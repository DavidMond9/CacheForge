#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(".."))

import re
import time
import csv
import json
import argparse
import sqlite3
import subprocess
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import ollama

from dotenv import load_dotenv
from RAG import ExperimentRAG
from PromptGenerator import PolicyPromptGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (defaults; can be overridden by CLI)
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH = "DB/funsearch.db"
LIB_PATH = "ChampSim_CRC2/lib/config1.a"
INCLUDE_DIR = "ChampSim_CRC2/inc"
EXAMPLE_DIR = Path("ChampSim_CRC2/new_policies")
RESULTS_DIR = Path("results")                 # logs for plots/ablation
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_INST = "1000000"
SIM_INST    = "10000000"
MODEL       = "qwen2.5-coder:32b"

# Seed hosts/ports (trimmed to the ones you verified)
HOSTS = ["gpu36", "gpu16"]
PORTS = [11432, 11434, 11433]
active_server = None

workloads = [
    {"name": "astar",   "trace_path": "ChampSim_CRC2/traces/astar_313B.trace.gz"},
    {"name": "lbm",     "trace_path": "ChampSim_CRC2/traces/lbm_564B.trace.gz"},
    {"name": "mcf",     "trace_path": "ChampSim_CRC2/traces/mcf_250B.trace.gz"},
    {"name": "milc",    "trace_path": "ChampSim_CRC2/traces/milc_409B.trace.gz"},
    {"name": "omnetpp", "trace_path": "ChampSim_CRC2/traces/omnetpp_17B.trace.gz"}
]

# Some patterns we reject up-front (compile-hallucination guard)
BANNED_SNIPPETS = [
    "access_count", "std::memmove", "memmove(", "LOG2_BLOCK_SIZE",
    '#include "../inc/', '#include "inc/', "std::list", "std::map<"
]

# ──────────────────────────────────────────────────────────────────────────────
# Ollama client with randomized + cached server
# ──────────────────────────────────────────────────────────────────────────────
def query_model(prompt: str, model: str, *, options: Optional[Dict[str, Any]]=None,
                hosts=HOSTS, ports=PORTS, rng: Optional[random.Random]=None) -> str:
    """Randomize order; try cached server first; return raw model text."""
    if rng is None:
        rng = random.Random(os.urandom(8))
    hosts = hosts[:] ; ports = ports[:]
    rng.shuffle(hosts); rng.shuffle(ports)

    def ask(h: str, p: int) -> str:
        client = ollama.Client(host=f"http://{h}:{p}")
        r = client.generate(model=model, prompt=prompt, options=options or {}, stream=False)
        return r["response"]

    global active_server
    if active_server:
        h, p = active_server
        try:
            print(f"✅ Using cached server {h}:{p}")
            return ask(h, p)
        except Exception as e:
            print(f"⚠️ Cached server {h}:{p} failed: {e}")
            active_server = None

    for h in hosts:
        for p in ports:
            try:
                resp_text = ask(h, p)
                print(f"✅ Found new active server {h}:{p}")
                active_server = (h, p)
                return resp_text
            except Exception as e:
                print(f"⚠️ Server {h}:{p} not available: {e}")

    print("⏳ No servers available.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def sanitize(name: str) -> str:
    print("     3. 🔧 [Sanitize] Cleaning policy name")
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()

def parse_policy_content(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def _extract(pattern: str):
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None
    name = _extract(r"##\s*Policy\s*Name\s*\n(.*?)\n")
    desc = _extract(r"##\s*Policy\s*Description\s*\n(.*?)\n")
    code = _extract(r"```cpp\s*(.*?)\s*```")
    return name, desc, code

def harden_code(code: str) -> str:
    """Patch common include issues; reject obviously bad patterns early."""
    # Normalize bogus include paths
    code = code.replace('#include "../inc/', '#include "')
    code = code.replace('#include "inc/',   '#include "')
    # Add common headers if functions/macros are used
    if "INT_MAX" in code and "<climits>" not in code:
        code = '#include <climits>\n' + code
    if ("memset(" in code or "memcpy(" in code) and "<cstring>" not in code:
        code = '#include <cstring>\n' + code
    return code

def contains_banned(code: str) -> Optional[str]:
    for bad in BANNED_SNIPPETS:
        if bad in code:
            return bad
    return None

def compile_policy(cc: Path) -> Path:
    print(f"     4. 🔨 [Compile] Compiling: {cc.name}\n")
    exe = cc.with_suffix(".out")
    subprocess.run(
        ["g++", "-Wall", "-std=c++17", f"-I{INCLUDE_DIR}", str(cc), LIB_PATH, "-o", str(exe)],
        check=True,
    )
    return exe

def run_policy(exe: Path, trace_path: Path) -> str:
    print(f"     5. ⏳ [Simulation] Starting simulation for: {exe.name} and {str(trace_path)}")
    start_time = time.time()
    res = subprocess.run(
        [str(exe), "-warmup_instructions", WARMUP_INST, "-simulation_instructions", SIM_INST,
         "-traces", str(trace_path)],
        check=True, capture_output=True, text=True,
    )
    duration = time.time() - start_time
    print(f"     6. 🏁 [Simulation] Finished in {duration:.2f} seconds for: {exe.name} and {trace_path}")
    return res.stdout

def parse_hit_rate(output: str) -> float:
    print("     7. 📊 [Metric] Parsing cache hit rate from output")
    m = re.search(r"LLC TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)", output)
    if not m:
        raise RuntimeError("LLC TOTAL not found")
    return int(m.group(2)) / int(m.group(1))

def record_to_db(workload, name, desc, cc: Path, rate, workload_desc):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
      INSERT INTO experiments
        (workload, policy, policy_description, workload_description,
         cpp_file_path, cache_hit_rate, score)
      VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (workload, name, desc, workload_desc, str(cc), rate, rate),
    )
    conn.commit(); conn.close()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="CacheForge Run Loop (Ollama backends)")
    ap.add_argument("--iterations", type=int, default=3, help="Outer iterations")
    ap.add_argument("--candidates", type=int, default=2, help="Candidates per iteration")
    ap.add_argument("--temp", type=float, default=0.8, help="Generation temperature")
    ap.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")
    ap.add_argument("--model", type=str, default=MODEL, help="Ollama model tag")
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else os.urandom(8))
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Setup RAG + prompt generator
    load_dotenv(dotenv_path=Path(".env"), override=False)
    rag = ExperimentRAG(DB_PATH)
    prompt_gen = PolicyPromptGenerator(DB_PATH)

    WORKLOAD = "all"
    top_policies = rag.get_top_policies_by_score(WORKLOAD, top_n=5)
    workload_desc, traces = rag.get_all_workloads_with_description_and_traces()

    best_hit = float(top_policies[0]["score"])
    policy_summary = "\n".join(
        f"Policy: {p['policy']}\nHit Rate: {float(p['score']):.2%}\nDescription:\n{p['policy_description']}\n"
        for p in top_policies
    )
    print(f"     📈 [Init] Starting best cache hit rate: {best_hit:.2%}")

    # CSV logger
    csv_path = RESULTS_DIR / "iteration_history.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "iter", "candidate_idx", "name",
                        "avg_hit_rate"] + [f"hit_{w['name']}" for w in workloads])

    prev_name = prev_desc = prev_code = None
    current_hit = best_hit

    for i in range(args.iterations):
        # Build the prompt
        if i == 0 or prev_code is None:
            prompt = (
                f"The following workloads are under consideration:\n"
                f"{workload_desc}\n\n"
                "The top-performing cache replacement policies from past experiments are:\n"
                f"{policy_summary}\n\n"
                "Your task: Propose a **single** cache replacement policy that **compiles** with ChampSim CRC2 "
                "and aims to outperform the above on average.\n\n"
                "Constraints:\n"
                "- Use ONLY the official ChampSim CRC2 headers and interfaces.\n"
                "- Do NOT add fields to BLOCK or use non-existent fields (e.g., access_count).\n"
                "- Do NOT reorder cache lines with std::memmove; only update replacement metadata.\n"
                "- Keep includes like: #include \"champsim_crc2.h\"\n"
                "- Provide a complete C++ file implementing the required hooks only.\n\n"
                "Use the exact output format:\n\n"
                "## Policy Name\n<name>\n\n"
                "## Policy Description\n<one paragraph>\n\n"
                "## C++ Implementation\n"
                f"{prompt_gen._get_code_template()}\n"
            )
        else:
            feedback = (
                f"Previous avg hit: {current_hit:.2%} vs best-so-far {best_hit:.2%}. "
                "Try a materially different angle if not beating best."
            )
            prompt = (
                f"The following workloads are under consideration:\n{workload_desc}\n\n"
                f"Your previous design was **{prev_name}**:\n\n"
                f"Description:\n{prev_desc}\n\n"
                f"Implementation:\n```cpp\n{prev_code}\n```\n\n"
                f"Feedback:\n{feedback}\n\n"
                "Constraints (must compile on ChampSim CRC2):\n"
                "- Use ONLY official interfaces; do NOT access non-existent fields.\n"
                "- No memmove/reordering of physical lines; only replacement metadata.\n"
                "- Keep includes simple: #include \"champsim_crc2.h\"\n\n"
                "Produce exactly:\n"
                "## Policy Name\n...\n\n"
                "## Policy Description\n...\n\n"
                "## C++ Implementation\n"
                f"{prompt_gen._get_code_template()}\n"
            )

        print(f"     1. 📤 [LLM] Iteration {i}: asking {args.candidates} candidates @ temp={args.temp}")
        candidates: List[Dict[str, Any]] = []

        for cidx in range(args.candidates):
            seed = rng.getrandbits(32)
            text = query_model(prompt, args.model,
                               options={"temperature": float(args.temp), "seed": int(seed)},
                               rng=rng)
            name, desc, code = parse_policy_content(text)
            if not (name and desc and code):
                print("     ↪️  Candidate parse failed; skipping.")
                continue

            code = harden_code(code)
            bad = contains_banned(code)
            if bad:
                print(f"     ↪️  Candidate rejected (banned token: {bad}).")
                continue

            base = sanitize(name)
            cc = EXAMPLE_DIR / f"{i:03}_{cidx:02}_{base}.cc"
            cc.write_text(code, encoding="utf-8")

            try:
                exe = compile_policy(cc)
            except subprocess.CalledProcessError as e:
                print(f"❌ [Compile Error candidate {cidx}]:\n{e}")
                continue

            # Evaluate across all workloads
            per_work_hits = []
            for w in workloads:
                out = run_policy(exe, Path(w["trace_path"]))
                hit = parse_hit_rate(out)
                per_work_hits.append(hit)
                record_to_db(w["name"], name, desc, cc, hit, "")

            avg_hit = sum(per_work_hits) / len(per_work_hits)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(time.time()), i, cidx, name, avg_hit, *per_work_hits
                ])
            print(f"✅ [Result] Iter {i} cand {cidx}: {name} → avg {avg_hit:.2%}")
            candidates.append({"avg": avg_hit, "name": name, "desc": desc, "cc": cc, "code": code})

        if not candidates:
            print("⚠️ No valid candidates this iteration (parse/compile guards). Continuing.")
            continue

        # Pick best of pool
        best = max(candidates, key=lambda x: x["avg"])
        current_hit = best["avg"]
        record_to_db("all", best["name"], best["desc"], best["cc"], current_hit, "")
        print(f"🌟 [Best of iter {i}] {best['name']} → avg {current_hit:.2%}\n")

        # Update rolling best and feedback memory
        if current_hit > best_hit:
            best_hit = current_hit
        prev_name, prev_desc, prev_code = best["name"], best["desc"], best["code"]

    prompt_gen.close()
    rag.close()
    print(f"🏁 Done. CSV log: {csv_path}")
    print("Tip: plot iteration_history.csv (iter vs avg_hit_rate) for the report.")

if __name__ == "__main__":
    main()
