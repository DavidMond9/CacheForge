#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(".."))

from dotenv import load_dotenv
import re
import time
import sqlite3
import subprocess
import random
from pathlib import Path
from typing import Optional, Tuple
import ollama
from RAG import ExperimentRAG
from PromptGenerator import PolicyPromptGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH = "DB/funsearch.db"
LIB_PATH = "ChampSim_CRC2/lib/config1.a"
INCLUDE_DIR = "ChampSim_CRC2/inc"
EXAMPLE_DIR = Path("ChampSim_CRC2/new_policies")

WARMUP_INST = "1000000"
SIM_INST = "10000000"
MODEL = "qwen2.5-coder:32b"
ITERATIONS = 1

EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

workloads = [
    {"name": "astar", "trace_path": "ChampSim_CRC2/traces/astar_313B.trace.gz"},
    {"name": "lbm", "trace_path": "ChampSim_CRC2/traces/lbm_564B.trace.gz"},
    {"name": "mcf", "trace_path": "ChampSim_CRC2/traces/mcf_250B.trace.gz"},
    {"name": "milc", "trace_path": "ChampSim_CRC2/traces/milc_409B.trace.gz"},
    {"name": "omnetpp", "trace_path": "ChampSim_CRC2/traces/omnetpp_17B.trace.gz"}
]
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

HOSTS = ["gpu36", "gpu16"]
PORTS = [11432, 11434, 11433]
active_server = None

def query_model(prompt: str, model: str, hosts=HOSTS, ports=PORTS) -> str:
    """Randomize order; try cached server first; return raw model text."""
    rng = random.Random(os.urandom(8))
    hosts = hosts[:]   # copy so we don’t mutate globals
    ports = ports[:]   # copy so we don’t mutate globals
    rng.shuffle(hosts)
    rng.shuffle(ports)

    def ask(h, p):
        client = ollama.Client(host=f"http://{h}:{p}")
        r = client.generate(model=model, prompt=prompt)
        return r["response"]             # ← return TEXT

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


def sanitize(name: str) -> str:
    print("     3. 🔧 [Sanitize] Cleaning policy name")

    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()

def parse_policy_content(text: str,) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def _extract(pattern: str):
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    name = _extract(r"##\s*Policy\s*Name\s*\n(.*?)\n")
    desc = _extract(r"##\s*Policy\s*Description\s*\n(.*?)\n")
    code = _extract(r"```cpp\s*(.*?)\s*```")

    # print(f"📦 [Parse] Extracted policy: {name}")
    return name, desc, code

def compile_policy(cc: Path) -> Path:
    print(f"     4. 🔨 [Compile] Compiling: {cc.name}\n")

    exe = cc.with_suffix(".out")
    subprocess.run(
        [
            "g++",
            "-Wall",
            "-std=c++17",
            f"-I{INCLUDE_DIR}",
            str(cc),
            LIB_PATH,
            "-o",
            str(exe),
        ],
        check=True,
    )
    return exe

def run_policy(exe: Path, trace_path: Path) -> str:

    print(f"     5. ⏳ [Simulation] Starting simulation for: {exe.name} and {str(trace_path)}")
    start_time = time.time()

    res = subprocess.run(
        [
            str(exe),
            "-warmup_instructions", WARMUP_INST,
            "-simulation_instructions", SIM_INST,
            "-traces", str(trace_path),
        ],
        check=True,
        capture_output=True,
        text=True,
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

def record(workload, name, desc, cc: Path, rate, workload_desc):
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
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main Feedback Loop with Reward/Penalty
# ──────────────────────────────────────────────────────────────────────────────
def main():
    
    WORKLOAD = "all"

    # 1) Setup RAG and PromptGenerator
    rag = ExperimentRAG(DB_PATH)
    prompt_gen = PolicyPromptGenerator(DB_PATH)
    load_dotenv(dotenv_path=Path(".env"), override=False)

    top_policies = rag.get_top_policies_by_score(WORKLOAD, top_n=5)
    workload_desc, traces = rag.get_all_workloads_with_description_and_traces()

    best_hit = top_policies[0]["score"]
    policy_summary = "\n".join(
            f"Policy: {p['policy']}\nHit Rate: {float(p['score']):.2%}\nDescription:\n{p['policy_description']}\n"
            for p in top_policies
        )

    print(f"     📈 [Init] Starting best cache hit rate: {best_hit:.2%}")


    prev_name = prev_desc = prev_code = None
    current_hit = best_hit
    i=0
    
    for i in range(ITERATIONS):
        if i == 0:
            prompt = (
                f"The following workloads are under consideration:\n"
                f"{workload_desc}\n\n"
                "The top-performing cache replacement policies from past experiments are:\n"
                f"{policy_summary}\n\n"
                "Your task: Propose a new cache replacement policy that aims to **outperform all of the above policies** "
                "across these workloads. Consider workload characteristics like branching, memory access patterns, spatial and temporal locality, and phase behavior.\n\n"
                "Suggested approach:\n"
                "1) Generate 3-4 distinct policy ideas (divergent thinking), briefly explain why each could help with different workloads.\n"
                "2) Choose the most promising policy and provide a complete C++ implementation.\n"
                "3) Include any tunable parameters or knobs, and note what telemetry/statistics should be tracked.\n\n"
                "Use the exact output format below:\n\n"
                "## Policy Name\n<name>\n\n"
                "## Policy Description\n<one paragraph describing the approach and why it helps>\n\n"
                "## C++ Implementation\n"
                f"{prompt_gen._get_code_template()}\n"
            )
            
        else:
            if current_hit > best_hit:
                feedback = (
                    f"Great! Policy improved from {best_hit:.2%} to "
                    f"{current_hit:.2%}. Please refine further."
                )
                best_hit = current_hit
            else:
                feedback = (
                    f"Policy hit rate was {current_hit:.2%}, not better than "
                    f"{best_hit:.2%}. Try a different approach."
                )

            prompt = (
                f"The following workloads are under consideration:\n"
                f"{workload_desc}\n\n"
                f"Your previous design was **{prev_name}**:\n\n"
                f"Description:\n{prev_desc}\n\n"
                f"Implementation:\n```cpp\n{prev_code}\n```\n\n"
                f"Feedback from the last run:\n{feedback}\n\n"
                "Task: Refine or redesign the policy to achieve better performance across all workloads. "
                "Consider workload characteristics such as branching behavior, memory access patterns, spatial and temporal locality, and phase changes. "
                "You may propose modifications, hybrid approaches, or completely new ideas if needed.\n\n"
                "Produce the output in the exact format below:\n\n"
                "## Policy Name\n<name>\n\n"
                "## Policy Description\n<one paragraph explaining the approach and why it improves performance>\n\n"
                "## C++ Implementation\n"
                f"{prompt_gen._get_code_template()}\n"
            )

        
        # 5) Call model
        print(f"     1. 📤 [LLM] Iteration {i}: sending prompt to Ollama backends")

        text = query_model(prompt, MODEL)
        print("     2. 📥 [LLM] Response received from Ollama")


        # 6) Parse LLM output
        name, desc, code = parse_policy_content(text)
        if not (name and desc and code):
            raise RuntimeError(f"❌ Parse failed")

        # 7) Write, compile, run
        base = sanitize(name)
        cc = EXAMPLE_DIR / f"{i:03}_{base}.cc"
        cc.write_text(code, encoding="utf-8")
       

        try:
            exe = compile_policy(cc)
        except subprocess.CalledProcessError as e:
            print(f"❌ [Compile Error]:\n{e}")
            #compile_error = True
            continue  # ← this restarts the loop at the top
        #compile_error=False

        current_hit_tmp=0
        
        for trace_info in workloads:
            WORKLOAD = trace_info["name"]
            trace_path = trace_info["trace_path"]

            out = run_policy(exe, trace_path)
            tmp = parse_hit_rate(out)
            current_hit_tmp += tmp
            record(WORKLOAD, name, desc, cc, tmp, "")
            print(f"      [+] {name} → workload: {WORKLOAD} → hit rate: {tmp}\n")

        current_hit = current_hit_tmp / len(workloads)
        print(f"✅ [Result] Iteration {i}: {name}  → average hit rate {current_hit:.2%}\n")

        # 8) Record experiment
        record("all",name, desc, cc, current_hit, "")

        i+=1

        # 9) Prepare for next iteration
        prev_name, prev_desc, prev_code = name, desc, code




    prompt_gen.close()
    rag.close()


if __name__ == "__main__":
    main()
