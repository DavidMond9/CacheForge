#!/usr/bin/env python3
"""
Collect ChampSim LLC statistics for all policies recorded in funsearch.db.

Output: policy_stats.csv  (columns: Policy, Accesses, Hits, Misses, HitRate)
"""

import csv
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Tuple

# ────────────────────────────────────────────────────────────
#  Project-specific paths — adjust if your repo layout differs
# ────────────────────────────────────────────────────────────
DB_PATH       = "funsearch.db"
TRACE_PATH    = Path("ChampSim_CRC2/trace/astar_313B.trace.gz")
LIB_PATH      = "ChampSim_CRC2/lib/config1.a"
INCLUDE_DIR   = "ChampSim_CRC2/inc"
WARMUP_INST   = "100000000"
SIM_INST      = "500000000"

#  Where *.cc files live (as recorded in the DB) and where we
#  put the compiled executables
OUT_DIR = Path("ChampSim_CRC2/example")
OUT_DIR.mkdir(exist_ok=True, parents=True)

CSV_PATH = "policy_stats.csv"

# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────
def compile_policy(src: Path) -> Path:
    """g++ <src> -> <src>.out  (always recompiles)"""
    exe = OUT_DIR / (src.stem + ".out")
    subprocess.run(
        [
            "g++",
            "-Wall",
            "-std=c++17",
            f"-I{INCLUDE_DIR}",
            str(src),
            LIB_PATH,
            "-o",
            str(exe),
        ],
        check=True,
    )
    return exe


def run_policy(exe: Path) -> str:
    """Execute ChampSim binary and capture stdout."""
    res = subprocess.run(
        [
            str(exe),
            "-warmup_instructions", WARMUP_INST,
            "-simulation_instructions", SIM_INST,
            "-traces", str(TRACE_PATH),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return res.stdout


def parse_llc_stats(text: str) -> Tuple[int, int]:
    """
    Extract 'LLC TOTAL ACCESS: <A> HIT: <H>' and return (A, H).
    Raises if pattern not found.
    """
    m = re.search(r"LLC TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)", text)
    if not m:
        raise RuntimeError("LLC TOTAL line not found in ChampSim output")
    access, hits = map(int, m.groups())
    return access, hits


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    # 1) Fetch policies and their C++ paths.
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT policy, cpp_file_path FROM experiments"
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No policies recorded in funsearch.db")

    # 2) Evaluate each policy.
    results = []
    for policy, cpp_path in rows:
        cpp_file = Path(cpp_path)
        if not cpp_file.is_file():
            print(f"[WARN] Missing source for '{policy}' → skipped")
            continue

        print(f"[INFO] Compiling {policy} …")
        exe = compile_policy(cpp_file)

        print(f"[INFO] Running   {policy} …")
        out = run_policy(exe)

        accesses, hits = parse_llc_stats(out)
        misses   = accesses - hits
        hit_rate = hits / accesses if accesses else 0.0

        results.append(
            (policy, accesses, hits, misses, f"{hit_rate:.4f}")
        )

    # 3) Write CSV.
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Policy Name",
                         "LLC Total Accesses",
                         "LLC Hits",
                         "LLC Misses",
                         "LLC Hit Rate"])
        writer.writerows(results)

    print(f"\n[✓] Wrote {len(results)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
