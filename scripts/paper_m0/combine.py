"""Combine the nine per-case tables, check them against the raw counts, and add kappa."""

import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

import primes_in_intervals as pii

OUT = Path(sys.argv[1])
cases = [
    tuple(int(x) for x in line.split())
    for line in (OUT / "cases.txt").read_text().split("\n")
    if line.strip()
]
long_parts, summary_parts, checks = [], [], []
for N, H, kappa in cases:
    stem = OUT / "per_case" / f"N{N}_H{H}"
    long = pd.read_csv(f"{stem}_by_N_m.csv")
    summ = pd.read_csv(f"{stem}_by_N.csv")
    # keep exactly the requested (M=0, N, H) case, never a neighbouring checkpoint
    long = long[(long["N"] == N) & (long["H"] == H)].copy()
    summ = summ[(summ["N"] == N) & (summ["H"] == H)].copy()
    assert len(summ) == 1, f"expected one summary row for N={N}, H={H}, found {len(summ)}"
    # the JSON export and the database hold identical counts, summing to N
    raw = pii.read_dataset_json(OUT / "raw" / f"N{N}_H{H}.json")
    counts = {m: v for m, v in raw["data"][N].items() if v}
    conn = sqlite3.connect(OUT / "counts.db")
    row = conn.execute(
        "SELECT * FROM overlap_raw WHERE lower_bound=0 AND upper_bound=? AND interval_length=?",
        (N, H),
    ).fetchone()
    conn.close()
    stored = {m - 3: v for m, v in enumerate(row) if m >= 3 and v}
    table = {int(m): int(c) for m, c in zip(long["m"], long["count"]) if c}
    ok = counts == stored == table and sum(counts.values()) == N
    status = "OK" if ok else "FAILED"
    checks.append(f"N={N} H={H}: JSON = database = CSV, total {sum(counts.values())} = N: {status}")
    long.insert(2, "kappa", kappa)
    summ.insert(2, "kappa", kappa)
    # [2, e) is omitted from B_avg: the omitted part lies in [0, (e - 2)/N] for every m
    summ["B_avg_omitted_bound"] = (math.e - 2) / N
    long_parts.append(long)
    summary_parts.append(summ)
pd.concat(long_parts).to_csv(OUT / "per_bin_N_H_m.csv", index=False)
summary = pd.concat(summary_parts)
first = ["N", "H", "kappa", "lambda", "mu", "mean", "m_T", "max_m_observed",
         "ratio_global_E1", "ratio_global_E2", "quad_error", "quad_neval"]
summary = summary[first + [c for c in summary.columns if c not in first]]
summary.to_csv(OUT / "summary_by_N_H.csv", index=False)
(OUT / "checks.txt").write_text("\n".join(checks) + "\n")
print("\n".join(checks))
