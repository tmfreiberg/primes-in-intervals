"""Write README.md for the results directory from the files the run produced."""

import platform
import subprocess
import sys
from pathlib import Path

import primes_in_intervals as pii

OUT = Path(sys.argv[1])
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
).stdout.strip()
log = (OUT / "run.log").read_text() if (OUT / "run.log").exists() else ""
timings = [line for line in log.splitlines() if line.startswith("H = ") or line.startswith("real")]
checks = (OUT / "checks.txt").read_text()
cases = (OUT / "cases.txt").read_text()
s = pii.QuadratureSettings()
text = f"""# Primes in intervals: cumulative experiment, M = 0

Code: primes-in-intervals {pii.__version__}, commit {commit}{" (working tree had uncommitted changes)" if dirty else ""}
Python {platform.python_version()} on {platform.platform()}
Counter: {pii.COUNTER_VERSION}
Formulas: {pii.FORMULA_VERSION}

## Cases (N, H, target kappa)

{cases}
H = round(kappa (log N - 1)); every prediction uses the actual N and H.

## Data

X(n) = pi(n + H) - pi(n) for all integers 1 <= n <= N (M = 0), counted exhaustively
by overlap_cp in one sweep per case. Exact counts: raw/N<N>_H<H>.json (dataset JSON,
checkpoints 0 and N) and counts.db (SQLite, table overlap_raw, row (0, N, H), with a
provenance table recording counter version, package version, time and note).

## Predictions (at integer m)

p_m(u) = e^-u u^m / m!;  eta(H) = (log H + log 2 pi + gamma - 1) / H
Q(m; u, H) = p_m(u) [1 - eta(H)/2 ((m - u)^2 - m)]
F   = (1/N) int_2^N Q(m; H/log t, H) dt       F0 = (1/N) int_2^N p_m(H/log t) dt
mu  = (H/N) int_2^N dt/log t  (closed form via the exponential integral)
lambda = H / (log N - 1)
B_const = Binom(m; H, mu/H)
B_avg   = (1/N) int_e^N Binom(m; H, 1/log t) dt  ([2, e) omitted, where 1/log t > 1;
          its contribution lies in [0, (e - 2)/N] for every m: column B_avg_omitted_bound)
Q_mu = Q(m; mu, H);  Q_lambda = Q(m; lambda, H)
Nothing is clipped or renormalized. F and F0 sum to 1 - 2/N over all m.

## Numerical settings

Quadrature: scipy quad_vec (Gauss-Kronrod 21) over sigma = log(t/2), epsabs {s.epsabs} (on the
averaged value), epsrel {s.epsrel}, limit {s.limit}; error estimates in quad_error (heuristic,
not a bound). Cross-check: `pii quadrature-check H 0 N` compares with a 30-digit mpmath value.

Discrepancies (global): L1 = global_E1_*, L2 = global_E2_*, L-infinity = global_Einf_*,
summed over m = 0..m_T, all integers included (zero-frequency bins too). m_T is the least
value >= max(largest observed m, H) for which the absolute tail of F beyond m_T is provably
<= 1e-12 (global_tail_bound_F, rigorous). For F0 the same argument gives global_tail_bound_F0.
The binomials vanish beyond H <= m_T (tail 0). For Q_mu and Q_lambda the tail column is the
computed sum of |Q| beyond m_T. ratio_global_E1 = L1(F)/L1(F0), ratio_global_E2 = L2(F)/L2(F0).
Mass: deficit_* = 1 - sum over 0..m_T, negative_mass_*, min_value_*.
Central columns (central_*) use m = 0..H.

## Files

per_bin_N_H_m.csv: one row per (N, H, m), m = 0..H: count, P, F, F0, F - F0 (correction),
B_const, Q_mu, Q_lambda, B_avg.
summary_by_N_H.csv: one row per (N, H).
figures/: representative case and the ratio plot (PDF and PNG).
per_case/, cache/: intermediate tables and the prediction cache (regenerable).

## Timings (from run.log; "real" is wall time including uv start-up)

{chr(10).join(timings)}

## Checks

{checks}
All cases are kept. The data are exhaustive counts of overlapping intervals; no
sampling error bars or significance tests apply.
"""
(OUT / "README.md").write_text(text)
print("wrote", OUT / "README.md")