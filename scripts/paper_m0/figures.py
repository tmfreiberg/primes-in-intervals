"""Representative-case figure and the optional ratio plot."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import primes_in_intervals as pii

OUT = Path(sys.argv[1])
N, H = int(sys.argv[2]), int(sys.argv[3])
plt.rcParams.update({"font.size": 11})

# 1. Representative case: exact counts from the raw JSON, predictions recomputed (cached).
ds = pii.read_dataset_json(OUT / "raw" / f"N{N}_H{H}.json")
cache = pii.PredictionCache(OUT / "cache" / f"pred_H{H}.json")
frame = [f for f in pii.build_frames(ds, models=("F0", "B_avg", "F"), cache=cache) if f.N == N][0]
cache.save()
m = np.arange(frame.m_axis + 1)
fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
)
pii.plot_cumulative_frame(top, frame, H, models=("F0", "B_avg", "F"), guides=False, overlay=False)
top.set_xlabel("")
top.legend(loc="upper right", framealpha=0.8)
top.set_title(rf"$N = {N:,}$, $H = {H}$, $\mu = {frame.predictions['mu']:.4f}$")
F0, F = frame.prediction("F0"), frame.prediction("F")
bottom.axhline(0, color="black", linewidth=0.8)
bottom.plot(m, frame.P - F0, "o", color="red", label=r"$P_m - F_0(m)$")
bottom.plot(
    m,
    F - F0,
    "s",
    markerfacecolor="none",
    markeredgewidth=1.5,
    color="green",
    label=r"$F(m) - F_0(m)$",
)
bottom.set_xlabel(r"$m$ (number of primes in $(n, n + H]$)")
bottom.legend(framealpha=0.6)
bottom.grid(True, alpha=0.5)
bottom.set_xticks(m)
fig.tight_layout()
pii.save_figure(fig, OUT / "figures" / f"representative_N{N}_H{H}", formats=("pdf", "png"), dpi=200)
plt.close(fig)

# 2. Optional: L1(F)/L1(F0) against N, one series per target kappa.
s = pd.read_csv(OUT / "summary_by_N_H.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for kappa, g in s.sort_values("N").groupby("kappa"):
    ax.plot(g["N"], g["ratio_global_E1"], "o-", label=rf"$\kappa = {kappa}$")
ax.set_xscale("log")
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$L_1(F)\,/\,L_1(F_0)$")
ax.grid(True, alpha=0.5)
ax.legend()
fig.tight_layout()
pii.save_figure(fig, OUT / "figures" / "ratio_L1_F_over_F0", formats=("pdf", "png"), dpi=200)
plt.close(fig)
print("figures written to", OUT / "figures")
