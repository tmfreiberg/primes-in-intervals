r"""The fixed-``H``, ``M = 0`` experiment: cumulative counts over ``1 <= n <= N``.

For a fixed interval length ``H`` the empirical distribution

    P(m; H, 0, N) = #{1 <= n <= N : pi(n + H) - pi(n) = m} / N

is recorded at a schedule of checkpoints ``N`` (dense at first, then
geometrically spaced), starting at ``N = 1``, in one sweep of the primes
(:func:`~primes_in_intervals.intervals.overlap_cp` with the internal
checkpoint ``0`` in front, which holds no intervals and is never displayed).
For every checkpoint from a configurable ``overlay_from`` onwards the
manuscript's predictions are recomputed for the actual range ``(0, N]``,
through :func:`~primes_in_intervals.predictions.predict_all`, and scored with
the discrepancy measures of :mod:`primes_in_intervals.discrepancies`.

The result is a list of :class:`Frame` objects, one per checkpoint, which the
plotting layer turns into static figures and animations, and which
:func:`frames_to_tables` flattens into two tables (one row per ``(N, m)``,
one row per ``N``) for CSV export and for the paper.

Predictions are cached separately from the raw counts
(:class:`PredictionCache`, a JSON file keyed by ``H``, ``M``, ``N``, the
largest ``m``, the formula version and the quadrature settings), so changing
a plot never recomputes an integral, and changing a formula never touches a
count.

**On the mean.**  For fixed ``H`` the averaged parameter ``mu`` decreases as
``N`` grows (the primes thin out), so this experiment does not show
convergence to a Poisson law with a fixed mean.  It shows the evolution of
the cumulative frequencies together with predictions that move with them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from primes_in_intervals.dataio import retrieve, save
from primes_in_intervals.discrepancies import (
    discrepancy,
    global_range,
    mass_report,
    tail_bound,
    tail_bound_local,
)
from primes_in_intervals.intervals import Dataset, overlap_cp
from primes_in_intervals.predictions import DEFAULT_MODELS, FORMULA_VERSION, predict_all
from primes_in_intervals.quadrature import QuadratureSettings
from primes_in_intervals.statistics import dictionary_statistics

__all__ = [
    "DEFAULT_OVERLAY_FROM",
    "Frame",
    "PredictionCache",
    "build_frames",
    "checkpoint_schedule",
    "frames_to_tables",
    "load",
    "run",
    "write_tables",
]

#: Default first checkpoint at which theoretical overlays are drawn and
#: scored.  This is a presentation choice, not a mathematical threshold: the
#: integral formulas are computed for every ``N >= 3`` (they are zero at
#: ``N = 2``), but at very small ``N`` they are not useful asymptotic
#: approximations and the shifted parameter ``H/(log N - 1)`` is unusable for
#: ``N <= e``.  Each model's parameters are checked separately at every
#: frame regardless of this setting.
DEFAULT_OVERLAY_FROM = 100


def checkpoint_schedule(N_max: int, dense_until: int = 100, ratio: float = 1.05) -> list[int]:
    """Return checkpoints ``1, 2, ..., dense_until`` then geometrically spaced up to ``N_max``.

    After the dense initial stretch each checkpoint is the previous one
    multiplied by ``ratio`` and rounded up (always at least one more than the
    previous), and ``N_max`` itself is always the last checkpoint.

    Parameters
    ----------
    N_max : int
        Largest checkpoint (the final ``N``), at least 1.
    dense_until : int, optional
        Every integer up to this value is a checkpoint (default 100).
    ratio : float, optional
        Growth factor of the later checkpoints (default 1.05), greater than 1.

    Returns
    -------
    list of int
        Strictly increasing, starting at 1 and ending at ``N_max``.
    """
    if N_max < 1:
        raise ValueError("N_max must be at least 1")
    if ratio <= 1:
        raise ValueError("ratio must exceed 1")
    out = list(range(1, min(dense_until, N_max) + 1))
    n = out[-1]
    while n < N_max:
        n = max(n + 1, int(math.ceil(n * ratio)))
        out.append(min(n, N_max))
    return out


def run(
    H: int,
    N_max: int | None = None,
    checkpoints: list[int] | None = None,
    dense_until: int = 100,
    ratio: float = 1.05,
    db_path: str | Path | None = None,
    save_to_db: bool = False,
    on_conflict: str = "error",
    note: str = "",
) -> Dataset:
    """Count primes in ``(n, n + H]`` for ``1 <= n <= N`` at every checkpoint, in one sweep.

    Parameters
    ----------
    H : int
        Interval length (fixed for the whole run).
    N_max : int, optional
        Final checkpoint; used with :func:`checkpoint_schedule` when
        ``checkpoints`` is not given.
    checkpoints : list of int, optional
        Explicit checkpoints (positive integers; ``0`` is added internally).
    dense_until, ratio : int and float, optional
        Passed to :func:`checkpoint_schedule`.
    db_path : str, Path, or None, optional
        Database for ``save_to_db``.
    save_to_db : bool, optional
        Store the rows ``(0, N, H)`` in ``overlap_raw`` (default False).
    on_conflict, note : str, optional
        Passed to :func:`~primes_in_intervals.dataio.save`.

    Returns
    -------
    dict
        The dataset from :func:`~primes_in_intervals.intervals.overlap_cp`
        with lower bound ``0``.
    """
    if checkpoints is None:
        if N_max is None:
            raise ValueError("give N_max or an explicit checkpoint list")
        checkpoints = checkpoint_schedule(N_max, dense_until, ratio)
    C = sorted({0, *(int(c) for c in checkpoints)})
    if any(c < 0 for c in C):
        raise ValueError("checkpoints must be non-negative")
    dataset = overlap_cp(C, H)
    if save_to_db:
        save(dataset, db_path=db_path, on_conflict=on_conflict, note=note)
    return dataset


def load(H: int, db_path: str | Path | None = None) -> Dataset:
    """Return the stored overlapping dataset with interval length ``H`` and lower bound ``0``.

    Parameters
    ----------
    H : int
        Interval length.
    db_path : str, Path, or None, optional
        Database file.

    Returns
    -------
    dict

    Raises
    ------
    LookupError
        If no such dataset is stored.
    """
    found = retrieve(H, "overlap", db_path=db_path)
    if found is None:
        raise LookupError(f"no overlapping data for H = {H}")
    if isinstance(found, dict):
        found = [found]
    for ds in found:
        if ds["header"]["lower_bound"] == 0:
            return ds
    raise LookupError(f"no overlapping dataset with lower bound 0 for H = {H}")


@dataclass
class Frame:
    """One checkpoint of the cumulative experiment.

    Attributes
    ----------
    N : int
        The checkpoint: starting points ``1 <= n <= N``.
    counts : dict
        ``{m: #{n <= N : X(n; H) = m}}``, nonzero entries only.
    P : numpy.ndarray
        Relative frequencies at ``m = 0, ..., m_axis``.
    m_axis : int
        Largest ``m`` of the shared horizontal axis.
    mean, var, median : float
        Empirical statistics of the counts.
    mode : list of int
        Empirical mode(s).
    predictions : dict or None
        Output of :func:`~primes_in_intervals.predictions.predict_all` at
        ``(H, 0, N)`` for ``m = 0, ..., m_pred``, or ``None`` before the
        overlay start.
    scores : dict or None
        Discrepancy scores per model (``'central'``, ``'global'``,
        ``'mass'``, ``'ratio'``, ``'m_T'``), or ``None``.
    """

    N: int
    counts: dict[int, int]
    P: np.ndarray
    m_axis: int
    mean: float
    var: float
    median: float
    mode: list[int]
    predictions: dict[str, Any] | None = None
    scores: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def prediction(self, model: str, m_max: int | None = None) -> np.ndarray | None:
        """Return the model's values at ``m = 0, ..., m_max`` (default ``m_axis``), or ``None``."""
        if self.predictions is None or model not in self.predictions:
            return None
        values = np.asarray(self.predictions[model], dtype=float)
        top = self.m_axis if m_max is None else m_max
        return values[: top + 1]


class PredictionCache:
    """A JSON file of prediction vectors, keyed by everything that determines them.

    Parameters
    ----------
    path : str or Path
        The JSON file (created on first save).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, Any] = {}
        if self.path.exists():
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(H: int, M: float, N: float, m_max: int, settings: dict[str, Any]) -> str:
        """Return the cache key of ``(H, M, N, m_max)`` under the formula version and tolerances.

        The key does not name the models: an entry holds whichever models
        have been computed for it so far, and a request for more merges the
        new ones in.
        """
        return json.dumps(
            {
                "H": H,
                "M": M,
                "N": N,
                "m_max": m_max,
                "formula_version": FORMULA_VERSION,
                "settings": settings,
            },
            sort_keys=True,
        )

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached entry (arrays restored), or ``None``."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in entry.items()}

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store an entry (NumPy arrays are written as lists)."""
        self._entries[key] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in value.items()
        }

    def save(self) -> None:
        """Write the cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries), encoding="utf-8")

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._entries)


def _predict_cached(
    cache: PredictionCache | None,
    H: int,
    M: float,
    N: float,
    m_max: int,
    models: tuple[str, ...],
    settings: QuadratureSettings,
) -> dict[str, Any]:
    key = PredictionCache.key(H, M, N, m_max, settings.as_dict())
    hit = cache.get(key) if cache is not None else None
    missing = tuple(name for name in models if hit is None or name not in hit)
    if hit is not None and not missing:
        return hit
    pred = predict_all(H, M, N, m_max, models=missing, settings=settings)
    if hit is not None:
        merged = dict(hit)
        merged.update({k: v for k, v in pred.items() if k not in merged or k in missing})
        pred = merged
    if cache is not None:
        cache.put(key, pred)
    return pred


def build_frames(
    dataset: Dataset,
    overlay_from: int = DEFAULT_OVERLAY_FROM,
    models: tuple[str, ...] = DEFAULT_MODELS,
    m_axis: int | None = None,
    central: tuple[int, int] | None = None,
    tail_tolerance: float = 1e-12,
    settings: QuadratureSettings | None = None,
    cache: PredictionCache | None = None,
) -> list[Frame]:
    """Turn a lower-bound-zero dataset into scored frames, one per checkpoint ``N >= 1``.

    Parameters
    ----------
    dataset : dict
        A dataset with ``'header'['lower_bound'] == 0``.
    overlay_from : int, optional
        First ``N`` at which predictions are computed and scored (default
        :data:`DEFAULT_OVERLAY_FROM`); earlier frames are empirical only.
    models : sequence of str, optional
        The models to evaluate (see
        :data:`~primes_in_intervals.predictions.MODELS`).
    m_axis : int, optional
        Largest ``m`` of the shared horizontal axis (default: the largest
        count observed at any checkpoint).
    central : tuple of int, optional
        The fixed central scoring range ``(m_lo, m_hi)`` (default
        ``(0, m_axis)``).
    tail_tolerance : float, optional
        Target for the omitted predicted mass of the global range.
    settings : QuadratureSettings, optional
        Quadrature tolerances.
    cache : PredictionCache, optional
        Cache to read and fill (call its ``save`` afterwards).

    Returns
    -------
    list of Frame
    """
    if dataset["header"]["lower_bound"] != 0:
        raise ValueError("build_frames needs a dataset with lower bound 0 (starting points 1..N)")
    H = dataset["header"]["interval_length"]
    settings = settings or QuadratureSettings()
    checkpoints = sorted(N for N in dataset["data"].keys() if N > 0)
    observed_max = max(
        (m for N in checkpoints for m, v in dataset["data"][N].items() if v), default=0
    )
    if m_axis is None:
        m_axis = observed_max
    if central is None:
        central = (0, m_axis)
    m_lo, m_hi = central
    frames: list[Frame] = []
    for N in checkpoints:
        counts = {m: v for m, v in dataset["data"][N].items() if v}
        total = sum(counts.values())
        if total != N:
            raise ValueError(f"checkpoint {N}: counts sum to {total}, not {N}")
        stats = dictionary_statistics(counts)
        P = np.zeros(m_axis + 1)
        for m, v in counts.items():
            if m <= m_axis:
                P[m] = v / N
        frame = Frame(
            N=N,
            counts=counts,
            P=P,
            m_axis=m_axis,
            mean=float(stats["mean"]),
            var=float(stats["var"]),
            median=float(stats["med"]),
            mode=[int(x) for x in stats["mode"]],
        )
        if N >= overlay_from:
            m_T = global_range(H, 0, N, tail_tolerance, m_min=max(observed_max, m_axis, H))
            pred = _predict_cached(cache, H, 0, N, m_T, tuple(models), settings)
            frame.predictions = pred
            frame.scores = _score_frame(H, N, counts, pred, m_lo, m_hi, m_T, models)
        frames.append(frame)
    return frames


def _score_frame(
    H: int,
    N: int,
    counts: dict[int, int],
    pred: dict[str, Any],
    m_lo: int,
    m_hi: int,
    m_T: int,
    models: tuple[str, ...],
) -> dict[str, Any]:
    """Score every valid model of ``pred`` against the counts at one checkpoint."""
    P = {m: v / N for m, v in counts.items()}
    entry: dict[str, Any] = {"central": {}, "global": {}, "mass": {}, "ratio": {}, "m_T": m_T}
    tails: dict[str, float] = {}
    for name in models:
        values = pred.get(name)
        if values is None or np.any(np.isnan(values)):
            entry["central"][name] = None
            entry["global"][name] = None
            entry["mass"][name] = None
            continue
        if name == "F":
            tails[name] = tail_bound(H, 0, N, m_T, corrected=True)
        elif name == "F0":
            tails[name] = tail_bound(H, 0, N, m_T, corrected=False)
        elif name == "Q_mu":
            tails[name] = tail_bound_local(H, pred["mu"], m_T)
        elif name == "Q_lambda":
            tails[name] = tail_bound_local(H, pred["lambda"], m_T)
        else:  # binomials: support ends at H <= m_T
            tails[name] = 0.0
        # F and F0 sum to 1 - 2/N over all m; Q and the constant-density
        # binomial sum to 1; the averaged binomial omits [2, e), whose
        # contribution is only bounded (by 'omitted_bound' in B_avg_info).
        expected_deficit: float | None = 2 / N if name in ("F", "F0") else 0.0
        if name == "B_avg":
            expected_deficit = None
        entry["central"][name] = discrepancy(P, values, m_lo, m_hi).as_dict()
        entry["global"][name] = discrepancy(P, values, 0, m_T, tails[name]).as_dict()
        entry["mass"][name] = mass_report(values, expected_deficit).as_dict()
    for which in ("central", "global"):
        f, f0 = entry[which].get("F"), entry[which].get("F0")
        entry["ratio"][which] = {
            key: (f[key] / f0[key] if f and f0 and f0[key] > 1e-15 else math.nan)
            for key in ("E1", "E2", "Einf")
        }
    return entry


def frames_to_tables(frames: list[Frame], H: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten frames into a per-``(N, m)`` table and a per-``N`` summary table.

    Parameters
    ----------
    frames : list of Frame
        As returned by :func:`build_frames`.
    H : int
        Interval length (recorded in every row).

    Returns
    -------
    tuple of pandas.DataFrame
        ``(long, summary)``.  ``long`` has one row per frame and ``m`` on the
        shared axis with the count, ``P`` and every model's value (``nan``
        where not computed).  ``summary`` has one row per frame with the
        empirical statistics, ``mu``, ``lambda``, the quadrature error
        estimate, the central and global scores of every model, the mass
        reports and the ``F``-to-``F0`` ratios.
    """
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for fr in frames:
        pred = fr.predictions or {}
        for m in range(fr.m_axis + 1):
            row: dict[str, Any] = {
                "H": H,
                "N": fr.N,
                "m": m,
                "count": fr.counts.get(m, 0),
                "P": fr.P[m],
            }
            for name in ("F", "F0", "correction", "B_const", "Q_mu", "Q_lambda", "B_avg"):
                if name in pred:
                    row[name] = float(pred[name][m])
            long_rows.append(row)
        srow: dict[str, Any] = {
            "H": H,
            "N": fr.N,
            "mean": fr.mean,
            "var": fr.var,
            "median": fr.median,
            "mode": ",".join(str(x) for x in fr.mode),
            "max_m_observed": max(fr.counts) if fr.counts else 0,
        }
        if fr.predictions is not None:
            srow["mu"] = pred["mu"]
            srow["lambda"] = pred["lambda"]
            srow["quad_error"] = pred.get("quadrature", {}).get("error", math.nan)
            srow["quad_neval"] = pred.get("quadrature", {}).get("neval", 0)
        if fr.scores is not None:
            srow["m_T"] = fr.scores["m_T"]
            first = next((s for s in fr.scores["central"].values() if s is not None), None)
            if first is not None:
                srow["central_m_lo"], srow["central_m_hi"] = first["m_lo"], first["m_hi"]
            for which in ("central", "global"):
                for name, sc in fr.scores[which].items():
                    if sc is None:
                        continue
                    for key in ("E1", "E2", "Einf"):
                        srow[f"{which}_{key}_{name}"] = sc[key]
                    if which == "global":
                        srow[f"global_tail_bound_{name}"] = sc["predicted_tail_bound"]
                for key in ("E1", "E2", "Einf"):
                    srow[f"ratio_{which}_{key}"] = fr.scores["ratio"][which][key]
            for name, mass in fr.scores["mass"].items():
                if mass is None:
                    continue
                srow[f"deficit_{name}"] = mass["deficit"]
                srow[f"negative_mass_{name}"] = mass["negative_mass"]
                srow[f"min_value_{name}"] = mass["min_value"]
        summary_rows.append(srow)
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def write_tables(
    frames: list[Frame], H: int, directory: str | Path, stem: str | None = None
) -> tuple[Path, Path]:
    """Write the two tables of :func:`frames_to_tables` as CSV files.

    Parameters
    ----------
    frames : list of Frame
        As returned by :func:`build_frames`.
    H : int
        Interval length.
    directory : str or Path
        Output directory (created if needed).
    stem : str, optional
        File name stem (default ``cumulative_H{H}``).

    Returns
    -------
    tuple of Path
        The paths of the long table and the summary table.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = stem or f"cumulative_H{H}"
    long, summary = frames_to_tables(frames, H)
    p1 = directory / f"{stem}_by_N_m.csv"
    p2 = directory / f"{stem}_by_N.csv"
    long.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)
    return p1, p2
