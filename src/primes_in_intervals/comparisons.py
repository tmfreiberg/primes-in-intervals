"""Comparing empirical counts against the theoretical predictions.

:func:`compare` attaches, for every checkpoint (or nested interval) and every
integer prime count ``m`` from ``0`` to the largest count observed anywhere in
the dataset, a pair of tuples: the values ``(actual, B, F, F0)`` as
probabilities and the same as (unrounded) numbers of intervals.  Here, for a
checkpoint ``c`` of a dataset with lower bound ``A``, the manuscript's
parameters are ``M = A`` and ``N = c - A`` (for a nested interval
``(c[0], c[1]]``, ``M = c[0]`` and ``N = c[1] - c[0]``), and

* ``B`` is the constant-density binomial ``Binom(H, mu/H)`` with ``mu`` the
  averaged parameter ``(H/N) int_{max(2,M)}^{M+N} dt/log t``,
* ``F`` is the integrated corrected prediction ``F(m; H, M, N)``,
* ``F0`` is the integrated Poisson expression ``F_0(m; H, M, N)``.

The former normalization by the density ``1/(log N - 1)`` at the midpoint
``N`` of the range is gone: it was inappropriate for narrow ranges centered
far from the origin, where the averaged density is essentially ``1/log N``
with no shift.  Every prediction is evaluated through
:func:`~primes_in_intervals.predictions.predict_all`, the same routine the
plots use.

:func:`score` is the principal assessment: the discrepancy measures
``E_1``, ``E_2`` and ``E_infinity`` of each prediction on an explicit common
range of ``m`` (a fixed central range, and a global range with the omitted
predicted tail bounded), together with mass deficits, negative mass, and the
ratios of corrected to uncorrected discrepancies.  :func:`winners` is the
older per-bin scoreboard, kept for the tables that use it; it ignores the
magnitude of the discrepancies and is not the principal assessment.

The refined predictions were derived for overlapping intervals only; the
comparisons are nevertheless permitted for disjoint and prime-start data (the
second-order terms may differ in those cases), and the display layer carries a
reminder to that effect.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from primes_in_intervals.discrepancies import (
    discrepancy,
    global_range,
    mass_report,
    tail_bound,
)
from primes_in_intervals.intervals import Dataset
from primes_in_intervals.predictions import predict_all
from primes_in_intervals.quadrature import QuadratureSettings

__all__ = ["COMPARISON_LABEL", "compare", "score", "winners"]

#: The header ``'contents'`` entry recorded by :func:`compare`.
COMPARISON_LABEL = "comparison - actual, binomial, F, F0"

#: Position of each prediction in the comparison tuples.
_SLOT = {"B": 1, "F": 2, "F0": 3}


def _range_parameters(dataset: Dataset, c: Any) -> tuple[int, int, int]:
    """Return ``(M, N, multiplier)`` for checkpoint ``c``: range and number of intervals."""
    interval_type = dataset["header"]["interval_type"]
    H = dataset["header"]["interval_length"]
    if isinstance(c, tuple):
        M, N = c[0], c[1] - c[0]
        counts = dataset["nested_interval_data"][c]
    else:
        A = dataset["header"]["lower_bound"]
        M, N = A, c - A
        counts = dataset["data"][c]
    if interval_type == "overlap":
        multiplier = N
    elif interval_type == "disjoint":
        multiplier = N // H
    else:  # prime_start
        multiplier = sum(counts.values())
    return M, N, multiplier


def _m_max(dataset: Dataset, datakey: str) -> int:
    """Return the largest ``m`` with a nonzero count anywhere in the dataset."""
    best = 0
    for counts in dataset[datakey].values():
        for m, v in counts.items():
            if v and m > best:
                best = m
    return best


def compare(
    dataset: Dataset,
    settings: QuadratureSettings | None = None,
) -> Dataset | None:
    """Attach prediction comparisons to an analyzed dataset, in place.

    For each checkpoint ``c`` (or nested interval ``c = (c[0], c[1])``) and
    each integer ``m`` from ``0`` to the largest count observed in the
    dataset, the ``'comparison'`` item receives::

        (dist, B, F, F0), (count, B * n, F * n, F0 * n)

    where ``dist`` and ``count`` are the empirical relative frequency and
    count (zero for an ``m`` that never occurs at ``c``), ``n`` is the number
    of intervals sampled (``N`` for overlapping data, ``N // H`` for disjoint
    data, the total prime count for prime-start data), and nothing is
    rounded.  The parameters used at each checkpoint (``M``, ``N``, ``mu``,
    ``lambda``, the quadrature diagnostics and the validity flags) are stored
    under ``'prediction_parameters'``.

    Parameters
    ----------
    dataset : dict
        An analyzed dataset (run
        :func:`~primes_in_intervals.statistics.analyze` first).
    settings : QuadratureSettings, optional
        Quadrature tolerances for ``F`` and ``F0``.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if the dataset has
        no data or has not been analyzed.
    """
    if "data" in dataset.keys():
        datakey = "data"
    elif "nested_interval_data" in dataset.keys():
        datakey = "nested_interval_data"
    else:
        return print("No data to compare.")
    if "distribution" not in dataset.keys():
        return print(
            "Analyze data first, to obtain distribution data for comparison "
            "with theoretical predictions."
        )
    H = dataset["header"]["interval_length"]
    C = list(dataset[datakey].keys())
    if datakey == "data":
        C.sort()
    m_max = _m_max(dataset, datakey)
    comparison: dict = {}
    parameters: dict = {}
    if datakey == "data":
        # For consistency with the keys: the initial checkpoint carries no intervals.
        comparison[C[0]] = {m: 0 for m in range(m_max + 1)}
        C = C[1:]
    for c in C:
        M, N, multiplier = _range_parameters(dataset, c)
        pred = predict_all(H, M, N, m_max, models=("F", "F0", "B_const"), settings=settings)
        parameters[c] = {
            "M": M,
            "N": N,
            "intervals": multiplier,
            "mu": pred["mu"],
            "lambda": pred["lambda"],
            "quadrature": pred["quadrature"],
            "validity": pred["validity"],
            "formula_version": pred["formula_version"],
        }
        counts = dataset[datakey][c]
        dist = dataset["distribution"][c]
        comparison[c] = {}
        for m in range(m_max + 1):
            b, f, f0 = float(pred["B_const"][m]), float(pred["F"][m]), float(pred["F0"][m])
            comparison[c][m] = (
                (dist.get(m, 0.0), b, f, f0),
                (counts.get(m, 0), b * multiplier, f * multiplier, f0 * multiplier),
            )
    dataset["comparison"] = comparison
    dataset["prediction_parameters"] = parameters
    dataset["header"]["contents"].append(COMPARISON_LABEL)
    return dataset


def score(
    dataset: Dataset,
    m_lo: int = 0,
    m_hi: int | None = None,
    tail_tolerance: float = 1e-12,
    settings: QuadratureSettings | None = None,
) -> Dataset | None:
    """Attach discrepancy scores to a compared dataset, in place.

    For every checkpoint ``c`` and each prediction in the comparison
    (``'B'``, ``'F'``, ``'F0'``), the ``'scores'`` item records

    * ``'central'``: ``E_1``, ``E_2``, ``E_infinity`` on the fixed range
      ``m_lo .. m_hi`` (default: ``0`` to the largest count observed in the
      dataset), the same for every checkpoint;
    * ``'global'``: the same measures on ``0 .. m_T``, where ``m_T`` is the
      least value for which the predicted mass of ``F`` beyond it is
      provably below ``tail_tolerance`` (see
      :func:`~primes_in_intervals.discrepancies.tail_bound`), with that bound
      recorded; the binomial has no tail beyond ``H``;
    * ``'mass'``: total, deficit, expected deficit, negative mass and the
      least value of each prediction over ``0 .. m_T``;
    * ``'ratio'``: ``E(F) / E(F0)`` for each measure and range, ``nan`` when
      ``E(F0)`` is below ``1e-15``.

    Parameters
    ----------
    dataset : dict
        A dataset to which :func:`compare` has been applied.
    m_lo, m_hi : int, optional
        The fixed central range.
    tail_tolerance : float, optional
        Target for the omitted predicted mass of the global range.
    settings : QuadratureSettings, optional
        Quadrature tolerances for the extra ``m`` of the global range.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if it has not
        been compared.
    """
    if "comparison" not in dataset.keys():
        return print("Compare the data first, with the compare function.")
    H = dataset["header"]["interval_length"]
    datakey = "nested_interval_data" if "nested_interval_data" in dataset else "data"
    observed_max = _m_max(dataset, datakey)
    if m_hi is None:
        m_hi = observed_max
    scores: dict = {}
    for c, params in dataset["prediction_parameters"].items():
        counts = dataset[datakey][c]
        n = params["intervals"]
        P = {m: v / n for m, v in counts.items()} if n else {}
        M, N = params["M"], params["N"]
        m_T = global_range(H, M, N, tail_tolerance, m_min=max(observed_max, H))
        pred = predict_all(H, M, N, m_T, models=("F", "F0", "B_const"), settings=settings)
        q = {"B": pred["B_const"], "F": pred["F"], "F0": pred["F0"]}
        tails = {
            "B": 0.0,
            "F": tail_bound(H, M, N, m_T, corrected=True),
            "F0": tail_bound(H, M, N, m_T, corrected=False),
        }
        expected_deficit = (max(2, M) - M) / N if N else math.nan
        entry: dict[str, Any] = {"central": {}, "global": {}, "mass": {}, "ratio": {}}
        for name, values in q.items():
            valid = not np.any(np.isnan(values))
            if not valid:
                entry["central"][name] = None
                entry["global"][name] = None
                entry["mass"][name] = None
                continue
            entry["central"][name] = discrepancy(P, values, m_lo, m_hi).as_dict()
            entry["global"][name] = discrepancy(P, values, 0, m_T, tails[name]).as_dict()
            entry["mass"][name] = mass_report(
                values, expected_deficit if name in ("F", "F0") else 0.0
            ).as_dict()
        for which in ("central", "global"):
            f, f0 = entry[which].get("F"), entry[which].get("F0")
            entry["ratio"][which] = {
                key: (f[key] / f0[key] if f and f0 and f0[key] > 1e-15 else math.nan)
                for key in ("E1", "E2", "Einf")
            }
        entry["m_T"] = m_T
        scores[c] = entry
    dataset["scores"] = scores
    dataset["scores_range"] = {"central": (m_lo, m_hi), "tail_tolerance": tail_tolerance}
    dataset["header"]["contents"].append("scores")
    return dataset


def winners(dataset: Dataset) -> Dataset | None:
    """Score the three predictions per interval by the older per-bin scoreboard, in place.

    "Best" is judged in two senses.  First, the sum over ``m`` of the squared
    error between the actual and predicted (unrounded) interval counts, over
    the full run of ``m`` from the smallest to the largest with a nonzero
    comparison entry; the three predictions are ranked 1, 2, 3 by this
    score.  Second, for each ``m`` in that run, whichever prediction's count
    lands closest to the actual count "wins" that ``m`` (ties shared); the
    lists of ``m`` won and the resulting most/2nd-most/least tallies are
    recorded.

    The number of bins won ignores the size of the discrepancies, so this is
    a secondary summary; see :func:`score` for the principal assessment.

    The result is stored as a ``'winners'`` item, with keys ``'B sq error'``,
    ``'F sq error'``, ``'F0 sq error'``, ``1``, ``2``, ``3``,
    ``'B wins for m in '``, ``'F wins for m in '``, ``'F0 wins for m in '``,
    ``'most wins'``, ``'2nd most wins'``, and ``'least wins'`` per interval
    (``'B'`` the binomial, ``'F'`` the corrected prediction, ``'F0'`` the
    uncorrected one), and noted in the header's ``'contents'``.

    Parameters
    ----------
    dataset : dict
        A dataset to which :func:`compare` has been applied.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if comparisons are
        missing or winners already computed.
    """
    if "winners" in dataset.keys():
        return print("This function has already been applied to the data.")
    if "comparison" not in dataset.keys():
        return print(
            "Compare the data first, to obtain distribution data for comparison "
            "with theoretical predictions."
        )
    if "nested_interval_data" in dataset.keys():
        datakey = "nested_interval_data"
    elif "data" in dataset.keys():
        datakey = "data"
    else:
        return print("No data.")
    names = ["B", "F", "F0"]
    C = list(dataset[datakey].keys())
    win: dict = {}
    for c in C:
        win[c] = {}
        M = [m for m in dataset["comparison"][c].keys() if dataset["comparison"][c][m] != 0]
        if M != []:
            min_m, max_m = min(M), max(M)
            M = list(range(min_m, max_m + 1))
            sq: dict[str, float] = {}
            for name in names:
                slot = _SLOT[name]
                sq[name] = sum(
                    (dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][slot]) ** 2
                    for m in M
                )
                win[c][f"{name} sq error"] = sq[name]
            ranked = sorted(sq.values())
            for i in range(3):
                for name in names:
                    if ranked[i] == sq[name]:
                        win[c][i + 1] = name
            tallies = {name: 0 for name in names}
            for name in names:
                win[c][f"{name} wins for m in "] = []
            for m in M:
                actual = dataset["comparison"][c][m][1][0]
                diffs = {
                    name: abs(actual - dataset["comparison"][c][m][1][_SLOT[name]])
                    for name in names
                }
                min_diff = min(diffs.values())
                for name in names:
                    if diffs[name] == min_diff:
                        win[c][f"{name} wins for m in "].append(m)
                        tallies[name] += 1
            max_wins = sorted(tallies.values(), reverse=True)
            for label, rank in (("most wins", 0), ("2nd most wins", 1), ("least wins", 2)):
                win[c][label] = "".join(name for name in names if tallies[name] == max_wins[rank])
        if M == []:
            win[c] = {
                "B sq error": "-",
                "F sq error": "-",
                "F0 sq error": "-",
                1: "-",
                2: "-",
                3: "-",
                "B wins for m in ": "-",
                "F wins for m in ": "-",
                "F0 wins for m in ": "-",
                "most wins": "-",
                "2nd most wins": "-",
                "least wins": "-",
            }
    dataset["winners"] = win
    dataset["header"]["contents"].append("winners")
    return dataset
