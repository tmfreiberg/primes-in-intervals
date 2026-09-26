r"""Discrepancy measures between empirical frequencies and predictions.

For an empirical distribution ``P_m`` (relative frequencies at integer
``m``) and a prediction ``q_m``, the measures reported are

    E_1(q)        = sum_m |P_m - q_m|,
    E_2(q)        = sqrt( sum_m (P_m - q_m)^2 ),
    E_infinity(q) = max_m |P_m - q_m|,

with the summation range stated explicitly in every result.  Two ranges are
distinguished:

* a fixed **central** range ``m_lo <= m <= m_hi`` chosen once for a whole
  dataset (the same for every checkpoint and every prediction), and
* a **global** range ``0 <= m <= m_T`` whose upper end is chosen so that the
  predicted mass beyond it is provably below a stated tolerance
  (:func:`tail_bound`); the bound is reported alongside the scores.

Both ranges include every integer between their ends, so zero-frequency bins
and integers that never occur in the data contribute their full
``|0 - q_m|``.  Nothing is rounded, nothing is renormalized, and negative
predicted values enter the sums as they are.

These are discrepancy measures between an empirical distribution and a
signed, not necessarily normalized, approximation; they are not total
variation distances between probability distributions, and no likelihood,
chi-square, or divergence is computed.  The data are deterministic and the
starting points strongly overlapping, so no sampling standard errors are
attached.  Separately reported are the mass deficit of each prediction
(``1 - sum_m q_m`` over the global range) and its negative mass
(``sum of -q_m over q_m < 0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from primes_in_intervals.predictions import eta, poisson_pmf

__all__ = [
    "DiscrepancyScores",
    "discrepancy",
    "global_range",
    "mass_report",
    "pad",
    "tail_bound",
    "tail_bound_local",
]


@dataclass
class DiscrepancyScores:
    """The three measures on one explicit range.

    Attributes
    ----------
    m_lo, m_hi : int
        The range summed over (inclusive).
    E1, E2, Einf : float
        The measures.
    n_terms : int
        ``m_hi - m_lo + 1``.
    empirical_outside : float
        Empirical mass at ``m`` outside the range (nonzero only when the
        range fails to cover the data).
    predicted_tail_bound : float or None
        For the global range, a bound on ``sum_{m > m_hi} |q_m|``; ``None``
        for a central range, where no such claim is made.
    """

    m_lo: int
    m_hi: int
    E1: float
    E2: float
    Einf: float
    n_terms: int
    empirical_outside: float
    predicted_tail_bound: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the scores as a plain dictionary."""
        return {
            "m_lo": self.m_lo,
            "m_hi": self.m_hi,
            "E1": self.E1,
            "E2": self.E2,
            "Einf": self.Einf,
            "n_terms": self.n_terms,
            "empirical_outside": self.empirical_outside,
            "predicted_tail_bound": self.predicted_tail_bound,
        }


def pad(values: dict[int, float] | np.ndarray, m_lo: int, m_hi: int) -> np.ndarray:
    """Return ``values`` as a dense array over ``m_lo .. m_hi``, zero where undefined.

    Parameters
    ----------
    values : dict or array
        Either ``{m: value}`` or an array indexed from ``m = 0``.
    m_lo, m_hi : int
        Range (inclusive).

    Returns
    -------
    numpy.ndarray
        Length ``m_hi - m_lo + 1``.
    """
    out = np.zeros(m_hi - m_lo + 1)
    if isinstance(values, dict):
        for m, v in values.items():
            if m_lo <= m <= m_hi:
                out[m - m_lo] = v
    else:
        arr = np.asarray(values, dtype=float)
        lo, hi = max(m_lo, 0), min(m_hi, len(arr) - 1)
        if hi >= lo:
            out[lo - m_lo : hi - m_lo + 1] = arr[lo : hi + 1]
    return out


def discrepancy(
    P: dict[int, float] | np.ndarray,
    q: dict[int, float] | np.ndarray,
    m_lo: int,
    m_hi: int,
    predicted_tail_bound: float | None = None,
) -> DiscrepancyScores:
    """Compute ``E_1``, ``E_2`` and ``E_infinity`` between ``P`` and ``q`` on ``m_lo .. m_hi``.

    Parameters
    ----------
    P : dict or array
        Empirical relative frequencies (``{m: P_m}`` or array from ``m = 0``).
    q : dict or array
        Predicted values, unrounded, possibly negative.
    m_lo, m_hi : int
        The explicit summation range (inclusive).
    predicted_tail_bound : float, optional
        Recorded in the result for a global range.

    Returns
    -------
    DiscrepancyScores
    """
    if m_hi < m_lo:
        raise ValueError("empty comparison range")
    p = pad(P, m_lo, m_hi)
    if isinstance(P, dict):
        outside = sum(v for m, v in P.items() if not m_lo <= m <= m_hi)
    else:
        arr = np.asarray(P, dtype=float)
        mask = np.ones(len(arr), dtype=bool)
        mask[max(m_lo, 0) : m_hi + 1] = False
        outside = float(arr[mask].sum())
    d = p - pad(q, m_lo, m_hi)
    return DiscrepancyScores(
        m_lo=m_lo,
        m_hi=m_hi,
        E1=float(np.sum(np.abs(d))),
        E2=float(math.sqrt(np.sum(d * d))),
        Einf=float(np.max(np.abs(d))),
        n_terms=m_hi - m_lo + 1,
        empirical_outside=float(outside),
        predicted_tail_bound=predicted_tail_bound,
    )


def _weighted_poisson_terms(
    u: float, m_start: int, weight: Any, max_terms: int = 5000
) -> np.ndarray:
    """Return the terms ``weight(m) p_m(u)`` for ``m = m_start, ..., m_start + max_terms - 1``.

    The weights are evaluated on the whole vector of ``m`` at once.
    """
    m = np.arange(m_start, m_start + max_terms)
    return np.asarray(weight(m), dtype=float) * poisson_pmf(m, u)


def _tail_from_terms(terms: np.ndarray) -> float:
    """Sum the terms and add a geometric bound for whatever lies beyond the last one.

    The terms of a weighted Poisson tail decrease once ``m`` is well beyond
    ``u``; with 5000 terms they underflow to exactly zero long before the
    end, so the remainder is zero.  Should the last term still be positive,
    the ratio of the last two terms (when below one) bounds the geometric
    remainder; otherwise no bound is claimed (``inf``).
    """
    total = float(terms.sum())
    last, prev = terms[-1], terms[-2]
    if last == 0.0:
        return total
    if 0 < last < prev:  # pragma: no cover - needs more than 5000 relevant terms
        ratio = last / prev
        return total + last * ratio / (1 - ratio)
    return math.inf  # pragma: no cover


def _poisson_tail_series(u: float, m_start: int, weight: Any, max_terms: int = 5000) -> float:
    """Return ``sum_{m >= m_start} weight(m) p_m(u)`` (see :func:`_tail_from_terms`)."""
    return _tail_from_terms(_weighted_poisson_terms(u, m_start, weight, max_terms))


def tail_bound(H: float, M: float, N: float, m_T: int, corrected: bool = True) -> float:
    r"""Bound the predicted mass beyond ``m_T``: ``sum_{m > m_T} |F(m; H, M, N)|``.

    For every ``t`` in ``[max(2, M), M + N]`` the Poisson parameter
    ``u = H / log t`` lies in ``[u_min, u_max]`` with ``u_max = H / log
    max(2, M)``.  For ``m >= u_max``, ``p_m(u)`` is increasing in ``u`` on
    ``[0, m]``, so ``p_m(u) <= p_m(u_max)``; and
    ``|(m - u)^2 - m| <= (m - u_min)^2 + m``.  Hence

        |F(m)| <= p_m(u_max) * [1 + eta(H)/2 * ((m - u_min)^2 + m)]

    (the bracket is dropped when ``corrected`` is false, giving the bound
    for ``F_0``), and the tail is bounded by the sum of the right-hand side
    over ``m > m_T``.  The averaging over ``t`` can only help, so this is a
    rigorous, if crude, bound.

    Parameters
    ----------
    H, M, N : int or float
        Parameters of the prediction.
    m_T : int
        Last ``m`` included in a global comparison range.
    corrected : bool, optional
        Bound ``F`` (default) or ``F_0``.

    Returns
    -------
    float
        The bound; ``inf`` when ``m_T < u_max`` (the monotonicity argument
        needs ``m > u_max``), signalling that ``m_T`` must be raised.
    """
    lower, upper = max(2.0, float(M)), float(M) + float(N)
    if upper <= lower:
        return 0.0
    u_max = H / math.log(lower)
    u_min = H / math.log(upper)
    if m_T + 1 < u_max:
        return math.inf
    eta_H = eta(H)
    if corrected:

        def weight(m: int) -> float:
            return 1 + eta_H / 2 * ((m - u_min) ** 2 + m)

    else:

        def weight(m: int) -> float:
            return 1.0

    return _poisson_tail_series(u_max, m_T + 1, weight)


def tail_bound_local(H: float, u: float, m_T: int, corrected: bool = True) -> float:
    """Return ``sum_{m > m_T} |Q(m; u, H)|`` (or of ``p_m(u)``), summed until the terms vanish.

    Parameters
    ----------
    H : int or float
        Interval length.
    u : float
        The Poisson parameter of the local expression.
    m_T : int
        Last ``m`` included in the range.
    corrected : bool, optional
        Sum ``|Q|`` (default) or ``p_m``.

    Returns
    -------
    float
    """
    if u <= 0:
        return 0.0
    eta_H = eta(H)
    if corrected:

        def weight(m: int) -> float:
            return abs(1 - eta_H / 2 * ((m - u) ** 2 - m))

    else:

        def weight(m: int) -> float:
            return 1.0

    return _poisson_tail_series(u, m_T + 1, weight)


def global_range(H: float, M: float, N: float, tolerance: float = 1e-12, m_min: int = 0) -> int:
    """Return the least ``m_T >= m_min`` with :func:`tail_bound` ``(H, M, N, m_T) <= tolerance``.

    Parameters
    ----------
    H, M, N : int or float
        Parameters of the prediction.
    tolerance : float, optional
        Target for the omitted predicted mass (default ``1e-12``).
    m_min : int, optional
        Lower limit for ``m_T`` (for instance the largest observed count).

    Returns
    -------
    int
    """
    lower, upper = max(2.0, float(M)), float(M) + float(N)
    if upper <= lower:
        return m_min
    u_max, u_min = H / math.log(lower), H / math.log(upper)
    start = max(m_min, int(math.ceil(u_max)))
    eta_H = eta(H)
    # The bound for m_T is the sum of the weighted terms over m > m_T; the
    # reverse cumulative sum gives every candidate at once.
    terms = _weighted_poisson_terms(
        u_max, start + 1, lambda m: 1 + eta_H / 2 * ((m - u_min) ** 2 + m)
    )
    remainder = _tail_from_terms(terms) - float(terms.sum())
    tails = np.cumsum(terms[::-1])[::-1] + remainder  # tails[j] = bound for m_T = start + j
    hits = np.flatnonzero(tails <= tolerance)
    if len(hits) == 0:  # pragma: no cover - would need an absurd tolerance
        raise ValueError("no m_T within 5000 terms meets the tolerance")
    return start + int(hits[0])


@dataclass
class MassReport:
    """Mass bookkeeping of one prediction over the global range.

    Attributes
    ----------
    total : float
        ``sum_m q_m`` over ``0 .. m_T``.
    deficit : float
        ``1 - total``.
    expected_deficit : float or None
        The deficit implied by the formula when known (``max(2, M) - M) / N``
        for the integrated predictions), else ``None``.
    negative_mass : float
        ``sum of -q_m over q_m < 0``.
    n_negative : int
        Number of negative entries.
    min_value : float
        The least ``q_m``.
    """

    total: float
    deficit: float
    expected_deficit: float | None
    negative_mass: float
    n_negative: int
    min_value: float
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary."""
        return {
            "total": self.total,
            "deficit": self.deficit,
            "expected_deficit": self.expected_deficit,
            "negative_mass": self.negative_mass,
            "n_negative": self.n_negative,
            "min_value": self.min_value,
            **self.extra,
        }


def mass_report(q: np.ndarray, expected_deficit: float | None = None) -> MassReport:
    """Summarize the mass of a prediction vector ``q`` indexed from ``m = 0``.

    Parameters
    ----------
    q : array
        Prediction values at ``m = 0, 1, ..., m_T``.
    expected_deficit : float, optional
        The formula's own mass deficit, for comparison.

    Returns
    -------
    MassReport
    """
    arr = np.asarray(q, dtype=float)
    neg = arr[arr < 0]
    return MassReport(
        total=float(arr.sum()),
        deficit=float(1 - arr.sum()),
        expected_deficit=expected_deficit,
        negative_mass=float(-neg.sum()) if len(neg) else 0.0,
        n_negative=int(len(neg)),
        min_value=float(arr.min()) if len(arr) else math.nan,
    )
