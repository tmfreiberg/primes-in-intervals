"""Independent reference counts for validating the sliding-window counters.

The counters in :mod:`primes_in_intervals.intervals` advance between
prime-crossing events and share one prime generator,
:func:`~primes_in_intervals.sieve.postponed_sieve`.  Agreement between
:func:`~primes_in_intervals.intervals.overlap` and
:func:`~primes_in_intervals.intervals.overlap_cp` therefore says nothing about
either one's correctness, and neither does agreement with the original script
they were transcribed from.  This module provides the slow, transparent
calculation they are checked against:

* a prime table built by a plain array sieve of Eratosthenes
  (:func:`prime_table`), which shares no code with the postponed sieve, and
  which is itself cross-checked against trial division
  (:func:`is_prime_trial`) in the test-suite;
* :func:`overlap_reference`, which visits every starting point ``a`` in
  ``(A, B]`` and evaluates ``pi(a + H) - pi(a)`` from a prefix-sum table;
* :func:`overlap_cp_reference`, the checkpoint form of the same thing.

Everything here is quadratic-ish in the range and meant for ranges of at most
a few million integers.  Nothing in the package's counting path imports this
module.
"""

from __future__ import annotations

import math

import numpy as np

from primes_in_intervals.intervals import Dataset, zeros

__all__ = [
    "first_moment_identity",
    "is_prime_trial",
    "overlap_cp_reference",
    "overlap_reference",
    "overlap_reference_segment",
    "prime_pi_table",
    "prime_table",
    "prime_table_segment",
]


def is_prime_trial(n: int) -> bool:
    """Decide primality by trial division up to the square root of ``n``.

    Parameters
    ----------
    n : int
        Any integer; negatives, 0 and 1 are not prime.

    Returns
    -------
    bool
        Whether ``n`` is prime.
    """
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def prime_table(limit: int) -> np.ndarray:
    """Return a boolean array ``t`` with ``t[n]`` true exactly when ``n`` is prime.

    A textbook array sieve of Eratosthenes over ``0 .. limit`` inclusive.  It
    is independent of :func:`~primes_in_intervals.sieve.postponed_sieve`,
    which is the point.

    Parameters
    ----------
    limit : int
        Largest integer covered (inclusive); may be negative, giving an empty
        table.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``max(limit, -1) + 1``.
    """
    if limit < 0:
        return np.zeros(0, dtype=bool)
    t = np.ones(limit + 1, dtype=bool)
    t[:2] = False
    for p in range(2, math.isqrt(limit) + 1):
        if t[p]:
            t[p * p :: p] = False
    return t


def prime_table_segment(lo: int, hi: int) -> np.ndarray:
    """Return a boolean array ``s`` with ``s[n - lo]`` true exactly when ``n`` is prime.

    A segmented sieve over ``lo .. hi`` inclusive: the base primes up to
    ``isqrt(hi)`` come from :func:`prime_table`, and their multiples are
    struck out of the segment.  This lets the reference counts be evaluated
    on a narrow range far from the origin (around ``10**9``, say) without
    tabulating everything below it.

    Parameters
    ----------
    lo, hi : int
        Segment endpoints (inclusive); ``lo`` may be 0 or negative.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``hi - lo + 1`` (empty when ``hi < lo``).
    """
    if hi < lo:
        return np.zeros(0, dtype=bool)
    s = np.ones(hi - lo + 1, dtype=bool)
    for n in range(lo, min(hi, 1) + 1):
        s[n - lo] = False  # 0, 1 and negatives are not prime
    base = prime_table(math.isqrt(max(hi, 0)))
    for q in np.flatnonzero(base):
        p = int(q)
        start = max(p * p, ((lo + p - 1) // p) * p)
        if start > hi:
            continue
        s[start - lo :: p] = False
    return s


def overlap_reference_segment(A: int, B: int, H: int) -> dict[int, int]:
    """As :func:`overlap_reference`, but using a segmented sieve over ``(A, B + H]``.

    Parameters
    ----------
    A, B : int
        The left endpoint ``a`` runs over ``(A, B]``; empty when ``B <= A``.
    H : int
        Interval length, ``H >= 1``.

    Returns
    -------
    dict
        ``{m: count}`` over the nonzero counts.
    """
    if B <= A:
        return {}
    seg = prime_table_segment(A + 1, B + H)
    # cum[j] = number of primes in (A, A + j]
    cum = np.concatenate([[0], np.cumsum(seg.astype(np.int64))])
    a = np.arange(A + 1, B + 1) - A
    m = cum[a + H] - cum[a]
    values, counts = np.unique(m, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def prime_pi_table(limit: int) -> np.ndarray:
    """Return the prefix sums ``pi(0), pi(1), ..., pi(limit)`` of :func:`prime_table`.

    Parameters
    ----------
    limit : int
        Largest argument covered (inclusive).

    Returns
    -------
    numpy.ndarray
        Integer array with ``pi[n]`` the number of primes ``<= n``.
    """
    return np.cumsum(prime_table(limit).astype(np.int64))


def overlap_reference(A: int, B: int, H: int, pi: np.ndarray | None = None) -> dict[int, int]:
    """Count primes in ``(a, a + H]`` for every integer ``a`` with ``A < a <= B``, directly.

    Every starting point is visited; no event-jumping, no generators.  The
    result has the same shape as :func:`~primes_in_intervals.intervals.overlap`:
    ``{m: count}`` restricted to nonzero counts.

    Parameters
    ----------
    A, B : int
        The left endpoint ``a`` runs over ``(A, B]``; empty when ``B <= A``.
    H : int
        Interval length, ``H >= 1``.
    pi : numpy.ndarray, optional
        A prefix-sum table from :func:`prime_pi_table` covering at least
        ``B + H``; built on demand when omitted.

    Returns
    -------
    dict
        ``{m: count}`` over the nonzero counts.
    """
    if B <= A:
        return {}
    if pi is None or len(pi) <= B + H:
        pi = prime_pi_table(B + H)
    a = np.arange(A + 1, B + 1)
    m = pi[a + H] - pi[a]
    values, counts = np.unique(m, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def overlap_cp_reference(C: list[int], H: int) -> Dataset:
    """Checkpoint form of :func:`overlap_reference`, built one checkpoint at a time.

    The checkpoints are sorted and deduplicated, exactly as the fixed
    :func:`~primes_in_intervals.intervals.overlap_cp` does, and each
    checkpoint ``c`` gets the counts for ``a`` in ``(C[0], c]``.  The
    ``'header'`` matches the counter's, so the two datasets can be compared
    item by item.

    Parameters
    ----------
    C : list of int
        Checkpoints (any order, duplicates allowed).
    H : int
        Interval length.

    Returns
    -------
    dict
        Meta-dictionary with ``'header'`` and ``'data'`` items.
    """
    Cs = sorted(set(C))
    pi = prime_pi_table(Cs[-1] + H)
    data = {c: {m: 0 for m in range(H + 1)} for c in Cs}
    for c in Cs[1:]:
        for m, count in overlap_reference(Cs[0], c, H, pi).items():
            data[c][m] = count
    return {
        "header": {
            "interval_type": "overlap",
            "lower_bound": Cs[0],
            "upper_bound": Cs[-1],
            "interval_length": H,
            "no_of_checkpoints": len(Cs),
            "contents": ["data"],
        },
        "data": zeros(data),
    }


def first_moment_identity(A: int, B: int, H: int, pi: np.ndarray | None = None) -> int:
    """Return ``sum_{h=1}^{H} [pi(B + h) - pi(A + h)]``.

    Summing ``pi(a + H) - pi(a)`` over ``A < a <= B`` and exchanging the order
    of summation gives this quantity, so it must equal ``sum_m m g(m)`` for the
    histogram ``g`` of the overlapping counts on ``(A, B]``.  Evaluated from
    the independent prime table.

    Parameters
    ----------
    A, B : int
        Range endpoints.
    H : int
        Interval length.
    pi : numpy.ndarray, optional
        Prefix-sum table covering at least ``B + H``.

    Returns
    -------
    int
        The value of the sum (zero when ``B <= A``).
    """
    if B <= A:
        return 0
    if pi is None or len(pi) <= B + H:
        pi = prime_pi_table(B + H)
    h = np.arange(1, H + 1)
    return int(np.sum(pi[B + h] - pi[A + h]))
