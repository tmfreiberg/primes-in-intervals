"""Counting primes in intervals.

This module contains the project's core computations.  For an interval length
``H`` and a range ``(A, B]``, we count how many intervals of the form
``(a, a + H]`` contain exactly ``m`` primes, as ``a`` runs over

* multiples of ``H`` shifted by ``A`` ("disjoint" intervals, one count per
  block of ``H`` consecutive integers),
* every integer in ``(A, B]`` ("overlapping" intervals, a sliding window), or
* every prime in ``(A, B]`` ("prime-start" intervals).

Each counter comes in two forms: a plain form returning a single frequency
dictionary ``{m: count}``, and a checkpoint form (suffix ``_cp``) that sweeps
the primes once and records the cumulative frequencies at each element of a
checkpoint list ``C``, wrapped in a "meta-dictionary" with a ``'header'``
describing the computation and a ``'data'`` item mapping each checkpoint to its
frequency dictionary.  Sweeping once matters: prime generation dominates the
cost, and the checkpoint forms make an entire animation's worth of data
available for the price of the final frame.

The functions :func:`anyIntervals` and :func:`anyIntervals_cp` generalize all
of the above: the left endpoints ``a`` are drawn from one arbitrary strictly
increasing generator, and the objects counted inside ``(a, a + H]`` from
another.

A warning to maintainers: the sliding-window logic in :func:`overlap` and
:func:`overlap_cp`, and the endpoint handling throughout, are delicate.  The
loops advance two independent prime generators (one trailing the window's left
edge, one leading its right edge) and jump the window by runs of prime-free
integers rather than one step at a time; the closing ``while`` blocks handle
the final stretch where the trailing generator has already passed the upper
bound.  These functions are transcribed verbatim from the original project and
validated against it; do not "tidy" the arithmetic without re-running the
equivalence tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from primes_in_intervals.sieve import postponed_sieve

__all__ = [
    "COUNTER_VERSION",
    "anyIntervals",
    "anyIntervals_cp",
    "disjoint",
    "disjoint_cp",
    "intervals",
    "overlap",
    "overlap_cp",
    "overlap_extension",
    "prime_start",
    "prime_start_cp",
    "zeros",
]

MetaDict = dict[Any, dict[int, int]]
Dataset = dict[str, Any]

#: Version tag of the counting implementations in this module, recorded by
#: :func:`primes_in_intervals.dataio.save` alongside every stored row.  Bump
#: it whenever the arithmetic of a counter changes.
COUNTER_VERSION = "intervals-2026-09-26 (sliding window; validated against reference.py)"


def zeros(meta_dictionary: MetaDict, pad: str = "yes") -> MetaDict:
    """Normalize the zero-count entries across a meta-dictionary's values.

    A meta-dictionary here is a mapping whose values are frequency
    dictionaries ``{m: count}``.  With ``pad='yes'`` (the default), every inner
    dictionary is padded so that they all share the same keys: the sorted union
    of every ``m`` that has a nonzero count somewhere.  With ``pad='no'``,
    zero-count items are instead stripped from each inner dictionary.

    Padding to a common key set is what lets the checkpoint dictionaries be
    compared, differenced, tabulated, and plotted against one another.

    Parameters
    ----------
    meta_dictionary : dict
        Mapping from keys (checkpoints, say) to frequency dictionaries.
    pad : str, optional
        ``'no'`` to strip zero items; any other value (default ``'yes'``) to
        pad to the common key set.

    Returns
    -------
    dict
        A new meta-dictionary; the input is not modified.
    """
    output: MetaDict = {}
    if pad == "no":
        for k in meta_dictionary.keys():
            output[k] = {
                m: meta_dictionary[k][m]
                for m in meta_dictionary[k].keys()
                if meta_dictionary[k][m] != 0
            }
        return output
    # if pad option is 'yes' or unspecified or anything other than the string 'no'...
    padding: set[int] = set()
    for k in meta_dictionary.keys():
        padding = padding.union(
            [m for m in meta_dictionary[k] if meta_dictionary[k][m] != 0]
        )
    padding_list = list(padding)
    padding_list.sort()
    for k in meta_dictionary.keys():
        output[k] = {}
        for m in padding_list:
            if m in meta_dictionary[k].keys():
                output[k][m] = meta_dictionary[k][m]
            else:
                output[k][m] = 0
    return output


def disjoint(A: int, B: int, H: int) -> dict[int, int]:
    """Count primes in the disjoint intervals of length ``H`` spanning ``(A, B]``.

    The intervals are ``(A, A + H], (A + H, A + 2H], ..., (A + (K-1)H, A + KH]``
    where ``K = (B - A) // H`` (so a ragged final stretch of ``(A, B]`` shorter
    than ``H`` is ignored).  The returned dictionary has an item ``m: g(m)``
    for each ``m`` with ``g(m)`` nonzero, where ``g(m)`` is the number of these
    intervals containing exactly ``m`` primes.

    Parameters
    ----------
    A, B : int
        Range endpoints; intervals tile ``(A, B]`` from the left.
    H : int
        Interval length.

    Returns
    -------
    dict
        ``{m: g(m)}`` over the nonzero counts.

    Examples
    --------
    ``disjoint(0, H, H)`` has a single item ``{m: 1}`` with ``m`` the number of
    primes up to ``H``:

    >>> disjoint(0, 10, 10)
    {4: 1}
    """
    K = (B - A) // H
    B = A + K * H  # re-define B in case the inputs are not of this form
    # Initialize the output dictionary covering all possible values for m.
    output = {m: 0 for m in range(H + 1)}
    P = postponed_sieve()
    p = next(P)  # initialize p as 2
    a = A  # start of the first interval, viz. (A, A + H]
    while p < a + 1:
        p = next(P)  # p is now the prime after a (= A initially)
    m = 0  # initialize m as 0
    for k in range(1, K + 1):
        while p < a + k * H + 1:
            m += 1
            p = next(P)
        output[m] += 1
        m = 0
    # Remove m if there are no intervals with m primes.
    output = {m: output[m] for m in output.keys() if output[m] != 0}
    return output


def disjoint_cp(C: list[int], H: int) -> Dataset:
    """Checkpoint version of :func:`disjoint`: one sweep, cumulative counts.

    The checkpoint list is first snapped onto the arithmetic progression
    ``C[0] mod H``: each ``C[i]`` is replaced by
    ``C[0] + ((C[i] - C[0]) // H) * H``, and duplicates are removed.  (For
    example, ``C = [0, 10, 100, 210, 350, 400]`` with ``H = 100`` becomes
    ``[0, 100, 200, 300, 400]``.)  The output's ``'data'`` item then maps each
    snapped checkpoint ``N[k]`` to the frequency dictionary of
    ``disjoint(N[0], N[k], H)``, computed cumulatively so the primes are
    enumerated only once.

    Parameters
    ----------
    C : list of int
        Checkpoints; snapped and deduplicated as described.
    H : int
        Interval length.

    Returns
    -------
    dict
        Meta-dictionary with a ``'header'`` item (interval type ``'disjoint'``,
        bounds, length, number of checkpoints, contents) and a ``'data'`` item
        mapping each checkpoint to its cumulative frequency dictionary, padded
        to a common key set by :func:`zeros`.
    """
    P = postponed_sieve()
    p = next(P)
    # If, e.g., H = 100 and C = [0,10,100,210,350,400], then we replace C by
    # N = [0,100,200,300,400]...
    K, N = [], []
    for i in range(len(C)):
        K.append((C[i] - C[0]) // H)
        N.append(C[0] + K[i] * H)
    # Could have repeated elements: in above e.g., K = [0,0,1,2,3,4] and
    # N = [0,0,100,200,300,400], whence
    K = list(set(K))
    N = list(set(N))
    K.sort()
    N.sort()
    output: Dataset = {
        "header": {
            "interval_type": "disjoint",
            "lower_bound": N[0],
            "upper_bound": N[-1],
            "interval_length": H,
            "no_of_checkpoints": len(N),
            "contents": [],
        }
    }
    # OK now N = [0,100,200,300,400] in our e.g., and
    # [N_0, N_0 + K_1*H,...,N_0 + K_n*H] in general.
    data: MetaDict = {}
    for n in N:
        data[n] = {}
    data[N[0]] = {m: 0 for m in range(H + 1)}
    for i in range(1, len(N)):
        for m in data[N[i - 1]].keys():
            data[N[i]][m] = data[N[i - 1]][m]
        while p < N[i - 1] + 1:
            p = next(P)
        m = 0
        for k in range(1, (N[i] - N[i - 1]) // H + 1):
            while p < N[i - 1] + k * H + 1:
                m += 1
                p = next(P)
            data[N[i]][m] += 1
            m = 0
    trimmed_data = zeros(data)
    output["data"] = trimmed_data
    output["header"]["contents"].append("data")
    return output


def overlap(A: int, B: int, H: int) -> dict[int, int]:
    """Count primes in the sliding window ``(a, a + H]`` for every ``a`` in ``(A, B]``.

    The returned dictionary has an item ``m: h(m)`` for each ``m`` with
    ``h(m)`` nonzero, where ``h(m)`` is the number of integers ``a`` with
    ``A < a <= B`` such that ``(a, a + H]`` contains exactly ``m`` primes.
    Note that the first interval considered is ``(A + 1, A + 1 + H]``, so the
    first prime that can be counted is at least ``A + 2``.

    The implementation slides a window of length ``H`` and only does work when
    an endpoint crosses a prime.  Two independent prime generators are
    maintained: ``p`` (from ``P``) is always the first prime after the window's
    left endpoint ``a``, and ``q`` (from ``Q``) the first prime after its right
    endpoint ``a + H``.  Writing ``b = p - a`` and ``c = q - (a + H)``, the
    window can jump ``min(b, c) - 1`` steps with the count ``m`` unchanged;
    then, depending on which endpoint reaches its prime first, ``m`` decreases
    by one (left endpoint passes a prime), increases by one (right endpoint
    does), or stays put (both at once).  A separate closing loop handles the
    stretch after ``p`` has passed ``B``, where the jump must be truncated at
    ``B``.

    Parameters
    ----------
    A, B : int
        The left endpoint ``a`` runs over ``(A, B]``.
    H : int
        Interval (window) length.

    Returns
    -------
    dict
        ``{m: h(m)}`` over the nonzero counts.

    Examples
    --------
    For ``a = 1, ..., 5``, the windows ``(a, a + 5]`` contain 3, 3, 2, 2, 1
    primes respectively:

    >>> overlap(0, 5, 5)
    {1: 1, 2: 2, 3: 2}
    """
    P = postponed_sieve()  # We'll need two prime generators (see below).
    Q = postponed_sieve()
    # Initialize the output dictionary covering all possible values for m.
    output = {m: 0 for m in range(H + 1)}
    a = A + 1  # start of the first interval, viz. (A + 1, A + 1 + H]
    p, q = next(P), next(Q)  # initialize p and q as 2
    while p < a + 1:
        p, q = next(P), next(Q)  # p and q are now the prime after a
    m = 0  # initialize m as 0
    while q < a + H + 1:
        m += 1
        q = next(Q)
    # q is now the prime after a + H;
    # m is the number of primes in our first interval (a, a + H].
    #
    # From now on, imagine a sliding window of length H, starting at a.  We
    # have m primes in the window.  Move the window one to the right.  If the
    # left endpoint is prime while the right endpoint is not, we lose a prime:
    # m -> m - 1.  If the right endpoint is prime while the left is not, we
    # gain a prime: m -> m + 1.  Otherwise, m remains unchanged.  Thus, we only
    # need to update our dictionary when either the left or right endpoint
    # passes a prime.  E.g. if the next prime after a is p = a + 10 and the
    # next prime after a + H is q = a + H + 12, then (a', a' + H] contains m
    # primes for a' = a, a + 1, ..., a + 9, so we can just update our m-counter
    # by nine.  Also, (a + 10, a + 10 + H] now contains m - 1 primes.  We'd let
    # p = a + 10 become the new a, m - 1 the new m, p_next the new p, q remains
    # the same, etc.  We have a small problem if the jump exceeds B, so we
    # treat that with a separate loop at the end.
    while p < B + 1:
        output[m] += 1
        b, c = p - a, q - (a + H)  # p = a + b, q = a + H + c
        output[m] = output[m] + min(b, c) - 1
        if b == c:
            a = p
            p = next(P)
        if b < c:
            a, m = p, m - 1
            p = next(P)
        if c < b:
            a, m = a + c, m + 1
        while q < a + H + 1:
            q = next(Q)
    while a < B + 1:  # now the prime after a is also bigger than B
        output[m] += 1
        b, c = p - a, q - (a + H)  # p = a + b, q = a + H + c
        if a + min(b, c) > B:
            output[m] = output[m] + B - a
            break
        else:  # must be that c < b, because p = a + b > B.
            output[m] = output[m] + c - 1
            a, m = a + c, m + 1
            while q < a + H + 1:
                q = next(Q)
    output = {m: output[m] for m in output.keys() if output[m] != 0}
    return output


def overlap_cp(C: list[int], H: int) -> Dataset:
    """Checkpoint version of :func:`overlap`: one sweep, cumulative counts.

    The output's ``'data'`` item maps each checkpoint ``C[k]`` to the frequency
    dictionary of ``overlap(C[0], C[k], H)``, computed in a single pass over
    the primes.  The sliding-window mechanics are those of :func:`overlap`; at
    each checkpoint boundary the closing loop truncates the current jump, a
    snapshot of the running counts is stored, and the sweep resumes.

    Note that ``C`` is sorted and deduplicated in place, and that the header
    describes the sorted, deduplicated list: ``'lower_bound'`` is its least
    element, ``'upper_bound'`` its greatest, and ``'no_of_checkpoints'`` the
    number of distinct checkpoints, which is also the number of keys of the
    ``'data'`` item.

    Parameters
    ----------
    C : list of int
        Checkpoints.  Sorted and deduplicated in place.
    H : int
        Interval (window) length.

    Returns
    -------
    dict
        Meta-dictionary with a ``'header'`` item (interval type ``'overlap'``,
        bounds, length, number of checkpoints, contents) and a ``'data'`` item
        mapping each checkpoint to its cumulative frequency dictionary, padded
        to a common key set by :func:`zeros`.
    """
    # Sort and deduplicate before the header is built, so that the recorded
    # bounds and checkpoint count describe the list that is actually swept.
    C[:] = sorted(set(C))
    output: Dataset = {
        "header": {
            "interval_type": "overlap",
            "lower_bound": C[0],
            "upper_bound": C[-1],
            "interval_length": H,
            "no_of_checkpoints": len(C),
            "contents": [],
        }
    }
    data: MetaDict = {C[0]: {m: 0 for m in range(H + 1)}}
    P = postponed_sieve()
    Q = postponed_sieve()
    p, q = next(P), next(Q)
    m = 0
    current_data = {m: 0 for m in range(H + 1)}
    for i in range(1, len(C)):
        M, N = C[i - 1], C[i]
        a = M + 1
        while p < a + 1:
            m -= 1
            p = next(P)
        while q < a + H + 1:
            m += 1
            q = next(Q)
        while p < N + 1:
            current_data[m] += 1
            b, c = p - a, q - (a + H)
            current_data[m] = current_data[m] + min(b, c) - 1
            if b == c:
                a = p
                p = next(P)
            if b < c:
                a, m = p, m - 1
                p = next(P)
            if c < b:
                a, m = a + c, m + 1
            while q < a + H + 1:
                q = next(Q)
        while a < N + 1:
            current_data[m] += 1
            b, c = p - a, q - (a + H)
            if a + min(b, c) > N:
                current_data[m] = current_data[m] + N - a
                data[N] = {}
                for k in current_data.keys():
                    data[N][k] = current_data[k]
                break
            else:
                current_data[m] = current_data[m] + c - 1
                a, m = a + c, m + 1
                while q < a + H + 1:
                    q = next(Q)
    trimmed_data = zeros(data)
    output["data"] = trimmed_data
    output["header"]["contents"].append("data")
    return output


def prime_start(M: int, N: int, H: int) -> dict[int, int]:
    """Count primes in ``(p, p + H]`` for every prime ``p`` in ``(M, N]``.

    The returned dictionary has an item ``m: count`` for each ``m`` such that
    some prime ``p`` with ``M < p <= N`` has exactly ``m`` primes in
    ``(p, p + H]``.

    Two generators are used: ``p`` (from ``P``) runs over the interval
    starting points, and ``q`` (from ``Q``) leads it by the window length,
    counting primes in ``(p, p + H]``.  When ``p`` advances to the next prime,
    the count drops by exactly one, since the old ``p`` leaves the window
    ``(p, p + H]`` and the new window's extra content is picked up by advancing
    ``q``.

    Parameters
    ----------
    M, N : int
        The starting prime ``p`` runs over ``(M, N]``.
    H : int
        Interval length.

    Returns
    -------
    dict
        ``{m: count}`` over the nonzero counts.

    Examples
    --------
    For primes 11, 13, 17, 19 in ``(10, 20]`` and ``H = 20``, the intervals
    contain 6, 5, 5, 4 primes respectively:

    >>> prime_start(10, 20, 20)
    {4: 1, 5: 2, 6: 1}
    """
    P = postponed_sieve()
    Q = postponed_sieve()
    p = next(P)
    q = next(Q)
    while p <= M:
        p = next(P)
    while q <= p:
        q = next(Q)
    output = {m: 0 for m in range(H + 1)}
    m = 0
    while p <= N:
        while q <= p + H:
            m += 1
            q = next(Q)
        output[m] += 1
        p = next(P)
        m += -1
    output = {m: output[m] for m in range(H + 1) if output[m] != 0}
    return output


def prime_start_cp(C: list[int], H: int) -> Dataset:
    """Checkpoint version of :func:`prime_start`: one sweep, cumulative counts.

    The output's ``'data'`` item maps each checkpoint ``C[k]`` to the frequency
    dictionary of ``prime_start(C[0], C[k], H)``, computed in a single pass.

    Note that ``C`` is sorted and deduplicated in place, and that the loop's
    first iteration (``i = 0``) pairs ``C[-1]`` with ``C[0]``; since the prime
    supply has already been advanced past ``C[0]``, that iteration counts
    nothing and records the all-zero dictionary at ``C[0]``, exactly as
    intended.

    Parameters
    ----------
    C : list of int
        Checkpoints.  Sorted and deduplicated in place.
    H : int
        Interval length.

    Returns
    -------
    dict
        Meta-dictionary with a ``'header'`` item (interval type
        ``'prime_start'``, bounds, length, number of checkpoints, contents) and
        a ``'data'`` item mapping each checkpoint to its cumulative frequency
        dictionary, padded to a common key set by :func:`zeros`.
    """
    C[:] = sorted(set(C))
    P = postponed_sieve()
    Q = postponed_sieve()
    p = next(P)
    q = next(Q)
    output: Dataset = {
        "header": {
            "interval_type": "prime_start",
            "lower_bound": C[0],
            "upper_bound": C[-1],
            "interval_length": H,
            "no_of_checkpoints": len(C),
            "contents": [],
        }
    }
    data: MetaDict = {C[0]: {m: 0 for m in range(H + 1)}}
    current = {m: 0 for m in range(H + 1)}
    m = 0
    while p <= C[0]:
        p = next(P)
    while q <= p:
        q = next(Q)
    for i in range(len(C)):
        # As in the original: M is assigned but never used here (a vestige of
        # the overlap pattern); at i = 0 this pairs C[-1] with C[0], and since
        # p has already been advanced past C[0], that iteration counts nothing.
        M, N = C[i - 1], C[i]  # noqa: F841
        while p <= N:
            while q <= p + H:
                m += 1
                q = next(Q)
            current[m] += 1
            p = next(P)
            m += -1
        data[N] = {}
        for k in range(H + 1):
            data[N][k] = current[k]
    trimmed_data = zeros(data)
    output["data"] = trimmed_data
    output["header"]["contents"].append("data")
    return output


def intervals(C: list[int], H: int, interval_type: str = "overlap") -> Dataset | None:
    """Dispatch to one of the three checkpointed counters.

    Parameters
    ----------
    C : list of int
        Checkpoints, passed through to the chosen counter.
    H : int
        Interval length.
    interval_type : str, optional
        ``'disjoint'``, ``'prime_start'``, or ``'overlap'`` (the default).
        Any other string returns ``None``.

    Returns
    -------
    dict or None
        The meta-dictionary produced by :func:`disjoint_cp`,
        :func:`prime_start_cp`, or :func:`overlap_cp`.
    """
    if interval_type == "disjoint":
        return disjoint_cp(C, H)
    if interval_type == "prime_start":
        return prime_start_cp(C, H)
    if interval_type == "overlap":
        return overlap_cp(C, H)
    return None


def anyIntervals(
    M: int,
    N: int,
    H: int,
    generator1: Iterator[int],
    generator2: Iterator[int],
) -> dict[int, int]:
    """Count elements of one sequence in intervals started by another.

    For each ``a`` from ``generator1`` with ``M < a <= N``, count the number
    ``m`` of elements ``b`` of ``generator2`` with ``a < b <= a + H``, and
    tally how many ``a`` produce each ``m``.  Both generators must yield
    strictly increasing nonnegative integers.  With two independent prime
    generators this reproduces :func:`prime_start`; it is slower than the
    specialized counters but works for arbitrary sequences.

    A buffer ``Blist`` holds the ``b``-values currently inside the window;
    after ``a`` advances, values that have fallen out on the left are popped
    and ``m`` decremented accordingly.

    Parameters
    ----------
    M, N : int
        The window start ``a`` runs over ``(M, N]``.
    H : int
        Interval length.
    generator1 : iterator of int
        Supplies the window starting points ``a`` (strictly increasing).
    generator2 : iterator of int
        Supplies the elements counted inside the windows (strictly
        increasing).

    Returns
    -------
    dict
        ``{m: count}`` over the nonzero counts.
    """
    A = generator1
    B = generator2
    a = next(A)
    b = next(B)
    while a <= M:
        a = next(A)
    output = {m: 0 for m in range(H + 1)}
    m = 0
    Blist: list[int] = []
    while a <= N:
        while b <= a:
            b = next(B)
        while b <= a + H:
            m += 1
            Blist.append(b)
            b = next(B)
        output[m] += 1
        a = next(A)
        temp_m = m
        for _ in range(temp_m):
            if Blist[0] <= a:
                m += -1
                Blist.pop(0)
    output = {m: output[m] for m in range(H + 1) if output[m] != 0}
    return output


def anyIntervals_cp(
    C: list[int],
    H: int,
    generator1: Iterator[int],
    generator2: Iterator[int],
) -> MetaDict:
    """Checkpoint version of :func:`anyIntervals`.

    Unlike the specialized ``_cp`` counters, this returns the bare
    checkpoint-to-frequencies mapping (padded by :func:`zeros`) without a
    ``'header'``: with arbitrary generators there is no obvious way to label
    the data, so wiring this into the save/retrieve machinery is left as
    future work.

    Note that ``C`` is sorted in place.

    Parameters
    ----------
    C : list of int
        Checkpoints.  Sorted in place.
    H : int
        Interval length.
    generator1, generator2 : iterator of int
        As in :func:`anyIntervals`.

    Returns
    -------
    dict
        ``{C[k]: {m: count}}`` with cumulative counts, padded to a common key
        set.
    """
    C.sort()
    A = generator1
    B = generator2
    a = next(A)
    b = next(B)
    data: MetaDict = {C[0]: {m: 0 for m in range(H + 1)}}
    current = {m: 0 for m in range(H + 1)}
    m = 0
    Blist: list[int] = []
    for i in range(1, len(C)):
        M, N = C[i - 1], C[i]
        while a <= M:
            a = next(A)
        while a <= N:
            while b <= a:
                b = next(B)
            while b <= a + H:
                m += 1
                Blist.append(b)
                b = next(B)
            current[m] += 1
            a = next(A)
            temp_m = m
            for _ in range(temp_m):
                if Blist[0] <= a:
                    m += -1
                    Blist.pop(0)
        data[N] = {}
        for k in range(H + 1):
            data[N][k] = current[k]
    trimmed_data = zeros(data)
    return trimmed_data


def overlap_extension(
    A: int, B: int, H: int, M: list[int]
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Like :func:`overlap`, but also record which windows hold ``m`` primes.

    In addition to the frequency dictionary of :func:`overlap`, return, for
    each ``m`` in the list ``M``, the list of left endpoints ``a`` in
    ``(A, B]`` for which ``(a, a + H]`` contains exactly ``m`` primes.  Useful
    for inspecting exactly which intervals realize a rare count.

    Parameters
    ----------
    A, B : int
        The left endpoint ``a`` runs over ``(A, B]``.
    H : int
        Interval (window) length.
    M : list of int
        The counts ``m`` whose realizing endpoints should be recorded.

    Returns
    -------
    tuple
        ``(show_me, output)`` where ``show_me`` maps each ``m`` in ``M`` to its
        list of endpoints and ``output`` is the frequency dictionary of
        :func:`overlap`.
    """
    P = postponed_sieve()
    Q = postponed_sieve()
    output = {m: 0 for m in range(H + 1)}
    show_me: dict[int, list[int]] = {m: [] for m in M}
    a = A + 1
    p, q = next(P), next(Q)
    while p < a + 1:
        p, q = next(P), next(Q)
    m = 0
    while q < a + H + 1:
        m += 1
        q = next(Q)
    while p < B + 1:
        if m in M:
            show_me[m].append(a)
        output[m] += 1
        b, c = p - a, q - (a + H)
        if m in M:
            show_me[m].extend([x for x in range(a + 1, a + min(b, c))])
        output[m] = output[m] + min(b, c) - 1
        if b == c:
            a = p
            p = next(P)
        if b < c:
            a, m = p, m - 1
            p = next(P)
        if c < b:
            a, m = a + c, m + 1
        while q < a + H + 1:
            q = next(Q)
    while a < B + 1:
        if m in M:
            show_me[m].append(a)
        output[m] += 1
        b, c = p - a, q - (a + H)
        if a + min(b, c) > B:
            if m in M:
                show_me[m].extend([x for x in range(a + 1, B + 1)])
            output[m] = output[m] + B - a
            break
        else:
            if m in M:
                show_me[m].extend([x for x in range(a + 1, a + c)])
            output[m] = output[m] + c - 1
            a, m = a + c, m + 1
            while q < a + H + 1:
                q = next(Q)
    output = {m: output[m] for m in output.keys() if output[m] != 0}
    return show_me, output
