"""Reshaping interval-count datasets.

The checkpointed counters produce cumulative counts: the frequencies at
checkpoint ``C[k]`` cover every interval from the very first checkpoint on.
The functions here reorganize that data without touching the primes again:

* :func:`extract` builds a new dataset over a sub-range or sub-list of
  checkpoints (re-basing the cumulative counts at the new lower bound);
* :func:`partition` and :func:`unpartition` convert between cumulative counts
  and per-checkpoint-gap counts (successive differences and cumulative sums);
* :func:`nest` converts a run of checkpoints into a family of nested intervals
  ``(C[k-1], C[k]], (C[k-2], C[k+1]], ...`` sharing a common center, the
  configuration used throughout the project's animations.

:func:`extract` and :func:`nest` return new datasets; :func:`partition` and
:func:`unpartition` modify their argument in place (adding a ``'partition'``
or ``'data'`` item and recording the addition in the header's ``'contents'``
list) and also return it.
"""

from __future__ import annotations

from primes_in_intervals.intervals import Dataset, zeros

__all__ = ["extract", "nest", "partition", "unpartition"]


def extract(meta_dictionary: Dataset, newC: list[int], option: str = "filter") -> Dataset | None:
    """Build a new dataset restricted to a sub-range or sub-list of checkpoints.

    With ``option='narrow'``, ``newC`` must be a two-element list ``[A, B]``
    and the new checkpoints are the old ones lying in ``(A, B]`` (in fact in
    ``[A, B]``, closed at both ends: the new lower bound is the smallest old
    checkpoint at or above ``A``).  With the default ``option='filter'``, the
    new checkpoints are the intersection of ``newC`` with the old checkpoints.
    If ``A`` and ``B`` are themselves checkpoints, the two options agree.

    In either case the counts are re-based at the new lower bound: the new
    count at checkpoint ``c`` is the old count at ``c`` minus the old count at
    the new lower bound, so the result describes intervals with left endpoints
    in the new range only.

    Parameters
    ----------
    meta_dictionary : dict
        A dataset with a ``'data'`` item.
    newC : list of int
        ``[A, B]`` for ``'narrow'``; a list of desired checkpoints for
        ``'filter'``.
    option : str, optional
        ``'narrow'`` or anything else (defaults to ``'filter'``).

    Returns
    -------
    dict or None
        A new dataset (the input is not modified), or ``None`` with a message
        if the request cannot be satisfied.
    """
    # newC is a list.
    # option is either 'narrow' or not (defaults to 'filter').
    # if option=='narrow', newC should be of the form [A,B] where (A, B] is the
    # desired range for checkpoints.
    # if A and B are already checkpoints, then both 'narrow' and 'filter' will
    # do the same thing.
    if "data" not in meta_dictionary.keys():
        return print("No data to filter.")
    if option == "narrow":
        if len(newC) != 2:
            return print("To narrow checkpoints to range (A, B], enter list [A,B].")
        oldC = list(meta_dictionary["data"].keys())
        oldC.sort()  # just in case: it's important that these are in increasing order
        C = [c for c in oldC if newC[0] <= c <= newC[-1]]
        if len(C) < 2:
            return print("At least two of the new checkpoints must lie in the given range.")
    else:
        old_set = set(meta_dictionary["data"].keys())
        C = list(old_set.intersection(set(newC)))
        C.sort()
        if len(C) < 2:
            return print(
                "At least two of the new checkpoints must coincide with the old checkpoints."
            )
    interval_type = meta_dictionary["header"]["interval_type"]
    A, B = C[0], C[-1]
    H = meta_dictionary["header"]["interval_length"]
    output: Dataset = {
        "header": {
            "interval_type": interval_type,
            "lower_bound": A,
            "upper_bound": B,
            "interval_length": H,
            "no_of_checkpoints": len(C),
            "contents": [],
        }
    }
    output["data"] = {}
    for c in C:
        output["data"][c] = {}
        for m in meta_dictionary["data"][c].keys():
            output["data"][c][m] = meta_dictionary["data"][c][m] - meta_dictionary["data"][A][m]
    trimmed_data = zeros(output["data"])
    output["data"] = trimmed_data
    output["header"]["contents"].append("data")
    return output


def partition(meta_dictionary: Dataset) -> Dataset | None:
    """Add per-gap counts to a dataset (successive differences), in place.

    For checkpoints ``A = C_0 < C_1 < ... < C_k = B``, the new ``'partition'``
    item maps each ``C_i`` (``i >= 1``) to the counts for intervals with left
    endpoints in ``(C_{i-1}, C_i]`` alone, obtained by differencing the
    cumulative ``'data'`` counts.  ``'partition'`` is appended to the header's
    ``'contents'``.

    Parameters
    ----------
    meta_dictionary : dict
        A dataset with a ``'data'`` item and no ``'partition'`` item yet.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if there is
        nothing to do.
    """
    if "data" not in meta_dictionary.keys():
        return print("No data to partition.")
    if "partition" in meta_dictionary.keys():
        return print("Partitioned data already exists.")
    C = list(meta_dictionary["data"].keys())
    C.sort()  # just in case: it's important that these are in increasing order
    partitioned_data = {C[0]: meta_dictionary["data"][C[0]]}
    for k in range(1, len(C)):
        partitioned_data[C[k]] = {}
        for m in meta_dictionary["data"][C[k]].keys():
            partitioned_data[C[k]][m] = (
                meta_dictionary["data"][C[k]][m] - meta_dictionary["data"][C[k - 1]][m]
            )
    meta_dictionary["partition"] = {}
    for c in C:
        meta_dictionary["partition"][c] = {}
        for m in partitioned_data[c].keys():
            meta_dictionary["partition"][c][m] = partitioned_data[c][m]
    meta_dictionary["header"]["contents"].append("partition")
    return meta_dictionary


def unpartition(meta_dictionary: Dataset) -> Dataset | None:
    """Rebuild cumulative counts from per-gap counts, in place.

    The inverse of :func:`partition`: the new ``'data'`` item is the running
    cumulative sum of the ``'partition'`` item.  ``'data'`` is appended to the
    header's ``'contents'``.

    Parameters
    ----------
    meta_dictionary : dict
        A dataset with a ``'partition'`` item and no ``'data'`` item.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if there is
        nothing to do.
    """
    if "partition" not in meta_dictionary.keys():
        return print("No data to unpartition.")
    if "data" in meta_dictionary.keys():
        return print("Unpartitioned data already exists.")
    C = list(meta_dictionary["partition"].keys())
    C.sort()  # just in case: it's important that these are in increasing order
    unpartitioned_data = {C[0]: meta_dictionary["partition"][C[0]]}
    for k in range(1, len(C)):
        unpartitioned_data[C[k]] = {}
        for m in meta_dictionary["partition"][C[k]].keys():
            unpartitioned_data[C[k]][m] = (
                meta_dictionary["partition"][C[k]][m] + unpartitioned_data[C[k - 1]][m]
            )
    meta_dictionary["data"] = {}
    for c in C:
        meta_dictionary["data"][c] = {}
        for m in unpartitioned_data[c].keys():
            meta_dictionary["data"][c][m] = unpartitioned_data[c][m]
    meta_dictionary["header"]["contents"].append("data")
    return meta_dictionary


def nest(dataset: Dataset) -> Dataset | None:
    """Convert checkpointed counts into counts over nested, centered intervals.

    Given checkpoints ``C_0 < C_1 < ... < C_K``, the new dataset's
    ``'nested_interval_data'`` item maps each pair ``(C_{k-i-1}, C_{k+i})``,
    ``i = 0, ..., k - 1`` with ``k = K // 2``, to the counts for intervals with
    left endpoints in ``(C_{k-i-1}, C_{k+i}]``: a family of intervals each
    contained in the next.  When the checkpoints form an arithmetic
    progression, all the nested intervals share a common midpoint, which is
    the configuration used for the project's centered animations.

    If the number of checkpoints is odd, the middle checkpoint is dropped
    first, so the innermost interval always straddles the center.  At least
    three checkpoints are required.

    If the dataset has been partitioned and its ``'data'`` item removed, it is
    first rebuilt with :func:`unpartition`.

    Parameters
    ----------
    dataset : dict
        A dataset with a ``'data'`` (or at least ``'partition'``) item, whose
        checkpoint keys are integers.

    Returns
    -------
    dict or None
        A new dataset whose data item is ``'nested_interval_data'``, keyed by
        ``(lower, upper)`` tuples; ``None`` with a message if the input is
        unsuitable.
    """
    if "data" not in dataset.keys():
        if "partition" not in dataset.keys():
            return print(
                "No data to work with, or data is not in a suitable configuration for nesting."
            )
        else:
            unpartition(dataset)
    C = list(dataset["data"].keys())
    C.sort()
    if len(C) < 3:
        return print("At least three checkpoints needed for a nontrivial nesting.")
    interval_type = dataset["header"]["interval_type"]
    A = dataset["header"]["lower_bound"]
    B = dataset["header"]["upper_bound"]
    H = dataset["header"]["interval_length"]
    no_of_checkpoints = dataset["header"]["no_of_checkpoints"]
    nested: Dataset = {
        "header": {
            "nested_intervals": 0,
            "interval_type": interval_type,
            "lower_bound": A,
            "upper_bound": B,
            "interval_length": H,
            "no_of_checkpoints": no_of_checkpoints,
            "contents": [],
        }
    }
    nested["nested_interval_data"] = {}
    if len(C) % 2 == 1:
        C.pop(len(C) // 2)
    k = len(C) // 2
    M = list(dataset["data"][C[-1]].keys())
    for i in range(k):
        nested["nested_interval_data"][C[k - i - 1], C[k + i]] = {}
        for m in M:
            nested["nested_interval_data"][C[k - i - 1], C[k + i]][m] = (
                dataset["data"][C[k + i]][m] - dataset["data"][C[k - i - 1]][m]
            )
    nested["header"]["nested_intervals"] = k
    nested["header"]["contents"].append("nested_interval_data")
    return nested
