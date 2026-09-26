"""Tabular presentation of datasets as pandas DataFrames.

:func:`display` is the project's one-stop table builder.  Depending on its
options it shows raw or partitioned counts, prediction comparisons (as
probability or absolute-count tuples), or the per-interval prediction
scoreboard produced by :func:`~primes_in_intervals.comparisons.winners`; it
handles both ordinary checkpointed datasets and nested-interval datasets, adds
prime tallies and totals where meaningful, and can attach an explanatory
caption.

The option values are strings, kept exactly as in the original project (for
example ``count='partition'``, ``comparisons='absolute'``,
``winners='show'``), so every call in the exposition and examples continues
to work unchanged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from primes_in_intervals.intervals import Dataset

__all__ = ["display"]


def display(
    dataset: Dataset,
    orient: str = "index",
    description: str = "on",
    zeroth_item: str = "show",
    count: str = "cumulative",
    comparisons: str = "off",
    single_cell: str = "true",
    winners: str = "no show",
) -> Any:
    """Present a dataset as a DataFrame, under a variety of views.

    Parameters
    ----------
    dataset : dict
        The dataset to display.  Which items it must contain depends on the
        options: ``'partition'`` for ``count='partition'``, ``'comparison'``
        for the comparison views, ``'winners'`` for ``winners='show'``.
    orient : str, optional
        Passed to ``pandas.DataFrame.from_dict``: ``'index'`` (default) for
        one row per checkpoint, ``'columns'`` for one column per checkpoint.
    description : str, optional
        ``'off'`` for a bare DataFrame; anything else (default ``'on'``)
        attaches a caption describing the dataset (ordinary datasets only).
    zeroth_item : str, optional
        ``'no show'`` to drop the first (all-zero) checkpoint row or column;
        anything else (default ``'show'``) keeps it.
    count : str, optional
        ``'partition'`` to display the per-gap counts (with a totals row);
        anything else (default ``'cumulative'``) displays the cumulative
        counts.
    comparisons : str, optional
        ``'absolute'`` for actual-vs-predicted interval counts,
        ``'probabilities'`` for the probability tuples, anything else
        (default ``'off'``) for the plain counts.
    single_cell : str, optional
        Nested datasets with comparisons only: ``'false'`` to expand each
        tuple into four columns (``m``, ``Bm``, ``Fm``, ``F0m``); anything
        else (default ``'true'``) keeps one tuple per cell.
    winners : str, optional
        ``'show'`` for the prediction scoreboard; anything else (default
        ``'no show'``) for the data views above.

    Returns
    -------
    pandas.DataFrame or pandas Styler or None
        The requested table (styled when a caption is attached), or ``None``
        with a message if the dataset lacks the item an option requires.
    """
    # DataFrame orient argument either 'index' or 'columns'.
    # description either 'off' or not (defaults to 'on').
    # zeroth_item either 'no show' or not (defaults to 'show').
    # count either 'partition' or not (defaults to 'cumulative').
    # comparisons either 'absolute', 'probabilities', or not (defaults to 'off').
    # single_cell either 'false' or not (defaults to 'true').
    # winners either 'show' or not (defaults to 'no show').
    if winners == "show":
        if "winners" not in dataset.keys():
            return print("Apply the 'winners' function first.")
        if "data" in dataset.keys():
            C = list(dataset["data"].keys())
            H = dataset["header"]["interval_length"]
            interval_type = dataset["header"]["interval_type"]
            output: dict = {}
            for i in range(1, len(C)):
                if interval_type == "overlap":
                    output[i] = {"B - A": C[i] - C[0], "A": C[0], "B": C[i], "H": H}
                if interval_type == "disjoint":
                    output[i] = {
                        "(B - A)/H": (C[i] - C[0]) // H,
                        "A": C[0],
                        "B": C[i],
                        "H": H,
                    }
                if interval_type == "prime_start":
                    output[i] = {
                        "pi(B) - pi(A)": sum(dataset["data"][C[i]].values()),
                        "A": C[0],
                        "B": C[i],
                        "H": H,
                    }
                for w in dataset["winners"][C[i]]:
                    output[i][w] = dataset["winners"][C[i]][w]

            df = pd.DataFrame.from_dict(output, orient=orient)
            return df
        if "nested_interval_data" in dataset.keys():
            C = list(dataset["nested_interval_data"].keys())
            H = dataset["header"]["interval_length"]
            interval_type = dataset["header"]["interval_type"]
            output = {}
            for i in range(len(C)):
                if interval_type == "overlap":
                    output[i] = {
                        "B - A": C[i][1] - C[i][0],
                        "A": C[i][0],
                        "B": C[i][1],
                        "H": H,
                    }
                if interval_type == "disjoint":
                    output[i] = {
                        "(B - A)/H": (C[i][1] - C[i][0]) // H,
                        "A": C[i][0],
                        "B": C[i][1],
                        "H": H,
                    }
                if interval_type == "prime_start":
                    output[i] = {
                        "pi(B) - pi(A)": sum(dataset["nested_interval_data"][C[i]].values()),
                        "A": C[i][0],
                        "B": C[i][1],
                        "H": H,
                    }
                for w in dataset["winners"][C[i]]:
                    output[i][w] = dataset["winners"][C[i]][w]
            df = pd.DataFrame.from_dict(output, orient=orient)
            return df
        return None
    else:
        if "data" in dataset.keys():
            if comparisons == "absolute" or comparisons == "probabilities":
                if comparisons == "absolute":
                    index = 1
                if comparisons == "probabilities":
                    index = 0
                if "comparison" not in dataset.keys():
                    return print("First compare the data to something with the compare function.")
                if count == "partition":
                    return print("We only compare cumulative (non-partitioned) data.")
                C = list(dataset["comparison"].keys())
                C.sort()
                output = {C[0]: {m: 0 for m in dataset["comparison"][C[0]].keys()}}
                for c in C[1:]:
                    output[c] = {}
                    for m in dataset["comparison"][c].keys():
                        output[c][m] = dataset["comparison"][c][m][index]

                df = pd.DataFrame.from_dict(output, orient=orient)

            else:
                if count == "partition":
                    if "partition" not in dataset.keys():
                        return print("First partition the data.")
                    datakey = "partition"
                else:
                    if "data" not in dataset.keys():
                        return print("First unpartition the data.")
                    datakey = "data"
                C = list(dataset[datakey].keys())
                C.sort()
                output = {}
                # In the case of disjoint intervals, we can display 'prime
                # tallies' for each checkpoint.  (Gives the total number of
                # primes from C[0] to C[k] in the cumulative count case, or
                # from C[k-1] to C[k] in the partial count case.)  In the case
                # of displaying the partitioned data (count 'partition' i.e.
                # non-cumulative), we can show totals at the end of each
                # row/column (depending on the orientation), giving the total
                # number of intervals between A and B that contain m primes.
                # (In the cumulative count case, the totals are just the last
                # row/column anyway.)
                for c in C:
                    output[c] = {}
                    for m in dataset[datakey][c].keys():
                        output[c][m] = dataset[datakey][c][m]
                if dataset["header"]["interval_type"] == "disjoint":
                    for c in C:
                        output[c]["prime_tally"] = {}
                        tally = sum(
                            [m * dataset[datakey][c][m] for m in dataset[datakey][c].keys()]
                        )
                        output[c]["prime_tally"] = tally
                if count == "partition":
                    output["totals"] = {}
                    for m in dataset[datakey][C[-1]].keys():
                        output["totals"][m] = sum([dataset[datakey][c][m] for c in C])
                    if dataset["header"]["interval_type"] == "disjoint":
                        # should be the same as summing m*totals[m] over m:
                        output["totals"]["prime_tally"] = sum(
                            [output[c]["prime_tally"] for c in C]
                        )

                df = pd.DataFrame.from_dict(output, orient=orient)

            if description == "off":
                if zeroth_item == "no show":
                    if orient == "columns":
                        A = dataset["header"]["lower_bound"]
                        return df.loc[:, df.columns != A]
                    else:
                        return df.tail(-1)
                else:
                    return df
            else:
                interval_type = dataset["header"]["interval_type"]
                if interval_type == "overlap":
                    word = "overlapping"
                if interval_type == "disjoint":
                    word = "disjoint"
                if interval_type == "prime_start":
                    word = "left endpoint prime"
                A = dataset["header"]["lower_bound"]
                B = dataset["header"]["upper_bound"]
                H = dataset["header"]["interval_length"]
                if count == "partition":
                    counts = "non-cumulative"
                else:
                    counts = "cumulative"
                text = (
                    f"Interval type: {word}. Lower bound: {A}. Upper bound: {B}. "
                    f"Interval length: {H}. Partial counts: {counts}."
                )
                if comparisons == "absolute" or comparisons == "probabilities":
                    text = text + (
                        "In tuple (a,b,c,d), a is actual data, b is the binomial "
                        "prediction Binom(H, mu/H), c is the corrected prediction F, "
                        "and d is the uncorrected prediction F0 (M = A, N = B - A)."
                    )
                if zeroth_item == "no show":
                    if orient == "columns":
                        return df.loc[:, df.columns != A].style.set_caption(text)
                    else:
                        return df.tail(-1).style.set_caption(text)
                else:
                    return df.style.set_caption(text)
        if "nested_interval_data" in dataset.keys():
            if comparisons == "absolute" or comparisons == "probabilities":
                if comparisons == "absolute":
                    index = 1
                if comparisons == "probabilities":
                    index = 0
                if "comparison" not in dataset.keys():
                    return print("First compare the data to something with the compare function.")
            C = list(dataset["nested_interval_data"].keys())
            H = dataset["header"]["interval_length"]
            interval_type = dataset["header"]["interval_type"]
            M = list(dataset["nested_interval_data"][C[-1]].keys())
            output = {}
            for i in range(len(C)):
                if interval_type == "overlap":
                    output[i] = {
                        "B - A": C[i][1] - C[i][0],
                        "A": C[i][0],
                        "B": C[i][1],
                        "H": H,
                    }
                if interval_type == "disjoint":
                    output[i] = {
                        "(B - A)/H": (C[i][1] - C[i][0]) // H,
                        "A": C[i][0],
                        "B": C[i][1],
                        "H": H,
                    }
                if interval_type == "prime_start":
                    output[i] = {
                        "pi(B) - pi(A)": sum(dataset["nested_interval_data"][C[i]].values()),
                        "A": C[i][0],
                        "B": C[i][1],
                        "H": H,
                    }
                if not (comparisons == "absolute" or comparisons == "probabilities"):
                    for m in M:
                        output[i][m] = dataset["nested_interval_data"][C[i]][m]
                    if interval_type == "disjoint":
                        tally = sum([m * output[i][m] for m in M])
                        output[i]["prime tally"] = tally
                else:
                    if single_cell == "true":
                        for m in M:
                            output[i][m] = dataset["comparison"][C[i]][m][index]
                    else:
                        Mexpand: list = []
                        for m in M:
                            Mexpand.extend([m, f"B{m}", f"F{m}", f"F0{m}"])
                        j = 0
                        while j < len(Mexpand):
                            m = Mexpand[j]
                            B = Mexpand[j + 1]
                            F = Mexpand[j + 2]
                            Falt = Mexpand[j + 3]
                            output[i][m] = dataset["comparison"][C[i]][m][index][0]
                            output[i][B] = dataset["comparison"][C[i]][m][index][1]
                            output[i][F] = dataset["comparison"][C[i]][m][index][2]
                            output[i][Falt] = dataset["comparison"][C[i]][m][index][3]
                            j += 4

            df = pd.DataFrame.from_dict(output, orient=orient)
            return df
        return None
