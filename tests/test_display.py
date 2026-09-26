"""Tests for the display DataFrame builder."""

from __future__ import annotations

import copy

import pandas as pd

import primes_in_intervals as pii


class TestFlatViews:
    def test_default_is_styled_with_caption(self, analyzed_overlap):
        styled = pii.display(analyzed_overlap)
        assert "Interval type: overlapping." in styled.caption
        assert "Partial counts: cumulative." in styled.caption

    def test_description_off_returns_dataframe(self, analyzed_overlap):
        df = pii.display(analyzed_overlap, description="off")
        assert isinstance(df, pd.DataFrame)
        C = list(analyzed_overlap["data"].keys())
        assert list(df.index) == C
        assert list(df.columns) == list(analyzed_overlap["data"][C[-1]].keys())

    def test_zeroth_item_no_show(self, analyzed_overlap):
        C = list(analyzed_overlap["data"].keys())
        df = pii.display(analyzed_overlap, description="off", zeroth_item="no show")
        assert list(df.index) == C[1:]
        dfc = pii.display(
            analyzed_overlap, description="off", zeroth_item="no show", orient="columns"
        )
        assert C[0] not in dfc.columns

    def test_partition_totals(self, disjoint_dataset):
        ds = copy.deepcopy(disjoint_dataset)
        pii.partition(ds)
        df = pii.display(ds, count="partition", description="off")
        assert "totals" in df.index
        assert "prime_tally" in df.columns
        C = list(ds["data"].keys())
        # The totals row reproduces the final cumulative counts.
        for m in ds["data"][C[-1]]:
            assert df.loc["totals", m] == ds["data"][C[-1]][m]
        # And the prime tally column really tallies primes.
        assert df.loc["totals", "prime_tally"] == sum(
            m * v for m, v in ds["data"][C[-1]].items()
        )

    def test_comparison_views(self, analyzed_overlap):
        pii.compare(analyzed_overlap)
        df_abs = pii.display(analyzed_overlap, comparisons="absolute", description="off")
        df_prob = pii.display(analyzed_overlap, comparisons="probabilities", description="off")
        C = list(analyzed_overlap["data"].keys())
        m = list(analyzed_overlap["data"][C[-1]].keys())[-1]
        assert df_abs.loc[C[-1], m] == analyzed_overlap["comparison"][C[-1]][m][1]
        assert df_prob.loc[C[-1], m] == analyzed_overlap["comparison"][C[-1]][m][0]
        cap = pii.display(analyzed_overlap, comparisons="absolute").caption
        assert "a is actual data" in cap

    def test_option_guards(self, analyzed_overlap, capsys):
        ds = copy.deepcopy(analyzed_overlap)
        assert pii.display(ds, comparisons="absolute") is None
        assert "First compare the data" in capsys.readouterr().out
        assert pii.display(ds, count="partition") is None
        assert "First partition the data." in capsys.readouterr().out
        pii.compare(ds)
        assert pii.display(ds, comparisons="absolute", count="partition") is None
        assert "only compare cumulative" in capsys.readouterr().out
        assert pii.display(ds, winners="show") is None
        assert "Apply the 'winners' function first." in capsys.readouterr().out


class TestNestedViews:
    def test_plain(self, nested_overlap):
        df = pii.display(nested_overlap)
        keys = list(nested_overlap["nested_interval_data"].keys())
        assert list(df["A"]) == [k[0] for k in keys]
        assert list(df["B"]) == [k[1] for k in keys]
        assert list(df["B - A"]) == [k[1] - k[0] for k in keys]

    def test_comparison_single_cell_and_expanded(self, nested_overlap):
        pii.compare(nested_overlap)
        df1 = pii.display(nested_overlap, comparisons="absolute")
        keys = list(nested_overlap["nested_interval_data"].keys())
        m = max(nested_overlap["nested_interval_data"][keys[-1]])
        assert df1.loc[len(keys) - 1, m] == nested_overlap["comparison"][keys[-1]][m][1]
        df4 = pii.display(nested_overlap, comparisons="absolute", single_cell="false")
        for col in [m, f"B{m}", f"F{m}", f"F0{m}"]:
            assert col in df4.columns
        row = df4.loc[len(keys) - 1]
        got = tuple(row[[m, f"B{m}", f"F{m}", f"F0{m}"]])
        assert got == nested_overlap["comparison"][keys[-1]][m][1]

    def test_winners_view(self, nested_overlap):
        pii.compare(nested_overlap)
        pii.winners(nested_overlap)
        df = pii.display(nested_overlap, winners="show")
        for col in ["B sq error", "F sq error", "F0 sq error", "most wins"]:
            assert col in df.columns
        assert len(df) == len(nested_overlap["nested_interval_data"])
