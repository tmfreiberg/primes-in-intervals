"""Tests for extract, partition/unpartition, and nest."""

from __future__ import annotations

import copy

import primes_in_intervals as pii


class TestExtract:
    def test_narrow_rebases_counts(self, overlap_dataset):
        C = list(overlap_dataset["data"].keys())
        lo, hi = C[10], C[30]
        got = pii.extract(overlap_dataset, [lo, hi], option="narrow")
        # Rebased data equals a fresh computation over the narrowed range.
        H = overlap_dataset["header"]["interval_length"]
        fresh = pii.overlap_cp([c for c in C if lo <= c <= hi], H)
        assert got["data"] == fresh["data"]
        assert got["header"]["lower_bound"] == lo
        assert got["header"]["upper_bound"] == hi

    def test_filter_and_narrow_agree_on_checkpoints(self, overlap_dataset):
        C = list(overlap_dataset["data"].keys())
        lo, hi = C[5], C[-5]
        narrow = pii.extract(overlap_dataset, [lo, hi], option="narrow")
        filt = pii.extract(overlap_dataset, [c for c in C if lo <= c <= hi], option="filter")
        assert narrow == filt

    def test_input_not_modified(self, overlap_dataset):
        before = copy.deepcopy(overlap_dataset)
        C = list(overlap_dataset["data"].keys())
        pii.extract(overlap_dataset, [C[1], C[-1]], option="narrow")
        assert overlap_dataset == before

    def test_error_messages(self, overlap_dataset, capsys):
        assert pii.extract({"header": {}}, [0, 1]) is None
        assert "No data to filter." in capsys.readouterr().out
        assert pii.extract(overlap_dataset, [1, 2, 3], option="narrow") is None
        assert "enter list [A,B]" in capsys.readouterr().out
        assert pii.extract(overlap_dataset, [-10, -5], option="narrow") is None
        assert "must lie in the given range" in capsys.readouterr().out
        assert pii.extract(overlap_dataset, [-10, -5], option="filter") is None
        assert "must coincide with the old checkpoints" in capsys.readouterr().out


class TestPartition:
    def test_partition_then_unpartition_is_identity(self, overlap_dataset):
        ds = copy.deepcopy(overlap_dataset)
        pii.partition(ds)
        assert "partition" in ds
        assert ds["header"]["contents"][-1] == "partition"
        data_before = copy.deepcopy(ds["data"])
        del ds["data"]
        pii.unpartition(ds)
        assert ds["data"] == data_before

    def test_partition_values_are_differences(self, disjoint_dataset):
        ds = copy.deepcopy(disjoint_dataset)
        pii.partition(ds)
        C = list(ds["data"].keys())
        for k in range(1, len(C)):
            for m in ds["data"][C[k]]:
                assert ds["partition"][C[k]][m] == ds["data"][C[k]][m] - ds["data"][C[k - 1]][m]

    def test_guards(self, overlap_dataset, capsys):
        ds = copy.deepcopy(overlap_dataset)
        pii.partition(ds)
        assert pii.partition(ds) is None
        assert "already exists" in capsys.readouterr().out
        assert pii.unpartition(ds) is None
        assert "Unpartitioned data already exists." in capsys.readouterr().out
        assert pii.partition({"header": {}}) is None
        assert "No data to partition." in capsys.readouterr().out
        assert pii.unpartition({"header": {}}) is None
        assert "No data to unpartition." in capsys.readouterr().out


class TestNest:
    def test_structure_odd_checkpoint_count(self, overlap_dataset):
        # 41 checkpoints: the middle one is dropped, leaving 20 nested pairs.
        nested = pii.nest(copy.deepcopy(overlap_dataset))
        keys = list(nested["nested_interval_data"].keys())
        assert nested["header"]["nested_intervals"] == 20
        assert len(keys) == 20
        C = list(overlap_dataset["data"].keys())
        mid = (C[0] + C[-1]) // 2
        for lo, hi in keys:
            assert lo + hi == 2 * mid  # common center
        # Innermost first, outermost last, each containing the previous.
        assert keys[0][1] - keys[0][0] < keys[-1][1] - keys[-1][0]
        assert keys[-1] == (C[0], C[-1])

    def test_values_are_differences(self, overlap_dataset):
        nested = pii.nest(copy.deepcopy(overlap_dataset))
        data = overlap_dataset["data"]
        for (lo, hi), freq in nested["nested_interval_data"].items():
            for m, v in freq.items():
                assert v == data[hi][m] - data[lo][m]

    def test_even_checkpoint_count(self, overlap_dataset):
        C = list(overlap_dataset["data"].keys())
        ds = pii.extract(overlap_dataset, C[:-1], option="filter")  # 40 checkpoints
        nested = pii.nest(ds)
        assert nested["header"]["nested_intervals"] == 20

    def test_rebuilds_from_partition(self, overlap_dataset):
        ds = copy.deepcopy(overlap_dataset)
        pii.partition(ds)
        del ds["data"]
        nested = pii.nest(ds)
        assert nested == pii.nest(copy.deepcopy(overlap_dataset))

    def test_guards(self, capsys):
        assert pii.nest({"header": {}}) is None
        assert "No data to work with" in capsys.readouterr().out
        tiny = pii.overlap_cp([0, 100], 10)
        assert pii.nest(tiny) is None
        assert "At least three checkpoints" in capsys.readouterr().out
