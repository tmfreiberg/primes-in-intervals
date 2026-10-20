"""Tests for the JSON dataset serialization."""

from __future__ import annotations

import copy

import pytest

import primes_in_intervals as pii
from primes_in_intervals.serialize import dataset_from_json, dataset_to_json


class TestRoundTrip:
    def test_flat_dataset_exact(self, overlap_dataset):
        assert dataset_from_json(dataset_to_json(overlap_dataset)) == overlap_dataset

    def test_nested_keys_restored(self, nested_overlap):
        rt = dataset_from_json(dataset_to_json(nested_overlap))
        assert all(isinstance(k, tuple) for k in rt["nested_interval_data"])
        assert rt["nested_interval_data"] == nested_overlap["nested_interval_data"]

    def test_comparison_tuples_restored(self, nested_overlap):
        ds = copy.deepcopy(nested_overlap)
        pii.compare(ds)
        pii.winners(ds)
        rt = dataset_from_json(dataset_to_json(ds))
        c = next(iter(rt["comparison"]))
        m = next(iter(rt["comparison"][c]))
        cell = rt["comparison"][c][m]
        assert isinstance(cell, tuple)
        assert isinstance(cell[0], tuple) and isinstance(cell[1], tuple)
        # Integer parts of the comparison survive exactly.
        assert cell[1] == ds["comparison"][c][m][1]

    def test_winners_keys_and_lists(self, nested_overlap):
        ds = copy.deepcopy(nested_overlap)
        pii.compare(ds)
        pii.winners(ds)
        rt = dataset_from_json(dataset_to_json(ds))
        c = next(iter(rt["winners"]))
        # The rank keys come back as ints, the label keys as strings, and
        # the win lists as lists (not tuples).
        assert 1 in rt["winners"][c] and "B sq error" in rt["winners"][c]
        assert isinstance(rt["winners"][c]["B wins for m in "], list)
        assert rt["winners"][c] == ds["winners"][c]

    def test_statistics_numerically_equal(self, nested_overlap):
        rt = dataset_from_json(dataset_to_json(nested_overlap))
        for c, stats in nested_overlap["statistics"].items():
            for key, value in stats.items():
                if isinstance(value, list):
                    assert rt["statistics"][c][key] == value
                else:
                    assert rt["statistics"][c][key] == pytest.approx(float(value))

    def test_contents_and_mode_stay_lists(self, nested_overlap):
        rt = dataset_from_json(dataset_to_json(nested_overlap))
        assert isinstance(rt["header"]["contents"], list)
        c = next(iter(rt["statistics"]))
        assert isinstance(rt["statistics"][c]["mode"], list)


class TestEdges:
    def test_unserializable_type_raises(self):
        with pytest.raises(TypeError, match="Cannot serialize"):
            dataset_to_json({"header": {"oops": object()}})

    def test_string_keys_pass_through(self):
        ds = {"header": {"contents": []}, "data": {0: {1: 2}}}
        rt = dataset_from_json(dataset_to_json(ds))
        assert rt == ds
        assert isinstance(next(iter(rt["data"])), int)
