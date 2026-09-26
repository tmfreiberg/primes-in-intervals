"""Tests for the fixed-H, M = 0 cumulative experiment."""

from __future__ import annotations

import math

import numpy as np
import pytest

import primes_in_intervals as pii
from primes_in_intervals import cumulative as cu
from primes_in_intervals import reference as ref


class TestSchedule:
    def test_shape(self):
        C = cu.checkpoint_schedule(1000, dense_until=50, ratio=1.1)
        assert C[:50] == list(range(1, 51))
        assert C[-1] == 1000
        assert all(b > a for a, b in zip(C, C[1:], strict=False))
        assert all(b - a >= 1 for a, b in zip(C[50:], C[51:], strict=False))

    def test_small_and_errors(self):
        assert cu.checkpoint_schedule(1) == [1]
        assert cu.checkpoint_schedule(5, dense_until=100) == [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            cu.checkpoint_schedule(0)
        with pytest.raises(ValueError):
            cu.checkpoint_schedule(10, ratio=1.0)


@pytest.fixture(scope="module")
def dataset():
    return cu.run(20, N_max=3000, dense_until=60, ratio=1.2)


@pytest.fixture(scope="module")
def frames(dataset):
    return cu.build_frames(
        dataset, overlay_from=100, models=("F", "F0", "B_const", "Q_mu", "Q_lambda", "B_avg")
    )


class TestRun:
    def test_counts_match_reference_and_start_at_one(self, dataset):
        assert dataset["header"]["lower_bound"] == 0
        C = sorted(dataset["data"])
        assert C[:3] == [0, 1, 2]
        want = ref.overlap_cp_reference(C, 20)
        assert dataset["data"] == want["data"]
        for N in C[1:]:
            assert sum(dataset["data"][N].values()) == N

    def test_first_window(self, dataset):
        # (1, 21] contains pi(21) = 8 primes
        assert {m: v for m, v in dataset["data"][1].items() if v} == {8: 1}

    def test_explicit_checkpoints_and_save(self, tmp_path):
        db = tmp_path / "db"
        ds = cu.run(10, checkpoints=[50, 10, 10, 200], db_path=db, save_to_db=True, note="test")
        assert sorted(ds["data"]) == [0, 10, 50, 200]
        loaded = cu.load(10, db_path=db)
        assert loaded["data"] == ds["data"]
        prov = pii.provenance_of(10, "overlap", db_path=db)
        assert list(prov["note"]) == ["test"] * 3
        with pytest.raises(LookupError):
            cu.load(11, db_path=db)
        with pytest.raises(ValueError):
            cu.run(10)


class TestFrames:
    def test_frames_cover_every_checkpoint(self, dataset, frames):
        C = sorted(N for N in dataset["data"] if N > 0)
        assert [f.N for f in frames] == C
        for f in frames:
            assert f.P.sum() == pytest.approx(1.0)
            assert sum(f.counts.values()) == f.N

    def test_overlay_start_is_honoured(self, frames):
        for f in frames:
            if f.N < 100:
                assert f.predictions is None and f.scores is None
            else:
                assert f.predictions is not None and f.scores is not None
                assert f.predictions["N"] == f.N and f.predictions["M"] == 0

    def test_scores_are_consistent(self, frames):
        f = frames[-1]
        H = 20
        sc = f.scores
        P = {m: v / f.N for m, v in f.counts.items()}
        lo, hi = sc["central"]["F"]["m_lo"], sc["central"]["F"]["m_hi"]
        assert (lo, hi) == (0, f.m_axis)
        direct = pii.discrepancy(P, f.predictions["F"], lo, hi)
        assert sc["central"]["F"]["E1"] == pytest.approx(direct.E1)
        assert sc["global"]["F"]["m_hi"] == sc["m_T"]
        assert sc["global"]["F"]["predicted_tail_bound"] <= 1e-12
        assert sc["global"]["Q_mu"]["predicted_tail_bound"] == pytest.approx(
            pii.tail_bound_local(H, f.predictions["mu"], sc["m_T"])
        )
        assert sc["global"]["B_const"]["predicted_tail_bound"] == 0.0
        assert sc["mass"]["F"]["expected_deficit"] == pytest.approx(2 / f.N)
        assert sc["mass"]["F0"]["deficit"] == pytest.approx(2 / f.N, abs=1e-9)
        assert sc["mass"]["B_avg"]["expected_deficit"] is None
        assert f.predictions["B_avg_info"]["omitted_bound"] == pytest.approx((math.e - 2) / f.N)
        r = sc["ratio"]["central"]["E1"]
        assert r == pytest.approx(sc["central"]["F"]["E1"] / sc["central"]["F0"]["E1"])

    def test_prediction_accessor(self, frames):
        f = frames[-1]
        assert f.prediction("F").shape == (f.m_axis + 1,)
        assert f.prediction("F", 3).shape == (4,)
        assert frames[0].prediction("F") is None

    def test_needs_lower_bound_zero(self):
        ds = pii.intervals([100, 200, 300], 10, "overlap")
        with pytest.raises(ValueError):
            cu.build_frames(ds)

    def test_cache_round_trip(self, dataset, tmp_path):
        path = tmp_path / "cache.json"
        cache = pii.PredictionCache(path)
        f1 = cu.build_frames(dataset, overlay_from=1000, models=("F", "F0"), cache=cache)
        cache.save()
        n = len(cache)
        assert n == sum(f.N >= 1000 for f in f1)
        cache2 = pii.PredictionCache(path)
        assert len(cache2) == n
        f2 = cu.build_frames(dataset, overlay_from=1000, models=("F", "F0"), cache=cache2)
        assert len(cache2) == n  # every request was a hit
        for a, b in zip(f1, f2, strict=True):
            if a.predictions is not None:
                assert np.allclose(a.predictions["F"], b.predictions["F"])
                assert a.scores["central"]["F"] == b.scores["central"]["F"]

    def test_tables(self, frames, tmp_path):
        long, summary = cu.frames_to_tables(frames, 20)
        assert set(long.columns) >= {
            "H",
            "N",
            "m",
            "count",
            "P",
            "F",
            "F0",
            "correction",
            "B_const",
        }
        assert len(long) == sum(f.m_axis + 1 for f in frames)
        assert summary["N"].tolist() == [f.N for f in frames]
        last = summary.iloc[-1]
        assert last["central_E1_F"] == pytest.approx(frames[-1].scores["central"]["F"]["E1"])
        assert last["ratio_central_E1"] == pytest.approx(
            frames[-1].scores["ratio"]["central"]["E1"]
        )
        assert math.isnan(summary.iloc[0]["mu"])  # before the overlay start
        p1, p2 = cu.write_tables(frames, 20, tmp_path)
        assert p1.exists() and p2.exists()
