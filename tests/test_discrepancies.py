"""Tests for the discrepancy measures, tail bounds and mass reports."""

from __future__ import annotations

import math

import numpy as np
import pytest

import primes_in_intervals as pii
from primes_in_intervals import discrepancies as di


class TestMeasures:
    def test_hand_example(self):
        P = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}
        q = np.array([0.1, 0.25, 0.45, 0.15, 0.05])
        s = di.discrepancy(P, q, 0, 4)
        assert s.E1 == pytest.approx(0.05 + 0.05 + 0.05 + 0.05)
        assert s.E2 == pytest.approx(math.sqrt(4 * 0.05**2))
        assert s.Einf == pytest.approx(0.05)
        assert s.n_terms == 5 and s.empirical_outside == 0.0
        assert s.predicted_tail_bound is None

    def test_zero_bins_and_gaps_count(self):
        # m = 1 never occurs, m = 3 occurs: the gap at m = 1 contributes |q_1|.
        P = {0: 0.5, 3: 0.5}
        q = {0: 0.5, 1: 0.2, 3: 0.3}
        s = di.discrepancy(P, q, 0, 3)
        assert s.E1 == pytest.approx(0.2 + 0.2)

    def test_range_restriction_reports_outside_mass(self):
        P = {0: 0.2, 1: 0.3, 5: 0.5}
        q = np.zeros(6)
        s = di.discrepancy(P, q, 1, 4)
        assert s.E1 == pytest.approx(0.3)
        assert s.empirical_outside == pytest.approx(0.7)
        with pytest.raises(ValueError):
            di.discrepancy(P, q, 4, 1)

    def test_negative_predictions_enter_as_they_are(self):
        P = {0: 1.0}
        q = np.array([1.0, -0.1])
        assert di.discrepancy(P, q, 0, 1).E1 == pytest.approx(0.1)


class TestTails:
    def test_tail_bound_is_a_bound(self):
        H, M, N = 40, 0, 10**5
        m_T = di.global_range(H, M, N, 1e-10)
        bound = di.tail_bound(H, M, N, m_T)
        assert bound <= 1e-10
        # the bound really dominates the omitted mass of F and F0
        res = pii.integrated(np.arange(m_T + 1, m_T + 120), H, M, N)
        assert np.abs(res.F).sum() <= bound
        assert res.F0.sum() <= di.tail_bound(H, M, N, m_T, corrected=False)

    def test_tail_bound_needs_m_beyond_u_max(self):
        assert math.isinf(di.tail_bound(40, 0, 1000, 10))  # u_max = 40 / log 2 > 11

    def test_global_range_is_minimal(self):
        H, M, N = 30, 0, 10**6
        m_T = di.global_range(H, M, N, 1e-12)
        assert di.tail_bound(H, M, N, m_T) <= 1e-12 < di.tail_bound(H, M, N, m_T - 1)
        assert di.global_range(H, M, N, 1e-12, m_min=500) == 500

    def test_local_tail(self):
        H, u = 40, 3.1
        m = np.arange(0, 400)
        total = np.abs(pii.local_corrected(m, u, H))[31:].sum()
        assert di.tail_bound_local(H, u, 30) == pytest.approx(total, rel=1e-12)
        assert di.tail_bound_local(H, 0.0, 30) == 0.0

    def test_empty_range(self):
        assert di.tail_bound(40, 5, 0, 10) == 0.0
        assert di.global_range(40, 5, 0, 1e-12, m_min=3) == 3


class TestMass:
    def test_report(self):
        r = di.mass_report(np.array([0.5, 0.6, -0.1]), expected_deficit=0.0)
        assert r.total == pytest.approx(1.0) and r.deficit == pytest.approx(0.0)
        assert r.negative_mass == pytest.approx(0.1) and r.n_negative == 1
        assert r.min_value == pytest.approx(-0.1)
        r = di.mass_report(np.array([0.7, 0.2]))
        assert r.negative_mass == 0.0 and r.expected_deficit is None
        assert "deficit" in r.as_dict()

    def test_pad(self):
        assert list(di.pad({2: 1.0, 5: 2.0}, 1, 4)) == [0.0, 1.0, 0.0, 0.0]
        assert list(di.pad(np.array([1.0, 2.0, 3.0]), 1, 4)) == [2.0, 3.0, 0.0, 0.0]
