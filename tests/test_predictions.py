"""Tests for the prediction functions, the quadrature, and the constant eta."""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma

import primes_in_intervals as pii
from primes_in_intervals.quadrature import QuadratureSettings


class TestConstants:
    def test_ms_value(self):
        # 1 - EulerGamma - log(2*pi) = 1 - 0.5772156... - 1.8378770... = -1.4150927...
        assert float(pii.MS) == pytest.approx(-1.4150927, abs=1e-6)

    def test_eta(self):
        H = 76
        assert pii.eta(H) == pytest.approx(
            (math.log(H) + math.log(2 * math.pi) + 0.5772156649 - 1) / H
        )
        assert pii.eta(H) == pytest.approx((math.log(H) - float(pii.MS)) / H)


class TestPoisson:
    def test_values_and_mass(self):
        u = 3.7
        m = np.arange(0, 60)
        p = pii.poisson_pmf(m, u)
        assert p.sum() == pytest.approx(1.0)
        assert pii.poisson_pmf(4, u) == pytest.approx(math.exp(-u) * u**4 / 24)
        assert pii.poisson_pmf(0, 0.0) == 1.0 and pii.poisson_pmf(3, 0.0) == 0.0

    def test_real_m_via_gamma(self):
        u = 2.0
        assert pii.poisson_pmf(2.5, u) == pytest.approx(math.exp(-u) * u**2.5 / gamma(3.5))


class TestLocal:
    def test_formula_and_old_name(self):
        H, m, t = 76, 5, 4.75
        Q2 = ((m - t) ** 2 - m) / 2
        expected = math.exp(-t) * (t**m / gamma(m + 1)) * (1 - pii.eta(H) * Q2)
        assert pii.local_corrected(m, t, H) == pytest.approx(expected)
        assert float(pii.frei(H, m, t)) == pytest.approx(expected)

    def test_sums_to_one_for_every_u(self):
        H = 40
        m = np.arange(0, 200)
        for u in (0.5, 3.0, 12.0, 57.0):
            assert pii.local_corrected(m, u, H).sum() == pytest.approx(1.0, abs=1e-12)

    def test_can_be_negative_and_is_not_clipped(self):
        H = 40
        values = pii.local_corrected(np.arange(0, 30), 3.0, H)
        assert values.min() < 0

    def test_reduces_to_poisson_at_large_H(self):
        m, t = 4, 4.0
        assert pii.local_corrected(m, t, 10**9) == pytest.approx(pii.poisson_pmf(m, t), rel=1e-6)

    def test_frei_alt_retained(self):
        H, m, t = 50, 6, 5.0
        base = math.exp(-t) * t**m / gamma(m + 1)
        diff = float(pii.frei_alt(H, m, t)) - float(pii.frei(H, m, t))
        assert diff == pytest.approx(base * (t / H) * (m - t))


class TestParameters:
    def test_shift_constant(self):
        assert pii.shift_constant(0.0) == pytest.approx(-1.0)
        a = 0.3
        assert pii.shift_constant(a) == pytest.approx(
            (a + 1) * math.log(a + 1) - a * math.log(a) - 1
        )
        with pytest.raises(ValueError):
            pii.shift_constant(-0.1)

    def test_shifted_parameter(self):
        assert pii.shifted_parameter(40, 0, 10**6) == pytest.approx(40 / (math.log(10**6) - 1))
        assert math.isnan(pii.shifted_parameter(40, 0, 2))  # log 2 - 1 < 0
        M, N = 1000.0, 500.0
        avg_log = quad(np.log, M, M + N)[0] / N
        assert pii.shifted_parameter(40, M, N) == pytest.approx(40 / avg_log)

    def test_averaged_parameter_matches_quadrature(self):
        for H, M, N in [(40, 0, 10**5), (76, 485065195, 200000), (10, 2, 5), (30, 1, 1)]:
            lower, upper = max(2, M), M + N
            want = H * quad(lambda t: 1 / math.log(t), lower, upper)[0] / N
            assert pii.averaged_parameter(H, M, N) == pytest.approx(want, rel=1e-10)
        assert pii.averaged_parameter(40, 0, 2) == 0.0
        assert pii.averaged_parameter(40, 0, 1) == 0.0

    def test_narrow_range_far_out_has_no_shift(self):
        # (A, A + N] around e^20: mu is H / 20 to three digits, not H / 19.
        H, M, N = 76, 485065195, 200000
        assert pii.averaged_parameter(H, M, N) == pytest.approx(76 / 20, rel=1e-3)


def _mp_F(m, H, M, N, corrected, dps=30):
    eta_H = pii.eta(H)

    def f(t):
        u = mp.mpf(H) / mp.log(t)
        p = mp.exp(-u) * u**m / mp.factorial(m)
        return p * (1 - mp.mpf(eta_H) / 2 * ((m - u) ** 2 - m)) if corrected else p

    return float(pii.mp_log_average(f, max(2, M), M + N, N, dps=dps))


class TestIntegrated:
    def test_mass_identity(self):
        for H, M, N in [(40, 0, 10**6), (100, 0, 10**9), (76, 485065195, 200000), (20, 0, 3)]:
            m_T = pii.global_range(H, M, N, 1e-13)
            res = pii.integrated(np.arange(0, m_T + 1), H, M, N)
            assert res.expected_mass == pytest.approx((M + N - max(2, M)) / N)
            assert res.F.sum() == pytest.approx(res.expected_mass, abs=1e-11)
            assert res.F0.sum() == pytest.approx(res.expected_mass, abs=1e-11)
            assert abs(res.correction.sum()) < 1e-11
            assert res.converged

    def test_against_mpmath(self):
        H, M, N = 40, 0, 10**6
        ms = [0, 1, 3, 5, 8, 12, 20, 40]
        res = pii.integrated(np.array(ms, dtype=float), H, M, N)
        for i, m in enumerate(ms):
            F_mp = _mp_F(m, H, M, N, True)
            F0_mp = _mp_F(m, H, M, N, False)
            assert res.F[i] == pytest.approx(F_mp, abs=1e-15)
            assert res.F0[i] == pytest.approx(F0_mp, abs=1e-15)
            assert res.correction[i] == pytest.approx(F_mp - F0_mp, abs=1e-15)

    def test_against_mpmath_far_range(self):
        H, M, N = 76, 485065195, 200000
        res = pii.integrated(np.array([2.0, 4.0, 6.0]), H, M, N)
        for i, m in enumerate([2, 4, 6]):
            assert res.F[i] == pytest.approx(_mp_F(m, H, M, N, True), abs=1e-14)

    def test_tightened_tolerances_agree(self):
        H, M, N = 40, 0, 10**7
        m = np.arange(0, 40)
        loose = pii.integrated(m, H, M, N)
        tight = pii.integrated(
            m, H, M, N, QuadratureSettings(epsabs=1e-16, epsrel=1e-14, limit=2000)
        )
        assert np.max(np.abs(loose.F - tight.F)) < 1e-13
        assert loose.error < 1e-12 and tight.error < 1e-12

    def test_huge_N_is_stable(self):
        res = pii.integrated(np.arange(0, 80), 100, 0, 10**15)
        assert res.converged
        assert res.F0.sum() == pytest.approx(1 - 2 / 10**15, abs=1e-9)

    def test_zero_at_N_two(self):
        res = pii.integrated(np.arange(0, 5), 40, 0, 2)
        assert np.all(res.F == 0) and np.all(res.F0 == 0) and res.error == 0

    def test_negative_values_reported(self):
        res = pii.integrated(np.arange(0, 40), 40, 0, 10**5)
        assert res.F.min() < 0
        assert res.negative_mass > 0
        assert np.all(res.F0 >= 0)

    def test_scalar_wrappers(self):
        H, M, N = 40, 0, 10**5
        res = pii.integrated(np.array([3.0]), H, M, N)
        assert pii.integrated_corrected(3, H, M, N) == pytest.approx(res.F[0])
        assert pii.integrated_poisson(3, H, M, N) == pytest.approx(res.F0[0])
        assert pii.integrated_correction(3, H, M, N) == pytest.approx(res.correction[0])


class TestBinomials:
    def test_binom_pmf_sums_to_one_and_matches_comb(self):
        from math import comb

        H, p = 12, 0.3
        assert sum(pii.binom_pmf(H, m, p) for m in range(H + 1)) == pytest.approx(1.0)
        for m in range(H + 1):
            assert pii.binom_pmf(H, m, p) == pytest.approx(comb(H, m) * p**m * (1 - p) ** (H - m))
        assert 0 < pii.binom_pmf(30, 2.5, 0.07) < 1  # gamma interpolation for guides

    def test_binomial_constant(self):
        H, mu = 30, 2.1
        values = pii.binomial_constant(np.arange(0, H + 1), H, mu)
        assert values.sum() == pytest.approx(1.0)
        assert values[3] == pytest.approx(pii.binom_pmf(H, 3, mu / H))
        assert math.isnan(pii.binomial_constant(3, H, 2 * H))  # p > 1 is not clipped
        with pytest.raises(ValueError):
            pii.binomial_constant(2.5, H, mu)

    def test_binomial_averaged_omits_below_e(self):
        H, M, N = 40, 0, 10**5
        values, info = pii.binomial_averaged(np.arange(0, H + 1), H, M, N)
        assert info["lower_used"] == pytest.approx(math.e)
        assert info["omitted_length"] == pytest.approx(math.e - 2)
        assert info["omitted_bound"] == pytest.approx((math.e - 2) / N)
        # the mass over 0..H is exactly (N - e)/N: each pmf sums to one
        assert values.sum() == pytest.approx((N - math.e) / N, abs=1e-12)
        # a range starting beyond e omits nothing
        _v, info2 = pii.binomial_averaged(np.arange(0, 5), 40, 100, 50)
        assert info2["omitted_length"] == 0.0
        with pytest.raises(ValueError):
            pii.binomial_averaged(np.array([1.5]), H, M, N)


class TestValidityAndPredictAll:
    def test_validity_flags(self):
        v = pii.parameter_validity(40, 0, 2)
        assert not v["integrated"]["valid"] and not v["lambda"]["valid"]
        v = pii.parameter_validity(40, 0, 3)
        assert all(entry["valid"] for entry in v.values())
        v = pii.parameter_validity(40, 2, 1)  # mu/H = avg of 1/log t on [2, 3] > 1
        assert not v["binomial_constant"]["valid"]
        assert v["binomial_averaged"]["valid"]  # [e, 3] is usable
        assert v["integrated"]["valid"]
        assert not pii.parameter_validity(40, 2, 0.5)["binomial_averaged"]["valid"]  # [2, 2.5]

    def test_predict_all_consistency(self):
        H, M, N = 40, 0, 10**5
        pred = pii.predict_all(
            H, M, N, 20, models=("F", "F0", "B_const", "Q_mu", "Q_lambda", "B_avg")
        )
        res = pii.integrated(np.arange(0, 21), H, M, N)
        assert np.allclose(pred["F"], res.F) and np.allclose(pred["F0"], res.F0)
        assert np.allclose(pred["Q_mu"], pii.local_corrected(np.arange(0, 21), pred["mu"], H))
        assert np.allclose(pred["B_const"], pii.binomial_constant(np.arange(0, 21), H, pred["mu"]))
        assert pred["formula_version"] == pii.FORMULA_VERSION
        assert pred["quadrature"]["converged"]
        assert "omitted_bound" in pred["B_avg_info"]
        with pytest.raises(ValueError):
            pii.predict_all(H, M, N, 5, models=("nonsense",))

    def test_predict_all_invalid_models_are_nan(self):
        pred = pii.predict_all(40, 0, 2, 5)
        assert np.all(np.isnan(pred["F"])) and np.all(np.isnan(pred["Q_lambda"]))
        assert not pred["validity"]["integrated"]["valid"]
