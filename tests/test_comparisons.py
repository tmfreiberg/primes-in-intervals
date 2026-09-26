"""Tests for compare, score and winners."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

import primes_in_intervals as pii


class TestCompare:
    def test_structure_and_values_flat(self, analyzed_overlap):
        ds = analyzed_overlap
        assert pii.compare(ds) is ds
        C = list(ds["data"].keys())
        A = C[0]
        H = ds["header"]["interval_length"]
        c = C[-1]
        M, N = A, c - A
        m_max = max(m for cc in C for m, v in ds["data"][cc].items() if v)
        pred = pii.predict_all(H, M, N, m_max, models=("F", "F0", "B_const"))
        multiplier = N  # overlap: one interval per starting point
        for m in range(m_max + 1):
            probs, preds = ds["comparison"][c][m]
            assert probs[0] == ds["distribution"][c].get(m, 0.0)
            assert probs[1] == pytest.approx(float(pred["B_const"][m]))
            assert probs[2] == pytest.approx(float(pred["F"][m]))
            assert probs[3] == pytest.approx(float(pred["F0"][m]))
            assert preds[0] == ds["data"][c].get(m, 0)
            # unrounded predicted counts
            assert preds[1] == pytest.approx(probs[1] * multiplier)
            assert preds[2] == pytest.approx(probs[2] * multiplier)
            assert preds[3] == pytest.approx(probs[3] * multiplier)
        params = ds["prediction_parameters"][c]
        assert params["M"] == M and params["N"] == N and params["intervals"] == multiplier
        assert params["mu"] == pytest.approx(pii.averaged_parameter(H, M, N))
        assert ds["header"]["contents"][-1] == pii.COMPARISON_LABEL

    def test_every_integer_m_is_present(self, analyzed_overlap):
        ds = pii.compare(analyzed_overlap)
        C = list(ds["data"].keys())
        keys = sorted(ds["comparison"][C[-1]].keys())
        assert keys == list(range(keys[-1] + 1))

    def test_no_midpoint_normalization(self, analyzed_overlap):
        # The range (A, c] far from the origin: mu is H times the average of
        # 1/log t over the range, not H/(log(midpoint) - 1).
        ds = pii.compare(analyzed_overlap)
        C = list(ds["data"].keys())
        A, c = C[0], C[-1]
        H = ds["header"]["interval_length"]
        mu = ds["prediction_parameters"][c]["mu"]
        assert mu == pytest.approx(H / np.log((A + c) / 2), rel=1e-3)
        assert abs(mu - H / (np.log((A + c) / 2) - 1)) > 0.05 * mu

    def test_multiplier_disjoint(self, disjoint_dataset):
        ds = copy.deepcopy(disjoint_dataset)
        pii.analyze(ds)
        pii.compare(ds)
        C = list(ds["data"].keys())
        c = C[-1]
        H = ds["header"]["interval_length"]
        n_intervals = (c - C[0]) // H
        assert ds["prediction_parameters"][c]["intervals"] == n_intervals
        preds = [ds["comparison"][c][m][1][1] for m in ds["comparison"][c]]
        # the tuples cover m up to the largest observed count; the binomial's
        # remaining mass lies beyond it
        assert 0.9 * n_intervals < sum(preds) <= n_intervals

    def test_multiplier_prime_start(self, prime_start_dataset):
        ds = copy.deepcopy(prime_start_dataset)
        pii.analyze(ds)
        pii.compare(ds)
        C = list(ds["data"].keys())
        c = C[-1]
        n_intervals = sum(ds["data"][c].values())
        assert ds["prediction_parameters"][c]["intervals"] == n_intervals

    def test_nested(self, nested_overlap):
        ds = nested_overlap
        pii.compare(ds)
        keys = list(ds["nested_interval_data"].keys())
        c = keys[-1]
        assert set(ds["nested_interval_data"][c]) <= set(ds["comparison"][c])
        probs, preds = ds["comparison"][c][max(ds["nested_interval_data"][c])]
        assert len(probs) == 4 and len(preds) == 4
        assert ds["prediction_parameters"][c]["M"] == c[0]
        assert ds["prediction_parameters"][c]["N"] == c[1] - c[0]

    def test_requires_analyze(self, overlap_dataset, capsys):
        ds = copy.deepcopy(overlap_dataset)
        assert pii.compare(ds) is None
        assert "Analyze data first" in capsys.readouterr().out

    def test_no_data(self, capsys):
        assert pii.compare({"header": {}}) is None
        assert "No data to compare." in capsys.readouterr().out


class TestScore:
    @pytest.fixture()
    def compared(self, analyzed_overlap):
        return pii.compare(analyzed_overlap)

    def test_scores_match_direct_computation(self, compared):
        ds = pii.score(compared)
        assert ds is compared
        C = list(ds["data"].keys())
        c = C[-1]
        entry = ds["scores"][c]
        params = ds["prediction_parameters"][c]
        n = params["intervals"]
        P = {m: v / n for m, v in ds["data"][c].items()}
        lo, hi = ds["scores_range"]["central"]
        for name, slot in (("B", 1), ("F", 2), ("F0", 3)):
            q = np.array([ds["comparison"][c][m][0][slot] for m in range(hi + 1)])
            direct = pii.discrepancy(P, q, lo, hi)
            assert entry["central"][name]["E1"] == pytest.approx(direct.E1)
            assert entry["central"][name]["E2"] == pytest.approx(direct.E2)
            assert entry["central"][name]["Einf"] == pytest.approx(direct.Einf)
        g = entry["global"]["F"]
        assert g["m_lo"] == 0 and g["m_hi"] == entry["m_T"]
        assert g["predicted_tail_bound"] <= 1e-12
        assert entry["global"]["B"]["predicted_tail_bound"] == 0.0
        mass = entry["mass"]["F0"]
        # F0 over 0..m_T carries essentially all of its mass 1 - (2 - M)_+/N
        assert mass["deficit"] == pytest.approx(mass["expected_deficit"], abs=1e-9)
        assert mass["negative_mass"] == 0.0
        r = entry["ratio"]["central"]
        assert r["E1"] == pytest.approx(entry["central"]["F"]["E1"] / entry["central"]["F0"]["E1"])
        assert "scores" in ds["header"]["contents"]

    def test_requires_compare(self, analyzed_overlap, capsys):
        assert pii.score(analyzed_overlap) is None
        assert "Compare the data first" in capsys.readouterr().out

    def test_custom_central_range(self, compared):
        pii.score(compared, m_lo=1, m_hi=4)
        c = list(compared["data"].keys())[-1]
        s = compared["scores"][c]["central"]["F"]
        assert (s["m_lo"], s["m_hi"], s["n_terms"]) == (1, 4, 4)
        assert s["empirical_outside"] > 0  # mass at m = 0 and m > 4 lies outside


class TestWinners:
    @pytest.fixture()
    def compared(self, nested_overlap):
        pii.compare(nested_overlap)
        return nested_overlap

    def test_rankings_match_square_errors(self, compared):
        ds = compared
        assert pii.winners(ds) is ds
        for _c, w in ds["winners"].items():
            errors = {"B": w["B sq error"], "F": w["F sq error"], "F0": w["F0 sq error"]}
            ranked = sorted(errors.values())
            assert errors[w[1]] == ranked[0]
            assert errors[w[2]] == ranked[1]
            assert errors[w[3]] == ranked[2]

    def test_win_lists_partition_the_range(self, compared):
        ds = pii.winners(compared) if "winners" not in compared else compared
        for c, w in ds["winners"].items():
            M = [m for m in ds["comparison"][c] if ds["comparison"][c][m] != 0]
            full = list(range(min(M), max(M) + 1))
            union = (
                set(w["B wins for m in "])
                | set(w["F wins for m in "])
                | set(w["F0 wins for m in "])
            )
            assert union == set(full)
            tallies = sorted(
                [
                    len(w["B wins for m in "]),
                    len(w["F wins for m in "]),
                    len(w["F0 wins for m in "]),
                ],
                reverse=True,
            )
            assert w["most wins"] != ""
            assert tallies[0] >= tallies[1] >= tallies[2]

    def test_guards(self, compared, capsys):
        pii.winners(compared)
        capsys.readouterr()
        assert pii.winners(compared) is None
        assert "already been applied" in capsys.readouterr().out
        assert pii.winners({"header": {}}) is None
        assert "Compare the data first" in capsys.readouterr().out


def test_shift_constant_is_the_average_of_log():
    # log N + c(M/N) is the average of log t over M < t <= M + N.
    M, N = 5.0, 40.0
    from scipy.integrate import quad

    avg = quad(np.log, M, M + N)[0] / N
    assert math.log(N) + pii.shift_constant(M / N) == pytest.approx(avg)
    assert pii.shifted_parameter(30, 0, 1000.0) == pytest.approx(30 / (math.log(1000) - 1))
