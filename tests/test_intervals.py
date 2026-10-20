"""Tests for the interval counters, anchored on the exposition's reference values."""

from __future__ import annotations

import primes_in_intervals as pii

# The full frequency dictionary for disjoint centades between 2 and 3 million,
# as printed in the exposition.
DISJOINT_2_3_MILLION = {
    0: 1,
    1: 25,
    2: 97,
    3: 337,
    4: 776,
    5: 1408,
    6: 1881,
    7: 1995,
    8: 1525,
    9: 1035,
    10: 559,
    11: 227,
    12: 98,
    13: 28,
    14: 6,
    15: 1,
    17: 1,
}

# The full frequency dictionary for prime-starting intervals of length 100
# with starting primes between 2 and 3 million, as printed in the exposition.
PRIME_START_2_3_MILLION = {
    0: 12,
    1: 155,
    2: 799,
    3: 2584,
    4: 6063,
    5: 10259,
    6: 13359,
    7: 12896,
    8: 10312,
    9: 6468,
    10: 3175,
    11: 1283,
    12: 396,
    13: 97,
    14: 15,
    15: 5,
    16: 2,
    17: 3,
}


class TestDisjoint:
    def test_single_interval_equals_prime_pi(self):
        # disjoint(0, H, H) is {pi(H): 1}.
        expected = {10: 4, 20: 8, 30: 10, 40: 12, 50: 15, 60: 17, 70: 19, 80: 22, 90: 24}
        for H, m in expected.items():
            assert pii.disjoint(0, H, H) == {m: 1}

    def test_two_to_three_million(self):
        got = pii.disjoint(2 * 10**6, 3 * 10**6, 100)
        assert got == DISJOINT_2_3_MILLION
        assert sum(got.values()) == 10**4  # number of intervals
        assert sum(m * v for m, v in got.items()) == 67883  # primes in (2e6, 3e6]

    def test_ragged_upper_bound_ignored(self):
        # (B - A) not a multiple of H: the ragged tail contributes nothing.
        assert pii.disjoint(0, 97, 10) == pii.disjoint(0, 90, 10)

    def test_counts_number_of_intervals(self):
        got = pii.disjoint(1000, 2000, 57)
        assert sum(got.values()) == (2000 - 1000) // 57


class TestDisjointCheckpoints:
    def test_snapping_and_dedup(self):
        # The docstring's example: checkpoints snap onto C[0] mod H, dedupe.
        ds = pii.disjoint_cp([0, 10, 100, 210, 350, 400], 100)
        assert list(ds["data"].keys()) == [0, 100, 200, 300, 400]
        assert ds["header"]["no_of_checkpoints"] == 5

    def test_final_checkpoint_matches_plain(self):
        ds = pii.disjoint_cp([0, 250, 500, 750, 1000], 25)
        trimmed = {m: v for m, v in ds["data"][1000].items() if v != 0}
        assert trimmed == pii.disjoint(0, 1000, 25)

    def test_header_contents(self):
        ds = pii.disjoint_cp([0, 100, 200], 50)
        h = ds["header"]
        assert h["interval_type"] == "disjoint"
        assert (h["lower_bound"], h["upper_bound"], h["interval_length"]) == (0, 200, 50)
        assert h["contents"] == ["data"]


class TestOverlap:
    def test_hand_counted_small_case(self):
        # For a = 1..5 the windows (a, a+5] contain 3, 3, 2, 2, 1 primes.
        assert pii.overlap(0, 5, 5) == {1: 1, 2: 2, 3: 2}

    def test_single_left_endpoint(self):
        # overlap(M, M+1, H) counts primes in (M+1, M+1+H].
        assert pii.overlap(999, 1000, 1000) == {135: 1}
        assert pii.prime_pi(1000, 2000) == 135

    def test_hundred_windows(self):
        got = pii.overlap(0, 100, 10)
        assert got == {1: 8, 2: 46, 3: 38, 4: 7, 5: 1}
        assert sum(got.values()) == 100
        assert sum(m * v for m, v in got.items()) == 247

    def test_total_windows(self):
        got = pii.overlap(500, 1500, 37)
        assert sum(got.values()) == 1000


class TestOverlapCheckpoints:
    def test_final_checkpoint_matches_plain(self):
        ds = pii.overlap_cp([0, 250, 500, 750, 1000], 25)
        trimmed = {m: v for m, v in ds["data"][1000].items() if v != 0}
        assert trimmed == pii.overlap(0, 1000, 25)

    def test_intermediate_checkpoints_match_plain(self):
        C = [100, 300, 700, 1100]
        ds = pii.overlap_cp(list(C), 40)
        for c in C[1:]:
            trimmed = {m: v for m, v in ds["data"][c].items() if v != 0}
            assert trimmed == pii.overlap(100, c, 40)

    def test_unsorted_checkpoints_are_sorted(self):
        ds = pii.overlap_cp([400, 0, 200, 100, 300], 50)
        assert list(ds["data"].keys()) == [0, 100, 200, 300, 400]


class TestPrimeStart:
    def test_hand_counted_small_case(self):
        # Primes 11, 13, 17, 19 start windows with 6, 5, 5, 4 primes.
        assert pii.prime_start(10, 20, 20) == {4: 1, 5: 2, 6: 1}

    def test_two_to_three_million(self):
        got = pii.prime_start(2 * 10**6, 3 * 10**6, 100)
        assert got == PRIME_START_2_3_MILLION
        # One window per starting prime in (2e6, 3e6].
        assert sum(got.values()) == 67883

    def test_checkpoint_final_matches_plain(self):
        ds = pii.prime_start_cp(list(range(2 * 10**6, 3 * 10**6 + 1, 10**5)), 100)
        assert ds["data"][3 * 10**6] == PRIME_START_2_3_MILLION


class TestDispatcher:
    def test_routes(self):
        C = [0, 100, 200]
        for itype, cp in [
            ("disjoint", pii.disjoint_cp),
            ("overlap", pii.overlap_cp),
            ("prime_start", pii.prime_start_cp),
        ]:
            assert pii.intervals(list(C), 20, itype) == cp(list(C), 20)

    def test_default_is_overlap(self):
        C = [0, 100, 200]
        assert pii.intervals(list(C), 20) == pii.overlap_cp(list(C), 20)

    def test_unknown_type_returns_none(self):
        assert pii.intervals([0, 100], 20, "banana") is None


class TestAnyIntervals:
    def test_two_prime_generators_reproduce_prime_start(self):
        got = pii.anyIntervals(0, 500, 20, pii.postponed_sieve(), pii.postponed_sieve())
        assert got == pii.prime_start(0, 500, 20)

    def test_checkpoint_version_matches(self):
        C = list(range(0, 501, 100))
        got = pii.anyIntervals_cp(list(C), 20, pii.postponed_sieve(), pii.postponed_sieve())
        ds = pii.prime_start_cp(list(C), 20)
        assert got == ds["data"]

    def test_integers_reproduce_overlap(self):
        from itertools import count

        got = pii.anyIntervals(0, 300, 10, count(1), pii.postponed_sieve())
        assert got == pii.overlap(0, 300, 10)


class TestZeros:
    def test_pad_to_common_keys(self):
        md = {1: {0: 0, 1: 2, 2: 0}, 2: {1: 0, 3: 4}}
        assert pii.zeros(md) == {1: {1: 2, 3: 0}, 2: {1: 0, 3: 4}}

    def test_strip_zero_items(self):
        md = {1: {0: 0, 1: 2, 2: 0}, 2: {1: 0, 3: 4}}
        assert pii.zeros(md, pad="no") == {1: {1: 2}, 2: {3: 4}}


class TestOverlapExtension:
    def test_counts_agree_with_overlap(self):
        show_me, output = pii.overlap_extension(1000, 2000, 50, [3, 11])
        assert output == pii.overlap(1000, 2000, 50)

    def test_show_me_lists_are_consistent(self):
        # First find every m that occurs, then request them all.
        counts = pii.overlap(0, 200, 15)
        show_me, output = pii.overlap_extension(0, 200, 15, list(counts.keys()))
        assert output == counts
        # Every listed a realizes its m, and the list lengths match the counts.
        for m, endpoints in show_me.items():
            assert len(endpoints) == output[m]
            for a in endpoints[:5]:
                assert pii.prime_pi(a, a + 15) == m
        # Together the lists cover (0, 200] exactly once.
        everything = sorted(a for lst in show_me.values() for a in lst)
        assert everything == list(range(1, 201))
