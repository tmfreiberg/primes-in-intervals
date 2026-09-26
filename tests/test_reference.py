"""Independent validation of the sliding-window counters.

The reference here shares nothing with the postponed sieve: an array sieve of
Eratosthenes (checked against trial division) and a direct visit of every
starting point.
"""

from __future__ import annotations

import random
from itertools import islice

import numpy as np
import pytest

import primes_in_intervals as pii
from primes_in_intervals import reference as ref

LIMIT = 12000


@pytest.fixture(scope="module")
def table():
    return ref.prime_table(LIMIT)


@pytest.fixture(scope="module")
def pi():
    return ref.prime_pi_table(LIMIT + 200)


class TestPrimeTables:
    def test_array_sieve_matches_trial_division(self, table):
        trial = np.array([ref.is_prime_trial(n) for n in range(LIMIT + 1)])
        assert np.array_equal(table, trial)

    def test_postponed_sieve_matches_array_sieve(self, table):
        want = [int(n) for n in np.flatnonzero(table)]
        got = list(islice(pii.postponed_sieve(), len(want)))
        assert got == want

    def test_segmented_sieve(self, table):
        for lo, hi in [
            (0, 100),
            (1, 1),
            (0, 0),
            (2, 2),
            (10, 9),
            (1000, 5000),
            (LIMIT - 500, LIMIT),
        ]:
            assert np.array_equal(ref.prime_table_segment(lo, hi), table[lo : hi + 1])
        lo = 1318715734  # around e^21, where the stored data live
        seg = ref.prime_table_segment(lo, lo + 2000)
        assert np.array_equal(seg, [ref.is_prime_trial(n) for n in range(lo, lo + 2001)])

    def test_empty_and_small(self):
        assert ref.overlap_reference(10, 10, 5) == {}
        assert ref.overlap_reference(10, 3, 5) == {}
        assert ref.first_moment_identity(10, 3, 5) == 0
        assert ref.overlap_reference(0, 5, 5) == {1: 1, 2: 2, 3: 2}


class TestOverlapAgainstReference:
    def test_random_ranges(self, pi):
        rng = random.Random(7)
        for _ in range(400):
            H = rng.choice([1, 2, 3, 5, 7, 10, 13, 20, 50, 100])
            A = rng.randint(0, 3000)
            B = A + rng.randint(-5, 400)
            got = pii.overlap(A, B, H)
            assert got == ref.overlap_reference(A, B, H, pi), (A, B, H)
            assert sum(got.values()) == max(B - A, 0)
            assert sum(m * g for m, g in got.items()) == ref.first_moment_identity(A, B, H, pi)

    def test_endpoint_cases(self, table, pi):
        primes = [int(p) for p in np.flatnonzero(table)][:150]
        for p in primes:
            for H in (1, 2, 3, 6, 10, 30):
                cases = [
                    (p - 1, p + 50),  # first starting point is the prime p
                    (p - 1, p),  # a single starting point, which is prime
                    (0, p),  # last starting point prime
                    (0, p - 1),  # B + 1 prime: the last window's right end
                    (p - H, p + H),  # window (p - H, p] contains the prime at its right end
                    (p - H - 1, p - H),  # single window ending at p
                    (p - 2, p - 1),  # single window (p - 1, p - 1 + H] starting just below p
                ]
                for A, B in cases:
                    if A < 0:
                        continue
                    assert pii.overlap(A, B, H) == ref.overlap_reference(A, B, H, pi), (A, B, H)

    def test_simultaneous_entry_and_exit_occur_and_agree(self, pi):
        # H = 2 over the twin primes (3, 5), (5, 7), (11, 13), ...: the window
        # (a, a + 2] gains and loses a prime at the same step repeatedly.
        assert pii.overlap(0, 200, 2) == ref.overlap_reference(0, 200, 2, pi)
        assert pii.overlap(2, 2000, 6) == ref.overlap_reference(2, 2000, 6, pi)


class TestOverlapCpAgainstReference:
    def test_random_checkpoints_with_duplicates_and_events(self, table, pi):
        rng = random.Random(11)
        primes = [int(p) for p in np.flatnonzero(table)][:300]
        for _ in range(120):
            H = rng.choice([1, 2, 3, 5, 7, 10, 20, 37, 100])
            start = rng.randint(0, 1500)
            C = [start] + [start + rng.randint(0, 1500) for _ in range(rng.randint(1, 10))]
            C.append(rng.choice(C))  # duplicate
            q = rng.choice(primes)
            C += [q, q - 1, max(0, q - H)]  # checkpoints at crossing events
            C_input = list(C)
            rng.shuffle(C_input)
            got = pii.overlap_cp(list(C_input), H)
            want = ref.overlap_cp_reference(C, H)
            assert got["data"] == want["data"], (C_input, H)
            assert got["header"] == want["header"], (C_input, H)
            Cs = sorted(set(C))
            assert got["header"]["no_of_checkpoints"] == len(Cs) == len(got["data"])
            for i in range(1, len(Cs)):
                standalone = pii.overlap(Cs[0], Cs[i], H)
                assert {m: v for m, v in got["data"][Cs[i]].items() if v} == standalone
                for j in range(1, i):
                    diff = {
                        m: got["data"][Cs[i]][m] - got["data"][Cs[j]][m] for m in got["data"][Cs[i]]
                    }
                    assert {m: v for m, v in diff.items() if v} == ref.overlap_reference(
                        Cs[j], Cs[i], H, pi
                    )

    def test_sorts_and_deduplicates_in_place(self):
        C = [300, 100, 200, 100, 0]
        pii.overlap_cp(C, 10)
        assert C == [0, 100, 200, 300]

    def test_dense_checkpoints_from_zero(self):
        for H in (1, 2, 3, 5, 10, 25):
            C = list(range(0, 61)) + list(range(70, 2001, 37)) + [2000]
            got = pii.overlap_cp(list(C), H)
            want = ref.overlap_cp_reference(C, H)
            assert got["data"] == want["data"]
            for N in range(1, 61):
                assert sum(got["data"][N].values()) == N

    def test_prime_start_cp_header_counts_distinct_checkpoints(self):
        ds = pii.prime_start_cp([100, 0, 50, 50], 10)
        assert ds["header"]["no_of_checkpoints"] == 3 == len(ds["data"])
        assert ds["header"]["lower_bound"] == 0 and ds["header"]["upper_bound"] == 100
