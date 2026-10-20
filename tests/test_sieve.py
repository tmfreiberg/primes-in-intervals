"""Tests for the prime generator and the counting helpers."""

from __future__ import annotations

import primes_in_intervals as pii


def _is_prime(n: int) -> bool:
    """Trial division: slow, but independent of the package's sieves."""
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def test_first_primes():
    P = pii.postponed_sieve()
    assert [next(P) for _ in range(10)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_thousandth_prime_and_primality():
    P = pii.postponed_sieve()
    primes = [next(P) for _ in range(1000)]
    assert primes[-1] == 7919  # the 1000th prime
    assert primes == [n for n in range(2, 7920) if _is_prime(n)]


def test_next_prime_readme_values():
    assert (pii.next_prime(100), pii.next_prime(101)) == (101, 103)


def test_prime_pi_readme_values():
    assert (pii.prime_pi(1, 100), pii.prime_pi(1, 101)) == (25, 26)


def test_prime_pi_interval_convention():
    # (x, y]: endpoints excluded on the left, included on the right.
    assert pii.prime_pi(2, 3) == 1
    assert pii.prime_pi(3, 3) == 0
    assert pii.prime_pi(1, 2) == 1
