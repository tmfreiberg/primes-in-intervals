"""Numerical integration for the manuscript's averaged predictions.

Every averaged quantity in :mod:`primes_in_intervals.predictions` has the form

    (1/N) * integral from lower to upper of f(H / log t) dt,

with ``lower = max(2, M)`` and ``upper = M + N``, where ``f`` may be
vector-valued (one component per prime count ``m``).  The range can be very
long (``N`` up to ``10**12`` or more) while the integrand varies only through
``log t``, so the integral is evaluated after the change of variable
``t = lower * exp(sigma)``:

    (1/N) * integral from 0 to log(upper / lower) of f(H / log t) t dsigma,

with ``t = lower * exp(sigma)`` (the upper limit is evaluated as
``log1p((upper - lower) / lower)``, which keeps a narrow range far from the
origin at full relative precision).

On this scale the integrand is smooth and the range is short (about ``log N``
units), and SciPy's adaptive vector quadrature :func:`scipy.integrate.quad_vec`
resolves it easily.  The tolerances are explicit inputs and the routine's
error estimate is returned with every value; the estimate is a heuristic,
not a bound, which is why :func:`mp_log_average` provides an independent
high-precision evaluation (with :mod:`mpmath`) for spot checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import quad_vec

__all__ = [
    "QuadratureSettings",
    "QuadratureResult",
    "log_average",
    "mp_log_average",
]


@dataclass(frozen=True)
class QuadratureSettings:
    """Tolerances handed to :func:`scipy.integrate.quad_vec`.

    Attributes
    ----------
    epsabs : float
        Absolute tolerance on the *averaged* value (the integral divided by
        ``N``); it is scaled by ``N`` before being passed to SciPy.
    epsrel : float
        Relative tolerance.
    limit : int
        Maximum number of subintervals.
    quadrature : str
        The rule (``'gk21'`` or ``'gk15'``).
    """

    epsabs: float = 1e-13
    epsrel: float = 1e-11
    limit: int = 500
    quadrature: str = "gk21"

    def as_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dictionary (for caches and reports)."""
        return asdict(self)


@dataclass
class QuadratureResult:
    """Value and diagnostics of one averaged integral.

    Attributes
    ----------
    value : numpy.ndarray
        The averaged integral, one entry per integrand component.
    error : float
        SciPy's estimate of the absolute error of ``value`` (max norm over
        components).  An estimate, not a rigorous bound.
    neval : int
        Number of integrand evaluations.
    converged : bool
        Whether SciPy reported success.
    lower, upper : float
        The integration range in the original variable ``t``.
    settings : dict
        The tolerances used.
    """

    value: np.ndarray
    error: float
    neval: int
    converged: bool
    lower: float
    upper: float
    settings: dict[str, Any] = field(default_factory=dict)


def log_average(
    f: Callable[[float], np.ndarray],
    lower: float,
    upper: float,
    N: float,
    settings: QuadratureSettings | None = None,
) -> QuadratureResult:
    """Evaluate ``(1/N) * integral_{lower}^{upper} f(t) dt`` via ``t = exp(s)``.

    Parameters
    ----------
    f : callable
        Maps a scalar ``t`` to a NumPy array (the integrand at ``t``).
    lower, upper : float
        Integration range in ``t``; must satisfy ``0 < lower <= upper``.
    N : float
        The normalizing length (the manuscript's ``N``), positive.
    settings : QuadratureSettings, optional
        Tolerances (defaults: absolute ``1e-13`` on the averaged value,
        relative ``1e-11``).

    Returns
    -------
    QuadratureResult
        Averaged value, error estimate, and diagnostics.  When
        ``upper <= lower`` the value is zero with zero error.
    """
    settings = settings or QuadratureSettings()
    if upper <= lower:
        probe = np.asarray(f(max(lower, 1.0)), dtype=float)
        return QuadratureResult(
            np.zeros_like(probe), 0.0, 0, True, lower, upper, settings.as_dict()
        )
    if lower <= 0:
        raise ValueError("the lower limit must be positive (the integrand uses log t)")

    # Integrate in sigma = log(t / lower), so that a narrow range far from the
    # origin (upper / lower close to 1) keeps its full relative precision: the
    # upper limit is log1p((upper - lower) / lower), not a difference of two
    # rounded logarithms.
    def g(sigma: float) -> np.ndarray:
        t = lower * np.exp(sigma)
        return np.asarray(f(t), dtype=float) * t

    value, err, info = quad_vec(
        g,
        0.0,
        float(np.log1p((upper - lower) / lower)),
        epsabs=settings.epsabs * N,
        epsrel=settings.epsrel,
        limit=settings.limit,
        norm="max",
        quadrature=settings.quadrature,
        full_output=True,
    )
    return QuadratureResult(
        value=np.asarray(value, dtype=float) / N,
        error=float(err) / N,
        neval=int(info.neval),
        converged=bool(info.success),
        lower=float(lower),
        upper=float(upper),
        settings=settings.as_dict(),
    )


def mp_log_average(
    f: Callable[[Any], Any],
    lower: float,
    upper: float,
    N: float,
    dps: int = 30,
) -> Any:
    """High-precision cross-check of :func:`log_average` for a scalar integrand.

    Uses :func:`mpmath.quad` (tanh-sinh) at ``dps`` decimal digits on the
    same ``s = log t`` formulation, with the range split at integer values
    of ``s`` so the exponential weight never spans more than a factor ``e``
    inside one panel.

    Parameters
    ----------
    f : callable
        Maps an ``mpmath.mpf`` argument ``t`` to an ``mpf`` value.
    lower, upper : float
        Integration range in ``t``.
    N : float
        Normalizing length.
    dps : int, optional
        Working precision in decimal digits (default 30).

    Returns
    -------
    mpmath.mpf
        The averaged integral.
    """
    import mpmath as mp

    if upper <= lower:
        return mp.mpf(0)
    with mp.workdps(dps):
        a, b = mp.log(mp.mpf(lower)), mp.log(mp.mpf(upper))
        knots = [a] + [mp.mpf(k) for k in range(int(mp.floor(a)) + 1, int(mp.ceil(b)))] + [b]
        knots = [k for i, k in enumerate(knots) if i == 0 or k > knots[i - 1]]
        total = mp.quad(lambda s: f(mp.exp(s)) * mp.exp(s), knots)
        return total / mp.mpf(N)
