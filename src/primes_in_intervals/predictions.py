r"""Theoretical predictions for the distribution of primes in intervals.

Notation follows the manuscript.  With ``X(n; H) = pi(n + H) - pi(n)`` and

    P(m; H, M, N) = #{M < n <= M + N : X(n; H) = m} / N,

the quantities predicted are, for integer ``m >= 0`` and ``u >= 0``,

* ``p_m(u) = exp(-u) u^m / m!`` (:func:`poisson_pmf`);
* ``eta(H) = (log H + log(2 pi) + gamma - 1) / H`` (:func:`eta`);
* the **local** corrected expression
  ``Q(m; u, H) = p_m(u) [1 - eta(H)/2 * ((m - u)^2 - m)]``
  (:func:`local_corrected`; the older name :func:`frei` is the same
  function with the arguments in the order ``(H, m, u)``);
* the **integrated** predictions

      F(m; H, M, N)   = (1/N) int_{max(2,M)}^{M+N} Q(m; H/log t, H) dt
      F_0(m; H, M, N) = (1/N) int_{max(2,M)}^{M+N} p_m(H/log t) dt

  (:func:`integrated_corrected`, :func:`integrated_poisson`), and their
  difference ``F - F_0``, evaluated directly from the correction integrand
  (:func:`integrated_correction`), all three returned together by
  :func:`integrated`;
* the averaged parameter ``mu = (H/N) int_{max(2,M)}^{M+N} dt/log t``
  (:func:`averaged_parameter`) and the shifted parameter
  ``lambda = H / (log N + c(M/N))``, ``c(a) = (a+1) log(a+1) - a log a - 1``
  (:func:`shifted_parameter`, :func:`shift_constant`); for ``M = 0``,
  ``lambda = H / (log N - 1)``;
* the constant-density binomial ``Binom(H, mu/H)`` (:func:`binomial_constant`),
  a constant-density approximation to Cramér's model, and the
  density-averaged local binomial
  ``(1/N) int Binom(m; H, 1/log t) dt`` over the part of the range where
  ``1/log t <= 1`` (:func:`binomial_averaged`).

``Q`` sums to one over ``m >= 0`` for every ``u``; ``F`` and ``F_0`` sum to
``(M + N - max(2, M)) / N`` (for ``M = 0`` and ``N >= 2``, to ``1 - 2/N``).
Nothing here renormalizes.  ``Q`` and ``F`` are signed and can be negative;
nothing here clips.

The function :func:`frei_alt` is retained for backward compatibility only: it
is the local expression with an extra first-order term appropriate to the
retired density ``1/log N`` at the midpoint of a range, and it plays no part
in the current workflow.

Everything that can be evaluated in closed form accepts a real ``m`` (via the
gamma function) so smooth guide curves can be drawn; the integrated
predictions accept real ``m`` for the same reason.  Only integer ``m`` carries
a probabilistic meaning, and every comparison score is computed at integer
``m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy
from scipy.special import binom, expi, gamma, gammaln

from primes_in_intervals.quadrature import QuadratureResult, QuadratureSettings, log_average

__all__ = [
    "DEFAULT_MODELS",
    "FORMULA_VERSION",
    "MODELS",
    "MS",
    "IntegratedPrediction",
    "averaged_parameter",
    "binom_pmf",
    "binomial_averaged",
    "binomial_constant",
    "eta",
    "frei",
    "frei_alt",
    "integrated",
    "integrated_corrected",
    "integrated_correction",
    "integrated_poisson",
    "local_corrected",
    "parameter_validity",
    "poisson_pmf",
    "predict_all",
    "shift_constant",
    "shifted_parameter",
]

#: Identifies the formulas implemented here, for prediction caches and reports.
FORMULA_VERSION = (
    "manuscript-2026-09: eta(H) = (log H + log 2pi + gamma - 1)/H; F, F0 on [max(2,M), M+N]"
)

#: The constant ``1 - gamma - log(2*pi)`` = -1.41509...  (kept under its
#: historical name; ``eta(H) = (log H - MS) / H``).
MS = 1 - sympy.EulerGamma.evalf() - np.log(2 * (np.pi))

_EULER_GAMMA = float(sympy.EulerGamma.evalf(30))


# --------------------------------------------------------------------------
# Closed-form pieces
# --------------------------------------------------------------------------


def eta(H: float) -> float:
    """Return ``eta(H) = (log H + log(2 pi) + gamma - 1) / H``.

    Parameters
    ----------
    H : int or float
        Interval length, positive.

    Returns
    -------
    float
    """
    return (math.log(H) + math.log(2 * math.pi) + _EULER_GAMMA - 1) / H


def poisson_pmf(m: Any, u: float) -> Any:
    """Return ``p_m(u) = exp(-u) u^m / m!``, extended to real ``m`` via the gamma function.

    Evaluated as ``exp(m log u - u - lgamma(m + 1))`` for ``u > 0``, which is
    stable far into the tails; for ``u = 0`` the value is ``1`` at ``m = 0``
    and ``0`` elsewhere.

    Parameters
    ----------
    m : int, float, or array
        Number of primes (non-negative; real values allowed).
    u : float
        Poisson parameter, ``u >= 0``.

    Returns
    -------
    float or numpy.ndarray
    """
    m_arr = np.asarray(m, dtype=float)
    if u <= 0:
        out = np.where(m_arr == 0, 1.0, 0.0)
    else:
        out = np.exp(m_arr * math.log(u) - u - gammaln(m_arr + 1))
    return out if np.ndim(m) else float(out)


def local_corrected(m: Any, u: float, H: float) -> Any:
    """Return the local corrected expression ``Q(m; u, H)``.

    ``Q(m; u, H) = p_m(u) * [1 - eta(H)/2 * ((m - u)^2 - m)]``.

    Parameters
    ----------
    m : int, float, or array
        Number of primes (real values allowed for guide curves).
    u : float
        The Poisson parameter (``mu``, ``lambda``, or ``H / log t``).
    H : int or float
        Interval length.

    Returns
    -------
    float or numpy.ndarray
        Signed; can be negative.
    """
    m_arr = np.asarray(m, dtype=float)
    out = poisson_pmf(m_arr, u) * (1 - eta(H) / 2 * ((m_arr - u) ** 2 - m_arr))
    return out if np.ndim(m) else float(out)


def shift_constant(alpha: float) -> float:
    """Return ``c(alpha) = (alpha + 1) log(alpha + 1) - alpha log(alpha) - 1`` (``0 log 0 = 0``).

    Parameters
    ----------
    alpha : float
        The ratio ``M / N``, non-negative.

    Returns
    -------
    float
    """
    if alpha < 0:
        raise ValueError("alpha = M/N must be non-negative")
    a_log_a = 0.0 if alpha == 0 else alpha * math.log(alpha)
    return (alpha + 1) * math.log(alpha + 1) - a_log_a - 1


def shifted_parameter(H: float, M: float, N: float) -> float:
    """Return ``lambda = H / (log N + c(M/N))``.

    This is ``H`` divided by the average of ``log t`` over ``M < t <= M + N``.
    For ``M = 0`` it is ``H / (log N - 1)``, which is only meaningful when
    ``N > e``; see :func:`parameter_validity`.

    Parameters
    ----------
    H : int or float
        Interval length.
    M : int or float
        Left end of the range of starting points.
    N : int or float
        Number of starting points, positive.

    Returns
    -------
    float
        ``nan`` when the denominator is not positive.
    """
    denominator = math.log(N) + shift_constant(M / N)
    if denominator <= 0:
        return math.nan
    return H / denominator


def _li(x: float) -> float:
    """Return the logarithmic integral ``li(x) = Ei(log x)`` for ``x > 1``."""
    return float(expi(math.log(x)))


def averaged_parameter(H: float, M: float, N: float) -> float:
    """Return ``mu = (H/N) * integral_{max(2,M)}^{M+N} dt / log t``.

    Evaluated in closed form as ``H * (li(M + N) - li(max(2, M))) / N``
    using :func:`scipy.special.expi`.

    Parameters
    ----------
    H : int or float
        Interval length.
    M : int or float
        Left end of the range of starting points.
    N : int or float
        Number of starting points, positive.

    Returns
    -------
    float
        Zero when ``M + N <= max(2, M)``.
    """
    lower, upper = max(2.0, float(M)), float(M) + float(N)
    if upper <= lower:
        return 0.0
    return H * (_li(upper) - _li(lower)) / N


def binom_pmf(H, m, p):
    """Binomial probability of ``m`` successes in ``H`` trials with probability ``p``.

    A constant-density approximation to Cramér's model: if every integer in
    an interval of length ``H`` were prime independently with probability
    ``p``, the number of primes would be ``Binom(H, p)``.  (Cramér's model
    proper assigns the probability ``1/log n`` to the integer ``n``, so its
    interval count is not exactly binomial.)

    Parameters
    ----------
    H : int
        Number of trials (interval length).
    m : int, float, or array
        Number of successes; real values are allowed (the generalized binomial
        coefficient ``Gamma(H+1)/(Gamma(m+1) Gamma(H-m+1))`` is used), so the
        curve can be drawn continuously.  This is an interpolation of the
        probability mass function, not a Gamma distribution, and has no
        probabilistic meaning between integers.
    p : float
        Success probability, ``0 <= p <= 1``.

    Returns
    -------
    float or array
        ``binom(H, m) * p**m * (1 - p)**(H - m)``.
    """
    return binom(H, m) * (p**m) * (1 - p) ** (H - m)


def binomial_constant(m: Any, H: int, mu: float) -> Any:
    """Return ``Binom(m; H, mu/H)`` at integer ``m``, the constant-density binomial prediction.

    Parameters
    ----------
    m : int or array of int
        Number of primes.
    H : int
        Interval length (number of trials).
    mu : float
        The averaged parameter; the success probability is ``mu / H`` and must
        lie in ``[0, 1]``.

    Returns
    -------
    float or numpy.ndarray
        ``nan`` when ``mu / H`` is not a valid probability.

    Raises
    ------
    ValueError
        If ``m`` is not integer-valued.
    """
    from scipy.stats import binom as binom_dist

    m_arr = np.asarray(m)
    if not np.all(np.equal(np.mod(m_arr, 1), 0)):
        raise ValueError("binomial_constant is defined at integer m only")
    p = mu / H
    if not 0 <= p <= 1:
        out = np.full(np.shape(m_arr), math.nan)
        return out if np.ndim(m) else math.nan
    out = binom_dist.pmf(m_arr.astype(int), H, p)
    return out if np.ndim(m) else float(out)


def frei(H, m, t):
    """Evaluate the local corrected expression ``Q(m; t, H)`` (older argument order).

    This is exactly :func:`local_corrected` with the arguments reordered as
    ``(H, m, t)``.  It is the **local** expression, not the integrated
    prediction ``F`` of the manuscript (see :func:`integrated_corrected`).

    Parameters
    ----------
    H : int
        Interval length.
    m : int, float, or array
        Number of primes; real values allowed for plotting.
    t : float
        The Poisson parameter (``mu`` or ``lambda``).

    Returns
    -------
    float or array
    """
    return local_corrected(m, t, H)


def frei_alt(H, m, t):
    """Evaluate the retired alternative expression ``F*(H, m, t)`` (backward compatibility only).

    As :func:`frei`, but with an additional first-order term ``(t/H) (m - t)``
    that compensated for using the density ``1/log N`` (at the midpoint of a
    range) in place of ``1/(log N - 1)``.  The current workflow averages the
    density over the range instead, so this expression is no longer used for
    comparisons or plots; it is kept so that older scripts continue to run.

    Parameters
    ----------
    H : int
        Interval length.
    m : int, float, or array
        Number of primes; real values allowed for plotting.
    t : float
        The Poisson parameter.

    Returns
    -------
    float or array
    """
    Q_1 = m - t
    Q_2 = ((m - t) ** 2 - m) / 2
    return np.exp(-t) * (t**m / gamma(m + 1)) * (1 + (t / H) * Q_1 - ((np.log(H) - MS) / (H)) * Q_2)


# --------------------------------------------------------------------------
# Integrated predictions
# --------------------------------------------------------------------------


@dataclass
class IntegratedPrediction:
    """The integrated predictions at one ``(H, M, N)``, for a vector of ``m``.

    Attributes
    ----------
    m : numpy.ndarray
        The prime counts at which the predictions were evaluated.
    F0 : numpy.ndarray
        ``F_0(m; H, M, N)``, the averaged Poisson expression.
    correction : numpy.ndarray
        ``F - F_0``, integrated directly from the correction integrand.
    F : numpy.ndarray
        ``F0 + correction``.
    error : float
        Quadrature error estimate (max over the stacked components, on the
        averaged scale).  A heuristic estimate, not a rigorous bound.
    neval : int
        Integrand evaluations used.
    converged : bool
        Whether the adaptive integrator reported success.
    H, M, N : float
        The parameters.
    lower, upper : float
        The integration range ``[max(2, M), M + N]``.
    settings : dict
        Quadrature tolerances.
    formula_version : str
        :data:`FORMULA_VERSION`.
    """

    m: np.ndarray
    F0: np.ndarray
    correction: np.ndarray
    F: np.ndarray
    error: float
    neval: int
    converged: bool
    H: float
    M: float
    N: float
    lower: float
    upper: float
    settings: dict[str, Any] = field(default_factory=dict)
    formula_version: str = FORMULA_VERSION

    @property
    def mass(self) -> float:
        """Return ``sum F`` over the evaluated ``m`` (compare with ``(upper - lower)/N``)."""
        return float(np.sum(self.F))

    @property
    def negative_mass(self) -> float:
        """Return the total of the negative values of ``F`` over the evaluated ``m``."""
        return float(-np.sum(self.F[self.F < 0]))

    @property
    def expected_mass(self) -> float:
        """Return ``(M + N - max(2, M)) / N``, the exact total of ``F`` over all ``m >= 0``."""
        return max(self.upper - self.lower, 0.0) / self.N


def _range(M: float, N: float) -> tuple[float, float]:
    return max(2.0, float(M)), float(M) + float(N)


def integrated(
    m: Any,
    H: float,
    M: float,
    N: float,
    settings: QuadratureSettings | None = None,
) -> IntegratedPrediction:
    """Evaluate ``F_0``, ``F - F_0`` and ``F`` at once for a vector of ``m``.

    One adaptive quadrature over ``s = log t`` handles the stacked integrand
    ``[p_m(H/s) ; -(eta(H)/2) ((m - H/s)^2 - m) p_m(H/s)]`` for all ``m``,
    so the Poisson part and the correction come from the same evaluations
    and ``F`` is assembled as ``F_0 + (F - F_0)``, avoiding cancellation
    when the correction is small.

    Parameters
    ----------
    m : int, float, or array
        Prime counts (real values are accepted for guide curves).
    H : int or float
        Interval length.
    M : int or float
        Left end of the range of starting points.
    N : int or float
        Number of starting points, positive.
    settings : QuadratureSettings, optional
        Quadrature tolerances.

    Returns
    -------
    IntegratedPrediction
    """
    m_arr = np.atleast_1d(np.asarray(m, dtype=float))
    lower, upper = _range(M, N)
    eta_H = eta(H)

    def integrand(t: float) -> np.ndarray:
        u = H / math.log(t)
        p = poisson_pmf(m_arr, u)
        corr = -(eta_H / 2) * ((m_arr - u) ** 2 - m_arr) * p
        return np.concatenate([p, corr])

    res: QuadratureResult = log_average(integrand, lower, upper, N, settings)
    k = len(m_arr)
    F0 = res.value[:k]
    corr = res.value[k:]
    return IntegratedPrediction(
        m=m_arr,
        F0=F0,
        correction=corr,
        F=F0 + corr,
        error=res.error,
        neval=res.neval,
        converged=res.converged,
        H=float(H),
        M=float(M),
        N=float(N),
        lower=lower,
        upper=upper,
        settings=res.settings,
    )


def integrated_poisson(m: Any, H: float, M: float, N: float, **kwargs: Any) -> Any:
    """Return ``F_0(m; H, M, N)``; see :func:`integrated`."""
    out = integrated(m, H, M, N, **kwargs).F0
    return out if np.ndim(m) else float(out[0])


def integrated_corrected(m: Any, H: float, M: float, N: float, **kwargs: Any) -> Any:
    """Return ``F(m; H, M, N)``; see :func:`integrated`."""
    out = integrated(m, H, M, N, **kwargs).F
    return out if np.ndim(m) else float(out[0])


def integrated_correction(m: Any, H: float, M: float, N: float, **kwargs: Any) -> Any:
    """Return ``F - F_0`` at ``(m; H, M, N)``, from the correction integrand.

    See :func:`integrated`.
    """
    out = integrated(m, H, M, N, **kwargs).correction
    return out if np.ndim(m) else float(out[0])


def binomial_averaged(
    m: Any,
    H: int,
    M: float,
    N: float,
    settings: QuadratureSettings | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Return the density-averaged local binomial prediction and its bookkeeping.

    The prediction is ``(1/N) int_{t_0}^{M+N} Binom(m; H, 1/log t) dt`` with
    ``t_0 = max(e, M)`` when ``M < e`` (otherwise ``t_0 = max(2, M)``, which
    is then at least ``e``).  The Bernoulli probability ``1/log t`` exceeds
    one for ``t < e``, so the stretch ``[max(2, M), e)`` cannot be used as a
    probability and is omitted rather than clipped.  Its length ``ell`` is
    reported, together with the bound ``ell / N`` on the omitted contribution
    (every probability mass function value lies in ``[0, 1]``, so the omitted
    part of the average is between ``0`` and ``ell / N`` for every ``m``).

    Parameters
    ----------
    m : int or array of int
        Prime counts (integers).
    H : int
        Interval length.
    M : int or float
        Left end of the range of starting points.
    N : int or float
        Number of starting points, positive.
    settings : QuadratureSettings, optional
        Quadrature tolerances.

    Returns
    -------
    tuple
        ``(values, info)`` where ``info`` has keys ``'omitted_length'``,
        ``'omitted_bound'`` (``= omitted_length / N``), ``'lower_used'``,
        ``'upper'``, ``'error'``, ``'neval'``, ``'converged'``.
    """
    from scipy.stats import binom as binom_dist

    m_arr = np.atleast_1d(np.asarray(m))
    if not np.all(np.equal(np.mod(m_arr, 1), 0)):
        raise ValueError("binomial_averaged is defined at integer m only")
    m_int = m_arr.astype(int)
    lower, upper = _range(M, N)
    lower_used = max(lower, math.e)
    omitted = max(0.0, min(upper, lower_used) - lower)

    def integrand(t: float) -> np.ndarray:
        p = 1.0 / math.log(t)
        return np.asarray(binom_dist.pmf(m_int, H, p), dtype=float)

    res = log_average(integrand, lower_used, upper, N, settings)
    info = {
        "omitted_length": omitted,
        "omitted_bound": omitted / N,
        "lower_used": lower_used,
        "upper": upper,
        "error": res.error,
        "neval": res.neval,
        "converged": res.converged,
        "settings": res.settings,
    }
    return res.value, info


#: The prediction models :func:`predict_all` knows, with their plot labels.
MODELS: dict[str, str] = {
    "F": r"$F(m; H, M, N)$",
    "F0": r"$F_0(m; H, M, N)$",
    "B_const": r"$\mathrm{Binom}(H, \mu/H)$",
    "Q_mu": r"$Q(m; \mu, H)$",
    "Q_lambda": r"$Q(m; \lambda, H)$",
    "B_avg": r"$\frac{1}{N}\int \mathrm{Binom}(m; H, 1/\log t)\,dt$",
}

#: The models computed by default (the density-averaged binomial costs a
#: second quadrature and is opt-in).
DEFAULT_MODELS: tuple[str, ...] = ("F", "F0", "B_const", "Q_mu", "Q_lambda")


def predict_all(
    H: int,
    M: float,
    N: float,
    m_max: int,
    models: tuple[str, ...] | list[str] = DEFAULT_MODELS,
    settings: QuadratureSettings | None = None,
) -> dict[str, Any]:
    """Evaluate every requested prediction at integer ``m = 0, ..., m_max`` for one ``(H, M, N)``.

    This is the single entry point the comparison, plotting and animation
    code all use, so that no two of them can evaluate a formula differently.
    Models whose parameters are invalid at ``(H, M, N)`` (see
    :func:`parameter_validity`) are returned as arrays of ``nan``, never
    silently extended.

    Parameters
    ----------
    H : int
        Interval length.
    M : int or float
        Left end of the range of starting points.
    N : int or float
        Number of starting points, positive.
    m_max : int
        Largest ``m`` evaluated.
    models : sequence of str, optional
        Any of the keys of :data:`MODELS`; ``'F'`` and ``'F0'`` are always
        computed together (one quadrature).
    settings : QuadratureSettings, optional
        Quadrature tolerances.

    Returns
    -------
    dict
        Keys ``'H'``, ``'M'``, ``'N'``, ``'m'`` (array), ``'mu'``,
        ``'lambda'``, ``'validity'``, one array per requested model,
        ``'correction'`` (``F - F_0``), ``'quadrature'`` (error estimate,
        evaluations, convergence, settings), ``'B_avg_info'`` when
        ``'B_avg'`` was requested, ``'formula_version'``.
    """
    unknown = [name for name in models if name not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    m = np.arange(0, m_max + 1)
    validity = parameter_validity(H, M, N)
    mu = averaged_parameter(H, M, N)
    lam = shifted_parameter(H, M, N)
    nan = np.full(len(m), math.nan)
    out: dict[str, Any] = {
        "H": H,
        "M": M,
        "N": N,
        "m": m,
        "mu": mu,
        "lambda": lam,
        "validity": validity,
        "formula_version": FORMULA_VERSION,
    }
    if "F" in models or "F0" in models:
        if validity["integrated"]["valid"]:
            res = integrated(m, H, M, N, settings)
            out["F"], out["F0"], out["correction"] = res.F, res.F0, res.correction
            out["quadrature"] = {
                "error": res.error,
                "neval": res.neval,
                "converged": res.converged,
                "lower": res.lower,
                "upper": res.upper,
                "settings": res.settings,
            }
        else:
            out["F"], out["F0"], out["correction"] = nan.copy(), nan.copy(), nan.copy()
            out["quadrature"] = {"error": math.nan, "neval": 0, "converged": False}
    if "B_const" in models:
        out["B_const"] = (
            binomial_constant(m, H, mu) if validity["binomial_constant"]["valid"] else nan.copy()
        )
    if "Q_mu" in models:
        out["Q_mu"] = local_corrected(m, mu, H) if validity["mu"]["valid"] else nan.copy()
    if "Q_lambda" in models:
        out["Q_lambda"] = local_corrected(m, lam, H) if validity["lambda"]["valid"] else nan.copy()
    if "B_avg" in models:
        if validity["binomial_averaged"]["valid"]:
            values, info = binomial_averaged(m, H, M, N, settings)
            out["B_avg"], out["B_avg_info"] = values, info
        else:
            out["B_avg"], out["B_avg_info"] = nan.copy(), {}
    return out


def parameter_validity(H: float, M: float, N: float) -> dict[str, dict[str, Any]]:
    """Report, model by model, whether the parameters at ``(H, M, N)`` are usable.

    The checks are structural, not statements about asymptotic accuracy:

    * ``'integrated'`` (``F`` and ``F_0``): the range ``[max(2, M), M + N]``
      must have positive length, i.e. ``M + N > max(2, M)``.
    * ``'lambda'`` (``Q`` at the shifted parameter): ``log N + c(M/N) > 0``;
      for ``M = 0`` this is ``N > e``.
    * ``'mu'`` (``Q`` at the averaged parameter): as ``'integrated'``, and
      ``mu > 0``.
    * ``'binomial_constant'``: ``0 <= mu / H <= 1``.
    * ``'binomial_averaged'``: the usable range ``[max(e, M, 2), M + N]`` must
      have positive length.

    Parameters
    ----------
    H, M, N : int or float
        Interval length, left end of the range, number of starting points.

    Returns
    -------
    dict
        ``{model: {'valid': bool, 'reason': str}}``.
    """
    lower, upper = _range(M, N)
    out: dict[str, dict[str, Any]] = {}
    ok = upper > lower
    out["integrated"] = {
        "valid": ok,
        "reason": "" if ok else f"empty integration range [{lower}, {upper}]",
    }
    lam = shifted_parameter(H, M, N) if N > 0 else math.nan
    ok = not math.isnan(lam)
    out["lambda"] = {
        "valid": ok,
        "reason": "" if ok else "log N + c(M/N) is not positive",
    }
    mu = averaged_parameter(H, M, N) if N > 0 else 0.0
    ok = upper > lower and mu > 0
    out["mu"] = {"valid": ok, "reason": "" if ok else "mu is not positive"}
    ok = upper > lower and 0 <= mu / H <= 1
    out["binomial_constant"] = {
        "valid": ok,
        "reason": "" if ok else f"mu/H = {mu / H:.4g} is not a probability",
    }
    ok = upper > max(lower, math.e)
    out["binomial_averaged"] = {
        "valid": ok,
        "reason": "" if ok else "no part of the range has 1/log t <= 1",
    }
    return out
