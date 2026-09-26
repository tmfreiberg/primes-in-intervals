"""Plotting and animating the distribution of primes in intervals.

Two families of figures live here.

**Checkpointed datasets** (:func:`plot_distribution_frame`,
:func:`animate_distribution`): for each checkpoint, a bar-and-dot histogram
of the empirical distribution with the manuscript's predictions marked at
integer ``m`` (and, optionally, smooth guide curves through them), summary
statistics in a text box, and the whole sequence of checkpoints strung into
an animation.  The predictions for the checkpoint ``c`` of a dataset with
lower bound ``A`` are those for the range of starting points ``(A, c]``, that
is ``M = A`` and ``N = c - A`` in the manuscript's notation (for a nested
interval ``(c[0], c[1]]``, ``M = c[0]`` and ``N = c[1] - c[0]``); the
former normalization by the density ``1/(log N - 1)`` at the midpoint of the
range is gone.

**The cumulative experiment** (:func:`plot_cumulative_frame`,
:func:`animate_cumulative`, :func:`plot_residuals`,
:func:`plot_discrepancies`, :func:`plot_means`): frames from
:func:`~primes_in_intervals.cumulative.build_frames`, with empirical-only
early frames, predictions that move with ``N``, residual panels, and the
discrepancy measures and means as functions of ``N``.

Conventions shared by both: the empirical distribution is drawn as magenta
bars (``#e0249a``) with red dots; every prediction is drawn as discrete
markers at integer ``m`` (the only points with a probabilistic meaning), each
model with its own colour and marker (see :data:`STYLE`), and a smooth
dashed guide through the markers can be added (it interpolates ``m!`` by the
gamma function and is a visual aid only).  Negative predicted values are
never hidden: the vertical axis extends below zero when needed and the
least value is annotated.  All predictions come from
:func:`~primes_in_intervals.predictions.predict_all`, so a plot cannot use a
formula the scoring code does not.

Matplotlib is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from primes_in_intervals.comparisons import _range_parameters
from primes_in_intervals.intervals import Dataset
from primes_in_intervals.predictions import (
    DEFAULT_MODELS,
    MODELS,
    binom_pmf,
    eta,
    integrated,
    local_corrected,
    predict_all,
)
from primes_in_intervals.quadrature import QuadratureSettings

__all__ = [
    "STYLE",
    "animate_cumulative",
    "animate_distribution",
    "distribution_axes_limits",
    "plot_cumulative_frame",
    "plot_discrepancies",
    "plot_distribution_frame",
    "plot_means",
    "plot_residuals",
    "save_figure",
    "save_gif",
    "save_mp4",
]

#: Colour and marker of each prediction model.
STYLE: dict[str, dict[str, str]] = {
    "F": {"color": "green", "marker": "s"},
    "F0": {"color": "blue", "marker": "^"},
    "B_const": {"color": "orange", "marker": "D"},
    "Q_mu": {"color": "purple", "marker": "v"},
    "Q_lambda": {"color": "saddlebrown", "marker": "<"},
    "B_avg": {"color": "teal", "marker": "x"},
}

_EMPIRICAL_BAR = "#e0249a"


def distribution_axes_limits(X: Dataset) -> tuple[list[int], float]:
    """Return the common horizontal axis and maximum height for a dataset.

    The horizontal axis is the key list of the final checkpoint's
    distribution (every checkpoint shares it, by
    :func:`~primes_in_intervals.intervals.zeros`), and the height is the
    largest relative frequency over all checkpoints, so an animation's axes
    can stay fixed across frames.

    Parameters
    ----------
    X : dict
        An analyzed dataset (with a ``'distribution'`` item).

    Returns
    -------
    tuple
        ``(hor_axis, y_max)``.
    """
    C = list(X["distribution"].keys())
    hor_axis = list(X["distribution"][C[-1]].keys())
    y_max = 0.0
    for c in C:
        for m in X["distribution"][c].keys():
            if y_max < X["distribution"][c][m]:
                y_max = X["distribution"][c][m]
    return hor_axis, y_max


def _guide(
    model: str, H: int, M: float, N: float, pred: dict[str, Any], x: np.ndarray
) -> np.ndarray | None:
    """Return the smooth guide curve of ``model`` on the real grid ``x``, or ``None``."""
    if model in ("F", "F0"):
        res = integrated(x, H, M, N, QuadratureSettings(epsabs=1e-10, epsrel=1e-8, limit=200))
        return res.F if model == "F" else res.F0
    if model == "B_const":
        p = pred["mu"] / H
        return binom_pmf(H, x, p) if 0 <= p <= 1 else None
    if model == "Q_mu":
        return local_corrected(x, pred["mu"], H)
    if model == "Q_lambda":
        lam = pred["lambda"]
        return local_corrected(x, lam, H) if not math.isnan(lam) else None
    return None  # the averaged binomial has no cheap real-m extension


def _draw_predictions(
    ax: Any,
    pred: dict[str, Any] | None,
    models: Sequence[str],
    m_axis: list[int] | np.ndarray,
    guides: bool,
    H: int,
    M: float,
    N: float,
    marker_size: float = 9,
) -> float:
    """Draw markers (and guides) for each valid model; return the least value drawn."""
    lowest = 0.0
    if pred is None:
        return lowest
    m_arr = np.asarray(m_axis)
    for model in models:
        values = pred.get(model)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)[: len(m_arr)]
        if np.any(np.isnan(values)):
            continue  # parameters invalid at this frame: nothing drawn
        style = STYLE[model]
        ax.plot(
            m_arr,
            values,
            linestyle="none",
            marker=style["marker"],
            markersize=marker_size,
            markerfacecolor="none",
            markeredgewidth=1.8,
            color=style["color"],
            zorder=4,
            label=MODELS[model],
        )
        lowest = min(lowest, float(values.min()))
        if guides:
            x = np.linspace(m_arr[0], m_arr[-1], 200)
            y = _guide(model, H, M, N, pred, x)
            if y is not None:
                ax.plot(x, y, "--", color=style["color"], linewidth=1.2, alpha=0.7, zorder=3.5)
    return lowest


def _annotate_negative(
    ax: Any, pred: dict[str, Any] | None, models: Sequence[str], m_top: int
) -> None:
    """Note the least value of any negative prediction on the axis."""
    if pred is None:
        return
    notes = []
    for model in models:
        values = pred.get(model)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)[: m_top + 1]
        if np.any(np.isnan(values)) or values.min() >= 0:
            continue
        j = int(np.argmin(values))
        label = MODELS[model].replace("(m; H, M, N)", "").replace("(m; \\mu, H)", "(\\mu)")
        notes.append(rf"min {label} $= {values[j]:.4f}$ at $m = {j}$")
    if notes:
        ax.text(
            0.99,
            0.02,
            "negative predicted values: " + "; ".join(notes),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )


def _default_overlay(X: Dataset, c: Any, H: int, pred: dict[str, Any] | None) -> str:
    """Build the statistics text box for a frame (the generic form).

    The random-variable line matches the interval type, the range line shows
    the manuscript's ``M`` and ``N`` for the checkpoint, and the statistics
    lines show ``mu``, ``lambda``, mean, variance, median, and mode(s).
    """
    mu = X["statistics"][c]["mean"]
    sigma = X["statistics"][c]["var"]
    med = X["statistics"][c]["med"]
    if med == int(med):
        med = int(med)
    modes = X["statistics"][c]["mode"]
    interval_type = X["header"]["interval_type"]
    M, N, _ = _range_parameters(X, c)
    if interval_type == "overlap":
        variable = r"$X = \pi(a + H) - \pi(a)$, " + r"$M < a \leq M + N$"
    elif interval_type == "disjoint":
        variable = r"$X = \pi(a + H) - \pi(a)$, " + r"$a = M + kH$, $1 \leq k \leq N/H$"
    else:  # prime_start
        variable = r"$X = \pi(p + H) - \pi(p)$, " + r"$M < p \leq M + N$, $p$ prime"
    lines = [
        variable,
        rf"$H = {H}$",
        rf"$M = {M}$",
        rf"$N = {N}$",
    ]
    if pred is not None:
        lines.append(rf"$\mu = {pred['mu']:.5f}$")
        if not math.isnan(pred["lambda"]):
            lines.append(rf"$\lambda = H/(\log N + c(M/N)) = {pred['lambda']:.5f}$")
    lines += [
        r"$\mathbb{E}[X] = $" + f"{mu:.5f}",
        r"$\mathrm{Var}(X) = $" + f"{sigma:.5f}",
        rf"median : ${med}$",
        rf"mode(s): ${modes}$",
    ]
    return "\n\n".join(lines)


def plot_distribution_frame(
    ax: Any,
    X: Dataset,
    c: Any,
    hor_axis: list[int] | None = None,
    y_max: float | None = None,
    models: Sequence[str] | None = None,
    guides: bool = True,
    overlay: str | Callable[..., str] | None = "auto",
    overlay_position: tuple[float, float] = (0.70, 0.15),
    note: str | None = None,
    x_pad: float = 0.5,
    ylim_decimals: int = 2,
    predictions: dict[str, Any] | None = None,
    settings: QuadratureSettings | None = None,
) -> dict[str, Any] | None:
    """Draw one checkpoint's distribution onto an existing axis.

    Renders the empirical distribution at checkpoint ``c`` as translucent
    bars with red dots, marks the requested predictions at integer ``m``
    (with dashed guides through them when ``guides`` is true), and adds the
    statistics text box.  The axis is cleared first, so this doubles as the
    animation's frame function.

    Parameters
    ----------
    ax : matplotlib axis
        The target axis (cleared before drawing).
    X : dict
        An analyzed dataset.
    c : int or tuple
        The checkpoint (or nested interval) to draw.
    hor_axis, y_max : list and float, optional
        The fixed axes from :func:`distribution_axes_limits`; computed from
        ``X`` when omitted.  Pass them explicitly when animating, so every
        frame shares the same axes.
    models : sequence of str, optional
        Prediction models to draw (keys of
        :data:`~primes_in_intervals.predictions.MODELS`).  The default draws
        ``F``, ``F0`` and the constant-density binomial for overlapping
        data, and only the binomial otherwise (the corrected prediction was
        derived for overlapping intervals).
    guides : bool, optional
        Draw smooth dashed guide curves through the markers (default True).
    overlay : str, callable, or None, optional
        ``'auto'`` (default) builds the generic statistics box; a callable is
        called as ``overlay(X, c, H, pred)`` and must return the text; any
        other string is used verbatim; ``None`` or ``'off'`` suppresses it.
    overlay_position : tuple, optional
        Axes-fraction position of the text box (default ``(0.70, 0.15)``).
    note : str, optional
        Extra text drawn at the top of the axes (for instance a reminder that
        ``F`` was derived for overlapping intervals, on prime-start data).
    x_pad : float, optional
        Horizontal padding beyond the first and last ``m`` (default 0.5).
    ylim_decimals : int, optional
        The vertical limit is ``y_max`` rounded up at this many decimals
        (default 2).
    predictions : dict, optional
        A precomputed :func:`~primes_in_intervals.predictions.predict_all`
        result for this checkpoint (for instance from a cache); computed
        here when omitted.
    settings : QuadratureSettings, optional
        Quadrature tolerances when predictions are computed here.

    Returns
    -------
    dict or None
        The predictions drawn (so callers can reuse or cache them).
    """
    ax.clear()

    H = X["header"]["interval_length"]
    interval_type = X["header"]["interval_type"]
    if hor_axis is None or y_max is None:
        hor_axis, y_max = distribution_axes_limits(X)
    if models is None:
        models = ("F", "F0", "B_const") if interval_type == "overlap" else ("B_const",)

    # The data and histogram
    ver_axis = list(X["distribution"][c].values())
    ax.bar(
        hor_axis,
        ver_axis,
        color=_EMPIRICAL_BAR,
        zorder=2.5,
        alpha=0.3,
        label=r"$\mathrm{Prob}(X = m)$",
    )
    ax.plot(hor_axis, ver_axis, "o", color="red", zorder=2.5)

    # Predictions for comparison, for the range (M, M + N] of this checkpoint
    M, N, _ = _range_parameters(X, c)
    pred = predictions
    if pred is None and models:
        pred = predict_all(H, M, N, max(hor_axis), models=tuple(models), settings=settings)
    lowest = _draw_predictions(ax, pred, models, hor_axis, guides, H, M, N)

    # Bounds for the plot, and horizontal axis tick marks.
    scale = 10**ylim_decimals
    top = np.ceil(scale * y_max) / scale
    bottom = 0.0 if lowest >= 0 else -np.ceil(scale * -lowest) / scale - 1 / scale
    ax.set(xlim=(hor_axis[0] - x_pad, hor_axis[-1] + x_pad), ylim=(bottom, top))
    if bottom < 0:
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    _annotate_negative(ax, pred, models, max(hor_axis))

    # Overlay information
    if note is not None:
        ax.text(
            0.25,
            0.90,
            note,
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.5),
            transform=ax.transAxes,
        )
    if overlay is not None and overlay != "off":
        if overlay == "auto":
            text = _default_overlay(X, c, H, pred)
        elif callable(overlay):
            text = overlay(X, c, H, pred)
        else:
            text = overlay  # a literal string, used as-is
        ax.text(
            overlay_position[0],
            overlay_position[1],
            text,
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.5),
            transform=ax.transAxes,
        )

    # Formatting/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r"$m$ (number of primes in an interval)")
    ax.set_ylabel("prop'n of intervals with" + r" $m$ " + "primes")
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else.
    ax.grid(True, zorder=0, alpha=0.7)
    return pred


def animate_distribution(
    X: Dataset,
    frames: list | None = None,
    interval: int = 100,
    figsize: tuple[float, float] = (22, 11),
    font_size: int = 22,
    suptitle: str = "Primes in intervals",
    **frame_kwargs: Any,
) -> tuple[Any, Any]:
    """Animate a dataset's distribution across its checkpoints.

    One frame per checkpoint, each drawn by :func:`plot_distribution_frame`
    with axes held fixed across the animation.  For an ordinary dataset the
    first (all-zero) checkpoint is skipped; for a nested dataset every
    interval is a frame, from the innermost outward.  The figure is left
    showing the final frame.  Predictions are computed once per checkpoint
    and reused when the frame is redrawn.

    Note that this updates ``matplotlib.rcParams`` (font size, white save
    background), as the original scripts did.

    Parameters
    ----------
    X : dict
        An analyzed dataset.
    frames : list, optional
        The checkpoints to animate; defaults as described above.
    interval : int, optional
        Delay between frames in milliseconds (default 100).
    figsize : tuple, optional
        Figure size in inches (default ``(22, 11)``).
    font_size : int, optional
        Global font size (default 22).
    suptitle : str, optional
        Figure title (default ``'Primes in intervals'``).
    **frame_kwargs
        Passed through to :func:`plot_distribution_frame` (models, guides,
        overlay, padding, and so on).

    Returns
    -------
    tuple
        ``(fig, anim)``: the matplotlib figure and the ``FuncAnimation``.
        Keep a reference to ``anim`` until it has been saved or displayed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams.update({"font.size": font_size})

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(suptitle)

    hor_axis, y_max = distribution_axes_limits(X)
    C = list(X["distribution"].keys())
    if frames is None:
        if "nested_interval_data" in X.keys():
            frames = C
        else:
            frames = C[1:]
    computed: dict[Any, dict[str, Any] | None] = {}

    def draw(c: Any) -> None:
        computed[c] = plot_distribution_frame(
            ax,
            X,
            c,
            hor_axis=hor_axis,
            y_max=y_max,
            predictions=computed.get(c),
            **frame_kwargs,
        )

    anim = FuncAnimation(
        fig,
        # Some matplotlib versions type the frame function more narrowly than
        # the callback we pass; the extra 'unused-ignore' code keeps this
        # comment harmless on versions where no error is raised at all.
        draw,  # type: ignore[arg-type, unused-ignore]
        frames=frames,
        interval=interval,
        blit=False,
        repeat=False,
    )

    # This is supposed to remedy the blurry axis ticks/labels.
    plt.rcParams["savefig.facecolor"] = "white"

    draw(frames[-1])
    return fig, anim


# --------------------------------------------------------------------------
# The cumulative experiment
# --------------------------------------------------------------------------


def _cumulative_overlay(frame: Any, H: int, models: Sequence[str]) -> str:
    """Text box for a cumulative frame: range, parameters, statistics, scores."""
    lines = [
        r"$X(n) = \pi(n + H) - \pi(n)$, $1 \leq n \leq N$",
        rf"$H = {H}$, $N = {frame.N}$",
        r"$\mathbb{E}[X] = $"
        + f"{frame.mean:.5f}"
        + r", $\mathrm{Var}(X) = $"
        + f"{frame.var:.5f}",
        rf"median ${frame.median:g}$, mode(s) ${frame.mode}$",
    ]
    pred = frame.predictions
    if pred is not None:
        lines.append(
            rf"$\mu = {pred['mu']:.5f}$"
            + (
                rf", $\lambda = H/(\log N - 1) = {pred['lambda']:.5f}$"
                if not math.isnan(pred["lambda"])
                else ""
            )
        )
        sc = frame.scores
        if sc is not None:
            lo, hi = None, None
            parts = []
            for model in models:
                s = sc["central"].get(model)
                if s is None:
                    continue
                lo, hi = s["m_lo"], s["m_hi"]
                short = (
                    MODELS[model].split("(")[0].strip("$")
                    if model not in ("B_const", "B_avg")
                    else (r"\mathrm{Binom}" if model == "B_const" else r"\overline{\mathrm{Binom}}")
                )
                parts.append(rf"$E_1({short}) = {s['E1']:.4f}$")
            if parts:
                lines.append(", ".join(parts) + rf" on $m = {lo}..{hi}$")
    else:
        lines.append("empirical frequencies only (predictions from the overlay start)")
    return "\n".join(lines)


def plot_cumulative_frame(
    ax: Any,
    frame: Any,
    H: int,
    models: Sequence[str] = DEFAULT_MODELS[:2],
    y_max: float | None = None,
    guides: bool = True,
    overlay: bool = True,
    overlay_position: tuple[float, float] = (0.99, 0.97),
    y_min: float | None = None,
    x_pad: float = 0.5,
    ylim_decimals: int = 2,
) -> None:
    """Draw one frame of the cumulative experiment onto an existing axis.

    Parameters
    ----------
    ax : matplotlib axis
        Cleared before drawing.
    frame : Frame
        From :func:`~primes_in_intervals.cumulative.build_frames`.
    H : int
        Interval length.
    models : sequence of str, optional
        Models to draw (default ``F`` and ``F0``); models without
        predictions at this frame are skipped.
    y_max : float, optional
        Top of the vertical axis (default: the frame's own maximum, rounded
        up).  Pass a fixed value when animating.
    guides : bool, optional
        Smooth dashed guides through the markers (default True).
    overlay : bool, optional
        Draw the text box (default True).
    overlay_position : tuple, optional
        Axes-fraction position of the text box.
    x_pad : float, optional
        Horizontal padding beyond ``m = 0`` and ``m = m_axis``.
    ylim_decimals : int, optional
        Rounding of the vertical limits.
    y_min : float, optional
        Extend the vertical axis down to at least this value (used by
        :func:`animate_cumulative` to hold the bottom of the axis fixed).
    """
    ax.clear()
    m_axis = list(range(frame.m_axis + 1))
    ax.bar(m_axis, frame.P, color=_EMPIRICAL_BAR, zorder=2.5, alpha=0.3, label=r"$P(m; H, 0, N)$")
    ax.plot(m_axis, frame.P, "o", color="red", zorder=2.5)
    lowest = _draw_predictions(ax, frame.predictions, models, m_axis, guides, H, 0, frame.N)
    scale = 10**ylim_decimals
    top = np.ceil(scale * (y_max if y_max is not None else float(frame.P.max()))) / scale
    if y_min is not None:
        lowest = min(lowest, y_min)
    bottom = 0.0 if lowest >= 0 else -np.ceil(scale * -lowest) / scale - 1 / scale
    ax.set(xlim=(-x_pad, frame.m_axis + x_pad), ylim=(bottom, top))
    if bottom < 0:
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    _annotate_negative(ax, frame.predictions, models, frame.m_axis)
    if overlay:
        ax.text(
            overlay_position[0],
            overlay_position[1],
            _cumulative_overlay(frame, H, models),
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.6),
            transform=ax.transAxes,
            fontsize="small",
            va="top",
            ha="right",
        )
    ax.set_xticks(m_axis)
    ax.set_xlabel(r"$m$ (number of primes in $(n, n + H]$)")
    ax.set_ylabel(r"proportion of $1 \leq n \leq N$ with $m$ primes")
    ax.legend(loc=2, ncol=1, framealpha=0.5)
    ax.grid(True, zorder=0, alpha=0.7)


def animate_cumulative(
    frames: list[Any],
    H: int,
    N_min: int | None = None,
    y_max: float | None = None,
    interval: int = 120,
    figsize: tuple[float, float] = (16, 9),
    font_size: int = 14,
    suptitle: str | None = None,
    **frame_kwargs: Any,
) -> tuple[Any, Any]:
    """Animate the cumulative experiment: one frame per checkpoint ``N``.

    Two presentations are useful.  The **full history** (``N_min=None``)
    starts at ``N = 1`` with the vertical axis running to 1, since a single
    interval puts all the mass in one bin.  The **later-range comparison**
    (``N_min`` set) shows only the frames with ``N >= N_min`` on a vertical
    axis fitted to them.  Either way every frame shows the cumulative
    frequencies over the full range ``1 <= n <= N``; nothing is dropped from
    the starting-point range.

    Parameters
    ----------
    frames : list of Frame
        From :func:`~primes_in_intervals.cumulative.build_frames`.
    H : int
        Interval length.
    N_min : int, optional
        First checkpoint shown.
    y_max : float, optional
        Fixed top of the vertical axis (default: the maximum over the frames
        shown, rounded up).  The bottom is fixed too, at the least value of
        any drawn prediction over the frames shown (zero if none is
        negative).
    interval : int, optional
        Delay between frames in milliseconds.
    figsize, font_size : tuple and int, optional
        Figure size and font size.
    suptitle : str, optional
        Title (default states ``H`` and the range).
    **frame_kwargs
        Passed to :func:`plot_cumulative_frame` (``models``, ``guides``,
        ``overlay``, ...).

    Returns
    -------
    tuple
        ``(fig, anim)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    shown = [f for f in frames if N_min is None or f.N >= N_min]
    if not shown:
        raise ValueError("no frames to animate")
    if y_max is None:
        y_max = max(float(f.P.max()) for f in shown)
    # Hold the bottom of the axis fixed as well: the least value of any drawn
    # prediction over all the frames shown (zero if none is negative).
    models = frame_kwargs.get("models", DEFAULT_MODELS[:2])
    y_min = 0.0
    for f in shown:
        for model in models:
            values = f.prediction(model)
            if values is not None and not np.any(np.isnan(values)):
                y_min = min(y_min, float(values.min()))
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        suptitle
        or rf"Cumulative prime-count frequencies over $1 \leq n \leq N$, $H = {H}$: "
        "evolution with $N$ of the frequencies and their predictions"
    )

    def draw(frame: Any) -> None:
        plot_cumulative_frame(ax, frame, H, y_max=y_max, y_min=y_min, **frame_kwargs)

    anim = FuncAnimation(
        fig,
        draw,  # type: ignore[arg-type, unused-ignore]
        frames=shown,
        interval=interval,
        blit=False,
        repeat=False,
    )
    plt.rcParams["savefig.facecolor"] = "white"
    draw(shown[-1])
    return fig, anim


def plot_residuals(
    frame: Any,
    H: int,
    ax_pair: tuple[Any, Any] | None = None,
    scale_by_eta: bool = False,
) -> Any:
    """Draw the residual panels for one frame.

    Left: ``P_m - F_0(m)`` as dots, with the predicted correction
    ``F(m) - F_0(m)`` superimposed as markers (so the eye can judge whether
    the correction has the right sign and size bin by bin).  Right:
    ``P_m - F(m)``, the residual after correction.  With ``scale_by_eta``
    both panels are divided by ``eta(H)``, the size of the correction; no
    division by the correction polynomial is ever done (it vanishes).

    Parameters
    ----------
    frame : Frame
        A frame with predictions.
    H : int
        Interval length.
    ax_pair : tuple of axes, optional
        Two axes to draw on; a new figure is created when omitted.
    scale_by_eta : bool, optional
        Divide the residuals and the correction by ``eta(H)``.

    Returns
    -------
    matplotlib figure
        The figure the panels live on.
    """
    import matplotlib.pyplot as plt

    if frame.predictions is None:
        raise ValueError(f"frame N = {frame.N} has no predictions")
    if ax_pair is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax1, ax2 = ax_pair
        fig = ax1.figure
    m = np.arange(frame.m_axis + 1)
    F0 = frame.prediction("F0")
    F = frame.prediction("F")
    corr = np.asarray(frame.predictions["correction"], dtype=float)[: frame.m_axis + 1]
    scale = eta(H) if scale_by_eta else 1.0
    unit = r"/\eta(H)" if scale_by_eta else ""
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.plot(m, (frame.P - F0) / scale, "o", color="red", label=rf"$(P_m - F_0(m)){unit}$")
    ax1.plot(
        m,
        corr / scale,
        linestyle="none",
        marker="s",
        markerfacecolor="none",
        markeredgewidth=1.8,
        color="green",
        label=rf"$(F(m) - F_0(m)){unit}$ (predicted correction)",
    )
    ax1.set_xlabel(r"$m$")
    ax1.set_title(rf"residual from $F_0$ and the predicted correction, $N = {frame.N}$")
    ax1.legend(framealpha=0.5)
    ax1.grid(True, alpha=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.plot(m, (frame.P - F) / scale, "s", color="green", label=rf"$(P_m - F(m)){unit}$")
    ax2.set_xlabel(r"$m$")
    ax2.set_title(rf"residual from $F$, $N = {frame.N}$")
    ax2.legend(framealpha=0.5)
    ax2.grid(True, alpha=0.5)
    for ax in (ax1, ax2):
        ax.set_xticks(m)
    fig.suptitle(rf"$H = {H}$, starting points $1 \leq n \leq N$, $\eta(H) = {eta(H):.5f}$")
    return fig


def plot_discrepancies(
    summary: Any,
    models: Sequence[str] = ("F", "F0", "B_const"),
    which: str = "central",
    measures: Sequence[str] = ("E1", "E2", "Einf"),
    ax_row: Sequence[Any] | None = None,
) -> Any:
    """Plot the discrepancy measures against ``N`` (log scale) for several models.

    Parameters
    ----------
    summary : pandas.DataFrame
        The per-``N`` table from
        :func:`~primes_in_intervals.cumulative.frames_to_tables`.
    models : sequence of str, optional
        Models to show.
    which : str, optional
        ``'central'`` (default) or ``'global'``.
    measures : sequence of str, optional
        Which of ``E1``, ``E2``, ``Einf`` to draw (one panel each).
    ax_row : sequence of axes, optional
        Axes to draw on (one per measure); a new figure when omitted.

    Returns
    -------
    matplotlib figure
    """
    import matplotlib.pyplot as plt

    if ax_row is None:
        fig, axes = plt.subplots(1, len(measures), figsize=(5.5 * len(measures), 4.5))
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(np.asarray(ax_row, dtype=object))
        fig = axes[0].figure
    scored = summary.dropna(subset=[f"{which}_E1_F0"]) if f"{which}_E1_F0" in summary else summary
    for ax, key in zip(axes, measures, strict=False):
        for model in models:
            col = f"{which}_{key}_{model}"
            if col not in scored:
                continue
            style = STYLE[model]
            ax.plot(
                scored["N"],
                scored[col],
                marker=style["marker"],
                markersize=4,
                color=style["color"],
                label=MODELS[model],
                linewidth=1,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$N$")
        pretty = {"E1": r"$E_1$", "E2": r"$E_2$", "Einf": r"$E_\infty$"}[key]
        ax.set_ylabel(pretty)
        ax.set_title(f"{pretty} ({which} range)")
        ax.grid(True, alpha=0.5)
        ax.legend(framealpha=0.5, fontsize="small")
    range_text = ""
    if which == "central" and "central_m_lo" in scored.columns and len(scored):
        lo, hi = int(scored["central_m_lo"].iloc[-1]), int(scored["central_m_hi"].iloc[-1])
        range_text = f", summed over $m = {lo}, \\ldots, {hi}$"
    elif which == "global" and "m_T" in scored.columns and len(scored):
        m_T = int(scored["m_T"].max())
        range_text = f", summed over $m = 0, \\ldots, m_T$ with $m_T \\leq {m_T}$"
    fig.suptitle(
        rf"discrepancy measures against $N$, $H = {int(summary['H'].iloc[0])}$"
        + range_text
    )
    return fig


def plot_means(summary: Any, ax: Any = None) -> Any:
    """Plot the empirical mean alongside ``mu`` and ``lambda`` against ``N``.

    Parameters
    ----------
    summary : pandas.DataFrame
        The per-``N`` table from
        :func:`~primes_in_intervals.cumulative.frames_to_tables`.
    ax : matplotlib axis, optional
        Axis to draw on; a new figure when omitted.

    Returns
    -------
    matplotlib figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    ax.plot(
        summary["N"],
        summary["mean"],
        "o-",
        color="red",
        markersize=3,
        linewidth=1,
        label=r"empirical mean $\mathbb{E}[X]$",
    )
    if "mu" in summary:
        s = summary.dropna(subset=["mu"])
        ax.plot(
            s["N"],
            s["mu"],
            "s--",
            color="green",
            markersize=3,
            linewidth=1,
            label=r"$\mu = (H/N)\int_2^N dt/\log t$",
        )
        ax.plot(
            s["N"],
            s["lambda"],
            "^:",
            color="blue",
            markersize=3,
            linewidth=1,
            label=r"$\lambda = H/(\log N - 1)$",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("mean number of primes")
    ax.set_title(rf"$H = {int(summary['H'].iloc[0])}$: the mean decreases as $N$ grows")
    ax.grid(True, alpha=0.5)
    ax.legend(framealpha=0.5)
    return fig


def save_figure(
    fig: Any, stem: str | Path, formats: Sequence[str] = ("pdf", "png"), dpi: int = 150
) -> list[Path]:
    """Save a figure under ``stem`` in each format (``stem.pdf``, ``stem.png``, ...).

    Parameters
    ----------
    fig : matplotlib figure
    stem : str or Path
        Path without extension.
    formats : sequence of str, optional
        File formats (default PDF for the paper and PNG for viewing).
    dpi : int, optional
        Resolution for raster formats.

    Returns
    -------
    list of Path
        The files written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append(path)
    return written


def save_gif(anim: Any, path: str, fps: int = 10, dpi: int = 100) -> None:
    """Save an animation as a GIF with matplotlib's ``PillowWriter``.

    Parameters
    ----------
    anim : matplotlib animation
        As returned by :func:`animate_distribution` or :func:`animate_cumulative`.
    path : str
        Output filename (conventionally ending in ``.gif``).
    fps : int, optional
        Frames per second (default 10).
    dpi : int, optional
        Resolution (default 100).
    """
    from matplotlib.animation import PillowWriter

    anim.save(path, dpi=dpi, writer=PillowWriter(fps=fps))


def save_mp4(anim: Any, path: str, fps: int = 10, dpi: int = 100) -> None:
    """Save an animation as an MP4 with matplotlib's ``FFMpegWriter``.

    Requires ``ffmpeg`` to be installed and on the path.

    Parameters
    ----------
    anim : matplotlib animation
        As returned by :func:`animate_distribution` or :func:`animate_cumulative`.
    path : str
        Output filename (conventionally ending in ``.mp4``).
    fps : int, optional
        Frames per second (default 10).
    dpi : int, optional
        Resolution (default 100).
    """
    from matplotlib.animation import FFMpegWriter

    anim.save(path, dpi=dpi, writer=FFMpegWriter(fps=fps))
