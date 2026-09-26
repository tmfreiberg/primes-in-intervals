"""Regenerate the animated figures of the worked-examples chapter.

Run from the repository root:

    python scripts/book_animations.py

It writes three GIFs into ``images/``: the nested animations about ``e^17``
(overlapping and prime-starting intervals, ``H = 76``) and about ``e^21``
(overlapping intervals, ``H = 60``, from the database).  The fourth GIF of
that chapter, ``li20_H_100.gif``, is not regenerated here.  Each frame shows
the predictions for the nested window it depicts, as drawn by
:func:`primes_in_intervals.plot_distribution_frame`.  Rendering takes a few
minutes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import primes_in_intervals as pii  # noqa: E402


def centred_overlay(x0, x0_label, variable=r"$X = \pi(a + H) - \pi(a)$", start="a", extra=None):
    """Return an overlay function for windows centred at ``x0`` (as in the book)."""

    def overlay(X, c, H, pred):
        upper = c[1] if isinstance(c, tuple) else c
        s = X["statistics"][c]
        med = int(s["med"]) if s["med"] == int(s["med"]) else s["med"]
        lines = [
            variable,
            rf"$x_0 - K < {start} \leq x_0 + K$",
            rf"$H = {H}$, $x_0 = {x0_label}$, $K = {upper - x0}$",
        ]
        if extra is not None:
            lines.append(extra(X, c))
        lines += [
            rf"$\mu = {pred['mu']:.5f}$",
            r"$\mathbb{E}[X] = $" + f"{s['mean']:.5f}",
            r"$\mathrm{Var}(X) = $" + f"{s['var']:.5f}",
            rf"median : ${med}$",
            rf"mode(s): ${s['mode']}$",
        ]
        return "\n\n".join(lines)

    return overlay


def write(X, path, **frame_kwargs):
    """Animate a nested dataset and save it as a GIF (22 by 11 inches at 72 dpi)."""
    fig, anim = pii.animate_distribution(X, **frame_kwargs)
    pii.save_gif(anim, path, fps=10, dpi=72)
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    """Build the three animations."""
    x17 = int(np.exp(17))
    C17 = list(range(x17 - 10**4, x17 + 10**4 + 1, 10**2))

    X = pii.nest(pii.intervals(list(C17), 76, "overlap"))
    pii.analyze(X)
    write(
        X,
        "images/EXP17_76_NESTanim.gif",
        ylim_decimals=3,
        overlay=centred_overlay(x17, "[e^{17}]"),
        overlay_position=(0.70, 0.15),
    )

    X = pii.nest(pii.intervals(list(C17), 76, "prime_start"))
    pii.analyze(X)

    def primes_in_window(X, c):
        return rf"$\pi(x_0 + K) - \pi(x_0 - K) = {sum(X['nested_interval_data'][c].values())}$"

    write(
        X,
        "images/PSEXP17_76_NESTanim.gif",
        models=("F", "F0", "B_const"),
        note="NB: $F$ and $F_0$ might not be applicable\nin this case, without modification.",
        overlay=centred_overlay(
            x17,
            "[e^{17}]",
            variable=r"$X = \pi(p + H) - \pi(p)$, $p$ prime",
            start="p",
            extra=primes_in_window,
        ),
        overlay_position=(0.70, 0.15),
    )

    x21 = int(np.exp(21))
    found = pii.retrieve(60)
    found = found if isinstance(found, list) else [found]
    lower = x21 - 100 * 10**3
    X = pii.nest(next(d for d in found if d["header"]["lower_bound"] == lower))
    pii.analyze(X)
    write(
        X,
        "images/N_exp21_H_60.gif",
        ylim_decimals=3,
        overlay=centred_overlay(x21, "[e^{21}]"),
        overlay_position=(0.70, 0.15),
    )


if __name__ == "__main__":
    main()
