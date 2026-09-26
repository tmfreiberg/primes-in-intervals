"""Smoke tests for the plotting layer (Agg backend, no display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import primes_in_intervals as pii  # noqa: E402


class TestAxesLimits:
    def test_common_axis_and_height(self, analyzed_overlap):
        hor_axis, y_max = pii.distribution_axes_limits(analyzed_overlap)
        C = list(analyzed_overlap["distribution"].keys())
        assert hor_axis == list(analyzed_overlap["distribution"][C[-1]].keys())
        top = max(
            v
            for c in C
            for v in analyzed_overlap["distribution"][c].values()
        )
        assert y_max == top


class TestFrame:
    def test_flat_frame_renders(self, analyzed_overlap):
        fig, ax = plt.subplots()
        C = list(analyzed_overlap["distribution"].keys())
        pii.plot_distribution_frame(ax, analyzed_overlap, C[-1])
        assert ax.get_legend() is not None
        assert ax.get_xlabel() == r"$m$ (number of primes in an interval)"
        # bars + dots + at least the two default curves for overlap data
        assert len(ax.lines) >= 3
        plt.close(fig)

    def test_nested_frame_with_all_curves_and_note(self, nested_overlap):
        fig, ax = plt.subplots()
        keys = list(nested_overlap["distribution"].keys())
        pred = pii.plot_distribution_frame(
            ax,
            nested_overlap,
            keys[0],
            models=("F", "F0", "B_const", "Q_mu", "Q_lambda"),
            note="NB: a reminder",
            ylim_decimals=3,
        )
        labels = [line.get_label() for line in ax.lines]
        for model in ("F", "F0", "B_const", "Q_mu", "Q_lambda"):
            assert pii.MODELS[model] in labels
        # the predictions drawn are those of the nested range (c[0], c[1]]
        assert pred["M"] == keys[0][0] and pred["N"] == keys[0][1] - keys[0][0]
        plt.close(fig)

    def test_negative_values_extend_the_axis(self, analyzed_overlap):
        fig, ax = plt.subplots()
        C = list(analyzed_overlap["distribution"].keys())
        pred = pii.plot_distribution_frame(ax, analyzed_overlap, C[-1], models=("F",))
        F = pred["F"][: max(analyzed_overlap["distribution"][C[-1]]) + 1]
        if F.min() < 0:
            assert ax.get_ylim()[0] < 0
            texts = [t.get_text() for t in ax.texts]
            assert any("negative predicted values" in t for t in texts)
        plt.close(fig)

    def test_frei_suppressed_off_overlap(self, prime_start_dataset):
        import copy

        ds = copy.deepcopy(prime_start_dataset)
        pii.analyze(ds)
        fig, ax = plt.subplots()
        C = list(ds["distribution"].keys())
        pii.plot_distribution_frame(ax, ds, C[-1])
        labels = [line.get_label() for line in ax.lines]
        assert pii.MODELS["F"] not in labels
        assert pii.MODELS["B_const"] in labels
        plt.close(fig)


class TestAnimation:
    def test_animate_and_save_gif(self, nested_overlap, tmp_path):
        keys = list(nested_overlap["distribution"].keys())
        fig, anim = pii.animate_distribution(nested_overlap, frames=keys[:3])
        out = tmp_path / "anim.gif"
        pii.save_gif(anim, str(out), fps=5, dpi=30)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore:Animation was deleted:UserWarning")
    def test_default_frames_skip_trivial_checkpoint(self, analyzed_overlap):
        fig, anim = pii.animate_distribution(analyzed_overlap)
        C = list(analyzed_overlap["distribution"].keys())
        assert list(anim._iter_gen()) == C[1:]  # noqa: SLF001 - matplotlib internal, test only
        plt.close(fig)
