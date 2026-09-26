"""Tests for the command-line interface.

Each test invokes :func:`primes_in_intervals.cli.main` with an argv list and
inspects stdout/stderr through capsys, the exit code, and any files written.
A session-scoped fixture prepares a small database and JSON files once.
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import pytest

import primes_in_intervals as pii
from primes_in_intervals import cli
from primes_in_intervals.serialize import read_dataset_json

# A small, fast configuration reused across the CLI tests.
RANGE_ARGS = ["--range", "exp(13)-2000", "exp(13)+2000", "400"]
H_ARGS = ["-H", "30"]


@pytest.fixture(scope="session")
def cli_workspace(tmp_path_factory):
    """Create a database and flat/nested JSON files via the CLI itself."""
    root = tmp_path_factory.mktemp("cli")
    db = root / "db"
    flat = root / "flat.json"
    nested = root / "nested.json"
    assert (
        cli.main(
            ["intervals", *RANGE_ARGS, *H_ARGS, "--save", "--db", str(db), "--json", str(flat)]
        )
        == 0
    )
    assert cli.main(["nest", "--from-json", str(flat), "--json", str(nested)]) == 0
    return {"db": db, "flat": flat, "nested": nested}


class TestNumberParser:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("24154952", 24154952),
            ("1_000_000", 1_000_000),
            ("2e6", 2_000_000),
            ("10**7", 10**7),
            ("exp(17)", 24154952),
            ("exp(17)-10**4", 24144952),
            ("exp(17)+10**4", 24164952),
            ("sqrt(49)", 7),
            ("-5", -5),
            ("(2+3)*4", 20),
            ("7//2", 3),
        ],
    )
    def test_accepted(self, text, value):
        assert cli.parse_int(text) == value

    @pytest.mark.parametrize(
        "text",
        [
            "__import__('os')",
            "x",
            "os.system",
            "exp(1, 2)",
            "[1,2]",
            "lambda: 0",
            "1 if 2 else 3",
        ],
    )
    def test_rejected(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_int(text)

    def test_int_list(self):
        assert cli.parse_int_list("0, 10**2, 2*10**2") == [0, 100, 200]

    def test_checkpoint_forms(self):
        assert cli.parse_checkpoint("last") == "last"
        assert cli.parse_checkpoint("2e2") == 200
        assert cli.parse_checkpoint("100,300") == (100, 300)
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_checkpoint("1,2,3")


class TestSieveCommands:
    def test_primes(self, capsys):
        assert cli.main(["primes", "5"]) == 0
        assert capsys.readouterr().out.split() == ["2", "3", "5", "7", "11"]

    def test_next_prime_and_prime_pi(self, capsys):
        assert cli.main(["next-prime", "100"]) == 0
        assert capsys.readouterr().out.strip() == "101"
        assert cli.main(["prime-pi", "1", "100"]) == 0
        assert capsys.readouterr().out.strip() == "25"


class TestCounterCommands:
    def test_overlap_table(self, capsys):
        assert cli.main(["overlap", "0", "5", "5"]) == 0
        out = capsys.readouterr().out
        assert "m  h(m)" in out
        assert "1  1" in out and "2  2" in out and "3  2" in out

    def test_prime_start_matches_readme(self, capsys):
        assert cli.main(["prime-start", "10", "20", "20"]) == 0
        out = capsys.readouterr().out
        assert "4  1" in out and "5  2" in out and "6  1" in out

    def test_json_and_csv_formats(self, capsys):
        assert cli.main(["disjoint", "0", "100", "10", "--format", "json"]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert sum(parsed.values()) == 10
        assert cli.main(["disjoint", "0", "100", "10", "--format", "csv"]) == 0
        assert capsys.readouterr().out.splitlines()[0] == "m,g(m)"

    def test_any_intervals_equals_prime_start(self, capsys):
        assert cli.main(["any-intervals", "0", "300", "20", "--format", "json"]) == 0
        via_any = json.loads(capsys.readouterr().out)
        assert cli.main(["prime-start", "0", "300", "20", "--format", "json"]) == 0
        via_ps = json.loads(capsys.readouterr().out)
        assert via_any == via_ps

    def test_overlap_extension_endpoints(self, capsys):
        assert cli.main(["overlap-extension", "0", "20", "5", "--m", "0,3"]) == 0
        out = capsys.readouterr().out
        assert "a with exactly 0 primes in (a, a+5]:" in out
        assert "a with exactly 3 primes in (a, a+5]:" in out

    def test_underscore_alias(self, capsys):
        assert cli.main(["prime_start", "10", "20", "20"]) == 0
        assert "6  1" in capsys.readouterr().out


class TestIntervalsCommand:
    def test_default_prints_display_with_caption(self, capsys):
        argv = ["intervals", "--checkpoints", "0,500,1000", "-H", "25", "--type", "disjoint"]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "Interval type: disjoint." in out
        assert "prime_tally" in out

    def test_range_is_inclusive(self, tmp_path, capsys):
        out_json = tmp_path / "ds.json"
        argv = ["intervals", "--range", "0", "1000", "500", "-H", "25", "--json", str(out_json)]
        assert cli.main(argv) == 0
        ds = read_dataset_json(out_json)
        assert list(ds["data"].keys()) == [0, 500, 1000]

    def test_save_writes_rows(self, cli_workspace):
        conn = sqlite3.connect(cli_workspace["db"])
        (count,) = conn.execute("SELECT COUNT(*) FROM overlap_raw").fetchone()
        conn.close()
        assert count == 10  # 11 checkpoints, one row per non-initial one

    def test_type_alias_presets(self, capsys):
        argv = ["disjoint-cp", "--checkpoints", "0,500,1000", "-H", "25"]
        assert cli.main(argv) == 0
        assert "Interval type: disjoint." in capsys.readouterr().out

    def test_json_to_stdout_is_clean(self, capsys):
        argv = ["intervals", "--checkpoints", "0,300,600", "-H", "20", "--json"]
        assert cli.main(argv) == 0
        captured = capsys.readouterr()
        ds = json.loads(captured.out)
        assert ds["header"]["interval_type"] == "overlap"


class TestRetrieveCommand:
    def test_summary_and_display(self, cli_workspace, capsys):
        argv = ["retrieve", "30", "--db", str(cli_workspace["db"])]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "Found 1 dataset" in out
        assert "Interval type: overlapping." in out

    def test_json_stdout_summary_on_stderr(self, cli_workspace, capsys):
        argv = ["retrieve", "30", "--db", str(cli_workspace["db"]), "--json"]
        assert cli.main(argv) == 0
        captured = capsys.readouterr()
        ds = json.loads(captured.out)  # would fail if the summary leaked in
        assert ds["header"]["interval_length"] == 30
        assert "Found 1 dataset" in captured.err

    def test_empty_retrieval_fails(self, cli_workspace, capsys):
        argv = ["retrieve", "999", "--db", str(cli_workspace["db"])]
        assert cli.main(argv) == 1
        assert "Found 0 datasets" in capsys.readouterr().out

    def test_multiple_datasets_need_index(self, tmp_path, capsys):
        db = tmp_path / "two"
        for start in ["0", "5000"]:
            argv = [
                "intervals", "--range", start, f"{start}+1000", "500", "-H", "20",
                "--save", "--db", str(db), "--json", str(tmp_path / "sink.json"),
            ]
            assert cli.main(argv) == 0
        # Summaries alone: fine without an index.
        assert cli.main(["retrieve", "20", "--db", str(db)]) == 0
        assert "Found 2 datasets" in capsys.readouterr().out
        # Asking for more without choosing: a clear failure.
        assert cli.main(["retrieve", "20", "--db", str(db), "--display"]) == 1
        assert "select one with --index" in capsys.readouterr().err
        # Choosing works.
        assert cli.main(["retrieve", "20", "--db", str(db), "--index", "1", "--display"]) == 0
        assert "Lower bound: 5000." in capsys.readouterr().out

    def test_pipeline_one_shot(self, cli_workspace, capsys):
        argv = [
            "retrieve", "30", "--db", str(cli_workspace["db"]),
            "--nest", "--analyze", "--compare", "--winners",
            "--display", "--view", "winners",
        ]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "B sq error" in out and "most wins" in out


class TestTransformCommands:
    def test_stepwise_equals_one_shot(self, cli_workspace, tmp_path, capsys):
        # Stepwise through JSON files...
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        c = tmp_path / "c.json"
        nested = str(cli_workspace["nested"])
        assert cli.main(["analyze", "--from-json", nested, "--json", str(a)]) == 0
        assert cli.main(["compare", "--from-json", str(a), "--json", str(b)]) == 0
        assert cli.main(["winners", "--from-json", str(b), "--json", str(c)]) == 0
        stepwise = read_dataset_json(c)
        # ...must agree with the one-shot pipeline.
        one_shot = tmp_path / "oneshot.json"
        argv = [
            "retrieve", "30", "--db", str(cli_workspace["db"]),
            "--nest", "--analyze", "--compare", "--winners", "--json", str(one_shot),
        ]
        assert cli.main(argv) == 0
        capsys.readouterr()
        combined = read_dataset_json(one_shot)
        assert stepwise["nested_interval_data"] == combined["nested_interval_data"]
        assert stepwise["winners"] == combined["winners"]

    def test_extract_narrow_and_filter(self, cli_workspace, capsys):
        base = read_dataset_json(cli_workspace["flat"])
        C = list(base["data"].keys())
        argv = [
            "extract", "--from-json", str(cli_workspace["flat"]),
            "--narrow", str(C[2]), str(C[-2]),
        ]
        assert cli.main(argv) == 0
        narrowed = json.loads(capsys.readouterr().out)
        assert narrowed["header"]["lower_bound"] == C[2]
        keep = ",".join(str(c) for c in C[::2])
        argv = ["extract", "--from-json", str(cli_workspace["flat"]), "--filter", keep]
        assert cli.main(argv) == 0
        filtered = json.loads(capsys.readouterr().out)
        assert filtered["header"]["no_of_checkpoints"] == len(C[::2])

    def test_partition_then_unpartition(self, cli_workspace, tmp_path, capsys):
        part = tmp_path / "part.json"
        assert cli.main(
            ["partition", "--from-json", str(cli_workspace["flat"]), "--json", str(part)]
        ) == 0
        ds = read_dataset_json(part)
        assert "partition" in ds
        # Remove 'data' and rebuild it through the CLI.
        del ds["data"]
        stripped = tmp_path / "stripped.json"
        stripped.write_text(pii.dataset_to_json(ds), encoding="utf-8")
        assert cli.main(["unpartition", "--from-json", str(stripped)]) == 0
        rebuilt = json.loads(capsys.readouterr().out)
        original = read_dataset_json(cli_workspace["flat"])
        assert {int(k): {int(m): v for m, v in row.items()} for k, row in rebuilt["data"].items()} \
            == original["data"]

    def test_nest_tuple_keys_in_json(self, cli_workspace):
        text = cli_workspace["nested"].read_text(encoding="utf-8")
        parsed = json.loads(text)
        keys = list(parsed["nested_interval_data"].keys())
        assert all("," in k for k in keys)  # "lower,upper" convention

    def test_guard_message_fails_cleanly(self, cli_workspace, capsys):
        # Nesting an already nested dataset: the library declines.
        argv = ["nest", "--from-json", str(cli_workspace["nested"])]
        assert cli.main(argv) == 1
        assert "No data to work with" in capsys.readouterr().err


class TestDisplayCommand:
    def test_requires_prerequisites(self, cli_workspace, capsys):
        argv = ["display", "--from-json", str(cli_workspace["flat"]), "--view", "winners"]
        assert cli.main(argv) == 1
        assert "Apply the 'winners' function first." in capsys.readouterr().out
        argv = ["display", "--from-json", str(cli_workspace["flat"]), "--count", "partition"]
        assert cli.main(argv) == 1
        assert "First partition the data." in capsys.readouterr().out

    def test_no_description_and_csv(self, cli_workspace, capsys):
        argv = [
            "display", "--from-json", str(cli_workspace["flat"]),
            "--no-description", "--format", "csv",
        ]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "Interval type" not in out
        assert out.splitlines()[0].startswith(",")  # CSV header row

    def test_hide_zeroth(self, cli_workspace, capsys):
        base = read_dataset_json(cli_workspace["flat"])
        first = str(list(base["data"].keys())[0])
        argv = ["display", "--from-json", str(cli_workspace["flat"]), "--hide-zeroth"]
        assert cli.main(argv) == 0
        table = capsys.readouterr().out
        assert not any(line.startswith(first) for line in table.splitlines())


class TestPredictionCommands:
    def test_values(self, capsys):
        assert cli.main(["binom-pmf", "76", "5", "1/16"]) == 0
        got = float(capsys.readouterr().out)
        assert got == pytest.approx(float(pii.binom_pmf(76, 5, 1 / 16)))
        assert cli.main(["frei", "76", "5", "76/16"]) == 0
        got = float(capsys.readouterr().out)
        assert got == pytest.approx(float(pii.frei(76, 5, 76 / 16)))
        assert cli.main(["ms"]) == 0
        assert float(capsys.readouterr().out) == pytest.approx(-1.4150927, abs=1e-6)


class TestStorageCommands:
    def test_save_from_json_roundtrip(self, cli_workspace, tmp_path, capsys):
        db = tmp_path / "roundtrip"
        assert cli.main(["save", "--from-json", str(cli_workspace["flat"]), "--db", str(db)]) == 0
        assert cli.main(["retrieve", "30", "--db", str(db), "--json"]) == 0
        got = json.loads(capsys.readouterr().out)
        want = json.loads(cli_workspace["flat"].read_text(encoding="utf-8"))
        assert got["data"] == want["data"]

    def test_show_table(self, cli_workspace, capsys):
        argv = ["show-table", "--db", str(cli_workspace["db"]), "--type", "overlap"]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "Overlapping intervals." in out  # the caption line
        argv += ["--no-description", "--format", "csv"]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "Overlapping intervals." not in out
        assert out.splitlines()[0].startswith(",A,B,H")

    def test_ensure_tables_creates_directories(self, tmp_path):
        db = tmp_path / "deep" / "down" / "db"
        assert cli.main(["ensure-tables", "--db", str(db)]) == 0
        assert db.exists()

    def test_env_var_is_honored(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "from_env" / "db"
        monkeypatch.setenv("PII_DB", str(db))
        argv = [
            "intervals", "--checkpoints", "0,300,600", "-H", "20",
            "--save", "--json", str(tmp_path / "sink.json"),
        ]
        assert cli.main(argv) == 0
        assert db.exists()
        assert cli.main(["retrieve", "20"]) == 0
        assert "Found 1 dataset" in capsys.readouterr().out


class TestPlotCommands:
    def test_plot_nested_last(self, cli_workspace, tmp_path, capsys):
        out = tmp_path / "frame.png"
        argv = [
            "plot", "--from-json", str(cli_workspace["nested"]),
            "-o", str(out), "--dpi", "35",
        ]
        assert cli.main(argv) == 0
        assert out.exists() and out.stat().st_size > 0
        assert "note: dataset not analyzed" in capsys.readouterr().err

    def test_plot_specific_tuple_checkpoint(self, cli_workspace, tmp_path):
        nested = read_dataset_json(cli_workspace["nested"])
        lo, hi = next(iter(nested["nested_interval_data"]))
        out = tmp_path / "inner.png"
        argv = [
            "plot", "--from-json", str(cli_workspace["nested"]),
            "--checkpoint", f"{lo},{hi}", "-o", str(out), "--dpi", "35",
        ]
        assert cli.main(argv) == 0
        assert out.exists()

    def test_plot_unknown_checkpoint_fails(self, cli_workspace, tmp_path, capsys):
        argv = [
            "plot", "--from-json", str(cli_workspace["flat"]),
            "--checkpoint", "123", "-o", str(tmp_path / "x.png"),
        ]
        assert cli.main(argv) == 1
        assert "not in dataset" in capsys.readouterr().err

    def test_animate_gif(self, cli_workspace, tmp_path):
        out = tmp_path / "anim.gif"
        argv = [
            "animate", "--from-json", str(cli_workspace["nested"]),
            "-o", str(out), "--max-frames", "2", "--dpi", "30", "--fps", "5",
        ]
        assert cli.main(argv) == 0
        assert out.exists() and out.stat().st_size > 0


class TestErrorsAndMeta:
    def test_missing_input_is_usage_error(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["nest"])
        assert exc.value.code == 2

    def test_both_inputs_is_usage_error(self, cli_workspace):
        with pytest.raises(SystemExit) as exc:
            cli.main(
                ["nest", "--from-json", str(cli_workspace["flat"]), "--retrieve", "30"]
            )
        assert exc.value.code == 2

    def test_bad_expression_is_usage_error(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["disjoint", "__import__('os')", "5", "5"])
        assert exc.value.code == 2

    def test_range_and_checkpoints_conflict(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["intervals", "-H", "20", "--range", "0", "100", "50",
                      "--checkpoints", "0,100"])
        assert exc.value.code == 2

    def test_version(self, capsys):
        assert cli.main(["version"]) == 0
        assert capsys.readouterr().out.strip() == pii.__version__

class TestManuscriptCommands:
    def test_eta_and_predict(self, capsys):
        assert cli.main(["eta", "40"]) == 0
        assert float(capsys.readouterr().out) == pytest.approx(pii.eta(40))
        assert cli.main(["predict", "40", "0", "10**4", "--m-max", "6", "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["info"]["mu"] == pytest.approx(pii.averaged_parameter(40, 0, 10**4))
        assert len(payload["table"]["F"]) == 7
        assert cli.main(["predict", "40", "0", "10**4", "--m-max", "4", "--models", "F,B_avg"]) == 0
        out = capsys.readouterr().out
        assert "B_avg omits" in out and "quadrature" in out

    def test_predict_rejects_unknown_model(self):
        with pytest.raises(SystemExit):
            cli.main(["predict", "40", "0", "100", "--models", "nonsense"])

    def test_quadrature_check(self, capsys):
        assert (
            cli.main(["quadrature-check", "30", "0", "10**4", "--m", "0,2,5", "--dps", "20"]) == 0
        )
        out = capsys.readouterr().out
        assert "largest absolute difference" in out
        worst = float(out.strip().split()[-1])
        assert worst < 1e-13

    def test_validate_counters(self, capsys):
        assert cli.main(["validate-counters", "--limit", "3000", "--cases", "40"]) == 0
        assert "all counter validations passed" in capsys.readouterr().out

    def test_provenance_and_conflict_flags(self, tmp_path, capsys):
        db = tmp_path / "db"
        argv = [
            "cumulative-run",
            "-H",
            "10",
            "--checkpoints",
            "5,20",
            "--save",
            "--db",
            str(db),
            "--note",
            "t",
        ]
        assert cli.main(argv) == 0
        assert cli.main(["provenance", "--db", str(db), "--format", "csv"]) == 0
        out = capsys.readouterr().out
        assert out.count("overlap_raw") == 2 and ",t" in out
        assert cli.main(["provenance", "--db", str(tmp_path / "none")]) == 0
        assert "no provenance entries" in capsys.readouterr().out


@pytest.fixture(scope="module")
def cumulative_workspace(tmp_path_factory):
    """Run a small cumulative experiment once for the command tests below."""
    root = tmp_path_factory.mktemp("cumulative")
    db = root / "db"
    data = root / "cum.json"
    argv = [
        "cumulative-run",
        "-H",
        "20",
        "--N-max",
        "1500",
        "--dense-until",
        "30",
        "--ratio",
        "1.3",
        "--save",
        "--db",
        str(db),
        "--json",
        str(data),
    ]
    assert cli.main(argv) == 0
    return {"db": db, "json": data, "root": root}


class TestCumulativeCommands:
    def test_run_output_and_resave(self, cumulative_workspace, capsys):
        ds = read_dataset_json(cumulative_workspace["json"])
        assert ds["header"]["lower_bound"] == 0 and sorted(ds["data"])[:3] == [0, 1, 2]
        # saving the same counts again is harmless
        argv = [
            "cumulative-run",
            "-H",
            "20",
            "--N-max",
            "1500",
            "--dense-until",
            "30",
            "--ratio",
            "1.3",
            "--save",
            "--db",
            str(cumulative_workspace["db"]),
        ]
        assert cli.main(argv) == 0
        assert cli.main(["cumulative-run", "-H", "20"]) == 1
        assert "give --N-max" in capsys.readouterr().err

    def test_frames_from_db_and_json(self, cumulative_workspace, capsys):
        out = cumulative_workspace["root"] / "frames"
        argv = [
            "cumulative-frames",
            "-H",
            "20",
            "--db",
            str(cumulative_workspace["db"]),
            "--out-dir",
            str(out),
            "--overlay-from",
            "200",
            "--cache",
            str(out / "cache.json"),
            "--models",
            "F,F0,B_const",
        ]
        assert cli.main(argv) == 0
        text = capsys.readouterr().out
        assert "final checkpoint N = 1500" in text and "central range m = 0.." in text
        assert (out / "cumulative_H20_by_N.csv").exists()
        assert (out / "cumulative_H20_by_N_m.csv").exists()
        argv = [
            "cumulative-frames",
            "--from-json",
            str(cumulative_workspace["json"]),
            "--out-dir",
            str(out),
            "--overlay-from",
            "200",
            "--cache",
            str(out / "cache.json"),
            "--models",
            "F,F0,B_const",
            "--stem",
            "again",
        ]
        assert cli.main(argv) == 0
        assert "prediction cache" in capsys.readouterr().err
        assert (out / "again_by_N.csv").exists()

    def test_missing_dataset_fails(self, cumulative_workspace, capsys):
        assert (
            cli.main(["cumulative-frames", "-H", "21", "--db", str(cumulative_workspace["db"])])
            == 1
        )
        assert "no overlapping data" in capsys.readouterr().err

    def test_plot_all_and_animate(self, cumulative_workspace, capsys):
        out = cumulative_workspace["root"] / "figs"
        argv = [
            "cumulative-plot",
            "--from-json",
            str(cumulative_workspace["json"]),
            "--out-dir",
            str(out),
            "--overlay-from",
            "200",
            "--formats",
            "png",
            "--dpi",
            "40",
            "--models",
            "F,F0",
        ]
        assert cli.main(argv) == 0
        text = capsys.readouterr().out
        for name in (
            "frame_N1500",
            "residuals_N1500",
            "residuals_N1500_over_eta",
            "discrepancies_central",
            "discrepancies_global",
            "means",
        ):
            assert (out / f"cumulative_H20_{name}.png").exists(), name
        assert text.count("wrote") == 6
        argv = [
            "cumulative-plot",
            "--from-json",
            str(cumulative_workspace["json"]),
            "--out-dir",
            str(out),
            "--overlay-from",
            "200",
            "--formats",
            "png",
            "--dpi",
            "40",
            "--kind",
            "frame",
            "--N",
            "7",
        ]
        assert cli.main(argv) == 0
        assert (out / "cumulative_H20_frame_N7.png").exists()
        argv = [
            "cumulative-plot",
            "--from-json",
            str(cumulative_workspace["json"]),
            "--out-dir",
            str(out),
            "--overlay-from",
            "200",
            "--kind",
            "residuals",
            "--N",
            "7",
        ]
        assert cli.main(argv) == 1
        assert "no predictions" in capsys.readouterr().err
        gif = out / "anim.gif"
        argv = [
            "cumulative-animate",
            "--from-json",
            str(cumulative_workspace["json"]),
            "-o",
            str(gif),
            "--overlay-from",
            "200",
            "--N-min",
            "1000",
            "--dpi",
            "25",
            "--fps",
            "5",
            "--models",
            "F",
        ]
        assert cli.main(argv) == 0
        assert gif.exists() and gif.stat().st_size > 0
        assert "frames, N = 1" in capsys.readouterr().out


def test_score_command_roundtrip(cli_workspace, tmp_path, capsys):
    a, b, c = (tmp_path / f"{x}.json" for x in "abc")
    nested = str(cli_workspace["nested"])
    assert cli.main(["analyze", "--from-json", nested, "--json", str(a)]) == 0
    assert cli.main(["compare", "--from-json", str(a), "--json", str(b)]) == 0
    assert cli.main(["score", "--from-json", str(b), "--json", str(c)]) == 0
    scored = read_dataset_json(c)
    key = next(iter(scored["scores"]))
    assert set(scored["scores"][key]["central"]) == {"B", "F", "F0"}
    assert scored["scores"][key]["central"]["F"]["E1"] >= 0
