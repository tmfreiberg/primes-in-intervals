"""Tests for the SQLite save/retrieve layer, on temporary databases."""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest

import primes_in_intervals as pii


def _small(itype):
    return pii.intervals(list(range(0, 1001, 250)), 25, itype)


class TestSaveAndRetrieve:
    def test_roundtrip_all_types(self, tmp_path, capsys):
        db = tmp_path / "db"
        for itype in ["disjoint", "overlap", "prime_start"]:
            ds = _small(itype)
            pii.save(ds, db_path=db)
            got = pii.retrieve(25, itype, db_path=db)
            out = capsys.readouterr().out
            assert "Found 1 dataset" in out
            assert "'header'" in out
            assert got["data"] == ds["data"]
            assert got["header"] == ds["header"]

    def test_multiple_lower_bounds_return_list(self, tmp_path, capsys):
        db = tmp_path / "db"
        ds1 = pii.intervals([0, 500, 1000], 25, "overlap")
        ds2 = pii.intervals([5000, 5500, 6000], 25, "overlap")
        pii.save(ds1, db_path=db)
        pii.save(ds2, db_path=db)
        got = pii.retrieve(25, "overlap", db_path=db)
        out = capsys.readouterr().out
        assert "Found 2 datasets" in out
        assert isinstance(got, list) and len(got) == 2
        assert got[0]["data"] == ds1["data"]
        assert got[1]["data"] == ds2["data"]

    def test_insert_or_ignore(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("overlap")
        pii.save(ds, db_path=db)
        pii.save(copy.deepcopy(ds), db_path=db)  # second save is a no-op
        conn = sqlite3.connect(db)
        (count,) = conn.execute("SELECT COUNT(*) FROM overlap_raw").fetchone()
        conn.close()
        assert count == len(ds["data"]) - 1  # one row per non-initial checkpoint

    def test_row_layout(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("disjoint")
        pii.save(ds, db_path=db)
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT * FROM disjoint_raw ORDER BY upper_bound LIMIT 1"
        ).fetchone()
        conn.close()
        C = list(ds["data"].keys())
        assert row[:3] == (C[0], C[1], 25)
        assert len(row) == pii.max_primes + 4
        for m, v in ds["data"][C[1]].items():
            assert row[m + 3] == v

    def test_save_without_data_prints_message(self, tmp_path, capsys):
        assert pii.save({"header": {}}, db_path=tmp_path / "db") is None
        assert "No data to save" in capsys.readouterr().out


class TestMissingTables:
    def test_retrieve_message(self, tmp_path, capsys):
        db = tmp_path / "empty"
        assert pii.retrieve(25, "overlap", db_path=db) is None
        assert (
            capsys.readouterr().out
            == "Database contains no table for overlapping intervals.\n"
        )

    def test_show_table_messages(self, tmp_path, capsys):
        db = tmp_path / "empty"
        expected = {
            "disjoint": "Database contains no table for disjoint intervals.",
            "overlap": "Database contains no table for overlapping intervals.",
            "prime_start": "Database contains no table for prime-starting intervals.",
        }
        for itype, message in expected.items():
            assert pii.show_table(itype, db_path=db) is None
            assert capsys.readouterr().out == message + "\n"


class TestShowTable:
    def test_bare_dataframe(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("overlap")
        pii.save(ds, db_path=db)
        df = pii.show_table("overlap", description="no description", db_path=db)
        assert list(df.columns[:3]) == ["A", "B", "H"]
        assert len(df) == len(ds["data"]) - 1

    def test_caption(self, tmp_path):
        db = tmp_path / "db"
        pii.save(_small("disjoint"), db_path=db)
        styled = pii.show_table("disjoint", db_path=db)
        assert "Disjoint intervals." in styled.caption


class TestSetDb:
    def test_default_path_is_switchable(self, tmp_path):
        old = pii.DB_PATH
        try:
            db = tmp_path / "switched"
            pii.set_db(db)
            ds = _small("overlap")
            pii.save(ds)
            assert db.exists()
            got = pii.retrieve(25, "overlap")
            assert got["data"] == ds["data"]
        finally:
            pii.set_db(old)


class TestPathResolution:
    def test_default_location(self):
        assert pii.DB_PATH == Path("data") / "primes_in_intervals_db"

    def test_env_var_is_used(self, tmp_path, monkeypatch):
        # Neutralize any set_db from earlier tests so the environment
        # variable is actually consulted; monkeypatch restores it after.
        monkeypatch.setattr(pii.dataio, "_DB_OVERRIDE", None)
        db = tmp_path / "made" / "by" / "env"
        monkeypatch.setenv("PII_DB", str(db))
        ds = _small("overlap")
        pii.save(ds)
        assert db.exists()
        got = pii.retrieve(25, "overlap")
        assert got["data"] == ds["data"]

    def test_set_db_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pii.dataio, "_DB_OVERRIDE", None)
        via_env = tmp_path / "env_db"
        via_set = tmp_path / "set_db"
        monkeypatch.setenv("PII_DB", str(via_env))
        pii.set_db(via_set)
        pii.save(_small("overlap"))
        assert via_set.exists()
        assert not via_env.exists()

class TestConflictsWidthAndProvenance:
    def test_identical_resave_is_skipped(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("overlap")
        first = pii.save(ds, db_path=db, note="run 1")
        assert first["inserted"] == len(ds["data"]) - 1 and first["conflicts"] == []
        again = pii.save(copy.deepcopy(ds), db_path=db)
        assert again["inserted"] == 0 and again["identical"] == len(ds["data"]) - 1

    def test_conflict_is_reported_not_overwritten(self, tmp_path, capsys):
        db = tmp_path / "db"
        ds = _small("overlap")
        pii.save(ds, db_path=db)
        bad = copy.deepcopy(ds)
        c = list(bad["data"])[-1]
        m = next(m for m, v in bad["data"][c].items() if v)
        bad["data"][c][m] += 1
        with pytest.raises(pii.StorageConflictError) as info:
            pii.save(bad, db_path=db)
        assert info.value.conflicts[0]["key"] == (0, c, 25)
        assert info.value.conflicts[0]["new"] == {m: bad["data"][c][m]}
        # nothing changed
        assert pii.retrieve(25, "overlap", db_path=db)["data"] == ds["data"]
        capsys.readouterr()
        summary = pii.save(bad, db_path=db, on_conflict="skip")
        assert summary["replaced"] == 0 and len(summary["conflicts"]) == 1
        assert "kept the stored versions" in capsys.readouterr().out
        assert pii.retrieve(25, "overlap", db_path=db)["data"] == ds["data"]
        capsys.readouterr()
        summary = pii.save(bad, db_path=db, on_conflict="replace")
        assert summary["replaced"] == 1
        capsys.readouterr()
        assert pii.retrieve(25, "overlap", db_path=db)["data"] == bad["data"]
        with pytest.raises(ValueError):
            pii.save(bad, db_path=db, on_conflict="whatever")

    def test_counts_beyond_the_default_width_are_kept(self, tmp_path, capsys):
        db = tmp_path / "db"
        ds = pii.intervals([0, 100, 200], 5, "overlap")
        ds["data"][100][150] = 0
        ds["data"][200][150] = 7  # a count at m = 150 > max_primes
        assert pii.table_width("overlap", db_path=db) is None
        summary = pii.save(ds, db_path=db)
        assert summary["widened_to"] == 150
        assert "widened" in capsys.readouterr().out
        assert pii.table_width("overlap", db_path=db) == 150
        got = pii.retrieve(5, "overlap", db_path=db)
        assert got["data"][200][150] == 7
        df = pii.show_table("overlap", "no description", db_path=db)
        assert list(df.columns)[-1] == 150 and df.shape[0] == 2

    def test_provenance_entries(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("disjoint")
        pii.save(ds, db_path=db, note="disjoint run")
        prov = pii.provenance_of(db_path=db)
        assert len(prov) == len(ds["data"]) - 1
        assert set(prov["table_name"]) == {"disjoint_raw"}
        assert set(prov["counter"]) == {"disjoint_cp"}
        assert set(prov["counter_version"]) == {pii.COUNTER_VERSION}
        assert set(prov["package_version"]) == {pii.__version__}
        assert set(prov["note"]) == {"disjoint run"}
        assert len(pii.provenance_of(25, "overlap", db_path=db)) == 0
        assert len(pii.provenance_of(db_path=tmp_path / "missing")) == 0

    def test_stored_rows_match_independent_reference(self):
        # The committed database: spot-check rows far from the origin with
        # the segmented reference sieve.
        from primes_in_intervals import reference as ref

        if not pii.DB_PATH.exists():
            pytest.skip("committed database not present")
        found = pii.retrieve(76, "overlap")
        if found is None:
            pytest.skip("no H = 76 data")
        ds = found if isinstance(found, dict) else found[0]
        A = ds["header"]["lower_bound"]
        C = sorted(ds["data"])
        for c in (C[1], C[2], C[-1]):
            got = {m: v for m, v in ds["data"][c].items() if v}
            assert got == ref.overlap_reference_segment(A, c, 76)


def test_resaving_stored_data_does_not_change_the_database(tmp_path):
    db = tmp_path / "db"
    ds = _small("overlap")
    pii.save(ds, db_path=db)
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE provenance")  # as in a database saved before provenance existed
    conn.commit()
    conn.close()
    before = db.read_bytes()
    pii.save(copy.deepcopy(ds), db_path=db)
    assert db.read_bytes() == before
