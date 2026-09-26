"""SQLite persistence for interval-count datasets.

The database (by default ``data/primes_in_intervals_db``, resolved relative
to the current working directory, so run from the repository root; override
with :func:`set_db`, the ``PII_DB`` environment variable, or a ``db_path``
argument) holds one table per interval type:

* ``disjoint_raw``
* ``overlap_raw``
* ``prime_start_raw``

Each table has columns ``lower_bound``, ``upper_bound``, ``interval_length``
(together the primary key) followed by ``m0, m1, ..., mK``: the number of
intervals ``(a, a + H]`` with ``a`` in ``(lower_bound, upper_bound]`` (``a``
in an arithmetic progression mod ``H`` in the disjoint case, ``a`` prime in
the prime-start case) containing exactly ``m`` primes.  New tables are created
with ``K`` = :data:`max_primes` = 100.  A dataset that needs a larger ``m``
does not lose anything: :func:`save` widens the table with further ``mK``
columns (default 0) before writing, and :func:`retrieve` and
:func:`show_table` read whatever columns a table has.

So ``lower_bound``, ``upper_bound``, ``interval_length``, ``m0``, ...,
``mK`` are columns ``0, 1, 2, 3, ..., K + 3`` respectively: ``mi`` is column
``i + 3``.

A dataset's checkpoint rows share their ``lower_bound``; :func:`save` writes
one row per checkpoint, and :func:`retrieve` groups rows by ``lower_bound``
to reconstruct the original meta-dictionaries.

**Conflicts.**  A row whose key ``(lower_bound, upper_bound,
interval_length)`` is already present is compared with the stored row.  If
the counts agree the row is skipped (re-saving is harmless); if they differ
the disagreement is a *conflict*, and :func:`save` reports it rather than
silently keeping either version: by default it raises
:class:`StorageConflictError` and writes nothing, and it can instead skip or
replace the conflicting rows when told to.

**Provenance.**  Every row written by :func:`save` gets a companion entry in
the ``provenance`` table recording the table, the key, the counter and its
version (:data:`~primes_in_intervals.intervals.COUNTER_VERSION`), the package
version, a timestamp, and a free-text note.  Rows saved before this table
existed have no entry; :func:`provenance_of` lists what is known.

Importing this module does not touch the filesystem; tables are created on
first use by :func:`save` (or explicitly by :func:`ensure_tables`), and every
function accepts a ``db_path`` argument, defaulting to :data:`DB_PATH`, which
can be changed globally with :func:`set_db`.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from primes_in_intervals.intervals import COUNTER_VERSION, Dataset, zeros

__all__ = [
    "DB_PATH",
    "StorageConflictError",
    "ensure_tables",
    "max_primes",
    "provenance_of",
    "retrieve",
    "save",
    "set_db",
    "show_table",
    "table_width",
]

#: Largest per-interval prime count in a freshly created table (columns
#: ``m0`` .. ``m{max_primes}``); tables grow beyond this on demand.
max_primes = 100

#: Default database location: ``data/primes_in_intervals_db``, resolved
#: relative to the current working directory (so run from the repo root).
DB_PATH: Path = Path("data") / "primes_in_intervals_db"

#: Set by :func:`set_db`; when not ``None`` it overrides both the ``PII_DB``
#: environment variable and the :data:`DB_PATH` default.
_DB_OVERRIDE: Path | None = None

_TABLES = {
    "disjoint": "disjoint_raw",
    "overlap": "overlap_raw",
    "prime_start": "prime_start_raw",
}

_MISSING_TABLE_MESSAGE = {
    "disjoint": "Database contains no table for disjoint intervals.",
    "overlap": "Database contains no table for overlapping intervals.",
    "prime_start": "Database contains no table for prime-starting intervals.",
}

_CAPTION = {
    "disjoint": (
        "Disjoint intervals. "
        r"Column with label $m$ shows $\#\{1 \le k \le (B - A)/H : "
        r"\pi(A + kH) - \pi(A + (k - 1)H) = m \}$"
    ),
    "overlap": (
        "Overlapping intervals. "
        r"Column with label $m$ shows $\#\{A < a \le B : "
        r"\pi(a + H) - \pi(a) = m \}$"
    ),
    "prime_start": (
        "Prime-starting intervals. "
        r"Column with label $m$ shows $\#\{A < p \le B : "
        r"\pi(p + H) - \pi(p) = m \}$, $p$ prime."
    ),
}

_PROVENANCE_TABLE = "provenance"


class StorageConflictError(ValueError):
    """Raised by :func:`save` when stored counts disagree with the counts being saved.

    Attributes
    ----------
    conflicts : list of dict
        One entry per conflicting row with keys ``'key'`` (the
        ``(lower_bound, upper_bound, interval_length)`` triple),
        ``'stored'`` and ``'new'`` (``{m: count}`` dictionaries of the
        differing entries only).
    """

    def __init__(self, table: str, conflicts: list[dict[str, Any]]):
        self.table = table
        self.conflicts = conflicts
        keys = ", ".join(str(c["key"]) for c in conflicts[:5])
        more = "" if len(conflicts) <= 5 else f" and {len(conflicts) - 5} more"
        super().__init__(
            f"{len(conflicts)} row(s) of {table} already hold different counts "
            f"(keys {keys}{more}); nothing was written. Inspect them, then call "
            "save(..., on_conflict='skip') to keep the stored rows or "
            "on_conflict='replace' to overwrite them."
        )


def _resolve(db_path: str | Path | None) -> str:
    """Return the database path to use, as a string for ``sqlite3.connect``.

    Resolution order, first match wins:

    1. an explicit ``db_path`` argument;
    2. a path set by :func:`set_db`;
    3. the ``PII_DB`` environment variable;
    4. the default :data:`DB_PATH` (``data/primes_in_intervals_db``).
    """
    if db_path is not None:
        return str(db_path)
    if _DB_OVERRIDE is not None:
        return str(_DB_OVERRIDE)
    env = os.environ.get("PII_DB")
    if env:
        return env
    return str(DB_PATH)


def set_db(db_path: str | Path) -> Path:
    """Set the process-wide database location.

    The location set here takes precedence over the ``PII_DB`` environment
    variable and the :data:`DB_PATH` default, and applies to every subsequent
    call that does not pass an explicit ``db_path``.

    Parameters
    ----------
    db_path : str or Path
        New location; subsequent calls without an explicit ``db_path`` use it.

    Returns
    -------
    Path
        The new location, for convenience.
    """
    global _DB_OVERRIDE
    _DB_OVERRIDE = Path(db_path)
    return _DB_OVERRIDE


def _count_columns(width: int) -> str:
    """Return ``'m0 int, m1 int, ..., m{width} int, '``."""
    return "".join(f"m{i} int, " for i in range(width + 1))


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _width(conn: sqlite3.Connection, table: str) -> int:
    """Return the largest ``K`` such that column ``mK`` exists in ``table``."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    ms = [int(c[1:]) for c in cols if c.startswith("m") and c[1:].isdigit()]
    return max(ms) if ms else -1


def _widen(conn: sqlite3.Connection, table: str, new_width: int) -> int:
    """Add columns ``m{K+1} .. m{new_width}`` (default 0) to ``table``; return columns added."""
    current = _width(conn, table)
    for i in range(current + 1, new_width + 1):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN m{i} int DEFAULT 0")
    return max(0, new_width - current)


def table_width(interval_type: str, db_path: str | Path | None = None) -> int | None:
    """Return the largest ``m`` a raw table can store, or ``None`` if the table is absent.

    Parameters
    ----------
    interval_type : str
        ``'disjoint'``, ``'overlap'``, or ``'prime_start'``.
    db_path : str, Path, or None, optional
        Database file; resolved as described in :func:`set_db`.

    Returns
    -------
    int or None
    """
    if interval_type not in _TABLES:
        return None
    resolved = _resolve(db_path)
    if not Path(resolved).exists():
        return None
    conn = sqlite3.connect(resolved)
    try:
        if not _table_exists(conn, _TABLES[interval_type]):
            return None
        return _width(conn, _TABLES[interval_type])
    finally:
        conn.close()


def ensure_tables(db_path: str | Path | None = None, width: int = max_primes) -> None:
    """Create the three raw tables and the provenance table if they do not exist.

    The raw schema is the original project's: three integer key columns
    forming the primary key, then ``m0`` through ``m{width}``.

    Parameters
    ----------
    db_path : str, Path, or None, optional
        Database file; resolved as described in :func:`set_db`.
    width : int, optional
        Largest ``m`` column in a newly created raw table (default
        :data:`max_primes`).  Existing tables are left as they are.
    """
    resolved = _resolve(db_path)
    # A fresh clone will not have the data/ directory yet; without this,
    # sqlite3 fails with an unhelpful "unable to open database file".
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    cols = _count_columns(width)
    conn = sqlite3.connect(resolved)
    for table in _TABLES.values():
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(lower_bound int, upper_bound int, interval_length int,"
            + cols
            + "PRIMARY KEY(lower_bound, upper_bound, interval_length))"
        )
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_PROVENANCE_TABLE} "
        "(table_name text, lower_bound int, upper_bound int, interval_length int, "
        "counter text, counter_version text, package_version text, saved_at text, "
        "note text, "
        "PRIMARY KEY(table_name, lower_bound, upper_bound, interval_length))"
    )
    conn.commit()
    conn.close()


def _row_counts(row: tuple, width: int) -> dict[int, int]:
    """Return ``{m: count}`` for the nonzero entries of a stored row."""
    return {m: row[m + 3] for m in range(width + 1) if row[m + 3]}


def save(
    data: Dataset,
    db_path: str | Path | None = None,
    on_conflict: str = "error",
    note: str = "",
) -> dict[str, Any] | None:
    """Store a dataset's checkpoint rows in the appropriate raw table.

    One row is inserted per checkpoint ``C[k]`` (``k >= 1``), of the form
    ``(C[0], C[k], H, g(0), g(1), ..., g(K))``, into the table named by the
    dataset's ``'interval_type'``, together with a provenance entry.  The
    table is widened first if the dataset has counts at ``m`` beyond its
    current last column, so no count is ever dropped.

    Rows whose key already exists are compared with the stored counts:
    identical rows are skipped, differing rows are conflicts.

    Parameters
    ----------
    data : dict
        A meta-dictionary with ``'header'`` and ``'data'`` items, as produced
        by the ``_cp`` counting functions.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.  Tables are created if
        absent.
    on_conflict : str, optional
        ``'error'`` (default): raise :class:`StorageConflictError` and write
        nothing; ``'skip'``: write the non-conflicting rows, keep the stored
        versions of the conflicting ones, and report; ``'replace'``: write
        everything, overwriting the conflicting rows, and report.
    note : str, optional
        Free text recorded in the provenance entries (what the run was for).

    Returns
    -------
    dict or None
        A summary with keys ``'table'``, ``'inserted'``, ``'identical'``,
        ``'replaced'``, ``'conflicts'`` (list of the conflicting keys with the
        differing counts), and ``'widened_to'`` (the table's last ``m``
        column after the call); ``None`` (with a message) if there is no
        data to save.

    Raises
    ------
    StorageConflictError
        With ``on_conflict='error'``, if any stored row disagrees.
    ValueError
        If ``on_conflict`` is not one of the three choices, or the interval
        type is unknown.
    """
    if "data" not in data.keys():
        return print("No data to save. Check contents.")
    if on_conflict not in ("error", "skip", "replace"):
        raise ValueError("on_conflict must be 'error', 'skip', or 'replace'")
    interval_type = data["header"]["interval_type"]
    if interval_type not in _TABLES:
        raise ValueError(f"unknown interval type {interval_type!r}")
    table = _TABLES[interval_type]
    ensure_tables(db_path)
    C = list(data["data"].keys())
    H = data["header"]["interval_length"]
    # The padded key set is the union of every m with a nonzero count
    # somewhere, so its maximum is the largest m that must be stored.
    needed = max((max(data["data"][c].keys(), default=0) for c in C), default=0)

    conn = sqlite3.connect(_resolve(db_path))
    try:
        widened = _widen(conn, table, needed)
        if widened:
            print(
                f"note: {table} widened to hold counts up to m = {needed} "
                f"({widened} column(s) added)"
            )
        width = _width(conn, table)
        new_rows: list[tuple] = []
        replace_rows: list[tuple] = []
        identical = 0
        conflicts: list[dict[str, Any]] = []
        for k in range(1, len(C)):
            row = [0] * (width + 4)
            row[0], row[1], row[2] = C[0], C[k], H
            for m, count in data["data"][C[k]].items():
                row[m + 3] = count
            stored = conn.execute(
                f"SELECT * FROM {table} WHERE lower_bound=? AND upper_bound=? "  # noqa: S608
                "AND interval_length=?",
                (C[0], C[k], H),
            ).fetchone()
            if stored is None:
                new_rows.append(tuple(row))
                continue
            old = _row_counts(stored, width)
            new = {m: count for m, count in data["data"][C[k]].items() if count}
            if old == new:
                identical += 1
                continue
            differing = sorted(set(old) | set(new))
            conflicts.append(
                {
                    "key": (C[0], C[k], H),
                    "stored": {
                        m: old.get(m, 0) for m in differing if old.get(m, 0) != new.get(m, 0)
                    },
                    "new": {m: new.get(m, 0) for m in differing if old.get(m, 0) != new.get(m, 0)},
                }
            )
            replace_rows.append(tuple(row))
        if conflicts and on_conflict == "error":
            conn.rollback()
            raise StorageConflictError(table, conflicts)
        qstring = ",".join("?" * (width + 4))
        conn.executemany(f"INSERT INTO {table} VALUES({qstring})", new_rows)  # noqa: S608
        replaced = 0
        if conflicts and on_conflict == "replace":
            conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES({qstring})", replace_rows)  # noqa: S608
            replaced = len(replace_rows)
        written = new_rows + (replace_rows if on_conflict == "replace" else [])
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        import primes_in_intervals

        conn.executemany(
            f"INSERT OR REPLACE INTO {_PROVENANCE_TABLE} VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    table,
                    r[0],
                    r[1],
                    r[2],
                    f"{interval_type}_cp",
                    COUNTER_VERSION,
                    primes_in_intervals.__version__,
                    stamp,
                    note,
                )
                for r in written
            ],
        )
        conn.commit()
    finally:
        conn.close()
    if conflicts:
        verb = "kept the stored versions of" if on_conflict == "skip" else "overwrote"
        print(
            f"warning: {len(conflicts)} row(s) of {table} held different counts; "
            f"{verb} them (keys {[c['key'] for c in conflicts[:5]]}"
            f"{'...' if len(conflicts) > 5 else ''})"
        )
    return {
        "table": table,
        "inserted": len(new_rows),
        "identical": identical,
        "replaced": replaced,
        "conflicts": conflicts,
        "widened_to": width,
    }


def provenance_of(
    H: int | None = None,
    interval_type: str | None = None,
    db_path: str | Path | None = None,
) -> Any:
    """Return the provenance entries as a DataFrame, optionally filtered.

    Parameters
    ----------
    H : int, optional
        Restrict to this interval length.
    interval_type : str, optional
        Restrict to this interval type.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.

    Returns
    -------
    pandas.DataFrame
        Columns ``table_name``, ``lower_bound``, ``upper_bound``,
        ``interval_length``, ``counter``, ``counter_version``,
        ``package_version``, ``saved_at``, ``note``; empty if the database or
        the provenance table does not exist (rows saved before provenance was
        recorded have no entry).
    """
    columns = [
        "table_name",
        "lower_bound",
        "upper_bound",
        "interval_length",
        "counter",
        "counter_version",
        "package_version",
        "saved_at",
        "note",
    ]
    resolved = _resolve(db_path)
    if not Path(resolved).exists():
        return pd.DataFrame(columns=columns)
    conn = sqlite3.connect(resolved)
    try:
        if not _table_exists(conn, _PROVENANCE_TABLE):
            return pd.DataFrame(columns=columns)
        clauses: list[str] = []
        params: list[Any] = []
        if H is not None:
            clauses.append("interval_length = ?")
            params.append(H)
        if interval_type is not None:
            clauses.append("table_name = ?")
            params.append(_TABLES.get(interval_type, interval_type))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM {_PROVENANCE_TABLE}{where} "  # noqa: S608
            "ORDER BY table_name, interval_length, lower_bound, upper_bound",
            params,
        ).fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


def show_table(
    interval_type: str,
    description: str = "description",
    db_path: str | Path | None = None,
) -> Any:
    """Return an entire raw table as a DataFrame.

    Parameters
    ----------
    interval_type : str
        ``'disjoint'``, ``'overlap'``, or ``'prime_start'``.
    description : str, optional
        ``'no description'`` for the bare DataFrame; anything else (the
        default) returns a styled DataFrame with an explanatory caption.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.

    Returns
    -------
    pandas.DataFrame or pandas Styler or None
        The table, ordered by ``lower_bound``, ``upper_bound``,
        ``interval_length``, with columns ``A``, ``B``, ``H``, ``0``, ...,
        ``K`` (``K`` the table's last ``m`` column); or ``None`` (with a
        message) if the table is absent.
    """
    if interval_type not in _TABLES:
        return None
    table = _TABLES[interval_type]
    resolved = _resolve(db_path)
    # Connecting to a nonexistent file would create an empty database as a
    # side effect (and fail outright if its directory is missing); a file
    # that is not there certainly contains no tables, so short-circuit.
    if not Path(resolved).exists():
        print(_MISSING_TABLE_MESSAGE[interval_type])
        return None
    conn = sqlite3.connect(resolved)
    if not _table_exists(conn, table):
        print(_MISSING_TABLE_MESSAGE[interval_type])
        conn.close()
        return None
    width = _width(conn, table)
    res = conn.execute(
        f"SELECT * FROM {table} "  # noqa: S608 - table name from fixed mapping
        "ORDER BY lower_bound ASC, upper_bound ASC, interval_length ASC"
    )
    rows = res.fetchall()
    conn.close()
    cols: list[Any] = ["A", "B", "H"]
    for m in range(0, width + 1):
        cols.append(m)
    df = pd.DataFrame(rows, columns=cols)
    if description == "no description":
        return df
    else:
        return df.style.set_caption(_CAPTION[interval_type])


def retrieve(
    H: int,
    interval_type: str = "overlap",
    db_path: str | Path | None = None,
) -> Dataset | list[Dataset] | None:
    """Reconstruct the dataset(s) with interval length ``H`` from the database.

    Rows are grouped by ``lower_bound``: each group of rows
    ``(A, C[k], H, g(0), ..., g(K))`` becomes one meta-dictionary with
    the usual ``'header'`` and a ``'data'`` item mapping each checkpoint to its
    frequency dictionary (zero-count keys re-trimmed by
    :func:`~primes_in_intervals.intervals.zeros`, exactly as when the data was
    first computed).

    A summary of what was found, including each dataset's header, is printed.

    Parameters
    ----------
    H : int
        Interval length to look up.
    interval_type : str, optional
        ``'disjoint'``, ``'overlap'`` (the default), or ``'prime_start'``.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.

    Returns
    -------
    dict, list of dict, or None
        A single meta-dictionary if exactly one dataset (one distinct
        ``lower_bound``) matches; a list of meta-dictionaries if several do;
        ``None`` (with a message) if the table is absent.
    """
    if interval_type not in _TABLES:
        return None
    table = _TABLES[interval_type]
    resolved = _resolve(db_path)
    # See show_table: avoid creating an empty database on a read.
    if not Path(resolved).exists():
        print(_MISSING_TABLE_MESSAGE[interval_type])
        return None
    conn = sqlite3.connect(resolved)
    if not _table_exists(conn, table):
        print(_MISSING_TABLE_MESSAGE[interval_type])
        conn.close()
        return None
    width = _width(conn, table)
    res = conn.execute(
        f"SELECT * FROM {table} "  # noqa: S608 - table name from fixed mapping
        "WHERE (interval_length) = (?) ORDER BY lower_bound ASC, upper_bound ASC",
        (H,),
    )
    rows = res.fetchall()
    # rows = [(C[0], C[k], H, g(0), ..., g(K)), k = 0,1,...),
    #         (C'[0], C'[k], H, g(0), ..., g(K)), k = 0,1,...), ...]
    conn.close()
    found: dict[int, dict[int, dict[int, int]]] = {}
    i = 0
    while i < len(rows):
        A = rows[i][0]  # C[0]
        found[A] = {}
        j = i
        while j < len(rows) and rows[j][0] == A:
            B = rows[j][1]
            found[A][B] = {m - 3: rows[j][m] for m in range(3, width + 4)}
            j += 1
        i = j
    output = []
    for A in found.keys():
        C = list(found[A].keys())
        C.insert(0, A)
        outputA: Dataset = {
            "header": {
                "interval_type": interval_type,
                "lower_bound": A,
                "upper_bound": C[-1],
                "interval_length": H,
                "no_of_checkpoints": len(C),
                "contents": ["data"],
            }
        }
        data = {C[0]: {m: 0 for m in range(H + 1)}}
        for cc in C[1:]:
            data[cc] = found[A][cc]
        trimmed_data = zeros(data)
        outputA["data"] = trimmed_data
        output.append(outputA)
    if len(output) == 1:
        print(
            f"Found {len(output)} dataset corresponding to interval of "
            f"length {H} ({interval_type} intervals)."
        )
        print(f"\n 'header' : {output[0]['header']}\n")
        return output[0]
    else:
        print(
            f"Found {len(output)} datasets corresponding to interval of "
            f"length {H} ({interval_type} intervals)."
        )
        for i in range(len(output)):
            print(f"\n [{i}] 'header' : {output[i]['header']}\n")
        return output
