"""JSON serialization for interval-count datasets.

Datasets are plain nested dictionaries, but two of their habits do not survive
:mod:`json` unassisted: dictionary keys are integers (prime counts,
checkpoints) or two-integer tuples (nested intervals), and some leaves are
tuples (the ``'comparison'`` item) or NumPy scalars (statistics computed with
NumPy).  The functions here define one documented round-trip convention so
datasets can be piped between command-line invocations, written to files for
inspection, and reloaded:

* **Keys.** Integer keys are written as their decimal strings (JSON object
  keys must be strings); two-integer tuple keys are written as
  ``"lower,upper"``; string keys pass through.  On reading, a key matching an
  optionally signed integer becomes an ``int``, a key matching two such
  integers joined by a comma becomes a tuple, and anything else stays a
  string.  This is unambiguous for the dataset schema because none of its
  string keys (``'header'``, ``'data'``, ``'mean'``, ``'B sq error'``, and so
  on) look like integers.
* **Tuples.** JSON has arrays only, so tuples are written as arrays.  On
  reading, arrays inside the ``'comparison'`` item are converted back to
  tuples (two levels deep, restoring the ``(probabilities, counts)`` pairs);
  arrays elsewhere (``'contents'``, ``'mode'``, the winners' lists) remain
  lists, as in the originals.
* **NumPy scalars.** Written as the corresponding Python ``int`` or
  ``float``.  A reloaded dataset is therefore numerically equal to, but not
  type-identical with, a freshly computed one (for example, a standard
  deviation computed as ``numpy.float64`` returns as ``float``); every
  consumer in this package treats the two alike.

The SQLite layer in :mod:`primes_in_intervals.dataio` remains the canonical
store for raw counts; JSON is the transport for everything downstream of the
counters, including analyzed, compared, and nested datasets, which the
database schema does not cover.
"""

from __future__ import annotations

import json
import numbers
import re
import sys
from pathlib import Path
from typing import Any

from primes_in_intervals.intervals import Dataset

__all__ = [
    "dataset_from_json",
    "dataset_to_json",
    "read_dataset_json",
    "write_dataset_json",
]

# An optionally signed run of digits: the string form of an int key.
_INT_KEY = re.compile(r"^-?\d+$")
# Two such runs joined by a comma: the string form of a (lower, upper) key.
_TUPLE_KEY = re.compile(r"^-?\d+,-?\d+$")


def _encode_key(key: Any) -> str:
    """Return the JSON object key for a dataset dictionary key."""
    if isinstance(key, tuple):
        return f"{key[0]},{key[1]}"
    return str(key)


def _decode_key(key: str) -> Any:
    """Invert :func:`_encode_key` by pattern-matching the string form."""
    if _INT_KEY.match(key):
        return int(key)
    if _TUPLE_KEY.match(key):
        lo, hi = key.split(",")
        return (int(lo), int(hi))
    return key


def _encode(value: Any) -> Any:
    """Recursively convert a dataset fragment to JSON-compatible types."""
    if isinstance(value, dict):
        return {_encode_key(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # Third-party scalars: NumPy integers and floats register with the
    # numbers ABCs; others, a SymPy Float for instance, do not, but expose
    # __float__.
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__float__"):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__!r} to JSON")


def _decode(value: Any, in_comparison: bool = False) -> Any:
    """Recursively invert :func:`_encode`.

    Parameters
    ----------
    value : object
        The JSON-decoded fragment.
    in_comparison : bool, optional
        Whether the fragment lies inside a ``'comparison'`` item, in which
        case arrays are restored to tuples.
    """
    if isinstance(value, dict):
        return {
            _decode_key(k): _decode(v, in_comparison or k == "comparison") for k, v in value.items()
        }
    if isinstance(value, list):
        decoded = [_decode(v, in_comparison) for v in value]
        return tuple(decoded) if in_comparison else decoded
    return value


def dataset_to_json(dataset: Dataset, indent: int | None = 2) -> str:
    """Serialize a dataset to a JSON string.

    Parameters
    ----------
    dataset : dict
        Any dataset produced by this package: raw, extracted, partitioned,
        nested, analyzed, compared, or scored.
    indent : int or None, optional
        Indentation passed to :func:`json.dumps` (default 2; ``None`` for the
        most compact form).

    Returns
    -------
    str
        The JSON text, following the module's key and tuple conventions.
    """
    return json.dumps(_encode(dataset), indent=indent)


def dataset_from_json(text: str) -> Dataset:
    """Reconstruct a dataset from the JSON produced by :func:`dataset_to_json`.

    Parameters
    ----------
    text : str
        JSON text following the module's conventions.

    Returns
    -------
    dict
        The dataset, with integer and tuple keys restored and the
        ``'comparison'`` item's tuples reinstated.
    """
    return _decode(json.loads(text))


def write_dataset_json(dataset: Dataset, path: str | Path) -> None:
    """Write a dataset as JSON to a file, or to standard output.

    Parameters
    ----------
    dataset : dict
        The dataset to write.
    path : str or Path
        Destination file; the special value ``"-"`` writes to standard
        output (the command-line convention).
    """
    text = dataset_to_json(dataset)
    if str(path) == "-":
        sys.stdout.write(text + "\n")
    else:
        Path(path).write_text(text + "\n", encoding="utf-8")


def read_dataset_json(path: str | Path) -> Dataset:
    """Read a dataset from a JSON file, or from standard input.

    Parameters
    ----------
    path : str or Path
        Source file; the special value ``"-"`` reads standard input (the
        command-line convention).

    Returns
    -------
    dict
        The reconstructed dataset.
    """
    if str(path) == "-":
        return dataset_from_json(sys.stdin.read())
    return dataset_from_json(Path(path).read_text(encoding="utf-8"))
