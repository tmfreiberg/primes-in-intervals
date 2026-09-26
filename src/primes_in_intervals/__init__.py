"""Primes in intervals: computation, storage, analysis, and visualization.

Count how many intervals of a given length contain each possible number of
primes, under three sampling schemes (disjoint blocks, a sliding window, and
windows started at primes); persist the counts in SQLite; reorganize them
(sub-ranges, partitions, nested centered intervals); summarize the resulting
distributions; compare them against the manuscript's predictions (the
integrated corrected prediction ``F``, its uncorrected form ``F_0``, the
local expression ``Q``, and binomial approximations to Cramér's model); and
present everything as tables, figures and animations.

The public API is flat::

    import primes_in_intervals as pii

    C = list(range(0, 10**6 + 1, 10**5))
    X = pii.intervals(C, 100, 'disjoint')
    pii.save(X)
    pii.analyze(X)
    pii.display(X)

The fixed-``H`` cumulative experiment (starting points ``1 <= n <= N``) is
in :mod:`primes_in_intervals.cumulative`, and the independent reference
counts used to validate the counters in :mod:`primes_in_intervals.reference`.
"""

from primes_in_intervals.comparisons import COMPARISON_LABEL, compare, score, winners
from primes_in_intervals.cumulative import (
    DEFAULT_OVERLAY_FROM,
    Frame,
    PredictionCache,
    build_frames,
    checkpoint_schedule,
    frames_to_tables,
    write_tables,
)
from primes_in_intervals.cumulative import load as load_cumulative
from primes_in_intervals.cumulative import run as run_cumulative
from primes_in_intervals.dataio import (
    DB_PATH,
    StorageConflictError,
    ensure_tables,
    max_primes,
    provenance_of,
    retrieve,
    save,
    set_db,
    show_table,
    table_width,
)
from primes_in_intervals.discrepancies import (
    DiscrepancyScores,
    discrepancy,
    global_range,
    mass_report,
    tail_bound,
    tail_bound_local,
)
from primes_in_intervals.display import display
from primes_in_intervals.intervals import (
    COUNTER_VERSION,
    Dataset,
    MetaDict,
    anyIntervals,
    anyIntervals_cp,
    disjoint,
    disjoint_cp,
    intervals,
    overlap,
    overlap_cp,
    overlap_extension,
    prime_start,
    prime_start_cp,
    zeros,
)
from primes_in_intervals.plotting import (
    STYLE,
    animate_cumulative,
    animate_distribution,
    distribution_axes_limits,
    plot_cumulative_frame,
    plot_discrepancies,
    plot_distribution_frame,
    plot_means,
    plot_residuals,
    save_figure,
    save_gif,
    save_mp4,
)
from primes_in_intervals.predictions import (
    DEFAULT_MODELS,
    FORMULA_VERSION,
    MODELS,
    MS,
    IntegratedPrediction,
    averaged_parameter,
    binom_pmf,
    binomial_averaged,
    binomial_constant,
    eta,
    frei,
    frei_alt,
    integrated,
    integrated_corrected,
    integrated_correction,
    integrated_poisson,
    local_corrected,
    parameter_validity,
    poisson_pmf,
    predict_all,
    shift_constant,
    shifted_parameter,
)
from primes_in_intervals.quadrature import (
    QuadratureResult,
    QuadratureSettings,
    log_average,
    mp_log_average,
)
from primes_in_intervals.reference import (
    first_moment_identity,
    is_prime_trial,
    overlap_cp_reference,
    overlap_reference,
    overlap_reference_segment,
    prime_pi_table,
    prime_table,
    prime_table_segment,
)
from primes_in_intervals.serialize import (
    dataset_from_json,
    dataset_to_json,
    read_dataset_json,
    write_dataset_json,
)
from primes_in_intervals.sieve import next_prime, postponed_sieve, prime_pi
from primes_in_intervals.statistics import analyze, dictionary_sort, dictionary_statistics
from primes_in_intervals.transforms import extract, nest, partition, unpartition

try:  # optional: lets `pii.dfi.export(df, 'table.png')` work as in the examples
    import dataframe_image as dfi  # noqa: F401
except ImportError:  # pragma: no cover - exercised only without the extra
    dfi = None

__version__ = "1.1.0"

__all__ = [
    "COMPARISON_LABEL",
    "COUNTER_VERSION",
    "DB_PATH",
    "DEFAULT_MODELS",
    "DEFAULT_OVERLAY_FROM",
    "FORMULA_VERSION",
    "MODELS",
    "MS",
    "STYLE",
    "Dataset",
    "DiscrepancyScores",
    "Frame",
    "IntegratedPrediction",
    "MetaDict",
    "PredictionCache",
    "QuadratureResult",
    "QuadratureSettings",
    "StorageConflictError",
    "analyze",
    "animate_cumulative",
    "animate_distribution",
    "anyIntervals",
    "anyIntervals_cp",
    "averaged_parameter",
    "binom_pmf",
    "binomial_averaged",
    "binomial_constant",
    "build_frames",
    "checkpoint_schedule",
    "compare",
    "dataset_from_json",
    "dataset_to_json",
    "dfi",
    "dictionary_sort",
    "dictionary_statistics",
    "discrepancy",
    "disjoint",
    "disjoint_cp",
    "display",
    "distribution_axes_limits",
    "ensure_tables",
    "eta",
    "extract",
    "first_moment_identity",
    "frames_to_tables",
    "frei",
    "frei_alt",
    "global_range",
    "integrated",
    "integrated_corrected",
    "integrated_correction",
    "integrated_poisson",
    "intervals",
    "is_prime_trial",
    "load_cumulative",
    "local_corrected",
    "log_average",
    "mass_report",
    "max_primes",
    "mp_log_average",
    "nest",
    "next_prime",
    "overlap",
    "overlap_cp",
    "overlap_cp_reference",
    "overlap_extension",
    "overlap_reference",
    "overlap_reference_segment",
    "parameter_validity",
    "partition",
    "plot_cumulative_frame",
    "plot_discrepancies",
    "plot_distribution_frame",
    "plot_means",
    "plot_residuals",
    "poisson_pmf",
    "postponed_sieve",
    "predict_all",
    "prime_pi",
    "prime_pi_table",
    "prime_start",
    "prime_start_cp",
    "prime_table",
    "prime_table_segment",
    "provenance_of",
    "read_dataset_json",
    "retrieve",
    "run_cumulative",
    "save",
    "save_figure",
    "save_gif",
    "save_mp4",
    "score",
    "set_db",
    "shift_constant",
    "shifted_parameter",
    "show_table",
    "table_width",
    "tail_bound",
    "tail_bound_local",
    "unpartition",
    "winners",
    "write_dataset_json",
    "write_tables",
    "zeros",
]
