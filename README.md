# Primes in intervals

How many of the intervals $(a, a + H]$ contain exactly $m$ primes? Counting them, for $a$ ranging over a window near some large $N$, gives an empirical distribution: a histogram of primes-per-interval. This project computes those distributions at scale, stores them, and compares them against predictions: those of Cramér's random model, and a refinement with a second-order correction. The empirical distributions are consistently narrower than Cramér's model predicts, and the refinement tracks that bias.

**The full exposition is a Quarto book: [Biases in the distribution of primes in intervals](https://tmfreiberg.github.io/primes-in-intervals/).** It derives the predictions, walks through the code, and animates the distributions as the sample grows. This README covers installation and the command-line interface; the book is where the mathematics and the worked examples live.

## What's here

The mathematics is elementary to state. Fix an interval length $H$ and a large $N$. For overlapping intervals, count how many $a$ in a window about $N$ give exactly $m$ primes in $(a, a + H]$; there are analogous counts for disjoint intervals (left endpoints in an arithmetic progression) and for prime-starting intervals (left endpoints prime). Cramér's model treats each integer $t$ as prime independently with probability $1/\log t$, which predicts a Binomial distribution, and in the limit a Poisson one. The data departs from that prediction in a specific direction. The refinement, written $F$, averages a corrected Poisson probability over the window, and its second-order correction follows the departure; $F_0$, the same average without the correction, shows what the correction contributes.

The computation leans on a single sliding-window pass over a prime sieve, so overlapping counts cost about twice a disjoint count and no more. Results are cached in SQLite, keyed by lower bound, upper bound, and interval length, so a run that takes an hour is paid for once. From stored counts the package builds distributions and summary statistics, attaches the predictions, measures how far each one is from the data, renders human-readable tables, and animates the distribution frame by frame as the window widens.

## Installation

Requires Python 3.12 or newer.

```
git clone https://github.com/tmfreiberg/primes-in-intervals.git
cd primes-in-intervals
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the package and its dependencies (NumPy, pandas, SciPy, matplotlib, mpmath, Jinja2) and puts two identical console commands on the path, `primes-in-intervals` and the short form `pii`.

## Command-line interface

Every public function in the package is reachable from the shell, so any claim in the book can be checked with a one-line command. A few examples:

```
pii disjoint "2*10**6" "3*10**6" 100        # count primes in disjoint intervals
pii prime-pi 1 100                          # 25 primes in (1, 100]
pii intervals --help                        # options for the main entry point
```

Anywhere a number is expected, a small arithmetic expression is accepted, so `exp(17)`, `2e6`, and `10**7` all work and match the notation in the book. Quote any argument containing `*` or parentheses so the shell does not expand it.

The full command list, the database rules, and the expression grammar are documented in **[docs/cli.md](docs/cli.md)**.

## Repository layout

```
src/primes_in_intervals/   the package
  sieve.py                 the prime generator
  intervals.py             the interval counters (disjoint, overlap, prime-start)
  transforms.py            narrow, partition, nest
  reference.py             independent reference counts for validating the counters
  dataio.py                SQLite storage and retrieval, with conflict checks and provenance
  statistics.py            distributions and summary statistics
  predictions.py           the manuscript's predictions: F, F0, Q, and binomial approximations
  quadrature.py            the averaged integrals, with tolerances and error estimates
  discrepancies.py         discrepancy measures, tail bounds, mass reports
  comparisons.py           compare data to predictions and score them
  cumulative.py            the fixed-H experiment over starting points 1 <= n <= N
  display.py               human-readable tables
  plotting.py              distribution frames, residual and discrepancy plots, animations
  serialize.py             JSON round-tripping for datasets
  cli.py                   the command-line interface
book/                      the Quarto book source
scripts/book_animations.py regenerates the book's animations
scripts/paper_m0/          the M = 0 runs behind the paper's numerical section
docs/cli.md                command-line reference
tests/                     the test suite
data/                      a small SQLite database of precomputed results
images/                    figures the book cannot generate at render time
```

## Development

The development environment is installed from the lock file with [uv](https://docs.astral.sh/uv/), and the checks run inside it:

```
uv sync --locked --extra dev
uv run pytest
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src
```

(Without uv: `pip install -e ".[dev]"`, then the same commands without `uv run`.) The tests cover the counters, the transforms, the storage layer, the statistics, the predictions, the display code, and the CLI. The same checks, and a full render of the book, run on every pull request.

## How to cite

Please cite the software through Zenodo. For the project as a whole, use the
DOI that always resolves to the latest version:

> Freiberg, T. *Primes in intervals*. Zenodo. https://doi.org/10.5281/zenodo.22982648

To cite the exact code behind a set of results, use the version DOI:

- version 1.2.0: https://doi.org/10.5281/zenodo.23505688
- version 1.1.0: https://doi.org/10.5281/zenodo.22982661
- version v2023.08, the code as committed on 2023-08-05:
  https://doi.org/10.5281/zenodo.22982649 (Software Heritage:
  `swh:1:rev:2a0b2e0ff8a137af1c325491901f7993cc95778b`)

```bibtex
@software{freiberg_primes_in_intervals,
  author    = {Freiberg, Tristan},
  title     = {Primes in intervals},
  publisher = {Zenodo},
  version   = {1.2.0},
  year      = {2026},
  doi       = {10.5281/zenodo.23505688},
  url       = {https://doi.org/10.5281/zenodo.23505688}
}
```

GitHub's "Cite this repository" button reads the same information from
`CITATION.cff`.

For the mathematics behind the predictions, cite the paper:

> Freiberg, T. *Biases in the distribution of primes in short intervals*. arXiv:2609.33692 [math.NT], 2026. https://doi.org/10.48550/arXiv.2609.33692

## About

This is a rewrite of a 2023 project, undertaken as a self-directed exercise in number theory and Python. The original was a single script with an expository README; this version packages the code into modules with tests and a command-line interface, and moves the exposition into a Quarto book. In 2026 the predictions were brought into line with the author's paper on the subject. The refinement $F$ is the author's own; its derivation, conditional on a uniform form of the Hardy-Littlewood prime tuples conjecture, is in a preprint that has not been peer reviewed: [arXiv:2609.33692](https://arxiv.org/abs/2609.33692).

Author: Tristan Freiberg.