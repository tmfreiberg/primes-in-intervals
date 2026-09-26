# Primes in intervals

How many of the intervals $(a, a + H]$ contain exactly $m$ primes? Counting them, for $a$ ranging over a window near some large $N$, gives an empirical distribution: a histogram of primes-per-interval. This project computes those distributions at scale, stores them, and compares them against three predictions, the naive one from Cramér's model and two refinements. The empirical distributions are consistently narrower than Cramér's model predicts, and the refinements track that bias.

**The full exposition is a Quarto book: [Biases in the distribution of primes in intervals](https://tmfreiberg.github.io/primes-in-intervals/).** It derives the predictions, walks through the code, and animates the distributions as the sample grows. This README covers installation and the command-line interface; the book is where the mathematics and the worked examples live.

## What's here

The mathematics is elementary to state. Fix an interval length $H$ and a large $N$. For overlapping intervals, count how many $a$ in a window about $N$ give exactly $m$ primes in $(a, a + H]$; there are analogous counts for disjoint intervals (left endpoints in an arithmetic progression) and for prime-starting intervals (left endpoints prime). Cramér's model treats each nearby integer as prime independently with probability $1/\log N$, which predicts a Binomial distribution. The data departs from that prediction in a specific direction, and the project's two refinements, written $F$ and $F^*$, add a second-order correction that follows the departure.

The computation leans on a single sliding-window pass over a prime sieve, so overlapping counts cost about twice a disjoint count and no more. Results are cached in SQLite, keyed by lower bound, upper bound, and interval length, so a run that takes an hour is paid for once. From stored counts the package builds distributions and summary statistics, attaches the three predictions, scores which prediction fits best, renders human-readable tables, and animates the distribution frame by frame as the window widens.

## Installation

Requires Python 3.10 or newer.

```
git clone https://github.com/tmfreiberg/primes-in-intervals.git
cd primes_in_intervals
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the package and its dependencies (NumPy, pandas, SciPy, SymPy, matplotlib) and puts two identical console commands on the path, `primes-in-intervals` and the short form `pii`.

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
docs/cli.md                command-line reference
tests/                     the test suite
data/                      a small SQLite database of precomputed results
images/static/             figures the book cannot generate at render time
```

## Development

The test suite runs with `pytest`:

```
pip install -e ".[dev]"
pytest
```

163 tests cover the counters, the transforms, the storage layer, the statistics, the predictions, the display code, and the CLI. The package is linted and type-checked with `ruff` and `mypy`.

## About

This is a rewrite of a 2023 project, undertaken as a self-directed exercise in number theory and Python. The original was a single script with an expository README; this version packages the code into modules with tests and a command-line interface, and moves the exposition into a Quarto book. The mathematical content is unchanged. The refinements $F$ and $F^*$ are the author's own; they are documented here and in the book but are not a published or peer-reviewed result.

Author: Tristan Freiberg.