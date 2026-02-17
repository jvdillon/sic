#!/usr/bin/env python3
"""Generate matplotlib code to plot accuracy from .log files.

Produces a 4x2 grid: rows = {test, test halt, train, train halt},
columns = {cell accuracy, puzzle accuracy}.

Parses Step lines of the form::

    Step <N>  Test <cell>% / <puzzle>% (halt: <cell>%/<puzzle>%)
      ... TrainAcc <cell>%/<puzzle>% (halt: <cell>%/<puzzle>%)

Usage:
    python plot_logs.py x07f.log x07m.log
    python plot_logs.py /path/to/*.log
"""

from pathlib import Path
from typing import NamedTuple

import re
import sys


_PATTERN = re.compile(
    r"Step\s+(\d+)\s+"
    r"Test\s+([\d.]+)%\s*/\s*([\d.]+)%\s+"
    r"\(halt:\s*([\d.]+)%/\s*([\d.]+)%\)\s+"
    r".*?"
    r"TrainAcc\s+([\d.]+)%/\s*([\d.]+)%\s+"
    r"\(halt:\s*([\d.]+)%/\s*([\d.]+)%\)"
)


class LogData(NamedTuple):
    steps: tuple[int, ...]
    test_cell: tuple[float, ...]
    test_puzzle: tuple[float, ...]
    test_halt_cell: tuple[float, ...]
    test_halt_puzzle: tuple[float, ...]
    train_cell: tuple[float, ...]
    train_puzzle: tuple[float, ...]
    train_halt_cell: tuple[float, ...]
    train_halt_puzzle: tuple[float, ...]


_PLOTS = [
    (
        "Test Cell",
        "Test Puzzle",
        "test_cell",
        "test_puzzle",
    ),
    (
        "Test Halt Cell",
        "Test Halt Puzzle",
        "test_halt_cell",
        "test_halt_puzzle",
    ),
    (
        "Train Cell",
        "Train Puzzle",
        "train_cell",
        "train_puzzle",
    ),
    (
        "Train Halt Cell",
        "Train Halt Puzzle",
        "train_halt_cell",
        "train_halt_puzzle",
    ),
]


def extract_data(log_path: Path) -> LogData:
    steps: list[int] = []
    tc, tp, thc, thp = ([] for _ in range(4))
    rc, rp, rhc, rhp = ([] for _ in range(4))
    with log_path.open() as f:
        for line in f:
            m = _PATTERN.search(line)
            if not m:
                continue
            steps.append(int(m.group(1)))
            tc.append(float(m.group(2)))
            tp.append(float(m.group(3)))
            thc.append(float(m.group(4)))
            thp.append(float(m.group(5)))
            rc.append(float(m.group(6)))
            rp.append(float(m.group(7)))
            rhc.append(float(m.group(8)))
            rhp.append(float(m.group(9)))
    return LogData(
        tuple(steps),
        tuple(tc),
        tuple(tp),
        tuple(thc),
        tuple(thp),
        tuple(rc),
        tuple(rp),
        tuple(rhc),
        tuple(rhp),
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    logs = [Path(p) for p in sys.argv[1:]]
    data = {log.stem: extract_data(log) for log in logs}
    data = {k: v for k, v in data.items() if v.steps}

    cell_fields = ("test_cell", "test_halt_cell", "train_cell", "train_halt_cell")
    puzzle_fields = (
        "test_puzzle",
        "test_halt_puzzle",
        "train_puzzle",
        "train_halt_puzzle",
    )
    cell_min = int(min(min(getattr(v, f)) for v in data.values() for f in cell_fields))
    puzzle_min = int(
        min(min(getattr(v, f)) for v in data.values() for f in puzzle_fields)
    )

    data_lines = "\n".join(
        f"    {name!r}: {tuple(vals)}," for name, vals in data.items()
    )
    field_indices = {f: i for i, f in enumerate(LogData._fields)}
    plot_rows = "\n".join(
        f"    ({field_indices[cf]}, {field_indices[pf]}, {ct!r}, {pt!r}),"
        for ct, pt, cf, pf in _PLOTS
    )
    print(f"""\
import matplotlib.pyplot as plt

data = {{
{data_lines}
}}

plots = [
{plot_rows}
]
ylim_home = (({cell_min}, 100), ({puzzle_min}, 100))  # (cell, puzzle)
ylim_zoom = ((98, 100), (40, 100))

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

for row, (ci, pi, ct, pt) in enumerate(plots):
    for col, (i, t) in enumerate([(ci, ct), (pi, pt)]):
        ax = axes[row][col]
        for name, vals in data.items():
            ax.plot(vals[0], vals[i], 'o-', label=name, markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(t)
        ax.legend(loc='lower right')
        ymin = ylim_home[col][0]
        ax.set_ylim(ymin, 100)
        ax.set_yticks(range(ymin, 101, 1))
        ax.set_yticklabels([str(i) if i % 5 == 0 else '' for i in range(ymin, 101, 1)])
        ax.grid(True, alpha=0.3)
        xticks = list(range(0, max(max(s[0]) for s in data.values()) + 2000, 2000))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) if x % 4000 == 0 else '' for x in xticks])

plt.tight_layout()

# Push ylim_home as "Home" in the nav stack, then zoom to preset range.
fig.canvas.draw()
if (tb := fig.canvas.toolbar) is not None:
    tb.push_current()  # home = ylim_home view
for row in range(len(plots)):
    for col in range(2):
        axes[row][col].set_ylim(*ylim_zoom[col])
fig.canvas.draw()
if (tb := fig.canvas.toolbar) is not None:
    tb.push_current()  # current = ylim_zoom view

plt.show()""")


if __name__ == "__main__":
    main()
