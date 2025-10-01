#!/usr/bin/env python3
"""Create an HTML report from sweep results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fbqc_snspe.objectives import BandMetrics
from fbqc_snspe.report import render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report from sweep results")
    parser.add_argument("results", type=Path, help="CSV file with sweep results")
    parser.add_argument("--report", type=Path, default=Path("outputs/report.html"), help="Output HTML path")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to include")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    df_sorted = df.sort_values("objective")
    best_row = df_sorted.iloc[0]
    metrics = BandMetrics(
        delta_db_max=float(best_row["delta_db_max"]),
        mean_absorptance=float(best_row["mean_absorptance"]),
        worst_case_absorptance=float(best_row["worst_absorptance"]),
        band_ok=bool(best_row.get("band_ok", True)),
    )
    top = df_sorted.head(args.top)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    render_report(args.report, metrics=metrics, candidates=top)
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
