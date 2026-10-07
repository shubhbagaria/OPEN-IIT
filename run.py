#!/usr/bin/env python3
"""
Phase 2 / Team B entry point.

    python run.py --data-dir csv_folder --out-dir out

Expects the nine required CSVs in one directory (copy the contents of
`shared/` and `ps1_fake_ptp/` into the same folder). annotated_ptp_sample.csv
is optional; when present it feeds annotation_check.md.

Writes to --out-dir:
    features_endcall.{parquet,csv}     Team C's main training table
    features_incall.{parquet,csv}      streaming table, one row per decision point
    features_postcall.{parquet,csv}    quarantined post-call signals
    labels.{parquet,csv}               label used for agent history; the
                                       label_source column says Phase 1 or fallback
    feature_dictionary.{csv,md}
    coverage_report.{md,json}
    leakage_audit.md
    annotation_check.{md,json}         only if annotated_ptp_sample.csv exists
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from ptp_features.audit import write_reports
from ptp_features.config import DEFAULT, Config
from ptp_features.pipeline import load_raw, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 2 feature store.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT.data_dir,
                        help="Directory holding the input CSVs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT.out_dir,
                        help="Where to write the feature store and reports.")
    parser.add_argument("--agent-history-pool", choices=("train", "all"),
                        default=DEFAULT.features.agent_history_pool,
                        help="Which accounts may contribute agent history. "
                             "'train' is the safe default.")
    parser.add_argument("--skip-reports", action="store_true",
                        help="Build features only, no audit.")
    args = parser.parse_args(argv)

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        label=DEFAULT.label,
        features=replace(
            DEFAULT.features, agent_history_pool=args.agent_history_pool
        ),
    )

    t0 = time.time()
    print(f"[1/3] Reading CSVs from {cfg.data_dir} ...")
    try:
        frames = load_raw(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    for name, frame in frames.items():
        print(f"      {name:<20} {len(frame):>9,} rows")

    print("[2/3] Building feature tables ...")
    outputs = run(cfg, frames)
    for name, table in outputs.items():
        print(f"      {name:<26} {len(table):>9,} rows x {table.shape[1]:>3} cols")

    if not args.skip_reports:
        print("[3/3] Writing audit reports ...")
        written = write_reports(frames, outputs, cfg)
        for name, path in written.items():
            print(f"      {name:<14} -> {path}")
    else:
        print("[3/3] Skipped.")

    print(f"\nDone in {time.time() - t0:.1f}s. Output in {cfg.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
