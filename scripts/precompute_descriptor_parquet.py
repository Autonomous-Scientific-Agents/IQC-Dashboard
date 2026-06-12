#!/usr/bin/env python3
"""Precompute IQC dashboard descriptors from a reaction JSON file."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iqc_dashboard.descriptor_precompute import (  # noqa: E402
    build_precomputed_descriptor_dataframe,
    default_worker_count,
    read_reaction_json,
)


DEFAULT_INPUT = Path(
    "/Users/keceli/Library/CloudStorage/Box-Box/Project_II_2024_2026/"
    "Project_5_Benchmarking_computational_methods_for_TM_structure_optimization/"
    "data/sarah_reaction_data.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read reaction JSON, precompute all descriptor-tab values, and write "
            "a dashboard-ready Parquet file containing the original fields."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Reaction JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Parquet path (default: <input stem>_descriptors.parquet)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Worker processes for descriptor calculation (default: CPU count minus one)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Rows sent to each worker per task batch (default: 8)",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=("zstd", "snappy", "gzip", "brotli", "none"),
        help="Parquet compression codec (default: zstd)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_descriptors.parquet")
    )

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be at least 1")
    if not input_path.is_file():
        raise SystemExit(f"Input JSON does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path} (use --overwrite)")

    start_time = time.perf_counter()
    reaction_df = read_reaction_json(input_path)
    print(f"Loaded {len(reaction_df):,} reaction rows from {input_path}")

    last_report = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_report
        if completed == total or completed - last_report >= 100:
            elapsed = time.perf_counter() - start_time
            rate = completed / elapsed if elapsed else 0.0
            print(
                f"Computed {completed:,}/{total:,} reactions "
                f"({completed / total:.1%}, {rate:.1f} rows/s)",
                flush=True,
            )
            last_report = completed

    output_df = build_precomputed_descriptor_dataframe(
        reaction_df,
        workers=args.workers,
        chunksize=args.chunksize,
        progress=report_progress,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    output_df.to_parquet(output_path, index=False, compression=compression)

    elapsed = time.perf_counter() - start_time
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {len(output_df):,} dashboard rows and {len(output_df.columns):,} "
        f"columns to {output_path} ({size_mb:.1f} MiB) in {elapsed / 60:.1f} minutes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
