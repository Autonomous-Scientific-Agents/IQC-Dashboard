"""
Streamlit Cloud entry point.

This file is used by Streamlit Cloud to launch the application.
It imports and runs the main function from the package.
"""

import argparse

from iqc_dashboard.app import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="Path to a parquet/json file or directory (can be repeated).",
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(data_paths=args.data_path)
