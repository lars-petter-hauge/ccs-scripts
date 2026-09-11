import argparse
from pathlib import Path


def build_argument_parser(skip_type: bool = False):
    parser = argparse.ArgumentParser(
        description="Extract dynamic plume targets from UNSMRY using a YAML config."
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Validate the configuration and exit without running the job.",
    )
    required_group = parser.add_argument_group("required named arguments")
    required_group.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path if not skip_type else str,
        help="Path to plume dynamic config YAML.",
    )
    required_group.add_argument(
        "-cn",
        "--case_name",
        required=True,
        type=Path if not skip_type else str,
        help="Case base path (accepted for workflow compatibility).",
    )
    parser.add_argument(
        "--egrid",
        default=None,
        type=Path if not skip_type else str,
        help="Path to EGRID file (overrides YAML).",
    )
    parser.add_argument(
        "--unrst",
        default=None,
        type=Path if not skip_type else str,
        help="Path to UNRST file (overrides YAML).",
    )
    parser.add_argument(
        "--output_date",
        default=None,
        help="Target report date in YYYY-MM-DD format. Overrides step when provided.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="UNRST time-step index, default -1 (last)",
    )
    return parser
