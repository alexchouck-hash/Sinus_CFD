#!/usr/bin/env python3
"""CLI: process one NasalSeg case into airway mask + STL surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: repo_root/src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.pipeline import (  # noqa: E402
    DEFAULT_AIRWAY_LABELS,
    DEFAULT_ALL_AIRWAY_LABELS,
    process_case,
    resolve_nasalseg_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="P001",
        help="NasalSeg case id (default: P001)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "NasalSeg",
        help="Path to NasalSeg root (contains images/ and labels/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/<case>)",
    )
    parser.add_argument(
        "--mask-source",
        choices=("labels", "hu", "labels_and_hu"),
        default="labels",
        help="How to build the airway mask (default: labels)",
    )
    parser.add_argument(
        "--include-sinuses",
        action="store_true",
        help="Include maxillary sinus labels (4,5) in label-based mask",
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=-400.0,
        help="Upper HU for air threshold (default: -400)",
    )
    args = parser.parse_args()

    image_path, label_path = resolve_nasalseg_case(args.data_root, args.case)
    labels = DEFAULT_ALL_AIRWAY_LABELS if args.include_sinuses else DEFAULT_AIRWAY_LABELS
    out = args.output_dir or (REPO_ROOT / "outputs" / args.case)

    process_case(
        image_path=image_path,
        label_path=label_path,
        output_dir=out,
        case_id=args.case,
        mask_source=args.mask_source,
        airway_labels=labels,
        hu_max=args.hu_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
