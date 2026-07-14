#!/usr/bin/env python3
"""Process a whole-head CT: head solid + airway + BCs + optional flow field."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.flow_field import compute_flow_field  # noqa: E402
from sinus_cfd.physiology import PatientBreathing  # noqa: E402
from sinus_cfd.whole_head import process_whole_head  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "data" / "VisibleHuman_Head" / "VHFCT1mm_Head.nrrd",
    )
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/<case>",
    )
    p.add_argument("--tidal-volume-L", type=float, default=0.50)
    p.add_argument("--respiratory-rate", type=float, default=12.0)
    p.add_argument("--skip-flow", action="store_true", help="Skip velocity field")
    p.add_argument("--flow-iterations", type=int, default=400)
    args = p.parse_args()

    if not args.image.is_file():
        raise SystemExit(
            f"Missing image: {args.image}\n"
            "Run: py -3.12 scripts/download_visible_human_head.py"
        )

    out = args.output_dir or (REPO_ROOT / "outputs" / args.case)
    breathing = PatientBreathing(
        patient_id=args.case,
        tidal_volume_L=args.tidal_volume_L,
        respiratory_rate_per_min=args.respiratory_rate,
    )

    process_whole_head(
        image_path=args.image,
        output_dir=out,
        case_id=args.case,
        breathing=breathing,
    )

    if not args.skip_flow:
        print(f"[{args.case}] computing flow field…")
        compute_flow_field(
            airway_mask_path=out / f"{args.case}_airway_mask.nrrd",
            boundary_json_path=out / f"{args.case}_boundary_conditions.json",
            output_dir=out,
            case_id=args.case,
            breathing=breathing,
            pressure_iterations=args.flow_iterations,
            n_streamline_seeds=48,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
