#!/usr/bin/env python3
"""Compute approximate airflow velocity field for a processed case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.flow_field import compute_flow_field  # noqa: E402
from sinus_cfd.physiology import PatientBreathing  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="P001")
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory containing <case>/ results from process_case.py",
    )
    p.add_argument("--tidal-volume-L", type=float, default=0.50)
    p.add_argument("--respiratory-rate", type=float, default=12.0)
    p.add_argument("--iterations", type=int, default=350)
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    mask = case_dir / f"{args.case}_airway_mask.nrrd"
    bc = case_dir / f"{args.case}_boundary_conditions.json"
    if not mask.is_file():
        raise SystemExit(f"Missing mask: {mask}  (run process_case.py first)")
    if not bc.is_file():
        raise SystemExit(f"Missing BCs: {bc}  (run process_case.py first)")

    breathing = PatientBreathing(
        tidal_volume_L=args.tidal_volume_L,
        respiratory_rate_per_min=args.respiratory_rate,
    )
    compute_flow_field(
        airway_mask_path=mask,
        boundary_json_path=bc,
        output_dir=case_dir,
        case_id=args.case,
        breathing=breathing,
        pressure_iterations=args.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
