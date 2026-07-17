#!/usr/bin/env python3
"""CLI: process one NasalSeg case into airway mask + STL + BC setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: repo_root/src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.physiology import PatientBreathing  # noqa: E402
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
        choices=("labels", "hu", "labels_and_hu", "nnunet"),
        default="labels",
        help="How to build the airway mask (default: labels)",
    )
    parser.add_argument(
        "--nnunet-dataset-id", type=int, default=501,
        help="nnU-Net dataset id (default: 501 = NasalSeg)",
    )
    parser.add_argument(
        "--nnunet-configuration", default="3d_fullres",
        help="nnU-Net configuration (default: 3d_fullres)",
    )
    parser.add_argument("--nnunet-fold", type=int, default=0, help="nnU-Net fold (default: 0)")
    parser.add_argument(
        "--nnunet-trainer", default="nnUNetTrainer_250epochs",
        help="nnU-Net trainer class name (default: nnUNetTrainer_250epochs)",
    )
    parser.add_argument(
        "--nnunet-plans", default="nnUNetPlans",
        help="nnU-Net plans identifier (default: nnUNetPlans)",
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
    # Physiology / patient matching
    parser.add_argument(
        "--tidal-volume-L",
        type=float,
        default=0.50,
        help="Tidal volume in liters (default: 0.50 = typical resting adult)",
    )
    parser.add_argument(
        "--respiratory-rate",
        type=float,
        default=12.0,
        help="Breaths per minute (default: 12)",
    )
    parser.add_argument(
        "--inspiratory-fraction",
        type=float,
        default=1.0 / 3.0,
        help="Inspiration as fraction of breath period (default: 1/3 ≈ I:E 1:2)",
    )
    parser.add_argument(
        "--inspiratory-time-s",
        type=float,
        default=None,
        help="Override Ti in seconds (else RR × inspiratory fraction)",
    )
    parser.add_argument(
        "--weight-kg",
        type=float,
        default=None,
        help="If set, scale VT ≈ 7 mL/kg (patient matching)",
    )
    parser.add_argument(
        "--left-flow-fraction",
        type=float,
        default=0.5,
        help="Fraction of total nasal flow through left nostril (default: 0.5)",
    )
    parser.add_argument(
        "--no-bcs",
        action="store_true",
        help="Skip boundary-condition JSON / OpenFOAM sketch",
    )
    args = parser.parse_args()

    image_path, label_path = resolve_nasalseg_case(args.data_root, args.case)
    labels = DEFAULT_ALL_AIRWAY_LABELS if args.include_sinuses else DEFAULT_AIRWAY_LABELS
    out = args.output_dir or (REPO_ROOT / "outputs" / args.case)

    right_frac = 1.0 - args.left_flow_fraction
    if args.weight_kg is not None:
        breathing = PatientBreathing.from_weight_kg(
            weight_kg=args.weight_kg,
            respiratory_rate_per_min=args.respiratory_rate,
            inspiratory_fraction=args.inspiratory_fraction,
            inspiratory_time_s=args.inspiratory_time_s,
            left_nostril_flow_fraction=args.left_flow_fraction,
            right_nostril_flow_fraction=right_frac,
            patient_id=args.case,
        )
    else:
        breathing = PatientBreathing(
            patient_id=args.case,
            tidal_volume_L=args.tidal_volume_L,
            respiratory_rate_per_min=args.respiratory_rate,
            inspiratory_fraction=args.inspiratory_fraction,
            inspiratory_time_s=args.inspiratory_time_s,
            left_nostril_flow_fraction=args.left_flow_fraction,
            right_nostril_flow_fraction=right_frac,
        )

    process_case(
        image_path=image_path,
        label_path=label_path,
        output_dir=out,
        case_id=args.case,
        mask_source=args.mask_source,
        airway_labels=labels,
        hu_max=args.hu_max,
        breathing=breathing,
        write_bcs=not args.no_bcs,
        nnunet_dataset_id=args.nnunet_dataset_id,
        nnunet_configuration=args.nnunet_configuration,
        nnunet_fold=args.nnunet_fold,
        nnunet_trainer=args.nnunet_trainer,
        nnunet_plans=args.nnunet_plans,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
