#!/usr/bin/env python3
"""
Stage 2 end-to-end: CT -> segmentation -> per-side airway geometry report.

Produces the clinical geometry summary the roadmap's §5 calls for without any
CFD: left/right nasal-cavity volume, minimal cross-sectional area (MCA) and
its anterior-posterior location, and the L/R MCA ratio that flags a
unilaterally obstructed airway.

Label source:
  --mask-source labels : NasalSeg expert labels (default; needs a labeled case)
  --mask-source nnunet : trained nnU-Net prediction (works on any head CT with
                         no expert labels — see docs/nnunet_colab_training.md)

Usage:
  # NasalSeg labeled case
  py -3.12 scripts/geometry_report.py --case P001 --data-root data

  # Any CT via the trained model
  py -3.12 scripts/geometry_report.py --image path/to/ct.nii.gz --mask-source nnunet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.passage_metrics import analyze_bilateral  # noqa: E402
from sinus_cfd.pipeline import resolve_nasalseg_case  # noqa: E402


def _plot(report, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for side_key, color in (("left", "tab:blue"), ("right", "tab:red")):
        side = report.__dict__[side_key]
        if not side["present"] or not side["area_profile"]:
            continue
        ap = [p["ap_mm"] for p in side["area_profile"]]
        area = [p["area_mm2"] for p in side["area_profile"]]
        ax.plot(ap, area, color=color, label=f"{side_key} (vol {side['volume_ml']:.1f} mL)")
        ax.scatter(
            [side["mca_ap_position_mm"]], [side["mca_mm2"]],
            color=color, marker="v", s=90, zorder=5,
            label=f"{side_key} MCA {side['mca_mm2']:.0f} mm²",
        )
    ax.set_xlabel("distance from anterior naris (mm)")
    ax.set_ylabel("cross-sectional area (mm²)")
    ax.set_title(f"{report.case_id} — nasal cross-sectional area (source: {report.mask_source})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=None, help="NasalSeg case id (e.g. P001)")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--image", type=Path, default=None, help="Explicit CT path (overrides --case)")
    p.add_argument("--label", type=Path, default=None, help="Explicit label path (for mask-source=labels)")
    p.add_argument("--mask-source", choices=("labels", "nnunet"), default="labels")
    p.add_argument("--output-dir", type=Path, default=None)
    # nnU-Net inference knobs (match the trained Colab model defaults)
    p.add_argument("--nnunet-dataset-id", type=int, default=501)
    p.add_argument("--nnunet-configuration", default="3d_fullres")
    p.add_argument("--nnunet-fold", type=int, default=0)
    p.add_argument("--nnunet-trainer", default="nnUNetTrainer_250epochs")
    p.add_argument("--nnunet-plans", default="nnUNetPlans")
    args = p.parse_args()

    # Resolve image + (optional) label paths.
    if args.image is not None:
        image_path = args.image
        label_path = args.label
        case_id = args.case or image_path.stem.split(".")[0].replace("_img", "").replace("_0000", "")
    elif args.case is not None:
        image_path, label_path = resolve_nasalseg_case(args.data_root, args.case)
        case_id = args.case
    else:
        print("ERROR: pass --case or --image", file=sys.stderr)
        return 1

    image = sitk.ReadImage(str(image_path))
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())

    if args.mask_source == "labels":
        if label_path is None:
            print("ERROR: --mask-source labels needs a label file (--label or a NasalSeg --case)", file=sys.stderr)
            return 1
        label_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    else:  # nnunet
        from sinus_cfd.nnunet_infer import predict_labels

        label_zyx = predict_labels(
            image,
            dataset_id=args.nnunet_dataset_id,
            configuration=args.nnunet_configuration,
            fold=args.nnunet_fold,
            trainer=args.nnunet_trainer,
            plans=args.nnunet_plans,
        )

    report = analyze_bilateral(
        label_zyx=label_zyx,
        spacing_xyz=spacing_xyz,
        case_id=case_id,
        mask_source=args.mask_source,
    )

    out_dir = args.output_dir or (REPO_ROOT / "outputs" / case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{case_id}_geometry_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    plot_path = out_dir / f"{case_id}_area_profile.png"
    _plot(report, plot_path)

    L, R = report.left, report.right
    print(f"[{case_id}] geometry report (source: {report.mask_source})")
    print(f"  left  : vol {L['volume_ml']:6.2f} mL   MCA {L['mca_mm2']:7.1f} mm²  @ {L['mca_ap_position_mm']:5.1f} mm from naris")
    print(f"  right : vol {R['volume_ml']:6.2f} mL   MCA {R['mca_mm2']:7.1f} mm²  @ {R['mca_ap_position_mm']:5.1f} mm from naris")
    if not np.isnan(report.mca_ratio):
        print(f"  L/R MCA ratio: {report.mca_ratio:.2f}  (more obstructed: {report.more_obstructed_side})")
    for n in report.notes + L.get("notes", []) + R.get("notes", []):
        print(f"  note: {n}")
    print(f"  wrote {json_path.name}, {plot_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
