#!/usr/bin/env python3
"""
Head-to-head: does the trained nnU-Net actually beat the classical HU-threshold
baseline at reproducing NasalSeg airway labels?

For every case that has an nnU-Net prediction (a *.nii.gz label map, one per
case id, e.g. from `nnUNetv2_predict` or the fold_0/validation/ folder nnU-Net
writes automatically), this computes on the *same* cases and the *same*
ground-truth definition used in docs/stage1_segmentation_baseline.md:

  - nnU-Net merged-airway Dice : predicted labels {1,2,3} vs truth {1,2,3}
  - nnU-Net per-structure Dice : each label 1..5 separately
  - classical Dice            : build_hu_threshold_mask(hu) vs truth {1,2,3}

so the headline nnU-Net number is directly comparable to the ~0.25 classical
baseline. Predictions are resampled onto each case's native label grid with
nearest-neighbour before scoring, so a prediction produced on nnU-Net's
resampled spacing is still scored on the native voxels the baseline used.

Usage (after downloading trained predictions from Drive):
  py -3.12 scripts/compare_nnunet_vs_classical.py \
      --pred-dir data/nnUNet_results/.../fold_0/validation \
      --nasalseg-root data
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.metrics import dice_coefficient, labels_to_mask, per_label_dice  # noqa: E402
from sinus_cfd.pipeline import DEFAULT_AIRWAY_LABELS, build_hu_threshold_mask  # noqa: E402

STRUCTURE_NAMES = {
    1: "left_nasal_cavity",
    2: "right_nasal_cavity",
    3: "nasopharynx",
    4: "left_maxillary_sinus",
    5: "right_maxillary_sinus",
}


def _aligned_pred_array(truth_img: sitk.Image, pred_img: sitk.Image) -> np.ndarray:
    """
    Return the prediction as a (z, y, x) array on the truth's voxel grid.

    When the voxel array sizes match — always true across NasalSeg, since
    nnU-Net writes predictions on the native image grid and image/label sizes
    always agree — the intended correspondence is voxel-index-to-voxel-index,
    exactly what training used. We compare by index rather than by physical
    coordinates. This matters because ~14 NasalSeg label NRRDs carry a
    spacing/origin/direction in their header that disagrees with the paired
    image (e.g. a flipped z-direction or a 220 mm origin shift); a
    physical-coordinate resample would misalign the prediction against those
    and report a spurious ~0 Dice. Only when sizes genuinely differ do we fall
    back to a nearest-neighbour resample onto the truth grid.
    """
    if pred_img.GetSize() == truth_img.GetSize():
        return sitk.GetArrayFromImage(pred_img)
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(truth_img)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(rs.Execute(pred_img))


def _find_prediction(pred_dir: Path, cid: str) -> Path | None:
    for name in (f"{cid}.nii.gz", f"{cid}_0000.nii.gz", f"{cid}.nrrd"):
        p = pred_dir / name
        if p.is_file():
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Folder of nnU-Net prediction label maps (one *.nii.gz per case id)",
    )
    p.add_argument(
        "--nasalseg-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="NasalSeg root with images/ and labels/ (default: data)",
    )
    p.add_argument(
        "--hu-max",
        type=float,
        default=-350.0,
        help="Classical threshold (default -350, the baseline's best mean-Dice cutoff)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "nnunet_vs_classical.json",
    )
    args = p.parse_args()

    images_dir = args.nasalseg_root / "images"
    labels_dir = args.nasalseg_root / "labels"
    if not args.pred_dir.is_dir():
        print(f"ERROR: prediction dir not found: {args.pred_dir}", file=sys.stderr)
        return 1

    airway = list(DEFAULT_AIRWAY_LABELS)  # (1, 2, 3)

    per_case: dict[str, dict] = {}
    # Discover cases from the prediction folder, not the whole dataset — we only
    # score cases nnU-Net actually predicted (e.g. its held-out fold-0 val set).
    pred_files = sorted(
        list(args.pred_dir.glob("P*.nii.gz")) + list(args.pred_dir.glob("P*.nrrd"))
    )
    seen: set[str] = set()
    for pred_path in pred_files:
        cid = pred_path.name.split(".")[0].replace("_0000", "")
        if cid in seen:
            continue
        seen.add(cid)
        label_path = labels_dir / f"{cid}_seg.nrrd"
        image_path = images_dir / f"{cid}_img.nrrd"
        if not label_path.is_file() or not image_path.is_file():
            print(f"  skip {cid}: no matching NasalSeg image/label")
            continue

        truth_img = sitk.ReadImage(str(label_path))
        truth = sitk.GetArrayFromImage(truth_img)
        hu = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))

        pred_img = sitk.ReadImage(str(_find_prediction(args.pred_dir, cid)))
        pred = _aligned_pred_array(truth_img, pred_img)

        nnunet_airway = dice_coefficient(
            labels_to_mask(pred, airway), labels_to_mask(truth, airway)
        )
        classical = build_hu_threshold_mask(hu, hu_max=args.hu_max)
        classical_airway = dice_coefficient(classical, labels_to_mask(truth, airway))
        struct = per_label_dice(pred, truth, STRUCTURE_NAMES.keys())

        per_case[cid] = {
            "nnunet_airway_dice": nnunet_airway,
            "classical_airway_dice": classical_airway,
            "nnunet_per_structure": {STRUCTURE_NAMES[k]: v for k, v in struct.items()},
        }
        print(
            f"  {cid}: nnU-Net={nnunet_airway:.3f}  classical={classical_airway:.3f}  "
            f"(Δ={nnunet_airway - classical_airway:+.3f})"
        )

    if not per_case:
        print("ERROR: no cases scored — check --pred-dir contents", file=sys.stderr)
        return 1

    cids = list(per_case)
    nn = [per_case[c]["nnunet_airway_dice"] for c in cids]
    cl = [per_case[c]["classical_airway_dice"] for c in cids]

    def _stats(xs: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(xs),
            "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    struct_means = {
        STRUCTURE_NAMES[k]: statistics.mean(
            per_case[c]["nnunet_per_structure"][STRUCTURE_NAMES[k]] for c in cids
        )
        for k in STRUCTURE_NAMES
    }

    print(f"\n[compare] {len(cids)} cases scored")
    print(f"  nnU-Net  airway Dice: {_stats(nn)['mean']:.3f} ± {_stats(nn)['stdev']:.3f}")
    print(f"  classical airway Dice: {_stats(cl)['mean']:.3f} ± {_stats(cl)['stdev']:.3f}")
    print(f"  mean improvement: {statistics.mean(nn) - statistics.mean(cl):+.3f}")
    print("\n  nnU-Net per-structure Dice (mean):")
    for name, v in struct_means.items():
        print(f"    {name:22s} {v:.3f}")

    report = {
        "pred_dir": str(args.pred_dir),
        "n_cases": len(cids),
        "ground_truth_airway_labels": airway,
        "classical_hu_max": args.hu_max,
        "nnunet_airway": _stats(nn),
        "classical_airway": _stats(cl),
        "mean_improvement": statistics.mean(nn) - statistics.mean(cl),
        "nnunet_per_structure_mean": struct_means,
        "per_case": per_case,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[compare] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
