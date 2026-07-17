#!/usr/bin/env python3
"""
Stage 1 validation: how well does classical HU threshold + region-grow
reproduce the expert NasalSeg airway labels?

For each case, builds a binary "nasal airway" mask two ways:
  - ground truth: NasalSeg labels 1,2,3 (L/R nasal cavity + nasopharynx)
  - prediction:   HU threshold + morphological closing + gap-bridging
                  region-grow (sinus_cfd.pipeline.build_hu_threshold_mask)

then reports the Dice coefficient at each of several HU thresholds, so a
per-scan or per-scanner threshold can be chosen with evidence rather than
guessing. This is the "sensitivity check" called for in
docs/architecture_and_roadmap.md section 2.

Usage (from repo root):
  py -3.12 scripts/evaluate_nasalseg_dice.py --n-cases 20
  py -3.12 scripts/evaluate_nasalseg_dice.py --n-cases all --thresholds -350,-400,-450,-500,-550,-600
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.pipeline import (  # noqa: E402
    DEFAULT_AIRWAY_LABELS,
    DEFAULT_ALL_AIRWAY_LABELS,
    build_hu_threshold_mask,
)

DEFAULT_THRESHOLDS = (-350.0, -400.0, -450.0, -500.0, -550.0, -600.0)


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    denom = int(pred.sum()) + int(truth.sum())
    if denom == 0:
        return 1.0  # both empty: trivially agree
    intersection = int((pred & truth).sum())
    return 2.0 * intersection / denom


def list_cases(nasalseg_root: Path) -> list[str]:
    images_dir = nasalseg_root / "images"
    labels_dir = nasalseg_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {labels_dir}. "
            "Run: py -3.12 scripts/download_nasalseg.py"
        )
    cases = []
    for img_path in sorted(images_dir.glob("P*_img.nrrd")):
        cid = img_path.stem.replace("_img", "")
        if (labels_dir / f"{cid}_seg.nrrd").is_file():
            cases.append(cid)
    return cases


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nasalseg-root",
        type=Path,
        default=REPO_ROOT / "data" / "NasalSeg",
        help="Path to NasalSeg root (contains images/ and labels/)",
    )
    p.add_argument(
        "--n-cases",
        default="20",
        help="Number of cases to evaluate, or 'all' (default: 20, evenly spaced)",
    )
    p.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated HU upper cutoffs to sweep",
    )
    p.add_argument(
        "--include-sinuses",
        action="store_true",
        help="Ground truth = labels 1-5 (also maxillary sinuses), not just 1-3",
    )
    p.add_argument(
        "--min-component-voxels",
        type=int,
        default=200,
        help="Drop connected components smaller than this (default: 200)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "nasalseg_dice_report.json",
        help="Where to write the JSON report (a sibling .csv is also written)",
    )
    args = p.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    airway_labels = DEFAULT_ALL_AIRWAY_LABELS if args.include_sinuses else DEFAULT_AIRWAY_LABELS

    all_cases = list_cases(args.nasalseg_root)
    if not all_cases:
        print(f"ERROR: no cases found under {args.nasalseg_root}", file=sys.stderr)
        return 1

    if args.n_cases == "all":
        cases = all_cases
    else:
        n = min(int(args.n_cases), len(all_cases))
        # Evenly spaced sample so we're not just looking at P001..P020
        idx = np.linspace(0, len(all_cases) - 1, n).round().astype(int)
        cases = [all_cases[i] for i in sorted(set(idx.tolist()))]

    print(f"[dice] {len(cases)}/{len(all_cases)} cases, thresholds={thresholds}")
    print(f"[dice] ground truth labels={list(airway_labels)}")

    images_dir = args.nasalseg_root / "images"
    labels_dir = args.nasalseg_root / "labels"

    per_case: dict[str, dict[str, float]] = {}
    for i, cid in enumerate(cases):
        img = sitk.ReadImage(str(images_dir / f"{cid}_img.nrrd"))
        hu = sitk.GetArrayFromImage(img)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_dir / f"{cid}_seg.nrrd")))
        truth = np.isin(lab, list(airway_labels))

        scores: dict[str, float] = {}
        for t in thresholds:
            pred = build_hu_threshold_mask(
                hu,
                hu_max=t,
                min_component_voxels=args.min_component_voxels,
            )
            scores[f"{t:g}"] = dice(pred, truth)
        per_case[cid] = scores
        print(f"  {cid} ({i + 1}/{len(cases)}): " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    summary = {
        str(t): {
            "mean": statistics.mean(per_case[c][f"{t:g}"] for c in cases),
            "stdev": statistics.pstdev(per_case[c][f"{t:g}"] for c in cases) if len(cases) > 1 else 0.0,
            "min": min(per_case[c][f"{t:g}"] for c in cases),
            "max": max(per_case[c][f"{t:g}"] for c in cases),
        }
        for t in thresholds
    }
    best_t = max(summary, key=lambda k: summary[k]["mean"])

    print("\n[dice] summary (mean ± stdev across cases):")
    for t in thresholds:
        s = summary[str(t)]
        print(f"  hu_max={t:g}: {s['mean']:.3f} ± {s['stdev']:.3f}  (min {s['min']:.3f}, max {s['max']:.3f})")
    print(f"\n[dice] best mean Dice: hu_max={best_t} ({summary[best_t]['mean']:.3f})")

    report = {
        "nasalseg_root": str(args.nasalseg_root),
        "ground_truth_labels": list(airway_labels),
        "min_component_voxels": args.min_component_voxels,
        "thresholds": thresholds,
        "cases": cases,
        "per_case": per_case,
        "summary": summary,
        "best_threshold_hu_max": best_t,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id"] + [f"dice_hu_{t:g}" for t in thresholds])
        for cid in cases:
            writer.writerow([cid] + [f"{per_case[cid][f'{t:g}']:.4f}" for t in thresholds])

    print(f"\n[dice] wrote {args.out} and {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
