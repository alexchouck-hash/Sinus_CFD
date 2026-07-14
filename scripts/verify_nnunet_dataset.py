#!/usr/bin/env python3
"""Quick sanity check for Dataset501_NasalSeg (or custom id)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--name", default="NasalSeg")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    folder = REPO / "data" / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.name}"
    if not folder.is_dir():
        print(f"Missing {folder}\nRun: py -3.12 scripts/prepare_nnunet_nasalseg.py", file=sys.stderr)
        return 1
    ds = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    cases = ds["sinus_cfd"]["case_ids"]
    print(f"{folder.name}: {ds['numTraining']} cases, labels={list(ds['labels'].keys())}")
    for cid in cases[: args.n]:
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / "labelsTr" / f"{cid}.nii.gz")))
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / "imagesTr" / f"{cid}_0000.nii.gz")))
        print(f"  {cid}: img {img.shape} HU[{img.min():.0f},{img.max():.0f}] lab {np.unique(lab).tolist()}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
