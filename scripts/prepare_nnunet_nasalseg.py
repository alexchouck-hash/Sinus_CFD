#!/usr/bin/env python3
"""
Convert NasalSeg (NRRD) into an nnU-Net v2 raw dataset.

Label IDs (NasalSeg / Sinus_CFD):
  0 background
  1 left_nasal_cavity
  2 right_nasal_cavity
  3 nasopharynx
  4 left_maxillary_sinus
  5 right_maxillary_sinus

Output (default):
  data/nnUNet_raw/Dataset501_NasalSeg/
    dataset.json
    imagesTr/P001_0000.nii.gz …
    labelsTr/P001.nii.gz …

Usage:
  py -3.12 scripts/download_nasalseg.py
  py -3.12 scripts/prepare_nnunet_nasalseg.py
  py -3.12 scripts/prepare_nnunet_nasalseg.py --verify-only

Then (with nnU-Net installed + GPU recommended):
  See docs/nnunet_nasal.md
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]

# nnU-Net v2 semantic labels (must match NasalSeg)
LABELS = {
    "background": 0,
    "left_nasal_cavity": 1,
    "right_nasal_cavity": 2,
    "nasopharynx": 3,
    "left_maxillary_sinus": 4,
    "right_maxillary_sinus": 5,
}

# Convenience merged classes for CFD-oriented experiments (optional remap)
MERGE_AIRWAY = {
    # background stays 0; L+R cavity + NP → 1; sinuses → 2
    "background": 0,
    "nasal_airway": 1,  # 1,2,3
    "maxillary_sinus": 2,  # 4,5
}


def _case_id_from_name(name: str) -> str | None:
    m = re.match(r"(P\d{3})", name)
    return m.group(1) if m else None


def _nrrd_to_nifti(src: Path, dst: Path) -> None:
    img = sitk.ReadImage(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(dst), useCompression=True)


def _geometry_disagrees(a: sitk.Image, b: sitk.Image, atol: float = 1e-3) -> bool:
    """True if spacing/origin/direction differ by more than float rounding noise."""
    return (
        not np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        or not np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        or not np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6)
    )


def _write_label_nifti(lab_src: Path, dst: Path, ref_img: sitk.Image, cid: str) -> None:
    """
    Write the label volume with the paired image's spacing/origin/direction.

    A handful of NasalSeg label NRRDs carry a spacing/origin/direction that
    genuinely disagrees with their paired image (14/130 cases as of this
    dataset release) even though the voxel array size always matches.
    nnU-Net's integrity check treats that as a fatal error. Since sizes
    agree, the intended correspondence is voxel-index-to-voxel-index, so we
    always stamp the image's geometry onto the label rather than trust the
    label's own (sometimes wrong) header — this also sidesteps NIfTI's
    float32 header vs NRRD's float64 introducing spurious rounding
    "mismatches" on every case if compared after a round-trip.
    """
    lab = sitk.ReadImage(str(lab_src))
    if lab.GetSize() != ref_img.GetSize():
        raise ValueError(
            f"{cid}: label/image voxel array size mismatch "
            f"({lab.GetSize()} vs {ref_img.GetSize()}) — cannot assume "
            "index correspondence, skipping geometry fix-up."
        )
    if _geometry_disagrees(lab, ref_img):
        print(f"  {cid}: label geometry disagreed with image header — copied image geometry onto label")
    lab.CopyInformation(ref_img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(lab, str(dst), useCompression=True)


def _remap_airway(lab: sitk.Image) -> sitk.Image:
    a = sitk.GetArrayFromImage(lab).astype(np.uint8)
    out = np.zeros_like(a)
    out[np.isin(a, [1, 2, 3])] = 1
    out[np.isin(a, [4, 5])] = 2
    o = sitk.GetImageFromArray(out)
    o.CopyInformation(lab)
    return o


def prepare(
    nasalseg_root: Path,
    nnunet_raw: Path,
    dataset_id: int = 501,
    dataset_name: str = "NasalSeg",
    remap: str = "full",
    max_cases: int | None = None,
) -> Path:
    img_dir = nasalseg_root / "images"
    lab_dir = nasalseg_root / "labels"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {img_dir} and {lab_dir}. Run: py -3.12 scripts/download_nasalseg.py"
        )

    folder = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = folder / "imagesTr"
    labels_tr = folder / "labelsTr"
    if folder.exists():
        shutil.rmtree(folder)
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    cases: list[str] = []
    for img_path in sorted(img_dir.glob("P*_img.nrrd")):
        cid = _case_id_from_name(img_path.name)
        if not cid:
            continue
        lab_path = lab_dir / f"{cid}_seg.nrrd"
        if not lab_path.is_file():
            print(f"  skip {cid}: missing {lab_path.name}")
            continue
        cases.append(cid)
        if max_cases is not None and len(cases) >= max_cases:
            break

    if not cases:
        raise RuntimeError("No NasalSeg cases found")

    print(f"[nnUNet] Converting {len(cases)} cases → {folder}")
    print(f"  remap={remap}")

    for i, cid in enumerate(cases):
        img_src = img_dir / f"{cid}_img.nrrd"
        lab_src = lab_dir / f"{cid}_seg.nrrd"
        # nnU-Net: CASE_0000.nii.gz for modality 0
        img_dst = images_tr / f"{cid}_0000.nii.gz"
        lab_dst = labels_tr / f"{cid}.nii.gz"
        _nrrd_to_nifti(img_src, img_dst)
        ref_img = sitk.ReadImage(str(img_dst))
        if remap == "airway":
            lab_img = sitk.ReadImage(str(lab_src))
            if lab_img.GetSize() != ref_img.GetSize():
                raise ValueError(f"{cid}: label/image size mismatch {lab_img.GetSize()} vs {ref_img.GetSize()}")
            if _geometry_disagrees(lab_img, ref_img):
                print(f"  {cid}: label geometry disagreed with image header — copied image geometry onto label")
            lab_img.CopyInformation(ref_img)
            lab_img = _remap_airway(lab_img)
            sitk.WriteImage(lab_img, str(lab_dst), useCompression=True)
        else:
            _write_label_nifti(lab_src, lab_dst, ref_img, cid)
        if (i + 1) % 20 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)}")

    if remap == "airway":
        labels_json = MERGE_AIRWAY
    else:
        labels_json = LABELS

    dataset = {
        "channel_names": {"0": "CT"},
        "labels": labels_json,
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "name": dataset_name,
        "description": (
            "NasalSeg (Zhang et al., Sci Data 2024) converted for nnU-Net v2. "
            "Cite the NasalSeg paper if you use this data."
        ),
        "reference": "https://zenodo.org/records/13893419",
        "licence": "See Zenodo NasalSeg record / Scientific Data paper",
        "release": "v1_sinus_cfd",
        "sinus_cfd": {
            "source_layout": "data/NasalSeg/{images,labels}",
            "remap": remap,
            "case_ids": cases,
            "label_meaning_full": LABELS,
        },
    }
    (folder / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    # 5-fold case lists (simple sequential splits for planning)
    folds = {str(k): {"train": [], "val": []} for k in range(5)}
    for i, cid in enumerate(cases):
        f = i % 5
        for k in range(5):
            if k == f:
                folds[str(k)]["val"].append(cid)
            else:
                folds[str(k)]["train"].append(cid)
    (folder / "splits_sinus_cfd.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    # Environment helper for Windows PowerShell
    env_ps1 = folder / "set_nnunet_env.ps1"
    raw = str(nnunet_raw.resolve())
    pre = str((nnunet_raw.parent / "nnUNet_preprocessed").resolve())
    res = str((nnunet_raw.parent / "nnUNet_results").resolve())
    env_ps1.write_text(
        f"""# nnU-Net paths for Sinus_CFD (run from any shell)
$env:nnUNet_raw = "{raw}"
$env:nnUNet_preprocessed = "{pre}"
$env:nnUNet_results = "{res}"
Write-Host "nnUNet_raw=$env:nnUNet_raw"
Write-Host "nnUNet_preprocessed=$env:nnUNet_preprocessed"
Write-Host "nnUNet_results=$env:nnUNet_results"
""",
        encoding="utf-8",
    )
    env_sh = folder / "set_nnunet_env.sh"
    env_sh.write_text(
        f"""#!/usr/bin/env bash
export nnUNet_raw="{raw}"
export nnUNet_preprocessed="{pre}"
export nnUNet_results="{res}"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
""",
        encoding="utf-8",
    )

    print(f"[nnUNet] Wrote dataset.json ({len(cases)} training cases)")
    print(f"[nnUNet] Env helpers: {env_ps1.name}, {env_sh.name}")
    return folder


def verify(folder: Path, n_check: int = 3) -> None:
    ds = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    cases = ds.get("sinus_cfd", {}).get("case_ids") or []
    print(f"[verify] dataset {folder.name} numTraining={ds['numTraining']}")
    print(f"[verify] labels={ds['labels']}")
    for cid in cases[:n_check]:
        img = folder / "imagesTr" / f"{cid}_0000.nii.gz"
        lab = folder / "labelsTr" / f"{cid}.nii.gz"
        assert img.is_file(), img
        assert lab.is_file(), lab
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(lab)))
        print(f"  {cid}: label unique={np.unique(a).tolist()} shape={a.shape}")
    print("[verify] OK")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nasalseg-root", type=Path, default=REPO_ROOT / "data" / "NasalSeg")
    p.add_argument(
        "--nnunet-raw",
        type=Path,
        default=REPO_ROOT / "data" / "nnUNet_raw",
        help="nnUNet_raw root (DatasetXXX_* created inside)",
    )
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--dataset-name", default="NasalSeg")
    p.add_argument(
        "--remap",
        choices=("full", "airway"),
        default="full",
        help="full=5 structures; airway=merged nasal air vs sinus",
    )
    p.add_argument("--max-cases", type=int, default=None, help="Debug: convert only N cases")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    folder = (
        args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    )
    if args.verify_only:
        if not folder.is_dir():
            print(f"Missing {folder}", file=sys.stderr)
            return 1
        verify(folder)
        return 0

    try:
        folder = prepare(
            nasalseg_root=args.nasalseg_root,
            nnunet_raw=args.nnunet_raw,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            remap=args.remap,
            max_cases=args.max_cases,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    verify(folder)
    print()
    print("Next steps:")
    print("  1. Install (GPU recommended):  py -3.12 -m pip install -r requirements-nn.txt")
    print("  2. Set env:  .\\data\\nnUNet_raw\\Dataset501_NasalSeg\\set_nnunet_env.ps1")
    print("  3. Plan:     nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity")
    print("  4. Train:    nnUNetv2_train 501 3d_fullres 0")
    print("  See docs/nnunet_nasal.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
