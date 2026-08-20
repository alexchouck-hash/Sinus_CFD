#!/usr/bin/env python3
"""
Automatic sinonasal segmentation.

Whole-head cases (Visible Human, THCA, skin-bounded CQ500): competing
geodesic flood — no operator labels, no slice confirm.

NasalSeg-style crops: Dataset501 nnU-Net if available, else leftover-air
expand from existing 5-class labels (IDs 1–5 never overwritten).

Usage (repo root):
  py -3.12 scripts\\autosegment_ct.py --case VisibleHuman_Head --no-legacy
  py -3.12 scripts\\autosegment_ct.py --image data\\images\\P001_img.nrrd --labels data\\labels\\P001_seg.nrrd
  py -3.12 scripts\\autosegment_ct.py --nasalseg-all --max-cases 5
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

from sinus_cfd.nasal_airway_ct import extract_ct_nasal_airway  # noqa: E402
from sinus_cfd.segment_sinonasal import expand_named_airspaces  # noqa: E402
from sinus_cfd.segmentation_labels import LABEL_NAMES  # noqa: E402


def _write_mask(mask: np.ndarray, path: Path, ref: sitk.Image) -> None:
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.CopyInformation(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def _write_like(ref: sitk.Image, array: np.ndarray, path: Path) -> None:
    _write_mask(array, path, ref)


def autosegment_nasalseg_crop(
    image_path: Path,
    out_dir: Path,
    labels_path: Path | None,
    use_nnunet: bool,
    case_id: str,
) -> dict:
    image = sitk.ReadImage(str(image_path))
    hu = sitk.GetArrayFromImage(image).astype(np.float32)
    named = None
    source = "expand"
    if labels_path is not None and labels_path.is_file() and not use_nnunet:
        named = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_path)))
        source = "labels+expand"
    elif use_nnunet:
        from sinus_cfd.nnunet_infer import predict_labels

        named = predict_labels(image)
        source = "nnunet+expand"
    result = expand_named_airspaces(hu, named, case_id=case_id)
    _write_like(image, result.labels, out_dir / f"{case_id}_sinonasal_labels.nrrd")
    _write_like(image, result.passage_mask().astype(np.uint8), out_dir / f"{case_id}_passage_mask.nrrd")
    meta = result.to_meta()
    meta["source"] = source
    (out_dir / f"{case_id}_sinonasal_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def autosegment_whole_head_case(
    case: str,
    outputs_root: Path,
    legacy: bool,
) -> dict:
    case_dir = outputs_root / case
    stats_path = case_dir / f"{case}_stats.json"
    tissues_path = case_dir / f"{case}_tissues.nrrd"
    head_path = case_dir / f"{case}_head_mask.nrrd"
    airway_path = case_dir / f"{case}_airway_mask.nrrd"
    if not stats_path.is_file() or not tissues_path.is_file() or not head_path.is_file():
        raise SystemExit(
            f"Missing process_whole_head outputs in {case_dir}. "
            f"Run: py -3.12 scripts\\process_whole_head.py --case {case}"
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    image_path = Path(stats.get("image_path") or "")
    if not image_path.is_file():
        image_path = REPO_ROOT / "data" / case / "VHFCT1mm_Head.nrrd"
    head_img = sitk.ReadImage(str(head_path))
    body = sitk.GetArrayFromImage(head_img).astype(bool)
    tissues = sitk.GetArrayFromImage(sitk.ReadImage(str(tissues_path)))
    flood = None
    if airway_path.is_file():
        flood = sitk.GetArrayFromImage(sitk.ReadImage(str(airway_path))).astype(bool)

    full = sitk.ReadImage(str(image_path)) if image_path.is_file() else head_img
    hu_full = sitk.GetArrayFromImage(full).astype(np.float32)
    crop = stats.get("crop_origin_zyx") or [0, 0, 0]
    cz, cy, cx = (int(crop[0]), int(crop[1]), int(crop[2]))
    nz, ny, nx = body.shape
    hu = np.full(body.shape, -1024.0, dtype=np.float32)
    z1 = min(nz, hu_full.shape[0] - cz)
    y1 = min(ny, hu_full.shape[1] - cy)
    x1 = min(nx, hu_full.shape[2] - cx)
    if image_path.is_file():
        hu[:z1, :y1, :x1] = hu_full[cz : cz + z1, cy : cy + y1, cx : cx + x1]

    spacing = tuple(float(v) for v in head_img.GetSpacing())
    origin = tuple(float(v) for v in head_img.GetOrigin())
    y_ant = True
    sup_high = True
    for note in stats.get("notes") or []:
        if "y_anterior_is_low=False" in str(note):
            y_ant = False
        if "superior_is_high_z=False" in str(note):
            sup_high = False

    prior_l = prior_r = None
    nares_path = case_dir / f"{case}_nares.json"
    if nares_path.is_file():
        nj = json.loads(nares_path.read_text(encoding="utf-8"))
        for pt in nj.get("naris_points") or []:
            if pt.get("name") == "left_nostril":
                prior_l = pt.get("center_mm")
            elif pt.get("name") == "right_nostril":
                prior_r = pt.get("center_mm")

    result = extract_ct_nasal_airway(
        hu=hu,
        body=body,
        interior_air=flood if flood is not None else (tissues == 1),
        soft_tissue=tissues == 2,
        spacing_xyz=spacing,
        origin_xyz=origin,
        y_anterior_is_low=y_ant,
        prior_left_mm=prior_l,
        prior_right_mm=prior_r,
        legacy_midplane_split=legacy,
        superior_is_high_z=sup_high,
        flood_domain=flood,
    )
    _write_mask(result.left_cavity, case_dir / f"{case}_cavity_left.nrrd", head_img)
    _write_mask(result.right_cavity, case_dir / f"{case}_cavity_right.nrrd", head_img)
    _write_mask(result.septum, case_dir / f"{case}_septum.nrrd", head_img)
    _write_mask(result.passage_lumen, case_dir / f"{case}_passage_lumen.nrrd", head_img)
    if not legacy:
        _write_mask(result.passage_lumen, case_dir / f"{case}_passage_lumen.nrrd", head_img)
        # Keep airway_mask as pre-strip whole-head path (A12 / check 2).
    if result.merge_zone is not None:
        _write_mask(result.merge_zone, case_dir / f"{case}_merge_zone.nrrd", head_img)
    if result.choanal_landmark is not None:
        _write_mask(result.choanal_landmark, case_dir / f"{case}_choanal_landmark.nrrd", head_img)
    if result.sinus_detour is not None:
        _write_mask(result.sinus_detour, case_dir / f"{case}_sinus_detour.nrrd", head_img)
    meta = result.to_meta()
    (case_dir / f"{case}_ct_nasal_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=str, default=None, help="whole-head outputs/<case>")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--nnunet", action="store_true")
    p.add_argument("--nasalseg-all", action="store_true")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument(
        "--no-legacy",
        action="store_true",
        help="competing geodesic flood (whole-head). Default is midplane split.",
    )
    args = p.parse_args()

    if args.nasalseg_all:
        images = sorted((REPO_ROOT / "data" / "images").glob("P*_img.nrrd"))
        if args.max_cases:
            images = images[: args.max_cases]
        n_ok = 0
        for img in images:
            case_id = img.stem.replace("_img", "")
            labels = REPO_ROOT / "data" / "labels" / f"{case_id}_seg.nrrd"
            out = args.out_dir or (args.outputs_root / case_id)
            try:
                meta = autosegment_nasalseg_crop(
                    img, out, labels if labels.is_file() else None, args.nnunet, case_id
                )
                print(
                    f"{case_id}: passage={sum(meta['voxel_counts'].get(LABEL_NAMES[i], 0) for i in (1, 2, 3))} "
                    f"unassigned_air={meta['unassigned_air_voxels']}"
                )
                n_ok += 1
            except Exception as exc:
                print(f"{case_id} FAILED: {exc}")
        print(f"done {n_ok}/{len(images)} NasalSeg crops")
        return 0 if n_ok else 1

    if args.image is not None:
        case_id = args.image.stem.replace("_img", "").replace("_0000", "")
        out = args.out_dir or (args.outputs_root / case_id)
        meta = autosegment_nasalseg_crop(
            args.image, out, args.labels, args.nnunet, case_id
        )
        print(json.dumps({k: meta[k] for k in ("voxel_counts", "ostial_contact_voxels", "unassigned_air_voxels", "source") if k in meta}, indent=2))
        return 0

    if args.case:
        meta = autosegment_whole_head_case(
            args.case, args.outputs_root, legacy=not args.no_legacy
        )
        print(f"method={meta.get('method')} L={meta.get('left_voxels')} R={meta.get('right_voxels')} "
              f"septum={meta.get('septum_voxels')} passage={meta.get('passage_voxels')}")
        for n in meta.get("notes") or []:
            print("  note:", n)
        return 0

    p.error("pass --case, --image, or --nasalseg-all")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
