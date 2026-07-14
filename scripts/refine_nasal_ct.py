#!/usr/bin/env python3
"""
Refine nasal airway from CT topology (true nostrils, L/R cavities, septum).

Uses the cropped head volume already produced by process_whole_head:
  tissues / head_mask / airway + original HU (cropped via stats crop_origin).

Writes:
  - passage lumen with CT nares (minimal tunnel only if needed)
  - left/right cavity masks + STLs
  - septum + mucosa wall masks + STLs
  - updated nares.json / BC inlet centers from CT openings
  - centerline face→trachea via analyze_nasal_passage

Example:
  py -3.12 scripts/refine_nasal_ct.py --case VisibleHuman_Head
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
from sinus_cfd.nasal_passage import (  # noqa: E402
    analyze_nasal_passage,
    write_passage_outputs,
)
from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402


def _write_mask(
    mask: np.ndarray,
    path: Path,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    ref: sitk.Image | None,
) -> None:
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if ref is not None:
        img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, str(path))


def _export_stl(mask: np.ndarray, path: Path, spacing, origin, target_faces: int = 20000) -> bool:
    if not mask.any() or int(mask.sum()) < 50:
        return False
    try:
        mesh = _mask_to_mesh(mask, spacing, origin)
        if len(mesh.faces) > target_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
            except Exception:
                pass
        mesh.export(path)
        return True
    except Exception as exc:
        print(f"  STL failed {path.name}: {exc}")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--air-hu-max", type=float, default=-300.0)
    p.add_argument("--tunnel-radius-mm", type=float, default=2.0,
                   help="Only used if CT openings still leave a residual gap")
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    stats_path = case_dir / f"{args.case}_stats.json"
    tissues_path = case_dir / f"{args.case}_tissues.nrrd"
    head_path = case_dir / f"{args.case}_head_mask.nrrd"
    bc_path = case_dir / f"{args.case}_boundary_conditions.json"

    if not stats_path.is_file():
        raise SystemExit(f"Missing {stats_path} — run process_whole_head first")
    if not tissues_path.is_file() or not head_path.is_file():
        raise SystemExit("Missing tissues/head masks — run process_whole_head first")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    image_path = Path(stats.get("image_path") or "")
    if not image_path.is_file():
        image_path = REPO_ROOT / "data" / "VisibleHuman_Head" / "VHFCT1mm_Head.nrrd"
    if not image_path.is_file():
        raise SystemExit(f"Missing CT volume: {image_path}")

    full = sitk.ReadImage(str(image_path))
    hu_full = sitk.GetArrayFromImage(full).astype(np.float32)
    crop = stats.get("crop_origin_zyx") or [0, 0, 0]
    cz, cy, cx = (int(crop[0]), int(crop[1]), int(crop[2]))

    head_img = sitk.ReadImage(str(head_path))
    body = sitk.GetArrayFromImage(head_img).astype(bool)
    tissues = sitk.GetArrayFromImage(sitk.ReadImage(str(tissues_path)))
    soft = tissues == 2
    air_from_tissues = tissues == 1

    nz, ny, nx = body.shape
    # Crop HU to match processed head grid
    hu = hu_full[cz : cz + nz, cy : cy + ny, cx : cx + nx]
    if hu.shape != body.shape:
        # pad or crop to match
        out = np.full(body.shape, -1024.0, dtype=np.float32)
        z1 = min(nz, hu_full.shape[0] - cz)
        y1 = min(ny, hu_full.shape[1] - cy)
        x1 = min(nx, hu_full.shape[2] - cx)
        src = hu_full[cz : cz + z1, cy : cy + y1, cx : cx + x1]
        out[:z1, :y1, :x1] = src
        hu = out
        print(f"[{args.case}] HU cropped/padded to {hu.shape}")

    spacing = tuple(float(v) for v in head_img.GetSpacing())
    origin = tuple(float(v) for v in head_img.GetOrigin())

    # Prior landmarks (optional)
    prior_l = prior_r = None
    nares_path = case_dir / f"{args.case}_nares.json"
    if nares_path.is_file():
        nj = json.loads(nares_path.read_text(encoding="utf-8"))
        pts = nj.get("naris_points") or []
        for pt in pts:
            if pt.get("name") == "left_nostril":
                prior_l = pt.get("center_mm")
            elif pt.get("name") == "right_nostril":
                prior_r = pt.get("center_mm")

    y_ant = True
    for note in stats.get("notes") or []:
        if "y_anterior_is_low=False" in str(note):
            y_ant = False

    print(f"[{args.case}] CT-native nasal extraction…")
    result = extract_ct_nasal_airway(
        hu=hu,
        body=body,
        interior_air=air_from_tissues,
        soft_tissue=soft,
        spacing_xyz=spacing,
        origin_xyz=origin,
        y_anterior_is_low=y_ant,
        air_hu_max=args.air_hu_max,
        prior_left_mm=prior_l,
        prior_right_mm=prior_r,
    )
    for n in result.notes:
        print(f"  note: {n}")

    meta = result.to_meta()
    meta_path = case_dir / f"{args.case}_ct_nasal_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {meta_path.name}")

    # Masks
    _write_mask(result.left_cavity, case_dir / f"{args.case}_cavity_left.nrrd", spacing, origin, head_img)
    _write_mask(result.right_cavity, case_dir / f"{args.case}_cavity_right.nrrd", spacing, origin, head_img)
    _write_mask(result.septum, case_dir / f"{args.case}_septum.nrrd", spacing, origin, head_img)
    _write_mask(result.mucosa_wall, case_dir / f"{args.case}_mucosa_wall.nrrd", spacing, origin, head_img)
    _write_mask(result.naris_opening, case_dir / f"{args.case}_naris_opening_ct.nrrd", spacing, origin, head_img)
    _write_mask(result.passage_lumen, case_dir / f"{args.case}_passage_lumen.nrrd", spacing, origin, head_img)
    _write_mask(result.passage_lumen, case_dir / f"{args.case}_airway_mask.nrrd", spacing, origin, head_img)

    # STLs
    _export_stl(result.left_cavity, case_dir / f"{args.case}_cavity_left.stl", spacing, origin, 15000)
    _export_stl(result.right_cavity, case_dir / f"{args.case}_cavity_right.stl", spacing, origin, 15000)
    _export_stl(result.septum, case_dir / f"{args.case}_septum.stl", spacing, origin, 12000)
    _export_stl(result.mucosa_wall, case_dir / f"{args.case}_mucosa_wall.stl", spacing, origin, 25000)
    _export_stl(result.passage_lumen, case_dir / f"{args.case}_airway.stl", spacing, origin, 30000)
    print(f"  wrote L/R cavities, septum, mucosa, airway STLs")

    # Update nares.json from CT centers
    skin_centers = []
    if result.left_naris_center_mm:
        skin_centers.append(
            {
                "name": "left_nostril",
                "center_mm": result.left_naris_center_mm,
                "skin_voxel_zyx": list(result.left_naris_center_zyx)
                if result.left_naris_center_zyx
                else None,
                "method": "ct_naris_opening",
            }
        )
    if result.right_naris_center_mm:
        skin_centers.append(
            {
                "name": "right_nostril",
                "center_mm": result.right_naris_center_mm,
                "skin_voxel_zyx": list(result.right_naris_center_zyx)
                if result.right_naris_center_zyx
                else None,
                "method": "ct_naris_opening",
            }
        )
    nares_out = {
        "method": "ct_topology_hu_edge",
        "y_anterior_is_low": y_ant,
        "naris_points": skin_centers,
        "notes": result.notes,
    }
    nares_path.write_text(json.dumps(nares_out, indent=2), encoding="utf-8")
    print(f"  wrote {nares_path.name}")

    # Update BC inlet centers
    if bc_path.is_file() and skin_centers:
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        by_name = {p["name"]: p["center_mm"] for p in skin_centers}
        for port in bc.get("ports", []):
            if port.get("name") in by_name:
                port["center_mm"] = by_name[port["name"]]
                port["method"] = "ct_naris_opening"
                port["notes"] = "External naris from CT topology (skin opening)."
        bc_path.write_text(json.dumps(bc, indent=2), encoding="utf-8")
        print(f"  updated BC inlet centers")

    # Passage analysis / centerline (uses CT passage lumen; light tunnel only if gap remains)
    outlets = []
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        outlets = [p["center_mm"] for p in bc.get("ports", []) if p.get("role") == "outlet"]
    if not outlets:
        # caudal air centroid fallback
        zz, yy, xx = np.where(result.passage_lumen)
        if len(zz):
            # most caudal (low z if superior high — stats)
            k = int(np.argmin(zz))
            outlets = [
                [
                    origin[0] + xx[k] * spacing[0],
                    origin[1] + yy[k] * spacing[1],
                    origin[2] + zz[k] * spacing[2],
                ]
            ]

    inlets = [p["center_mm"] for p in skin_centers] or [prior_l, prior_r]
    inlets = [c for c in inlets if c]

    print(f"[{args.case}] passage + centerline…")
    masks, passage, metrics = analyze_nasal_passage(
        lumen=result.passage_lumen,
        spacing=spacing,
        origin=origin,
        inlet_centers_mm=inlets,
        outlet_center_mm=outlets[0],
        case_id=args.case,
        open_radius_mm=6.0,
        skin_naris_centers_mm=inlets,
        tunnel_radius_mm=args.tunnel_radius_mm,
    )
    # Prefer CT septum over recompute
    masks["septum"] = result.septum
    masks["mucosa_wall"] = result.mucosa_wall
    masks["left_cavity"] = result.left_cavity
    masks["right_cavity"] = result.right_cavity
    masks["naris_opening"] = result.naris_opening

    passage["ct_nasal"] = meta
    passage["includes_external_nares"] = True
    passage["method"] = "ct_topology_hu_edge"
    paths = write_passage_outputs(
        args.case, case_dir, masks, passage, spacing, origin, reference_image=head_img
    )
    # Also write septum/mucosa into passage outputs if not already
    _write_mask(result.septum, case_dir / f"{args.case}_septum.nrrd", spacing, origin, head_img)
    _export_stl(result.septum, case_dir / f"{args.case}_septum.stl", spacing, origin, 12000)

    print(
        f"[{args.case}] lumen={metrics.lumen_volume_ml:.1f} mL  "
        f"centerline={metrics.centerline_length_mm:.1f} mm  "
        f"septum={int(result.septum.sum())} vx  "
        f"L/R={int(result.left_cavity.sum())}/{int(result.right_cavity.sum())}"
    )
    for k, path in paths.items():
        print(f"  wrote {k}: {path.name}")
    print("Done. Reload Streamlit viewer (Clear cache).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
