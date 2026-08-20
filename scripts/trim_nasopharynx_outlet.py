#!/usr/bin/env python3
"""
Trim a whole-head airway to a clean nostrils -> nasopharynx domain.

Whole-head extraction can run a thin, off-midline conduit down the neck and place
the outlet on that lateral tube (patient-right "pharynx"), which is anatomically
wrong for nasal airflow. Real nasal CFD terminates at a coronal plane through the
posterior choanae / nasopharynx.

This finds that plane from the coronal-slice area profile (the airway is broad and
midline through the nasal cavity, then collapses to a narrow off-midline tube at
the choanae) and:
  * keeps only the nostril-connected airway anterior to that plane,
  * places the outlet at the midline choanal face,
  * rewrites <case>_airway_mask.nrrd and the outlet in <case>_boundary_conditions.json
    (originals backed up to *.pretrim).

Then re-solves the flow so streamlines run nostrils -> nasopharynx.

    py -3.12 scripts/trim_nasopharynx_outlet.py --case VisibleHuman_Male_Head
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

for _s in (sys.stdout, sys.stderr):  # this script prints Unicode (→, ≈, ²)
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _idx(center_mm, img) -> list[int]:
    """Physical mm (x,y,z) -> integer index; spacing + Direction via ITK (Fix 1)."""
    size = img.GetSize()  # x, y, z
    ijk = img.TransformPhysicalPointToIndex(
        (float(center_mm[0]), float(center_mm[1]), float(center_mm[2])))
    return [int(min(max(int(ijk[i]), 0), size[i] - 1)) for i in range(3)]


def _load_ap_orientation(cdir: Path, case: str, bc: dict) -> tuple[bool, bool]:
    """y_anterior_is_low / superior_is_high_z — the flags whole_head derives (Fix 3).

    Sources (first hit wins): BC keys, <case>_nares.json, <case>_stats.json notes.
    Defaults match Visible Human (anterior=low-y, superior=high-z). The A-P axis
    stays array-y (axis 1) as elsewhere in the repo; only the posterior *sign* is
    made orientation-aware here.
    """
    y_ant_low, sup_high_z = True, True
    if "y_anterior_is_low" in bc:
        y_ant_low = bool(bc["y_anterior_is_low"])
    if "superior_is_high_z" in bc:
        sup_high_z = bool(bc["superior_is_high_z"])
    npath = cdir / f"{case}_nares.json"
    if npath.is_file():
        nj = json.loads(npath.read_text(encoding="utf-8"))
        if "y_anterior_is_low" in nj:
            y_ant_low = bool(nj["y_anterior_is_low"])
    spath = cdir / f"{case}_stats.json"
    if spath.is_file():
        sj = json.loads(spath.read_text(encoding="utf-8"))
        if "superior_is_high_z" in sj:
            sup_high_z = bool(sj["superior_is_high_z"])
        for note in sj.get("notes") or []:
            if "y_anterior_is_low=False" in str(note):
                y_ant_low = False
            elif "y_anterior_is_low=True" in str(note):
                y_ant_low = True
    return y_ant_low, sup_high_z


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="VisibleHuman_Male_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO / "outputs")
    p.add_argument("--midband-mm", type=float, default=30.0,
                   help="Half-width of the midline x-band used to detect the choanal collapse")
    p.add_argument("--min-midband", type=int, default=40,
                   help="Midline air voxels below this (posterior to the cavity) marks the choana")
    p.add_argument("--no-flow", action="store_true", help="Only trim; skip the flow re-solve")
    p.add_argument("--flow-iterations", type=int, default=450)
    p.add_argument("--streamline-seeds", type=int, default=120)
    args = p.parse_args()

    cdir = args.outputs_root / args.case
    mask_path = cdir / f"{args.case}_airway_mask.nrrd"
    bc_path = cdir / f"{args.case}_boundary_conditions.json"
    img = sitk.ReadImage(str(mask_path))
    a = sitk.GetArrayFromImage(img).astype(bool)  # z,y,x
    origin = tuple(float(v) for v in img.GetOrigin())
    spacing = tuple(float(v) for v in img.GetSpacing())  # x, y, z
    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    y_ant_low, sup_high_z = _load_ap_orientation(cdir, args.case, bc)
    post_sign = 1.0 if y_ant_low else -1.0  # +y posterior when anterior is low-y
    print(f"[{args.case}] orientation: y_anterior_is_low={y_ant_low} "
          f"superior_is_high_z={sup_high_z}  (A-P=array-y; posterior={'+' if y_ant_low else '-'}y)")

    inlets = [pt for pt in bc["ports"] if pt.get("role") == "inlet"]
    nose_idx = [_idx(pt["center_mm"], img) for pt in inlets]
    xmid = float(np.mean([n[0] for n in nose_idx]))
    nose_y = float(np.mean([n[1] for n in nose_idx]))
    mb = int(round(args.midband_mm / spacing[0]))
    xs_grid = np.arange(a.shape[2])
    ys = np.arange(a.shape[1])

    # Coronal-slice profile: total air and midline-band air per y.
    total = np.array([a[:, y, :].sum() for y in range(a.shape[1])])
    midband = np.array([(a[:, y, :][:, np.abs(xs_grid - xmid) < mb]).sum() for y in range(a.shape[1])])

    # Posterior half-space (orientation-aware: posterior is +y iff anterior is low-y).
    posterior = (ys > nose_y + 5) if y_ant_low else (ys < nose_y - 5)
    if not (total * posterior).any():
        raise SystemExit(f"No posterior airway found — check orientation (y_anterior_is_low={y_ant_low}).")
    y_peak = int(np.argmax(total * posterior))  # widest nasal-cavity slice

    # Walk posterior from the cavity peak until midline air collapses => choanal plane.
    if y_ant_low:
        y_cut, y_walk = a.shape[1] - 1, range(y_peak, a.shape[1])
    else:
        y_cut, y_walk = 0, range(y_peak, -1, -1)
    for y in y_walk:
        if midband[y] < args.min_midband:
            y_cut = y
            break
    cut_y_mm = float(origin[1] + y_cut * spacing[1])
    print(f"[{args.case}] nostril midline x={xmid:.0f}  cavity peak y={y_peak} "
          f"(area {total[y_peak]})  choanal cut y={y_cut} (mm {cut_y_mm:.0f})")

    # Keep nostril-connected airway on the anterior side of the cut (incl. plane).
    keep_band = np.zeros_like(a)
    if y_ant_low:
        keep_band[:, : y_cut + 1, :] = True   # anterior = low y
    else:
        keep_band[:, y_cut:, :] = True         # anterior = high y
    kept = a & keep_band
    lab, n = ndi.label(kept)
    seed_labels = {lab[nz, ny, nx] for nx, ny, nz in nose_idx if lab[nz, ny, nx] > 0}
    if not seed_labels:  # nearest kept voxel to any nostril
        zz, yy, xx = np.where(kept)
        nx, ny, nz = nose_idx[0]
        j = np.argmin((zz - nz) ** 2 + (yy - ny) ** 2 + (xx - nx) ** 2)
        seed_labels = {lab[zz[j], yy[j], xx[j]]}
    final = np.isin(lab, list(seed_labels))
    removed = a & ~final
    print(f"[{args.case}] airway {a.sum():,} -> {final.sum():,} voxels "
          f"(removed {removed.sum():,}; posterior tube + off-plane bits)")

    # Outlet = choanal face on the posterior side of the kept domain (few slices at cut).
    face = final & ((ys[None, :, None] >= y_cut - 4) if y_ant_low else (ys[None, :, None] <= y_cut + 4))
    zz, yy, xx = np.where(face)
    if len(zz) == 0:
        raise SystemExit("Choanal face empty after trim — cut plane may be wrong.")
    outlet_mm = list(img.TransformContinuousIndexToPhysicalPoint(
        (float(xx.mean()), float(yy.mean()), float(zz.mean()))))
    outlet_area = float(len(np.unique(list(zip(xx.tolist(), zz.tolist())), axis=0)) * spacing[0] * spacing[2])
    print(f"[{args.case}] new outlet (choana) mm=({outlet_mm[0]:.0f},{outlet_mm[1]:.0f},{outlet_mm[2]:.0f}) "
          f"area≈{outlet_area:.0f} mm²  (was patient-right lateral tube)")

    # --- write corrected mask + BC (back up originals once) ---
    for pth in (mask_path, bc_path):
        bak = pth.with_suffix(pth.suffix + ".pretrim")
        if not bak.exists():
            shutil.copy2(pth, bak)
    out_img = sitk.GetImageFromArray(final.astype("uint8"))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(mask_path))

    for pt in bc["ports"]:
        if pt.get("role") == "outlet":
            pt["center_mm"] = outlet_mm
            pt["area_mm2"] = outlet_area
            pt["normal_xyz"] = [0.0, post_sign, 0.0]  # posterior along +/- y (orientation-aware)
            pt["method"] = "nasopharynx_choanal_plane"
            pt["notes"] = ("Outlet at the midline choanal coronal plane (nostrils→nasopharynx). "
                           f"Lateral neck conduit removed. y_anterior_is_low={y_ant_low}.")
    bc.setdefault("trim", {})
    bc["trim"] = {"choanal_cut_y_mm": cut_y_mm,
                  "removed_voxels": int(removed.sum()),
                  "y_anterior_is_low": y_ant_low,
                  "superior_is_high_z": sup_high_z}
    bc_path.write_text(json.dumps(bc, indent=1), encoding="utf-8")
    print(f"[{args.case}] wrote corrected mask + BC (originals -> *.pretrim)")

    if args.no_flow:
        return 0

    from sinus_cfd.flow_field import compute_flow_field
    from sinus_cfd.physiology import PatientBreathing
    b = bc.get("breathing") or {}
    breathing = (PatientBreathing(
        patient_id=args.case, tidal_volume_L=float(b.get("tidal_volume_L", 0.5)),
        respiratory_rate_per_min=float(b.get("respiratory_rate_per_min", 12)),
        inspiratory_fraction=float(b.get("inspiratory_fraction", 1 / 3)),
        left_nostril_flow_fraction=float(b.get("left_nostril_flow_fraction", 0.5)),
        right_nostril_flow_fraction=float(b.get("right_nostril_flow_fraction", 0.5)))
        if b else PatientBreathing.typical_resting_adult(patient_id=args.case))
    print(f"[{args.case}] re-solving flow (nostrils → nasopharynx)…")
    compute_flow_field(
        airway_mask_path=mask_path, boundary_json_path=bc_path, output_dir=cdir,
        case_id=args.case, breathing=breathing,
        pressure_iterations=args.flow_iterations, n_streamline_seeds=args.streamline_seeds)
    print(f"[{args.case}] done — flow.npz + streamlines.json refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
