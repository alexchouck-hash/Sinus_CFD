#!/usr/bin/env python3
"""
Surgical guidance layers for the Streamlit viewer:

  1. Label frontal / sphenoid / maxillary L/R sinuses (CT air heuristics)
  2. Least-resistance instrument path: nostril → frontal sinus
     (most-open: high distance-to-wall + dark HU air)
  3. Magenta/pink "areas to remove" = narrowest bottlenecks along that path

Example:
  py -3.12 scripts/compute_surgical_guidance.py --case VisibleHuman_Head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage import morphology

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.open_path import (  # noqa: E402
    most_open_cost_hu,
    most_open_path_zyx,
    path_length_mm,
    path_restriction_highlights,
    path_zyx_to_mm,
    nearest_air_index,
    _mm_to_zyx,
)
from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402
from sinus_cfd.sinus_anatomy import detect_paranasal_sinuses  # noqa: E402


def _write_mask(mask, path: Path, spacing, origin) -> None:
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, str(path))


def _export_stl(mask, path: Path, spacing, origin, faces: int = 12000) -> int:
    if not mask.any() or int(mask.sum()) < 30:
        return 0
    mesh = _mask_to_mesh(mask, spacing, origin)
    try:
        if len(mesh.faces) > faces:
            mesh = mesh.simplify_quadric_decimation(faces)
    except Exception:
        pass
    mesh.export(str(path))
    return int(len(mesh.faces))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="VisibleHuman_Head")
    ap.add_argument("--air-hu-max", type=float, default=-280.0)
    args = ap.parse_args()

    case_id = args.case
    out = REPO_ROOT / "outputs" / case_id
    notes: list[str] = []

    stats = json.loads((out / f"{case_id}_stats.json").read_text(encoding="utf-8"))
    npz = np.load(out / f"{case_id}_flow.npz")
    spacing = tuple(float(v) for v in npz["spacing_xyz_mm"])
    origin = tuple(float(v) for v in npz["origin_xyz_mm"])
    shape = tuple(int(v) for v in npz["airway"].shape)

    crop = [int(v) for v in (stats.get("crop_origin_zyx") or [0, 0, 0])]
    image_path = Path(stats.get("image_path") or "")
    if not image_path.is_file():
        image_path = REPO_ROOT / "data" / "VisibleHuman_Head" / "VHFCT1mm_Head.nrrd"
    hu_full = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    hu = hu_full[
        crop[0] : crop[0] + shape[0],
        crop[1] : crop[1] + shape[1],
        crop[2] : crop[2] + shape[2],
    ]
    body = sitk.GetArrayFromImage(
        sitk.ReadImage(str(out / f"{case_id}_head_mask.nrrd"))
    ).astype(bool)

    nasal = np.zeros(shape, dtype=bool)
    for side in ("left", "right"):
        cp = out / f"{case_id}_cavity_{side}.nrrd"
        if cp.is_file():
            nasal |= sitk.GetArrayFromImage(sitk.ReadImage(str(cp))).astype(bool)

    superior = bool(stats.get("superior_is_high_z", True))
    y_ant_low = True
    for n in stats.get("notes") or []:
        if "y_anterior_is_low=False" in str(n):
            y_ant_low = False

    anatomy = detect_paranasal_sinuses(
        hu,
        body,
        spacing,
        origin,
        case_id=case_id,
        air_hu_max=args.air_hu_max,
        superior_is_high_z=superior,
        y_anterior_is_low=y_ant_low,
        nasal_mask=nasal if nasal.any() else None,
    )
    notes.extend(anatomy.notes)

    # Write sinus masks + STLs
    _write_mask(anatomy.frontal, out / f"{case_id}_sinus_frontal.nrrd", spacing, origin)
    _write_mask(anatomy.sphenoid, out / f"{case_id}_sinus_sphenoid.nrrd", spacing, origin)
    _write_mask(
        anatomy.maxillary_left, out / f"{case_id}_sinus_maxillary_left.nrrd", spacing, origin
    )
    _write_mask(
        anatomy.maxillary_right,
        out / f"{case_id}_sinus_maxillary_right.nrrd",
        spacing,
        origin,
    )
    for name, m in (
        ("sinus_frontal", anatomy.frontal),
        ("sinus_sphenoid", anatomy.sphenoid),
        ("sinus_maxillary_left", anatomy.maxillary_left),
        ("sinus_maxillary_right", anatomy.maxillary_right),
    ):
        nf = _export_stl(m, out / f"{case_id}_{name}.stl", spacing, origin)
        notes.append(f"STL {name}: {nf} faces")

    # Domain for instrument path: connected air including nasal + frontal
    air_domain = body & (hu <= args.air_hu_max + 40) & (hu >= -1024)
    air_domain = morphology.closing(air_domain, footprint=morphology.ball(1))
    air_domain |= nasal
    air_domain |= anatomy.frontal
    # Keep largest component touching a naris if possible
    lab, nlab = __import__("scipy.ndimage", fromlist=["label"]).label(air_domain)
    # Naris seeds
    nares_path = out / f"{case_id}_nares.json"
    naris_pts: list[list[float]] = []
    if nares_path.is_file():
        nj = json.loads(nares_path.read_text(encoding="utf-8"))
        for p in nj.get("naris_points") or []:
            if p.get("center_mm"):
                naris_pts.append([float(v) for v in p["center_mm"]])
    if len(naris_pts) < 1:
        bc = json.loads((out / f"{case_id}_boundary_conditions.json").read_text())
        for port in bc.get("ports", []):
            if port.get("role") == "inlet" and port.get("center_mm"):
                naris_pts.append([float(v) for v in port["center_mm"]])

    if not naris_pts:
        print("No naris centers found.")
        return 1
    if not anatomy.frontal.any():
        print("No frontal sinus air found — check HU / ROI.")
        # still write anatomy meta
        (out / f"{case_id}_sinus_anatomy.json").write_text(
            json.dumps(anatomy.to_meta(), indent=2), encoding="utf-8"
        )
        return 1

    # Keep air components that touch naris or frontal
    keep = np.zeros_like(air_domain)
    for nm in naris_pts:
        zyx = nearest_air_index(air_domain, _mm_to_zyx(nm, spacing, origin, shape))
        keep[lab == lab[zyx]] = True
    fz, fy, fx = np.where(anatomy.frontal)
    for i in range(0, len(fz), max(1, len(fz) // 40)):
        keep[lab == lab[fz[i], fy[i], fx[i]]] = True
    if keep.any():
        air_domain = keep
    notes.append(f"Instrument domain air voxels={int(air_domain.sum())}")

    # Frontal target: centroid of frontal mask on domain
    fz, fy, fx = np.where(anatomy.frontal & air_domain)
    if len(fz) == 0:
        fz, fy, fx = np.where(anatomy.frontal)
    frontal_zyx = (int(np.median(fz)), int(np.median(fy)), int(np.median(fx)))
    frontal_zyx = nearest_air_index(air_domain, frontal_zyx)

    # Paths from each naris (and best overall)
    cost, radius = most_open_cost_hu(air_domain, hu, spacing, power=2.2, hu_weight=0.6)
    paths: dict[str, list] = {}
    path_meta: list[dict] = []
    all_path_zyx: list[tuple[int, int, int]] = []

    # higher x = patient left in this CT
    ordered_nares = sorted(naris_pts[:2], key=lambda c: -float(c[0]))
    for i, nm in enumerate(ordered_nares):
        side = "left" if i == 0 else "right"
        start = nearest_air_index(air_domain, _mm_to_zyx(nm, spacing, origin, shape))
        idx = most_open_path_zyx(
            air_domain,
            start,
            frontal_zyx,
            spacing,
            power=2.2,
            hu=hu,
            hu_weight=0.6,
        )
        pts = path_zyx_to_mm(idx, spacing, origin)
        key = f"naris_{side}_to_frontal"
        paths[key] = pts.tolist()
        plen = path_length_mm(pts)
        r_along = [float(radius[p]) for p in idx]
        path_meta.append(
            {
                "name": key,
                "start_mm": list(nm),
                "end_mm": path_zyx_to_mm([frontal_zyx], spacing, origin)[0].tolist(),
                "length_mm": plen,
                "min_radius_mm": float(min(r_along)) if r_along else 0.0,
                "mean_radius_mm": float(np.mean(r_along)) if r_along else 0.0,
                "n_points": len(idx),
            }
        )
        all_path_zyx.extend(idx)
        notes.append(
            f"{key}: length={plen:.1f} mm min_r={path_meta[-1]['min_radius_mm']:.2f} mm"
        )

    # Prefer dual if both exist; also store primary = shorter min-radius-aware score
    if path_meta:
        best = min(
            path_meta,
            key=lambda m: m["length_mm"] / max(m["mean_radius_mm"], 0.5),
        )
        paths["primary_naris_to_frontal"] = paths[best["name"]]
        notes.append(f"Primary instrument path: {best['name']}")

    # Magenta removal zones: bottlenecks on all frontal access paths
    highlight, bottlenecks = path_restriction_highlights(
        all_path_zyx,
        air_domain,
        spacing,
        origin,
        radius=radius,
        narrow_percentile=28.0,
        ball_radius=2,
    )
    # Also flag narrow junctions near frontal (ostium-like)
    if anatomy.frontal.any():
        fringe = morphology.binary_dilation(anatomy.frontal, footprint=morphology.ball(2))
        fringe = fringe & air_domain & ~anatomy.frontal
        narrow = fringe & (radius <= max(1.2, float(np.percentile(radius[air_domain], 20))))
        highlight |= narrow
    notes.append(
        f"Removal highlight voxels={int(highlight.sum())} bottlenecks={len(bottlenecks)}"
    )

    _write_mask(highlight, out / f"{case_id}_removal_highlight.nrrd", spacing, origin)
    _export_stl(highlight, out / f"{case_id}_removal_highlight.stl", spacing, origin, faces=8000)

    # Point cloud for viewer (magenta)
    zz, yy, xx = np.where(highlight)
    if len(zz) > 5000:
        rng = np.random.default_rng(5)
        pick = rng.choice(len(zz), size=5000, replace=False)
        zz, yy, xx = zz[pick], yy[pick], xx[pick]
    rem_pts = np.column_stack(
        [
            origin[0] + xx * spacing[0],
            origin[1] + yy * spacing[1],
            origin[2] + zz * spacing[2],
            radius[zz, yy, xx],
        ]
    ).astype(np.float32)
    np.savez_compressed(
        out / f"{case_id}_removal_highlight.npz",
        points_xyz_r_mm=rem_pts,
        n_points=np.int32(len(zz)),
    )

    guidance = {
        "case_id": case_id,
        "method": "most_open_hu_edt_naris_to_frontal",
        "sinus_anatomy": anatomy.to_meta(),
        "paths_mm": paths,
        "path_metrics": path_meta,
        "bottlenecks": bottlenecks,
        "notes": notes,
        "viewer": {
            "frontal_path_color": "purple",
            "removal_color": "magenta",
            "sinus_colors": {
                "frontal": "#ffcc80",
                "sphenoid": "#80cbc4",
                "maxillary_left": "#90caf9",
                "maxillary_right": "#81d4fa",
            },
        },
    }
    (out / f"{case_id}_surgical_guidance.json").write_text(
        json.dumps(guidance, indent=2), encoding="utf-8"
    )
    (out / f"{case_id}_sinus_anatomy.json").write_text(
        json.dumps(anatomy.to_meta(), indent=2), encoding="utf-8"
    )

    print(f"OK surgical guidance case={case_id}")
    print(f"  frontal={int(anatomy.frontal.sum())} sphenoid={int(anatomy.sphenoid.sum())}")
    print(
        f"  maxillary L/R={int(anatomy.maxillary_left.sum())}/"
        f"{int(anatomy.maxillary_right.sum())}"
    )
    print(f"  paths={list(paths.keys())}")
    print(f"  removal voxels={int(highlight.sum())} bottlenecks={len(bottlenecks)}")
    for n in notes[-8:]:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
