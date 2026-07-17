#!/usr/bin/env python3
"""
Surgical guidance for the Streamlit viewer (docs/surgical_guidance.md).

  1. Label frontal / sphenoid / maxillary L/R (CT air heuristics)
  2. Dual instrument paths: L/R naris → ipsilateral frontal
  3. High-|u| corridor along naris → trachea
  4. Split into inferior turbinate / middle turbinate / septum
  5. Rank least-invasive treatment options

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
    split_frontal_lr,
    straighten_path_in_air,
    smooth_instrument_path,
    build_lateral_diverge_frontal_path,
    restriction_along_paths_high_speed,
    _mm_to_zyx,
)
from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402
from sinus_cfd.sinus_anatomy import detect_paranasal_sinuses  # noqa: E402
from sinus_cfd.surgical_zones import (  # noqa: E402
    classify_removal_zones,
    points_from_mask,
    recommend_treatments,
    zones_to_meta,
)


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

    # Split frontal into L/R targets (ipsilateral instrument paths)
    frontal_on_domain = anatomy.frontal & air_domain
    if not frontal_on_domain.any():
        frontal_on_domain = anatomy.frontal
    frontal_left, frontal_right, x_sep = split_frontal_lr(frontal_on_domain)
    notes.append(
        f"Frontal split L={int(frontal_left.sum())} R={int(frontal_right.sum())} "
        f"x_sep={x_sep:.1f}"
    )

    def _frontal_target_lateral(mask: np.ndarray, side: str) -> tuple[int, int, int]:
        """Most superior-anterior-lateral voxel in this frontal half."""
        m = mask & air_domain
        if not m.any():
            m = mask if mask.any() else frontal_on_domain
        zz, yy, xx = np.where(m)
        z_thr = float(np.percentile(zz, 60))
        y_thr = float(np.percentile(yy, 45))
        sel = (zz >= z_thr) & (yy <= y_thr)
        if int(sel.sum()) < 8:
            sel = zz >= float(np.percentile(zz, 50))
        zz2, yy2, xx2 = zz[sel], yy[sel], xx[sel]
        if side == "left":
            # extreme high-x among superior/anterior
            j2 = int(np.argmax(xx2.astype(float) + 0.15 * zz2 - 0.1 * yy2))
        else:
            j2 = int(np.argmin(xx2.astype(float) - 0.15 * zz2 + 0.1 * yy2))
        return nearest_air_index(air_domain, (int(zz2[j2]), int(yy2[j2]), int(xx2[j2])))

    tgt_left = _frontal_target_lateral(frontal_left, "left")
    tgt_right = _frontal_target_lateral(frontal_right, "right")
    # Force ends farther lateral than starts for clear coronal diverge
    def _push_lateral(tgt, naris_mm, side):
        tmm = path_zyx_to_mm([tgt], spacing, origin)[0]
        if side == "left" and tmm[0] < naris_mm[0] + 6:
            tmm[0] = naris_mm[0] + 10.0
        if side == "right" and tmm[0] > naris_mm[0] - 6:
            tmm[0] = naris_mm[0] - 10.0
        return nearest_air_index(
            air_domain, _mm_to_zyx(tmm, spacing, origin, shape)
        )

    cost, radius = most_open_cost_hu(
        air_domain, hu, spacing, power=1.15, hu_weight=0.35
    )
    paths: dict[str, list] = {}
    path_meta: list[dict] = []
    all_path_zyx: list[tuple[int, int, int]] = []

    ordered_nares = sorted(naris_pts[:2], key=lambda c: -float(c[0]))
    side_targets = {
        "left": (ordered_nares[0], tgt_left),
        "right": (
            ordered_nares[1] if len(ordered_nares) > 1 else ordered_nares[0],
            tgt_right,
        ),
    }
    for side, (nm, tgt) in side_targets.items():
        tgt = _push_lateral(tgt, nm, side)
        end_mm = path_zyx_to_mm([tgt], spacing, origin)[0]
        # Pure design path: straight in sagittal, flare laterally in coronal
        pts = build_lateral_diverge_frontal_path(
            nm,
            end_mm,
            air_domain,
            spacing,
            origin,
            side=side,
            n=56,
            lateral_flare=1.4,
        )
        # Do NOT re-route with nearest-air smooth (destroys sagittal straightness)
        idx_s = [
            nearest_air_index(air_domain, _mm_to_zyx(p, spacing, origin, shape))
            for p in pts
        ]
        key = f"naris_{side}_to_frontal_{side}"
        paths[key] = pts.tolist()
        paths[f"naris_{side}_to_frontal"] = pts.tolist()
        plen = path_length_mm(pts)
        r_along = [float(radius[p]) for p in idx_s]
        chord = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
        if len(pts) >= 3:
            t = np.linspace(0, 1, len(pts))
            y_line = (1 - t) * pts[0, 1] + t * pts[-1, 1]
            z_line = (1 - t) * pts[0, 2] + t * pts[-1, 2]
            sag_err = float(
                np.sqrt(np.mean((pts[:, 1] - y_line) ** 2 + (pts[:, 2] - z_line) ** 2))
            )
        else:
            sag_err = 0.0
        path_meta.append(
            {
                "name": key,
                "side": side,
                "start_mm": list(nm),
                "end_mm": pts[-1].tolist(),
                "length_mm": plen,
                "chord_mm": chord,
                "straightness": float(chord / max(plen, 1e-3)),
                "sagittal_rms_mm": sag_err,
                "dx_lateral_mm": float(pts[-1, 0] - pts[0, 0]),
                "min_radius_mm": float(min(r_along)) if r_along else 0.0,
                "mean_radius_mm": float(np.mean(r_along)) if r_along else 0.0,
                "n_points": len(pts),
            }
        )
        all_path_zyx.extend(idx_s)
        notes.append(
            f"{key}: len={plen:.1f} mm straight={path_meta[-1]['straightness']:.2f} "
            f"sag_rms={sag_err:.1f} mm dx={path_meta[-1]['dx_lateral_mm']:.1f} mm "
            f"x {pts[0,0]:.0f}→{pts[-1,0]:.0f}"
        )

    notes.append(
        "Dual instrument paths: sagittal-straight, coronal lateral diverge L/R"
    )

    # Magenta = high-velocity zones along naris→trachea inhale pathways
    # (places a larger opening could relieve resistance / peak speed)
    speed = npz["speed"].astype(float)
    airway_flow = npz["airway"].astype(bool)

    # Build inhale path corridors from BC nares → trachea (open-path style)
    trachea_mm = None
    bc_path = out / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports", []):
            if port.get("role") == "outlet" and port.get("center_mm"):
                trachea_mm = [float(v) for v in port["center_mm"]]
    inhale_paths_mm: list = []
    if trachea_mm is not None:
        for nm in naris_pts[:2]:
            s = nearest_air_index(airway_flow, _mm_to_zyx(nm, spacing, origin, shape))
            e = nearest_air_index(
                airway_flow, _mm_to_zyx(trachea_mm, spacing, origin, shape)
            )
            idx_inh = most_open_path_zyx(
                airway_flow, s, e, spacing, power=1.8, hu=hu, hu_weight=0.3
            )
            inhale_paths_mm.append(path_zyx_to_mm(idx_inh, spacing, origin))
        notes.append(f"Inhale corridors for restriction: {len(inhale_paths_mm)} naris→trachea")

    # Prefer streamlines if present (true flow)
    sl_path = out / f"{case_id}_streamlines.json"
    if sl_path.is_file():
        try:
            sl = json.loads(sl_path.read_text(encoding="utf-8"))
            for line in (sl.get("lines") or [])[:80]:
                arr = np.asarray(line, dtype=float)
                if len(arr) >= 8:
                    inhale_paths_mm.append(arr[:: max(1, len(arr) // 40)])
            notes.append("Added streamline samples to restriction corridor.")
        except Exception as exc:
            notes.append(f"Streamline load for restriction skipped: {exc}")

    highlight, rem_pts = restriction_along_paths_high_speed(
        inhale_paths_mm,
        speed,
        airway_flow,
        spacing,
        origin,
        speed_percentile=80.0,
        tube_radius_mm=5.0,
        max_points=6500,
    )
    # Also keep narrow bottlenecks on frontal instrument paths (secondary)
    fb_mask, bottlenecks = path_restriction_highlights(
        all_path_zyx,
        air_domain,
        spacing,
        origin,
        radius=radius,
        narrow_percentile=22.0,
        ball_radius=1,
    )
    highlight = highlight | (fb_mask & airway_flow)
    notes.append(
        f"Removal (high |u| on naris→trachea): {int(highlight.sum())} vx; "
        f"frontal bottlenecks={len(bottlenecks)}"
    )

    # Classify into inferior turbinate / middle turbinate / septum
    nasal_for_zone = nasal if nasal.any() else airway_flow
    zone_masks, zone_stats, znotes = classify_removal_zones(
        highlight,
        speed,
        spacing,
        origin,
        naris_mm=naris_pts,
        nasal_mask=nasal_for_zone,
    )
    notes.extend(znotes)
    treatments = recommend_treatments(zone_stats)
    zone_meta = zones_to_meta(zone_stats, treatments, znotes)

    _write_mask(highlight, out / f"{case_id}_removal_highlight.nrrd", spacing, origin)
    _export_stl(highlight, out / f"{case_id}_removal_highlight.stl", spacing, origin, faces=8000)

    # Combined + per-zone point clouds for viewer toggles
    zone_pts = {}
    for zkey, zmask in zone_masks.items():
        _write_mask(zmask, out / f"{case_id}_removal_{zkey}.nrrd", spacing, origin)
        zone_pts[zkey] = points_from_mask(
            zmask, speed, spacing, origin, max_points=3500, seed=hash(zkey) % 997
        )
        notes.append(f"Zone {zkey}: {int(zmask.sum())} vx → {len(zone_pts[zkey])} pts")

    zz, yy, xx = np.where(highlight)
    if len(zz) > 6000:
        rng = np.random.default_rng(5)
        pick = rng.choice(len(zz), size=6000, replace=False)
        zz, yy, xx = zz[pick], yy[pick], xx[pick]
    rem_pts = np.column_stack(
        [
            origin[0] + xx * spacing[0],
            origin[1] + yy * spacing[1],
            origin[2] + zz * spacing[2],
            speed[zz, yy, xx] if len(zz) else np.array([]),
        ]
    ).astype(np.float32)
    np.savez_compressed(
        out / f"{case_id}_removal_highlight.npz",
        points_xyz_r_mm=rem_pts,
        n_points=np.int32(len(zz)),
        metric="speed_m_s_along_naris_trachea",
        inferior_turbinate=zone_pts["inferior_turbinate"],
        middle_turbinate=zone_pts["middle_turbinate"],
        septum=zone_pts["septum"],
    )

    guidance = {
        "case_id": case_id,
        "method": "straight_most_open_dual_naris_to_ipsilateral_frontal",
        "sinus_anatomy": anatomy.to_meta(),
        "paths_mm": paths,
        "path_metrics": path_meta,
        "bottlenecks": bottlenecks,
        "dual_frontal_paths": True,
        "removal_zones": zone_meta,
        "notes": notes,
        "viewer": {
            "frontal_path_color": "purple",
            "show_both_frontal_paths": True,
            "removal_color": "magenta",
            "zone_keys": [
                "inferior_turbinate",
                "middle_turbinate",
                "septum",
            ],
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
    (out / f"{case_id}_treatment_recommendations.json").write_text(
        json.dumps(zone_meta, indent=2), encoding="utf-8"
    )

    print(f"OK surgical guidance case={case_id}")
    print(f"  frontal={int(anatomy.frontal.sum())} sphenoid={int(anatomy.sphenoid.sum())}")
    print(
        f"  maxillary L/R={int(anatomy.maxillary_left.sum())}/"
        f"{int(anatomy.maxillary_right.sum())}"
    )
    print(f"  paths={list(paths.keys())}")
    print(f"  removal voxels={int(highlight.sum())}")
    for z in zone_stats:
        print(
            f"  zone {z.name}: n={z.voxels} severity={z.severity} "
            f"mean|u|={z.mean_speed_m_s:.2f}"
        )
    print("  Recommended treatments:")
    for t in treatments:
        if t.recommended:
            print(f"    * {t.name} ({t.reason})")
    for n in notes[-6:]:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
