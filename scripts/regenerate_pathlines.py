#!/usr/bin/env python3
"""
Regenerate dense inhale pathlines + tip-accurate naris markers from existing
OpenFOAM-mapped flow field (no foam re-solve).

Also writes a restriction field (distance-to-wall / 1/r) for the viewer.

Example:
  py -3.12 scripts/regenerate_pathlines.py --case VisibleHuman_Head --seeds 240
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.flow_field import (  # noqa: E402
    compute_inhale_streamlines,
    extend_paths_to_outlet_via_centerline,
)


def _world(origin: np.ndarray, spacing: np.ndarray, z: int, y: int, x: int) -> list[float]:
    return [
        float(origin[0] + x * spacing[0]),
        float(origin[1] + y * spacing[1]),
        float(origin[2] + z * spacing[2]),
    ]


def tip_naris_from_opening(
    opening: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    y_anterior_is_low: bool = True,
    anterior_pct: float = 25.0,
    air_mask: np.ndarray | None = None,
) -> tuple[list[float] | None, list[float] | None, list[str]]:
    """
    Place each naris on the *anterior air opening* (must touch real air).

    Earlier "tip fringe" walked off the lumen into exterior free air, so
    markers sat outside the airway and all pathlines collapsed to one side.
    Prefer opening ∩ (air / inlet), then the more-anterior half of that set.
    """
    notes: list[str] = []
    open_m = opening.astype(bool)
    if air_mask is not None:
        air_m = air_mask.astype(bool)
        # Prefer opening voxels that are true air, else air dilated into opening
        on_air = open_m & air_m
        if int(on_air.sum()) < 30:
            on_air = open_m & ndi.binary_dilation(air_m, iterations=2)
        if int(on_air.sum()) >= 20:
            open_m = on_air
            notes.append("Naris markers constrained to opening ∩ airway.")
        else:
            notes.append("Opening∩air small — using full opening shell.")

    zz, yy, xx = np.where(open_m)
    if len(zz) < 10:
        return None, None, ["Opening mask too small for tip naris."]

    # L/R by x median (high x = patient left in this CT convention)
    xmed = float(np.median(xx))
    left_mm = right_mm = None
    for side, mask in (
        ("left", xx >= xmed),
        ("right", xx < xmed),
    ):
        if int(mask.sum()) < 5:
            notes.append(f"Tip naris {side}: too few voxels.")
            continue
        zs, ys, xs = zz[mask], yy[mask], xx[mask]
        if y_anterior_is_low:
            thr = float(np.percentile(ys, anterior_pct))
            keep = ys <= thr
        else:
            thr = float(np.percentile(ys, 100.0 - anterior_pct))
            keep = ys >= thr
        if int(keep.sum()) < 5:
            keep = np.ones(len(ys), dtype=bool)
        zs, ys, xs = zs[keep], ys[keep], xs[keep]
        zc = int(round(float(np.median(zs))))
        yc = int(round(float(np.median(ys))))
        xc = int(round(float(np.median(xs))))
        d = (zs - zc) ** 2 + (ys - yc) ** 2 + (xs - xc) ** 2
        j = int(np.argmin(d))
        zc, yc, xc = int(zs[j]), int(ys[j]), int(xs[j])
        # Snap onto air if available
        if air_mask is not None and not air_mask[zc, yc, xc]:
            az, ay, ax = np.where(air_mask)
            if len(az):
                dd = (az - zc) ** 2 + (ay - yc) ** 2 + (ax - xc) ** 2
                j2 = int(np.argmin(dd))
                zc, yc, xc = int(az[j2]), int(ay[j2]), int(ax[j2])
        mm = _world(origin, spacing, zc, yc, xc)
        notes.append(
            f"Naris {side}: zyx=({zc},{yc},{xc}) mm={mm} "
            f"(anterior {anterior_pct:.0f}% of opening∩air)."
        )
        if side == "left":
            left_mm = mm
        else:
            right_mm = mm
    return left_mm, right_mm, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="VisibleHuman_Head")
    ap.add_argument("--seeds", type=int, default=240, help="Total seed budget (~half per naris)")
    ap.add_argument("--max-lines", type=int, default=200)
    ap.add_argument("--step-mm", type=float, default=0.28)
    args = ap.parse_args()

    case_id = args.case
    out = REPO_ROOT / "outputs" / case_id
    npz_path = out / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        print(f"Missing {npz_path}")
        return 1

    data = np.load(npz_path)
    airway = data["airway"].astype(bool)
    ux = data["ux"].astype(float)
    uy = data["uy"].astype(float)
    uz = data["uz"].astype(float)
    speed = data["speed"].astype(float)
    spacing = tuple(float(v) for v in data["spacing_xyz_mm"])
    origin = tuple(float(v) for v in data["origin_xyz_mm"])
    sp_arr = np.asarray(spacing, dtype=float)
    org_arr = np.asarray(origin, dtype=float)

    notes: list[str] = []

    # Domain for pathlines: L+R nasal cavities + passage (NOT lumen alone —
    # VisibleHuman passage_lumen historically only covered the left side).
    domain = airway.copy()
    cav = np.zeros_like(airway)
    for side in ("left", "right"):
        cp = out / f"{case_id}_cavity_{side}.nrrd"
        if cp.is_file():
            carr = sitk.GetArrayFromImage(sitk.ReadImage(str(cp))).astype(bool)
            if carr.shape == airway.shape:
                cav |= carr
    lumen_p = out / f"{case_id}_passage_lumen.nrrd"
    if lumen_p.is_file():
        limg = sitk.ReadImage(str(lumen_p))
        lumen = sitk.GetArrayFromImage(limg).astype(bool)
        if lumen.shape == airway.shape:
            cav |= lumen
    if cav.any():
        domain = airway & cav
        notes.append(
            f"Pathlines on L/R cavities + passage ({int(domain.sum())} vx; "
            f"avoids left-only lumen bias)."
        )
    else:
        notes.append(f"Pathlines on full airway mask ({int(domain.sum())} vx).")

    # --- Naris markers on real air openings (not exterior free air) ---
    left_tip = right_tip = None
    opening_path = out / f"{case_id}_naris_opening_ct.nrrd"
    if opening_path.is_file():
        oimg = sitk.ReadImage(str(opening_path))
        opening = sitk.GetArrayFromImage(oimg).astype(bool)
        left_tip, right_tip, nnotes = tip_naris_from_opening(
            opening,
            sp_arr,
            org_arr,
            y_anterior_is_low=True,
            anterior_pct=30.0,
            air_mask=domain,
        )
        notes.extend(nnotes)
    else:
        notes.append("No naris_opening_ct.nrrd — keeping existing BC naris centers.")

    # Prefer existing skin-tip vestibule nares (from extend_nasal_to_tip.py)
    inlet_centers: list[list[float]] = []
    skin_naris: list[list[float]] = []
    nares_json = out / f"{case_id}_nares.json"
    tip_method = None
    if nares_json.is_file():
        nj = json.loads(nares_json.read_text(encoding="utf-8"))
        tip_method = nj.get("method")
        for npnt in nj.get("naris_points") or []:
            if npnt.get("center_mm"):
                skin_naris.append([float(v) for v in npnt["center_mm"]])
        if tip_method == "skin_tip_vestibule" and len(skin_naris) >= 2:
            inlet_centers = list(skin_naris)
            left_tip, right_tip = skin_naris[0], skin_naris[1]
            notes.append(
                f"Using skin-tip vestibule nares as inlets: L={left_tip} R={right_tip}"
            )

    # Else: cavity most-anterior air (strict anterior fringe, not 20% of whole cavity)
    if len(inlet_centers) < 2:
        inlet_centers = []
        for side, mask_name in (("left", "cavity_left"), ("right", "cavity_right")):
            cp = out / f"{case_id}_{mask_name}.nrrd"
            if not cp.is_file():
                continue
            carr = sitk.GetArrayFromImage(sitk.ReadImage(str(cp))).astype(bool)
            if carr.shape != airway.shape or not carr.any():
                continue
            m = carr & domain
            if not m.any():
                m = carr & airway
            zz, yy, xx = np.where(m)
            if len(zz) < 5:
                continue
            # Only the very front of each cavity (true opening)
            y_thr = float(np.percentile(yy, 8))
            keep = yy <= y_thr
            if int(keep.sum()) < 5:
                keep = yy <= float(np.percentile(yy, 20))
            zz, yy, xx = zz[keep], yy[keep], xx[keep]
            inlet_centers.append(
                [
                    float(org_arr[0] + xx.mean() * sp_arr[0]),
                    float(org_arr[1] + yy.mean() * sp_arr[1]),
                    float(org_arr[2] + zz.mean() * sp_arr[2]),
                ]
            )
            notes.append(
                f"Cavity {side} tip-fringe inlet: "
                f"{[round(v, 1) for v in inlet_centers[-1]]}"
            )

    # Fallback: passage inlet_open split
    if len(inlet_centers) < 2:
        inlet_open_p = out / f"{case_id}_passage_inlet_open.nrrd"
        if inlet_open_p.is_file():
            iimg = sitk.ReadImage(str(inlet_open_p))
            im = sitk.GetArrayFromImage(iimg).astype(bool)
            iz_, iy_, ix_ = np.where(im)
            if len(ix_):
                sp_i = iimg.GetSpacing()
                org_i = iimg.GetOrigin()
                xmed = float(np.median(ix_))
                inlet_centers = []
                for mask_x in (ix_ >= xmed, ix_ < xmed):
                    if not mask_x.any():
                        continue
                    inlet_centers.append(
                        [
                            float(org_i[0] + ix_[mask_x].mean() * sp_i[0]),
                            float(org_i[1] + iy_[mask_x].mean() * sp_i[1]),
                            float(org_i[2] + iz_[mask_x].mean() * sp_i[2]),
                        ]
                    )
                notes.append(f"Fallback inlet_open L/R: {inlet_centers}")

    # Keep tip markers: do not overwrite skin_tip_vestibule with deeper points
    if tip_method == "skin_tip_vestibule" and len(skin_naris) >= 2:
        left_tip, right_tip = skin_naris[0], skin_naris[1]
        notes.append("Preserved skin_tip_vestibule naris markers (not deep cavity).")
    elif len(inlet_centers) >= 2:
        left_tip, right_tip = inlet_centers[0], inlet_centers[1]
        skin_naris = list(inlet_centers)
        notes.append("Naris markers set to tip-fringe air openings (L, R).")
    elif not skin_naris and left_tip and right_tip:
        skin_naris = [left_tip, right_tip]
    if not inlet_centers and skin_naris:
        inlet_centers = list(skin_naris)

    # Outlet / centerline for path completion
    outlet_center = None
    centerline_mm = None
    passage_json = out / f"{case_id}_passage.json"
    if passage_json.is_file():
        pj = json.loads(passage_json.read_text(encoding="utf-8"))
        cl = pj.get("centerline_mm") or []
        if len(cl) >= 2:
            centerline_mm = cl
            outlet_center = [float(v) for v in cl[-1]]
    open_paths = out / f"{case_id}_open_paths.json"
    if open_paths.is_file() and centerline_mm is None:
        op = json.loads(open_paths.read_text(encoding="utf-8"))
        cl = op.get("centerline_mid_mm") or op.get("centerline_left_mm") or []
        if len(cl) >= 2:
            centerline_mm = cl
            outlet_center = [float(v) for v in cl[-1]]

    bc_path = out / f"{case_id}_boundary_conditions.json"
    if outlet_center is None and bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports", []):
            if port.get("role") == "outlet" and port.get("center_mm"):
                outlet_center = [float(v) for v in port["center_mm"]]

    # Dual-naris pathlines (~50% each): integrate per inlet, keep balanced
    n_sides = max(1, len(inlet_centers))
    n_per = max(24, args.seeds // n_sides)
    per_side_keep = max(20, args.max_lines // n_sides)
    lines: list = []
    for si, inlet in enumerate(inlet_centers):
        skin_one = None
        if skin_naris:
            # Match skin naris to this inlet by nearest x
            skin_one = [
                min(
                    skin_naris,
                    key=lambda s: abs(float(s[0]) - float(inlet[0])),
                )
            ]
        side_lines = compute_inhale_streamlines(
            ux,
            uy,
            uz,
            domain,
            spacing,
            origin,
            inlet_centers_mm=[inlet],
            outlet_center_mm=outlet_center,
            skin_naris_centers_mm=skin_one,
            n_per_naris=n_per,
            max_steps=1400,
            step_mm=args.step_mm,
            reach_outlet_mm=14.0,
            max_lines=per_side_keep,
            seed_radius_mm=9.0,
        )
        notes.append(
            f"Side {si} inlet={ [round(v,1) for v in inlet] }: "
            f"{len(side_lines)} pathlines (target ~{per_side_keep})."
        )
        lines.extend(side_lines)

    if centerline_mm is not None and outlet_center is not None and lines:
        lines = extend_paths_to_outlet_via_centerline(
            lines,
            np.asarray(centerline_mm, dtype=float),
            outlet_center,
            max_end_dist_mm=14.0,
        )
    notes.append(
        f"Balanced dual-naris pathlines: {len(lines)} total "
        f"(~50% seeds per nostril)."
    )

    # Sample speed along each path for viewer coloring
    ox, oy, oz = origin
    sx, sy, sz = spacing
    nz, ny, nx = speed.shape
    lines_out: list[dict] = []
    for line in lines:
        arr = np.asarray(line, dtype=float)
        if len(arr) < 4:
            continue
        ix = np.clip(np.rint((arr[:, 0] - ox) / sx).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((arr[:, 1] - oy) / sy).astype(int), 0, ny - 1)
        iz = np.clip(np.rint((arr[:, 2] - oz) / sz).astype(int), 0, nz - 1)
        sp = speed[iz, iy, ix].astype(float)
        lines_out.append(
            {
                "xyz_mm": arr.tolist(),
                "speed_m_s": sp.tolist(),
            }
        )

    sl_path = out / f"{case_id}_streamlines.json"
    with sl_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "source": "regenerate_pathlines",
                "n_lines": len(lines_out),
                "lines": [L["xyz_mm"] for L in lines_out],
                "speeds_m_s": [L["speed_m_s"] for L in lines_out],
                "notes": notes,
            },
            f,
        )
    notes.append(f"Wrote {len(lines_out)} pathlines → {sl_path.name}")

    # Restriction: narrow *medial* lumen (not wall surface shell).
    # EDT surface voxels always have r≈1 vx; use local-max radius (medial ridge)
    # and blend with speed so CFD constriction peaks light up.
    smin = float(min(spacing))
    edt = ndi.distance_transform_edt(domain) * smin
    inside = domain & (edt > 0)
    # Medial ridge: local maxima of radius inside airway
    local_max = (edt >= ndi.maximum_filter(edt, size=3) - 1e-6) & inside
    # Also include high-speed cores (continuity → narrow passages)
    sp_in = speed[inside]
    sp_thr = float(np.percentile(sp_in, 88)) if sp_in.size else 0.0
    fast = inside & (speed >= sp_thr) & (edt <= float(np.percentile(edt[inside], 55)))
    # Score: high when radius small on medial axis, or speed high in tight lumen
    restriction = np.zeros_like(edt, dtype=np.float32)
    inv_r = 1.0 / np.maximum(edt, 0.35)
    restriction[local_max] = (inv_r[local_max] * (1.0 + speed[local_max] / max(sp_thr, 1e-3))).astype(
        np.float32
    )
    restriction[fast] = np.maximum(
        restriction[fast],
        (inv_r[fast] * (1.5 + speed[fast] / max(sp_thr, 1e-3))).astype(np.float32),
    )
    # Keep only the tightest fraction among scored voxels
    scored = restriction > 0
    vals = restriction[scored]
    if vals.size:
        thr = float(np.percentile(vals, 70))
        hot = scored & (restriction >= thr)
        # Prefer truly narrow: drop wide ridges (r above ~median of hot)
        r_hot = edt[hot]
        if r_hot.size:
            r_cut = float(np.percentile(r_hot, 60))
            hot = hot & (edt <= max(r_cut, smin * 1.5))
    else:
        thr = 0.0
        hot = np.zeros_like(domain)
    zz, yy, xx = np.where(hot)
    # cap point count
    if len(zz) > 6000:
        rng = np.random.default_rng(3)
        pick = rng.choice(len(zz), size=6000, replace=False)
        zz, yy, xx = zz[pick], yy[pick], xx[pick]
    rest_pts = np.column_stack(
        [
            org_arr[0] + xx * sp_arr[0],
            org_arr[1] + yy * sp_arr[1],
            org_arr[2] + zz * sp_arr[2],
            restriction[zz, yy, xx],
            edt[zz, yy, xx],
        ]
    )
    rest_path = out / f"{case_id}_restriction.npz"
    min_r_hot = float(edt[hot].min()) if hot.any() else 0.0
    np.savez_compressed(
        rest_path,
        points_xyz_score_r_mm=rest_pts.astype(np.float32),
        threshold_1_over_r=np.float32(thr),
        max_restriction=np.float32(float(restriction[scored].max()) if scored.any() else 0),
        min_radius_mm=np.float32(min_r_hot),
        n_hot=np.int32(len(zz)),
    )
    notes.append(
        f"Restriction: score thr>={thr:.3f}, n_hot={len(zz)}, "
        f"min_r_hot={min_r_hot:.2f} mm, speed_thr={sp_thr:.3f} m/s"
    )

    # Update nares.json + BC — keep skin_tip_vestibule method if already set
    if left_tip and right_tip:
        nares_path = out / f"{case_id}_nares.json"
        method = "skin_tip_vestibule" if tip_method == "skin_tip_vestibule" else "ct_naris_opening_air"
        nares_obj = {
            "method": method,
            "y_anterior_is_low": True,
            "naris_points": [
                {
                    "name": "left_nostril",
                    "center_mm": left_tip,
                    "method": method,
                    "notes": (
                        "Skin tip vestibule (left) — open air to end of nose."
                        if method == "skin_tip_vestibule"
                        else "Anterior CT naris opening ∩ airway (left)."
                    ),
                },
                {
                    "name": "right_nostril",
                    "center_mm": right_tip,
                    "method": method,
                    "notes": (
                        "Skin tip vestibule (right) — open air to end of nose."
                        if method == "skin_tip_vestibule"
                        else "Anterior CT naris opening ∩ airway (right)."
                    ),
                },
            ],
            "notes": notes[:12],
        }
        if nares_path.is_file():
            try:
                old = json.loads(nares_path.read_text(encoding="utf-8"))
                nares_obj["notes"] = list(old.get("notes") or []) + notes[:8]
            except Exception:
                pass
        nares_path.write_text(json.dumps(nares_obj, indent=2), encoding="utf-8")
        notes.append(f"Updated nares → {nares_path.name} method={method}")

        if bc_path.is_file():
            bc = json.loads(bc_path.read_text(encoding="utf-8"))
            for port in bc.get("ports", []):
                if port.get("name") == "left_nostril":
                    port["center_mm"] = left_tip
                    port["method"] = method
                    port["notes"] = "Left naris at nose tip; ~50% inspiratory flow."
                elif port.get("name") == "right_nostril":
                    port["center_mm"] = right_tip
                    port["method"] = method
                    port["notes"] = "Right naris at nose tip; ~50% inspiratory flow."
            bc_path.write_text(json.dumps(bc, indent=2), encoding="utf-8")
            notes.append("Updated BC naris centers.")

        ct_meta = out / f"{case_id}_ct_nasal_meta.json"
        if ct_meta.is_file():
            try:
                cm = json.loads(ct_meta.read_text(encoding="utf-8"))
                cm["left_naris_center_mm"] = left_tip
                cm["right_naris_center_mm"] = right_tip
                cm["naris_marker_method"] = method
                cm.setdefault("notes", []).append(f"Naris markers method={method}.")
                ct_meta.write_text(json.dumps(cm, indent=2), encoding="utf-8")
            except Exception as exc:
                notes.append(f"ct_nasal_meta update skipped: {exc}")

    print(f"OK pathlines={len(lines_out)} case={case_id}")
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
