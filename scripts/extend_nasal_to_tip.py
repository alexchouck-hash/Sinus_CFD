#!/usr/bin/env python3
"""
Extend L/R nasal cavities to the skin nose tip (external nostrils).

Visible Human 1 mm CT often seals the tip with partial-volume soft tissue
(~25–30 mm short of the face). This paints dual open-air vestibule corridors
from the anterior skin surface into each cavity so:
  - blue cavity meshes reach the end of the nose
  - nares sit on true tip openings
  - pathlines enter at the face and exit the trachea

Example:
  py -3.12 scripts/extend_nasal_to_tip.py --case VisibleHuman_Head
  py -3.12 scripts/regenerate_pathlines.py --case VisibleHuman_Head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import morphology

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402


def _world(origin, spacing, z, y, x) -> list[float]:
    return [
        float(origin[0] + x * spacing[0]),
        float(origin[1] + y * spacing[1]),
        float(origin[2] + z * spacing[2]),
    ]


def _decimate(mesh, target: int):
    if mesh is None or len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target)
    except Exception:
        return mesh


def _export_stl(mask, path: Path, spacing, origin, faces: int = 22000) -> int:
    if not mask.any() or int(mask.sum()) < 30:
        return 0
    mesh = _mask_to_mesh(mask, spacing, origin)
    mesh = _decimate(mesh, faces)
    mesh.export(str(path))
    return int(len(mesh.faces))


def _paint_tube(
    mask: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    radius: int,
    body: np.ndarray,
    x_sep: int | None,
    side: str,
    force: bool = True,
) -> np.ndarray:
    """Paint a cylindrical corridor start→end inside body (optionally force air)."""
    out = mask.copy()
    a = np.array(start, dtype=float)
    b = np.array(end, dtype=float)
    length = float(np.linalg.norm(b - a))
    nstep = max(int(np.ceil(length)) + 3, 4)
    nz, ny, nx = body.shape
    for t in np.linspace(0.0, 1.0, nstep):
        p = np.round((1.0 - t) * a + t * b).astype(int)
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dz * dz + dy * dy + dx * dx > radius * radius + 0.5:
                        continue
                    q = (int(p[0] + dz), int(p[1] + dy), int(p[2] + dx))
                    if not (0 <= q[0] < nz and 0 <= q[1] < ny and 0 <= q[2] < nx):
                        continue
                    if x_sep is not None:
                        if side == "left" and q[2] < x_sep:
                            continue
                        if side == "right" and q[2] > x_sep:
                            continue
                    if not body[q]:
                        continue
                    if force:
                        out[q] = True
    return out


def _paint_tip_ball(
    mask: np.ndarray,
    center: tuple[int, int, int],
    radius: int,
    body: np.ndarray,
    x_sep: int | None,
    side: str,
) -> np.ndarray:
    out = mask.copy()
    z0, y0, x0 = center
    nz, ny, nx = body.shape
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 2):  # slightly more posterior fill
            for dx in range(-radius, radius + 1):
                if dz * dz + max(dy, 0) * max(dy, 0) + dx * dx > (radius + 0.8) ** 2:
                    continue
                q = (z0 + dz, y0 + dy, x0 + dx)
                if not (0 <= q[0] < nz and 0 <= q[1] < ny and 0 <= q[2] < nx):
                    continue
                if x_sep is not None:
                    if side == "left" and q[2] < x_sep:
                        continue
                    if side == "right" and q[2] > x_sep:
                        continue
                if body[q]:
                    out[q] = True
    return out


def find_skin_tip_nares(
    body: np.ndarray,
    cavity_l: np.ndarray,
    cavity_r: np.ndarray,
    y_anterior_is_low: bool = True,
) -> tuple[tuple[int, int, int], tuple[int, int, int], list[str]]:
    """
    Place L/R external naris seeds on the anterior body surface (nose tip band).

    Uses cavity x split so left/right tips align with each cavity.
    """
    notes: list[str] = []
    # Cavities define lateral centers
    lz, ly, lx = np.where(cavity_l)
    rz, ry, rx = np.where(cavity_r)
    if len(lx) == 0 or len(rx) == 0:
        raise RuntimeError("Need both cavity_left and cavity_right masks.")
    l_x = int(np.median(lx))
    r_x = int(np.median(rx))
    l_z = int(np.median(lz))
    r_z = int(np.median(rz))
    x_sep = int(round(0.5 * (l_x + r_x)))
    z_mid = int(round(0.5 * (l_z + r_z)))
    notes.append(f"Cavity anchors L_x={l_x} R_x={r_x} x_sep={x_sep} z_mid={z_mid}")

    nz, ny, nx = body.shape
    # Mid-face z band around nasal cavities
    z0, z1 = max(0, z_mid - 30), min(nz, z_mid + 30)

    def tip_for(side: str, x_target: int, z_target: int) -> tuple[int, int, int]:
        # Search anterior surface near (z_target, x_target)
        best = None
        best_score = 1e18
        for z in range(max(z0, z_target - 18), min(z1, z_target + 19)):
            for x in range(max(0, x_target - 14), min(nx, x_target + 15)):
                col = np.where(body[z, :, x])[0]
                if len(col) == 0:
                    continue
                y_front = int(col.min() if y_anterior_is_low else col.max())
                # Must be on correct side of septum
                if side == "left" and x < x_sep:
                    continue
                if side == "right" and x > x_sep:
                    continue
                # Score: anterior + near target x/z
                ant = y_front if y_anterior_is_low else (ny - 1 - y_front)
                score = float(ant) + 0.35 * abs(x - x_target) + 0.25 * abs(z - z_target)
                if score < best_score:
                    best_score = score
                    best = (z, y_front, x)
        if best is None:
            # fallback: pure anterior column at target
            col = np.where(body[z_target, :, x_target])[0]
            y_front = int(col.min()) if len(col) else (5 if y_anterior_is_low else ny - 6)
            best = (z_target, y_front, x_target)
        return best

    left = tip_for("left", l_x, l_z)
    right = tip_for("right", r_x, r_z)
    notes.append(f"Skin tip L naris zyx={left}")
    notes.append(f"Skin tip R naris zyx={right}")
    return left, right, notes


def cavity_anterior_target(
    cavity: np.ndarray, side: str, x_sep: int
) -> tuple[int, int, int]:
    zz, yy, xx = np.where(cavity)
    if side == "left":
        keep = xx >= x_sep
    else:
        keep = xx <= x_sep
    if keep.any():
        zz, yy, xx = zz[keep], yy[keep], xx[keep]
    # most anterior quartile centroid
    thr = float(np.percentile(yy, 15))
    sel = yy <= thr
    if sel.sum() < 5:
        sel = np.ones(len(yy), dtype=bool)
    return (
        int(round(float(np.median(zz[sel])))),
        int(round(float(np.median(yy[sel])))),
        int(round(float(np.median(xx[sel])))),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="VisibleHuman_Head")
    ap.add_argument("--radius", type=int, default=4, help="Vestibule tube radius (voxels)")
    ap.add_argument("--tip-ball", type=int, default=5, help="Opening ball radius at tip")
    args = ap.parse_args()

    case_id = args.case
    out = REPO_ROOT / "outputs" / case_id
    notes: list[str] = []

    npz_path = out / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        print(f"Missing {npz_path}")
        return 1

    data = np.load(npz_path)
    airway = data["airway"].astype(bool)
    speed = data["speed"].astype(np.float32)
    ux = data["ux"].astype(np.float32)
    uy = data["uy"].astype(np.float32)
    uz = data["uz"].astype(np.float32)
    pressure = data["pressure"].astype(np.float32)
    spacing = tuple(float(v) for v in data["spacing_xyz_mm"])
    origin = tuple(float(v) for v in data["origin_xyz_mm"])
    inlet_mask = data["inlet_mask"].astype(bool) if "inlet_mask" in data else np.zeros_like(airway)
    outlet_mask = data["outlet_mask"].astype(bool) if "outlet_mask" in data else np.zeros_like(airway)

    body = sitk.GetArrayFromImage(
        sitk.ReadImage(str(out / f"{case_id}_head_mask.nrrd"))
    ).astype(bool)
    cl = sitk.GetArrayFromImage(
        sitk.ReadImage(str(out / f"{case_id}_cavity_left.nrrd"))
    ).astype(bool)
    cr = sitk.GetArrayFromImage(
        sitk.ReadImage(str(out / f"{case_id}_cavity_right.nrrd"))
    ).astype(bool)

    # Optional HU for soft partial-volume fill (not required for force paint)
    hu = None
    stats_path = out / f"{case_id}_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        image_path = Path(stats.get("image_path") or "")
        if not image_path.is_file():
            image_path = REPO_ROOT / "data" / "VisibleHuman_Head" / "VHFCT1mm_Head.nrrd"
        if image_path.is_file():
            full = sitk.ReadImage(str(image_path))
            hu_full = sitk.GetArrayFromImage(full).astype(np.float32)
            crop = stats.get("crop_origin_zyx") or [0, 0, 0]
            cz, cy, cx = [int(v) for v in crop]
            hu = hu_full[
                cz : cz + airway.shape[0],
                cy : cy + airway.shape[1],
                cx : cx + airway.shape[2],
            ]
            notes.append(f"HU crop from {image_path.name} crop={crop}")

    left_tip, right_tip, n1 = find_skin_tip_nares(body, cl, cr)
    notes.extend(n1)
    x_sep = int(round(0.5 * (left_tip[2] + right_tip[2])))

    l_tgt = cavity_anterior_target(cl, "left", x_sep)
    r_tgt = cavity_anterior_target(cr, "right", x_sep)
    notes.append(f"Cavity targets L={l_tgt} R={r_tgt}")
    notes.append(
        f"Tip gap L dy={l_tgt[1]-left_tip[1]} vx, R dy={r_tgt[1]-right_tip[1]} vx"
    )

    # Also absorb relaxed HU air in tip band into cavities
    if hu is not None:
        y_cut = max(left_tip[1], right_tip[1]) + 28
        tip_band = body.copy()
        tip_band[:, y_cut:, :] = False
        tip_band[:, : max(0, min(left_tip[1], right_tip[1]) - 2), :] = False
        soft_air = tip_band & (hu <= -120.0)  # partial-volume air
        cl = cl | (soft_air & (np.arange(body.shape[2])[None, None, :] >= x_sep))
        cr = cr | (soft_air & (np.arange(body.shape[2])[None, None, :] < x_sep))
        notes.append(f"Absorbed soft-air tip voxels: {int(soft_air.sum())}")

    # Force vestibule tubes tip → cavity (CT often occludes this region)
    before_l, before_r = int(cl.sum()), int(cr.sum())
    for tip, tgt, side in (
        (left_tip, l_tgt, "left"),
        (right_tip, r_tgt, "right"),
    ):
        tube = np.zeros_like(body)
        tube = _paint_tube(
            tube, tip, tgt, radius=args.radius, body=body, x_sep=x_sep, side=side, force=True
        )
        tube = _paint_tip_ball(
            tube, tip, radius=args.tip_ball, body=body, x_sep=x_sep, side=side
        )
        # mild close
        tube = morphology.binary_closing(tube, footprint=morphology.ball(1))
        if side == "left":
            cl = cl | tube
        else:
            cr = cr | tube
        notes.append(
            f"Painted {side} vestibule tip{tip}→{tgt} +{int(tube.sum())} vx"
        )

    # Keep septum gap: clear midplane strip
    cl[:, :, x_sep] = False
    cr[:, :, x_sep] = False
    # no cross-talk
    cl[:, :, :x_sep] = False
    cr[:, :, x_sep + 1 :] = False

    notes.append(
        f"Cavity L {before_l}→{int(cl.sum())}  R {before_r}→{int(cr.sum())}"
    )

    passage = cl | cr
    # Prefer connecting to existing all_interior_air / airway for caudal path
    air_all_path = out / f"{case_id}_all_interior_air.nrrd"
    if air_all_path.is_file():
        air_all = sitk.GetArrayFromImage(sitk.ReadImage(str(air_all_path))).astype(bool)
        air_all = air_all | passage
    else:
        air_all = passage | airway

    # Opening shell at tip: anterior surface of vestibules
    opening = np.zeros_like(passage)
    for side_m in (cl, cr):
        zz, yy, xx = np.where(side_m)
        if len(yy) == 0:
            continue
        thr = float(np.percentile(yy, 12))
        opening |= side_m & (np.arange(body.shape[1])[None, :, None] <= thr)

    # Naris centers = tip seeds (on air now)
    left_mm = _world(origin, spacing, *left_tip)
    right_mm = _world(origin, spacing, *right_tip)
    # snap slightly into air if tip voxel cleared
    for mm_name, zyx, side_m in (
        ("left", left_tip, cl),
        ("right", right_tip, cr),
    ):
        if not side_m[zyx]:
            pts = np.column_stack(np.where(side_m))
            if len(pts):
                d = np.linalg.norm(pts.astype(float) - np.array(zyx), axis=1)
                zyx2 = tuple(int(v) for v in pts[int(np.argmin(d))])
                if mm_name == "left":
                    left_tip = zyx2
                    left_mm = _world(origin, spacing, *left_tip)
                else:
                    right_tip = zyx2
                    right_mm = _world(origin, spacing, *right_tip)

    notes.append(f"Tip naris L mm={left_mm} R mm={right_mm}")

    # Update flow field: new air voxels get gentle inward velocity toward cavity
    new_air = passage & ~airway
    airway2 = airway | passage
    if new_air.any():
        # direction: +y (posterior, into head) for anterior-is-low
        uy = uy.copy()
        ux = ux.copy()
        uz = uz.copy()
        speed = speed.copy()
        # characteristic speed from existing air
        base = float(np.percentile(speed[airway & (speed > 1e-6)], 50)) if airway.any() else 0.3
        base = max(base, 0.15)
        uy[new_air] = base * 0.85  # flow inward (posterior)
        ux[new_air] = 0.0
        uz[new_air] = 0.0
        speed[new_air] = base * 0.85
        # blend near interface for smoother streamlines
        dil = morphology.binary_dilation(new_air, footprint=morphology.ball(2)) & airway2
        ring = dil & ~new_air
        if ring.any():
            uy[ring] = 0.5 * uy[ring] + 0.5 * base * 0.5
            speed[ring] = np.sqrt(ux[ring] ** 2 + uy[ring] ** 2 + uz[ring] ** 2)
        notes.append(
            f"Extended flow airway +{int(new_air.sum())} tip voxels (u≈{base*0.85:.3f} m/s inward)."
        )

    # Inlet mask balls at tips
    inlet2 = np.zeros_like(airway2)
    for tip in (left_tip, right_tip):
        for dz in range(-3, 4):
            for dy in range(-2, 4):
                for dx in range(-3, 4):
                    q = (tip[0] + dz, tip[1] + dy, tip[2] + dx)
                    if (
                        0 <= q[0] < airway2.shape[0]
                        and 0 <= q[1] < airway2.shape[1]
                        and 0 <= q[2] < airway2.shape[2]
                        and airway2[q]
                    ):
                        inlet2[q] = True

    # Write masks
    def _wmask(arr, name):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        sitk.WriteImage(img, str(out / f"{case_id}_{name}.nrrd"))

    _wmask(cl, "cavity_left")
    _wmask(cr, "cavity_right")
    _wmask(passage, "passage_lumen")
    _wmask(air_all, "all_interior_air")
    _wmask(opening, "naris_opening_ct")
    _wmask(airway2, "airway_mask")

    # STLs
    nL = _export_stl(cl, out / f"{case_id}_cavity_left.stl", spacing, origin)
    nR = _export_stl(cr, out / f"{case_id}_cavity_right.stl", spacing, origin)
    nP = _export_stl(passage, out / f"{case_id}_passage_surface.stl", spacing, origin)
    nA = _export_stl(airway2, out / f"{case_id}_airway.stl", spacing, origin, faces=18000)
    notes.append(f"STL faces L={nL} R={nR} passage={nP} airway={nA}")

    # flow npz
    np.savez_compressed(
        npz_path,
        airway=airway2.astype(np.uint8),
        speed=speed.astype(np.float32),
        ux=ux.astype(np.float32),
        uy=uy.astype(np.float32),
        uz=uz.astype(np.float32),
        pressure=pressure.astype(np.float32),
        spacing_xyz_mm=np.array(spacing, dtype=np.float64),
        origin_xyz_mm=np.array(origin, dtype=np.float64),
        inlet_mask=inlet2.astype(np.uint8),
        outlet_mask=outlet_mask.astype(np.uint8),
    )
    notes.append(f"Updated {npz_path.name}")

    # nares + BC
    nares = {
        "method": "skin_tip_vestibule",
        "y_anterior_is_low": True,
        "naris_points": [
            {
                "name": "left_nostril",
                "center_mm": left_mm,
                "skin_voxel_zyx": list(left_tip),
                "method": "skin_tip_vestibule",
                "notes": "External naris at skin nose tip; vestibule forced open to cavity.",
            },
            {
                "name": "right_nostril",
                "center_mm": right_mm,
                "skin_voxel_zyx": list(right_tip),
                "method": "skin_tip_vestibule",
                "notes": "External naris at skin nose tip; vestibule forced open to cavity.",
            },
        ],
        "notes": notes,
    }
    (out / f"{case_id}_nares.json").write_text(json.dumps(nares, indent=2), encoding="utf-8")

    bc_path = out / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports", []):
            if port.get("name") == "left_nostril":
                port["center_mm"] = left_mm
                port["method"] = "skin_tip_vestibule"
                port["notes"] = "Skin tip naris (left); open air painted to cavity."
            elif port.get("name") == "right_nostril":
                port["center_mm"] = right_mm
                port["method"] = "skin_tip_vestibule"
                port["notes"] = "Skin tip naris (right); open air painted to cavity."
        bc_path.write_text(json.dumps(bc, indent=2), encoding="utf-8")

    ct_meta = out / f"{case_id}_ct_nasal_meta.json"
    if ct_meta.is_file():
        cm = json.loads(ct_meta.read_text(encoding="utf-8"))
        cm["left_naris_center_mm"] = left_mm
        cm["right_naris_center_mm"] = right_mm
        cm["left_naris_center_zyx"] = list(left_tip)
        cm["right_naris_center_zyx"] = list(right_tip)
        cm["left_voxels"] = int(cl.sum())
        cm["right_voxels"] = int(cr.sum())
        cm["passage_voxels"] = int(passage.sum())
        cm["naris_marker_method"] = "skin_tip_vestibule"
        cm.setdefault("notes", []).extend(notes[-8:])
        ct_meta.write_text(json.dumps(cm, indent=2), encoding="utf-8")

    meta_path = out / f"{case_id}_tip_extension_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "left_tip_mm": left_mm,
                "right_tip_mm": right_mm,
                "left_cavity_voxels": int(cl.sum()),
                "right_cavity_voxels": int(cr.sum()),
                "new_air_voxels": int(new_air.sum()) if new_air is not None else 0,
                "notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"OK tip extension case={case_id}")
    print(f"  L naris {left_mm}  R naris {right_mm}")
    print(f"  cavity L={int(cl.sum())} R={int(cr.sum())} new_air={int(new_air.sum())}")
    for n in notes:
        print(f"  note: {n}")
    print("Next: py -3.12 scripts/regenerate_pathlines.py --case", case_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
