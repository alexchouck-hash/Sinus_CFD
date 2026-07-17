"""
Geometry metrics along naris → trachea centerlines (no CFD required).

Primary outputs:
  - Cross-sectional area (CSA) profile along each side (L/R)
  - Minimal cross-sectional area (MCA) value, index, and physical location
  - Summary JSON for viewer / virtual-surgery compare
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .nasal_passage import compute_centerline


def _zyx_from_mm(
    pt_mm: list[float] | np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    sx, sy, sz = spacing
    ox, oy, oz = origin
    x, y, z = float(pt_mm[0]), float(pt_mm[1]), float(pt_mm[2])
    ix = int(np.clip(round((x - ox) / sx), 0, shape[2] - 1))
    iy = int(np.clip(round((y - oy) / sy), 0, shape[1] - 1))
    iz = int(np.clip(round((z - oz) / sz), 0, shape[0] - 1))
    return iz, iy, ix


def _nearest_lumen(
    lumen: np.ndarray, zyx: tuple[int, int, int]
) -> tuple[int, int, int]:
    if lumen[zyx]:
        return zyx
    dist = ndi.distance_transform_edt(~lumen)
    # climb to nearest True
    zz, yy, xx = np.where(lumen)
    if len(zz) == 0:
        return zyx
    d2 = (zz - zyx[0]) ** 2 + (yy - zyx[1]) ** 2 + (xx - zyx[2]) ** 2
    i = int(np.argmin(d2))
    return int(zz[i]), int(yy[i]), int(xx[i])


def _path_arc_length_mm(pts_mm: np.ndarray) -> np.ndarray:
    """Cumulative distance along polyline (N,)."""
    if len(pts_mm) == 0:
        return np.zeros(0, dtype=float)
    if len(pts_mm) == 1:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def enrich_cross_sections(
    sections: list[dict[str, Any]],
    centerline_mm: np.ndarray,
    sample_every: int = 3,
) -> list[dict[str, Any]]:
    """Add path_s_mm and xyz_mm to CSA samples."""
    s_all = _path_arc_length_mm(np.asarray(centerline_mm, dtype=float))
    out: list[dict[str, Any]] = []
    for sec in sections:
        idx = int(sec.get("index", 0))
        idx = max(0, min(idx, len(centerline_mm) - 1))
        pt = centerline_mm[idx]
        s_mm = float(s_all[idx]) if len(s_all) else 0.0
        row = dict(sec)
        row["path_s_mm"] = s_mm
        row["xyz_mm"] = [float(pt[0]), float(pt[1]), float(pt[2])]
        out.append(row)
    return out


def mca_from_sections(
    sections: list[dict[str, Any]],
    skip_tip_mm: float = 8.0,
    # Nasal valve / cavity only: ignore nasopharynx–trachea tail
    max_path_fraction: float = 0.55,
    max_path_mm: float | None = 85.0,
) -> dict[str, Any] | None:
    """Pick minimal CSA in the *nasal* portion of the path (not trachea)."""
    if not sections:
        return None
    s_vals = [float(s.get("path_s_mm", 0)) for s in sections]
    s_max = max(s_vals) if s_vals else 0.0
    s_hi = s_max * max_path_fraction
    if max_path_mm is not None:
        s_hi = min(s_hi, float(max_path_mm))
    candidates = [
        s
        for s in sections
        if skip_tip_mm <= float(s.get("path_s_mm", 0)) <= s_hi
    ]
    if not candidates:
        # mid-path band fallback
        candidates = [
            s
            for s in sections
            if 0.15 * s_max <= float(s.get("path_s_mm", 0)) <= 0.55 * s_max
        ]
    if not candidates:
        candidates = list(sections)
    best = min(candidates, key=lambda s: float(s.get("area_mm2", 1e18)))
    return {
        "mca_mm2": float(best.get("area_mm2", 0.0)),
        "mca_radius_mm": float(best.get("radius_mm", 0.0)),
        "mca_path_s_mm": float(best.get("path_s_mm", 0.0)),
        "mca_xyz_mm": list(best.get("xyz_mm") or []),
        "mca_zyx": list(best.get("zyx") or []),
        "n_samples": len(sections),
        "n_candidates": len(candidates),
        "search_path_mm": [skip_tip_mm, s_hi],
        "method": "plane_csa_nasal_segment",
    }


def plane_csa_along_centerline(
    lumen: np.ndarray,
    centerline_mm: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    sample_every: int = 2,
    half_width_mm: float = 18.0,
) -> list[dict[str, Any]]:
    """
    Cross-section area by counting lumen voxels near a plane perpendicular
    to the local centerline tangent (more stable than π r² alone).
    Also records EDT radius for reference.
    """
    sx, sy, sz = spacing
    ox, oy, oz = origin
    cl = np.asarray(centerline_mm, dtype=float)
    dist = ndi.distance_transform_edt(lumen, sampling=(sz, sy, sx))
    voxel_area = float(min(sx * sy, sx * sz, sy * sz))  # conservative face

    # Precompute physical coords of lumen voxels (subsample if huge)
    zz, yy, xx = np.where(lumen)
    if len(zz) > 250_000:
        step = max(1, len(zz) // 200_000)
        zz, yy, xx = zz[::step], yy[::step], xx[::step]
        voxel_area *= step  # rough compensation
    wx = ox + xx.astype(float) * sx
    wy = oy + yy.astype(float) * sy
    wz = oz + zz.astype(float) * sz
    pts = np.column_stack([wx, wy, wz])

    out: list[dict[str, Any]] = []
    n = len(cl)
    for i in range(n):
        if i % sample_every != 0 and i != n - 1:
            continue
        p = cl[i]
        # tangent
        if i == 0:
            t = cl[min(1, n - 1)] - cl[0]
        elif i >= n - 1:
            t = cl[-1] - cl[-2]
        else:
            t = cl[i + 1] - cl[i - 1]
        tn = float(np.linalg.norm(t))
        if tn < 1e-9:
            t = np.array([0.0, 1.0, 0.0])
        else:
            t = t / tn
        # slab around plane: | (x-p)·t | < half voxel
        slab = 0.6 * float(min(sx, sy, sz))
        d = pts - p.reshape(1, 3)
        along = d @ t
        radial = np.linalg.norm(d - along[:, None] * t.reshape(1, 3), axis=1)
        in_plane = (np.abs(along) <= slab) & (radial <= half_width_mm)
        count = int(in_plane.sum())
        # area ≈ count * mean voxel face perpendicular to t
        # use isotropic spacing product / mean spacing as area scale
        area = float(count) * float((sx * sy * sz) ** (2.0 / 3.0))
        iz, iy, ix = _zyx_from_mm(p, spacing, origin, lumen.shape)
        iz, iy, ix = _nearest_lumen(lumen, (iz, iy, ix))
        r = float(dist[iz, iy, ix])
        area_edt = float(np.pi * r * r)
        # blend: prefer plane count when enough voxels, else EDT
        if count >= 3:
            area_use = max(area, area_edt * 0.5)
        else:
            area_use = area_edt
        out.append(
            {
                "index": i,
                "zyx": [int(iz), int(iy), int(ix)],
                "radius_mm": r,
                "area_mm2": area_use,
                "area_edt_mm2": area_edt,
                "plane_voxel_count": count,
            }
        )
    return out


def side_geometry(
    name: str,
    lumen: np.ndarray,
    centerline_mm: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    sample_every: int = 2,
) -> dict[str, Any]:
    """CSA profile + MCA for one naris→trachea centerline."""
    cl = np.asarray(centerline_mm, dtype=float)
    if cl.ndim != 2 or len(cl) < 2:
        return {
            "name": name,
            "centerline_length_mm": 0.0,
            "n_nodes": int(len(cl)),
            "cross_sections": [],
            "mca": None,
            "notes": ["Centerline missing or too short."],
        }
    sections = plane_csa_along_centerline(
        lumen, cl, spacing, origin, sample_every=sample_every
    )
    sections = enrich_cross_sections(sections, cl, sample_every=sample_every)
    mca = mca_from_sections(sections)
    length = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
    areas = [float(s["area_mm2"]) for s in sections] if sections else [0.0]
    return {
        "name": name,
        "centerline_length_mm": length,
        "n_nodes": len(cl),
        "cross_sections": sections,
        "area_min_mm2": float(np.min(areas)),
        "area_mean_mm2": float(np.mean(areas)),
        "area_max_mm2": float(np.max(areas)),
        "mca": mca,
        "centerline_mm": cl.tolist(),
        "notes": [],
    }


def load_lumen_for_geometry(
    case_dir: Path,
    case_id: str,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float], str]:
    """Prefer passage lumen, then airway mask. Optionally OR L/R cavities."""
    candidates = [
        case_dir / f"{case_id}_passage_lumen.nrrd",
        case_dir / f"{case_id}_airway_mask.nrrd",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No lumen mask for {case_id} under {case_dir}"
        )
    img = sitk.ReadImage(str(path))
    lumen = sitk.GetArrayFromImage(img).astype(bool)
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())

    # Optionally enrich with L/R cavities (both sides of septum). Skip when
    # caller uses a dedicated edited passage lumen only (filename match).
    if "passage_lumen" not in path.name:
        for side in ("left", "right"):
            cp = case_dir / f"{case_id}_cavity_{side}.nrrd"
            if cp.is_file():
                cimg = sitk.ReadImage(str(cp))
                cav = sitk.GetArrayFromImage(cimg).astype(bool)
                if cav.shape == lumen.shape:
                    lumen = lumen | cav
    else:
        # Still OR cavities for dual-side MCA when baseline passage is thin
        # on one side — but only if cavities exist and are smaller enrichment.
        for side in ("left", "right"):
            cp = case_dir / f"{case_id}_cavity_{side}.nrrd"
            if cp.is_file():
                cimg = sitk.ReadImage(str(cp))
                cav = sitk.GetArrayFromImage(cimg).astype(bool)
                if cav.shape == lumen.shape:
                    lumen = lumen | cav
    return lumen, spacing, origin, path.name


def resolve_centerlines(
    case_dir: Path,
    case_id: str,
    lumen: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Load dual centerlines from passage/nares or recompute."""
    notes: list[str] = []
    left = right = None
    passage_path = case_dir / f"{case_id}_passage.json"
    if passage_path.is_file():
        pj = json.loads(passage_path.read_text(encoding="utf-8"))
        if pj.get("centerline_left_mm"):
            left = np.asarray(pj["centerline_left_mm"], dtype=float)
            notes.append("Left centerline from passage.json")
        if pj.get("centerline_right_mm"):
            right = np.asarray(pj["centerline_right_mm"], dtype=float)
            notes.append("Right centerline from passage.json")

    # Nares + trachea for recompute if needed
    naris_pts: list[list[float]] = []
    nares_path = case_dir / f"{case_id}_nares.json"
    if nares_path.is_file():
        nj = json.loads(nares_path.read_text(encoding="utf-8"))
        for npnt in nj.get("naris_points") or []:
            if npnt.get("center_mm"):
                naris_pts.append([float(v) for v in npnt["center_mm"]])

    outlet_mm: list[float] | None = None
    bc_path = case_dir / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports") or []:
            if port.get("role") == "outlet" and port.get("center_mm"):
                outlet_mm = [float(v) for v in port["center_mm"]]
            if port.get("role") == "inlet" and port.get("center_mm") and len(naris_pts) < 2:
                naris_pts.append([float(v) for v in port["center_mm"]])

    if outlet_mm is None and passage_path.is_file():
        pj = json.loads(passage_path.read_text(encoding="utf-8"))
        cl = pj.get("centerline_mm") or []
        if len(cl) >= 2:
            outlet_mm = [float(v) for v in cl[-1]]

    def _recompute(skin_mm: np.ndarray) -> np.ndarray:
        start = _nearest_lumen(
            lumen, _zyx_from_mm(skin_mm, spacing, origin, lumen.shape)
        )
        end = _nearest_lumen(
            lumen, _zyx_from_mm(outlet_mm, spacing, origin, lumen.shape)  # type: ignore[arg-type]
        )
        path, _ = compute_centerline(lumen, start, end, spacing, origin)
        if len(path) == 0:
            return skin_mm.reshape(1, 3)
        if float(np.linalg.norm(path[0] - skin_mm)) > 1.0:
            path = np.vstack([skin_mm, path])
        return path

    need = (left is None or right is None or len(left) < 2 or len(right) < 2)
    if need and len(naris_pts) >= 2 and outlet_mm is not None:
        arr = np.asarray(naris_pts, dtype=float)
        order = np.argsort(arr[:, 0])  # low x = patient right
        right = _recompute(arr[order[0]])
        left = _recompute(arr[order[-1]])
        notes.append("Recomputed dual centerlines naris→trachea on lumen")
    elif need and len(naris_pts) == 1 and outlet_mm is not None:
        left = _recompute(np.asarray(naris_pts[0], dtype=float))
        notes.append("Single naris centerline only")

    return left, right, notes


def compute_geometry_metrics(
    case_dir: Path | str,
    case_id: str,
    sample_every: int = 2,
) -> dict[str, Any]:
    """Full geometry report for a case directory."""
    case_dir = Path(case_dir)
    lumen, spacing, origin, lumen_src = load_lumen_for_geometry(case_dir, case_id)
    left_cl, right_cl, cl_notes = resolve_centerlines(
        case_dir, case_id, lumen, spacing, origin
    )

    sides: list[dict[str, Any]] = []
    if left_cl is not None and len(left_cl) >= 2:
        sides.append(
            side_geometry("left", lumen, left_cl, spacing, origin, sample_every)
        )
    if right_cl is not None and len(right_cl) >= 2:
        sides.append(
            side_geometry("right", lumen, right_cl, spacing, origin, sample_every)
        )

    mcas = [s["mca"] for s in sides if s.get("mca")]
    global_mca = None
    if mcas:
        global_mca = min(mcas, key=lambda m: float(m["mca_mm2"]))
        # annotate which side
        for s in sides:
            if s.get("mca") is global_mca:
                global_mca = dict(global_mca)
                global_mca["side"] = s["name"]
                break

    report: dict[str, Any] = {
        "case_id": case_id,
        "kind": "geometry_metrics",
        "lumen_source": lumen_src,
        "spacing_xyz_mm": list(spacing),
        "origin_xyz_mm": list(origin),
        "lumen_voxels": int(lumen.sum()),
        "lumen_volume_ml": float(lumen.sum() * float(np.prod(spacing)) / 1000.0),
        "sides": [
            {k: v for k, v in s.items() if k != "centerline_mm"} for s in sides
        ],
        # keep short polylines for viewer markers (MCA only needs mca xyz)
        "centerlines": {
            s["name"]: s.get("centerline_mm") for s in sides if s.get("centerline_mm")
        },
        "global_mca": global_mca,
        "mca_markers": [
            {
                "side": s["name"],
                "xyz_mm": (s.get("mca") or {}).get("mca_xyz_mm"),
                "mca_mm2": (s.get("mca") or {}).get("mca_mm2"),
                "path_s_mm": (s.get("mca") or {}).get("mca_path_s_mm"),
            }
            for s in sides
            if s.get("mca") and (s["mca"] or {}).get("mca_xyz_mm")
        ],
        "notes": cl_notes
        + [
            "CSA from lumen voxels in a plane ⊥ centerline (fallback π r² EDT).",
            "MCA ignores the first ~8 mm from the naris (vestibule).",
            "Research metric — not a clinical measurement device.",
        ],
    }
    return report


def write_geometry_metrics(
    case_dir: Path | str,
    case_id: str,
    report: dict[str, Any] | None = None,
    sample_every: int = 2,
) -> Path:
    case_dir = Path(case_dir)
    if report is None:
        report = compute_geometry_metrics(case_dir, case_id, sample_every=sample_every)
    out = case_dir / f"{case_id}_geometry_metrics.json"
    # Drop bulky centerlines from on-disk report (still have CSA + MCA)
    slim = dict(report)
    slim.pop("centerlines", None)
    for side in slim.get("sides") or []:
        # keep cross_sections (needed for CSA plot); drop if huge later
        pass
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return out
