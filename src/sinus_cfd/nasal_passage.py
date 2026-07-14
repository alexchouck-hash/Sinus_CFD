"""
Nasal passage domain for airflow: lumen, walls, open ports, centerline.

Given a connected airway mask and port locations (nares + trachea):

  1. Ensure a single fluid domain connecting nares → trachea
  2. Classify surface voxels as wall vs open boundary (inlets/outlets)
  3. Extract centerline path for path-aware flow / visualization
  4. Measure cross-sectional area along the passage
  5. Export wall mesh + passage JSON for CFD setup

Walls = mucosa / tissue interface (no-slip).
Open ports = nostrils (inflow) and trachea (outflow).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import morphology
from skimage.graph import route_through_array

from .pipeline import _mask_to_mesh


@dataclass
class PassageMetrics:
    case_id: str
    lumen_voxels: int
    lumen_volume_ml: float
    wall_voxels: int
    inlet_open_voxels: int
    outlet_open_voxels: int
    centerline_length_mm: float
    n_centerline_nodes: int
    min_cross_section_mm2: float
    max_cross_section_mm2: float
    mean_cross_section_mm2: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phys_from_zyx(
    z: float, y: float, x: float,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> list[float]:
    sx, sy, sz = spacing
    ox, oy, oz = origin
    return [float(ox + x * sx), float(oy + y * sy), float(oz + z * sz)]


def _zyx_from_mm(
    center_mm: list[float],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    sx, sy, sz = spacing
    ox, oy, oz = origin
    x = int(np.clip(round((center_mm[0] - ox) / sx), 0, shape[2] - 1))
    y = int(np.clip(round((center_mm[1] - oy) / sy), 0, shape[1] - 1))
    z = int(np.clip(round((center_mm[2] - oz) / sz), 0, shape[0] - 1))
    return z, y, x


def nearest_lumen_index(
    lumen: np.ndarray,
    zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Snap a point to the nearest true lumen voxel."""
    if (
        0 <= zyx[0] < lumen.shape[0]
        and 0 <= zyx[1] < lumen.shape[1]
        and 0 <= zyx[2] < lumen.shape[2]
        and lumen[zyx]
    ):
        return zyx
    pts = np.column_stack(np.where(lumen))
    if len(pts) == 0:
        raise ValueError("Empty lumen")
    d = np.linalg.norm(pts.astype(float) - np.array(zyx, dtype=float), axis=1)
    return tuple(int(v) for v in pts[int(np.argmin(d))])


def _line_zyx(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Voxel indices along a straight segment a→b (inclusive)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    dist = float(np.linalg.norm(b_arr - a_arr))
    n = max(int(np.ceil(dist)) + 1, 2)
    out: list[tuple[int, int, int]] = []
    for t in np.linspace(0.0, 1.0, n):
        p = np.round((1.0 - t) * a_arr + t * b_arr).astype(int)
        out.append((int(p[0]), int(p[1]), int(p[2])))
    # unique while preserving order
    seen = set()
    uniq: list[tuple[int, int, int]] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _paint_ball(
    mask: np.ndarray,
    center: tuple[int, int, int],
    radius_vox: int,
) -> None:
    """OR a ball of radius_vox into mask (in-place)."""
    z0, y0, x0 = center
    r = max(int(radius_vox), 1)
    zmin = max(0, z0 - r)
    zmax = min(mask.shape[0], z0 + r + 1)
    ymin = max(0, y0 - r)
    ymax = min(mask.shape[1], y0 + r + 1)
    xmin = max(0, x0 - r)
    xmax = min(mask.shape[2], x0 + r + 1)
    zz, yy, xx = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
    ball = (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2 <= r * r
    mask[zmin:zmax, ymin:ymax, xmin:xmax] |= ball


def extend_lumen_to_external_nares(
    lumen: np.ndarray,
    skin_naris_centers_mm: list[list[float]],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    tunnel_radius_mm: float = 3.5,
) -> tuple[np.ndarray, list[str], list[tuple[int, int, int]]]:
    """
    Add open-air corridors from external skin nares into the internal airway.

    The CT/airway mask often stops inside the nose (~3 cm deep of the face).
    Without these tunnels the fluid domain and centerline never reach the
    nostrils where air actually enters.

    Returns (extended_lumen, notes, naris_seed_zyx_list).
    """
    notes: list[str] = []
    if not skin_naris_centers_mm:
        return lumen.astype(bool), ["No skin naris centers — lumen not extended."], []

    extended = lumen.astype(bool).copy()
    sp = float(np.mean(spacing))
    r_vox = max(int(round(tunnel_radius_mm / sp)), 2)
    shape = lumen.shape
    naris_seeds: list[tuple[int, int, int]] = []
    gaps_mm: list[float] = []

    for c_mm in skin_naris_centers_mm:
        skin_zyx = _zyx_from_mm(c_mm, spacing, origin, shape)
        if not lumen.any():
            continue
        # Target: nearest *original* lumen voxel (internal nasal cavity)
        target = nearest_lumen_index(lumen, skin_zyx)
        a = np.array(skin_zyx, dtype=float)
        b = np.array(target, dtype=float)
        gap = float(np.linalg.norm((b - a) * np.array([spacing[2], spacing[1], spacing[0]])))
        # spacing is xyz; zyx indices → physical: dz*sz, dy*sy, dx*sx
        gap = float(
            np.linalg.norm(
                np.array(
                    [
                        (b[2] - a[2]) * spacing[0],
                        (b[1] - a[1]) * spacing[1],
                        (b[0] - a[0]) * spacing[2],
                    ]
                )
            )
        )
        gaps_mm.append(gap)
        for pt in _line_zyx(skin_zyx, target):
            if (
                0 <= pt[0] < shape[0]
                and 0 <= pt[1] < shape[1]
                and 0 <= pt[2] < shape[2]
            ):
                _paint_ball(extended, pt, r_vox)
        # Ensure a full opening at the skin surface
        _paint_ball(extended, skin_zyx, r_vox + 1)
        naris_seeds.append(skin_zyx)

    # Keep only components that touch the original lumen (drop stray paint)
    lab, nlab = ndi.label(extended)
    if nlab > 1:
        keep = np.zeros(nlab + 1, dtype=bool)
        touch = lab[lumen]
        for i in np.unique(touch):
            if i > 0:
                keep[i] = True
        # Always keep components containing naris seeds
        for s in naris_seeds:
            li = lab[s]
            if li > 0:
                keep[li] = True
        extended = keep[lab]

    # Fill small holes in tunnels
    extended = morphology.closing(extended, footprint=morphology.ball(1))
    extended = ndi.binary_fill_holes(extended) | extended

    # Re-keep largest component that includes a naris (or original lumen)
    lab2, n2 = ndi.label(extended)
    if n2 > 1:
        best_i, best_score = 1, -1
        for i in range(1, n2 + 1):
            comp = lab2 == i
            score = int(comp.sum())
            # bonus if contains naris
            if any(comp[s] for s in naris_seeds):
                score += 1_000_000
            if score > best_score:
                best_score = score
                best_i = i
        extended = lab2 == best_i

    notes.append(
        f"Extended lumen to {len(naris_seeds)} external skin nares "
        f"(tunnel radius ≈ {tunnel_radius_mm:.1f} mm; "
        f"skin→cavity gap was {np.mean(gaps_mm):.1f} mm mean)."
        if gaps_mm
        else f"Extended lumen to {len(naris_seeds)} external skin nares."
    )
    notes.append(
        "Open-air model now includes nostrils at the face; centerline starts at skin nares."
    )
    return extended.astype(bool), notes, naris_seeds


def open_ports_on_lumen(
    lumen: np.ndarray,
    inlet_centers_mm: list[list[float]],
    outlet_center_mm: list[float],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    open_radius_mm: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]], tuple[int, int, int]]:
    """
    Mark open-boundary masks near nares and trachea inside the lumen.

    Returns inlet_open, outlet_open, inlet_seed_zyx_list, outlet_seed_zyx.
    """
    shape = lumen.shape
    sp = float(np.mean(spacing))
    r = max(int(round(open_radius_mm / sp)), 2)

    inlet_open = np.zeros(shape, dtype=bool)
    inlet_seeds: list[tuple[int, int, int]] = []
    for c in inlet_centers_mm:
        zyx = nearest_lumen_index(lumen, _zyx_from_mm(c, spacing, origin, shape))
        inlet_seeds.append(zyx)
        seed = np.zeros(shape, dtype=bool)
        seed[zyx] = True
        seed = ndi.binary_dilation(seed, iterations=r) & lumen
        inlet_open |= seed

    out_zyx = nearest_lumen_index(
        lumen, _zyx_from_mm(outlet_center_mm, spacing, origin, shape)
    )
    outlet_open = np.zeros(shape, dtype=bool)
    outlet_open[out_zyx] = True
    outlet_open = ndi.binary_dilation(outlet_open, iterations=r) & lumen

    # Avoid overlapping inlet/outlet opens if domain is tiny
    outlet_open &= ~inlet_open
    return inlet_open, outlet_open, inlet_seeds, out_zyx


def wall_mask_from_lumen(
    lumen: np.ndarray,
    inlet_open: np.ndarray,
    outlet_open: np.ndarray,
) -> np.ndarray:
    """
    Wall = lumen surface voxels that are NOT open ports.

    Surface = lumen voxels with a non-lumen 6-neighbor (or domain boundary).
    """
    # Erode lumen: surface = lumen & ~eroded
    eroded = morphology.erosion(lumen, footprint=morphology.ball(1))
    surface = lumen & ~eroded
    # Also include voxels on the array boundary that are lumen
    surface[0] |= lumen[0]
    surface[-1] |= lumen[-1]
    surface[:, 0] |= lumen[:, 0]
    surface[:, -1] |= lumen[:, -1]
    surface[:, :, 0] |= lumen[:, :, 0]
    surface[:, :, -1] |= lumen[:, :, -1]

    wall = surface & ~inlet_open & ~outlet_open
    return wall.astype(bool)


def compute_centerline(
    lumen: np.ndarray,
    start_zyx: tuple[int, int, int],
    end_zyx: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """
    Centerline via geodesic through the lumen, biased to the medial axis
    (high distance-from-wall).

    Returns (N,3) points in mm and path length mm.
    """
    # Distance to wall (inside lumen)
    # EDT of inverted lumen gives distance to outside
    dist = ndi.distance_transform_edt(lumen)
    # Cost: prefer high dist (center) → cost = 1/(dist+eps)
    costs = np.where(lumen, 1.0 / (dist + 0.25), 1.0e6)

    start = nearest_lumen_index(lumen, start_zyx)
    end = nearest_lumen_index(lumen, end_zyx)
    try:
        indices, _ = route_through_array(
            costs, start=start, end=end, fully_connected=True, geometric=True
        )
    except Exception:
        # Fallback: straight-ish list of lumen voxels by z
        indices = [start, end]

    pts_mm = []
    for iz, iy, ix in indices:
        pts_mm.append(_phys_from_zyx(iz, iy, ix, spacing, origin))
    pts = np.array(pts_mm, dtype=float)
    if len(pts) < 2:
        return pts, 0.0
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, float(seglen.sum())


def cross_sections_along_centerline(
    lumen: np.ndarray,
    centerline_zyx: list[tuple[int, int, int]],
    spacing: tuple[float, float, float],
    sample_every: int = 3,
) -> list[dict[str, Any]]:
    """
    Approximate cross-sectional area at centerline nodes via local
    distance-to-wall (π r² with r = dist_to_wall).
    """
    dist = ndi.distance_transform_edt(lumen, sampling=spacing[::-1])  # z,y,x sampling
    # sampling for EDT: (sz, sy, sx) matching array axes
    sx, sy, sz = spacing
    dist = ndi.distance_transform_edt(lumen, sampling=(sz, sy, sx))

    out = []
    for i, (iz, iy, ix) in enumerate(centerline_zyx):
        if i % sample_every != 0 and i != len(centerline_zyx) - 1:
            continue
        r = float(dist[iz, iy, ix])
        area = float(np.pi * r * r)
        out.append(
            {
                "index": i,
                "zyx": [int(iz), int(iy), int(ix)],
                "radius_mm": r,
                "area_mm2": area,
            }
        )
    return out


def centerline_indices_from_mm_path(
    pts_mm: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    idxs = []
    for p in pts_mm:
        idxs.append(_zyx_from_mm(p.tolist(), spacing, origin, shape))
    return idxs


def analyze_nasal_passage(
    lumen: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    inlet_centers_mm: list[list[float]],
    outlet_center_mm: list[float],
    case_id: str = "case",
    open_radius_mm: float = 6.0,
    skin_naris_centers_mm: list[list[float]] | None = None,
    tunnel_radius_mm: float = 3.5,
) -> tuple[dict[str, np.ndarray], dict[str, Any], PassageMetrics]:
    """
    Full passage analysis.

    Extends the lumen to external skin nares when provided so the open-air
    model and centerline include the nostrils on the face.

    Returns:
      masks: lumen, wall, inlet_open, outlet_open, fluid_interior
      passage_json-ready dict
      metrics
    """
    notes: list[str] = []
    lumen = lumen.astype(bool)
    if not lumen.any():
        raise ValueError("Empty lumen mask")

    # Single component (internal cavity) first
    lab, n = ndi.label(lumen)
    if n > 1:
        notes.append(f"Lumen had {n} components; keeping largest only for passage.")
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        lumen = lab == int(np.argmax(counts))

    # Prefer explicit skin naris points; fall back to inlet BC centers
    skin_pts = skin_naris_centers_mm or list(inlet_centers_mm)
    lumen, ext_notes, skin_seeds = extend_lumen_to_external_nares(
        lumen,
        skin_pts,
        spacing,
        origin,
        tunnel_radius_mm=tunnel_radius_mm,
    )
    notes.extend(ext_notes)

    # Open ports: use skin nares for inlets so openings are at the face
    port_inlets = skin_pts if skin_pts else inlet_centers_mm
    inlet_open, outlet_open, inlet_seeds, out_seed = open_ports_on_lumen(
        lumen, port_inlets, outlet_center_mm, spacing, origin, open_radius_mm
    )
    # Prefer true skin seeds if extension created them
    if skin_seeds:
        inlet_seeds = skin_seeds
    wall = wall_mask_from_lumen(lumen, inlet_open, outlet_open)
    # Interior fluid = lumen (ports are still fluid, just open BC)
    fluid = lumen.copy()

    # Dual centerlines: each external naris → trachea (magenta paths in viewer)
    left_cl = right_cl = None
    skin_arr = np.array(skin_pts, dtype=float) if skin_pts else np.zeros((0, 3))

    def _naris_path(skin_mm: np.ndarray) -> np.ndarray:
        start = nearest_lumen_index(
            lumen, _zyx_from_mm(skin_mm.tolist(), spacing, origin, lumen.shape)
        )
        path, _ = compute_centerline(lumen, start, out_seed, spacing, origin)
        if len(path) == 0:
            return skin_mm.reshape(1, 3)
        if float(np.linalg.norm(path[0] - skin_mm)) > 1.0:
            path = np.vstack([skin_mm, path])
        return path

    if len(skin_arr) >= 2:
        # LPS: higher x = patient left
        order = np.argsort(skin_arr[:, 0])  # low x first = patient right
        right_mm = skin_arr[order[0]]
        left_mm = skin_arr[order[-1]]
        left_cl = _naris_path(left_mm)
        right_cl = _naris_path(right_mm)
        notes.append(
            "Dual centerlines: left naris→trachea and right naris→trachea."
        )
    elif len(skin_arr) == 1:
        left_cl = _naris_path(skin_arr[0])
        notes.append("Single naris centerline only (one skin point).")
    elif inlet_seeds:
        start = nearest_lumen_index(
            lumen,
            tuple(int(np.round(np.mean([s[i] for s in inlet_seeds]))) for i in range(3)),  # type: ignore
        )
        left_cl, _ = compute_centerline(lumen, start, out_seed, spacing, origin)

    # Combined reference path (mean of dual if both exist) for metrics
    if left_cl is not None and right_cl is not None and len(left_cl) and len(right_cl):
        # For metrics length use average of both paths
        length_mm = 0.5 * (
            float(np.linalg.norm(np.diff(left_cl, axis=0), axis=1).sum())
            + float(np.linalg.norm(np.diff(right_cl, axis=0), axis=1).sum())
        )
        cl_mm = left_cl  # metrics sample along left; both stored separately
    elif left_cl is not None and len(left_cl):
        cl_mm = left_cl
        length_mm = float(np.linalg.norm(np.diff(cl_mm, axis=0), axis=1).sum()) if len(cl_mm) > 1 else 0.0
    else:
        start = nearest_lumen_index(
            lumen, _zyx_from_mm(port_inlets[0], spacing, origin, lumen.shape)
        )
        cl_mm, length_mm = compute_centerline(lumen, start, out_seed, spacing, origin)

    cl_idx = centerline_indices_from_mm_path(cl_mm, spacing, origin, lumen.shape)
    sections = cross_sections_along_centerline(lumen, cl_idx, spacing)

    areas = [s["area_mm2"] for s in sections] or [0.0]
    sp_vol = float(np.prod(spacing))
    metrics = PassageMetrics(
        case_id=case_id,
        lumen_voxels=int(lumen.sum()),
        lumen_volume_ml=float(lumen.sum() * sp_vol / 1000.0),
        wall_voxels=int(wall.sum()),
        inlet_open_voxels=int(inlet_open.sum()),
        outlet_open_voxels=int(outlet_open.sum()),
        centerline_length_mm=length_mm,
        n_centerline_nodes=len(cl_mm),
        min_cross_section_mm2=float(np.min(areas)),
        max_cross_section_mm2=float(np.max(areas)),
        mean_cross_section_mm2=float(np.mean(areas)),
        notes=notes
        + [
            "Wall = lumen surface excluding open nares/trachea ports.",
            "Flow domain = open air from external nostrils through nasal passage to trachea.",
            "Dual magenta centerlines: each external naris back to caudal trachea.",
        ],
    )

    masks = {
        "lumen": lumen,
        "wall": wall,
        "inlet_open": inlet_open,
        "outlet_open": outlet_open,
        "fluid": fluid,
    }
    passage = {
        "case_id": case_id,
        "spacing_xyz_mm": list(spacing),
        "origin_xyz_mm": list(origin),
        "skin_naris_centers_mm": skin_pts,
        "includes_external_nares": True,
        "inlet_seeds_zyx": [list(s) for s in inlet_seeds],
        "outlet_seed_zyx": list(out_seed),
        "centerline_mm": cl_mm.tolist() if len(cl_mm) else [],
        "centerline_left_mm": left_cl.tolist() if left_cl is not None and len(left_cl) else [],
        "centerline_right_mm": right_cl.tolist() if right_cl is not None and len(right_cl) else [],
        "cross_sections": sections,
        "boundary_roles": {
            "wall": "no_slip_mucosa",
            "inlet_open": "volumetric_flow_nares_at_skin",
            "outlet_open": "pressure_outlet_trachea",
        },
        "metrics": metrics.to_dict(),
    }
    return masks, passage, metrics


def write_passage_outputs(
    case_id: str,
    output_dir: Path | str,
    masks: dict[str, np.ndarray],
    passage: dict[str, Any],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    direction: tuple[float, ...] | None = None,
    reference_image: sitk.Image | None = None,
) -> dict[str, Path]:
    """Write NRRD masks, wall STL, passage JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _write(mask: np.ndarray, name: str) -> Path:
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        if reference_image is not None:
            img.SetDirection(reference_image.GetDirection())
        elif direction is not None:
            img.SetDirection(direction)
        p = output_dir / name
        sitk.WriteImage(img, str(p))
        return p

    paths["lumen"] = _write(masks["lumen"], f"{case_id}_passage_lumen.nrrd")
    paths["wall"] = _write(masks["wall"], f"{case_id}_passage_wall.nrrd")
    paths["inlet_open"] = _write(masks["inlet_open"], f"{case_id}_passage_inlet_open.nrrd")
    paths["outlet_open"] = _write(masks["outlet_open"], f"{case_id}_passage_outlet_open.nrrd")

    # Wall surface mesh (boundary of lumen — includes open ends as surface;
    # for CFD, open ends are BCs not walls)
    try:
        wall_for_mesh = masks["lumen"]  # outer surface of fluid domain
        mesh = _mask_to_mesh(wall_for_mesh, spacing, origin)
        # Prefer decimated wall
        if len(mesh.faces) > 25000:
            try:
                mesh = mesh.simplify_quadric_decimation(25000)
            except Exception:
                pass
        p = output_dir / f"{case_id}_passage_surface.stl"
        mesh.export(p)
        paths["surface_stl"] = p
    except Exception as exc:
        passage.setdefault("metrics", {}).setdefault("notes", []).append(
            f"Surface mesh failed: {exc}"
        )

    # Explicit wall-only (no open ports) for visualization of mucosa
    try:
        if masks["wall"].any():
            # Dilate wall slightly to make a thin shell meshable
            shell = morphology.dilation(masks["wall"], footprint=morphology.ball(1))
            shell = shell & masks["lumen"]
            if shell.sum() > 100:
                wmesh = _mask_to_mesh(shell, spacing, origin)
                if len(wmesh.faces) > 20000:
                    try:
                        wmesh = wmesh.simplify_quadric_decimation(20000)
                    except Exception:
                        pass
                p = output_dir / f"{case_id}_passage_wall.stl"
                wmesh.export(p)
                paths["wall_stl"] = p
    except Exception:
        pass

    jp = output_dir / f"{case_id}_passage.json"
    with jp.open("w", encoding="utf-8") as f:
        json.dump(passage, f, indent=2)
    paths["passage_json"] = jp

    return paths
