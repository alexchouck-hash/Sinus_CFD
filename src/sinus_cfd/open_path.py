"""
Most-open dual centerlines (nares → trachea) and open-space inference.

Algorithm (demo-quality, classical):
  1. Domain = air / passage mask (binary).
  2. "Most open path" = geodesic minimizing cost = 1 / (r + ε)^p
     where r = distance-to-wall (EDT inside air). Paths prefer the lumen center.
  3. Dual paths from left and right naris seeds to a shared trachea seed.
  4. Soft L/R symmetry: resample by arc length and blend each path with the
     midplane reflection of the other (approximately symmetric trajectories).
  5. Open space = air within a local radius of either path (radius from EDT
     along the path, scaled), so the domain is "inferred" from the centerlines.

References / related ideas:
  - Vascular centerlines: cost ∝ 1/radius (or Frangi + shortest path)
  - skimage.graph.route_through_array for discrete geodesics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.graph import route_through_array


@dataclass
class OpenPathResult:
    case_id: str
    centerline_left_mm: list[list[float]]
    centerline_right_mm: list[list[float]]
    centerline_mid_mm: list[list[float]]
    open_space: np.ndarray
    length_left_mm: float
    length_right_mm: float
    x_midplane_mm: float
    method: str = "most_open_dual_symmetric"
    notes: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("open_space", None)
        d["open_space_voxels"] = int(self.open_space.sum())
        return d


def _mm_to_zyx(
    mm: list[float] | np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    x = int(np.clip(round((float(mm[0]) - ox) / sx), 0, shape[2] - 1))
    y = int(np.clip(round((float(mm[1]) - oy) / sy), 0, shape[1] - 1))
    z = int(np.clip(round((float(mm[2]) - oz) / sz), 0, shape[0] - 1))
    return z, y, x


def _zyx_to_mm(
    zyx: tuple[int, int, int] | np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> np.ndarray:
    z, y, x = (int(zyx[0]), int(zyx[1]), int(zyx[2]))
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    return np.array([ox + x * sx, oy + y * sy, oz + z * sz], dtype=float)


def nearest_air_index(
    air: np.ndarray,
    zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    if (
        0 <= zyx[0] < air.shape[0]
        and 0 <= zyx[1] < air.shape[1]
        and 0 <= zyx[2] < air.shape[2]
        and air[zyx]
    ):
        return zyx
    pts = np.column_stack(np.where(air))
    if len(pts) == 0:
        raise ValueError("Empty air mask")
    d = np.linalg.norm(pts.astype(float) - np.array(zyx, dtype=float), axis=1)
    return tuple(int(v) for v in pts[int(np.argmin(d))])


def most_open_cost(
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    power: float = 2.0,
    eps: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cost volume for most-open geodesic: low cost in wide lumen (high EDT).

    sampling for EDT matches array axes (z,y,x) with spacing (sz, sy, sx).
    """
    sx, sy, sz = spacing_xyz
    # radius to wall in mm
    radius = ndi.distance_transform_edt(air, sampling=(sz, sy, sx))
    cost = np.full(air.shape, 1.0e6, dtype=np.float64)
    inside = air & (radius > 0)
    # Prefer centerline: cost falls as radius grows
    cost[inside] = 1.0 / np.power(radius[inside] + eps, power)
    # Thin air still traversable but expensive
    thin = air & ~inside
    cost[thin] = 1.0 / (eps**power)
    return cost.astype(np.float64), radius.astype(np.float32)


def most_open_path_zyx(
    air: np.ndarray,
    start_zyx: tuple[int, int, int],
    end_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    power: float = 2.0,
) -> list[tuple[int, int, int]]:
    """Discrete most-open path indices (z,y,x)."""
    start = nearest_air_index(air, start_zyx)
    end = nearest_air_index(air, end_zyx)
    cost, _ = most_open_cost(air, spacing_xyz, power=power)
    try:
        indices, _ = route_through_array(
            cost, start=start, end=end, fully_connected=True, geometric=True
        )
    except Exception:
        indices = [start, end]
    return [(int(a), int(b), int(c)) for a, b, c in indices]


def path_zyx_to_mm(
    indices: list[tuple[int, int, int]],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> np.ndarray:
    if not indices:
        return np.zeros((0, 3), dtype=float)
    return np.array(
        [_zyx_to_mm(p, spacing_xyz, origin_xyz) for p in indices], dtype=float
    )


def path_length_mm(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def resample_polyline(pts: np.ndarray, n: int = 80) -> np.ndarray:
    """Arc-length resample to n points."""
    if pts is None or len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    if u[-1] < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    u /= u[-1]
    t = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(t, u, pts[:, k])
    return out


def soft_symmetric_pair(
    path_l: np.ndarray,
    path_r: np.ndarray,
    x_mid_mm: float,
    blend: float = 0.35,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft L/R symmetry about midplane x = x_mid_mm.

    Each path is pulled toward the mirror of the other so trajectories are
    approximately symmetric without forcing perfect mirror anatomy.
    blend=0 → independent; blend=1 → pure mirrors of each other.
    """
    if path_l is None or path_r is None or len(path_l) < 2 or len(path_r) < 2:
        return path_l, path_r
    pl = resample_polyline(path_l, n)
    pr = resample_polyline(path_r, n)
    # Mirror across midplane: x' = 2*x_mid - x
    pl_m = pl.copy()
    pr_m = pr.copy()
    pl_m[:, 0] = 2.0 * x_mid_mm - pl[:, 0]
    pr_m[:, 0] = 2.0 * x_mid_mm - pr[:, 0]
    # Left should match mirror of right, and vice versa
    pl_new = (1.0 - blend) * pl + blend * pr_m
    pr_new = (1.0 - blend) * pr + blend * pl_m
    return pl_new, pr_new


def project_path_to_air(
    path_mm: np.ndarray,
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Snap each sample to nearest air voxel (after symmetry blend)."""
    if path_mm is None or len(path_mm) == 0:
        return path_mm
    out = []
    for p in path_mm:
        zyx = nearest_air_index(air, _mm_to_zyx(p, spacing_xyz, origin_xyz, air.shape))
        out.append(_zyx_to_mm(zyx, spacing_xyz, origin_xyz))
    return np.array(out, dtype=float)


def infer_open_space_from_paths(
    air: np.ndarray,
    paths_mm: list[np.ndarray],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    radius_scale: float = 1.15,
    min_radius_mm: float = 1.5,
    max_radius_mm: float = 12.0,
) -> np.ndarray:
    """
    Open space = air voxels within a tube around the centerlines.

    Local tube radius = clamp(scale * EDT(path point), min, max).
    """
    sx, sy, sz = spacing_xyz
    radius_edt = ndi.distance_transform_edt(air, sampling=(sz, sy, sx))
    open_sp = np.zeros_like(air, dtype=bool)

    for path in paths_mm:
        if path is None or len(path) < 2:
            continue
        # denser samples along path
        path_d = resample_polyline(path, n=max(120, len(path) * 2))
        for p in path_d:
            zyx = _mm_to_zyx(p, spacing_xyz, origin_xyz, air.shape)
            if not air[zyx]:
                zyx = nearest_air_index(air, zyx)
            r = float(radius_edt[zyx]) * radius_scale
            r = float(np.clip(r, min_radius_mm, max_radius_mm))
            # voxel radius (approx isotropic mean spacing)
            sp = float(np.mean(spacing_xyz))
            rv = max(int(np.ceil(r / sp)), 1)
            z, y, x = zyx
            z0, z1 = max(0, z - rv), min(air.shape[0], z + rv + 1)
            y0, y1 = max(0, y - rv), min(air.shape[1], y + rv + 1)
            x0, x1 = max(0, x - rv), min(air.shape[2], x + rv + 1)
            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            # ellipsoid in mm
            ball = (
                ((zz - z) * sz) ** 2 + ((yy - y) * sy) ** 2 + ((xx - x) * sx) ** 2
            ) <= (r * r)
            open_sp[z0:z1, y0:y1, x0:x1] |= ball & air[z0:z1, y0:y1, x0:x1]

    # Keep only largest component touching both path starts if possible
    lab, n = ndi.label(open_sp)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        open_sp = lab == int(np.argmax(counts))
    return open_sp.astype(bool)


def compute_dual_most_open_paths(
    air: np.ndarray,
    left_naris_mm: list[float],
    right_naris_mm: list[float],
    trachea_mm: list[float],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    case_id: str = "case",
    power: float = 2.0,
    symmetry_blend: float = 0.35,
    radius_scale: float = 1.15,
) -> OpenPathResult:
    """
    Full demo pipeline: dual most-open centerlines + soft symmetry + open space.
    """
    notes: list[str] = []
    air = air.astype(bool)
    if not air.any():
        raise ValueError("Empty air domain")

    # Order L/R by LPS (left = higher x)
    ln = np.array(left_naris_mm, dtype=float)
    rn = np.array(right_naris_mm, dtype=float)
    if ln[0] < rn[0]:
        ln, rn = rn, ln
        notes.append("Swapped naris inputs so left = higher x (LPS).")

    x_mid = 0.5 * (float(ln[0]) + float(rn[0]))
    start_l = _mm_to_zyx(ln, spacing_xyz, origin_xyz, air.shape)
    start_r = _mm_to_zyx(rn, spacing_xyz, origin_xyz, air.shape)
    end = _mm_to_zyx(trachea_mm, spacing_xyz, origin_xyz, air.shape)

    idx_l = most_open_path_zyx(air, start_l, end, spacing_xyz, power=power)
    idx_r = most_open_path_zyx(air, start_r, end, spacing_xyz, power=power)
    path_l = path_zyx_to_mm(idx_l, spacing_xyz, origin_xyz)
    path_r = path_zyx_to_mm(idx_r, spacing_xyz, origin_xyz)

    # Prepend exact naris mm so path starts at the opening
    if len(path_l) and np.linalg.norm(path_l[0] - ln) > 0.5:
        path_l = np.vstack([ln, path_l])
    if len(path_r) and np.linalg.norm(path_r[0] - rn) > 0.5:
        path_r = np.vstack([rn, path_r])

    notes.append(
        f"Most-open paths: cost=1/(r+ε)^{power} (EDT radius); "
        f"len L={path_length_mm(path_l):.1f} mm R={path_length_mm(path_r):.1f} mm."
    )

    if symmetry_blend > 0:
        path_l, path_r = soft_symmetric_pair(
            path_l, path_r, x_mid_mm=x_mid, blend=symmetry_blend, n=80
        )
        path_l = project_path_to_air(path_l, air, spacing_xyz, origin_xyz)
        path_r = project_path_to_air(path_r, air, spacing_xyz, origin_xyz)
        # Re-anchor endpoints
        path_l[0] = ln
        path_r[0] = rn
        path_l[-1] = path_zyx_to_mm(
            [nearest_air_index(air, end)], spacing_xyz, origin_xyz
        )[0]
        path_r[-1] = path_l[-1].copy()
        notes.append(
            f"Soft symmetry blend={symmetry_blend} about midplane x={x_mid:.2f} mm."
        )

    # Midline path for reference (average of dual after resample)
    pl = resample_polyline(path_l, 80)
    pr = resample_polyline(path_r, 80)
    mid = 0.5 * (pl + pr)
    mid = project_path_to_air(mid, air, spacing_xyz, origin_xyz)

    open_space = infer_open_space_from_paths(
        air,
        [path_l, path_r],
        spacing_xyz,
        origin_xyz,
        radius_scale=radius_scale,
    )
    notes.append(
        f"Open space inferred from dual tubes: {int(open_space.sum())} voxels "
        f"(scale={radius_scale})."
    )

    return OpenPathResult(
        case_id=case_id,
        centerline_left_mm=path_l.tolist(),
        centerline_right_mm=path_r.tolist(),
        centerline_mid_mm=mid.tolist(),
        open_space=open_space,
        length_left_mm=path_length_mm(path_l),
        length_right_mm=path_length_mm(path_r),
        x_midplane_mm=float(x_mid),
        notes=notes,
    )
