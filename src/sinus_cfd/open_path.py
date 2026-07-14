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


def most_open_cost_hu(
    air: np.ndarray,
    hu: np.ndarray | None,
    spacing_xyz: tuple[float, float, float],
    power: float = 2.0,
    eps: float = 0.5,
    hu_weight: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Most-open cost favoring wide lumen (high EDT) and dark air (low HU).

    Instrument path: stay centered between bone (EDT) and in the darkest
    air voxels on DICOM (most negative HU).
    """
    cost, radius = most_open_cost(air, spacing_xyz, power=power, eps=eps)
    if hu is None or hu_weight <= 0:
        return cost, radius
    inside = air.astype(bool)
    if not inside.any():
        return cost, radius
    # Map HU: darker (more negative) → lower multiplier
    h = hu.astype(np.float64)
    h_in = h[inside]
    # clip typical air range
    h_clip = np.clip(h_in, -1000.0, -50.0)
    # 0 at darkest, 1 at least-dark air
    h_norm = (h_clip - h_clip.min()) / max(float(h_clip.max() - h_clip.min()), 1.0)
    # cost *= (1 + hu_weight * h_norm) so bright partial-volume is pricier
    mult = np.ones_like(cost)
    mult[inside] = 1.0 + float(hu_weight) * h_norm
    cost = cost * mult
    return cost.astype(np.float64), radius


def most_open_path_zyx(
    air: np.ndarray,
    start_zyx: tuple[int, int, int],
    end_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    power: float = 2.0,
    hu: np.ndarray | None = None,
    hu_weight: float = 0.55,
    straight_bias: float = 0.0,
) -> list[tuple[int, int, int]]:
    """
    Discrete most-open path indices (z,y,x).

    straight_bias > 0 penalizes deviation from the straight start→end chord
    so instrument corridors stay straighter (still stay in open dark air).
    """
    start = nearest_air_index(air, start_zyx)
    end = nearest_air_index(air, end_zyx)
    if hu is not None:
        cost, _ = most_open_cost_hu(
            air, hu, spacing_xyz, power=power, hu_weight=hu_weight
        )
    else:
        cost, _ = most_open_cost(air, spacing_xyz, power=power)

    if straight_bias > 0:
        # Distance (voxels) from the start–end line segment
        a = np.array(start, dtype=float)
        b = np.array(end, dtype=float)
        ab = b - a
        ab2 = float(np.dot(ab, ab)) + 1e-9
        zz, yy, xx = np.where(air)
        pts = np.column_stack([zz, yy, xx]).astype(float)
        t = np.clip(np.dot(pts - a, ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.linalg.norm(pts - proj, axis=1)
        # Mild penalty so path prefers a straight corridor without leaving air
        pen = 1.0 + float(straight_bias) * (dist / (dist.max() + 1e-6))
        cost = cost.copy()
        cost[zz, yy, xx] *= pen

    try:
        indices, _ = route_through_array(
            cost, start=start, end=end, fully_connected=True, geometric=True
        )
    except Exception:
        indices = [start, end]
    return [(int(a), int(b), int(c)) for a, b, c in indices]


def split_frontal_lr(
    frontal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split frontal mask into L/R by x median (high x = patient left)."""
    zz, yy, xx = np.where(frontal.astype(bool))
    if len(xx) == 0:
        z = np.zeros_like(frontal, dtype=bool)
        return z, z, 0.0
    x_mid = float(np.median(xx))
    left = frontal & (np.arange(frontal.shape[2])[None, None, :] >= x_mid)
    right = frontal & (np.arange(frontal.shape[2])[None, None, :] < x_mid)
    # If one side empty, fall back to full frontal for both
    if not left.any():
        left = frontal.copy()
    if not right.any():
        right = frontal.copy()
    return left.astype(bool), right.astype(bool), x_mid


def straighten_path_in_air(
    path_mm: np.ndarray,
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    blend: float = 0.65,
    n: int = 48,
) -> np.ndarray:
    """
    Pull an instrument path toward the start→end chord while snapping to air.

    Makes naris→frontal corridors look straighter for surgical planning without
    leaving the lumen.
    """
    if path_mm is None or len(path_mm) < 3:
        return path_mm
    pts = resample_polyline(np.asarray(path_mm, dtype=float), n)
    a, b = pts[0].copy(), pts[-1].copy()
    out = []
    for i, p in enumerate(pts):
        t = i / max(n - 1, 1)
        chord = (1.0 - t) * a + t * b
        blended = (1.0 - blend) * p + blend * chord
        zyx = nearest_air_index(
            air, _mm_to_zyx(blended, spacing_xyz, origin_xyz, air.shape)
        )
        out.append(_zyx_to_mm(zyx, spacing_xyz, origin_xyz))
    # ensure exact start/end anchors
    out[0] = a
    out[-1] = b
    return np.asarray(out, dtype=float)


def _local_air_snap(
    p_mm: np.ndarray,
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    rad: int = 4,
) -> np.ndarray:
    """Snap to nearest air within a small ball (preserve designed path shape)."""
    z0, y0, x0 = _mm_to_zyx(p_mm, spacing_xyz, origin_xyz, air.shape)
    if air[z0, y0, x0]:
        return np.asarray(p_mm, dtype=float)
    nz, ny, nx = air.shape
    best = None
    best_d = 1e18
    for dz in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                q = (z0 + dz, y0 + dy, x0 + dx)
                if not (0 <= q[0] < nz and 0 <= q[1] < ny and 0 <= q[2] < nx):
                    continue
                if not air[q]:
                    continue
                d = float(dz * dz + dy * dy + dx * dx)
                if d < best_d:
                    best_d = d
                    best = q
    if best is None:
        best = nearest_air_index(air, (z0, y0, x0))
    return _zyx_to_mm(best, spacing_xyz, origin_xyz)


def build_lateral_diverge_frontal_path(
    start_mm: np.ndarray | list[float],
    end_mm: np.ndarray | list[float],
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    side: str,
    n: int = 56,
    lateral_flare: float = 1.15,
) -> np.ndarray:
    """
    Instrument path naris → frontal:

    - **Sagittal (y–z):** nearly straight (linear y(t), z(t)).
    - **Coronal (x–z):** diverges laterally (L → +x, R → −x on this CT).

    x is prescribed for lateral flare; only y/z are adjusted to stay in air.
    """
    a = np.asarray(start_mm, dtype=float).copy()
    b = np.asarray(end_mm, dtype=float).copy()
    # Designed end: superior, slightly deeper, clearly lateral
    lateral_mm = 12.0 * float(lateral_flare)
    if side == "left":
        x_end = a[0] + lateral_mm
    else:
        x_end = a[0] - lateral_mm
    y_end = float(np.clip(b[1], a[1] + 10.0, a[1] + 32.0))
    z_end = float(max(b[2], a[2] + 28.0))

    t = np.linspace(0.0, 1.0, n)
    # Stay medial early, then slight lateral diverge only in superior half
    # (clinical: corridor tracks near septum, flares into frontal when high)
    mid_hold = 0.55  # fraction of path that stays near start x (medial corridor)
    lat_t = np.clip((t - mid_hold) / max(1e-6, 1.0 - mid_hold), 0.0, 1.0)
    lat = lat_t * lat_t * (3.0 - 2.0 * lat_t)  # smoothstep after hold
    lat = lat ** 1.15  # gentle late flare
    # Only mild total lateral offset (not aggressive)
    if side == "left":
        x_end = a[0] + 6.0 * float(lateral_flare)
    else:
        x_end = a[0] - 6.0 * float(lateral_flare)
    xs = (1.0 - lat) * a[0] + lat * x_end
    ys = (1.0 - t) * a[1] + t * y_end
    zs = (1.0 - t) * a[2] + t * z_end

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    nz, ny, nx = air.shape
    out = []
    for i in range(n):
        p = np.array([xs[i], ys[i], zs[i]], dtype=float)
        # Search air near (y,z) allowing small x drift but prefer designed x
        z0 = int(np.clip(round((p[2] - oz) / sz), 0, nz - 1))
        y0 = int(np.clip(round((p[1] - oy) / sy), 0, ny - 1))
        x0 = int(np.clip(round((p[0] - ox) / sx), 0, nx - 1))
        best = None
        best_score = 1e18
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                for dx in range(-2, 3):
                    q = (z0 + dz, y0 + dy, x0 + dx)
                    if not (0 <= q[0] < nz and 0 <= q[1] < ny and 0 <= q[2] < nx):
                        continue
                    if not air[q]:
                        continue
                    # Prefer designed x strongly; mild y/z preference
                    score = (
                        4.0 * abs(dx)
                        + 1.2 * abs(dy)
                        + 1.2 * abs(dz)
                    )
                    if score < best_score:
                        best_score = score
                        best = q
        if best is None:
            # fall back local snap
            p2 = _local_air_snap(p, air, spacing_xyz, origin_xyz, rad=5)
            # still enforce lateral x trend
            if side == "left":
                p2[0] = max(p2[0], xs[i] - 1.5)
            else:
                p2[0] = min(p2[0], xs[i] + 1.5)
            out.append(p2)
        else:
            out.append(_zyx_to_mm(best, spacing_xyz, origin_xyz))
    arr = np.asarray(out, dtype=float)
    arr[0] = a
    # Enforce monotonic lateral after search
    if side == "left":
        for i in range(1, len(arr)):
            arr[i, 0] = max(arr[i, 0], arr[i - 1, 0])
            arr[i, 0] = max(arr[i, 0], xs[i] - 2.0)
    else:
        for i in range(1, len(arr)):
            arr[i, 0] = min(arr[i, 0], arr[i - 1, 0])
            arr[i, 0] = min(arr[i, 0], xs[i] + 2.0)
    # Soft-blend y,z back to linear sagittal (instrument straight in sagittal)
    for i in range(len(arr)):
        arr[i, 1] = 0.35 * arr[i, 1] + 0.65 * ys[i]
        arr[i, 2] = 0.35 * arr[i, 2] + 0.65 * zs[i]
    arr[0] = a
    return arr


def smooth_instrument_path(
    path_mm: np.ndarray,
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    n: int = 64,
    smooth_passes: int = 4,
    max_turn_deg: float = 22.0,
) -> np.ndarray:
    """
    Smooth instrument corridor: relatively straight, **no sharp corners**.

    Moving-average smooth + turn-angle clamp, then snap back into air.
    Suitable for a rigid endoscope that bends gently only.
    """
    if path_mm is None or len(path_mm) < 3:
        return path_mm
    pts = resample_polyline(np.asarray(path_mm, dtype=float), n)
    a0, a1 = pts[0].copy(), pts[-1].copy()

    for _ in range(max(1, smooth_passes)):
        sm = pts.copy()
        for i in range(1, len(pts) - 1):
            sm[i] = 0.25 * pts[i - 1] + 0.5 * pts[i] + 0.25 * pts[i + 1]
        pts = sm
        # Clamp sharp turns
        max_rad = np.deg2rad(max_turn_deg)
        for i in range(1, len(pts) - 1):
            v0 = pts[i] - pts[i - 1]
            v1 = pts[i + 1] - pts[i]
            n0 = float(np.linalg.norm(v0))
            n1 = float(np.linalg.norm(v1))
            if n0 < 1e-9 or n1 < 1e-9:
                continue
            u0, u1 = v0 / n0, v1 / n1
            cos_a = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
            ang = float(np.arccos(cos_a))
            if ang > max_rad:
                # Rotate v1 toward v0 to reduce turn
                # Place next point along bisector closer to u0
                blend = max_rad / max(ang, 1e-6)
                u_new = u0 * (1.0 - blend) + u1 * blend
                un = float(np.linalg.norm(u_new)) + 1e-12
                pts[i + 1] = pts[i] + (u_new / un) * n1

    # Snap to air (except endpoints)
    out = [a0]
    for p in pts[1:-1]:
        zyx = nearest_air_index(air, _mm_to_zyx(p, spacing_xyz, origin_xyz, air.shape))
        out.append(_zyx_to_mm(zyx, spacing_xyz, origin_xyz))
    out.append(a1)
    # Final light smooth on air-snapped points (preserve ends)
    arr = np.asarray(out, dtype=float)
    for _ in range(2):
        sm = arr.copy()
        for i in range(1, len(arr) - 1):
            sm[i] = 0.2 * arr[i - 1] + 0.6 * arr[i] + 0.2 * arr[i + 1]
        arr = sm
        arr[0], arr[-1] = a0, a1
    # Re-snap midpoints once more
    final = [a0]
    for p in arr[1:-1]:
        zyx = nearest_air_index(air, _mm_to_zyx(p, spacing_xyz, origin_xyz, air.shape))
        final.append(_zyx_to_mm(zyx, spacing_xyz, origin_xyz))
    final.append(a1)
    return np.asarray(final, dtype=float)


def restriction_along_paths_high_speed(
    path_mm_list: list[np.ndarray],
    speed: np.ndarray,
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    speed_percentile: float = 82.0,
    tube_radius_mm: float = 4.0,
    max_points: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Magenta surgical targets: high |u| along naris→trachea pathways.

    A larger opening here would relieve velocity / resistance.
    Returns (mask, points_xyz_speed Nx4).
    """
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    mask = np.zeros_like(air, dtype=bool)
    if not path_mm_list:
        return mask, np.zeros((0, 4), dtype=np.float32)

    # Corridor around paths
    zz, yy, xx = np.where(air)
    if len(zz) == 0:
        return mask, np.zeros((0, 4), dtype=np.float32)
    px = ox + xx * sx
    py = oy + yy * sy
    pz = oz + zz * sz
    pts_air = np.column_stack([px, py, pz])

    near = np.zeros(len(zz), dtype=bool)
    for path in path_mm_list:
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            continue
        # subsample path for speed
        if len(path) > 80:
            path = path[:: max(1, len(path) // 80)]
        # distance to polyline (approx nearest vertex)
        # batch in chunks
        dmin = np.full(len(zz), np.inf)
        for i in range(0, len(path), 8):
            chunk = path[i : i + 8]
            # (N_air, K)
            d = np.linalg.norm(pts_air[:, None, :] - chunk[None, :, :], axis=2).min(axis=1)
            dmin = np.minimum(dmin, d)
        near |= dmin <= tube_radius_mm

    if not near.any():
        return mask, np.zeros((0, 4), dtype=np.float32)

    sp = speed[zz[near], yy[near], xx[near]]
    thr = float(np.percentile(sp, speed_percentile)) if sp.size else 0.0
    thr = max(thr, float(np.percentile(speed[air], 75)) if air.any() else thr)
    hot = near.copy()
    hot[near] = sp >= thr
    mask[zz[hot], yy[hot], xx[hot]] = True

    # Dilate slightly for visibility
    if mask.any():
        mask = morphology.binary_dilation(mask, footprint=morphology.ball(1)) & air

    hz, hy, hx = np.where(mask)
    if len(hz) == 0:
        return mask, np.zeros((0, 4), dtype=np.float32)
    if len(hz) > max_points:
        rng = np.random.default_rng(9)
        pick = rng.choice(len(hz), size=max_points, replace=False)
        hz, hy, hx = hz[pick], hy[pick], hx[pick]
    pts = np.column_stack(
        [
            ox + hx * sx,
            oy + hy * sy,
            oz + hz * sz,
            speed[hz, hy, hx],
        ]
    ).astype(np.float32)
    return mask, pts


def path_restriction_highlights(
    path_zyx: list[tuple[int, int, int]],
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    radius: np.ndarray | None = None,
    narrow_percentile: float = 30.0,
    ball_radius: int = 2,
) -> tuple[np.ndarray, list[dict]]:
    """
    Magenta surgical targets: narrowest segments along an access path.

    Returns (highlight_mask, list of bottleneck records with mm coords).
    """
    sx, sy, sz = spacing_xyz
    if radius is None:
        radius = ndi.distance_transform_edt(air, sampling=(sz, sy, sx)).astype(np.float32)
    if not path_zyx:
        return np.zeros_like(air), []
    r_along = np.array([float(radius[p]) for p in path_zyx], dtype=float)
    thr = float(np.percentile(r_along, narrow_percentile))
    thr = max(thr, float(min(sx, sy, sz)) * 0.9)
    highlight = np.zeros_like(air, dtype=bool)
    bottlenecks: list[dict] = []
    for i, p in enumerate(path_zyx):
        if r_along[i] <= thr:
            highlight[p] = True
    if highlight.any():
        highlight = morphology.binary_dilation(
            highlight, footprint=morphology.ball(ball_radius)
        )
        highlight &= air
    # Cluster bottleneck centers for labels
    lab, n = ndi.label(highlight)
    for li in range(1, n + 1):
        zz, yy, xx = np.where(lab == li)
        if len(zz) < 3:
            continue
        zc, yc, xc = int(zz.mean()), int(yy.mean()), int(xx.mean())
        bottlenecks.append(
            {
                "center_mm": [
                    float(origin_xyz[0] + xc * sx),
                    float(origin_xyz[1] + yc * sy),
                    float(origin_xyz[2] + zc * sz),
                ],
                "voxels": int(len(zz)),
                "mean_radius_mm": float(radius[zz, yy, xx].mean()),
                "min_radius_mm": float(radius[zz, yy, xx].min()),
            }
        )
    bottlenecks.sort(key=lambda b: b["min_radius_mm"])
    return highlight, bottlenecks


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
