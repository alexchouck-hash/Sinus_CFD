"""
Approximate steady airflow field for visualization and early analysis.

Method: potential / Darcy flow on the voxel airway mask
  - Solve Laplace pressure ∇²p ≈ 0 inside the airway
  - Inlets (nostrils): p = 1
  - Outlet (trachea proxy): p = 0
  - Walls: no-flux (Neumann via not updating exterior)
  - Velocity u = -∇p, then rescaled so total inlet flux matches physiology Q

This is NOT full Navier–Stokes CFD. It gives a continuous, anatomically
guided velocity field for the viewer until OpenFOAM (or similar) is wired in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .physiology import PatientBreathing


@dataclass
class FlowFieldResult:
    case_id: str
    spacing_xyz_mm: list[float]
    origin_xyz_mm: list[float]
    size_zyx: list[int]
    method: str
    target_flow_L_per_min: float
    achieved_inlet_flux_L_per_min: float
    max_speed_m_s: float
    mean_speed_m_s: float
    n_airway_voxels: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phys_to_index(
    points_mm: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Map Nx3 physical (x,y,z) mm → integer (z,y,x) indices, clipped."""
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    x = np.rint((points_mm[:, 0] - ox) / sx).astype(int)
    y = np.rint((points_mm[:, 1] - oy) / sy).astype(int)
    z = np.rint((points_mm[:, 2] - oz) / sz).astype(int)
    z = np.clip(z, 0, shape_zyx[0] - 1)
    y = np.clip(y, 0, shape_zyx[1] - 1)
    x = np.clip(x, 0, shape_zyx[2] - 1)
    return np.column_stack([z, y, x])


def _port_seed_mask(
    shape_zyx: tuple[int, int, int],
    center_mm: list[float],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    airway: np.ndarray,
    radius_mm: float = 8.0,
) -> np.ndarray:
    """
    Spherical seed around a port center, restricted to airway.

    Skin-surface naris centers may lie just outside the lumen; we always fall
    back to the nearest airway voxels so inlet/outlet BCs remain inside the fluid.
    """
    zz, yy, xx = np.indices(shape_zyx)
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    px = ox + xx * sx
    py = oy + yy * sy
    pz = oz + zz * sz
    cx, cy, cz = center_mm
    dist2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
    seed = (dist2 <= radius_mm**2) & airway
    if not seed.any():
        # Nearest airway voxels to the port (handles external skin naris centers)
        idx = _phys_to_index(
            np.array([center_mm], dtype=float),
            spacing_xyz,
            origin_xyz,
            shape_zyx,
        )[0]
        air_idx = np.column_stack(np.where(airway))
        if len(air_idx) == 0:
            return seed
        d = np.linalg.norm(air_idx.astype(float) - idx.astype(float), axis=1)
        # Take a neighborhood of nearest airway voxels
        k = min(80, len(d))
        nearest_ids = np.argpartition(d, k - 1)[:k]
        seed = np.zeros(shape_zyx, dtype=bool)
        for j in nearest_ids:
            seed[tuple(air_idx[j])] = True
        seed = ndi.binary_dilation(seed, iterations=2) & airway
    return seed


def solve_pressure_potential(
    airway: np.ndarray,
    inlet_mask: np.ndarray,
    outlet_mask: np.ndarray,
    iterations: int = 600,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Stable Jacobi Laplace solve for pressure on the airway mask.

    p=1 on inlets, p=0 on outlets. Walls are Neumann: only airway neighbors
    contribute (6-point stencil). Pressure is clipped to [0, 1] each step.
    """
    airway = airway.astype(bool)
    air_f = airway.astype(np.float64)
    p = np.full(airway.shape, 0.5, dtype=np.float64)
    p[~airway] = 0.0
    p[inlet_mask & airway] = 1.0
    p[outlet_mask & airway] = 0.0

    interior = airway & ~inlet_mask & ~outlet_mask
    kernel = np.zeros((3, 3, 3), dtype=np.float64)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0

    neighbor_count = ndi.convolve(air_f, kernel, mode="constant", cval=0.0)
    neighbor_count = np.maximum(neighbor_count, 1.0)

    for _it in range(iterations):
        p_air = p * air_f  # zero outside airway so walls don't pull from exterior
        neighbor_sum = ndi.convolve(p_air, kernel, mode="constant", cval=0.0)
        avg = neighbor_sum / neighbor_count
        p_new = np.clip(avg, 0.0, 1.0)
        delta = np.abs(p_new - p)[interior]
        p = np.where(interior, p_new, p)
        p[inlet_mask & airway] = 1.0
        p[outlet_mask & airway] = 0.0
        p[~airway] = 0.0
        if delta.size and float(delta.max()) < tol:
            break

    p = p.astype(np.float64)
    p[~airway] = np.nan
    return p


def pressure_to_velocity(
    p: np.ndarray,
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    u = -∇p with physical spacing (mm → m for gradient in SI-ish units later).

    Returns ux, uy, uz, speed in *relative* units (before flux scaling).
    Arrays shaped (z,y,x); components in +x,+y,+z physical axes.
    """
    # np.gradient on z,y,x with spacing sz,sy,sx in mm
    sx, sy, sz = spacing_xyz_mm
    # p is (z,y,x); gradient returns d/dz, d/dy, d/dx
    # Replace nan with local fill for gradient stability
    p_fill = np.where(np.isfinite(p), p, 0.0)
    gz, gy, gx = np.gradient(p_fill, sz, sy, sx)
    # u = -grad p  → (ux, uy, uz) corresponding to physical axes
    ux = -gx
    uy = -gy
    uz = -gz
    # Zero outside airway
    mask = airway.astype(bool)
    ux = np.where(mask, ux, 0.0)
    uy = np.where(mask, uy, 0.0)
    uz = np.where(mask, uz, 0.0)
    speed = np.sqrt(ux**2 + uy**2 + uz**2)
    return ux, uy, uz, speed


def scale_velocity_to_flow_rate(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    airway: np.ndarray,
    inlet_mask: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    target_L_per_min: float,
    inlet_area_mm2: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Scale relative velocity so bulk inlet speed ≈ Q / A_inlet.

    Uses labeled port open-area when provided; otherwise estimates A from the
    inlet seed voxel count × in-plane pixel area.
    """
    sx, sy, sz = spacing_xyz_mm
    inlet = inlet_mask & airway
    sp = np.sqrt(ux**2 + uy**2 + uz**2)
    if not inlet.any():
        return ux, uy, uz, sp, 0.0

    if inlet_area_mm2 is None or inlet_area_mm2 <= 0:
        face_mm2 = float(np.prod(sorted([sx, sy, sz])[:2]))
        # surface-like shell: use count^{2/3} * face as rough area
        inlet_area_mm2 = float(max(inlet.sum(), 1) ** (2.0 / 3.0) * face_mm2)

    area_m2 = max(inlet_area_mm2, 1.0) * 1e-6
    target_m3_s = target_L_per_min / 1000.0 / 60.0
    u_bulk = target_m3_s / area_m2  # m/s characteristic

    sp_in = float(sp[inlet].mean())
    if sp_in < 1e-30:
        scale = 1.0
    else:
        scale = u_bulk / sp_in

    ux_s = ux * scale
    uy_s = uy * scale
    uz_s = uz * scale
    speed_s = np.sqrt(ux_s**2 + uy_s**2 + uz_s**2)
    achieved = float(speed_s[inlet].mean() * area_m2) * 1000.0 * 60.0  # L/min
    return ux_s, uy_s, uz_s, speed_s, achieved


def sample_velocity_trilinear(
    pos_mm: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
) -> np.ndarray | None:
    """
    Smooth velocity sample so streamlines bend continuously around corners
    (nearest-neighbor makes angular, “straight segment” paths).
    """
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm
    nz, ny, nx = airway.shape
    x, y, z = float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2])
    fx = (x - ox) / sx
    fy = (y - oy) / sy
    fz = (z - oz) / sz
    i0 = int(np.floor(fz))
    j0 = int(np.floor(fy))
    k0 = int(np.floor(fx))
    if not (0 <= i0 < nz - 1 and 0 <= j0 < ny - 1 and 0 <= k0 < nx - 1):
        return None
    if not airway[i0, j0, k0]:
        # allow sample if any corner of cell is air
        corners = airway[i0 : i0 + 2, j0 : j0 + 2, k0 : k0 + 2]
        if not corners.any():
            return None
    di = fz - i0
    dj = fy - j0
    dk = fx - k0

    def _corner(ii: int, jj: int, kk: int) -> np.ndarray:
        if not airway[ii, jj, kk]:
            return np.zeros(3, dtype=float)
        return np.array([ux[ii, jj, kk], uy[ii, jj, kk], uz[ii, jj, kk]], dtype=float)

    c000 = _corner(i0, j0, k0)
    c001 = _corner(i0, j0, k0 + 1)
    c010 = _corner(i0, j0 + 1, k0)
    c011 = _corner(i0, j0 + 1, k0 + 1)
    c100 = _corner(i0 + 1, j0, k0)
    c101 = _corner(i0 + 1, j0, k0 + 1)
    c110 = _corner(i0 + 1, j0 + 1, k0)
    c111 = _corner(i0 + 1, j0 + 1, k0 + 1)
    c00 = c000 * (1 - dk) + c001 * dk
    c01 = c010 * (1 - dk) + c011 * dk
    c10 = c100 * (1 - dk) + c101 * dk
    c11 = c110 * (1 - dk) + c111 * dk
    c0 = c00 * (1 - dj) + c01 * dj
    c1 = c10 * (1 - dj) + c11 * dj
    v = c0 * (1 - di) + c1 * di
    if float(np.linalg.norm(v)) < 1e-14:
        return None
    return v


def compute_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    seed_points_mm: np.ndarray,
    max_steps: int = 400,
    step_mm: float = 0.4,
    reverse: bool = False,
    use_trilinear: bool = True,
    swirl: float = 0.0,
    rng: np.random.Generator | None = None,
    attract_mm: np.ndarray | None = None,
    attract_strength: float = 0.0,
) -> list[np.ndarray]:
    """
    Integrate streamlines from seed points (physical mm).

    - use_trilinear: smooth U sampling so paths curve around bends.
    - RK2 mid-point step; reduced step near walls.
    - swirl: perpendicular meander for a curvy / turbulent look (demo).
    - attract_mm + attract_strength: soft pull toward a point (e.g. trachea
      on inhale) so paths generally head downstream without erasing bends.

    Returns list of (N_i, 3) arrays of polyline vertices in mm.
    If reverse=True, integrates against the velocity field.
    """
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm
    shape = airway.shape
    lines: list[np.ndarray] = []
    sign = -1.0 if reverse else 1.0
    rng = rng or np.random.default_rng(0)
    attract = (
        np.asarray(attract_mm, dtype=float)
        if attract_mm is not None and attract_strength > 0
        else None
    )

    def sample(pos_mm: np.ndarray) -> np.ndarray | None:
        if use_trilinear:
            v = sample_velocity_trilinear(
                pos_mm, ux, uy, uz, airway, spacing_xyz_mm, origin_xyz_mm
            )
            if v is None:
                return None
            return sign * v
        x, y, z = pos_mm
        ix = (x - ox) / sx
        iy = (y - oy) / sy
        iz = (z - oz) / sz
        i0, j0, k0 = int(np.floor(iz)), int(np.floor(iy)), int(np.floor(ix))
        if not (0 <= i0 < shape[0] - 1 and 0 <= j0 < shape[1] - 1 and 0 <= k0 < shape[2] - 1):
            return None
        if not airway[i0, j0, k0]:
            return None
        return sign * np.array([ux[i0, j0, k0], uy[i0, j0, k0], uz[i0, j0, k0]], dtype=float)

    for seed_i, seed in enumerate(seed_points_mm):
        pts = [seed.astype(float)]
        pos = seed.astype(float)
        # Per-seed phase so meander doesn't lock-step across all lines
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        # Stable perpendicular basis for smooth helical meander
        meander_axis = rng.normal(0.0, 1.0, size=3)
        for step_i in range(max_steps):
            v = sample(pos)
            if v is None:
                break
            speed = float(np.linalg.norm(v))
            if speed < 1e-12:
                break
            direction = v / speed

            # Soft pull toward trachea (or other attractor) — stronger when far
            if attract is not None and not reverse:
                to_att = attract - pos
                d_att = float(np.linalg.norm(to_att))
                if d_att > 1e-6:
                    to_att = to_att / d_att
                    # Fade pull as we near outlet so we don't overshoot
                    fade = min(1.0, d_att / 40.0)
                    direction = direction + float(attract_strength) * fade * to_att
                    nrm = float(np.linalg.norm(direction))
                    if nrm > 1e-12:
                        direction = direction / nrm

            # Curvy meander: smooth helical offset (not white noise spikes)
            if swirl > 1e-6:
                # Build orthonormal frame around flow direction
                tmp = meander_axis - direction * float(np.dot(meander_axis, direction))
                tn = float(np.linalg.norm(tmp))
                if tn < 1e-8:
                    tmp = np.array([1.0, 0.0, 0.0])
                    tmp = tmp - direction * float(np.dot(tmp, direction))
                    tn = float(np.linalg.norm(tmp)) + 1e-12
                e1 = tmp / tn
                e2 = np.cross(direction, e1)
                e2n = float(np.linalg.norm(e2)) + 1e-12
                e2 = e2 / e2n
                # Spatial frequency ~ one full turn every ~18 mm of travel
                ang = phase + (step_i * float(step_mm) / 18.0) * 2.0 * np.pi
                # Amplitude grows then settles; stronger when flow is faster
                amp = float(swirl) * (0.55 + 0.45 * min(1.0, speed / (speed + 0.25)))
                amp *= 0.85 + 0.15 * np.sin(0.37 * step_i)  # slight modulation
                direction = direction + amp * (np.cos(ang) * e1 + np.sin(ang) * e2)
                nrm = float(np.linalg.norm(direction))
                if nrm > 1e-12:
                    direction = direction / nrm

            # RK2
            h = float(step_mm)
            mid = pos + 0.5 * h * direction
            v2 = sample(mid)
            if v2 is not None and float(np.linalg.norm(v2)) > 1e-12:
                d2 = v2 / float(np.linalg.norm(v2))
                # Keep some meander at midpoint blend
                direction = 0.55 * direction + 0.45 * d2
                direction = direction / (float(np.linalg.norm(direction)) + 1e-12)
            new_pos = pos + h * direction
            idx = _phys_to_index(new_pos.reshape(1, 3), spacing_xyz_mm, origin_xyz_mm, shape)[0]
            if not airway[tuple(idx)]:
                advanced = False
                for frac in (0.5, 0.25, 0.12):
                    trial = pos + frac * h * direction
                    idx2 = _phys_to_index(
                        trial.reshape(1, 3), spacing_xyz_mm, origin_xyz_mm, shape
                    )[0]
                    if airway[tuple(idx2)]:
                        new_pos = trial
                        advanced = True
                        break
                if not advanced:
                    break
            pos = new_pos
            pts.append(pos.copy())
        if len(pts) >= 5:
            lines.append(np.vstack(pts))
    return lines


def seeds_throughout_volume(
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    n_seeds: int = 200,
    speed: np.ndarray | None = None,
    prefer_speed: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Scatter seeds through the full air domain (sinuses + nasal + pharynx).

    When prefer_speed, mix high-speed cores with uniform coverage so secondary
    passages still get streamlines.
    """
    rng = rng or np.random.default_rng(11)
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm
    mask = airway.astype(bool)
    if speed is not None:
        active = mask & (speed > 1e-6)
        if not active.any():
            active = mask
    else:
        active = mask
    zz, yy, xx = np.where(active)
    if len(zz) == 0:
        return np.zeros((0, 3), dtype=float)
    n = min(int(n_seeds), len(zz))
    if prefer_speed and speed is not None and n >= 8:
        sp = speed[zz, yy, xx]
        # half by speed weight, half uniform
        n_fast = n // 2
        n_uni = n - n_fast
        w = np.clip(sp, 1e-6, None)
        w = w / w.sum()
        try:
            pick_fast = rng.choice(len(zz), size=n_fast, replace=False, p=w)
        except ValueError:
            pick_fast = rng.choice(len(zz), size=n_fast, replace=True, p=w)
        pick_uni = rng.choice(len(zz), size=n_uni, replace=False)
        pick = np.unique(np.concatenate([pick_fast, pick_uni]))
        if len(pick) < n:
            extra = rng.choice(len(zz), size=n - len(pick), replace=True)
            pick = np.concatenate([pick, extra])
    else:
        pick = rng.choice(len(zz), size=n, replace=False)
    pts = np.column_stack(
        [
            ox + xx[pick] * sx,
            oy + yy[pick] * sy,
            oz + zz[pick] * sz,
        ]
    ).astype(float)
    # small jitter inside voxel so lines don't stack
    pts += rng.normal(0.0, 0.25 * min(sx, sy, sz), size=pts.shape)
    return pts


def seeds_near_ports(
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    port_centers_mm: list[list[float] | np.ndarray],
    n_per_port: int = 20,
    radius_mm: float = 6.0,
    speed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Seed points on airway voxels nearest each port center (for inhale streamlines).
    """
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm
    zz, yy, xx = np.where(airway if speed is None else (airway & (speed > 1e-5)))
    if len(zz) == 0:
        zz, yy, xx = np.where(airway)
    if len(zz) == 0:
        return np.zeros((0, 3), dtype=float)
    px = ox + xx * sx
    py = oy + yy * sy
    pz = oz + zz * sz
    seeds: list[np.ndarray] = []
    rng = np.random.default_rng(7)
    for center in port_centers_mm:
        c = np.asarray(center, dtype=float)
        d2 = (px - c[0]) ** 2 + (py - c[1]) ** 2 + (pz - c[2]) ** 2
        near = d2 <= radius_mm**2
        if not near.any():
            k = min(n_per_port, len(d2))
            pick = np.argpartition(d2, k - 1)[:k]
        else:
            ids = np.where(near)[0]
            if len(ids) > n_per_port:
                pick = rng.choice(ids, size=n_per_port, replace=False)
            else:
                pick = ids
        for j in pick:
            seeds.append(np.array([px[j], py[j], pz[j]], dtype=float))
            # small jitter inside lumen
            for _ in range(1):
                seeds.append(
                    np.array([px[j], py[j], pz[j]], dtype=float)
                    + rng.normal(0, 0.6, size=3)
                )
    return np.array(seeds, dtype=float) if seeds else np.zeros((0, 3), dtype=float)


def project_skin_naris_into_lumen(
    skin_naris_mm: np.ndarray,
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    toward_mm: np.ndarray | None = None,
    max_steps: int = 80,
    step_mm: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk from external skin naris into the lumen along the path toward the airway.

    Returns (entry_on_lumen_mm, polyline_skin_to_entry including both ends).
    """
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm
    shape = airway.shape
    start = np.asarray(skin_naris_mm, dtype=float)

    # Direction: prefer toward lumen centroid / given target; default +y (posterior)
    if toward_mm is not None:
        direction = np.asarray(toward_mm, dtype=float) - start
    else:
        # Nearest airway voxel direction
        zz, yy, xx = np.where(airway)
        if len(zz) == 0:
            return start, start.reshape(1, 3)
        pts = np.column_stack(
            [ox + xx * sx, oy + yy * sy, oz + zz * sz]
        )
        j = int(np.argmin(np.sum((pts - start) ** 2, axis=1)))
        direction = pts[j] - start
    n = np.linalg.norm(direction)
    if n < 1e-9:
        direction = np.array([0.0, 1.0, 0.0])
    else:
        direction = direction / n

    poly = [start.copy()]
    pos = start.copy()
    entry = start.copy()
    for _ in range(max_steps):
        pos = pos + step_mm * direction
        poly.append(pos.copy())
        idx = _phys_to_index(pos.reshape(1, 3), spacing_xyz_mm, origin_xyz_mm, shape)[0]
        if airway[tuple(idx)]:
            entry = pos.copy()
            break
    return entry, np.vstack(poly)


def compute_inhale_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    inlet_centers_mm: list[list[float] | np.ndarray],
    outlet_center_mm: list[float] | np.ndarray | None = None,
    skin_naris_centers_mm: list[list[float] | np.ndarray] | None = None,
    n_per_naris: int = 22,
    max_steps: int = 900,
    step_mm: float = 0.35,
    reach_outlet_mm: float = 12.0,
    max_lines: int = 200,
    seed_radius_mm: float = 10.0,
) -> list[np.ndarray]:
    """
    Streamlines for inspiration: skin nares → nasal passage → trachea.

    - Prefer integration only on the provided airway mask (use *passage lumen*,
      not sinus-inflated solid, to avoid maxillary detours).
    - Seeds from lumen near inlets; prepend skin→lumen segment when skin
      naris centers are given (external openings on the face).
    - Prefer paths that approach the trachea / caudal outlet.
    """
    # Lumen seeds near open ports (internal openings)
    seeds = seeds_near_ports(
        airway,
        spacing_xyz_mm,
        origin_xyz_mm,
        inlet_centers_mm,
        n_per_port=n_per_naris,
        radius_mm=seed_radius_mm,
        speed=np.sqrt(ux * ux + uy * uy + uz * uz),
    )
    if seeds.size == 0:
        return []

    outlet = (
        np.asarray(outlet_center_mm, dtype=float)
        if outlet_center_mm is not None
        else None
    )

    # Map each seed to a skin naris (for entry stub) by nearest skin point
    skin_list = (
        [np.asarray(s, dtype=float) for s in skin_naris_centers_mm]
        if skin_naris_centers_mm
        else []
    )

    def score_lines(lines: list[np.ndarray]) -> float:
        if not lines:
            return -1.0
        scores = []
        for line in lines:
            if len(line) < 8:
                continue
            progress = float(np.linalg.norm(line[-1] - line[0]))
            if outlet is not None:
                d0 = float(np.linalg.norm(line[0] - outlet))
                d1 = float(np.linalg.norm(line[-1] - outlet))
                progress += max(0.0, d0 - d1) * 3.0
                if d1 < reach_outlet_mm:
                    progress += 80.0
            scores.append(progress)
        if not scores:
            return 0.0
        return float(np.mean(sorted(scores, reverse=True)[: min(12, len(scores))]))

    fwd = compute_streamlines(
        ux, uy, uz, airway, spacing_xyz_mm, origin_xyz_mm, seeds,
        max_steps=max_steps, step_mm=step_mm, reverse=False,
    )
    rev = compute_streamlines(
        ux, uy, uz, airway, spacing_xyz_mm, origin_xyz_mm, seeds,
        max_steps=max_steps, step_mm=step_mm, reverse=True,
    )
    lines = fwd if score_lines(fwd) >= score_lines(rev) else rev

    # Prepend skin → lumen entry so paths start at the face, not deep inside
    if skin_list:
        lumen_centroid = None
        zz, yy, xx = np.where(airway)
        if len(zz):
            sx, sy, sz = spacing_xyz_mm
            ox, oy, oz = origin_xyz_mm
            lumen_centroid = np.array(
                [
                    ox + xx.mean() * sx,
                    oy + yy.mean() * sy,
                    oz + zz.mean() * sz,
                ],
                dtype=float,
            )
        extended: list[np.ndarray] = []
        for line in lines:
            # nearest skin naris to line start
            skin = min(skin_list, key=lambda s: float(np.linalg.norm(s - line[0])))
            _entry, stub = project_skin_naris_into_lumen(
                skin,
                airway,
                spacing_xyz_mm,
                origin_xyz_mm,
                toward_mm=lumen_centroid if lumen_centroid is not None else line[0],
            )
            # join stub (skin→entry) + streamline from first point near entry
            if len(stub) >= 2:
                # drop first streamline points if far from stub end
                joined = np.vstack([stub, line])
            else:
                joined = np.vstack([skin.reshape(1, 3), line])
            extended.append(joined)
        lines = extended

    # Keep paths that travel toward the outlet
    kept: list[np.ndarray] = []
    for line in lines:
        travel = float(np.linalg.norm(line[-1] - line[0]))
        if travel < 15.0:  # mm
            continue
        if outlet is not None:
            d0 = float(np.linalg.norm(line[0] - outlet))
            d1 = float(np.linalg.norm(line[-1] - outlet))
            # must get closer to trachea overall
            if d1 > d0 - 5.0 and d1 > reach_outlet_mm * 2:
                # allow if still long and y-progress toward outlet
                if line[-1, 1] <= line[0, 1] + 10 and travel < 40:
                    continue
        kept.append(line)

    def sort_key(line: np.ndarray) -> float:
        travel = float(np.linalg.norm(line[-1] - line[0]))
        if outlet is None:
            return travel
        d0 = float(np.linalg.norm(line[0] - outlet))
        d1 = float(np.linalg.norm(line[-1] - outlet))
        bonus = 100.0 if d1 < reach_outlet_mm else 0.0
        return travel + max(0.0, d0 - d1) * 2.0 + bonus

    kept.sort(key=sort_key, reverse=True)
    n_keep = max(1, int(max_lines))
    return kept[:n_keep]


def _nearest_naris_mm(
    point: np.ndarray, naris_list: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    best = naris_list[0]
    best_d = float(np.linalg.norm(point - best))
    for n in naris_list[1:]:
        d = float(np.linalg.norm(point - n))
        if d < best_d:
            best, best_d = n, d
    return best, best_d


def _orient_naris_to_trachea(
    line: np.ndarray,
    naris_list: list[np.ndarray],
    trachea: np.ndarray | None,
) -> np.ndarray | None:
    """
    Flip polyline so it runs naris → trachea (inspiration direction).
    Returns None if the line does not approach a naris at either end.
    """
    line = np.asarray(line, dtype=float)
    if len(line) < 6:
        return None
    d0_n = min(float(np.linalg.norm(line[0] - n)) for n in naris_list)
    d1_n = min(float(np.linalg.norm(line[-1] - n)) for n in naris_list)
    # Prefer the end closer to a naris as the start
    if d1_n + 2.0 < d0_n:
        line = line[::-1].copy()
        d0_n, d1_n = d1_n, d0_n
    # Must start reasonably near a naris (or we'll prepend later)
    if trachea is not None:
        d0_t = float(np.linalg.norm(line[0] - trachea))
        d1_t = float(np.linalg.norm(line[-1] - trachea))
        # If still pointing wrong (start closer to trachea than end), flip
        if d0_t + 5.0 < d1_t and d1_n < d0_n + 8.0:
            # start is nearer trachea — reverse so we leave nares
            line = line[::-1].copy()
    return line


def compute_curvy_volume_pathlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    airway: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    naris_centers_mm: list[list[float]] | None = None,
    outlet_center_mm: list[float] | None = None,
    n_volume_seeds: int = 220,
    n_naris_seeds: int = 80,
    max_steps: int = 1600,
    step_mm: float = 0.22,
    swirl: float = 0.08,
    max_lines: int = 320,
    bidirectional: bool = True,
    rng_seed: int = 19,
    naris_start_max_mm: float = 14.0,
    trachea_end_max_mm: float = 18.0,
    centerline_mm: np.ndarray | list | None = None,
) -> list[np.ndarray]:
    """
    Curvy inhale pathlines seeded **throughout the airspace**, generally
    traveling toward the trachea (low-pressure outlet).

    - Volume seeds fill the lumen; naris seeds add entry density.
    - Higher seed speed → longer traces.
    - Soft trachea attract + meander for curvy wisps.
    """
    rng = np.random.default_rng(rng_seed)
    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm

    naris_list = (
        [np.asarray(n, dtype=float) for n in naris_centers_mm]
        if naris_centers_mm
        else []
    )
    trachea = (
        np.asarray(outlet_center_mm, dtype=float)
        if outlet_center_mm is not None
        else None
    )

    seeds_list: list[np.ndarray] = []
    # Primary: seeds throughout airspace
    if n_volume_seeds > 0:
        vol = seeds_throughout_volume(
            airway,
            spacing_xyz_mm,
            origin_xyz_mm,
            n_seeds=max(n_volume_seeds, 120),
            speed=speed,
            prefer_speed=True,
            rng=rng,
        )
        if len(vol):
            seeds_list.append(vol)
    # Secondary: naris entry cloud
    if naris_centers_mm and n_naris_seeds > 0:
        port = seeds_near_ports(
            airway,
            spacing_xyz_mm,
            origin_xyz_mm,
            naris_centers_mm,
            n_per_port=max(12, n_naris_seeds // max(1, len(naris_centers_mm))),
            radius_mm=10.0,
            speed=speed,
        )
        if len(port):
            seeds_list.append(port)

    if not seeds_list:
        return []
    seeds = np.vstack(seeds_list)

    # Mean speed for length scaling
    sp_vals = speed[airway & (speed > 1e-6)]
    sp_mean = float(sp_vals.mean()) if sp_vals.size else 0.3
    sp_mean = max(sp_mean, 0.05)

    finished: list[np.ndarray] = []
    seed_speeds: list[float] = []
    for seed in seeds:
        # Sample seed speed for length scaling
        idx0 = _phys_to_index(
            seed.reshape(1, 3), spacing_xyz_mm, origin_xyz_mm, airway.shape
        )[0]
        sp0 = float(speed[tuple(idx0)]) if airway[tuple(idx0)] else sp_mean
        # Higher velocity → longer lines (more steps)
        length_scale = 0.55 + 1.1 * min(2.2, sp0 / sp_mean)
        steps = int(max(80, min(max_steps, max_steps * length_scale * 0.85)))

        base_kw = dict(
            ux=ux,
            uy=uy,
            uz=uz,
            airway=airway,
            spacing_xyz_mm=spacing_xyz_mm,
            origin_xyz_mm=origin_xyz_mm,
            max_steps=steps,
            step_mm=step_mm,
            use_trilinear=True,
            swirl=swirl,
            rng=rng,
            attract_mm=trachea,
            attract_strength=0.34 if trachea is not None else 0.0,
        )
        s = seed.reshape(1, 3)
        fwd = compute_streamlines(**base_kw, seed_points_mm=s, reverse=False)
        f = fwd[0] if fwd else None
        if f is None or len(f) < 6:
            continue
        line = np.asarray(f, dtype=float)

        # Orient generally toward trachea if both ends known
        if trachea is not None:
            d0 = float(np.linalg.norm(line[0] - trachea))
            d1 = float(np.linalg.norm(line[-1] - trachea))
            if d0 + 3.0 < d1:
                line = line[::-1].copy()
            # Soft finish toward trachea when close enough / progressing
            d_end = float(np.linalg.norm(line[-1] - trachea))
            d_start = float(np.linalg.norm(line[0] - trachea))
            if d_end > trachea_end_max_mm and d_end < d_start - 5.0:
                if centerline_mm is not None and len(np.asarray(centerline_mm)) >= 2:
                    line = extend_paths_to_outlet_via_centerline(
                        [line],
                        np.asarray(centerline_mm, dtype=float),
                        trachea,
                        max_end_dist_mm=trachea_end_max_mm,
                    )[0]
                elif d_end < 45.0:
                    n_extra = max(3, int(d_end / max(step_mm, 0.2)))
                    tail = np.linspace(line[-1], trachea, num=n_extra + 1)[1:]
                    line = np.vstack([line, tail])

        travel = float(np.linalg.norm(line[-1] - line[0]))
        if travel < 8.0:
            continue
        if trachea is not None:
            # Must make progress toward trachea overall
            if float(np.linalg.norm(line[-1] - trachea)) > float(
                np.linalg.norm(line[0] - trachea)
            ) - 3.0:
                # allow short mid-passage wisps if still long enough
                if travel < 25.0:
                    continue
        finished.append(line)
        seed_speeds.append(sp0)

    # Score: trachea progress + seed speed (prefer fast / long wisps)
    scored: list[tuple[float, np.ndarray]] = []
    for line, sp0 in zip(finished, seed_speeds):
        arc = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        score = arc * 0.4 + sp0 * 25.0
        if trachea is not None:
            d0 = float(np.linalg.norm(line[0] - trachea))
            d1 = float(np.linalg.norm(line[-1] - trachea))
            score += max(0.0, d0 - d1) * 5.0
            score += max(0.0, 60.0 - d1) * 2.0
        scored.append((score, line))
    scored.sort(key=lambda t: t[0], reverse=True)
    n_keep = max(1, int(max_lines))
    return [ln for _, ln in scored[:n_keep]]


def extend_paths_to_outlet_via_centerline(
    lines: list[np.ndarray],
    centerline_mm: np.ndarray,
    outlet_center_mm: np.ndarray | list[float],
    max_end_dist_mm: float = 14.0,
) -> list[np.ndarray]:
    """
    If a streamline stops short of the trachea, append the remaining centerline.

    This keeps skin→lumen CFD paths but completes the anatomical route to the
    caudal outlet when the discrete CFD field is weak near the outlet.
    """
    if centerline_mm is None or len(centerline_mm) < 2:
        return lines
    cl = np.asarray(centerline_mm, dtype=float)
    outlet = np.asarray(outlet_center_mm, dtype=float)
    out: list[np.ndarray] = []
    for line in lines:
        line = np.asarray(line, dtype=float)
        end = line[-1]
        if float(np.linalg.norm(end - outlet)) <= max_end_dist_mm:
            out.append(line)
            continue
        # nearest centerline node to current end
        d = np.linalg.norm(cl - end, axis=1)
        i0 = int(np.argmin(d))
        # walk centerline toward the end that is closer to the outlet
        d_start = float(np.linalg.norm(cl[0] - outlet))
        d_end = float(np.linalg.norm(cl[-1] - outlet))
        if d_end <= d_start:
            tail = cl[i0:]
        else:
            tail = cl[: i0 + 1][::-1]
        if len(tail) < 2:
            # direct finish
            tail = np.vstack([end, outlet])
        else:
            # ensure last point is outlet
            if float(np.linalg.norm(tail[-1] - outlet)) > 1.0:
                tail = np.vstack([tail, outlet])
        joined = np.vstack([line, tail])
        out.append(joined)
    return out


def compute_flow_field(
    airway_mask_path: Path | str,
    boundary_json_path: Path | str,
    output_dir: Path | str | None = None,
    case_id: str | None = None,
    breathing: PatientBreathing | None = None,
    pressure_iterations: int = 350,
    port_radius_mm: float = 6.0,
    n_streamline_seeds: int = 40,
) -> FlowFieldResult:
    """
    Full pipeline: mask + BC JSON → pressure, velocity, streamlines, NPZ export.
    """
    airway_mask_path = Path(airway_mask_path)
    boundary_json_path = Path(boundary_json_path)
    case_id = case_id or airway_mask_path.stem.replace("_airway_mask", "")
    output_dir = Path(output_dir or airway_mask_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    with boundary_json_path.open(encoding="utf-8") as f:
        bc = json.load(f)

    img = sitk.ReadImage(str(airway_mask_path))
    airway = sitk.GetArrayFromImage(img).astype(bool)
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())
    shape = airway.shape

    ports = {p["name"]: p for p in bc["ports"]}
    inlet_mask = np.zeros(shape, dtype=bool)
    outlet_mask = np.zeros(shape, dtype=bool)
    for name, port in ports.items():
        seed = _port_seed_mask(
            shape, port["center_mm"], spacing, origin, airway, radius_mm=port_radius_mm
        )
        if port["role"] == "inlet":
            inlet_mask |= seed
        elif port["role"] == "outlet":
            outlet_mask |= seed

    notes = [
        "Potential-flow / Darcy approximation (Laplace pressure), not full CFD.",
        "Velocity scaled to match mean inspiratory flow from physiology.",
    ]
    if bc.get("outlet_is_proxy"):
        notes.append("Outlet is nasopharynx proxy (no true trachea in FOV).")

    breathing = breathing or PatientBreathing.typical_resting_adult()
    # Prefer BC-stored flow if present
    target_q = float(
        bc.get("flow_assignment", {}).get(
            "total_inflow_L_per_min", breathing.mean_inspiratory_flow_L_per_min
        )
    )

    p = solve_pressure_potential(
        airway, inlet_mask, outlet_mask, iterations=pressure_iterations
    )
    # Finite pressure only inside airway (avoid float32 overflow on nan)
    p_out = np.where(airway, np.nan_to_num(p, nan=0.0), 0.0)

    ux, uy, uz, _speed_rel = pressure_to_velocity(p_out, airway, spacing)
    inlet_area = sum(
        float(port.get("area_mm2", 0.0))
        for port in ports.values()
        if port.get("role") == "inlet"
    )
    ux, uy, uz, speed, achieved_q = scale_velocity_to_flow_rate(
        ux,
        uy,
        uz,
        airway,
        inlet_mask,
        spacing,
        target_q,
        inlet_area_mm2=inlet_area if inlet_area > 0 else None,
    )

    # Streamline seeds: inlet face + optional centerline jitter (passage JSON)
    sx, sy, sz = spacing
    ox, oy, oz = origin
    seeds: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    zz, yy, xx = np.where(inlet_mask & airway)
    if len(zz) > 0:
        n_in = min(max(n_streamline_seeds // 2, 20), len(zz))
        pick = rng.choice(len(zz), size=n_in, replace=False)
        for i in pick:
            seeds.append(
                np.array(
                    [ox + xx[i] * sx, oy + yy[i] * sy, oz + zz[i] * sz],
                    dtype=float,
                )
            )

    passage_path = output_dir / f"{case_id}_passage.json"
    if passage_path.is_file():
        try:
            with passage_path.open(encoding="utf-8") as f:
                passage = json.load(f)
            cl = passage.get("centerline_mm") or []
            # Seed along first third of centerline (near nares) with radial jitter
            n_cl = min(max(n_streamline_seeds // 2, 20), max(len(cl) // 2, 1))
            if cl:
                step = max(len(cl) // (n_cl + 1), 1)
                for i in range(0, min(len(cl) // 2 + 1, len(cl)), step):
                    base = np.array(cl[i], dtype=float)
                    for _ in range(2):
                        jitter = rng.normal(0, 1.2, size=3)
                        seeds.append(base + jitter)
            notes.append(
                "Streamlines seeded from inlet ports and nasal-passage centerline."
            )
        except Exception as exc:
            notes.append(f"Centerline seed skip: {exc}")

    if not seeds:
        # Fallback: random lumen seeds near high pressure
        zz, yy, xx = np.where(airway)
        if len(zz):
            n_take = min(n_streamline_seeds, len(zz))
            pick = rng.choice(len(zz), size=n_take, replace=False)
            for i in pick:
                seeds.append(
                    np.array(
                        [ox + xx[i] * sx, oy + yy[i] * sy, oz + zz[i] * sz],
                        dtype=float,
                    )
                )

    seed_arr = np.array(seeds, dtype=float) if seeds else np.zeros((0, 3))
    streamlines = compute_streamlines(
        ux, uy, uz, airway, spacing, origin, seed_arr, max_steps=800, step_mm=0.4
    )

    # Save compact NPZ for the viewer
    npz_path = output_dir / f"{case_id}_flow.npz"
    # Downcast for size
    np.savez_compressed(
        npz_path,
        airway=airway.astype(np.uint8),
        pressure=p_out.astype(np.float32),
        ux=ux.astype(np.float32),
        uy=uy.astype(np.float32),
        uz=uz.astype(np.float32),
        speed=speed.astype(np.float32),
        spacing_xyz_mm=np.array(spacing, dtype=np.float64),
        origin_xyz_mm=np.array(origin, dtype=np.float64),
        inlet_mask=inlet_mask.astype(np.uint8),
        outlet_mask=outlet_mask.astype(np.uint8),
    )
    # Streamlines as separate JSON (list of polylines)
    sl_path = output_dir / f"{case_id}_streamlines.json"
    with sl_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "unit": "mm",
                "n_lines": len(streamlines),
                "lines": [line.tolist() for line in streamlines],
            },
            f,
        )

    # Also write a SimpleITK speed map for 3D Slicer / ITK-SNAP
    speed_img = sitk.GetImageFromArray(speed.astype(np.float32))
    speed_img.CopyInformation(img)
    sitk.WriteImage(speed_img, str(output_dir / f"{case_id}_speed.nrrd"))

    air_speeds = speed[airway]
    result = FlowFieldResult(
        case_id=case_id,
        spacing_xyz_mm=list(spacing),
        origin_xyz_mm=list(origin),
        size_zyx=list(shape),
        method="potential_flow_laplace_scaled",
        target_flow_L_per_min=target_q,
        achieved_inlet_flux_L_per_min=achieved_q,
        max_speed_m_s=float(air_speeds.max()) if air_speeds.size else 0.0,
        mean_speed_m_s=float(air_speeds.mean()) if air_speeds.size else 0.0,
        n_airway_voxels=int(airway.sum()),
        notes=notes,
    )
    with (output_dir / f"{case_id}_flow_meta.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"[{case_id}] flow field: max |u|={result.max_speed_m_s:.4f} m/s  "
          f"mean={result.mean_speed_m_s:.4f} m/s")
    print(f"[{case_id}] target Q={target_q:.2f} L/min  "
          f"flux proxy={achieved_q:.2f} L/min  streamlines={len(streamlines)}")
    print(f"[{case_id}] wrote {npz_path.name}, {sl_path.name}, {case_id}_speed.nrrd")
    return result


def load_flow_npz(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
