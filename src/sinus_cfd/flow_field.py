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
) -> list[np.ndarray]:
    """
    RK2 integration of streamlines from seed points (physical mm).

    Returns list of (N_i, 3) arrays of polyline vertices in mm.
    """
    sx, sy, sz = spacing_xyz_mm
    ox, oy, oz = origin_xyz_mm
    shape = airway.shape
    lines: list[np.ndarray] = []

    def sample(pos_mm: np.ndarray) -> np.ndarray | None:
        x, y, z = pos_mm
        # continuous index
        ix = (x - ox) / sx
        iy = (y - oy) / sy
        iz = (z - oz) / sz
        i0, j0, k0 = int(np.floor(iz)), int(np.floor(iy)), int(np.floor(ix))
        if not (0 <= i0 < shape[0] - 1 and 0 <= j0 < shape[1] - 1 and 0 <= k0 < shape[2] - 1):
            return None
        if not airway[i0, j0, k0]:
            return None
        # nearest-neighbor sample (fast, good enough for viz)
        return np.array([ux[i0, j0, k0], uy[i0, j0, k0], uz[i0, j0, k0]], dtype=float)

    for seed in seed_points_mm:
        pts = [seed.astype(float)]
        pos = seed.astype(float)
        for _ in range(max_steps):
            v = sample(pos)
            if v is None:
                break
            speed = np.linalg.norm(v)
            if speed < 1e-12:
                break
            direction = v / speed
            # RK2
            mid = pos + 0.5 * step_mm * direction
            v2 = sample(mid)
            if v2 is not None and np.linalg.norm(v2) > 1e-12:
                direction = v2 / np.linalg.norm(v2)
            new_pos = pos + step_mm * direction
            idx = _phys_to_index(new_pos.reshape(1, 3), spacing_xyz_mm, origin_xyz_mm, shape)[0]
            if not airway[tuple(idx)]:
                # try smaller step once
                new_pos = pos + 0.25 * step_mm * direction
                idx = _phys_to_index(new_pos.reshape(1, 3), spacing_xyz_mm, origin_xyz_mm, shape)[0]
                if not airway[tuple(idx)]:
                    break
            pos = new_pos
            pts.append(pos.copy())
        if len(pts) >= 5:
            lines.append(np.vstack(pts))
    return lines


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

    # Streamline seeds from actual inlet airway voxels (physical mm)
    sx, sy, sz = spacing
    ox, oy, oz = origin
    seeds: list[np.ndarray] = []
    zz, yy, xx = np.where(inlet_mask & airway)
    if len(zz) > 0:
        rng = np.random.default_rng(42)
        n_take = min(n_streamline_seeds, len(zz))
        pick = rng.choice(len(zz), size=n_take, replace=False)
        for i in pick:
            seeds.append(
                np.array(
                    [
                        ox + xx[i] * sx,
                        oy + yy[i] * sy,
                        oz + zz[i] * sz,
                    ],
                    dtype=float,
                )
            )
    # Slight inward offset along pressure gradient (toward outlet)
    seed_arr = np.array(seeds, dtype=float) if seeds else np.zeros((0, 3))
    streamlines = compute_streamlines(
        ux, uy, uz, airway, spacing, origin, seed_arr, max_steps=600, step_mm=0.45
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
