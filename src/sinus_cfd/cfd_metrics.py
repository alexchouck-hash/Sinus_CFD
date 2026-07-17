"""
CFD summary metrics from imported velocity/pressure fields.

Reports:
  - Mean pressure at nares (inlets) vs trachea (outlet) → ΔP
  - Nasal resistance R = ΔP / Q  (using target or estimated Q)
  - Approximate L/R inlet volume-flux split from mapped U near each naris
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


def _sphere_mask(
    shape: tuple[int, int, int],
    center_mm: list[float] | np.ndarray,
    radius_mm: float,
    spacing: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    nz, ny, nx = shape
    zz, yy, xx = np.ogrid[0:nz, 0:ny, 0:nx]
    x = ox + xx * sx
    y = oy + yy * sy
    z = oz + zz * sz
    cx, cy, cz = float(center_mm[0]), float(center_mm[1]), float(center_mm[2])
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return d2 <= radius_mm**2


def _port_centers(case_dir: Path, case_id: str) -> tuple[list[list[float]], list[float] | None, list[str]]:
    notes: list[str] = []
    inlets: list[list[float]] = []
    outlet: list[float] | None = None

    nares = case_dir / f"{case_id}_nares.json"
    if nares.is_file():
        nj = json.loads(nares.read_text(encoding="utf-8"))
        for npnt in nj.get("naris_points") or []:
            if npnt.get("center_mm"):
                inlets.append([float(v) for v in npnt["center_mm"]])
        if inlets:
            notes.append(f"Inlet centers from nares.json ({len(inlets)})")

    bc_path = case_dir / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports") or []:
            c = port.get("center_mm")
            if not c:
                continue
            if port.get("role") == "outlet":
                outlet = [float(v) for v in c]
            elif port.get("role") == "inlet" and len(inlets) < 2:
                inlets.append([float(v) for v in c])
        if outlet:
            notes.append("Outlet center from boundary_conditions.json")

    # Passage outlet_open centroid overrides if present
    o_path = case_dir / f"{case_id}_passage_outlet_open.nrrd"
    if o_path.is_file():
        img = sitk.ReadImage(str(o_path))
        m = sitk.GetArrayFromImage(img).astype(bool)
        zz, yy, xx = np.where(m)
        if len(zz):
            sp = img.GetSpacing()
            org = img.GetOrigin()
            outlet = [
                float(org[0] + xx.mean() * sp[0]),
                float(org[1] + yy.mean() * sp[1]),
                float(org[2] + zz.mean() * sp[2]),
            ]
            notes.append("Outlet from passage_outlet_open centroid")

    # Lumen inlet_open L/R split by x
    i_path = case_dir / f"{case_id}_passage_inlet_open.nrrd"
    if i_path.is_file() and len(inlets) < 2:
        img = sitk.ReadImage(str(i_path))
        m = sitk.GetArrayFromImage(img).astype(bool)
        zz, yy, xx = np.where(m)
        if len(xx):
            sp = img.GetSpacing()
            org = img.GetOrigin()
            xmid = float(np.median(xx))
            for side, sel in (
                ("left", xx >= xmid),
                ("right", xx < xmid),
            ):
                if not sel.any():
                    continue
                inlets.append(
                    [
                        float(org[0] + xx[sel].mean() * sp[0]),
                        float(org[1] + yy[sel].mean() * sp[1]),
                        float(org[2] + zz[sel].mean() * sp[2]),
                    ]
                )
            notes.append("Inlets from passage_inlet_open L/R centroids")

    return inlets, outlet, notes


def _mean_pressure(
    pressure: np.ndarray,
    airway: np.ndarray,
    mask: np.ndarray,
    min_voxels: int = 5,
    require_mapped: bool = True,
) -> tuple[float | None, int]:
    region = airway & mask & np.isfinite(pressure)
    if require_mapped:
        # Prefer voxels with non-zero pressure (actually mapped from foam)
        mapped = region & (np.abs(pressure) > 1e-9)
        if int(mapped.sum()) >= min_voxels:
            vals = pressure[mapped]
            return float(np.mean(vals)), int(mapped.sum())
    n = int(region.sum())
    if n < min_voxels:
        return None, n
    vals = pressure[region]
    nz = vals[np.abs(vals) > 1e-9]
    if len(nz) >= max(3, min_voxels // 2):
        return float(np.mean(nz)), int(len(nz))
    return float(np.mean(vals)), n


def _inlet_flux_estimate(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    airway: np.ndarray,
    center_mm: list[float],
    spacing: np.ndarray,
    origin: np.ndarray,
    radius_mm: float = 8.0,
) -> dict[str, Any]:
    """
    Approximate volumetric flux through a spherical probe at the naris.

    Uses mean |U| * (π r_eff²) with r_eff from airway voxels in the probe —
    order-of-magnitude, not a mesh-face flux.
    """
    sph = _sphere_mask(airway.shape, center_mm, radius_mm, spacing, origin)
    region = airway & sph
    n = int(region.sum())
    if n == 0:
        return {
            "center_mm": list(map(float, center_mm)),
            "n_voxels": 0,
            "mean_speed_m_s": 0.0,
            "flux_m3_s_approx": 0.0,
            "flux_L_per_min_approx": 0.0,
        }
    speed = np.sqrt(ux[region] ** 2 + uy[region] ** 2 + uz[region] ** 2)
    mean_u = float(np.mean(speed))
    # Effective open area ≈ n_voxels * face_area of one voxel (approx sx*sy)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    # Use geometric mean of two smallest voxel face areas as aperture scale
    faces = sorted([sx * sy, sx * sz, sy * sz])
    area_m2 = (faces[0] * 1e-6) * max(n, 1) ** (2.0 / 3.0)  # rough
    # Better: π * r_edt^2 style using count
    r_eff_mm = (3.0 * n * sx * sy * sz / (4.0 * np.pi)) ** (1.0 / 3.0)
    area_m2 = float(np.pi * (r_eff_mm * 1e-3) ** 2)
    # Directed flux ≈ mean speed * area (inspiration into nose)
    q_m3_s = mean_u * area_m2
    q_lpm = q_m3_s * 1000.0 * 60.0
    return {
        "center_mm": list(map(float, center_mm)),
        "n_voxels": n,
        "mean_speed_m_s": mean_u,
        "effective_radius_mm": float(r_eff_mm),
        "area_m2_approx": area_m2,
        "flux_m3_s_approx": float(q_m3_s),
        "flux_L_per_min_approx": float(q_lpm),
        "notes": [
            "Flux is a CT-grid probe estimate (not OpenFOAM patch integral).",
        ],
    }


def compute_cfd_metrics(
    case_dir: Path | str,
    case_id: str,
    port_radius_mm: float = 10.0,
) -> dict[str, Any]:
    case_dir = Path(case_dir)
    npz_path = case_dir / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing flow field: {npz_path}")

    data = np.load(npz_path)
    airway = data["airway"].astype(bool)
    ux = data["ux"].astype(np.float32)
    uy = data["uy"].astype(np.float32)
    uz = data["uz"].astype(np.float32)
    pressure = data["pressure"].astype(np.float32)
    spacing = data["spacing_xyz_mm"].astype(float)
    origin = data["origin_xyz_mm"].astype(float)
    inlet_mask = data["inlet_mask"].astype(bool) if "inlet_mask" in data.files else None
    outlet_mask = data["outlet_mask"].astype(bool) if "outlet_mask" in data.files else None

    meta: dict[str, Any] = {}
    meta_path = case_dir / f"{case_id}_flow_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    inlets, outlet, port_notes = _port_centers(case_dir, case_id)

    # Order inlets: low x = right, high x = left (VH LPS-ish)
    left_c = right_c = None
    if len(inlets) >= 2:
        arr = sorted(inlets, key=lambda c: c[0])
        right_c, left_c = arr[0], arr[-1]
    elif len(inlets) == 1:
        left_c = inlets[0]

    notes = list(port_notes)
    method = str(meta.get("method", "unknown"))
    target_q = float(meta.get("target_flow_L_per_min") or 18.0)

    # Pressures
    p_in_vals: list[float] = []
    p_in_n = 0
    for c in (left_c, right_c):
        if c is None:
            continue
        m = _sphere_mask(airway.shape, c, port_radius_mm, spacing, origin)
        pv, n = _mean_pressure(pressure, airway, m)
        if pv is not None:
            p_in_vals.append(pv)
            p_in_n += n
    if not p_in_vals and inlet_mask is not None and inlet_mask.any():
        pv, n = _mean_pressure(pressure, airway, inlet_mask)
        if pv is not None:
            p_in_vals.append(pv)
            p_in_n = n
            notes.append("Inlet pressure from flow.npz inlet_mask")

    p_out = None
    p_out_n = 0
    if outlet is not None:
        m = _sphere_mask(airway.shape, outlet, port_radius_mm * 1.2, spacing, origin)
        p_out, p_out_n = _mean_pressure(pressure, airway, m)
    if p_out is None and outlet_mask is not None and outlet_mask.any():
        p_out, p_out_n = _mean_pressure(pressure, airway, outlet_mask)
        notes.append("Outlet pressure from flow.npz outlet_mask")

    # Fallback bands when port spheres miss mapped pressure
    if (not p_in_vals or p_out is None) and airway.any():
        zz, yy, xx = np.where(airway)
        y_ant = float(np.percentile(yy, 20))  # low y = anterior on VH
        y_post = float(np.percentile(yy, 85))
        if not p_in_vals:
            mask_ant = np.zeros_like(airway)
            mask_ant[zz[yy <= y_ant], yy[yy <= y_ant], xx[yy <= y_ant]] = True
            pv, n = _mean_pressure(pressure, airway, mask_ant, min_voxels=3)
            if pv is not None:
                p_in_vals.append(pv)
                p_in_n = n
                notes.append("Inlet pressure fallback: anterior airway band")
        if p_out is None:
            mask_c = np.zeros_like(airway)
            mask_c[zz[yy >= y_post], yy[yy >= y_post], xx[yy >= y_post]] = True
            pv, n = _mean_pressure(pressure, airway, mask_c, min_voxels=3)
            if pv is not None:
                p_out, p_out_n = pv, n
                notes.append("Outlet pressure fallback: posterior airway band")

    # If nares spheres sit outside the foam map (common at skin tip), use
    # anterior vs posterior *mapped* pressure bands.
    mapped_p = airway & (np.abs(pressure) > 1e-9)
    if mapped_p.any() and (not p_in_vals or p_in_vals[0] == 0.0 or p_out is None):
        zz, yy, xx = np.where(mapped_p)
        y_ant_cut = float(np.percentile(yy, 30))
        y_post_cut = float(np.percentile(yy, 75))
        ant = mapped_p.copy()
        ant[:, :, :] = False
        sel_a = yy <= y_ant_cut
        ant[zz[sel_a], yy[sel_a], xx[sel_a]] = True
        post = mapped_p.copy()
        post[:, :, :] = False
        sel_p = yy >= y_post_cut
        post[zz[sel_p], yy[sel_p], xx[sel_p]] = True
        if not p_in_vals or float(np.mean(p_in_vals)) == 0.0:
            pv, n = _mean_pressure(pressure, airway, ant, min_voxels=3)
            if pv is not None:
                p_in_vals = [pv]
                p_in_n = n
                notes.append(
                    "Inlet pressure from anterior mapped-p band (naris spheres unmapped)."
                )
        if p_out is None or (p_out is not None and abs(p_out) < 1e-12):
            pv, n = _mean_pressure(pressure, airway, post, min_voxels=3)
            if pv is not None:
                p_out, p_out_n = pv, n
                notes.append("Outlet pressure from posterior mapped-p band.")

    p_inlet = float(np.mean(p_in_vals)) if p_in_vals else None
    delta_p = None
    delta_p_abs = None
    if p_inlet is not None and p_out is not None:
        # Incompressible simpleFoam: p is kinematic (m²/s²) or Pa depending case;
        # report as stored field units with note.
        delta_p = float(p_inlet - p_out)
        delta_p_abs = float(abs(delta_p))

    # L/R flux probes
    left_flux = (
        _inlet_flux_estimate(ux, uy, uz, airway, left_c, spacing, origin)
        if left_c is not None
        else None
    )
    right_flux = (
        _inlet_flux_estimate(ux, uy, uz, airway, right_c, spacing, origin)
        if right_c is not None
        else None
    )

    q_l = float((left_flux or {}).get("flux_L_per_min_approx") or 0.0)
    q_r = float((right_flux or {}).get("flux_L_per_min_approx") or 0.0)
    q_sum = q_l + q_r
    if q_sum > 1e-9:
        frac_l = q_l / q_sum
        frac_r = q_r / q_sum
        # Scale probe fluxes to target total Q for reporting allocation
        q_l_scaled = frac_l * target_q
        q_r_scaled = frac_r * target_q
    else:
        frac_l = frac_r = 0.5
        q_l_scaled = q_r_scaled = 0.5 * target_q
        notes.append("Flux probes empty — assumed 50/50 split for scaled Q")

    # Resistance using target Q (more stable than probe Q)
    q_m3_s = target_q / (1000.0 * 60.0)
    resistance = None
    resistance_abs = None
    if delta_p is not None and q_m3_s > 0:
        resistance = float(delta_p / q_m3_s)
        resistance_abs = float(delta_p_abs / q_m3_s) if delta_p_abs is not None else abs(resistance)

    # Speed stats
    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    mapped = airway & (speed > 1e-6)

    report: dict[str, Any] = {
        "case_id": case_id,
        "kind": "cfd_metrics",
        "method": method,
        "openfoam_time": meta.get("openfoam_time"),
        "target_flow_L_per_min": target_q,
        "pressure": {
            "inlet_mean": p_inlet,
            "outlet_mean": p_out,
            "delta_p_inlet_minus_outlet": delta_p,
            "delta_p_abs": delta_p_abs,
            "inlet_sample_voxels": p_in_n,
            "outlet_sample_voxels": p_out_n,
            "port_radius_mm": port_radius_mm,
            "units_note": (
                "Field as stored in flow.npz (OpenFOAM simpleFoam p is often "
                "kinematic pressure m^2/s^2 = Pa / rho). Sign depends on BC "
                "setup; use delta_p_abs and R_abs for magnitude comparisons."
            ),
        },
        "resistance": {
            "R_delta_p_over_Q": resistance,
            "R_abs": resistance_abs,
            "Q_used_L_per_min": target_q,
            "Q_used_m3_s": q_m3_s,
            "definition": "R = (p_inlet - p_outlet) / Q_target; R_abs = |ΔP|/Q",
        },
        "inlet_allocation": {
            "left_fraction_from_probe": frac_l,
            "right_fraction_from_probe": frac_r,
            "left_Q_L_per_min_scaled_to_target": q_l_scaled,
            "right_Q_L_per_min_scaled_to_target": q_r_scaled,
            "left_probe": left_flux,
            "right_probe": right_flux,
            "target_split": "nominally 50/50 inspiration",
        },
        "speed": {
            "max_m_s": float(speed[mapped].max()) if mapped.any() else 0.0,
            "mean_m_s": float(speed[mapped].mean()) if mapped.any() else 0.0,
            "n_mapped_voxels": int(mapped.sum()),
        },
        "ports": {
            "left_naris_mm": left_c,
            "right_naris_mm": right_c,
            "outlet_mm": outlet,
        },
        "notes": notes
        + [
            "Research metrics from CT-mapped foam/potential fields — not validated clinical numbers.",
            "L/R fractions use local |U| probes; absolute probe flux is uncalibrated.",
        ],
    }
    return report


def write_cfd_metrics(
    case_dir: Path | str,
    case_id: str,
    report: dict[str, Any] | None = None,
    port_radius_mm: float = 10.0,
) -> Path:
    case_dir = Path(case_dir)
    if report is None:
        report = compute_cfd_metrics(case_dir, case_id, port_radius_mm=port_radius_mm)
    out = case_dir / f"{case_id}_cfd_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
