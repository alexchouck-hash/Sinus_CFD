"""
Boundary-condition geometry and flow assignment for nasal CFD.

Target anatomy (user intent):
  - Inlets: both nostrils (flow prescribed)
  - Outlet: trachea (pressure reference / outflow)
  - Mouth: closed / blocked (not part of fluid domain, or wall)

NasalSeg FOV typically ends at the nasopharynx — there is no true tracheal
segment. We place the outlet on the distal nasopharynx face as a *proxy*
until a full head–neck CT including trachea is used.

Port detection uses expert labels when available:
  - left_nostril  ← anterior tip of left nasal cavity (label 1)
  - right_nostril ← anterior tip of right nasal cavity (label 2)
  - trachea_outlet_proxy ← distal tip of nasopharynx (label 3)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from scipy import ndimage as ndi

from .physiology import PatientBreathing, summary_text

# Label IDs matching pipeline.LABEL_NAMES
LAB_LEFT_NASAL = 1
LAB_RIGHT_NASAL = 2
LAB_NASOPHARYNX = 3


@dataclass
class Port:
    name: str
    role: str  # "inlet" | "outlet" | "wall"
    # Physical center of the port patch (mm, same frame as mesh)
    center_mm: list[float]
    # Approximate open area from label cross-section (mm^2)
    area_mm2: float
    # Unit normal pointing *into* the fluid domain along mean flow sense
    # (inlet: inward from outside; outlet: toward trachea / out of domain)
    normal_xyz: list[float]
    # How this port was found
    method: str
    notes: str = ""
    # Face indices on the surface mesh tagged for this port (if available)
    face_indices: list[int] = field(default_factory=list)
    n_faces: int = 0


@dataclass
class BoundarySetup:
    case_id: str
    mouth: str
    inlet_names: list[str]
    outlet_name: str
    outlet_is_proxy: bool
    ports: list[Port]
    breathing: dict[str, Any]
    flow_assignment: dict[str, Any]
    mesh_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mouth": self.mouth,
            "inlet_names": self.inlet_names,
            "outlet_name": self.outlet_name,
            "outlet_is_proxy": self.outlet_is_proxy,
            "ports": [asdict(p) for p in self.ports],
            "breathing": self.breathing,
            "flow_assignment": self.flow_assignment,
            "mesh_path": self.mesh_path,
            "warnings": self.warnings,
            "boundary_policy": {
                "inlets": "both_nostrils",
                "outlet": "trachea",
                "mouth": "closed",
                "wall": "airway_mucosa_no_slip",
            },
        }


def _voxel_to_phys(
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Map voxel indices (z,y,x arrays) → physical Nx3 (x,y,z) mm."""
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    return np.column_stack(
        [
            x.astype(np.float64) * sx + ox,
            y.astype(np.float64) * sy + oy,
            z.astype(np.float64) * sz + oz,
        ]
    )


def _structure_tip(
    label_zyx: np.ndarray,
    label_id: int,
    airway_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    tip_mode: str,
    reference_centroid_phys: np.ndarray | None = None,
    percentile: float = 5.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Locate a port tip for a labeled structure.

    tip_mode:
      - "farthest_from_ref": voxels farthest from a reference point (good for nostrils
        relative to nasopharynx centroid)
      - "min_y" / "max_y" / "min_x" / "max_x" / "min_z" / "max_z": axis extremes

    Returns (center_xyz_mm, area_mm2, outward_normal_xyz).
    """
    # Prefer voxels that remain in the CFD mask; fall back to raw labels so a
    # nostril is still located if bridging/cleanup temporarily dropped it.
    region = (label_zyx == label_id) & airway_mask
    if not region.any():
        region = label_zyx == label_id
    if not region.any():
        raise ValueError(f"No voxels for label {label_id}")

    zz, yy, xx = np.where(region)
    pts = _voxel_to_phys(zz, yy, xx, spacing_xyz, origin_xyz)

    if tip_mode == "farthest_from_ref":
        if reference_centroid_phys is None:
            raise ValueError("farthest_from_ref needs reference_centroid_phys")
        dist = np.linalg.norm(pts - reference_centroid_phys, axis=1)
        thr = np.percentile(dist, 100.0 - percentile)
        tip = pts[dist >= thr]
        # Outward = away from reference (toward exterior / trachea)
        direction = tip.mean(axis=0) - reference_centroid_phys
    else:
        axis = tip_mode[-1]  # x/y/z
        col = {"x": 0, "y": 1, "z": 2}[axis]
        vals = pts[:, col]
        if tip_mode.startswith("min"):
            thr = np.percentile(vals, percentile)
            tip = pts[vals <= thr]
            direction = np.zeros(3)
            direction[col] = -1.0
        else:
            thr = np.percentile(vals, 100.0 - percentile)
            tip = pts[vals >= thr]
            direction = np.zeros(3)
            direction[col] = 1.0

    center = tip.mean(axis=0)
    # Approximate area: project tip voxels onto plane ⟂ direction
    n = direction.astype(np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        n = np.array([0.0, 1.0, 0.0])
    else:
        n = n / n_norm

    # Rough area from count of tip voxels × in-plane pixel area
    sx, sy, sz = spacing_xyz
    # Use geometric mean of the two smallest spacings as face pixel area
    face_area = float(np.prod(sorted([sx, sy, sz])[:2]))
    area_mm2 = float(len(tip) * face_area)

    return center, area_mm2, n


def _nasopharynx_centroid(
    label_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> np.ndarray:
    m = label_zyx == LAB_NASOPHARYNX
    if not m.any():
        # fall back to all airway labels 1-3
        m = np.isin(label_zyx, [1, 2, 3])
    zz, yy, xx = np.where(m)
    return _voxel_to_phys(zz, yy, xx, spacing_xyz, origin_xyz).mean(axis=0)


def detect_ports_from_labels(
    label_zyx: np.ndarray,
    airway_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> tuple[list[Port], list[str]]:
    """
    Detect left/right nostril inlets and distal nasopharynx outlet proxy.
    """
    warnings: list[str] = []
    ports: list[Port] = []

    np_centroid = _nasopharynx_centroid(label_zyx, spacing_xyz, origin_xyz)

    # Nostrils: farthest part of each nasal cavity from nasopharynx (toward face)
    for lab_id, name in (
        (LAB_LEFT_NASAL, "left_nostril"),
        (LAB_RIGHT_NASAL, "right_nostril"),
    ):
        if not (label_zyx == lab_id).any():
            warnings.append(f"Missing label for {name} (id={lab_id})")
            continue
        center, area, normal = _structure_tip(
            label_zyx,
            lab_id,
            airway_mask,
            spacing_xyz,
            origin_xyz,
            tip_mode="farthest_from_ref",
            reference_centroid_phys=np_centroid,
            percentile=8.0,
        )
        ports.append(
            Port(
                name=name,
                role="inlet",
                center_mm=center.tolist(),
                area_mm2=area,
                normal_xyz=normal.tolist(),
                method="label_tip_farthest_from_nasopharynx",
                notes="Inspiratory flow enters here (nostril).",
            )
        )

    # Outlet: nasopharynx tip farthest from mean of nasal cavities → toward trachea
    nasal = np.isin(label_zyx, [LAB_LEFT_NASAL, LAB_RIGHT_NASAL])
    if nasal.any() and (label_zyx == LAB_NASOPHARYNX).any():
        zz, yy, xx = np.where(nasal)
        nasal_centroid = _voxel_to_phys(zz, yy, xx, spacing_xyz, origin_xyz).mean(0)
        center, area, normal = _structure_tip(
            label_zyx,
            LAB_NASOPHARYNX,
            airway_mask,
            spacing_xyz,
            origin_xyz,
            tip_mode="farthest_from_ref",
            reference_centroid_phys=nasal_centroid,
            percentile=8.0,
        )
        ports.append(
            Port(
                name="trachea_outlet_proxy",
                role="outlet",
                center_mm=center.tolist(),
                area_mm2=area,
                normal_xyz=normal.tolist(),
                method="nasopharynx_tip_farthest_from_nasal",
                notes=(
                    "NasalSeg does not include the trachea. This is the distal "
                    "nasopharynx face used as an outlet proxy until a full "
                    "head–neck CT with trachea is available."
                ),
            )
        )
        warnings.append(
            "Outlet is nasopharynx proxy (no trachea in NasalSeg FOV)."
        )
    else:
        warnings.append("Could not place trachea outlet proxy from labels.")

    return ports, warnings


def tag_mesh_faces_near_ports(
    mesh: trimesh.Trimesh,
    ports: list[Port],
    radius_mm: float = 8.0,
) -> list[Port]:
    """
    Assign surface faces whose centroids fall within radius_mm of a port center.

    Overlapping claims: nearest port wins. Remaining faces are walls.
    """
    if len(mesh.faces) == 0:
        return ports

    face_centers = mesh.triangles_center  # (F, 3)
    claimed = np.full(len(face_centers), -1, dtype=int)
    claim_dist = np.full(len(face_centers), np.inf)

    port_centers = np.array([p.center_mm for p in ports], dtype=np.float64)
    for pi, c in enumerate(port_centers):
        d = np.linalg.norm(face_centers - c, axis=1)
        nearer = d < claim_dist
        in_radius = d <= radius_mm
        take = nearer & in_radius
        claimed[take] = pi
        claim_dist[take] = d[take]

    updated: list[Port] = []
    for pi, port in enumerate(ports):
        idxs = np.where(claimed == pi)[0].astype(int).tolist()
        updated.append(
            Port(
                name=port.name,
                role=port.role,
                center_mm=port.center_mm,
                area_mm2=port.area_mm2,
                normal_xyz=port.normal_xyz,
                method=port.method,
                notes=port.notes,
                face_indices=idxs,
                n_faces=len(idxs),
            )
        )
    return updated


def assign_flow(
    ports: list[Port],
    breathing: PatientBreathing,
) -> dict[str, Any]:
    """
    Map physiology → per-port volumetric flow.

    Inlets share total mean inspiratory flow by configured fractions.
    Outlet is a zero-gauge (or fixed pressure) boundary — net mass conserved.
    """
    split = breathing.flow_split_L_per_min()
    q_total = split["total"]
    q_m3s = breathing.mean_inspiratory_flow_m3_s

    per_port: dict[str, Any] = {}
    for p in ports:
        if p.role == "inlet" and p.name == "left_nostril":
            q_lpm = split["left_nostril"]
            per_port[p.name] = _inlet_entry(p, q_lpm, q_m3s * breathing.left_nostril_flow_fraction)
        elif p.role == "inlet" and p.name == "right_nostril":
            q_lpm = split["right_nostril"]
            per_port[p.name] = _inlet_entry(p, q_lpm, q_m3s * breathing.right_nostril_flow_fraction)
        elif p.role == "outlet":
            area_m2 = max(p.area_mm2, 1.0) * 1e-6
            # Characteristic outflow speed if uniform over proxy area
            u = q_m3s / area_m2
            per_port[p.name] = {
                "role": "outlet",
                "type": "fixed_pressure_or_outflow",
                "gauge_pressure_Pa": 0.0,
                "expected_total_outflow_L_per_min": q_total,
                "expected_bulk_speed_m_s_if_uniform": u,
                "area_mm2": p.area_mm2,
                "notes": p.notes,
            }
        else:
            per_port[p.name] = {"role": p.role, "notes": p.notes}

    return {
        "mode": "quasi_steady_inspiration",
        "duration_s": breathing.Ti_s,
        "total_inflow_L_per_min": q_total,
        "total_inflow_m3_s": q_m3s,
        "mouth": "closed",
        "per_port": per_port,
        "fluid": {
            "name": "air",
            "density_kg_m3": breathing.density_kg_m3,
            "dynamic_viscosity_Pa_s": breathing.dynamic_viscosity_Pa_s,
            "temperature_C_nominal": 37.0,
        },
        "wall": {
            "type": "no_slip",
            "includes": ["airway_mucosa", "closed_mouth", "sealed_sinus_ostia_if_excluded"],
        },
    }


def _inlet_entry(port: Port, q_lpm: float, q_m3s: float) -> dict[str, Any]:
    area_m2 = max(port.area_mm2, 1.0) * 1e-6
    u = q_m3s / area_m2
    return {
        "role": "inlet",
        "type": "volumetric_flow_or_normal_velocity",
        "flow_L_per_min": q_lpm,
        "flow_m3_s": q_m3s,
        "area_mm2_estimate": port.area_mm2,
        "bulk_normal_speed_m_s": u,
        "direction": "into_domain_along_port_normal",
        "normal_xyz": port.normal_xyz,
    }


def build_boundary_setup(
    case_id: str,
    label_zyx: np.ndarray | None,
    airway_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    mesh: trimesh.Trimesh | None,
    breathing: PatientBreathing | None = None,
    mesh_path: str | Path | None = None,
    face_tag_radius_mm: float = 8.0,
) -> BoundarySetup:
    breathing = breathing or PatientBreathing.typical_resting_adult()
    warnings: list[str] = []

    if label_zyx is None:
        raise ValueError(
            "Label map required to place nostrils / trachea proxy. "
            "Use NasalSeg labels or provide equivalent segmentation."
        )

    ports, w = detect_ports_from_labels(
        label_zyx, airway_mask, spacing_xyz, origin_xyz
    )
    warnings.extend(w)

    if mesh is not None and ports:
        ports = tag_mesh_faces_near_ports(mesh, ports, radius_mm=face_tag_radius_mm)
        for p in ports:
            if p.n_faces == 0:
                warnings.append(
                    f"No mesh faces tagged for {p.name} within {face_tag_radius_mm} mm; "
                    "increase radius or check coordinates."
                )

    flow = assign_flow(ports, breathing)
    inlets = [p.name for p in ports if p.role == "inlet"]
    outlets = [p.name for p in ports if p.role == "outlet"]
    outlet_name = outlets[0] if outlets else "trachea_outlet_proxy"

    return BoundarySetup(
        case_id=case_id,
        mouth="closed — oral cavity excluded from fluid domain (labels 1–3 only; not meshed as an opening)",
        inlet_names=inlets,
        outlet_name=outlet_name,
        outlet_is_proxy=any("proxy" in p.name for p in ports if p.role == "outlet"),
        ports=ports,
        breathing=breathing.to_dict(),
        flow_assignment=flow,
        mesh_path=str(mesh_path) if mesh_path else None,
        warnings=warnings,
    )


def write_boundary_setup(setup: BoundarySetup, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(setup.to_dict(), f, indent=2)
    return path


def write_openfoam_bc_notes(setup: BoundarySetup, path: Path | str) -> Path:
    """
    Human-readable OpenFOAM-oriented BC sketch (not a full case).

    Velocity inlet uses bulk normal speed from estimated port area; replace with
    flowRateInletVelocity once patch areas are exact after meshing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = setup.breathing
    fa = setup.flow_assignment
    lines = [
        "/* Sinus_CFD - boundary condition sketch (OpenFOAM-oriented) */",
        f"// case: {setup.case_id}",
        f"// mode: quasi-steady inspiration for Ti = {b['Ti_s_effective']:.3f} s",
        f"// total Q = {fa['total_inflow_L_per_min']:.3f} L/min "
        f"= {fa['total_inflow_m3_s']:.6e} m^3/s",
        f"// mouth: CLOSED (wall)",
        f"// outlet: {setup.outlet_name}"
        + (" (PROXY - not true trachea)" if setup.outlet_is_proxy else ""),
        "",
        summary_text(PatientBreathing(
            tidal_volume_L=b["tidal_volume_L"],
            respiratory_rate_per_min=b["respiratory_rate_per_min"],
            inspiratory_fraction=b["inspiratory_fraction"],
            inspiratory_time_s=b.get("inspiratory_time_s"),
            left_nostril_flow_fraction=b["left_nostril_flow_fraction"],
            right_nostril_flow_fraction=b["right_nostril_flow_fraction"],
        )),
        "",
        "// Suggested patches after surface meshing:",
        "//   left_nostril, right_nostril  -> inlet",
        f"//   {setup.outlet_name}        -> outlet (p=0 gauge)",
        "//   wall_airway, wall_mouth     -> noSlip",
        "",
        "/* 0/U (sketch) */",
        "boundaryField",
        "{",
        "    left_nostril",
        "    {",
        "        type            flowRateInletVelocity;",
        f"        volumetricFlowRate constant {fa['per_port'].get('left_nostril', {}).get('flow_m3_s', 0):.6e};",
        "        // value is m^3/s; direction = patch normal into domain",
        "        value           uniform (0 0 0);",
        "    }",
        "    right_nostril",
        "    {",
        "        type            flowRateInletVelocity;",
        f"        volumetricFlowRate constant {fa['per_port'].get('right_nostril', {}).get('flow_m3_s', 0):.6e};",
        "        value           uniform (0 0 0);",
        "    }",
        f"    {setup.outlet_name}",
        "    {",
        "        type            pressureInletOutletVelocity;",
        "        value           uniform (0 0 0);",
        "    }",
        "    wall_airway",
        "    {",
        "        type            noSlip;",
        "    }",
        "    wall_mouth",
        "    {",
        "        type            noSlip;  // mouth closed / blocked",
        "    }",
        "}",
        "",
        "/* 0/p (sketch) */",
        "boundaryField",
        "{",
        "    left_nostril",
        "    {",
        "        type            zeroGradient;",
        "    }",
        "    right_nostril",
        "    {",
        "        type            zeroGradient;",
        "    }",
        f"    {setup.outlet_name}",
        "    {",
        "        type            fixedValue;",
        "        value           uniform 0;  // gauge Pa",
        "    }",
        "    \".*wall.*\"",
        "    {",
        "        type            zeroGradient;",
        "    }",
        "}",
        "",
    ]
    if setup.warnings:
        lines.append("// warnings:")
        for w in setup.warnings:
            lines.append(f"//  - {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_port_markers_ply(
    ports: Iterable[Port],
    path: Path | str,
    radius_mm: float = 2.0,
) -> Path:
    """Write small icosphere markers at port centers for visual QC in MeshLab."""
    path = Path(path)
    geoms = []
    # Distinct-ish colors via face colors not needed if separate; merge with vertex colors
    for p in ports:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius_mm)
        sphere.apply_translation(p.center_mm)
        geoms.append(sphere)
    if not geoms:
        raise ValueError("No ports to export")
    scene = trimesh.util.concatenate(geoms)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path)
    return path
