"""
Export CFD geometry for OpenFOAM (or similar): solid air body + open-port STLs.

OpenFOAM needs a clear fluid domain boundary split into named patches:

  wall              — mucosa (no-slip)
  left_nostril      — open inlet
  right_nostril     — open inlet
  trachea           — open outlet

The *solid body of air* is the connected lumen (nasal passage + attached sinuses
if included): air exists *inside* that solid; walls are its surface except at
open ports, which are flat caps where flow enters/leaves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure, morphology

from .pipeline import _mask_to_mesh


@dataclass
class OpenFoamExportResult:
    case_id: str
    out_dir: str
    solid_air_volume_ml: float
    patches: dict[str, str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target)
    except Exception:
        idx = np.linspace(0, len(mesh.faces) - 1, target, dtype=int)
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)


def _planar_cap_mesh(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    n_ring: int = 48,
) -> trimesh.Trimesh | None:
    """
    Build a flat disk-like patch at an open-port region.

    Uses PCA of port voxels: center + normal + radius from in-plane spread.
    """
    pts_idx = np.column_stack(np.where(mask))
    if len(pts_idx) < 5:
        return None
    sx, sy, sz = spacing
    ox, oy, oz = origin
    # physical points (x,y,z)
    pts = np.column_stack(
        [
            ox + pts_idx[:, 2] * sx,
            oy + pts_idx[:, 1] * sy,
            oz + pts_idx[:, 0] * sz,
        ]
    ).astype(float)
    center = pts.mean(axis=0)
    # PCA for plane normal
    centered = pts - center
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
    except Exception:
        normal = np.array([0.0, 1.0, 0.0])
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # orthonormal basis in plane
    tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, tmp)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(normal, e1)
    # radius from projected points
    u = centered @ e1
    v = centered @ e2
    radius = float(np.sqrt(np.max(u * u + v * v))) * 1.05
    radius = max(radius, 2.0)

    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring = np.array(
        [center + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]
    )
    verts = np.vstack([center, ring])
    faces = []
    for i in range(n_ring):
        j = 1 + i
        k = 1 + ((i + 1) % n_ring)
        faces.append([0, j, k])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    # Ensure consistent normal (outward-ish: flip if needed later by user)
    return mesh


def _surface_from_mask_region(
    mask: np.ndarray,
    parent_lumen: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> trimesh.Trimesh | None:
    """
    Extract surface triangles of the fluid domain that lie on open-port voxels.

    More accurate than a pure disk when the opening is irregular.
    """
    if not mask.any():
        return None
    # Marching cubes on lumen; keep faces whose centers are near open mask
    try:
        mesh = _mask_to_mesh(parent_lumen, spacing, origin)
    except Exception:
        return _planar_cap_mesh(mask, spacing, origin)

    sx, sy, sz = spacing
    ox, oy, oz = origin
    # face centers → nearest voxel
    centers = mesh.triangles_center
    # map to indices
    ix = np.clip(np.rint((centers[:, 0] - ox) / sx).astype(int), 0, mask.shape[2] - 1)
    iy = np.clip(np.rint((centers[:, 1] - oy) / sy).astype(int), 0, mask.shape[1] - 1)
    iz = np.clip(np.rint((centers[:, 2] - oz) / sz).astype(int), 0, mask.shape[0] - 1)
    # dilate open mask so faces near port are selected
    open_d = morphology.dilation(mask, footprint=morphology.ball(2))
    keep = open_d[iz, iy, ix]
    if not np.any(keep):
        return _planar_cap_mesh(mask, spacing, origin)
    faces = mesh.faces[keep]
    # reindex
    used = np.unique(faces.ravel())
    remap = -np.ones(len(mesh.vertices), dtype=int)
    remap[used] = np.arange(len(used))
    new_faces = remap[faces]
    new_verts = mesh.vertices[used]
    patch = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
    if len(patch.faces) < 3:
        return _planar_cap_mesh(mask, spacing, origin)
    return patch


def build_solid_air_body(
    passage_lumen: np.ndarray,
    all_interior_air: np.ndarray | None,
    include_sinuses: bool,
) -> np.ndarray:
    """
    Solid body of air = connected domain for CFD interior.

    If include_sinuses, merge all interior air that touches the main passage
    (maxillary, ethmoid, etc. when connected).
    """
    solid = passage_lumen.astype(bool)
    if include_sinuses and all_interior_air is not None:
        extra = all_interior_air.astype(bool)
        # Keep only air components that touch the passage lumen
        lab, n = ndi.label(extra)
        keep = np.zeros(n + 1, dtype=bool)
        for i in range(1, n + 1):
            comp = lab == i
            if (comp & solid).any():
                keep[i] = True
        solid = keep[lab] | solid
        # Single largest component containing passage
        lab2, n2 = ndi.label(solid)
        if n2 > 1:
            # pick component that maximizes overlap with original passage
            best_i, best_o = 1, -1
            for i in range(1, n2 + 1):
                o = int(((lab2 == i) & passage_lumen).sum())
                if o > best_o:
                    best_o = o
                    best_i = i
            solid = lab2 == best_i
    return solid.astype(bool)


def export_openfoam_geometry(
    case_id: str,
    output_dir: Path | str,
    lumen: np.ndarray,
    inlet_open: np.ndarray,
    outlet_open: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    port_names_inlet: list[str] | None = None,
    port_name_outlet: str = "trachea",
    all_interior_air: np.ndarray | None = None,
    include_sinuses: bool = True,
    reference_image: sitk.Image | None = None,
    left_inlet_mask: np.ndarray | None = None,
    right_inlet_mask: np.ndarray | None = None,
) -> OpenFoamExportResult:
    """
    Write solid air body + named open-port STLs + wall STL + OpenFOAM notes.
    """
    output_dir = Path(output_dir)
    geom_dir = output_dir / "openfoam_geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    solid = build_solid_air_body(lumen, all_interior_air, include_sinuses)
    sp_vol = float(np.prod(spacing))
    vol_ml = float(solid.sum() * sp_vol / 1000.0)
    notes.append(
        f"Solid air body volume = {vol_ml:.2f} mL "
        f"({'passage + connected sinuses' if include_sinuses else 'passage only'})."
    )

    # Save solid air mask
    solid_img = sitk.GetImageFromArray(solid.astype(np.uint8))
    solid_img.SetSpacing(spacing)
    solid_img.SetOrigin(origin)
    if reference_image is not None:
        solid_img.SetDirection(reference_image.GetDirection())
    solid_mask_path = geom_dir / f"{case_id}_solid_air_body.nrrd"
    sitk.WriteImage(solid_img, str(solid_mask_path))

    # Solid air surface (closed envelope of the fluid domain)
    solid_mesh = _decimate(_mask_to_mesh(solid, spacing, origin), 40000)
    solid_stl = geom_dir / f"{case_id}_solid_air_body.stl"
    solid_mesh.export(solid_stl)
    notes.append(
        "solid_air_body.stl = outer surface of the air solid (nasal airway ± sinuses). "
        "Air is modeled *inside* this body."
    )

    # Split inlets L/R if separate masks not provided: use x-median of inlet_open
    if left_inlet_mask is None or right_inlet_mask is None:
        zz, yy, xx = np.where(inlet_open)
        if len(xx) > 0:
            xmed = float(np.median(xx))
            left_inlet_mask = inlet_open & (
                np.arange(inlet_open.shape[2])[None, None, :] >= xmed
            )
            right_inlet_mask = inlet_open & (
                np.arange(inlet_open.shape[2])[None, None, :] < xmed
            )
        else:
            left_inlet_mask = inlet_open.copy()
            right_inlet_mask = np.zeros_like(inlet_open)

    names_in = port_names_inlet or ["left_nostril", "right_nostril"]
    patches: dict[str, str] = {}

    # Open-port patches
    for name, m in (
        (names_in[0], left_inlet_mask),
        (names_in[1] if len(names_in) > 1 else "right_nostril", right_inlet_mask),
    ):
        patch = _surface_from_mask_region(m, solid, spacing, origin)
        if patch is None:
            patch = _planar_cap_mesh(m, spacing, origin)
        if patch is not None and len(patch.faces) > 0:
            p = geom_dir / f"{case_id}_patch_{name}.stl"
            patch.export(p)
            patches[name] = p.name
            notes.append(f"Open inlet patch: {p.name} ({len(patch.faces)} faces)")

    out_patch = _surface_from_mask_region(outlet_open, solid, spacing, origin)
    if out_patch is None:
        out_patch = _planar_cap_mesh(outlet_open, spacing, origin)
    if out_patch is not None and len(out_patch.faces) > 0:
        p = geom_dir / f"{case_id}_patch_{port_name_outlet}.stl"
        out_patch.export(p)
        patches[port_name_outlet] = p.name
        notes.append(f"Open outlet patch: {p.name} ({len(out_patch.faces)} faces)")

    # Wall = solid surface minus open-port neighborhoods
    # Approximate: mesh solid, remove faces near open masks
    wall_open = left_inlet_mask | right_inlet_mask | outlet_open
    wall_open = morphology.dilation(wall_open, footprint=morphology.ball(2))
    try:
        full = _mask_to_mesh(solid, spacing, origin)
        centers = full.triangles_center
        sx, sy, sz = spacing
        ox, oy, oz = origin
        ix = np.clip(np.rint((centers[:, 0] - ox) / sx).astype(int), 0, solid.shape[2] - 1)
        iy = np.clip(np.rint((centers[:, 1] - oy) / sy).astype(int), 0, solid.shape[1] - 1)
        iz = np.clip(np.rint((centers[:, 2] - oz) / sz).astype(int), 0, solid.shape[0] - 1)
        keep = ~wall_open[iz, iy, ix]
        faces = full.faces[keep]
        used = np.unique(faces.ravel())
        remap = -np.ones(len(full.vertices), dtype=int)
        remap[used] = np.arange(len(used))
        wall_mesh = trimesh.Trimesh(
            vertices=full.vertices[used], faces=remap[faces], process=True
        )
        wall_mesh = _decimate(wall_mesh, 35000)
        p = geom_dir / f"{case_id}_patch_wall.stl"
        wall_mesh.export(p)
        patches["wall"] = p.name
        notes.append(f"Wall patch: {p.name} ({len(wall_mesh.faces)} faces, mucosa no-slip)")
    except Exception as exc:
        notes.append(f"Wall patch failed: {exc}")

    # Combined multi-body scene note file for snappyHexMesh
    readme = geom_dir / "README_OPENFOAM.txt"
    readme.write_text(
        _openfoam_readme_text(case_id, patches, vol_ml, include_sinuses),
        encoding="utf-8",
    )

    # JSON manifest
    manifest = {
        "case_id": case_id,
        "solid_air_volume_ml": vol_ml,
        "include_sinuses": include_sinuses,
        "solid_air_body_stl": solid_stl.name,
        "solid_air_body_nrrd": solid_mask_path.name,
        "patches": patches,
        "notes": notes,
        "openfoam_hint": {
            "mesh": "snappyHexMesh or cfMesh on solid_air_body.stl with named patches",
            "U_inlets": "flowRateInletVelocity on left_nostril / right_nostril",
            "U_outlet": "pressureInletOutletVelocity on trachea",
            "U_wall": "noSlip on wall",
            "p_outlet": "fixedValue 0 on trachea",
        },
    }
    man_path = geom_dir / f"{case_id}_openfoam_manifest.json"
    with man_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return OpenFoamExportResult(
        case_id=case_id,
        out_dir=str(geom_dir),
        solid_air_volume_ml=vol_ml,
        patches=patches,
        notes=notes,
    )


def _openfoam_readme_text(
    case_id: str,
    patches: dict[str, str],
    vol_ml: float,
    include_sinuses: bool,
) -> str:
    lines = [
        f"OpenFOAM geometry export — {case_id}",
        "=" * 60,
        "",
        "WHAT IS THE SOLID AIR BODY?",
        "  The file *_solid_air_body.stl is the OUTER SURFACE of the region that",
        "  contains air: nasal cavity, path to trachea"
        + (", and connected sinuses" if include_sinuses else "")
        + ".",
        "  Think of it as a hollow 'pipe network' shaped like the anatomy.",
        f"  Approximate enclosed air volume: {vol_ml:.1f} mL.",
        "  CFD solves for velocity/pressure of air INSIDE this solid.",
        "",
        "WHAT ARE OPEN-PORT PATCHES?",
        "  Separate STLs for openings where air enters/leaves:",
    ]
    for name, fname in patches.items():
        role = {
            "wall": "WALL (mucosa) — no-slip, no flow through",
            "left_nostril": "INLET — prescribed inspiratory flow",
            "right_nostril": "INLET — prescribed inspiratory flow",
            "trachea": "OUTLET — fixed pressure (to lungs/ambient gauge)",
        }.get(name, "boundary")
        lines.append(f"    {fname:40s}  {role}")
    lines += [
        "",
        "Together, wall + open ports should cover the full fluid boundary.",
        "OpenFOAM (snappyHexMesh) uses these named surfaces as boundary patches.",
        "",
        "MINIMAL OPENFOAM WORKFLOW",
        "  1. Create a case directory (e.g. foam/VisibleHuman_Head).",
        "  2. Copy STLs into constant/triSurface/.",
        "  3. blockMesh: coarse background box around the anatomy.",
        "  4. snappyHexMesh: snap to solid_air_body.stl; assign patch names",
        "     from the open-port STLs (or use surfaceFeatureExtract + refinements).",
        "  5. Set 0/U and 0/p BCs:",
        "       left_nostril / right_nostril : flowRateInletVelocity",
        "       trachea                     : fixedValue p=0, pressureInletOutletVelocity",
        "       wall                        : noSlip",
        "  6. Use simpleFoam (steady laminar/RAS) or pimpleFoam (transient).",
        "",
        "AIRFLOW DIRECTION (this project)",
        "  In: both nostrils  →  through nasal passage / sinuses  →  Out: trachea",
        "  Mouth is closed (not an open patch).",
        "",
        "See docs/openfoam.md in the Sinus_CFD repo for a fuller explanation.",
        "",
    ]
    return "\n".join(lines)
