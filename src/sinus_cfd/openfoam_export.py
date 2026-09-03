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
from scipy.spatial import cKDTree
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


# A decimation that changes the enclosed volume by more than this is not a
# simplification of the same body and is rejected. Quadric decimation on a real
# airway surface moves the volume by ~0.2%; the failure mode it guards against
# moved it by 100%.
DECIMATION_MAX_VOLUME_CHANGE = 0.05
# The exported surface must enclose the same volume as the sealed voxel mask.
# Marching cubes plus Taubin smoothing on a real airway lands within a few
# percent; anything past this is a broken surface, not a smoothed one.
MESH_VS_VOXEL_MAX_REL_DIFF = 0.15


def _decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh | None:
    """Quadric-decimate to ``target`` faces, or None if that is not possible.

    ``face_count=`` is passed by KEYWORD. trimesh 4.x changed the signature to
    ``(percent, face_count, aggression)``, so a positional call sets *percent*
    and raises ``target_reduction must be between 0 and 1``.

    Returning None rather than falling back is deliberate. The old fallback took
    ``np.linspace`` over the face array and kept every Nth triangle, which is not
    decimation at all -- it shreds the surface into disconnected triangles. On
    CQ500CT390 the caller's ``_largest_component`` then picked 6 triangles out of
    266,060, ``fill_holes`` closed them into a degenerate shell, ``is_watertight``
    returned True, and the export announced "watertight (good for
    snappyHexMesh)" over a 0.00 mL body. A heavier mesh is a cost; a wrong one is
    a wrong answer.
    """
    if len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=int(target))
    except Exception:
        return None


def _decimation_kept_the_body(
    before: trimesh.Trimesh, after: trimesh.Trimesh
) -> tuple[bool, str]:
    """Did decimation produce a simplification of the same solid, or a ruin?"""
    if after is None or len(after.faces) < 4:
        return False, f"collapsed to {0 if after is None else len(after.faces)} faces"
    try:
        v0 = abs(float(before.volume))
        v1 = abs(float(after.volume))
    except Exception:
        return True, ""  # volume undefined on an open surface; topology tests stand
    if v0 <= 0.0:
        return True, ""
    rel = abs(v1 - v0) / v0
    if rel > DECIMATION_MAX_VOLUME_CHANGE:
        return False, (f"volume changed {100 * rel:.1f}% "
                       f"({v0 / 1000.0:.2f} -> {v1 / 1000.0:.2f} mL)")
    return True, ""


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


def seal_solid_for_watertight_mesh(
    solid: np.ndarray,
    port_masks: list[np.ndarray | None],
    close_radius: int = 2,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Close open nares/trachea holes so marching cubes yields a closed shell.

    Strategy:
      1. Dilate open-port voxels into the solid (caps the openings on the mask)
      2. Morphological closing
      3. binary_fill_holes (multiple passes)
      4. Keep largest connected component

    ``exclude`` is air that must stay OUT of the sealed solid: the sinus bodies
    the strip removed. Each is a 3-D hole in the passage, so the two
    whole-solid fill passes below refilled every one of them -- on VH male the
    passage went in clean (47.6 mL, 0.00 mL sinus) and the sealed solid came
    out at 75.4 mL holding 23.55 of the 23.5 mL that had been stripped. The
    exclusion is applied after every morphological step.
    """
    sealed = solid.astype(bool)
    excl = None if exclude is None else np.asarray(exclude).astype(bool)

    def _keep_out(m: np.ndarray) -> np.ndarray:
        return m if excl is None else (m & ~excl)

    ball_port = morphology.ball(max(close_radius + 1, 2))
    for pm in port_masks:
        if pm is None:
            continue
        pm = pm.astype(bool)
        if not pm.any():
            continue
        sealed = sealed | morphology.dilation(pm, footprint=ball_port)
    sealed = _keep_out(sealed)

    ball = morphology.ball(close_radius)
    sealed = _keep_out(morphology.closing(sealed, footprint=ball))
    sealed = _keep_out(ndi.binary_fill_holes(sealed))
    # Second close catches residual tunnels
    sealed = _keep_out(morphology.closing(sealed, footprint=morphology.ball(1)))
    sealed = _keep_out(ndi.binary_fill_holes(sealed))

    lab, n = ndi.label(sealed)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        sealed = lab == int(np.argmax(counts))
    return sealed


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep the connected component with the most faces (drops slivers)."""
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: len(m.faces))


def _smoothed_field_mesh(
    solid_closed: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    sigma_vox: float = 0.8,
    pad: int = 3,
) -> trimesh.Trimesh:
    """
    Marching cubes on a Gaussian-blurred field instead of the raw binary mask.

    Raw binary voxel marching cubes on thin, complex topology (nasal
    passages are exactly this) can hit non-manifold cube configurations —
    the root cause of the open-edge / high-skewness surfaces that made
    snappyHexMesh choke on prism layers. Blurring the mask to a continuous
    field before thresholding at 0.5 removes the voxel-adjacency ambiguity
    that triggers those configurations, at the cost of ~0.3-0.5 voxel of
    rounding at the surface (negligible next to CT partial-volume blur).
    Padding avoids clipping the surface where it nears the array boundary.
    """
    padded = np.pad(solid_closed.astype(np.float32), pad, mode="constant", constant_values=0.0)
    field = ndi.gaussian_filter(padded, sigma=sigma_vox)
    origin_adj = (
        origin[0] - pad * spacing[0],
        origin[1] - pad * spacing[1],
        origin[2] - pad * spacing[2],
    )
    return _mask_to_mesh(field, spacing, origin_adj, level=0.5)


def solid_mask_to_watertight_mesh(
    solid_closed: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    target_faces: int = 40000,
    smooth_sigma_vox: float = 0.8,
    taubin_iterations: int = 15,
) -> tuple[trimesh.Trimesh, list[str]]:
    """
    Marching cubes + hole fill + careful decimation for snappyHexMesh.

    Returns (mesh, notes). Prefers a watertight result over aggressive decimation.
    """
    notes: list[str] = []
    try:
        mesh = _smoothed_field_mesh(solid_closed, spacing, origin, sigma_vox=smooth_sigma_vox)
        notes.append(f"Smoothed-field marching cubes (Gaussian sigma={smooth_sigma_vox} vox).")
    except Exception as e:
        mesh = _mask_to_mesh(solid_closed, spacing, origin)
        notes.append(f"Smoothed-field marching cubes failed ({e}); used raw binary mask.")

    mesh = _largest_component(mesh)
    try:
        mesh.process(validate=True)
    except Exception:
        pass

    if taubin_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=taubin_iterations)
            notes.append(f"Taubin-smoothed surface ({taubin_iterations} iterations, shrink-free).")
        except Exception as e:
            notes.append(f"Taubin smoothing skipped: {e}")

    for _ in range(6):
        if bool(mesh.is_watertight):
            break
        try:
            mesh.fill_holes()
        except Exception:
            break
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    if not bool(mesh.is_watertight):
        # Slight dilation of the mask often closes remaining pinholes
        notes.append("First mesh not watertight — retry after +1 voxel dilation.")
        dil = morphology.dilation(solid_closed, footprint=morphology.ball(1))
        dil = ndi.binary_fill_holes(dil)
        mesh = _mask_to_mesh(dil, spacing, origin)
        for _ in range(6):
            if bool(mesh.is_watertight):
                break
            try:
                mesh.fill_holes()
            except Exception:
                break

    was_wt = bool(mesh.is_watertight)
    n_faces_raw = len(mesh.faces)
    if n_faces_raw > target_faces:
        simplified = _decimate(mesh, target_faces)
        if simplified is None:
            notes.append(
                f"Quadric decimation unavailable (install fast_simplification); "
                f"keeping the full-resolution mesh ({n_faces_raw} faces)."
            )
        else:
            simplified = _largest_component(simplified)
            # Re-fill after decimation (often opens small holes)
            for _ in range(4):
                if bool(simplified.is_watertight):
                    break
                try:
                    simplified.fill_holes()
                except Exception:
                    break
            kept, why = _decimation_kept_the_body(mesh, simplified)
            if not kept:
                # is_watertight alone cannot catch this: a degenerate closed
                # shell is watertight and bounds nothing.
                notes.append(
                    f"REJECTED decimation of the solid mesh -- {why}. "
                    f"Keeping the full-resolution mesh ({n_faces_raw} faces)."
                )
            elif bool(simplified.is_watertight) or not was_wt:
                mesh = simplified
                notes.append(
                    f"Decimated solid mesh {n_faces_raw} → {len(mesh.faces)} faces."
                )
            else:
                # Keep denser mesh to preserve watertightness
                notes.append(
                    f"Kept unsimplified solid mesh ({n_faces_raw} faces) to stay watertight."
                )
    try:
        mesh.fix_normals()
    except Exception:
        pass

    if bool(mesh.is_watertight):
        try:
            vol = float(abs(mesh.volume))
            notes.append(f"Watertight solid mesh volume ≈ {vol / 1000.0:.2f} mL.")
        except Exception:
            notes.append("Solid mesh is watertight.")
    else:
        notes.append(
            f"WARNING: solid_air_body still not watertight "
            f"(faces={len(mesh.faces)}, verts={len(mesh.vertices)})."
        )
    return mesh, notes


def _write_multiregion_stl(
    mesh: trimesh.Trimesh,
    face_region: np.ndarray,
    path: Path,
) -> dict[str, int]:
    """
    Write ASCII STL with named solids (OpenFOAM multi-region triSurfaceMesh).

    face_region: object array of region name per face.
    """
    counts: dict[str, int] = {}
    lines: list[str] = []
    verts = mesh.vertices
    faces = mesh.faces
    # group face indices by region
    regions: dict[str, list[int]] = {}
    for i, name in enumerate(face_region):
        regions.setdefault(str(name), []).append(i)

    for rname, idxs in sorted(regions.items()):
        lines.append(f"solid {rname}")
        for fi in idxs:
            f = faces[fi]
            tri = verts[f]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            else:
                n = np.array([0.0, 0.0, 1.0])
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            for v in tri:
                lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {rname}")
        counts[rname] = len(idxs)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def label_solid_faces_by_ports(
    solid_mesh: trimesh.Trimesh,
    port_meshes: dict[str, trimesh.Trimesh | None],
    max_dist_mm: float = 5.0,
) -> np.ndarray:
    """
    Assign each solid face to nearest open-port patch if within max_dist_mm, else wall.
    """
    centers = solid_mesh.triangles_center
    labels = np.full(len(centers), "wall", dtype=object)
    best_d = np.full(len(centers), np.inf, dtype=float)

    for name, pm in port_meshes.items():
        if pm is None or len(pm.faces) == 0:
            continue
        tree = cKDTree(pm.triangles_center)
        d, _ = tree.query(centers, k=1, workers=-1)
        nearer = d < best_d
        take = nearer & (d <= max_dist_mm)
        labels[take] = name
        best_d[take] = d[take]
    return labels


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
    include_sinuses: bool = False,
    sinus_exclude: np.ndarray | None = None,
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
    # Hard guard: the solid the mesher receives must not hold the sinus air the
    # strip removed. include_sinuses=True merged every interior-air component
    # touching the passage -- the stripped sinuses at their ostia -- and was the
    # default; it put 23.5 mL back into VH male after autosegment and
    # analyze_passage had both kept the domain clean.
    if sinus_exclude is not None:
        excl = np.asarray(sinus_exclude).astype(bool)
        if excl.shape == solid.shape and excl.any():
            inside = float((solid & excl).sum()) / float(excl.sum())
            if inside > 0.05:
                raise ValueError(
                    f"[{case_id}] {100 * inside:.0f}% of the stripped sinus air "
                    f"({(solid & excl).sum() * float(np.prod(spacing)) / 1000.0:.1f} mL) "
                    "is inside the solid air body; refusing to export a CFD domain "
                    "that contains sinuses (CLAUDE.md goal 1)."
                )
            notes.append(
                f"Stripped sinus air inside the solid body: {100 * inside:.1f}% -- "
                "sinus-free domain preserved."
            )
    sp_vol = float(np.prod(spacing))
    vol_ml = float(solid.sum() * sp_vol / 1000.0)
    notes.append(
        f"Solid air body volume = {vol_ml:.2f} mL "
        f"({'passage + connected sinuses' if include_sinuses else 'passage only'})."
    )

    # Split inlets L/R early — needed to seal ports for a closed solid mesh
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

    # Seal open ports + morph close so snappy can use locationInMesh / mode inside
    solid_closed = seal_solid_for_watertight_mesh(
        solid,
        port_masks=[left_inlet_mask, right_inlet_mask, outlet_open],
        close_radius=2,
        exclude=sinus_exclude,
    )
    vol_closed_ml = float(solid_closed.sum() * sp_vol / 1000.0)
    notes.append(
        f"Sealed solid (ports closed) voxel volume = {vol_closed_ml:.2f} mL."
    )
    # The guard that matters is on the SEALED solid -- that is what is written
    # and meshed. The pre-seal check above passed on VH male while the seal's
    # fill passes put 23.55 of 23.5 mL of sinus back.
    if sinus_exclude is not None:
        excl = np.asarray(sinus_exclude).astype(bool)
        if excl.shape == solid_closed.shape and excl.any():
            inside = float((solid_closed & excl).sum()) / float(excl.sum())
            if inside > 0.05:
                raise ValueError(
                    f"[{case_id}] {100 * inside:.0f}% of the stripped sinus air "
                    f"({(solid_closed & excl).sum() * sp_vol / 1000.0:.1f} mL) is inside "
                    "the SEALED solid air body; refusing to export a CFD domain that "
                    "contains sinuses (CLAUDE.md goal 1)."
                )
            notes.append(
                f"Stripped sinus air inside the sealed solid: {100 * inside:.1f}% -- "
                "sinus-free domain preserved through sealing."
            )

    # Save sealed solid air mask (what snappy should keep as fluid)
    solid_img = sitk.GetImageFromArray(solid_closed.astype(np.uint8))
    solid_img.SetSpacing(spacing)
    solid_img.SetOrigin(origin)
    if reference_image is not None:
        solid_img.SetDirection(reference_image.GetDirection())
    solid_mask_path = geom_dir / f"{case_id}_solid_air_body.nrrd"
    sitk.WriteImage(solid_img, str(solid_mask_path))

    solid_mesh, mesh_notes = solid_mask_to_watertight_mesh(
        solid_closed, spacing, origin, target_faces=40000
    )
    notes.extend(mesh_notes)
    is_wt = bool(solid_mesh.is_watertight)
    # The surface must bound the SAME body the voxel mask does. Watertightness is
    # a topological test and passes on a degenerate closed shell: CQ500CT390 once
    # exported a 6-face, 0.00 mL "watertight" solid against a 38.91 mL mask and
    # the run continued. This is the check that makes that a hard failure.
    mesh_ml = 0.0
    if is_wt:
        try:
            mesh_ml = abs(float(solid_mesh.volume)) / 1000.0
        except Exception:
            mesh_ml = 0.0
        rel = abs(mesh_ml - vol_closed_ml) / max(vol_closed_ml, 1e-9)
        if rel > MESH_VS_VOXEL_MAX_REL_DIFF:
            raise ValueError(
                f"[{case_id}] solid_air_body surface encloses {mesh_ml:.2f} mL but the "
                f"sealed voxel mask is {vol_closed_ml:.2f} mL ({100 * rel:.0f}% apart, "
                f"limit {100 * MESH_VS_VOXEL_MAX_REL_DIFF:.0f}%). The surface does not "
                f"represent the airway -- refusing to hand it to snappyHexMesh. "
                f"faces={len(solid_mesh.faces)}, verts={len(solid_mesh.vertices)}."
            )
        notes.append(
            f"solid_air_body mesh is watertight and encloses {mesh_ml:.2f} mL "
            f"against a {vol_closed_ml:.2f} mL voxel mask "
            f"({100 * rel:.1f}% apart) -- good for snappyHexMesh."
        )
    else:
        notes.append(
            "WARNING: solid_air_body mesh is not fully watertight; "
            "snappy may keep the background box."
        )
    # Write watertight flag for scaffold
    (geom_dir / f"{case_id}_solid_watertight.flag").write_text(
        "1" if is_wt else "0", encoding="utf-8"
    )

    # Interior seed for snappy locationInMesh (mm): the voxel FARTHEST from any
    # wall, never the centroid. A nasal airway is a thin curved sheet, so its
    # centroid lands beside a wall -- CQ500CT390's sat 0.85 mm from one, and
    # snappy then keeps the background box instead of the lumen. The same
    # averaging-positions trap already bit the naris and outlet ports.
    sx, sy, sz = spacing
    ox, oy, oz = origin
    if solid_closed.any():
        edt = ndi.distance_transform_edt(solid_closed, sampling=(sz, sy, sx))
        iz, iy, ix = np.unravel_index(int(np.argmax(edt)), edt.shape)
        loc_mm = [
            float(ox + ix * sx),
            float(oy + iy * sy),
            float(oz + iz * sz),
        ]
        notes.append(
            f"locationInMesh seed is {float(edt[iz, iy, ix]):.2f} mm from the nearest "
            "wall (deepest interior voxel)."
        )
    else:
        b = solid_mesh.bounds
        loc_mm = (0.5 * (b[0] + b[1])).tolist()
    (geom_dir / f"{case_id}_locationInMesh_mm.json").write_text(
        json.dumps({"locationInMesh_mm": loc_mm}, indent=2), encoding="utf-8"
    )
    notes.append(f"locationInMesh (mm) = {loc_mm}")

    names_in = port_names_inlet or ["left_nostril", "right_nostril"]
    patches: dict[str, str] = {}

    # Open-port patches (separate STLs for QC + face labeling)
    port_meshes: dict[str, trimesh.Trimesh | None] = {}
    for name, m in (
        (names_in[0], left_inlet_mask),
        (names_in[1] if len(names_in) > 1 else "right_nostril", right_inlet_mask),
    ):
        patch = _surface_from_mask_region(m, solid, spacing, origin)
        if patch is None:
            patch = _planar_cap_mesh(m, spacing, origin)
        port_meshes[name] = patch
        if patch is not None and len(patch.faces) > 0:
            p = geom_dir / f"{case_id}_patch_{name}.stl"
            patch.export(p)
            patches[name] = p.name
            notes.append(f"Open inlet patch: {p.name} ({len(patch.faces)} faces)")

    out_patch = _surface_from_mask_region(outlet_open, solid, spacing, origin)
    if out_patch is None:
        out_patch = _planar_cap_mesh(outlet_open, spacing, origin)
    port_meshes[port_name_outlet] = out_patch
    if out_patch is not None and len(out_patch.faces) > 0:
        p = geom_dir / f"{case_id}_patch_{port_name_outlet}.stl"
        out_patch.export(p)
        patches[port_name_outlet] = p.name
        notes.append(f"Open outlet patch: {p.name} ({len(out_patch.faces)} faces)")

    # Multi-region closed solid (single STL with named solids for snappy patches)
    face_regions = label_solid_faces_by_ports(solid_mesh, port_meshes, max_dist_mm=5.0)
    solid_stl = geom_dir / f"{case_id}_solid_air_body.stl"
    region_counts = _write_multiregion_stl(solid_mesh, face_regions, solid_stl)
    notes.append(
        "solid_air_body.stl = multi-region closed surface "
        f"(regions={region_counts}). Air is *inside* this body."
    )
    for rname, nfaces in region_counts.items():
        if rname not in patches:
            patches[rname] = solid_stl.name
        notes.append(f"  multi-region '{rname}': {nfaces} faces")

    # Also export wall-only faces for preview
    wall_ids = np.where(face_regions == "wall")[0]
    if len(wall_ids) > 0:
        try:
            wf = solid_mesh.faces[wall_ids]
            used = np.unique(wf.ravel())
            remap = -np.ones(len(solid_mesh.vertices), dtype=int)
            remap[used] = np.arange(len(used))
            wall_mesh = trimesh.Trimesh(
                vertices=solid_mesh.vertices[used], faces=remap[wf], process=False
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
        "solid_air_sealed_volume_ml": vol_closed_ml,
        "solid_mesh_watertight": is_wt,
        "locationInMesh_mm": loc_mm,
        "multi_region_face_counts": region_counts,
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
