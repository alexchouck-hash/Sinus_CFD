"""
Whole-head CT processing (Visible Human and similar).

Produces:
  - multi-tissue labels (air / soft tissue / cartilage / bone / exterior)
  - solid head surface (body shell) for semi-transparent display
  - airway lumen (nostrils → pharynx → trachea, caudal direction)
  - BCs: both nostrils inlet, trachea outlet, mouth closed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import morphology
from skimage.morphology import convex_hull_image

from .boundary_conditions import (
    BoundarySetup,
    Port,
    assign_flow,
    export_port_markers_ply,
    tag_mesh_faces_near_ports,
    write_boundary_setup,
    write_openfoam_bc_notes,
)
from .physiology import PatientBreathing, summary_text
from .pipeline import _mask_to_mesh
from .edge_segment import run_edge_segmentation
from .skin_and_nares import (
    detect_external_nares,
    extract_skin_shell,
    mesh_skin_surface,
)
from .tissues import TISSUE_LABELS


@dataclass
class WholeHeadResult:
    case_id: str
    image_path: str
    size_xyz: list[int]
    spacing_xyz_mm: list[float]
    origin_xyz_mm: list[float]
    crop_origin_zyx: list[int]
    head_voxels: int
    airway_voxels: int
    airway_volume_ml: float
    head_mesh_faces: int
    airway_mesh_faces: int
    outlet_is_proxy: bool
    superior_is_high_z: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bbox(mask: np.ndarray, margin: int = 4) -> tuple[slice, slice, slice]:
    zz, yy, xx = np.where(mask)
    if len(zz) == 0:
        raise ValueError("Empty mask for bounding box")
    z0 = max(int(zz.min()) - margin, 0)
    z1 = min(int(zz.max()) + margin + 1, mask.shape[0])
    y0 = max(int(yy.min()) - margin, 0)
    y1 = min(int(yy.max()) + margin + 1, mask.shape[1])
    x0 = max(int(xx.min()) - margin, 0)
    x1 = min(int(xx.max()) + margin + 1, mask.shape[2])
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def interior_air_within_hull(
    hu: np.ndarray,
    body: np.ndarray,
    air_hu_max: float,
    voxel_ml: float,
    min_speck_ml: float = 0.02,
) -> np.ndarray:
    """
    Recover interior (nasal / sinus / pharyngeal) air on OPEN-NARES living scans.

    On a living head CT the nasal lumen is 26-connected to the ambient FOV air
    through the nostrils (and to the exterior through the caudal FOV cut at the
    neck), so ``binary_fill_holes`` cannot reclaim it and ``body & (hu<=air)``
    collapses to a handful of sealed-sinus voxels (~1e3). This mirrors the
    body-hull method proven in ``scripts/assess_dicom_incoming.internal_air_ml``:
    take the per-axial-slice convex hull of the body silhouette (which seals the
    in-plane nostril gap and the neck outline), then keep low-HU voxels inside
    that hull. Works at any spacing because the hull is silhouette-geometric and
    the speck filter is expressed in mL, not voxels.
    """
    hull = np.zeros_like(body)
    for z in range(body.shape[0]):
        sl = body[z]
        if int(sl.sum()) < 50:  # too little tissue to define a head silhouette
            continue
        try:
            hull[z] = convex_hull_image(sl)
        except Exception:
            hull[z] = sl  # degenerate slice: fall back to raw body silhouette

    interior = (hu <= air_hu_max) & hull
    # Light closing to reconnect thin passages without bridging tissue walls
    interior = morphology.closing(interior, footprint=morphology.ball(1))
    labeled, n = ndi.label(interior)
    if n == 0:
        return interior.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    min_speck = max(int(round(min_speck_ml / max(voxel_ml, 1e-9))), 8)
    keep = counts >= min_speck
    keep[0] = False
    return keep[labeled].astype(bool)


def infer_superior_is_high_z(
    image: sitk.Image,
    body: np.ndarray | None = None,
) -> tuple[bool, str]:
    """
    Determine whether array index z increases toward the head (superior).

    Preferred: DICOM ImagePositionPatient along the slice normal (LPS: +Z superior).
    Fallback: body cross-section (neck often smaller than mid-face; crown also small —
    use IPP when available).
    """
    # Try direction + origin: if z-component of spacing direction points +superior
    # SimpleITK GetOrigin()[2] is physical z of index (0,0,0)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    size = image.GetSize()
    # Physical z of first and last slice (index k along image z)
    # For a pure axial LPS stack, origin_z of last slice = origin[2] + (nz-1)*spacing[2]*dir
    direction = image.GetDirection()  # 3x3 row-major
    # Column 2 is the image-z axis in physical space
    z_axis = np.array([direction[2], direction[5], direction[8]], dtype=float)
    # Physical position of last voxel center along k
    # sign of z_axis[2] * spacing[2]: if positive, higher k → higher physical Z (superior in LPS)
    k_sign = float(z_axis[2] * spacing[2])
    if abs(k_sign) > 1e-6:
        superior_is_high_z = k_sign > 0
        # Also check origin vs last
        z0 = origin[2]
        z1 = origin[2] + (size[2] - 1) * spacing[2] * z_axis[2]
        method = (
            f"image_z_axis LPS: physical_z first={z0:.1f} last={z1:.1f} "
            f"→ superior_is_high_z={superior_is_high_z}"
        )
        return superior_is_high_z, method

    # Fallback body-area heuristic (unreliable alone on head FOV)
    if body is not None:
        n_z = body.shape[0]
        a0 = float(body[: max(n_z // 10, 1)].sum())
        a1 = float(body[-max(n_z // 10, 1) :].sum())
        # Smaller end often crown OR neck; prefer "neck has shoulders" → larger inferior
        # If low-z larger → low-z inferior (shoulders) → superior is high z
        superior_is_high_z = a0 > a1
        return superior_is_high_z, f"body_area fallback a0={a0:.0f} a1={a1:.0f}"

    return True, "default superior_is_high_z=True"


# Below this, the array-y axis is treated as degenerate (oblique or coronal
# acquisition) and the anatomy heuristic is used instead. Matches the z-axis
# test above.
Y_AXIS_DEGENERATE_TOL = 1e-6


def infer_y_anterior_is_low_from_image(
    image: sitk.Image,
) -> tuple[bool | None, str]:
    """Array-y anterior/posterior polarity from the direction matrix.

    The sibling z-axis flag has always been derived from image geometry
    (``infer_superior_is_high_z``); the A-P flag was derived from *anatomy* —
    "the body y-extreme nearer the air centroid is the face". That heuristic
    inverts whenever the interior-air mask is unrepresentative, which is
    exactly what happens on an open-nares living scan where the enclosed-air
    mask collapses to a handful of sealed mastoid cells (CQ500CT105: 0.15 mL,
    median y=275 of a head spanning y[19,482] -> "anterior is high y", wrong).

    This reads direction COLUMN 1 (the array-y axis). In LPS, +y is POSTERIOR,
    so array-y increasing toward +y means anterior is LOW array-y.

    Returns ``None`` when the matrix is oblique or degenerate, so "geometry
    could not decide" stays distinguishable from "geometry said True" and the
    caller can fall back rather than silently defaulting.
    """
    direction = image.GetDirection()  # 3x3 row-major
    y_axis = np.array([direction[1], direction[4], direction[7]], dtype=float)
    j_sign = float(y_axis[1] * image.GetSpacing()[1])
    if abs(j_sign) <= Y_AXIS_DEGENERATE_TOL:
        return None, "image_y_axis degenerate (oblique/coronal); anatomy fallback"
    y_anterior_is_low = j_sign > 0
    return y_anterior_is_low, (
        f"image_y_axis LPS: y_axis={tuple(round(float(v), 3) for v in y_axis)} "
        f"j_sign={j_sign:+.3f} -> y_anterior_is_low={y_anterior_is_low}"
    )


def _orient_face_and_neck(
    body: np.ndarray,
    superior_is_high_z: bool,
    y_anterior_is_low: bool,
    spacing_xyz: tuple[float, float, float],
    neck_mm: float = 40.0,
    crown_mm: float = 30.0,
    anterior_depth_mm: float = 45.0,
) -> dict[str, Any]:
    """
    Face (nostrils) vs neck (trachea) bands in PHYSICAL mm.

    Orientation is supplied by the single upstream inference (``y_anterior_is_low``
    from ``edge_segment`` and ``superior_is_high_z`` from the DICOM direction) — this
    function no longer runs its own A–P heuristic, so the seed bands can never
    disagree with the nares / trachea ports. All band widths are converted from mm
    using the image spacing so they behave identically at 0.5 mm and 1.5 mm.

    Nostrils: anterior mid-face band (not the cranial crown, not the neck).
    Trachea: most caudal ``neck_mm`` of the head.
    """
    sy = float(spacing_xyz[1])  # array axis 1 (y) spacing
    sz = float(spacing_xyz[2])  # array axis 0 (z) spacing
    n_z = body.shape[0]
    n_y = body.shape[1]

    # Anchor bands to the body's actual z-extent (crop already trims to bbox).
    zb = np.where(body.any(axis=(1, 2)))[0]
    z0b, z1b = (int(zb.min()), int(zb.max())) if len(zb) else (0, n_z - 1)
    neck_sl = max(int(round(neck_mm / sz)), 8)
    crown_sl = max(int(round(crown_mm / sz)), 8)

    if superior_is_high_z:
        # high z = superior (crown), low z = inferior (neck)
        z_neck = slice(z0b, min(z0b + neck_sl, n_z))
        z_crown = slice(max(z1b - crown_sl + 1, 0), n_z)
        z_face = slice(min(z0b + neck_sl, n_z), max(z1b - crown_sl + 1, 0))
    else:
        # low z = superior (crown), high z = inferior (neck)
        z_neck = slice(max(z1b - neck_sl + 1, 0), n_z)
        z_crown = slice(z0b, min(z0b + crown_sl, n_z))
        z_face = slice(min(z0b + crown_sl, n_z), max(z1b - neck_sl + 1, 0))

    # Anterior face band anchored to the body's anterior y-extreme, width in mm.
    yb = np.where(body.any(axis=(0, 2)))[0]
    y0b, y1b = (int(yb.min()), int(yb.max())) if len(yb) else (0, n_y - 1)
    ant_sl = max(int(round(anterior_depth_mm / sy)), 12)
    if y_anterior_is_low:
        y_ant = slice(y0b, min(y0b + ant_sl, n_y))
    else:
        y_ant = slice(max(y1b - ant_sl + 1, 0), min(y1b + 1, n_y))

    return {
        "superior_is_high_z": superior_is_high_z,
        "z_face": z_face,
        "z_neck": z_neck,
        "z_crown": z_crown,
        "y_anterior_is_low": y_anterior_is_low,
        "y_ant": y_ant,
    }


def _geodesic_airway_bridge(
    hu: np.ndarray,
    body: np.ndarray,
    air: np.ndarray,
    bone: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    superior_is_high_z: bool,
    tube_radius: int = 3,
) -> np.ndarray:
    """
    Low-HU geodesic from nostrils → neck; heavily penalize going superior
    and through bone.
    """
    from skimage.graph import route_through_array

    costs = np.full(hu.shape, 1.0e6, dtype=np.float64)
    inside = body.astype(bool)
    # Prefer low HU (air / soft cavity)
    costs[inside] = np.clip((hu[inside] + 1000.0) / 6.0, 1.0, 500.0)
    costs[air] = 0.4
    # Bone is nearly forbidden
    costs[bone] = 1.0e5
    # Soft bias: do not travel through crown (superior to start)
    sz, sy, sx = start
    zz = np.arange(hu.shape[0])[:, None, None]
    if superior_is_high_z:
        # higher z than start is superior — penalize
        superior_zone = zz > (sz + 2)
    else:
        superior_zone = zz < (sz - 2)
    costs = np.where(superior_zone & inside, costs * 25.0 + 50.0, costs)

    try:
        indices, _cost = route_through_array(
            costs,
            start=start,
            end=end,
            fully_connected=True,
            geometric=True,
        )
    except Exception:
        return np.zeros_like(air, dtype=bool)

    path = np.zeros_like(air, dtype=bool)
    for iz, iy, ix in indices:
        path[iz, iy, ix] = True
    if tube_radius > 0:
        path = morphology.dilation(path, footprint=morphology.ball(tube_radius))
    path &= body
    path &= ~bone  # keep lumen out of bone
    return path


def select_nasal_to_trachea_path(
    air: np.ndarray,
    body: np.ndarray,
    hu: np.ndarray,
    bone: np.ndarray,
    superior_is_high_z: bool,
    y_anterior_is_low: bool,
    spacing_xyz: tuple[float, float, float],
    trachea_half_width_mm: float = 22.0,
    min_component_ml: float = 0.5,
    min_airway_ml: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build fluid domain: nostrils (anterior mid-face) → caudal to trachea.

    Direction is constrained inferiorly (down the neck), not into the cranial vault.
    Orientation ``(y_anterior_is_low, superior_is_high_z)`` is passed in from the
    single upstream inference so seed placement matches the nares / trachea ports.
    All seed bands and size filters are physical (mm / mL) via ``spacing_xyz``.
    """
    info: dict[str, Any] = {
        "superior_is_high_z": superior_is_high_z,
        "y_anterior_is_low": y_anterior_is_low,
    }
    if not air.any():
        raise ValueError("Empty air mask")

    voxel_ml = float(np.prod(spacing_xyz)) / 1000.0
    sx = float(spacing_xyz[0])  # array axis 2 (x) spacing
    x_half = max(int(round(trachea_half_width_mm / sx)), 6)
    min_component_vox = max(int(round(min_component_ml / max(voxel_ml, 1e-9))), 30)

    orient = _orient_face_and_neck(
        body, superior_is_high_z, y_anterior_is_low, spacing_xyz
    )
    z_face, z_neck, z_crown = orient["z_face"], orient["z_neck"], orient["z_crown"]
    y_ant = orient["y_ant"]

    zz, yy, xx = np.where(air)
    x_mid = int(np.median(xx))

    # --- Nostril seeds: anterior + mid-face air (exclude crown / intracranial) ---
    nostril_region = np.zeros_like(air)
    nostril_region[z_face, y_ant, :] = True
    nostril_region &= air
    # Remove air that is mostly superior vault
    crown_air = np.zeros_like(air)
    crown_air[z_crown] = True
    nostril_region &= ~crown_air
    info["nostril_seed_voxels"] = int(nostril_region.sum())
    if not nostril_region.any():
        raise ValueError(
            "Empty nostril seed region: no anterior mid-face air found for "
            f"orientation y_anterior_is_low={y_anterior_is_low}, "
            f"superior_is_high_z={superior_is_high_z}. Check orientation inference "
            "and interior-air recovery (would otherwise fall back to a ~0 mL airway)."
        )

    # --- Trachea seeds: inferior neck, midline, air or very low HU ---
    x0, x1 = max(x_mid - x_half, 0), min(x_mid + x_half, air.shape[2])
    trachea_region = np.zeros_like(air)
    trachea_region[z_neck, :, x0:x1] = True
    trachea_region &= body & (hu <= -80)
    if int(trachea_region.sum()) < min_component_vox // 20:
        trachea_region = np.zeros_like(air)
        trachea_region[z_neck, :, x0:x1] = True
        trachea_region &= air
    info["trachea_seed_voxels"] = int(trachea_region.sum())
    if not trachea_region.any():
        raise ValueError(
            "Empty trachea seed region: no caudal midline neck air found for "
            f"orientation superior_is_high_z={superior_is_high_z} "
            f"(neck band {z_neck}, x±{x_half} vox). Check orientation / FOV extent."
        )

    labeled, n = ndi.label(air)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    both, nose_only = [], []
    for lab_id in range(1, n + 1):
        if counts[lab_id] < min_component_vox:
            continue
        comp = labeled == lab_id
        # Exclude pure crown components
        if (comp & crown_air).mean() > 0.7 and not (comp & nostril_region).any():
            continue
        t_nose = bool((comp & nostril_region).any())
        t_trach = bool((comp & trachea_region).any())
        if t_nose and t_trach:
            both.append(lab_id)
        elif t_nose:
            nose_only.append(lab_id)

    if both:
        best = max(both, key=lambda i: counts[i])
        keep = labeled == best
        # Strip residual superior vault blobs
        keep = _clip_superior_to_start(keep, nostril_region, superior_is_high_z)
        info["selection"] = "connected_nose_to_trachea"
        return keep.astype(bool), info

    # Closing bridge (still constrained)
    for radius in (2, 3, 4):
        bridged = morphology.closing(air, footprint=morphology.ball(radius))
        labeled2, n2 = ndi.label(bridged)
        counts2 = np.bincount(labeled2.ravel())
        counts2[0] = 0
        both2 = []
        for i in range(1, n2 + 1):
            comp = labeled2 == i
            if (comp & nostril_region).any() and (comp & trachea_region).any():
                both2.append(i)
        if both2:
            best = max(both2, key=lambda i: counts2[i])
            keep = (labeled2 == best) & (
                air | morphology.dilation(air, footprint=morphology.ball(2))
            )
            keep = _clip_superior_to_start(keep, nostril_region, superior_is_high_z)
            info["selection"] = f"bridged_closing_r{radius}"
            return keep.astype(bool), info

    # Geodesic conduit nostrils → neck (caudal only)
    if nostril_region.any():
        nose_pts = np.column_stack(np.where(nostril_region))
        # Start at most inferior nostril seed (already caudal of crown)
        if superior_is_high_z:
            # inferior = low z → pick lower-z nose points
            order = np.argsort(nose_pts[:, 0])
            start = tuple(int(v) for v in nose_pts[order[len(order) // 4]])
        else:
            order = np.argsort(-nose_pts[:, 0])
            start = tuple(int(v) for v in nose_pts[order[len(order) // 4]])

        # Neck target: midline body in neck band
        z_list = list(range(*z_neck.indices(air.shape[0])))
        if not z_list:
            z_list = [0 if superior_is_high_z else air.shape[0] - 1]
        # Pick the most inferior neck slice that still has body
        if superior_is_high_z:
            z_candidates = sorted(z_list)  # low first
        else:
            z_candidates = sorted(z_list, reverse=True)
        end = None
        for z_t in z_candidates:
            ys, xs = np.where(body[z_t, :, x0:x1])
            if len(ys) > 50:
                end = (z_t, int(np.median(ys)), int(np.median(xs)) + x0)
                break
        if end is None:
            z_t = z_candidates[0]
            end = (z_t, body.shape[1] // 2, x_mid)

        tube = _geodesic_airway_bridge(
            hu, body, air, bone, start, end, superior_is_high_z, tube_radius=3
        )
        if nose_only:
            nose_comp = labeled == max(nose_only, key=lambda i: counts[i])
        else:
            # largest air that touches nostrils
            nose_comp = np.zeros_like(air)
            if nostril_region.any():
                for lab_id in range(1, n + 1):
                    comp = labeled == lab_id
                    if (comp & nostril_region).any():
                        if counts[lab_id] > nose_comp.sum():
                            nose_comp = comp
            if not nose_comp.any():
                nose_comp = labeled == int(np.argmax(counts))

        # Exclude pure crown air from fluid domain
        nose_comp = nose_comp & ~(_mostly_crown(nose_comp, z_crown))

        keep = nose_comp | tube
        labk, nk = ndi.label(keep)
        best_k, best_n = None, 0
        for i in range(1, nk + 1):
            comp = labk == i
            if (comp & nostril_region).any() and int(comp.sum()) > best_n:
                # must reach neck band
                if (comp[z_neck]).any() or tube.any():
                    best_n = int(comp.sum())
                    best_k = i
        if best_k is not None:
            out = labk == best_k
            out = _clip_superior_to_start(out, nostril_region, superior_is_high_z)
            _assert_airway_plausible(out, voxel_ml, min_airway_ml, "geodesic_nose_to_neck_caudal")
            info["selection"] = "geodesic_nose_to_neck_caudal"
            info["note"] = (
                "Path constrained caudally to trachea/neck. "
                "Geodesic tube used where free air is discontinuous."
            )
            info["start_zyx"] = list(start)
            info["end_zyx"] = list(end)
            return out.astype(bool), info

    # Fallback: largest mid-face air, clip superior vault
    if nose_only:
        best = max(nose_only, key=lambda i: counts[i])
    else:
        best = int(np.argmax(counts))
    keep = labeled == best
    keep = _clip_superior_to_start(keep, nostril_region if nostril_region.any() else keep, superior_is_high_z)
    _assert_airway_plausible(keep, voxel_ml, min_airway_ml, "fallback_largest_nasal_caudal_clip")
    info["selection"] = "fallback_largest_nasal_caudal_clip"
    info["warning"] = "Limited nose–trachea connection; outlet at caudal tip of path."
    return keep.astype(bool), info


def _assert_airway_plausible(
    mask: np.ndarray, voxel_ml: float, min_airway_ml: float, selection: str
) -> None:
    """Fail loudly if a non-connected selection path collapsed to a ~0 mL airway."""
    vol_ml = float(int(mask.sum()) * voxel_ml)
    if vol_ml < min_airway_ml:
        raise ValueError(
            f"Airway selection '{selection}' produced only {int(mask.sum())} voxels "
            f"({vol_ml:.3f} mL), below the {min_airway_ml:.2f} mL plausibility floor. "
            "This is the silent-failure mode (orientation / seed placement / empty "
            "interior air) — refusing to emit a degenerate airway."
        )


def _mostly_crown(comp: np.ndarray, z_crown: slice) -> np.ndarray:
    crown = np.zeros_like(comp)
    crown[z_crown] = True
    if not comp.any():
        return comp
    # If component is mostly in crown, drop all of it
    if (comp & crown).sum() > 0.6 * comp.sum():
        return comp
    return np.zeros_like(comp)


def _clip_superior_to_start(
    mask: np.ndarray,
    seed_region: np.ndarray,
    superior_is_high_z: bool,
    margin: int = 4,
) -> np.ndarray:
    """Remove voxels superior to the nostril seeds (no upward intracranial path)."""
    if not seed_region.any() or not mask.any():
        return mask
    zz = np.where(seed_region)[0]
    if superior_is_high_z:
        z_max_allowed = int(zz.max()) + margin
        out = mask.copy()
        out[z_max_allowed + 1 :, :, :] = False
    else:
        z_min_allowed = int(zz.min()) - margin
        out = mask.copy()
        out[: max(z_min_allowed, 0), :, :] = False
    # Keep only component still touching seeds
    if seed_region.any():
        lab, n = ndi.label(out)
        keep_ids = []
        for i in range(1, n + 1):
            if ((lab == i) & seed_region).any():
                keep_ids.append(i)
        if keep_ids:
            out = np.isin(lab, keep_ids)
    return out


def _ports_from_edge_nares(
    airway: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    superior_is_high_z: bool,
    y_anterior_is_low: bool,
    naris_L: tuple[int, int, int] | None,
    naris_R: tuple[int, int, int] | None,
    nose_tip: tuple[int, int, int] | None,
) -> tuple[list[Port], list[str], dict[str, Any]]:
    """Build BC ports from edge-detected skin nares + caudal trachea."""
    warnings: list[str] = []
    meta: dict[str, Any] = {
        "nose_tip_zyx": list(nose_tip) if nose_tip else None,
        "y_anterior_is_low": y_anterior_is_low,
    }
    zz, yy, xx = np.where(airway)
    if len(zz) == 0:
        raise ValueError("Empty airway")
    pts = np.column_stack(
        [
            xx * spacing_xyz[0] + origin_xyz[0],
            yy * spacing_xyz[1] + origin_xyz[1],
            zz * spacing_xyz[2] + origin_xyz[2],
        ]
    )
    face = float(np.prod(sorted(spacing_xyz)[:2]))
    if superior_is_high_z:
        tip_idx = zz <= np.percentile(zz, 8)
        trach_normal = [0.0, 0.0, -1.0]
    else:
        tip_idx = zz >= np.percentile(zz, 92)
        trach_normal = [0.0, 0.0, 1.0]
    trach_center = pts[tip_idx].mean(axis=0)
    trach_area = float(max(int(tip_idx.sum()), 1) ** (2 / 3) * face)

    ports: list[Port] = []
    nrm = [0.0, 1.0 if y_anterior_is_low else -1.0, 0.0]
    ox, oy, oz = origin_xyz
    sx, sy, sz = spacing_xyz

    def _add_naris(name: str, zyx: tuple[int, int, int] | None) -> None:
        if zyx is None:
            warnings.append(f"Missing edge-detected {name}")
            return
        iz, iy, ix = zyx
        center = [ox + ix * sx, oy + iy * sy, oz + iz * sz]
        ports.append(
            Port(
                name=name,
                role="inlet",
                center_mm=center,
                area_mm2=80.0,
                normal_xyz=nrm,
                method="edge_nose_tip_skin_naris",
                notes="External naris at nose tip (edge/geometry), not orbits.",
            )
        )
        meta.setdefault("naris_points", []).append(
            {
                "name": name,
                "center_mm": center,
                "skin_voxel_zyx": list(zyx),
                "depth_mm": 0.0,
            }
        )

    _add_naris("left_nostril", naris_L)
    _add_naris("right_nostril", naris_R)
    if len([p for p in ports if p.role == "inlet"]) < 2 and nose_tip is not None:
        # Split tip left/right by a few mm in x
        iz, iy, ix = nose_tip
        _add_naris("left_nostril", (iz, iy, min(ix + 8, airway.shape[2] - 1)))
        _add_naris("right_nostril", (iz, iy, max(ix - 8, 0)))
        warnings.append("Nares inferred by splitting nose tip L/R.")

    ports.append(
        Port(
            name="trachea",
            role="outlet",
            center_mm=trach_center.tolist(),
            area_mm2=trach_area,
            normal_xyz=trach_normal,
            method="whole_head_caudal_airway",
            notes="Caudal airway outlet (trachea direction).",
        )
    )
    return ports, warnings, meta


def detect_ports_whole_head(
    hu: np.ndarray,
    body: np.ndarray,
    airway: np.ndarray,
    interior_air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    superior_is_high_z: bool,
) -> tuple[list[Port], list[str], dict[str, Any]]:
    """
    Ports: external nares on skin + caudal trachea.

    Nostrils use skin-projected external naris positions (not deep vestibule).
    """
    warnings: list[str] = []
    meta: dict[str, Any] = {}
    zz, yy, xx = np.where(airway)
    if len(zz) == 0:
        raise ValueError("Empty airway")

    pts = np.column_stack(
        [
            xx * spacing_xyz[0] + origin_xyz[0],
            yy * spacing_xyz[1] + origin_xyz[1],
            zz * spacing_xyz[2] + origin_xyz[2],
        ]
    )
    face = float(np.prod(sorted(spacing_xyz)[:2]))

    # Trachea = caudal-most air
    if superior_is_high_z:
        z_thr = np.percentile(zz, 8)
        tip_idx = zz <= z_thr
        trach_normal = np.array([0.0, 0.0, -1.0])
    else:
        z_thr = np.percentile(zz, 92)
        tip_idx = zz >= z_thr
        trach_normal = np.array([0.0, 0.0, 1.0])
    trach_center = pts[tip_idx].mean(axis=0)
    trach_area = float(max(int(tip_idx.sum()), 1) ** (2 / 3) * face)

    # External nares on skin
    nares, ninfo = detect_external_nares(
        hu,
        body,
        interior_air,
        spacing_xyz,
        origin_xyz,
        superior_is_high_z=superior_is_high_z,
    )
    meta["nares"] = ninfo
    y_ant_is_low = bool(ninfo.get("y_anterior_is_low", True))

    ports: list[Port] = []
    for naris in nares:
        nrm = np.array([0.0, 1.0 if y_ant_is_low else -1.0, 0.0])
        ports.append(
            Port(
                name=naris.name,
                role="inlet",
                center_mm=naris.center_mm,
                area_mm2=float(max(naris.n_support_voxels, 1) ** (2 / 3) * face),
                normal_xyz=nrm.tolist(),
                method="external_naris_on_skin",
                notes=(
                    f"External naris on skin surface "
                    f"(depth_to_exterior≈{naris.depth_to_exterior_mm:.1f} mm)."
                ),
            )
        )
        meta.setdefault("naris_points", []).append(
            {
                "name": naris.name,
                "center_mm": naris.center_mm,
                "skin_voxel_zyx": naris.skin_voxel_zyx,
                "depth_mm": naris.depth_to_exterior_mm,
            }
        )

    if len(ports) < 2:
        warnings.append(
            "Could not place both external nares on skin; check face air near surface."
        )

    ports.append(
        Port(
            name="trachea",
            role="outlet",
            center_mm=trach_center.tolist(),
            area_mm2=trach_area,
            normal_xyz=trach_normal.tolist(),
            method="whole_head_caudal_airway",
            notes="Caudal airway outlet (trachea / subglottis direction).",
        )
    )
    return ports, warnings, meta


def process_whole_head(
    image_path: Path | str,
    output_dir: Path | str | None = None,
    case_id: str = "VisibleHuman_Head",
    breathing: PatientBreathing | None = None,
    body_hu_min: float = -200.0,
    air_hu_max: float = -300.0,
    mesh_decimate_head: int = 25000,
    mesh_decimate_airway: int = 20000,
    mesh_decimate_skin: int = 30000,
) -> WholeHeadResult:
    image_path = Path(image_path)
    output_dir = Path(output_dir or Path("outputs") / case_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    breathing = breathing or PatientBreathing.typical_resting_adult()
    notes: list[str] = []

    image = sitk.ReadImage(str(image_path))
    hu_full = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = tuple(float(v) for v in image.GetSpacing())
    origin = tuple(float(v) for v in image.GetOrigin())

    superior_is_high_z, orient_method = infer_superior_is_high_z(image, None)
    notes.append(f"Orientation: {orient_method}")
    print(f"[{case_id}] {orient_method}")

    y_ant_low_img, ap_method = infer_y_anterior_is_low_from_image(image)
    notes.append(f"Orientation: {ap_method}")
    print(f"[{case_id}] {ap_method}")

    print(f"[{case_id}] edge-aware tissue + air segmentation (crop shoulders)…")
    seg = run_edge_segmentation(
        hu_full,
        superior_is_high_z=superior_is_high_z,
        body_hu_min=body_hu_min,
        air_hu_max=air_hu_max,
        y_anterior_is_low=y_ant_low_img,  # None -> keep the anatomy inference
    )
    notes.extend(seg.notes)

    # Crop to head bbox (shoulders already removed from body mask)
    bb = seg.head_bbox_zyx
    crop_origin_zyx = [bb[0].start, bb[1].start, bb[2].start]
    hu = hu_full[bb]
    body = seg.body[bb]
    bone = seg.bone[bb]
    labels_crop = seg.labels[bb].copy()
    edge_crop = seg.edge_magnitude[bb]

    # Interior (nasal / sinus / pharyngeal) air. The enclosed-air segmentation
    # (body & low-HU on the hole-filled body) works when the airway is enclosed
    # (e.g. VH cadaver, closed nares). On an OPEN-NARES living scan the nasal lumen
    # is 26-connected to the ambient FOV air through the nostrils (and vents to the
    # FOV cut via the nasopharynx), so it is neither enclosed nor hole-fillable and
    # the mask collapses to a few sealed-sinus voxels (~1e3 → ~0 mL). Detect that
    # collapse in physical mL and recover interior air via the per-axial body-hull
    # method (proven in assess_dicom_incoming.internal_air_ml). Using enclosed air
    # when healthy keeps VH byte-for-byte identical to the prior pipeline.
    voxel_ml = float(np.prod(spacing)) / 1000.0
    air_enclosed = seg.air[bb]
    air_enclosed_ml = float(int(air_enclosed.sum()) * voxel_ml)
    open_nares_min_ml = 5.0
    if air_enclosed_ml >= open_nares_min_ml:
        air_all = air_enclosed
        notes.append(
            f"Interior air from enclosed-air segmentation ({air_enclosed_ml:.1f} mL)."
        )
        print(f"[{case_id}] interior air (enclosed) = {air_enclosed_ml:.1f} mL")
    else:
        air_all = interior_air_within_hull(hu, body, air_hu_max, voxel_ml)
        air_hull_ml = float(int(air_all.sum()) * voxel_ml)
        if not air_all.any():
            raise ValueError(
                f"[{case_id}] Enclosed-air segmentation collapsed to "
                f"{air_enclosed_ml:.2f} mL (open nares?) AND body-hull recovery found "
                f"no interior air (air_hu_max={air_hu_max}). Refusing to emit an empty "
                "airway."
            )
        notes.append(
            f"Enclosed-air segmentation collapsed to {air_enclosed_ml:.2f} mL "
            f"(open-nares living scan); recovered {air_hull_ml:.1f} mL via body-hull."
        )
        print(
            f"[{case_id}] interior air (enclosed={air_enclosed_ml:.2f} mL collapsed) "
            f"→ body-hull recovery = {air_hull_ml:.1f} mL"
        )
    # Keep tissue labels consistent with the interior air actually used.
    labels_crop[labels_crop == 1] = 2  # old AIR label → SOFT
    labels_crop[air_all] = 1  # AIR

    # Remap landmark coords into crop space
    def _to_crop(zyx: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if zyx is None:
            return None
        return (
            zyx[0] - crop_origin_zyx[0],
            zyx[1] - crop_origin_zyx[1],
            zyx[2] - crop_origin_zyx[2],
        )

    nose_tip_c = _to_crop(seg.nose_tip_zyx)
    naris_L_c = _to_crop(seg.naris_left_zyx)
    naris_R_c = _to_crop(seg.naris_right_zyx)

    origin_crop = (
        origin[0] + crop_origin_zyx[2] * spacing[0],
        origin[1] + crop_origin_zyx[1] * spacing[1],
        origin[2] + crop_origin_zyx[0] * spacing[2],
    )

    print(
        f"[{case_id}] head crop zyx={hu.shape} body={int(body.sum()):,} "
        f"air={int(air_all.sum()):,} bone={int(bone.sum()):,} "
        f"shoulder_z0={seg.shoulder_crop_z0}"
    )
    print(f"[{case_id}] nose_tip={nose_tip_c} naris_L={naris_L_c} naris_R={naris_R_c}")
    print(f"[{case_id}] selecting nasal→trachea path (caudal)…")
    airway, path_info = select_nasal_to_trachea_path(
        air_all,
        body,
        hu,
        bone,
        superior_is_high_z=superior_is_high_z,
        y_anterior_is_low=seg.y_anterior_is_low,
        spacing_xyz=spacing,
    )
    notes.append(f"Airway selection: {path_info.get('selection')}")
    if path_info.get("note"):
        notes.append(path_info["note"])
    if path_info.get("warning"):
        notes.append(path_info["warning"])
    notes.append(
        "Mouth closed: domain is nasal path to caudal outlet; oral cavity excluded when separable."
    )
    notes.append(
        "Edge-aware tissue classes: exterior=0, air=1, soft_tissue=2, cartilage=3, bone=4."
    )
    notes.append("Shoulders cropped from body mask using inferior cross-section filter.")

    def _write_arr(arr: np.ndarray, name: str, dtype=np.uint8) -> Path:
        img = sitk.GetImageFromArray(arr.astype(dtype))
        img.SetSpacing(spacing)
        img.SetOrigin(origin_crop)
        img.SetDirection(image.GetDirection())
        path = output_dir / name
        sitk.WriteImage(img, str(path))
        return path

    _write_arr(body.astype(np.uint8), f"{case_id}_head_mask.nrrd")
    _write_arr(airway.astype(np.uint8), f"{case_id}_airway_mask.nrrd")
    _write_arr(air_all.astype(np.uint8), f"{case_id}_all_interior_air.nrrd")
    _write_arr(labels_crop, f"{case_id}_tissues.nrrd", dtype=np.int16)
    _write_arr(bone.astype(np.uint8), f"{case_id}_bone_mask.nrrd")
    _write_arr((labels_crop == 2).astype(np.uint8), f"{case_id}_soft_tissue_mask.nrrd")
    _write_arr((edge_crop * 255).astype(np.uint8), f"{case_id}_edges.nrrd")

    print(f"[{case_id}] meshing skin surface + head solid + airway…")
    skin_shell = extract_skin_shell(body, thickness=1)
    _write_arr(skin_shell.astype(np.uint8), f"{case_id}_skin_shell_mask.nrrd")

    try:
        # Slightly less smoothing keeps facial features (nose) readable
        skin_mesh = mesh_skin_surface(
            body, spacing, origin_crop, smooth_sigma=0.55, level=0.45
        )
        skin_mesh = _decimate(skin_mesh, mesh_decimate_skin)
        skin_stl = output_dir / f"{case_id}_skin.stl"
        skin_mesh.export(skin_stl)
        notes.append(
            f"Skin surface mesh: {len(skin_mesh.faces):,} faces (head-only, shoulders cropped)."
        )
        print(f"[{case_id}] skin mesh faces={len(skin_mesh.faces):,}")
    except Exception as exc:
        skin_stl = None
        notes.append(f"Skin mesh failed: {exc}")
        print(f"[{case_id}] skin mesh failed: {exc}")

    head_mesh = _decimate(_mask_to_mesh(body, spacing, origin_crop), mesh_decimate_head)
    airway_mesh_full = _mask_to_mesh(airway, spacing, origin_crop)
    airway_mesh = _decimate(airway_mesh_full, mesh_decimate_airway)
    if bone.any():
        try:
            bone_mesh = _decimate(_mask_to_mesh(bone, spacing, origin_crop), 15000)
            bone_mesh.export(output_dir / f"{case_id}_bone.stl")
        except Exception as exc:
            notes.append(f"Bone mesh skipped: {exc}")

    head_stl = output_dir / f"{case_id}_head.stl"
    airway_stl = output_dir / f"{case_id}_airway.stl"
    head_mesh.export(head_stl)
    airway_mesh.export(airway_stl)
    airway_mesh_full.export(output_dir / f"{case_id}_airway_full.stl")

    # Ports from edge-detected external nares + caudal trachea
    ports, port_warnings, port_meta = _ports_from_edge_nares(
        airway=airway,
        spacing_xyz=spacing,
        origin_xyz=origin_crop,
        superior_is_high_z=superior_is_high_z,
        y_anterior_is_low=seg.y_anterior_is_low,
        naris_L=naris_L_c,
        naris_R=naris_R_c,
        nose_tip=nose_tip_c,
    )
    notes.extend(port_warnings)
    for p in ports:
        if p.role == "inlet":
            notes.append(
                f"Inlet {p.name} at skin: {[round(c,1) for c in p.center_mm]} ({p.method})"
            )
    ports = tag_mesh_faces_near_ports(airway_mesh, ports, radius_mm=16.0)
    for p in ports:
        if p.role == "outlet" and p.n_faces == 0:
            ports = tag_mesh_faces_near_ports(airway_mesh, ports, radius_mm=26.0)
            break

    flow = assign_flow(ports, breathing)
    outlet_proxy = str(path_info.get("selection", "")).startswith("fallback")
    setup = BoundarySetup(
        case_id=case_id,
        mouth="closed — oral cavity excluded from nasal→trachea domain when separable",
        inlet_names=[p.name for p in ports if p.role == "inlet"],
        outlet_name=next((p.name for p in ports if p.role == "outlet"), "trachea"),
        outlet_is_proxy=outlet_proxy,
        ports=ports,
        breathing=breathing.to_dict(),
        flow_assignment=flow,
        mesh_path=str(airway_stl),
        warnings=port_warnings
        + ([path_info["warning"]] if path_info.get("warning") else [])
        + ([path_info["note"]] if path_info.get("note") else []),
    )
    write_boundary_setup(setup, output_dir / f"{case_id}_boundary_conditions.json")
    write_openfoam_bc_notes(setup, output_dir / f"{case_id}_openfoam_bc_sketch.txt")
    try:
        export_port_markers_ply(ports, output_dir / f"{case_id}_port_markers.ply")
    except ValueError:
        pass
    with (output_dir / f"{case_id}_nares.json").open("w", encoding="utf-8") as f:
        json.dump(port_meta, f, indent=2)

    _save_whole_head_preview(
        hu,
        body,
        airway,
        bone,
        output_dir / f"{case_id}_preview.png",
        case_id,
        superior_is_high_z=superior_is_high_z,
        naris_points=port_meta.get("naris_points"),
        skin_shell=skin_shell,
    )

    # Direction QC: airway z should extend more caudally than cranially from centroid
    az = np.where(airway)[0]
    if len(az):
        z_mean = float(az.mean())
        z_min, z_max = float(az.min()), float(az.max())
        if superior_is_high_z:
            caudal_extent = z_mean - z_min
            cranial_extent = z_max - z_mean
        else:
            caudal_extent = z_max - z_mean
            cranial_extent = z_mean - z_min
        notes.append(
            f"Airway z extent: caudal={caudal_extent:.1f} slices, "
            f"cranial={cranial_extent:.1f} slices (caudal should dominate)."
        )
        if cranial_extent > caudal_extent * 1.2:
            notes.append(
                "WARNING: airway still extends more cranially than caudally — check seeds."
            )

    result = WholeHeadResult(
        case_id=case_id,
        image_path=str(image_path),
        size_xyz=list(image.GetSize()),
        spacing_xyz_mm=list(spacing),
        origin_xyz_mm=list(origin_crop),
        crop_origin_zyx=crop_origin_zyx,
        head_voxels=int(body.sum()),
        airway_voxels=int(airway.sum()),
        airway_volume_ml=float(airway.sum() * voxel_ml),
        head_mesh_faces=int(len(head_mesh.faces)),
        airway_mesh_faces=int(len(airway_mesh.faces)),
        outlet_is_proxy=outlet_proxy,
        superior_is_high_z=superior_is_high_z,
        notes=notes,
    )
    with (output_dir / f"{case_id}_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **result.to_dict(),
                "tissue_labels": TISSUE_LABELS,
                "path_info": {
                    k: v
                    for k, v in path_info.items()
                    if not isinstance(v, slice)
                },
                "boundary_summary": {
                    "inlets": setup.inlet_names,
                    "outlet": setup.outlet_name,
                    "outlet_is_proxy": outlet_proxy,
                    "Q_L_per_min": breathing.mean_inspiratory_flow_L_per_min,
                    "Ti_s": breathing.Ti_s,
                },
            },
            f,
            indent=2,
        )

    print(summary_text(breathing))
    print(
        f"[{case_id}] head={result.head_voxels:,}  airway={result.airway_voxels:,} "
        f"({result.airway_volume_ml:.1f} mL)  superior_is_high_z={superior_is_high_z}"
    )
    print(f"[{case_id}] wrote tissues, head STL, airway STL, BCs → {output_dir}")
    return result


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        idx = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
        return trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[idx], process=False
        )


def _save_whole_head_preview(
    hu: np.ndarray,
    body: np.ndarray,
    airway: np.ndarray,
    bone: np.ndarray,
    path: Path,
    case_id: str,
    superior_is_high_z: bool,
    naris_points: list[dict] | None = None,
    skin_shell: np.ndarray | None = None,
) -> None:
    z, y, x = hu.shape
    # Prefer axial through naris z if available
    if naris_points:
        zs = [p["skin_voxel_zyx"][0] for p in naris_points]
        ys = [p["skin_voxel_zyx"][1] for p in naris_points]
        xs = [p["skin_voxel_zyx"][2] for p in naris_points]
        mid = (int(np.mean(zs)), int(np.mean(ys)), int(np.mean(xs)))
    elif airway.any():
        az, ay, ax = np.where(airway)
        mid = (int(np.median(az)), int(np.median(ay)), int(np.median(ax)))
    else:
        mid = (z // 2, y // 2, x // 2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    views = [
        ("axial @ nares", 0, hu[mid[0]], body[mid[0]], airway[mid[0]], bone[mid[0]],
         skin_shell[mid[0]] if skin_shell is not None else None),
        ("coronal", 1, hu[:, mid[1], :], body[:, mid[1], :], airway[:, mid[1], :], bone[:, mid[1], :],
         skin_shell[:, mid[1], :] if skin_shell is not None else None),
        ("sagittal", 2, hu[:, :, mid[2]], body[:, :, mid[2]], airway[:, :, mid[2]], bone[:, :, mid[2]],
         skin_shell[:, :, mid[2]] if skin_shell is not None else None),
    ]
    for ax, (title, axis, img, b, a, bn, sk) in zip(axes, views):
        disp = np.clip(img, -200, 600)
        ax.imshow(disp, cmap="gray", origin="lower")
        if sk is not None and sk.any():
            ax.contour(sk.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
        else:
            ax.contour(b.astype(float), levels=[0.5], colors=["#4fc3f7"], linewidths=0.7)
        if bn is not None and bn.any():
            ax.contour(bn.astype(float), levels=[0.5], colors=["#f5f5f5"], linewidths=0.35, alpha=0.6)
        ov = np.ma.masked_where(~a, a.astype(float))
        ax.imshow(ov, cmap="autumn", alpha=0.45, origin="lower")
        # Mark external nares
        if naris_points:
            for p in naris_points:
                iz, iy, ix = p["skin_voxel_zyx"]
                if axis == 0 and abs(iz - mid[0]) <= 3:
                    ax.plot(ix, iy, "o", color="#00ff66", markersize=8, markeredgecolor="white")
                    ax.text(ix + 2, iy, p["name"].replace("_", "\n"), color="#00ff66", fontsize=7)
                elif axis == 1 and abs(iy - mid[1]) <= 3:
                    ax.plot(ix, iz, "o", color="#00ff66", markersize=8, markeredgecolor="white")
                elif axis == 2 and abs(ix - mid[2]) <= 3:
                    ax.plot(iy, iz, "o", color="#00ff66", markersize=8, markeredgecolor="white")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"{case_id} — skin (cyan), airway (red), external nares (green)  |  "
        f"superior_is_high_z={superior_is_high_z}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Extra face close-up axial for naris QC
    if naris_points:
        fig2, ax = plt.subplots(1, 1, figsize=(6, 6))
        iz = mid[0]
        disp = np.clip(hu[iz], -200, 600)
        ax.imshow(disp, cmap="gray", origin="lower")
        if skin_shell is not None:
            ax.contour(skin_shell[iz].astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.2)
        ov = np.ma.masked_where(~airway[iz], airway[iz].astype(float))
        ax.imshow(ov, cmap="autumn", alpha=0.4, origin="lower")
        for p in naris_points:
            _, iy, ix = p["skin_voxel_zyx"]
            ax.plot(ix, iy, "o", color="#00ff66", markersize=12, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(
                p["name"],
                (ix, iy),
                textcoords="offset points",
                xytext=(6, 6),
                color="#00ff66",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(f"{case_id} face axial — external nares on skin")
        ax.axis("off")
        face_path = path.with_name(path.stem + "_face_nares.png")
        fig2.savefig(face_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
