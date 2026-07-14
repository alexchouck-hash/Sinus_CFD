"""
Outer skin surface mesh and true external nostril (naris) placement.

Nostrils are placed where internal nasal air comes closest to exterior free air
on the anterior mid-face, then projected onto the outer skin surface. This
fixes ports that previously sat too deep inside the vestibule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import gaussian

from .pipeline import _mask_to_mesh


@dataclass
class Naris:
    name: str  # left_nostril / right_nostril
    center_mm: list[float]
    center_zyx: list[int]
    skin_voxel_zyx: list[int]
    n_support_voxels: int
    depth_to_exterior_mm: float


def body_unfilled_and_filled(
    hu: np.ndarray,
    body_hu_min: float = -200.0,
    min_component: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unfilled_body, filled_body) boolean masks."""
    seed = hu > body_hu_min
    seed = morphology.opening(seed, footprint=morphology.ball(1))
    labeled, n = ndi.label(seed)
    if n == 0:
        raise ValueError("No body found")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts >= min_component
    keep[0] = False
    if not keep[1:].any():
        keep[int(np.argmax(counts))] = True
    unfilled = keep[labeled]
    filled = ndi.binary_fill_holes(unfilled)
    filled = morphology.closing(filled, footprint=morphology.ball(1))
    filled = ndi.binary_fill_holes(filled)
    return unfilled.astype(bool), filled.astype(bool)


def extract_skin_shell(
    body_filled: np.ndarray,
    thickness: int = 1,
) -> np.ndarray:
    """
    Outer skin shell voxels: body surface facing exterior only.

    Uses filled body so internal cavities (sinuses) are not part of the shell.
    """
    eroded = morphology.erosion(body_filled, footprint=morphology.ball(thickness))
    shell = body_filled & ~eroded
    return shell.astype(bool)


def mesh_skin_surface(
    body_filled: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    smooth_sigma: float = 0.8,
    level: float = 0.5,
) -> Any:
    """
    Smooth outer skin isosurface (trimesh) from filled body mask.

    Gaussian smoothing of the binary mask yields a cleaner skin surface for
    visualization than raw voxel stair-steps.
    """
    # float mask for smooth surface
    vol = body_filled.astype(np.float32)
    if smooth_sigma and smooth_sigma > 0:
        vol = gaussian(vol, sigma=smooth_sigma, preserve_range=True)
    # Marching cubes via shared helper expects binary-ish mask; use thresholded smooth field
    mask = vol >= level
    # Ensure single outer component
    labeled, n = ndi.label(mask)
    if n > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        mask = labeled == int(np.argmax(counts))
    mesh = _mask_to_mesh(mask, spacing_xyz, origin_xyz)
    return mesh


def detect_external_nares(
    hu: np.ndarray,
    body_filled: np.ndarray,
    interior_air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    superior_is_high_z: bool,
    max_depth_mm: float = 12.0,
    face_z_fraction: tuple[float, float] = (0.28, 0.78),
) -> tuple[list[Naris], dict[str, Any]]:
    """
    Locate left/right external nostrils on the skin.

    Method:
      1. Distance-to-exterior for every body voxel.
      2. Keep interior air within max_depth_mm of exterior (near openings).
      3. Restrict to mid-face band (not crown, not neck).
      4. Keep the most anterior air cluster(s); split L/R by x.
      5. Project each naris center to the nearest outer skin voxel
         (most exterior point of the cluster, then snap to skin shell).
    """
    info: dict[str, Any] = {}
    sx, sy, sz = spacing_xyz
    # distance in voxels → mm (approx isotropic mean spacing)
    sp_mean = float(np.mean(spacing_xyz))
    max_depth_vox = max(max_depth_mm / sp_mean, 2.0)

    exterior = ~body_filled
    dist_vox = ndi.distance_transform_edt(body_filled)
    skin = extract_skin_shell(body_filled, thickness=1)

    # Air near exterior = candidate vestibule / naris air
    near = interior_air & (dist_vox <= max_depth_vox)
    info["near_surface_air_voxels"] = int(near.sum())
    if not near.any():
        # Fall back: shallowest air overall
        if not interior_air.any():
            return [], {**info, "error": "no interior air"}
        air_dist = dist_vox.copy()
        air_dist[~interior_air] = 1e9
        thr = np.percentile(air_dist[interior_air], 5)
        near = interior_air & (air_dist <= thr + 1)
        info["near_surface_air_voxels_fallback"] = int(near.sum())

    n_z, n_y, n_x = body_filled.shape
    z0 = int(n_z * face_z_fraction[0])
    z1 = int(n_z * face_z_fraction[1])
    if superior_is_high_z:
        # high z = superior crown → face is mid band already
        face_band = np.zeros_like(near)
        face_band[z0:z1, :, :] = True
    else:
        face_band = np.zeros_like(near)
        face_band[z0:z1, :, :] = True
    near_face = near & face_band
    if near_face.sum() < 20:
        near_face = near
    info["near_face_air_voxels"] = int(near_face.sum())

    # Anterior direction: which y has smaller exterior distance for face air
    # (true nares sit on the front of the face)
    zz, yy, xx = np.where(near_face)
    if len(zz) == 0:
        return [], {**info, "error": "no near-face air"}

    # Anterior = body y-extreme closer to nasal air (not free-air mass outside head).
    # Nose sits near the face surface; occiput is the opposite y-extreme.
    body_y = np.where(body_filled)[1]
    y_body_low, y_body_high = int(body_y.min()), int(body_y.max())
    air_y_mean = float(yy.mean())
    y_ant_is_low = abs(air_y_mean - y_body_low) <= abs(air_y_mean - y_body_high)
    info["air_y_mean"] = air_y_mean
    info["body_y_range"] = [y_body_low, y_body_high]

    # Keep only the anterior third of the head for naris search
    span = max(y_body_high - y_body_low, 1)
    if y_ant_is_low:
        y_face_cut = y_body_low + int(0.32 * span)
        anterior = near_face & (np.arange(n_y)[None, :, None] <= y_face_cut)
    else:
        y_face_cut = y_body_high - int(0.32 * span)
        anterior = near_face & (np.arange(n_y)[None, :, None] >= y_face_cut)

    if anterior.sum() < 15:
        # Fall back: most anterior 25% of near_face air by y percentile
        if y_ant_is_low:
            thr_y = np.percentile(yy, 25)
            anterior = near_face.copy()
            anterior[:, int(thr_y) + 1 :, :] = False
        else:
            thr_y = np.percentile(yy, 75)
            anterior = near_face.copy()
            anterior[:, : int(thr_y), :] = False
    info["y_anterior_is_low"] = bool(y_ant_is_low)
    info["y_face_cut"] = int(y_face_cut)
    info["anterior_air_voxels"] = int(anterior.sum())

    # Among anterior near-surface air, keep shallowest (closest to exterior)
    d = dist_vox.copy()
    d[~anterior] = 1e9
    if not anterior.any():
        return [], {**info, "error": "no anterior near-surface air"}
    d_thr = np.percentile(d[anterior], 40)
    shallow = anterior & (dist_vox <= d_thr + 0.5)
    if shallow.sum() < 10:
        shallow = anterior

    # Split left/right by median x of shallow air
    szz, syy, sxx = np.where(shallow)
    x_med = float(np.median(sxx))
    left = shallow & (np.arange(n_x)[None, None, :] >= x_med)
    right = shallow & (np.arange(n_x)[None, None, :] < x_med)

    nares: list[Naris] = []
    for name, mask in (("left_nostril", left), ("right_nostril", right)):
        if mask.sum() < 3:
            info.setdefault("warnings", []).append(f"sparse support for {name}")
            # try full shallow on that side of all interior air
            if name.startswith("left"):
                mask = shallow & (np.arange(n_x)[None, None, :] >= x_med)
            else:
                mask = shallow & (np.arange(n_x)[None, None, :] < x_med)
        if mask.sum() < 1:
            continue
        # Most exterior voxel in cluster = min dist_to_exterior
        md = dist_vox.copy()
        md[~mask] = 1e9
        # pick the set of shallowest voxels
        min_d = float(md[mask].min())
        core = mask & (dist_vox <= min_d + 1.0)
        cz = int(np.median(np.where(core)[0]))
        cy = int(np.median(np.where(core)[1]))
        cx = int(np.median(np.where(core)[2]))

        # Project to skin: from core center, walk toward exterior along -grad(dist)
        # or nearest skin voxel among those more exterior
        skin_zyx = _project_to_skin((cz, cy, cx), skin, dist_vox, y_ant_is_low)

        ox, oy, oz = origin_xyz
        sx_, sy_, sz_ = spacing_xyz
        iz, iy, ix = skin_zyx
        center_mm = [
            float(ox + ix * sx_),
            float(oy + iy * sy_),
            float(oz + iz * sz_),
        ]
        nares.append(
            Naris(
                name=name,
                center_mm=center_mm,
                center_zyx=[cz, cy, cx],
                skin_voxel_zyx=list(skin_zyx),
                n_support_voxels=int(mask.sum()),
                depth_to_exterior_mm=float(min_d * sp_mean),
            )
        )

    info["n_nares"] = len(nares)
    return nares, info


def _project_to_skin(
    start_zyx: tuple[int, int, int],
    skin: np.ndarray,
    dist_vox: np.ndarray,
    y_anterior_is_low: bool,
    max_steps: int = 40,
) -> tuple[int, int, int]:
    """
    Move from an internal point toward the exterior until hitting the skin shell.
    Prefer anterior direction if gradient is ambiguous.
    """
    z, y, x = [int(v) for v in start_zyx]
    shape = skin.shape
    # Walk downhill on distance_to_exterior (toward 0)
    for _ in range(max_steps):
        if (
            0 <= z < shape[0]
            and 0 <= y < shape[1]
            and 0 <= x < shape[2]
            and skin[z, y, x]
        ):
            return z, y, x
        # 6-neighborhood step to lower dist
        best = None
        best_d = dist_vox[z, y, x] if 0 <= z < shape[0] else 1e9
        for dz, dy, dx in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            nz, ny, nx = z + dz, y + dy, x + dx
            if not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                continue
            d = float(dist_vox[nz, ny, nx])
            # Prefer anterior move slightly
            bias = 0.0
            if y_anterior_is_low and dy < 0:
                bias = -0.15
            elif (not y_anterior_is_low) and dy > 0:
                bias = -0.15
            score = d + bias
            if score < best_d:
                best_d = score
                best = (nz, ny, nx)
        if best is None or best == (z, y, x):
            break
        z, y, x = best

    # Nearest skin voxel fallback
    skin_pts = np.column_stack(np.where(skin))
    if len(skin_pts) == 0:
        return start_zyx
    d2 = ((skin_pts - np.array(start_zyx)) ** 2).sum(axis=1)
    return tuple(int(v) for v in skin_pts[int(np.argmin(d2))])


def voxel_to_mm(
    zyx: tuple[int, int, int] | list[int],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> list[float]:
    iz, iy, ix = zyx
    ox, oy, oz = origin_xyz
    sx, sy, sz = spacing_xyz
    return [float(ox + ix * sx), float(oy + iy * sy), float(oz + iz * sz)]
