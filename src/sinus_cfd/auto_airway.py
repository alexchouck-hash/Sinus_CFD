"""Automatic L/R nasal assignment without a naris-mid sagittal plane.

See docs/segmentation_strategy.md. This is the geometric teacher:
competing geodesic flood, bone-first choanal landmark, through-path
sinus strip, EDT septum ridge. No operator labels.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

# v1 constants (mm / mL / HU). Names match docs/segmentation_strategy.md.
AIR_HU_MAX_FLOOD = -300.0
AIR_HU_MAX_VESTIBULE = -120.0
NARIS_SNAP_RADIUS_MM = 10.0
NASAL_BOX_POSTERIOR_MM = 90.0
NASAL_BOX_Z_HALF_MM = 35.0
RIDGE_DILATE_MM = 3.0
MIDSURFACE_TAU_MM = 0.5
MIDSURFACE_DMAX_MM = 8.0
OSTIUM_NECK_MM = 2.5
GATE_SLAB_MM = 6.0
PALATE_MIN_ML = 0.5
PALATE_ANTERIOR_OF_NARIS_MM = 10.0
SPACING_ISOTROPIC_RATIO_MAX = 1.05
PALATE_RIDGE_DISAGREE_MM = 15.0
THROUGH_PATH_SLACK_MM = 4.0
BONE_HU_MIN = 300.0


def _spacing_zyx(spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    return sz, sy, sx


def _offsets_26(spacing_xyz: tuple[float, float, float]) -> list[tuple[int, int, int, float]]:
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    out: list[tuple[int, int, int, float]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                length = float(np.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2))
                out.append((dz, dy, dx, length))
    return out


def _isotropic(spacing_xyz: tuple[float, float, float]) -> bool:
    s = np.array(spacing_xyz, dtype=float)
    return float(s.max() / max(s.min(), 1e-9)) <= SPACING_ISOTROPIC_RATIO_MAX


def snap_seed_to_air(
    air: np.ndarray,
    seed_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    radius_mm: float = NARIS_SNAP_RADIUS_MM,
) -> tuple[int, int, int] | None:
    """Nearest air voxel within a physical ball. None if the seed cannot snap."""
    z, y, x = (int(seed_zyx[0]), int(seed_zyx[1]), int(seed_zyx[2]))
    nz, ny, nx = air.shape
    if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx and air[z, y, x]:
        return (z, y, x)
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    rz = max(int(np.ceil(radius_mm / max(sz, 1e-6))), 1)
    ry = max(int(np.ceil(radius_mm / max(sy, 1e-6))), 1)
    rx = max(int(np.ceil(radius_mm / max(sx, 1e-6))), 1)
    best: tuple[float, tuple[int, int, int]] | None = None
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                q = (z + dz, y + dy, x + dx)
                if not (0 <= q[0] < nz and 0 <= q[1] < ny and 0 <= q[2] < nx):
                    continue
                if not air[q]:
                    continue
                dist = float(np.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2))
                if dist > radius_mm:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, q)
    return None if best is None else best[1]


def geodesic_distance_mm(
    air: np.ndarray,
    seed_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Spacing-aware 26-connected geodesic distance (mm) from one seed in ``air``."""
    dist = np.full(air.shape, np.inf, dtype=np.float64)
    seed = (int(seed_zyx[0]), int(seed_zyx[1]), int(seed_zyx[2]))
    if not air[seed]:
        return dist
    dist[seed] = 0.0
    heap: list[tuple[float, int, int, int]] = [(0.0, seed[0], seed[1], seed[2])]
    offsets = _offsets_26(spacing_xyz)
    nz, ny, nx = air.shape
    while heap:
        d, z, y, x = heapq.heappop(heap)
        if d > dist[z, y, x]:
            continue
        for dz, dy, dx, length in offsets:
            z2, y2, x2 = z + dz, y + dy, x + dx
            if not (0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx):
                continue
            if not air[z2, y2, x2]:
                continue
            nd = d + length
            if nd < dist[z2, y2, x2]:
                dist[z2, y2, x2] = nd
                heapq.heappush(heap, (nd, z2, y2, x2))
    return dist


def competing_naris_flood(
    air: np.ndarray,
    left_seed_zyx: tuple[int, int, int],
    right_seed_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assign every air voxel to the nearer naris (geodesic in air).

    Returns (left_air, right_air, d_L_mm, d_R_mm, notes).
    Ties (d_L <= d_R) go left. Independent binary_propagation is not used.
    """
    notes: list[str] = []
    air = air.astype(bool)
    d_l = geodesic_distance_mm(air, left_seed_zyx, spacing_xyz)
    d_r = geodesic_distance_mm(air, right_seed_zyx, spacing_xyz)
    reachable = np.isfinite(d_l) | np.isfinite(d_r)
    left = reachable & (d_l <= d_r)
    right = reachable & ~left
    notes.append(
        f"competing geodesic flood: left={int(left.sum())} right={int(right.sum())} "
        f"unreached={int(air.sum()) - int(reachable.sum())} "
        f"isotropic_shortcut={_isotropic(spacing_xyz)}"
    )
    return left, right, d_l, d_r, notes


def nasal_box_mask(
    shape: tuple[int, int, int],
    left_seed: tuple[int, int, int],
    right_seed: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    y_anterior_is_low: bool,
) -> np.ndarray:
    """Crop the graph for speed. Not a laterality prior."""
    nz, ny, nx = shape
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    z_mid = 0.5 * (left_seed[0] + right_seed[0])
    y_face = 0.5 * (left_seed[1] + right_seed[1])
    z_half = max(int(round(NASAL_BOX_Z_HALF_MM / max(sz, 1e-6))), 4)
    y_post = max(int(round(NASAL_BOX_POSTERIOR_MM / max(sy, 1e-6))), 8)
    z0, z1 = max(0, int(z_mid) - z_half), min(nz, int(z_mid) + z_half + 1)
    if y_anterior_is_low:
        y0, y1 = max(0, int(y_face) - 2), min(ny, int(y_face) + y_post)
    else:
        y0, y1 = max(0, int(y_face) - y_post), min(ny, int(y_face) + 3)
    box = np.zeros(shape, dtype=bool)
    box[z0:z1, y0:y1, :] = True
    return box


def posterior_air_seed(
    air: np.ndarray,
    y_anterior_is_low: bool,
) -> tuple[int, int, int] | None:
    zz, yy, xx = np.where(air)
    if len(zz) == 0:
        return None
    y_post = int(yy.max()) if y_anterior_is_low else int(yy.min())
    sel = yy == y_post
    return (
        int(np.round(zz[sel].mean())),
        y_post,
        int(np.round(xx[sel].mean())),
    )


def choanal_landmark_from_bone(
    hu: np.ndarray,
    air: np.ndarray,
    left_seed_zyx: tuple[int, int, int],
    right_seed_zyx: tuple[int, int, int],
    *,
    y_anterior_is_low: bool,
    superior_is_high_z: bool,
    spacing_xyz: tuple[float, float, float],
    meeting_set: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Bone-first coronal landmark + posterior merge zone (half-space)."""
    notes: list[str] = []
    nz, ny, nx = hu.shape
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    voxel_ml = float(np.prod(spacing_xyz)) / 1000.0
    box = nasal_box_mask(hu.shape, left_seed_zyx, right_seed_zyx, spacing_xyz, y_anterior_is_low)
    z_mid = int(round(0.5 * (left_seed_zyx[0] + right_seed_zyx[0])))
    y_face = int(round(0.5 * (left_seed_zyx[1] + right_seed_zyx[1])))
    x_mid = int(round(0.5 * (left_seed_zyx[2] + right_seed_zyx[2])))

    # Inferior half of the nasal box.
    if superior_is_high_z:
        inferior = np.zeros(hu.shape, dtype=bool)
        inferior[: z_mid + 1] = True
    else:
        inferior = np.zeros(hu.shape, dtype=bool)
        inferior[z_mid:] = True
    bone = box & inferior & (hu >= BONE_HU_MIN)
    lab, n = ndi.label(bone)
    landmark_y: int | None = None
    palate_ok = False
    if n:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        order = np.argsort(counts)[::-1]
        min_vox = max(int(round(PALATE_MIN_ML / max(voxel_ml, 1e-9))), 20)
        y_naris = y_face
        for cid in order:
            if counts[cid] < min_vox:
                break
            comp = lab == int(cid)
            yy = np.where(comp)[1]
            y_post = int(yy.max()) if y_anterior_is_low else int(yy.min())
            anterior_of_naris = (
                (y_naris - y_post) * sy if y_anterior_is_low else (y_post - y_naris) * sy
            )
            # "Anterior of naris" is missing-palate if the plate sits in front of the face.
            if y_anterior_is_low:
                too_anterior = y_post < y_naris - (PALATE_ANTERIOR_OF_NARIS_MM / max(sy, 1e-6))
            else:
                too_anterior = y_post > y_naris + (PALATE_ANTERIOR_OF_NARIS_MM / max(sy, 1e-6))
            if too_anterior:
                notes.append(f"skip bone CC {cid}: posterior-y anterior of naris")
                continue
            landmark_y = y_post
            palate_ok = True
            notes.append(f"palate CC {cid} vox={int(counts[cid])} landmark_y={landmark_y}")
            break

    # Vomer hunt: central x strip, take posterior bone edge.
    x_win = max(int(round(15.0 / max(sx, 1e-6))), 3)
    vomer = box & (hu >= BONE_HU_MIN)
    vomer[:, :, : max(0, x_mid - x_win)] = False
    vomer[:, :, min(nx, x_mid + x_win + 1) :] = False
    vomer_y: int | None = None
    if vomer.any():
        yy = np.where(vomer)[1]
        vomer_y = int(yy.max()) if y_anterior_is_low else int(yy.min())
        notes.append(f"vomer posterior y={vomer_y}")

    if landmark_y is not None and vomer_y is not None:
        if y_anterior_is_low:
            landmark_y = max(landmark_y, vomer_y)
        else:
            landmark_y = min(landmark_y, vomer_y)

    if landmark_y is None and meeting_set is not None and meeting_set.any():
        yy = np.where(meeting_set)[1]
        # Posterior cluster of the meeting set.
        landmark_y = int(np.percentile(yy, 90 if y_anterior_is_low else 10))
        notes.append(f"landmark from meeting-set posterior y={landmark_y} (WARN missing bone)")
        palate_ok = False

    landmark = np.zeros(hu.shape, dtype=bool)
    merge = np.zeros(hu.shape, dtype=bool)
    if landmark_y is None:
        notes.append("WARN: no choanal landmark")
        meta = {"landmark_y": None, "palate_ok": False, "notes": notes}
        return landmark, merge, meta

    slab = max(int(round(GATE_SLAB_MM / max(sy, 1e-6))), 1)
    y0 = max(0, landmark_y - slab // 2)
    y1 = min(ny, landmark_y + slab // 2 + 1)
    landmark[:, y0:y1, :] = box[:, y0:y1, :]
    if y_anterior_is_low:
        merge[:, landmark_y:, :] = True
    else:
        merge[:, : landmark_y + 1, :] = True
    merge &= air
    # Palate CC often runs to the nasal-box posterior face; that yields an empty
    # merge zone. Fall back to the posterior third of remaining air.
    if int(merge.sum()) < 50 and air.any():
        yy = np.where(air)[1]
        y_cut = int(np.percentile(yy, 75 if y_anterior_is_low else 25))
        merge = np.zeros_like(air)
        if y_anterior_is_low:
            merge[:, y_cut:, :] = True
        else:
            merge[:, : y_cut + 1, :] = True
        merge &= air
        landmark_y = y_cut
        notes.append(f"WARN: bone merge empty; fallback landmark_y={landmark_y} from air p75")
    notes.append(f"merge zone voxels={int(merge.sum())} landmark_y={landmark_y}")
    meta = {
        "landmark_y": int(landmark_y),
        "palate_ok": bool(palate_ok),
        "vomer_y": vomer_y,
        "notes": notes,
    }
    return landmark, merge, meta


def through_path_passage(
    air: np.ndarray,
    naris_seeds: list[tuple[int, int, int]],
    outlet_seed: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    slack_mm: float = THROUGH_PATH_SLACK_MM,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep voxels on a naris→outlet geodesic (plus slack). Detours = sinus.

    A thin valve on the through-path stays. A maxillary pocket is a detour.
    """
    notes: list[str] = []
    air = air.astype(bool)
    d_out = geodesic_distance_mm(air, outlet_seed, spacing_xyz)
    d_naris = np.full(air.shape, np.inf, dtype=np.float64)
    for seed in naris_seeds:
        d_naris = np.minimum(d_naris, geodesic_distance_mm(air, seed, spacing_xyz))
    shortest = float(d_naris[outlet_seed])
    if not np.isfinite(shortest):
        notes.append("through-path: no naris→outlet path; keeping flooded air")
        return air.copy(), np.zeros_like(air), notes
    through = air & np.isfinite(d_naris) & np.isfinite(d_out)
    through &= (d_naris + d_out) <= (shortest + slack_mm)
    sinus = air & ~through
    notes.append(
        f"through-path slack={slack_mm} mm shortest={shortest:.1f} mm "
        f"passage={int(through.sum())} sinus_detour={int(sinus.sum())}"
    )
    return through, sinus, notes


def septum_ridge_from_cavities(
    body: np.ndarray,
    left_air: np.ndarray,
    right_air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    tau_mm: float = MIDSURFACE_TAU_MM,
    dmax_mm: float = MIDSURFACE_DMAX_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tissue ridge between anterior L/R air. No naris-mid plane.

    Returns (septum_tissue, dL_euc_mm, dR_euc_mm) where d* are Euclidean
    distances to each air set (for confinement use geodesic d from the flood).
    """
    sampling = _spacing_zyx(spacing_xyz)
    d_l = ndi.distance_transform_edt(~left_air.astype(bool), sampling=sampling)
    d_r = ndi.distance_transform_edt(~right_air.astype(bool), sampling=sampling)
    tissue = body.astype(bool) & ~left_air & ~right_air
    ridge = (
        tissue
        & (np.abs(d_l - d_r) <= tau_mm)
        & (np.minimum(d_l, d_r) <= dmax_mm)
    )
    dil_vox = max(int(round(RIDGE_DILATE_MM / max(float(np.mean(sampling)), 1e-6))), 1)
    if ridge.any():
        septum = morphology.dilation(ridge, footprint=morphology.ball(dil_vox)) & tissue
    else:
        # Fallback: tissue that neighbors both air sets.
        near_l = morphology.dilation(left_air, footprint=morphology.ball(dil_vox))
        near_r = morphology.dilation(right_air, footprint=morphology.ball(dil_vox))
        septum = tissue & near_l & near_r
    return septum.astype(bool), d_l.astype(np.float32), d_r.astype(np.float32)


def meeting_set(
    left: np.ndarray,
    right: np.ndarray,
    d_l: np.ndarray,
    d_r: np.ndarray,
    tau_mm: float = 2.0,
) -> np.ndarray:
    air = left | right
    finite = np.isfinite(d_l) & np.isfinite(d_r)
    delta = np.zeros_like(d_l, dtype=np.float64)
    np.subtract(d_l, d_r, out=delta, where=finite)
    near = air & finite & (np.abs(delta) <= tau_mm)
    dil_l = morphology.dilation(left, footprint=morphology.ball(1))
    dil_r = morphology.dilation(right, footprint=morphology.ball(1))
    border = (dil_l & right) | (dil_r & left)
    return near | border
