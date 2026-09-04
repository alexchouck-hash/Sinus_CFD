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
# The box must extend ANTERIOR of the CT-air-shell naris all the way to the skin
# nares, or passage_lumen stops short of the boundary-condition inlet and there
# is no flow path to solve (VH: BC inlet at y=4, box started at y=36).
NASAL_BOX_ANTERIOR_MM = 45.0
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

# --- Dead-end sinus strip (supersedes the corridor test; see the docstring of
# dead_end_sinus_strip and docs/segmentation_strategy.md K9).
# Thickness of the anterior/posterior opening slabs used as flow terminals.
NARIS_TERMINAL_SLAB_MM = 1.5
# A chamber whose local radius exceeds this multiple of its widest-path radius
# to any opening is "roomy space reached only through a neck" -> sinus seed.
# Measured stable over 1.15-2.0 on both Visible Human and CQ500CT105; the
# lateral (maxillary) basins do not move across that range.
SINUS_SEED_RATIO = 2.0
# Local radius within this multiple of the widest-path radius means open
# corridor -> passage marker. Raising it erodes the sinuses (VH 8.9 -> 4.8 mL
# at 1.6), so it stays tight.
SINUS_CORRIDOR_RATIO = 1.05
SINUS_MIN_BODY_ML = 0.3


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


def main_air_component(air: np.ndarray) -> np.ndarray:
    """The largest 26-connected component of an air mask.

    A naris seed has to land on the air the flood can actually traverse. THCA's
    right seed once snapped onto a 55-voxel pocket painted at the nostril that was
    not joined to the lumen, and the competing flood then returned 55 voxels for
    the whole right cavity while the left took everything. Snapping against this
    mask cannot fabricate a connection: the snap radius still bounds how far a
    seed may move, so a genuinely unreachable lumen fails loudly instead.
    """
    air = air.astype(bool)
    if not air.any():
        return air
    lab, n = ndi.label(air, np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return air
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(counts.argmax())


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
    y_ant = max(int(round(NASAL_BOX_ANTERIOR_MM / max(sy, 1e-6))), 4)
    z0, z1 = max(0, int(z_mid) - z_half), min(nz, int(z_mid) + z_half + 1)
    if y_anterior_is_low:
        y0, y1 = max(0, int(y_face) - y_ant), min(ny, int(y_face) + y_post)
    else:
        y0, y1 = max(0, int(y_face) - y_post), min(ny, int(y_face) + y_ant)
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


def widest_path_bottleneck_mm(
    air: np.ndarray,
    terminal: np.ndarray,
    edt_mm: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Widest-path (maximin) radius from ``terminal`` to every air voxel.

    ``bottleneck[v]`` is the largest, over all paths from the terminal to ``v``,
    of the smallest wall-distance along that path — i.e. the radius of the
    tightest constriction you must squeeze through to reach ``v``.

    Terminals are **openings, not constrictions**. Seeding the terminal with its
    own ``edt`` caps every downstream bottleneck by it: the outlet slab sits on
    the FOV cut where ``edt`` is small, which made the nasopharynx (2.50 mm wide,
    directly behind the outlet) report a 1.18 mm bottleneck and get classified as
    sinus. Seeding at effectively infinite width makes the first constraint the
    first genuine constriction inside the airway.
    """
    air = air.astype(bool)
    nz, ny, nx = air.shape
    bott = np.zeros(air.shape, dtype=np.float32)
    big = np.float32(1e6)
    heap: list[tuple[float, int, int, int]] = []
    for z, y, x in zip(*np.where(terminal & air)):
        z, y, x = int(z), int(y), int(x)
        if big > bott[z, y, x]:
            bott[z, y, x] = big
            heap.append((-float(big), z, y, x))
    heapq.heapify(heap)
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    while heap:
        neg, z, y, x = heapq.heappop(heap)
        cur = -neg
        if cur < bott[z, y, x]:
            continue
        for dz, dy, dx in offsets:
            z2, y2, x2 = z + dz, y + dy, x + dx
            if not (0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx):
                continue
            if not air[z2, y2, x2]:
                continue
            w = min(cur, float(edt_mm[z2, y2, x2]))
            if w > bott[z2, y2, x2]:
                bott[z2, y2, x2] = w
                heapq.heappush(heap, (-w, z2, y2, x2))
    bott[bott >= 1e5] = 0.0  # terminal voxels themselves carry no bottleneck
    return bott


def _opening_slabs(
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Anterior and posterior opening slabs of the airway, in physical mm."""
    ny = air.shape[1]
    sy = float(spacing_xyz[1])
    yy = np.where(air)[1]
    y0, y1 = int(yy.min()), int(yy.max())
    t = max(int(round(NARIS_TERMINAL_SLAB_MM / max(sy, 1e-6))), 1)
    idx = np.arange(ny)[None, :, None]
    return air & (idx <= y0 + t), air & (idx >= y1 - t)


def merge_zone_reaching_opening(
    merge_zone: np.ndarray, opening: np.ndarray, vox_ml: float
) -> tuple[np.ndarray, list[str]]:
    """Keep only the 26-connected parts of ``merge_zone`` that touch ``opening``.

    Air behind the choanal landmark that cannot reach the outlet without
    leaving the half-space is not nasopharynx (a sphenoid, a posterior ethmoid
    cell) and must not carry the nasopharynx's veto. If no part touches the
    opening the zone is returned untouched with a WARN: losing the veto would
    let the dead-end test strip the nasopharynx, which is worse than keeping a
    sphenoid.
    """
    notes: list[str] = []
    merge_zone = merge_zone.astype(bool)
    if not merge_zone.any():
        return merge_zone, notes
    lab, n = ndi.label(merge_zone, np.ones((3, 3, 3), dtype=bool))
    hit = np.unique(lab[merge_zone & opening.astype(bool)])
    hit = hit[hit > 0]
    if hit.size == 0:
        notes.append("WARN: merge zone does not touch the posterior opening; "
                     "keeping the whole half-space as nasopharynx")
        return merge_zone, notes
    kept = np.isin(lab, hit)
    dropped = merge_zone & ~kept
    if dropped.any():
        notes.append(
            f"merge zone: {int(n)} parts behind the landmark, "
            f"{hit.size} reach the opening; {dropped.sum() * vox_ml:.1f} mL "
            f"behind the landmark is not nasopharynx and may be stripped")
    return kept, notes


def dead_end_sinus_strip(
    air: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    merge_zone: np.ndarray | None = None,
    naris_seeds: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split the flooded airway into (passage, sinus) by a dead-end test.

    A sinus is a **roomy chamber whose only route to any opening squeezes
    through a neck**. Because openings sit at both ends of the airway (nares and
    nasopharynx/outlet), cavity behind a tight nasal valve still has a wide route
    to the posterior opening and is kept — the discriminator the strategy doc
    says caliber alone cannot provide falls out of the geometry rather than
    needing a special case.

    Supersedes ``through_path_passage``, which kept only voxels within a slack of
    *the single shortest* naris→outlet geodesic. A nasal cavity is a broad
    volume, and the second nostril's route is longer than "shortest", so the
    contralateral cavity always read as a detour ("WARN: through-path dropped a
    cavity") and the strip was disabled on every real head.

    ``merge_zone`` — air posterior of the choanal landmark — is the nasopharynx
    and is passage by definition (strategy K5). It is never seeded as sinus and
    any basin touching it is rejected. Without it the nasopharynx, a wide chamber
    behind the relatively narrow choanae, is misread as a dead end.

    Returns ``(passage, sinus, notes)``.
    """
    notes: list[str] = []
    air = air.astype(bool)
    if not air.any():
        return air.copy(), np.zeros_like(air), ["dead-end strip: empty airway"]
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    vox_ml = (sz * sy * sx) / 1000.0
    edt = ndi.distance_transform_edt(air, sampling=(sz, sy, sx)).astype(np.float32)
    if merge_zone is not None:
        merge_zone = merge_zone.astype(bool) & air
    else:
        merge_zone = np.zeros_like(air)
    # Terminals must be the ANATOMICAL openings. Deriving them from the array
    # extent works on a whole airway but not on a box-cropped flood domain,
    # where the extent faces are artificial cuts (CQ500CT105: 6358 voxels of
    # fake "opening" vs 240 at the real nares) and nothing then reads as being
    # behind a neck.
    ant_slab, post_slab = _opening_slabs(air, spacing_xyz)
    if naris_seeds:
        ant = np.zeros_like(air)
        rz = max(int(round(NARIS_TERMINAL_SLAB_MM / max(sz, 1e-6))), 1)
        ry = max(int(round(NARIS_TERMINAL_SLAB_MM / max(sy, 1e-6))), 1)
        rx = max(int(round(NARIS_TERMINAL_SLAB_MM / max(sx, 1e-6))), 1)
        nz_, ny_, nx_ = air.shape
        for s in naris_seeds:
            z, y, x = int(s[0]), int(s[1]), int(s[2])
            ant[
                max(0, z - rz) : z + rz + 1,
                max(0, y - ry) : y + ry + 1,
                max(0, x - rx) : x + rx + 1,
            ] = True
        ant &= air
        if not ant.any():
            ant = ant_slab
    else:
        ant = ant_slab
    # The merge zone is a passage MARKER and a rejection test, not a bottleneck
    # source. Seeding the widest-path from it makes the maxillary sinuses
    # reachable widely from behind and they stop looking like dead ends
    # (Visible Human dropped from two maxillary bodies to one).
    post = post_slab
    terminal = ant | post
    bott = np.maximum(
        widest_path_bottleneck_mm(air, ant, edt, spacing_xyz),
        widest_path_bottleneck_mm(air, post, edt, spacing_xyz),
    )
    # A basin must also survive the extent slabs: on a full airway those ARE the
    # openings, and a sinus must not touch them either.
    terminal = terminal | ant_slab | post_slab
    # merge_zone arrives as a posterior HALF-SPACE, which holds more than the
    # nasopharynx: the sphenoid sits behind the choanae too, and the veto handed
    # it back to the flow domain (THCA: 9.1 mL, both sphenoids, inside
    # passage_lumen). Narrowing the veto by calibre ("half-space that is not
    # itself behind a neck") was tried and REJECTED -- it carved the nasopharynx
    # (bodies bounded at 6.5-11.5 mm openings). The nasopharynx is instead
    # defined by CONNECTIVITY: the part of the half-space that reaches the
    # posterior opening without leaving the half-space. The sphenoid's only air
    # route to the nasopharynx runs forward through its ostium into the nasal
    # cavity, in front of the landmark, so it is not connected inside the
    # half-space and loses the veto; the nasopharynx contains the opening and
    # keeps it whole, with no calibre judgement anywhere.
    merge_zone, mz_notes = merge_zone_reaching_opening(merge_zone, post | post_slab, vox_ml)
    notes.extend(mz_notes)
    behind = air & (bott > 0) & (edt > SINUS_SEED_RATIO * bott) & ~merge_zone
    struct = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndi.label(behind, struct)
    if n == 0:
        notes.append("dead-end strip: no sinus seed; whole airway is passage")
        return air.copy(), np.zeros_like(air), notes
    sizes = ndi.sum(behind, lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if sizes[i] * vox_ml >= SINUS_MIN_BODY_ML]
    if not keep:
        notes.append("dead-end strip: all sinus seeds below the volume floor")
        return air.copy(), np.zeros_like(air), notes
    corridor = air & (bott > 0) & (edt <= SINUS_CORRIDOR_RATIO * bott)
    markers = np.zeros(air.shape, dtype=np.int32)
    markers[corridor | terminal | merge_zone] = 1
    for k, old in enumerate(keep, start=2):
        markers[lab == old] = k
    try:
        from skimage.segmentation import watershed
    except Exception as exc:  # pragma: no cover
        notes.append(f"dead-end strip unavailable ({exc}); keeping whole airway")
        return air.copy(), np.zeros_like(air), notes
    ws = watershed(-edt, markers, mask=air)
    sinus = np.zeros_like(air)
    n_kept = n_rej = 0
    for k in range(2, 2 + len(keep)):
        basin = air & (ws == k)
        if not basin.any() or basin.sum() * vox_ml < SINUS_MIN_BODY_ML:
            continue
        if (basin & terminal).any() or (basin & merge_zone).any():
            n_rej += 1  # reaches an opening or the nasopharynx -> passage
            continue
        sinus |= basin
        n_kept += 1
    passage = air & ~sinus
    # Never let the strip delete a whole side or the openings.
    if not (passage & ant).any() or not (passage & post).any():
        notes.append("WARN: dead-end strip would drop an opening; keeping whole airway")
        return air.copy(), np.zeros_like(air), notes
    notes.append(
        f"dead-end strip: {len(keep)} seeds, {n_kept} sinus / {n_rej} rejected; "
        f"sinus={sinus.sum() * vox_ml:.1f} mL passage={passage.sum() * vox_ml:.1f} mL "
        f"(merge_zone={merge_zone.sum() * vox_ml:.1f} mL)"
    )
    return passage, sinus, notes


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
