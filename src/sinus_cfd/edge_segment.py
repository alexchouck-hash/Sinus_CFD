"""
Edge-aware CT segmentation for head tissue vs open air space.

Pipeline:
  1. HU thresholds for coarse tissue classes
  2. 3D gradient magnitude (edge strength) to refine boundaries
  3. Head-only crop (remove shoulders)
  4. Nose-tip / external naris landmarks from geometry + edges
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import gaussian, sobel


# Label IDs (match tissues.py)
EXT, AIR, SOFT, CART, BONE = 0, 1, 2, 3, 4


@dataclass
class EdgeSegResult:
    labels: np.ndarray
    body: np.ndarray
    air: np.ndarray
    soft_tissue: np.ndarray
    cartilage: np.ndarray
    bone: np.ndarray
    edge_magnitude: np.ndarray
    head_bbox_zyx: tuple[slice, slice, slice]
    shoulder_crop_z0: int
    nose_tip_zyx: tuple[int, int, int]
    naris_left_zyx: tuple[int, int, int] | None
    naris_right_zyx: tuple[int, int, int] | None
    superior_is_high_z: bool
    y_anterior_is_low: bool
    notes: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        return {
            "shoulder_crop_z0": self.shoulder_crop_z0,
            "nose_tip_zyx": list(self.nose_tip_zyx),
            "naris_left_zyx": list(self.naris_left_zyx) if self.naris_left_zyx else None,
            "naris_right_zyx": list(self.naris_right_zyx) if self.naris_right_zyx else None,
            "superior_is_high_z": self.superior_is_high_z,
            "y_anterior_is_low": self.y_anterior_is_low,
            "notes": self.notes,
        }


def gradient_magnitude(hu: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """3D edge strength via Gaussian-smoothed gradient magnitude."""
    sm = gaussian(hu.astype(np.float32), sigma=sigma, preserve_range=True)
    # Per-axis sobel-like gradients
    gz = ndi.sobel(sm, axis=0)
    gy = ndi.sobel(sm, axis=1)
    gx = ndi.sobel(sm, axis=2)
    mag = np.sqrt(gz * gz + gy * gy + gx * gx)
    # Normalize robustly
    p99 = float(np.percentile(mag, 99))
    if p99 > 1e-6:
        mag = np.clip(mag / p99, 0, 1)
    return mag.astype(np.float32)


def segment_body_edge_aware(
    hu: np.ndarray,
    edge: np.ndarray,
    body_hu_min: float = -200.0,
) -> np.ndarray:
    """
    Body mask: HU threshold, suppress weak exterior noise, fill holes.
    Edges reinforce the outer contour (high edge near skin).
    """
    seed = hu > body_hu_min
    # Remove isolated speckles
    seed = morphology.opening(seed, footprint=morphology.ball(1))
    labeled, n = ndi.label(seed)
    if n == 0:
        raise ValueError("No body tissue found")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    body = labeled == int(np.argmax(counts))
    body = ndi.binary_fill_holes(body)
    # Light close; keep outer edge crisp using edge map as stop (optional)
    body = morphology.closing(body, footprint=morphology.ball(1))
    body = ndi.binary_fill_holes(body)
    return body.astype(bool)


def crop_shoulders(
    body: np.ndarray,
    superior_is_high_z: bool,
) -> tuple[int, int, str]:
    """
    Return (z0, z1) inclusive-exclusive head range excluding shoulders.

    Shoulders: large cross-section at the inferior end of the FOV.
    """
    n_z = body.shape[0]
    areas = np.array([float(body[z].sum()) for z in range(n_z)], dtype=np.float64)
    # Reference = median area in superior half of volume
    if superior_is_high_z:
        # inferior = low z
        ref = float(np.median(areas[n_z // 2 :]))
        # Walk from inferior: first z where area within 15% of ref mid-head
        z0 = 0
        for z in range(0, n_z // 2):
            if areas[z] <= ref * 1.18:
                z0 = z
                break
        # Also require area not still dropping steeply from shoulders
        # Prefer local minimum after shoulder bulge
        if z0 < 5:
            # find where d(area)/dz becomes small after large area
            for z in range(5, n_z // 2):
                if areas[z] < areas[0] * 0.55:
                    z0 = z
                    break
        z1 = n_z
        note = f"shoulder crop inferior z0={z0} (areas[0]={areas[0]:.0f}, ref={ref:.0f})"
    else:
        ref = float(np.median(areas[: n_z // 2]))
        z1 = n_z
        for z in range(n_z - 1, n_z // 2, -1):
            if areas[z] <= ref * 1.18:
                z1 = z + 1
                break
        z0 = 0
        note = f"shoulder crop superior-end z1={z1}"
    z0 = int(np.clip(z0, 0, n_z - 10))
    z1 = int(np.clip(z1, z0 + 10, n_z))
    return z0, z1, note


def find_nose_tip(
    body: np.ndarray,
    y_anterior_is_low: bool,
    superior_is_high_z: bool,
    z0: int,
    z1: int,
) -> tuple[int, int, int]:
    """
    Nose tip = most anterior body voxel in mid-cranio-caudal head band.
    Returns (z, y, x) of tip.
    """
    # Search band: avoid crown and neck
    n_z = body.shape[0]
    if superior_is_high_z:
        zs = range(max(z0, int(z0 + 0.15 * (z1 - z0))), int(z0 + 0.70 * (z1 - z0)))
    else:
        zs = range(int(z0 + 0.30 * (z1 - z0)), min(z1, int(z0 + 0.85 * (z1 - z0))))

    best = None
    best_score = None
    for z in zs:
        sl = body[z]
        if not sl.any():
            continue
        yy, xx = np.where(sl)
        if y_anterior_is_low:
            y_tip = int(yy.min())
            # tip x = mean x of voxels at that y
            xs = xx[yy == y_tip]
            x_tip = int(np.median(xs))
            # more anterior = smaller y → lower score better
            score = y_tip
            if best_score is None or score < best_score:
                best_score = score
                best = (z, y_tip, x_tip)
        else:
            y_tip = int(yy.max())
            xs = xx[yy == y_tip]
            x_tip = int(np.median(xs))
            score = -y_tip
            if best_score is None or score < best_score:
                best_score = score
                best = (z, y_tip, x_tip)
    if best is None:
        # fallback center
        zz, yy, xx = np.where(body)
        best = (int(np.median(zz)), int(np.median(yy)), int(np.median(xx)))
    return best


def detect_nares_at_nose_tip(
    hu: np.ndarray,
    body: np.ndarray,
    air: np.ndarray,
    edge: np.ndarray,
    nose_tip: tuple[int, int, int],
    y_anterior_is_low: bool,
    superior_is_high_z: bool,
    search_half_z: int = 12,
    anterior_depth: int = 28,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, dict[str, Any]]:
    """
    External L/R nares near the nose tip on the skin, NOT eyes.

    Eyes are more superior and lateral; we lock z near nose tip and stay
    in the anterior snout band.
    """
    info: dict[str, Any] = {"nose_tip": list(nose_tip)}
    nz, ny, nx = body.shape
    zt, yt, xt = nose_tip

    z_lo = max(zt - search_half_z, 0)
    z_hi = min(zt + search_half_z + 1, nz)
    # Prefer slightly inferior to tip (vestibule floor) but never far superior (eyes)
    if superior_is_high_z:
        # superior = high z → clamp upper bound tightly above tip
        z_hi = min(zt + 8, nz)
        z_lo = max(zt - 15, 0)
    else:
        z_lo = max(zt - 8, 0)
        z_hi = min(zt + 15, nz)

    # Skin shell
    eroded = morphology.erosion(body, footprint=morphology.ball(1))
    skin = body & ~eroded

    # Anterior snout ROI
    roi = np.zeros_like(body)
    roi[z_lo:z_hi, :, :] = True
    if y_anterior_is_low:
        y_cut = min(yt + anterior_depth, ny - 1)
        roi[:, y_cut + 1 :, :] = False
        roi[:, : max(yt - 2, 0), :] = False  # keep from tip inward a bit
    else:
        y_cut = max(yt - anterior_depth, 0)
        roi[:, :y_cut, :] = False
        roi[:, min(yt + 2, ny) :, :] = False

    # Candidate: air or very low HU near skin in ROI, or skin voxels adjacent to air
    low = (hu <= -200) & body
    near_skin = morphology.dilation(skin, footprint=morphology.ball(2))
    cand = roi & near_skin & (air | low)
    # Also include skin voxels that touch air (true openings)
    skin_open = skin & morphology.dilation(air | low, footprint=morphology.ball(2)) & roi
    cand = cand | skin_open

    info["candidate_voxels"] = int(cand.sum())
    if cand.sum() < 5:
        # Fallback: anterior-most air in z band near tip
        band = np.zeros_like(air)
        band[z_lo:z_hi] = True
        aa = air & band
        if not aa.any():
            aa = low & band
        if not aa.any():
            return None, None, {**info, "error": "no candidates"}
        zz, yy, xx = np.where(aa)
        if y_anterior_is_low:
            thr = np.percentile(yy, 15)
            aa = aa & (np.arange(ny)[None, :, None] <= thr)
        else:
            thr = np.percentile(yy, 85)
            aa = aa & (np.arange(ny)[None, :, None] >= thr)
        cand = aa
        info["candidate_voxels_fallback"] = int(cand.sum())

    zz, yy, xx = np.where(cand)
    if len(zz) == 0:
        return None, None, {**info, "error": "empty cand"}

    # Split L/R with a minimum lateral offset from midline (avoid collapsing to tip)
    mid_gap = 6
    left_m = xx >= (xt + mid_gap)
    right_m = xx <= (xt - mid_gap)

    def _pick(mask_bool: np.ndarray, side_sign: int) -> tuple[int, int, int] | None:
        """side_sign: +1 left (higher x), -1 right (lower x)."""
        if not np.any(mask_bool):
            # Seed from tip offset along x on skin
            x0 = int(np.clip(xt + side_sign * 10, 0, nx - 1))
            return _snap_skin(zt, yt, x0, skin, y_anterior_is_low)
        idx = np.where(mask_bool)[0]
        scores = []
        for i in idx:
            z, y, x = int(zz[i]), int(yy[i]), int(xx[i])
            ant = -(y if y_anterior_is_low else -y)
            sk = 3.0 if skin[z, y, x] else 0.0
            ed = float(edge[z, y, x])
            zpen = -0.08 * abs(z - zt)
            # Prefer modest lateral distance from midline (~8–18 px)
            lat = abs(x - xt)
            lat_score = -0.05 * abs(lat - 12)
            scores.append(ant + sk + 1.5 * ed + zpen + lat_score)
        j = idx[int(np.argmax(scores))]
        z, y, x = int(zz[j]), int(yy[j]), int(xx[j])
        return _snap_skin(z, y, x, skin, y_anterior_is_low)

    left = _pick(left_m, +1)
    right = _pick(right_m, -1)
    # Ensure distinct
    if left is not None and right is not None and left == right:
        left = _snap_skin(zt, yt, min(xt + 12, nx - 1), skin, y_anterior_is_low)
        right = _snap_skin(zt, yt, max(xt - 12, 0), skin, y_anterior_is_low)
    info["naris_left"] = list(left) if left else None
    info["naris_right"] = list(right) if right else None
    return left, right, info


def _snap_skin(
    z: int, y: int, x: int, skin: np.ndarray, y_anterior_is_low: bool, rad: int = 4
) -> tuple[int, int, int]:
    nz, ny, nx = skin.shape
    best = (z, y, x)
    best_s = -1e9
    for dz in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                zz, yy, xx = z + dz, y + dy, x + dx
                if not (0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx):
                    continue
                if not skin[zz, yy, xx]:
                    continue
                ant = -(yy if y_anterior_is_low else -yy)
                if ant > best_s:
                    best_s = ant
                    best = (zz, yy, xx)
    return best


def infer_y_anterior_is_low(body: np.ndarray, air: np.ndarray) -> bool:
    """
    Anterior = body y-extreme closer to the *centroid of internal air*.

    Nasal/pharyngeal air sits nearer the face than the occiput, so the face
    is the y-extreme with smaller distance to the air centroid.
    """
    by = np.where(body)[1]
    if len(by) == 0:
        return True
    y_min, y_max = int(by.min()), int(by.max())
    az = np.where(air & body)[1]
    if len(az) < 30:
        # Fallback: more “pointed” end = smaller width near that extreme
        return True
    air_cy = float(np.median(az))
    return abs(air_cy - y_min) <= abs(air_cy - y_max)


def segment_air_edge_aware(
    hu: np.ndarray,
    body: np.ndarray,
    edge: np.ndarray,
    air_hu_max: float = -300.0,
) -> np.ndarray:
    """Open air space inside body; edges help separate soft tissue walls."""
    air = body & (hu <= air_hu_max) & (hu >= -1024)
    # Small closing to reconnect thin airways without filling soft tissue
    air = morphology.closing(air, footprint=morphology.ball(1))
    labeled, n = ndi.label(air)
    if n == 0:
        return air
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts >= 80
    keep[0] = False
    air = keep[labeled]
    return air.astype(bool)


def run_edge_segmentation(
    hu: np.ndarray,
    superior_is_high_z: bool,
    body_hu_min: float = -200.0,
    air_hu_max: float = -300.0,
) -> EdgeSegResult:
    notes: list[str] = []
    edge = gradient_magnitude(hu, sigma=1.0)
    body = segment_body_edge_aware(hu, edge, body_hu_min=body_hu_min)
    z0, z1, crop_note = crop_shoulders(body, superior_is_high_z)
    notes.append(crop_note)

    # Zero body outside head crop (shoulders removed from masks)
    if superior_is_high_z:
        body[:z0] = False
    else:
        body[z1:] = False
    body = morphology.opening(body, footprint=morphology.ball(1))
    # Re-fill after shoulder cut
    body = ndi.binary_fill_holes(body)

    air = segment_air_edge_aware(hu, body, edge, air_hu_max=air_hu_max)
    y_ant_low = infer_y_anterior_is_low(body, air)
    notes.append(f"y_anterior_is_low={y_ant_low}")

    nose = find_nose_tip(body, y_ant_low, superior_is_high_z, z0 if superior_is_high_z else 0, z1)
    notes.append(f"nose_tip_zyx={list(nose)}")

    nL, nR, ninfo = detect_nares_at_nose_tip(
        hu, body, air, edge, nose, y_ant_low, superior_is_high_z
    )
    notes.append(f"nares L={nL} R={nR}")

    # Multi-class labels
    labels = np.zeros(hu.shape, dtype=np.int16)
    labels[body] = SOFT
    bone = body & (hu >= 300)
    bone = morphology.opening(bone, footprint=morphology.ball(1))
    labels[bone] = BONE
    cart = body & (hu >= 80) & (hu < 300) & ~bone
    lab_c, nc = ndi.label(cart)
    if nc:
        cc = np.bincount(lab_c.ravel())
        cc[0] = 0
        cart = cc[lab_c] >= 40
    labels[cart] = CART
    labels[air] = AIR
    labels[~body] = EXT

    # BBox around remaining body
    zz, yy, xx = np.where(body)
    margin = 4
    bb = (
        slice(max(int(zz.min()) - margin, 0), min(int(zz.max()) + margin + 1, body.shape[0])),
        slice(max(int(yy.min()) - margin, 0), min(int(yy.max()) + margin + 1, body.shape[1])),
        slice(max(int(xx.min()) - margin, 0), min(int(xx.max()) + margin + 1, body.shape[2])),
    )

    return EdgeSegResult(
        labels=labels,
        body=body,
        air=air,
        soft_tissue=(labels == SOFT),
        cartilage=(labels == CART),
        bone=(labels == BONE),
        edge_magnitude=edge,
        head_bbox_zyx=bb,
        shoulder_crop_z0=z0 if superior_is_high_z else 0,
        nose_tip_zyx=nose,
        naris_left_zyx=nL,
        naris_right_zyx=nR,
        superior_is_high_z=superior_is_high_z,
        y_anterior_is_low=y_ant_low,
        notes=notes,
    )
