"""
CT-native nasal airway extraction: real nostril openings + L/R cavities + septum.

Unlike geometric naris tunnels, this uses DICOM/HU topology:
  - Interior air = low HU inside the body
  - Exterior free air = outside the body
  - True naris openings = body surface that touches BOTH interior and exterior air
  - Left/right cavities grown only through air (septum tissue blocks crossing)
  - Septum = soft/cartilage/bone between left and right nasal air

Future upgrade path: same interfaces can wrap a neural net (nnU-Net / NasalSeg
labels) without changing downstream passage/OpenFOAM export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

# --- Naris-pair geometry, in physical millimetres -------------------------
# These are spacing-aware by contract (docs/segmentation_strategy.md): never
# express a naris threshold in voxels, because living CT is anisotropic
# (THCA is 0.977 / 0.977 / 1.5 mm) while Visible Human is 1 mm isotropic.

# Two nostril centres closer than this laterally are not two nostrils.
MIN_NARIS_SEPARATION_MM = 5.0
# Wider than this is not one pair of nostrils (e.g. seeds on opposite cheeks).
MAX_NARIS_SEPARATION_MM = 50.0
# Nostrils sit at roughly the same depth and height; a pair that does not is a
# collapsed or relocated detection, not anatomy.
NARIS_PAIR_PLANE_TOL_MM = 20.0
# Confidence bar for the x-histogram two-peak search. 15 mm reproduces the old
# hard-coded ``sep >= 15`` exactly on 1 mm isotropic data (Visible Human) while
# becoming correct on anisotropic living CT.
MIN_NARIS_PEAK_SEPARATION_MM = 15.0
# The air-HU preference filter can silently relocate the naris shell backwards
# into the mid-cavity when the anterior airway is a painted geodesic tube rather
# than resolved air. Note it when it moves the shell more than this.
# Measured signed-median shift: Visible Human Female +1.0 mm, Male -3.0 mm,
# THCA +32.2 mm. (The *mean* would put VH Female at +4.4 mm, i.e. 0.6 mm from a
# 5 mm bar — which is why this is a median against a 10 mm bar, not a mean.)
NARIS_HU_FILTER_SHIFT_WARN_MM = 10.0
# Separation alone does not make a pair bilateral: on CQ500CT390 the two-peak
# search failed, the percentile fallback drew BOTH seeds from one blob 16.5 mm
# apart and both on the patient-left side of the midline, and the competing flood
# came out 12:1. A valid pair must straddle the midline, not merely be apart.
NARIS_MIDLINE_TOL_MM = 5.0


@dataclass
class CTNasalAirwayResult:
    """Cropped-volume arrays (z, y, x) matching existing head outputs."""

    interior_air: np.ndarray
    left_cavity: np.ndarray
    right_cavity: np.ndarray
    passage_lumen: np.ndarray  # L∪R (+ nasopharynx path if connected)
    septum: np.ndarray
    mucosa_wall: np.ndarray  # air-adjacent tissue (includes septum)
    naris_opening: np.ndarray  # CT-derived open skin at nares
    left_naris_center_zyx: tuple[int, int, int] | None
    right_naris_center_zyx: tuple[int, int, int] | None
    left_naris_center_mm: list[float] | None
    right_naris_center_mm: list[float] | None
    method: str = "ct_topology_hu_edge"
    # Where the accepted naris seeds actually came from: "ct_air_shell",
    # "prior", "mixed(...)" or "none". Never infer this from the method string.
    naris_source: str = "ct_air_shell"
    notes: list[str] = field(default_factory=list)
    merge_zone: np.ndarray | None = None
    choanal_landmark: np.ndarray | None = None
    sinus_detour: np.ndarray | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "naris_source": self.naris_source,
            "left_naris_center_zyx": list(self.left_naris_center_zyx)
            if self.left_naris_center_zyx
            else None,
            "right_naris_center_zyx": list(self.right_naris_center_zyx)
            if self.right_naris_center_zyx
            else None,
            "left_naris_center_mm": self.left_naris_center_mm,
            "right_naris_center_mm": self.right_naris_center_mm,
            "left_voxels": int(self.left_cavity.sum()),
            "right_voxels": int(self.right_cavity.sum()),
            "passage_voxels": int(self.passage_lumen.sum()),
            "septum_voxels": int(self.septum.sum()),
            "naris_opening_voxels": int(self.naris_opening.sum()),
            "merge_zone_voxels": int(self.merge_zone.sum()) if self.merge_zone is not None else 0,
            "sinus_detour_voxels": int(self.sinus_detour.sum())
            if self.sinus_detour is not None
            else 0,
            "notes": self.notes,
        }


def _zyx_to_mm(
    zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> list[float]:
    z, y, x = zyx
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    return [float(ox + x * sx), float(oy + y * sy), float(oz + z * sz)]


def validate_naris_pair(
    left_zyx: tuple[int, int, int] | None,
    right_zyx: tuple[int, int, int] | None,
    spacing_xyz: tuple[float, float, float],
    x_midline: float | None = None,
) -> tuple[bool, str]:
    """Is this a plausible pair of two *distinct* nostril seeds?

    Every predicate is in physical millimetres so the verdict holds on
    anisotropic CT. A pair that fails here must **not** be repaired by inventing
    a seed (the old 2-voxel x nudge put both seeds on the same side of the
    septum). The caller falls back to the whole-head prior, then HARD-fails.

    Returns ``(ok, reason)``; ``reason`` describes the pair when ok.
    """
    if left_zyx is None or right_zyx is None:
        return False, "missing seed"
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    dx = abs(int(left_zyx[2]) - int(right_zyx[2])) * sx
    dy = abs(int(left_zyx[1]) - int(right_zyx[1])) * sy
    dz = abs(int(left_zyx[0]) - int(right_zyx[0])) * sz
    sep = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if dx < MIN_NARIS_SEPARATION_MM:
        return False, f"lateral separation {dx:.2f} mm < {MIN_NARIS_SEPARATION_MM:.1f} mm"
    if sep > MAX_NARIS_SEPARATION_MM:
        return False, f"separation {sep:.2f} mm > {MAX_NARIS_SEPARATION_MM:.1f} mm"
    if dy > NARIS_PAIR_PLANE_TOL_MM:
        return False, (
            f"anterior-posterior offset {dy:.2f} mm > {NARIS_PAIR_PLANE_TOL_MM:.1f} mm"
        )
    if dz > NARIS_PAIR_PLANE_TOL_MM:
        return False, (
            f"superior-inferior offset {dz:.2f} mm > {NARIS_PAIR_PLANE_TOL_MM:.1f} mm"
        )
    # Repo convention (AGENTS.md): high x is patient left.
    if int(left_zyx[2]) <= int(right_zyx[2]):
        return False, "left seed is not lateral of right (high x is patient left)"
    # Straddle: each seed must be on its OWN side of the midline. Without this a
    # pair can sit 16 mm apart with both seeds in the same cavity and pass.
    if x_midline is not None:
        tol = NARIS_MIDLINE_TOL_MM / max(sx, 1e-6)
        if not (int(left_zyx[2]) > x_midline - tol and int(right_zyx[2]) < x_midline + tol):
            return False, (
                f"pair does not straddle the midline x={x_midline:.0f} "
                f"(L x={int(left_zyx[2])}, R x={int(right_zyx[2])}); both seeds are "
                f"on the same side"
            )
    return True, f"sep {sep:.2f} mm (dx {dx:.2f}, dy {dy:.2f}, dz {dz:.2f})"


def detect_nares_from_ct_air(
    hu: np.ndarray,
    body: np.ndarray,
    interior_air: np.ndarray,
    y_anterior_is_low: bool = True,
    air_hu_max: float = -150.0,
    z_hint: int | None = None,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[
    np.ndarray,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
    list[str],
]:
    """
    Interpret nostrils from CT as the **anterior openings of nasal air**.

    On 1 mm CT, free exterior air and skin partial-volume rarely form a clean
    "hole." The reliable CT signal is the most-anterior shell of *interior*
    nasal air (where the cavity opens toward the face), clustered into L/R.

    Returns (opening_mask, left_center_zyx, right_center_zyx, notes).
    """
    notes: list[str] = []
    air = interior_air.astype(bool) & body.astype(bool)
    if not air.any():
        return np.zeros_like(air), None, None, ["No interior air for naris detection."]

    nz, ny, nx = air.shape
    zz_all, yy_all, xx_all = np.where(air)

    # Mid-face z band (around densest nasal air or prior tip hint)
    if z_hint is None:
        z_hint = int(np.median(zz_all))
    z0, z1 = max(0, z_hint - 28), min(nz, z_hint + 32)
    nasal = np.zeros_like(air)
    nasal[z0:z1] = True
    nasal &= air

    # Anterior shell: for each (z,x) column, keep the front-most air voxels
    opening = np.zeros_like(air)
    if y_anterior_is_low:
        for z in range(z0, z1):
            for x in range(nx):
                col = np.where(nasal[z, :, x])[0]
                if len(col) == 0:
                    continue
                ymin = int(col.min())
                opening[z, ymin : min(ymin + 3, ny), x] = nasal[
                    z, ymin : min(ymin + 3, ny), x
                ]
    else:
        for z in range(z0, z1):
            for x in range(nx):
                col = np.where(nasal[z, :, x])[0]
                if len(col) == 0:
                    continue
                ymax = int(col.max())
                opening[z, max(0, ymax - 2) : ymax + 1, x] = nasal[
                    z, max(0, ymax - 2) : ymax + 1, x
                ]

    opening &= air
    zz, yy, xx = np.where(opening)
    if len(zz) < 20:
        notes.append("Anterior air shell too small for naris clustering.")
        return opening, None, None, notes

    # Keep the more anterior half of the shell (true openings, not deep cavity wall)
    if y_anterior_is_low:
        y_cut = float(np.percentile(yy, 45))
        opening = opening & (np.arange(ny)[None, :, None] <= y_cut)
    else:
        y_cut = float(np.percentile(yy, 55))
        opening = opening & (np.arange(ny)[None, :, None] >= y_cut)

    # Prefer air-like HU. When the anterior airway is a painted geodesic tube
    # through soft tissue (living CT with unresolved vestibule air), every tube
    # voxel fails this test and the "nostril" shell silently jumps backwards
    # into the mid-cavity. That is not recoverable here, but it must be visible.
    opening_hu = opening & (hu <= air_hu_max)
    # Measure the relocation *independently* of whether the filter is applied: a
    # guard nested inside the apply-branch goes dead on exactly the cases that
    # starve that branch. Signed median, not abs(mean) — only a POSTERIOR shift
    # is the failure mode, and the median separates the cases an order of
    # magnitude better than the mean (see NARIS_HU_FILTER_SHIFT_WARN_MM).
    if int(opening_hu.sum()) > 0 and int(opening.sum()) > 0:
        y_before = float(np.median(np.where(opening)[1]))
        y_after = float(np.median(np.where(opening_hu)[1]))
        shift_mm = (y_after - y_before) * float(spacing_xyz[1])
        if not y_anterior_is_low:
            shift_mm = -shift_mm
        if shift_mm > NARIS_HU_FILTER_SHIFT_WARN_MM:
            notes.append(
                f"WARN: air-HU filter moves the naris shell {shift_mm:.1f} mm "
                f"posteriorly (median y {y_before:.1f} to {y_after:.1f}); the "
                f"anterior airway is likely a painted tube, not resolved air."
            )
    if int(opening_hu.sum()) >= 40:
        opening = opening_hu

    zz, yy, xx = np.where(opening)
    notes.append(
        f"CT naris shell: {int(opening.sum())} anterior air voxels "
        f"(y {int(yy.min())}–{int(yy.max())}, z-band [{z0},{z1}))."
    )

    # Two lateral peaks in x histogram → L/R nostrils
    from scipy.ndimage import gaussian_filter1d

    h = np.bincount(xx, minlength=nx).astype(float)
    hs = gaussian_filter1d(h, sigma=2.0)
    peaks: list[tuple[float, int]] = []
    for i in range(2, nx - 2):
        if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > 3:
            peaks.append((float(hs[i]), i))
    peaks.sort(reverse=True)
    chosen: list[int] | None = None
    best_sep = 0
    # Spacing-aware confidence bar. At 1 mm isotropic this is 15 voxels, i.e.
    # byte-identical to the old hard-coded threshold; on 1.5 mm or sub-mm CT it
    # now means the same physical distance instead of a different one.
    min_peak_sep_vox = max(int(round(MIN_NARIS_PEAK_SEPARATION_MM / max(float(spacing_xyz[0]), 1e-6))), 1)
    top = [p for p in peaks if p[0] >= peaks[0][0] * 0.2][:12]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            sep = abs(top[i][1] - top[j][1])
            if sep >= min_peak_sep_vox and sep > best_sep:
                best_sep = sep
                chosen = sorted([top[i][1], top[j][1]])
    if chosen is None:
        # fallback: median split of opening x
        xmed = float(np.median(xx))
        chosen = [int(np.percentile(xx, 25)), int(np.percentile(xx, 75))]
        notes.append(f"Naris x-peaks weak; using percentiles {chosen}.")
    else:
        notes.append(f"Naris x-peaks from CT air histogram: {chosen} (sep={best_sep}).")

    right_x, left_x = chosen[0], chosen[1]  # low x = patient right, high x = left

    def _center_near_peak(peak_x: int, side: str) -> tuple[int, int, int] | None:
        band = opening & (np.abs(np.arange(nx)[None, None, :] - peak_x) <= 12)
        band = band & (hu <= air_hu_max + 50)
        bz, by, bx = np.where(band)
        if len(bz) < 5:
            # Widen the lateral band but KEEP the HU constraint.
            band = opening & (np.abs(np.arange(nx)[None, None, :] - peak_x) <= 14)
            band = band & (hu <= air_hu_max + 50)
            bz, by, bx = np.where(band)
        if len(bz) < 5:
            # Last resort: drop the HU constraint. Say so — this is precisely
            # how a painted-tube voxel gets promoted to a "naris".
            band = opening & (np.abs(np.arange(nx)[None, None, :] - peak_x) <= 14)
            bz, by, bx = np.where(band)
            if len(bz):
                notes.append(
                    f"WARN: {side} naris band relaxed off the HU constraint "
                    f"(n={len(bz)})."
                )
        if len(bz) == 0:
            return None
        # Prefer more anterior points
        if y_anterior_is_low:
            ythr = float(np.percentile(by, 40))
            keep = by <= ythr
        else:
            ythr = float(np.percentile(by, 60))
            keep = by >= ythr
        if keep.sum() >= 5:
            bz, by, bx = bz[keep], by[keep], bx[keep]
        # Weight anterior + low HU
        w = (1.0 / (by - by.min() + 1.0)) * np.clip((-hu[bz, by, bx] - 100) / 400.0, 0.2, 2.0)
        zc = int(round(np.average(bz, weights=w)))
        yc = int(round(np.average(by, weights=w)))
        xc = int(round(np.average(bx, weights=w)))
        # Snap to air, in physical millimetres — an unweighted index norm makes
        # a 1.5 mm z step look the same as a 0.98 mm x step on anisotropic CT.
        if not air[zc, yc, xc]:
            pts = np.column_stack([bz, by, bx]).astype(float)
            sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
            d = np.sqrt(
                ((pts[:, 0] - zc) * sz) ** 2
                + ((pts[:, 1] - yc) * sy) ** 2
                + ((pts[:, 2] - xc) * sx) ** 2
            )
            zc, yc, xc = map(int, pts[int(np.argmin(d))])
        notes.append(
            f"CT {side} naris zyx=({zc},{yc},{xc}) HU={float(hu[zc, yc, xc]):.0f} "
            f"air={bool(air[zc, yc, xc])} n={len(bz)}"
        )
        return (zc, yc, xc)

    left = _center_near_peak(left_x, "left")
    right = _center_near_peak(right_x, "right")
    return opening.astype(bool), left, right, notes


def detect_ct_naris_openings(
    hu: np.ndarray,
    body: np.ndarray,
    interior_air: np.ndarray,
    y_anterior_is_low: bool = True,
    air_hu_max: float = -250.0,
    min_component: int = 5,
    prior_left_zyx: tuple[int, int, int] | None = None,
    prior_right_zyx: tuple[int, int, int] | None = None,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, list[str]]:
    """
    Compatibility wrapper: CT anterior-air naris shell (preferred method).
    """
    z_hint = None
    if prior_left_zyx is not None and prior_right_zyx is not None:
        z_hint = int(0.5 * (prior_left_zyx[0] + prior_right_zyx[0]))
    opening, _l, _r, notes = detect_nares_from_ct_air(
        hu,
        body,
        interior_air,
        y_anterior_is_low=y_anterior_is_low,
        air_hu_max=air_hu_max,
        z_hint=z_hint,
        spacing_xyz=spacing_xyz,
    )
    return opening, notes


def _cluster_lr_opening_centers(
    openings: np.ndarray,
    tip_x: int | None = None,
    hu: np.ndarray | None = None,
    air_hu_max: float = -150.0,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, list[str]]:
    """Legacy helper — prefer detect_nares_from_ct_air centers."""
    notes: list[str] = []
    if not openings.any():
        return None, None, ["No opening voxels to cluster."]
    zz, yy, xx = np.where(openings)
    xmed = float(np.median(xx)) if tip_x is None else float(tip_x)
    left_m = xx >= xmed
    right_m = xx < xmed

    def _centroid(mask: np.ndarray) -> tuple[int, int, int] | None:
        if not mask.any():
            return None
        return (
            int(np.round(zz[mask].mean())),
            int(np.round(yy[mask].mean())),
            int(np.round(xx[mask].mean())),
        )

    left = _centroid(left_m)
    right = _centroid(right_m)
    notes.append(f"Naris centers from opening mask: L={left} R={right}")
    return left, right, notes


def _paint_corridor_to_air(
    mask: np.ndarray,
    body: np.ndarray,
    hu: np.ndarray,
    start: tuple[int, int, int],
    air: np.ndarray,
    y_anterior_is_low: bool,
    radius: int = 3,
    air_hu_max: float = -150.0,
    side: str = "left",
    x_sep: int | None = None,
) -> np.ndarray:
    """
    Paint a naris vestibule from a skin landmark into interior air.
    Stays on the correct side of the septum plane (x_sep) so L/R don't merge.
    """
    out = mask.copy()
    z0, y0, x0 = start
    # Target: nearest air voxel on the correct side of septum
    pts = np.column_stack(np.where(air))
    if len(pts) == 0:
        return out
    if x_sep is not None:
        if side == "left":
            side_pts = pts[pts[:, 2] >= x_sep]
        else:
            side_pts = pts[pts[:, 2] <= x_sep]
        if len(side_pts) > 0:
            pts = side_pts
    d = np.linalg.norm(pts.astype(float) - np.array(start, dtype=float), axis=1)
    tgt = tuple(int(v) for v in pts[int(np.argmin(d))])

    a = np.array(start, dtype=float)
    b = np.array(tgt, dtype=float)
    nstep = max(int(np.ceil(np.linalg.norm(b - a))) + 2, 2)
    for t in np.linspace(0.0, 1.0, nstep):
        p = np.round((1.0 - t) * a + t * b).astype(int)
        # Prefer stepping posteriorly from the face
        for r in range(radius + 1):
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        q = (int(p[0] + dz), int(p[1] + dy), int(p[2] + dx))
                        if not (
                            0 <= q[0] < body.shape[0]
                            and 0 <= q[1] < body.shape[1]
                            and 0 <= q[2] < body.shape[2]
                        ):
                            continue
                        if x_sep is not None:
                            if side == "left" and q[2] < x_sep:
                                continue
                            if side == "right" and q[2] > x_sep:
                                continue
                        if body[q] and hu[q] <= air_hu_max:
                            out[q] = True
    # Explicit ball at the naris on the correct side
    for dz in range(-radius, radius + 1):
        for dy in range(0, radius + 4):
            for dx in range(-radius, radius + 1):
                q = (
                    z0 + dz,
                    y0 + (dy if y_anterior_is_low else -dy),
                    x0 + dx,
                )
                if not (
                    0 <= q[0] < body.shape[0]
                    and 0 <= q[1] < body.shape[1]
                    and 0 <= q[2] < body.shape[2]
                ):
                    continue
                if x_sep is not None:
                    if side == "left" and q[2] < x_sep - 0:
                        continue
                    if side == "right" and q[2] > x_sep + 0:
                        continue
                if body[q] and (hu[q] <= air_hu_max or dy <= 2):
                    # near skin allow slightly higher HU (partial volume)
                    if hu[q] <= air_hu_max + 80 or dy <= 1:
                        out[q] = True
    return out


def split_left_right_by_septum_plane(
    interior_air: np.ndarray,
    left_naris_zyx: tuple[int, int, int],
    right_naris_zyx: tuple[int, int, int],
    y_anterior_is_low: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """
    Split nasal air by the septum plane centered BETWEEN the two nostrils.

    Septum plane x = midpoint of left/right naris x (LPS: left = higher x).
    In axial slices the plane is a vertical line that continues straight
    down (through z) between the nostrils — matching CT anatomy.
    """
    notes: list[str] = []
    air = interior_air.astype(bool)
    x_sep = int(round(0.5 * (left_naris_zyx[2] + right_naris_zyx[2])))
    z_sep = int(round(0.5 * (left_naris_zyx[0] + right_naris_zyx[0])))
    y_face = int(round(0.5 * (left_naris_zyx[1] + right_naris_zyx[1])))

    # Nasal ROI: from face posterior into the cavity, mid-face z band
    nz, ny, nx = air.shape
    z0, z1 = max(0, z_sep - 28), min(nz, z_sep + 35)
    if y_anterior_is_low:
        y0, y1 = max(0, y_face - 2), min(ny, y_face + 90)
    else:
        y0, y1 = max(0, y_face - 90), min(ny, y_face + 2)

    nasal = np.zeros_like(air)
    nasal[z0:z1, y0:y1, :] = True
    nasal &= air

    # Hard split on septum plane (no air on the plane itself → wall gap)
    xx = np.arange(nx)[None, None, :]
    left = nasal & (xx > x_sep)
    right = nasal & (xx < x_sep)
    # Leave x==x_sep as non-cavity (septum slot)

    # Posterior shared air (choana / NP) still split by plane for L/R display
    # Outside nasal ROI: keep nothing (avoid sinus dump) unless connected
    # Optionally attach air components that touch nasal L/R
    lab, nlab = ndi.label(air & ~nasal)
    for i in range(1, nlab + 1):
        comp = lab == i
        if int(comp.sum()) < 80:
            continue
        dil = morphology.dilation(comp, footprint=morphology.ball(1))
        if (dil & left).any() and not (dil & right).any():
            left |= comp & (xx > x_sep)
        elif (dil & right).any() and not (dil & left).any():
            right |= comp & (xx < x_sep)
        elif (dil & left).any() and (dil & right).any():
            # shared posterior — still split by plane
            left |= comp & (xx > x_sep)
            right |= comp & (xx < x_sep)

    notes.append(
        f"Septum plane x_index={x_sep} (midpoint of nares x={left_naris_zyx[2]} / "
        f"{right_naris_zyx[2]}); nasal z=[{z0},{z1}) y=[{y0},{y1})."
    )
    notes.append(
        f"L/R by septum plane: left={int(left.sum())} right={int(right.sum())} vx."
    )
    return left.astype(bool), right.astype(bool), x_sep, notes


def extract_septum_and_walls(
    body: np.ndarray,
    left_cavity: np.ndarray,
    right_cavity: np.ndarray,
    interior_air: np.ndarray,
    soft_or_tissue: np.ndarray | None = None,
    x_sep: int | None = None,
    left_naris_zyx: tuple[int, int, int] | None = None,
    right_naris_zyx: tuple[int, int, int] | None = None,
    y_anterior_is_low: bool = True,
    half_width: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Septum from CT tissue between L/R cavities, guided by the naris midplane.

    Primary: soft/cartilage tissue that neighbors both left and right air.
    Guide: keep only candidates near the plane centered between the nostrils
    (so the wall stays between nares) but do **not** force a synthetic slab
    of non-CT material. Light closing fills partial-volume gaps only.
    """
    notes: list[str] = []
    body = body.astype(bool)
    tissue = body & ~interior_air.astype(bool)
    if soft_or_tissue is not None:
        tissue = tissue | (soft_or_tissue.astype(bool) & body)

    if x_sep is None and left_naris_zyx is not None and right_naris_zyx is not None:
        x_sep = int(round(0.5 * (left_naris_zyx[2] + right_naris_zyx[2])))
    if x_sep is None and (left_cavity.any() or right_cavity.any()):
        xl = np.where(left_cavity)[2]
        xr = np.where(right_cavity)[2]
        if len(xl) and len(xr):
            x_sep = int(round(0.5 * (np.median(xl) + np.median(xr))))
        else:
            x_sep = body.shape[2] // 2

    if left_naris_zyx is not None and right_naris_zyx is not None:
        z_face = int(round(0.5 * (left_naris_zyx[0] + right_naris_zyx[0])))
        y_face = int(round(0.5 * (left_naris_zyx[1] + right_naris_zyx[1])))
    elif (left_cavity | right_cavity).any():
        zz, yy, _ = np.where(left_cavity | right_cavity)
        z_face, y_face = int(np.percentile(zz, 20)), int(np.percentile(yy, 10))
    else:
        z_face, y_face = body.shape[0] // 2, 0

    z0, z1 = max(0, z_face - 8), min(body.shape[0], z_face + 50)
    if y_anterior_is_low:
        y0, y1 = max(0, y_face - 1), min(body.shape[1], y_face + 90)
    else:
        y0, y1 = max(0, y_face - 90), min(body.shape[1], y_face + 1)

    roi = np.zeros_like(body)
    roi[z0:z1, y0:y1, :] = True

    # CT primary: tissue between left and right air neighborhoods
    near_l = morphology.dilation(left_cavity, footprint=morphology.ball(3))
    near_r = morphology.dilation(right_cavity, footprint=morphology.ball(3))
    between = near_l & near_r & tissue & roi

    # Soft guide: allow a wider band around naris midplane (does not invent tissue)
    xx = np.arange(body.shape[2])[None, None, :]
    guide = (xx >= x_sep - half_width) & (xx <= x_sep + half_width)
    # Prefer CT between-air tissue; soft-weight toward midplane but keep off-plane CT too
    septum = between & (guide | morphology.dilation(guide, footprint=morphology.ball(2)))
    # If guide is empty of CT, fall back to all between-air tissue in ROI
    if int(septum.sum()) < 80:
        septum = between
        notes.append("Septum mostly CT between L/R (midplane guide sparse).")
    else:
        notes.append(
            f"Septum = CT tissue between L/R air, guided by naris midplane x={x_sep}."
        )

    # Light close to reconnect partial-volume gaps (still only where body/tissue)
    if septum.any():
        septum = morphology.closing(septum, footprint=morphology.ball(1))
        septum &= tissue & roi

    # Face segment: only real tissue between the two vestibules, not a full synthetic wall
    if left_naris_zyx is not None and right_naris_zyx is not None:
        face = np.zeros_like(body)
        if y_anterior_is_low:
            face[z0:z1, y0 : min(y0 + 25, y1), :] = True
        else:
            face[z0:z1, max(y0, y1 - 25) : y1, :] = True
        face_sep = face & between
        septum = septum | face_sep

    notes.append(
        f"Septum voxels={int(septum.sum())} (CT-following, naris-mid guided); "
        f"roi z=[{z0},{z1}) y=[{y0},{y1})."
    )

    passage = left_cavity | right_cavity
    mucosa = tissue & morphology.dilation(passage, footprint=morphology.ball(1))
    mucosa = mucosa & ~passage
    notes.append(f"Mucosa/wall voxels={int(mucosa.sum())}.")
    return septum.astype(bool), mucosa.astype(bool), notes


def extract_ct_nasal_airway(
    hu: np.ndarray,
    body: np.ndarray,
    interior_air: np.ndarray | None = None,
    soft_tissue: np.ndarray | None = None,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    y_anterior_is_low: bool = True,
    air_hu_max: float = -300.0,
    prior_left_mm: list[float] | None = None,
    prior_right_mm: list[float] | None = None,
    *,
    legacy_midplane_split: bool = True,
    superior_is_high_z: bool = True,
    flood_domain: np.ndarray | None = None,
) -> CTNasalAirwayResult:
    """
    Full CT-native nasal airway package for a cropped head volume.

    ``legacy_midplane_split=True`` (default until Visible Human --strict-septum)
    keeps ``split_left_right_by_septum_plane``. False uses competing geodesic
    flood (docs/segmentation_strategy.md).
    """
    notes: list[str] = []
    body = body.astype(bool)
    if interior_air is None:
        interior_air = body & (hu <= air_hu_max) & (hu >= -1024)
        interior_air = morphology.closing(interior_air, footprint=morphology.ball(1))
        notes.append(f"Interior air from HU ≤ {air_hu_max}.")
    else:
        interior_air = interior_air.astype(bool)

    def _mm_to_zyx(mm: list[float]) -> tuple[int, int, int]:
        sx, sy, sz = spacing_xyz
        ox, oy, oz = origin_xyz
        x = int(np.clip(round((mm[0] - ox) / sx), 0, hu.shape[2] - 1))
        y = int(np.clip(round((mm[1] - oy) / sy), 0, hu.shape[1] - 1))
        z = int(np.clip(round((mm[2] - oz) / sz), 0, hu.shape[0] - 1))
        return z, y, x

    prior_l_zyx = _mm_to_zyx(prior_left_mm) if prior_left_mm else None
    prior_r_zyx = _mm_to_zyx(prior_right_mm) if prior_right_mm else None
    z_hint = None
    if prior_l_zyx is not None and prior_r_zyx is not None:
        z_hint = int(0.5 * (prior_l_zyx[0] + prior_r_zyx[0]))

    # Primary: nostrils = anterior openings of CT nasal air (L/R clusters)
    openings, left_zyx, right_zyx, n1 = detect_nares_from_ct_air(
        hu,
        body,
        interior_air,
        y_anterior_is_low=y_anterior_is_low,
        air_hu_max=max(air_hu_max, -150.0),
        z_hint=z_hint,
        spacing_xyz=spacing_xyz,
    )
    notes.extend(n1)
    notes.append("Naris method: CT anterior air shell (not geometric skin tunnels).")

    # Fallback only if CT clustering failed. Track provenance per side: a seed
    # spliced in here is NOT a CT-shell seed, and must not be reported as one.
    src_l = src_r = "ct_air_shell"
    if left_zyx is None and prior_l_zyx is not None:
        left_zyx = prior_l_zyx
        src_l = "prior"
        notes.append("Left naris fell back to prior landmark.")
    if right_zyx is None and prior_r_zyx is not None:
        right_zyx = prior_r_zyx
        src_r = "prior"
        notes.append("Right naris fell back to prior landmark.")

    # Two distinct seeds or HARD-fail (docs/handoff.md §9 item 1). The detector
    # can return a collapsed or relocated pair without returning None — on THCA
    # it returned the *same voxel twice* — so validate the pair rather than
    # trusting non-None. Never invent a seed: the old repair nudged x by two
    # voxels, which left both seeds on the same side of the septum.
    # Midline reference for the straddle test. The prior naris midpoint is the
    # right anchor; the body centroid is skewed on an off-centre head (CQ500CT390:
    # centroid x=221 against a true midline of ~189) so it is only a fallback.
    if prior_l_zyx is not None and prior_r_zyx is not None:
        x_midline = 0.5 * (float(prior_l_zyx[2]) + float(prior_r_zyx[2]))
    elif body.any():
        x_midline = float(np.where(body)[2].mean())
    else:
        x_midline = None
    ct_ok, ct_why = validate_naris_pair(
        left_zyx, right_zyx, spacing_xyz, x_midline=x_midline
    )
    if ct_ok:
        naris_source = src_l if src_l == src_r else f"mixed(L={src_l},R={src_r})"
        notes.append(f"Naris pair accepted [{naris_source}]: {ct_why}.")
    else:
        notes.append(f"WARN: naris pair rejected [{src_l}/{src_r}]: {ct_why}.")
        prior_ok, prior_why = validate_naris_pair(
            prior_l_zyx, prior_r_zyx, spacing_xyz, x_midline=x_midline
        )
        if prior_ok:
            left_zyx, right_zyx = prior_l_zyx, prior_r_zyx
            naris_source = "prior"
            notes.append(
                f"Naris pair from whole-head prior landmarks: {prior_why}."
            )
        else:
            notes.append(f"ERROR: prior naris pair also rejected: {prior_why}.")
            left_zyx = right_zyx = None
            naris_source = "none"

    if left_zyx is None or right_zyx is None:
        notes.append("ERROR: need two distinct naris landmarks; HARD fail.")
        empty = np.zeros_like(body)
        return CTNasalAirwayResult(
            interior_air=interior_air,
            left_cavity=empty,
            right_cavity=empty,
            passage_lumen=interior_air,
            septum=empty,
            mucosa_wall=empty,
            naris_opening=openings,
            left_naris_center_zyx=left_zyx,
            right_naris_center_zyx=right_zyx,
            left_naris_center_mm=_zyx_to_mm(left_zyx, spacing_xyz, origin_xyz)
            if left_zyx
            else None,
            right_naris_center_mm=_zyx_to_mm(right_zyx, spacing_xyz, origin_xyz)
            if right_zyx
            else None,
            method="naris_detection_HARD_fail",
            naris_source="none",
            notes=notes,
        )

    x_sep = int(round(0.5 * (left_zyx[2] + right_zyx[2])))
    paint_sep = x_sep if legacy_midplane_split else None
    if not legacy_midplane_split:
        notes.append("legacy_midplane_split=False: competing geodesic flood (no x_sep paint)")
    else:
        notes.append(
            f"Septum plane x={x_sep} centered between nares "
            f"L_x={left_zyx[2]} R_x={right_zyx[2]}."
        )

    # Teacher 3b: vestibule paint. No x_sep on the new path (sealed 1 mm nares).
    if openings.any():
        bridge = morphology.dilation(openings, footprint=morphology.ball(3)) & body
        vestibule = bridge & (hu <= air_hu_max + 100)
        interior_air = interior_air | vestibule

    for seed, side in ((left_zyx, "left"), (right_zyx, "right")):
        before = int(interior_air.sum())
        interior_air = _paint_corridor_to_air(
            interior_air,
            body,
            hu,
            seed,
            interior_air,
            y_anterior_is_low=y_anterior_is_low,
            radius=3,
            air_hu_max=air_hu_max + 100,
            side=side,
            x_sep=paint_sep,
        )
        notes.append(
            f"Vestibule {side}: +{int(interior_air.sum()) - before} voxels "
            f"from naris {seed}."
        )

    merge_zone = None
    choanal_landmark = None
    sinus_detour = None
    method = "ct_topology_hu_edge"

    if legacy_midplane_split:
        left, right, x_sep, n3 = split_left_right_by_septum_plane(
            interior_air,
            left_zyx,
            right_zyx,
            y_anterior_is_low=y_anterior_is_low,
        )
        notes.extend(n3)
        passage = left | right
        septum, mucosa, n4 = extract_septum_and_walls(
            body,
            left,
            right,
            interior_air,
            soft_or_tissue=soft_tissue,
            x_sep=x_sep,
            left_naris_zyx=left_zyx,
            right_naris_zyx=right_zyx,
            y_anterior_is_low=y_anterior_is_low,
            half_width=3,
        )
        notes.extend(n4)
    else:
        from .auto_airway import (
            choanal_landmark_from_bone,
            competing_naris_flood,
            dead_end_sinus_strip,
            main_air_component,
            meeting_set,
            NARIS_SNAP_RADIUS_MM,
            nasal_box_mask,
            septum_ridge_from_cavities,
            snap_seed_to_air,
        )

        domain = interior_air.copy()
        if flood_domain is not None:
            domain = domain | flood_domain.astype(bool)
        box = nasal_box_mask(
            domain.shape, left_zyx, right_zyx, spacing_xyz, y_anterior_is_low
        )
        domain = domain & box
        # Snap onto the air the flood can actually traverse. A naris that lands on
        # a stranded pocket is not a naris: THCA's right seed once took a 55-voxel
        # island painted at the nostril and the flood returned 55 voxels for the
        # entire right cavity. The snap RADIUS still bounds the move, so this
        # cannot invent a connection across an obstruction.
        snap_domain = main_air_component(domain)
        snap_l = snap_seed_to_air(snap_domain, left_zyx, spacing_xyz)
        snap_r = snap_seed_to_air(snap_domain, right_zyx, spacing_xyz)
        if snap_l is None or snap_r is None:
            miss = " and ".join(
                s for s, v in (("left", snap_l), ("right", snap_r)) if v is None)
            notes.append(
                f"ERROR: {miss} naris seed did not snap onto the main flood-domain "
                f"air component within {NARIS_SNAP_RADIUS_MM:.0f} mm.")
            empty = np.zeros_like(body)
            return CTNasalAirwayResult(
                interior_air=interior_air,
                left_cavity=empty,
                right_cavity=empty,
                passage_lumen=interior_air,
                septum=empty,
                mucosa_wall=empty,
                naris_opening=openings,
                left_naris_center_zyx=left_zyx,
                right_naris_center_zyx=right_zyx,
                left_naris_center_mm=_zyx_to_mm(left_zyx, spacing_xyz, origin_xyz),
                right_naris_center_mm=_zyx_to_mm(right_zyx, spacing_xyz, origin_xyz),
                method="competing_geodesic_HARD_fail",
                naris_source=naris_source,
                notes=notes,
            )
        left_zyx, right_zyx = snap_l, snap_r
        left, right, d_l, d_r, n3 = competing_naris_flood(
            domain, left_zyx, right_zyx, spacing_xyz
        )
        notes.extend(n3)
        meet = meeting_set(left, right, d_l, d_r)
        choanal_landmark, merge_zone, gate_meta = choanal_landmark_from_bone(
            hu,
            domain,
            left_zyx,
            right_zyx,
            y_anterior_is_low=y_anterior_is_low,
            superior_is_high_z=superior_is_high_z,
            spacing_xyz=spacing_xyz,
            meeting_set=meet,
        )
        notes.extend(gate_meta.get("notes") or [])
        if merge_zone is not None and merge_zone.any():
            left_ant = left & ~merge_zone
            right_ant = right & ~merge_zone
        else:
            left_ant, right_ant = left, right
        flooded = left | right
        # Run the strip on the ANATOMICAL airway, not on the nasal-box-cropped
        # flood domain: the box's faces are artificial cuts that read as wide
        # openings, so nothing there looks like it sits behind a neck. Then
        # apply the verdict to the flooded region.
        strip_domain = (
            flood_domain.astype(bool) if flood_domain is not None else flooded
        )
        _strip_passage, sinus_full, n_tp = dead_end_sinus_strip(
            strip_domain, spacing_xyz, merge_zone=merge_zone
        )
        notes.extend(n_tp)
        # The strip's verdict is on the WHOLE flood domain. A sinus behind the
        # nasal box's posterior face (the sphenoid, THCA: 8.9 mL core) is still
        # a sinus: keep it in sinus_detour so drainage names it and the export
        # excludes it, instead of clipping the record to the box.
        sinus_detour = sinus_full.copy()
        passage = flooded & ~sinus_detour
        # The strip must never delete a cavity. It cannot by construction (a
        # basin reaching an opening is rejected), but keep the guard so a future
        # change cannot silently ship a one-sided domain.
        left_keep = int((left_ant & passage).sum())
        right_keep = int((right_ant & passage).sum())
        if left_keep == 0 or right_keep == 0:
            notes.append(
                "WARN: sinus strip emptied a cavity; keeping competing assignment"
            )
            passage = flooded
            sinus_detour = np.zeros_like(flooded)
            sinus_full = np.zeros_like(flooded)
        left = left_ant & passage
        right = right_ant & passage
        if merge_zone is not None:
            # The merge zone is a half-space: it held the sphenoid, and OR-ing
            # it back wholesale undid the strip's verdict behind the landmark
            # (THCA 2026-09-04: the strip removed the sphenoid, this line put
            # it back). Only merge-zone air the strip did not call sinus.
            passage = left | right | (merge_zone & ~sinus_full)
        else:
            passage = left | right
        # Glue the whole-head nasopharynx/trachea continuation back on. The
        # nasal box crops the flood domain for speed, but the CFD outlet BC
        # lives at the trachea, outside that box: without this the outlet port
        # does not resolve onto passage_lumen and there is no flow path to
        # solve (VH Male and THCA both failed exactly here). Only air OUTSIDE
        # the box and connected to the passage is added, so the sinuses --
        # which are inside the box -- stay stripped.
        if flood_domain is not None:
            # ... and never glue back what the strip called sinus: the sphenoid
            # can lie behind the box's posterior face, exactly where this looks.
            outside = flood_domain.astype(bool) & ~box & ~sinus_full
            if outside.any():
                s26 = np.ones((3, 3, 3), dtype=bool)
                lab_g, _n = ndi.label(passage | outside, s26)
                ids = [int(v) for v in np.unique(lab_g[passage]) if v]
                if ids:
                    grown = np.isin(lab_g, ids)
                    added = int(grown.sum()) - int(np.asarray(passage).sum())
                    passage = grown
                    notes.append(
                        f"outlet glue: +{added} voxels of whole-head airway "
                        f"posterior of the nasal box (nasopharynx/trachea)"
                    )
        septum, _, _ = septum_ridge_from_cavities(body, left, right, spacing_xyz)
        mucosa = body & morphology.dilation(passage, footprint=morphology.ball(1))
        mucosa = mucosa & ~passage
        method = "competing_geodesic_flood"
        notes.append(
            f"anterior cavities L={int(left.sum())} R={int(right.sum())} "
            f"passage={int(np.asarray(passage).sum())}"
        )

    if not np.asarray(passage).any():
        passage = interior_air
        notes.append("WARNING: L/R empty — using full interior air as passage.")

    left_mm = _zyx_to_mm(left_zyx, spacing_xyz, origin_xyz) if left_zyx else None
    right_mm = _zyx_to_mm(right_zyx, spacing_xyz, origin_xyz) if right_zyx else None

    return CTNasalAirwayResult(
        interior_air=interior_air,
        left_cavity=left,
        right_cavity=right,
        passage_lumen=np.asarray(passage).astype(bool),
        septum=septum,
        mucosa_wall=mucosa,
        naris_opening=openings,
        left_naris_center_zyx=left_zyx,
        right_naris_center_zyx=right_zyx,
        left_naris_center_mm=left_mm,
        right_naris_center_mm=right_mm,
        method=method,
        naris_source=naris_source,
        notes=notes,
        merge_zone=merge_zone,
        choanal_landmark=choanal_landmark,
        sinus_detour=sinus_detour,
    )
