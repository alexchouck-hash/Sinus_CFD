"""
Virtual surgery — parameterized edits to the airway label map (Stage 4).

Digitally mimics the common nasal-airway-obstruction operations by growing the
air lumen into adjacent tissue, so the same geometry -> mesh -> CFD pipeline can
be re-run on the edited anatomy and compared pre/post (resistance, MCA, airflow
allocation, mucosal cooling).

Honest scope. These are *geometric* mimics, not tissue-accurate surgery — we
grow the air region rather than model removed turbinate/septal tissue, because
the NasalSeg labels give the air lumen (L/R nasal cavity, nasopharynx) but not
turbinate or septal cartilage. The literature validates the *direction* and
rough *magnitude* of the simulated change for turbinate reduction and
septoplasty, not exact per-patient outcomes; the tool ranks candidate edits by
predicted metric improvement, it does not auto-prescribe which tissue to remove.

Anatomical convention (matches the rest of the repo): array axes are (z, y, x);
the septum midline is in x, between the left (label 1) and right (label 2)
nasal-cavity centroids. "Lateral" = away from that midline; "medial" = toward it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEFT_CAVITY = 1
RIGHT_CAVITY = 2
NASOPHARYNX = 3


@dataclass
class SurgeryResult:
    edited_label: np.ndarray
    procedure: str
    side: str
    depth_mm: float
    added_voxels: int
    added_volume_ml: float
    notes: list[str] = field(default_factory=list)


def _shift_x(mask: np.ndarray, dx: int) -> np.ndarray:
    """Shift a boolean volume by dx voxels along x (axis 2), zero-filled."""
    out = np.zeros_like(mask)
    if dx > 0:
        out[:, :, dx:] = mask[:, :, :-dx]
    elif dx < 0:
        out[:, :, :dx] = mask[:, :, -dx:]
    else:
        out[:] = mask
    return out


def _septum_x(label_zyx: np.ndarray) -> int | None:
    """Midline x index between the L and R nasal-cavity centroids."""
    xl = np.where(label_zyx == LEFT_CAVITY)[2]
    xr = np.where(label_zyx == RIGHT_CAVITY)[2]
    if len(xl) == 0 or len(xr) == 0:
        return None
    return int(round(0.5 * (np.median(xl) + np.median(xr))))


def turbinate_reduction(
    label_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    side: str = "left",
    depth_mm: float = 3.0,
) -> SurgeryResult:
    """
    Inferior/lateral turbinate reduction: grow the nasal-cavity air *laterally*
    (away from the septum midline) by ``depth_mm``, into adjacent tissue voxels.

    The turbinates protrude from the lateral nasal wall, so lateral expansion of
    the lumen mimics shaving them back. Growth only fills background/tissue
    voxels (label 0) — it never overwrites the contralateral cavity or the
    nasopharynx.
    """
    side_label = LEFT_CAVITY if side == "left" else RIGHT_CAVITY
    notes: list[str] = []
    cavity = label_zyx == side_label
    if not cavity.any():
        return SurgeryResult(label_zyx.copy(), "turbinate_reduction", side, depth_mm, 0, 0.0,
                             [f"{side} nasal cavity absent — no edit applied."])

    x_sep = _septum_x(label_zyx)
    x_med = float(np.median(np.where(cavity)[2]))
    if x_sep is None:
        x_sep = int(round(label_zyx.shape[2] / 2))
        notes.append("Septum midline fell back to volume centre (one cavity missing).")
    lateral_sign = 1 if x_med >= x_sep else -1

    sx = spacing_xyz[0]
    n = max(1, int(round(depth_mm / sx)))
    grown = cavity.copy()
    for k in range(1, n + 1):
        grown |= _shift_x(cavity, lateral_sign * k)

    new_air = grown & (label_zyx == 0)  # only into tissue/background
    out = label_zyx.copy()
    out[new_air] = side_label

    added = int(new_air.sum())
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0
    notes.append(
        f"Grew {side} cavity {n} voxels (~{depth_mm:.1f} mm) laterally "
        f"(x {'+' if lateral_sign > 0 else '-'}), +{added} air voxels."
    )
    return SurgeryResult(out, "turbinate_reduction", side, depth_mm, added, added * vox_ml, notes)


def septoplasty(
    label_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    side: str = "left",
    depth_mm: float = 3.0,
) -> SurgeryResult:
    """
    Septoplasty: straighten a deviated septum by growing the *obstructed* side's
    air *medially* (toward the septum midline) by ``depth_mm``, mimicking the
    deviated septal wall being moved back to midline.

    ``side`` is the narrower/obstructed side (from the Stage-2 L/R MCA ratio).
    Growth is clipped at the midline so it cannot cross into the other cavity.
    """
    side_label = LEFT_CAVITY if side == "left" else RIGHT_CAVITY
    notes: list[str] = []
    cavity = label_zyx == side_label
    if not cavity.any():
        return SurgeryResult(label_zyx.copy(), "septoplasty", side, depth_mm, 0, 0.0,
                             [f"{side} nasal cavity absent — no edit applied."])

    x_sep = _septum_x(label_zyx)
    if x_sep is None:
        return SurgeryResult(label_zyx.copy(), "septoplasty", side, depth_mm, 0, 0.0,
                             ["Cannot locate septum midline (one cavity missing)."])
    x_med = float(np.median(np.where(cavity)[2]))
    medial_sign = -1 if x_med >= x_sep else 1  # toward the midline

    sx = spacing_xyz[0]
    n = max(1, int(round(depth_mm / sx)))
    grown = cavity.copy()
    for k in range(1, n + 1):
        grown |= _shift_x(cavity, medial_sign * k)

    # Clip at the midline so the widened side stays on its own side of the septum.
    xx = np.arange(label_zyx.shape[2])[None, None, :]
    if medial_sign < 0:  # side is on the high-x side, grow toward lower x but not past x_sep
        grown &= xx >= x_sep
    else:
        grown &= xx <= x_sep

    new_air = grown & (label_zyx == 0)
    out = label_zyx.copy()
    out[new_air] = side_label

    added = int(new_air.sum())
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0
    notes.append(
        f"Grew {side} cavity {n} voxels (~{depth_mm:.1f} mm) medially toward "
        f"midline x={x_sep} (clipped at midline), +{added} air voxels."
    )
    return SurgeryResult(out, "septoplasty", side, depth_mm, added, added * vox_ml, notes)


PROCEDURES = {
    "turbinate_reduction": turbinate_reduction,
    "septoplasty": septoplasty,
}


def apply(
    procedure: str,
    label_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    side: str = "left",
    depth_mm: float = 3.0,
) -> SurgeryResult:
    if procedure not in PROCEDURES:
        raise ValueError(f"Unknown procedure {procedure!r}; choose from {list(PROCEDURES)}")
    return PROCEDURES[procedure](label_zyx, spacing_xyz, side=side, depth_mm=depth_mm)
