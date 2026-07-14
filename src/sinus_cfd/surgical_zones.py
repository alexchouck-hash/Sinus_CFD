"""
Anatomic classification of high-flow / constriction zones for surgical planning.

Zones (along naris→trachea airway):
  - inferior_turbinate: lateral, inferior — along maxillary / inferior meatus
  - middle_turbinate: mid-height, para-septal — splits nasal airflow
  - septum: distal–medial (near midline of nasal passage)

Treatment recommendations prioritize least invasive options that address
the dominant high-|u| sites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


@dataclass
class ZoneStats:
    name: str
    label: str
    voxels: int
    mean_speed_m_s: float
    max_speed_m_s: float
    center_mm: list[float]
    severity: str  # mild / moderate / marked
    notes: str = ""


@dataclass
class TreatmentOption:
    name: str
    category: str  # airflow | sinus_drainage
    addresses: list[str]  # zone keys
    invasiveness: int  # 1=least … 5=most
    description: str
    recommended: bool = False
    reason: str = ""


def _severity(mean_sp: float, max_sp: float, ref_mean: float) -> str:
    if mean_sp >= ref_mean * 1.4 or max_sp >= ref_mean * 2.5:
        return "marked"
    if mean_sp >= ref_mean * 1.1 or max_sp >= ref_mean * 1.6:
        return "moderate"
    return "mild"


def classify_removal_zones(
    highlight: np.ndarray,
    speed: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    naris_mm: list[list[float]] | None = None,
    nasal_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], list[ZoneStats], list[str]]:
    """
    Split high-|u| highlight into inferior turbinate, middle turbinate, septum.
    Returns (masks_dict, stats, notes).
    """
    notes: list[str] = []
    hl = highlight.astype(bool)
    empty = np.zeros_like(hl)
    if not hl.any():
        return (
            {
                "inferior_turbinate": empty,
                "middle_turbinate": empty,
                "septum": empty,
            },
            [],
            ["No highlight voxels to classify."],
        )

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    nz, ny, nx = hl.shape
    zz, yy, xx = np.where(hl)

    # Midplane from nares if available, else from highlight x
    if naris_mm and len(naris_mm) >= 2:
        x_mid = 0.5 * (float(naris_mm[0][0]) + float(naris_mm[1][0]))
    else:
        x_mid = float(ox + np.median(xx) * sx)

    # Nasal AP/sup bounds for region-relative bands
    if nasal_mask is not None and nasal_mask.any():
        nz_a, ny_a, nx_a = np.where(nasal_mask)
        z_lo, z_hi = float(nz_a.min()), float(nz_a.max())
        y_lo, y_hi = float(ny_a.min()), float(ny_a.max())
    else:
        z_lo, z_hi = float(zz.min()), float(zz.max())
        y_lo, y_hi = float(yy.min()), float(yy.max())

    z_span = max(z_hi - z_lo, 1.0)
    y_span = max(y_hi - y_lo, 1.0)
    # Inferior = lower z (superior_is_high_z)
    z_inf_cut = z_lo + 0.42 * z_span
    z_mid_lo = z_lo + 0.28 * z_span
    z_mid_hi = z_lo + 0.72 * z_span
    # Distal septum: more posterior y (away from tip)
    y_distal = y_lo + 0.35 * y_span

    # Physical half-widths
    septum_half = 7.0  # mm from midplane
    mid_turb_half_lo = 6.0
    mid_turb_half_hi = 16.0
    inf_turb_lat = 9.0  # mm lateral of midplane

    wx = ox + xx * sx
    lat = np.abs(wx - x_mid)

    # Septum: medial + spans mid/distal nasal (not pure tip vestibule only)
    septum_sel = (lat <= septum_half) & (yy.astype(float) >= y_lo + 0.12 * y_span)
    # Inferior turbinate: lateral + inferior
    inf_sel = (lat >= inf_turb_lat) & (zz.astype(float) <= z_inf_cut)
    # Middle turbinate: para-septal to mid-lateral, mid height (splits airflow)
    mid_sel = (
        (lat >= mid_turb_half_lo)
        & (lat <= mid_turb_half_hi)
        & (zz.astype(float) >= z_mid_lo)
        & (zz.astype(float) <= z_mid_hi)
    )
    # Avoid double-count: priority septum > middle > inferior for overlaps
    mid_sel = mid_sel & ~septum_sel
    inf_sel = inf_sel & ~septum_sel & ~mid_sel

    # Unclassified high-flow → assign by nearest rule
    unassigned = ~(septum_sel | mid_sel | inf_sel)
    if unassigned.any():
        # medial leftovers → septum; inferior leftovers → IT; else MT
        med = unassigned & (lat <= 10.0)
        septum_sel = septum_sel | med
        unassigned = unassigned & ~med
        inf2 = unassigned & (zz.astype(float) <= z_inf_cut)
        inf_sel = inf_sel | inf2
        unassigned = unassigned & ~inf2
        mid_sel = mid_sel | unassigned

    masks = {
        "septum": empty.copy(),
        "middle_turbinate": empty.copy(),
        "inferior_turbinate": empty.copy(),
    }
    masks["septum"][zz[septum_sel], yy[septum_sel], xx[septum_sel]] = True
    masks["middle_turbinate"][zz[mid_sel], yy[mid_sel], xx[mid_sel]] = True
    masks["inferior_turbinate"][zz[inf_sel], yy[inf_sel], xx[inf_sel]] = True

    # Small dilation for visibility (keep separate)
    for k in masks:
        if masks[k].any():
            masks[k] = morphology.binary_dilation(
                masks[k], footprint=morphology.ball(1)
            ) & hl

    notes.append(
        f"Zone split: septum={int(masks['septum'].sum())} "
        f"middle_turb={int(masks['middle_turbinate'].sum())} "
        f"inferior_turb={int(masks['inferior_turbinate'].sum())} "
        f"x_mid={x_mid:.1f} mm"
    )

    # Global ref for severity
    sp_all = speed[hl]
    ref_mean = float(sp_all.mean()) if sp_all.size else 0.3

    label_map = {
        "inferior_turbinate": "Inferior turbinate (lateral / maxillary corridor)",
        "middle_turbinate": "Middle turbinate (splits nasal airflow)",
        "septum": "Septum (distal–medial)",
    }
    stats: list[ZoneStats] = []
    for key, lab in label_map.items():
        m = masks[key]
        if not m.any():
            stats.append(
                ZoneStats(
                    name=key,
                    label=lab,
                    voxels=0,
                    mean_speed_m_s=0.0,
                    max_speed_m_s=0.0,
                    center_mm=[x_mid, float(oy + 0.5 * (y_lo + y_hi) * sy), float(oz)],
                    severity="none",
                    notes="No high-|u| focus in this zone.",
                )
            )
            continue
        sp = speed[m]
        zz2, yy2, xx2 = np.where(m)
        center = [
            float(ox + xx2.mean() * sx),
            float(oy + yy2.mean() * sy),
            float(oz + zz2.mean() * sz),
        ]
        sev = _severity(float(sp.mean()), float(sp.max()), ref_mean)
        stats.append(
            ZoneStats(
                name=key,
                label=lab,
                voxels=int(m.sum()),
                mean_speed_m_s=float(sp.mean()),
                max_speed_m_s=float(sp.max()),
                center_mm=center,
                severity=sev,
                notes=f"High-flow focus ({sev}) — candidate tissue reduction site.",
            )
        )

    return masks, stats, notes


def points_from_mask(
    mask: np.ndarray,
    speed: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    max_points: int = 4000,
    seed: int = 3,
) -> np.ndarray:
    """Nx4 array x,y,z,speed."""
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    zz, yy, xx = np.where(mask.astype(bool))
    if len(zz) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if len(zz) > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(zz), size=max_points, replace=False)
        zz, yy, xx = zz[pick], yy[pick], xx[pick]
    return np.column_stack(
        [
            ox + xx * sx,
            oy + yy * sy,
            oz + zz * sz,
            speed[zz, yy, xx],
        ]
    ).astype(np.float32)


def recommend_treatments(zone_stats: list[ZoneStats]) -> list[TreatmentOption]:
    """
    Least-invasive first recommendations for airflow and CRS drainage.
    """
    by = {z.name: z for z in zone_stats}
    active = {
        k: z
        for k, z in by.items()
        if z.voxels > 0 and z.severity in ("mild", "moderate", "marked")
    }

    options: list[TreatmentOption] = [
        TreatmentOption(
            name="Inferior turbinate reduction (RF or microdebrider)",
            category="airflow",
            addresses=["inferior_turbinate"],
            invasiveness=2,
            description=(
                "Reduce inferior turbinate bulk to open the inferior meatus / "
                "lateral nasal corridor. RF is tissue-sparing; microdebrider for "
                "more controlled volume reduction."
            ),
        ),
        TreatmentOption(
            name="Middle turbinate reduction (RF or microdebrider)",
            category="airflow",
            addresses=["middle_turbinate"],
            invasiveness=2,
            description=(
                "Address middle turbinate contortion/concha that splits and "
                "accelerates nasal airflow; preserve medial lamella when possible."
            ),
        ),
        TreatmentOption(
            name="Septoplasty — caudal / anterior deviation",
            category="airflow",
            addresses=["septum"],
            invasiveness=3,
            description=(
                "Correct caudal septal deviation near the valve / anterior septum "
                "to enlarge the primary inspiratory aperture with limited dissection."
            ),
        ),
        TreatmentOption(
            name="Septoplasty — posterior / distal deviation",
            category="airflow",
            addresses=["septum"],
            invasiveness=3,
            description=(
                "Address distal–medial septal spur/deviation where high-speed "
                "jets form along the septum further into the nasal passage."
            ),
        ),
        TreatmentOption(
            name="Nasal valve support (graft / implant / lateral wall)",
            category="airflow",
            addresses=["septum", "inferior_turbinate"],
            invasiveness=3,
            description=(
                "If collapse or dynamic narrowing dominates near the entry "
                "(high speed at anterior medial/lateral), consider valve support "
                "before or with limited turbinate work."
            ),
        ),
        TreatmentOption(
            name="Balloon sinus dilation",
            category="sinus_drainage",
            addresses=["middle_turbinate"],
            invasiveness=2,
            description=(
                "Office or OR balloon dilation of obstructed ostia to improve "
                "drainage with minimal tissue removal — first-line for selected CRS."
            ),
        ),
        TreatmentOption(
            name="Maxillary antrostomy",
            category="sinus_drainage",
            addresses=["inferior_turbinate", "middle_turbinate"],
            invasiveness=3,
            description=(
                "Widen the maxillary ostium for chronic maxillary disease when "
                "medical therapy and balloon fail; often combined with limited "
                "uncinectomy / middle meatus work."
            ),
        ),
        TreatmentOption(
            name="Frontal sinus drillout (Draf-type)",
            category="sinus_drainage",
            addresses=["middle_turbinate"],
            invasiveness=5,
            description=(
                "Expanded frontal drainage for refractory frontal CRS; reserve "
                "when less invasive frontal pathways and balloon have failed."
            ),
        ),
    ]

    # Score recommendation: match severity-weighted zones, prefer low invasiveness
    sev_w = {"none": 0, "mild": 1, "moderate": 2, "marked": 3}
    for opt in options:
        score = 0
        reasons = []
        for zname in opt.addresses:
            z = by.get(zname)
            if z is None or z.voxels == 0:
                continue
            w = sev_w.get(z.severity, 0)
            if w > 0:
                score += w * 10 + min(z.voxels // 50, 20)
                reasons.append(f"{z.label}: {z.severity} (n={z.voxels})")
        # Prefer less invasive when scores similar
        opt.recommended = score > 0
        opt.reason = "; ".join(reasons) if reasons else "No matching high-flow zone."
        # store temp score on object via description? use invasiveness for sort
        opt._score = score - opt.invasiveness  # type: ignore[attr-defined]

    # Rank recommended by score
    ranked = sorted(
        [o for o in options if o.recommended],
        key=lambda o: (-getattr(o, "_score", 0), o.invasiveness),
    )
    # Keep top airflow (up to 3) + top drainage (up to 2), always list others as optional
    top: list[TreatmentOption] = []
    n_air, n_sin = 0, 0
    for o in ranked:
        if o.category == "airflow" and n_air < 3:
            top.append(o)
            n_air += 1
        elif o.category == "sinus_drainage" and n_sin < 2:
            top.append(o)
            n_sin += 1
    # Mark only top as recommended; demote rest
    top_names = {o.name for o in top}
    for o in options:
        if o.name not in top_names:
            o.recommended = False
        else:
            o.recommended = True

    # Sort: recommended first, then invasiveness
    options.sort(key=lambda o: (not o.recommended, o.invasiveness, o.name))

    # Minimal-intervention narrative note
    if not any(o.recommended for o in options):
        options.insert(
            0,
            TreatmentOption(
                name="Medical / observation first",
                category="airflow",
                addresses=[],
                invasiveness=1,
                description=(
                    "No marked high-flow surgical target on current CFD map. "
                    "Optimize saline, topical steroids, allergy control; reassess."
                ),
                recommended=True,
                reason="Low zone severity on current map.",
            ),
        )

    return options


def zones_to_meta(
    zone_stats: list[ZoneStats],
    treatments: list[TreatmentOption],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "zones": [asdict(z) for z in zone_stats],
        "treatments": [
            {
                "name": t.name,
                "category": t.category,
                "addresses": t.addresses,
                "invasiveness": t.invasiveness,
                "description": t.description,
                "recommended": t.recommended,
                "reason": t.reason,
            }
            for t in treatments
        ],
        "notes": notes,
    }
