"""Instrument fit: can THIS tool reach THAT ostium from a nostril?

A centreline is not a path. A seeker is a rigid object with a diameter, a
curve or a fixed bend, and a working tip of a stated length; it must be placed
with its tip at the target, its whole body inside the airway with clearance
for its radius, and its shaft leaving through a naris -- without touching the
walls it must not lever against. That is a rigid-body placement problem in
three rotational degrees of freedom, and this module solves it by exhaustive
orientation search against the airway's distance field.

Tool geometry is a list of segments from the tip backwards:

  ("straight", length_mm)                 the working tip, or the shaft
  ("arc", radius_mm, angle_deg)           a curved section (a seeker's curve,
                                          or a fixed bend modelled as a tight
                                          arc)

Placing a tool: put the tip point at the target, choose the tip's direction
(a unit vector, sampled over the sphere) and a roll about it (sampled), then
march segment by segment, sampling points every ``STEP_MM``. A placement
FITS when every sample has wall distance >= tool radius AND the shaft reaches
the naris opening before it runs out. The result reports, for the best
placement found, the minimum clearance and where along the tool it occurs --
so a failure says which structure the tool would touch, not just "no".

What this does NOT do: bend the tool (no deformation), model soft-tissue
displacement, or know about structures the tool must not touch beyond the
airway wall itself. Those are stated limits, not omissions to hide.

Tool specifications are the roadmap's numbers where it gives them (frontal
seeker 2 mm; ET seeker 45 degree bend with an 18.5 mm working tip). Where the
roadmap gives no number the tool carries ``assumed=True`` and the report says
so: a verdict on an assumed diameter is not a verdict.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

STEP_MM = 0.5            # sample spacing along the tool
N_TIP_DIRECTIONS = 400   # Fibonacci sphere samples for the tip direction
N_ROLL = 24              # roll samples about the tip direction
SHAFT_REACH_MM = 120.0   # how far the shaft is followed looking for the naris


@dataclass(frozen=True)
class Instrument:
    name: str
    target: str                       # "frontal" | "maxillary" | "sphenoid" | "eustachian"
    diameter_mm: float
    segments: tuple[tuple, ...]       # from the tip backwards, see module doc
    assumed: tuple[str, ...] = ()     # which numbers are NOT from the roadmap

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm


# The roadmap (CLAUDE.md, goal 4) states: frontal seeker 2 mm diameter, curved;
# maxillary and sphenoid seekers "their own curvature"; ET seeker ~4 in shaft,
# 45 degree bend, 18.5 mm working tip past the bend. Everything else here is
# assumed and flagged. Curves are modelled as an arc of the stated radius; a
# fixed bend as a tight arc.
INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument(
        "frontal sinus seeker", "frontal", 2.0,
        (("straight", 10.0), ("arc", 20.0, 60.0), ("straight", SHAFT_REACH_MM)),
        assumed=("tip 10 mm", "curve radius 20 mm over 60 deg"),
    ),
    Instrument(
        "maxillary seeker", "maxillary", 2.0,
        (("straight", 8.0), ("arc", 12.0, 90.0), ("straight", SHAFT_REACH_MM)),
        assumed=("diameter 2 mm", "tip 8 mm", "curve radius 12 mm over 90 deg"),
    ),
    Instrument(
        "sphenoid seeker", "sphenoid", 2.0,
        (("straight", 10.0), ("arc", 30.0, 30.0), ("straight", SHAFT_REACH_MM)),
        assumed=("diameter 2 mm", "tip 10 mm", "curve radius 30 mm over 30 deg"),
    ),
    Instrument(
        "Eustachian tube seeker", "eustachian", 2.0,
        (("straight", 18.5), ("arc", 3.0, 45.0), ("straight", SHAFT_REACH_MM)),
        assumed=("diameter 2 mm", "bend radius 3 mm"),
    ),
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _frame(direction: np.ndarray, roll_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame (t, n, b): t = direction, n = bend plane normal-ish."""
    t = direction / max(np.linalg.norm(direction), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t, helper)
    n0 /= max(np.linalg.norm(n0), 1e-12)
    b0 = np.cross(t, n0)
    n = math.cos(roll_rad) * n0 + math.sin(roll_rad) * b0
    b = np.cross(t, n)
    return t, n, b


def tool_samples_mm(
    instrument: Instrument, tip_mm: np.ndarray, direction: np.ndarray, roll_rad: float,
    step_mm: float = STEP_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along the tool placed tip-first, marching backwards.

    ``direction`` is the direction the tip POINTS (from the tool towards the
    target), so the body extends along -direction. Returns (points, arc_mm) with
    arc_mm the distance from the tip.
    """
    t, n, _b = _frame(np.asarray(direction, dtype=np.float64), roll_rad)
    pos = np.asarray(tip_mm, dtype=np.float64).copy()
    heading = -t                      # marching direction along the body
    pts: list[np.ndarray] = [pos.copy()]
    arcs: list[float] = [0.0]
    s_total = 0.0
    for seg in instrument.segments:
        if seg[0] == "straight":
            length = float(seg[1])
            n_steps = max(int(math.ceil(length / step_mm)), 1)
            for k in range(1, n_steps + 1):
                d = min(k * step_mm, length)
                pts.append(pos + heading * d)
                arcs.append(s_total + d)
            pos = pos + heading * length
            s_total += length
        elif seg[0] == "arc":
            radius, angle = float(seg[1]), math.radians(float(seg[2]))
            length = radius * angle
            n_steps = max(int(math.ceil(length / step_mm)), 1)
            # rotate the heading about axis n (the bend plane normal) by angle,
            # walking the circle of the given radius
            centre = pos + np.cross(n, heading) * radius
            h0 = heading.copy()
            r0 = pos - centre
            for k in range(1, n_steps + 1):
                a = min(k * length / n_steps, length) / radius
                ca, sa = math.cos(a), math.sin(a)
                r = r0 * ca + np.cross(n, r0) * sa + n * np.dot(n, r0) * (1 - ca)
                pts.append(centre + r)
                arcs.append(s_total + a * radius)
            ca, sa = math.cos(angle), math.sin(angle)
            heading = h0 * ca + np.cross(n, h0) * sa + n * np.dot(n, h0) * (1 - ca)
            heading /= max(np.linalg.norm(heading), 1e-12)
            pos = pts[-1].copy()
            s_total += length
        else:
            raise ValueError(f"unknown segment type {seg[0]!r}")
    return np.asarray(pts), np.asarray(arcs)


# --------------------------------------------------------------------------
# placement search
# --------------------------------------------------------------------------


def _sample_field(field_arr: np.ndarray, pts_mm: np.ndarray, spacing_xyz, origin_xyz, order=1) -> np.ndarray:
    """Trilinear sample of a zyx array at xyz-mm points (outside the array -> 0)."""
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    coords = np.stack([
        (pts_mm[:, 2] - oz) / sz,   # z index
        (pts_mm[:, 1] - oy) / sy,   # y index
        (pts_mm[:, 0] - ox) / sx,   # x index
    ], axis=0)
    return ndi.map_coordinates(field_arr, coords, order=order, mode="constant", cval=0.0)


@dataclass
class Placement:
    fits: bool
    min_clearance_mm: float          # wall distance minus tool radius, worst sample
    worst_arc_mm: float              # distance from the tip where it occurs
    worst_point_mm: list[float]
    reaches_naris: bool
    exit_arc_mm: float | None        # where along the tool it leaves through the naris
    first_contact_arc_mm: float | None       # first wall contact from the tip, if any
    tip_direction: list[float]
    roll_deg: float
    points_mm: list[list[float]] = field(default_factory=list)


def place_instrument(
    instrument: Instrument,
    target_mm: np.ndarray,
    edt_mm: np.ndarray,
    inlet_open: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    n_dirs: int = N_TIP_DIRECTIONS,
    n_roll: int = N_ROLL,
    tip_hint: np.ndarray | None = None,
) -> Placement:
    """Exhaustive rigid placement: best orientation for the tool at the target.

    ``edt_mm``: distance-to-wall inside the airway (0 outside it), zyx.
    ``inlet_open``: the naris port voxels (the shaft must reach them), zyx.
    Score = min clearance over the part of the tool INSIDE the airway up to the
    naris exit; a placement whose shaft never reaches the naris is scored on
    its whole length and marked unreachable. ``tip_hint`` (a unit vector) keeps
    only tip directions within 90 degrees of it, halving the search when the
    approach side is known (e.g. from the ostium's outward normal).
    """
    dirs = fibonacci_sphere(n_dirs)
    if tip_hint is not None:
        h = np.asarray(tip_hint, dtype=np.float64)
        h /= max(np.linalg.norm(h), 1e-12)
        keep = dirs @ h > 0.0
        if keep.any():
            dirs = dirs[keep]
    rolls = np.linspace(0.0, 2.0 * math.pi, n_roll, endpoint=False)
    radius = instrument.radius_mm
    inlet_f = inlet_open.astype(np.float32)

    def evaluate(d: np.ndarray, roll: float) -> tuple[Placement, np.ndarray, int]:
        pts, arcs = tool_samples_mm(instrument, target_mm, d, float(roll))
        dist = _sample_field(edt_mm, pts, spacing_xyz, origin_xyz)
        at_naris = _sample_field(inlet_f, pts, spacing_xyz, origin_xyz) > 0.5
        # the tool is followed until it exits through the naris; beyond that
        # it is outside the head and clearance means nothing
        exit_idx = int(np.argmax(at_naris)) if at_naris.any() else None
        last = exit_idx if exit_idx is not None else len(pts) - 1
        clear = dist[: last + 1] - radius
        k = int(np.argmin(clear))
        bad = np.where(clear < 0.0)[0]
        cand = Placement(
            fits=bool(clear[k] >= 0.0 and exit_idx is not None),
            min_clearance_mm=float(clear[k]),
            worst_arc_mm=float(arcs[k]),
            worst_point_mm=[float(v) for v in pts[k]],
            reaches_naris=exit_idx is not None,
            exit_arc_mm=float(arcs[exit_idx]) if exit_idx is not None else None,
            first_contact_arc_mm=float(arcs[bad[0]]) if bad.size else None,
            tip_direction=[float(v) for v in d],
            roll_deg=float(math.degrees(roll)),
        )
        return cand, pts, last

    best: Placement | None = None
    best_pts = None
    for d in dirs:
        for roll in rolls:
            cand, pts, last = evaluate(d, roll)
            if best is None or _better(cand, best):
                best, best_pts = cand, pts[: last + 1]
    assert best is not None
    # A straight run down a narrow channel is missed by a coarse sphere: 400
    # directions are ~10 degrees apart and 10 degrees over 36 mm is 6 mm off
    # axis. Refine in shrinking cones about the best orientation found.
    for half_deg, n_c, roll_half_deg, n_r in ((15.0, 120, 30.0, 9), (3.0, 80, 6.0, 7), (0.6, 40, 1.2, 5)):
        d0 = np.asarray(best.tip_direction)
        r0 = math.radians(best.roll_deg)
        cone = _cone_dirs(d0, math.radians(half_deg), n_c)
        rr = r0 + np.linspace(-math.radians(roll_half_deg), math.radians(roll_half_deg), n_r)
        for d in cone:
            for roll in rr:
                cand, pts, last = evaluate(d, float(roll))
                if _better(cand, best):
                    best, best_pts = cand, pts[: last + 1]
    best.points_mm = [[float(v) for v in p] for p in best_pts]
    return best


def _cone_dirs(axis: np.ndarray, half_angle: float, n: int) -> np.ndarray:
    """``n`` unit vectors within ``half_angle`` of ``axis`` (axis included)."""
    t, nrm, b = _frame(np.asarray(axis, dtype=np.float64), 0.0)
    i = np.arange(n, dtype=np.float64) + 0.5
    a = half_angle * np.sqrt(i / n)                     # uniform in solid angle (small cone)
    phi = math.pi * (1.0 + 5.0 ** 0.5) * i
    out = (np.cos(a)[:, None] * t[None, :]
           + np.sin(a)[:, None] * (np.cos(phi)[:, None] * nrm[None, :] + np.sin(phi)[:, None] * b[None, :]))
    return np.vstack([t[None, :], out])


def _better(a: Placement, b: Placement) -> bool:
    """Order placements so the search has a gradient even when every coarse
    orientation fails: fitting beats not, then reaching the naris, then the
    tool that gets FURTHEST before touching a wall, then the larger clearance.
    Without the contact-depth term every leaving-the-airway placement scores
    the same -radius and refinement circles an arbitrary one."""
    if a.fits != b.fits:
        return a.fits
    if a.reaches_naris != b.reaches_naris:
        return a.reaches_naris
    ca, cb = a.first_contact_arc_mm, b.first_contact_arc_mm
    if (ca is None) != (cb is None):
        return ca is None
    if ca is not None and cb is not None and abs(ca - cb) > 1e-9:
        return ca > cb
    return a.min_clearance_mm > b.min_clearance_mm


# --------------------------------------------------------------------------
# case-level driver
# --------------------------------------------------------------------------


def airway_wall_distance_mm(airway: np.ndarray, spacing_xyz) -> np.ndarray:
    sx, sy, sz = spacing_xyz
    return ndi.distance_transform_edt(airway.astype(bool), sampling=(sz, sy, sx)).astype(np.float32)


def zyx_to_mm(zyx, spacing_xyz, origin_xyz) -> np.ndarray:
    z, y, x = zyx
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    return np.array([ox + x * sx, oy + y * sy, oz + z * sz], dtype=np.float64)


def fit_instruments_to_ostia(
    sinus_records: list[dict[str, Any]],
    airway: np.ndarray,
    inlet_open: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    instruments: tuple[Instrument, ...] = INSTRUMENTS,
    keep_points: bool = True,
) -> dict[str, Any]:
    """One verdict per (instrument, detected ostium of its target sinus).

    Sinuses without a valid ostium, and instruments whose target has no
    detection (the Eustachian tube orifice has no detector yet), are reported
    as ``no target`` -- explicitly, so the gap is visible in the output.
    """
    edt = airway_wall_distance_mm(airway, spacing_xyz)
    out: list[dict[str, Any]] = []
    for inst in instruments:
        recs = [r for r in sinus_records
                if r.get("name") == inst.target and r.get("ostium_zyx") is not None
                and r.get("ostium_valid", True)]
        if not recs:
            out.append({
                "instrument": inst.name, "target": inst.target, "side": None,
                "verdict": "no target",
                "reason": (f"no detected {inst.target} ostium in this case"
                           if inst.target != "eustachian"
                           else "no Eustachian tube orifice detector exists yet"),
                "assumed": list(inst.assumed),
            })
            continue
        for r in recs:
            tgt = zyx_to_mm(r["ostium_zyx"], spacing_xyz, origin_xyz)
            pl = place_instrument(inst, tgt, edt, inlet_open, spacing_xyz, origin_xyz)
            if pl.fits:
                verdict, reason = "fits", (
                    f"min clearance {pl.min_clearance_mm:.2f} mm at {pl.worst_arc_mm:.1f} mm "
                    f"from the tip; shaft exits the naris {pl.exit_arc_mm:.1f} mm from the tip")
            elif not pl.reaches_naris:
                verdict, reason = "unreachable", (
                    "no rigid placement with the tip at the ostium brings the shaft out "
                    f"through a naris within {SHAFT_REACH_MM:.0f} mm")
            else:
                verdict, reason = "touches wall", (
                    f"best placement is {-pl.min_clearance_mm:.2f} mm short of clearance at "
                    f"{pl.worst_arc_mm:.1f} mm from the tip (point {np.round(pl.worst_point_mm, 1).tolist()} mm)")
            rec = {
                "instrument": inst.name, "target": inst.target, "side": r.get("side"),
                "ostium_diameter_mm": r.get("ostium_diameter_mm"),
                "ostium_mm": [float(v) for v in tgt],
                "verdict": verdict, "reason": reason,
                "assumed": list(inst.assumed),
                "placement": {k: v for k, v in asdict(pl).items() if k != "points_mm"},
            }
            if keep_points:
                rec["path_mm"] = pl.points_mm
            out.append(rec)
    return {
        "method": "rigid placement, exhaustive orientation search",
        "step_mm": STEP_MM, "n_tip_directions": N_TIP_DIRECTIONS, "n_roll": N_ROLL,
        "limits": [
            "rigid tool: no bending, no soft-tissue displacement",
            "clearance is against the airway wall only; no list of forbidden structures yet",
            "diameters and curves marked assumed are placeholders, not specifications",
        ],
        "fits": out,
    }
