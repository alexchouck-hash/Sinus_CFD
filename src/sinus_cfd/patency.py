"""Patency assurance: is there a flow path, and can each sinus drain?

Two questions this module answers numerically, because "the segmentation looks
right" is not an answer (docs/handoff.md records several cases that looked right
and were not):

1. **Flow path** -- does a connected route exist inside the CFD domain from each
   naris to the outlet, and how tight is it at its narrowest?
2. **Drainage** -- for each named sinus, where is its ostium, how wide is it, and
   does it reach the nasal passage?

Both are prerequisites for CLAUDE.md goals 2, 3 and 4. Neither is a Dice score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .auto_airway import _spacing_zyx, geodesic_distance_mm, widest_path_bottleneck_mm

# Snap radius when resolving a boundary-condition port onto a mask. Generous on
# purpose: the point is to MEASURE the gap, not to hide it.
PORT_SNAP_MM = 10.0
# An ostium narrower than this reads as obstructed. Balloon dilation targets
# 3-4 mm (CLAUDE.md goal 5), so this is the clinically interesting bar.
OSTIUM_PATENT_MM = 2.0
# Anatomical bounds on a real ostium (operator, 2026-08-25). Used BOTH to pick
# the calibre statistic and to reject a body that is not bounded at an ostium:
# a 15-18 mm "ostium" is not a narrow one, it means the watershed boundary did
# not land on an ostium at all.
OSTIUM_MIN_DIAMETER_MM = 0.2
OSTIUM_MAX_DIAMETER_MM = 6.0
# Anatomically there is ONE of these per side, so two bodies sharing a name and a
# side are one sinus split by a partly-resolved ostium (CQ500CT390 reported
# maxillary L twice: 3.01 mL disconnected + 1.36 mL draining). Ethmoid is
# deliberately absent -- it is a cluster of separate cells and must not be fused.
MERGE_SINGLETON_SINUSES = ("maxillary", "frontal", "sphenoid")
# Only fuse bodies whose centroids are this close; guards against welding two
# genuinely different structures that happened to get the same label.
MERGE_MAX_CENTROID_MM = 40.0
STRUCT26 = np.ones((3, 3, 3), dtype=bool)


def _snap(mask, zyx, spacing_xyz, radius_mm=PORT_SNAP_MM):
    """Nearest ``mask`` voxel to ``zyx`` within a physical ball; (None, dist) if none."""
    z, y, x = (int(zyx[0]), int(zyx[1]), int(zyx[2]))
    nz, ny, nx = mask.shape
    if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx and mask[z, y, x]:
        return (z, y, x), 0.0
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    rz = max(int(np.ceil(radius_mm / max(sz, 1e-6))), 1)
    ry = max(int(np.ceil(radius_mm / max(sy, 1e-6))), 1)
    rx = max(int(np.ceil(radius_mm / max(sx, 1e-6))), 1)
    z0, z1 = max(0, z - rz), min(nz, z + rz + 1)
    y0, y1 = max(0, y - ry), min(ny, y + ry + 1)
    x0, x1 = max(0, x - rx), min(nx, x + rx + 1)
    sub = mask[z0:z1, y0:y1, x0:x1]
    if not sub.any():
        return None, float("inf")
    zz, yy, xx = np.where(sub)
    d = np.sqrt(
        ((zz + z0 - z) * sz) ** 2 + ((yy + y0 - y) * sy) ** 2 + ((xx + x0 - x) * sx) ** 2
    )
    i = int(np.argmin(d))
    if float(d[i]) > radius_mm:
        return None, float(d[i])
    return (int(zz[i] + z0), int(yy[i] + y0), int(xx[i] + x0)), float(d[i])


def flow_path(passage, inlets_zyx, outlet_zyx, spacing_xyz):
    """Is there a route from each naris to the outlet inside ``passage``?

    Reports, per inlet: whether the port resolves onto the domain and how far it
    had to snap (a large snap means the CFD domain does not reach the boundary
    condition), whether it is connected to the outlet, the route length, and the
    tightest radius along that route.

    The tight-radius figure is a **proxy for minimum cross-sectional area**: the
    smallest wall-distance on the route, not a true perpendicular cross-section.
    Reported as a radius, never converted into an area that could be mistaken for
    a measured MCA.
    """
    passage = passage.astype(bool)
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    out: dict[str, Any] = {"inlets": {}, "ok": False, "notes": []}
    if not passage.any():
        out["notes"].append("empty passage")
        return out
    edt = ndi.distance_transform_edt(passage, sampling=(sz, sy, sx)).astype(np.float32)
    o_snap, o_dist = _snap(passage, outlet_zyx, spacing_xyz)
    out["outlet_snap_mm"] = round(o_dist, 2)
    if o_snap is None:
        out["notes"].append(
            f"outlet does not resolve onto the passage (nearest {o_dist:.1f} mm)"
        )
        return out
    d_out = geodesic_distance_mm(passage, o_snap, spacing_xyz)
    all_ok = True
    for name, zyx in inlets_zyx.items():
        rec: dict[str, Any] = {}
        snap, dist = _snap(passage, zyx, spacing_xyz)
        rec["snap_mm"] = round(dist, 2) if np.isfinite(dist) else None
        if snap is None:
            rec["connected"] = False
            rec["reason"] = f"port does not resolve onto the passage (nearest {dist:.1f} mm)"
            all_ok = False
            out["inlets"][name] = rec
            continue
        length = float(d_out[snap])
        rec["connected"] = bool(np.isfinite(length))
        rec["path_len_mm"] = round(length, 1) if np.isfinite(length) else None
        if not rec["connected"]:
            rec["reason"] = "no connected route to the outlet inside the passage"
            all_ok = False
        else:
            d_in = geodesic_distance_mm(passage, snap, spacing_xyz)
            corridor = passage & np.isfinite(d_in) & np.isfinite(d_out)
            corridor &= (d_in + d_out) <= (length + 2.0 * max(sz, sy, sx))
            rec["min_radius_mm"] = (
                round(float(edt[corridor].min()), 2) if corridor.any() else None
            )
        out["inlets"][name] = rec
    out["ok"] = bool(all_ok and out["inlets"])
    return out


def _frame(mask):
    zz, yy, xx = np.where(mask)
    return {
        "z0": int(zz.min()),
        "z1": int(zz.max()),
        "y0": int(yy.min()),
        "y1": int(yy.max()),
        "xmid": float(xx.mean()),
    }


def name_sinus_bodies(
    sinus,
    airway,
    spacing_xyz,
    y_anterior_is_low=True,
    superior_is_high_z=True,
    x_midline=None,
):
    """Label each sinus body maxillary / frontal / sphenoid / ethmoid.

    Heuristic anatomy in the airway's own frame, not a trained classifier:

    * maxillary -- lateral, inferior, mid anterior-posterior
    * frontal   -- superior AND anterior
    * sphenoid  -- posterior, near midline
    * ethmoid   -- paramedian, superior-ish (between the orbits)

    Ambiguous bodies are named ``unknown`` rather than forced into a class.
    """
    sinus = sinus.astype(bool)
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    vox_ml = (sz * sy * sx) / 1000.0
    fr = _frame(airway)
    xmid = float(x_midline) if x_midline is not None else fr["xmid"]
    dy = max(fr["y1"] - fr["y0"], 1)
    dz = max(fr["z1"] - fr["z0"], 1)
    lab, n = ndi.label(sinus, STRUCT26)
    out = []
    for i in range(1, n + 1):
        b = lab == i
        v = float(b.sum()) * vox_ml
        if v < 0.05:
            continue
        zz, yy, xx = np.where(b)
        off = (float(xx.mean()) - xmid) * sx
        fy = (float(yy.mean()) - fr["y0"]) / dy
        fz = (float(zz.mean()) - fr["z0"]) / dz
        if not y_anterior_is_low:
            fy = 1.0 - fy
        if not superior_is_high_z:
            fz = 1.0 - fz
        lateral = abs(off)
        if lateral > 12.0 and fz < 0.78 and 0.10 < fy < 0.80:
            name = "maxillary"
        elif fz > 0.72 and fy < 0.42:
            name = "frontal"
        elif fy > 0.62 and lateral <= 14.0:
            name = "sphenoid"
        elif 3.0 < lateral <= 20.0 and fz >= 0.42:
            name = "ethmoid"
        else:
            name = "unknown"
        side = "L" if off > 3.0 else ("R" if off < -3.0 else "midline")
        out.append(
            {
                "name": name,
                "side": side,
                "volume_ml": round(v, 2),
                "off_midline_mm": round(off, 1),
                "frac_posterior": round(fy, 2),
                "frac_superior": round(fz, 2),
                "_label": i,
            }
        )
    out.sort(key=lambda r: -r["volume_ml"])
    return out


def _merge_split_sinuses(recs, spacing_xyz, notes):
    """Fuse bodies that are one sinus split by a partly-resolved ostium.

    A maxillary antrum whose ostium is only half-resolved appears twice: the part
    the strip carved out of the airway (draining, with a calibre) and the part
    that stayed a separate air component (not draining). They are one sinus. The
    merged record keeps the BEST evidence -- if any part drains, the sinus drains,
    at the widest calibre measured.
    """
    if not recs:
        return recs
    out, used = [], set()
    for i, r in enumerate(recs):
        if i in used:
            continue
        if r["name"] not in MERGE_SINGLETON_SINUSES:
            out.append(r)
            continue
        group = [r]
        for j in range(i + 1, len(recs)):
            if j in used:
                continue
            o = recs[j]
            if o["name"] != r["name"] or o["side"] != r["side"]:
                continue
            if abs(o["off_midline_mm"] - r["off_midline_mm"]) > MERGE_MAX_CENTROID_MM:
                continue
            group.append(o)
            used.add(j)
        if len(group) == 1:
            out.append(r)
            continue
        vol = sum(g["volume_ml"] for g in group)
        drains = any(g["drains"] for g in group)
        best = max(group, key=lambda g: g["ostium_diameter_mm"])
        merged = dict(r)
        merged["volume_ml"] = round(vol, 2)
        merged["off_midline_mm"] = round(
            sum(g["off_midline_mm"] * g["volume_ml"] for g in group) / max(vol, 1e-9), 1
        )
        merged["drains"] = drains
        merged["ostium_radius_mm"] = best["ostium_radius_mm"]
        merged["ostium_diameter_mm"] = best["ostium_diameter_mm"]
        merged["patent"] = bool(best["ostium_diameter_mm"] >= OSTIUM_PATENT_MM) and drains
        merged["ostium_zyx"] = best.get("ostium_zyx")
        merged["merged_from"] = len(group)
        merged["connection"] = (
            best.get("connection", "") if drains
            else "no ostium resolved at this resolution"
        )
        parts = " + ".join(f"{g['volume_ml']:.2f}" for g in group)
        notes.append(
            f"merged {len(group)} {r['name']} {r['side']} bodies "
            f"({parts} mL) -- one sinus split by a partly-resolved ostium"
        )
        out.append(merged)
    return out


def drainage(
    airway,
    sinus,
    passage,
    spacing_xyz,
    y_anterior_is_low=True,
    superior_is_high_z=True,
    interior_air=None,
    min_leftover_ml=0.25,
):
    """Per-sinus ostium location and calibre, and whether it reaches the passage.

    Calibre is the **widest-path radius** from the passage into the sinus: the
    radius of the tightest point drainage must pass through. Diameter (2x) is
    what an antrostomy or a balloon changes, so both are reported. A sinus below
    ``OSTIUM_PATENT_MM`` diameter is flagged obstructed.
    """
    airway = airway.astype(bool)
    sinus = sinus.astype(bool) & airway
    passage = passage.astype(bool) & airway
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    vox_ml = (sz * sy * sx) / 1000.0
    res: dict[str, Any] = {"sinuses": [], "notes": []}
    if not sinus.any() and interior_air is None:
        res["notes"].append("no sinus bodies segmented")
        return res
    if not passage.any():
        res["notes"].append("empty passage; cannot measure drainage")
        return res
    edt = ndi.distance_transform_edt(airway, sampling=(sz, sy, sx)).astype(np.float32)
    bodies = name_sinus_bodies(
        sinus, airway, spacing_xyz, y_anterior_is_low, superior_is_high_z
    )
    lab, _ = ndi.label(sinus, STRUCT26)
    # Sinuses whose ostium is not resolved at this HU/resolution are SEPARATE air
    # components, never part of the nares->trachea path, so the dead-end strip
    # cannot see them. On CQ500CT390 both maxillary antra (3.0 / 2.8 mL) and a
    # sphenoid (0.4 mL) sit entirely outside airway_mask. Pick them up from the
    # leftover interior air and report them as found-but-not-drained -- NOT as
    # "obstructed", because an ostium below the threshold looks the same here.
    leftover_recs = []
    if interior_air is not None:
        left = interior_air.astype(bool) & ~airway
        if left.any():
            llab, ln = ndi.label(left, STRUCT26)
            lsz = ndi.sum(left, llab, range(1, ln + 1))
            big = np.zeros_like(left)
            for i in range(ln):
                if lsz[i] * vox_ml >= min_leftover_ml:
                    big |= llab == i + 1
            if big.any():
                for rec in name_sinus_bodies(
                    big, airway, spacing_xyz, y_anterior_is_low, superior_is_high_z
                ):
                    rec.pop("_label", None)
                    rec["ostium_radius_mm"] = 0.0
                    rec["ostium_diameter_mm"] = 0.0
                    rec["touches_passage"] = False
                    rec["drains"] = False
                    rec["patent"] = False
                    rec["ostium_zyx"] = None
                    rec["connection"] = "no ostium resolved at this resolution"
                    leftover_recs.append(rec)
    bott = widest_path_bottleneck_mm(airway, passage, edt, spacing_xyz)
    for rec in bodies:
        b = lab == rec.pop("_label")
        interface = ndi.binary_dilation(b, STRUCT26) & passage
        # Calibre is the MEDIAN half-width across the interface, not the max.
        # The interface is the whole watershed contact surface between sinus and
        # passage; its widest single voxel is an outlier sitting wherever the
        # lumen happens to be roomiest, which reported an 18.11 mm "ostium" on
        # THCA's sphenoid and 15.83 mm on CQ500CT390's left maxillary. Measured
        # against the 0.2-6 mm anatomical range, the max lands outside it on 7 of
        # 11 bodies and the median lands inside on all 11.
        if interface.any():
            iface_vals = edt[interface]
            calibre = float(np.median(iface_vals))
            iface_max = float(iface_vals.max())
        else:
            vals = bott[b]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            calibre = float(vals.max()) if vals.size else 0.0
            iface_max = calibre
        rec["interface_max_mm"] = round(2.0 * iface_max, 2)
        rec["ostium_radius_mm"] = round(calibre, 2)
        rec["ostium_diameter_mm"] = round(2.0 * calibre, 2)
        rec["touches_passage"] = bool(interface.any())
        rec["drains"] = bool(interface.any() and calibre > 0.0)
        diam = 2.0 * calibre
        rec["patent"] = bool(diam >= OSTIUM_PATENT_MM)
        # A connection outside the anatomical range is not an ostium: the sinus
        # body is not bounded at one. Say so rather than quoting the number.
        if interface.any() and not (OSTIUM_MIN_DIAMETER_MM <= diam <= OSTIUM_MAX_DIAMETER_MM):
            rec["ostium_valid"] = False
            rec["ostium_note"] = (
                f"{diam:.2f} mm is outside the anatomical "
                f"{OSTIUM_MIN_DIAMETER_MM}-{OSTIUM_MAX_DIAMETER_MM} mm range; "
                f"this body is not bounded at an ostium"
            )
            rec["patent"] = False
        else:
            rec["ostium_valid"] = True
        if interface.any():
            zz, yy, xx = np.where(interface)
            rec["ostium_zyx"] = [int(zz.mean()), int(yy.mean()), int(xx.mean())]
        else:
            rec["ostium_zyx"] = None
        rec["connection"] = "drains through a resolved ostium"
        res["sinuses"].append(rec)
    # Keep only anatomically named leftovers; unnamed blobs at this stage are
    # mastoid / orbital air, not sinuses.
    res["sinuses"].extend(r for r in leftover_recs if r["name"] != "unknown")
    res["sinuses"] = _merge_split_sinuses(res["sinuses"], spacing_xyz, res["notes"])
    res["sinuses"].sort(key=lambda r: -r["volume_ml"])
    have = {r["name"] for r in res["sinuses"] if r["name"] != "unknown"}
    for want in ("maxillary", "frontal", "sphenoid"):
        if want not in have:
            res["notes"].append(f"no {want} sinus found")
    return res
