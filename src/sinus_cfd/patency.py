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
# No single sinus straddles the midline: maxillary, frontal and ethmoid are
# paired, and the sphenoid is midline but small. A large body with substantial
# volume on BOTH sides is therefore several sinuses fused by 26-connectivity --
# on THCA both antra came out as one 43.2 mL "unknown" mass 74 mm wide.
SPLIT_MIN_VOLUME_ML = 5.0
SPLIT_MIN_SIDE_FRACTION = 0.25
SPLIT_ERODE_MAX_MM = 4.0
# A paranasal sinus drains INTO the nasal cavity, so it lies against it. Air this
# far from the passage is not a sinus. Measured across all 5 cases: every genuine
# body is 0.0-11.2 mm from the passage, while CQ500CT390 produced a 0.40 mL body
# 78.5 mm away -- air behind the head, caught by the per-slice convex hull in
# interior_air_within_hull, and then named "sphenoid" because the classifier only
# tested direction and never distance.
SINUS_MAX_DISTANCE_TO_PASSAGE_MM = 30.0
# Where on the sinus/passage interface the ostium actually sits. This is anatomy,
# not geometry, and it differs per sinus -- which is why a single rule kept
# failing. Operator marks on 4 ostia: all three MAXILLARY ostia were 18-25 mm
# SUPERIOR of a typical-width pick (dz mean +14.8 mm, |mean|/std 1.47, i.e.
# systematic), while dy and dx were -0.4 and -0.3 mm, pure scatter. The ethmoid
# was already a 2.0 mm hit.
#   maxillary - drains UP through the superomedial wall into the middle meatus
#   frontal   - drains DOWN the frontal recess into the middle meatus
#   sphenoid  - drains FORWARD into the sphenoethmoidal recess
#   ethmoid   - drains medially; small enough that any interface voxel is close
OSTIUM_DRAINAGE_DIR = {
    "maxillary": ("z", +1),
    "frontal": ("z", -1),
    "sphenoid": ("y", -1),
    "ethmoid": (None, 0),
}
# Fraction of the interface, taken from the drainage end, to search within.
OSTIUM_DIR_QUANTILE = 0.15
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


def _split_midline_straddlers(mask, spacing_xyz, x_midline):
    """Separate sinuses that 26-connectivity fused into one component.

    The strip's watershed already knew these as distinct basins, but that
    identity is lost when the basins are OR-ed into a boolean mask and written
    out. Rather than re-plumb the file format, recover the lobes here: erode
    until the body falls apart, then hand the pieces back their territory by
    watershed, which puts the boundary at the narrowest link between them.

    Returns a labelled array. Only bodies that straddle the midline are split,
    so a genuinely single sinus and a cluster of ethmoid cells are left alone.
    """
    lab, n = ndi.label(mask, STRUCT26)
    if n == 0:
        return lab, 0
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    vox_ml = (sz * sy * sx) / 1000.0
    out = np.zeros_like(lab)
    nxt = 0
    for i in range(1, n + 1):
        b = lab == i
        vol = float(b.sum()) * vox_ml
        xs = np.where(b)[2]
        left = float((xs > x_midline).mean())
        straddles = min(left, 1.0 - left) >= SPLIT_MIN_SIDE_FRACTION
        if vol < SPLIT_MIN_VOLUME_ML or not straddles:
            nxt += 1
            out[b] = nxt
            continue
        edt_b = ndi.distance_transform_edt(b, sampling=(sz, sy, sx)).astype(np.float32)
        markers = None
        step = max(min(sz, sy, sx), 0.25)
        r = step
        while r <= SPLIT_ERODE_MAX_MM:
            core = b & (edt_b >= r)
            cl, cn = ndi.label(core, STRUCT26)
            if cn >= 2:
                csz = ndi.sum(core, cl, range(1, cn + 1))
                keep = [k + 1 for k in range(cn) if csz[k] * vox_ml >= 0.3]
                if len(keep) >= 2:
                    markers = np.zeros_like(cl)
                    for j, k in enumerate(keep, start=1):
                        markers[cl == k] = j
                    break
            r += step
        if markers is None:
            nxt += 1
            out[b] = nxt
            continue
        try:
            from skimage.segmentation import watershed
        except Exception:
            nxt += 1
            out[b] = nxt
            continue
        ws = watershed(-edt_b, markers, mask=b)
        for j in range(1, int(markers.max()) + 1):
            piece = b & (ws == j)
            if not piece.any():
                continue
            nxt += 1
            out[piece] = nxt
    return out, nxt


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
    lab, n = _split_midline_straddlers(sinus, spacing_xyz, xmid)
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
        # Carry EVERY part's label, not just the first. dict(r) copies r's
        # body_id alone, so the merged record used to name a sinus while
        # pointing at a fraction of it -- and a tool path or a virtual
        # antrostomy built on that mask would operate on the wrong volume.
        merged["body_ids"] = [i for g in group for i in g.get("body_ids", [])]
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
    hu=None,
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
    # Every body gets a unique id in this map. Recovering a body by matching its
    # VOLUME is ambiguous -- two frontal cells of 0.42 and 0.36 mL both matched a
    # 0.40 mL component and silently shared one mask.
    body_labels = np.zeros(airway.shape, dtype=np.int32)
    next_id = 0
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
    # MUST use the same labelling name_sinus_bodies used, or a record's name and
    # its mask refer to different bodies -- and the mismatch is silent, because
    # the names still look right while the geometry behind them is wrong.
    x_mid = _frame(airway)["xmid"]
    lab, _ = _split_midline_straddlers(sinus, spacing_xyz, x_mid)
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
                # Drop anything too far from the passage to be a sinus.
                d_pas = ndi.distance_transform_edt(
                    ~passage, sampling=(sz, sy, sx)).astype(np.float32)
                llab, ln2 = ndi.label(big, STRUCT26)
                dropped = 0
                for i in range(1, ln2 + 1):
                    m = llab == i
                    if float(d_pas[m].min()) > SINUS_MAX_DISTANCE_TO_PASSAGE_MM:
                        big[m] = False
                        dropped += 1
                if dropped:
                    res["notes"].append(
                        f"dropped {dropped} leftover air body(ies) more than "
                        f"{SINUS_MAX_DISTANCE_TO_PASSAGE_MM:.0f} mm from the passage "
                        "-- too far to be a sinus")
            if big.any():
                lmap, _ln = _split_midline_straddlers(
                    big, spacing_xyz, _frame(airway)["xmid"])
                for rec in name_sinus_bodies(
                    big, airway, spacing_xyz, y_anterior_is_low, superior_is_high_z
                ):
                    _lid = rec.pop("_label", None)
                    if _lid is not None:
                        next_id += 1
                        body_labels[lmap == _lid] = next_id
                        rec["body_id"] = next_id
                        rec["body_ids"] = [next_id]
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
        next_id += 1
        body_labels[b] = next_id
        rec["body_id"] = next_id
        rec["body_ids"] = [next_id]
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
            # WHERE on the interface: the NARROWEST voxel. An ostium is by
            # definition the tightest constriction between sinus and passage,
            # so this is the physical definition rather than a tuned choice.
            #
            # Measured against operator marks on 3 maxillary ostia:
            #   narrowest voxel        1.5 / 2.5 / 4.0 mm   median 2.5   <-- this
            #   typical width (old)    4.0 / 2.9 / 4.8 mm   median 4.0
            #   best-connected         2.6 / 3.7 / 5.1 mm   median 3.7
            #   edt ~ ostium radius   12.7 /14.7 /18.1 mm   median 14.7
            #   centroid of narrowest 10%  11.9 /12.4 /15.2 mm
            #
            # That last one is the recurring trap: AVERAGING POSITIONS lands
            # inside the sinus, because narrow voxels are scattered around the
            # whole interface rim. Never take a centroid of a surface and call
            # it a location -- pick a real voxel.
            zz, yy, xx = np.where(interface)
            widths = edt[interface]
            j = int(np.argmin(widths))
            rec["ostium_zyx"] = [int(zz[j]), int(yy[j]), int(xx[j])]
            # Narrowest point, alongside the median calibre reported above.
            # They answer different questions: the median is the effective
            # opening (and matches the clinical 2-4 mm for a maxillary ostium),
            # the minimum is the tightest squeeze a probe would meet.
            rec["ostium_min_diameter_mm"] = round(2.0 * float(widths[j]), 2)
        else:
            rec["ostium_zyx"] = None
        rec["connection"] = "drains through a resolved ostium"
        res["sinuses"].append(rec)
    # Keep only anatomically named leftovers; unnamed blobs at this stage are
    # mastoid / orbital air, not sinuses.
    res["sinuses"].extend(r for r in leftover_recs if r["name"] != "unknown")
    res["sinuses"] = _merge_split_sinuses(res["sinuses"], spacing_xyz, res["notes"])
    res["sinuses"].sort(key=lambda r: -r["volume_ml"])
    res["body_labels"] = body_labels
    # For a named sinus with no resolved ostium, estimate where it WOULD drain.
    if hu is not None:
        for rec in res["sinuses"]:
            if rec.get("drains") or rec["name"] in ("unknown",):
                continue
            bid = rec.get("body_id")
            if not bid:
                continue
            est = probable_ostium(hu, body_labels == bid, passage, spacing_xyz)
            if est.get("found"):
                rec["probable_ostium"] = est
    have = {r["name"] for r in res["sinuses"] if r["name"] != "unknown"}
    for want in ("maxillary", "frontal", "sphenoid"):
        if want not in have:
            res["notes"].append(f"no {want} sinus found")
    return res


# Voxels whose geodesic distance to the two nares differs by less than this are
# fed by both -- the convergence zone. Not a hard anatomical boundary: it is the
# width of the band where the streams meet, and it widens where the airway is wide.
NARIS_MIXING_TOL_MM = 2.0


def naris_territory(passage, inlets_zyx, spacing_xyz, mixing_tol_mm=NARIS_MIXING_TOL_MM,
                    outlet_zyx=None):
    """Which naris feeds each voxel of the CFD domain.

    Competing geodesic flood inside ``passage`` from the two naris ports. Air
    reaching a voxel takes the shorter route, so the nearer naris feeds it; where
    the two distances are within ``mixing_tol_mm`` the streams converge and
    neither owns the voxel.

    Distance is geodesic IN THE LUMEN, not Euclidean -- a Euclidean split would
    put the boundary on a plane through the septum, which is the midplane error
    this project spent a cycle removing (docs/segmentation_strategy.md K3).

    Returns ``(label, meta)`` with label 0=outside, 1=left-fed, 2=right-fed,
    3=convergence.
    """
    passage = passage.astype(bool)
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    vox_ml = (sz * sy * sx) / 1000.0
    label = np.zeros(passage.shape, dtype=np.uint8)
    meta: dict[str, Any] = {"notes": []}
    if not passage.any() or len(inlets_zyx) < 2:
        meta["notes"].append("need a passage and two naris ports")
        return label, meta
    names = list(inlets_zyx)
    left_name = next((n for n in names if "left" in n.lower()), names[0])
    right_name = next((n for n in names if n != left_name), names[-1])
    snaps = {}
    for nm in (left_name, right_name):
        snap, dist = _snap(passage, inlets_zyx[nm], spacing_xyz)
        if snap is None:
            meta["notes"].append(f"{nm} does not resolve onto the passage")
            return label, meta
        snaps[nm] = (snap, dist)
    d_l = geodesic_distance_mm(passage, snaps[left_name][0], spacing_xyz)
    d_r = geodesic_distance_mm(passage, snaps[right_name][0], spacing_xyz)
    fl, fr = np.isfinite(d_l), np.isfinite(d_r)
    reach = passage & (fl | fr)
    # Reachable from ONE naris only -> that naris feeds it, whatever the other
    # distance is. Only compare where both are finite; inf - inf is not a
    # tie, it is "unreachable", and subtracting them would make it a silent NaN.
    label[passage & fl & ~fr] = 1
    label[passage & fr & ~fl] = 2
    both = passage & fl & fr
    if both.any():
        diff = np.zeros(passage.shape, dtype=np.float64)
        diff[both] = d_l[both] - d_r[both]
        label[both & (diff < -mixing_tol_mm)] = 1
        label[both & (diff > mixing_tol_mm)] = 2
        label[both & (np.abs(diff) <= mixing_tol_mm)] = 3
    # Once the two streams have met they stay met: everything DOWNSTREAM of the
    # convergence band is fed by both nares. Without this, a long shared channel
    # (THCA carries the whole pharynx and larynx) is handed entirely to whichever
    # naris happens to be geodesically nearer its entrance -- THCA came out
    # 57.9 / 11.9 mL, balance 0.21, which is an artefact of the tie-break, not
    # anatomy. Downstream = closer to the outlet than the far edge of the band.
    if outlet_zyx is not None and (label == 3).any():
        o_snap, _o_d = _snap(passage, outlet_zyx, spacing_xyz)
        if o_snap is not None:
            d_out = geodesic_distance_mm(passage, o_snap, spacing_xyz)
            band = d_out[(label == 3) & np.isfinite(d_out)]
            if band.size:
                # The tie band is not a single front: it runs the whole length of
                # the septum, where the two routes are equal all the way forward.
                # The streams actually MERGE at its posterior end -- the point
                # nearest the outlet -- so threshold on the MINIMUM. Using the
                # maximum swallowed the nasal cavities whole (VH: left-fed went
                # to 0.00 mL).
                thr = float(band.min())
                downstream = passage & np.isfinite(d_out) & (d_out <= thr) & (label != 0)
                meta["downstream_of_convergence_ml"] = round(
                    float((downstream & (label != 3)).sum()) * vox_ml, 2)
                label[downstream] = 3
    unreached = int(passage.sum()) - int(reach.sum())
    meta.update({
        "left_port": left_name, "right_port": right_name,
        "left_snap_mm": round(snaps[left_name][1], 2),
        "right_snap_mm": round(snaps[right_name][1], 2),
        "left_ml": round(float((label == 1).sum()) * vox_ml, 2),
        "right_ml": round(float((label == 2).sum()) * vox_ml, 2),
        "convergence_ml": round(float((label == 3).sum()) * vox_ml, 2),
        "unreached_ml": round(float(unreached) * vox_ml, 2),
        "mixing_tol_mm": mixing_tol_mm,
    })
    tot = meta["left_ml"] + meta["right_ml"]
    if tot > 0:
        meta["balance"] = round(min(meta["left_ml"], meta["right_ml"]) / max(meta["left_ml"], meta["right_ml"]), 3)
    if unreached:
        meta["notes"].append(
            f"{meta['unreached_ml']:.2f} mL of the passage is not reachable from "
            "either naris -- disconnected domain")
    return label, meta


# --- Probable ostium: a hypothesis, not a measurement -----------------------
# HU above this is treated as air and costs nothing to traverse.
OSTIUM_PATH_AIR_HU = -300.0
# Peak HU on the route, used to judge whether it is a plausible unresolved
# ostium (mucosa / thin bone, partial-volumed) or a solid wall.
OSTIUM_PATH_PLAUSIBLE_HU = 250.0    # below: consistent with an unresolved ostium
OSTIUM_PATH_BLOCKED_HU = 700.0      # above: cortical bone, not a drainage route
OSTIUM_PATH_SEARCH_MM = 30.0


def probable_ostium(hu, sinus_body, passage, spacing_xyz,
                    search_mm=OSTIUM_PATH_SEARCH_MM):
    """Where a sinus would most likely drain, when no ostium resolves as air.

    A frontal recess or sphenoethmoidal recess is often sub-millimetre and simply
    is not resolved at 0.4-0.6 mm, so the sinus appears as disconnected air. The
    CT still carries partial-volume evidence: along a real but unresolved channel
    the HU is depressed toward air, while a solid wall stays at cortical values.

    This walks the least-resistance route from the sinus to the passage, with
    per-voxel cost rising from 0 at air to high at bone, and reports where that
    route leaves the sinus.

    THIS IS A HYPOTHESIS, NOT A MEASUREMENT. It says "if this sinus drains, here
    is the most likely place and this is what stands in the way". A route through
    cortical bone means no pathway was found, not a narrow one.

    Returns a dict with the exit point, the route's peak HU (the barrier), the
    millimetres of non-air crossed, and a verdict.
    """
    import heapq

    sinus_body = sinus_body.astype(bool)
    passage = passage.astype(bool)
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    if not sinus_body.any() or not passage.any():
        return {"found": False, "reason": "empty sinus or passage"}
    # crop to the sinus plus a search margin, for speed
    zz, yy, xx = np.where(sinus_body)
    rz = max(int(round(search_mm / sz)), 1)
    ry = max(int(round(search_mm / sy)), 1)
    rx = max(int(round(search_mm / sx)), 1)
    z0, z1 = max(0, zz.min() - rz), min(hu.shape[0], zz.max() + rz + 1)
    y0, y1 = max(0, yy.min() - ry), min(hu.shape[1], yy.max() + ry + 1)
    x0, x1 = max(0, xx.min() - rx), min(hu.shape[2], xx.max() + rx + 1)
    H = hu[z0:z1, y0:y1, x0:x1].astype(np.float32)
    S = sinus_body[z0:z1, y0:y1, x0:x1]
    P = passage[z0:z1, y0:y1, x0:x1]
    if not P.any():
        return {"found": False, "reason": "no passage within the search window"}
    # cost: 0 in air, rising through soft tissue, steep in bone
    cost = np.clip((H - OSTIUM_PATH_AIR_HU) / 300.0, 0.0, None).astype(np.float64)
    nz_, ny_, nx_ = H.shape
    INF = np.inf
    dist = np.full(H.shape, INF)
    prev = np.full(H.shape + (3,), -1, dtype=np.int32)
    heap = []
    for z, y, x in zip(*np.where(S)):
        dist[z, y, x] = 0.0
        heap.append((0.0, int(z), int(y), int(x)))
    heapq.heapify(heap)
    offs = [(dz, dy, dx, float(np.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2)))
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)]
    target = None
    while heap:
        d, z, y, x = heapq.heappop(heap)
        if d > dist[z, y, x]:
            continue
        if P[z, y, x]:
            target = (z, y, x)
            break
        for dz, dy, dx, L in offs:
            z2, y2, x2 = z + dz, y + dy, x + dx
            if not (0 <= z2 < nz_ and 0 <= y2 < ny_ and 0 <= x2 < nx_):
                continue
            nd = d + L * (1.0 + 0.5 * (cost[z, y, x] + cost[z2, y2, x2]))
            if nd < dist[z2, y2, x2]:
                dist[z2, y2, x2] = nd
                prev[z2, y2, x2] = (z, y, x)
                heapq.heappush(heap, (nd, z2, y2, x2))
    if target is None:
        return {"found": False, "reason": "no route to the passage within the window"}
    path = [target]
    while True:
        pz, py, px = prev[path[-1]]
        if pz < 0:
            break
        path.append((int(pz), int(py), int(px)))
    path.reverse()
    hus = np.array([H[p] for p in path], dtype=float)
    lens = [0.0] + [float(np.sqrt(((path[i][0] - path[i - 1][0]) * sz) ** 2
                                  + ((path[i][1] - path[i - 1][1]) * sy) ** 2
                                  + ((path[i][2] - path[i - 1][2]) * sx) ** 2))
                    for i in range(1, len(path))]
    non_air_mm = float(sum(l for l, h in zip(lens, hus) if h > OSTIUM_PATH_AIR_HU))
    peak = float(hus.max())
    # the exit point is the last voxel still inside the sinus
    exit_idx = 0
    for i, p in enumerate(path):
        if S[p]:
            exit_idx = i
    ez, ey, ex = path[exit_idx]
    verdict = ("consistent with an unresolved ostium" if peak <= OSTIUM_PATH_PLAUSIBLE_HU
               else "blocked by bone -- no drainage route found"
               if peak >= OSTIUM_PATH_BLOCKED_HU
               else "uncertain -- thin bone or partial volume")
    return {
        "found": True,
        "exit_zyx": [int(ez + z0), int(ey + y0), int(ex + x0)],
        "entry_zyx": [int(path[-1][0] + z0), int(path[-1][1] + y0), int(path[-1][2] + x0)],
        "route_mm": round(float(sum(lens)), 2),
        "non_air_mm": round(non_air_mm, 2),
        "peak_hu": round(peak, 0),
        "verdict": verdict,
    }
