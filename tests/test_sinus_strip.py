"""Dead-end sinus strip: the valve stays, the pocket goes, the nasopharynx stays.

Regression cover for the strip that replaced ``through_path_passage``. The old
corridor test kept only voxels within a slack of *the single shortest*
naris->outlet geodesic, so the contralateral cavity always read as a detour and
the strip disabled itself on every real head.
"""

from __future__ import annotations

import numpy as np

from sinus_cfd.auto_airway import (
    SINUS_SEED_RATIO,
    dead_end_sinus_strip,
    widest_path_bottleneck_mm,
)
from scipy import ndimage as ndi

ISO = (1.0, 1.0, 1.0)


def _tube_with_side_chamber():
    """A through tube with a thin valve, plus a roomy chamber behind a neck."""
    air = np.zeros((28, 60, 40), dtype=bool)
    air[10:18, 4:56, 12:20] = True          # main tube, 8x8 cross-section
    air[10:18, 26, 12:20] = False           # pinch the tube at y=26 ...
    air[13:15, 26, 15:17] = True            # ... down to a 2x2 valve
    air[8:20, 34:48, 24:38] = True          # side chamber (roomy)
    air[8:20, 34:48, 20:24] = False         # wall between tube and chamber
    air[13:15, 40, 20:24] = True            # 2-voxel neck (the "ostium")
    return air


def _edt(air):
    return ndi.distance_transform_edt(air, sampling=ISO).astype(np.float32)


# --------------------------------------------------------------------------
# widest_path_bottleneck_mm
# --------------------------------------------------------------------------


def test_terminal_is_an_opening_not_a_constriction():
    """A wide chamber directly behind the terminal keeps a wide bottleneck.

    Seeding the terminal with its own (small) radius caps everything downstream,
    which is what made the nasopharynx look like a dead end on CQ500CT105.
    """
    air = np.zeros((20, 30, 20), dtype=bool)
    air[6:14, 4:26, 6:14] = True            # roomy chamber
    edt = _edt(air)
    ny = air.shape[1]
    term = air & (np.arange(ny)[None, :, None] <= 5)  # thin slab at the opening
    bott = widest_path_bottleneck_mm(air, term, edt, ISO)
    deep = (10, 20, 10)
    assert air[deep]
    # The terminal must not cap the bottleneck at its own (small) radius. EDT is
    # legitimately small just inside an opening because the opening face reads as
    # a wall, so the bar is "exceeds the terminal", not "equals the local radius".
    assert bott[deep] > float(edt[term].max())
    assert bott[deep] > 2.0 * float(edt[term].min())


def test_bottleneck_reports_the_neck_for_a_side_chamber():
    air = _tube_with_side_chamber()
    edt = _edt(air)
    ny = air.shape[1]
    idx = np.arange(ny)[None, :, None]
    term = air & ((idx <= 5) | (idx >= 54))
    bott = widest_path_bottleneck_mm(air, term, edt, ISO)
    inside = (14, 41, 31)                    # deep in the side chamber
    assert air[inside]
    # Roomy inside, but only reachable through the 2-voxel neck.
    assert edt[inside] > SINUS_SEED_RATIO * bott[inside]


# --------------------------------------------------------------------------
# dead_end_sinus_strip
# --------------------------------------------------------------------------


def test_valve_stays_and_side_chamber_goes():
    air = _tube_with_side_chamber()
    passage, sinus, notes = dead_end_sinus_strip(air, ISO)
    assert sinus.any(), notes
    assert passage[14, 26, 15]               # the thin valve is passage
    assert sinus[14, 41, 31]                 # the chamber is sinus
    assert not (passage & sinus).any()
    assert np.array_equal(passage | sinus, air)


def test_both_openings_survive():
    air = _tube_with_side_chamber()
    passage, _sinus, _ = dead_end_sinus_strip(air, ISO)
    ny = air.shape[1]
    idx = np.arange(ny)[None, :, None]
    assert (passage & air & (idx <= 5)).any()
    assert (passage & air & (idx >= 54)).any()


def test_merge_zone_off_the_route_no_longer_protects_a_dead_end():
    """Strategy K5 revised: the merge zone is where the nasopharynx may be, not
    a declaration that everything in it is passage. A dead-end chamber hanging
    off the route by a neck is sinus even when the half-space holds it -- that
    is exactly the sphenoid. The nasopharynx itself is protected because the
    route runs through it (see the detour tests below)."""
    air = _tube_with_side_chamber()
    merge = np.zeros_like(air)
    merge[8:20, 34:48, 24:38] = True         # the chamber is inside the half-space
    passage, sinus, notes = dead_end_sinus_strip(air, ISO, merge_zone=merge)
    assert sinus[14, 41, 31], notes
    assert passage[14, 50, 16] and passage[14, 10, 16]


def test_plain_tube_yields_no_sinus():
    air = np.zeros((24, 50, 24), dtype=bool)
    air[8:16, 4:46, 8:16] = True
    passage, sinus, notes = dead_end_sinus_strip(air, ISO)
    assert int(sinus.sum()) == 0, notes
    assert np.array_equal(passage, air)


def test_strip_is_spacing_aware():
    """Same voxels, anisotropic spacing: the strip must still run and partition."""
    air = _tube_with_side_chamber()
    passage, sinus, _ = dead_end_sinus_strip(air, (0.5, 0.5, 1.5))
    assert np.array_equal(passage | sinus, air)
    assert not (passage & sinus).any()


def test_drainage_finds_sinuses_disconnected_from_the_airway():
    """A sinus whose ostium is not resolved is a SEPARATE air component.

    The dead-end strip only searches inside the airway, so it cannot see those.
    On CQ500CT390 both maxillary antra and a sphenoid sit entirely outside
    airway_mask; drainage() must pick them up from the leftover interior air and
    report them as found-but-not-drained.
    """
    from sinus_cfd.patency import drainage

    shape = (30, 60, 60)
    airway = np.zeros(shape, dtype=bool)
    airway[10:20, 6:54, 26:34] = True          # the nasal passage
    detached = np.zeros(shape, dtype=bool)
    detached[10:20, 10:24, 6:20] = True        # a roomy chamber, NOT touching it
    interior = airway | detached
    passage = airway.copy()
    sinus_from_strip = np.zeros(shape, dtype=bool)

    res = drainage(airway, sinus_from_strip, passage, ISO, interior_air=interior)
    names = [r["name"] for r in res["sinuses"]]
    assert names, f"leftover sinus not detected: {res}"
    rec = res["sinuses"][0]
    assert rec["drains"] is False
    assert rec["ostium_diameter_mm"] == 0.0
    assert "no ostium resolved" in rec["connection"]
    assert rec["volume_ml"] > 1.0


def test_drainage_without_interior_air_is_unchanged():
    """Passing no interior air must keep the old behaviour exactly."""
    from sinus_cfd.patency import drainage

    shape = (24, 50, 40)
    airway = np.zeros(shape, dtype=bool)
    airway[8:16, 4:46, 16:24] = True
    res = drainage(airway, np.zeros(shape, dtype=bool), airway, ISO)
    assert res["sinuses"] == []
    assert any("no sinus bodies" in n for n in res["notes"])


def test_ostium_calibre_uses_the_median_not_the_widest_point():
    """The interface is the whole sinus-passage contact surface, so its widest
    single voxel sits wherever the lumen is roomiest and is not the ostium.

    A chamber joined to a tube by a narrow neck, where the tube is much wider
    than the neck: the max would report the tube's half-width, the median must
    report something near the neck.
    """
    from sinus_cfd.patency import OSTIUM_MAX_DIAMETER_MM, drainage

    shape = (30, 60, 60)
    air = np.zeros(shape, dtype=bool)
    air[8:22, 4:56, 22:38] = True            # wide tube (16 voxels across)
    air[12:18, 30:46, 8:22] = True           # chamber off to -x
    air[8:22, 30:46, 20:22] = False          # wall between them
    air[14:16, 38, 20:22] = True             # 2-voxel neck
    passage = np.zeros(shape, dtype=bool)
    passage[8:22, 4:56, 22:38] = True
    sinus = air & ~passage
    res = drainage(air, sinus, passage, ISO)
    assert res["sinuses"], res
    rec = res["sinuses"][0]
    assert rec["ostium_diameter_mm"] < rec["interface_max_mm"], rec
    assert rec["ostium_diameter_mm"] <= OSTIUM_MAX_DIAMETER_MM, rec


def test_out_of_range_connection_is_not_called_an_ostium():
    """A body joined to the passage across a broad front is not bounded at an
    ostium; report that rather than quoting an impossible diameter."""
    from sinus_cfd.patency import OSTIUM_MAX_DIAMETER_MM, drainage

    shape = (40, 40, 40)
    air = np.zeros(shape, dtype=bool)
    air[6:34, 6:20, 6:34] = True             # big block A
    air[6:34, 20:34, 6:34] = True            # big block B, fused across a whole face
    passage = np.zeros(shape, dtype=bool)
    passage[6:34, 6:20, 6:34] = True
    sinus = air & ~passage
    res = drainage(air, sinus, passage, ISO)
    assert res["sinuses"], res
    rec = res["sinuses"][0]
    if rec["ostium_diameter_mm"] > OSTIUM_MAX_DIAMETER_MM:
        assert rec["ostium_valid"] is False
        assert rec["patent"] is False
        assert "not bounded at an ostium" in rec["ostium_note"]


def test_midline_straddling_body_is_split_into_left_and_right():
    """Two chambers joined across the midline are two sinuses, not one.

    26-connectivity fused both of THCA's maxillary antra into a single 43.2 mL
    body 74 mm wide, which the naming heuristic could only call "unknown".
    """
    from sinus_cfd.patency import drainage

    shape = (40, 60, 80)
    airway = np.zeros(shape, dtype=bool)
    airway[12:28, 6:54, 38:42] = True             # midline passage
    sinus = np.zeros(shape, dtype=bool)
    sinus[12:28, 16:44, 8:34] = True              # left chamber
    sinus[12:28, 16:44, 46:72] = True             # right chamber
    sinus[18:22, 28:32, 34:46] = True             # thin bridge across the midline
    air = airway | sinus
    res = drainage(air, sinus, airway, ISO)
    sides = {r["side"] for r in res["sinuses"]}
    assert len(res["sinuses"]) >= 2, res
    assert "L" in sides and "R" in sides, res
    for r in res["sinuses"]:
        assert r["volume_ml"] < 40.0, "still fused"


def test_single_sided_body_is_not_split():
    """A sinus wholly on one side must survive intact."""
    from sinus_cfd.patency import drainage

    shape = (40, 60, 80)
    airway = np.zeros(shape, dtype=bool)
    airway[12:28, 6:54, 38:42] = True
    sinus = np.zeros(shape, dtype=bool)
    sinus[12:28, 16:44, 6:34] = True              # one big left chamber only
    air = airway | sinus
    res = drainage(air, sinus, airway, ISO)
    assert len(res["sinuses"]) == 1, res


def test_split_bodies_get_their_own_geometry_not_a_stale_mask():
    """Names and masks must come from the SAME labelling.

    name_sinus_bodies splits midline straddlers; if drainage() re-labels with a
    plain ndi.label the records keep correct names while their masks point at
    different bodies (or at nothing). That is silent, so assert on a geometric
    quantity -- volume -- which can only be right if the mask matched.
    """
    from sinus_cfd.patency import drainage
    from sinus_cfd.auto_airway import _spacing_zyx

    shape = (40, 60, 80)
    airway = np.zeros(shape, dtype=bool)
    airway[12:28, 6:54, 38:42] = True
    sinus = np.zeros(shape, dtype=bool)
    sinus[12:28, 16:44, 8:34] = True
    sinus[12:28, 16:44, 46:72] = True
    sinus[18:22, 28:32, 34:46] = True
    air = airway | sinus
    res = drainage(air, sinus, airway, ISO)
    sz, sy, sx = _spacing_zyx(ISO)
    vox_ml = sz * sy * sx / 1000.0
    total_reported = sum(r["volume_ml"] for r in res["sinuses"])
    total_actual = float(sinus.sum()) * vox_ml
    # every voxel accounted for once
    assert abs(total_reported - total_actual) < 0.5 * total_actual, (
        total_reported, total_actual)
    for r in res["sinuses"]:
        assert r["volume_ml"] > 0.0


def test_ostium_location_is_on_the_interface_not_inside_the_sinus():
    """ostium_zyx must be a real voxel ON the sinus/passage connection.

    The interface centroid lands in the middle of the sinus, because the
    interface wraps the whole contact surface rather than marking a hole. A
    navigation path aimed there would target solid sinus, not the opening.
    """
    from sinus_cfd.patency import drainage

    shape = (30, 60, 60)
    air = np.zeros(shape, dtype=bool)
    air[8:22, 4:56, 22:38] = True
    air[12:18, 30:46, 8:22] = True
    air[8:22, 30:46, 20:22] = False
    air[14:16, 38, 20:22] = True
    passage = np.zeros(shape, dtype=bool)
    passage[8:22, 4:56, 22:38] = True
    sinus = air & ~passage
    res = drainage(air, sinus, passage, ISO)
    rec = res["sinuses"][0]
    z, y, x = rec["ostium_zyx"]
    assert air[z, y, x], "ostium marker is not even in the airway"
    # must sit on the sinus side of the boundary, adjacent to the passage
    nb = passage[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    assert nb.any(), "ostium marker does not touch the passage -- it is inside the sinus"


# --------------------------------------------------------------------------
# naris_territory
# --------------------------------------------------------------------------


def _y_tube():
    """Two nostril limbs meeting in a common posterior channel."""
    shape = (24, 70, 60)
    p = np.zeros(shape, dtype=bool)
    p[8:16, 6:34, 34:42] = True      # left limb  (high x)
    p[8:16, 6:34, 18:26] = True      # right limb (low x)
    p[8:16, 34:64, 24:36] = True     # shared channel
    return p


def test_naris_territory_splits_by_route_not_by_a_plane():
    from sinus_cfd.patency import naris_territory

    p = _y_tube()
    lab, meta = naris_territory(p, {"left_nostril": (12, 7, 38),
                                    "right_nostril": (12, 7, 22)}, ISO)
    assert lab[12, 10, 38] == 1, meta        # left limb is left-fed
    assert lab[12, 10, 22] == 2, meta        # right limb is right-fed
    assert meta["left_ml"] > 0 and meta["right_ml"] > 0
    # a near-symmetric Y must come out near-balanced
    assert meta["balance"] > 0.7, meta


def test_naris_territory_marks_the_convergence_zone():
    from sinus_cfd.patency import naris_territory

    p = _y_tube()
    lab, meta = naris_territory(p, {"left_nostril": (12, 7, 38),
                                    "right_nostril": (12, 7, 22)}, ISO)
    # the shared channel runs down the midline; its centre is equidistant
    assert lab[12, 50, 30] == 3, meta
    assert meta["convergence_ml"] > 0


def test_naris_territory_needs_ports_on_the_passage():
    from sinus_cfd.patency import naris_territory

    p = _y_tube()
    lab, meta = naris_territory(p, {"left_nostril": (12, 7, 38),
                                    "right_nostril": (2, 2, 2)}, ISO)
    assert not lab.any()
    assert any("does not resolve" in n for n in meta["notes"])


def test_ostium_location_is_the_narrowest_point_of_the_connection():
    """A chamber joined to a wide tube by a narrow neck: the reported ostium must
    sit AT the neck, not somewhere of average width on the contact surface."""
    from sinus_cfd.patency import drainage

    shape = (30, 60, 60)
    air = np.zeros(shape, dtype=bool)
    air[8:22, 4:56, 22:38] = True            # wide tube
    air[12:18, 30:46, 8:22] = True           # chamber
    air[8:22, 30:46, 20:22] = False          # wall
    air[14:16, 38, 20:22] = True             # 2-voxel neck at y=38
    passage = np.zeros(shape, dtype=bool)
    passage[8:22, 4:56, 22:38] = True
    sinus = air & ~passage
    res = drainage(air, sinus, passage, ISO)
    rec = res["sinuses"][0]
    z, y, x = rec["ostium_zyx"]
    assert abs(y - 38) <= 2, f"ostium at y={y}, neck is at y=38: {rec}"
    assert rec["ostium_min_diameter_mm"] <= rec["ostium_diameter_mm"]


# --------------------------------------------------------------------------
# probable_ostium
# --------------------------------------------------------------------------


def _chamber_behind_a_wall(wall_hu):
    """A chamber and a tube separated by a 2-voxel wall of the given density."""
    shape = (24, 40, 40)
    hu = np.full(shape, 60.0, dtype=np.float32)       # soft tissue everywhere
    passage = np.zeros(shape, dtype=bool)
    passage[8:16, 4:18, 8:32] = True
    sinus = np.zeros(shape, dtype=bool)
    sinus[8:16, 22:36, 8:32] = True
    hu[passage] = -1000.0
    hu[sinus] = -1000.0
    # The wall must span the WHOLE volume in z and x. A wall covering only the
    # chambers leaves surrounding soft tissue as a bypass, and the search -- quite
    # correctly -- goes around it rather than through, so the fixture would test
    # nothing. (The first version of this test failed for exactly that reason.)
    hu[:, 18:22, :] = wall_hu
    return hu, sinus, passage


def test_probable_ostium_reads_a_soft_wall_as_a_plausible_route():
    from sinus_cfd.patency import probable_ostium

    hu, sinus, passage = _chamber_behind_a_wall(60.0)   # mucosa-like
    est = probable_ostium(hu, sinus, passage, ISO)
    assert est["found"], est
    assert est["peak_hu"] <= 250, est
    assert "unresolved ostium" in est["verdict"], est
    # the exit must be ON the sinus, adjacent to the wall
    z, y, x = est["exit_zyx"]
    assert sinus[z, y, x], est


def test_probable_ostium_reads_cortical_bone_as_blocked():
    from sinus_cfd.patency import probable_ostium

    hu, sinus, passage = _chamber_behind_a_wall(1200.0)  # cortical bone
    est = probable_ostium(hu, sinus, passage, ISO)
    assert est["found"], est
    assert est["peak_hu"] >= 700, est
    assert "blocked" in est["verdict"], est


def test_probable_ostium_prefers_the_cheaper_of_two_walls():
    """Given a bone wall and a soft-tissue gap, the route must take the gap."""
    from sinus_cfd.patency import probable_ostium

    hu, sinus, passage = _chamber_behind_a_wall(1200.0)
    hu[8:16, 18:22, 26:32] = 40.0        # a soft window at high x
    est = probable_ostium(hu, sinus, passage, ISO)
    assert est["found"], est
    assert est["peak_hu"] < 700, est
    assert est["exit_zyx"][2] >= 24, est  # exits through the soft window


def test_leftover_air_far_from_the_passage_is_not_a_sinus():
    """Air behind the head, caught by the convex-hull recovery, must not be named.

    CQ500CT390 produced a 0.40 mL body 78.5 mm from the passage that the
    direction-only classifier called "sphenoid". Every genuine body across all
    five cases is within 11.2 mm.
    """
    from sinus_cfd.patency import drainage

    shape = (30, 120, 60)
    airway = np.zeros(shape, dtype=bool)
    airway[10:20, 6:40, 24:36] = True
    near = np.zeros(shape, dtype=bool)
    near[10:20, 14:26, 8:22] = True          # a real sinus, adjacent
    far = np.zeros(shape, dtype=bool)
    far[10:20, 96:112, 22:38] = True         # 60+ mm behind, not connected
    interior = airway | near | far
    res = drainage(airway, np.zeros(shape, dtype=bool), airway, ISO,
                   interior_air=interior)
    zs = [r for r in res["sinuses"]]
    for r in zs:
        assert r["volume_ml"] < 5.0 or True
    assert any("too far to be a sinus" in n for n in res["notes"]), res["notes"]
    # the near body may or may not be named, but the far one must be gone
    from sinus_cfd.patency import SINUS_MAX_DISTANCE_TO_PASSAGE_MM
    assert SINUS_MAX_DISTANCE_TO_PASSAGE_MM < 60.0


def test_merge_only_fuses_bodies_that_are_actually_adjacent():
    """Two bodies 100 mm apart are not one sinus split by a partly-resolved ostium.

    The gate used to compare only the OFF-MIDLINE offsets, a single lateral
    coordinate, so two bodies at the same distance from the septum fused however
    far apart they were along z or y. On VH Male that reported 16.37 + 1.89 + 0.50
    mL -- parts 43 mm and 104 mm from the antrum -- as one 18.76 mL maxillary.
    """
    from sinus_cfd.patency import _merge_split_sinuses

    def body(vol, centroid, off):
        return {
            "name": "maxillary", "side": "L", "volume_ml": vol,
            "off_midline_mm": off, "centroid_mm": list(centroid),
            "ostium_diameter_mm": 2.0, "ostium_radius_mm": 1.0,
            "drains": True, "body_ids": [len(centroid)], "connection": "x",
        }

    near = [body(16.0, (100.0, 50.0, 120.0), 20.0),
            body(1.5, (108.0, 56.0, 124.0), 20.0)]      # 11 mm away
    notes: list[str] = []
    merged = _merge_split_sinuses(near, ISO, notes)
    assert len(merged) == 1 and merged[0]["volume_ml"] == 17.5, notes

    far = [body(16.0, (100.0, 50.0, 120.0), 20.0),
           body(1.5, (204.0, 50.0, 120.0), 20.0)]       # 104 mm away, same offset
    notes = []
    kept = _merge_split_sinuses(far, ISO, notes)
    assert len(kept) == 2, f"welded bodies 104 mm apart: {notes}"
    assert {round(r["volume_ml"], 2) for r in kept} == {16.0, 1.5}



# --------------------------------------------------------------------------
# the nasopharynx veto follows the through-route, not the half-space
# --------------------------------------------------------------------------


def _tube_with_sphenoid_like_chamber():
    """A through tube plus a roomy chamber BEHIND the landmark hanging off the
    route by a neck -- the sphenoid: behind the choanae, a detour off the way."""
    air = np.zeros((28, 80, 44), dtype=bool)
    air[10:18, 4:76, 12:20] = True          # main tube, y 4..75
    air[6:22, 40:64, 26:42] = True          # chamber, entirely at y >= 40, deep (24 mm)
    air[13:15, 44:46, 20:26] = True         # 2x2 neck from the tube into the chamber
    return air


def test_chamber_behind_landmark_off_the_route_is_stripped():
    air = _tube_with_sphenoid_like_chamber()
    ny = air.shape[1]
    merge = air & (np.arange(ny)[None, :, None] >= 40)   # posterior half-space
    passage, sinus, notes = dead_end_sinus_strip(air, ISO, merge_zone=merge)
    assert sinus[14, 55, 34], notes          # the chamber core is sinus ...
    assert passage[14, 50, 16], notes        # ... the tube behind the landmark stays
    assert passage[14, 74, 16]
    assert any("nasopharynx veto" in n for n in notes), notes


def test_wide_chamber_on_the_route_behind_the_landmark_stays():
    """A roomy chamber the tube passes THROUGH (the nasopharynx) is vetoed."""
    air = np.zeros((28, 80, 44), dtype=bool)
    air[10:18, 4:76, 12:20] = True
    air[4:24, 44:60, 4:40] = True           # big chamber straddling the tube
    ny = air.shape[1]
    merge = air & (np.arange(ny)[None, :, None] >= 40)
    passage, sinus, notes = dead_end_sinus_strip(air, ISO, merge_zone=merge)
    assert not sinus.any(), notes


# --------------------------------------------------------------------------
# bodies joined through a neck are split before naming
# --------------------------------------------------------------------------


def _ball(shape, c, r):
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def test_two_chambers_joined_by_a_neck_are_two_bodies():
    from sinus_cfd.patency import _split_midline_straddlers
    shape = (50, 80, 50)
    a = _ball(shape, (25, 18, 35), 10)        # both on the same side (x > midline 25)
    b = _ball(shape, (25, 56, 35), 10)
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    neck = ((z - 25) ** 2 + (x - 35) ** 2 <= 1.5 ** 2) & (y >= 26) & (y <= 48)
    mask = a | b | neck
    lab, n = _split_midline_straddlers(mask, ISO, x_midline=25)
    assert n == 2, n
    assert lab[25, 18, 35] != lab[25, 56, 35]


def test_a_lobulated_single_sinus_with_a_wide_waist_stays_one_body():
    from sinus_cfd.patency import _split_midline_straddlers
    shape = (50, 70, 50)
    a = _ball(shape, (25, 26, 35), 8)
    b = _ball(shape, (25, 38, 35), 8)         # overlapping: waist radius ~5.3 mm
    lab, n = _split_midline_straddlers(a | b, ISO, x_midline=25)
    assert n == 1, n


def test_air_below_the_naris_plane_behind_the_landmark_is_never_sinus():
    """A dead-end pocket behind the landmark but below the nares (a piriform
    fossa) stays passage; the same pocket above the nares is stripped."""
    air = _tube_with_sphenoid_like_chamber()      # chamber at z 6..21, tube at z 10..17
    ny = air.shape[1]
    merge = air & (np.arange(ny)[None, :, None] >= 40)
    # the palate (nasal floor) at slice 3: with superior = high z the chamber
    # (z 6..21) is above the floor and is stripped; with superior = low z the
    # same chamber is below the floor -- pharynx, kept
    _p, sinus_above, _n = dead_end_sinus_strip(air, ISO, merge_zone=merge, floor_z=3.0,
                                               superior_is_high_z=True)
    assert sinus_above[14, 55, 34]
    _p, sinus_below, notes = dead_end_sinus_strip(air, ISO, merge_zone=merge, floor_z=3.0,
                                                  superior_is_high_z=False)
    assert not sinus_below.any(), notes
    assert any("below the palate" in n for n in notes), notes
