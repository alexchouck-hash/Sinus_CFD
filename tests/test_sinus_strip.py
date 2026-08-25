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


def test_merge_zone_is_never_sinus():
    """A wide chamber inside the merge zone stays passage (strategy K5)."""
    air = _tube_with_side_chamber()
    merge = np.zeros_like(air)
    merge[8:20, 34:48, 24:38] = True         # declare the chamber nasopharynx
    passage, sinus, notes = dead_end_sinus_strip(air, ISO, merge_zone=merge)
    assert not (sinus & merge).any(), notes
    assert passage[14, 41, 31]


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
