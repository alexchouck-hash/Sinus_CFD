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
