"""A centreline is not a path: the tool has a diameter, a bend and a length.

Synthetic airway: a straight tube from a "naris" opening at low y to a target
deep inside. A 2 mm seeker fits a 6 mm tube and does not fit a 1.5 mm tube;
a fixed-bend tool whose bend cannot be accommodated in a straight tube is
reported as touching the wall, not as fitting because its centreline exists.
"""

from __future__ import annotations

import numpy as np

from sinus_cfd.instrument_fit import (
    Instrument,
    airway_wall_distance_mm,
    fit_instruments_to_ostia,
    place_instrument,
    tool_samples_mm,
)

ISO = (1.0, 1.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)


def _tube(radius_vox: int, length: int = 60, size: int = 40):
    """Straight tube along y, centred at (z, x) = (size/2, size/2)."""
    z, y, x = np.mgrid[0:size, 0:length, 0:size]
    c = size // 2
    air = ((z - c) ** 2 + (x - c) ** 2) <= radius_vox ** 2
    air &= (y >= 2) & (y < length - 2)
    inlet = np.zeros_like(air)
    inlet[:, 2:5, :] = air[:, 2:5, :]        # the naris: first 3 voxels of tube
    return air, inlet


STRAIGHT_SEEKER = Instrument("straight probe", "maxillary", 2.0, (("straight", 60.0),))
FAT_PROBE = Instrument("fat probe", "maxillary", 4.0, (("straight", 60.0),))
ET_SEEKER = Instrument(
    "ET seeker", "eustachian", 2.0,
    (("straight", 18.5), ("arc", 3.0, 45.0), ("straight", 80.0)),
)


def test_tool_samples_have_the_stated_length_and_start_at_the_tip():
    tip = np.array([10.0, 20.0, 30.0])
    pts, arcs = tool_samples_mm(ET_SEEKER, tip, np.array([0.0, 1.0, 0.0]), 0.0)
    assert np.allclose(pts[0], tip)
    expected = 18.5 + 3.0 * np.radians(45.0) + 80.0
    assert abs(arcs[-1] - expected) < 1e-6
    # a 45 degree bend turns the heading by 45 degrees
    h_tip = pts[1] - pts[0]
    h_end = pts[-1] - pts[-2]
    cos = np.dot(h_tip, h_end) / (np.linalg.norm(h_tip) * np.linalg.norm(h_end))
    assert abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 45.0) < 1.0


def test_two_mm_probe_fits_a_six_mm_tube_and_exits_the_naris():
    air, inlet = _tube(radius_vox=3)
    edt = airway_wall_distance_mm(air, ISO)
    target = np.array([20.0, 40.0, 20.0])     # x, y, z in mm; deep in the tube
    pl = place_instrument(STRAIGHT_SEEKER, target, edt, inlet, ISO, ORIGIN, n_dirs=200, n_roll=4)
    assert pl.reaches_naris and pl.fits, pl
    assert pl.min_clearance_mm >= 0.0
    assert 30.0 < pl.exit_arc_mm < 42.0       # the naris is ~35-38 mm back along y


def test_four_mm_probe_touches_the_wall_of_a_three_mm_tube():
    air, inlet = _tube(radius_vox=1)          # ~1.5 mm radius
    edt = airway_wall_distance_mm(air, ISO)
    target = np.array([20.0, 40.0, 20.0])
    pl = place_instrument(FAT_PROBE, target, edt, inlet, ISO, ORIGIN, n_dirs=200, n_roll=4)
    assert not pl.fits
    assert pl.min_clearance_mm < 0.0
    assert pl.worst_arc_mm >= 0.0


def test_fixed_bend_tool_does_not_fit_a_straight_narrow_tube():
    """The 45 degree bend puts the shaft into the wall of a 4 mm tube; the
    tip's centreline exists, the tool does not fit."""
    air, inlet = _tube(radius_vox=2)
    edt = airway_wall_distance_mm(air, ISO)
    target = np.array([20.0, 45.0, 20.0])
    pl = place_instrument(ET_SEEKER, target, edt, inlet, ISO, ORIGIN, n_dirs=300, n_roll=8)
    assert not pl.fits, pl


def test_fixed_bend_tool_fits_where_the_airway_turns():
    """A tube with a 45 degree elbow of generous radius takes the bent tool."""
    size, L = 60, 70
    z, y, x = np.mgrid[0:size, 0:L, 0:size]
    c = 30
    # leg A along y (the "nasal cavity"), leg B along +x at 45 degrees from y
    leg_a = ((z - c) ** 2 + (x - c) ** 2) <= 4 ** 2
    leg_a &= (y >= 2) & (y <= 40)
    u = (x - c) - (y - 40)                     # distance from the 45 degree axis, in the x-y plane
    along = ((x - c) + (y - 40)) / np.sqrt(2)
    leg_b = ((z - c) ** 2 + (u ** 2) / 2.0) <= 4 ** 2
    leg_b &= (along >= 0) & (along <= 26)
    air = leg_a | leg_b
    inlet = np.zeros_like(air)
    inlet[:, 2:5, :] = air[:, 2:5, :]
    edt = airway_wall_distance_mm(air, ISO)
    # target 16 mm along leg B from the elbow
    d = 16.0 / np.sqrt(2)
    target = np.array([c + d, 40 + d, float(c)])
    pl = place_instrument(ET_SEEKER, target, edt, inlet, ISO, ORIGIN, n_dirs=400, n_roll=12)
    assert pl.fits, pl


def test_case_driver_reports_missing_targets_explicitly():
    air, inlet = _tube(radius_vox=3)
    recs = [{"name": "maxillary", "side": "L", "ostium_zyx": [20, 40, 20],
             "ostium_diameter_mm": 3.0, "ostium_valid": True}]
    out = fit_instruments_to_ostia(recs, air, inlet, ISO, ORIGIN,
                                   instruments=(STRAIGHT_SEEKER, ET_SEEKER), keep_points=False)
    by = {f["instrument"]: f for f in out["fits"]}
    assert by["straight probe"]["verdict"] == "fits"
    assert by["ET seeker"]["verdict"] == "no target"
    assert "no Eustachian tube orifice detector" in by["ET seeker"]["reason"]
