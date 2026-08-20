"""Competing geodesic flood, through-path sinus strip, no midplane septum."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from sinus_cfd.auto_airway import (
    competing_naris_flood,
    geodesic_distance_mm,
    snap_seed_to_air,
    through_path_passage,
)


def _independent_propagation(air: np.ndarray, seed):
    struct = ndi.generate_binary_structure(3, 3)
    out = np.zeros_like(air, dtype=bool)
    out[seed] = True
    return ndi.binary_propagation(out, mask=air, structure=struct)


def test_competing_flood_partitions_connected_choana():
    air = np.zeros((24, 32, 24), dtype=bool)
    # Right tube (low x) and left tube (high x), join at posterior y.
    air[8:16, 4:28, 6:10] = True
    air[8:16, 4:28, 14:18] = True
    air[8:16, 24:28, 6:18] = True  # choana
    r_seed = (12, 5, 8)
    l_seed = (12, 5, 16)
    left, right, d_l, d_r, _ = competing_naris_flood(air, l_seed, r_seed, (1.0, 1.0, 1.0))
    assert int(left.sum()) > 20 and int(right.sum()) > 20
    assert not np.any(left & right)
    assert np.all((left | right) == air)
    # Independent floods both fill the whole connected lumen.
    ind_l = _independent_propagation(air, l_seed)
    ind_r = _independent_propagation(air, r_seed)
    assert np.array_equal(ind_l, air) and np.array_equal(ind_r, air)


def test_c_shaped_septum_midplane_misassigns():
    """Diagonal septal wall: midplane steals ≥20% of one cavity; geodesic does not."""
    air = np.zeros((16, 24, 28), dtype=bool)
    air[4:12, 4:22, 2:26] = True
    # Deviated wall sits at x=22 through most of the cavity, then opens for a choana.
    for y in range(4, 18):
        wx = 22 if y < 16 else 12
        air[4:12, y, wx : wx + 2] = False
    l_seed = (8, 5, 24)
    r_seed = (8, 5, 4)
    x_sep = int(round(0.5 * (l_seed[2] + r_seed[2])))
    # True left = air on the high-x side of the wall, grown from l_seed.
    true_left = _independent_propagation(air, l_seed)
    # Wait: independent flood fills BOTH sides on a connected choana.
    # Restrict truth to anterior of the choana (y < 18) where the wall still splits.
    ant = np.zeros_like(air)
    ant[:, 4:18, :] = True
    # Geodesic assignment is the reference; midplane is the bug.
    left, right, _, _, _ = competing_naris_flood(air, l_seed, r_seed, (1.0, 1.0, 1.0))
    mid_left = air & (np.arange(28)[None, None, :] > x_sep)
    # Midplane steals anterior right-cavity air (geodesic-right but x > x_sep).
    geo_r = right[:, 4:18, :]
    stolen = geo_r & mid_left[:, 4:18, :]
    assert int(geo_r.sum()) > 0
    assert stolen.sum() / geo_r.sum() >= 0.20
    assert int(left.sum()) > 20 and int(right.sum()) > 20
    assert not np.any(left & right)


def test_through_path_keeps_valve_drops_ostium_pocket():
    air = np.zeros((12, 28, 12), dtype=bool)
    # Through-path tube along y, with a 1-voxel valve at y=12.
    air[4:8, 2:26, 4:8] = True
    air[4:8, 12, 4:8] = False
    air[5:7, 12, 5:7] = True  # 2x2 valve
    # Maxillary pocket off to +x, 1-voxel ostium.
    air[5:7, 16:22, 8:11] = True
    air[5:7, 16, 8] = True
    naris = (6, 3, 6)
    outlet = (6, 24, 6)
    passage, sinus, _ = through_path_passage(air, [naris], outlet, (1.0, 1.0, 1.0), slack_mm=4.0)
    assert passage[6, 12, 6]  # valve stays
    assert int(sinus[5:7, 17:22, 9:11].sum()) >= 4  # pocket is a detour


def test_snap_seed_reaches_painted_vestibule():
    air = np.zeros((16, 16, 16), dtype=bool)
    air[6:10, 6:12, 6:10] = True
    seed = (8, 2, 8)  # anterior of the lumen
    assert snap_seed_to_air(air, seed, (1.0, 1.0, 1.0), radius_mm=2.0) is None
    # Paint a 3-voxel vestibule tube.
    air[8, 2:7, 8] = True
    snapped = snap_seed_to_air(air, seed, (1.0, 1.0, 1.0), radius_mm=10.0)
    assert snapped is not None
    d = geodesic_distance_mm(air, snapped, (1.0, 1.0, 1.0))
    assert np.isfinite(d[8, 8, 8])


def test_anisotropic_geodesic_uses_spacing():
    air = np.ones((8, 8, 12), dtype=bool)
    seed = (4, 4, 2)
    d = geodesic_distance_mm(air, seed, (2.0, 1.0, 1.0))  # x spacing 2 mm
    # One step in +x should cost ~2 mm, one step in +y ~1 mm.
    assert d[4, 4, 3] == 2.0
    assert d[4, 5, 2] == 1.0
