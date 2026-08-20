"""Regression guard for **bilateral (L/R) seeding balance**.

``scripts/vestibule_to_pharynx_streamlines.py`` (and the identical block in
``flow_field.compute_flow_field``) seed streamlines by *stratifying moving-airway
voxels left/right of the naris midline* and drawing up to ``seeds_per_side`` from
each side. This is the fix for one-sided streamlines: naive global sampling on a
weak field starves one passage; stratified L/R seeding guarantees both nasal
passages get seeds.

The seeding is inline in ``main()`` (not importable without files + argparse), so
this test *faithfully reproduces* that block — see
scripts/vestibule_to_pharynx_streamlines.py lines ~98-112:

    smax = float(speed[domain].max())
    moving = interior & (speed > max(speed_frac * smax, 1e-9))
    zz, yy, xx = np.where(moving)
    xmm = ox + xx * sx
    rng = np.random.default_rng(0)
    seeds = []
    for side in (xmm < x_mid, xmm >= x_mid):
        idx = np.where(side)[0]
        if len(idx):
            pick = rng.choice(idx, size=min(seeds_per_side, len(idx)), replace=False)
            for i in pick:
                seeds.append([ox + xx[i]*sx, oy + yy[i]*sy, oz + zz[i]*sz])

and asserts a symmetric moving field yields a balanced L/R seed split.
"""
from __future__ import annotations

import numpy as np


def _stratified_lr_seeds(moving, speed, spacing, origin, x_mid, seeds_per_side, rng_seed=0):
    """Faithful reproduction of the inline L/R stratified seeding block."""
    ox, oy, oz = origin
    sx, sy, sz = spacing
    zz, yy, xx = np.where(moving)
    xmm = ox + xx * sx
    rng = np.random.default_rng(rng_seed)
    seeds = []
    for side in (xmm < x_mid, xmm >= x_mid):
        idx = np.where(side)[0]
        if len(idx):
            pick = rng.choice(idx, size=min(seeds_per_side, len(idx)), replace=False)
            for i in pick:
                seeds.append([ox + xx[i] * sx, oy + yy[i] * sy, oz + zz[i] * sz])
    return np.asarray(seeds, dtype=float)


def _symmetric_field(n_per_side=256):
    """Two mirror-image lobes of moving airway either side of x_mid = 20."""
    Nz, Ny, Nx = 10, 20, 40
    speed = np.zeros((Nz, Ny, Nx), dtype=float)
    airway = np.zeros((Nz, Ny, Nx), dtype=bool)
    airway[3:7, 6:14, 6:14] = True    # left lobe:  x in [6, 13]
    airway[3:7, 6:14, 26:34] = True   # right lobe: x in [26, 33]
    speed[airway] = 1.0
    moving = airway & (speed > 0.02 * speed.max())
    return moving, speed


def test_symmetric_field_gives_balanced_lr_split():
    """Balance: enough voxels per side -> exactly seeds_per_side on each side."""
    moving, speed = _symmetric_field()
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)
    x_mid = 20.0
    seeds_per_side = 30

    seeds = _stratified_lr_seeds(moving, speed, spacing, origin, x_mid, seeds_per_side)
    L = int((seeds[:, 0] < x_mid).sum())
    R = int((seeds[:, 0] >= x_mid).sum())

    assert L == seeds_per_side and R == seeds_per_side
    assert L > 0 and R > 0
    # Perfectly balanced on a symmetric field.
    assert abs(L - R) <= 1


def test_capped_when_fewer_voxels_than_requested():
    """Balance holds even when each side has fewer moving voxels than requested."""
    moving, speed = _symmetric_field()
    per_side_voxels = (np.where(moving)[2] < 20).sum()  # count on the left half
    assert per_side_voxels > 0
    seeds_per_side = int(per_side_voxels) + 500  # ask for far more than exist

    seeds = _stratified_lr_seeds(moving, speed, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                                 20.0, seeds_per_side)
    L = int((seeds[:, 0] < 20.0).sum())
    R = int((seeds[:, 0] >= 20.0).sum())
    # Both sides fully consumed and still equal (symmetric).
    assert L == R == int(per_side_voxels)


def test_neither_passage_is_starved():
    """The whole point of stratifying: no side gets zero seeds."""
    moving, speed = _symmetric_field()
    seeds = _stratified_lr_seeds(moving, speed, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                                 20.0, 40)
    L = int((seeds[:, 0] < 20.0).sum())
    R = int((seeds[:, 0] >= 20.0).sum())
    assert min(L, R) > 0
    # Split ratio within a tight band of 50/50 on a symmetric field.
    frac = min(L, R) / max(L, R)
    assert frac >= 0.8
