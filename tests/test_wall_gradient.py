"""Regression guard for **Fix 2** — wall-gradient interior-only velocity.

Audit claim 4 (CONFIRMED severe, grok_inbox/2026-07-21_verification.md): the
approximate solver ran ``np.gradient`` on a pressure array whose exterior was 0.
Interior wall voxels then saw a fake pressure cliff (interior p ~0.5 -> 0),
inflating the mucosal |u| ~70-130x so the reported "peak speed" was a wall
artifact, not flow. The fix (``flow_field.pressure_to_velocity``) nearest-fills
exterior pressure from the closest airway voxel before the gradient and zeroes
u off-airway, so the wall gradient is Neumann-like.

This drives the real ``pressure_to_velocity`` on a synthetic thin airway prism
carrying a gentle along-flow (z) pressure ramp and asserts the
boundary-shell / deep-interior mean-|u| ratio is O(1) (< 5x). It also reproduces
the OLD buggy gradient inline and asserts *it* blows the ratio up (>10x), so the
test fails if the fix is ever reverted.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from sinus_cfd.flow_field import pressure_to_velocity


def _thin_airway_and_pressure():
    """A 20x9x9-voxel airway prism with p = 0.5 + 0.02*z inside, NaN outside.

    The constant term (0.5) is what the old code let the wall gradient bite into;
    the small z-slope (0.02) is the only *real* gradient, so a clean field has
    near-uniform |u| and a polluted one spikes at the walls.
    """
    Nz, Ny, Nx = 24, 16, 16
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    mask[2:22, 4:13, 4:13] = True
    zc = np.arange(Nz, dtype=float).reshape(Nz, 1, 1)
    p = 0.5 + 0.02 * np.broadcast_to(zc, mask.shape)
    p = p.copy()
    p[~mask] = np.nan  # mimic solve_pressure_potential (exterior = NaN)
    return mask, p


def _shell_and_deep(mask):
    shell = mask & ~ndi.binary_erosion(mask, iterations=1)
    deep = ndi.binary_erosion(mask, iterations=2)
    assert shell.sum() > 0 and deep.sum() > 0
    return shell, deep


def test_wall_gradient_ratio_is_order_one():
    """Fix 2: fixed pressure_to_velocity keeps shell/deep mean |u| ratio O(1)."""
    mask, p = _thin_airway_and_pressure()
    spacing = (1.0, 1.0, 1.0)

    ux, uy, uz, speed = pressure_to_velocity(p, mask, spacing)

    # Velocity is finite everywhere and exactly zero outside the airway.
    assert np.isfinite(speed).all()
    assert float(np.abs(speed[~mask]).max()) == 0.0

    shell, deep = _shell_and_deep(mask)
    ratio = float(speed[shell].mean() / speed[deep].mean())
    assert ratio < 5.0, f"boundary shell / deep interior |u| ratio too high: {ratio:.1f}x"


def test_old_buggy_gradient_would_pollute_the_wall():
    """Fix 2: the pre-fix exterior-0 gradient inflates wall |u| >10x (guard)."""
    mask, p = _thin_airway_and_pressure()
    sx, sy, sz = 1.0, 1.0, 1.0

    # Reproduce the OLD behaviour: exterior pressure = 0, NO nearest-fill.
    p_buggy = np.where(mask, np.nan_to_num(p, nan=0.0), 0.0)
    gz, gy, gx = np.gradient(p_buggy, sz, sy, sx)
    ux_b = np.where(mask, -gx, 0.0)
    uy_b = np.where(mask, -gy, 0.0)
    uz_b = np.where(mask, -gz, 0.0)
    speed_b = np.sqrt(ux_b**2 + uy_b**2 + uz_b**2)

    shell, deep = _shell_and_deep(mask)
    ratio_buggy = float(speed_b[shell].mean() / speed_b[deep].mean())
    assert ratio_buggy > 10.0, (
        "buggy gradient should pollute the wall; if this is now small the "
        "synthetic no longer exercises the bug"
    )

    # And the fix must be dramatically cleaner than the bug on the same input.
    _ux, _uy, _uz, speed_fixed = pressure_to_velocity(p, mask, (sx, sy, sz))
    ratio_fixed = float(speed_fixed[shell].mean() / speed_fixed[deep].mean())
    assert ratio_fixed < ratio_buggy / 3.0
