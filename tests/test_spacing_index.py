"""Regression guard for **Fix 1** — spacing-aware mm->voxel-index conversion.

Audit claim 1 (CONFIRMED, grok_inbox/2026-07-21_verification.md): the old code
mapped a physical mm point to a voxel index with ``round(mm - origin)`` and
ignored voxel spacing, so on any non-1mm scan the wrong voxel was sampled and
``offset_mm`` was wrong. The fix converts with ``(mm - origin) / spacing`` via
``img.TransformPhysicalPointToIndex`` (which also honours ITK Direction).

These tests build tiny synthetic SimpleITK images with non-1mm spacing and
assert (a) the ITK conversion is spacing-correct and (b) the OLD buggy formula
would land on a *different* (wrong) voxel — so the fix demonstrably matters.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def _make_image(size_xyz, spacing_xyz, origin_xyz) -> sitk.Image:
    img = sitk.Image(int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2]), sitk.sitkUInt8)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetOrigin(tuple(float(o) for o in origin_xyz))
    return img


def _buggy_round_idx(pt_mm, origin_xyz):
    """The pre-fix conversion: round(mm - origin), NO division by spacing."""
    return tuple(int(round(float(pt_mm[i]) - float(origin_xyz[i]))) for i in range(3))


def test_transform_point_to_index_is_spacing_correct():
    """Fix 1: ITK mm->index divides by spacing; the buggy round() does not."""
    spacing = (0.5, 1.0, 1.5)  # x, y, z  (two non-1mm axes)
    origin = (0.0, 0.0, 0.0)
    img = _make_image((24, 20, 16), spacing, origin)

    # Point that lands exactly on integer index (10, 3, 4) once spacing applies.
    pt = (5.0, 3.0, 6.0)  # 5/0.5=10, 3/1.0=3, 6/1.5=4
    correct = (10, 3, 4)

    ijk = tuple(int(v) for v in img.TransformPhysicalPointToIndex(pt))
    assert ijk == correct

    buggy = _buggy_round_idx(pt, origin)
    assert buggy == (5, 3, 6)
    # The bug matters: buggy disagrees on the two non-1mm axes.
    assert buggy != ijk
    assert buggy[0] != ijk[0] and buggy[2] != ijk[2]


def test_transform_point_to_index_with_nonzero_origin():
    """Fix 1: spacing-correct conversion holds under a shifted origin too."""
    spacing = (0.5, 2.0, 1.5)
    origin = (-10.0, 5.0, 2.0)
    img = _make_image((24, 20, 16), spacing, origin)

    target = (12, 4, 5)  # desired integer index
    pt = tuple(origin[i] + target[i] * spacing[i] for i in range(3))  # (-4.0, 13.0, 9.5)

    ijk = tuple(int(v) for v in img.TransformPhysicalPointToIndex(pt))
    assert ijk == target

    buggy = _buggy_round_idx(pt, origin)  # round(6.0, 8.0, 7.5)
    assert buggy != ijk


def test_script_idx_helper_uses_spacing(trim_module):
    """Fix 1: the actual trim_nasopharynx_outlet._idx helper is spacing-aware."""
    spacing = (0.5, 1.0, 1.5)
    origin = (0.0, 0.0, 0.0)
    img = _make_image((24, 20, 16), spacing, origin)
    pt = (5.0, 3.0, 6.0)

    got = trim_module._idx(pt, img)
    assert list(got) == [10, 3, 4]
    assert list(got) != list(_buggy_round_idx(pt, origin))


def test_script_idx_helper_clips_out_of_bounds(trim_module):
    """Fix 1: _idx clamps indices into the valid voxel range (no negative/oob)."""
    spacing = (0.5, 1.0, 1.5)
    origin = (0.0, 0.0, 0.0)
    size = (24, 20, 16)  # x, y, z
    img = _make_image(size, spacing, origin)

    # Physical point well outside the volume on the high side.
    far = (1000.0, 1000.0, 1000.0)
    idx = trim_module._idx(far, img)
    assert idx == [size[0] - 1, size[1] - 1, size[2] - 1]

    # And on the low side.
    neg = (-1000.0, -1000.0, -1000.0)
    assert trim_module._idx(neg, img) == [0, 0, 0]
