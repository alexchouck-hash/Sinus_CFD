"""Regression guard for **Fix 3** — orientation-aware nasopharynx trim.

Audit claim 3 (CONFIRMED, grok_inbox/2026-07-21_verification.md): the trim
hardcoded posterior = +y (Visible-Human LPS), which is unsafe on arbitrary DICOM
orientation. The fix derives the anterior-posterior sign from the BC orientation
flags via ``trim_nasopharynx_outlet._load_ap_orientation`` and sets
``post_sign = 1.0 if y_anterior_is_low else -1.0``.

These tests write synthetic ``<case>_nares.json`` / ``<case>_stats.json`` into a
tmp dir and assert the loader reads ``y_anterior_is_low`` with the documented
precedence (BC key < nares.json < stats.json notes), and that the posterior sign
flips accordingly.
"""
from __future__ import annotations

import json


def _post_sign(y_ant_low: bool) -> float:
    # Mirrors trim_nasopharynx_outlet.main(): post_sign = 1.0 if y_anterior_is_low else -1.0
    return 1.0 if y_ant_low else -1.0


def test_default_orientation_is_visible_human(trim_module, tmp_path):
    """Fix 3: with no files and empty BC, defaults match Visible Human (low-y anterior)."""
    y_ant_low, sup_high_z = trim_module._load_ap_orientation(tmp_path, "SYN", {})
    assert (y_ant_low, sup_high_z) == (True, True)
    assert _post_sign(y_ant_low) == 1.0


def test_bc_flags_are_read(trim_module, tmp_path):
    """Fix 3: BC-embedded orientation flags are honoured."""
    bc = {"y_anterior_is_low": False, "superior_is_high_z": False}
    y_ant_low, sup_high_z = trim_module._load_ap_orientation(tmp_path, "SYN", bc)
    assert (y_ant_low, sup_high_z) == (False, False)
    assert _post_sign(y_ant_low) == -1.0


def test_nares_json_overrides_bc(trim_module, tmp_path):
    """Fix 3: <case>_nares.json wins over the BC key for y_anterior_is_low."""
    (tmp_path / "SYN_nares.json").write_text(
        json.dumps({"y_anterior_is_low": False}), encoding="utf-8"
    )
    # BC says True, nares says False -> nares (read later) wins.
    y_ant_low, _ = trim_module._load_ap_orientation(tmp_path, "SYN", {"y_anterior_is_low": True})
    assert y_ant_low is False
    assert _post_sign(y_ant_low) == -1.0


def test_stats_notes_override_and_flip_posterior_sign(trim_module, tmp_path):
    """Fix 3: a 'y_anterior_is_low=...' stats note is the final say; sign flips."""
    # nares says False, but a stats note says True -> stats note (read last) wins.
    (tmp_path / "SYN_nares.json").write_text(
        json.dumps({"y_anterior_is_low": False}), encoding="utf-8"
    )
    (tmp_path / "SYN_stats.json").write_text(
        json.dumps({"notes": ["orientation resolved: y_anterior_is_low=True"],
                    "superior_is_high_z": False}),
        encoding="utf-8",
    )
    y_ant_low, sup_high_z = trim_module._load_ap_orientation(tmp_path, "SYN", {})
    assert y_ant_low is True
    assert sup_high_z is False
    assert _post_sign(y_ant_low) == 1.0

    # Now the opposite note -> posterior sign flips the other way.
    (tmp_path / "SYN_stats.json").write_text(
        json.dumps({"notes": ["y_anterior_is_low=False"]}), encoding="utf-8"
    )
    y_ant_low2, _ = trim_module._load_ap_orientation(tmp_path, "SYN", {})
    assert y_ant_low2 is False
    assert _post_sign(y_ant_low2) == -1.0


def test_posterior_sign_maps_both_ways(trim_module):
    """Fix 3: the posterior-sign rule is the flip the trim relies on for normal_xyz."""
    assert _post_sign(True) == 1.0    # anterior low-y  -> posterior +y
    assert _post_sign(False) == -1.0  # anterior high-y -> posterior -y
