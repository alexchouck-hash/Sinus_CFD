"""An import must read THIS solve, not whatever older run left files behind.

Re-solving VisibleHuman_Male_Head in a case directory that still held July's
results produced a "settled, dP 0.19 Pa" verdict for September -- July's. Two
separate mechanisms, both OpenFOAM behaviour the reader did not account for:

1. Old time directories are left in place. The new run wrote 400/ (127,198
   cells); July's 500/ (240,479 cells) survived, and latest_time_dir took 500.
   The velocities were then padded onto the new mesh and reported as new.
2. An existing surfaceFieldValue.dat is not overwritten; the new history goes
   to surfaceFieldValue_0.dat beside it. Globbing the bare name read July's.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from sinus_cfd.openfoam_import import (
    _read_surface_field_value,
    latest_time_dir,
    select_time_dir_matching_cells,
)


def _u_field(path: Path, n: int) -> None:
    body = "\n".join("(0.1 0 0)" for _ in range(n))
    path.write_text(
        f"internalField nonuniform List<vector>\n{n}\n(\n{body}\n);\n", encoding="utf-8"
    )


def _case(tmp_path):
    for name, n in (("0", 1), ("400", 5), ("500", 9)):
        d = tmp_path / name
        d.mkdir()
        _u_field(d / "U", n)
        (d / "p").write_text("solved", encoding="utf-8")   # a solve writes p; U alone is not one
    (tmp_path / "constant").mkdir()
    (tmp_path / "system").mkdir()
    return tmp_path


def test_latest_time_dir_is_still_the_numerically_latest(tmp_path):
    assert latest_time_dir(_case(tmp_path)) == "500"


def test_matching_cells_skips_a_newer_dir_from_another_mesh(tmp_path):
    case = _case(tmp_path)
    assert select_time_dir_matching_cells(case, 5) == "400"
    assert select_time_dir_matching_cells(case, 9) == "500"


def test_no_fitting_field_returns_none_so_the_caller_fails_loudly(tmp_path):
    assert select_time_dir_matching_cells(_case(tmp_path), 7) is None


def _history(root: Path, fname: str, rows, mtime: float) -> Path:
    d = root / "postProcessing" / "p_left_nostril" / "0"
    d.mkdir(parents=True, exist_ok=True)
    p = d / fname
    p.write_text("# t v\n" + "\n".join(f"{t}\t{v}" for t, v in rows) + "\n", encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def test_history_reader_takes_the_newest_file_only(tmp_path):
    old = time.time() - 3600
    _history(tmp_path, "surfaceFieldValue.dat", [(450, 0.162), (500, 0.1623)], old)
    _history(tmp_path, "surfaceFieldValue_0.dat", [(350, 6.20), (400, 6.17)], old + 1800)
    rows = _read_surface_field_value(tmp_path, "p_left_nostril")
    assert rows == [(350.0, 6.20), (400.0, 6.17)]


def test_history_reader_does_not_merge_two_runs(tmp_path):
    """Concatenating both files would have produced a plausible, wrong curve."""
    now = time.time()
    _history(tmp_path, "surfaceFieldValue.dat", [(50, 1.0), (100, 2.0)], now - 10)
    _history(tmp_path, "surfaceFieldValue_0.dat", [(50, 9.0)], now)
    assert _read_surface_field_value(tmp_path, "p_left_nostril") == [(50.0, 9.0)]
