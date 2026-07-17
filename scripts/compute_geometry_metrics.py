#!/usr/bin/env python3
"""Compute dual-side CSA profiles and MCA (geometry-only; no CFD)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.geometry_metrics import (  # noqa: E402
    compute_geometry_metrics,
    write_geometry_metrics,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--sample-every", type=int, default=2)
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    report = compute_geometry_metrics(
        case_dir, args.case, sample_every=args.sample_every
    )
    path = write_geometry_metrics(case_dir, args.case, report=report)
    print(f"[{args.case}] wrote {path.name}")
    g = report.get("global_mca") or {}
    print(
        f"  global MCA={g.get('mca_mm2', 0):.2f} mm²  "
        f"side={g.get('side')}  s={g.get('mca_path_s_mm', 0):.1f} mm  "
        f"xyz={g.get('mca_xyz_mm')}"
    )
    for s in report.get("sides") or []:
        m = s.get("mca") or {}
        print(
            f"  {s['name']}: len={s.get('centerline_length_mm', 0):.1f} mm  "
            f"MCA={m.get('mca_mm2', 0):.2f} mm²  "
            f"area min/mean/max="
            f"{s.get('area_min_mm2', 0):.1f}/"
            f"{s.get('area_mean_mm2', 0):.1f}/"
            f"{s.get('area_max_mm2', 0):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
