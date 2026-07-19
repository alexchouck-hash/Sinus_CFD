#!/usr/bin/env python3
"""
Mesh-independence study for the P001 nasal CFD case.

Reads the three resolution runs produced by foam/mesh_study.docker
(foam/P001_m1, _m2, _m3 — snappyHexMesh refine levels 1/2/3) and reports
resistance / ΔP at each, plus the relative change between successive meshes.
The mesh is "independent" once refining further changes ΔP by less than a
tolerance (default 5%) — at that point the number can be quoted as validated
rather than resolution-dependent.

Usage:
  py -3.12 scripts/mesh_independence_report.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compute_nasal_resistance import resistance_from_postprocessing  # noqa: E402


def _cell_count(foam_dir: Path) -> int | None:
    owner = foam_dir / "constant" / "polyMesh" / "owner"
    if not owner.is_file():
        return None
    head = owner.read_text(encoding="utf-8", errors="replace")[:4000]
    m = re.search(r"nCells:\s*(\d+)", head)
    return int(m.group(1)) if m else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="P001")
    p.add_argument("--foam-root", type=Path, default=REPO_ROOT / "foam")
    p.add_argument("--levels", default="1,2,3")
    p.add_argument("--rho", type=float, default=1.14)
    p.add_argument("--tol", type=float, default=0.05, help="relative-change convergence tolerance")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    rows: list[dict] = []
    for lvl in levels:
        foam_dir = args.foam_root / f"{args.case}_m{lvl}"
        pp = foam_dir / "postProcessing"
        res = resistance_from_postprocessing(pp, rho=args.rho) if pp.is_dir() else None
        cells = _cell_count(foam_dir)
        if res is None:
            print(f"  level {lvl}: no result (dir {foam_dir.name}) — did the run finish?")
            continue
        rows.append({"level": lvl, "cells": cells, **res})

    if len(rows) < 2:
        print("ERROR: need at least 2 completed resolutions.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["cells"] or r["level"])

    print(f"[{args.case}] mesh-independence study (resting 18 L/min, laminar simpleFoam)")
    print(f"  {'level':>5} {'cells':>10} {'ΔP (Pa)':>10} {'R (Pa·s/mL)':>13} {'Δ vs prev':>10}")
    prev_dp = None
    max_change = 0.0
    for r in rows:
        if prev_dp is None:
            change_str = "—"
        else:
            change = abs(r["dp_pa"] - prev_dp) / prev_dp
            max_change = max(max_change, change)
            change_str = f"{change * 100:.1f}%"
        print(
            f"  {r['level']:>5} {r['cells'] or 0:>10,} {r['dp_pa']:>10.2f} "
            f"{r['r_pa_s_ml']:>13.4f} {change_str:>10}"
        )
        prev_dp = r["dp_pa"]

    finest_change = None
    if len(rows) >= 3:
        finest_change = abs(rows[-1]["dp_pa"] - rows[-2]["dp_pa"]) / rows[-2]["dp_pa"]

    print()
    if finest_change is not None and finest_change <= args.tol:
        print(
            f"  VERDICT: mesh-INDEPENDENT — finest refinement changed ΔP by "
            f"{finest_change * 100:.1f}% (< {args.tol * 100:.0f}%).\n"
            f"  Quote R = {rows[-1]['r_pa_s_ml']:.3f} Pa·s/mL (finest mesh) as the "
            "validated resting value."
        )
    elif finest_change is not None:
        print(
            f"  VERDICT: NOT yet mesh-independent — finest step still moved ΔP "
            f"{finest_change * 100:.1f}% (> {args.tol * 100:.0f}%). Trend direction: "
            f"{'rising' if rows[-1]['dp_pa'] > rows[-2]['dp_pa'] else 'falling'}. "
            "Add a finer level before quoting a validated number."
        )
    else:
        print("  (need 3 levels to judge convergence of the finest step)")

    out = REPO_ROOT / "outputs" / args.case / f"{args.case}_mesh_independence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": args.case, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
