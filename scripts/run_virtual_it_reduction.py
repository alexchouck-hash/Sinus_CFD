#!/usr/bin/env python3
"""
Virtual inferior turbinate reduction: edit lumen → geometry (+ optional flow).

Writes:
  outputs/{case}_virtual_IT/
  outputs/{case}/{case}_virtual_IT_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.virtual_surgery import run_virtual_it_reduction  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--shave-mm", type=float, default=2.0)
    p.add_argument("--variant-name", default="virtual_IT")
    p.add_argument("--skip-pathlines", action="store_true")
    p.add_argument("--pathline-seeds", type=int, default=80)
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    if not case_dir.is_dir():
        raise SystemExit(f"Missing case dir: {case_dir}")

    compare = run_virtual_it_reduction(
        case_dir,
        args.case,
        shave_mm=args.shave_mm,
        variant_name=args.variant_name,
        recompute_pathlines=not args.skip_pathlines,
        pathline_seeds=args.pathline_seeds,
    )
    print(json.dumps(compare, indent=2)[:2500])
    g = compare.get("geometry") or {}
    print(
        f"\n[{args.case}] virtual IT: removed {compare.get('voxels_removed')} vx "
        f"({compare.get('volume_removed_ml'):.3f} mL)"
    )
    print(
        f"  MCA baseline→virtual: {g.get('mca_mm2_baseline')} → "
        f"{g.get('mca_mm2_virtual')}  (Δ={g.get('mca_delta_mm2')}) mm²"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
