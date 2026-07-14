#!/usr/bin/env python3
"""
Import OpenFOAM simpleFoam U/p onto the CT airway grid for the Streamlit viewer.

Example:
  py -3.12 scripts/import_openfoam_results.py --case VisibleHuman_Head
  py -3.12 scripts/import_openfoam_results.py --case VisibleHuman_Head --time 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.openfoam_import import import_openfoam_to_grid  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--foam-root", type=Path, default=None)
    p.add_argument("--outputs-root", type=Path, default=None)
    p.add_argument("--time", default=None, help="Time directory name (default: latest)")
    p.add_argument("--seeds", type=int, default=48)
    args = p.parse_args()

    foam = args.foam_root or (REPO_ROOT / "foam" / args.case)
    outputs = args.outputs_root or (REPO_ROOT / "outputs" / args.case)

    result = import_openfoam_to_grid(
        case_id=args.case,
        foam_root=foam,
        outputs_root=outputs,
        time_name=args.time,
        n_streamline_seeds=args.seeds,
    )
    print(f"OK method={result.method} time={result.time_name}")
    for n in result.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
