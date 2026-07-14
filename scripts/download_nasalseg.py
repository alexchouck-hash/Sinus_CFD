#!/usr/bin/env python3
"""
Download and extract the NasalSeg public CT dataset (130 labeled volumes).

Source: https://zenodo.org/records/13893419
Paper: Zhang et al., Scientific Data 2024 (NasalSeg)

Usage (from repo root):
  py -3.12 scripts/download_nasalseg.py
  py -3.12 scripts/download_nasalseg.py --force   # re-download zip
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_URL = "https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1"
EXPECTED_MD5 = "ba83df47974d907798542101fa43ac7d"


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory for NasalSeg.zip and NasalSeg/",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    p.add_argument("--skip-md5", action="store_true", help="Skip checksum verification")
    args = p.parse_args()

    data_root = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "NasalSeg.zip"
    out_dir = data_root / "NasalSeg"

    n_img = len(list((out_dir / "images").glob("*.nrrd"))) if (out_dir / "images").is_dir() else 0
    n_lab = len(list((out_dir / "labels").glob("*.nrrd"))) if (out_dir / "labels").is_dir() else 0
    if n_img >= 130 and n_lab >= 130 and not args.force:
        print(f"[NasalSeg] Already extracted: {n_img} images, {n_lab} labels under {out_dir}")
        print("  Next: py -3.12 scripts/prepare_nnunet_nasalseg.py")
        return 0

    if args.force and zip_path.is_file():
        zip_path.unlink()

    if not zip_path.is_file():
        print(f"[NasalSeg] Downloading (~224 MB)…\n  {ZENODO_URL}")
        print(f"  → {zip_path}")

        def _progress(block: int, block_size: int, total: int) -> None:
            if total <= 0:
                return
            done = min(block * block_size, total)
            pct = 100.0 * done / total
            if block % 50 == 0 or done >= total:
                print(f"\r  {pct:5.1f}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)

        try:
            urlretrieve(ZENODO_URL, zip_path, reporthook=_progress)
            print()
        except Exception as exc:
            print(f"\nERROR: download failed: {exc}", file=sys.stderr)
            print("Manual: browser download from https://zenodo.org/records/13893419", file=sys.stderr)
            return 1
    else:
        print(f"[NasalSeg] Using existing zip: {zip_path}")

    if not args.skip_md5:
        print("[NasalSeg] Verifying MD5…")
        got = _md5(zip_path)
        if got != EXPECTED_MD5:
            print(f"ERROR: MD5 mismatch got={got} expected={EXPECTED_MD5}", file=sys.stderr)
            return 1
        print(f"  OK {got}")

    print(f"[NasalSeg] Extracting to {out_dir}…")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    # Zip may create data/NasalSeg/ or nested folder
    if not (out_dir / "images").is_dir():
        # search
        for cand in data_root.rglob("P001_img.nrrd"):
            root = cand.parent.parent
            if (root / "labels").is_dir():
                if root.resolve() != out_dir.resolve():
                    print(f"[NasalSeg] Found nested layout at {root}")
                    out_dir = root
                break

    n_img = len(list((out_dir / "images").glob("*.nrrd")))
    n_lab = len(list((out_dir / "labels").glob("*.nrrd")))
    print(f"[NasalSeg] Done: {n_img} images, {n_lab} labels")
    if n_img < 130 or n_lab < 130:
        print("WARNING: expected 130/130 — check extract path", file=sys.stderr)
        return 1
    print("  Next: py -3.12 scripts/prepare_nnunet_nasalseg.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
