#!/usr/bin/env python3
"""
Download the NLM Visible Human Female 1 mm head CT from Harvard Dataverse
and stack it into a single NRRD volume for Sinus_CFD.

Source dataset:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3JDZCT
  Iowa MRRF mirror notes:
  https://mri.medicine.uiowa.edu/equipment-information/scanner-images/visible-human-project-ct-datasets

Series used: VHFCT1mm-Head (~234 DICOM slices, ~120 MB total).
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PID = "doi:10.7910/DVN/3JDZCT"
API_BASE = "https://dataverse.harvard.edu/api"
# Both Visible Human cadavers' head CT live on the same Dataverse record:
#   VHFCT1mm-Head = Female (234 slices), VHMCT1mm-Head = Male (245 slices).
SERIES_BY_SUBJECT = {"female": "VHFCT1mm-Head", "male": "VHMCT1mm-Head"}


def _api_get_json(url: str) -> dict:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Sinus_CFD/0.1"})
    with urllib.request.urlopen(url=req, timeout=120, context=ctx) as r:
        return json.load(r)


def list_head_files(series_prefix: str) -> list[dict]:
    url = f"{API_BASE}/datasets/:persistentId/?persistentId={DATASET_PID}"
    data = _api_get_json(url)
    files = data["data"]["latestVersion"]["files"]
    out = []
    for f in files:
        df = f["dataFile"]
        name = df["filename"]
        if not name.startswith(series_prefix):
            continue
        m = re.search(r"\((\d+)\)", name)
        idx = int(m.group(1)) if m else -1
        out.append(
            {
                "id": df["id"],
                "filename": name,
                "index": idx,
                "size": df.get("filesize", 0),
            }
        )
    out.sort(key=lambda x: x["index"])
    return out


def download_file(file_id: int, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    url = f"{API_BASE}/access/datafile/{file_id}"
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Sinus_CFD/0.1"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, context=ctx, timeout=180) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def stack_dicom_series(dicom_dir: Path, out_nrrd: Path) -> sitk.Image:
    """Read DICOM series with SimpleITK and write NRRD."""
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        # Fallback: sorted .dcm files (may lack full geometry metadata)
        paths = sorted(dicom_dir.glob("*.dcm"), key=lambda p: _slice_index(p.name))
        if not paths:
            raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
        names = [str(p) for p in paths]
    reader.SetFileNames(names)
    image = reader.Execute()
    out_nrrd.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_nrrd))
    return image


def _slice_index(name: str) -> int:
    m = re.search(r"\((\d+)\)", name)
    return int(m.group(1)) if m else 0


def write_preview(image: sitk.Image, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    arr = sitk.GetArrayFromImage(image)  # z,y,x
    z = arr.shape[0] // 2
    y = arr.shape[1] // 2
    x = arr.shape[2] // 2
    # Soft-tissue window-ish
    lo, hi = -200, 400
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sl, title in zip(
        axes,
        [arr[z], arr[:, y, :], arr[:, :, x]],
        [f"Axial z={z}", f"Coronal y={y}", f"Sagittal x={x}"],
    ):
        ax.imshow(np.clip(sl, lo, hi), cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Visible Human Female — Head CT (1 mm)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subject",
        choices=("female", "male"),
        default="female",
        help="Which Visible Human cadaver's head CT (both on the same Dataverse)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/VisibleHuman_Head (female) or data/VisibleHuman_Male_Head (male)",
    )
    p.add_argument("--max-slices", type=int, default=None, help="Debug: limit slice count")
    args = p.parse_args()

    series_prefix = SERIES_BY_SUBJECT[args.subject]
    tag = "VHFCT1mm" if args.subject == "female" else "VHMCT1mm"
    case_name = "VisibleHuman_Head" if args.subject == "female" else "VisibleHuman_Male_Head"
    out_dir = args.out_dir or (REPO_ROOT / "data" / case_name)

    dicom_dir = out_dir / "dicom"
    nrrd_path = out_dir / f"{tag}_Head.nrrd"
    preview_path = out_dir / f"{tag}_Head_preview.png"
    manifest_path = out_dir / "manifest.json"

    print(f"Listing {series_prefix} files from Harvard Dataverse…")
    files = list_head_files(series_prefix)
    if args.max_slices:
        files = files[: args.max_slices]
    if not files:
        print("No head files found.", file=sys.stderr)
        return 1
    print(f"Found {len(files)} slices (~{sum(f['size'] for f in files)/1e6:.0f} MB)")

    for i, f in enumerate(files, 1):
        dest = dicom_dir / f["filename"]
        print(f"[{i}/{len(files)}] {f['filename']}")
        download_file(f["id"], dest)

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": f"NLM Visible Human Project — {args.subject.title()} CT 1mm Head",
                "dataverse": f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={DATASET_PID}",
                "series": series_prefix,
                "n_slices": len(files),
                "files": files,
                "citation": (
                    "Visible Human Project CT Datasets, Harvard Dataverse, "
                    "doi:10.7910/DVN/3JDZCT"
                ),
            },
            fh,
            indent=2,
        )

    print("Stacking DICOM → NRRD…")
    image = stack_dicom_series(dicom_dir, nrrd_path)
    size = image.GetSize()
    spacing = image.GetSpacing()
    print(f"Volume size (xyz)={size}  spacing_mm={spacing}")
    write_preview(image, preview_path)
    print(f"Wrote {nrrd_path}")
    print(f"Wrote {preview_path}")
    print("Done. Cite doi:10.7910/DVN/3JDZCT when using this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
