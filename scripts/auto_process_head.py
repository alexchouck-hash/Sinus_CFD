#!/usr/bin/env python3
"""
Auto-processing entry point: whole-head CT -> viewer-ready nasal-airway analysis.

One command runs the whole pipeline that was previously a manual chain of steps
(the Visible Human / living-subject whole-head path):

    CT volume
      -> process_whole_head    (airway extraction, skin/bone, nares+trachea ports)
      -> analyze_passage       (centerline, cross-sections, path-aware flow field)
      -> export_openfoam_geometry (watertight solid_air_body + patches)
      -> scaffold_openfoam_case   (prism-layer mesh + thermal case)
      -> [--cfd] Docker simpleFoam + import_openfoam_results (physiological flow)

After the flow stage the case already shows up in the Streamlit viewer
("Whole airway (zones)" and "Nasal airflow"); --cfd replaces the fast
approximate field with a real OpenFOAM solve.

Examples
--------
    # A living-subject whole-head CT (TCIA / CQ500 / your own), fast path:
    py -3.12 scripts/auto_process_head.py --image data/MyPatient/head.nrrd --case MyPatient

    # Fetch a Visible Human cadaver and run it (geometry + approximate flow):
    py -3.12 scripts/auto_process_head.py --download male

    # Full physiological solve (needs Docker Desktop running):
    py -3.12 scripts/auto_process_head.py --image data/MyPatient/head.nrrd --case MyPatient --cfd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
IS_WINDOWS = sys.platform == "win32"

# This entry point prints Unicode glyphs; a cp1252 console would garble them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# download_visible_human_head.py writes these subject-specific paths.
VH = {
    "female": ("VisibleHuman_Head", "data/VisibleHuman_Head/VHFCT1mm_Head.nrrd"),
    "male": ("VisibleHuman_Male_Head", "data/VisibleHuman_Male_Head/VHMCT1mm_Head.nrrd"),
}


class StageError(RuntimeError):
    pass


def run(title: str, argv: list[str], *, optional: bool = False) -> float:
    """Run one child step, streaming its output. Returns elapsed seconds."""
    print(f"\n{'=' * 70}\n>> {title}\n   {' '.join(str(a) for a in argv)}\n{'=' * 70}", flush=True)
    # Child scripts print Unicode (e.g. "nares -> trachea"); a cp1252 console
    # would crash them, so force UTF-8 I/O for every child (and its children).
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    proc = subprocess.run(argv, cwd=str(REPO_ROOT), env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        msg = f"{title} failed (exit {proc.returncode}) after {dt:.0f}s"
        if optional:
            print(f"!! {msg} — continuing (stage was optional).", flush=True)
            raise StageError(msg)
        raise SystemExit(f"\nABORT: {msg}")
    print(f"-- {title} OK ({dt:.0f}s)", flush=True)
    return dt


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *args]


def _trachea_is_reliable(case: str) -> tuple[bool, str]:
    """After process_whole_head, decide whether the trachea outlet is trustworthy.

    Returns (reliable, reason). Unreliable => fall back to the nasopharynx: the
    outlet is a fallback/proxy, is off-midline (a synthesized neck conduit, as on
    the Male cadaver), or is not connected to the nostrils (collapsed pharynx).
    """
    import json as _json

    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage as ndi

    cdir = REPO_ROOT / "outputs" / case
    img = sitk.ReadImage(str(cdir / f"{case}_airway_mask.nrrd"))
    a = sitk.GetArrayFromImage(img).astype(bool)
    org = img.GetOrigin()
    size = img.GetSize()  # x, y, z
    bc = _json.loads((cdir / f"{case}_boundary_conditions.json").read_text())
    ports = {p["name"]: p for p in bc["ports"]}
    if bc.get("outlet_is_proxy"):
        return False, "outlet is a proxy/fallback (no real trachea found)"

    def idx(name):
        """Physical mm (x,y,z) -> integer index; spacing + Direction via ITK (Fix 1)."""
        c = ports[name]["center_mm"]
        ijk = img.TransformPhysicalPointToIndex((float(c[0]), float(c[1]), float(c[2])))
        return [int(min(max(int(ijk[i]), 0), size[i] - 1)) for i in range(3)]

    noL, noR, out = idx("left_nostril"), idx("right_nostril"), idx("trachea")
    # Midline offset in physical mm (robust to non-1 mm spacing).
    xmid_mm = 0.5 * (float(ports["left_nostril"]["center_mm"][0])
                     + float(ports["right_nostril"]["center_mm"][0]))
    offset_mm = abs(float(ports["trachea"]["center_mm"][0]) - xmid_mm)

    lab, _ = ndi.label(a)
    nl = lab[noL[2], noL[1], noL[0]]
    # nearest airway voxel to the outlet centre, and is it the nostril component?
    zz, yy, xx = np.where(a)
    j = int(np.argmin((zz - out[2]) ** 2 + (yy - out[1]) ** 2 + (xx - out[0]) ** 2))
    reachable = nl > 0 and lab[zz[j], yy[j], xx[j]] == nl
    if not reachable:
        return False, "trachea outlet not connected to the nostrils (collapsed pharynx)"
    if offset_mm > 15.0:
        return False, f"trachea outlet {offset_mm:.0f} mm off the nostril midline (lateral conduit)"
    return True, f"clean midline trachea (outlet {offset_mm:.0f} mm off midline)"


def _stack_dicom(dicom_dir: Path, out_nrrd: Path) -> Path:
    """Stack a DICOM series folder into a single NRRD volume (SimpleITK)."""
    import SimpleITK as sitk  # local import: only needed on the DICOM path

    if not dicom_dir.is_dir():
        raise SystemExit(f"DICOM folder not found: {dicom_dir}")
    print(f">> stacking DICOM series: {dicom_dir}", flush=True)
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:  # recurse for nested series
        ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if ids:
            names = reader.GetGDCMSeriesFileNames(str(dicom_dir), ids[0])
    if not names:
        raise SystemExit(f"No DICOM series found under {dicom_dir}")
    reader.SetFileNames(names)
    img = reader.Execute()
    sz, sp = img.GetSize(), img.GetSpacing()
    print(f"   {len(names)} slices  size(xyz)={sz}  spacing_mm={tuple(round(v,3) for v in sp)}", flush=True)
    if sz[2] < 60:
        print(f"   WARNING: only {sz[2]} slices — confirm this covers nostrils→nasopharynx", flush=True)
    out_nrrd.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_nrrd))
    print(f"   wrote {out_nrrd}", flush=True)
    return out_nrrd


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Whole-head CT volume (.nrrd)")
    src.add_argument("--dicom-dir", type=Path,
                     help="Folder of a whole-head CT DICOM series (stacked to NRRD first)")
    src.add_argument(
        "--download",
        choices=("female", "male"),
        help="Fetch a Visible Human cadaver head CT first, then process it",
    )
    p.add_argument("--case", default=None, help="Case name (default: image stem)")
    p.add_argument("--cfd", action="store_true",
                   help="Also run the OpenFOAM solve in Docker + import results")
    p.add_argument("--tidal-volume-L", type=float, default=0.50)
    p.add_argument("--respiratory-rate", type=float, default=12.0)
    p.add_argument("--streamline-seeds", type=int, default=120)
    p.add_argument("--flow-iterations", type=int, default=450)
    p.add_argument("--preview-flow", action="store_true",
                   help="Also run the potential-flow preview in analyze_passage "
                        "(~18 min on a 0.4 mm scan). Off by default: the CFD never "
                        "reads it and the 20-minute budget is a hard constraint.")
    p.add_argument("--foam-image", default="opencfd/openfoam-run:2412",
                   help="Docker image for the --cfd solve")
    p.add_argument("--stop-after",
                   choices=("geometry", "flow", "export", "scaffold", "cfd"),
                   default="cfd",
                   help="Stop the pipeline early after the named stage")
    p.add_argument("--outlet", choices=("auto", "trachea", "nasopharynx"), default="auto",
                   help="Outlet handling. auto: keep a clean midline trachea, else fall "
                        "back to the nasopharynx (handles a disconnected/collapsed "
                        "pharynx). trachea: always keep it. nasopharynx: always trim there.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    args = p.parse_args()

    # Resolve source image + case name -------------------------------------
    if args.download:
        case_default, rel = VH[args.download]
        image = REPO_ROOT / rel
        case = args.case or case_default
    elif args.dicom_dir:
        ddir = args.dicom_dir if args.dicom_dir.is_absolute() else (REPO_ROOT / args.dicom_dir)
        case = args.case or ddir.parent.name if ddir.name.lower() in ("dicom", "dcm") else (args.case or ddir.name)
        case = case.replace(" ", "_")
        image = REPO_ROOT / "data" / case / f"{case}.nrrd"
        if args.dry_run:
            print(f"(dicom) would stack {ddir} -> {image}")
        else:
            image = _stack_dicom(ddir, image)
    else:
        image = args.image if args.image.is_absolute() else (REPO_ROOT / args.image)
        case = args.case or image.stem.replace(" ", "_")

    # Which stages will run (respecting --stop-after) ----------------------
    order = ["geometry", "flow", "export", "scaffold", "cfd"]
    last = order.index(args.stop_after)
    if not args.cfd and last == order.index("cfd"):
        last = order.index("scaffold")  # don't imply CFD unless asked
    stages = order[: last + 1]

    print(f"case         : {case}")
    print(f"image        : {image}")
    print(f"stages       : {' -> '.join(stages)}")
    print(f"cfd (Docker) : {'yes (' + args.foam_image + ')' if 'cfd' in stages else 'no'}")
    print(f"outputs      : outputs/{case}/   foam: foam/{case}/")
    if args.dry_run:
        return 0

    if args.download:
        run("download Visible Human head CT",
            py("download_visible_human_head.py", "--subject", args.download))
    if not image.is_file():
        raise SystemExit(
            f"Missing image: {image}\n"
            "Pass a whole-head .nrrd via --image, or use --download female|male."
        )

    timings: dict[str, float] = {}

    # 1. geometry: airway + skin/bone + BC ports (flow deferred to analyze_passage)
    timings["process_whole_head"] = run(
        "process_whole_head (airway + boundary ports)",
        py("process_whole_head.py",
           "--image", str(image), "--case", case, "--skip-flow",
           "--tidal-volume-L", str(args.tidal_volume_L),
           "--respiratory-rate", str(args.respiratory_rate)),
    )

    # 1b. outlet resolution: keep a clean midline trachea, else fall back to the
    #     nasopharynx so a disconnected/collapsed pharynx still runs in one command.
    trim = False
    if args.outlet == "nasopharynx":
        trim, why = True, "forced by --outlet nasopharynx"
    elif args.outlet == "auto":
        reliable, why = _trachea_is_reliable(case)
        trim = not reliable
    else:
        why = "kept by --outlet trachea"
    print(f"\n[{case}] outlet: {'TRIM to nasopharynx' if trim else 'keep trachea'} — {why}")
    if trim:
        timings["trim_nasopharynx_outlet"] = run(
            "trim_nasopharynx_outlet (terminate at the choanae)",
            py("trim_nasopharynx_outlet.py", "--case", case, "--no-flow"),
        )

    # 2. passage: centerline, cross-sections and the inlet/outlet port masks the
    #    export needs. The potential-flow PREVIEW is opt-in: on CQ500CT390 it
    #    cost 1,075 s of an 18-minute stage whose port masks take 100 s, and the
    #    CFD never reads it. With the 20-minute budget a hard constraint, the
    #    default path is the fast one.
    if "flow" in stages:
        ap_args = ["analyze_passage.py", "--case", case]
        if args.preview_flow:
            label = "analyze_passage (centerline + potential-flow preview)"
            ap_args += ["--flow-iterations", str(args.flow_iterations),
                        "--streamline-seeds", str(args.streamline_seeds)]
        else:
            label = "analyze_passage (centerline + port masks; preview skipped)"
            ap_args += ["--skip-flow"]
        timings["analyze_passage"] = run(label, py(*ap_args))

    # 3-4. OpenFOAM geometry + case scaffold
    if "export" in stages:
        timings["export_openfoam_geometry"] = run(
            "export_openfoam_geometry (watertight solid + patches)",
            py("export_openfoam_geometry.py", "--case", case),
        )
    if "scaffold" in stages:
        timings["scaffold_openfoam_case"] = run(
            "scaffold_openfoam_case (mesh + thermal case)",
            py("scaffold_openfoam_case.py", "--case", case),
        )

    # 5. Physiological CFD solve in Docker, then import the real field
    cfd_ran = False
    if "cfd" in stages:
        if not IS_WINDOWS:
            print("\n!! --cfd auto-run uses the Windows PowerShell Docker wrapper; "
                  "on this platform run the solve manually:\n"
                  f"   docker run --rm -v <repo>/foam/{case}:/case -w /case "
                  f"{args.foam_image} bash Allrun.docker\n"
                  f"   then: {Path(sys.executable).name} scripts/import_openfoam_results.py --case {case}")
        else:
            try:
                timings["docker_cfd"] = run(
                    "OpenFOAM solve (Docker simpleFoam)",
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                     str(SCRIPTS / "run_openfoam_docker.ps1"),
                     "-Case", case, "-Image", args.foam_image],
                    optional=True,
                )
                timings["import_openfoam_results"] = run(
                    "import_openfoam_results (real CFD field -> viewer)",
                    py("import_openfoam_results.py", "--case", case,
                       "--seeds", str(args.streamline_seeds)),
                )
                cfd_ran = True
            except StageError:
                print("   Keeping the approximate flow field from analyze_passage; "
                      "start Docker Desktop and re-run with --cfd to get the OpenFOAM solve.")

    # Summary --------------------------------------------------------------
    out = REPO_ROOT / "outputs" / case
    have = lambda name: (out / f"{case}_{name}").is_file()  # noqa: E731
    print(f"\n{'=' * 70}\nDONE — case '{case}'")
    total = sum(timings.values())
    for k, v in timings.items():
        print(f"  {k:28s} {v:6.0f}s")
    print(f"  {'total':28s} {total:6.0f}s")

    print("\nProduced (outputs/%s/):" % case)
    for marker, label in [
        ("passage.json", "whole-airway centerline/zones"),
        ("flow.npz", "flow field" + (" (OpenFOAM)" if cfd_ran else " (approximate)")),
        ("streamlines.json", "streamlines"),
        ("airway.stl", "airway surface"),
    ]:
        print(f"  [{'x' if have(marker) else ' '}] {case}_{marker:20s} {label}")

    print("\nViewer — the case now appears in:")
    if have("passage.json"):
        print("  • Whole airway (zones)")
    if have("flow.npz"):
        print("  • Nasal airflow")
    print("\nLaunch:  py -3.12 -m streamlit run app/viewer.py")
    if "cfd" in stages and not cfd_ran and IS_WINDOWS:
        print("Physiological CFD: start Docker Desktop, then re-run this command with --cfd.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
