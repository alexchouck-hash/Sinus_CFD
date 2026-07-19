#!/usr/bin/env python3
"""
Sinus_CFD interactive viewer (Streamlit + Plotly).

Demo layers (Visible Human):
  - Skin + L/R cavities + frontal sinus
  - Turbulent wispy pathlines (volume/naris seeds → trachea)
  - Purple dual naris→frontal instrument paths
  - Pink high-|u| zones: inferior turbinate / middle turbinate / septum
  - Treatment recommendations (heuristic)

Sidebar: toggles for frontal paths and each removal zone.
See docs/viewer.md and AGENTS.md.

Run from repo root:
  py -3.12 -m streamlit run app/viewer.py --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Bump when viewer behavior or expected data layout changes (shown in UI).
APP_VERSION = "0.17.0-mvp-geometry"
APP_VERSION_LABEL = (
    "MVP geometry view: interactive L/R constriction charts · tri-planar CT "
    "with cavity overlay · rotatable 3D airway"
)

DATA_ROOT = REPO_ROOT / "data"
LEFT_COLOR = "#2b8cbe"   # blue = left cavity
RIGHT_COLOR = "#d6604d"  # red = right cavity

DEFAULT_CASE = "P001"
OUTPUTS = REPO_ROOT / "outputs"


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def case_data_fingerprint(case_id: str) -> str:
    """Stamp from key output mtimes so cache invalidates after reprocess."""
    case_dir = OUTPUTS / case_id
    keys = [
        f"{case_id}_flow.npz",
        f"{case_id}_flow_meta.json",
        f"{case_id}_streamlines.json",
        f"{case_id}_boundary_conditions.json",
        f"{case_id}_stats.json",
        f"{case_id}_skin.stl",
        f"{case_id}_airway.stl",
        f"{case_id}_nares.json",
        f"{case_id}_passage.json",
        f"{case_id}_open_paths.json",
        f"{case_id}_restriction.npz",
        f"{case_id}_surgical_guidance.json",
        f"{case_id}_removal_highlight.npz",
        f"{case_id}_sinus_frontal.stl",
        f"{case_id}_septum.stl",
        f"{case_id}_cavity_left.stl",
        f"{case_id}_cavity_right.stl",
        f"{case_id}_ct_nasal_meta.json",
    ]
    return "|".join(f"{n}:{int(_file_mtime(case_dir / n))}" for n in keys)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading flow field…")
def load_case(case_id: str, data_fingerprint: str) -> dict:
    """data_fingerprint must change when on-disk outputs are reprocessed."""
    case_dir = OUTPUTS / case_id
    npz_path = case_dir / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        return {
            "error": (
                f"Missing {npz_path}. Run: "
                f"py -3.12 scripts/process_whole_head.py --case {case_id}"
            )
        }

    data = np.load(npz_path)
    out = {
        "case_id": case_id,
        "data_fingerprint": data_fingerprint,
        "airway": data["airway"].astype(bool),
        "speed": data["speed"].astype(np.float32),
        "ux": data["ux"].astype(np.float32),
        "uy": data["uy"].astype(np.float32),
        "uz": data["uz"].astype(np.float32),
        "pressure": data["pressure"].astype(np.float32),
        "spacing": data["spacing_xyz_mm"].astype(float),
        "origin": data["origin_xyz_mm"].astype(float),
        "inlet": data["inlet_mask"].astype(bool),
        "outlet": data["outlet_mask"].astype(bool),
    }

    sl_path = case_dir / f"{case_id}_streamlines.json"
    if sl_path.is_file():
        with sl_path.open(encoding="utf-8") as f:
            sl = json.load(f)
        out["streamlines"] = sl.get("lines") or []
        out["streamline_speeds"] = sl.get("speeds_m_s") or []
    else:
        out["streamlines"] = []
        out["streamline_speeds"] = []

    meta_path = case_dir / f"{case_id}_flow_meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            out["meta"] = json.load(f)
    else:
        out["meta"] = {}

    open_paths = case_dir / f"{case_id}_open_paths.json"
    if open_paths.is_file():
        with open_paths.open(encoding="utf-8") as f:
            out["open_paths"] = json.load(f)
    else:
        out["open_paths"] = {}

    rest_path = case_dir / f"{case_id}_restriction.npz"
    if rest_path.is_file():
        rd = np.load(rest_path)
        out["restriction_pts"] = rd["points_xyz_score_r_mm"].astype(np.float32)
        out["restriction_thr"] = float(rd["threshold_1_over_r"]) if "threshold_1_over_r" in rd else 0.0
        out["restriction_min_r"] = float(rd["min_radius_mm"]) if "min_radius_mm" in rd else 0.0
    else:
        out["restriction_pts"] = None
        out["restriction_thr"] = 0.0
        out["restriction_min_r"] = 0.0

    nares_path = case_dir / f"{case_id}_nares.json"
    if nares_path.is_file():
        with nares_path.open(encoding="utf-8") as f:
            out["nares"] = json.load(f)
    else:
        out["nares"] = {}

    surg_path = case_dir / f"{case_id}_surgical_guidance.json"
    if surg_path.is_file():
        with surg_path.open(encoding="utf-8") as f:
            out["surgical"] = json.load(f)
    else:
        out["surgical"] = {}

    rem_npz = case_dir / f"{case_id}_removal_highlight.npz"
    if rem_npz.is_file():
        rd = np.load(rem_npz)
        out["removal_pts"] = rd["points_xyz_r_mm"].astype(np.float32)
        out["removal_inferior_turbinate"] = (
            rd["inferior_turbinate"].astype(np.float32)
            if "inferior_turbinate" in rd.files
            else None
        )
        out["removal_middle_turbinate"] = (
            rd["middle_turbinate"].astype(np.float32)
            if "middle_turbinate" in rd.files
            else None
        )
        out["removal_septum"] = (
            rd["septum"].astype(np.float32) if "septum" in rd.files else None
        )
    else:
        out["removal_pts"] = None
        out["removal_inferior_turbinate"] = None
        out["removal_middle_turbinate"] = None
        out["removal_septum"] = None

    for key, fname in (
        ("frontal_stl", f"{case_id}_sinus_frontal.stl"),
        ("sphenoid_stl", f"{case_id}_sinus_sphenoid.stl"),
        ("maxillary_left_stl", f"{case_id}_sinus_maxillary_left.stl"),
        ("maxillary_right_stl", f"{case_id}_sinus_maxillary_right.stl"),
    ):
        fp = case_dir / fname
        out[key] = str(fp) if fp.is_file() else None

    stl_path = case_dir / f"{case_id}_airway.stl"
    out["stl_path"] = str(stl_path) if stl_path.is_file() else None
    head_stl = case_dir / f"{case_id}_head.stl"
    out["head_stl_path"] = str(head_stl) if head_stl.is_file() else None
    skin_stl = case_dir / f"{case_id}_skin.stl"
    out["skin_stl_path"] = str(skin_stl) if skin_stl.is_file() else None
    bone_stl = case_dir / f"{case_id}_bone.stl"
    out["bone_stl_path"] = str(bone_stl) if bone_stl.is_file() else None
    septum_stl = case_dir / f"{case_id}_septum.stl"
    out["septum_stl_path"] = str(septum_stl) if septum_stl.is_file() else None
    left_stl = case_dir / f"{case_id}_cavity_left.stl"
    out["left_cavity_stl_path"] = str(left_stl) if left_stl.is_file() else None
    right_stl = case_dir / f"{case_id}_cavity_right.stl"
    out["right_cavity_stl_path"] = str(right_stl) if right_stl.is_file() else None
    mucosa_stl = case_dir / f"{case_id}_mucosa_wall.stl"
    out["mucosa_stl_path"] = str(mucosa_stl) if mucosa_stl.is_file() else None
    ct_meta = case_dir / f"{case_id}_ct_nasal_meta.json"
    if ct_meta.is_file():
        with ct_meta.open(encoding="utf-8") as f:
            out["ct_nasal"] = json.load(f)
    else:
        out["ct_nasal"] = {}

    bc_path = case_dir / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        with bc_path.open(encoding="utf-8") as f:
            out["bc"] = json.load(f)
        out["bc_mtime"] = datetime.fromtimestamp(
            _file_mtime(bc_path), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        out["bc"] = {}
        out["bc_mtime"] = None

    stats_path = case_dir / f"{case_id}_stats.json"
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8") as f:
            out["stats"] = json.load(f)
        out["stats_mtime"] = datetime.fromtimestamp(
            _file_mtime(stats_path), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        out["stats"] = {}
        out["stats_mtime"] = None

    face_qc = case_dir / f"{case_id}_preview_face_nares.png"
    out["face_qc_path"] = str(face_qc) if face_qc.is_file() else None
    preview_path = case_dir / f"{case_id}_preview.png"
    out["preview_path"] = str(preview_path) if preview_path.is_file() else None

    passage_path = case_dir / f"{case_id}_passage.json"
    if passage_path.is_file():
        with passage_path.open(encoding="utf-8") as f:
            out["passage"] = json.load(f)
    else:
        out["passage"] = {}

    return out


def list_cases() -> list[str]:
    if not OUTPUTS.is_dir():
        return []
    cases = []
    for d in sorted(OUTPUTS.iterdir()):
        if d.is_dir() and (d / f"{d.name}_flow.npz").is_file():
            cases.append(d.name)
    return cases


def list_geometry_cases() -> list[str]:
    """Cases with a Stage-2 geometry report (scripts/geometry_report.py output)."""
    if not OUTPUTS.is_dir():
        return []
    return [
        d.name
        for d in sorted(OUTPUTS.iterdir())
        if d.is_dir() and (d / f"{d.name}_geometry_report.json").is_file()
    ]


def load_geometry_report(case_id: str) -> dict | None:
    path = OUTPUTS / case_id / f"{case_id}_geometry_report.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_case_ct_label(case_id: str) -> tuple[Path | None, Path | None]:
    """Find (image, label) NRRD paths for a geometry-report case."""
    # NasalSeg native case (P001, P065, …)
    img = DATA_ROOT / "images" / f"{case_id}_img.nrrd"
    lab = DATA_ROOT / "labels" / f"{case_id}_seg.nrrd"
    if img.is_file() and lab.is_file():
        return img, lab
    # Post-op virtual-surgery case: edited label under outputs/, base image from
    # the parent NasalSeg case id (e.g. P065_postop_septoplasty_left -> P065).
    postop = OUTPUTS / case_id / f"{case_id.split('_')[0]}_postop_label.nrrd"
    if not postop.is_file():
        hits = list((OUTPUTS / case_id).glob("*_postop_label.nrrd"))
        postop = hits[0] if hits else postop
    base_id = case_id.split("_")[0]
    base_img = DATA_ROOT / "images" / f"{base_id}_img.nrrd"
    if postop.is_file() and base_img.is_file():
        return base_img, postop
    return None, None


@st.cache_data(show_spinner=False)
def _load_ct_label(case_id: str) -> dict | None:
    """Load CT + label arrays (z,y,x) and spacing for a case, or None."""
    import SimpleITK as sitk

    img_p, lab_p = _resolve_case_ct_label(case_id)
    if img_p is None or lab_p is None:
        return None
    img = sitk.ReadImage(str(img_p))
    lab_img = sitk.ReadImage(str(lab_p))
    ct = sitk.GetArrayFromImage(img)
    label = sitk.GetArrayFromImage(lab_img)
    if ct.shape != label.shape:
        # geometry mismatch on some NasalSeg labels — index correspondence holds
        label = label if label.shape == ct.shape else None
    return {
        "ct": ct,
        "label": label,
        "spacing_xyz": tuple(float(v) for v in img.GetSpacing()),
    }


def _area_profile_fig(report: dict) -> go.Figure:
    """Interactive L/R cross-sectional-area-vs-distance chart with MCA markers."""
    fig = go.Figure()
    for side_key, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
        s = report.get(side_key, {})
        prof = s.get("area_profile") or []
        if not s.get("present") or not prof:
            continue
        ap = [p["ap_mm"] for p in prof]
        area = [p["area_mm2"] for p in prof]
        fig.add_trace(go.Scatter(
            x=ap, y=area, mode="lines", name=f"{side_key} ({s['volume_ml']:.1f} mL)",
            line=dict(color=color, width=2.5),
            hovertemplate=f"{side_key}<br>%{{x:.1f}} mm<br>%{{y:.0f}} mm²<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[s["mca_ap_position_mm"]], y=[s["mca_mm2"]], mode="markers+text",
            marker=dict(color=color, size=14, symbol="triangle-down",
                        line=dict(color="white", width=1)),
            text=[f"MCA {s['mca_mm2']:.0f}"], textposition="bottom center",
            textfont=dict(color=color, size=11), showlegend=False,
            hovertemplate=f"{side_key} MCA<br>%{{y:.0f}} mm² @ %{{x:.1f}} mm<extra></extra>",
        ))
    fig.update_layout(
        height=420,
        xaxis_title="distance from anterior naris (mm)",
        yaxis_title="cross-sectional area (mm²)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=20, t=30, b=50),
        hovermode="x unified",
    )
    return fig


def _airway_center(label: np.ndarray) -> tuple[int, int, int]:
    airway = np.isin(label, (1, 2, 3))
    if airway.any():
        zz, yy, xx = np.where(airway)
        return int(np.median(zz)), int(np.median(yy)), int(np.median(xx))
    return tuple(s // 2 for s in label.shape)  # type: ignore[return-value]


def _triplanar_ct_fig(ct: np.ndarray, label: np.ndarray, cz: int, cy: int, cx: int) -> go.Figure:
    """CT tri-planar with L/R cavity overlay at the given slice indices."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f"Axial z={cz}", f"Coronal y={cy}", f"Sagittal x={cx}"), horizontal_spacing=0.04)

    planes = [
        (ct[cz], label[cz]),
        (ct[:, cy, :], label[:, cy, :]),
        (ct[:, :, cx], label[:, :, cx]),
    ]
    for col, (ct_slice, lab_slice) in enumerate(planes, start=1):
        disp = np.clip(ct_slice, -1000, 400).astype(np.float32)
        fig.add_trace(go.Heatmap(z=disp, colorscale="gray", showscale=False,
                                 hoverinfo="skip"), row=1, col=col)
        # L (blue) and R (red) cavity overlays
        for lid, color in ((1, LEFT_COLOR), (2, RIGHT_COLOR)):
            m = lab_slice == lid
            if m.any():
                ov = np.where(m, 1.0, np.nan)
                fig.add_trace(go.Heatmap(
                    z=ov, colorscale=[[0, color], [1, color]], showscale=False,
                    opacity=0.45, hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(visible=False, row=1, col=col)
        fig.update_yaxes(visible=False, scaleanchor=f"x{col if col>1 else ''}",
                         row=1, col=col)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig


@st.cache_data(show_spinner=False)
def _airway_3d_traces(case_id: str, spacing_xyz: tuple) -> list:
    """Marching-cubes L/R cavity + nasopharynx surfaces as Plotly Mesh3d."""
    from skimage import measure

    data = _load_ct_label(case_id)
    if data is None or data["label"] is None:
        return []
    label = data["label"]
    sx, sy, sz = spacing_xyz
    traces = []
    parts = [(1, "left cavity", LEFT_COLOR, 0.55),
             (2, "right cavity", RIGHT_COLOR, 0.55),
             (3, "nasopharynx", "#7fbf7b", 0.35)]
    for lid, nm, color, op in parts:
        mask = label == lid
        if mask.sum() < 50:
            continue
        try:
            # step_size=2 ~quarters the face count for a lighter/faster WebGL mesh
            verts, faces, _n, _v = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=(sz, sy, sx), step_size=2)
        except (ValueError, RuntimeError):
            continue
        # verts are (z,y,x) physical → plot as (x,y,z)
        traces.append(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=op, name=nm, showlegend=True,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2), flatshading=False,
        ))
    return traces


def render_geometry_report() -> None:
    """
    MVP nasal-airway geometry view: interactive L/R constriction charts, a
    tri-planar CT with cavity overlay, and a rotatable 3D airway. Reads the
    Stage-2 geometry report (scripts/geometry_report.py) plus the case CT/label.
    """
    st.title("Sinus_CFD — Nasal Airway Geometry")
    st.caption("Per-side volume · minimal cross-sectional area (MCA) · L/R asymmetry")

    geo_cases = list_geometry_cases()
    if not geo_cases:
        st.info(
            "No geometry reports found under `outputs/`. Generate one with:\n\n"
            "```\npy -3.12 scripts/geometry_report.py --case P001 --data-root data\n```"
        )
        return

    default_idx = geo_cases.index("P065") if "P065" in geo_cases else 0
    case_id = st.selectbox("Case", geo_cases, index=default_idx)
    report = load_geometry_report(case_id)
    if report is None:
        st.error(f"Could not read geometry report for {case_id}.")
        return

    left, right = report["left"], report["right"]
    st.caption(f"Segmentation source: **{report.get('mask_source', 'labels')}**")

    # Headline metrics row
    ratio = report.get("mca_ratio")
    m1, m2, m3 = st.columns(3)
    with m1:
        if left.get("present"):
            st.metric("Left MCA", f"{left['mca_mm2']:.0f} mm²",
                      f"{left['volume_ml']:.1f} mL volume", delta_color="off")
    with m2:
        if right.get("present"):
            st.metric("Right MCA", f"{right['mca_mm2']:.0f} mm²",
                      f"{right['volume_ml']:.1f} mL volume", delta_color="off")
    with m3:
        if ratio is not None and not (isinstance(ratio, float) and np.isnan(ratio)):
            more = report.get("more_obstructed_side", "unknown")
            if ratio >= 0.85:
                verdict, color = "roughly symmetric", "normal"
            elif ratio >= 0.6:
                verdict, color = f"{more} side narrower", "off"
            else:
                verdict, color = f"{more} side obstructed", "inverse"
            st.metric("L/R MCA ratio", f"{ratio:.2f}", verdict, delta_color=color)
        else:
            st.metric("L/R MCA ratio", "n/a")

    tab_chart, tab_ct, tab_3d = st.tabs(
        ["📉 Constriction charts", "🩻 CT slices", "🌐 3D airway"])

    with tab_chart:
        st.plotly_chart(_area_profile_fig(report), use_container_width=True)
        st.caption(
            "Cross-sectional area along each passage from the anterior naris. "
            "The ▼ marks each side's minimal cross-sectional area (MCA) — the "
            "constriction. A large L/R gap is the obstruction signal.")

    with tab_ct:
        data = _load_ct_label(case_id)
        if data is None or data["label"] is None:
            st.info("CT/label not found for this case (needs data/images + data/labels).")
        else:
            ct, label = data["ct"], data["label"]
            nz, ny, nx = ct.shape
            dz, dy, dx = _airway_center(label)
            s1, s2, s3 = st.columns(3)
            cz = s1.slider("Axial (z)", 0, nz - 1, dz, key=f"cz_{case_id}")
            cy = s2.slider("Coronal (y)", 0, ny - 1, dy, key=f"cy_{case_id}")
            cx = s3.slider("Sagittal (x)", 0, nx - 1, dx, key=f"cx_{case_id}")
            st.plotly_chart(_triplanar_ct_fig(ct, label, cz, cy, cx), use_container_width=True)
            st.caption("CT windowed to soft tissue; left cavity blue, right cavity red. "
                       "Drag the sliders to scroll through slices.")

    with tab_3d:
        data = _load_ct_label(case_id)
        if data is None or data["label"] is None:
            st.info("CT/label not found for this case.")
        else:
            traces = _airway_3d_traces(case_id, data["spacing_xyz"])
            if not traces:
                st.info("Could not build 3D surfaces for this case.")
            else:
                fig = go.Figure(data=traces)
                fig.update_layout(
                    height=560, margin=dict(l=0, r=0, t=10, b=0),
                    scene=dict(aspectmode="data",
                               xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
                    legend=dict(orientation="h", yanchor="bottom", y=0.0),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Rotatable nasal airway — drag to orbit. Left blue, right red, "
                           "nasopharynx green.")

    with st.expander("Method & caveats"):
        st.markdown(
            "- Area = actual lumen voxel count per coronal slice × pixel area "
            "(faithful for the slit-shaped nasal valve, unlike a π·r² disk).\n"
            "- Profiles are typically unimodal, so the MCA sits at an end of the "
            "airway body (end narrowing, not a focal internal stenosis).\n"
            "- The **L/R MCA ratio is the most robust output**; on near-symmetric "
            "cases the 'more obstructed side' can flip within noise.\n"
            "- See `docs/stage2_geometry_metrics.md`."
        )
        st.json({k: v for k, v in report.items() if k not in ("left", "right")})


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _slice_fig(
    speed: np.ndarray,
    airway: np.ndarray,
    iz: int,
    iy: int,
    ix: int,
    spacing: np.ndarray,
    vmax: float,
) -> go.Figure:
    """Tri-planar speed maps (axial / coronal / sagittal)."""
    # speed shape (z,y,x)
    axial = np.where(airway[iz], speed[iz], np.nan)
    coronal = np.where(airway[:, iy, :], speed[:, iy, :], np.nan)
    sagittal = np.where(airway[:, :, ix], speed[:, :, ix], np.nan)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Axial  z={iz}",
            f"Coronal  y={iy}",
            f"Sagittal  x={ix}",
        ),
        horizontal_spacing=0.06,
    )

    common = dict(
        colorscale="Turbo",
        zmin=0,
        zmax=max(vmax, 1e-9),
        colorbar=dict(title="|u| m/s", x=1.02),
    )

    fig.add_trace(
        go.Heatmap(z=axial, **common, showscale=True),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=coronal, **{**common, "showscale": False}),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=sagittal, **{**common, "showscale": False}),
        row=1,
        col=3,
    )

    # Crosshairs
    for col, (h, v) in enumerate(
        [(iy, ix), (iz, ix), (iz, iy)], start=1
    ):
        fig.add_hline(y=h, line_width=1, line_color="white", opacity=0.5, row=1, col=col)
        fig.add_vline(x=v, line_width=1, line_color="white", opacity=0.5, row=1, col=col)

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    fig.update_xaxes(showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showticklabels=False)
    # Match anatomical feel: axial origin lower
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    return fig


def _load_mesh_decimated(stl_path: str, target_faces: int = 14000):
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except Exception:
            idx = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
    return mesh


def _mesh_wireframe_trace(mesh, max_edges: int = 8000, color: str = "#d8e6f0") -> go.Scatter3d | None:
    """Dark-on-light edge overlay so the cavity silhouette reads clearly."""
    try:
        edges = mesh.edges_unique
    except Exception:
        return None
    if edges is None or len(edges) == 0:
        return None
    if len(edges) > max_edges:
        idx = np.linspace(0, len(edges) - 1, max_edges, dtype=int)
        edges = edges[idx]
    v = mesh.vertices
    # NaN breaks create separate segments
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [v[a, 0], v[b, 0], None]
        ys += [v[a, 1], v[b, 1], None]
        zs += [v[a, 2], v[b, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=1.5),
        name="Cavity edges",
        hoverinfo="skip",
        opacity=0.55,
    )


def _add_surface_mesh(
    fig: go.Figure,
    mesh,
    *,
    name: str,
    color: str,
    opacity: float,
    show_wireframe: bool,
    edge_color: str,
    wireframe_max_edges: int = 6000,
    is_skin: bool = False,
) -> None:
    v = mesh.vertices
    f = mesh.faces
    # Solid surface (not a point cloud): Mesh3d with lighting.
    # Skin gets slightly stronger lighting so the head silhouette reads clearly.
    if is_skin:
        lighting = dict(
            ambient=0.42,
            diffuse=0.95,
            specular=0.55,
            roughness=0.35,
            fresnel=0.25,
        )
        lightpos = dict(x=120, y=180, z=280)
    else:
        lighting = dict(
            ambient=0.55,
            diffuse=0.85,
            specular=0.35,
            roughness=0.45,
            fresnel=0.15,
        )
        lightpos = dict(x=80, y=120, z=200)
    mesh_kwargs = dict(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=float(np.clip(opacity, 0.02, 1.0)),
        name=name,
        flatshading=False,
        lighting=lighting,
        lightposition=lightpos,
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_trace(go.Mesh3d(**mesh_kwargs))
    if show_wireframe:
        wf = _mesh_wireframe_trace(mesh, max_edges=wireframe_max_edges, color=edge_color)
        if wf is not None:
            wf.name = f"{name} edges"
            wf.opacity = 0.75 if is_skin else 0.55
            fig.add_trace(wf)


def _sample_speed_along_line(
    line: np.ndarray,
    speed: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor sample of |u| (m/s) at each pathline vertex."""
    ox, oy, oz = origin
    sx, sy, sz = spacing
    nz, ny, nx = speed.shape
    ix = np.clip(np.rint((line[:, 0] - ox) / sx).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((line[:, 1] - oy) / sy).astype(int), 0, ny - 1)
    iz = np.clip(np.rint((line[:, 2] - oz) / sz).astype(int), 0, nz - 1)
    return speed[iz, iy, ix].astype(float)


def _fig_3d(
    mesh,
    head_mesh,
    skin_mesh,
    bone_mesh,
    streamlines: list,
    streamline_speeds: list | None,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    speed: np.ndarray,
    airway: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    show_head: bool,
    head_opacity: float,
    show_skin: bool,
    skin_opacity: float,
    show_bone: bool,
    bone_opacity: float,
    show_mesh: bool,
    mesh_opacity: float,
    show_wireframe: bool,
    show_streamlines: bool,
    streamline_width: float,
    streamline_opacity: float,
    max_pathlines: int,
    show_vectors: bool,
    vector_stride: int,
    max_vector_speed: float,
    ports: list[dict],
    bg_mode: str = "dark",
    max_vectors: int = 6000,
    centerline_mm: list | None = None,
    centerline_left_mm: list | None = None,
    centerline_right_mm: list | None = None,
    show_centerlines: bool = False,
    left_mesh=None,
    right_mesh=None,
    septum_mesh=None,
    mucosa_mesh=None,
    show_left: bool = False,
    show_right: bool = False,
    show_septum: bool = False,
    show_mucosa: bool = False,
    left_opacity: float = 0.5,
    right_opacity: float = 0.5,
    septum_opacity: float = 0.9,
    mucosa_opacity: float = 0.35,
    restriction_pts: np.ndarray | None = None,
    show_restriction: bool = True,
    animate_pathlines: bool = False,
    n_anim_frames: int = 24,
    frontal_path_mm: list | None = None,
    frontal_path_left_mm: list | None = None,
    frontal_path_right_mm: list | None = None,
    show_frontal_path: bool = False,
    removal_pts: np.ndarray | None = None,
    show_removal: bool = False,
    removal_inferior: np.ndarray | None = None,
    removal_middle: np.ndarray | None = None,
    removal_septum: np.ndarray | None = None,
    show_removal_inferior: bool = True,
    show_removal_middle: bool = True,
    show_removal_septum: bool = True,
    frontal_mesh=None,
    sphenoid_mesh=None,
    max_l_mesh=None,
    max_r_mesh=None,
    show_frontal_sinus: bool = False,
    show_sphenoid: bool = False,
    show_maxillary: bool = False,
    sinus_opacity: float = 0.35,
) -> go.Figure:
    fig = go.Figure()
    dark = bg_mode == "dark"
    scene_bg = "#1a1f2b" if dark else "#f4f6f8"
    paper_bg = "#0e1117" if dark else "#ffffff"
    font_c = "#fafafa" if dark else "#111111"
    mesh_color = "#5ec8ff"  # airway cyan
    head_color = "#c4a484"  # soft tissue volume
    skin_color = "#e8b896"  # outer skin surface
    bone_color = "#f0e6d8"  # bone
    edge_color = "#e8f1f8" if dark else "#1a3a52"
    head_edge = "#8b7355" if dark else "#5c4033"
    stream_width = float(streamline_width)

    # --- Outer skin surface (preferred visual for head shape) ---
    if show_skin and skin_mesh is not None:
        _add_surface_mesh(
            fig,
            skin_mesh,
            name="Skin surface",
            color=skin_color,
            opacity=skin_opacity,
            show_wireframe=show_wireframe,
            edge_color="#3d2314" if not dark else "#ffd7b0",
            wireframe_max_edges=18000,
            is_skin=True,
        )
    elif show_head and head_mesh is not None:
        _add_surface_mesh(
            fig,
            head_mesh,
            name="Soft tissue / head",
            color=head_color,
            opacity=head_opacity,
            show_wireframe=False,
            edge_color=head_edge,
        )

    # --- Bone ---
    if show_bone and bone_mesh is not None:
        _add_surface_mesh(
            fig,
            bone_mesh,
            name="Bone",
            color=bone_color,
            opacity=bone_opacity,
            show_wireframe=False,
            edge_color="#dddddd",
        )

    # --- Airway cavity (combined air space) ---
    if show_mesh and mesh is not None and not (show_left or show_right):
        _add_surface_mesh(
            fig,
            mesh,
            name="Air space",
            color=mesh_color,
            opacity=mesh_opacity,
            show_wireframe=show_wireframe,
            edge_color=edge_color,
            wireframe_max_edges=5000,
        )

    # --- L/R cavities (CT-native; septum is the gap/wall between them) ---
    if show_left and left_mesh is not None:
        _add_surface_mesh(
            fig,
            left_mesh,
            name="Left nasal cavity",
            color="#4fc3f7",
            opacity=left_opacity,
            show_wireframe=False,
            edge_color=edge_color,
        )
    if show_right and right_mesh is not None:
        _add_surface_mesh(
            fig,
            right_mesh,
            name="Right nasal cavity",
            color="#81d4fa",
            opacity=right_opacity,
            show_wireframe=False,
            edge_color=edge_color,
        )
    if show_septum and septum_mesh is not None:
        _add_surface_mesh(
            fig,
            septum_mesh,
            name="Nasal septum",
            color="#ff8a65",
            opacity=septum_opacity,
            show_wireframe=False,
            edge_color="#bf360c",
        )
    if show_mucosa and mucosa_mesh is not None:
        _add_surface_mesh(
            fig,
            mucosa_mesh,
            name="Mucosa / walls",
            color="#ce93d8",
            opacity=mucosa_opacity,
            show_wireframe=False,
            edge_color="#6a1b9a",
        )

    # --- Paranasal sinuses (labeled) ---
    if show_frontal_sinus and frontal_mesh is not None:
        _add_surface_mesh(
            fig,
            frontal_mesh,
            name="Frontal sinus",
            color="#ffb74d",
            opacity=sinus_opacity,
            show_wireframe=False,
            edge_color="#e65100",
        )
    if show_sphenoid and sphenoid_mesh is not None:
        _add_surface_mesh(
            fig,
            sphenoid_mesh,
            name="Sphenoid sinus",
            color="#4db6ac",
            opacity=sinus_opacity,
            show_wireframe=False,
            edge_color="#00695c",
        )
    if show_maxillary:
        if max_l_mesh is not None:
            _add_surface_mesh(
                fig,
                max_l_mesh,
                name="Maxillary sinus (L)",
                color="#64b5f6",
                opacity=sinus_opacity,
                show_wireframe=False,
                edge_color="#1565c0",
            )
        if max_r_mesh is not None:
            _add_surface_mesh(
                fig,
                max_r_mesh,
                name="Maxillary sinus (R)",
                color="#4fc3f7",
                opacity=sinus_opacity,
                show_wireframe=False,
                edge_color="#0277bd",
            )

    # --- Wispy turbulent pathlines (no velocity colorbar on 3D) ---
    path_arrays: list[np.ndarray] = []
    path_speeds: list[np.ndarray] = []
    if show_streamlines and streamlines:
        n_avail = len(streamlines)
        n_show = min(int(max_pathlines), n_avail)
        if n_show < n_avail:
            idx = np.linspace(0, n_avail - 1, n_show, dtype=int)
        else:
            idx = np.arange(n_avail)
        speed_lists = streamline_speeds or []
        cmax_u = max(float(max_vector_speed), 1e-6)
        base_op = float(np.clip(streamline_opacity, 0.12, 1.0))
        n_fade_seg = 7
        # Soft cyan/white wisps (no blue→red scale)
        wisp_color = "#7ecbff" if dark else "#1565c0"
        legend_done = False
        for k, li in enumerate(idx):
            full = np.asarray(streamlines[li], dtype=float)
            if full.ndim != 2 or full.shape[0] < 2 or full.shape[1] < 3:
                continue
            if li < len(speed_lists) and speed_lists[li]:
                sp_full = np.asarray(speed_lists[li], dtype=float)
                if len(sp_full) != len(full):
                    sp_full = _sample_speed_along_line(full, speed, spacing, origin)
            else:
                sp_full = _sample_speed_along_line(full, speed, spacing, origin)
            if len(full) > 300:
                step = max(1, len(full) // 240)
                arr = full[::step]
                sp = sp_full[::step]
            else:
                arr = full
                sp = sp_full
            if len(sp) != len(arr):
                n = min(len(sp), len(arr))
                arr, sp = arr[:n], sp[:n]
            if len(arr) < 4:
                continue
            path_arrays.append(arr)
            path_speeds.append(sp)
            mean_sp = float(np.mean(sp)) if len(sp) else 0.0
            w_scale = 0.75 + 0.55 * min(1.5, mean_sp / max(cmax_u * 0.35, 1e-3))
            n_pts = len(arr)
            n_seg = min(n_fade_seg, max(2, n_pts // 8))
            edges = np.linspace(0, n_pts - 1, n_seg + 1, dtype=int)
            for si in range(n_seg):
                i0, i1 = int(edges[si]), int(edges[si + 1])
                if i1 <= i0:
                    continue
                i0s = max(0, i0 - (1 if si > 0 else 0))
                seg = arr[i0s : i1 + 1]
                t_mid = (si + 0.5) / n_seg
                op = base_op * (1.0 - 0.88 * (t_mid ** 1.05))
                op = float(np.clip(op, 0.04, 1.0))
                w = stream_width * w_scale * (1.0 - 0.78 * (t_mid ** 0.9))
                w = float(np.clip(w, 0.5, stream_width * 1.5))
                show_leg = (not legend_done) and si == 0
                fig.add_trace(
                    go.Scatter3d(
                        x=seg[:, 0],
                        y=seg[:, 1],
                        z=seg[:, 2],
                        mode="lines",
                        line=dict(width=w, color=wisp_color),
                        name="Turbulent pathlines" if show_leg else None,
                        showlegend=show_leg,
                        hoverinfo="skip",
                        opacity=op,
                    )
                )
                if show_leg:
                    legend_done = True

    # --- Reddish-pink constriction (hex + opacity; rgba strings can wash to white) ---
    def _add_pink_zone(pts, name: str, color: str = "#e53935") -> None:
        if pts is None or len(pts) == 0:
            return
        rp = np.asarray(pts, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=rp[:, 0],
                y=rp[:, 1],
                z=rp[:, 2],
                mode="markers",
                marker=dict(
                    size=4.0,
                    color=color,  # solid hex — Plotly 3D often ignores rgba fill
                    opacity=0.28,  # more transparent
                    line=dict(width=0),
                    symbol="circle",
                ),
                name=name,
                hoverinfo="skip",
            )
        )

    any_zone = False
    if show_removal_inferior and removal_inferior is not None and len(removal_inferior) > 0:
        _add_pink_zone(
            removal_inferior,
            "Areas to remove: inferior turbinate",
            "#ef5350",  # light red
        )
        any_zone = True
    if show_removal_middle and removal_middle is not None and len(removal_middle) > 0:
        _add_pink_zone(
            removal_middle,
            "Areas to remove: middle turbinate",
            "#e91e63",  # pink-red
        )
        any_zone = True
    if show_removal_septum and removal_septum is not None and len(removal_septum) > 0:
        _add_pink_zone(
            removal_septum,
            "Areas to remove: septum (distal–medial)",
            "#c62828",  # deeper red
        )
        any_zone = True
    if (
        show_removal
        and not any_zone
        and removal_pts is not None
        and len(removal_pts) > 0
    ):
        _add_pink_zone(removal_pts, "Areas to remove (high |u|)", "#e53935")
    elif (
        show_restriction
        and not any_zone
        and restriction_pts is not None
        and len(restriction_pts) > 0
    ):
        _add_pink_zone(restriction_pts, "Constriction", "#e53935")

    # --- Purple: dual instrument paths L/R naris → ipsilateral frontal ---
    if show_frontal_path:
        dual_paths = []
        if frontal_path_left_mm is not None and len(frontal_path_left_mm) >= 2:
            dual_paths.append(("L naris → L frontal", frontal_path_left_mm, "#ab47bc"))
        if frontal_path_right_mm is not None and len(frontal_path_right_mm) >= 2:
            dual_paths.append(("R naris → R frontal", frontal_path_right_mm, "#8e24aa"))
        # Fallback single path
        if not dual_paths and frontal_path_mm is not None and len(frontal_path_mm) >= 2:
            dual_paths.append(("Naris → frontal", frontal_path_mm, "#9c27b0"))
        for pi, (name, path, color) in enumerate(dual_paths):
            fp = np.asarray(path, dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=fp[:, 0],
                    y=fp[:, 1],
                    z=fp[:, 2],
                    mode="lines",
                    line=dict(color=color, width=9),
                    name=name,
                    hoverinfo="skip",
                    opacity=0.95,
                )
            )
            side_tag = "L" if "L naris" in name else ("R" if "R naris" in name else "")
            fig.add_trace(
                go.Scatter3d(
                    x=[fp[0, 0], fp[-1, 0]],
                    y=[fp[0, 1], fp[-1, 1]],
                    z=[fp[0, 2], fp[-1, 2]],
                    mode="markers+text",
                    marker=dict(
                        size=[4, 5],
                        color=["#ce93d8", "#6a1b9a"],
                        symbol=["circle", "diamond"],
                        line=dict(width=0.5, color="#333333"),
                    ),
                    text=[
                        f"{side_tag} naris" if side_tag else "Naris",
                        f"{side_tag} frontal" if side_tag else "Frontal",
                    ],
                    textposition="top center",
                    textfont=dict(size=9, color="#6a1b9a"),
                    name=f"{name} ends",
                    showlegend=False,
                )
            )

    # --- Velocity cones (optional, denser cones) ---
    if show_vectors:
        zz, yy, xx = np.where(airway)
        step = max(int(vector_stride), 1)
        # 3D striding for more uniform density
        sel = (
            (zz % step == 0)
            & (yy % step == 0)
            & (xx % step == 0)
        )
        zz, yy, xx = zz[sel], yy[sel], xx[sel]
        # Prefer higher-speed samples when over max
        if len(zz) > max_vectors:
            sp_all = speed[zz, yy, xx]
            # mix top-speed and random coverage
            n_top = max_vectors // 2
            n_rest = max_vectors - n_top
            top_idx = np.argpartition(sp_all, -n_top)[-n_top:]
            rest_pool = np.setdiff1d(np.arange(len(zz)), top_idx, assume_unique=False)
            rng = np.random.default_rng(0)
            if len(rest_pool) > n_rest:
                rest_idx = rng.choice(rest_pool, size=n_rest, replace=False)
            else:
                rest_idx = rest_pool
            keep = np.concatenate([top_idx, rest_idx])
            zz, yy, xx = zz[keep], yy[keep], xx[keep]

        sx, sy, sz = spacing
        ox, oy, oz = origin
        px = ox + xx * sx
        py = oy + yy * sy
        pz = oz + zz * sz
        vx = ux[zz, yy, xx]
        vy = uy[zz, yy, xx]
        vz = uz[zz, yy, xx]
        scale = 2.2 / max(max_vector_speed, 1e-9)
        fig.add_trace(
            go.Cone(
                x=px,
                y=py,
                z=pz,
                u=vx,
                v=vy,
                w=vz,
                colorscale="Turbo",
                cmin=0,
                cmax=max_vector_speed,
                sizemode="absolute",
                sizeref=max(scale * 1.1, 0.15),
                anchor="tail",
                name=f"Velocity ({len(px)} cones)",
                showscale=False,
                opacity=0.85,
            )
        )

    # Optional dual centerlines (off by default — low visual value vs pathlines)
    if show_centerlines:
        dual_drawn = False
        for name, cl_pts, width in (
            ("Centerline left naris → trachea", centerline_left_mm, 5),
            ("Centerline right naris → trachea", centerline_right_mm, 5),
        ):
            if cl_pts is not None and len(cl_pts) >= 2:
                cl = np.asarray(cl_pts, dtype=float)
                fig.add_trace(
                    go.Scatter3d(
                        x=cl[:, 0],
                        y=cl[:, 1],
                        z=cl[:, 2],
                        mode="lines",
                        line=dict(color="rgba(255,0,170,0.45)", width=width),
                        name=name,
                        hoverinfo="skip",
                    )
                )
                dual_drawn = True
        if not dual_drawn and centerline_mm is not None and len(centerline_mm) >= 2:
            cl = np.asarray(centerline_mm, dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=cl[:, 0],
                    y=cl[:, 1],
                    z=cl[:, 2],
                    mode="lines",
                    line=dict(color="rgba(255,0,170,0.5)", width=6),
                    name="Centerline (nares → trachea)",
                    hoverinfo="skip",
                )
            )

    # Port markers — tip-accurate nares + trachea only (no per-path clutter)
    for port in ports:
        c = port.get("center_mm", [0, 0, 0])
        is_inlet = port.get("role") == "inlet"
        name = str(port.get("name", "port"))
        if is_inlet:
            color = "#00c853"
            short = "L naris" if "left" in name.lower() else (
                "R naris" if "right" in name.lower() else "Naris"
            )
            label = short
            size = 5
            symbol = "circle"
        else:
            color = "#ff1744"
            label = "Trachea"
            size = 6
            symbol = "diamond"
        fig.add_trace(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=0.5, color="#333333"),
                    opacity=0.9,
                ),
                text=[label],
                textposition="top center",
                textfont=dict(size=9, color=color, family="Arial"),
                name=name,
            )
        )

    # Animated particles riding pathlines (optional)
    frames: list = []
    if animate_pathlines and path_arrays:
        particle_trace_idx = len(fig.data)
        xs0 = [float(a[0, 0]) for a in path_arrays]
        ys0 = [float(a[0, 1]) for a in path_arrays]
        zs0 = [float(a[0, 2]) for a in path_arrays]
        cs0 = [float(s[0]) for s in path_speeds]
        cmax_u = max(float(max_vector_speed), 1e-6)
        fig.add_trace(
            go.Scatter3d(
                x=xs0,
                y=ys0,
                z=zs0,
                mode="markers",
                marker=dict(
                    size=5,
                    color=cs0,
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=cmax_u,
                    line=dict(width=0),
                ),
                name="Flow particles",
                showlegend=True,
            )
        )
        n_f = max(8, int(n_anim_frames))
        for fi in range(n_f):
            frac = fi / max(n_f - 1, 1)
            xs, ys, zs, cs = [], [], [], []
            for arr, sp in zip(path_arrays, path_speeds):
                j = int(frac * (len(arr) - 1))
                xs.append(float(arr[j, 0]))
                ys.append(float(arr[j, 1]))
                zs.append(float(arr[j, 2]))
                cs.append(float(sp[min(j, len(sp) - 1)]))
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=cs,
                                colorscale="Turbo",
                                cmin=0.0,
                                cmax=cmax_u,
                                line=dict(width=0),
                            ),
                        )
                    ],
                    name=str(fi),
                    traces=[particle_trace_idx],
                )
            )
        fig.frames = frames

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
            bgcolor=scene_bg,
            xaxis=dict(backgroundcolor=scene_bg, gridcolor="#444" if dark else "#ccc", showbackground=True),
            yaxis=dict(backgroundcolor=scene_bg, gridcolor="#444" if dark else "#ccc", showbackground=True),
            zaxis=dict(backgroundcolor=scene_bg, gridcolor="#444" if dark else "#ccc", showbackground=True),
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
        ),
        paper_bgcolor=paper_bg,
        font_color=font_c,
        legend=dict(bgcolor="rgba(0,0,0,0.3)" if dark else "rgba(255,255,255,0.7)"),
    )
    if frames:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.05,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=80, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ]
        )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Sinus_CFD Viewer",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    mode = st.sidebar.radio(
        "View",
        ("Geometry (MVP)", "Airflow demo"),
        help="Geometry: per-side nasal airway metrics + 3D/CT (Stage 2). Airflow: Visible Human flow demo.",
    )
    if mode == "Geometry (MVP)":
        render_geometry_report()
        return

    st.title("Sinus_CFD — Airflow Viewer")
    st.caption(f"`{APP_VERSION}` · {APP_VERSION_LABEL}")

    cases = list_cases()
    if not cases:
        st.error(
            "No flow fields found under `outputs/`. "
            "Run process_whole_head / compute_flow / regenerate_curvy_pathlines."
        )
        return

    case_id = "VisibleHuman_Head" if "VisibleHuman_Head" in cases else cases[0]

    # Fixed display defaults
    show_skin = True
    skin_opacity = 0.32
    show_head = False
    head_opacity = 0.25
    show_bone = False
    bone_opacity = 0.45
    show_mesh = False
    mesh_opacity = 0.40
    show_left = True
    show_right = True
    show_septum = False
    show_mucosa = False
    left_opacity = 0.38
    right_opacity = 0.38
    septum_opacity = 0.55
    show_wireframe = True
    show_streamlines = True
    streamline_width = 2.4
    streamline_opacity = 0.52
    max_pathlines = 320
    animate_pathlines = False
    show_restriction = False
    show_centerlines = False
    show_frontal_sinus = True
    show_sphenoid = False
    show_maxillary = False
    sinus_opacity = 0.30
    show_vectors = False
    vector_stride = 4
    max_vectors = 2000
    bg_mode = "light"

    with st.sidebar:
        st.markdown(f"**`{APP_VERSION}`**")
        st.caption("Surgical view toggles")
        show_frontal_path = st.checkbox(
            "Paths to frontal sinuses (purple)",
            value=True,
            help="L/R naris → ipsilateral frontal instrument corridors.",
        )
        st.markdown("**Areas to remove (pink)**")
        show_removal_inferior = st.checkbox(
            "Inferior turbinates (lateral / maxillary)",
            value=True,
            help="High-|u| along inferior turbinate / lateral nasal corridor.",
        )
        show_removal_middle = st.checkbox(
            "Middle turbinates (split nasal airflow)",
            value=True,
            help="High-|u| at middle turbinate level that partitions flow.",
        )
        show_removal_septum = st.checkbox(
            "Septum (distal & medial)",
            value=True,
            help="High-|u| along distal–medial septum.",
        )
        show_removal = (
            show_removal_inferior or show_removal_middle or show_removal_septum
        )
        if st.button("Reload data"):
            st.cache_data.clear()
            st.rerun()

    fp = case_data_fingerprint(case_id)
    data = load_case(case_id, fp)
    if "error" in data:
        st.error(data["error"])
        return

    speed = data["speed"]
    airway = data["airway"]
    nz, ny, nx = speed.shape
    meta = data.get("meta", {})
    bc = data.get("bc", {})
    stats = data.get("stats", {})

    # Data version panel (collapsed)
    with st.expander("Loaded data version", expanded=False):
        st.markdown(
            f"- **App:** `{APP_VERSION}`\n"
            f"- **Case:** `{case_id}`\n"
            f"- **BC file time:** `{data.get('bc_mtime')}`\n"
            f"- **Stats file time:** `{data.get('stats_mtime')}`\n"
            f"- **Fingerprint:** `{fp[-80:]}`"
        )
        # Expected for current pipeline
        expected_methods = []
        for p in bc.get("ports", []):
            if p.get("role") == "inlet":
                expected_methods.append(
                    f"**{p.get('name')}** · method=`{p.get('method')}` · "
                    f"xyz_mm=`{[round(x,1) for x in p.get('center_mm', [])]}`"
                )
        if expected_methods:
            st.markdown("**Inlet ports currently loaded:**")
            for line in expected_methods:
                st.markdown(f"- {line}")
        else:
            st.warning("No inlet ports in loaded BC JSON.")

        # Guardrail: tip-accurate CT naris openings
        methods = [p.get("method") for p in bc.get("ports", []) if p.get("role") == "inlet"]
        tip_ok = {
            "skin_tip_vestibule",
            "ct_naris_opening_air",
            "ct_naris_opening_tip",
            "ct_naris_opening",
            "edge_nose_tip_skin_naris",
        }
        if methods and all(m in tip_ok for m in methods):
            if all(m == "skin_tip_vestibule" for m in methods):
                st.success(
                    "Nares at **skin nose tip** with **open vestibules** painted into "
                    "each cavity (CT often seals this region). "
                    "Pathlines ~50% L / ~50% R → trachea."
                )
            elif all(m == "ct_naris_opening_air" for m in methods):
                st.info(
                    "Nares on CT opening∩air. For tip openings run: "
                    "`py -3.12 scripts/extend_nasal_to_tip.py`"
                )
            else:
                st.info(f"Nares methods: `{methods}`.")
        elif methods:
            st.warning(
                f"Unexpected naris methods `{methods}`. "
                "Click **Clear cache & reload data**, or re-run pathline regen."
            )

        notes = stats.get("notes") or []
        if notes:
            st.markdown("**Pipeline notes (from stats):**")
            for n in notes[:8]:
                st.caption(f"• {n}")

        qc1, qc2 = st.columns(2)
        if data.get("face_qc_path"):
            with qc1:
                st.markdown("**Face QC (nares should be at nose tip, not orbits)**")
                st.image(data["face_qc_path"], use_container_width=True)
        if data.get("preview_path"):
            with qc2:
                st.markdown("**Tri-planar QC**")
                st.image(data["preview_path"], use_container_width=True)

        passage = data.get("passage") or {}
        pm = passage.get("metrics") or {}
        if pm:
            st.markdown("**Nasal passage domain (walls + open ports)**")
            st.markdown(
                f"- Lumen volume: **{pm.get('lumen_volume_ml', 0):.1f} mL**\n"
                f"- Centerline length: **{pm.get('centerline_length_mm', 0):.1f} mm**\n"
                f"- Cross-section min/mean/max: "
                f"**{pm.get('min_cross_section_mm2', 0):.1f} / "
                f"{pm.get('mean_cross_section_mm2', 0):.1f} / "
                f"{pm.get('max_cross_section_mm2', 0):.1f} mm²**\n"
                f"- Wall voxels: `{pm.get('wall_voxels')}` · "
                f"inlet open: `{pm.get('inlet_open_voxels')}` · "
                f"outlet open: `{pm.get('outlet_open_voxels')}`"
            )
        n_sl = len(data.get("streamlines") or [])
        st.caption(
            f"**{n_sl} pathlines** (Turbo = |u|) · "
            "**Purple** = least-resistance naris→frontal · "
            "**Magenta** = areas to remove (narrow bottlenecks)."
        )
        surg = data.get("surgical") or {}
        if surg.get("path_metrics"):
            for pm_ in surg["path_metrics"][:2]:
                st.caption(
                    f"• `{pm_.get('name')}`: **{pm_.get('length_mm', 0):.1f} mm** · "
                    f"min r **{pm_.get('min_radius_mm', 0):.2f} mm**"
                )
        labels = (surg.get("sinus_anatomy") or {}).get("labels") or []
        if labels:
            st.markdown("**Sinus labels (CT air heuristics):**")
            for L in labels:
                st.caption(
                    f"• **{L.get('name')}**: {L.get('voxels')} vx · "
                    f"center mm `{[round(x,1) for x in (L.get('center_mm') or [])]}`"
                )
        if not surg:
            st.info(
                "No surgical guidance yet. Run: "
                "`py -3.12 scripts/compute_surgical_guidance.py --case VisibleHuman_Head`"
            )

    method = str(meta.get("method", "potential_flow"))
    is_openfoam = "openfoam" in method.lower()
    if is_openfoam:
        st.success(
            f"**Flow source: OpenFOAM simpleFoam** · time=`{meta.get('openfoam_time', '?')}` · "
            f"cells=`{meta.get('n_cells', '?')}` · mapped voxels=`{meta.get('n_mapped_voxels', '?')}`"
        )
        st.markdown(
            "**Inspiration pathlines** (Turbo = |u|): **nostrils → trachea**. "
            "**Purple** corridor = least-resistance **naris → frontal sinus** (instrument path). "
            "**Magenta/pink** = **areas to remove** (narrow bottlenecks along frontal access)."
        )
    else:
        st.warning(
            f"**Flow source: `{method}`** (not OpenFOAM). "
            "After a Docker run, import with: "
            "`py -3.12 scripts/import_openfoam_results.py --case VisibleHuman_Head`"
        )

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max |u|", f"{meta.get('max_speed_m_s', float(speed[airway].max() if airway.any() else 0)):.3f} m/s")
    c2.metric("Mean |u|", f"{meta.get('mean_speed_m_s', float(speed[airway].mean() if airway.any() else 0)):.3f} m/s")
    c3.metric("Target Q", f"{meta.get('target_flow_L_per_min', 18):.1f} L/min")
    c4.metric("App ver", APP_VERSION.split("-")[0])

    if meta.get("notes"):
        with st.expander("Method notes / caveats", expanded=is_openfoam):
            st.markdown(f"**Method:** `{method}`")
            for n in meta["notes"]:
                st.write(f"- {n}")
            if bc.get("outlet_is_proxy"):
                st.warning("Outlet is a nasopharynx proxy — NasalSeg FOV has no true trachea.")

    # ---- Tri-planar ----
    st.subheader("Tri-planar velocity")
    z0, y0, x0 = nz // 2, ny // 2, nx // 2
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        iz = st.slider("Axial (z index)", 0, nz - 1, z0, key="iz")
    with sc2:
        iy = st.slider("Coronal (y index)", 0, ny - 1, y0, key="iy")
    with sc3:
        ix = st.slider("Sagittal (x index)", 0, nx - 1, x0, key="ix")

    vmax = float(np.percentile(speed[airway], 99)) if airway.any() else 1.0
    fig2d = _slice_fig(speed, airway, iz, iy, ix, data["spacing"], vmax)
    st.plotly_chart(fig2d, use_container_width=True)

    # ---- 3D ----
    st.subheader("3D head + airway + airflow")
    mesh = None
    head_mesh = None
    skin_mesh = None
    bone_mesh = None
    if data.get("stl_path"):
        try:
            mesh = _load_mesh_decimated(data["stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Could not load airway STL: {exc}")
    if data.get("skin_stl_path"):
        try:
            # Keep more faces so skin reads as a continuous surface + wireframe
            skin_mesh = _load_mesh_decimated(data["skin_stl_path"], target_faces=40000)
        except Exception as exc:
            st.warning(f"Could not load skin STL: {exc}")
    if data.get("head_stl_path"):
        try:
            head_mesh = _load_mesh_decimated(data["head_stl_path"], target_faces=18000)
        except Exception as exc:
            st.warning(f"Could not load head STL: {exc}")
    if show_skin and skin_mesh is None and show_head is False:
        st.info("No skin mesh for this case. Enable soft-tissue solid or run process_whole_head.")
    if data.get("bone_stl_path"):
        try:
            bone_mesh = _load_mesh_decimated(data["bone_stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Could not load bone STL: {exc}")

    left_mesh = right_mesh = septum_mesh = mucosa_mesh = None
    if data.get("left_cavity_stl_path"):
        try:
            left_mesh = _load_mesh_decimated(data["left_cavity_stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Left cavity STL: {exc}")
    if data.get("right_cavity_stl_path"):
        try:
            right_mesh = _load_mesh_decimated(data["right_cavity_stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Right cavity STL: {exc}")
    if data.get("septum_stl_path"):
        try:
            septum_mesh = _load_mesh_decimated(data["septum_stl_path"], target_faces=10000)
        except Exception as exc:
            st.warning(f"Septum STL: {exc}")
    if data.get("mucosa_stl_path"):
        try:
            mucosa_mesh = _load_mesh_decimated(data["mucosa_stl_path"], target_faces=18000)
        except Exception as exc:
            st.warning(f"Mucosa STL: {exc}")

    frontal_mesh = sphenoid_mesh = max_l_mesh = max_r_mesh = None
    if data.get("frontal_stl"):
        try:
            frontal_mesh = _load_mesh_decimated(data["frontal_stl"], target_faces=10000)
        except Exception as exc:
            st.warning(f"Frontal sinus STL: {exc}")
    if data.get("sphenoid_stl"):
        try:
            sphenoid_mesh = _load_mesh_decimated(data["sphenoid_stl"], target_faces=10000)
        except Exception as exc:
            st.warning(f"Sphenoid STL: {exc}")
    if data.get("maxillary_left_stl"):
        try:
            max_l_mesh = _load_mesh_decimated(data["maxillary_left_stl"], target_faces=8000)
        except Exception as exc:
            st.warning(f"Maxillary L STL: {exc}")
    if data.get("maxillary_right_stl"):
        try:
            max_r_mesh = _load_mesh_decimated(data["maxillary_right_stl"], target_faces=8000)
        except Exception as exc:
            st.warning(f"Maxillary R STL: {exc}")

    ct_nasal = data.get("ct_nasal") or {}
    if ct_nasal:
        st.info(
            f"**CT-native nasal model** (`{ct_nasal.get('method', '?')}`): "
            f"L={ct_nasal.get('left_voxels', '?')} · R={ct_nasal.get('right_voxels', '?')} · "
            f"septum={ct_nasal.get('septum_voxels', '?')} voxels · "
            f"naris openings={ct_nasal.get('naris_opening_voxels', '?')}"
        )
    elif show_septum or show_left or show_right:
        st.warning(
            "No CT L/R/septum meshes yet. Run: "
            "`py -3.12 scripts/refine_nasal_ct.py --case VisibleHuman_Head`"
        )

    streamlines = data["streamlines"] if show_streamlines else []
    streamline_speeds = data.get("streamline_speeds") or []
    ports = bc.get("ports", [])
    # Prefer tip-accurate centers from nares.json when present
    tip_by_name: dict[str, list[float]] = {}
    for npnt in (data.get("nares") or {}).get("naris_points") or []:
        nm = str(npnt.get("name") or "")
        if npnt.get("center_mm"):
            tip_by_name[nm] = [float(v) for v in npnt["center_mm"]]
    ports_lite = []
    for p in ports:
        center = p.get("center_mm")
        name = str(p.get("name") or "")
        if name in tip_by_name:
            center = tip_by_name[name]
        ports_lite.append(
            {
                "name": name,
                "role": p.get("role"),
                "center_mm": center,
                "method": p.get("method"),
            }
        )

    # Prefer dual open-paths when passage dual missing
    op = data.get("open_paths") or {}
    passage = data.get("passage") or {}
    cl_left = passage.get("centerline_left_mm") or op.get("centerline_left_mm")
    cl_right = passage.get("centerline_right_mm") or op.get("centerline_right_mm")
    cl_mid = passage.get("centerline_mm") or op.get("centerline_mid_mm")

    fig3d = _fig_3d(
        mesh=mesh,
        head_mesh=head_mesh,
        skin_mesh=skin_mesh,
        bone_mesh=bone_mesh,
        streamlines=streamlines if show_streamlines else [],
        streamline_speeds=streamline_speeds if show_streamlines else [],
        ux=data["ux"],
        uy=data["uy"],
        uz=data["uz"],
        speed=speed,
        airway=airway,
        spacing=data["spacing"],
        origin=data["origin"],
        show_head=show_head and head_mesh is not None,
        head_opacity=head_opacity,
        show_skin=show_skin and skin_mesh is not None,
        skin_opacity=skin_opacity,
        show_bone=show_bone and bone_mesh is not None,
        bone_opacity=bone_opacity,
        show_mesh=show_mesh,
        mesh_opacity=mesh_opacity,
        show_wireframe=show_wireframe,
        show_streamlines=show_streamlines,
        streamline_width=streamline_width,
        streamline_opacity=streamline_opacity,
        max_pathlines=max_pathlines,
        show_vectors=show_vectors,
        vector_stride=vector_stride,
        max_vector_speed=vmax,
        ports=ports_lite,
        bg_mode=bg_mode,
        max_vectors=max_vectors,
        centerline_mm=cl_mid,
        centerline_left_mm=cl_left,
        centerline_right_mm=cl_right,
        show_centerlines=show_centerlines,
        left_mesh=left_mesh,
        right_mesh=right_mesh,
        septum_mesh=septum_mesh,
        mucosa_mesh=mucosa_mesh,
        show_left=show_left and left_mesh is not None,
        show_right=show_right and right_mesh is not None,
        show_septum=show_septum and septum_mesh is not None,
        show_mucosa=show_mucosa and mucosa_mesh is not None,
        left_opacity=left_opacity,
        right_opacity=right_opacity,
        septum_opacity=septum_opacity,
        restriction_pts=data.get("restriction_pts"),
        show_restriction=show_restriction,
        animate_pathlines=animate_pathlines and show_streamlines,
        n_anim_frames=28,
        frontal_path_mm=(
            ((data.get("surgical") or {}).get("paths_mm") or {}).get(
                "naris_left_to_frontal"
            )
        ),
        frontal_path_left_mm=(
            ((data.get("surgical") or {}).get("paths_mm") or {}).get(
                "naris_left_to_frontal"
            )
        ),
        frontal_path_right_mm=(
            ((data.get("surgical") or {}).get("paths_mm") or {}).get(
                "naris_right_to_frontal"
            )
        ),
        show_frontal_path=show_frontal_path,
        removal_pts=data.get("removal_pts"),
        show_removal=show_removal,
        removal_inferior=data.get("removal_inferior_turbinate"),
        removal_middle=data.get("removal_middle_turbinate"),
        removal_septum=data.get("removal_septum"),
        show_removal_inferior=show_removal_inferior,
        show_removal_middle=show_removal_middle,
        show_removal_septum=show_removal_septum,
        frontal_mesh=frontal_mesh,
        sphenoid_mesh=sphenoid_mesh,
        max_l_mesh=max_l_mesh,
        max_r_mesh=max_r_mesh,
        show_frontal_sinus=show_frontal_sinus and frontal_mesh is not None,
        show_sphenoid=show_sphenoid and sphenoid_mesh is not None,
        show_maxillary=show_maxillary,
        sinus_opacity=sinus_opacity,
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption(
        "Wispy turbulent pathlines (seeded through airspace, toward trachea). "
        "**Purple** = naris → frontal instrument paths. "
        "**Pink** = high-|u| tissue to consider removing (IT / MT / septum). "
        "Use the left panel to toggle layers."
    )

    # ---- Identified removal zones + treatment recommendations ----
    surg = data.get("surgical") or {}
    rz = surg.get("removal_zones") or {}
    zones = rz.get("zones") or []
    treatments = rz.get("treatments") or []

    st.subheader("Identified areas to remove")
    if zones:
        cols = st.columns(3)
        for i, z in enumerate(zones):
            with cols[i % 3]:
                sev = z.get("severity", "none")
                st.markdown(f"**{z.get('label', z.get('name'))}**")
                st.caption(
                    f"Severity: **{sev}** · voxels `{z.get('voxels', 0)}` · "
                    f"mean |u| **{z.get('mean_speed_m_s', 0):.2f}** m/s · "
                    f"max **{z.get('max_speed_m_s', 0):.2f}** m/s"
                )
                cmm = z.get("center_mm") or []
                if cmm:
                    st.caption(f"Center ≈ `[{cmm[0]:.0f}, {cmm[1]:.0f}, {cmm[2]:.0f}]` mm")
                st.caption(z.get("notes") or "")
    else:
        st.info(
            "No zone map yet. Run: "
            "`py -3.12 scripts/compute_surgical_guidance.py --case VisibleHuman_Head`"
        )

    st.subheader("Recommended treatment options")
    st.markdown(
        "Prioritized to **increase airflow** with **minimal intervention** first; "
        "sinus-drainage options for CRS when indicated."
    )
    if treatments:
        recs = [t for t in treatments if t.get("recommended")]
        others = [t for t in treatments if not t.get("recommended")]
        if recs:
            st.markdown("#### Prefer first")
            for t in recs:
                st.markdown(
                    f"- **{t.get('name')}**  \n"
                    f"  _{t.get('description')}_  \n"
                    f"  Why: {t.get('reason') or '—'} · "
                    f"invasiveness {t.get('invasiveness')}/5 · "
                    f"`{t.get('category')}`"
                )
        if others:
            with st.expander("Additional options (context)", expanded=False):
                for t in others:
                    st.markdown(
                        f"- **{t.get('name')}** — {t.get('description')} "
                        f"(invasiveness {t.get('invasiveness')}/5)"
                    )
    else:
        st.caption("Treatment recommendations appear after surgical guidance is computed.")

    st.markdown(
        """
**Airflow toolbox (reference)**  
Inferior / middle turbinate reduction (RF or microdebrider) · septoplasty (caudal or posterior) ·
nasal valve support  

**CRS / drainage toolbox (reference)**  
Balloon sinus dilation · maxillary antrostomy · frontal drillout (when less invasive paths fail)
"""
    )

    # ---- BC summary ----
    with st.expander("Boundary conditions & breathing"):
        breath = bc.get("breathing", {})
        flow = bc.get("flow_assignment", {})
        st.write(
            f"**Inlets:** {', '.join(bc.get('inlet_names', []))}  ·  "
            f"**Outlet:** {bc.get('outlet_name', '?')}  ·  "
            f"**Mouth:** closed"
        )
        if breath:
            st.write(
                f"VT={breath.get('tidal_volume_L')} L · "
                f"RR={breath.get('respiratory_rate_per_min')}/min · "
                f"Ti={breath.get('Ti_s_effective', 0):.2f} s · "
                f"Q={flow.get('total_inflow_L_per_min', 0):.1f} L/min"
            )
        st.json(
            {
                "inlets": bc.get("inlet_names"),
                "outlet": bc.get("outlet_name"),
                "flow_split_L_per_min": breath.get("flow_split_L_per_min"),
            }
        )

    st.markdown("---")
    st.markdown(
        "**Future product modules** (not yet computed): ostium pathways · mucus "
        "clearance · CRS / NAO / NVC diagnosis · polyp ID · secure CT upload."
    )


if __name__ == "__main__":
    main()
