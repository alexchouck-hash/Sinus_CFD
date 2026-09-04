# Sinus_CFD — handoff (2026-08-20)

Read this if you are taking over the repo. It is the current snapshot, not a
roadmap rewrite. Research prototype — **not a medical device.**

| | |
|---|---|
| **Repo** | `C:\Users\houck\Sinus_CFD` |
| **GitHub** | https://github.com/alexchouck-hash/Sinus_CFD |
| **Branch** | `MVP` (tracks `origin/MVP`) |
| **Python** | 3.12 (`py -3.12`) |
| **Owner intent** | CT → nasal/sinus airflow + surgical-planning **demo**. Operator prefers **automatic** segmentation; will label 3–5 full volumes only if QA gates fail. **No slice-by-slice confirm/deny.** |

**Start here after this file:**

1. [`AGENTS.md`](../AGENTS.md) — pipeline commands, conventions, what not to commit
2. [`docs/segmentation_strategy.md`](segmentation_strategy.md) — how L/R + septum are *supposed* to work
3. [`docs/airway_connectivity_limitation.md`](airway_connectivity_limitation.md) — why NasalSeg Dice is a trap
4. [`PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) — product goal vs what actually ships

---

## 1. What this project is

Turn a head CT into:

- a **sinus-free** inspiratory fluid domain (both nares → choanae → pharynx/trachea)
- named sinus air (maxillary now; frontal/ethmoid/sphenoid still weak)
- OpenFOAM or a fast potential-flow preview
- a Streamlit 3D viewer (pathlines, naris→frontal corridors, pink “remove” zones)

The long-term product vision (disease detection, ranked treatment, navigation
export) is in `PROJECT_OVERVIEW.md`. **Most of that is not built.** What ships
is a whole-head CFD prototype plus a viewer.

---

## 2. What has been done (through 2026-08-20)

### Working product pieces

| Piece | Where | Honest status |
|-------|--------|----------------|
| Visible Human Female 1 mm demo | `outputs/VisibleHuman_Head/`, viewer | Working geometry + pathlines + OpenFOAM import |
| NasalSeg nnU-Net 5-class | Dataset501, Colab train, fold 0 | **0.885 Dice vs NasalSeg labels 1–3** — those labels do **not** connect both cavities to the nasopharynx |
| P001 OpenFOAM | `foam/P001/` | Resistance in published range on **n=1**; right-side flow was HU-bridged (can punch septum / maxillary) |
| Approximate flow + streamlines | `src/sinus_cfd/flow_field.py` | Preview only; magnitude uncalibrated |
| Connectivity QA + pytest | `scripts/qa_connectivity.py`, `tests/` | Exists; check 3 (sinus-free passage) is vacuous while `passage_lumen` ≡ `airway_mask` (AGENT_QUEUE **A12**) |
| Virtual surgery (geometry) | `src/sinus_cfd/virtual_surgery.py` | Edits L/R air using a **midplane** `_septum_x` — wrong for a deviated septum |

### Segmentation work in this cycle (2026-08-19/20)

Goal: auto-segment airway + septum **without** the operator painting slices.

| Deliverable | Path |
|-------------|------|
| Label taxonomy (IDs 0–10 air, 11 planned septum tissue) | `src/sinus_cfd/segmentation_labels.py` |
| Leftover-HU expander (names frontal/ethmoid/sphenoid after NasalSeg 1–5) | `src/sinus_cfd/segment_sinonasal.py`, `scripts/segment_sinonasal.py` |
| Method survey of NasalSeg / SinusSegment / Craneal_CT / etc. | `docs/segmentation_method.md` |
| Strategy (competing geodesic flood, no HITL, last-resort 3–5 Slicer volumes) | `docs/segmentation_strategy.md` |
| Geometric teacher | `src/sinus_cfd/auto_airway.py` |
| Wired into extractor behind a flag | `extract_ct_nasal_airway(..., legacy_midplane_split=True)` in `nasal_airway_ct.py` |
| CLI | `scripts/autosegment_ct.py` |
| Tests | `tests/test_auto_airway.py`, `tests/test_segment_sinonasal.py` (12 passed) |
| Preview image | `outputs/VisibleHuman_Head/VisibleHuman_Head_autosegment_preview.png` |

**Default refine is still the old midplane split** so the Streamlit demo does not
regress. New path: `--no-legacy` / `--no-legacy-midplane-split`.

### Measured autosegment runs

| Case | Command | Result |
|------|---------|--------|
| Visible Human Female | `autosegment_ct.py --case VisibleHuman_Head --no-legacy` | **L=19148 R=17983 septum=9821** — bilateral; septum ridge too fat on sagittal (spills posterior) |
| Visible Human Male | `--case VisibleHuman_Male_Head --no-legacy` | L=9563 R=6449; **septum=0** |
| NasalSeg P001–P005 | `--nasalseg-all --max-cases 5` | 5/5 wrote labels; leftover expander **over-paints frontal** with FOV-edge air |
| THCA living head-neck | `--case THCA_HeadNeck --no-legacy` | **Fixed 2026-08-20.** Was L=1172 R=14 (naris detector collapsed to one voxel). Now **L=1644 R=1664** (balance 0.99) via naris-pair validation + whole-head prior fallback |
| CQ500 tight crop | `--case CQ500CT100 --no-legacy` | **HARD-fails cleanly** (`naris_detection_HARD_fail`: CT pair 0.00 mm apart, prior pair 113.5 mm apart). Previously returned L=0 R=0 while reporting success. **Do not label** (operator decision); airway 26-connected to room air |

---

## 3. Where you are at (honest)

**Demo (legacy midplane)** still runs. **New extractor exists, is not the default, and is not viewer-wired.**

```
                    ┌─ legacy midplane (DEFAULT) ─ refine_nasal_ct.py ─ viewer
whole-head CT ──────┤
                    └─ competing flood (--no-legacy) ─ masks on disk; viewer still uses old L/R if present
```

NasalSeg crops can be auto-named (`labels+expand` or `--nnunet`) but that does
**not** give a CFD-correct bilateral passage. See connectivity doc: **0/130**
NasalSeg cases have both cavities connected to nasopharynx via labels {1,2,3}.
Retraining Dataset501 cannot invent a choana.

**Git:** working tree is **dirty**. Many docs, foam cases, scripts, and
`src/sinus_cfd/auto_airway.py` are **untracked**. Nothing from this cycle is
guaranteed on `origin/MVP` until someone commits. Do not commit `data/` or
`outputs/` (DICOM/NRRD/STL).

**nnU-Net weights** live under `data/nnUNet_results/` (gitignored). Colab
notebook: `notebooks/train_nnunet_colab.ipynb`.

---

## 4. How to run (handoff smoke)

From repo root, Python 3.12:

```powershell
cd C:\Users\houck\Sinus_CFD

# Demo viewer (legacy masks already in outputs/VisibleHuman_Head)
py -3.12 -m streamlit run app\viewer.py --server.address 127.0.0.1 --server.port 8501

# New autosegment (does not flip the demo)
py -3.12 scripts\autosegment_ct.py --case VisibleHuman_Head --no-legacy
# Preview: outputs\VisibleHuman_Head\VisibleHuman_Head_autosegment_preview.png

# NasalSeg crop (uses expert labels if present; --nnunet if weights + GPU)
py -3.12 scripts\autosegment_ct.py --image data\images\P001_img.nrrd --labels data\labels\P001_seg.nrrd

# Tests for this cycle
py -3.12 -m pytest tests\test_auto_airway.py tests\test_segment_sinonasal.py -q
```

Whole-head `--case` requires `process_whole_head` outputs already under
`outputs/<case>/` (`*_tissues.nrrd`, `*_head_mask.nrrd`, `*_stats.json`).

---

## 5. Data on disk (gitignored)

| Path | What |
|------|------|
| `data/images/` `data/labels/` | NasalSeg 130 CT + 5-class NRRD |
| `data/VisibleHuman_Head/` `data/VisibleHuman_Male_Head/` | 1 mm cadaver whole-head |
| `data/THCA_HeadNeck/` | Living IDC head-neck (~1.5 mm) |
| `data/CQ500CT100/` `data/incoming/qct*/` | CQ500 DICOM library (many series) |
| `data/nnUNet_raw/` `data/nnUNet_results/` | Dataset501 + fold-0 weights |
| `outputs/<case>/` | Masks, STL, flow, streamlines |

Never commit patient trees or large volumes.

---

## 6. Code map (segmentation)

| File | Role |
|------|------|
| `src/sinus_cfd/auto_airway.py` | Competing geodesic flood, bone landmark, through-path strip, EDT septum ridge |
| `src/sinus_cfd/nasal_airway_ct.py` | Naris detector, **legacy midplane split**, vestibule paint, `extract_ct_nasal_airway` |
| `src/sinus_cfd/segment_sinonasal.py` | Hybrid 11-class map from NasalSeg 1–5 + leftover HU |
| `src/sinus_cfd/segmentation_labels.py` | Canonical IDs; `PASSAGE_IDS = (1,2,3)` |
| `src/sinus_cfd/whole_head.py` | HU body/airway; `select_nasal_to_trachea_path` (flood **domain**) |
| `src/sinus_cfd/pipeline.py` | NasalSeg-crop `process_case`; `_bridge_through_air` can punch the septum |
| `scripts/autosegment_ct.py` | One CLI for whole-head + NasalSeg |
| `scripts/refine_nasal_ct.py` | `--legacy-midplane-split` (default on) |
| `scripts/extend_nasal_to_tip.py` | **Still recuts L/R on naris-mid plane** — will undo `--no-legacy` if you run the canonical AGENTS.md pipeline |
| `scripts/qa_connectivity.py` | Topology QA; **must not import `src/`** |

Midplane clones that still exist (must stay on the same flag before default flip):
`extend_nasal_to_tip.py`, `virtual_surgery._septum_x`, viewer/artifact
`sign(x - septum_x)`, `surgical_zones` 7 mm heuristic, `open_path.split_frontal_lr`.

---

## 7. What is a struggle (do not re-learn the hard way)

### Septum

The old definition is **the sagittal plane through the midpoint of the two
nostrils** (`split_left_right_by_septum_plane`). A deviated septum is not that
plane. That is why L/R, septoplasty, and streamline side-assignment are wrong
on the anatomy that matters.

The new septum is an **EDT ridge between anterior L and R air**. On Visible
Human Female it exists but is **too thick** (sagittal yellow blob into
sphenoid/ethmoid). Male VH ridge came out **empty**. Thickness priors and
merge-zone placement are unfinished.

### NasalSeg is the wrong teacher for connectivity

Labels {1,2,3} never form one bilateral vestibule→NP tube (0/130). nnU-Net
copies that gap. HU-bridging P001 connected the right side **through the
maxillary sinus or across the septum**. **Do not retrain Dataset501 to “fix”
choanae.**

### Nares on living CT — **fixed 2026-08-20, but know why it broke**

`detect_nares_from_ct_air` needs a clean anterior air shell. On THCA it did not
have one, and the failure was silent. The chain, measured end to end:

1. THCA has **no resolved air anterior of y=43** — `tissues==1` spans only
   y[43,83]. The nostrils and vestibule are not air in this reconstruction.
2. `whole_head.select_nasal_to_trachea_path` therefore painted a **synthetic
   geodesic tube** through soft tissue ("Geodesic tube used where free air is
   discontinuous"), so `airway_mask` spans y[3,138]. Tube voxels carry
   soft-tissue HU (+34 / −32), not air HU.
3. The detector builds its anterior shell on that tube, then the air-HU
   preference filter (`opening & (hu <= -150)`) **discards every tube voxel**,
   silently relocating the "nostrils" ~24 mm posterior into the mid-cavity.
4. The relocated blob is unimodal, so the two-peak search fails and the
   percentile fallback splits one blob; the ±12-voxel bands in
   `_center_near_peak` then overlap and return the **same voxel twice**.
5. The old repair nudged x by two voxels, leaving both seeds on the same side of
   the septum → competing flood left=4153 right=116 → L=1172 R=14.

**The fix is not a better detector — it is a contract.** `validate_naris_pair`
(spacing-aware, all thresholds in mm) rejects a pair that is not two distinct
nostrils; the caller then falls back to the **whole-head prior** from
`<case>_nares.json`, and HARD-fails if that is invalid too. The prior was always
being passed in and was always correct — it was only ever used when the detector
returned `None`, which a collapsed pair does not.

The HU filter still relocates the shell; it now says so
(`WARN: air-HU filter moves the naris shell ... mm posteriorly`). The statistic is
the **signed median**, measured *before* the apply-decision, against a 10 mm bar:

| Case | signed median shift | WARN |
|---|---|---|
| Visible Human Female | +1.0 mm | silent |
| Visible Human Male | −3.0 mm (anterior, not the failure mode) | silent |
| THCA | **+32.2 mm** | fires |
| CQ500CT100 | below bar | silent (it HARD-fails on the pair instead) |

Do **not** revert this to `abs(mean)` inside the `>= 40` branch. The mean puts VH
Female at +4.4 mm against a 5 mm bar — 0.6 mm from a false alarm on the one case
that works — and it fired spuriously on VH Male. Nesting the measurement inside
the apply-branch also makes it dead code on any case that starves that branch.

Treat a firing WARN as the marker for "the anterior airway here is painted, not
imaged." `naris_source` in `<case>_ct_nasal_meta.json` records where the accepted
seeds actually came from: `ct_air_shell` (VH F/M), `prior` (THCA), `none` (CQ500).

Sealed 1 mm nares are why vestibule paint exists (`extend_nasal_to_tip.py`).

**Do not** "fix" this by preferring the prior unconditionally: on Visible Human
Female the prior sits **36 mm** from the (correct) detected seed, so a
prior-always-wins rule would regress the one case that works.

### Choanal landmark / sinus strip

Bone-first palate often sits on the **nasal-box posterior face** → empty merge
zone. Fallback is “posterior 25% of air” (WARN). Through-path
naris→outlet then treats **the other cavity** as a detour; the extractor
**refuses** the strip if it zeros a side (`WARN: through-path dropped a cavity`).
So ostium exclusion is **not** actually on for real heads yet.

### Tight crops (NasalSeg, CQ500)

Airway is 26-connected to **room air** through the nares. `segment_body`
hole-fill cannot draw a nostril plane. Five auto isolations of CQ500 were
abandoned (~0 mL). Leftover-air expander then names **FOV-edge air as frontal**
(P001 left_frontal ~183k voxels — not anatomy).

### What we explicitly will not do

- Slice-wise human-in-the-loop / “accept this slice”
- Vendor Craneal_CT weights (CC-BY-ND-NC); that model is **bone**, not air
- Use SPESIS (MRI peri-sinus *space*, wrong anatomy)
- Replace nnU-Net with SinusSegment’s public trainer (`num_classes=1` binary)
- Ask the operator to label CQ500’s tight crop

---

## 8. Operator labeling policy (locked)

Default: **automatic.** Manual work is a **last resort**: 3–5 **complete** 3D
Slicer volumes with a starter to carve, IDs 0–10 plus **11 = septum_tissue**.
Tripwire (not yet coded as a library script): ≥40% of ≥3 eligible whole-head
volumes HARD-fail after the extractor exists, or Visible Human HARD-fails.

Procedure if tripped: `docs/slicer_labeling.md` (extend with class 11).

---

## 9. Next work (priority order)

Do **not** flip `legacy_midplane_split` default until Visible Human passes a
real `--strict-septum` QA.

1. ~~**Naris detector on living CT** (THCA / CQ500 body-bounded). Two distinct seeds or HARD-fail. Do not invent a midplane.~~ **DONE 2026-08-20.** `validate_naris_pair` + prior fallback + `naris_detection_HARD_fail` in `nasal_airway_ct.py`; `tests/test_naris_validation.py` (13 tests). THCA L=1644 R=1664; VH F/M byte-identical; CQ500 HARD-fails with a reason. Remaining sub-item: the CQ500 **prior itself** is bad (113.5 mm apart) because that crop's airway touches room air — that is the tight-crop topology problem, not a detector problem.
2. **Choanal landmark** that is not the crop’s last y-slice. Merge zone must contain NP air.
3. ~~**Through-path / dead-end ostium cut** that cannot delete a cavity (valve stays, maxillary detour goes).~~ **DONE 2026-08-22.** `auto_airway.dead_end_sinus_strip` replaces `through_path_passage` (K9 → K9a). Measured sinus: VH Female 15.1 mL (8.9 + 7.0 bilateral maxillary), VH Male 27.5 mL, CQ500CT105 3.3 mL vs 4.1 mL independently measured in-FOV. **A12 is closed as a side effect** — `passage_lumen` is now genuinely distinct from `airway_mask` on the `--no-legacy` path (VH 34.9 vs 55.3 mL; CQ500 26.9 vs 35.8 mL), so `qa_connectivity` check 3 is no longer vacuous. Two things to know: the strip runs on the **whole-head airway**, not the nasal-box flood domain (the box faces are artificial cuts that read as wide openings); and the merge zone is a passage **marker and rejection test**, never a bottleneck source (seeding widest-path from it cost VH one maxillary). Residual: ~1.2 mL of nasopharynx still reads as sinus on CQ500 because its merge zone came from the `air p75` fallback, not a bone landmark — that is bounded by item 2, not by the strip.
4. **Thin septum ridge** (τ / dmax / peel). Viewer sagittal should look like a wall, not a blob.
5. **Stop `extend_nasal_to_tip.py` recutting** L/R on `x_sep` when `--no-legacy`.
6. **QA `--strict-septum`** in `qa_connectivity.py` (no `src/` import). Check 3 SKIP until passage ≠ airway (A12).
7. **Viewer** consumes `sign(d_L − d_R)` behind the same flag; bump `APP_VERSION` only on default flip.
8. Optional Dataset504 distill **after** QA-passing pseudo-labels. Slim IDs 0–6, remap 6→11. Do not block CFD on this.
9. `forbid_transseptal` on `_bridge_through_air` (1–2 **and** 2–3). NasalSeg-crop CFD only.

Full PR list: `docs/segmentation_strategy.md` § PR Plan. Implementation has
**collapsed PR-1/2/3/6 into working code** behind the flag; remaining PRs are
hardening, QA, and consumers.

Product stages B–D (disease, treatment report, ostia navigation) wait on a
trustworthy passage + septum.

---

## 9b. Patency assurance (2026-08-22)

`scripts/qa_patency.py` answers the two questions that gate CLAUDE.md goals 2-4.
It imports from `src/` (unlike `qa_connectivity.py`, which is import-free by contract).

```powershell
py -3.12 scripts\qa_patency.py --case CQ500CT105
py -3.12 scripts\qa_patency.py --all --json outputs\patency.json
```

**[1] Flow path** — is each naris connected to the outlet *inside `passage_lumen`*, how far
did the BC port have to snap, how long is the route, and how tight is it at its narrowest?
Exit non-zero if any case lacks a bilateral path. **[2] Drainage** — per sinus: anatomical
name, side, volume, ostium diameter, whether it reaches the passage, whether it is patent.
Drainage is reported, not gated: an obstructed ostium is a clinical finding, not a bug.

| case | flow path | maxillary L / R (mL, ostium mm) | frontal | sphenoid |
|---|---|---|---|---|
| VisibleHuman_Head | **PASS** 117.5 / 109.7 mm, tightest r=1.0 mm | 7.70 @14.0 / 7.01 @4.9 | none | none |
| VisibleHuman_Male_Head | **PASS** 126.4 / 121.1 mm | 4.54 @20.4 / 1.67 @19.8 | none | none |
| THCA_HeadNeck | **PASS** 155.4 / 155.9 mm | none | none | none |
| CQ500CT105 | **FAIL** — BC inlets do not resolve | 0.73 @2.5 / 0.60 @3.1 | none | none |

Two defects it found and that are now fixed: the nasal box began *at* the CT-air-shell
naris, so `passage_lumen` stopped 32.8 mm short of the BC inlet
(`NASAL_BOX_ANTERIOR_MM = 45.0`); and the box cropped away the trachea, so the outlet port
did not resolve either (outlet glue — whole-head airway outside the box, connected to the
passage, is unioned back on; sinuses are inside the box and stay stripped).

**Still failing, and worth knowing why each is different:**

- **CQ500CT105** — its BC ports are wrong at source: `whole_head`'s edge naris detector put
  the left naris at x=304 when the airway spans x[124,298]. The diagnostic line says
  `on airway_mask (pre-strip) 0/2 inlet(s) connect`, i.e. the ports miss the anatomy
  entirely. This is a `_ports_from_edge_nares` bug, **not** a segmentation bug — the
  segmentation of that case is the best in the repo.
- **No frontal or sphenoid sinus is found on any case.** On VH/THCA that is largely honest
  (1 mm cadaver data, and THCA has no resolved sinus air at all), but it is unverified on
  CQ500CT105, whose FOV clips inferiorly. Goal 3 needs these, so this is the next gap.
- **Ostium calibre is only trustworthy on sub-millimetre data.** VH reports 14-20 mm
  "ostia" because its maxillary wall is perforated by partial volume (19 connections) —
  broad connection is the truth for that scan. CQ500CT105 at 0.39 mm reports **2.5 and
  3.1 mm**, which is anatomically correct for a maxillary ostium. Do not compare the two.

---

## 9c. Pipeline status and queue (2026-08-30)

Answers the question "does the pipeline work, end to end?" stage by stage, with
the evidence for each answer. Supersedes the stage claims in §12, which predate
the 2026-08-24/30 fixes.

### Stage status

| # | Stage | State | Evidence |
|---|-------|-------|----------|
| 1 | Import CT + auto-segment | **works** | `qa_patency.py --all`: flow path PASS on all 5 audited cases (VH F, VH M, THCA, CQ500CT105, CQ500CT390). Bounded domain, L/R split, sinus bodies named. |
| — | Operator labelling | **works, low yield** | 22 ostium sheets marked. Ostium location validated to **2.5 mm median** — at the noise floor, since two operator marks of the *same* ostium differ by 4.5 mm. |
| — | Neural net | **works, wrong target** | nnU-Net Dataset501 airway Dice **0.885** vs classical 0.260. But **0/130** NasalSeg cases connect both cavities to the nasopharynx, so the labels cannot teach CFD topology. Not a retraining problem. |
| 2 | Mesh boundary | **works; sinus-free at every layer since 2026-09-03** | `export_openfoam_geometry.py` writes a solid air body plus open inlet/outlet/wall STL patches. Foam cases for VH F, VH M, THCA, P001, CQ500CT390. Re-exporting CQ500CT390 produced a **6-face, 0.00 mL** "watertight" body against a 38.91 mL mask and reported it as good — see §9d. Now guarded: the surface is compared against the voxel mask and the export raises if they are >15% apart. |
| 3 | Mesh refinement | **works** | snappyHexMesh + layers. `checkMesh` **Mesh OK** on 3 of 4; non-orthogonality max ~64.5, average 13.4–14.3. Independence study on P001 at 74.9k / 259k / 817k cells → R = 0.0541 / 0.0522 / **0.0520** Pa·s/mL. 259k is independent to 0.3%. |
| 4 | Airflow nares → trachea | **works end to end on sinus-free domains** | CQ500CT390 (living, 0.38 mm) went CT → segmentation → export → snappyHexMesh (312,652 cells) → simpleFoam → import on 2026-09-01: dP **7.87 Pa** at 18 L/min, R 0.0262 Pa·s/mL, per-side 7.41 / 8.33 Pa, inlet pressure flat to 0.26%, outlet flux 0.01% off the imposed value. Solver is OpenFOAM 2412 in WSL Ubuntu 24.04 (Docker Desktop is wedged — see §9e). The four July solves are still stale against their August masks; THCA's never settled at all. |
| 5 | Sinus drainage | **works** | Per-sinus volume, ostium calibre and drains yes/no. Calibres 1.46–2.83 mm, inside the 0.2–6 mm anatomical range. Where no ostium resolves, `probable_ostium` returns a labelled hypothesis, never a number. |
| 6 | Seeker paths to ostia | **centreline only** | `compute_surgical_guidance.py` produces naris→frontal paths and a high-\|u\| corridor. It models **no instrument geometry at all** — no diameter, no curvature, no bend, no clearance test. |
| 7 | Virtual surgery | **geometry only** | `virtual_surgery.py` edits the airway and reports pre/post volume, MCA and L/R ratio. It writes an edited label so CFD *could* re-run; no pre/post CFD comparison has been run. |

### The 20-minute budget — now measurable (2026-09-01, CQ500CT390)

Full chain on a living 0.38 mm scan, 312,652 cells, wall clock:

| stage | time | note |
|-------|------|------|
| `process_whole_head` + `autosegment_ct` | ~4 min | segmentation |
| `analyze_passage --skip-flow` | **140 s** | port masks + centreline; the potential-flow *preview* (another ~935 s) is now opt-in via `auto_process_head --preview-flow` and the CFD never reads it |
| `export_openfoam_geometry` | 86 s | |
| `scaffold_openfoam_case` | 3 s | |
| `snappyHexMesh` + layers | 309 s | |
| `simpleFoam` to a settled dP | **~236 s** (≈200 iter × 1.18 s) | ran 2,000 iterations = 2,361 s; the last 1,800 changed dP by <0.2% |

**The whole chain is now inside the budget: ~4 + 2.3 + 1.4 + 0.05 + 5.1 + 3.9
≈ 17 minutes** on a 0.38 mm living scan, with the solve stopped where dP settles.
The 18-minute potential-flow preview that used to sit on the road to export is
opt-in. Verified the honest way: the fast path reproduces all five masks the
solved case was built from **byte for byte** — after fixing what §9f describes.

**What "converged" now means.** Every archived run had hit its 500-iteration
cap without tripping `residualControl`. SIMPLEC (p 1 / U 0.9, replacing SIMPLE
p 0.3 / U 0.5) brought the first-corrector p residual from 5.5e-3 to 3.8e-3 by
iteration 200 — then it sat there to 2,000, pinned by two highly-skew boundary
faces where the flat nostril cap meets the curved wall (`checkMesh` max
skewness 6.79, faces on `left_nostril` and `wall`). The *answer* was flat from
iteration 50. So `endTime` is capped at 400 and convergence is judged by
`import_openfoam_results.py` on the `surfaceFieldValue` history: it computes
dP, per-side split and resistance and **raises** if the inlet pressure moved
more than 1% over the last 3 samples.

Re-solved on the sinus-free domains (2026-09-03, evening), choanal outlet on
every whole-head case, judged by drift / wobble / outlet patch:

| case | cells | max skew | dP (Pa) | L / R (Pa) | R (Pa·s/mL) | drift | wobble | outlet | verdict |
|------|-------|----------|---------|------------|-------------|-------|--------|--------|---------|
| CQ500CT390 | 358,234 | 4.86 | 20.55 | 19.37 / 21.73 | 0.0685 | 0.04% | 0.03% | 1.8× | settled |
| P001 | 257,780 | 3.07 | 16.72 | 21.99 / 11.46 | 0.0557 | 0.16% | 0.51% | 3.0× | settled |
| VH male | 271,010 | 2.67 | 5.97 | 4.53 / 7.40 | 0.0199 | 0.57% | 1.22% | 2.8× | settled |
| THCA | 237,438 | 3.27 | 14.42 | 13.72 / 15.13 | 0.0481 | 0.09% | 0.16% | 3.4× | settled |
| VH female | 162,566 | 4.56 | — | — | — | — | — | — | **refused — choked: max face speed is 9x the patch mean** |

Every earlier number, July and this morning's, is superseded: those
domains held all of their sinus air (§9h). See §9g for the outlet story.

**Magnitudes are not calibrated.** Physiological nasal resistance at 300 mL/s
is roughly 0.1–0.3 Pa·s/mL; these sit 4–500× below it, and the two 1 mm
cadaver cases give near-identical, near-zero drops. Likely contributors: planar
nostril caps that erase entrance loss, a nasal valve not resolved at 1 mm, and
laminar treatment. *Relative* outputs — P001's 2:1 left/right asymmetry,
CQ500CT390's right side 12% more resistive — are the kind of number goals 3
and 5 need; absolute resistance is not yet one of them. Queue item 2.

### 9e. Docker Desktop is wedged; the solver runs in WSL (2026-09-01)

Docker Desktop 4.82 crashes on start with `remove …/dockerInference: The file
cannot be accessed by the system`, then the same on
`docker-secrets-engine/engine.sock`. These are AF_UNIX socket files Windows
will not delete, and every restart mints a fresh one — moving `run/` aside and
disabling the inference engine in `settings-store.json` both got past one
socket and hit the next. The OS-level fix is a reboot. Rather than wait,
OpenFOAM 2412 (the same version as the `opencfd/openfoam-run:2412` image) was
installed into a fresh WSL `Ubuntu-24.04` distro as root, and `run_in_wsl.ps1`
now targets it (`$env:SINUS_CFD_WSL_DISTRO` overrides). `Allrun` sources the
2412 bashrc. The Docker route still exists and should work after a reboot.

### 9d. The export collapse (found and fixed 2026-08-30)

Worth recording because the failure was silent and three faults deep, each
covering for the last:

1. trimesh 4.x changed `simplify_quadric_decimation` to
   `(percent, face_count, aggression)`. The positional call passed the 40,000
   face target as *percent* and raised `target_reduction must be between 0 and 1`.
2. A bare `except Exception` caught that and fell back to keeping every Nth face
   via `np.linspace` — not decimation, but shredding the surface into loose
   triangles. `_largest_component` then picked **6 of 266,060**, and `fill_holes`
   closed them into a degenerate shell.
3. The acceptance test asked only `is_watertight`. A closed shell of 6 triangles
   *is* watertight and bounds nothing, so the ruin passed and the export printed
   "watertight (good for snappyHexMesh)".

Reproduced exactly on an icosphere: 5,120 faces → 8, volume 4.18 → 0.01.
`fast_simplification` was also missing from the environment, which is what made
the argument bug fatal rather than merely wrong; it is now installed, but none of
the three fixes depend on it.

The lesson generalises: **watertightness is a topological test and cannot detect
a surface that bounds the wrong body.** The guard that catches it is comparing
the mesh volume against the voxel mask it came from.

### 9f. The import that moved the outlet (found and fixed 2026-09-01)

The byte compare above failed the first time: `outlet_open` came back at 2,450
voxels against the 1,780 the solver used — same lumen, same code. The cause was
`import_openfoam_results.py`. It overwrote the outlet port's `center_mm` — the
domain definition `analyze_passage` seeds the outlet cap from — with the
*centroid* of the previous cap, "for viewer labels". The lumen clips the cap
asymmetrically, so its centroid sits anterior of its own seed; writing it back
moved the seed, and the next cap grew from there. One import walked the seed
3 voxels (1.1 mm) anteriorly. Every import → re-analyze cycle would have walked
it further. The CFD domain changed because a plot label was saved.

Proven, not argued: rebuilding the cap from candidate seeds against the real
port builder, only **(27,232,220)** — the pre-import seed — reproduces the
solved geometry byte for byte; the post-import seed (27,229,220) gives 2,450.

The marker now lives in `ports[].viewer_marker_mm`; `center_mm` is never
touched by anything downstream of the geometry stages, and the viewer prefers
the marker when present. This is the repo's "averaging positions" trap in a
new coat: a downstream consumer editing an upstream definition.

### 9g. The outlet cap: ratchet, choke, guard (2026-09-03)

Re-solving the four stale cases turned up the second half of the outlet story.
Both Visible Human heads, solved with the caudal trachea as outlet, gave drops
~1000× off (182 Pa; and a run that never settled). July's 0.18 Pa had been
wrong the other way. On the solved field the inlets sat at 1.0× their mean
face speed while the **trachea patch carried 57× (female) and 8× (male)** —
the whole pressure drop lived on a few outlet faces (implied outlet area
44 mm² against a 133 mm² cap). Moving the outlet to the choana on the same
masks gave clean patches (2.5×) and a settled 4.70 Pa on the male.

Two pre-solve predictors were measured and **rejected** — recorded so nobody
retries them: the cap's PCA axes (every cap is a fat 6 mm ball clipped by the
lumen; the choked 12×6×6 mm cap looks like P001's clean 19×8×8) and whether
the cap touches the image boundary (none does). Only the solved field
separates the populations, at 15×. So:

- `auto_process_head --outlet` defaults to **nasopharynx** — the
  nares→nasopharynx domain the roadmap specifies. `auto` / `trachea` remain.
- The import reads the outlet patch of the solved `U` and **raises above 10×**
  max/mean face speed. Clean caps measure 1.8–3.5×; the choked ones 8× and
  41–57×. It tries the mesh patch name `trachea` before the BC's port name
  (P001's BC says `trachea_outlet_proxy`) and says so loudly if neither reads.
- **Drift vs wobble.** THCA sat at a per-sample wobble of 1.53% for 400 extra
  iterations while its resistance moved 0.6%; a per-sample rule at 1% called
  that unconverged. The verdict now reports two statistics with their own
  bounds: *drift* (window mean against the previous window, ≤1%) and *wobble*
  (amplitude within the window, ≤3%). Oscillation averages out; a trend does
  not.

Also found on the way, both now fixed and tested: an import in a case
directory that still held an older run read the older run (`latest_time_dir`
took July's `500/` over today's `400/`, and OpenFOAM writes a new history to
`surfaceFieldValue_0.dat` beside an old one); and the chain must clean stale
time directories, `postProcessing/` and `log.*` before a solve. July results
are snapshotted under the session scratchpad `july_foam/`.

### 9h. The domain was never sinus-free (found and fixed 2026-09-03)

Measured on the files each solve was built from, every CFD domain ever meshed
here held **all** of its stripped sinus air: VH male 30.9 of 30.9 mL, THCA 36.3
of 36.3, CQ500CT390 5.6 of 5.6. The roadmap forbids this (goal 1). Four
re-entry points, each found by measuring the array the next stage received
rather than trusting the previous stage's name, each fixed with a guard that
refuses to pass on a contaminated array:

1. **`analyze_passage` never read the stripped passage.** The line said
   `...passage_lumen.nrrd).is_file() and False:` — hard-disabled — so it started
   from the raw airway and then overwrote both `airway_mask.nrrd` and
   `passage_lumen.nrrd` with its own lumen (which also put the drainage audit
   on a different mask from the sinus bodies: 6.63 mm "maxillary ostia" on VH
   male, real 2.83). Now starts from the stripped passage, says which lumen it
   used, never writes `airway_mask.nrrd`.
2. **The nostril-tunnel extension refilled the sinuses.** A stripped sinus is
   a 3-D hole in the passage; the whole-lumen `binary_fill_holes` meant to
   close pits in the painted tunnels put 93–97% of each body back (CQ500CT390:
   0.29 → 5.60 mL). Closing and fill are confined to one voxel around the
   paint; the stripped air is passed in as `exclude`; the analysis raises
   before writing at >5%.
3. **The export merged them back by default.** `include_sinuses=True` merged
   every interior-air component touching the passage — the sinuses at their
   ostia (VH male: 47.6 mL clean in, 75.5 mL out with 23.55 of 23.5 mL back).
   Sinus-free is the default; `--include-sinuses` is the opt-in; the export
   refuses at >5%.
4. **The mesh seal refilled them after the export's own guard passed.** Two
   whole-solid `binary_fill_holes` passes in `seal_solid_for_watertight_mesh`
   (VH male: closing 0.91 mL, first fill 23.55 mL). The seal takes `exclude`;
   the guard moved onto the *sealed* solid that is written and meshed.

And a fifth, of a different kind: with the sinuses out, they are **cavities**
in the solid, marching cubes gives an outer shell plus one inner shell each,
and the mesher kept only the largest component — the surface wrapped the
sinuses back in with no inner wall (VH male: 74.1 mL enclosed vs 51.8 voxel;
inner shells 13.6 + 9.5 mL). Every closed shell above 20 mm³ is kept now,
cavity walls are oriented away from the fluid, and the mesh-vs-voxel guard
(added for the 6-face collapse) is what caught it. VH male's surface now
encloses 49.23 mL against 51.82; CQ500CT390 33.17 against 33.31.

CQ500CT390 measures what that fifth one was worth: on the export whose
sinuses were wrapped back in without inner walls, simpleFoam tripped
residualControl at 250 and reported dP 8.02 Pa (R 0.0267). With the inner
walls back, 20.55 Pa (R 0.0685). A converged solve on the wrong domain is
still the wrong number; the mesh-vs-voxel guard is what stands between them.

Also fixed on the way: `_merge_split_sinuses` glued bodies 104 mm apart
(previous commit); air more than 45 mm below the antral roof is not a sinus
(VH male's neck and skull-base "maxillary L" bodies); a stale
`*_solid_watertight.flag` let a chain whose export had just been refused fall
through to solving the previous export.

CQ500CT390: 6.79 → 4.86 max skewness, two faces remain, both now on the wall (one was on the left-nostril cap); VH female still one wall face at 4.56. The residual is the cap/wall corner geometry, not the layer template. VH female is refused at the choana regardless: the outlet patch's hottest face runs 9× its mean (limit 5×), so on that 1 mm cadaver the outlet section, not the airway, sets the pressure drop; the importer now leaves `<case>_flow_refused.json` with that reason and no field.

### Queue

Tagged by what unblocks each item: **[CODE]** implementable now, **[LABEL]**
needs operator marks, **[NN]** needs a network, **[CFD]** needs a solve, **[DATA]**
needs a better scan.

**Blocking the end-to-end claim**

1. **[CFD]** ~~Calibrate magnitudes~~ — **closed by evidence, one follow-up.** The "4–500× below physiological" compared a rest-breathing CFD resistance against rhinomanometry, which is measured at ~150 Pa driving pressure, ten times quiet breathing. The July sweep on P001 (`foam/P001/sweep`, `flow_sweep_report.py`) already answers it: R = 0.052 / 0.098 / 0.156 / 0.229 Pa·s/mL at 18 / 36 / 60 / 90 L/min, ΔP = 15.7 / 58.7 / 156 / 344 Pa — ΔP ∝ Q^1.9, the expected inertial nonlinearity. At 60 L/min (ΔP ≈ 156 Pa, the rhinomanometry driving pressure) R = 0.156, inside the published 0.10–0.35 band. The solver is laminar at ν = 1.5e-5, flow-rate inlets, p = 0 outlet; Re ≈ 300–1300 at rest, where laminar is defensible. **Follow-up:** re-run the sweep on the rebuilt, sinus-free P001 and quote R at 18 and 60 L/min side by side wherever a resistance is shown. **Done 2026-09-03 on the sinus-free P001** (`foam/P001_sweep`): R = 0.0557 / 0.1026 / 0.1685 / 0.2387 Pa·s/mL at 18 / 36 / 60 / 90 L/min, ΔP = 16.7 / 61.5 / 168.5 / 358 Pa, at ρ = 1.2 kg/m³ — the one constant every reporter now uses (the July figures above were at 1.14, which is why they read 5% lower). R at 60 L/min = 0.169, inside the band; the 18 L/min row reproduces the import's 16.72 Pa exactly.
2. **[CODE]** The two highly-skew faces at the inlet-cap/wall corner. The layer template now names the open ports with `nSurfaceLayers 0` and adds `slipFeatureAngle 30` so the wall's prism stack can slide along a cap instead of pinning to it. Harmless on a clean case (THCA: Mesh OK, max skewness 3.23 before and after). CQ500CT390: 6.79 → 4.86 max skewness, two faces remain, both now on the wall (one was on the left-nostril cap); VH female still one wall face at 4.56. The residual is the cap/wall corner geometry, not the layer template. VH female is refused at the choana regardless: the outlet patch's hottest face runs 9× its mean (limit 5×), so on that 1 mm cadaver the outlet section, not the airway, sets the pressure drop; the importer now leaves `<case>_flow_refused.json` with that reason and no field.

**Goal 1/3 correctness**

3. **[CODE]** Sphenoid air is inside the CFD domain (THCA: 9.1 mL). `merge_zone`
   is a posterior half-space, so it holds the sphenoid as well as the
   nasopharynx. Needs a merge zone bounded by the choanal aperture. Narrowing the
   veto to "half-space not itself behind a neck" was tried and **rejected** — it
   carved the nasopharynx instead (bodies bounded at 6.5–11.5 mm openings).
4. **[CODE]** CQ500CT390's right maxillary antrum is never found. Its left is
   found at 1.36 mL. Likely opacified or below the FOV — screen before fixing.
5. **[DATA]** CQ500CT390's naris ports are not at the nostrils. The FOV is
   brain-framed and the nostrils sit at or below its bottom edge, so the ports
   fall back to the airway's anterior opening ~33 mm higher. Flow path still
   passes, but "flow from the nares" is not literally true for that case.

**Goal 4 — the biggest gap**

6. **[CODE]** Instrument-fit checking does not exist. A centreline is not a path.
   Needs clearance against each tool's real geometry: frontal seeker **2 mm
   diameter** curved, maxillary and sphenoid seekers with their own curvature,
   ET seeker ~4 in shaft with a **45° bend** and **18.5 mm** working tip past the
   bend. This is a geometric feasibility problem, not a shortest-path problem.
7. **[LABEL]** Frontal and sphenoid ostia resolve on **no** usable case — those
   recesses are sub-millimetre. Operator marks on the frontal recess would give
   `probable_ostium` something to be scored against.

**Goal 5**

8. **[CFD]** Run the pre/post pair through CFD. `virtual_surgery.py` already
   writes the edited label; nothing has been solved on one.

**Data**

9. **[DATA]** The corrected screen (`assess_dicom_incoming.py`: SliceThickness, hole-filled silhouette, `--report`/`--csv`) is running over `data/incoming` and persisting every verdict to `outputs/incoming_screen.json`. Final tally 2026-09-03: 411 scans measured, 381 viable (346 at 0.62 mm, 31 at 1.0 mm; median 70 mL internal air); 29 rejected for too few slices, 1 for no internal air. Next: promote the viable cases to NRRD and run `qa_patency` on each.


---

## 10. Conventions and landmines

- Arrays `(z, y, x)`; spacing/origin from SimpleITK. **High x ≈ patient left** on Visible Human.
- `y_anterior_is_low` / `superior_is_high_z` are per-case; persist them (A12 still open).
- Passage for CFD = labels **1,2,3** or L∪R∪NP — **not** sinus air.
- mm↔index must use spacing (`TransformPhysicalPointToIndex`). Old `round(mm - origin)` was a confirmed bug.
- Python 3.12; `sys.path` → `src`.
- Do not commit `data/` volumes, `outputs/`, OpenFOAM `polyMesh` binaries, or nnU-Net checkpoints.

---

## 11. Agent protocol (if you use one)

`PROJECT_OVERVIEW.md` §6: Grok was meant to be **read-only** (audits in
`grok_inbox/`); an integrator owns `src/`. In the 2026-08-19/20 cycle Grok
**did implement** `auto_airway.py` and the CLI because the operator asked to
continue until CTs autosegment. Verify-then-adopt still applies: do not treat
VH voxel counts as “validated septum.”

Live task list: `AGENT_QUEUE.md` (Wave 1 stale in places; B1 is this
segmentation work). Audits: `grok_inbox/2026-07-21_*.md`.

---

## 12. One-screen status

| Question | Answer |
|----------|--------|
| Can I run the old demo? | Yes. Viewer + `outputs/VisibleHuman_Head`. |
| Can I autosegment without labeling? | **Yes on Visible Human** (`--no-legacy`) **and now on THCA** (bilateral, L≈R). **Partial** on NasalSeg crops (names, not CFD topology). **No** on tight CQ500 — it now HARD-fails honestly instead of returning zeros. |
| Is the septum solved? | **No.** Midplane banned in the new path; ridge is a first cut (too fat or empty). |
| Should I retrain nnU-Net next? | **No**, not to fix connectivity. Optional Dataset504 only after topology QA. |
| Should I ask the owner to paint slices? | **No**, unless the library tripwire fires. Then 3–5 full volumes, not per-slice. |
| What do I commit first? | New `src/`, `scripts/autosegment_ct.py`, `tests/`, `docs/` — not `data/` or `outputs/`. Working tree is currently uncommitted. |
