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
