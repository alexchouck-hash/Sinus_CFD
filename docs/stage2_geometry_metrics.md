# Stage 2: per-side airway geometry metrics (no CFD)

Roadmap §5 ("Geometry first — no CFD needed") and staged plan item 2. This is
the clinical geometry summary that falls out of a trustworthy segmentation
without any flow solve: which nasal passage is narrower, by how much, and
where.

## What it produces

`scripts/geometry_report.py` → per case:

- `<case>_geometry_report.json` — structured metrics
- `<case>_area_profile.png` — left/right cross-sectional area vs distance from
  the anterior naris, with each side's MCA marked (the standard nasal
  "area-distance curve").

Per nasal cavity (left = label 1, right = label 2):

| Metric | Meaning |
|---|---|
| `volume_ml` | cavity air volume |
| `mca_mm2` | **minimal cross-sectional area** — the classic constriction metric |
| `mca_ap_position_mm` | where the MCA sits, measured from the anterior naris |
| `mca_location` | anterior / middle / posterior third |
| `mca_at_terminal_slice` | whether the MCA is an end-narrowing vs a focal internal stenosis |
| `area_profile` | full area-vs-AP-distance curve |

And the bilateral summary: `mca_ratio` = min(L,R)/max(L,R) and
`more_obstructed_side` — the asymmetry that flags a unilaterally obstructed
airway (deviated septum, turbinate hypertrophy). Ratio near 1.0 = symmetric;
toward 0 = strongly one-sided.

## Run it

```powershell
# NasalSeg labeled case (works today, no GPU)
py -3.12 scripts/geometry_report.py --case P001 --data-root data

# Any head CT via the trained nnU-Net (no expert labels needed)
py -3.12 scripts/geometry_report.py --image path/to/ct.nii.gz --mask-source nnunet
```

Example (NasalSeg expert labels):

| Case | L vol | R vol | L MCA | R MCA | L/R ratio | flag |
|------|------:|------:|------:|------:|----------:|------|
| P001 | 15.0 mL | 14.1 mL | 19.9 mm² | 19.0 mm² | 0.96 | symmetric |
| P002 | 20.3 mL | 21.7 mL | 24.0 mm² | 36.9 mm² | 0.65 | left narrower |
| P065 | 13.7 mL | 13.1 mL |  9.1 mm² | 19.6 mm² | 0.46 | left obstructed |

## Method & honest limitations

- **Cross-sectional area = actual lumen voxel count per coronal (fixed-y)
  slice × pixel area**, not the π·r² distance-to-wall disk used in
  `nasal_passage.cross_sections_along_centerline`. That disk approximation
  collapses for slit-shaped passages, and the nasal valve — the usual MCA
  location — is a tall thin slit, so slice-area is the faithful primitive
  here (`src/sinus_cfd/passage_metrics.py`).
- **AP axis is assumed to be array axis y** (valid for axial-acquired
  NasalSeg crops). Coronal slices are perpendicular to the dominant
  anterior-posterior airflow direction. Where the airway turns vertical
  (nasopharynx) this assumption weakens, but the MCA is anterior, so it
  doesn't affect the reported number.
- **Profiles are typically unimodal** (widen from naris to a mid-cavity peak,
  narrow to the choana) with no separate internal notch, so the MCA usually
  sits at the anterior or posterior end of the airway body. That's reported
  honestly (`mca_at_terminal_slice`, `mca_location`) rather than hidden — it
  represents genuine end narrowing, not a fabricated focal stenosis. NasalSeg's
  nasal-cavity label tends to start near the nasal-valve region, so the
  anterior MCA is clinically meaningful.
- The **L/R asymmetry ratio is the most robust output** — it compares each
  side's narrowest real cross-section and is insensitive to the exact AP
  location of the minimum.

## How this connects

Runs on either segmentation source, so it's the first metric that works on a
real new-patient CT (via `--mask-source nnunet`) rather than only NasalSeg's
pre-labeled cases. Volume + MCA + asymmetry are also the pre/post quantities a
later virtual-surgery loop (Stage 4) would compare.
