# Training nnU-Net on NasalSeg via Google Colab

This machine has no CUDA GPU (Intel Iris Plus integrated graphics only), and
nnU-Net's default `3d_fullres` training schedule (~1000 epochs) is not
practical on CPU. This doc covers training on a free/cheap Colab GPU instead,
using `notebooks/train_nnunet_colab.ipynb`.

## Why nnU-Net at all

`docs/stage1_segmentation_baseline.md` measured the classical HU-threshold
pipeline against NasalSeg ground truth: ~0.25 Dice, and neither exterior-air
masking nor nostril-seeded growth improved it. The blocking issue is
structural — nothing in the classical pipeline distinguishes nasal-cavity air
from paranasal-sinus air within the same connected air blob. A learned
segmentation model is the next thing to try.

## Step 1 — prepare the dataset locally (done, CPU only)

```powershell
py -3.12 scripts\download_nasalseg.py
py -3.12 scripts\prepare_nnunet_nasalseg.py --nasalseg-root data
```

Produces `data/nnUNet_raw/Dataset501_NasalSeg/` (130 cases, ~216 MB): NIfTI
images/labels, `dataset.json`, and a 5-fold split file. NasalSeg's own zip
happens to extract `images/`/`labels/` directly under `data/` rather than
`data/NasalSeg/` — hence `--nasalseg-root data`.

## Step 2 — upload to Google Drive

Zip `data/nnUNet_raw/Dataset501_NasalSeg/` (already done by the assistant:
`data/NasalSeg_nnUNet_raw.zip`, gitignored, not committed) and upload it to
your Google Drive, e.g. `MyDrive/sinus_cfd/NasalSeg_nnUNet_raw.zip`. That
path is what the notebook expects by default (`DRIVE_ZIP` variable, edit it
if you put it elsewhere).

**If you already uploaded a zip and hit `nnUNetv2_plan_and_preprocess`
errors** (`Spacing mismatch between segmentation and corresponding images`,
or a wall of `Origin`/`Direction mismatch` warnings followed by
`RuntimeError: Some images have errors`): that was a real bug in the
original conversion, now fixed — 12-14 of the 130 NasalSeg label NRRDs carry
a spacing/origin/direction in their own header that disagrees with the
paired image, even though the voxel array size always matches. `prepare_nnunet_nasalseg.py`
now stamps the image's geometry onto the label before writing (safe, since
size always agrees — the intended correspondence is voxel-index-to-voxel-index,
not the label's own header). **Re-run** `prepare_nnunet_nasalseg.py`, re-zip,
and **re-upload to Drive** before retrying the notebook.

## Step 3 — run the notebook

Open `notebooks/train_nnunet_colab.ipynb` in Colab — either upload it
directly, or once this branch is pushed:

```
https://colab.research.google.com/github/alexchouck-hash/Sinus_CFD/blob/MVP/notebooks/train_nnunet_colab.ipynb
```

Runtime → Change runtime type → GPU (T4 on the free tier is fine to start).
The notebook: mounts Drive, installs `nnunetv2`, stages the dataset onto
Colab's local disk (faster than Drive for nnU-Net's small-file I/O), runs
`nnUNetv2_plan_and_preprocess`, trains fold 0 of `3d_fullres`, and rsyncs
`nnUNet_results/` back to Drive so a session disconnect doesn't lose the
trained weights.

**Free-tier Colab disconnects** after ~90 min idle and has a session cap
(often ~12h, sometimes less). If training stops partway, rerun the training
cell with `--c` appended (`nnUNetv2_train 501 3d_fullres 0 --c`) to resume
from the last checkpoint rather than restarting.

MVP scope: **train fold 0 only**, not the full 5-fold CV — enough for a
usable model and one held-out validation Dice. Add folds 1-4 later for a
proper cross-validated estimate.

## Step 4 — bring the trained model back

Download `nnUNet_results/` (or just the best checkpoint) from Drive to
`data/nnUNet_results/` locally. Run inference:

```powershell
. .\data\nnUNet_raw\Dataset501_NasalSeg\set_nnunet_env.ps1
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 501 -c 3d_fullres -f 0
```

Then compare its Dice against the same ground truth
`scripts/evaluate_nasalseg_dice.py` uses (labels 1-3), so the learned model's
score is directly comparable to the ~0.25 classical baseline.

## Rough cost/time expectations

| | |
|---|---|
| Free Colab T4 | $0, but session limits/disconnects are the main friction |
| Colab Pro (~$10/mo) | Longer sessions, priority GPU access, faster A100/V100 sometimes available |
| Dataset size | Small crops (~50-200 voxels per axis), so even a T4 should train fold 0 in a few hours, not days |
| Do not commit | `data/nnUNet_raw`, `data/nnUNet_preprocessed`, `data/nnUNet_results`, and the zip are all gitignored — trained weights and dataset stay out of git |
