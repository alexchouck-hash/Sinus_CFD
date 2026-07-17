"""
nnU-Net inference wrapper: run a trained model on one CT volume.

Shells out to the `nnUNetv2_predict` CLI rather than nnU-Net's Python API,
since the CLI is the documented, stable entry point across nnU-Net internal
refactors. Requires `pip install -r requirements-nn.txt` and the
`nnUNet_results` (and typically `nnUNet_raw`/`nnUNet_preprocessed`)
environment variables set — see docs/nnunet_colab_training.md for how the
weights get here after training on Colab.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_DATASET_ID = 501
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_FOLD = 0
DEFAULT_TRAINER = "nnUNetTrainer_250epochs"
DEFAULT_PLANS = "nnUNetPlans"


def predict_labels(
    image: sitk.Image,
    dataset_id: int = DEFAULT_DATASET_ID,
    configuration: str = DEFAULT_CONFIGURATION,
    fold: int = DEFAULT_FOLD,
    trainer: str = DEFAULT_TRAINER,
    plans: str = DEFAULT_PLANS,
) -> np.ndarray:
    """
    Run nnU-Net inference on one CT volume, returning a (z, y, x) label array
    on the same voxel grid as ``image`` (nnU-Net resamples its internal
    prediction back to the input's native spacing before writing output).

    Raises RuntimeError with the CLI's stderr if inference fails, rather than
    letting a cryptic subprocess/file-not-found error propagate.
    """
    if shutil.which("nnUNetv2_predict") is None:
        raise RuntimeError(
            "nnUNetv2_predict not found on PATH. Install with "
            "`py -3.12 -m pip install -r requirements-nn.txt` and set "
            "nnUNet_raw / nnUNet_preprocessed / nnUNet_results (see "
            "docs/nnunet_colab_training.md)."
        )

    with tempfile.TemporaryDirectory(prefix="sinus_cfd_nnunet_") as tmp:
        tmp_path = Path(tmp)
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # nnU-Net expects CASE_0000.nii.gz for a single-modality dataset.
        sitk.WriteImage(image, str(in_dir / "case_0000.nii.gz"), useCompression=True)

        cmd = [
            "nnUNetv2_predict",
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-d", str(dataset_id),
            "-c", configuration,
            "-f", str(fold),
            "-tr", trainer,
            "-p", plans,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"nnUNetv2_predict failed (exit {result.returncode}):\n"
                f"{result.stderr[-4000:]}"
            )

        candidates = sorted(out_dir.glob("*.nii.gz"))
        if not candidates:
            raise RuntimeError(
                f"nnUNetv2_predict produced no output in {out_dir} "
                f"(stdout tail: {result.stdout[-2000:]})"
            )

        pred_img = sitk.ReadImage(str(candidates[0]))
        return sitk.GetArrayFromImage(pred_img)
