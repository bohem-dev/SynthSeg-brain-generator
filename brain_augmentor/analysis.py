"""Shared utilities for comparing and validating BrainAugmentor vs SynthSeg."""

import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .core import BrainAugmentor

_HERE = Path(__file__).resolve().parent.parent
_SYNTHSEG_REPO = _HERE / "synthseg"
_SYNTHSEG_PY = _HERE / "venv_synthseg" / "bin" / "python"


def make_augmentor(output_shape=None):
    """BrainAugmentor configured to match BrainGenerator defaults exactly."""
    return BrainAugmentor(
        output_shape=output_shape,
        scaling_bounds=0.2,
        rotation_bounds=15.0,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=4.0,
        nonlin_scale=0.04,
        bias_field_std=0.7,
        bias_scale=0.025,
        clip=300,
        gamma_std=0.5,
        flipping=True,
        randomise_res=True,
        max_res_iso=4.0,
        max_res_aniso=8.0,
    )


def per_label_intensity(image, label):
    """Mean/std of image intensity within each label region."""
    img = image.float().numpy().ravel()
    lbl = label.numpy().ravel()
    out = {}
    for lid in np.unique(lbl):
        vals = img[lbl == lid]
        if vals.size:
            out[int(lid)] = (float(vals.mean()), float(vals.std()))
    return out


def dice_vs_input(label_out, label_in_np):
    """Per-label Dice between a deformed output label map and the input label map."""
    out = label_out.numpy()
    if out.shape != label_in_np.shape:
        return {}
    result = {}
    for lid in np.unique(label_in_np):
        if lid == 0:
            continue
        a = label_in_np == lid
        b = out == lid
        denom = a.sum() + b.sum()
        if denom:
            result[int(lid)] = float(2.0 * (a & b).sum() / denom)
    return result


_RUNNER = """
import sys, os
sys.path.insert(0, {repo!r})
import numpy as np, nibabel as nib
from SynthSeg.brain_generator import BrainGenerator

gen = BrainGenerator(labels_dir={label!r}, target_res={target_res})
for i in range({n}):
    img, lbl = gen.generate_brain()
    nib.save(nib.Nifti1Image(np.squeeze(img).astype(np.float32), np.eye(4)),
             os.path.join({out!r}, "ss_image_%03d.nii.gz" % i))
    nib.save(nib.Nifti1Image(np.squeeze(lbl).astype(np.int32), np.eye(4)),
             os.path.join({out!r}, "ss_label_%03d.nii.gz" % i))
    print("  [synthseg] sample %d done" % i, flush=True)
print("SYNTHSEG_DONE", flush=True)
"""


def run_synthseg(label_path, n, out_dir, repo=None, python_exe=None, target_res=1.0):
    """Run the original TF BrainGenerator in its own venv; return list[(img,lbl)] or None."""
    if repo is None:
        repo = _SYNTHSEG_REPO
    if python_exe is None:
        python_exe = _SYNTHSEG_PY

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    python_exe = Path(python_exe)
    if not python_exe.exists():
        print(f"  [synthseg] python not found at {python_exe} -- skipping.")
        return None

    runner = out_dir / "_run_synthseg.py"
    runner.write_text(
        _RUNNER.format(
            repo=str(repo),
            label=str(label_path),
            out=str(out_dir),
            n=n,
            target_res=target_res,
        )
    )
    env = {**__import__("os").environ, "PYTHONPATH": str(repo)}
    proc = subprocess.run(
        [str(python_exe), str(runner)],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0 or "SYNTHSEG_DONE" not in proc.stdout:
        print("  [synthseg] FAILED:\n" + proc.stderr[-2000:])
        return None

    results = []
    for i in range(n):
        ip = out_dir / f"ss_image_{i:03d}.nii.gz"
        lp = out_dir / f"ss_label_{i:03d}.nii.gz"
        if not ip.exists() or not lp.exists():
            break
        img = np.squeeze(np.asarray(nib.load(ip).dataobj, dtype=np.float32))
        lbl = np.squeeze(np.asarray(nib.load(lp).dataobj, dtype=np.int32))
        lo, hi = float(img.min()), float(img.max())
        if hi > 1.0 + 1e-3 or lo < -1e-3:
            img = (img - lo) / (hi - lo + 1e-8)
        results.append((torch.from_numpy(img), torch.from_numpy(lbl)))
    return results or None
