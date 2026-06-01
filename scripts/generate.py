#!/usr/bin/env python
"""Generate synthetic MRI images from a label map.

Supports two backends:
  augmentor   PyTorch reimplementation (default, no TF required)
  synthseg    Original SynthSeg TF BrainGenerator (requires separate venv)

The synthseg backend is invoked via subprocess using the Python interpreter
from the synthseg venv (see --synthseg-python). Outputs are saved as NIfTI.

Usage:
    python scripts/generate.py --label data/sample/label.nii.gz --out /tmp/out
    python scripts/generate.py --label data/sample/label.nii.gz --out /tmp/out \\
        --backend synthseg --synthseg-python venv_synthseg/bin/python
"""

import argparse
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from brain_augmentor import BrainAugmentor
from brain_augmentor.analysis import run_synthseg, _SYNTHSEG_REPO, _SYNTHSEG_PY


def _save(array, affine, path):
    nib.save(nib.Nifti1Image(array, affine), str(path))


def run_augmentor(args):
    aug = BrainAugmentor(
        output_shape=tuple(args.output_shape) if args.output_shape else None,
        scaling_bounds=args.scaling,
        rotation_bounds=args.rotation,
        shearing_bounds=args.shearing,
        nonlin_std=args.nonlin_std,
        nonlin_scale=args.nonlin_scale,
        bias_field_std=args.bias_std,
        bias_scale=args.bias_scale,
        gamma_std=args.gamma_std,
        flipping=not args.no_flip,
        randomise_res=not args.no_randomise_res,
        max_res_iso=args.max_res_iso,
        max_res_aniso=args.max_res_aniso,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = str(Path(args.label).resolve())

    for i in range(args.n):
        seed = args.seed + i * 100
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        image, label = aug(label_path, None)
        img_np = image.float().numpy()
        lbl_np = label.numpy().astype(np.int32)
        _save(img_np, np.eye(4), out_dir / f"image_{i:03d}.nii.gz")
        _save(lbl_np, np.eye(4), out_dir / f"label_{i:03d}.nii.gz")
        print(f"  sample {i:3d}: image range=[{img_np.min():.3f},{img_np.max():.3f}]  "
              f"labels={len(np.unique(lbl_np))}")

    print(f"\nOutputs in: {out_dir}")


def run_synthseg_backend(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = str(Path(args.label).resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = run_synthseg(
        label_path,
        args.n,
        out_dir,
        repo=args.synthseg_repo,
        python_exe=args.synthseg_python,
        target_res=args.target_res,
    )
    if results is None:
        print("SynthSeg backend failed or is unavailable.")
        sys.exit(1)

    for i, (img, lbl) in enumerate(results):
        print(f"  sample {i:3d}: image range=[{img.min():.3f},{img.max():.3f}]  "
              f"labels={len(torch.unique(lbl))}")
    print(f"\nOutputs in: {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic MRI images from a label map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label", "-l", required=True, help="input label map (.nii.gz)")
    p.add_argument("--out", "-o", required=True, help="output directory")
    p.add_argument("--n", type=int, default=1, help="number of samples to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["augmentor", "synthseg"], default="augmentor",
                   help="augmentor: PyTorch port (default); synthseg: original TF code")

    aug = p.add_argument_group("augmentor parameters (ignored for synthseg backend)")
    aug.add_argument("--scaling", type=float, default=0.2)
    aug.add_argument("--rotation", type=float, default=15.0, help="degrees")
    aug.add_argument("--shearing", type=float, default=0.012)
    aug.add_argument("--nonlin-std", type=float, default=4.0, dest="nonlin_std")
    aug.add_argument("--nonlin-scale", type=float, default=0.04, dest="nonlin_scale")
    aug.add_argument("--bias-std", type=float, default=0.7, dest="bias_std")
    aug.add_argument("--bias-scale", type=float, default=0.025, dest="bias_scale")
    aug.add_argument("--gamma-std", type=float, default=0.5, dest="gamma_std")
    aug.add_argument("--no-flip", action="store_true")
    aug.add_argument("--no-randomise-res", action="store_true")
    aug.add_argument("--max-res-iso", type=float, default=4.0, dest="max_res_iso")
    aug.add_argument("--max-res-aniso", type=float, default=8.0, dest="max_res_aniso")
    aug.add_argument("--output-shape", type=int, nargs=3, metavar="N",
                     dest="output_shape", help="crop shape, e.g. 128 128 128")

    ss = p.add_argument_group("synthseg backend parameters")
    ss.add_argument("--synthseg-python", default=str(_SYNTHSEG_PY), dest="synthseg_python",
                    help="path to the synthseg venv Python interpreter")
    ss.add_argument("--synthseg-repo", default=str(_SYNTHSEG_REPO), dest="synthseg_repo",
                    help="path to the synthseg source directory")
    ss.add_argument("--target-res", type=float, default=1.0, dest="target_res",
                    help="target resolution in mm for the synthseg backend")

    args = p.parse_args()

    if args.backend == "augmentor":
        run_augmentor(args)
    else:
        run_synthseg_backend(args)


if __name__ == "__main__":
    main()
