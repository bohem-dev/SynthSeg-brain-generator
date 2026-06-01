"""Stage-by-stage visualisation of the BrainAugmentor pipeline.

Replays the generation pipeline one stage at a time and saves a snapshot after
each step so you can see what each stage does to the label map and image:

    0. input label map
    1. spatial deformation (affine + diffeomorphic elastic)
    2. random crop
    3. random flip
    4. conditional GMM sampling
    5. bias field corruption
    6. intensity augmentation (clip / normalise / gamma)
    7. resolution simulation (blur + downsample/resample)
    8. convert labels

Saves one combined figure (stages.png) plus per-stage PNGs under <out>/stages/.

Usage:
    python scripts/visualize.py \\
        --label data/sample/label.nii.gz \\
        --out /tmp/stages
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import brain_augmentor as ba


def _centre_slices(vol):
    s0, s1, s2 = vol.shape[:3]
    return [vol[s0 // 2], vol[:, s1 // 2], vol[:, :, s2 // 2]]


def _save_panel(vol, kind, title, out_path):
    planes = ["axial", "coronal", "sagittal"]
    sl = _centre_slices(vol)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for c in range(3):
        if kind == "label":
            axes[c].imshow(sl[c], cmap="tab20", origin="lower", interpolation="nearest")
        else:
            axes[c].imshow(sl[c], cmap="gray", origin="lower")
        axes[c].set_title(f"{title} | {planes[c]}", fontsize=8)
        axes[c].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_combined(snapshots, out_path):
    planes = ["axial", "coronal", "sagittal"]
    n = len(snapshots)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    fig.suptitle("BrainAugmentor -- stage by stage", fontsize=13)
    for r, (title, vol, kind) in enumerate(snapshots):
        sl = _centre_slices(vol)
        for c in range(3):
            ax = axes[r][c]
            if kind == "label":
                ax.imshow(sl[c], cmap="tab20", origin="lower", interpolation="nearest")
            else:
                ax.imshow(sl[c], cmap="gray", origin="lower")
            ax.set_title(f"{r}. {title} | {planes[c]}", fontsize=7)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visualise BrainAugmentor stages")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    stage_dir = out_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    sample = ba.load_sample(args.label, None)
    label = sample["label"]
    affine = sample["affine"]
    atlas_res = sample["vox_size"]
    gen_labels = label.unique().cpu().numpy().astype(np.int64)
    gen_classes = np.arange(len(gen_labels), dtype=np.int64)
    n_classes = len(gen_labels)
    n_neutral = len(gen_labels)

    snapshots = []

    def snap(title, vol, kind):
        v = vol.detach().cpu().numpy()
        snapshots.append((title, v, kind))
        idx = len(snapshots) - 1
        _save_panel(v, kind, title, stage_dir / f"stage_{idx}_{title.replace(' ', '_')}.png")
        if kind == "label":
            print(f"  stage {idx}: {title:<24} labels={len(np.unique(v))}")
        else:
            print(f"  stage {idx}: {title:<24} range=[{v.min():.3f},{v.max():.3f}] mean={v.mean():.3f}")

    snap("input label", label, "label")

    label, _ = ba.random_spatial_deformation(
        label, None, scaling_bounds=0.2, rotation_bounds=15.0, shearing_bounds=0.012,
        translation_bounds=False, nonlin_std=4.0, nonlin_scale=0.04,
    )
    snap("spatial deformation", label, "label")

    label, _ = ba.random_crop(label, None, None)
    snap("crop", label, "label")

    label, _ = ba.random_flip(label, None, affine, flipping=True,
                              label_list=gen_labels, n_neutral_labels=n_neutral)
    snap("flip", label, "label")

    means, stds = ba.build_gmm_params(n_classes, gen_classes, None, None, "uniform")
    image = ba.sample_conditional_gmm(label, gen_labels, means, stds)
    snap("GMM synth", image, "image")

    image = ba.bias_field_corruption(image, bias_field_std=0.7, bias_scale=0.025)
    snap("bias field", image, "image")

    image = ba.intensity_augmentation(image, clip=300, gamma_std=0.5)
    snap("intensity aug", image, "image")

    image = ba.simulate_resolution(
        image, atlas_res, tuple(label.shape), randomise_res=True,
        max_res_iso=4.0, max_res_aniso=8.0,
    )
    snap("resolution sim", image, "image")

    label = ba.convert_labels(label, gen_labels, None)
    snap("final label", label, "label")

    _save_combined(snapshots, out_dir / "stages.png")
    print(f"\nCombined figure -> {out_dir / 'stages.png'}")
    print(f"Per-stage PNGs  -> {stage_dir}/stage_*.png")


if __name__ == "__main__":
    main()
