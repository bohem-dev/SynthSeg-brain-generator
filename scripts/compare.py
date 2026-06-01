"""Compare BrainAugmentor (PyTorch) against SynthSeg BrainGenerator (TensorFlow).

Both augmentors go label map -> synthetic MRI. They draw random numbers in
different libraries so sample-for-sample matching is impossible; distributions
are compared instead:

  1. Whole-image intensity statistics and histograms (over N samples each).
  2. Per-label intensity mean/std.
  3. Dice overlap of each augmentor's deformed label map against the input.
  4. Visual PNGs: central axial/coronal/sagittal slices, histogram, per-label plot.

The original SynthSeg runs in its own venv via subprocess to avoid TF/torch
conflicts. See README for setup instructions.

Usage:
    python scripts/compare.py \\
        --label data/sample/label.nii.gz \\
        --image data/sample/t1.nii.gz \\
        --out /tmp/compare --n 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from brain_augmentor.analysis import (
    make_augmentor,
    per_label_intensity,
    dice_vs_input,
    run_synthseg,
    _SYNTHSEG_REPO,
    _SYNTHSEG_PY,
)


def image_stats(image):
    v = image.float().numpy().ravel()
    v = v[np.isfinite(v)]
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
    }


def intensity_histogram(image, n_bins=64):
    v = image.float().numpy().ravel()
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=n_bins, range=(0.0, 1.0))
    return counts.tolist(), edges.tolist()


def _centre_slices(vol):
    s0, s1, s2 = vol.shape[:3]
    return [vol[s0 // 2], vol[:, s1 // 2], vol[:, :, s2 // 2]]


def save_comparison_grid(panels, out_path, sample_idx):
    planes = ["axial", "coronal", "sagittal"]
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows), squeeze=False)
    fig.suptitle(f"Sample {sample_idx}", fontsize=12)
    for r, (title, vol, kind) in enumerate(panels):
        sl = _centre_slices(vol)
        for c in range(3):
            ax = axes[r][c]
            if kind == "label":
                ax.imshow(sl[c], cmap="tab20", origin="lower", interpolation="nearest")
            else:
                ax.imshow(sl[c], cmap="gray", origin="lower")
            ax.set_title(f"{title} | {planes[c]}", fontsize=7)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_histogram_comparison(hist_by_mode, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, samples in hist_by_mode.items():
        counts = np.array([s[0] for s in samples], dtype=float).mean(axis=0)
        edges = np.array(samples[0][1])
        centres = (edges[:-1] + edges[1:]) / 2
        total = counts.sum() or 1.0
        ax.plot(centres, counts / total, label=mode, alpha=0.8)
    ax.set_xlabel("Normalised intensity")
    ax.set_ylabel("Frequency (normalised)")
    ax.set_title("Mean intensity histogram")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_per_label_means(label_means_by_mode, out_path, max_labels=40):
    label_ids = sorted(set(l for m in label_means_by_mode.values() for l in m))[:max_labels]
    modes = list(label_means_by_mode.keys())
    fig, ax = plt.subplots(figsize=(max(8, len(label_ids) * 0.3), 5))
    x = np.arange(len(label_ids))
    width = 0.8 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        means = [label_means_by_mode[mode].get(l, (0.0, 0.0))[0] for l in label_ids]
        ax.bar(x + i * width, means, width=width, label=mode, alpha=0.8)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels([str(l) for l in label_ids], rotation=90, fontsize=6)
    ax.set_ylabel("Mean intensity")
    ax.set_title(f"Per-label mean intensity (first {len(label_ids)} labels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare BrainAugmentor vs SynthSeg")
    p.add_argument("--label", required=True, help="input label map (.nii.gz)")
    p.add_argument("--image", default=None, help="reference T1 (.nii.gz) for overlays")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n", type=int, default=3, help="samples per augmentor")
    p.add_argument("--target-res", type=float, default=1.0, dest="target_res")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthseg-repo", default=str(_SYNTHSEG_REPO), dest="synthseg_repo")
    p.add_argument("--synthseg-python", default=str(_SYNTHSEG_PY), dest="synthseg_python")
    p.add_argument("--skip-synthseg", action="store_true", dest="skip_synthseg",
                   help="run only our augmentor")
    args = p.parse_args()

    args.label = str(Path(args.label).resolve())
    if args.image:
        args.image = str(Path(args.image).resolve())

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_dir = out_dir / "volumes"
    vol_dir.mkdir(exist_ok=True)

    label_in = np.asarray(nib.load(args.label).dataobj, dtype=np.int32)
    print(f"Input label map: shape={label_in.shape}, {len(np.unique(label_in))} labels")

    ref_t1 = None
    if args.image:
        ref_t1 = np.asarray(nib.load(args.image).dataobj, dtype=np.float32)
        lo, hi = ref_t1.min(), ref_t1.max()
        ref_t1 = (ref_t1 - lo) / (hi - lo + 1e-8)

    results = {}
    samples = {}

    def _accumulate(mode, gen_iter):
        results[mode] = {"stats": [], "hist": [], "label_int": [], "dice": []}
        samples[mode] = []
        for i, (img, lbl) in enumerate(gen_iter):
            if img.ndim == 4:
                img = img[..., 0]
            results[mode]["stats"].append(image_stats(img))
            results[mode]["hist"].append(intensity_histogram(img))
            results[mode]["label_int"].append(per_label_intensity(img, lbl))
            results[mode]["dice"].append(dice_vs_input(lbl, label_in))
            if i == 0:
                samples[mode].append((img, lbl))
            print(f"  {mode} sample {i}: mean={results[mode]['stats'][-1]['mean']:.3f} "
                  f"std={results[mode]['stats'][-1]['std']:.3f}")

    print("\nRunning: augmentor (BrainAugmentor)")
    aug = make_augmentor(output_shape=None)

    def _augmentor_iter():
        for i in range(args.n):
            seed = args.seed + i * 100
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            yield aug(args.label, None)

    _accumulate("augmentor", _augmentor_iter())

    if not args.skip_synthseg:
        print("\nRunning: synthseg (original TF BrainGenerator)")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        ss = run_synthseg(
            args.label, args.n, out_dir / "synthseg_tmp",
            args.synthseg_repo, args.synthseg_python, args.target_res,
        )
        if ss is not None:
            _accumulate("synthseg", iter(ss))
        else:
            print("  [synthseg] unavailable -- comparison will show augmentor only.")

    summary = {}
    for mode, data in results.items():
        keys = data["stats"][0].keys()
        summary[mode] = {
            k: {
                "mean": float(np.mean([s[k] for s in data["stats"]])),
                "std": float(np.std([s[k] for s in data["stats"]])),
            }
            for k in keys
        }
        dvals = [d for dd in data["dice"] for d in dd.values()]
        summary[mode]["dice_vs_input"] = {
            "mean": float(np.mean(dvals)) if dvals else None,
            "std": float(np.std(dvals)) if dvals else None,
        }
    (out_dir / "stats_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nStats -> {out_dir / 'stats_summary.json'}")

    label_means = {}
    for mode, data in results.items():
        agg = {}
        all_ids = set(l for s in data["label_int"] for l in s)
        for lid in all_ids:
            means = [s[lid][0] for s in data["label_int"] if lid in s]
            stds = [s[lid][1] for s in data["label_int"] if lid in s]
            agg[lid] = (float(np.mean(means)), float(np.mean(stds)))
        label_means[mode] = agg
    (out_dir / "per_label_intensity.json").write_text(
        json.dumps({m: {str(k): v for k, v in d.items()} for m, d in label_means.items()}, indent=2)
    )

    n_grids = max(len(s) for s in samples.values())
    for i in range(n_grids):
        panels = []
        if ref_t1 is not None:
            panels.append(("original MRI (T1)", ref_t1, "image"))
        panels.append(("input label map", label_in, "label"))
        for mode in samples:
            if i < len(samples[mode]):
                _, lbl = samples[mode][i]
                panels.append((f"{mode}: deformed label", lbl.numpy(), "label"))
        for mode in samples:
            if i < len(samples[mode]):
                img, _ = samples[mode][i]
                panels.append((f"{mode}: synth image", img.numpy(), "image"))
        save_comparison_grid(panels, out_dir / f"sample_{i:02d}_slices.png", i)
    print(f"Slice grids -> {out_dir}/sample_*_slices.png")

    save_histogram_comparison(
        {m: results[m]["hist"] for m in results}, out_dir / "histogram_comparison.png"
    )
    save_per_label_means(label_means, out_dir / "per_label_intensity.png")

    print("\n== Whole-image summary ==")
    print(f"{'mode':<12}{'mean':>9}{'std':>9}{'p5':>9}{'p95':>9}{'dice':>9}")
    print("-" * 57)
    for mode, agg in summary.items():
        dice = agg["dice_vs_input"]["mean"]
        dice_s = f"{dice:9.4f}" if dice is not None else f"{'n/a':>9}"
        print(f"{mode:<12}{agg['mean']['mean']:9.4f}{agg['std']['mean']:9.4f}"
              f"{agg['p5']['mean']:9.4f}{agg['p95']['mean']:9.4f}{dice_s}")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
