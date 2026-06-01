"""Validation suite for the BrainAugmentor port.

Per-stage intensity progression, distributional comparison against SynthSeg,
and invariant tests (hard pass/fail). Exits non-zero if any invariant fails.

Usage:
    python scripts/validate.py \\
        --label data/sample/label.nii.gz \\
        --out /tmp/validate --n 3
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
from scipy.stats import ks_2samp

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import brain_augmentor as ba
from brain_augmentor.analysis import (
    make_augmentor,
    per_label_intensity,
    dice_vs_input,
    run_synthseg,
    _SYNTHSEG_REPO,
    _SYNTHSEG_PY,
)


def per_stage_progression(label_path, seed=0):
    """Run the pipeline stage by stage, logging intensity stats at each step."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    s = ba.load_sample(label_path, None)
    label, affine, atlas_res = s["label"], s["affine"], s["vox_size"]
    gen_labels = label.unique().cpu().numpy().astype(np.int64)
    gen_classes = np.arange(len(gen_labels), dtype=np.int64)

    stages = []

    def log(name, vol, kind):
        v = vol.detach().cpu().numpy()
        if kind == "image":
            stages.append({"stage": name, "min": float(v.min()), "max": float(v.max()),
                            "mean": float(v.mean())})
        else:
            stages.append({"stage": name, "n_labels": int(len(np.unique(v)))})

    label, _ = ba.random_spatial_deformation(label, None, nonlin_std=4.0, nonlin_scale=0.04)
    log("deformed label", label, "label")
    label, _ = ba.random_flip(label, None, affine, flipping=True,
                              label_list=gen_labels, n_neutral_labels=len(gen_labels))
    means, stds = ba.build_gmm_params(len(gen_labels), gen_classes, None, None, "uniform")
    image = ba.sample_conditional_gmm(label, gen_labels, means, stds)
    log("GMM synth", image, "image")
    image = ba.bias_field_corruption(image, 0.7, 0.025)
    log("bias field", image, "image")
    image = ba.intensity_augmentation(image, clip=300, gamma_std=0.5)
    log("intensity aug", image, "image")
    image = ba.simulate_resolution(image, atlas_res, tuple(label.shape),
                                   randomise_res=True, max_res_iso=4.0, max_res_aniso=8.0)
    log("resolution sim", image, "image")
    return stages


def jacobian_det(shape=(96, 96, 96), nonlin_std=4.0, nonlin_scale=0.04, seed=0):
    """Compute Jacobian determinant of our diffeomorphic elastic field at shape."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n_dims = len(shape)
    small = [int(np.ceil(shape[i] * nonlin_scale)) for i in range(n_dims)]
    std = random.uniform(0.0, nonlin_std)
    field = torch.randn(*small, n_dims) * std
    resize_shape = [max(int(shape[i] / 2), small[i]) for i in range(n_dims)]
    field = ba._resize(field, resize_shape, "linear")
    field = ba._integrate_vec(field, nb_steps=7)
    disp = ba._resize(field, list(shape), "linear")
    mesh = ba._meshgrid(shape)
    phi = (mesh + disp).numpy()
    g0 = np.stack(np.gradient(phi[..., 0]), axis=-1)
    g1 = np.stack(np.gradient(phi[..., 1]), axis=-1)
    g2 = np.stack(np.gradient(phi[..., 2]), axis=-1)
    J = np.stack([g0, g1, g2], axis=-2)
    return np.linalg.det(J)


def gradient_energy(vol):
    g = np.gradient(vol.astype(np.float32))
    return float(np.mean(sum(gi ** 2 for gi in g)))


def run_invariant_tests(label_path, ours_samples, ss_samples, label_in, out_dir):
    tests = []
    in_set = set(int(x) for x in np.unique(label_in))

    for mode, samples in [("augmentor", ours_samples), ("synthseg", ss_samples)]:
        if not samples:
            continue
        ok = all(set(int(x) for x in torch.unique(lbl).tolist()) <= in_set for _, lbl in samples)
        tests.append((f"{mode}: output labels subset of input", ok,
                      "all output labels present in input" if ok else "FOUND new labels"))

    for mode, samples in [("augmentor", ours_samples), ("synthseg", ss_samples)]:
        if not samples:
            continue
        mn = min(float(img.min()) for img, _ in samples)
        mx = max(float(img.max()) for img, _ in samples)
        ok = mn >= -1e-3 and mx <= 1.0 + 1e-3
        tests.append((f"{mode}: image in [0,1]", ok, f"observed range [{mn:.3f}, {mx:.3f}]"))

    det = jacobian_det()
    frac_folded = float(np.mean(det <= 0))
    ok = frac_folded < 0.001
    tests.append(("augmentor: deformation diffeomorphic (Jac det>0)", ok,
                  f"folded voxel fraction = {frac_folded:.5f} (det range "
                  f"[{det.min():.3f}, {det.max():.3f}])"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(det.ravel(), bins=80, color="steelblue")
    ax.axvline(0, color="red", lw=1)
    ax.set_title("Jacobian determinant of elastic deformation")
    ax.set_xlabel("det(J)"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(out_dir / "jacobian_hist.png", dpi=120); plt.close(fig)

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    s = ba.load_sample(label_path, None)
    lab, _ = ba.random_spatial_deformation(s["label"], None, nonlin_std=0.0,
                                           scaling_bounds=False, rotation_bounds=False,
                                           shearing_bounds=False)
    gl = lab.unique().cpu().numpy().astype(np.int64)
    m, st = ba.build_gmm_params(len(gl), np.arange(len(gl)), None, None, "uniform")
    img0 = ba.intensity_augmentation(ba.sample_conditional_gmm(lab, gl, m, st),
                                     clip=300, gamma_std=0.0)
    e_before = gradient_energy(img0.numpy())
    img1 = ba.simulate_resolution(img0, s["vox_size"], tuple(lab.shape),
                                  randomise_res=True, max_res_iso=4.0, max_res_aniso=8.0)
    e_after = gradient_energy(img1.numpy())
    ok = e_after < e_before
    tests.append(("augmentor: resolution sim reduces HF energy", ok,
                  f"gradient energy {e_before:.4f} -> {e_after:.4f}"))
    return tests


def main():
    p = argparse.ArgumentParser(description="Validate BrainAugmentor")
    p.add_argument("--label", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-res", type=float, default=1.0, dest="target_res")
    p.add_argument("--skip-synthseg", action="store_true", dest="skip_synthseg")
    p.add_argument("--synthseg-repo", default=str(_SYNTHSEG_REPO), dest="synthseg_repo")
    p.add_argument("--synthseg-python", default=str(_SYNTHSEG_PY), dest="synthseg_python")
    args = p.parse_args()

    args.label = str(Path(args.label).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_in = np.asarray(nib.load(args.label).dataobj, dtype=np.int32)

    print(f"Generating {args.n} samples with augmentor...")
    aug = make_augmentor(output_shape=None)
    ours = []
    for i in range(args.n):
        seed = args.seed + i * 100
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        ours.append(aug(args.label, None))

    ss = None
    if not args.skip_synthseg:
        print(f"Generating {args.n} samples with original SynthSeg...")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        ss = run_synthseg(args.label, args.n, out_dir / "synthseg_tmp",
                          args.synthseg_repo, args.synthseg_python, args.target_res)
    ss = ss or []

    print("\n[stage progression] augmentor pipeline (seed 0):")
    stages = per_stage_progression(args.label, seed=0)
    for st in stages:
        if "mean" in st:
            print(f"    {st['stage']:<18} range=[{st['min']:.3f},{st['max']:.3f}] mean={st['mean']:.3f}")
        else:
            print(f"    {st['stage']:<18} n_labels={st['n_labels']}")

    print("\n[distributional comparison]")
    rng = np.random.default_rng(0)

    def pooled_intensities(samples, per=200_000):
        vals = []
        for img, _ in samples:
            v = img.numpy().ravel()
            vals.append(rng.choice(v, size=min(per, v.size), replace=False))
        return np.concatenate(vals) if vals else np.array([])

    ks_stat = ks_p = None
    ours_pool = pooled_intensities(ours)
    if ss:
        ss_pool = pooled_intensities(ss)
        ks_stat, ks_p = ks_2samp(ours_pool, ss_pool)
        print(f"    intensity KS: stat={ks_stat:.4f} p={ks_p:.3e}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ours_pool, bins=60, range=(0, 1), density=True, alpha=0.5, label="augmentor")
        ax.hist(ss_pool, bins=60, range=(0, 1), density=True, alpha=0.5, label="synthseg")
        ax.set_title("Pooled intensity distribution"); ax.set_xlabel("normalised intensity")
        ax.legend(); plt.tight_layout()
        plt.savefig(out_dir / "intensity_distribution.png", dpi=120); plt.close(fig)

    def mean_label_intensity(samples):
        acc = {}
        for img, lbl in samples:
            for lid, (mean, _) in per_label_intensity(img, lbl).items():
                acc.setdefault(lid, []).append(mean)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    pearson = None
    if ss:
        mi_o, mi_s = mean_label_intensity(ours), mean_label_intensity(ss)
        common = sorted(set(mi_o) & set(mi_s))
        xo = np.array([mi_o[l] for l in common]); xs = np.array([mi_s[l] for l in common])
        if len(common) > 2:
            pearson = float(np.corrcoef(xo, xs)[0, 1])
            print(f"    per-label mean-intensity Pearson r = {pearson:.4f} (over {len(common)} labels)")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(xs, xo, s=8, alpha=0.6)
            lim = [0, max(xo.max(), xs.max()) * 1.05]
            ax.plot(lim, lim, "r--", lw=1)
            ax.set_xlabel("synthseg mean intensity"); ax.set_ylabel("augmentor mean intensity")
            ax.set_title(f"Per-label mean intensity (r={pearson:.3f})")
            plt.tight_layout(); plt.savefig(out_dir / "per_label_scatter.png", dpi=120); plt.close(fig)

    def mean_dice(samples):
        d = [v for _, lbl in samples for v in dice_vs_input(lbl, label_in).values()]
        return float(np.mean(d)) if d else None

    print(f"    mean Dice(deformed, input): augmentor={mean_dice(ours)}  synthseg={mean_dice(ss)}")

    print("\n[invariant tests]")
    tests = run_invariant_tests(args.label, ours, ss, label_in, out_dir)
    n_fail = 0
    for name, ok, detail in tests:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"    [{flag}] {name:<46} {detail}")

    report = {
        "n_samples": args.n,
        "synthseg_available": bool(ss),
        "per_stage": stages,
        "ks_intensity": {"stat": ks_stat, "p": ks_p} if ks_stat is not None else None,
        "per_label_pearson_r": pearson,
        "dice_vs_input": {"augmentor": mean_dice(ours), "synthseg": mean_dice(ss)},
        "invariant_tests": [{"name": n, "passed": bool(o), "detail": d} for n, o, d in tests],
        "n_failed": n_fail,
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out_dir / 'validation_report.json'}")
    print(f"\n{'ALL INVARIANTS PASSED' if n_fail == 0 else f'{n_fail} INVARIANT TEST(S) FAILED'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
