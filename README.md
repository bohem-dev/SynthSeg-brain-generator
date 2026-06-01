# SynthSeg Brain Generator

A PyTorch reimplementation of the SynthSeg brain image generation pipeline, alongside the original TensorFlow code for comparison.

SynthSeg generates synthetic MRI images from label maps using a Gaussian Mixture Model (GMM). Given a segmentation label map, it applies random spatial deformations, samples per-label intensities from a GMM, adds a bias field, and simulates variable acquisition resolution. The result is a realistic-looking synthetic MRI with a paired label map.

This repository contains:

- `brain_augmentor/` -- PyTorch port (`BrainAugmentor`), faithful to the original pipeline step for step
- `synthseg/` -- original TF/Keras code isolated from [SynthSeg](https://github.com/BBillot/SynthSeg) (reference only)
- `scripts/` -- CLI tools for generation, comparison, and validation


## Installation

The PyTorch augmentor requires Python >= 3.10. The original SynthSeg code requires Python 3.11 and an older TensorFlow (see below).

Install the augmentor package and its dependencies:

```
pip install -e .
```

This installs the `brain-generate` command and the `brain_augmentor` package.


## Sample data

A small synthetic label map is included for quick testing. Generate it with:

```
python data/sample/create_sample.py
```

This creates `data/sample/label.nii.gz` (96x96x96, 14 labels).

Comparison scripts were originally tested with IXI subjects (e.g. `IXI002-Guys-0828-SEG.nii.gz`).


## Usage

### Generate synthetic images

```
python scripts/generate.py --label data/sample/label.nii.gz --out /tmp/out --n 3
```

Use the original SynthSeg backend instead (requires venv_synthseg, see below):

```
python scripts/generate.py \
    --label data/sample/label.nii.gz \
    --out /tmp/out \
    --backend synthseg \
    --synthseg-python venv_synthseg/bin/python
```

Key parameters (augmentor backend):

| flag | default | description |
|---|---|---|
| `--n` | 1 | number of samples |
| `--scaling` | 0.2 | random scaling range |
| `--rotation` | 15.0 | random rotation range (degrees) |
| `--nonlin-std` | 4.0 | elastic deformation std |
| `--bias-std` | 0.7 | bias field std |
| `--gamma-std` | 0.5 | gamma augmentation std |
| `--output-shape N N N` | None | crop to this shape after deformation |
| `--no-randomise-res` | off | disable resolution randomisation |

Run `python scripts/generate.py --help` for the full list.

### Visualise pipeline stages

```
python scripts/visualize.py --label data/sample/label.nii.gz --out /tmp/stages
```

Saves `stages.png` (all stages in one figure) and per-stage PNGs.

### Compare augmentor vs SynthSeg

```
python scripts/compare.py \
    --label data/sample/label.nii.gz \
    --out /tmp/compare \
    --n 5 \
    --skip-synthseg
```

Without `--skip-synthseg`, the script invokes the original TF code via subprocess (requires venv_synthseg).

### Validate

```
python scripts/validate.py \
    --label data/sample/label.nii.gz \
    --out /tmp/validate \
    --skip-synthseg
```

Runs invariant tests (diffeomorphic deformation, output range [0,1], label subset) and exits non-zero on failure.


## Original SynthSeg setup

The original code lives in `synthseg/` and requires TensorFlow. It cannot share a virtual environment with PyTorch due to dependency conflicts. Set up a separate environment:

```
python3.11 -m venv venv_synthseg
source venv_synthseg/bin/activate
pip install tensorflow==2.12.* keras==2.12.* nibabel numpy
```

Pass the interpreter path to any script that calls the SynthSeg backend:

```
--synthseg-python venv_synthseg/bin/python
```

The `synthseg/` directory is self-contained; no installation is needed beyond setting `PYTHONPATH=synthseg` when running it, which the scripts handle automatically.


## Repository layout

```
brain_augmentor/     PyTorch package
  core.py            full pipeline implementation
  analysis.py        shared utilities for comparison and validation
  cli.py             entry point for brain-generate command

synthseg/            original TensorFlow code (reference)
  SynthSeg/
  ext/lab2im/
  ext/neuron/

scripts/
  generate.py        main CLI (augmentor or synthseg backend)
  compare.py         distributional comparison between the two
  validate.py        invariant and statistical validation
  visualize.py       stage-by-stage pipeline visualisation

data/sample/
  create_sample.py   generates a small synthetic label map for testing
```


## Citation

If you use the original SynthSeg code or method, please cite:

```
@article{billot2023synthseg,
  title={SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
  author={Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel
          and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio},
  journal={Medical Image Analysis},
  volume={86},
  pages={102789},
  year={2023}
}
```

The original SynthSeg code is copyright 2020 Benjamin Billot, licensed under Apache 2.0.


## Licence

Apache 2.0. See the licence headers in individual files.
