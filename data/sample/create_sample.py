"""Create a small synthetic label map for testing.

Generates a 96x96x96 label map with anatomically-inspired regions (spheres and
ellipsoids), saved as label.nii.gz in the same directory. No external data
needed; no internet connection required.

This is intended for quick functional tests. For realistic experiments use real
label maps such as those from the IXI dataset (see README).

Usage:
    python data/sample/create_sample.py
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def ellipsoid_mask(shape, centre, radii):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        ((z - centre[0]) / radii[0]) ** 2
        + ((y - centre[1]) / radii[1]) ** 2
        + ((x - centre[2]) / radii[2]) ** 2
    ) <= 1.0


def main():
    out_dir = Path(__file__).resolve().parent
    shape = (96, 96, 96)
    c = [s // 2 for s in shape]
    label = np.zeros(shape, dtype=np.int32)

    # background (0) fills everything; assign regions from outside in

    # skull-like outer shell: label 1
    label[ellipsoid_mask(shape, c, [44, 44, 44])] = 1

    # white matter: label 2
    label[ellipsoid_mask(shape, c, [36, 36, 36])] = 2

    # grey matter cortex: label 3
    label[ellipsoid_mask(shape, c, [40, 40, 40])] = 3
    label[ellipsoid_mask(shape, c, [36, 36, 36])] = 2  # white matter inside cortex

    # ventricles: label 4 (bilateral)
    label[ellipsoid_mask(shape, [c[0], c[1] - 8, c[2]], [6, 4, 10])] = 4
    label[ellipsoid_mask(shape, [c[0], c[1] + 8, c[2]], [6, 4, 10])] = 4

    # brainstem: label 5
    label[ellipsoid_mask(shape, [c[0] + 20, c[1], c[2]], [10, 7, 7])] = 5

    # cerebellum left/right: labels 6 and 7
    label[ellipsoid_mask(shape, [c[0] + 15, c[1] - 18, c[2]], [8, 8, 8])] = 6
    label[ellipsoid_mask(shape, [c[0] + 15, c[1] + 18, c[2]], [8, 8, 8])] = 7

    # hippocampus left/right: labels 8 and 9
    label[ellipsoid_mask(shape, [c[0] + 5, c[1] - 14, c[2] - 5], [4, 3, 7])] = 8
    label[ellipsoid_mask(shape, [c[0] + 5, c[1] + 14, c[2] - 5], [4, 3, 7])] = 9

    # thalamus left/right: labels 10 and 11
    label[ellipsoid_mask(shape, [c[0], c[1] - 7, c[2]], [5, 5, 5])] = 10
    label[ellipsoid_mask(shape, [c[0], c[1] + 7, c[2]], [5, 5, 5])] = 11

    # putamen left/right: labels 12 and 13
    label[ellipsoid_mask(shape, [c[0] - 3, c[1] - 16, c[2] + 5], [4, 4, 6])] = 12
    label[ellipsoid_mask(shape, [c[0] - 3, c[1] + 16, c[2] + 5], [4, 4, 6])] = 13

    vox_mm = 1.0
    affine = np.diag([vox_mm, vox_mm, vox_mm, 1.0])
    out_path = out_dir / "label.nii.gz"
    nib.save(nib.Nifti1Image(label, affine), str(out_path))

    labels_present = np.unique(label)
    print(f"Saved: {out_path}")
    print(f"  shape : {shape}")
    print(f"  voxel : {vox_mm} mm isotropic")
    print(f"  labels: {labels_present.tolist()}")


if __name__ == "__main__":
    main()
