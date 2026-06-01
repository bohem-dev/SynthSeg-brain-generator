"""
BrainAugmentor — faithful PyTorch/nibabel port of SynthSeg's BrainGenerator.

This is a 1-to-1 reimplementation of the SynthSeg generation pipeline found in
``histoseg/tools/SynthSeg-brain-augment``, with TensorFlow/Keras replaced by
torch + numpy + nibabel. It is a *literal transcription* of the reference layers
(``ext/lab2im/layers.py``), the tensor helpers (``ext/lab2im/edit_tensors.py``),
and the neuron warping primitives (``ext/neuron/utils.py`` /
``ext/neuron/layers.py``), following ``SynthSeg/labels_to_image_model.py`` step
for step. Where the reference relies on neuron's ``interpn`` / ``resize`` (which
use *clamped* — i.e. border-padded — interpolation and a ``coord = j * old/new``
resampling convention, NOT ``align_corners``), this port reproduces those exact
conventions rather than substituting torch's defaults.

Pipeline (labels_to_image_model order), all for batchsize=1, n_channels=1:

    1. RandomSpatialDeformation  — random affine (scaling @ shearing @ rotation)
       composed with a diffeomorphic SVF (sample small N(0, U(0,nonlin_std)),
       resize to half, scaling-and-squaring integrate, resize to full); labels
       warped with nearest, images with linear. §3.1.1
    2. RandomCrop                — random crop to output_shape (if smaller).
    3. RandomFlip                — flip the RAS left-right axis with p=0.5 (+ L/R
       label swap when label_list / n_neutral_labels describe sided structures).
    4. SampleConditionalGMM      — per-label N(mu_k, sigma_k) intensity synth. §3.1.2
    5. BiasFieldCorruption       — multiply by exp(resize(N(0, U(0,bias_std)))). §3.1.3
    6. IntensityAugmentation     — clip -> min-max normalise -> gamma. §3.1.4
    7. resolution simulation     — SampleResolution -> blurring_sigma -> dynamic
       Gaussian blur -> MimicAcquisition (downsample to LR, resample to output). §3.1.5
    8. ConvertLabels             — generation_labels -> output_labels. §3.1.6

The GMM means/stds and the SynthSeg background-reset logic are drawn per call,
matching ``model_inputs.build_model_inputs``.

Public API:
    aug = BrainAugmentor(...)
    image, label = aug(label_path, image_path=None)

Synthetic mode (image_path=None): intensities synthesised from the GMM.
Real mode (image_path given): the real MRI is carried through the SAME spatial
deformation / crop / flip / resolution path as the labels, but the GMM step is
skipped (real intensities are kept).
"""

import math
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

#  neuron.utils primitives (interpn / transform / resize / integrate_vec)
#
#  neuron.interpn clamps sampling locations to [0, N-1] (border padding) and does
#  clamped multilinear (or rounded nearest) interpolation. grid_sample with
#  align_corners=True + padding_mode='border' is mathematically identical: it maps
#  absolute index x -> normalised 2x/(N-1)-1, clamps out-of-range to the border,
#  and interpolates linearly between the clamped integer neighbours.


def _meshgrid(shape, device=None):
    """ij ndgrid of voxel indices -> (*shape, n_dims) float32 (neuron.volshape_to_ndgrid)."""
    ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    return torch.stack(grids, dim=-1)


def _interpn(vol, loc, interp_method="linear"):
    """Port of neuron.utils.interpn: sample ``vol`` at absolute voxel locations ``loc``.

    vol : (D0, D1, D2) or (D0, D1, D2, C) float tensor.
    loc : (*out_shape, n_dims) float tensor of absolute input-voxel coordinates.
    Returns: (*out_shape) if vol is single-channel, else (*out_shape, C).

    Uses grid_sample(align_corners=True, padding_mode='border') which reproduces
    neuron's coordinate clamping and clamped multilinear / nearest interpolation.
    """
    n_dims = loc.shape[-1]
    single = vol.ndim == n_dims
    sizes = vol.shape[:n_dims]
    if single:
        vol_c = vol[None, None].float()  # (1, 1, D0, D1, D2)
    else:
        vol_c = vol.permute(n_dims, *range(n_dims))[None].float()  # (1, C, D0, D1, D2)

    # normalise absolute coords to [-1, 1] per the align_corners=True convention
    norm_axes = []
    for d in range(n_dims):
        n = sizes[d]
        if n > 1:
            norm_axes.append(2.0 * loc[..., d] / (n - 1) - 1.0)
        else:
            norm_axes.append(torch.zeros_like(loc[..., d]))
    # grid_sample expects (x=last spatial axis, y=mid, z=first): reverse axis order
    grid = torch.stack(norm_axes[::-1], dim=-1)[None]  # (1, *out, n_dims)

    mode = "nearest" if interp_method == "nearest" else "bilinear"
    out = F.grid_sample(
        vol_c, grid, mode=mode, padding_mode="border", align_corners=True
    )
    out = out[0]  # (C, *out)
    if single:
        return out[0]
    return out.permute(*range(1, n_dims + 1), 0)  # (*out, C)


def _transform(vol, loc_shift, interp_method="linear"):
    """Port of neuron.utils.transform: interpolate ``vol`` at (ndgrid + loc_shift)."""
    volshape = loc_shift.shape[:-1]
    mesh = _meshgrid(volshape, device=loc_shift.device)
    loc = mesh + loc_shift
    return _interpn(vol, loc, interp_method=interp_method)


def _resize(vol, new_shape, interp_method="linear"):
    """Port of neuron.utils.resize / neuron.layers.Resize.

    For output index j along axis d, samples the input at ``j / zoom_d`` where
    ``zoom_d = new_shape_d / old_shape_d`` (i.e. ``j * old_d / new_d``), with
    border clamping. This is neuron's convention, NOT align_corners.
    """
    n_dims = len(new_shape)
    old_shape = vol.shape[:n_dims]
    zoom = [new_shape[i] / old_shape[i] for i in range(n_dims)]
    grid = _meshgrid(new_shape, device=vol.device)  # (*new_shape, n_dims)
    loc = torch.empty_like(grid)
    for d in range(n_dims):
        loc[..., d] = grid[..., d] / zoom[d]
    return _interpn(vol, loc, interp_method=interp_method)


def _integrate_vec(vec, nb_steps=7):
    """Scaling-and-squaring integration of a stationary velocity field.

    Port of neuron.utils.integrate_vec(method='ss'): vec /= 2**nb_steps; then
    repeat nb_steps times: vec += transform(vec, vec). vec is (*shape, n_dims),
    in voxel-displacement units.
    """
    vec = vec / (2**nb_steps)
    for _ in range(nb_steps):
        vec = vec + _transform(vec, vec, interp_method="linear")
    return vec


#  lab2im.utils.draw_value_from_distribution (numpy, uniform/normal)


def draw_value_from_distribution(
    hyperparameter,
    size=1,
    distribution="uniform",
    centre=0.0,
    default_range=10.0,
    positive_only=False,
):
    """Port of utils.draw_value_from_distribution (numpy path).

    hyperparameter:
      False  -> None
      None   -> [centre-default_range, centre+default_range]
      number -> [centre-h, centre+h]
      [a, b] -> [a, b] (tiled to ``size``)
      ndarray (2, m)   -> per-column bounds
      ndarray (2n, m)  -> randomly pick a 2-row block, then per-column
    Returns a float / numpy 1d array (or None).
    """
    if hyperparameter is False:
        return None

    if not isinstance(hyperparameter, np.ndarray):
        if hyperparameter is None:
            hyperparameter = np.array(
                [[centre - default_range] * size, [centre + default_range] * size]
            )
        elif isinstance(hyperparameter, (int, float)):
            hyperparameter = np.array(
                [[centre - hyperparameter] * size, [centre + hyperparameter] * size]
            )
        elif isinstance(hyperparameter, (list, tuple)):
            assert (
                len(hyperparameter) == 2
            ), "if list, hyperparameter should be length 2"
            hyperparameter = np.transpose(np.tile(np.array(hyperparameter), (size, 1)))
        else:
            raise ValueError(
                "hyperparameter should be None, number, sequence, or ndarray"
            )
    else:
        assert hyperparameter.shape[0] % 2 == 0, "ndarray rows should be divisible by 2"
        n_modalities = int(hyperparameter.shape[0] / 2)
        modality_idx = 2 * np.random.randint(n_modalities)
        hyperparameter = hyperparameter[modality_idx : modality_idx + 2, :]

    if distribution == "uniform":
        value = np.random.uniform(low=hyperparameter[0, :], high=hyperparameter[1, :])
    elif distribution == "normal":
        value = np.random.normal(loc=hyperparameter[0, :], scale=hyperparameter[1, :])
    else:
        raise ValueError("distribution should be 'uniform' or 'normal'")

    if positive_only:
        value = np.asarray(value)
        value[value < 0] = 0
    return value


#  Affine sampling (lab2im.utils.sample_affine_transform + create_* helpers)


def create_rotation_transform(rotation, n_dims=3):
    """Port of utils.create_rotation_transform (3D): T_rot = Rx @ Ry @ Rz, degrees."""
    r = np.asarray(rotation, dtype=np.float64) * np.pi / 180.0
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(r[0]), -np.sin(r[0])], [0, np.sin(r[0]), np.cos(r[0])]]
    )
    Ry = np.array(
        [[np.cos(r[1]), 0, np.sin(r[1])], [0, 1, 0], [-np.sin(r[1]), 0, np.cos(r[1])]]
    )
    Rz = np.array(
        [[np.cos(r[2]), -np.sin(r[2]), 0], [np.sin(r[2]), np.cos(r[2]), 0], [0, 0, 1]]
    )
    return Rx @ Ry @ Rz


def create_shearing_transform(shearing, n_dims=3):
    """Port of utils.create_shearing_transform (3D): 6 off-diagonal coefficients."""
    s = np.asarray(shearing, dtype=np.float64)
    return np.array([[1, s[0], s[1]], [s[2], 1, s[3]], [s[4], s[5], 1]])


def sample_affine_transform(
    n_dims=3,
    rotation_bounds=False,
    scaling_bounds=False,
    shearing_bounds=False,
    translation_bounds=False,
):
    """Port of utils.sample_affine_transform (batchsize=1). Returns (M 3x3, t 3,).

    T = T_scaling @ T_shearing @ T_rotation, translation appended as last column.
    Default ranges match the reference: rotation 15 deg, scaling 0.15, shearing
    0.01, translation 5 (all centred on 0 except scaling on 1).
    """
    if rotation_bounds is not False:
        rot = draw_value_from_distribution(
            rotation_bounds, size=n_dims, centre=0.0, default_range=15.0
        )
        T_rot = create_rotation_transform(rot, n_dims)
    else:
        T_rot = np.eye(n_dims)

    if shearing_bounds is not False:
        shear = draw_value_from_distribution(
            shearing_bounds, size=n_dims**2 - n_dims, centre=0.0, default_range=0.01
        )
        T_shear = create_shearing_transform(shear, n_dims)
    else:
        T_shear = np.eye(n_dims)

    if scaling_bounds is not False:
        scale = draw_value_from_distribution(
            scaling_bounds, size=n_dims, centre=1.0, default_range=0.15
        )
        T_scale = np.diag(np.asarray(scale, dtype=np.float64))
    else:
        T_scale = np.eye(n_dims)

    M = T_scale @ T_shear @ T_rot

    if translation_bounds is not False:
        t = draw_value_from_distribution(
            translation_bounds, size=n_dims, centre=0.0, default_range=5.0
        ).astype(np.float64)
    else:
        t = np.zeros(n_dims)

    return M, t


#  Step 1: RandomSpatialDeformation (affine + diffeomorphic elastic)


def random_spatial_deformation(
    label,
    image=None,
    scaling_bounds=0.2,
    rotation_bounds=15.0,
    shearing_bounds=0.012,
    translation_bounds=False,
    nonlin_std=4.0,
    nonlin_scale=0.0625,
    svf_int_steps=7,
):
    """Port of layers.RandomSpatialDeformation (§3.1.1).

    Builds the diffeomorphic elastic field exactly as the reference:
      small = ceil(shape * nonlin_scale); std ~ U(0, nonlin_std);
      field ~ N(0, std); resize to [max(shape//2, small)]; integrate (scaling &
      squaring); resize to full shape.
    Then composes with the random affine and warps (labels: nearest, image:
    linear), reproducing SpatialTransformer + combine_non_linear_and_aff_to_shift:
      loc = M @ ((mesh - centre) + nonlin) + t + centre.
    """
    shape = tuple(label.shape)
    n_dims = len(shape)
    device = label.device

    apply_affine = any(
        b is not False
        for b in (scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds)
    )
    apply_elastic = nonlin_std > 0

    nonlin = None
    if apply_elastic:
        small = [math.ceil(shape[i] * nonlin_scale) for i in range(n_dims)]
        trans_std = random.uniform(0.0, nonlin_std)  # U(0, nonlin_std)
        field = torch.randn(*small, n_dims, device=device) * trans_std
        resize_shape = [max(int(shape[i] / 2), small[i]) for i in range(n_dims)]
        field = _resize(field, resize_shape, interp_method="linear")
        field = _integrate_vec(field, nb_steps=svf_int_steps)
        nonlin = _resize(field, list(shape), interp_method="linear")

    mesh = _meshgrid(shape, device=device)
    centre = torch.tensor([(s - 1) / 2.0 for s in shape], device=device)
    centred = mesh - centre
    if apply_elastic:
        centred = centred + nonlin

    if apply_affine:
        M, t = sample_affine_transform(
            n_dims, rotation_bounds, scaling_bounds, shearing_bounds, translation_bounds
        )
        M_t = torch.tensor(M, dtype=torch.float32, device=device)
        t_t = torch.tensor(t, dtype=torch.float32, device=device)
        loc = torch.einsum("ij,...j->...i", M_t, centred) + t_t + centre
    else:
        loc = centred + centre

    warped_label = (
        _interpn(label.float(), loc, interp_method="nearest").round().to(torch.int32)
    )
    warped_image = None
    if image is not None:
        warped_image = _interpn(image.float(), loc, interp_method="linear")
    return warped_label, warped_image


#  Step 2: RandomCrop  /  Step 3: RandomFlip


def random_crop(label, image, crop_shape):
    """Port of layers.RandomCrop. Same random offset applied to label and image."""
    shape = tuple(label.shape)
    n_dims = len(shape)
    if crop_shape is None or tuple(crop_shape) == shape:
        return label, image
    crop_shape = [min(int(crop_shape[i]), shape[i]) for i in range(n_dims)]
    offs = [random.randint(0, shape[i] - crop_shape[i]) for i in range(n_dims)]
    sl = tuple(slice(offs[i], offs[i] + crop_shape[i]) for i in range(n_dims))
    label = label[sl]
    if image is not None:
        image = image[sl]
    return label, image


def get_ras_axes(aff, n_dims=3):
    """Port of edit_volumes.get_ras_axes — array axes corresponding to RAS axes."""
    aff_inverted = np.linalg.inv(np.asarray(aff, dtype=np.float64))
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i
    return img_ras_axes


def _build_swap_lut(label_list, n_neutral_labels):
    """Port of RandomFlip swap LUT: neutral, then left<->right swapped."""
    label_arr = np.asarray(label_list, dtype=np.int64)
    n_labels = len(label_arr)
    if n_neutral_labels == n_labels:
        return None
    n_side = (n_labels - n_neutral_labels) // 2
    rl = np.split(label_arr, [n_neutral_labels, n_neutral_labels + n_side])
    label_list_swap = np.concatenate((rl[0], rl[2], rl[1]))
    lut = np.arange(int(label_arr.max()) + 1, dtype=np.int64)
    lut[label_arr] = label_list_swap
    return lut


def random_flip(
    label,
    image,
    affine,
    flipping=True,
    prob=0.5,
    label_list=None,
    n_neutral_labels=None,
):
    """Port of layers.RandomFlip.

    SynthSeg flips along the RAS left-right array axis (get_ras_axes(aff)[0]) with
    probability ``prob``. If label_list/n_neutral_labels describe sided structures
    and an odd number of flips occurs, left/right labels are swapped first.
    """
    if not flipping:
        return label, image
    n_dims = label.ndim
    lr_axis = int(get_ras_axes(affine, n_dims)[0]) if affine is not None else 0

    do_flip = random.random() < prob
    if do_flip and label_list is not None and n_neutral_labels is not None:
        lut = _build_swap_lut(label_list, n_neutral_labels)
        if lut is not None:
            lut_t = torch.from_numpy(lut).to(label.device)
            label = lut_t[label.to(torch.int64)].to(torch.int32)

    if do_flip:
        label = torch.flip(label, dims=[lr_axis])
        if image is not None:
            image = torch.flip(image, dims=[lr_axis])
    return label, image


#  Step 4: GMM parameters (model_inputs) + SampleConditionalGMM


def build_gmm_params(
    n_classes,
    generation_classes,
    prior_means=None,
    prior_stds=None,
    prior_distribution="uniform",
):
    """Port of model_inputs.build_model_inputs (batchsize=1, n_channels=1).

    Per-class means ~ draw(prior_means, default centre/range 125,125, positive),
    stds ~ draw(prior_stds, 15,15, positive); then the SynthSeg background reset
    (class 0): 5% -> 0, next 25% -> low Gaussian; finally scattered to per-label
    vectors through generation_classes.
    """
    class_means = draw_value_from_distribution(
        prior_means, n_classes, prior_distribution, 125.0, 125.0, positive_only=True
    )
    class_stds = draw_value_from_distribution(
        prior_stds, n_classes, prior_distribution, 15.0, 15.0, positive_only=True
    )
    class_means = np.asarray(class_means, dtype=np.float64).copy()
    class_stds = np.asarray(class_stds, dtype=np.float64).copy()

    random_coef = np.random.uniform()
    if random_coef > 0.95:  # 5%: pure-black background
        class_means[0] = 0
        class_stds[0] = 0
    elif random_coef > 0.7:  # 25%: low-Gaussian background
        class_means[0] = np.random.uniform(0, 15)
        class_stds[0] = np.random.uniform(0, 5)

    means = class_means[generation_classes]
    stds = class_stds[generation_classes]
    return means.astype(np.float32), stds.astype(np.float32)


def sample_conditional_gmm(label, generation_labels, means, stds):
    """Port of layers.SampleConditionalGMM (n_channels=1): mu_L + sigma_L * N(0,1)."""
    gen = np.asarray(generation_labels, dtype=np.int64)
    max_label = int(gen.max()) + 1
    mean_lut = np.zeros(max_label, dtype=np.float32)
    std_lut = np.zeros(max_label, dtype=np.float32)
    mean_lut[gen] = means
    std_lut[gen] = stds

    mean_lut = torch.from_numpy(mean_lut).to(label.device)
    std_lut = torch.from_numpy(std_lut).to(label.device)
    idx = label.to(torch.int64).clamp(0, max_label - 1)
    mean_map = mean_lut[idx]
    std_map = std_lut[idx]
    return std_map * torch.randn_like(mean_map) + mean_map


#  Step 5: BiasFieldCorruption


def bias_field_corruption(image, bias_field_std=0.7, bias_scale=0.025, prob=0.95):
    """Port of layers.BiasFieldCorruption (§3.1.3, n_channels=1).

    small = ceil(shape * bias_scale); std ~ U(0, bias_field_std);
    bias ~ N(0, std); resize to full (neuron linear); image *= exp(bias).
    Applied with probability ``prob``.
    """
    if bias_field_std <= 0 or random.random() >= prob:
        return image
    shape = tuple(image.shape)
    n_dims = len(shape)
    small = [math.ceil(shape[i] * bias_scale) for i in range(n_dims)]
    std = random.uniform(0.0, bias_field_std)  # U(0, bias_field_std)
    small_field = torch.randn(*small, device=image.device) * std
    bias = _resize(small_field, list(shape), interp_method="linear")
    return image * torch.exp(bias)


#  Step 6: IntensityAugmentation


def intensity_augmentation(
    image, clip=300, normalise=True, gamma_std=0.5, prob_gamma=1.0
):
    """Port of layers.IntensityAugmentation (§3.1.4; no noise / inversion).

    Order: clip -> min-max normalise -> gamma. gamma ~ N(0, gamma_std) in log
    domain; image = image ** exp(gamma) (applied with probability prob_gamma).
    """
    if clip:
        if isinstance(clip, (int, float)):
            lo, hi = 0.0, float(clip)
        else:
            lo, hi = float(clip[0]), float(clip[1])
        image = image.clamp(lo, hi)
    if normalise:
        m = image.min()
        M = image.max()
        image = (image - m) / (M - m + 1e-7)
    if gamma_std > 0 and random.random() < prob_gamma:
        gamma = math.exp(random.gauss(0.0, gamma_std))
        image = image.pow(gamma)
    return image


#  Step 7: resolution simulation (SampleResolution + blur + MimicAcquisition)


def sample_resolution(
    atlas_res, max_res_iso, max_res_aniso, prob_iso=0.1, prob_min=0.05
):
    """Port of layers.SampleResolution (return_thickness=True).

    Both iso and aniso resolutions are sampled *per axis* (matching the reference,
    which uses tf.random.uniform over the n_dims shape). For the anisotropic
    branch only one randomly-chosen axis takes the sampled value (others = min).
    With prob_min the result is reset to min (= atlas). Thickness ~ U(min, res)
    per axis.
    """
    min_res = np.asarray(atlas_res, dtype=np.float32)
    n_dims = len(min_res)
    iso = np.asarray(max_res_iso, dtype=np.float32)
    aniso = np.asarray(max_res_aniso, dtype=np.float32)

    res_iso = np.random.uniform(min_res, iso).astype(np.float32)
    res_aniso_full = np.random.uniform(min_res, aniso).astype(np.float32)
    mask = np.zeros(n_dims, dtype=bool)
    mask[np.random.randint(n_dims)] = True
    res_aniso = np.where(mask, res_aniso_full, min_res).astype(np.float32)

    if np.random.uniform() < prob_iso:
        resolution = res_iso
    else:
        resolution = res_aniso
    if np.random.uniform() < prob_min:
        resolution = min_res.copy()

    thickness = np.random.uniform(min_res, resolution).astype(np.float32)
    return resolution, thickness


def blurring_sigma_for_downsampling(current_res, downsample_res, thickness=None):
    """Port of edit_tensors.blurring_sigma_for_downsampling (mult_coef=None)."""
    current = np.asarray(current_res, dtype=np.float32)
    down = np.asarray(downsample_res, dtype=np.float32)
    if thickness is not None:
        down = np.minimum(down, np.asarray(thickness, dtype=np.float32))
    sigma = 0.75 * down / current
    sigma[down == current] = 0.5
    sigma[down == 0] = 0.0
    return sigma


def _gaussian_blur(image, sigma, max_sigma, blur_range=1.03):
    """Port of layers.GaussianBlur / DynamicGaussianBlur + edit_tensors.gaussian_kernel.

    Gaussian kernels are separable, so per-axis 1-D convolutions reproduce the
    reference exactly. Kernel half-width per axis comes from ``max_sigma``
    (windowsize = int(ceil(2.5*max_sigma)/2)*2+1); the actual std is jittered by
    U(1/blur_range, blur_range).
    """
    sigma = np.asarray(sigma, dtype=np.float32).copy()
    max_sigma = np.asarray(max_sigma, dtype=np.float32)
    if blur_range is not None and blur_range != 1:
        sigma = sigma * np.random.uniform(
            1.0 / blur_range, blur_range, size=sigma.shape
        ).astype(np.float32)

    windowsize = (np.int32(np.ceil(2.5 * max_sigma) / 2) * 2 + 1).astype(int)

    x = image[None, None]  # (1, 1, D0, D1, D2)
    n_dims = image.ndim
    for axis in range(n_dims):
        sig = float(sigma[axis])
        wsize = int(windowsize[axis])
        if wsize <= 1 or sig <= 0:
            continue
        locations = (
            torch.arange(wsize, dtype=torch.float32, device=image.device)
            - (wsize - 1) / 2.0
        )
        g = torch.exp(
            -(locations**2) / (2 * sig**2) - math.log(math.sqrt(2 * math.pi) * sig)
        )
        g = g / g.sum()
        view = [1, 1, 1, 1, 1]
        view[2 + axis] = wsize
        kernel = g.view(view)
        radius = (wsize - 1) // 2
        pad = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # F.pad pads last dim first: axis d -> positions [(n_dims-1-d)*2 : +2]
        pad[(n_dims - 1 - axis) * 2] = radius
        pad[(n_dims - 1 - axis) * 2 + 1] = radius
        x = F.conv3d(F.pad(x, pad, mode="constant", value=0.0), kernel)
    return x[0, 0]


def mimic_acquisition(
    image, volume_res, min_subsample_res, resample_shape, subsample_res
):
    """Port of layers.MimicAcquisition (build_dist_map=False, n_channels=1).

    Literal transcription: build a down-grid sized by volume_res/min_subsample_res,
    sample (nearest) at down_grid / (down_shape/inshape) clamped to inshape, then
    sample (linear) at up_grid / (resample_shape/down_shape).
    """
    inshape = np.array(image.shape, dtype=np.float64)
    volume_res = np.asarray(volume_res, dtype=np.float64)
    min_subsample_res = np.asarray(min_subsample_res, dtype=np.float64)
    subsample_res = np.asarray(subsample_res, dtype=np.float64)
    resample_shape = np.asarray(resample_shape, dtype=np.float64)
    n_dims = len(inshape)
    device = image.device

    down_tensor_shape = (inshape * volume_res / min_subsample_res).astype(int)
    down_shape = (inshape * volume_res / subsample_res).astype(int)
    down_zoom_factor = down_shape / inshape
    up_zoom_factor = resample_shape / down_shape

    # downsample (nearest)
    down_grid = _meshgrid([int(s) for s in down_tensor_shape], device=device)
    down_loc = torch.empty_like(down_grid)
    for d in range(n_dims):
        down_loc[..., d] = (down_grid[..., d] / float(down_zoom_factor[d])).clamp(
            0.0, float(inshape[d])
        )
    vol = _interpn(image, down_loc, interp_method="nearest")

    # upsample (linear)
    up_grid = _meshgrid([int(s) for s in resample_shape.astype(int)], device=device)
    up_loc = torch.empty_like(up_grid)
    for d in range(n_dims):
        up_loc[..., d] = up_grid[..., d] / float(up_zoom_factor[d])
    vol = _interpn(vol, up_loc, interp_method="linear")
    return vol


def simulate_resolution(
    image,
    atlas_res,
    output_shape,
    randomise_res=True,
    data_res=None,
    thickness=None,
    max_res_iso=4.0,
    max_res_aniso=8.0,
    prob_iso=0.1,
    prob_min=0.05,
    blur_range=1.03,
):
    """Port of the per-channel resolution block in labels_to_image_model (§3.1.5)."""
    n_dims = image.ndim
    atlas = np.asarray(atlas_res, dtype=np.float32)
    if atlas.ndim == 0:
        atlas = np.full(n_dims, float(atlas), np.float32)

    if randomise_res:
        iso = (
            np.full(n_dims, max_res_iso, np.float32)
            if np.isscalar(max_res_iso)
            else np.asarray(max_res_iso, np.float32)
        )
        aniso = (
            np.full(n_dims, max_res_aniso, np.float32)
            if np.isscalar(max_res_aniso)
            else np.asarray(max_res_aniso, np.float32)
        )
        max_res = np.maximum(iso, aniso)
        resolution, blur_res = sample_resolution(atlas, iso, aniso, prob_iso, prob_min)
        sigma = blurring_sigma_for_downsampling(atlas, resolution, thickness=blur_res)
        max_sigma = 0.75 * max_res / atlas  # DynamicGaussianBlur kernel-size bound
        image = _gaussian_blur(image, sigma, max_sigma, blur_range=blur_range)
        # MimicAcquisition(atlas_res, atlas_res, output_shape)([channel, resolution])
        image = mimic_acquisition(image, atlas, atlas, output_shape, resolution)
    else:
        if data_res is None:
            data = atlas.copy()  # SynthSeg default: mild blur, no real downsample
        else:
            data = (
                np.full(n_dims, data_res, np.float32)
                if np.isscalar(data_res)
                else np.asarray(data_res, np.float32)
            )
        thick = (
            data
            if thickness is None
            else (
                np.full(n_dims, thickness, np.float32)
                if np.isscalar(thickness)
                else np.asarray(thickness, np.float32)
            )
        )
        sigma = blurring_sigma_for_downsampling(atlas, data, thickness=thick)
        image = _gaussian_blur(image, sigma, sigma, blur_range=blur_range)
        # MimicAcquisition(atlas_res, data_res, output_shape)([channel, data_res])
        image = mimic_acquisition(image, atlas, data, output_shape, data)
    return image


#  Step 8: ConvertLabels


def convert_labels(label, source_values, dest_values=None):
    """Port of layers.ConvertLabels. dest_values=None keeps source values."""
    if dest_values is None:
        return label
    src = np.asarray(source_values, dtype=np.int64)
    dst = np.asarray(dest_values, dtype=np.int64)
    lut = np.zeros(int(src.max()) + 1, dtype=np.int64)
    lut[src] = dst
    lut_t = torch.from_numpy(lut).to(label.device)
    return lut_t[label.to(torch.int64).clamp(0, len(lut) - 1)].to(torch.int32)


#  Loading


def load_sample(label_path, image_path=None):
    """Load label map (int32) and optional MRI (float32), plus affine + voxel size."""
    label_nib = nib.load(str(label_path))
    label = torch.from_numpy(np.asarray(label_nib.dataobj, dtype=np.int32))
    affine = label_nib.affine.astype(np.float64)
    vox_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)

    image = None
    if image_path is not None:
        image_nib = nib.load(str(image_path))
        image = torch.from_numpy(np.asarray(image_nib.dataobj, dtype=np.float32))
    return {"label": label, "image": image, "affine": affine, "vox_size": vox_size}


#  BrainAugmentor — full pipeline (BrainGenerator + labels_to_image_model)


class BrainAugmentor:
    """Faithful torch port of SynthSeg's BrainGenerator (batchsize=1, n_channels=1).

    Constructor names mirror the SynthSeg parameters and defaults
    (BrainGenerator.__init__). ``output_shape=None`` keeps the full deformed label
    shape with no cropping; a tuple triggers a random crop before GMM synthesis.
    """

    def __init__(
        self,
        # output
        output_shape=None,
        # GMM
        generation_labels=None,
        n_neutral_labels=None,
        output_labels=None,
        generation_classes=None,
        prior_distributions="uniform",
        prior_means=None,
        prior_stds=None,
        # spatial deformation
        flipping=True,
        scaling_bounds=0.2,
        rotation_bounds=15.0,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=4.0,
        nonlin_scale=0.0625,
        # resolution
        randomise_res=True,
        max_res_iso=4.0,
        max_res_aniso=8.0,
        data_res=None,
        thickness=None,
        # bias / intensity
        bias_field_std=0.7,
        bias_scale=0.025,
        clip=300,
        gamma_std=0.5,
        # misc
        atlas_res=None,
        svf_int_steps=7,
    ):
        self.output_shape = output_shape
        self.generation_labels = generation_labels
        self.n_neutral_labels = n_neutral_labels
        self.output_labels = output_labels
        self.generation_classes = generation_classes
        self.prior_distributions = prior_distributions
        self.prior_means = prior_means
        self.prior_stds = prior_stds
        self.flipping = flipping
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.data_res = data_res
        self.thickness = thickness
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.clip = clip
        self.gamma_std = gamma_std
        self.atlas_res = atlas_res
        self.svf_int_steps = svf_int_steps

    def __call__(self, label_path, image_path=None):
        sample = load_sample(label_path, image_path)
        label = sample["label"]
        image = sample["image"]
        affine = sample["affine"]
        vox_size = sample["vox_size"]

        atlas_res = self.atlas_res if self.atlas_res is not None else vox_size

        # generation labels / classes
        if self.generation_labels is None:
            gen_labels = label.unique().cpu().numpy().astype(np.int64)
        else:
            gen_labels = np.asarray(self.generation_labels, dtype=np.int64)
        n_neutral = (
            self.n_neutral_labels
            if self.n_neutral_labels is not None
            else len(gen_labels)
        )
        if self.generation_classes is None:
            gen_classes = np.arange(len(gen_labels), dtype=np.int64)
        else:
            gen_classes = np.asarray(self.generation_classes, dtype=np.int64)
        n_classes = int(np.unique(gen_classes).shape[0])

        # Step 1: spatial deformation
        label, image = random_spatial_deformation(
            label,
            image,
            scaling_bounds=self.scaling_bounds,
            rotation_bounds=self.rotation_bounds,
            shearing_bounds=self.shearing_bounds,
            translation_bounds=self.translation_bounds,
            nonlin_std=self.nonlin_std,
            nonlin_scale=self.nonlin_scale,
            svf_int_steps=self.svf_int_steps,
        )

        # Step 2: crop
        label, image = random_crop(label, image, self.output_shape)
        output_shape = tuple(label.shape)

        # Step 3: flip
        label, image = random_flip(
            label,
            image,
            affine,
            flipping=self.flipping,
            label_list=gen_labels if self.flipping else None,
            n_neutral_labels=n_neutral if self.flipping else None,
        )

        # Step 4: GMM synthesis (synthetic mode only)
        if image is None:
            means, stds = build_gmm_params(
                n_classes,
                gen_classes,
                self.prior_means,
                self.prior_stds,
                self.prior_distributions,
            )
            image = sample_conditional_gmm(label, gen_labels, means, stds)

        # Step 5: bias field
        image = bias_field_corruption(image, self.bias_field_std, self.bias_scale)

        # Step 6: intensity augmentation
        image = intensity_augmentation(image, clip=self.clip, gamma_std=self.gamma_std)

        # Step 7: resolution simulation
        image = simulate_resolution(
            image,
            atlas_res,
            output_shape,
            randomise_res=self.randomise_res,
            data_res=self.data_res,
            thickness=self.thickness,
            max_res_iso=self.max_res_iso,
            max_res_aniso=self.max_res_aniso,
        )

        # Step 8: convert labels
        label = convert_labels(label, gen_labels, self.output_labels)

        return image, label
