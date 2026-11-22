"""Augmentations sur lots : recadrages valides, symétrie, évaluation sans hasard."""

import numpy as np
import torch
import torch.nn.functional as F

from facegeom.data.augment import GpuAugment, crop_theta, sample_crop_boxes, to_float

SCALE = (0.16, 1.0)
RATIO = (3 / 4, 4 / 3)
MEAN = (0.0, 0.0, 0.0)
STD = (1.0, 1.0, 1.0)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _aug(out_size: int = 32) -> GpuAugment:
    return GpuAugment(out_size, MEAN, STD, SCALE, RATIO, 0.5, "cpu")


def test_boxes_inside_image_and_within_ranges():
    boxes = sample_crop_boxes(20_000, _gen(), SCALE, RATIO)
    x0, y0, w, h = boxes.unbind(1)
    assert torch.all(x0 >= 0) and torch.all(y0 >= 0)
    assert torch.all(x0 + w <= 1 + 1e-6) and torch.all(y0 + h <= 1 + 1e-6)
    area = w * h
    assert area.min() >= SCALE[0] - 1e-6 and area.max() <= SCALE[1] + 1e-6
    aspect = w / h
    assert aspect.min() >= RATIO[0] - 1e-5 and aspect.max() <= RATIO[1] + 1e-5
    # Les surfaces couvrent bien la plage (pas toutes à 1 par repli).
    assert area.mean() < 0.75


def test_boxes_depend_only_on_seed():
    a = sample_crop_boxes(64, _gen(3), SCALE, RATIO)
    b = sample_crop_boxes(64, _gen(3), SCALE, RATIO)
    c = sample_crop_boxes(64, _gen(4), SCALE, RATIO)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_fallback_is_whole_image_when_nothing_fits():
    # Surface imposée de 100 % et rapport très allongé : rien ne tient, image entière.
    boxes = sample_crop_boxes(10, _gen(), (1.0, 1.0), (3.0, 4.0))
    assert torch.allclose(boxes, torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 10))


def test_identity_crop_equals_plain_resize():
    """Recadrage = image entière, sans symétrie : même résultat que l'évaluation."""
    images = torch.randint(0, 256, (3, 16, 16, 3), dtype=torch.uint8)
    theta = crop_theta(torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3), torch.zeros(3, dtype=torch.bool))
    x = to_float(images)
    grid = F.affine_grid(theta, [3, 3, 32, 32], align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ref = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
    assert torch.allclose(out, ref, atol=1e-5)


def test_flip_mirrors_the_crop():
    images = torch.randint(0, 256, (2, 16, 16, 3), dtype=torch.uint8)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 2)
    x = to_float(images)
    outs = []
    for flip in (False, True):
        theta = crop_theta(boxes, torch.full((2,), flip))
        grid = F.affine_grid(theta, [2, 3, 16, 16], align_corners=False)
        outs.append(F.grid_sample(x, grid, mode="bilinear", align_corners=False))
    assert torch.allclose(outs[1], outs[0].flip(-1), atol=1e-5)


def test_corner_crop_picks_the_right_pixels():
    """Le quart haut gauche agrandi ne contient que les pixels du quart haut gauche."""
    img = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    img[0, :4, :4] = 200  # quart haut gauche clair, le reste noir
    theta = crop_theta(torch.tensor([[0.0, 0.0, 0.5, 0.5]]), torch.zeros(1, dtype=torch.bool))
    grid = F.affine_grid(theta, [1, 3, 16, 16], align_corners=False)
    out = F.grid_sample(to_float(torch.from_numpy(img)), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)
    # À l'intérieur du recadrage (loin du bord droit et du bas), tout est clair.
    assert torch.allclose(out[0, :, :12, :12], torch.full((3, 12, 12), 200 / 255), atol=1e-5)


def test_train_and_eval_shapes_and_normalization():
    aug = GpuAugment(24, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), SCALE, RATIO, 0.5, "cpu")
    images = torch.randint(0, 256, (5, 16, 16, 3), dtype=torch.uint8)
    tr = aug.train(images, _gen())
    ev = aug.eval(images)
    assert tr.shape == ev.shape == (5, 3, 24, 24)
    assert tr.dtype == ev.dtype == torch.float32
    assert ev.min() >= -1 - 1e-6 and ev.max() <= 1 + 1e-6


def test_eval_is_deterministic_and_train_is_seeded():
    aug = _aug()
    images = torch.randint(0, 256, (4, 16, 16, 3), dtype=torch.uint8)
    assert torch.equal(aug.eval(images), aug.eval(images))
    assert torch.equal(aug.train(images, _gen(1)), aug.train(images, _gen(1)))
    assert not torch.equal(aug.train(images, _gen(1)), aug.train(images, _gen(2)))
