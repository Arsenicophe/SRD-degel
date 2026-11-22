"""Tests du recadrage (boîte fixe, choisie le 9 novembre et reprise telle quelle)."""

from pathlib import Path

import pytest
from PIL import Image

from facegeom.data.crop import crop_box, crop_face
from facegeom.data.dataset import cache_box
from facegeom.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CROP = load_config(ROOT / "configs" / "data.yaml")["crop"]
DATA = load_config(ROOT / "configs" / "data.yaml")


@pytest.mark.parametrize("size", [100, 128, 150, 178])
def test_crop_box_is_square_and_inside_image(size):
    l, t, r, b = crop_box(CROP, size)
    assert r - l == size and b - t == size
    assert 0 <= l and 0 <= t and r <= 178 and b <= 218


def test_chosen_crop_box():
    box = crop_box(CROP)
    assert box == (24, 52, 152, 180)
    # Les yeux tombent à ~46 % de la hauteur du carré
    assert abs((CROP["eye_y"] - box[1]) / CROP["size"] - CROP["eye_rel"]) < 0.01


def test_crop_rejects_wrong_image_size():
    with pytest.raises(ValueError):
        crop_face(Image.new("RGB", (200, 200)), (0, 0, 128, 128))


def test_cache_box_matches_config():
    assert cache_box(DATA) == tuple(DATA["crop"]["box"]) == (24, 52, 152, 180)


def test_cache_box_refuses_inconsistent_config():
    with pytest.raises(ValueError):
        cache_box({**DATA, "crop": {**DATA["crop"], "box": [0, 0, 128, 128]}})
