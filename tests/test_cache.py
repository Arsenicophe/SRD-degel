"""Cache des carrés : contenu, ordre, reprise après coupure, refus d'un cache incomplet."""

import json

import numpy as np
import pytest
from PIL import Image

from facegeom.data.cache import build_cache, is_complete, open_cache

BOX = (24, 52, 152, 180)


@pytest.fixture()
def images(tmp_path):
    """Dix fausses images CelebA (178 × 218), chacune d'une couleur unie différente."""
    folder = tmp_path / "img_align_celeba"
    folder.mkdir()
    names = []
    for i in range(10):
        name = f"{i + 1:06d}.png"
        Image.new("RGB", (178, 218), color=(20 * i, 255 - 20 * i, 7 * i)).save(folder / name)
        names.append(name)
    return folder, names


def test_build_reads_crops_in_split_order(tmp_path, images):
    folder, names = images
    path = build_cache(names, folder, BOX, tmp_path / "cache" / "val_128.npy", chunk_size=3,
                       workers=1, log=lambda s: None)
    arr = open_cache(path, names)
    assert arr.shape == (10, 128, 128, 3) and arr.dtype == np.uint8
    for i in range(10):
        assert tuple(arr[i, 64, 64]) == (20 * i, 255 - 20 * i, 7 * i)


def test_resume_rebuilds_only_missing_chunks(tmp_path, images):
    folder, names = images
    path = tmp_path / "val_128.npy"
    build_cache(names, folder, BOX, path, chunk_size=3, workers=1, log=lambda s: None)
    # Simule une coupure : le morceau 2 n'avait pas été écrit, la fin n'était pas marquée
    done = path.with_name("val_128.done.json")
    digest = json.loads(done.read_text())["names_sha1"]
    done.unlink()
    path.with_name("val_128.progress").write_text(f"{digest}\n0\n1\n3\n")
    arr = np.lib.format.open_memmap(path, mode="r+")
    arr[6:9] = 0
    del arr
    assert not is_complete(path, names)
    messages = []
    build_cache(names, folder, BOX, path, chunk_size=3, workers=1, log=messages.append)
    assert any("1 à faire" in m for m in messages)
    arr = open_cache(path, names)
    assert tuple(arr[7, 0, 0]) == (140, 115, 49)


def test_cache_for_another_name_list_is_refused(tmp_path, images):
    folder, names = images
    path = build_cache(names, folder, BOX, tmp_path / "val_128.npy", chunk_size=5, workers=1,
                       log=lambda s: None)
    with pytest.raises(FileNotFoundError):
        open_cache(path, names[::-1])
