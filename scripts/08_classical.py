"""Méthodes classiques : Eigenfaces, Fisherfaces, LBP (processeur seulement, à tout moment).

Mêmes carrés de 128 px que le réseau (cache), en niveaux de gris. ACP et LDA
ajustées sur les pixels d'entraînement ; vecteurs écrits pour la validation et
le test. Aucune métrique ici : l'évaluation (scripts/06_evaluate.py) les mesure
avec les autres méthodes, le test une seule fois.

    python scripts/08_classical.py --cache-dir /content/cache --runs-dir $RUNS
"""

import time
from pathlib import Path

import numpy as np

from facegeom.classical import PixelProjections, extract_pixels
from facegeom.cli import parser
from facegeom.data.cache import cache_file, open_cache
from facegeom.data.dataset import read_split
from facegeom.stages import DONE, select_stage, stage_lock, write_stage_meta
from facegeom.utils import load_config, resolve_workers, write_json


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--classical-config", default="configs/classical.yaml")
    p.add_argument("--new", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    dcfg, ccfg = load_config(args.data_config), load_config(args.classical_config)
    runs = Path(args.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    workers = max(1, resolve_workers(ccfg.get("workers", "auto"), cap=64, reserve=0))
    size = dcfg["crop"]["size"]

    with stage_lock(runs, "classical", None, args.force) as lock:
        stage = select_stage(runs, "classical", None, args.new)
        if stage is None:
            return
        write_stage_meta(stage, {"data": dcfg, "classical": ccfg}, vars(args), "cpu")

        pixels, labels = {}, {}
        for split in ["train", "val", "test"]:
            names, labels[split] = read_split(dcfg["splits"]["output_dir"], split)
            path = cache_file(args.cache_dir, split, size)
            open_cache(path, names)  # vérifie le cache
            t = time.perf_counter()
            lbp = ccfg["lbp"] if split != "train" else None
            pixels[split], hist = extract_pixels(str(path), len(names), ccfg["gray_size"], lbp,
                                                 workers)
            if hist is not None:
                np.save(stage / f"lbp_{split}.npy", hist)
            (stage / f"{split}_images.txt").write_text("\n".join(names) + "\n")
            print(f"{split} : {len(names)} images en {time.perf_counter() - t:.0f} s")
            lock.heartbeat()

        t = time.perf_counter()
        proj = PixelProjections(ccfg["dim"], ccfg["fisher_pca_dim"]).fit(pixels["train"],
                                                                         labels["train"])
        print(f"ACP et LDA ajustées en {time.perf_counter() - t:.0f} s ; variance gardée "
              f"{proj.pca_.explained_variance_ratio_[:ccfg['dim']].sum():.3f} (Eigenfaces)")
        for split in ["val", "test"]:
            np.save(stage / f"eigenfaces_{split}.npy", proj.eigenfaces(pixels[split]))
            np.save(stage / f"fisherfaces_{split}.npy", proj.fisherfaces(pixels[split]))
        np.save(stage / "eigenfaces_components.npy",
                proj.pca_.components_[: ccfg["dim"]].astype(np.float32))
        write_json(stage / DONE, {"methods": ["eigenfaces", "fisherfaces", "lbp"]})
        print("Résultats dans", stage)


if __name__ == "__main__":
    main()
