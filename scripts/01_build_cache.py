"""Étape 1 : cache des carrés de 128 px, une fois par session Colab.

Chaque image est lue (dossier décompressé ou zip), recadrée et stockée en
octets dans ``<cache-dir>/<partie>_128.npy``, dans l'ordre de ``splits/``.
Relancer la commande reprend un cache coupé et saute un cache complet.

    python scripts/01_build_cache.py --source /content/data/img_align_celeba --cache-dir /content/cache
    python scripts/01_build_cache.py --source data/celeba/img_align_celeba.zip --splits train val
"""

import time

from facegeom.cli import parser
from facegeom.data.cache import build_cache, cache_file
from facegeom.data.dataset import SPLITS, cache_box, read_split
from facegeom.utils import load_config, resolve_workers


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--source", required=True, help="dossier img_align_celeba ou zip de CelebA")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--workers", default=None, help="processus de décodage (défaut : configuration)")
    args = p.parse_args()

    dcfg = load_config(args.data_config)
    box = cache_box(dcfg)
    workers = resolve_workers(args.workers or dcfg["cache"]["workers"], cap=64, reserve=0)
    print(f"Recadrage {box}, {workers} processus, source {args.source}")
    for split in args.splits:
        t = time.perf_counter()
        names, _ = read_split(dcfg["splits"]["output_dir"], split)
        build_cache(names, args.source, box, cache_file(args.cache_dir, split, dcfg["crop"]["size"]),
                    chunk_size=int(dcfg["cache"]["chunk_size"]), workers=max(1, workers))
        print(f"{split} : {len(names)} images en {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()
