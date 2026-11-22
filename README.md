# srd-degel

Projet SRD sujet 09 (M1 informatique, Aix-Marseille Université) : qu'est-ce que la
fonction de coût change à la façon dont les visages sont rangés dans l'espace des
vecteurs, et est-ce que ça aide à reconnaître des personnes jamais vues ?

Dispositif : ResNet-18 supervisé ImageNet entraîné en entier (BatchNorm gelés),
tête linéaire 512 → 128, CelebA en 224 px, 4 pertes (`nsoftmax`, `arcface`,
`contrastive`, `triplet`). Seule la perte change.

- Protocole : [`docs/protocole.md`](docs/protocole.md) (source de vérité)
- Lancer les calculs : [`docs/guide_colab.md`](docs/guide_colab.md) et
  [`notebooks/colab.ipynb`](notebooks/colab.ipynb)

## Étapes

| Script | Étape |
|---|---|
| `scripts/01_build_cache.py` | cache des carrés de 128 px |
| `scripts/02_baselines.py` | mesures de base : brut, ACP 128, LDA 128 |
| `scripts/03_pilot.py` | pilote : une époque par perte, durée, budget |
| `scripts/04_tune.py` | réglage : 10 tirages par perte |
| `scripts/05_final.py` | entraînements finaux : graines 0, 1, 2 |
| `scripts/06_evaluate.py` | évaluation (validation, puis test une seule fois) |
| `scripts/07_figures.py` | figures d'une évaluation |
| `scripts/08_classical.py` | Eigenfaces, Fisherfaces, LBP |

## Sur le Mac

```bash
cd ~/Projets/SRD-degel
PYTHONPATH=src python -m pytest -q
ruff check .
```

Ce dépôt remplace `srd-facegeom` (ResNet-50 gelé), dont il reprend la partition,
l'évaluation et les bonnes pratiques.
