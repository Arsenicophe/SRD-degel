# Guide : lancer le protocole sur Colab

Pas à pas, dans l'ordre. Le détail des choix est dans `docs/protocole.md`.

## 0. Une seule fois : mettre le code sur GitHub

Sur le Mac, dans un terminal :

1. Créer sur GitHub un dépôt **public** nommé `srd-degel` (vide, sans README).
2. Pousser :

```bash
cd ~/Projets/SRD-degel
git remote add origin https://github.com/Arsenicophe/srd-degel.git
git push -u origin main
```

3. Ouvrir le notebook dans Colab :
   <https://colab.research.google.com/github/Arsenicophe/srd-degel/blob/main/notebooks/colab.ipynb>

## 1. À chaque session Colab

1. **Exécution → Modifier le type d'exécution → GPU** (A100 ou L4 si possible).
2. Cellule **1** : monte Drive, clone ou met à jour le code, installe
   pytorch-metric-learning, affiche la carte graphique.
3. Cellule **2** : copie le zip de CelebA sur le disque local et construit le cache
   des carrés de 128 px (environ 3 minutes). Le cache disparaît à la fin de la
   session : il faut refaire cette cellule à chaque fois.
4. Facultatif : les tests (`pytest`, environ 1 minute).

## 2. Ce qui est déjà fait

- **Étape 2, mesures de base** : faites sur le Mac le 22 novembre. Le résumé est
  dans le dépôt (`results/2_baselines_val.json`) ; les scripts suivants s'en
  contentent. Résultats sur la validation (ResNet-18 ImageNet, 224 px) :

  | Méthode | MAP@R | P@1 | AUC |
  |---|---|---|---|
  | brut (512) | 3,02 | 23,3 | 74,8 |
  | brut (ACP 128) | 3,01 | 22,0 | 81,1 |
  | LDA 128 | 5,13 | 30,3 | 82,9 |

## 3. Le pilote, en premier

Cellule **Étape 3**. Une époque par perte, réglages au centre des plages. Le
tableau final donne, pour chaque perte :

- `map_at_r` après une époque, et `apprend` (vrai si plus haut que le brut) ;
- `seconds_epoch_full` : durée d'une époque complète ;
- en bas, le **temps prévu** du réglage et des graines finales, au plus.

C'est ce tableau qui permet de décider du nombre d'époques et de la règle de
repli (protocole, section 1). **À regarder en particulier** : si `nsoftmax` ou
`arcface` ont un `map_at_r` proche de 0 (au niveau du hasard), le réseau s'est
effondré ; c'est arrivé sur le Mac avec une montée du taux comprimée. Le groupe
peut alors décider d'activer l'écrêtage des gradients (`grad_clip_norm: 5` dans
`configs/train.yaml`, comme l'essai du 17 novembre) et relancer le pilote avec
`--new`. Une décision prise s'écrit dans
`docs/protocole.md`, avec une ligne au journal.

## 4. Réglage et graines finales

- Cellule **Étape 4** avec `LOSS = "all"` : les 4 pertes à la suite ; ou une perte
  par session (`"arcface"`, etc.) pour aller plus vite avec plusieurs sessions.
- Cellule **Étape 5** : les graines 0, 1, 2 avec le meilleur tirage.
- Ou la cellule **Tout enchaîner**.
- **Après une coupure** : refaire les cellules 1 et 2, puis relancer la même
  cellule. Les runs finis sont sautés, le run coupé reprend à sa dernière époque.
- Si une cellule refuse de démarrer à cause d'un **verrou**, c'est qu'une autre
  session travaille sur la même perte. Si cette session est bien arrêtée,
  attendre 15 minutes ou ajouter `--force` à la commande. Avec `--loss all`, une
  perte prise par une autre session est sautée et les autres continuent.
- Un tirage qui **diverge** est arrêté et noté « diverged » dans son
  `result.json` ; le réglage passe au tirage suivant.
- **Corbeille de Drive** : chaque point de reprise remplacé part dans la
  corbeille, qui compte dans le quota. Après le pilote, regarder la taille de la
  corbeille de Drive ; la vider de temps en temps pendant le réglage.

## 5. Évaluation, figures, test

- Cellule **Méthodes classiques** : sur processeur, à n'importe quel moment.
- Cellule **Étape 6** : évaluation sur la validation et figures, dans
  `MyDrive/srd-degel/runs/<date>_6_eval_val/`.
- Cellule **Étape 7** : le **test, une seule fois**, quand tout est décidé. Il
  faut les 4 pertes avec leurs 3 graines et les méthodes classiques. Si la
  session est coupée pendant la lecture du test, la relancer avec
  `--retry-interrupted-test` (la tentative est notée).

## Où sont les résultats

Tout est dans `MyDrive/srd-degel/runs/`, un dossier par étape, jamais écrasé :

| Dossier | Contenu |
|---|---|
| `*_3_pilot/` | `pilot.json` : durées, apprentissage, budget |
| `*_4_tune_<perte>/` | `trials.json`, un dossier par tirage, `summary.csv`, `best.json` |
| `*_5_final_<perte>/` | un dossier par graine (`best.pt`, `history.csv`, `result.json`) |
| `*_6_eval_val/` | `per_seed.csv`, `summary.csv`, `paired.csv`, `bootstrap.csv`, `figures/` |
| `*_8_classical/` | vecteurs Eigenfaces, Fisherfaces, LBP |
