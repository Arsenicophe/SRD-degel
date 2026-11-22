# CLAUDE.md — Projet SRD sujet 09, réseau entraîné en entier

Contexte court, chargé à chaque session. Le détail est ailleurs :

- **`docs/protocole.md`** — le protocole expérimental. **Source de vérité** :
  en cas de doute ou de contradiction, c'est lui qui fait foi.
- `docs/guide_colab.md` — comment lancer les calculs sur Colab, pas à pas.
- `docs/notes/` — les notes de travail d'Arsène qui ont fixé ce protocole.
- `references/musgrave2020_fiche.md` — fiche de lecture de l'article de référence.

## Le projet

Module SRD, M1 informatique, Aix-Marseille Université, groupe de 4. Sujet 09
« Algorithmique de la reconnaissance faciale, avant et après IA ».
Livrables en anglais : synthèse d'avancement (2 pages), poster A0 soutenu à l'oral.
Rendu le **vendredi 27 novembre 2026**.

**Question** : qu'est-ce que la fonction de coût change à la façon dont les
visages sont rangés dans l'espace des vecteurs, et est-ce que ça aide à
reconnaître des personnes jamais vues ?

**Dispositif** : ResNet-18 supervisé ImageNet **entraîné en entier** (BatchNorm
gelés), tête linéaire 512 → 128, CelebA en 224 px, 4 pertes (`nsoftmax`,
`arcface`, `contrastive`, `triplet`). Seule la perte change. Méthode inspirée de
Musgrave et al. (2020), sans la reprendre en entier.

Ce dépôt remplace `srd-facegeom` (ResNet-50 gelé). Le dégel a été décidé le
22 novembre 2026 avec l'accord de l'encadrant.

## Où tourne quoi

Les calculs tournent sur **Google Colab** : la machine est effacée à chaque
déconnexion, les sessions sont limitées en durée.

| Quoi | Où |
|---|---|
| Code | GitHub `Arsenicophe/srd-degel` ; Colab le clone à chaque session |
| CelebA | Google Drive (compte d'Arsène), un seul zip, copié et décompressé à chaque session |
| Cache des images (carrés de 128 px) | disque local de Colab, reconstruit à chaque session |
| Résultats des runs | Google Drive, `MyDrive/srd-degel/runs/` |
| Copie de travail du code | `~/Projets/SRD-degel` sur le Mac d'Arsène, hors d'iCloud |

Conséquences pour le code : le notebook ne fait que monter Drive, cloner,
installer et lancer des scripts ; chaque entraînement écrit un point de reprise
à chaque époque et sait reprendre ; chaque run enregistre les versions des
bibliothèques ; un verrou empêche deux sessions d'écrire dans le même dossier.

## Façon de travailler

- Découper chaque demande en **petites étapes**. Faire la première, s'arrêter,
  attendre la validation d'Arsène, puis passer à la suivante.
- Proposer des options avec une recommandation, ne pas décider à la place du
  groupe. Les choix non discutés sont marqués **[à valider]**.
- Ce qui n'est pas clair se tranche par une **mesure** (pilote, essai court),
  pas par une prédiction.
- Une décision prise est écrite dans `docs/protocole.md`, avec une ligne dans son
  journal.
- Expliquer simplement, en français.

## Règles qu'on ne viole jamais

1. Aucune personne du test n'apparaît à l'entraînement ni en validation. Un test
   automatique le vérifie et n'est jamais désactivé.
2. Le test est consulté **une seule fois**, à la fin. Tous les choix (réglages,
   seuils, nombre d'époques) se font sur la validation.
3. Mêmes conditions pour les 4 pertes, même nombre d'essais de réglage.
4. Mêmes graines pour toutes les pertes, intervalle de confiance sur chaque chiffre.
5. Tout est consigné, runs ratés compris. Un dossier par run, jamais écrasé.
6. Aucun entraînement de perte avant les mesures de base (brut + LDA).

## Ce qu'on ne fait pas

- Entraîner un réseau depuis zéro (on part toujours des poids ImageNet).
- Utiliser des modèles publics entraînés sur visages comme terme de comparaison.
- Implémenter la détection ou l'alignement.
- Ajouter une cinquième perte.
- Comparer nos chiffres à des chiffres publiés.
- Utiliser des photos produites par nous ou nos proches.

## Conventions de code

- Python ≥ 3.10, layout `src/`, paquet `facegeom`.
- Tous les réglages dans des fichiers de configuration (`configs/`), aucune
  valeur en dur.
- Scripts numérotés dans l'ordre d'exécution, orchestration seulement ; la logique
  vit dans le paquet.
- Graines fixées pour `random`, `numpy`, `torch`.
- Pertes via `pytorch-metric-learning`, pas d'implémentation maison.
- Docstrings en français, noms de symboles en anglais.
- Linter (`ruff check .`) et tests (`pytest -q`) avant chaque commit.
- Messages de commit sans ligne « Co-Authored-By » ni mention de Claude.

**Tests prioritaires** : non-recouvrement des identités ; `arcface` avec m = 0
égale `nsoftmax` ; métriques à 1,0 sur des vecteurs synthétiques séparables ;
même graine, même résultat (sur processeur) ; reprise après coupure identique à
un run sans coupure.

## Répartition

| Qui | Travail |
|---|---|
| Arsène | partie mathématique, conception du dispositif, garantie d'équité, relecture |
| Membre 2 | entraînements des pertes |
| Membre 3 | mesures de géométrie, courbes, figures |
| Membre 4 | Eigenfaces, Fisherfaces, LBP |

## État du projet

- [x] Protocole « réseau entraîné en entier » rédigé (`docs/protocole.md`)
- [x] Code écrit et testé sur le Mac (petit sous-ensemble)
- [ ] Étape 1 : cache des images sur Colab
- [ ] Étape 2 : mesures de base (brut, LDA) sur la validation
- [ ] Étape 3 : pilote, mesure du budget
- [ ] Étapes 4 et 5 : réglage, entraînements finaux
- [ ] Étapes 6 et 7 : évaluation sur la validation, puis test une seule fois

Mettre cette liste à jour quand une étape est terminée.
