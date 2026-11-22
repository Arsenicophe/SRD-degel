# Protocole expérimental, SRD sujet 09 : réseau entraîné en entier

Document de travail du groupe. Il fixe **comment** l'expérience est menée.
Toute modification se discute en groupe et se date ici, en bas, dans le journal.

Les points marqués **[à valider]** sont des précisions techniques ajoutées à la
rédaction, pas encore discutées en groupe.

Ce protocole remplace celui du dépôt `srd-facegeom` (ResNet-50 Barlow Twins
gelé, tête linéaire). Le passage au réseau entraîné en entier a été décidé le
22 novembre 2026, avec l'accord de l'encadrant. Les décisions viennent des notes
de travail d'Arsène (versions 1 à 7, `docs/notes/`). Les bonnes pratiques de
l'ancien protocole sont reprises telles quelles (section 12).

**Principe** : ce qui n'est pas clair se tranche par une mesure, pas par une
prédiction.

Référence principale : Musgrave, Belongie & Lim (2020), *A Metric Learning
Reality Check*, fiche de lecture dans `references/musgrave2020_fiche.md`.
Valeurs optimales de ses réglages : matériel supplémentaire en ligne,
<https://kevinmusgrave.github.io/powerful-benchmarker/papers/mlrc>.

---

## 1. Objectif et question de recherche

**Question.** Qu'est-ce que la fonction de coût change à la façon dont les
visages sont rangés dans l'espace des vecteurs, et est-ce que ça aide à
reconnaître des personnes **jamais vues** à l'entraînement ?

**Principe.** Une comparaison contrôlée : même réseau de départ, même tête,
mêmes données, même recette d'entraînement, même évaluation. **Seule la
fonction de coût change**, avec ses propres réglages.

**Ce qui change par rapport à l'ancien protocole.** Le réseau entier est
entraîné, comme chez Musgrave. La perte façonne donc le réseau lui-même, et pas
seulement une réorganisation linéaire de vecteurs figés.

**Périmètre.** La chaîne complète d'un système est détection, alignement,
description, comparaison. On ne traite que la **description** (image vers
vecteur), sur les recadrages alignés fournis par CelebA.

**Calendrier.** Rendu le **vendredi 27 novembre 2026**. Le budget de calcul est
mesuré pendant le pilote (section 8) et comparé à cette date. Règle de repli si
le budget ne tient pas : **à décider après le pilote**.

### Ordre d'exécution

| Étape | Contenu | Script | Section |
|---|---|---|---|
| 0 | Dépôt, environnement, partition reprise, test de non-recouvrement | `tests/` | 3 |
| 1 | Cache des images recadrées (128 px) | `scripts/01_build_cache.py` | 3 |
| 2 | Mesures de base sur le ResNet-18 de départ : brut (ACP 128) et LDA | `scripts/02_baselines.py` | 5 |
| 3 | Pilote : une époque par perte, durée, budget | `scripts/03_pilot.py` | 8 |
| 4 | Réglage : 10 tirages par perte | `scripts/04_tune.py` | 8 |
| 5 | Entraînements finaux : meilleur tirage × graines 0, 1, 2 | `scripts/05_final.py` | 9 |
| 6 | Évaluation sur la validation, figures | `scripts/06_evaluate.py`, `scripts/07_figures.py` | 10 |
| 7 | Test, **une seule fois**, à la fin | `scripts/06_evaluate.py --split test` | 10 |
| à tout moment | Méthodes classiques (processeur seulement) | `scripts/08_classical.py` | 11 |

Aucun entraînement de perte avant les mesures de base de l'étape 2 : le pilote
et le réglage refusent de démarrer tant qu'elles n'existent pas.

---

## 2. Ce qu'on reprend de Musgrave, et ce qu'on laisse

### Repris

| Élément | Pourquoi |
|---|---|
| Réseau pré-entraîné sur ImageNet, entraîné en entier | C'est son dispositif |
| BatchNorm gelés pendant l'entraînement | Son choix, contre le surapprentissage |
| Augmentations : recadrage aléatoire redimensionné, symétrie horizontale | Ses augmentations, chiffrées (section 7) |
| Taux d'apprentissage propre aux poids des pertes (nsoftmax, ArcFace), réglé | Son choix |
| Plages de réglage centrées sur ses valeurs optimales | Meilleur point de départ que les valeurs par défaut |
| Personnes du test jamais vues à l'entraînement ni en validation | Sans ça, le résultat ne vaut rien |
| Test consulté une seule fois, à la fin | Éviter de se régler sur le test |
| Mêmes conditions et même nombre d'essais pour toutes les pertes | Comparaison équitable |
| Plusieurs graines et intervalles de confiance | Distinguer un vrai écart du hasard |
| MAP@R et P@1 | Ses métriques |
| Ligne de référence sans entraînement (« Pretrained ») | Savoir ce que les pertes apportent |

### Laissé de côté

| Chez Musgrave | Chez nous | Raison |
|---|---|---|
| BN-Inception | ResNet-18 supervisé ImageNet | Disponible dans torchvision, déjà essayé sur nos données |
| RMSprop, taux 10⁻⁶ | Recette du ResNet-18 : SGD (section 7) | Garder le plus possible la recette d'origine du réseau |
| 50 essais bayésiens par perte | 10 tirages au hasard | Budget de calcul |
| Validation croisée à 4 plis | Un seul découpage entraînement / validation | Budget de calcul |
| 10 relances | 3 graines | Budget de calcul |
| Lots de 32 (et une série en lots de 256) | Lots de 256 | Recette du ResNet-18 |
| 13 pertes | 4 pertes | On garde celles qui comptent |

### Ajouté (absent chez Musgrave)

Vérification (ROC, TAR à FAR fixé), histogrammes de similarité, alignement et
uniformité, PCA + t-SNE, bootstrap sur les identités, méthodes classiques.

---

## 3. Données

### CelebA

- Environ 200 000 images, 10 177 identités, environ 20 images par personne.
- Images alignées fournies : 178 × 218 pixels.
- Découpage officiel fourni (`list_eval_partition.txt`).

### Découpage et filtre (repris tels quels)

Les fichiers `splits/` sont **copiés de l'ancien dépôt sans modification** :
découpage officiel, identités disjointes entre parties, identités gardées si
elles ont **au moins 4 images** (les lots de 64 personnes × 4 images
l'exigent).

| Partie | Identités | Images |
|---|---|---|
| train | 7 689 | 161 602 |
| val | 924 | 19 720 |
| test | 951 | 19 850 |

Un **test automatique** vérifie qu'aucune identité n'apparaît dans deux parties.
Il n'est jamais désactivé.

### Recadrage (inchangé)

Carré de **128 px** centré sur le visage, boîte (24, 52, 152, 180) dans l'image
178 × 218, choisi le 9 novembre (voir l'ancien protocole, section 3).

### Cache des images

- Chaque image recadrée est stockée **une fois** en 128 × 128 × 3 octets, dans
  un fichier par partie (`train_128.npy`, `val_128.npy`, `test_128.npy`), dans
  l'ordre exact des fichiers `splits/`.
- Le cache est construit sur le disque local de Colab à chaque session, depuis
  le zip de CelebA, et sait reprendre après une coupure.
- Contrôle : formes, ordre identique aux fichiers `splits/`, aucune image
  manquante.

### Règles

- Jeux de données publics uniquement. Aucune photo produite par nous ou nos
  proches.

---

## 4. Réseau et tête

- **ResNet-18 supervisé sur ImageNet**, poids torchvision V1
  (`IMAGENET1K_V1`, fichier `resnet18-f37072fd.pth`), téléchargés depuis leur
  source officielle et **vérifiés par SHA-256** avant chargement. Pas de
  `torch.hub.load`.
- La couche de classification est retirée : le réseau sort un vecteur de **512**
  après la moyenne globale.
- **Tête** : une couche linéaire **512 → 128**, puis normalisation L2.
- **BatchNorm gelés** : leurs statistiques et leurs paramètres restent ceux
  d'ImageNet pendant tout l'entraînement.
- **Taille d'entrée : 224 px.** Le carré de 128 px est redimensionné en
  224 × 224 (interpolation bilinéaire) ; son contenu ne change pas.
  Normalisation ImageNet standard (moyenne et écart-type).
- Les matrices de poids propres à `nsoftmax` et `arcface` ne servent qu'à
  l'entraînement. L'évaluation n'utilise que les vecteurs de 128.

---

## 5. Mesures de base (sans entraînement)

À faire **avant tout entraînement de perte**, sur le **ResNet-18 de départ**,
exactement comme dans l'ancien protocole.

- **Brut** : vecteurs de 512 du réseau de départ, transformation d'évaluation
  (section 7). Version de référence : **ACP à 128 dimensions** ajustée sur
  l'entraînement (comme la ligne « Pretrained » de Musgrave). Les vecteurs de
  512 tels quels sont aussi mesurés.
- **LDA** : analyse discriminante linéaire ajustée sur les identités
  d'entraînement, projection en **128 dimensions**, appliquée aux autres
  parties. C'est la meilleure transformation linéaire pour séparer les
  personnes, obtenue par une formule.
- **Mesures** : toutes celles de la section 10 (recherche, vérification,
  histogrammes, alignement, uniformité).
- Pas de règle de décision : le réseau est fixé. Ces mesures servent de repères
  pour lire ce que les pertes apportent.

### Résultat (étape 2, validation, 22 novembre 2026)

Calculé sur le Mac (GPU Apple), dossier `runs/20261122-021645_2_baselines`,
résumé versionné dans `results/2_baselines_val.json`. En % sauf alignement et
uniformité.

| Version | MAP@R | P@1 | R-Precision | AUC | TAR à FAR 10⁻³ | Alignement | Uniformité |
|---|---|---|---|---|---|---|---|
| brut (512) | 3,02 | 23,3 | 5,70 | 74,8 | 2,9 | 0,32 | −0,84 |
| brut (ACP 128) | 3,01 | 22,0 | 5,75 | 81,1 | 3,7 | 1,43 | −3,56 |
| LDA 128 | 5,13 | 30,3 | 9,05 | 82,9 | 8,8 | 1,66 | −3,91 |

Lecture : le brut (3,0) est un peu sous celui de l'ancien ResNet-50 Barlow Twins
gelé (4,01), et la LDA ne gagne que 2 points sur le brut (5,13, contre 13,23
pour l'ancien réseau) : une transformation linéaire du réseau de départ récupère
peu d'information d'identité.

---

## 6. Les quatre fonctions de coût

| Nom | Type | Paramètres réglés |
|---|---|---|
| `nsoftmax` | classification, softmax normalisée | taux de la tête, taux des poids propres, *s* |
| `arcface` | classification à marge angulaire | taux de la tête, taux des poids propres, *s*, marge *m* |
| `contrastive` | paires | taux de la tête, marge positive, marge négative |
| `triplet` | triplets, sélection semi-difficile | taux de la tête, marge |

- **Lots identiques pour les 4 pertes : 64 personnes × 4 images = 256.** Les
  pertes de classification marchent avec n'importe quelle composition ; les
  pertes métriques ont besoin de plusieurs images par personne.
- `nsoftmax` est la softmax normalisée, c'est-à-dire ArcFace avec *m* = 0
  (vérifié par un test). *s* = 1 / température.
- `arcface` : la marge **monte de 0 à *m*** pendant les **3 premières
  époques**, linéairement, à chaque pas, comme dans l'essai dégelé du
  17 novembre.
- `contrastive` : marges positive et négative réglées, comme chez Musgrave.
- `triplet` : sélection des triplets **semi-difficiles** toujours active, non
  réglée (reprise de l'ancien protocole).
- Implémentation : **pytorch-metric-learning 3.x**, marge d'ArcFace en
  **radians**.

---

## 7. Recette d'entraînement

Gardée **le plus possible proche de celle du ResNet-18** (He et al., 2016,
reprise par torchvision : SGD, taux 0,1, momentum 0,9, weight decay 10⁻⁴, lots
de 256).

- **Optimiseur** : SGD, momentum 0,9 (sans Nesterov), **weight decay 10⁻⁴** sur
  tous les paramètres entraînés.
- **Taux d'apprentissage** :
  - tête : réglé (section 8) ;
  - réseau : **3 fois plus bas que la tête**, rapport fixé, non réglé (comme
    l'essai du 17 novembre : 0,03 et 0,1) ;
  - poids propres de `nsoftmax` et `arcface` : réglés à part, comme chez
    Musgrave.
- **Évolution du taux** : montée linéaire pendant la **1re époque** (de presque
  0 à sa valeur), puis **décroissance en cosinus** jusqu'à 0 sur la durée
  prévue de l'entraînement (20 époques au réglage, 30 au final). La recette
  d'origine (division par 10 aux époques 30 et 60 sur 90) ne peut pas être
  gardée telle quelle avec 20 ou 30 époques.
- **Augmentations** (entraînement seulement), reprises de Musgrave, les mêmes
  pour les 4 pertes :
  - recadrage aléatoire couvrant de **16 % à 100 %** de la surface du carré, rapport
    largeur / hauteur entre **3/4 et 4/3**, redimensionné en 224 × 224 ;
  - **symétrie horizontale** avec une probabilité de 50 %.
- **En évaluation**, sans hasard : le carré est redimensionné en 224 × 224,
  sans augmentation. Une même image donne toujours le même vecteur.
- **Arrêt précoce** (repris) : MAP@R mesuré sur la validation à la fin de chaque
  époque ; on garde la meilleure époque ; arrêt après **5 époques sans
  progrès**.
- **Précision mixte** **[à valider]** : sur GPU, le réseau calcule en bfloat16
  sur les cartes qui le gèrent nativement (A100, L4), en float16 avec mise à
  l'échelle des gradients sur les autres (T4) ; la perte est toujours calculée
  en float32. Même réglage pour les 4 pertes ; la précision de chaque époque est
  notée dans `history.csv`. Raison : vitesse. Les mesures de base (section 5)
  restent en float32 partout.
- **Point de reprise à chaque époque**, sur Drive (`checkpoint_every` dans
  `configs/train.yaml`) : un entraînement coupé reprend à l'époque suivante.
  Un run déjà commencé n'est repris que si ses réglages et sa recette sont
  identiques (`params.json`) ; sinon il faut un nouveau dossier.

---

## 8. Pilote et réglage

### Pilote (étape 3)

- **Une époque par perte**, réglages au **centre des plages** (milieu pour les
  plages linéaires, moyenne géométrique pour les plages logarithmiques),
  graine 100. **[à valider]**
- Cette époque est exactement la première époque d'un tirage de réglage : montée
  du taux pendant l'époque, cosinus calculé sur les 20 époques d'un tirage.
- Sorties : durée d'une époque, MAP@R de validation après une époque comparé au
  brut (ACP 128), estimation du temps total du réglage et des entraînements
  finaux.
- But : vérifier que chaque perte apprend avec ses plages, et mesurer le budget
  avant de s'engager. Si une perte n'apprend pas, on en discute avant le
  réglage.

### Réglage (étape 4)

- **Méthode** : tirage au hasard dans des plages **fixées avant tout
  lancement**.
- **Budget** : **10 tirages par perte**, **20 époques au plus par tirage**
  (valeur de départ, ajustable selon le pilote, la même pour les 4 pertes).
- **Graine de réglage : 100**, pour les tirages et pour l'entraînement.
- **Critère** : meilleur **MAP@R en validation** (meilleure époque du tirage).
- **Tirage qui diverge** (perte non finie, ou réseau qui ne sort plus que des
  vecteurs nuls) : arrêté et consigné « divergé » dans son `result.json` ; il
  compte dans les 10 tirages, avec son meilleur MAP@R avant la divergence (aucun
  s'il n'a fini aucune époque). Si tous les tirages d'une perte divergent, pas de
  graines finales pour elle : discussion en groupe. **[à valider]**

### Plages retenues

Chaque plage contient les valeurs optimales de Musgrave (colonnes CUB200,
Cars196, SOP et CUB200 en lots de 256) **et** celles qui ont marché dans l'essai
dégelé du 17 novembre, avec de la marge. Marges d'ArcFace converties de degrés
en radians.

| Perte | Paramètre | Musgrave (4 valeurs) | Essai du 17 nov. | Plage retenue |
|---|---|---|---|---|
| `nsoftmax` | *s* = 1 / température | 9,2 ; 11,3 ; 15,9 ; 14,1 | 32 | **6 à 40** |
| `arcface` | marge *m* | 0,41 ; 0,36 ; 0,33 ; 0,40 rad | 0,5 | **0,25 à 0,55 rad** |
| `arcface` | *s* | 100 ; 50 ; 220 ; 79 | 32 | **24 à 256**, logarithmique |
| `contrastive` | marge positive | ≤ 0 (équivaut à 0) ; 0,27 ; 0,29 ; 0,22 | non essayé | **0 à 0,40** |
| `contrastive` | marge négative | 0,38 ; 0,54 ; 0,51 ; 0,77 | non essayé | **0,30 à 1,0** |
| `triplet` | marge | 0,096 ; 0,119 ; 0,045 ; 0,137 | non essayé | **0,03 à 0,20** |
| toutes | taux de la tête (SGD) | non réglé (10⁻⁶, RMSprop) | 0,1 | **0,03 à 0,3**, logarithmique |
| `nsoftmax`, `arcface` | taux des poids propres (SGD) | 7×10⁻⁶ à 7×10⁻² avec RMSprop, non transposable | 0,1 | **0,01 à 0,3**, logarithmique |

Les valeurs trouvées avec l'ancien ResNet-50 gelé ne servent pas de repère :
dispositif différent (réseau gelé, RMSprop, lots de 32), et chacune n'était que
le meilleur de 5 tirages.

### Pourquoi MAP@R et pas la vérification pour choisir (repris)

- La vérification demande un seuil ou un FAR, donc un choix arbitraire de plus.
- La plupart des paires de personnes différentes sont faciles : l'aire sous la
  ROC départage mal les configurations.
- MAP@R n'a pas de seuil, regarde les voisins les plus proches et est moins
  bruitée (Musgrave : autocorrélation 0,81 contre 0,73 pour P@1).

---

## 9. Graines et statistiques

### Graines

- **Graines 0, 1, 2** pour les entraînements finaux, les mêmes pour les 4 pertes.
- **Graine 100** pour le réglage.
- Une graine fixe `random`, `numpy` et `torch` : initialisation de la tête et
  des poids propres, ordre des lots, augmentations.
- Sur GPU, certains calculs parallèles ne sont pas reproductibles au bit près :
  deux runs de même graine donnent des résultats très proches, pas identiques.
  Le test « même graine, même résultat » tourne sur processeur.
- Coût : 4 pertes × 3 graines = 12 entraînements finaux, 30 époques au plus.

### Intervalles de confiance

- **IC à 95 % de Student** sur les 3 graines, pour chaque chiffre. Avec 3
  graines, le coefficient vaut 4,30 : l'intervalle est large.
- **Comparaison appariée graine par graine** : pour chaque graine, écart entre
  deux pertes ; moyenne et IC à 95 % sur ces écarts. Deux pertes sont déclarées
  différentes si l'IC de leur écart exclut 0.
- **Bootstrap sur les identités** (nouveau pour les pertes ; l'ancien protocole
  s'en servait pour choisir le backbone) : les graines mesurent le hasard de
  l'entraînement, pas celui du choix des personnes du test.
  - On tire avec remise autant d'identités qu'il y en a dans la partie (951 au
    test, 924 en validation), 1 000 fois, graine 0.
  - Score d'un tirage : moyenne des AP@R par requête des images des identités
    tirées, chaque requête restant comparée à toute la partie. **[à valider]**
  - IC : du 2,5e au 97,5e centile. Pour comparer deux pertes, **même tirage**
    pour les deux.

### Consignation (reprise)

- Chaque run écrit son propre dossier, **jamais écrasé** : configuration,
  graine, métriques par époque, révision du dépôt, versions des bibliothèques,
  carte graphique, journal.
- Tout est consigné, y compris les runs ratés.

---

## 10. Évaluation

Toutes les mesures sortent de la même matrice de similarités cosinus entre
vecteurs de 128, normalisés L2.

### Sur la validation (pendant le travail)

MAP@R pour le réglage et l'arrêt ; mesures de base ; seuil de vérification.

### Sur le test (une seule fois, à la fin)

**Méthodes évaluées** : brut (ACP 128), LDA, les 4 pertes × 3 graines,
Eigenfaces, Fisherfaces, LBP.

**Vérification** :
- courbe ROC et aire sous la courbe ;
- **TAR à FAR = 10⁻¹, 10⁻², 10⁻³** ;
- seuil choisi **sur la validation**, au point d'égale erreur ; on rapporte au
  test le TAR, le FAR et l'exactitude équilibrée à ce seuil.

Paires (reprises) : toutes les paires positives, et un échantillon fixe de
**1 000 000 de paires négatives** tirées avec la graine 0, identiques pour
toutes les méthodes. Les paires négatives partagent des images : elles ne sont
pas indépendantes.

**Recherche par plus proches voisins** : **MAP@R**, R-Precision, **P@1**,
chaque image servant tour à tour de requête, les autres de références.

**Géométrie** :
- **histogrammes de similarité** des paires « même personne » et « personnes
  différentes », superposés ;
- **alignement et uniformité** (Wang & Isola, 2020), avec *t* = 2 :
  alignement sur toutes les paires positives, uniformité sur 1 000 000 de
  paires d'images tirées au hasard (graine 0). Descriptif seulement pour les
  pertes à marge.

**Visualisation** : PCA à 50 dimensions puis t-SNE en 2 dimensions, sur
**20 identités du test** tirées avec la graine 0, graine et perplexité (30)
fixées, mêmes identités et même réglage pour toutes les méthodes. C'est une
illustration : t-SNE garde les voisins proches mais déforme les distances
entre groupes. Les conclusions viennent des mesures chiffrées.

**Planche des plus proches voisins** sur les modèles finaux (reprise).

### Garde du test

Avant de lire le test, le script d'évaluation vérifie tout ce qui peut l'être :
les 4 pertes et toutes leurs graines, aucune divergée ni raccourcie, les
méthodes classiques calculées, les caches complets. Il écrit ensuite
`TEST_CONSULTED.json` dans le dossier des runs, puis refuse toute seconde
lecture. Une lecture interrompue avant la fin (coupure de Colab) ne peut être
refaite qu'avec `--retry-interrupted-test`, et chaque tentative est notée dans
le fichier.

---

## 11. Méthodes classiques

Mêmes partitions, même recadrage, mêmes métriques. Code repris de la copie de
travail du 16 novembre. **[à valider]** pour les réglages ci-dessous, choisis
dans cette copie sans discussion de groupe.

| Méthode | Référence | Principe | Comparaison |
|---|---|---|---|
| Eigenfaces | Turk & Pentland, 1991 | ACP à 128 dimensions sur les pixels | cosinus |
| Fisherfaces | Belhumeur et al., 1997 | ACP à 512 dimensions, puis LDA à 128 | cosinus |
| LBP | Ahonen et al., 2006 | histogrammes de motifs binaires locaux (8 voisins, rayon 2, uniformes) sur une grille 7 × 7 | distance du khi-deux |

- Eigenfaces et Fisherfaces : images en niveaux de gris, 64 × 64, tirées du
  même carré de 128 px. LBP : sur le carré de 128 px entier, en niveaux de gris
  (choix de la copie de travail). **[à valider]**
- LBP se compare par khi-deux et non par cosinus : entorse à « même mesure
  partout », à écrire.

---

## 12. Repris du protocole précédent

- Arrêt précoce sur MAP@R de validation, patience 5, meilleure époque gardée.
- Normalisation L2 des vecteurs avant tout calcul de similarité.
- Paires de vérification fixes (1 000 000 de négatives, graine 0) et seuil au
  point d'égale erreur sur la validation.
- Alignement et uniformité avec *t* = 2.
- Garde du test : une seule évaluation, marquée par `TEST_CONSULTED.json`.
- Tests automatiques : non-recouvrement des identités ; ArcFace avec *m* = 0
  égale nsoftmax ; métriques à 1 sur des vecteurs séparables ; même graine,
  même résultat.
- Un dossier par run, jamais écrasé ; tout est consigné, runs ratés compris.
- Graines fixées pour `random`, `numpy`, `torch` ; tous les réglages dans des
  fichiers de configuration ; scripts numérotés, la logique dans le paquet.

---

## 13. Calcul et pratique sur Colab

- **Volume prévu** : pilote 4 époques ; réglage 10 tirages × 4 pertes × 20
  époques = 800 époques au plus ; final 4 pertes × 3 graines × 30 époques = 360
  époques au plus. Environ 1 165 époques au plus, moins avec l'arrêt précoce.
  Budget recalculé avec la durée mesurée dans le pilote.
- **Où sont les choses** :
  - code : dépôt GitHub `srd-degel`, cloné à chaque session ;
  - CelebA : le zip déjà sur Drive, copié et décompressé à chaque session ;
  - cache des images : disque local de Colab, reconstruit à chaque session ;
  - résultats : `MyDrive/srd-degel/runs/` sur Drive.
- **Reprise** : chaque entraînement écrit un point de reprise à chaque époque ;
  le réglage et les entraînements finaux sautent ce qui est déjà terminé et
  reprennent ce qui a été coupé (le dossier le plus récent de l'étape, s'il n'est
  pas terminé).
- **Journal** : chaque run écrit ses messages, et une éventuelle erreur, dans
  `journal.txt` ; un run divergé garde son `result.json` (règle 5).
- **Essais rapides** (`--max-steps`) : dossiers à part, suffixés `_essai`, jamais
  pris pour des résultats du protocole ni pour le test.
- **Corbeille de Drive** : remplacer un point de reprise chaque époque envoie
  l'ancien dans la corbeille de Drive, qui compte dans le quota. À vérifier
  après le pilote ; si elle grossit trop, la vider ou augmenter
  `checkpoint_every`. **[à valider]**
- **Verrou** : un dossier de réglage ou d'entraînement final ne peut être
  utilisé que par une seule session à la fois (leçon de l'incident triplet du
  15 novembre). Une session qui trouve un verrou récent refuse de démarrer.
- **Plusieurs sessions en parallèle** : possible en donnant à chacune une perte
  différente.

---

## 14. Limites à écrire dans le rapport

**Sur ce qu'on mesure**
- ResNet-18 supervisé ImageNet, entraîné en entier sur CelebA : nos chiffres ne
  sont comparables à **aucun chiffre publié**.
- Le carré de 128 px est agrandi en 224 : l'agrandissement n'ajoute aucun
  détail.
- Détection et alignement hors périmètre.

**Sur l'équité de la comparaison**
- Nombre de paramètres réglés différent : 4 pour `arcface`, 3 pour `nsoftmax`
  et `contrastive`, 2 pour `triplet`. Avec le même nombre de tirages, les pertes
  à plus de paramètres explorent moins bien leur espace.
- Plages centrées sur les valeurs de Musgrave, trouvées sur d'autres images
  (oiseaux, voitures, produits), avec un autre réseau et RMSprop.
- 10 tirages au hasard au lieu de 50 essais bayésiens ; un seul découpage au
  lieu de 4 plis ; 3 graines au lieu de 10.

**Sur les statistiques**
- Avec 3 graines, les IC de Student sont larges.
- Les paires négatives ne sont pas indépendantes ; aux faibles FAR, peu de
  paires décident du résultat.
- Sur GPU, deux runs de même graine ne sont pas identiques au bit près.

**Sur les données**
- Célébrités : le style (coiffure, maquillage, époque) peut aider à reconnaître
  sans regarder le visage. Le recadrage le réduit sans l'éliminer.
- Pas d'annotations démographiques fiables : l'équité entre groupes n'est pas
  mesurable.

**Sur les mesures**
- Alignement et uniformité sont fondés théoriquement pour les pertes
  contrastives de type InfoNCE ; pour les pertes à marge, c'est descriptif.
- LBP comparé par khi-deux, pas par cosinus.
- t-SNE est une illustration, pas une mesure.

---

## Journal des modifications

| Date | Modification |
|---|---|
| 2026-11-22 | Création du protocole « réseau entraîné en entier », avec l'accord de l'encadrant. Décisions reprises des notes de travail d'Arsène, versions 1 à 7 (`docs/notes/`). Bonnes pratiques de l'ancien protocole reprises (section 12) |
| 2026-11-22 | Étape 1 sur le Mac : cache des carrés (train 161 602 images en 94 s, val 19 720 en 7 s). Étape 2 sur le Mac : mesures de base sur la validation (section 5, résultat). Le test n'est pas lu |
| 2026-11-22 | Après l'essai réel sur le Mac et la relecture du code : tirage divergé consigné sans arrêter le réglage ; pilote = première époque d'un tirage ; essais rapides à part ; garde du test renforcée ; précision float16 sur T4 ; journal par run ; points [à valider] ajoutés (sections 7, 8, 11, 13) |
