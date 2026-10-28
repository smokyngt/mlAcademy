---
id: aud.03
title: "Classification audio : du k-NN sur MFCC aux CNN et à SpecAugment"
summary: "Transformer un extrait en vecteur de taille fixe ou en image, le classifier avec un k-NN ou un CNN, augmenter les données avec des masques temporels et fréquentiels, et évaluer honnêtement avec des matrices de confusion, le rappel par classe et le F1 macro."
minutes: 85
teaches: [aud.classify, aud.specaugment]
requires: [aud.mfcc, aud.mel, ml.generalization, nn.layer, dl.early-stopping]
sources:
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/03-audio-classification/docs/en.md (k-NN on MFCC mean+variance, 2D CNN on log-mels, AST/BEATs/Whisper-encoder backbones, class imbalance, mixup, SpecAugment, metrics, the CNN code and its parameter-count claim). READ."
  - "math/coursAvanceApprentissageAudio.pdf, PDF p.3-8 (principle of data augmentation: label-preserving and realistic; time shift, noise, volume, time stretching, pitch shifting, compressor) and p.14 (quiz on which augmentation to avoid for music identification). READ."
  - "math/coursIntroApprentissageAudio.pdf, PDF p.40-47 (three families of audio networks, long raw sequences, variable durations, adaptive/global pooling, zero-padding batches) and p.49-53 (spectrogram as an image, rectangular kernels, CNN comparison, summary). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/en.md (metric per task: top-1, mAP, macro F1, per-class recall). READ."
---

## Où cela s'inscrit

« Est-ce un chien ou une sirène ? », « Quelle langue est-ce ? », « Quelle commande a été prononcée : oui, non, stop ? », « Le locuteur est-il en colère ? » : tout cela relève de la *classification audio*, une étiquette (ou plusieurs) par extrait. C'est le premier endroit où les caractéristiques de [[aud.02]] rencontrent un algorithme d'apprentissage, et c'est la façon la plus simple de voir toute la recette de l'apprentissage automatique audio : **forme d'onde, spectrogramme, caractéristiques, modèle, métrique**. La même recette, avec une couche de sortie différente, alimente la reconnaissance vocale ([[aud.04]]).

Tu as besoin des caractéristiques log-mel et MFCC de [[aud.02]], de l'idée de généralisation et de validation (concept `ml.generalization`), et de la notion de couche de réseau de neurones (concept `nn.layer`).

## L'idée en termes simples

Un extrait a une longueur variable, mais un classifieur veut une entrée de taille fixe. Il y a deux issues.

- **Résumer.** Calcule des caractéristiques trame par trame, puis remplace tout l'extrait par quelques nombres : la moyenne de chaque caractéristique dans le temps et sa variation. « Cet extrait est, en moyenne, brillant et aigu, et assez stable » est une description parfaitement valable d'une sirène ou d'un sifflet. Une règle du plus proche voisin dit alors « l'extrait ressemble le plus à ces extraits étiquetés ».
- **Regarder l'image.** Traite le spectrogramme log-mel comme une image avec des axes temps et fréquence, et laisse un réseau convolutif trouver des motifs locaux (un balayage montant, un clic vertical, un bourdonnement horizontal), puis agrège sur le temps pour ignorer *quand* ils se sont produits. Le réseau peut apprendre quoi chercher, mais il lui faut beaucoup d'exemples.

Les jeux de données audio sont petits, déséquilibrés et bruités. Deux habitudes font l'essentiel du travail : l'**augmentation**, qui montre au réseau des versions légèrement dégradées de chaque extrait pour qu'il ne s'appuie pas sur des accidents ; et l'**évaluation honnête**, qui examine les erreurs classe par classe au lieu d'un seul chiffre d'exactitude.

## Caractéristiques au niveau de l'extrait et k-NN

Soit $C\in\mathbb R^{T\times K}$ la matrice des MFCC (ou log-mel) d'un extrait : $T$ trames, $K$ coefficients. Le vecteur de résumé est

$$
\mathbf z=\big[\ \mu_1,\dots,\mu_K,\ \sigma_1,\dots,\sigma_K\ \big]\in\mathbb R^{2K},\qquad
\mu_j=\frac1T\sum_t C_{tj},\quad \sigma_j=\sqrt{\frac1T\sum_t (C_{tj}-\mu_j)^2 } .
$$

Le $T$ variable disparaît. (Le cours en français énonce le même problème pour les réseaux : les sons ont des durées différentes, et le pooling global transforme une carte de caractéristiques `(canaux, longueur)` en un vecteur `(canaux)` indépendant de la longueur.)

**k plus proches voisins.** Stocke les vecteurs d'entraînement étiquetés $\mathbf z_i$. Pour classifier un nouvel extrait, trouve les $k$ extraits d'entraînement de plus grande similarité cosinus

$$
\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\,\|\mathbf b\|}
$$

et renvoie l'étiquette la plus fréquente parmi eux. Il n'y a pas d'entraînement ; le modèle *est* les données. La similarité cosinus ignore l'échelle globale d'un vecteur (qui dépend de l'intensité sonore), la distance euclidienne non ; l'une ou l'autre peut servir, et le bon choix est une affaire empirique.

**Formes.** `Xtr: (N_train, 2K)`, `Xte: (N_test, 2K)`, similarité `Xte @ Xtr.T` après normalisation des lignes : `(N_test, N_train)` ; pour chaque ligne, prends les indices des $k$ plus grandes entrées et vote.

### Exemple détaillé 1 : un problème bruité à quatre classes

Nous synthétisons des extraits de 0,25 s à 8 kHz en quatre classes (le générateur exact est dans l'exercice) : un son à 440 Hz, un son à 520 Hz, un balayage montant et du bruit blanc, chacun avec un bruit gaussien additif d'écart type 3 (le signal est donc noyé : chaque son a une amplitude de 1). Caractéristiques : puissance logarithmique dans 16 bandes de fréquence de largeur égale sur des trames de 200 échantillons, résumée par la moyenne et l'écart type dans le temps (32 nombres par extrait). Avec 30 extraits par classe et la graine 0, nous séparons 80 extraits d'entraînement / 40 de test (effectifs de test par classe 13, 8, 9, 10).

| $k$ | exactitude en test |
|---|---|
| 1 | 0.775 |
| 3 | 0.900 |
| 5 | 0.875 |

Pour $k=3$, la matrice de confusion (lignes = classe vraie, colonnes = classe prédite) est

$$
\begin{pmatrix}12&0&1&0\\1&6&1&0\\1&0&8&0\\0&0&0&10\end{pmatrix}
$$

Les deux sons proches (440 et 520 Hz) sont les classes les plus difficiles : la classe 520 Hz a un rappel de $6/8=0.75$. Le bruit n'est jamais confondu avec un son. L'exactitude globale $36/40=0.9$ masque cela ; la matrice de confusion montre où le modèle échoue.

## CNN sur des images log-mel

Un spectrogramme log-mel `(T, M)` devient une image à un canal `(1, T, M)` ; un lot est `(B, 1, T, M)`. Une convolution 2D avec $C_{in}$ canaux d'entrée, $C_{out}$ canaux de sortie et un noyau $k\times k$ a $C_{in}C_{out}k^2+C_{out}$ paramètres et réduit chaque axe selon

$$
o=\left\lfloor\frac{i+2p-k}{s}\right\rfloor+1 ,
$$

(longueur d'entrée $i$, remplissage $p$, pas $s$ ; la même formule apparaît dans le cours en français pour les convolutions 1D). Avec $k=3$, $p=1$, $s=1$, la longueur est conservée ; un max-pooling $2\times2$ divise chaque axe par deux.

La leçon du dépôt donne un petit réseau : trois blocs `Conv(3x3, pad 1) + ReLU` avec 32, 64, 128 canaux, deux max-poolings $2\times2$ après les deux premiers, un average pooling global, et une couche linéaire vers les classes. Pour une entrée `(1, 100, 80)` (100 trames, 80 mels), les formes sont

| étape | forme (canaux, temps, mel) |
|---|---|
| entrée | (1, 100, 80) |
| conv1 | (32, 100, 80) |
| pool | (32, 50, 40) |
| conv2 | (64, 50, 40) |
| pool | (64, 25, 20) |
| conv3 | (128, 25, 20) |
| average pooling global | (128,) |
| linéaire | (50,) pour 50 classes |

Paramètres : $320+18\,496+73\,856+6\,450=99\,122$ pour 50 classes (vérifié par l'exercice). **La leçon du dépôt décrit ce réseau comme « 3M de paramètres » ; notre décompte est d'environ 0,1M**, donc ce chiffre de la source est incohérent avec son propre code. Compter les paramètres soi-même est une bonne habitude.

**Notes de conception.** (1) Les deux axes d'un spectrogramme ne sont *pas* interchangeables, contrairement aux axes d'une photographie : un motif décalé dans le temps est le même événement, un motif décalé en fréquence est un son différent. Le cours en français suggère donc des noyaux rectangulaires dans certaines architectures. (2) Le pooling global sur le temps rend le réseau indépendant de la longueur de l'extrait. (3) Le cours en français liste trois familles d'entrée : les modèles sur spectrogramme mel, les modèles sur des fenêtres de forme d'onde brute (la première couche apprend ses propres filtres) et les modèles sur échantillons bruts individuels ; la famille spectrogramme est peu coûteuse et classique, les modèles sur échantillons bruts sont coûteux mais souvent les plus performants. (4) Une comparaison citée dans le même cours, de CNN d'images sur des spectrogrammes audio, a trouvé que des architectures d'images plus profondes (ResNet-50 et Inception) étaient nettement meilleures qu'un réseau entièrement connecté sur cette tâche d'étiquetage de vidéos.

**Le choix par défaut de 2026, d'après la leçon du dépôt :** ne pas entraîner depuis zéro. Prends un backbone audio pré-entraîné (l'Audio Spectrogram Transformer, un ViT sur des patchs de spectrogramme ; BEATs ; ou un encodeur Whisper gelé) et entraîne une petite tête. La leçon du dépôt rapporte des exactitudes sur ESC-50 d'environ 97 % pour les meilleurs modèles pré-entraînés ; nous n'avons pas vérifié ces chiffres.

## Augmentation de données

L'augmentation génère de nouveaux exemples d'entraînement en transformant des exemples existants. Le cours en français énonce deux exigences : la transformation ne doit **pas changer l'information utilisée pour la supervision**, et elle doit correspondre à des **distorsions réalistes**. Sa liste pour l'audio : décalage temporel, bruit additif, changement de volume et étirement temporel (change la durée, pas la hauteur), plus changement de hauteur (change la hauteur, pas la durée) et un compresseur (compression de la dynamique).

*Laquelle éviter ?* Le cours demande quelle augmentation est dangereuse pour identifier une chanson à partir d'un enregistrement (à la Shazam) : le changement de hauteur, car la hauteur *est* l'information d'identification. De même, le changement de hauteur serait nuisible pour le genre d'un locuteur ou la classification de notes.

### SpecAugment : masquer le spectrogramme

SpecAugment (Park et al. ; résumé par la leçon du dépôt, article original non lu) travaille sur l'image log-mel `S` de forme `(T, M)` :

- **Masque de fréquence.** Choisis une largeur $f$ uniformément dans $\{0,\dots,F\}$ et un début $f_0$ uniformément dans $\{0,\dots,M-f\}$ ; mets $S[:,f_0:f_0+f]$ à une constante (zéro, ou la moyenne).
- **Masque temporel.** Choisis $t$ uniformément dans $\{0,\dots,T_{\max}\}$ et un début $t_0\in\{0,\dots,T-t\}$ ; mets $S[t_0:t_0+t,:]$ à une constante.

Répète plusieurs fois. Le réseau apprend à classifier avec des parties de l'image manquantes, comme un auditeur qui reconnaît encore une chanson malgré une mauvaise ligne téléphonique ou une toux. La leçon du dépôt suggère un masque temporel de 20 et un masque fréquentiel de 10 comme point de départ pour l'exercice. Les masques ne font qu'effacer, donc l'étiquette est préservée tant que la zone masquée n'est pas toute la preuve.

**Exemple détaillé 2 : des masques sur un tableau jouet.** Sur un tableau $10\times6$ de uns, un masque temporel de largeur 3 démarrant à la ligne 7 et un masque fréquentiel de largeur 1 à la colonne 3 mettent à zéro $3\cdot6+1\cdot10-3\cdot1=25$ des 60 entrées (le recouvrement, 3 cellules, est compté une fois) ; la somme passe de 60 à 35. C'est exactement ce que produisent les fonctions `time_mask`/`freq_mask` de l'exercice.

### Mixup

Le mixup mélange deux exemples et leurs étiquettes, $x=\lambda x_1+(1-\lambda)x_2$ et $y=\lambda y_1+(1-\lambda)y_2$ (étiquettes one-hot). Avec $\lambda=0.7$, $y_1=(1,0)$, $y_2=(0,1)$, la nouvelle cible est $(0.7,0.3)$. Entraîne avec l'entropie croisée contre la cible douce. L'audio mélangé est aussi physiquement plausible : deux sons qui jouent ensemble.

### Déséquilibre des classes

Les jeux de données réels sont déséquilibrés (jeux de type UrbanSound environ 10:1 ; AudioSet a une très longue traîne, d'après la leçon du dépôt). Remèdes : échantillonnage équilibré *à l'entraînement seulement*, augmentation des classes rares, et métriques par classe.

## Évaluation

Pour $K$ classes, construis la matrice de confusion $C_{ab}$ = nombre d'extraits de vraie classe $a$ prédits comme $b$. Alors

$$
\text{recall}_a=\frac{C_{aa}}{\sum_b C_{ab}},\qquad
\text{precision}_a=\frac{C_{aa}}{\sum_b C_{ba}},\qquad
F1_a=\frac{2\,\text{precision}_a\,\text{recall}_a}{\text{precision}_a+\text{recall}_a},
$$

exactitude $=\sum_a C_{aa}/\sum_{ab}C_{ab}$, et le **F1 macro** est la moyenne non pondérée des $F1_a$ sur les classes. Utilise l'exactitude top-1 / top-5 pour des classes exclusives avec des données équilibrées, la **mAP** (précision moyenne moyenne) pour l'étiquetage multi-étiquettes comme AudioSet, et le F1 macro plus le rappel par classe quand les classes sont déséquilibrées (leçon d'évaluation du dépôt).

**Exemple détaillé 3.** $C=\begin{pmatrix}8&1&1\\2&6&2\\0&1&9\end{pmatrix}$ (30 extraits). Rappels $0.8, 0.6, 0.9$ ; précisions $0.8, 0.75, 0.75$ ; $F1=0.8,\ 0.667,\ 0.818$ ; F1 macro $=0.7616$ ; exactitude $=23/30=0.767$.

**Pourquoi le F1 macro bat l'exactitude en cas de déséquilibre.** Avec 95 négatifs et 5 positifs, un modèle qui répond toujours « négatif » a une exactitude de 0,95, mais un F1 de $0.974$ pour la classe négative et de $0$ pour la positive : F1 macro $=0.487$.

**Évite la fuite.** Les extraits découpés dans le même long enregistrement ne sont pas indépendants. Si certains sont en entraînement et d'autres en test, le k-NN et les CNN peuvent reconnaître l'enregistrement plutôt que la classe sonore et rapporter des scores gonflés. Sépare par enregistrement source ou par locuteur (des benchmarks comme ESC-50 fournissent des plis prédéfinis pour cette raison). Choisis les hyperparamètres ($k$, force de l'augmentation) sur une séparation de validation, jamais sur le jeu de test (concept `ml.generalization`).

## Implémentation

```python
import numpy as np

def knn_predict(Xtr, ytr, Xte, k=3):
    a = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    b = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
    sim = b @ a.T                                   # (N_test, N_train) cosine similarities
    out = []
    for row in sim:
        nn = np.argsort(-row)[:k]
        out.append(np.bincount(ytr[nn]).argmax())  # majority vote
    return np.array(out)

def confusion_matrix(y, p, K):
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C

def macro_f1(C):
    tp = np.diag(C).astype(float)
    prec = tp / np.maximum(C.sum(0), 1)
    rec = tp / np.maximum(C.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return f1.mean()

def spec_augment(spec, rng, n_time=2, T_max=20, n_freq=2, F_max=10, fill=0.0):
    out = spec.copy()
    T, F = spec.shape
    for _ in range(n_freq):
        f = int(rng.integers(0, F_max + 1)); f0 = int(rng.integers(0, F - f + 1))
        out[:, f0:f0 + f] = fill
    for _ in range(n_time):
        t = int(rng.integers(0, T_max + 1)); t0 = int(rng.integers(0, T - t + 1))
        out[t0:t0 + t, :] = fill
    return out
```

Bibliothèques : `torchaudio.transforms.TimeMasking`/`FrequencyMasking`, `torch_audiomentations` pour l'augmentation de la forme d'onde, `sklearn.neighbors.KNeighborsClassifier`, `sklearn.metrics.classification_report`, et Hugging Face `transformers` (`ASTForAudioClassification`) pour les backbones pré-entraînés.

## Idées reçues courantes

- **« L'exactitude suffit. »** En cas de déséquilibre, elle cache les classes qui échouent ; rapporte toujours le rappel par classe.
- **« Augmente tout. »** Une transformation qui change l'étiquette (changement de hauteur pour l'identification de chansons) revient à injecter du bruit au mauvais endroit.
- **« Un CNN sur spectrogrammes est un classifieur d'images, donc les astuces d'ImageNet se transfèrent telles quelles. »** Les recadrages aléatoires en fréquence, les rotations et les retournements changent le sens ; les décalages temporels et le masquage sont les plus sûrs.
- **« Un $k$ plus grand est toujours plus lisse et meilleur. »** Il échange variance contre biais ; dans l'exemple détaillé 1, $k=5$ n'était pas meilleur que $k=3$ sur 40 extraits de test (et 40 extraits donnent une erreur standard de plusieurs pourcents, donc des différences d'un ou deux extraits sont du bruit).
- **« L'exactitude en test d'une séparation aléatoire d'extraits estime l'exactitude en déploiement. »** Pas quand des extraits d'un même enregistrement se trouvent des deux côtés.

## Points clés à retenir

- Des résumés de taille fixe (moyenne et écart type des MFCC dans le temps) plus un k-NN cosinus constituent une solide référence pour de petits jeux de données ; les CNN sur des log-mels `(1,T,M)` avec pooling global gèrent les longueurs variables.
- Taille de sortie d'une convolution : $\lfloor(i+2p-k)/s\rfloor+1$ ; nombre de paramètres $C_{in}C_{out}k^2+C_{out}$.
- Les augmentations doivent préserver l'étiquette et imiter des distorsions réelles ; SpecAugment masque des bandes aléatoires de temps et de fréquence ; le mixup mélange entrées et étiquettes.
- Évalue avec la matrice de confusion, le rappel par classe et le F1 macro ; évite la fuite entre entraînement et test.
- En pratique, pars d'un backbone pré-entraîné et entraîne une tête.

## Explique-le toi-même

1. Un collègue augmente un classifieur de chants d'oiseaux avec un changement de hauteur allant jusqu'à 6 demi-tons. Argumente pour et contre.
2. Ton modèle à 10 classes a 96 % d'exactitude mais deux classes avec 20 % de rappel. Explique comment c'est possible et ce que tu rapporterais.

## Pour aller plus loin

- Dépôt, phase 06, leçon 03 (choix de backbones et pile 2026) ; la leçon cite Gong et al. (AST), Chen et al. (BEATs), Park et al. (SpecAugment) et Piczak (ESC-50), dont nous n'avons lu aucun.
- Hershey et al. (2016) est cité dans le cours en français pour la comparaison de CNN d'images sur l'audio ; non lu directement.
- [[aud.04]] montre comment les mêmes encodeurs gèrent des sorties de longueur variable.

## Critères de maîtrise

Tu sais construire des statistiques MFCC au niveau de l'extrait, implémenter un k-NN cosinus, calculer à la main et en NumPy une matrice de confusion, le rappel par classe et le F1 macro, dériver la taille de sortie et le nombre de paramètres d'un petit CNN, implémenter un masquage de type SpecAugment et le mixup, et nommer une augmentation risquée pour une tâche donnée.
