---
id: vis.01
title: "Les images comme données : pixels, canaux, normalisation et filtres"
summary: "Traite une image comme un tenseur de nombres, prépare-la exactement comme un modèle l'attend, et détecte des contours avec un filtre à fenêtre glissante écrit à partir de zéro."
minutes: 60
teaches: [vis.image-tensor, vis.filter, vis.augment]
requires: [la.matrix, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/01-image-fundamentals/docs/en.md (whole file: pixels as samples, channels, HWC vs CHW, byte ranges, colour spaces, grayscale weights, resizing and interpolation, ImageNet standardisation, preprocessing order). READ."
  - "ai-engineering-from-scratch/phases/04-computer-vision/04-image-classification/docs/en.md (Sec. 'Why augmentation works', flip, reflect-padded random crop, mixup; used for the augmentation section). READ."
---

## Où ça s'inscrit

Tout système de vision, du déverrouillage facial d'un téléphone au détecteur de tumeurs, commence de la même façon : une caméra produit des nombres, et un programme transforme ces nombres en le tenseur exact sur lequel un modèle a été entraîné. La plupart des bugs du type « le modèle est mauvais » en vision appliquée ne sont pas des bugs de modèle. Ce sont des bugs de prétraitement : BGR au lieu de RGB, `uint8` au lieu de `float`, canaux au mauvais endroit, ou mauvaise moyenne et mauvais écart-type. Cette leçon pose les fondations de tout le cours de vision.

Elle introduit aussi le filtre à fenêtre glissante. Si tu connais déjà les convolutions et les CNN (voir [[seq.01]]), les filtres d'ici sont les ancêtres conçus à la main des filtres qu'apprend un CNN ; sinon, rien ici ne demande plus que des boucles et des multiplications. Les leçons suivantes s'appuient sur celle-ci : [[vis.02]] entraîne des classifieurs sur ces tenseurs, [[vis.03]] y place des boîtes, [[vis.04]] étiquette chaque pixel.

## L'idée en termes simples

Une photographie est une grille de minuscules posemètres. Chaque posemètre indique « quelle quantité de lumière m'a atteint », et un pixel n'est que cette mesure, pas un petit carré coloré ayant une surface. Une image en niveaux de gris a une mesure par position. Une image en couleur en a trois : la quantité de lumière arrivée à travers un filtre rouge, vert et bleu. Une image en couleur est donc trois images en niveaux de gris empilées, une par canal.

Un modèle ressemble à un cuisinier très tatillon qui a appris une recette avec des ingrédients mesurés en grammes : si tu lui donnes les mêmes ingrédients en onces, le plat rate en silence. L'« unité » des images, c'est l'échelle et le centrage des nombres (entiers de 0 à 255, décimaux de 0 à 1, ou valeurs centrées et remises à l'échelle) ainsi que l'ordre des axes. Tu dois préparer ton image comme l'ont été les images d'entraînement.

Un filtre est un petit pochoir que l'on fait glisser sur l'image. À chaque position, tu multiplies les nombres du pochoir par les pixels qui se trouvent dessous et tu additionnes le tout. Un pochoir à poids égaux floute (il moyenne un voisinage). Un pochoir négatif à gauche et positif à droite répond à la question « est-ce plus clair à droite qu'à gauche ici ? », ce qui est exactement un détecteur de contours.

## Mise en place : les images comme tenseurs

Une image en couleur de hauteur $H$ et de largeur $W$ est un tableau à trois axes.

- Disposition **HWC** : forme $(H, W, 3)$, pixels entrelacés. Les fichiers, PIL, OpenCV et matplotlib l'utilisent.
- Disposition **CHW** : forme $(3, H, W)$, trois plans empilés. Les couches de convolution de PyTorch l'attendent, et un lot est $(N, 3, H, W)$ (NCHW).

Passer de l'une à l'autre n'est qu'une permutation d'axes : `chw = hwc.transpose(2, 0, 1)`.

Trois conventions numériques que tu rencontres en pratique :

| nom | dtype | plage | où |
|---|---|---|---|
| brut | `uint8` | 0..255 | fichiers, décodeurs |
| normalisé | `float` | 0..1 | après division par 255 |
| standardisé | `float` | environ $-2$..$2$ | après soustraction de la moyenne par canal et division par l'écart-type par canal |

La standardisation par canal $c$ est

$$
x'_{c} = \frac{x_{c}/255 - \mu_c}{\sigma_c}.
$$

Les valeurs ImageNet courantes sont $\mu = (0.485, 0.456, 0.406)$ et $\sigma = (0.229, 0.224, 0.225)$, la moyenne et l'écart-type de chaque canal sur l'ensemble d'entraînement ImageNet, calculés sur des pixels dans 0..1. Un modèle pré-entraîné sur ImageNet les attend ; un modèle entraîné sur un autre jeu de données attend les statistiques de ce jeu.

L'ordre complet suivi par les pipelines de production est : décoder (`uint8` HWC), corriger l'ordre des couleurs, redimensionner (petit côté), recadrer à la taille du modèle, diviser par 255, standardiser, transposer en CHW, ajouter l'axe de lot. Changer l'ordre des étapes change ce que voit le modèle.

### Espaces colorimétriques et niveaux de gris

RGB est le format de capture, mais d'autres coordonnées sont utiles. HSV sépare la couleur (teinte) de la luminosité (valeur), ce qui aide pour le seuillage de couleur classique. YCbCr sépare la luminosité (Y) de deux canaux de différence de couleur ; JPEG et les codecs vidéo compressent plus fortement les canaux de couleur parce que l'œil y est moins sensible. Pour un modèle, tu fournis presque toujours du RGB, ou du gris quand la couleur est une gêne (documents, OCR).

Le gris est une somme pondérée, pas une simple moyenne, parce que l'œil est le plus sensible au vert :

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B .
$$

### Redimensionnement

Les modèles ont une taille d'entrée fixe, donc les images sont redimensionnées. Le plus proche voisin copie le pixel source le plus proche (en blocs, mais le seul choix sûr pour les images d'étiquettes contenant des identifiants de classe). Bilinéaire mélange les quatre pixels environnants selon la distance (lisse ; le choix par défaut pour l'entraînement). Bicubique et Lanczos sont plus nets et plus lents, utilisés pour des images que des personnes vont regarder. Redimensionner puis recadrer au centre conserve le rapport d'aspect et supprime une bordure ; redimensionner avec remplissage conserve chaque pixel (détection, OCR) ; l'étirement direct est bon marché mais déforme les formes.

## Dérivation : un filtre comme produit scalaire glissant

Prends une image en niveaux de gris $I\in\mathbb{R}^{H\times W}$ et un noyau $K\in\mathbb{R}^{k_h\times k_w}$. L'image filtrée **valid** (sans remplissage, pas de 1) est

$$
O[i,j]=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1} K[u,v]\; I[i+u,\,j+v].
$$

C'est le produit scalaire entre le noyau et la fenêtre de l'image commençant en $(i,j)$. Les bibliothèques de deep learning appellent cela « convolution » bien que mathématiquement ce soit de l'intercorrélation : la vraie convolution retourne d'abord le noyau. Pour un noyau appris, le retournement n'a pas d'importance, car le réseau peut apprendre la version retournée.

**Taille de la sortie.** La fenêtre doit tenir dans l'image, donc $i$ parcourt $H-k_h+1$ valeurs et $j$ en parcourt $W-k_w+1$ :

$$
O\in\mathbb{R}^{(H-k_h+1)\times(W-k_w+1)}.
$$

Un noyau $3\times3$ perd donc un pixel sur chaque bord ; compléter l'image avec des zéros ou des valeurs réfléchies conserve la taille.

**Flou.** $K=\tfrac19\mathbf 1_{3\times3}$ remplace chaque pixel par la moyenne de son voisinage, ce qui lisse le bruit mais adoucit aussi les contours.

**Contours de Sobel.** Le noyau

$$
K_x=\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}
$$

prend les pixels de la colonne de droite moins ceux de la colonne de gauche, en pondérant la ligne du milieu deux fois. Sur une région de luminosité constante, les poids ont une somme nulle, donc la réponse est nulle. À travers une marche verticale, elle est grande. Le noyau transposé $K_y=K_x^\top$ répond aux contours horizontaux. L'intensité du contour combine les deux :

$$
G=\sqrt{G_x^2+G_y^2},\qquad G_x=I\star K_x,\ G_y=I\star K_x^\top .
$$

**Histogramme.** Compter combien de pixels tombent dans chaque classe de luminosité donne un résumé unidimensionnel de l'image qui ignore la position. C'est la vérification la plus rapide qu'une image est sous-exposée (masse accumulée près de 0) ou écrêtée (accumulée à 255).

## Exemple travaillé

Toutes les valeurs ci-dessous sont produites par le code de l'exercice.

**Standardiser un pixel.** Le pixel $(R,G,B)=(210,140,30)$ divisé par 255 donne $(0.8235, 0.5490, 0.1176)$. Soustrais la moyenne ImageNet et divise par l'écart-type : $(1.478,\ 0.415,\ -1.282)$. Sa valeur de gris est $0.299\cdot210+0.587\cdot140+0.114\cdot30=148.39$.

**Une marche de contour verticale.** L'image $5\times5$ dont chaque ligne est $(0,0,0,10,10)$ a un contour entre les colonnes 2 et 3.

- Le $G_x$ de Sobel avec le noyau $3\times3$ valid a la forme $3\times3$ et chaque ligne vaut $(0,\ 40,\ 40)$. Pour la fenêtre sur les colonnes 1..3, la colonne de droite moins la colonne de gauche vaut $10-0$ pour chacune des trois lignes, pondérées par $1+2+1=4$, ce qui donne $40$. Pour la fenêtre sur les colonnes 0..2, tout vaut 0.
- $G_y=0$ partout, donc la magnitude vaut aussi $(0,40,40)$ par ligne.
- Le flou moyenneur $3\times3$ donne des lignes $(0,\ 3.33,\ 6.67)$ : la marche nette est devenue une rampe.

**Histogramme.** Les 6 nombres `[0,10,200,255,128,64]` dans 4 classes de largeur 64 produisent les effectifs $(2,1,1,2)$.

## Augmentation (aperçu)

Un réseau n'a aucune invariance intégrée aux retournements, aux recadrages ou aux changements d'éclairage. Lui montrer des copies modifiées de chaque image d'entraînement, avec la même étiquette, lui apprend à ignorer ces changements. La règle : une augmentation doit **préserver l'étiquette**. Un retournement horizontal est sûr pour « chien » mais pas pour du texte ni pour une classification gauche/droite ; une rotation peut transformer un 6 en 9. Choix courants :

- **Retournement aléatoire** : miroir avec probabilité $0.5$.
- **Recadrage aléatoire** : compléter (le remplissage par réflexion vaut mieux que des bordures noires) et recadrer à la taille d'origine avec un décalage aléatoire.
- **Variation de couleur** : petits changements aléatoires de luminosité et de contraste.
- **Mixup** : mélanger deux images et leurs étiquettes, $x=\lambda x_i+(1-\lambda)x_j$, $y=\lambda y_i+(1-\lambda)y_j$.

L'augmentation s'applique uniquement aux images d'entraînement. Les images de validation et de test ne reçoivent que la partie déterministe du pipeline (redimensionner, recadrer, standardiser). [[vis.02]] montre comment mesurer si elle aide.

## Implémentation

```python
import numpy as np

def to_gray(rgb):                       # (H, W, 3) -> (H, W)
    return rgb.astype(float) @ np.array([0.299, 0.587, 0.114])

def standardize(img_uint8, mean, std):  # (H, W, 3) uint8 -> (3, H, W) float
    x = img_uint8.astype(float) / 255.0
    x = (x - np.asarray(mean)) / np.asarray(std)
    return x.transpose(2, 0, 1)

def conv2d_valid(img, kernel):          # (H, W), (kh, kw) -> (H-kh+1, W-kw+1)
    H, W = img.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out

Kx = np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
def sobel_magnitude(gray):
    return np.sqrt(conv2d_valid(gray, Kx) ** 2 + conv2d_valid(gray, Kx.T) ** 2)
```

Les bibliothèques de production font la même chose avec des tenseurs par lots et conscients du périphérique : `torchvision.transforms` pour le pipeline (`Resize`, `CenterCrop`, `ToTensor`, `Normalize`) et `torch.nn.functional.conv2d` ou `filter2D` d'OpenCV pour les filtres. Les deux boucles Python imbriquées ci-dessus sont lentes volontairement ; les bibliothèques les vectorisent.

## Formes

| objet | forme |
|---|---|
| image RGB, HWC | $(H,W,3)$ |
| même image, CHW | $(3,H,W)$ |
| lot pour un modèle | $(N,3,H,W)$ |
| moyenne / écart-type par canal | $(3,)$, diffusé sur le dernier axe de HWC (donc standardise **avant** de transposer) |
| filtrage valid avec un noyau $k_h\times k_w$ | $(H-k_h+1,\ W-k_w+1)$ |

## Idées reçues

- **« Un pixel est un carré coloré. »** C'est un échantillon d'intensité lumineuse par canal ; redimensionner et pivoter rééchantillonnent ces mesures.
- **« Diviser par 255 suffit comme normalisation. »** Cela ne fait que ramener les valeurs dans 0..1. Un modèle entraîné sur des entrées standardisées voit toujours la mauvaise distribution.
- **« La convolution retourne le noyau en deep learning. »** Les bibliothèques implémentent l'intercorrélation ; pour des noyaux appris la différence n'importe pas, mais copier le sens d'un noyau de manuel, si.
- **« Le gris est la moyenne de R, G, B. »** L'œil pondère surtout le vert ; utilise la somme pondérée.
- **« N'importe quel redimensionnement convient. »** Le plus proche voisin sur un masque de segmentation est correct ; le bilinéaire sur un masque invente des identifiants de classe qui n'existent pas.
- **« Plus d'augmentation, c'est toujours mieux. »** Une augmentation qui change l'étiquette (faire pivoter des chiffres, retourner du texte) abîme les données.

## Points clés à retenir

- Une image est un tenseur $(H,W,C)$ ou $(C,H,W)$ ; vérifie le dtype, la plage, l'ordre des canaux et la disposition avant toute chose.
- Reproduis exactement le prétraitement d'entraînement : ordre des couleurs, redimensionnement, recadrage, division par 255, standardisation avec les bonnes statistiques, transposition.
- Un filtre est un produit scalaire glissant ; la sortie valid a pour taille $(H-k_h+1)\times(W-k_w+1)$.
- Le flou moyenne, Sobel différencie ; un noyau dont les poids ont une somme nulle ignore les régions plates.
- Les augmentations doivent préserver l'étiquette et ne servent qu'à l'entraînement.

## Explique-le toi-même

1. Un collègue donne directement un tableau `uint8` à un modèle pré-entraîné et obtient des prédictions assurées mais fausses. Explique ce qui ne va pas et liste les étapes exactes qui le corrigeraient.
2. Pourquoi le noyau de Sobel a-t-il des poids de somme nulle, et que signifierait une réponse exactement nulle en un pixel ?

## Pour aller plus loin

- La liste de lectures de la leçon source (Poynton sur les espaces colorimétriques, la documentation des transformations et des modèles de torchvision).
- [[seq.01]] pour la convolution comme couche apprenable avec canaux, pas et pooling.

## Critères de maîtrise

Tu sais convertir entre HWC et CHW et entre `uint8`, 0..1 et plages standardisées sans aide-mémoire, prédire la forme de sortie d'un filtre valid, implémenter et tester un filtre à partir de zéro, et expliquer pourquoi chaque étape d'un pipeline de type ImageNet existe.
