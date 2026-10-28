---
id: vis.05
title: "Vision transformers et vision auto-supervisée"
summary: "Transforme une image en une séquence de tokens de patchs pour un transformer, compare ses biais inductifs à ceux d'un CNN, et apprends des caractéristiques sans étiquettes avec la perte contrastive NT-Xent."
minutes: 80
teaches: [vis.patch-embed, vis.contrastive]
requires: [trf.attention, trf.position, vis.image-tensor, vis.augment, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/14-vision-transformers/docs/en.md (Sec. 'The Problem' and 'The Concept': pipeline, patch embedding, class token, positional embedding, encoder block, pre-LN, patch-size trade-off, DeiT recipe, Swin vs ConvNeXt, MAE; the Build It code was not read). READ."
  - "ai-engineering-from-scratch/phases/04-computer-vision/17-self-supervised-vision/docs/en.md (Sec. 'The Problem' and 'The Concept': three families, SimCLR/InfoNCE, DINO teacher-student, MAE masking ratio, linear-probe evaluation; the Build It code was not read). READ."
---

## Où ça s'inscrit

Jusqu'en 2020 environ, « modèle de vision » voulait dire « réseau convolutif ». Le transformer ([[trf.01]], [[trf.04]]) a ensuite pris le contrôle du langage, et la question était de savoir s'il pouvait aussi lire des images. La réponse, le **vision transformer (ViT)**, est le backbone de la plupart des modèles de vision actuels et du côté image de CLIP dans [[vis.06]]. La seconde moitié de cette leçon aborde l'autre goulot d'étranglement : les étiquettes sont chères, les images ne le sont pas. L'**apprentissage auto-supervisé** produit des caractéristiques utiles à partir d'images non étiquetées, qui alimentent ensuite les recettes de sonde linéaire et de fine-tuning de [[vis.02]]. La perte contrastive que tu implémentes ici est l'ancêtre direct de la perte de CLIP.

## L'idée en termes simples

Un transformer lit une séquence de tokens. Une phrase est naturellement une séquence de mots ; une image ne l'est pas. L'astuce du ViT est brutale : découper l'image en une grille de petits carrés (disons $16\times16$ pixels), aplatir chaque carré en un long vecteur, et traiter les carrés comme des mots. Tout le reste est un transformer standard. Comme l'attention n'a aucune notion de « gauche » ou de « au-dessus », une étiquette de position est ajoutée à chaque patch, et un token spécial appris de « résumé » est placé en tête, son état final étant lu comme la description de l'image.

Un réseau convolutif vient avec des hypothèses intégrées : des pixels proches comptent ensemble (localité) et le même motif signifie la même chose partout (équivariance par translation). Un ViT n'en a presque aucune ; il doit les apprendre. C'est une faiblesse avec peu de données et une force avec beaucoup, ce qui explique pourquoi les premiers ViT perdaient face aux CNN sur ImageNet seul mais gagnaient une fois pré-entraînés sur des jeux de données bien plus grands, et pourquoi de meilleures recettes d'entraînement ont ensuite comblé l'écart.

Pour l'apprentissage auto-supervisé, prends cette expérience de pensée. Prends une photo, fabrique-en deux versions éditées différentes (recadrées et recolorées différemment). Un bon extracteur de caractéristiques devrait dire « même chose » pour ces deux-là, et « chose différente » pour des versions éditées d'autres photos. Personne n'a besoin d'étiqueter la photo « chat » ; la supervision vient du fait de savoir quelles deux éditions proviennent d'un même original. S'entraîner avec cette règle force le réseau à conserver ce qui survit aux éditions (le contenu) et à écarter ce qui ne survit pas (le recadrage exact, la teinte).

## Plongement de patchs

Une image de taille $H\times W$ avec $C$ canaux et une taille de patch $p$ (avec $p$ divisant $H$ et $W$) donne $N=(H/p)(W/p)$ patchs. Aplatis chaque patch en un vecteur de longueur $C p^2$ :

$$
x_{\text{patches}}\in\mathbb{R}^{N\times Cp^2}.
$$

Une projection linéaire apprise $W_E\in\mathbb{R}^{Cp^2\times D}$ avec un biais $b\in\mathbb{R}^D$ associe chacun à un token de largeur $D$ :

$$
z_0^{\text{patch}}=x_{\text{patches}}W_E+b\in\mathbb{R}^{N\times D}.
$$

Dans les frameworks, c'est une seule convolution de taille de noyau $p$ et de pas $p$ (les patchs ne se chevauchent donc pas) avec $D$ canaux de sortie, suivie de l'aplatissement de la grille spatiale ; c'est exactement la même application linéaire appliquée à chaque patch. Voir [[seq.01]] pour les formes de convolution.

Puis :

1. **Token de classe.** Un vecteur appris $x_{\text{cls}}\in\mathbb{R}^D$ est placé en tête, donnant $N+1$ tokens. Sa sortie après le dernier bloc est la représentation de l'image fournie au classifieur.
2. **Plongement positionnel.** Une matrice apprise $E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$ est ajoutée : $z_0=[x_{\text{cls}};z_0^{\text{patch}}]+E_{\text{pos}}$. Sans elle, l'attention est équivariante par permutation et ne peut pas distinguer le patch en haut à gauche de celui en bas à droite (voir [[trf.03]]).
3. **Blocs d'encodeur.** $L$ blocs de transformer à pré-LayerNorm : $z\leftarrow z+\mathrm{MSA}(\mathrm{LN}(z))$, puis $z\leftarrow z+\mathrm{MLP}(\mathrm{LN}(z))$, où le MLP est $D\to4D\to D$ avec GELU. La pré-normalisation entraîne les empilements profonds plus stablement que la post-normalisation.
4. **Tête.** Une couche linéaire ou un petit MLP sur la sortie du token de classe.

Pour ViT-B/16 en $224\times224$ : $N=(224/16)^2=196$ patchs, $197$ tokens avec le token de classe, largeur $D=768$, $12$ blocs et $12$ têtes (environ 86M de paramètres). Remarque la coïncidence $Cp^2=3\cdot16\cdot16=768=D$ ici, donc la projection de patch est carrée ; en général ce n'est pas le cas.

### Taille de patch et coût

L'attention compare chaque token à tous les autres, donc une matrice d'attention a $(N+1)^2$ entrées par tête et par couche et son coût croît quadratiquement avec le nombre de tokens. Diviser par deux le côté du patch quadruple les tokens et multiplie par $16$ la taille de la matrice d'attention. Des patchs plus grands sont plus rapides mais perdent du détail spatial ; $16$ est le compromis standard, et les conceptions hiérarchiques (Swin) utilisent de petits patchs avec des fenêtres locales.

### Biais inductif : CNN contre ViT

| propriété | CNN | ViT |
|---|---|---|
| localité | intégrée (petits noyaux) | apprise (l'attention peut porter partout) |
| équivariance par translation | intégrée (partage de poids) | seulement via le plongement de patchs et les positions apprises |
| champ réceptif | croît avec la profondeur | global dès le premier bloc |
| appétit en données | plus faible | plus élevé, sauf si régularisé ou pré-entraîné |
| passage à l'échelle | sature plus tôt | continue de s'améliorer avec les données et la taille |

Conséquences pratiques. Un ViT simple entraîné sur ImageNet-1k seul était moins bon que les ResNet dans le travail original ; avec un pré-entraînement sur un jeu de données bien plus grand, il a gagné. DeiT a ensuite atteint de solides résultats sur ImageNet-1k par la seule recette d'entraînement : forte augmentation (RandAugment, mixup, CutMix, effacement aléatoire), profondeur stochastique, augmentation répétée, et distillation optionnelle depuis un enseignant CNN. Swin ajoute des fenêtres locales décalées (un a priori de localité), et ConvNeXt montre qu'un CNN modernisé peut les égaler, donc la différence durable tient autant à la recette d'entraînement qu'à l'attention contre la convolution. Sur du matériel embarqué, les CNN restent compétitifs.

## Apprentissage auto-supervisé

Trois familles :

- **Contrastive** (SimCLR, MoCo, CLIP) : rapprocher les plongements des paires positives, écarter les négatives.
- **Enseignant-élève** (DINO, BYOL) : un élève prédit la sortie d'un enseignant mis à jour lentement ; pas de négatifs explicites.
- **Reconstruction masquée** (MAE, BEiT) : cacher la plus grande partie de l'image et la reconstruire.

### Dérivation : la perte NT-Xent (SimCLR)

Prends un lot de $N$ images. Fais deux augmentations aléatoires de chacune, ce qui donne $2N$ vues. Passe-les dans un encodeur et une petite tête de projection pour obtenir des plongements $z_1,\dots,z_{2N}\in\mathbb{R}^d$, et normalise chacun en L2. La similarité de deux vues est le cosinus

$$
s_{ij}=\frac{z_i^\top z_j}{\lVert z_i\rVert\,\lVert z_j\rVert}=z_i^\top z_j\quad(\text{after normalising}).
$$

Pour une vue $i$, soit $i^+$ son partenaire positif (l'autre vue de la même image). La tâche est : parmi les $2N-1$ autres vues, identifier $i^+$. C'est une classification à $(2N-1)$ classes, on utilise donc l'entropie croisée softmax avec les logits $s_{ik}/\tau$ pour tout $k\ne i$, où $\tau>0$ est une **température** :

$$
\ell_i=-\log\frac{\exp(s_{i,i^+}/\tau)}{\sum_{k\ne i}\exp(s_{ik}/\tau)}
=-\frac{s_{i,i^+}}{\tau}+\log\sum_{k\ne i}\exp\!\Big(\frac{s_{ik}}{\tau}\Big),
$$

$$
L=\frac1{2N}\sum_{i=1}^{2N}\ell_i .
$$

C'est l'« entropie croisée normalisée à température », NT-Xent. C'est l'entropie croisée de [[vis.02]] où les « classes » sont les autres échantillons du lot. Propriétés à connaître :

- **L'auto-similarité est exclue** ($k\ne i$) ; sinon l'appariement trivial $s_{ii}=1$ dominerait.
- **Température.** Diviser par un petit $\tau$ affûte le softmax, concentrant le gradient sur les négatifs les plus difficiles (les mauvaises vues les plus similaires) ; un grand $\tau$ traite tous les négatifs de la même façon. Les valeurs typiques sont autour de $0.1$ (la source cite $0.1$ comme standard).
- **Négatifs.** La perte a $2N-2$ négatifs par ancre, donc plus d'échantillons dans le lot donne une tâche plus difficile et plus informative. La source dit que SimCLR a besoin de lots de 512 à 8192, et que MoCo découple le nombre de négatifs de la taille du lot en gardant une file de plongements passés.
- **Valeur de contrôle.** Si toutes les similarités sont égales, la perte vaut $\log(2N-1)$, le niveau du hasard. Quand $\tau\to\infty$ les logits s'aplatissent, donc la perte tend vers cette valeur quels que soient les plongements.
- **Invariance d'échelle.** À cause de la normalisation, multiplier un plongement par un nombre positif quelconque ne change rien.

### Enseignant-élève (DINO) en bref

Deux réseaux partagent une architecture : l'**élève**, entraîné par descente de gradient, et l'**enseignant**, une moyenne mobile exponentielle des poids de l'élève ($\theta_t\leftarrow m\theta_t+(1-m)\theta_s$ avec $m\approx0.996$). Tous deux voient des vues augmentées différentes ; l'élève est entraîné pour que sa distribution de sortie corresponde à celle de l'enseignant sur l'autre vue (une entropie croisée entre les deux). Il n'y a pas de négatifs, donc l'effondrement vers une sortie constante est le danger. DINO l'évite en **centrant** les sorties de l'enseignant (en soustrayant une moyenne glissante, ce qui empêche une dimension de dominer) et en les **affûtant** (une basse température, ce qui empêche l'effondrement vers l'uniforme). DINOv2 passe cela à l'échelle et ses caractéristiques sont, selon la source, un choix par défaut actuel pour la recherche et la prédiction dense.

### Auto-encodage masqué (MAE) en bref

Masque environ 75 % des patchs, encode seulement les 25 % visibles avec un grand ViT, et entraîne un petit décodeur à reconstruire les pixels des patchs masqués (erreur quadratique moyenne sur les patchs masqués uniquement). Le ratio est élevé parce que les pixels voisins sont fortement corrélés : ne cacher que 15 % (comme BERT pour le texte) permet au modèle de combler les trous par simple interpolation, alors que 75 % le force à représenter le contenu. Après le pré-entraînement, le décodeur est jeté.

### Évaluer les caractéristiques auto-supervisées

La mesure standard est la **sonde linéaire** ([[vis.02]]) : geler l'encodeur, entraîner une couche linéaire avec des étiquettes, rapporter la précision top-1. La source cite des précisions de sonde linéaire sur ImageNet d'environ 71 % (SimCLR ResNet-50), 77 % (DINO ViT-S/16), 76 % (MAE ViT-L/16) et 86 % (DINOv2 ViT-g/14) ; considère-les comme indicatives, car les configurations d'entraînement diffèrent. Le fine-tuning ajoute typiquement quelques points mais mélange aussi l'effet du ré-entraînement.

## Exemple travaillé

Tous les nombres sont reproduits par le code de l'exercice.

**Découper une minuscule image en patchs.** L'image $4\times4$ à un canal avec les pixels $0..15$ (ligne par ligne) et $p=2$ donne $4$ patchs de $4$ nombres : patch 0 (haut-gauche) $=[0,1,4,5]$, patch 1 (haut-droite) $=[2,3,6,7]$, patch 2 (bas-gauche) $=[8,9,12,13]$, patch 3 $=[10,11,14,15]$. Reconstituer l'image restaure exactement l'image.

**Comptabilité de ViT-B/16.** $3\times224\times224$ avec $p=16$ : $196$ patchs de $768$ nombres ; la projection de patch a $768\cdot768+768=590{,}592$ paramètres ; $197$ tokens (avec le token de classe) ; une matrice d'attention par tête a $197^2=38{,}809$ entrées. Avec $p=8$ : $784$ patchs et $784^2=614{,}656$ entrées sans le token de classe, soit $16$ fois les $196^2=38{,}416$ du cas $p=16$.

**NT-Xent.** $N=2$ images avec des plongements orthogonaux $(1,0)$ et $(0,1)$, les deux vues identiques, $\tau=0.5$. Pour l'ancre $(1,0)$ les logits contre les trois autres vues sont : partenaire $s=1\to2$, les deux vues de l'autre image $s=0\to0$ chacune. Donc $\ell=-\log\frac{e^2}{e^2+1+1}=\log(1+2e^{-2})=0.2395$. Par symétrie les quatre ancres donnent la même valeur, donc $L=0.2395$. Avec des plongements aléatoires sans lien et un $\tau$ très grand, la perte vaut environ $\log(2N-1)$ ; pour $N=8$ cela fait $\log15=2.708$.

## Implémentation

```python
import numpy as np

def patchify(img, p):                 # (C, H, W) -> (N, C*p*p)
    C, H, W = img.shape
    x = img.reshape(C, H // p, p, W // p, p).transpose(1, 3, 0, 2, 4)
    return x.reshape((H // p) * (W // p), C * p * p)

def nt_xent(z1, z2, tau):             # (N, d), (N, d) -> scalar
    N = len(z1)
    Z = np.concatenate([z1, z2], axis=0)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T / tau
    np.fill_diagonal(S, -np.inf)                 # exclude self-similarity
    pos = (np.arange(2 * N) + N) % (2 * N)       # partner of view i
    m = S.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(S - m).sum(axis=1, keepdims=True))).ravel()
    return float(np.mean(lse - S[np.arange(2 * N), pos]))
```

En production : `timm` et `torchvision.models.vision_transformer` fournissent ViT ; `transformers` fournit les checkpoints ViT, DeiT, DINOv2 et MAE avec la même API de chargement que les modèles de texte ; `lightly` et les dépôts officiels DINO/MAE implémentent les recettes auto-supervisées à grande échelle.

## Formes

| objet | forme |
|---|---|
| image | $(C,H,W)$, lot $(B,C,H,W)$ |
| patchs | $(N,Cp^2)$, $N=HW/p^2$ |
| tokens après plongement et token de classe | $(N+1,D)$, lot $(B,N+1,D)$ |
| matrice d'attention par tête | $(N+1,N+1)$ |
| plongements de la tête de projection pour NT-Xent | $(2N_{\text{batch}},d)$ |
| matrice de similarité | $(2N_{\text{batch}},2N_{\text{batch}})$ |

## Idées reçues

- **« Les ViT ne peuvent pas fonctionner sans énormément de données. »** Ils ont besoin d'a priori ou de recettes solides (augmentation, pré-entraînement, auto-supervision), pas forcément de données à l'échelle de JFT.
- **« Un ViT regarde l'image globalement, donc la taille de patch n'importe pas. »** La taille de patch fixe le nombre de tokens, donc la résolution du détail et le coût quadratique de l'attention.
- **« Le token de classe contient une étiquette. »** C'est un vecteur appris de type requête ; il rassemble l'information des patchs par l'attention.
- **« L'apprentissage contrastif a besoin d'étiquettes. »** Il a seulement besoin de savoir quelles vues proviennent de la même image.
- **« Les négatifs sont d'autres classes. »** Dans un lot aléatoire, certains négatifs peuvent partager la classe de l'ancre (« faux négatifs ») ; ce bruit est toléré à grande échelle.
- **« La température est un détail. »** Elle contrôle à quel point la perte se concentre sur les négatifs difficiles et affecte fortement les résultats.
- **« Le pré-entraînement auto-supervisé rend le fine-tuning inutile. »** La sonde mesure les caractéristiques ; le fine-tuning ajoute généralement de la précision.

## Points clés à retenir

- ViT = découpage en patchs, projection linéaire, token de classe, plongement positionnel, transformer standard ; $N=HW/p^2$ et le coût de l'attention évolue en $(N+1)^2$.
- Les CNN intègrent la localité et l'équivariance par translation ; les ViT les apprennent, donc ils dépendent des données et de la recette d'entraînement.
- NT-Xent est une entropie croisée softmax sur les autres vues du lot, avec similarité cosinus et température.
- DINO évite l'effondrement avec un enseignant EMA, le centrage et l'affûtage ; MAE masque 75 % des patchs.
- Évalue avec une sonde linéaire sur des caractéristiques gelées.

## Explique-le toi-même

1. Explique pourquoi un ViT a besoin de plongements positionnels et comment le nombre de tokens change quand la taille de patch est divisée par deux.
2. Dérive la perte NT-Xent comme un problème de classification et explique les rôles de la normalisation, de la température et du nombre de négatifs.

## Pour aller plus loin

- Les deux leçons sources (références ViT, DeiT, Swin, ConvNeXt, MAE, SimCLR, DINO).
- [[trf.04]] pour le bloc transformer, [[vis.06]] pour l'objectif contrastif étendu aux paires image-texte.

## Critères de maîtrise

Tu sais calculer le nombre de tokens et de paramètres pour une taille de patch donnée, écrire le découpage en patchs et son inverse, expliquer la différence de biais inductif entre CNN et ViT, et implémenter et vérifier NT-Xent (valeur au niveau du hasard, invariance d'échelle, vues alignées contre vues sans lien).
