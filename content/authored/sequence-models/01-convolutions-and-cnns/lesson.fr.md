---
id: seq.01
title: Convolutions et CNN
summary: "La corrélation croisée comme produit scalaire glissant, l'arithmétique des tailles de sortie, les canaux, le pooling, la rétropropagation complète d'une convolution dérivée et vérifiée par gradient numérique, les champs récepteurs et la route de LeNet à ResNet."
minutes: 100
teaches: [seq.conv, seq.cnn-arch]
requires: [nn.backprop, la.matrix, calc.chain, nn.activation]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/02-convolutions-from-scratch/docs/en.md (whole file up to the Build It / Use It sections: sliding kernel, 5x5 worked input and Sobel kernel, output-size formula, padding, stride, channels, parameter count, im2col, receptive field). READ."
  - "ai-engineering-from-scratch/phases/04-computer-vision/03-cnns-lenet-to-resnet/docs/en.md (The Concept: LeNet-5 layout, AlexNet, VGG, Inception, degradation problem, ResNet; Build It steps 1-2 for the LeNet parameter count 61,706). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md (The Concept and TextCNN: 1D convolution over word embeddings as a learnable n-gram detector). READ."
---

## Objectifs d'apprentissage

- Expliquer une couche de convolution comme une petite couche dense glissée sur l'entrée avec des **poids partagés**, et calculer sa sortie par un produit scalaire à la main.
- Calculer les tailles de sortie, les nombres de paramètres et les champs récepteurs pour n'importe quelle pile de couches de convolution et de pooling.
- Dériver la passe arrière d'une couche de convolution (gradients pour $W$, $b$ et l'entrée) et l'exprimer avec des produits matriciels.
- Implémenter conv2d en passe avant et arrière avec NumPy et les vérifier par différences finies.
- Dire ce que LeNet, AlexNet, VGG et ResNet ont chacun apporté, et pourquoi une connexion résiduelle aide les gradients.

## Prérequis

La rétropropagation avec les dimensions ([[f2n.10]]) et la boucle d'entraînement ([[f2n.11]]). Tu dois être à l'aise avec les produits matriciels et la règle de la chaîne.

## Où cela s'inscrit

Jusqu'ici, chaque entrée était un vecteur plat. Les données réelles ont une **structure** : les pixels ont des voisins, les échantillons audio ont des voisins, les mots d'une phrase ont des voisins. Une couche entièrement connectée l'ignore. Donner une image RVB $224\times224$ à une couche dense de 1000 unités demande $224\cdot224\cdot3\cdot1000+1000=150{,}529{,}000$ paramètres, et un chat qui se déplace de trois pixels vers la droite ressemble à une entrée complètement différente.

Les convolutions règlent les deux problèmes avec une seule idée : regarder un petit voisinage, avec le même détecteur partout. Cela rend les images abordables et c'est aussi la première des trois façons de traiter les **séquences** dans ce cours. Une convolution 1D sur des embeddings de mots est un détecteur de n-grammes entraînable (l'idée du TextCNN), et la même pensée par fenêtre glissante réapparaît quand nous rencontrons les réseaux récurrents ([[seq.03]]) et l'attention ([[seq.05]]).

## L'idée en mots simples

Imagine un petit tampon sur lequel est imprimé un motif, disons un motif en forme de bord vertical. Tu presses le tampon sur chaque endroit possible d'une photo et, à chaque endroit, tu notes à quel point le tampon correspond à ce qu'il y a dessous. La feuille de scores que tu obtiens est une **carte de caractéristiques** (feature map) : claire là où la photo contient ce motif, sombre ailleurs. Le tampon est le **noyau** (ou filtre).

Trois choses rendent cela puissant.

- **Un tampon, beaucoup d'endroits.** Les mêmes quelques nombres sont réutilisés à chaque position. C'est le *partage de paramètres*, et c'est pourquoi une couche de convolution est minuscule comparée à une couche dense.
- **Local.** Chaque score ne regarde qu'un petit morceau, donc les détecteurs apprennent des motifs locaux comme des bords, des coins et des textures.
- **Beaucoup de tampons.** Une couche possède de nombreux noyaux, chacun produisant sa propre carte. Empile les couches et les tampons suivants correspondent à des motifs de motifs : les bords deviennent des coins, les coins des yeux, les yeux des visages.

Le pooling est l'autre ingrédient : après la mise en correspondance, rétrécis la feuille en ne gardant que le score le plus fort de chaque petit bloc. Tu perds la position exacte, tu gardes « ce motif était quelque part par ici ».

## Définition formelle

Pour un canal d'entrée, une image $x\in\mathbb{R}^{H\times W}$ et un noyau $w\in\mathbb{R}^{K\times K}$, la sortie (valide, pas de 1) est

$$
y_{ij}=\sum_{u=0}^{K-1}\sum_{v=0}^{K-1} w_{uv}\,x_{i+u,\,j+v}+b .
$$

À proprement parler, c'est une **corrélation croisée** (le noyau n'est pas retourné comme dans la convolution des manuels). Toutes les bibliothèques d'apprentissage profond font cela ; comme le noyau est appris, le retournement ne changerait rien sauf un renommage des poids.

Avec plusieurs canaux, une couche a une entrée $x\in\mathbb{R}^{C_{in}\times H\times W}$, des poids $W\in\mathbb{R}^{C_{out}\times C_{in}\times K\times K}$ et un biais $b\in\mathbb{R}^{C_{out}}$. Avec un remplissage (padding) $P$ (des zéros ajoutés sur les quatre côtés) et un pas (stride) $S$ :

$$
y_{o,i,j}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1} W_{o,c,u,v}\;x^{\mathrm{pad}}_{c,\;iS+u,\;jS+v}.
$$

Chaque valeur de sortie est un produit scalaire entre le noyau $W_{o}$ (tous les canaux d'entrée à la fois, un bloc de $C_{in}K^2$ nombres) et une fenêtre de l'entrée remplie.

### Taille de sortie

Le coin supérieur gauche de la fenêtre peut se trouver aux lignes $0,S,2S,\dots$ tant que la fenêtre tient dans la hauteur remplie $H+2P$. Le dernier départ valide $iS$ vérifie $iS+K\le H+2P$, donc $i\le (H+2P-K)/S$, et en comptant à partir de 0 :

$$
H_{out}=\Big\lfloor\frac{H-K+2P}{S}\Big\rfloor+1 .
$$

Pareil pour la largeur. Un remplissage « same » avec un pas de 1 signifie $H_{out}=H$, c'est-à-dire $P=(K-1)/2$ pour $K$ impair.

### Paramètres

$$
\#\text{params}=C_{out}\,C_{in}\,K^2+C_{out}.
$$

Il ne **dépend pas** de $H,W$. Une couche $3\to64$ avec des noyaux $3\times3$ a $64\cdot3\cdot9+64=1792$ paramètres que l'image soit de $32\times32$ ou de $4000\times4000$. Faire la même application avec une couche dense sur une image $32\times32$ demanderait $(3\cdot32\cdot32)(64\cdot32\cdot32)\approx2.0\times10^{8}$ poids.

### Pooling

Le max pooling de fenêtre $k$ et de pas $k$ prend le maximum de chaque bloc $k\times k$ sans recouvrement, indépendamment par canal. Il n'a **aucun paramètre**, divise la résolution par deux pour $k=2$, et ajoute un peu de robustesse aux petits décalages. L'average pooling prend la moyenne à la place (LeNet l'utilisait).

### Champ récepteur

Le champ récepteur d'une unité est le morceau de l'*entrée* qui peut l'influencer. Suis deux nombres en parcourant les couches : la taille courante du champ $r$ (départ à 1) et le « saut » $j$, la distance en pixels d'entrée entre deux unités voisines (départ à 1). Une couche de noyau $K$ et de pas $S$ met à jour

$$
r\leftarrow r+(K-1)\,j,\qquad j\leftarrow j\,S.
$$

Trois convolutions $3\times3$ de pas 1 empilées donnent $r=1+3\cdot2=7$. Deux convolutions $3\times3$ donnent $r=5$, comme une seule convolution $5\times5$, mais utilisent $2\cdot9C^2=18C^2$ poids au lieu de $25C^2$ (pour $C$ canaux en entrée et en sortie) et ont une non-linéarité de plus entre les deux. Cette observation est toute la conception de VGG.

## Dérivation : la passe arrière

Soit $L$ une perte scalaire et $\bar Y:=\partial L/\partial Y$ le gradient amont, de la forme de $Y$ : $(C_{out},H_{out},W_{out})$. Écris $x^{\mathrm{pad}}$ sous la forme $\tilde x$. La formule avant, de nouveau :

$$
Y_{o,i,j}=b_o+\sum_{c,u,v}W_{o,c,u,v}\,\tilde x_{c,\,iS+u,\,jS+v}.
$$

**Biais.** $b_o$ apparaît à chaque position du canal $o$, donc par la règle de la chaîne $\partial Y_{o,i,j}/\partial b_o=1$ et

$$
\bar b_o=\sum_{i,j}\bar Y_{o,i,j}\qquad\text{(shape }C_{out}\text{).}
$$

**Poids.** $\partial Y_{o,i,j}/\partial W_{o,c,u,v}=\tilde x_{c,iS+u,jS+v}$ et le poids est utilisé à *chaque* $(i,j)$ (c'est exactement le partage de paramètres), donc les contributions s'additionnent :

$$
\bar W_{o,c,u,v}=\sum_{i,j}\bar Y_{o,i,j}\;\tilde x_{c,\,iS+u,\,jS+v}.
$$

C'est elle-même une corrélation croisée de l'entrée remplie avec le gradient amont.

**Entrée.** $\partial Y_{o,i,j}/\partial\tilde x_{c,p,q}=W_{o,c,u,v}$ chaque fois que $p=iS+u$ et $q=jS+v$, et 0 sinon. Un pixel $\tilde x_{c,p,q}$ est utilisé par toutes les fenêtres qui le couvrent, donc

$$
\bar{\tilde x}_{c,p,q}=\sum_{o}\sum_{\substack{i,j,u,v:\\ iS+u=p,\ jS+v=q}}\bar Y_{o,i,j}\,W_{o,c,u,v}.
$$

Pour $S=1$ c'est une corrélation « complète » de $\bar Y$ avec le noyau retourné spatialement. Enfin $\bar x$ est $\bar{\tilde x}$ avec le remplissage rogné (les pixels de remplissage sont des constantes, pas des variables).

### La même chose en produits matriciels (im2col)

Empile chaque fenêtre de $\tilde x$ comme une colonne d'une matrice $X_{col}\in\mathbb{R}^{(C_{in}K^2)\times(H_{out}W_{out})}$ (c'est `im2col`). Aplatis les poids en $W_{mat}\in\mathbb{R}^{C_{out}\times(C_{in}K^2)}$ et $\bar Y$ en $\bar Y_{mat}\in\mathbb{R}^{C_{out}\times(H_{out}W_{out})}$. Alors

$$
Y_{mat}=W_{mat}X_{col}+b\mathbf{1}^\top .
$$

C'est une couche linéaire ordinaire où le « batch » est l'ensemble des fenêtres, donc les formules de [[f2n.10]] s'appliquent telles quelles, avec les transpositions de l'autre côté parce qu'ici les échantillons sont des colonnes :

$$
\bar W_{mat}=\bar Y_{mat}\,X_{col}^\top,\qquad \bar b=\bar Y_{mat}\mathbf 1,\qquad \bar X_{col}=W_{mat}^\top\,\bar Y_{mat}.
$$

Dimensions : $(C_{out}\times HW)(HW\times C_{in}K^2)=C_{out}\times C_{in}K^2$, et $(C_{in}K^2\times C_{out})(C_{out}\times HW)=C_{in}K^2\times HW$. La dernière étape, `col2im`, prend chaque colonne de $\bar X_{col}$ et l'**additionne** dans la fenêtre d'où elle a été copiée. Les fenêtres se chevauchent, donc chaque pixel d'entrée reçoit la somme de ses contributions : c'est la somme sur $(i,j,u,v)$ dans la formule ci-dessus.

## Exemple détaillé

Prenons l'entrée $5\times5$ de la leçon source et un noyau de type Sobel (pas 1, sans remplissage) :

$$
x=\begin{pmatrix}1&2&0&1&2\\0&1&3&1&0\\2&1&0&2&1\\1&0&2&1&3\\2&1&1&0&1\end{pmatrix},\qquad
w=\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}.
$$

Taille de sortie : $(5-3+0)/1+1=3$. La valeur en haut à gauche est le produit scalaire de $w$ avec la fenêtre $3\times3$ en haut à gauche $\begin{pmatrix}1&2&0\\0&1&3\\2&1&0\end{pmatrix}$ :

- ligne 0 : $1\cdot1+2\cdot0+0\cdot(-1)=1$
- ligne 1 : $0\cdot2+1\cdot0+3\cdot(-2)=-6$
- ligne 2 : $2\cdot1+1\cdot0+0\cdot(-1)=2$

donc $Y_{00}=1-6+2=-3$. Faire les neuf fenêtres donne (vérifié par le code de l'exercice)

$$
Y=\begin{pmatrix}-3&0&3\\0&-3&0\\1&-2&-3\end{pmatrix}.
$$

Avec un pas de 2 et un remplissage de 1, la taille de sortie est $\lfloor(5-3+2)/2\rfloor+1=3$ de nouveau, avec les valeurs $\begin{pmatrix}-5&2&3\\-3&-3&6\\-2&1&1\end{pmatrix}$.

**Passe arrière**, avec le gradient amont le plus simple $\bar Y=\mathbf{1}$ (que des uns, c'est-à-dire perte $=\sum Y$) : alors $\bar b=9$, et $\bar W_{uv}$ est la somme des $x_{i+u,j+v}$ sur les neuf fenêtres, c'est-à-dire la somme d'un bloc $3\times3$ de $x$ décalé de $(u,v)$ :

$$
\bar W=\begin{pmatrix}10&11&10\\10&11&13\\10&8&11\end{pmatrix},\qquad
\bar x=\begin{pmatrix}1&1&0&-1&-1\\3&3&0&-3&-3\\4&4&0&-4&-4\\3&3&0&-3&-3\\1&1&0&-1&-1\end{pmatrix}.
$$

Vérifie le coin : $x_{00}$ appartient à une seule fenêtre (celle en haut à gauche) à la position de noyau $(0,0)$, donc $\bar x_{00}=w_{00}=1$. Le pixel central $x_{22}$ appartient aux neuf fenêtres, chaque fois à une position de noyau différente, donc $\bar x_{22}=\sum_{uv}w_{uv}=0$ ✓.

## Implémentation

```python
import numpy as np

def im2col(x, K, stride, pad):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - K) // stride + 1
    Wo = (W + 2 * pad - K) // stride + 1
    cols = np.zeros((C * K * K, Ho * Wo))
    for i in range(Ho):
        for j in range(Wo):
            cols[:, i * Wo + j] = xp[:, i*stride:i*stride+K, j*stride:j*stride+K].reshape(-1)
    return cols, Ho, Wo

def conv2d_forward(x, W, b, stride=1, pad=0):
    Co, Ci, K, _ = W.shape
    cols, Ho, Wo = im2col(x, K, stride, pad)
    out = W.reshape(Co, -1) @ cols + b[:, None]
    return out.reshape(Co, Ho, Wo), (cols, x.shape, W, stride, pad)

def conv2d_backward(dout, cache):
    cols, x_shape, W, stride, pad = cache
    Co = W.shape[0]
    dY = dout.reshape(Co, -1)
    dW = (dY @ cols.T).reshape(W.shape)
    db = dY.sum(axis=1)
    dcols = W.reshape(Co, -1).T @ dY
    dx = col2im(dcols, x_shape, W.shape[2], stride, pad)   # scatter-add, crop padding
    return dx, dW, db
```

`col2im` est la boucle qui fait `xp[:, rows, cols] += column.reshape(C, K, K)` pour chaque position de sortie et rogne le remplissage à la fin ; tu l'écris dans l'exercice.

### Dimensions

| Grandeur | Dimensions |
|---|---|
| entrée $x$ | $(C_{in},H,W)$ |
| poids $W$, biais $b$ | $(C_{out},C_{in},K,K)$, $(C_{out})$ |
| $X_{col}$ | $(C_{in}K^2,\;H_{out}W_{out})$ |
| sortie $Y$ | $(C_{out},H_{out},W_{out})$ |
| $\bar W$, $\bar b$, $\bar x$ | comme $W$, $b$, $x$ |

Avec un batch, tu ajoutes simplement un axe initial $B$ à $x$ et $Y$.

## Vérification par gradient numérique

Choisis des tenseurs aléatoires $R$ de la forme de $Y$ et utilise le scalaire $L=\sum R\odot Y$, pour lequel $\bar Y=R$. Compare `conv2d_backward(R, cache)` avec des différences centrées, comme dans [[f2n.10]]. Le premier exercice le fait pour quatre combinaisons (pas, remplissage) et attend un accord à $\sim10^{-6}$ près. Le bug classique est d'écrire `=` au lieu de `+=` dans `col2im` : les gradients sont alors trop petits dans les zones de chevauchement et les différences finies l'attrapent immédiatement.

## De LeNet à ResNet en bref

- **LeNet-5 (1998)** : conv $5\times5$ ($1\to6$) $\to$ pool $\to$ conv $5\times5$ ($6\to16$) $\to$ pool $\to$ dense $400\to120\to84\to10$. Les cartes passent par $32\to28\to14\to10\to5$ ; le réseau entier a $61{,}706$ paramètres (vérification : $156+2416+48120+10164+850$). Ce modèle conv-pool-dense est toujours le squelette de tout CNN.
- **AlexNet (2012)** : ReLU au lieu de tanh, dropout, et beaucoup plus de profondeur et de largeur sur GPU.
- **VGG (2014)** : uniquement des convolutions $3\times3$, empilées, avec pooling ; justifié par l'arithmétique du champ récepteur ci-dessus.
- **Inception (2014)** : exécuter plusieurs tailles de noyau en parallèle et concaténer le long de l'axe des canaux ; les convolutions $1\times1$ réduisent les canaux à bas coût.
- **ResNet (2015)** : le bloc résiduel $y=x+F(x)$.

### Pourquoi la connexion résiduelle aide les gradients

Les réseaux simples très profonds obtenaient une perte d'entraînement *pire* que des réseaux moins profonds (le problème de dégradation) : les gradients sont des produits de nombreux facteurs locaux, chacun typiquement de magnitude inférieure à 1, donc ils rétrécissent exponentiellement avec la profondeur. Pour un bloc résiduel $y=x+F(x)$, la jacobienne est

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x},
$$

donc le gradient amont devient $\bar x=\bar y+\bar y\,\partial F/\partial x$ : il contient toujours une **copie directe** de $\bar y$ qui ne passe pas du tout par $F$. À travers de nombreux blocs, le gradient a donc un chemin d'identités pures de la perte jusqu'aux premières couches. Le bloc peut aussi représenter « ne rien faire » en posant $F\approx0$, donc ajouter un bloc n'est jamais forcé de nuire. Le même truc réapparaît dans le chemin additif de l'état de cellule du LSTM ([[seq.04]]) et dans chaque bloc de transformer.

## Idées reçues

- **« Une couche de convolution retourne son noyau. »** En pratique non (corrélation croisée) ; comme le noyau est appris, cela ne change rien.
- **« Le pooling a des poids. »** Le max et l'average pooling n'ont aucun paramètre. La passe arrière du max pooling route simplement le gradient vers la position de l'arg-max.
- **« Plus de pixels signifie plus de paramètres. »** Le nombre de paramètres d'une couche de convolution est indépendant de la taille de l'image ; son *calcul* et sa *mémoire d'activations* croissent avec elle.
- **« Le champ récepteur égale la taille du noyau. »** Il grandit avec la profondeur (et multiplicativement avec les pas) : les unités profondes voient de grandes parties de l'image.
- **« Le remplissage est appris ou est une donnée. »** C'est une constante (des zéros), donc aucun gradient n'est conservé pour les positions remplies.

## Limites et compromis

- Les convolutions supposent la **localité** et l'**équivariance par translation**. C'est un a priori fort : excellent pour les images et l'audio, faux quand des éléments éloignés doivent interagir directement (les dépendances à longue portée demandent de nombreuses couches ou un autre mécanisme comme l'attention).
- Cet a priori signifie aussi que les CNN ne sont pas invariants à la rotation ou à l'échelle par eux-mêmes ; il faut de l'augmentation de données ou des changements d'architecture.
- im2col échange de la mémoire contre de la vitesse : $X_{col}$ est environ $K^2$ fois plus grand que $x$. Les bibliothèques de production utilisent des noyaux de calcul plus malins.
- Le max pooling jette l'information de position. On utilise souvent à la place des convolutions à pas.
- Pour le texte, une convolution 1D de largeur $k$ sur des embeddings de mots détecte des $k$-grammes en parallèle et à bas coût, mais ne peut pas transporter d'information au-delà de sa fenêtre, contrairement à un réseau récurrent.

## Points clés

- Une couche de convolution est une couche dense appliquée à chaque fenêtre avec des poids partagés : $\#\text{params}=C_{out}C_{in}K^2+C_{out}$.
- $H_{out}=\lfloor (H-K+2P)/S\rfloor+1$ ; le champ récepteur croît avec $r\leftarrow r+(K-1)j$, $j\leftarrow jS$.
- Passe arrière : $\bar W=\bar Y_{mat}X_{col}^\top$, $\bar b=\sum\bar Y$, $\bar X_{col}=W_{mat}^\top\bar Y_{mat}$ suivi d'un scatter-**add** (`col2im`).
- Les blocs résiduels donnent aux gradients un chemin d'identité : $\partial y/\partial x=I+\partial F/\partial x$.

## Explique-le toi-même

1. Pourquoi le partage de paramètres réduit-il à la fois le nombre de poids et fait-il du gradient des poids une *somme sur les positions* ?
2. Dérive $\bar x$ pour un noyau $3\times3$ de pas 1 pour un pixel intérieur et explique pourquoi `col2im` doit additionner plutôt qu'affecter.

## Pour aller plus loin

- Bibliothèque source : `ai-engineering-from-scratch` phase 04, leçons 02 (convolutions from scratch) et 03 (LeNet to ResNet).
- Même bibliothèque, phase 05 leçon 08, pour les convolutions 1D sur du texte.
- Ensuite : [[seq.02]] (plongements de mots), la représentation qui alimente à la fois les TextCNN et les réseaux récurrents.

## Critères de maîtrise

Tu sais calculer sur papier la taille de sortie, le nombre de paramètres et le champ récepteur d'un petit CNN, écrire les trois formules de la passe arrière d'une couche de convolution avec leurs dimensions, et ton conv2d NumPy passe une vérification par différences finies pour plusieurs pas et remplissages.
