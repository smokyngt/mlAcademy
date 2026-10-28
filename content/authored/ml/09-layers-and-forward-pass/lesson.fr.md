---
id: f2n.09
title: Neurones, couches et passe avant
summary: "Une couche dense est activation(XW + b). Pourquoi la non-linéarité est obligatoire, les activations courantes, le décompte des paramètres et l'échelle initiale."
minutes: 55
teaches: [nn.layer, nn.activation]
requires: [la.matrix, ml.loss, calc.derivative]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 5.1 Feed-forward network functions, PDF p.247
  - math/math for ML.pdf (Sec. 5.6 Backpropagation and automatic differentiation: the deep-network chain, PDF p.145-176)
  - ai-engineering-from-scratch/phases/03-deep-learning-core/01-the-perceptron/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/02-multi-layer-networks/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/04-activation-functions/docs/en.md
---

## Objectifs d'apprentissage

- Écrire un neurone et une couche dense, avec toutes les formes, pour un seul exemple et pour un batch.
- Prouver qu'empiler des couches linéaires sans non-linéarité n'apporte rien de nouveau.
- Donner les formules et les dérivées de ReLU, sigmoïde et tanh.
- Compter les paramètres et choisir une échelle initiale des poids qui maintient les activations à une taille stable.

## Prérequis

Matrices et formes ([[f2n.03]]), dérivées ([[f2n.04]]), pertes ([[f2n.08]]).

## Intuition

Un **neurone** calcule une somme pondérée de ses entrées (un produit scalaire : « dans quelle mesure cette entrée ressemble-t-elle à mon motif ? »), ajoute un biais (un seuil), et fait passer le résultat dans une **activation** non linéaire. Une **couche** est un ensemble de neurones qui lisent la même entrée, ce qui revient à une multiplication matricielle. Un **réseau** est une composition de couches, $h=f_L\circ\cdots\circ f_1$ (leçon 1).

## Définitions formelles

Un neurone : $a=\phi(w\cdot x+b)$ avec $w,x\in\mathbb{R}^n$, $b\in\mathbb{R}$.

Couche dense pour un batch $X\in\mathbb{R}^{B\times n}$ :

$$
Z=XW+\mathbf{1}b^\top,\qquad A=\phi(Z),\qquad W\in\mathbb{R}^{n\times m},\ b\in\mathbb{R}^m,\ Z,A\in\mathbb{R}^{B\times m}.
$$

Le biais est ajouté à chaque ligne (« broadcasting »). La colonne $k$ de $W$ est le vecteur de poids du neurone $k$.

**Activations et dérivées**

| Nom | $\phi(z)$ | $\phi'(z)$ | Remarques |
|---|---|---|---|
| ReLU | $\max(0,z)$ | $1$ si $z>0$ sinon $0$ | peu coûteuse ; choix par défaut pour les couches cachées ; « morte » si toujours $z\le0$ |
| Sigmoïde | $\dfrac1{1+e^{-z}}$ | $\sigma(1-\sigma)$ | sortie dans $(0,1)$ ; sature, pente maximale $0.25$ |
| Tanh | $\tanh z$ | $1-\tanh^2z$ | centrée en zéro, sortie dans $(-1,1)$, pente maximale $1$ |
| Identité | $z$ | $1$ | utilisée pour les sorties de régression et les logits |

La dérivée de tanh découle de $\tanh z=\frac{e^z-e^{-z}}{e^z+e^{-z}}$ et de la règle du quotient : $\tanh'=1-\tanh^2$.

Explore-les, dérivées comprises :

```viz
activation-plot
```

## Pourquoi la non-linéarité est obligatoire

Deux couches linéaires : $W_2(W_1x)=(W_2W_1)x$. Le produit $W_2W_1$ est une seule matrice, donc n'importe quelle profondeur de couches purement linéaires équivaut à **une seule** couche linéaire. Sans $\phi$, un réseau ne peut tracer que des frontières de décision droites et ne peut même pas calculer le XOR.

**XOR avec deux unités ReLU.** Entrées $x\in\{0,1\}^2$. Posons $h_1=\mathrm{ReLU}(x_1+x_2)$, $h_2=\mathrm{ReLU}(x_1+x_2-1)$, sortie $o=h_1-2h_2$. Alors

| $x_1,x_2$ | $h_1$ | $h_2$ | $o$ |
|---|---|---|---|
| 0,0 | 0 | 0 | 0 |
| 1,0 | 1 | 0 | 1 |
| 0,1 | 1 | 0 | 1 |
| 1,1 | 2 | 1 | 0 |

C'est le XOR, calculé par un réseau de poids $W_1=\begin{pmatrix}1&1\\1&1\end{pmatrix}$, $b_1=(0,-1)$, $W_2=(1,-2)^\top$. Tu le construiras dans l'exercice.

## Compter les paramètres

Une couche $n\to m$ a $nm+m$ paramètres. Un classifieur $784\to128\to10$ a $(784\cdot128+128)+(128\cdot10+10)=100{,}480+1{,}290=101{,}770$ paramètres.

## Dérivation : choisir l'échelle initiale

Prenons $z=\sum_{j=1}^{n}w_jx_j$ avec des poids indépendants de moyenne nulle et de variance $\sigma_w^2$, et des entrées de variance $\sigma_x^2$ (et de moyenne 0). Alors

$$
\operatorname{Var}(z)=n\,\sigma_w^2\,\sigma_x^2 .
$$

Pour empêcher la variance de croître ou de décroître de couche en couche, il faut $n\sigma_w^2=1$, c'est-à-dire $\sigma_w^2=1/n$ (style Xavier/Glorot pour tanh). ReLU annule la moitié des pré-activations, ce qui divise par deux le moment d'ordre deux ; en compensant, on obtient $\sigma_w^2=2/n$ (initialisation de He). Avec une échelle plus grande, les activations explosent avec la profondeur ; avec une plus petite, elles s'évanouissent, et dans les deux cas les gradients héritent du problème.

## Implémentation

```python
import numpy as np

def dense(X, W, b, phi):
    Z = X @ W + b            # (B,n)@(n,m) + (m,) -> (B,m)
    return phi(Z), Z         # keep Z: backprop needs it (lesson 10)

relu = lambda z: np.maximum(0, z)
rng = np.random.default_rng(0)
W = rng.normal(0, np.sqrt(2 / 784), size=(784, 128))   # He init
```

## Idées reçues

- **« Les unités cachées n'ont besoin d'un biais que parfois. »** Un biais permet au neurone de décaler son seuil ; il coûte $m$ paramètres et est presque toujours inclus.
- **ReLU mortes** : si la pré-activation d'une unité est négative pour toutes les entrées, son gradient est exactement 0 et elle ne s'en remet jamais. Les grands taux d'apprentissage et une mauvaise initialisation en sont la cause.
- **Poids initiaux nuls** : tous les neurones d'une couche calculent la même chose et reçoivent le même gradient, donc la symétrie n'est jamais brisée. Initialise toujours au hasard.
- **La sigmoïde pour les couches cachées** est rarement judicieuse : une pente $\le0.25$ fait rétrécir les gradients d'un facteur $4\times$ par couche, au mieux.

## Points clés à retenir

- Une couche dense est `phi(XW + b)` avec `X: (B, n)`, `W: (n, m)`, `b: (m)`.
- Sans non-linéarité, la profondeur s'effondre en une seule matrice.
- ReLU, sigmoïde, tanh et leurs dérivées : connais l'allure de leurs courbes et leurs pentes maximales.
- L'initialisation de He utilise la variance `2/n_in` pour que les activations n'explosent ni ne s'évanouissent.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Explique en deux phrases pourquoi un réseau purement linéaire ne peut pas calculer le XOR, puis donne les poids qui y parviennent avec ReLU.
2. Compte les paramètres d'un réseau 784-128-10 et montre d'où vient chaque terme.

## Pour aller plus loin

- Bishop, *PRML*, Sec. 5.1, PDF p.247.
- Bibliothèque source : `ai-engineering-from-scratch` phase 03, leçons 01, 02, 04 et 08.

## Critères de maîtrise

Tu sais donner chaque forme dans `A = relu(X @ W + b)`, prouver que la profondeur sans non-linéarité est inutile, compter les paramètres d'un MLP, et expliquer pourquoi l'initialisation de He utilise $2/n$.
