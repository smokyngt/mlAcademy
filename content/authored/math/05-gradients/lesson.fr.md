---
id: f2n.05
title: "Dérivées partielles et gradient"
summary: "Les dérivées des fonctions de plusieurs variables, et pourquoi le gradient pointe vers le haut de la pente."
minutes: 55
teaches: [calc.partial, calc.gradient]
requires: [calc.derivative, la.dot]
sources:
  - math/math for ML.pdf (Ch. 5 Vector Calculus, Sec. 5.2 Partial differentiation and gradients, PDF p.145-176)
  - ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md
---

## Objectifs d'apprentissage

- Calculer des dérivées partielles et assembler un gradient.
- Énoncer la forme du gradient de $f:\mathbb{R}^n\to\mathbb{R}$ et pourquoi elle est égale à celle de l'entrée.
- Démontrer que la direction du gradient est la direction de plus forte augmentation locale.
- Établir le gradient de l'erreur quadratique moyenne pour la régression linéaire, avec toutes les formes.
- Vérifier numériquement n'importe quel gradient.

## Prérequis

Dérivées ([[f2n.04]]) et produit scalaire ([[f2n.02]]).

## Où cela s'inscrit

Un vrai modèle n'a pas un seul bouton ; il a des milliers ou des millions de paramètres. L'entraînement a besoin, pour chacun d'eux, de la réponse à « dans quel sens dois-je te tourner ? ». Faire la dérivée à un bouton de la leçon 4 paramètre par paramètre, puis ranger toutes les réponses dans un seul objet, c'est exactement ce qu'est le **gradient**.

Cette leçon contient aussi la première vraie formule du machine learning que tu vas établir toi-même : le gradient de l'erreur d'un modèle linéaire. Si tu sais le faire avec les bonnes formes, la version pour les réseaux de neurones, plus tard, est le même film avec plus d'acteurs.

## L'idée en mots simples

- Imagine que tu te tiens sur le **flanc d'une colline dans un épais brouillard** et que tu veux descendre. Tu ne vois pas la vallée, mais tu sens la pente sous tes pieds dans chaque direction de la boussole : raide vers le nord, montant doucement vers l'est, et ainsi de suite.
- Chacune de ces pentes par direction est une **dérivée partielle** : « si je ne bouge que dans cette direction et que je gèle les autres, à quelle vitesse la hauteur change-t-elle ? ».
- Mets-les toutes dans une liste et tu obtiens le **gradient**. Il se comporte comme une **flèche pointant droit vers le haut de la pente**, et sa longueur te dit à quel point la colline est raide. Pour *descendre*, marche dans le sens opposé.
- Il ne connaît que le sol juste sous tes pieds. Il ne pointe pas vers la vallée, seulement vers là où c'est localement le plus raide. C'est pourquoi on marche à petits pas et on regarde de nouveau.
- Une règle de comptabilité t'évite de nombreux bogues : le gradient a **la même forme que ce que tu modifies**. Dix boutons, une liste de dix pentes. Une matrice de poids, une matrice de pentes.

## Intuition

Une perte dépend de milliers ou de millions de paramètres. Placé en un point du « paysage » de la perte, chaque paramètre est une direction de la boussole. La **dérivée partielle** par rapport à un paramètre est la pente si l'on marche uniquement dans cette direction. Le **gradient** empile toutes ces pentes en un seul vecteur, et ce vecteur pointe droit vers le haut. Pour diminuer la perte, marche dans le sens opposé.

## Définitions formelles

Soit $f:\mathbb{R}^n\to\mathbb{R}$ qui envoie un vecteur $x=(x_1,\dots,x_n)$ sur un nombre.

**Dérivée partielle.** Gèle toutes les coordonnées sauf $x_i$ et dérive :

$$
\frac{\partial f}{\partial x_i}(x)=\lim_{h\to0}\frac{f(x+he_i)-f(x)}{h},
$$

où $e_i$ est le $i$-ième vecteur de la base canonique.

**Gradient.**

$$
\nabla f(x)=\begin{pmatrix}\partial f/\partial x_1\\ \vdots\\ \partial f/\partial x_n\end{pmatrix}\in\mathbb{R}^n .
$$

> **Règle des formes :** le gradient d'une fonction scalaire a exactement la forme de son entrée. Si les paramètres forment une matrice $W$ de taille $d\times k$, alors $\nabla_W L$ est aussi de taille $d\times k$. Tu t'en serviras tout le temps pour attraper des bogues.

**Dérivée directionnelle.** Le taux de variation de $f$ le long d'un vecteur *unitaire* $u$ est $D_uf(x)=\lim_{t\to0}\frac{f(x+tu)-f(x)}{t}$. Pour $f$ dérivable, le développement au premier ordre $f(x+tu)\approx f(x)+t\,\nabla f(x)\cdot u$ donne

$$
D_uf(x)=\nabla f(x)\cdot u .
$$

## Démonstration : pourquoi le gradient est la direction de plus forte montée

D'après la leçon 2, pour un vecteur unitaire $u$ faisant un angle $\theta$ avec $\nabla f$ :

$$
D_uf=\nabla f\cdot u=\|\nabla f\|\,\|u\|\cos\theta=\|\nabla f\|\cos\theta .
$$

C'est maximal quand $\cos\theta=1$, c'est-à-dire $u=\nabla f/\|\nabla f\|$, et le taux maximal est $\|\nabla f\|$. C'est le plus négatif pour $u=-\nabla f/\|\nabla f\|$, et nul pour les directions perpendiculaires à $\nabla f$ (le long des courbes de niveau). Les mots *local* et *premier ordre* comptent : pour un grand pas, la surface s'incurve et l'énoncé peut tomber en défaut. $\blacksquare$

## Exemple traité 1

$f(x,y)=x^2+3y^2$. Traite $y$ comme une constante pour obtenir $\partial f/\partial x=2x$ ; traite $x$ comme une constante pour obtenir $\partial f/\partial y=6y$. Donc $\nabla f=(2x,6y)$. En $(1,1)$ : $\nabla f=(2,6)$, avec $\|\nabla f\|=\sqrt{40}\approx6.32$. La pente le long de $u=(1,0)$ est $2$ ; le long de la direction la plus raide $(2,6)/6.32$ elle est $6.32$ ; le long de la perpendiculaire $(3,-1)/\sqrt{10}$ elle est $0$ (vérification : $2\cdot3-6\cdot1=0$).

## Exemple traité 2 : gradient de l'erreur quadratique moyenne

Matrice de données $X\in\mathbb{R}^{N\times d}$ (les lignes sont les exemples), cibles $y\in\mathbb{R}^{N}$, poids $w\in\mathbb{R}^{d}$. Prédictions $\hat y=Xw\in\mathbb{R}^N$ et résidus $r=Xw-y\in\mathbb{R}^N$.

$$
L(w)=\frac1N\sum_{i=1}^{N}\big(x_i\cdot w-y_i\big)^2=\frac1N\|r\|^2 .
$$

Dérivons par rapport à un poids $w_j$. Seul $x_i\cdot w=\sum_k x_{ik}w_k$ dépend de $w_j$, avec $\partial(x_i\cdot w)/\partial w_j=x_{ij}$. En utilisant $(u^2)'=2u$ :

$$
\frac{\partial L}{\partial w_j}=\frac1N\sum_{i=1}^N 2\,r_i\,x_{ij}=\frac2N\sum_i x_{ij}r_i=\frac2N\,(X^\top r)_j .
$$

En empilant tous les $j$ :

$$
\boxed{\nabla_wL=\frac2N X^\top(Xw-y)}\qquad\text{shapes: }(d\times N)(N)=(d).
$$

La forme de $\nabla_wL$ est $d$, la même que $w$, comme promis.

## Implémentation

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float); e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

mse_grad = lambda X, y, w: 2 / len(y) * X.T @ (X @ w - y)
```

## Idées fausses courantes

- **Le gradient n'est pas la dérivée d'un *vecteur* ; c'est un vecteur de dérivées d'un scalaire.** Pour les fonctions à valeurs vectorielles, il faut la jacobienne (leçon suivante).
- **Un gradient nul ne signifie pas un minimum.** Ce peut être un maximum ou un point selle ; en grande dimension, les points selles sont le cas courant.
- **La plus forte montée est locale.** Elle ne pointe pas vers le minimum global ; c'est pourquoi l'optimisation est itérative.
- **Confusion sur la forme du gradient** : écrire $\nabla_wL$ comme $Xr^\top$ ou oublier la transposée est le bogue classique. Vérifie les formes par rapport à $w$.

## Points clés à retenir

- Le gradient empile les dérivées partielles ; il a exactement la forme de l'entrée.
- `D_u f = grad f . u`, donc la direction de plus forte montée est `grad f` (Cauchy-Schwarz de la leçon 2).
- Pour la MSE : `grad_w = (2/N) X^T (Xw - y)`. Vérifie sa forme par rapport à `w`.
- Un gradient nul indique seulement un point stationnaire, pas nécessairement un minimum.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Démontre avec tes propres mots que le gradient est la direction de plus forte montée.
2. Établis le gradient de la MSE composante par composante `dL/dw_j`, puis assemble-le en `X^T r`.

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 5.2-5.3, PDF p.145-176.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01, leçon 04.

## Critères de maîtrise

Tu sais calculer à la main le gradient d'un polynôme à deux variables, établir $\frac{2}{N}X^\top(Xw-y)$ sans notes, et écrire un vérificateur de gradient numérique.
