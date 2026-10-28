---
id: f2n.06
title: "Règle de la chaîne, jacobiennes et produits vecteur-jacobienne"
summary: "Comment les dérivées circulent à travers les compositions, avec toutes les formes, et les trois formules arrière qui font fonctionner la rétropropagation."
minutes: 65
teaches: [calc.chain, calc.jacobian]
requires: [calc.derivative, calc.gradient, la.matrix]
sources:
  - math/math for ML.pdf (Ch. 5 Vector Calculus, Sec. 5.3 Gradients of vector-valued functions and 5.6 Backpropagation and automatic differentiation, PDF p.145-176)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 5.3 Error Backpropagation, PDF p.261
  - ai-engineering-from-scratch/phases/01-math-foundations/05-chain-rule-and-autodiff/docs/en.md
---

## Objectifs d'apprentissage

- Appliquer la règle de la chaîne scalaire à des fonctions imbriquées.
- Définir la jacobienne d'une fonction à valeurs vectorielles et énoncer sa forme.
- Écrire la règle de la chaîne multivariée comme un produit matriciel et vérifier les formes.
- Établir, avec les formes, les formules arrière d'une application linéaire $y=Wx$ et d'une activation terme à terme.
- Expliquer pourquoi on multiplie le gradient amont par $J^\top$ au lieu de former des jacobiennes.

## Prérequis

Dérivées ([[f2n.04]]), gradients ([[f2n.05]]), produits matriciels ([[f2n.03]]).

## Où cela s'inscrit

Un réseau de neurones est une longue chaîne de petites étapes : l'entrée est transformée par la première couche, le résultat par la deuxième, et ainsi de suite jusqu'au calcul d'une erreur finale. Un bouton de la première couche n'affecte l'erreur qu'*à travers* tout ce qui suit. Pour entraîner le réseau, il faut savoir combien chaque bouton, tôt ou tard dans la chaîne, contribue à l'erreur finale.

La **règle de la chaîne** est la règle qui dit comment les effets se transmettent le long d'une chaîne. Sa version matricielle (les jacobiennes) permet de l'appliquer à des couches comportant beaucoup de nombres. Cette leçon est le pont entre la partie Mathématiques et la partie Machine Learning : les trois petites formules de la fin sont, littéralement, toute la rétropropagation.

## L'idée en mots simples

- **Taux de change.** Si 1 euro achète 1,1 dollar et 1 dollar achète 150 yens, alors 1 euro achète 1,1 x 150 = 165 yens. Les taux le long d'une chaîne se **multiplient**. C'est la règle de la chaîne : la sensibilité de la dernière étape à la première est le produit des sensibilités de chaque maillon.
- Si une entrée peut atteindre la sortie par **deux chemins différents**, les effets des deux chemins **s'additionnent**.
- Avec un seul nombre par étape, un taux de change par maillon suffit. Avec des *listes* de nombres, il faut tout un **tableau de taux de change** (de combien chaque sortie réagit à chaque entrée). Ce tableau est la **jacobienne**, et multiplier des tableaux, c'est la multiplication matricielle.
- **Remonter est moins coûteux.** Supposons que l'erreur finale soit un seul nombre. Partir de là et demander « quelle part de responsabilité pour chaque étape précédente ? » permet de réutiliser les réponses au fil de la remontée, donc un seul balayage arrière donne la responsabilité de chaque bouton. C'est l'« attribution de la responsabilité » (credit assignment, ou plutôt blame assignment), et c'est ce que fait la rétropropagation.

## Intuition

Si $y$ dépend de $u$ et $u$ dépend de $x$, alors pousser $x$ de $\delta$ pousse $u$ de $u'\delta$, ce qui pousse $y$ de $y'(u)\,u'\delta$. **Les taux se multiplient le long du chemin.** Un réseau de neurones est une longue chaîne de telles dépendances, donc son gradient est un produit de taux locaux, chacun calculé une seule fois, de la sortie vers l'arrière.

## Règle de la chaîne scalaire

Pour $y=g(f(x))$,

$$
\frac{dy}{dx}=g'\big(f(x)\big)\cdot f'(x).
$$

*Exemple.* $y=\sigma(wx+b)$ comme fonction de $w$ : posons $z=wx+b$, donc $dz/dw=x$ et $dy/dz=\sigma'(z)=\sigma(z)(1-\sigma(z))$. Alors $dy/dw=\sigma'(z)\,x$.

*Plusieurs chemins.* Si $y=g(u_1(x),u_2(x))$ alors $\dfrac{dy}{dx}=\dfrac{\partial g}{\partial u_1}\dfrac{du_1}{dx}+\dfrac{\partial g}{\partial u_2}\dfrac{du_2}{dx}$ : somme sur chaque chemin de $x$ à $y$. Cette somme explique pourquoi un poids qui influence plusieurs unités suivantes reçoit plusieurs contributions.

## La jacobienne

Pour $f:\mathbb{R}^n\to\mathbb{R}^m$ de sorties $f_1,\dots,f_m$, la **jacobienne** rassemble toutes les dérivées partielles premières :

$$
J_f(x)=\begin{pmatrix}\partial f_1/\partial x_1&\cdots&\partial f_1/\partial x_n\\ \vdots&&\vdots\\ \partial f_m/\partial x_1&\cdots&\partial f_m/\partial x_n\end{pmatrix}\in\mathbb{R}^{m\times n},\qquad (J_f)_{ij}=\frac{\partial f_i}{\partial x_j}.
$$

La ligne $i$ est le gradient (transposé) de la sortie $f_i$. L'image au premier ordre : $f(x+\delta)\approx f(x)+J_f(x)\,\delta$. La jacobienne est la meilleure *application linéaire* approchant $f$ près de $x$.

**Règle de la chaîne multivariée.** Pour $f:\mathbb{R}^n\to\mathbb{R}^m$ et $g:\mathbb{R}^m\to\mathbb{R}^k$, $h=g\circ f$ :

$$
J_h(x)=J_g\big(f(x)\big)\,J_f(x),\qquad (k\times n)=(k\times m)(m\times n).
$$

Cela vaut parce que linéariser chaque application et composer les applications linéaires multiplie leurs matrices ; terme à terme, c'est la règle de la somme sur les chemins : $\frac{\partial h_i}{\partial x_j}=\sum_{l=1}^m\frac{\partial g_i}{\partial f_l}\frac{\partial f_l}{\partial x_j}$.

## Des jacobiennes à la rétropropagation : le produit vecteur-jacobienne

Pendant l'entraînement, le bout de la chaîne est une perte **scalaire** $L$, donc $k=1$ et $J_L$ est une ligne. Notons le *gradient amont* $g=\nabla_yL\in\mathbb{R}^m$ pour la sortie $y=f(x)$ d'une étape. Alors

$$
\nabla_xL=J_f(x)^\top\,g\in\mathbb{R}^n .
$$

Calculer $J_f^\top g$ (un « produit vecteur-jacobienne ») n'exige jamais de construire une énorme jacobienne, et pour les couches courantes il a une forme close. Trois formules portent toute la rétropropagation :

### 1. Application linéaire $y=Wx$ avec $W\in\mathbb{R}^{m\times n}$

$y_i=\sum_jW_{ij}x_j$, donc $\partial y_i/\partial x_j=W_{ij}$ et $J=W$. D'où

$$
\nabla_xL=W^\top g\ \ (n),\qquad\text{and since }\frac{\partial L}{\partial W_{ij}}=\sum_{l}\frac{\partial L}{\partial y_l}\frac{\partial y_l}{\partial W_{ij}}=g_i\,x_j:\qquad\nabla_WL=g\,x^\top\ \ (m\times n).
$$

### 2. Ajout d'un biais $y=z+b$

$J=I$, donc $\nabla_zL=g$ et $\nabla_bL=g$.

### 3. Activation terme à terme $y=\phi(z)$

$y_i=\phi(z_i)$ ne dépend que de $z_i$, donc $J=\operatorname{diag}(\phi'(z_1),\dots,\phi'(z_m))$ et $\operatorname{diag}(a)\,g=a\odot g$ :

$$
\nabla_zL=g\odot\phi'(z).
$$

(On ne construit jamais la matrice diagonale $m\times m$.)

## Exemple traité avec des nombres

$W=\begin{pmatrix}1&2\\3&4\end{pmatrix}$, $x=\begin{pmatrix}1\\1\end{pmatrix}$, donc $y=Wx=(3,7)^\top$. Supposons que la perte donne le gradient amont $g=(1,-1)^\top$.

$$
\nabla_xL=W^\top g=\begin{pmatrix}1&3\\2&4\end{pmatrix}\begin{pmatrix}1\\-1\end{pmatrix}=\begin{pmatrix}-2\\-2\end{pmatrix},\qquad\nabla_WL=gx^\top=\begin{pmatrix}1\\-1\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix}=\begin{pmatrix}1&1\\-1&-1\end{pmatrix}.
$$

Vérification avec $L=g\cdot y=y_1-y_2$ : alors $L=(W_{11}x_1+W_{12}x_2)-(W_{21}x_1+W_{22}x_2)$ ; $\partial L/\partial x_1=W_{11}-W_{21}=1-3=-2$ ✓ et $\partial L/\partial W_{22}=-x_2=-1$ ✓.

## Implémentation

```python
import numpy as np

def linear_backward(W, x, g):
    dx = W.T @ g              # (n,)   = (n,m)(m,)
    dW = np.outer(g, x)       # (m,n)  = (m,1)(1,n)
    return dx, dW

def relu_backward(z, g):
    return g * (z > 0)        # elementwise: g * phi'(z)
```

## Idées fausses courantes

- **« La rétropropagation est un algorithme spécial. »** C'est la règle de la chaîne, appliquée dans l'ordre inverse pour que chaque gradient intermédiaire soit réutilisé. Une évaluation dans l'ordre direct coûterait une passe par entrée ; l'ordre inverse coûte une passe par sortie (ici, une).
- **Les transposées ne sont pas décoratives.** L'application arrière de $W$ est $W^\top$ parce que les lignes et les colonnes échangent leurs rôles quand on passe de « effet de l'entrée sur la sortie » à « responsabilité de l'entrée vue depuis la sortie ».
- **Les formes sont la preuve.** Si ton $\nabla_WL$ n'est pas $m\times n$, la formule est fausse.
- **La dérivée de ReLU en 0** n'est pas définie ; les implémentations utilisent $0$ (ou $1$). Cela compte rarement en pratique.

## Points clés à retenir

- Les taux se multiplient le long d'un chemin et s'additionnent entre chemins.
- La jacobienne de `f: R^n -> R^m` est `m x n` ; la règle de la chaîne est le produit `J_g J_f`.
- Formules arrière : linéaire `dx = W^T g`, `dW = g x^T` ; biais `db = g` ; terme à terme `dz = g * phi'(z)`.
- On multiplie le gradient amont par `J^T` au lieu de construire `J` : c'est un produit vecteur-jacobienne.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Pourquoi l'application arrière de `W` est-elle égale à `W^T` ? Explique avec un exemple 2x2.
2. Quelles sont les formes de `dW`, `dx` et `db` pour `y = Wx + b` avec `W: (4, 3)` ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Ch. 5, PDF p.145-176 (jacobiennes, rétropropagation, différentiation automatique).
- Bishop, *PRML*, Sec. 5.3, PDF p.261.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01, leçon 05.

## Critères de maîtrise

Tu sais établir $W^\top g$ et $g\,x^\top$ à partir de zéro, énoncer chaque forme, et les vérifier par un test aux différences finies.
