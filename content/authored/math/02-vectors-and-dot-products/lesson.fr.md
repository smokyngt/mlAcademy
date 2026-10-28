---
id: f2n.02
title: "Vecteurs, normes et produit scalaire"
summary: "Les vecteurs vus comme des flèches et comme des listes, et le produit scalaire que calcule chaque neurone."
minutes: 45
teaches: [la.vector, la.dot]
requires: [alg.functions, alg.summation]
sources:
  - math/math for ML.pdf (Ch. 2 Linear Algebra, PDF p.23; Ch. 3 Analytic Geometry: norms, inner products, angles, projections, PDF p.76)
  - ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md
  - ai-engineering-from-scratch/phases/01-math-foundations/14-norms-and-distances/docs/en.md
---

## Objectifs d'apprentissage

- Additionner des vecteurs, les multiplier par un scalaire et former des combinaisons linéaires.
- Calculer le produit scalaire de deux façons (somme de produits ; $\|a\|\|b\|\cos\theta$) et démontrer pourquoi elles coïncident.
- Calculer des normes, la similarité cosinus et la projection d'un vecteur sur un autre.
- Reconnaître la pré-activation d'un neurone $w\cdot x + b$ comme un produit scalaire.

## Prérequis

La notation des fonctions et la sommation ([[f2n.01]]).

## Où cela s'inscrit

Un modèle ne peut comparer des choses que si elles sont décrites par des nombres. Une maison est (surface, pièces, âge) ; une chanson est quelques centaines de mesures audio ; un mot est une liste de 768 nombres. Une telle liste est un **vecteur**. Presque toute question posée par un modèle est « ces deux listes se ressemblent-elles ? » (cet e-mail ressemble-t-il aux spams que j'ai vus ? ce mot est-il proche de celui-là ?), et l'outil qui y répond est le **produit scalaire**.

Tu le retrouveras immédiatement : un neurone artificiel n'est rien de plus qu'un produit scalaire suivi d'un petit ajustement. Le comprendre maintenant fera que la leçon 9 (les couches) ressemblera à une répétition, pas à une idée nouvelle.

## L'idée en mots simples

- Un vecteur est une **flèche** sur une carte : de combien vers l'est, de combien vers le nord. Avec plus de deux ou trois directions on ne peut plus le dessiner, mais l'idée (une liste de nombres = une flèche) reste valable.
- La **longueur** d'une flèche, c'est le Pythagore que tu connais déjà, étendu à plus de directions.
- Le **produit scalaire** répond à « dans quelle mesure ces deux flèches pointent-elles dans le même sens ? ». Imagine deux personnes qui marchent : si elles vont dans la même direction, le nombre est grand et positif ; si elles marchent à angle droit, il est nul ; si elles vont dans des directions opposées, il est négatif. Plus les marches sont longues, plus le nombre est grand.
- Une **projection** est une ombre : tiens un bâton (flèche `a`) sous une lampe située juste au-dessus d'une droite (flèche `b`) et regarde l'ombre qu'il projette sur cette droite. L'ombre est la partie de `a` qui va le long de `b` ; ce qui reste est perpendiculaire à `b`.

Essaie la figure ci-dessous avant de lire les formules : déplace les curseurs et observe le produit scalaire changer de signe quand les flèches tournent.

## Intuition

Un vecteur de $\mathbb{R}^n$ est une liste ordonnée de $n$ nombres. Géométriquement, c'est une flèche partant de l'origine ; en termes de données, c'est *un exemple décrit par $n$ caractéristiques* (une maison : surface, pièces, âge) ou *un ensemble de paramètres*.

Le **produit scalaire** mesure *dans quelle mesure deux vecteurs pointent dans le même sens*. Il est grand et positif quand ils sont alignés, nul quand ils sont perpendiculaires, négatif quand ils s'opposent. Un neurone calcule exactement cela entre son vecteur de poids et l'entrée : « dans quelle mesure l'entrée ressemble-t-elle au motif auquel je suis accordé ? »

Essaie ci-dessous. Déplace les curseurs et observe le produit scalaire suivre l'angle.

```viz
vector-playground
```

## Définitions formelles

Pour $a,b\in\mathbb{R}^n$ de composantes $a_i,b_i$ :

- **Addition et multiplication par un scalaire** se font composante par composante : $(a+b)_i=a_i+b_i$, $(\lambda a)_i=\lambda a_i$.
- **Combinaison linéaire :** $\lambda_1 v_1+\cdots+\lambda_k v_k$.
- **Produit scalaire :** $a\cdot b = a^\top b=\sum_{i=1}^{n} a_ib_i \in\mathbb{R}$.
- **Norme (longueur euclidienne) :** $\|a\|=\sqrt{a\cdot a}=\sqrt{\sum_i a_i^2}$.
- **Distance :** $\|a-b\|$.
- **Similarité cosinus :** $\cos\theta = \dfrac{a\cdot b}{\|a\|\,\|b\|}$ (définie quand aucun des deux n'est le vecteur nul).
- **Orthogonaux :** $a\cdot b=0$.

## Démonstration : pourquoi $a\cdot b=\|a\|\|b\|\cos\theta$

Place $a$ et $b$ comme deux côtés d'un triangle ; le troisième côté est $a-b$ et l'angle entre $a$ et $b$ est $\theta$. La loi des cosinus dit que

$$
\|a-b\|^2=\|a\|^2+\|b\|^2-2\|a\|\|b\|\cos\theta. \tag{1}
$$

Développons maintenant le membre de gauche en utilisant seulement la définition du produit scalaire, qui est bilinéaire et symétrique :

$$
\|a-b\|^2=(a-b)\cdot(a-b)=a\cdot a-2\,a\cdot b+b\cdot b=\|a\|^2-2\,a\cdot b+\|b\|^2. \tag{2}
$$

En égalant (1) et (2) et en simplifiant $\|a\|^2+\|b\|^2$, il reste $-2\,a\cdot b=-2\|a\|\|b\|\cos\theta$, c'est-à-dire $a\cdot b=\|a\|\|b\|\cos\theta$. $\blacksquare$

Comme $|\cos\theta|\le1$, on obtient immédiatement l'**inégalité de Cauchy–Schwarz** $|a\cdot b|\le\|a\|\|b\|$, avec égalité exactement quand un vecteur est un multiple de l'autre.

**Projection.** La composante de $a$ le long de $b$ est un multiple $\lambda b$ tel que le reste $a-\lambda b$ soit orthogonal à $b$ :

$$
(a-\lambda b)\cdot b=0\;\Rightarrow\;\lambda=\frac{a\cdot b}{b\cdot b},\qquad \operatorname{proj}_b(a)=\frac{a\cdot b}{b\cdot b}\,b .
$$

## Exemples traités

1. $a=(1,2,3)$, $b=(4,5,6)$. Produit scalaire : $1\cdot4+2\cdot5+3\cdot6=4+10+18=32$. Normes : $\|a\|=\sqrt{14}\approx3.742$, $\|b\|=\sqrt{77}\approx8.775$. Donc $\cos\theta=32/(3.742\times8.775)\approx0.9746$ et $\theta\approx12.9^\circ$.
2. $a=(3,4)$, $b=(4,-3)$ : $a\cdot b=12-12=0$, ils sont donc orthogonaux. Et $\|a\|=5$ (le triangle 3-4-5).
3. Projeter $a=(2,3)$ sur $b=(1,0)$ : $\lambda=(2)/(1)=2$, projection $=(2,0)$, reste $(0,3)\perp b$.

## Implémentation

```python
import numpy as np

a, b = np.array([1., 2., 3.]), np.array([4., 5., 6.])
dot = np.sum(a * b)                      # same as a @ b
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
proj = (a @ b) / (b @ b) * b             # projection of a onto b
```

## Formes

| Expression | Formes | Résultat |
|---|---|---|
| `a @ b` | $(n,)\cdot(n,)$ | scalaire |
| `a * b` | $(n,)\odot(n,)$ | $(n,)$ terme à terme, **pas** le produit scalaire |
| `W @ x + b` (neurone) | poids $(n,)$, entrée $(n,)$, biais scalaire | pré-activation scalaire |

## Idées fausses courantes

- **`*` n'est pas le produit scalaire.** `a * b` conserve $n$ nombres ; `np.sum(a*b)` est le produit scalaire.
- **La similarité cosinus ignore la longueur.** $(1,1)$ et $(100,100)$ ont une similarité de $1$ mais des distances très différentes. Choisis la mesure qui correspond à ta question.
- **Un produit scalaire nul ne signifie pas que l'un des vecteurs est nul**, seulement qu'ils sont perpendiculaires.
- **En grande dimension, des vecteurs aléatoires sont presque orthogonaux**, donc « presque orthogonal » est une faible preuve de dissemblance.

## Points clés à retenir

- Le produit scalaire est une somme de produits et aussi `|a||b|cos(theta)` ; la loi des cosinus prouve que c'est la même chose.
- Orthogonal signifie produit scalaire nul ; la similarité cosinus ne regarde que la direction.
- La projection de `a` sur `b` est `(a.b / b.b) b`, et le reste est perpendiculaire à `b`.
- La pré-activation d'un neurone `w.x + b` est un produit scalaire : elle mesure l'alignement avec les poids.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Sans formules : pourquoi le produit scalaire est-il grand quand deux flèches pointent dans le même sens ?
2. Deux vecteurs ont une similarité cosinus de 1 mais sont éloignés en distance euclidienne. Donne un exemple.

## Pour aller plus loin

- *Mathematics for Machine Learning*, Ch. 3 (normes, produits scalaires, angles, projections orthogonales), PDF p.76.
- Leçons de la bibliothèque source sur l'intuition de l'algèbre linéaire et les normes (`ai-engineering-from-scratch`, phase 01, leçons 01 et 14).

## Critères de maîtrise

Sans calculatrice, tu sais calculer un produit scalaire et une norme pour de petits vecteurs à coordonnées entières, décider si deux vecteurs sont orthogonaux, et expliquer pourquoi la sortie d'un neurone est grande quand l'entrée est alignée avec ses poids.
