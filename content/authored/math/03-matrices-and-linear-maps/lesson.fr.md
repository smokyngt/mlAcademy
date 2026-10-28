---
id: f2n.03
title: "Matrices, produit matriciel et applications linéaires"
summary: "Les formes, la règle ligne par colonne, et l'image d'une matrice comme transformation de l'espace."
minutes: 55
teaches: [la.matrix, la.linear-map]
requires: [la.dot]
sources:
  - math/math for ML.pdf (Ch. 2 Linear Algebra, Sec. 2.2 Matrices and 2.7 Linear Mappings, PDF p.23-70)
  - ai-engineering-from-scratch/phases/01-math-foundations/02-vectors-matrices-operations/docs/en.md
  - ai-engineering-from-scratch/phases/01-math-foundations/03-matrix-transformations/docs/en.md
---

## Objectifs d'apprentissage

- Prédire la forme de la sortie de n'importe quel produit matriciel et repérer les erreurs de forme par simple inspection.
- Calculer à la main un produit matrice-vecteur et un produit matrice-matrice, selon la vue par *lignes* et la vue par *colonnes*.
- Expliquer pourquoi $AB\neq BA$ en général et pourquoi $(AB)^\top=B^\top A^\top$.
- Lire une matrice $2\times2$ comme une transformation : ses colonnes sont les endroits où atterrissent les vecteurs de base.

## Prérequis

Produit scalaire ([[f2n.02]]).

## Où cela s'inscrit

Une couche d'un réseau de neurones prend une liste de nombres (disons les 784 pixels d'une image) et produit une nouvelle liste (disons 128 nombres qui décrivent mieux l'image). En pratique, on le fait pour 32 images à la fois. L'objet qui le fait est une **matrice**, et savoir lire sa forme est la compétence la plus utile pour éviter les bogues en machine learning : la plupart des vraies erreurs sont du type « ces deux formes ne s'emboîtent pas ».

La leçon a deux objectifs. Premièrement, rendre la multiplication matricielle mécanique (pour que les erreurs de forme sautent aux yeux). Deuxièmement, te donner une image, la matrice comme manière de *déformer l'espace*, pour que l'algèbre cesse de paraître arbitraire.

## L'idée en mots simples

- Une matrice est un **tableau de recettes**. Chaque ligne dit comment mélanger les entrées pour fabriquer une sortie : « prends 2 parts du premier nombre, 1 part du second, et additionne-les ». Multiplier une matrice par une liste revient à suivre chaque recette une fois.
- Géométriquement, imagine une feuille de **papier quadrillé** que tu peux étirer, tourner ou cisailler, mais jamais déchirer ni plier : les droites restent droites, et l'origine reste en place. Une telle déformation est ce que fait une matrice au plan.
- Pour savoir exactement ce que fait une matrice donnée, il suffit de savoir **où vont deux flèches** : la flèche pointant à droite (d'une unité) et celle pointant vers le haut. Leurs points d'arrivée sont les deux colonnes de la matrice, et tous les autres points suivent.
- La **règle des formes** est une simple règle de prise et de fiche : une matrice `m x n` a `n` prises du côté de l'entrée ; elle s'adapte à quelque chose qui offre `n` fiches, et te rend `m` sorties.

Joue d'abord avec la figure : change un coefficient et observe toute la grille bouger.

## Intuition

Une matrice est un *tableau de nombres* et aussi une *fonction sur les vecteurs*. En tant que fonction, $x\mapsto Ax$ envoie un vecteur de $\mathbb{R}^n$ vers un vecteur de $\mathbb{R}^m$ et le fait **linéairement** : elle peut étirer, tourner, cisailler, réfléchir et projeter l'espace, mais elle garde toujours l'origine fixe et les droites droites. Chaque couche dense d'un réseau de neurones est l'une de ces applications suivie d'une non-linéarité.

```viz
matrix-transform
```

## Définitions formelles

$A\in\mathbb{R}^{m\times n}$ a $m$ lignes et $n$ colonnes ; le coefficient $a_{ij}$ est à la ligne $i$, colonne $j$.

**Produit matrice-vecteur.** Pour $x\in\mathbb{R}^n$, $Ax\in\mathbb{R}^m$ avec

$$
(Ax)_i=\sum_{j=1}^{n} a_{ij}x_j = (\text{row } i \text{ of } A)\cdot x .
$$

**Produit matrice-matrice.** Pour $A\in\mathbb{R}^{m\times n}$, $B\in\mathbb{R}^{n\times p}$, $AB\in\mathbb{R}^{m\times p}$ avec

$$
(AB)_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
$$

**Règle des formes :** $(m\times n)(n\times p)=(m\times p)$. Les deux tailles intérieures doivent coïncider, puis elles disparaissent.

**Transposée.** $(A^\top)_{ij}=a_{ji}$, de forme $n\times m$. L'**identité** $I$ a des uns sur la diagonale ; $AI=IA=A$.

## Deux façons de voir le même produit

*Vue par lignes :* chaque coefficient de sortie est un produit scalaire (leçon 2). *Vue par colonnes :* en écrivant $A$ par ses colonnes $a^{(1)},\dots,a^{(n)}$,

$$
Ax=x_1a^{(1)}+x_2a^{(2)}+\cdots+x_na^{(n)}.
$$

Ainsi $Ax$ est une **combinaison linéaire des colonnes de $A$**, dont les coefficients sont les composantes de $x$. En appliquant cela au vecteur de base canonique $e_j$ (un 1 en position $j$, des zéros ailleurs), on obtient $Ae_j=a^{(j)}$ : **la $j$-ième colonne de $A$ est l'image de $e_j$**. Ce seul fait te permet de dessiner n'importe quelle transformation $2\times2$.

## Démonstration : $(AB)^\top=B^\top A^\top$

Le coefficient $(i,j)$ de $(AB)^\top$ est le coefficient $(j,i)$ de $AB$, c'est-à-dire $\sum_k a_{jk}b_{ki}$. Le coefficient $(i,j)$ de $B^\top A^\top$ est $\sum_k (B^\top)_{ik}(A^\top)_{kj}=\sum_k b_{ki}a_{jk}$. Ce sont les mêmes sommes. $\blacksquare$

## Pourquoi $AB\neq BA$

Soit $A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$ et $S=\begin{pmatrix}0&1\\1&0\end{pmatrix}$.

$$
AS=\begin{pmatrix}1\cdot0+2\cdot1&1\cdot1+2\cdot0\\3\cdot0+4\cdot1&3\cdot1+4\cdot0\end{pmatrix}=\begin{pmatrix}2&1\\4&3\end{pmatrix}\quad(\text{columns swapped}),
$$

$$
SA=\begin{pmatrix}0\cdot1+1\cdot3&0\cdot2+1\cdot4\\1\cdot1+0\cdot3&1\cdot2+0\cdot4\end{pmatrix}=\begin{pmatrix}3&4\\1&2\end{pmatrix}\quad(\text{rows swapped}).
$$

La multiplication à droite agit sur les colonnes ; la multiplication à gauche agit sur les lignes. Opérations différentes, résultats différents. Remarque aussi que si $m\ne p$, l'un de $AB$, $BA$ peut même ne pas être défini.

## Exemple traité : une rotation

Tourner le plan de $\theta$ dans le sens trigonométrique envoie $e_1=(1,0)\mapsto(\cos\theta,\sin\theta)$ et $e_2=(0,1)\mapsto(-\sin\theta,\cos\theta)$. D'après la règle des colonnes,

$$
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}.
$$

Pour $\theta=90^\circ$ : $R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$, donc $R(1,1)^\top=(-1,1)^\top$. La rotation conserve les longueurs : $\|Rx\|=\|x\|$.

Un exemple matrice-vecteur : $\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}1+2\\3+4\end{pmatrix}=\begin{pmatrix}3\\7\end{pmatrix}=1\cdot\begin{pmatrix}1\\3\end{pmatrix}+1\cdot\begin{pmatrix}2\\4\end{pmatrix}$.

## Implémentation

```python
import numpy as np

def matmul(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"inner dimensions differ: {n} vs {n2}"
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            C[i, j] = np.sum(A[i, :] * B[:, j])   # row i . column j
    return C
```

## Formes des tenseurs dans un réseau de neurones

Pour un lot de $B$ exemples ayant chacun $n$ caractéristiques, empilés comme lignes de $X\in\mathbb{R}^{B\times n}$, et une matrice de poids $W\in\mathbb{R}^{n\times m}$ :

$$
XW\in\mathbb{R}^{B\times m}.
$$

La ligne $b$ du résultat est $x_b^\top W$, la couche appliquée à l'exemple $b$. On écrit les couches sous la forme $XW$ (et non $Wx$) parce que les frameworks stockent un lot sous forme de lignes. Les deux conventions sont les mêmes mathématiques à une transposée près : $(XW)^\top=W^\top X^\top$.

## Idées fausses courantes

- **La multiplication matricielle n'est pas terme à terme.** La multiplication terme à terme (`A * B`) exige des formes égales et est une opération différente.
- **L'ordre compte.** $ABx$ signifie qu'on applique d'abord $B$, puis $A$.
- **$AB=0$ n'implique pas $A=0$ ni $B=0$.**
- **« Linéaire » ne veut pas dire « une droite ».** Cela signifie $A(\alpha x+\beta y)=\alpha Ax+\beta Ay$. Ajouter un biais $b$ rend l'application *affine*, pas linéaire.

## Points clés à retenir

- `(m x n)(n x p) = (m x p)` : les tailles intérieures doivent coïncider puis disparaissent.
- Deux vues de `Ax` : les lignes donnent des produits scalaires ; les colonnes donnent une combinaison linéaire de colonnes.
- Les colonnes de `A` sont les endroits où atterrissent les vecteurs de base, donc tu peux dessiner n'importe quelle application 2x2 à partir de ses colonnes.
- `AB != BA`, et `(AB)^T = B^T A^T`. Un biais rend une application affine, pas linéaire.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Dessine où va le carré unité sous `[[2,0],[0,3]]` et sous une rotation de 90 degrés. En quoi les colonnes t'ont-elles aidé ?
2. Pourquoi `X @ W` avec `X: (B, n)` applique-t-il la couche à tous les exemples d'un coup ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 2.2 (matrices) et 2.7 (applications linéaires), PDF p.23-70.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01, leçons 02 et 03.

## Critères de maîtrise

Face à trois matrices dont les formes sont données, tu sais dire instantanément quels produits existent et quelle forme ils ont, et tu sais esquisser où va le carré unité sous une matrice $2\times2$ à partir de ses colonnes.
