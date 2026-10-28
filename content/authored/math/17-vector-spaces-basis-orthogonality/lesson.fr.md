---
id: math.17
title: "Espaces vectoriels, bases, orthogonalité et QR"
summary: Ce que sont vraiment un sous-espace, une base et une dimension, comment changer de coordonnées, comment projeter sur un sous-espace, et comment Gram-Schmidt et la décomposition QR en font une méthode stable pour les moindres carrés.
minutes: 90
teaches: [la.vector-space, la.basis, la.projection, la.qr]
requires: [la.matrix, la.linear-map, la.dot, la.linear-systems]
sources:
  - "math/math for ML.pdf, Sec. 2.4 Vector Spaces (PDF p.41-42): groups and the vector-space definition; subspaces. READ (p.41-42 in detail)."
  - "math/math for ML.pdf, Sec. 2.6 Basis and Rank (PDF p.50-53): generating set, span, basis as minimal generating set and maximal independent set, dimension, finding a basis by pivot columns (Example 2.17), rank. READ."
  - "math/math for ML.pdf, Sec. 2.7 Linear Mappings, basis change (PDF p.62-63): composition of basis changes, Def. 2.21 equivalence and Def. 2.22 similarity. READ."
  - "math/math for ML.pdf, Sec. 3.4-3.6 (PDF p.82-86): orthogonality, orthogonal matrices and length/angle preservation (3.31-3.32), orthonormal basis, Gram-Schmidt mention, orthogonal complement and unique decomposition. READ."
  - "math/math for ML.pdf, Sec. 3.8 Orthogonal Projections (PDF p.87-94): projection definition P^2 = P, projection onto a line and onto a subspace, normal equation, P = B (B^T B)^-1 B^T, Example 3.11. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/17-linear-systems/docs/en.md (QR section, lines ~160-200): the route R x = Q^T b and the Gram-Schmidt steps. READ."
---

## Où cela s'inscrit

La leçon [[math.11]] t'a appris à résoudre $Ax=b$ et a introduit le rang. Deux questions restaient ouvertes. Première : quel est exactement l'*espace* dans lequel vivent les colonnes de $A$, et comment mesurer sa taille ? (Espaces vectoriels, bases, dimension.) Seconde : quand $Ax=b$ n'a pas de solution exacte, quel $x$ est le « plus proche », et comment le calculer sans perdre de chiffres ? (Projection orthogonale, Gram-Schmidt, QR.)

Les réponses servent partout. La régression linéaire ([[cml.01]]) est une projection orthogonale de la cible sur l'espace engendré par les colonnes de caractéristiques. L'ACP ([[math.07]]) est la recherche d'une bonne base orthonormée. Le blanchiment, les dimensions d'embedding, l'initialisation orthogonale des poids et la manière numériquement sûre d'ajuster un modèle sont toutes des conséquences de cette leçon. Elle s'appuie sur les produits scalaires et les normes ([[f2n.02]]) et sur les matrices vues comme applications linéaires ([[f2n.03]]).

## L'idée en termes simples

- Un **espace vectoriel** est une collection d'objets que l'on peut additionner et étirer par des nombres, sans jamais sortir de la collection. Les flèches du plan conviennent. Les listes de 768 nombres aussi, et les polynômes aussi. Seul compte que les deux opérations se comportent raisonnablement.
- Un **sous-espace** est un espace vectoriel vivant à l'intérieur d'un plus grand : une feuille plate passant par l'origine. Une droite par l'origine en 3D en est un ; un plan par l'origine en est un. Un plan qui rate l'origine n'en est *pas* un : on ne peut même pas réduire un de ses points à zéro en restant dessus.
- Le **sous-espace engendré** (span) est « tout ce que tu peux construire à partir de ces flèches en les étirant et en les additionnant ». Des flèches **indépendantes** sont telles qu'aucune n'est inutile : aucune ne peut être construite à partir des autres. Une **base** est une équipe sans gaspillage qui atteint pourtant tout. La taille de l'équipe est la **dimension**, le nombre de directions réellement indépendantes.
- Les **coordonnées** sont des indications pour atteindre un point avec une équipe donnée : « deux pas le long de la première flèche, trois le long de la seconde ». Change d'équipe et le *même point* a d'autres coordonnées. Un changement de base n'est qu'une traduction entre deux telles langues.
- **Orthogonal** veut dire à angle droit. Une équipe *orthonormée* (angles droits, longueur unité) est le rêve : les coordonnées viennent de simples produits scalaires, sans équation à résoudre.
- La **projection** est l'ombre d'un point sur une feuille, projetée par une lumière perpendiculaire à la feuille. L'ombre est le point de la feuille le plus proche de l'original, et la droite qui va du point à son ombre est perpendiculaire à la feuille.
- **Gram-Schmidt** fabrique une équipe orthonormée à partir de n'importe quelle équipe : prends les flèches une à une, soustrais la part qui pointe dans la direction de ce que tu as déjà, et ce qui reste est nouveau. Garder la trace de ce que tu as soustrait donne la **décomposition QR**.

## Espaces vectoriels et sous-espaces

Un **espace vectoriel réel** est un ensemble $V$ muni d'une addition $u+v$ et d'une multiplication par un scalaire $c\,v$ ($c\in\mathbb{R}$) obéissant aux règles suivantes, en mots : additionner ou multiplier ne fait jamais sortir de $V$ (fermeture) ; l'addition ne dépend ni de l'ordre ni du regroupement ; il existe un vecteur nul $\mathbf 0$ ; chaque $v$ a un opposé $-v$ ; et la multiplication par un scalaire se comporte comme la multiplication ordinaire et se distribue sur l'addition ($1\,v=v$, $(ab)v=a(bv)$, $(a+b)v=av+bv$, $c(u+v)=cu+cv$). Le livre (Sec. 2.4) construit cela à partir de la notion de groupe, et appelle le premier ensemble de règles « $(V,+)$ est un groupe abélien ». Tu vérifieras rarement les axiomes pour $\mathbb{R}^n$, mais tu les vérifieras constamment pour des *sous-ensembles*.

Un **sous-espace** $U\subseteq V$ est un sous-ensemble qui est lui-même un espace vectoriel. En pratique, trois vérifications suffisent :

1. $\mathbf 0\in U$ ;
2. fermé par addition : $u,w\in U\Rightarrow u+w\in U$ ;
3. fermé par multiplication scalaire : $u\in U,\ c\in\mathbb{R}\Rightarrow cu\in U$.

| Sous-ensemble de $\mathbb{R}^3$ | Sous-espace ? | Pourquoi |
|---|---|---|
| $\{(x,y,0)\}$ (le sol) | oui | contient $\mathbf 0$ ; sommes et multiples restent sur le sol |
| $\{t(1,2,2)\}$ (une droite par $\mathbf 0$) | oui | multiples d'un seul vecteur |
| ensemble des solutions de $Ax=\mathbf 0$ | oui | $A(u+w)=\mathbf 0$ et $A(cu)=\mathbf 0$ (le noyau) |
| ensemble des solutions de $Ax=b$, $b\neq\mathbf 0$ | non | $\mathbf 0$ n'est pas une solution |
| $\{(x,y,1)\}$ (un plan au-dessus du sol) | non | pas de vecteur nul ; la somme de deux points a une troisième coordonnée égale à $2$ |
| $\{(x,y,z): x\ge 0\}$ (un demi-espace) | non | multiplier par $-1$ en fait sortir |
| sphère unité $\|v\|=1$ | non | pas de zéro ; $2v$ n'y est pas |

Les espaces vectoriels ne sont pas que des listes de nombres. Les polynômes de degré au plus 2 forment un espace de dimension 3 (base $1,x,x^2$) ; les matrices $n\times n$ forment un espace de dimension $n^2$. Cette généralité explique pourquoi les espaces de caractéristiques, de fonctions et d'embeddings peuvent tous utiliser la même machinerie.

### Sous-espace engendré, indépendance et espace des colonnes

Le **sous-espace engendré** (span) par des vecteurs $v_1,\dots,v_k$ est l'ensemble de toutes les combinaisons linéaires $c_1v_1+\dots+c_kv_k$. C'est toujours un sous-espace. Pour une matrice $A=[a_1,\dots,a_n]$ de colonnes $a_j$, le sous-espace engendré par les colonnes est l'**espace des colonnes** $\operatorname{col}(A)=\{Ax\}$. Cela donne la lecture géométrique de $Ax=b$ : il est résoluble *exactement quand $b$ appartient à l'espace des colonnes*. Les moindres carrés sont ce que l'on fait quand ce n'est pas le cas.

Des vecteurs sont **linéairement indépendants** si la seule façon de les combiner en $\mathbf 0$ est de prendre tous les coefficients nuls ; de manière équivalente, aucun n'est combinaison des autres ; de manière équivalente, $Ax=\mathbf 0$ n'a que la solution $x=\mathbf 0$. Test pratique (livre Sec. 2.5, via l'élimination de Gauss) : place-les dans les colonnes d'une matrice et échelonne ; ils sont indépendants si et seulement si chaque colonne est une colonne pivot.

## Base, dimension et théorème rang-noyau

Une **base** de $V$ est une liste indépendante qui engendre $V$. Le livre énonce des caractérisations équivalentes : une base est une famille génératrice *minimale*, une famille indépendante *maximale*, et une liste telle que chaque vecteur possède un ensemble de coefficients **unique**. Les bases ne sont pas uniques (dans $\mathbb{R}^3$, la base canonique et $(1,0,0),(1,1,0),(1,1,1)$ conviennent toutes les deux) mais toutes les bases de $V$ ont la même taille, la **dimension** $\dim V$. La dimension compte des directions indépendantes, pas des composantes : la droite $\operatorname{span}\{(0,1)\}$ est dans $\mathbb{R}^2$ mais a pour dimension 1.

Pour trouver une base d'un sous-espace engendré : place les vecteurs générateurs en colonnes de $A$, échelonne, et garde les vecteurs d'origine situés dans les *colonnes pivots* (exemple 2.17 du livre : sur quatre vecteurs de $\mathbb{R}^5$, trois sont des colonnes pivots).

Le **rang** est $\operatorname{rk}(A)=\dim\operatorname{col}(A)$, le nombre de pivots ; il est aussi égal à la dimension de l'espace des lignes. Le **noyau** $\operatorname{null}(A)=\{x:Ax=\mathbf 0\}$ est l'ensemble des entrées que l'application écrase en zéro. Les deux sont liés :

$$
\boxed{\operatorname{rk}(A)+\dim\operatorname{null}(A)=n}\qquad (A\text{ has }n\text{ columns}).
$$

Pourquoi : échelonne $A$. Chaque colonne est soit une colonne pivot (elle ajoute une dimension à l'espace des colonnes), soit une colonne libre (elle ajoute un paramètre libre aux solutions de $Ax=\mathbf 0$, donc une dimension au noyau). Les $n$ colonnes se répartissent entre les deux types. En mots : sur les $n$ dimensions d'entrée, l'application en conserve $\operatorname{rk}(A)$ et détruit le reste. Ainsi $\operatorname{rk}(A)=n$ signifie que l'application ne perd rien, et une matrice large ($n>m$) a toujours un noyau non trivial.

## Coordonnées et changement de base

Soit $b_1,\dots,b_n$ une base de $\mathbb{R}^n$, empilée en colonnes d'une matrice inversible $S$. Un vecteur $x$ (en coordonnées standard) a pour coordonnées $c$ par rapport à cette base si

$$
x = c_1b_1+\dots+c_nb_n = S\,c \quad\Longrightarrow\quad c=S^{-1}x .
$$

Donc $S$ transforme les coordonnées de la base en coordonnées standard et $S^{-1}$ fait le chemin inverse. Exemple : $S=\begin{pmatrix}1&1\\0&1\end{pmatrix}$, $x=(3,2)$ : $c=S^{-1}x=(1,2)$ ; vérification $1\cdot(1,0)+2\cdot(1,1)=(3,2)$. Même point, nouvelle langue.

Une application linéaire de matrice $M$ en coordonnées standard agit sur les nouvelles coordonnées ainsi : convertir en standard ($S$), appliquer $M$, reconvertir ($S^{-1}$) :

$$
\tilde M = S^{-1} M S .
$$

C'est la Def. 2.22 du livre : $\tilde M$ et $M$ sont **semblables**. Des matrices semblables décrivent la *même* application dans des bases différentes, donc elles ont les mêmes valeurs propres, la même trace, le même déterminant et le même rang. La diagonalisation ([[math.07]]) est précisément la recherche d'une base (les vecteurs propres) dans laquelle $\tilde M$ est diagonale. Avec $M=\begin{pmatrix}2&1\\0&3\end{pmatrix}$ et le $S$ ci-dessus, $\tilde M=\operatorname{diag}(2,3)$ : les colonnes de $S$ sont les vecteurs propres $(1,0)$ et $(1,1)$.

## Orthogonalité, bases orthonormées et matrices orthogonales

Des vecteurs $x,y$ sont **orthogonaux** si $x^\top y=0$. Des vecteurs non nuls orthogonaux sont automatiquement indépendants. Une **base orthonormée** a des vecteurs deux à deux orthogonaux de longueur 1. Empile-les en colonnes de $Q$ :

$$
Q^\top Q = I .
$$

L'entrée $(i,j)$ de $Q^\top Q$ est $q_i^\top q_j$, qui vaut 1 sur la diagonale et 0 ailleurs. Si $Q$ est carrée, c'est une **matrice orthogonale** et $Q^{-1}=Q^\top$ : l'inverse ne coûte rien. Deux propriétés clés (livre 3.31-3.32) :

$$
\|Qx\|^2=(Qx)^\top(Qx)=x^\top Q^\top Q x=x^\top x,\qquad (Qx)^\top(Qy)=x^\top y .
$$

Donc $Q$ préserve les longueurs, les distances et les angles ; c'est une rotation, éventuellement avec une réflexion. Et dans une base orthonormée, les coordonnées sont gratuites : comme $x=\sum_j c_jq_j$ et $q_i^\top q_j=\delta_{ij}$, on obtient $c_i=q_i^\top x$, sans système linéaire à résoudre.

Le **complément orthogonal** $U^\perp$ d'un sous-espace $U\subseteq\mathbb{R}^n$ est l'ensemble des vecteurs orthogonaux à tout $U$. Si $\dim U=m$ alors $\dim U^\perp=n-m$, $U\cap U^\perp=\{\mathbf 0\}$, et tout $x$ se décompose de façon unique en $x=u+w$ avec $u\in U$, $w\in U^\perp$ (livre Sec. 3.6). Ce $u$ est la projection de la section suivante. Une conséquence utile : $\operatorname{null}(A)$ est le complément orthogonal de l'espace des lignes de $A$, puisque chaque équation $a_i^\top x=0$ dit $x\perp a_i$.

## Dérivation : la matrice de projection

Soit $U=\operatorname{col}(A)$ où $A\in\mathbb{R}^{n\times m}$ a des colonnes indépendantes ($m\le n$). Étant donné $x\in\mathbb{R}^n$, nous voulons le $p\in U$ le plus proche de $x$.

1. $p\in U$ signifie $p=A\lambda$ pour un certain vecteur de coefficients $\lambda\in\mathbb{R}^m$.
2. « Le plus proche » signifie que l'erreur $x-p$ est orthogonale à $U$ (la propriété de l'ombre ; c'est aussi ce que donne le calcul différentiel quand on minimise $\|x-A\lambda\|^2$). Orthogonale à $U$ signifie orthogonale à chaque colonne : $a_j^\top(x-A\lambda)=0$ pour tout $j$, soit, empilé, $A^\top(x-A\lambda)=\mathbf 0$.
3. Développe : $A^\top A\,\lambda=A^\top x$ (les **équations normales**, livre 3.56).
4. $A^\top A$ est $m\times m$ et inversible car les colonnes sont indépendantes, donc $\lambda=(A^\top A)^{-1}A^\top x$.
5. Alors $p=A\lambda$ :

$$
\boxed{P=A\,(A^\top A)^{-1}A^\top},\qquad p=Px .
$$

Propriétés, chacune en une ligne :

- $P^2=A(A^\top A)^{-1}\,(A^\top A)\,(A^\top A)^{-1}A^\top=P$ : projeter deux fois ne change rien (c'est la définition d'une projection dans le livre).
- $P^\top=P$ car $(A^\top A)^{-1}$ est symétrique. Idempotente plus symétrique, c'est ce qui en fait une projection *orthogonale*.
- Ses valeurs propres sont $1$ (sur $U$) et $0$ (sur $U^\perp$), donc $\operatorname{tr}P=m=\dim U$.
- Si les colonnes sont orthonormées ($A=Q$, $Q^\top Q=I$), elle se réduit à $P=QQ^\top$, la somme des ombres de rang un $q_jq_j^\top$.

Avec une seule colonne $b$, c'est $P=bb^\top/(b^\top b)$.

## Exemple détaillé : projection

Exemple 3.11 du livre : $A=\begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix}$, $x=(6,0,0)$.

- $A^\top A=\begin{pmatrix}3&3\\3&5\end{pmatrix}$, $A^\top x=(6,0)$, donc $\lambda=(5,-3)$.
- $p=A\lambda=(5,2,-1)$ ; l'erreur $x-p=(1,-2,1)$ a pour longueur $\sqrt6\approx2.449$.
- $P=\tfrac16\begin{pmatrix}5&2&-1\\2&2&2\\-1&2&5\end{pmatrix}$, et $\operatorname{tr}P=2$.
- Vérification : $A^\top(x-p)=(1-2+1,\ 0-2+2)=(0,0)$.

## Gram-Schmidt et la décomposition QR

**But.** Étant donné des colonnes indépendantes $a_1,\dots,a_n$, construire des $q_1,\dots,q_n$ orthonormés tels que $\operatorname{span}(q_1..q_k)=\operatorname{span}(a_1..a_k)$ pour tout $k$.

**Dérivation.** Étape 1 : $q_1=a_1/\|a_1\|$. Étape $k$ : on veut la part de $a_k$ qui est nouvelle. La part déjà couverte est la projection de $a_k$ sur le sous-espace engendré par $q_1..q_{k-1}$, qui, d'après la formule orthonormée ci-dessus, vaut $\sum_{i<k}(q_i^\top a_k)q_i$. Soustrais-la :

$$
v_k=a_k-\sum_{i<k}(q_i^\top a_k)\,q_i,\qquad q_k=\frac{v_k}{\|v_k\|}.
$$

$v_k$ est orthogonal à tous les $q_i$ précédents (l'erreur de projection est perpendiculaire au sous-espace) et non nul car $a_k$ est indépendant des colonnes précédentes. Lis maintenant les mêmes équations à l'envers : $a_k=\sum_{i<k}(q_i^\top a_k)q_i+\|v_k\|q_k$. Sous forme matricielle :

$$
\boxed{A=QR},\qquad R_{ik}=q_i^\top a_k\ (i<k),\quad R_{kk}=\|v_k\|>0,\quad R_{ik}=0\ (i>k).
$$

$Q$ est $n\times m$ à colonnes orthonormées, $R$ est $m\times m$ triangulaire supérieure. Dimensions : $A:(n,m)=Q:(n,m)\cdot R:(m,m)$ (QR « réduite » ; la QR « complète » complète $Q$ en $n\times n$ et $R$ en $n\times m$).

**Exemple détaillé.** $a_1=(1,1,0)$, $a_2=(1,0,1)$, $a_3=(0,1,1)$.

- $q_1=(1,1,0)/\sqrt2$, $R_{11}=\sqrt2\approx1.4142$.
- $R_{12}=q_1^\top a_2=1/\sqrt2\approx0.7071$ ; $v_2=a_2-R_{12}q_1=(0.5,-0.5,1)$ ; $R_{22}=\|v_2\|=\sqrt{1.5}\approx1.2247$ ; $q_2=v_2/R_{22}$.
- $R_{13}\approx0.7071$, $R_{23}\approx0.4082$, $R_{33}\approx1.1547$.

Cela coïncide avec `np.linalg.qr` au signe près : NumPy (LAPACK) renvoie une première colonne de $Q$ égale à $-q_1$ et un $R_{11}$ négatif. La décomposition n'est unique qu'à l'inversion près d'une colonne de $Q$ avec la ligne correspondante de $R$ ; exiger $R_{kk}>0$ la rend unique, donc fixe les signes avant de comparer.

**Classique contre modifié.** La formule ci-dessus calcule chaque coefficient $q_i^\top a_k$ à partir du $a_k$ *d'origine* (Gram-Schmidt classique). En virgule flottante, quand les colonnes sont presque dépendantes, les $q$ calculés perdent leur orthogonalité. La variante **modifiée** soustrait les projections une à une du vecteur courant, $v\leftarrow v-(q_i^\top v)q_i$. Elle est identique en arithmétique exacte et nettement meilleure en virgule flottante. Les bibliothèques de production utilisent à la place des réflexions de Householder (ce que `np.linalg.qr` appelle via LAPACK), qui sont encore meilleures.

## Moindres carrés via QR

Système surdéterminé : $A\in\mathbb{R}^{n\times m}$ avec $n>m$ et $b$ hors de l'espace des colonnes. Minimise $\|Ax-b\|^2$. D'après la dérivation de la projection, le minimiseur résout les équations normales $A^\top Ax=A^\top b$ (les équations de [[cml.01]]). Insère $A=QR$ :

$$
A^\top A=R^\top Q^\top QR=R^\top R,\qquad A^\top b=R^\top Q^\top b .
$$

Donc $R^\top Rx=R^\top Q^\top b$, et comme $R$ est inversible, $R^\top$ se simplifie :

$$
\boxed{Rx=Q^\top b}
$$

Résous par substitution arrière (le système est triangulaire, $O(m^2)$). Le produit $A^\top A$ n'est jamais formé.

**Pourquoi c'est plus stable.** Le conditionnement $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ mesure de combien les erreurs relatives des données ou de l'arithmétique peuvent être amplifiées dans la solution (voir [[math.15]]). Former $A^\top A$ le **met au carré** : $\kappa(A^\top A)=\kappa(A)^2$. Résoudre les équations normales perd donc environ deux fois plus de chiffres que ce qu'exige le problème. En float64 (environ 16 chiffres) avec $\kappa(A)=10^{5}$, les équations normales laissent à peu près $16-10=6$ chiffres et QR à peu près $16-5=11$. Comme $Q$ est orthogonale, multiplier par $Q^\top$ ne change pas le conditionnement et $\kappa(R)=\kappa(A)$.

Chiffres de l'exercice : ajuste un polynôme de degré 7 à 40 points équidistants de $[0,1]$ (matrice de Vandermonde $V$, de forme $40\times8$) avec des coefficients vrais tous égaux à $1$ et $y=V\mathbf 1$ sans bruit. Alors $\kappa(V)\approx1.1\times10^5$ et $\kappa(V^\top V)\approx1.2\times10^{10}$ ; les équations normales retrouvent les coefficients à environ $10^{-6}$ près (nous avons mesuré $8\times10^{-7}$), QR avec réflexions de Householder (`np.linalg.qr`) à environ $10^{-12}$. Une mise en garde découverte en écrivant l'exercice : notre propre *Gram-Schmidt modifié* suivi de $Q^\top b$ n'atteint qu'environ $2\times10^{-7}$, car $Q$ n'est orthonormée qu'à environ $10^{-12}$ près et cette erreur est amplifiée par $\kappa$. Appliquer Gram-Schmidt à la matrice augmentée $[A\mid b]$ et lire $Q^\top b$ dans la dernière colonne de $R$ rétablit environ $10^{-13}$. La QR de Householder est le choix robuste par défaut. Si la matrice peut être de rang déficient, utilise la voie de la SVD ([[math.07]]), ce que fait `np.linalg.lstsq`.

## Implémentation

```python
import numpy as np

def gram_schmidt(A):
    """Modified Gram-Schmidt. A: (n, m), independent columns -> Q (n, m), R (m, m)."""
    n, m = A.shape
    Q = np.zeros((n, m)); R = np.zeros((m, m))
    V = A.astype(float).copy()
    for k in range(m):
        R[k, k] = np.linalg.norm(V[:, k])
        Q[:, k] = V[:, k] / R[k, k]
        for j in range(k + 1, m):          # remove q_k from all later columns now
            R[k, j] = Q[:, k] @ V[:, j]
            V[:, j] -= R[k, j] * Q[:, k]
    return Q, R

A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1.]])
Q, R = gram_schmidt(A)
print(np.allclose(Q @ R, A), np.allclose(Q.T @ Q, np.eye(3)))
B = np.array([[1, 0], [1, 1], [1, 2.]])
P = B @ np.linalg.inv(B.T @ B) @ B.T      # use solve() when you only need P @ x
```

En production, `np.linalg.qr(A)` (Householder), `scipy.linalg.solve_triangular` et `np.linalg.lstsq` (basé sur la SVD) font ce travail.

## Dimensions

$A:(n,m)$ ; $P:(n,n)$ de rang $m$ ; $Q:(n,m)$ réduite ou $(n,n)$ complète ; $R:(m,m)$ réduite ou $(n,m)$ complète ; coordonnées $c=S^{-1}x$ avec $S:(n,n)$, $x:(n,)$, $c:(n,)$ ; $\tilde M=S^{-1}MS:(n,n)$.

## Pourquoi c'est important en ML

- **Espaces de caractéristiques.** Les colonnes d'une matrice de conception engendrent l'espace des fonctions qu'un modèle linéaire peut exprimer. Des caractéristiques colinéaires signifient des colonnes dépendantes, un rang déficient et un ajustement non unique ; un $\kappa$ grand signifie « presque ».
- **La régression est une projection.** Les valeurs ajustées $\hat y=X\hat w$ sont la projection de $y$ sur $\operatorname{col}(X)$, le résidu est orthogonal à chaque caractéristique, et $P$ est la « matrice chapeau » (hat matrix).
- **Base de l'ACP.** L'ACP choisit une base orthonormée de l'espace des caractéristiques ordonnée par variance ; les coordonnées d'un point dans cette base sont des produits scalaires $q_i^\top x$, et garder $k$ axes est une projection orthogonale sur un sous-espace de dimension $k$ ([[math.07]]).
- **Blanchiment.** Tourner dans la base propre de la covariance (un changement de base orthogonal) et diviser chaque axe par son écart type donne des caractéristiques non corrélées de variance unité.
- **Embeddings.** La dimension d'embedding est la dimension de l'espace ; les directions lisibles sont celles que l'on peut distinguer par des produits scalaires, et des vecteurs aléatoires de grande dimension sont presque orthogonaux. L'initialisation orthogonale des poids (prendre $Q$ de la QR d'une matrice aléatoire) évite que les normes explosent ou s'évanouissent, puisque $\|Qx\|=\|x\|$.

## Idées reçues

- **« Un sous-espace est n'importe quel ensemble plat. »** Il doit passer par l'origine. Les ensembles de solutions de $Ax=b$ avec $b\neq\mathbf 0$ sont des copies décalées, pas des sous-espaces.
- **« La dimension est la longueur des vecteurs. »** C'est le nombre de directions indépendantes ; une droite dans $\mathbb{R}^{100}$ a pour dimension 1.
- **« Matrice orthogonale veut dire lignes orthogonales seulement. »** Les lignes et les colonnes sont toutes orthonormées ; « matrice orthonormée » serait un meilleur nom. Une $Q$ non carrée avec $Q^\top Q=I$ a $QQ^\top\neq I$ (c'est une projection à la place).
- **« Orthogonal et non corrélé, c'est pareil. »** Seulement pour des données centrées.
- **« Une projection peut être inversée. »** Elle écrase $U^\perp$ en zéro ; l'information est perdue.
- **« Gram-Schmidt classique suffit. »** Mathématiquement oui, numériquement il peut perdre l'orthogonalité sur des colonnes presque dépendantes ; utilise la version modifiée ou Householder.
- **« Utilise simplement les équations normales ; c'est une ligne. »** Elles mettent le conditionnement au carré. Correct pour de petits problèmes bien conditionnés, risqué sinon.

## Points clés

- Test du sous-espace : contient $\mathbf 0$, fermé par $+$ et par multiplication scalaire. Le noyau et l'espace des colonnes sont des sous-espaces.
- Base = indépendante + génératrice ; dimension = sa taille ; $\operatorname{rk}(A)+\dim\operatorname{null}(A)=n$.
- Coordonnées : $c=S^{-1}x$ ; une application dans la nouvelle base est $S^{-1}MS$ (matrices semblables).
- $Q^\top Q=I$ donne des applications qui préservent les longueurs et des coordonnées gratuites $c=Q^\top x$.
- Projection sur $\operatorname{col}(A)$ : $P=A(A^\top A)^{-1}A^\top$, avec $P^2=P=P^\top$.
- Gram-Schmidt construit $Q$ et enregistre $R$ dans $A=QR$ ; les moindres carrés deviennent $Rx=Q^\top b$, plus stable que les équations normales car $\kappa(A^\top A)=\kappa(A)^2$.

## Explique-le toi-même

1. Pourquoi le plan $x+y+z=1$ n'est-il pas un sous-espace de $\mathbb{R}^3$, et comment est-il lié à un plan qui en est un ?
2. Sans formules : pourquoi l'erreur d'un ajustement par moindres carrés doit-elle être perpendiculaire à chaque colonne de caractéristiques ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 2.4-2.7 (PDF p.41-63) et Sec. 3.4-3.8 (PDF p.82-94).
- La leçon sur les systèmes linéaires du dépôt compagnon (`01-math-foundations/17-linear-systems`) pour LU, Cholesky et le conditionnement.

## Critères de maîtrise

Tu sais tester si un ensemble est un sous-espace, trouver une base et la dimension d'un espace des colonnes et d'un noyau et énoncer le théorème rang-noyau, convertir des coordonnées entre bases, dériver $P=A(A^\top A)^{-1}A^\top$ et vérifier $P^2=P=P^\top$, exécuter Gram-Schmidt à la main sur trois vecteurs, et expliquer pourquoi QR bat les équations normales sur un problème mal conditionné.
