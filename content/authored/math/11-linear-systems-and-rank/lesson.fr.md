---
id: math.11
title: "Systèmes linéaires, élimination de Gauss et rang"
summary: "Résous Ax = b par élimination, lis l'existence et l'unicité dans le rang et le noyau, replie-toi sur les moindres carrés quand il n'y a pas de réponse exacte, et comprends pourquoi on appelle solve plutôt qu'inv."
minutes: 80
teaches: [la.linear-systems]
requires: [la.matrix, la.linear-map]
sources:
  - "math/math for ML.pdf, Sec. 2.3 Solving Systems of Linear Equations (PDF p.33-41): particular and general solution, elementary transformations, row-echelon form, pivots, basic and free variables, reduced row-echelon form, the minus-1 trick, inverse by Gaussian elimination, pseudo-inverse and iterative methods. READ."
  - "math/math for ML.pdf, Sec. 2.4 Vector Spaces (PDF p.41-45): subspaces; the solution set of Ax = 0 is a subspace, that of Ax = b (b != 0) is not. READ (skimmed groups/vector-space axioms)."
  - "math/math for ML.pdf, Sec. 2.5 Linear Independence (PDF p.46-47): linear combination, dependence, the Gaussian-elimination test. READ (p.46-47 only)."
  - "math/math for ML.pdf, Sec. 2.6.2 Rank (PDF p.53-54): rank properties, rk(A) = rk(A|b) for solvability, dim of null space = n - rk(A), full rank, worked rank example. READ."
  - "math/math for ML.pdf, Sec. 2.7.3 Image and Kernel (PDF p.64-65): kernel, image, column space, Ax = 0 and the kernel. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/17-linear-systems/docs/en.md (first 14000 characters): row and column picture, Gaussian elimination with partial pivoting, LU, QR, Cholesky, least squares and the normal equations, pseudoinverse, condition number. READ."
---

## Où cela s'inscrit

Une énorme part du machine learning consiste à « trouver les inconnues qui satisfont ces équations ». Régression linéaire : trouver les poids $\mathbf w$ tels que $X\mathbf w$ colle aux cibles. Ajuster un processus gaussien, inverser une covariance pour la distance de Mahalanobis, calculer la pseudo-inverse utilisée par l'ACP et les moindres carrés, résoudre le pas suivant de la méthode de Newton (leçon 12) : chacun de ces problèmes est « résoudre $A\mathbf x=\mathbf b$ » déguisé.

Deux questions précèdent tout algorithme. *Une solution existe-t-elle ? Est-elle unique ?* La réponse se trouve dans deux nombres attachés à la matrice : son **rang** et la dimension de son **noyau**. Cette leçon construit l'image, puis la méthode standard (l'élimination de Gauss), puis ce qu'il faut faire quand il n'y a **pas de solution exacte** (presque toujours, en science des données : plus d'équations que d'inconnues, donc on utilise les moindres carrés), et enfin l'avertissement numérique qui explique pourquoi les bibliothèques numériques te donnent `solve` et non un bouton « inverser, puis multiplier ».

## L'idée en mots simples

- **Vue par les lignes.** Chaque équation est une feuille plate (une droite en 2-D, un plan en 3-D). Une solution est un point situé sur toutes les feuilles à la fois. Deux droites peuvent se croiser en un point, être parallèles et ne jamais se rencontrer, ou être confondues et partager une infinité de points : exactement les trois possibilités pour tout système linéaire (zéro, une ou une infinité de solutions ; jamais exactement deux).
- **Vue par les colonnes.** Lis le même système comme une recette : « quelles quantités $x_1,x_2,\dots$ des vecteurs colonnes, mélangées, donnent le vecteur $\mathbf b$ ? » L'ensemble de tout ce que tu peux construire est l'**espace des colonnes**. Une solution existe exactement quand $\mathbf b$ s'y trouve.
- L'**élimination de Gauss**, c'est ce que tu faisais à l'école avec deux équations : utiliser une équation pour éliminer une variable des autres, jusqu'à ce que le système devienne un escalier que tu résous de bas en haut. L'escalier montre combien d'équations sont réellement indépendantes (le **rang**) et quelles inconnues restent libres.
- **Variables libres = redondance.** Si les colonnes en contiennent une redondante (une colonne que tu pourrais construire à partir des autres), tu peux en ajouter un peu et en retirer autant ailleurs : la solution n'est pas unique, il y a toute une droite ou un plan de solutions. Cet ensemble de déplacements « invisibles », les vecteurs que $A$ envoie sur zéro, est le **noyau**.
- **Pas de solution exacte ?** Si $\mathbf b$ n'est pas dans l'espace des colonnes, trouve le point de l'espace des colonnes qui est le *plus proche* de $\mathbf b$ : les moindres carrés. C'est l'ombre de $\mathbf b$ sur l'espace des colonnes, et le vecteur d'erreur restant est perpendiculaire à cet espace.
- **Pourquoi ne pas simplement inverser ?** Imagine mesurer avec une règle dont les graduations sont presque identiques (deux droites presque parallèles). Un minuscule tremblement de la mesure déplace énormément le point de croisement. Cette sensibilité appartient au problème (le **conditionnement**) ; un bon algorithme ne peut pas l'éliminer, mais un algorithme négligent (comme calculer d'abord l'inverse complète) peut y ajouter ses propres erreurs et demande plus de travail.

## Définitions

Soit $A\in\mathbb R^{m\times n}$, $\mathbf x\in\mathbb R^n$, $\mathbf b\in\mathbb R^m$. Le système $A\mathbf x=\mathbf b$ a $m$ équations et $n$ inconnues.

- **Espace des colonnes** (image) : $\operatorname{Im}(A)=\{A\mathbf x\}=\operatorname{span}(\mathbf a_1,\dots,\mathbf a_n)\subseteq\mathbb R^m$.
- **Noyau** : $\ker(A)=\{\mathbf x:A\mathbf x=\mathbf 0\}\subseteq\mathbb R^n$, un sous-espace.
- **Rang** $\operatorname{rk}(A)$ : le nombre de colonnes linéairement indépendantes, qui égale le nombre de lignes linéairement indépendantes, et égale $\dim\operatorname{Im}(A)$. Il vérifie $\operatorname{rk}(A)\le\min(m,n)$ ; l'égalité est le **rang plein**.
- **Rang-nullité** (Deisenroth Sec. 2.6.2) : $\dim\ker(A)=n-\operatorname{rk}(A)$.

### Ce qui décide de l'existence et de l'unicité

| Condition | Signification |
|---|---|
| Une solution existe si et seulement si $\operatorname{rk}(A)=\operatorname{rk}(A\,|\,\mathbf b)$ | ajouter $\mathbf b$ comme colonne supplémentaire n'agrandit pas l'espace des colonnes |
| Solution unique si et seulement si elle existe et $\ker(A)=\{\mathbf 0\}$, c.-à-d. $\operatorname{rk}(A)=n$ | pas de variable libre |
| Une infinité de solutions si et seulement si elle existe et $\operatorname{rk}(A)<n$ | la solution générale est une solution particulière plus n'importe quoi dans $\ker(A)$ |
| Une $A$ carrée est inversible si et seulement si $\operatorname{rk}(A)=n$ | |

**Solution générale = particulière + noyau.** Si $A\mathbf x_p=\mathbf b$ et $A\mathbf n=\mathbf 0$, alors $A(\mathbf x_p+\mathbf n)=\mathbf b+\mathbf 0=\mathbf b$. Réciproquement, deux solutions quelconques diffèrent d'un vecteur du noyau. Donc l'ensemble des solutions de $A\mathbf x=\mathbf b$ est une copie décalée de $\ker(A)$ ; pour $\mathbf b\ne\mathbf 0$ ce n'est *pas* un sous-espace (il ne contient pas $\mathbf 0$).

## Élimination de Gauss

Trois **opérations élémentaires sur les lignes** ne changent pas l'ensemble des solutions : échanger deux lignes, multiplier une ligne par une constante non nulle, ajouter un multiple d'une ligne à une autre. Écris le système sous forme de matrice augmentée $[A\,|\,\mathbf b]$ et applique-les pour atteindre la **forme échelonnée** (REF) : des lignes nulles en bas, et le premier coefficient non nul de chaque ligne (son **pivot**) strictement à droite du pivot au-dessus. Cet escalier donne :

- **Rang** = nombre de pivots.
- **Variables de base** = inconnues dans les colonnes de pivot ; **variables libres** = les autres.
- Une ligne de la forme $[0\ \cdots\ 0\,|\,c]$ avec $c\ne0$ signifie « $0=c$ » : pas de solution.

Avec chaque pivot mis à 1 et des zéros aussi au-dessus des pivots, on obtient la **forme échelonnée réduite** (RREF), d'où l'on peut lire directement le noyau (l'*astuce du moins-1* du livre : pour chaque colonne libre, place $-1$ à sa position diagonale et lis les colonnes).

### Un premier exemple (celui du livre, Sec. 2.3.1)

$$
\begin{pmatrix}1&0&8&-4\\0&1&2&12\end{pmatrix}\mathbf x=\begin{pmatrix}42\\8\end{pmatrix}.
$$

Deux équations, quatre inconnues : attends-toi à une infinité de solutions. Les colonnes de pivot sont 1 et 2. Une solution particulière prend $42$ fois la colonne 1 et $8$ fois la colonne 2 : $\mathbf x_p=(42,8,0,0)$. La colonne 3 égale $8\cdot$col1 $+\,2\cdot$col2, donc $(8,2,-1,0)$ est envoyé sur zéro ; la colonne 4 égale $-4\cdot$col1 $+\,12\cdot$col2, donc $(-4,12,0,-1)$ l'est aussi. Par conséquent

$$
\mathbf x=\begin{pmatrix}42\\8\\0\\0\end{pmatrix}+\lambda_1\begin{pmatrix}8\\2\\-1\\0\end{pmatrix}+\lambda_2\begin{pmatrix}-4\\12\\0\\-1\end{pmatrix},\qquad\lambda_1,\lambda_2\in\mathbb R .
$$

Ici $n=4$, rang $2$, nullité $2$ : deux paramètres libres, comme promis.

### Une élimination entièrement numérique (carrée, solution unique)

$$
\begin{pmatrix}2&1&1\\4&3&3\\2&3&1\end{pmatrix}\mathbf x=\begin{pmatrix}8\\20\\12\end{pmatrix}.
$$

Soustrais $2\times$ligne 1 de la ligne 2 et $1\times$ligne 1 de la ligne 3 :

$$
\left(\begin{array}{ccc|c}2&1&1&8\\0&1&1&4\\0&2&0&4\end{array}\right)\ \to\ \left(\begin{array}{ccc|c}2&1&1&8\\0&1&1&4\\0&0&-2&-4\end{array}\right)\quad(\text{row 3}-2\times\text{row 2}).
$$

Substitution remontante depuis le bas : $-2x_3=-4\Rightarrow x_3=2$ ; $x_2+x_3=4\Rightarrow x_2=2$ ; $2x_1+x_2+x_3=8\Rightarrow x_1=2$. Les multiplicateurs $2,1,2$ sont les entrées du facteur inférieur dans $A=LU$ ; le résultat est la matrice triangulaire supérieure $U$. Trois pivots, rang 3, solution unique $(2,2,2)$.

### Pivotage et coût

Éliminer coûte environ $n^3$ opérations (le livre note que la croissance cubique est impraticable pour des millions d'inconnues, où l'on emploie plutôt des méthodes itératives). Si un pivot est nul, tu dois échanger des lignes ; s'il est simplement *minuscule*, tu devrais aussi échanger : diviser par un pivot minuscule crée d'énormes multiplicateurs qui amplifient les erreurs d'arrondi. Le **pivotage partiel** choisit le plus grand coefficient de la colonne à chaque étape. Avec un système $2\times2$ $\begin{psmallmatrix}10^{-20}&1\\1&1\end{psmallmatrix}\mathbf x=\begin{psmallmatrix}1\\2\end{psmallmatrix}$ (vraie solution $\approx(1,1)$), l'élimination sans pivotage renvoie $(0,1)$ en float64, avec pivotage $(1,1)$.

Si tu dois résoudre de nombreux systèmes avec le même $A$ et des $\mathbf b$ différents, factorise une fois $PA=LU$ (coût $\sim n^3$) et alors chaque second membre ne coûte que $\sim n^2$ (substitutions descendante et remontante).

### Le rang par élimination (l'exemple du livre, p.54)

$$
A=\begin{pmatrix}1&2&1\\-2&-3&1\\3&5&0\end{pmatrix}\ \rightsquigarrow\ \begin{pmatrix}1&2&1\\0&1&3\\0&0&0\end{pmatrix}:
$$

deux pivots, rang 2 : la troisième ligne est $\text{row}_1+\text{row}_2$ dans la matrice d'origine. La matrice est carrée mais *non* inversible (nullité $3-2=1$). En virgule flottante, on compte les valeurs singulières au-dessus d'une tolérance plutôt que des pivots exactement nuls.

### L'existence dépend de $\mathbf b$ (l'exemple 2.6 du livre)

Le système de matrice des coefficients de lignes $(-2,4,-2,-1,4)$, $(4,-8,3,-3,1)$, $(1,-2,1,-1,1)$, $(1,-2,0,-3,4)$ et de second membre $(-3,2,0,a)$ se réduit à un escalier à trois pivots et une dernière ligne $0=a+1$. Donc $\operatorname{rk}(A)=3$, et le système n'est résoluble que pour $a=-1$ (alors $\operatorname{rk}(A|\mathbf b)=3$ aussi ; pour $a=0$ le rang augmenté vaut $4$). Pour $a=-1$ une solution particulière est $(2,0,-1,1,0)$ et le noyau est engendré par $(2,1,0,0,0)$ et $(2,0,-1,2,1)$, donc cinq inconnues, rang 3, deux paramètres libres.

## Quand il n'y a pas de solution exacte : les moindres carrés

En ajustement de données $m\gg n$ (beaucoup de points, peu de paramètres), et $\mathbf b\notin\operatorname{Im}(A)$ presque sûrement. On minimise le résidu :

$$
\min_{\mathbf x}\ \|A\mathbf x-\mathbf b\|^2 .
$$

**Démonstration.** Développe $\|A\mathbf x-\mathbf b\|^2=\mathbf x^\top A^\top A\mathbf x-2\mathbf b^\top A\mathbf x+\mathbf b^\top\mathbf b$. Son gradient par rapport à $\mathbf x$ est $2A^\top A\mathbf x-2A^\top\mathbf b$ (avec les règles de gradient de la leçon 5 ; $A^\top A$ est symétrique). L'annuler donne les **équations normales**

$$
A^\top A\,\mathbf x=A^\top\mathbf b .
$$

Géométriquement, réécris-les $A^\top(\mathbf b-A\mathbf x)=\mathbf 0$ : le résidu est orthogonal à chaque colonne de $A$, c.-à-d. $A\mathbf x$ est la projection orthogonale de $\mathbf b$ sur l'espace des colonnes. Si $A$ a des colonnes linéairement indépendantes ($\operatorname{rk}A=n$), alors $A^\top A\in\mathbb R^{n\times n}$ est inversible et $\mathbf x=(A^\top A)^{-1}A^\top\mathbf b$, la solution par pseudo-inverse de Moore-Penrose mentionnée dans le livre (Sec. 2.3.4), qui est aussi la solution des moindres carrés de norme minimale. C'est la régression linéaire : $A$ est la matrice de données (de conception), $\mathbf x$ les poids.

**Avertissement numérique.** Former $A^\top A$ élève au carré le conditionnement (voir plus bas). Le code de production utilise une factorisation QR ou la SVD (`np.linalg.lstsq` utilise la SVD) plutôt que les équations normales, et le livre lui-même dit que, pour la précision numérique, il n'est généralement pas recommandé de calculer l'inverse ou la pseudo-inverse. Si $\operatorname{rk}(A)<n$, le minimiseur n'est pas unique ; la pseudo-inverse via la SVD choisit celui de plus petite norme.

## Conditionnement et pourquoi on appelle solve, pas inv

Le **conditionnement** $\kappa(A)=\|A\|\,\|A^{-1}\|=\sigma_{\max}/\sigma_{\min}$ (rapport de la plus grande à la plus petite valeur singulière, leçons précédentes) borne l'amplification des erreurs relatives sur $\mathbf b$ ou $A$ dans $\mathbf x$. Règle empirique : tu perds environ $\log_{10}\kappa$ chiffres décimaux sur les 16 environ dont tu disposes en float64. Si $\kappa$ est proche de $10^{16}$, la matrice est numériquement singulière.

Conséquences pratiques :

1. **Des colonnes presque dépendantes** (variables corrélées) signifient un $\sigma_{\min}$ minuscule, un $\kappa$ grand et des poids instables. La régularisation ridge $A^\top A+\lambda I$ relève $\sigma_{\min}^2$ à $\sigma_{\min}^2+\lambda$ et stabilise le problème.
2. **N'écris jamais `inv(A) @ b`.** `np.linalg.solve(A, b)` effectue une factorisation LU avec pivotage partiel et deux résolutions triangulaires ; elle ne forme jamais $A^{-1}$, fait moins de travail et est généralement au moins aussi précise. Inverser n'est justifié que si tu as réellement besoin des coefficients de $A^{-1}$.
3. Pour $A$ symétrique définie positive (covariances, matrices noyaux, $A^\top A+\lambda I$), utilise une factorisation de Cholesky : environ la moitié du travail d'une LU.
4. Ne te fie pas à un déterminant proche de zéro pour tester la singularité (il dépend de l'échelle) ; utilise le rang ou $\kappa$.

## Exemple détaillé

Tous les nombres sont reproduits dans l'exercice.

**1. Solution unique.** Le système $3\times3$ ci-dessus : `solve` renvoie $(2,2,2)$, déterminant $-4$ (non nul, donc inversible).

**2. Rang et nullité.** La matrice $3\times3$ de lignes $(1,2,1),(-2,-3,1),(3,5,0)$ a pour rang $2$ (valeurs singulières $7.21,\ 1.42,\ \approx10^{-16}$) ; la matrice $4\times5$ de l'exemple 2.6 a pour rang 3, nullité 2. Vérification des vecteurs du noyau : $(2,1,0,0,0)$ et $(2,0,-1,2,1)$ sont tous deux envoyés sur $\mathbf 0$ par cette matrice, et $(2,0,-1,1,0)$ est envoyé sur $(-3,2,0,-1)$, le second membre pour $a=-1$.

**3. Moindres carrés.** Ajuste une droite $t\approx x_0+x_1s$ aux points $(s,t)=(1,3),(2,5),(3,6),(4,8)$. Ici

$$
A=\begin{pmatrix}1&1\\1&2\\1&3\\1&4\end{pmatrix},\ \mathbf b=\begin{pmatrix}3\\5\\6\\8\end{pmatrix},\quad A^\top A=\begin{pmatrix}4&10\\10&30\end{pmatrix},\quad A^\top\mathbf b=\begin{pmatrix}22\\63\end{pmatrix}.
$$

Résolution : déterminant $4\cdot30-100=20$, donc $\mathbf x=\tfrac1{20}\begin{psmallmatrix}30&-10\\-10&4\end{psmallmatrix}\begin{psmallmatrix}22\\63\end{psmallmatrix}=\tfrac1{20}(30,\,32)=(1.5,\,1.6)$. Ordonnée à l'origine $1.5$, pente $1.6$. Le résidu est $\mathbf b-A\mathbf x=(-0.1,\,0.3,\,-0.3,\,0.1)$, de norme au carré $0.2$, et $A^\top(\mathbf b-A\mathbf x)=\mathbf 0$ aux arrondis près : le résidu est orthogonal aux deux colonnes. `np.linalg.lstsq` donne le même $(1.5,1.6)$. (La leçon du dépôt compagnon cite $1.7$ pour la pente ; le recalcul à la main et par le code donne $1.6$.)

**4. Conditionnement.** $\kappa(A)=7.47$ mais $\kappa(A^\top A)=55.78=7.47^2$ : les équations normales élèvent le conditionnement au carré. Pour $B=\begin{psmallmatrix}1&1\\1&1.0001\end{psmallmatrix}$ ($\kappa\approx4\times10^4$) : avec $\mathbf b=(2,2.0001)$ la solution est $(1,1)$ ; change $\mathbf b$ en $(2,\,2.0002)$ (une variation relative de $5\times10^{-5}$) et la solution saute à $(0,2)$.

## Implémentation

```python
import numpy as np

A = np.array([[2., 1, 1], [4, 3, 3], [2, 3, 1]])
b = np.array([8., 20, 12])
x = np.linalg.solve(A, b)                 # LU with partial pivoting, shape (3,)

rank = np.linalg.matrix_rank(np.array([[1., 2, 1], [-2, -3, 1], [3, 5, 0]]))   # 2

X = np.array([[1., 1], [1, 2], [1, 3], [1, 4]])      # (m, n) = (4, 2)
t = np.array([3., 5, 6, 8])                          # (m,)
w, *_ = np.linalg.lstsq(X, t, rcond=None)            # (n,) = [1.5, 1.6]
np.linalg.cond(X)                                    # 7.47
```

## Formes

$A:(m,n)$, $\mathbf x:(n,)$, $\mathbf b:(m,)$. Matrice augmentée $(m,n+1)$. Base du noyau : une matrice $(n,n-r)$ dont les colonnes sont des solutions de $A\mathbf x=\mathbf 0$. Équations normales : $A^\top A:(n,n)$, $A^\top\mathbf b:(n,)$. Pour `np.linalg.solve`, $A$ doit être carrée $(n,n)$ ; pour une $A$ rectangulaire, utilise `lstsq`.

## Idées fausses courantes

- **Un système carré a toujours exactement une solution.** Seulement si la matrice est de rang plein ; sinon il y en a zéro ou une infinité.
- **Plus d'équations que d'inconnues signifie pas de solution.** Pas nécessairement : si les équations supplémentaires sont cohérentes (redondantes), une solution existe. Typiquement pourtant, des données bruitées les rendent incohérentes, d'où les moindres carrés.
- **`x = inv(A) @ b` est la méthode standard.** Utilise `solve` ; c'est plus rapide et plus précis.
- **Les moindres carrés donnent une solution de $A\mathbf x=\mathbf b$.** Ils donnent le point le plus proche de l'espace des colonnes ; le résidu est non nul.
- **Un petit déterminant signifie mal conditionné.** Multiplier une matrice par $10^{-3}$ change énormément le déterminant et pas du tout le conditionnement.

## Points clés à retenir

- Des solutions existent si et seulement si $\operatorname{rk}(A)=\operatorname{rk}(A|\mathbf b)$ ; elles sont uniques si en plus $\operatorname{rk}(A)=n$ ; sinon elles forment solution particulière $+\ \ker(A)$ avec $\dim\ker(A)=n-\operatorname{rk}(A)$.
- L'élimination de Gauss transforme le système en escalier : les pivots donnent le rang, les colonnes libres donnent le noyau ; utilise le pivotage partiel.
- Pas de solution exacte : $\min\|A\mathbf x-\mathbf b\|^2$, équations normales $A^\top A\mathbf x=A^\top\mathbf b$, résidu orthogonal à l'espace des colonnes. Préfère QR ou la SVD pour la précision.
- $\kappa=\sigma_{\max}/\sigma_{\min}$ mesure la sensibilité ; appelle `solve` (ou `lstsq`), ne forme pas l'inverse.

## Explique-le avec tes mots

1. Une matrice $3\times3$ a pour rang 2. Que peut-il arriver à $A\mathbf x=\mathbf b$, selon $\mathbf b$, et combien y a-t-il de paramètres libres ?
2. Pourquoi le résidu des moindres carrés est-il perpendiculaire aux colonnes de $A$, et pourquoi cela donne-t-il les équations normales ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 2.3 (PDF p.33), Sec. 2.6.2 (PDF p.53), Sec. 2.7.3 (PDF p.64).
- Leçon du dépôt `17-linear-systems` pour la LU, la QR, Cholesky, la méthode du gradient conjugué et les solveurs itératifs.

## Critères de maîtrise

Tu sais amener à la main un petit système en forme échelonnée, compter le rang et les variables libres, écrire la solution générale comme particulière plus noyau, décider de la résolubilité à partir de $\operatorname{rk}(A)$ contre $\operatorname{rk}(A|\mathbf b)$, poser et résoudre les équations normales d'un ajustement de droite, et expliquer ce que le conditionnement dit de la confiance à accorder à une solution.
