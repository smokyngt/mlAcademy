---
id: math.07
title: "Valeurs propres, vecteurs propres et SVD"
summary: "Les directions qu'une matrice se contente d'étirer, comment elles permettent d'élever une matrice à une puissance, et la décomposition qui marche pour toute matrice."
minutes: 70
teaches: [la.eigen, la.svd]
requires: [la.linear-map]
sources:
  - "math/math for ML.pdf, Sec. 4.2 Eigenvalues and Eigenvectors (PDF p.111-114): definition, equivalent statements, characteristic polynomial, worked example A = [[4,2],[1,3]], multiplicities. READ."
  - "math/math for ML.pdf, Sec. 4.4 Eigendecomposition and Diagonalization (PDF p.121-123): A = P D P^-1 iff the eigenvectors form a basis. READ."
  - "math/math for ML.pdf, Sec. 4.5 Singular Value Decomposition (PDF p.125-128): SVD theorem, geometric intuition, worked example with A in R^{3x2}. READ."
---

## Où cela s'inscrit

Dans la leçon 3, une matrice était une transformation qui déforme le plan. La plupart des directions sont tournées *et* étirées, ce qui rend une transformation difficile à se représenter. Deux questions simplifient tout : **existe-t-il une direction qui est seulement étirée, pas tournée ?** (les vecteurs propres) et, pour les matrices qui ne sont pas carrées ou pas sympathiques, **quelle est la description la plus simple de « tourner, étirer, tourner » ?** (la SVD).

Ces outils sont partout en aval : l'ACP (la leçon sur les composantes principales) est un problème de valeurs propres ; la stabilité de la descente de gradient (leçon 7) est décidée par les valeurs propres de la courbure de la perte ; l'adaptation de bas rang des grands modèles (LoRA) est une idée de SVD tronquée ; et les valeurs singulières d'une matrice de poids te disent comment elle amplifie les signaux pendant la rétropropagation.

## L'idée en mots simples

- Imagine une feuille de caoutchouc sur laquelle est dessinée une grille, et une matrice comme une manière d'étirer la feuille. Presque toute flèche dessinée sur la feuille change de direction quand on l'étire. Mais il existe généralement quelques directions spéciales le long desquelles la feuille est **seulement tirée ou comprimée, jamais tordue** : une flèche pointant dans ce sens reste sur sa propre droite. Ce sont les **vecteurs propres**, et l'ampleur de l'étirement le long de chacun est sa **valeur propre** (une valeur négative signifie que la flèche est aussi retournée). Le mot « eigen » est allemand pour « propre » ou « caractéristique » (le livre fait la même remarque) : ce sont les directions propres de la transformation.
- Si tu utilises ces directions spéciales comme axes de coordonnées, la transformation devient trivialement simple : un simple étirement indépendant le long de chaque axe. C'est la **diagonalisation**, et elle rend l'application répétée peu coûteuse : appliquer la matrice 100 fois revient simplement à élever chaque facteur d'étirement à la puissance 100.
- Toutes les matrices ne sont pas carrées ni n'ont assez de directions spéciales. La **SVD** est la version universelle : *toute* matrice, de n'importe quelle forme, est égale à « tourner (ou réfléchir), puis étirer le long des axes, puis tourner de nouveau ». Les facteurs d'étirement sont les **valeurs singulières**.

## Valeurs propres et vecteurs propres

**Définition** (Deisenroth et al., Déf. 4.6). Pour une matrice carrée `A ∈ R^{n×n}`, un nombre `λ` est une *valeur propre* et un vecteur non nul `x` le *vecteur propre* associé si

$$
Ax=\lambda x .
$$

Quatre énoncés équivalents (le livre les énumère) : `λ` est une valeur propre ⇔ `(A − λI)x = 0` a une solution non triviale ⇔ `A − λI` est singulière (rang `< n`) ⇔ `det(A − λI) = 0`.

La dernière forme donne une recette. `p_A(λ) = det(A − λI)` est le **polynôme caractéristique**, et les valeurs propres en sont les racines. Pour une matrice `2×2` `[[a,b],[c,d]]` :

$$
p(\lambda)=\lambda^2-(a+d)\lambda+(ad-bc)=\lambda^2-\operatorname{tr}(A)\,\lambda+\det(A).
$$

Deux faits utiles en découlent : les **valeurs propres ont pour somme la trace** et **pour produit le déterminant**. Les vecteurs propres ne sont pas uniques : si `x` convient, `cx` aussi pour tout `c ≠ 0`, car `A(cx) = λ(cx)`. L'ensemble de tous les vecteurs propres d'une valeur propre, plus zéro, est un sous-espace appelé **espace propre**, le noyau de `A − λI`.

### Exemple traité (celui utilisé dans le livre)

$$
A=\begin{pmatrix}4&2\\1&3\end{pmatrix}.
$$

1. Polynôme caractéristique : `(4 − λ)(3 − λ) − 2·1 = λ² − 7λ + 10 = (λ − 2)(λ − 5)`. Vérification : trace `7 = 2 + 5`, déterminant `10 = 2·5`. ✓
2. Pour `λ = 5` : `(A − 5I)x = [[−1,2],[1,−2]]x = 0` donne `x₁ = 2x₂`, donc `E₅ = span{(2, 1)}`. Vérification : `A(2,1) = (4·2+2·1, 1·2+3·1) = (10, 5) = 5·(2,1)`. ✓
3. Pour `λ = 2` : `[[2,2],[1,1]]x = 0` donne `x₂ = −x₁`, donc `E₂ = span{(1, −1)}`. Vérification : `A(1,−1) = (4−2, 1−3) = (2, −2) = 2·(1,−1)`. ✓

La matrice tire le plan d'un facteur 5 le long de `(2,1)` et d'un facteur 2 le long de `(1,−1)`.

### Faits à retenir

- `A` et `Aᵀ` ont les mêmes valeurs propres (pas forcément les mêmes vecteurs propres).
- Une matrice symétrique a des valeurs propres **réelles** et des vecteurs propres de valeurs propres différentes **orthogonaux** ; une matrice symétrique *définie positive* a toutes ses valeurs propres `> 0`. C'est le cas qui compte pour les matrices de covariance (ACP) et pour la hessienne d'une perte convexe (descente de gradient).
- Les valeurs propres avec multiplicité peuvent être défectueuses : `[[2,1],[0,2]]` a la valeur propre 2 deux fois mais une seule direction `(1,0)` (l'exemple 4.6 du livre).
- Une rotation de 90 degrés n'a aucun vecteur propre réel (elle tourne toutes les directions) ; ses valeurs propres sont complexes.

## Diagonalisation

Place les vecteurs propres comme colonnes de `P` et les valeurs propres sur la diagonale de `D`. Alors `AP = PD`, car la `j`-ième colonne de `AP` est `Ap_j = λ_j p_j`, la `j`-ième colonne de `PD`. Si les vecteurs propres forment une base (donc `P` est inversible), alors

$$
\boxed{A=PDP^{-1}}\qquad\text{(Theorem 4.20 in the book: possible iff the eigenvectors span }\mathbb{R}^n\text{).}
$$

Lis de droite à gauche, comme le fait la figure du livre : `P⁻¹` passe aux coordonnées des vecteurs propres, `D` étire chaque axe, `P` revient en arrière. Conséquences : `A^k = P D^k P⁻¹` (seuls les nombres de la diagonale sont élevés à une puissance), `det A = ∏ λ_i`, et `A⁻¹ = P D⁻¹ P⁻¹`.

Exemple : avec `P = [[2,1],[1,−1]]` (colonnes `(2,1)` et `(1,−1)`) et `D = diag(5, 2)` : `A³ = P·diag(125, 8)·P⁻¹ = [[86,78],[39,47]]`, ce qui coïncide avec le produit de `A` trois fois à la main.

## La décomposition en valeurs singulières

**Théorème** (livre, Th. 4.22). Toute matrice `A ∈ R^{m×n}`, carrée ou non, de rang `r`, peut s'écrire

$$
\boxed{A=U\,\Sigma\,V^\top}\qquad U:(m\times m)\text{ orthogonal},\ \ \Sigma:(m\times n)\text{ diagonal},\ \ V:(n\times n)\text{ orthogonal},
$$

avec `Σ_ii = σ_i ≥ 0` les **valeurs singulières**, ordonnées `σ₁ ≥ σ₂ ≥ … ≥ 0`, dont `r` sont non nulles. Les colonnes de `U` sont les vecteurs singuliers à gauche, celles de `V` les vecteurs singuliers à droite.

**Géométrie** (la fig. 4.8 du livre) : `Vᵀ` tourne (un changement de base) dans l'espace d'entrée, `Σ` étire chaque axe de `σ_i` et ajoute ou supprime des dimensions pour atteindre l'espace de sortie, `U` tourne dans l'espace de sortie. Un cercle unité devient une ellipse dont les demi-axes ont pour longueurs `σ_i`. La décomposition en valeurs propres applique un changement de base et le défait dans le même espace ; la SVD utilise deux bases différentes, reliées par `Σ`.

**D'où cela vient.** `AᵀA = VΣᵀUᵀUΣVᵀ = V(ΣᵀΣ)Vᵀ`. Donc `V` contient les vecteurs propres de la matrice symétrique `AᵀA`, et les valeurs propres sont `σ_i²`. D'où `σ_i = √λ_i(AᵀA)`, et `u_i = A v_i / σ_i` pour `σ_i > 0`. (C'est aussi pourquoi l'ACP, qui diagonalise `XᵀX`, peut se faire avec la SVD de `X`.)

### Exemple traité (tiré du livre, Exemple 4.12)

$$
A=\begin{pmatrix}1&-0.8\\0&1\\1&0\end{pmatrix}\ (3\times2).\qquad A^\top A=\begin{pmatrix}2&-0.8\\-0.8&1.64\end{pmatrix}.
$$

Sa trace est `3.64` et son déterminant `2·1.64 − 0.64 = 2.64`, donc les valeurs propres sont `(3.64 ± 1.64)/2 = 2.64` et `1.00`. Par conséquent `σ₁ = √2.64 ≈ 1.625` et `σ₂ = 1`, ce qui concorde avec les `1.62` et `1.0` affichés dans le livre. Une matrice `3×2` envoie le plan sur une feuille 2D inclinée à l'intérieur de l'espace 3D.

### Approximation de bas rang

Ne garde que les `k` plus grandes valeurs singulières : `A_k = Σ_{i≤k} σ_i u_i v_iᵀ`. C'est la meilleure approximation de rang `k` de `A` au sens des moindres carrés (Eckart-Young ; la Sec. 4.6 du livre), et l'erreur vaut exactement

$$
\|A-A_k\|_F=\sqrt{\sigma_{k+1}^2+\dots+\sigma_r^2}.
$$

Stocker `A_k` coûte `k(m+n+1)` nombres au lieu de `mn` : c'est la base de la compression d'images, de l'ACP, et des adaptateurs de bas rang pour les grands modèles de langage.

## Implémentation

```python
import numpy as np

A = np.array([[4., 2.], [1., 3.]])
vals, vecs = np.linalg.eig(A)             # 5, 2 (order not guaranteed); columns are eigenvectors
Ak = vecs @ np.diag(vals ** 3) @ np.linalg.inv(vecs)   # A cubed

U, S, Vt = np.linalg.svd(np.array([[1, -.8], [0, 1], [1, 0]]), full_matrices=False)
```

Pour les matrices symétriques, utilise `np.linalg.eigh` (plus rapide, renvoie des vecteurs propres orthonormés, par ordre croissant).

## Formes

`A: (m,n)` ; SVD réduite `U: (m,r)`, `S: (r,)`, `Vt: (r,n)` avec `r = min(m,n)` ; `A ≈ (U[:, :k] * S[:k]) @ Vt[:k]`. Décomposition en valeurs propres : `P: (n,n)`, `D: (n,n)`.

## Idées fausses courantes

- **Toute matrice a des vecteurs propres réels.** Non (les rotations), et toute matrice n'est pas diagonalisable (les matrices défectueuses). La SVD existe toujours.
- **Les valeurs singulières sont les valeurs propres de `A`.** Ce sont les racines carrées des valeurs propres de `AᵀA`. Elles ne coïncident que pour les matrices symétriques semi-définies positives.
- **Les vecteurs propres sont uniques.** Seulement à un facteur d'échelle (et un signe) près ; ne les compare pas terme à terme.
- **Une valeur propre nulle est inoffensive.** Elle signifie que la matrice écrase une direction et n'est pas inversible.

## Points clés à retenir

- `Ax = λx` : les vecteurs propres sont les directions seulement étirées ; `tr A = Σλ`, `det A = Πλ`.
- Diagonalise pour calculer des puissances : `A^k = P D^k P⁻¹`.
- Matrices symétriques : valeurs propres réelles, vecteurs propres orthogonaux (le cas de l'ACP et de la hessienne).
- La SVD `A = UΣVᵀ` existe pour toute matrice ; `σ_i² = λ_i(AᵀA)` ; la tronquer donne la meilleure approximation de bas rang.

## Explique-le toi-même

1. Pourquoi `det(A − λI) = 0` traduit-il « il existe une direction seulement étirée » ?
2. Pourquoi la SVD existe-t-elle pour une matrice `3×2` alors que la décomposition en valeurs propres n'existe pas ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 4.2 (PDF p.111), 4.4 (p.121), 4.5 (p.125), 4.6 approximation de bas rang (p.135).

## Critères de maîtrise

Tu sais trouver à la main les valeurs propres et les vecteurs propres d'une matrice `2×2`, la diagonaliser, énoncer ce que les valeurs singulières signifient géométriquement, et calculer une approximation de bas rang avec son erreur.
