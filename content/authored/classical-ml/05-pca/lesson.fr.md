---
id: cml.05
title: "Analyse en composantes principales"
summary: "Trouver les quelques directions qui portent l'essentiel de la variance, les dériver avec un multiplicateur de Lagrange, et mesurer exactement ce que l'on perd en abandonnant le reste."
minutes: 65
teaches: [ml.pca]
requires: [la.matrix, la.dot, ml.kmeans]
sources:
  - math/math for ML.pdf (Ch. 10 Dimensionality Reduction with Principal Component Analysis, PDF p.323)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 12.1 Principal Component Analysis, PDF p.581; Appendix E Lagrange Multipliers, PDF p.727
  - ai-engineering-from-scratch/phases/01-math-foundations/10-dimensionality-reduction/docs/en.md
  - ai-engineering-from-scratch/phases/01-math-foundations/11-singular-value-decomposition/docs/en.md
---

## Où cela s'inscrit

Les données réelles ont beaucoup de colonnes, mais elles sont rarement indépendantes : taille et poids évoluent ensemble, des pixels voisins sont presque égaux, des centaines de comptages de mots reflètent quelques thèmes. Travailler avec quelques combinaisons bien choisies des colonnes rend les données plus faciles à **visualiser** (projeter 100 dimensions en 2), **moins coûteuses** à traiter, et souvent **moins bruitées**. L'ACP trouve ces combinaisons. C'est aussi la première fois que tu utilises des vecteurs propres, qui reviennent dans les méthodes spectrales, dans l'analyse de la dynamique d'entraînement et dans l'adaptation de bas rang des grands modèles.

## L'idée en mots simples

Imagine un nuage de points en forme de long cigare mince incliné dans le plan. Si tu dois décrire chaque point avec **un** seul nombre, tu devrais noter où il se trouve *le long du cigare*, car c'est là que les points diffèrent le plus ; la position en travers du cigare est presque la même pour tous et porte peu d'information. L'ACP trouve ce grand axe, puis l'axe suivant le plus long qui lui est perpendiculaire, et ainsi de suite. Garder les premiers axes conserve l'essentiel de la dispersion (variance) des données.

## Cadre

Centre les données : soustrais les moyennes des colonnes, de sorte que `X ∈ R^{N×d}` ait des colonnes de moyenne nulle. La **matrice de covariance**

$$
C=\frac{1}{N-1}X^\top X\in\mathbb{R}^{d\times d}
$$

est symétrique ; `C_{jk}` est la covariance des variables `j` et `k`, et `C_{jj}` la variance de la variable `j`.

## Dérivation : la meilleure direction est un vecteur propre

Projette les données sur une direction unitaire `w` (`‖w‖ = 1`) : les projections sont `Xw ∈ R^N`. Leur variance vaut

$$
\operatorname{Var}(Xw)=\frac{1}{N-1}\|Xw\|^2=w^\top C\,w .
$$

Maximise `wᵀCw` sous la contrainte `wᵀw = 1`. Utilise un **multiplicateur de Lagrange** `λ` (la méthode d'optimisation sous contrainte : à un optimum contraint, le gradient de l'objectif est parallèle au gradient de la contrainte). Le lagrangien est `ℒ = wᵀCw − λ(wᵀw − 1)`. Son gradient est

$$
\nabla_w\mathcal{L}=2Cw-2\lambda w=0\;\Longrightarrow\;\boxed{Cw=\lambda w}.
$$

Donc `w` est un **vecteur propre** de `C`, et la variance le long de celui-ci est `wᵀCw = λ wᵀw = λ`, la valeur propre. Pour maximiser la variance, prends le vecteur propre de **plus grande valeur propre**. Répéter pour la direction suivante perpendiculaire à la première donne le deuxième plus grand vecteur propre, et ainsi de suite. Comme `C` est symétrique, ses vecteurs propres sont orthogonaux et ses valeurs propres réelles et positives ou nulles.

**Variance expliquée.** La variance totale est la somme de la diagonale de `C`, qui est égale à la somme de toutes les valeurs propres. Garder `k` composantes en conserve la part `(λ₁+…+λ_k) / (λ₁+…+λ_d)`.

**Erreur de reconstruction.** Avec `W_k ∈ R^{d×k}` les `k` premiers vecteurs propres, les données compressées sont `Z = X W_k ∈ R^{N×k}` et la reconstruction est `X̂ = Z W_kᵀ`. L'erreur quadratique sommée sur tous les points vaut

$$
\|X-\hat X\|_F^2=(N-1)\,(\lambda_{k+1}+\dots+\lambda_d),
$$

exactement la variance que tu as jetée. L'ACP est la meilleure compression linéaire de rang `k` au sens des moindres carrés.

**Lien avec la SVD.** Si `X = UΣVᵀ` est la décomposition en valeurs singulières, alors `XᵀX = VΣ²Vᵀ` : les colonnes de `V` sont les directions principales et `λ_i = σ_i²/(N−1)`. Numériquement, la SVD de `X` est plus stable que de former `C`.

## Exemple détaillé

Points centrés `(2,2), (−2,−2), (1,−1), (−1,1)` (`N = 4`).

- `XᵀX = [[10,6],[6,10]]`, donc `C = XᵀX/3 = [[3.333, 2],[2, 3.333]]`.
- Décomposition en éléments propres de `[[10,6],[6,10]]` : valeurs propres `16` et `4`, vecteurs propres `(1,1)/√2` et `(1,−1)/√2`. Donc `λ₁ = 16/3 = 5.333`, `λ₂ = 4/3 = 1.333`.
- Variance expliquée de la première composante : `16/20 = 0.8`.
- Garde `k = 1` : les projections sont `Xw₁ = (2.83, −2.83, 0, 0)` ; erreur de reconstruction `= 4` (la valeur propre abandonnée de `XᵀX`), soit `(N−1)λ₂ = 3 · 1.333`. ✓

## Implémentation

```python
import numpy as np

def pca_fit(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    C = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)          # ascending, symmetric matrix
    order = np.argsort(vals)[::-1]
    return mean, vecs[:, order[:k]].T, vals[order]
```

## Formes

`X: (N, d)`, `C: (d, d)`, composantes `(k, d)` (les lignes sont des directions), projection `Z = Xc @ comps.T : (N, k)`, reconstruction `Z @ comps + mean : (N, d)`.

## Idées reçues

- **Oublier de centrer** fait pointer la première composante vers la moyenne plutôt que vers la direction de dispersion.
- **L'échelle compte.** L'ACP sur des variables brutes est dominée par celle qui a les plus grandes unités. Standardise d'abord, sauf si les unités sont comparables.
- **Les signes sont arbitraires** : `w` et `−w` sont la même direction ; ne compare pas les composantes par leur signe.
- **Une variance élevée n'est pas synonyme d'importance pour ta tâche.** L'ACP est non supervisée ; une direction de faible variance peut être celle qui sépare tes classes.
- **Elle est linéaire.** Les structures courbes (une spirale, un swiss roll) demandent des méthodes non linéaires.

## Points clés à retenir

- Les directions principales sont les vecteurs propres de la matrice de covariance ; les valeurs propres sont les variances le long de celles-ci.
- Garde les `k` premières : la variance conservée est la part de la somme des valeurs propres ; l'erreur de reconstruction est exactement la somme des valeurs propres abandonnées.
- Centre d'abord, standardise en général, et préfère la SVD pour la stabilité numérique.

## Explique-le toi-même

1. Dérive `Cw = λw` à partir de la maximisation de `wᵀCw` sous `‖w‖ = 1`.
2. Pourquoi l'erreur de reconstruction est-elle égale à la somme des valeurs propres écartées ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Ch. 10 (PDF p.323) ; Bishop, *PRML*, Sec. 12.1.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01, leçons 10-11.

## Critères de maîtrise

Tu sais dériver l'ACP avec un multiplicateur de Lagrange, l'implémenter à partir de `eigh` et de la SVD, et quantifier l'information perdue pour un `k` donné.
