---
id: cml.01
title: "Régression linéaire et équations normales"
summary: "Ajuster une droite (ou un hyperplan) aux données en une ligne d'algèbre linéaire, comprendre pourquoi cela fonctionne géométriquement, et la dompter avec la régularisation ridge."
minutes: 60
teaches: [ml.linreg, ml.ridge]
requires: [la.matrix, calc.gradient, ml.loss]
sources:
  - math/math for ML.pdf (Ch. 9 Linear Regression, PDF p.295; Ch. 3 orthogonal projections, PDF p.76)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 3.1 Linear Basis Function Models, PDF p.158
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/02-linear-regression/docs/en.md
---

## Où cela s'inscrit

Prédire un nombre à partir d'autres nombres (un prix à partir d'une surface et d'un quartier, un retard à partir du trafic) est la plus ancienne tâche de l'apprentissage automatique, et le modèle le plus simple pour cela, une somme pondérée des entrées, reste la première chose à essayer. C'est aussi le seul modèle qui possède une réponse exacte : pas d'itération, pas de taux d'apprentissage, juste de l'algèbre linéaire. L'étudier soigneusement te donne le gabarit de tout le reste : un modèle, une perte, un gradient, et un moyen de vérifier que tu as juste.

## L'idée en mots simples

Tu as un nuage de points et tu veux la droite qui passe « le plus près possible » de tous. Mesure la proximité par les **écarts verticaux** entre les points et la droite, élève-les au carré (pour que les écarts au-dessus et en dessous ne s'annulent pas, et que les grands écarts coûtent plus cher), puis additionne-les. La meilleure droite est celle dont le total est le plus petit.

Il y a une belle image derrière la formule : le vecteur de toutes tes cibles vit dans un espace avec un axe par point de données. L'ensemble de tout ce que ton modèle *peut* prédire est une nappe plane (un plan, ou un espace de dimension supérieure) à l'intérieur de cet espace. Le meilleur ajustement est simplement **l'ombre des cibles sur cette nappe**, et l'erreur est la partie qui dépasse à angle droit. Un angle droit, c'est un produit scalaire égal à zéro, et ce seul fait produit toute la solution.

## Cadre formel

Données : une matrice `X ∈ R^{N×d}` (les lignes sont les exemples ; ajoute une colonne de uns pour obtenir une ordonnée à l'origine) et des cibles `y ∈ R^N`. Modèle `ŷ = Xw`, avec `w ∈ R^d`. Perte (somme des carrés, qui a le même minimiseur que la moyenne) :

$$
L(w)=\tfrac12\|Xw-y\|^2 .
$$

## Dérivation 1 : les équations normales

D'après la leçon 5, le gradient est `∇L = Xᵀ(Xw − y)`. Au minimum il est nul :

$$
X^\top(Xw-y)=0\quad\Longleftrightarrow\quad \boxed{X^\top X\,w = X^\top y}\qquad(d\times d)(d)=(d).
$$

Si `XᵀX` est inversible, `w = (XᵀX)⁻¹ Xᵀ y`. Pourquoi est-ce un *minimum* et non un maximum ? La hessienne de `L` est `XᵀX`, et `vᵀXᵀXv = ‖Xv‖² ≥ 0` pour tout `v`, donc la perte est une cuvette convexe et l'unique point stationnaire est le minimum global.

**Lecture géométrique.** Avec le résidu `r = y − Xw`, l'équation `Xᵀ r = 0` dit : *le résidu est orthogonal à chaque colonne de `X`*. Ainsi `Xw` est exactement la projection orthogonale de `y` sur l'espace engendré par les colonnes de `X` (la projection de la leçon 2, avec de nombreuses directions à la fois).

## Exemple détaillé

Points (0, 1), (1, 2), (2, 2). Avec une ordonnée à l'origine, `X = [[1,0],[1,1],[1,2]]`, `y = (1,2,2)`.

- `XᵀX = [[3,3],[3,5]]`, `Xᵀy = (5, 6)`.
- `det = 15 − 9 = 6`, donc `(XᵀX)⁻¹ = (1/6)[[5,−3],[−3,3]]`.
- `w = (1/6)(25 − 18, −15 + 18) = (7/6, 1/2) ≈ (1.167, 0.5)` : la droite `y = 1.167 + 0.5x`.
- Prédictions `(1.167, 1.667, 2.167)` ; résidus `(−0.167, 0.333, −0.167)`.
- Vérifie l'orthogonalité : `1·(−0.167)+1·0.333+1·(−0.167) = 0` ✓ et `0·(−0.167)+1·0.333+2·(−0.167) = 0` ✓.

## Dérivation 2 : la régression ridge

Quand deux colonnes sont presque des copies l'une de l'autre, `XᵀX` est presque singulière et la solution varie violemment pour de minuscules changements dans les données. **Ridge** ajoute une pénalité sur la taille des poids :

$$
L_\lambda(w)=\tfrac12\|Xw-y\|^2+\tfrac{\lambda}{2}\|w\|^2\;\Rightarrow\;\nabla L_\lambda=X^\top(Xw-y)+\lambda w=0\;\Rightarrow\;\boxed{(X^\top X+\lambda I)\,w=X^\top y}.
$$

Ajouter `λI` relève chaque valeur propre de `XᵀX` de `λ`, donc la matrice est toujours inversible pour `λ > 0` et bien conditionnée (leçon 7 : le conditionnement détermine la difficulté du problème). Le prix à payer est un léger biais : les poids sont rétrécis vers zéro. Ne pénalise pas l'ordonnée à l'origine (centre les données, ou exclus cette colonne).

## Implémentation

```python
import numpy as np

def fit_ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)      # solve, never invert

def fit_ridge(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
```

Utilise `solve` (ou `lstsq`) plutôt que de calculer `inv(XᵀX)` : c'est plus rapide et numériquement plus sûr. Pour les problèmes très mal conditionnés, préfère `np.linalg.lstsq`, qui utilise une SVD.

## Formes

| Quantité | Forme |
|---|---|
| `X` | `(N, d)` |
| `XᵀX`, `XᵀX + λI` | `(d, d)` |
| `Xᵀy`, `w` | `(d,)` |
| résidu `r = y − Xw` | `(N,)`, et `Xᵀ r = 0` a pour forme `(d,)` |

## Idées reçues

- **« Linéaire » signifie linéaire en les poids, pas en les entrées.** Ajouter les colonnes `x, x², x³` donne une courbe et reste de la régression linéaire.
- **Inversibilité.** Plus de colonnes que de lignes (`d > N`) ou des colonnes dupliquées rendent `XᵀX` singulière ; ridge règle cela.
- **Un R² élevé ne prouve pas un bon modèle** : il augmente toujours quand tu ajoutes des colonnes. La leçon 3 explique comment détecter le surapprentissage.
- **Valeurs aberrantes** : l'erreur quadratique est dominée par les plus grandes erreurs. Un seul point extrême peut entraîner toute la droite.

## Points clés à retenir

- Les meilleurs poids satisfont `XᵀX w = Xᵀy` parce que le résidu est orthogonal aux colonnes de `X`.
- La perte est une cuvette convexe, donc le point stationnaire est le minimum global.
- Ridge ajoute `λI` : toujours résoluble, plus stable, légèrement biaisé.
- Résous un système linéaire ; n'inverse pas une matrice.

## Explique-le toi-même

1. Pourquoi « le résidu est orthogonal à chaque colonne de X » implique-t-il les équations normales ?
2. Que fait ridge aux valeurs propres de `XᵀX`, et pourquoi cela rend-il la solution plus stable ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Ch. 9 (PDF p.295) et Sec. 3.8 sur la projection orthogonale.
- Bishop, *PRML*, Sec. 3.1.
- Bibliothèque source : `ai-engineering-from-scratch` phase 02, leçon 02.

## Critères de maîtrise

Tu sais dériver les équations normales et la solution ridge à partir de la perte, énoncer leur signification géométrique, et implémenter les deux avec les bonnes formes.
