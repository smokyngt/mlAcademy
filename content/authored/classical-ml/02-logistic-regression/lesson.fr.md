---
id: cml.02
title: "Régression logistique"
summary: "Transformer un modèle linéaire en probabilité, l'entraîner avec l'entropie croisée, et comprendre pourquoi sa frontière de décision est une droite."
minutes: 60
teaches: [ml.logreg]
requires: [ml.linreg, opt.gradient-descent, ml.loss]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 4.3 Probabilistic Discriminative Models, PDF p.223
  - math/math for ML.pdf (Sec. 8.3 maximum likelihood; Ch. 12 classification for the margin view, PDF p.257 and p.376)
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/03-logistic-regression/docs/en.md
---

## Où cela s'inscrit

Beaucoup de questions réelles ont une réponse oui/non : cet e-mail est-il un spam, ce client va-t-il partir, cette image médicale est-elle anormale ? Tu ne veux pas seulement une étiquette, tu veux une **probabilité** (pour pouvoir fixer le seuil qui convient au coût des erreurs). La régression logistique est le modèle le plus simple qui en fournit une. C'est aussi exactement un réseau de neurones sans couche cachée : la perte, le gradient et la boucle d'entraînement que tu as appris dans le cours d'apprentissage automatique sont les mêmes, donc cette leçon est une répétition sur un modèle assez petit pour tout voir.

## L'idée en mots simples

Calcule un score comme le fait la régression linéaire (une somme pondérée), puis écrase-le dans l'intervalle 0 à 1 avec la **sigmoïde** en forme de S, de sorte qu'un score très négatif signifie « presque sûrement non », zéro signifie « 50-50 » et un score très positif signifie « presque sûrement oui ». L'entraînement ajuste les poids pour que les probabilités prédites soient les *plus élevées possible pour ce qui s'est réellement produit*, c'est-à-dire le maximum de vraisemblance (leçon 8).

Comme la sigmoïde est monotone, « probabilité supérieure à un demi » équivaut à « score supérieur à zéro », et le score est linéaire en les entrées. La frontière entre les deux classes est donc une **droite** (un hyperplan en dimension supérieure). C'est à la fois la force (simple, difficile à surapprendre) et la limite (il ne peut pas séparer ce qui nécessite une frontière courbe).

## Modèle formel

Variables `x ∈ R^d` (avec un 1 en tête pour l'ordonnée à l'origine), poids `w ∈ R^d`, étiquettes `y ∈ {0,1}` :

$$
z=w^\top x,\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}=P(y=1\mid x).
$$

Pour `N` exemples, `z = Xw ∈ R^N` et `p = σ(z)`.

## Dérivation : perte et gradient

Vraisemblance d'une étiquette : `p^y (1−p)^{1−y}`. L'opposé de la log-vraisemblance moyenne (entropie croisée binaire) :

$$
L(w)=-\frac1N\sum_i\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big].
$$

D'après la leçon 8, la dérivée par rapport au score est `∂L/∂z_i = (p_i − y_i)/N`, et `z = Xw` donne `∂z_i/∂w_j = X_{ij}`, donc (règle de la couche linéaire de la leçon 6)

$$
\boxed{\nabla_wL=\frac1N X^\top (p-y)}\qquad(d\times N)(N)=(d).
$$

Compare avec la régression linéaire, `Xᵀ(Xw − y)` : c'est la *même* formule en remplaçant la prédiction par `p`. Contrairement aux moindres carrés, il n'y a pas de forme fermée (l'équation `Xᵀ(p − y) = 0` est non linéaire en `w`), donc on utilise la descente de gradient. La perte reste convexe (sa hessienne est `Xᵀ diag(p(1−p)) X ⪰ 0`), donc la descente de gradient avec un taux assez petit atteint le minimum global.

**Régularisation.** Ajoute `(λ/2)‖w‖²` (pas sur l'ordonnée à l'origine) : le gradient gagne `+ λw`. Cela compte davantage que pour la régression linéaire : si les classes sont parfaitement séparables, la vraisemblance continue de s'améliorer quand `‖w‖ → ∞` (probabilités poussées vers 0 et 1) et aucun optimum fini n'existe. La pénalité rend l'optimum fini.

## Exemple détaillé

Un exemple `x = (1, 2)` (ordonnée à l'origine et une variable), étiquette `y = 1`, poids `w = (−1, 0.5)`.

- `z = −1·1 + 0.5·2 = 0`, donc `p = σ(0) = 0.5`.
- Perte `= −ln 0.5 = 0.693`, la valeur « on ne sait rien » de la leçon 8.
- Gradient `= (p − y) x = (−0.5)(1, 2) = (−0.5, −1)`.
- Un pas avec `η = 1` donne `w = (−0.5, 1.5)` : maintenant `z = −0.5 + 3 = 2.5`, `p = 0.924`. Le modèle est devenu plus confiant dans la bonne réponse.

## Implémentation

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def loss(w, X, y):
    z = X @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)   # stable cross-entropy

def grad(w, X, y):
    return X.T @ (sigmoid(X @ w) - y) / len(y)
```

`log(1+e^z) − yz` est l'entropie croisée réécrite pour ne jamais prendre le log de 0.

## Formes

`X: (N, d)`, `w: (d)`, `z, p, y: (N)`, gradient `(d)`. Règle de décision : prédire la classe 1 quand `Xw > 0`, c'est-à-dire `p > 0.5`.

## Idées reçues

- **C'est un classifieur, pas une régression**, malgré le nom : la régression porte sur le log-odds `ln(p/(1−p)) = w·x`, qui est linéaire.
- **Les probabilités ne valent que ce que vaut le modèle.** Sans vérification de calibration, `0.9` ne signifie pas forcément « juste 90 % du temps ».
- **La précision au seuil 0.5 cache de l'information.** Change le seuil pour échanger faux positifs contre faux négatifs.
- **Les variables non mises à l'échelle** ralentissent fortement la descente de gradient (leçon 7 : mauvais conditionnement). Standardise les entrées.

## Points clés à retenir

- `p = σ(Xw)` ; la perte est l'entropie croisée ; le gradient est `Xᵀ(p − y)/N`.
- Pas de forme fermée, mais la perte est convexe : la descente de gradient trouve l'optimum global.
- La frontière de décision `w·x = 0` est linéaire.
- Sur des données séparables, les poids divergent sans régularisation.

## Explique-le toi-même

1. Pourquoi la frontière de la régression logistique est-elle une droite alors que le modèle est non linéaire (il contient une sigmoïde) ?
2. Pourquoi des données séparables font-elles croître indéfiniment les poids non régularisés ?

## Pour aller plus loin

- Bishop, *PRML*, Sec. 4.3.2-4.3.3 (PDF p.223).
- Bibliothèque source : `ai-engineering-from-scratch` phase 02, leçon 03.

## Critères de maîtrise

Tu sais dériver `Xᵀ(p − y)/N` à partir de la perte, implémenter une version stable, la vérifier numériquement, et expliquer ce que change la régularisation.
