---
id: cml.04
title: "Clustering par k-means"
summary: "Trouver des groupes dans des données non étiquetées en alternant deux étapes simples, et prouver que l'objectif ne peut que baisser."
minutes: 50
teaches: [ml.kmeans]
requires: [la.dot, alg.summation]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 9.1 K-means Clustering, PDF p.444
  - math/math for ML.pdf (Ch. 11 Density Estimation with Gaussian Mixture Models, for the soft-assignment generalisation, PDF p.354)
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/07-unsupervised-learning/docs/en.md
---

## Où cela s'inscrit

Jusqu'ici, chaque point de données venait avec une réponse (un prix, une étiquette). Souvent il n'y a pas de réponse : tu as les achats de 100 000 clients et tu veux savoir s'il existe des groupes naturels, ou tu veux compresser une image en 16 couleurs. C'est l'**apprentissage non supervisé** : trouver de la structure sans étiquettes. Le k-means est la méthode la plus simple et la plus utilisée. Il introduit aussi un schéma que tu retrouveras (dans les mélanges gaussiens et dans l'entraînement avec variables latentes) : *alterner entre deux étapes faciles, dont aucune ne peut dégrader l'objectif*.

## L'idée en mots simples

Tu veux ouvrir `k` pizzerias de sorte que chacun habite près de l'une d'elles. Commence par placer les pizzerias n'importe où. Puis répète deux étapes :

1. **Affecter :** chaque personne va à la pizzeria la plus proche.
2. **Déplacer :** chaque pizzeria déménage au centre (la position moyenne) des personnes qui l'ont choisie.

Déplacer une pizzeria à la moyenne de ses clients ne peut que réduire la distance totale parcourue, et réaffecter chaque personne à la pizzeria la plus proche ne peut que la réduire aussi. Le total ne monte donc jamais, et quand plus rien ne change, c'est terminé.

## Objectif formel

Données `x_1..x_N ∈ R^d`, `k` centres `μ_1..μ_k`, affectation `c(i) ∈ {1..k}`. L'objectif est la **somme des carrés intra-cluster** (inertie) :

$$
J=\sum_{i=1}^{N}\|x_i-\mu_{c(i)}\|^2 .
$$

## Dérivation : pourquoi chaque étape aide

**Étape d'affectation** (centres fixés). Chaque point contribue `‖x_i − μ_{c(i)}‖²` indépendamment, donc choisir `c(i)` comme le centre le plus proche minimise `J`.

**Étape de mise à jour** (affectations fixées). Pour un cluster `S`, minimise `f(μ) = Σ_{i∈S} ‖x_i − μ‖²`. Son gradient (leçon 5) est `∇f = −2 Σ_{i∈S}(x_i − μ)`. L'annuler donne

$$
\mu=\frac{1}{|S|}\sum_{i\in S}x_i ,
$$

la moyenne. `f` est une quadratique convexe, donc c'est le minimum.

Les deux étapes ne peuvent que diminuer ou conserver `J`, et `J ≥ 0`, donc la suite converge. Il n'y a qu'un nombre fini d'affectations possibles, et `J` baisse strictement dès qu'une affectation change, donc l'algorithme s'arrête après un nombre fini d'itérations. **Il s'arrête à un minimum local, pas nécessairement global.**

## Exemple détaillé (1-D)

Points `0, 1, 2, 10, 11, 12`, `k = 2`, centres initiaux `0` et `1`.

- Affecter : `0 → c₀` ; `1, 2, 10, 11, 12 → c₁` (|2−1| < |2−0|). Mettre à jour : `c₀ = 0`, `c₁ = (1+2+10+11+12)/5 = 7.2`.
- Affecter : `0, 1, 2 → c₀` (par ex. |2−0| = 2 < |2−7.2|) ; `10, 11, 12 → c₁`. Mettre à jour : `c₀ = 1`, `c₁ = 11`.
- Affecter à nouveau : rien ne change. Convergé avec `J = (1+0+1) + (1+0+1) = 4`.

## Questions pratiques

- **L'initialisation compte.** De mauvais départs donnent de mauvais minima locaux (deux centres dans un même vrai cluster). Remède : lancer plusieurs redémarrages et garder le `J` le plus bas ; ou utiliser k-means++, qui choisit des centres initiaux éloignés les uns des autres (chaque nouveau centre est tiré avec une probabilité proportionnelle à sa distance au carré au centre déjà choisi le plus proche).
- **Des clusters vides** peuvent apparaître : garde l'ancien centre (ou ré-initialise-le au point le plus éloigné).
- **Choisir `k`** : `J` baisse toujours quand `k` augmente, donc trace `J` en fonction de `k` et cherche un « coude », ou utilise un critère de validation ; il n'y a pas de réponse universellement correcte.
- **Hypothèses** : les clusters sont à peu près ronds et de taille similaire, et l'échelle des variables est comparable. Standardise d'abord les variables. Pour des formes allongées ou imbriquées, utilise d'autres méthodes.

## Implémentation

```python
import numpy as np

def assign(X, centers):
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (N, k)
    return d2.argmin(axis=1)

def update(X, labels, centers):
    new = centers.copy()
    for j in range(len(centers)):
        pts = X[labels == j]
        if len(pts):
            new[j] = pts.mean(axis=0)
    return new
```

## Formes

`X: (N, d)`, `centers: (k, d)`, distances au carré deux à deux `(N, k)` par broadcasting `(N,1,d) − (1,k,d)`, `labels: (N)`.

## Idées reçues

- **Il trouve toujours la même réponse.** Non ; cela dépend de l'initialisation.
- **Une inertie plus faible signifie de meilleurs clusters.** Seulement pour le même `k` et les mêmes données ; l'inertie baisse toujours avec plus de clusters.
- **Il marche pour n'importe quelles données.** Il utilise la distance euclidienne et des moyennes ; des données catégorielles ou très non sphériques demandent une autre approche.

## Points clés à retenir

- Alterner « affecter au centre le plus proche » et « déplacer le centre à la moyenne ».
- Chaque étape diminue (ou conserve) la somme des distances au carré ; il converge vers un minimum local.
- Relance plusieurs fois ; standardise les variables ; choisis `k` avec soin.

## Explique-le toi-même

1. Pourquoi la moyenne est-elle le meilleur centre pour un cluster fixé ? (Utilise le gradient.)
2. Pourquoi l'algorithme doit-il s'arrêter, et pourquoi le résultat n'est-il pas nécessairement l'optimum global ?

## Pour aller plus loin

- Bishop, *PRML*, Sec. 9.1 (PDF p.444) ; *Mathematics for Machine Learning*, Ch. 11 pour la version douce.
- Bibliothèque source : `ai-engineering-from-scratch` phase 02, leçon 07.

## Critères de maîtrise

Tu sais implémenter l'algorithme de Lloyd avec le broadcasting, montrer que l'objectif est monotone, et diagnostiquer un mauvais minimum local.
