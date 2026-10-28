---
id: cml.03
title: "Généralisation, biais-variance et validation croisée"
summary: "Pourquoi un modèle qui ajuste parfaitement les données d'entraînement peut être inutile, comment le mesurer honnêtement, et comment choisir la complexité du modèle."
minutes: 65
teaches: [ml.generalization]
requires: [ml.linreg, ml.ridge]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 1.1 Polynomial Curve Fitting, Sec. 3.2 The Bias-Variance Decomposition, PDF p.24 and p.167
  - math/math for ML.pdf (Ch. 8 When Models Meet Data: empirical risk minimisation, cross-validation, PDF p.257)
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/10-bias-variance/docs/en.md
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/09-model-evaluation/docs/en.md
---

## Où cela s'inscrit

Tout ce qu'on a vu jusqu'ici minimisait une erreur **sur les données que l'on avait**. Mais un modèle n'est utile que sur des données qu'il *n'a pas vues* : les clients de demain, de nouvelles images, de nouvelles phrases. Un modèle assez flexible pour mémoriser son jeu d'entraînement a une erreur d'entraînement nulle et peut pourtant être désastreux en pratique. Cette leçon apprend à distinguer les deux cas, et c'est la discipline la plus importante de l'apprentissage automatique appliqué : tout résultat honnête que tu publieras un jour a été mesuré sur des données que le modèle n'a jamais utilisées pour s'entraîner.

## L'idée en mots simples

Imagine un élève qui prépare un examen. Un élève apprend la *méthode* derrière les exercices (bon ajustement à l'idée, bon sur de nouvelles questions). Un autre mémorise mot pour mot les réponses de l'examen de l'an dernier (parfait sur le sujet de l'an dernier, perdu sur celui de cette année). Un élève qui a à peine étudié échoue aux deux. Le mémoriseur fait du **surapprentissage**, celui qui a à peine étudié fait du **sous-apprentissage**, et la seule façon de les distinguer est de leur donner un *nouvel* examen.

En termes de modèle : un modèle trop simple se trompe de façon systématique (**biais** élevé) ; un modèle trop flexible change beaucoup selon les exemples d'entraînement qu'il a rencontrés (**variance** élevée). Les bons modèles équilibrent les deux.

## Le protocole de mesure

Découpe les données en trois parties et respecte-les :

- **Jeu d'entraînement** : sert à ajuster les poids.
- **Jeu de validation** : sert à choisir les réglages (degré du polynôme, `λ`, taux d'apprentissage...). Comme tu choisis d'après lui, il cesse peu à peu d'être indépendant.
- **Jeu de test** : touché **une seule fois**, tout à la fin, pour rapporter le chiffre final.

Quand les données sont rares, utilise la **validation croisée à k blocs** : découpe les données en `k` blocs ; pour chaque bloc, entraîne sur les `k−1` autres et mesure sur lui ; fais la moyenne des `k` erreurs. Chaque point sert exactement une fois à la validation.

## Une expérience réelle

Cibles issues de `f(x) = x³ − x` plus un bruit d'écart-type 0.1 (variance 0.01), 30 points d'entraînement, 500 nouveaux points de test, ajustements polynomiaux par moindres carrés :

| Degré | MSE entraînement | MSE test | Lecture |
|---|---|---|---|
| 1 | 0.042 | 0.036 | trop simple : les deux erreurs sont élevées (**sous-apprentissage**) |
| 3 | 0.0081 | 0.0103 | bonne complexité : erreur de test ≈ plancher de bruit 0.01 |
| 9 | 0.0045 | 0.0153 | l'entraînement baisse, le test monte : **surapprentissage** |
| 15 | 0.0035 | 0.0173 | encore mieux à l'entraînement, pire au test |

L'erreur d'entraînement *diminue toujours* avec la complexité ; l'erreur de test a une forme en U. La variance du bruit `σ² = 0.01` est un plancher que rien ne peut battre.

## Dérivation : la décomposition biais-variance

Fixe un point de test `x`. La vérité est `y = f(x) + ε`, avec `E[ε] = 0`, `Var(ε) = σ²`. Notre prédicteur appris `f̂(x)` dépend du jeu d'entraînement aléatoire. Note `f̄ = E[f̂(x)]` (moyenne sur les jeux d'entraînement). Ajoute et retranche :

$$
y-\hat f=(f-\bar f)+(\bar f-\hat f)+\varepsilon .
$$

Élève au carré et prends l'espérance. Les termes croisés s'annulent : `ε` est indépendant de `f̂` et de moyenne nulle, et `E[\bar f - \hat f]=0`. Il reste :

$$
\boxed{E\big[(y-\hat f)^2\big]=\underbrace{(f-\bar f)^2}_{\text{bias}^2}+\underbrace{E\big[(\hat f-\bar f)^2\big]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{irreducible noise}}.}
$$

Les modèles plus flexibles diminuent le biais et augmentent la variance ; la régularisation (ridge, leçon 1) fait l'inverse. La meilleure erreur de test se situe à un compromis.

## Choisir les réglages

Choisis le degré, `λ`, ou tout hyperparamètre ayant la **plus faible erreur de validation croisée**, puis réentraîne sur toutes les données d'entraînement. Dans l'expérience ci-dessus, une validation croisée à 5 blocs sur 60 points donne des erreurs de 0.031 (degré 1), 0.037 (2), 0.0070 (3), 0.0066 (4), puis à peu près constantes autour de 0.007 à 0.008 : la grosse chute au degré 3 est le signal ; le meilleur degré exact entre 3 et 6 est dans le bruit. Une règle courante consiste à choisir le modèle *le plus simple* à moins d'un écart-type de l'erreur du meilleur.

## Implémentation

```python
import numpy as np

def kfold_indices(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)

def cv_mse(x, y, degree, lam, k=5, seed=0):
    errs = []
    for val in kfold_indices(len(x), k, seed):
        tr = np.setdiff1d(np.arange(len(x)), val)
        A = np.vander(x[tr], degree + 1, increasing=True)
        w = np.linalg.solve(A.T @ A + lam * np.eye(degree + 1), A.T @ y[tr])
        errs.append(np.mean((np.vander(x[val], degree + 1, increasing=True) @ w - y[val]) ** 2))
    return np.mean(errs)
```

## Idées reçues

- **L'erreur d'entraînement n'est pas une estimation de la performance.** Elle est optimiste, systématiquement.
- **Fuite de données.** Calculer des statistiques de normalisation, sélectionner des variables ou régler des paramètres sur tout le jeu de données avant de le découper contamine la validation. Fais chaque étape dépendante des données *à l'intérieur* de chaque bloc.
- **Réutiliser le jeu de test** pour départager des idées en fait un jeu de validation. Garde de côté des données que tu ne regardes vraiment jamais.
- **Les découpages aléatoires ne sont pas toujours adaptés** : pour les séries temporelles, découpe selon le temps ; pour les données groupées (plusieurs lignes par patient), découpe par groupe.
- **Plus de données réduit la variance, pas le biais.** Si les erreurs d'entraînement et de validation sont toutes deux élevées, collecter plus de données n'aidera pas ; utilise un modèle plus riche.

## Points clés à retenir

- Ne juge un modèle que sur des données avec lesquelles il ne s'est pas entraîné.
- Erreur de test = biais² + variance + bruit ; la complexité échange les deux premiers.
- L'erreur d'entraînement baisse toujours avec la complexité ; l'erreur de validation a une forme en U.
- Choisis les hyperparamètres par validation croisée ; rapporte une seule fois sur un jeu de test mis de côté.

## Explique-le toi-même

1. Les erreurs d'entraînement et de validation sont toutes deux élevées. Que cela t'indique-t-il, et que changerais-tu ?
2. L'erreur d'entraînement est minuscule mais l'erreur de validation est grande. Même question.

## Pour aller plus loin

- Bishop, *PRML*, Sec. 1.1 et 3.2 ; *Mathematics for Machine Learning*, Ch. 8.
- Bibliothèque source : `ai-engineering-from-scratch` phase 02, leçons 09-10.

## Critères de maîtrise

Tu sais dériver la décomposition, lire une courbe entraînement/validation et diagnostiquer un sous- ou un surapprentissage, et implémenter la validation croisée à k blocs sans fuite de données.
