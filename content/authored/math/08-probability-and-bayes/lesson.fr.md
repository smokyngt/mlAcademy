---
id: math.08
title: "Probabilités, conditionnement et théorème de Bayes"
summary: "Les deux règles auxquelles se réduisent toutes les probabilités, et comment le théorème de Bayes transforme des observations en croyances mises à jour."
minutes: 60
teaches: [prob.basics]
requires: [alg.summation]
sources:
  - "math/math for ML.pdf, Sec. 6.3 Sum Rule, Product Rule, and Bayes' Theorem (PDF p.189-192): sum rule (marginalisation), product rule, Bayes' theorem with prior / likelihood / posterior / evidence naming. READ."
  - "math/math for ML.pdf, Table 6.1 nomenclature (PDF p.189): pmf vs pdf vs cdf. READ."
---

## Où cela s'inscrit

Le machine learning est un raisonnement sous incertitude : les données sont bruitées, le modèle est imparfait, et l'avenir est inconnu. Les probabilités sont le langage pour dire à quel point nous sommes sûrs. Trois choses de ce cours s'appuient directement sur elles : les **fonctions de perte** de la leçon 8 sont des log-probabilités négatives ; les sorties de **classification** sont des probabilités (régression logistique, softmax) ; et le **théorème de Bayes** est la façon de mettre à jour une croyance après avoir vu des données, ce qui est le cœur du Bayes naïf, de l'inférence bayésienne et, sous un déguisement, de toute procédure « entraîner sur des données ». Le livre dit la même chose : une fois écrites les distributions décrivant tes données et ton problème, « il n'y a que deux règles fondamentales, la règle de la somme et la règle du produit », et tout le reste en découle.

## L'idée en mots simples

- Une probabilité est un nombre entre 0 et 1 qui dit à quel point quelque chose est plausible ; 1 est certain, 0 est impossible, et les probabilités de toutes les possibilités exclusives s'additionnent pour faire 1.
- Pense à **une grande foule de 1000 personnes**, chacune étiquetée avec deux propriétés (par exemple, a la maladie ou non, test positif ou non). Une probabilité est alors simplement une *fraction de la foule*. Une probabilité **conditionnelle** est la fraction *au sein d'une sous-foule* : « parmi les personnes dont le test est positif, quelle fraction a la maladie ? ».
- La **règle de la somme** dit : pour obtenir la probabilité d'une propriété indépendamment de l'autre, additionne sur l'autre propriété (marginalise). La **règle du produit** dit : la probabilité des deux est la probabilité de l'une fois la probabilité de l'autre *sachant* la première.
- Le **théorème de Bayes** est la règle du produit lue dans les deux sens. Il te dit comment inverser une conditionnelle : tu sais « si la personne est malade, le test est positif 90 % du temps » mais tu veux « si le test est positif, quelle est la probabilité d'être malade ? ». Ce sont des questions différentes, et les confondre est l'une des erreurs les plus courantes dans le raisonnement sur les données.

## Règles formelles

Pour des variables aléatoires `x`, `y` de loi conjointe `p(x, y)` :

$$
\textbf{Sum rule (marginalisation):}\quad p(x)=\sum_{y}p(x,y)\ \ \big(\text{or }\textstyle\int p(x,y)\,dy\big).
$$

$$
\textbf{Product rule:}\quad p(x,y)=p(y\mid x)\,p(x)=p(x\mid y)\,p(y).
$$

La probabilité **conditionnelle** en découle : `p(y | x) = p(x, y) / p(x)` pour `p(x) > 0`. Deux variables sont **indépendantes** si `p(x, y) = p(x) p(y)`, ce qui équivaut à `p(y | x) = p(y)` : apprendre `x` ne t'apprend rien sur `y`.

### Théorème de Bayes (démonstration)

Écris la règle du produit dans les deux sens, `p(x | y) p(y) = p(y | x) p(x)`, et divise par `p(y)` :

$$
\boxed{\underbrace{p(x\mid y)}_{\text{posterior}}=\frac{\overbrace{p(y\mid x)}^{\text{likelihood}}\ \overbrace{p(x)}^{\text{prior}}}{\underbrace{p(y)}_{\text{evidence}}}},\qquad p(y)=\sum_x p(y\mid x)\,p(x).
$$

En lisant les termes comme le fait le livre : l'**a priori** `p(x)` est ce que tu crois à propos de la quantité non observée `x` avant de voir des données ; la **vraisemblance** `p(y | x)` dit à quel point l'observation `y` est probable si `x` était vrai (remarque que c'est une distribution sur `y`, pas sur `x`) ; l'**évidence** `p(y)` ne fait que normaliser pour que l'a posteriori somme à 1 ; l'**a posteriori** `p(x | y)` est ce que tu sais de `x` après avoir vu `y`. L'a priori doit donner une probabilité non nulle à tout ce qui est plausible, car une hypothèse d'a priori 0 ne peut jamais être ranimée par aucune observation.

## Exemple traité : le test médical

Une maladie touche 1 % des personnes. Un test la détecte 90 % du temps quand elle est présente (sensibilité) et donne une fausse alerte 5 % du temps quand elle est absente. Quelqu'un est testé positif. Quelle est la probabilité qu'il ait la maladie ?

Prenons 10 000 personnes :

- 100 ont la maladie ; 90 d'entre elles sont testées positives.
- 9 900 ne l'ont pas ; 5 % d'entre elles, soit `495`, sont aussi testées positives.
- Positifs au total : `90 + 495 = 585`. Parmi eux, seuls 90 sont malades.

$$
P(D\mid +)=\frac{0.9\cdot0.01}{0.9\cdot0.01+0.05\cdot0.99}=\frac{0.009}{0.0585}\approx0.154 .
$$

Seulement environ **15 %**, bien que le test soit « fiable à 90 % ». La raison est l'a priori faible : la plupart des positifs viennent du gigantesque groupe des personnes saines. C'est le *biais de négligence du taux de base*, et l'image de la foule le rend évident.

**Mettre à jour deux fois.** Les a posteriori deviennent l'a priori suivant. Un deuxième test positif indépendant transforme `0.154` en `0.9·0.154 / (0.9·0.154 + 0.05·0.846) = 0.1386/(0.1386+0.0423) ≈ 0.766`. Faire les deux mises à jour d'un coup, avec les vraisemblances `0.9²` contre `0.05²`, donne le même `0.766` : l'ordre n'a pas d'importance.

## Implémentation

```python
import numpy as np

def posterior(prior, likelihood):
    """prior[h] = P(h); likelihood[h] = P(data | h). Returns P(h | data)."""
    joint = prior * likelihood           # product rule
    return joint / joint.sum()           # divide by the evidence (sum rule)

print(posterior(np.array([0.01, 0.99]), np.array([0.9, 0.05])))   # [0.1538, 0.8462]
```

## Formes

Pour une loi conjointe discrète sur `x` (n états) et `y` (m états), stocke `P` avec la forme `(n, m)`, coefficients `≥ 0`, total `1`. Alors `p(x) = P.sum(axis=1)` de forme `(n,)`, `p(y) = P.sum(axis=0)` de forme `(m,)`, `p(y | x) = P / P.sum(axis=1, keepdims=True)` (chaque ligne somme à 1).

## Idées fausses courantes

- **`P(A | B) = P(B | A)`.** Non ; elles diffèrent du rapport des a priori (Bayes). « La plupart des positifs sont faux » et « le test est fiable » peuvent être vrais en même temps.
- **Indépendance contre exclusivité.** Des événements mutuellement exclusifs (qui ne peuvent pas se produire ensemble) sont fortement *dépendants* : savoir que l'un s'est produit te dit que l'autre non.
- **Une densité n'est pas une probabilité.** Pour une variable continue, `p(x)` peut dépasser 1 ; seules ses intégrales sur des intervalles sont des probabilités (le livre le note explicitement).
- **La vraisemblance n'est pas une distribution sur l'hypothèse.** Elle somme à 1 sur les données, pas sur `x`.
- **Les a priori ne disparaissent pas.** Avec peu de données, l'a priori domine ; avec beaucoup de données, il est submergé par la vraisemblance.

## Points clés à retenir

- Deux règles : la somme (marginaliser) et le produit (chaîner). Tout le reste en est dérivé.
- Bayes : `a posteriori ∝ vraisemblance × a priori`, normalisé par l'évidence.
- Pense en foules de personnes : une probabilité conditionnelle est une fraction d'une sous-foule.
- La mise à jour bayésienne séquentielle équivaut à une mise à jour en bloc.

## Explique-le toi-même

1. Un ami dit « le test est fiable à 90 %, donc un test positif veut dire 90 % de chances d'être malade ». Explique, avec la foule, pourquoi c'est faux dans cet exemple.
2. Pourquoi un a priori ne doit-il jamais être exactement 0 pour une hypothèse simplement improbable ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 6.3 (PDF p.189-192) ; Bishop, *PRML*, Sec. 1.2.

## Critères de maîtrise

Étant donné n'importe quel petit tableau de contingence, tu sais calculer les marginales, les conditionnelles et l'a posteriori de Bayes, et expliquer le biais de négligence du taux de base.
