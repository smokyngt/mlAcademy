---
id: cml.07
title: "Gradient boosting"
summary: "Construire un modèle puissant comme une somme de nombreux arbres faibles, chacun ajusté sur ce que la somme précédente prédit encore mal, et comprendre pourquoi c'est une descente de gradient dans l'espace des fonctions."
minutes: 80
teaches: [ml.boosting]
requires: [ml.trees, ml.bagging, opt.gradient-descent, ml.loss]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 14.3 Boosting incl. 14.3.1-14.3.2 (PDF p.677-683): AdaBoost algorithm, exponential error derivation, comparison of error functions, extension to squared error and residuals. READ."
  - "Bishop, PRML, Sec. 14.2 (PDF p.676-677): why plain averaging of correlated models helps little, motivating boosting. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/11-ensemble-methods/docs/en.md (boosting, AdaBoost, gradient boosting, shrinkage). READ (concept sections)."
---

## Où cela s'inscrit

Une forêt aléatoire ([[cml.06]]) lutte contre la *variance* : elle moyenne des arbres profonds à faible biais. Le boosting lutte contre le *biais* : il part d'arbres très simples et faibles et construit un modèle complexe en ajoutant des corrections. Les arbres à gradient boosté (XGBoost, LightGBM, CatBoost et les variantes de scikit-learn implémentent tous la même idée avec de l'ingénierie par-dessus) sont le gagnant habituel sur les problèmes tabulaires : prédiction de clics, scoring de crédit, prévision de la demande, classement. L'idée centrale est aussi une recette générale : optimiser une perte par **descente de gradient, mais où les paramètres mis à jour sont les prédictions du modèle** et où chaque pas est réalisé en entraînant un petit modèle. Une fois que tu vois cela, les pertes autres que l'erreur quadratique viennent gratuitement.

## L'idée en mots simples

Imagine que tu donnes des cours particuliers à un élève examen après examen. Après le premier examen, tu ne reprends pas tout depuis le début ; tu regardes les questions ratées et tu conçois la leçon suivante uniquement sur celles-ci. Après l'examen suivant, tu vises à nouveau ce qui est *encore* faux. Chaque leçon est simple et courte, mais la somme de nombreuses leçons ciblées est complète.

Dans le boosting, l'« élève » est la prédiction courante et chaque « leçon » est un petit arbre. En régression, « ce qui est encore faux » est littéralement le **résidu** : valeur vraie moins prédiction courante. L'arbre suivant est entraîné à prédire ces résidus, et sa sortie est ajoutée (un peu réduite) à la prédiction cumulée. En classification, AdaBoost exprime « encore faux » autrement : il augmente le *poids* des exemples mal classés pour que le classifieur faible suivant se concentre sur eux.

Contraste avec le bagging : le bagging entraîne des arbres indépendamment et en parallèle et les moyenne ; le boosting les entraîne l'un après l'autre, chacun dépendant des précédents, et les additionne.

## Modèles additifs par étapes

Le boosting construit

$$
F_M(x)=F_0+\sum_{m=1}^{M}\nu\, h_m(x),
$$

une somme d'apprenants de base $h_m$ (ici de petits arbres) avec un **taux d'apprentissage** (rétrécissement, shrinkage) $\nu\in(0,1]$. « Par étapes » signifie qu'à l'étape $m$ on fige $F_{m-1}$ et on ne choisit que le nouveau terme ; on ne revient jamais réajuster les termes précédents. Cela fait de chaque pas un petit problème d'ajustement abordable, même si le modèle final a des centaines de termes.

## AdaBoost à partir de la perte exponentielle

Bishop (Sec. 14.3, avec des cibles $t_n\in\{-1,+1\}$) montre qu'AdaBoost est exactement la minimisation par étapes de l'**erreur exponentielle** $E=\sum_n e^{-t_n f_m(x_n)}$ avec $f_m=\tfrac12\sum_{l\le m}\alpha_l y_l$. Fige les $m-1$ premiers termes et définis $w_n^{(m)}=e^{-t_nf_{m-1}(x_n)}$, constantes pour cette étape. Alors

$$
E=\sum_n w_n^{(m)}\exp\!\big(-\tfrac12 t_n\alpha_m y_m(x_n)\big).
$$

Sépare les points bien classés ($t_ny_m=+1$) et mal classés ($t_ny_m=-1$) :

$$
E=e^{-\alpha_m/2}\sum_{\text{correct}}w_n+e^{\alpha_m/2}\sum_{\text{wrong}}w_n .
$$

Soit $\epsilon_m$ le taux d'erreur pondéré (poids des points faux sur tous les poids) et divise par $\sum_n w_n$ :

$$
\frac{E}{\sum w}=e^{-\alpha/2}(1-\epsilon)+e^{\alpha/2}\epsilon .
$$

- Sur le classifieur $y_m$ : c'est minimal quand $\epsilon$ est minimal, donc ajuste l'apprenant faible par l'erreur de classification **pondérée**.
- Sur $\alpha$ : annule la dérivée, $-\tfrac12e^{-\alpha/2}(1-\epsilon)+\tfrac12e^{\alpha/2}\epsilon=0$, donc $e^{\alpha}=\frac{1-\epsilon}{\epsilon}$ :

$$
\alpha_m=\ln\frac{1-\epsilon_m}{\epsilon_m}.
$$

Les apprenants précis ($\epsilon$ proche de $0$) obtiennent un grand vote, et un apprenant pas meilleur que le hasard ($\epsilon=\tfrac12$) obtient $\alpha=0$. Mettre à jour les poids, $w^{(m+1)}_n=w^{(m)}_n e^{-\frac12t_n\alpha_my_m(x_n)}$, et retirer un facteur commun à tous les $n$ donne l'Éq. 14.18 de Bishop : **multiplie le poids de chaque point mal classé par $e^{\alpha_m}$**. Le classifieur final est $\operatorname{sign}\big(\sum_m\alpha_my_m(x)\big)$.

Limites (Bishop Sec. 14.3.2) : la perte exponentielle croît exponentiellement pour les points très mal classés, donc AdaBoost est bien moins robuste aux valeurs aberrantes et aux données mal étiquetées que l'entropie croisée, et il ne se généralise pas proprement à un modèle probabiliste multiclasse.

## Dérivation : gradient boosting avec perte quadratique

Passons à la régression, avec une perte générale. On veut minimiser $\mathcal L(F)=\sum_{i=1}^N \ell\big(y_i,F(x_i)\big)$. Traite les $N$ nombres $F_i=F(x_i)$ comme les paramètres. La descente de gradient ordinaire sur eux serait

$$
F_i\leftarrow F_i-\eta\,\frac{\partial\ell(y_i,F_i)}{\partial F_i}.
$$

On ne peut pas fixer librement les $F_i$, car il nous faut une fonction qui marche aussi sur de nouveaux points. On **ajuste donc un modèle $h$ sur le gradient négatif** $-g_i$, avec $g_i=\partial\ell/\partial F_i$, par moindres carrés, puis on avance le long de $h$ : $F\leftarrow F+\nu h$. C'est une descente de gradient dans l'espace des fonctions.

Pour la perte quadratique $\ell=\tfrac12(y-F)^2$, la dérivée est $g_i=-(y_i-F_i)$, donc le gradient négatif est exactement le résidu :

$$
r_i=y_i-F_{m-1}(x_i).
$$

**Pourquoi c'est aussi le meilleur nouveau terme possible.** À l'étape $m$ on choisit $h$ pour minimiser $\sum_i\big(y_i-F_{m-1}(x_i)-h(x_i)\big)^2=\sum_i (r_i-h(x_i))^2$ : c'est une *régression ordinaire de $h$ sur les résidus*. Ainsi, avec la perte quadratique, la vue par le gradient et la vue « ajuste ce qui reste » coïncident (Bishop fait la même remarque pour la somme des carrés : chaque nouvel apprenant est ajusté sur les résidus $t_n-f_{m-1}(x_n)$).

L'algorithme :

1. $F_0=\bar y$ (la constante qui minimise l'erreur quadratique).
2. Pour $m=1..M$ : calcule $r_i=y_i-F_{m-1}(x_i)$ ; ajuste un petit arbre de régression $h_m$ sur $(x_i,r_i)$ ; pose $F_m=F_{m-1}+\nu h_m$.

Pour d'autres pertes, seul le résidu change : pour l'erreur absolue $\ell=|y-F|$ le gradient négatif est $\operatorname{sign}(y-F)$, une alternative robuste (Bishop note que l'erreur quadratique n'est pas robuste aux valeurs aberrantes et que l'écart absolu en est un remède) ; pour la perte logistique c'est $y-\sigma(F)$, la même « erreur » que dans la régression logistique ([[cml.02]]). Dans l'algorithme complet, les valeurs des feuilles sont ensuite affinées par une recherche linéaire par feuille ; on s'en passe avec la perte quadratique puisque les moyennes de feuilles sont déjà optimales.

## Contrôler le surapprentissage

Contrairement à une forêt, un modèle boosté surapprend si l'on continue d'ajouter des tours. Les leviers :

- **Rétrécissement $\nu$** (par ex. 0.01-0.3) : chaque arbre ne déplace la prédiction que d'une partie du chemin. Un petit $\nu$ demande plus de tours mais généralise en général mieux ; c'est le régularisateur le plus important.
- **Nombre de tours $M$** : choisis-le par l'erreur de validation (arrêt précoce).
- **Profondeur des arbres** : la profondeur 1 (souches) donne un modèle additif en les variables ; la profondeur $d$ permet des interactions entre jusqu'à $d$ variables. Les profondeurs 3-6 sont typiques.
- **Taille des feuilles et sous-échantillonnage** des lignes ou des colonnes à chaque tour ajoutent d'autre aléa et de la régularisation (courant dans les bibliothèques modernes).

## Exemple détaillé

Six points, $x=1..6$, $y=(1,1.2,0.9,3,3.2,2.8)$, souches, $\nu=0.5$.

- $F_0=\bar y=2.0167$. Résidus : $(-1.017,-0.817,-1.117,0.983,1.183,0.783)$.
- Meilleure souche : coupure à $x\le3.5$, moyennes des feuilles $-0.983$ (gauche) et $+0.983$ (droite). Mise à jour : $F_1=2.0167+0.5(\mp0.983)=1.525$ (gauche) et $2.508$ (droite). La MSE d'entraînement passe de la variance de $y$ à $0.2628$.
- Tour 2 : les moyennes des résidus des deux côtés valent maintenant $\mp0.4917$ (divisées par deux, car $\nu=0.5$ nous a fait parcourir la moitié du chemin). Même coupure. $F_2=1.279$ / $2.754$, MSE $0.0815$.
- Tour 3 : $F_3=1.156$ / $2.877$, MSE $0.0362$.

La MSE ne peut pas descendre sous la variance intra-groupe, $0.1267/6=0.0211$ (une souche à une coupure prédit une constante par côté) ; ici le résidu restant n'est que du bruit à l'intérieur de chaque côté. Le rétrécissement apparaît comme une division géométrique par deux : après $m$ tours, le résidu au niveau de la souche vaut $(1-\nu)^m\cdot0.983$.

Un tour d'AdaBoost : avec cinq poids égaux de $0.2$ et une erreur, $\epsilon=0.2$, $\alpha=\ln4=1.386$ ; le poids de l'erreur devient $0.2\cdot4=0.8$, les autres restent à $0.2$ ; normalisés : $(0.125,0.125,0.5,0.125,0.125)$. Après repondération, le point mal classé porte exactement la moitié du poids total, donc l'apprenant faible suivant est forcé de le prendre au sérieux.

## Implémentation

```python
import numpy as np

def fit_stump(X, r):
    best = None
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        for t in (u[:-1] + u[1:]) / 2:
            L = X[:, j] <= t
            lv, rv = r[L].mean(), r[~L].mean()
            sse = ((r[L] - lv) ** 2).sum() + ((r[~L] - rv) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, j, t, lv, rv)
    return best[1:]

def gb_fit(X, y, rounds, lr):
    F = np.full(len(y), y.mean())
    stumps = []
    for _ in range(rounds):
        j, t, lv, rv = fit_stump(X, y - F)          # fit the residuals
        F = F + lr * np.where(X[:, j] <= t, lv, rv) # shrunken update
        stumps.append((j, t, lv, rv))
    return y.mean(), stumps
```

## Formes

`X: (N, d)`, `y, F, r: (N,)`. Chaque souche est quatre nombres ; le modèle est $M$ souches plus la constante. La prédiction coûte $O(M\cdot\text{depth})$ par point. L'entraînement est séquentiel en $M$ (contrairement au bagging, les tours ne peuvent pas être exécutés en parallèle).

## Idées reçues

- **« Le boosting repondère les données » ne vaut que pour AdaBoost.** Le gradient boosting ne repondère pas les lignes ; il change les *cibles* de régression en résidus (gradients négatifs).
- **« Plus de tours est toujours mieux. »** La perte d'entraînement n'augmente jamais avec les tours, mais la perte de validation finit par le faire ; utilise l'arrêt précoce.
- **« Ajuster les résidus » ne vaut que pour la perte quadratique.** En général on ajuste le gradient négatif, qui n'égale le résidu que pour l'erreur quadratique.
- **Les arbres profonds seraient de meilleurs apprenants de base.** Des arbres faibles et peu profonds sont tout l'intérêt ; des arbres très profonds font surapprendre chaque pas.
- **Boosting et bagging seraient interchangeables.** Le bagging réduit la variance d'apprenants à faible biais ; le boosting réduit le biais d'apprenants à fort biais (et peut augmenter la variance).
- **Étiquettes bruitées.** Comme le boosting se concentre sans cesse sur les cas difficiles, il peut poursuivre le bruit d'étiquetage, surtout avec la perte exponentielle.

## Points clés à retenir

- Le boosting construit $F_M=F_0+\nu\sum h_m$ par étapes ; chaque $h_m$ est un apprenant faible ajusté sur ce qui reste.
- AdaBoost = minimisation par étapes de l'erreur exponentielle : $\alpha=\ln\frac{1-\epsilon}{\epsilon}$, poids des points faux multipliés par $e^\alpha$.
- Gradient boosting = descente de gradient dans l'espace des fonctions : ajuste $h_m$ sur le gradient négatif ; pour la perte quadratique c'est le résidu.
- Le rétrécissement, la profondeur des arbres et le nombre de tours sont les régularisateurs ; règle-les avec des données de validation.

## Explique-le toi-même

1. Montre que pour $\ell=\tfrac12(y-F)^2$ le gradient négatif par rapport à $F_i$ est le résidu.
2. Pourquoi un petit taux d'apprentissage demande-t-il en général plus d'arbres mais généralise-t-il mieux ?

## Pour aller plus loin

- Bishop, *PRML*, Sec. 14.3 (PDF p.677-683).
- Leçon du dépôt `phases/02-ml-fundamentals/11-ensemble-methods` (couvre aussi le stacking et la régularisation de type XGBoost).
- Précédente : [[cml.06]] ; plus tard tu retrouveras la même structure « ajuste le gradient négatif » chaque fois qu'une perte est minimisée par mises à jour itératives.

## Critères de maîtrise

Tu sais dériver $\alpha_m$ à partir de la perte exponentielle, expliquer pourquoi les résidus sont des gradients négatifs pour la perte quadratique, calculer à la main quelques tours de boosting avec des souches, et nommer les trois principaux leviers de régularisation.
