---
id: cml.06
title: "Arbres de décision et forêts aléatoires"
summary: "Faire pousser un arbre par des coupures gloutonnes qui réduisent l'impureté, comprendre pourquoi il surapprend, puis moyenner de nombreux arbres décorrélés (bagging, forêts aléatoires) et lire gratuitement l'erreur hors-sac."
minutes: 75
teaches: [ml.trees, ml.bagging]
requires: [ml.generalization, prob.basics, info.entropy]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 14.2 Committees (PDF p.675-677): bootstrap aggregation, error of an averaged committee. READ."
  - "Bishop, PRML, Sec. 14.4 Tree-based Models (PDF p.683-686): CART, greedy growth, cost-complexity pruning, Gini and cross-entropy, weaknesses of trees. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/04-decision-trees/docs/en.md (split criteria, stopping rules, random forest, MDI vs permutation importance). READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/11-ensemble-methods/docs/en.md (bagging, 63.2 percent in-bag, out-of-bag samples). READ (bagging and boosting sections)."
---

## Où cela s'inscrit

La plupart des données d'entreprise sont des tableaux : clients, transactions, mesures de capteurs, dossiers de patients. Les colonnes ont des unités différentes, certaines sont des catégories, les relations sont irrégulières (« le risque saute dès que le revenu passe sous un seuil *et* que le compte est récent »). Les modèles linéaires (leçons 01-02) exigent que tu construises ces interactions à la main. Les arbres les découvrent : ils posent une suite de questions oui/non sur des colonnes isolées, et chaque réponse resserre le problème. Sur les données tabulaires, les arbres et les forêts (et leurs cousins boostés, leçon suivante) sont la référence solide par défaut et gagnent régulièrement des compétitions. Leur faiblesse est le thème de la leçon 03 : un arbre seul a un biais faible et une variance très élevée. En moyenner beaucoup est le remède, et c'est l'une des applications les plus nettes de l'idée biais-variance que tu rencontreras.

## L'idée en mots simples

Pense au jeu des « vingt questions ». Tu ne commences pas par demander « est-ce un golden retriever ? » ; tu commences par une question qui partage les possibilités aussi équitablement que possible (« est-ce vivant ? »), puis tu poses la meilleure question suivante *compte tenu de la réponse*. Un arbre de décision est cette stratégie, apprise à partir des données : au sommet, choisis l'unique question sur une variable (« âge < 30 ? ») qui rend les deux groupes obtenus aussi *purs* que possible, c'est-à-dire que chaque groupe soit surtout d'une seule classe. Puis répète à l'intérieur de chaque groupe, indépendamment, jusqu'à ce que les groupes soient purs ou trop petits. Pour prédire, fais descendre un nouvel exemple le long des questions et renvoie la classe majoritaire (ou la valeur moyenne, en régression) du groupe où il atterrit.

Deux choses en découlent. D'abord, un arbre découpe l'espace d'entrée en boîtes dont les côtés sont parallèles aux axes et prédit une constante dans chaque boîte. Ensuite, si on le laisse poser des questions sans fin, il finit avec une boîte par point d'entraînement : il a mémorisé les données, bruit compris. Change quelques lignes d'entraînement et les questions du sommet peuvent changer complètement, ce qui explique l'instabilité d'un arbre seul.

Une **forêt** exploite cette instabilité. Entraîne beaucoup d'arbres sur des versions aléatoires légèrement différentes des données et laisse-les voter. Leurs erreurs individuelles pointent dans des directions différentes et s'annulent en grande partie, comme un jury de témoins indépendants.

## Impureté : ce que « pur » veut dire

Soit un nœud contenant $n$ exemples et soit $p_k$ la fraction dans la classe $k$. Deux mesures standard du désordre (Bishop les écrit pour une région $R_\tau$ comme dans ses Éq. 14.32-14.33) :

$$
G=\sum_k p_k(1-p_k)=1-\sum_k p_k^2 \qquad\text{(Gini)}
$$

$$
H=-\sum_k p_k\log_2 p_k \qquad\text{(entropy, in bits)}
$$

Les deux valent $0$ pour un nœud pur et sont maximales quand les classes sont également mélangées ($G=0.5$, $H=1$ bit pour deux classes). Gini a une jolie lecture : c'est la probabilité qu'un exemple tiré au hasard dans le nœud soit mal étiqueté si on l'étiquette en tirant une classe selon la distribution de classes du nœud lui-même. Pourquoi ne pas simplement utiliser le taux d'erreur de classification ? Bishop souligne que Gini et l'entropie réagissent plus régulièrement aux variations des proportions du nœud et sont dérivables ; l'erreur de classification peut rester plate sur des coupures manifestement utiles. (Il note que le taux d'erreur de classification reste le choix habituel pour l'élagage.)

En régression, l'« impureté » est la somme des erreurs quadratiques autour de la moyenne du nœud.

## Dérivation : la coupure gloutonne

Une coupure est un couple (variable $j$, seuil $\theta$). Elle envoie $n_L$ exemples à gauche ($x_j\le\theta$) et $n_R$ à droite. Sa **diminution d'impureté** (gain d'information quand on utilise $H$) est

$$
\Delta = I(\text{parent})-\frac{n_L}{n}\,I(\text{left})-\frac{n_R}{n}\,I(\text{right}).
$$

Les poids $n_L/n$ et $n_R/n$ comptent : une coupure qui détache un seul exemple pur ne vaut presque rien, une coupure qui envoie la moitié des données dans un fils pur est excellente.

**Pourquoi la recherche exhaustive est peu coûteuse.** Pour la variable $j$, seuls les seuils entre *valeurs distinctes triées* consécutives changent la partition, donc il y a au plus $n-1$ candidats par variable : teste les milieux. Chercher parmi toutes les variables et tous les seuils donne la meilleure coupure unique au nœud. Trouver le meilleur *arbre entier* est infaisable combinatoirement (Bishop, p.664-665), donc on est glouton : on prend la meilleure coupure maintenant, on ne la revoit jamais, on recurse sur les deux fils. Le glouton peut rater une mauvaise première coupure qui en permet deux brillantes (le XOR est le cas classique), mais ça marche assez bien en pratique.

**Feuilles de régression.** Si la partition est fixée et que l'on minimise $\sum_n (t_n-y_{\tau(n)})^2$, annuler la dérivée par rapport à la constante $y_\tau$ de chaque feuille donne $y_\tau=\frac1{N_\tau}\sum_{x_n\in R_\tau}t_n$, la **moyenne de la feuille** (Bishop Éq. 14.29). Ainsi, en régression, une coupure est évaluée par l'erreur quadratique totale après avoir remplacé chaque côté par sa moyenne ; on choisit la coupure qui la réduit le plus.

**Arrêt et élagage.** Faire pousser l'arbre jusqu'à ce que chaque feuille soit pure donne une erreur d'entraînement nulle et une mauvaise généralisation. Bishop observe que s'arrêter dès que la meilleure coupure apporte peu de gain est myope, car une coupure d'apparence inutile peut en permettre une très utile en dessous. La recette courante est donc : faire pousser un grand arbre $T_0$ (limité seulement par une taille minimale de feuille), puis l'élaguer en minimisant le critère de **coût-complexité** (Éq. 14.31)

$$
C(T)=\sum_{\tau=1}^{|T|}Q_\tau(T)+\lambda|T|,
$$

où $Q_\tau$ est l'erreur de la feuille (erreur quadratique, Gini ou entropie), $|T|$ le nombre de feuilles et $\lambda\ge0$ est choisi par validation croisée. C'est la même structure que la régression ridge : ajustement aux données plus un prix de complexité. Des contrôles plus simples (pré-élagage) comme `max_depth`, `min_samples_leaf` jouent un rôle similaire de façon plus brute et plus rapide.

## D'un arbre à une forêt

**Pourquoi moyenner aide.** Supposons que $M$ modèles prédisent chacun $y_m(x)=h(x)+\epsilon_m(x)$, fonction vraie plus erreur. Si les erreurs sont de moyenne nulle et *non corrélées*, l'erreur de la moyenne vaut $1/M$ de l'erreur individuelle moyenne (Bishop Éq. 14.14). Les erreurs réelles sont corrélées parce que tous les modèles voient les mêmes données. Soit chacune de variance $\sigma^2$ et de corrélation deux à deux $\rho$. La variance de leur moyenne est

$$
\operatorname{Var}\Big(\frac1M\sum_m \epsilon_m\Big)=\frac{1}{M^2}\Big(M\sigma^2+M(M-1)\rho\sigma^2\Big)=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2 .
$$

Ajouter des arbres tue le second terme mais jamais le premier. Le levier est donc de rendre les arbres **moins corrélés**, pas seulement plus nombreux.

**Bagging** (bootstrap aggregation, Bishop Éq. 14.7) : tire $M$ échantillons bootstrap (taille $N$, avec remise) du jeu d'entraînement, ajuste un arbre sur chacun, moyenne (régression) ou fais voter (classification). Les arbres profonds ont un biais faible et une variance élevée, donc les moyenner est le bon compromis.

**Forêt aléatoire** : le bagging plus une seconde randomisation. À *chaque coupure*, seul un sous-ensemble aléatoire de $m$ variables est éligible (un défaut courant est d'environ $\sqrt d$ pour la classification, à peu près $d/3$ pour la régression). Sans cela, une variable dominante se retrouve à la racine de chaque arbre et les arbres sont fortement corrélés ($\rho$ grand). Restreindre le choix les décorrèle, ce qui diminue $\rho\sigma^2$ au prix de rendre chaque arbre légèrement moins bon (un peu plus de biais, un peu plus de variance dans chaque arbre).

**Erreur hors-sac (OOB).** Un échantillon bootstrap de taille $N$ rate un exemple donné avec probabilité $(1-\tfrac1N)^N\to e^{-1}\approx0.368$, donc chaque arbre ne voit qu'environ $63.2\%$ des exemples distincts. Pour l'exemple $i$, moyenne les prédictions des seuls arbres qui ne se sont *pas* entraînés dessus ; l'erreur de ces prédictions est une estimation honnête de l'erreur de généralisation sans jeu de validation séparé. C'est de la validation croisée offerte avec le bagging.

## Exemple détaillé

Dix points sur une droite, $x=1,\dots,10$, étiquettes $y=(0,0,1,0,0,1,1,1,1,1)$ (surtout `0` pour $x$ petit et `1` pour $x$ grand, avec un `1` égaré en $x=3$).

- Parent : 4 zéros et 6 uns, $G=1-(0.4^2+0.6^2)=0.48$, $H=0.971$ bits.
- Seuil $5.5$ : gauche $=\{0,0,1,0,0\}$ a $G=1-(0.8^2+0.2^2)=0.32$ ; droite $=\{1,1,1,1,1\}$ est pure. Gain $=0.48-\tfrac{5}{10}0.32-\tfrac{5}{10}0=0.32$. En termes d'entropie, le gain est de $0.610$ bit.
- Seuil $2.5$ (isolant les deux premiers zéros) : gain $0.18$ (Gini). Tous les autres seuils sont moins bons, donc le choix glouton est $\theta=5.5$.
- Bootstrap : pour $N=10$, la probabilité qu'un exemple soit raté est $0.9^{10}=0.349$, donc la fraction dans l'échantillon est $0.651$ ; pour $N=1000$ elle vaut $0.632$, ce qui correspond à $1-e^{-1}$.

L'exercice reproduit ces nombres.

## Implémentation

```python
import numpy as np

def gini(y):
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)

def best_split(X, y):
    n, d = X.shape
    best = (None, None, 0.0)                       # feature, threshold, gain
    parent = gini(y)
    for j in range(d):
        xs = np.unique(X[:, j])
        for t in (xs[:-1] + xs[1:]) / 2:           # midpoints
            L = X[:, j] <= t
            gain = parent - (L.sum() * gini(y[L]) + (~L).sum() * gini(y[~L])) / n
            if gain > best[2] + 1e-12:
                best = (j, t, gain)
    return best
```

Un arbre complet appelle `best_split`, s'arrête sur une limite de profondeur ou de taille de feuille ou sur un nœud pur, et sinon recurse sur les sous-ensembles gauche et droit. Une forêt est une boucle : `idx = rng.integers(0, n, n)` ; ajuste un arbre sur `X[idx]` ; retiens quelles lignes ont été laissées de côté pour l'estimation OOB.

## Formes

`X: (N, d)`, `y: (N,)` classes entières. Une coupure candidate produit le masque booléen `L: (N,)`. Un ensemble bagging de $B$ arbres contient `B` tableaux d'indices de forme `(N,)` ; la prédiction OOB pour toutes les lignes est `(N,)`, chacune moyennée sur environ $0.368B$ arbres.

## Idées reçues

- **« Les arbres n'ont pas besoin de réglage. »** Un arbre non élagué surapprend fortement ; la profondeur et la taille des feuilles sont les principaux boutons.
- **« Plus d'arbres signifie surapprentissage. »** Pour une forêt, plus d'arbres ne fait que réduire le terme de variance $\frac{1-\rho}{M}\sigma^2$ ; la précision plafonne. (C'est la profondeur des arbres individuels qui contrôle le surapprentissage.)
- **« L'importance des variables par impureté est la vérité. »** La diminution moyenne d'impureté favorise les variables ayant beaucoup de valeurs distinctes et se calcule sur les données d'entraînement ; l'importance par permutation sur des données mises de côté est plus fiable mais plus lente (leçon 04 du dépôt).
- **« Les arbres extrapolent. »** Un arbre de régression prédit une constante dans chaque boîte, donc au-delà de la plage d'entraînement il répète simplement la valeur du bord.
- **Limites des axes.** Une frontière de classe diagonale demande de nombreuses coupures en escalier (Bishop p.666), là où un modèle linéaire n'en demande qu'une.
- **Réserve sur l'interprétabilité.** Un petit arbre seul est lisible, mais Bishop note que la structure apprise est très sensible aux petits changements des données ; une forêt de 500 arbres n'est pas lisible.

## Points clés à retenir

- Un arbre choisit goulûment le (variable, seuil) à la plus grande diminution d'impureté pondérée ; les feuilles prédisent la classe majoritaire ou la moyenne.
- Gini et entropie se comportent de façon similaire ; la régression utilise l'erreur quadratique et les moyennes de feuilles.
- Fais pousser grand, puis élague avec $\sum Q_\tau+\lambda|T|$, ou limite la profondeur et la taille des feuilles.
- Moyenner $M$ modèles de variance $\sigma^2$ et de corrélation $\rho$ laisse une variance $\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$ : décorrèle les arbres.
- L'erreur OOB est une estimation de validation gratuite car environ 36.8 pour cent des lignes sont laissées de côté dans chaque bootstrap.

## Explique-le toi-même

1. Pourquoi la diminution d'impureté pondère-t-elle les fils par $n_L/n$ et $n_R/n$ ?
2. Un collègue dit « plus d'arbres dans une forêt finiront par surapprendre ». Réponds avec la formule de la variance.

## Pour aller plus loin

- Bishop, *PRML*, Sec. 14.2 et 14.4 (PDF p.675-686).
- Leçons du dépôt `phases/02-ml-fundamentals/04-decision-trees` et `11-ensemble-methods`.
- Suite : [[cml.07]] le boosting utilise les arbres à l'inverse, comme apprenants à fort biais corrigés l'un après l'autre.

## Critères de maîtrise

Tu sais calculer Gini et l'entropie à la main, trouver le meilleur seuil sur un jeu de données jouet, écrire un arbre récursif avec limites de profondeur et de taille de feuille, expliquer le bagging via la formule $\rho\sigma^2$, et calculer une erreur OOB à partir d'ensembles d'indices.
