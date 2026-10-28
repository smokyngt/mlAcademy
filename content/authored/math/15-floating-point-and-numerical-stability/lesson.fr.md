---
id: math.15
title: "Virgule flottante et stabilité numérique"
summary: "Pourquoi les nombres réels d'un ordinateur ne sont que des approximations, quelles formules cassent à cause de cela, et les réécritures standard (log-sum-exp, Welford, sommes compensées) qui les réparent."
minutes: 75
teaches: [num.floatingpoint, num.stability]
requires: [alg.exp-log, calc.numerical-diff, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md: IEEE 754 layout and format table (float64/float32/float16/bfloat16), 0.1+0.2, catastrophic cancellation, overflow/underflow limits of exp and log, log-sum-exp, mixed precision and loss scaling, bfloat16 vs float16. READ."
  - "math/math for ML.pdf, Sec. 6.4.3 (PDF p.199, book p.193): the raw-score formula V[x]=E[x^2]-(E[x])^2, computable in one pass but numerically unstable when the two terms are huge and nearly equal; pairwise-difference identity (6.45). READ."
  - "academy/content/authored/math/01-functions-and-notation/lesson.md (frontmatter and opening): the exponential/log notation and why ML works in log space; cross-referenced. READ."
  - "academy/content/authored/math/04-derivatives/lesson.md (frontmatter and opening): finite differences as a numerical derivative; cross-referenced. READ."
---

## Où cela s'inscrit

Ton entraînement affiche `loss = nan` à l'étape 40 000. Ou un softmax renvoie `[nan, nan, nan]` dès qu'un logit atteint 1000. Ou une variance sort *négative*. Aucun de ces cas n'est une erreur des mathématiques : chacun est la mathématique exécutée sur une machine qui stocke les nombres réels avec un nombre fixe et fini de chiffres significatifs. L'algèbre dit que deux formules sont égales ; la virgule flottante dit que l'une fonctionne et que l'autre explose.

Cette leçon explique comment les nombres réels sont stockés, d'où viennent les erreurs, et les quelques réécritures (l'**astuce log-sum-exp**, l'**algorithme de Welford**, la **sommation compensée**) sur lesquelles repose toute base de code ML sérieuse. Elle se relie en arrière à la leçon 1 (on calcule en espace logarithmique parce que des probabilités se multiplient en nombres minuscules), à la leçon 4 (les dérivées par différences finies ont une taille de pas optimale, que nous pouvons maintenant expliquer), et à la leçon 13 (le texte de Deisenroth avertit que la variance par scores bruts est instable ; nous mesurons à quel point). Elle explique aussi les choix du matériel moderne : pourquoi les grands modèles s'entraînent en bfloat16, et ce qu'est la mise à l'échelle de la perte (loss scaling).

## L'idée en mots simples

- **Notation scientifique avec un budget.** Un flottant, c'est comme écrire un nombre en notation scientifique mais avec un nombre fixe de chiffres pour la partie de tête (la *mantisse* ou *significande*) et une plage fixe pour la puissance de dix (l'*exposant*). Un float64 a environ 16 chiffres significatifs, un float32 environ 7, un float16 environ 3. Les nombres n'ont pas un espacement uniforme : près de 1 la grille est fine, près de $10^{15}$ les flottants voisins sont espacés de 0.125 ou plus. Comme une règle dont les graduations s'écartent d'autant plus que le nombre mesuré est grand.
- **La plupart des décimaux ne peuvent pas être stockés.** En base 2, $0.1$ est une fraction périodique (tout comme $1/3$ se répète en base 10), donc l'ordinateur stocke le voisin représentable le plus proche. C'est pourquoi `0.1 + 0.2 == 0.3` est faux.
- **Soustraire des nombres presque égaux détruit des chiffres.** Si tu connais deux longueurs à 7 chiffres, $1.234567$ et $1.234561$, leur différence $0.000006$ n'a plus qu'un chiffre correct. Les chiffres de tête s'annulent et ce qui reste est surtout le bruit d'arrondi antérieur promu en tête. C'est l'**annulation catastrophique**.
- **Deux falaises : trop grand et trop petit.** Au-delà du plus grand nombre représentable, tu tombes dans `inf` ; en dessous du plus petit, tu tombes à `0`. Puis `inf/inf` et `0*inf` donnent `nan`, qui empoisonne tout ce qu'il touche.
- **Le remède est la réorganisation, pas plus de précision.** Soustrais le maximum avant d'exponentier. Travaille avec les écarts à la moyenne plutôt qu'avec des carrés de grands nombres. Additionne les petits nombres entre eux avant de les ajouter à un grand. Les mathématiques restent les mêmes ; seul l'ordre des opérations change.

## Comment les flottants sont stockés

Un flottant binaire IEEE-754 a un bit de signe, un champ d'exposant et un champ de mantisse. Pour les nombres normaux

$$
x=(-1)^{s}\times 2^{\,e-\text{bias}}\times(1.m)_2 .
$$

La mantisse donne 1 bit de tête implicite plus ses bits stockés, donc l'espacement *relatif* entre flottants voisins est compris entre $2^{-p}$ et $2^{-p+1}$ où $p$ est le nombre de bits de mantisse stockés.

| format | bits | exposant | mantisse | epsilon machine | plus grand fini | plus petit normal |
|---|---|---|---|---|---|---|
| float64 | 64 | 11 | 52 | $2.2\times10^{-16}$ | $1.8\times10^{308}$ | $2.2\times10^{-308}$ |
| float32 | 32 | 8 | 23 | $1.19\times10^{-7}$ | $3.40\times10^{38}$ | $1.18\times10^{-38}$ |
| float16 | 16 | 5 | 10 | $9.77\times10^{-4}$ | $65504$ | $6.1\times10^{-5}$ |
| bfloat16 | 16 | 8 | 7 | $7.8\times10^{-3}$ ($2^{-7}$) | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |

(Les trois premières lignes des colonnes numériques sont affichées par `np.finfo` ; bfloat16 n'est pas un type NumPy, et ses valeurs découlent de son exposant sur 8 bits comme float32 et de ses 7 bits de mantisse, comme l'énonce la leçon du dépôt.)

L'**epsilon machine** $\varepsilon$ est l'écart entre 1 et le flottant suivant ($2^{-52}$ en float64). Il borne l'erreur d'arrondi relative d'un seul nombre stocké ou d'une seule opération arithmétique : $\text{fl}(x)=x(1+\delta)$ avec $|\delta|\le\varepsilon/2$ pour l'arrondi au plus proche. Concrètement `1 + 1e-16 == 1` vaut `True` en float64, tandis que `1 + 2e-16 == 1` vaut `False`.

**Quel format et quand ?** float16 a plus de précision mais une plage minuscule (max 65 504) : logits et activations débordent. bfloat16 est « un float32 dont on a supprimé les 16 bits de mantisse du bas » : la même plage, une précision bien plus grossière. Pour l'entraînement, la plage compte plus que la précision, donc bfloat16 est le choix par défaut sur les TPU et les GPU récents. Avec float16, on utilise la **mise à l'échelle de la perte** : multiplier la perte par ex. par 1024 avant la passe arrière pour que les petits gradients (en dessous des plus petites valeurs de float16, environ $6\times10^{-8}$) ne tombent pas à zéro par underflow, et diviser les gradients par le même facteur avant la mise à jour. Les accumulations et les copies maîtresses des poids sont gardées en float32.

Souviens-toi aussi des **dénormaux** : en dessous du plus petit nombre normal, les flottants perdent de la précision progressivement (float32 atteint $1.4\times10^{-45}$), puis s'écrasent à 0.

## Erreur d'arrondi et annulation (avec une démonstration)

Supposons que $a$ et $b$ soient stockés avec des erreurs relatives jusqu'à $\delta$, donc $\tilde a=a(1+\delta_a)$, $\tilde b=b(1+\delta_b)$. Alors

$$
\tilde a-\tilde b=(a-b)+(a\delta_a-b\delta_b),\qquad
\frac{|\text{error}|}{|a-b|}\ \le\ \frac{|a|+|b|}{|a-b|}\,\delta .
$$

Quand $a\approx b$, le facteur d'amplification $\frac{|a|+|b|}{|a-b|}$ est énorme : l'erreur absolue reste de la taille de $\varepsilon\,|a|$ mais la vraie différence est bien plus petite, donc l'erreur *relative* explose. L'addition de nombres de même signe n'amplifie jamais ainsi ; la soustraction de quasi-égaux, si.

### Le piège de la variance (l'avertissement de Deisenroth, mesuré)

La formule « des scores bruts » $\text{Var}[x]=\mathbb E[x^2]-(\mathbb E[x])^2$ ne demande qu'un passage sur les données, mais Deisenroth et al. avertissent qu'elle est instable quand les deux termes sont énormes et presque égaux. Prends $x=10^9+\{4,7,13,16\}$. Les vraies valeurs d'écart à la moyenne sont $-6,-3,3,6$, donc la variance de la population est $(36+9+9+36)/4=22.5$ (variance empirique $90/3=30$). Les deux termes de la formule des scores bruts valent environ $10^{18}$, où l'espacement float64 est de 128 :

- `np.mean(x**2) - np.mean(x)**2` renvoie **$-128.0$**, une variance négative.
- Soustraire d'abord la moyenne, `x.var()`, renvoie $22.5$.

L'**algorithme de Welford** fait le centrage en ligne en un seul passage. Garde un compteur $n$, une moyenne $m$ et une somme des carrés des écarts $M_2$ courants. Pour chaque nouvelle valeur $x$ :

$$
\delta=x-m,\qquad m\leftarrow m+\frac{\delta}{n},\qquad M_2\leftarrow M_2+\delta\,(x-m_{\text{new}}),
$$

et à la fin $\text{Var}=M_2/n$ (ou $M_2/(n-1)$). *Pourquoi c'est juste :* après $n$ points, $M_2=\sum(x_i-m_n)^2$ ; la mise à jour ajoute le terme $\delta(x-m_{new})=\frac{n-1}{n}\delta^2$, qui est l'incrément exact de cette somme quand on ajoute un point (un court exercice d'algèbre : développe $\sum(x_i-m_n)^2$ en fonction de $m_{n-1}$). Toutes les quantités manipulées sont des écarts, jamais d'énormes carrés. Sur l'exemple, il renvoie exactement $22.5$ et $30.0$.

## Overflow, underflow et softmax stable

$\text{softmax}(z)_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$. En float32, $e^{88.7}\approx3.3\times10^{38}$ est la dernière valeur finie et $e^{89}$ vaut `inf` ; en float64 la limite est d'environ $e^{709}$. Avec les logits $z=(1000,1001,1002)$, la formule naïve donne `inf/inf = nan`.

**Identité clé.** Le softmax est inchangé si tu soustrais la même constante $c$ de chaque logit :

$$
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}=\frac{e^{-c}e^{z_i}}{e^{-c}\sum_j e^{z_j}}=\frac{e^{z_i}}{\sum_je^{z_j}} .
$$

Choisis $c=\max_j z_j$. Alors chaque exposant est $\le0$, donc aucun terme ne déborde, et au moins un terme vaut $e^0=1$, donc le dénominateur vaut au moins 1 et ne peut pas tomber à zéro par underflow. Pour $z=(1000,1001,1002)$ cela donne $(0.0900,\,0.2447,\,0.6652)$.

### Log-sum-exp

Les pertes ont besoin de $\log\sum_j e^{z_j}$ (le log-normalisateur). Le même décalage donne

$$
\operatorname{LSE}(z)=c+\log\sum_j e^{z_j-c},\qquad c=\max_j z_j .
$$

Pour $z=(1000,1001,1002)$ : $c=1002$, la somme est $e^{-2}+e^{-1}+1=1.5032$, et $\operatorname{LSE}=1002.4076$. La version naïve renvoie `inf`. L'entropie croisée pour une classe correcte $y$ est alors $-z_y+\operatorname{LSE}(z)$ : calcule-la directement à partir des logits plutôt que de prendre le $\log$ d'une sortie softmax, car une probabilité qui a fait un underflow à $0$ rend $\log 0=-\infty$. (C'est pourquoi les pertes des bibliothèques prennent des logits, pas des probabilités.)

## Conditionnement : le problème lui-même peut être instable

Même un algorithme parfait ne peut pas battre un problème mal conditionné. Le **conditionnement** est le facteur par lequel les erreurs relatives d'entrée peuvent être amplifiées en sortie. Pour résoudre $Ax=b$ :

$$
\frac{\|\delta x\|}{\|x\|}\ \le\ \kappa(A)\,\frac{\|\delta b\|}{\|b\|},\qquad \kappa(A)=\|A\|\,\|A^{-1}\|=\frac{\sigma_{\max}}{\sigma_{\min}} \ \ (\text{2-norm}),
$$

le rapport de la plus grande à la plus petite valeur singulière (leçon 7). Exemple : $A=\begin{pmatrix}1&1\\1&1.0001\end{pmatrix}$ a $\kappa\approx4\times10^{4}$. Pour $b=(2,\,2.0001)$ la solution est $(1,1)$ ; modifier $b_2$ de seulement $10^{-4}$, à $2.0002$, déplace la solution en $(0,2)$. Un minuscule changement des données change complètement la réponse. En ML, c'est pourquoi les équations normales $X^\top Xw=X^\top y$ sont risquées : $\kappa(X^\top X)=\kappa(X)^2$, donc élever la matrice au carré élève l'amplification au carré, et pourquoi la régularisation ridge (ajouter $\lambda I$) la stabilise.

## Ordre de sommation et différences finies

L'addition en virgule flottante n'est **pas associative** : $(a+b)+c\ne a+(b+c)$ en général. Float32 ne peut pas ajouter $1$ à $16{,}777{,}216=2^{24}$ (le résultat est arrondi à $2^{24}$) : une fois qu'un total courant est grand, les petits termes disparaissent. Additionner $10^6$ copies de $0.1$ en float32 une à une (comme le fait `np.cumsum`) finit à $100958.34$, une erreur d'environ 1 %, alors que `np.sum` (qui additionne par paires/blocs) donne $100000.01$. Deux remèdes standard :

- **La sommation par paires (en arbre)**, utilisée par `np.sum` : l'erreur croît comme $\log N$ plutôt que $N$.
- **La sommation compensée de Kahan** : garde une correction $c$ contenant les bits de poids faible perdus à chaque addition.

```text
y = x - c ; t = s + y ; c = (t - s) - y ; s = t
```

Sur les $10^6$ valeurs float32 $0.1$, elle finit exactement à $100000.0$. (Ici l'ordre en virgule flottante explique aussi pourquoi les réductions GPU peuvent donner des résultats légèrement différents d'une exécution à l'autre.)

**Différences finies revisitées (leçon 4).** Pour $f(x)=e^x$ en $x=1$, la différence centrée $\frac{f(x+h)-f(x-h)}{2h}$ a une erreur absolue de $5.9\times10^{-11}$ pour $h=10^{-5}$, mais $6.6\times10^{-9}$ pour $h=10^{-8}$, $2.1\times10^{-4}$ pour $h=10^{-12}$ et $0.17$ pour $h=10^{-15}$. Deux erreurs sont en compétition : l'erreur de troncature de la formule décroît comme $h^2$, tandis que l'erreur d'arrondi due à la soustraction de deux valeurs presque égales $f(x+h)-f(x-h)$ et à la division par un minuscule $2h$ croît comme $\varepsilon/h$. L'équilibre est à $h\approx\varepsilon^{1/3}\approx6\times10^{-6}$ en float64. Un $h$ plus petit n'est pas meilleur ; c'est de l'annulation catastrophique.

## Exemple détaillé

Quatre vérifications, toutes reproduites par l'exercice :

1. `softmax([1000, 1001, 1002])` : naïf `[nan, nan, nan]`, stable `[0.09003, 0.24473, 0.66524]` (somme égale à 1).
2. `logsumexp([1000, 1001, 1002]) = 1002.4076` ; et `logsumexp([-1000, -1000]) = -1000 + log 2 = -999.3069`, où la formule naïve donne `-inf` car $e^{-1000}$ fait un underflow à 0.
3. Variance de $10^9+\{4,7,13,16\}$ : scores bruts $-128.0$, Welford $22.5$ (population) et $30.0$ (empirique).
4. Somme float32 de $10^5$ copies de $0.1$ : séquentielle $9998.557$, Kahan $10000.0$.

## Implémentation

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def logsumexp(z):
    c = z.max()
    return c + np.log(np.exp(z - c).sum())

def welford(xs):
    n, m, M2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        d = x - m
        m += d / n
        M2 += d * (x - m)
    return m, M2 / n
```

## Formes

Pour un lot de logits `z: (B, K)`, prends le max sur l'axe des classes avec `keepdims=True` pour que le décalage `(B, 1)` se diffuse contre `(B, K)` (leçon 16) ; `softmax(z): (B, K)`, `logsumexp(z): (B,)`. Welford sur un flux de `N` vecteurs de `R^D` garde `m: (D,)` et `M2: (D,)`.

## Idées fausses courantes

- **« Utilise float64 et le problème disparaît. »** Cela ne fait que déplacer la falaise : la variance par scores bruts avec des décalages de $10^{9}$ échoue déjà en float64, et le matériel de ML utilise délibérément des formats 16 bits.
- **« `nan` signifie que mes données sont mauvaises. »** Très souvent c'est `inf - inf`, `0 * inf` ou `log(0)` produit par une formule instable.
- **« `a == b` est un bon test pour des flottants. »** Compare avec une tolérance, `np.isclose(a, b)`.
- **« Additionner est toujours sûr. »** Ajouter un tout petit nombre à un énorme total courant le fait silencieusement disparaître.
- **« Un algorithme stable répare un problème mal conditionné. »** Non : un mauvais $\kappa$ amplifie même l'arrondi d'entrée inévitable.

## Points clés à retenir

- Les flottants ont une précision relative $\varepsilon$ (float64 $2.2\times10^{-16}$, float32 $1.2\times10^{-7}$, float16 $10^{-3}$, bfloat16 $8\times10^{-3}$) et une plage finie ; bfloat16 échange de la précision contre la plage de float32.
- Soustraire des nombres presque égaux perd des chiffres ; réorganise (centre d'abord, utilise Welford) plutôt que de soustraire de grands carrés.
- Soustrais le max avant d'exponentier : softmax stable et $\operatorname{LSE}(z)=c+\log\sum e^{z-c}$ ; donne des logits aux pertes, pas des probabilités.
- $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ mesure l'amplification des erreurs ; l'ordre de sommation compte (par paires, Kahan).

## Explique-le avec tes mots

1. Pourquoi soustraire le logit maximal ne change-t-il pas le softmax, mais empêche-t-il `nan` ? Couvre à la fois le côté overflow et le côté underflow.
2. Explique pourquoi la formule de variance « moyenne des carrés moins carré de la moyenne » peut donner un nombre négatif sur un ordinateur, et ce que fait Welford de différent.

## Pour aller plus loin

- Dépôt, phase 01, leçon 13 : précision mixte, écrêtage de gradient, détection de NaN/Inf, vérification de gradient.
- Deisenroth et al., Sec. 6.4.3 et les remarques sur la stabilité numérique au chapitre 7 sur l'optimisation.
- Goldberg, « What every computer scientist should know about floating-point arithmetic » (une référence classique ; ne fait pas partie des sources lues pour cette leçon).

## Critères de maîtrise

- Donner l'epsilon machine et la plage de float64, float32, float16 et bfloat16 et dire pourquoi bfloat16 est utilisé pour l'entraînement.
- Montrer avec des chiffres pourquoi la variance par scores bruts échoue et implémenter Welford.
- Implémenter le softmax stable et le log-sum-exp et démontrer l'identité de décalage.
- Expliquer ce que mesure le conditionnement et pourquoi $\kappa(X^\top X)=\kappa(X)^2$ importe.
