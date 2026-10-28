---
id: math.09
title: "Variables aléatoires, espérance, variance et gaussienne"
summary: "Décris des quantités incertaines par des distributions, résume-les avec la moyenne, la variance et la covariance, et vois pourquoi la gaussienne et le maximum de vraisemblance se ramènent à l'erreur quadratique."
minutes: 80
teaches: [prob.rv]
requires: [prob.basics, calc.derivative]
sources:
  - "math/math for ML.pdf, Sec. 6.2 Discrete and Continuous Probabilities (PDF p.184-189): pmf, pdf, cdf, density can exceed 1, nomenclature table. READ."
  - "math/math for ML.pdf, Sec. 6.3 Sum Rule, Product Rule, Bayes (PDF p.189-192): marginalisation and the product rule, recalled from the previous lesson. READ."
  - "math/math for ML.pdf, Sec. 6.4 Summary Statistics and Independence (PDF p.192-202): expected value and its linearity, covariance, correlation, empirical mean/covariance (1/N is biased), three expressions for the variance, affine transformation, uncorrelated vs independent. READ."
  - "math/math for ML.pdf, Sec. 6.5 Gaussian Distribution (PDF p.203-210): univariate and multivariate density, conditionals/marginals, sums, A mu and A Sigma A^T, sampling with the Cholesky factor. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 2.3 (PDF p.98-102): central limit theorem, Mahalanobis distance, eigen-decomposition of the covariance and the elliptical contours. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 2.3.4 (PDF p.113-114): maximum likelihood for the Gaussian, mu_ML and Sigma_ML, E[Sigma_ML] = (N-1)/N Sigma. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/06-probability-and-distributions/docs/en.md: pmf vs pdf, CLT story, log-probabilities. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/15-statistics-for-ml/docs/en.md (first part): descriptive statistics, Bessel correction, Pearson correlation. READ."
---

## Où cela s'inscrit

Un classifieur ne produit pas « chat ». Il produit des nombres comme `[0.03, 0.91, 0.06]`. Un modèle de régression ne connaît pas le prix exact d'une maison : les mesures sont bruitées, les acheteurs sont lunatiques. Chaque fois qu'un modèle doit parler de quelque chose dont il n'est pas sûr, il lui faut un moyen de dire *à quel point* il est incertain. Ce langage, c'est une **variable aléatoire munie d'une distribution**.

Cette leçon est le vocabulaire que tout ce qui suit utilise. La perte que tu minimises est une *espérance* (une moyenne sur la distribution des données). La dispersion de tes gradients est une *variance*. Des variables qui bougent ensemble ont une *covariance*, et la décomposition en valeurs propres de cette matrice de covariance (leçon précédente) est l'ACP. L'initialisation, le bruit dans les modèles de diffusion, le filtre de Kalman, les processus gaussiens : tout cela, ce sont des gaussiennes. Et à la fin nous démontrons quelque chose que tu rencontreras sans cesse : **ajuster une gaussienne par maximum de vraisemblance revient à minimiser l'erreur quadratique moyenne**. La leçon suivante (théorie de l'information) utilise exactement la même astuce du « log d'une probabilité » pour la classification.

## L'idée en mots simples

- Une **variable aléatoire** est un nombre qui n'est pas encore décidé. « La valeur du prochain lancer de dé », « la température de demain », « la taille d'une personne tirée au hasard ». Avant de regarder, tu ne peux pas dire ce que c'est ; après avoir regardé, ce n'est qu'un nombre. La *distribution* est la recette qui dit quelle est la probabilité de chaque valeur possible.
- Si les valeurs viennent une à une (faces d'un dé, étiquettes de classe), la recette est un **tableau de probabilités** dont la somme vaut 1. Si la valeur glisse le long d'un continuum (une taille), aucune valeur exacte n'a de probabilité visible, donc la recette est une **courbe dont l'aire sur un intervalle est la probabilité de tomber dans cet intervalle**. La hauteur de la courbe est une *densité*, pas une probabilité : une cloche très étroite peut dépasser 1 en hauteur, tant que l'aire totale reste égale à 1.
- L'**espérance** est le point d'équilibre : pose la distribution sur une balançoire, la moyenne est l'endroit où elle s'équilibre. C'est aussi ce que tu obtiendrais en moyenne en répétant l'expérience indéfiniment. La **variance** dit de combien les valeurs s'écartent typiquement de ce point d'équilibre (au carré, pour que les écarts au-dessus et en dessous ne s'annulent pas).
- Avec deux variables, on se demande si elles **bougent ensemble**. Les personnes grandes ont tendance à être plus lourdes : quand l'une est au-dessus de sa moyenne, l'autre l'est généralement aussi. Cette tendance est la **covariance**. Divise-la par les deux dispersions et tu obtiens la **corrélation**, un nombre sans unité compris entre -1 et 1.
- La **gaussienne** (courbe en cloche) est ce que tu obtiens quand de nombreuses petites perturbations indépendantes s'additionnent (le théorème central limite). Deux nombres la décrivent entièrement : où se trouve le centre et quelle est sa largeur. En plusieurs dimensions, la cloche devient une **colline aux courbes de niveau elliptiques** : la moyenne dit où est le sommet, la matrice de covariance dit comment l'ellipse est étirée et inclinée. Comme elle n'est décrite que par une moyenne et une covariance, tu peux la faire passer par une application linéaire en ne faisant passer que ces deux objets.

## Définitions

### Discret et continu

Une **fonction de masse** (pmf) donne $P(X=x)$ pour chaque valeur, avec $\sum_x P(X=x)=1$. Une **fonction de densité de probabilité** (pdf) est une fonction $p(x)\ge 0$ avec $\int p(x)\,dx = 1$, et

$$
P(a\le X\le b)=\int_a^b p(x)\,dx .
$$

La probabilité d'une valeur exacte est $0$ pour une variable continue, et $p(x)$ peut dépasser $1$ (l'exemple du livre : une densité uniforme sur $[0.9,1.6]$ a une hauteur de $1/0.7\approx1.43$). La **fonction de répartition** (cdf) est $F(x)=P(X\le x)=\int_{-\infty}^x p(t)\,dt$.

### Espérance

Pour une fonction $g$ d'une variable aléatoire $X$ (Deisenroth Def. 6.3) :

$$
\mathbb E[g(X)]=\sum_x g(x)P(X=x)\quad\text{or}\quad \int g(x)\,p(x)\,dx .
$$

La moyenne correspond à $g(x)=x$. L'espérance est **linéaire** : $\mathbb E[aX+bY+c]=a\,\mathbb E[X]+b\,\mathbb E[Y]+c$, que $X$ et $Y$ soient indépendantes ou non (la remarque du livre : l'intégrale d'une somme est la somme des intégrales).

### La variance, en trois expressions

$$
\operatorname{Var}[X]=\mathbb E\big[(X-\mu)^2\big]=\mathbb E[X^2]-\mu^2,\qquad \mu=\mathbb E[X].
$$

L'expression du milieu est la définition. Celle de droite (« moyenne du carré moins carré de la moyenne », la **formule des scores bruts**) s'obtient en développant le carré et en utilisant la linéarité :

$$
\mathbb E[(X-\mu)^2]=\mathbb E[X^2-2\mu X+\mu^2]=\mathbb E[X^2]-2\mu\,\mathbb E[X]+\mu^2=\mathbb E[X^2]-\mu^2 .
$$

(Chaque étape est une seule règle : développer, séparer l'espérance par linéarité, remplacer $\mathbb E[X]$ par $\mu$, et $-2\mu^2+\mu^2=-\mu^2$.) La troisième expression donnée par le livre est une somme sur toutes les paires d'observations, $\frac1{N^2}\sum_{i,j}(x_i-x_j)^2 = 2\big(\frac1N\sum x_i^2-(\frac1N\sum x_i)^2\big)$ : le double de la variance des scores bruts.

**Piège numérique.** La formule des scores bruts ne demande qu'un seul passage sur les données, mais elle soustrait deux nombres énormes et presque égaux. Cela détruit la précision en virgule flottante (Deisenroth met en garde contre cela). Dans l'exemple détaillé ci-dessous, tu la verras renvoyer une variance *négative*.

### Covariance et corrélation

$$
\operatorname{Cov}[X,Y]=\mathbb E\big[(X-\mathbb E X)(Y-\mathbb E Y)\big]=\mathbb E[XY]-\mathbb E[X]\,\mathbb E[Y],\qquad \operatorname{corr}[X,Y]=\frac{\operatorname{Cov}[X,Y]}{\sqrt{\operatorname{Var}[X]\operatorname{Var}[Y]}}\in[-1,1].
$$

Pour un vecteur aléatoire $\mathbf x\in\mathbb R^D$, la **matrice de covariance** est $\Sigma=\mathbb E[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top]\in\mathbb R^{D\times D}$ : les variances sur la diagonale, les covariances en dehors. Elle est symétrique et semi-définie positive. La corrélation est le cosinus de l'angle entre les deux variables centrées, ce qui explique pourquoi elle se trouve dans $[-1,1]$ (Sec. 6.4.6 du livre).

**L'indépendance implique une covariance nulle, pas l'inverse.** Prends $X$ symétrique autour de $0$ et $Y=X^2$. Alors $Y$ est entièrement déterminée par $X$, et pourtant $\operatorname{Cov}[X,Y]=\mathbb E[X^3]-\mathbb E[X]\mathbb E[X^2]=0$. La covariance ne voit que les co-variations *linéaires*.

### Moyenne et covariance empiriques : $N$ contre $N-1$

À partir d'un jeu de données $\mathbf x_1,\dots,\mathbf x_N$, on estime

$$
\bar{\mathbf x}=\frac1N\sum_{n}\mathbf x_n,\qquad \hat\Sigma=\frac1N\sum_n(\mathbf x_n-\bar{\mathbf x})(\mathbf x_n-\bar{\mathbf x})^\top .
$$

La version en $1/N$ est l'estimateur du maximum de vraisemblance et elle est **biaisée** : $\mathbb E[\hat\Sigma]=\frac{N-1}{N}\Sigma$ (Bishop 2.124). La raison est que $\bar{\mathbf x}$ a lui-même été ajusté sur ces mêmes points, donc les points sont un peu plus proches de $\bar{\mathbf x}$ que de la vraie moyenne. Diviser par $N-1$ donne l'estimateur non biaisé. Avec des milliers de points la différence est invisible ; avec une poignée, elle compte. `np.cov` de NumPy utilise $N-1$ par défaut, `np.var` utilise $N$.

### Transformations affines

Soit $\mathbf x$ de moyenne $\boldsymbol\mu$ et de covariance $\Sigma$, et soit $\mathbf y=A\mathbf x+\mathbf b$ avec $A\in\mathbb R^{m\times D}$, $\mathbf b\in\mathbb R^m$. Alors

$$
\mathbb E[\mathbf y]=A\boldsymbol\mu+\mathbf b,\qquad \operatorname{Cov}[\mathbf y]=A\,\Sigma\,A^\top\in\mathbb R^{m\times m}.
$$

Démonstration. La moyenne découle de la linéarité. Pour la covariance, $\mathbf y-\mathbb E\mathbf y=A(\mathbf x-\boldsymbol\mu)$ (la constante $\mathbf b$ s'annule : décaler ne change jamais la dispersion), donc

$$
\operatorname{Cov}[\mathbf y]=\mathbb E\big[A(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top A^\top\big]=A\,\mathbb E\big[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top\big]A^\top=A\Sigma A^\top ,
$$

puisque $A$ est une constante et peut sortir de l'espérance de part et d'autre. En dimension un, on retrouve la formule familière $\operatorname{Var}[aX+b]=a^2\operatorname{Var}[X]$.

### La gaussienne

Univariée, de moyenne $\mu$ et de variance $\sigma^2$ :

$$
\mathcal N(x\mid\mu,\sigma^2)=\frac1{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).
$$

Multivariée, avec $\mathbf x,\boldsymbol\mu\in\mathbb R^D$ et $\Sigma\in\mathbb R^{D\times D}$ symétrique définie positive :

$$
\mathcal N(\mathbf x\mid\boldsymbol\mu,\Sigma)=\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\!\Big(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Big).
$$

L'exposant contient le carré de la **distance de Mahalanobis** $\Delta^2=(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$, qui se réduit au carré de la distance ordinaire quand $\Sigma=I$. En écrivant $\Sigma=\sum_i\lambda_i\mathbf u_i\mathbf u_i^\top$ (décomposition en valeurs propres, leçon précédente), Bishop montre que $\Delta^2=\sum_i y_i^2/\lambda_i$ avec $y_i=\mathbf u_i^\top(\mathbf x-\boldsymbol\mu)$ : dans les coordonnées tournées, la gaussienne se factorise en $D$ gaussiennes 1-D indépendantes. Les courbes de niveau de densité constante sont des ellipses dont les axes pointent le long des vecteurs propres $\mathbf u_i$ et ont des demi-longueurs proportionnelles à $\sqrt{\lambda_i}$.

Trois faits utilisés constamment (Deisenroth Sec. 6.5) :

1. **Les applications affines restent gaussiennes :** $\mathbf x\sim\mathcal N(\boldsymbol\mu,\Sigma)\Rightarrow A\mathbf x+\mathbf b\sim\mathcal N(A\boldsymbol\mu+\mathbf b,\ A\Sigma A^\top)$.
2. **Les sommes de gaussiennes indépendantes sont gaussiennes :** les moyennes s'ajoutent, les covariances s'ajoutent.
3. **Les marginales et les conditionnelles d'une gaussienne jointe sont gaussiennes.** Dans l'exemple bivarié du livre (moyenne $(0,2)$, covariance de lignes $(0.3,-1)$ et $(-1,5)$), la marginale de $x_1$ est $\mathcal N(0,0.3)$ et la conditionnelle sachant $x_2=-1$ est $\mathcal N(0.6,0.1)$.

### Échantillonner avec le facteur de Cholesky

Les ordinateurs te donnent des échantillons de $\mathbf z\sim\mathcal N(\mathbf 0,I)$ (des normales standard indépendantes). Pour obtenir $\mathbf y\sim\mathcal N(\boldsymbol\mu,\Sigma)$, utilise le fait 1 avec $A=L$ où $LL^\top=\Sigma$ :

$$
\mathbf y=\boldsymbol\mu+L\mathbf z,\qquad \operatorname{Cov}[\mathbf y]=L\,I\,L^\top=\Sigma .
$$

La factorisation de Cholesky donne un $L$ triangulaire inférieur (peu coûteux), et elle existe parce que $\Sigma$ est symétrique définie positive. C'est ainsi que toute gaussienne multivariée est échantillonnée en pratique, et c'est l'*astuce de reparamétrisation* des autoencodeurs variationnels.

## Maximum de vraisemblance pour la gaussienne

Supposons que $x_1,\dots,x_N$ soient tirés indépendamment de $\mathcal N(\mu,\sigma^2)$ et que nous voulions les meilleurs paramètres. L'indépendance fait de la vraisemblance un produit ; prendre le log le transforme en somme (et évite de multiplier des centaines de tout petits nombres, ce qui provoque un underflow) :

$$
\ln p(\mathbf x\mid\mu,\sigma^2)=-\frac N2\ln(2\pi\sigma^2)-\frac1{2\sigma^2}\sum_{n=1}^N(x_n-\mu)^2 .
$$

Dérivons par rapport à $\mu$ et annulons :

$$
\frac{\partial}{\partial\mu}=\frac1{\sigma^2}\sum_n(x_n-\mu)=0\ \Longrightarrow\ \mu_{\rm ML}=\frac1N\sum_n x_n .
$$

Dérivons par rapport à $\sigma^2$ (écris $v=\sigma^2$) : $-\frac N{2v}+\frac1{2v^2}\sum_n(x_n-\mu)^2=0$, ce qui donne $v_{\rm ML}=\frac1N\sum_n(x_n-\mu_{\rm ML})^2$. Le résultat multivarié est le même avec des produits extérieurs (Bishop 2.121, 2.122). Ainsi l'ajustement par maximum de vraisemblance d'une gaussienne est : **la moyenne empirique, et la covariance empirique biaisée (en $1/N$).**

**Lien avec l'EQM.** Pour $\sigma^2$ fixé, le seul terme qui dépend de $\mu$ est $\frac1{2\sigma^2}\sum_n(x_n-\mu)^2$. Maximiser la vraisemblance revient donc à minimiser la somme des carrés des erreurs. Dans un modèle de régression $t_n=f(\mathbf x_n;\mathbf w)+\varepsilon$ avec un bruit gaussien $\varepsilon\sim\mathcal N(0,\sigma^2)$, le même calcul dit : **minimiser la perte EQM, c'est le maximum de vraisemblance sous bruit gaussien.** L'erreur quadratique n'est pas un choix arbitraire ; c'est l'opposé du log d'une courbe en cloche.

## Exemple détaillé

Tous les nombres ci-dessous sont reproduits par le code de l'exercice.

**1. Espérance discrète.** $X\in\{0,1,2,3\}$ avec $P=(0.1,0.2,0.4,0.3)$. Moyenne : $0\cdot0.1+1\cdot0.2+2\cdot0.4+3\cdot0.3=1.9$. $\mathbb E[X^2]=0.2+1.6+2.7=4.5$. Variance $=4.5-1.9^2=0.89$ ; la définition $\sum_x(x-1.9)^2P(x)$ donne aussi $0.89$.

**2. Le piège des scores bruts.** Données $10^9+(4,7,13,16)$. La vraie variance est $22.5$ (invariante par translation). En float64, `mean(x**2) - mean(x)**2` renvoie **-128.0** : chaque terme vaut environ $10^{18}$, où les flottants voisins sont espacés de $128$, donc le vrai $22.5$ est arrondi et perdu. Soustraire d'abord la moyenne (deux passages) donne exactement $22.5$.

**3. Covariance empirique.** Cinq points $(2,1),(4,3),(6,2),(8,6),(10,8)$ (tableau $X$ de forme $(5,2)$). Moyenne $(6,4)$. Avec la matrice centrée $X_c=X-\bar x$, le produit $X_c^\top X_c/5$ a pour lignes $(8,\,6.8)$ et $(6.8,\,6.8)$, tandis que la division par $N-1=4$ donne les lignes $(10,\,8.5)$ et $(8.5,\,8.5)$ (rapport $5/4$). Corrélation $=6.8/\sqrt{8\cdot6.8}=0.922$.

**4. Application affine.** $\boldsymbol\mu=(1,2)$, $\Sigma$ de lignes $(2,0.6)$ et $(0.6,1)$, $A$ de lignes $(1,1)$ et $(0,2)$, $\mathbf b=(0,1)$. Alors $A\boldsymbol\mu+\mathbf b=(3,5)$ et $A\Sigma A^\top$ a pour lignes $(4.2,\,3.2)$ et $(3.2,\,4)$. Vérification à la main : $A\Sigma$ a pour lignes $(2.6,\,1.6)$ et $(1.2,\,2)$ ; multiplie par $A^\top$, dont les lignes sont $(1,0)$ et $(1,2)$.

**5. Cholesky.** Pour la même $\Sigma$ : $L$ a pour lignes $(1.4142,\,0)$ et $(0.4243,\,0.9055)$, et $LL^\top=\Sigma$. Les échantillons $\boldsymbol\mu+L\mathbf z$ issus de $200\,000$ tirages normaux standard ont une covariance empirique proche de $\Sigma$.

**6. Maximum de vraisemblance.** Données $(2,4,4,5,5)$ : $\mu_{\rm ML}=4$, écarts $(-2,0,0,1,1)$, $\sigma^2_{\rm ML}=6/5=1.2$, non biaisé $6/4=1.5$. À l'optimum, la log-vraisemblance vaut $-\frac N2\ln(2\pi\sigma^2_{\rm ML})-\frac N2=-7.55$.

## Implémentation

```python
import numpy as np

x = 1e9 + np.array([4., 7, 13, 16])
raw = np.mean(x**2) - np.mean(x)**2          # -128.0  (catastrophic cancellation)
two_pass = np.mean((x - x.mean())**2)        # 22.5

X = np.array([[2., 1], [4, 3], [6, 2], [8, 6], [10, 8]])   # (N, D) = (5, 2)
Xc = X - X.mean(axis=0)
cov_biased = Xc.T @ Xc / len(X)               # (D, D)
cov_unbiased = np.cov(X, rowvar=False)        # divides by N-1

Sigma = np.array([[2., .6], [.6, 1.]])
L = np.linalg.cholesky(Sigma)
z = np.random.default_rng(0).standard_normal((1000, 2))   # (n, D)
y = np.array([1., 2.]) + z @ L.T                           # rows are samples
```

## Formes

Matrice de données `X: (N, D)` avec les échantillons en lignes. Moyenne `(D,)`. Covariance `(D, D)`. Si les lignes sont les échantillons, une application affine s'applique comme `X @ A.T + b` (la formule $A\mathbf x$ vaut pour un seul vecteur colonne), et la covariance du résultat est `A @ Sigma @ A.T` de forme `(m, m)`. Échantillonnage : `Z: (n, D)` normales standard, `Z @ L.T: (n, D)`.

## Idées fausses courantes

- **Une valeur de densité est une probabilité.** Non : elle peut dépasser 1 ; seules les aires sont des probabilités.
- **Covariance nulle signifie indépendance.** Seulement pour des variables conjointement gaussiennes. $Y=X^2$ est le contre-exemple.
- **Il faut toujours diviser par $N-1$.** C'est l'estimateur non biaisé de la variance ; la version en $1/N$ est celle du maximum de vraisemblance. Les deux sont consistantes ; sache laquelle ta bibliothèque utilise (`np.var` utilise $N$, `np.cov` utilise $N-1$).
- **La variance en deux lignes par scores bruts est un raccourci sûr.** Elle s'annule de façon catastrophique quand la moyenne est grande devant la dispersion. Centre d'abord.
- **Une matrice de covariance peut être n'importe quelle matrice.** Elle est symétrique semi-définie positive ; c'est ce qui permet l'existence du facteur de Cholesky et de la densité gaussienne.

## Points clés à retenir

- Une variable aléatoire est décrite par une pmf (tableaux, somme égale à 1) ou une pdf (courbes, intégrale égale à 1, peut dépasser 1).
- $\mathbb E$ est linéaire ; $\operatorname{Var}[X]=\mathbb E[X^2]-\mathbb E[X]^2$ mais calcule-la en centrant d'abord.
- $\mathbf y=A\mathbf x+\mathbf b$ donne la moyenne $A\boldsymbol\mu+\mathbf b$ et la covariance $A\Sigma A^\top$ ; pour des gaussiennes, le résultat est à nouveau gaussien.
- Échantillonne $\mathcal N(\boldsymbol\mu,\Sigma)$ avec $\boldsymbol\mu+L\mathbf z$, $LL^\top=\Sigma$.
- Maximum de vraisemblance pour une gaussienne : moyenne empirique et covariance en $1/N$ ; à variance fixée, c'est exactement les moindres carrés.

## Explique-le avec tes mots

1. Pourquoi la covariance de $A\mathbf x+\mathbf b$ ne dépend-elle pas de $\mathbf b$, et pourquoi $A$ apparaît-elle des deux côtés ?
2. Avec tes propres mots : pourquoi minimiser l'EQM revient-il à supposer un bruit gaussien ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 6.4 (PDF p.192) et 6.5 (PDF p.203).
- Bishop, *PRML*, Sec. 2.3 (PDF p.98), en particulier les gaussiennes conditionnelles et marginales (PDF p.105-110, hors des sources de cette leçon).

## Critères de maîtrise

Tu sais calculer à la main la moyenne, la variance et la covariance d'un petit jeu de données discret ou empirique, expliquer le choix entre $N$ et $N-1$, propager une gaussienne à travers une application affine, échantillonner une gaussienne multivariée à l'aide d'un facteur de Cholesky, et démontrer $\mu_{\rm ML}$ et $\sigma^2_{\rm ML}$ ainsi que le lien avec l'erreur quadratique.
