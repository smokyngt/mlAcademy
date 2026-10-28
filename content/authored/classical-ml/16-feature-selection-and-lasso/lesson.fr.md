---
id: cml.16
title: "Sélection de variables et lasso : chemins de régularisation et stabilité"
summary: "Classer les méthodes de sélection en familles filtre, enveloppe (wrapper) et intégrée, dériver le seuillage doux pour la pénalité L1, implémenter le lasso par descente de coordonnées et tracer son chemin de régularisation, puis juger à quelles variables sélectionnées se fier grâce à la sélection par stabilité."
minutes: 80
teaches: [ml.featsel, ml.lasso]
requires: [ml.ridge, ml.linreg, ml.features, info.entropy, opt.convex]
sources:
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/docs/en.md (filter, wrapper, embedded families; variance threshold, mutual information, RFE, L1, tree importance, permutation importance, comparison). READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 3.1.4 Regularized least squares (PDF p.163-166): quadratic regulariser and weight decay, generalised regulariser with exponent q, lasso as q = 1, equivalent constrained form, sparsity from the constraint geometry. READ."
---

## Où cela s'inscrit

Les jeux de données réels comptent des centaines de colonnes, dont la plupart sont redondantes ou du pur bruit. Plus de colonnes n'aide pas un modèle : elles ajoutent de la variance (surapprentissage, [[cml.03]]), ralentissent l'entraînement et rendent le résultat impossible à expliquer. La **sélection de variables** choisit un sous-ensemble utile. Cette leçon relie trois choses que tu connais déjà : la pénalité L2 de la régression ridge ([[cml.01]]), l'information mutuelle comme grandeur d'entropie (entropie et divergence KL), et l'avertissement sur la fuite de [[cml.15]] (la sélection est une étape ajustée et doit vivre à l'intérieur de la boucle de validation croisée).

La vedette de la leçon est le **lasso**, la régression linéaire avec pénalité L1. Il réalise sélection et ajustement en une seule optimisation, et sa solution a une mise à jour par coordonnée en forme close (le seuillage doux) que tu vas dériver et implémenter.

## L'idée en mots simples

Trois façons de recruter une équipe parmi 200 candidats.

- **Filtre** : note chaque candidat seul avec un test rapide (corrélation avec la performance au travail, ou un score plus général « combien cela m'apprend-il ») et garde les meilleurs. Rapide, mais aveugle au travail d'équipe : deux candidats peuvent être excellents seuls et redondants ensemble.
- **Wrapper (enveloppe)** : essaie des équipes. Ajoute le membre qui améliore le plus le résultat de l'équipe, recommence (sélection ascendante, forward), ou pars de tout le monde et retire le moins utile (élimination descendante, backward). Précis pour le modèle choisi, coûteux : beaucoup d'ajustements de modèle.
- **Intégrée (embedded)** : la procédure d'entraînement elle-même écarte des gens. Le lasso fait payer chaque unité de poids que porte une variable, donc les variables qui ne se rentabilisent pas finissent avec un poids exactement nul.

Pourquoi *exactement* nul ? Imagine un budget sur la taille absolue totale des poids, dessiné comme un losange dans l'espace des poids avec ses coins sur les axes. Le meilleur ajustement qui touche le losange tend à toucher un coin, et un coin signifie que certains poids sont nuls. Le budget de ridge est un disque rond : l'ajustement le touche en un point quelconque où rien n'est exactement nul.

## Filtre, wrapper, intégrée

**Filtres.** Seuil de variance (retirer les colonnes constantes), corrélation avec la cible, et **information mutuelle** (MI). Pour des variables discrètes

$$
I(X;Y)=\sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)\,p(y)}\ \ge 0,
$$

c'est la divergence KL entre la loi conjointe et le produit des marginales (entropie et divergence KL) ; elle est nulle exactement quand $X$ et $Y$ sont indépendantes. Contrairement à la corrélation, elle détecte la dépendance non linéaire : pour $y=x^2+\text{noise}$ avec $x$ symétrique, la corrélation est à peu près nulle, mais la MI est grande. Pour des variables continues, on l'estime avec des histogrammes (découpage en quantiles), des estimateurs à noyau ou aux plus proches voisins ; les estimations dépendent du découpage et de la taille d'échantillon, donc compare les scores des variables entre eux, pas leurs valeurs absolues.

**Wrappers.** Sélection ascendante, élimination descendante, élimination récursive de variables (RFE : ajuster, retirer les variables de plus petit poids, réajuster). Ils optimisent la métrique qui t'intéresse mais coûtent beaucoup d'ajustements, et avec de nombreux sous-ensembles candidats ils surapprennent les données de validation.

**Intégrée.** La régularisation L1, et les mesures d'importance intégrées aux arbres. L'importance d'arbre fondée sur les coupures est biaisée en faveur des variables à forte cardinalité (une colonne d'identifiants « coupe bien ») ; l'importance par permutation ([[cml.17]]) est un meilleur contrôle.

## Le lasso

Soit $X\in\mathbb R^{n\times d}$ à colonnes standardisées (moyenne 0, variance 1), $y\in\mathbb R^n$ centré. Le lasso résout

$$
\min_{w\in\mathbb R^d}\ \frac1{2n}\|y-Xw\|_2^2+\lambda\|w\|_1,\qquad \|w\|_1=\sum_j|w_j| .
$$

Bishop (Sec. 3.1.4) écrit la famille $\frac12\sum_n\{t_n-w^\top\phi(x_n)\}^2+\frac\lambda2\sum_j|w_j|^q$ ; $q=2$ est la décroissance des poids (ridge) et $q=1$ est le lasso. Minimiser avec la pénalité équivaut, par multiplicateurs de Lagrange (voir le matériel sur la convexité et les multiplicateurs de Lagrange), à minimiser l'erreur quadratique sous la contrainte $\sum_j|w_j|\le\eta$ pour un certain $\eta$ dépendant de $\lambda$. Les lignes de niveau de l'erreur quadratique sont des ellipses ; le losange $\sum|w_j|\le\eta$ a ses coins sur les axes, ce qui est l'origine de la parcimonie. Standardise d'abord : la pénalité traite tous les poids pareillement, donc des variables d'échelles différentes seraient pénalisées inégalement.

### Dérivation : le seuillage doux

Fixe tous les poids sauf $w_j$ et soit $r^{(j)}=y-\sum_{k\ne j}X_{:k}w_k$ le résidu partiel. Définis $z_j=\frac1n X_{:j}^\top X_{:j}$ (égal à 1 pour des colonnes standardisées) et $\rho_j=\frac1n X_{:j}^\top r^{(j)}$. En développant le carré, l'objectif en fonction de $w=w_j$ est, à une constante près,

$$
g(w)=\tfrac12 z_j w^2-\rho_j w+\lambda|w| .
$$

Il est convexe mais non dérivable en $0$. Traitons les cas.

- Si $w>0$ : $g'(w)=z_jw-\rho_j+\lambda=0$ donne $w=(\rho_j-\lambda)/z_j$, valable seulement si $\rho_j>\lambda$.
- Si $w<0$ : $g'(w)=z_jw-\rho_j-\lambda=0$ donne $w=(\rho_j+\lambda)/z_j$, valable seulement si $\rho_j<-\lambda$.
- Sinon le minimum se situe en $w=0$ : la condition de sous-gradient $0\in -\rho_j+\lambda[-1,1]$ est vérifiée exactement quand $|\rho_j|\le\lambda$.

Ensemble :

$$
\boxed{\,w_j=\frac{S(\rho_j,\lambda)}{z_j},\qquad S(\rho,\lambda)=\operatorname{sign}(\rho)\max(|\rho|-\lambda,0)\,}
$$

l'opérateur de **seuillage doux** : rétrécir vers zéro de $\lambda$, et tronquer à zéro si la corrélation est sous le seuil. L'analogue ridge, $w_j=\rho_j/(z_j+\lambda)$, ne fait que redimensionner et ne donne jamais de zéros exacts.

La **descente de coordonnées** parcourt $j=1,\dots,d$ en appliquant cette mise à jour, en maintenant le résidu $r=y-Xw$ pour que chaque mise à jour coûte $O(n)$. L'objectif est convexe et sa partie non lisse se sépare par coordonnée, donc les balayages convergent vers le minimum global.

**Plus grande pénalité utile.** En $w=0$, la condition d'optimalité pour chaque coordonnée est $|\rho_j|=|\frac1nX_{:j}^\top y|\le\lambda$, d'où

$$
\lambda_{\max}=\max_j\frac1n|X_{:j}^\top y| ,
$$

et pour $\lambda\ge\lambda_{\max}$ le modèle est vide. Diminuer $\lambda$ depuis $\lambda_{\max}$ laisse les variables entrer une à une : c'est le **chemin de régularisation**. Une recette courante consiste à calculer le chemin sur une grille espacée logarithmiquement, en initialisant chaque solution à partir de la précédente (warm start), et à choisir $\lambda$ par validation croisée (avec toute la procédure dans un Pipeline, [[cml.15]]).

**Conditions d'optimalité (KKT)** que tu peux tester numériquement : avec $g=\frac1nX^\top(y-Xw)$, les coordonnées actives satisfont $g_j=\lambda\,\mathrm{sign}(w_j)$ et les inactives satisfont $|g_j|\le\lambda$.

### Sélection par stabilité

L'ensemble des coefficients non nuls du lasso est fragile : avec des variables corrélées, il en choisit souvent une arbitrairement dans un groupe, et de petits changements dans les données changent le gagnant. La **sélection par stabilité** relance le lasso sur de nombreux demi-échantillons aléatoires et enregistre la fréquence à laquelle chaque variable est sélectionnée. Les variables dont la fréquence de sélection est proche de 1 sont fiables ; les variables de bruit apparaissent sporadiquement. Garde celles au-dessus d'un seuil (0.6-0.9 est courant). Cela transforme une sélection oui/non en un score de preuve.

## Exemple détaillé

Données synthétiques (graine 0 de l'exercice) : $n=100$, $d=10$ colonnes gaussiennes standardisées, poids vrais $(3,-2,0,0,1.5,0,0,0,0,0)$, écart-type du bruit $0.5$.

- $\lambda_{\max}=2.9795$ : le modèle vide pour tout $\lambda$ plus grand.
- Les nombres de coefficients non nuls le long de $\lambda=\lambda_{\max}\times(1,0.5,0.2,0.05)$ sont $0,2,3,3$.
- À $0.2\lambda_{\max}$, l'ensemble actif est exactement la vérité $\{0,1,4\}$ avec des poids **rétrécis** $(2.464,-1.309,0.777)$ contre des moindres carrés ordinaires proches de $(3.1,-2.1,1.5)$ : la sélection coûte du biais. À $\lambda=0.001\lambda_{\max}$, neuf des dix coefficients sont non nuls (ceux du bruit minuscules) et la solution est proche des moindres carrés.
- Vérification KKT à $0.2\lambda_{\max}$ : les gradients actifs valent $\pm0.2\lambda_{\max}=\pm0.596$ et tous les autres sont en dessous.
- Sélection par stabilité (50 demi-échantillons, $\lambda=0.03\lambda_{\max}$) : les variables 0, 1, 4 sont sélectionnées dans $100\%$ des sous-échantillons, les variables de bruit dans $20$-$36\%$. À $\lambda=0.01\lambda_{\max}$, les variables de bruit sont sélectionnées dans $58$-$80\%$ : une pénalité trop petite sélectionne du rebut.
- Information mutuelle : pour $y=x^2+0.1\varepsilon$ avec $x\sim\mathcal N(0,1)$, 2000 échantillons, la corrélation est de $0.008$ mais la MI avec 8 classes de quantiles est de $1.02$ nats ; une colonne de bruit indépendante obtient $0.012$.

## Implémentation

```python
import numpy as np

def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def lasso_cd(X, y, lam, n_iter=500, tol=1e-10):
    n, d = X.shape
    w = np.zeros(d); r = y.astype(float).copy(); z = (X ** 2).sum(0) / n
    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            r += X[:, j] * w[j]                       # remove coordinate j
            w[j] = soft_threshold(X[:, j] @ r / n, lam) / z[j]
            r -= X[:, j] * w[j]                       # put it back
        if np.max(np.abs(w - w_old)) < tol: break
    return w
```

Bibliothèques : `sklearn.linear_model.Lasso`/`LassoCV`/`lasso_path` (note : l'objectif de scikit-learn utilise la même échelle $1/(2n)$), `ElasticNet` (mélange de L1 et L2, plus stable avec des groupes corrélés), `sklearn.feature_selection.mutual_info_regression`, `RFE`, `SequentialFeatureSelector`.

## Dimensions

`X: (n, d)`, `y: (n,)`, `w: (d,)`, résidu `r: (n,)`, `z: (d,)` ; un chemin est `(n_lambdas, d)` ; les fréquences de stabilité `(d,)`.

## Idées reçues

- **« Sélectionné signifie causal ou important. »** La sélection reflète l'utilité prédictive sous ce modèle et cet échantillon ; des variables corrélées se substituent les unes aux autres.
- **« Les coefficients du lasso sont non biaisés. »** Ils sont rétrécis ; réajuster les moindres carrés ordinaires sur l'ensemble sélectionné (lasso relaxé) supprime une partie du biais.
- **« Sélectionner d'abord, puis valider par validation croisée. »** La sélection utilise les étiquettes ; fais-la à l'intérieur de chaque pli, sinon tu verras de la compétence dans du bruit pur ([[cml.15]]).
- **« Le filtrage par corrélation suffit. »** Il rate les effets non linéaires et d'interaction (la MI attrape les premiers, les wrappers ou les arbres les seconds).
- **« Des variables non mises à l'échelle, ce n'est pas grave. »** La pénalité L1 dépend de l'échelle des variables ; standardise.
- **« Les scores de MI sont comparables entre estimateurs. »** Le découpage et la taille d'échantillon changent les valeurs absolues ; utilise-les pour classer.

## Points clés

- Les filtres sont rapides et indépendants du modèle, les wrappers précis et coûteux, les méthodes intégrées un compromis.
- $w_j=S(\rho_j,\lambda)/z_j$ : la pénalité L1 rétrécit et annule les coefficients ; $\lambda_{\max}=\max_j|X_{:j}^\top y|/n$ démarre le chemin.
- La parcimonie vient des coins de la boule L1 ; ridge rétrécit mais ne sélectionne pas.
- Vérifie la stabilité sur des sous-échantillons avant de croire un ensemble sélectionné.

## Explique-le toi-même

1. Dérive la mise à jour par seuillage doux à partir de l'objectif unidimensionnel $\frac12z w^2-\rho w+\lambda|w|$, et explique pourquoi $\lambda_{\max}$ en découle.
2. Pourquoi deux variables parfaitement bonnes et fortement corrélées peuvent-elles rendre instable la sélection par le lasso, et que rapporte la sélection par stabilité ?

## Pour aller plus loin

- Bishop, *PRML*, Sec. 3.1.4 et Fig. 3.3-3.4 (contours du régulariseur ; image de la parcimonie).
- La leçon source `phases/02-ml-fundamentals/18-feature-selection` pour RFE, l'importance d'arbre et le tableau de comparaison.

## Critères de maîtrise

Tu sais classer une méthode de sélection comme filtre, wrapper ou intégrée, dériver le seuillage doux et $\lambda_{\max}$, implémenter la descente de coordonnées et la vérifier avec les conditions KKT, lire un chemin de régularisation et justifier un ensemble sélectionné par des preuves de stabilité.
