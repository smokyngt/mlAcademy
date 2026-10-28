---
id: math.13
title: "Estimation et tests d'hypothèses"
summary: "Comment apprendre des paramètres à partir de données (maximum de vraisemblance, MAP), jusqu'où faire confiance à la réponse (biais, variance, intervalles de confiance, bootstrap), et comment distinguer une vraie amélioration d'un modèle du simple bruit."
minutes: 80
teaches: [stat.estimation, stat.testing]
requires: [prob.basics, prob.rv, ml.generalization, ml.ridge]
sources:
  - "math/math for ML.pdf, Sec. 8.3 Parameter Estimation, 8.3.1 Maximum Likelihood Estimation (PDF p.271-273) and 8.3.2 MAP Estimation (PDF p.275-276): negative log-likelihood, Gaussian likelihood gives least squares (Example 8.5), posterior proportional to likelihood times prior, zero-mean Gaussian prior as regularisation. READ."
  - "math/math for ML.pdf, Sec. 6.4.3 (PDF p.199, book p.193): raw-score variance and its numerical caveat; used here for the variance definition. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.2.4 Gaussian distribution (PDF p.47): ML mean and variance, E[sigma^2_ML] = (N-1)/N sigma^2 and the unbiased correction. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.2.5 Curve fitting re-visited (PDF p.49-50): sum-of-squares from Gaussian noise, Gaussian prior on weights gives the regularised error with lambda = alpha/beta. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/15-statistics-for-ml/docs/en.md: hypothesis tests, p-values, confidence intervals, paired t-test, effect size, multiple comparisons, bootstrap for model comparison. READ."
---

## Où cela s'inscrit

Tu as entraîné deux classifieurs. Le modèle A obtient 0.87 sur ton jeu de test et le modèle B obtient 0.89. Livres-tu B ? Tout ce qui figure dans cette leçon existe pour répondre honnêtement à cette question. Le jeu de test est un petit échantillon aléatoire de toutes les données que le modèle verra jamais, donc chaque nombre que tu en calcules (exactitude, perte, poids appris) est lui-même une quantité aléatoire : refais l'expérience avec un autre échantillon et il change. **La statistique est l'étude de la confiance que l'on peut accorder à un nombre calculé à partir de données.**

La leçon comporte deux moitiés. D'abord l'*estimation* : comment transformer des données en valeurs de paramètres, et que veut même dire « une bonne estimation » ? Tu verras que les fonctions de perte de la leçon 8 (erreur quadratique moyenne, entropie croisée) ne sont pas arbitraires : elles découlent du maximum de vraisemblance, et la régularisation L2 découle d'un a priori gaussien. Ensuite les *tests* : comment quantifier l'incertitude (intervalles de confiance, bootstrap) et décider si une différence entre deux modèles est réelle.

Elle s'appuie sur les probabilités et Bayes (leçon 8), les variables aléatoires et la gaussienne (leçon 9), et l'idée de généralisation (le matériel antérieur sur le biais et la variance). Elle alimente la leçon 14 (échantillonnage, la manière d'explorer les a posteriori bayésiens quand il n'existe pas de formule).

## L'idée en mots simples

- **Un estimateur est une recette ; une estimation est un résultat de la recette.** Pense à mesurer la hauteur d'une tour en marchant autour. La recette « moyenne de mes résultats de pas » est l'estimateur. Si tu refaisais la mesure demain, tu obtiendrais une réponse légèrement différente : l'estimation est aléatoire parce que les données sont aléatoires. Un bon estimateur a deux vertus : il n'est pas *systématiquement* à côté (faible **biais**) et il ne saute pas beaucoup d'un jeu de données à l'autre (faible **variance**). Un joueur de fléchettes dont les fléchettes se regroupent serrées mais loin du centre a une faible variance et un fort biais ; celui dont les fléchettes se dispersent uniformément autour du centre a une forte variance et un faible biais.
- Le **maximum de vraisemblance** demande : *parmi tous les mondes dans lesquels je pourrais vivre, lequel aurait rendu mes données observées le moins surprenantes ?* Tu essaies chaque valeur candidate du paramètre et tu la notes selon la probabilité qu'elle donne aux données ; tu gardes le gagnant. C'est « l'explication qui colle le mieux ».
- Le **MAP (maximum a posteriori)** ajoute du bon sens avant de regarder les données : *« je crois déjà que les poids sont probablement petits ».* L'a priori agit comme un doux élastique qui tire l'estimation vers zéro. Avec peu de données, l'élastique compte beaucoup ; avec beaucoup de données, les données l'emportent. Cet élastique est exactement la régularisation ridge.
- Un **intervalle de confiance** est une barre d'erreur honnête. Il dit : « si je pouvais refaire toute l'expérience de nombreuses fois, des intervalles construits de cette manière captureraient la vraie valeur 95 % du temps. »
- Un **test d'hypothèse** est une preuve par l'absurde rendue probabiliste. Suppose l'explication ennuyeuse (les deux modèles sont aussi bons l'un que l'autre). Demande : à quel point les données que j'ai vues seraient-elles surprenantes si l'explication ennuyeuse était vraie ? Si elles le sont beaucoup, doute de l'explication ennuyeuse.
- Le **bootstrap** répond à « de combien mon nombre aurait-il changé avec un autre échantillon ? » en faisant comme si ton échantillon *était* le monde et en retirant dedans avec remise, des milliers de fois.

## Estimateurs, biais et variance

Soient $x_1,\dots,x_N$ des tirages indépendants d'une distribution de paramètre inconnu $\theta$. Un **estimateur** est une fonction $\hat\theta = g(x_1,\dots,x_N)$. Comme les $x_i$ sont aléatoires, $\hat\theta$ est une variable aléatoire avec sa propre moyenne et sa propre variance.

$$
\operatorname{bias}(\hat\theta)=\mathbb E[\hat\theta]-\theta,\qquad
\operatorname{Var}(\hat\theta)=\mathbb E\big[(\hat\theta-\mathbb E\hat\theta)^2\big].
$$

L'erreur quadratique espérée se décompose en les deux (c'est le compromis biais-variance de la leçon sur la généralisation, un niveau plus bas) :

$$
\mathbb E\big[(\hat\theta-\theta)^2\big]=\operatorname{bias}^2+\operatorname{Var}.
$$

*Pourquoi :* écris $\hat\theta-\theta=(\hat\theta-\mathbb E\hat\theta)+(\mathbb E\hat\theta-\theta)$ et élève au carré. Le terme croisé est $2(\mathbb E\hat\theta-\theta)\,\mathbb E[\hat\theta-\mathbb E\hat\theta]$, et la dernière espérance est nulle par définition de la moyenne, donc seuls les deux carrés survivent.

### La moyenne empirique et son erreur standard

Pour la moyenne empirique $\bar x=\frac1N\sum x_i$ de vraie moyenne $\mu$ et de variance $\sigma^2$ : $\mathbb E[\bar x]=\mu$ (non biaisée), et comme les tirages sont indépendants, $\operatorname{Var}(\bar x)=\sigma^2/N$. L'écart-type de l'estimateur, $\sigma/\sqrt N$, s'appelle l'**erreur standard**. Pour diviser ta barre d'erreur par deux, il te faut quatre fois plus de données. Garde cette règle en $1/\sqrt N$ en tête : elle revient dans la leçon sur Monte Carlo.

### Pourquoi l'estimation de la variance divise par $N-1$

Bishop (Sec. 1.2.4) montre que la variance du maximum de vraisemblance $\sigma^2_{ML}=\frac1N\sum(x_n-\mu_{ML})^2$ vérifie $\mathbb E[\sigma^2_{ML}]=\frac{N-1}{N}\sigma^2$ : elle est biaisée vers le bas. L'intuition : les points de données sont, en moyenne, plus proches de *leur propre* moyenne empirique que de la vraie moyenne (la moyenne empirique est par construction le point qui minimise les distances au carré), donc les écarts au carré à $\bar x$ sont trop petits. Multiplier par $\frac{N}{N-1}$ corrige cela :

$$
s^2=\frac1{N-1}\sum_{n=1}^N (x_n-\bar x)^2 .
$$

Pour $N=5$ le facteur est $4/5=0.8$ : une simulation de 200 000 jeux de données de cinq nombres normaux standard donne une moyenne de $0.802$ pour l'estimateur divisant par $N$ et de $1.002$ pour l'estimateur divisant par $(N-1)$ (reproduit dans l'exercice). Note que non biaisé ne veut pas dire meilleur : un estimateur avec un peu de biais et beaucoup moins de variance peut avoir une erreur totale plus petite, ce qui est précisément la raison pour laquelle la régularisation aide.

## Maximum de vraisemblance

Modélise les données par une densité $p(x\mid\theta)$. En fixant les données et en faisant varier $\theta$, $p(x\mid\theta)$ est la **vraisemblance**. Pour des exemples indépendants, la vraisemblance du jeu de données entier est un produit, et un produit de nombreux petits nombres est à la fois pénible à optimiser et hostile numériquement (voir la leçon 15). Prendre des logs le transforme en somme. Deisenroth et al. (Sec. 8.3.1) minimisent la **log-vraisemblance négative** :

$$
L(\theta)=-\log p(Y\mid X,\theta)=-\sum_{n=1}^{N}\log p(y_n\mid x_n,\theta),\qquad
\hat\theta_{ML}=\arg\min_\theta L(\theta).
$$

(Le signe moins n'est qu'une convention pour que « maximiser la vraisemblance » devienne « minimiser une perte ».)

### Démonstration : un bruit gaussien donne les moindres carrés

Suppose $y_n=x_n^\top\theta+\varepsilon_n$ avec un bruit $\varepsilon_n\sim\mathcal N(0,\sigma^2)$, donc $p(y_n\mid x_n,\theta)=\mathcal N(y_n\mid x_n^\top\theta,\sigma^2)$. Insère la densité gaussienne :

$$
L(\theta)=-\sum_{n}\log\Big[\tfrac1{\sqrt{2\pi\sigma^2}}\exp\Big(-\tfrac{(y_n-x_n^\top\theta)^2}{2\sigma^2}\Big)\Big].
$$

Le log d'un produit est une somme de logs, et $\log\exp(u)=u$, donc chaque terme devient $\frac{(y_n-x_n^\top\theta)^2}{2\sigma^2}+\log\sqrt{2\pi\sigma^2}$ :

$$
L(\theta)=\frac1{2\sigma^2}\sum_{n=1}^{N}(y_n-x_n^\top\theta)^2+N\log\sqrt{2\pi\sigma^2}.
$$

Le second terme ne dépend pas de $\theta$, et $\frac1{2\sigma^2}$ est une constante positive qui ne peut pas déplacer le minimiseur. Donc le maximum de vraisemblance avec un bruit gaussien **est** les moindres carrés (l'exemple 8.5 du livre et l'éq. 1.62 de Bishop). La même recette pour une étiquette de Bernoulli donne l'entropie croisée. C'est la réponse à « pourquoi cette perte ? » : chaque perte standard est la log-vraisemblance négative d'un modèle de bruit.

Pour un simple échantillon gaussien de $\mu$ et $\sigma^2$ inconnus, annuler les dérivées donne la moyenne empirique et la variance divisée par $N$ (Bishop éq. 1.55-1.56), d'où venait le biais ci-dessus. Propriétés (la remarque de Deisenroth, p.269-270) : l'EMV est *consistant* (il converge vers la vérité quand $N\to\infty$) et son erreur décroît comme $1/\sqrt N$, mais pour $N$ petit il peut surapprendre.

## MAP et pourquoi la régularisation L2 est un a priori gaussien

Le maximum de vraisemblance fait entièrement confiance aux données. Le théorème de Bayes (leçon 8) nous permet d'encoder aussi une croyance a priori $p(\theta)$ :

$$
p(\theta\mid x)=\frac{p(x\mid\theta)\,p(\theta)}{p(x)}\ \propto\ p(x\mid\theta)\,p(\theta).
$$

Le dénominateur ne dépend pas de $\theta$, donc pour trouver le meilleur $\theta$ on peut l'ignorer. L'estimation MAP minimise la **log-a-posteriori négative** :

$$
\hat\theta_{MAP}=\arg\min_\theta\ \big[-\log p(x\mid\theta)-\log p(\theta)\big].
$$

### Démonstration : un a priori gaussien donne ridge

Prends la vraisemblance à bruit gaussien ci-dessus et l'a priori $\theta\sim\mathcal N(0,\tau^2 I)$, donc $-\log p(\theta)=\frac1{2\tau^2}\|\theta\|^2+\text{const}$. En supprimant les constantes :

$$
-\log p(\theta\mid x)\ =\ \frac1{2\sigma^2}\sum_n (y_n-x_n^\top\theta)^2+\frac1{2\tau^2}\|\theta\|_2^2 .
$$

Multiplie par la constante positive $2\sigma^2$ (cela ne change pas le minimiseur) :

$$
\sum_n (y_n-x_n^\top\theta)^2+\lambda\|\theta\|_2^2,\qquad \lambda=\frac{\sigma^2}{\tau^2}.
$$

C'est exactement la régression ridge. Bishop (éq. 1.67) arrive à la même forme avec des précisions : $\lambda=\alpha/\beta$ où $\alpha=1/\tau^2$ est la précision a priori et $\beta=1/\sigma^2$ la précision du bruit. En mots : **une croyance plus forte que les poids sont proches de zéro (petit $\tau$) donne un $\lambda$ plus grand ; des données plus bruitées (grand $\sigma$) donnent aussi un $\lambda$ plus grand**, parce qu'on doit s'appuyer sur l'a priori quand les données sont peu fiables. Un a priori de Laplace donnerait la pénalité L1 à la place. Souviens-toi toutefois que le MAP renvoie toujours un seul point ; le traitement pleinement bayésien garde tout l'a posteriori (Deisenroth Sec. 8.4, non couvert ici).

## Intervalles de confiance

Une estimation ponctuelle sans barre d'erreur est une demi-vérité. Pour une moyenne empirique de $\sigma$ inconnu, remplace $\sigma$ par $s$ et utilise la loi de Student $t$ à $N-1$ degrés de liberté (elle a des queues plus épaisses que la normale pour payer l'estimation de $\sigma$) :

$$
\bar x\ \pm\ t_{N-1,\,0.975}\cdot\frac{s}{\sqrt N}.
$$

Lecture correcte : la *procédure* capture la vraie valeur dans 95 % des expériences répétées. Elle ne dit **pas** qu'il y a 95 % de probabilité que la vraie valeur se trouve dans l'unique intervalle que tu as calculé (la vraie valeur est fixe ; c'est l'intervalle qui est aléatoire).

Pour une exactitude de test $\hat p$ mesurée sur $N$ exemples indépendants, l'erreur standard est $\sqrt{\hat p(1-\hat p)/N}$ (une moyenne de Bernoulli). Avec $\hat p=0.87$ et $N=500$ elle vaut $0.0150$, donc un intervalle à 95 % est d'environ $\pm1.96\times0.0150=\pm0.029$ : **0.87 et 0.89 sont statistiquement indistinguables sur un jeu de test de 500 exemples.** Ce seul calcul évite de nombreux mauvais déploiements.

## Tests d'hypothèses et p-valeurs

1. Énonce une hypothèse nulle $H_0$ (l'explication ennuyeuse, par ex. « les deux modèles ont le même score espéré ») et une alternative $H_1$.
2. Choisis une statistique de test $T$ dont tu connais la distribution sous $H_0$.
3. La **p-valeur** est $P(\text{a statistic at least as extreme as the observed one}\mid H_0)$.
4. Une petite p-valeur (en dessous d'un $\alpha$ choisi à l'avance, couramment $0.05$) signifie que les données sont surprenantes sous $H_0$, donc tu la rejettes.

### Pièges (chacun a causé de vraies mauvaises décisions)

- Une p-valeur n'est **pas** $P(H_0\mid\text{data})$. C'est un énoncé sur les données sachant $H_0$, jamais l'inverse (la confusion est la même que confondre $P(A\mid B)$ et $P(B\mid A)$ dans la leçon sur Bayes).
- « Non significatif » ne veut pas dire « pas de différence ». Cela peut simplement signifier trop peu de données.
- **Signification statistique contre signification pratique.** Avec un million d'exemples de test, une amélioration de 0.03 % peut avoir $p=0.001$ et rester sans valeur. Rapporte une *taille d'effet* (par exemple la différence moyenne dans les unités propres de la métrique, ou le $d$ de Cohen = différence divisée par l'écart-type poolé) à côté de la p-valeur.
- **Comparaisons multiples.** Lance $m$ tests indépendants au niveau $\alpha$ avec toutes les nulles vraies et $P(\text{at least one false positive})=1-(1-\alpha)^m$. Pour $m=20$, $\alpha=0.05$ cela fait $1-0.95^{20}=0.64$. Essayer 20 réglages d'hyperparamètres et rapporter le meilleur, c'est exactement cela. Le correctif de Bonferroni teste chacun à $\alpha/m=0.0025$.
- **Coup d'œil et réutilisation du jeu de test.** Si tu choisis un modèle *parce qu'*il a fait le mieux sur le jeu de test, ce score de test est biaisé de façon optimiste. Garde un jeu de test final intact.

## Comparer deux modèles équitablement

Règles qui découlent de ce qui précède : évalue les deux modèles sur les **mêmes** découpages de données ; regarde les **différences appariées** (un pli difficile abaisse les deux scores, et l'appariement annule cette difficulté partagée) ; rapporte un intervalle, pas un seul nombre.

**Test t apparié.** Avec les différences par pli $d_i=b_i-a_i$ ($i=1..K$), $t=\bar d\,/\,(s_d/\sqrt K)$ avec $K-1$ degrés de liberté, un test à un échantillon de $\bar d=0$. Réserve : les plis de validation croisée partagent des données d'entraînement, donc les $d_i$ ne sont pas vraiment indépendants et le test est quelque peu optimiste ; considère-le comme un guide, pas un verdict.

**Bootstrap.** Rééchantillonne avec remise les $K$ différences (ou, avec un grand jeu de test, les *exemples* de test), recalcule la statistique, répète $B$ fois, et lis les 2.5e et 97.5e percentiles des résultats comme un intervalle à 95 %. Il n'a besoin d'aucune formule et fonctionne pour des métriques délicates (AUC, F1).

## Exemple détaillé

Dix plis de validation croisée, exactitudes des modèles A et B :

| pli | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | .842 | .851 | .839 | .860 | .848 | .855 | .845 | .852 | .840 | .858 |
| B | .850 | .862 | .841 | .871 | .850 | .866 | .858 | .861 | .847 | .865 |

Les différences $d_i=B_i-A_i$ ont pour moyenne $\bar d=0.0081$ et pour écart-type empirique $s_d=0.003755$. L'erreur standard est $s_d/\sqrt{10}=0.001187$, donc $t=0.0081/0.001187=6.82$ avec 9 degrés de liberté : bien au-delà de la valeur critique à 5 % $2.262$, donc l'amélioration n'est pas du bruit. Un bootstrap de 5000 rééchantillonnages (initialisé par `default_rng(0)`, tirant 10 différences à chaque fois) donne un intervalle à 95 % d'environ $[0.0058,\ 0.0101]$ pour le gain moyen, qui exclut zéro. Conclusion : B est meilleur d'environ 0.8 point, plausiblement entre 0.6 et 1.0. Remarque que les données sont ici *appariées* : B a battu A sur les dix plis.

Un second petit exemple pour le MAP. Régression à un poids passant par l'origine avec $x=(1,2,3)$, $y=(1.2,1.9,3.4)$. Minimiser $\sum(y-wx)^2+\lambda w^2$ donne $\hat w=\sum xy\,/\,(\sum x^2+\lambda)$. Ici $\sum xy=1.2+3.8+10.2=15.2$ et $\sum x^2=14$, donc $\hat w=15.2/(14+\lambda)$ : $1.0857$ pour $\lambda=0$ (maximum de vraisemblance), $1.0133$ pour $\lambda=1$ et $0.8$ pour $\lambda=5$. Un $\lambda$ plus grand (a priori plus fort) tire le poids vers zéro.

## Implémentation

```python
import numpy as np

def bootstrap_ci(d, B=5000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d)).mean() for _ in range(B)])
    a = (1 - level) / 2
    return np.percentile(means, [100 * a, 100 * (1 - a)])

def ridge_1d(x, y, lam):
    return (x @ y) / (x @ x + lam)      # MAP with Gaussian prior, lam = sigma^2 / tau^2
```

## Formes

Matrice de données `X: (N, D)`, cibles `y: (N,)`. Les log-vraisemblances par exemple ont la forme `(N,)` et la perte est leur somme (ou moyenne) opposée, un scalaire. Scores appariés pour $K$ plis : deux tableaux `(K,)`, différences `(K,)`. Un bootstrap tire un tableau d'indices de forme `(B, K)`, donnant `B` statistiques rééchantillonnées.

## Idées fausses courantes

- **« p = 0.03 signifie que l'hypothèse nulle a 3 % de chances d'être vraie. »** Non : c'est la probabilité de telles données si la nulle était vraie.
- **« L'intervalle à 95 % contient la vérité avec une probabilité de 95 %. »** La procédure le fait ; un intervalle unique calculé la contient ou ne la contient pas.
- **« Non biaisé est toujours mieux. »** Les estimateurs régularisés (ridge) acceptent un peu de biais en échange de beaucoup moins de variance, et ont souvent une erreur totale plus petite.
- **« Significatif veut dire important. »** Cela veut seulement dire distinguable de zéro compte tenu de la taille de l'échantillon.
- **« Le bootstrap crée de l'information nouvelle. »** Il ne le peut pas ; il montre seulement à quel point la taille de ton échantillon te limite. Avec un échantillon biaisé ou minuscule, il se trompera avec assurance.

## Points clés à retenir

- Une estimation est aléatoire ; juge les estimateurs par le biais et la variance, avec erreur $=\text{bias}^2+\text{variance}$. L'erreur standard d'une moyenne est $\sigma/\sqrt N$.
- L'EMV minimise la log-vraisemblance négative : un bruit gaussien donne les moindres carrés, Bernoulli donne l'entropie croisée. La variance non biaisée divise par $N-1$.
- Le MAP ajoute un a priori : un a priori gaussien $\mathcal N(0,\tau^2I)$ donne ridge avec $\lambda=\sigma^2/\tau^2$.
- Attache toujours un intervalle à une métrique, compare les modèles sur des plis appariés, corrige les comparaisons multiples, et rapporte des tailles d'effet à côté des p-valeurs.

## Explique-le avec tes mots

1. Un coéquipier dit « la p-valeur est 0.04, donc il y a 96 % de chances que le nouveau modèle soit meilleur ». Explique précisément ce qui ne va pas et ce que signifie réellement 0.04.
2. Sans symboles, explique pourquoi ajouter une pénalité L2 aux moindres carrés revient à croire, avant de voir les données, que les poids sont petits, et ce qui arrive à cette croyance quand tu collectes plus de données.

## Pour aller plus loin

- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*, chap. 8 (EMV, MAP, puis régression linéaire bayésienne complète au chap. 9).
- Bishop, *Pattern Recognition and Machine Learning*, Sec. 1.2.3-1.2.6.
- Dépôt, phase 01, leçon 15 (statistiques pour le ML) pour le test t, le khi-deux et des alternatives non paramétriques comme Wilcoxon.

## Critères de maîtrise

- Démontrer les moindres carrés à partir d'une vraisemblance gaussienne et ridge à partir d'un a priori gaussien, en nommant ce que vaut $\lambda$.
- Calculer un intervalle de confiance fondé sur $t$ et un intervalle bootstrap à la main ou en code.
- Énoncer le sens correct d'une p-valeur et d'un intervalle à 95 %, et nommer trois façons dont les tests dérapent.
- Décider, chiffres à l'appui, si deux scores sur un jeu de test de taille donnée sont distinguables.
