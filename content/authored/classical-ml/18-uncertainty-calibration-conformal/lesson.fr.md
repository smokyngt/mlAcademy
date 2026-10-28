---
id: cml.18
title: "Incertitude : réparation de la calibration, ensembles, variance prédictive bayésienne et prédiction conforme"
summary: Sépare le bruit de l'ignorance, répare des probabilités trop confiantes par la mise à l'échelle de température, mesure l'incertitude épistémique avec des ensembles et la régression bayésienne, et enveloppe n'importe quel modèle dans des intervalles de prédiction dont tu peux prouver la couverture.
minutes: 90
teaches: [ml.uncertainty, ml.conformal]
requires: [ml.calibration, ml.generalization, ml.linreg, ml.ridge, prob.rv, stat.testing]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf (Sec. 3.3.2 Predictive distribution, PDF p.176-178; Sec. 3.5.2-3.5.3 evidence and effective number of parameters, PDF p.188-190): the predictive variance as noise plus parameter uncertainty (eq. 3.59) and its shrinkage with data. READ."
  - "math/math for ML.pdf (Sec. 8.2-8.3, PDF p.263-275, the pages on empirical risk, maximum likelihood and negative log-likelihood): the NLL is the training objective that temperature scaling re-minimises on held-out data. READ."
---

Note sur la couverture de la bibliothèque : la bibliothèque lue pour cette leçon couvre la variance prédictive bayésienne et la NLL. Elle ne couvre **pas** la mise à l'échelle de température, les ensembles profonds ni la prédiction conforme ; ces parties sont écrites à partir de connaissances générales, et tout ce qui est numérique est vérifié par le code des exercices.

## Où cela s'inscrit

Un modèle qui dit « sûr à 93 % » et qui a raison 70 % du temps est dangereux : un médecin, une table de trading ou un programme en aval agira sur ce chiffre. [[cml.10]] t'a donné la *mesure* de ce problème (bacs de fiabilité, erreur de calibration attendue). Cette leçon te donne les *réparations* et la moitié manquante de l'histoire : **à quel point le modèle est-il incertain parce qu'il a vu peu de données**, et **comment attacher à n'importe quelle prédiction un intervalle assorti d'une garantie** ? La calibration, les ensembles, la régression bayésienne et la prédiction conforme sont les quatre outils qui servent à poser des barres d'erreur honnêtes sur les prédictions. [[cml.03]] t'a donné la discipline entraînement/validation/test que nous réutilisons ici.

Prérequis : espérance, variance et gaussienne ([[math.09]]) ; maximum de vraisemblance et bootstrap ([[math.13]]) ; régression ridge. Le point de vue des modèles de mélange de [[cml.12]] (données produites par des causes cachées plus du bruit) aide pour l'image des « deux sortes d'aléa » mais n'est pas indispensable.

## L'idée en mots simples

Imagine un prévisionniste météo et un nouveau stagiaire.

- Même le meilleur prévisionniste ne peut pas dire s'il pleuvra demain à 15 h : une part d'aléa est dans le monde lui-même. Collecter plus de données ne l'élimine pas. C'est l'incertitude **aléatoire** (latin *alea*, dé).
- Le stagiaire n'a vu que deux semaines de données d'une seule ville. Sa prévision est vacillante *parce qu'il sait peu de choses*. Donne-lui dix ans de relevés et il s'améliore. C'est l'incertitude **épistémique** (de *episteme*, connaissance).

Un bon système rapporte les deux : de larges barres d'erreur là où le monde est bruité, et une largeur *supplémentaire* aux endroits où il n'a pas vu de données.

Trois questions pratiques en découlent.

1. Si le modèle produit des probabilités systématiquement trop confiantes, peut-on les corriger avec un seul nombre ? (Oui : la mise à l'échelle de température.)
2. Comment voir l'ignorance du modèle ? Entraîne plusieurs modèles sur des données légèrement différentes et regarde où ils divergent. Là où ils s'accordent, les données fixent la réponse ; là où ils divergent, personne ne sait. (Les ensembles ; la régression bayésienne est la version exacte de cette idée.)
3. Peut-on obtenir une promesse du type « la vraie valeur se trouve dans cet intervalle 90 % du temps » sans supposer que les données sont gaussiennes ni que le modèle est juste ? (La prédiction conforme : oui, avec une seule hypothèse, à savoir que le nouveau point ressemble aux points de calibration.)

## Aléatoire contre épistémique, formellement

Suppose $y = f(x) + \varepsilon$ avec un bruit $\varepsilon$ de variance $\sigma^2$. Un modèle ajusté $\hat f$ dépend de l'échantillon d'entraînement, donc $\hat f(x)$ est lui-même aléatoire. Comme le bruit d'une nouvelle observation est indépendant de l'échantillon d'entraînement, les variances s'additionnent :

$$
\operatorname{Var}(y \mid x)=\underbrace{\sigma^2}_{\text{aleatoric}}+\underbrace{\operatorname{Var}\big(\hat f(x)\big)}_{\text{epistemic}} .
$$

Le premier terme reste le même quelle que soit la quantité de données collectées ; le second diminue quand l'échantillon grandit (c'est le terme de variance de la décomposition biais-variance de [[cml.03]]). Mise en garde : la séparation est un énoncé de *modélisation*. Ce que tu appelles bruit dépend des variables que tu as incluses ; ajoute une variable et une partie du bruit aléatoire devient explicable.

## Réparer la calibration

Rappel de [[cml.10]] : un classifieur est **calibré** si, parmi toutes les entrées pour lesquelles il dit « p = 0,8 », environ 80 % sont effectivement positives. Les réseaux entraînés avec l'entropie croisée pendant de nombreuses époques ont tendance à être **trop confiants** : par exemple une précision de 68 % alors que la probabilité maximale moyenne est de 86 %.

### Mise à l'échelle de température

Le modèle produit des **logits** $z\in\mathbb{R}^K$ (un nombre réel par classe ; forme $(N,K)$ pour un lot). Le softmax les transforme en probabilités. Introduis un scalaire $T>0$, la **température** :

$$
p_k(T)=\frac{e^{z_k/T}}{\sum_j e^{z_j/T}} .
$$

$T>1$ aplatit la distribution (moins de confiance), $T<1$ l'affûte, $T=1$ est le modèle d'origine. Diviser tous les logits par le même nombre positif ne change jamais quelle classe a le plus grand logit, donc la **précision est inchangée** ; seule la confiance change.

Pour choisir $T$, minimise la log-vraisemblance négative sur un **ensemble de calibration** mis de côté (jamais l'ensemble d'entraînement, où le modèle est déjà trop confiant) :

$$
\mathrm{NLL}(T)=-\frac1N\sum_{i=1}^{N}\log p_{y_i}(T) .
$$

C'est le même objectif de NLL que le maximum de vraisemblance (la somme des $-\log p(y_n\mid x_n)$ ; math for ML Sec. 8.3), mais avec un seul paramètre. Un seul paramètre signifie qu'une **recherche à une dimension** suffit. $\mathrm{NLL}(T)$ est lisse et possède un seul minimum dans la plage pratique, donc la recherche par section dorée fonctionne : on réduit un intervalle $[a,b]$ du rapport d'or $0.618$ à chaque étape, en gardant la partie où la fonction est plus basse. Elle n'a besoin d'aucun gradient.

### Régression isotonique, l'alternative non paramétrique

Si une seule température est trop rigide, ajuste une fonction $g$ **monotone non décroissante** du score brut vers la probabilité calibrée, en minimisant l'erreur quadratique par rapport aux résultats 0/1. L'algorithme des « pool-adjacent-violators » (regroupement des violateurs adjacents) le fait : parcours les points triés par score ; chaque fois que le résultat moyen d'un bloc est inférieur à celui du bloc précédent, fusionne les deux blocs. Le résultat est un escalier. C'est plus flexible que la mise à l'échelle de température mais cela demande plus de données de calibration (des centaines à des milliers de points) et peut surajuster avec peu de données. Aucune des deux méthodes ne répare un modèle dont le *classement* est mauvais : la calibration change la confiance, pas la discrimination (l'AUC est inchangée par toute application strictement croissante).

## Incertitude épistémique par rééchantillonnage : bootstrap et ensembles profonds

**Ensemble bootstrap.** Tire $B$ rééchantillons de l'ensemble d'entraînement avec remise (un tableau d'indices de forme $(N,)$ chacun). Ajuste un modèle par rééchantillon. Pour une entrée de test $x$, tu obtiens $B$ prédictions $\hat f_b(x)$ ; leur écart-type estime $\sqrt{\operatorname{Var}(\hat f(x))}$, la part épistémique. Le bootstrap lui-même est dans [[math.13]].

**Ensemble profond.** Entraîne $M$ réseaux de neurones à partir d'initialisations aléatoires différentes (et de mini-lots mélangés différemment) sur les *mêmes* données. La surface de perte possède de nombreux minima différents et bons, donc les réseaux font des erreurs différentes, et leur désaccord est une estimation d'incertitude bon marché, simple et, en pratique, solide. Si chaque membre produit une moyenne $\mu_m(x)$ et sa propre variance de bruit $\sigma_m^2(x)$, la variance prédictive de l'ensemble est

$$
\sigma^2(x)=\underbrace{\frac1M\sum_m \sigma_m^2(x)}_{\text{mean aleatoric}}+\underbrace{\frac1M\sum_m\big(\mu_m(x)-\bar\mu(x)\big)^2}_{\text{epistemic: spread of the means}},
\qquad \bar\mu=\frac1M\sum_m\mu_m .
$$

Coût : $M$ fois le calcul d'entraînement et d'inférence. Limite : des membres de même architecture peuvent partager un angle mort, donc l'accord n'est pas une preuve de connaissance.

## Régression linéaire bayésienne : la variance prédictive exacte

Ici, l'« ensemble de modèles » est remplacé par une distribution de probabilité sur les poids, et la dispersion peut être calculée en forme close. (Source : Bishop Sec. 3.3.2 ; l'algèbre ci-dessous est détaillée avec mes propres étapes.)

**Modèle.** Variables $\phi(x)\in\mathbb{R}^M$ (par exemple des puissances de $x$) ; matrice de conception $\Phi\in\mathbb{R}^{N\times M}$ ; cibles $t\in\mathbb{R}^N$.

- Vraisemblance : $t_n = w^\top\phi(x_n)+\varepsilon_n$, $\varepsilon_n\sim\mathcal N(0,\beta^{-1})$. Ici $\beta=1/\sigma^2$ est la **précision du bruit**.
- A priori sur les poids : $w\sim\mathcal N(0,\alpha^{-1}I)$, avec une **précision a priori** $\alpha$. (Cet a priori produit la régression ridge, comme on le montre plus bas.)

**A posteriori.** À une constante près, la log-postérieure vaut

$$
-\tfrac{\beta}{2}\|t-\Phi w\|^2-\tfrac{\alpha}{2}\|w\|^2 .
$$

C'est quadratique en $w$, donc la loi a posteriori est gaussienne. Développe et regroupe les termes en $w$ :

$$
-\tfrac12\Big[w^\top(\alpha I+\beta\Phi^\top\Phi)w-2\beta\,w^\top\Phi^\top t\Big]+\text{const}.
$$

Le logarithme de $\mathcal N(m_N,S_N)$ est $-\tfrac12[w^\top S_N^{-1}w-2w^\top S_N^{-1}m_N]+\text{const}$. En identifiant les coefficients :

$$
S_N^{-1}=\alpha I+\beta\Phi^\top\Phi,\qquad m_N=\beta S_N\Phi^\top t .
$$

Remarque que $m_N$ est exactement la solution ridge avec la pénalité $\lambda=\alpha/\beta$, donc la régression ridge ([[cml.01]]) est la *moyenne* d'un modèle bayésien ; la covariance $S_N\in\mathbb{R}^{M\times M}$ est ce que ridge jette.

**Distribution prédictive.** Pour une nouvelle entrée de variables $\phi=\phi(x)$, la nouvelle cible est $t^\ast=w^\top\phi+\varepsilon$ avec $w$ et $\varepsilon$ indépendants. Une fonction linéaire de gaussiennes indépendantes est gaussienne, donc seules la moyenne et la variance sont nécessaires :

$$
\mathbb{E}[t^\ast]=m_N^\top\phi,\qquad
\operatorname{Var}(t^\ast)=\operatorname{Var}(w^\top\phi)+\operatorname{Var}(\varepsilon)=\phi^\top S_N\phi+\beta^{-1}.
$$

Donc

$$
\boxed{\sigma_N^2(x)=\underbrace{\beta^{-1}}_{\text{aleatoric}}+\underbrace{\phi(x)^\top S_N\,\phi(x)}_{\text{epistemic}}}
$$

qui est l'équation (3.59) de Bishop. Propriétés notées là : le terme épistémique est le plus petit près des entrées d'entraînement ; quand $N\to\infty$ il tend vers zéro, ne laissant que le bruit pur $\beta^{-1}$. Bishop avertit aussi qu'avec des fonctions de base localisées, le terme s'annule loin des centres des fonctions de base, ce qui rend le modèle trop confiant en extrapolation ; les processus gaussiens évitent cela.

## Intervalles de prédiction

Un **intervalle de prédiction** de niveau $1-\alpha$ est un intervalle $C(x)$ avec $P(y\in C(x))\ge 1-\alpha$. Ne le confonds pas avec un **intervalle de confiance** pour la *moyenne* $f(x)$, qui omet le terme de bruit et est plus étroit. Avec la distribution prédictive gaussienne, l'intervalle à 90 % est $\mu\pm1.645\,\sigma_N$. Cela n'est correct que si le modèle, le bruit gaussien et l'a priori sont tous justes. La prédiction conforme supprime cette exigence.

## Prédiction conforme fractionnée (split)

**Mise en place.** Sépare les données en un ensemble d'*entraînement propre* (ajuste n'importe quel modèle $\hat f$) et un ensemble de *calibration* de $n$ points $(x_i,y_i)$ non utilisés pour l'ajustement. Choisis un **score** qui mesure à quel point le modèle se trompe sur un point ; le plus simple est le résidu absolu $s_i=|y_i-\hat f(x_i)|$.

**Recette.** Soit $k=\lceil (n+1)(1-\alpha)\rceil$ et soit $\hat q$ le $k$-ième plus petit score de calibration (si $k>n$, pose $\hat q=\infty$ : pas assez de données pour ce niveau). Pour un nouveau $x$, renvoie

$$
C(x)=\big[\hat f(x)-\hat q,\ \hat f(x)+\hat q\big].
$$

**Théorème (couverture à échantillon fini).** Si les $n$ points de calibration et le nouveau point de test sont *échangeables* (tout réordonnancement a la même loi jointe ; des données i.i.d. conviennent) et si les scores n'ont pas d'ex æquo, alors

$$
1-\alpha\ \le\ P\big(y_{n+1}\in C(x_{n+1})\big)\ \le\ 1-\alpha+\frac1{n+1}.
$$

**Preuve.** Le modèle $\hat f$ a été ajusté sans ces $n+1$ points, donc on le traite comme fixé ; les scores $s_1,\dots,s_n,s_{n+1}$ (le dernier pour le point de test) sont alors échangeables. L'événement $y_{n+1}\in C(x_{n+1})$ est exactement $s_{n+1}\le\hat q$, c'est-à-dire que le score de test n'est pas plus grand que le $k$-ième plus petit des $n$ autres scores. Regarde le rang de $s_{n+1}$ parmi les $n+1$ scores (rang 1 = le plus petit). Par échangeabilité et absence d'ex æquo, chacun des $n+1$ rangs a la probabilité $\frac1{n+1}$. Si le rang est $r\le k$, alors moins de $k$ scores de calibration se trouvent en dessous, donc il ne dépasse pas le $k$-ième plus petit score de calibration ; si $r>k$, au moins $k$ scores de calibration se trouvent en dessous et il le dépasse. Donc

$$
P(s_{n+1}\le\hat q)=\frac{k}{n+1}.
$$

Comme $(n+1)(1-\alpha)\le k<(n+1)(1-\alpha)+1$, diviser par $n+1$ donne les deux bornes. $\square$

Remarques. (i) La garantie est **marginale** : moyennée sur les ensembles de calibration et les points de test, pas « pour chaque $x$ ». Là où le bruit est plus grand, l'intervalle peut sous-couvrir, là où il est plus petit il sur-couvre. Des correctifs (scores normalisés, régression quantile conformalisée) existent mais dépassent cette leçon. (ii) Elle vaut pour n'importe quel modèle ; un mauvais modèle donne seulement des intervalles *larges*. (iii) Le $+1$ rend la garantie exacte à échantillon fini ; le simple quantile $(1-\alpha)$ de $n$ scores sous-couvre légèrement. (iv) L'échangeabilité échoue pour des séries temporelles dérivantes, et sous un décalage de covariables la garantie est perdue.

## Exemple détaillé

Tous les nombres sont reproduits par le code des exercices (avec graine).

1. **Mise à l'échelle de température.** Problème à 3 classes, $N=2000$, logits égaux à $2.5\times$ les vrais logits (un modèle trop confiant). À $T=1$ : NLL $0.952$, ECE (10 bacs) $0.181$, précision $0.680$, probabilité maximale moyenne $0.855$. La recherche 1-D renvoie $T^\ast=2.48$ (proche du $2.5$ que nous avons injecté), NLL $0.715$, ECE $0.023$, probabilité maximale moyenne $0.689$, qui correspond à la précision. La précision est identique avant et après.
2. **Ensemble contre bayésien.** Données : $y=\sin 3x+0.15\varepsilon$ sur deux groupes $x\in[-1,-0.2]\cup[0.6,1]$ (40 points chacun), variables polynomiales de degré 7. Écart-type bootstrap ($B=200$) de la prédiction en $x=-0.6$ (dans les données), $0.2$ (le trou) et $0.8$ (dans les données) : $0.023,\ 0.119,\ 0.046$ (pénalité ridge $10^{-3}$, graine 3). Écart-type épistémique bayésien $\sqrt{\phi^\top S_N\phi}$ aux mêmes points ($\alpha=1$, $\beta=1/0.15^2$) : $0.033,\ 0.097,\ 0.034$ ; écart-type total $0.154,\ 0.178,\ 0.154$ contre un bruit pur de $0.15$. Les deux méthodes disent que le modèle est ignorant dans le trou, avec le même ordre de grandeur. Elles ne sont pas égales : le bootstrap n'a pas d'a priori, et aucune des deux ne tient compte d'une mauvaise spécification.
3. **Couverture conforme.** Bruit à queues lourdes ($t_3$ avec une échelle dépendant de l'entrée), une droite ajustée sur 100 points d'entraînement, $n=100$ points de calibration, $\alpha=0.1$, donc $k=\lceil 101\cdot 0.9\rceil=91$ et la théorie donne une couverture attendue de $91/101=0.901$. Sur 300 essais indépendants, chacun jugé sur 2000 nouveaux points de test, la couverture moyenne était de $0.904$ ; les essais individuels allaient de $0.796$ vers le haut avec un écart-type de $0.030$, ce qui est la dispersion à attendre d'un seul ensemble de calibration. Avec $n=19$ : $k=\lceil 20\cdot0.9\rceil=18$, couverture attendue exactement $18/20=0.90$, observée $0.904$ sur 300 essais.

## Implémentation

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)       # (N, K)

def nll(T, z, y):
    p = softmax(z / T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

def golden(f, lo, hi, iters=60):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if f(c) < f(d): b = d
        else:           a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return (a + b) / 2

def blr_posterior(Phi, t, alpha, beta):
    SN = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)   # (M, M)
    return beta * SN @ Phi.T @ t, SN                                        # (M,), (M, M)

def conformal_q(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]
```

En production, scikit-learn propose `CalibratedClassifierCV` (sigmoïde et isotonique) et le paquet MAPIE implémente les intervalles conformes ; les utilisateurs de PyTorch écrivent en général le court calibrateur de température ci-dessus avec L-BFGS. Je n'ai pas lu la documentation de ces bibliothèques pour cette leçon, donc vérifie leurs API actuelles.

## Formes

- Logits $z$ : $(N,K)$ ; étiquettes $y$ : $(N,)$ entiers ; $T$ : scalaire.
- $\Phi$ : $(N,M)$ ; $S_N$ : $(M,M)$ ; $m_N$ : $(M,)$ ; pour $P$ points de requête $\Phi_\ast$ : $(P,M)$ et variances prédictives $(P,)$ via `einsum('ij,jk,ik->i', Phi_star, SN, Phi_star)`.
- Prédictions bootstrap : $(B,P)$ ; dispersion = `.std(axis=0)`, forme $(P,)$.
- Scores de calibration : $(n,)$ ; $\hat q$ : scalaire.

## Idées reçues

- « Calibré veut dire précis. » Non. Un modèle qui prédit toujours le taux de base est calibré et inutile ; calibration et discrimination sont distinctes ([[cml.10]]).
- « La mise à l'échelle de température change les prédictions. » La classe arg-max ne change pas ; seule la confiance change.
- « Ajuste $T$ sur l'ensemble d'entraînement. » Le sur-excès de confiance apparaît sur l'ensemble d'entraînement ; ajuste sur des données mises de côté.
- « La dispersion de l'ensemble est l'incertitude totale. » Elle estime la partie épistémique ; ajoute le bruit pour un intervalle de prédiction.
- « Un intervalle conforme à 90 % couvre 90 % des points pour chaque $x$. » La garantie est marginale ; vérifie la couverture conditionnelle séparément.
- « La prédiction conforme demande un bon modèle. » La validité non ; l'efficacité (intervalles étroits) oui.
- « L'incertitude bayésienne est toujours juste. » Elle est juste étant donnés le modèle et l'a priori ; sous un mauvais modèle elle est sûre d'elle à tort, d'où l'intérêt d'un contrôle conforme par-dessus.

## Points clés

- Variance prédictive = bruit (irréductible) + ignorance sur les paramètres (diminue avec les données).
- Mise à l'échelle de température : un paramètre, ajusté par minimisation 1-D de la NLL sur données mises de côté, précision intacte.
- Les ensembles estiment l'incertitude épistémique à partir du désaccord ; la régression linéaire bayésienne la donne exactement : $\sigma_N^2=\beta^{-1}+\phi^\top S_N\phi$.
- Conforme fractionnée : utilise le $\lceil(n+1)(1-\alpha)\rceil$-ième plus petit score de calibration ; la couverture est d'au moins $1-\alpha$ à échantillon fini sous échangeabilité, par un argument de symétrie des rangs.

## Explique-le toi-même

1. Pourquoi la variance prédictive bayésienne contient-elle deux termes, et lequel peux-tu réduire en collectant des données ?
2. Pourquoi la prédiction conforme a-t-elle besoin du « +1 » dans $n+1$, et qu'est-ce qui casserait si les points de calibration servaient aussi à ajuster le modèle ?

## Pour aller plus loin

- Bishop, *Pattern Recognition and Machine Learning*, Sec. 3.3 et 3.5 (lues ici) et chap. 6 sur les processus gaussiens (non lu pour cette leçon).
- Écrits à partir de connaissances générales, pas de la bibliothèque : mise à l'échelle de température (Guo et al., 2017), ensembles profonds (Lakshminarayanan et al., 2017), prédiction conforme (Vovk et al. ; tutoriel d'Angelopoulos et Bates). Consulte ces articles pour les détails exacts.

## Critères de maîtrise

- Calculer la NLL, trouver la meilleure température et montrer que l'ECE diminue alors que la précision est inchangée.
- Dériver $S_N$, $m_N$ et la variance prédictive à partir de la log-postérieure.
- Énoncer le théorème conforme avec son hypothèse, reproduire la preuve par les rangs et vérifier la couverture par simulation.
- Expliquer avec un exemple pourquoi la couverture marginale n'est pas la couverture conditionnelle.
