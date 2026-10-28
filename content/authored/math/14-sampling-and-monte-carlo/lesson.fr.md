---
id: math.14
title: "Échantillonnage et Monte Carlo"
summary: "Comment un ordinateur tire des nombres aléatoires selon une distribution choisie, et comment des moyennes de ces tirages remplacent des intégrales impossibles à résoudre à la main."
minutes: 80
teaches: [prob.sampling, prob.montecarlo]
requires: [prob.basics, prob.rv, calc.derivative]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 11.1 Basic Sampling Algorithms (PDF p.546, 548, 551, 553, 555): transformation method z = h(y), Box-Muller, Cholesky for multivariate Gaussians, rejection sampling and its failure in high dimensions, importance sampling with weights r_l = p/q, self-normalised weights, SIR. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 11.2 MCMC (PDF p.557, 559): random-walk behaviour (distance grows like sqrt(tau)), definition of a first-order Markov chain. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 11.3 Gibbs Sampling (PDF p.563): the coordinate-by-coordinate update scheme. READ."
  - "math/math for ML.pdf, Sec. 6.7 Change of Variables / Inverse Transform (PDF p.221-222, book p.215-216): distribution function technique via the cdf, transformation of random variables. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/16-sampling-methods/docs/en.md: inverse CDF for the exponential, rejection sampling with bound M and acceptance 1/M, importance sampling, Monte Carlo O(1/sqrt N), Metropolis-Hastings algorithm and acceptance rule, Gibbs sampling. READ."
---

## Où cela s'inscrit

De nombreuses questions en machine learning se ramènent à une moyenne sur une distribution que nous ne savons pas traiter analytiquement : la perte espérée d'un modèle sur toutes les données possibles, la moyenne a posteriori d'un paramètre, le rendement espéré d'une politique de jeu, la probabilité d'une défaillance rare. Écrites comme une intégrale $\int f(x)\,p(x)\,dx$, elles n'ont pas de forme fermée dès que $p$ n'est pas une loi de manuel, et l'intégration sur grille meurt quand le nombre de dimensions croît. Pendant ce temps, un ordinateur est très bon dans une chose : produire des nombres qui semblent uniformément aléatoires entre 0 et 1.

Cette leçon est le pont entre ces deux faits. D'abord, **comment transformer des nombres aléatoires uniformes en tirages de la distribution voulue** (transformée inverse, rejet, importance). Ensuite, **pourquoi la moyenne de tirages estime une intégrale, et à quelle vitesse** (la loi en $1/\sqrt N$). Enfin, **que faire quand on ne peut même pas échantillonner directement** (Monte Carlo par chaînes de Markov). Tu retrouveras ces idées dans le dropout, en apprentissage par renforcement (poids d'importance dans les gradients de politique), en apprentissage profond bayésien, dans les modèles de diffusion et dans la façon dont les modèles de langage échantillonnent le mot suivant.

Elle utilise les probabilités et les variables aléatoires (leçons 8 et 9) et la leçon d'estimation juste avant (une moyenne de tirages est un estimateur, avec l'erreur standard que tu y as vue).

## L'idée en mots simples

- **Mesurer un lac en lançant des pierres.** Pour trouver l'aire d'un lac irrégulier à l'intérieur d'un champ carré d'aire connue, lance des pierres au hasard dans le champ et compte la fraction qui éclabousse. L'aire du lac vaut à peu près cette fraction fois le champ. Plus de pierres, meilleure estimation, et tu n'as jamais besoin d'une formule pour la rive. C'est Monte Carlo.
- **La machine uniforme-vers-n'importe-quoi (transformée inverse).** Imagine toutes les valeurs de la quantité voulue triées de la plus petite à la plus grande, posées le long d'une règle de longueur 1 où chaque tronçon a une longueur égale à sa probabilité. Choisis un point uniforme au hasard sur la règle et lis sur quelle valeur il tombe. Les valeurs courantes possèdent de longs tronçons, donc elles sont souvent choisies. Mathématiquement, la règle est la fonction de répartition et la lire revient à l'inverser.
- **Lancer des fléchettes sur une cible plus grande (échantillonnage par rejet).** Si tu ne peux pas échantillonner la forme voulue, échantillonne dans une forme plus grande et facile qui la recouvre, et écarte (rejette) chaque point qui tombe en dehors. Ce qui reste est uniforme à l'intérieur de ta forme. Coût : les lancers rejetés sont gaspillés.
- **Interroger la mauvaise foule, puis repondérer (échantillonnage préférentiel).** Tu veux l'avis d'experts rares mais tu n'as accès qu'à une foule générale ? Sonde la foule et multiplie chaque réponse par la sur- ou sous-représentation de cette personne par rapport aux experts. Rien n'est jeté, mais si la foule ne contient presque jamais d'expert, quelques réponses portent d'énormes poids et l'estimation est fragile.
- **Un marcheur qui préfère les hauteurs (MCMC).** Place un randonneur dans un paysage dont la hauteur est la probabilité. Une règle : propose un pas aléatoire ; accepte toujours s'il monte, et accepte parfois un pas descendant (avec une probabilité égale au rapport des hauteurs). Longtemps après le départ, la fraction du temps que le randonneur passe en chaque lieu correspond à la probabilité de ce lieu. Tu n'as jamais eu besoin de connaître la constante de normalisation, seulement les rapports de hauteurs.

## Échantillonnage par transformée inverse

Soit $F(x)=P(X\le x)$ la fonction de répartition (cdf) de la cible. Elle croît de 0 à 1. Si $U\sim\text{Uniform}(0,1)$, alors $X=F^{-1}(U)$ a pour cdf $F$.

**Démonstration.** $P(F^{-1}(U)\le x)=P(U\le F(x))$ parce que $F$ est croissante (applique $F$ aux deux membres). Pour une variable uniforme, $P(U\le c)=c$ pour $c\in[0,1]$. Donc $P(F^{-1}(U)\le x)=F(x)$, qui est la cdf voulue. C'est la « technique de la fonction de répartition » de Deisenroth parcourue en sens inverse, et elle égale l'éq. 11.6 de Bishop : transformer des nombres uniformes par l'inverse de l'intégrale indéfinie de la densité voulue.

**Exemple : l'exponentielle.** La densité $\lambda e^{-\lambda x}$ pour $x\ge0$ a pour cdf $F(x)=1-e^{-\lambda x}$. Pose $u=1-e^{-\lambda x}$ et résous : $e^{-\lambda x}=1-u$, donc $x=-\ln(1-u)/\lambda$. Avec $\lambda=2$ : $u=0.5$ donne $x=0.3466$ et $u=0.9$ donne $x=1.1513$. Comme $1-U$ est aussi uniforme, tu peux écrire $-\ln(u)/\lambda$.

Limite : il te faut $F^{-1}$ en forme fermée. La gaussienne n'en a pas, c'est pourquoi Bishop présente Box-Muller (deux uniformes donnent deux normales standard indépendantes) et l'astuce $\mu+L z$ avec $\Sigma=LL^\top$ (Cholesky) pour une gaussienne multivariée de covariance $\Sigma$ : si $z$ a des composantes indépendantes de variance unité, $\mathrm{Cov}(\mu+Lz)=L I L^\top=\Sigma$.

## Échantillonnage par rejet

Supposons que nous puissions évaluer la cible $p(x)$ (même non normalisée) mais pas l'échantillonner. Choisis une proposition $q(x)$ que nous savons échantillonner et une constante $M$ avec $p(x)\le Mq(x)$ partout.

1. Tire $x\sim q$ et $u\sim\text{Uniform}(0,1)$.
2. Accepte $x$ si $u<\dfrac{p(x)}{Mq(x)}$, sinon rejette et recommence.

**Pourquoi ça marche.** La densité acceptée en $x$ est proportionnelle à $q(x)\cdot\dfrac{p(x)}{Mq(x)}=\dfrac{p(x)}{M}$, proportionnelle à $p(x)$. La probabilité d'acceptation globale est $\int q(x)\frac{p(x)}{Mq(x)}dx=\frac1M$ (pour $p$ normalisée), donc un $M$ serré est peu coûteux.

**Exemple.** Cible $p(x)=6x(1-x)$ sur $[0,1]$ (une densité Beta(2,2), maximum $1.5$ en $x=0.5$), proposition uniforme, $M=1.5$. Le taux d'acceptation est $1/1.5=0.667$ ; une exécution de 100 000 propositions en a accepté $66.5\%$, avec une moyenne empirique de $0.4998$ et une variance de $0.0496$ (valeurs exactes $0.5$ et $0.05$).

**Grande dimension.** Bishop (livre p.531) utilise une cible gaussienne et une proposition gaussienne un peu plus large pour montrer pourquoi le rejet s'effondre. En faisant le calcul : avec des variances isotropes $\sigma_p^2<\sigma_q^2$, la plus petite borne valide est $M=(\sigma_q/\sigma_p)^D$, qui croît exponentiellement avec $D$, donc le taux d'acceptation $1/M$ s'évanouit (pour $\sigma_q/\sigma_p=1.1$ et $D=100$ il vaut environ $7\times10^{-5}$). Le rejet est un outil pour 1 à 3 dimensions.

## Échantillonnage préférentiel et intégration de Monte Carlo

Pour estimer une espérance, nous n'avons pas réellement besoin d'échantillons de $p$. Réécris

$$
\mathbb E_p[f]=\int f(x)p(x)\,dx=\int f(x)\frac{p(x)}{q(x)}q(x)\,dx\ \approx\ \frac1L\sum_{l=1}^L r_l f(x_l),\qquad x_l\sim q,\ \ r_l=\frac{p(x_l)}{q(x_l)}.
$$

Les $r_l$ sont les poids d'importance (Bishop éq. 11.19). Si $p$ n'est connue qu'à une constante près, utilise la forme autonormalisée $\sum_l w_l f(x_l)$ avec $w_l=r_l/\sum_m r_m$ (éq. 11.22-11.23) ; tous les échantillons sont conservés, contrairement au rejet. Le rééchantillonnage préférentiel (sampling-importance-resampling, Bishop p.535) retire ensuite dans l'ensemble pondéré pour obtenir des échantillons approchés de $p$.

**Pourquoi il peut échouer.** L'estimateur est non biaisé, mais sa variance dépend de la façon dont $q$ recouvre $p$. Si $q$ met peu de masse là où $|f|p$ est grand, quelques poids énormes dominent. L'application standard est l'apprentissage par renforcement, où le poids est le rapport de la probabilité d'une action sous la nouvelle politique à celle sous l'ancienne.

### Intégration de Monte Carlo et l'erreur en $1/\sqrt N$

Le Monte Carlo simple est l'échantillonnage préférentiel avec $q=p$ : $I=\mathbb E[f(X)]\approx\hat I=\frac1N\sum f(x_i)$. Il est non biaisé et, d'après la leçon d'estimation,

$$
\operatorname{Var}(\hat I)=\frac{\operatorname{Var}(f(X))}{N},\qquad \text{error}\ \sim\ \frac{\sigma_f}{\sqrt N}.
$$

Deux conséquences. Pour gagner un chiffre de précision supplémentaire, il faut 100 fois plus d'échantillons. Et le taux ne mentionne pas la dimension, ce qui explique pourquoi Monte Carlo l'emporte sur les grilles en grande dimension (une grille de 10 points par axe demande $10^{D}$ évaluations ; voir le fléau de la dimension dans la leçon 16).

**Exemple : estimer $\pi$.** Tire $(x,y)$ uniformément dans le carré unité ; $P(x^2+y^2\le1)=\pi/4$, donc $\hat\pi=4\times$ (fraction à l'intérieur). Le comptage est de Bernoulli avec $p=\pi/4$, donc l'écart-type de $\hat\pi$ est $4\sqrt{p(1-p)/N}=1.642/\sqrt N$. Avec la graine 1, les estimations sont $3.04$ ($N=100$), $3.1564$ ($N=10^4$) et $3.1432$ ($N=10^6$) ; les tailles d'erreur prédites sont $0.16$, $0.016$, $0.0016$.

**Exemple : un événement rare.** $P(Z>3)$ pour une normale standard vaut $0.00135$. Un Monte Carlo naïf avec $N=10{,}000$ voit environ 13 succès, donc chaque estimation a un écart-type d'environ $3.7\times10^{-4}$ (27 % en relatif). L'échantillonnage préférentiel depuis $q=\mathcal N(3,1)$ avec les poids $p(x)/q(x)=\exp(-x^2/2+(x-3)^2/2)$ place la moitié de ses échantillons au-delà de 3 ; l'écart-type sur 200 répétitions tombe à $2.5\times10^{-5}$, environ 15 fois plus petit, pour le même $N$.

## Monte Carlo par chaînes de Markov

Le rejet et l'échantillonnage préférentiel se dégradent en grande dimension. Le remède est de construire une **chaîne de Markov** : une suite $z^{(1)},z^{(2)},\dots$ où chaque état ne dépend que du précédent (Bishop éq. 11.37). La chaîne est conçue pour que, sur le long terme, elle visite les états proportionnellement à la cible $p$. Les échantillons sont corrélés (les voisins dans la chaîne sont proches), mais les moyennes sur la chaîne convergent quand même. Bishop note qu'une marche aléatoire non guidée ne se déplace que d'une distance $\propto\sqrt\tau$ après $\tau$ pas, donc un bon échantillonneur doit éviter le comportement de pure marche aléatoire.

### Metropolis (et Metropolis-Hastings)

Cible $p(z)=\tilde p(z)/Z$ où seule $\tilde p$ peut être évaluée.

1. Depuis le $z$ courant, propose $z'\sim q(z'\mid z)$. Dans l'algorithme de Metropolis original, $q$ est symétrique, par ex. $z'=z+\text{Gaussian noise}$.
2. Accepte avec la probabilité $\alpha=\min\!\big(1,\ \tilde p(z')/\tilde p(z)\big)$ (pour une $q$ non symétrique, multiplie par $q(z\mid z')/q(z'\mid z)$ : Metropolis-Hastings).
3. Si accepté, passe à $z'$ ; sinon reste en $z$ **et compte à nouveau l'état répété**.

La constante inconnue $Z$ s'annule dans le rapport, ce qui explique pourquoi le MCMC est le cheval de trait des a posteriori bayésiens connus à un normalisateur près. La correction vient de l'**équilibre détaillé** : la règle rend le flux $z\to z'$ égal au flux $z'\to z$ sous $p$, ce qui implique que $p$ est stationnaire pour la chaîne. Points pratiques : écarte une *période de chauffe* initiale (burn-in) ; l'échelle de la proposition doit être réglée (trop petite, la chaîne rampe ; trop grande, presque tout est rejeté) ; des échantillons corrélés portent moins d'information que le même nombre d'échantillons indépendants.

**Exemple.** Cible la normale standard ($\log\tilde p=-z^2/2$), pas de proposition $2.4\times\mathcal N(0,1)$, 20 000 pas depuis $z=0$ (graine 0), après avoir écarté 1 000 échantillons de chauffe : taux d'acceptation $0.44$, moyenne empirique $0.0006$, variance empirique $0.975$, proches de $0$ et $1$.

### Échantillonnage de Gibbs

Si tu sais échantillonner chaque variable selon sa **conditionnelle** sachant toutes les autres, mets-les à jour une à une (Bishop p.543) :

$$
z_1\sim p(z_1\mid z_2,z_3,\dots),\quad z_2\sim p(z_2\mid z_1^{\text{new}},z_3,\dots),\ \dots
$$

en parcourant toutes les variables, toujours avec les valeurs les plus récentes. C'est Metropolis avec des propositions toujours acceptées, donc il n'y a pas de taille de pas à régler. Il peine quand les variables sont fortement corrélées : des déplacements d'une coordonnée à la fois ne peuvent pas traverser une crête diagonale étroite, donc la chaîne mélange lentement.

## Implémentation

```python
import numpy as np

rng = np.random.default_rng(0)

def exp_inverse(u, lam):                    # inverse transform
    return -np.log(1 - u) / lam

def metropolis(logp, n, step, x0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x, out, acc = x0, np.empty(n), 0
    for t in range(n):
        xp = x + step * rng.normal()
        if np.log(rng.random()) < logp(xp) - logp(x):   # work in log space
            x, acc = xp, acc + 1
        out[t] = x
    return out, acc / n
```

La comparaison se fait en espace logarithmique pour que les densités minuscules ne produisent jamais d'underflow (leçon 15).

## Formes

Échantillonnage vectorisé : `u: (N,)` tirages uniformes, `x = f_inverse(u): (N,)`. Le rejet avec `N` propositions renvoie un masque booléen `(N,)` et `x[mask]` a une longueur $\approx N/M$. Échantillonnage préférentiel : échantillons `(N,)` ou `(N, D)`, poids `(N,)`, estimation `np.mean(w * f(x))`. Une chaîne de Metropolis est `(n,)` ou `(n, D)` ; tirages gaussiens multivariés : `mu + z @ L.T` avec `z: (N, D)`, `L: (D, D)`.

## Idées fausses courantes

- **« Monte Carlo est lent parce qu'il demande beaucoup d'échantillons. »** Il est lent pour obtenir une haute précision ($1/\sqrt N$) mais son taux ne dépend pas de la dimension, ce qui est sa grande force.
- **« Les échantillons MCMC sont indépendants. »** Ils sont corrélés ; la taille d'échantillon effective est plus petite que la longueur de la chaîne.
- **« Les propositions rejetées sont écartées dans Metropolis. »** La chaîne reste en place et répète l'ancien état ; l'omettre biaise le résultat.
- **« Plus d'échantillons d'une mauvaise $q$ réparent l'échantillonnage préférentiel. »** Si $q$ rate la région qui compte, les poids ont une variance énorme ou infinie et plus d'échantillons n'aident que très lentement.
- **« Les nombres pseudo-aléatoires sont vraiment aléatoires. »** Ce sont des suites déterministes qui passent des tests statistiques (Bishop p.526) ; c'est pourquoi fixer une graine reproduit les résultats.

## Points clés à retenir

- Transformée inverse : $X=F^{-1}(U)$. Rejet : accepter avec la probabilité $p/(Mq)$, taux $1/M$. Préférentiel : pondérer par $p/q$.
- Erreur de Monte Carlo $\approx\sigma_f/\sqrt N$ quelle que soit la dimension.
- MCMC (Metropolis : accepter avec $\min(1,\tilde p(z')/\tilde p(z))$) n'a pas besoin de constante de normalisation ; Gibbs échantillonne une conditionnelle à la fois.
- Le rejet et l'échantillonnage préférentiel s'effondrent en grande dimension ; le MCMC échange l'indépendance contre le passage à l'échelle.

## Explique-le avec tes mots

1. Explique à un ami pourquoi lancer des fléchettes au hasard sur un carré peut estimer $\pi$, et pourquoi 100 fois plus de fléchettes ne donnent qu'une décimale de plus.
2. Pourquoi l'algorithme de Metropolis n'a-t-il pas besoin de la constante de normalisation de la cible, et que se passe-t-il si tu prends un pas de proposition beaucoup trop grand ?

## Pour aller plus loin

- Bishop, *Pattern Recognition and Machine Learning*, chap. 11 (y compris l'échantillonnage par tranches et le Monte Carlo hybride, que cette leçon omet).
- Deisenroth et al., Sec. 6.7 pour le point de vue du changement de variables dans la transformation de variables aléatoires.
- Dépôt, phase 01, leçon 16, pour la température, l'échantillonnage top-k et top-p, l'astuce de reparamétrisation utilisée dans les VAE et Gumbel-softmax.

## Critères de maîtrise

- Démontrer l'échantillonneur par transformée inverse de la loi exponentielle et prouver pourquoi $F^{-1}(U)$ a pour cdf $F$.
- Expliquer la probabilité d'acceptation $1/M$ et pourquoi le rejet échoue en grande dimension.
- Écrire l'estimateur d'échantillonnage préférentiel et dire quand sa variance est grande.
- Implémenter Metropolis en espace logarithmique et nommer la période de chauffe, la taille de pas et l'autocorrélation comme ses pièges.
