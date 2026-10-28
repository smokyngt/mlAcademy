---
id: cml.12
title: "Modèles de mélange gaussien et algorithme EM"
summary: "Modéliser les données comme un mélange de gaussiennes avec une étiquette de cluster cachée, dériver l'étape E et l'étape M, prouver que la vraisemblance ne diminue jamais, et voir le k-means comme la limite dure."
minutes: 80
teaches: [ml.gmm, ml.em]
requires: [ml.kmeans, prob.basics, prob.rv, alg.exp-log]
sources:
  - "math for ML.pdf (Deisenroth, Faisal, Ong), Ch. 11 Density Estimation with Gaussian Mixture Models (PDF p.354-374): mixture model, responsibilities, update theorems for means/covariances/weights, EM algorithm, running 1-D example with its numbers, latent-variable view. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 9.2 Mixtures of Gaussians (PDF p.450-459): latent variable z, posterior gamma(z_nk), singularities, identifiability, EM for Gaussian mixtures summary, k-means initialisation. READ."
---

## Où cela s'inscrit

Le k-means (leçon 4) donne chaque point à exactement un cluster. Les données réelles sont rarement aussi ordonnées. Un client peut se comporter en partie comme un chasseur de bonnes affaires et en partie comme un acheteur fidèle ; une mesure d'un capteur peut venir de l'un de deux régimes de fonctionnement ; les tailles dans un groupe de personnes sont un mélange de deux populations. Dans ces cas nous voulons deux choses que le k-means ne peut pas donner : une **densité de probabilité** (quelle est la vraisemblance d'un nouveau point, est-ce une valeur aberrante ?) et une **affectation douce** (ce point est à 90 % du groupe A et à 10 % du groupe B).

Un **modèle de mélange gaussien (GMM)** fournit les deux. Son algorithme d'ajustement, l'**espérance-maximisation (EM)**, est l'une des grandes idées récurrentes de l'apprentissage automatique : chaque fois qu'un modèle a des variables cachées, alterne entre *deviner les variables cachées étant donné les paramètres* et *réajuster les paramètres étant donné les suppositions*. Le même schéma anime les modèles de Markov cachés, les modèles de thèmes et une partie de l'inférence variationnelle. Le k-means était les petites roues ; cette leçon est le vélo.

## L'idée en mots simples

Imagine une usine avec deux machines qui produisent des boulons de longueurs légèrement différentes, et un bac qui mélange les boulons des deux. Tu mesures un boulon et tu veux savoir quelle machine l'a fabriqué. Si tu connaissais la longueur typique et la dispersion de chaque machine, et combien de boulons chacune fabrique, tu pourrais calculer pour n'importe quel boulon mesuré à quel point il est plausible sous chaque machine, et pondérer cela par l'activité de la machine. Cette part de plausibilité est la **responsabilité** de la machine pour le boulon.

À l'inverse, si quelqu'un te disait exactement quelle machine a fabriqué chaque boulon, estimer la longueur moyenne et la dispersion de chaque machine serait facile : il suffirait de faire la moyenne des boulons de cette machine.

Tu ne connais ni l'un ni l'autre. L'EM brise le cercle en alternant :

1. **Étape E (espérance) :** avec la supposition courante sur les machines, calcule pour chaque boulon la responsabilité de chaque machine (une affectation douce, probabiliste).
2. **Étape M (maximisation) :** traite les responsabilités comme une appartenance fractionnaire et ré-estime chaque machine : sa moyenne est la moyenne pondérée par les responsabilités de tous les boulons, sa dispersion la dispersion pondérée par les responsabilités, sa part la responsabilité moyenne.

Répète. À chaque tour, l'ajustement aux données ne peut que s'améliorer. Si chaque responsabilité valait exactement 0 ou 1, ce serait précisément le k-means.

## Le modèle

Un GMM écrit la densité de $x\in\mathbb R^d$ comme une combinaison convexe de $K$ gaussiennes :

$$
p(x\mid\theta)=\sum_{k=1}^{K}\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k),\qquad 0\le\pi_k\le1,\quad\sum_k\pi_k=1,
$$

avec les paramètres $\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$. Pour $K=1$ c'est une seule gaussienne ; avec plus de composantes elle peut être multimodale.

### Vue par variable latente

Introduis un vecteur caché one-hot $z\in\{0,1\}^K$ avec $\sum_kz_k=1$ indiquant quelle composante a produit $x$ :

$$
p(z_k=1)=\pi_k,\qquad p(x\mid z_k=1)=\mathcal N(x\mid\mu_k,\Sigma_k).
$$

Marginaliser $z$ redonne le mélange : $p(x)=\sum_kp(z_k=1)\,p(x\mid z_k=1)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)$. Cela donne aussi une recette générative (**échantillonnage ancestral**) : tire une composante $k$ avec probabilité $\pi_k$, puis tire $x\sim\mathcal N(\mu_k,\Sigma_k)$, et oublie $k$. Un jeu de données « complet » contiendrait $(x_n,z_n)$ ; le nôtre est « incomplet » : nous ne voyons que $x_n$.

## Dérivation 1 : l'étape E est le théorème de Bayes

Étant donné $\theta$, quelle est la probabilité que la composante $k$ ait engendré le $x_n$ observé ? Par le théorème de Bayes,

$$
r_{nk}\equiv p(z_{nk}=1\mid x_n)=\frac{p(z_{nk}=1)\,p(x_n\mid z_{nk}=1)}{\sum_{j=1}^K p(z_{nj}=1)\,p(x_n\mid z_{nj}=1)}=\frac{\pi_k\,\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\,\mathcal N(x_n\mid\mu_j,\Sigma_j)}.
$$

L'a priori est $\pi_k$, la vraisemblance est la densité gaussienne, et le dénominateur normalise. $r_n=(r_{n1},\dots,r_{nK})$ est un vecteur de probabilités, une affectation douce. Définis $N_k=\sum_nr_{nk}$, le nombre effectif de points de la composante $k$ ; $\sum_kN_k=N$.

### Exemple détaillé 1 : une responsabilité à la main

L'exemple courant du chapitre de Deisenroth : données $X=\{-3,-2.5,-1,0,2,4,5\}$, trois composantes de moyennes initiales $-4,0,8$, de variances $1,0.2,3$ et de poids égaux $1/3$. Pour $x=-1$ :

- composante 1 : $\tfrac13\mathcal N(-1\mid-4,1)=\tfrac13\cdot0.00443=0.00148$,
- composante 2 : $\tfrac13\mathcal N(-1\mid0,0.2)=\tfrac13\cdot0.0732=0.0244$,
- composante 3 : $\tfrac13\mathcal N(-1\mid8,3)\approx 10^{-6}$ (négligeable).

En normalisant : $r=(0.057,\,0.943,\,0.000)$. Le même calcul pour les sept points donne des responsabilités totales $N_k=(2.057,\,2.009,\,2.934)$. (L'exercice reproduit toute la matrice.)

## Dérivation 2 : l'étape M

Pour des responsabilités fixées, on maximise l'espérance de la log-vraisemblance des données complètes

$$
Q(\theta)=\sum_{n=1}^N\sum_{k=1}^Kr_{nk}\Bigl[\log\pi_k+\log\mathcal N(x_n\mid\mu_k,\Sigma_k)\Bigr].
$$

Pourquoi cette fonction ? Un jeu de données complet aurait une log-vraisemblance $\sum_n\log\pi_{z_n}\mathcal N(x_n\mid\mu_{z_n},\Sigma_{z_n})$ ; nous ne connaissons pas $z_n$, donc nous remplaçons l'indicatrice $z_{nk}$ par son espérance $r_{nk}$. Le caractère (non) fermé du problème original vient de la somme à l'intérieur du logarithme, $\log\sum_k\pi_k\mathcal N(\cdot)$, qui ne se sépare pas par composante. Dans $Q$ le logarithme agit sur chaque gaussienne séparément, et les composantes se découplent. (Le livre obtient les mêmes trois formules en annulant directement le gradient de la log-vraisemblance $\log p(X\mid\theta)$, et les responsabilités apparaissent automatiquement par la règle de la chaîne.)

**Moyennes.** Pour une composante, $\nabla_{\mu_k}Q=\sum_nr_{nk}\Sigma_k^{-1}(x_n-\mu_k)=0$. Multiplie par $\Sigma_k$ :

$$
\sum_nr_{nk}x_n=\mu_k\sum_nr_{nk}\;\Longrightarrow\;\mu_k=\frac1{N_k}\sum_{n=1}^Nr_{nk}\,x_n.
$$

La nouvelle moyenne est la moyenne de **tous** les points de données, chacun pondéré par la responsabilité de la composante $k$ pour lui. Avec des responsabilités 0/1, c'est la moyenne de cluster du k-means.

**Variance (vérification en 1-D).** Avec $\Sigma_k=\sigma_k^2$ : $\frac{\partial Q}{\partial\sigma_k^2}=\sum_nr_{nk}\bigl(-\frac1{2\sigma_k^2}+\frac{(x_n-\mu_k)^2}{2\sigma_k^4}\bigr)=0$. Multiplie par $2\sigma_k^4$ : $-N_k\sigma_k^2+\sum_nr_{nk}(x_n-\mu_k)^2=0$, donc

$$
\sigma_k^2=\frac1{N_k}\sum_nr_{nk}(x_n-\mu_k)^2 .
$$

**Covariance (cas général $d$).** Le même calcul, utilisant les dérivées de $\log\det\Sigma$ et de $(x-\mu)^\top\Sigma^{-1}(x-\mu)$, donne

$$
\Sigma_k=\frac1{N_k}\sum_nr_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top ,
$$

une covariance empirique pondérée, calculée avec la **nouvelle** $\mu_k$.

**Poids.** Maximise $\sum_n\sum_kr_{nk}\log\pi_k$ sous la contrainte $\sum_k\pi_k=1$ avec un multiplicateur de Lagrange : $\frac{\partial}{\partial\pi_k}\bigl[\sum_{n,k}r_{nk}\log\pi_k+\lambda(\sum_k\pi_k-1)\bigr]=\frac{N_k}{\pi_k}+\lambda=0$, donc $\pi_k=-N_k/\lambda$. Sommer sur $k$ donne $1=-N/\lambda$, soit $\lambda=-N$ et

$$
\pi_k=\frac{N_k}{N},
$$

la responsabilité moyenne.

### Algorithme

1. Initialise $\pi_k,\mu_k,\Sigma_k$.
2. **Étape E :** $r_{nk}=\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)/\sum_j\pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)$.
3. **Étape M :** $N_k$, $\mu_k$, $\Sigma_k$, $\pi_k$ comme ci-dessus.
4. Calcule $\log p(X\mid\theta)=\sum_n\log\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$ ; arrête-toi quand elle (ou les paramètres) cesse de changer.

### Exemple détaillé 2 : l'exemple courant, exécuté

En partant de la supposition initiale ci-dessus (log-vraisemblance $-28.33$), une étape E plus une étape M donnent des moyennes $-2.70,\,-0.40,\,3.70$, des variances $0.14,\,0.44,\,1.53$, des poids $0.29,\,0.29,\,0.42$, et la log-vraisemblance saute à $-14.41$. En itérant, après environ cinq tours l'EM se stabilise sur

$$
p(x)=0.29\,\mathcal N(x\mid-2.75,0.06)+0.28\,\mathcal N(x\mid-0.50,0.25)+0.43\,\mathcal N(x\mid3.64,1.63),
$$

avec une log-vraisemblance $\approx-13.97$. Tous ces nombres correspondent au livre et sont reproduits par le code de l'exercice.

## Pourquoi la log-vraisemblance ne diminue jamais

Soit $\ell(\theta)=\sum_n\log\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$. Pour tout vecteur de probabilités $q_n=(q_{n1},\dots,q_{nK})$ par point de données, multiplie et divise par $q_{nk}$ à l'intérieur de la somme et utilise la concavité du $\log$ (inégalité de Jensen, $\log\mathbb E[Y]\ge\mathbb E[\log Y]$) :

$$
\ell(\theta)=\sum_n\log\sum_kq_{nk}\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{q_{nk}}\;\ge\;\sum_n\sum_kq_{nk}\log\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{q_{nk}}=:F(q,\theta).
$$

L'égalité a lieu exactement quand le rapport $\pi_k\mathcal N(\cdot)/q_{nk}$ est le même pour tout $k$, c'est-à-dire quand $q_{nk}$ est proportionnel à $\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$ : autrement dit, quand $q_{nk}=r_{nk}(\theta)$, la probabilité a posteriori de l'étape E. Ainsi :

- **Étape E :** avec $\theta^{(t)}$ fixé, choisir $q=r(\theta^{(t)})$ rend la borne tangente à la vraisemblance : $F(q,\theta^{(t)})=\ell(\theta^{(t)})$.
- **Étape M :** avec $q$ fixé, $F(q,\theta)=Q(\theta)-\sum q\log q$, et le dernier terme ne dépend pas de $\theta$. Maximiser $Q$ maximise donc $F$ : $F(q,\theta^{(t+1)})\ge F(q,\theta^{(t)})$.

Enchaîne :

$$
\ell(\theta^{(t+1)})\;\ge\;F(q,\theta^{(t+1)})\;\ge\;F(q,\theta^{(t)})\;=\;\ell(\theta^{(t)}).
$$

Chaque itération d'EM augmente donc (ou conserve) la log-vraisemblance. La suite n'est bornée supérieurement que si aucune singularité n'apparaît (section suivante), et elle converge vers un maximum **local** ou un point selle de $\ell$, pas nécessairement le global. (L'énoncé de la propriété dans le livre est en PDF p.367 ; l'argument ci-dessus est la dérivation standard par Jensen, que le livre présente via l'espérance de la log-vraisemblance des données complètes $Q$, Éq. 11.73.)

## Lien avec le k-means

Considère un GMM à poids égaux et à covariance sphérique partagée $\Sigma_k=\sigma^2I$. Les responsabilités de l'étape E sont

$$
r_{nk}\propto\exp\!\Bigl(-\frac{\|x_n-\mu_k\|^2}{2\sigma^2}\Bigr).
$$

Quand $\sigma^2\to0$, le centre le plus proche l'emporte massivement et $r_{nk}\to1$ pour le $k$ le plus proche, 0 pour les autres : **affectation dure**. La moyenne de l'étape M est alors la simple moyenne de cluster. Le k-means est donc la limite de l'EM pour des gaussiennes sphériques égales de variance tendant vers zéro. Les deux diffèrent par ce qu'ils offrent :

| | k-means | GMM + EM |
|---|---|---|
| affectation | dure | douce (responsabilités) |
| forme des clusters | ronds, de taille égale | ellipsoïdes avec leurs propres $\Sigma_k$ et taille $\pi_k$ |
| sortie | centres et étiquettes | densité $p(x)$, probabilités, détection possible de valeurs aberrantes |
| coût par itération | moins cher | plus cher (densités, covariances) |

### Exemple détaillé 3 : doux contre dur

Deux centres $1$ et $11$, poids égaux, variance $4$. Pour $x=5$, le log-rapport des deux densités pondérées vaut $\frac{(5-11)^2-(5-1)^2}{2\cdot4}=\frac{36-16}{8}=2.5$, donc $r_1=1/(1+e^{-2.5})=0.924$ : le k-means dirait « cluster 1 », l'EM dit « 92.4 % cluster 1, 7.6 % cluster 2 ». Pour $x=6$ (le milieu) $r=(0.5,0.5)$.

## Questions pratiques

**Singularités.** Une composante peut placer sa moyenne sur un seul point de données et réduire sa variance : la densité en ce point est $\mathcal N(x_n\mid x_n,\sigma^2)=1/\sqrt{2\pi\sigma^2}\to\infty$. Pour $\sigma^2=10^{-12}$ la log-vraisemblance gagne $-\tfrac12\log(2\pi\sigma^2)\approx12.9$ nats grâce à ce seul point, et sans limite quand $\sigma^2\to0$, tandis qu'une seconde composante garde tous les autres points à une probabilité finie. Une seule gaussienne ne peut pas faire cela, car s'effondrer sur un point envoie la vraisemblance de tous les autres points à zéro. La vraisemblance est donc *non bornée*, une forme extrême de surapprentissage (Bishop 9.2.1). Remèdes : ajouter un petit ridge $\varepsilon I$ à chaque covariance (un plancher de variance), détecter les composantes qui s'effondrent et les ré-initialiser, ou utiliser un a priori bayésien/MAP.

**Initialisation et maxima locaux.** L'EM trouve un maximum local qui dépend du départ. Lance plusieurs départs aléatoires et garde la meilleure log-vraisemblance. Recommandation de Bishop : lance d'abord le k-means, puis initialise $\mu_k$ avec les centres, $\Sigma_k$ avec les covariances des clusters et $\pi_k$ avec les fractions des clusters. L'EM demande typiquement bien plus d'itérations que le k-means.

**Identifiabilité.** Renommer les composantes ($K!$ façons) donne la même densité, donc la composante « 1 » n'a pas de signification intrinsèque.

**Choisir $K$ et la forme de covariance.** La log-vraisemblance s'améliore toujours avec plus de composantes, donc $K$ ne peut pas être choisi sur la vraisemblance d'entraînement ; utilise la vraisemblance sur données mises de côté ou un critère d'information. Un GMM à covariance pleine a $K\bigl(d+\tfrac{d(d+1)}2\bigr)+K-1$ paramètres (pour $d=2$, $K=3$ : $17$) ; se restreindre à des covariances diagonales ou sphériques réduit ce nombre au prix de la flexibilité.

**Numérique.** Les densités sous-dépassent (underflow). Travaille avec des logs et l'astuce log-sum-exp : $\log\sum_ke^{a_k}=m+\log\sum_ke^{a_k-m}$ avec $m=\max_ka_k$.

## Implémentation

```python
import numpy as np

def e_step(x, pi, mu, var):              # x: (N,), pi/mu/var: (K,)
    w = pi * np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    return w / w.sum(axis=1, keepdims=True)           # (N, K)

def m_step(x, r):
    Nk = r.sum(axis=0)                                # (K,)
    mu = (r * x[:, None]).sum(axis=0) / Nk
    var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    return Nk / len(x), mu, var
```

Pour $d>1$ : `mu: (K, d)`, `S: (K, d, d)`, et la covariance de l'étape M est `(r[:, k, None] * D).T @ D / Nk[k]` avec `D = X - mu[k]`.

## Formes

`X: (N, d)`, responsabilités `r: (N, K)` dont les lignes somment à 1, `Nk: (K,)`, `pi: (K,)`, `mu: (K, d)`, `S: (K, d, d)`, log-poids `(N, K)`, historique : un scalaire par itération.

## Idées reçues

- **« L'EM trouve l'optimum global. »** Il augmente la vraisemblance de façon monotone mais seulement jusqu'à un maximum local ; relance et compare.
- **« Les responsabilités sont des probabilités des données. »** Ce sont des probabilités a posteriori de la composante cachée *étant donné* les données et les paramètres courants.
- **« Plus de composantes ajuste toujours mieux, donc choisis le plus grand $K$. »** Sur les données d'entraînement, oui ; c'est du surapprentissage, et une composante peut même s'effondrer sur un point.
- **« Un GMM est juste un k-means avec des probabilités. »** Il apprend aussi les formes et les tailles des clusters et définit une densité.
- **« La log-vraisemblance peut se calculer à partir des seules responsabilités. »** Elle demande les densités (le dénominateur de l'étape E, en logs).

## Points clés à retenir

- Un GMM est un modèle à variable latente : choisis une composante avec probabilité $\pi_k$, puis échantillonne sa gaussienne. Marginaliser la composante donne la densité de mélange.
- Étape E : responsabilités par le théorème de Bayes. Étape M : moyennes pondérées, covariances pondérées, responsabilités moyennes.
- L'EM augmente la log-vraisemblance à chaque pas (la borne inférieure de Jensen est tangente à l'étape E, l'étape M la relève), convergeant vers un maximum local.
- Le k-means est l'EM à la limite de composantes sphériques égales de variance $\to0$.
- Protège-toi des covariances singulières, utilise plusieurs initialisations ou celles du k-means, travaille en logs.

## Explique-le toi-même

1. Pourquoi ne peut-on pas simplement annuler le gradient de $\sum_n\log\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$ et résoudre pour $\mu_k$ en forme fermée ?
2. Décris en mots ce qui arrive aux responsabilités et aux moyennes quand toutes les gaussiennes deviennent très étroites.

## Pour aller plus loin

- *Mathematics for Machine Learning*, Ch. 11 (modèles de mélange, EM, vue par variable latente).
- Bishop, *PRML*, Sec. 9.2 (mélanges de gaussiennes) et Sec. 9.3 (la vue générale de l'EM, non requise ici).

## Critères de maîtrise

Tu sais énoncer la densité de mélange et sa forme à variable latente, calculer les responsabilités et les trois formules de l'étape M à la main et en NumPy pour $d=1$ et $d>1$, expliquer pourquoi la vraisemblance est non décroissante, décrire le problème de singularité et comment s'en prémunir, et relier l'EM au k-means.
