---
id: mx.04
title: Chaînes de Markov et processus stochastiques
summary: "Des systèmes dont l'étape suivante ne dépend que du présent, leurs matrices de transition, les distributions stationnaires comme vecteurs propres, quand les chaînes convergent, et comment cela alimente l'échantillonnage MCMC."
minutes: 75
teaches: [mx.markov, mx.stationary]
requires: [prob.basics, prob.rv, la.matrix, la.eigen]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 13.1 Markov Models (PDF p.627-630): first-order chain factorisation, higher-order chains and their K^(M-1)(K-1) parameter count, latent-variable chains as the bridge to HMMs. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md (random walk, Markov chain, MCMC and Langevin sections; section headings for the rest). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/21-graph-theory/docs/en.md (random-walk view used for PageRank). READ."
---

## Où cela s'inscrit

Les modèles de langage produisent un token à la fois, les agents d'apprentissage par renforcement passent d'état en état, et les modèles de diffusion ajoutent du bruit pas à pas. Tout cela, ce sont des **suites de variables aléatoires où chaque étape dépend de ce qui l'a précédée**. La plus simple structure maniable de ce genre est la chaîne de Markov : le futur ne dépend du passé qu'à travers le présent.

Les chaînes de Markov comptent en ML de trois façons concrètes : (1) elles modélisent des séquences (modèles de langage n-grammes, modèles de Markov cachés, processus de décision markoviens) ; (2) elles sont le moteur des **méthodes de Monte-Carlo par chaînes de Markov** (MCMC), qui échantillonnent des distributions trop compliquées pour être échantillonnées directement (lois a posteriori bayésiennes) ; (3) leur comportement à long terme est un problème de vecteurs propres, le même que celui que tu as résolu pour PageRank dans [[mx.03]] et pour les matrices dans [[math.07]]. Tu as besoin des probabilités ([[math.08]]) et des vecteurs propres.

## L'idée en mots simples

- Imagine un jeu de plateau où tu déplaces un pion selon des dés. Pour prédire où va le pion ensuite, tu as seulement besoin de savoir où il est *maintenant*, pas comment il y est arrivé. Cet oubli de l'histoire est la **propriété de Markov**.
- Les règles du jeu forment un tableau : pour chaque case, les probabilités d'atterrir sur chaque autre case. Multiplie le tableau par lui-même et tu obtiens les probabilités après deux coups, et ainsi de suite.
- Joue très longtemps et quelque chose de remarquable arrive souvent : la fraction du temps passée sur chaque case se stabilise sur une liste fixe de nombres qui ne dépend pas de l'endroit où tu as commencé. Le jeu « oublie » son point de départ. Cette liste est la **distribution stationnaire**.
- Tous les jeux ne font pas cela. Si le pion peut être piégé dans un coin, ou seulement alterner entre deux cases aux tours impairs et pairs, la fraction du temps ne se stabilise pas de la même façon. Les jeux exempts de tels pièges sont dits **ergodiques**.
- Le MCMC renverse la situation : au lieu qu'on nous donne les règles et qu'on demande la distribution stationnaire, on nous donne la distribution que nous voulons et nous *concevons des règles* dont le comportement à long terme la reproduit.

## Définitions

Une suite $X_0,X_1,\dots$ à valeurs dans un ensemble fini de $K$ états est une **chaîne de Markov** si

$$
P(X_{t+1}=j\mid X_t=i,X_{t-1},\dots,X_0)=P(X_{t+1}=j\mid X_t=i)=:P_{ij}.
$$

C'est la propriété de Markov ; Bishop (Sec. 13.1) dit la même chose par la factorisation de la probabilité jointe de $N$ observations, $p(x_1,\dots,x_N)=p(x_1)\prod_{n=2}^Np(x_n\mid x_{n-1})$. Quand $P_{ij}$ ne dépend pas de $t$, la chaîne est **homogène** (règle de transition stationnaire). La **matrice de transition** $P\in[0,1]^{K\times K}$ est **stochastique par lignes** : chaque ligne somme à 1, car depuis l'état $i$ tu dois aller quelque part.

*Ordres supérieurs.* Une chaîne d'ordre deux laisse $x_n$ dépendre de $x_{n-1},x_{n-2}$ ; une chaîne d'ordre $M$ avec des tableaux généraux demande $K^{M-1}(K-1)$ paramètres (Bishop p.629), exponentiel en $M$. Un modèle de langage n-gramme est une chaîne d'ordre $(n-1)$, c'est pourquoi les simples tableaux de comptage s'arrêtent à de petits $n$ et pourquoi les modèles de séquences neuronaux remplacent le tableau par une fonction à paramètres partagés. Ajouter une chaîne cachée (latente), avec des observations conditionnées sur elle, donne les modèles de Markov cachés (Bishop Sec. 13.2, non couverts ici) : les observations elles-mêmes n'ont alors plus de propriété de Markov d'ordre fini.

### Probabilités à $n$ pas

Soit $\mu_t\in\mathbb{R}^{1\times K}$ le vecteur ligne avec $(\mu_t)_i=P(X_t=i)$. Par la loi des probabilités totales,

$$
(\mu_{t+1})_j=\sum_i(\mu_t)_iP_{ij}\quad\Longleftrightarrow\quad\mu_{t+1}=\mu_tP,\qquad\mu_n=\mu_0P^n .
$$

L'entrée $(P^n)_{ij}$ est la probabilité d'aller de $i$ à $j$ en exactement $n$ pas (même récurrence que pour compter les marches dans [[mx.03]]).

## Distribution stationnaire

Une distribution $\pi$ (vecteur ligne, $\pi_i\ge0$, $\sum\pi_i=1$) est **stationnaire** si

$$
\boxed{\pi P=\pi .}
$$

Si la chaîne démarre dans $\pi$, elle reste dans $\pi$ pour toujours. De façon équivalente, $\pi^\top$ est un vecteur propre de $P^\top$ de valeur propre $1$ (un vecteur propre *à gauche* de $P$).

**Pourquoi une valeur propre 1 existe-t-elle ?** Parce que les lignes somment à 1, $P\mathbf 1=\mathbf 1$ : le vecteur de uns est un vecteur propre à droite de valeur propre 1. Une matrice et sa transposée ont les mêmes valeurs propres ([[math.07]]), donc $P^\top$ a aussi la valeur propre 1, et un vecteur propre correspondant existe. (Montrer qu'il peut être choisi non négatif demande le théorème de Perron-Frobenius, que nous citons.) De plus, aucune valeur propre n'a un module supérieur à 1, car $\|P^n\|_\infty=1$ pour tout $n$.

**Convergence.** Supposons $P$ diagonalisable avec des valeurs propres $1=\lambda_1>|\lambda_2|\ge\dots$ et des vecteurs propres à gauche $\ell_1=\pi,\ell_2,\dots$. Écris $\mu_0=\sum_ic_i\ell_i$ avec $c_1=1$ (la coordonnée selon $\pi$ vaut 1 car $\mu_0\mathbf 1=1$ et $\ell_i\mathbf 1=0$ pour $i\ge2$). Alors

$$
\mu_n=\mu_0P^n=\pi+\sum_{i\ge2}c_i\lambda_i^n\ell_i\ \longrightarrow\ \pi,
$$

avec une erreur qui diminue comme $|\lambda_2|^n$. Donc le **deuxième plus grand module de valeur propre** (SLEM) fixe la **vitesse de mélange** : $1-|\lambda_2|$ est le trou spectral, et une chaîne avec $|\lambda_2|$ proche de 1 oublie lentement son départ. C'est le même argument que l'itération de puissance pour PageRank.

**Ergodicité.** Une chaîne finie est dite **irréductible** si chaque état peut atteindre tous les autres, et **apériodique** si les temps de retour à un état n'ont pas tous un diviseur commun supérieur à 1. Irréductible et apériodique ensemble donnent **ergodique** (une chaîne finie est ergodique si et seulement si une puissance $P^k$ a toutes ses entrées strictement positives). Pour une chaîne ergodique :

1. La distribution stationnaire $\pi$ est unique et strictement positive.
2. $\mu_n\to\pi$ à partir de n'importe quel départ.
3. **Les moyennes temporelles égalent les moyennes spatiales :** la fraction du temps passée dans l'état $i$ le long d'une longue trajectoire tend vers $\pi_i$ ; plus généralement $\frac1T\sum_{t}f(X_t)\to\sum_i\pi_if(i)$. C'est ce qui permet à une seule simulation d'estimer des espérances.

Contre-exemples : la bascule à deux états $\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$ est périodique (la distribution oscille pour toujours), et la matrice identité est réductible (chaque état est son propre piège, et toute distribution est stationnaire).

## Marches aléatoires et processus de Poisson, en bref

Une **marche aléatoire simple** ajoute $\pm1$ avec probabilité égale à chaque pas : $S_n=\xi_1+\dots+\xi_n$. C'est une chaîne de Markov sur les entiers. Comme les pas sont indépendants de moyenne 0 et de variance 1, $\mathbb{E}[S_n]=0$ et $\operatorname{Var}S_n=n$, donc la distance typique au départ croît comme $\sqrt n$ (la signature d'un bruit indépendant : bruit de la SGD, mouvement brownien, modèles de diffusion). Elle n'a pas de distribution stationnaire sur les entiers : elle continue de s'étaler.

Un **processus de Poisson** de taux $\lambda$ compte des événements aléatoires en temps continu (requêtes arrivant sur un serveur, photons). Ses traits définissants : les temps d'attente entre événements sont des exponentielles indépendantes de moyenne $1/\lambda$, et le nombre d'événements dans un intervalle de longueur $T$ suit une loi de Poisson dont la moyenne et la variance valent toutes deux $\lambda T$. L'exponentielle a la propriété **sans mémoire**, $P(\text{wait}>s+t\mid\text{wait}>s)=P(\text{wait}>t)$, qui est la version en temps continu de la propriété de Markov. Les chaînes de Markov en temps continu sont construites à partir de tels temps de séjour exponentiels.

## Des chaînes de Markov au MCMC

Supposons que nous voulions des échantillons d'une distribution $\pi$ que nous ne pouvons évaluer qu'à une constante près (une loi a posteriori bayésienne). Construis une chaîne dont la distribution stationnaire est $\pi$ et fais-la tourner. Une condition suffisante est l'**équilibre détaillé** :

$$
\pi_iP_{ij}=\pi_jP_{ji}\quad\text{for all }i,j.
$$

Somme sur $i$ : $\sum_i\pi_iP_{ij}=\pi_j\sum_iP_{ji}=\pi_j$, ce qui est exactement $\pi P=\pi$. La recette de **Metropolis-Hastings** : depuis l'état $i$, propose $j$ avec probabilité $q_{ij}$ (symétrique : $q_{ij}=q_{ji}$) et accepte avec probabilité $\min(1,\pi_j/\pi_i)$. Alors, pour $i\ne j$,

$$
\pi_iP_{ij}=\pi_iq_{ij}\min\!\Big(1,\frac{\pi_j}{\pi_i}\Big)=q_{ij}\min(\pi_i,\pi_j),
$$

ce qui est symétrique en $i,j$, donc l'équilibre détaillé tient. Seul le *rapport* $\pi_j/\pi_i$ est nécessaire : les constantes de normalisation inconnues s'annulent. Les propositions rejetées laissent la chaîne où elle est (l'entrée diagonale). Dans les espaces continus, les sommes deviennent des intégrales ([[mx.05]]) et la proposition est, par exemple, un pas gaussien autour du point courant ; la dynamique de Langevin ajoute une dérive de gradient à la proposition, ce qui fait le lien avec les méthodes à gradient bruité de la modélisation générative. Les échantillons MCMC sont *corrélés* ; la taille d'échantillon effective est plus petite que le nombre de pas, la quantité de type SLEM (le temps d'autocorrélation) contrôlant de combien.

## Exemple travaillé

Chaîne à trois états utilisée dans tout le code de l'exercice :

$$
P=\begin{pmatrix}0.9&0.1&0\\0.2&0.6&0.2\\0&0.3&0.7\end{pmatrix}.
$$

- En partant de l'état 0, les distributions sont $\mu_1=(0.9,0.1,0)$, $\mu_2=(0.83,0.15,0.02)$, $\mu_5=(0.70125,0.21247,0.08628)$.
- Valeurs propres de $P$ : $1,\ 0.8236,\ 0.3764$ (à partir de $\lambda^2-1.2\lambda+0.31=0$ après avoir factorisé $\lambda=1$). Le SLEM vaut $0.8236$.
- Distribution stationnaire : résous $\pi P=\pi$ avec $\sum\pi_i=1$. Les équations donnent $0.1\pi_0=0.2\pi_1$ et $0.2\pi_1=0.3\pi_2$, c'est-à-dire $\pi=(6,3,2)/11=(0.5455,0.2727,0.1818)$. Vérification : $\pi P$ reproduit $\pi$.
- Convergence : l'erreur maximale $\max_i|\mu_n-\pi|_i$ depuis l'état 0 vaut $0.156$ en $n=5$, $0.059$ en $n=10$, $0.0085$ en $n=20$, $1.7\times10^{-4}$ en $n=40$ ; compare avec $0.8236^{40}\approx4.4\times10^{-4}$ : le même ordre de grandeur, comme le prédit le SLEM.
- Simulation : une trajectoire de 40 000 pas (graine 0) passe les fractions $(0.559,\ 0.268,\ 0.173)$ de son temps dans les trois états, à moins de 0.03 de $\pi$. Des graines différentes diffèrent d'environ cette quantité parce que les états successifs sont corrélés.
- Metropolis sur la cible $(0.5,0.3,0.2)$ avec une proposition uniforme sur les trois états donne la première ligne $(0.667,\ 0.2,\ 0.133)$ ; la matrice complète a la cible pour distribution stationnaire, avec équilibre détaillé.
- Un processus de Poisson avec $\lambda=3$ sur $T=2$ : les comptages simulés (4000 exécutions) ont une moyenne $\approx6.06$ et une variance $\approx6.19$, toutes deux proches de $\lambda T=6$.

## Implémentation

```python
import numpy as np

def stationary(P):
    w, V = np.linalg.eig(P.T)            # left eigenvectors of P
    i = np.argmin(np.abs(w - 1))
    v = np.real(V[:, i])
    return v / v.sum()                   # normalise (also fixes the sign)

def simulate_chain(P, start, steps, rng):
    cum = np.cumsum(P, axis=1)
    u = rng.random(steps)
    s, path = start, np.empty(steps, dtype=int)
    for t in range(steps):
        s = min(int(np.searchsorted(cum[s], u[t], side="right")), len(P) - 1)
        path[t] = s
    return path
```

En pratique : `np.linalg.matrix_power(P, n)` pour des probabilités à $n$ pas exactes ; pour d'énormes chaînes creuses, utilise l'itération $\mu\leftarrow\mu P$. En MCMC, utilise des échantillonneurs établis (Stan, PyMC, emcee, NumPyro) avec des diagnostics de convergence (tracés, $\hat R$).

## Formes (shapes)

- $P$ : $(K,K)$, les lignes somment à 1. Distribution $\mu$ : vecteur ligne $(1,K)$, multiplié *à gauche* : $\mu P$.
- Les vecteurs propres à gauche de $P$ sont les vecteurs propres à droite de $P^\top$ : `np.linalg.eig(P.T)` renvoie les vecteurs propres comme *colonnes* de forme $(K,K)$.
- Une trajectoire de $T$ pas : tableau d'entiers de forme $(T,)$ ; distribution empirique `np.bincount(path, minlength=K)/T` : forme $(K,)$.

## Idées fausses courantes

- « Markov signifie indépendant. » Les états successifs sont dépendants ; ce qui est indépendant, c'est le futur et le passé *sachant le présent*.
- « Lignes ou colonnes, peu importe. » Les conventions diffèrent d'un livre à l'autre (les matrices stochastiques par colonnes agissent sur des vecteurs colonnes). Vérifie toujours quel indice somme à 1.
- « Stationnaire signifie que la chaîne cesse de bouger. » La chaîne continue de bouger ; c'est sa *distribution* qui ne change pas.
- « Un long parcours garantit la convergence. » Les chaînes réductibles ou périodiques ne convergent pas vers une limite unique ; les chaînes MCMC à SLEM élevé peuvent sembler avoir convergé alors qu'elles sont coincées dans un seul mode.
- « Sans mémoire signifie sans état. » L'état peut encoder un long historique (la fenêtre de contexte d'un transformer définit un énorme état), donc la propriété de Markov est un énoncé sur la façon dont tu définis l'état.

## Points clés à retenir

- Une chaîne de Markov est une distribution initiale plus une matrice stochastique par lignes ; $n$ pas plus loin, c'est $\mu_0P^n$.
- Une distribution stationnaire résout $\pi P=\pi$ : un vecteur propre à gauche de valeur propre 1, qui existe toujours.
- Les chaînes ergodiques convergent vers un unique $\pi$ au taux $|\lambda_2|^n$ et les moyennes temporelles égalent les moyennes en $\pi$.
- L'équilibre détaillé donne un moyen de concevoir des chaînes avec une distribution stationnaire choisie ; Metropolis-Hastings n'a besoin que de rapports de densités.
- Les marches aléatoires s'étalent comme $\sqrt n$ ; un processus de Poisson a des intervalles exponentiels et des comptages de Poisson.

## Explique-le toi-même

1. Explique pourquoi les fractions de temps à long terme ne dépendent pas de l'état de départ d'une chaîne ergodique, en utilisant l'image des valeurs propres.
2. Explique à un collègue pourquoi le MCMC n'a pas besoin de la constante de normalisation de la cible.

## Pour aller plus loin

- Bishop, Sec. 13.1-13.2 (modèles de Markov et modèles de Markov cachés), PDF p.627 et suivantes.
- Leçon du dépôt `phases/01-math-foundations/22-stochastic-processes` (mouvement brownien, dynamique de Langevin, code de Metropolis-Hastings).
- Ensuite : [[mx.05]] transforme les sommes sur les états en intégrales sur des variables continues.

## Critères de maîtrise

- [ ] Je sais écrire une matrice de transition à partir d'une description verbale et calculer des probabilités à $n$ pas.
- [ ] Je sais trouver une distribution stationnaire à la fois en résolvant des équations linéaires et comme vecteur propre.
- [ ] Je sais dire laquelle de l'irréductibilité et de l'apériodicité échoue dans un contre-exemple donné.
- [ ] Je sais montrer l'équilibre détaillé pour une chaîne de Metropolis.
