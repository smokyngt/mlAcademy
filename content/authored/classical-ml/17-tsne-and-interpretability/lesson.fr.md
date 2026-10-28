---
id: cml.17
title: "Au-delà de l'ACP, et regarder à l'intérieur des modèles : t-SNE, ACP à noyau, importance par permutation, dépendance partielle, valeurs de Shapley"
summary: "Plonger les données de façon non linéaire (affinités de t-SNE et son gradient, ACP à noyau), puis expliquer un modèle ajusté avec l'importance par permutation, la dépendance partielle et les valeurs de Shapley, et apprendre exactement ce que ces explications peuvent et ne peuvent pas t'apprendre."
minutes: 90
teaches: [ml.tsne, ml.kpca, ml.interpret, ml.shap]
requires: [ml.pca, la.eigen, info.entropy, calc.gradient, ml.features]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/10-dimensionality-reduction/docs/en.md (curse of dimensionality, t-SNE properties and perplexity, kernel PCA algorithm and comparison table). READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/docs/en.md (permutation importance section, bias of tree importance). READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 12.3 Kernel PCA (PDF p.606-610): kernel-only eigenproblem K a = lambda N a, normalisation, centred Gram matrix, projections, N x N cost, linear kernel recovers PCA, pre-image problem. READ."
---

## Où cela s'inscrit

[[cml.05]] t'a donné l'ACP (PCA), la meilleure compression *linéaire*. Deux choses tournent mal en pratique. D'abord, les données reposent souvent sur une surface courbe (une spirale, un anneau, des variations d'écriture manuscrite) qu'aucun sous-espace plat ne capture ; **t-SNE** et l'**ACP à noyau** sont les réponses non linéaires standards, la première pour fabriquer une image 2D, la seconde comme extracteur de variables. Ensuite, une fois que tu as un bon modèle (une forêt, un ensemble boosté, un réseau), tu dois pouvoir dire *quelles entrées il utilise et comment*. L'**importance par permutation**, la **dépendance partielle** et les **valeurs de Shapley** sont des sondes indépendantes du modèle qui fonctionnent sur n'importe quelle fonction `predict` ajustée.

Les deux moitiés répondent à la même question, « quelle est la structure ici ? », l'une sur les données et l'autre sur le modèle. La seconde moitié se rattache aussi à la sélection de variables ([[cml.16]]) et à la détection de fuites : une variable à l'importance suspectement énorme est souvent une fuite.

## L'idée en mots simples

**t-SNE.** Suppose que chaque point de données est une personne à une fête et que tu demandes à chacun « qui sont tes plus proches amis ? ». Dans l'espace d'origine, de grande dimension, chaque personne répartit une unité d'attention sur les autres, surtout sur les plus proches. Demande maintenant à un plan de table 2D de reproduire ces amitiés : les amis s'assoient ensemble, les étrangers à l'écart, et là où le plan est bondé, une règle à queues lourdes permet aux non-amis de s'asseoir confortablement loin. t-SNE conserve les voisinages ; il ne conserve ni les distances entre groupes, ni leurs tailles.

**ACP à noyau.** Un anneau ne peut pas être aplati par une droite. Mais si chaque point est décrit par *sa ressemblance avec chaque autre point* (un noyau), l'anneau devient facile : fais une ACP sur ces similarités.

**Importance par permutation.** Pour tester si un modèle utilise une variable, brouille cette colonne entre les lignes et regarde de combien l'erreur augmente. Si rien ne change, le modèle l'ignore.

**Dépendance partielle.** Force une variable à chaque valeur d'une grille pour chaque ligne, moyenne les prédictions, et trace la courbe moyennée : la réponse moyenne du modèle à cette variable.

**Valeurs de Shapley.** Traite les variables comme une équipe qui a produit conjointement la prédiction et répartis le mérite équitablement, en moyennant la contribution marginale de chaque variable sur tous les ordres dans lesquels l'équipe pourrait être constituée.

## t-SNE

Soit $x_1,\dots,x_N\in\mathbb R^D$. Le but est d'obtenir des points $y_1,\dots,y_N\in\mathbb R^2$.

**Affinités en grande dimension.** Pour chaque point $i$, utilise une gaussienne centrée en $x_i$ :

$$
p_{j\mid i}=\frac{\exp\!\big(-\|x_i-x_j\|^2/2\sigma_i^2\big)}{\sum_{k\ne i}\exp\!\big(-\|x_i-x_k\|^2/2\sigma_i^2\big)},\qquad p_{i\mid i}=0 .
$$

La largeur $\sigma_i$ est choisie point par point de sorte que la **perplexité** $\mathrm{Perp}(P_i)=\exp\big(H(P_i)\big)$, avec $H(P_i)=-\sum_jp_{j\mid i}\ln p_{j\mid i}$, égale un réglage de l'utilisateur (typiquement 5-50) : c'est le nombre effectif de voisins. Les régions denses obtiennent un petit $\sigma_i$, les régions clairsemées un grand $\sigma_i$. Comme $H$ croît avec $\sigma_i$, on trouve $\sigma_i$ (ou $\beta_i=1/2\sigma_i^2$) par bissection. Puis on symétrise pour que chaque point contribue :

$$
p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2N},\qquad \sum_{i\ne j}p_{ij}=1 .
$$

**Affinités en basse dimension.** Avec un noyau de Student (un degré de liberté, queues lourdes) :

$$
w_{ij}=\frac1{1+\|y_i-y_j\|^2},\qquad q_{ij}=\frac{w_{ij}}{\sum_{k\ne l}w_{kl}} .
$$

Les queues lourdes résolvent le « problème d'entassement » (crowding problem) : une carte 2D a moins de place autour d'un point qu'un espace à 10 dimensions, et un noyau à queues lourdes permet de placer des points modérément éloignés très loin les uns des autres sans pénalité.

**Objectif.** Faire ressembler $Q$ à $P$ : minimiser $C=\mathrm{KL}(P\|Q)=\sum_{i\ne j}p_{ij}\ln\frac{p_{ij}}{q_{ij}}$.

### Dérivation du gradient

Écris $d_{ij}^2=\|y_i-y_j\|^2$ et $Z=\sum_{k\ne l}w_{kl}$. Comme $\sum p_{ij}=1$,

$$
C=\sum p_{ij}\ln p_{ij}-\sum p_{ij}\ln w_{ij}+\ln Z .
$$

Avec $\ln w_{ij}=-\ln(1+d_{ij}^2)$ : $\ \partial(-p_{ij}\ln w_{ij})/\partial d_{ij}^2=p_{ij}w_{ij}$, et $\partial w_{ij}/\partial d_{ij}^2=-w_{ij}^2$ donc $\partial\ln Z/\partial d_{ij}^2=-w_{ij}^2/Z=-q_{ij}w_{ij}$. Chaque paire non ordonnée apparaît deux fois dans les sommes, d'où $\partial C/\partial d_{ij}^2=2(p_{ij}-q_{ij})w_{ij}$. Enfin $\partial d_{ij}^2/\partial y_i=2(y_i-y_j)$, donc

$$
\boxed{\frac{\partial C}{\partial y_i}=4\sum_{j}(p_{ij}-q_{ij})\,w_{ij}\,(y_i-y_j)}
$$

Sous forme matricielle avec $M=(P-Q)\odot W$ : $\ \nabla_Y C=4\big(\mathrm{diag}(M\mathbf 1)-M\big)Y$. Cela se lit comme un système de ressorts : les paires avec $p_{ij}>q_{ij}$ (trop éloignées sur la carte) s'attirent, les paires avec $p_{ij}<q_{ij}$ se repoussent. Un pas de descente de gradient est $Y\leftarrow Y-\eta\nabla_YC$ (les implémentations réelles ajoutent de l'élan, une « exagération précoce » de $P$, et des approximations de Barnes-Hut ou par FFT pour éviter le coût en $O(N^2)$).

J'ai vérifié numériquement cette formule contre des différences finies dans l'exercice plutôt que de la citer d'après les sources que j'ai lues ; fais toi-même la vérification.

**Ce que t-SNE ne promet pas.** Les distances *entre* amas, les tailles d'amas et les densités n'ont pas de sens ; des graines différentes donnent des dispositions différentes ; la perplexité change l'image (trop basse, elle disloque les amas, trop haute, elle les fusionne) ; la carte sert à regarder, pas à prétraiter un modèle en aval, et les nouveaux points n'ont pas de projection hors échantillon. UMAP est un cousin plus rapide qui conserve un peu plus de structure globale.

## ACP à noyau

Remplace chaque point par un vecteur de variables $\phi(x)$ et fais l'ACP dans cet espace, mais uniquement à travers le noyau $k(x,x')=\phi(x)^\top\phi(x')$. Suivant Bishop Sec. 12.3 : si les vecteurs propres dans l'espace des variables sont $v_i=\sum_na_{in}\phi(x_n)$, substituer dans $Cv_i=\lambda_iv_i$ et multiplier par $\phi(x_l)^\top$ donne un problème aux valeurs propres de la matrice de Gram $N\times N$, $K\mathbf a_i=\lambda_iN\mathbf a_i$ ; exiger $\|v_i\|=1$ normalise $\lambda_iN\,\mathbf a_i^\top\mathbf a_i=1$. La projection d'un point est $y_i(x)=\sum_na_{in}k(x,x_n)$. Comme $\phi$ est rarement de moyenne nulle, la matrice de Gram doit être **centrée** :

$$
\tilde K=K-\mathbf 1_NK-K\mathbf 1_N+\mathbf 1_NK\mathbf 1_N=JKJ,\qquad J=I-\tfrac1N\mathbf 1\mathbf 1^\top ,
$$

avec $\mathbf 1_N$ la matrice dont toutes les entrées valent $1/N$. Recette pratique : construis le noyau RBF $K_{ij}=\exp(-\gamma\|x_i-x_j\|^2)$, centre-le, prends les premiers couples propres $(\mu_i,u_i)$ de $\tilde K$ ; les coordonnées de l'ensemble d'entraînement sont $u_i\sqrt{\mu_i}$. Un noyau linéaire redonne l'ACP ordinaire. Coûts : une matrice $N\times N$ (mémoire $O(N^2)$, décomposition en valeurs propres $O(N^3)$), pas de reconstruction directe (le problème de la pré-image), et $\gamma$ doit être réglé.

## Importance par permutation

Étant donnés un modèle ajusté, des données mises de côté et une métrique, le score de base est $s_0$. Pour la variable $j$, permute sa colonne entre les lignes (ce qui casse son lien avec $y$ tout en gardant sa distribution marginale), recalcule le score $s_j$, répète plusieurs fois et moyenne : $\mathrm{imp}_j=\mathbb E[s_j-s_0]$ (pour une métrique d'erreur). Utilise des données mises de côté : sur l'ensemble d'entraînement, un modèle qui surapprend peut faire paraître importantes des variables de bruit. Mises en garde : avec deux variables fortement corrélées, le modèle peut se rabattre sur l'autre copie, si bien que les deux paraissent sans importance ; permuter crée des lignes irréalistes quand les variables sont dépendantes ; l'importance mesure la *dépendance de ce modèle*, pas la pertinence réelle de la variable. Elle évite le biais de cardinalité de l'importance d'arbre fondée sur l'impureté.

## Dépendance partielle

$$
\mathrm{PD}_j(g)=\frac1N\sum_{n=1}^N f\big(x_n^{(1)},\dots,x_n^{(j)}\!=\!g,\dots,x_n^{(d)}\big).
$$

Elle montre l'effet moyen de la variable $j$ et est exacte quand le modèle est additif en $j$. Faiblesses : les moyennes cachent l'hétérogénéité (une variable qui aide la moitié des lignes et nuit à l'autre moitié montre une droite plate ; trace des courbes individuelles, **ICE**, pour le voir), et avec des variables corrélées elle évalue le modèle en des points irréalistes.

## Valeurs de Shapley (l'idée derrière SHAP)

Pour une instance $x$ à $d$ variables, définis une fonction de valeur $v(S)$ = prédiction espérée quand les variables de la coalition $S$ sont fixées aux valeurs de $x$ et que les autres sont tirées d'un jeu de données de fond (background). La valeur de Shapley de la variable $j$ est sa contribution marginale moyennée sur toutes les coalitions des autres variables :

$$
\phi_j=\sum_{S\subseteq\{1..d\}\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\Big[v(S\cup\{j\})-v(S)\Big].
$$

Propriétés : **efficience** $\sum_j\phi_j=f(x)-\mathbb E[f]$ (l'explication s'additionne exactement), symétrie, une valeur nulle pour une variable qui ne compte jamais, et linéarité. Pour un modèle linéaire $f(x)=w^\top x$, cela donne $\phi_j=w_j(x_j-\mathbb E x_j)$. Le calcul exact demande $2^{d}$ coalitions ; les bibliothèques SHAP utilisent l'échantillonnage (KernelSHAP) ou exploitent la structure des arbres (TreeSHAP). Les choix comptent : utiliser une distribution de fond qui ignore les corrélations plutôt que des espérances conditionnelles donne des réponses différentes.

## Limites de l'interprétabilité

- Les explications décrivent le **modèle**, pas le monde : une variable à forte attribution peut être un indicateur indirect (proxy) ou une fuite.
- Elles sont **locales** (une prédiction) ou **moyennées** (une courbe) ; aucune ne révèle entièrement les interactions.
- Les variables corrélées se partagent ou se cachent le mérite ; les résultats dépendent des données de fond et de la métrique.
- Les explications peuvent être instables d'un ré-entraînement à l'autre, et peuvent servir à paraître convaincant (explanation washing).
- Utilise-les pour déboguer (trouver des fuites, vérifier la cohérence, repérer la dépendance à des variables sensibles), et utilise un modèle intrinsèquement simple quand les enjeux exigent une règle fidèle et auditable.

## Exemple détaillé

**Entrées de t-SNE.** 20 points dans $\mathbb R^5$ formant deux amas serrés de 10 (centres 0 et 3, dispersion $0.3$). Perplexité 5 : chaque ligne conditionnelle a $\exp H=5.000$. La matrice $P$ symétrisée somme à 1, chaque amas porte exactement une masse de $0.5$ et la masse inter-amas est inférieure à $10^{-6}$. En partant de $Y\sim\mathcal N(0,10^{-4})$ (donc $Q$ presque uniforme), $\mathrm{KL}(P\|Q)=1.2303$ ; le gradient analytique concorde avec les différences finies centrées à environ $2\times10^{-10}$ près ; un pas avec $\eta=10$ abaisse la KL à $1.2300$ (un pas minuscule car le gradient est petit à ce départ quasi effondré ; les vraies exécutions utilisent plusieurs centaines d'itérations avec exagération précoce).

**ACP à noyau.** Deux cercles concentriques (rayon 1 et 3, 40 points chacun), RBF avec $\gamma=0.5$ : les plus grandes valeurs propres du noyau centré sont $10.70$ et $8.64$, et la première composante du noyau est parfaitement corrélée ($|r|=1$) à l'étiquette du cercle, alors que le premier score d'ACP linéaire ne l'est pas.

**Interprétation.** Données : $n=400$, quatre variables normales standard, $y=2x_0+x_1^2+0.3\varepsilon$. Un modèle des moindres carrés avec les variables $[1,x_0,x_1,x_1^2,x_2,x_3]$ trouve des coefficients proches de $1.97$ pour $x_0$ et $1.00$ pour $x_1^2$. L'EQM de base (calculée ici sur les données d'entraînement, par simplicité) est $0.0967$. Importance par permutation (20 répétitions) : $x_0$ : $7.63$, $x_1$ : $3.12$, $x_2$ : $\approx0.00001$, $x_3$ : $\approx0.001$. Dépendance partielle de $x_1$ en $(-2,-1,0,1,2)$ : $(4.05,1.03,-0.01,0.96,3.92)$, une parabole ; de $x_0$ : une droite de pente $1.974$. Valeurs de Shapley pour $x=(1,2,0,0)$ : $(1.949,3.009,0.0005,0.0017)$ dont la somme $4.960$ égale exactement $f(x)-\mathbb E f$. Tout est reproduit par les exercices.

## Implémentation

```python
import numpy as np

def tsne_grad(P, Y):
    D = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    W = 1.0 / (1.0 + D); np.fill_diagonal(W, 0)
    Q = W / W.sum()
    M = (P - Q) * W
    return 4 * (np.diag(M.sum(1)) - M) @ Y            # (N, 2)

def permutation_importance(predict, X, y, n_repeats, rng):
    base = np.mean((y - predict(X)) ** 2); imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        v = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            v.append(np.mean((y - predict(Xp)) ** 2) - base)
        imp[j] = np.mean(v)
    return base, imp
```

Production : `sklearn.manifold.TSNE`, `sklearn.decomposition.KernelPCA`, `sklearn.inspection.permutation_importance` et `partial_dependence`, le paquet `shap`, `umap-learn`.

## Dimensions

t-SNE : `X: (N, D)`, `P, Q, W: (N, N)`, `Y, grad: (N, 2)`. ACP à noyau : `K: (N, N)`, scores `(N, k)`. Interprétation : `predict: (N, d) -> (N,)` ; importances `(d,)` ; courbe PD `(len(grid),)` ; valeurs de Shapley `(d,)` par instance.

## Idées reçues

- **« Les distances entre amas de t-SNE ont un sens. »** Non ; les tailles d'amas non plus.
- **« Une exécution de t-SNE est la vérité. »** Relance avec d'autres graines et perplexités.
- **« L'ACP à noyau réduit la dimension gratuitement. »** Elle exige une matrice $N\times N$ et ne donne pas d'application inverse facile.
- **« Une forte importance par permutation signifie causal. »** Elle signifie que ce modèle s'appuie sur la variable.
- **« La dépendance partielle, c'est ce qui arrive à une prédiction. »** C'est une moyenne ; utilise ICE ou les valeurs de Shapley pour des lignes individuelles.
- **« Les valeurs de Shapley expliquent le monde. »** Elles attribuent équitablement la sortie du modèle, étant donnée une distribution de fond choisie.

## Points clés

- t-SNE fait coïncider des distributions de voisinage $P$ (gaussienne, calibrée par la perplexité) et $Q$ (Student) en minimisant la KL ; gradient $4\sum_j(p_{ij}-q_{ij})w_{ij}(y_i-y_j)$.
- L'ACP à noyau est une ACP sur la matrice de Gram centrée $JKJ$.
- L'importance par permutation, la dépendance partielle et les valeurs de Shapley sondent n'importe quel modèle ; chacune a des modes de défaillance avec les variables corrélées.
- Les valeurs de Shapley sont efficientes : les attributions somment exactement à $f(x)-\mathbb E f$.

## Explique-le toi-même

1. Pourquoi t-SNE utilise-t-il un noyau à queues lourdes sur la carte, et qu'est-ce qui irait mal avec une gaussienne ?
2. Deux variables sont presque des copies l'une de l'autre et le modèle utilise les deux. Prédis ce que rapporte l'importance par permutation et pourquoi.

## Pour aller plus loin

- Bishop, *PRML*, Sec. 12.3 (ACP à noyau, dont la figure à noyau gaussien p. 610 du PDF).
- van der Maaten et Hinton, « Visualizing data using t-SNE » (2008) ; Wattenberg et al., « How to use t-SNE effectively » (distill.pub, interactif ; cité dans la leçon source).
- Lundberg et Lee (2017) pour SHAP. Aucun de ces trois n'a été ouvert pour cette leçon.

## Critères de maîtrise

Tu sais calculer les affinités de t-SNE avec une recherche de perplexité, dériver et vérifier son gradient, réaliser une ACP à noyau avec une matrice de Gram centrée, implémenter l'importance par permutation, la dépendance partielle et les valeurs de Shapley exactes, et énoncer précisément ce que chaque explication montre et ne montre pas.
