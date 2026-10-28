---
id: cml.08
title: "Machines à vecteurs de support"
summary: "Choisir l'hyperplan séparateur de marge maximale, dériver le primal à marge souple et son dual lagrangien, voir la lecture par la perte charnière (hinge), et ajouter des noyaux pour des frontières courbes."
minutes: 85
teaches: [ml.svm]
requires: [ml.logreg, la.dot, opt.gradient-descent, ml.loss]
sources:
  - "math/math for ML.pdf (Deisenroth et al.), Ch. 12 Classification with Support Vector Machines, Sec. 12.1-12.4 (PDF p.378-396): separating hyperplanes, margin, hard and soft margin, hinge loss view, Lagrangian dual, kernels. READ (read selectively; 12.3.2 convex-hull view read, only mentioned)."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 7.1 Maximum Margin Classifiers (PDF p.346-348 and p.352-354): margin, canonical representation, slack variables, dual with box constraints, which points are support vectors. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/05-support-vector-machines/docs/en.md (hinge loss, subgradient training of a linear SVM, when SVMs still win). READ (hinge, gradient descent and 'why SVMs lost' sections)."
---

## Où cela s'inscrit

La régression logistique ([[cml.02]]) te donne *une* droite séparatrice et une probabilité. Mais quand deux classes sont séparables, il existe une infinité de droites séparatrices, et certaines sont fragiles : un minuscule déplacement d'un point d'entraînement pourrait inverser une prédiction de test. La machine à vecteurs de support (SVM) choisit celle qui est la *plus éloignée* des deux classes. Elle a dominé l'apprentissage automatique sur les problèmes de taille moyenne (classification de textes avec des variables creuses de mots, bio-informatique, chiffres manuscrits) avant l'apprentissage profond, et reste un bon choix pour les données petites ou de grande dimension. C'est aussi le lieu de rencontre de trois idées que tu réutiliseras : les marges comme régularisation, la dualité lagrangienne (voir la leçon d'optimisation sous contraintes `math.12`), et l'astuce du noyau, qui réapparaît dans les processus gaussiens et, dans l'esprit, dans l'attention.

## L'idée en mots simples

Imagine deux groupes de maisons de part et d'autre d'une rivière, et tu dois construire une route entre elles. Toute route qui garde toutes les maisons d'un groupe d'un côté convient, mais tu préfères la construire aussi loin que possible de chaque maison, afin qu'une maison en bordure légèrement déplacée reste du bon côté. La bande vide entre la route et les maisons les plus proches est la **marge**. Seules les maisons qui touchent la bande décident du tracé de la route : supprime n'importe quelle autre maison et la route ne bouge pas ; ces maisons de bordure sont les **vecteurs de support**.

Les données réelles sont désordonnées : quelques maisons peuvent se trouver dans la bande ou du mauvais côté. La **marge souple** l'autorise mais fait payer une pénalité par violation, et un cadran $C$ dit combien coûte une violation. Un grand $C$ insiste pour classer correctement les points d'entraînement (marge étroite, risque de surapprentissage) ; un petit $C$ tolère des erreurs pour une route plus large et plus simple.

Quand aucune route droite ne convient (une classe encercle l'autre), tu peux soulever les points dans un espace de dimension supérieure où une nappe plane les sépare. L'**astuce du noyau** réalise ce soulèvement implicitement : tu ne calcules jamais les nouvelles coordonnées, seulement les similarités entre paires de points.

## Cadre et marge

Variables $x\in\mathbb R^D$, étiquettes $y_n\in\{-1,+1\}$, score $f(x)=\langle w,x\rangle+b$, prédiction $\operatorname{sign}f(x)$. L'hyperplan est $\{x:f(x)=0\}$ et $w$ lui est normal (prends deux points $x_a,x_b$ dessus : $f(x_a)-f(x_b)=\langle w,x_a-x_b\rangle=0$).

**Distance à l'hyperplan.** Écris $x_a=x_a'+r\,\frac{w}{\|w\|}$ avec $x_a'$ la projection orthogonale de $x_a$ sur l'hyperplan, donc $f(x_a')=0$. Alors

$$
f(x_a)=\langle w,x_a'\rangle+b+r\frac{\langle w,w\rangle}{\|w\|}=0+r\|w\|\quad\Longrightarrow\quad r=\frac{f(x_a)}{\|w\|}.
$$

Pour un point bien classé, la distance est $y_nf(x_n)/\|w\|$ (Bishop Éq. 7.2). La marge est la plus petite de ces distances.

**Fixer l'échelle.** Remettre à l'échelle $(w,b)\to(\kappa w,\kappa b)$ laisse toutes les distances inchangées, donc on peut choisir l'échelle de sorte que le point le plus proche vérifie $y_nf(x_n)=1$. C'est la représentation *canonique*. Tous les points vérifient alors $y_nf(x_n)\ge1$ et la marge vaut $r=1/\|w\|$. Maximiser $1/\|w\|$ revient à minimiser $\tfrac12\|w\|^2$ (transformation monotone, et le $\tfrac12$ simplifie le gradient).

## Les problèmes primaux

**Marge dure** (données séparables) :

$$
\min_{w,b}\ \tfrac12\|w\|^2\quad\text{s.t.}\quad y_n(\langle w,x_n\rangle+b)\ge1,\ n=1..N.
$$

**Marge souple.** Donne à chaque point une variable d'écart (slack) $\xi_n\ge0$ qui lui permet de violer la marge ($0<\xi\le1$ : dans la marge mais correct ; $\xi>1$ : mal classé) :

$$
\min_{w,b,\xi}\ \tfrac12\|w\|^2+C\sum_n\xi_n\quad\text{s.t.}\quad y_n(\langle w,x_n\rangle+b)\ge1-\xi_n,\ \ \xi_n\ge0 .
$$

$C\to\infty$ retrouve la marge dure. Remarque que $\sum_n\xi_n$ majore le nombre d'erreurs d'entraînement (Bishop p.332). $b$ n'est pas régularisé.

## La lecture par la perte charnière (hinge)

Pour $(w,b)$ fixé, le meilleur écart est le plus petit autorisé : $\xi_n=\max\{0,\,1-y_nf(x_n)\}$. En substituant, on obtient un problème sans contrainte

$$
\min_{w,b}\ \underbrace{\tfrac12\|w\|^2}_{\text{regulariser}}+C\sum_n\underbrace{\max\{0,\,1-y_nf(x_n)\}}_{\text{hinge loss}} .
$$

Avec $t=yf(x)$, la perte charnière vaut $0$ quand $t\ge1$ (correct et hors de la marge), et $1-t$ sinon, croissant linéairement même pour des points corrects dans la marge. C'est un majorant convexe de la perte 0-1, qui est elle-même combinatoire à optimiser (Deisenroth Sec. 12.2.5). Une SVM est donc de la « minimisation du risque empirique avec perte charnière + régularisation $\ell_2$ » ; la marge est le régularisateur. Compare avec la perte logistique $\ln(1+e^{-t})$ : lisse et jamais exactement nulle, alors que la charnière est exactement nulle au-delà de la marge, ce qui explique pourquoi seuls certains points comptent.

## Dérivation : le dual lagrangien

(Contexte d'optimisation sous contraintes : `math.12` ; voici ce dont on a besoin.) Introduis des multiplicateurs $\alpha_n\ge0$ pour les contraintes de marge et $\gamma_n\ge0$ pour $\xi_n\ge0$ :

$$
\mathcal L=\tfrac12\|w\|^2+C\sum_n\xi_n-\sum_n\alpha_n\big(y_n(\langle w,x_n\rangle+b)-1+\xi_n\big)-\sum_n\gamma_n\xi_n .
$$

Annule les dérivées par rapport aux variables primales :

$$
\frac{\partial\mathcal L}{\partial w}=0\Rightarrow w=\sum_n\alpha_ny_nx_n,\qquad
\frac{\partial\mathcal L}{\partial b}=0\Rightarrow\sum_n\alpha_ny_n=0,\qquad
\frac{\partial\mathcal L}{\partial \xi_n}=0\Rightarrow C-\alpha_n-\gamma_n=0 .
$$

La dernière relation avec $\gamma_n\ge0$ donne $\alpha_n\le C$. Substitue en retour : le terme en $b$ disparaît par la deuxième condition, les termes en $\xi$ par la troisième, et les termes en $w$ se combinent en $\tfrac12\|w\|^2-\|w\|^2=-\tfrac12\|w\|^2$ où $\|w\|^2=\sum_{i,j}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle$. Le résultat est le **dual** :

$$
\min_{\alpha}\ \tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle-\sum_i\alpha_i\quad\text{s.t.}\quad\sum_i\alpha_iy_i=0,\ \ 0\le\alpha_i\le C .
$$

Conséquences :

- **Forme de représentation.** $w=\sum_n\alpha_ny_nx_n$ : la solution est une combinaison de points d'entraînement. Le relâchement complémentaire ($\alpha_n(y_nf(x_n)-1+\xi_n)=0$) dit que $\alpha_n>0$ seulement pour les points sur la marge, dans la marge ou mal classés : les **vecteurs de support**. Les points avec $0<\alpha_n<C$ sont exactement sur la marge ($\xi_n=0$), ceux avec $\alpha_n=C$ peuvent la violer (Bishop p.334).
- **Récupérer $b$** à partir d'un point de marge : $b=y_n-\langle w,x_n\rangle$ (moyenne sur ces points pour la stabilité).
- **Seuls les produits scalaires apparaissent**, $\langle x_i,x_j\rangle$. Le dual a $N$ variables au lieu de $D$ : intéressant quand $D\gg N$, et c'est la porte vers les noyaux.

## Noyaux

Remplace $x$ par une application de variables $\phi(x)$. Le dual n'a besoin que de $k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$, donc on peut *définir* directement une similarité $k$. Un noyau valide donne une matrice de Gram $K_{ij}=k(x_i,x_j)$ symétrique semi-définie positive pour tout jeu de données. La prédiction devient

$$
f(x)=\sum_n\alpha_ny_n\,k(x_n,x)+b .
$$

Exemples. Le noyau polynomial $k(x,z)=(\langle x,z\rangle+1)^2$ en 2D égale $\langle\phi(x),\phi(z)\rangle$ avec $\phi(x)=(1,\sqrt2x_1,\sqrt2x_2,x_1^2,\sqrt2x_1x_2,x_2^2)$ (développe le carré pour vérifier), et coûte un produit scalaire au lieu de six variables. Le **noyau RBF (gaussien)**

$$
k(x,z)=\exp(-\gamma\|x-z\|^2)
$$

correspond à un espace de variables de dimension infinie. Il vaut 1 pour des points identiques et décroît vers 0 avec la distance ; $\gamma$ fixe son degré de localité : un grand $\gamma$ donne des frontières ondulées qui peuvent surapprendre, un petit $\gamma$ s'approche d'un classifieur linéaire. Le modèle appris reste un classifieur *linéaire* dans l'espace des variables ; la frontière n'est courbe que dans l'espace d'origine. $C$ et $\gamma$ se règlent normalement par validation croisée.

## Entraîner une SVM linéaire par descente de sous-gradient

La perte charnière a un point anguleux en $t=1$, donc on utilise un **sous-gradient**. Divise l'objectif par $CN$ et pose $\lambda=1/(CN)$ :

$$
J(w,b)=\frac\lambda2\|w\|^2+\frac1N\sum_n\max\{0,\,1-y_n(\langle w,x_n\rangle+b)\}.
$$

Soit $\mathcal V=\{n:y_nf(x_n)<1\}$ l'ensemble des violateurs de marge. Un sous-gradient valide est

$$
g_w=\lambda w-\frac1N\sum_{n\in\mathcal V}y_nx_n,\qquad g_b=-\frac1N\sum_{n\in\mathcal V}y_n .
$$

Les non-violateurs ne contribuent à rien : c'est la même parcimonie que dans le dual. Mets à jour $w\leftarrow w-\eta_tg_w$ avec un pas décroissant $\eta_t$, car avec un objectif non lisse un pas fixe rebondit autour du minimum. La perte n'est pas monotone, donc juge selon la tendance de l'objectif plutôt qu'à chaque pas.

## Exemple détaillé

Prends $x_1=(2,0),\,y_1=+1$ et $x_2=(-2,0),\,y_2=-1$ (plus deux points éloignés $(3,1)^+,(-3,-1)^-$).

- Par symétrie, la meilleure frontière est la droite verticale $x_1=0$ : $w=(w_1,0),\,b=0$. Contraintes $y f\ge1$ aux points les plus proches : $2w_1\ge1$, donc $w_1=\tfrac12$. Marge $=1/\|w\|=2$ (en effet la distance de $(2,0)$ à la droite $x_1=0$). Les points éloignés ont $yf=1.5>1$ : ce ne sont pas des vecteurs de support.
- Vérification par le dual : $w=\sum\alpha_ny_nx_n=\alpha\cdot(2,0)+\alpha\cdot(2,0)=(4\alpha,0)$, donc $\alpha_1=\alpha_2=\tfrac18$, et $\sum\alpha_ny_n=0$ est vérifié. L'objectif dual vaut $\sum\alpha-\tfrac12\sum_{ij}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle=\tfrac14-\tfrac12\cdot\tfrac{16}{64}=\tfrac18$, et le primal $\tfrac12\|w\|^2=\tfrac12\cdot\tfrac14=\tfrac18$ : égaux, comme l'exige la dualité forte.
- Sous-gradient en $w=0,\,b=0$, avec les quatre points, $\lambda=0.1$ : chaque point viole la marge ($yf=0<1$). $g_w=-\tfrac14\big[(2,0)+(2,0)+(3,1)+(3,1)\big]=(-2.5,-0.5)$ et $g_b=-\tfrac14(1-1+1-1)=0$.

Les exercices reproduisent ces nombres.

## Implémentation

```python
import numpy as np

def svm_subgradient(w, b, X, y, lam):
    viol = y * (X @ w + b) < 1                   # margin violators
    gw = lam * w - (y[viol, None] * X[viol]).sum(0) / len(y)
    gb = -y[viol].sum() / len(y)
    return gw, gb

def svm_fit(X, y, lam=0.01, lr=0.5, epochs=500):
    w, b = np.zeros(X.shape[1]), 0.0
    for t in range(epochs):
        gw, gb = svm_subgradient(w, b, X, y, lam)
        step = lr / (1 + 0.05 * t)              # decaying step size
        w, b = w - step * gw, b - step * gb
    return w, b
```

## Formes

`X: (N, D)`, `y: (N,)` dans $\{-1,+1\}$, `w: (D,)`, `b: scalaire`, scores `X @ w + b: (N,)`. Matrice de noyau (Gram) `K: (N, N)` ; pour des points de test `K_test: (M, N)` et prédictions `K_test @ (alpha * y) + b: (M,)`. La matrice de Gram explique pourquoi les SVM à noyau passent mal à l'échelle au-delà de quelques dizaines de milliers de points : mémoire $O(N^2)$, entraînement entre $O(N^2)$ et $O(N^3)$.

## Idées reçues

- **« La SVM maximise la marge de tous les points. »** Seuls les vecteurs de support déterminent la solution ; déplacer les autres points (sans franchir la marge) ne change rien.
- **Oublier de mettre les variables à l'échelle.** La marge se mesure en distance euclidienne, et le noyau RBF l'utilise aussi ; une variable à grandes unités domine. Standardise d'abord.
- **« Un grand $C$ est meilleur. »** Un grand $C$ ajuste étroitement les données d'entraînement et peut surapprendre des données bruitées ; choisis $C$ par validation.
- **« Une SVM à noyau travaille dans l'espace d'origine. »** Elle est linéaire dans un espace de variables implicite, donc sa frontière est courbe dans l'espace d'entrée.
- **Les sorties d'une SVM ne sont pas des probabilités.** Le score $f(x)$ est une quantité de type distance signée ; des probabilités demandent une étape de calibration supplémentaire.
- **« Le dual est toujours meilleur. »** Pour un grand $N$ avec un $D$ modéré, le primal (SVM linéaire par SGD) est bien moins coûteux.

## Points clés à retenir

- Marge $=1/\|w\|$ dans l'échelle canonique ; marge dure : $\min\tfrac12\|w\|^2$ s.t. $y_nf(x_n)\ge1$.
- La marge souple ajoute des écarts et une pénalité $C\sum\xi_n$ ; de façon équivalente, perte charnière + régularisateur $\ell_2$.
- Le dual (issu du lagrangien) a $0\le\alpha_n\le C$, $\sum\alpha_ny_n=0$, seulement des produits scalaires, et $w=\sum\alpha_ny_nx_n$ ; les $\alpha_n$ non nuls sont les vecteurs de support.
- Les noyaux remplacent les produits scalaires ; le RBF donne des frontières courbes avec $\gamma$ contrôlant la localité.
- Une SVM linéaire peut être entraînée directement dans le primal par descente de sous-gradient sur la perte charnière.

## Explique-le toi-même

1. Pourquoi peut-on supposer que le point le plus proche a $y_nf(x_n)=1$ sans perte de généralité ?
2. À partir du lagrangien, explique pourquoi le dual ne contient que des produits scalaires $\langle x_i,x_j\rangle$.

## Pour aller plus loin

- Deisenroth, Faisal, Ong, *Mathematics for ML*, Ch. 12 (PDF p.376-398) ; Bishop, *PRML*, Sec. 7.1.
- Leçon du dépôt `phases/02-ml-fundamentals/05-support-vector-machines`.
- Lié : [[cml.02]] (perte logistique contre perte charnière).

## Critères de maîtrise

Tu sais dériver $r=1/\|w\|$, écrire le primal à marge souple et sa forme par perte charnière, dériver le dual à partir du lagrangien, identifier les vecteurs de support à partir de $\alpha_n$, évaluer une fonction de décision à noyau RBF, et entraîner une SVM linéaire avec une boucle de sous-gradient.
