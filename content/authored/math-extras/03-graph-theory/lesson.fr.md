---
id: mx.03
title: La théorie des graphes pour le machine learning
summary: "Transforme des réseaux en matrices, lis la connectivité dans les valeurs propres du laplacien, coupe un graphe avec un seul vecteur propre, et dérive PageRank comme le vecteur propre dominant d'une marche aléatoire."
minutes: 75
teaches: [mx.graph, mx.laplacian, mx.pagerank]
requires: [la.matrix, la.eigen, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/21-graph-theory/docs/en.md (graphs, adjacency matrix, degree, BFS/DFS, Laplacian, spectral properties, message passing, spectral clustering build steps). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md (Markov chain and transition-matrix section, read for the random-walk view of PageRank). READ."
---

## Où cela s'inscrit

Les lignes d'un tableau sont indépendantes ; mais les molécules, les cartes routières, les réseaux de citations, les réseaux sociaux et les bases de connaissances sont définis par des *connexions*. Les réseaux de neurones sur graphes, les systèmes de recommandation, le classement des recherches et le clustering de données non vectorielles partent tous des mêmes quelques objets : la **matrice d'adjacence**, la **matrice des degrés** et le **laplacien**. Le plus remarquable est que les vecteurs propres rencontrés dans [[math.07]] te renseignent sur la forme du graphe : combien de morceaux il a, où le couper, et quels nœuds comptent.

Cette leçon s'appuie sur les matrices ([[math.07]] pour les valeurs propres, les matrices symétriques ont des valeurs propres réelles et des vecteurs propres orthogonaux) et sur les probabilités de base. Elle mène à [[mx.04]], où la marche aléatoire sur un graphe devient une chaîne de Markov générale.

## L'idée en mots simples

- Un graphe, ce sont des points et des traits : des personnes et des amitiés, des villes et des routes, des pages web et des liens. Une matrice peut enregistrer qui est connecté à qui, et alors *multiplier* des matrices répond à des questions sur la marche autour du graphe (deux pas, trois pas...).
- Imagine que chaque nœud porte un nombre, comme une température. La chaleur circule le long des arêtes jusqu'à ce que les voisins s'accordent. Le **laplacien** est la machine qui mesure à quel point les nombres sont en désaccord à travers les arêtes. Si le graphe est en deux morceaux déconnectés, chaque morceau peut avoir pour toujours sa propre température constante : c'est un motif « plat » que le laplacien ne voit pas.
- Si le graphe est *presque* en deux morceaux reliés par un mince pont, il existe un motif presque plat de chaque côté avec un saut au milieu. Il coûte peu en désaccord, et il révèle les deux communautés. Le signe de ce motif te dit de quel côté se trouve chaque nœud. C'est le **clustering spectral**.
- Imagine maintenant un surfeur qui clique sur un lien au hasard sur chaque page, éternellement. Les pages vers lesquelles pointent beaucoup de pages importantes sont plus visitées. La fraction de temps à long terme passée sur chaque page est son **PageRank**, et c'est un vecteur propre de la matrice du surfeur.

## Définitions

Un graphe $G=(V,E)$ a $n$ nœuds et des arêtes entre des paires d'entre eux. Les arêtes non orientées fonctionnent dans les deux sens ; les arêtes orientées (liens, abonnements) non. Numérote les nœuds $0,\dots,n-1$.

- **Matrice d'adjacence** $A\in\{0,1\}^{n\times n}$ avec $A_{ij}=1$ s'il y a une arête $i\to j$ (forme $n\times n$). Graphe non orienté : $A=A^\top$.
- **Degré** du nœud $i$ : $d_i=\sum_jA_{ij}$. **Matrice des degrés** $D=\operatorname{diag}(d_0,\dots,d_{n-1})$.
- Une **marche** (walk) de longueur $k$ de $i$ à $j$ est une suite de $k$ arêtes. Un **chemin** (path) est une marche sans nœud répété. Deux nœuds sont dans la même **composante connexe** si un chemin les relie.

**Compter les marches.** $(A^k)_{ij}$ est égal au nombre de marches de longueur $k$ de $i$ à $j$. Preuve par récurrence : $(A^{k+1})_{ij}=\sum_m(A^k)_{im}A_{mj}$, c'est-à-dire des marches de longueur $k$ jusqu'à un certain $m$, puis une arête de plus jusqu'à $j$. En particulier, $\operatorname{trace}(A^3)=6\times$(nombre de triangles) dans un graphe simple non orienté, puisque chaque triangle est compté depuis 3 nœuds de départ dans 2 sens.

## Le laplacien d'un graphe

$$
L=D-A\qquad(n\times n,\ \text{symmetric}).
$$

Pour un vecteur $x\in\mathbb{R}^n$ (un nombre par nœud), calcule la forme quadratique :

$$
x^\top Lx=x^\top Dx-x^\top Ax=\sum_id_ix_i^2-\sum_{i,j}A_{ij}x_ix_j .
$$

Comme $d_i=\sum_jA_{ij}$, on peut écrire $\sum_id_ix_i^2=\frac12\sum_{i,j}A_{ij}(x_i^2+x_j^2)$. Donc

$$
\boxed{x^\top Lx=\frac12\sum_{i,j}A_{ij}(x_i-x_j)^2=\sum_{\{i,j\}\in E}(x_i-x_j)^2\ \ge 0 .}
$$

(la dernière forme somme chaque arête non orientée une seule fois.) Conséquences :

1. $L$ est **semi-définie positive** : toutes les valeurs propres sont $\ge0$.
2. $L\mathbf 1=0$ (le vecteur constant) : toutes les lignes de $L$ somment à zéro. Donc $0$ est toujours une valeur propre.
3. $x^\top Lx=0$ exactement quand $x_i=x_j$ sur chaque arête, c'est-à-dire que $x$ est constant sur chaque composante connexe. Par conséquent **la multiplicité de la valeur propre 0 est égale au nombre de composantes connexes**, et l'espace propre est engendré par les vecteurs indicateurs des composantes.

### Bissection spectrale

Coupe les nœuds en deux groupes avec des étiquettes $s_i=\pm1$. Une arête coupée a $(s_i-s_j)^2=4$ et une arête non coupée a $0$, donc d'après la formule encadrée $s^\top Ls=4\cdot\text{cut}$, c'est-à-dire $\text{cut}=\tfrac14s^\top Ls$ (le nombre d'arêtes entre les groupes). Minimiser cela sur tous les vecteurs $\pm1$ est un problème combinatoire difficile, et tout mettre dans un seul groupe donne trivialement une coupe nulle. La **relaxation** abandonne l'exigence $\pm1$ : minimise $x^\top Lx$ sous les contraintes $\|x\|=1$ et $x\perp\mathbf 1$ (pour exclure la solution plate triviale). Par l'argument du quotient de Rayleigh de [[math.07]], le minimiseur est le vecteur propre de la **deuxième plus petite valeur propre** $\lambda_2$, le **vecteur de Fiedler**, et la valeur atteinte est $\lambda_2$. Arrondis-le par le signe : $s_i=\operatorname{sign}(x_i)$.

$\lambda_2>0$ si et seulement si le graphe est connexe, et un petit $\lambda_2$ signifie que le graphe se scinde presque (un goulot d'étranglement). Utilise plus de vecteurs propres, puis lance $k$-means ([[cml.04]] : $k$-means appliqué aux lignes de la matrice de vecteurs propres) pour obtenir $k$ clusters. C'est le cœur du **clustering spectral**, utilisé quand les clusters sont des formes connexes plutôt que des blobs ronds, et des variantes normalisées où $L$ est remis à l'échelle par $D^{-1/2}$.

Une remarque qui fait le lien avec [[mx.02]] : sur un anneau de $n$ nœuds, le laplacien est une matrice circulante et ses vecteurs propres sont exactement les ondes de Fourier discrètes. Les vecteurs propres du laplacien sont « l'analyse de Fourier sur un graphe ».

## Marches aléatoires et PageRank

Une **marche aléatoire** sur un graphe orienté passe de façon répétée du nœud courant à l'un de ses voisins sortants choisi uniformément. La matrice de transition est la matrice d'adjacence normalisée par ligne

$$
P=D_{\text{out}}^{-1}A,\qquad P_{ij}=\frac{A_{ij}}{d_i^{\text{out}}},\qquad\text{each row sums to }1 .
$$

Si $r^{(t)}\in\mathbb{R}^n$ est le vecteur (ligne) des probabilités d'être en chaque nœud au temps $t$, alors $r^{(t+1)}=r^{(t)}P$ (formes $1\times n$ fois $n\times n$). Une distribution qui ne change pas, $r=rP$, est un vecteur propre à gauche de $P$ de valeur propre 1 ; nous le rencontrons proprement dans [[mx.04]].

Deux problèmes pour le web : une page sans lien sortant (un nœud **pendant**, *dangling*) est une impasse, et le graphe peut avoir des groupes où le marcheur peut entrer mais dont il ne peut jamais sortir. PageRank corrige les deux avec la **téléportation** : avec probabilité $d$ (le facteur d'amortissement, couramment $0.85$) suis un lien, avec probabilité $1-d$ saute vers une page tirée uniformément au hasard. La **matrice de Google** qui en résulte est

$$
G=dP+\frac{1-d}{n}\mathbf 1\mathbf 1^\top,\qquad\text{(dangling rows of }P\text{ replaced by }\tfrac1n\mathbf 1^\top).
$$

Toutes les entrées de $G$ sont strictement positives et les lignes somment à 1. D'après le théorème de Perron-Frobenius (utilisé ici sans preuve), une telle matrice a une valeur propre 1 qui est simple, toutes les autres valeurs propres ont un module $<1$ (en fait $\le d$), et le vecteur propre à gauche de valeur propre 1 peut être choisi positif. **PageRank est ce vecteur propre**, normalisé pour sommer à 1 :

$$
r=rG,\qquad\sum_ir_i=1 .
$$

De façon équivalente, puisque $r\mathbf 1\mathbf 1^\top=\mathbf 1^\top$ quand $r$ somme à 1,

$$
r=d\,rP+\frac{1-d}{n}\mathbf 1^\top .
$$

L'**itération de puissance** le calcule : pars du vecteur uniforme et applique cette mise à jour de façon répétée. L'erreur diminue au moins comme $d^t$, donc $0.85^{50}\approx3\times10^{-4}$. Avec $n$ en milliards, la matrice $P$ est stockée de façon creuse et une itération est un produit matrice-vecteur creux.

## Exemple travaillé

Tous les nombres sont produits par le code de l'exercice.

**Deux triangles reliés par un pont.** Les nœuds $0,1,2$ forment un triangle, $3,4,5$ un autre, et l'arête $(2,3)$ les relie. Degrés : $(2,2,3,3,2,2)$.

- Valeurs propres du laplacien : $0,\ 0.4384,\ 3,\ 3,\ 3,\ 4.5616$. Exactement un zéro : connexe. $\lambda_2=0.4384$ est petit par rapport aux autres : un goulot d'étranglement.
- Vecteur de Fiedler $\approx(-0.465,-0.465,-0.261,\ 0.261,\ 0.465,\ 0.465)$. Les signes séparent $\{0,1,2\}$ de $\{3,4,5\}$, exactement les deux triangles. Les extrémités du pont ont les plus petites amplitudes.
- Retire le pont : les valeurs propres deviennent $0,0,3,3,3,3$. Deux zéros = deux composantes.
- Marches : $(A^2)_{03}=1$ (seulement via le nœud 2) et $\operatorname{trace}(A^3)=12=6\times2$ triangles.
- Vérification de la forme quadratique : $x=(1,2,0,3,-1,5)$ donne $x^\top Lx=\sum_{\text{edges}}(x_i-x_j)^2$.

**PageRank** sur 4 pages avec les liens $0\to1,\ 0\to2,\ 1\to2,\ 2\to0,\ 2\to3,\ 3\to2$ et $d=0.85$ : 

$$
P=\begin{pmatrix}0&.5&.5&0\\0&0&1&0\\.5&0&0&.5\\0&0&1&0\end{pmatrix}.
$$

L'itération de puissance converge vers $r=(0.2199,\ 0.1310,\ 0.4292,\ 0.2199)$, de somme 1. La page 2 gagne : trois pages pointent vers elle. Les pages 0 et 3 sont à égalité à $0.2199$. Le même vecteur sort du vecteur propre de $G^\top$ pour la valeur propre 1 (calculé avec `np.linalg.eig`) : la valeur propre est $1.0$ aux arrondis près.

## Implémentation

```python
import numpy as np

def laplacian(A):
    return np.diag(A.sum(axis=1)) - A

def spectral_bisection(A):
    w, V = np.linalg.eigh(laplacian(A))      # ascending eigenvalues
    lab = (V[:, 1] > 0).astype(int)           # sign of the Fiedler vector
    return 1 - lab if lab[0] == 1 else lab

def pagerank(M, d=0.85, tol=1e-12):
    n = M.shape[0]
    out = M.sum(axis=1, keepdims=True)
    P = np.where(out > 0, M / np.where(out > 0, out, 1), 1.0 / n)  # dangling -> uniform
    r = np.ones(n) / n
    while True:
        r_new = d * (r @ P) + (1 - d) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
```

En production : `scipy.sparse` et `scipy.sparse.linalg.eigsh` pour de grands laplaciens creux (ne calculer que quelques vecteurs propres), `networkx.pagerank`, `sklearn.cluster.SpectralClustering` ; les bibliothèques de GNN (PyTorch Geometric, DGL) stockent l'adjacence comme une liste d'arêtes, pas comme une matrice dense.

## Formes (shapes)

- $A,D,L,P,G$ : $(n,n)$. Caractéristiques des nœuds $H$ : $(n,f)$. Un tour de passage de messages de type GNN est $\hat A H$ avec $\hat A=D^{-1}(A+I)$ ou une normalisation symétrique : $(n,n)\times(n,f)\to(n,f)$, chaque nœud faisant la moyenne de lui-même et de ses voisins.
- Vecteur de rang : ligne $(1,n)$ quand il est multiplié à gauche de $P$ ; en colonne, on écrirait $P^\top r$.
- Vecteurs propres du laplacien : colonnes de $V\in\mathbb{R}^{n\times n}$ issues de `eigh`, ordonnées par valeur propre croissante.

## Idées fausses courantes

- « La valeur propre 0 apparaît une fois. » Elle apparaît une fois *par composante connexe*.
- « Le vecteur de Fiedler donne la coupe minimale. » Il donne une solution relaxée qui est arrondie ; elle peut être fausse, surtout pour des groupes déséquilibrés. Les variantes normalisées fonctionnent mieux sur des graphes aux degrés très différents.
- « PageRank exige que le graphe soit connexe. » La téléportation supprime cette exigence.
- « $\sum_j P_{ij}=1$ signifie que $P$ est symétrique. » Les lignes somment à 1 ; les colonnes n'ont pas à le faire.
- « Les vecteurs propres du laplacien à grandes valeurs propres sont du bruit. » Ce sont les motifs de haute fréquence, qui alternent à travers les arêtes ; les petits sont les lisses. Les GNN qui lissent les caractéristiques agissent comme un filtre passe-bas sur ce spectre.

## Points clés à retenir

- $A$ stocke les arêtes, $D$ les degrés, $L=D-A$ ; $(A^k)_{ij}$ compte les marches.
- $x^\top Lx=\sum_{\text{edges}}(x_i-x_j)^2$ : $L$ mesure la rugosité sur le graphe et est semi-définie positive.
- Le nombre de valeurs propres nulles est le nombre de composantes ; le signe du vecteur de Fiedler coupe un graphe à son goulot d'étranglement le plus étroit.
- PageRank est la distribution stationnaire d'une marche aléatoire amortie, un vecteur propre de la matrice de Google, trouvé par itération de puissance.

## Explique-le toi-même

1. Pourquoi la forme quadratique $x^\top Lx$ s'annule-t-elle exactement sur les vecteurs constants sur chaque composante ?
2. Pourquoi la téléportation est-elle nécessaire dans PageRank, et que contrôle le facteur d'amortissement $d$ ?

## Pour aller plus loin

- Leçon du dépôt `phases/01-math-foundations/21-graph-theory` (BFS/DFS, passage de messages).
- Ensuite : [[mx.04]] pour la théorie générale des chaînes de Markov derrière PageRank.

## Critères de maîtrise

- [ ] Je sais construire $A$, $D$, $L$ pour un petit graphe et vérifier $x^\top Lx$ à la main.
- [ ] Je sais expliquer pourquoi la multiplicité de la valeur propre 0 compte les composantes.
- [ ] Je sais lancer la bissection spectrale et PageRank à partir de zéro.
- [ ] Je sais dire ce que chacune des corrections (amortissement et nœuds pendants) répare.
