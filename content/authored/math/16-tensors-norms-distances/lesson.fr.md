---
id: math.16
title: "Tenseurs, normes et distances"
summary: "Les formes que prennent les données en apprentissage profond, les règles qui permettent de combiner des tableaux de formes différentes, et les normes, distances et similarités servant à mesurer la taille et la proximité, y compris pourquoi les distances perdent leur sens en grande dimension."
minutes: 75
teaches: [la.tensor, la.norms]
requires: [la.vector, la.dot, la.matrix, la.svd, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/12-tensor-operations/docs/en.md: tensor rank/shape, typical deep-learning shapes (B,C,H,W), (B,T,D), (B,H,T,D), strides and contiguity, broadcasting rule with the (8,1,6,1) x (7,1,5) example, einsum patterns. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/14-norms-and-distances/docs/en.md: L1, L2, Lp, Linf norms, cosine similarity, Mahalanobis distance (definition and intuition via whitening), links to MAE/MSE and L1/L2 regularisation. READ."
  - "math/math for ML.pdf, Sec. 3.1 Norms (PDF p.77-78, book p.71-72): Definition 3.1 (absolutely homogeneous, triangle inequality, positive definite), Manhattan norm, Euclidean norm. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.4 The Curse of Dimensionality (PDF p.53-57): exponential growth of grid cells, volume of a sphere concentrated in a thin shell 1-(1-eps)^D, Gaussian mass in a thin shell, why learning still works (low effective dimension, smoothness). READ."
---

## Où cela s'inscrit

Chaque nombre qu'un réseau de neurones manipule vit dans un tableau. Un lot d'images est de dimension quatre, un lot de phrases de dimension trois, un tableau de scores d'attention de dimension quatre. Écrire et déboguer du code d'apprentissage profond, c'est en grande partie **suivre les formes** : quel axe est le lot, lequel est la caractéristique, et comment les tableaux s'alignent quand tu les additionnes ou les multiplies. La première moitié de cette leçon te donne ce vocabulaire : tenseurs, formes, diffusion (broadcasting) et einsum.

La seconde moitié répond à une question différente mais liée : **quelle est la taille d'un vecteur, et à quelle distance sont deux vecteurs ?** Les fonctions de perte de la leçon 8, les régulariseurs des leçons ridge et lasso, la recherche du plus proche voisin, les plongements et la recherche d'information, la détection d'anomalies et le partitionnement ont tous besoin d'une mesure de taille ou de proximité, et le choix de la mesure change la réponse. Nous terminons par le fait surprenant qu'en grande dimension « le plus proche » et « le plus lointain » deviennent presque la même chose, ce qui explique pourquoi les données de grande dimension exigent une réflexion soigneuse.

Tu utiliseras les leçons 2 et 3 (vecteurs, produits scalaires, matrices), 7 (les valeurs singulières apparaissent dans la norme spectrale) et 9 (la covariance pour la distance de Mahalanobis).

## L'idée en mots simples

- **Un tenseur est un tableur généralisé à un nombre quelconque d'axes.** Un nombre seul est un point. Une liste de nombres est une colonne. Un tableau est une grille de lignes et de colonnes. Une pile de tableaux (par exemple un tableau par canal de couleur d'une image, ou un par exemple d'un lot) ajoute un troisième axe, et ainsi de suite. La **forme** est la liste des longueurs le long de chaque axe : un lot de 32 images couleur de taille 224 par 224 a pour forme `(32, 3, 224, 224)`. La forme est le contrat : les opérations sont légales ou non selon elle.
- **La diffusion (broadcasting) consiste à « étirer le petit pour qu'il corresponde ».** Si tu ajoutes un seul prix à toute une colonne de prix, ou un vecteur de biais à chaque ligne d'un lot, tu ne veux pas le copier à la main. NumPy étire conceptuellement le plus petit tableau le long des axes où sa longueur vaut 1 (ou manque) pour que les formes correspondent, sans réellement copier la mémoire.
- **Einsum est de la comptabilité par étiquettes.** Donne une lettre à chaque axe ; les lettres qui apparaissent en entrée mais pas en sortie sont sommées. Le produit matriciel, le produit scalaire, la trace et les scores d'attention deviennent tous des énoncés d'une ligne en étiquettes.
- **Une norme est une règle.** Des règles différentes donnent des longueurs différentes à la même flèche. La règle « marcher le long des pâtés de maisons » (L1), la règle « voler en ligne droite » (L2) et la règle « seule la pire direction compte » (L-infini) ne s'accordent pas. La **distance** entre deux points est la longueur de la flèche qui les relie.
- **La similarité cosinus compare des directions, pas des tailles.** Deux documents sur le même sujet dont l'un est dix fois plus long ont des vecteurs qui pointent dans la même direction mais de longueurs différentes ; l'angle voit à travers cela.
- **La distance de Mahalanobis est une distance mesurée en unités de la propre dispersion des données.** Une personne de 20 cm plus grande que la moyenne n'est pas remarquable parmi des basketteurs, mais l'est dans la population générale. Et quelqu'un d'inhabituellement grand *et* inhabituellement léger est plus surprenant que ne le suggèrent les écarts individuels, car taille et poids vont ensemble. Mahalanobis tient compte à la fois de l'échelle et de la corrélation.
- **Les grandes dimensions sont étranges.** En mille dimensions, un point aléatoire est presque à la même distance de tout autre point aléatoire. Presque tout le volume d'une boule se trouve dans une fine peau près de sa surface.

## Tenseurs et formes

Un **tenseur** est un tableau multidimensionnel de nombres d'un seul type. Son **rang** (nombre d'axes) et sa **forme** (n-uplet de longueurs) :

| objet | forme | rang |
|---|---|---|
| scalaire | `()` | 0 |
| vecteur de $\mathbb R^3$ | `(3,)` | 1 |
| matrice $2\times3$ | `(2, 3)` | 2 |
| lot d'images | `(B, C, H, W)` | 4 |
| lot de plongements de tokens | `(B, T, D)` | 3 |
| scores d'attention | `(B, heads, T, T)` | 4 |
| poids d'une couche linéaire | `(out, in)` | 2 |

Le nombre d'éléments est le produit des tailles : la forme `(2, 3, 4)` en a 24. `reshape` regroupe autrement les mêmes éléments dans le même ordre mémoire (le total doit correspondre ; une taille peut valoir `-1`, ce qui signifie « la déduire »). `transpose` et `permute` réordonnent les axes. En mémoire, un tenseur est un bloc plat et les **pas** (strides) disent combien d'éléments sauter pour avancer d'un cran le long de chaque axe : la forme `(3, 4)` en ordre ligne-majeur a pour pas `(4, 1)`. Transposer échange les pas sans déplacer de données, ce qui explique pourquoi un tableau transposé n'est pas contigu et pourquoi `view` de PyTorch peut échouer là où `reshape` fonctionne.

Les **réductions** (`sum`, `mean`, `max`) écrasent les axes que tu nommes : la moyenne de `(B, T, D)` selon `axis=1` donne `(B, D)` ; le pooling moyen global de `(B, C, H, W)` sur les axes `(2, 3)` donne `(B, C)`.

## Diffusion (broadcasting)

**Règle.** Aligne les formes par la **droite**. Les axes de tête manquants comptent pour une taille 1. Deux tailles alignées sont compatibles si elles sont égales ou si l'une vaut 1 ; le résultat prend la plus grande. Sinon, c'est une erreur.

Exemple (celui de la leçon du dépôt) : `(8, 1, 6, 1)` avec `(7, 1, 5)`. Complète la seconde en `(1, 7, 1, 5)`, compare axe par axe depuis la droite : $1$ contre $5\to5$, $6$ contre $1\to6$, $1$ contre $7\to7$, $8$ contre $1\to8$. Forme du résultat `(8, 7, 6, 5)`.

Schémas courants : un biais `(D,)` ajouté à des activations `(B, T, D)` ; la standardisation `X - X.mean(axis=0)` où la moyenne a la forme `(D,)` face à `(N, D)` ; et le bug classique où `(N,)` moins `(N, 1)` se diffuse silencieusement en `(N, N)`. Quand tu gardes un axe réduit pour une diffusion ultérieure, utilise `keepdims=True` (de sorte que le max dans un softmax ait la forme `(B, 1)`, pas `(B,)`).

## Einsum

`np.einsum("subscripts->output", ...)`. Règle : une lettre absente de la sortie est sommée ; une lettre répétée dans un même opérande prend la diagonale.

| indices | signification |
|---|---|
| `i,i->` | produit scalaire |
| `i,j->ij` | produit extérieur |
| `ii->` | trace |
| `ij->ji` | transposition |
| `ik,kj->ij` | produit matriciel |
| `bij,bjk->bik` | produit matriciel par lots |
| `bhtd,bhsd->bhts` | scores d'attention : requêtes `(B,H,T,D)` contre clés `(B,H,S,D)` |

Pour la multiplication matricielle, `ik,kj->ij` dit littéralement $C_{ij}=\sum_k A_{ik}B_{kj}$ : l'indice $k$ n'apparaît pas à droite, donc il est sommé. Lire le code d'un modèle devient bien plus facile une fois que tu sais traduire une chaîne d'indices en formule de somme.

## Normes

Une **norme** est une fonction $\|\cdot\|$ qui assigne une longueur à chaque vecteur telle que (Deisenroth Déf. 3.1), pour tous scalaires $\lambda$ et vecteurs $x,y$ :

- absolument homogène : $\|\lambda x\|=|\lambda|\,\|x\|$ ;
- inégalité triangulaire : $\|x+y\|\le\|x\|+\|y\|$ ;
- définie positive : $\|x\|\ge0$ et $\|x\|=0\iff x=0$.

La famille $\ell_p$ : $\|x\|_p=\big(\sum_i|x_i|^p\big)^{1/p}$ pour $p\ge1$.

| norme | formule | boule unité en 2-D | où on la rencontre |
|---|---|---|---|
| L1 (Manhattan) | $\sum_i\lvert x_i\rvert$ | losange | lasso (parcimonie), perte MAE |
| L2 (euclidienne) | $\sqrt{\sum_i x_i^2}=\sqrt{x^\top x}$ | cercle | ridge, MSE, par défaut |
| $L_\infty$ (Tchebychev) | $\max_i\lvert x_i\rvert$ | carré | erreur dans le pire cas, bornes de perturbations adversariales |

$\|x\|_\infty$ est la limite de $\|x\|_p$ quand $p\to\infty$ : la plus grande composante domine de plus en plus. Pour $x=(3,-4,12)$ : $\|x\|_1=19$, $\|x\|_2=13$, $\|x\|_\infty=12$, $\|x\|_3=12.21$. Les normes vérifient $\|x\|_\infty\le\|x\|_2\le\|x\|_1$. Le losange a des coins sur les axes, ce qui est la raison géométrique pour laquelle une pénalité L1 pousse des coordonnées à exactement zéro alors qu'une pénalité L2 ne fait que les rétrécir.

### Normes matricielles

- **Frobenius** : traite la matrice comme un long vecteur, $\|A\|_F=\sqrt{\sum_{ij}a_{ij}^2}=\sqrt{\operatorname{tr}(A^\top A)}=\sqrt{\sum_k\sigma_k^2}$ (les égalités utilisent la SVD de la leçon 7 : $A^\top A=V\Sigma^\top\Sigma V^\top$ a pour valeurs propres $\sigma_k^2$, et la trace est leur somme).
- **Spectrale (norme 2 d'opérateur)** : le plus grand étirement que la matrice peut appliquer à un vecteur unité, $\|A\|_2=\max_{\|x\|_2=1}\|Ax\|_2=\sigma_{\max}$.
- Également utilisées : la somme maximale des colonnes (norme 1 d'opérateur) et la somme maximale des lignes (norme $\infty$ d'opérateur).

Pour $A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$ : valeurs singulières $5.4650$ et $0.3660$, donc $\|A\|_2=5.4650$ et $\|A\|_F=\sqrt{30}=5.4772$ (vérification : $5.4650^2+0.3660^2=29.866+0.134=30$). La norme spectrale est ce qui contrôle de combien une couche peut amplifier les signaux et les gradients, et c'est la grandeur derrière la normalisation spectrale et le rapport $\sigma_{\max}/\sigma_{\min}$ du conditionnement de la leçon 15.

## Distances et similarités

Toute norme donne une distance : $d(a,b)=\|a-b\|$.

**Euclidienne** $d_2(a,b)=\|a-b\|_2$. **Manhattan** $d_1(a,b)=\sum_i|a_i-b_i|$. Pour $a=(1,1)$, $b=(4,5)$ : $d_1=7$, $d_2=5$, $d_\infty=4$.

La **similarité cosinus** mesure l'angle (elle réutilise l'identité du produit scalaire $a\cdot b=\|a\|\|b\|\cos\theta$ de la leçon 2) :

$$
\cos(a,b)=\frac{a^\top b}{\|a\|_2\,\|b\|_2}\in[-1,1],\qquad d_{\cos}=1-\cos(a,b).
$$

Elle ignore l'échelle : $(1,2,3)$ et $(2,4,6)$ ont un cosinus de 1 bien que leur distance euclidienne soit $\sqrt{14}=3.74$. Une fois les vecteurs normalisés à la longueur unité, $\|a-b\|_2^2=2-2\cos(a,b)$, donc le classement euclidien et le classement cosinus coïncident. (Démonstration : $\|a-b\|^2=\|a\|^2+\|b\|^2-2a^\top b=1+1-2\cos$.) Remarque : la distance cosinus ne vérifie pas l'inégalité triangulaire, donc ce n'est pas une métrique au sens strict.

**Distance de Mahalanobis** pour des données de moyenne $\mu$ et de covariance $\Sigma$ :

$$
d_M(x,\mu)=\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)} .
$$

*Pourquoi cette formule.* Factorise $\Sigma=U\Lambda U^\top$ (décomposition en valeurs propres, leçon 7). Alors $\Sigma^{-1/2}=U\Lambda^{-1/2}U^\top$ et $d_M=\|\Sigma^{-1/2}(x-\mu)\|_2$ : tourne sur les axes principaux, divise chaque coordonnée par son écart-type $\sqrt{\lambda_k}$, puis prends la longueur euclidienne ordinaire. Le nuage de données devient une boule ronde (« blanchiment »). Quand $\Sigma=I$, c'est la simple distance euclidienne.

**Exemple détaillé.** $\Sigma=\begin{pmatrix}4&2\\2&3\end{pmatrix}$ (corrélée positivement, $\mu=0$), $\det\Sigma=8$, $\Sigma^{-1}=\frac18\begin{pmatrix}3&-2\\-2&4\end{pmatrix}$.

| point | norme euclidienne | Mahalanobis |
|---|---|---|
| $(2,0)$ | 2.000 | $\sqrt{4\cdot3/8}=1.2247$ |
| $(0,2)$ | 2.000 | $\sqrt{4\cdot4/8}=1.4142$ |
| $(2,2)$ | 2.828 | $\sqrt{(12-16+16)/8}=1.2247$ |
| $(2,-2)$ | 2.828 | $\sqrt{(12+16+16)/8}=2.3452$ |

(Pour $(2,2)$ : $x^\top\Sigma^{-1}x=\frac18(3\cdot4-2\cdot2\cdot4+4\cdot4)=\frac{12}{8}$, d'où $1.2247$ ; pour $(2,-2)$ : $\frac18(12+16+16)=5.5$, d'où $2.3452$.) Le point $(2,2)$ est plus loin de l'origine au sens euclidien que $(2,0)$ et pourtant tout aussi typique, car il se trouve le long de la direction dans laquelle les données s'étalent naturellement. Le point $(2,-2)$ va contre la corrélation et est le vrai point aberrant. C'est pourquoi la distance de Mahalanobis sert à la détection de points aberrants, et pourquoi les densités gaussiennes en dépendent (l'exposant de la loi normale multivariée est $-\tfrac12 d_M^2$).

## Le fléau de la dimension

Bishop (Sec. 1.4) fait trois remarques liées.

1. **Les grilles explosent.** Diviser chacun des $D$ axes en $k$ cellules donne $k^D$ cellules ; il faudrait exponentiellement beaucoup de données pour que toutes contiennent des exemples. Un polynôme d'ordre $M$ en $D$ variables a environ $D^M$ coefficients.
2. **Le volume passe dans la coquille.** Le volume d'une $D$-boule de rayon $r$ varie comme $r^D$. La fraction du volume de la boule unité située à moins de $\varepsilon$ de la surface est $\dfrac{V(1)-V(1-\varepsilon)}{V(1)}=1-(1-\varepsilon)^D$. Avec $\varepsilon=0.05$ : $D=2$ donne $9.8\%$, $D=20$ donne $64.2\%$, $D=100$ donne $99.4\%$. La masse de probabilité d'une gaussienne se concentre aussi à un rayon d'environ $\sqrt D$ : pour des tirages normaux standard, la norme a une moyenne $\approx\sqrt D$ (10.0 pour $D=100$) avec un écart-type d'environ $0.7$, presque indépendant de $D$.
3. **Les distances se concentrent.** Pour des points aléatoires, la distance au plus proche voisin et au plus lointain deviennent presque égales. Prends 200 points uniformes dans le cube unité et une requête aléatoire $q$ ; le contraste relatif $(d_{\max}-d_{\min})/d_{\min}$ vaut $230.6$ pour $D=2$, $2.20$ pour $D=10$, $0.316$ pour $D=100$, $0.118$ pour $D=1000$ et $0.029$ pour $D=10000$ (graine 0, reproduit dans l'exercice). Intuitivement, chaque distance au carré est une somme de $D$ termes de coordonnées indépendants, donc par la loi des grands nombres elle se concentre autour de sa moyenne $\propto D$ tandis que la dispersion ne croît que comme $\sqrt D$. La recherche du plus proche voisin et toute méthode reposant sur la proximité euclidienne brute se dégradent.

Pourquoi le machine learning marche quand même (Bishop p.37) : les données réelles se trouvent près d'une variété de dimension plus faible, et la fonction cible est généralement localement lisse. C'est pourquoi on réduit la dimension avec l'ACP (leçon 7 et la leçon sur l'ACP), on apprend des plongements, et on préfère souvent le cosinus ou des métriques apprises à la distance euclidienne brute.

## Implémentation

```python
import numpy as np

def lp_norm(x, p):
    return np.max(np.abs(x)) if np.isinf(p) else (np.abs(x) ** p).sum() ** (1 / p)

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

def mahalanobis(x, mu, S):
    d = x - mu
    return np.sqrt(d @ np.linalg.solve(S, d))   # solve instead of an explicit inverse

# attention scores: queries (B,H,T,D), keys (B,H,S,D) -> (B,H,T,S)
# scores = np.einsum("bhtd,bhsd->bhts", Q, K)
```

Utilise `np.linalg.solve` plutôt que `np.linalg.inv(S) @ d` : c'est plus rapide et plus précis (leçon 15).

## Formes

Distances euclidiennes deux à deux entre `X: (N, D)` et `Y: (M, D)` : développe le carré, $\|x-y\|^2=\|x\|^2+\|y\|^2-2x^\top y$, donc `D2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T` avec les formes `(N,1)`, `(1,M)` et `(N,M)` qui se diffusent en `(N, M)`. (De minuscules valeurs négatives peuvent apparaître par annulation ; borne à 0 avant la racine carrée.) La matrice de similarité cosinus de `Xn: (N, D)`, `Yn: (M, D)` normalisées par ligne est `Xn @ Yn.T: (N, M)`.

## Idées fausses courantes

- **« La diffusion copie des données. »** Non ; elle réutilise la mémoire en donnant à l'axe étiré un pas de 0. Mais elle peut cacher des bugs comme `(N,)` moins `(N,1)` donnant `(N,N)`.
- **« La distance L2 est toujours la bonne distance. »** Des tâches différentes veulent des métriques différentes : L1 pour la robustesse et la parcimonie, le cosinus pour la direction, Mahalanobis pour des variables corrélées ou d'échelles différentes.
- **« Une similarité cosinus de 0 signifie sans rapport. »** Elle signifie orthogonal ; pour des données centrées c'est une corrélation nulle, pour des données positives (comptages de mots) cela signifie aucune caractéristique partagée.
- **« Plus de variables aident toujours les méthodes de plus proches voisins. »** Au-delà de la dimension effective, des variables bruitées supplémentaires diluent le contraste entre proche et lointain.
- **« Les normes de Frobenius et spectrale sont identiques. »** $\|A\|_F^2=\sum\sigma_k^2$ tandis que $\|A\|_2=\sigma_{\max}$ ; elles ne coïncident que pour les matrices de rang un.

## Points clés à retenir

- La forme est le contrat. La diffusion aligne par la droite et étire les axes de taille 1 ; les lettres d'einsum absentes de la sortie sont sommées.
- $\|x\|_\infty\le\|x\|_2\le\|x\|_1$ ; L1 a des coins (parcimonie), L2 est rond (rétrécissement). $\|A\|_F=\sqrt{\sum\sigma^2}$, $\|A\|_2=\sigma_{\max}$.
- Le cosinus ignore la longueur ; pour des vecteurs unité $\|a-b\|^2=2-2\cos$. Mahalanobis blanchit par $\Sigma^{-1/2}$.
- En grande dimension le volume se trouve dans une fine coquille et les distances se concentrent ; compte sur la faible dimension effective et la régularité.

## Explique-le avec tes mots

1. Un coéquipier ajoute un vecteur `(32,)` de poids par exemple à une colonne `(32, 1)` de pertes et obtient une matrice `(32, 32)`. Explique exactement ce qu'a fait la diffusion et comment corriger.
2. Explique à un non-spécialiste pourquoi deux points distants de 2 unités au sens euclidien peuvent être « tout aussi typiques » que la moyenne pour une paire et un net point aberrant pour une autre, et quelle distance corrige cela.

## Pour aller plus loin

- Deisenroth et al., chap. 3 (produits scalaires, normes, angles, projections orthogonales).
- Bishop, Sec. 1.4 et ses figures 1.21-1.23.
- Dépôt, phase 01, leçons 12 et 14 (davantage de distances : Jaccard, distance d'édition, divergence KL, Wasserstein ; organisation mémoire et pas en profondeur).

## Critères de maîtrise

- Prédire la forme de sortie d'une diffusion et d'une chaîne einsum sans les exécuter.
- Calculer les normes L1, L2, $L_\infty$, de Frobenius et spectrale et expliquer comment Frobenius et spectrale se rattachent aux valeurs singulières.
- Montrer que, pour des vecteurs unité, la distance euclidienne est une fonction monotone du cosinus.
- Calculer une distance de Mahalanobis et l'expliquer comme un blanchiment suivi d'une longueur euclidienne.
- Expliquer la concentration des distances et une raison pour laquelle l'apprentissage reste possible en grande dimension.
