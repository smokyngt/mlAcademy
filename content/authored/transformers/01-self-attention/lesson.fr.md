---
id: trf.01
title: "L'auto-attention à partir de zéro"
summary: "Requêtes, clés et valeurs, la formule de l'attention par produit scalaire normalisé démontrée (y compris pourquoi on divise par la racine carrée de d_k), les formes des tenseurs à chaque étape, et une implémentation NumPy avec un test numérique de la variance."
minutes: 80
teaches: [trf.attention]
requires: [la.matrix, la.dot, nn.layer, calc.chain]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/docs/en.md (why recurrence was the bottleneck, O(N^2) attention memory). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md (Q/K/V projections, attention matrix, softmax, formula, scaling motivation). READ."
  - "LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb (Sec. 3.3 simple attention with the 6-token 'Your journey starts with one step' example, Sec. 3.4 trainable Q/K/V with scaling by sqrt(d_k)). READ."
---

## Objectifs d'apprentissage

- Expliquer ce que sont les requêtes, les clés et les valeurs grâce à l'image de la recherche « floue », et pourquoi une même entrée produit trois vecteurs différents.
- Démontrer $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ et prouver que le facteur $1/\sqrt{d_k}$ maintient la variance des scores à 1.
- Donner la forme de chaque tenseur, avec et sans dimension de batch.
- Énoncer le coût en temps et en mémoire de l'attention en fonction de la longueur de séquence $T$.
- Implémenter l'attention par batch en NumPy et tester l'affirmation sur la variance par simulation.

## Où cela s'inscrit

Le langage, le code, l'audio et l'ADN arrivent tous sous forme de **séquences** : un ordre de tokens où le sens de l'un dépend des autres, parfois très éloignés (« Le trophée ne rentrait pas dans la valise parce qu'**il** était trop gros » : que désigne « il » ?). Avant 2017, l'outil standard était un réseau récurrent, qui lit les tokens un par un et comprime tout ce qu'il a vu jusque-là dans un unique vecteur caché. Deux problèmes en découlent : le calcul ne peut pas être parallélisé le long de la séquence (l'étape $t+1$ a besoin de l'étape $t$), et l'information du token 1 doit survivre à de nombreuses compressions pour atteindre le token 50.

**L'auto-attention** supprime les deux problèmes avec une idée différente : laisser chaque token regarder directement chaque autre token et décider lui-même de l'importance de chacun. Toutes ces consultations s'écrivent comme quelques multiplications matricielles, ce que les GPU font très bien. Cette leçon construit cette unique opération. Les leçons suivantes l'étendent (plusieurs têtes et masques, puis l'information de position, puis le bloc complet) et forment ensemble l'architecture à la base de tous les modèles de langage modernes.

Tu vas utiliser ce que tu connais déjà : les produits matriciels (la.matrix), les produits scalaires comme mesure de similarité (la.dot) et la notation de couche `X @ W` de la leçon 9.

## L'idée en mots simples

Imagine une bibliothèque avec un bibliothécaire très accommodant. Tu entres avec une question écrite sur une fiche (la **requête**). Chaque livre sur les étagères porte une étiquette sur sa tranche (la **clé**) et contient un vrai contenu (la **valeur**). Une bibliothèque normale fait une correspondance exacte : ta fiche correspond à une étiquette, tu obtiens un livre. Le bibliothécaire accommodant, lui, compare ta fiche avec *toutes* les étiquettes, attribue à chaque livre un score de pertinence, et te remet un **mélange** de tous les livres, les plus pertinents contribuant le plus.

L'auto-attention, c'est cela, effectué par chaque token en même temps, où les « livres » sont les autres tokens de la même phrase :

- La **requête** du token $i$ dit « quel type d'information est-ce que je cherche ? ».
- La **clé** du token $j$ dit « quel type d'information est-ce que je détiens ? ».
- La **valeur** du token $j$ dit « voici ce que je transmettrai si tu me choisis ».

Pourquoi trois vecteurs différents à partir d'un seul embedding ? Parce que *trouver* un token et *utiliser* un token sont deux tâches différentes. Le mot « banque » peut annoncer (clé) qu'il est un nom lié à des lieux, tandis que ce qu'il transmet (valeur) est une description plus riche. Trois applications linéaires apprises permettent au modèle de décider indépendamment ce qu'il cherche, ce qu'il annonce et ce qu'il livre.

Deux images de plus :

- Le score de pertinence est un **produit scalaire** : grand quand la requête et la clé pointent dans la même direction. C'est une similarité apprise.
- Les scores sont transformés en **pourcentages** dont la somme vaut 100 % (softmax), et la sortie est la moyenne des valeurs pondérée par ces pourcentages. La sortie d'un token est donc un nouveau vecteur qui contient déjà de l'information collectée dans toute la phrase.

## Définitions

Soit une séquence de $T$ tokens représentée par une matrice d'embeddings, une ligne par token :

$$
X\in\mathbb{R}^{T\times d_{\text{model}}}.
$$

Trois matrices de poids apprises la projettent :

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V,
$$

avec $W_Q,W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}$ et $W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}$, de sorte que $Q,K\in\mathbb{R}^{T\times d_k}$ et $V\in\mathbb{R}^{T\times d_v}$. L'attention par produit scalaire normalisé est

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V,
$$

où le softmax est appliqué **à chaque ligne** de la matrice de scores $T\times T$.

## Démonstration : construire la formule en cinq petites étapes

**Étape 1 : un score de pertinence pour une paire.** Le token $i$ a une requête $q_i\in\mathbb{R}^{d_k}$ (ligne $i$ de $Q$) et le token $j$ a une clé $k_j$. Leur score brut est le produit scalaire $s_{ij}=q_i\cdot k_j=\sum_{m=1}^{d_k}q_{im}k_{jm}$.

**Étape 2 : toutes les paires d'un coup.** L'entrée $(i,j)$ de $QK^\top$ vaut exactement $q_i\cdot k_j$, car $(QK^\top)_{ij}=\sum_m Q_{im}K_{jm}$. Formes : $(T\times d_k)(d_k\times T)=(T\times T)$. La ligne $i$ indique « combien le token $i$ s'intéresse à chaque token ».

**Étape 3 : normalisation d'échelle (justifiée plus bas).** $S=QK^\top/\sqrt{d_k}$, toujours $T\times T$.

**Étape 4 : normaliser chaque ligne.** Les poids du token $i$ sont $a_{ij}=\dfrac{e^{S_{ij}}}{\sum_{j'}e^{S_{ij'}}}$. Alors $a_{ij}\ge0$ et $\sum_j a_{ij}=1$. Sous forme matricielle $A=\operatorname{softmax}_{\text{rows}}(S)$, de forme $(T\times T)$.

**Étape 5 : mélanger les valeurs.** La sortie du token $i$ est $z_i=\sum_j a_{ij}v_j$. En empilant les lignes : $Z=AV$, de forme $(T\times T)(T\times d_v)=(T\times d_v)$. Chaque ligne de $Z$ est une combinaison convexe (une moyenne pondérée) de lignes de $V$.

### Pourquoi diviser par $\sqrt{d_k}$ : l'argument de la variance

Suppose, comme modèle d'un réseau fraîchement initialisé, que les entrées de $q$ et $k$ sont indépendantes, de moyenne 0 et de variance 1. Considère un score $s=\sum_{m=1}^{d_k}q_mk_m$.

- Chaque produit a pour moyenne $\mathbb{E}[q_mk_m]=\mathbb{E}[q_m]\,\mathbb{E}[k_m]=0$ (indépendance).
- Chaque produit a pour variance $\operatorname{Var}(q_mk_m)=\mathbb{E}[q_m^2]\,\mathbb{E}[k_m^2]-0=1\cdot1=1$.
- Les $d_k$ produits sont indépendants, et la variance d'une somme de termes indépendants est la somme des variances. Donc

$$
\operatorname{Var}(s)=d_k.
$$

L'écart type d'un score brut croît donc comme $\sqrt{d_k}$ : avec $d_k=64$ les scores typiques sont d'environ $\pm8$, avec $d_k=512$ d'environ $\pm23$. Diviser par $\sqrt{d_k}$ utilise $\operatorname{Var}(cs)=c^2\operatorname{Var}(s)$ avec $c=1/\sqrt{d_k}$ :

$$
\operatorname{Var}\!\Big(\frac{s}{\sqrt{d_k}}\Big)=\frac{d_k}{d_k}=1.
$$

**Pourquoi de grands scores posent problème.** Le softmax de nombres très dispersés est presque one-hot : un poids vaut $\approx1$, les autres $\approx0$. La dérivée du softmax $p=\operatorname{softmax}(s)$ est $\partial p_i/\partial s_j=p_i(\delta_{ij}-p_j)$. Si $p$ est one-hot, alors $p_i\in\{0,1\}$ et chaque entrée de cette jacobienne vaut $0$ : le gradient s'évanouit, et la couche cesse d'apprendre précisément là où elle a fait un choix tranché (peut-être faux). Garder la variance à 1 maintient le softmax dans sa zone sensible à l'initialisation.

### Coût

La matrice de scores a $T^2$ entrées, chacune coûtant $d_k$ multiplications-additions : $O(T^2d_k)$ en temps. La stocker coûte $O(T^2)$ en mémoire par tête. Le produit $AV$ coûte encore $O(T^2d_v)$. Doubler la longueur du contexte **quadruple** donc le coût de l'attention. C'est le prix à payer pour que chaque token voie tous les autres, et la raison d'être de la recherche sur les contextes longs (fenêtres glissantes, noyaux économes en mémoire). En échange, la *profondeur* du calcul est constante en $T$ : tous les tokens sont traités en parallèle, contrairement aux $T$ étapes séquentielles d'un réseau récurrent.

## Formes

| Tenseur | Séquence unique | Avec batch $B$ |
|---|---|---|
| entrée $X$ | $(T,d_{\text{model}})$ | $(B,T,d_{\text{model}})$ |
| $W_Q,W_K$ | $(d_{\text{model}},d_k)$ | identique (partagé par le batch) |
| $W_V$ | $(d_{\text{model}},d_v)$ | identique |
| $Q,K$ | $(T,d_k)$ | $(B,T,d_k)$ |
| $V$ | $(T,d_v)$ | $(B,T,d_v)$ |
| scores $S=QK^\top/\sqrt{d_k}$ | $(T,T)$ | $(B,T,T)$ |
| poids $A$ | $(T,T)$, les lignes somment à 1 | $(B,T,T)$ |
| sortie $Z=AV$ | $(T,d_v)$ | $(B,T,d_v)$ |

En NumPy, la transposée par batch des deux derniers axes s'écrit `K.transpose(0, 2, 1)` (ou `np.swapaxes(K, -1, -2)`), et `@` diffuse (broadcast) sur l'axe du batch.

## Exemple détaillé (vérifié en exécutant le code)

Trois tokens, $d_{\text{model}}=d_k=d_v=2$ :

$$
X=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad
W_Q=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad
W_K=\begin{pmatrix}1&0\\1&1\end{pmatrix},\quad
W_V=\begin{pmatrix}1&0\\0&2\end{pmatrix}.
$$

Projections (chacune $3\times2$) :

$$
Q=\begin{pmatrix}1&1\\0&1\\1&2\end{pmatrix},\quad
K=\begin{pmatrix}1&0\\1&1\\2&1\end{pmatrix},\quad
V=\begin{pmatrix}1&0\\0&2\\1&2\end{pmatrix}.
$$

Scores bruts $QK^\top$ (forme $3\times3$) et scores normalisés avec $\sqrt{d_k}=\sqrt2\approx1.4142$ :

$$
QK^\top=\begin{pmatrix}1&2&3\\0&1&1\\1&3&4\end{pmatrix},\qquad
S=\begin{pmatrix}0.7071&1.4142&2.1213\\0&0.7071&0.7071\\0.7071&2.1213&2.8284\end{pmatrix}.
$$

Le softmax ligne par ligne donne les poids d'attention (chaque ligne somme à 1) :

$$
A=\begin{pmatrix}0.1400&0.2840&0.5760\\0.1978&0.4011&0.4011\\0.0743&0.3057&0.6200\end{pmatrix}.
$$

Vérifie la ligne 1 à la main : $e^{0.7071}=2.028$, $e^{1.4142}=4.113$, $e^{2.1213}=8.341$, somme $=14.482$, donc les poids sont $0.140,\,0.284,\,0.576$. La sortie $Z=AV$ (forme $3\times2$) :

$$
Z=\begin{pmatrix}0.7160&1.7199\\0.5989&1.6044\\0.6943&1.8514\end{pmatrix}.
$$

Par exemple, $z_1=0.14\,(1,0)+0.284\,(0,2)+0.576\,(1,2)=(0.716,\,1.720)$. Le token 3 place 62 % de son attention sur lui-même (plus fort alignement de clé) ; le token 1 n'en place que 14 % sur lui-même et écoute surtout le token 3. L'exercice reproduit ces nombres.

**Simulation de la variance.** En tirant 20 000 paires de vecteurs gaussiens standard, la variance du score brut vaut $4.0,\ 64.5,\ 508.2$ pour $d_k=4,64,512$ (théorie : $d_k$), et après division par $\sqrt{d_k}$ elle vaut $1.00,\ 1.01,\ 0.99$. L'effet sur le softmax avec 8 clés aléatoires : le plus grand poids d'attention vaut $0.63,\ 1.00,\ 1.00$ sans normalisation contre $0.35,\ 0.48,\ 0.50$ avec (pour $d_k=4,64,512$, un seul tirage aléatoire chacun ; c'est la tendance, et non les chiffres exacts, qui compte).

## Implémentation

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))   # subtract max: stability
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """Q,K: (..., T, d_k)   V: (..., T, d_v)   mask: (T, T) bool, True = forbidden."""
    d_k = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)      # (..., T, T)
    if mask is not None:
        S = np.where(mask, -np.inf, S)                  # forbidden pairs get weight 0
    A = softmax(S, axis=-1)                             # (..., T, T)
    return A @ V, A                                     # (..., T, d_v), weights
```

La soustraction du maximum ne change pas le résultat (le softmax est invariant par ajout d'une constante à une ligne) mais évite le débordement de `exp`.

## Idées reçues courantes

- **« Les poids d'attention sont ce à quoi le modèle "prête attention", point final. »** Ils ne sont qu'une quantité intermédiaire ; la sortie dépend aussi de $V$ et de tout ce qui suit. Considère les cartes d'attention comme un outil de diagnostic, pas comme une preuve d'explication.
- **« Les requêtes, clés et valeurs sont des données différentes. »** En auto-attention, les trois sont des projections linéaires du *même* $X$. (En attention croisée, leçon 4, les requêtes viennent d'une séquence et les clés et valeurs d'une autre.)
- **« Le softmax porte sur les colonnes / sur le batch. »** Il porte sur l'axe des clés (le dernier axe de la matrice $T\times T$) afin que les poids de chaque requête somment à 1.
- **« Le facteur d'échelle est cosmétique. »** C'est lui qui garde les gradients en vie à l'initialisation quand $d_k$ vaut 64 ou 128.
- **« L'attention est en $O(T)$ parce qu'elle est parallèle. »** Parallèle en profondeur, mais en $O(T^2)$ en travail et en mémoire.

## Points clés à retenir

- $\operatorname{Attn}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ : noter chaque paire requête-clé, normaliser par requête, moyenner les valeurs.
- Formes : $Q,K:(T,d_k)$, $V:(T,d_v)$, scores et poids $(T,T)$, sortie $(T,d_v)$.
- Les produits scalaires bruts ont une variance $d_k$ pour des entrées de variance unité ; diviser par $\sqrt{d_k}$ rétablit une variance de 1 et empêche un softmax saturé aux gradients évanescents.
- Le coût est de $O(T^2d_k)$ en temps et $O(T^2)$ en mémoire, mais avec une profondeur séquentielle constante.

## Explique-le toi-même

1. Sans formules, explique à un ami les rôles de la requête, de la clé et de la valeur, en utilisant l'analogie de la bibliothèque mais avec un autre exemple de ton choix.
2. Un collègue retire le facteur $1/\sqrt{d_k}$ et constate que la perte d'entraînement reste bloquée pendant les premières centaines de pas avec $d_k=128$. À l'aide de l'argument de la variance et de la jacobienne du softmax, explique une cause plausible.

## Pour aller plus loin

- Raschka, *Build a Large Language Model from Scratch*, ch. 3 (Sec. 3.3 à 3.4) : l'exemple à six tokens et les projections entraînables.
- Vaswani et al., 2017, *Attention Is All You Need*, Sec. 3.2.1 (cité dans les références de la leçon de la phase 07).
- La leçon 02 de la bibliothèque source sur l'auto-attention (section Build It) pour une implémentation alternative à base de classes.

## Critères de maîtrise

Tu sais, sur une feuille blanche, écrire la formule de l'attention, démontrer la variance d'un score brut, donner toutes les formes pour un appel par batch, reproduire à trois décimales la matrice de poids de l'exemple détaillé, et implémenter `attention` de façon à ce que les tests de l'exercice passent.
