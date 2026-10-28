---
id: trf.02
title: "Attention multi-têtes et masquage"
summary: "Pourquoi un seul motif d'attention ne suffit pas, comment les têtes sont des sous-espaces obtenus par remodelage, les formes exactes de découpage et de fusion, la projection de sortie, les masques causaux et de padding, et le nombre de paramètres, avec une implémentation NumPy testée contre les fuites du futur."
minutes: 80
teaches: [trf.multihead, trf.mask]
requires: [trf.attention, la.matrix]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/03-multi-head-attention/docs/en.md (split/combine heads, W_o, MQA/GQA/MLA variants, typical d_head). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md (the causal mask as lower-triangular row-stochastic matrix, -inf before softmax, parallel training). READ."
  - "LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb (Sec. 3.5 causal attention: mask before vs after softmax; Sec. 3.6 MultiHeadAttention with view/transpose and out_proj). READ."
---

## Objectifs d'apprentissage

- Expliquer pourquoi plusieurs petites têtes d'attention peuvent faire mieux qu'une seule grande tête pour le même coût.
- Écrire les remodelages de découpage et de fusion avec la forme exacte après chaque opération.
- Construire un masque causal et un masque de padding, et dire pourquoi ils s'ajoutent avant le softmax sous forme de $-\infty$.
- Compter les paramètres d'une couche d'attention multi-têtes.
- Implémenter l'attention multi-têtes en NumPy et tester que les tokens futurs ne peuvent pas modifier les sorties antérieures.

## Où cela s'inscrit

La leçon 1 produisait une moyenne pondérée par token : un seul motif d'attention $T\times T$. Mais une phrase contient plusieurs types de relations à la fois : « quel est le sujet de ce verbe », « à quel nom ce pronom renvoie-t-il », « quel était le mot précédent ». Une seule distribution softmax doit étaler tout cela ensemble. **L'attention multi-têtes** exécute plusieurs attentions en parallèle, chacune dans son propre espace plus petit, et laisse une matrice finale mélanger les résultats.

La seconde moitié de la leçon porte sur **ce qu'un token a le droit de voir**. Un modèle de langage est entraîné à prédire le token suivant ; pendant l'entraînement, le token $t$ ne doit donc pas pouvoir regarder le token $t+1$ (ce serait lire la réponse). Et les séquences d'un batch ont des longueurs différentes, donc les tokens de padding doivent être ignorés. Les deux cas se traitent avec la même astuce : les masques.

C'est exactement la couche d'attention à l'intérieur des modèles de type GPT (décodeur seul, leçon 4) et, sans le masque causal, à l'intérieur des modèles de type BERT.

## L'idée en mots simples

**Les têtes.** Imagine une salle de rédaction couvrant un événement. Un reporter suit qui a fait quoi, un autre suit les dates, un troisième les citations. Chacun travaille à partir de la même matière brute, mais avec un centre d'intérêt différent, et chacun écrit une courte note. Un rédacteur en chef (la projection de sortie) lit toutes les notes et rédige l'article final. Huit reporters spécialisés capturent généralement plus qu'un seul reporter disposant de huit fois plus de place dans son carnet, car un seul reporter ne peut suivre qu'une intrigue par paragraphe.

Concrètement, le vecteur de caractéristiques du modèle (disons 768 nombres) est **découpé en $h$ tranches égales** ; chaque tête ne regarde que les projections de sa propre tranche, calcule ses propres poids d'attention et renvoie un court vecteur. Les courts vecteurs sont recollés pour retrouver 768 nombres, puis passés dans une matrice de mélange apprise. Le coût total est le même que pour une grande tête, car les tranches sont plus petites.

**Les masques.** Un masque est un pochoir : une feuille de papier percée de trous. La matrice de scores a une entrée pour chaque paire (requête, clé) ; le pochoir bloque les paires que le modèle ne doit pas utiliser. Pour un modèle de langage, les paires interdites sont les paires « futures » (position de la clé postérieure à celle de la requête), qui forment le triangle supérieur de la matrice. Pour le padding, les paires interdites sont toutes celles dont la clé est un token factice.

## Définitions

Soit $d=d_{\text{model}}$, $h$ le nombre de têtes, et $d_h=d/h$ (on exige $h\mid d$). Les matrices apprises sont $W_Q,W_K,W_V,W_O\in\mathbb{R}^{d\times d}$. Pour une entrée $X\in\mathbb{R}^{B\times T\times d}$ :

1. Projeter : $Q=XW_Q$, $K=XW_K$, $V=XW_V$, chacun $(B,T,d)$.
2. **Découper** chacun en $h$ têtes : $(B,T,d)\to(B,T,h,d_h)\to(B,h,T,d_h)$.
3. Attention par tête : $\text{scores}=QK^\top/\sqrt{d_h}$ dans $(B,h,T,T)$, ajouter le masque, softmax, multiplier par $V$ : $(B,h,T,d_h)$.
4. **Fusionner** : $(B,h,T,d_h)\to(B,T,h,d_h)\to(B,T,d)$.
5. Projection de sortie : $\text{out}=\text{merged}\,W_O$, $(B,T,d)$.

Remarque : la normalisation utilise $\sqrt{d_h}$, pas $\sqrt{d}$ : les produits scalaires portent désormais sur $d_h$ termes, donc d'après l'argument de la variance de la leçon 1, c'est le bon facteur.

### Pourquoi on parle de « têtes comme sous-espaces »

Écris $W_Q=[W_Q^{(1)}\,|\,\dots\,|\,W_Q^{(h)}]$ avec chaque bloc de taille $d\times d_h$ (colonnes regroupées). La requête de la tête $i$ est alors $XW_Q^{(i)}$ : une projection apprise de $X$ sur un sous-espace de dimension $d_h$, et de même pour les clés et les valeurs. Le remodelage de l'étape 2 n'est que ce regroupement de colonnes réalisé par une grosse multiplication matricielle. Le score de la tête $i$ est $\sum_{m\in\text{slice }i}q_mk_m$, un produit scalaire partiel sur ses propres coordonnées, qui peut exprimer une notion de similarité différente de celle des autres têtes.

La fusion suivie de $W_O$ équivaut à sommer les contributions de chaque tête : découpe $W_O$ en blocs de lignes $W_O^{(i)}\in\mathbb{R}^{d_h\times d}$, alors $\text{out}=\sum_i Z^{(i)}W_O^{(i)}$. Les têtes n'interagissent jamais *à l'intérieur* de l'attention ; elles n'interagissent qu'à travers $W_O$ (et les couches suivantes).

## Démonstration : les masques

**Masque causal.** On veut que la sortie $i$ ne dépende que des entrées $j\le i$. La sortie $i$ vaut $\sum_ja_{ij}v_j$, il suffit donc que $a_{ij}=0$ pour $j>i$. Le softmax ne renvoie jamais un zéro exact pour des scores finis, mais $e^{-\infty}=0$. On ajoute donc une matrice $M$ aux scores avant le softmax :

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}
\qquad
a_{ij}=\frac{e^{S_{ij}+M_{ij}}}{\sum_{j'}e^{S_{ij'}+M_{ij'}}}.
$$

Le numérateur vaut $0$ pour les clés interdites, et le dénominateur ne somme que sur les clés autorisées, donc chaque ligne somme toujours à 1. Forme de $M$ : $(T,T)$, diffusée sur $(B,h)$.

**Pourquoi avant le softmax et non après.** Si tu mets les poids à zéro *après* le softmax, la ligne ne somme plus à 1 et il faut renormaliser ; masquer les scores avant le softmax donne le même résultat final en une seule étape et est numériquement plus propre (le chapitre 3 du livre montre les deux et note qu'ils coïncident après renormalisation).

**Masque de padding.** Les séquences d'un batch sont complétées jusqu'à une longueur commune $T$. Soit $p\in\{0,1\}^{B\times T}$ marquant les clés de padding. Le masque de padding des clés vaut $M^{\text{pad}}_{b,i,j}=-\infty$ si $p_{b,j}=1$, c'est-à-dire de forme $(B,1,1,T)$ diffusée sur les têtes et les requêtes. Les requêtes de padding produisent quand même des sorties (ignorées), donc la fonction de perte doit aussi sauter ces positions. Les deux masques se combinent en prenant le OU logique des indicateurs « interdit » (ou en additionnant les matrices de $-\infty$).

**Un coin dangereux.** Si *toutes* les clés d'une ligne sont masquées, la ligne vaut $\operatorname{softmax}(-\infty,\dots,-\infty)=0/0=\mathrm{NaN}$. Avec un masque causal seul, cela ne peut pas arriver (la diagonale est autorisée) ; avec un padding à gauche, cela peut arriver pour les requêtes de padding. Les implémentations gardent soit la diagonale autorisée, soit remplacent les lignes NaN par des zéros.

## Formes

| Étape | Forme | Remarque |
|---|---|---|
| $X$ | $(B,T,d)$ | |
| $Q,K,V$ après projection | $(B,T,d)$ | $W_{Q,K,V}:(d,d)$ |
| après `reshape` | $(B,T,h,d_h)$ | $d_h$ caractéristiques consécutives forment une tête |
| après `transpose(0,2,1,3)` | $(B,h,T,d_h)$ | les têtes deviennent un axe de type batch |
| $K^\top$ (deux derniers axes échangés) | $(B,h,d_h,T)$ | |
| scores / poids | $(B,h,T,T)$ | un masque $(T,T)$ ou $(B,1,1,T)$ se diffuse |
| poids @ V | $(B,h,T,d_h)$ | |
| `transpose(0,2,1,3)` | $(B,T,h,d_h)$ | |
| `reshape` de fusion | $(B,T,d)$ | inverse du découpage |
| après $W_O$ | $(B,T,d)$ | $W_O:(d,d)$ |

Le seul point subtil est l'ordre **remodeler d'abord, transposer ensuite**. Remodeler directement `(B,T,d)` en `(B,h,T,d_h)` mélangerait les tokens dans les têtes. Petite vérification tirée du code : pour une séquence avec $T=3$, $d=4$, $h=2$ et $X=\begin{pmatrix}0&1&2&3\\4&5&6&7\\8&9&10&11\end{pmatrix}$, la tête 0 reçoit les colonnes $\{0,1\}$ : les lignes $(0,1),(4,5),(8,9)$ et la tête 1 reçoit les colonnes $\{2,3\}$ : les lignes $(2,3),(6,7),(10,11)$. La fusion redonne exactement $X$.

## Nombre de paramètres

Chacune des quatre matrices contient $d^2$ poids :

$$
N_{\text{attn}}=4d^2\quad(\text{+ }4d\text{ if the projections have biases}).
$$

Pour $d=768$, $h=12$ ($d_h=64$) : $4\cdot768^2=2\,359\,296$ paramètres (2 362 368 avec biais). **Ce nombre ne dépend pas de $h$** : le découpage en têtes réutilise les mêmes matrices $d\times d$. Les matrices d'attention ne sont *pas* des paramètres, ce sont des activations calculées : $B\cdot h\cdot T^2$ nombres, ce qui explique la croissance de la mémoire avec la longueur du contexte.

Variantes (tirées de la leçon source) : l'attention multi-requêtes (multi-query) partage une seule tête K/V entre toutes les têtes de requête, et l'attention à requêtes groupées (grouped-query) utilise $G<h$ têtes K/V. Les deux réduisent $W_K,W_V$ à $d\times (G\,d_h)$ et, surtout, réduisent le cache clés/valeurs utilisé lors de la génération de texte, au prix d'une qualité légèrement inférieure. Dans cette leçon, nous implémentons l'attention multi-têtes classique.

## Exemple détaillé (vérifié en exécutant le code)

Prends l'exemple à trois tokens de la leçon 1 (mêmes $Q,K,V$, une seule tête) et applique le masque causal. Les scores normalisés étaient

$$
S=\begin{pmatrix}0.7071&1.4142&2.1213\\0&0.7071&0.7071\\0.7071&2.1213&2.8284\end{pmatrix}.
$$

Mettre les entrées au-dessus de la diagonale à $-\infty$ donne les poids

$$
A_{\text{causal}}=\begin{pmatrix}1&0&0\\0.3302&0.6698&0\\0.0743&0.3057&0.6200\end{pmatrix}.
$$

La ligne 1 ne peut voir qu'elle-même : poids 1, donc sa sortie est exactement $v_1=(1,0)$. La ligne 2 a pour scores $(0,0.7071)$ : $e^0=1$, $e^{0.7071}=2.028$, donc les poids sont $1/3.028=0.3302$ et $0.6698$. La ligne 3 est inchangée par rapport au cas non masqué : c'est le dernier token, donc les clés qu'il a le droit de voir (les trois) sont les mêmes qu'à la leçon 1. Les sorties sont

$$
Z_{\text{causal}}=\begin{pmatrix}1&0\\0.3302&1.3395\\0.6943&1.8514\end{pmatrix}.
$$

La ligne 3 est égale à la ligne 3 de la sortie non masquée de la leçon 1, ce qui est exactement ce qu'on veut : *le dernier token n'a pas de futur, donc le masquage ne le modifie pas*.

**Padding.** Suppose qu'une quatrième clé soit du padding avec un score de $5.0$ pour la première requête. La masquer donne les poids $(0.1400,\,0.2840,\,0.5760,\,0)$, identiques à ceux de la séquence sans padding. Sans le masque, la clé de padding absorberait l'essentiel de la masse de probabilité car son score est le plus grand. (Appliquer le softmax à une ligne de trois valeurs $-\infty$ renvoie bien NaN, comme annoncé plus haut.)

## Implémentation

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def split_heads(x, h):                      # (B, T, d) -> (B, h, T, d_h)
    B, T, d = x.shape
    return x.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)

def merge_heads(x):                         # (B, h, T, d_h) -> (B, T, d)
    B, h, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, h * dh)

def mha(X, Wq, Wk, Wv, Wo, h, causal=True, pad=None):
    """X: (B,T,d). pad: (B,T) bool, True = padding key. Returns (out, weights)."""
    B, T, d = X.shape
    Q, K, V = (split_heads(X @ W, h) for W in (Wq, Wk, Wv))   # (B,h,T,d_h)
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d // h)          # (B,h,T,T)
    forbid = np.zeros((B, 1, T, T), dtype=bool)
    if causal:
        forbid |= np.triu(np.ones((T, T), dtype=bool), k=1)
    if pad is not None:
        forbid |= pad[:, None, None, :]
    A = softmax(np.where(forbid, -np.inf, S))                  # (B,h,T,T)
    return merge_heads(A @ V) @ Wo, A                           # (B,T,d)
```

## Idées reçues courantes

- **« Plus de têtes signifie plus de paramètres. »** Le nombre est $4d^2$ pour tout $h$ qui divise $d$. Plus de têtes signifie des têtes plus petites.
- **« Les têtes apprennent automatiquement des choses différentes. »** Elles sont libres de le faire, et les études par sondage montrent des têtes spécialisées, mais rien n'impose la diversité ; beaucoup de têtes peuvent être élaguées sans grande perte.
- **« Le masque causal s'applique à la sortie. »** Il s'applique aux scores avant le softmax ; l'appliquer à la sortie n'empêcherait pas l'information de circuler à travers $V$.
- **« Remodeler `(B,T,d)` directement en `(B,h,T,d_h)`. »** Cela brouille silencieusement tokens et caractéristiques. Remodèle en `(B,T,h,d_h)`, puis transpose.
- **« Le masquage rend le modèle plus rapide. »** La matrice complète $T\times T$ est quand même calculée dans l'implémentation simple ; les noyaux efficaces sautent la moitié masquée.

## Points clés à retenir

- Les têtes sont $h$ attentions parallèles dans des sous-espaces de dimension $d_h=d/h$ ; on découpe par reshape+transpose, on fusionne par l'inverse, puis on mélange avec $W_O$.
- Les scores utilisent $\sqrt{d_h}$ ; les formes sont $(B,h,T,T)$ pour les poids et $(B,T,d)$ pour la sortie de la couche.
- Masque causal : $-\infty$ au-dessus de la diagonale, ajouté avant le softmax ; masque de padding : $-\infty$ sur les clés de padding.
- Paramètres : $4d^2$ (indépendant de $h$) ; les matrices d'attention sont des activations coûtant $B\,h\,T^2$.

## Explique-le toi-même

1. En partant de `(B, T, d)`, écris la suite des remodelages et transpositions d'un appel d'attention multi-têtes, avec la forme après chacun, et dis ce qui se passerait si tu échangeais le premier reshape et la transposition.
2. Pourquoi la sortie du token $i$ ne change-t-elle pas quand tu modifies le token $i+1$ sous un masque causal ? Suis le raisonnement à travers le softmax et la somme pondérée.

## Pour aller plus loin

- Raschka, *Build a Large Language Model from Scratch*, ch. 3, Sec. 3.5-3.6 (attention causale, `MultiHeadAttention` avec `view`/`transpose`).
- Bibliothèque source, phase 07, leçon 03 (variantes de têtes MQA, GQA, MLA) et leçon 07 (le masque triangulaire comme moyenne de préfixe raffinée).
- Vaswani et al., 2017, Sec. 3.2.2 (cité par la leçon source).

## Critères de maîtrise

Tu sais écrire de mémoire les fonctions de découpage et de fusion avec leurs formes, construire un masque causal et un masque de padding, calculer $4d^2$ pour un $d$ donné, et ton implémentation passe un test montrant que modifier les tokens après la position $t$ laisse inchangées les sorties aux positions $\le t$.
