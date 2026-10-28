---
id: seq.02
title: Plongements de mots
summary: "Des vecteurs one-hot aux vecteurs denses appris à partir du contexte, avec la perte skip-gram à échantillonnage négatif et ses gradients dérivés, plus les idées de GloVe et fastText, la similarité cosinus et les analogies."
minutes: 90
teaches: [seq.embedding]
requires: [nn.backprop, la.dot, ml.logreg, info.entropy]
sources:
  - "math/cours-tal-01.pdf (CNAM RCP217, PDF p.32-47: one-hot representation, <UNKNOWN>/<PADDING> markers, distributional semantics, word embeddings, word2vec CBOW and skip-gram, cosine, geometric properties, GloVe and its analogy argument). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md (distributional hypothesis, skip-gram pairs, two tables W and W', negative-sampling update, nearest neighbours and analogy). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md (GloVe objective with weighting, fastText character n-grams, BPE overview). READ."
---

## Objectifs d'apprentissage

- Expliquer pourquoi les vecteurs one-hot ne peuvent pas exprimer la similarité et ce qu'apporte un plongement (embedding) dense.
- Énoncer l'hypothèse distributionnelle et comment skip-gram la transforme en tâche de prédiction.
- Dériver la perte à échantillonnage négatif et ses gradients pour les deux tables de plongements.
- Décrire l'objectif de GloVe et l'idée des sous-mots de fastText, et quand chacun aide.
- Utiliser la similarité cosinus, les plus proches voisins et les analogies vectorielles, et connaître leurs limites.
- Entraîner un minuscule skip-gram en NumPy et lire le résultat.

## Prérequis

La régression logistique et la sigmoïde ([[f2n.08]]), la rétropropagation avec les dimensions ([[f2n.10]]).

## Où cela s'inscrit

Chaque modèle de langue, traducteur, moteur de recherche et chatbot doit nourrir une machine numérique avec des mots. La première décision est *comment représenter un mot par des nombres*. Tout ce qui suit dans ce cours (les réseaux récurrents dans [[seq.03]], les cellules à portes dans [[seq.04]], le séquence-à-séquence et l'attention dans [[seq.05]], et plus tard les transformers) prend en entrée une séquence de **vecteurs**. Les plongements sont ces vecteurs.

Ils sont aussi le premier exemple d'**apprentissage auto-supervisé** : il n'y a pas d'annotateur, le texte lui-même fournit les cibles. (Les notes de cours du CNAM remarquent que l'on appelle souvent word2vec « non supervisé » par abus de langage ; la tâche d'entraînement est un problème de prédiction supervisé dont les étiquettes viennent gratuitement.)

## L'idée en mots simples

Supposons que le vocabulaire compte 50 000 mots. Le code le plus direct donne à chaque mot son propre interrupteur parmi 50 000 : un seul interrupteur est allumé. Ce code **one-hot** est aveugle : « chat » et « chien » sont exactement aussi différents que « chat » et « hypothèque ». Deux vecteurs one-hot distincts sont toujours à distance $\sqrt2$, et leur produit scalaire vaut 0. Un modèle entraîné sur des phrases sur les chats n'apprend rien qu'il puisse réutiliser pour les chiens.

Que voudrait-on à la place ? Une *carte des mots* où les mots semblables sont voisins. Le linguiste Firth a mis la recette en une ligne : « on reconnaît un mot à la compagnie qu'il fréquente ». Regarde les mots qui tendent à apparaître autour de « chat » : *ronronne, poils, nourrit, chasse*. « Chien » apparaît dans presque la même compagnie. Alors on place les mots sur la carte de sorte que **les mots à compagnie semblable obtiennent des positions voisines**.

Comment trouver de telles positions ? On en fait un jeu. Prends un mot, cache ses voisins, et demande à un petit modèle de deviner quels mots apparaissent près de lui. Pour gagner, le modèle doit donner à « chat » et « chien » des positions à partir desquelles on peut faire des devinettes semblables. Les positions qui gagnent le jeu sont les plongements. Jette le modèle, garde la carte.

Deux astuces pratiques gardent le jeu bon marché. Deviner « lequel des 50 000 mots est le voisin ? » est coûteux, donc on pose plutôt des questions oui/non : « *ronronne* est-il un vrai voisin de *chat* ? et *hypothèque* ? » (le second est un mot faux tiré au hasard, un **échantillon négatif**). C'est l'**échantillonnage négatif**.

## Cadre formel

Un corpus est une suite de tokens. Fixe une taille de fenêtre $m$. Pour chaque position, prends le mot **central** $c$ et chaque mot $o$ à moins de $m$ positions comme mot de **contexte** ; chaque $(c,o)$ de ce type est une paire d'entraînement positive. Pour « the cat sat on the mat » avec $m=1$, les paires pour le centre « sat » sont (sat, cat) et (sat, on).

Skip-gram (tel que présenté dans les diapositives du CNAM et la leçon source) prédit les mots de contexte à partir du mot central ; CBOW prédit le centre à partir de la somme des mots de contexte. Les diapositives notent que CBOW fonctionne bien pour les mots fréquents et skip-gram mieux pour les mots rares.

Chaque mot $w$ reçoit **deux** vecteurs dans $\mathbb{R}^d$ : $v_w$ quand c'est un mot central (lignes de $V\in\mathbb{R}^{n\times d}$) et $u_w$ quand c'est un mot de contexte (lignes de $U\in\mathbb{R}^{n\times d}$). En termes de réseau, $V$ est la première couche (one-hot fois $V$ sélectionne simplement une ligne) et $U$ la couche de sortie. Après l'entraînement, le plongement final est $v_w$, ou $v_w+u_w$.

### Le modèle softmax complet et pourquoi il est coûteux

Modélise $p(o\mid c)=\dfrac{e^{u_o\cdot v_c}}{\sum_{w=1}^{n}e^{u_w\cdot v_c}}$. La perte pour une paire est $-\log p(o\mid c)$, et son gradient par rapport à $v_c$ est $-u_o+\sum_w p(w\mid c)\,u_w$ (le gradient du softmax de [[f2n.08]]). La somme parcourt tout le vocabulaire : un travail en $O(nd)$ par paire d'entraînement, avec $n$ de 50 000 ou plus.

## Dérivation : l'échantillonnage négatif

Remplace la classification à $n$ classes par une classification binaire. Pour une vraie paire $(c,o)$ on veut une forte probabilité de « réel », pour $k$ mots aléatoires $w_1,\dots,w_k$ on veut une forte probabilité de « faux » :

$$
P(\text{real}\mid c,w)=\sigma(u_w\cdot v_c),\qquad \sigma(s)=\frac1{1+e^{-s}}.
$$

La log-vraisemblance négative des étiquettes observées (c'est exactement la perte de la régression logistique) est

$$
\ell=-\log\sigma(u_o\cdot v_c)-\sum_{j=1}^{k}\log\sigma(-u_{w_j}\cdot v_c),
$$

en utilisant $1-\sigma(s)=\sigma(-s)$. Les négatifs $w_j$ sont tirés d'une distribution de bruit ; la leçon source les tire au hasard dans le vocabulaire, et le choix courant est la fréquence unigramme élevée à la puissance $3/4$, qui échantillonne les mots rares plus souvent que leur fréquence brute.

Écris les scores $s_o=u_o\cdot v_c$ et $s_j=u_{w_j}\cdot v_c$. Il nous faut la dérivée des deux sortes de termes. Pour le terme positif : puisque $\frac{d}{ds}\log\sigma(s)=1-\sigma(s)$,

$$
\frac{\partial}{\partial s_o}\big[-\log\sigma(s_o)\big]=-(1-\sigma(s_o))=\sigma(s_o)-1 .
$$

Pour un terme négatif : $\frac{d}{ds}\log\sigma(-s)=-(1-\sigma(-s))=-\sigma(s)$, donc

$$
\frac{\partial}{\partial s_j}\big[-\log\sigma(-s_j)\big]=\sigma(s_j).
$$

Définis $g_o=\sigma(s_o)-1$ (dans $(-1,0)$) et $g_j=\sigma(s_j)$ (dans $(0,1)$). Les scores sont des produits scalaires, donc par la règle de la chaîne :

$$
\frac{\partial\ell}{\partial v_c}=g_o\,u_o+\sum_{j=1}^{k}g_j\,u_{w_j},\qquad
\frac{\partial\ell}{\partial u_o}=g_o\,v_c,\qquad
\frac{\partial\ell}{\partial u_{w_j}}=g_j\,v_c .
$$

**Lire le gradient.** La descente de gradient soustrait ces quantités. Pour $u_o$, le pas est $-\eta g_ov_c$ avec $g_o<0$ : $u_o$ se déplace **vers** $v_c$. Chaque négatif s'**éloigne** de $v_c$. Et $v_c$ lui-même se déplace vers le vrai vecteur de contexte et s'éloigne des vecteurs de bruit. L'ampleur de chaque poussée est l'erreur actuelle du modèle : un négatif qui a déjà un score très bas a $g_j\approx0$ et est laissé tranquille.

**Coût.** Seules $1+k$ lignes de $U$ et une ligne de $V$ changent par paire : $O((k+1)d)$ au lieu de $O(nd)$. Si un indice négatif se répète, ses contributions au gradient s'additionnent (utilise `np.add.at`).

**Dimensions.** $V,U:(n,d)$ ; $v_c,u_o:(d)$ ; matrice des négatifs $U[\text{negs}]:(k,d)$ ; scores $(k)$ ; $\partial\ell/\partial v_c=g_ou_o+g^\top U[\text{negs}]:(d)$ ; le gradient des négatifs est le produit extérieur $g\,v_c^\top:(k,d)$.

## Exemple détaillé

Prenons $d=2$, le vecteur central $v_c=(0.5,\,1.0)$, le vrai contexte $u_o=(1.0,\,0.5)$ et deux négatifs $u_1=(-0.5,\,0.5)$, $u_2=(1.0,\,-1.0)$ (les nombres ci-dessous viennent du code de l'exercice).

- Scores : $s_o=0.5+0.5=1.0$, $s_1=-0.25+0.5=0.25$, $s_2=0.5-1.0=-0.5$.
- Perte : $-\log\sigma(1)=0.3133$ ; $-\log\sigma(-0.25)=0.8259$ ; $-\log\sigma(0.5)=0.4741$. Total $\ell=1.6133$.
- Erreurs : $g_o=\sigma(1)-1=-0.2689$, $g_1=\sigma(0.25)=0.5622$, $g_2=\sigma(-0.5)=0.3775$.
- $\partial\ell/\partial v_c=-0.2689(1,0.5)+0.5622(-0.5,0.5)+0.3775(1,-1)=(-0.1725,\,-0.2309)$.
- $\partial\ell/\partial u_o=-0.2689\,(0.5,1.0)=(-0.1345,-0.2689)$ ; $\partial\ell/\partial u_1=0.5622\,(0.5,1.0)=(0.2811,0.5622)$ ; $\partial\ell/\partial u_2=(0.1888,0.3775)$.

Le premier négatif a obtenu $0.25$, positif, c'est donc lui qui est le plus repoussé. Une valeur de contrôle : avec tous les vecteurs nuls, chaque score vaut 0 et $\ell=(1+k)\ln2$.

## Implémentation

```python
import numpy as np

def sigmoid(s):
    return 0.5 * (1.0 + np.tanh(0.5 * s))

def sgns_loss_grads(V, U, c, o, negs):
    negs = np.asarray(negs)
    v = V[c]
    s_pos, s_neg = U[o] @ v, U[negs] @ v
    loss = float(np.logaddexp(0, -s_pos) + np.logaddexp(0, s_neg).sum())
    g_pos, g_neg = sigmoid(s_pos) - 1.0, sigmoid(s_neg)
    dV = np.zeros_like(V); dU = np.zeros_like(U)
    dV[c] = g_pos * U[o] + g_neg @ U[negs]
    dU[o] += g_pos * v
    np.add.at(dU, negs, np.outer(g_neg, v))
    return loss, dV, dU
```

`np.logaddexp(0, -s)` est un $-\log\sigma(s)$ numériquement stable. La boucle d'entraînement : construis les paires à partir du corpus, mélange à chaque époque, tire $k$ négatifs, mets à jour $V[c]$, $U[o]$ et $U[\text{negs}]$ par SGD. Sur un corpus jouet de 13 phrases avec $d=8$, fenêtre 2, $k=5$, taux d'apprentissage $0.05$ et 40 époques, la perte moyenne tombe de $4.13$ à environ $1.96$, et dans l'espace appris « king » et « queen » (qui ne diffèrent jamais par leurs contextes) ont un cosinus $\approx0.99$ alors que « cat » et « king » sont alignés négativement. Le corpus minuscule en fait un jouet, pas une preuve de quoi que ce soit sur la langue réelle.

## Similarité et analogies

La distance euclidienne dépend trop de la fréquence des mots ; les diapositives du CNAM préfèrent l'angle entre vecteurs, mesuré par

$$
\cos(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}\in[-1,1].
$$

Il vaut 1 pour la même direction, 0 pour l'orthogonalité (comme les vecteurs one-hot le sont toujours), et est invariant par changement d'échelle.

Empiriquement, sur de grands corpus, les décalages entre vecteurs capturent des relations : « woman + (king - man) est proche de queen ». Les diapositives et la leçon source présentent cela comme une propriété observée, pas une garantie. Pour répondre à une analogie $a:b::c:?$, calcule $b-a+c$ et renvoie le mot le plus proche par cosinus **en excluant les trois mots de la requête** ; sans l'exclusion, la réponse est généralement l'une des entrées.

## GloVe : utiliser directement les comptages globaux

Word2vec apprend en ligne à partir de fenêtres. **GloVe** construit la matrice de co-occurrence $X$ ($X_{ij}$ = combien de fois $j$ apparaît dans le contexte de $i$) et ajuste des vecteurs dessus. L'argument du CNAM part des rapports de probabilités. Pour $i=$ ice, $j=$ steam, et un mot sonde $k$ : $P(k\mid\text{ice})/P(k\mid\text{steam})$ est grand pour $k=$ solid (8.9 dans le tableau de la diapositive), petit pour $k=$ gas (0.085), et proche de 1 pour les mots liés aux deux (water, 1.36) ou à aucun (fashion, 0.96). Les rapports isolent l'axe significatif (solide à gaz). Pour que les rapports deviennent des différences, modélise le **logarithme** :

$$
\log P(k\mid i)\approx w_i^\top\tilde w_k\quad\Rightarrow\quad \log\frac{P(k\mid i)}{P(k\mid j)}\approx(w_i-w_j)^\top\tilde w_k .
$$

Si $A:B$ et $C:D$ expriment la même relation, cela doit valoir pour tous les mots sondes, donc $w_A-w_B\approx w_C-w_D$. C'est la raison géométrique pour laquelle les décalages encodent des analogies. L'objectif implémenté (leçon source) est un ajustement par moindres carrés pondérés :

$$
J=\sum_{i,j}f(X_{ij})\big(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij}\big)^2,\qquad f(x)=\min\!\big((x/x_{max})^{\alpha},1\big).
$$

Le poids $f$ empêche les paires très fréquentes comme (the, and) de dominer ; la source utilise $\alpha=0.75$. Le vecteur final est $w+\tilde w$. La diapositive du CNAM dit que sa dérivation est simplifiée (le modèle de l'article est plus complexe).

## fastText : plonger les morceaux

Les tables au niveau du mot n'ont pas de vecteur pour un mot jamais vu (« zoomer-approved ») et traitent « run », « runs », « running » comme sans lien. **fastText** représente un mot comme la somme des plongements de ses n-grammes de caractères plus le mot lui-même. Avec des marqueurs de frontière, « where » avec $n=3$ donne `<wh`, `whe`, `her`, `ere`, `re>`. Un mot jamais vu comme « whereupon » partage plusieurs n-grammes avec « where » et obtient donc un vecteur proche. Le prix est une table plus grande (beaucoup de n-grammes) et des vecteurs un peu plus flous pour les mots courts irréguliers. Les LLM modernes vont un pas plus loin avec des vocabulaires de sous-mots appris (BPE) ; c'est le sujet des leçons sur les tokenizers.

## Idées reçues

- **« Le plongement est une couche cachée qui a besoin d'une non-linéarité. »** Il n'y en a pas : multiplier un vecteur one-hot par $V$ sélectionne simplement une ligne. Le gain vient de la tâche d'entraînement, pas de la profondeur.
- **« Des vecteurs semblables signifient des synonymes. »** Le cosinus mesure la similarité des *contextes*. « Hot » et « cold » apparaissent dans les mêmes contextes et finissent proches ; les antonymes ne sont pas séparés.
- **« Les plongements sont des vérités universelles. »** Ils reflètent le corpus, y compris ses biais et son mélange de sens : un vecteur par type de mot brouille « bank » la rive et « bank » l'institution. (Les plongements contextuels, plus loin dans le cours, corrigent cela.)
- **« L'arithmétique des analogies est exacte. »** C'est une régularité bruitée sur de grands corpus ; elle échoue souvent et exige la règle d'exclusion.
- **« Les échantillons négatifs sont garantis faux. »** Un mot aléatoire peut occasionnellement être un vrai voisin. Le bruit est toléré parce qu'il est rare.

## Limites et compromis

- Un vecteur par type de mot : pas de polysémie, pas de gestion des mots jamais vus (fastText ou les sous-mots atténuent ce dernier point).
- La fenêtre définit la « compagnie » ; de petites fenêtres donnent une similarité syntaxique, de grandes fenêtres une similarité thématique.
- Les plongements statiques ne portent aucune information au niveau de la phrase : l'ordre des mots est perdu, c'est pourquoi les couches récurrentes viennent ensuite.
- La qualité demande beaucoup de texte : sur un corpus jouet, la géométrie n'est qu'évocatrice.

## Points clés

- Les vecteurs one-hot sont orthogonaux (distance $\sqrt2$) : aucune similarité. Les plongements denses de dimension 50-300 placent les mots à contextes semblables à proximité.
- Skip-gram à échantillonnage négatif minimise $-\log\sigma(u_o\cdot v_c)-\sum_j\log\sigma(-u_j\cdot v_c)$.
- Gradients : $g_o=\sigma(s_o)-1$, $g_j=\sigma(s_j)$ ; $\partial_{v_c}=g_ou_o+\sum g_ju_j$, $\partial_{u_o}=g_ov_c$, $\partial_{u_j}=g_jv_c$.
- GloVe ajuste $w_i^\top\tilde w_k+b_i+\tilde b_k$ à $\log X_{ik}$ avec une pondération ; fastText somme des vecteurs de n-grammes de caractères.
- Compare avec le cosinus ; pour les analogies, exclus les mots de la requête.

## Explique-le toi-même

1. Pourquoi les négatifs de score très bas ne contribuent-ils presque pas au gradient, et pourquoi est-ce souhaitable ?
2. En partant de $\log P(k\mid i)\approx w_i^\top\tilde w_k$, explique pourquoi $w_{\text{king}}-w_{\text{man}}\approx w_{\text{queen}}-w_{\text{woman}}$ est plausible.

## Pour aller plus loin

- Bibliothèque source : `ai-engineering-from-scratch` phase 05, leçons 03 (word2vec) et 04 (GloVe, fastText, BPE).
- Diapositives CNAM RCP217 `cours-tal-01.pdf`, p. 32-47 du PDF.
- Ensuite : [[seq.03]], transformer une séquence de plongements en une prédiction grâce à la récurrence.

## Critères de maîtrise

Tu sais dériver à partir de zéro les trois gradients de la perte à échantillonnage négatif, ton implémentation NumPy passe une vérification par différences finies (y compris avec des négatifs répétés), et tu sais expliquer en quelques phrases ce que la géométrie apprise signifie et ne signifie pas.
