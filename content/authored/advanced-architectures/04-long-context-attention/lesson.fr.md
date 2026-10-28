---
id: adv.04
title: "Variantes d'attention pour le contexte long"
summary: "Masques à fenêtre glissante, creux et global-local, partage de clés-valeurs MQA/GQA/MLA et attention différentielle, avec du code qui construit les masques et compte les octets de KV de chaque variante."
minutes: 75
teaches: [adv.attn-variants, adv.kv-sharing]
requires: [trf.attention, trf.multihead, trf.mask]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/15-attention-variants/docs/en.md: sliding-window, sparse and differential attention, masks, comparison table, Gemma-style 5:1 layout. READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/16-differential-attention-v2/docs/en.md: softmax noise floor, differential formula, V1 vs V2. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/17-native-sparse-attention/docs/en.md: three NSA branches (compressed, selected, window), natively trainable. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/docs/en.md: MHA, MQA, GQA, MLA comparison. READ (knob 4)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/docs/en.md: MLA latent cache 7.6 GB vs 30.5 GB GQA at 128k. READ."
---

## Où cela s'inscrit

L'attention complète ([[trf.01]]) compare chaque token à chaque token précédent : un calcul en $O(N^2)$, et un cache clé-valeur (KV) ([[sys.01]]) qui croît linéairement avec le contexte dans chaque couche. À 128 000 tokens, les deux deviennent le goulot d'étranglement : le cache d'un modèle à 32 couches et 32 têtes fait à lui seul 64 GiB. Deux familles d'idées combattent cela. **Regarder moins** (fenêtres, motifs creux, résumés) réduit le calcul et parfois le cache. **Stocker moins par token** (partager ou compresser les clés et les valeurs) ne réduit que le cache. Une troisième idée, l'**attention différentielle**, améliore la qualité de ce sur quoi l'on porte l'attention. Les modèles modernes mélangent tout cela ; cette leçon te donne les masques sous forme de code et l'arithmétique du KV sous forme de calculateur.

## L'idée en mots simples

En lisant un long rapport, tu ne relis pas chaque page précédente pour chaque nouvelle phrase. Tu gardes en tête en détail les **dernières pages** (fenêtre glissante), tu gardes quelques **marque-pages** vers des endroits importants comme le titre et la table des matières (tokens globaux), et de temps en temps tu sautes à un **résumé de chapitre** (blocs compressés). L'ensemble est bien moins coûteux que tout relire, et chaque page que tu lis a elle-même été écrite en connaissance des pages précédentes, de sorte que l'information se diffuse tout de même progressivement.

Pour le stockage : si 32 relecteurs gardent chacun leur copie complète des notes (multi-têtes), la mémoire explose. Laisser des groupes de relecteurs partager une copie (requêtes groupées), laisser tout le monde en partager une seule (multi-requête), ou garder une sténographie compacte à partir de laquelle chaque relecteur peut reconstruire sa copie (attention latente) réduit ce qu'il faut transporter.

Enfin, l'attention softmax est comme une pièce avec un faible sifflement : même les tokens non pertinents reçoivent un peu de poids. Prends deux enregistrements bruités et soustrais l'un de l'autre : le sifflement commun s'annule tandis que la voix demeure : c'est l'attention différentielle.

## Masques : regarder moins

Les scores d'attention sont $S=QK^\top/\sqrt{d}\in\mathbb{R}^{N\times N}$ ; un masque $M$ met à $-\infty$ les entrées interdites avant le softmax ([[trf.02]]), de sorte que ces clés reçoivent exactement un poids nul. Le masque booléen $M_{ij}=1$ signifie que la requête $i$ peut voir la clé $j$.

| Variante | Ensemble autorisé pour la requête $i$ | Entrées (causal) | KV conservé par couche |
|---|---|---|---|
| Causal complet | $j\le i$ | $N(N+1)/2$ | $N$ |
| Fenêtre glissante $W$ | $i-W<j\le i$ | environ $NW$ | $W$ |
| Local + à pas | fenêtre, plus $j\le i$ avec $j \bmod s=0$ | environ $N(W+N/s)$ | tout, ou seulement ceux à pas |
| Global + local | fenêtre, plus quelques positions globales | environ $N(W+g)$ | $W+g$ |

Exemples vérifiés par l'exercice : pour $N=8$, $W=4$ : 26 des 36 entrées causales. Pour $N=8192$, $W=1024$ : 7 864 832 sur 33 558 528, un facteur 4.3. (La source cite « 8x » pour ce cas ; cela compare avec le carré complet $N^2$, pas avec la moitié causale que l'attention complète calcule réellement. Les deux points de vue sont instructifs, mais compare calcul contre calcul.)

**L'empilement restaure la portée.** Un token de la couche $\ell$ ne voit que les $W$ positions précédentes de la couche $\ell-1$, dont les valeurs résumaient elles-mêmes $W$ positions antérieures. Après $L$ couches, l'information peut voyager sur $L(W-1)+1$ positions : 131 041 pour $W=4096$, $L=32$. En pratique, le rappel à cette distance est faible, donc les modèles entrelacent **une couche d'attention complète toutes les quelques couches**. Un ratio de style Gemma de 5 couches à fenêtre glissante pour 1 globale est le choix par défaut classique.

Les **motifs creux** (sparse transformer, Longformer, BigBird) choisissent le motif à l'avance : fenêtre locale, plus des positions à pas ou globales, plus parfois des liens aléatoires. Le masque est facile ; le gain n'apparaît que lorsque les noyaux sautent les blocs nuls. L'**attention creuse native (DeepSeek NSA)** apprend au contraire le motif pendant le pré-entraînement avec trois branches parallèles : l'attention sur des résumés de blocs *compressés* (vue globale grossière), l'attention sur les *tokens fins des blocs les mieux notés* (les scores compressés choisissent quels blocs charger), et une fenêtre glissante (locale), combinées par une porte apprise. Comme la branche compressée est dérivable et pilote la sélection, le routage creux est entraînable de bout en bout, contrairement à un élagage à l'inférence appliqué à un modèle entraîné en dense.

## Mémoire KV : stocker moins par token

La taille du cache est

$$
\text{KV}=2\cdot L\cdot h_{kv}\cdot d_h\cdot N_{\text{kept}}\cdot b,
$$

avec $L$ couches, $h_{kv}$ têtes KV, dimension de tête $d_h$, tokens conservés $N_{\text{kept}}$ et $b$ octets par valeur (2 pour BF16).

- **MHA** : $h_{kv}=h$ (une paire KV par tête de requête).
- **MQA** (multi-requête) : $h_{kv}=1$ ; le cache rétrécit d'un facteur $h$, avec un petit coût en précision.
- **GQA** (requêtes groupées) : $G$ têtes KV, chacune partagée par $h/G$ têtes de requête. La tête de requête $j$ utilise la tête KV $\lfloor j/(h/G)\rfloor$. Style Llama-3-8B : $h=32$, $G=8$.
- **Fenêtre glissante** : $N_{\text{kept}}=\min(N,W)$.
- **MLA** (attention latente multi-têtes, DeepSeek) : compresser l'état caché $x_t\in\mathbb{R}^d$ en un latent $c_t=x_tW_{DKV}\in\mathbb{R}^{r}$ avec $r\ll h\,d_h$, et **ne mettre en cache que $c_t$**. Les clés et valeurs sont reconstruites à la volée : $k_t=c_tW_{UK}$, $v_t=c_tW_{UV}$, chacune produisant toutes les têtes. Comme $q_t k_s^\top=q_t W_{UK}^\top c_s^\top$, la projection ascendante $W_{UK}$ peut être repliée (absorbée) du côté des requêtes, de sorte que les clés complètes n'ont jamais besoin d'être matérialisées. Le cache par token et par couche vaut $r$ nombres, pas $2h_{kv}d_h$. DeepSeek-V3 a $r=512$ (le décompte de la source). Les vraies implémentations mettent aussi en cache une petite clé de position rotative séparée, environ 64 nombres supplémentaires d'après mon souvenir des articles DeepSeek-V2/V3, ce que je n'ai pas vérifié dans les sources ; l'exercice a un paramètre `rope_dim` pour que tu puisses l'inclure.

## Attention différentielle

Le softmax ne donne jamais exactement zéro, et le score d'une clé non pertinente est un petit nombre aléatoire autour de 0, donc avec $N$ clés la masse non pertinente s'additionne en $O(1)$ quelle que soit la longueur du contexte. L'attention différentielle sépare les requêtes et les clés de chaque tête en deux groupes et soustrait les deux cartes d'attention :

$$
\operatorname{DiffAttn}(X)=\Big(\operatorname{softmax}\!\big(\tfrac{Q_1K_1^\top}{\sqrt d}\big)-\lambda\,\operatorname{softmax}\!\big(\tfrac{Q_2K_2^\top}{\sqrt d}\big)\Big)V ,
$$

avec $\lambda$ un scalaire appris par tête. Les deux cartes partagent le plancher de bruit diffus, qui s'annule ; la carte qui apprend la vraie dépendance n'est pas annulée. Deux vérifications que tu peux faire : si les deux cartes sont identiques et $\lambda=1$, la sortie est exactement nulle, et si $V$ ne contient que des uns, la sortie vaut $1-\lambda$ puisque chaque ligne de softmax somme à un. Effets rapportés (de la source, non vérifiés par moi) : perplexité plus basse et récupération de type aiguille dans une botte de foin plus nette. DIFF V2 réorganise les têtes (plus de têtes de requête, mêmes têtes KV) de sorte que la vitesse de décodage et la compatibilité avec FlashAttention égalent celles de la référence.

## Exemple détaillé (tout est reproduit par l'exercice)

Modèle de la forme de Llama-3-8B : 32 couches, $d_h=128$, 128K tokens (131 072), BF16.

| Variante | Configuration | Cache KV |
|---|---|---|
| MHA | 32 têtes KV | 64 GiB |
| GQA | 8 têtes KV | 16 GiB |
| MQA | 1 tête KV | 2 GiB |
| GQA + fenêtre glissante 4096 sur chaque couche | 8 têtes KV, 4096 tokens | 0.5 GiB |
| Hybride 5:1 (30 couches : 5 globales, 25 à fenêtre 1024), GQA | 8 têtes KV | 2.60 GiB, contre 15 GiB tout-global |

Forme de DeepSeek-V3 (61 couches, 128K) : MLA avec $r=512$ stocke $61\cdot512\cdot131072\cdot2$ octets $=7.625$ GiB ; un GQA de style Llama-70B (8 têtes KV, $d_h=128$) stocke 30.5 GiB, quatre fois plus ; ajouter la clé rotative de 64 dimensions à MLA donne 8.58 GiB.

## Implémentation

```python
import numpy as np

def sliding_mask(n, window):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return (j <= i) & (i - j < window)

def gqa_attention(Q, K, V, mask):            # Q:(H,N,d)  K,V:(G,N,d)
    H, N, d = Q.shape; G = K.shape[0]
    out = np.zeros_like(Q)
    for h in range(H):
        g = h // (H // G)
        s = np.where(mask, Q[h] @ K[g].T / np.sqrt(d), -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        out[h] = (s / s.sum(-1, keepdims=True)) @ V[g]
    return out
```

Production : FlashAttention et FlexAttention de PyTorch prennent des fonctions de masque et sautent les blocs vides ; Hugging Face expose `num_key_value_heads` dans les configurations de modèles pour la GQA, et des tailles de fenêtre glissante par couche pour Gemma et Mistral.

## Formes

| Tenseur | Forme |
|---|---|
| requêtes | (H, N, d) |
| clés, valeurs (GQA) | (G, N, d), avec H divisible par G |
| masque | (N, N) booléen |
| cache MLA | (N, r) par couche (+ clé rope) |
| attention différentielle | deux cartes de scores (N, N), soustraites avant de multiplier par V |

## Idées reçues

- **« Fenêtre glissante = le modèle oublie tout ce qui est plus ancien que W. »** L'information se propage à travers la profondeur, et les couches globales la rafraîchissent ; le rappel reste toutefois plus faible.
- **« La GQA change le nombre de têtes de requête. »** Seulement le nombre de têtes clé/valeur distinctes.
- **« MLA est un repas gratuit. »** Il échange de la mémoire contre un peu de calcul supplémentaire (projection ascendante ou matrices absorbées) et une complexité de conception supplémentaire.
- **« Un masque économise automatiquement du calcul. »** Seulement si le noyau saute les blocs masqués ; un masque dense sur un noyau dense n'économise rien.
- **« L'attention différentielle soustrait la carte d'attention des valeurs. »** Elle soustrait deux *cartes* d'attention avant de multiplier par V.

## Points clés à retenir

- Les masques réduisent le calcul (fenêtre $W$ : environ $NW$ au lieu de $N^2/2$ paires) ; les couches ou tokens globaux restaurent l'accès à longue portée.
- Cache KV $=2Lh_{kv}d_hN_{\text{kept}}b$ : GQA/MQA réduisent $h_{kv}$, les fenêtres réduisent $N_{\text{kept}}$, MLA stocke un latent de taille $r$.
- L'attention différentielle annule le bruit de mode commun du softmax : $(A_1-\lambda A_2)V$.
- Compte toujours les octets pour ton propre modèle avant de choisir une variante.

## Explique-le toi-même

1. Dérive le cache KV de la GQA pour $h=32,G=8$ relativement à la MHA, et explique pourquoi la qualité des requêtes n'est que faiblement affectée.
2. Pourquoi la projection ascendante absorbée de MLA te permet-elle de ne jamais former les clés complètes par tête ?

## Pour aller plus loin

- Longformer, BigBird, Gemma 2/3, GQA (Ainslie et al.), DeepSeek-V2/V3, Differential Transformer, Native Sparse Attention, tels que résumés dans les leçons sources.
- [[adv.02]] pour les coûts de KV de la vidéo, [[adv.05]] pour remplacer le cache par un état, [[adv.03]] pour le MoE de DeepSeek-V3.

## Critères de maîtrise

Tu sais écrire des masques causal, glissant, à pas et global-local, implémenter la GQA et l'attention différentielle, et calculer la taille du cache KV pour des dispositions MHA, MQA, GQA, à fenêtre glissante, hybrides et MLA.
