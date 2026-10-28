---
id: adv.02
title: "Modèles vision-langage à résolution arbitraire et vidéo"
summary: "Empaqueter des images de forme quelconque dans une seule séquence, donner aux patchs des positions 2D et 3D, et calculer le coût en tokens et en mémoire KV de plusieurs heures de vidéo."
minutes: 65
teaches: [adv.dynamic-res, adv.video-tokens]
requires: [adv.vlm, trf.mask, trf.rope, adv.visual-tokens]
sources:
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/06-any-resolution-patch-n-pack/docs/en.md: NaViT patch-n-pack, block-diagonal mask, AnyRes, M-RoPE, token budgets, min/max pixels. READ."
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps/docs/en.md: native dynamic resolution (multiples of 28, 960 tokens for 1120x672), M-RoPE, dynamic FPS, absolute time. READ (first part)."
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/17-video-language-temporal-grounding/docs/en.md: video token reduction strategies, per-frame pooling, TMRoPE, frame sampling. READ (first part)."
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/18-long-video-million-token/docs/en.md: token counts of long video, brute context vs ring attention vs compression vs agentic retrieval. READ (first part)."
---

## Où cela s'inscrit

[[adv.01]] donnait à chaque image une taille fixe et donc un nombre de tokens fixe. Les vraies entrées ne sont pas carrées : un ticket de caisse est haut, un graphique est large, une capture de téléphone est en 9:19,5, un scan peut faire 4096 pixels de large, et une vidéo est des centaines d'images à la suite. Écraser tout dans un carré de 336x336 déforme le texte et gaspille des tokens, et une vidéo naïve « chaque image en pleine résolution » fait exploser n'importe quelle fenêtre de contexte. Cette leçon couvre les trois façons dont les modèles évitent ces problèmes (découpage en tuiles, empaquetage, résolution native avec positions rotatives généralisées) et construit le calculateur que tu devrais lancer avant de t'engager dans une conception : combien de tokens, et combien de gigaoctets de cache KV.

## L'idée en mots simples

Imagine un labo photo qui imprime les photos de tout le monde sur du papier carré identique. Un long panorama est écrasé, un ticket de caisse haut est complété par du papier blanc, et une carte immense est réduite jusqu'à ce que les étiquettes disparaissent. Il y a trois remèdes.

1. **Découper la grande image en tirages de taille standard** et imprimer aussi un petit aperçu d'ensemble (découpage AnyRes).
2. **Imprimer chaque photo à sa taille naturelle**, et empiler les tirages en une seule pile, en veillant à ce qu'un lecteur qui lit une photo ne regarde jamais sa voisine (patch-n-pack avec un masque en blocs).
3. **Décrire où se trouve chaque patch par des coordonnées** (ligne, colonne, et pour la vidéo, temps) plutôt que par un emplacement numéroté dans une table fixe, de sorte que toute taille soit légale (M-RoPE).

Pour la vidéo, ajoute une idée de plus : tu n'as pas besoin de trente photographies par seconde d'une personne assise immobile. Échantillonne moins d'images, réduis chacune en moyennant les patchs voisins, et étiquette chaque image avec son temps réel.

## Résolution native et empaquetage

**Redimensionner en carré, rogner, compléter** échouent chacun : le redimensionnement déforme le texte, le rognage jette du contenu, le complément gaspille plus de la moitié des tokens sur une page haute et l'attention paie les tokens de remplissage de façon quadratique.

**Patch-n-pack (NaViT).** Pour chaque image, calculer sa propre grille de patchs ; aplatir chacune en une séquence de longueur $n_b$ ; concaténer les $B$ séquences en une seule de longueur $S=\sum_b n_b$ sans remplissage. Pour garder les images séparées, utiliser un masque d'attention **bloc-diagonal** $M\in\{0,1\}^{S\times S}$, avec $M_{ij}=1$ si et seulement si les positions $i$ et $j$ se trouvent dans la même image (les masques ont été introduits dans [[trf.02]]). Entrées de scores calculées : $\sum_b n_b^2$, au lieu de $B\cdot(\max_b n_b)^2$ avec remplissage. Pour quatre images de 256, 576, 729 et 1024 tokens : $S=2585$, entrées empaquetées $1{,}977{,}329$, entrées avec remplissage $4\cdot1024^2=4{,}194{,}304$, et une séquence indivise noterait $2585^2=6{,}682{,}225$ paires. (Le chemin à longueur variable de FlashAttention prend des longueurs cumulées au lieu d'un masque dense et saute les zéros.)

**Résolution dynamique native (style Qwen2-VL).** Le transformer de vision accepte toute hauteur et largeur divisible par $u=p\cdot m$ (patch $p=14$, une fusion spatiale $m\times m=2\times2$ de patchs voisins en un token, donc $u=28$) :

$$
N=\frac{H'}{u}\cdot\frac{W'}{u},\qquad H',W' \text{ the sides rounded to multiples of } u .
$$

Une image 1120x672 donne $40\times24=960$ tokens. Un plafond (le `max_pixels` de Qwen ; l'exercice utilise `max_tokens`) te protège de la photo 4000x3000, et un plancher (`min_pixels`) de l'icône 5x5.

**Position sans table : M-RoPE.** Les embeddings rotatifs ([[trf.03]]) font tourner les coordonnées de requête et de clé d'un angle proportionnel à la position, de sorte que les scores dépendent du décalage relatif. Le RoPE multimodal donne à chaque token un **triplet** $(t, r, c)$ et affecte des tranches différentes des dimensions tournées au temps, à la ligne et à la colonne. Pour une image, $t=0$. Il n'y a pas de table apprise dont on puisse dépasser la fin, donc toute taille fonctionne sans redimensionnement. `mrope_positions(2,2,3)` produit douze triplets allant de $(0,0,0)$ à $(1,1,2)$.

**Le découpage AnyRes (LLaVA-NeXT)** est l'alternative quand l'encodeur est figé à une seule résolution (336) : choisir une disposition en grille (1x2, 2x2, 3x1...) qui s'adapte au rapport d'aspect, découper des tuiles, ajouter une miniature réduite pour le contexte global. Le coût est multiplicatif : $(r c+1)\times 576$ tokens. Une grille 2x2 donne 2880 ; une grille 4x4 sur une image 1344x1344 donne 9792.

| Stratégie | Tokens pour une grande image | Faut-il un encodeur entraîné pour cela ? |
|---|---|---|
| Redimensionnement carré 336 | 576 (détail perdu) | non |
| Tuiles AnyRes + miniature | $(rc+1)\cdot576$ | non |
| Natif + M-RoPE | environ $HW/u^2$ jusqu'au plafond | oui |

Règle empirique de la source : choisir un plafond de tokens par tâche (OCR et documents 1000-4000, graphiques et interfaces 700-1000, photos naturelles 250-600) et encoder au rapport d'aspect natif jusqu'à ce plafond.

## Vidéo : tokens et temps

Une vidéo de $D$ secondes échantillonnée à $f$ images par seconde avec $\tau$ tokens par image a

$$
N_{\text{video}} = D\,f\,\tau
$$

tokens. Trois leviers la réduisent : **moins d'images** (un $f$ plus bas, ou un échantillonnage dynamique qui dépense des images là où il y a du mouvement), **moins de tokens par image** (moyenner par pooling la grille de tokens d'un facteur $s$ par côté : $\tau\to\lceil g_h/s\rceil\lceil g_w/s\rceil$, donc une grille $24\times24=576$ devient $8\times8=64$ pour $s=3$), et **un compresseur appris** (un Q-Former par clip, ou un token de résumé par clip).

**Positions temporelles.** Dire au modèle que l'image 5 précède l'image 6 exige une position temporelle. Les indices d'image appris sont les plus simples. Avec un FPS dynamique, l'indice est trompeur (les images sont espacées de façon inégale), donc Qwen2.5-VL lie la position rotative temporelle à des **horodatages absolus** ; le modèle peut alors répondre à « à quelle seconde le chat saute-t-il ? » plutôt qu'à « au début du clip ».

**Mémoire.** Le cache KV stocke les clés et valeurs de chaque token dans chaque couche :

$$
\text{KV bytes}=2\cdot L\cdot h_{kv}\cdot d_h\cdot N\cdot b ,
$$

avec $L$ couches, $h_{kv}$ têtes clé-valeur, dimension de tête $d_h$, $N$ tokens et $b$ octets par valeur (2 pour BF16). Il croît **linéairement** en $N$ ; le calcul d'attention du prefill croît **quadratiquement** : l'attention causale note $N(N+1)/2$ paires. (Le cache est introduit dans [[sys.01]] ; [[adv.04]] montre comment le réduire.)

## Exemple détaillé

LLM : 32 couches, 8 têtes KV, $d_h=128$, BF16 (un modèle de la forme de Llama-3-8B, utilisé comme choix illustratif).

| Entrée | Tokens | Cache KV |
|---|---|---|
| 5 min à 1 fps, 576 tokens/image | $300\cdot576=172{,}800$ | 21.1 GiB |
| idem, pooling 3x3 (64 tokens/image) | $300\cdot64=19{,}200$ | 2.3 GiB |
| 30 min à 1 fps, 81 tokens/image | $1800\cdot81=145{,}800$ | 17.80 GiB |
| 2 h à 1 fps, 81 tokens/image | $583{,}200$ | 71.2 GiB |

Par token, le cache coûte $2\cdot32\cdot8\cdot128\cdot2=131{,}072$ octets = 128 KiB. La vidéo de 2 heures ne laisse donc presque rien d'une carte de 80 Go pour les poids, c'est pourquoi les systèmes de vidéo longue utilisent l'attention en anneau (ring attention) entre appareils, une compression agressive des tokens, ou une **récupération agentique** (un LLM interroge la vidéo comme une base de données et n'encode que les clips pertinents). Le contexte brut (style Gemini 1.5, 1M+ tokens) cache l'ingénierie mais pas l'arithmétique.

Exemple du ticket de caisse : un ticket de 600x1500 avec un patch de 14 sans fusion donne $43\times107=4601$ tokens en résolution native contre 576 après un redimensionnement carré à 336. L'écart de huit fois est l'endroit où se loge la précision de l'OCR.

## Implémentation

```python
import numpy as np

def packed_mask(lengths):
    S = sum(lengths); M = np.zeros((S, S), bool); o = 0
    for l in lengths:
        M[o:o+l, o:o+l] = True; o += l
    return M

def kv_gib(layers, kv_heads, head_dim, n_tokens, b=2):
    return 2 * layers * kv_heads * head_dim * n_tokens * b / 2**30
```

Code de production : les processeurs de Hugging Face exposent `min_pixels` et `max_pixels` pour Qwen-VL, et l'interface varlen de FlashAttention (`cu_seqlens`) implémente l'empaquetage sans masque dense.

## Formes

| Tenseur | Forme |
|---|---|
| séquence de patchs empaquetée | (S, d), $S=\sum_b n_b$ |
| masque en blocs | (S, S) booléen |
| positions M-RoPE | (N, 3) |
| tokens vidéo | (images x tokens_par_image, d_llm) |
| cache KV (par séquence) | (L, 2, h_kv, N, d_h) |

## Idées reçues

- **« La résolution native supprime le problème des tokens. »** Elle rend les tokens proportionnels aux pixels ; un grand scan coûte encore des milliers de tokens sans plafond.
- **« L'empaquetage change les mathématiques de l'attention. »** À l'intérieur de chaque bloc elles sont identiques ; le masque ne fait qu'interdire les scores entre images.
- **« Plus de FPS signifie une meilleure compréhension vidéo. »** Au-delà d'un certain point, on ajoute des tokens redondants ; l'échantillonnage dynamique ou piloté par les événements et les horodatages comptent souvent davantage.
- **« Le pooling est gratuit. »** Il perd le détail spatial fin (lire de petits textes) ; il convient mieux aux questions d'action qu'à l'OCR.
- **« La mémoire KV est quadratique en la longueur. »** Le cache est linéaire ; le calcul du prefill est quadratique.

## Points clés à retenir

- Les tokens valent environ les pixels divisés par $u^2$ en résolution native ; AnyRes multiplie par le nombre de tuiles plus une miniature.
- L'empaquetage utilise un masque bloc-diagonal pour que les images ne se mélangent pas et que le remplissage ne soit jamais payé.
- M-RoPE donne aux patchs des coordonnées (temps, ligne, colonne), donc les tailles n'ont pas besoin de table.
- Le coût d'une vidéo est $Df\tau$ tokens et $2Lh_{kv}d_hNb$ octets ; utilise l'échantillonnage, le pooling, la compression ou la récupération pour rester dans le budget.

## Explique-le toi-même

1. Pourquoi un masque bloc-diagonal rend-il un batch empaqueté équivalent à des images traitées séparément ?
2. Un utilisateur veut analyser une vidéo de cours de 90 minutes. Cite trois stratégies, leurs coûts et l'information que chacune perd.

## Pour aller plus loin

- NaViT (Dehghani et al., 2023), rapports techniques de Qwen2-VL et Qwen2.5-VL, LLaVA-NeXT, tels que résumés dans les leçons sources.
- [[adv.01]] pour les connecteurs ; [[sys.01]] pour les caches KV ; [[adv.04]] pour les moyens de les réduire.

## Critères de maîtrise

Tu sais calculer les nombres de tokens pour des entrées en résolution native, AnyRes et vidéo, construire un masque d'empaquetage, dire ce qu'ajoute M-RoPE, et estimer les GiB de cache KV pour une forme de modèle et une longueur d'entrée données.
