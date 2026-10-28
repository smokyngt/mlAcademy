---
id: sys.01
title: "Cache KV et génération autorégressive efficace"
summary: "Pourquoi la génération naïve token par token refait sans cesse le même travail, comment la mise en cache des clés et des valeurs l'élimine, ce que le cache coûte en mémoire, et comment l'attention à requêtes groupées le réduit."
minutes: 70
teaches: [llm.kv-cache]
requires: [la.matrix, nn.layer, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/12-kv-cache-flash-attention/docs/en.md: problem statement, per-token cache formula, FlashAttention and continuous batching overview. READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/12-inference-optimization/docs/en.md (Problem, Prefill vs Decode): compute-bound prefill, memory-bound decode, ops:byte ratio. READ (first part)."
  - "LLMs-from-scratch/ch04/03_kv-cache/README.md: the 'Time flies' picture of reusable key/value vectors. READ."
  - "LLMs-from-scratch/ch04/04_gqa/README.md: grouped-query attention, heads sharing K/V. READ."
  - "LLMs-from-scratch/pkg/llms_from_scratch/kv_cache/generate.py: structure of generation with and without a cache (prime the cache with the prompt, then feed one new token per step). READ."
---

## Où cela s'inscrit

Quand tu discutes avec un modèle de langage, les mots apparaissent les uns après les autres. Ce n'est pas un artifice d'interface : le modèle produit réellement un token par passe avant, et chaque token dépend de tous les précédents. Une réponse de 500 tokens exige donc 500 passes séquentielles. Si chaque passe relisait et retraitait toute la conversation depuis le début, les longues conversations deviendraient inutilisables de lenteur et de coût. Tous les systèmes de serving sérieux (vLLM, llama.cpp, TensorRT-LLM) évitent cela grâce à un **cache KV**, et la mémoire que ce cache requiert est, en pratique, ce qui limite le nombre d'utilisateurs qu'un GPU peut servir et la longueur du contexte.

Avant cette leçon, tu as construit l'attention (requêtes, clés, valeurs, softmax) et un bloc décodeur. Après, tu pourras raisonner sur le coût de la génération en FLOPs et en gigaoctets, ce qui est le ticket d'entrée pour la leçon suivante (la quantification et le décodage spéculatif sont deux autres façons d'attaquer le même goulot d'étranglement).

## L'idée en mots simples

Imagine que tu écrives une histoire avec un ami qui doit relire toute l'histoire depuis la première page chaque fois que tu ajoutes un mot, avant de pouvoir te suggérer le mot suivant. Les notes de l'ami sur la première page ne changent jamais, mais il les réécrit à chaque fois. Un ami plus malin tient un carnet : pour chaque mot déjà écrit, une courte note (« ce que ce mot offre aux mots suivants » et « ce que ce mot cherche »). Quand un nouveau mot arrive, il écrit une seule nouvelle note puis consulte tout le carnet.

Dans l'attention, chaque token produit trois vecteurs : une **requête** (ce que je cherche), une **clé** (ce que j'offre pour la mise en correspondance) et une **valeur** (ce que je transmets si je suis retenu). Point crucial : dans un décodeur avec masque causal, la clé et la valeur du token $i$ ne dépendent que des tokens jusqu'à $i$. Une fois calculées, elles ne changent plus quand de nouveaux tokens sont ajoutés. On peut donc les stocker, et chaque nouveau pas n'a qu'à calculer les trois vecteurs du nouveau token et à comparer sa requête à toutes les clés stockées.

Deux phases en découlent :

- **Prefill** : le prompt est connu à l'avance, donc tous ses tokens sont traités en parallèle dans une grosse multiplication matricielle (rapide par token, limitée par le calcul) et leurs clés et valeurs remplissent le cache.
- **Decode** : ensuite, un token par pas. Chaque pas fait très peu d'arithmétique mais doit faire transiter les poids du modèle et tout le cache depuis la mémoire ; il est donc limité par la bande passante mémoire plutôt que par le calcul.

Le prix du carnet est la mémoire, qui croît à chaque token.

## Cadre formel et coût de la méthode naïve

Une seule tête, largeur du modèle $d$ (notée $d_\text{model}$), largeur de tête $d_h$. Soit $X \in \mathbb{R}^{t\times d}$ contenant les $t$ embeddings de tokens jusqu'ici et $W_q, W_k, W_v \in \mathbb{R}^{d\times d_h}$.

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v,\qquad
O = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}} + M\Big)V
$$

où $M$ est le masque causal ($-\infty$ au-dessus de la diagonale). Seule la dernière ligne de $O$ est nécessaire pour choisir le token suivant.

**Convention de FLOPs.** Le produit d'une matrice $(a,b)$ par une matrice $(b,c)$ coûte $2abc$ FLOPs (une multiplication et une addition par terme). Nous ignorons le softmax, la layer norm, le MLP et la projection de sortie ; ils sont plus petits ou identiques pour les deux méthodes.

## Dérivation : compter le travail

**Pas naïf avec $t$ tokens présents.**

1. Les trois projections des $t$ lignes : $3\times 2\,t\,d\,d_h = 6\,t\,d\,d_h$.
2. Scores $QK^\top$ : $2\,t^2 d_h$ si on calcule le carré complet (un noyau causal n'a besoin que du triangle inférieur, soit $t(t+1)/2$ entrées, ce qui donne $t(t+1)d_h$).
3. Somme pondérée $PV$ : de nouveau $t(t+1)d_h$ avec le triangle.

Total avec le comptage triangulaire :

$$
F_\text{naive}(t) = 6\,t\,d\,d_h + 2\,t(t+1)\,d_h .
$$

Remarque que c'est $O(t^2)$ par pas, donc $O(T^3)$ pour générer une séquence de longueur $T$ si on somme sur les pas (les projections donnent $O(T^2)$, l'attention $O(T^3)$).

**Pas avec cache, $t$ tokens présents.**

1. Projections d'**un seul** token : $6\,d\,d_h$.
2. Une requête contre $t$ clés : $2\,t\,d_h$ ; poids fois $t$ valeurs : $2\,t\,d_h$.

$$
F_\text{cached}(t) = 6\,d\,d_h + 4\,t\,d_h .
$$

C'est $O(t)$ par pas : linéaire au lieu de quadratique. Le rapport $F_\text{naive}/F_\text{cached}$ vaut environ $t$ pour $t$ grand, ce qui correspond exactement à « nous avons cessé de refaire $t-1$ tokens ».

Le cache ne change pas le *résultat* : les entrées de $K$ et $V$ pour les tokens précédents sont, bit pour bit, les mêmes nombres dans les deux cas. C'est pourquoi le test d'un cache correct est l'égalité exacte des sorties.

## Formule de la mémoire

Par couche, par token, par tête KV, le cache stocke un vecteur clé et un vecteur valeur de longueur $d_h$ :

$$
\text{bytes} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot T \cdot b \cdot B
$$

où $L$ est le nombre de couches, $n_{kv}$ le nombre de têtes clé/valeur, $T$ la longueur de séquence, $b$ le nombre d'octets par nombre (2 pour fp16/bf16, 1 pour int8) et $B$ la taille de batch. Le facteur 2 correspond à « K et V ».

## Exemple détaillé

Tous les nombres ci-dessous sont reproduits par le code de l'exercice.

**Un modèle de classe 7B** ($L=32$, $n_{kv}=32$, $d_h=128$, fp16, $T=4096$, $B=1$) :

$2\cdot 32\cdot 32\cdot 128\cdot 4096\cdot 2 = 2{,}147{,}483{,}648$ octets $= 2$ GiB pour **une** conversation. Seize utilisateurs simultanés exigent 32 GiB de cache seul, avant de compter les 14 Go de poids.

**Attention à requêtes groupées (GQA).** En attention multi-têtes, chaque tête de requête a ses propres K et V. La GQA conserve le nombre complet de têtes de requête mais en fait partager un même couple K/V à $g$ d'entre elles, donc $n_{kv} = n_\text{heads}/g$. Avec $n_{kv}=8$, le même modèle n'a besoin que de $0.5$ GiB, soit quatre fois moins. L'attention multi-requête est le cas extrême $n_{kv}=1$. Le matériel du livre indique que les modèles GQA se comportent de façon comparable à l'attention multi-têtes complète dans les ablations publiées ; c'est donc surtout un gain de mémoire gratuit, bien qu'il doive être décidé à l'entraînement.

Un cas plus grand : $L=80$, $n_{kv}=8$, $d_h=128$, $T=32768$, fp16 donne $10.0$ GiB par séquence ; avec 64 têtes K/V, ce serait $80$ GiB.

**FLOPs pour un modèle jouet** ($d=16$, $d_h=8$, prompt de 5 tokens, 20 nouveaux tokens) : la méthode naïve utilise $305{,}280$ FLOPs au total, la méthode avec cache $24{,}640$ : un gain de $12.4\times$, et les sorties concordent à $2\times10^{-16}$ près. L'écart se creuse avec la longueur. Pour une tête réaliste ($d=4096$, $d_h=128$) à un seul pas :

| tokens présents $t$ | FLOPs naïfs | FLOPs avec cache | rapport |
|---|---|---|---|
| 5 | 15,736,320 | 3,148,288 | 5.0 |
| 100 | 317,158,400 | 3,196,928 | 99.2 |
| 1000 | 3,401,984,000 | 3,657,728 | 930.1 |

## Implémentation

```python
def generate_cached(X0, steps, W):
    d_head = W["Wq"].shape[1]
    K, V = X0 @ W["Wk"], X0 @ W["Wv"]          # prefill: all prompt tokens at once
    out = softmax((X0[-1] @ W["Wq"]) @ K.T / np.sqrt(d_head)) @ V
    rows = list(X0)
    for _ in range(steps):                      # decode: one token per step
        nxt = np.tanh(out @ W["Wo"])
        rows.append(nxt)
        K = np.vstack([K, nxt @ W["Wk"]])       # append, never recompute
        V = np.vstack([V, nxt @ W["Wv"]])
        out = softmax((nxt @ W["Wq"]) @ K.T / np.sqrt(d_head)) @ V
    return np.array(rows)
```

Dans les implémentations réelles, le cache est un tenseur préalloué et les tokens y sont écrits sur place ; `np.vstack` copie et ne sert qu'à la clarté. Le fichier `kv_cache/generate.py` du livre a la même forme à deux branches : avec un cache, il fait d'abord passer le prompt dans le modèle avec le cache attaché, puis n'envoie que le token le plus récent à chaque pas.

## Formes

| Objet | Forme |
|---|---|
| embeddings du prompt $X_0$ | $(T_0, d)$ |
| $W_q, W_k, W_v$ | $(d, d_h)$ |
| cache $K$, $V$ après $t$ tokens | $(t, d_h)$ par tête, par couche |
| requête du nouveau token | $(d_h,)$ |
| scores | $(t,)$ (avec cache) au lieu de $(t,t)$ (naïf) |
| tenseur de cache complet (toutes couches, têtes, batch) | $(L, 2, B, n_{kv}, T, d_h)$ |

## Limites et compromis

- **La mémoire, pas les FLOPs, domine.** Le decode avec cache fait peu d'arithmétique par octet déplacé (faible rapport ops:octets), donc le GPU attend la mémoire. Réduire le cache (GQA, cache quantifié, fenêtres glissantes, allocation paginée) accélère le decode autant qu'il économise de l'espace. Les sources estiment que lire une fois tous les poids d'un modèle 70B en fp16 prend environ 70 ms sur un GPU à 2 To/s, ce qui constitue un plancher pour un pas de decode, quelles que soient les astuces de cache.
- **Pas pour l'entraînement.** À l'entraînement, toute la séquence est connue et traitée en parallèle, donc il n'y a rien à réutiliser d'un pas à l'autre.
- **Quantifier le cache** (int8/fp8) le divise de nouveau par deux, mais les erreurs sur les clés et les valeurs affectent tous les tokens suivants ; la perte de qualité croît donc avec la longueur du contexte.
- **Idées voisines, non dérivées ici.** FlashAttention découpe en tuiles le calcul d'attention pour que la matrice de scores $t\times t$ ne vive jamais en mémoire lente ; le continuous batching et la mémoire KV paginée améliorent l'utilisation. Ils changent la vitesse et la mémoire, jamais le résultat mathématique.
- Les chiffres exacts de modèles de production précis changent d'une version à l'autre ; utilise la formule avec la configuration propre du modèle.

## Idées reçues

1. « Le cache rend la génération approximative. » Non : les $K$, $V$ stockés sont exactement les valeurs qu'un recalcul complet donnerait. Seuls des changements comme la quantification du cache introduisent de l'erreur.
2. « Le prefill et le decode coûtent la même chose par token. » Le prefill est parallèle et limité par le calcul, le decode est séquentiel et limité par la bande passante ; le temps jusqu'au premier token et les tokens par seconde sont des métriques différentes.
3. « La GQA réduit le nombre de têtes de requête. » Elle réduit uniquement le nombre de têtes K/V ; les requêtes restent.
4. « Il faudrait aussi mettre les requêtes en cache. » Une requête n'est utilisée qu'une fois, à son propre pas.

## Points clés à retenir

- Sans cache, un pas de decode coûte $O(t^2)$ (somme sur une génération : $O(T^3)$) ; avec cache, $O(t)$ par pas.
- Taille du cache $=2\,L\,n_{kv}\,d_h\,T\,b\,B$ octets ; elle croît linéairement avec le contexte et la taille de batch.
- Le prefill remplit le cache en parallèle ; le decode ajoute une entrée par pas et est limité par la mémoire.
- La GQA divise le cache par la taille de groupe $g=n_\text{heads}/n_{kv}$.
- Un cache correct se vérifie par un accord exact avec le calcul sans cache.

## Explique-le toi-même

1. Pourquoi la clé et la valeur du token 3 peuvent-elles être stockées et réutilisées quand le token 10 est généré, alors que le modèle est relancé ? Quelle propriété du masque causal le garantit ?
2. Un ami dit « achète simplement plus de calcul, le cache n'est qu'une optimisation ». À l'aide de la formule de la mémoire, explique pourquoi un long contexte et de nombreux utilisateurs sont limités par la mémoire.

## Pour aller plus loin

- Raschka, *Build a Large Language Model From Scratch*, matériel bonus `ch04/03_kv-cache` et `ch04/04_gqa`.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023), l'article vers lequel pointe le README de la GQA.
- Dao et al., FlashAttention (2022), pour le versant trafic mémoire de l'attention.

## Critères de maîtrise

- [ ] Je sais dériver $F_\text{naive}(t)$ et $F_\text{cached}(t)$ et dire lequel est quadratique.
- [ ] Je sais calculer la taille du cache KV pour une configuration et voir ce que change la GQA.
- [ ] Je sais expliquer prefill contre decode et pourquoi le decode est limité par la mémoire.
- [ ] Je sais tester une implémentation de cache pour l'égalité exacte avec l'implémentation naïve.
