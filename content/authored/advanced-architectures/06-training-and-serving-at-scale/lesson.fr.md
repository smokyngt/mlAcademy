---
id: adv.06
title: "Entraînement à grande échelle : parallélisme, ZeRO, précision mixte, checkpointing, MTP"
summary: "Répartir un modèle et son batch sur des GPU, compter les octets par GPU sous ZeRO/FSDP, utiliser fp16/bf16 avec mise à l'échelle de la perte, échanger du recalcul contre de la mémoire d'activations, et ajouter la prédiction multi-tokens."
minutes: 85
teaches: [adv.parallelism, adv.zero, adv.mixed-precision, adv.checkpointing, adv.mtp]
requires: [nn.training, nn.backprop, dl.adam, trf.block, info.entropy, adv.moe-balance]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/05-scaling-distributed/docs/en.md: memory table, data/tensor/pipeline/3D parallelism, bubble fraction, ZeRO stages, mixed precision and loss scaling. READ (most sections; code skimmed)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/19-dualpipe-parallelism/docs/en.md: pipeline bubbles, 1F1B, zero bubble, bidirectional DualPipe overlap with all-to-all. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/34-gradient-checkpointing/docs/en.md: what backward needs, full vs selective checkpointing, recompute cost model, offload. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/18-multi-token-prediction/docs/en.md: sequential MTP modules, joint loss, parallel vs sequential, parameter overhead, speculative-decoding reuse. READ (first part)."
---

## Où cela s'inscrit

Tout ce que nous avons vu jusqu'ici (contextes longs, MoE, nombreux experts) met à rude épreuve la mémoire et la communication. Un modèle de 7 milliards de paramètres a déjà besoin de plus d'un GPU pour **s'entraîner** (pas seulement pour s'exécuter), car l'optimiseur garde plusieurs copies de chaque paramètre. Cette leçon est la boîte à outils pour faire tenir et accélérer l'entraînement : trois façons de répartir le travail (parallélisme de données, de tenseurs, de pipeline), comment ZeRO/FSDP partitionne l'état redondant, pourquoi l'arithmétique en basse précision exige une mise à l'échelle de la perte, comment le gradient checkpointing échange du calcul contre de la mémoire, et comment la prédiction multi-tokens donne une supervision plus dense. Elle s'appuie sur l'optimiseur de la boucle d'entraînement et sur Adam, ainsi que sur le bloc de transformer ([[trf.04]]). La mémoire côté inférence (cache KV, quantification) est traitée dans [[sys.01]] et [[sys.02]].

## L'idée en mots simples

Imagine écrire une immense encyclopédie en équipe.

- **Parallélisme de données** : chaque rédacteur a une copie complète du brouillon, chacun édite de nouvelles pages différentes, et à la fin de chaque journée ils moyennent leurs modifications pour que toutes les copies restent identiques. Rapide, mais chacun a besoin d'une copie complète.
- **Parallélisme de tenseurs** : une très large table est découpée en bandes de colonnes, chaque rédacteur remplit sa bande, et les bandes sont recollées. Cela exige de discuter en permanence, donc les rédacteurs doivent siéger dans la même pièce (même nœud).
- **Parallélisme de pipeline** : une chaîne de montage : le rédacteur 1 traite les chapitres 1-8, le rédacteur 2 les chapitres 9-16, et ainsi de suite. Au début et à la fin de chaque journée, certains rédacteurs sont inactifs (la **bulle**), sauf si l'on alimente la chaîne avec de nombreux petits lots.
- **ZeRO / FSDP** : les rédacteurs travaillent toujours dans le style parallèle de données, mais la volumineuse comptabilité (les notes de l'optimiseur) est découpée pour que chaque rédacteur n'en stocke qu'une tranche, et les tranches sont empruntées au besoin.
- **Basse précision** : écrire les nombres avec moins de chiffres est plus rapide et plus compact, mais les très petits nombres disparaissent ; multiplie tout par une grande constante avant d'écrire, et divise ensuite.
- **Checkpointing** : au lieu de garder chaque note intermédiaire de la passe avant, garde quelques marque-pages et redérive le reste quand tu en as besoin.

## Modèle mémoire : où vont les octets ?

Avec Adam en précision mixte, chaque paramètre coûte

| Élément | Octets par paramètre |
|---|---|
| paramètres (fp16/bf16) | 2 |
| gradients (fp16/bf16) | 2 |
| copie maîtresse fp32 des poids | 4 |
| premier moment d'Adam $m$ (fp32) | 4 |
| second moment d'Adam $v$ (fp32) | 4 |
| **état total du modèle** | **16** |

Donc $16\psi$ octets pour $\psi$ paramètres, avant toute activation. Un modèle de 7.5 Md a besoin de $16\cdot7.5\times10^9/2^{30}=111.8$ GiB, plus qu'un GPU de 80 Go. (Le tableau de la leçon source compte 12 octets par paramètre : 2 pour les poids, 2 pour les gradients et 8 pour les moments d'Adam, en omettant la copie maîtresse fp32 ; le décompte à 16 octets est le plus courant pour l'entraînement en précision mixte avec poids maîtres, que je cite d'après ma connaissance générale de l'article ZeRO plutôt que d'après la source.) Le parallélisme de données réplique les 16 octets sur chaque GPU, donc ajouter des GPU achète de la vitesse, pas de la capacité.

**ZeRO / FSDP** supprime la redondance en partitionnant sur $N$ GPU :

| Étape | Partitionné | Octets par GPU et par paramètre |
|---|---|---|
| 0 (DP simple) | rien | $2+2+12=16$ |
| 1 | état de l'optimiseur | $2+2+12/N$ |
| 2 | + gradients | $2+(2+12)/N$ |
| 3 (= FSDP) | + paramètres | $16/N$ |

Pour $\psi=7.5$ Md et $N=8$ : 111.76, 38.42, 26.19 et 13.97 GiB. L'étape 3 doit **all-gather** (rassembler) les paramètres de chaque couche juste avant son utilisation (et les libérer ensuite), ce qui coûte de la communication supplémentaire qui se recouvre avec le calcul. ZeRO-Offload / Infinity déplacent l'état de l'optimiseur vers la RAM du CPU ou du NVMe, échangeant de la vitesse contre de la capacité. Pour 70 Md sur 64 GPU : l'étape 0 fait 1043 GiB par GPU (impossible), l'étape 3 fait 16.3 GiB ; avec un budget de 80 GiB, l'étape 3 a besoin de 14 GPU ($1043.1/80=13.04$) avant même de compter les activations.

## Stratégies de parallélisme

**Parallèle de données (DP).** Découper le batch en $N$ tranches ; chaque GPU calcule les gradients sur sa tranche ; un **all-reduce** les moyenne ; chaque réplique applique la même mise à jour. Batch effectif $=$ batch par GPU $\times N$.

**Parallèle de tenseurs (TP).** Découper une matrice de poids. *Parallèle en colonnes* : $W=[W_1\,|\,W_2]$, chaque GPU calcule $xW_i$, les résultats sont concaténés : $x[W_1|W_2]=[xW_1|xW_2]$. *Parallèle en lignes* : $x=[x_1\,|\,x_2]$, $W=\begin{bmatrix}W_1\\W_2\end{bmatrix}$, donc $xW=x_1W_1+x_2W_2$ et les sommes partielles sont additionnées par un all-reduce. Dans un MLP de transformer, la première matrice est parallèle en colonnes et la seconde parallèle en lignes, donc un seul all-reduce par MLP suffit. Le TP est exact (identique au calcul non découpé) mais communique à l'intérieur de chaque couche, il est donc confiné aux GPU à interconnexion rapide.

**Parallèle de pipeline (PP).** Affecter des groupes de couches à $p$ étapes ; découper le batch en $m$ micro-batchs pour que les étapes se recouvrent. Pendant le remplissage et la vidange, des étapes sont inactives. Avec un ordonnancement de style GPipe, la fraction d'inactivité de l'exécution complète est

$$
\frac{p-1}{m+p-1},
$$

par exemple $p=4$ : $m=1$ donne 75 % d'inactivité, $m=16$ donne $3/19=15.8\%$. Le temps supplémentaire par rapport à l'idéal est $(p-1)/m=18.75\%$ pour $m=16$ (la source cite ce dernier chiffre). Des ordonnancements comme 1F1B (démarrer tôt la rétropropagation, moins de mémoire d'activations), zero-bubble (séparer la rétropropagation en parties gradient d'entrée et gradient de poids pour combler les trous) et **DualPipe** la réduisent encore. DualPipe alimente des micro-batchs depuis les deux extrémités du pipeline et recouvre l'échange all-to-all de tokens des couches MoE ([[adv.03]]) avec le calcul, au prix de conserver deux copies des couches de chaque étape ; DeepSeek-V3 rapporte plus de 95 % d'utilisation des GPU.

Le **parallélisme 3D** combine les trois : par exemple 8 voies de TP dans un nœud, 16 voies de PP entre nœuds et 128 voies de DP donnent 16 384 GPU. Exemple de la source : Llama 3 405B sur 16 384 H100.

## Précision mixte et mise à l'échelle de la perte

fp32 a 8 bits d'exposant et 23 de mantisse ; **fp16** en a 5 et 10 (plage jusqu'à 65 504, plus petit dénormalisé environ $6\times10^{-8}$) ; **bf16** en a 8 et 7 (la plage de fp32, précision plus grossière). La passe avant et la rétropropagation s'exécutent en 16 bits pour la vitesse et la mémoire, tandis que les **poids maîtres et les moments d'Adam restent en fp32**, car sinon les minuscules mises à jour $\eta\hat m/\sqrt{\hat v}$ seraient arrondies à néant.

**Sous-dépassement du gradient (underflow).** En fp16, un gradient de $10^{-8}$ devient exactement 0. La **mise à l'échelle de la perte** multiplie la perte (donc tous les gradients, par linéarité de la rétropropagation) par $S$ avant la passe arrière et divise les gradients par $S$ en fp32 avant le pas d'optimisation. Avec $S=1024$, $10^{-8}\to1.025\times10^{-5}$, représentable ; en divisant de nouveau on obtient $1.0012\times10^{-8}$ (0.1 % d'erreur). Un $S$ trop grand déborde vers $\infty$. **Mise à l'échelle dynamique de la perte** : sur un gradient non fini, sauter le pas et diviser $S$ par deux ; après une série (par exemple 2000) de pas propres, doubler $S$. Le bf16 a rarement besoin de mise à l'échelle car sa plage égale celle de fp32, c'est pourquoi il est devenu le choix par défaut.

## Gradient checkpointing

La rétropropagation a besoin des activations de la passe avant de chaque couche. Disons que chacune des $L$ couches stocke $A$ unités et que l'entrée d'une couche coûte $a$ unités ($a\ll A$). Le **checkpointing** ne garde que les entrées d'une couche sur $k$ ; pendant la rétropropagation, chaque segment de $k$ couches est ré-exécuté en avant pour régénérer ses activations. Mémoire :

$$
M(k)=\frac{L}{k}\,a+k\,A .
$$

En minimisant sur $k$ : $dM/dk=-La/k^2+A=0\Rightarrow k^\star=\sqrt{La/A}$, donnant $M^\star=2\sqrt{LaA}$. Avec $a=A=1$ et $L=64$ : $k^\star=8$ et $M=16$ au lieu de 64, un gain de 4x.

**Coût.** Un pas normal est avant (1 unité) + arrière (2 unités) $=3$ par couche. Tout recalculer ajoute une passe avant : $4/3$, un surcoût de **33 %**. Le checkpointing **sélectif** ne recalcule que la partie coûteuse à stocker et peu coûteuse à recalculer (les scores d'attention en $O(L^2)$), disons 15 % d'une passe avant : $(3+0.15)/3=1.05$, 5 % de surcoût, selon le modèle de coût de la source. Alternative : **décharger** (offload) les activations vers la RAM du CPU si la bande passante PCIe est disponible.

## Prédiction multi-tokens (MTP)

L'entraînement standard supervise chaque position sur une seule cible, le token suivant. Le MTP ajoute des têtes pour des tokens plus éloignés. La conception de DeepSeek-V3 est **séquentielle** : le module $d$ prend l'état caché de la profondeur précédente, le combine (après normalisation et une projection) avec l'embedding du vrai token à cette profondeur, exécute un bloc de transformer, et prédit le décalage suivant en utilisant l'embedding et la tête de sortie *partagés* ; l'enchaînement préserve la structure causale (contrairement aux têtes parallèles indépendantes de Gloeckle et al.). La perte est

$$
\mathcal{L}=\mathcal{L}_{\text{main}}+\frac{\lambda}{D}\sum_{d=1}^{D}\mathcal{L}_d,\qquad \mathcal{L}_d=\operatorname{CE}\big(\text{head}_d(\cdot),\;t_{i+1+d}\big),
$$

avec $\lambda=0.3$ au début de l'entraînement et $0.1$ plus tard (source). Dans l'exercice, j'implémente la perte avec la pondération $\lambda/D$ et des têtes indépendantes par simplicité ; l'enchaînement séquentiel est décrit, pas implémenté. Avantages : un signal d'entraînement plus dense, et le module entraîné peut servir de brouillon de décodage spéculatif à l'inférence (acceptation rapportée de 80 %+ et débit environ 1.8x ; voir [[sys.02]]). Surcoût : 14 Md de paramètres sur 671 Md (2 %).

## Exemple détaillé

1. **Mémoire.** 7.5 Md de paramètres sur 8 GPU : ZeRO-0 111.76, ZeRO-1 38.42, ZeRO-2 26.19, ZeRO-3 13.97 GiB par GPU (état du modèle seulement).
2. **Bulle.** $p=4$ : $m=1,4,16$ donnent des fractions d'inactivité $75\%,\ 42.9\%,\ 15.8\%$.
3. **Checkpoint.** $L=64$ : $k=8$ est optimal pour des tailles égales ; si les entrées sont quatre fois moins coûteuses ($a=0.25$), le meilleur est $k=4$ avec $M=16\cdot0.25+4=8$.
4. **Mise à l'échelle de la perte.** $g=10^{-8}$ : le fp16 sans mise à l'échelle donne 0 ; $S=1024$ récupère $1.001\times10^{-8}$.
5. **MTP.** Avec des logits uniformes, chaque tête a une perte $\ln V$ ; pour $V=5$ c'est 1.609, et le terme supplémentaire vaut $0.3\ln5=0.483$.

Tous les nombres proviennent du code de l'exercice.

## Implémentation

```python
import numpy as np

def model_state_gib(psi, n, stage, bp=2, bg=2, bo=12):
    p, g, o = bp * psi, bg * psi, bo * psi
    if stage >= 1: o /= n
    if stage >= 2: g /= n
    if stage >= 3: p /= n
    return (p + g + o) / 2**30

def checkpoint_memory(L, k, a_in, a_layer):
    return -(-L // k) * a_in + k * a_layer      # ceil(L/k) inputs + one live segment
```

Production : `DistributedDataParallel` et `FullyShardedDataParallel` (FSDP2) de PyTorch, DeepSpeed ZeRO, Megatron-Core (TP, PP, recalcul sélectif), `torch.utils.checkpoint`, `torch.autocast` avec `GradScaler`.

## Formes

| Quantité | Forme / taille |
|---|---|
| état du modèle par GPU (ZeRO-3) | $16\psi/N$ octets |
| bloc parallèle en colonnes | $W_i$ : (d_in, d_out/n) |
| bloc parallèle en lignes | $x_i$ : (B, d_in/n), $W_i$ : (d_in/n, d_out) ; sorties partielles (B, d_out) sommées |
| étape de pipeline | L/p couches consécutives |
| logits MTP | un tenseur (T, V) par tête |

## Idées reçues

- **« Plus de GPU en parallélisme de données me permet d'entraîner un modèle plus gros. »** Chaque réplique détient tout ; utilise ZeRO/FSDP, TP ou PP pour la capacité.
- **« La précision mixte stocke tout en fp16. »** Les poids maîtres et les moments de l'optimiseur restent en fp32.
- **« La mise à l'échelle de la perte change le résultat de l'entraînement. »** Elle s'annule exactement (aux arrondis près) ; elle ne fait que préserver les petits gradients.
- **« Le checkpointing rend la rétropropagation indépendante des activations de la passe avant. »** Il les recalcule, ce qui coûte environ un tiers de FLOPs en plus pour le checkpointing complet.
- **« La bulle du pipeline disparaît avec assez d'étapes. »** Plus d'étapes agrandissent la bulle pour un $m$ donné.
- **« ZeRO-3 est gratuit. »** Il ajoute des all-gathers par couche ; le recouvrement en cache beaucoup mais pas tout.

## Points clés à retenir

- Adam en précision mixte coûte 16 octets par paramètre ; les étapes ZeRO 1-3 partitionnent l'état de l'optimiseur, les gradients, puis les paramètres sur $N$ GPU.
- Le DP met à l'échelle le batch, le TP découpe les matrices (exact, lourd en communication), le PP découpe les couches (bulle $\frac{p-1}{m+p-1}$) ; la production combine les trois.
- La mise à l'échelle de la perte maintient en vie les petits gradients fp16 ; le bf16 évite en grande partie le problème.
- Checkpointer toutes les $\sqrt{La/A}$ couches : mémoire $O(\sqrt L)$ pour environ 33 % de calcul en plus (environ 5 % avec recalcul sélectif).
- Le MTP ajoute $\frac{\lambda}{D}\sum_d\mathcal{L}_d$ pour une supervision plus dense et un modèle brouillon gratuit.

## Explique-le toi-même

1. Pourquoi ajouter des répliques en parallélisme de données n'aide-t-il jamais un modèle qui ne tient pas sur un GPU, et quelle étape de ZeRO le permet en premier ?
2. Dérive l'intervalle de checkpoint optimal et explique pourquoi les scores d'attention sont la première chose que le checkpointing sélectif recalcule.

## Pour aller plus loin

- ZeRO (Rajbhandari et al., 2020), Megatron-LM (Narayanan et al., 2021), Chen et al. (2016) sur l'entraînement à mémoire sous-linéaire, Micikevicius et al. (2018) sur la précision mixte, rapport DeepSeek-V3 ; résumés dans les leçons sources.
- [[adv.03]] pour le MoE et le parallélisme d'experts, [[adv.04]] pour la mémoire côté inférence.

## Critères de maîtrise

Tu sais calculer l'état du modèle par GPU pour toute étape de ZeRO, découper une multiplication matricielle par colonnes et par lignes, calculer les bulles de pipeline, choisir un intervalle de checkpoint et son surcoût, simuler l'underflow fp16 avec et sans mise à l'échelle de la perte, et écrire la perte MTP.
