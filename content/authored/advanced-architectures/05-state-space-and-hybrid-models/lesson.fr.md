---
id: adv.05
title: "Modèles à espace d'états, Mamba et architectures hybrides"
summary: "Récurrences linéaires à mémoire constante, l'idée sélective (Mamba) sous forme de code exécutable, le scan parallèle qui les rend entraînables, et l'hybride attention/SSM de style Jamba."
minutes: 80
teaches: [adv.ssm, adv.scan]
requires: [la.matrix, alg.exp-log, trf.architectures, adv.kv-sharing]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/docs/en.md: SSM recurrence in brief, S4 vs Mamba vs Mamba-2/3, Jamba 1:7 layout with MoE every other layer, memory budget at 256k. READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/docs/en.md: only searched (grep) for Mamba mentions; not used for content."
---

## Où cela s'inscrit

L'attention ([[trf.01]]) se souvient exactement de chaque token passé, au prix d'un cache KV qui croît avec la séquence ([[adv.04]] a montré à quel point il devient gros) et d'un calcul d'entraînement en $O(N^2)$. Un modèle **récurrent** comprime au contraire le passé dans un état de taille fixe : mémoire constante, temps constant par token généré. Les RNN classiques le faisaient mais étaient lents à entraîner (strictement séquentiels) et instables. Les **modèles à espace d'états (SSM)** sont des récurrences linéaires qui peuvent s'entraîner en parallèle ; **Mamba** les a rendus *sélectifs* (ce dont on se souvient dépend de l'entrée) et est devenu compétitif avec les transformers. En pratique, les meilleurs systèmes à contexte long mélangent les deux : quelques couches d'attention pour un rappel exact, de nombreuses couches SSM pour un traitement de masse bon marché. Cette leçon implémente la récurrence, ses vues par convolution et par scan, un SSM sélectif, et l'arithmétique mémoire d'un hybride.

## L'idée en mots simples

Une couche d'attention est un classeur : chaque page jamais lue reste dans un tiroir et peut être retrouvée mot pour mot, mais le classeur grossit à chaque page. Un SSM est un **carnet à nombre de pages fixe** : après chaque nouvelle phrase, tu réécris le carnet en estompant légèrement les anciennes notes et en ajoutant un résumé de la nouvelle phrase. La mémoire ne croît jamais, mais les anciens détails s'estompent et le carnet peut ne plus avoir de place pour le seul détail dont tu auras besoin plus tard.

**Sélectif** signifie que le rédacteur décide, phrase par phrase, de combien estomper les anciennes notes et de combien écrire. Pour du texte de remplissage, n'écris rien et laisse les notes intactes. À un changement de sujet, estompe beaucoup et repars à neuf. Un taux d'estompage fixe (un SSM non sélectif) ne peut pas faire cela.

Un **hybride** garde quelques couches-classeurs là où la recherche exacte compte et utilise des carnets partout ailleurs.

## La récurrence linéaire

Pour un canal d'entrée $x_t\in\mathbb{R}$, un état $h_t\in\mathbb{R}^n$ et une dynamique diagonale,

$$
h_t=a\odot h_{t-1}+b\,x_t,\qquad y_t=c^\top h_t ,\qquad a,b,c\in\mathbb{R}^n,\ h_{-1}=0 .
$$

$a$ (élément par élément, $|a_i|<1$) est la part de chaque coordonnée d'état qui survit, $b$ écrit l'entrée, $c$ lit la sortie. Calculer $y_t$ n'exige que $h_{t-1}$ et $x_t$ : **mémoire et temps en $O(1)$ par token** à l'inférence, indépendants de la longueur de l'historique.

**D'où cela vient-il ?** Un système en temps continu $\dot h=Ah+Bx$ échantillonné avec un pas $\Delta$ et l'entrée maintenue constante (blocage d'ordre zéro) donne

$$
h_t=\bar A\,h_{t-1}+\bar B\,x_t,\qquad \bar A=e^{\Delta A},\quad \bar B=A^{-1}(e^{\Delta A}-I)B .
$$

Pour $\Delta A$ petit, $\bar B\approx\Delta B$. Pour $A$ diagonale négative, $\bar A=e^{\Delta A}\in(0,1)$ est la décroissance, et $\Delta$ est la durée pendant laquelle le système « écoute » : un $\Delta$ plus grand signifie plus d'oubli du passé et une écriture plus forte du présent. (Les formules de discrétisation sont standard ; la leçon source n'énonce la récurrence qu'à haut niveau, ce paragraphe relève donc de la culture générale.)

**Vue par convolution.** En déroulant, $y_t=\sum_{k=0}^{t}c^\top\operatorname{diag}(a)^k b\;x_{t-k}=\sum_k K_k x_{t-k}$ avec le noyau $K_k=\sum_i c_ia_i^kb_i$. Si $a,b,c$ ne dépendent pas de $t$, le modèle est une convolution **linéaire invariante dans le temps**, entraînable en parallèle pour tous les $t$ (S4 l'évalue avec des FFT). L'exercice vérifie que les calculs récurrent et convolutif concordent.

## L'idée sélective (Mamba)

Faire dépendre les paramètres de l'entrée : $\Delta_t,B_t,C_t$ sont calculés à partir de $x_t$ par de petites couches linéaires ($\Delta_t=\operatorname{softplus}(\cdot)>0$). Pour $D$ canaux, chacun avec son propre état de dimension $n$,

$$
h_t=\exp(\Delta_t A)\odot h_{t-1}+(\Delta_t\otimes B_t)\,x_t ,\qquad y_t=h_tC_t ,
$$

avec les formes $h_t\in\mathbb{R}^{D\times n}$, $A\in\mathbb{R}^{D\times n}$ (négative), $\Delta_t\in\mathbb{R}^{D}$, $B_t,C_t\in\mathbb{R}^{n}$, $x_t\in\mathbb{R}^D$, $y_t\in\mathbb{R}^D$. Vérifie le comportement que teste l'exercice :

- $\Delta_t\to0$ : $\bar A\to1$ et $\bar B\to0$ : le token est **ignoré** et l'état reporté inchangé.
- $\Delta_t$ grand : $\bar A\to0$ : l'état est **réinitialisé** et le token courant domine.

Ainsi $\Delta_t$ agit comme une porte d'oubli dépendant de l'entrée ; $B_t$ choisit ce qu'il faut écrire et $C_t$ ce qu'il faut lire. Cela permet au modèle de faire des choses qu'un filtre fixe ne peut pas, comme copier certains tokens en sautant le remplissage. Le prix : avec des paramètres dépendant de l'entrée, le modèle n'est plus une convolution fixe, donc nous ne pouvons pas utiliser la FFT. Il nous faut une autre façon de paralléliser.

## Scan parallèle

Chaque pas est une **application affine** $h\mapsto a_th+b_t$, la paire $e_t=(a_t,b_t)$. Composer deux applications, appliquer $e_1$ puis $e_2$ :

$$
e_2\circ e_1:\; h\mapsto a_2(a_1h+b_1)+b_2=(a_1a_2)\,h+(a_2b_1+b_2),
$$

donc $\operatorname{combine}(e_1,e_2)=(a_1a_2,\;a_2b_1+b_2)$. Cet opérateur est **associatif** (la composition de fonctions l'est toujours), et $h_t$ est la composante $b$ de $e_t\circ\dots\circ e_0$ appliquée à $h_{-1}=0$. Un opérateur associatif signifie que nous pouvons parenthéser le produit comme nous le voulons, donc un *scan préfixe* peut se faire en $\lceil\log_2T\rceil$ tours : au tour $r$, chaque élément $i\ge2^r$ se combine avec l'élément $i-2^r$ (Hillis-Steele). Séquentiel : $T$ pas dépendants. Parallèle : $\lceil\log_2 T\rceil$ tours dépendants, chacun entièrement parallèle. La version simple fait $O(T\log T)$ de travail total ; les variantes efficaces en travail (Blelloch) atteignent $O(T)$. Les vraies implémentations de Mamba utilisent un noyau GPU fusionné pour ce scan qui garde l'état étendu dans la mémoire rapide sur puce (un détail de systèmes qui dépasse cette leçon).

## Exemple détaillé

État scalaire, $a=0.5$, $b=1$, $c=1$, entrée $x=(1,2,0,1)$.

- Récurrence : $h=(1,\;0.5\cdot1+2=2.5,\;0.5\cdot2.5+0=1.25,\;0.5\cdot1.25+1=1.625)$, donc $y=h$.
- L'impulsion $(1,0,0,0)$ donne $1,0.5,0.25,0.125$ : le noyau $K_k=0.5^k$.
- Éléments du scan $e_t=(0.5,\,b_t)$ avec $b=(1,2,0,1)$. Tour 1 (décalage 1) : $e_1'=(0.25,\,0.5\cdot1+2)=(0.25,2.5)$, $e_2'=(0.25,\,0.5\cdot2+0)=(0.25,1)$, $e_3'=(0.25,\,0.5\cdot0+1)=(0.25,1)$. Tour 2 (décalage 2) : $e_2''=(0.0625,\,0.25\cdot1+1)=(0.0625,1.25)$, $e_3''=(0.0625,\,0.25\cdot2.5+1)=(0.0625,1.625)$. Les composantes $b$ $(1,2.5,1.25,1.625)$ concordent. Deux tours pour quatre éléments.

## Hybrides et quand chacun gagne

**Jamba** (AI21) entrelace les couches par blocs de huit : 7 Mamba + 1 attention, avec un feed-forward en mélange d'experts ([[adv.03]]) une couche sur deux. La source rapporte 52 Md de paramètres au total / 12 Md actifs avec une fenêtre de 256k sur un seul GPU de 80 Go. Raisonnement : les couches d'attention fournissent le rappel exact et le suivi d'état ; les couches SSM fournissent un traitement bon marché. Les ablations rapportées là-bas ont trouvé le meilleur compromis autour de 1:7 à 1:8 ; trop peu d'attention nuit à la récupération en contexte, trop coûte de la mémoire. Jamba 1.5 ajoute des positions rotatives aux couches d'attention.

**Mémoire à 256k tokens (BF16), 32 couches, 32 têtes, dimension de tête 128**, calculée par l'exercice :

| Conception | KV / état | Total |
|---|---|---|
| transformer pur, MHA | 32 couches d'attention | 128 GiB |
| transformer pur, GQA (8 têtes KV) | 32 couches | 32 GiB |
| de forme Jamba 28 SSM + 4 attention (MHA) | KV pour 4 couches + états minuscules | 16 GiB |

(Le texte de la source imprime le KV de l'hybride comme « 8.4 GB » à côté de la formule $2\cdot4\cdot32\cdot128\cdot256\text{k}\cdot2$, qui vaut 16 GiB, et dit plus loin 16 GB ; je suis la formule. L'état SSM par couche fait de quelques dizaines de kilooctets à des mégaoctets et ne croît pas avec la longueur.)

| | Transformer | SSM (Mamba) | Hybride |
|---|---|---|---|
| coût d'entraînement | $O(N^2)$, parallèle | $O(N)$, scan | intermédiaire |
| mémoire par token généré | croît avec $N$ | constante | croît lentement (peu de couches d'attention) |
| rappel exact / copie depuis un long contexte | fort | faible (état de taille fixe) | assez fort |
| contextes courts, maturité de l'outillage | meilleur | bon | complexe |

Les modèles purement SSM égalent les transformers en perplexité à petite échelle mais sont en retard sur la récupération en contexte et le suivi d'état ; d'où les hybrides. Mamba-3 (tel que rapporté dans la source) ajoute une discrétisation plus précise, des mises à jour d'état à valeurs complexes et des projections à entrées multiples et sorties multiples ; je n'ai pas étudié ses détails.

## Implémentation

```python
import numpy as np

def scan_combine(e1, e2):                       # apply e1 first, then e2
    return (e1[0] * e2[0], e2[0] * e1[1] + e2[1])

def parallel_scan(a, b):
    A, B = np.array(a, float), np.array(b, float)
    T, offset = len(A), 1
    while offset < T:
        nA, nB = A.copy(), B.copy()
        for i in range(offset, T):
            nA[i], nB[i] = scan_combine((A[i-offset], B[i-offset]), (A[i], B[i]))
        A, B, offset = nA, nB, offset * 2
    return B

def selective_scan(x, delta, A, B, C):          # x:(T,D) A:(D,n) B,C:(T,n)
    h = np.zeros(A.shape); y = np.zeros_like(x)
    for t in range(len(x)):
        h = np.exp(delta[t][:, None] * A) * h + delta[t][:, None] * B[t] * x[t][:, None]
        y[t] = h @ C[t]
    return y
```

La boucle interne sur $i$ est la partie qu'un GPU exécute en parallèle. Production : le paquet `mamba-ssm` (noyaux CUDA fusionnés) et les classes Mamba/Jamba de Hugging Face.

## Formes

| Tenseur | Forme |
|---|---|
| séquence d'entrée | (T, D) |
| état $h$ | (D, n), constant en T |
| $A$ | (D, n), négative |
| $\Delta_t$ | (T, D) |
| $B_t,C_t$ | (T, n) |
| éléments du scan $(a_t,b_t)$ | chacun (T, ...) |

## Idées reçues

- **« Les SSM n'ont pas de limite mémoire. »** L'état est de taille fixe ; les longs historiques sont comprimés avec perte.
- **« Les SSM sélectifs sont des convolutions. »** Seuls les invariants dans le temps le sont ; la sélectivité casse cela et exige un scan.
- **« Un scan parallèle est plus rapide en FLOPs. »** Il fait plus de travail total que la boucle séquentielle ; il gagne par la profondeur et l'utilisation du matériel.
- **« Les hybrides sont juste deux modèles collés. »** Les couches sont entrelacées dans une seule pile résiduelle, et le ratio est un hyperparamètre réglé.
- **« Un temps constant par token signifie toujours plus rapide. »** Les contextes courts sont dominés par les poids, où un bon noyau de transformer est difficile à battre.

## Points clés à retenir

- Une récurrence linéaire $h_t=a\odot h_{t-1}+bx_t$ donne une inférence en $O(1)$ par token ; déroulée, c'est une convolution.
- Mamba rend $\Delta,B,C$ dépendants de l'entrée : un petit $\Delta$ ignore un token, un grand $\Delta$ réinitialise l'état.
- Les applications affines se composent de façon associative, $(a_1a_2,\,a_2b_1+b_2)$, donc la récurrence peut être scannée en $\lceil\log_2T\rceil$ tours parallèles.
- Les hybrides utilisent quelques couches d'attention pour le rappel exact et de nombreuses couches SSM pour un contexte bon marché ; Jamba utilise environ 1 couche d'attention pour 7 couches Mamba.

## Explique-le toi-même

1. Pourquoi la sélectivité empêche-t-elle l'astuce de la convolution par FFT, et qu'est-ce qui la remplace ?
2. Un chat de 256k tokens a besoin de citations exactes du premier message. Sur quel type de couche t'appuierais-tu, et que mettrais-tu dans le reste du modèle ?

## Pour aller plus loin

- S4 (Gu et al., 2021), Mamba (Gu et Dao, 2023), Mamba-2, Jamba et Jamba 1.5, tels que résumés dans la leçon source.
- [[adv.04]] pour les variantes de cache KV ; [[adv.03]] pour la partie MoE de Jamba.

## Critères de maîtrise

Tu sais implémenter la récurrence, la convolution et le scan sélectif, prouver que l'opérateur de scan est associatif et l'utiliser pour calculer la récurrence en profondeur logarithmique, et estimer la mémoire d'un hybride contre un transformer.
