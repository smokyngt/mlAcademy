---
id: trf.03
title: "Information de position, encodages sinusoïdaux et RoPE"
summary: "Pourquoi l'attention ne peut pas connaître l'ordre des mots par elle-même, comment les encodages de position sinusoïdaux, appris et rotatifs y remédient, et pourquoi RoPE rend le score d'attention dépendant uniquement de la distance relative, démontré avec des rotations 2-D et vérifié numériquement."
minutes: 75
teaches: [trf.position, trf.rope]
requires: [trf.attention, la.matrix, la.dot]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/04-positional-encoding/docs/en.md (permutation invariance, sinusoidal formula, RoPE and the relative-distance property, ALiBi, comparison table). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/05-full-transformer/docs/en.md (where the positional signal enters the model; RoPE vs sinusoidal in the 2026 block table). READ."
  - "LLMs-from-scratch/ch04/01_main-chapter-code/ch04.ipynb (Sec. 4.1: GPT-2 uses learned token and positional embedding tables tok_emb and pos_emb). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/docs/en.md (transformers assume no locality, unlike RNNs). READ."
---

## Objectifs d'apprentissage

- Prouver que l'attention sans information de position est équivariante par permutation, donc qu'elle voit un sac de mots.
- Écrire l'encodage sinusoïdal, donner sa forme, et montrer qu'un décalage de $k$ positions est une rotation fixe de chaque paire sinus/cosinus.
- Comparer les positions absolues apprises aux positions sinusoïdales.
- Démontrer les embeddings de position rotatifs (RoPE) comme des rotations 2-D et prouver que $\langle R_mq,R_nk\rangle$ ne dépend que de $m-n$.
- Implémenter les encodages sinusoïdaux et RoPE et tester numériquement la propriété de décalage relatif.

## Où cela s'inscrit

Un réseau récurrent connaît l'ordre des mots gratuitement : il les lit dans l'ordre. Un transformer lit tous les tokens d'un coup ; si on ne fait rien, « le chien mord l'homme » et « l'homme mord le chien » sont le *même ensemble de tokens* et donnent le même ensemble de sorties. Pour le langage, le code et l'audio, l'ordre porte du sens, nous devons donc injecter **la position de chaque token**.

Le choix a de vraies conséquences. Il décide si un modèle entraîné sur des documents de 2 000 tokens peut lire 20 000 tokens, si les scores d'attention peuvent exprimer « le mot situé deux rangs à ma gauche », et quelle est la taille de la première couche. Cette leçon couvre les trois réponses principales : les sinusoïdes fixes (l'original de 2017), les tables apprises (GPT-2) et les embeddings rotatifs (RoPE, utilisés par la plupart des modèles ouverts récents d'après la leçon source).

## L'idée en mots simples

**Le problème.** Imagine une salle de classe où les élèves lèvent la main pour répondre, et où l'enseignante recueille les réponses sans noter qui est assis où. Tout ce qu'elle connaît, c'est l'ensemble des réponses. L'attention traite les tokens de la même façon : elle mesure la similarité par paires entre des vecteurs de contenu et fait la moyenne ; les numéros de place n'entrent jamais en jeu.

**Solution 1 : coller un numéro de place sur chaque élève (position absolue).** On ajoute à l'embedding de chaque token un vecteur qui dit « je suis le token numéro 7 ». Soit conçu à l'avance à partir d'ondes sinus et cosinus de nombreuses longueurs d'onde différentes (comme un jeu d'aiguilles d'horloge : la rapide donne la minute exacte, la lente donne à peu près l'heure), soit appris à partir des données, comme une table de correspondance avec une ligne par place.

**Solution 2 : mesurer l'angle entre les élèves (position relative, RoPE).** Au lieu d'ajouter de l'information au token, on *fait tourner* ses vecteurs requête et clé d'un angle qui croît avec la position. Deux vecteurs tournés de la même quantité gardent leur angle mutuel ; deux vecteurs tournés de quantités différentes voient leur angle changer exactement de la différence. Ainsi, quand requête et clé se rencontrent dans un produit scalaire, seule la différence des rotations survit : la **distance** entre les tokens. Les numéros de place absolus s'annulent, et « deux places à ma gauche » est quelque chose que l'attention peut exprimer directement.

## Pourquoi l'attention est équivariante par permutation (preuve)

Soit $P$ une matrice de permutation ($T\times T$, un seul 1 par ligne et par colonne, $P^\top P=I$) et soit $X'=PX$ qui réordonne les lignes (tokens). Les projections agissent ligne par ligne, donc $Q'=PQ$, $K'=PK$, $V'=PV$. Les scores deviennent

$$
S'=\frac{Q'K'^\top}{\sqrt{d_k}}=\frac{PQK^\top P^\top}{\sqrt{d_k}}=PSP^\top .
$$

Le softmax ligne par ligne commute avec une permutation simultanée des lignes et des colonnes : $\operatorname{softmax}(PSP^\top)=P\operatorname{softmax}(S)P^\top$ (chaque ligne est un ensemble de nombres, le softmax ne se soucie pas de l'ordre dans lequel on les liste). Donc

$$
Z'=A'V'=PAP^\top\,PV=PA\,(P^\top P)\,V=PAV=PZ .
$$

Mélanger les lignes d'entrée mélange les lignes de sortie de la même façon et ne change rien d'autre. Une couche d'attention pure ne peut pas distinguer les ordres. (Avec un masque causal, ce n'est plus exactement vrai, puisque le masque lui-même brise la symétrie : un décodeur tire un peu d'information d'ordre du masque seul, mais pas assez pour savoir *à quelle distance* se trouve un token.)

## Encodages sinusoïdaux

Pour une position $p\in\{0,\dots,T-1\}$ et une dimension de modèle $d$ paire, définis $\omega_i=10000^{-2i/d}$ pour $i=0,\dots,d/2-1$ et

$$
PE_{p,2i}=\sin(p\,\omega_i),\qquad PE_{p,2i+1}=\cos(p\,\omega_i).
$$

La matrice $PE$ a la forme $(T,d)$ et est **ajoutée** aux embeddings de tokens : $X\leftarrow X+PE$, les deux de forme $(T,d)$ (diffusée sur le batch). La paire $i$ est une sinusoïde de longueur d'onde $2\pi/\omega_i$, de $2\pi$ tokens pour $i=0$ jusqu'à $2\pi\cdot10000$ tokens pour la dernière paire. Les paires rapides résolvent les positions voisines, les paires lentes encodent la localisation grossière.

**Un décalage est une application linéaire.** Fixe une paire $i$ et note $\omega=\omega_i$. En utilisant $\sin(a+b)=\sin a\cos b+\cos a\sin b$ et $\cos(a+b)=\cos a\cos b-\sin a\sin b$,

$$
\begin{pmatrix}\sin((p+k)\omega)\\\cos((p+k)\omega)\end{pmatrix}
=\begin{pmatrix}\cos k\omega&\sin k\omega\\-\sin k\omega&\cos k\omega\end{pmatrix}
\begin{pmatrix}\sin p\omega\\\cos p\omega\end{pmatrix}.
$$

La matrice ne dépend que du décalage $k$, pas de $p$. Un modèle *peut* donc apprendre des applications linéaires qui lisent des décalages relatifs à partir d'entrées sinusoïdales. Mais il doit le faire à travers les projections $W_Q,W_K$ et le contenu se mélange ; rien ne l'y oblige, et la sinusoïde est ajoutée au contenu au lieu d'être un canal séparé.

**Positions absolues apprises.** GPT-2, tel que construit au chapitre 4 du livre, utilise une table d'embeddings `pos_emb` de forme $(T_{\max},d)$ consultée par indice de position et ajoutée aux embeddings de tokens (la même opération que pour les sinusoïdes, mais la table est entraînée). Coût : $T_{\max}d$ paramètres (pour $T_{\max}=1024$, $d=768$ : $786\,432$). Faiblesse : les lignes au-delà de $T_{\max}$ n'existent pas, et les lignes jamais vues à l'entraînement ne sont pas entraînées. La leçon source note aussi que les sinusoïdes fixes extrapolent mal en pratique.

## Démonstration de RoPE

**Objectif.** Trouver des applications $f_q(x,m)$, $f_k(x,n)$ (position $m$ pour la requête, $n$ pour la clé) telles que le score soit une fonction de $x_q,x_k$ et $m-n$ uniquement :

$$
\langle f_q(x_q,m),\,f_k(x_k,n)\rangle=g(x_q,x_k,m-n).
$$

**On commence en deux dimensions.** Soit $R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}$ la rotation d'angle $\phi$. Elle est orthogonale : $R(\phi)^\top=R(-\phi)$ et $R(a)R(b)=R(a+b)$. Définis $f(x,m)=R(m\theta)x$, c'est-à-dire une rotation d'un angle proportionnel à la position, avec une fréquence fixe $\theta$. Alors

$$
\langle R(m\theta)q,\,R(n\theta)k\rangle=q^\top R(m\theta)^\top R(n\theta)\,k=q^\top R(-m\theta)R(n\theta)\,k=q^\top R\big((n-m)\theta\big)\,k .
$$

Seul $n-m$ apparaît. En développant avec $q=(q_1,q_2)$, $k=(k_1,k_2)$ et $\phi=(m-n)\theta$, on obtient la forme concrète

$$
\langle R(m\theta)q,R(n\theta)k\rangle=\cos\phi\,(q\cdot k)+\sin\phi\,(q_1k_2-q_2k_1).
$$

(De façon équivalente, en notation complexe avec $q\leftrightarrow q_1+iq_2$, le score vaut $\operatorname{Re}(q\,\bar k\,e^{i\phi})$.) Le score d'attention est le produit scalaire ordinaire, « tourné » de l'angle relatif.

**Plusieurs dimensions.** Pour une dimension de tête $d_h$ (paire), regroupe les coordonnées en paires consécutives $(2i,2i+1)$, $i=0,\dots,d_h/2-1$, et fais tourner la paire $i$ avec sa propre fréquence $\theta_i=\text{base}^{-2i/d_h}$ (base $=10000$ par défaut, comme dans la leçon source). L'opérateur complet est diagonal par blocs,

$$
R_m=\operatorname{diag}\big(R(m\theta_0),\dots,R(m\theta_{d_h/2-1})\big)\in\mathbb{R}^{d_h\times d_h},
$$

toujours orthogonal, et $R_m^\top R_n=R_{n-m}$ car chaque bloc obéit à la règle 2-D. Par conséquent

$$
\langle R_mq,R_nk\rangle=q^\top R_{n-m}k .
$$

Propriétés, toutes conséquences de ce qui précède :

- **Relatif** : le score dépend du décalage, donc décaler toute la phrase de $s$ positions ne change aucun score.
- **Conserve la norme** : $\|R_mq\|=\|q\|$ (les rotations sont des isométries). L'amplitude du contenu n'est pas touchée ; seule la direction encode la position.
- **Aucun paramètre supplémentaire** et rien n'est ajouté à l'embedding : RoPE s'applique à $Q$ et $K$ (pas à $V$) dans chaque couche d'attention, par tête.
- Les différentes paires tournent à des vitesses différentes : les paires rapides résolvent les petits décalages, les lentes les grands (même idée multi-échelle que les sinusoïdes).

En pratique, on ne construit jamais la matrice $d_h\times d_h$. Pour la paire $i$ avec $c=\cos(m\theta_i)$, $s=\sin(m\theta_i)$ :

$$
x'_{2i}=x_{2i}c-x_{2i+1}s,\qquad x'_{2i+1}=x_{2i}s+x_{2i+1}c .
$$

## Formes

| Objet | Forme |
|---|---|
| positions | $(T,)$ entiers |
| table sinusoïdale $PE$ | $(T,d)$, ajoutée aux embeddings $(B,T,d)$ |
| fréquences $\theta_i$ | $(d_h/2,)$ |
| angles $m\theta_i$ | $(T,d_h/2)$ |
| $Q$ ou $K$ par tête, avant et après RoPE | $(B,h,T,d_h)$ (forme inchangée) |
| tables $\cos,\sin$ | $(T,d_h/2)$, diffusées sur $(B,h)$ |

## Exemple détaillé (vérifié en exécutant le code)

**Table sinusoïdale**, $T=4$, $d=4$ (donc $\omega_0=1$ et $\omega_1=10000^{-1/2}=0.01$) :

$$
PE=\begin{pmatrix}0&1&0&1\\0.8415&0.5403&0.0100&1.0000\\0.9093&-0.4161&0.0200&0.9998\\0.1411&-0.9900&0.0300&0.9996\end{pmatrix}.
$$

Les colonnes 0-1 oscillent avec une période de $2\pi$ ; les colonnes 2-3 bougent à peine sur quatre positions (aiguille lente). Ligne 1, colonne 0 : $\sin1=0.8415$.

**RoPE en 2-D**, $q=(1,2)$, $k=(3,1)$, $\theta=0.5$ (donc $q\cdot k=5$ et $q_1k_2-q_2k_1=1-6=-5$). Le score $\langle R(m\theta)q,R(n\theta)k\rangle$ vaut :

| $(m,n)$ | $m-n$ | score |
|---|---|---|
| $(0,0)$ | 0 | $5.0000$ |
| $(5,5)$ | 0 | $5.0000$ |
| $(2,1)$ | 1 | $1.9908$ |
| $(7,6)$ | 1 | $1.9908$ |
| $(3,0)$ | 3 | $-4.6338$ |
| $(10,7)$ | 3 | $-4.6338$ |

Même décalage, même score, quelle que soit la position absolue. Vérifie $m-n=1$ avec la forme close : $\phi=0.5$, $\cos\phi=0.8776$, $\sin\phi=0.4794$, donc $0.8776\cdot5+0.4794\cdot(-5)=1.9908$ ✓. Pour $m-n=3$, $\phi=1.5$ : $0.0707\cdot5+0.9975\cdot(-5)=-4.6338$ ✓.

**Tête complète**, $d_h=8$ avec $q,k$ aléatoires (graine 0) : $\langle R_5q,R_2k\rangle=-1.5865$ et $\langle R_{105}q,R_{102}k\rangle=-1.5865$ (même décalage 3), et $\|R_9q\|=\|q\|=1.8627$.

## Implémentation

```python
import numpy as np

def sinusoidal(T, d, base=10000.0):
    p = np.arange(T)[:, None]                       # (T, 1)
    i = np.arange(d // 2)[None, :]                  # (1, d/2)
    ang = p / base ** (2 * i / d)                   # (T, d/2)
    pe = np.zeros((T, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe                                       # (T, d)

def rope(x, pos, base=10000.0):
    """x: (..., T, d_h) with d_h even; pos: (T,) integer positions."""
    d = x.shape[-1]
    th = base ** (-2 * np.arange(d // 2) / d)       # (d/2,)
    ang = pos[:, None] * th[None, :]                # (T, d/2)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    a, b = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out
```

## Limites et compromis

- Les sinusoïdes et les tables apprises ajoutent la position au *contenu* ; le modèle doit les démêler. RoPE garde le contenu et la position séparés dans le score.
- RoPE encode la distance relative dans le score mais ne donne pas directement la position absolue ; pour les tâches exigeant un indice absolu, le modèle doit la déduire (par exemple à partir d'un masque causal).
- Le comportement en contexte long au-delà de la longueur d'entraînement dépend de la base de fréquence et de techniques de mise à l'échelle ; la leçon source en liste plusieurs (rééchelonnement de la base, interpolation, pénalité linéaire de distance d'ALiBi $-m_h|i-j|$). Nous ne les avons pas implémentées ici.
- La rotation coûte $O(Td_h)$ par tête, négligeable devant l'attention.

## Idées reçues courantes

- **« L'encodage de position est aussi ajouté à $V$. »** Les sinusoïdes sont ajoutées aux embeddings (donc elles atteignent $Q,K,V$), mais RoPE ne fait tourner que $Q$ et $K$.
- **« RoPE encode des positions absolues. »** Il utilise des indices absolus pour construire les rotations, mais le score ne dépend que de la différence.
- **« Sinus et cosinus sont redondants. »** La paire $(\sin,\cos)$ est un point sur un cercle ; elle fait du décalage de $k$ une rotation (linéaire) et garde la paire à norme constante.
- **« La rotation change la taille du vecteur. »** Non : $\|R_mq\|=\|q\|$.

## Points clés à retenir

- L'attention seule est équivariante par permutation : $Z(PX)=PZ(X)$.
- Sinusoïdal : $PE_{p,2i}=\sin(p\omega_i)$, $PE_{p,2i+1}=\cos(p\omega_i)$, forme $(T,d)$, ajouté aux embeddings ; un décalage est une rotation paire par paire.
- Appris : une table entraînable $(T_{\max},d)$, aucune valeur au-delà de $T_{\max}$.
- RoPE : faire tourner chaque paire de coordonnées de $Q,K$ de $m\theta_i$ ; $\langle R_mq,R_nk\rangle=q^\top R_{n-m}k$, fonction du seul décalage.

## Explique-le toi-même

1. Avec tes propres mots, prouve qu'une couche d'attention pure ne peut pas distinguer « le chien mord l'homme » de « l'homme mord le chien ».
2. Pourquoi les positions absolues $m$ et $n$ disparaissent-elles de $\langle R_mq,R_nk\rangle$ ? Utilise $R(a)^\top R(b)=R(b-a)$.

## Pour aller plus loin

- Bibliothèque source, phase 07, leçon 04 (sinusoïdal, RoPE, ALiBi, méthodes de mise à l'échelle) et ses références (Vaswani 2017 Sec. 3.5 ; Su et al. 2021, RoFormer).
- Raschka, *LLMs from scratch*, ch. 4, Sec. 4.1 pour l'embedding de position appris de GPT-2.

## Critères de maîtrise

Tu sais prouver l'équivariance par permutation, écrire la formule sinusoïdale, démontrer $\langle R_mq,R_nk\rangle=q^\top R_{n-m}k$, et ton implémentation passe les tests de décalage relatif et de conservation de la norme.
