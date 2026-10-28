---
id: sys.02
title: "Quantification, décodage spéculatif et mélange d'experts"
summary: "Trois façons de rendre les gros modèles moins chers à servir - moins de bits par poids, plusieurs tokens par passe coûteuse, et n'activer qu'une partie du réseau - avec les mathématiques de l'échelle et du point zéro, la règle d'acceptation/rejet qui garde exacte la distribution de sortie, et la formule d'accélération."
minutes: 90
teaches: [llm.quantization, llm.speculative]
requires: [llm.kv-cache, prob.rv, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/11-quantization/docs/en.md (Number Formats, How Quantization Works, Sensitivity Hierarchy, Quality Measurement, Real Numbers): formats, scale formulas, per-channel, asymmetric, sensitivity ranking, PTQ vs QAT. READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/16-speculative-decoding/docs/en.md: draft/verify algorithm, residual distribution, speed-up rule of thumb. READ (Problem, Core algorithm, What determines speedup)."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/16-speculative-decoding/code/main.py: accept test min(1,q/p), residual, chi-square check of equivalence, expected_tokens_per_verify. READ (first half)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/12-inference-optimization/docs/en.md (Prefill vs Decode): why decode is memory-bound. READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md (Problem, FFN swap, load balancing, shared and fine-grained experts): router, top-k, total vs active parameters. READ."
---

## Où cela s'inscrit

Un modèle de 70 milliards de paramètres stocké en nombres de 16 bits occupe 140 Go. Aucune carte grand public ne les contient, et même sur un GPU de centre de données, le coût de génération d'un token est dominé par la lecture de ces 140 Go depuis la mémoire. La leçon précédente a montré que le decode est limité par la mémoire. Cette leçon donne trois familles de réponses, chacune s'attaquant à un terme différent :

1. **Quantification** : stocker les nombres sur 8 ou 4 bits. Moins d'octets à lire, moins d'octets à stocker.
2. **Décodage spéculatif** : laisser un petit modèle deviner plusieurs tokens, et laisser le gros modèle vérifier toutes les suppositions en une seule passe. Moins de *passes* du gros modèle.
3. **Mélange d'experts (MoE)** : construire le réseau de façon que chaque token n'utilise qu'une fraction de ses paramètres. Moins de *travail par token*.

La quantification modifie légèrement les poids, il faut donc mesurer les dégâts. Le décodage spéculatif ressemble à une approximation mais, avec la bonne règle, produit exactement la même distribution de sorties que le gros modèle. Ce « exactement » est la partie délicate, et nous la démontrons.

## L'idée en mots simples

**Quantification.** Une règle graduée en millimètres mesure plus finement qu'une règle graduée en centimètres, mais exige dix fois plus de graduations. La quantification choisit une règle avec moins de graduations (256 pour 8 bits) et la place soigneusement sur la plage où vivent les nombres. Chaque nombre est arrondi à la graduation la plus proche ; l'erreur vaut au plus la moitié d'un intervalle. Les seules décisions sont *où placer la règle* (la plage) et *combien de règles* utiliser (une pour toute la matrice, ou une par ligne). Si un seul nombre énorme étire la règle, les graduations deviennent grossières pour tous les autres : c'est le problème des **valeurs aberrantes** (outliers).

**Décodage spéculatif.** Un rédacteur en chef (le gros modèle) est lent ; un stagiaire (le petit modèle) est rapide mais parfois dans l'erreur. Le stagiaire rédige les cinq mots suivants. Le rédacteur les lit tous d'un coup, ce qui coûte à peu près autant qu'écrire un mot car lire en parallèle est bon marché, et approuve les mots de gauche à droite jusqu'au premier qui lui déplaît, en écrivant un remplaçant à cet endroit. Si le stagiaire a généralement raison, l'équipe produit plusieurs mots par tour de rédacteur. La surprise est que, avec une règle d'approbation soigneusement choisie, le texte final est statistiquement *exactement* ce que le rédacteur aurait écrit seul.

**Mélange d'experts.** Un hôpital n'envoie pas chaque patient chez chaque spécialiste. Un réceptionniste (le routeur) envoie chaque patient vers un ou deux spécialistes pertinents. L'hôpital emploie beaucoup de spécialistes (beaucoup de paramètres), mais chaque patient n'en utilise que quelques-uns (peu de calcul).

## Les formats de nombres en un tableau

| Format | Bits | Disposition (signe/exposant/mantisse) | Remarque de la source |
|---|---|---|---|
| FP32 | 32 | 1/8/23 | accumulation, entraînement classique |
| FP16 | 16 | 1/5/10 | petite plage ; mise à l'échelle de la perte nécessaire à l'entraînement |
| BF16 | 16 | 1/8/7 | plage de FP32, moins de précision ; format d'entraînement courant |
| FP8 | 8 | E4M3 ou E5M2 | inférence (E4M3), gradients (E5M2) |
| INT8 | 8 | 256 niveaux régulièrement espacés + une échelle | l'arithmétique entière est bon marché |
| INT4 | 4 | 16 niveaux + une échelle | exige des méthodes soignées |

Les formats à virgule flottante dépensent des bits sur un exposant pour que l'espacement entre nombres représentables croisse avec la magnitude. Les formats entiers ont un espacement uniforme et reposent sur une **échelle** (scale) qui renvoie les entiers vers des valeurs réelles.

## Quantification : définition formelle

Choisis une largeur de bits $b$, une **échelle** réelle $s>0$ et un **point zéro** entier $z$. Quantifier et déquantifier :

$$
q = \operatorname{clip}\big(\operatorname{round}(x/s) + z,\; q_{\min},\; q_{\max}\big),\qquad \hat{x} = s\,(q - z).
$$

Schéma **symétrique** : $z=0$, entiers dans $[-q_{\max}, q_{\max}]$ avec $q_{\max}=2^{b-1}-1$ (127 pour int8). Choisis $s$ pour que la plus grande magnitude tombe sur le plus grand entier :

$$
s = \frac{\max_i |x_i|}{q_{\max}} .
$$

Schéma **asymétrique** : entiers non signés $0\ldots 2^b-1$ et une plage $[\ell, u]$ qui doit contenir le réel 0 (pour que zéro, qui est partout - padding, sorties de ReLU - soit représenté exactement). Dérivation : on veut une application affine $x\mapsto q$ avec $\ell\mapsto 0$ et $u\mapsto 2^b-1$. La pente est $1/s$ avec

$$
s=\frac{u-\ell}{2^b-1},\qquad q=\frac{x-\ell}{s}=\frac{x}{s}+\underbrace{\Big(-\frac{\ell}{s}\Big)}_{z},
$$

et comme $z$ doit être un entier, $z=\operatorname{round}(-\ell/s)$. Alors $x=0$ donne $q=z$ et se déquantifie en $s(z-z)=0$ exactement.

## Dérivation : l'erreur d'arrondi

Arrondir à l'entier le plus proche déplace $x/s$ d'au plus $1/2$, donc $|\hat x - x|\le s/2$ pour chaque élément non écrêté. Si l'erreur d'arrondi est répartie à peu près uniformément sur $[-s/2, s/2]$, sa variance est

$$
\mathbb{E}\big[(\hat x - x)^2\big]=\frac1s\int_{-s/2}^{s/2} e^2\,de=\frac{s^2}{12}.
$$

Deux conséquences. L'erreur dépend de l'*échelle*, pas des valeurs elles-mêmes : une plage large (grand $\max|x|$) signifie un grand $s$ et une erreur plus grande pour chaque élément. Et chaque bit retiré double $s$, donc la MSE est multipliée par 4 par bit.

**Par tenseur contre par canal.** Une seule échelle pour toute une matrice de poids est dictée par le plus grand poids. Avec une échelle par ligne de sortie (par canal), chaque ligne a sa propre règle, et une valeur aberrante ne nuit qu'à sa propre ligne. Le prix est de stocker une échelle par ligne.

## Exemple détaillé

Tous les nombres proviennent du code de l'exercice.

**Int8 symétrique.** $x=[-1,\,-0.3,\,0,\,0.4,\,2.54]$. $\max|x|=2.54$, donc $s=2.54/127=0.02$ et $q=[-50,-15,0,20,127]$. Chaque erreur de reconstruction vaut au plus $s/2=0.01$ et vaut ici exactement 0.

**Int8 asymétrique.** $y=[-1,\,0,\,0.5,\,3]$ : plage $[-1,3]$, $s=4/255\approx0.01569$, $z=\operatorname{round}(63.75)=64$, $q=[0,64,96,255]$. Le réel 0 est envoyé sur $q=64$ et revient exactement à $0$.

**Données unilatérales (comme une sortie de ReLU).** $[0,0.5,1,2,6]$. Le schéma symétrique dépense la moitié de ses entiers sur des valeurs négatives qui n'arrivent jamais : $s=6/127=0.0472$, MSE $1.40\times10^{-4}$. L'asymétrique utilise les 256 niveaux : $s=6/255=0.0235$, MSE $3.46\times10^{-5}$, quatre fois plus petite.

**Valeurs aberrantes.** Une matrice gaussienne $64\times32$ d'écart-type 0.05, dont la ligne 5 est multipliée par 40. L'échelle par tenseur $s=0.0306$ est fixée par la ligne aberrante : MSE $7.74\times10^{-5}$. Échelles par canal : MSE $1.19\times10^{-6}$, une amélioration d'un facteur $65\times$. Sur les 63 *autres* lignes, l'erreur passe de $7.7\times10^{-5}$ à $7.2\times10^{-8}$ : la ligne aberrante a ruiné toutes les autres lignes sous l'échelle par tenseur.

**Balayage du nombre de bits** (2000 valeurs normales centrées réduites) : MSE $5.5\times10^{-5}$ à 8 bits, $9.1\times10^{-4}$ à 6, $1.9\times10^{-2}$ à 4, $0.73$ à 2. La formule $s^2/12$ prédit $5.5\times10^{-5}$, $9.2\times10^{-4}$, $1.8\times10^{-2}$ et $0.88$. À 2 bits, l'écrêtage et les niveaux grossiers invalident l'hypothèse d'erreur uniforme.

**Mémoire :** 70 milliards de paramètres occupent 140 Go en 16 bits, 70 Go en 8 bits et 35 Go en 4 bits (en ignorant le stockage des échelles et les couches gardées en plus haute précision).

## Ce qu'on peut quantifier sans risque

Selon la hiérarchie de sensibilité du matériel sur la quantification : les **poids** (à peu près gaussiens autour de zéro) sont les plus robustes ; les **activations** ont des valeurs aberrantes (un seul canal peut être 100 fois plus grand que la moyenne) et demandent des échelles par canal/par token ou de garder les canaux aberrants en plus haute précision ; le **cache KV** est plus sensible car les erreurs s'accumulent sur tous les tokens suivants ; les **logits d'attention** avant le softmax sont les plus sensibles et sont généralement gardés en 16 bits. La *quantification post-entraînement* (PTQ) arrondit un modèle existant, rapide et peu coûteuse ; l'*entraînement conscient de la quantification* (QAT) entraîne avec l'arrondi dans la boucle. Des méthodes comme GPTQ et AWQ utilisent des données de calibration pour choisir des arrondis qui minimisent l'erreur de sortie ; elles sont mentionnées ici, non dérivées. Le tableau de chiffres de la source pour un modèle 70B (changements de perplexité et de benchmarks par format) provient de ce matériel et doit être lu comme indicatif, car les résultats diffèrent selon les modèles, les outils et les benchmarks. Mesure toujours la perplexité et la métrique de ta propre tâche avant et après.

## Décodage spéculatif

### L'algorithme

Soit $q(\cdot\mid \text{ctx})$ le modèle cible (gros, lent) et $p(\cdot\mid\text{ctx})$ le brouillon (petit, rapide). Un tour avec $N$ brouillons :

1. Tire les brouillons $d_1,\dots,d_N$ de façon autorégressive depuis $p$.
2. Exécute $q$ **une seule fois** sur le contexte plus tous les brouillons ; cela donne les distributions de $q$ aux $N+1$ positions simultanément (c'est une passe parallèle de type prefill).
3. Pour $i=1,2,\dots$ : accepte $d_i$ avec la probabilité $\min\!\big(1,\ q_i(d_i)/p_i(d_i)\big)$.
4. Au premier rejet, écarte les brouillons restants et tire le token de remplacement dans la distribution **résiduelle**
$$
r(x)=\frac{\max\big(0,\;q(x)-p(x)\big)}{\sum_y \max\big(0,\;q(y)-p(y)\big)} .
$$
5. Si les $N$ brouillons ont tous été acceptés, tire un token bonus dans $q_{N+1}$.

### Pourquoi la sortie a exactement la distribution $q$

Regarde une seule position. La probabilité que le token final soit égal à $x$ est

$$
\Pr[x]=\underbrace{p(x)\min\!\Big(1,\tfrac{q(x)}{p(x)}\Big)}_{\text{drafted as }x\text{ and accepted}}+\underbrace{\Pr[\text{reject}]\; r(x)}_{\text{replacement}} .
$$

Le premier terme vaut $\min(p(x),q(x))$. La probabilité de rejet est $1-\sum_y\min(p(y),q(y))=:1-\alpha$, où $\alpha=\sum_y\min(p(y),q(y))$ est le taux d'acceptation. Comme $q-\min(p,q)=\max(0,q-p)$, le normalisateur du résidu est $\sum_y\max(0,q(y)-p(y))=1-\alpha$. Le second terme vaut donc $(1-\alpha)\cdot\frac{\max(0,q(x)-p(x))}{1-\alpha}=\max(0,q(x)-p(x))$, et

$$
\Pr[x]=\min(p(x),q(x))+\max(0,q(x)-p(x))=q(x).
$$

En appliquant cela position après position (chaque position conditionnée par le préfixe accepté), on montre que toute la séquence générée suit $q$. Le brouillon n'influence que la *vitesse*, jamais la distribution. Si $p$ est une mauvaise supposition, $\alpha$ est petit et tu gaspilles du travail, mais le texte reste distribué comme si la cible l'avait écrit seule.

### Accélération attendue

Suppose que chaque token brouillon soit accepté indépendamment avec la même probabilité $\alpha$ (hypothèse de modélisation : dans les vrais modèles, $\alpha$ varie avec le contexte). Un tour produit le préfixe accepté plus exactement un token de plus (la correction, ou le bonus). Le nombre attendu de tokens est

$$
\mathbb{E}[\text{tokens}]=1+\alpha+\alpha^2+\dots+\alpha^N=\frac{1-\alpha^{N+1}}{1-\alpha}.
$$

Soit un pas de brouillon coûtant une fraction $c$ d'une passe cible (par exemple $c=0.1$ si le brouillon est dix fois moins cher). Un tour coûte $Nc+1$ équivalents de passe cible, donc

$$
\text{speed-up}\;\approx\;\frac{1-\alpha^{N+1}}{(1-\alpha)(Nc+1)} .
$$

Pour $\alpha=0.8$, $N=4$ : $3.36$ tokens par tour ; avec $c=0.1$, le coût est $1.4$, ce qui donne $2.40\times$. Si le brouillon est coûteux ($c=0.5$), on tombe à $1.12\times$ ; si le brouillon est mauvais ($\alpha=0.5$, $c=0.1$), on obtient $1.38\times$. Choisir $N$ est un équilibre : plus de brouillons augmentent le numérateur avec des rendements décroissants ($\alpha^{N+1}$) tandis que le coût croît linéairement. La source mentionne des accélérations pratiques typiques de 2 à 4 fois et des variantes (têtes Medusa, EAGLE, lookahead) qui évitent un modèle brouillon séparé ; ces chiffres viennent du matériel du cours et dépendent du matériel et de la charge de travail.

### Exemple détaillé sur une position

Cible $q=[0.5,0.25,0.15,0.07,0.03]$, brouillon $p=[0.3,0.3,0.2,0.1,0.1]$. Alors $\min(p,q)=[0.3,0.25,0.15,0.07,0.03]$, donc $\alpha=0.8$. Le numérateur du résidu est $[0.2,0,0,0,0]$, normalisé en $[1,0,0,0,0]$. Probabilités de sortie : le token 0 obtient $0.3+0.2\cdot1=0.5$, le token 1 obtient $0.25$, ... exactement $q$. L'exercice vérifie cela empiriquement avec un test du khi-deux sur un modèle bigramme à deux tokens : l'échantillonneur spéculatif réussit, et l'échantillonnage depuis le seul brouillon échoue largement.

## Le mélange d'experts en bref

Un transformer dense applique son réseau feed-forward (FFN) à chaque token, donc le calcul par token est proportionnel au nombre de paramètres. Dans une couche MoE, le FFN est remplacé par $E$ FFN experts indépendants et un **routeur**, une couche linéaire $x\mapsto \text{scores}\in\mathbb{R}^E$. Pour chaque token, prends les $k$ meilleurs scores, applique un softmax dessus, et produis la somme pondérée des sorties des $k$ experts sélectionnés :

$$
y=\sum_{e\in\text{top-}k}g_e(x)\,\text{Expert}_e(x).
$$

Le nombre total de paramètres croît avec $E$ ; le calcul par token croît avec $k$. La source donne des exemples comme 671B paramètres au total avec 37B actifs par token, et décrit des réglages modernes typiques de nombreux petits experts avec un $k$ plus grand plus quelques experts partagés toujours actifs ; ces chiffres sont cités du matériel, non vérifiés ici. La difficulté centrale est l'**équilibrage de charge** : si le routeur envoie la plupart des tokens vers quelques experts, les autres n'apprennent jamais et les plus sollicités deviennent un goulot d'étranglement. Remèdes dans les sources : une perte auxiliaire sur la variance d'utilisation, des limites de capacité qui abandonnent les tokens en excès, ou un biais par expert qui infléchit la sélection sans toucher à la perte d'entraînement. Le MoE économise du calcul mais pas de mémoire : tous les experts doivent quand même être stockés, donc le serving exige beaucoup de mémoire (et de la communication entre appareils).

## Implémentation

```python
def quantize_sym(x, bits=8, axis=None):
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(x), axis=axis, keepdims=axis is not None)
    scale = np.where(amax == 0, 1.0, amax / qmax)
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int32)
    return q, scale

def accept_or_replace(q, p, d, u, u2):
    """d drafted from p; u, u2 uniform. Returns the emitted token."""
    if u < min(1.0, q[d] / p[d]):
        return d
    r = np.maximum(q - p, 0.0)
    return int(np.searchsorted(np.cumsum(r / r.sum()), u2, side="right"))
```

## Formes

| Objet | Forme |
|---|---|
| matrice de poids $W$ | $(d_\text{out}, d_\text{in})$ |
| échelle par tenseur | scalaire |
| échelle par canal (par ligne) | $(d_\text{out}, 1)$ pour qu'elle se diffuse contre $W$ |
| poids quantifiés | même forme que $W$, int8 |
| distributions brouillon/cible à une position | $(V,)$ |
| passe de vérification sur $N$ brouillons | logits cibles $(N+1, V)$ en un seul appel |
| scores du routeur MoE | $(\text{tokens}, E)$ ; indices top-$k$ $(\text{tokens}, k)$ |

## Idées reçues

1. « Le décodage spéculatif échange de la qualité contre de la vitesse. » Avec la règle d'acceptation/rejet ci-dessus, non : la distribution est exactement celle de la cible. (Le non-déterminisme matériel et les différences numériques entre passes batchées et non batchées peuvent toutefois changer des échantillons individuels.)
2. « N'accepter un brouillon que s'il égale l'argmax de la cible revient au même. » C'est une règle différente, valable uniquement en glouton. Pour l'échantillonnage, c'est la règle $\min(1,q/p)$ avec le résidu qui préserve la distribution.
3. « 4 bits signifie 4 fois pire. » L'erreur dépend de l'échelle ; une mise à l'échelle soignée par canal/groupe et la calibration limitent la perte, et un int4 naïf par tenseur peut échouer gravement.
4. « Les modèles MoE sont petits parce que peu de paramètres sont actifs. » Ils sont peu coûteux à exécuter par token mais volumineux à stocker.
5. « Plus de tokens brouillons est toujours mieux. » Au-delà du point où $\alpha^{N+1}$ est petit, des brouillons supplémentaires n'ajoutent que du coût.

## Points clés à retenir

- La quantification envoie les réels vers des entiers avec une échelle (et un point zéro si asymétrique) ; l'erreur d'arrondi vaut au plus $s/2$ et environ $s^2/12$ en MSE.
- Les valeurs aberrantes gonflent l'échelle ; des échelles par canal (ou plus fines) les isolent.
- Zéro doit être exactement représentable ; les plages asymétriques gèrent les données unilatérales.
- Échantillonnage spéculatif : accepter avec $\min(1,q/p)$, sinon échantillonner depuis $(q-p)_+$ normalisé ; la loi de sortie est exactement $q$.
- Tokens attendus par tour $=(1-\alpha^{N+1})/(1-\alpha)$ ; l'accélération divise par le coût du tour $Nc+1$.
- Le MoE découple le nombre total de paramètres du calcul par token, au prix de la mémoire et de l'équilibre du routage.

## Explique-le toi-même

1. Pourquoi un poids aberrant dans une matrice rend-il les autres poids moins précis sous la quantification par tenseur, et qu'est-ce que la quantification par canal change ?
2. Dans le décodage spéculatif, pourquoi le token de remplacement doit-il être échantillonné dans le résidu plutôt que directement dans $q$ ? (Calcule ce qui arrive au token 0 dans l'exemple détaillé si tu échantillonnais dans $q$ lors d'un rejet.)

## Pour aller plus loin

- Leviathan et al., "Fast inference from transformers via speculative decoding" (2023), et Chen et al. (2023) : les sources citent les deux pour la règle d'acceptation/rejet.
- Dettmers et al., LLM.int8() (gestion des valeurs aberrantes), articles GPTQ et AWQ : cités dans le matériel sur la quantification.
- Rapports Switch Transformer, Mixtral et DeepSeek-V3 pour le routage et l'équilibrage MoE.

## Critères de maîtrise

- [ ] Je sais dériver l'échelle et le point zéro pour int8 symétrique et asymétrique, et borner l'erreur.
- [ ] Je sais expliquer pourquoi la quantification par canal aide face aux valeurs aberrantes et le vérifier numériquement.
- [ ] Je sais prouver que la règle acceptation/rejet plus résidu échantillonne depuis $q$.
- [ ] Je sais calculer les tokens attendus par tour et l'accélération pour $\alpha$, $N$, $c$ donnés.
- [ ] Je sais dire ce que le MoE économise (calcul par token) et ce qu'il n'économise pas (mémoire).
