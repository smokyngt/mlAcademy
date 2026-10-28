---
id: adv.03
title: "Mélange d'experts : routage, équilibre et parallélisme d'experts"
summary: "Remplacer le bloc feed-forward par de nombreux experts et un routeur top-k, dériver et implémenter la perte d'équilibrage de charge, et comprendre la capacité, l'abandon de tokens, les experts partagés et le parallélisme d'experts."
minutes: 75
teaches: [adv.moe, adv.moe-balance]
requires: [trf.block, prob.basics, calc.gradient]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md: FFN swap, router, load-balancing options, shared and fine-grained experts, memory catch, when to use MoE. READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/docs/en.md: knob 5 (dense vs MoE), Mixtral and DeepSeek-V3 parameter figures. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/docs/en.md: DeepSeek-V3 config (256 routed + 1 shared, top-8), auxiliary-loss-free bias balancing, 3 dense + 58 MoE layers. READ."
---

## Où cela s'inscrit

Dans un bloc de transformer ([[trf.04]]), le réseau feed-forward (FFN) contient environ les deux tiers des paramètres, et chaque token les paie tous. Faire croître la qualité en agrandissant un modèle dense fait donc croître le coût par token. Un **mélange d'experts (MoE)** rompt ce lien : il garde de nombreux FFN (les « experts ») mais n'envoie chaque token qu'à travers quelques-uns. Le nombre total de paramètres, qui stocke la connaissance, croît avec le nombre d'experts $E$ ; le calcul par token ne croît qu'avec le nombre $k$ choisi. La plupart des modèles ouverts de pointe sont des MoE (DeepSeek-V3 : 671 Md de paramètres au total, environ 37 Md actifs par token). Le prix est un problème d'entraînement plus difficile (les experts doivent être utilisés de façon homogène), et un problème de systèmes (tous les experts doivent résider quelque part et les tokens doivent voyager jusqu'à eux). Cette leçon dérive la perte d'équilibrage qui traite le premier et explique le second. Le coût de garder les paramètres résidents est le versant mémoire de [[sys.02]] ; le partitionnement (sharding) à l'entraînement vient dans [[adv.06]].

## L'idée en mots simples

Un hôpital n'envoie pas chaque patient chez chaque médecin. Un réceptionniste (le **routeur**) jette un coup d'œil à chaque patient et l'envoie aux deux spécialistes les plus pertinents (les **experts**), puis combine leurs avis, en pondérant peut-être davantage le plus pertinent. L'hôpital peut employer des centaines de spécialistes, mais chaque patient ne coûte que deux consultations.

Trois choses peuvent mal tourner. Si le réceptionniste envoie tout le monde chez le même médecin vedette, ce médecin est submergé et les autres n'apprennent jamais rien (**déséquilibre de charge**, qui fait aussi s'effondrer l'entraînement vers un seul expert). Si chaque médecin a un nombre fixe de créneaux par jour, les patients en excès restent sans soins (**abandon de tokens**). Et un hôpital réparti sur de nombreux bâtiments oblige les patients à marcher entre les bâtiments (**parallélisme d'experts** et sa communication all-to-all). Quelques généralistes voient tout le monde (**experts partagés**) pour que la connaissance commune n'ait pas à être dupliquée dans chaque spécialiste.

## La couche MoE

Entrée : $N$ vecteurs de tokens $x_t\in\mathbb{R}^d$. Les poids du routeur $W_r\in\mathbb{R}^{d\times E}$ donnent des logits $z_t=x_tW_r\in\mathbb{R}^E$ et des probabilités $p_t=\operatorname{softmax}(z_t)$. Soit $S_t$ l'ensemble des indices des $k$ plus grands scores. La sortie de la couche est

$$
y_t=\sum_{e\in S_t} g_{t,e}\;\mathrm{Expert}_e(x_t),\qquad g_{t,e}=\frac{\exp z_{t,e}}{\sum_{e'\in S_t}\exp z_{t,e'}} .
$$

Les portes (gates) sont un softmax sur les seuls logits sélectionnés, donc elles somment à un. Chaque expert est un FFN ordinaire (typiquement SwiGLU avec trois matrices $d\times d_{ff}$). Avec des **experts partagés**, quelques FFN supplémentaires traitent chaque token et leurs sorties sont simplement ajoutées.

Comptage des paramètres (feed-forward uniquement, SwiGLU) : total $=(E+s)\cdot 3 d\, d_{ff}$, actifs $=(k+s)\cdot 3 d\, d_{ff}$, où $s$ est le nombre d'experts partagés. Les conceptions à grain fin (DeepSeek, Qwen-MoE) utilisent de nombreux petits experts, par exemple 256 experts routés, top 8, plus 1 partagé : le nombre de sous-ensembles d'experts possibles par token, $\binom{256}{8}\approx 4.1\times10^{14}$, écrase les 28 d'une disposition 2 parmi 8, à coût actif similaire. Dans DeepSeek-V3, les trois premières couches sont denses et les 58 autres sont des MoE.

## Le problème d'équilibrage de charge et sa perte

Laissé à lui-même, un routeur tend à l'**effet Matthieu** (les riches s'enrichissent) : un expert légèrement meilleur reçoit plus de tokens, s'entraîne davantage, devient encore meilleur. Le remède classique ajoute une perte auxiliaire. Définis, sur un batch de $N$ tokens avec $k$ choix chacun,

- $f_i=\dfrac{1}{Nk}\#\{(t,j): \text{token } t\text{'s } j\text{th choice is expert } i\}$, la **fraction d'affectations** vers l'expert $i$ (dure, issue du top-$k$),
- $P_i=\dfrac1N\sum_t p_{t,i}$, la **probabilité moyenne du routeur** pour l'expert $i$ (douce).

$$
\mathcal{L}_{\text{bal}}=\alpha\,E\sum_{i=1}^E f_iP_i,\qquad \alpha\approx10^{-2}\text{ to }10^{-3}.
$$

**Pourquoi cette forme (dérivation).**

1. $f$ et $P$ sont tous deux des vecteurs de probabilité : $\sum_if_i=\sum_iP_i=1$.
2. Suppose que le routage soit cohérent, $f_i\approx P_i$. Alors $\mathcal{L}_{\text{bal}}=\alpha E\sum_iP_i^2$. Par Cauchy-Schwarz, $1=(\sum_i P_i)^2\le E\sum_iP_i^2$, donc $\mathcal{L}_{\text{bal}}\ge\alpha$, avec égalité exactement quand $P_i=1/E$ pour tout $i$ (uniforme). Si tout s'effondre sur un seul expert, $\sum P_i^2=1$ et $\mathcal{L}_{\text{bal}}=\alpha E$. Le facteur $E$ est choisi pour que la valeur parfaitement équilibrée soit $\alpha$ quel que soit $E$.
3. Pourquoi à la fois $f$ et $P$ ? $f$ est ce qui compte vraiment (la charge) mais est constant par morceaux par rapport aux poids du routeur (l'argmax a un gradient nul). $P$ est dérivable. Le produit donne un substitut dérivable dont le gradient est pondéré par la vraie charge.

**Gradient.** Traite $f$ comme une constante. Comme $\partial\mathcal{L}/\partial P_i=\alpha E f_i$ et $P_i=\frac1N\sum_tp_{t,i}$, avec $\partial p_{t,i}/\partial z_{t,j}=p_{t,i}(\delta_{ij}-p_{t,j})$ (la jacobienne du softmax),

$$
\frac{\partial\mathcal{L}_{\text{bal}}}{\partial z_{t,j}}=\frac{\alpha E}{N}\sum_i f_i\,p_{t,i}(\delta_{ij}-p_{t,j})=\frac{\alpha E}{N}\,p_{t,j}\Big(f_j-\sum_if_ip_{t,i}\Big).
$$

Lecture : le logit de l'expert $j$ est poussé vers le **bas** (la descente de gradient soustrait le gradient) quand $f_j$ dépasse la charge moyenne pondérée par les probabilités du token $\sum_if_ip_{t,i}$, et poussé vers le haut quand l'expert $j$ est en dessous. Les experts surchargés deviennent moins attractifs, proportionnellement à leur surcharge. Cette formule a été vérifiée contre des différences finies pour l'exemple détaillé ci-dessous.

**Alternative : équilibrage sans perte auxiliaire (DeepSeek-V3).** Ajouter un biais par expert $b_e$ aux scores utilisés uniquement pour la **sélection**, $S_t=\operatorname{top-}k(z_t+b)$, tandis que les portes utilisent toujours les $z$ bruts. Après chaque pas, poser $b_e\leftarrow b_e-\gamma\,\operatorname{sign}(c_e-\bar c)$ où $c_e$ est le nombre de tokens de l'expert : les experts surchargés reçoivent un biais plus bas, les sous-chargés un biais plus haut. Aucun terme de perte supplémentaire ne rivalise avec l'objectif de modélisation du langage, et aucun $\alpha$ à régler ; l'équilibre est piloté hors du gradient.

## Capacité, abandon et parallélisme

Sur les accélérateurs, le batch de chaque expert est un tampon de taille fixe. Définis la **capacité**

$$
C=\Big\lceil \text{cf}\cdot\frac{N k}{E}\Big\rceil ,
$$

où $Nk/E$ est la charge parfaitement équilibrée et le **facteur de capacité** cf (1.0 à 2.0) est une marge. Les tokens sont traités dans l'ordre ; une affectation arrivant sur un expert plein est **abandonnée** : ce token saute l'expert et ne passe que par la connexion résiduelle. Un cf plus grand abandonne moins de tokens mais gaspille de la mémoire tampon et du calcul sur du remplissage ; un cf plus petit est moins coûteux et nuit à la qualité. De nombreux systèmes modernes évitent l'abandon en utilisant des noyaux sans abandon (multiplications matricielles groupées de taille variable) plus l'équilibrage.

**Parallélisme d'experts.** Avec $G$ GPU, chacun détient $E/G$ experts. Par couche MoE : un **all-to-all** envoie le vecteur de chaque token aux GPU qui détiennent ses experts choisis, les experts calculent, et un second all-to-all renvoie les sorties. Chaque token déplace environ $2k\,d\,b$ octets à travers le réseau par couche ($k$ destinations, aller et retour). À grande échelle, cette communication, et non les multiplications matricielles, domine la latence, c'est pourquoi le DualPipe de DeepSeek (voir [[adv.06]]) la recouvre avec le calcul. De plus, **tous les experts restent en mémoire** qu'ils s'activent ou non : un modèle de 671 Md exige environ 1,3 To pour des poids en 16 bits.

## Exemple détaillé

Six tokens, $E=3$ experts, top-1, logits

| token | $z_0$ | $z_1$ | $z_2$ |
|---|---|---|---|
| 0 | 2.0 | 0 | 0 |
| 1 | 1.5 | 0 | 0.5 |
| 2 | 2.0 | 1 | 0 |
| 3 | 0 | 1 | 0 |
| 4 | 0 | 0.2 | 1.0 |
| 5 | 1.0 | 0 | 0 |

- Routage : les tokens 0, 1, 2, 5 vont à l'expert 0 ; le token 3 à l'expert 1 ; le token 4 à l'expert 2. Donc $f=(4/6,1/6,1/6)=(0.667,0.167,0.167)$.
- Probabilités moyennes (lignes de softmax moyennées) : $P=(0.512,0.255,0.234)$.
- $\mathcal{L}_{\text{bal}}/\alpha=E\sum f_iP_i=3\,(0.667\cdot0.512+0.167\cdot0.255+0.167\cdot0.234)=1.268$, au-dessus de la valeur équilibrée 1.
- Capacité avec cf $=1.0$ : $C=\lceil6\cdot1/3\rceil=2$. L'expert 0 reçoit quatre tokens dans l'ordre 0, 1, 2, 5 : il garde 0 et 1, **abandonne 2 et 5**. Avec cf $=1.25$ : $C=3$, seul le token 5 est abandonné.
- Gradient sur le token 0 : $\frac{\alpha E}{N}p_{0,j}(f_j-\sum_if_ip_{0,i})$ est positif pour l'expert 0 et négatif pour les experts 1 et 2, donc la descente abaisse $z_{0,0}$ et relève les autres pour ce token.

## Implémentation

```python
import numpy as np

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def topk_route(logits, k, bias=None):
    scores = logits if bias is None else logits + bias
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return idx, softmax(np.take_along_axis(logits, idx, axis=1))

def balance_loss(logits, idx, alpha=1.0):
    N, E = logits.shape
    f = np.bincount(idx.ravel(), minlength=E) / idx.size
    return alpha * E * np.sum(f * softmax(logits).mean(0))
```

Production : les classes Mixtral et DeepSeek de `transformers` de Hugging Face, avec vLLM et SGLang pour l'inférence à experts parallèles.

## Formes

| Tenseur | Forme |
|---|---|
| tokens $x$ | (N, d) |
| poids du routeur | (d, E) |
| logits, probabilités | (N, E) |
| indices sélectionnés, portes | (N, k) |
| poids des experts (jouet linéaire) | (E, d, d) ; expert SwiGLU : trois (d, d_ff) |
| tampon par expert | (C, d) avec $C=\lceil\text{cf}\,Nk/E\rceil$ |

## Idées reçues

- **« Les paramètres actifs sont tout ce qui compte pour la mémoire. »** La mémoire contient les $E$ experts ; seul le calcul est creux.
- **« La perte d'équilibrage rend les experts également bons. »** Elle égalise la charge ; la spécialisation s'apprend séparément.
- **« Les tokens abandonnés sont perdus. »** Ils traversent toujours le flux résiduel et l'attention, mais cette couche ne contribue en rien pour eux.
- **« Un $\alpha$ plus grand est toujours plus sûr. »** Une pénalité trop forte impose un routage uniforme contre ce que veulent les données et nuit à la qualité ; cette tension a motivé l'équilibrage par biais.
- **« Le MoE bat toujours le dense. »** À petite échelle (quelques milliards de paramètres actifs), en périphérie, ou pour un serving mono-utilisateur limité par la latence, le dense est souvent meilleur.

## Points clés à retenir

- Le routeur choisit $k$ experts parmi $E$ par token ; les portes sont un softmax sur les logits sélectionnés ; les paramètres totaux croissent avec $E$, le calcul avec $k$.
- $\mathcal{L}_{\text{bal}}=\alpha E\sum f_iP_i$ est minimisée ($=\alpha$) à charge uniforme ; son gradient pousse vers le bas les experts dont la charge dépasse la moyenne pondérée par le token.
- La capacité $\lceil\text{cf}\,Nk/E\rceil$ borne le travail d'un expert au prix de tokens abandonnés ; l'équilibrage par biais évite la perte auxiliaire.
- Le parallélisme d'experts coûte deux all-to-all par couche ; les experts partagés détiennent la connaissance commune.

## Explique-le toi-même

1. Dérive pourquoi $\alpha E\sum f_iP_i\ge\alpha$ quand $f=P$, et quand l'égalité a lieu.
2. Une équipe double le nombre d'experts mais garde $k$. Dis ce qui arrive aux paramètres totaux, aux paramètres actifs et à la mémoire par GPU sur 8 GPU.

## Pour aller plus loin

- Switch Transformer (Fedus et al.), rapports de Mixtral, DeepSeek-V2/V3, tels que résumés dans les leçons sources.
- [[adv.06]] pour le partitionnement et les pipelines, [[adv.04]] pour le versant attention de DeepSeek-V3 (MLA).

## Critères de maîtrise

Tu sais implémenter un routage top-$k$ avec un biais pour la sélection et des portes sur logits bruts, calculer la perte d'équilibrage et expliquer son gradient, dimensionner la capacité d'un expert et compter les tokens abandonnés, et calculer les paramètres totaux contre actifs en incluant les experts partagés.
