---
id: gen.05
title: "Diffusion latente, conditionnement et fine-tuning"
summary: "Exécuter la diffusion dans un espace latent compressé, la piloter par du texte via la cross-attention, dériver et implémenter le guidage sans classifieur, et voir comment ControlNet et LoRA adaptent un modèle gelé."
minutes: 80
teaches: [gen.latent-diffusion, gen.cfg]
requires: [gen.diffusion, gen.vae, trf.attention]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/07-latent-diffusion-stable-diffusion/docs/en.md: two-stage design (VAE then diffusion on latents), text conditioning by cross-attention, model table, classifier-free guidance formula and typical scales, common pitfalls. READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai/08-controlnet-lora-conditioning/docs/en.md: ControlNet (trainable encoder copy, zero-convolutions), LoRA (low-rank delta, parameter counts, alpha scaling, stacking). READ."
---

## Où cela s'inscrit

[[gen.04]] a fourni un modèle de diffusion qui fonctionne mais qui est coûteux : chacun des ~1000 appels au réseau traite une image en pleine résolution. Les systèmes texte-vers-image de production résolvent cela de trois façons que cette leçon couvre. La **diffusion latente** exécute le processus dans un espace compressé fabriqué par un VAE ([[gen.02]]). Le **conditionnement** injecte un prompt texte, via le mécanisme d'attention de [[trf.01]]. Le **guidage sans classifieur** (*classifier-free guidance*) échange de la diversité contre de la fidélité au prompt avec une formule peu coûteuse. Enfin, **ControlNet** et **LoRA** permettent d'adapter un énorme modèle gelé avec un tout petit nombre de nouveaux paramètres ; LoRA est la même idée que tu retrouveras pour les modèles de langage ([[llm.04]]).

## L'idée en mots simples

*Diffusion latente.* La plupart des pixels d'une photographie portent une texture fine qu'un bon compresseur peut jeter et réinventer plus tard. On entraîne donc d'abord un compresseur (un autoencodeur) qui transforme une grande image en une petite grille de nombres, et inversement. Puis on fait tout le débruitage coûteux sur la petite grille, et on décompresse une seule fois à la fin. C'est comme éditer la structure d'une vignette puis laisser un retoucheur habile ajouter le détail au niveau du pixel.

*Conditionnement.* Un prompt est une liste de vecteurs de mots. À chaque étape de débruitage, chaque position de l'image demande « quels mots sont pertinents pour moi ? » et mélange leur information. Ce jeu de question-réponse est la cross-attention.

*Guidage.* Exécute le débruiteur deux fois : une fois en lui donnant le prompt, une fois sans. La différence entre les deux réponses est la direction qui « signifie le prompt ». Exagère cette direction et l'image suit le prompt plus fortement, au prix de la variété.

*Adaptateurs.* Pour apprendre un nouveau tour à un modèle géant, ne le réécris pas. Fixe un petit module bon marché à côté du modèle gelé : un réseau latéral qui lit un signal supplémentaire (ControlNet), ou une fine correction de bas rang sur certaines matrices de poids (LoRA).

## Diffusion latente

**Étape 1.** Entraîner un autoencodeur avec un encodeur $E$ et un décodeur $D$ (perte de reconstruction, perte perceptuelle, un petit poids KL, souvent une perte antagoniste) de sorte que $z=E(x)$ soit un latent compact et $D(z)\approx x$. Un réglage typique sous-échantillonne d'un facteur 8 dans chaque direction spatiale. Pour une image de $512\times512\times3$, le latent est de $64\times64\times4$ : $786{,}432$ nombres deviennent $16{,}384$, une réduction de taille d'un facteur **48** (le texte source cite environ 16 fois moins de pixels ; le sous-échantillonnage 8x par côté donne $8^2=64$ fois moins de positions spatiales et 4 canaux au lieu de 3, ce qui donne le 48 ci-dessus).

**Étape 2.** Traiter $z_0=E(x_0)$ comme les données et entraîner dessus le DDPM de [[gen.04]] : la perte de prédiction du bruit est *identique*, seul le tenseur est plus petit. Échantillonnage : tirer $z_T\sim\mathcal N(0,I)$, débruiter jusqu'à $z_0$, sortir $x=D(z_0)$.

Comme l'autoencodeur est gelé après l'étape 1, la diffusion n'a besoin de modéliser que la structure de haut niveau ; le décodeur fournit le détail haute fréquence. Le calcul diminue à peu près proportionnellement à la réduction de taille.

## Conditionnement par cross-attention

Un encodeur de texte gelé transforme le prompt en embeddings de tokens $C\in\mathbb R^{n_\text{tok}\times d_c}$. À l'intérieur du réseau de débruitage, les caractéristiques de l'image $F\in\mathbb R^{n_\text{pix}\times d}$ portent attention au texte :

$$
\mathrm{CrossAttn}(F,C)=\mathrm{softmax}\!\Big(\frac{(FW_Q)(CW_K)^\top}{\sqrt{d_k}}\Big)(CW_V),
$$

avec $Q=FW_Q\in\mathbb R^{n_\text{pix}\times d_k}$, $K=CW_K,\ V=CW_V\in\mathbb R^{n_\text{tok}\times d_k}$. C'est l'attention de [[trf.01]], sauf que les requêtes viennent de l'image et que les clés et valeurs viennent du texte. Chaque position de pixel obtient une combinaison convexe de valeurs textuelles, donc le texte n'influence l'image que par ces couches (dans les transformers de diffusion plus récents, les tokens de texte et d'image sont traités conjointement).

## Guidage sans classifieur : dérivation

Nous voudrions des échantillons non seulement plausibles mais *fortement cohérents avec* une condition $c$. La règle de Bayes sur la densité conditionnelle donne $p(x\mid c)\propto p(x)\,p(c\mid x)$, d'où, pour les scores (gradients de la log-densité)

$$
\nabla_x\log p(x\mid c)=\nabla_x\log p(x)+\nabla_x\log p(c\mid x).
$$

Le second terme est le gradient d'un classifieur. Aiguisons le classifieur en l'élevant à une puissance $w$ : $p_w(x\mid c)\propto p(x)\,p(c\mid x)^{w}$, de score

$$
\nabla_x\log p_w(x\mid c)=\nabla_x\log p(x)+w\,\nabla_x\log p(c\mid x)
=\nabla_x\log p(x)+w\big(\nabla_x\log p(x\mid c)-\nabla_x\log p(x)\big),
$$

où la seconde égalité utilise de nouveau l'identité de Bayes. Aucun classifieur n'est nécessaire : les deux scores sont exactement ce que fournissent un débruiteur conditionnel et un débruiteur inconditionnel. Comme un prédicteur de bruit est un score remis à l'échelle ($\varepsilon=-\sqrt{1-\bar\alpha_t}\,\nabla\log p_t$, [[gen.04]]), multiplier par $-\sqrt{1-\bar\alpha_t}$ donne la règle du **guidage sans classifieur**

$$
\boxed{\tilde\varepsilon=\varepsilon_\theta(x_t,\varnothing)+w\big(\varepsilon_\theta(x_t,c)-\varepsilon_\theta(x_t,\varnothing)\big)}
$$

Ici $\varnothing$ est la condition « vide ». Un seul réseau est entraîné pour les deux : avec une probabilité d'environ 10 %, la condition est supprimée pendant l'entraînement. $w=0$ ignore le prompt, $w=1$ est le modèle conditionnel simple, $w>1$ extrapole au-delà. La leçon source écrit la même règle $(1+w')\varepsilon_c-w'\varepsilon_u$, c'est-à-dire que son échelle $w'$ vaut $w-1$ ici ; vérifie toujours quelle convention une bibliothèque utilise (les bibliothèques de diffusion appellent généralement notre $w$ l'« échelle de guidage »). Les valeurs typiques sont d'environ 3-8 dans ces conventions ; des valeurs beaucoup plus grandes donnent des images saturées, trop nettes et peu diversifiées. Le coût : deux évaluations du réseau par étape.

## Exemple résolu : le guidage sur un mélange 1-D à deux classes

Données : deux classes d'a priori égaux, de moyennes $-2$ et $+2$, d'écart-type commun $0.5$. Le prédicteur de bruit conditionnel (le débruiteur gaussien de [[gen.04]]) et l'inconditionnel (un mélange pondéré par les responsabilités des deux moyennes a posteriori) sont tous deux exacts. On échantillonne 20 000 points avec $T=500$ étapes, condition $c=$ « classe +2 », graine 0 :

| $w$ | moyenne des échantillons | écart-type des échantillons |
|---|---|---|
| 0 (inconditionnel) | $-0.026$ | $2.058$ (les deux modes) |
| 1 (conditionnel) | $1.992$ | $0.503$ |
| 2 | $2.078$ | $0.486$ |
| 4 | $2.138$ | $0.483$ |
| 8 | $2.190$ | $0.483$ |

Ainsi $w=1$ retrouve la classe (moyenne 2, écart-type 0.5) ; $w>1$ pousse les échantillons au-delà de la moyenne de la classe, vers l'extrême de la direction de la classe, loin de l'autre classe. Dans de vraies images, ce dépassement se manifeste par des caractéristiques sursaturées et exagérées. Également vérifié dans l'exercice : le prédicteur de bruit conditionnel gaussien est égal à $-\sqrt{1-\bar\alpha_t}$ fois le score de $\mathcal N(\sqrt{\bar\alpha_t}m,\bar\alpha_ts^2+1-\bar\alpha_t)$ à la précision machine.

## ControlNet et LoRA

**ControlNet.** Le texte ne peut pas dire où se tient le chien. ControlNet garde le débruiteur pré-entraîné gelé, fait une *copie entraînable* de son encodeur qui reçoit aussi une condition spatiale (carte de contours, profondeur, pose), et ajoute les sorties de la copie aux caractéristiques du décodeur du réseau gelé via des **convolutions initialisées à zéro** :

$$
\text{features}\leftarrow\text{features}+\mathrm{zeroconv}\big(\text{encoder\_copy}(x_t,\text{condition})\big).
$$

Comme les convolutions nulles démarrent à 0, le modèle se comporte d'abord exactement comme l'original, et l'entraînement ne peut pas l'abîmer ; il apprend un delta. Plusieurs ControlNet peuvent être sommés avec des poids.

**LoRA** (adaptation de bas rang). Pour un poids gelé $W\in\mathbb R^{d_\text{out}\times d_\text{in}}$, on apprend une mise à jour de bas rang $\Delta W=\frac{\alpha}{r}BA$ avec $A\in\mathbb R^{r\times d_\text{in}}$, $B\in\mathbb R^{d_\text{out}\times r}$, $r\ll d$. On initialise $B=0$ (donc $\Delta W=0$ au départ) et $A$ aléatoirement. Paramètres entraînables : $r(d_\text{out}+d_\text{in})$ au lieu de $d_\text{out}d_\text{in}$. Pour $d=640$ et $r=16$, cela fait $20{,}480$ contre $409{,}600$, une réduction de $20\times$ (5 % des poids). Une matrice de rang $r$ convient bien quand la mise à jour vit dans un sous-espace de faible dimension, ce que font empiriquement les mises à jour de fine-tuning (voir [[cml.05]] pour le point de vue SVD du bas rang). À l'inférence, tu peux fusionner $W'=W+\Delta W$ (aucune latence supplémentaire) ou mettre l'adaptateur à l'échelle, et empiler plusieurs adaptateurs de façon additive. Les adaptateurs LoRA enseignent des concepts, des styles et des personnages ; ControlNet enseigne de nouveaux *signaux de conditionnement*.

## Implémentation

```python
import numpy as np

def cfg(eps_u, eps_c, w):
    return eps_u + w * (eps_c - eps_u)

def cross_attention(F, C, Wq, Wk, Wv):
    Q, K, V = F @ Wq, C @ Wk, C @ Wv
    S = Q @ K.T / np.sqrt(K.shape[1])
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return P @ V, P

def lora_delta(A, B, alpha=1.0):
    r = A.shape[0]
    return (alpha / r) * (B @ A)
```

En pratique : les pipelines `diffusers` prennent `guidance_scale`, exécutent les passes inconditionnelle et conditionnelle comme un seul batch de deux, et `peft`/`load_lora_weights` attachent les adaptateurs.

## Formes (shapes)

Latent : `(B, 4, 64, 64)`. Embeddings du prompt `C: (B, n_tok, d_c)` avec `n_tok` autour de 77 pour les encodeurs de type CLIP. Cross-attention : `F: (B, n_pix, d)`, scores `(B, n_pix, n_tok)`, sortie `(B, n_pix, d_k)`. LoRA : `A: (r, d_in)`, `B: (d_out, r)`, `ΔW: (d_out, d_in)`.

## Idées reçues

- **« Une échelle de guidage plus élevée donne une meilleure image. »** Au-delà d'une valeur modérée, la diversité s'effondre et les couleurs saturent.
- **« Le guidage nécessite un classifieur séparé. »** C'était la méthode précédente ; le guidage sans classifieur utilise le même réseau avec la condition supprimée.
- **« Le VAE de Stable Diffusion est entraîné conjointement avec le modèle de diffusion. »** Il est entraîné d'abord puis gelé.
- **« LoRA est un petit modèle. »** C'est un petit *delta* sur un grand modèle ; il est inutile sans les poids de base.
- **« ControlNet fait le fine-tuning du modèle de base. »** La base reste gelée ; une copie est entraînée.

## Points clés à retenir

- La diffusion dans l'espace latent d'un autoencodeur gelé réduit le calcul à peu près du facteur de compression ; la perte est inchangée.
- Le texte entre par la cross-attention : requêtes issues de l'image, clés/valeurs issues du prompt.
- $\tilde\varepsilon=\varepsilon_u+w(\varepsilon_c-\varepsilon_u)$ vient de l'aiguisage de $p(c\mid x)$ ; $w=1$ est le modèle conditionnel simple.
- ControlNet ajoute un contrôle spatial avec des connexions latérales initialisées à zéro ; LoRA ajoute $\frac{\alpha}{r}BA$ avec $B=0$ au départ.

## Explique-le toi-même

1. Dérive la formule de guidage à partir de $p_w(x|c)\propto p(x)p(c|x)^w$.
2. Pourquoi le $B$ de LoRA (ou l'un de ses facteurs) doit-il démarrer à zéro, et pourquoi ControlNet utilise-t-il des convolutions initialisées à zéro ?

## Pour aller plus loin

- Leçons sources : phase 08, leçons 07 (diffusion latente) et 08 (ControlNet, LoRA). Les articles originaux (Rombach et al. ; Ho et Salimans ; Zhang et al. ; Hu et al.) n'ont pas été lus pour cette leçon.
- Suite : [[gen.06]].

## Critères de maîtrise

Tu sais expliquer les deux étapes de la diffusion latente, dériver et implémenter le guidage sans classifieur, écrire la cross-attention avec des formes explicites, et calculer l'économie de paramètres et le comportement initial d'un adaptateur LoRA.
