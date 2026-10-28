---
id: llm.04
title: "Fine-tuning : réglage supervisé sur instructions et LoRA"
summary: "Comment un modèle de base qui ne fait que prolonger du texte devient un suiveur d'instructions (format des données, masquage de la perte sur le prompt), pourquoi mettre à jour tous les poids est coûteux, et comment LoRA entraîne une mise à jour de rang faible W + (alpha/r)BA avec quelques pour cent ou moins des paramètres, avec une couche NumPy et une vérification de gradient."
minutes: 80
teaches: [llm.sft, llm.lora]
requires: [llm.lm, nn.backprop, la.svd, la.matrix]
sources:
  - "LLMs-from-scratch/ch07/01_main-chapter-code/ch07.ipynb (Sec. on instruction data formatting: Alpaca-style prompt template, padding collate function, ignore_index -100, remark that masking instruction targets is common practice): READ (the formatting, collate and masking cells)."
  - "LLMs-from-scratch/appendix-E/01_main-chapter-code/appendix-E.ipynb (Sec. E.1 LoRA idea W+AB, E.4 LoRALayer with zero-initialised B, alpha/rank scaling, LinearWithLoRA, replacing Linear layers): READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/06-instruction-tuning-sft/docs/en.md (what SFT does, data formats Alpaca/ShareGPT/ChatML, masked loss, SFT hyperparameters, catastrophic forgetting): READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/08-fine-tuning-lora/docs/en.md and code/lora.py (LoRA parameter arithmetic, alpha, which layers, rank choice, QLoRA NF4/double quantisation/paged optimisers, merging, memory table): READ."
---

## Où cela s'inscrit

Le pré-entraînement ([[llm.03]]) produit un **modèle de base** : un excellent continuateur de texte et un piètre assistant. Demande-lui « Quelle est la capitale de la France ? » et il pourra poursuivre avec d'autres questions de quiz, car les listes de questions sont un motif courant dans ses données. Le transformer en quelque chose qui *répond* est le travail du **fine-tuning supervisé** (SFT), aussi appelé réglage sur instructions. La même mécanique adapte un modèle général à un domaine, au ton d'une entreprise ou à une tâche de classification.

Deux murs pratiques apparaissent immédiatement :

1. **Sur quoi entraîner, et sur quels tokens ?** La réponse : des paires de démonstrations (instruction, bonne réponse), avec la perte comptée sur la réponse seulement.
2. **Que mettre à jour ?** Mettre à jour tous les poids d'un modèle de 7 milliards de paramètres demande des gradients et un état d'optimiseur pour chacun (voir la facture mémoire de la leçon précédente) et produit une copie complète du modèle par tâche. **LoRA** gèle au contraire le modèle et apprend une toute petite correction.

Cette leçon démontre les deux éléments et les implémente. L'alignement à partir de données de préférence, qui affine encore le modèle SFT, est la leçon suivante ([[llm.05]]).

## L'idée en mots simples

**SFT.** Le pré-entraînement a donné au modèle des connaissances ; le SFT lui enseigne les manières. Il n'a pas besoin d'apprendre que Paris est la capitale de la France, il le sait déjà. Il doit apprendre que lorsqu'un message ressemble à une instruction, la prochaine chose à écrire est une *réponse*. C'est un petit changement de comportement, ce qui explique que des dizaines de milliers de bons exemples, et non des milliards, suffisent. Un point subtil : dans chaque exemple d'entraînement, l'instruction est le contexte, la réponse est ce que nous voulons produire. Si nous évaluions aussi le modèle sur la prédiction de l'instruction, nous dépenserions du signal d'apprentissage à lui apprendre à *poser* des questions. La perte est donc désactivée sur les tokens du prompt.

**LoRA.** Imagine que la matrice de poids pré-entraînée soit un immense tableau achevé et que le fine-tuning soit une retouche. Une retouche complète repeint chaque pixel et exige une toile aussi grande que l'original. Mais une retouche est généralement simple : quelques larges coups de pinceau, un décalage de couleur. LoRA dit : ne stocke pas la retouche pixel par pixel ; stocke-la comme le produit de deux matrices minces, une « grande et fine » fois une « courte et large », dont le produit ne peut modifier l'image que dans quelques directions (son **rang** est petit). Tu gardes le tableau original intact et tu ajoutes la fine correction par-dessus. Après l'entraînement, tu peux fusionner la correction dans le tableau, ou la garder comme calque amovible et échanger les calques selon les tâches.

## Le SFT en détail

**Formats de données.** Un exemple est une conversation transformée en une seule séquence de tokens. Dispositions courantes : Alpaca (`instruction`, `input` optionnel, `output`, rendus dans un gabarit fixe avec des en-têtes), listes multi-tours de style ShareGPT de (rôle, texte), et ChatML, qui délimite les rôles avec des tokens spéciaux tels que `<|im_start|>` et `<|im_end|>` ([[llm.01]]). Le gabarit utilisé au moment du fine-tuning doit être reproduit exactement au moment de l'inférence.

**La séquence.** Pour un prompt $p=(p_1..p_m)$ et une réponse $r=(r_1..r_n)$, construis $s=p\,\|\,r\,\|\,\text{eos}$, de longueur $m+n+1$. Les entrées sont $s_{1:-1}$ et les cibles $s_{2:}$. Les étiquettes aux positions du prompt sont remplacées par une valeur sentinelle (`-100`, la valeur que l'entropie croisée de PyTorch ignore ; le livre l'utilise aussi pour le padding).

**Perte masquée.** Soit $M_t\in\{0,1\}$ valant 1 là où la cible est un token de réponse ou de fin de texte. La perte est

$$
\mathcal L=-\frac{1}{\sum_t M_t}\sum_t M_t\,\ln p_\theta(y_t\mid s_{\le t}).
$$

Le gradient par rapport aux logits à la position $t$ est $M_t\,(p_t-\text{onehot}(y_t))/\sum M$, donc les positions masquées n'envoient *aucun* signal, mais les tokens du prompt influencent quand même la sortie par l'attention parce qu'ils sont dans le contexte. Dans un batch, les séquences sont complétées jusqu'à la plus longue ; les cibles de padding sont masquées aussi. Le même mécanisme gère les deux.

**Les hyperparamètres diffèrent du pré-entraînement.** La leçon source oppose un taux d'apprentissage de pic d'environ $3\times10^{-4}$ en pré-entraînement à environ $2\times10^{-5}$ en SFT, 1 à 3 époques au lieu d'une passe sur des billions de tokens, et quelques dizaines de milliers d'exemples de haute qualité (elle cite environ 27 000 paires triées pour un modèle de chat et 52 000 paires générées pour Alpaca). La qualité l'emporte sur la quantité.

**Oubli catastrophique.** De grands pas sur un petit jeu de données écrasent les capacités générales. Remèdes : petit taux d'apprentissage, peu d'époques, réinjecter quelques pour cent de données de type pré-entraînement, ou restreindre ce qui peut changer (LoRA ci-dessous).

## LoRA, démontré

### La paramétrisation

Fine-tuner une couche de poids $W\in\mathbb R^{d_{out}\times d_{in}}$ revient à apprendre une mise à jour $\Delta W$, de sorte que la couche adaptée est $W+\Delta W$. LoRA restreint la mise à jour à la forme

$$
\Delta W=\frac{\alpha}{r}\,B A,\qquad B\in\mathbb R^{d_{out}\times r},\quad A\in\mathbb R^{r\times d_{in}},\quad r\ll\min(d_{in},d_{out}),
$$

et n'entraîne que $A$ et $B$, avec $W$ gelé. (Le livre écrit le produit $AB$ avec les matrices transposées, pour correspondre à une convention où les entrées multiplient par la gauche ; c'est la même construction.) Le scalaire $\alpha$ est un hyperparamètre ; le facteur $s=\alpha/r$ garde la taille de la mise à jour comparable d'un rang à l'autre, ce qui, note le livre, permet de comparer des exécutions de rangs différents sans réajuster le taux d'apprentissage.

**Rang.** Le rang d'un produit est au plus le rang de chaque facteur, donc $\operatorname{rank}(\Delta W)\le r$. Le pari (Hu et al.) est que la mise à jour *utile* pendant l'adaptation a un « rang intrinsèque » faible, donc un petit $r$ perd peu. Une matrice de rang $r$ est exactement ce que donne la troncature d'une SVD à $r$ valeurs singulières ; LoRA apprend directement une telle factorisation au lieu de calculer une mise à jour complète puis de la tronquer.

**Nombre de paramètres.** Une mise à jour complète a $d_{out}d_{in}$ nombres ; LoRA en a $r(d_{out}+d_{in})$. Pour $d_{out}=d_{in}=4096$ et $r=16$ : $16\cdot8192=131{,}072$ contre $16{,}777{,}216$, soit $0.78\%$. Appliqué aux quatre projections d'attention de chaque couche de GPT-2 small ($d=768$, 12 couches, $r=8$), l'adaptateur a $4\cdot12\cdot8\cdot(768+768)=589{,}824$ paramètres, soit $0.47\%$ des 124,439,808 poids.

### Passe avant sans former $\Delta W$

Pour un batch $x\in\mathbb R^{B\times d_{in}}$ (les lignes sont des exemples) :

$$
y=xW^\top+s\,(xA^\top)B^\top,\qquad h=xA^\top\in\mathbb R^{B\times r}.
$$

Par distributivité, $x(W+sBA)^\top=xW^\top+s\,xA^\top B^\top$, donc nous ne construisons jamais la mise à jour $d_{out}\times d_{in}$ ; le coût supplémentaire est de deux produits matriciels minces à travers le goulot de largeur $r$.

### Passe arrière

Soit $G=\partial\mathcal L/\partial y\in\mathbb R^{B\times d_{out}}$. Terme à terme, $y_{bk}=\sum_j x_{bj}W_{kj}+s\sum_{i}h_{bi}B_{ki}$ et $h_{bi}=\sum_j x_{bj}A_{ij}$. Par conséquent :

$$
\frac{\partial\mathcal L}{\partial B_{ki}}=s\sum_b G_{bk}h_{bi}\;\Longrightarrow\;\bar B=s\,G^\top h\quad(d_{out}\times r),
$$

$$
\frac{\partial\mathcal L}{\partial h_{bi}}=s\sum_k G_{bk}B_{ki}\;\Longrightarrow\;\bar h=s\,GB\quad(B\times r),
$$

$$
\frac{\partial\mathcal L}{\partial A_{ij}}=\sum_b\bar h_{bi}x_{bj}\;\Longrightarrow\;\bar A=\bar h^\top x=s\,(GB)^\top x\quad(r\times d_{in}),
$$

$$
\bar x=GW+\bar hA=GW+s\,(GB)A\quad(B\times d_{in}).
$$

(La dernière ligne est ce qu'il faut transmettre aux couches précédentes ; le $W$ gelé n'a besoin d'aucun gradient propre.)

### Initialisation : pourquoi $B=0$ et $A$ aléatoire

Avec $B=0$, l'adaptateur ne contribue exactement à rien, donc le modèle démarre comme le modèle pré-entraîné : $\Delta W=0$. Les formules ci-dessus montrent pourquoi il ne faut pas mettre *les deux* à zéro : $\bar A\propto B$ et $\bar B\propto h=xA^\top$. Si $A=0$ aussi, alors $h=0$ et $\bar B=0$ ; si $B=0$ alors $\bar A=0$. Avec $B=0$ et $A$ aléatoire, $\bar A=0$ à la première étape mais $\bar B\neq0$, donc $B$ s'écarte de zéro et dès lors les deux facteurs apprennent. $A$ aléatoire avec $B=0$ est le choix standard (gaussien dans l'article, Kaiming uniforme dans le `LoRALayer` du livre).

### Fusion et échange

Après l'entraînement, $W_{\text{merged}}=W+\frac{\alpha}{r}BA$ donne une couche ordinaire sans latence supplémentaire. Autrement, garde $A,B$ séparés : un modèle de base gelé en mémoire et un petit fichier par tâche, commutable à l'exécution.

### Où l'appliquer, quel rang

La leçon source rapporte qu'adapter les projections de requête et de valeur est un bon choix par défaut, les quatre projections d'attention un peu mieux, et que l'ajout des matrices du MLP double environ les paramètres pour de petits gains ; les rangs 4 à 16 sont courants, avec des rendements décroissants au-delà de 64. Considère cela comme des règles empiriques tirées de cette source, à vérifier sur ta tâche. Le livre remplace chaque couche `Linear` du modèle par un `LinearWithLoRA` après avoir gelé les originales.

### QLoRA

QLoRA garde les poids de base *gelés* en précision 4 bits (la leçon source nomme le format NF4, dont les 16 niveaux sont placés aux quantiles d'une loi normale car les poids sont à peu près normaux), stocke les constantes de quantification de façon compacte (« double quantification »), et entraîne les matrices LoRA en plus haute précision ; un état d'optimiseur paginé évite les pics de dépassement de mémoire. Les gradients passent à travers les poids déquantifiés jusqu'à $A$ et $B$. Le tableau mémoire de la leçon source pour un modèle 7B (approximatif, tel que rapporté là-bas) : fine-tuning complet environ 56 GB, LoRA sur une base 16 bits environ 18 GB, QLoRA environ 6 GB. L'économie vient du fait de ne pas stocker de gradients ni d'état d'optimiseur pour les poids gelés, et de réduire les poids eux-mêmes.

## Exemple détaillé

Une seule couche LoRA assez petite pour le crayon. $d_{in}=3$, $d_{out}=2$, $r=1$, $\alpha=2$ donc $s=2$. $A=(1,0,-1)$, $B=(0.5,-1)^\top$, entrée $x=(1,2,3)$.

- $h=xA^\top=1\cdot1+2\cdot0+3\cdot(-1)=-2$.
- Sortie de l'adaptateur $s\,h\,B^\top=2\cdot(-2)\cdot(0.5,-1)=(-2,\,4)$.
- Mise à jour fusionnée équivalente $\Delta W=sBA=\begin{pmatrix}1&0&-1\\-2&0&2\end{pmatrix}$, et $\Delta W x=(1-3,\,-2+6)=(-2,4)$. Même résultat, comme le promet la distributivité. Rang 1 : les deux lignes sont proportionnelles.
- Gradient amont $G=(1,-1)$ : $\bar B=s\,G^\top h=2\cdot(-2)\cdot(1,-1)=(-4,4)$ ; $GB=1\cdot0.5+(-1)(-1)=1.5$ ; $\bar A=s\,(GB)\,x=2\cdot1.5\cdot(1,2,3)=(3,6,9)$.

J'ai vérifié ces valeurs en exécutant les fonctions de l'exercice. Nombre de paramètres : 5 nombres entraînables contre 6 pour la matrice complète $2\times3$ ; l'économie ne devient spectaculaire qu'à grande largeur.

**Test de capacité.** Dans l'exercice, le $W$ pré-entraîné est $8\times10$ et la cible en diffère par une matrice de rang 2. La descente de gradient sur $A,B$ avec $r=2$ fait passer l'erreur quadratique de $13.17$ à environ $10^{-30}$ (la mise à jour est retrouvée exactement), tandis que $r=1$ stagne à environ $2.9$ : une mise à jour de rang 1 ne peut pas représenter un changement de rang 2. C'est le compromis en miniature.

Du côté SFT, `build_example([5,6,7],[8,9],eos=1)` donne les ids `5 6 7 8 9 1` et les étiquettes `-100 -100 -100 8 9 1` ; après le décalage, entrées `5 6 7 8 9` et cibles `-100 -100 8 9 1`, donc trois positions sur cinq portent de la perte : le dernier token du prompt `7` est la première entrée dont la cible est supervisée.

## Implémentation

```python
import numpy as np

def lora_forward(x, W, A, B, alpha):          # x:(Bt,d_in) W:(d_out,d_in) A:(r,d_in) B:(d_out,r)
    s = alpha / A.shape[0]
    h = x @ A.T                                # (Bt, r)  -- cache for backward
    return x @ W.T + s * (h @ B.T), h

def lora_backward(G, x, h, W, A, B, alpha):   # G = dL/dy : (Bt, d_out)
    s = alpha / A.shape[0]
    GB = G @ B                                 # (Bt, r)
    dB = s * G.T @ h                           # (d_out, r)
    dA = s * GB.T @ x                          # (r, d_in)
    dx = G @ W + s * GB @ A                    # (Bt, d_in)
    return dx, dA, dB

def merge(W, A, B, alpha):
    return W + (alpha / A.shape[0]) * B @ A
```

```python
IGNORE = -100
def masked_ce_grad(p, y):                      # p: (B,T,V) probabilities, y: (B,T) with IGNORE
    valid = y != IGNORE
    g = np.zeros_like(p)
    b, t = np.nonzero(valid)
    g[b, t] = p[b, t]; g[b, t, y[b, t]] -= 1
    return g / max(1, valid.sum())
```

## Formes

- $W:(d_{out},d_{in})$ gelé ; $A:(r,d_{in})$ ; $B:(d_{out},r)$ ; $x:(B,d_{in})$ ; $h:(B,r)$ ; $y,G:(B,d_{out})$.
- Batch SFT : entrées $X:(B,T)$ entiers, cibles $Y:(B,T)$ entiers avec `-100` ; logits $(B,T,V)$ ; la perte divise par le nombre de cibles non masquées, pas par $B\cdot T$.
- Taille de l'adaptateur : $r(d_{out}+d_{in})$ par matrice adaptée ; un modèle avec $n$ matrices adaptées stocke $n$ telles paires.

## Limites et compromis

- **Capacité.** Une mise à jour de rang faible ne peut pas représenter des changements qui exigent un rang élevé (l'échec de rang 1 ci-dessus). Les tâches qui demandent beaucoup de nouvelles connaissances, et pas seulement un nouveau comportement, peuvent préférer un rang plus élevé, plus de couches, ou un fine-tuning complet.
- **Hyperparamètres.** $r$, $\alpha$, les couches ciblées et le taux d'apprentissage interagissent ; la mise à l'échelle $\alpha/r$ réduit cela sans le supprimer.
- **L'erreur de quantification** de QLoRA est une petite approximation supplémentaire sur les poids gelés ; la qualité rapportée est proche de LoRA 16 bits sur les benchmarks des tableaux de la source, mais vérifie sur ta propre tâche.
- **Le SFT enseigne le format, pas la vérité.** Si les démonstrations contiennent des erreurs ou inventent des faits, le modèle apprend à produire du texte assuré dans ce style. Le SFT ne peut pas non plus exprimer une *préférence entre deux réponses acceptables*, ce qui motive la leçon suivante.
- **L'oubli persiste** à fort taux d'apprentissage même avec LoRA ; la base gelée ne fait que limiter la distance dont la fonction peut se déplacer.

## Idées reçues courantes

- « Le fine-tuning apprend de façon fiable de nouveaux faits au modèle. » Il façonne surtout le comportement et le style ; les faits sont surtout fixés au pré-entraînement.
- « LoRA est une approximation du modèle fine-tuné qui est toujours moins bonne. » C'est un fine-tuning contraint ; quand la mise à jour nécessaire est de rang faible, il peut égaler le fine-tuning complet, sinon non.
- « Masquer le prompt signifie que le modèle l'ignore. » Les tokens masqués ne portent pas de perte, mais ils restent dans le contexte et la réponse porte son attention sur eux.
- « Initialise A et B à zéro par sécurité. » Alors aucun des deux facteurs ne reçoit jamais de gradient.
- « LoRA fusionné et non fusionné donnent des sorties différentes. » Ils sont algébriquement identiques, aux arrondis en virgule flottante près.

## Points clés à retenir

- SFT = entraînement au token suivant sur des séquences (prompt, réponse) avec la perte restreinte aux tokens de réponse et de fin ; faible taux d'apprentissage, peu d'époques, la qualité avant la quantité.
- LoRA gèle $W$ et apprend $\Delta W=\frac{\alpha}{r}BA$ de rang $\le r$ avec $r(d_{in}+d_{out})$ paramètres ; $B=0$ au départ pour que le modèle soit inchangé, et $A$ aléatoire pour que $B$ reçoive un gradient.
- Gradients : $\bar B=sG^\top h$, $\bar A=s(GB)^\top x$, $\bar x=GW+s(GB)A$.
- Fusionner par $W+sBA$ pour une inférence sans latence, ou garder des adaptateurs séparés par tâche.
- QLoRA stocke la base gelée en 4 bits et entraîne des adaptateurs par-dessus.

## Explique-le toi-même

1. Pourquoi mettre *les deux* matrices LoRA à zéro gèle-t-il l'entraînement pour toujours, et pourquoi ne mettre que $B$ à zéro ne le fait-il pas ?
2. Un collègue ne masque rien et fait du fine-tuning sur prompt plus réponse. Qu'est-ce qui change dans la perte, et quel comportement indésirable cela peut-il encourager ?

## Pour aller plus loin

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* ; Dettmers et al. (2023), *QLoRA*.
- Raschka, *Build a Large Language Model From Scratch*, chapitre 7 et annexe E.
- Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, pour le SFT comme première étape du pipeline.

## Critères de maîtrise

- Construire à la main un exemple SFT avec des entrées correctement décalées et des étiquettes masquées.
- Établir $\bar A,\bar B,\bar x$ pour une couche LoRA et passer une vérification par différences finies.
- Calculer le nombre de paramètres LoRA et la fraction pour une matrice et un rang donnés.
- Expliquer l'initialisation, l'identité de fusion, et ce qui limite la méthode.
