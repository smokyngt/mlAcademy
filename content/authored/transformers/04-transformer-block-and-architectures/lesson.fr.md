---
id: trf.04
title: "Le bloc transformer et les architectures"
summary: "Connexions résiduelles, LayerNorm vs RMSNorm et pre-norm, la couche feed-forward et sa variante à porte SwiGLU, l'empilement de blocs, et les trois familles (encodeur seul, décodeur seul, encodeur-décodeur), assemblées en une passe avant de décodeur seul avec chaque forme et un décompte des paramètres."
minutes: 90
teaches: [trf.block, trf.architectures]
requires: [trf.attention, trf.multihead, trf.position, nn.layer]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/05-full-transformer/docs/en.md (six pieces of the block, pre-norm vs post-norm, 2017 vs 2026 block table, parameter count, encoder/decoder/encoder-decoder choice). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/06-bert-masked-language-modeling/docs/en.md (masked language modelling, 15% / 80-10-10 rule, bidirectional encoder, use cases). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md (next-token prediction, causal mask, parallel training vs serial generation). READ."
  - "LLMs-from-scratch/ch04/01_main-chapter-code/ch04.ipynb (Sec. 4.1 GPT-2 124M config, 4.2 LayerNorm with scale and shift, 4.3 GELU feed-forward with 4x expansion, 4.4 shortcut connections and gradients, 4.5 TransformerBlock, 4.6 GPTModel, parameter counts 163,009,536 and 124,412,160 with weight tying). READ."
---

## Objectifs d'apprentissage

- Dessiner un bloc transformer pre-norm et écrire ses deux équations de mise à jour.
- Expliquer ce qu'apportent respectivement les connexions résiduelles et la normalisation, et calculer LayerNorm et RMSNorm sur un vecteur.
- Écrire la couche feed-forward standard et la variante SwiGLU avec leurs formes, et expliquer la taille cachée $8d/3$.
- Comparer les modèles encodeur seul, décodeur seul et encodeur-décodeur par leur masque, leur objectif d'entraînement et leur usage.
- Assembler une passe avant complète de décodeur seul, compter ses paramètres sous forme close, et vérifier chaque forme de sortie.

## Où cela s'inscrit

Les leçons 1 à 3 ont donné trois ingrédients : l'attention (mélanger l'information entre les positions), le multi-têtes et les masques (plusieurs points de vue, avec un contrôle sur qui voit qui), et les positions (pour que l'ordre compte). Une couche d'attention reste un extracteur de caractéristiques peu profond. Le langage demande de la profondeur, et les piles profondes sont difficiles à entraîner sans la bonne plomberie.

Cette leçon, c'est la plomberie. Elle explique l'unité répétée, le **bloc transformer**, et comment l'empilement de blocs et le choix des masques donnent trois familles d'architectures : les encodeurs de type BERT (embeddings, recherche, classification), les décodeurs de type GPT (chat, code, génération de texte) et les modèles encodeur-décodeur (traduction, reconnaissance vocale). Après cette leçon, tu peux lire la section architecture de presque n'importe quel article sur les modèles de langage.

## L'idée en mots simples

Pense à un document qui traverse une rédaction avec $L$ bureaux alignés. À chaque bureau, deux personnes travaillent sur la *même copie* et écrivent des notes en marge ; elles n'effacent jamais ce qui s'y trouve.

1. Le commis **attention** lit tout le document et ajoute des notes en marge du type « ce mot est lié à ce mot précédent ».
2. Le commis **feed-forward** relit chaque mot isolément, y réfléchit plus profondément, et ajoute une note en marge.

L'habitude des notes en marge, c'est la **connexion résiduelle** : chaque sous-couche *ajoute* à un vecteur courant (le « flux résiduel ») au lieu de le remplacer. Une étape de **normalisation**, appliquée avant que chaque commis lise, garde l'écriture à une taille lisible pour que les notes de 40 bureaux n'explosent pas et ne s'estompent pas.

Empile $L$ bureaux et tu obtiens un transformer. Les seules décisions restantes sont qui peut lire quoi (le masque) et ce qu'on demande au modèle de prédire (l'objectif), ce qui donne les trois familles.

## Le bloc, formellement

Soit $x\in\mathbb{R}^{B\times T\times d}$ le flux résiduel. Un bloc **pre-norm** calcule

$$
x' = x+\operatorname{Attn}\big(\operatorname{Norm}_1(x)\big),\qquad
x'' = x'+\operatorname{FFN}\big(\operatorname{Norm}_2(x')\big).
$$

Les deux sous-couches envoient $(B,T,d)\to(B,T,d)$, donc les formes permettent la somme et un nombre quelconque de répétitions. Le modèle est

$$
\text{tokens}\;(B,T)\;\xrightarrow{\text{embed}}\;(B,T,d)\;\xrightarrow{L\text{ blocks}}\;(B,T,d)\;\xrightarrow{\text{Norm}}\;(B,T,d)\;\xrightarrow{\text{head}}\;(B,T,V)
$$

où $V$ est la taille du vocabulaire et le dernier tenseur contient un vecteur de logits par position.

### Connexions résiduelles

Avec $x_{l+1}=x_l+F_l(x_l)$, la jacobienne est $\partial x_{l+1}/\partial x_l=I+\partial F_l/\partial x_l$. En déroulant $L$ blocs, la dérivée du flux final par rapport au premier contient un terme égal à l'identité, un chemin le long duquel les gradients circulent sans être multipliés par aucune matrice de poids. Le chapitre 4 du livre le montre sur un petit réseau jouet de 5 couches : sans raccourcis, les gradients des premières couches sont bien plus petits que ceux des dernières ; avec raccourcis, ils restent comparables. La leçon source fait le même constat et affirme que sans résidus les gradients s'évanouissent au-delà d'une poignée de couches (elle indique environ 6 ; considère ce chiffre comme approximatif).

### Normalisation : LayerNorm et RMSNorm

Pour le vecteur d'un token $x\in\mathbb{R}^d$ (la normalisation agit indépendamment sur chacune des $B\cdot T$ lignes, le long du dernier axe) :

$$
\operatorname{LayerNorm}(x)=g\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+b,\quad
\mu=\tfrac1d\textstyle\sum_ix_i,\ \sigma^2=\tfrac1d\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\tfrac1d\sum_ix_i^2+\varepsilon}}.
$$

$g,b\in\mathbb{R}^d$ sont appris (initialisés à 1 et 0). RMSNorm abandonne la soustraction de la moyenne et le décalage $b$ : légèrement moins coûteux, avec $d$ paramètres au lieu de $2d$ par norme, et d'après la leçon source au moins aussi stable ; c'est le choix moderne par défaut là-bas. Les deux utilisent des statistiques *par token*, donc le résultat ne dépend ni du batch ni des autres positions (contrairement à la normalisation par batch).

**Pre-norm vs post-norm.** La disposition originale de 2017 était $\operatorname{Norm}(x+\operatorname{sublayer}(x))$ (post-norm) ; la disposition ci-dessus normalise l'*entrée* de chaque sous-couche et laisse le chemin résiduel complètement intact. La leçon source rapporte que le post-norm est plus difficile à entraîner en profondeur et que le pre-norm est le choix par défaut de 2026. Une raison que tu peux vérifier toi-même : en pre-norm, le chemin d'identité de l'entrée à la sortie ne porte aucune norme, donc le terme d'identité du gradient vu plus haut survit intact. Une conséquence est que l'amplitude du flux croît avec la profondeur, d'où la norme finale supplémentaire avant la tête.

### Le réseau feed-forward

Un MLP position par position appliqué de façon identique à chaque token (aucun mélange entre positions) :

$$
\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2,\qquad W_1:(d,4d),\ W_2:(4d,d),
$$

avec $\phi=$ GELU dans GPT-2 (le chapitre 4 du livre utilise l'approximation tanh de $x\Phi(x)$ et un facteur d'expansion 4). Formes : $(B,T,d)\to(B,T,4d)\to(B,T,d)$. Grossièrement, l'attention déplace l'information *entre* les tokens et le FFN la traite *à l'intérieur* de chaque token ; le FFN contient environ deux tiers des paramètres d'un bloc.

**SwiGLU (à porte).** Les blocs modernes utilisent une porte :

$$
\operatorname{SwiGLU}(x)=\big(\operatorname{SiLU}(xW_1)\odot xW_3\big)W_2,\quad
\operatorname{SiLU}(z)=z\,\sigma(z),
$$

avec $W_1,W_3:(d,f)$ et $W_2:(f,d)$. La branche $xW_3$ porte le contenu et $\operatorname{SiLU}(xW_1)$ est une porte douce apprise qui met à l'échelle chaque unité cachée. Trois matrices au lieu de deux : pour garder le budget de paramètres égal à la conception en $4d$, il faut $3df=2\cdot d\cdot4d$, c'est-à-dire

$$
f=\tfrac{8}{3}d\approx2.67\,d .
$$

(La leçon source cite 2.6 ; les vrais modèles arrondissent $f$ à un multiple commode. Dans notre modèle de test, $d=64$ donne $f=170.7$ et nous utilisons $f=172$.) La leçon source rapporte un petit gain de perplexité pour les variantes à porte dans des modèles publiés ; je ne l'ai pas vérifié de manière indépendante.

### Bloc moderne vs classique

| Élément | Vaswani 2017 | Bloc « 2026 » de la leçon source |
|---|---|---|
| Placement de la norme | post-norm | pre-norm |
| Norme | LayerNorm | RMSNorm |
| FFN | ReLU, 4x | SwiGLU, environ 2.6x |
| Position | sinusoïdale ajoutée | RoPE sur $Q,K$ |
| Attention | MHA complète | GQA ou MLA |
| Biais | oui | non |

## Familles d'architectures

Les trois utilisent le même bloc ; elles diffèrent par les masques, les entrées et l'objectif d'entraînement.

**Encodeur seul (BERT).** Pas de masque causal : chaque token voit tous les autres (bidirectionnel). Entraîné par **modélisation de langage masquée** : on cache environ 15 % des tokens et on les prédit. Parmi les tokens sélectionnés, 80 % sont remplacés par `[MASK]`, 10 % par un token aléatoire, 10 % laissés inchangés (pour que le modèle ne puisse pas compter sur l'apparition de `[MASK]`, qui n'arrive jamais à l'inférence). Sortie : un vecteur contextuel par token, $(B,T,d)$, qui alimente un classifieur, un étiqueteur, ou est agrégé en un embedding pour la recherche. Il ne peut pas naturellement générer du texte token par token. La leçon source dit que les encodeurs restent l'outil de la classification et de la recherche d'information et sont bien plus rapides par token que les décodeurs.

**Décodeur seul (GPT).** Masque causal ; entraîné par **prédiction du token suivant** : les logits à la position $t$ sont comparés au token $t+1$, avec la perte d'entropie croisée softmax moyennée sur les $B\cdot T$ positions, de sorte qu'une passe avant entraîne $T$ prédictions en parallèle. La génération est séquentielle : on échantillonne un token, on l'ajoute, on relance (un cache clé/valeur évite de tout recalculer). Pas d'encodeur séparé, pas d'attention croisée. Cette famille domine la génération de texte.

**Encodeur-décodeur (T5, type Whisper).** Un encodeur lit la source de façon bidirectionnelle, donnant $(B,T_{src},d)$. Le décodeur a une auto-attention causale plus une sous-couche d'**attention croisée** dont les requêtes viennent du flux du décodeur $(B,T_{tgt},d)$ et dont les clés et valeurs viennent de la sortie de l'encodeur $(B,T_{src},d)$. Les scores ont la forme $(B,h,T_{tgt},T_{src})$, rectangulaire et non carrée. Il convient aux tâches avec une source et une cible claires (traduction, reconnaissance vocale, résumé).

| | Encodeur seul | Décodeur seul | Encodeur-décodeur |
|---|---|---|---|
| Masque d'auto-attention | aucun (complet) | causal | encodeur complet, décodeur causal |
| Attention croisée | non | non | oui (décodeur) |
| Objectif | tokens masqués | token suivant | token cible suivant sachant la source |
| Usage typique | classer, plonger, étiqueter | générer | traduire, transcrire |

## Formes d'un modèle décodeur seul

Notation : vocabulaire $V$, largeur $d$, couches $L$, têtes $h$, largeur FFN $f$, batch $B$, longueur $T$.

| Étape | Forme | Paramètres appris |
|---|---|---|
| ids de tokens | $(B,T)$ | |
| consultation de l'embedding | $(B,T,d)$ | $Vd$ |
| par bloc : Norm1 | $(B,T,d)$ | $d$ (RMSNorm) |
| projections Q, K, V, découpage | $(B,h,T,d/h)$ | $3d^2$ |
| scores, poids | $(B,h,T,T)$ | aucun |
| fusion, $W_O$ | $(B,T,d)$ | $d^2$ |
| addition résiduelle | $(B,T,d)$ | |
| Norm2 | $(B,T,d)$ | $d$ |
| couche cachée SwiGLU | $(B,T,f)$ | $3df$ |
| retour au flux | $(B,T,d)$ | |
| norme finale | $(B,T,d)$ | $d$ |
| tête | $(B,T,V)$ | $dV$ |

**Nombre de paramètres du modèle de style Llama** (RMSNorm, SwiGLU, sans biais, tête non partagée) :

$$
N=\underbrace{Vd}_{\text{embed}}+L\big(\underbrace{4d^2}_{\text{attn}}+\underbrace{3df}_{\text{FFN}}+\underbrace{2d}_{\text{norms}}\big)+\underbrace{d}_{\text{final norm}}+\underbrace{dV}_{\text{head}} .
$$

Exemple $V=1000$, $d=64$, $L=4$, $h=4$, $f=172$ : un bloc a $4\cdot4096+3\cdot64\cdot172+128=16\,384+33\,024+128=49\,536$ paramètres ; le total est $64\,000+4\cdot49\,536+64+64\,000=326\,208$. Le code ci-dessous reproduit ce décompte et la forme des logits $(2,5,1000)$ pour un batch de 2 séquences de 5 tokens.

**Le modèle GPT-2 124M du livre** (chapitre 4 : $V=50257$, contexte 1024, $d=768$, $L=12$, $h=12$, pas de biais qkv, positions apprises, LayerNorm, FFN GELU avec biais) : embeddings de tokens $38\,597\,376$, positions $786\,432$, chaque bloc $7\,085\,568$, LayerNorm final $1\,536$ et une tête non partagée $38\,597\,376$ donnent $163\,009\,536$, ce qui correspond au total affiché par le notebook. Le GPT-2 original *partage* la tête de sortie avec l'embedding de tokens, ce qui supprime les $38\,597\,376$ paramètres de la tête : $124\,412\,160$, de nouveau le nombre affiché dans le notebook.

Deux enseignements de ces décomptes. Premièrement, par bloc, l'attention prend $4d^2$ et le FFN $8d^2$ : le FFN domine. Deuxièmement, pour les petits modèles, l'embedding et la tête représentent une grande fraction de l'ensemble (ici $128\,000$ sur $326\,208$).

## Implémentation

```python
import numpy as np

def rms_norm(x, g, eps=1e-6):                       # (B,T,d) -> (B,T,d)
    return g * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

def swiglu(x, W1, W3, W2):                          # (B,T,d) -> (B,T,f) -> (B,T,d)
    a = x @ W1
    return ((a / (1 + np.exp(-a))) * (x @ W3)) @ W2

def block(x, p, h):                                 # pre-norm residual block
    x = x + attn_rope(rms_norm(x, p["n1"]), p, h)   # attention from lessons 2 and 3
    x = x + swiglu(rms_norm(x, p["n2"]), p["W1"], p["W3"], p["W2"])
    return x

def forward(params, tokens, h):                     # tokens: (B,T) ints
    x = params["tok"][tokens]                       # (B,T,d)
    for p in params["blocks"]:
        x = block(x, p, h)                          # (B,T,d)
    return rms_norm(x, params["nf"]) @ params["head"]   # (B,T,V)
```

L'exercice fournit `attn_rope` (attention multi-têtes causale avec RoPE) et te demande d'écrire les éléments ci-dessus, plus le compteur de paramètres et les tests de fuite et d'identité résiduelle.

## Exemple détaillé (vérifié en exécutant le code)

Normalisation de $x=(1,2,3,4)$ (avec $\varepsilon=0$) :

- LayerNorm : $\mu=2.5$, $\sigma^2=1.25$, sortie $(-1.3416,-0.4472,0.4472,1.3416)$ : moyenne 0, variance 1.
- RMSNorm : $\operatorname{mean}(x^2)=7.5$, RMS $=2.7386$, sortie $(0.3651,0.7303,1.0954,1.4606)$ : moyenne quadratique 1, mais la moyenne n'est pas 0 et l'ordre ainsi que le motif de signes de $x$ sont préservés.

La première recentre, la seconde ne fait que remettre à l'échelle, donc la sortie de RMSNorm reste positive quand l'entrée est positive.

## Idées reçues courantes

- **« Le FFN mélange les tokens. »** Il agit sur chaque position indépendamment ; tout le mélange entre tokens se fait dans l'attention.
- **« Les connexions résiduelles sont un détail d'optimisation. »** Elles définissent l'architecture : chaque couche est une petite modification d'un flux partagé, pas une nouvelle représentation.
- **« LayerNorm normalise sur le batch. »** Elle normalise sur les caractéristiques d'un seul token ; le batch n'interagit pas.
- **« Les modèles décodeur seul ne peuvent pas comprendre le texte, seulement le générer. »** La même pile causale est entraînée à prédire et sert aussi aux tâches de compréhension ; le coût est que chaque token ne voit que son passé.
- **« Les paramètres croissent avec la longueur de séquence. »** Seule une table de positions apprise le fait ; RoPE et les sinusoïdes n'en ajoutent aucun. Le *calcul et la mémoire* de l'attention croissent en $T^2$ quoi qu'il arrive.
- **« BERT prédit le mot suivant. »** Il prédit des mots masqués à partir des deux côtés.

## Points clés à retenir

- Bloc : $x\leftarrow x+\operatorname{Attn}(\operatorname{Norm}(x))$, puis $x\leftarrow x+\operatorname{FFN}(\operatorname{Norm}(x))$ ; chaque tenseur du flux est $(B,T,d)$.
- Les résidus donnent aux gradients un chemin d'identité ; le pre-norm garde ce chemin propre.
- RMSNorm = LayerNorm sans soustraction de la moyenne ni décalage. SwiGLU = FFN à porte avec $f\approx8d/3$ pour égaler le budget de paramètres en $4d$.
- Encodeur seul : bidirectionnel + tokens masqués. Décodeur seul : causal + token suivant. Encodeur-décodeur : ajoute l'attention croisée $(B,h,T_{tgt},T_{src})$.
- Paramètres : $4d^2+3df+2d$ par bloc plus les embeddings ; le nombre est indépendant de $T$ et de $h$.

## Explique-le toi-même

1. En partant d'ids de tokens de forme $(B,T)$, raconte une passe avant de décodeur seul en nommant la forme après chaque opération, jusqu'aux logits.
2. Pourquoi la largeur cachée de SwiGLU est-elle proche de $\tfrac83d$ et non $4d$ ? Et pourquoi le FFN, et non l'attention, est-il la plus grosse partie d'un bloc ?

## Pour aller plus loin

- Raschka, *LLMs from scratch*, ch. 4 (Sec. 4.2 à 4.6), y compris la discussion sur le partage de poids.
- Bibliothèque source, phase 07, leçons 05 à 07 (transformer complet, BERT, GPT) et les leçons suivantes de cette phase sur les modèles encodeur-décodeur et le cache clé/valeur.

## Critères de maîtrise

Tu sais dessiner le bloc pre-norm, écrire les deux normes et SwiGLU, énoncer les trois familles avec leurs masques et leurs objectifs, et produire une passe avant de décodeur seul dont les logits ont la forme $(B,T,V)$, dont le nombre de paramètres correspond au $N$ ci-dessus, et qui passe le test d'absence de fuite du futur.
