---
id: seq.05
title: Séquence-à-séquence et chemin vers l'attention
summary: "Les modèles encodeur-décodeur, le teacher forcing, le décodage glouton et par faisceau (beam search), le goulot d'étranglement du vecteur de taille fixe, l'attention additive de Bahdanau dérivée avec sa passe arrière, et comment elle devient l'auto-attention par produit scalaire normalisé."
minutes: 110
teaches: [seq.seq2seq, seq.attention]
requires: [seq.rnn, seq.lstm, seq.gru, nn.backprop, prob.basics]
sources:
  - "math/cours-tal-03.pdf (CNAM RCP217 'Sequence vers sequence', PDF p.2-23, 36: Sutskever and Cho encoder-decoder architecture, encoder/decoder inputs and outputs, greedy decoding, the objective argmax P(y|x), chain rule, Markov/Viterbi remark, log-probabilities, beam search and its pseudocode). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/docs/en.md (encoder, decoder, teacher forcing and exposure bias, greedy decode, bottleneck demonstration, scheduled sampling). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/10-attention-mechanism/docs/en.md (Bahdanau additive score and shapes table, Luong variants, s_{t-1} vs s_t convention, worked weights, Q/K/V reading, masking, attention-is-not-explanation caveat). READ."
---

## Objectifs d'apprentissage

- Dessiner un encodeur-décodeur, dire ce que reçoit et produit chaque partie, et expliquer le teacher forcing et le biais d'exposition.
- Expliquer le décodage glouton face à la recherche par faisceau comme approximations de $\arg\max_yP(y\mid x)$.
- Expliquer le goulot d'étranglement d'un vecteur de contexte de taille fixe.
- Dériver l'attention additive (Bahdanau) avec les dimensions, y compris la passe arrière.
- Montrer comment l'attention par produit scalaire et l'auto-attention la généralisent.
- Implémenter le calcul d'attention et une minuscule passe avant seq2seq avec attention en NumPy.

## Prérequis

Les réseaux récurrents et les cellules à portes ([[seq.03]], [[seq.04]]), l'entropie croisée et le softmax ([[f2n.08]]), la rétropropagation avec les dimensions ([[f2n.10]]).

## Où cela s'inscrit

La traduction, le résumé, la réécriture de questions et la reconnaissance vocale envoient toutes une séquence vers une *autre* séquence, dont la longueur n'est pas connue d'avance et n'a pas à égaler celle de l'entrée (la catégorie « réécriture » du cours du CNAM). Les articles de 2014 de Sutskever et al. et Cho et al. ont montré que deux réseaux récurrents collés ensemble peuvent le faire. Les diapositives du CNAM décrivent le système de Sutskever comme une pile de 4 couches LSTM de 1000 cellules chacune avec environ 380 millions de paramètres, presque aussi bon que les meilleurs systèmes de traduction (même non neuronaux) de l'époque.

Cette leçon est aussi la charnière de tout le cours : la faiblesse de l'encodeur-décodeur (un seul vecteur doit porter toute l'entrée) est exactement ce que l'**attention** supprime, et l'attention, une fois la récurrence retirée, c'est le transformer.

## L'idée en mots simples

Un interprète écoute toute une phrase en français, se forme une compréhension, puis la prononce en anglais mot à mot. Deux rôles : un **auditeur** (l'encodeur) et un **locuteur** (le décodeur). Dans la conception d'origine, l'auditeur remet au locuteur une petite fiche, l'état de mémoire final, et rien d'autre. Le locuteur doit produire toute la traduction à partir de cette seule fiche. Pour une phrase de quatre mots, cela marche. Pour une phrase de quarante mots, cela échoue, car tout le détail doit être comprimé sur une fiche de taille fixe.

L'**attention** change la règle : le locuteur garde la *transcription entière* des notes de l'auditeur (une note par mot source). Avant de prononcer chaque mot, le locuteur jette un coup d'œil en arrière, décide quelles notes comptent *maintenant* (en disant « cat », regarder « chat » ; en disant « black », regarder « noir », même si l'ordre est inversé), mélange ces notes en un résumé tout frais, et parle. Le « combien regarder chaque note » correspond aux **poids d'attention** ; ce sont aussi un alignement doux entre les deux langues.

## Encodeur-décodeur

Source $x_1,\dots,x_{T_s}$, cible $y_1,\dots,y_{T_t}$ (avec des tokens spéciaux `<start>`/`<end>`).

- **Encodeur** : un RNN (LSTM/GRU/tanh) sur la source plongée, produisant des états $h_1,\dots,h_{T_s}$, $h_i\in\mathbb{R}^{H}$. Le modèle classique ne garde que $h_{T_s}$. Il peut être bidirectionnel car toute la source est connue (diapositives, p. 6).
- **Décodeur** : un RNN dont le premier état est initialisé avec le dernier état de l'encodeur. À chaque pas, il prend le token cible précédent (et l'état précédent) et produit un softmax sur le vocabulaire cible plus un nouvel état, transmis au pas suivant (diapositives, p. 3 et 6).

Le modèle définit, par la règle de la chaîne des probabilités,

$$
P(y\mid x)=\prod_{k=1}^{T_t}P(y_k\mid x,\;y_{<k}).
$$

Chaque facteur est une sortie softmax du décodeur. Remarque que $y_k$ dépend des mots *précédemment choisis* : traduire « I like potatoes » après avoir écrit « j'aime les pommes » n'est pas pareil qu'après « j'aime les patates » (diapositives, p. 7).

### Entraînement : teacher forcing

Perte : $-\log P(y\mid x)=-\sum_k\log P(y_k\mid x,y_{<k})$, la somme des entropies croisées par pas ([[f2n.08]]), par BPTT ordinaire à travers le décodeur **et** l'encodeur. Pendant l'entraînement, l'entrée au pas $k$ est le token précédent **de référence** (gold) $y_{k-1}$, pas la prédiction du modèle : c'est le *teacher forcing*. Il stabilise l'apprentissage (une erreur précoce ne corrompt pas toutes les entrées suivantes) et permet de calculer les pertes de tous les pas à partir de la séquence de référence. Le prix est le **biais d'exposition** : à l'inférence, le modèle doit consommer ses propres sorties, possiblement fausses, une situation qu'il n'a jamais pratiquée. L'échantillonnage planifié (mélanger tokens de référence et prédits, en annulant progressivement vers les prédictions) est une atténuation (leçon source).

### Inférence : glouton vs recherche par faisceau

On veut $y^\star=\arg\max_yP(y\mid x)$, une recherche exponentielle sur $|V|^{T_t}$ séquences. Les diapositives soulignent que Viterbi ne peut pas être utilisé car le réseau n'a pas d'hypothèse de Markov : chaque pas dépend de tout le préfixe via son état caché.

- **Glouton** : à chaque pas, prendre l'$\arg\max$ et le réinjecter. Bon marché ; un premier mot localement bon peut mener à une mauvaise phrase.
- **Recherche par faisceau** (beam search) de largeur $b$ : garder les $b$ meilleures hypothèses partielles, étendre chacune avec les $b$ meilleurs mots suivants, garder les $b$ meilleurs des $b\cdot b$ candidats, recommencer ; les hypothèses se terminant par `<end>` sont finies. Les scores sont des **sommes de log-probabilités**, $\log P(y_{\le k}\mid x)=\log P(y_k\mid x,y_{<k})+\log P(y_{<k}\mid x)$ (multiplier des probabilités provoque un sous-dépassement ; remarque qu'on somme les log-*softmax*, pas les logits bruts). $b=1$ est le glouton. C'est une heuristique sans garantie d'optimalité.

## Le goulot d'étranglement

Avec l'état final de l'encodeur comme seul lien, l'accès du décodeur à la source est un seul vecteur de taille $H$ :

$$
c=h_{T_s}\in\mathbb{R}^{H},\qquad\text{the same for every output step.}
$$

Deux problèmes en découlent. (1) **Capacité** : l'information nécessaire croît avec la longueur de la source mais pas $c$. La leçon source montre une tâche de copie où la précision s'effondre quand la séquence grandit, bien que l'information soit disponible à chaque pas de l'encodeur. (2) **Attribution du crédit** : les gradients des derniers mots cibles doivent remonter la longue chaîne de pas récurrents pour atteindre les premiers mots source ([[seq.03]]). Les diapositives notent qu'inverser la phrase source peut améliorer les résultats selon la langue (p. 4) ; une lecture plausible, que je donne comme interprétation, est que cela raccourcit le chemin entre les premiers mots source et cible.

## L'attention de Bahdanau, dérivée

Garde **tous** les états de l'encodeur $H=(h_1;\dots;h_{T_s})\in\mathbb{R}^{T_s\times H}$ et laisse le décodeur former un contexte $c_t$ *différent* à chaque pas. Au pas $t$, avec l'état précédent du décodeur $s_{t-1}\in\mathbb{R}^{d_s}$ comme **requête** :

1. **Noter** chaque position source avec un petit réseau feed-forward (attention additive) :

$$
e_{t,i}=v^\top\tanh\big(W_as_{t-1}+U_ah_i\big),\qquad W_a:(A,d_s),\;U_a:(A,H),\;v:(A).
$$

   $W_as_{t-1}+U_ah_i$ a la forme $(A)$ ; le vecteur $v$ l'écrase en un scalaire. Pour toutes les positions à la fois : $Z=s_{t-1}W_a^\top+HU_a^\top:(T_s,A)$, $u=\tanh Z$, $e=uv:(T_s)$.

2. **Normaliser** en poids de somme 1 (un softmax sur les positions source) :

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T_s}\exp(e_{t,j})}.
$$

3. **Contexte** : la moyenne pondérée des états de l'encodeur (c'est l'« alignement » à l'œuvre) :

$$
c_t=\sum_{i=1}^{T_s}\alpha_{t,i}\,h_i\in\mathbb{R}^{H}.
$$

4. **Décoder** : $s_t=\tanh(W_d[\,\mathrm{emb}(y_{t-1});c_t\,]+U_ds_{t-1}+b_d)$ et $\text{logits}_t=W_o[\,s_t;c_t\,]+b_o$.

Empiler les poids sur tous les pas de sortie donne la **matrice d'alignement** $\alpha\in\mathbb{R}^{T_t\times T_s}$, dont chaque ligne somme à 1 ; tracée, elle montre quels mots source chaque mot de sortie a regardés.

**Avertissement de convention.** Bahdanau note avec l'état *précédent* du décodeur $s_{t-1}$ ; les variantes de Luong utilisent l'état *courant* $s_t$. Les mélanger donne des résultats subtilement faux (leçon source). La fonction de score peut aussi être multiplicative : produit scalaire $s^\top h_i$ (exige $d_s=H$), ou générale $s^\top Wh_i$.

### Passe arrière du bloc d'attention

Soit $\bar c$ le gradient qui arrive au contexte. Tout est une application de règles déjà vues.

- Contexte $c=\sum_i\alpha_ih_i$ : $\ \bar\alpha_i=\bar c^\top h_i$ et le **chemin des valeurs** $\bar h_i\mathrel{+}=\alpha_i\bar c$.
- Softmax (jacobienne $\operatorname{diag}(\alpha)-\alpha\alpha^\top$) : $\ \bar e_i=\alpha_i\big(\bar\alpha_i-\sum_j\alpha_j\bar\alpha_j\big)$.
- Score $e_i=v^\top u_i$ : $\ \bar v=\sum_i\bar e_iu_i$ et $\bar u_i=\bar e_iv$.
- tanh : $\bar z_i=\bar u_i\odot(1-u_i^2)$ où $z_i=W_as+U_ah_i$.
- Parties linéaires : $\bar W_a=\big(\sum_i\bar z_i\big)s^\top$, $\bar s=W_a^\top\sum_i\bar z_i$, $\bar U_a=\sum_i\bar z_ih_i^\top$, et le **chemin des clés** $\bar h_i\mathrel{+}=U_a^\top\bar z_i$.

Chaque état de l'encodeur reçoit donc du gradient de deux endroits : directement par la moyenne pondérée (chemin des valeurs) et par le score (chemin des clés). Les deux sont vérifiés par différences finies dans le premier exercice. Remarque que le gradient de chaque pas de sortie atteint chaque position source via $\alpha$ en un seul saut : aucune longue chaîne récurrente n'est nécessaire. C'est pourquoi l'attention répare aussi le problème (2).

## Exemple détaillé

Trois états d'encodeur dans $\mathbb{R}^2$ : $H=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$ (pense « chat », « noir », « le »), requête $s=(0.5,-0.5)$, et par transparence $W_a=U_a=I$, $v=(1,1)$.

- $u_1=\tanh(1.5,-0.5)=(0.9051,-0.4621)$, $e_1=0.4430$
- $u_2=\tanh(0.5,0.5)=(0.4621,0.4621)$, $e_2=0.9242$
- $u_3=\tanh(1.5,0.5)=(0.9051,0.4621)$, $e_3=1.3672$
- $\alpha=\operatorname{softmax}(0.4430,0.9242,1.3672)=(0.1946,\;0.3149,\;0.4905)$
- $c=0.1946(1,0)+0.3149(0,1)+0.4905(1,1)=(0.6851,\;0.8054)$

Le troisième état, dont la clé s'accorde le mieux à la requête, reçoit 49 % du poids. Si l'état du décodeur bouge, les poids bougent avec lui, et le contexte aussi.

## Implémentation

```python
import numpy as np

def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def additive_attention(s, H, Wa, Ua, v, mask=None):
    u = np.tanh(s @ Wa.T + H @ Ua.T)      # (T, A)
    e = u @ v                             # (T,)
    if mask is not None:                  # True = real token
        e = np.where(mask, e, -1e9)       # padded positions get weight ~0
    alpha = softmax(e)                    # (T,)
    return alpha @ H, alpha               # (d_h,), (T,)
```

Le second exercice câble ceci dans un petit seq2seq : plongement + encodeur RNN tanh, une cellule de décodeur alimentée avec `[embedding of previous token; context]`, teacher forcing avec une perte d'entropie croisée (environ $\ln V$ à l'initialisation aléatoire : 2.30 pour $V=10$), la référence `use_attention=False` dont le contexte est toujours le dernier état de l'encodeur, et le décodage glouton. Les tests vérifient que la matrice d'alignement a la forme $(T_t,T_s)$ avec des lignes de somme 1, que le teacher forcing est causal (changer la dernière entrée de référence ne modifie pas les logits précédents), et que forcer par teacher forcing la sortie gloutonne reproduit les arg-max gloutons.

### Masquage et dimensions

| Grandeur | Dimensions |
|---|---|
| états de l'encodeur $H$ | $(T_s,H)$ (par batch $(B,T_s,H)$) |
| requête $s_{t-1}$ | $(d_s)$ |
| scores $e$, poids $\alpha_t$ | $(T_s)$ |
| contexte $c_t$ | $(H)$ |
| matrice d'alignement | $(T_t,T_s)$, lignes de somme 1 |

Dans un batch de longueurs inégales, les positions source remplies (padding) doivent être exclues : mets leurs scores à un nombre très négatif *avant* le softmax pour que leur poids soit 0 et que le reste somme toujours à 1 (testé dans l'exercice).

## De l'attention à l'auto-attention

Renomme les pièces : l'état du décodeur est la **requête** (query) $q$, les états de l'encodeur sont les **clés** (keys) $k_i$ (ce contre quoi on compare) et les **valeurs** $v_i$ (ce que l'on renvoie, ici les mêmes $h_i$). L'attention, c'est « softmax des scores requête-clé, puis moyenne des valeurs ». Remplace le score additif par un produit scalaire et exécute toutes les requêtes à la fois :

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big)V,\qquad Q:(T_q,d),\;K:(T_k,d),\;V:(T_k,d_v).
$$

Le facteur $1/\sqrt d$ : si les entrées de $q$ et $k$ sont indépendantes de moyenne 0 et de variance 1, alors $q\cdot k=\sum_{j=1}^dq_jk_j$ a pour variance $d$, donc les scores croîtraient comme $\sqrt d$ et saturerait le softmax ; diviser par $\sqrt d$ restaure une variance unité.

Deux étapes de plus donnent le transformer :

1. **Auto-attention** : prendre $Q,K,V$ de la **même** séquence, via des projections apprises $XW_Q,XW_K,XW_V$. Chaque position se relie désormais directement à chaque autre en un seul pas, quelle que soit la distance (longueur de chemin 1 au lieu de jusqu'à $T$ pas récurrents), et le calcul sur les positions est parallèle plutôt que séquentiel.
2. **Abandonner la récurrence.** Comme l'attention seule est équivariante par permutation (l'exercice le teste : mélanger les entrées mélange les sorties), l'ordre doit être injecté séparément par une information de position. Empiler de telles couches avec des connexions résiduelles ([[seq.01]]) et des couches feed-forward, c'est le transformer, traité dans le cours suivant.

## Idées reçues

- **« L'attention remplace l'encodeur. »** L'encodeur produit toujours les états que l'attention lit.
- **« Les poids d'attention expliquent le modèle. »** Ils sont intuitifs mais ne sont pas en général une explication fidèle ; la leçon source cite des travaux montrant que les poids peuvent être altérés sans changer les prédictions sur certaines tâches. Vérifie par des ablations avant de prétendre à un raisonnement.
- **« Le teacher forcing, c'est tricher au moment du test. »** C'est seulement une technique d'entraînement ; au test, le modèle utilise ses propres sorties (glouton ou faisceau).
- **« La recherche par faisceau trouve la meilleure traduction. »** C'est une heuristique ; des faisceaux plus larges ne sont pas garantis meilleurs.
- **« Le goulot d'étranglement est un problème d'entraînement. »** Il est structurel : entraîner davantage n'agrandit pas un vecteur de taille fixe.
- **« Les poids softmax doivent être piqués. »** Ce sont des mélanges doux ; piqués ou diffus selon les scores.

## Limites et compromis

- L'attention coûte $O(T_tT_s)$ scores par séquence ; pour l'auto-attention $O(T^2)$ en mémoire et en calcul, ce qui limite la longueur du contexte.
- Les décodeurs RNN restent séquentiels à l'inférence (un token à la fois) ; l'attention ne parallélise que sur les positions source et, à l'entraînement avec teacher forcing, sur la cible si le décodeur n'est pas récurrent.
- Le biais d'exposition et le décalage entraînement/inférence persistent dans tous les générateurs autorégressifs, y compris les LLM.
- L'attention additive a des paramètres supplémentaires ($W_a,U_a,v$) mais fonctionne avec $d_s\neq H$ ; l'attention par produit scalaire est moins coûteuse et se traduit en multiplications matricielles rapides.

## Points clés

- Encodeur-décodeur : $P(y\mid x)=\prod_kP(y_k\mid x,y_{<k})$ ; entraîne avec le teacher forcing, décode en glouton ou par faisceau sur des log-probabilités sommées.
- Un seul vecteur fixe est un goulot d'étranglement de capacité et de chemin de gradient.
- Attention de Bahdanau : $e_{t,i}=v^\top\tanh(W_as_{t-1}+U_ah_i)$, $\alpha_t=\operatorname{softmax}(e_t)$, $c_t=\sum_i\alpha_{t,i}h_i$.
- Passe arrière : $\bar\alpha_i=\bar c^\top h_i$, $\bar e=\alpha\odot(\bar\alpha-\alpha^\top\bar\alpha)$, avec des chemins de valeurs et de clés vers $\bar h_i$.
- Requête/clé/valeur plus la mise à l'échelle par $\sqrt d$ donne l'attention par produit scalaire normalisé ; $Q,K,V$ de la même séquence, c'est l'auto-attention, cœur des transformers.

## Explique-le toi-même

1. Pourquoi le goulot d'étranglement nuit-il davantage aux longues phrases qu'aux courtes, et comment l'attention change-t-elle le chemin du gradient d'un mot cible tardif vers un mot source précoce ?
2. Dérive $\bar e_i=\alpha_i(\bar\alpha_i-\sum_j\alpha_j\bar\alpha_j)$ à partir de la jacobienne du softmax.

## Pour aller plus loin

- Diapositives CNAM RCP217 `cours-tal-03.pdf` (seq2seq, décodage, recherche par faisceau).
- Bibliothèque source : `ai-engineering-from-scratch` phase 05 leçons 09 (sequence-to-sequence) et 10 (attention mechanism), qui citent Bahdanau et al. (2014) et Luong et al. (2015).
- Cours suivant : les transformers, où l'auto-attention remplace la récurrence.

## Critères de maîtrise

Tu sais dériver l'attention additive et sa passe arrière avec les dimensions, expliquer le teacher forcing, le biais d'exposition et la recherche par faisceau, et ton attention NumPy et ton minuscule seq2seq passent les tests des exercices. Tu sais expliquer en quelques phrases comment l'auto-attention par produit scalaire normalisé découle de l'attention de Bahdanau.
