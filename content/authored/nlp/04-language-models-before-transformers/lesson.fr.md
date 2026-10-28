---
id: nlp.04
title: "Les modèles de langage avant les transformers"
summary: "Construis des modèles de langage n-grammes à partir de comptes, corrige le problème des comptes nuls avec le lissage et l'interpolation, évalue les modèles avec la perplexité, et vois pourquoi les modèles de langage neuronaux les ont remplacés."
minutes: 70
teaches: [nlp.ngram-lm]
requires: [prob.basics, info.entropy, alg.exp-log]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer/docs/en.md: n-gram probabilities, zero-count problem, list of smoothing methods, perplexity, Shannon's prediction game. READ."
  - "math/cours-tal-01.pdf (PDF p.28): n-gram counts and the zero-count problem. READ."
  - "math/cours-tal-03.pdf (PDF p.1-6, French slides): translation seen as predicting the next item of a sequence, and the encoder-decoder idea. Only the opening pages were read. READ."
---

## Où ça s'inscrit

Un **modèle de langage** (LM) attribue une probabilité à une phrase, ou de façon équivalente prédit le mot suivant à partir des précédents. Cela paraît abstrait, mais les systèmes de reconnaissance vocale, les correcteurs orthographiques, l'autocomplétion et les anciens systèmes de traduction automatique en utilisaient tous un pour choisir parmi des sorties candidates : « recognise speech » est bien plus probable que « wreck a nice beach ». Les grands modèles de langage modernes ([[llm.02]]) sont la même idée passée à l'échelle avec des transformers ; le nombre d'évaluation, la **perplexité**, est le même que celui que tu verras dans leurs articles. Cette leçon construit le modèle à base de comptes qui a dominé des années 1980 aux années 2010, et le problème qui l'a tué (les comptes épars) montre pourquoi les modèles neuronaux ont été inventés. Elle utilise les n-grammes de [[nlp.01]] et les log-probabilités de [[nlp.02]].

## L'idée en termes simples

Joue à un jeu de devinettes : je te montre « the cat sat on the » et tu devines le mot suivant. Tu dis « mat », ou « floor », ou « sofa », parce que tu as déjà vu ce genre de phrase. Un modèle n-gramme joue à ce jeu par pure mémoire : il se souvient, à partir d'un gros tas de texte, de la fréquence à laquelle chaque mot a suivi chaque court contexte, et répond avec ces fréquences relatives. Un modèle bigramme se souvient d'un mot précédent, un trigramme de deux.

Deux choses tournent mal. D'abord, la mémoire a des limites : avec un contexte assez long, rien n'est jamais vu deux fois, donc le modèle doit oublier les mots lointains. Ensuite, et c'est pire, le monde est plus grand que le tas de texte : des paires de mots parfaitement bonnes ne sont jamais apparues dans ton tas, et le modèle dit qu'elles sont impossibles. Le **lissage** est l'art de réserver un peu de croyance pour ce que tu n'as pas vu. La **perplexité** note tout le jeu : entre combien d'options également probables, en moyenne, le modèle choisissait-il ?

## Définitions

Une phrase $w_1,\dots,w_m$ est complétée par un symbole de début $\langle s\rangle$ et un symbole de fin $\langle/s\rangle$ pour que le modèle apprenne aussi où les phrases commencent et finissent. Par la règle de la chaîne des probabilités (exacte, sans hypothèse)

$$
P(w_1,\dots,w_m)=\prod_{i=1}^{m+1}P(w_i\mid w_1,\dots,w_{i-1}).
$$

L'**hypothèse de Markov** d'ordre $n-1$ ne garde que les $n-1$ derniers mots de contexte. Pour un bigramme ($n=2$) :

$$
P(w_i\mid w_{1:i-1})\approx P(w_i\mid w_{i-1}).
$$

**Estimation du maximum de vraisemblance.** Avec $c(\cdot)$ le compte dans le corpus d'entraînement,

$$
P_{\mathrm{MLE}}(w\mid v)=\frac{c(v,w)}{c(v)} .
$$

C'est la même estimation par fréquence relative que dans Naive Bayes ([[nlp.02]]), dérivée de la même façon (maximiser la vraisemblance d'entraînement sous normalisation).

## Lissage

Sous le MLE, tout bigramme absent des données d'entraînement a une probabilité $0$, donc toute phrase de test en contenant un obtient la probabilité $0$ et une perplexité infinie. Les vrais corpus ont une longue queue : une grande fraction des bigrammes d'un nouveau texte n'a jamais été vue. Nous devons déplacer de la masse de probabilité des événements vus vers les événements non vus.

**Add-$k$ (Laplace pour $k=1$).** Fais comme si chacun des $|V|$ mots avait suivi chaque contexte $k$ fois de plus :

$$
P_{\mathrm{add}\text{-}k}(w\mid v)=\frac{c(v,w)+k}{c(v)+k|V|}.
$$

Le dénominateur est $\sum_w (c(v,w)+k)$, c'est donc une distribution valide. Avec $|V|$ de l'ordre de dizaines de milliers, $k=1$ vole beaucoup trop de masse aux événements observés ; utilise donc un petit $k$ (réglé sur des données de validation) ou de meilleures méthodes.

**Interpolation.** Mélange des modèles d'ordres différents, pour qu'une estimation fiable d'ordre bas comble là où celle d'ordre élevé n'est pas fiable :

$$
P_{\mathrm{interp}}(w\mid v)=\lambda\,P_{\mathrm{MLE}}(w\mid v)+(1-\lambda)\,P_{\mathrm{MLE}}(w),\qquad 0\le\lambda\le1 .
$$

C'est un mélange de deux distributions, donc lui-même une distribution. Choisis $\lambda$ sur du texte de validation (les poids peuvent dépendre du compte du contexte).

**Repli (backoff).** Utilise l'estimation d'ordre le plus élevé si son compte est positif ; sinon « replie-toi » sur l'ordre inférieur suivant. Le backoff de Katz décote les comptes d'ordre élevé pour que le total reste 1. Une variante peu coûteuse et non normalisée utilisée à l'échelle du web, le *stupid backoff*, multiplie simplement par une constante (environ $0.4$) à chaque étape de repli ; elle donne des scores, pas des probabilités. La méthode classique la plus réussie, **Kneser-Ney**, combine une décote absolue avec une distribution d'ordre inférieur plus intelligente qui compte dans combien de contextes différents un mot apparaît (l'exemple de la source : « Francisco » est fréquent mais ne suit presque que « San », donc il ne devrait pas être un mot probable dans un contexte neuf). Nous ne la dérivons pas ici.

## Perplexité

Pour un jeu de test de $N$ tokens (en comptant les symboles de fin), la **log-vraisemblance négative moyenne** (entropie croisée en nats) est

$$
H=-\frac{1}{N}\sum_{i=1}^{N}\ln P(w_i\mid \text{context}_i),
$$

et la **perplexité** est $\mathrm{PP}=e^{H}$ (égale à $2^{H_{\text{bits}}}$ quand les logs sont en base 2). Interprétation : un modèle qui répartit uniformément sa probabilité sur $b$ mots à chaque pas a une perplexité exactement égale à $b$, car $H=\ln b$. La perplexité est donc le facteur de branchement effectif ; plus bas, c'est mieux. Elle ne peut être comparée qu'entre modèles de même vocabulaire et de même tokenisation, et le jeu de test ne doit avoir servi à rien régler. Une probabilité nulle n'importe où la rend infinie.

## Modèles de langage neuronaux

La table n-gramme traite « cat » et « dog » comme des symboles sans lien, donc voir « the cat sat on the mat » n'apprend rien sur « the dog sat on the mat ». Un LM neuronal remplace la table par une fonction : associer chacun des $n-1$ mots précédents à un vecteur de plongement dense ([[seq.02]]), les concaténer, les passer dans une couche cachée, et produire un softmax sur le vocabulaire. Des mots similaires obtiennent des plongements similaires et donc des prédictions similaires, de sorte que le modèle **généralise** à des contextes non vus sans aucun lissage, et le nombre de paramètres croît avec la taille du plongement plutôt qu'en $|V|^n$. Les réseaux récurrents ([[seq.03]]) suppriment la fenêtre fixe ; les diapositives françaises des sources présentent la traduction comme exactement cette prédiction de l'élément suivant, avec un agencement encodeur-décodeur. Les transformers ([[llm.02]]) étendent le contexte à des milliers de tokens. La perte d'entraînement de tous est l'entropie croisée $H$ ci-dessus, donc la perplexité est $e^{\text{loss}}$.

## Exemple travaillé

Corpus d'entraînement (chacun complété par $\langle s\rangle$ et $\langle/s\rangle$) : `the cat sat on the mat`, `the dog sat on the rug`, `the cat saw the dog`. Le vocabulaire des tokens prédits compte $|V|=9$ (the, cat, sat, on, mat, dog, rug, saw et $\langle/s\rangle$). Comptes : $c(\text{the})=6$ comme contexte, $c(\text{the},\text{cat})=2$.

- $P_{\mathrm{MLE}}(\text{cat}\mid\text{the})=2/6=0.3333$.
- Add-1 : $(2+1)/(6+9)=0.2$. Les événements vus perdent de la masse : $0.333\to0.2$. La paire (the, rug) a été vue une fois, donc le MLE donne $1/6=0.1667$ et add-1 donne $(1+1)/15=0.1333$.
- Interpolation avec $\lambda=0.7$ : $P(\text{rug}\mid\text{the})=0.7\cdot\frac16+0.3\cdot\frac{1}{20}=0.1317$ (il y a $20$ tokens prédits au total).

Perplexité de la phrase de test `the mat saw the cat`. Son bigramme (mat, saw) n'apparaît jamais à l'entraînement. MLE : infinie. Add-1 : $6.289$. Add-0.1 : $5.396$. Interpolé ($\lambda=0.7$) : $6.396$. Perplexité de `the dog sat on the mat`, dont tous les bigrammes ont été vus : MLE $1.669$, add-1 $4.569$, add-0.1 $2.168$ : add-1 fait paraître un bon modèle bien pire parce qu'il déplace tant de masse vers des événements qui n'arrivent jamais. Le modèle uniforme sur 9 mots a une perplexité de 9 ; tous les modèles lissés sont bien en dessous. Toutes les valeurs viennent du code de l'exercice.

## Implémentation

```python
import numpy as np
from collections import Counter

def build_counts(sentences):
    uni, bi, ctx = Counter(), Counter(), Counter()
    for s in sentences:
        t = ["<s>"] + s.split() + ["</s>"]
        uni.update(t[1:])                     # predicted tokens
        ctx.update(t[:-1])                    # tokens used as context
        bi.update(zip(t[:-1], t[1:]))
    return uni, bi, ctx

def perplexity(sentences, prob):
    ll, n = 0.0, 0
    for s in sentences:
        t = ["<s>"] + s.split() + ["</s>"]
        for a, b in zip(t[:-1], t[1:]):
            p = prob(a, b)
            ll += np.log(p) if p > 0 else -np.inf
            n += 1
    return float(np.exp(-ll / n))
```

`nltk.lm` de NLTK et KenLM implémentent les modèles n-grammes lissés de production (KenLM avec Kneser-Ney). Calcule toujours en espace logarithmique.

## Formes

Pour un vocabulaire de taille $V$, une table de bigrammes a $V^2$ cases ($V\times V$), une table de trigrammes $V^3$ ; cette explosion explique pourquoi les modèles n-grammes s'arrêtent à $n\approx5$ et pourquoi la plupart des cases sont vides. Un LM neuronal avec une taille de plongement $d$ et une fenêtre $n-1$ a environ $Vd+(n-1)d\,h+hV$ paramètres pour une taille cachée $h$.

## Idées reçues

- **« Probabilité nulle signifie impossible. »** Cela signifie seulement non observé ; le langage est ouvert.
- **« Une perplexité plus basse signifie toujours une meilleure application. »** Elle n'est comparable que sur le même vocabulaire et le même tokeniseur, et ne garantit pas une meilleure qualité en aval.
- **« Le lissage add-one est le standard. »** C'est un outil pédagogique ; il se comporte mal avec de grands vocabulaires.
- **« Règle le lissage sur le jeu de test. »** Règle $k$ et $\lambda$ sur un jeu de validation.
- **« Un $n$ plus grand est toujours meilleur. »** Un ordre plus élevé signifie des comptes plus épars ; au-delà d'un certain point, il faut plus de données et un meilleur lissage.

## Points clés à retenir

- Un LM n-gramme est la règle de la chaîne plus une hypothèse de Markov, estimé par fréquence relative.
- Les n-grammes non vus imposent le lissage : add-$k$, interpolation, repli, Kneser-Ney.
- La perplexité $=e^{H}$ est l'entropie croisée par token exponentiée : le nombre effectif de choix.
- Les LM neuronaux remplacent la table de comptes par des plongements, généralisent entre mots similaires, et sont entraînés sur la même entropie croisée.

## Explique-le toi-même

1. Pourquoi un seul bigramme non vu rend-il infinie la perplexité d'un modèle MLE, et comment l'interpolation avec un modèle unigramme y remédie-t-elle ?
2. Un ami dit qu'un modèle de langage de perplexité 10 « connaît 10 mots ». Quelle est la lecture correcte ?

## Pour aller plus loin

- Jurafsky et Martin, *Speech and Language Processing*, chapitre sur les modèles de langage n-grammes (lissage, Kneser-Ney, perplexité).
- Bengio et al., « A neural probabilistic language model » (2003).

## Critères de maîtrise

- Tu sais calculer à la main les probabilités MLE, add-$k$ et interpolées d'un bigramme.
- Tu sais calculer la perplexité à partir de probabilités par token et l'interpréter.
- Tu sais énoncer la faiblesse des tables de comptes que corrigent les LM neuronaux.
