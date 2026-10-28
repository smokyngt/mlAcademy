---
id: nlp.06
title: "Traduction, résumé automatique et question-réponse"
summary: "Comprends comment les systèmes seq2seq traduisent, résument et répondent à des questions, et implémente et critique les métriques servant à les noter (BLEU, ROUGE, correspondance exacte et F1 par token) ainsi que la sélection extractive de segments."
minutes: 80
teaches: [nlp.seq-eval]
requires: [nlp.ngram-lm, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/11-machine-translation/docs/en.md: MT pipeline (tokeniser, encoder-decoder, beam search), BLEU and chrF evaluation. Opening sections read. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/12-text-summarization/docs/en.md: extractive (TextRank) versus abstractive summarisation, ROUGE-1/2/L. Opening sections read. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/13-question-answering/docs/en.md: extractive, open-domain/retrieval-augmented and generative QA, start/end span heads. Opening sections read. READ."
  - "math/cours-tal-03.pdf (PDF p.1-6, French slides): language-model view of translation, encoder-decoder architecture of Cho et al. and Sutskever et al. (2014). READ."
---

## Où ça s'inscrit

Trois des applications du langage les plus visibles partagent une même structure : lire un texte, en écrire un autre. La **traduction** associe une phrase d'une langue à une autre ; le **résumé automatique** comprime un document ; la **question-réponse** (QA) transforme une question et un contexte en une réponse. Tous trois étaient autrefois des domaines de recherche séparés, et aujourd'hui ils sont traités par les mêmes transformers encodeur-décodeur ou décodeur seul ([[llm.02]], [[trf.04]]). Ce qui reste différent, et ce sur quoi cette leçon se concentre, c'est comment les *évaluer* : il n'y a pas de sortie correcte unique, donc nous notons le recouvrement avec des références humaines, et ces scores ont des angles morts bien connus. Tu as besoin des n-grammes de [[nlp.01]], de la notion de précision/rappel de [[nlp.02]], et de la réflexion sur la perplexité de [[nlp.04]]. La recherche pour la QA en domaine ouvert est le sujet de [[sys.03]] ; l'évaluation de systèmes LLM entiers se poursuit dans [[sys.04]].

## L'idée en termes simples

Un **traducteur** lit toute la phrase source, forme une compréhension interne, puis écrit la phrase cible mot par mot, en regardant à chaque fois ce qui a déjà été écrit. La traduction neuronale fait exactement cela avec un *encodeur* qui lit et un *décodeur* qui écrit. La première version largement utilisée (2014) comprimait toute la source en un seul vecteur et laissait un décodeur se dérouler à partir de lui ; l'attention ([[seq.05]]) a ensuite permis au décodeur de regarder en arrière chaque mot source, ce qui a aidé pour les longues phrases.

Un **résumé** peut être construit de deux façons. Un résumé *extractif* est un surligneur : choisir les phrases les plus importantes et les copier telles quelles. Il est toujours grammatical, n'invente jamais de faits, mais ne peut ni reformuler ni fusionner. Un résumé *abstractif* est une réécriture : le modèle écrit de nouvelles phrases. Il peut être fluide et compact, mais peut affirmer quelque chose que le document n'a jamais dit (une « hallucination »).

La **question-réponse extractive** est un surligneur pour une seule réponse : étant donné un passage, marquer où la réponse commence et où elle finit. La QA **générative** écrit la réponse, de mémoire ou à partir de passages récupérés.

Et comment noter ? Nous ne pouvons pas attendre de la machine qu'elle égale un humain mot pour mot, donc nous comptons quelle part de son texte recoupe une ou plusieurs références humaines. C'est peu coûteux et reproductible, mais une paraphrase parfaitement bonne obtient un score bas et une réponse fausse mais fluide peut obtenir un score élevé.

## Traduction seq2seq en un paragraphe

Les diapositives présentent la traduction comme la prédiction de l'élément suivant d'une séquence, puis comme une paire encodeur-décodeur : un réseau encodeur lit la source et son état final initialise un réseau décodeur qui émet la cible token par token, en partant d'un symbole de début et en s'arrêtant à un symbole de fin. L'entraînement est l'entropie croisée de [[nlp.04]] : le décodeur est un modèle de langage conditionnel $P(y_t\mid y_{<t},x)$ sur les tokens cibles étant donnée la source $x$. À l'inférence nous cherchons une sortie de forte probabilité : le décodage glouton prend l'arg max à chaque pas, la **recherche en faisceau** (beam search) garde les $B$ meilleures sorties partielles (typiquement $B=4$ ou $5$ en traduction). Les vocabulaires sont des sous-mots ([[llm.01]]) pour que les mots rares ne soient jamais inconnus. Les systèmes performants d'aujourd'hui sont des transformers encodeur-décodeur multilingues ; les sources mentionnent des modèles de la famille NLLB, mais leurs détails ne sont pas nécessaires ici.

## BLEU

BLEU compare une traduction candidate $c$ avec une ou plusieurs références. Définis la **précision de n-grammes écrêtée** : compte chaque n-gramme du candidat, mais jamais plus de fois que son compte maximal dans une seule référence.

$$
p_n=\frac{\sum_{g\in n\text{-grams}(c)}\min\big(\mathrm{count}_c(g),\ \max_{r}\mathrm{count}_r(g)\big)}{\sum_{g\in n\text{-grams}(c)}\mathrm{count}_c(g)} .
$$

L'écrêtage arrête un candidat dégénéré qui répète un mot courant (« the the the the the the the » aurait sinon une précision de 1). La précision seule récompense les sorties très courtes, donc on multiplie par une **pénalité de brièveté** avec $|c|$ la longueur du candidat et $\rho$ la longueur de la référence la plus proche en longueur :

$$
\mathrm{BP}=\begin{cases}1&|c|>\rho\\ e^{1-\rho/|c|}&|c|\le\rho\end{cases}
\qquad
\mathrm{BLEU}=\mathrm{BP}\cdot\exp\Big(\sum_{n=1}^{N}\tfrac1N\ln p_n\Big).
$$

L'intérieur est la moyenne géométrique des précisions, généralement avec $N=4$. Propriétés : si un $p_n=0$, le BLEU au niveau de la phrase vaut $0$ (une raison pour laquelle BLEU est normalement calculé sur un corpus entier, en additionnant les comptes avant de diviser, ou lissé au niveau de la phrase, par exemple en ajoutant 1 au numérateur et au dénominateur pour $n>1$). Il n'y a pas de terme de rappel ; la pénalité de brièveté en tient lieu.

## ROUGE

ROUGE est la contrepartie orientée rappel, standard pour les résumés. **ROUGE-$n$** compte le recouvrement de n-grammes entre candidat et référence :

$$
R_n=\frac{\text{overlap}}{\#\text{n-grams in reference}},\quad P_n=\frac{\text{overlap}}{\#\text{n-grams in candidate}},\quad F_n=\frac{2P_nR_n}{P_n+R_n},
$$

où le recouvrement est la somme sur les n-grammes du minimum des deux comptes (le même écrêtage). **ROUGE-L** utilise la **plus longue sous-séquence commune** (LCS) : la plus longue suite de tokens apparaissant dans les deux textes dans le même ordre, pas nécessairement adjacents. Avec $\ell=\mathrm{LCS}(c,r)$ : $P=\ell/|c|$, $R=\ell/|r|$, et $F$ comme ci-dessus. La LCS se calcule par le programme dynamique $D_{i+1,j+1}=D_{ij}+1$ si $c_i=r_j$, sinon $\max(D_{i,j+1},D_{i+1,j})$, en $O(|c||r|)$.

## QA extractive : prédiction de segment

Encode ensemble la question et le passage avec un transformer, de sorte que chaque token $j$ du passage obtienne un vecteur $h_j$. Deux petits vecteurs appris $u_s$ et $u_e$ les transforment en un **score de début** $s_j=u_s\cdot h_j$ et un **score de fin** $e_j=u_e\cdot h_j$. L'entraînement utilise l'entropie croisée d'un softmax sur les positions, une fois pour le vrai début et une fois pour la vraie fin. À l'inférence, choisis le segment $(i,j)$ avec $i\le j$ et $j-i+1\le L_{\max}$ maximisant $s_i+e_j$ ; le plafond de longueur empêche le modèle de sélectionner d'énormes segments. Les modèles entraînés avec des questions sans réponse comparent aussi le meilleur segment avec un score « pas de réponse » à une position spéciale. La QA est notée par **correspondance exacte** (EM) et **F1 par token** après normalisation (minuscules, retrait de la ponctuation et des articles), où le F1 traite la prédiction et la référence comme des sacs de tokens.

## Pièges de l'évaluation

- **Cécité aux paraphrases.** `a feline rested on the rug` est une traduction correcte de `the cat sat on the mat`, pourtant elle ne partage presque aucun n-gramme. Les métriques de recouvrement mesurent la similarité de surface, pas le sens. Les métriques apprises (scores à base de plongements ou de modèles) et l'évaluation humaine corrèlent mieux avec le jugement au niveau du système ; elles ne sont pas couvertes ici.
- **Dépendance aux références.** Une seule référence pénalise des alternatives valides ; BLEU n'est comparable qu'entre systèmes notés sur les mêmes références, la même tokenisation et la même implémentation (de petits choix d'implémentation changent le score de points entiers).
- **Bruit au niveau de la phrase.** BLEU est conçu pour des corpus ; les valeurs par phrase sont instables et nulles dès qu'un 4-gramme manque.
- **Effets de longueur et de copie.** Le rappel de ROUGE peut être manipulé par de longues sorties ; des références extractives comme « les trois premières phrases » sont fortes sur les actualités, compare donc toujours avec elles. Pour des résumés abstractifs, un ROUGE élevé ne garantit pas la cohérence factuelle ; vérifie la fidélité séparément.
- **Fuites en QA.** Des réponses mémorisées dans les modèles génératifs, et une normalisation indulgente, gonflent EM/F1.

## Exemple travaillé

Candidat $c$ = `the cat sat on the mat`, référence $r$ = `the cat is on the mat`, tous deux de longueur 6.

| $n$ | n-grammes du candidat | appariés (écrêtés) | $p_n$ |
|---|---|---|---|
| 1 | 6 | 5 (tous sauf « sat ») | 0.8333 |
| 2 | 5 | 3 (the cat, on the, the mat) | 0.6 |
| 3 | 4 | 1 (on the mat) | 0.25 |
| 4 | 3 | 0 | 0 |

La pénalité de brièveté vaut 1 (longueurs égales). BLEU-2 $=\sqrt{0.8333\cdot0.6}=0.7071$. BLEU-4 vaut $0$ parce que $p_4=0$ ; avec un lissage add-one sur $n\ge2$ les précisions deviennent $0.8333,\ 0.6667,\ 0.4,\ 0.25$ et BLEU-4 $=0.4855$. Le candidat dégénéré `the the the the the the the` contre la même référence a une précision de unigrammes écrêtée de $2/7=0.2857$ (la référence contient « the » deux fois). Un candidat `the cat sat` a $p_1=2/3$ mais une pénalité de brièveté $e^{1-6/3}=0.3679$, donc BLEU-1 $=0.2453$. Si une seconde référence `a cat sat on the mat` est ajoutée, le candidat lui correspond exactement et BLEU-2 devient $1$.

ROUGE pour la même paire : ROUGE-1 F1 $=0.8333$ (5 sur 6 de chaque côté), ROUGE-2 F1 $=0.6$, et la LCS est `the cat on the mat`, de longueur 5, donc ROUGE-L F1 $=0.8333$. La paraphrase `a feline rested on the rug` obtient BLEU-2 $=0.2582$ et ROUGE-1 F1 $=0.3333$ contre la même référence bien qu'elle signifie la même chose.

Sélection de segment : les tokens `the iphone launched on june 29 , 2007 .` avec des scores de début culminant à `june` (3.0) et des scores de fin culminant à `2007` (2.5) donnent le segment `june 29 , 2007` (score 5.5) quand $L_{\max}=5$ et seulement `june` quand $L_{\max}=3$. Scores de QA : la prédiction `June 29` contre la référence `June 29, 2007` a EM $0$ et F1 par token $=0.8$ ($P=1$, $R=2/3$). Tout cela est reproduit par les exercices.

## Implémentation

```python
from collections import Counter
import numpy as np

def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

def clipped(c, refs, n):
    cc, mx = ngrams(c, n), Counter()
    for r in refs:
        for g, v in ngrams(r, n).items():
            mx[g] = max(mx[g], v)
    return sum(min(v, mx[g]) for g, v in cc.items()), sum(cc.values())

def lcs(a, b):
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            D[i + 1][j + 1] = D[i][j] + 1 if a[i] == b[j] else max(D[i][j + 1], D[i + 1][j])
    return D[-1][-1]
```

Les fonctions BLEU et ROUGE complètes sont l'exercice. En pratique, utilise `sacrebleu` (tokenisation et signature standardisées, le choix reproductible), `rouge-score`, et le script d'évaluation de SQuAD.

## Formes

- Candidat et références : listes de tokens, de longueurs $|c|$ et $|r|$. Table de LCS : $(|c|+1,|r|+1)$.
- QA : `start_scores`, `end_scores` de forme $(T,)$ pour $T$ tokens de passage ; la matrice de scores de paires $s_i+e_j$ a la forme $(T,T)$ et est masquée à $i\le j<i+L_{\max}$.
- Sortie du décodeur : logits $(T_{\text{target}},|V|)$, une distribution par token généré.

## Idées reçues

- **« BLEU mesure la qualité de traduction. »** Il mesure le recouvrement de n-grammes avec des références ; un BLEU élevé n'est ni nécessaire ni suffisant pour une bonne traduction.
- **« Compare les nombres BLEU d'un article à l'autre. »** Valable seulement avec des références, une tokenisation et un outil de notation identiques.
- **« ROUGE récompense les bons résumés. »** Il récompense le recouvrement de mots ; la fidélité et la cohérence demandent des vérifications séparées.
- **« L'abstractif est strictement meilleur que l'extractif. »** L'extractif n'hallucine jamais et est une référence solide.
- **« L'EM est à lui seul une métrique de QA équitable. »** Des réponses équivalentes formulées différemment échouent à l'EM ; rapporte aussi le F1.

## Points clés à retenir

- La traduction, le résumé et la QA générative sont des modèles de langage conditionnels entraînés par entropie croisée et décodés par recherche gloutonne ou en faisceau.
- BLEU = pénalité de brièveté multipliée par la moyenne géométrique des précisions de n-grammes écrêtées ; ROUGE-$n$ et ROUGE-L sont des scores F de recouvrement (n-grammes, LCS).
- La QA extractive choisit le segment maximisant le score de début plus le score de fin sous un plafond de longueur ; note avec EM et F1 par token.
- Les métriques de recouvrement sont des indicateurs peu coûteux : traite-les avec des références de base, plusieurs références et des vérifications humaines.

## Explique-le toi-même

1. Pourquoi l'écrêtage est-il nécessaire dans BLEU et pourquoi y a-t-il une pénalité de brièveté mais pas de terme de rappel ?
2. Donne une paire de phrases à fort recouvrement de sens mais à BLEU bas, et explique quelle autre évaluation tu ajouterais.

## Pour aller plus loin

- Papineni et al., « BLEU: a method for automatic evaluation of machine translation » (2002) ; Lin, « ROUGE: a package for automatic evaluation of summaries » (2004) ; Post, « A call for clarity in reporting BLEU scores » (2018).
- Rajpurkar et al., SQuAD (2016) pour la QA par segments.

## Critères de maîtrise

- Tu sais calculer à la main les précisions écrêtées, la pénalité de brièveté et BLEU pour une petite paire.
- Tu sais calculer ROUGE-1, ROUGE-2 et ROUGE-L, y compris une LCS.
- Tu sais décoder le meilleur segment sous un plafond de longueur et calculer EM/F1.
- Tu sais nommer au moins trois façons dont les métriques de recouvrement induisent en erreur.
