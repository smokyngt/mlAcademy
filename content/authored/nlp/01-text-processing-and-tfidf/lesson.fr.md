---
id: nlp.01
title: "Traitement du texte et représentations classiques"
summary: "Transforme des chaînes brutes en tokens puis en vecteurs creux, en dérivant pas à pas TF-IDF et la similarité cosinus et en les implémentant avec NumPy."
minutes: 70
teaches: [nlp.text-preprocessing, nlp.tfidf]
requires: [alg.exp-log, la.dot]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/01-text-processing/docs/en.md: tokenisation, stemming versus lemmatisation, failure modes. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/docs/en.md: bag of words, smoothed IDF, L2 normalisation. READ."
  - "math/cours-tal-01.pdf (PDF p.1-7, 28, 30, 32-33, French slides): overview of NLP tasks, difficulty of defining a word, one-hot vectors with unknown and padding tokens, n-gram counting. READ."
---

## Où ça s'inscrit

Tout système qui traite le langage, d'un filtre anti-spam à un moteur de recherche en passant par un grand modèle de langage, commence par le même obstacle : un ordinateur stocke `"The cats weren't running."` sous forme de chaîne, mais tous les modèles de ce cours consomment des nombres. Cette leçon couvre les deux étapes classiques qui comblent l'écart : le **traitement du texte** (découper la chaîne en unités propres) et la **représentation** (transformer les unités en un vecteur). L'approche est ancienne et toujours solide : pour une tâche étroite comme « ce ticket concerne-t-il la facturation ? », un vecteur TF-IDF plus un modèle linéaire est souvent difficile à battre, et c'est la référence qu'il faut dépasser avant de chercher quelque chose de plus lourd.

Tu réutiliseras cette leçon dans les suivantes : [[nlp.02]] classe ces vecteurs, [[nlp.04]] compte les mêmes n-grammes pour construire des modèles de langage, et [[nlp.05]] les modèles de sujets lisent des comptages de sacs de mots. Les représentations denses apprises qui remplacent ces vecteurs creux sont dans [[seq.02]], et les tokeniseurs de sous-mots utilisés par les LLM modernes sont dans [[llm.01]].

## L'idée en termes simples

Imagine que tu doives classer mille lettres par sujet sans les lire attentivement. D'abord tu mets chaque lettre au propre : ignorer les majuscules, séparer les mots de la ponctuation, peut-être traiter « running » et « runs » comme le même mot. C'est le **traitement du texte**. Puis tu décris chaque lettre par une feuille de pointage : une ligne par mot du dictionnaire, avec un compte à côté. L'ordre est perdu, seul le pointage reste ; c'est un **sac de mots**, comme si tu vidais un sac de jetons de Scrabble sur la table.

Un pointage brut a un défaut : le mot « the » obtient un score élevé dans toutes les lettres et ne t'apprend rien, alors que « mortgage » apparaît dans peu de lettres et t'apprend beaucoup. Tu sous-pondères donc les mots présents partout et tu sur-pondères les mots rares globalement mais fréquents dans cette lettre. Cette repondération est **TF-IDF**. Enfin, tu compares les lettres par l'angle entre leurs feuilles de pointage, pour qu'une longue lettre ne soit pas considérée comme plus similaire à tout simplement parce qu'elle a plus de mots.

## Traitement du texte

Les étapes courantes, dans l'ordre :

1. **Normalisation.** Passer en minuscules, unifier les formes Unicode (accents, guillemets courbes), éventuellement retirer le balisage. Les minuscules fusionnent « Apple » et « apple », ce qui aide les comptes, mais perd le signal entreprise-contre-fruit. Chaque étape de normalisation jette de l'information, choisis donc selon la tâche.
2. **Tokenisation.** Découper la chaîne en tokens. Un « mot » est plus difficile à définir qu'il n'y paraît : les diapositives françaises des sources pointent « pomme de terre » (trois mots écrits, un concept), « aujourd'hui » (une apostrophe à l'intérieur d'un mot) et les mots composés allemands. Une règle simple est une expression régulière pour des suites de lettres et de chiffres qui autorise une apostrophe interne. Les vrais systèmes gèrent aussi les URL, les hashtags et les emojis. Les LLM modernes évitent le problème en apprenant des unités de sous-mots ([[llm.01]]).
3. **Mots vides** (optionnel). Retirer les mots extrêmement courants. Avec TF-IDF c'est surtout inutile parce que l'IDF les réduit déjà, et pour des tâches comme le sentiment, retirer « not » serait nuisible.
4. **Racinisation ou lemmatisation** (optionnel), pour fusionner les formes d'un mot.

La **racinisation** (stemming) coupe les suffixes avec des règles de réécriture ordonnées (le stemmer de Porter est le classique) : `caresses -> caress`, `cats -> cat`. Elle est rapide et n'a pas besoin de dictionnaire, mais elle est brutale : `ponies -> poni`, et des mots différents peuvent s'effondrer ensemble (la source cite `organization -> organ`). La **lemmatisation** associe un mot à sa forme du dictionnaire à l'aide d'un lexique et de la catégorie grammaticale du mot : `ran -> run` (verbe), `better -> good` (adjectif). Elle est plus précise mais plus lente, et elle a besoin d'un étiqueteur de catégories grammaticales ([[nlp.03]]) car `saw` est `see` comme verbe et `saw` comme nom. Règle empirique : raciniser pour l'indexation de recherche et la classification approximative ; lemmatiser quand le sens de la sortie compte.

Deux marqueurs pratiques apparaissent dans les diapositives : un token `<UNK>` pour les mots jamais vus à l'entraînement et un token `<PAD>` pour donner la même longueur aux séquences.

## Sac de mots et vecteurs creux

Fixe un vocabulaire $V$ de $|V|$ tokens distincts, triés pour que chaque token ait un indice de colonne. Pour un corpus de $N$ documents, la **matrice de comptage** est

$$
X\in\mathbb{N}^{N\times |V|},\qquad X_{dj}=\text{number of times token } j \text{ occurs in document } d.
$$

Les vrais vocabulaires ont de $10^4$ à $10^6$ entrées mais un document en utilise quelques dizaines, donc presque toutes les entrées sont nulles : la matrice est **creuse** et se stocke comme une liste de triplets (ligne, colonne, valeur) (scikit-learn utilise `scipy.sparse`). L'implémentation ci-dessous utilise un tableau dense parce que le vocabulaire jouet est minuscule. Le vecteur one-hot des diapositives est le cas particulier d'un seul mot ; le sac de mots est la somme des vecteurs one-hot des tokens du document.

Les **n-grammes** récupèrent un peu de l'ordre perdu : un bigramme est une paire de tokens adjacents, et `("not", "good")` est une caractéristique à part entière. Le prix est la croissance du vocabulaire : avec $|V|$ mots il y a jusqu'à $|V|^n$ $n$-grammes possibles, bien que seule une petite fraction apparaisse. Les n-grammes reviennent au cœur de [[nlp.04]].

## Dérivation : TF-IDF

On veut un poids $w_{dj}$ qui soit élevé quand le token $j$ est fréquent dans le document $d$ et rare dans le corpus.

**Fréquence du terme.** Utilise le compte brut $\mathrm{tf}_{dj}=X_{dj}$ (des variantes divisent par la longueur du document ou utilisent $\log(1+\mathrm{tf})$ ; la normalisation finale supprime de toute façon l'essentiel de l'effet de longueur).

**Fréquence documentaire.** $\mathrm{df}_j$ est le nombre de documents contenant le token $j$, entre $1$ et $N$. La fraction $\mathrm{df}_j/N$ est la probabilité qu'un document au hasard contienne le token. Un token présent dans chaque document a une fraction de $1$ et aucun pouvoir discriminant.

**Fréquence inverse de document.** Définis $\mathrm{idf}_j$ comme l'auto-information de « ce token apparaît » (compare avec la leçon sur l'entropie) : $-\log(\mathrm{df}_j/N)=\log(N/\mathrm{df}_j)$. Elle vaut $0$ pour un token omniprésent et croît lentement (à cause du logarithme) pour les plus rares ; sans le log, un token vu dans un seul document pèserait $N$ fois un token vu partout, ce qui est trop extrême.

Deux correctifs pratiques donnent la version utilisée par scikit-learn :

- Le lissage, pour éviter la division par zéro pour un token avec $\mathrm{df}=0$ (possible quand on ajuste le vocabulaire sur un corpus et qu'on transforme un autre) : ajouter un au numérateur et au dénominateur.
- Ajouter un en dehors du log, pour qu'un token présent dans chaque document garde un poids de $1$ plutôt que d'être effacé :

$$
\mathrm{idf}_j=\ln\frac{1+N}{1+\mathrm{df}_j}+1 .
$$

**Pondération et normalisation.** Le poids non normalisé est $\mathrm{tf}_{dj}\cdot\mathrm{idf}_j$. Puis divise chaque ligne par sa norme euclidienne pour que chaque vecteur de document ait une longueur de 1 :

$$
\tilde w_{dj}=\frac{\mathrm{tf}_{dj}\,\mathrm{idf}_j}{\sqrt{\sum_{k}(\mathrm{tf}_{dk}\,\mathrm{idf}_k)^2}} .
$$

**Similarité cosinus.** Pour deux vecteurs $a,b$, le cosinus de l'angle entre eux est $\frac{a\cdot b}{\|a\|\|b\|}$. Après normalisation L2 les dénominateurs valent $1$, donc la similarité des documents $d$ et $e$ est simplement le produit scalaire $\tilde w_d\cdot\tilde w_e$. Toute la matrice de similarité est $\tilde W\tilde W^\top$ de forme $(N,N)$.

## Exemple travaillé

Trois documents : `"The cat sat"`, `"The dog sat"`, `"The cat ran"`. Tokenisé et passé en minuscules, le vocabulaire trié est `[cat, dog, ran, sat, the]`, donc $|V|=5$, $N=3$ et la matrice de comptage (de forme $3\times5$) est

| doc | cat | dog | ran | sat | the |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 | 1 |
| 3 | 1 | 0 | 1 | 0 | 1 |

Les fréquences documentaires sont $[2,1,1,2,3]$, donc $\mathrm{idf}=[\ln(4/3)+1,\ \ln(4/2)+1,\ \ln 2+1,\ \ln(4/3)+1,\ \ln(4/4)+1]=[1.2877,\,1.6931,\,1.6931,\,1.2877,\,1.0000]$. « the » reçoit le poids minimal $1$, « dog » et « ran » le poids maximal.

Le document 1 non normalisé : $(1.2877,0,0,1.2877,1)$ de norme $\sqrt{2\cdot1.2877^2+1}=2.0776$, d'où $(0.6198,0,0,0.6198,0.4813)$. Le document 2 se normalise en $(0,0.7203,0,0.5478,0.4254)$ et le document 3 en $(0.5478,0,0.7203,0,0.4254)$.

Similarités cosinus : les docs 1 et 2 partagent « sat » et « the » : $0.6198\cdot0.5478+0.4813\cdot0.4254=0.5443$. Les docs 1 et 3 partagent « cat » et « the », également $0.5443$ par symétrie. Les docs 2 et 3 ne partagent que « the » : $0.4254^2=0.1810$. La paire n'ayant qu'un mot vide en commun est correctement la moins similaire. L'exercice reproduit chacun de ces nombres.

## Implémentation

```python
import re
import numpy as np

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())

def count_matrix(docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for d, tokens in enumerate(docs):
        for w in tokens:
            if w in index:           # unknown words are dropped
                X[d, index[w]] += 1
    return X

def tfidf(X):
    N = X.shape[0]
    df = (X > 0).sum(axis=0)                  # shape (V,)
    idf = np.log((1 + N) / (1 + df)) + 1      # shape (V,)
    T = X * idf                               # (N, V) * (V,) broadcasts
    norm = np.linalg.norm(T, axis=1, keepdims=True)
    return T / np.where(norm == 0, 1, norm)   # empty documents stay zero
```

## Formes

- `docs` : liste de $N$ listes de tokens ; `vocab` : liste de $V$ chaînes.
- `X`, `T`, `tfidf(X)` : $(N,V)$. `df`, `idf` : $(V,)$.
- Similarité `W @ W.T` : $(N,N)$, avec des uns sur la diagonale.

En production, utilise `sklearn.feature_extraction.text.TfidfVectorizer`, qui renvoie une matrice creuse et applique par défaut cette formule lissée et la normalisation L2. Ajuste le vocabulaire et l'IDF uniquement sur les données d'entraînement, puis transforme les nouvelles données avec les valeurs stockées.

## Idées reçues

- **« TF-IDF comprend le sens. »** Il ne fait que compter. « Car » et « automobile » sont des colonnes sans lien ; c'est le fossé que comblent les plongements ([[seq.02]]).
- **« Toujours retirer les mots vides et raciniser. »** Pour le sentiment, les négations et les petits mots fonctionnels comptent ; pour les transformers, le texte brut fonctionne le mieux.
- **« Ajuste le vectoriseur sur toutes les données. »** Calculer l'IDF sur le jeu de test fait fuir de l'information du test vers les caractéristiques.
- **« Un sac de mots ne peut pas capturer la négation. »** Des unigrammes simples ne le peuvent pas, mais des bigrammes comme `not good` le peuvent, au prix d'un vocabulaire plus grand.
- **« La lemmatisation n'a pas besoin de contexte. »** Sans la catégorie grammaticale, les formes ambiguës sont mal lemmatisées.

## Points clés à retenir

- Le traitement du texte est une suite de choix avec perte ; choisis-les par tâche et applique le même pipeline au texte d'entraînement et de test.
- Un sac de mots est une matrice de comptage creuse de forme $(N,|V|)$.
- $\mathrm{idf}_j=\ln\frac{1+N}{1+\mathrm{df}_j}+1$ réduit les mots omniprésents ; la normalisation L2 fait de la similarité cosinus un simple produit scalaire.
- Les n-grammes ajoutent l'ordre local au prix de la croissance du vocabulaire.

## Explique-le toi-même

1. Pourquoi un mot qui apparaît dans chaque document reçoit-il un petit poids, et pourquoi gardons-nous quand même un « +1 » pour qu'il ne soit pas exactement nul ?
2. Un ami dit « la racinisation et la lemmatisation sont la même chose ». Donne deux entrées sur lesquelles elles divergent et dis quel outil tu utiliserais pour un index de recherche.

## Pour aller plus loin

- Jurafsky et Martin, *Speech and Language Processing*, les chapitres sur les expressions régulières, la tokenisation et la sémantique vectorielle.
- La section du guide utilisateur de scikit-learn sur l'extraction de caractéristiques de texte.

## Critères de maîtrise

- Tu sais calculer à la main une matrice TF-IDF et une similarité cosinus pour trois documents courts.
- Tu sais dire quand la racinisation nuit et quand la lemmatisation a besoin d'un étiqueteur.
- Tu sais expliquer pourquoi le vectoriseur est ajusté uniquement sur le découpage d'entraînement.
