---
id: nlp.01
title: Text processing and classical representations
summary: Turn raw strings into tokens and then into sparse vectors, deriving TF-IDF and cosine similarity step by step and implementing them in NumPy.
minutes: 70
teaches: [nlp.text-preprocessing, nlp.tfidf]
requires: [alg.exp-log, la.dot]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/01-text-processing/docs/en.md: tokenisation, stemming versus lemmatisation, failure modes. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/docs/en.md: bag of words, smoothed IDF, L2 normalisation. READ."
  - "math/cours-tal-01.pdf (PDF p.1-7, 28, 30, 32-33, French slides): overview of NLP tasks, difficulty of defining a word, one-hot vectors with unknown and padding tokens, n-gram counting. READ."
---

## Where this fits

Every system that handles language, from a spam filter to a search engine to a large language model, begins with the same obstacle: a computer stores `"The cats weren't running."` as a string, but every model in this course consumes numbers. This lesson covers the two classical steps that bridge the gap: **text processing** (cut the string into clean units) and **representation** (turn the units into a vector). The approach is old and still strong: for a narrow task such as "is this ticket about billing?", a TF-IDF vector plus a linear model is often hard to beat, and it is the baseline you should beat before reaching for something heavier.

You will use this lesson in the next ones: [[nlp.02]] classifies these vectors, [[nlp.04]] counts the same n-grams to build language models, and [[nlp.05]] topic models read bag-of-words counts. The dense learned representations that replace these sparse vectors are in [[seq.02]], and the subword tokenisers used by modern LLMs are in [[llm.01]].

## The idea in plain words

Imagine you must file a thousand letters by topic without reading them closely. First you tidy each letter: ignore capital letters, separate words from punctuation, perhaps treat "running" and "runs" as the same word. That is **text processing**. Then you describe each letter by a tally sheet: one line per word of the dictionary, with a count next to it. Order is lost, only the tally remains; this is a **bag of words**, like emptying a bag of Scrabble tiles on the table.

A raw tally has a flaw: the word "the" scores high in every letter and tells you nothing, whereas "mortgage" appears in few letters and tells you a lot. So you down-weight words that are everywhere and up-weight words that are rare overall but frequent in this letter. That re-weighting is **TF-IDF**. Finally you compare letters by the angle between their tally sheets, so that a long letter is not considered more similar to everything just because it has more words.

## Text processing

The common steps, in order:

1. **Normalisation.** Lowercase, unify Unicode forms (accents, curly quotes), perhaps strip markup. Lowercasing merges "Apple" and "apple", which helps counts, but it loses the company-versus-fruit signal. Every normalisation step throws information away, so choose by task.
2. **Tokenisation.** Split the string into tokens. A "word" is harder to define than it looks: the French slides in the sources point at "pomme de terre" (three written words, one concept), "aujourd'hui" (an apostrophe inside a word) and German compounds. A simple rule is a regular expression for runs of letters and digits that allows an inner apostrophe. Real systems also handle URLs, hashtags and emojis. Modern LLMs avoid the problem by learning subword units ([[llm.01]]).
3. **Stop words** (optional). Remove extremely common words. With TF-IDF this is mostly unnecessary because IDF already shrinks them, and for tasks such as sentiment removing "not" would be harmful.
4. **Stemming or lemmatisation** (optional), to merge word forms.

**Stemming** chops suffixes with ordered rewrite rules (the Porter stemmer is the classic): `caresses -> caress`, `cats -> cat`. It is fast and needs no dictionary, but it is blunt: `ponies -> poni`, and different words can collapse together (the source cites `organization -> organ`). **Lemmatisation** maps a word to its dictionary form using a lexicon and the word's part of speech: `ran -> run` (verb), `better -> good` (adjective). It is more accurate but slower, and it needs a part-of-speech tagger ([[nlp.03]]) because `saw` is `see` as a verb and `saw` as a noun. Rule of thumb: stem for search indexing and rough classification; lemmatise when the output meaning matters.

Two practical markers appear in the slides: an `<UNK>` token for words never seen in training and a `<PAD>` token to make sequences the same length.

## Bag of words and sparse vectors

Fix a vocabulary $V$ of $|V|$ distinct tokens, sorted so each token has a column index. For a corpus of $N$ documents the **count matrix** is

$$
X\in\mathbb{N}^{N\times |V|},\qquad X_{dj}=\text{number of times token } j \text{ occurs in document } d.
$$

Real vocabularies have $10^4$ to $10^6$ entries but a document uses a few dozen, so almost every entry is zero: the matrix is **sparse** and is stored as a list of (row, column, value) triples (scikit-learn uses `scipy.sparse`). The implementation below uses a dense array because the toy vocabulary is tiny. The one-hot vector of the slides is the special case of a single word; the bag of words is the sum of the one-hot vectors of the document's tokens.

**N-grams** recover a little of the lost order: a bigram is a pair of adjacent tokens, and `("not", "good")` is a feature on its own. The price is vocabulary growth: with $|V|$ words there are up to $|V|^n$ possible $n$-grams, though only a small fraction occur. N-grams reappear as the heart of [[nlp.04]].

## Derivation: TF-IDF

We want a weight $w_{dj}$ that is high when token $j$ is frequent in document $d$ and rare across the corpus.

**Term frequency.** Use the raw count $\mathrm{tf}_{dj}=X_{dj}$ (variants divide by document length or use $\log(1+\mathrm{tf})$; the final normalisation removes most of the length effect anyway).

**Document frequency.** $\mathrm{df}_j$ is the number of documents containing token $j$, between $1$ and $N$. The fraction $\mathrm{df}_j/N$ is the probability that a random document contains the token. A token present in every document has fraction $1$ and no discriminating power.

**Inverse document frequency.** Define $\mathrm{idf}_j$ as the self-information of "this token occurs" (compare the entropy lesson): $-\log(\mathrm{df}_j/N)=\log(N/\mathrm{df}_j)$. It is $0$ for a ubiquitous token and grows slowly (because of the logarithm) for rarer ones; without the log, a token seen in one document would weigh $N$ times a token seen everywhere, which is too extreme.

Two practical fixes give the version used by scikit-learn:

- Smoothing, so no division by zero for a token with $\mathrm{df}=0$ (possible when you fit the vocabulary on one corpus and transform another): add one to numerator and denominator.
- Add one outside the log, so a token present in every document keeps weight $1$ rather than being erased:

$$
\mathrm{idf}_j=\ln\frac{1+N}{1+\mathrm{df}_j}+1 .
$$

**Weighting and normalisation.** The unnormalised weight is $\mathrm{tf}_{dj}\cdot\mathrm{idf}_j$. Then divide each row by its Euclidean norm so every document vector has length 1:

$$
\tilde w_{dj}=\frac{\mathrm{tf}_{dj}\,\mathrm{idf}_j}{\sqrt{\sum_{k}(\mathrm{tf}_{dk}\,\mathrm{idf}_k)^2}} .
$$

**Cosine similarity.** For two vectors $a,b$ the cosine of the angle between them is $\frac{a\cdot b}{\|a\|\|b\|}$. After L2 normalisation the denominators are $1$, so the similarity of documents $d$ and $e$ is just the dot product $\tilde w_d\cdot\tilde w_e$. The whole similarity matrix is $\tilde W\tilde W^\top$ with shape $(N,N)$.

## Worked example

Three documents: `"The cat sat"`, `"The dog sat"`, `"The cat ran"`. Tokenised and lowercased, the sorted vocabulary is `[cat, dog, ran, sat, the]`, so $|V|=5$, $N=3$ and the count matrix (shape $3\times5$) is

| doc | cat | dog | ran | sat | the |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 | 1 |
| 3 | 1 | 0 | 1 | 0 | 1 |

Document frequencies are $[2,1,1,2,3]$, so $\mathrm{idf}=[\ln(4/3)+1,\ \ln(4/2)+1,\ \ln 2+1,\ \ln(4/3)+1,\ \ln(4/4)+1]=[1.2877,\,1.6931,\,1.6931,\,1.2877,\,1.0000]$. "the" gets the minimum weight $1$, "dog" and "ran" the maximum.

Document 1 unnormalised: $(1.2877,0,0,1.2877,1)$ with norm $\sqrt{2\cdot1.2877^2+1}=2.0776$, hence $(0.6198,0,0,0.6198,0.4813)$. Document 2 normalises to $(0,0.7203,0,0.5478,0.4254)$ and document 3 to $(0.5478,0,0.7203,0,0.4254)$.

Cosine similarities: docs 1 and 2 share "sat" and "the": $0.6198\cdot0.5478+0.4813\cdot0.4254=0.5443$. Docs 1 and 3 share "cat" and "the", also $0.5443$ by symmetry. Docs 2 and 3 share only "the": $0.4254^2=0.1810$. The pair with only a stop word in common is correctly the least similar. The exercise reproduces every one of these numbers.

## Implementation

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

## Shapes

- `docs`: list of $N$ token lists; `vocab`: list of $V$ strings.
- `X`, `T`, `tfidf(X)`: $(N,V)$. `df`, `idf`: $(V,)$.
- Similarity `W @ W.T`: $(N,N)$, with ones on the diagonal.

In production use `sklearn.feature_extraction.text.TfidfVectorizer`, which returns a sparse matrix and applies this smoothed formula and L2 normalisation by default. Fit the vocabulary and IDF on training data only, then transform new data with the stored values.

## Common misconceptions

- **"TF-IDF understands meaning."** It only counts. "Car" and "automobile" are unrelated columns; that is the gap that embeddings ([[seq.02]]) fill.
- **"Always remove stop words and stem."** For sentiment, negations and small function words matter; for transformers, raw text works best.
- **"Fit the vectoriser on all the data."** Computing IDF on the test set leaks information from it into the features.
- **"Bag of words cannot capture negation."** Plain unigrams cannot, but bigrams such as `not good` can, at the cost of a larger vocabulary.
- **"Lemmatisation does not need context."** Without the part of speech, ambiguous forms are lemmatised wrongly.

## Key takeaways

- Text processing is a sequence of lossy choices; pick them per task and apply the same pipeline to training and test text.
- A bag of words is a sparse count matrix of shape $(N,|V|)$.
- $\mathrm{idf}_j=\ln\frac{1+N}{1+\mathrm{df}_j}+1$ shrinks ubiquitous words; L2 normalisation makes cosine similarity a plain dot product.
- N-grams add local order at the price of vocabulary growth.

## Explain it yourself

1. Why does a word that occurs in every document get a small weight, and why do we still keep a "+1" so it is not exactly zero?
2. A friend says "stemming and lemmatisation are the same thing". Give two inputs where they disagree and say which tool you would use for a search index.

## Further reading

- Jurafsky and Martin, *Speech and Language Processing*, the chapters on regular expressions, tokenisation and vector semantics.
- The scikit-learn user guide section on text feature extraction.

## Mastery criteria

- You can compute a TF-IDF matrix and a cosine similarity by hand for three short documents.
- You can state when stemming harms and when lemmatisation needs a tagger.
- You can explain why the vectoriser is fitted on the training split only.
