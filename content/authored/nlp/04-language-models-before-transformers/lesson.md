---
id: nlp.04
title: Language models before transformers
summary: Build n-gram language models from counts, fix the zero-count problem with smoothing and interpolation, score models with perplexity, and see why neural language models replaced them.
minutes: 70
teaches: [nlp.ngram-lm]
requires: [prob.basics, info.entropy, alg.exp-log]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer/docs/en.md: n-gram probabilities, zero-count problem, list of smoothing methods, perplexity, Shannon's prediction game. READ."
  - "math/cours-tal-01.pdf (PDF p.28): n-gram counts and the zero-count problem. READ."
  - "math/cours-tal-03.pdf (PDF p.1-6, French slides): translation seen as predicting the next item of a sequence, and the encoder-decoder idea. Only the opening pages were read. READ."
---

## Where this fits

A **language model** (LM) assigns a probability to a sentence, or equivalently predicts the next word given the previous ones. That sounds abstract, but speech recognisers, spell checkers, autocomplete and the older machine translation systems all used one to choose among candidate outputs: "recognise speech" is far more probable than "wreck a nice beach". Modern large language models ([[llm.02]]) are the same idea scaled up with transformers; the evaluation number, **perplexity**, is the same one you will see in their papers. This lesson builds the count-based model that dominated from the 1980s to the 2010s, and the problem that killed it (sparse counts) shows why neural models were invented. It uses the n-grams of [[nlp.01]] and the log-probabilities of [[nlp.02]].

## The idea in plain words

Play a guessing game: I show you "the cat sat on the" and you guess the next word. You say "mat", or "floor", or "sofa", because you have seen this kind of sentence before. An n-gram model plays this game by pure memory: it remembers, from a big pile of text, how often each word followed each short context, and answers with those relative frequencies. A bigram model remembers one previous word, a trigram two.

Two things go wrong. First, memory has limits: with a long enough context nothing is ever seen twice, so the model must forget distant words. Second, and worse, the world is bigger than the pile of text: perfectly good word pairs never occurred in your pile, and the model says they are impossible. **Smoothing** is the art of reserving a little belief for what you have not seen. **Perplexity** scores the whole game: how many equally likely options, on average, was the model choosing between?

## Definitions

A sentence $w_1,\dots,w_m$ is padded with a start symbol $\langle s\rangle$ and an end symbol $\langle/s\rangle$ so the model also learns where sentences begin and end. By the chain rule of probability (exact, no assumption)

$$
P(w_1,\dots,w_m)=\prod_{i=1}^{m+1}P(w_i\mid w_1,\dots,w_{i-1}).
$$

The **Markov assumption** of order $n-1$ keeps only the last $n-1$ words of context. For a bigram ($n=2$):

$$
P(w_i\mid w_{1:i-1})\approx P(w_i\mid w_{i-1}).
$$

**Maximum likelihood estimate.** With $c(\cdot)$ the count in the training corpus,

$$
P_{\mathrm{MLE}}(w\mid v)=\frac{c(v,w)}{c(v)} .
$$

This is the same relative-frequency estimate as in Naive Bayes ([[nlp.02]]), derived the same way (maximise the training likelihood under normalisation).

## Smoothing

Under the MLE any bigram not in the training data has probability $0$, so any test sentence containing one gets probability $0$ and infinite perplexity. Real corpora have a long tail: a large fraction of the bigrams in new text are unseen. We must move probability mass from seen to unseen events.

**Add-$k$ (Laplace for $k=1$).** Pretend each of the $|V|$ words followed each context $k$ extra times:

$$
P_{\mathrm{add}\text{-}k}(w\mid v)=\frac{c(v,w)+k}{c(v)+k|V|}.
$$

The denominator is $\sum_w (c(v,w)+k)$, so it is a valid distribution. With $|V|$ in the tens of thousands, $k=1$ steals far too much mass from observed events, so use small $k$ (tuned on held-out data) or better methods.

**Interpolation.** Mix models of different orders, so a reliable low-order estimate fills in where the high-order one is unreliable:

$$
P_{\mathrm{interp}}(w\mid v)=\lambda\,P_{\mathrm{MLE}}(w\mid v)+(1-\lambda)\,P_{\mathrm{MLE}}(w),\qquad 0\le\lambda\le1 .
$$

This is a mixture of two distributions, hence itself a distribution. Choose $\lambda$ on held-out text (the weights can depend on the context count).

**Backoff.** Use the highest-order estimate if its count is positive; otherwise "back off" to the next lower order. Katz backoff discounts the high-order counts so the total remains 1. A cheap, unnormalised variant used at web scale, *stupid backoff*, simply multiplies by a constant (about $0.4$) at each back-off step; it gives scores, not probabilities. The most successful classical method, **Kneser-Ney**, combines absolute discounting with a smarter lower-order distribution that counts in how many different contexts a word appears (the source's example: "Francisco" is frequent but almost only follows "San", so it should not be a likely word in a fresh context). We do not derive it here.

## Perplexity

For a test set of $N$ tokens (counting the end symbols), the **average negative log-likelihood** (cross-entropy in nats) is

$$
H=-\frac{1}{N}\sum_{i=1}^{N}\ln P(w_i\mid \text{context}_i),
$$

and the **perplexity** is $\mathrm{PP}=e^{H}$ (equal to $2^{H_{\text{bits}}}$ when logs are base 2). Interpretation: a model that spreads its probability uniformly over $b$ words at every step has perplexity exactly $b$, because $H=\ln b$. So perplexity is the effective branching factor; lower is better. It can be compared only between models with the same vocabulary and tokenisation, and the test set must not have been used to tune anything. A zero probability anywhere makes it infinite.

## Neural language models

The n-gram table treats "cat" and "dog" as unrelated symbols, so seeing "the cat sat on the mat" teaches nothing about "the dog sat on the mat". A neural LM replaces the table with a function: map each of the previous $n-1$ words to a dense embedding vector ([[seq.02]]), concatenate them, pass through a hidden layer, and output a softmax over the vocabulary. Similar words get similar embeddings and therefore similar predictions, so the model **generalises** to unseen contexts without any smoothing, and the number of parameters grows with the embedding size rather than as $|V|^n$. Recurrent networks ([[seq.03]]) remove the fixed window; the French slides in the sources present translation as exactly this next-item prediction, with an encoder-decoder arrangement. Transformers ([[llm.02]]) extend the context to thousands of tokens. The training loss of all of them is the cross-entropy $H$ above, so perplexity is $e^{\text{loss}}$.

## Worked example

Training corpus (each padded with $\langle s\rangle$ and $\langle/s\rangle$): `the cat sat on the mat`, `the dog sat on the rug`, `the cat saw the dog`. The vocabulary of predicted tokens has $|V|=9$ (the, cat, sat, on, mat, dog, rug, saw and $\langle/s\rangle$). Counts: $c(\text{the})=6$ as a context, $c(\text{the},\text{cat})=2$.

- $P_{\mathrm{MLE}}(\text{cat}\mid\text{the})=2/6=0.3333$.
- Add-1: $(2+1)/(6+9)=0.2$. Seen events lose mass: $0.333\to0.2$. The pair (the, rug) was seen once, so MLE gives $1/6=0.1667$ and add-1 gives $(1+1)/15=0.1333$.
- Interpolation with $\lambda=0.7$: $P(\text{rug}\mid\text{the})=0.7\cdot\frac16+0.3\cdot\frac{1}{20}=0.1317$ (there are $20$ predicted tokens in total).

Perplexity of the test sentence `the mat saw the cat`. Its bigram (mat, saw) never occurs in training. MLE: infinite. Add-1: $6.289$. Add-0.1: $5.396$. Interpolated ($\lambda=0.7$): $6.396$. Perplexity of `the dog sat on the mat`, whose bigrams were all seen: MLE $1.669$, add-1 $4.569$, add-0.1 $2.168$: add-1 makes a good model look much worse because it moves so much mass to events that never happen. The uniform model over 9 words has perplexity 9; every smoothed model is far below it. All values come from the exercise code.

## Implementation

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

NLTK's `nltk.lm` and KenLM implement the production smoothed n-gram models (KenLM with Kneser-Ney). Always compute in log space.

## Shapes

For a vocabulary of size $V$ a bigram table has $V^2$ cells ($V\times V$), a trigram table $V^3$; this blow-up is why n-gram models stop at $n\approx5$ and why most of the cells are empty. A neural LM with embedding size $d$ and window $n-1$ has about $Vd+(n-1)d\,h+hV$ parameters for hidden size $h$.

## Common misconceptions

- **"Zero probability means impossible."** It only means unobserved; language is open-ended.
- **"Lower perplexity always means a better application."** It is comparable only on the same vocabulary and tokeniser, and does not guarantee better downstream quality.
- **"Add-one smoothing is the standard."** It is a teaching device; it performs badly with large vocabularies.
- **"Tune smoothing on the test set."** Tune $k$ and $\lambda$ on a validation set.
- **"Bigger $n$ is always better."** Higher order means sparser counts; beyond a point it needs more data and better smoothing.

## Key takeaways

- An n-gram LM is the chain rule plus a Markov assumption, estimated by relative frequency.
- Unseen n-grams force smoothing: add-$k$, interpolation, backoff, Kneser-Ney.
- Perplexity $=e^{H}$ is the exponentiated per-token cross-entropy: the effective number of choices.
- Neural LMs replace the count table with embeddings, generalise across similar words, and are trained on the same cross-entropy.

## Explain it yourself

1. Why does a single unseen bigram make the perplexity of an MLE model infinite, and how does interpolation with a unigram model repair it?
2. A friend says a language model with perplexity 10 "knows 10 words". What is the correct reading?

## Further reading

- Jurafsky and Martin, *Speech and Language Processing*, chapter on n-gram language models (smoothing, Kneser-Ney, perplexity).
- Bengio et al., "A neural probabilistic language model" (2003).

## Mastery criteria

- You can compute bigram MLE, add-$k$ and interpolated probabilities by hand.
- You can compute perplexity from per-token probabilities and interpret it.
- You can state the weakness of count tables that neural LMs fix.
