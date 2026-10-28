---
id: seq.02
title: Word embeddings
summary: From one-hot vectors to dense vectors learned from context, with the skip-gram negative-sampling loss and its gradients derived, plus the GloVe and fastText ideas, cosine similarity and analogies.
minutes: 90
teaches: [seq.embedding]
requires: [nn.backprop, la.dot, ml.logreg, info.entropy]
sources:
  - "math/cours-tal-01.pdf (CNAM RCP217, PDF p.32-47: one-hot representation, <UNKNOWN>/<PADDING> markers, distributional semantics, word embeddings, word2vec CBOW and skip-gram, cosine, geometric properties, GloVe and its analogy argument). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md (distributional hypothesis, skip-gram pairs, two tables W and W', negative-sampling update, nearest neighbours and analogy). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md (GloVe objective with weighting, fastText character n-grams, BPE overview). READ."
---

## Learning objectives

- Explain why one-hot vectors cannot express similarity and what a dense embedding adds.
- State the distributional hypothesis and how skip-gram turns it into a prediction task.
- Derive the negative-sampling loss and its gradients for both embedding tables.
- Describe the GloVe objective and the fastText subword idea, and when each helps.
- Use cosine similarity, nearest neighbours and vector analogies, and know their limits.
- Train a tiny skip-gram in NumPy and read the result.

## Prerequisites

Logistic regression and the sigmoid ([[f2n.08]]), backpropagation with shapes ([[f2n.10]]).

## Where this fits

Every language model, translator, search engine and chatbot must feed words into a numeric machine. The first decision is *how to represent a word as numbers*. Everything that follows in this course (recurrent networks in [[seq.03]], gated cells in [[seq.04]], sequence-to-sequence and attention in [[seq.05]], and later transformers) takes a sequence of **vectors** as input. Embeddings are those vectors.

They are also the first example of **self-supervised learning**: there is no labeller, the text itself supplies the targets. (The CNAM course notes that people often call word2vec "unsupervised" loosely; the training task is a supervised prediction problem whose labels come for free.)

## The idea in plain words

Suppose the vocabulary has 50,000 words. The most direct code gives every word its own light switch among 50,000: only one switch is on. This **one-hot** code is blind: "cat" and "dog" are exactly as different as "cat" and "mortgage". Any two distinct one-hot vectors are at distance $\sqrt2$, and their dot product is 0. A model trained on sentences about cats learns nothing it can reuse for dogs.

What would we want instead? A *map of words* in which similar words are neighbours. The linguist Firth put the recipe in one line: "you shall know a word by the company it keeps". Look at the words that tend to appear around "cat": *purrs, fur, feeds, chases*. "Dog" appears among almost the same company. So we place words on the map such that **words with similar company get nearby positions**.

How do we find such positions? Turn it into a game. Take a word, hide its neighbours, and ask a small model to guess which words appear near it. To win, the model must give "cat" and "dog" positions from which similar guesses can be made. The positions that win the game are the embeddings. Throw away the model, keep the map.

Two practical tricks keep the game cheap. Guessing "which of the 50,000 words is the neighbour?" is expensive, so instead we ask yes/no questions: "is *purrs* a real neighbour of *cat*? is *mortgage*?" (the second one is a random wrong word, a **negative sample**). That is **negative sampling**.

## Formal setup

A corpus is a sequence of tokens. Fix a window size $m$. For each position take the **centre** word $c$ and each word $o$ within $m$ positions as a **context** word; every such $(c,o)$ is a positive training pair. For "the cat sat on the mat" with $m=1$ the pairs for the centre "sat" are (sat, cat) and (sat, on).

Skip-gram (as presented in the CNAM slides and the source lesson) predicts the context words from the centre word; CBOW predicts the centre from the sum of the context words. The slides note that CBOW works well for frequent words and skip-gram better for rare ones.

Each word $w$ gets **two** vectors in $\mathbb{R}^d$: $v_w$ when it is a centre word (rows of $V\in\mathbb{R}^{n\times d}$) and $u_w$ when it is a context word (rows of $U\in\mathbb{R}^{n\times d}$). In network terms, $V$ is the first layer (one-hot times $V$ just selects a row) and $U$ the output layer. After training the final embedding is $v_w$, or $v_w+u_w$.

### The full softmax model and why it is expensive

Model $p(o\mid c)=\dfrac{e^{u_o\cdot v_c}}{\sum_{w=1}^{n}e^{u_w\cdot v_c}}$. The loss for one pair is $-\log p(o\mid c)$, and its gradient with respect to $v_c$ is $-u_o+\sum_w p(w\mid c)\,u_w$ (the softmax gradient of [[f2n.08]]). The sum runs over the whole vocabulary: $O(nd)$ work per training pair, with $n$ of 50,000 or more.

## Derivation: negative sampling

Replace the $n$-way classification with binary classification. For a true pair $(c,o)$ we want a high probability of "real", for $k$ random words $w_1,\dots,w_k$ we want a high probability of "fake":

$$
P(\text{real}\mid c,w)=\sigma(u_w\cdot v_c),\qquad \sigma(s)=\frac1{1+e^{-s}}.
$$

The negative log-likelihood of the observed labels (this is exactly the logistic-regression loss) is

$$
\ell=-\log\sigma(u_o\cdot v_c)-\sum_{j=1}^{k}\log\sigma(-u_{w_j}\cdot v_c),
$$

using $1-\sigma(s)=\sigma(-s)$. The negatives $w_j$ are drawn from a noise distribution; the source lesson draws them at random from the vocabulary, and the common choice is the unigram frequency raised to the power $3/4$, which samples rare words more often than their raw frequency.

Write the scores $s_o=u_o\cdot v_c$ and $s_j=u_{w_j}\cdot v_c$. We need the derivative of the two kinds of terms. For the positive term: since $\frac{d}{ds}\log\sigma(s)=1-\sigma(s)$,

$$
\frac{\partial}{\partial s_o}\big[-\log\sigma(s_o)\big]=-(1-\sigma(s_o))=\sigma(s_o)-1 .
$$

For a negative term: $\frac{d}{ds}\log\sigma(-s)=-(1-\sigma(-s))=-\sigma(s)$, hence

$$
\frac{\partial}{\partial s_j}\big[-\log\sigma(-s_j)\big]=\sigma(s_j).
$$

Define $g_o=\sigma(s_o)-1$ (in $(-1,0)$) and $g_j=\sigma(s_j)$ (in $(0,1)$). The scores are dot products, so by the chain rule:

$$
\frac{\partial\ell}{\partial v_c}=g_o\,u_o+\sum_{j=1}^{k}g_j\,u_{w_j},\qquad
\frac{\partial\ell}{\partial u_o}=g_o\,v_c,\qquad
\frac{\partial\ell}{\partial u_{w_j}}=g_j\,v_c .
$$

**Reading the gradient.** Gradient descent subtracts these. For $u_o$ the step is $-\eta g_ov_c$ with $g_o<0$: $u_o$ moves **toward** $v_c$. Each negative moves **away** from $v_c$. And $v_c$ itself moves toward the true context vector and away from the noise vectors. The size of each push is the model's current error: a negative that already has a very low score has $g_j\approx0$ and is left alone.

**Cost.** Only $1+k$ rows of $U$ and one row of $V$ change per pair: $O((k+1)d)$ instead of $O(nd)$. If a negative index repeats, its gradient contributions add (use `np.add.at`).

**Shapes.** $V,U:(n,d)$; $v_c,u_o:(d)$; negatives' matrix $U[\text{negs}]:(k,d)$; scores $(k)$; $\partial\ell/\partial v_c=g_ou_o+g^\top U[\text{negs}]:(d)$; the negatives' gradient is the outer product $g\,v_c^\top:(k,d)$.

## Worked example

Take $d=2$, centre vector $v_c=(0.5,\,1.0)$, true context $u_o=(1.0,\,0.5)$ and two negatives $u_1=(-0.5,\,0.5)$, $u_2=(1.0,\,-1.0)$ (the numbers below come from the exercise code).

- Scores: $s_o=0.5+0.5=1.0$, $s_1=-0.25+0.5=0.25$, $s_2=0.5-1.0=-0.5$.
- Loss: $-\log\sigma(1)=0.3133$; $-\log\sigma(-0.25)=0.8259$; $-\log\sigma(0.5)=0.4741$. Total $\ell=1.6133$.
- Errors: $g_o=\sigma(1)-1=-0.2689$, $g_1=\sigma(0.25)=0.5622$, $g_2=\sigma(-0.5)=0.3775$.
- $\partial\ell/\partial v_c=-0.2689(1,0.5)+0.5622(-0.5,0.5)+0.3775(1,-1)=(-0.1725,\,-0.2309)$.
- $\partial\ell/\partial u_o=-0.2689\,(0.5,1.0)=(-0.1345,-0.2689)$; $\partial\ell/\partial u_1=0.5622\,(0.5,1.0)=(0.2811,0.5622)$; $\partial\ell/\partial u_2=(0.1888,0.3775)$.

The first negative scored $0.25$, positive, so it is the one most pushed away. A sanity value: with all vectors zero, every score is 0 and $\ell=(1+k)\ln2$.

## Implementation

```python
import numpy as np

def sigmoid(s):
    return 0.5 * (1.0 + np.tanh(0.5 * s))

def sgns_loss_grads(V, U, c, o, negs):
    negs = np.asarray(negs)
    v = V[c]
    s_pos, s_neg = U[o] @ v, U[negs] @ v
    loss = float(np.logaddexp(0, -s_pos) + np.logaddexp(0, s_neg).sum())
    g_pos, g_neg = sigmoid(s_pos) - 1.0, sigmoid(s_neg)
    dV = np.zeros_like(V); dU = np.zeros_like(U)
    dV[c] = g_pos * U[o] + g_neg @ U[negs]
    dU[o] += g_pos * v
    np.add.at(dU, negs, np.outer(g_neg, v))
    return loss, dV, dU
```

`np.logaddexp(0, -s)` is a numerically stable $-\log\sigma(s)$. The training loop: build pairs from the corpus, shuffle each epoch, draw $k$ negatives, update $V[c]$, $U[o]$ and $U[\text{negs}]$ by SGD. On a 13-sentence toy corpus with $d=8$, window 2, $k=5$, learning rate $0.05$ and 40 epochs, the mean loss falls from $4.13$ to about $1.96$, and in the learned space "king" and "queen" (which never differ in their contexts) have cosine $\approx0.99$ while "cat" and "king" are negatively aligned. The tiny corpus makes this a toy, not evidence of anything about real language.

## Similarity and analogies

Euclidean distance depends too much on word frequency; the CNAM slides prefer the angle between vectors, measured by

$$
\cos(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}\in[-1,1].
$$

It is 1 for the same direction, 0 for orthogonal (as one-hot vectors always are), and invariant to scaling.

Empirically, on large corpora, offsets between vectors capture relations: "woman + (king - man) is near queen". The slides and the source lesson present this as an observed property, not a guarantee. To answer an analogy $a:b::c:?$, compute $b-a+c$ and return the nearest word by cosine **excluding the three query words**; without the exclusion the answer is usually one of the inputs.

## GloVe: use the global counts directly

Word2vec learns online from windows. **GloVe** builds the co-occurrence matrix $X$ ($X_{ij}$ = how often $j$ appears in the context of $i$) and fits vectors to it. The CNAM argument starts from ratios of probabilities. For $i=$ ice, $j=$ steam, and a probe word $k$: $P(k\mid\text{ice})/P(k\mid\text{steam})$ is large for $k=$ solid (8.9 in the slide's table), small for $k=$ gas (0.085), and close to 1 for words related to both (water, 1.36) or to neither (fashion, 0.96). Ratios isolate the meaningful axis (solid to gas). To make ratios become differences, model the **log**:

$$
\log P(k\mid i)\approx w_i^\top\tilde w_k\quad\Rightarrow\quad \log\frac{P(k\mid i)}{P(k\mid j)}\approx(w_i-w_j)^\top\tilde w_k .
$$

If $A:B$ and $C:D$ express the same relation, this must hold for all probe words, so $w_A-w_B\approx w_C-w_D$. That is the geometric reason offsets encode analogies. The implemented objective (source lesson) is a weighted least-squares fit:

$$
J=\sum_{i,j}f(X_{ij})\big(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij}\big)^2,\qquad f(x)=\min\!\big((x/x_{max})^{\alpha},1\big).
$$

The weight $f$ keeps very frequent pairs such as (the, and) from dominating; the source uses $\alpha=0.75$. The final vector is $w+\tilde w$. The CNAM slide says its derivation is simplified (the paper's model is more complex).

## fastText: embed the pieces

Word-level tables have no vector for an unseen word ("zoomer-approved") and treat "run", "runs", "running" as unrelated. **fastText** represents a word as the sum of the embeddings of its character n-grams plus the word itself. With boundary markers, "where" with $n=3$ gives `<wh`, `whe`, `her`, `ere`, `re>`. An unseen word like "whereupon" shares several n-grams with "where" and so gets a nearby vector. The price is a larger table (many n-grams) and slightly blurrier vectors for short irregular words. Modern LLMs go one step further with learned subword vocabularies (BPE); that is the topic of the tokenizer lessons.

## Common misconceptions

- **"The embedding is a hidden layer that needs a nonlinearity."** There is none: multiplying a one-hot vector by $V$ just selects a row. The gain comes from the training task, not from depth.
- **"Similar vectors mean synonyms."** Cosine measures similarity of *contexts*. "Hot" and "cold" appear in the same contexts and end up close; antonyms are not separated.
- **"Embeddings are universal truths."** They reflect the corpus, including its biases and its sense mixing: one vector per word type blurs "bank" the river and "bank" the institution. (Contextual embeddings, later in the course, fix this.)
- **"Analogy arithmetic is exact."** It is a noisy regularity on large corpora; it often fails and needs the exclusion rule.
- **"Negative samples are guaranteed wrong."** A random word may occasionally be a real neighbour. The noise is tolerated because it is rare.

## Limits and trade-offs

- One vector per word type: no polysemy, no handling of unseen words (fastText or subwords mitigate the latter).
- The window defines "company"; small windows give syntactic similarity, large windows give topical similarity.
- Static embeddings carry no sentence-level information: word order is lost, which is why recurrent layers come next.
- Quality needs lots of text: on a toy corpus the geometry is only suggestive.

## Key takeaways

- One-hot vectors are orthogonal (distance $\sqrt2$): no similarity. Dense embeddings of dimension 50-300 place similar-context words nearby.
- Skip-gram with negative sampling minimises $-\log\sigma(u_o\cdot v_c)-\sum_j\log\sigma(-u_j\cdot v_c)$.
- Gradients: $g_o=\sigma(s_o)-1$, $g_j=\sigma(s_j)$; $\partial_{v_c}=g_ou_o+\sum g_ju_j$, $\partial_{u_o}=g_ov_c$, $\partial_{u_j}=g_jv_c$.
- GloVe fits $w_i^\top\tilde w_k+b_i+\tilde b_k$ to $\log X_{ik}$ with a weighting; fastText sums character n-gram vectors.
- Compare with cosine; for analogies exclude the query words.

## Explain it yourself

1. Why do negatives with a very low score contribute almost no gradient, and why is that desirable?
2. Starting from $\log P(k\mid i)\approx w_i^\top\tilde w_k$, explain why $w_{\text{king}}-w_{\text{man}}\approx w_{\text{queen}}-w_{\text{woman}}$ is plausible.

## Further reading

- Source library: `ai-engineering-from-scratch` phase 05, lessons 03 (word2vec) and 04 (GloVe, fastText, BPE).
- CNAM RCP217 slides `cours-tal-01.pdf`, PDF p.32-47.
- Next: [[seq.03]], turning a sequence of embeddings into a prediction with recurrence.

## Mastery criteria

You can derive the three gradients of the negative-sampling loss from scratch, your NumPy implementation passes a finite-difference check (including repeated negatives), and you can explain in a few sentences what the learned geometry does and does not mean.
