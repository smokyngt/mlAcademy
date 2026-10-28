---
id: nlp.05
title: Topic modelling with LDA
summary: Understand the LDA generative story and Dirichlet priors, derive the collapsed Gibbs sampling update, implement a tiny LDA in NumPy, and learn how topics are evaluated and how embedding-based topics differ.
minutes: 85
teaches: [nlp.topic-models]
requires: [prob.basics, nlp.tfidf]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/15-topic-modeling/docs/en.md: LDA generative story, BERTopic pipeline, coherence and diversity, when to pick which. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 11.3 Gibbs Sampling (PDF p.564-565): why sampling one variable from its conditional leaves the joint invariant, the ergodicity condition, and the acceptance probability of 1. READ."
---

## Where this fits

You have 50,000 support tickets or news articles and no labels. What are they about? **Topic modelling** answers without supervision: it returns a handful of word lists ("topics") and, for each document, how much of each topic it contains. It is exploratory data analysis for text, used to organise archives, to summarise trends and to create features. It builds on the bag-of-words counts of [[nlp.01]] (LDA uses raw counts, not TF-IDF) and on Bayesian ideas: a generative story, a prior, and sampling. The Gibbs sampling here is the same tool Bishop presents for general probabilistic models, and the Dirichlet smoothing is the same as the add-$\alpha$ of [[nlp.02]].

## The idea in plain words

Imagine a newspaper written by a committee. The committee has a few themes in a drawer: sport, cooking, politics. Each theme is a bag of words with its own frequencies (sport contains "goal" often, "oven" never). To write an article, the author first decides a *mix* (say 70% sport, 30% politics). Then for each word position they flip a coin by that mix to choose a theme, and pull a random word out of that theme's bag. We see only the finished articles. **Latent Dirichlet Allocation** (LDA) runs this story backwards: given all the articles, find bags of words and mixes that would make them likely.

Two practical consequences. Documents are mixtures, not single labels: an article on a cooking competition can be 50% sport. And the number of themes is a choice you make.

## The model

Let $K$ be the number of topics, $V$ the vocabulary size, $D$ the number of documents. For each topic $k$ there is a word distribution $\phi_k$ over the vocabulary (a vector of length $V$ summing to 1), and for each document $d$ a topic mixture $\theta_d$ (length $K$, summing to 1). The generative story:

1. For each topic $k$: draw $\phi_k\sim\mathrm{Dirichlet}(\beta,\dots,\beta)$.
2. For each document $d$: draw $\theta_d\sim\mathrm{Dirichlet}(\alpha,\dots,\alpha)$.
3. For each word position $i$ of document $d$: draw a topic $z_{di}\sim\mathrm{Categorical}(\theta_d)$, then a word $w_{di}\sim\mathrm{Categorical}(\phi_{z_{di}})$.

**Dirichlet intuition.** A Dirichlet distribution is a distribution over probability vectors. With a symmetric concentration parameter $\alpha$: if $\alpha=1$ all mixtures are equally likely; if $\alpha<1$ it prefers sparse vectors, a few large entries and the rest near zero (documents with one or two dominant topics, topics with a small set of characteristic words); if $\alpha>1$ it prefers near-uniform vectors. Typical choices: $\alpha\approx0.1$ to $50/K$, $\beta\approx0.01$ to $0.1$. It is the conjugate prior of the categorical distribution: a Dirichlet prior plus categorical observations gives a Dirichlet posterior with the counts added to the parameters, which is why the formulas below are "counts plus $\alpha$".

We observe the words $w$. Unknown: $z$, $\theta$, $\phi$. Exact posterior inference is intractable, so we use sampling (here) or variational methods (scikit-learn's default).

## Derivation: collapsed Gibbs sampling

**Gibbs sampling** (Bishop Sec. 11.3) draws samples from a joint distribution by repeatedly resampling one variable from its conditional distribution given all the others. Bishop shows two things: the joint distribution is invariant under each such step (the conditioning variables are untouched and the resampled one comes from its correct conditional), and the chain is ergodic if no conditional is zero anywhere. Gibbs sampling is also Metropolis-Hastings with acceptance probability always 1, so no proposals are rejected.

For LDA we **collapse** (integrate out) $\theta$ and $\phi$ and sample only the topic assignments $z$. Notation: $n_{dk}$ = number of tokens in document $d$ assigned to topic $k$; $n_{kw}$ = number of times word $w$ is assigned to topic $k$; $n_k=\sum_w n_{kw}$.

For one document, integrating a Dirichlet against the multinomial counts gives

$$
\int \mathrm{Dir}(\theta\mid\alpha)\prod_k\theta_k^{n_{dk}}\,d\theta=\frac{\Gamma(K\alpha)}{\Gamma(K\alpha+N_d)}\prod_{k}\frac{\Gamma(\alpha+n_{dk})}{\Gamma(\alpha)},
$$

and similarly for each topic's word counts with $\beta$ and $V$. The joint $P(w,z)$ is the product of these terms. To resample the topic of one token we need $P(z_{di}=k\mid z_{-di},w)\propto P(w,z)$ viewed as a function of $k$ with everything else fixed. Remove token $i$ from the counts (call the reduced counts $n^{-di}$). Putting it back in topic $k$ changes only $n_{dk}$ and $n_{kw}$ (with $w=w_{di}$) by $+1$. Each Gamma ratio is of the form $\Gamma(a+1)/\Gamma(a)=a$, and the normalisers depend only on totals that are the same for every $k$ except $n_k$, so

$$
P(z_{di}=k\mid z_{-di},w)\ \propto\ \big(n^{-di}_{dk}+\alpha\big)\cdot\frac{n^{-di}_{kw}+\beta}{n^{-di}_{k}+V\beta}.
$$

Read it as: *how much does this document like topic $k$* times *how much does topic $k$ like this word*. Normalise over the $K$ values and sample.

**Algorithm.** Initialise every $z_{di}$ at random and fill the counts. Repeat for many sweeps: for each token, decrement its counts, compute the $K$ weights above, sample a new topic, increment the counts. After burn-in, estimate

$$
\hat\theta_{dk}=\frac{n_{dk}+\alpha}{N_d+K\alpha},\qquad \hat\phi_{kw}=\frac{n_{kw}+\beta}{n_k+V\beta}.
$$

Cost: $O(K)$ per token per sweep.

## Evaluation

There is no ground truth, so evaluation combines several signals.

- **Held-out perplexity or log-likelihood** of unseen documents: principled but weakly correlated with human judgement.
- **Coherence** (for instance $c_v$, built on normalised pointwise mutual information of the top words of a topic over a text window): do the top words co-occur in real text? Higher is better. Libraries: `gensim.models.CoherenceModel`.
- **Topic diversity**: fraction of distinct words among all topics' top-$n$ words. Low values mean repeated, redundant topics.
- **Human reading**: look at the top ten words and sample documents of each topic. This remains the last line of defence.

Practical choices: remove stop words and very rare or very frequent words before counting; try several $K$ and compare coherence and your reading, rather than trusting one number; run with different seeds (the sampler finds local optima) and check stability.

## Embedding-based topics: the BERTopic idea

LDA sees only word counts, which fails on short text (tweets, headlines) where each document has too few words. The BERTopic pipeline described in the source works differently: embed each document with a sentence encoder (dense vectors, see [[seq.02]]), reduce dimensions (UMAP), cluster (HDBSCAN, which also labels outliers), then name each cluster by its most characteristic words using a class-based TF-IDF (the TF-IDF of [[nlp.01]] with each cluster treated as one big document). Output: one topic per document plus an outlier group, with semantics captured by the embeddings. Trade-off: LDA gives mixtures and handles long documents and small compute; BERTopic is stronger on short text and semantic coherence but gives hard assignments and truncates long inputs. Both can be inspected with the same coherence and diversity checks.

## Worked example

First one update step. A token with word $w$ has, after removing it, document counts $n_{d\cdot}=(3,1)$, word counts $n_{\cdot w}=(4,0)$, topic totals $n_\cdot=(10,8)$, with $V=8$, $\alpha=0.1$, $\beta=0.01$. Unnormalised weights: topic 0: $(3+0.1)\cdot\frac{4+0.01}{10+0.08}=1.2333$; topic 1: $(1+0.1)\cdot\frac{0+0.01}{8+0.08}=0.001361$. Normalised: $(0.99890,\ 0.00110)$. The document is mostly topic 0 and topic 0 already owns this word, so the token almost surely stays in topic 0.

Now the whole sampler. Vocabulary of 8 word ids: ids 0-3 are "sport" words, ids 4-7 "cooking" words. Nine documents of 20 tokens: four drawn only from ids 0-3, four only from ids 4-7, and one drawn from all eight. With $K=2$, $\alpha=0.1$, $\beta=0.01$ and 60 sweeps, every one of the ten seeds we tried recovers two topics whose top-4 words are exactly $\{0,1,2,3\}$ and $\{4,5,6,7\}$ (which topic index gets which group depends on the seed), the sport and cooking documents get $\hat\theta$ essentially $(1,0)$ or $(0,1)$, and the mixed document gets a split mixture (about $(0.5,0.5)$ for the last seed). The exercise checks these properties.

## Implementation

```python
import numpy as np

def conditional(ndk_d, nkw_w, nk, alpha, beta, V):
    p = (ndk_d + alpha) * (nkw_w + beta) / (nk + V * beta)   # shape (K,)
    return p / p.sum()

def lda_gibbs(docs, K, V, alpha, beta, iters, seed):
    rng = np.random.default_rng(seed)
    ndk = np.zeros((len(docs), K)); nkw = np.zeros((K, V)); nk = np.zeros(K)
    z = [rng.integers(K, size=len(d)) for d in docs]
    for d, doc in enumerate(docs):                       # fill the counts
        for i, w in enumerate(doc):
            k = z[d][i]; ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1
    for _ in range(iters):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                k = z[d][i]; ndk[d, k] -= 1; nkw[k, w] -= 1; nk[k] -= 1
                k = rng.choice(K, p=conditional(ndk[d], nkw[:, w], nk, alpha, beta, V))
                z[d][i] = k; ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1
    return ndk, nkw
```

In production use `gensim.models.LdaModel` (variational, streaming) or `sklearn.decomposition.LatentDirichletAllocation`; for BERTopic use the `bertopic` package.

## Shapes

- `docs`: list of $D$ lists of word ids. `z[d]`: $(N_d,)$.
- `ndk`: $(D,K)$, rows sum to $N_d$. `nkw`: $(K,V)$, total sum is the number of tokens. `nk`: $(K,)$.
- $\hat\theta$: $(D,K)$, $\hat\phi$: $(K,V)$, each row summing to 1.

## Common misconceptions

- **"Topics are labelled categories."** A topic is a distribution over words; the name is your interpretation.
- **"LDA finds the right number of topics."** $K$ is a hyperparameter; compare several values.
- **"Use TF-IDF input for LDA."** The generative model is over counts; use raw counts.
- **"A run of the sampler gives the answer."** It is a random search for a local optimum; check stability over seeds.
- **"The counts must be decremented only at the end of the sweep."** The token being resampled must be removed from the counts before computing its conditional, otherwise it votes for its own current topic.

## Key takeaways

- LDA = documents are mixtures of topics, topics are distributions over words, with Dirichlet priors.
- Collapsed Gibbs: $P(z=k)\propto(n_{dk}+\alpha)\frac{n_{kw}+\beta}{n_k+V\beta}$, with the current token removed.
- Evaluate with coherence, diversity, held-out likelihood and reading the topics.
- BERTopic clusters embeddings instead; it helps on short text but loses mixtures.

## Explain it yourself

1. Describe in words what each of the two factors in the Gibbs update measures, and what happens when $\alpha$ is very small.
2. When would you prefer BERTopic over LDA, and what do you give up?

## Further reading

- Blei, Ng and Jordan, "Latent Dirichlet Allocation" (2003); Griffiths and Steyvers, "Finding scientific topics" (2004) for the collapsed sampler.
- Bishop, *Pattern Recognition and Machine Learning*, Sec. 11.3.

## Mastery criteria

- You can write the LDA generative story and name $\theta$, $\phi$, $z$.
- You can derive (or at least justify each factor in) the collapsed Gibbs conditional.
- You can run a tiny LDA, read its output and say how you would judge it.
