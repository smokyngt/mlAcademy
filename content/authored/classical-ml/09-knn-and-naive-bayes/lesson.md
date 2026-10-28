---
id: cml.09
title: k-nearest neighbours and naive Bayes
summary: Two opposite classifiers, one that memorises the data and votes among neighbours, one that models how each class generates features under a bold independence assumption; implement both and learn when each breaks.
minutes: 70
teaches: [ml.knn, ml.naive-bayes]
requires: [prob.basics, prob.rv, ml.generalization, la.dot]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 2.5.2 Nearest-neighbour methods (PDF p.144-147): K-NN density estimate, derivation of the K-NN classifier via Bayes' theorem, smoothing role of K, Cover-Hart bound, storage and search cost. READ."
  - "Bishop, PRML, Sec. 4.2 Probabilistic Generative Models, incl. 4.2.1 and 4.2.3 (PDF p.216-223): Bayes' theorem for classification, sigmoid/softmax of log-odds, Gaussian class-conditionals, naive Bayes assumption for discrete features. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/06-knn-and-distances/docs/en.md (choice of K, curse of dimensionality). READ (those sections)."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/14-naive-bayes/docs/en.md (independence assumption, variants, Laplace smoothing, log-space computation). READ (concept sections)."
---

## Where this fits

Two very different answers to "how do I classify a new example from labelled ones?"

- **k-nearest neighbours (kNN)**: do no training at all. Keep the data, and for a new point look at its closest labelled neighbours and let them vote. It is the baseline for recommendations ("users like you"), anomaly checks and retrieval of similar items (the same idea behind vector search over embeddings).
- **Naive Bayes**: build a tiny probabilistic story of how each class produces its features, then invert it with Bayes' theorem. It powered the first practical spam filters because it trains in one pass, needs little data, handles thousands of word features and is still a respectable text baseline.

Together they show the two basic families: **non-parametric / instance-based** versus **generative / parametric**, and the trade-off between flexibility (kNN) and strong assumptions (naive Bayes) that you met in [[cml.03]].

## The idea in plain words

**kNN.** You move to a new city and wonder if a restaurant is good. You ask the five people who live closest to it. Those neighbours are your prediction. Two choices matter: *what counts as close* (distance, and therefore units!) and *how many neighbours* to ask. Ask one person and a single eccentric opinion decides; ask the whole city and you always hear the majority view. So the number $k$ is a smoothness dial. And "close" breaks down when the points live in many dimensions: with hundreds of features, almost everybody is about equally far away.

**Naive Bayes.** A mail clerk learned that spam often contains "free" and "win" and rarely "meeting". For a new email they tally how typical each word is for spam versus ordinary mail and combine the evidence, as if each word were an independent witness. That independence is obviously false (words come in phrases), hence "naive", yet it makes the bookkeeping trivial: one small table of word frequencies per class instead of an impossible table over all word combinations. It is often wrong about the *probabilities* but right about *which class wins*.

## kNN: definition and derivation

Training set $\{(x_n,y_n)\}$, a distance $d$. For a query $x$: find the $k$ training points with smallest $d(x,x_n)$ and predict the majority class (or, for regression, the mean of their targets). Common distance: Euclidean $d(x,z)=\|x-z\|_2$.

**Where "majority vote" comes from** (Bishop Sec. 2.5.2). Grow a sphere around $x$ until it contains exactly $K$ training points; call its volume $V$, and let $K_c$ of the points belong to class $c$ (class $c$ has $N_c$ points of the $N$ total). A local density estimate is "count over (number of points times volume)":

$$
p(x\mid c)\approx\frac{K_c}{N_cV},\qquad p(x)\approx\frac{K}{NV},\qquad p(c)=\frac{N_c}{N}.
$$

Bayes' theorem:

$$
p(c\mid x)=\frac{p(x\mid c)p(c)}{p(x)}=\frac{\frac{K_c}{N_cV}\cdot\frac{N_c}{N}}{\frac{K}{NV}}=\frac{K_c}{K}.
$$

Everything cancels: the posterior is the fraction of the $K$ neighbours that belong to class $c$, and picking the largest posterior is the majority vote. Bishop adds a classical result: as $N\to\infty$ the 1-nearest-neighbour rule has error at most twice the best achievable (Bayes) error (Cover and Hart).

**Choosing $k$.** $k=1$: the boundary follows every point, zero training error, high variance. Large $k$: smoother, more bias; $k=N$ predicts the global majority. Pick $k$ (and the distance) by validation error ([[cml.03]]); an odd $k$ avoids ties in binary problems.

**Feature scaling is mandatory.** Distances add squared differences of raw columns, so a feature measured in dollars swamps one measured in years. Standardise each column ($z=(x-\mu)/\sigma$) using *training-set* statistics, and apply the same transform to queries.

**The curse of dimensionality.** In high dimensions the nearest and farthest points become nearly equally distant, so "nearest" carries little information. A quick measure is the relative contrast $(d_{\max}-d_{\min})/d_{\min}$ from a query to a random cloud. Moreover, to capture a fixed share of the data your neighbourhood must cover most of the range of each feature, so it is no longer "local". Remedies: fewer, better features (PCA, [[cml.05]]), feature selection, or learned embeddings. Cost: kNN stores all data and, naively, computes $N$ distances per query ($O(Nd)$), so prediction is slow for big $N$; tree-based indices help only in low-to-moderate dimension.

## Naive Bayes: definition and derivation

Generative classification (Bishop Sec. 4.2): model $p(x\mid C_k)$ and $p(C_k)$, then

$$
p(C_k\mid x)=\frac{p(x\mid C_k)\,p(C_k)}{\sum_jp(x\mid C_j)p(C_j)}\ \propto\ p(C_k)\,p(x\mid C_k).
$$

For $d$ features the joint $p(x\mid C_k)$ is a huge table (for $d$ binary features, $2^d-1$ numbers per class; Bishop p.202). The **naive Bayes assumption** is conditional independence of the features given the class:

$$
p(x\mid C_k)=\prod_{i=1}^dp(x_i\mid C_k),
$$

which needs only $d$ one-dimensional distributions per class. To avoid multiplying hundreds of tiny numbers (underflow) work with logs:

$$
\log p(C_k\mid x)=\text{const}+\log p(C_k)+\sum_{i=1}^d\log p(x_i\mid C_k),\qquad \hat y=\arg\max_k\ [\ldots].
$$

Training is pure counting or averaging; prediction is one matrix product.

**Multinomial variant (word counts).** Let $n_{ki}$ be the total count of word $i$ over all training documents of class $k$, $V$ the vocabulary size. Maximum-likelihood estimate $\theta_{ki}=n_{ki}/\sum_in_{ki}$, and a document with counts $x_i$ has log-score $\log p(C_k)+\sum_ix_i\log\theta_{ki}$. This is linear in the count vector $x$ (compare Bishop Eq. 4.82 for binary features, which is also linear in $x$: naive Bayes yields a linear classifier, like logistic regression, but fitted by counting rather than by optimisation).

**Laplace smoothing.** If a word never appeared in a class, $\theta_{ki}=0$, and one such word zeroes the whole product (or $-\infty$ log-score) no matter how much other evidence there is. Add a pseudo-count $\alpha$ (usually 1) to every word:

$$
\theta_{ki}=\frac{n_{ki}+\alpha}{\sum_in_{ki}+\alpha V}.
$$

It is equivalent to a uniform Dirichlet prior over the word probabilities, so $\alpha$ is a strength-of-prior dial: $\alpha\to0$ trusts the data, large $\alpha$ flattens towards uniform.

**Gaussian variant (continuous features).** Per class and feature estimate the mean $\mu_{ki}$ and variance $\sigma_{ki}^2$ and use

$$
\log p(x_i\mid C_k)=-\tfrac12\log(2\pi\sigma_{ki}^2)-\frac{(x_i-\mu_{ki})^2}{2\sigma_{ki}^2}.
$$

Bishop's Gaussian generative model with a *shared* full covariance gives a linear boundary; naive Gaussian with class-specific variances gives quadratic boundaries but ignores correlations between features. A tiny variance floor avoids division by zero.

**Why does it work when the assumption is false?** Classification needs only the argmax. Correlated features double-count evidence, which makes the posterior over-confident (probabilities near 0 or 1), but the ranking of classes is frequently still right. The strong assumption also means few parameters, hence low variance: with little data it often beats flexible models. It fails when correlated features carry the signal (e.g. XOR-like interactions) and its probability outputs are poorly calibrated.

## Worked example 1: scaling decides the neighbour

Query: age 30, income $50{,}000$. Candidate A: age 31, income $60{,}000$ (class 0). Candidate B: age 60, income $50{,}500$ (class 1).

- Raw Euclidean distances: $d(q,A)=\sqrt{1+10000^2}=10000.00005$, $d(q,B)=\sqrt{30^2+500^2}=500.90$. B is "nearest": the age gap of 30 years is invisible next to dollars.
- Standardise with $\sigma_{\text{age}}=10$, $\sigma_{\text{income}}=10{,}000$: $d(q,A)=\sqrt{0.1^2+1^2}=1.005$, $d(q,B)=\sqrt{3^2+0.05^2}=3.000$. Now A is nearest.

## Worked example 2: a tiny spam filter

Vocabulary (free, win, meeting, report). Three spam emails with count rows $(3,2,0,0),(2,1,0,1),(1,2,1,0)$; three ham emails $(0,0,2,3),(1,0,3,1),(0,1,2,2)$. Priors are $\tfrac12,\tfrac12$.

- Spam word totals: $(6,5,1,1)$, sum $13$. With $\alpha=1$, $V=4$: $\theta_{\text{spam}}=(7,6,2,2)/17=(0.412,0.353,0.118,0.118)$.
- Ham totals: $(1,1,7,6)$, sum $15$: $\theta_{\text{ham}}=(2,2,8,7)/19=(0.105,0.105,0.421,0.368)$.
- New email "free free meeting": counts $x=(2,0,1,0)$.
  - $\log$ score spam $=\ln0.5+2\ln0.412+\ln0.118=-4.608$.
  - $\log$ score ham $=\ln0.5+2\ln0.105+\ln0.421=-6.061$.
  - Posterior spam $=\sigma(-4.608-(-6.061))=\sigma(1.453)=0.810$, where $\sigma(a)=1/(1+e^{-a})$.

The email is classified as spam with posterior $0.81$. If "free" had never appeared in ham, the unsmoothed ham score would be $-\infty$ regardless of "meeting"; smoothing prevents that.

## Implementation

```python
import numpy as np

def knn_predict(Xtr, ytr, Xte, k):
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)    # (M, N)
    nn = np.argsort(d2, axis=1)[:, :k]                         # (M, k)
    return np.array([np.bincount(ytr[r]).argmax() for r in nn])

def mnb_fit(X, y, alpha=1.0):                                   # X: (N, V) counts
    classes = np.unique(y)
    log_prior = np.log(np.array([(y == c).mean() for c in classes]))
    counts = np.array([X[y == c].sum(0) for c in classes]) + alpha
    log_theta = np.log(counts / counts.sum(1, keepdims=True))
    return classes, log_prior, log_theta

def mnb_scores(X, log_prior, log_theta):
    return X @ log_theta.T + log_prior                          # (N, classes)
```

## Shapes

kNN: `Xtr (N, d)`, `Xte (M, d)`, pairwise distances `(M, N)` (memory $MNd$ with the naive broadcast above; use the identity $\|a-b\|^2=\|a\|^2+\|b\|^2-2a\cdot b$ for $(M,N)$ only). Multinomial NB: counts `(N, V)`, `log_theta (C, V)`, `log_prior (C,)`, scores `(N, C)`. Gaussian NB: means and variances `(C, d)`.

## Common misconceptions

- **"kNN needs no hyperparameters."** $k$, the distance and the scaling are all hyperparameters; fit them on training folds only.
- **Scaling on the full data set.** Compute means and standard deviations on the training split only; otherwise test information leaks.
- **"Zero training time means cheap."** kNN moves the cost to prediction time and memory.
- **"Naive Bayes outputs reliable probabilities."** They are usually too extreme because of double-counting; use them to rank, or calibrate.
- **"Independence is required for naive Bayes to work."** It is an assumption of the model, not of the data; it often works despite violations, but fails under strong, label-relevant dependencies.
- **Forgetting to smooth, or smoothing the prior.** Laplace smoothing is applied to the feature probabilities $\theta_{ki}$.
- **Multiplying probabilities directly** instead of adding logs: underflow for long documents.

## Key takeaways

- kNN posterior $=K_c/K$ follows from local density estimates plus Bayes' theorem; $k$ controls smoothness; scale features; beware high dimensions.
- Naive Bayes: $\arg\max_k\ \log p(C_k)+\sum_i\log p(x_i\mid C_k)$, trained by counting (multinomial) or by per-class mean and variance (Gaussian).
- Laplace smoothing $(n+\alpha)/(\text{total}+\alpha V)$ prevents zero probabilities; compute in log space.
- kNN is flexible but slow and fragile in many dimensions; naive Bayes is fast and stable with little data but biased by its independence assumption.

## Explain it yourself

1. Derive $p(c\mid x)=K_c/K$ and say which quantities cancel.
2. Explain why naive Bayes can classify well even though its probability estimates are badly calibrated.

## Further reading

- Bishop, *PRML*, Sec. 2.5.2 (PDF p.144-147) and Sec. 4.2 (PDF p.216-223).
- Repo lessons `phases/02-ml-fundamentals/06-knn-and-distances` and `14-naive-bayes`.
- Related: [[cml.02]] (a discriminative linear classifier for the same job), [[cml.05]] (reducing dimension before kNN).

## Mastery criteria

You can predict with kNN by hand for a few points, show how feature scaling changes the neighbours, measure distance concentration as dimension grows, and train and apply multinomial and Gaussian naive Bayes with smoothing in log space.
