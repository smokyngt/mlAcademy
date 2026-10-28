---
id: cml.14
title: "Anomaly detection: z-scores, Mahalanobis distance and isolation forests"
summary: Model what is normal instead of classifying what is rare, from one-dimensional z-scores through Mahalanobis distance to an isolation forest you build yourself, and learn how to judge detectors when you have almost no labels.
minutes: 70
teaches: [ml.anomaly, ml.iforest]
requires: [prob.rv, la.norms, ml.pca, ml.eval-metrics]
sources:
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/16-anomaly-detection/docs/en.md (anomaly types, unsupervised framing, z-score/IQR, isolation forest, LOF, evaluation challenges, pipeline). READ."
---

## Where this fits

A card used in two countries five minutes apart, a sensor reading far outside its usual band, a server suddenly answering a thousand times more requests. Such events are rare, costly, and different every time. You cannot gather thousands of labelled examples of "tomorrow's fraud", so the classifiers of [[cml.02]] and [[cml.10]] have little to learn from. The standard answer is to flip the problem: **learn what normal looks like and flag whatever does not fit**.

This lesson builds a ladder of detectors. A z-score works on one number; Mahalanobis distance handles correlated features (it uses the covariance matrix of [[cml.05]]); an isolation forest makes no distributional assumption; local outlier factor adapts to regions of different density. Then comes the part most tutorials skip: how to evaluate a detector when labels are scarce or missing, and what the word "contamination" really means.

## The idea in plain words

Imagine a crowd at a station. If you must find one person to hand a note to, and you ask random yes/no questions ("is the person left of this pillar?"), then someone standing far from everybody is singled out after a couple of questions, while someone in the middle of the crowd needs many more. **Isolation forest** is exactly that game: the points that are easiest to isolate with random cuts are the suspicious ones.

The other detectors phrase the same intuition differently:

- **z-score**: "how many typical spreads away from the average is this value?"
- **Mahalanobis distance**: the same, but the notion of "typical spread" is a tilted ellipse that follows how the features move together. A person who is average in height and average in weight but with an impossible *combination* is caught here and missed by per-feature z-scores.
- **Local outlier factor (LOF)**: "is this point lonelier than its own neighbours?" A point that is ordinary in a sparse region is fine; the same distance from a dense cluster is not.

Three kinds of anomaly matter: **point** anomalies (one value is odd), **contextual** ones (90 degrees is odd in winter, not in summer: you need time or location features) and **collective** ones (fifty failed logins in a row, each single event harmless). Most of the detectors below find point anomalies; contextual and collective ones need features that carry the context (rolling statistics, lags as in [[cml.13]]).

## Z-score and its failure mode

For a one-dimensional sample with mean $\mu$ and standard deviation $\sigma$, the **z-score** of $x$ is $z=(x-\mu)/\sigma$, and a common rule flags $|z|>3$. The weakness is **masking**: the outlier itself inflates $\mu$ and $\sigma$ and hides from the very test that should catch it. The **robust z-score** replaces the mean by the median and the standard deviation by the **MAD** (median absolute deviation, $\operatorname{MAD}=\operatorname{median}|x_i-\tilde x|$):

$$
z^{\text{rob}}_i=0.6745\,\frac{x_i-\tilde x}{\operatorname{MAD}} .
$$

The constant 0.6745 makes the MAD comparable to $\sigma$ for Gaussian data (it is the 75th percentile of the standard normal). The median and MAD barely move when a few extreme points are present.

## Mahalanobis distance

For $X\in\mathbb R^{N\times d}$ with mean $\mu$ and covariance $\Sigma\in\mathbb R^{d\times d}$ (symmetric positive definite), the **Mahalanobis distance** of a point $x$ is

$$
D(x)=\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)} .
$$

### Why this is the right ruler

Write $\Sigma=LL^\top$ (Cholesky, or via the eigendecomposition $\Sigma=Q\Lambda Q^\top$, see [[cml.05]]). Define $u=L^{-1}(x-\mu)$. This **whitens** the data: $\operatorname{Cov}(u)=L^{-1}\Sigma L^{-\top}=I$, so $u$ has uncorrelated, unit-variance coordinates. Then

$$
\|u\|^2=(x-\mu)^\top L^{-\top}L^{-1}(x-\mu)=(x-\mu)^\top\Sigma^{-1}(x-\mu)=D(x)^2 .
$$

So $D$ is simply the ordinary Euclidean distance after whitening, i.e. a z-score generalised to correlated features. If the data are Gaussian, $D^2$ follows a chi-square distribution with $d$ degrees of freedom and mean $d$, which gives a principled cut-off (for example, the 99th percentile of $\chi^2_d$). On the sample itself the average of $D^2$ equals $d\,(N-1)/N$ exactly when $\Sigma$ is estimated with divisor $N-1$, a handy sanity check.

Weakness: the mean and covariance are themselves contaminated by outliers (same masking problem), the covariance cannot be inverted when features are collinear or $d>N$, and everything assumes one elliptical blob.

## Isolation forest

Build many **isolation trees**. Each tree uses a small random subsample of $\psi$ rows (256 is a typical default). A node picks a random feature and a random threshold uniformly between that feature's minimum and maximum in the node, and splits. Stop at depth $\lceil\log_2\psi\rceil$ or when a single point remains. The **path length** $h(x)$ is the number of splits needed to reach the leaf containing $x$.

Anomalies are few and different, so random cuts separate them early: short average path. To turn path length into a comparable score we need to know what "typical" is. A tree built on $n$ points is like a binary search tree, whose average path length for an unsuccessful search is

$$
c(n)=2\big(\ln(n-1)+\gamma\big)-\frac{2(n-1)}{n},\qquad \gamma\approx0.5772,
$$

with $c(2)=1$ and $c(n)=0$ for $n\le1$. When a leaf still holds $s>1$ points (depth limit reached) we add $c(s)$ as an estimate of the splits that would have been needed. The **anomaly score** is

$$
s(x)=2^{-\,\mathbb E[h(x)]/c(\psi)} ,
$$

so $E[h]=c(\psi)$ gives $0.5$ (nothing special), very short paths give scores near $1$, very long paths give scores well below $0.5$. Strengths: no distribution assumption, cost grows with the subsample size not the dataset, works with many features. Weaknesses: axis-parallel cuts struggle with diagonal structures, and a cluster of anomalies can hide each other.

## Local outlier factor (idea only)

For each point find its $k$ nearest neighbours and define a local density (roughly the inverse of the average distance to those neighbours). LOF is the ratio of the neighbours' densities to the point's own: about $1$ means "as dense as my neighbourhood", clearly above $1$ means "much sparser than the points around me". It finds anomalies sitting next to a dense cluster that a global threshold would miss, at the price of $O(N^2)$ distance computations, sensitivity to $k$ and the curse of dimensionality for distances ([[cml.09]]).

## Evaluation without (many) labels

- **Accuracy is useless** at 0.1% anomalies: predicting "normal" scores 99.9%.
- With a few labels, rank by score and report **precision@k** (of the $k$ top-scored items, how many are real anomalies) and the area under the precision-recall curve ([[cml.10]]); ROC-AUC can look rosy under heavy imbalance.
- With no labels: **inject synthetic anomalies** (shifted or swapped values) and check they are recovered; check **stability** (do two seeds, or two subsamples, flag the same points?); compare detectors by **rank agreement**; and above all send the top of the list to a human and measure the hit rate. Labels from this feedback loop are your future evaluation set.
- Evaluate on data **after** the training period (walk-forward, [[cml.13]]): the definition of normal drifts.

## Contamination

Many libraries take a parameter `contamination`: the assumed fraction of anomalies. It does **not** change the scores. It only places the threshold: flag the top $\lceil \text{contamination}\cdot N\rceil$ scores. So it is really a decision about how many alerts you can afford to review. Pick it from the cost of a false alarm versus a missed event, not from wishful knowledge of the true rate. Fitting a detector on data that already contains anomalies (the usual situation) makes methods based on means and covariances less reliable, which motivates robust statistics and the subsampling used by isolation forests.

## Worked example

The ten numbers $(10,11,9,10,12,10,11,9,10,50)$ have mean $14.2$ and standard deviation $11.96$, so the obvious outlier 50 has $z=2.99<3$: **masked**. The median is $10$ and the MAD is $1$, so its robust z-score is $0.6745\cdot40=26.98$.

For a cloud with unit variances and correlation $0.9$, the point $(1,1)$ has Mahalanobis distance $1.03$ (it follows the correlation), while $(1,-1)$ has distance $4.47$ even though each coordinate alone is perfectly ordinary ($|z|=1$).

For $\psi=256$, $c(256)=10.2448$. On 200 standard-normal points in 2-D plus three planted points $(6,6),(-5,5),(0,7)$, a 100-tree forest with $\psi=64$ (seed 0) gives the three planted points the three highest scores (about 0.78, 0.75, 0.69) while the median score is about 0.43. The exercise reproduces these.

## Implementation

```python
import numpy as np

def mahalanobis(X):
    Xc = X - X.mean(axis=0)
    Sinv = np.linalg.inv(np.cov(X, rowvar=False))
    return np.sqrt(np.einsum("ij,jk,ik->i", Xc, Sinv, Xc))

def c_factor(n):
    if n <= 1: return 0.0
    if n == 2: return 1.0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
```

The exercise adds a complete `IsolationForest` (random trees on subsamples, score $2^{-E[h]/c(\psi)}$). Production code (`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `sklearn.covariance.MinCovDet` for a robust covariance) adds speed and parallelism.

## Shapes

`X: (N, d)`, `Xc: (N, d)`, `Sigma: (d, d)`, distances `(N,)`. A forest holds `n_trees` trees, each trained on `(psi, d)` rows; `score(X)` returns `(N,)`.

## Common misconceptions

- **"Anomaly means error."** An anomaly is merely rare; it may be the most valuable event in the data.
- **"Contamination changes the model."** It only sets the threshold.
- **"|z|>3 is a safe rule."** It is masked by the outliers themselves and assumes a roughly Gaussian shape.
- **"One detector fits all."** Choose by data: Mahalanobis for one elliptical blob, isolation forest for generic tabular data, LOF for varying densities.
- **"High AUC proves it works."** With 0.1% anomalies, look at precision at the number of alerts you can actually review.

## Key takeaways

- Model normality, rank by a score, then choose a threshold as a business decision.
- Robust statistics (median, MAD) defeat masking; Mahalanobis distance is a whitened z-score.
- Isolation forests score by average random-cut path length normalised by $c(\psi)$.
- Evaluate with precision@k, synthetic injections, stability and human feedback, on later data.

## Explain it yourself

1. Why do points that are easy to isolate with random splits deserve a high anomaly score?
2. Show with whitening why the squared Mahalanobis distance of Gaussian data has mean $d$.

## Further reading

- The source lesson `phases/02-ml-fundamentals/16-anomaly-detection` for IQR fences, one-class SVM and ensemble detectors.
- scikit-learn user guide, "Novelty and outlier detection".

## Mastery criteria

You can explain masking and fix it with robust statistics, derive the Mahalanobis distance by whitening, implement an isolation forest and its score normalisation, and design an evaluation for a detector with very few labels.
