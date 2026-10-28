---
id: cml.04
title: k-means clustering
summary: Find groups in unlabelled data by alternating two simple steps, and prove the objective can only go down.
minutes: 50
teaches: [ml.kmeans]
requires: [la.dot, alg.summation]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 9.1 K-means Clustering, PDF p.444
  - math/math for ML.pdf (Ch. 11 Density Estimation with Gaussian Mixture Models, for the soft-assignment generalisation, PDF p.354)
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/07-unsupervised-learning/docs/en.md
---

## Where this fits

So far every data point came with an answer (a price, a label). Often there is no answer: you have the purchases of 100,000 customers and want to know whether natural groups exist, or you want to compress an image to 16 colours. This is **unsupervised learning**: find structure without labels. k-means is the simplest and most widely used method. It also introduces a pattern you will meet again (in Gaussian mixtures and in training with latent variables): *alternate between two easy steps, each of which cannot make the objective worse*.

## The idea in plain words

You want to open `k` pizza shops so that everyone lives close to one. Start by putting the shops anywhere. Then repeat two steps:

1. **Assign:** every person goes to their nearest shop.
2. **Move:** each shop relocates to the centre (the average position) of the people who chose it.

Moving a shop to the average of its customers can only reduce the total walking distance, and reassigning each person to the nearest shop can only reduce it too. So the total never goes up, and when nothing changes any more you are done.

## Formal objective

Data `x_1..x_N ∈ R^d`, `k` centres `μ_1..μ_k`, assignment `c(i) ∈ {1..k}`. The objective is the **within-cluster sum of squares** (inertia):

$$
J=\sum_{i=1}^{N}\|x_i-\mu_{c(i)}\|^2 .
$$

## Derivation: why each step helps

**Assignment step** (centres fixed). Each point contributes `‖x_i − μ_{c(i)}‖²` independently, so choosing `c(i)` as the nearest centre minimises `J`.

**Update step** (assignments fixed). For one cluster `S`, minimise `f(μ) = Σ_{i∈S} ‖x_i − μ‖²`. Its gradient (lesson 5) is `∇f = −2 Σ_{i∈S}(x_i − μ)`. Setting it to zero gives

$$
\mu=\frac{1}{|S|}\sum_{i\in S}x_i ,
$$

the mean. `f` is a convex quadratic, so this is the minimum.

Both steps can only lower or keep `J`, and `J ≥ 0`, so the sequence converges. There are finitely many possible assignments, and `J` strictly drops whenever an assignment changes, so the algorithm stops after finitely many iterations. **It stops at a local minimum, not necessarily the global one.**

## Worked example (1-D)

Points `0, 1, 2, 10, 11, 12`, `k = 2`, initial centres `0` and `1`.

- Assign: `0 → c₀`; `1, 2, 10, 11, 12 → c₁` (|2−1| < |2−0|). Update: `c₀ = 0`, `c₁ = (1+2+10+11+12)/5 = 7.2`.
- Assign: `0, 1, 2 → c₀` (e.g. |2−0| = 2 < |2−7.2|); `10, 11, 12 → c₁`. Update: `c₀ = 1`, `c₁ = 11`.
- Assign again: nothing changes. Converged with `J = (1+0+1) + (1+0+1) = 4`.

## Practical issues

- **Initialisation matters.** Bad starts give bad local minima (two centres in one true cluster). Remedy: run several restarts and keep the lowest `J`; or use k-means++, which picks initial centres spread apart (each new centre is drawn with probability proportional to its squared distance to the nearest chosen centre).
- **Empty clusters** can happen: keep the old centre (or re-seed it at the farthest point).
- **Choosing `k`**: `J` always falls as `k` grows, so plot `J` against `k` and look for an "elbow", or use a validation criterion; there is no universally correct answer.
- **Assumptions**: clusters are roughly round and similar in size and the scale of features is comparable. Standardise features first. For elongated or nested shapes use other methods.

## Implementation

```python
import numpy as np

def assign(X, centers):
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (N, k)
    return d2.argmin(axis=1)

def update(X, labels, centers):
    new = centers.copy()
    for j in range(len(centers)):
        pts = X[labels == j]
        if len(pts):
            new[j] = pts.mean(axis=0)
    return new
```

## Shapes

`X: (N, d)`, `centers: (k, d)`, pairwise squared distances `(N, k)` via broadcasting `(N,1,d) − (1,k,d)`, `labels: (N)`.

## Common misconceptions

- **It always finds the same answer.** No; it depends on initialisation.
- **Lower inertia means better clusters.** Only for the same `k` and the same data; inertia always drops with more clusters.
- **It works for any data.** It uses Euclidean distance and means; categorical or very non-spherical data needs another approach.

## Key takeaways

- Alternate "assign to nearest centre" and "move centre to the mean".
- Each step lowers (or keeps) the sum of squared distances; it converges to a local minimum.
- Restart several times; standardise features; choose `k` with care.

## Explain it yourself

1. Why is the mean the best centre for a fixed cluster? (Use the gradient.)
2. Why does the algorithm have to stop, and why is the result not necessarily the global optimum?

## Further reading

- Bishop, *PRML*, Sec. 9.1 (PDF p.444); *Mathematics for Machine Learning*, Ch. 11 for the soft version.
- Source library: `ai-engineering-from-scratch` phase 02, lesson 07.

## Mastery criteria

You can implement Lloyd's algorithm with broadcasting, show the objective is monotone, and diagnose a bad local minimum.
