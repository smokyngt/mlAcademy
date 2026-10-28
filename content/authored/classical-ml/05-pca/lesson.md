---
id: cml.05
title: Principal component analysis
summary: Find the few directions that carry most of the variance, derive them with a Lagrange multiplier, and measure exactly what you lose by dropping the rest.
minutes: 65
teaches: [ml.pca]
requires: [la.matrix, la.dot, ml.kmeans]
sources:
  - math/math for ML.pdf (Ch. 10 Dimensionality Reduction with Principal Component Analysis, PDF p.323)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 12.1 Principal Component Analysis, PDF p.581; Appendix E Lagrange Multipliers, PDF p.727
  - ai-engineering-from-scratch/phases/01-math-foundations/10-dimensionality-reduction/docs/en.md
  - ai-engineering-from-scratch/phases/01-math-foundations/11-singular-value-decomposition/docs/en.md
---

## Where this fits

Real data has many columns, but they are rarely independent: height and weight move together, neighbouring pixels are nearly equal, hundreds of word counts reflect a few topics. Working with a few well-chosen combinations of the columns makes data easier to **visualise** (project 100 dimensions to 2), **cheaper** to process, and often **less noisy**. PCA finds those combinations. It is also the first time you use eigenvectors, which appear again in spectral methods, in the analysis of training dynamics and in low-rank adaptation of large models.

## The idea in plain words

Picture a cloud of points shaped like a long thin cigar tilted in the plane. If you must describe each point with **one** number, you should record where it lies *along the cigar*, because that is where the points differ the most; the position across the cigar is nearly the same for everyone and carries little information. PCA finds that long axis, then the next-longest axis perpendicular to it, and so on. Keeping the first few axes keeps most of the spread (variance) of the data.

## Setup

Centre the data: subtract the column means, so `X ∈ R^{N×d}` has zero-mean columns. The **covariance matrix**

$$
C=\frac{1}{N-1}X^\top X\in\mathbb{R}^{d\times d}
$$

is symmetric; `C_{jk}` is the covariance of features `j` and `k`, and `C_{jj}` is the variance of feature `j`.

## Derivation: the best direction is an eigenvector

Project the data on a unit direction `w` (`‖w‖ = 1`): the projections are `Xw ∈ R^N`. Their variance is

$$
\operatorname{Var}(Xw)=\frac{1}{N-1}\|Xw\|^2=w^\top C\,w .
$$

Maximise `wᵀCw` subject to `wᵀw = 1`. Use a **Lagrange multiplier** `λ` (the method for constrained optimisation: at a constrained optimum the gradient of the objective is parallel to the gradient of the constraint). The Lagrangian is `ℒ = wᵀCw − λ(wᵀw − 1)`. Its gradient is

$$
\nabla_w\mathcal{L}=2Cw-2\lambda w=0\;\Longrightarrow\;\boxed{Cw=\lambda w}.
$$

So `w` is an **eigenvector** of `C`, and the variance along it is `wᵀCw = λ wᵀw = λ`, the eigenvalue. To maximise the variance pick the eigenvector with the **largest eigenvalue**. Repeating for the next direction perpendicular to the first gives the second-largest eigenvector, and so on. Because `C` is symmetric, its eigenvectors are orthogonal and its eigenvalues real and non-negative.

**Explained variance.** The total variance is the sum of the diagonal of `C`, which equals the sum of all eigenvalues. Keeping `k` components retains `(λ₁+…+λ_k) / (λ₁+…+λ_d)` of it.

**Reconstruction error.** With `W_k ∈ R^{d×k}` the top `k` eigenvectors, the compressed data is `Z = X W_k ∈ R^{N×k}` and the reconstruction is `X̂ = Z W_kᵀ`. The squared error summed over all points equals

$$
\|X-\hat X\|_F^2=(N-1)\,(\lambda_{k+1}+\dots+\lambda_d),
$$

exactly the variance you threw away. PCA is the best rank-`k` linear compression in the least-squares sense.

**Link to the SVD.** If `X = UΣVᵀ` is the singular value decomposition, then `XᵀX = VΣ²Vᵀ`: the columns of `V` are the principal directions and `λ_i = σ_i²/(N−1)`. Numerically, the SVD of `X` is more stable than forming `C`.

## Worked example

Centred points `(2,2), (−2,−2), (1,−1), (−1,1)` (`N = 4`).

- `XᵀX = [[10,6],[6,10]]`, so `C = XᵀX/3 = [[3.333, 2],[2, 3.333]]`.
- Eigen-decomposition of `[[10,6],[6,10]]`: eigenvalues `16` and `4`, eigenvectors `(1,1)/√2` and `(1,−1)/√2`. So `λ₁ = 16/3 = 5.333`, `λ₂ = 4/3 = 1.333`.
- Explained variance of the first component: `16/20 = 0.8`.
- Keep `k = 1`: the projections are `Xw₁ = (2.83, −2.83, 0, 0)`; reconstruction error `= 4` (the dropped eigenvalue of `XᵀX`), which is `(N−1)λ₂ = 3 · 1.333`. ✓

## Implementation

```python
import numpy as np

def pca_fit(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    C = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)          # ascending, symmetric matrix
    order = np.argsort(vals)[::-1]
    return mean, vecs[:, order[:k]].T, vals[order]
```

## Shapes

`X: (N, d)`, `C: (d, d)`, components `(k, d)` (rows are directions), projection `Z = Xc @ comps.T : (N, k)`, reconstruction `Z @ comps + mean : (N, d)`.

## Common misconceptions

- **Forgetting to centre** makes the first component point to the mean instead of the direction of spread.
- **Scale matters.** PCA on raw features is dominated by the one with the biggest units. Standardise first unless units are comparable.
- **Signs are arbitrary**: `w` and `−w` are the same direction; do not compare components by sign.
- **High variance is not the same as importance for your task.** PCA is unsupervised; a low-variance direction can be the one that separates your classes.
- **It is linear.** Curved structures (a spiral, a swiss roll) need non-linear methods.

## Key takeaways

- Principal directions are eigenvectors of the covariance matrix; eigenvalues are the variances along them.
- Keep the top `k`: retained variance is the share of the eigenvalue sum; reconstruction error is exactly the dropped eigenvalues.
- Centre first, usually standardise, and prefer the SVD for numerical stability.

## Explain it yourself

1. Derive `Cw = λw` from maximising `wᵀCw` under `‖w‖ = 1`.
2. Why does the reconstruction error equal the sum of the discarded eigenvalues?

## Further reading

- *Mathematics for Machine Learning*, Ch. 10 (PDF p.323); Bishop, *PRML*, Sec. 12.1.
- Source library: `ai-engineering-from-scratch` phase 01, lessons 10-11.

## Mastery criteria

You can derive PCA with a Lagrange multiplier, implement it from `eigh` and from the SVD, and quantify the information lost for a given `k`.
