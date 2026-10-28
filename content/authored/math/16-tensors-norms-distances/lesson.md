---
id: math.16
title: Tensors, norms and distances
summary: The shapes data takes in deep learning, the rules that let arrays of different shapes combine, and the norms, distances and similarities used to measure size and closeness, including why distances lose meaning in high dimensions.
minutes: 75
teaches: [la.tensor, la.norms]
requires: [la.vector, la.dot, la.matrix, la.svd, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/12-tensor-operations/docs/en.md: tensor rank/shape, typical deep-learning shapes (B,C,H,W), (B,T,D), (B,H,T,D), strides and contiguity, broadcasting rule with the (8,1,6,1) x (7,1,5) example, einsum patterns. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/14-norms-and-distances/docs/en.md: L1, L2, Lp, Linf norms, cosine similarity, Mahalanobis distance (definition and intuition via whitening), links to MAE/MSE and L1/L2 regularisation. READ."
  - "math/math for ML.pdf, Sec. 3.1 Norms (PDF p.77-78, book p.71-72): Definition 3.1 (absolutely homogeneous, triangle inequality, positive definite), Manhattan norm, Euclidean norm. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.4 The Curse of Dimensionality (PDF p.53-57): exponential growth of grid cells, volume of a sphere concentrated in a thin shell 1-(1-eps)^D, Gaussian mass in a thin shell, why learning still works (low effective dimension, smoothness). READ."
---

## Where this fits

Every number a neural network touches lives in an array. An image batch is four-dimensional, a batch of sentences is three-dimensional, an attention score table is four-dimensional. Writing and debugging deep learning code is, to a large extent, **keeping track of shapes**: which axis is the batch, which is the feature, and how the arrays line up when you add or multiply them. The first half of this lesson gives you that vocabulary: tensors, shapes, broadcasting and einsum.

The second half answers a different but related question: **how big is a vector, and how far apart are two vectors?** The loss functions of lesson 8, the regularisers of the ridge and lasso lessons, nearest-neighbour search, embeddings and retrieval, anomaly detection and clustering all need a measure of size or closeness, and the choice of measure changes the answer. We finish with the surprising fact that in many dimensions "nearest" and "farthest" become almost the same thing, which is why high-dimensional data forces careful thinking.

You will use lessons 2 and 3 (vectors, dot products, matrices), 7 (singular values appear in the spectral norm) and 9 (covariance for the Mahalanobis distance).

## The idea in plain words

- **A tensor is a spreadsheet generalised to any number of axes.** A single number is a point. A list of numbers is a column. A table is a grid with rows and columns. A stack of tables (for instance one table per image colour channel, or one per example in a batch) adds a third axis, and so on. The **shape** is the list of lengths along each axis: a batch of 32 colour images of size 224 by 224 has shape `(32, 3, 224, 224)`. The shape is the contract: operations are legal or not depending on it.
- **Broadcasting is "stretch the small thing to fit".** If you add a single price to a whole column of prices, or one bias vector to every row of a batch, you do not want to copy it manually. NumPy conceptually stretches the smaller array along axes where its length is 1 (or missing) so that the shapes match, without actually copying memory.
- **Einsum is bookkeeping by labels.** Give every axis a letter; letters that appear on the input but not on the output are summed over. Matrix multiplication, dot products, traces and attention scores all become one-line label statements.
- **A norm is a ruler.** Different rulers give different lengths to the same arrow. The "walk along city blocks" ruler (L1), the "fly in a straight line" ruler (L2) and the "only the worst direction counts" ruler (L-infinity) disagree. The **distance** between two points is the length of the arrow between them.
- **Cosine similarity compares directions, not sizes.** Two documents with the same topic but one ten times longer have vectors pointing the same way but of different lengths; the angle sees through that.
- **Mahalanobis distance is distance measured in units of the data's own spread.** A person 20 cm taller than average is not remarkable among basketball players, but notable in the general population. And someone unusually tall *and* unusually light is more surprising than the individual deviations suggest, because height and weight go together. Mahalanobis accounts for both scale and correlation.
- **High dimensions are strange.** In a thousand dimensions, a random point is at nearly the same distance from every other random point. Nearly all the volume of a ball sits in a thin skin near its surface.

## Tensors and shapes

A **tensor** is a multidimensional array of numbers of one type. Its **rank** (number of axes) and **shape** (tuple of lengths):

| object | shape | rank |
|---|---|---|
| scalar | `()` | 0 |
| vector in $\mathbb R^3$ | `(3,)` | 1 |
| matrix $2\times3$ | `(2, 3)` | 2 |
| batch of images | `(B, C, H, W)` | 4 |
| batch of token embeddings | `(B, T, D)` | 3 |
| attention scores | `(B, heads, T, T)` | 4 |
| linear layer weights | `(out, in)` | 2 |

The number of elements is the product of the sizes: shape `(2, 3, 4)` has 24. `reshape` re-groups the same elements in the same memory order (the total must match; one size may be `-1` meaning "infer it"). `transpose` and `permute` reorder axes. In memory a tensor is a flat block and **strides** say how many elements to skip to advance one step along each axis: shape `(3, 4)` row-major has strides `(4, 1)`. Transposing swaps the strides without moving data, which is why a transposed array is not contiguous and why PyTorch `view` can fail where `reshape` works.

**Reductions** (`sum`, `mean`, `max`) collapse the axes you name: mean over `(B, T, D)` along `axis=1` gives `(B, D)`; global average pooling of `(B, C, H, W)` over axes `(2, 3)` gives `(B, C)`.

## Broadcasting

**Rule.** Align the shapes from the **right**. Missing leading axes count as size 1. Two aligned sizes are compatible if they are equal or one of them is 1; the result takes the larger. Otherwise it is an error.

Example (the repo lesson's): `(8, 1, 6, 1)` with `(7, 1, 5)`. Pad the second to `(1, 7, 1, 5)`, compare axis by axis from the right: $1$ vs $5\to5$, $6$ vs $1\to6$, $1$ vs $7\to7$, $8$ vs $1\to8$. Result shape `(8, 7, 6, 5)`.

Common patterns: bias `(D,)` added to activations `(B, T, D)`; standardising `X - X.mean(axis=0)` where the mean has shape `(D,)` against `(N, D)`; and the classic bug where `(N,)` minus `(N, 1)` silently broadcasts to `(N, N)`. When keeping a reduced axis for later broadcasting use `keepdims=True` (so the max in a softmax has shape `(B, 1)`, not `(B,)`).

## Einsum

`np.einsum("subscripts->output", ...)`. Rule: a letter missing from the output is summed over; a letter repeated within one operand picks the diagonal.

| subscripts | meaning |
|---|---|
| `i,i->` | dot product |
| `i,j->ij` | outer product |
| `ii->` | trace |
| `ij->ji` | transpose |
| `ik,kj->ij` | matrix product |
| `bij,bjk->bik` | batched matrix product |
| `bhtd,bhsd->bhts` | attention scores: queries `(B,H,T,D)` against keys `(B,H,S,D)` |

For matrix multiplication, `ik,kj->ij` literally says $C_{ij}=\sum_k A_{ik}B_{kj}$: the index $k$ does not appear on the right, so it is summed. Reading a model's code becomes much easier once you can translate a subscript string into a sum formula.

## Norms

A **norm** is a function $\|\cdot\|$ assigning a length to each vector such that (Deisenroth Def. 3.1), for all scalars $\lambda$ and vectors $x,y$:

- absolutely homogeneous: $\|\lambda x\|=|\lambda|\,\|x\|$;
- triangle inequality: $\|x+y\|\le\|x\|+\|y\|$;
- positive definite: $\|x\|\ge0$ and $\|x\|=0\iff x=0$.

The $\ell_p$ family: $\|x\|_p=\big(\sum_i|x_i|^p\big)^{1/p}$ for $p\ge1$.

| norm | formula | unit ball in 2-D | where it shows up |
|---|---|---|---|
| L1 (Manhattan) | $\sum_i\lvert x_i\rvert$ | diamond | lasso (sparsity), MAE loss |
| L2 (Euclidean) | $\sqrt{\sum_i x_i^2}=\sqrt{x^\top x}$ | circle | ridge, MSE, default |
| $L_\infty$ (Chebyshev) | $\max_i\lvert x_i\rvert$ | square | worst-case error, adversarial perturbation bounds |

$\|x\|_\infty$ is the limit of $\|x\|_p$ as $p\to\infty$: the largest component dominates more and more. For $x=(3,-4,12)$: $\|x\|_1=19$, $\|x\|_2=13$, $\|x\|_\infty=12$, $\|x\|_3=12.21$. Norms satisfy $\|x\|_\infty\le\|x\|_2\le\|x\|_1$. The diamond has corners on the axes, which is the geometric reason an L1 penalty drives coordinates to exactly zero while an L2 penalty only shrinks them.

### Matrix norms

- **Frobenius**: treat the matrix as a long vector, $\|A\|_F=\sqrt{\sum_{ij}a_{ij}^2}=\sqrt{\operatorname{tr}(A^\top A)}=\sqrt{\sum_k\sigma_k^2}$ (the equalities use the SVD of lesson 7: $A^\top A=V\Sigma^\top\Sigma V^\top$ has eigenvalues $\sigma_k^2$, and the trace is their sum).
- **Spectral (operator 2-norm)**: the largest stretch the matrix can apply to a unit vector, $\|A\|_2=\max_{\|x\|_2=1}\|Ax\|_2=\sigma_{\max}$.
- Also used: the max column sum (operator 1-norm) and max row sum (operator $\infty$-norm).

For $A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$: singular values $5.4650$ and $0.3660$, so $\|A\|_2=5.4650$ and $\|A\|_F=\sqrt{30}=5.4772$ (check: $5.4650^2+0.3660^2=29.866+0.134=30$). The spectral norm is what controls how much a layer can amplify signals and gradients, and it is the quantity behind spectral normalisation and the ratio $\sigma_{\max}/\sigma_{\min}$ of lesson 15's condition number.

## Distances and similarities

Any norm gives a distance: $d(a,b)=\|a-b\|$.

**Euclidean** $d_2(a,b)=\|a-b\|_2$. **Manhattan** $d_1(a,b)=\sum_i|a_i-b_i|$. For $a=(1,1)$, $b=(4,5)$: $d_1=7$, $d_2=5$, $d_\infty=4$.

**Cosine similarity** measures the angle (it reuses the dot product identity $a\cdot b=\|a\|\|b\|\cos\theta$ of lesson 2):

$$
\cos(a,b)=\frac{a^\top b}{\|a\|_2\,\|b\|_2}\in[-1,1],\qquad d_{\cos}=1-\cos(a,b).
$$

It ignores scale: $(1,2,3)$ and $(2,4,6)$ have cosine 1 though their Euclidean distance is $\sqrt{14}=3.74$. Once vectors are normalised to unit length, $\|a-b\|_2^2=2-2\cos(a,b)$, so Euclidean ranking and cosine ranking coincide. (Derivation: $\|a-b\|^2=\|a\|^2+\|b\|^2-2a^\top b=1+1-2\cos$.) Note cosine distance does not satisfy the triangle inequality, so it is not a metric in the strict sense.

**Mahalanobis distance** for data with mean $\mu$ and covariance $\Sigma$:

$$
d_M(x,\mu)=\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)} .
$$

*Why this formula.* Factor $\Sigma=U\Lambda U^\top$ (eigendecomposition, lesson 7). Then $\Sigma^{-1/2}=U\Lambda^{-1/2}U^\top$ and $d_M=\|\Sigma^{-1/2}(x-\mu)\|_2$: rotate onto the principal axes, divide each coordinate by its standard deviation $\sqrt{\lambda_k}$, then take the ordinary Euclidean length. The data cloud becomes a round ball ("whitening"). When $\Sigma=I$, it is plain Euclidean distance.

**Worked example.** $\Sigma=\begin{pmatrix}4&2\\2&3\end{pmatrix}$ (positively correlated, $\mu=0$), $\det\Sigma=8$, $\Sigma^{-1}=\frac18\begin{pmatrix}3&-2\\-2&4\end{pmatrix}$.

| point | Euclidean norm | Mahalanobis |
|---|---|---|
| $(2,0)$ | 2.000 | $\sqrt{4\cdot3/8}=1.2247$ |
| $(0,2)$ | 2.000 | $\sqrt{4\cdot4/8}=1.4142$ |
| $(2,2)$ | 2.828 | $\sqrt{(12-16+16)/8}=1.2247$ |
| $(2,-2)$ | 2.828 | $\sqrt{(12+16+16)/8}=2.3452$ |

(For $(2,2)$: $x^\top\Sigma^{-1}x=\frac18(3\cdot4-2\cdot2\cdot4+4\cdot4)=\frac{12}{8}$, giving $1.2247$; for $(2,-2)$: $\frac18(12+16+16)=5.5$, giving $2.3452$.) The point $(2,2)$ is farther from the origin in Euclidean terms than $(2,0)$ yet equally typical, because it lies along the direction in which the data naturally spreads. The point $(2,-2)$ goes against the correlation and is the real outlier. This is why Mahalanobis distance is used for outlier detection, and why Gaussian densities depend on it (the exponent in the multivariate normal is $-\tfrac12 d_M^2$).

## The curse of dimensionality

Bishop (Sec. 1.4) makes three related points.

1. **Grids explode.** Dividing each of $D$ axes into $k$ cells gives $k^D$ cells; you would need exponentially much data for them all to contain examples. A polynomial of order $M$ in $D$ variables has about $D^M$ coefficients.
2. **Volume goes to the shell.** The volume of a $D$-ball of radius $r$ scales as $r^D$. The fraction of the unit ball's volume lying within $\varepsilon$ of the surface is $\dfrac{V(1)-V(1-\varepsilon)}{V(1)}=1-(1-\varepsilon)^D$. With $\varepsilon=0.05$: $D=2$ gives $9.8\%$, $D=20$ gives $64.2\%$, $D=100$ gives $99.4\%$. A Gaussian's probability mass also concentrates at radius about $\sqrt D$: for standard normal draws the norm has mean $\approx\sqrt D$ (10.0 for $D=100$) with standard deviation about $0.7$, nearly independent of $D$.
3. **Distances concentrate.** For random points the distance to the nearest and to the farthest neighbour become almost equal. Take 200 uniform points in the unit cube and a random query $q$; the relative contrast $(d_{\max}-d_{\min})/d_{\min}$ is $230.6$ for $D=2$, $2.20$ for $D=10$, $0.316$ for $D=100$, $0.118$ for $D=1000$ and $0.029$ for $D=10000$ (seed 0, reproduced in the exercise). Intuitively, each distance squared is a sum of $D$ independent coordinate terms, so by the law of large numbers it concentrates around its mean $\propto D$ while the spread grows only like $\sqrt D$. Nearest-neighbour search and any method that relies on raw Euclidean closeness degrade.

Why machine learning still works (Bishop p.37): real data lie near a lower-dimensional manifold, and the target function is usually smooth locally. This is why we reduce dimension with PCA (lesson 7 and the PCA lesson), learn embeddings, and often prefer cosine or learned metrics over raw Euclidean distance.

## Implementation

```python
import numpy as np

def lp_norm(x, p):
    return np.max(np.abs(x)) if np.isinf(p) else (np.abs(x) ** p).sum() ** (1 / p)

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

def mahalanobis(x, mu, S):
    d = x - mu
    return np.sqrt(d @ np.linalg.solve(S, d))   # solve instead of an explicit inverse

# attention scores: queries (B,H,T,D), keys (B,H,S,D) -> (B,H,T,S)
# scores = np.einsum("bhtd,bhsd->bhts", Q, K)
```

Use `np.linalg.solve` rather than `np.linalg.inv(S) @ d`: it is faster and more accurate (lesson 15).

## Shapes

Pairwise Euclidean distances between `X: (N, D)` and `Y: (M, D)`: expand the square, $\|x-y\|^2=\|x\|^2+\|y\|^2-2x^\top y$, so `D2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T` with shapes `(N,1)`, `(1,M)` and `(N,M)` broadcasting to `(N, M)`. (Tiny negative values can appear from cancellation; clip at 0 before the square root.) Cosine similarity matrix of row-normalised `Xn: (N, D)`, `Yn: (M, D)` is `Xn @ Yn.T: (N, M)`.

## Common misconceptions

- **"Broadcasting copies data."** It does not; it re-uses memory by giving the stretched axis stride 0. But it can hide bugs such as `(N,)` minus `(N,1)` giving `(N,N)`.
- **"L2 distance is always the right distance."** Different tasks want different metrics: L1 for robustness and sparsity, cosine for direction, Mahalanobis for correlated or differently scaled features.
- **"Cosine similarity of 0 means unrelated."** It means orthogonal; for centred data that is zero correlation, for non-negative data (word counts) it means no shared features.
- **"More features always help nearest-neighbour methods."** Beyond the effective dimension, extra noisy features wash out the contrast between near and far.
- **"Frobenius and spectral norms are the same."** $\|A\|_F^2=\sum\sigma_k^2$ while $\|A\|_2=\sigma_{\max}$; they coincide only for rank-one matrices.

## Key takeaways

- Shape is the contract. Broadcasting aligns from the right and stretches size-1 axes; einsum letters absent from the output are summed.
- $\|x\|_\infty\le\|x\|_2\le\|x\|_1$; L1 has corners (sparsity), L2 is round (shrinkage). $\|A\|_F=\sqrt{\sum\sigma^2}$, $\|A\|_2=\sigma_{\max}$.
- Cosine ignores length; for unit vectors $\|a-b\|^2=2-2\cos$. Mahalanobis whitens by $\Sigma^{-1/2}$.
- In high dimensions volume sits in a thin shell and distances concentrate; rely on low effective dimension and smoothness.

## Explain it yourself

1. A teammate adds a `(32,)` vector of per-example weights to a `(32, 1)` column of losses and gets a `(32, 32)` matrix. Explain exactly what broadcasting did and how to fix it.
2. Explain to a non-specialist why two points that are 2 units apart in Euclidean distance can be "equally typical" as the mean for one pair and a clear outlier for another, and which distance fixes this.

## Further reading

- Deisenroth et al., Ch. 3 (inner products, norms, angles, orthogonal projections).
- Bishop, Sec. 1.4 and its Figures 1.21-1.23.
- Repo phase 01, lessons 12 and 14 (more distances: Jaccard, edit distance, KL divergence, Wasserstein; memory layout and strides in depth).

## Mastery criteria

- Predict the output shape of a broadcast and of an einsum string without running them.
- Compute L1, L2, $L_\infty$, Frobenius and spectral norms and explain how Frobenius and spectral relate to singular values.
- Show that for unit vectors Euclidean distance is a monotone function of cosine.
- Compute a Mahalanobis distance and explain it as whitening followed by a Euclidean length.
- Explain concentration of distances and one reason learning is still possible in high dimensions.
