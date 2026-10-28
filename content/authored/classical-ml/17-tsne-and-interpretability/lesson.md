---
id: cml.17
title: "Beyond PCA, and looking inside models: t-SNE, kernel PCA, permutation importance, partial dependence, Shapley values"
summary: Embed data non-linearly (t-SNE affinities and its gradient, kernel PCA), then explain a fitted model with permutation importance, partial dependence and Shapley values, and learn exactly what such explanations can and cannot tell you.
minutes: 90
teaches: [ml.tsne, ml.kpca, ml.interpret, ml.shap]
requires: [ml.pca, la.eigen, info.entropy, calc.gradient, ml.features]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/10-dimensionality-reduction/docs/en.md (curse of dimensionality, t-SNE properties and perplexity, kernel PCA algorithm and comparison table). READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/docs/en.md (permutation importance section, bias of tree importance). READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 12.3 Kernel PCA (PDF p.606-610): kernel-only eigenproblem K a = lambda N a, normalisation, centred Gram matrix, projections, N x N cost, linear kernel recovers PCA, pre-image problem. READ."
---

## Where this fits

[[cml.05]] gave you PCA, the best *linear* compression. Two things go wrong in practice. First, data often lies on a curved surface (a spiral, a ring, handwriting variations) that no flat subspace captures; **t-SNE** and **kernel PCA** are the standard non-linear answers, the first for making a 2-D picture, the second as a feature extractor. Second, once you have a good model (a forest, a boosted ensemble, a network) you must be able to say *which inputs it uses and how*. **Permutation importance**, **partial dependence** and **Shapley values** are model-agnostic probes that work on any fitted `predict` function.

Both halves answer the same question, "what is the structure here?", one about the data and one about the model. The second half also ties to feature selection ([[cml.16]]) and to leakage detection: a feature with suspiciously huge importance is often a leak.

## The idea in plain words

**t-SNE.** Suppose each data point is a person at a party and you ask everyone "who are your closest friends?". In the original, high-dimensional space each person spreads one unit of attention over the others, mostly on the nearest ones. Now ask a 2-D seating plan to reproduce those friendships: friends sit together, strangers apart, and where the plan is crowded, a heavy-tailed rule lets non-friends sit comfortably far away. t-SNE keeps neighbourhoods; it does not keep distances between groups, nor their sizes.

**Kernel PCA.** A ring cannot be flattened by any straight line. But if each point is described by *how similar it is to every other point* (a kernel), the ring becomes easy: do PCA on those similarities.

**Permutation importance.** To test whether a model uses a feature, scramble that column among the rows and see how much the error grows. If nothing changes, the model ignores it.

**Partial dependence.** Force a feature to each value on a grid for every row, average the predictions, and plot the averaged curve: the model's average response to that feature.

**Shapley values.** Treat the features as a team that jointly produced the prediction and split the credit fairly, by averaging each feature's marginal contribution over all orders in which the team could be assembled.

## t-SNE

Let $x_1,\dots,x_N\in\mathbb R^D$. The goal is points $y_1,\dots,y_N\in\mathbb R^2$.

**High-dimensional affinities.** For each point $i$ use a Gaussian centred on $x_i$:

$$
p_{j\mid i}=\frac{\exp\!\big(-\|x_i-x_j\|^2/2\sigma_i^2\big)}{\sum_{k\ne i}\exp\!\big(-\|x_i-x_k\|^2/2\sigma_i^2\big)},\qquad p_{i\mid i}=0 .
$$

The bandwidth $\sigma_i$ is chosen per point so that the **perplexity** $\mathrm{Perp}(P_i)=\exp\big(H(P_i)\big)$, with $H(P_i)=-\sum_jp_{j\mid i}\ln p_{j\mid i}$, equals a user setting (typically 5-50): it is the effective number of neighbours. Dense regions get small $\sigma_i$, sparse regions large $\sigma_i$. Because $H$ grows with $\sigma_i$, $\sigma_i$ (or $\beta_i=1/2\sigma_i^2$) is found by bisection. Then symmetrise so every point contributes:

$$
p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2N},\qquad \sum_{i\ne j}p_{ij}=1 .
$$

**Low-dimensional affinities.** With a Student-t kernel (one degree of freedom, heavy tails):

$$
w_{ij}=\frac1{1+\|y_i-y_j\|^2},\qquad q_{ij}=\frac{w_{ij}}{\sum_{k\ne l}w_{kl}} .
$$

Heavy tails solve the "crowding problem": a 2-D map has less room around a point than a 10-D space, and a heavy-tailed kernel lets moderately distant points be placed far apart without penalty.

**Objective.** Make $Q$ resemble $P$: minimise $C=\mathrm{KL}(P\|Q)=\sum_{i\ne j}p_{ij}\ln\frac{p_{ij}}{q_{ij}}$.

### Derivation of the gradient

Write $d_{ij}^2=\|y_i-y_j\|^2$ and $Z=\sum_{k\ne l}w_{kl}$. Since $\sum p_{ij}=1$,

$$
C=\sum p_{ij}\ln p_{ij}-\sum p_{ij}\ln w_{ij}+\ln Z .
$$

With $\ln w_{ij}=-\ln(1+d_{ij}^2)$: $\ \partial(-p_{ij}\ln w_{ij})/\partial d_{ij}^2=p_{ij}w_{ij}$, and $\partial w_{ij}/\partial d_{ij}^2=-w_{ij}^2$ so $\partial\ln Z/\partial d_{ij}^2=-w_{ij}^2/Z=-q_{ij}w_{ij}$. Each unordered pair appears twice in the sums, hence $\partial C/\partial d_{ij}^2=2(p_{ij}-q_{ij})w_{ij}$. Finally $\partial d_{ij}^2/\partial y_i=2(y_i-y_j)$, so

$$
\boxed{\frac{\partial C}{\partial y_i}=4\sum_{j}(p_{ij}-q_{ij})\,w_{ij}\,(y_i-y_j)}
$$

In matrix form with $M=(P-Q)\odot W$: $\ \nabla_Y C=4\big(\mathrm{diag}(M\mathbf 1)-M\big)Y$. It reads like a spring system: pairs with $p_{ij}>q_{ij}$ (too far apart in the map) attract, pairs with $p_{ij}<q_{ij}$ repel. One step of gradient descent is $Y\leftarrow Y-\eta\nabla_YC$ (real implementations add momentum, "early exaggeration" of $P$, and Barnes-Hut or FFT approximations to avoid the $O(N^2)$ cost).

I verified this formula numerically against finite differences in the exercise rather than citing it from the sources I read; run the check yourself.

**What t-SNE does not promise.** Distances *between* clusters, cluster sizes and densities are not meaningful; different seeds give different layouts; perplexity changes the picture (too low breaks clusters apart, too high merges them); the map is for looking, not for preprocessing a downstream model, and new points have no out-of-sample mapping. UMAP is a faster relative that keeps somewhat more global structure.

## Kernel PCA

Replace each point by a feature vector $\phi(x)$ and run PCA there, but only through the kernel $k(x,x')=\phi(x)^\top\phi(x')$. Following Bishop Sec. 12.3: if the feature-space eigenvectors are $v_i=\sum_na_{in}\phi(x_n)$, substituting into $Cv_i=\lambda_iv_i$ and multiplying by $\phi(x_l)^\top$ gives an eigenproblem of the $N\times N$ Gram matrix, $K\mathbf a_i=\lambda_iN\mathbf a_i$; requiring $\|v_i\|=1$ normalises $\lambda_iN\,\mathbf a_i^\top\mathbf a_i=1$. The projection of a point is $y_i(x)=\sum_na_{in}k(x,x_n)$. Because $\phi$ rarely has zero mean, the Gram matrix must be **centred**:

$$
\tilde K=K-\mathbf 1_NK-K\mathbf 1_N+\mathbf 1_NK\mathbf 1_N=JKJ,\qquad J=I-\tfrac1N\mathbf 1\mathbf 1^\top ,
$$

with $\mathbf 1_N$ the matrix of entries $1/N$. Practical recipe: build the RBF kernel $K_{ij}=\exp(-\gamma\|x_i-x_j\|^2)$, centre it, take the top eigenpairs $(\mu_i,u_i)$ of $\tilde K$; the training-set coordinates are $u_i\sqrt{\mu_i}$. A linear kernel recovers ordinary PCA. Costs: an $N\times N$ matrix (memory $O(N^2)$, eigen-decomposition $O(N^3)$), no direct reconstruction (the pre-image problem), and $\gamma$ must be tuned.

## Permutation importance

Given a fitted model, held-out data and a metric, the baseline score is $s_0$. For feature $j$ permute its column among the rows (breaking its link with $y$ while keeping its marginal distribution), recompute the score $s_j$, repeat several times and average: $\mathrm{imp}_j=\mathbb E[s_j-s_0]$ (for an error metric). Use held-out data: on the training set a model that overfits can make noise features look important. Caveats: with two strongly correlated features the model can fall back on the other copy, so both look unimportant; permuting creates unrealistic rows when features are dependent; importance measures *reliance of this model*, not the feature's real relevance. It avoids the cardinality bias of impurity-based tree importance.

## Partial dependence

$$
\mathrm{PD}_j(g)=\frac1N\sum_{n=1}^N f\big(x_n^{(1)},\dots,x_n^{(j)}\!=\!g,\dots,x_n^{(d)}\big).
$$

It shows the average effect of feature $j$ and is exact when the model is additive in $j$. Weaknesses: averages hide heterogeneity (a feature that helps half the rows and hurts the other half shows a flat line; plot individual curves, **ICE**, to see it), and with correlated features it evaluates the model on unrealistic points.

## Shapley values (the idea behind SHAP)

For one instance $x$ with $d$ features, define a value function $v(S)$ = expected prediction when the features in the coalition $S$ are fixed to $x$'s values and the others are drawn from a background dataset. The Shapley value of feature $j$ is its marginal contribution averaged over all coalitions of the other features:

$$
\phi_j=\sum_{S\subseteq\{1..d\}\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\Big[v(S\cup\{j\})-v(S)\Big].
$$

Properties: **efficiency** $\sum_j\phi_j=f(x)-\mathbb E[f]$ (the explanation adds up exactly), symmetry, a zero value for a feature that never matters, and linearity. For a linear model $f(x)=w^\top x$ it gives $\phi_j=w_j(x_j-\mathbb E x_j)$. Exact computation needs $2^{d}$ coalitions; SHAP libraries use sampling (KernelSHAP) or exploit tree structure (TreeSHAP). Choices matter: using a background distribution that ignores correlations versus conditional expectations gives different answers.

## Limits of interpretability

- Explanations describe the **model**, not the world: a feature with large attribution may be a proxy or a leak.
- They are **local** (one prediction) or **averaged** (one curve); neither reveals interactions fully.
- Correlated features share or hide credit; results depend on the background data and on the metric.
- Explanations can be unstable between retrainings, and can be used to look convincing (explanation washing).
- Use them for debugging (finding leaks, sanity checks, spotting reliance on sensitive features), and use an intrinsically simple model when the stakes demand a faithful, auditable rule.

## Worked example

**t-SNE inputs.** 20 points in $\mathbb R^5$ forming two tight clusters of 10 (centres 0 and 3, spread $0.3$). Perplexity 5: every conditional row has $\exp H=5.000$. The symmetrised $P$ sums to 1, each cluster holds mass exactly $0.5$ and cross-cluster mass is below $10^{-6}$. Starting from $Y\sim\mathcal N(0,10^{-4})$ (so $Q$ is almost uniform), $\mathrm{KL}(P\|Q)=1.2303$; the analytic gradient agrees with central finite differences to about $2\times10^{-10}$; one step with $\eta=10$ lowers the KL to $1.2300$ (a tiny step because the gradient is small at this near-collapsed start; real runs use many hundreds of iterations with early exaggeration).

**Kernel PCA.** Two concentric circles (radius 1 and 3, 40 points each), RBF with $\gamma=0.5$: top centred-kernel eigenvalues $10.70$ and $8.64$, and the first kernel component is perfectly correlated ($|r|=1$) with the circle label, whereas the first linear PCA score is uncorrelated with it.

**Interpretation.** Data: $n=400$, four standard normal features, $y=2x_0+x_1^2+0.3\varepsilon$. A least-squares model with features $[1,x_0,x_1,x_1^2,x_2,x_3]$ finds coefficients near $1.97$ for $x_0$ and $1.00$ for $x_1^2$. The baseline MSE (computed on the training data here, for simplicity) is $0.0967$. Permutation importance (20 repeats): $x_0$: $7.63$, $x_1$: $3.12$, $x_2$: $\approx0.00001$, $x_3$: $\approx0.001$. Partial dependence of $x_1$ at $(-2,-1,0,1,2)$: $(4.05,1.03,-0.01,0.96,3.92)$, a parabola; of $x_0$: a straight line of slope $1.974$. Shapley values for $x=(1,2,0,0)$: $(1.949,3.009,0.0005,0.0017)$ whose sum $4.960$ equals $f(x)-\mathbb E f$ exactly. All reproduced by the exercises.

## Implementation

```python
import numpy as np

def tsne_grad(P, Y):
    D = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    W = 1.0 / (1.0 + D); np.fill_diagonal(W, 0)
    Q = W / W.sum()
    M = (P - Q) * W
    return 4 * (np.diag(M.sum(1)) - M) @ Y            # (N, 2)

def permutation_importance(predict, X, y, n_repeats, rng):
    base = np.mean((y - predict(X)) ** 2); imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        v = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            v.append(np.mean((y - predict(Xp)) ** 2) - base)
        imp[j] = np.mean(v)
    return base, imp
```

Production: `sklearn.manifold.TSNE`, `sklearn.decomposition.KernelPCA`, `sklearn.inspection.permutation_importance` and `partial_dependence`, the `shap` package, `umap-learn`.

## Shapes

t-SNE: `X: (N, D)`, `P, Q, W: (N, N)`, `Y, grad: (N, 2)`. Kernel PCA: `K: (N, N)`, scores `(N, k)`. Interpretation: `predict: (N, d) -> (N,)`; importances `(d,)`; PD curve `(len(grid),)`; Shapley values `(d,)` per instance.

## Common misconceptions

- **"Distances between t-SNE clusters mean something."** They do not; neither do cluster sizes.
- **"One t-SNE run is the truth."** Rerun with other seeds and perplexities.
- **"Kernel PCA reduces dimension for free."** It needs an $N\times N$ matrix and gives no easy inverse map.
- **"High permutation importance means causal."** It means this model relies on the feature.
- **"Partial dependence is what happens to one prediction."** It is an average; use ICE or Shapley values for individual rows.
- **"Shapley values explain the world."** They fairly attribute the model's output, given a chosen background distribution.

## Key takeaways

- t-SNE matches neighbourhood distributions $P$ (Gaussian, perplexity-calibrated) and $Q$ (Student-t) by minimising KL; gradient $4\sum_j(p_{ij}-q_{ij})w_{ij}(y_i-y_j)$.
- Kernel PCA is PCA on the centred Gram matrix $JKJ$.
- Permutation importance, partial dependence and Shapley values probe any model; each has failure modes with correlated features.
- Shapley values are efficient: attributions sum exactly to $f(x)-\mathbb E f$.

## Explain it yourself

1. Why does t-SNE use a heavy-tailed kernel in the map, and what would go wrong with a Gaussian there?
2. Two features are near-copies of each other and the model uses both. Predict what permutation importance reports and why.

## Further reading

- Bishop, *PRML*, Sec. 12.3 (kernel PCA, including the Gaussian-kernel figure on PDF p.610).
- van der Maaten and Hinton, "Visualizing data using t-SNE" (2008); Wattenberg et al., "How to use t-SNE effectively" (distill.pub, interactive; listed in the source lesson).
- Lundberg and Lee (2017) for SHAP. None of these three were opened for this lesson.

## Mastery criteria

You can compute t-SNE affinities with a perplexity search, derive and check its gradient, perform kernel PCA with a centred Gram matrix, implement permutation importance, partial dependence and exact Shapley values, and state precisely what each explanation does and does not show.
