---
id: cml.16
title: "Feature selection and the lasso: regularisation paths and stability"
summary: Sort selection methods into filter, wrapper and embedded families, derive soft-thresholding for the L1 penalty, implement the lasso by coordinate descent and trace its regularisation path, and judge which selected features you can trust with stability selection.
minutes: 80
teaches: [ml.featsel, ml.lasso]
requires: [ml.ridge, ml.linreg, ml.features, info.entropy, opt.convex]
sources:
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/docs/en.md (filter, wrapper, embedded families; variance threshold, mutual information, RFE, L1, tree importance, permutation importance, comparison). READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 3.1.4 Regularized least squares (PDF p.163-166): quadratic regulariser and weight decay, generalised regulariser with exponent q, lasso as q = 1, equivalent constrained form, sparsity from the constraint geometry. READ."
---

## Where this fits

Real datasets come with hundreds of columns, most of them redundant or pure noise. More columns do not help a model: they add variance (overfitting, [[cml.03]]), slow training, and make the result impossible to explain. **Feature selection** chooses a useful subset. This lesson ties together three things you already met: the L2 penalty of ridge regression ([[cml.01]]), mutual information as an entropy quantity (entropy and KL divergence), and the leakage warning of [[cml.15]] (selection is a fitted step and must live inside the cross-validation loop).

The star of the lesson is the **lasso**, linear regression with an L1 penalty. It does selection and fitting in one optimisation, and its solution has a closed-form coordinate update (soft-thresholding) that you will derive and implement.

## The idea in plain words

Three ways to hire a team from 200 applicants.

- **Filter**: score each applicant alone with a quick test (correlation with job performance, or a more general "how much does this tell me" score) and keep the top few. Fast, but blind to teamwork: two applicants may be great alone and redundant together.
- **Wrapper**: try teams. Add the member who improves the team's result most, repeat (forward selection), or start with everyone and drop the least useful (backward elimination). Accurate for the chosen model, expensive: many model fits.
- **Embedded**: the training procedure itself drops people. The lasso charges a fee for each unit of weight a feature carries, so features that do not pay for themselves end up with exactly zero weight.

Why *exactly* zero? Imagine a budget on the total absolute size of the weights, drawn as a diamond in weight space with its corners on the axes. The best fit that touches the diamond tends to touch a corner, and a corner means some weights are zero. The ridge budget is a round disc: the fit touches it at a generic point where nothing is exactly zero.

## Filter, wrapper, embedded

**Filters.** Variance threshold (drop constant columns), correlation with the target, and **mutual information** (MI). For discrete variables

$$
I(X;Y)=\sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)\,p(y)}\ \ge 0,
$$

the KL divergence between the joint distribution and the product of marginals (entropy and KL divergence); it is zero exactly when $X$ and $Y$ are independent. Unlike correlation it detects non-linear dependence: for $y=x^2+\text{noise}$ with $x$ symmetric, the correlation is about zero, but MI is large. For continuous variables you estimate it with histograms (binning at quantiles), kernel or nearest-neighbour estimators; estimates depend on the binning and on sample size, so compare scores of features, not their absolute values.

**Wrappers.** Forward selection, backward elimination, recursive feature elimination (fit, drop the smallest-weight features, refit). They optimise the metric you care about but cost many fits, and with many candidate subsets they overfit the validation data.

**Embedded.** L1 regularisation, and importance measures built into trees. Tree split-based importance is biased towards high-cardinality features (a random id column "splits well"); permutation importance ([[cml.17]]) is a better check.

## The lasso

Let $X\in\mathbb R^{n\times d}$ have standardised columns (mean 0, variance 1), $y\in\mathbb R^n$ be centred. The lasso solves

$$
\min_{w\in\mathbb R^d}\ \frac1{2n}\|y-Xw\|_2^2+\lambda\|w\|_1,\qquad \|w\|_1=\sum_j|w_j| .
$$

Bishop (Sec. 3.1.4) writes the family $\frac12\sum_n\{t_n-w^\top\phi(x_n)\}^2+\frac\lambda2\sum_j|w_j|^q$; $q=2$ is weight decay (ridge) and $q=1$ is the lasso. Minimising with the penalty is equivalent, by Lagrange multipliers (see the convexity and Lagrange-multiplier material), to minimising the squared error subject to $\sum_j|w_j|\le\eta$ for some $\eta$ depending on $\lambda$. The level sets of the squared error are ellipses; the diamond $\sum|w_j|\le\eta$ has corners on the axes, which is where sparsity comes from. Standardise first: the penalty treats all weights alike, so features on different scales would be penalised unequally.

### Derivation: soft-thresholding

Fix all weights except $w_j$ and let $r^{(j)}=y-\sum_{k\ne j}X_{:k}w_k$ be the partial residual. Define $z_j=\frac1n X_{:j}^\top X_{:j}$ (equal to 1 for standardised columns) and $\rho_j=\frac1n X_{:j}^\top r^{(j)}$. Expanding the square, the objective as a function of $w=w_j$ is, up to a constant,

$$
g(w)=\tfrac12 z_j w^2-\rho_j w+\lambda|w| .
$$

It is convex but not differentiable at $0$. Treat the cases.

- If $w>0$: $g'(w)=z_jw-\rho_j+\lambda=0$ gives $w=(\rho_j-\lambda)/z_j$, valid only if $\rho_j>\lambda$.
- If $w<0$: $g'(w)=z_jw-\rho_j-\lambda=0$ gives $w=(\rho_j+\lambda)/z_j$, valid only if $\rho_j<-\lambda$.
- Otherwise the minimum sits at $w=0$: the subgradient condition $0\in -\rho_j+\lambda[-1,1]$ holds exactly when $|\rho_j|\le\lambda$.

Together:

$$
\boxed{\,w_j=\frac{S(\rho_j,\lambda)}{z_j},\qquad S(\rho,\lambda)=\operatorname{sign}(\rho)\max(|\rho|-\lambda,0)\,}
$$

the **soft-threshold** operator: shrink towards zero by $\lambda$, and clip to zero if the correlation is below the threshold. The ridge analogue, $w_j=\rho_j/(z_j+\lambda)$, only rescales and never gives exact zeros.

**Coordinate descent** cycles through $j=1,\dots,d$ applying this update, maintaining the residual $r=y-Xw$ so each update costs $O(n)$. The objective is convex and its non-smooth part separates across coordinates, so the sweeps converge to the global minimum.

**Largest useful penalty.** At $w=0$ the optimality condition for every coordinate is $|\rho_j|=|\frac1nX_{:j}^\top y|\le\lambda$, hence

$$
\lambda_{\max}=\max_j\frac1n|X_{:j}^\top y| ,
$$

and for $\lambda\ge\lambda_{\max}$ the model is empty. Decreasing $\lambda$ from $\lambda_{\max}$ lets features enter one at a time: that is the **regularisation path**. A common recipe is to compute the path on a log-spaced grid, warm-starting each solution from the previous one, and choose $\lambda$ by cross-validation (with the whole procedure in a Pipeline, [[cml.15]]).

**Optimality (KKT) conditions** you can test numerically: with $g=\frac1nX^\top(y-Xw)$, active coordinates satisfy $g_j=\lambda\,\mathrm{sign}(w_j)$ and inactive ones satisfy $|g_j|\le\lambda$.

### Stability selection

The set of non-zero lasso coefficients is fragile: with correlated features it often picks one of a group arbitrarily, and small changes in the data change the winner. **Stability selection** reruns the lasso on many random half-samples and records how often each feature is selected. Features with selection frequency near 1 are trustworthy; noise features appear sporadically. Keep those above a threshold (0.6-0.9 is common). It turns a yes/no selection into an evidence score.

## Worked example

Synthetic data (exercise seed 0): $n=100$, $d=10$ standardised Gaussian columns, true weights $(3,-2,0,0,1.5,0,0,0,0,0)$, noise standard deviation $0.5$.

- $\lambda_{\max}=2.9795$: the empty model for any larger $\lambda$.
- Non-zero counts along $\lambda=\lambda_{\max}\times(1,0.5,0.2,0.05)$ are $0,2,3,3$.
- At $0.2\lambda_{\max}$ the active set is exactly the truth $\{0,1,4\}$ with **shrunken** weights $(2.464,-1.309,0.777)$ against ordinary least squares near $(3.1,-2.1,1.5)$: selection costs bias. At $\lambda=0.001\lambda_{\max}$ nine of the ten coefficients are non-zero (the noise ones tiny) and the solution is close to least squares.
- KKT check at $0.2\lambda_{\max}$: the active gradients equal $\pm0.2\lambda_{\max}=\pm0.596$ and all others are below it.
- Stability selection (50 half-samples, $\lambda=0.03\lambda_{\max}$): features 0, 1, 4 are selected in $100\%$ of the subsamples, noise features in $20$-$36\%$. At $\lambda=0.01\lambda_{\max}$ the noise features are selected in $58$-$80\%$: too small a penalty selects junk.
- Mutual information: for $y=x^2+0.1\varepsilon$ with $x\sim\mathcal N(0,1)$, 2000 samples, the correlation is $0.008$ but 8-quantile-bin MI is $1.02$ nats; an independent noise column gets $0.012$.

## Implementation

```python
import numpy as np

def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def lasso_cd(X, y, lam, n_iter=500, tol=1e-10):
    n, d = X.shape
    w = np.zeros(d); r = y.astype(float).copy(); z = (X ** 2).sum(0) / n
    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            r += X[:, j] * w[j]                       # remove coordinate j
            w[j] = soft_threshold(X[:, j] @ r / n, lam) / z[j]
            r -= X[:, j] * w[j]                       # put it back
        if np.max(np.abs(w - w_old)) < tol: break
    return w
```

Libraries: `sklearn.linear_model.Lasso`/`LassoCV`/`lasso_path` (note: scikit-learn's objective uses the same $1/(2n)$ scaling), `ElasticNet` (mix of L1 and L2, steadier with correlated groups), `sklearn.feature_selection.mutual_info_regression`, `RFE`, `SequentialFeatureSelector`.

## Shapes

`X: (n, d)`, `y: (n,)`, `w: (d,)`, residual `r: (n,)`, `z: (d,)`; a path is `(n_lambdas, d)`; stability frequencies `(d,)`.

## Common misconceptions

- **"Selected means causal or important."** Selection reflects predictive usefulness under this model and sample; correlated features substitute for each other.
- **"Lasso coefficients are unbiased."** They are shrunken; refitting ordinary least squares on the selected set (relaxed lasso) removes part of the bias.
- **"Select first, then cross-validate."** Selection uses the labels; do it inside each fold, or you will see skill in pure noise ([[cml.15]]).
- **"Correlation filtering is enough."** It misses non-linear and interaction effects (MI catches the first, wrappers or trees the second).
- **"Unscaled features are fine."** The L1 penalty depends on feature scale; standardise.
- **"MI scores are comparable across estimators."** Binning and sample size change absolute values; use them to rank.

## Key takeaways

- Filters are fast and model-free, wrappers accurate and costly, embedded methods a compromise.
- $w_j=S(\rho_j,\lambda)/z_j$: the L1 penalty shrinks and zeroes coefficients; $\lambda_{\max}=\max_j|X_{:j}^\top y|/n$ starts the path.
- Sparsity comes from the corners of the L1 ball; ridge shrinks but does not select.
- Check stability across subsamples before believing a selected set.

## Explain it yourself

1. Derive the soft-threshold update from the one-dimensional objective $\frac12z w^2-\rho w+\lambda|w|$, and explain why $\lambda_{\max}$ follows from it.
2. Why can two perfectly good, highly correlated features make lasso selection unstable, and what does stability selection report?

## Further reading

- Bishop, *PRML*, Sec. 3.1.4 and Fig. 3.3-3.4 (contours of the regulariser; sparsity picture).
- The source lesson `phases/02-ml-fundamentals/18-feature-selection` for RFE, tree importance and the comparison table.

## Mastery criteria

You can classify a selection method as filter, wrapper or embedded, derive soft-thresholding and $\lambda_{\max}$, implement coordinate descent and verify it with the KKT conditions, read a regularisation path and justify a selected set with stability evidence.
