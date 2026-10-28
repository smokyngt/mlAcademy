---
id: math.09
title: Random variables, expectation, variance and the Gaussian
summary: Describe uncertain quantities with distributions, summarise them with mean, variance and covariance, and see why the Gaussian and maximum likelihood turn into squared error.
minutes: 80
teaches: [prob.rv]
requires: [prob.basics, calc.derivative]
sources:
  - "math/math for ML.pdf, Sec. 6.2 Discrete and Continuous Probabilities (PDF p.184-189): pmf, pdf, cdf, density can exceed 1, nomenclature table. READ."
  - "math/math for ML.pdf, Sec. 6.3 Sum Rule, Product Rule, Bayes (PDF p.189-192): marginalisation and the product rule, recalled from the previous lesson. READ."
  - "math/math for ML.pdf, Sec. 6.4 Summary Statistics and Independence (PDF p.192-202): expected value and its linearity, covariance, correlation, empirical mean/covariance (1/N is biased), three expressions for the variance, affine transformation, uncorrelated vs independent. READ."
  - "math/math for ML.pdf, Sec. 6.5 Gaussian Distribution (PDF p.203-210): univariate and multivariate density, conditionals/marginals, sums, A mu and A Sigma A^T, sampling with the Cholesky factor. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 2.3 (PDF p.98-102): central limit theorem, Mahalanobis distance, eigen-decomposition of the covariance and the elliptical contours. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 2.3.4 (PDF p.113-114): maximum likelihood for the Gaussian, mu_ML and Sigma_ML, E[Sigma_ML] = (N-1)/N Sigma. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/06-probability-and-distributions/docs/en.md: pmf vs pdf, CLT story, log-probabilities. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/15-statistics-for-ml/docs/en.md (first part): descriptive statistics, Bessel correction, Pearson correlation. READ."
---

## Where this fits

A classifier does not output "cat". It outputs numbers like `[0.03, 0.91, 0.06]`. A regression model does not know the exact house price: the measurements are noisy, the buyers are moody. Every time a model has to talk about something it cannot be sure of, it needs a way to say *how* unsure it is. That language is a **random variable with a distribution**.

This lesson is the vocabulary that everything after it uses. The loss you minimise is an *expectation* (an average over the data distribution). The spread of your gradients is a *variance*. Features that move together have a *covariance*, and the eigen-decomposition of that covariance matrix (previous lesson) is PCA. Initialisation, noise in diffusion models, the Kalman filter, Gaussian processes: all Gaussians. And at the end we derive something you will meet again and again: **fitting a Gaussian by maximum likelihood is the same as minimising the mean squared error**. The next lesson (information theory) uses exactly the same "log of a probability" trick for classification.

## The idea in plain words

- A **random variable** is a number that is not decided yet. "The value of the next die roll", "tomorrow's temperature", "the height of a random person". Before you look, you cannot say what it is; after you look, it is just a number. The *distribution* is the recipe that says how likely each possible value is.
- If the values come one by one (die faces, class labels) the recipe is a **table of probabilities** that adds up to 1. If the value slides along a continuum (a height), no single exact value has a visible probability, so the recipe is a **curve whose area over an interval is the probability of landing in that interval**. The height of the curve is a *density*, not a probability: a very narrow bell can be taller than 1, as long as the total area stays 1.
- **Expectation** is the balance point: put the distribution on a see-saw, and the mean is where it balances. It is also what you would get on average if you repeated the experiment forever. **Variance** says how far values typically stray from that balance point (squared, so that being above and below do not cancel).
- With two variables we ask whether they **move together**. Tall people tend to be heavier: when one is above its average, the other usually is too. That tendency is the **covariance**. Divide it by both spreads and you get the **correlation**, a unit-free number between -1 and 1.
- The **Gaussian** (bell curve) is what you get when many small independent nudges add up (the central limit theorem). Two numbers describe it fully: where the middle is and how wide it is. In several dimensions the bell becomes a **hill with elliptical contour lines**: the mean says where the peak is, the covariance matrix says how the ellipse is stretched and tilted. Because it is described by just a mean and a covariance, you can push it through a linear map by pushing only those two things through.

## Definitions

### Discrete and continuous

A **probability mass function** (pmf) gives $P(X=x)$ for each value, with $\sum_x P(X=x)=1$. A **probability density function** (pdf) is a function $p(x)\ge 0$ with $\int p(x)\,dx = 1$, and

$$
P(a\le X\le b)=\int_a^b p(x)\,dx .
$$

The probability of one exact value is $0$ for a continuous variable, and $p(x)$ may exceed $1$ (the book's example: a uniform density on $[0.9,1.6]$ has height $1/0.7\approx1.43$). The **cumulative distribution function** (cdf) is $F(x)=P(X\le x)=\int_{-\infty}^x p(t)\,dt$.

### Expectation

For a function $g$ of a random variable $X$ (Deisenroth Def. 6.3):

$$
\mathbb E[g(X)]=\sum_x g(x)P(X=x)\quad\text{or}\quad \int g(x)\,p(x)\,dx .
$$

The mean is $g(x)=x$. It is **linear**: $\mathbb E[aX+bY+c]=a\,\mathbb E[X]+b\,\mathbb E[Y]+c$, whether or not $X$ and $Y$ are independent (the book's remark: the integral of a sum is the sum of integrals).

### Variance, in three expressions

$$
\operatorname{Var}[X]=\mathbb E\big[(X-\mu)^2\big]=\mathbb E[X^2]-\mu^2,\qquad \mu=\mathbb E[X].
$$

The middle expression is the definition. The right-hand one ("mean of the square minus square of the mean", the **raw-score formula**) follows by expanding the square and using linearity:

$$
\mathbb E[(X-\mu)^2]=\mathbb E[X^2-2\mu X+\mu^2]=\mathbb E[X^2]-2\mu\,\mathbb E[X]+\mu^2=\mathbb E[X^2]-\mu^2 .
$$

(Each step is one rule: expand, split the expectation by linearity, replace $\mathbb E[X]$ by $\mu$, and $-2\mu^2+\mu^2=-\mu^2$.) The third expression the book gives is a sum over all pairs of observations, $\frac1{N^2}\sum_{i,j}(x_i-x_j)^2 = 2\big(\frac1N\sum x_i^2-(\frac1N\sum x_i)^2\big)$: twice the raw-score variance.

**Numerical trap.** The raw-score formula needs only one pass over the data, but it subtracts two huge, nearly equal numbers. That destroys floating-point precision (Deisenroth warns about this). In the worked example below you will see it return a *negative* variance.

### Covariance and correlation

$$
\operatorname{Cov}[X,Y]=\mathbb E\big[(X-\mathbb E X)(Y-\mathbb E Y)\big]=\mathbb E[XY]-\mathbb E[X]\,\mathbb E[Y],\qquad \operatorname{corr}[X,Y]=\frac{\operatorname{Cov}[X,Y]}{\sqrt{\operatorname{Var}[X]\operatorname{Var}[Y]}}\in[-1,1].
$$

For a random vector $\mathbf x\in\mathbb R^D$ the **covariance matrix** is $\Sigma=\mathbb E[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top]\in\mathbb R^{D\times D}$: variances on the diagonal, covariances off it. It is symmetric and positive semi-definite. Correlation is the cosine of the angle between the two centred variables, which is why it lies in $[-1,1]$ (the book's Sec. 6.4.6).

**Independence implies zero covariance, not the reverse.** Take $X$ symmetric around $0$ and $Y=X^2$. Then $Y$ is completely determined by $X$, yet $\operatorname{Cov}[X,Y]=\mathbb E[X^3]-\mathbb E[X]\mathbb E[X^2]=0$. Covariance only sees *linear* co-movement.

### Empirical mean and covariance: $N$ versus $N-1$

From a dataset $\mathbf x_1,\dots,\mathbf x_N$ we estimate

$$
\bar{\mathbf x}=\frac1N\sum_{n}\mathbf x_n,\qquad \hat\Sigma=\frac1N\sum_n(\mathbf x_n-\bar{\mathbf x})(\mathbf x_n-\bar{\mathbf x})^\top .
$$

The $1/N$ version is the maximum-likelihood estimate and is **biased**: $\mathbb E[\hat\Sigma]=\frac{N-1}{N}\Sigma$ (Bishop 2.124). The reason is that $\bar{\mathbf x}$ was itself fitted to these same points, so the points sit slightly closer to $\bar{\mathbf x}$ than to the true mean. Dividing by $N-1$ instead gives the unbiased estimate. With thousands of points the difference is invisible; with a handful it matters. NumPy's `np.cov` uses $N-1$ by default, `np.var` uses $N$.

### Affine transformations

Let $\mathbf x$ have mean $\boldsymbol\mu$ and covariance $\Sigma$, and let $\mathbf y=A\mathbf x+\mathbf b$ with $A\in\mathbb R^{m\times D}$, $\mathbf b\in\mathbb R^m$. Then

$$
\mathbb E[\mathbf y]=A\boldsymbol\mu+\mathbf b,\qquad \operatorname{Cov}[\mathbf y]=A\,\Sigma\,A^\top\in\mathbb R^{m\times m}.
$$

Derivation. The mean is linearity. For the covariance, $\mathbf y-\mathbb E\mathbf y=A(\mathbf x-\boldsymbol\mu)$ (the constant $\mathbf b$ cancels: shifting never changes spread), so

$$
\operatorname{Cov}[\mathbf y]=\mathbb E\big[A(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top A^\top\big]=A\,\mathbb E\big[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top\big]A^\top=A\Sigma A^\top ,
$$

since $A$ is a constant and can leave the expectation on both sides. In one dimension this is the familiar $\operatorname{Var}[aX+b]=a^2\operatorname{Var}[X]$.

### The Gaussian

Univariate, with mean $\mu$ and variance $\sigma^2$:

$$
\mathcal N(x\mid\mu,\sigma^2)=\frac1{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).
$$

Multivariate, with $\mathbf x,\boldsymbol\mu\in\mathbb R^D$ and $\Sigma\in\mathbb R^{D\times D}$ symmetric positive definite:

$$
\mathcal N(\mathbf x\mid\boldsymbol\mu,\Sigma)=\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\!\Big(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Big).
$$

The exponent contains the squared **Mahalanobis distance** $\Delta^2=(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$, which reduces to ordinary squared distance when $\Sigma=I$. Writing $\Sigma=\sum_i\lambda_i\mathbf u_i\mathbf u_i^\top$ (eigen-decomposition, previous lesson), Bishop shows $\Delta^2=\sum_i y_i^2/\lambda_i$ with $y_i=\mathbf u_i^\top(\mathbf x-\boldsymbol\mu)$: in the rotated coordinates the Gaussian factorises into $D$ independent 1-D Gaussians. Contours of constant density are ellipses whose axes point along the eigenvectors $\mathbf u_i$ and have half-lengths proportional to $\sqrt{\lambda_i}$.

Three facts used constantly (Deisenroth Sec. 6.5):

1. **Affine maps stay Gaussian:** $\mathbf x\sim\mathcal N(\boldsymbol\mu,\Sigma)\Rightarrow A\mathbf x+\mathbf b\sim\mathcal N(A\boldsymbol\mu+\mathbf b,\ A\Sigma A^\top)$.
2. **Sums of independent Gaussians are Gaussian:** means add, covariances add.
3. **Marginals and conditionals of a joint Gaussian are Gaussian.** In the book's bivariate example (mean $(0,2)$, covariance rows $(0.3,-1)$ and $(-1,5)$) the marginal of $x_1$ is $\mathcal N(0,0.3)$ and the conditional given $x_2=-1$ is $\mathcal N(0.6,0.1)$.

### Sampling with the Cholesky factor

Computers give you samples of $\mathbf z\sim\mathcal N(\mathbf 0,I)$ (independent standard normals). To get $\mathbf y\sim\mathcal N(\boldsymbol\mu,\Sigma)$, use fact 1 with $A=L$ where $LL^\top=\Sigma$:

$$
\mathbf y=\boldsymbol\mu+L\mathbf z,\qquad \operatorname{Cov}[\mathbf y]=L\,I\,L^\top=\Sigma .
$$

The Cholesky factorisation gives a lower-triangular $L$ (cheap), and exists because $\Sigma$ is symmetric positive definite. This is how every multivariate Gaussian is sampled in practice, and it is the *reparameterisation trick* of variational autoencoders.

## Maximum likelihood for the Gaussian

Suppose $x_1,\dots,x_N$ are drawn independently from $\mathcal N(\mu,\sigma^2)$ and we want the best parameters. Independence makes the likelihood a product; taking the log turns it into a sum (and avoids multiplying hundreds of tiny numbers, which underflows):

$$
\ln p(\mathbf x\mid\mu,\sigma^2)=-\frac N2\ln(2\pi\sigma^2)-\frac1{2\sigma^2}\sum_{n=1}^N(x_n-\mu)^2 .
$$

Differentiate with respect to $\mu$ and set to zero:

$$
\frac{\partial}{\partial\mu}=\frac1{\sigma^2}\sum_n(x_n-\mu)=0\ \Longrightarrow\ \mu_{\rm ML}=\frac1N\sum_n x_n .
$$

Differentiate with respect to $\sigma^2$ (write $v=\sigma^2$): $-\frac N{2v}+\frac1{2v^2}\sum_n(x_n-\mu)^2=0$, which gives $v_{\rm ML}=\frac1N\sum_n(x_n-\mu_{\rm ML})^2$. The multivariate result is the same with outer products (Bishop 2.121, 2.122). So the maximum-likelihood fit of a Gaussian is: **sample mean, and the biased ($1/N$) sample covariance.**

**Link to MSE.** For fixed $\sigma^2$, the only term that depends on $\mu$ is $\frac1{2\sigma^2}\sum_n(x_n-\mu)^2$. Maximising the likelihood therefore means minimising the sum of squared errors. In a regression model $t_n=f(\mathbf x_n;\mathbf w)+\varepsilon$ with Gaussian noise $\varepsilon\sim\mathcal N(0,\sigma^2)$, the same computation says: **minimising the MSE loss is maximum likelihood under Gaussian noise.** The squared error is not an arbitrary choice; it is the negative log of a bell curve.

## Worked example

All numbers below are reproduced by the exercise code.

**1. Discrete expectation.** $X\in\{0,1,2,3\}$ with $P=(0.1,0.2,0.4,0.3)$. Mean: $0\cdot0.1+1\cdot0.2+2\cdot0.4+3\cdot0.3=1.9$. $\mathbb E[X^2]=0.2+1.6+2.7=4.5$. Variance $=4.5-1.9^2=0.89$; the definition $\sum_x(x-1.9)^2P(x)$ also gives $0.89$.

**2. The raw-score trap.** Data $10^9+(4,7,13,16)$. The true variance is $22.5$ (shift-invariant). In float64, `mean(x**2) - mean(x)**2` returns **-128.0**: each term is about $10^{18}$, where neighbouring floats are $128$ apart, so the true $22.5$ is rounded away. Subtracting the mean first (two-pass) gives $22.5$ exactly.

**3. Empirical covariance.** Five points $(2,1),(4,3),(6,2),(8,6),(10,8)$ (array $X$ of shape $(5,2)$). Mean $(6,4)$. With the centred matrix $X_c=X-\bar x$, the product $X_c^\top X_c/5$ has rows $(8,\,6.8)$ and $(6.8,\,6.8)$, while dividing by $N-1=4$ gives rows $(10,\,8.5)$ and $(8.5,\,8.5)$ (ratio $5/4$). Correlation $=6.8/\sqrt{8\cdot6.8}=0.922$.

**4. Affine map.** $\boldsymbol\mu=(1,2)$, $\Sigma$ with rows $(2,0.6)$ and $(0.6,1)$, $A$ with rows $(1,1)$ and $(0,2)$, $\mathbf b=(0,1)$. Then $A\boldsymbol\mu+\mathbf b=(3,5)$ and $A\Sigma A^\top$ has rows $(4.2,\,3.2)$ and $(3.2,\,4)$. Check by hand: $A\Sigma$ has rows $(2.6,\,1.6)$ and $(1.2,\,2)$; multiply by $A^\top$, whose rows are $(1,0)$ and $(1,2)$.

**5. Cholesky.** For the same $\Sigma$: $L$ has rows $(1.4142,\,0)$ and $(0.4243,\,0.9055)$, and $LL^\top=\Sigma$. Samples $\boldsymbol\mu+L\mathbf z$ from $200\,000$ standard normal draws have sample covariance close to $\Sigma$.

**6. Maximum likelihood.** Data $(2,4,4,5,5)$: $\mu_{\rm ML}=4$, deviations $(-2,0,0,1,1)$, $\sigma^2_{\rm ML}=6/5=1.2$, unbiased $6/4=1.5$. At the optimum, the log-likelihood equals $-\frac N2\ln(2\pi\sigma^2_{\rm ML})-\frac N2=-7.55$.

## Implementation

```python
import numpy as np

x = 1e9 + np.array([4., 7, 13, 16])
raw = np.mean(x**2) - np.mean(x)**2          # -128.0  (catastrophic cancellation)
two_pass = np.mean((x - x.mean())**2)        # 22.5

X = np.array([[2., 1], [4, 3], [6, 2], [8, 6], [10, 8]])   # (N, D) = (5, 2)
Xc = X - X.mean(axis=0)
cov_biased = Xc.T @ Xc / len(X)               # (D, D)
cov_unbiased = np.cov(X, rowvar=False)        # divides by N-1

Sigma = np.array([[2., .6], [.6, 1.]])
L = np.linalg.cholesky(Sigma)
z = np.random.default_rng(0).standard_normal((1000, 2))   # (n, D)
y = np.array([1., 2.]) + z @ L.T                           # rows are samples
```

## Shapes

Data matrix `X: (N, D)` with samples in rows. Mean `(D,)`. Covariance `(D, D)`. If rows are samples, an affine map is applied as `X @ A.T + b` (the formula $A\mathbf x$ is for one column vector), and the covariance of the result is `A @ Sigma @ A.T` with shape `(m, m)`. Sampling: `Z: (n, D)` standard normals, `Z @ L.T: (n, D)`.

## Common misconceptions

- **A density value is a probability.** No: it can exceed 1; only areas are probabilities.
- **Zero covariance means independent.** Only for jointly Gaussian variables. $Y=X^2$ is the counterexample.
- **Always divide by $N-1$.** It is the unbiased estimator of the variance; the $1/N$ version is the maximum-likelihood one. Both are consistent; know which your library uses (`np.var` uses $N$, `np.cov` uses $N-1$).
- **The two-line raw-score variance is a safe shortcut.** It cancels catastrophically when the mean is large compared to the spread. Centre first.
- **Covariance matrices can be any matrix.** They are symmetric positive semi-definite; that is what makes the Cholesky factor and the Gaussian density exist.

## Key takeaways

- A random variable is described by a pmf (tables, sum to 1) or a pdf (curves, integrate to 1, may exceed 1).
- $\mathbb E$ is linear; $\operatorname{Var}[X]=\mathbb E[X^2]-\mathbb E[X]^2$ but compute it by centring first.
- $\mathbf y=A\mathbf x+\mathbf b$ gives mean $A\boldsymbol\mu+\mathbf b$ and covariance $A\Sigma A^\top$; for Gaussians the result is again Gaussian.
- Sample from $\mathcal N(\boldsymbol\mu,\Sigma)$ with $\boldsymbol\mu+L\mathbf z$, $LL^\top=\Sigma$.
- Maximum likelihood for a Gaussian: sample mean and $1/N$ covariance; with fixed variance it is exactly least squares.

## Explain it yourself

1. Why does the covariance of $A\mathbf x+\mathbf b$ not depend on $\mathbf b$, and why does $A$ appear on both sides?
2. In your own words: why is minimising MSE the same as assuming Gaussian noise?

## Further reading

- *Mathematics for Machine Learning*, Sec. 6.4 (PDF p.192) and 6.5 (PDF p.203).
- Bishop, *PRML*, Sec. 2.3 (PDF p.98), in particular the conditional and marginal Gaussians (PDF p.105-110, not part of this lesson's sources).

## Mastery criteria

You can compute the mean, variance and covariance of a small discrete or empirical dataset by hand, explain the $N$ versus $N-1$ choice, propagate a Gaussian through an affine map, sample from a multivariate Gaussian using a Cholesky factor, and derive $\mu_{\rm ML}$ and $\sigma^2_{\rm ML}$ and the link to squared error.
