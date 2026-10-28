---
id: mx.05
title: Integration for machine learning
summary: Area as a limit of sums, the fundamental theorem of calculus, substitution and parts, why expectations are integrals, the Gaussian integral by polar coordinates, and three numerical methods compared.
minutes: 80
teaches: [mx.integral, mx.expect-integral, mx.numint]
requires: [calc.derivative, alg.summation, prob.rv, calc.numerical-diff]
sources:
  - "math/math for ML.pdf, Sec. 6.2 Discrete and Continuous Probabilities (PDF p.184-190): pdf definition (non-negative, integrates to 1), P(a<=X<=b) as an integral, cdf as an integral, densities may exceed 1 (uniform on [0.9,1.6]). READ."
  - "math/math for ML.pdf, Sec. 6.5 Gaussian Distribution (PDF p.203-205 and p.209): univariate and multivariate Gaussian density, mixture second moment E[x^2] as integrals and the variance of a mixture. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Appendix B Probability Distributions (PDF p.705-706, p.710): Bernoulli and Beta summaries (E[mu]=a/(a+b)), Gaussian marginal formula (B.51). Read selectively. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md (MCMC section, for the link from integrals to sampling). READ."
---

## Where this fits

The word **expectation** hides an integral. The cross-entropy loss, the KL divergence, the evidence in Bayesian inference, the ELBO of a variational autoencoder, the normalising constant of a distribution, the average reward of a policy: all are integrals of some function against a probability density. Most of the time they cannot be solved with pencil and paper, so ML replaces them by sums over data (the training loss is a Monte Carlo estimate of an expected loss) or by sampling ([[mx.04]] showed how MCMC does that).

This lesson gives you the tools to read those formulas and to evaluate them numerically. It builds on derivatives ([[f2n.04]]) and the idea of a density from [[math.09]]. The Gaussian integral computed here justifies the constant in front of the Gaussian density, which you have been using since [[math.09]].

## The idea in plain words

- To find the area under a curve, slice it into thin vertical strips, treat each strip as a rectangle, and add the rectangles. Use thinner strips and the sum approaches a single number, the **integral**.
- The **fundamental theorem of calculus** says something surprising: if you know a function whose slope is your curve (an *antiderivative*), then the area between two points is just the difference of that function at the two ends. Integrating undoes differentiating.
- A **probability density** is a curve whose total area is 1. The probability of landing in an interval is the area over it. A single point has no area, so it has probability zero. The *height* of the curve is not a probability, so it may exceed 1 (squash a distribution into a narrow range and it must grow tall).
- The **average value** of some quantity when the input is random is the area under "quantity times density". That is the expectation. For several variables, the area becomes a volume, and you integrate one variable at a time.
- When no neat formula exists, evaluate the area numerically: join the curve's points with straight lines (trapezoid), or with parabolas (Simpson), or just throw random darts and count (Monte Carlo). The darts are slow in one dimension but their accuracy does not care how many dimensions there are.

## Definite integrals and the fundamental theorem

Divide $[a,b]$ into $n$ pieces of width $h=(b-a)/n$ with points $x_k=a+kh$. The **Riemann sum** is

$$
S_n=\sum_{k=0}^{n-1}f(x_k)\,h ,
$$

and the **definite integral** is its limit as $n\to\infty$: $\int_a^bf(x)\,dx=\lim S_n$ (for continuous $f$). Areas below the axis count as negative.

**Fundamental theorem of calculus.** Define $F(x)=\int_a^xf(t)\,dt$. Then $F'(x)=f(x)$, and consequently for any antiderivative $G$ (any $G$ with $G'=f$)

$$
\int_a^bf(x)\,dx=G(b)-G(a).
$$

*Why $F'=f$.* $F(x+h)-F(x)=\int_x^{x+h}f(t)\,dt\approx f(x)\,h$ for small $h$, since $f$ barely changes on a short interval. Dividing by $h$ and letting $h\to0$ gives $F'(x)=f(x)$. Any two antiderivatives differ by a constant, which cancels in $G(b)-G(a)$.

### Two techniques

**Substitution** is the chain rule read backwards. If $u=g(x)$ then $du=g'(x)dx$ and

$$
\int_a^bf(g(x))\,g'(x)\,dx=\int_{g(a)}^{g(b)}f(u)\,du .
$$

Example: $\int_0^12x\,e^{x^2}dx$; with $u=x^2$, $du=2x\,dx$, this is $\int_0^1e^u\,du=e-1=1.71828$. Substitution is also how a density transforms under a change of variables: a Jacobian factor appears (see [[f2n.06]]).

**Integration by parts** is the product rule read backwards. From $(uv)'=u'v+uv'$, integrating from $a$ to $b$:

$$
\int_a^bu\,v'\,dx=\big[uv\big]_a^b-\int_a^bu'\,v\,dx .
$$

Example: $\int_0^1x\,e^x\,dx$ with $u=x$, $v'=e^x$: $[xe^x]_0^1-\int_0^1e^x\,dx=e-(e-1)=1$.

## Expectations are integrals

A continuous random variable $X$ has a density $p(x)\ge0$ with $\int_{-\infty}^{\infty}p(x)\,dx=1$ (Deisenroth Def. 6.1), and $P(a\le X\le b)=\int_a^bp(x)\,dx$. The cumulative distribution function $F(x)=P(X\le x)=\int_{-\infty}^xp(t)\,dt$ is the running area. Two consequences the book stresses: $P(X=x)=0$ for every single $x$ (an interval of zero width), and $p(x)$ can exceed 1 as long as the area is 1. Example from the book: the uniform density on $[0.9,1.6]$ has height $1/0.7\approx1.43$.

The expectation of a function $g(X)$ replaces the discrete weighted sum $\sum_ig(x_i)p_i$ by

$$
\mathbb{E}[g(X)]=\int g(x)\,p(x)\,dx,
$$

so the mean is $g(x)=x$, and the variance is $\mathbb{E}[X^2]-(\mathbb{E}X)^2$ with $g(x)=x^2$. Deisenroth uses exactly this to compute the mean and variance of a mixture of two densities, $\alpha p_1+(1-\alpha)p_2$: the integral of a sum is the sum of the integrals, so $\mathbb{E}[X^2]=\alpha(\mu_1^2+\sigma_1^2)+(1-\alpha)(\mu_2^2+\sigma_2^2)$.

**Several variables.** For a joint density $p(x,y)$: $\iint p\,dx\,dy=1$, and the **marginal** is obtained by integrating out the other variable, $p(x)=\int p(x,y)\,dy$ (the integral analogue of summing a row of a table). Iterated integrals can be done in either order for well-behaved functions (Fubini's theorem, quoted). Bishop's Appendix B lists closed-form results of such integrals for standard distributions: for the Beta density on $[0,1]$ the mean is $a/(a+b)$ and the Gaussian marginal of a subset of variables is the Gaussian with the corresponding block of the mean and covariance (B.51).

## The Gaussian integral

The univariate Gaussian density (Deisenroth Eq. 6.62) is $p(x)=\frac1{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$. Why that constant? Because the density must integrate to 1, and that requires

$$
I=\int_{-\infty}^{\infty}e^{-x^2/2}\,dx=\sqrt{2\pi}.
$$

The function $e^{-x^2/2}$ has no elementary antiderivative, so the trick is to square the integral and go to two dimensions:

$$
I^2=\int e^{-x^2/2}dx\int e^{-y^2/2}dy=\iint_{\mathbb{R}^2}e^{-(x^2+y^2)/2}\,dx\,dy .
$$

Switch to polar coordinates $x=r\cos\theta$, $y=r\sin\theta$, for which $x^2+y^2=r^2$ and the area element is $dx\,dy=r\,dr\,d\theta$ (the Jacobian from [[f2n.06]]):

$$
I^2=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta=2\pi\int_0^\infty e^{-r^2/2}\,r\,dr .
$$

The inner integral has the substitution $u=r^2/2$, $du=r\,dr$: $\int_0^\infty e^{-u}du=1$. So $I^2=2\pi$ and $I=\sqrt{2\pi}$. Substituting $z=(x-\mu)/\sigma$ gives the general constant $\sqrt{2\pi\sigma^2}$. The multivariate version (Eq. 6.63) has the constant $(2\pi)^{-D/2}|\Sigma|^{-1/2}$.

**The second moment.** For $X\sim\mathcal N(0,1)$, integrate by parts with $u=x$ and $v'=xe^{-x^2/2}$ ($v=-e^{-x^2/2}$):

$$
\int x\cdot xe^{-x^2/2}dx=\big[-xe^{-x^2/2}\big]_{-\infty}^{\infty}+\int e^{-x^2/2}dx=0+\sqrt{2\pi},
$$

so $\mathbb{E}[X^2]=\sqrt{2\pi}/\sqrt{2\pi}=1$: the variance of the standard normal is 1, as it should be.

## Numerical integration

Let $h=(b-a)/n$ and $y_k=f(a+kh)$.

- **Left Riemann sum:** $h\sum_{k=0}^{n-1}y_k$. Error $O(h)$.
- **Trapezoid rule:** replace the curve on each strip by a straight segment: $h\big(\tfrac12y_0+y_1+\dots+y_{n-1}+\tfrac12y_n\big)$. Error $O(h^2)$: doubling $n$ cuts the error by about 4.
- **Simpson's rule** ($n$ even): replace the curve by a parabola through every three points: $\frac h3\big(y_0+4y_1+2y_2+4y_3+\dots+4y_{n-1}+y_n\big)$. Error $O(h^4)$: doubling $n$ cuts the error by about 16; it is exact for polynomials up to degree 3.
- **Monte Carlo:** draw $u_1,\dots,u_n$ uniformly in $[a,b]$ and estimate $\int_a^bf\approx\frac{b-a}n\sum f(u_i)$. This is an average of independent terms, so the error is random with standard deviation $\sigma/\sqrt n$, where $\sigma$ is the standard deviation of $(b-a)f(u)$. Error $O(n^{-1/2})$: 100 times more samples gives 10 times less error. Slow in 1D... but see the next paragraph.

**The curse of dimensionality.** A grid rule with $m$ points per axis in $d$ dimensions needs $m^d$ evaluations: $m=100$ and $d=5$ already means $10^{10}$ points. Monte Carlo keeps the rate $n^{-1/2}$ whatever $d$ is (the constant depends on the variance of the integrand, not on $d$ directly). That is why every expectation in high-dimensional ML (Bayesian posteriors, ELBOs, policy gradients) is estimated by sampling. Training with mini-batches is Monte Carlo integration of the expected loss. Variance reduction (control variates, importance sampling) improves the constant.

Gaussians are a special case: the trapezoid rule on a smooth, rapidly decaying function converges much faster than $O(h^2)$, as the example shows.

## Worked example

All values come from the exercise code.

1. Left Riemann sum of $f(x)=x$ on $[0,1]$ with $n=100$: $0.495$ (the exact area is $0.5$; the error is $h/2=0.005$, order $h$).
2. $\int_0^\pi\sin x\,dx=2$ exactly. Errors against the exact value:

| $n$ | trapezoid error | Simpson error |
|---|---|---|
| 4 | $-0.10388$ | $+0.004560$ |
| 8 | $-0.025768$ | $+0.000269$ |
| 16 | $-0.006430$ | $+0.0000166$ |
| 32 | $-0.001607$ | $+0.00000103$ |

Trapezoid error falls by 4 each time $n$ doubles; Simpson's by about 16.

3. Monte Carlo for the same integral (uniform samples, seed 0): $n=100$ gives $1.866\pm0.105$; $n=10^4$ gives $1.9970\pm0.0097$; $n=10^6$ gives $1.99914\pm0.00097$. The standard error shrinks by 10 for each 100x increase of $n$.
4. Gaussian integral: the trapezoid rule for $\int e^{-x^2/2}dx$ on $[-8,8]$ has error $0.036$ with $8$ intervals, $1.3\times10^{-8}$ with $16$, and about $10^{-15}$ with $32$. On $[-10,10]$ with $2000$ intervals it returns $2.5066283=\sqrt{2\pi}$.
5. Expectations as integrals (Simpson on $[-8,8]$): $\int p=1$, $\mathbb{E}[X]=0$, $\mathbb{E}[X^2]=1$ for the standard normal, and $P(|X|\le1.96)=0.95000$.
6. The Beta$(2,5)$ density is $30x(1-x)^4$; its mean computed by Simpson is $0.285714=2/7=a/(a+b)$, matching Bishop's formula (B.7).
7. By parts: $\int_0^1xe^x\,dx=1$ (Simpson with $n=20$: $1.0000003$). By substitution: $\int_0^12xe^{x^2}dx=e-1$ (Simpson $n=100$: $1.7182818$).
8. Volume of the unit ball: Monte Carlo with $2\times10^5$ points in the cube $[-1,1]^d$ gives $3.142$ for $d=2$ (exact $\pi$) and $5.225$ for $d=5$ (exact $5.264$). A grid with 100 points per axis would need $10^{10}$ evaluations for $d=5$.

## Implementation

```python
import numpy as np

def trapezoid(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2)

def simpson(f, a, b, n):                      # n must be even
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])

def mc_integrate(f, a, b, n, rng):
    vals = (b - a) * f(rng.uniform(a, b, n))
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n)

def expectation(pdf, g, a, b, n=400):
    return simpson(lambda x: g(x) * pdf(x), a, b, n)
```

Libraries: `scipy.integrate.quad` (adaptive Gauss-Kronrod), `scipy.integrate.simpson`, `np.trapz`/`np.trapezoid`; for high dimension, quasi-Monte Carlo (`scipy.stats.qmc`) and MCMC.

## Shapes

- A function sampled on $n+1$ grid points: array of shape $(n+1,)$. For a $d$-dimensional grid the array has $(n+1)^d$ entries.
- Monte Carlo points in $d$ dimensions: shape $(n,d)$; function values: $(n,)$; the estimate is a scalar and its error scales with $1/\sqrt n$ whatever $d$ is.

## Common misconceptions

- "The density is a probability." Only areas are probabilities; the density may be larger than 1.
- "Probability zero means impossible." Every individual value has probability zero for a continuous variable, yet one of them occurs.
- "The antiderivative exists in closed form for any reasonable function." $e^{-x^2/2}$ has none; the Gaussian integral needs a trick, and most integrals in ML are done numerically.
- "Monte Carlo is a bad method because it converges slowly." In high dimension it is often the only feasible one.
- "A higher-order rule is always better." Simpson needs smoothness; for functions with kinks or jumps (such as an indicator) it loses its advantage.

## Key takeaways

- The integral is the limit of Riemann sums; the fundamental theorem says $\int_a^bf=G(b)-G(a)$ for an antiderivative $G$.
- Substitution reverses the chain rule; parts reverse the product rule.
- A density is non-negative with total area 1; $\mathbb{E}[g(X)]=\int g\,p$; marginals integrate out variables.
- $\int e^{-x^2/2}dx=\sqrt{2\pi}$ via polar coordinates, which fixes the Gaussian normalisation.
- Trapezoid is $O(h^2)$, Simpson $O(h^4)$, Monte Carlo $O(n^{-1/2})$ independently of the dimension.

## Explain it yourself

1. Explain, without formulas, why the average of a function over a random input is an area under "function times density".
2. Explain why the polar-coordinates trick computes the Gaussian integral, and where the extra factor $r$ comes from.

## Further reading

- Deisenroth et al., Ch. 6.2 and 6.5; Bishop Appendix B (table of distributions and their moments).
- Repo lesson `phases/01-math-foundations/22-stochastic-processes` for MCMC as integration by sampling.
- Return to [[mx.04]] for Metropolis-Hastings and to [[mx.02]] where the continuous Fourier transform is defined by an integral.

## Mastery criteria

- [ ] I can evaluate simple definite integrals by the fundamental theorem, substitution and parts.
- [ ] I can write $P(a\le X\le b)$, a marginal and an expectation as integrals.
- [ ] I can derive $\int e^{-x^2/2}dx=\sqrt{2\pi}$.
- [ ] I can implement trapezoid, Simpson and Monte Carlo and say how each error scales.
