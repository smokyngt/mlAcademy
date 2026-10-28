---
id: f2n.07
title: Gradient descent
summary: Why stepping against the gradient lowers the loss, when it converges, and how the learning rate and conditioning decide everything.
minutes: 60
teaches: [opt.gradient-descent]
requires: [calc.gradient, calc.numerical-diff]
sources:
  - math/math for ML.pdf (Ch. 7 Continuous Optimization, Sec. 7.1 Optimization using gradient descent, PDF p.231)
  - ai-engineering-from-scratch/phases/01-math-foundations/08-optimization/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/06-optimizers/docs/en.md
---

## Learning objectives

- Write the gradient descent update and justify it with a first-order argument.
- Derive the exact convergence condition $0<\eta<2/\lambda$ for a quadratic and the effect of conditioning.
- Predict divergence, slow convergence, and oscillation from the learning rate.
- Implement gradient descent and use it to fit a linear model.

## Prerequisites

Gradients ([[f2n.05]]) and numerical checking ([[f2n.04]]).

## Intuition

You are on a foggy hillside and want the valley. You can only feel the slope under your feet. Take a small step downhill, feel again, repeat. The gradient tells you "uphill"; you step the opposite way. The step size $\eta$ (the *learning rate*) is the whole difficulty: too small and you crawl, too large and you overshoot the valley and bounce out.

Experiment first. Pick a learning rate and a condition number $\kappa$ (how much steeper one axis is than the other) and watch the path.

```viz
gradient-descent
```

## The algorithm

Minimise $L:\mathbb{R}^n\to\mathbb{R}$. Starting from $\theta_0$,

$$
\theta_{t+1}=\theta_t-\eta\,\nabla L(\theta_t).
$$

**Why it decreases the loss (for small $\eta$).** The first-order expansion from lesson 5 with step $\Delta=-\eta g$, where $g=\nabla L(\theta)$:

$$
L(\theta-\eta g)\approx L(\theta)+\nabla L(\theta)\cdot(-\eta g)=L(\theta)-\eta\|g\|^2 .
$$

Because $\|g\|^2\ge0$, the loss falls whenever $\eta>0$ is small enough that the approximation is valid, and it falls fastest per unit step along $-g$ (lesson 5). It stops moving when $g=0$.

## Derivation: when does it converge? (the quadratic)

One dimension: $L(\theta)=\tfrac12\lambda\theta^2$ with curvature $\lambda>0$ and minimum at 0. Then $L'(\theta)=\lambda\theta$ and

$$
\theta_{t+1}=\theta_t-\eta\lambda\theta_t=(1-\eta\lambda)\,\theta_t\quad\Rightarrow\quad\theta_t=(1-\eta\lambda)^t\theta_0 .
$$

This converges to 0 iff $|1-\eta\lambda|<1$, i.e. $0<\eta<\dfrac{2}{\lambda}$. Three regimes:

- $0<\eta<1/\lambda$: monotone convergence (factor in $(0,1)$).
- $1/\lambda<\eta<2/\lambda$: converges **with oscillation** (factor in $(-1,0)$).
- $\eta>2/\lambda$: $|1-\eta\lambda|>1$, **divergence**.

In $n$ dimensions with $L(\theta)=\tfrac12\theta^\top H\theta$ and eigenvalues $0<\lambda_{\min}\le\dots\le\lambda_{\max}$ of $H$, each eigen-direction behaves like the 1-D case with its own $\lambda_i$. The step must satisfy $\eta<2/\lambda_{\max}$ to avoid blowing up the steepest direction, but then the flattest direction contracts by only $1-\eta\lambda_{\min}$ per step, which is slow. Choosing the best $\eta=\dfrac{2}{\lambda_{\max}+\lambda_{\min}}$ gives the per-step contraction

$$
\rho=\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}=\frac{\kappa-1}{\kappa+1},\qquad\kappa=\frac{\lambda_{\max}}{\lambda_{\min}} .
$$

With $\kappa=1$ it converges in one step; with $\kappa=100$, $\rho\approx0.98$ and it takes ~100 steps to shrink the error by $e^{-2}$. **Ill-conditioning, not dimensionality alone, is what makes optimisation slow.** This motivates momentum, Adam, normalisation layers and good initialisation.

## Worked examples

1. $L(\theta)=\theta^2$ ($\lambda=2$), $\theta_0=3$, $\eta=0.1$: $\theta_1=3-0.1\cdot6=2.4$, $\theta_2=2.4-0.1\cdot4.8=1.92$. The ratio is $0.8=1-\eta\lambda$. ✓ With $\eta=1.1$: ratio $1-2.2=-1.2$, so $3\to-3.6\to4.32\to\cdots$ diverges.
2. $L(x,y)=\tfrac12(x^2+10y^2)$, so $\lambda_{\max}=10,\lambda_{\min}=1$, $\kappa=10$. Stability needs $\eta<0.2$. Best $\eta=2/11\approx0.182$ with $\rho=9/11\approx0.818$. At $\eta=0.19$ the $y$-direction factor is $1-1.9=-0.9$: slowly oscillating convergence.

## Stochastic gradient descent

For a loss that is an average over data, $L=\frac1N\sum_iL_i$, the gradient costs $O(N)$. Using a random mini-batch $B$ of size $b$ gives $\hat g=\frac1b\sum_{i\in B}\nabla L_i$ with $\mathbb{E}[\hat g]=\nabla L$ (unbiased) and variance shrinking as $1/b$. SGD trades exact gradients for many cheap steps; the noise also helps leave sharp minima and saddles. Momentum, RMSProp and Adam (phase 03 of the source library) change the *direction/scale* of the step but the update shape is the same.

## Implementation

```python
import numpy as np

def gradient_descent(grad, x0, lr, steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad(x)
        path.append(x.copy())
    return np.array(path)        # shape (steps+1, n)
```

## Common misconceptions

- **"Gradient descent finds the global minimum."** Only for convex losses (such as the linear regression MSE). Neural networks are non-convex; you reach a good *local* solution.
- **"Lower learning rate is always safer."** It is safer against divergence but can leave you stuck on a plateau or use your whole compute budget.
- **"The loss must decrease every step."** True for exact gradient descent with small $\eta$; false for SGD, where it is noisy but decreasing on average.
- **A diverging loss (NaN/inf) is usually the learning rate**, not "bad data". Divide by 10 and retry.

## Key takeaways

- `theta <- theta - eta grad L`; for small `eta` the loss changes by about `-eta |grad|^2`.
- On a quadratic with curvature `lambda`, it converges iff `0 < eta < 2/lambda`.
- The condition number `kappa = lambda_max / lambda_min` sets how slowly it converges: `(kappa-1)/(kappa+1)` per step at best.
- NaN early in training usually means the learning rate is too large.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Predict what happens for `eta` = 0.5/lambda, 1.5/lambda, 2.5/lambda, then check with the figure.
2. Why does an ill-conditioned loss make gradient descent slow even in two dimensions?

## Further reading

- *Mathematics for Machine Learning*, Sec. 7.1 (gradient descent, step-size, momentum, SGD), PDF p.231.
- Source library: `ai-engineering-from-scratch` phase 01 lesson 08 and phase 03 lesson 06.

## Mastery criteria

You can predict, from the curvature, the range of learning rates that converge, compute the optimal rate for a 2-D quadratic, and diagnose divergence from a loss curve.
