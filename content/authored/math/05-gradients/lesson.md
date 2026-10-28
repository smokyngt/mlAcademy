---
id: f2n.05
title: Partial derivatives and the gradient
summary: Derivatives of functions of many variables, and why the gradient points uphill.
minutes: 55
teaches: [calc.partial, calc.gradient]
requires: [calc.derivative, la.dot]
sources:
  - math/math for ML.pdf (Ch. 5 Vector Calculus, Sec. 5.2 Partial differentiation and gradients, PDF p.145-176)
  - ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md
---

## Learning objectives

- Compute partial derivatives and assemble a gradient.
- State the shape of the gradient of $f:\mathbb{R}^n\to\mathbb{R}$ and why it equals the shape of the input.
- Prove that the gradient direction is the direction of steepest local increase.
- Derive the gradient of mean squared error for linear regression, with every shape.
- Verify any gradient numerically.

## Prerequisites

Derivatives ([[f2n.04]]) and the dot product ([[f2n.02]]).

## Where this fits

A real model does not have one knob; it has thousands or millions of parameters. Training needs, for every one of them, the answer to "which way should I turn you?". Doing the single-knob derivative of lesson 4 one parameter at a time, and then packaging all the answers into one object, is exactly what the **gradient** is.

This lesson also contains the first real formula of machine learning that you will derive yourself: the gradient of the error of a linear model. If you can do that one with the shapes right, the neural network version later is the same movie with more actors.

## The idea in plain words

- Imagine you are standing on a **hillside in thick fog** and want to go down. You cannot see the valley, but you can feel the slope under your feet in each compass direction: steep toward the north, gently rising toward the east, and so on.
- Each of those per-direction slopes is a **partial derivative**: "if I only move along this one direction and freeze the others, how fast does the height change?".
- Put all of them into one list and you have the **gradient**. It behaves like an **arrow pointing straight uphill**, and its length tells you how steep the hill is. To go *down*, walk the opposite way.
- It only knows about the ground right under your feet. It does not point to the valley, only to where it is locally steepest. That is why we walk in small steps and look again.
- One bookkeeping rule saves you from many bugs: the gradient has **the same shape as the thing you are changing**. Ten knobs, a list of ten slopes. A matrix of weights, a matrix of slopes.

## Intuition

A loss depends on thousands or millions of parameters. Standing at a point on the loss "landscape", each parameter is a compass direction. The **partial derivative** with respect to one parameter is the slope if you walk in only that direction. The **gradient** stacks all those slopes into one vector, and that vector points straight uphill. To lower the loss, walk the opposite way.

## Formal definitions

Let $f:\mathbb{R}^n\to\mathbb{R}$ take a vector $x=(x_1,\dots,x_n)$ to a number.

**Partial derivative.** Freeze all coordinates except $x_i$ and differentiate:

$$
\frac{\partial f}{\partial x_i}(x)=\lim_{h\to0}\frac{f(x+he_i)-f(x)}{h},
$$

where $e_i$ is the $i$-th standard basis vector.

**Gradient.**

$$
\nabla f(x)=\begin{pmatrix}\partial f/\partial x_1\\ \vdots\\ \partial f/\partial x_n\end{pmatrix}\in\mathbb{R}^n .
$$

> **Shape rule:** the gradient of a scalar function has exactly the shape of its input. If the parameters are a $d\times k$ matrix $W$, then $\nabla_W L$ is also $d\times k$. You will use this to catch bugs all the time.

**Directional derivative.** The rate of change of $f$ along a *unit* vector $u$ is $D_uf(x)=\lim_{t\to0}\frac{f(x+tu)-f(x)}{t}$. For differentiable $f$, the first-order expansion $f(x+tu)\approx f(x)+t\,\nabla f(x)\cdot u$ gives

$$
D_uf(x)=\nabla f(x)\cdot u .
$$

## Derivation: why the gradient is the direction of steepest ascent

By lesson 2, for a unit vector $u$ at angle $\theta$ to $\nabla f$:

$$
D_uf=\nabla f\cdot u=\|\nabla f\|\,\|u\|\cos\theta=\|\nabla f\|\cos\theta .
$$

This is largest when $\cos\theta=1$, i.e. $u=\nabla f/\|\nabla f\|$, and the maximal rate is $\|\nabla f\|$. It is most negative for $u=-\nabla f/\|\nabla f\|$, and zero for directions perpendicular to $\nabla f$ (along level curves). *Local* and *first-order* matter: for a large step the surface curves away and the statement can fail. $\blacksquare$

## Worked example 1

$f(x,y)=x^2+3y^2$. Treat $y$ as constant to get $\partial f/\partial x=2x$; treat $x$ as constant to get $\partial f/\partial y=6y$. So $\nabla f=(2x,6y)$. At $(1,1)$: $\nabla f=(2,6)$, with $\|\nabla f\|=\sqrt{40}\approx6.32$. The slope along $u=(1,0)$ is $2$; along the steepest direction $(2,6)/6.32$ it is $6.32$; along the perpendicular $(3,-1)/\sqrt{10}$ it is $0$ (check: $2\cdot3-6\cdot1=0$).

## Worked example 2: gradient of mean squared error

Data matrix $X\in\mathbb{R}^{N\times d}$ (rows are examples), targets $y\in\mathbb{R}^{N}$, weights $w\in\mathbb{R}^{d}$. Predictions $\hat y=Xw\in\mathbb{R}^N$ and residuals $r=Xw-y\in\mathbb{R}^N$.

$$
L(w)=\frac1N\sum_{i=1}^{N}\big(x_i\cdot w-y_i\big)^2=\frac1N\|r\|^2 .
$$

Differentiate with respect to one weight $w_j$. Only $x_i\cdot w=\sum_k x_{ik}w_k$ depends on $w_j$, with $\partial(x_i\cdot w)/\partial w_j=x_{ij}$. Using $(u^2)'=2u$:

$$
\frac{\partial L}{\partial w_j}=\frac1N\sum_{i=1}^N 2\,r_i\,x_{ij}=\frac2N\sum_i x_{ij}r_i=\frac2N\,(X^\top r)_j .
$$

Stacking all $j$:

$$
\boxed{\nabla_wL=\frac2N X^\top(Xw-y)}\qquad\text{shapes: }(d\times N)(N)=(d).
$$

The shape of $\nabla_wL$ is $d$, the same as $w$, as promised.

## Implementation

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float); e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

mse_grad = lambda X, y, w: 2 / len(y) * X.T @ (X @ w - y)
```

## Common misconceptions

- **The gradient is not a derivative of the *vector*; it is a vector of derivatives of a scalar.** For vector-valued functions you need the Jacobian (next lesson).
- **Zero gradient does not mean minimum.** It may be a maximum or a saddle point; in high dimensions saddles are the common case.
- **Steepest ascent is local.** It does not point at the global minimum; that is why optimisation is iterative.
- **Gradient shape confusion**: writing $\nabla_wL$ as $Xr^\top$ or forgetting the transpose is the classic bug. Check shapes against $w$.

## Key takeaways

- The gradient stacks the partial derivatives; it has exactly the shape of the input.
- `D_u f = grad f . u`, so the steepest ascent direction is `grad f` (Cauchy-Schwarz from lesson 2).
- For MSE: `grad_w = (2/N) X^T (Xw - y)`. Check its shape against `w`.
- A zero gradient only means a stationary point, not necessarily a minimum.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Prove the gradient is the direction of steepest ascent in your own words.
2. Derive the MSE gradient one component `dL/dw_j` at a time, then assemble it into `X^T r`.

## Further reading

- *Mathematics for Machine Learning*, Sec. 5.2-5.3, PDF p.145-176.
- Source library: `ai-engineering-from-scratch` phase 01, lesson 04.

## Mastery criteria

You can compute the gradient of a two-variable polynomial by hand, derive $\frac{2}{N}X^\top(Xw-y)$ without notes, and write a numerical gradient checker.
