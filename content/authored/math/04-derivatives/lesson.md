---
id: f2n.04
title: Derivatives and numerical differentiation
summary: The derivative as an instantaneous rate of change, the rules that compute it, and finite differences that check them.
minutes: 55
teaches: [calc.derivative, calc.numerical-diff]
requires: [alg.functions, alg.exp-log]
sources:
  - math/math for ML.pdf (Ch. 5 Vector Calculus, Sec. 5.1 Differentiation of univariate functions, PDF p.145)
  - ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md
---

## Learning objectives

- State the limit definition of the derivative and use it to differentiate $x^2$.
- Apply the power, sum, product, exponential and logarithm rules.
- Derive $\sigma'(x)=\sigma(x)(1-\sigma(x))$ for the sigmoid.
- Compute forward and central finite differences, and explain their error orders and the best step size.

## Prerequisites

Functions, exponentials and logarithms ([[f2n.01]]).

## Where this fits

Learning is a loop: try the model, measure how wrong it is, adjust, repeat. The second half, adjusting, needs one piece of information per knob: **if I turn this knob a tiny bit up, does the error go up or down, and by how much?** That quantity is the derivative.

Everything in the next lessons (gradients, the chain rule, gradient descent, backpropagation) is this one idea applied to more and more knobs. So this lesson is about getting the single-knob case completely clear, plus a practical way to check your answers with a computer.

## The idea in plain words

- Think of the **tap of a shower**. Turn it a little and the water temperature changes. "A little turn gives this much temperature change, per unit of turn" is the derivative: a *sensitivity*. Steep means the water reacts strongly; flat means it barely reacts; negative means turning up makes it colder.
- On a graph, it is the **slope** of the curve at a point. For a straight line the slope is the same everywhere. For a curve it changes, but if you **zoom in** far enough on any smooth curve it looks like a straight line, and the slope of that line is the derivative.
- Speedometer versus odometer: the odometer says where you are, the speedometer says how fast the position is changing. The derivative is the speedometer of a function.
- You can also *estimate* the derivative without algebra: nudge the input by a tiny amount `h`, see how far the output moved, and divide. That is what a computer check does, and its one subtlety (make `h` small, but not too small) is explained below.

Drag `h` in the figure and watch the slanted line turn into the tangent.

## Intuition

The derivative $f'(x)$ answers: *if I nudge the input by a tiny amount, how much does the output move, per unit nudge?* It is the slope of the tangent line. Training a model is nothing but repeating "nudge each parameter in the direction that lowers the loss", so derivatives are the engine of learning.

Drag $h$ toward zero: the secant line through $(x,f(x))$ and $(x+h,f(x+h))$ becomes the tangent.

```viz
derivative-explorer
```

## Formal definition

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
$$

If the limit exists, $f$ is differentiable at $x$. Equivalent first-order statement: $f(x+h)\approx f(x)+f'(x)\,h$ for small $h$. This **local linear approximation** is the idea that later generalises to gradients and Jacobians.

## Derivation 1: the power rule for $x^2$

$$
\frac{(x+h)^2-x^2}{h}=\frac{x^2+2xh+h^2-x^2}{h}=\frac{2xh+h^2}{h}=2x+h\;\xrightarrow{h\to0}\;2x .
$$

The general rule is $\frac{d}{dx}x^n=nx^{n-1}$ (the binomial theorem gives $(x+h)^n=x^n+nx^{n-1}h+O(h^2)$).

## Rules you will use constantly

| Rule | Statement |
|---|---|
| Constant multiple | $(cf)'=cf'$ |
| Sum | $(f+g)'=f'+g'$ |
| Product | $(fg)'=f'g+fg'$ |
| Power | $(x^n)'=nx^{n-1}$ |
| Exponential | $(e^x)'=e^x$ |
| Logarithm | $(\ln x)'=1/x$ for $x>0$ |

(The chain rule is the next-but-one lesson, after gradients.)

## Derivation 2: the sigmoid

$\sigma(x)=\dfrac{1}{1+e^{-x}}=(1+e^{-x})^{-1}$. Treat $u=1+e^{-x}$, so $\sigma=u^{-1}$, $\sigma'=-u^{-2}u'$ and $u'=-e^{-x}$. Therefore

$$
\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\sigma(x)\,\big(1-\sigma(x)\big),
$$

since $\frac{e^{-x}}{1+e^{-x}}=1-\frac{1}{1+e^{-x}}$. At $x=0$: $\sigma(0)=\tfrac12$ and $\sigma'(0)=\tfrac14$, the largest value it takes, which is why deep sigmoid networks suffer vanishing gradients (each layer multiplies by at most $0.25$).

## Numerical differentiation

We can approximate $f'$ without algebra:

- **Forward difference:** $D_+ f(x)=\dfrac{f(x+h)-f(x)}{h}$
- **Central difference:** $D_0 f(x)=\dfrac{f(x+h)-f(x-h)}{2h}$

**Error analysis.** Taylor's theorem: $f(x\pm h)=f(x)\pm hf'(x)+\tfrac{h^2}{2}f''(x)\pm\tfrac{h^3}{6}f'''(x)+\cdots$.

- Forward: $D_+f=f'+\tfrac h2f''+\cdots$, so the *truncation error* is $O(h)$.
- Central: subtracting the two expansions cancels the even terms, $f(x+h)-f(x-h)=2hf'+\tfrac{h^3}{3}f'''+\cdots$, so $D_0f=f'+\tfrac{h^2}{6}f'''+\cdots$, error $O(h^2)$.

But floating point adds *round-off error*. The numerator subtracts two nearly equal numbers, losing roughly $\varepsilon|f|$ in absolute accuracy, and dividing by $h$ gives error $\sim\varepsilon/h$. Total error $\approx Ah^p+\varepsilon/h$; it falls as $h$ shrinks, then *rises again*. Minimising gives $h\approx\varepsilon^{1/2}\approx10^{-8}$ for forward and $h\approx\varepsilon^{1/3}\approx10^{-5}$ for central differences in float64 ($\varepsilon\approx2.2\times10^{-16}$).

## Worked example

$f(x)=x^3$ at $x=2$; true derivative $3x^2=12$.

- Forward, $h=0.1$: $(2.1^3-2^3)/0.1=(9.261-8)/0.1=12.61$ (error $0.61$).
- Central, $h=0.1$: $(2.1^3-1.9^3)/0.2=(9.261-6.859)/0.2=12.01$ (error $0.01$, matching $\tfrac{h^2}{6}f'''=\tfrac{0.01}{6}\cdot6$).

Central differences are about 60 times more accurate here at the same $h$.

## Implementation

```python
import numpy as np

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
print(central_diff(sigmoid, 0.0))   # 0.25
```

## Common misconceptions

- **"Smaller $h$ is always better."** No: below the optimum, round-off dominates and the answer gets *worse*. The exercise makes you see this.
- **Differentiable is stronger than continuous.** $|x|$ is continuous at 0 but has no derivative there; ReLU has this kink, and frameworks simply pick a value (0) for the derivative at 0.
- **$(fg)'\neq f'g'$.**
- **Numerical derivatives are for checking, not training**: they cost one function evaluation per parameter. Backpropagation computes all derivatives in one backward pass.

## Key takeaways

- The derivative is the slope of the tangent: `f(x+h) ~ f(x) + f'(x) h`.
- Learn the six rules (constant, sum, product, power, exp, ln) and derive the sigmoid derivative once yourself.
- Central differences are `O(h^2)`; forward differences are `O(h)`.
- Smaller `h` is not always better: round-off grows like `eps/h`, so the best `h` is around `1e-5`.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Derive `sigma'(x) = sigma(x)(1 - sigma(x))` from memory. Why is its maximum 0.25, and what does that do to deep sigmoid networks?
2. Why is a numerical derivative used to test code but not to train a network?

## Further reading

- *Mathematics for Machine Learning*, Sec. 5.1, PDF p.145.
- Source library: `ai-engineering-from-scratch` phase 01, lesson 04.

## Mastery criteria

You can differentiate a polynomial/exponential/log expression by hand, derive $\sigma'$, and explain why a gradient check uses central differences with $h\approx10^{-5}$.
