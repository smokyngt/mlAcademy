---
id: f2n.06
title: The chain rule, Jacobians and vector-Jacobian products
summary: How derivatives flow through compositions, with every shape, and the three backward formulas that make backpropagation work.
minutes: 65
teaches: [calc.chain, calc.jacobian]
requires: [calc.derivative, calc.gradient, la.matrix]
sources:
  - math/math for ML.pdf (Ch. 5 Vector Calculus, Sec. 5.3 Gradients of vector-valued functions and 5.6 Backpropagation and automatic differentiation, PDF p.145-176)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 5.3 Error Backpropagation, PDF p.261
  - ai-engineering-from-scratch/phases/01-math-foundations/05-chain-rule-and-autodiff/docs/en.md
---

## Learning objectives

- Apply the scalar chain rule to nested functions.
- Define the Jacobian of a vector-valued function and state its shape.
- Write the multivariate chain rule as a matrix product and verify the shapes.
- Derive, with shapes, the backward formulas for a linear map $y=Wx$ and an elementwise activation.
- Explain why we multiply the upstream gradient by $J^\top$ instead of forming Jacobians.

## Prerequisites

Derivatives ([[f2n.04]]), gradients ([[f2n.05]]), matrix products ([[f2n.03]]).

## Where this fits

A neural network is a long chain of small steps: the input is transformed by the first layer, the result by the second, and so on until a final error is computed. A knob in the first layer affects the error only *through* everything after it. To train the network we need to know how much each knob, early or late, contributes to the final error.

The **chain rule** is the rule that tells you how effects pass along a chain. Its matrix version (Jacobians) is what makes it work for layers with many numbers. This lesson is the bridge between the Mathematics part and the Machine Learning part: the three little formulas at the end are, literally, the whole of backpropagation.

## The idea in plain words

- **Exchange rates.** If 1 euro buys 1.1 dollars and 1 dollar buys 150 yen, then 1 euro buys 1.1 x 150 = 165 yen. Rates along a chain **multiply**. That is the chain rule: the sensitivity of the last step to the first is the product of the sensitivities of each link.
- If an input can reach the output by **two different routes**, the effects of the two routes **add**.
- With a single number per step, one exchange rate per link is enough. With *lists* of numbers, you need a whole **table of exchange rates** (how much each output reacts to each input). That table is the **Jacobian**, and multiplying tables is matrix multiplication.
- **Backwards is cheaper.** Suppose the final error is a single number. Starting there and asking "how much is each earlier step to blame?" lets you reuse the answers as you go back, so one backward sweep gives the blame for every knob. This is "blame assignment", and it is what backpropagation does.

## Intuition

If $y$ depends on $u$ and $u$ depends on $x$, then nudging $x$ by $\delta$ nudges $u$ by $u'\delta$, which nudges $y$ by $y'(u)\,u'\delta$. **Rates multiply along the path.** A neural network is a long chain of such dependencies, so its gradient is a product of local rates, each computed once, from the output backwards.

## Scalar chain rule

For $y=g(f(x))$,

$$
\frac{dy}{dx}=g'\big(f(x)\big)\cdot f'(x).
$$

*Example.* $y=\sigma(wx+b)$ as a function of $w$: let $z=wx+b$, so $dz/dw=x$ and $dy/dz=\sigma'(z)=\sigma(z)(1-\sigma(z))$. Then $dy/dw=\sigma'(z)\,x$.

*Several paths.* If $y=g(u_1(x),u_2(x))$ then $\dfrac{dy}{dx}=\dfrac{\partial g}{\partial u_1}\dfrac{du_1}{dx}+\dfrac{\partial g}{\partial u_2}\dfrac{du_2}{dx}$: sum over every path from $x$ to $y$. This sum is why a weight that influences several later units gets several contributions.

## The Jacobian

For $f:\mathbb{R}^n\to\mathbb{R}^m$ with outputs $f_1,\dots,f_m$, the **Jacobian** collects all first partial derivatives:

$$
J_f(x)=\begin{pmatrix}\partial f_1/\partial x_1&\cdots&\partial f_1/\partial x_n\\ \vdots&&\vdots\\ \partial f_m/\partial x_1&\cdots&\partial f_m/\partial x_n\end{pmatrix}\in\mathbb{R}^{m\times n},\qquad (J_f)_{ij}=\frac{\partial f_i}{\partial x_j}.
$$

Row $i$ is the (transposed) gradient of output $f_i$. The first-order picture: $f(x+\delta)\approx f(x)+J_f(x)\,\delta$. The Jacobian is the best *linear map* approximating $f$ near $x$.

**Multivariate chain rule.** For $f:\mathbb{R}^n\to\mathbb{R}^m$ and $g:\mathbb{R}^m\to\mathbb{R}^k$, $h=g\circ f$:

$$
J_h(x)=J_g\big(f(x)\big)\,J_f(x),\qquad (k\times n)=(k\times m)(m\times n).
$$

This holds because linearising each map and composing the linear maps multiplies their matrices; entrywise it is the sum-over-paths rule: $\frac{\partial h_i}{\partial x_j}=\sum_{l=1}^m\frac{\partial g_i}{\partial f_l}\frac{\partial f_l}{\partial x_j}$.

## From Jacobians to backpropagation: the vector-Jacobian product

In training, the end of the chain is a **scalar** loss $L$, so $k=1$ and $J_L$ is a row. Write the *upstream gradient* $g=\nabla_yL\in\mathbb{R}^m$ for the output $y=f(x)$ of one stage. Then

$$
\nabla_xL=J_f(x)^\top\,g\in\mathbb{R}^n .
$$

Computing $J_f^\top g$ (a "vector-Jacobian product") never requires building a huge Jacobian, and for the common layers it has a closed form. Three formulas carry all of backpropagation:

### 1. Linear map $y=Wx$ with $W\in\mathbb{R}^{m\times n}$

$y_i=\sum_jW_{ij}x_j$, so $\partial y_i/\partial x_j=W_{ij}$ and $J=W$. Hence

$$
\nabla_xL=W^\top g\ \ (n),\qquad\text{and since }\frac{\partial L}{\partial W_{ij}}=\sum_{l}\frac{\partial L}{\partial y_l}\frac{\partial y_l}{\partial W_{ij}}=g_i\,x_j:\qquad\nabla_WL=g\,x^\top\ \ (m\times n).
$$

### 2. Bias addition $y=z+b$

$J=I$, so $\nabla_zL=g$ and $\nabla_bL=g$.

### 3. Elementwise activation $y=\phi(z)$

$y_i=\phi(z_i)$ depends only on $z_i$, so $J=\operatorname{diag}(\phi'(z_1),\dots,\phi'(z_m))$ and $\operatorname{diag}(a)\,g=a\odot g$:

$$
\nabla_zL=g\odot\phi'(z).
$$

(We never build the $m\times m$ diagonal matrix.)

## Worked example with numbers

$W=\begin{pmatrix}1&2\\3&4\end{pmatrix}$, $x=\begin{pmatrix}1\\1\end{pmatrix}$, so $y=Wx=(3,7)^\top$. Suppose the loss gives upstream gradient $g=(1,-1)^\top$.

$$
\nabla_xL=W^\top g=\begin{pmatrix}1&3\\2&4\end{pmatrix}\begin{pmatrix}1\\-1\end{pmatrix}=\begin{pmatrix}-2\\-2\end{pmatrix},\qquad\nabla_WL=gx^\top=\begin{pmatrix}1\\-1\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix}=\begin{pmatrix}1&1\\-1&-1\end{pmatrix}.
$$

Sanity check with $L=g\cdot y=y_1-y_2$: then $L=(W_{11}x_1+W_{12}x_2)-(W_{21}x_1+W_{22}x_2)$; $\partial L/\partial x_1=W_{11}-W_{21}=1-3=-2$ ✓ and $\partial L/\partial W_{22}=-x_2=-1$ ✓.

## Implementation

```python
import numpy as np

def linear_backward(W, x, g):
    dx = W.T @ g              # (n,)   = (n,m)(m,)
    dW = np.outer(g, x)       # (m,n)  = (m,1)(1,n)
    return dx, dW

def relu_backward(z, g):
    return g * (z > 0)        # elementwise: g * phi'(z)
```

## Common misconceptions

- **"Backpropagation is a special algorithm."** It is the chain rule, applied in reverse order so that each intermediate gradient is reused. Forward-order evaluation would cost one pass per input; reverse order costs one pass per output (here, one).
- **Transposes are not decoration.** The backward map of $W$ is $W^\top$ because rows and columns swap roles when we go from "effect of input on output" to "blame on input from output".
- **Shapes are the proof.** If your $\nabla_WL$ is not $m\times n$, the formula is wrong.
- **The derivative of ReLU at 0** is undefined; implementations use $0$ (or $1$). It rarely matters in practice.

## Key takeaways

- Rates multiply along a path and add across paths.
- The Jacobian of `f: R^n -> R^m` is `m x n`; chain rule is the product `J_g J_f`.
- Backward formulas: linear `dx = W^T g`, `dW = g x^T`; bias `db = g`; elementwise `dz = g * phi'(z)`.
- We multiply the upstream gradient by `J^T` instead of building `J`: that is a vector-Jacobian product.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Why is the backward map of `W` equal to `W^T`? Explain with a 2x2 example.
2. What are the shapes of `dW`, `dx` and `db` for `y = Wx + b` with `W: (4, 3)`?

## Further reading

- *Mathematics for Machine Learning*, Ch. 5, PDF p.145-176 (Jacobians, backpropagation, automatic differentiation).
- Bishop, *PRML*, Sec. 5.3, PDF p.261.
- Source library: `ai-engineering-from-scratch` phase 01, lesson 05.

## Mastery criteria

You can derive $W^\top g$ and $g\,x^\top$ from scratch, state every shape, and check them with a finite-difference test.
