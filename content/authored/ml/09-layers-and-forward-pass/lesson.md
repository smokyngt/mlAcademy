---
id: f2n.09
title: Neurons, layers and the forward pass
summary: A dense layer is activation(XW + b). Why nonlinearity is mandatory, the common activations, parameter counts and initial scale.
minutes: 55
teaches: [nn.layer, nn.activation]
requires: [la.matrix, ml.loss, calc.derivative]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 5.1 Feed-forward network functions, PDF p.247
  - math/math for ML.pdf (Sec. 5.6 Backpropagation and automatic differentiation: the deep-network chain, PDF p.145-176)
  - ai-engineering-from-scratch/phases/03-deep-learning-core/01-the-perceptron/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/02-multi-layer-networks/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/04-activation-functions/docs/en.md
---

## Learning objectives

- Write a neuron and a dense layer, with all shapes, for a single example and for a batch.
- Prove that stacking linear layers without nonlinearity gives nothing new.
- Give the formulas and derivatives of ReLU, sigmoid and tanh.
- Count parameters and choose an initial weight scale that keeps activations at a stable size.

## Prerequisites

Matrices and shapes ([[f2n.03]]), derivatives ([[f2n.04]]), losses ([[f2n.08]]).

## Intuition

A **neuron** takes a weighted sum of its inputs (a dot product: "how much does this input look like my pattern?"), adds a bias (a threshold), and passes the result through a nonlinear **activation**. A **layer** is many neurons reading the same input, which is one matrix multiplication. A **network** is layers composed, $h=f_L\circ\cdots\circ f_1$ (lesson 1).

## Formal definitions

One neuron: $a=\phi(w\cdot x+b)$ with $w,x\in\mathbb{R}^n$, $b\in\mathbb{R}$.

Dense layer for a batch $X\in\mathbb{R}^{B\times n}$:

$$
Z=XW+\mathbf{1}b^\top,\qquad A=\phi(Z),\qquad W\in\mathbb{R}^{n\times m},\ b\in\mathbb{R}^m,\ Z,A\in\mathbb{R}^{B\times m}.
$$

The bias is added to every row ("broadcasting"). Column $k$ of $W$ is the weight vector of neuron $k$.

**Activations and derivatives**

| Name | $\phi(z)$ | $\phi'(z)$ | Notes |
|---|---|---|---|
| ReLU | $\max(0,z)$ | $1$ if $z>0$ else $0$ | cheap; default for hidden layers; "dead" if always $z\le0$ |
| Sigmoid | $\dfrac1{1+e^{-z}}$ | $\sigma(1-\sigma)$ | output in $(0,1)$; saturates, max slope $0.25$ |
| Tanh | $\tanh z$ | $1-\tanh^2z$ | zero-centred, output in $(-1,1)$, max slope $1$ |
| Identity | $z$ | $1$ | used for regression outputs and logits |

The tanh derivative follows from $\tanh z=\frac{e^z-e^{-z}}{e^z+e^{-z}}$ and the quotient rule: $\tanh'=1-\tanh^2$.

Explore them, including their derivatives:

```viz
activation-plot
```

## Why nonlinearity is mandatory

Two linear layers: $W_2(W_1x)=(W_2W_1)x$. The product $W_2W_1$ is a single matrix, so any depth of purely linear layers is equivalent to **one** linear layer. Without $\phi$ a network can only draw straight decision boundaries and cannot even compute XOR.

**XOR with two ReLU units.** Inputs $x\in\{0,1\}^2$. Let $h_1=\mathrm{ReLU}(x_1+x_2)$, $h_2=\mathrm{ReLU}(x_1+x_2-1)$, output $o=h_1-2h_2$. Then

| $x_1,x_2$ | $h_1$ | $h_2$ | $o$ |
|---|---|---|---|
| 0,0 | 0 | 0 | 0 |
| 1,0 | 1 | 0 | 1 |
| 0,1 | 1 | 0 | 1 |
| 1,1 | 2 | 1 | 0 |

That is XOR, computed by a network of weights $W_1=\begin{pmatrix}1&1\\1&1\end{pmatrix}$, $b_1=(0,-1)$, $W_2=(1,-2)^\top$. You will build it in the exercise.

## Counting parameters

A layer $n\to m$ has $nm+m$ parameters. A $784\to128\to10$ classifier has $(784\cdot128+128)+(128\cdot10+10)=100{,}480+1{,}290=101{,}770$ parameters.

## Derivation: choosing the initial scale

Take $z=\sum_{j=1}^{n}w_jx_j$ with independent zero-mean weights of variance $\sigma_w^2$ and inputs of variance $\sigma_x^2$ (and mean 0). Then

$$
\operatorname{Var}(z)=n\,\sigma_w^2\,\sigma_x^2 .
$$

To keep the variance from growing or shrinking layer after layer we need $n\sigma_w^2=1$, i.e. $\sigma_w^2=1/n$ (Xavier/Glorot-style for tanh). ReLU zeroes half of the pre-activations, halving the second moment, so compensating gives $\sigma_w^2=2/n$ (He initialisation). With a larger scale activations explode with depth; with a smaller one they vanish, and either way gradients inherit the problem.

## Implementation

```python
import numpy as np

def dense(X, W, b, phi):
    Z = X @ W + b            # (B,n)@(n,m) + (m,) -> (B,m)
    return phi(Z), Z         # keep Z: backprop needs it (lesson 10)

relu = lambda z: np.maximum(0, z)
rng = np.random.default_rng(0)
W = rng.normal(0, np.sqrt(2 / 784), size=(784, 128))   # He init
```

## Common misconceptions

- **"Hidden units need a bias only sometimes."** A bias lets the neuron shift its threshold; it costs $m$ parameters and is almost always included.
- **Dead ReLUs**: if a unit's pre-activation is negative for every input, its gradient is exactly 0 and it never recovers. Large learning rates and poor initialisation cause this.
- **Zero initial weights**: every neuron in a layer computes the same thing and gets the same gradient, so symmetry is never broken. Always initialise randomly.
- **Sigmoid for hidden layers** is rarely wise: slope $\le0.25$ makes gradients shrink by $4\times$ per layer at best.

## Key takeaways

- A dense layer is `phi(XW + b)` with `X: (B, n)`, `W: (n, m)`, `b: (m)`.
- Without a nonlinearity, depth collapses to one matrix.
- ReLU, sigmoid, tanh and their derivatives: know the shapes of their curves and the maximum slopes.
- He initialisation uses variance `2/n_in` so activations neither blow up nor vanish.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Explain in two sentences why a purely linear network cannot compute XOR, then give the weights that do with ReLU.
2. Count the parameters of a 784-128-10 network and show where each term comes from.

## Further reading

- Bishop, *PRML*, Sec. 5.1, PDF p.247.
- Source library: `ai-engineering-from-scratch` phase 03, lessons 01, 02, 04 and 08.

## Mastery criteria

You can state every shape in `A = relu(X @ W + b)`, prove that depth without nonlinearity is useless, count the parameters of an MLP, and explain why He initialisation uses $2/n$.
