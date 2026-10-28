---
id: f2n.10
title: Backpropagation, derived with shapes
summary: The chain rule through a two-layer network, one formula at a time, with a numeric walk-through and a batched implementation.
minutes: 75
teaches: [nn.backprop]
requires: [calc.jacobian, nn.activation, nn.layer]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 5.3 Error Backpropagation, PDF p.261-269
  - math/math for ML.pdf (Sec. 5.6 Backpropagation and automatic differentiation, PDF p.145-176)
  - ai-engineering-from-scratch/phases/03-deep-learning-core/03-backpropagation/docs/en.md
  - LLMs-from-scratch/appendix-A/01_main-chapter-code (automatic differentiation in PyTorch)
---

## Learning objectives

- Derive the gradients of every parameter of a two-layer MLP with softmax cross-entropy.
- Give the shape of every intermediate quantity and every gradient.
- Explain what must be cached in the forward pass and why.
- Walk a small numeric example through forward and backward by hand.
- Implement batched backpropagation and verify it with a finite-difference check.

## Prerequisites

The three backward formulas of lesson 6 ([[f2n.06]]), the loss gradient $p-y$ of lesson 8 ([[f2n.08]]) and the layer notation of lesson 9 ([[f2n.09]]).

## Intuition

The forward pass computes a chain of values $X\to Z_1\to A_1\to Z_2\to P\to L$. We want $\partial L/\partial(\text{every weight})$. Start at the end, where the derivative of $L$ with respect to the last values is easy ($P-Y$), and **walk backwards**, converting "how $L$ changes with the output of this stage" into "how $L$ changes with the input of this stage and with this stage's parameters". Each stage does one local multiplication, so the total cost of all gradients is about the cost of **two** forward passes.

## The network

Batch $X\in\mathbb{R}^{B\times n}$, hidden width $h$, $K$ classes, labels $y\in\{0,\dots,K-1\}^B$:

$$
\begin{aligned}
Z_1&=XW_1+b_1&&(B\times h),\ W_1:(n\times h),\ b_1:(h)\\
A_1&=\mathrm{ReLU}(Z_1)&&(B\times h)\\
Z_2&=A_1W_2+b_2&&(B\times K),\ W_2:(h\times K),\ b_2:(K)\\
P&=\operatorname{softmax}(Z_2)&&(B\times K)\\
L&=-\tfrac1B\textstyle\sum_{b}\ln P_{b,y_b}&&\text{scalar}
\end{aligned}
$$

## Backward derivation, step by step

Notation: $\bar{Q}:=\partial L/\partial Q$ has the same shape as $Q$.

**Step 1: loss and softmax** (lesson 8). $\;\bar Z_2=\dfrac1B\,(P-Y)\;\;(B\times K)$ where $Y$ is the one-hot label matrix.

**Step 2: the second linear layer $Z_2=A_1W_2+b_2$.** Entrywise, $(Z_2)_{bk}=\sum_j(A_1)_{bj}(W_2)_{jk}+(b_2)_k$.

- Weights: $\dfrac{\partial L}{\partial(W_2)_{jk}}=\sum_b\dfrac{\partial L}{\partial(Z_2)_{bk}}\dfrac{\partial(Z_2)_{bk}}{\partial(W_2)_{jk}}=\sum_b(A_1)_{bj}(\bar Z_2)_{bk}$, i.e.
$$
\bar W_2=A_1^\top\bar Z_2\qquad(h\times B)(B\times K)=(h\times K).
$$
- Bias: each $(b_2)_k$ appears in every row, so $\bar b_2=\sum_b(\bar Z_2)_{b,:}$, shape $(K)$.
- Input: $\dfrac{\partial L}{\partial(A_1)_{bj}}=\sum_k(\bar Z_2)_{bk}(W_2)_{jk}$, i.e.
$$
\bar A_1=\bar Z_2W_2^\top\qquad(B\times K)(K\times h)=(B\times h).
$$

These are lesson 6's $\nabla_xL=W^\top g$ and $\nabla_WL=gx^\top$ with the batch stacked as rows: the transposes move to the other side, and the sum over the batch in $A_1^\top\bar Z_2$ is the sum of the per-example outer products $g_bx_b^\top$.

**Step 3: ReLU.** Elementwise (lesson 6): $\bar Z_1=\bar A_1\odot\mathbb{1}[Z_1>0]\ \ (B\times h)$.

**Step 4: the first linear layer** (same pattern as step 2):
$$
\bar W_1=X^\top\bar Z_1\ (n\times h),\qquad\bar b_1=\textstyle\sum_b(\bar Z_1)_{b,:}\ (h).
$$
(We would also get $\bar X=\bar Z_1W_1^\top$, needed only if $X$ itself came from another layer.)

**Summary of what to cache from the forward pass:** $X$ (for $\bar W_1$), $Z_1$ (the ReLU mask), $A_1$ (for $\bar W_2$), $P$ (for step 1). This is why training memory scales with depth × batch size.

## Numeric walk-through (one example, scalar output)

To keep numbers tiny: $x=(1,2)$, one ReLU hidden unit, one sigmoid output with binary cross-entropy, true label $y=1$. Parameters $w_1=(0.5,-0.25)$, $b_1=0.1$, $w_2=2$, $b_2=-0.3$.

Forward:

- $z_1=0.5\cdot1+(-0.25)\cdot2+0.1=0.1$, $a_1=\mathrm{ReLU}(0.1)=0.1$
- $z_2=2\cdot0.1-0.3=-0.1$, $p=\sigma(-0.1)=0.47502$
- $L=-\ln0.47502=0.7444$

Backward (every line is one formula):

- $\bar z_2=p-y=-0.52498$
- $\bar w_2=\bar z_2\,a_1=-0.052498$, $\bar b_2=\bar z_2=-0.52498$
- $\bar a_1=\bar z_2\,w_2=-1.04996$
- $\bar z_1=\bar a_1\cdot\mathbb{1}[z_1>0]=-1.04996$ (the unit is active)
- $\bar w_1=\bar z_1\,x=(-1.04996,\,-2.09992)$, $\bar b_1=-1.04996$

A finite-difference check on $w_{1,1}$ with $h=10^{-6}$ gives $-1.04996$ ✓. All gradients are negative, so gradient descent *increases* $w_1,w_2,b$: that raises $z_2$ and therefore $p$ toward the label 1. The model is learning in the right direction.

## Implementation

```python
import numpy as np

def forward_backward(params, X, y):
    W1, b1, W2, b2 = params
    B = len(y)
    Z1 = X @ W1 + b1;  A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z = Z2 - Z2.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(B), y]))

    dZ2 = P.copy(); dZ2[np.arange(B), y] -= 1; dZ2 /= B
    dW2 = A1.T @ dZ2;  db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1;   db1 = dZ1.sum(axis=0)
    return loss, (dW1, db1, dW2, db2)
```

## Debugging checklist (shapes catch most bugs)

1. Every gradient must have exactly the shape of its parameter.
2. Compare with a finite-difference gradient on a tiny network (exercise below). Relative error below $10^{-6}$ is good, above $10^{-3}$ is a bug.
3. Initial loss should be near $\ln K$ (lesson 8).
4. A network that cannot overfit 8 examples has a bug, not a capacity problem.

## Common misconceptions

- **"Backprop is only for neural networks."** It is reverse-mode automatic differentiation (lesson 12) and works on any computation graph.
- **"The weights are updated layer by layer during the backward pass."** No: compute all gradients first, then update. Updating mid-way changes the values later formulas depend on.
- **"Gradients flow through the whole network equally."** They are multiplied by every local derivative on the way; factors below 1 (sigmoid) shrink them (vanishing), factors above 1 grow them (exploding).
- **Forgetting the sum over the batch** in $\bar b$ or the $1/B$ in $\bar Z_2$ produces gradients that are wrong by exactly a factor $B$: a subtle bug that finite differences expose.

## Key takeaways

- Backprop is the chain rule applied from the loss backwards, reusing intermediate gradients.
- Batch formulas: `dW = X^T dZ`, `db = sum_b dZ`, `dX = dZ W^T`.
- Cache `X`, `Z1` (ReLU mask), `A1`, `P` during the forward pass.
- Every gradient must have the shape of its parameter; finite differences settle disputes.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. From the forward equations alone, derive all four gradients of the two-layer network with shapes.
2. Why does forgetting the `1/B` produce gradients that are off by exactly a factor `B`?

## Further reading

- Bishop, *PRML*, Sec. 5.3.1-5.3.2, PDF p.261-265.
- *Mathematics for Machine Learning*, Sec. 5.6, PDF p.145-176.
- Source library: `ai-engineering-from-scratch` phase 03 lesson 03; Raschka, *LLMs from scratch*, appendix A.

## Mastery criteria

On paper, from the forward equations alone, you can produce all four gradient formulas with shapes, and your implementation passes a gradient check with relative error $<10^{-6}$.
