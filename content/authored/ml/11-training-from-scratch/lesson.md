---
id: f2n.11
title: Training a network from scratch
summary: Initialisation, the training loop, mini-batches, momentum, and a systematic way to debug a network that will not learn.
minutes: 70
teaches: [nn.training]
requires: [nn.backprop, opt.gradient-descent]
sources:
  - ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/10-mini-framework/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/docs/en.md
  - math/math for ML.pdf (Sec. 7.1 gradient descent with momentum and SGD, PDF p.231)
---

## Learning objectives

- Assemble forward, loss, backward and update into a training loop.
- Initialise weights with He scaling and say why.
- Explain mini-batch SGD and momentum with their update equations.
- Apply the standard debugging ladder: initial loss, gradient check, overfit one batch, learning-rate sweep.
- Train a two-layer network to separate two interleaved half-moons.

## Prerequisites

Backpropagation ([[f2n.10]]) and gradient descent ([[f2n.07]]).

## Intuition

Training is just gradient descent (lesson 7) on the loss of the network (lesson 8) with gradients from backpropagation (lesson 10). The craft is in making it *start well* (initialisation), *run cheaply* (mini-batches), *move efficiently* (momentum, learning rate) and *tell you when something is wrong* (diagnostics).

## The loop

For each epoch: shuffle the data; for each mini-batch of size $b$:

1. **Forward:** compute predictions and the batch loss.
2. **Backward:** compute $\nabla_\theta L$ (lesson 10).
3. **Update:** $\theta\leftarrow\theta-\eta\,\nabla_\theta L$.

Never update before all gradients are computed, and always zero (or recompute) gradients per batch: accumulating stale gradients is the classic PyTorch bug.

## Initialisation (derivation recap)

Lesson 9 showed $\operatorname{Var}(z)=n\sigma_w^2\sigma_x^2$. For ReLU networks use $W_{ij}\sim\mathcal{N}(0,2/n_{\text{in}})$ (He) and $b=0$. Check the effect empirically: push random data through 20 ReLU layers with $\sigma_w^2=1/n$ (activations decay by $2^{-20}$ in second moment) versus $2/n$ (stable).

## Momentum

Plain SGD zig-zags in a narrow valley (lesson 7: large $\kappa$). **Momentum** keeps a running velocity:

$$
v\leftarrow\mu v-\eta\,g,\qquad\theta\leftarrow\theta+v,\qquad\mu\approx0.9 .
$$

Components of the gradient that keep flipping sign cancel in $v$; components that agree accumulate, so the effective step along a consistent direction is about $\eta/(1-\mu)$ ($10\times$ larger for $\mu=0.9$). Adam additionally rescales each parameter by a running estimate of its gradient magnitude; you will meet it in the source library (phase 03, lesson 06).

## A systematic debugging ladder

Work down this list; each step isolates a different class of bug.

1. **Initial loss $\approx\ln K$?** If not, the output scale or loss is wrong.
2. **Gradient check** on a tiny network and batch (lesson 10). Catches backward bugs.
3. **Overfit 8 examples.** With enough capacity, loss must go to ~0 and accuracy to 100%. If not: bug in data/labels, update sign, or learning rate.
4. **Learning-rate sweep**: try $\eta\in\{10^{-3},10^{-2},10^{-1},1\}$ for a few hundred steps and pick the largest that does not diverge.
5. **Only then** train on the full data and watch the *validation* loss. Divergence of train and validation curves means overfitting, not a bug.

Typical symptom → cause: loss stuck at $\ln K$ → $\eta$ far too small, dead ReLUs, or a detached graph; `nan` → $\eta$ too large or log of 0; training loss falls but accuracy flat → label/ordering mismatch between $X$ and $y$ after shuffling.

## Worked example: the half-moons

Two interleaved arcs of 2-D points are not linearly separable, so logistic regression (a single linear boundary) plateaus around 89% on the exercise data. A $2\to16\to2$ ReLU network with He initialisation and $\eta=0.5$ full-batch gradient descent for 400 steps reaches about 98% training accuracy (measured: 98.5%, loss 0.60 to 0.04); the exercise gives you a harness to check that claim and to experiment with other widths, learning rates and momentum.

## Implementation sketch

```python
params = init_params(2, 16, 2, seed=0)
for epoch in range(400):
    loss, grads = forward_backward(params, X, y)
    params = sgd_step(params, grads, lr=0.5)
```

## Common misconceptions

- **"Loss going down means the model works."** Check a held-out set; also check that predictions are not just the majority class.
- **"Bigger learning rate is faster."** Only up to the stability limit; beyond it you diverge.
- **"Random seeds do not matter."** On small networks they can change final accuracy by several points; report mean and spread over seeds (see statistics-for-ML in the source library, phase 01 lesson 15).
- **Mini-batch size changes the effective noise and the best learning rate**; when you scale the batch, retune $\eta$.

## Key takeaways

- Training is forward, backward, update; compute all gradients before updating.
- Debug in order: initial loss, gradient check, overfit 8 examples, learning-rate sweep, then full training.
- Momentum accumulates consistent directions: effective step about `eta/(1-mu)`.
- A network that cannot overfit 8 examples has a bug, not a capacity problem.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Your loss is stuck at `ln K`. List three possible causes and the check that separates them.
2. What would you expect to see in the loss curve for `eta` too small, right, and too large?

## Further reading

- Source library: `ai-engineering-from-scratch` phase 03 lessons 08, 10 and 13.
- *Mathematics for Machine Learning*, Sec. 7.1.

## Mastery criteria

You can start from an empty file and write a working training loop for an MLP, state the five-rung debugging ladder, and predict what happens to the loss for $\eta$ too small, right, and too large.
