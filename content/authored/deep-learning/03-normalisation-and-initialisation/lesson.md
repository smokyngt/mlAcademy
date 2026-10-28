---
id: dl.03
title: Normalisation and initialisation
summary: Variance-preserving initialisation derived from scratch, vanishing and exploding gradients measured, and batch norm, layer norm and RMSNorm derived, implemented and gradient-checked.
minutes: 85
teaches: [dl.init, dl.batchnorm, dl.layernorm]
requires: [nn.backprop, nn.training, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/docs/en.md (symmetry problem, variance propagation, Xavier and He, transformer residual scaling). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/code/main.py (function list: zero/random/Xavier/Kaiming init, 50-layer forward experiment). READ (outline)."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/docs/en.md (batch norm, layer norm, RMSNorm sections: formulas, train/inference split, comparison). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/code/main.py (BatchNorm class with running statistics). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/docs/en.md (dead ReLUs, vanishing and exploding gradients, activation statistics table, gradient flow per layer, gradient checking, overfit-one-batch). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/code/debug_neural_nets.py (NetworkDebugger hooks, gradient_check; function list). READ (outline)."
---

## Where this fits

Take the training loop from [[f2n.11]] and make the network 50 layers deep. Without care, nothing trains: the signal that flows forward either shrinks to zero or blows up to $10^{50}$, and the gradient flowing backward does the same. For two decades this limited networks to a few layers. Two families of fixes turned depth from a liability into the default:

- **Initialisation** chooses the starting weights so that signals neither shrink nor grow from layer to layer (Xavier, He).
- **Normalisation layers** re-standardise activations inside the network, continuously, during training (batch norm for vision, layer norm and RMSNorm for transformers).

This lesson derives both, measures the failures they cure, and has you implement batch-norm forward and backward and prove them correct with a gradient check.

## The idea in plain words

Imagine passing a message down a line of 50 people where each person repeats it at their own volume. If everyone is slightly quieter than the previous speaker, the message is inaudible at the end; slightly louder, and the last people are screaming. Only if each person preserves the volume does the message arrive intact. Initialisation sets every person's volume knob at the start, based on how many people they listen to.

Normalisation is different: it is a sound engineer who, after every few speakers, measures the loudness and brings it back to a standard level. The engineer can re-measure continuously, so the line stays audible even while the speakers keep changing their knobs during training. Three kinds of engineer exist:

- **Batch norm** measures each channel across the whole group of messages in the batch.
- **Layer norm** measures each message across its own channels (independent of the other messages).
- **RMSNorm** is layer norm without bothering to remove the average, just rescaling to a standard size.

## Formal definitions and derivations

### Variance propagation

Consider a layer $z=\sum_{j=1}^{n_{in}}w_jx_j$ with independent zero-mean weights of variance $\mathrm{Var}(w)$, independent of the inputs. Then $\mathbb{E}[z]=0$ and

$$
\mathrm{Var}(z)=n_{in}\,\mathrm{Var}(w)\,\mathbb{E}[x^2].
$$

If the inputs have zero mean, $\mathbb{E}[x^2]=\mathrm{Var}(x)$ and each layer multiplies the variance by $n_{in}\mathrm{Var}(w)$. After $L$ layers the factor is $(n_{in}\mathrm{Var}(w))^L$: any value other than 1 explodes or vanishes exponentially. The aim is therefore $n_{in}\mathrm{Var}(w)=1$ in the linear case.

**ReLU.** For $x=\max(0,z_{prev})$ with $z_{prev}$ symmetric around zero, half of the units are zeroed and $\mathbb{E}[x^2]=\tfrac12\mathrm{Var}(z_{prev})$. Therefore

$$
\mathrm{Var}(z_l)=\tfrac12\,n_{in}\mathrm{Var}(w)\,\mathrm{Var}(z_{l-1}),
$$

and the condition for a constant scale is

$$
\mathrm{Var}(w)=\frac{2}{n_{in}}\qquad\text{(He / Kaiming).}
$$

**Backward direction.** Running the same argument for the gradient through $W^\top$ replaces $n_{in}$ by $n_{out}$. One cannot satisfy both $n_{in}\mathrm{Var}(w)=1$ and $n_{out}\mathrm{Var}(w)=1$ unless $n_{in}=n_{out}$; Xavier (Glorot) takes a compromise,

$$
\mathrm{Var}(w)=\frac{2}{n_{in}+n_{out}},
$$

derived for roughly linear activations (tanh and sigmoid near zero). For ReLU the extra factor 2 (He) is needed, otherwise the second moment halves each layer. For transformers with residual branches $x\leftarrow x+f(x)$ the variances of the branches add up, so the source lesson reports that GPT-2 scales the output weights of each residual layer by $1/\sqrt{2N}$ for $N$ layers; we do not derive that here.

**Symmetry.** All-zero (or all-equal) weights make every unit in a layer compute the same function and receive the same gradient, so they stay identical forever; random initialisation is what breaks the symmetry, and its scale is what the formulas above fix.

### Measured: activation scale through 50 layers

Width 256, batch 512, standard-normal inputs, square layers, no biases, one random draw. The table shows the standard deviation of the activations after layer 1, 5, 10, 25, 50:

| Init and activation | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| ReLU, $\mathcal N(0,1)$ | 9.4 | $1.6\times10^{5}$ | $3.1\times10^{10}$ | $2.2\times10^{26}$ | $5.9\times10^{52}$ |
| ReLU, $\mathcal N(0,0.01^2)$ | 0.093 | $1.6\times10^{-5}$ | $3.1\times10^{-10}$ | $2.1\times10^{-24}$ | $2.9\times10^{-48}$ |
| ReLU, Xavier | 0.59 | 0.16 | 0.026 | $1.4\times10^{-4}$ | $1.4\times10^{-8}$ |
| ReLU, He | 0.83 | 0.92 | 0.89 | 1.07 | 0.92 |
| tanh, $\mathcal N(0,0.01^2)$ | 0.16 | $1.0\times10^{-4}$ | $1.1\times10^{-8}$ | $1.3\times10^{-20}$ | $1.4\times10^{-40}$ |
| tanh, $\mathcal N(0,1)$ | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 |
| tanh, Xavier | 0.63 | 0.32 | 0.23 | 0.14 | 0.093 |

Observations that match the derivation: He keeps ReLU activations at order 1; Xavier on ReLU loses about half the second moment per layer; the two naive scales explode or vanish exponentially. Two details to read carefully: tanh with $\mathcal N(0,1)$ does not "explode" (tanh is bounded) but every unit saturates near $\pm1$, where the derivative is about zero, so gradients die; and tanh with Xavier decays slowly rather than staying exactly constant, because the derivation assumes the linear regime and tanh is only approximately linear.

### Measured: vanishing gradients

A 20-layer sigmoid network (width 128, Xavier-scale weights, batch 256, a dummy loss with unit upstream gradient) has the following Frobenius norm of the weight gradient at layers 1, 5, 10, 15, 20 (counting from the input):

- without normalisation: $3\times10^{-12}$, $7\times10^{-9}$, $9\times10^{-6}$, $1\times10^{-2}$, $15$,
- with batch norm before every sigmoid: $0.58$, $0.57$, $0.52$, $0.46$, $0.37$.

The sigmoid's derivative is at most $0.25$, so each layer multiplies the backward signal by a factor typically well below one; twenty layers compound to a ratio of about $10^{12}$ between the last and the first layer. Batch norm keeps pre-activations in the sigmoid's sensitive range and the gradient profile flat. (A ReLU network with He initialisation showed no such decay in the same measurement, but note that weight-gradient norms also include the forward scale of the layer input, so compare profiles, not absolute values.)

### Diagnostics you should always log

1. **Activation statistics per layer** (mean, std, fraction of exactly-zero ReLU outputs). Healthy: bounded, mean near 0 and std not collapsing or growing; "dead" means all zero; "saturated" means pinned at extremes.
2. **Gradient norm per layer.** Layers whose norm is $10^3$ times smaller than the last layer's are vanishing; $\texttt{nan}$ or growing without bound is exploding. Cures: better initialisation, normalisation, residual connections, gradient clipping (see [[dl.01]]), a lower learning rate.
3. **Dead ReLUs.** A unit with a large negative pre-activation has zero gradient forever. If more than about half the activations are exactly zero, change initialisation, learning rate or activation.
4. **Gradient check** of any hand-written layer ([[f2n.10]]), and the **overfit-one-batch** test from [[f2n.11]].

## Batch normalisation

### Forward

For a mini-batch $x\in\mathbb{R}^{N\times D}$ (rows are examples, columns are features), per feature $j$:

$$
\mu_j=\frac1N\sum_{i}x_{ij},\qquad \sigma_j^2=\frac1N\sum_i(x_{ij}-\mu_j)^2,
$$

$$
\hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}},\qquad y_{ij}=\gamma_j\hat x_{ij}+\beta_j .
$$

The learned $\gamma,\beta\in\mathbb{R}^D$ let the network undo the standardisation if that is what it wants. At evaluation time the batch statistics are replaced by running averages gathered during training, $\mu_{run}\leftarrow(1-m)\mu_{run}+m\mu$, same for the variance, with $m=0.1$ typical. Shapes: input $(N,D)$, statistics $(D)$, output $(N,D)$. For convolutional features $(N,C,H,W)$ the statistics are taken per channel over $N,H,W$.

Because training uses batch statistics, each example's output depends on the other examples in the batch. With tiny batches the statistics are noisy (the source lesson warns about batches below about 32), and with $N=1$ they are meaningless. This is the reason transformers do not use it.

### Backward

Let $\bar y$ be the upstream gradient $(N,D)$, and write $\mathrm{inv}=1/\sqrt{\sigma^2+\varepsilon}$ and $\bar{\hat x}=\bar y\odot\gamma$. The parameter gradients are immediate:

$$
\bar\gamma_j=\sum_i\bar y_{ij}\hat x_{ij},\qquad \bar\beta_j=\sum_i\bar y_{ij}.
$$

For the input, follow the three paths along which $x_{ij}$ affects $\hat x$: directly, through $\mu$ and through $\sigma^2$ (drop the feature index $j$):

$$
\bar{\sigma^2}=\sum_i\bar{\hat x}_i\,(x_i-\mu)\cdot\big(-\tfrac12\big)(\sigma^2+\varepsilon)^{-3/2},
$$

$$
\bar\mu=-\,\mathrm{inv}\sum_i\bar{\hat x}_i\;+\;\bar{\sigma^2}\cdot\frac{-2}{N}\sum_i(x_i-\mu),
$$

$$
\bar x_i=\bar{\hat x}_i\,\mathrm{inv}+\bar{\sigma^2}\,\frac{2(x_i-\mu)}{N}+\frac{\bar\mu}{N}.
$$

The last term in $\bar\mu$ vanishes because $\sum_i(x_i-\mu)=0$. Substituting $\bar{\sigma^2}\cdot\frac{2(x_i-\mu)}{N}=-\frac{\mathrm{inv}}{N}\hat x_i\sum_k\bar{\hat x}_k\hat x_k$ gives the compact form used in code:

$$
\boxed{\;\bar x_i=\frac{\mathrm{inv}}{N}\Big(N\,\bar{\hat x}_i-\sum_k\bar{\hat x}_k-\hat x_i\sum_k\bar{\hat x}_k\hat x_k\Big)\;}
$$

Read it as: take the incoming gradient, remove its mean over the batch, and remove its component along $\hat x$, then rescale. Two consequences are worth checking: the gradient sums to zero over the batch (shifting every input by a constant changes nothing), and it is orthogonal to $\hat x$ (scaling every input changes nothing).

### Layer norm and RMSNorm

Layer norm standardises each example across its own $D$ features:

$$
\mu_i=\frac1D\sum_jx_{ij},\quad \sigma_i^2=\frac1D\sum_j(x_{ij}-\mu_i)^2,\quad y_{ij}=\gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}+\beta_j .
$$

It behaves identically in training and evaluation and for any batch size, including 1. The backward pass is the same boxed formula with the sums taken over $j$ (the feature axis) and $N$ replaced by $D$. In transformers it is applied before (pre-LN, reported more stable to train) or after (post-LN) each attention and feed-forward block.

RMSNorm drops the centring and the shift:

$$
y_{ij}=\gamma_j\,\frac{x_{ij}}{\sqrt{\tfrac1D\sum_kx_{ik}^2+\varepsilon}} .
$$

With $r=(\tfrac1D\sum x^2+\varepsilon)^{-1/2}$ one has $\partial r/\partial x_j=-x_jr^3/D$, so the backward is $\bar x=r\,\bar{\hat x}-x\,\frac{r^3}{D}\sum_k\bar{\hat x}_kx_k$ with $\bar{\hat x}=\bar y\odot\gamma$. The source lesson reports that LLaMA-style models use RMSNorm and that it saves about 10% of the normalisation cost with comparable accuracy; we have not measured that.

Why does normalisation help? The original explanation (reduced "internal covariate shift") is, according to the source lesson, contradicted by later work that points to a smoother loss landscape and gradients that are more predictive. Treat the mechanism as partly open; the empirical benefit (stable gradient profile above, larger usable learning rates) is not in doubt.

## Worked example

Batch of one feature, $x=(1,2,3,4)^\top$, $\gamma=1$, $\beta=0$, $\varepsilon=0$. Then $\mu=2.5$, $\sigma^2=1.25$, $\mathrm{inv}=1/\sqrt{1.25}=0.8944$, and

$$
\hat x=(-1.342,\,-0.447,\,0.447,\,1.342)^\top .
$$

The output has mean $0$ and variance $1$. Now let the upstream gradient be $\bar y=(1,0,0,0)^\top$. Then $\sum\bar{\hat x}=1$ and $\sum\bar{\hat x}\hat x=-1.342$, so for $i=1$:

$$
\bar x_1=\frac{0.8944}{4}\big(4\cdot1-1-(-1.342)(-1.342)\big)=0.2236\cdot(4-1-1.8)=0.2683 ,
$$

and the full gradient is $(0.268,-0.358,-0.089,0.179)^\top$, which sums to zero as predicted. $\bar\gamma=\sum\bar y\hat x=-1.342$ and $\bar\beta=1$. These values are reproduced by the code below.

Gradient check on random data ($N=8$, $D=5$, inputs with mean 2 and std 3): central differences with $h=10^{-6}$ against the backward formulas give maximum relative errors of $1.7\times10^{-9}$ (batch-norm input), $7.6\times10^{-10}$ ($\gamma$) and $1.7\times10^{-10}$ ($\beta$); layer norm gives $1.8\times10^{-9}$, $2.3\times10^{-10}$, $1.0\times10^{-10}$; RMSNorm gives absolute errors below $1.5\times10^{-9}$.

## Implementation

```python
import numpy as np

def bn_forward(x, gamma, beta, eps=1e-5):           # x: (N, D)
    mu, var = x.mean(0), x.var(0)                    # (D,)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return gamma * xh + beta, (xh, inv, gamma)

def bn_backward(dy, cache):
    xh, inv, gamma = cache
    N = dy.shape[0]
    dxh = dy * gamma
    dx = (inv / N) * (N * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0))
    return dx, (dy * xh).sum(0), dy.sum(0)           # dx, dgamma, dbeta

def layer_norm(x, gamma, beta, eps=1e-5):            # normalise each row
    mu, var = x.mean(1, keepdims=True), x.var(1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
```

A debugging helper in the spirit of the source lesson's `NetworkDebugger` records, per layer, `x.std()`, `(x == 0).mean()` and the gradient norm; log them every few hundred steps.

## Common misconceptions

- **"Batch norm is a regulariser, so remove dropout."** Batch norm can regularise slightly through batch noise, but its main role is optimisation; test the combination.
- **"Eval mode only matters for dropout."** Batch norm switches to running statistics too; forgetting $\texttt{model.eval()}$ gives wrong, batch-dependent predictions.
- **"Initialisation only matters at step 0."** It decides whether the first steps have usable gradients; a bad start can be unrecoverable (dead ReLUs, saturated tanh).
- **"Xavier is fine for ReLU."** Measured above: the second moment halves every layer.
- **"Layer norm is batch norm over the other axis, same thing."** Different statistics, different behaviour at batch size 1 and at inference.
- **"The gradient check can use the same data as training."** Use a tiny batch and double precision; a float32 check is too noisy to trust.

## Key takeaways

- $\mathrm{Var}(z)=n_{in}\mathrm{Var}(w)\mathbb{E}[x^2]$ is the whole story of initialisation; ReLU halves $\mathbb{E}[x^2]$, giving $\mathrm{Var}(w)=2/n_{in}$.
- Wrong scale means exponential explosion or vanishing in depth; the table shows $10^{52}$ and $10^{-48}$ after 50 layers.
- Batch norm standardises per feature over the batch with learned $\gamma,\beta$, and its backward is $\frac{\mathrm{inv}}{N}(N\bar{\hat x}-\sum\bar{\hat x}-\hat x\sum\bar{\hat x}\hat x)$.
- Layer norm normalises per example, RMSNorm drops the mean; both are batch-independent, which is why transformers use them.
- Log activation statistics and per-layer gradient norms; they reveal vanishing, exploding and dead units before the loss does.

## Explain it yourself

1. Starting from $z=\sum w_jx_j$, derive the He initialisation and say where the factor 2 comes from.
2. Why does the batch-norm input gradient sum to zero over the batch, and what invariance does that reflect?

## Further reading

- Source library: phase 03 lessons 07 (normalisation sections), 08 and 13.
- Glorot and Bengio (Xavier), He et al. (Kaiming), Ioffe and Szegedy (batch norm): cited by the source lessons; not opened here.

## Mastery criteria

You can derive the variance-preserving initialisation for tanh and ReLU, implement batch-norm forward and backward and confirm them with a gradient check, state when to use batch, layer and RMS normalisation, and read an activation/gradient statistics table to diagnose vanishing or exploding signals.
