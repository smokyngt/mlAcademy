---
id: gen.03
title: "Generative adversarial networks"
summary: The minimax game between generator and discriminator, the optimal discriminator and the Jensen-Shannon divergence derived, why training is unstable, mode collapse, and the Wasserstein fix, with a 1-D toy gradient analysis.
minutes: 75
teaches: [gen.gan]
requires: [gen.taxonomy, ml.logreg, nn.training, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/03-gans-generator-discriminator/docs/en.md: generator/discriminator losses, non-saturating generator loss, failure modes, table of variants (DCGAN, WGAN, spectral norm, StyleGAN). READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai/04-conditional-gans-pix2pix/docs/en.md: conditional GAN, Pix2Pix (U-Net generator, PatchGAN, adversarial + L1 loss), CycleGAN cycle-consistency. READ."
---

## Where this fits

VAEs ([[gen.02]]) need a likelihood bound and give blurry samples. Generative adversarial networks (GANs) take the opposite bet: forget densities, learn only a **sampler**, and train it by making a second network unable to tell its output from real data. They produced the first photorealistic faces (2017-2019) and remain a fast one-pass option; their discriminators still appear as perceptual losses and in one-step distilled diffusion models. They are also the best case study in why training dynamics matter: a mathematically beautiful objective that is hard to optimise. You need logistic regression ([[cml.02]]) and KL divergence ([[math.10]]).

## The idea in plain words

A forger paints fake banknotes. An inspector examines notes and says real or fake. Every time the inspector catches a fake, the forger learns what gave it away and improves; every time the forger improves, the inspector must look for subtler flaws. If this race goes well, the forger ends up so good that the inspector can only guess, and the forger's notes are indistinguishable from real ones. The forger never sees a real banknote directly; all it learns comes through the inspector's verdicts. That is both the power of the idea (no density needed) and its danger: if the inspector has a blind spot, the forger will find it and produce only the one kind of fake that slips through (**mode collapse**).

## The game

Generator $G_\theta:\mathbb R^{k}\to\mathbb R^d$ maps noise $z\sim p_z$ (usually $\mathcal N(0,I)$) to a sample; its output distribution is $p_g$. Discriminator $D_\psi:\mathbb R^d\to(0,1)$ outputs the probability that its input is real. The **minimax** objective is

$$
\min_G\max_D V(D,G)=\mathbb E_{x\sim p_\text{data}}[\log D(x)]+\mathbb E_{x\sim p_g}[\log(1-D(x))].
$$

For fixed $G$, maximising over $D$ is just binary cross-entropy of a classifier that separates real (label 1) from generated (label 0) points, i.e. logistic regression with a neural network.

## Derivation: the optimal discriminator and Jensen-Shannon

Fix $G$. Write $V$ as an integral over $x$ (use that sampling $x=G(z)$ gives $x\sim p_g$):

$$
V(D,G)=\int\big[p_\text{data}(x)\log D(x)+p_g(x)\log(1-D(x))\big]\,dx .
$$

The integrand can be maximised separately at each $x$. For numbers $a=p_\text{data}(x)$, $b=p_g(x)$ the function $f(y)=a\log y+b\log(1-y)$ has $f'(y)=a/y-b/(1-y)=0$, i.e. $y=a/(a+b)$, and $f''<0$ so it is a maximum. Hence

$$
D^*(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)} .
$$

Plug it back. Using $1-D^*=\dfrac{p_g}{p_\text{data}+p_g}$ and the mixture $m=\tfrac12(p_\text{data}+p_g)$:

$$
V(D^*,G)=\mathbb E_{p_\text{data}}\!\left[\log\frac{p_\text{data}}{2m}\right]+\mathbb E_{p_g}\!\left[\log\frac{p_g}{2m}\right]
=-\log4+\mathrm{KL}(p_\text{data}\|m)+\mathrm{KL}(p_g\|m).
$$

The **Jensen-Shannon divergence** is $\mathrm{JS}(p\|q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$ (symmetric, between $0$ and $\log2$). So

$$
\boxed{V(D^*,G)=-\log4+2\,\mathrm{JS}(p_\text{data}\|p_g)} .
$$

Consequences. The generator's best response to a perfect discriminator is to minimise the JS divergence, and the global minimum $-\log4$ is reached exactly when $p_g=p_\text{data}$, where $D^*\equiv\tfrac12$ (the inspector can only guess). The exercise checks this identity numerically on a small discrete example: with $p_\text{data}=(0.1,0.4,0.4,0.1)$ and $p_g=(0.4,0.1,0.1,0.4)$ we get $D^*=(0.2,0.8,0.8,0.2)$, $\mathrm{JS}=0.19274$ and $V(D^*,G)=-1.00080=-\log4+2\cdot0.19274$.

## The practical generator loss

At the start of training the generator is bad and the discriminator confident: $D(G(z))\approx0$. With the minimax generator loss $\log(1-D(G(z)))$, write $D=\sigma(s)$ for a logit $s$. Then $\partial\log(1-\sigma(s))/\partial s=-\sigma(s)$, which is $\approx0$ when $D\approx0$. The gradient **vanishes exactly when the generator most needs it**. The **non-saturating** loss $-\log D(G(z))$ has gradient $-(1-\sigma(s))\to-1$ in that regime. It has the same fixed point but strong gradients early, and is what everyone uses. Numbers from the exercise: at $s=-6$, $D=0.0025$, the saturating gradient is $-0.0025$ and the non-saturating one is $-0.9975$.

Training alternates: one (or a few) discriminator steps, then one generator step. The generator never sees real data: its gradient flows through $D$ into $G$.

## Why training is unstable

- **A moving target.** $G$ is optimising against a $D$ that changes, and vice versa. Gradient descent on a minimax game can circle around the equilibrium instead of converging (in the simplest bilinear game $\min_a\max_b ab$ simultaneous gradient steps spiral outwards).
- **Disjoint supports.** If $p_\text{data}$ and $p_g$ live on different low-dimensional sets (real images lie on a thin manifold), a discriminator can separate them perfectly and $\mathrm{JS}=\log 2$ regardless of how far apart they are. The gradient carries no information about *how to improve*. The exercise shows this: for two point masses at distance $\theta=1,2,5$ the JS is $0.6931$ for all three.
- **Mode collapse.** The generator maps many $z$ to the single output the discriminator currently likes best; the discriminator then penalises that mode, the generator hops to another mode, and so on. Diversity is never recovered because nothing in the loss rewards covering all modes.

## Wasserstein GAN

The Wasserstein-1 distance measures the cost of moving the mass of one distribution onto the other. For two point masses at $0$ and $\theta$ it is $|\theta|$: it *grows smoothly with the distance*, unlike the JS, so the generator gets a useful gradient even with disjoint supports. In 1-D it is simply the mean absolute difference of sorted samples (the exercise does this). The WGAN replaces the probability-output discriminator by a real-valued **critic** $f$ constrained to be 1-Lipschitz and maximises $\mathbb E_{p_\text{data}}f(x)-\mathbb E_{p_g}f(x)$ (Kantorovich-Rubinstein duality). Enforcing the Lipschitz constraint is done by weight clipping (crude), a gradient penalty (WGAN-GP) or spectral normalisation. The source lesson lists these among the stabilisers that made GANs practical, together with DCGAN's conv architecture and progressive growing.

## Conditional GANs and Pix2Pix

Feed a condition $c$ (a class, a sketch) to both networks: $G(c,z)$ and $D(c,x)$. The discriminator then judges whether $x$ is *consistent with* $c$. Pix2Pix (paired image-to-image translation) uses a U-Net generator, a **PatchGAN** discriminator that scores overlapping local patches (realism is local, which needs fewer parameters), and the loss $-\log D(x,G(x))+\lambda\|y-G(x)\|_1$. The L1 term stabilises training and prefers sharp edges over the blur that L2 gives. CycleGAN trains without pairs by adding cycle consistency: translating there and back must reproduce the input.

## Worked example: a 1-D toy GAN gradient step

Real data $x\sim\mathcal N(3,1)$. Generator $G_\theta(z)=\theta+z$ with $z\sim\mathcal N(0,1)$ (it can only shift). Discriminator $D(x)=\sigma(ax+b)$. Discriminator loss $L_D=-\overline{\log\sigma(ax_r+b)}-\overline{\log(1-\sigma(ax_f+b))}$ with gradients

$$
\frac{\partial L_D}{\partial a}=\overline{-(1-D_r)x_r}+\overline{D_fx_f},\qquad
\frac{\partial L_D}{\partial b}=\overline{-(1-D_r)}+\overline{D_f}.
$$

Non-saturating generator loss $L_G=-\overline{\log\sigma(a(\theta+z)+b)}$, so $\partial L_G/\partial\theta=\overline{-(1-D_f)\,a}$. With 64 seeded samples, $\theta=0$, $a=0.5$, $b=-0.5$: $\partial L_D/\partial a=-0.62107$, $\partial L_D/\partial b=0.11934$, $\partial L_G/\partial\theta=-0.30459$, all matching finite differences. The sign of the last one is the point: since $a>0$ the discriminator thinks larger $x$ is more real, so descending $L_G$ increases $\theta$, moving fake samples towards the real mean 3. Running 300 alternating steps (learning rate 0.1, batch 128) from $a=b=\theta=0$ moves $\theta$ to about $3.2$ in the exercise-seed run (it then oscillates around the real mean instead of settling, which is the instability in miniature).

## Implementation

```python
import numpy as np
sigmoid = lambda s: 1 / (1 + np.exp(-s))

def d_loss(a, b, xr, xf):
    return -np.mean(np.log(sigmoid(a*xr + b))) - np.mean(np.log(1 - sigmoid(a*xf + b)))

def g_grad(theta, a, b, z):                    # non-saturating loss
    return np.mean(-(1 - sigmoid(a*(theta + z) + b)) * a)
```

In PyTorch you write the same two losses with `BCEWithLogitsLoss`, call `d_opt.step()` and `g_opt.step()` alternately, and `detach()` the fake batch when updating $D$ so no gradient reaches $G$.

## Shapes

`z: (B, k)` noise, `G(z): (B, d)`, `D(x): (B,)` or `(B,1)`. For PatchGAN `D(x)` is `(B, 1, H', W')`, one score per patch, averaged. The two optimisers hold disjoint parameter sets.

## Common misconceptions

- **"D should be trained to optimality each step."** A perfect $D$ saturates and gives vanishing gradients; in practice the two are kept roughly balanced.
- **"The loss curve shows progress."** GAN losses oscillate and do not track sample quality; inspect samples and use metrics (FID, [[gen.06]]).
- **"WGAN removes all instability."** It improves gradients but the Lipschitz constraint must be enforced well.
- **"Mode collapse means low loss."** The generator can reach a low loss on one mode while ignoring all others; the loss does not measure coverage.

## Key takeaways

- $D^*=p_\text{data}/(p_\text{data}+p_g)$ and $V(D^*,G)=-\log4+2\,\mathrm{JS}(p_\text{data}\|p_g)$.
- Use the non-saturating generator loss; the original one has vanishing gradients.
- JS is flat for disjoint supports; the Wasserstein distance is not, which motivates WGAN.
- Mode collapse and oscillation are properties of the game, not bugs of one implementation.

## Explain it yourself

1. Derive the optimal discriminator and show the value function equals $-\log4+2\,\mathrm{JS}$.
2. Why does a perfect discriminator give no useful gradient when the supports are disjoint, and how does the Wasserstein distance differ?

## Further reading

- Source lessons: phase 08 lessons 03 and 04 (GANs; conditional GANs and Pix2Pix). The original GAN paper by Goodfellow et al. (not read here) contains the same derivation.
- Next: [[gen.04]], a generative model with one stable regression loss.

## Mastery criteria

You can derive $D^*$ and the JS connection, write the non-saturating loss and explain why it is needed, describe mode collapse and the WGAN remedy, and compute the discriminator and generator gradients for a toy 1-D GAN.
