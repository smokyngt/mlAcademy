---
id: gen.04
title: "Diffusion models (DDPM)"
summary: Destroy data with Gaussian noise in a closed-form forward process, learn to undo it one step at a time by predicting the noise, and sample by running the learned reverse chain - derived step by step and tested against an exact analytic denoiser on 1-D Gaussian data.
minutes: 90
teaches: [gen.diffusion]
requires: [gen.vae, prob.rv, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/06-diffusion-ddpm-from-scratch/docs/en.md: forward process with closed form, noise-prediction loss L_simple, reverse step formula, linear beta schedule 1e-4 to 0.02 over T=1000, intuitions (denoising is easy, score matching). READ."
---

## Where this fits

GANs ([[gen.03]]) are fast but unstable; VAEs ([[gen.02]]) are stable but blurry. **Denoising diffusion probabilistic models (DDPM)** keep the stability of a plain regression loss and reach the best sample quality, which is why they power today's image, audio and video generators. The key trick is to split one impossible task ("draw an image from nothing") into a thousand tiny ones ("remove a little noise"). This lesson derives the whole method with Gaussian algebra only, and then tests it end to end on data simple enough that the perfect denoiser can be written down. The next lesson ([[gen.05]]) scales it up; [[gen.06]] shows a straighter alternative.

## The idea in plain words

Take a photograph and add a pinch of static. Do it again, and again, a thousand times. The picture dissolves into pure static, and it does so along a path you control completely. Now play the movie backwards: starting from static, if you had a helper who could look at a slightly noisy picture and say "this much of it is noise", you could remove that noise, step by step, and the static would turn into a picture.

Why is the helper's job feasible? When the image is almost pure noise, guessing a tiny correction is easy: the average image is a good guess. When the image is almost clean, the correction is small and local. The hard middle is handled by one network that is told how noisy the input is. Crucially, we can *manufacture* training data for the helper for free: take a real image, add a known amount of noise, and ask the network to name the noise. No competition, no latent codes, just regression.

## Forward process

Choose a **noise schedule** $\beta_1,\dots,\beta_T\in(0,1)$ (small numbers). Starting from data $x_0$, each step shrinks the signal a little and adds fresh Gaussian noise:

$$
q(x_t\mid x_{t-1})=\mathcal N\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\ \beta_tI\big),\qquad x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon_t .
$$

Define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$.

### Closed form for any $t$

Unroll two steps: $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_t$ and $x_{t-1}=\sqrt{\alpha_{t-1}}\,x_{t-2}+\sqrt{1-\alpha_{t-1}}\,\varepsilon_{t-1}$. Then

$$
x_t=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\,\varepsilon_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_t .
$$

The two noise terms are independent zero-mean Gaussians, so their sum is Gaussian with variance the sum of the variances: $\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$. The result has the same form as one step with $\alpha_t\alpha_{t-1}$ in place of $\alpha_t$. By induction

$$
\boxed{q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ \ \varepsilon\sim\mathcal N(0,I).}
$$

This is what makes training cheap: to get a noisy example at *any* time $t$ you do not simulate $t$ steps, you jump straight there. With the standard linear schedule $\beta_t$ from $10^{-4}$ to $0.02$ over $T=1000$ steps, $\bar\alpha_{10}=0.998$, $\bar\alpha_{100}=0.897$, $\bar\alpha_{300}=0.396$, and $\bar\alpha_{1000}=4.0\times10^{-5}$: the final $x_T$ is essentially $\mathcal N(0,I)$ (signal weight $\sqrt{\bar\alpha_T}=0.006$). The numbers are computed by the exercise.

## The reverse process and its target

We want to sample by starting at $x_T\sim\mathcal N(0,I)$ and drawing $x_{t-1}\sim p_\theta(x_{t-1}\mid x_t)$, modelled as $\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I)$. What should $\mu_\theta$ be? Bayes' rule gives the **true reverse step when $x_0$ is known**: $q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$, a product of two Gaussians in $x_{t-1}$, hence Gaussian. Completing the square in the exponent gives

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_tI),\quad
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\quad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .
$$

At test time $x_0$ is unknown. But from the closed form $x_0=\dfrac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}$, so $x_0$ is determined by the noise $\varepsilon$ inside $x_t$. Substitute this and simplify (using $1-\bar\alpha_t=\beta_t+\alpha_t(1-\bar\alpha_{t-1})$):

$$
\tilde\mu_t=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\Big).
$$

So the only unknown in the reverse mean is the noise $\varepsilon$. Let a network $\varepsilon_\theta(x_t,t)$ predict it and set $\mu_\theta=\frac1{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\big)$.

## The loss

Training maximises a variational lower bound on $\log p_\theta(x_0)$, exactly as for the VAE ([[gen.02]]) but with the $T$ noisy variables as the latents. The bound splits into one KL per step, each between two Gaussians with the same variance, $\mathrm{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)=\frac1{2\sigma_t^2}\|\tilde\mu_t-\mu_\theta\|^2+\text{const}$. Substituting the two means, the difference is $\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\varepsilon-\varepsilon_\theta)$, so each term is

$$
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\ \|\varepsilon-\varepsilon_\theta(x_t,t)\|^2 .
$$

Dropping the $t$-dependent weights (as Ho et al. found works better) gives the **simple loss**

$$
\boxed{L_\text{simple}=\mathbb E_{x_0,\,t\sim\mathrm{U}\{1..T\},\,\varepsilon}\Big[\big\|\varepsilon-\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\big)\big\|^2\Big]}
$$

Training loop: draw a data point, a random $t$, noise $\varepsilon$; form $x_t$ in one line; regress the noise. One stable loss, no minimax. In real systems $\varepsilon_\theta$ is a U-Net or a transformer taking $(x_t,t)$ with $t$ embedded; the loss is the same.

## Sampling

Start $x_T\sim\mathcal N(0,I)$. For $t=T,\dots,1$:

$$
x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\Big)+\sigma_t z,\qquad z\sim\mathcal N(0,I)\ (z=0\text{ at the last step}),
$$

with $\sigma_t^2=\beta_t$ (or $\tilde\beta_t$). This is **ancestral sampling**. It needs $T$ network evaluations, which is why diffusion is slow compared with a GAN; faster samplers (DDIM, fewer steps, distillation) and flow matching ([[gen.06]]) reduce the cost. Another view: $\varepsilon_\theta$ is, up to a factor $-\sqrt{1-\bar\alpha_t}$, the **score** $\nabla_{x_t}\log q(x_t)$, so each step moves $x_t$ uphill in the log-density of the noisy data plus a noise injection (Langevin-like dynamics).

## Worked example with an exact denoiser

If the data are $x_0\sim\mathcal N(m_0,s_0^2)$ in 1-D, everything is Gaussian and the perfect noise predictor is known in closed form. The marginal is $x_t\sim\mathcal N(\sqrt{\bar\alpha_t}m_0,\ \bar\alpha_ts_0^2+1-\bar\alpha_t)$. The conditional mean of $x_0$ given $x_t$ is the Gaussian regression formula

$$
\mathbb E[x_0\mid x_t]=m_0+\frac{\sqrt{\bar\alpha_t}\,s_0^2}{\bar\alpha_ts_0^2+1-\bar\alpha_t}\big(x_t-\sqrt{\bar\alpha_t}m_0\big),\qquad
\varepsilon^*(x_t,t)=\frac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]}{\sqrt{1-\bar\alpha_t}} .
$$

($\varepsilon^*$ is the minimiser of $L_\text{simple}$: the conditional mean of $\varepsilon$ given $x_t$.) Take $m_0=3,\ s_0=0.5$, the linear schedule with $T=1000$, and the sampling loop above with this $\varepsilon^*$ standing in for a trained network.

- At $t=300$ with $x_t=2.0$: $\mathbb E[x_0|x_t]=3.0249$, $\varepsilon^*=0.1229$.
- Forward check (200k draws at $t=500$): empirical mean $0.8401$ and std $0.9704$ against theory $\sqrt{\bar\alpha}m_0=0.8410$ and $\sqrt{\bar\alpha s_0^2+1-\bar\alpha}=0.9701$.
- The loss $\mathbb E\|\varepsilon-\hat\varepsilon\|^2$ at $t=300$ is $0.139$ for $\varepsilon^*$, against $1.0$ for the "predict zero" baseline and $0.144$ for predicting the standardised $x_t$ without using the structure of the data.
- Sampling 20,000 points from $\mathcal N(0,1)$ noise: mean $3.004$, std $0.4986$, i.e. recovering $\mathcal N(3,0.5^2)$ to two decimals. With this tiny model of the data the diffusion machinery works exactly.

## Implementation

```python
import numpy as np

def linear_betas(T, b0=1e-4, b1=0.02): return np.linspace(b0, b1, T)
def alpha_bars(betas): return np.cumprod(1 - betas)

def q_sample(x0, t, abar, eps):                       # jump straight to step t
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps

def sample(n, betas, eps_fn, rng):
    al = 1 - betas; ab = alpha_bars(betas)
    x = rng.normal(size=n)                            # x_T
    for t in range(len(betas) - 1, -1, -1):
        e = eps_fn(x, t)
        x = (x - betas[t] / np.sqrt(1 - ab[t]) * e) / np.sqrt(al[t])
        if t > 0: x = x + np.sqrt(betas[t]) * rng.normal(size=n)
    return x
```

(Arrays are zero-indexed: index $t-1$ in code is step $t$ in the maths.) With PyTorch the network is `eps_model(x_t, t)` and the training step is `loss = ((eps - eps_model(x_t, t))**2).mean()`.

## Shapes

For images: `x0, eps, x_t: (B, C, H, W)`; `t: (B,)` integer step per example; `abar[t]` is gathered to `(B, 1, 1, 1)` so it broadcasts. The network output has the same shape as `x_t`.

## Common misconceptions

- **"The network removes noise in one shot."** It predicts the noise at every level; sampling still needs many small steps.
- **"The forward process is learned."** It is fixed; only the reverse model is trained.
- **"More steps always give better samples."** Beyond a point they only cost time; many samplers get good results in 20-50 steps.
- **"Predicting the noise is arbitrary."** It is an equivalent re-parameterisation of predicting $x_0$ or the score; it simply gives a well-scaled target at all noise levels.
- **"Sampling noise $\sigma_tz$ is optional."** Dropping it changes the process (a deterministic sampler is a different, valid method, e.g. DDIM, but this formula needs the noise).

## Key takeaways

- $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$, so training examples at any noise level are free.
- The reverse mean depends on the unknown noise: $\tilde\mu_t=\frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon)$.
- Training is regression on the noise; sampling is $T$ small denoising steps from pure noise.
- With a known Gaussian data distribution an exact denoiser exists, and the sampler reproduces the data distribution.

## Explain it yourself

1. Derive $q(x_t\mid x_0)$ by adding independent Gaussians and say why it matters for training speed.
2. Explain why the optimal noise predictor is a conditional mean and why its error is never zero.

## Further reading

- Source lesson: phase 08 lesson 06 (DDPM from scratch). Original DDPM paper by Ho, Jain and Abbeel (not read here).
- Next: [[gen.05]] (latent diffusion, guidance) and [[gen.06]] (flow matching).

## Mastery criteria

You can derive the closed-form forward marginal and the noise-parameterised reverse mean, write the training step and the sampling loop, and verify a sampler against an analytic denoiser.
