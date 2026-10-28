---
id: gen.02
title: "Autoencoders and VAEs"
summary: From the bottleneck autoencoder to the variational autoencoder - derive the ELBO and the Gaussian KL term, make sampling differentiable with the reparameterisation trick (checked numerically), and understand posterior collapse.
minutes: 75
teaches: [gen.vae]
requires: [gen.taxonomy, info.entropy, nn.backprop, calc.chain]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/02-autoencoders-vae/docs/en.md: autoencoder vs VAE, reparameterisation, loss with the closed-form Gaussian KL, beta-VAE. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 10.1 Variational Inference (PDF p.482-483 read in detail): functionals and the idea of approximating a posterior by an analytic family. READ."
---

## Where this fits

[[gen.01]] ended on a problem: for a latent-variable model $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$ the integral cannot be computed, so we cannot maximise the likelihood directly. The **variational autoencoder** (VAE) solves this by optimising a computable lower bound, using a neural network to approximate the hard posterior. It is also the first stage of every latent-diffusion model ([[gen.05]]): the VAE decoder is what turns a denoised latent back into an image. This lesson builds on backpropagation ([[f2n.10]]), the chain rule and KL divergence ([[math.10]]).

## The idea in plain words

An **autoencoder** is a pair of machines. The encoder squeezes a photo into a short list of numbers (the code); the decoder tries to rebuild the photo from that code. Because the code is much shorter than the photo, the network must keep only what matters (this is a non-linear cousin of PCA). A **denoising** autoencoder is trained to rebuild the clean photo from a corrupted one, which forces it to learn structure rather than copy pixels.

But an ordinary autoencoder is a poor *generator*. Its code space is full of holes: codes of two real cats may sit far apart with garbage in between, and you do not know which codes are legal. The VAE fixes that by making the code **fuzzy and tidy**. The encoder does not output one code but a small cloud (a centre and a spread), the decoder must work from a random point of that cloud, and a penalty keeps all clouds close to a standard round blob. Now the code space is a smooth, filled region, and picking a random point in the round blob and decoding it gives a plausible new image.

## Autoencoders in one paragraph

Encoder $z=f_\phi(x)$ with $x\in\mathbb R^d$, $z\in\mathbb R^k$, $k\ll d$. Decoder $\hat x=g_\theta(z)$. Loss $\|x-\hat x\|^2$ averaged over data. A linear autoencoder with squared loss recovers the PCA subspace (spanned by the top-$k$ eigenvectors; see [[cml.05]]). Non-linear ones bend it. A **denoising** autoencoder feeds $\tilde x=x+\text{noise}$ and still asks for $x$. Remember that this idea returns, much amplified, in diffusion models ([[gen.04]]).

## From a latent-variable model to the ELBO

Model: prior $p(z)=\mathcal N(0,I)$, decoder $p_\theta(x\mid z)$. We want $\log p_\theta(x)=\log\int p_\theta(x\mid z)p(z)\,dz$. Introduce any distribution $q_\phi(z\mid x)$ (the encoder). Multiply and divide inside the integral:

$$
\log p_\theta(x)=\log\int q_\phi(z\mid x)\,\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\,dz
=\log\mathbb E_{q_\phi}\!\left[\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\right].
$$

**Jensen's inequality**: for a concave function such as $\log$, $\log\mathbb E[Y]\ge\mathbb E[\log Y]$. Therefore

$$
\log p_\theta(x)\;\ge\;\mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)=:\mathrm{ELBO}(x).
$$

This is the **evidence lower bound**. To see how loose it is, expand the KL between the encoder and the *true* posterior $p_\theta(z\mid x)$. Using $p_\theta(z\mid x)=p_\theta(x\mid z)p(z)/p_\theta(x)$:

$$
\mathrm{KL}\big(q\,\|\,p_\theta(z\mid x)\big)=\mathbb E_q[\log q-\log p_\theta(x\mid z)-\log p(z)]+\log p_\theta(x)
=\log p_\theta(x)-\mathrm{ELBO}(x).
$$

So $\log p_\theta(x)=\mathrm{ELBO}(x)+\mathrm{KL}(q\|p_\theta(z\mid x))$. Since a KL is non-negative the bound is valid, and it is **tight exactly when the encoder equals the true posterior**. Maximising the ELBO over $\phi$ makes the encoder a better posterior approximation; over $\theta$ it raises the likelihood. This is the variational idea of Bishop Sec. 10.1: replace an intractable distribution by the closest member of a tractable family, where "closest" is measured by a KL divergence (Bishop works with the same decomposition of the log marginal into a bound plus a KL).

### The two terms

- **Reconstruction** $\mathbb E_q[\log p_\theta(x\mid z)]$: if the decoder is Gaussian with fixed variance, $\log p_\theta(x\mid z)=-\frac1{2\sigma^2}\|x-g_\theta(z)\|^2+\text{const}$, i.e. squared error (up to scale).
- **Regulariser** $\mathrm{KL}(q_\phi(z\mid x)\|p(z))$: pulls every code cloud towards the prior.

The loss to *minimise* is $-\mathrm{ELBO}=\text{reconstruction error}+\beta\,\mathrm{KL}$ with $\beta=1$ for the true bound. Choosing $\beta\ne1$ gives the $\beta$-VAE: $\beta<1$ favours sharper reconstructions, $\beta>1$ favours a more regular code space and blurrier output (the trade-off stated in the source lesson).

## The KL term between Gaussians

Take $q=\mathcal N(\mu,\operatorname{diag}(\sigma^2))$ and $p=\mathcal N(0,I)$. Both factorise over the $k$ latent coordinates, so the KL is a sum of 1-D KLs. For one coordinate:

$$
\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1)\big)=\mathbb E_q\!\left[\log q(z)-\log p(z)\right].
$$

Write $\log q(z)=-\tfrac12\log(2\pi\sigma^2)-\frac{(z-\mu)^2}{2\sigma^2}$ and $\log p(z)=-\tfrac12\log(2\pi)-\frac{z^2}{2}$. Under $q$: $\mathbb E[(z-\mu)^2]=\sigma^2$ and $\mathbb E[z^2]=\mu^2+\sigma^2$. Therefore

$$
\mathrm{KL}=-\tfrac12\log\sigma^2-\tfrac12+\tfrac12(\mu^2+\sigma^2)=\tfrac12\big(\mu^2+\sigma^2-\log\sigma^2-1\big).
$$

Summed over coordinates: $\mathrm{KL}=\tfrac12\sum_{i=1}^k(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1)$. It is $0$ only at $\mu=0,\sigma=1$ and grows when a cloud moves away from the origin or becomes too narrow or too wide. The general formula between two 1-D Gaussians is $\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12$, which reduces to the above for $(\mu_2,\sigma_2)=(0,1)$.

In practice the encoder outputs $\log\sigma^2$ (call it `logvar`), not $\sigma$: it is unconstrained and $\sigma=e^{\mathrm{logvar}/2}>0$ automatically.

## The reparameterisation trick

The reconstruction term is an expectation over $z\sim q_\phi(z\mid x)$ and we estimate it with one sample. But *sampling* is not a differentiable operation: if $z$ is drawn at random, what is $\partial z/\partial\phi$? The trick moves the randomness out of the parameters:

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I).
$$

Now $z$ is a deterministic, differentiable function of $(\mu,\sigma)$ given the fixed noise $\varepsilon$, and the chain rule works. For one sample, with decoder loss $L$ and $\delta_z=\partial L/\partial z$:

$$
\frac{\partial L}{\partial\mu}=\delta_z,\qquad
\frac{\partial L}{\partial\,\mathrm{logvar}}=\delta_z\odot\tfrac12\sigma\odot\varepsilon,
$$

because $\partial z/\partial\mathrm{logvar}=\varepsilon\cdot\partial\sigma/\partial\mathrm{logvar}=\varepsilon\cdot\tfrac12 e^{\mathrm{logvar}/2}$. The KL term has its own gradients: $\partial\mathrm{KL}/\partial\mu=\mu$ and $\partial\mathrm{KL}/\partial\mathrm{logvar}=\tfrac12(e^{\mathrm{logvar}}-1)$. The exercise verifies every one of these with finite differences.

## Posterior collapse

If the decoder is very powerful (for example an autoregressive text decoder), it can model $x$ well *without using $z$*. The cheapest way to reduce the loss is then to set $q_\phi(z\mid x)=p(z)$ for every $x$: KL goes to $0$ and the code carries no information. This is **posterior collapse**. Typical remedies: KL annealing (start with $\beta\approx0$ and increase to 1), "free bits" (do not penalise KL below a floor per dimension), a weaker decoder, or word dropout in text. A rising number of "dead" dimensions with KL $\approx0$ is the symptom to monitor.

## Worked example

Take $x=(1,-0.5,2)$, $\mu=(0.3,-0.2)$, $\mathrm{logvar}=(-0.5,0.4)$, noise $\varepsilon=(0.7,-1.1)$, and a linear decoder $\hat x=Wz+b$ with a seeded random $W\in\mathbb R^{3\times2}$, $b=(0.1,0,-0.1)$. (The exercise fixes these.) The sampled code is $z=\mu+e^{\mathrm{logvar}/2}\varepsilon$; the squared reconstruction error is $10.7986$, the KL is $0.1642$ and the total loss $10.9628$. The analytic gradient of the loss with respect to $\mu_1$ is $4.61077$ and the central finite difference agrees to nine digits; for $\mathrm{logvar}_2$ the analytic value is $1.52861$, matching the numeric one.

KL value check: $\mu=(1,0)$, $\sigma^2=(1,4)$ gives $\tfrac12(1+1-0-1)+\tfrac12(0+4-\log4-1)=0.5+0.8069=1.3069$.

## Implementation

```python
import numpy as np

def kl_to_std_normal(mu, logvar):
    return 0.5 * np.sum(np.exp(logvar) + mu**2 - logvar - 1, axis=-1)

def reparameterize(mu, logvar, eps):
    return mu + np.exp(0.5 * logvar) * eps

def vae_loss(x, mu, logvar, eps, W, b, beta=1.0):
    z = reparameterize(mu, logvar, eps)          # (k,)
    xh = W @ z + b                               # (d,)
    rec = np.sum((x - xh) ** 2)
    kl = kl_to_std_normal(mu, logvar)
    return rec + beta * kl, rec, kl
```

Production code (PyTorch) computes the same thing on batches, e.g. `mu, logvar = encoder(x); z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)`, and lets autograd produce all gradients.

## Shapes

For a batch of $B$ points: `x: (B, d)`, `mu, logvar, eps, z: (B, k)`, `x_hat: (B, d)`. The KL is summed over $k$ giving `(B,)`, then averaged over the batch. The reconstruction error is summed over $d$ then averaged. If you average over $d$ instead of summing, you silently change $\beta$ by a factor $d$.

## Common misconceptions

- **"The VAE maximises the likelihood."** It maximises a lower bound; the gap is $\mathrm{KL}(q\|p_\theta(z\mid x))$.
- **"The KL term is a nuisance."** Without it the encoder shrinks $\sigma\to0$ and the model degrades to a plain autoencoder with a holey code space.
- **"Blurry samples mean a bug."** A Gaussian decoder with squared error predicts the *mean* of the plausible images, which is blurry. This is intrinsic to the loss, not a bug.
- **"Use $\sigma$ as the encoder output."** Outputting $\log\sigma^2$ avoids sign constraints and numerical trouble near $\sigma=0$.
- **"The sampling step blocks gradients."** Only if you sample $z$ directly; the reparameterisation exposes the path through $\mu$ and $\sigma$.

## Key takeaways

- $\log p_\theta(x)=\mathrm{ELBO}+\mathrm{KL}(q\|p_\theta(z\mid x))\ge\mathrm{ELBO}$; the bound is tight when the encoder is the true posterior.
- ELBO = reconstruction $-$ KL to the prior; for diagonal Gaussians KL $=\tfrac12\sum(\mu^2+\sigma^2-\log\sigma^2-1)$.
- $z=\mu+\sigma\varepsilon$ makes the sample differentiable; gradients: $\delta_z$ for $\mu$ and $\delta_z\cdot\tfrac12\sigma\varepsilon$ for logvar.
- Strong decoders can ignore $z$ (posterior collapse); anneal $\beta$ or weaken the decoder.

## Explain it yourself

1. Derive the ELBO with Jensen's inequality and explain what the gap measures.
2. Why can we not backpropagate through "draw $z\sim\mathcal N(\mu,\sigma^2)$" and how does $z=\mu+\sigma\varepsilon$ fix it?

## Further reading

- Bishop, PRML, Sec. 10.1 (PDF p.482) for the variational framework; Kingma and Welling's original VAE paper (not read for this lesson).
- Next: [[gen.03]] drops the density and the encoder altogether.

## Mastery criteria

You can derive the ELBO and the Gaussian KL by hand, implement a VAE forward pass and loss, verify the reparameterised gradients numerically, and recognise and mitigate posterior collapse.
