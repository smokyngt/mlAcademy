---
id: gen.01
title: "Generative models: taxonomy and the maths they share"
summary: What it means to model a data distribution, why discriminative and generative models differ, the families of generative model, and the small set of quantities (likelihood, KL divergence, sampling) that all of them are judged by.
minutes: 60
teaches: [gen.taxonomy]
requires: [prob.rv, info.entropy, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/01-generative-models-taxonomy-history/docs/en.md: the five-family taxonomy (explicit tractable, explicit approximate, implicit, score/continuous-time, token-based) and the history table. READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai (directory listing of lessons 01-19): course map of the phase. READ."
---

## Where this fits

Up to now almost every model in this academy answered a question of the form "given an input $x$, what is the label $y$?". That is **discriminative** modelling. A second family of problems asks for the data itself: write a sentence, draw a face, compose a melody, propose a molecule. A model that can do this is **generative**. This lesson draws the map of the whole generative course: autoencoders and VAEs ([[gen.02]]), GANs ([[gen.03]]), diffusion ([[gen.04]], [[gen.05]]) and flow matching with evaluation ([[gen.06]]). It also sets up the few mathematical objects they all share, so that each later lesson is a variation on a theme you already know. You build directly on random variables and the Gaussian ([[math.09]]) and on entropy and KL divergence ([[math.10]]).

## The idea in plain words

Imagine a huge pile of photographs of cats. A **discriminative** model is a bouncer who looks at a photo and says "cat" or "not cat". A **generative** model is an artist who has studied the pile so well that she can paint a new cat that was never in it. The bouncer only needs to know where the boundary between cats and non-cats lies. The artist must know what cats look like *everywhere*: she needs a full picture of the pile, not just its edge. That is why generation is the harder job.

Different artists work in different ways:

- One **writes the picture one brushstroke at a time**, each stroke depending on the previous ones (autoregressive models; language models do exactly this with words).
- One **compresses every picture to a short secret code and learns to un-compress**, then paints new pictures by inventing new codes (autoencoders, VAEs).
- One **competes against a critic**: she paints, the critic says "fake", she improves, the critic sharpens (GANs). She never writes down what the pile looks like, she only learns to fool.
- One **starts from a sheet of static and cleans it up a little at a time** until a picture appears (diffusion and flow models).

Every artist is a different answer to the same question: how do you represent a distribution in millions of dimensions, and how do you draw from it?

## Setup: a distribution over data

Let $x\in\mathbb{R}^d$ be a data point (an image is a vector of $d$ pixel values). We assume the training set $x_1,\dots,x_N$ was drawn independently from an unknown distribution $p_\text{data}(x)$. A **model** is a distribution $p_\theta(x)$ with parameters $\theta$. A generative model has two jobs:

1. **Density estimation**: make $p_\theta$ close to $p_\text{data}$, and (ideally) be able to evaluate $p_\theta(x)$ for a given $x$.
2. **Sampling**: produce new $x\sim p_\theta$.

Discriminative models learn $p(y\mid x)$. Generative models learn $p(x)$, or the joint $p(x,y)$, which gives both by Bayes' rule:

$$
p(y\mid x)=\frac{p(x\mid y)\,p(y)}{\sum_{y'}p(x\mid y')\,p(y')} .
$$

So a generative model with class-conditional densities $p(x\mid y)$ can *also* classify (this is naive Bayes and Gaussian discriminant analysis). The reverse is not true: knowing only the boundary $p(y\mid x)$ gives you no way to draw an $x$.

## Maximum likelihood is minimising a KL divergence

How do we measure "close"? The usual answer is the **likelihood**. Under independence the probability the model assigns to the data is $\prod_i p_\theta(x_i)$, and we maximise its logarithm (a sum, easier to optimise):

$$
\hat\theta=\arg\max_\theta \frac1N\sum_{i=1}^N\log p_\theta(x_i).
$$

Now the link to information theory. The KL divergence from $p_\text{data}$ to $p_\theta$ is

$$
\mathrm{KL}(p_\text{data}\,\|\,p_\theta)=\mathbb{E}_{x\sim p_\text{data}}\big[\log p_\text{data}(x)-\log p_\theta(x)\big]
=-H(p_\text{data})-\mathbb{E}_{p_\text{data}}[\log p_\theta(x)] .
$$

The entropy $H(p_\text{data})$ does not depend on $\theta$. So minimising the KL is the same as maximising $\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$, and the average over the training sample is a Monte Carlo estimate of that expectation (law of large numbers). Therefore:

$$
\text{maximum likelihood}\;\Longleftrightarrow\;\text{minimise }\mathrm{KL}(p_\text{data}\|p_\theta)\;\Longleftrightarrow\;\text{minimise cross-entropy } H(p_\text{data},p_\theta).
$$

The quantity $-\frac1N\sum_i\log p_\theta(x_i)$ is the **average negative log-likelihood (NLL)**. Reported in bits per dimension, it is the standard score for likelihood models. Two practical warnings. The NLL on training data can always be pushed down by memorising, so report it on **held-out** data. And NLL values only compare models of the same kind of data (continuous densities depend on the units of $x$).

## The taxonomy

The source lesson sorts models into five families by what they model and how they pay for tractability. Here is the same map organised around the two questions a model can answer: *can I compute $p_\theta(x)$?* and *can I sample cheaply?*

| Family | Idea | Density? | Sampling | Examples |
|---|---|---|---|---|
| Explicit, tractable | write $p_\theta(x)$ as a computable product or invertible transform | exact | autoregressive: one step per component (slow) | autoregressive models, language models, normalising flows |
| Explicit, approximate | bound $\log p_\theta(x)$ from below and optimise the bound | lower bound (ELBO) | latent-variable decoder, or iterative denoising | VAE ([[gen.02]]), diffusion ([[gen.04]]) |
| Implicit | learn a sampler $x=G(z)$ with no density, trained against a critic | none | one network pass (fast) | GANs ([[gen.03]]) |
| Score / continuous-time | learn the vector field that moves noise to data | via an ODE, costly | integrate an ODE or SDE | score models, flow matching ([[gen.06]]) |
| Token-based autoregressive | compress data to discrete tokens, then model the token sequence | exact on tokens | one token at a time | image/audio token models |

Two ideas run through the whole table.

**Autoregressive factorisation** uses the chain rule of probability, which is exact:

$$
p_\theta(x)=\prod_{j=1}^{d}p_\theta(x_j\mid x_{<j}).
$$

Each factor is a small classifier or regression, so training is plain cross-entropy and $\log p_\theta(x)$ is exact. The price is that sampling needs $d$ sequential steps.

**Latent variables** assume a hidden cause $z$ (a short code) that generates $x$: $p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz$. Integrating over $z$ is usually intractable, which forces the approximations of the next lessons (VAE) or avoids the density altogether (GAN).

A short history, from the source's table: VAE (2013), GAN (2014), flows such as RealNVP and Glow (2017), DDPM diffusion (2020), latent-diffusion text-to-image systems and ControlNet/LoRA adapters (2022), flow matching in large image models (2023 onward). The order of this course follows that history, because each method was invented to fix a weakness of the previous one: VAEs are blurry, GANs are unstable and drop modes, diffusion is slow, flow matching shortens the path.

## Evaluation overview

A model with a tractable density can be scored by held-out NLL. Implicit models and many modern models cannot, so the field also uses **sample-based** scores: compare statistics of generated samples with statistics of real ones (FID, precision/recall), measure agreement with a prompt, or ask humans. [[gen.06]] derives and implements the main ones. Remember the basic tension: *quality* (are samples realistic?) and *diversity/coverage* (does the model reach all of the data?) are different properties and a single number often hides which one failed.

## Worked example: density, sampling and Bayes on a 1-D mixture

Data: a mixture of two Gaussians, weights $(0.5,0.5)$, means $(-2,2)$, standard deviations $(0.5,0.5)$. Draw 2000 training and 2000 test points with a seeded generator (the exercise reproduces all numbers).

- **Model A**, one Gaussian fitted by maximum likelihood: the MLE is the sample mean and (biased) standard deviation, giving about $\hat\mu\approx-0.015$, $\hat\sigma\approx2.05$. It puts its mass at $0$, where there is almost no data, and its samples would mostly land between the two real modes.
- **Held-out average NLL**: Model A $\approx2.148$ nats; the true mixture $\approx1.437$ nats. The gap of $\approx0.71$ nats is the price of the wrong model family, an estimate of $\mathrm{KL}(p_\text{data}\|p_A)$ because the entropy term is shared.
- **Generative classification**: with class-conditional densities $\mathcal N(\mp2,0.5^2)$ and equal priors, $p(y=2\mid x=0)=0.5$ exactly (by symmetry). With priors $(0.9,0.1)$ and $x=0.5$, the posterior of the second class is $0.997$: the likelihood ratio $e^{8}\approx2981$ overwhelms the $9{:}1$ prior.

## Implementation

```python
import numpy as np

def gauss_logpdf(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - (x - mu)**2/(2*sigma**2)

def gmm_logpdf(x, w, mu, sigma):
    x = np.asarray(x, float)[:, None]                    # (N,1)
    lp = np.log(w) + gauss_logpdf(x, np.asarray(mu), np.asarray(sigma))   # (N,K)
    m = lp.max(1, keepdims=True)                          # log-sum-exp for stability
    return (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))[:, 0]

def gmm_sample(n, w, mu, sigma, rng):
    k = rng.choice(len(w), size=n, p=w)                   # ancestral sampling: pick a component...
    return rng.normal(np.asarray(mu)[k], np.asarray(sigma)[k])   # ...then draw from it
```

`gmm_sample` is the simplest case of **ancestral sampling**: sample the latent (the component), then sample $x$ given it. Every latent-variable model in the next lessons samples this way.

## Shapes

`x: (N,)` for 1-D data, or `(N, d)` for $d$-dimensional data. Mixture log-densities are built in a `(N, K)` array (one column per component) and reduced over `K` with log-sum-exp. Posterior class probabilities have shape `(N, n_classes)` and rows sum to 1.

## Common misconceptions

- **"A generative model must be a neural network."** A histogram, a Gaussian and a mixture are generative models. Deep models just scale the same idea.
- **"Higher likelihood always means better samples."** Likelihood and sample quality can disagree: a model can have good NLL and blurry samples, or sharp samples and terrible NLL. That is why several metrics are reported.
- **"GANs and diffusion models estimate a density."** A GAN has no density at all. Diffusion optimises a bound on it.
- **"Density values are probabilities."** A density can exceed 1 (a narrow Gaussian). Only integrals over regions are probabilities.
- **"Generative classifiers are always worse."** They need more assumptions but often learn faster from little data; with wrong assumptions they lose to discriminative models.

## Key takeaways

- Generative models learn $p(x)$ (or $p(x,y)$) and can sample; discriminative models learn $p(y\mid x)$.
- Maximum likelihood equals minimising $\mathrm{KL}(p_\text{data}\|p_\theta)$ equals minimising cross-entropy.
- Families differ in what they trade away: exact likelihood, cheap sampling, or training stability.
- Always evaluate on held-out data and look at both quality and coverage.

## Explain it yourself

1. Why does minimising the KL divergence from the data distribution to the model reduce to maximising average log-likelihood, even though we never know $p_\text{data}$?
2. Give one advantage and one disadvantage of autoregressive models versus GANs.

## Further reading

- Source lesson: phase 08 lesson 01 of the course library (taxonomy and history).
- Next: [[gen.02]], where the intractable latent integral is bounded below.

## Mastery criteria

You can sort a model into a family, derive "MLE = minimise KL", compute a held-out NLL and sample from a mixture by ancestral sampling, and explain why a classifier cannot be turned into a generator.
