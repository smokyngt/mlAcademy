---
id: math.14
title: Sampling and Monte Carlo
summary: How a computer draws random numbers from a chosen distribution, and how averages of those draws replace integrals that cannot be solved by hand.
minutes: 80
teaches: [prob.sampling, prob.montecarlo]
requires: [prob.basics, prob.rv, calc.derivative]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 11.1 Basic Sampling Algorithms (PDF p.546, 548, 551, 553, 555): transformation method z = h(y), Box-Muller, Cholesky for multivariate Gaussians, rejection sampling and its failure in high dimensions, importance sampling with weights r_l = p/q, self-normalised weights, SIR. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 11.2 MCMC (PDF p.557, 559): random-walk behaviour (distance grows like sqrt(tau)), definition of a first-order Markov chain. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 11.3 Gibbs Sampling (PDF p.563): the coordinate-by-coordinate update scheme. READ."
  - "math/math for ML.pdf, Sec. 6.7 Change of Variables / Inverse Transform (PDF p.221-222, book p.215-216): distribution function technique via the cdf, transformation of random variables. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/16-sampling-methods/docs/en.md: inverse CDF for the exponential, rejection sampling with bound M and acceptance 1/M, importance sampling, Monte Carlo O(1/sqrt N), Metropolis-Hastings algorithm and acceptance rule, Gibbs sampling. READ."
---

## Where this fits

Many questions in machine learning boil down to an average over a distribution we cannot handle analytically: the expected loss of a model over all possible data, the posterior mean of a parameter, the expected return of a game-playing policy, the probability of a rare failure. Written as an integral $\int f(x)\,p(x)\,dx$ these have no closed form once $p$ is anything but textbook-friendly, and grid-based integration dies as the number of dimensions grows. Meanwhile a computer is very good at one thing: producing numbers that look uniformly random between 0 and 1.

This lesson is the bridge between those two facts. First, **how to turn uniform random numbers into draws from the distribution you want** (inverse transform, rejection, importance). Second, **why averaging draws estimates an integral, and how fast** (the $1/\sqrt N$ law). Third, **what to do when you cannot even sample directly** (Markov chain Monte Carlo). You will meet these ideas again in dropout, in reinforcement learning (importance weights in policy gradients), in Bayesian deep learning, in diffusion models and in how language models sample the next word.

It uses probability and random variables (lessons 8 and 9) and the estimation lesson just before (an average of draws is an estimator, with the standard error you saw there).

## The idea in plain words

- **Measuring a lake by throwing stones.** To find the area of an irregular lake inside a square field of known area, throw stones at random into the field and count the fraction that splash. Area of the lake is about that fraction times the field. More stones, better estimate, and you never need a formula for the shore. That is Monte Carlo.
- **The uniform-to-anything machine (inverse transform).** Imagine all values of the quantity you want sorted from smallest to largest, laid along a ruler of length 1 where each stretch has length equal to its probability. Pick a uniform random point on the ruler and read off which value it lands on. Common values own long stretches, so they are picked often. Mathematically the ruler is the cumulative distribution function and reading it off is inverting it.
- **Throw darts at a bigger board (rejection sampling).** If you cannot sample the shape you want, sample from an easy bigger shape that covers it, and discard (reject) every point that falls outside. What stays is uniform inside your shape. Cost: the rejected throws are wasted.
- **Ask the wrong crowd, then re-weight (importance sampling).** Want the opinion of rare experts but only have access to a general crowd? Poll the crowd and multiply each answer by how over- or under-represented that person is relative to the experts. Nothing is thrown away, but if the crowd almost never contains an expert, a few answers carry huge weights and the estimate is shaky.
- **A walker who prefers high ground (MCMC).** Put a hiker in a landscape whose height is the probability. A rule: propose a random step; always accept if it goes uphill, and sometimes accept a downhill step (with probability equal to the height ratio). Long after starting, the fraction of time the hiker spends at each place matches the probability there. You never needed to know the normalising constant, only height ratios.

## Inverse transform sampling

Let $F(x)=P(X\le x)$ be the cumulative distribution function (cdf) of the target. It increases from 0 to 1. If $U\sim\text{Uniform}(0,1)$, then $X=F^{-1}(U)$ has cdf $F$.

**Derivation.** $P(F^{-1}(U)\le x)=P(U\le F(x))$ because $F$ is increasing (apply $F$ to both sides). For a uniform variable $P(U\le c)=c$ for $c\in[0,1]$. So $P(F^{-1}(U)\le x)=F(x)$, which is the cdf we wanted. This is the Deisenroth "distribution function technique" run in reverse, and equals Bishop's Eq. 11.6: transform uniform numbers by the inverse of the indefinite integral of the desired density.

**Example: exponential.** Density $\lambda e^{-\lambda x}$ for $x\ge0$ has cdf $F(x)=1-e^{-\lambda x}$. Set $u=1-e^{-\lambda x}$ and solve: $e^{-\lambda x}=1-u$, so $x=-\ln(1-u)/\lambda$. With $\lambda=2$: $u=0.5$ gives $x=0.3466$ and $u=0.9$ gives $x=1.1513$. Since $1-U$ is also uniform you can write $-\ln(u)/\lambda$.

Limit: you need $F^{-1}$ in closed form. The Gaussian does not have one, which is why Bishop presents Box-Muller (two uniforms to two independent standard normals) and the trick $\mu+L z$ with $\Sigma=LL^\top$ (Cholesky) for a multivariate Gaussian with covariance $\Sigma$: if $z$ has independent unit-variance components, $\mathrm{Cov}(\mu+Lz)=L I L^\top=\Sigma$.

## Rejection sampling

Suppose we can evaluate the target $p(x)$ (even unnormalised) but not sample from it. Pick a proposal $q(x)$ we can sample from and a constant $M$ with $p(x)\le Mq(x)$ everywhere.

1. Draw $x\sim q$ and $u\sim\text{Uniform}(0,1)$.
2. Accept $x$ if $u<\dfrac{p(x)}{Mq(x)}$, otherwise reject and repeat.

**Why it works.** The accepted density at $x$ is proportional to $q(x)\cdot\dfrac{p(x)}{Mq(x)}=\dfrac{p(x)}{M}$, which is proportional to $p(x)$. The overall acceptance probability is $\int q(x)\frac{p(x)}{Mq(x)}dx=\frac1M$ (for normalised $p$), so a tight $M$ is cheap.

**Example.** Target $p(x)=6x(1-x)$ on $[0,1]$ (a Beta(2,2) density, maximum $1.5$ at $x=0.5$), proposal uniform, $M=1.5$. Acceptance rate is $1/1.5=0.667$; a run of 100,000 proposals accepted $66.5\%$ with sample mean $0.4998$ and variance $0.0496$ (exact values $0.5$ and $0.05$).

**High dimensions.** Bishop (book p.531) uses a Gaussian target and a slightly wider Gaussian proposal to show why rejection collapses. Working it out: with isotropic variances $\sigma_p^2<\sigma_q^2$ the smallest valid bound is $M=(\sigma_q/\sigma_p)^D$, which grows exponentially with $D$, so the acceptance rate $1/M$ vanishes (for $\sigma_q/\sigma_p=1.1$ and $D=100$ it is about $7\times10^{-5}$). Rejection is a 1-to-3 dimension tool.

## Importance sampling and Monte Carlo integration

To estimate an expectation we do not actually need samples from $p$. Rewrite

$$
\mathbb E_p[f]=\int f(x)p(x)\,dx=\int f(x)\frac{p(x)}{q(x)}q(x)\,dx\ \approx\ \frac1L\sum_{l=1}^L r_l f(x_l),\qquad x_l\sim q,\ \ r_l=\frac{p(x_l)}{q(x_l)}.
$$

The $r_l$ are the importance weights (Bishop Eq. 11.19). If $p$ is known only up to a constant, use the self-normalised form $\sum_l w_l f(x_l)$ with $w_l=r_l/\sum_m r_m$ (Eq. 11.22-11.23); all samples are kept, unlike rejection. Sampling-importance-resampling (Bishop p.535) then redraws from the weighted set to get approximate samples from $p$.

**Why it can fail.** The estimator is unbiased, but its variance depends on how well $q$ covers $p$. If $q$ puts little mass where $|f|p$ is large, a few huge weights dominate. The standard application is reinforcement learning, where the weight is the ratio of the new policy's to the old policy's probability of an action.

### Monte Carlo integration and the $1/\sqrt N$ error

Plain Monte Carlo is importance sampling with $q=p$: $I=\mathbb E[f(X)]\approx\hat I=\frac1N\sum f(x_i)$. It is unbiased and by the estimation lesson

$$
\operatorname{Var}(\hat I)=\frac{\operatorname{Var}(f(X))}{N},\qquad \text{error}\ \sim\ \frac{\sigma_f}{\sqrt N}.
$$

Two consequences. To gain one extra digit of accuracy you need 100 times more samples. And the rate does not mention the dimension, which is why Monte Carlo wins over grids in high dimensions (a grid with 10 points per axis needs $10^{D}$ evaluations; see the curse of dimensionality in lesson 16).

**Example: estimating $\pi$.** Draw $(x,y)$ uniformly in the unit square; $P(x^2+y^2\le1)=\pi/4$, so $\hat\pi=4\times$ (fraction inside). The count is Bernoulli with $p=\pi/4$, hence the standard deviation of $\hat\pi$ is $4\sqrt{p(1-p)/N}=1.642/\sqrt N$. With seed 1 the estimates are $3.04$ ($N=100$), $3.1564$ ($N=10^4$) and $3.1432$ ($N=10^6$); the predicted error sizes are $0.16$, $0.016$, $0.0016$.

**Example: a rare event.** $P(Z>3)$ for a standard normal is $0.00135$. Naive Monte Carlo with $N=10{,}000$ sees about 13 hits, so each estimate has standard deviation about $3.7\times10^{-4}$ (27% relative). Importance sampling from $q=\mathcal N(3,1)$ with weights $p(x)/q(x)=\exp(-x^2/2+(x-3)^2/2)$ puts half its samples beyond 3; the standard deviation over 200 repeats drops to $2.5\times10^{-5}$, about 15 times smaller, for the same $N$.

## Markov chain Monte Carlo

Rejection and importance sampling degrade in high dimensions. The remedy is to build a **Markov chain**: a sequence $z^{(1)},z^{(2)},\dots$ where each state depends only on the previous one (Bishop Eq. 11.37). The chain is designed so that, in the long run, it visits states in proportion to the target $p$. Samples are correlated (neighbours in the chain are close), but averages over the chain still converge. Bishop notes that an unguided random walk moves only a distance $\propto\sqrt\tau$ after $\tau$ steps, so a good sampler must avoid pure random-walk behaviour.

### Metropolis (and Metropolis-Hastings)

Target $p(z)=\tilde p(z)/Z$ where only $\tilde p$ can be evaluated.

1. From the current $z$ propose $z'\sim q(z'\mid z)$. In the original Metropolis algorithm $q$ is symmetric, e.g. $z'=z+\text{Gaussian noise}$.
2. Accept with probability $\alpha=\min\!\big(1,\ \tilde p(z')/\tilde p(z)\big)$ (for non-symmetric $q$ multiply by $q(z\mid z')/q(z'\mid z)$: Metropolis-Hastings).
3. If accepted, move to $z'$; otherwise stay at $z$ **and count the repeated state again**.

The unknown $Z$ cancels in the ratio, which is why MCMC is the workhorse for Bayesian posteriors known only up to a normaliser. Correctness comes from **detailed balance**: the rule makes the flow $z\to z'$ equal the flow $z'\to z$ under $p$, which implies that $p$ is stationary for the chain. Practical points: discard an initial *burn-in*; the proposal scale must be tuned (too small, and the chain crawls; too large, and nearly everything is rejected); correlated samples carry less information than the same number of independent ones.

**Example.** Target the standard normal ($\log\tilde p=-z^2/2$), proposal step $2.4\times\mathcal N(0,1)$, 20,000 steps from $z=0$ (seed 0), after dropping 1,000 burn-in samples: acceptance rate $0.44$, sample mean $0.0006$, sample variance $0.975$, close to $0$ and $1$.

### Gibbs sampling

If you can sample each variable from its **conditional** given all the others, update them one at a time (Bishop p.543):

$$
z_1\sim p(z_1\mid z_2,z_3,\dots),\quad z_2\sim p(z_2\mid z_1^{\text{new}},z_3,\dots),\ \dots
$$

cycling through all variables, always using the freshest values. It is Metropolis with proposals that are always accepted, so there is no step size to tune. It struggles when variables are strongly correlated: one-coordinate-at-a-time moves cannot cut across a narrow diagonal ridge, so the chain mixes slowly.

## Implementation

```python
import numpy as np

rng = np.random.default_rng(0)

def exp_inverse(u, lam):                    # inverse transform
    return -np.log(1 - u) / lam

def metropolis(logp, n, step, x0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x, out, acc = x0, np.empty(n), 0
    for t in range(n):
        xp = x + step * rng.normal()
        if np.log(rng.random()) < logp(xp) - logp(x):   # work in log space
            x, acc = xp, acc + 1
        out[t] = x
    return out, acc / n
```

The comparison is done in log space so that tiny densities never underflow (lesson 15).

## Shapes

Vectorised sampling: `u: (N,)` uniform draws, `x = f_inverse(u): (N,)`. Rejection with `N` proposals returns a boolean mask `(N,)` and `x[mask]` has length $\approx N/M$. Importance sampling: samples `(N,)` or `(N, D)`, weights `(N,)`, estimate `np.mean(w * f(x))`. A Metropolis chain is `(n,)` or `(n, D)`; multivariate Gaussian draws: `mu + z @ L.T` with `z: (N, D)`, `L: (D, D)`.

## Common misconceptions

- **"Monte Carlo is slow because it needs many samples."** It is slow to get high precision ($1/\sqrt N$) but its rate does not depend on the dimension, which is its great strength.
- **"MCMC samples are independent."** They are correlated; the effective sample size is smaller than the chain length.
- **"Rejected proposals are discarded in Metropolis."** The chain stays put and repeats the old state; dropping it biases the result.
- **"More samples from a bad $q$ fix importance sampling."** If $q$ misses the region that matters, weights have enormous or infinite variance and more samples help very slowly.
- **"Pseudo-random numbers are truly random."** They are deterministic sequences that pass statistical tests (Bishop p.526); that is why fixing a seed reproduces results.

## Key takeaways

- Inverse transform: $X=F^{-1}(U)$. Rejection: accept with probability $p/(Mq)$, rate $1/M$. Importance: weight by $p/q$.
- Monte Carlo error $\approx\sigma_f/\sqrt N$ whatever the dimension.
- MCMC (Metropolis: accept with $\min(1,\tilde p(z')/\tilde p(z))$) needs no normalising constant; Gibbs samples one conditional at a time.
- Rejection and importance sampling break down in high dimensions; MCMC trades independence for scalability.

## Explain it yourself

1. Explain to a friend why throwing random darts at a square can estimate $\pi$, and why 100 times more darts only gives one more decimal digit.
2. Why does the Metropolis algorithm not need the normalising constant of the target, and what happens if you set the proposal step far too large?

## Further reading

- Bishop, *Pattern Recognition and Machine Learning*, Ch. 11 (including slice sampling and hybrid Monte Carlo, which this lesson skips).
- Deisenroth et al., Sec. 6.7 for the change-of-variables view of transforming random variables.
- Repo phase 01, lesson 16, for temperature, top-k, top-p sampling, the reparameterisation trick used in VAEs and Gumbel-softmax.

## Mastery criteria

- Derive the inverse-transform sampler for the exponential distribution and prove why $F^{-1}(U)$ has cdf $F$.
- Explain acceptance probability $1/M$ and why rejection fails in high dimensions.
- Write the importance-sampling estimator and say when its variance is large.
- Implement Metropolis in log space and name burn-in, step size and autocorrelation as its pitfalls.
