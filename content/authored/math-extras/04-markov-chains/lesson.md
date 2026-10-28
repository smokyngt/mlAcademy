---
id: mx.04
title: Markov chains and stochastic processes
summary: Systems whose next step depends only on the present, their transition matrices, stationary distributions as eigenvectors, when chains converge, and how this powers MCMC sampling.
minutes: 75
teaches: [mx.markov, mx.stationary]
requires: [prob.basics, prob.rv, la.matrix, la.eigen]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 13.1 Markov Models (PDF p.627-630): first-order chain factorisation, higher-order chains and their K^(M-1)(K-1) parameter count, latent-variable chains as the bridge to HMMs. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md (random walk, Markov chain, MCMC and Langevin sections; section headings for the rest). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/21-graph-theory/docs/en.md (random-walk view used for PageRank). READ."
---

## Where this fits

Language models produce one token at a time, reinforcement-learning agents move from state to state, and diffusion models add noise step by step. All of these are **sequences of random variables where each step depends on what came before**. The simplest tractable such structure is the Markov chain: the future depends on the past only through the present.

Markov chains matter for ML in three concrete ways: (1) they model sequences (n-gram language models, hidden Markov models, Markov decision processes); (2) they are the engine of **Markov chain Monte Carlo** (MCMC), which samples from distributions too complicated to sample directly (Bayesian posteriors); (3) their long-run behaviour is an eigenvector problem, the same one you solved for PageRank in [[mx.03]] and for matrices in [[math.07]]. You need probability ([[math.08]]) and eigenvectors.

## The idea in plain words

- Picture a board game where you move a token according to dice. To predict where the token goes next you only need to know where it is *now*, not how it got there. That forgetting of history is the **Markov property**.
- The rules of the game are a table: for each square, the odds of landing on each other square. Multiply the table by itself and you get the odds after two moves, and so on.
- Play for a very long time and something remarkable often happens: the fraction of time spent on each square settles to a fixed list of numbers that does not depend on where you started. The game "forgets" its starting point. That list is the **stationary distribution**.
- Not every game does this. If the token can be trapped in a corner, or only alternates between two squares on odd and even turns, the fraction of time does not settle in the same way. Games that are free of such traps are called **ergodic**.
- MCMC turns this around: instead of being given the rules and asking for the stationary distribution, we are given the distribution we want and *design rules* whose long-run behaviour reproduces it.

## Definitions

A sequence $X_0,X_1,\dots$ taking values in a finite set of $K$ states is a **Markov chain** if

$$
P(X_{t+1}=j\mid X_t=i,X_{t-1},\dots,X_0)=P(X_{t+1}=j\mid X_t=i)=:P_{ij}.
$$

This is the Markov property; Bishop (Sec. 13.1) states the same thing through the factorisation of the joint probability of $N$ observations, $p(x_1,\dots,x_N)=p(x_1)\prod_{n=2}^Np(x_n\mid x_{n-1})$. When $P_{ij}$ does not depend on $t$ the chain is **homogeneous** (stationary transition rule). The **transition matrix** $P\in[0,1]^{K\times K}$ is **row-stochastic**: every row sums to 1, because from state $i$ you must go somewhere.

*Higher orders.* A second-order chain lets $x_n$ depend on $x_{n-1},x_{n-2}$; an $M$-th order chain with general tables needs $K^{M-1}(K-1)$ parameters (Bishop p.629), exponential in $M$. An n-gram language model is an $(n-1)$-th order chain, which is why plain count tables stop at small $n$ and why neural sequence models replace the table by a function with shared parameters. Adding a hidden (latent) chain, with observations conditioned on it, gives hidden Markov models (Bishop Sec. 13.2, not covered here): the observations themselves then have no finite-order Markov property.

### $n$-step probabilities

Let $\mu_t\in\mathbb{R}^{1\times K}$ be the row vector with $(\mu_t)_i=P(X_t=i)$. By the law of total probability,

$$
(\mu_{t+1})_j=\sum_i(\mu_t)_iP_{ij}\quad\Longleftrightarrow\quad\mu_{t+1}=\mu_tP,\qquad\mu_n=\mu_0P^n .
$$

The entry $(P^n)_{ij}$ is the probability of going from $i$ to $j$ in exactly $n$ steps (same induction as counting walks in [[mx.03]]).

## Stationary distribution

A distribution $\pi$ (row vector, $\pi_i\ge0$, $\sum\pi_i=1$) is **stationary** if

$$
\boxed{\pi P=\pi .}
$$

If the chain starts in $\pi$ it stays in $\pi$ forever. Equivalently, $\pi^\top$ is an eigenvector of $P^\top$ with eigenvalue $1$ (a *left* eigenvector of $P$).

**Why does an eigenvalue 1 exist?** Because rows sum to 1, $P\mathbf 1=\mathbf 1$: the all-ones vector is a right eigenvector with eigenvalue 1. A matrix and its transpose have the same eigenvalues ([[math.07]]), so $P^\top$ also has eigenvalue 1, and a corresponding eigenvector exists. (Showing that it can be chosen non-negative needs the Perron-Frobenius theorem, which we quote.) Moreover no eigenvalue has modulus above 1, because $\|P^n\|_\infty=1$ for all $n$.

**Convergence.** Suppose $P$ is diagonalisable with eigenvalues $1=\lambda_1>|\lambda_2|\ge\dots$ and left eigenvectors $\ell_1=\pi,\ell_2,\dots$. Write $\mu_0=\sum_ic_i\ell_i$ with $c_1=1$ (the coordinate along $\pi$ is 1 because $\mu_0\mathbf 1=1$ and $\ell_i\mathbf 1=0$ for $i\ge2$). Then

$$
\mu_n=\mu_0P^n=\pi+\sum_{i\ge2}c_i\lambda_i^n\ell_i\ \longrightarrow\ \pi,
$$

with error shrinking like $|\lambda_2|^n$. So the **second largest eigenvalue modulus** (SLEM) sets the **mixing speed**: $1-|\lambda_2|$ is the spectral gap, and a chain with $|\lambda_2|$ close to 1 forgets its start slowly. This is the same argument as power iteration for PageRank.

**Ergodicity.** A finite chain is called **irreducible** if every state can reach every other, and **aperiodic** if the return times to a state do not all share a common divisor above 1. Irreducible and aperiodic together are **ergodic** (a finite chain is ergodic iff some power $P^k$ has all entries strictly positive). For an ergodic chain:

1. The stationary distribution $\pi$ is unique and strictly positive.
2. $\mu_n\to\pi$ from any start.
3. **Time averages equal space averages:** the fraction of time spent in state $i$ along one long trajectory tends to $\pi_i$; more generally $\frac1T\sum_{t}f(X_t)\to\sum_i\pi_if(i)$. This is what lets a single simulation estimate expectations.

Counter-examples: the two-state flip $\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$ is periodic (the distribution oscillates forever), and the identity matrix is reducible (each state is its own trap, and every distribution is stationary).

## Random walks and the Poisson process, briefly

A **simple random walk** adds $\pm1$ with equal probability at each step: $S_n=\xi_1+\dots+\xi_n$. It is a Markov chain on the integers. Because the steps are independent with mean 0 and variance 1, $\mathbb{E}[S_n]=0$ and $\operatorname{Var}S_n=n$, so typical distance from the start grows like $\sqrt n$ (the signature of independent noise: SGD noise, Brownian motion, diffusion models). It has no stationary distribution on the integers: it keeps spreading out.

A **Poisson process** with rate $\lambda$ counts random events in continuous time (requests arriving at a server, photons). Its defining features: the waiting times between events are independent exponentials with mean $1/\lambda$, and the number of events in an interval of length $T$ is Poisson-distributed with mean and variance both equal to $\lambda T$. The exponential has the **memoryless** property, $P(\text{wait}>s+t\mid\text{wait}>s)=P(\text{wait}>t)$, which is the continuous-time version of the Markov property. Continuous-time Markov chains are built from such exponential holding times.

## From Markov chains to MCMC

Suppose we want samples from a distribution $\pi$ that we can evaluate only up to a constant (a Bayesian posterior). Build a chain whose stationary distribution is $\pi$ and run it. A sufficient condition is **detailed balance**:

$$
\pi_iP_{ij}=\pi_jP_{ji}\quad\text{for all }i,j.
$$

Sum over $i$: $\sum_i\pi_iP_{ij}=\pi_j\sum_iP_{ji}=\pi_j$, which is exactly $\pi P=\pi$. The **Metropolis-Hastings** recipe: from state $i$ propose $j$ with probability $q_{ij}$ (symmetric: $q_{ij}=q_{ji}$) and accept with probability $\min(1,\pi_j/\pi_i)$. Then for $i\ne j$,

$$
\pi_iP_{ij}=\pi_iq_{ij}\min\!\Big(1,\frac{\pi_j}{\pi_i}\Big)=q_{ij}\min(\pi_i,\pi_j),
$$

which is symmetric in $i,j$, so detailed balance holds. Only the *ratio* $\pi_j/\pi_i$ is needed: unknown normalising constants cancel. Rejected proposals leave the chain where it is (the diagonal entry). In continuous spaces the sums become integrals ([[mx.05]]) and the proposal is, for example, a Gaussian step around the current point; Langevin dynamics adds a gradient drift to the proposal, which links to the noisy gradient methods of generative modelling. MCMC samples are *correlated*; the effective sample size is smaller than the number of steps, with the SLEM-like quantity (autocorrelation time) controlling by how much.

## Worked example

Three-state chain used throughout the exercise code:

$$
P=\begin{pmatrix}0.9&0.1&0\\0.2&0.6&0.2\\0&0.3&0.7\end{pmatrix}.
$$

- Starting in state 0, the distributions are $\mu_1=(0.9,0.1,0)$, $\mu_2=(0.83,0.15,0.02)$, $\mu_5=(0.70125,0.21247,0.08628)$.
- Eigenvalues of $P$: $1,\ 0.8236,\ 0.3764$ (from $\lambda^2-1.2\lambda+0.31=0$ after factoring out $\lambda=1$). The SLEM is $0.8236$.
- Stationary distribution: solve $\pi P=\pi$ with $\sum\pi_i=1$. The equations give $0.1\pi_0=0.2\pi_1$ and $0.2\pi_1=0.3\pi_2$, i.e. $\pi=(6,3,2)/11=(0.5455,0.2727,0.1818)$. Check: $\pi P$ reproduces $\pi$.
- Convergence: the maximum error $\max_i|\mu_n-\pi|_i$ from state 0 is $0.156$ at $n=5$, $0.059$ at $n=10$, $0.0085$ at $n=20$, $1.7\times10^{-4}$ at $n=40$; compare $0.8236^{40}\approx4.4\times10^{-4}$: the same order, as the SLEM predicts.
- Simulation: one trajectory of 40,000 steps (seed 0) spends fractions $(0.559,\ 0.268,\ 0.173)$ of its time in the three states, within 0.03 of $\pi$. Different seeds differ by about this amount because successive states are correlated.
- Metropolis on the target $(0.5,0.3,0.2)$ with a uniform proposal over the three states gives first row $(0.667,\ 0.2,\ 0.133)$; the full matrix has the target as its stationary distribution, with detailed balance.
- A Poisson process with $\lambda=3$ over $T=2$: simulated counts (4000 runs) have mean $\approx6.06$ and variance $\approx6.19$, both near $\lambda T=6$.

## Implementation

```python
import numpy as np

def stationary(P):
    w, V = np.linalg.eig(P.T)            # left eigenvectors of P
    i = np.argmin(np.abs(w - 1))
    v = np.real(V[:, i])
    return v / v.sum()                   # normalise (also fixes the sign)

def simulate_chain(P, start, steps, rng):
    cum = np.cumsum(P, axis=1)
    u = rng.random(steps)
    s, path = start, np.empty(steps, dtype=int)
    for t in range(steps):
        s = min(int(np.searchsorted(cum[s], u[t], side="right")), len(P) - 1)
        path[t] = s
    return path
```

Practice: `np.linalg.matrix_power(P, n)` for exact $n$-step probabilities; for huge sparse chains use iteration $\mu\leftarrow\mu P$. In MCMC use established samplers (Stan, PyMC, emcee, NumPyro) with diagnostics for convergence (trace plots, $\hat R$).

## Shapes

- $P$: $(K,K)$, rows sum to 1. Distribution $\mu$: $(1,K)$ row, multiplied on the *left*: $\mu P$.
- Left eigenvectors of $P$ are right eigenvectors of $P^\top$: `np.linalg.eig(P.T)` returns eigenvectors as *columns* of shape $(K,K)$.
- A trajectory of $T$ steps: integer array of shape $(T,)$; empirical distribution `np.bincount(path, minlength=K)/T`: shape $(K,)$.

## Common misconceptions

- "Markov means independent." Successive states are dependent; what is independent is the future and the past *given the present*.
- "Rows or columns, whichever." Conventions differ between books (column-stochastic matrices act on column vectors). Always check which index sums to 1.
- "Stationary means the chain stops moving." The chain keeps moving; its *distribution* is unchanged.
- "A long run guarantees convergence." Reducible or periodic chains do not converge to a unique limit; MCMC chains with high SLEM can look converged while stuck in one mode.
- "Memoryless means no state." The state can encode a long history (a transformer context window defines a huge state), so the Markov property is a statement about how you define the state.

## Key takeaways

- A Markov chain is a start distribution plus a row-stochastic matrix; $n$ steps ahead is $\mu_0P^n$.
- A stationary distribution solves $\pi P=\pi$: a left eigenvector with eigenvalue 1, which always exists.
- Ergodic chains converge to a unique $\pi$ at rate $|\lambda_2|^n$ and time averages equal $\pi$-averages.
- Detailed balance gives a way to design chains with a chosen stationary distribution; Metropolis-Hastings needs only density ratios.
- Random walks spread like $\sqrt n$; a Poisson process has exponential gaps and Poisson counts.

## Explain it yourself

1. Explain why the long-run fractions of time do not depend on the starting state of an ergodic chain, using the eigenvalue picture.
2. Explain to a colleague why MCMC does not need the normalising constant of the target.

## Further reading

- Bishop, Sec. 13.1-13.2 (Markov models and hidden Markov models), PDF p.627 onward.
- Repo lesson `phases/01-math-foundations/22-stochastic-processes` (Brownian motion, Langevin dynamics, Metropolis-Hastings code).
- Next: [[mx.05]] makes the sums over states into integrals over continuous variables.

## Mastery criteria

- [ ] I can write down a transition matrix from a verbal description and compute $n$-step probabilities.
- [ ] I can find a stationary distribution both by solving linear equations and as an eigenvector.
- [ ] I can say which of irreducibility and aperiodicity fails in a given counter-example.
- [ ] I can show detailed balance for a Metropolis chain.
