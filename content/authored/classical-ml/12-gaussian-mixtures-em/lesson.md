---
id: cml.12
title: Gaussian mixture models and the EM algorithm
summary: Model data as a blend of Gaussians with a hidden cluster label, derive the E-step and M-step, prove that the likelihood never decreases, and see k-means as the hard limit.
minutes: 80
teaches: [ml.gmm, ml.em]
requires: [ml.kmeans, prob.basics, prob.rv, alg.exp-log]
sources:
  - "math for ML.pdf (Deisenroth, Faisal, Ong), Ch. 11 Density Estimation with Gaussian Mixture Models (PDF p.354-374): mixture model, responsibilities, update theorems for means/covariances/weights, EM algorithm, running 1-D example with its numbers, latent-variable view. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 9.2 Mixtures of Gaussians (PDF p.450-459): latent variable z, posterior gamma(z_nk), singularities, identifiability, EM for Gaussian mixtures summary, k-means initialisation. READ."
---

## Where this fits

k-means (lesson 4) gives every point to exactly one cluster. Real data is rarely so tidy. A customer may behave partly like a bargain hunter and partly like a loyal buyer; a measurement from a sensor may come from either of two operating regimes; heights in a group of people are a blend of two populations. In such cases we want two things k-means cannot give: a **probability density** (how likely is a new point, is it an outlier?) and a **soft assignment** (this point is 90% group A and 10% group B).

A **Gaussian mixture model (GMM)** provides both. Its fitting algorithm, **expectation-maximisation (EM)**, is one of the great recurring ideas in machine learning: whenever a model has hidden variables, alternate between *guessing the hidden variables given the parameters* and *refitting the parameters given the guesses*. The same pattern powers hidden Markov models, topic models and parts of variational inference. k-means was the training wheels; this lesson is the bike.

## The idea in plain words

Imagine a factory with two machines that produce bolts of slightly different lengths, and a bin that mixes bolts from both. You measure a bolt and want to know which machine made it. If you knew each machine's typical length and spread, and how many bolts each makes, you could compute for any measured bolt how plausible it is under each machine, and weigh that by how busy the machine is. That plausibility share is the machine's **responsibility** for the bolt.

Conversely, if someone told you exactly which machine made each bolt, estimating each machine's average length and spread would be easy: just average the bolts of that machine.

You know neither. EM breaks the circle by alternating:

1. **E-step (expectation):** with the current guess of the machines, compute for every bolt the responsibility of each machine (a soft, probabilistic assignment).
2. **M-step (maximisation):** treat the responsibilities as fractional membership and re-estimate each machine: its average is the responsibility-weighted average of all bolts, its spread the responsibility-weighted spread, its share the average responsibility.

Repeat. Each round, the fit to the data can only improve. If every responsibility were exactly 0 or 1, this would be precisely k-means.

## The model

A GMM writes the density of $x\in\mathbb R^d$ as a convex combination of $K$ Gaussians:

$$
p(x\mid\theta)=\sum_{k=1}^{K}\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k),\qquad 0\le\pi_k\le1,\quad\sum_k\pi_k=1,
$$

with parameters $\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$. For $K=1$ it is a single Gaussian; with more components it can be multimodal.

### Latent-variable view

Introduce a hidden one-hot vector $z\in\{0,1\}^K$ with $\sum_kz_k=1$ saying which component produced $x$:

$$
p(z_k=1)=\pi_k,\qquad p(x\mid z_k=1)=\mathcal N(x\mid\mu_k,\Sigma_k).
$$

Marginalising out $z$ recovers the mixture: $p(x)=\sum_kp(z_k=1)\,p(x\mid z_k=1)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)$. This also gives a generative recipe (**ancestral sampling**): draw a component $k$ with probability $\pi_k$, then draw $x\sim\mathcal N(\mu_k,\Sigma_k)$, and forget $k$. A "complete" dataset would contain $(x_n,z_n)$; ours is "incomplete": we see only $x_n$.

## Derivation 1: the E-step is Bayes' theorem

Given $\theta$, what is the probability that component $k$ generated the observed $x_n$? By Bayes' theorem,

$$
r_{nk}\equiv p(z_{nk}=1\mid x_n)=\frac{p(z_{nk}=1)\,p(x_n\mid z_{nk}=1)}{\sum_{j=1}^K p(z_{nj}=1)\,p(x_n\mid z_{nj}=1)}=\frac{\pi_k\,\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\,\mathcal N(x_n\mid\mu_j,\Sigma_j)}.
$$

The prior is $\pi_k$, the likelihood is the Gaussian density, and the denominator normalises. $r_n=(r_{n1},\dots,r_{nK})$ is a probability vector, a soft assignment. Define $N_k=\sum_nr_{nk}$, the effective number of points of component $k$; $\sum_kN_k=N$.

### Worked example 1: one responsibility by hand

The running example of Deisenroth's chapter: data $X=\{-3,-2.5,-1,0,2,4,5\}$, three components with initial means $-4,0,8$, variances $1,0.2,3$ and equal weights $1/3$. For $x=-1$:

- component 1: $\tfrac13\mathcal N(-1\mid-4,1)=\tfrac13\cdot0.00443=0.00148$,
- component 2: $\tfrac13\mathcal N(-1\mid0,0.2)=\tfrac13\cdot0.0732=0.0244$,
- component 3: $\tfrac13\mathcal N(-1\mid8,3)\approx 10^{-6}$ (negligible).

Normalising: $r=(0.057,\,0.943,\,0.000)$. The same computation for all seven points gives total responsibilities $N_k=(2.057,\,2.009,\,2.934)$. (The exercise reproduces the whole matrix.)

## Derivation 2: the M-step

For fixed responsibilities we maximise the expected complete-data log-likelihood

$$
Q(\theta)=\sum_{n=1}^N\sum_{k=1}^Kr_{nk}\Bigl[\log\pi_k+\log\mathcal N(x_n\mid\mu_k,\Sigma_k)\Bigr].
$$

Why this function? A complete dataset would have a log-likelihood $\sum_n\log\pi_{z_n}\mathcal N(x_n\mid\mu_{z_n},\Sigma_{z_n})$; we do not know $z_n$, so we replace the indicator $z_{nk}$ by its expectation $r_{nk}$. The (non-)closed-form nature of the original problem comes from the sum inside the logarithm, $\log\sum_k\pi_k\mathcal N(\cdot)$, which does not split by component. In $Q$ the logarithm acts on each Gaussian separately, and the components decouple. (The book obtains the same three formulas by setting the gradient of the log-likelihood $\log p(X\mid\theta)$ directly to zero, and the responsibilities appear automatically from the chain rule.)

**Means.** For one component, $\nabla_{\mu_k}Q=\sum_nr_{nk}\Sigma_k^{-1}(x_n-\mu_k)=0$. Multiply by $\Sigma_k$:

$$
\sum_nr_{nk}x_n=\mu_k\sum_nr_{nk}\;\Longrightarrow\;\mu_k=\frac1{N_k}\sum_{n=1}^Nr_{nk}\,x_n.
$$

The new mean is the average of **all** data points, each weighted by how much component $k$ is responsible for it. With 0/1 responsibilities this is the cluster mean of k-means.

**Variance (1-D check).** With $\Sigma_k=\sigma_k^2$: $\frac{\partial Q}{\partial\sigma_k^2}=\sum_nr_{nk}\bigl(-\frac1{2\sigma_k^2}+\frac{(x_n-\mu_k)^2}{2\sigma_k^4}\bigr)=0$. Multiply by $2\sigma_k^4$: $-N_k\sigma_k^2+\sum_nr_{nk}(x_n-\mu_k)^2=0$, so

$$
\sigma_k^2=\frac1{N_k}\sum_nr_{nk}(x_n-\mu_k)^2 .
$$

**Covariance (general $d$).** The same calculation, using the derivatives of $\log\det\Sigma$ and of $(x-\mu)^\top\Sigma^{-1}(x-\mu)$, gives

$$
\Sigma_k=\frac1{N_k}\sum_nr_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top ,
$$

a weighted sample covariance, computed with the **new** $\mu_k$.

**Weights.** Maximise $\sum_n\sum_kr_{nk}\log\pi_k$ subject to $\sum_k\pi_k=1$ with a Lagrange multiplier: $\frac{\partial}{\partial\pi_k}\bigl[\sum_{n,k}r_{nk}\log\pi_k+\lambda(\sum_k\pi_k-1)\bigr]=\frac{N_k}{\pi_k}+\lambda=0$, so $\pi_k=-N_k/\lambda$. Summing over $k$ gives $1=-N/\lambda$, i.e. $\lambda=-N$ and

$$
\pi_k=\frac{N_k}{N},
$$

the average responsibility.

### Algorithm

1. Initialise $\pi_k,\mu_k,\Sigma_k$.
2. **E-step:** $r_{nk}=\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)/\sum_j\pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)$.
3. **M-step:** $N_k$, $\mu_k$, $\Sigma_k$, $\pi_k$ as above.
4. Compute $\log p(X\mid\theta)=\sum_n\log\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$; stop when it (or the parameters) stops changing.

### Worked example 2: the running example, run

Starting from the initial guess above (log-likelihood $-28.33$), one E-step plus one M-step gives means $-2.70,\,-0.40,\,3.70$, variances $0.14,\,0.44,\,1.53$, weights $0.29,\,0.29,\,0.42$, and the log-likelihood jumps to $-14.41$. Iterating, after about five rounds EM settles at

$$
p(x)=0.29\,\mathcal N(x\mid-2.75,0.06)+0.28\,\mathcal N(x\mid-0.50,0.25)+0.43\,\mathcal N(x\mid3.64,1.63),
$$

with log-likelihood $\approx-13.97$. All of these numbers match the book and are reproduced by the code of the exercise.

## Why the log-likelihood never decreases

Let $\ell(\theta)=\sum_n\log\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$. For any probability vector $q_n=(q_{n1},\dots,q_{nK})$ per data point, multiply and divide by $q_{nk}$ inside the sum and use that $\log$ is concave (Jensen's inequality, $\log\mathbb E[Y]\ge\mathbb E[\log Y]$):

$$
\ell(\theta)=\sum_n\log\sum_kq_{nk}\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{q_{nk}}\;\ge\;\sum_n\sum_kq_{nk}\log\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{q_{nk}}=:F(q,\theta).
$$

Equality holds exactly when the ratio $\pi_k\mathcal N(\cdot)/q_{nk}$ is the same for every $k$, i.e. when $q_{nk}$ is proportional to $\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$: that is, when $q_{nk}=r_{nk}(\theta)$, the posterior from the E-step. So:

- **E-step:** with $\theta^{(t)}$ fixed, choosing $q=r(\theta^{(t)})$ makes the bound touch the likelihood: $F(q,\theta^{(t)})=\ell(\theta^{(t)})$.
- **M-step:** with $q$ fixed, $F(q,\theta)=Q(\theta)-\sum q\log q$, and the last term does not depend on $\theta$. Maximising $Q$ therefore maximises $F$: $F(q,\theta^{(t+1)})\ge F(q,\theta^{(t)})$.

Chain them:

$$
\ell(\theta^{(t+1)})\;\ge\;F(q,\theta^{(t+1)})\;\ge\;F(q,\theta^{(t)})\;=\;\ell(\theta^{(t)}).
$$

So every EM iteration increases (or keeps) the log-likelihood. The sequence is bounded above only if no singularity occurs (next section), and it converges to a **local** maximum or a saddle point of $\ell$, not necessarily the global one. (The book's statement of the property is on PDF p.367; the argument above is the standard Jensen derivation, which the book frames through the expected complete-data log-likelihood $Q$, Eq. 11.73.)

## Relation to k-means

Consider a GMM with equal weights and a shared spherical covariance $\Sigma_k=\sigma^2I$. The E-step responsibilities are

$$
r_{nk}\propto\exp\!\Bigl(-\frac{\|x_n-\mu_k\|^2}{2\sigma^2}\Bigr).
$$

As $\sigma^2\to0$ the closest centre wins overwhelmingly and $r_{nk}\to1$ for the nearest $k$, 0 for the rest: **hard assignment**. The M-step mean is then the plain cluster mean. So k-means is the limit of EM for spherical equal Gaussians with vanishing variance. The two differ in what they offer:

| | k-means | GMM + EM |
|---|---|---|
| assignment | hard | soft (responsibilities) |
| cluster shape | round, equal size | ellipsoids with their own $\Sigma_k$ and size $\pi_k$ |
| output | centres and labels | density $p(x)$, probabilities, can detect outliers |
| cost per iteration | cheaper | more expensive (densities, covariances) |

### Worked example 3: soft vs hard

Two centres $1$ and $11$, equal weights, variance $4$. For $x=5$ the log-ratio of the two weighted densities is $\frac{(5-11)^2-(5-1)^2}{2\cdot4}=\frac{36-16}{8}=2.5$, so $r_1=1/(1+e^{-2.5})=0.924$: k-means would say "cluster 1", EM says "92.4% cluster 1, 7.6% cluster 2". For $x=6$ (the midpoint) $r=(0.5,0.5)$.

## Practical issues

**Singularities.** A component can put its mean on a single data point and shrink its variance: the density at that point is $\mathcal N(x_n\mid x_n,\sigma^2)=1/\sqrt{2\pi\sigma^2}\to\infty$. For $\sigma^2=10^{-12}$ the log-likelihood gains $-\tfrac12\log(2\pi\sigma^2)\approx12.9$ nats from that single point, and without limit as $\sigma^2\to0$, while a second component keeps all other points at finite probability. A single Gaussian cannot do this, because collapsing onto one point sends the likelihood of all other points to zero. The likelihood is therefore *unbounded*, an extreme form of overfitting (Bishop 9.2.1). Remedies: add a small ridge $\varepsilon I$ to every covariance (a variance floor), detect collapsing components and re-initialise them, or use a Bayesian/MAP prior.

**Initialisation and local maxima.** EM finds a local maximum that depends on the start. Run several random starts and keep the best log-likelihood. Bishop's recommendation: run k-means first, then initialise $\mu_k$ with the centres, $\Sigma_k$ with the cluster covariances and $\pi_k$ with the cluster fractions. EM typically needs many more iterations than k-means.

**Identifiability.** Relabelling the components ($K!$ ways) gives the same density, so component "1" has no intrinsic meaning.

**Choosing $K$ and the covariance form.** The log-likelihood always improves with more components, so $K$ cannot be chosen on training likelihood; use held-out likelihood or an information criterion. A full-covariance GMM has $K\bigl(d+\tfrac{d(d+1)}2\bigr)+K-1$ parameters (for $d=2$, $K=3$: $17$); restricting to diagonal or spherical covariances reduces this at the price of flexibility.

**Numerics.** Densities underflow. Work with logs and the log-sum-exp trick: $\log\sum_ke^{a_k}=m+\log\sum_ke^{a_k-m}$ with $m=\max_ka_k$.

## Implementation

```python
import numpy as np

def e_step(x, pi, mu, var):              # x: (N,), pi/mu/var: (K,)
    w = pi * np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    return w / w.sum(axis=1, keepdims=True)           # (N, K)

def m_step(x, r):
    Nk = r.sum(axis=0)                                # (K,)
    mu = (r * x[:, None]).sum(axis=0) / Nk
    var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    return Nk / len(x), mu, var
```

For $d>1$: `mu: (K, d)`, `S: (K, d, d)`, and the M-step covariance is `(r[:, k, None] * D).T @ D / Nk[k]` with `D = X - mu[k]`.

## Shapes

`X: (N, d)`, responsibilities `r: (N, K)` with rows summing to 1, `Nk: (K,)`, `pi: (K,)`, `mu: (K, d)`, `S: (K, d, d)`, log-weights `(N, K)`, history: one scalar per iteration.

## Common misconceptions

- **"EM finds the global optimum."** It increases the likelihood monotonically but only to a local maximum; restart and compare.
- **"Responsibilities are probabilities of the data."** They are posterior probabilities of the hidden component *given* the data and current parameters.
- **"More components always fit better, so choose the largest $K$."** On training data, yes; that is overfitting, and a component can even collapse onto a point.
- **"A GMM is just k-means with probabilities."** It also learns shapes and sizes of clusters and defines a density.
- **"The log-likelihood can be computed from the responsibilities alone."** It needs the densities (the denominator of the E-step, in logs).

## Key takeaways

- A GMM is a latent-variable model: pick a component with probability $\pi_k$, then sample from its Gaussian. Marginalising the component gives the mixture density.
- E-step: responsibilities by Bayes' theorem. M-step: weighted means, weighted covariances, average responsibilities.
- EM increases the log-likelihood at every step (Jensen lower bound touches at the E-step, M-step raises it), converging to a local maximum.
- k-means is EM in the limit of equal spherical components with variance $\to0$.
- Guard against singular covariances, use several or k-means initialisations, work in logs.

## Explain it yourself

1. Why can't we simply set the gradient of $\sum_n\log\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)$ to zero and solve for $\mu_k$ in closed form?
2. Describe in words what happens to the responsibilities and to the means when all the Gaussians become very narrow.

## Further reading

- *Mathematics for Machine Learning*, Ch. 11 (mixture models, EM, latent-variable view).
- Bishop, *PRML*, Sec. 9.2 (mixtures of Gaussians) and Sec. 9.3 (the general EM view, not required here).

## Mastery criteria

You can state the mixture density and its latent-variable form, compute responsibilities and the three M-step formulas by hand and in NumPy for $d=1$ and $d>1$, explain why the likelihood is non-decreasing, describe the singularity problem and how to guard against it, and relate EM to k-means.
