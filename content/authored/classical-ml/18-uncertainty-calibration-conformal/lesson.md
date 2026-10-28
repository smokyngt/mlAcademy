---
id: cml.18
title: "Uncertainty: calibration repair, ensembles, Bayesian predictive variance and conformal prediction"
summary: Separate noise from ignorance, repair overconfident probabilities with temperature scaling, measure epistemic uncertainty with ensembles and Bayesian regression, and wrap any model in prediction intervals whose coverage you can prove.
minutes: 90
teaches: [ml.uncertainty, ml.conformal]
requires: [ml.calibration, ml.generalization, ml.linreg, ml.ridge, prob.rv, stat.testing]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf (Sec. 3.3.2 Predictive distribution, PDF p.176-178; Sec. 3.5.2-3.5.3 evidence and effective number of parameters, PDF p.188-190): the predictive variance as noise plus parameter uncertainty (eq. 3.59) and its shrinkage with data. READ."
  - "math/math for ML.pdf (Sec. 8.2-8.3, PDF p.263-275, the pages on empirical risk, maximum likelihood and negative log-likelihood): the NLL is the training objective that temperature scaling re-minimises on held-out data. READ."
---

Library coverage note: the library read for this lesson covers the Bayesian predictive variance and the NLL. It does **not** cover temperature scaling, deep ensembles or conformal prediction; those parts are written from general knowledge, and everything numeric is verified by the exercise code.

## Where this fits

A model that says "93 % sure" and is right 70 % of the time is dangerous: a doctor, a trading desk or a downstream program will act on that number. [[cml.10]] gave you the *measurement* of this problem (reliability bins, expected calibration error). This lesson gives you the *repairs* and the missing half of the story: **how uncertain is the model because it has seen little data**, and **how can you attach an interval to any prediction with a guarantee**? Calibration, ensembles, Bayesian regression and conformal prediction are the four tools used to put honest error bars on predictions. [[cml.03]] gave the train/validation/test discipline we reuse here.

Prerequisites: expectation, variance and the Gaussian ([[math.09]]); maximum likelihood and the bootstrap ([[math.13]]); ridge regression. The mixture-model viewpoint of [[cml.12]] (data produced by hidden causes plus noise) helps for the "two kinds of randomness" picture but is not required.

## The idea in plain words

Imagine a weather forecaster and a new intern.

- Even the best forecaster cannot say whether it will rain at 3 pm tomorrow: some randomness is in the world itself. Collecting more data does not remove it. This is **aleatoric** uncertainty (Latin *alea*, dice).
- The intern has only seen two weeks of data from one town. Their guess is wobbly *because they know little*. Give them ten years of records and they improve. This is **epistemic** uncertainty (from *episteme*, knowledge).

A good system reports both: wide error bars where the world is noisy, and *additional* width in places where it has not seen data.

Three practical questions follow.

1. If the model outputs probabilities that are systematically too confident, can we fix them with one number? (Yes: temperature scaling.)
2. How do we see the model's ignorance? Train several models on slightly different data and look at where they disagree. Where they agree, the data pins the answer down; where they disagree, nobody knows. (Ensembles; Bayesian regression is the exact version of this idea.)
3. Can we get a promise like "the true value lies in this interval 90 % of the time" without assuming the data are Gaussian or the model is right? (Conformal prediction: yes, with one assumption, that the new point looks like the calibration points.)

## Aleatoric versus epistemic, formally

Suppose $y = f(x) + \varepsilon$ with noise $\varepsilon$ of variance $\sigma^2$. A fitted model $\hat f$ depends on the training sample, so $\hat f(x)$ is itself random. Because the noise of a new observation is independent of the training sample, the variances add:

$$
\operatorname{Var}(y \mid x)=\underbrace{\sigma^2}_{\text{aleatoric}}+\underbrace{\operatorname{Var}\big(\hat f(x)\big)}_{\text{epistemic}} .
$$

The first term is the same however much data you collect; the second shrinks as the sample grows (it is the variance term of the bias-variance decomposition in [[cml.03]]). Caveat: the split is a *modelling* statement. What you call noise depends on which features you included; add a feature and some aleatoric noise becomes explainable.

## Calibration repair

Recall from [[cml.10]]: a classifier is **calibrated** if, among all inputs where it says "p = 0.8", about 80 % are in fact positive. Networks trained with cross-entropy for many epochs tend to be **overconfident**: for example accuracy 68 % while the average top probability is 86 %.

### Temperature scaling

The model produces **logits** $z\in\mathbb{R}^K$ (one real number per class; shape $(N,K)$ for a batch). The softmax turns them into probabilities. Introduce one scalar $T>0$, the **temperature**:

$$
p_k(T)=\frac{e^{z_k/T}}{\sum_j e^{z_j/T}} .
$$

$T>1$ flattens the distribution (less confident), $T<1$ sharpens it, $T=1$ is the original model. Dividing all logits by the same positive number never changes which class has the largest logit, so **accuracy is unchanged**; only confidence changes.

To pick $T$, minimise the negative log-likelihood on a held-out **calibration set** (never the training set, where the model is already overconfident):

$$
\mathrm{NLL}(T)=-\frac1N\sum_{i=1}^{N}\log p_{y_i}(T) .
$$

This is the same NLL objective as maximum likelihood (the sum of $-\log p(y_n\mid x_n)$; math for ML Sec. 8.3), now with a single parameter. One parameter means a **one-dimensional search** suffices. $\mathrm{NLL}(T)$ is smooth and has a single minimum in the practical range, so golden-section search works: shrink an interval $[a,b]$ by the golden ratio $0.618$ at each step, keeping the part where the function is lower. It needs no gradients.

### Isotonic regression, the non-parametric alternative

If one temperature is too rigid, fit a **monotone non-decreasing** function $g$ from raw score to calibrated probability, minimising squared error to the 0/1 outcomes. The pool-adjacent-violators algorithm does it: scan points sorted by score; whenever a block's average outcome is lower than the previous block's, merge the two blocks. The result is a staircase. It is more flexible than temperature scaling but needs more calibration data (hundreds to thousands of points) and can overfit with few. Neither method repairs a model whose *ranking* is bad: calibration changes confidence, not discrimination (AUC is unchanged by any strictly increasing map).

## Epistemic uncertainty by resampling: bootstrap and deep ensembles

**Bootstrap ensemble.** Draw $B$ resamples of the training set with replacement (index array of shape $(N,)$ each). Fit one model per resample. At a test input $x$ you get $B$ predictions $\hat f_b(x)$; their standard deviation estimates $\sqrt{\operatorname{Var}(\hat f(x))}$, the epistemic part. The bootstrap itself is in [[math.13]].

**Deep ensemble.** Train $M$ neural networks from different random initialisations (and shuffled mini-batches) on the *same* data. The loss surface has many different good minima, so the networks make different errors, and their disagreement is a cheap, simple and in practice strong uncertainty estimate. If each member outputs a mean $\mu_m(x)$ and its own noise variance $\sigma_m^2(x)$, the ensemble predictive variance is

$$
\sigma^2(x)=\underbrace{\frac1M\sum_m \sigma_m^2(x)}_{\text{mean aleatoric}}+\underbrace{\frac1M\sum_m\big(\mu_m(x)-\bar\mu(x)\big)^2}_{\text{epistemic: spread of the means}},
\qquad \bar\mu=\frac1M\sum_m\mu_m .
$$

Cost: $M$ times the training and inference compute. Limit: members of one architecture can share a blind spot, so agreement is not proof of knowledge.

## Bayesian linear regression: the exact predictive variance

Here the "ensemble of models" is replaced by a probability distribution over weights, and the spread can be computed in closed form. (Source: Bishop Sec. 3.3.2; the algebra below is written out in my own steps.)

**Model.** Features $\phi(x)\in\mathbb{R}^M$ (for example powers of $x$); design matrix $\Phi\in\mathbb{R}^{N\times M}$; targets $t\in\mathbb{R}^N$.

- Likelihood: $t_n = w^\top\phi(x_n)+\varepsilon_n$, $\varepsilon_n\sim\mathcal N(0,\beta^{-1})$. Here $\beta=1/\sigma^2$ is the **noise precision**.
- Prior on weights: $w\sim\mathcal N(0,\alpha^{-1}I)$, with **prior precision** $\alpha$. (This prior produces ridge regression, as shown below.)

**Posterior.** Up to a constant the log-posterior is

$$
-\tfrac{\beta}{2}\|t-\Phi w\|^2-\tfrac{\alpha}{2}\|w\|^2 .
$$

This is quadratic in $w$, so the posterior is Gaussian. Expand and collect terms in $w$:

$$
-\tfrac12\Big[w^\top(\alpha I+\beta\Phi^\top\Phi)w-2\beta\,w^\top\Phi^\top t\Big]+\text{const}.
$$

The log of $\mathcal N(m_N,S_N)$ is $-\tfrac12[w^\top S_N^{-1}w-2w^\top S_N^{-1}m_N]+\text{const}$. Matching coefficients:

$$
S_N^{-1}=\alpha I+\beta\Phi^\top\Phi,\qquad m_N=\beta S_N\Phi^\top t .
$$

Notice that $m_N$ is exactly the ridge solution with penalty $\lambda=\alpha/\beta$, so ridge regression ([[cml.01]]) is the *mean* of a Bayesian model; the covariance $S_N\in\mathbb{R}^{M\times M}$ is what ridge throws away.

**Predictive distribution.** For a new input with features $\phi=\phi(x)$, the new target is $t^\ast=w^\top\phi+\varepsilon$ with $w$ and $\varepsilon$ independent. A linear function of independent Gaussians is Gaussian, so only the mean and variance are needed:

$$
\mathbb{E}[t^\ast]=m_N^\top\phi,\qquad
\operatorname{Var}(t^\ast)=\operatorname{Var}(w^\top\phi)+\operatorname{Var}(\varepsilon)=\phi^\top S_N\phi+\beta^{-1}.
$$

So

$$
\boxed{\sigma_N^2(x)=\underbrace{\beta^{-1}}_{\text{aleatoric}}+\underbrace{\phi(x)^\top S_N\,\phi(x)}_{\text{epistemic}}}
$$

which is Bishop's equation (3.59). Properties noted there: the epistemic term is smallest near the training inputs; as $N\to\infty$ it tends to zero, leaving pure noise $\beta^{-1}$. Bishop also warns that with localised basis functions the term vanishes away from the basis centres, making the model over-confident when extrapolating; Gaussian processes avoid this.

## Prediction intervals

A **prediction interval** at level $1-\alpha$ is an interval $C(x)$ with $P(y\in C(x))\ge 1-\alpha$. Do not confuse it with a **confidence interval** for the *mean* $f(x)$, which omits the noise term and is narrower. With the Gaussian predictive distribution, the 90 % interval is $\mu\pm1.645\,\sigma_N$. That is correct only if the model, the Gaussian noise and the prior are all right. Conformal prediction removes that requirement.

## Split conformal prediction

**Setup.** Split the data into a *proper training* set (fit any model $\hat f$) and a *calibration* set of $n$ points $(x_i,y_i)$ not used for fitting. Choose a **score** measuring how badly the model does on a point; the simplest is the absolute residual $s_i=|y_i-\hat f(x_i)|$.

**Recipe.** Let $k=\lceil (n+1)(1-\alpha)\rceil$ and let $\hat q$ be the $k$-th smallest calibration score (if $k>n$ set $\hat q=\infty$: not enough data for that level). For a new $x$ output

$$
C(x)=\big[\hat f(x)-\hat q,\ \hat f(x)+\hat q\big].
$$

**Theorem (finite-sample coverage).** If the $n$ calibration points and the new test point are *exchangeable* (any reordering has the same joint distribution; i.i.d. data qualify) and the scores have no ties, then

$$
1-\alpha\ \le\ P\big(y_{n+1}\in C(x_{n+1})\big)\ \le\ 1-\alpha+\frac1{n+1}.
$$

**Proof.** The model $\hat f$ was fitted without these $n+1$ points, so treat it as fixed; the scores $s_1,\dots,s_n,s_{n+1}$ (the last for the test point) are then exchangeable. The event $y_{n+1}\in C(x_{n+1})$ is exactly $s_{n+1}\le\hat q$, i.e. the test score is no larger than the $k$-th smallest of the other $n$ scores. Look at the rank of $s_{n+1}$ among all $n+1$ scores (rank 1 = smallest). By exchangeability and no ties, each of the $n+1$ ranks has probability $\frac1{n+1}$. If the rank is $r\le k$, then fewer than $k$ calibration scores lie below it, so it does not exceed the $k$-th smallest calibration score; if $r>k$, at least $k$ calibration scores lie below it and it does exceed it. Hence

$$
P(s_{n+1}\le\hat q)=\frac{k}{n+1}.
$$

Since $(n+1)(1-\alpha)\le k<(n+1)(1-\alpha)+1$, dividing by $n+1$ gives both bounds. $\square$

Notes. (i) The guarantee is **marginal**: averaged over calibration sets and test points, not "for every $x$". Where the noise is larger the interval may under-cover, where it is smaller it over-covers. Fixes (normalised scores, conformalised quantile regression) exist but are beyond this lesson. (ii) It holds for any model; a bad model only gives *wide* intervals. (iii) The $+1$ is what makes it exact in finite samples; the plain $(1-\alpha)$ quantile of $n$ scores slightly under-covers. (iv) Exchangeability fails for drifting time series, and under covariate shift the guarantee is lost.

## Worked example

All numbers are reproduced by the code in the exercises (seeded).

1. **Temperature scaling.** 3-class problem, $N=2000$, logits equal to $2.5\times$ the true logits (an overconfident model). At $T=1$: NLL $0.952$, ECE (10 bins) $0.181$, accuracy $0.680$, mean top probability $0.855$. The 1-D search returns $T^\ast=2.48$ (close to the $2.5$ we injected), NLL $0.715$, ECE $0.023$, mean top probability $0.689$, matching the accuracy. Accuracy is identical before and after.
2. **Ensemble versus Bayesian.** Data: $y=\sin 3x+0.15\varepsilon$ on two clusters $x\in[-1,-0.2]\cup[0.6,1]$ (40 points each), degree-7 polynomial features. Bootstrap ($B=200$) standard deviation of the prediction at $x=-0.6$ (inside data), $0.2$ (the gap) and $0.8$ (inside data): $0.023,\ 0.119,\ 0.046$ (ridge penalty $10^{-3}$, seed 3). Bayesian epistemic standard deviation $\sqrt{\phi^\top S_N\phi}$ at the same points ($\alpha=1$, $\beta=1/0.15^2$): $0.033,\ 0.097,\ 0.034$; total standard deviation $0.154,\ 0.178,\ 0.154$ against pure noise $0.15$. Both methods say the model is ignorant in the gap, with the same order of magnitude. They are not equal: the bootstrap has no prior, and neither accounts for misspecification.
3. **Conformal coverage.** Heavy-tailed noise ($t_3$ with input-dependent scale), a straight line fitted on 100 training points, $n=100$ calibration points, $\alpha=0.1$, so $k=\lceil 101\cdot 0.9\rceil=91$ and theory gives expected coverage $91/101=0.901$. Over 300 independent trials, each judged on 2000 fresh test points, the mean coverage was $0.904$; single trials ranged from $0.796$ upward with standard deviation $0.030$, which is the spread to expect from one calibration set. With $n=19$: $k=\lceil 20\cdot0.9\rceil=18$, expected coverage exactly $18/20=0.90$, observed $0.904$ over 300 trials.

## Implementation

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)       # (N, K)

def nll(T, z, y):
    p = softmax(z / T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

def golden(f, lo, hi, iters=60):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if f(c) < f(d): b = d
        else:           a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return (a + b) / 2

def blr_posterior(Phi, t, alpha, beta):
    SN = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)   # (M, M)
    return beta * SN @ Phi.T @ t, SN                                        # (M,), (M, M)

def conformal_q(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]
```

In production, scikit-learn offers `CalibratedClassifierCV` (sigmoid and isotonic) and the MAPIE package implements conformal intervals; PyTorch users usually write the short temperature scaler above with L-BFGS. I did not read those libraries' documentation for this lesson, so check their current APIs.

## Shapes

- Logits $z$: $(N,K)$; labels $y$: $(N,)$ integers; $T$: scalar.
- $\Phi$: $(N,M)$; $S_N$: $(M,M)$; $m_N$: $(M,)$; for $P$ query points $\Phi_\ast$: $(P,M)$ and predictive variances $(P,)$ via `einsum('ij,jk,ik->i', Phi_star, SN, Phi_star)`.
- Bootstrap predictions: $(B,P)$; spread = `.std(axis=0)`, shape $(P,)$.
- Calibration scores: $(n,)$; $\hat q$: scalar.

## Common misconceptions

- "Calibrated means accurate." No. A model that always predicts the base rate is calibrated and useless; calibration and discrimination are separate ([[cml.10]]).
- "Temperature scaling changes the predictions." The arg-max class does not change; only confidence does.
- "Fit $T$ on the training set." Overconfidence arises on the training set; fit on held-out data.
- "The ensemble spread is the total uncertainty." It estimates the epistemic part; add noise for a prediction interval.
- "A 90 % conformal interval covers 90 % of points for every $x$." The guarantee is marginal; check conditional coverage separately.
- "Conformal needs a good model." Validity does not; efficiency (narrow intervals) does.
- "Bayesian uncertainty is always right." It is right given the model and prior; under a wrong model it is confidently wrong, which is why a conformal check on top is useful.

## Key takeaways

- Predictive variance = noise (irreducible) + parameter ignorance (shrinks with data).
- Temperature scaling: one parameter, fitted by 1-D minimisation of held-out NLL, accuracy untouched.
- Ensembles estimate epistemic uncertainty from disagreement; Bayesian linear regression gives it exactly: $\sigma_N^2=\beta^{-1}+\phi^\top S_N\phi$.
- Split conformal: use the $\lceil(n+1)(1-\alpha)\rceil$-th smallest calibration score; coverage is at least $1-\alpha$ in finite samples under exchangeability, by a rank-symmetry argument.

## Explain it yourself

1. Why does the Bayesian predictive variance contain two terms, and which one can you reduce by collecting data?
2. Why does conformal prediction need the "+1" in $n+1$, and what would break if the calibration points were also used to fit the model?

## Further reading

- Bishop, *Pattern Recognition and Machine Learning*, Sec. 3.3 and 3.5 (read here) and Ch. 6 on Gaussian processes (not read for this lesson).
- Written from general knowledge, not from the library: temperature scaling (Guo et al., 2017), deep ensembles (Lakshminarayanan et al., 2017), conformal prediction (Vovk et al.; Angelopoulos and Bates' tutorial). Check those papers for exact details.

## Mastery criteria

- Compute NLL, find the best temperature and show ECE falls while accuracy is unchanged.
- Derive $S_N$, $m_N$ and the predictive variance from the log-posterior.
- State the conformal theorem with its assumption, reproduce the rank proof, and verify coverage by simulation.
- Explain with an example why marginal coverage is not conditional coverage.
