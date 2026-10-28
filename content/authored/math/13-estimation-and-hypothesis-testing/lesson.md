---
id: math.13
title: Estimation and hypothesis testing
summary: How to learn parameters from data (maximum likelihood, MAP), how far to trust the answer (bias, variance, confidence intervals, the bootstrap), and how to tell a real model improvement from noise.
minutes: 80
teaches: [stat.estimation, stat.testing]
requires: [prob.basics, prob.rv, ml.generalization, ml.ridge]
sources:
  - "math/math for ML.pdf, Sec. 8.3 Parameter Estimation, 8.3.1 Maximum Likelihood Estimation (PDF p.271-273) and 8.3.2 MAP Estimation (PDF p.275-276): negative log-likelihood, Gaussian likelihood gives least squares (Example 8.5), posterior proportional to likelihood times prior, zero-mean Gaussian prior as regularisation. READ."
  - "math/math for ML.pdf, Sec. 6.4.3 (PDF p.199, book p.193): raw-score variance and its numerical caveat; used here for the variance definition. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.2.4 Gaussian distribution (PDF p.47): ML mean and variance, E[sigma^2_ML] = (N-1)/N sigma^2 and the unbiased correction. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.2.5 Curve fitting re-visited (PDF p.49-50): sum-of-squares from Gaussian noise, Gaussian prior on weights gives the regularised error with lambda = alpha/beta. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/15-statistics-for-ml/docs/en.md: hypothesis tests, p-values, confidence intervals, paired t-test, effect size, multiple comparisons, bootstrap for model comparison. READ."
---

## Where this fits

You trained two classifiers. Model A scores 0.87 on your test set and Model B scores 0.89. Do you ship B? Everything in this lesson exists to answer that question honestly. The test set is a small random sample of all the data the model will ever see, so every number you compute from it (accuracy, loss, a learned weight) is itself a random quantity: run the experiment again with another sample and it changes. **Statistics is the study of how much a number computed from data can be trusted.**

The lesson has two halves. First, *estimation*: how do we turn data into parameter values, and what does "a good estimate" even mean? You will see that the loss functions of lesson 8 (mean squared error, cross-entropy) are not arbitrary: they fall out of maximum likelihood, and L2 regularisation falls out of a Gaussian prior. Second, *testing*: how do we quantify uncertainty (confidence intervals, the bootstrap) and decide whether a difference between two models is real.

This builds on probability and Bayes (lesson 8), random variables and the Gaussian (lesson 9), and the idea of generalisation (the earlier bias-variance material). It feeds lesson 14 (sampling, which is how Bayesian posteriors are explored when no formula exists).

## The idea in plain words

- **An estimator is a recipe; an estimate is one result of the recipe.** Think of measuring the height of a tower by pacing around it. The recipe "average of my pacing results" is the estimator. If you did the pacing again tomorrow you would get a slightly different answer: the estimate is random because the data is random. A good estimator has two virtues: it is not *systematically* off (low **bias**) and it does not jump around much from one data set to the next (low **variance**). A dart player whose darts cluster tightly but away from the bull has low variance and high bias; one whose darts scatter evenly around the bull has high variance and low bias.
- **Maximum likelihood** asks: *among all the worlds I could be living in, which one would have made my observed data least surprising?* You try each candidate parameter value and score it by how probable it makes the data; you keep the winner. It is "the explanation that fits best".
- **MAP (maximum a posteriori)** adds common sense before looking at the data: *"I already believe weights are probably small."* The prior acts like a gentle rubber band pulling the estimate toward zero. With little data the rubber band matters a lot; with a lot of data the data wins. This rubber band is exactly what ridge regularisation is.
- **A confidence interval** is an honest error bar. It says: "if I could redo the whole experiment many times, intervals built this way would capture the true value 95% of the time."
- **A hypothesis test** is a proof by contradiction made probabilistic. Assume the boring explanation (the two models are equally good). Ask: how surprising would the data I saw be if the boring explanation were true? If very surprising, doubt the boring explanation.
- **The bootstrap** answers "how much would my number have changed with a different sample?" by pretending your sample *is* the world and re-drawing from it with replacement, thousands of times.

## Estimators, bias and variance

Let $x_1,\dots,x_N$ be independent draws from a distribution with unknown parameter $\theta$. An **estimator** is a function $\hat\theta = g(x_1,\dots,x_N)$. Because the $x_i$ are random, $\hat\theta$ is a random variable with its own mean and variance.

$$
\operatorname{bias}(\hat\theta)=\mathbb E[\hat\theta]-\theta,\qquad
\operatorname{Var}(\hat\theta)=\mathbb E\big[(\hat\theta-\mathbb E\hat\theta)^2\big].
$$

The expected squared error splits into the two (this is the bias-variance trade-off of the generalisation lesson, one level down):

$$
\mathbb E\big[(\hat\theta-\theta)^2\big]=\operatorname{bias}^2+\operatorname{Var}.
$$

*Why:* write $\hat\theta-\theta=(\hat\theta-\mathbb E\hat\theta)+(\mathbb E\hat\theta-\theta)$ and square. The cross term is $2(\mathbb E\hat\theta-\theta)\,\mathbb E[\hat\theta-\mathbb E\hat\theta]$, and the last expectation is zero by definition of the mean, so only the two squares survive.

### The sample mean and its standard error

For the sample mean $\bar x=\frac1N\sum x_i$ with true mean $\mu$ and variance $\sigma^2$: $\mathbb E[\bar x]=\mu$ (unbiased), and because the draws are independent, $\operatorname{Var}(\bar x)=\sigma^2/N$. The standard deviation of the estimator, $\sigma/\sqrt N$, is called the **standard error**. To halve your error bar you need four times as much data. Keep this $1/\sqrt N$ rule in mind: it returns in the Monte Carlo lesson.

### Why the variance estimate divides by $N-1$

Bishop (Sec. 1.2.4) shows that the maximum likelihood variance $\sigma^2_{ML}=\frac1N\sum(x_n-\mu_{ML})^2$ satisfies $\mathbb E[\sigma^2_{ML}]=\frac{N-1}{N}\sigma^2$: it is biased low. The intuition: the data points are, on average, closer to *their own* sample mean than to the true mean (the sample mean is by construction the point that minimises the squared distances), so the squared deviations from $\bar x$ are too small. Multiplying by $\frac{N}{N-1}$ fixes it:

$$
s^2=\frac1{N-1}\sum_{n=1}^N (x_n-\bar x)^2 .
$$

For $N=5$ the factor is $4/5=0.8$: a simulation of 200,000 data sets of five standard normal numbers gives an average of the divide-by-$N$ estimator of $0.802$ and of the divide-by-$(N-1)$ estimator of $1.002$ (reproduced in the exercise). Note that unbiased does not mean best: an estimator with a little bias and much less variance can have a smaller total error, which is the very reason regularisation helps.

## Maximum likelihood

Model the data by a density $p(x\mid\theta)$. Holding the data fixed and varying $\theta$, $p(x\mid\theta)$ is the **likelihood**. For independent examples the likelihood of the whole data set is a product, and a product of many small numbers is both awkward to optimise and numerically hostile (see lesson 15). Taking logs turns it into a sum. Deisenroth et al. (Sec. 8.3.1) minimise the **negative log-likelihood**:

$$
L(\theta)=-\log p(Y\mid X,\theta)=-\sum_{n=1}^{N}\log p(y_n\mid x_n,\theta),\qquad
\hat\theta_{ML}=\arg\min_\theta L(\theta).
$$

(The minus sign is only a convention so that "maximise likelihood" becomes "minimise a loss".)

### Derivation: Gaussian noise gives least squares

Assume $y_n=x_n^\top\theta+\varepsilon_n$ with noise $\varepsilon_n\sim\mathcal N(0,\sigma^2)$, so $p(y_n\mid x_n,\theta)=\mathcal N(y_n\mid x_n^\top\theta,\sigma^2)$. Insert the Gaussian density:

$$
L(\theta)=-\sum_{n}\log\Big[\tfrac1{\sqrt{2\pi\sigma^2}}\exp\Big(-\tfrac{(y_n-x_n^\top\theta)^2}{2\sigma^2}\Big)\Big].
$$

The log of a product is a sum of logs, and $\log\exp(u)=u$, so each term becomes $\frac{(y_n-x_n^\top\theta)^2}{2\sigma^2}+\log\sqrt{2\pi\sigma^2}$:

$$
L(\theta)=\frac1{2\sigma^2}\sum_{n=1}^{N}(y_n-x_n^\top\theta)^2+N\log\sqrt{2\pi\sigma^2}.
$$

The second term does not depend on $\theta$, and $\frac1{2\sigma^2}$ is a positive constant that cannot move the minimiser. So maximum likelihood with Gaussian noise **is** least squares (the book's Example 8.5 and Bishop's Eq. 1.62). The same recipe for a Bernoulli label gives cross-entropy. This is the answer to "why that loss?": each standard loss is the negative log-likelihood of a noise model.

For a plain Gaussian sample with unknown $\mu$ and $\sigma^2$, setting derivatives to zero gives the sample mean and the divide-by-$N$ variance (Bishop Eq. 1.55-1.56), which is where the bias above came from. Properties (the Deisenroth remark, p.269-270): the MLE is *consistent* (converges to the truth as $N\to\infty$) and its error shrinks like $1/\sqrt N$, but for small $N$ it can overfit.

## MAP and why L2 regularisation is a Gaussian prior

Maximum likelihood trusts the data completely. Bayes' theorem (lesson 8) lets us also encode a prior belief $p(\theta)$:

$$
p(\theta\mid x)=\frac{p(x\mid\theta)\,p(\theta)}{p(x)}\ \propto\ p(x\mid\theta)\,p(\theta).
$$

The denominator does not depend on $\theta$, so for finding the best $\theta$ it can be ignored. The MAP estimate minimises the **negative log-posterior**:

$$
\hat\theta_{MAP}=\arg\min_\theta\ \big[-\log p(x\mid\theta)-\log p(\theta)\big].
$$

### Derivation: Gaussian prior gives ridge

Take the Gaussian-noise likelihood above and the prior $\theta\sim\mathcal N(0,\tau^2 I)$, so $-\log p(\theta)=\frac1{2\tau^2}\|\theta\|^2+\text{const}$. Dropping constants:

$$
-\log p(\theta\mid x)\ =\ \frac1{2\sigma^2}\sum_n (y_n-x_n^\top\theta)^2+\frac1{2\tau^2}\|\theta\|_2^2 .
$$

Multiply by the positive constant $2\sigma^2$ (does not change the minimiser):

$$
\sum_n (y_n-x_n^\top\theta)^2+\lambda\|\theta\|_2^2,\qquad \lambda=\frac{\sigma^2}{\tau^2}.
$$

That is exactly ridge regression. Bishop (Eq. 1.67) reaches the same form with precisions: $\lambda=\alpha/\beta$ where $\alpha=1/\tau^2$ is the prior precision and $\beta=1/\sigma^2$ the noise precision. Reading it in words: **a stronger belief that weights are near zero (small $\tau$) means a larger $\lambda$; noisier data (large $\sigma$) also means a larger $\lambda$**, because you should lean on the prior when the data is unreliable. A Laplace prior would give the L1 penalty instead. Remember, though, that MAP still returns a single point; the fully Bayesian treatment keeps the whole posterior (Deisenroth Sec. 8.4, not covered here).

## Confidence intervals

A point estimate without an error bar is a half-truth. For a sample mean with unknown $\sigma$, replace $\sigma$ by $s$ and use the Student $t$ distribution with $N-1$ degrees of freedom (it has fatter tails than the normal to pay for estimating $\sigma$):

$$
\bar x\ \pm\ t_{N-1,\,0.975}\cdot\frac{s}{\sqrt N}.
$$

Correct reading: the *procedure* captures the true value in 95% of repeated experiments. It does **not** say there is a 95% probability that the true value lies in the one interval you computed (the true value is fixed; the interval is what is random).

For a test accuracy $\hat p$ measured on $N$ independent examples, the standard error is $\sqrt{\hat p(1-\hat p)/N}$ (a Bernoulli mean). With $\hat p=0.87$ and $N=500$ it is $0.0150$, so a 95% interval is about $\pm1.96\times0.0150=\pm0.029$: **0.87 and 0.89 are statistically indistinguishable on a 500-example test set.** This one calculation prevents many wrong deployments.

## Hypothesis tests and p-values

1. State a null hypothesis $H_0$ (the boring explanation, e.g. "the two models have the same expected score") and an alternative $H_1$.
2. Pick a test statistic $T$ whose distribution under $H_0$ you know.
3. The **p-value** is $P(\text{a statistic at least as extreme as the observed one}\mid H_0)$.
4. Small p-value (below a pre-chosen $\alpha$, commonly $0.05$) means the data is surprising under $H_0$, so you reject it.

### Pitfalls (each one has caused real bad decisions)

- A p-value is **not** $P(H_0\mid\text{data})$. It is a statement about data given $H_0$, never the reverse (the confusion is the same as confusing $P(A\mid B)$ with $P(B\mid A)$ in the Bayes lesson).
- "Not significant" does not mean "no difference". It may just mean too little data.
- **Statistical vs practical significance.** With a million test examples a 0.03% improvement can have $p=0.001$ and still be worthless. Report an *effect size* (for example the mean difference in the metric's own units, or Cohen's $d$ = difference divided by the pooled standard deviation) next to the p-value.
- **Multiple comparisons.** Run $m$ independent tests at level $\alpha$ with all nulls true and $P(\text{at least one false positive})=1-(1-\alpha)^m$. For $m=20$, $\alpha=0.05$ this is $1-0.95^{20}=0.64$. Trying 20 hyper-parameter settings and reporting the best one is exactly this. The Bonferroni fix tests each at $\alpha/m=0.0025$.
- **Peeking and test-set reuse.** If you pick a model *because* it did best on the test set, that test score is optimistically biased. Keep a final untouched test set.

## Comparing two models fairly

Rules that follow from the above: evaluate both models on the **same** data splits; look at **paired differences** (a hard fold makes both scores low, and pairing cancels that shared difficulty); report an interval, not a single number.

**Paired t-test.** With per-fold differences $d_i=b_i-a_i$ ($i=1..K$), $t=\bar d\,/\,(s_d/\sqrt K)$ with $K-1$ degrees of freedom, a one-sample test of $\bar d=0$. Caveat: cross-validation folds share training data, so the $d_i$ are not truly independent and the test is somewhat optimistic; treat it as a guide, not a verdict.

**Bootstrap.** Resample the $K$ differences (or, with a big test set, the test *examples*) with replacement, recompute the statistic, repeat $B$ times, and read the 2.5th and 97.5th percentiles of the results as a 95% interval. It needs no formula and works for awkward metrics (AUC, F1).

## Worked example

Ten cross-validation folds, accuracies of model A and B:

| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | .842 | .851 | .839 | .860 | .848 | .855 | .845 | .852 | .840 | .858 |
| B | .850 | .862 | .841 | .871 | .850 | .866 | .858 | .861 | .847 | .865 |

Differences $d_i=B_i-A_i$ have mean $\bar d=0.0081$ and sample standard deviation $s_d=0.003755$. The standard error is $s_d/\sqrt{10}=0.001187$, so $t=0.0081/0.001187=6.82$ with 9 degrees of freedom: far beyond the 5% critical value $2.262$, so the improvement is not noise. A bootstrap with 5000 resamples (seeded `default_rng(0)`, drawing 10 differences each time) gives a 95% interval of roughly $[0.0058,\ 0.0101]$ for the mean gain, which excludes zero. Conclusion: B is better by about 0.8 points, plausibly between 0.6 and 1.0. Notice the data here is *paired*: B beat A on all ten folds.

A second tiny example for MAP. One-weight regression through the origin with $x=(1,2,3)$, $y=(1.2,1.9,3.4)$. Minimising $\sum(y-wx)^2+\lambda w^2$ gives $\hat w=\sum xy\,/\,(\sum x^2+\lambda)$. Here $\sum xy=1.2+3.8+10.2=15.2$ and $\sum x^2=14$, so $\hat w=15.2/(14+\lambda)$: $1.0857$ for $\lambda=0$ (maximum likelihood), $1.0133$ for $\lambda=1$ and $0.8$ for $\lambda=5$. Larger $\lambda$ (stronger prior) pulls the weight toward zero.

## Implementation

```python
import numpy as np

def bootstrap_ci(d, B=5000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d)).mean() for _ in range(B)])
    a = (1 - level) / 2
    return np.percentile(means, [100 * a, 100 * (1 - a)])

def ridge_1d(x, y, lam):
    return (x @ y) / (x @ x + lam)      # MAP with Gaussian prior, lam = sigma^2 / tau^2
```

## Shapes

Data matrix `X: (N, D)`, targets `y: (N,)`. Per-example log-likelihoods have shape `(N,)` and the loss is their (negated) sum or mean, a scalar. Paired scores for $K$ folds: two arrays `(K,)`, differences `(K,)`. A bootstrap draws an index array of shape `(B, K)`, giving `B` resampled statistics.

## Common misconceptions

- **"p = 0.03 means the null has a 3% chance of being true."** No: it is the chance of such data if the null were true.
- **"The 95% interval contains the truth with 95% probability."** The procedure does; a single computed interval either does or does not.
- **"Unbiased is always better."** Regularised estimators (ridge) accept a little bias in exchange for a lot less variance, and often have smaller total error.
- **"Significant means important."** It only means distinguishable from zero given the sample size.
- **"Bootstrap creates new information."** It cannot; it only shows how much your sample size limits you. With a biased or tiny sample it will be confidently wrong.

## Key takeaways

- An estimate is random; judge estimators by bias and variance, with error $=\text{bias}^2+\text{variance}$. The standard error of a mean is $\sigma/\sqrt N$.
- MLE minimises the negative log-likelihood: Gaussian noise gives least squares, Bernoulli gives cross-entropy. The unbiased variance divides by $N-1$.
- MAP adds a prior: a Gaussian prior $\mathcal N(0,\tau^2I)$ gives ridge with $\lambda=\sigma^2/\tau^2$.
- Always attach an interval to a metric, compare models on paired folds, correct for multiple comparisons, and report effect sizes alongside p-values.

## Explain it yourself

1. A teammate says "the p-value is 0.04, so there is a 96% chance the new model is better". Explain precisely what is wrong and what the 0.04 does mean.
2. Without symbols, explain why adding an L2 penalty to least squares is the same as believing, before seeing data, that weights are small, and what happens to that belief as you collect more data.

## Further reading

- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*, Ch. 8 (MLE, MAP, then full Bayesian linear regression in Ch. 9).
- Bishop, *Pattern Recognition and Machine Learning*, Sec. 1.2.3-1.2.6.
- Repo phase 01, lesson 15 (statistics for ML) for t-test, chi-squared and non-parametric alternatives such as Wilcoxon.

## Mastery criteria

- Derive least squares from a Gaussian likelihood and ridge from a Gaussian prior, naming what $\lambda$ equals.
- Compute a $t$-based confidence interval and a bootstrap interval by hand or in code.
- State the correct meaning of a p-value and of a 95% interval, and name three ways testing goes wrong.
- Decide, with numbers, whether two scores on a test set of given size are distinguishable.
