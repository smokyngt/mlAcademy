---
id: ops.06
title: "Monitoring, drift and continual learning"
summary: Why deployed models fail, how to name and detect covariate, label and concept shift (KS test and PSI from scratch), what to monitor, how continual learning updates models, and how to test a new model in production without fooling yourself with peeking.
minutes: 80
teaches: [ops.drift, ops.monitoring, ops.continual, ops.abtest]
requires: [stat.testing, prob.rv, ops.requirements, ops.serving]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.8 Data Distribution Shifts and Monitoring: causes of ML system failures, covariate/label/concept shift, detecting shifts (summary statistics, two-sample and KS tests, time-scale windows), monitoring and observability, alerts (PDF p.264-287, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.9 Continual Learning and Test in Production: stateless retraining versus stateful training, why continual learning, the four stages, shadow deployment, A/B testing, canary release, interleaving, bandits (PDF p.290-325, read in selected parts). READ."
---

## Where this fits

Deployment ([[ops.05]]) is not the finish line. The world changes: users change, competitors change, upstream systems change a column. A model that scored well offline ([[ops.04]]) slowly or suddenly gets worse, usually without any error message. This lesson closes the loop of the course: name the ways a system fails, detect shifts in the data with statistical tests, monitor the right signals, update the model continually, and compare a new model with the old one in production through experiments. The skew checks of [[ops.03]] reappear here as continuous monitors, and the z test and sampling ideas come from the statistics lessons ([[ops.02]] for sampling, and the earlier probability and testing material).

## The idea in plain words

A weather forecaster trained on twenty years of data is hired in a city whose climate is shifting. Three different things can go wrong. The kinds of days that occur may change (more heatwaves than in the training years), though a given set of readings still means the same thing: that is **covariate shift**. The base rate of the thing you predict may change (rain is now frequent), though rainy days still look the same: **label shift**. Or the same readings may now mean something different (a pressure drop that used to bring rain no longer does): **concept drift**. Each calls for a different response.

A smoke alarm is a good model for monitoring. It cannot wait for the fire's insurance claim (the true label arrives late, if at all), so it watches indirect signals that change first: temperature, particles in the air, battery state. For an ML system the indirect signals are the distribution of the inputs, the distribution of the predictions, and the health of the service. Testing a new model in production is like trying a new recipe: you can have the cook prepare it without serving it (shadow), serve it to a few tables (canary), split the room randomly and compare (A/B), or shift tables toward the better dish as evidence accumulates (bandit).

## Why ML systems fail

A failure is a violated expectation of the system. Two families:

1. **Software failures**, which any system has: dependency breaks, wrong binary deployed, hardware failure, downtime, distributed-system and data-pipeline bugs. The book reports a study of 96 failures of a large production ML pipeline in which 60 were not directly ML-related.
2. **ML-specific failures**:
   - **Production data differs from training data.** The test set is supposed to represent unseen data, but the world is not stationary and a finite training set cannot cover it.
   - **Edge cases.** Rare inputs that cause catastrophic mistakes (a self-driving car that is right 99.99% of the time may still be unusable); a sudden increase of badly handled cases can also signal a distribution shift.
   - **Degenerate feedback loops.** The system's predictions influence the data it later learns from. A recommender shows song A first, so A is clicked more, so the system ranks A higher still; the result is popularity bias and filter bubbles, and new items cannot break through. Exploration (bandits, below) and randomisation mitigate it.

## Data distribution shift

Let $X$ be the inputs and $Y$ the labels. The joint distribution factors in two ways:

$$
P(X,Y)=P(Y\mid X)\,P(X)=P(X\mid Y)\,P(Y).
$$

Each factor can change, giving three named shifts.

| Shift | What changes | What stays the same | Example |
|---|---|---|---|
| **Covariate shift** | $P(X)$ | $P(Y\mid X)$ | More patients over 40 in production than in training; the relation between age and risk is unchanged. |
| **Label shift** | $P(Y)$ | $P(X\mid Y)$ | The disease becomes more prevalent; given the disease, patients look the same. |
| **Concept drift** | $P(Y\mid X)$ | $P(X)$ | The same house features now sell for less after a market change ("same input, different output"). Often cyclic or seasonal. |

Other changes also hurt: a feature's units change, columns are added or removed, or the set of possible values changes (the age feature switches from years to months). Not every shift matters; the ones that matter reduce performance.

### Detecting shifts

When labels are available quickly, monitor accuracy-related metrics directly. When they are not (delayed or absent), monitor distributions you can observe without labels: mostly $P(X)$, and the distribution of the model's predictions.

*Summary statistics.* Compare mean, median, variance, quantiles between a reference window and the current window. Cheap and widespread, but necessary-not-sufficient: equal means do not imply equal distributions.

*Two-sample tests.* Treat the reference window and the current window as two samples and ask whether they could come from the same distribution. A caveat from the book: a statistically significant difference is not necessarily a practically important one, but a difference that is detectable from a small sample is probably large.

**Kolmogorov-Smirnov (KS) test.** Let $F_a$ and $F_b$ be the empirical cumulative distribution functions of two samples of sizes $n$ and $m$. The KS statistic is the largest vertical gap

$$
D=\max_x\left|F_a(x)-F_b(x)\right| .
$$

It is non-parametric (no assumption on the shape) and it takes values in $[0,1]$: 0 for identical samples, 1 for samples that do not overlap. Under the null hypothesis that both samples come from the same continuous distribution, the scaled statistic $\sqrt{nm/(n+m)}\,D$ has an asymptotic distribution with tail

$$
P(D\text{ at least this large})\approx 2\sum_{k=1}^{\infty}(-1)^{k-1}e^{-2k^2\lambda^2},\qquad
\lambda=\Big(\sqrt{n_e}+0.12+\tfrac{0.11}{\sqrt{n_e}}\Big)D,\quad n_e=\tfrac{nm}{n+m},
$$

a standard small-sample correction. Practical limits (stated in the book): KS works for one-dimensional data, so you apply it per feature or on predictions and labels; it can produce many false alarms when you test many features and large samples; for high-dimensional data reduce the dimension first or use multivariate tests such as MMD.

**Population stability index (PSI).** Not covered in the book; it is a widely used industry score (stated here from general practice). Cut the reference sample into bins of equal mass using its quantiles (say deciles), compute the proportion $p_j$ of the reference and $q_j$ of the new sample in each bin $j$, and

$$
\mathrm{PSI}=\sum_{j}(q_j-p_j)\ln\frac{q_j}{p_j}.
$$

Each term is non-negative (the numerator and the log have the same sign), PSI is 0 only if the histograms are equal, and it equals the sum of the two directions of the KL divergence (a symmetric measure of how different the histograms are). Proportions are clipped to a small value so that empty bins do not give infinity. Rules of thumb in practice use thresholds around 0.1 (look) and 0.2-0.25 (alert); these are conventions, not laws, so calibrate on your own data.

*Time-scale windows.* Whether data "shifted" depends on the reference window. With a weekly cycle, comparing today with the last six days can look like a shift, while comparing with the last two weeks shows it is normal seasonality. Shifts can be spatial (a new group of users or devices) or temporal, and sudden shifts are easier to catch than gradual ones. Use sliding rather than cumulative statistics for recency.

### Addressing shifts

Three approaches: train on a large and varied dataset hoping to cover production; adapt a trained model to the target distribution without new labels (mostly research); or retrain regularly on fresh data, which is what most companies do. How often to retrain is usually a gut decision; the next sections show how to make it a measured one.

## Monitoring and observability

**Monitoring** is tracking, measuring and logging metrics to know when something goes wrong; **observability** (instrumentation) is setting the system up so you can investigate why: timers on functions, counts of NaNs in features, logging of unusual inputs, tracing inputs through transformations. Without observability, monitoring has nothing to measure.

Two classes of metrics:

- **Operational metrics**: latency, throughput, request counts, share of 2xx responses, CPU/GPU and memory utilisation, uptime (the percentage of time the service meets its service level objective). Four nines (99.99%) allows roughly four minutes of downtime a month.
- **ML-specific metrics**, from four artifacts at increasing distance from the model: *raw inputs*, *features*, *predictions* and *accuracy-related metrics*. The deeper the artifact is in the pipeline, the closer it is to what you care about and the easier to interpret, but the more likely it is that a change originates in one of the many earlier transformations; so monitor several levels. Use any feedback signal you can capture (clicks, downvotes, completion rates); a click-through rate that stays flat while completion rate falls suggests that recommendations got worse.

*Alerts* have three parts: an **alert policy** (the condition and duration, for example accuracy below 90%, or latency over 1 s for 10 minutes), **notification channels** (who is told, where), and a **description** (what happened and what to check). Too many alerts cause alert fatigue; the thresholds of the drift scores above must be tuned to your tolerance for false alarms.

## Continual learning

**Continual learning** means setting up infrastructure so that models can be updated as fast as needed, not constant retraining for its own sake. Two ways to update:

- **Stateless retraining**: train from scratch on a window of data (for example the last three months) each time.
- **Stateful training** (fine-tuning): load yesterday's checkpoint and continue on only the new data. It needs less data and compute (the book cites a company that cut training cost 45-fold and raised its purchase-through rate by 20% by moving from daily stateless to daily stateful training), and in principle each sample is used once, so data need not be stored (a privacy benefit). Occasional from-scratch training is still used to recalibrate.

*Why:* respond to sudden shifts (a surge in ride demand at an event), adapt to rare events (Black Friday has too little history).

*Four stages* of maturity: (1) **manual, stateless retraining** whenever the model is clearly bad and someone has time; (2) **automated retraining** by a scheduled script, with a frequency chosen by gut feeling; (3) **automated stateful training**, which needs tracking of data and model lineage; (4) **continual learning** triggered by shifts or performance drops rather than a calendar. Stage 4 is a steep step: it needs good monitoring, safe evaluation of each candidate (below) and reliable infrastructure.

## Test in production

Offline evaluation ([[ops.04]]) cannot tell how a model behaves on live traffic with real feedback, so new models are tested on production traffic in controlled ways.

- **Shadow deployment**: the candidate receives every request in parallel but only the incumbent's prediction is served; you log and compare. Safest, but doubles inference cost.
- **A/B testing**: randomly split users between incumbent (A) and candidate (B), serve each their model, and compare a metric with a statistical test. Randomisation is essential, and sample size and metric must be fixed in advance.
- **Canary release**: send a small share of traffic to the candidate, increase it if its metrics hold, abort and roll back if they degrade. It does not need randomisation (you could canary in one market first).
- **Interleaving**: show each user a *mixed* list containing items from both rankers and count which model's items get clicks; each user experiences both models, which makes the comparison need fewer users than A/B (Netflix reports reliable identification with considerably smaller samples). It suits ranking and recommendation.
- **Bandits**: treat each model as a slot machine whose payoff is unknown and route traffic adaptively, balancing **exploration** (try others) and **exploitation** (use the best so far). They need online prediction and short feedback loops, and they are stateful (you must track each model's payoff). Simple strategies: $\varepsilon$-greedy (a fraction of traffic goes to a random model); **Thompson sampling** (sample a plausible success rate for each model from its posterior and play the largest); **UCB** (play the model with the highest optimistic upper confidence bound). In an experiment cited in the book, Thompson sampling decided a 5% difference with under 12,000 samples where a fixed A/B test needed over 630,000. Bandits are harder to implement and are mainly used at large companies.

### The A/B test and the peeking problem

For a binary outcome, with $x_A$ successes out of $n_A$ and $x_B$ out of $n_B$, the pooled proportion is $\hat p=(x_A+x_B)/(n_A+n_B)$, and the two-proportion $z$ statistic for $B-A$ is

$$
z=\frac{x_B/n_B-x_A/n_A}{\sqrt{\hat p(1-\hat p)\left(\tfrac1{n_A}+\tfrac1{n_B}\right)}} .
$$

Under the null (equal rates), $z$ is approximately standard normal, and rejecting when $|z|>1.96$ gives a 5% false-positive rate **if you look exactly once, at a sample size fixed in advance**. The required size per arm to detect a difference $\delta$ at 5% significance with 80% power is about $n\approx 2\,(1.96+0.84)^2\,p(1-p)/\delta^2$. For $p=0.10$ and $\delta=0.03$ that is $2\cdot7.84\cdot0.09/0.0009=1{,}568$ users per arm.

Now imagine checking the dashboard every day and stopping at the first $|z|>1.96$. Each look is another chance for noise to cross the threshold, and the tests are correlated but not identical, so the chance of *some* look crossing is much more than 5%. This is the **peeking problem**. In an **A/A test** (both arms identical, so any "winner" is a false positive), with 1,000 users per arm and ten equally spaced looks, our simulation declares a winner in about 20% of experiments, versus 5% for one look. Remedies: fix the sample size and look once; use a correction (a stricter threshold per look, or alpha-spending); use proper sequential methods designed for continuous monitoring; or use a bandit when your goal is reward rather than a clean hypothesis test.

Beyond peeking: pick a metric tied to the business objective ([[ops.01]]), check that randomisation has no bug (sample-ratio check), run for whole weekly cycles, and beware of testing many metrics or many variants without correcting.

## Worked example

All values come from the exercise code with fixed seeds.

*Drift detection.* Reference: 1,000 draws from $N(0,1)$. New samples of 1,000:

| New sample | KS $D$ | KS p-value | PSI | Verdict |
|---|---|---|---|---|
| same $N(0,1)$ | 0.052 | 0.13 | 0.020 | no drift |
| $N(0.5,1)$ (mean shift) | 0.210 | $\approx9\times10^{-20}$ | 0.240 | drift |
| $N(0,2)$ (wider) | 0.191 | $\approx2\times10^{-16}$ | 0.591 | drift; mean unchanged |

The last row is why you look at distributions, not just means: the mean differs by less than 0.2, yet both KS and PSI flag the variance change.

*A/B test.* 100 of 1,000 versus 130 of 1,000 gives pooled $\hat p=0.115$, standard error $0.01427$, $z=2.10$, just past 1.96. It is only borderline, and the needed size was 1,568 per arm, so a careful team would collect more before concluding. *Peeking*: A/A, $p=0.1$, 1,000 users per arm, 2,000 simulated experiments: one look declares a false winner in 5.05% of runs, ten looks in 20.5% (about four times more). *Bandit*: Thompson sampling with true rates 0.10 and 0.20 over 3,000 rounds pulls the worse arm only 29 times and the better arm 2,971 times, whereas a 50/50 A/B test would send 1,500 users to the worse arm.

## Implementation

```python
import numpy as np

def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))

def psi(reference, new, bins=10, eps=1e-4):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])
    p = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=bins) / len(reference)
    q = np.bincount(np.searchsorted(edges, new, side="right"), minlength=bins) / len(new)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))

def z_test(xa, na, xb, nb):
    p = (xa + xb) / (na + nb)
    return float((xb / nb - xa / na) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb)))
```

Libraries: `scipy.stats.ks_2samp` (exact and asymptotic p-values), Alibi Detect and Evidently for drift detection, Prometheus/Grafana or cloud monitoring for operational metrics, and experimentation platforms for A/B tests and bandits.

## Shapes

Reference and new samples are 1-D arrays of lengths $n$ and $m$ (one feature at a time); `edges: (bins-1,)`; bin proportions `(bins,)`; counts in a 2x2 A/B table are scalars. In simulations, outcome matrices have shape `(n_sims, n_per_arm)`.

## Common misconceptions

- "If accuracy was fine at launch, it stays fine." Distributions move; monitor continuously.
- "No error means no failure." ML systems fail silently.
- "Any significant drift needs retraining." Significance is not importance; check the impact on performance.
- "Means are enough to detect drift." A variance or shape change can leave the mean untouched.
- "Peeking at an A/B test is harmless if I only stop when it is clearly significant." Repeated looks inflate false positives severalfold.
- "Continual learning means retraining constantly." It means being able to update as fast as needed, safely.
- "Bandits always beat A/B tests." They need fast feedback and online prediction, and give less clean statistical conclusions.

## Key takeaways

- Failures are software-related or ML-specific (data mismatch, edge cases, degenerate feedback loops).
- Covariate shift changes $P(X)$, label shift $P(Y)$, concept drift $P(Y\mid X)$.
- KS (largest CDF gap) and PSI (binned divergence) detect shifts in one-dimensional features without labels; always compare distributions, not only means.
- Monitor operational metrics and ML metrics at several pipeline stages; alerts need a policy, a channel and a description.
- Stateful training and automation lead toward continual learning; test candidates by shadow, canary, A/B, interleaving or bandits.
- Fix sample size before an A/B test; repeated looks inflate the false-positive rate (about 5% to about 20% with ten looks).

## Explain it yourself

1. A feature's mean is unchanged but its variance doubled. Which checks catch it, and which do not?
2. Why does checking an A/B test every day and stopping at the first $p<0.05$ give far more than 5% false positives, and what are two remedies?

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch.8-9 (PDF p.264-327). Alibi Detect documentation for multivariate drift detection (not read for this lesson).
- [[ops.04]] for offline evaluation and slices that decide what to monitor.

## Mastery criteria

You can state the three shifts in terms of $P(X)$, $P(Y)$ and $P(Y\mid X)$, implement KS and PSI and interpret them, design a monitoring plan with operational and ML metrics and alerts, explain stateless versus stateful updates, choose between shadow, canary, A/B, interleaving and bandits, and demonstrate the peeking problem by simulation.
