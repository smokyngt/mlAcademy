---
id: saf.05
title: "Bias, fairness and privacy"
summary: Measure representational and allocational harm, compute the group-fairness criteria and see why they cannot all hold at once, then build the Laplace and Gaussian mechanisms and DP-SGD clipping and check their privacy guarantees empirically.
minutes: 90
teaches: [saf.fairness-criteria, saf.differential-privacy]
requires: [prob.rv, prob.basics, ml.logreg, stat.testing]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/20-bias-representational-harm/docs/en.md (whole lesson): representational vs allocational harm, metric families, intersectionality, mechanistic approaches, meta-critique. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual/docs/en.md (whole lesson): group, individual and counterfactual fairness, impossibility, backtracking counterfactuals. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/docs/en.md (whole lesson): (epsilon, delta)-DP, DP-SGD recipe, LoRA+DP, MIA vs extraction tension, alternatives. READ."
---

## Where this fits

The earlier safety lessons were about adversaries and optimisers. This lesson is about harm that needs no villain: a model that is accurate on average but treats groups of people differently, or that remembers and reveals data about the individuals it was trained on. These are measurable properties, so the lesson is mostly *definitions you can compute* and *theorems that tell you what cannot be had together*. It depends on conditional probability and random variables (`prob.basics`, `prob.rv`) and classification basics (`ml.logreg`), and the evaluation habits from [[saf.04]] (report uncertainty) apply here too. The deployment context is in [[saf.06]].

## The idea in plain words

**Bias.** A system can be unfair in two different ways. It can *portray* people unfairly (always picturing nurses as women: a **representational** harm), or it can *decide* unfairly (scoring equally qualified applicants differently: an **allocational** harm). A model can pass tests for one and fail the other, so you measure both.

**Fairness.** "Treat people equally" has several precise meanings that sound alike and conflict:

- *Equal outcomes for groups*: the same share of each group gets the loan.
- *Equal mistakes for groups*: qualified people in each group are equally likely to be approved, and unqualified people equally likely to be wrongly approved.
- *Equal meaning of a score*: when the system says "70% risk", it is right 70% of the time in every group.
- *Similar people, similar treatment* (individual fairness).
- *Same decision if only the protected attribute had been different* (counterfactual fairness).

When two groups differ in how often the outcome really occurs, you cannot satisfy the first three together except in degenerate cases. This is a mathematical fact, not a modelling failure, so choosing a criterion is a policy decision that must be made explicitly.

**Privacy.** Releasing a statistic about a dataset reveals something about each person in it. Differential privacy promises that the output would look almost the same whether or not any single person's record were present, by adding carefully calibrated random noise. The price is accuracy; the guarantee is mathematical, not a matter of trust.

## Bias and representational harm

From the source lesson (a survey-based summary; we did not open the underlying papers):

- **Metric families.** *Embedding-based* tests measure associations between identity words and attribute words in representations; *probability-based* ones compare the likelihood of stereotype-consistent versus inconsistent completions; *generated-text* ones evaluate downstream outputs (such as scoring résumés). The last is the most realistic and the hardest to reproduce.
- **Intersectionality.** Bias along a single axis (say gender) can hide bias that appears only for combinations (gender and race together). The lesson reports a 2025 study of résumé scoring in which a combination was penalised more than either factor alone, and a benchmark that measures differences in the model's *uncertainty*, not only its point predictions, across intersectional groups.
- **Mechanistic approaches** (reported): locating neurons, sparse-autoencoder features or attention heads that correlate with identity-sensitive behaviour and ablating or re-weighting them, with limited capability cost in the cited studies.
- **A meta-critique** (reported): the literature over-focuses on binary gender, so a model may look debiased on the axis everyone tests while remaining biased on axes nobody checks (disability, religion, migration status, language).

## Group fairness criteria

Let $A$ be the group, $Y\in\{0,1\}$ the true outcome, $\hat Y\in\{0,1\}$ the decision, and $S$ a score from which $\hat Y$ is obtained by thresholding.

| Criterion | Requirement |
|---|---|
| Demographic parity | $P(\hat Y=1\mid A=a)$ equal across groups |
| Equalised odds | $P(\hat Y=1\mid Y=y,A=a)$ equal across groups for $y=0$ and $y=1$ (equal FPR and TPR) |
| Predictive parity / conditional use accuracy | $P(Y=y\mid \hat Y=y,A=a)$ equal across groups (equal precision, PPV) |
| Calibration within groups | $P(Y=1\mid S=s,A=a)=s$ for every group |

Equalised odds is the ability of the system to treat *qualified* and *unqualified* people alike across groups; demographic parity ignores $Y$ altogether and can force different error rates when base rates differ.

### Why they conflict (derivation)

Fix a group with base rate $p=P(Y=1)$, true-positive rate $\mathrm{TPR}$ and false-positive rate $\mathrm{FPR}$. By Bayes' rule the precision is

$$
\mathrm{PPV}=P(Y=1\mid\hat Y=1)=\frac{\mathrm{TPR}\cdot p}{\mathrm{TPR}\cdot p+\mathrm{FPR}\cdot(1-p)} .
$$

Suppose two groups have equal TPR and equal FPR (equalised odds), but different base rates $p_1\ne p_2$. The expression above is strictly increasing in $p$ whenever $\mathrm{FPR}>0$ and $\mathrm{TPR}>0$, so the PPVs must differ. Therefore equalised odds and equal precision cannot both hold, unless the classifier is perfect ($\mathrm{FPR}=0$ or $\mathrm{TPR}=1$ with $\mathrm{FPR}=0$) or the base rates coincide. This is the content of the impossibility results (Chouldechova; Kleinberg, Mullainathan and Raghavan, as cited by the source). A score that is calibrated within each group, together with equal TPR and FPR, is likewise impossible under different base rates.

### Individual and counterfactual fairness

- **Individual fairness** (Dwork et al., as the source states): a decision map $f$ is fair if $|f(x)-f(x')|\le L\,d(x,x')$ for a task-specific similarity metric $d$. The statistical content is a Lipschitz condition; the hard part is policy, because someone must define "similar".
- **Counterfactual fairness** (Kusner et al.): a decision is fair to an individual if it would not change had the protected attribute been different in a causal model of the world. It requires a causal graph, which is a modelling assumption; the lesson reports a theoretical accuracy trade-off for enforcing it.
- **Backtracking counterfactuals** (reported, January 2024): rather than intervene on the protected attribute (legally awkward), ask what other changes in the person's features would have produced the counterfactual outcome.
- The source notes that with a causal graph some group criteria entail counterfactual fairness, so the families are less opposed than they look, but the impossibility result under unequal base rates remains.

## Differential privacy

### Definition

A randomised algorithm $M$ is $(\varepsilon,\delta)$-differentially private if for all datasets $D,D'$ that differ in one person's record, and all sets of outputs $S$,

$$
P(M(D)\in S)\le e^{\varepsilon}\,P(M(D')\in S)+\delta .
$$

Smaller $\varepsilon$ means the output distributions are closer, so stronger privacy; $\delta$ is a small probability with which the guarantee may fail. There is no universal "safe" $\varepsilon$: the source states plainly that published values vary with threat model and utility target.

### The Laplace mechanism

For a numeric query $f$ whose value changes by at most $\Delta$ (its **sensitivity**) when one record changes, release $f(D)+\text{Lap}(\Delta/\varepsilon)$ where Laplace noise with scale $b$ has density $\frac{1}{2b}e^{-|x|/b}$.

**Why $\varepsilon$-DP holds.** For any output $x$, the density ratio is

$$
\frac{p_D(x)}{p_{D'}(x)}=\exp\!\Big(\frac{|x-f(D')|-|x-f(D)|}{b}\Big)\le\exp\!\Big(\frac{|f(D)-f(D')|}{b}\Big)\le e^{\Delta/b}=e^{\varepsilon}.
$$

The first inequality is the triangle inequality, the second uses the definition of sensitivity, and the last sets $b=\Delta/\varepsilon$. Integrating the density ratio over any set $S$ gives the guarantee with $\delta=0$.

### The Gaussian mechanism

Release $f(D)+\mathcal{N}(0,\sigma^2)$. The **privacy loss** at output $x$ is $L(x)=\log\frac{p_D(x)}{p_{D'}(x)}$. With $f(D')=f(D)+\Delta$ and $x=f(D)+\sigma Z$ for $Z\sim\mathcal N(0,1)$:

$$
L=\frac{(x-f(D)-\Delta)^2-(x-f(D))^2}{2\sigma^2}=\frac{\Delta^2-2\Delta\sigma Z}{2\sigma^2}=\frac{\Delta^2}{2\sigma^2}-\frac{\Delta}{\sigma}Z,
$$

so $L\sim\mathcal N\big(\tfrac{\Delta^2}{2\sigma^2},\tfrac{\Delta^2}{\sigma^2}\big)$. The mechanism is $(\varepsilon,\delta)$-DP when $P(L>\varepsilon)\le\delta$. The classical sufficient calibration (a standard result we state without proof) is

$$
\sigma\ge\frac{\Delta\sqrt{2\ln(1.25/\delta)}}{\varepsilon}\qquad(\varepsilon<1).
$$

It is conservative: for $\Delta=1,\varepsilon=0.9,\delta=0.01$ it gives $\sigma=3.453$, where the loss has mean $0.042$ and standard deviation $0.290$, so $P(L>0.9)=P(Z>2.96)\approx0.0015$, well below $\delta$.

### DP-SGD

To train a model with a DP guarantee, the standard recipe (Abadi et al., as the source describes) changes how gradients are used:

1. Sample a mini-batch; compute **per-example** gradients $g_i\in\mathbb R^d$.
2. **Clip** each to L2 norm at most $C$: $\bar g_i=g_i\cdot\min(1,C/\|g_i\|)$. Now each example can move the *sum* by at most $C$: the sensitivity is bounded by construction.
3. Add Gaussian noise: $\tilde g=\sum_i\bar g_i+\mathcal N(0,\sigma^2C^2I_d)$, then divide by the batch size.
4. Take the optimiser step with $\tilde g$.

The total $(\varepsilon,\delta)$ over many steps is tracked by a *privacy accountant* (the source names the moments and Rényi accountants), which uses privacy amplification from random sub-sampling of batches and tighter composition than simply adding the per-step $\varepsilon$ (basic composition adds them and is loose). The exercise implements clipping, the noisy sum and basic composition, not an accountant. In practice for language models the source reports the common 2025 setup is to apply DP-SGD to low-rank adapter parameters only (LoRA), so the guarantee covers the adapter while the base model is held fixed.

### What the guarantee covers, and the tension in the evidence

The source reports two bodies of evidence that look contradictory: canary-based membership-inference studies report limited attack success against language models, while training-data extraction studies recover substantial verbatim memorisation. A 2025 analysis reconciled them: they measure different things, and the *most extractable* examples drive privacy risk, which randomly inserted canaries under-represent. Alternatives to DP training were also described: private prediction at inference time by aggregating experts trained on shards (PMixED), and DP synthetic data. An emerging attack uses a DP-trained model's *confidence scores* as an oracle, so a deployment should not expose raw confidences. Treat all these as reported findings.

## Worked example: fairness

Synthetic data (20,000 people, two groups), score $S$ calibrated by construction in both groups (label drawn with probability $S$), group 1 having higher scores. Base rates: 0.286 and 0.500.

At a common threshold 0.5:

| | group 0 | group 1 |
|---|---|---|
| selection rate | 0.109 | 0.493 |
| TPR | 0.228 | 0.629 |
| FPR | 0.061 | 0.357 |
| PPV | 0.598 | 0.638 |
| calibration error (10 bins) | 0.009 | 0.014 |

Demographic-parity gap $0.384$; equalised-odds gap $0.401$ (the TPR gap), although the score is calibrated in both groups (ECE below 0.02). Next, lower group 0's threshold to $0.312$ so that its TPR matches group 1's ($0.629$). Now TPR is equal, but PPV is $0.455$ against $0.638$ and FPR $0.302$ against $0.357$. If both groups had exactly TPR $0.629$ and FPR $0.357$, Bayes' rule would force PPV $0.414$ for group 0 against $0.638$ for group 1: the numbers behave as the derivation predicts.

## Worked example: privacy

Count query, sensitivity 1, neighbouring counts 100 and 101.

- Gaussian: $\varepsilon=0.9,\delta=10^{-2}\Rightarrow\sigma=3.453$; $\varepsilon=0.5,\delta=10^{-5}\Rightarrow\sigma=9.690$ (smaller $\varepsilon$ and $\delta$ cost much more noise).
- Empirical check with 200,000 samples per dataset: for events $\{x>t\}$ the largest excess of $P(M(D'))-e^\varepsilon P(M(D))-\delta$ is zero up to sampling error (about 0.001), for the calibrated Gaussian and for Laplace with $\varepsilon=1$. With a quarter of the noise the excess is above 0.1 in both cases (Gaussian, and Laplace with claimed $\varepsilon=1$ but scale 0.25; in one run about 0.19 and 0.75): the empirical test detects an invalid claim.
- DP-SGD clipping: with per-example gradients scaled to norm 3 and $C=1$, each clipped row has norm at most 1, and removing any example changes the clipped sum by at most 1.
- Basic composition of $(0.5,10^{-6})$, $(0.25,10^{-6})$, $(0.25,0)$ gives $(1.0,\,2\times10^{-6})$.

An empirical check can only *refute* a privacy claim for the events tested; it can never prove one. The proof is the derivation.

## Implementation

```python
import math
import numpy as np

def gaussian_sigma(sensitivity, eps, delta):
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps

def clip_rows(G, C):
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    return G * np.minimum(1.0, C / np.maximum(norms, 1e-12))

def dp_sgd_sum(G, C, sigma, rng):
    return clip_rows(G, C).sum(axis=0) + sigma * C * rng.normal(size=G.shape[1])
```

In production, libraries such as Opacus (PyTorch) or TensorFlow Privacy compute per-example gradients efficiently and include the accountants.

## Shapes

- Per-example gradients `G`: `(B, d)`; clipped `(B, d)`; noisy aggregate `(d,)`.
- Fairness: `g`, `s`, `y`, `pred`: `(n,)`; masks are boolean `(n,)`.

## Common misconceptions

- **"A calibrated score is a fair score."** The example is calibrated in both groups and still has a 0.40 gap in error rates.
- **"We can pick the fairest model by optimising all metrics."** Under unequal base rates the criteria are provably incompatible; the choice is a policy decision, made with the affected people in mind.
- **"Removing the protected attribute removes bias."** Other features correlate with it; the metrics above are computed from outcomes and need the attribute.
- **"DP means the data are anonymised."** DP is a property of the *algorithm*, bounding what any output reveals about one record; $\varepsilon$ values that are large provide little protection.
- **"An empirical test that finds no violation proves privacy."** It only fails to refute the claim for the tested events.
- **"Clipping merely stabilises training."** In DP-SGD it is what bounds each person's influence, so the noise can be calibrated.

## Key takeaways

- Distinguish representational from allocational harm and test both, including intersections.
- Demographic parity, equalised odds and equal precision/calibration conflict when base rates differ; derive it from $\mathrm{PPV}=\mathrm{TPR}\,p/(\mathrm{TPR}\,p+\mathrm{FPR}(1-p))$.
- Individual and counterfactual fairness need a similarity metric and a causal model respectively; both are assumptions.
- Laplace noise of scale $\Delta/\varepsilon$ gives $\varepsilon$-DP; Gaussian noise gives $(\varepsilon,\delta)$-DP; clipping makes the sensitivity of a gradient sum known.
- Privacy accounting, off-distribution memorisation and exposed confidences are the practical difficulties.

## Explain it yourself

1. Explain to a non-specialist why a loan model can be "equally accurate for everyone" and still be unfair under another definition.
2. Why do we clip per-example gradients before adding noise in DP-SGD, instead of only adding noise?

## Further reading

- Named in the source lessons: the representational-bias survey (arXiv:2309.00770), the Dwork et al. individual-fairness paper, Kusner et al. on counterfactual fairness, the impossibility papers of Chouldechova and of Kleinberg, Mullainathan and Raghavan, Abadi et al. on DP-SGD, and the 2025 analysis reconciling MIA with extraction (arXiv:2503.06808). Only the repo summaries were read.

## Mastery criteria

- Compute demographic parity, equalised odds, PPV and ECE for a set of predictions.
- Derive the PPV expression and use it to show the impossibility under unequal base rates.
- Prove the Laplace mechanism's $\varepsilon$-DP and derive the Gaussian privacy-loss distribution.
- Describe DP-SGD and what clipping and noise each contribute.
