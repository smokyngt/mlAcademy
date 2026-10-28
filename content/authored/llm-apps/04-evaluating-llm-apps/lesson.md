---
id: app.04
title: "Evaluating and testing LLM applications"
summary: Build an eval set, score it with checks and calibrated judges, and decide with a bootstrap interval and an exact permutation test whether prompt B really beats prompt A.
minutes: 75
teaches: [app.eval]
requires: [app.prompting, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/11-llm-engineering/10-evaluation/docs/en.md (opening sections: the silent-regression story, eval taxonomy, LLM-as-judge criteria and rubric design; first ~6.5k characters only). READ."
---

## Where this fits

Every prompt change, model swap or temperature tweak changes the distribution of outputs in ways you cannot judge by reading ten examples. Software engineers do not ship without tests; for LLM applications the equivalent is an **evaluation harness** run on every change. This lesson is about building one and, above all, about *reading its numbers honestly*: outputs are random, test sets are small, and judges are imperfect. The previous lessons gave you things to evaluate (prompts in [[app.01]], schemas and tools in [[app.02]], context layouts in [[app.03]]). Evaluation of the model itself and of RAG pipelines (faithfulness, retrieval metrics, benchmark contamination) is covered in lesson `sys.04`; here we focus on application-level testing and statistics. The statistics use the idea of a random variable and its variance from `prob.rv`.

## The idea in plain words

Suppose a baker tweaks a recipe and asks three friends whether the new bread is better. Two say yes. Is it better? You cannot tell: three friends is a tiny sample, they may be polite, and the baker may have fed them the good slices. A fair test uses many tasters, the same slices for both recipes, a clear scoring sheet, and a rule decided *before* tasting for what counts as a real difference.

Evaluating an LLM app is the same. The **eval set** is the tasters. The **scoring sheet** is a rubric or a check. Pairing both recipes on the same slices is a **paired comparison**, which cancels the difficulty of each item. And the rule for "real difference" is a **significance test** or a **confidence interval**, which tell you how much of the gap could be luck.

## Building an eval set

- **Source it from reality.** Sample real (anonymised) user inputs; add the failures you have seen; add edge cases (empty input, very long input, other languages, adversarial text).
- **Label what matters.** For each item store the input, a reference answer or acceptance rule, and tags (topic, difficulty) so you can slice results.
- **Keep two sets.** A development set you look at while tuning, and a held-out set you touch rarely; otherwise you overfit the prompt to the set, exactly like overfitting a model.
- **Size it for the effect you care about.** Small sets only detect large differences (see the sample-size formula below).
- **Version it** with the prompt and model version.

## Scoring: three kinds of scorer

1. **Deterministic checks**: does it parse against the schema ([[app.02]]), contain a required field, stay under a length, avoid forbidden strings? Fast, free, unambiguous. Use them first and as gates.
2. **Reference metrics** (exact match, n-gram overlap, embedding similarity): cheap but blind to meaning; two correct answers can share no words.
3. **LLM-as-judge**: a strong model grades an output against a rubric. It captures relevance, correctness and tone, costs money, and can be biased. The source reports correlations with human judgement in the 80s percent for well-designed rubrics (figures we cannot verify and that depend on the task).

**Rubrics must be anchored.** "Rate 1-5 how good" gives noisy scores. Anchor each level to observable behaviour ("5: factually correct, answers the question directly, cites the source; 3: correct but misses a required element; 1: contains a factual error").

### Judge pitfalls

These are widely reported in practice (they are not all in the source we read, so test them on your own judge):

- **Position bias**: in pairwise comparison the judge may favour the first (or second) answer. Run both orders and count only consistent verdicts.
- **Verbosity bias**: longer answers look better. Control for length or say so in the rubric.
- **Self-preference**: a model may favour text in its own style; use a different judge model than the generator when possible.
- **Drift**: the judge is a model too; pin its version.
- **Leakage**: do not show the judge the label you want it to confirm.

**Calibrate against humans.** Label 50 to 100 items by hand, run the judge, and measure agreement beyond chance with **Cohen's kappa**:

$$
\kappa=\frac{p_o-p_e}{1-p_e},\qquad p_e=\sum_c P_{\text{human}}(c)\,P_{\text{judge}}(c).
$$

Here $p_o$ is the observed fraction of agreement and $p_e$ the agreement expected if both labelled independently with their own base rates. Example: human labels `1,1,1,0,0,1,0,1,1,0` against judge `1,1,0,0,0,1,1,1,1,0` agree on 8 of 10 ($p_o=0.8$); both say 1 for 6 of 10, so $p_e=0.6\cdot0.6+0.4\cdot0.4=0.52$ and $\kappa=0.28/0.48=0.583$. A judge that always says "pass" has $\kappa=0$ whatever the base rate, even if it is right most of the time, which is why raw accuracy is a poor yardstick.

## Comparing two prompts

You run prompts A and B on the same $n$ items and score each 0/1 (or a rubric score). For each item $i$ define the paired difference $d_i=b_i-a_i$. Its mean $\bar d$ estimates how much better B is. The question is how much $\bar d$ would vary by chance.

### Percentile bootstrap

Treat the items as a sample from the population of future inputs. Resample $n$ items with replacement, recompute $\bar d$, repeat $B$ times (say 2,000), and take the 2.5th and 97.5th percentiles of the $B$ means as a 95 percent **confidence interval** for the true improvement. No normality assumption is needed. If the interval comfortably excludes zero, the improvement is credible; if it straddles zero, it is not.

### Exact permutation (sign-flip) test

Null hypothesis: the two prompts are interchangeable, so for each item the labels A and B could have been swapped. Swapping flips the sign of $d_i$; items with $d_i=0$ do not change. Hence under the null every assignment of signs to the $m$ non-zero differences is equally likely: $2^m$ equally likely patterns. Let $S=\sum_i d_i$ be the observed sum. The two-sided p-value is

$$
p=\frac{\#\{\text{sign patterns with }|S'|\ge |S|\}}{2^m}.
$$

If $j$ of the $m$ differences are negative then $|S'|=|m-2j|$, so the count is a sum of binomial coefficients and the test is exactly the sign test.

### Worked example

Forty items. B passes where A fails on 9 items, A passes where B fails on 3, both pass on 16, both fail on 12. A scores $19/40=0.475$, B $25/40=0.625$; $\bar d=+0.15$. Only $m=12$ items differ, observed $S=9-3=6$. The sign patterns with $|S'|\ge6$ are those with $j\le3$ or $j\ge9$ negatives: $2\,(1+12+66+220)=598$ out of $2^{12}=4096$, so $p=598/4096=0.146$. The bootstrap interval (seed 0, $B=2000$) is $(0.000,\,0.325)$. Both say the same thing: a 15-point gain on 40 items is **not** significant at the 5 percent level. It might be real; the test just cannot distinguish it from luck. With 12 wins against 2 losses the p-value drops below 0.05 (exercise check).

### How many items?

Suppose per-item differences have standard deviation $\sigma$ and you want to detect a true mean difference $\delta$ with 5 percent two-sided significance and 80 percent power. The mean of $n$ differences is approximately normal with standard error $\sigma/\sqrt n$. You reject when $\bar d$ exceeds $z_{\alpha/2}\,\sigma/\sqrt n$; to do so with probability 0.8 when the truth is $\delta$, the shift $\delta$ must sit $z_{\text{power}}$ standard errors beyond that threshold:

$$
\delta=(z_{\alpha/2}+z_{\text{power}})\frac{\sigma}{\sqrt n}\;\Longrightarrow\; n=\left(\frac{(z_{\alpha/2}+z_{\text{power}})\,\sigma}{\delta}\right)^{2},
$$

with $z_{\alpha/2}=1.96$ and $z_{\text{power}}=0.8416$. For $\delta=0.05$ and $\sigma=0.4$ (typical for 0/1 scores where about 16 percent of items differ), $n=\lceil(2.8016\cdot0.4/0.05)^2\rceil=503$. Halving $\delta$ quadruples $n$. Small improvements need big eval sets, or a metric with less noise.

## Regression testing and monitoring

- **Gate in CI.** Run the eval on every prompt or model change. Block when a deterministic check fails, or when the lower end of the confidence interval of the difference falls below a margin you accept.
- **Look at slices.** An average can rise while one slice collapses. The source tells of a prompt change that cut hallucinations but made answers 34 percent less complete, unnoticed for 11 days because only one metric was tracked. Track several metrics (correctness, completeness, format, latency, cost) and per-tag results.
- **Cost-gate the eval itself.** Judge calls cost money; run a small smoke set on every commit and the full set nightly.
- **Monitor production.** Log inputs, outputs, retrieval results, tool calls, latency, token cost and guardrail triggers ([[app.05]]); sample a fraction for judge or human review; watch for drift in input distribution and in refusal or retry rates; feed new failures back into the eval set.

## Implementation

```python
import numpy as np

def bootstrap_ci(d, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def sign_flip_pvalue(d):
    nz = d[d != 0]; m = len(nz)
    bits = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1) * 2 - 1
    return np.mean(np.abs((bits * nz).sum(axis=1)) >= abs(nz.sum()) - 1e-12)
```

Production code uses `scipy.stats` (`bootstrap`, `permutation_test`), evaluation frameworks such as RAGAS or DeepEval for ready-made judge prompts, and experiment trackers to store runs. For many items enumerating $2^m$ patterns is impossible; sample random sign patterns instead.

## Common misconceptions

- "The new prompt scored higher, so it is better." Only if the difference exceeds what noise produces on your set size.
- "A high judge-human agreement percentage means the judge is good." With skewed labels, chance agreement is high; use kappa.
- "More metrics is always better." Running many comparisons inflates false positives; decide the primary metric in advance.
- "Temperature 0 makes evaluation deterministic." Often not exactly (batching and hardware non-determinism), and it hides variance you will see in production.
- "The eval set is finished." It must grow with each production failure.

## Key takeaways

- Evaluate with a versioned, sliced eval set, a dev/held-out split and anchored rubrics.
- Use deterministic checks first, then reference metrics, then calibrated judges.
- Compare prompts as paired differences; report a bootstrap CI and, for small sets, an exact sign-flip p-value.
- Sample size scales as $(\sigma/\delta)^2$; a 15-point gain on 40 items was not significant.
- Gate changes in CI and watch slices and production metrics.

## Explain it yourself

1. Why does pairing the two prompts on the same items give a more sensitive comparison than computing two separate accuracies?
2. Your judge agrees with humans on 90 percent of items, but 90 percent of items are "good". What do you compute next and why?

## Further reading

- Lesson `sys.04` for RAG and LLM benchmark evaluation; RAGAS and DeepEval documentation for judge-based metrics.
- Efron and Tibshirani on the bootstrap, for the theory behind resampling.

## Mastery criteria

- You can build an eval set plan with splits and tags.
- You can compute a paired bootstrap interval and an exact sign-flip p-value from per-item scores.
- You can compute the sample size for a target effect.
- You can compute kappa and explain why accuracy misleads.
