---
id: sys.05
title: "Research practice: reading, reproducing and ablating"
summary: How to read a paper in three passes, reproduce a result, design a controlled experiment (seeds, baselines, variance), run a paired ablation with honest statistics, and spot weak claims, with a worked ablation on the moons MLP.
minutes: 80
teaches: [res.ablation]
requires: [ml.generalization, nn.training, llm.eval]
sources:
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (markdown cells, Sec. 4.4-4.6 and the results table): self-consistency recipe and the table of accuracies for combinations of CoT, top-p and voting on 500 problems. READ."
  - "reasoning-from-scratch/ch07/01_main-chapter-code/ch07_main.ipynb (markdown cells, Sec. 7.2-7.3): which training metrics to track, sanity metrics that should stay at zero, interpretation of curves. READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (results tables) and chF/01_main-chapter-code/chF_main.ipynb: accuracy on 10 vs 500 problems, devices and runs. READ."
  - "Designing Machine Learning Systems.pdf, Ch.6 (PDF p.201-214) baselines and evaluation methods, and Ch.9 'Test in Production' (PDF p.314-317): shadow deployment, A/B testing requirements (random assignment, enough samples), canary release. READ."
  - "content/authored/classical-ml/03-generalization/lesson.md (measuring protocol: train/validation/test, cross-validation; the experiment table): the discipline this lesson extends from one model to a comparison of two. READ (first half)."
---

## Where this fits

Sooner or later you will want to know whether an idea "works": a learning-rate warmup, a different initialisation, a trick from a new paper. The honest answer needs a small experiment, not a feeling. Research practice is the set of habits that turn "it seemed better" into evidence you could defend to a sceptical colleague (or to yourself in three months). It reuses the statistics of the previous lesson (intervals, paired comparisons) and the discipline of the generalisation lesson (never judge on data you tuned on), and applies them to the question every ML engineer faces: *did my change cause the improvement, or is it noise?*

## The idea in plain words

Imagine two bakers comparing recipes. One bakes recipe A on Monday in a cold oven and recipe B on Tuesday in a hot one, once each, and declares B better. Everything is confounded: oven, day, flour. A fair test bakes both recipes many times, in the same oven, alternating, with the same flour, and compares the *differences* batch by batch. An **ablation** is the same idea applied to a system: take the full system, remove (or change) exactly one component, and see what changes. If you change two things at once, you cannot say which one mattered.

Three more ideas:

- **Variance is real.** A neural network trained twice with different random seeds gives different results. Any claim from one run is a claim about one random draw.
- **Baselines give meaning.** A number is only interpretable next to a reference: the simple method, the previous best, chance.
- **Honesty is a method.** Report what you tried, including what did not work; say what the evidence supports and what it does not.

## Reading a paper in three passes

Reading everything in full detail is slow and most papers do not deserve it. A staged approach (the "three-pass" method many researchers use) saves time:

1. **First pass, 5 to 10 minutes.** Title, abstract, introduction, section headings, figures, conclusion. Goal: *what problem, what claim, is it relevant to me?* Decide whether to continue.
2. **Second pass, about an hour.** Read the text with attention to figures, tables and the experimental setup; skip proofs. Goal: understand the method well enough to explain it, and list the *claims* with the *evidence* offered for each (which table or figure).
3. **Third pass, hours.** Re-derive or re-implement mentally: what are the assumptions, what would I have done differently, where could it fail? This is the pass in which you reproduce part of the result.

At each pass keep a note with four lines: the claim, the evidence, the baseline, what is missing.

## Reproducing a result

Reproduction is the best way to understand a method and to calibrate your trust.

1. Start with the **smallest** version that should show the effect (a tiny model, a subset of data). The reasoning-from-scratch notebooks do this: they evaluate 10 problems for a quick demo and report the full 500-problem numbers separately; the two differ in how trustworthy they are.
2. Fix everything that can be fixed: seeds, data split, library versions, hardware notes. Write the command that reproduces the run.
3. Reproduce the **baseline** first. If you cannot get the published baseline number, you cannot trust a comparison against it.
4. Expect small mismatches. Compare using the paper's own variance if given; if not, run several seeds yourself.
5. **Sanity-check metrics.** The chapter on tracking training runs gives an example: with the group-relative advantage used there, the average advantage must stay at zero by construction; if it drifts, that points to a bug. Find such invariants in your own experiments (a loss that must start near $\ln K$ for $K$ classes, gradients that must match finite differences) and check them before interpreting results.
6. Log curves, not only final numbers: the same chapter notes that evaluation accuracy can rise and then fall during training, and that some metrics (like the loss) are less meaningful than others.

## Designing a controlled experiment

**Hypothesis.** State it so it can fail: "Linear warmup of the learning rate over 50 steps changes test accuracy of this MLP", not "warmup helps".

**Variables.** One *manipulated* variable per ablation (warmup on/off), everything else held fixed (data, architecture, step budget, learning rate, evaluation set). Pre-register the primary metric.

**Baseline.** The configuration without the change, tuned *as carefully* as the new one. A strong tuned baseline beats a fancy method against a neglected baseline; an unfair comparison is one of the commonest weaknesses in papers.

**Seeds and variance.** Run $n$ seeds per configuration (5 is a minimum for a quick check, more for small effects). Report mean $\pm$ standard deviation (sample std, $\text{ddof}=1$) and $n$. Say what the seed controls (initialisation, batch order, data split).

**Common random numbers (pairing).** Use the *same* seed for both configurations, so they share initial weights and batch order. Then the difference between configurations is measured *within* seeds, where much of the noise cancels. For paired differences $d_i=a_i-b_i$ over $n$ seeds:

$$
\bar d=\frac1n\sum_i d_i,\quad s_d=\sqrt{\tfrac{1}{n-1}\sum_i(d_i-\bar d)^2},\quad t=\frac{\bar d}{s_d/\sqrt n}.
$$

Compare $|t|$ with the Student critical value for $n-1$ degrees of freedom (2.776 for 4 d.o.f. at the two-sided 5 percent level). The unpaired (Welch) statistic uses each group's own variance, $t=\frac{\bar a-\bar b}{\sqrt{s_a^2/n+s_b^2/n}}$; when seed-to-seed noise is large compared with the effect, it is far less powerful.

**Statistical vs practical significance.** A tiny effect can be statistically solid and practically irrelevant; a big effect with 3 seeds can be a fluke. Always report the effect size (here, accuracy points) next to the test.

**Test set discipline.** Choose settings on validation data and touch the test set once (generalisation lesson). If you run 30 ablations and report the best on the test set, the test set has become a validation set.

**Test in production** (Huyen, Ch. 9): the offline protocol above cannot capture everything, so deployed systems add *shadow deployment* (run the new model in parallel, log but do not serve its predictions, at the cost of doubling inference), *A/B tests* (random assignment of traffic; the book stresses that assignment must really be random and the sample large enough, and that significance is not infallible), and *canary releases* (roll out to a small fraction first and abort if key metrics degrade).

## Worked example: does warmup or initialisation scale matter for the moons MLP?

**Setup.** Data: 200 training and 1000 test points from two noisy moons (noise 0.2, fixed data). Model: $2\to16\to1$, tanh hidden units, logistic output, minibatch SGD with batch 32, 300 steps, base learning rate 0.5. The seed controls the initial weights and the minibatch order. The initial weights are drawn as $W\sim\mathcal N(0,\sigma^2/\text{fan\_in})$ with $\sigma$ the "init scale". Five seeds, 0 to 4. All the numbers below come from the harness in the exercise.

**Ablation 1: warmup (0 vs 50 steps).**

| seed | no warmup | warmup 50 | difference |
|---|---|---|---|
| 0 | 0.911 | 0.904 | +0.007 |
| 1 | 0.873 | 0.873 | 0.000 |
| 2 | 0.900 | 0.893 | +0.007 |
| 3 | 0.935 | 0.929 | +0.006 |
| 4 | 0.918 | 0.908 | +0.010 |

Means: $0.907\pm0.023$ and $0.901\pm0.021$ (mean $\pm$ sample std). Look at the spread: the seed-to-seed std (about 0.02) is much larger than the gap (0.006), so the unpaired Welch statistic is only $t=0.43$ and the two means look indistinguishable. But the *paired* differences are consistently non-negative: $\bar d=0.006$, $s_d=0.0037$, $t=3.65>2.776$, with no-warmup better in 4 seeds and tied in 1. With 20 seeds the paired difference is $0.0053$ with $t=9.1$ (means 0.906 vs 0.901).

**How to read this honestly.** Without warmup is *slightly* better here, by about half an accuracy point, and the paired test says it is not luck. But the effect is tiny and has a boring explanation: warmup spends the first 50 of 300 steps at a reduced learning rate, so the warmed-up run has simply trained a bit less. This network is shallow, tanh-based and uses a gentle learning rate, so there is no early instability for warmup to cure. The correct claim is "at this learning rate and budget, warmup does not help and costs about 0.5 points", not "warmup is useless". Warmup is mostly justified when large learning rates or deep/attention models are unstable in the first steps, a regime this experiment does not test. As a quick extra probe, at learning rate 5.0 the harness gives $0.941\pm0.013$ without and $0.914\pm0.034$ with warmup: still no benefit, and larger variance. We did not tune learning rates separately for each arm, which is a limitation.

**Ablation 2: initialisation scale (0.1 vs 1.0).** Paired by seed, test accuracy is $0.864\pm0.005$ with $\sigma=0.1$ and $0.907\pm0.023$ with $\sigma=1$: a gain of $0.044$, $s_d=0.019$, $t=5.2$. This is a large effect compared with warmup. A natural suspicion is that small initialisation merely trains more slowly. We tested it by training 10 times longer (3000 steps): $\sigma=0.1$ reaches $0.867$ while $\sigma=1$ reaches $0.953$, so the gap does not close. At $\sigma=0.1$ the network stays at roughly the accuracy of a linear classifier on this data. A plausible mechanism is that tiny weights keep the tanh units in their near-linear region and the gradient reaching the first layer is multiplied by tiny second-layer weights, but we have not tested that mechanism directly, so we state it as a hypothesis. For completeness, $\sigma=3$ gives $0.943$ at 300 steps and $0.948$ at 3000 steps: larger is not better beyond a point (and the seed spread is a few points, so $0.953$ vs $0.948$ is not a difference we can claim).

**A real-world caution from the course material.** In the reasoning-model chapter on inference-time scaling, a table of accuracies on 500 problems lists one configuration, "top-p with self-consistency", with $n=3$ samples at 29.6 percent, $n=5$ at 27.8 percent and $n=10$ at 31.6 percent. More votes did not give monotonic improvement. With accuracy near 0.3 on 500 problems, the standard error is $\sqrt{0.3\cdot0.7/500}\approx0.02$, so the 1.8-point difference between $n=3$ and $n=5$ is within noise (and each row is a single run; the notebook does not report repeated seeds). It is exactly the kind of table where a careful reader asks for variance.

## Reporting honestly

- State the full protocol: data, split, seeds, steps, every hyper-parameter, and what was tuned for which arm.
- Report mean $\pm$ std over $n$ seeds, the effect size and an interval or test, and also the *negative* and *null* results.
- Show the baseline and the strongest simple alternative.
- Separate observations ("accuracy rose by 0.044") from interpretations ("because tanh saturates"), and mark interpretations that you did not test.
- Release code and seeds so that someone else can rerun your table.
- Do not hide the number of things tried. If you tried 20 variants and report the best, the reader needs to know (multiple comparisons inflate false positives).

## Spotting weaknesses in a paper (or in your own work)

1. A gain smaller than the seed noise, from a single run.
2. Baselines that are untuned, outdated, or given less compute.
3. Tuning on the test set, or an evaluation set that overlaps training data (contamination).
4. Comparing at unequal budgets (parameters, steps, tokens, wall-clock) without saying so.
5. Averages without slices, or metrics chosen after the results were seen.
6. No ablation: several changes introduced together and the credit given to the most fashionable one.
7. Claims that go beyond the settings tested (one dataset, one model size).
8. Missing details that would prevent reproduction.

## Implementation

```python
def run_seeds(cfg, seeds):
    return np.array([train_mlp(s, **cfg) for s in seeds])

def paired_stats(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return {"mean_diff": d.mean(), "sd_diff": sd,
            "t": d.mean() / (sd / np.sqrt(len(d))), "wins": int((d > 0).sum())}

# usage: same seeds for both arms = common random numbers
a = run_seeds({"warmup": 0},  range(5))
b = run_seeds({"warmup": 50}, range(5))
print(summarize(a), summarize(b), paired_stats(a, b))
```

## Shapes

| Object | Shape |
|---|---|
| training data | $X\in\mathbb{R}^{200\times2}$, $y\in\{0,1\}^{200}$ |
| $W_1,\ b_1$ | $(2,16)$, $(16,)$ |
| $W_2,\ b_2$ | $(16,1)$, $(1,)$ |
| minibatch | $x\in\mathbb{R}^{32\times2}$ |
| accuracies of one configuration | $(n_\text{seeds},)$ |
| paired differences | $(n_\text{seeds},)$ |

## Common misconceptions

1. "One run is enough if the number is good." It is one draw from a distribution; the spread may exceed the effect.
2. "Statistically significant means important." Here a 0.5 point effect is significant with pairing and still negligible.
3. "No significant difference means the two are equal." With 5 seeds you may simply lack power; report an interval and say what effect sizes you could have detected.
4. "Ablation means removing a feature and watching the accuracy drop." It means a controlled change of one factor with a fair, equally tuned baseline.
5. "Pairing is a trick." It is legitimate precisely because the arms share seeds; it is invalid if the pairs are not truly matched.

## Key takeaways

- Read in passes; write down claim, evidence, baseline and what is missing.
- Reproduce the baseline before testing a change.
- Change one thing at a time, hold the budget fixed, tune baselines fairly.
- Use several seeds, report mean $\pm$ std, and compare arms with common random numbers and a paired statistic.
- Separate effect size from significance; separate observation from interpretation.
- Report nulls and limitations; production adds shadow, A/B and canary tests on top of offline evaluation.

## Explain it yourself

1. In the warmup table the unpaired $t$ is 0.43 and the paired $t$ is 3.65 for the same data. Explain why pairing makes such a difference and what condition must hold for pairing to be valid.
2. Why is the statement "warmup is useless" not supported by the experiment, even though the paired test is significant in favour of no warmup?

## Further reading

- Keshav, "How to Read a Paper" (2007), the three-pass method.
- Huyen, *Designing Machine Learning Systems*, Ch. 6 and Ch. 9.
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021), on seeds and variance sources.
- Henderson et al., "Deep Reinforcement Learning that Matters" (2018), on reproducibility failures.

## Mastery criteria

- [ ] I can read a paper in three passes and list its claims with the evidence for each.
- [ ] I can design an ablation with one manipulated variable, fixed budget and a fair baseline.
- [ ] I can run several seeds and report mean $\pm$ std and a paired comparison.
- [ ] I can explain common random numbers and compute the paired $t$ statistic.
- [ ] I can distinguish what an experiment shows from what I merely suspect.
