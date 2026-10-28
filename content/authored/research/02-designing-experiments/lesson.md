---
id: res.02
title: "Designing and evaluating original experiments"
summary: From research question to hypothesis, variables, controls and baselines; sample size and power (derived and simulated); randomisation; multiple comparisons with Bonferroni and Benjamini-Hochberg; ablation design and interactions; compute budgets; pre-registration, p-hacking and honest reporting.
minutes: 95
teaches: [res.design]
requires: [res.reproduction, res.ablation, stat.testing]
sources:
  - "content/authored/llm-systems/05-research-practice/lesson.md (Designing a controlled experiment, Reporting honestly, Spotting weaknesses): the one-factor-at-a-time ablation, paired seeds and the checklist of weak papers that this lesson turns into a planning procedure. READ."
  - "content/authored/math/13-estimation-and-hypothesis-testing/lesson.md (Confidence intervals, Hypothesis tests and p-values, Pitfalls on multiple comparisons, Comparing two models fairly): the t interval, p-value definition and the 1-(1-alpha)^m calculation, used and extended here. READ."
  - "content/authored/classical-ml/03-generalization/lesson.md (measuring protocol: train/validation/test, cross-validation): why the test set is touched once, the root of the forking-paths problem. READ."
  - "content/authored/research/01-reproducing-a-result/lesson.md (the previous lesson, res.01: report template, moons dropout reproduction with 10 seeds and the resampled-data variant): the experiment whose variance is used for the sample-size example. READ."
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (markdown results table on MATH-500, 500 problems; accuracies and run times of 12 configurations, e.g. self-consistency n=3: 29.6 percent / 97.6 min, n=5: 27.8 percent / 116.8 min, n=10: 31.6 percent / 300.4 min, and the combination with CoT up to 862.6 min): used for the sampling error of an accuracy, the minimum detectable difference on 500 problems and compute budgeting. READ."
  - "Designing Machine Learning Systems.pdf, Ch.6 (PDF p.201-205 Model Offline Evaluation, Baselines: random, simple heuristic, zero-rule, human, existing solutions; p.208 slice-based evaluation, majority/minority example): baselines that give a number meaning and the danger of coarse aggregate metrics. READ."
---

## Where this fits

[[res.01]] checked somebody else's claim. Now you want to ask your *own* question: does a new learning-rate schedule help? Is my retrieval change worth the extra latency? The difficulty is that you now control every choice, and every choice is a chance to fool yourself. You could run twenty variants, look at twenty metrics on three data subsets, and almost certainly find something "significant". Experiment design is the discipline of deciding the question, the comparison, the number of runs and the analysis **before** you see the data, so that the result means what it appears to mean.

This lesson builds on the statistics of [[math.13]] (intervals, p-values, the multiple-comparison warning), on the generalisation protocol of [[cml.03]] and on the paired-seed ablation of [[sys.05]]. It goes further in four directions: how many runs you need (power), what to do when you test many things (Bonferroni, Benjamini-Hochberg), how to arrange an ablation (grids and interactions), and how to stay honest when the answer is "nothing happened".

## The idea in plain words

Picture a taste test between two brands of coffee. A bad design: you try each once, on different mornings, and decide from how you felt. A good design: you decide in advance what "better" means (taste score out of ten), you have a friend pour the cups in random order without telling you which is which, you taste enough pairs that a real difference would show up, and you write down the verdict rule before the first sip. Then something subtler happens when you are allowed to peek. If after each cup you can decide to stop, or switch the question to "which smells better?", or to "which is better for tall people only?", you will eventually find a difference even between two cups from the *same* pot. That is not dishonesty about the coffee, it is the arithmetic of lots of chances.

Five ideas carry the lesson:

- **A question that can be answered.** A hypothesis says what would count as the answer and what would count against it.
- **Controls and baselines.** Change one thing, keep everything else the same, and compare against a baseline that was given an equal chance.
- **Enough samples.** Too few runs and you will miss real effects (low power). Compute the number you need *before* running.
- **Counting your chances.** Every extra comparison is a lottery ticket. Either buy fewer tickets (pre-register) or raise the bar for each (multiple-comparison corrections).
- **Report the whole story.** Including what you tried, what failed, and how big the effect is, not just whether it crossed a line.

## From question to hypothesis

Write the chain explicitly:

1. **Research question** (open): "Does cosine learning-rate decay help this model?"
2. **Hypothesis** (specific, directional if you can defend it): "Cosine decay to zero over the full run gives higher mean validation accuracy than a constant rate with the same peak value, for this model and dataset."
3. **Operational definitions**: metric (validation accuracy at the final step), unit of replication (a full training run with its seed), what counts as "higher" (a paired difference whose 95 percent interval excludes 0), and the smallest difference you would care about, the **minimum effect of interest** $\delta$ (say 0.5 accuracy points).
4. **Decision rule** written before running: "Adopt cosine decay if the paired interval excludes 0 *and* the mean gain is at least $\delta$."

The minimum effect of interest is not decoration. It tells you how many runs to plan (next sections) and prevents the classic ending "statistically significant, practically irrelevant".

## Variables, controls and baselines

- **Independent variable (factor)**: what you manipulate (the schedule).
- **Dependent variable (outcome)**: what you measure (validation accuracy).
- **Control variables**: everything held fixed (model, data, steps, batch size, peak learning rate, evaluation set, precision). Write them down; this is the setup table of [[res.01]].
- **Nuisance variables you cannot fix**: random initialisation, batch order, dropout masks, data split. You do not remove them; you **randomise** over them with seeds and average.
- **Confounders**: a variable that changes together with the factor and could explain the result. If the cosine runs also got more steps, steps is a confounder.

**Baselines.** A number is only meaningful next to a reference. The offline-evaluation chapter of Huyen (Ch.6, PDF p.201-205) lists the ones to think of: a random baseline (given the label distribution, a classifier that guesses gets accuracy 0.82 and F1 0.1 on a 90/10 problem, versus accuracy 0.5 and F1 0.167 when guessing uniformly), a simple heuristic, the zero-rule baseline that always predicts the majority class, a human baseline and the existing solution. Add the **strongest simple alternative** and the **previous best** for your own comparison. The baseline gets the same tuning effort as the new method; an under-tuned baseline is the commonest way to manufacture a win.

**Slices.** The same chapter (p.208) warns against living on one aggregate: in its example a model with 98 percent accuracy on a majority group (90 percent of the data) and 80 percent on a minority has 96.2 percent overall, ahead of a model with 95 percent on both groups (95 percent overall). Decide before running which slices you will report; adding slices after the fact is a forking path (below).

## Randomisation and seeds

Randomise whatever you cannot control, and use the same random draws across arms where you can.

- **Seeds as replicates.** One full run with one seed is one observation. The runs, not the mini-batches or the evaluation examples, are the independent units for a claim about the *training procedure*.
- **Common random numbers.** Give both arms the same seeds (same initial weights, same batch order). The per-seed difference $d_i=a_i-b_i$ cancels shared noise, which shrinks the variance of the comparison (the paired statistic of [[sys.05]]).
- **Randomise the order of execution** across arms when hardware or time-of-day effects could matter (a shared machine that gets slower in the afternoon should not always run arm B last).
- **Decide what a seed varies** and say so (initialisation only, or also the data split). In [[res.01]] widening the seed to resample the dataset changed the verdict; the right choice depends on the population you want to speak about.
- **Evaluation noise is separate.** On a fixed test set of $N$ examples, accuracy $\hat p$ has sampling error $\sqrt{\hat p(1-\hat p)/N}$ ([[math.13]]). It adds to run-to-run noise rather than replacing it.

## Sample size and power

**The two errors.** With a null hypothesis $H_0$ (no difference) and an alternative $H_1$ (a difference $\delta$):

- Type I error: reject $H_0$ when it is true; its probability is the significance level $\alpha$ (typically 0.05).
- Type II error: fail to reject $H_0$ when $H_1$ is true; its probability is $\beta$.
- **Power** is $1-\beta$: the probability of detecting a real difference of size $\delta$. A common target is 0.8.

### Derivation for a two-sample comparison

Take two independent groups of $n$ runs each. Group A has outcomes $X\sim\mathcal N(\mu_A,\sigma^2)$, group B has $Y\sim\mathcal N(\mu_B,\sigma^2)$ with the same standard deviation $\sigma$, and $\delta=\mu_B-\mu_A$.

1. The difference of sample means $D=\bar Y-\bar X$ has mean $\delta$ and variance $\sigma^2/n+\sigma^2/n=2\sigma^2/n$. Write the standard error $\mathrm{se}=\sigma\sqrt{2/n}$.
2. Standardise: $Z=D/\mathrm{se}$. Under $H_0$ ($\delta=0$) it is $\mathcal N(0,1)$, so the two-sided test rejects when $|Z|>z_{1-\alpha/2}$, where $z_q$ is the $q$-quantile of the standard normal ($z_{0.975}=1.960$).
3. Under $H_1$, $Z=(D-\delta)/\mathrm{se}+\delta/\mathrm{se}$ is $\mathcal N(\delta/\mathrm{se},\,1)$: the null bell curve shifted right by $\delta/\mathrm{se}$.
4. Power is the probability that this shifted variable exceeds the cut-off (ignoring the negligible chance of landing below $-z_{1-\alpha/2}$):

$$
1-\beta \;\approx\; P\big(Z>z_{1-\alpha/2}\big)=\Phi\!\Big(\frac{\delta}{\mathrm{se}}-z_{1-\alpha/2}\Big).
$$

5. Apply $\Phi^{-1}$ to both sides: $z_{1-\beta}=\delta/\mathrm{se}-z_{1-\alpha/2}$, hence $\delta/\mathrm{se}=z_{1-\alpha/2}+z_{1-\beta}$.
6. Substitute $\mathrm{se}=\sigma\sqrt{2/n}$ and solve for $n$:

$$
n \;=\; \frac{2\,\sigma^{2}\,\big(z_{1-\alpha/2}+z_{1-\beta}\big)^{2}}{\delta^{2}}\quad\text{per group.}
$$

With $\alpha=0.05$ and power 0.8, $z_{1-\alpha/2}=1.960$ and $z_{0.8}=0.842$, so $(1.960+0.842)^2=7.85$ and $n\approx 15.7\,(\sigma/\delta)^2$. Read the formula: halving the effect you want to detect **quadruples** the runs; doubling the noise quadruples them too. For a **paired** design with differences of standard deviation $\sigma_d$, the same steps give $n=(z_{1-\alpha/2}+z_{1-\beta})^2\sigma_d^2/\delta^2$ pairs.

The formula uses the normal distribution; with small $n$ the test uses a $t$ distribution with heavier tails, so the true power is slightly lower. That is why we also **simulate**.

### Simulation-based power

When the test or the data model is awkward, estimate power directly: simulate the experiment many times under $H_1$ and count how often it rejects.

```python
def simulated_power(delta, sigma, n, n_sims=2000, seed=0):
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sims):
        a = rng.normal(0.0, sigma, n)
        b = rng.normal(delta, sigma, n)
        t, df = welch_t(a, b)                      # Welch statistic and degrees of freedom
        rejections += abs(t) > t_crit(int(round(df)))
    return rejections / n_sims
```

Setting `delta=0` estimates the type I error instead; it should come out near $\alpha$. This is a good test of the simulator itself.

### Numbers (reproduced by the first exercise)

For $\delta/\sigma=0.5$ (a "medium" effect) the formula gives $n=63$ per group; for $\delta/\sigma=0.2$ it gives $393$, and for $0.8$ it gives $25$. Simulated power with 4,000 simulations:

| $n$ per group | 10 | 32 | 63 | 100 |
|---|---|---|---|---|
| power at $\delta/\sigma=0.5$ | 0.167 | 0.499 | 0.798 | 0.936 |

And with $\delta=0$ and $n=10$ the rejection rate is 0.046, close to the nominal 0.05. The headline: ten runs per arm detect a half-sigma effect about one time in six. A "not significant" result from 10 seeds is therefore weak evidence of no effect, a point made in [[sys.05]] that now has a number.

**Using the previous lesson's experiment.** When the dataset was resampled for each seed, the paired differences of the gap had mean 0.0082 and standard deviation $\sigma_d=0.0259$. The paired formula gives $n=7.85\times0.0259^2/0.0082^2\approx 80$ pairs for 80 percent power; ten seeds were far too few to settle that question.

**Evaluation sets are small samples too.** In the reasoning-from-scratch evaluation table (500 MATH-500 problems), accuracies cluster around 0.3. The sampling error of one accuracy is about $\sqrt{0.3\times0.7/500}=0.020$ (2 points). For two independent configurations the standard error of the difference is $\sqrt{2\times0.21/500}=0.029$, so at 80 percent power the *minimum detectable difference* is about $2.80\times0.029\approx 8$ points. Seen this way the table's self-consistency rows (29.6 percent at $n=3$, 27.8 percent at $n=5$, 31.6 percent at $n=10$, each from a single evaluation) differ by 2 to 4 points, well inside what 500 problems and one sampling seed can resolve, so none of those orderings is established. To detect a 1.8-point difference with the same formula one would need about 10,000 problems per arm. (Evaluating on the same problems and pairing helps; the calculation here is the unpaired worst case.)

## Multiple comparisons

If you test $m$ hypotheses at level $\alpha$ and all nulls are true, the chance of at least one false positive (the **family-wise error rate**, FWER) is $1-(1-\alpha)^m$ when tests are independent ([[math.13]]): 0.226 for $m=5$, 0.642 for $m=20$.

### Bonferroni

Test each hypothesis at $\alpha/m$. By the union bound $P(\bigcup_i A_i)\le\sum_i P(A_i)$, the FWER is at most $m\cdot\alpha/m=\alpha$, with no independence assumption. It is simple and safe, and conservative when $m$ is large.

### Benjamini-Hochberg (BH)

Sometimes you do not need zero false positives, only a controlled *fraction*. The **false discovery rate** (FDR) is the expected proportion of false positives among the rejections. The Benjamini-Hochberg step-up procedure at level $q$:

1. Sort the p-values ascending: $p_{(1)}\le p_{(2)}\le\dots\le p_{(m)}$.
2. Find the largest rank $k$ with $p_{(k)}\le q\,k/m$.
3. Reject the hypotheses with the $k$ smallest p-values (none if no rank passes).

Its guarantee (for independent p-values, and under some positive dependence): FDR $\le q$. The thresholds $qk/m$ grow with rank, so the procedure is looser than Bonferroni for the ordered tail. "Step-up" means a larger p-value passing its threshold rescues all smaller ones, even if some of those fail their own thresholds.

**Worked example.** Eight p-values from eight ablations: 0.001, 0.008, 0.012, 0.021, 0.034, 0.041, 0.30, 0.62.

| Rank | $p_{(k)}$ | BH threshold $0.05k/8$ | Pass? |
|---|---|---|---|
| 1 | 0.001 | 0.00625 | yes |
| 2 | 0.008 | 0.01250 | yes |
| 3 | 0.012 | 0.01875 | yes |
| 4 | 0.021 | 0.02500 | yes |
| 5 | 0.034 | 0.03125 | no |
| 6 | 0.041 | 0.03750 | no |
| 7 | 0.30 | 0.04375 | no |
| 8 | 0.62 | 0.05000 | no |

The largest passing rank is 4, so BH rejects the four smallest p-values. Bonferroni uses the single threshold $0.05/8=0.00625$ and rejects only $p=0.001$. Uncorrected testing at 0.05 would "find" six effects. Which to use? Bonferroni when one false positive is costly (a claim in a paper that must stand alone); BH when you screen many candidates and can tolerate a fraction of false leads that you will confirm later. (Holm's step-down procedure keeps Bonferroni's guarantee while rejecting at least as many hypotheses; we do not implement it here.)

## Ablation design

An **ablation** removes or changes a component to find out how much it contributes ([[sys.05]]). Two layouts:

- **One factor at a time (OFAT)**: start from the baseline and change one factor per run. For factors with $l_1,\dots,l_k$ levels this needs $1+\sum_i (l_i-1)$ configurations. Cheap, easy to read, but blind to interactions.
- **Full grid (factorial)**: every combination, $\prod_i l_i$ configurations. It reveals interactions and costs exponentially more.

**Interactions.** Two factors interact when the effect of one depends on the level of the other. In a $2\times2$ table of mean results $y_{ij}$ (factor A at level $i$, factor B at level $j$, level 0 is "off"), the interaction is

$$
I=(y_{11}-y_{10})-(y_{01}-y_{00}).
$$

Example: baseline $y_{00}=0.80$, A alone $y_{10}=0.80$, B alone $y_{01}=0.80$, both $y_{11}=0.90$. OFAT would report that neither A nor B does anything and discard both; the grid shows a $+0.10$ gain only when both are on, $I=0.10$. If the effects were additive (0.70, 0.75, 0.80, 0.85) then $I=0$.

**Planning table.** Four factors with 2, 2, 3 and 2 levels, 5 seeds, 3 minutes per run:

| Layout | Configurations | Runs | Minutes |
|---|---|---|---|
| OFAT | $1+1+1+2+1=6$ | 30 | 90 |
| Full grid | $2\cdot2\cdot3\cdot2=24$ | 120 | 360 |

With a 240-minute budget the grid does not fit; it would fit with 3 seeds ($24\times3\times3=216$ minutes), but 3 seeds give a very wide interval. A practical compromise is OFAT first to find the factors that matter, then a full grid only over those; fractional factorial designs (running a carefully chosen subset of the grid) are the standard next step if you need more (not implemented here). Remember that each configuration needs the *same* number of seeds, and that each extra comparison enters the multiple-comparison count.

## Compute budgeting

Total cost $=\text{configurations}\times\text{seeds}\times\text{minutes per run}$. Allocate in stages:

1. **Pilot** (about 10 percent of the budget): a few seeds on the smallest configuration to estimate $\sigma$, check for bugs, and verify the metric.
2. **Power calculation** with the pilot's $\sigma$: how many seeds does your minimum effect of interest need? If the answer exceeds the budget, change the design: pair the arms, increase $\delta$ (ask a bigger question), reduce noise (more evaluation data, lower-variance metric), or accept that the experiment can only exclude large effects and say so.
3. **Main experiment**, with the analysis script written and tested on the pilot data.
4. **Reserve** for follow-ups (confirmation of a surprising result).

The evaluation table in the reasoning-from-scratch chapter shows how fast cost grows when quality is bought with samples: self-consistency with 3, 5, and 10 samples plus chain-of-thought took 211.6, 452.9 and 862.6 minutes for 42.2, 48.0 and 52.0 percent accuracy, in single evaluations on 500 problems. Even an honest comparison of these rows with several seeds each would multiply the cost by the number of seeds, which is why the design must fix the question before the budget is spent.

## Pre-registration

**Pre-registration** means writing down, with a timestamp (a commit is enough for solo work), the hypothesis, the metric, the data and the exclusion rules, the sample size, the comparisons, the correction for multiple comparisons and the decision rule, *before* you look at outcomes. Exploratory analyses are allowed and valuable, but they are labelled as exploratory. The benefit is not bureaucracy; it removes your freedom to choose the analysis after seeing the data, which is the engine of the next section.

A minimal pre-registration for a solo ML experiment:

```text
question        Does cosine decay raise final validation accuracy vs constant lr?
hypothesis      mean paired gain >= 0.5 points (minimum effect of interest)
design          2 arms, same peak lr, same steps; seeds 0..N-1 shared by both arms
sample size     N from power analysis (sigma_d from pilot): N = ...
primary metric  validation accuracy at final step (single, pre-declared)
secondary       training loss; reported as exploratory
analysis        paired t interval; adopt if interval excludes 0 and mean >= 0.5
multiplicity    1 primary comparison; secondary p-values via BH at q = 0.05
stop rule       fixed N; no interim looks
deviations      logged in the report
```

## p-hacking and the garden of forking paths

**p-hacking** is any flexible analysis that increases the chance of reaching "significant". The **garden of forking paths** names the subtler version: you never run twenty tests deliberately; at each step you make a reasonable-looking choice (which metric, which subset, drop that outlier seed, try a bigger learning rate for the new method, stop early because it looks promising), and the choice depends on the data. The number of analyses you *could* have run, had the data looked different, is what counts, not the number you ran.

**Simulation.** Make the world boring: no true effect at all, two groups of 10 draws from $\mathcal N(0,1)$. A disciplined analyst tests one pre-registered outcome. A flexible analyst has $k$ independent outcomes ("forks") and reports the study if any is significant at 5 percent. The simulation (3,000 simulated studies, Welch test) gives:

| forks $k$ | pre-registered fork significant | any fork significant | $1-0.95^k$ |
|---|---|---|---|
| 1 | 0.047 | 0.047 | 0.050 |
| 5 | 0.046 | 0.219 | 0.226 |
| 20 | 0.043 | 0.621 | 0.642 |

The pre-registered analysis keeps its 5 percent error rate; the flexible one reports a "discovery" in about three studies out of five when nothing exists. The match with $1-(1-\alpha)^k$ confirms the arithmetic (the small shortfall is simulation noise and the discreteness of the $t$ cut-offs). Real forks are correlated (the same data underlie different metrics), which usually makes the inflation smaller than for independent forks but still far above 5 percent.

Typical forks in ML work: choose the best of several seeds, tune the new method's hyper-parameters but not the baseline's, evaluate on many checkpoints and report the best on the test set, pick the benchmark subset after seeing results, change the metric because the original one was "noisy". Defences: pre-registration, a single held-out test evaluation ([[cml.03]]), reporting every variant you tried and applying a multiplicity correction to the whole family, and out-of-sample confirmation of any surprise.

## Negative results and honest reporting

- **A null result is a result.** "No difference detected with 10 seeds, interval $[-0.8,+0.6]$ points" says that effects larger than about one point are unlikely; it does not say the effect is zero. Quote the interval and the **minimum detectable effect**.
- **Report the denominator.** How many variants, metrics, subsets and seeds did you try? The reader needs the count to judge multiplicity.
- **Separate confirmatory from exploratory** results, and mark interpretations you did not test.
- **Report effect sizes with intervals,** not only p-values or stars.
- **Fail loudly.** If the planned experiment was underpowered, say so rather than switching to a more flattering analysis.
- **Publish the code, seeds and configs** so that the experiment can be reproduced as in [[res.01]].

Doing this is not only ethical hygiene; it is what makes your own later work trustworthy to you.

## Implementation: the planning functions

```python
def bonferroni(p, alpha=0.05):
    p = np.asarray(p, float)
    return p <= alpha / len(p)

def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    passing = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    reject = np.zeros(m, bool)
    if len(passing):
        reject[order[: passing.max() + 1]] = True
    return reject

def ablation_plan(levels, seeds, minutes_per_run, budget_minutes):
    ofat = 1 + sum(l - 1 for l in levels); grid = int(np.prod(levels))
    return {"ofat_configs": ofat, "grid_configs": grid,
            "ofat_minutes": ofat * seeds * minutes_per_run,
            "grid_minutes": grid * seeds * minutes_per_run,
            "grid_fits": grid * seeds * minutes_per_run <= budget_minutes,
            "max_seeds_grid": int(budget_minutes // (grid * minutes_per_run))}
```

Production tools: `scipy.stats` and `statsmodels` provide power calculators, t-tests and multiple-testing corrections (`multipletests` with the `fdr_bh` method); experiment trackers store configs and seeds. Writing the few lines yourself, as the two exercises do, is the way to understand what those functions promise.

## Shapes

| Object | Shape |
|---|---|
| outcomes of one arm | $(n,)$ |
| p-values of a family | $(m,)$; BH returns a boolean array $(m,)$ in the original order |
| $2\times2$ table of means | $(2,2)$, indexed `[level of A][level of B]` |
| grid of results | $(l_1,\dots,l_k,\ n_\text{seeds})$ |

## Common misconceptions

1. "Not significant means no effect." With 10 runs per arm a half-sigma effect is detected only about one time in six; the absence of significance is mostly absence of power.
2. "A smaller p-value means a bigger effect." It means the data are more surprising under $H_0$; with enough runs a negligible effect gets a tiny p-value.
3. "Bonferroni and BH control the same thing." Bonferroni controls the probability of any false positive; BH controls the expected fraction of false positives among rejections.
4. "OFAT ablations tell you each component's contribution." Only if components do not interact; the $2\times2$ example above has two components that each look useless alone.
5. "I only ran one test, so there is no multiplicity." If the choice of that test depended on the data you saw, the relevant count is the number you could have run.
6. "Pre-registration forbids exploration." It separates exploration from confirmation; it does not prevent either.
7. "Power analysis needs the true effect, which I do not know." You use the *smallest effect you would care about*, plus a variance estimate from a pilot; that is a decision, not a guess about nature.

## Key takeaways

- State the hypothesis, metric, minimum effect of interest and decision rule before running.
- Control everything except the factor; randomise nuisance variables with seeds; use common random numbers.
- $n=2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2/\delta^2$ per group; halving $\delta$ quadruples $n$. Simulate power when in doubt, and check that the simulator gives $\alpha$ under $H_0$.
- Many comparisons need a correction: Bonferroni (FWER) or Benjamini-Hochberg (FDR).
- OFAT is cheap but misses interactions; grids reveal them at multiplicative cost; budget runs as configurations x seeds x minutes.
- Flexible analysis inflates false positives toward $1-(1-\alpha)^k$; pre-registration and a single confirmatory test prevent it.
- Report nulls with intervals and minimum detectable effects, and say how many things you tried.

## Explain it yourself

1. Your pilot gives $\sigma_d=0.02$ for the paired accuracy difference between two training recipes and you care about $\delta=0.005$. Compute the number of seeds for 80 percent power and explain what you would do if it exceeds your budget.
2. Explain, using the forking-paths simulation, why an analyst who honestly reports "I tried several metrics and one was significant at $p=0.03$" has not provided evidence of an effect, and what the paper should have said.

## Further reading

- Benjamini and Hochberg, "Controlling the False Discovery Rate" (1995), the original BH paper.
- Gelman and Loken, "The garden of forking paths" (2013).
- Ioannidis, "Why Most Published Research Findings Are False" (2005).
- Cohen, *Statistical Power Analysis for the Behavioral Sciences*, on effect sizes and power.
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021).

## Mastery criteria

- [ ] I can go from a research question to a hypothesis, operational metric, minimum effect of interest and decision rule.
- [ ] I can derive $n=2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2/\delta^2$ and say why it scales as $1/\delta^2$.
- [ ] I can estimate power by simulation and verify the simulator at $\delta=0$.
- [ ] I can apply Bonferroni and Benjamini-Hochberg by hand and in code, and explain which guarantee each gives.
- [ ] I can choose between OFAT and a grid, detect an interaction in a $2\times2$ table, and budget the runs.
- [ ] I can show with a simulation how flexible analysis inflates false positives and write a pre-registration.
- [ ] I can report a null result with an interval and a minimum detectable effect.
