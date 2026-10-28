---
id: res.01
title: "Reproducing a published result end to end"
summary: How to choose a claim, extract its exact setup, plan the compute, reproduce it at small scale over many seeds, compare with the reported number using intervals and tolerances, document deviations and diagnose failures, with a full NumPy reproduction of "dropout narrows the generalisation gap".
minutes: 85
teaches: [res.reproduction]
requires: [res.ablation, stat.testing, nn.training, dl.dropout]
sources:
  - "content/authored/llm-systems/05-research-practice/lesson.md (whole file): the three-pass reading, the short reproduction checklist and the paired-seed ablation that this lesson deepens without repeating. READ."
  - "content/authored/classical-ml/03-generalization/lesson.md (measuring protocol, experiment table) and content/authored/math/13-estimation-and-hypothesis-testing/lesson.md (Sec. Confidence intervals, Hypothesis tests, Comparing two models fairly): the Student interval, paired differences and multiple-comparison warnings used here. READ."
  - "content/authored/ml/11-training-from-scratch/lesson.md (Worked example: the half-moons): the moons problem and the idea of a small harness around a training loop; the model in this lesson is a variant (tanh, 64 hidden units, 60 training points) chosen so that it overfits. READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (markdown results table): the same 10-problem evaluation of the base model gives 30 percent on CPU, CUDA and XPU but 20 percent on MPS, an example of hardware-dependent results. READ."
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (code cells with torch.manual_seed(123) before every sampling demo; self_consistency_vote(seed=...) which seeds each sample as seed+i+1; markdown on stochastic decoding): how a stochastic procedure is made repeatable only by seeding it explicitly. READ."
  - "Designing Machine Learning Systems.pdf, Ch.6 (PDF p.190-191 Experiment Tracking and Versioning; p.201-205 Model Offline Evaluation and Baselines): artifacts, versioning code and data so that a promising run can be recreated, and why a number needs a baseline to mean anything. READ."
---

## Where this fits

A paper says "method X improves metric Y by 2 points". Before you build on it, you would like to know whether that is true on your hardware, with your code. Reproduction is how you turn somebody else's sentence into evidence you own. It is also the best way to learn a method: you cannot reproduce what you do not understand, and the places where the reproduction disagrees with the paper are where the interesting lessons hide.

[[sys.05]] gave you the short version: read the paper in three passes, reproduce the baseline first, expect small mismatches, and run a paired ablation. This lesson goes deeper on the part that lesson only listed. How do you pick a claim that is worth reproducing? How do you extract the *complete* setup from a paper that omits half of it? How do you decide, with numbers, that your result and the paper's result agree? What do you write down so that the next person (usually you, in six months) can trust your verdict? We use the statistical tools of [[math.13]] (intervals, paired differences) and the generalisation protocol of [[cml.03]], and the training loop of [[f2n.11]].

After this lesson the follow-up [[res.02]] turns the same machinery around: instead of checking somebody else's claim you design your own experiment so that it can fail honestly.

## The idea in plain words

Imagine a friend gives you a cake recipe and says "it always rises beautifully". You follow it once and the cake is flat. What does that tell you? Maybe the recipe is bad. Maybe your oven runs cold. Maybe the recipe forgot to say the eggs must be at room temperature. Maybe you got unlucky this one time. To find out you do not bake the same flat cake ten more times; you bake several cakes, write down everything you did, change one thing at a time, and compare the *spread* of your results to what the friend promised.

Reproducing a result is exactly that, with five habits:

- **Pick a claim, not a paper.** A paper contains many claims. You reproduce one sentence with one number or one direction ("A beats B"), because that is a question with an answer.
- **Write the recipe before cooking.** Data, model, hyper-parameters, metric, how many seeds, which hardware. Anything you cannot find in the paper becomes an explicit "I assumed this".
- **Start small.** A scaled-down version that should show the same *qualitative* effect is cheap enough to run many times.
- **Compare ranges, not points.** Your ten runs give a range. The paper's single number is one draw from its own range. They agree when the paper's number is plausible under your range, not when the two decimals match.
- **Write down what differs.** A reproduction that ends with "close enough" and no list of deviations teaches nobody anything.

## What counts as "reproduced"? A vocabulary

Researchers use several words loosely; fix them before you start:

| Term | Same code | Same data | Question answered |
|---|---|---|---|
| Rerun (repeatability) | yes | yes | Does the original script give the same number twice on the same machine? |
| Reproduction | yes (theirs) | yes | Does their code and data give their number on my machine? |
| Replication | no (mine) | yes or new | Does the *finding* survive an independent implementation? |
| Generalisation of the finding | no | different | Does the claim hold on other data or scale? |

This lesson is about the second and third rows, run at a deliberately small scale. A small-scale reproduction can usually support only a **qualitative** verdict ("the gap narrows", "A converges faster than B") and a **quantitative** one only when you can match the setup closely. Saying so in the report is part of the job.

## Step 1: choose the claim

Good claims for a first reproduction have four properties:

1. **Specific.** "Dropout of 0.5 reduces the train-test accuracy gap of a one-hidden-layer network" rather than "dropout helps".
2. **Falsifiable at your scale.** If it needs 1,000 GPU-hours, shrink the question (smaller model, subset of data) or pick another claim.
3. **Measurable with a metric you can compute exactly.** Train-test gap, accuracy, loss at step $k$, steps to reach a threshold.
4. **Interesting if false.** Some claims are folklore everyone relies on; confirming them is cheap and valuable.

Write the claim as a sentence of the form *"In setting S, configuration A has a (lower/higher) metric M than configuration B, by about Δ."* The "by about Δ" part is optional but turns a vague direction into a testable number.

## Step 2: extract the exact setup

Read the paper's method, experiments and appendix with a table open and fill every row. Rows that remain empty are your *unreported details*.

| Field | What to record | Typical gap in papers |
|---|---|---|
| Data | source, version, size, preprocessing, split sizes, who made the split | exact split or filtering rule unstated |
| Model | architecture, size, initialisation | init scheme or normalisation placement unstated |
| Training | optimiser, learning-rate schedule, batch size, steps or epochs, regularisation | warmup length, weight-decay exclusions |
| Metric | exact definition, averaging, which split it is measured on | macro vs micro average, tuned threshold |
| Seeds | how many, what each seed controls | usually one, or not mentioned |
| Compute | hardware, precision, library versions, wall-clock | rarely given |
| Baselines | which, how they were tuned | baseline tuned less than the proposed method |

For the details you cannot find, in order of preference: look in the released code, then the appendix, then related work by the same authors, and finally choose a default and **log it as a deviation**. Huyen's chapter on experiment tracking (Ch.6, PDF p.190-191) makes the same point from the engineering side: keep every definition needed to recreate an experiment and its artifacts (logs, curves, intermediate results), and version the code *and* the data, because a promising run you cannot recreate is worth nothing.

## Step 3: plan the compute

Estimate cost before you commit. If one run takes $c$ minutes, the plan has $R$ runs, and your budget is $B$ minutes, you need $R\,c \le B$. The runs are

$$
R = (\text{configurations}) \times (\text{seeds}) .
$$

For a two-arm comparison with 10 seeds that is 20 runs. If the full-scale run is out of reach, reduce **one** thing at a time and note it: fewer steps, a smaller width, a subset of data. Scale-down is itself a deviation, because some effects (for example, large-model emergent behaviour) vanish when you shrink. A useful sanity check is to run the smallest case first and verify that it produces *something* resembling the published trend before you invest in more seeds.

Also decide now what you will do with the answer. If the claim is reproduced you build on it; if not, you will spend time diagnosing. Set a time limit for diagnosis, otherwise a failed reproduction swallows a week.

## Step 4: compare with the paper's number

Suppose the paper reports a number $r$ and you obtain $n$ runs $x_1,\dots,x_n$ with sample mean $\bar x$ and sample standard deviation $s$ (with $n-1$ in the denominator). The 95 percent Student interval from [[math.13]] is

$$
\bar x \pm t_{n-1,\,0.975}\,\frac{s}{\sqrt n}.
$$

Three outcomes are worth distinguishing:

1. **Reproduced**: $r$ lies inside the interval. The paper's number is a plausible draw from the process you measured.
2. **Within tolerance**: $r$ lies outside the interval but $|\bar x - r| \le \tau$ where $\tau$ is a tolerance you fixed *before* running (for example "within one percentage point", or "within the paper's own reported standard deviation"). Say that it is within tolerance, not that it is reproduced.
3. **Not reproduced**: neither holds. This is a finding, not a failure, as long as you can say what you checked.

Two cautions. First, the interval describes the uncertainty of **your mean**; the spread of single runs is $s$, which is $\sqrt n$ times wider. If the paper reports *one* run, the right comparison for "is the paper's run typical?" is against $\bar x \pm 2s$, not against the narrow interval for the mean. Second, a wide interval is not a reproduction. With $n=3$ the critical value is $4.30$ and almost any $r$ falls inside; "reproduced" is only meaningful when the interval is narrow enough to rule out something. Always report the interval together with the verdict.

For a **qualitative** claim ("A is better than B") compare the two arms with the paired difference. Run both arms with the same seeds and let $d_i = a_i - b_i$ be the per-seed differences. The interval for the mean difference is $\bar d \pm t_{n-1,0.975}\, s_d/\sqrt n$. If it excludes 0 the direction is supported in your setting; its width tells you how big the effect is. This is the paired statistic of [[sys.05]]; here we use it as the *verdict* of the reproduction.

## Why reproductions fail

When your number disagrees with the paper, work through this list in order, cheapest check first.

1. **Unreported details.** Initialisation, warmup, weight-decay on bias terms, data filtering, evaluation tricks (test-time augmentation, checkpoint selection). Search the code release; try the obvious defaults; log each as a deviation.
2. **Different evaluation.** The metric, the split, the averaging or the threshold differs. Recompute the *baseline* with your evaluation code; if you cannot match the baseline's published number, the evaluation is suspect, and no comparison against the proposed method can be trusted.
3. **Data leakage in the original or in yours.** If the paper selected hyper-parameters or checkpoints on the test set its number is optimistic ([[cml.03]]: the test set is touched once). Your own pipeline can leak too: normalising with statistics computed on the full dataset, or augmenting before splitting. A reproduction that comes out *better* than the paper is as suspicious as one that comes out worse.
4. **Seeds and variance.** The paper may report the best of several runs, or one lucky run. If its number sits in the tail of your distribution, "not reproduced" may really mean "reported from a favourable draw". Seeds can control initial weights, batch order, dropout masks and the data split; say which ones you vary. (The code in the reasoning-from-scratch notebooks calls `torch.manual_seed(123)` before each sampling demo, and its self-consistency routine seeds the $i$-th sample with `seed + i + 1`: a stochastic procedure is repeatable only because someone seeded it deliberately.)
5. **Non-determinism beyond seeds.** The same code on different hardware or library versions can give different numbers. The results table of the reasoning-from-scratch chapter on evaluation shows a base model scored on the same 10 problems: 30 percent on CPU, CUDA and XPU, but 20 percent on Apple's MPS backend. With only 10 problems one problem flips the number by ten points, so the difference is a mix of numerical non-determinism and tiny sample size; the lesson is that "the same code" does not guarantee "the same number". Typical sources: floating-point summation order on GPU, non-deterministic kernels, different default precision, different library versions.
6. **Scale.** The effect exists at the paper's scale and vanishes at yours (or the other way round). This is the one failure you cannot fix by debugging; you report it as a limit of the reproduction.
7. **The claim was weaker than its summary.** Read the table again. Often the headline says "improves by 2 points" and the table shows 2 points on one dataset with overlapping error bars.

Decide on a stopping rule in advance: for instance, "after exhausting items 1-5 and a day of effort I stop and report 'not reproduced, here is what I tried'".

## Step 5: document deviations and write the report

Every difference between your setup and the paper's is a **deviation**, and a reproduction report is mainly a table of them. A complete report has these sections (this is exactly what the checker in the second exercise looks for):

```text
claim            one sentence: setting, A vs B, metric, direction and size
source           paper, table or figure, version, code release used or not
dataset          source, sizes, splits, preprocessing
model            architecture and size
hyperparameters  every value, and which were tuned for which arm
metric           exact definition and the split it is measured on
seeds            list of seeds, and what each seed controls
environment      hardware, library versions, precision, run time
deviations       every difference from the paper, or the word "none"
results          per-arm mean, sample sd, 95 percent interval, n, paired difference
verdict          reproduced / within tolerance / not reproduced / partly, in one sentence
limits           what the result does NOT show (scale, data, untested settings)
artifacts        commit hash, command to rerun, log files
```

Rules for the verdict sentence: say what you observed in numbers, name the interval, separate observation from interpretation, and never use a stronger word than the evidence supports. "Partly reproduced: the gap narrows with dropout in setting 1 (10 of 10 seeds), but we could not show it in setting 2" is a legitimate and useful conclusion.

## Worked example: does dropout narrow the generalisation gap?

**The claim (a textbook-level one).** "On a small, noisy dataset, a network with dropout 0.5 on its hidden layer has a smaller gap between training accuracy and test accuracy than the same network without dropout." (Dropout was covered in [[dl.02]]; here we treat it as somebody else's claim to be checked.)

**Setup, extracted before running.**

- Data: noisy two-moons ([[f2n.11]]), noise standard deviation 0.3, **60** training points (small on purpose, so the network can overfit) and 2,000 test points. Data seed 1234 fixes one dataset in setting 1.
- Model: $2 \to 64 \to 1$, tanh hidden units, logistic output, weights $W_1\sim\mathcal N(0,1)$, $W_2\sim\mathcal N(0,1/64)$, zero biases.
- Training: full-batch gradient descent, learning rate 0.1, 1,500 steps, cross-entropy loss. Dropout (inverted, rate 0.5) is applied to the hidden activations during training only, a fresh mask per step.
- Metric: gap $=$ training accuracy minus test accuracy, both measured without dropout. The seed controls initial weights and dropout masks.
- Plan: 10 seeds (0 to 9), both arms with the same seeds (common random numbers), 20 runs, each takes about 0.4 s, so the budget is trivial. Tolerance for a hypothetical reported number: 0.02.

```python
import numpy as np

def make_moons(n, noise, rng):
    t = rng.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.r_[a, b] + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y

def train_mlp(seed, p_drop=0.0, hidden=64, steps=1500, lr=0.1,
              n_train=60, noise=0.3, data_seed=1234):
    drng = np.random.default_rng(data_seed)
    Xtr, ytr = make_moons(n_train, noise, drng)
    Xte, yte = make_moons(2000, noise, drng)
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, (2, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1)); b2 = np.zeros(1)
    n = len(Xtr)
    for _ in range(steps):
        H = np.tanh(Xtr @ W1 + b1)                      # (n, hidden)
        if p_drop > 0:
            mask = (rng.random(H.shape) >= p_drop) / (1 - p_drop)
        else:
            mask = 1.0
        Hd = H * mask
        p = 1 / (1 + np.exp(-(Hd @ W2 + b2)[:, 0]))
        dz = ((p - ytr) / n)[:, None]                   # (n, 1)
        gW2 = Hd.T @ dz; gb2 = dz.sum(0)
        dH = (dz @ W2.T) * mask * (1 - H ** 2)          # (n, hidden)
        gW1 = Xtr.T @ dH; gb1 = dH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    def acc(X, y):
        H = np.tanh(X @ W1 + b1)
        return float((((H @ W2 + b2)[:, 0] > 0) == (y == 1)).mean())
    return acc(Xtr, ytr), acc(Xte, yte)             # (train acc, test acc)
```

Before any statistics, run the **determinism check**: `train_mlp(3, 0.5) == train_mlp(3, 0.5)` is `True` here (identical tuple on the same machine). Without that, differences between arms could be noise from the machine rather than from the method.

**Setting 1: one fixed dataset, ten seeds.** The seeds vary initial weights and dropout masks. Results (mean over seeds, 95 percent interval in brackets):

| Quantity | No dropout | Dropout 0.5 |
|---|---|---|
| Train accuracy | 0.962 | 0.888 |
| Test accuracy | 0.891 | 0.873 |
| Gap (train $-$ test) | 0.0708 [0.0614, 0.0801] | 0.0157 [0.0088, 0.0226] |

Paired by seed, the gap reduction is $0.0551$ with 95 percent interval $[0.0447,\ 0.0655]$ (paired $t = 11.95$ on 9 degrees of freedom) and dropout has the smaller gap in **10 of 10** seeds. The test accuracy changes by $-0.0183$, interval $[-0.0210,\ -0.0155]$, and dropout is better in **0 of 10** seeds.

Suppose a paper had reported a gap of 0.020 with dropout. That value lies inside $[0.0088, 0.0226]$, so by the rule above the verdict on that number is "reproduced". (This reported value is hypothetical; the point is how to compare.)

**Setting 2: change what the seed controls.** Setting 1 held the dataset fixed, so its intervals contain only initialisation and dropout noise. A skeptical reader will ask what happens when the *training set itself* is redrawn. Repeat with `data_seed = 100 + seed`, so each seed also gets a new 60-point training set and a new test set:

| Quantity | Value |
|---|---|
| Gap, no dropout | 0.0589 (sd 0.041) |
| Gap, dropout 0.5 | 0.0508 (sd 0.052) |
| Gap reduction, paired | 0.0082, 95 percent interval $[-0.0104,\ 0.0267]$ |
| Test accuracy change | $-0.0168$, interval $[-0.0238,\ -0.0099]$, dropout better in 0 of 10 |

Now the interval for the gap reduction contains zero. The paired differences are scattered (from $-0.025$ to $+0.055$) because a single 60-point dataset can overfit more or less whatever you do.

**Written verdict.** *Partly reproduced.* On one fixed 60-point dataset, dropout 0.5 reduced the train-test accuracy gap from 0.071 to 0.016 (paired difference 0.055, 95 percent interval 0.045 to 0.066, 10 of 10 seeds). When the dataset is also resampled, the reduction shrinks to 0.008 and its interval $[-0.010, 0.027]$ includes zero with 10 seeds; to see an effect of that size with 80 percent power one would need about 80 paired runs (the power calculation is derived in [[res.02]]). In both settings dropout lowered test accuracy (by about 0.017 to 0.018) in every seed, so the narrower gap came from lower training accuracy, not from better generalisation. Deviations: custom NumPy implementation, tanh network, full-batch updates, no tuning of the dropout rate or of the learning rate for either arm. Limits: one architecture, one dataset family, and 1,500 steps; whether dropout would help with a longer schedule or a tuned baseline is untested.

Notice what the report does *not* say: it does not say dropout "works" or "does not work". A narrower gap is the claim, and in setting 1 that claim holds; whether a narrower gap is useful is a separate question the same numbers answer negatively.

## Implementation: the harness

The same structure is the first exercise. The code below is the entire harness.

```python
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]  # df 1..10 (extend)

def run_seeds(fn, seeds):
    return np.array([fn(s) for s in seeds], dtype=float)

def summarize(x):
    x = np.asarray(x, dtype=float); n = len(x)
    half = T975[n - 2] * x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": x.mean(), "sd": x.std(ddof=1), "lo": x.mean() - half, "hi": x.mean() + half}

gaps_a = run_seeds(lambda s: np.subtract(*train_mlp(s, 0.0)), range(10))   # train - test
gaps_b = run_seeds(lambda s: np.subtract(*train_mlp(s, 0.5)), range(10))
print(summarize(gaps_a - gaps_b))      # paired difference: same seeds in both arms
```

In production research code you would use the experiment trackers mentioned by Huyen (for example MLflow or Weights and Biases) to store these arrays, configs and commit hashes automatically; the minimal version is a JSON file per run containing the config, seed, environment and metrics.

## Shapes

| Object | Shape |
|---|---|
| training inputs / labels | $(60, 2)$ / $(60,)$ |
| $W_1$, $b_1$, $W_2$, $b_2$ | $(2, 64)$, $(64,)$, $(64, 1)$, $(1,)$ |
| dropout mask, hidden activations | $(60, 64)$ each |
| results of one arm (train, test) | $(n_\text{seeds}, 2)$ |
| paired differences | $(n_\text{seeds},)$ |

## Common misconceptions

1. "If my number matches the paper to two decimals, I reproduced it." It may be luck; a match inside a wide interval carries little information. Report the interval and the number of runs.
2. "If my number is below the paper's, the paper is wrong." Check the evaluation, the baseline, the seed distribution and the unreported details first. Authors can also report favourable runs without bad intent.
3. "A single seed reproduces a result if the code is the same." The code fixes the procedure, not the random draw. A single seed shows one sample from the distribution.
4. "Seeding makes the run deterministic." Seeding fixes the pseudo-random streams you control; floating-point order on parallel hardware, library versions and nondeterministic kernels can still change results.
5. "A failed reproduction is a wasted week." A well-documented failure (what you tried, what the intervals were) is a result; the waste is an undocumented one.
6. "Qualitative agreement means the paper's effect size is right." In the worked example the sign of the effect was reproduced while the practical conclusion (is the model better?) was the opposite of what the gap alone suggests.

## Key takeaways

- Choose one specific claim; write it as a sentence with setting, arms, metric and direction.
- Extract the setup into a table; each empty row is an unreported detail you must fill and log as a deviation.
- Plan runs as configurations times seeds and check the budget before starting.
- Compare to the paper with an interval over your own runs and a pre-declared tolerance; use paired differences for "A beats B".
- Check determinism and reproduce the baseline first; investigate failures in a fixed order and with a stopping rule.
- State which sources of variance your seeds cover; widening them (new data, not only new initialisation) can change the verdict.
- A report is a table of deviations plus a verdict sentence that claims no more than the evidence.

## Explain it yourself

1. In setting 1 the interval for the gap reduction was $[0.045, 0.066]$; in setting 2 it was $[-0.010, 0.027]$. Both use the same method and the same ten seeds. Explain what changed and which experiment better supports a claim about "dropout in general".
2. A colleague reproduces a paper with one run and writes "reproduced, 71.2 vs 71.5". Write the three things you would ask for before accepting that sentence.

## Further reading

- Henderson et al., "Deep Reinforcement Learning that Matters" (2018): reproducibility failures from seeds and implementation details.
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021): which sources of variance to randomise.
- Pineau et al., "Improving Reproducibility in Machine Learning Research" (2020): checklists for papers.
- Huyen, *Designing Machine Learning Systems*, Ch. 6: experiment tracking and versioning.

## Mastery criteria

- [ ] I can turn a paper sentence into a specific, falsifiable claim and fill a setup table with the unreported details marked.
- [ ] I can budget a reproduction as configurations times seeds times minutes per run.
- [ ] I can compare my runs to a reported number with a Student interval and a tolerance fixed in advance.
- [ ] I can say which sources of variance a seed controls and widen them deliberately.
- [ ] I can run a determinism check and list the failure causes in the order I would investigate them.
- [ ] I can write a reproduction report whose verdict does not claim more than the evidence.
