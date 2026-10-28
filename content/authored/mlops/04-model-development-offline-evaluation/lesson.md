---
id: ops.04
title: "Model development and offline evaluation"
summary: Choose models without fooling yourself, combine them with ensembles, track and version experiments, understand distributed training at a glance, and evaluate offline with baselines, behavioural tests and slice metrics.
minutes: 70
teaches: [ops.offline-eval, ops.slice-eval]
requires: [ml.eval-metrics, ml.generalization, ops.skew]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.6 Model Development and Training: six tips for model selection, ensembles (bagging, boosting, stacking), experiment tracking and versioning, distributed training (data and model parallelism) (PDF p.180-200, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.6 Model Offline Evaluation: baselines, perturbation tests, invariance and directional expectation tests, slice-based evaluation, Simpson's paradox (PDF p.201-214). READ."
---

## Where this fits

By now you have framed the problem ([[ops.01]]), built a training set ([[ops.02]]) and consistent features ([[ops.03]]). This lesson is the part most people think of as "machine learning": picking and training models and judging them. The production twist is that judging is harder than training. One accuracy number computed on a held-out split says little about whether the model will survive contact with real users. You will learn to compare against baselines, to probe behaviour with perturbation and invariance tests, and to slice the metric so that a model that is good on average but bad for an important group cannot hide. Distribution shift after deployment is the subject of [[ops.06]]; the evaluation metrics themselves are covered in [[cml.10]].

## The idea in plain words

Choosing a model is like hiring. The résumé (a leaderboard score) was earned on someone else's tasks. A fair hiring process gives every candidate a comparable trial period, starts from a simple reference (what would a sensible beginner achieve?), tests how the candidate reacts to messy real-world situations, and checks performance for each type of customer rather than only the grand average. An ensemble is a committee: independent opinions that average out each other's mistakes, at the price of a bigger payroll. Experiment tracking is the lab notebook that lets you repeat a good result or work out why a bad one happened.

## Model selection: six habits

1. **Avoid the state-of-the-art trap.** A model that tops a benchmark on a static dataset is not necessarily fast enough, cheap enough or better on your data. Use the simpler solution if it solves your problem.
2. **Start with the simplest model.** Simple models deploy earlier (which validates that your serving pipeline matches training, see [[ops.03]]), are easier to debug, and give you a baseline. Simplest does not mean least effort: a pretrained model can be easy to start with, though hard to improve.
3. **Avoid human biases in comparisons.** If you try 100 configurations of the architecture you like and 3 of the one you do not, the first wins by effort. Give architectures comparable tuning budgets, and remember a claim about an architecture is only true in the context where it was tested.
4. **Compare good performance now versus later.** A tree model may win with little data while a neural network wins after you double it; a learning curve (performance versus training-set size) hints which.
5. **Evaluate trade-offs**, such as false positives versus false negatives, and accuracy versus latency or interpretability.
6. **Understand the model's assumptions** (for example that features are independent, or that the data is smooth), because they tell you when it will fail.

## Ensembles

An **ensemble** combines several **base learners**. Why does it help? Suppose three classifiers are each correct with probability $p=0.7$ on every example, and their errors are independent. The majority vote is correct when at least two are correct:

$$
P(\text{correct})=\sum_{k=2}^{3}\binom{3}{k}p^k(1-p)^{3-k}=3(0.7)^2(0.3)+(0.7)^3=0.441+0.343=0.784 .
$$

So three 70% models give 78.4%. For $n$ independent voters it is $\sum_{k>n/2}\binom{n}{k}p^k(1-p)^{n-k}$, which grows toward 1 as $n$ grows when $p>0.5$. The assumption that matters is **independence**: if the models always agree, the ensemble is exactly as good as one of them. This is why practitioners combine different kinds of models. Three ways to build ensembles:

- **Bagging** (bootstrap aggregating): train each model on a bootstrap sample (drawn with replacement), then vote or average. It reduces variance and helps unstable learners such as deep trees. A random forest is bagging plus a random subset of features at each split.
- **Boosting**: train weak learners one after another on the same data, re-weighting examples so later learners focus on previous mistakes; combine them with weights favouring accurate learners. Gradient boosting (XGBoost, LightGBM) is the best-known family.
- **Stacking**: a meta-learner (even a plain average, or a logistic regression) combines the base learners' outputs.

Ensembles dominate competitions but are less common in production, because they cost more to serve and maintain and are harder to interpret. They survive where a small gain is worth a lot of money, such as ad click-through prediction.

## Experiment tracking and versioning

Models differ by learning rates of 0.003 versus 0.002 and can behave very differently, so you must be able to recreate any experiment. **Experiment tracking** records, for every run: the loss curves for training and each evaluation split, the evaluation metrics, the predictions or samples, the speed (iterations per second) and system usage (memory, CPU/GPU), and the hyper-parameters. **Versioning** records everything needed to reproduce the run: code, data (or a data snapshot), configuration and environment. Tools like MLflow and Weights & Biases grew from tracking, DVC from versioning, and they now overlap. Data is the hardest to version because it is large; at minimum keep immutable snapshots and hashes. A practical test: can a colleague, six months from now, rebuild the model you shipped and obtain the same metrics?

## Distributed training in one paragraph

**Data parallelism** gives each worker a copy of the model and a share of each batch, then accumulates their gradients. Synchronous SGD waits for all workers, so a single slow worker (a *straggler*) delays everyone, and the chance that some worker is slow grows with the number of workers. Asynchronous SGD updates as gradients arrive and suffers from *stale* gradients computed on older weights. **Model parallelism** splits the model itself across machines when it does not fit on one; naively, machine 2 waits for machine 1, which **pipeline parallelism** reduces by splitting each batch into smaller parts that flow through the stages like an assembly line. Frameworks (PyTorch DistributedDataParallel and similar) hide most of this.

## Offline evaluation

### Baselines

A metric means nothing alone. A model with $F_1=0.90$ on a task where 90% of labels are positive might be no better than random. Always compare with baselines:

- **Random baseline**: predict at random, following the uniform distribution or the label distribution.
- **Simple heuristic** (forget ML): rank a feed by recency, route by keywords.
- **Zero-rule baseline**: always predict the most common class (or the most-used app).
- **Human baseline**: how well do experts do? This is the relevant bar when automating human work.
- **Existing solution**: the currently deployed system, because the business compares you with that.

Derivation of the random baseline. Let the positive class have prevalence $p$ and let the model predict positive with probability $q$, independently of the input. Then precision $=p$ (a random prediction is positive-class with probability $p$), recall $=q$, so

$$
F_1=\frac{2\,p\,q}{p+q},\qquad \text{accuracy}=p\,q+(1-p)(1-q).
$$

With $p=0.1$ and uniform guessing $q=0.5$: $F_1=0.1/0.6\approx0.167$ and accuracy $=0.5$. Guessing with the label distribution $q=0.1$: $F_1=0.1$ and accuracy $=0.01+0.81=0.82$. A model with 82% accuracy on this task is therefore at chance level.

### Perturbation tests

Production inputs are noisier than your curated test set (coughs recorded on phones in a crowded room versus in a hospital). Perturb the test inputs with small, realistic changes (noise, clipping, typos) and measure how performance falls. Prefer the model that is best on perturbed data, since sensitivity to noise makes a model hard to maintain: when user behaviour shifts slightly, a brittle model degrades.

### Invariance and directional expectation tests

Some input changes **should not** change the output: changing a name or a protected attribute in a résumé or loan application. Keep everything else fixed, change the sensitive field, and count how many predictions flip (better, do not use the field at all). Other changes should move the output in a **known direction**: raising a house's lot size, other things equal, must not lower the predicted price. Counting the rows where the direction is violated exposes a model that learned something strange. These tests need no labels.

### Slice-based evaluation

An aggregate metric is an average, and averages hide structure. **Slices** are subsets of the data (mobile versus desktop, region, rare classes, new users). A model can be excellent on the whole and poor on a slice that matters: one team found great overall performance and terrible results on mobile traffic because a button was half-hidden on small screens, a non-ML bug that slicing revealed.

**Simpson's paradox** shows that comparing aggregates can even reverse the right conclusion. With the counts of the book's Table 6-4 (taken from a kidney-stone treatment study), model A is correct on 81 of 87 in group A and 192 of 263 in group B; model B is correct on 234 of 270 and 55 of 80. Per group, A wins (93.1% vs 86.7%, and 73.0% vs 68.8%). Overall, A scores 273/350 = 78.0% while B scores 289/350 = 82.6%, so B wins. The reversal occurs because the groups are weighted differently in the two totals: most of B's evaluation examples come from the easier group A. Moral: compare models on the same slices with the same data, and always report per-slice numbers.

**Finding slices.** (1) Heuristics from domain knowledge; (2) error analysis, reading misclassified examples and finding patterns; (3) automatic slice finders that generate candidate subsets (beam search, clustering, decision trees), prune and rank them. Each critical slice needs enough correctly labelled examples to give a stable estimate, so the quality of your evaluation is capped by your evaluation data.

## Worked example

Every number below is computed in the exercises.

- Ensemble: `majority_accuracy(0.7, 3)=0.784`; with 11 independent voters it rises further.
- Random baselines on 10% positives: uniform guessing gives $F_1\approx0.167$, accuracy $0.5$; label-distribution guessing gives $F_1=0.1$, accuracy $0.82$.
- Simpson: per-slice accuracies of model A are $81/87\approx0.931$ and $192/263\approx0.730$, those of model B $234/270\approx0.867$ and $55/80\approx0.688$; overall A $=0.780$, B $=0.826$.
- Perturbation: a model that predicts `x0 > 0` on a dataset whose true label is `x0 + 0.1 x1 > 0` scores 96.9% clean. Adding Gaussian noise with standard deviation 0.1, 1 and 2 to every feature lowers accuracy by 0.7, 22.3 and 33.0 points respectively (fixed seed).
- Invariance: a model that adds `0.5 * (sensitive - 0.5)` to its score changes its prediction for 9.55% of rows when the sensitive column is forced to 1; the clean model changes for none.
- Directional: the score $\sin(3x_0)$ goes the wrong way for 50.4% of rows when $x_0$ increases by 0.5, whereas $2x_0+x_1$ never does.

## Implementation

```python
import numpy as np
from math import comb

def slice_metrics(y_true, y_pred, slice_ids):
    out = {}
    for s in np.unique(slice_ids):
        m = slice_ids == s
        out[s.item()] = (int(m.sum()), float(np.mean(y_true[m] == y_pred[m])))
    return out

def majority_accuracy(p, n):
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))

def perturbation_drop(predict, X, y, noise_std, rng):
    clean = np.mean(predict(X) == y)
    noisy = np.mean(predict(X + rng.normal(0, noise_std, X.shape)) == y)
    return float(clean - noisy)
```

Libraries: scikit-learn `RandomForestClassifier`, `BaggingClassifier`, `StackingClassifier`; `xgboost`/`lightgbm`; MLflow or Weights & Biases for tracking; the slice and behavioural testing ideas appear in toolkits such as checklist-style NLP test suites and in evaluation dashboards.

## Shapes

`y_true, y_pred, slice_ids: (n,)`; metrics dictionary: slice id to `(count, accuracy)`; `predict: (n, d) -> (n,)`; noise has the shape of `X`.

## Common misconceptions

- "The best model on the leaderboard is best for us." Different data, latency and cost.
- "Ensembling always helps by the formula." Only if the base learners are not strongly correlated.
- "Overall accuracy up means every group improved." Simpson's paradox and slice regressions say otherwise.
- "A high F1 is good." Compare with the random and zero-rule baselines first.
- "Removing the sensitive feature removes the bias." Other features can act as proxies; invariance tests check the model's behaviour, not just its inputs.

## Key takeaways

- Start simple, compare architectures with comparable effort, and ask which model will be best later as data grows.
- Independent base learners combine into better predictions (three 70% voters give 78.4%); bagging cuts variance, boosting focuses on errors, stacking learns the combination.
- Track and version everything needed to reproduce a run.
- Data parallelism is limited by stragglers (sync) or stale gradients (async); model parallelism needs pipelining.
- Evaluate against baselines, perturb inputs, test invariances and directions, and report metrics by slice.

## Explain it yourself

1. Explain, with numbers, why 3 independent 70% classifiers combine to 78.4% and why correlated ones do not.
2. Construct (or describe) a scenario in which model A is better in every slice yet worse overall.

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch.6 (PDF p.180-216), including the parts not detailed here (further evaluation methods such as calibration).
- [[cml.10]] for the metrics themselves, and [[cml.03]] for generalisation.

## Mastery criteria

You can compute the baseline scores of a random model, derive the accuracy of a majority-vote ensemble, state when ensembling fails, write perturbation, invariance and directional tests as short functions, compute slice metrics, and explain Simpson's paradox with an example.
