---
id: cml.10
title: Evaluation metrics for classification and regression
summary: Turn "is this model any good?" into numbers that match the real cost of mistakes - confusion matrix, precision/recall, ROC and AUC (derived as a probability), calibration, and regression errors.
minutes: 70
teaches: [ml.eval-metrics, ml.roc-auc, ml.calibration]
requires: [prob.basics, ml.logreg, ml.generalization]
sources:
  - "Designing Machine Learning Systems.pdf, Ch. 6 'Model Offline Evaluation' (PDF p.201-214): baselines (Table 6-2), perturbation/invariance tests, calibration, slice-based evaluation and Simpson's paradox. READ."
  - "Designing Machine Learning Systems.pdf, Ch. 4 'Using the right evaluation metrics' (PDF p.134-145): the cancer models A and B (Tables 4-4 to 4-7), asymmetry of F1, ROC and precision-recall curves. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 1.5 Decision Theory (PDF p.58-66): minimum misclassification rate, loss matrix and expected loss (1.81), reject option, compensating for class priors. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/09-model-evaluation/docs/en.md: confusion matrix, precision/recall/F1/AUC, MSE/RMSE/MAE/R^2 definitions, common evaluation mistakes. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/17-imbalanced-data/docs/en.md (metrics section): F-beta, AUPRC baseline equals the positive rate. READ."
---

## Where this fits

A hospital deploys a model that reads chest X-rays. In the validation set 1 patient in 100 has cancer. The model has 99% accuracy. Is it good? It might be a model that says "healthy" for everybody. A fraud team compares two models: one is slightly better on average error, the other ranks fraudulent transactions higher. Which should go live? An ads team wants to forecast how many clicks next week will bring, not only which ad ranks first.

Each question needs a different yardstick. Training a model (lessons 1 to 5) is only half the job; the other half is **measuring** it in a way that reflects what the mistakes cost. A wrong metric does not just give a wrong number: it makes you pick the wrong model, tune the wrong thing, and report success on a system that does not work. This lesson builds the standard toolbox and, more importantly, the habit of choosing from it by asking "what does each kind of error cost?".

It connects back to logistic regression (lesson 2), whose output is a probability, and to generalisation (lesson 3), because every number here must be computed on data the model has not trained on. It connects forward to class imbalance and leakage (lesson 11).

## The idea in plain words

**Counting mistakes properly.** A classifier can be wrong in two different ways. It can raise an alarm that should not have been raised (a false alarm), or it can stay silent when it should have spoken (a miss). Smoke detectors, spam filters, cancer screening: in each of them the two errors hurt differently. A table that counts the four outcomes (alarm/no alarm times truly fire/no fire) is the **confusion matrix**, and every classification metric is a different way to summarise it.

**Accuracy is an average over everybody.** It is dominated by whichever group is biggest. If 99% of people are healthy, "everyone is healthy" scores 99%. The question we really care about is usually about the rare group: of the people who are sick, how many did we catch (**recall**)? Of the people we flagged, how many were really sick (**precision**)?

**A threshold is a dial, not a fact.** Most models output a score or probability, and we choose a cut-off to turn it into a yes/no. Slide the cut-off down and you catch more real cases but raise more false alarms. The **ROC curve** draws that whole trade-off in one picture, and the **AUC** summarises it with a single number with a lovely meaning: *pick one random positive and one random negative; AUC is the chance the model gives the positive the higher score.* It only measures ranking.

**Ranking is not the same as being honest.** A weather forecaster who says "70% rain" should see rain on about 70% of those days. That property is **calibration**. A model can rank perfectly (AUC = 1) and still say "99%" when the truth is 60%. If you only sort items, calibration does not matter; if you multiply probabilities by money or count expected events, it matters a great deal.

**Regression has the same story.** Errors can be measured in units of the target (MAE, RMSE) or relative to a trivial guess (R squared). Squaring errors punishes big misses heavily; absolute values treat them linearly.

**A number alone is meaningless.** "F1 = 0.90" sounds great until you learn that predicting the majority class at random also gives about 0.90 on that data. Always compare to a baseline.

## The confusion matrix and its metrics

For a binary problem with labels $y\in\{0,1\}$ (1 = positive) and predictions $\hat y$:

| | predicted 1 | predicted 0 |
|---|---|---|
| actual 1 | TP (hit) | FN (miss) |
| actual 0 | FP (false alarm) | TN |

$$
\text{accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad \text{precision}=\frac{TP}{TP+FP},\qquad \text{recall}=\frac{TP}{TP+FN}.
$$

The **false positive rate** is $FPR = FP/(FP+TN)$, and recall is also called the true positive rate, $TPR$. The **F1** score is the harmonic mean of precision $P$ and recall $R$:

$$
F_1=\frac{2PR}{P+R}.
$$

The harmonic mean is pulled towards the smaller of the two, so you cannot get a high $F_1$ by maximising one and ignoring the other. The weighted version $F_\beta=(1+\beta^2)PR/(\beta^2P+R)$ treats recall as $\beta$ times as important as precision.

Precision, recall and F1 are **asymmetric**: they change when you swap which class is called positive. Accuracy does not.

## Worked example 1: two models, same accuracy

Take the book's example of 1000 patients, 100 with cancer (positive) and 900 normal.

| | TP | FP | FN | TN |
|---|---|---|---|---|
| Model A | 10 | 10 | 90 | 890 |
| Model B | 90 | 90 | 10 | 810 |

Both have accuracy $(10+890)/1000=(90+810)/1000=0.9$. But:

- Model A: precision $10/20=0.5$, recall $10/100=0.1$, $F_1=2(0.5)(0.1)/0.6=0.167$.
- Model B: precision $90/180=0.5$, recall $90/100=0.9$, $F_1=2(0.5)(0.9)/1.4=0.643$.

Model A misses 90 of the 100 sick people. Same accuracy, completely different usefulness. (The exercise reproduces these numbers.)

**Baselines.** With 10% positives, a model that predicts positive with probability 0.5 at random has accuracy $0.5$ and $F_1=0.167$ (precision 0.1, recall 0.5). A model that predicts positive 10% of the time at random (matching the label distribution) has accuracy $0.9\cdot0.9+0.1\cdot0.1=0.82$ and $F_1=0.1$. A "zero rule" model that always predicts the most frequent class is the simplest reference. Any model you build has to beat these by a margin that justifies its cost. Other useful baselines: a simple heuristic, a human expert, the existing system.

## Decision theory: where the threshold really comes from

A classifier that outputs $p(C_k\mid x)$ has done the *inference*. Choosing the action is a separate *decision* step. Bishop's Sec. 1.5 shows that the rule that makes the fewest mistakes assigns $x$ to the class with the largest posterior. But mistakes rarely cost the same, so we introduce a **loss matrix** $L_{kj}$ = loss when the true class is $k$ and we decide $j$. For each $x$ the expected loss of deciding $j$ is $\sum_k L_{kj}\,p(C_k\mid x)$, and we choose the $j$ that minimises it.

### Derivation: the cost threshold

Two classes: positive (1) and negative (0). Let a correct decision cost 0, a false positive cost $c_{FP}$ and a false negative cost $c_{FN}$. With $p=p(y=1\mid x)$:

$$
\mathbb E[\text{loss of predicting }1]=(1-p)\,c_{FP},\qquad
\mathbb E[\text{loss of predicting }0]=p\,c_{FN}.
$$

Predict 1 exactly when the first is smaller:

$$
(1-p)\,c_{FP}<p\,c_{FN}\iff p>\frac{c_{FP}}{c_{FP}+c_{FN}}.
$$

With equal costs the threshold is $0.5$. In Bishop's cancer matrix a missed cancer costs 1000 and a false alarm costs 1, so the threshold is $1/1001\approx0.001$: flag anyone with even a 0.1% chance. Notice what this says: **the 0.5 default is a special case, not a law.** It also says why probabilities (not just labels) are valuable: if the costs change next quarter you move the threshold and need no retraining. Bishop also describes a **reject option**: if the largest posterior is below some $\theta$, hand the case to a human.

## ROC curve and AUC

Given scores $s_i$ (any real numbers, higher means "more positive"), each threshold $t$ gives a point $(\mathrm{FPR}(t),\mathrm{TPR}(t))$ with predictions $\hat y=\mathbb 1[s\ge t]$. Sweeping $t$ from $+\infty$ down to $-\infty$ traces the **ROC curve** from $(0,0)$ (nothing flagged) to $(1,1)$ (everything flagged). A perfect model hugs the top-left corner; random scores follow the diagonal.

### Derivation: AUC is a probability

Let $S^+$ be the score of a randomly drawn positive, with density $f_+$, and $S^-$ the score of a random negative, with density $f_-$. Then

$$
\mathrm{TPR}(t)=P(S^+>t),\qquad \mathrm{FPR}(t)=P(S^->t).
$$

As $t$ decreases the curve moves right; the infinitesimal step in the horizontal direction is $d\,\mathrm{FPR}=-f_-(t)\,dt$. The area under the curve is therefore

$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}\;d\,\mathrm{FPR}
=\int_{-\infty}^{\infty} P(S^+>t)\,f_-(t)\,dt .
$$

The right-hand side reads: "draw a negative score $t$ from $f_-$, and ask how likely a positive beats it." Averaging over $t$ gives

$$
\mathrm{AUC}=P(S^+>S^-)\quad(\text{plus }\tfrac12P(S^+=S^-)\text{ for ties}).
$$

For a finite sample the same statement is a count: among all $n_+n_-$ (positive, negative) pairs, count the pairs where the positive scores higher (ties count half) and divide. This also equals the trapezoid area under the empirical ROC staircase. Consequences:

- $\mathrm{AUC}=0.5$ is random ranking; $\mathrm{AUC}<0.5$ means the ranking is backwards (flip the sign).
- It depends only on the *order* of scores, so any increasing transformation of them leaves it unchanged. This is exactly why AUC says nothing about calibration.
- It does not depend on the class ratio, which is both a strength and a weakness (next section).

### Worked example 2: a six-point ROC

Labels $y=(1,1,0,1,0,0)$ with scores $s=(0.9,0.8,0.7,0.6,0.4,0.2)$. There are 3 positives and 3 negatives, so TPR moves in steps of $1/3$ and FPR in steps of $1/3$. Walking down the sorted scores gives the ROC points

$$
(\mathrm{FPR},\mathrm{TPR}):\ (0,0),(0,\tfrac13),(0,\tfrac23),(\tfrac13,\tfrac23),(\tfrac13,1),(\tfrac23,1),(1,1).
$$

The trapezoid area is $\tfrac13\cdot\tfrac23+\tfrac13\cdot1+\tfrac13\cdot 1=\tfrac89$. Pair counting: positives $\{0.9,0.8,0.6\}$ against negatives $\{0.7,0.4,0.2\}$; the positive wins $3+3+2=8$ of $9$ pairs, so $8/9\approx0.889$. Same answer, as the derivation promised.

## Precision-recall curves and imbalance

The ROC curve uses $\mathrm{FPR}=FP/(FP+TN)$. When negatives vastly outnumber positives, a large number of false alarms barely moves FPR, so ROC/AUC can look excellent while precision is poor. Davis and Goadrich (as reported in the DMLS chapter on imbalance) suggested the **precision-recall curve**, which plots precision against recall as the threshold moves and ignores true negatives altogether. A random classifier has a PR curve at the level of the positive rate (for example 0.01) rather than 0.5, so improvements are easier to see. For the six-point example the thresholds $0.85, 0.65, 0.5, 0.3$ give (precision, recall) of $(1,\tfrac13)$, $(\tfrac23,\tfrac23)$, $(0.75,1)$, $(0.6,1)$.

## Calibration

A probabilistic model is **calibrated** if among all cases where it says $p$, the event happens a fraction $p$ of the time. The DMLS chapter uses the example of a team predicted to win with probability 70% but winning only 60% of the games, and an ad whose predicted click chance of 10% is really 5%: ranking is fine but the click forecast is off by a factor of two.

**Reliability diagram.** Split $[0,1]$ into bins; in each bin plot the mean predicted probability (x) against the observed fraction of positives (y). A calibrated model lies on the diagonal. The **expected calibration error** summarises the gaps:

$$
\mathrm{ECE}=\sum_{b}\frac{n_b}{N}\,\bigl|\mathrm{freq}_b-\mathrm{conf}_b\bigr| .
$$

**Brier score.** The mean squared error between predicted probability and the 0/1 outcome, $\frac1N\sum_i(p_i-y_i)^2$. It rewards both sharpness and honesty; lower is better; always predicting the base rate is the reference.

### Worked example 3: calibration bins

Fifteen predictions: five at $0.1$ (one of them positive), five at $0.9$ (four positive), five at $0.6$ (two positive). Bin frequencies are $0.2, 0.4, 0.8$ against confidences $0.1, 0.6, 0.9$. Each bin has weight $1/3$:

$$
\mathrm{ECE}=\tfrac13(0.1+0.2+0.1)=0.1333.
$$

The Brier score is $\frac1{15}(0.85+1.40+0.85)=0.2067$ (the 0.1 group contributes $4(0.01)+0.81=0.85$, the 0.6 group $2(0.16)+3(0.36)=1.40$, the 0.9 group $4(0.01)+0.81=0.85$).

If a model ranks well but is miscalibrated you can fix the probabilities afterwards by fitting a monotone map (for example logistic regression on the score) on a held-out set; since the map is increasing, AUC is unchanged. Also note a subtle consequence from Bishop: if you rebalance the training set (say 50/50 instead of 1/1000), the posteriors are shifted by the class priors, and you must divide by the training class fractions and multiply by the true ones, then renormalise, to get calibrated probabilities for the real population.

## Regression metrics

For targets $y_i$ and predictions $\hat y_i$, with $e_i=\hat y_i-y_i$:

$$
\mathrm{MAE}=\frac1N\sum|e_i|,\qquad \mathrm{RMSE}=\sqrt{\frac1N\sum e_i^2},\qquad R^2=1-\frac{\sum e_i^2}{\sum(y_i-\bar y)^2}.
$$

MAE is in the units of the target and treats every unit of error equally (it is the loss whose optimum is the median). RMSE is also in target units but squares errors first, so a few large misses dominate (its optimum is the mean). $R^2$ compares your squared error with that of the trivial predictor "always $\bar y$": $R^2=1$ is perfect, $0$ is no better than the mean, negative is worse.

### Worked example 4

$y=(3,5,2,7)$, $\hat y=(2.5,5,4,8)$. Errors are $(-0.5,0,2,1)$. MAE $=3.5/4=0.875$. MSE $=(0.25+0+4+1)/4=1.3125$, RMSE $=1.1456$. $\bar y=4.25$, $\sum(y-\bar y)^2=1.5625+0.5625+5.0625+7.5625=14.75$, $\sum e^2=5.25$, so $R^2=1-5.25/14.75=0.644$. A single bad miss (the 2 on the third point) contributes most of the squared error, which is why RMSE (1.15) is noticeably larger than MAE (0.875).

## Beyond one number: what DMLS adds

- **Slice-based evaluation.** An overall number hides subgroups. In the book's example, Model A has 98% accuracy on the majority (90% of data) and 80% on the minority, so 96.2% overall; Model B has 95% on both, so 95% overall. The overall metric prefers A, but A is far worse for the minority. Simpson's paradox shows aggregation can even reverse a comparison. Always look at the slices that matter (paid users, mobile traffic, protected groups).
- **Perturbation tests**: add realistic noise to the test inputs and see how the metric degrades. **Invariance tests**: changing a sensitive attribute alone should not change the output. **Directional tests**: increasing lot size should not lower a predicted house price.
- Ideally evaluation in development and production should match; where labels arrive late you must monitor instead.

## Choosing the metric from the business cost

1. Write down the cost (or benefit) of each cell of the confusion matrix in money, harm or time.
2. If the decision is a yes/no action with known costs, use the expected-cost threshold above and report **total expected cost** at that threshold.
3. If the downstream use is a ranking (top-k list, review queue), use AUC or precision at k.
4. If the output is a probability that will be multiplied or summed (revenue, risk, number of events), require **calibration** (ECE, Brier).
5. If positives are rare, report precision-recall, not accuracy and not ROC alone.
6. In regression, choose MAE when all errors matter proportionally and outliers are noise, RMSE when large errors are disproportionately bad, a relative error when a 100% miss on a big value matters more.
7. Compare with a baseline and report uncertainty; do not tune on the test set.

## Implementation

```python
import numpy as np

def roc_curve(y, s):
    order = np.argsort(-s, kind="stable")
    y_s, s_s = y[order], s[order]
    last = np.r_[np.where(np.diff(s_s) != 0)[0], len(s_s) - 1]   # one point per distinct score
    tpr = np.r_[0, np.cumsum(y_s)[last] / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y_s)[last] / (1 - y).sum()]
    return fpr, tpr

def auc_pairs(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
```

## Shapes

`y: (N,)` of 0/1, `scores: (N,)`, `fpr, tpr: (T+1,)` where $T$ is the number of distinct scores, pairwise comparison matrix `(n_pos, n_neg)`. Calibration bins return three arrays of length (number of non-empty bins).

## Common misconceptions

- **"High accuracy means a good model."** Not when classes are imbalanced; compare with the majority-class baseline.
- **"AUC = 0.9 means 90% accuracy."** It is a probability about *pairs*, independent of any threshold and of calibration.
- **"A well-calibrated model is accurate."** A model that always outputs the base rate is perfectly calibrated and useless for ranking.
- **"The threshold should be 0.5."** The optimal threshold depends on the costs (and the base rate); it is a decision, not a property of the model.
- **"F1 is symmetric."** It changes if you swap the positive class; also it ignores true negatives.
- **"RMSE, MAE and R squared rank models identically."** They can disagree, especially in the presence of outliers.

## Key takeaways

- Start from the confusion matrix and from the cost of each kind of error; accuracy is rarely the right yardstick.
- AUC is the probability that a random positive outranks a random negative; it measures ranking only. Use precision-recall when positives are rare.
- Calibration (reliability diagram, ECE, Brier) is separate from ranking and essential when probabilities are used as numbers.
- The cost-optimal threshold is $c_{FP}/(c_{FP}+c_{FN})$ on a calibrated probability.
- Report slices and baselines, not only a global number.

## Explain it yourself

1. Why can two models with identical accuracy be very different, and which numbers would you show a doctor?
2. Explain in your own words why the area under the ROC curve equals the probability that a positive outscores a negative.

## Further reading

- *Designing Machine Learning Systems*, Ch. 6 (offline evaluation) and Ch. 4 (metrics under class imbalance).
- Bishop, *PRML*, Sec. 1.5 (decision theory, loss matrix, reject option).
- Source library: `ai-engineering-from-scratch` phase 02, lesson 09.

## Mastery criteria

You can compute any confusion-matrix metric by hand, draw an ROC curve from scores, prove and use the pairwise reading of AUC, build a reliability table, derive the cost threshold, and justify the metric for a given business problem.
