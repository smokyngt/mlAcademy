---
id: cml.11
title: Features, class imbalance and data leakage
summary: Prepare inputs the model can use (scaling, encoding, crosses, missing values, selection), cope with rare classes, and avoid the silent bug that makes validation scores lie - leakage.
minutes: 75
teaches: [ml.features, ml.imbalance, ml.leakage]
requires: [ml.generalization, ml.logreg, ml.eval-metrics]
sources:
  - "Designing Machine Learning Systems.pdf, Ch. 5 Feature Engineering (PDF p.157-178): missing values (MNAR/MAR/MCAR), scaling, discretization, encoding categories and the hashing trick, feature crossing, causes and detection of data leakage, feature importance and generalization. READ."
  - "Designing Machine Learning Systems.pdf, Ch. 4 Class Imbalance (PDF p.132-148): why imbalance is hard, metrics, resampling (undersample/oversample, Tomek links, SMOTE), cost-sensitive and class-balanced loss, focal loss. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 1.5.4 'Compensating for class priors' (PDF p.65): correcting posteriors after rebalancing. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/08-feature-engineering/docs/en.md: numerical transforms, one-hot/label/target encoding, missing-value strategies, interactions. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/17-imbalanced-data/docs/en.md: class weights formula, SMOTE, threshold tuning, cost-sensitive learning, MCC. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/docs/en.md: filter, wrapper and embedded selection, variance threshold, mutual information, RFE, L1, permutation importance. READ."
---

## Where this fits

Most real projects spend more time on the inputs than on the model. Raw data has columns on wildly different scales (age 20 to 40 next to income 10,000 to 150,000), text-like categories ("brand = Acme"), holes (income not filled in), and a target that is rarely balanced (0.01% of X-rays show cancer, a fraction of a percent of transactions are fraud). A good model of the wrong inputs loses to a plain model of good ones.

On top of that sits the most expensive mistake in applied machine learning: **data leakage**, where information that will not exist at prediction time sneaks into training or into the choice of features. The symptom is a validation score that looks wonderful and a production result that collapses. Lessons 3 and 10 told you how to measure honestly; this lesson tells you how to prepare data so the measurement stays honest, and how to deal with rare classes where accuracy is useless.

## The idea in plain words

**Features are the model's eyes.** The model only sees numbers. If one column spans 10,000 to 150,000 and another 20 to 40, an algorithm that uses distances or gradients will treat the big-number column as more important simply because its numbers are big. So we put columns on a comparable scale. Categories ("red", "blue") are not numbers at all; we must choose a numeric representation that does not invent an order that is not there.

**Holes carry information.** The reason a value is missing is itself a clue. If rich people tend to skip the income question, then "income missing" predicts something. Throwing away every row with a hole may discard the very pattern you need, so we usually fill holes and also remember where they were.

**Rare classes drown in the crowd.** If 999 of 1000 examples are normal, the cheapest way for a model to be "mostly right" is to ignore the rare class. We can fight this by showing the rare class more often (resampling), by making its mistakes count more (class weights), or by changing the decision cut-off afterwards (threshold moving). We must also use metrics that look at the rare class.

**Leakage is peeking.** Imagine a student who studies from the exam paper. Their score on that exam tells you nothing about the next one. Any time the model (or *you*, while choosing features, scaling, or filling holes) gets to look at the test data, or at the future, or at a proxy of the answer, your score is the score of a student who saw the paper. The cure is a discipline: **split first; learn every statistic from the training split only; apply it unchanged to the rest.**

## Scaling

Let $x$ be a numeric feature. Common transforms:

- **Min-max:** $x' = (x-\min x)/(\max x-\min x)\in[0,1]$.
- **Standardisation (z-score):** $x'=(x-\bar x)/\sigma$, giving mean 0 and standard deviation 1.
- **Log transform:** $x'=\log(1+x)$ to tame right-skewed features (income, counts).

Scaling matters for anything with distances or gradient steps (k-means from lesson 4, logistic regression, SVMs, neural networks) and is nearly irrelevant for decision trees. Two warnings from the DMLS chapter: scaling needs *global statistics* (mean, std, min, max), and these must be computed on the training split and **reused** at inference; and if production data drifts away from the training statistics, the transform degrades, so models need retraining.

### Worked example 1: the same outlier, two scalers

Train values $10,12,14$ and one test value $30$.

- Fit on train only: $\mu=12$, $\sigma=\sqrt{8/3}=1.633$. The test point has $z=(30-12)/1.633=11.02$: clearly an outlier, as it should look.
- Fit on train and test together (the leaky way): $\mu=16.5$, $\sigma=\sqrt{62.75}=7.92$, so $z=(30-16.5)/7.92=1.70$. The outlier looks ordinary because it influenced its own yardstick.

This is mild, but the principle is exact: the test row has changed the numbers the model is given.

## Encoding categories

**One-hot encoding** turns a category with $m$ values into $m$ binary columns, exactly one of which is 1. It invents no order. Cost: $m$ columns, which explodes for high-cardinality features (a million brands). **Label encoding** (red=0, blue=1, green=2) is compact but suggests that green > blue > red; only tree models can safely ignore this. **Target encoding** replaces a category by the mean of the target in that category; it is powerful but a classic leak, because a row's own label is inside its encoding unless you compute it on training data only (and preferably out-of-fold).

Production adds a twist (DMLS): categories change over time. A model that has never seen a new brand has either to crash or to lump it with an "UNKNOWN" bucket that it never trained on. In the one-hot code later, an unseen value simply produces an all-zero row. For open-ended categories the **hashing trick** maps each string into one of a fixed number of buckets, $h(s)=\bigl(\sum_i \mathrm{ord}(s_i)\,31^i\bigr)\bmod B$. With a hash space of 18 bits there are $2^{18}=262{,}144$ buckets, new categories always get a valid index, and collisions are random rather than systematic. The price is collisions and loss of interpretability.

## Feature crosses and other constructions

A **feature cross** combines features into one: "marital status" and "number of children" become "Single, 0", "Married, 2", and so on. It lets a linear model represent an interaction it could not otherwise express, such as "married with two children is different from the sum of 'married' and 'two children'". The caveat is size: crossing a feature with 100 values and one with 100 values gives 10,000 values, which needs a lot more data and invites overfitting. Other classical constructions: polynomial terms ($x^2$, $x_1x_2$), ratios such as BMI $=$ weight / height$^2$, **binning** (turning a continuous value into buckets, at the price of discontinuities at the boundaries), and domain features such as "is rush hour". There is a trade-off between **generalisation** and **specificity**: `is_rush_hour` generalises better than `hour_of_day` but throws information away.

## Missing values

The DMLS chapter distinguishes three mechanisms:

- **MNAR** (missing not at random): the value is missing *because of its own value* (high earners hide income).
- **MAR** (missing at random): missingness depends on *another observed variable* (people of one group skip the age question).
- **MCAR** (completely at random): no pattern; rare in practice.

Options: **deletion** of a column or of rows (only reasonable for MCAR and a tiny fraction; otherwise it loses information and can bias the sample), or **imputation** with a default, the mean, the median or the mode. Avoid filling with a value that is itself plausible (filling "number of children" with 0 confuses "missing" with "none"). A robust recipe: impute with the training median and add a 0/1 **missingness indicator** per column so the model can use the pattern.

## Feature selection

More features are not always better: they add noise, memory, latency and opportunities for leakage. The repo lesson groups methods in three families:

- **Filter** (look at each feature, no model): drop near-constant features (variance threshold), drop one of two highly correlated features, rank by correlation or **mutual information** $I(X;Y)=\sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$, which also catches nonlinear relationships.
- **Wrapper** (use a model to score subsets): recursive feature elimination trains, drops the weakest feature, and repeats; accurate but expensive.
- **Embedded** (selection happens during training): L1 regularisation drives weights to exactly zero; tree importances; **permutation importance** (shuffle one feature on validation data and see how much the score drops) avoids tree importance's bias towards high-cardinality features.

DMLS adds a practical criterion for a good feature: *importance* (how much the model relies on it) and *generalisation* (does it exist at prediction time with similar coverage and values?). An ID column has no generalisation value; `day_of_week` is useless if the training window contains only Monday to Saturday and you must predict Sunday. A feature whose coverage differs wildly between train and test splits is a sign that the splits are not comparable, or that the feature leaks.

**Selection is training.** Choosing features by looking at the labels is part of fitting the model, so it must happen inside the training split. That fact is the heart of the next section.

## Data leakage

**Definition.** Leakage is when information from outside the training data (the test set, the future, or the label itself) is used to build the model, so that evaluation overestimates real performance. The DMLS chapter lists common causes:

1. **Random split of time-correlated data.** Predicting day 7 from day 1 to 6 is a different problem from predicting a random day. If you split randomly, the market on day 7 leaks into training. Split by time.
2. **Scaling before splitting** (worked example 1): the statistics of the test rows enter the model.
3. **Imputing with statistics from the whole dataset**: same mechanism.
4. **Duplicates across splits.** Near-copies in train and test let the model memorise. Deduplicate before splitting, and if you oversample, do it **after** splitting (oversampling creates duplicates!).
5. **Group leakage.** Two scans of the same patient, in train and in test.
6. **Leakage through the data generating process.** A model "detects" disease from the scanner model used at the sicker hospital.
7. **Feature selection or target encoding using labels from the test rows.**
8. **A feature that is a consequence of the label** (a "claim paid" column in a model predicting whether a claim will be filed), or two features that together reveal it (start date and end date predict tenure).

**Detection.** A suspiciously high score; a single feature with unusually high predictive power; ablation studies (remove a feature, see a big drop); a large jump after adding a feature; and touching the test set more than once (every peek turns it into a second validation set).

**Prevention**, in one rule: *the test set is sealed; anything that is learned (means, medians, vocabularies, selected columns, encodings, thresholds) is learned on train and applied to the rest; split by time or by group when the data requires it.* The pipeline view is: `split -> fit transformers on train -> transform train/val/test -> fit model -> evaluate`.

### Worked example 2: leakage on pure noise

Take $n=40$ examples with $d=500$ standard-normal features and labels that are random coin flips (balanced): there is nothing to learn. Choose the $k=10$ features most correlated with the label and train a nearest-class-mean classifier on a random half, testing on the other half.

- **Leaky:** select the 10 features using all 40 rows, then split. The selected columns happen to correlate with the labels of the test rows too, because those rows took part in the choice. Mean test accuracy over 20 seeds: about **0.91**.
- **Honest:** split first, select using the 20 training rows only. Mean test accuracy over 20 seeds: about **0.47**, i.e. chance.

Same data, same model, same code for the classifier: a 44-point difference created purely by *when* the labels were consulted. With 500 candidate features, some always look informative on any small sample; picking them with the test rows included just hides the multiple-testing effect. (The exercise reproduces this.)

## Class imbalance

Imbalance is the norm: fraud, churn, disease screening, defect detection, resume screening. DMLS gives three reasons it is hard: too little signal about the rare class (a few-shot problem), a trivial "predict the majority" solution that is hard to beat, and **asymmetric costs** (a missed cancer is far worse than a false alarm). Imbalance can also be an artefact of sampling (the spam that never reached your database) or of labelling errors, so look at where it came from.

### Step 0: use the right metrics

Evaluate with precision, recall, F1, the precision-recall curve, or the Matthews correlation coefficient $\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ (lesson 10). A model that says "no fraud" 1000 times in a set of 990 legitimate and 10 fraudulent transactions has accuracy 0.99 and recall 0. **Never evaluate on resampled data**; evaluate on data with the real class ratio.

### Data-level methods: resampling

- **Random oversampling** copies minority rows until the classes balance. Simple, but exact duplicates invite overfitting.
- **Random undersampling** drops majority rows. Fast, but throws information away.
- **SMOTE** creates new minority points on the segment between a minority point $x$ and one of its minority neighbours $x'$: $x_{\text{new}}=x+u\,(x'-x)$ with $u\sim U(0,1)$. It works in low-dimensional feature spaces; DMLS notes that SMOTE and Tomek links have only been shown to be effective in low dimensions.
- Resampling is part of training: do it on the training split **after** splitting.

### Algorithm-level methods: class weights and cost-sensitive loss

Instead of changing the data, change the loss. Give a training example of class $c$ the weight

$$
w_c=\frac{N}{C\,n_c},
$$

with $N$ examples, $C$ classes and $n_c$ examples in class $c$. In the repo example, 950 negatives and 50 positives give $w_0=1000/(2\cdot950)=0.526$ and $w_1=1000/(2\cdot50)=10$: one missed positive counts as much as 19 missed negatives. For logistic regression the loss becomes $-\sum_i w_{y_i}\bigl[y_i\log p_i+(1-y_i)\log(1-p_i)\bigr]$. In expectation this behaves like oversampling without duplicating rows. A full **cost matrix** (Elkan) generalises it to arbitrary costs $C_{ij}$; **focal loss** instead down-weights examples the model already gets right.

### Threshold moving

Most models output a probability. Rather than fixing 0.5, sweep the threshold on a validation set and choose the one that optimises your metric or minimises expected cost (lesson 10: $c_{FP}/(c_{FP}+c_{FN})$ for calibrated probabilities).

### Worked example 3: a better threshold

Ten validation cases, three positives. Probabilities for the positives: $0.30, 0.25, 0.08$; for the seven negatives: $0.20, 0.15, 0.07, 0.05, 0.04, 0.03, 0.02$. At threshold 0.5 nothing is flagged, so $F_1=0$. Sweeping the distinct values:

| threshold | flagged | TP | FP | $F_1$ |
|---|---|---|---|---|
| 0.30 | 1 | 1 | 0 | 0.50 |
| 0.25 | 2 | 2 | 0 | **0.80** |
| 0.20 | 3 | 2 | 1 | 0.67 |
| 0.08 | 5 | 3 | 2 | 0.75 |

The best threshold is $0.25$. The model's ranking was fine; the default cut-off was wrong.

### Correcting probabilities after rebalancing

If you train on a balanced set but deploy on a population where the positive prior is $\pi_{\text{true}}$ (training prior $\pi_{\text{bal}}$), the posteriors from the balanced model are proportional to the training priors. Following Bishop's recipe: divide by the training class fractions, multiply by the true ones, renormalise:

$$
p_{\text{true}}(1\mid x)=\frac{p_{\text{bal}}(1\mid x)\,\pi_{\text{true}}/\pi_{\text{bal}}}{p_{\text{bal}}(1\mid x)\,\pi_{\text{true}}/\pi_{\text{bal}}+p_{\text{bal}}(0\mid x)\,(1-\pi_{\text{true}})/(1-\pi_{\text{bal}})}.
$$

For a model trained 50/50 that outputs $0.8$ in a population with 1% positives: numerator $0.8\cdot0.02=0.016$, other term $0.2\cdot1.98=0.396$, result $0.016/0.412=0.039$. A "80% sure" flag from the balanced model means only about 4% in the real world, a number you need when you budget manual reviews.

## Implementation

```python
import numpy as np

class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        self.std_ = np.where(s == 0, 1.0, s)      # constant column: avoid 0/0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_

# the only correct order
# scaler = Standardizer().fit(X_train)
# X_train, X_val, X_test = (scaler.transform(A) for A in (X_train, X_val, X_test))
```

## Shapes

`X: (N, d)`, `mean_, std_: (d,)`, one-hot of `n` values and `m` categories: `(n, m)`, imputer output `(n, 2d)` (filled values plus indicators), cross of two columns of length `n`: a list of `n` strings, class weights: one scalar per class.

## Common misconceptions

- **"Scaling before the split is harmless, it's just preprocessing."** It leaks test statistics (and any other learned statistic does too).
- **"Resampling the whole dataset first and then splitting is fine."** Duplicates and synthetic points end up on both sides and inflate the test score.
- **"Imbalance must always be fixed."** If the metric and threshold are right, a model can learn the true imbalance; resampling distorts probabilities.
- **"Delete rows with missing values, it's cleaner."** It biases the sample unless values are MCAR and rare, and the missingness may be informative.
- **"More features can't hurt."** They raise overfitting, cost and leakage risk.
- **"A great validation score is good news."** An unexpectedly great score is the first sign to look for leakage.

## Key takeaways

- Split first; learn scalers, imputers, vocabularies, encodings and selected features on the training split only.
- Encode categories without inventing order; plan for unseen categories; crosses add interactions at the price of size.
- Missing values are information: impute and add indicators instead of deleting.
- Under imbalance use PR/F1/MCC, class weights or resampling on the training split only, and move the threshold.
- Suspect leakage whenever a score is too good; split by time or group when the data demands it.

## Explain it yourself

1. Why does selecting the 10 best-correlated features on the whole dataset inflate the test score even when labels are random?
2. A model trained on a 50/50 resampled set says 0.8. What must you do before reading that as a probability for real traffic, and why?

## Further reading

- *Designing Machine Learning Systems*, Ch. 4 (class imbalance) and Ch. 5 (feature engineering, data leakage).
- Source library: `ai-engineering-from-scratch` phase 02, lessons 08, 17, 18.

## Mastery criteria

You can build train-only preprocessing transformers, explain MNAR/MAR/MCAR, choose between resampling, class weights and threshold moving, correct posteriors for a changed prior, and design a pipeline that cannot leak.
