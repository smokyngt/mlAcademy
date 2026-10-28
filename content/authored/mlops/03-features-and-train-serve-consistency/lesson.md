---
id: ops.03
title: "Feature pipelines, leakage and train/serve consistency"
summary: Build features whose training-time and serving-time versions agree, recognise the leakage patterns that creep into pipelines, and write a skew checker that compares the two worlds automatically.
minutes: 65
teaches: [ops.feature-pipelines, ops.skew]
requires: [ml.features, ml.leakage, ops.sampling]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.5 Feature Engineering: learned versus engineered features, scaling, discretization, categorical encoding and the hashing trick, missing values, feature crossing (PDF p.157-172, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.5: Data Leakage (causes and detection) and Engineering Good Features (importance, generalization, coverage) (PDF p.172-178). READ."
---

## Where this fits

[[cml.11]] taught what features are and what leakage looks like on a single notebook dataset. Production changes the problem. The *same* feature is now computed twice: once by an offline job that builds the training set, and again by a service that must answer a request in milliseconds. If the two computations differ even slightly, the model that looked excellent offline behaves differently online, and nothing crashes. This lesson is about the pipeline: consistent transformations, safe splits, and automated checks. It uses the batch/stream distinction from [[ops.02]] and prepares the monitoring of [[ops.06]], where the very same checks run continuously on live traffic.

## The idea in plain words

A recipe card says "add a pinch of salt". In the kitchen at home (training) you use your fingers; in the restaurant (serving) the new cook uses a measuring spoon and a smaller pinch. Each cook followed the card, yet the dish tastes different. The model is the diner who got used to the home version.

Leakage is a different failure: the home cook peeked at the diners' final verdict while cooking. The dish scores brilliantly in the home tasting and is mediocre at the restaurant, where nobody can tell you the verdict in advance.

Both failures have one cure: write the recipe once, use the same code in both kitchens, never look at information that will not exist at serving time, and weigh the dish at both places to see whether they agree.

## Learned versus engineered features

Deep networks can learn features from raw pixels, audio or text, which removes much manual work. But most production ML is not deep learning on raw signals, and even deep models are fed engineered inputs (user history, counts, ratios). A classic text classifier needed lowercasing, stemming and word counts by hand; a modern one learns its representation, yet still consumes things like "number of purchases in the last 7 days" that someone has to define. Feature engineering remains one of the highest-leverage activities: once a workable model exists, better features often beat cleverer algorithms.

## Common operations, with the production catch

| Operation | What it does | Pipeline catch |
|---|---|---|
| Missing values | Impute or delete. Values can be missing not at random (the value itself caused the gap), at random (another variable explains it) or completely at random (rare). | Fill values must come from the training split only. Missingness itself can be informative and worth a flag feature. |
| Scaling | Map a feature to a comparable range, e.g. standardise $(x-\mu)/\sigma$ or min-max. | $\mu,\sigma$ are global statistics: fit on train, store them, reuse them at serving. If new data drifts, the stored statistics become stale. |
| Discretization | Bucket a continuous value. | Creates cliffs at bucket edges; choose edges from training quantiles and keep them fixed. |
| Categorical encoding | Map categories to integers or vectors. | Categories change in production (new brands appear daily). An `UNKNOWN` bucket stops crashes but a model never trained on it will ignore it. The **hashing trick** assigns each category the hash of its name modulo a fixed space (for example $2^{18}$ slots), so unseen categories are handled with no vocabulary; collisions are random and, according to the book's cited study, even 50% colliding features cost under 0.5% in log loss. |
| Feature crossing | Combine features (marital status x children) to give simple models non-linear power. | The cross multiplies the cardinality and sparsity. |

## Data leakage

**Leakage** means information about the label (or about the future) is present in the features at training time that will not be available at serving time. It is dangerous because evaluation looks great. The causes catalogued by the book:

1. **Splitting time-correlated data randomly.** Rows from the same day share trends. A random split puts day-7 rows in training and day-7 rows in test. Split by time: train on the first four weeks, test on week five.
2. **Scaling (or any fitting) before splitting.** Computing means or variances on the full dataset lets test statistics into training. Split first, fit on train, transform everything with those statistics. The same holds for imputation values, vocabularies, PCA, and target encodings.
3. **Filling missing values with statistics from the test split**, same mechanism.
4. **Duplicates and near-duplicates across splits.** Typical after merging sources or after oversampling; the book reports 3.3% of CIFAR-10 test images and 10% of CIFAR-100 test images having duplicates in the training set. Deduplicate before splitting, check again after, and oversample only after splitting.
5. **Group leakage.** Rows that share a label-determining entity (two scans of one patient, frames of one object) must be in the same split.
6. **Leakage from the data generation process.** The model learns the artefact of how the data was collected (the book's example: a scanner type that correlates with doctors suspecting cancer), which does not exist at the second hospital. Only knowledge of how data is collected finds this; normalise across sources and involve domain experts.

**Detecting leakage.** Measure each feature's predictive power and investigate suspiciously strong ones (two features may only leak together, such as start date and end date for tenure). Run ablation studies: remove a feature and see how much performance drops. Treat any new feature that suddenly boosts the metric with suspicion. And look at the test set only to report the final number: using it to design features or tune hyper-parameters leaks it.

## Engineering good features

More features are not automatically better: more chances for leakage, more overfitting, more memory and latency at serving, and every unused feature is technical debt that must be maintained whenever the upstream data changes. Judge a feature on two axes.

- **Importance**: how much does performance drop without it (feature importance, SHAP values, permutation or ablation)? Often a few features carry most of the importance.
- **Generalization**: will it behave the same on unseen data? Check **coverage** (the share of rows that have a value; a feature present in 1% of rows is rarely useful unless the missingness is itself informative) and the **distribution of values**. If coverage in train is 90% and in test 20%, the splits differ. If the values seen at training do not overlap those at test time (the book's example: `DAY_OF_THE_WEEK` has Monday-Saturday in training and Sunday at test time), the feature can hurt.

## Feature stores and shared code

A **feature store** is infrastructure that defines each feature once and serves it both to offline training jobs (historical values as of each training example's time) and to online services (current values at low latency). The book defers feature stores to its infrastructure chapter, which is not part of this lesson's reading; the point to retain here is the principle they implement: **one definition, two consumers**. Without a store the same discipline can be achieved by sharing a transformation library between the training job and the service, versioning it with the model, and testing that both paths give the same outputs for the same raw input.

## Train/serve skew and how to check it

**Skew** is any systematic difference between the features the model saw in training and those it receives when served. Causes: the two pipelines are separate code (Python batch job versus a service in another language), different units (age in years versus months), stale statistics, missing upstream data, new categories, and different time semantics (a feature computed "as of now" at serving but "as of the label date" in training).

A practical **skew checker** compares a training matrix with a sample of served features, per feature:

- **missing**: the NaN rate differs by more than a tolerance.
- **shift**: the mean moves by more than `mean_z` training standard deviations, $|\bar x_{\text{serve}}-\bar x_{\text{train}}|/\sigma_{\text{train}}>z$.
- **range**: more than a tolerance of serving values fall outside the training minimum and maximum.
- for categoricals, the **unseen-category rate**, the share of served values never seen in training.

These checks need no labels, so they can run on live traffic. They are cheap, crude and effective. They do not prove the absence of skew (two distributions can share a mean and differ in shape); [[ops.06]] adds the KS test and the PSI for that.

## Worked example

All numbers come from the exercise. Train on 2,000 rows of three features: `age ~ N(30, 5)`, `income ~ N(100, 20)`, `z ~ N(0, 1)`. Take 1,500 clean serving rows from the same distributions: the checker reports no issues. Now corrupt the serving data three ways:

- age multiplied by 12 (months instead of years) puts the serving mean about 66 training standard deviations away (360 versus 30, with sd 5) and almost every value above the training maximum, so the issues are `["range", "shift"]`;
- 300 of the 1,500 income values replaced by NaN (20% missing, versus 0% in training) gives `["missing"]` only: the mean of the non-missing values is unchanged;
- `z` shifted by +3 gives `["range", "shift"]`.

The checker returns `{"age": ["range","shift"], "income": ["missing"], "z": ["range","shift"]}`. In the same exercise, a time split of ten rows with timestamps 5,1,4,2,3,10,9,8,7,6 and `test_frac=0.3` sends timestamps 8, 9, 10 to test and 1 to 7 to training, so every training row is earlier than every test row.

## Implementation

```python
import numpy as np

def fit_scaler(X):
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0)     # TRAIN split only

def apply_scaler(X, mean, std):
    return (X - mean) / std                                 # reuse at serving

def time_split(timestamps, test_frac):
    order = np.argsort(timestamps, kind="stable")
    n_train = int(round(len(order) * (1 - test_frac)))
    return order[:n_train], order[n_train:]
```

Libraries: scikit-learn `Pipeline` objects bundle fitted transformers with the model so the fitted statistics travel together; `TimeSeriesSplit` for time-ordered splits; `GroupKFold` for group leakage; feature stores such as Feast; data-validation tools that compare training and serving statistics.

## Shapes

`X: (n, d)` features; `mean, std: (d,)`; a skew report maps feature name to a list of issues; `time_split` returns two index vectors into the original `n` rows.

## Common misconceptions

- "Normalise the whole dataset once, then split." That leaks the test statistics. Fit on train.
- "If the offline metric is high, the pipeline is right." Leakage makes offline metrics too good; skew makes online worse than offline.
- "Unknown-category bucket solves new categories." It stops the crash, but a model that never saw the bucket in training ignores it.
- "Same mean means no skew." Means hide variance, shape and range changes.
- "Feature stores remove the need for tests." They help, but the two code paths must still be checked for agreement.

## Key takeaways

- Define each transformation once and reuse the fitted parameters at serving.
- Split by time for time-correlated data; fit scalers, imputers and encoders on the train split only; deduplicate before splitting; keep groups together.
- Suspect any feature that is too good; do ablations; touch the test set once.
- Judge features on importance and generalization (coverage and value distribution).
- A label-free skew checker (missing rate, mean shift, range violations, unseen categories) catches the most common train/serve mismatches.

## Explain it yourself

1. Give three different ways a test set can leak into training through preprocessing, and the fix for each.
2. Why can an `UNKNOWN` category fail silently, and how does hashing change that?

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch.5 (PDF p.157-179) and Ch.10 for feature stores (not read for this lesson).
- [[cml.11]] for the notebook-level treatment of features, imbalance and leakage.

## Mastery criteria

You can list the six leakage patterns with a fix for each, explain why every fitted statistic must come from the training split, design the offline/online paths of a feature so they share code, and implement and interpret a skew checker.
