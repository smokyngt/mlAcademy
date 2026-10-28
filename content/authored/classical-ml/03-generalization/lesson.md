---
id: cml.03
title: Generalisation, bias-variance and cross-validation
summary: Why a model that fits the training data perfectly can be useless, how to measure that honestly, and how to choose model complexity.
minutes: 65
teaches: [ml.generalization]
requires: [ml.linreg, ml.ridge]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 1.1 Polynomial Curve Fitting, Sec. 3.2 The Bias-Variance Decomposition, PDF p.24 and p.167
  - math/math for ML.pdf (Ch. 8 When Models Meet Data: empirical risk minimisation, cross-validation, PDF p.257)
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/10-bias-variance/docs/en.md
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/09-model-evaluation/docs/en.md
---

## Where this fits

Everything so far minimised an error **on the data we had**. But a model is useful only on data it has *not* seen: tomorrow's customers, new images, new sentences. A model flexible enough to memorise its training set has zero training error and can still be terrible in practice. This lesson is about telling the two apart, and it is the single most important discipline in applied machine learning: every honest result you will ever report was measured on data the model never trained on.

## The idea in plain words

Imagine a student preparing for an exam. One student learns the *method* behind the exercises (a good fit to the idea, good on new questions). Another memorises the answers to last year's exam word for word (perfect on last year's paper, lost on this year's). A student who barely studied does badly on both. The memoriser is **overfitting**, the one who barely studied is **underfitting**, and the only way to tell them apart is to give them a *new* exam.

In model terms: a too-simple model is wrong in a consistent way (high **bias**); a too-flexible model changes a lot depending on which training examples it happened to see (high **variance**). Good models balance them.

## The measuring protocol

Split the data into three parts and respect them:

- **Training set**: used to fit the weights.
- **Validation set**: used to choose settings (polynomial degree, `λ`, learning rate...). Because you choose based on it, it slowly stops being independent.
- **Test set**: touched **once**, at the very end, to report the final number.

When data is scarce use **k-fold cross-validation**: cut the data into `k` folds; for each fold, train on the other `k−1` and measure on it; average the `k` errors. Every point is used for validation exactly once.

## A real experiment

Targets from `f(x) = x³ − x` plus noise of standard deviation 0.1 (variance 0.01), 30 training points, 500 fresh test points, polynomial fits by least squares:

| Degree | Train MSE | Test MSE | Reading |
|---|---|---|---|
| 1 | 0.042 | 0.036 | too simple: both errors high (**underfit**) |
| 3 | 0.0081 | 0.0103 | right complexity: test error ≈ noise floor 0.01 |
| 9 | 0.0045 | 0.0153 | train falls, test rises: **overfit** |
| 15 | 0.0035 | 0.0173 | even better on train, worse on test |

Training error *always* decreases with complexity; test error has a U shape. The noise variance `σ² = 0.01` is a floor nothing can beat.

## Derivation: the bias-variance decomposition

Fix a test point `x`. The truth is `y = f(x) + ε`, with `E[ε] = 0`, `Var(ε) = σ²`. Our learned predictor `f̂(x)` depends on the random training set. Write `f̄ = E[f̂(x)]` (average over training sets). Add and subtract:

$$
y-\hat f=(f-\bar f)+(\bar f-\hat f)+\varepsilon .
$$

Square and take expectations. The cross terms vanish: `ε` is independent of `f̂` with mean 0, and `E[\bar f - \hat f]=0`. What remains:

$$
\boxed{E\big[(y-\hat f)^2\big]=\underbrace{(f-\bar f)^2}_{\text{bias}^2}+\underbrace{E\big[(\hat f-\bar f)^2\big]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{irreducible noise}}.}
$$

More flexible models lower the bias and raise the variance; regularisation (ridge, lesson 1) does the opposite. The best test error sits at a compromise.

## Choosing settings

Pick the degree, `λ`, or any hyper-parameter with the **lowest cross-validation error**, then retrain on all training data. In the experiment above, 5-fold CV on 60 points gives errors 0.031 (degree 1), 0.037 (2), 0.0070 (3), 0.0066 (4), then roughly flat around 0.007 to 0.008: the big drop at degree 3 is the signal; the exact best degree among 3-6 is within noise. A common rule is to choose the *simplest* model within one standard error of the best.

## Implementation

```python
import numpy as np

def kfold_indices(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)

def cv_mse(x, y, degree, lam, k=5, seed=0):
    errs = []
    for val in kfold_indices(len(x), k, seed):
        tr = np.setdiff1d(np.arange(len(x)), val)
        A = np.vander(x[tr], degree + 1, increasing=True)
        w = np.linalg.solve(A.T @ A + lam * np.eye(degree + 1), A.T @ y[tr])
        errs.append(np.mean((np.vander(x[val], degree + 1, increasing=True) @ w - y[val]) ** 2))
    return np.mean(errs)
```

## Common misconceptions

- **Training error is not a performance estimate.** It is optimistic, systematically.
- **Data leakage.** Computing normalisation statistics, feature selection, or tuning on the whole data set before splitting contaminates the validation. Do every data-dependent step *inside* each fold.
- **Reusing the test set** to pick between ideas turns it into a validation set. Hold out data you truly never look at.
- **Random splits are not always right**: for time series, split by time; for grouped data (several rows per patient), split by group.
- **More data reduces variance, not bias.** If both train and validation errors are high, collecting more data will not help; use a richer model.

## Key takeaways

- Judge a model only on data it did not train on.
- Test error = bias² + variance + noise; complexity trades the first two.
- Train error always falls with complexity; validation error is U-shaped.
- Choose hyper-parameters by cross-validation; report once on a held-out test set.

## Explain it yourself

1. Both training and validation error are high. What does that tell you, and what would you change?
2. Training error is tiny but validation error is large. Same question.

## Further reading

- Bishop, *PRML*, Sec. 1.1 and 3.2; *Mathematics for Machine Learning*, Ch. 8.
- Source library: `ai-engineering-from-scratch` phase 02, lessons 09-10.

## Mastery criteria

You can derive the decomposition, read a train/validation curve and diagnose under- or over-fitting, and implement k-fold cross-validation without leaking data.
