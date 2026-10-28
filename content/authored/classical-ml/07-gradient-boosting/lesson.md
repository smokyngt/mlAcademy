---
id: cml.07
title: Gradient boosting
summary: Build a strong model as a sum of many weak trees, each one fitted to what the sum so far still gets wrong, and understand why this is gradient descent in function space.
minutes: 80
teaches: [ml.boosting]
requires: [ml.trees, ml.bagging, opt.gradient-descent, ml.loss]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 14.3 Boosting incl. 14.3.1-14.3.2 (PDF p.677-683): AdaBoost algorithm, exponential error derivation, comparison of error functions, extension to squared error and residuals. READ."
  - "Bishop, PRML, Sec. 14.2 (PDF p.676-677): why plain averaging of correlated models helps little, motivating boosting. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/11-ensemble-methods/docs/en.md (boosting, AdaBoost, gradient boosting, shrinkage). READ (concept sections)."
---

## Where this fits

A random forest ([[cml.06]]) fights *variance*: it averages deep, low-bias trees. Boosting fights *bias*: it starts from very simple, weak trees and builds a complex model by adding corrections. Gradient-boosted trees (XGBoost, LightGBM, CatBoost and scikit-learn's variants all implement the same idea with engineering on top) are the usual winner on tabular problems: click-through prediction, credit scoring, demand forecasting, ranking. The core idea is also a general recipe: optimise a loss by **gradient descent, but where the parameters being updated are the model's predictions** and each step is realised by training a small model. Once you see this, losses other than squared error come for free.

## The idea in plain words

Imagine tutoring a student by exam. After the first exam you do not re-teach everything; you look at which questions they got wrong and design the next lesson only on those. After the next exam you again target what is *still* wrong. Each lesson is simple and short, but the sum of many targeted lessons is thorough.

In boosting the "student" is the current prediction and each "lesson" is a small tree. For regression, "what is still wrong" is literally the **residual**: true value minus current prediction. The next tree is trained to predict those residuals, and its output is added (scaled down a bit) to the running prediction. For classification, AdaBoost expresses "still wrong" differently: it increases the *weight* of misclassified examples so the next weak classifier concentrates on them.

Contrast with bagging: bagging trains trees independently and in parallel and averages them; boosting trains them one after another, each depending on the previous ones, and adds them.

## Stagewise additive models

Boosting builds

$$
F_M(x)=F_0+\sum_{m=1}^{M}\nu\, h_m(x),
$$

a sum of base learners $h_m$ (here small trees) with a **learning rate** (shrinkage) $\nu\in(0,1]$. "Stagewise" means that at stage $m$ we freeze $F_{m-1}$ and choose only the new term; we never go back and re-fit earlier terms. That makes each step a small, tractable fitting problem even though the final model has hundreds of terms.

## AdaBoost from exponential loss

Bishop (Sec. 14.3, with targets $t_n\in\{-1,+1\}$) shows that AdaBoost is exactly stagewise minimisation of the **exponential error** $E=\sum_n e^{-t_n f_m(x_n)}$ with $f_m=\tfrac12\sum_{l\le m}\alpha_l y_l$. Freeze the first $m-1$ terms and define $w_n^{(m)}=e^{-t_nf_{m-1}(x_n)}$, constants for this stage. Then

$$
E=\sum_n w_n^{(m)}\exp\!\big(-\tfrac12 t_n\alpha_m y_m(x_n)\big).
$$

Split points into correctly classified ($t_ny_m=+1$) and misclassified ($t_ny_m=-1$):

$$
E=e^{-\alpha_m/2}\sum_{\text{correct}}w_n+e^{\alpha_m/2}\sum_{\text{wrong}}w_n .
$$

Let $\epsilon_m$ be the weighted error rate (weights of the wrong points over all weights) and divide by $\sum_n w_n$:

$$
\frac{E}{\sum w}=e^{-\alpha/2}(1-\epsilon)+e^{\alpha/2}\epsilon .
$$

- Over the classifier $y_m$: this is smallest when $\epsilon$ is smallest, so fit the weak learner by **weighted** misclassification error.
- Over $\alpha$: set the derivative to zero, $-\tfrac12e^{-\alpha/2}(1-\epsilon)+\tfrac12e^{\alpha/2}\epsilon=0$, so $e^{\alpha}=\frac{1-\epsilon}{\epsilon}$:

$$
\alpha_m=\ln\frac{1-\epsilon_m}{\epsilon_m}.
$$

Accurate learners ($\epsilon$ near $0$) get a large vote, and a learner no better than chance ($\epsilon=\tfrac12$) gets $\alpha=0$. Updating the weights, $w^{(m+1)}_n=w^{(m)}_n e^{-\frac12t_n\alpha_my_m(x_n)}$, and dropping a factor common to all $n$ gives Bishop's Eq. 14.18: **multiply the weight of each misclassified point by $e^{\alpha_m}$**. The final classifier is $\operatorname{sign}\big(\sum_m\alpha_my_m(x)\big)$.

Limits (Bishop Sec. 14.3.2): the exponential loss grows exponentially for badly misclassified points, so AdaBoost is much less robust to outliers and mislabelled data than cross-entropy, and it does not generalise cleanly to a probabilistic multiclass model.

## Derivation: gradient boosting with squared loss

Now regression, with a general loss. We want to minimise $\mathcal L(F)=\sum_{i=1}^N \ell\big(y_i,F(x_i)\big)$. Treat the $N$ numbers $F_i=F(x_i)$ as the parameters. Ordinary gradient descent on them would be

$$
F_i\leftarrow F_i-\eta\,\frac{\partial\ell(y_i,F_i)}{\partial F_i}.
$$

We cannot set the $F_i$ freely, because we need a function that also works on new points. So we **fit a model $h$ to the negative gradient** $-g_i$, with $g_i=\partial\ell/\partial F_i$, by least squares, then step along $h$: $F\leftarrow F+\nu h$. This is gradient descent in function space.

For the squared loss $\ell=\tfrac12(y-F)^2$ the derivative is $g_i=-(y_i-F_i)$, so the negative gradient is exactly the residual:

$$
r_i=y_i-F_{m-1}(x_i).
$$

**Why that is also the best possible new term.** At stage $m$ we choose $h$ to minimise $\sum_i\big(y_i-F_{m-1}(x_i)-h(x_i)\big)^2=\sum_i (r_i-h(x_i))^2$: that is *ordinary regression of $h$ on the residuals*. So with squared loss the gradient view and the "fit what is left over" view coincide (Bishop makes the same remark for sum-of-squares: each new learner is fitted to the residuals $t_n-f_{m-1}(x_n)$).

The algorithm:

1. $F_0=\bar y$ (the constant minimising squared error).
2. For $m=1..M$: compute $r_i=y_i-F_{m-1}(x_i)$; fit a small regression tree $h_m$ to $(x_i,r_i)$; set $F_m=F_{m-1}+\nu h_m$.

For other losses only the residual changes: for absolute error $\ell=|y-F|$ the negative gradient is $\operatorname{sign}(y-F)$, a robust alternative (Bishop notes squared error is not robust to outliers and the absolute deviation is a remedy); for logistic loss it is $y-\sigma(F)$, the same "error" you saw in logistic regression ([[cml.02]]). Leaf values are then refined by a line search per leaf in the full algorithm; we skip that with squared loss since leaf means are already optimal.

## Controlling overfitting

Unlike a forest, a boosted model does overfit if you keep adding rounds. The levers:

- **Shrinkage $\nu$** (e.g. 0.01-0.3): each tree only moves the prediction part of the way. Small $\nu$ needs more rounds but usually generalises better; it is the most important regulariser.
- **Number of rounds $M$**: choose by validation error (early stopping).
- **Tree depth**: depth 1 (stumps) gives an additive model in the features; depth $d$ allows interactions among up to $d$ features. Depths 3-6 are typical.
- **Leaf size and subsampling** of rows or columns per round add further randomness and regularisation (common in modern libraries).

## Worked example

Six points, $x=1..6$, $y=(1,1.2,0.9,3,3.2,2.8)$, stumps, $\nu=0.5$.

- $F_0=\bar y=2.0167$. Residuals: $(-1.017,-0.817,-1.117,0.983,1.183,0.783)$.
- Best stump: split at $x\le3.5$, leaf means $-0.983$ (left) and $+0.983$ (right). Update: $F_1=2.0167+0.5(\mp0.983)=1.525$ (left) and $2.508$ (right). Training MSE falls from the variance of $y$ to $0.2628$.
- Round 2: the residual means on the two sides are now $\mp0.4917$ (halved, because $\nu=0.5$ moved us half-way). Same split. $F_2=1.279$ / $2.754$, MSE $0.0815$.
- Round 3: $F_3=1.156$ / $2.877$, MSE $0.0362$.

The MSE cannot go below the within-group variance, $0.1267/6=0.0211$ (a stump with one split predicts a constant per side); here the remaining residual is just noise inside each side. Shrinkage shows up as the geometric halving: after $m$ rounds the stump-level residual is $(1-\nu)^m\cdot0.983$.

AdaBoost round: with five equal weights $0.2$ and one mistake, $\epsilon=0.2$, $\alpha=\ln4=1.386$; the mistake's weight becomes $0.2\cdot4=0.8$, the others stay $0.2$; normalised: $(0.125,0.125,0.5,0.125,0.125)$. After reweighting, the misclassified point carries exactly half of the total weight, so the next weak learner is forced to treat it seriously.

## Implementation

```python
import numpy as np

def fit_stump(X, r):
    best = None
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        for t in (u[:-1] + u[1:]) / 2:
            L = X[:, j] <= t
            lv, rv = r[L].mean(), r[~L].mean()
            sse = ((r[L] - lv) ** 2).sum() + ((r[~L] - rv) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, j, t, lv, rv)
    return best[1:]

def gb_fit(X, y, rounds, lr):
    F = np.full(len(y), y.mean())
    stumps = []
    for _ in range(rounds):
        j, t, lv, rv = fit_stump(X, y - F)          # fit the residuals
        F = F + lr * np.where(X[:, j] <= t, lv, rv) # shrunken update
        stumps.append((j, t, lv, rv))
    return y.mean(), stumps
```

## Shapes

`X: (N, d)`, `y, F, r: (N,)`. Each stump is four numbers; the model is $M$ stumps plus the constant. Prediction costs $O(M\cdot\text{depth})$ per point. Training is sequential in $M$ (unlike bagging, rounds cannot be run in parallel).

## Common misconceptions

- **"Boosting reweights data" is only AdaBoost.** Gradient boosting does not reweight rows; it changes the regression *targets* to residuals (negative gradients).
- **"More rounds are always better."** Training loss never increases with rounds, but validation loss eventually does; use early stopping.
- **"Fit the residuals" is only for squared loss.** In general you fit the negative gradient, which equals the residual only for squared error.
- **Deep trees are better base learners.** Weak, shallow trees are the point; very deep trees make each step overfit.
- **Boosting and bagging are interchangeable.** Bagging reduces variance of low-bias learners; boosting reduces bias of high-bias learners (and can increase variance).
- **Noisy labels.** Because boosting keeps concentrating on hard cases, it can chase label noise, especially with the exponential loss.

## Key takeaways

- Boosting builds $F_M=F_0+\nu\sum h_m$ stagewise; each $h_m$ is a weak learner fitted to what remains.
- AdaBoost = stagewise minimisation of the exponential error: $\alpha=\ln\frac{1-\epsilon}{\epsilon}$, wrong points' weights multiplied by $e^\alpha$.
- Gradient boosting = gradient descent in function space: fit $h_m$ to the negative gradient; for squared loss this is the residual.
- Shrinkage, tree depth and the number of rounds are the regularisers; tune them with validation data.

## Explain it yourself

1. Show that for $\ell=\tfrac12(y-F)^2$ the negative gradient with respect to $F_i$ is the residual.
2. Why does a small learning rate usually need more trees but generalise better?

## Further reading

- Bishop, *PRML*, Sec. 14.3 (PDF p.677-683).
- Repo lesson `phases/02-ml-fundamentals/11-ensemble-methods` (also covers stacking and XGBoost-style regularisation).
- Previous: [[cml.06]]; later you will see the same "fit the negative gradient" structure whenever a loss is minimised with iterative updates.

## Mastery criteria

You can derive $\alpha_m$ from the exponential loss, explain why residuals are negative gradients for squared loss, hand-compute a few boosting rounds with stumps, and name the three main regularisation levers.
