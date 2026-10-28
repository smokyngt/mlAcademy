---
id: cml.02
title: Logistic regression
summary: Turn a linear model into a probability, train it with cross-entropy, and see why its decision boundary is a straight line.
minutes: 60
teaches: [ml.logreg]
requires: [ml.linreg, opt.gradient-descent, ml.loss]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 4.3 Probabilistic Discriminative Models, PDF p.223
  - math/math for ML.pdf (Sec. 8.3 maximum likelihood; Ch. 12 classification for the margin view, PDF p.257 and p.376)
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/03-logistic-regression/docs/en.md
---

## Where this fits

Many real questions have a yes/no answer: is this email spam, will this customer leave, is this scan abnormal? You do not only want a label, you want a **probability** (so you can set the threshold that suits the cost of mistakes). Logistic regression is the simplest model that outputs one. It is also exactly a neural network with no hidden layer: the loss, the gradient and the training loop you learned in the Machine Learning course are the same ones, so this lesson is a rehearsal on a model small enough to see everything.

## The idea in plain words

Compute a score the way linear regression does (a weighted sum), then squash it into the range 0 to 1 with the S-shaped **sigmoid**, so that a very negative score means "almost surely no", zero means "50-50" and a very positive score means "almost surely yes". Training adjusts the weights so that the predicted probabilities are as *high as possible for what actually happened*, which is maximum likelihood (lesson 8).

Because the sigmoid is monotone, "probability above one half" is the same as "score above zero", and the score is linear in the inputs. So the boundary between the two classes is a **straight line** (a flat hyperplane in higher dimensions). This is both the strength (simple, hard to overfit) and the limit (it cannot separate things that need a curved boundary).

## Formal model

Features `x ∈ R^d` (with a leading 1 for the intercept), weights `w ∈ R^d`, labels `y ∈ {0,1}`:

$$
z=w^\top x,\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}=P(y=1\mid x).
$$

For `N` examples, `z = Xw ∈ R^N` and `p = σ(z)`.

## Derivation: loss and gradient

Likelihood of one label: `p^y (1−p)^{1−y}`. The mean negative log-likelihood (binary cross-entropy):

$$
L(w)=-\frac1N\sum_i\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big].
$$

From lesson 8, the derivative with respect to the score is `∂L/∂z_i = (p_i − y_i)/N`, and `z = Xw` gives `∂z_i/∂w_j = X_{ij}`, so (lesson 6's linear-layer rule)

$$
\boxed{\nabla_wL=\frac1N X^\top (p-y)}\qquad(d\times N)(N)=(d).
$$

Compare it with linear regression, `Xᵀ(Xw − y)`: it is the *same* formula with the prediction replaced by `p`. Unlike least squares there is no closed form (the equation `Xᵀ(p − y) = 0` is nonlinear in `w`), so we use gradient descent. The loss is still convex (its Hessian is `Xᵀ diag(p(1−p)) X ⪰ 0`), so gradient descent with a small enough rate reaches the global minimum.

**Regularisation.** Add `(λ/2)‖w‖²` (not on the intercept): the gradient gains `+ λw`. This matters more than for linear regression: if the classes are perfectly separable, the likelihood keeps improving as `‖w‖ → ∞` (probabilities pushed to 0 and 1) and no finite optimum exists. The penalty makes the optimum finite.

## Worked example

One example `x = (1, 2)` (intercept and one feature), label `y = 1`, weights `w = (−1, 0.5)`.

- `z = −1·1 + 0.5·2 = 0`, so `p = σ(0) = 0.5`.
- Loss `= −ln 0.5 = 0.693`, the "know nothing" value from lesson 8.
- Gradient `= (p − y) x = (−0.5)(1, 2) = (−0.5, −1)`.
- A step with `η = 1` gives `w = (−0.5, 1.5)`: now `z = −0.5 + 3 = 2.5`, `p = 0.924`. The model became more confident in the right answer.

## Implementation

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def loss(w, X, y):
    z = X @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)   # stable cross-entropy

def grad(w, X, y):
    return X.T @ (sigmoid(X @ w) - y) / len(y)
```

`log(1+e^z) − yz` is the cross-entropy rewritten so that it never takes the log of 0.

## Shapes

`X: (N, d)`, `w: (d)`, `z, p, y: (N)`, gradient `(d)`. Decision rule: predict class 1 when `Xw > 0`, i.e. `p > 0.5`.

## Common misconceptions

- **It is a classifier, not a regression** in spite of the name: the regression is on the log-odds `ln(p/(1−p)) = w·x`, which is linear.
- **The probabilities are only as good as the model.** Without calibration checks, `0.9` need not mean "right 90% of the time".
- **Accuracy at threshold 0.5 hides information.** Change the threshold to trade false positives for false negatives.
- **Unscaled features** slow gradient descent badly (lesson 7: ill-conditioning). Standardise inputs.

## Key takeaways

- `p = σ(Xw)`; the loss is cross-entropy; the gradient is `Xᵀ(p − y)/N`.
- No closed form, but the loss is convex: gradient descent finds the global optimum.
- The decision boundary `w·x = 0` is linear.
- On separable data the weights diverge without regularisation.

## Explain it yourself

1. Why is the boundary of logistic regression a straight line even though the model is nonlinear (it has a sigmoid)?
2. Why does separable data make the unregularised weights grow forever?

## Further reading

- Bishop, *PRML*, Sec. 4.3.2-4.3.3 (PDF p.223).
- Source library: `ai-engineering-from-scratch` phase 02, lesson 03.

## Mastery criteria

You can derive `Xᵀ(p − y)/N` from the loss, implement a stable version, verify it numerically, and explain what regularisation changes.
