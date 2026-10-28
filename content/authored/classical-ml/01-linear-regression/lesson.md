---
id: cml.01
title: Linear regression and the normal equations
summary: Fit a line (or hyperplane) to data in one line of linear algebra, understand why it works geometrically, and tame it with ridge regularisation.
minutes: 60
teaches: [ml.linreg, ml.ridge]
requires: [la.matrix, calc.gradient, ml.loss]
sources:
  - math/math for ML.pdf (Ch. 9 Linear Regression, PDF p.295; Ch. 3 orthogonal projections, PDF p.76)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 3.1 Linear Basis Function Models, PDF p.158
  - ai-engineering-from-scratch/phases/02-ml-fundamentals/02-linear-regression/docs/en.md
---

## Where this fits

Predicting a number from other numbers (a price from a surface and a location, a delay from traffic) is the oldest machine learning task, and the simplest model for it, a weighted sum of the inputs, is still the first thing worth trying. It is also the one model that has an exact answer: no iteration, no learning rate, just linear algebra. Studying it carefully gives you the template for everything else: a model, a loss, a gradient, and a way to check you got it right.

## The idea in plain words

You have a cloud of points and want the straight line that passes "as close as possible" to all of them. Measure closeness by the **vertical gaps** between the points and the line, square them (so gaps above and below do not cancel, and big gaps hurt more), and add them up. The best line is the one with the smallest total.

There is a lovely picture behind the formula: the vector of all your targets lives in a space with one axis per data point. The set of everything your model *can* predict is a flat sheet (a plane, or higher-dimensional) inside that space. The best fit is simply the **shadow of the targets on that sheet**, and the error is the part sticking out of it at a right angle. A right angle is a dot product equal to zero, and that single fact produces the whole solution.

## Formal setup

Data: a matrix `X ∈ R^{N×d}` (rows are examples; add a column of ones to get an intercept) and targets `y ∈ R^N`. Model `ŷ = Xw`, with `w ∈ R^d`. Loss (sum of squares, which has the same minimiser as the mean):

$$
L(w)=\tfrac12\|Xw-y\|^2 .
$$

## Derivation 1: the normal equations

From lesson 5, the gradient is `∇L = Xᵀ(Xw − y)`. At the minimum it is zero:

$$
X^\top(Xw-y)=0\quad\Longleftrightarrow\quad \boxed{X^\top X\,w = X^\top y}\qquad(d\times d)(d)=(d).
$$

If `XᵀX` is invertible, `w = (XᵀX)⁻¹ Xᵀ y`. Why is this a *minimum* and not a maximum? The Hessian of `L` is `XᵀX`, and `vᵀXᵀXv = ‖Xv‖² ≥ 0` for any `v`, so the loss is a convex bowl and the single stationary point is the global minimum.

**Geometric reading.** With the residual `r = y − Xw`, the equation `Xᵀ r = 0` says: *the residual is orthogonal to every column of `X`*. So `Xw` is exactly the orthogonal projection of `y` onto the column space of `X` (lesson 2's projection, with many directions at once).

## Worked example

Points (0, 1), (1, 2), (2, 2). With an intercept, `X = [[1,0],[1,1],[1,2]]`, `y = (1,2,2)`.

- `XᵀX = [[3,3],[3,5]]`, `Xᵀy = (5, 6)`.
- `det = 15 − 9 = 6`, so `(XᵀX)⁻¹ = (1/6)[[5,−3],[−3,3]]`.
- `w = (1/6)(25 − 18, −15 + 18) = (7/6, 1/2) ≈ (1.167, 0.5)`: the line `y = 1.167 + 0.5x`.
- Predictions `(1.167, 1.667, 2.167)`; residuals `(−0.167, 0.333, −0.167)`.
- Check orthogonality: `1·(−0.167)+1·0.333+1·(−0.167) = 0` ✓ and `0·(−0.167)+1·0.333+2·(−0.167) = 0` ✓.

## Derivation 2: ridge regression

When two columns are nearly copies of each other, `XᵀX` is almost singular and the solution swings wildly with tiny changes in the data. **Ridge** adds a penalty on the size of the weights:

$$
L_\lambda(w)=\tfrac12\|Xw-y\|^2+\tfrac{\lambda}{2}\|w\|^2\;\Rightarrow\;\nabla L_\lambda=X^\top(Xw-y)+\lambda w=0\;\Rightarrow\;\boxed{(X^\top X+\lambda I)\,w=X^\top y}.
$$

Adding `λI` lifts every eigenvalue of `XᵀX` by `λ`, so the matrix is always invertible for `λ > 0` and well conditioned (lesson 7: conditioning controls how hard the problem is). The price is a small bias: the weights are shrunk toward zero. Do not penalise the intercept (centre the data, or exclude that column).

## Implementation

```python
import numpy as np

def fit_ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)      # solve, never invert

def fit_ridge(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
```

Use `solve` (or `lstsq`) instead of computing `inv(XᵀX)`: it is faster and numerically safer. For very ill-conditioned problems prefer `np.linalg.lstsq`, which uses an SVD.

## Shapes

| Quantity | Shape |
|---|---|
| `X` | `(N, d)` |
| `XᵀX`, `XᵀX + λI` | `(d, d)` |
| `Xᵀy`, `w` | `(d,)` |
| residual `r = y − Xw` | `(N,)`, and `Xᵀ r = 0` has shape `(d,)` |

## Common misconceptions

- **"Linear" means linear in the weights, not in the inputs.** Adding columns `x, x², x³` gives a curve and is still linear regression.
- **Invertibility.** More columns than rows (`d > N`) or duplicated columns make `XᵀX` singular; ridge fixes this.
- **A high R² is not proof of a good model**: it always rises when you add columns. Lesson 3 explains how to detect overfitting.
- **Outliers**: squared error is dominated by the largest errors. One wild point can drag the whole line.

## Key takeaways

- The best weights satisfy `XᵀX w = Xᵀy` because the residual is orthogonal to the columns of `X`.
- The loss is a convex bowl, so the stationary point is the global minimum.
- Ridge adds `λI`: always solvable, more stable, slightly biased.
- Solve a linear system; do not invert a matrix.

## Explain it yourself

1. Why does "the residual is orthogonal to every column of X" imply the normal equations?
2. What does ridge do to the eigenvalues of `XᵀX`, and why does that make the solution more stable?

## Further reading

- *Mathematics for Machine Learning*, Ch. 9 (PDF p.295) and Sec. 3.8 on orthogonal projection.
- Bishop, *PRML*, Sec. 3.1.
- Source library: `ai-engineering-from-scratch` phase 02, lesson 02.

## Mastery criteria

You can derive the normal equations and the ridge solution from the loss, state the geometric meaning, and implement both with the correct shapes.
