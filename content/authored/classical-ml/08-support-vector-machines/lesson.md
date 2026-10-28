---
id: cml.08
title: Support vector machines
summary: Pick the separating hyperplane with the widest margin, derive the soft-margin primal and its Lagrangian dual, see the hinge-loss view, and add kernels for curved boundaries.
minutes: 85
teaches: [ml.svm]
requires: [ml.logreg, la.dot, opt.gradient-descent, ml.loss]
sources:
  - "math/math for ML.pdf (Deisenroth et al.), Ch. 12 Classification with Support Vector Machines, Sec. 12.1-12.4 (PDF p.378-396): separating hyperplanes, margin, hard and soft margin, hinge loss view, Lagrangian dual, kernels. READ (read selectively; 12.3.2 convex-hull view read, only mentioned)."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 7.1 Maximum Margin Classifiers (PDF p.346-348 and p.352-354): margin, canonical representation, slack variables, dual with box constraints, which points are support vectors. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/05-support-vector-machines/docs/en.md (hinge loss, subgradient training of a linear SVM, when SVMs still win). READ (hinge, gradient descent and 'why SVMs lost' sections)."
---

## Where this fits

Logistic regression ([[cml.02]]) gives you *a* separating line and a probability. But when two classes are separable there are infinitely many separating lines, and some are fragile: a tiny shift in a training point would flip a test prediction. The support vector machine (SVM) picks the one *farthest* from both classes. It dominated machine learning on medium-sized problems (text classification with sparse word features, bioinformatics, handwritten digits) before deep learning, and it is still a strong choice for small or high-dimensional data. It is also where three ideas meet that you will reuse: margins as regularisation, Lagrangian duality (see the constrained-optimisation lesson `math.12`), and the kernel trick, which reappears in Gaussian processes and, in spirit, in attention.

## The idea in plain words

Picture two groups of houses on opposite sides of a river and you must build a road between them. Any road that keeps all houses of one group on one side works, but you would rather build it as far from every house as possible, so that a house on the edge being moved a bit still leaves it on the right side. The empty strip between the road and the nearest houses is the **margin**. Only the houses touching the strip decide where the road goes: delete any other house and the road stays; those boundary houses are the **support vectors**.

Real data is messy: a few houses may sit in the strip or on the wrong side. The **soft margin** allows that but charges a penalty per violation, and a dial $C$ says how expensive a violation is. A big $C$ insists on classifying training points correctly (narrow margin, risk of overfitting); a small $C$ tolerates mistakes for a wider, simpler road.

When no straight road works (one class encircles the other), you can lift the points into a higher-dimensional space where a flat sheet does separate them. The **kernel trick** performs that lift implicitly: you never compute the new coordinates, only similarities between pairs of points.

## Setup and the margin

Features $x\in\mathbb R^D$, labels $y_n\in\{-1,+1\}$, score $f(x)=\langle w,x\rangle+b$, prediction $\operatorname{sign}f(x)$. The hyperplane is $\{x:f(x)=0\}$ and $w$ is normal to it (take two points $x_a,x_b$ on it: $f(x_a)-f(x_b)=\langle w,x_a-x_b\rangle=0$).

**Distance to the hyperplane.** Write $x_a=x_a'+r\,\frac{w}{\|w\|}$ with $x_a'$ the orthogonal projection of $x_a$ onto the hyperplane, so $f(x_a')=0$. Then

$$
f(x_a)=\langle w,x_a'\rangle+b+r\frac{\langle w,w\rangle}{\|w\|}=0+r\|w\|\quad\Longrightarrow\quad r=\frac{f(x_a)}{\|w\|}.
$$

For a correctly classified point the distance is $y_nf(x_n)/\|w\|$ (Bishop Eq. 7.2). The margin is the smallest such distance.

**Fix the scale.** Rescaling $(w,b)\to(\kappa w,\kappa b)$ leaves every distance unchanged, so we may choose the scale such that the closest point satisfies $y_nf(x_n)=1$. This is the *canonical* representation. All points then satisfy $y_nf(x_n)\ge1$ and the margin is $r=1/\|w\|$. Maximising $1/\|w\|$ is minimising $\tfrac12\|w\|^2$ (monotone transformation, and the $\tfrac12$ tidies the gradient).

## The primal problems

**Hard margin** (separable data):

$$
\min_{w,b}\ \tfrac12\|w\|^2\quad\text{s.t.}\quad y_n(\langle w,x_n\rangle+b)\ge1,\ n=1..N.
$$

**Soft margin.** Give each point a slack $\xi_n\ge0$ that lets it violate the margin ($0<\xi\le1$: inside the margin but correct; $\xi>1$: misclassified):

$$
\min_{w,b,\xi}\ \tfrac12\|w\|^2+C\sum_n\xi_n\quad\text{s.t.}\quad y_n(\langle w,x_n\rangle+b)\ge1-\xi_n,\ \ \xi_n\ge0 .
$$

$C\to\infty$ recovers the hard margin. Note $\sum_n\xi_n$ upper-bounds the number of training errors (Bishop p.332). $b$ is not regularised.

## The hinge-loss view

For a fixed $(w,b)$ the best slack is the smallest allowed: $\xi_n=\max\{0,\,1-y_nf(x_n)\}$. Substituting gives an unconstrained problem

$$
\min_{w,b}\ \underbrace{\tfrac12\|w\|^2}_{\text{regulariser}}+C\sum_n\underbrace{\max\{0,\,1-y_nf(x_n)\}}_{\text{hinge loss}} .
$$

With $t=yf(x)$ the hinge loss is $0$ when $t\ge1$ (correct and outside the margin), and $1-t$ otherwise, rising linearly even for correct points inside the margin. It is a convex upper bound on the 0-1 loss, which is itself combinatorial to optimise (Deisenroth Sec. 12.2.5). So an SVM is "ERM with hinge loss + $\ell_2$ regularisation"; the margin is the regulariser. Compare logistic loss $\ln(1+e^{-t})$: smooth and never exactly zero, whereas hinge is exactly zero beyond the margin, which is why only some points matter.

## Derivation: the Lagrangian dual

(Constrained optimisation background: `math.12`; here is what we need.) Introduce multipliers $\alpha_n\ge0$ for the margin constraints and $\gamma_n\ge0$ for $\xi_n\ge0$:

$$
\mathcal L=\tfrac12\|w\|^2+C\sum_n\xi_n-\sum_n\alpha_n\big(y_n(\langle w,x_n\rangle+b)-1+\xi_n\big)-\sum_n\gamma_n\xi_n .
$$

Set the derivatives with respect to the primal variables to zero:

$$
\frac{\partial\mathcal L}{\partial w}=0\Rightarrow w=\sum_n\alpha_ny_nx_n,\qquad
\frac{\partial\mathcal L}{\partial b}=0\Rightarrow\sum_n\alpha_ny_n=0,\qquad
\frac{\partial\mathcal L}{\partial \xi_n}=0\Rightarrow C-\alpha_n-\gamma_n=0 .
$$

The last relation with $\gamma_n\ge0$ gives $\alpha_n\le C$. Substitute back: the $b$ term vanishes by the second condition, the $\xi$ terms vanish by the third, and the $w$ terms combine into $\tfrac12\|w\|^2-\|w\|^2=-\tfrac12\|w\|^2$ where $\|w\|^2=\sum_{i,j}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle$. The result is the **dual**:

$$
\min_{\alpha}\ \tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle-\sum_i\alpha_i\quad\text{s.t.}\quad\sum_i\alpha_iy_i=0,\ \ 0\le\alpha_i\le C .
$$

Consequences:

- **Representer form.** $w=\sum_n\alpha_ny_nx_n$: the solution is a combination of training points. Complementary slackness ($\alpha_n(y_nf(x_n)-1+\xi_n)=0$) says $\alpha_n>0$ only for points on or inside the margin or misclassified: the **support vectors**. Points with $0<\alpha_n<C$ lie exactly on the margin ($\xi_n=0$), those with $\alpha_n=C$ may violate it (Bishop p.334).
- **Recover $b$** from a margin point: $b=y_n-\langle w,x_n\rangle$ (average over such points for stability).
- **Only inner products appear**, $\langle x_i,x_j\rangle$. The dual has $N$ variables instead of $D$: attractive when $D\gg N$, and it is the door to kernels.

## Kernels

Replace $x$ by a feature map $\phi(x)$. The dual only needs $k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$, so we may *define* a similarity $k$ directly. A valid kernel gives a symmetric positive semi-definite Gram matrix $K_{ij}=k(x_i,x_j)$ for any data set. Prediction becomes

$$
f(x)=\sum_n\alpha_ny_n\,k(x_n,x)+b .
$$

Examples. The polynomial kernel $k(x,z)=(\langle x,z\rangle+1)^2$ in 2D equals $\langle\phi(x),\phi(z)\rangle$ with $\phi(x)=(1,\sqrt2x_1,\sqrt2x_2,x_1^2,\sqrt2x_1x_2,x_2^2)$ (expand the square to check), and costs one dot product instead of six features. The **RBF (Gaussian) kernel**

$$
k(x,z)=\exp(-\gamma\|x-z\|^2)
$$

corresponds to an infinite-dimensional feature space. It is 1 for identical points and decays towards 0 with distance; $\gamma$ sets how local it is: large $\gamma$ gives wiggly boundaries that can overfit, small $\gamma$ approaches a linear classifier. The learned model is still a *linear* classifier in feature space; the boundary is curved only in the original space. $C$ and $\gamma$ are normally tuned by cross-validation.

## Training a linear SVM by subgradient descent

The hinge loss has a kink at $t=1$, so use a **subgradient**. Divide the objective by $CN$ and set $\lambda=1/(CN)$:

$$
J(w,b)=\frac\lambda2\|w\|^2+\frac1N\sum_n\max\{0,\,1-y_n(\langle w,x_n\rangle+b)\}.
$$

Let $\mathcal V=\{n:y_nf(x_n)<1\}$ be the margin violators. A valid subgradient is

$$
g_w=\lambda w-\frac1N\sum_{n\in\mathcal V}y_nx_n,\qquad g_b=-\frac1N\sum_{n\in\mathcal V}y_n .
$$

Non-violators contribute nothing: this is the same sparsity as the dual. Update $w\leftarrow w-\eta_tg_w$ with a decaying step $\eta_t$, because with a non-smooth objective a fixed step bounces around the minimum. The loss is not monotone, so judge by the objective trend rather than each step.

## Worked example

Take $x_1=(2,0),\,y_1=+1$ and $x_2=(-2,0),\,y_2=-1$ (plus two far points $(3,1)^+,(-3,-1)^-$).

- By symmetry the best boundary is the vertical line $x_1=0$: $w=(w_1,0),\,b=0$. Constraints $y f\ge1$ at the closest points: $2w_1\ge1$, so $w_1=\tfrac12$. Margin $=1/\|w\|=2$ (indeed the distance from $(2,0)$ to the line $x_1=0$). The far points have $yf=1.5>1$: not support vectors.
- Dual check: $w=\sum\alpha_ny_nx_n=\alpha\cdot(2,0)+\alpha\cdot(2,0)=(4\alpha,0)$, so $\alpha_1=\alpha_2=\tfrac18$, and $\sum\alpha_ny_n=0$ holds. The dual objective is $\sum\alpha-\tfrac12\sum_{ij}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle=\tfrac14-\tfrac12\cdot\tfrac{16}{64}=\tfrac18$, and the primal $\tfrac12\|w\|^2=\tfrac12\cdot\tfrac14=\tfrac18$: equal, as strong duality requires.
- Subgradient at $w=0,\,b=0$, with all four points, $\lambda=0.1$: every point violates ($yf=0<1$). $g_w=-\tfrac14\big[(2,0)+(2,0)+(3,1)+(3,1)\big]=(-2.5,-0.5)$ and $g_b=-\tfrac14(1-1+1-1)=0$.

The exercises reproduce these numbers.

## Implementation

```python
import numpy as np

def svm_subgradient(w, b, X, y, lam):
    viol = y * (X @ w + b) < 1                   # margin violators
    gw = lam * w - (y[viol, None] * X[viol]).sum(0) / len(y)
    gb = -y[viol].sum() / len(y)
    return gw, gb

def svm_fit(X, y, lam=0.01, lr=0.5, epochs=500):
    w, b = np.zeros(X.shape[1]), 0.0
    for t in range(epochs):
        gw, gb = svm_subgradient(w, b, X, y, lam)
        step = lr / (1 + 0.05 * t)              # decaying step size
        w, b = w - step * gw, b - step * gb
    return w, b
```

## Shapes

`X: (N, D)`, `y: (N,)` in $\{-1,+1\}$, `w: (D,)`, `b: scalar`, scores `X @ w + b: (N,)`. Kernel (Gram) matrix `K: (N, N)`; for test points `K_test: (M, N)` and predictions `K_test @ (alpha * y) + b: (M,)`. The Gram matrix is why kernel SVMs scale poorly beyond tens of thousands of points: memory $O(N^2)$, training between $O(N^2)$ and $O(N^3)$.

## Common misconceptions

- **"The SVM maximises the margin of all points."** Only the support vectors determine the solution; moving other points (without crossing the margin) changes nothing.
- **Forgetting to scale features.** The margin is measured in Euclidean distance, and the RBF kernel uses it too; a feature with large units dominates. Standardise first.
- **"A big $C$ is better."** A large $C$ fits training data tightly and can overfit noisy data; choose $C$ by validation.
- **"A kernel SVM works in the original space."** It is linear in an implicit feature space, so its boundary is curved in input space.
- **SVM outputs are not probabilities.** The score $f(x)$ is a signed distance-like quantity; probabilities need an extra calibration step.
- **"Dual is always better."** For large $N$ with moderate $D$, the primal (linear SVM by SGD) is far cheaper.

## Key takeaways

- Margin $=1/\|w\|$ in the canonical scaling; hard margin: $\min\tfrac12\|w\|^2$ s.t. $y_nf(x_n)\ge1$.
- Soft margin adds slacks and a penalty $C\sum\xi_n$; equivalently hinge loss + $\ell_2$ regulariser.
- The dual (from the Lagrangian) has $0\le\alpha_n\le C$, $\sum\alpha_ny_n=0$, only inner products, and $w=\sum\alpha_ny_nx_n$; non-zero $\alpha_n$ are the support vectors.
- Kernels replace inner products; RBF gives curved boundaries with $\gamma$ controlling locality.
- A linear SVM can be trained directly in the primal by subgradient descent on the hinge loss.

## Explain it yourself

1. Why can we assume that the closest point has $y_nf(x_n)=1$ without losing generality?
2. Starting from the Lagrangian, explain why the dual contains only inner products $\langle x_i,x_j\rangle$.

## Further reading

- Deisenroth, Faisal, Ong, *Mathematics for ML*, Ch. 12 (PDF p.376-398); Bishop, *PRML*, Sec. 7.1.
- Repo lesson `phases/02-ml-fundamentals/05-support-vector-machines`.
- Related: [[cml.02]] (logistic loss versus hinge loss).

## Mastery criteria

You can derive $r=1/\|w\|$, write the soft-margin primal and the hinge-loss form, derive the dual from the Lagrangian, identify support vectors from $\alpha_n$, evaluate an RBF kernel decision function, and train a linear SVM with a subgradient loop.
