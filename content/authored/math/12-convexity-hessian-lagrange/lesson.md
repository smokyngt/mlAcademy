---
id: math.12
title: Convexity, the Hessian and Lagrange multipliers
summary: Use second derivatives to read the shape of a loss surface, see why convex problems have no bad local minima, take Newton steps, and handle constraints with Lagrange multipliers and a first glimpse of duality.
minutes: 90
teaches: [opt.convex]
requires: [calc.gradient, opt.gradient-descent, la.eigen]
sources:
  - "math/math for ML.pdf, Sec. 5.7 Higher-Order Derivatives (PDF p.170-171): second partial derivatives, symmetry of mixed partials, the Hessian as curvature. READ."
  - "math/math for ML.pdf, Sec. 5.8 Linearization and Multivariate Taylor Series (PDF p.171-172, start only): first-order Taylor expansion and the definition of the multivariate Taylor series. READ (p.171-172 only)."
  - "math/math for ML.pdf, Sec. 7.2 Constrained Optimization and Lagrange Multipliers (PDF p.239-242): constrained problem, indicator function, Lagrangian, primal and dual problems, minimax inequality, weak duality, equality constraints. READ."
  - "math/math for ML.pdf, Sec. 7.3 Convex Optimization (PDF p.242-245): convex sets and functions, first- and second-order conditions, negative entropy example, closure under non-negative sums, Jensen, linear program and its dual. READ (p.242-245 only)."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Appendix E Lagrange Multipliers (PDF p.727-729): gradient of g is normal to the surface, grad f parallel to grad g, the Lagrangian, worked example, inequality constraints. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/18-convex-optimization/docs/en.md (first 12000 characters): convex sets/functions, tests, Hessian eigenvalues, Newton's method, Lagrange example, KKT conditions. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/08-optimization/docs/en.md (Convex vs non-convex section and headings): non-convex landscapes and saddle points. READ."
---

## Where this fits

Gradient descent (lesson 7) only looks at the **slope**: which way is downhill. It does not know whether the valley ahead is wide or narrow, whether you stand at the bottom of a bowl or on a mountain pass. The information about the *shape* sits in the second derivatives, collected in the **Hessian** matrix. With it you can (1) tell a minimum from a maximum and from a saddle point, (2) choose smarter steps than "a fixed fraction of the slope" (Newton's method), and (3) explain a deep dichotomy of machine learning: linear and logistic regression, SVMs, ridge and lasso are **convex**, so there are no bad local minima; neural networks are not.

The second half deals with **constraints**: minimise a loss *while* keeping weights on a budget, probabilities summing to one, or margins non-negative. The tool is the Lagrange multiplier, which also opens the door to duality (the idea behind support vector machines). You have already met a Lagrange multiplier in the information-theory lesson (maximum entropy and the frequencies $\mu_k=m_k/N$); here is the method itself.

## The idea in plain words

- **Curvature.** Walk across a landscape. The slope tells you the steepness; the *curvature* tells you how fast the steepness changes. On a bowl, the ground tilts up in every direction. On the top of a hill it tilts down in every direction. On a **mountain pass** (a saddle) it tilts up along the path and down across it. At a flat spot, the curvature in each direction decides which of the three you are on.
- **The Hessian is the table of curvatures.** For a function of $n$ variables it is an $n\times n$ symmetric table. Its eigenvectors are the principal directions of the landscape and its eigenvalues the curvatures along them (lesson 7): all positive means a bowl, all negative a hilltop, mixed signs a saddle.
- **Convex = a single bowl.** A function is convex if the straight line between any two points of its graph never dips below the graph. Then there is no way to build a second valley: a place that looks like a bottom locally *is* the global bottom. That is why convex problems are "solved problems" while neural-network losses are not.
- **Newton's method = "fit a bowl and jump to its bottom".** Instead of taking a small step down the slope, approximate the landscape around you by the best parabola (using slope and curvature) and jump straight to that parabola's minimum. For a true bowl it arrives in one jump. The price is the cost of the curvature table.
- **Constraints = you must stay on a path.** Hike along a fixed trail: the best spot on the trail is where the trail runs *tangent* to a contour line of the height map. If the trail crossed the contour, you could slide along the trail and go lower. At the optimum the downhill direction is perpendicular to the trail. The **Lagrange multiplier** is the number that says how strongly the constraint is pushing back.

## Definitions and derivations

### Hessian and second-order Taylor expansion

For $f:\mathbb R^n\to\mathbb R$ twice continuously differentiable, the **Hessian** is

$$
H(\mathbf x)=\nabla^2f(\mathbf x),\qquad H_{ij}=\frac{\partial^2f}{\partial x_i\,\partial x_j}\in\mathbb R^{n\times n}.
$$

Because mixed partials commute ($\partial^2f/\partial x\partial y=\partial^2f/\partial y\partial x$, the book's (5.146)), $H$ is **symmetric**, so it has real eigenvalues and orthogonal eigenvectors (lesson 7). The Hessian is the derivative of the gradient: each row is the gradient of one component of $\nabla f$.

The first-order Taylor expansion (book (5.148)) is $f(\mathbf x)\approx f(\mathbf x_0)+\nabla f(\mathbf x_0)^\top(\mathbf x-\mathbf x_0)$: a tilted plane. Keeping the next term (the second-order term of the Taylor series, whose coefficient is the Hessian; the book defines the general series in Def. 5.7) gives a **quadratic**:

$$
f(\mathbf x_0+\boldsymbol\delta)\ \approx\ f(\mathbf x_0)+\nabla f(\mathbf x_0)^\top\boldsymbol\delta+\tfrac12\,\boldsymbol\delta^\top H(\mathbf x_0)\,\boldsymbol\delta .
$$

Shapes: $\boldsymbol\delta:(n,)$, $\nabla f:(n,)$, $H:(n,n)$; every term is a scalar. In one dimension this is $f(x_0)+f'(x_0)\delta+\tfrac12f''(x_0)\delta^2$.

### Classifying stationary points

At a stationary point $\nabla f(\mathbf x^\star)=\mathbf 0$ the linear term vanishes, so the local shape is decided by $\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$. Write $\boldsymbol\delta=\sum_ic_i\mathbf u_i$ in the orthonormal eigenvectors of $H$; then $\boldsymbol\delta^\top H\boldsymbol\delta=\sum_i\lambda_ic_i^2$. Hence:

| Eigenvalues of $H(\mathbf x^\star)$ | Type |
|---|---|
| all $>0$ (positive definite) | local minimum |
| all $<0$ | local maximum |
| both signs | saddle point |
| some $=0$, others of one sign | inconclusive (flat direction, look at higher order terms) |

A saddle is the typical obstacle in high-dimensional training: a point with zero gradient that is not a minimum. (The repository's optimisation lesson notes that in high-dimensional networks saddles are a bigger practical obstacle than poor local minima; treat this as a rule of thumb, not a theorem.)

### Convex sets and convex functions

A set $C$ is **convex** if for all $\mathbf x,\mathbf y\in C$ and $\theta\in[0,1]$: $\theta\mathbf x+(1-\theta)\mathbf y\in C$ (the whole segment stays inside). A function $f$ on a convex domain is **convex** if

$$
f(\theta\mathbf x+(1-\theta)\mathbf y)\ \le\ \theta f(\mathbf x)+(1-\theta)f(\mathbf y)\qquad\forall\ \theta\in[0,1] \quad(\text{the chord lies above the graph}).
$$

Equivalent tests for differentiable $f$ (Deisenroth Sec. 7.3):

- **First-order:** $f(\mathbf y)\ge f(\mathbf x)+\nabla f(\mathbf x)^\top(\mathbf y-\mathbf x)$ for all $\mathbf x,\mathbf y$: the graph lies above each of its tangent planes.
- **Second-order:** $H(\mathbf x)$ is positive semi-definite for **every** $\mathbf x$ in the domain. In 1-D, $f''\ge0$ everywhere.

Closure: a non-negative weighted sum $\alpha f_1+\beta f_2$ ($\alpha,\beta\ge0$) of convex functions is convex (add the two defining inequalities, as in the book's Example 7.4). That is why a convex loss plus a convex regulariser (ridge $\lambda\|\mathbf w\|^2$, lasso $\lambda\|\mathbf w\|_1$) is convex. Examples: $x^2$, $e^x$, $|x|$, $-\ln x$, $\max(0,x)$, linear functions, $x\log x$ for $x>0$ (the book's check at $x=2,4$: midpoint value $3\log_23\approx4.755\le$ chord value $5$), and the quadratic $\tfrac12\mathbf x^\top Q\mathbf x-\mathbf c^\top\mathbf x$ with $Q$ positive semi-definite, whose Hessian is the constant $Q$.

### Why convex problems have no bad local minima

**Claim.** If $f$ is convex, every local minimum is a global minimum.

*Proof.* Suppose $\mathbf x$ is a local minimum but some $\mathbf y$ has $f(\mathbf y)<f(\mathbf x)$. For $\theta\in(0,1]$ small, the point $\mathbf z_\theta=\theta\mathbf y+(1-\theta)\mathbf x$ is as close to $\mathbf x$ as we like, and by convexity
$$
f(\mathbf z_\theta)\le\theta f(\mathbf y)+(1-\theta)f(\mathbf x)=f(\mathbf x)-\theta\big(f(\mathbf x)-f(\mathbf y)\big)<f(\mathbf x).
$$
So there are points arbitrarily close to $\mathbf x$ with strictly lower value, contradicting local minimality. $\square$

Equivalently with the first-order condition: if $\nabla f(\mathbf x^\star)=\mathbf 0$ then $f(\mathbf y)\ge f(\mathbf x^\star)+0$ for every $\mathbf y$, so a stationary point of a convex function is automatically a global minimum. Gradient descent with a sensible step size therefore cannot be trapped. By contrast $f(x)=x^4-2x^2$ has two minima at $x=\pm1$ (value $-1$) and a local maximum between them; the convexity inequality fails: the chord between $x=-1$ and $x=1$ sits at height $\frac12(f(1)+f(-1))=-1$, below the graph at the midpoint, where $f(0)=0$. Gradient descent started at $+0.5$ ends at $+1$, started at $-0.5$ ends at $-1$.

### Newton's method

Minimise the second-order model $m(\boldsymbol\delta)=f+\mathbf g^\top\boldsymbol\delta+\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$ ($\mathbf g=\nabla f(\mathbf x_k)$, $H=H(\mathbf x_k)$). Its gradient with respect to $\boldsymbol\delta$ is $\mathbf g+H\boldsymbol\delta$ (using the gradient rules for linear and quadratic forms; $H$ symmetric). Setting it to zero:

$$
H\,\boldsymbol\delta=-\mathbf g\quad\Longrightarrow\quad\mathbf x_{k+1}=\mathbf x_k-H(\mathbf x_k)^{-1}\nabla f(\mathbf x_k).
$$

Compare with gradient descent $\mathbf x_{k+1}=\mathbf x_k-\eta\nabla f(\mathbf x_k)$: Newton replaces the scalar learning rate by the inverse Hessian, which takes small steps along high-curvature directions and large steps along flat ones. In practice you do not invert: solve the linear system $H\boldsymbol\delta=-\mathbf g$ (lesson 11). For a quadratic $f$ one step lands exactly on the minimum. Near a minimum of a general smooth function the error roughly **squares** at each step (quadratic convergence). The cost: $O(n^2)$ memory for $H$ and $O(n^3)$ to solve, impossible for networks with millions of weights; and where $H$ is not positive definite the Newton step may point uphill (toward a saddle or maximum), so practical variants regularise $H$.

### Constrained optimisation and Lagrange multipliers

**Problem (equality constraint).** Minimise $f(\mathbf x)$ subject to $g(\mathbf x)=0$.

**Geometry** (Bishop Appendix E). At any point of the constraint surface, $\nabla g$ is perpendicular to the surface: if $\mathbf x$ and $\mathbf x+\boldsymbol\epsilon$ both lie on it then $g(\mathbf x+\boldsymbol\epsilon)\approx g(\mathbf x)+\boldsymbol\epsilon^\top\nabla g$ and $g$ is the same at both points, so $\boldsymbol\epsilon^\top\nabla g\approx0$. At an optimum on the surface, $\nabla f$ must also be perpendicular to the surface (otherwise sliding along the surface would change $f$ to first order). Two vectors perpendicular to the same surface are parallel, so there is a number $\lambda$ with

$$
\nabla f+\lambda\nabla g=\mathbf 0 \qquad(\lambda\text{ may have either sign}).
$$

Package this as the **Lagrangian** $\mathcal L(\mathbf x,\lambda)=f(\mathbf x)+\lambda g(\mathbf x)$. Setting $\nabla_{\mathbf x}\mathcal L=\mathbf 0$ gives the condition above, and setting $\partial\mathcal L/\partial\lambda=0$ returns the constraint $g=0$. So **constrained optimisation becomes finding a stationary point of $\mathcal L$**: $n+1$ equations in $n+1$ unknowns. You may solve for $\mathbf x$ without ever needing $\lambda$ (hence "undetermined multiplier").

**Worked example.** Minimise $f(x,y)=x^2+y^2$ subject to $x+y-1=0$.

$$
\mathcal L=x^2+y^2+\lambda(x+y-1),\qquad \frac{\partial\mathcal L}{\partial x}=2x+\lambda=0,\quad \frac{\partial\mathcal L}{\partial y}=2y+\lambda=0,\quad \frac{\partial\mathcal L}{\partial\lambda}=x+y-1=0 .
$$

The first two give $x=y=-\lambda/2$; the third gives $-\lambda=1$, so $\lambda=-1$ and $x=y=\tfrac12$ with $f=\tfrac12$: the point of the line closest to the origin. Check with elimination: substitute $y=1-x$: $x^2+(1-x)^2$ has derivative $4x-2=0$, $x=\tfrac12$. The same method applied to Bishop's example $1-x_1^2-x_2^2$ (maximise) with $x_1+x_2-1=0$ gives $(\tfrac12,\tfrac12)$ and $\lambda=1$; the sign of $\lambda$ depends on whether you minimise or maximise and on how you write $g$.

**Regularisation as a constraint.** "Minimise the loss subject to $\|\mathbf w\|^2\le t$" and "minimise loss $+\lambda\|\mathbf w\|^2$" are two views of the same problem: $\lambda$ is the multiplier of the budget constraint (repository lesson 18).

### Inequality constraints and a first look at duality

For $\min f(\mathbf x)$ subject to $g_i(\mathbf x)\le0$, the book associates a multiplier $\lambda_i\ge0$ with each constraint:

$$
\mathcal L(\mathbf x,\boldsymbol\lambda)=f(\mathbf x)+\boldsymbol\lambda^\top\mathbf g(\mathbf x),\qquad D(\boldsymbol\lambda)=\min_{\mathbf x}\mathcal L(\mathbf x,\boldsymbol\lambda).
$$

Why: the hard constraint can be written as an infinite penalty (an indicator function). For $\lambda\ge0$ the linear penalty $\lambda g$ is never larger than that infinite penalty (where $g\le0$ it is $\le0$; where $g>0$ the indicator is $+\infty$), and $\max_{\boldsymbol\lambda\ge0}\mathcal L$ equals $f(\mathbf x)$ when $\mathbf x$ is feasible and $+\infty$ otherwise, which is exactly the penalised objective. The **primal** problem is $\min_{\mathbf x}\max_{\boldsymbol\lambda\ge0}\mathcal L$, the **dual** problem is $\max_{\boldsymbol\lambda\ge0}D(\boldsymbol\lambda)=\max_{\boldsymbol\lambda\ge0}\min_{\mathbf x}\mathcal L$. Swapping min and max can only lower the value (the minimax inequality, book (7.23)), which gives **weak duality**: every dual value $D(\boldsymbol\lambda)$ is a lower bound on the primal optimum. $D$ is always concave, even when $f$ is not convex (it is a pointwise minimum of functions that are affine in $\boldsymbol\lambda$). For convex problems (convex $f$, $g_i$) the book states that **strong duality** holds: the two optimal values coincide.

*Example.* Minimise $x^2$ subject to $1-x\le0$ (that is $x\ge1$). Then $\mathcal L=x^2+\lambda(1-x)$; minimising over $x$ gives $x=\lambda/2$ and $D(\lambda)=\lambda-\lambda^2/4$. Maximising over $\lambda\ge0$: $D'=1-\lambda/2=0$, so $\lambda=2$ and $D=1$. The primal optimum is $x=1$ with value $1$: equal. For other $\lambda$ the dual value is lower: $D(1)=0.75$, $D(3)=0.75$, $D(0)=0$. At the solution the constraint is *active* ($1-x=0$) and $\lambda>0$; had the unconstrained minimum $x=0$ already satisfied the constraint, the multiplier would be $0$ (**complementary slackness**: $\lambda_ig_i(\mathbf x^\star)=0$, as in Bishop's discussion of active and inactive constraints and in the KKT conditions of the repository lesson). Support vector machines are exactly this construction, with $\lambda_i>0$ for the support vectors.

## Worked example

All numbers are reproduced by the exercise code.

**1. Classify stationary points.** $f(x,y)=x^3-3x+y^2$. Gradient $(3x^2-3,\ 2y)$ vanishes at $(1,0)$ and $(-1,0)$. Hessian $\operatorname{diag}(6x,\,2)$. At $(1,0)$: $\operatorname{diag}(6,2)$, positive definite, a local minimum with $f=-2$. At $(-1,0)$: $\operatorname{diag}(-6,2)$, mixed signs, a saddle with $f=2$.

**2. A saddle from cross terms.** $f(x,y)=x^2+3xy+y^2$ has $H=\begin{psmallmatrix}2&3\\3&2\end{psmallmatrix}$ with eigenvalues $5$ and $-1$: a saddle at the origin, although both diagonal entries are positive. Along the eigenvector $(1,-1)$: $f(t,-t)=t^2-3t^2+t^2=-t^2<0$.

**3. Newton versus gradient descent on a quadratic.** $f(\mathbf x)=\tfrac12\mathbf x^\top Q\mathbf x-\mathbf c^\top\mathbf x$ with $Q$ having rows $(4,1)$ and $(1,3)$, $\mathbf c=(1,2)$ (eigenvalues $2.38$ and $4.62$, so $f$ is convex). The minimiser solves $Q\mathbf x=\mathbf c$: $\mathbf x^\star=(1/11,\,7/11)=(0.0909,\,0.6364)$. From $(5,5)$ one Newton step gives exactly $(0.0909,0.6364)$. Gradient descent with $\eta=0.2$ has errors $0.771$ after 1 step, $0.0445$ after 5, $0.00175$ after 10, and $4\times10^{-9}$ after 30.

**4. Quadratic convergence.** Minimise $f(x)=e^x-2x$ (convex, minimum at $\ln2=0.693147$). Newton: $x_{k+1}=x_k-(e^{x_k}-2)/e^{x_k}$ from $x_0=0$ gives errors $0.307,\ 0.0426,\ 8.95\times10^{-4},\ 4.0\times10^{-7},\ 8.0\times10^{-14}$: the number of correct digits roughly doubles each step. Gradient descent with $\eta=0.3$ has errors $0.393,\ 0.198,\ 0.090,\ 0.038,\ 0.016,\ 0.0064$: a constant factor per step.

**5. Non-convex example.** $h(x)=x^4-2x^2$: $h(1)=h(-1)=-1$, $h(0)=0>\frac12(h(1)+h(-1))$, so it is not convex. Gradient descent with step $0.05$ from $\pm0.5$ ends at $\pm1$.

**6. Lagrange.** The example above: $(0.5,0.5)$, $\lambda=-1$, $f=0.5$. Dual example: $D(\lambda)=\lambda-\lambda^2/4$, maximum $1$ at $\lambda=2$.

## Implementation

```python
import numpy as np

def grad(x):  return np.array([3*x[0]**2 - 3, 2*x[1]])      # f = x^3 - 3x + y^2
def hess(x):  return np.array([[6*x[0], 0.], [0., 2.]])

x_star = np.array([1., 0.])
eig = np.linalg.eigvalsh(hess(x_star))      # ascending, real because H is symmetric
kind = "min" if (eig > 0).all() else "max" if (eig < 0).all() else "saddle"

# Newton step for a quadratic: solve, never invert
Q = np.array([[4., 1], [1, 3]]); c = np.array([1., 2]); x = np.array([5., 5])
x = x - np.linalg.solve(Q, Q @ x - c)       # lands on Q^{-1} c = [1/11, 7/11]
```

## Shapes

$\mathbf x,\nabla f:(n,)$; Hessian $(n,n)$ symmetric, `np.linalg.eigvalsh` returns `(n,)`. Newton step: solve $H\boldsymbol\delta=-\mathbf g$ with `np.linalg.solve(H, -g)`, result `(n,)`. For $m$ constraints $\mathbf g(\mathbf x)\in\mathbb R^m$ and multipliers $\boldsymbol\lambda\in\mathbb R^m$. Computing the Hessian of a network with $n=10^6$ weights would need $10^{12}$ numbers; that is why Newton's method is not used there.

## Common misconceptions

- **Positive diagonal entries of $H$ mean a minimum.** The eigenvalues decide, not the diagonal ($x^2+3xy+y^2$ is a saddle).
- **A zero gradient means a minimum.** It may be a maximum or saddle; check the Hessian.
- **Convex means "has a unique minimum".** Convex guarantees every local minimum is global; the minimiser may be a whole flat set (e.g. $f=0$). Strict convexity gives uniqueness.
- **Newton's method is always better.** It needs a positive-definite Hessian and costs $O(n^3)$; far from the minimum it can diverge.
- **Neural networks are convex in their weights.** Hidden layers with non-linear activations break convexity; SGD works well anyway for reasons that are partly empirical.
- **The Lagrange multiplier is just a trick.** It has meaning: the sensitivity of the optimal value to relaxing the constraint (not proved here), and in SVMs it flags the support vectors.

## Key takeaways

- $f(\mathbf x_0+\boldsymbol\delta)\approx f+\mathbf g^\top\boldsymbol\delta+\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$; at a stationary point the eigenvalues of $H$ say minimum, maximum or saddle.
- Convex: chord above graph, $H\succeq0$ everywhere, and every local minimum is global.
- Newton: $\mathbf x\leftarrow\mathbf x-H^{-1}\mathbf g$ (solve, do not invert); exact on quadratics, quadratically convergent near a minimum, too costly for huge models.
- Constraints: $\nabla f=-\lambda\nabla g$; find stationary points of $\mathcal L=f+\lambda g$. Inequalities need $\lambda\ge0$ and complementary slackness; the dual gives lower bounds, equal for convex problems.

## Explain it yourself

1. Why is a stationary point of a convex function always a global minimum? Use the picture of the chord.
2. Draw (or describe) why, at the optimum of a problem constrained to a curve, the gradient of the objective must be perpendicular to the curve.

## Further reading

- *Mathematics for Machine Learning*, Sec. 7.1 (gradient descent, PDF p.233, not part of this lesson's sources), 7.3.2 quadratic programming (PDF p.246) and Sec. 7.3.3 Legendre-Fenchel duality, which continue where this lesson stops.
- Boyd and Vandenberghe, *Convex Optimization* (cited by the book for the definitions used here).

## Mastery criteria

You can compute a Hessian and classify stationary points by its eigenvalues, check convexity by the definition or the Hessian, explain why convexity rules out bad local minima, perform a Newton step, and solve a small equality-constrained problem with a Lagrangian and interpret the multiplier and the dual value.
