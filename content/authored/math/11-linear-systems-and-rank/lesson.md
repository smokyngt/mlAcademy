---
id: math.11
title: Linear systems, Gaussian elimination and rank
summary: Solve Ax = b by elimination, read existence and uniqueness off the rank and null space, fall back on least squares when there is no exact answer, and understand why we call solve instead of inv.
minutes: 80
teaches: [la.linear-systems]
requires: [la.matrix, la.linear-map]
sources:
  - "math/math for ML.pdf, Sec. 2.3 Solving Systems of Linear Equations (PDF p.33-41): particular and general solution, elementary transformations, row-echelon form, pivots, basic and free variables, reduced row-echelon form, the minus-1 trick, inverse by Gaussian elimination, pseudo-inverse and iterative methods. READ."
  - "math/math for ML.pdf, Sec. 2.4 Vector Spaces (PDF p.41-45): subspaces; the solution set of Ax = 0 is a subspace, that of Ax = b (b != 0) is not. READ (skimmed groups/vector-space axioms)."
  - "math/math for ML.pdf, Sec. 2.5 Linear Independence (PDF p.46-47): linear combination, dependence, the Gaussian-elimination test. READ (p.46-47 only)."
  - "math/math for ML.pdf, Sec. 2.6.2 Rank (PDF p.53-54): rank properties, rk(A) = rk(A|b) for solvability, dim of null space = n - rk(A), full rank, worked rank example. READ."
  - "math/math for ML.pdf, Sec. 2.7.3 Image and Kernel (PDF p.64-65): kernel, image, column space, Ax = 0 and the kernel. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/17-linear-systems/docs/en.md (first 14000 characters): row and column picture, Gaussian elimination with partial pivoting, LU, QR, Cholesky, least squares and the normal equations, pseudoinverse, condition number. READ."
---

## Where this fits

A huge fraction of machine learning is "find the unknowns that satisfy these equations". Linear regression: find the weights $\mathbf w$ such that $X\mathbf w$ matches the targets. Fitting a Gaussian process, inverting a covariance for the Mahalanobis distance, computing the pseudo-inverse used by PCA and least squares, solving for the next step of Newton's method (lesson 12): each of these is "solve $A\mathbf x=\mathbf b$" in disguise.

Two questions come before any algorithm. *Does a solution exist? Is it unique?* The answer lives in two numbers attached to the matrix: its **rank** and the dimension of its **null space**. This lesson builds the picture, then the standard method (Gaussian elimination), then what to do when there is **no exact solution** (almost always, in data science: more equations than unknowns, so we use least squares), and finally the numerical warning that explains why numerical libraries give you `solve` and not a button for "invert, then multiply".

## The idea in plain words

- **Row picture.** Every equation is a flat sheet (a line in 2-D, a plane in 3-D). A solution is a point lying on all sheets at once. Two lines can cross at one point, be parallel and never meet, or lie on top of each other and share infinitely many points: exactly the three possibilities for any linear system (zero, one or infinitely many solutions; never exactly two).
- **Column picture.** Read the same system as a recipe: "which amounts $x_1,x_2,\dots$ of the column vectors, mixed together, give the vector $\mathbf b$?" The set of everything you can build is the **column space**. A solution exists exactly when $\mathbf b$ lies in it.
- **Gaussian elimination** is what you did at school with two equations: use one equation to cancel a variable from the others, until the system becomes a staircase that you solve from the bottom up. The staircase shows how many equations are really independent (the **rank**) and which unknowns are left free.
- **Free variables = redundancy.** If the columns contain a redundant one (a column you could build from the others), then you can add a bit of it and take the same amount away elsewhere: the solution is not unique, there is a whole line or plane of solutions. That set of "invisible" moves, the vectors that $A$ sends to zero, is the **null space**.
- **No exact solution?** If $\mathbf b$ is not in the column space, find the point in the column space that is *closest* to $\mathbf b$: least squares. It is the shadow of $\mathbf b$ on the column space, and the leftover error vector is perpendicular to that space.
- **Why not just invert?** Imagine measuring with a ruler whose markings are almost identical (two nearly parallel lines). A tiny wobble in the measurement moves the crossing point enormously. That sensitivity belongs to the problem (the **condition number**); a good algorithm cannot remove it but a careless one (like computing the full inverse first) can add its own errors and costs more work.

## Definitions

Let $A\in\mathbb R^{m\times n}$, $\mathbf x\in\mathbb R^n$, $\mathbf b\in\mathbb R^m$. The system $A\mathbf x=\mathbf b$ has $m$ equations and $n$ unknowns.

- **Column space** (image): $\operatorname{Im}(A)=\{A\mathbf x\}=\operatorname{span}(\mathbf a_1,\dots,\mathbf a_n)\subseteq\mathbb R^m$.
- **Null space** (kernel): $\ker(A)=\{\mathbf x:A\mathbf x=\mathbf 0\}\subseteq\mathbb R^n$, a subspace.
- **Rank** $\operatorname{rk}(A)$: the number of linearly independent columns, which equals the number of linearly independent rows, and equals $\dim\operatorname{Im}(A)$. It satisfies $\operatorname{rk}(A)\le\min(m,n)$; equality is **full rank**.
- **Rank-nullity** (Deisenroth Sec. 2.6.2): $\dim\ker(A)=n-\operatorname{rk}(A)$.

### What decides existence and uniqueness

| Condition | Meaning |
|---|---|
| Solution exists iff $\operatorname{rk}(A)=\operatorname{rk}(A\,|\,\mathbf b)$ | adding $\mathbf b$ as an extra column does not enlarge the column space |
| Solution unique iff exists and $\ker(A)=\{\mathbf 0\}$, i.e. $\operatorname{rk}(A)=n$ | no free variable |
| Infinitely many solutions iff exists and $\operatorname{rk}(A)<n$ | the general solution is a particular solution plus anything in $\ker(A)$ |
| Square $A$ is invertible iff $\operatorname{rk}(A)=n$ | |

**General solution = particular + null space.** If $A\mathbf x_p=\mathbf b$ and $A\mathbf n=\mathbf 0$, then $A(\mathbf x_p+\mathbf n)=\mathbf b+\mathbf 0=\mathbf b$. Conversely any two solutions differ by a vector of the null space. So the solution set of $A\mathbf x=\mathbf b$ is a shifted copy of $\ker(A)$; for $\mathbf b\ne\mathbf 0$ it is *not* a subspace (it does not contain $\mathbf 0$).

## Gaussian elimination

Three **elementary row operations** do not change the solution set: swap two rows, multiply a row by a non-zero constant, add a multiple of one row to another. Write the system as the augmented matrix $[A\,|\,\mathbf b]$ and apply them to reach **row-echelon form** (REF): zero rows at the bottom, and the first non-zero entry of each row (its **pivot**) strictly to the right of the pivot above. This staircase gives:

- **Rank** = number of pivots.
- **Basic variables** = unknowns in pivot columns; **free variables** = the others.
- A row of the form $[0\ \cdots\ 0\,|\,c]$ with $c\ne0$ means "$0=c$": no solution.

With every pivot scaled to 1 and zeros above the pivots as well, we get **reduced row-echelon form** (RREF), from which the null space can be read directly (the book's *minus-1 trick*: for each free column, put $-1$ on its diagonal position and read the columns).

### A first example (the book's, Sec. 2.3.1)

$$
\begin{pmatrix}1&0&8&-4\\0&1&2&12\end{pmatrix}\mathbf x=\begin{pmatrix}42\\8\end{pmatrix}.
$$

Two equations, four unknowns: expect infinitely many solutions. The pivot columns are 1 and 2. A particular solution takes $42$ of column 1 and $8$ of column 2: $\mathbf x_p=(42,8,0,0)$. Column 3 equals $8\cdot$col1 $+\,2\cdot$col2, so $(8,2,-1,0)$ is sent to zero; column 4 equals $-4\cdot$col1 $+\,12\cdot$col2, so $(-4,12,0,-1)$ is too. Therefore

$$
\mathbf x=\begin{pmatrix}42\\8\\0\\0\end{pmatrix}+\lambda_1\begin{pmatrix}8\\2\\-1\\0\end{pmatrix}+\lambda_2\begin{pmatrix}-4\\12\\0\\-1\end{pmatrix},\qquad\lambda_1,\lambda_2\in\mathbb R .
$$

Here $n=4$, rank $2$, nullity $2$: two free parameters, as promised.

### A fully numeric elimination (square, unique solution)

$$
\begin{pmatrix}2&1&1\\4&3&3\\2&3&1\end{pmatrix}\mathbf x=\begin{pmatrix}8\\20\\12\end{pmatrix}.
$$

Subtract $2\times$row 1 from row 2 and $1\times$row 1 from row 3:

$$
\left(\begin{array}{ccc|c}2&1&1&8\\0&1&1&4\\0&2&0&4\end{array}\right)\ \to\ \left(\begin{array}{ccc|c}2&1&1&8\\0&1&1&4\\0&0&-2&-4\end{array}\right)\quad(\text{row 3}-2\times\text{row 2}).
$$

Back substitution from the bottom: $-2x_3=-4\Rightarrow x_3=2$; $x_2+x_3=4\Rightarrow x_2=2$; $2x_1+x_2+x_3=8\Rightarrow x_1=2$. The multipliers $2,1,2$ are the entries of the lower factor in $A=LU$; the result is the upper triangular matrix $U$. Three pivots, rank 3, unique solution $(2,2,2)$.

### Pivoting and cost

Eliminating costs about $n^3$ operations (the book notes the cubic scaling is impractical for millions of unknowns, where iterative methods are used instead). If a pivot is zero you must swap rows; if it is merely *tiny* you should also swap: dividing by a tiny pivot creates huge multipliers that amplify rounding errors. **Partial pivoting** picks the largest entry in the column at each step. With a $2\times2$ system $\begin{psmallmatrix}10^{-20}&1\\1&1\end{psmallmatrix}\mathbf x=\begin{psmallmatrix}1\\2\end{psmallmatrix}$ (true solution $\approx(1,1)$) elimination without pivoting returns $(0,1)$ in float64, with pivoting $(1,1)$.

If you must solve many systems with the same $A$ and different $\mathbf b$, factor once as $PA=LU$ (cost $\sim n^3$) and then each right-hand side costs only $\sim n^2$ (forward and back substitution).

### Rank by elimination (the book's example, p.54)

$$
A=\begin{pmatrix}1&2&1\\-2&-3&1\\3&5&0\end{pmatrix}\ \rightsquigarrow\ \begin{pmatrix}1&2&1\\0&1&3\\0&0&0\end{pmatrix}:
$$

two pivots, rank 2: the third row is $\text{row}_1+\text{row}_2$ in the original. The matrix is square but *not* invertible (nullity $3-2=1$). In floating point we count singular values above a tolerance instead of exact zero pivots.

### Existence depends on $\mathbf b$ (the book's Example 2.6)

The system with coefficient matrix rows $(-2,4,-2,-1,4)$, $(4,-8,3,-3,1)$, $(1,-2,1,-1,1)$, $(1,-2,0,-3,4)$ and right-hand side $(-3,2,0,a)$ reduces to a staircase with three pivots and a last row $0=a+1$. So $\operatorname{rk}(A)=3$, and the system is solvable only for $a=-1$ (then $\operatorname{rk}(A|\mathbf b)=3$ too; for $a=0$ the augmented rank is $4$). For $a=-1$ a particular solution is $(2,0,-1,1,0)$ and the null space is spanned by $(2,1,0,0,0)$ and $(2,0,-1,2,1)$, so five unknowns, rank 3, two free parameters.

## When there is no exact solution: least squares

In data fitting $m\gg n$ (many data points, few parameters), and $\mathbf b\notin\operatorname{Im}(A)$ almost surely. We minimise the residual:

$$
\min_{\mathbf x}\ \|A\mathbf x-\mathbf b\|^2 .
$$

**Derivation.** Expand $\|A\mathbf x-\mathbf b\|^2=\mathbf x^\top A^\top A\mathbf x-2\mathbf b^\top A\mathbf x+\mathbf b^\top\mathbf b$. Its gradient with respect to $\mathbf x$ is $2A^\top A\mathbf x-2A^\top\mathbf b$ (using the gradient rules of lesson 5; $A^\top A$ is symmetric). Setting it to zero gives the **normal equations**

$$
A^\top A\,\mathbf x=A^\top\mathbf b .
$$

Geometrically, rewrite as $A^\top(\mathbf b-A\mathbf x)=\mathbf 0$: the residual is orthogonal to every column of $A$, i.e. $A\mathbf x$ is the orthogonal projection of $\mathbf b$ onto the column space. If $A$ has linearly independent columns ($\operatorname{rk}A=n$) then $A^\top A\in\mathbb R^{n\times n}$ is invertible and $\mathbf x=(A^\top A)^{-1}A^\top\mathbf b$, the Moore-Penrose pseudo-inverse solution mentioned in the book (Sec. 2.3.4), which is also the minimum-norm least-squares solution. This is linear regression: $A$ is the data (design) matrix, $\mathbf x$ the weights.

**Numerical warning.** Forming $A^\top A$ squares the condition number (see below). Production code uses a QR factorisation or the SVD (`np.linalg.lstsq` uses the SVD) instead of the normal equations, and the book itself says that for numerical precision it is generally not recommended to compute the inverse or pseudo-inverse. If $\operatorname{rk}(A)<n$ the minimiser is not unique; the pseudo-inverse via the SVD picks the one with smallest norm.

## Conditioning and why we call solve, not inv

The **condition number** $\kappa(A)=\|A\|\,\|A^{-1}\|=\sigma_{\max}/\sigma_{\min}$ (ratio of the largest to the smallest singular value, previous lessons) bounds how much relative errors in $\mathbf b$ or $A$ can be amplified in $\mathbf x$. Rule of thumb: you lose about $\log_{10}\kappa$ decimal digits out of the roughly 16 you have in float64. If $\kappa$ is near $10^{16}$ the matrix is numerically singular.

Practical consequences:

1. **Nearly dependent columns** (correlated features) mean tiny $\sigma_{\min}$, large $\kappa$, and unstable weights. Ridge regularisation $A^\top A+\lambda I$ raises $\sigma_{\min}^2$ to $\sigma_{\min}^2+\lambda$ and makes the problem stable.
2. **Never write `inv(A) @ b`.** `np.linalg.solve(A, b)` performs an LU factorisation with partial pivoting and two triangular solves; it never forms $A^{-1}$, does less work, and is typically at least as accurate. Inverting is only justified if you truly need the entries of $A^{-1}$.
3. For $A$ symmetric positive definite (covariances, kernel matrices, $A^\top A+\lambda I$) use a Cholesky factorisation: about half the work of LU.
4. Do not trust a determinant near zero as a test for singularity (it depends on scale); use the rank or $\kappa$.

## Worked example

All numbers are reproduced in the exercise.

**1. Unique solution.** The $3\times3$ system above: `solve` returns $(2,2,2)$, determinant $-4$ (non-zero, so invertible).

**2. Rank and nullity.** The $3\times3$ matrix with rows $(1,2,1),(-2,-3,1),(3,5,0)$ has rank $2$ (singular values $7.21,\ 1.42,\ \approx10^{-16}$); the $4\times5$ matrix of Example 2.6 has rank 3, nullity 2. Check of the null-space vectors: both $(2,1,0,0,0)$ and $(2,0,-1,2,1)$ are mapped to $\mathbf 0$ by that matrix, and $(2,0,-1,1,0)$ is mapped to $(-3,2,0,-1)$, the right-hand side for $a=-1$.

**3. Least squares.** Fit a line $t\approx x_0+x_1s$ to the points $(s,t)=(1,3),(2,5),(3,6),(4,8)$. Here

$$
A=\begin{pmatrix}1&1\\1&2\\1&3\\1&4\end{pmatrix},\ \mathbf b=\begin{pmatrix}3\\5\\6\\8\end{pmatrix},\quad A^\top A=\begin{pmatrix}4&10\\10&30\end{pmatrix},\quad A^\top\mathbf b=\begin{pmatrix}22\\63\end{pmatrix}.
$$

Solving: determinant $4\cdot30-100=20$, so $\mathbf x=\tfrac1{20}\begin{psmallmatrix}30&-10\\-10&4\end{psmallmatrix}\begin{psmallmatrix}22\\63\end{psmallmatrix}=\tfrac1{20}(30,\,32)=(1.5,\,1.6)$. Intercept $1.5$, slope $1.6$. The residual is $\mathbf b-A\mathbf x=(-0.1,\,0.3,\,-0.3,\,0.1)$, with squared norm $0.2$, and $A^\top(\mathbf b-A\mathbf x)=\mathbf 0$ up to rounding: the residual is orthogonal to both columns. `np.linalg.lstsq` gives the same $(1.5,1.6)$. (The companion repository lesson quotes $1.7$ for the slope; recomputing by hand and by code gives $1.6$.)

**4. Conditioning.** $\kappa(A)=7.47$ but $\kappa(A^\top A)=55.78=7.47^2$: the normal equations square the condition number. For $B=\begin{psmallmatrix}1&1\\1&1.0001\end{psmallmatrix}$ ($\kappa\approx4\times10^4$): with $\mathbf b=(2,2.0001)$ the solution is $(1,1)$; change $\mathbf b$ to $(2,\,2.0002)$ (a relative change of $5\times10^{-5}$) and the solution jumps to $(0,2)$.

## Implementation

```python
import numpy as np

A = np.array([[2., 1, 1], [4, 3, 3], [2, 3, 1]])
b = np.array([8., 20, 12])
x = np.linalg.solve(A, b)                 # LU with partial pivoting, shape (3,)

rank = np.linalg.matrix_rank(np.array([[1., 2, 1], [-2, -3, 1], [3, 5, 0]]))   # 2

X = np.array([[1., 1], [1, 2], [1, 3], [1, 4]])      # (m, n) = (4, 2)
t = np.array([3., 5, 6, 8])                          # (m,)
w, *_ = np.linalg.lstsq(X, t, rcond=None)            # (n,) = [1.5, 1.6]
np.linalg.cond(X)                                    # 7.47
```

## Shapes

$A:(m,n)$, $\mathbf x:(n,)$, $\mathbf b:(m,)$. Augmented matrix $(m,n+1)$. Null-space basis: a matrix $(n,n-r)$ whose columns are solutions of $A\mathbf x=\mathbf 0$. Normal equations: $A^\top A:(n,n)$, $A^\top\mathbf b:(n,)$. For `np.linalg.solve`, $A$ must be square $(n,n)$; for rectangular $A$ use `lstsq`.

## Common misconceptions

- **A square system always has exactly one solution.** Only if the matrix has full rank; otherwise there are none or infinitely many.
- **More equations than unknowns means no solution.** Not necessarily: if the extra equations are consistent (redundant) a solution exists. Typically though, noisy data makes them inconsistent, hence least squares.
- **`x = inv(A) @ b` is the standard way.** Use `solve`; it is faster and more accurate.
- **Least squares gives a solution of $A\mathbf x=\mathbf b$.** It gives the closest point in the column space; the residual is non-zero.
- **A small determinant means ill-conditioned.** Scaling a matrix by $10^{-3}$ changes the determinant enormously and the condition number not at all.

## Key takeaways

- Solutions exist iff $\operatorname{rk}(A)=\operatorname{rk}(A|\mathbf b)$; they are unique iff additionally $\operatorname{rk}(A)=n$; otherwise they form particular solution $+\ \ker(A)$ with $\dim\ker(A)=n-\operatorname{rk}(A)$.
- Gaussian elimination turns the system into a staircase: pivots give the rank, free columns give the null space; use partial pivoting.
- No exact solution: $\min\|A\mathbf x-\mathbf b\|^2$, normal equations $A^\top A\mathbf x=A^\top\mathbf b$, residual orthogonal to the column space. Prefer QR or the SVD for accuracy.
- $\kappa=\sigma_{\max}/\sigma_{\min}$ measures sensitivity; call `solve` (or `lstsq`), do not form the inverse.

## Explain it yourself

1. A $3\times3$ matrix has rank 2. What can happen to $A\mathbf x=\mathbf b$, depending on $\mathbf b$, and how many free parameters are there?
2. Why is the least-squares residual perpendicular to the columns of $A$, and why does that give the normal equations?

## Further reading

- *Mathematics for Machine Learning*, Sec. 2.3 (PDF p.33), Sec. 2.6.2 (PDF p.53), Sec. 2.7.3 (PDF p.64).
- Repository lesson `17-linear-systems` for LU, QR, Cholesky, the conjugate gradient method and iterative solvers.

## Mastery criteria

You can bring a small system to row-echelon form by hand, count the rank and free variables, write the general solution as particular plus null space, decide solvability from $\operatorname{rk}(A)$ versus $\operatorname{rk}(A|\mathbf b)$, set up and solve the normal equations for a line fit, and explain what the condition number says about trusting a solution.
