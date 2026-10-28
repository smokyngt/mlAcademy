---
id: math.07
title: Eigenvalues, eigenvectors and the SVD
summary: The directions a matrix merely stretches, how they let you take powers of a matrix, and the decomposition that works for every matrix.
minutes: 70
teaches: [la.eigen, la.svd]
requires: [la.linear-map]
sources:
  - "math/math for ML.pdf, Sec. 4.2 Eigenvalues and Eigenvectors (PDF p.111-114): definition, equivalent statements, characteristic polynomial, worked example A = [[4,2],[1,3]], multiplicities. READ."
  - "math/math for ML.pdf, Sec. 4.4 Eigendecomposition and Diagonalization (PDF p.121-123): A = P D P^-1 iff the eigenvectors form a basis. READ."
  - "math/math for ML.pdf, Sec. 4.5 Singular Value Decomposition (PDF p.125-128): SVD theorem, geometric intuition, worked example with A in R^{3x2}. READ."
---

## Where this fits

In lesson 3 a matrix was a transformation that bends the plane. Most directions get turned *and* stretched, which makes a transformation hard to picture. Two questions simplify everything: **is there a direction that is only stretched, not turned?** (eigenvectors) and, for matrices that are not square or not nice, **what is the simplest description of "rotate, stretch, rotate"?** (the SVD).

These tools are everywhere downstream: PCA (lesson on principal components) is an eigenvalue problem; the stability of gradient descent (lesson 7) is decided by the eigenvalues of the loss curvature; low-rank adaptation of large models (LoRA) is a truncated SVD idea; and the singular values of a weight matrix tell you how it amplifies signals during backpropagation.

## The idea in plain words

- Picture a rubber sheet with a grid drawn on it, and a matrix as a way of stretching the sheet. Almost every arrow drawn on the sheet changes direction when you stretch it. But there are usually a few special directions along which the sheet is **only pulled or squeezed, never twisted**: an arrow pointing that way stays on its own line. Those are the **eigenvectors**, and the amount of stretching along each is its **eigenvalue** (a negative one means the arrow is also flipped). The word "eigen" is German for "own" or "characteristic" (the book makes the same remark): these are the transformation's own directions.
- If you use those special directions as your coordinate axes, the transformation becomes trivially simple: just independent stretching along each axis. That is **diagonalisation**, and it makes repeated application cheap: applying the matrix 100 times just raises each stretch factor to the 100th power.
- Not every matrix is square or has enough special directions. The **SVD** is the universal version: *every* matrix, of any shape, equals "rotate (or reflect), then stretch along the axes, then rotate again". The stretch factors are the **singular values**.

## Eigenvalues and eigenvectors

**Definition** (Deisenroth et al., Def. 4.6). For a square `A ∈ R^{n×n}`, a number `λ` is an *eigenvalue* and a non-zero vector `x` the matching *eigenvector* if

$$
Ax=\lambda x .
$$

Four equivalent statements (the book lists them): `λ` is an eigenvalue ⇔ `(A − λI)x = 0` has a non-trivial solution ⇔ `A − λI` is singular (rank `< n`) ⇔ `det(A − λI) = 0`.

The last form gives a recipe. `p_A(λ) = det(A − λI)` is the **characteristic polynomial**, and eigenvalues are its roots. For a `2×2` matrix `[[a,b],[c,d]]`:

$$
p(\lambda)=\lambda^2-(a+d)\lambda+(ad-bc)=\lambda^2-\operatorname{tr}(A)\,\lambda+\det(A).
$$

Two useful facts follow: the **eigenvalues add up to the trace** and **multiply to the determinant**. Eigenvectors are not unique: if `x` works so does `cx` for any `c ≠ 0`, because `A(cx) = λ(cx)`. The set of all eigenvectors of one eigenvalue, plus zero, is a subspace called the **eigenspace**, the null space of `A − λI`.

### Worked example (the one used in the book)

$$
A=\begin{pmatrix}4&2\\1&3\end{pmatrix}.
$$

1. Characteristic polynomial: `(4 − λ)(3 − λ) − 2·1 = λ² − 7λ + 10 = (λ − 2)(λ − 5)`. Check: trace `7 = 2 + 5`, determinant `10 = 2·5`. ✓
2. For `λ = 5`: `(A − 5I)x = [[−1,2],[1,−2]]x = 0` gives `x₁ = 2x₂`, so `E₅ = span{(2, 1)}`. Check: `A(2,1) = (4·2+2·1, 1·2+3·1) = (10, 5) = 5·(2,1)`. ✓
3. For `λ = 2`: `[[2,2],[1,1]]x = 0` gives `x₂ = −x₁`, so `E₂ = span{(1, −1)}`. Check: `A(1,−1) = (4−2, 1−3) = (2, −2) = 2·(1,−1)`. ✓

The matrix pulls the plane by a factor 5 along `(2,1)` and by 2 along `(1,−1)`.

### Facts worth remembering

- `A` and `Aᵀ` have the same eigenvalues (not necessarily the same eigenvectors).
- A symmetric matrix has **real** eigenvalues and eigenvectors for different eigenvalues that are **orthogonal**; a symmetric *positive definite* matrix has all eigenvalues `> 0`. This is the case that matters for covariance matrices (PCA) and for the Hessian of a convex loss (gradient descent).
- Eigenvalues with multiplicity can be defective: `[[2,1],[0,2]]` has eigenvalue 2 twice but only the single direction `(1,0)` (the book's Example 4.6).
- A rotation by 90 degrees has no real eigenvector (it turns every direction); its eigenvalues are complex.

## Diagonalisation

Put the eigenvectors as columns of `P` and the eigenvalues on the diagonal of `D`. Then `AP = PD`, because the `j`-th column of `AP` is `Ap_j = λ_j p_j`, the `j`-th column of `PD`. If the eigenvectors form a basis (so `P` is invertible), then

$$
\boxed{A=PDP^{-1}}\qquad\text{(Theorem 4.20 in the book: possible iff the eigenvectors span }\mathbb{R}^n\text{).}
$$

Read right to left, as the book's figure does: `P⁻¹` changes to the eigenvector coordinates, `D` stretches each axis, `P` changes back. Consequences: `A^k = P D^k P⁻¹` (only the diagonal numbers are raised to a power), `det A = ∏ λ_i`, and `A⁻¹ = P D⁻¹ P⁻¹`.

Example: with `P = [[2,1],[1,−1]]` (columns `(2,1)` and `(1,−1)`) and `D = diag(5, 2)`: `A³ = P·diag(125, 8)·P⁻¹ = [[86,78],[39,47]]`, which matches multiplying `A` three times by hand.

## The singular value decomposition

**Theorem** (book Thm 4.22). Every `A ∈ R^{m×n}`, square or not, of rank `r`, can be written

$$
\boxed{A=U\,\Sigma\,V^\top}\qquad U:(m\times m)\text{ orthogonal},\ \ \Sigma:(m\times n)\text{ diagonal},\ \ V:(n\times n)\text{ orthogonal},
$$

with `Σ_ii = σ_i ≥ 0` the **singular values**, ordered `σ₁ ≥ σ₂ ≥ … ≥ 0`, and `r` of them non-zero. Columns of `U` are the left-singular vectors, columns of `V` the right-singular vectors.

**Geometry** (the book's Fig. 4.8): `Vᵀ` rotates (a change of basis) in the input space, `Σ` stretches each axis by `σ_i` and adds or drops dimensions to reach the output space, `U` rotates in the output space. A unit circle becomes an ellipse whose semi-axes have lengths `σ_i`. The eigendecomposition applies one basis change and undoes it in the same space; the SVD uses two different bases, linked by `Σ`.

**Where it comes from.** `AᵀA = VΣᵀUᵀUΣVᵀ = V(ΣᵀΣ)Vᵀ`. So `V` holds the eigenvectors of the symmetric matrix `AᵀA`, and the eigenvalues are `σ_i²`. Hence `σ_i = √λ_i(AᵀA)`, and `u_i = A v_i / σ_i` for `σ_i > 0`. (This is also why PCA, which diagonalises `XᵀX`, can be done with the SVD of `X`.)

### Worked example (from the book, Example 4.12)

$$
A=\begin{pmatrix}1&-0.8\\0&1\\1&0\end{pmatrix}\ (3\times2).\qquad A^\top A=\begin{pmatrix}2&-0.8\\-0.8&1.64\end{pmatrix}.
$$

Its trace is `3.64` and determinant `2·1.64 − 0.64 = 2.64`, so the eigenvalues are `(3.64 ± 1.64)/2 = 2.64` and `1.00`. Therefore `σ₁ = √2.64 ≈ 1.625` and `σ₂ = 1`, which agree with the `1.62` and `1.0` printed in the book. A `3×2` matrix maps the plane into a 2-D tilted sheet inside 3-D space.

### Low-rank approximation

Keep only the `k` largest singular values: `A_k = Σ_{i≤k} σ_i u_i v_iᵀ`. This is the best rank-`k` approximation of `A` in the least-squares sense (Eckart-Young; the book's Sec. 4.6), and the error is exactly

$$
\|A-A_k\|_F=\sqrt{\sigma_{k+1}^2+\dots+\sigma_r^2}.
$$

Storing `A_k` costs `k(m+n+1)` numbers instead of `mn`: that is the basis of image compression, of PCA, and of low-rank adapters for large language models.

## Implementation

```python
import numpy as np

A = np.array([[4., 2.], [1., 3.]])
vals, vecs = np.linalg.eig(A)             # 5, 2 (order not guaranteed); columns are eigenvectors
Ak = vecs @ np.diag(vals ** 3) @ np.linalg.inv(vecs)   # A cubed

U, S, Vt = np.linalg.svd(np.array([[1, -.8], [0, 1], [1, 0]]), full_matrices=False)
```

For symmetric matrices use `np.linalg.eigh` (faster and returns orthonormal eigenvectors, ascending order).

## Shapes

`A: (m,n)`; reduced SVD `U: (m,r)`, `S: (r,)`, `Vt: (r,n)` with `r = min(m,n)`; `A ≈ (U[:, :k] * S[:k]) @ Vt[:k]`. Eigendecomposition: `P: (n,n)`, `D: (n,n)`.

## Common misconceptions

- **Every matrix has real eigenvectors.** No (rotations), and not every matrix is diagonalisable (defective ones). The SVD always exists.
- **Singular values are eigenvalues of `A`.** They are the square roots of the eigenvalues of `AᵀA`. They coincide only for symmetric positive semi-definite matrices.
- **Eigenvectors are unique.** Only up to scale (and sign); do not compare them elementwise.
- **A zero eigenvalue is harmless.** It means the matrix collapses a direction and is not invertible.

## Key takeaways

- `Ax = λx`: eigenvectors are the directions only stretched; `tr A = Σλ`, `det A = Πλ`.
- Diagonalise to take powers: `A^k = P D^k P⁻¹`.
- Symmetric matrices: real eigenvalues, orthogonal eigenvectors (the PCA and Hessian case).
- The SVD `A = UΣVᵀ` exists for every matrix; `σ_i² = λ_i(AᵀA)`; truncating gives the best low-rank approximation.

## Explain it yourself

1. Why does `det(A − λI) = 0` capture "there is a direction that is only stretched"?
2. Why does the SVD exist for a `3×2` matrix when the eigendecomposition does not?

## Further reading

- *Mathematics for Machine Learning*, Sec. 4.2 (PDF p.111), 4.4 (p.121), 4.5 (p.125), 4.6 low-rank approximation (p.135).

## Mastery criteria

You can find the eigenvalues and eigenvectors of a `2×2` matrix by hand, diagonalise it, state what the singular values mean geometrically, and compute a low-rank approximation with its error.
