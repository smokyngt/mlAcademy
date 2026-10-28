---
id: math.17
title: "Vector spaces, basis, orthogonality and QR"
summary: What a subspace, a basis and a dimension really are, how to change coordinates, how to project onto a subspace, and how Gram-Schmidt and the QR decomposition turn that into a stable way to do least squares.
minutes: 90
teaches: [la.vector-space, la.basis, la.projection, la.qr]
requires: [la.matrix, la.linear-map, la.dot, la.linear-systems]
sources:
  - "math/math for ML.pdf, Sec. 2.4 Vector Spaces (PDF p.41-42): groups and the vector-space definition; subspaces. READ (p.41-42 in detail)."
  - "math/math for ML.pdf, Sec. 2.6 Basis and Rank (PDF p.50-53): generating set, span, basis as minimal generating set and maximal independent set, dimension, finding a basis by pivot columns (Example 2.17), rank. READ."
  - "math/math for ML.pdf, Sec. 2.7 Linear Mappings, basis change (PDF p.62-63): composition of basis changes, Def. 2.21 equivalence and Def. 2.22 similarity. READ."
  - "math/math for ML.pdf, Sec. 3.4-3.6 (PDF p.82-86): orthogonality, orthogonal matrices and length/angle preservation (3.31-3.32), orthonormal basis, Gram-Schmidt mention, orthogonal complement and unique decomposition. READ."
  - "math/math for ML.pdf, Sec. 3.8 Orthogonal Projections (PDF p.87-94): projection definition P^2 = P, projection onto a line and onto a subspace, normal equation, P = B (B^T B)^-1 B^T, Example 3.11. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/17-linear-systems/docs/en.md (QR section, lines ~160-200): the route R x = Q^T b and the Gram-Schmidt steps. READ."
---

## Where this fits

Lesson [[math.11]] taught you to solve $Ax=b$ and introduced rank. Two questions were left open. First: what exactly is the *space* that the columns of $A$ live in, and how do we measure its size? (Vector spaces, bases, dimension.) Second: when $Ax=b$ has no exact answer, which $x$ is "closest", and how do we compute it without losing digits? (Orthogonal projection, Gram-Schmidt, QR.)

The answers are used everywhere. Linear regression ([[cml.01]]) is an orthogonal projection of the target onto the span of the feature columns. PCA ([[math.07]]) is the search for a good orthonormal basis. Whitening, embedding dimensions, orthogonal weight initialisation and the numerically safe way to fit a model are all consequences of this lesson. It builds on dot products and norms ([[f2n.02]]) and matrices as linear maps ([[f2n.03]]).

## The idea in plain words

- A **vector space** is a collection of things you may add together and stretch by numbers, and you never leave the collection. Arrows in the plane qualify. So do lists of 768 numbers, and so do polynomials. All that matters is that the two operations behave sensibly.
- A **subspace** is a vector space living inside a bigger one: a flat sheet through the origin. A line through the origin in 3-D is one; a plane through the origin is one. A plane that misses the origin is *not*: you cannot even shrink one of its points to zero and stay on it.
- **Span** is "everything you can build from these arrows by stretching and adding". **Independent** arrows are ones where none is wasted: none can be built from the others. A **basis** is a crew with no waste that still reaches everything. The crew size is the **dimension**, the number of genuinely independent directions.
- **Coordinates** are directions for reaching a point with a particular crew: "two steps along the first arrow, three along the second". Change the crew and the *same point* gets different coordinates. A change of basis is just translation between two such languages.
- **Orthogonal** means at right angles. An *orthonormal* crew (right angles, unit length) is the dream: coordinates come from plain dot products, with no equations to solve.
- **Projection** is the shadow of a point on a sheet, cast by a light shining perpendicular to the sheet. The shadow is the closest point of the sheet to the original, and the line from point to shadow is perpendicular to the sheet.
- **Gram-Schmidt** manufactures an orthonormal crew from any crew: take the arrows one at a time, subtract the part that points along what you already have, and what is left is new. Recording what you subtracted gives the **QR decomposition**.

## Vector spaces and subspaces

A **real vector space** is a set $V$ with addition $u+v$ and scaling $c\,v$ ($c\in\mathbb{R}$) obeying these rules, in words: adding or scaling never leaves $V$ (closure); addition is order-independent and regrouping-independent; there is a zero vector $\mathbf 0$; every $v$ has an opposite $-v$; and scaling behaves like ordinary multiplication and distributes over addition ($1\,v=v$, $(ab)v=a(bv)$, $(a+b)v=av+bv$, $c(u+v)=cu+cv$). The book (Sec. 2.4) builds this from the notion of a group, and calls the first set of rules "$(V,+)$ is an Abelian group". You will rarely check the axioms for $\mathbb{R}^n$, but you will constantly check them for *subsets*.

A **subspace** $U\subseteq V$ is a subset that is itself a vector space. In practice three checks suffice:

1. $\mathbf 0\in U$;
2. closed under addition: $u,w\in U\Rightarrow u+w\in U$;
3. closed under scaling: $u\in U,\ c\in\mathbb{R}\Rightarrow cu\in U$.

| Subset of $\mathbb{R}^3$ | Subspace? | Why |
|---|---|---|
| $\{(x,y,0)\}$ (the floor) | yes | contains $\mathbf 0$; sums and multiples stay on the floor |
| $\{t(1,2,2)\}$ (a line through $\mathbf 0$) | yes | multiples of one vector |
| solution set of $Ax=\mathbf 0$ | yes | $A(u+w)=\mathbf 0$ and $A(cu)=\mathbf 0$ (the null space) |
| solution set of $Ax=b$, $b\neq\mathbf 0$ | no | $\mathbf 0$ is not a solution |
| $\{(x,y,1)\}$ (a plane above the floor) | no | no zero vector; the sum of two points has third coordinate $2$ |
| $\{(x,y,z): x\ge 0\}$ (a half-space) | no | scaling by $-1$ leaves it |
| unit sphere $\|v\|=1$ | no | no zero; $2v$ is not on it |

Vector spaces are not only lists of numbers. The polynomials of degree at most 2 form a 3-dimensional space (basis $1,x,x^2$); $n\times n$ matrices form an $n^2$-dimensional one. This generality is why feature spaces, function spaces and embedding spaces can all use the same machinery.

### Span, independence and the column space

The **span** of vectors $v_1,\dots,v_k$ is the set of all linear combinations $c_1v_1+\dots+c_kv_k$. It is always a subspace. For a matrix $A=[a_1,\dots,a_n]$ with columns $a_j$, the span of the columns is the **column space** $\operatorname{col}(A)=\{Ax\}$. This gives the geometric reading of $Ax=b$: it is solvable *exactly when $b$ lies in the column space*. Least squares is what you do when it does not.

Vectors are **linearly independent** if the only way to combine them into $\mathbf 0$ is with all coefficients zero; equivalently, none is a combination of the others; equivalently, $Ax=\mathbf 0$ has only the solution $x=\mathbf 0$. Practical test (book Sec. 2.5, via Gaussian elimination): put them in the columns of a matrix and row-reduce; they are independent iff every column is a pivot column.

## Basis, dimension and rank-nullity

A **basis** of $V$ is an independent list that spans $V$. The book states equivalent characterisations: a basis is a *minimal* spanning set, a *maximal* independent set, and a list such that every vector has a **unique** set of coefficients. Bases are not unique (in $\mathbb{R}^3$ the standard basis and $(1,0,0),(1,1,0),(1,1,1)$ both work) but all bases of $V$ have the same size, the **dimension** $\dim V$. Dimension counts independent directions, not entries: the line $\operatorname{span}\{(0,1)\}$ sits in $\mathbb{R}^2$ but has dimension 1.

To find a basis of a span: put the spanning vectors as columns of $A$, row-reduce, and keep the original vectors in the *pivot columns* (book Example 2.17: of four vectors in $\mathbb{R}^5$, three are pivot columns).

**Rank** is $\operatorname{rk}(A)=\dim\operatorname{col}(A)$, the number of pivots; it also equals the dimension of the row space. The **null space** $\operatorname{null}(A)=\{x:Ax=\mathbf 0\}$ is the set of inputs the map crushes to zero. The two fit together:

$$
\boxed{\operatorname{rk}(A)+\dim\operatorname{null}(A)=n}\qquad (A\text{ has }n\text{ columns}).
$$

Why: row-reduce $A$. Each column is either a pivot column (it adds one dimension to the column space) or a free column (it adds one free parameter to the solutions of $Ax=\mathbf 0$, hence one dimension of the null space). The $n$ columns are split between the two kinds. In words: of the $n$ input dimensions, the map keeps $\operatorname{rk}(A)$ and destroys the rest. So $\operatorname{rk}(A)=n$ means the map loses nothing, and a wide matrix ($n>m$) always has a non-trivial null space.

## Coordinates and change of basis

Let $b_1,\dots,b_n$ be a basis of $\mathbb{R}^n$, stacked as the columns of an invertible matrix $S$. A vector $x$ (in standard coordinates) has coordinates $c$ with respect to this basis if

$$
x = c_1b_1+\dots+c_nb_n = S\,c \quad\Longrightarrow\quad c=S^{-1}x .
$$

So $S$ turns basis coordinates into standard ones and $S^{-1}$ goes back. Example: $S=\begin{pmatrix}1&1\\0&1\end{pmatrix}$, $x=(3,2)$: $c=S^{-1}x=(1,2)$; check $1\cdot(1,0)+2\cdot(1,1)=(3,2)$. Same point, new language.

A linear map with matrix $M$ in standard coordinates acts on new coordinates as follows: convert to standard ($S$), apply $M$, convert back ($S^{-1}$):

$$
\tilde M = S^{-1} M S .
$$

This is the book's Def. 2.22: $\tilde M$ and $M$ are **similar**. Similar matrices describe the *same* map in different bases, so they share eigenvalues, trace, determinant and rank. Diagonalisation ([[math.07]]) is precisely the search for a basis (the eigenvectors) in which $\tilde M$ is diagonal. With $M=\begin{pmatrix}2&1\\0&3\end{pmatrix}$ and the $S$ above, $\tilde M=\operatorname{diag}(2,3)$: the columns of $S$ are the eigenvectors $(1,0)$ and $(1,1)$.

## Orthogonality, orthonormal bases and orthogonal matrices

Vectors $x,y$ are **orthogonal** if $x^\top y=0$. Orthogonal non-zero vectors are automatically independent. An **orthonormal basis** has pairwise orthogonal vectors of length 1. Stack them as the columns of $Q$:

$$
Q^\top Q = I .
$$

Entry $(i,j)$ of $Q^\top Q$ is $q_i^\top q_j$, which is 1 on the diagonal and 0 elsewhere. If $Q$ is square this is an **orthogonal matrix** and $Q^{-1}=Q^\top$: the inverse costs nothing. Two key properties (book 3.31-3.32):

$$
\|Qx\|^2=(Qx)^\top(Qx)=x^\top Q^\top Q x=x^\top x,\qquad (Qx)^\top(Qy)=x^\top y .
$$

So $Q$ preserves lengths, distances and angles; it is a rotation, possibly with a reflection. And in an orthonormal basis the coordinates are free: since $x=\sum_j c_jq_j$ and $q_i^\top q_j=\delta_{ij}$, we get $c_i=q_i^\top x$, with no linear system to solve.

The **orthogonal complement** $U^\perp$ of a subspace $U\subseteq\mathbb{R}^n$ is the set of vectors orthogonal to all of $U$. If $\dim U=m$ then $\dim U^\perp=n-m$, $U\cap U^\perp=\{\mathbf 0\}$, and every $x$ splits uniquely as $x=u+w$ with $u\in U$, $w\in U^\perp$ (book Sec. 3.6). That $u$ is the projection of the next section. A useful consequence: $\operatorname{null}(A)$ is the orthogonal complement of the row space of $A$, since each equation $a_i^\top x=0$ says $x\perp a_i$.

## Derivation: the projection matrix

Let $U=\operatorname{col}(A)$ where $A\in\mathbb{R}^{n\times m}$ has independent columns ($m\le n$). Given $x\in\mathbb{R}^n$ we want the $p\in U$ closest to $x$.

1. $p\in U$ means $p=A\lambda$ for some coefficient vector $\lambda\in\mathbb{R}^m$.
2. "Closest" means the error $x-p$ is orthogonal to $U$ (the shadow property; it is also what calculus gives when you minimise $\|x-A\lambda\|^2$). Orthogonal to $U$ means orthogonal to every column: $a_j^\top(x-A\lambda)=0$ for all $j$, i.e. stacked, $A^\top(x-A\lambda)=\mathbf 0$.
3. Expand: $A^\top A\,\lambda=A^\top x$ (the **normal equations**, book 3.56).
4. $A^\top A$ is $m\times m$ and invertible because the columns are independent, so $\lambda=(A^\top A)^{-1}A^\top x$.
5. Then $p=A\lambda$:

$$
\boxed{P=A\,(A^\top A)^{-1}A^\top},\qquad p=Px .
$$

Properties, each a one-liner:

- $P^2=A(A^\top A)^{-1}\,(A^\top A)\,(A^\top A)^{-1}A^\top=P$: projecting twice changes nothing (this is the book's definition of a projection).
- $P^\top=P$ because $(A^\top A)^{-1}$ is symmetric. Idempotent plus symmetric is what makes it an *orthogonal* projection.
- Its eigenvalues are $1$ (on $U$) and $0$ (on $U^\perp$), so $\operatorname{tr}P=m=\dim U$.
- If the columns are orthonormal ($A=Q$, $Q^\top Q=I$) it collapses to $P=QQ^\top$, the sum of the rank-one shadows $q_jq_j^\top$.

With one column $b$ this is $P=bb^\top/(b^\top b)$.

## Worked example: projection

Book Example 3.11: $A=\begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix}$, $x=(6,0,0)$.

- $A^\top A=\begin{pmatrix}3&3\\3&5\end{pmatrix}$, $A^\top x=(6,0)$, so $\lambda=(5,-3)$.
- $p=A\lambda=(5,2,-1)$; the error $x-p=(1,-2,1)$ has length $\sqrt6\approx2.449$.
- $P=\tfrac16\begin{pmatrix}5&2&-1\\2&2&2\\-1&2&5\end{pmatrix}$, and $\operatorname{tr}P=2$.
- Check $A^\top(x-p)=(1-2+1,\ 0-2+2)=(0,0)$.

## Gram-Schmidt and the QR decomposition

**Goal.** Given independent columns $a_1,\dots,a_n$, build orthonormal $q_1,\dots,q_n$ such that $\operatorname{span}(q_1..q_k)=\operatorname{span}(a_1..a_k)$ for every $k$.

**Derivation.** Step 1: $q_1=a_1/\|a_1\|$. Step $k$: we want the part of $a_k$ that is new. The part already covered is the projection of $a_k$ onto the span of $q_1..q_{k-1}$, which by the orthonormal formula above is $\sum_{i<k}(q_i^\top a_k)q_i$. Subtract it:

$$
v_k=a_k-\sum_{i<k}(q_i^\top a_k)\,q_i,\qquad q_k=\frac{v_k}{\|v_k\|}.
$$

$v_k$ is orthogonal to all earlier $q_i$ (the projection error is perpendicular to the subspace) and non-zero because $a_k$ is independent of the earlier columns. Now read the same equations backwards: $a_k=\sum_{i<k}(q_i^\top a_k)q_i+\|v_k\|q_k$. In matrix form:

$$
\boxed{A=QR},\qquad R_{ik}=q_i^\top a_k\ (i<k),\quad R_{kk}=\|v_k\|>0,\quad R_{ik}=0\ (i>k).
$$

$Q$ is $n\times m$ with orthonormal columns, $R$ is $m\times m$ upper triangular. Shapes: $A:(n,m)=Q:(n,m)\cdot R:(m,m)$ ("reduced" QR; the "full" QR pads $Q$ to $n\times n$ and $R$ to $n\times m$).

**Worked example.** $a_1=(1,1,0)$, $a_2=(1,0,1)$, $a_3=(0,1,1)$.

- $q_1=(1,1,0)/\sqrt2$, $R_{11}=\sqrt2\approx1.4142$.
- $R_{12}=q_1^\top a_2=1/\sqrt2\approx0.7071$; $v_2=a_2-R_{12}q_1=(0.5,-0.5,1)$; $R_{22}=\|v_2\|=\sqrt{1.5}\approx1.2247$; $q_2=v_2/R_{22}$.
- $R_{13}\approx0.7071$, $R_{23}\approx0.4082$, $R_{33}\approx1.1547$.

These match `np.linalg.qr` up to signs: NumPy (LAPACK) returns a first column of $Q$ equal to $-q_1$ and a negative $R_{11}$. The decomposition is unique only up to flipping a column of $Q$ together with the matching row of $R$; demanding $R_{kk}>0$ makes it unique, so fix the signs before comparing.

**Classical vs modified.** The formula above computes every coefficient $q_i^\top a_k$ from the *original* $a_k$ (classical Gram-Schmidt). In floating point, when columns are nearly dependent, the computed $q$'s lose orthogonality. The **modified** variant subtracts the projections one at a time from the running vector, $v\leftarrow v-(q_i^\top v)q_i$. It is identical in exact arithmetic and noticeably better in floating point. Production libraries use Householder reflections instead (what `np.linalg.qr` calls via LAPACK), which are better still.

## Least squares via QR

Overdetermined system: $A\in\mathbb{R}^{n\times m}$ with $n>m$ and $b$ outside the column space. Minimise $\|Ax-b\|^2$. By the projection derivation the minimiser solves the normal equations $A^\top Ax=A^\top b$ (the equations of [[cml.01]]). Insert $A=QR$:

$$
A^\top A=R^\top Q^\top QR=R^\top R,\qquad A^\top b=R^\top Q^\top b .
$$

So $R^\top Rx=R^\top Q^\top b$, and since $R$ is invertible, $R^\top$ cancels:

$$
\boxed{Rx=Q^\top b}
$$

Solve by back-substitution (the system is triangular, $O(m^2)$). The product $A^\top A$ is never formed.

**Why this is more stable.** The condition number $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ measures how much relative errors in the data or arithmetic can be amplified in the solution (see [[math.15]]). Forming $A^\top A$ **squares** it: $\kappa(A^\top A)=\kappa(A)^2$. Solving the normal equations therefore loses about twice as many digits as the problem demands. In float64 (about 16 digits) with $\kappa(A)=10^{5}$, the normal equations leave roughly $16-10=6$ digits and QR roughly $16-5=11$. Since $Q$ is orthogonal, multiplying by $Q^\top$ does not change conditioning and $\kappa(R)=\kappa(A)$.

Numbers from the exercise: fit a degree-7 polynomial to 40 equally spaced points in $[0,1]$ (Vandermonde matrix $V$, shape $40\times8$) with true coefficients all $1$ and noiseless $y=V\mathbf 1$. Then $\kappa(V)\approx1.1\times10^5$ and $\kappa(V^\top V)\approx1.2\times10^{10}$; the normal equations recover the coefficients to about $10^{-6}$ (we measured $8\times10^{-7}$), QR with Householder reflections (`np.linalg.qr`) to about $10^{-12}$. A caveat found while writing the exercise: our own *modified Gram-Schmidt* followed by $Q^\top b$ only reaches about $2\times10^{-7}$, because $Q$ is orthonormal only to about $10^{-12}$ and that error is amplified by $\kappa$. Running Gram-Schmidt on the augmented matrix $[A\mid b]$ and reading $Q^\top b$ off the last column of $R$ restores about $10^{-13}$. Householder QR is the robust default. If the matrix may be rank-deficient, use the SVD route ([[math.07]]), which is what `np.linalg.lstsq` does.

## Implementation

```python
import numpy as np

def gram_schmidt(A):
    """Modified Gram-Schmidt. A: (n, m), independent columns -> Q (n, m), R (m, m)."""
    n, m = A.shape
    Q = np.zeros((n, m)); R = np.zeros((m, m))
    V = A.astype(float).copy()
    for k in range(m):
        R[k, k] = np.linalg.norm(V[:, k])
        Q[:, k] = V[:, k] / R[k, k]
        for j in range(k + 1, m):          # remove q_k from all later columns now
            R[k, j] = Q[:, k] @ V[:, j]
            V[:, j] -= R[k, j] * Q[:, k]
    return Q, R

A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1.]])
Q, R = gram_schmidt(A)
print(np.allclose(Q @ R, A), np.allclose(Q.T @ Q, np.eye(3)))
B = np.array([[1, 0], [1, 1], [1, 2.]])
P = B @ np.linalg.inv(B.T @ B) @ B.T      # use solve() when you only need P @ x
```

In production, `np.linalg.qr(A)` (Householder), `scipy.linalg.solve_triangular` and `np.linalg.lstsq` (SVD-based) do this work.

## Shapes

$A:(n,m)$; $P:(n,n)$ with rank $m$; $Q:(n,m)$ reduced or $(n,n)$ full; $R:(m,m)$ reduced or $(n,m)$ full; coordinates $c=S^{-1}x$ with $S:(n,n)$, $x:(n,)$, $c:(n,)$; $\tilde M=S^{-1}MS:(n,n)$.

## Why this matters for ML

- **Feature spaces.** The columns of a design matrix span the space of functions a linear model can express. Collinear features mean dependent columns, rank deficiency and a non-unique fit; a large $\kappa$ means "nearly so".
- **Regression is projection.** The fitted values $\hat y=X\hat w$ are the projection of $y$ onto $\operatorname{col}(X)$, the residual is orthogonal to every feature, and $P$ is the "hat matrix".
- **PCA basis.** PCA picks an orthonormal basis of feature space ordered by variance; a datapoint's coordinates in that basis are dot products $q_i^\top x$, and keeping $k$ axes is an orthogonal projection onto a $k$-dimensional subspace ([[math.07]]).
- **Whitening.** Rotating into the eigenbasis of the covariance (an orthogonal change of basis) and dividing each axis by its standard deviation yields uncorrelated, unit-variance features.
- **Embeddings.** The embedding dimension is the dimension of the space; the readable directions are those you can tell apart by dot products, and random high-dimensional vectors are nearly orthogonal. Orthogonal weight initialisation (take $Q$ from the QR of a random matrix) keeps norms from exploding or vanishing, since $\|Qx\|=\|x\|$.

## Common misconceptions

- **"A subspace is any flat set."** It must pass through the origin. Solution sets of $Ax=b$ with $b\neq\mathbf 0$ are shifted copies, not subspaces.
- **"Dimension is the length of the vectors."** It is the number of independent directions; a line in $\mathbb{R}^{100}$ has dimension 1.
- **"Orthogonal matrix means orthogonal rows only."** Rows and columns are both orthonormal; "orthonormal matrix" would be the better name. A non-square $Q$ with $Q^\top Q=I$ has $QQ^\top\neq I$ (it is a projection instead).
- **"Orthogonal and uncorrelated are the same."** Only for mean-centred data.
- **"A projection can be undone."** It crushes $U^\perp$ to zero; the information is gone.
- **"Classical Gram-Schmidt is fine."** Mathematically yes, numerically it can lose orthogonality on near-dependent columns; use the modified version or Householder.
- **"Just use the normal equations; it is one line."** It squares the condition number. Fine for small well-conditioned problems, risky otherwise.

## Key takeaways

- Subspace test: contains $\mathbf 0$, closed under $+$ and scaling. Null space and column space are subspaces.
- Basis = independent + spanning; dimension = its size; $\operatorname{rk}(A)+\dim\operatorname{null}(A)=n$.
- Coordinates: $c=S^{-1}x$; a map in the new basis is $S^{-1}MS$ (similar matrices).
- $Q^\top Q=I$ gives length-preserving maps and free coordinates $c=Q^\top x$.
- Projection onto $\operatorname{col}(A)$: $P=A(A^\top A)^{-1}A^\top$, with $P^2=P=P^\top$.
- Gram-Schmidt builds $Q$ and records $R$ in $A=QR$; least squares becomes $Rx=Q^\top b$, more stable than the normal equations because $\kappa(A^\top A)=\kappa(A)^2$.

## Explain it yourself

1. Why is the plane $x+y+z=1$ not a subspace of $\mathbb{R}^3$, and how is it related to one that is?
2. Without formulas: why must the error of a least-squares fit be perpendicular to every feature column?

## Further reading

- *Mathematics for Machine Learning*, Sec. 2.4-2.7 (PDF p.41-63) and Sec. 3.4-3.8 (PDF p.82-94).
- The linear-systems lesson of the companion repository (`01-math-foundations/17-linear-systems`) for LU, Cholesky and conditioning.

## Mastery criteria

You can test whether a set is a subspace, find a basis and the dimension of a column and null space and state rank-nullity, convert coordinates between bases, derive $P=A(A^\top A)^{-1}A^\top$ and verify $P^2=P=P^\top$, run Gram-Schmidt by hand on three vectors, and explain why QR beats the normal equations on an ill-conditioned problem.
