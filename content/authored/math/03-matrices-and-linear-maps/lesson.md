---
id: f2n.03
title: Matrices, matrix multiplication and linear maps
summary: Shapes, the row-by-column rule, and the picture of a matrix as a transformation of space.
minutes: 55
teaches: [la.matrix, la.linear-map]
requires: [la.dot]
sources:
  - math/math for ML.pdf (Ch. 2 Linear Algebra, Sec. 2.2 Matrices and 2.7 Linear Mappings, PDF p.23-70)
  - ai-engineering-from-scratch/phases/01-math-foundations/02-vectors-matrices-operations/docs/en.md
  - ai-engineering-from-scratch/phases/01-math-foundations/03-matrix-transformations/docs/en.md
---

## Learning objectives

- Predict the output shape of any matrix product and detect shape errors by inspection.
- Compute a matrix-vector and matrix-matrix product by hand, in both the *rows* view and the *columns* view.
- Explain why $AB\neq BA$ in general and why $(AB)^\top=B^\top A^\top$.
- Read a $2\times2$ matrix as a transformation: its columns are where the basis vectors land.

## Prerequisites

Dot product ([[f2n.02]]).

## Where this fits

A layer of a neural network takes a list of numbers (say the 784 pixels of an image) and produces a new list (say 128 numbers that describe the image better). In practice you do this for 32 images at once. The object that does it is a **matrix**, and knowing how to read its shape is the single most useful skill for avoiding bugs in the machine learning part: most real errors are "these two shapes do not fit".

The lesson has two goals. First, make matrix multiplication mechanical (so shape mistakes are obvious). Second, give you a picture, a matrix as a way of *deforming space*, so the algebra stops feeling arbitrary.

## The idea in plain words

- A matrix is a **recipe table**. Each row says how to mix the inputs to make one output: "take 2 parts of the first number, 1 part of the second, and add them". Multiplying a matrix by a list just follows every recipe once.
- Geometrically, picture a sheet of **graph paper** that you can stretch, rotate or shear but never tear and never bend: straight lines stay straight, and the origin stays where it is. Such a deformation is what a matrix does to the plane.
- To know exactly what a given matrix does, you only need to know **where two arrows go**: the arrow pointing right (one unit) and the arrow pointing up. Their landing places are the two columns of the matrix, and every other point follows.
- The **shape rule** is just a plug-and-socket rule: an `m x n` matrix has `n` sockets on the input side; it fits something that offers `n` plugs, and it hands you back `m` outputs.

Play with the figure first: change one entry and watch the whole grid move.

## Intuition

A matrix is a *table of numbers* and also a *function on vectors*. As a function, $x\mapsto Ax$ takes a vector in $\mathbb{R}^n$ to one in $\mathbb{R}^m$ and does so **linearly**: it can stretch, rotate, shear, reflect and project space, but it always keeps the origin fixed and straight lines straight. Every dense layer of a neural network is one of these maps followed by a nonlinearity.

```viz
matrix-transform
```

## Formal definitions

$A\in\mathbb{R}^{m\times n}$ has $m$ rows and $n$ columns; entry $a_{ij}$ is in row $i$, column $j$.

**Matrix-vector product.** For $x\in\mathbb{R}^n$, $Ax\in\mathbb{R}^m$ with

$$
(Ax)_i=\sum_{j=1}^{n} a_{ij}x_j = (\text{row } i \text{ of } A)\cdot x .
$$

**Matrix-matrix product.** For $A\in\mathbb{R}^{m\times n}$, $B\in\mathbb{R}^{n\times p}$, $AB\in\mathbb{R}^{m\times p}$ with

$$
(AB)_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
$$

**Shape rule:** $(m\times n)(n\times p)=(m\times p)$. The two inner sizes must agree and then disappear.

**Transpose.** $(A^\top)_{ij}=a_{ji}$, shape $n\times m$. **Identity** $I$ has ones on the diagonal; $AI=IA=A$.

## Two ways to see the same product

*Rows view:* each output entry is a dot product (lesson 2). *Columns view:* writing $A$ by its columns $a^{(1)},\dots,a^{(n)}$,

$$
Ax=x_1a^{(1)}+x_2a^{(2)}+\cdots+x_na^{(n)}.
$$

So $Ax$ is a **linear combination of the columns of $A$**, with the entries of $x$ as the coefficients. Applying this to the standard basis vector $e_j$ (a 1 in position $j$, zeros elsewhere) gives $Ae_j=a^{(j)}$: **the $j$-th column of $A$ is the image of $e_j$**. That one fact lets you draw any $2\times2$ transformation.

## Derivation: $(AB)^\top=B^\top A^\top$

The $(i,j)$ entry of $(AB)^\top$ is the $(j,i)$ entry of $AB$, namely $\sum_k a_{jk}b_{ki}$. The $(i,j)$ entry of $B^\top A^\top$ is $\sum_k (B^\top)_{ik}(A^\top)_{kj}=\sum_k b_{ki}a_{jk}$. These are the same sum. $\blacksquare$

## Why $AB\neq BA$

Let $A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$ and $S=\begin{pmatrix}0&1\\1&0\end{pmatrix}$.

$$
AS=\begin{pmatrix}1\cdot0+2\cdot1&1\cdot1+2\cdot0\\3\cdot0+4\cdot1&3\cdot1+4\cdot0\end{pmatrix}=\begin{pmatrix}2&1\\4&3\end{pmatrix}\quad(\text{columns swapped}),
$$

$$
SA=\begin{pmatrix}0\cdot1+1\cdot3&0\cdot2+1\cdot4\\1\cdot1+0\cdot3&1\cdot2+0\cdot4\end{pmatrix}=\begin{pmatrix}3&4\\1&2\end{pmatrix}\quad(\text{rows swapped}).
$$

Right-multiplication acts on columns; left-multiplication acts on rows. Different operations, different results. Also note that if $m\ne p$ one of $AB$, $BA$ may not even be defined.

## Worked example: a rotation

Rotating the plane counter-clockwise by $\theta$ sends $e_1=(1,0)\mapsto(\cos\theta,\sin\theta)$ and $e_2=(0,1)\mapsto(-\sin\theta,\cos\theta)$. By the columns rule,

$$
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}.
$$

For $\theta=90^\circ$: $R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$, so $R(1,1)^\top=(-1,1)^\top$. Rotation preserves lengths: $\|Rx\|=\|x\|$.

A matrix-vector example: $\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}1+2\\3+4\end{pmatrix}=\begin{pmatrix}3\\7\end{pmatrix}=1\cdot\begin{pmatrix}1\\3\end{pmatrix}+1\cdot\begin{pmatrix}2\\4\end{pmatrix}$.

## Implementation

```python
import numpy as np

def matmul(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"inner dimensions differ: {n} vs {n2}"
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            C[i, j] = np.sum(A[i, :] * B[:, j])   # row i . column j
    return C
```

## Tensor shapes in a neural network

For a batch of $B$ examples with $n$ features each, stacked as rows of $X\in\mathbb{R}^{B\times n}$, and a weight matrix $W\in\mathbb{R}^{n\times m}$:

$$
XW\in\mathbb{R}^{B\times m}.
$$

Row $b$ of the result is $x_b^\top W$, the layer applied to example $b$. We write layers as $XW$ (not $Wx$) because frameworks store a batch as rows. Both conventions are the same math up to a transpose: $(XW)^\top=W^\top X^\top$.

## Common misconceptions

- **Matrix multiplication is not elementwise.** Elementwise multiplication (`A * B`) requires equal shapes and is a different operation.
- **Order matters.** $ABx$ means apply $B$ first, then $A$.
- **$AB=0$ does not imply $A=0$ or $B=0$.**
- **"Linear" does not mean "a line".** It means $A(\alpha x+\beta y)=\alpha Ax+\beta Ay$. Adding a bias $b$ makes the map *affine*, not linear.

## Key takeaways

- `(m x n)(n x p) = (m x p)`: the inner sizes must match and then vanish.
- Two views of `Ax`: rows give dot products; columns give a linear combination of columns.
- The columns of `A` are where the basis vectors land, so you can draw any 2x2 map from its columns.
- `AB != BA`, and `(AB)^T = B^T A^T`. A bias makes a map affine, not linear.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Draw where the unit square goes under `[[2,0],[0,3]]` and under a 90 degree rotation. How did the columns help?
2. Why does `X @ W` with `X: (B, n)` apply the layer to every example at once?

## Further reading

- *Mathematics for Machine Learning*, Sec. 2.2 (matrices) and 2.7 (linear mappings), PDF p.23-70.
- Source library: `ai-engineering-from-scratch` phase 01, lessons 02 and 03.

## Mastery criteria

Given three matrices with stated shapes you can say instantly which products exist and what shape they give, and you can sketch where the unit square goes under a $2\times2$ matrix using its columns.
