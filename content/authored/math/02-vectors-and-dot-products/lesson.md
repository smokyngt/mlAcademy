---
id: f2n.02
title: Vectors, norms and the dot product
summary: Vectors as arrows and as lists, and the dot product that every neuron computes.
minutes: 45
teaches: [la.vector, la.dot]
requires: [alg.functions, alg.summation]
sources:
  - math/math for ML.pdf (Ch. 2 Linear Algebra, PDF p.23; Ch. 3 Analytic Geometry: norms, inner products, angles, projections, PDF p.76)
  - ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md
  - ai-engineering-from-scratch/phases/01-math-foundations/14-norms-and-distances/docs/en.md
---

## Learning objectives

- Add vectors, scale them, and form linear combinations.
- Compute the dot product two ways (sum of products; $\|a\|\|b\|\cos\theta$) and derive why they agree.
- Compute norms, cosine similarity and the projection of one vector onto another.
- Recognise a neuron's pre-activation $w\cdot x + b$ as a dot product.

## Prerequisites

Function notation and summation ([[f2n.01]]).

## Where this fits

A model can only compare things if they are described by numbers. A house is (surface, rooms, age); a song is a few hundred audio measurements; a word is a list of 768 numbers. Such a list is a **vector**. Almost every question a model asks is "how similar are these two lists?" (is this email like the spam I have seen? is this word close to that word?), and the tool that answers it is the **dot product**.

You will meet it again immediately: a single artificial neuron is nothing more than a dot product followed by a small adjustment. Understanding it now means lesson 9 (layers) will feel like a repeat, not a new idea.

## The idea in plain words

- A vector is an **arrow** on a map: how far east, how far north. With more directions than two or three you cannot draw it, but the idea (a list of numbers = an arrow) still holds.
- The **length** of an arrow is the Pythagoras you already know, extended to more directions.
- The **dot product** answers "how much do these two arrows point the same way?". Picture two people walking: if they walk in the same direction the number is big and positive, if they walk at right angles it is zero, if they walk in opposite directions it is negative. The longer the walks, the bigger the number.
- A **projection** is a shadow: hold a stick (arrow `a`) under a lamp directly above a line (arrow `b`) and look at the shadow it casts on that line. The shadow is the part of `a` that goes along `b`; what is left over is perpendicular to it.

Try the figure below before reading the formulas: drag the sliders and watch the dot product change sign as the arrows turn.

## Intuition

A vector in $\mathbb{R}^n$ is an ordered list of $n$ numbers. Geometrically it is an arrow from the origin; in data terms it is *one example described by $n$ features* (a house: area, rooms, age) or *one set of parameters*.

The **dot product** measures *how much two vectors point the same way*. It is large and positive when they align, zero when they are perpendicular, negative when they oppose. A neuron computes exactly this between its weight vector and the input: "how much does the input look like the pattern I am tuned to?"

Try it below. Drag the sliders and watch the dot product follow the angle.

```viz
vector-playground
```

## Formal definitions

For $a,b\in\mathbb{R}^n$ with components $a_i,b_i$:

- **Addition and scaling** are componentwise: $(a+b)_i=a_i+b_i$, $(\lambda a)_i=\lambda a_i$.
- **Linear combination:** $\lambda_1 v_1+\cdots+\lambda_k v_k$.
- **Dot product:** $a\cdot b = a^\top b=\sum_{i=1}^{n} a_ib_i \in\mathbb{R}$.
- **Norm (Euclidean length):** $\|a\|=\sqrt{a\cdot a}=\sqrt{\sum_i a_i^2}$.
- **Distance:** $\|a-b\|$.
- **Cosine similarity:** $\cos\theta = \dfrac{a\cdot b}{\|a\|\,\|b\|}$ (defined when neither is the zero vector).
- **Orthogonal:** $a\cdot b=0$.

## Derivation: why $a\cdot b=\|a\|\|b\|\cos\theta$

Place $a$ and $b$ as two sides of a triangle; the third side is $a-b$ and the angle between $a$ and $b$ is $\theta$. The law of cosines says

$$
\|a-b\|^2=\|a\|^2+\|b\|^2-2\|a\|\|b\|\cos\theta. \tag{1}
$$

Now expand the left side using only the definition of the dot product, which is bilinear and symmetric:

$$
\|a-b\|^2=(a-b)\cdot(a-b)=a\cdot a-2\,a\cdot b+b\cdot b=\|a\|^2-2\,a\cdot b+\|b\|^2. \tag{2}
$$

Equating (1) and (2) and cancelling $\|a\|^2+\|b\|^2$ leaves $-2\,a\cdot b=-2\|a\|\|b\|\cos\theta$, i.e. $a\cdot b=\|a\|\|b\|\cos\theta$. $\blacksquare$

Because $|\cos\theta|\le1$ we immediately get the **Cauchy–Schwarz inequality** $|a\cdot b|\le\|a\|\|b\|$, with equality exactly when one vector is a multiple of the other.

**Projection.** The component of $a$ along $b$ is a multiple $\lambda b$ such that the leftover $a-\lambda b$ is orthogonal to $b$:

$$
(a-\lambda b)\cdot b=0\;\Rightarrow\;\lambda=\frac{a\cdot b}{b\cdot b},\qquad \operatorname{proj}_b(a)=\frac{a\cdot b}{b\cdot b}\,b .
$$

## Worked examples

1. $a=(1,2,3)$, $b=(4,5,6)$. Dot: $1\cdot4+2\cdot5+3\cdot6=4+10+18=32$. Norms: $\|a\|=\sqrt{14}\approx3.742$, $\|b\|=\sqrt{77}\approx8.775$. So $\cos\theta=32/(3.742\times8.775)\approx0.9746$ and $\theta\approx12.9^\circ$.
2. $a=(3,4)$, $b=(4,-3)$: $a\cdot b=12-12=0$, so they are orthogonal. And $\|a\|=5$ (the 3-4-5 triangle).
3. Project $a=(2,3)$ onto $b=(1,0)$: $\lambda=(2)/(1)=2$, projection $=(2,0)$, leftover $(0,3)\perp b$.

## Implementation

```python
import numpy as np

a, b = np.array([1., 2., 3.]), np.array([4., 5., 6.])
dot = np.sum(a * b)                      # same as a @ b
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
proj = (a @ b) / (b @ b) * b             # projection of a onto b
```

## Shapes

| Expression | Shapes | Result |
|---|---|---|
| `a @ b` | $(n,)\cdot(n,)$ | scalar |
| `a * b` | $(n,)\odot(n,)$ | $(n,)$ elementwise, **not** the dot product |
| `W @ x + b` (neuron) | weights $(n,)$, input $(n,)$, bias scalar | scalar pre-activation |

## Common misconceptions

- **`*` is not the dot product.** `a * b` keeps $n$ numbers; `np.sum(a*b)` is the dot product.
- **Cosine similarity ignores length.** $(1,1)$ and $(100,100)$ have similarity $1$ but very different distance. Pick the measure that matches your question.
- **A dot product of 0 does not mean either vector is zero**, only that they are perpendicular.
- **In high dimensions random vectors are nearly orthogonal**, so "nearly orthogonal" is weak evidence of dissimilarity.

## Key takeaways

- The dot product is a sum of products and also `|a||b|cos(theta)`; the law of cosines proves they are the same.
- Orthogonal means dot product zero; cosine similarity looks at direction only.
- Projection of `a` on `b` is `(a.b / b.b) b`, and the leftover is perpendicular to `b`.
- A neuron's pre-activation `w.x + b` is a dot product: it measures alignment with the weights.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Without formulas: why is the dot product big when two arrows point the same way?
2. Two vectors have cosine similarity 1 but are far apart in Euclidean distance. Give an example.

## Further reading

- *Mathematics for Machine Learning*, Ch. 3 (norms, inner products, angles, orthogonal projections), PDF p.76.
- Source library lessons on linear algebra intuition and norms (`ai-engineering-from-scratch`, phase 01, lessons 01 and 14).

## Mastery criteria

Without a calculator you can compute a dot product and norm for small integer vectors, decide whether two vectors are orthogonal, and explain why a neuron's output is large when the input aligns with its weights.
