---
id: mx.01
title: Complex numbers for machine learning
summary: Numbers with a built-in rotation, Euler's formula derived from Taylor series, roots of unity, and why a rotation matrix has complex eigenvalues.
minutes: 60
teaches: [mx.complex, mx.euler, mx.roots-unity]
requires: [la.linear-map, la.eigen, alg.exp-log, calc.derivative]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/19-complex-numbers/docs/en.md (concept section: arithmetic, polar form, Euler's formula, roots of unity, links to DFT and RoPE; section list of Build It). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/docs/en.md (DFT definition, used only to see where complex numbers lead next). READ."
---

## Where this fits

Sooner or later a paper shows you an `i`. The Fourier transform ([[mx.02]]) is written with it, rotary position embeddings (RoPE) in transformers are described as rotating complex numbers, and the eigenvalues of some perfectly real matrices turn out to be complex. In [[math.07]] you saw that a 90-degree rotation of the plane has no real eigenvector. This lesson explains what its eigenvalues are and why.

You do not need complex numbers to do ML day to day, but they are the cleanest language for **anything that rotates or oscillates**. After this lesson you can read `e^{i\theta}` as "the point at angle theta on the unit circle" and see the Fourier transform as a change of basis rather than a magic trick.

## The idea in plain words

- Real numbers live on a line. Multiplying by $-1$ flips the line around zero: it is a half-turn. Now ask: is there a number whose square is $-1$, that is, a "quarter-turn" which done twice gives a half-turn? On a line there is none, so mathematicians add a second direction. The new unit pointing "up" is called $i$, and numbers become **points in a plane**.
- The name "imaginary" is a historical accident. Nothing about $i$ is less real than a negative number; it is simply "turn a quarter circle".
- A point in a plane can be described by two walks (right and up) or by one walk in a direction: "go this far, at this angle". The first is the Cartesian description, the second the polar one. Adding is easy with the first; multiplying is easy with the second, because **multiplying by a complex number means turning and stretching**.
- A number of length 1 only turns. The numbers of length 1 form the unit circle, and Euler's formula is the statement that the exponential function, fed with a pure turn angle, walks around that circle.
- If you split a circle into $n$ equal slices starting from the point 1, the slice boundaries are the **roots of unity**. They are the sample frequencies of the Fourier transform.

## Definitions

The **imaginary unit** $i$ is defined by $i^2=-1$. A **complex number** is $z=a+bi$ with real $a$ (real part) and $b$ (imaginary part). The set of all of them is $\mathbb{C}$, identified with the plane $\mathbb{R}^2$ via $z\leftrightarrow(a,b)$.

Arithmetic follows ordinary algebra plus $i^2=-1$:

$$
(a+bi)+(c+di)=(a+c)+(b+d)i,
$$

$$
(a+bi)(c+di)=ac+adi+bci+bd\,i^2=(ac-bd)+(ad+bc)i .
$$

The **conjugate** of $z=a+bi$ is $\bar z=a-bi$ (mirror image across the real axis). The product $z\bar z=a^2+b^2$ is real and non-negative, which defines the **modulus** $|z|=\sqrt{a^2+b^2}$ (distance from the origin). Division uses it: $\dfrac{1}{z}=\dfrac{\bar z}{|z|^2}$.

The **argument** (angle) is $\theta=\operatorname{atan2}(b,a)$, so $z=r(\cos\theta+i\sin\theta)$ with $r=|z|$. This is the **polar form**.

### A matrix hiding inside

Write $z=a+bi$ as the matrix

$$
M_z=\begin{pmatrix}a&-b\\ b&a\end{pmatrix}\qquad(\text{shape }2\times2).
$$

Multiplying $M_z$ by the column $(c,d)^\top$ gives $(ac-bd,\ bc+ad)^\top$, exactly the product $z\cdot(c+di)$. So complex multiplication **is** a particular family of $2\times2$ matrices: rotation matrices times a scale. With $r=|z|$ and $a=r\cos\theta$, $b=r\sin\theta$:

$$
M_z=r\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}.
$$

## Derivation: Euler's formula from Taylor series

We want a meaning for $e^{i\theta}$. The exponential has the series (which converges for every real or complex $x$)

$$
e^{x}=\sum_{k=0}^{\infty}\frac{x^k}{k!}=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots
$$

Use this series as the **definition** for $x=i\theta$. The powers of $i$ cycle with period 4: $i^0=1,\ i^1=i,\ i^2=-1,\ i^3=-i,\ i^4=1,\dots$ Hence

$$
e^{i\theta}=1+i\theta-\frac{\theta^2}{2!}-i\frac{\theta^3}{3!}+\frac{\theta^4}{4!}+i\frac{\theta^5}{5!}-\cdots
$$

Group the terms without an $i$ (even powers) and with an $i$ (odd powers):

$$
e^{i\theta}=\underbrace{\Big(1-\frac{\theta^2}{2!}+\frac{\theta^4}{4!}-\cdots\Big)}_{\cos\theta}+i\underbrace{\Big(\theta-\frac{\theta^3}{3!}+\frac{\theta^5}{5!}-\cdots\Big)}_{\sin\theta}.
$$

The two brackets are the Taylor series of cosine and sine (the cosine series has only even powers with alternating signs; the sine series only odd ones). Therefore

$$
\boxed{e^{i\theta}=\cos\theta+i\sin\theta.}
$$

Consequences:

- $|e^{i\theta}|^2=\cos^2\theta+\sin^2\theta=1$: it always lies on the unit circle.
- $e^{i\pi}=-1$, i.e. $e^{i\pi}+1=0$.
- A general complex number is $z=re^{i\theta}$, and the product rule is just the exponent rule $e^{x}e^{y}=e^{x+y}$:

$$
r_1e^{i\theta_1}\cdot r_2e^{i\theta_2}=r_1r_2\,e^{i(\theta_1+\theta_2)}.
$$

**Multiply the lengths, add the angles.** Multiplying by $e^{i\varphi}$ rotates every point by $\varphi$; multiplying by a real $r>0$ scales it. Combined with the matrix view, this explains why $M_{e^{i\varphi}}$ is a rotation matrix.

Reading the product's real and imaginary parts also proves the angle-addition identities for free: $\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$ and so on.

## Roots of unity

The solutions of $z^n=1$ are the **$n$-th roots of unity**. Write $z=e^{i\theta}$; then $e^{in\theta}=1$ requires $n\theta=2\pi k$, so

$$
\omega_k=e^{2\pi i k/n},\qquad k=0,1,\dots,n-1.
$$

They are the vertices of a regular $n$-gon on the unit circle. Writing $\omega=\omega_1$ we have $\omega_k=\omega^k$. Two facts you will use in [[mx.02]]:

1. **They sum to zero** for $n\ge2$: $\sum_{k=0}^{n-1}\omega^k=\dfrac{1-\omega^n}{1-\omega}=0$ (geometric series, with $\omega^n=1$ and $\omega\ne1$). Geometrically the polygon is balanced around the origin.
2. **Orthogonality.** For $\omega^j$, the sum $\sum_{n}\omega^{jn}$ equals $n$ if $j\equiv0 \pmod n$ and $0$ otherwise. The vectors $(\omega^{jm})_{m}$ for different $j$ are therefore orthogonal (in the complex inner product, which conjugates one factor).

## Complex eigenvalues of rotations

Take the rotation matrix $R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}$. From [[math.07]], eigenvalues solve $\lambda^2-\operatorname{tr}(R)\lambda+\det R=0$. Here $\operatorname{tr}=2\cos\theta$ and $\det=\cos^2\theta+\sin^2\theta=1$:

$$
\lambda^2-2\cos\theta\,\lambda+1=0\ \Longrightarrow\ \lambda=\cos\theta\pm\sqrt{\cos^2\theta-1}=\cos\theta\pm i\sin\theta=e^{\pm i\theta}.
$$

For $\theta=90^\circ$ the eigenvalues are $\pm i$. This is consistent with the story: no real direction survives a turn (so no real eigenvector), but over the complex numbers there is one, with an eigenvalue that is itself "a rotation by $\theta$". The corresponding eigenvectors are $(1,\mp i)^\top$ (up to scale). Two lessons for ML:

- **Eigenvalues of modulus 1 mean pure rotation; modulus above or below 1 means spiral out or in.** For a recurrent network $h_{t+1}=Wh_t$, eigenvalues of $W$ with $|\lambda|>1$ explode and $|\lambda|<1$ vanish. A complex pair with $|\lambda|=1$ oscillates forever.
- Real matrices with complex eigenvalues always have them in conjugate pairs $\lambda,\bar\lambda$ (the characteristic polynomial has real coefficients).

## Worked example

All of these numbers come from the exercise code.

1. $z=3+4i$: $|z|=5$, $\theta=\operatorname{atan2}(4,3)=0.9273$ rad $=53.13^\circ$.
2. $(1+2i)(3-i)=3-i+6i-2i^2=3+5i+2=5+5i$. Moduli: $|1+2i|\,|3-i|=\sqrt5\sqrt{10}=\sqrt{50}=7.0711$, and $|5+5i|=\sqrt{50}$ as well. The angles add: $\operatorname{atan2}(2,1)+\operatorname{atan2}(-1,3)=1.1071-0.3217=0.7854=45^\circ$, the angle of $5+5i$.
3. Rotate $1+i$ by $90^\circ$: multiply by $i$, getting $i+i^2=-1+i$. A quarter turn of the point $(1,1)$ is $(-1,1)$. ✓
4. Taylor partial sums of $e^{i\pi/3}=0.5+0.8660i$. With $T$ terms the error is $0.5318$ ($T=2$), $0.0494$ ($T=4$), $0.0018$ ($T=6$) and $4\times10^{-7}$ ($T=10$). The factorial in the denominator makes the series converge very quickly.
5. Fourth roots of unity: $1,\ i,\ -1,\ -i$. Sixth roots: $1,\ 0.5\pm0.866i,\ -0.5\pm0.866i,\ -1$.
6. $R_{60^\circ}$ has eigenvalues $0.5\pm0.866i=e^{\pm i\pi/3}$; $R_{90^\circ}$ has $\pm i$.

## Implementation

```python
import numpy as np

def to_polar(z):
    return abs(z), np.arctan2(z.imag, z.real)

def euler_series(theta, terms):
    # partial sum of the Taylor series of exp(i*theta)
    total, term = 0j, 1 + 0j
    for k in range(terms):
        total += term
        term = term * (1j * theta) / (k + 1)
    return total

def roots_of_unity(n):
    return np.exp(2j * np.pi * np.arange(n) / n)

def rotation_eigenvalues(theta):
    tr, det = 2 * np.cos(theta), 1.0
    disc = np.sqrt(complex(tr * tr - 4 * det))
    return (tr + disc) / 2, (tr - disc) / 2
```

In practice NumPy has native complex dtypes (`complex128`), `np.abs`, `np.angle`, `np.exp`, and `np.linalg.eig` returns complex eigenvalues when needed. PyTorch supports complex tensors (`torch.view_as_real` converts to a trailing dimension of size 2), which is how RoPE implementations often rotate pairs of coordinates.

## Shapes

- A complex scalar is two reals. A complex vector of length $n$ is $2n$ reals.
- $\mathbb{C}^n$ vectors use the **conjugate transpose** $x^{H}=\bar x^\top$ for inner products: $\langle x,y\rangle=x^{H}y$. Plain transposes give the wrong lengths ($z^2$ can be negative, $\bar zz$ cannot).
- A rotation by $\theta$ acting on $n/2$ pairs of coordinates of a length-$n$ real vector is a complex multiplication of an $n/2$-vector by $e^{i\theta_j}$ componentwise.

## Common misconceptions

- "Imaginary numbers are not real." They are a coordinate on a plane. Everything here can be done with $2\times2$ matrices.
- "$\sqrt{ab}=\sqrt a\sqrt b$ always." It fails for negative numbers: $\sqrt{(-1)(-1)}=1$ but $\sqrt{-1}\sqrt{-1}=-1$.
- "The angle is unique." The argument is defined only modulo $2\pi$; $\operatorname{atan2}$ picks one in $(-\pi,\pi]$.
- "$|z|^2=z^2$." The correct identity is $|z|^2=z\bar z$.
- "A real matrix has real eigenvalues." Only guaranteed for symmetric matrices ([[math.07]]).

## Key takeaways

- $z=a+bi=re^{i\theta}$: Cartesian for adding, polar for multiplying.
- Multiplication adds angles and multiplies lengths, so unit-modulus numbers are rotations.
- $e^{i\theta}=\cos\theta+i\sin\theta$ follows from the exponential series by splitting even and odd powers.
- The $n$-th roots of unity are $e^{2\pi ik/n}$; they sum to zero and are mutually orthogonal as vectors.
- A real rotation by $\theta$ has eigenvalues $e^{\pm i\theta}$.

## Explain it yourself

1. Explain to a friend why multiplying by $i$ is a quarter turn, and why that makes $i^2=-1$ unavoidable.
2. Explain why the Taylor series of $e^{i\theta}$ splits into cosine and sine, and what that says about circles.

## Further reading

- Repo lesson `phases/01-math-foundations/19-complex-numbers` (connections to RoPE and sinusoidal positional encodings).
- Next: [[mx.02]] uses roots of unity as the basis of the Fourier transform.

## Mastery criteria

- [ ] I can convert between $a+bi$ and $re^{i\theta}$ and multiply in either form.
- [ ] I can derive Euler's formula without notes.
- [ ] I can list the $n$-th roots of unity and explain why they sum to zero.
- [ ] I can compute the eigenvalues of a $2\times2$ rotation and interpret their modulus.
