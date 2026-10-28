---
id: mf.05
title: "Trigonometry and the unit circle"
summary: Angles and radians, sine, cosine and tangent from the unit circle, the identities you will reuse (including rotation matrices), and periodic waves, the doorway to Fourier analysis.
minutes: 95
teaches: [mf.trig]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/03-matrix-transformations/docs/en.md (The Problem, 'Transformations as matrices', and the Rotation section): rotation matrix R(theta) and its role in ML. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/docs/en.md (The Problem, DFT definition and coefficient meanings, window formulas, connection to positional encodings): frequency, amplitude, phase and the sinusoidal position encoding formula. READ."
---

## Where this fits

Anything that rotates, repeats or oscillates uses trigonometry. In machine learning that includes: rotating images and point clouds for data augmentation, rotation matrices inside geometric models, **sinusoidal positional encodings** that tell a transformer where each word sits, and the **Fourier transform** that analyses audio, images and time series into waves. Trigonometry looks like a pile of triangle formulas in school; here we reorganise it around one picture, the unit circle, and then every formula becomes something you can see.

This lesson needs no earlier lesson. You use function ideas from `mf.03` (period, shifting and scaling a graph) and the code habits of `mf.01`, but each is re-explained when needed.

## The idea in plain words

Imagine a ferris wheel of radius 1 turning at constant speed, and a lamp far above casting its shadow. As the wheel turns, a rider's **height** goes up, reaches the top, comes down, goes below the centre, and returns: a smooth wave. The rider's **horizontal position** does the same wave, but a quarter-turn behind. Those two waves are *sine* and *cosine*. Everything else is bookkeeping about where the rider is.

- An **angle** says how far around the wheel the rider has gone. We measure it either in degrees (a full turn is 360) or in **radians** (a full turn is $2\pi$): the radian is the *distance travelled along the rim*, which is the natural unit because on a circle of radius 1 the distance equals the angle.
- **Cosine** of the angle: how far right the rider is. **Sine**: how high.
- **Tangent**: how steep the line from the centre to the rider is (height divided by right-distance).
- **Periodic**: after a full turn the rider is back where they started, so the waves repeat forever.
- **Rotation**: if you turn the whole picture by some angle, a point at coordinates $(x,y)$ moves to a point whose new coordinates are mixtures of sine and cosine of that angle.

## Angles and radians

A full circle is $360^\circ$ or $2\pi$ radians. The circumference of a circle of radius $r$ is $2\pi r$; on the **unit circle** ($r=1$) a trip of arc length $\theta$ corresponds to angle $\theta$ radians. Conversions:

$$
\text{radians}=\text{degrees}\times\frac{\pi}{180},\qquad \text{degrees}=\text{radians}\times\frac{180}{\pi}.
$$

Landmarks: $30^\circ=\pi/6$, $45^\circ=\pi/4$, $60^\circ=\pi/3$, $90^\circ=\pi/2$, $180^\circ=\pi$, $270^\circ=3\pi/2$, $360^\circ=2\pi$. Angles are measured counter-clockwise from the positive horizontal axis; negative angles go clockwise. Angles that differ by $2\pi$ describe the same direction.

Why radians in mathematics and code? Formulas such as the slope of $\sin$ become clean only in radians, and every library (`np.sin`, `math.cos`) expects radians. Forgetting this is the most common trig bug: `np.sin(90)` is not 1, because it means 90 radians.

## Sine, cosine, tangent

**Unit-circle definition.** For an angle $\theta$, let $(x,y)$ be the point on the unit circle at that angle. Then

$$
\cos\theta=x,\qquad \sin\theta=y,\qquad \tan\theta=\frac{y}{x}=\frac{\sin\theta}{\cos\theta}.
$$

**Right-triangle view.** For an acute angle in a right triangle: $\sin=\frac{\text{opposite}}{\text{hypotenuse}}$, $\cos=\frac{\text{adjacent}}{\text{hypotenuse}}$, $\tan=\frac{\text{opposite}}{\text{adjacent}}$. It agrees with the circle because a triangle with hypotenuse 1 inscribed in the circle has adjacent side $x$ and opposite side $y$.

**Values worth knowing.**

| $\theta$ | 0 | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ |
|---|---|---|---|---|---|
| $\sin\theta$ | 0 | $1/2$ | $\sqrt2/2\approx0.707$ | $\sqrt3/2\approx0.866$ | 1 |
| $\cos\theta$ | 1 | $\sqrt3/2$ | $\sqrt2/2$ | $1/2$ | 0 |

Why $\sin(\pi/6)=\frac12$? Take an equilateral triangle with side 1 and cut it in half: you get a right triangle with angles $30^\circ,60^\circ,90^\circ$, hypotenuse 1, and short side $\frac12$, opposite the $30^\circ$ angle. The other values come from the $45^\circ$ right isosceles triangle (both legs equal, so $2x^2=1$, $x=\sqrt2/2$).

**Signs by quadrant.** Right/up is positive. In the second quadrant ($\pi/2$ to $\pi$) the point is left and up: $\cos<0,\sin>0$. In the third both are negative; in the fourth $\cos>0,\sin<0$. Symmetries follow from mirroring: $\sin(-\theta)=-\sin\theta$ (odd), $\cos(-\theta)=\cos\theta$ (even), $\sin(\theta+\pi)=-\sin\theta$.

**Tangent** is undefined where $\cos\theta=0$ ($\theta=\pi/2+k\pi$), where the radius is vertical and its slope infinite.

## The Pythagorean identity

The point $(\cos\theta,\sin\theta)$ lies on the unit circle $x^2+y^2=1$. Therefore, for every angle,

$$
\sin^2\theta+\cos^2\theta=1 .
$$

Use: if you know $\cos\theta=0.6$ and $\theta$ is in the first quadrant, then $\sin\theta=\sqrt{1-0.36}=0.8$. Check numerically with code below.

## Angle addition and rotation matrices

Rotating a point by $\theta$ counter-clockwise about the origin. Start with the two basis points $(1,0)$ and $(0,1)$:

- $(1,0)$ is at angle 0, rotated by $\theta$ it is at angle $\theta$ on the unit circle: $(\cos\theta,\sin\theta)$.
- $(0,1)$ is at angle $\pi/2$; rotated by $\theta$ it sits at angle $\theta+\pi/2$, which is the point $(-\sin\theta,\cos\theta)$ (the previous point turned a further quarter-turn).

A rotation treats a general point $(x,y)=x(1,0)+y(0,1)$ linearly, so its image is $x(\cos\theta,\sin\theta)+y(-\sin\theta,\cos\theta)$. In matrix notation:

$$
R(\theta)\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}x\cos\theta-y\sin\theta\\ x\sin\theta+y\cos\theta\end{pmatrix}.
$$

Check: $\theta=\pi/2$ sends $(1,0)$ to $(0,1)$ and $(0,2)$ to $(-2,0)$. Rotation keeps distance from the origin (squares: $(x\cos\theta-y\sin\theta)^2+(x\sin\theta+y\cos\theta)^2=x^2+y^2$ by the Pythagorean identity).

**Angle addition, derived.** Rotating by $b$ and then by $a$ must equal rotating by $a+b$. Apply the matrices to $(1,0)$: rotating by $b$ gives $(\cos b,\sin b)$, then rotating by $a$ gives first coordinate $\cos a\cos b-\sin a\sin b$ and second coordinate $\sin a\cos b+\cos a\sin b$. But rotating $(1,0)$ by $a+b$ gives $(\cos(a+b),\sin(a+b))$. Equate:

$$
\cos(a+b)=\cos a\cos b-\sin a\sin b,\qquad \sin(a+b)=\sin a\cos b+\cos a\sin b .
$$

Numeric check with $a=\pi/4$, $b=\pi/6$: $\sin(75^\circ)=0.9659258$ on both sides. The matrix view is the one that matters for ML: rotations compose by multiplying matrices. Positions in some language models are encoded by rotating pairs of coordinates, and the angle-addition identity is why relative offsets then fall out cleanly.

## Periodic functions and waves

$\sin$ and $\cos$ repeat every $2\pi$: $\sin(\theta+2\pi)=\sin\theta$. A general wave is

$$
y(t)=A\sin(2\pi ft+\varphi).
$$

- **Amplitude** $A$: the height of the peaks.
- **Frequency** $f$: cycles per unit of time. The **period** is $T=1/f$, since the sine argument advances by $2\pi$ when $t$ advances by $1/f$.
- **Phase** $\varphi$: shifts the wave sideways (a shift of the graph, as in `mf.03`; the wave is delayed by $\varphi/(2\pi f)$ time units).

So $\sin(3x)$ has period $2\pi/3$, and $\sin(2\pi\cdot5t)$ completes five cycles per unit time. Also $\cos\theta=\sin(\theta+\pi/2)$: cosine is a sine shifted by a quarter period.

## Why Fourier analysis exists

The big idea (developed in a later lesson): any reasonable periodic signal is a sum of such waves of different frequencies. To find how much of frequency $k$ is hidden in $N$ samples $x_n$ over one unit of time, correlate the signal with a cosine and a sine of that frequency: if the signal contains a wave of that frequency, the products line up and the sums are large, otherwise positive and negative parts cancel. For a pure wave $A\sin(2\pi kn/N)$ the amplitude estimate $\frac2N\sqrt{(\sum x_n\cos)^2+(\sum x_n\sin)^2}$ equals $A$. This is the discrete Fourier transform in disguise (the repository material expresses it with complex numbers $e^{-2\pi ikn/N}$, which package a cosine and a sine together).

A related use: the original transformer encodes position $pos$ with the pairs $\sin(pos/10000^{2i/d})$ and $\cos(pos/10000^{2i/d})$: many sine/cosine pairs at different frequencies, so that every position has a distinct fingerprint and nearby positions have similar ones.

## Worked example in code

Every number is reproduced by the exercise functions.

```python
import numpy as np

print(np.sin(np.pi / 6), np.cos(np.pi / 6), np.tan(np.pi / 6))
# 0.49999999999999994 0.8660254037844387 0.5773502691896257   (float rounding: 0.5, sqrt(3)/2, 1/sqrt(3))

def R(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

print(R(np.pi / 2) @ np.array([1, 0]))     # [6.1e-17 1.0]  i.e. (0, 1) up to rounding
print(R(np.pi / 4) @ np.array([1, 0]))     # [0.7071 0.7071]
print(np.hypot(3, 4), np.arctan2(4, 3))    # 5.0 0.9272952180016122

N = 256
t = np.arange(N) / N
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
amp = lambda k: 2 / N * np.hypot(x @ np.cos(2*np.pi*k*t), x @ np.sin(2*np.pi*k*t))
print(round(amp(5), 6), round(amp(3), 6), round(amp(4), 6))   # 1.0 0.5 0.0
```

The "6.1e-17" is floating-point noise for a true 0; compare with `np.allclose`, never `==`.

## Common misconceptions

- "`np.sin(90)` is 1." The inputs are radians. Convert: `np.sin(np.deg2rad(90))`.
- "$\sin(a+b)=\sin a+\sin b$." False; use the addition formula.
- "$\sin^2\theta$ means $\sin(\theta^2)$." It means $(\sin\theta)^2$.
- "$\sin^{-1}$ is $1/\sin$." The notation $\sin^{-1}$ (also called arcsin) is the inverse function, which returns an angle; $1/\sin\theta$ is called cosecant.
- "$\tan$ is bounded." It blows up at $\pi/2+k\pi$.
- "Rotation by $\theta$ changes lengths." It does not; it is an isometry.
- Mixing up the order of the rotation matrix entries: the minus sign sits in the top-right.

## Key takeaways

- Radians measure arc length on the unit circle; $2\pi$ is a full turn.
- $(\cos\theta,\sin\theta)$ is the point on the unit circle; hence $\sin^2+\cos^2=1$ and $\tan=\sin/\cos$.
- Rotation matrices preserve length and compose by adding angles; the addition formulas are that statement written out.
- $A\sin(2\pi ft+\varphi)$ has amplitude $A$, frequency $f$, period $1/f$ and phase $\varphi$.
- Fourier analysis asks how much of each frequency a signal contains; sinusoidal positional encodings use many frequencies at once.

## Explain it yourself

1. Why must $\sin^2\theta+\cos^2\theta=1$? Argue from the unit circle only.
2. Explain why rotating by $b$ then by $a$ is the same as rotating by $a+b$, and how that gives the formula for $\sin(a+b)$.

## Further reading

- Later lessons: matrix transformations (rotations, scalings), complex numbers and the Fourier transform, and transformer positional encodings.

## Mastery criteria

You can: convert between degrees and radians; read $\sin$, $\cos$, $\tan$ off the unit circle for the standard angles including signs; use $\sin^2+\cos^2=1$; build a $2\times2$ rotation matrix and apply it; state period, amplitude, frequency and phase of $A\sin(2\pi ft+\varphi)$; explain in words what a Fourier coefficient measures.
