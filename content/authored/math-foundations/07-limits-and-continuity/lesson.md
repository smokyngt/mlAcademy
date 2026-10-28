---
id: mf.07
title: "Limits and continuity"
summary: Build the idea of a limit from numbers approaching a target, add one-sided limits, limit laws and continuity, meet the epsilon-delta definition gently, and see why derivatives are limits of slopes.
minutes: 100
teaches: [mf.limits]
requires: []
sources:
  - "math/math for ML.pdf, Ch. 5 intro and Sec. 5.1 (PDF p.145-149): function notation, the difference quotient as the slope of a secant, the derivative defined as the limit as h -> 0, and the worked derivation of the derivative of x^n by shrinking terms. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (The Problem, 'What is a derivative?' and its limit definition; numerical derivative with a small h): why derivatives matter for learning and how a small h replaces the limit in code. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md (floating-point precision and catastrophic cancellation discussion): why numerical limits cannot take h too small. READ."
---

## Where this fits

Training a model means nudging parameters in the direction that lowers a loss. "Which direction?" is answered by the **derivative**, the slope of the loss at the current point. But the slope of a curve at a *single* point sounds impossible: a slope needs two points, and one point is only one. The resolution is the idea of a **limit**: take two points, pull them together, and ask what the slope *approaches*. Limits are the foundation under derivatives, integrals, convergent series (`mf.06`), and the definition of $e$ (`mf.04`).

This is the last lesson of the foundations stage. It needs only the function ideas of `mf.03` and the code habits of `mf.01`; the geometric-series example refers back to `mf.06`, but nothing else is assumed.

## The idea in plain words

A limit is about **where a journey is heading, not where it ends**. Walk toward a door in steps that always halve the remaining distance. You never "touch" the door in finitely many steps, yet it is perfectly clear where you are heading. The limit is the door.

Three pictures:

1. **A hole in a road.** The road has a missing plank at position 1. You cannot stand there, but every plank on both sides leads straight to it at the same height. The limit exists (it is that height) even though the value there does not.
2. **A cliff.** The road to the left of position 0 is at height $-1$, and to the right at height $+1$. Approaching from the left you head to $-1$, from the right to $+1$. The two directions disagree, so there is no single destination: the limit does not exist.
3. **A continuous road.** No missing planks and no cliffs: where you are heading is where you are. That is **continuity**: you can draw the graph without lifting the pen.

## Limits, intuitively

We write

$$
\lim_{x\to a}f(x)=L
$$

and say: *as $x$ gets close to $a$ (from either side, but not equal to $a$), $f(x)$ gets close to $L$, as close as we please.* The phrase "not equal to $a$" is deliberate: a limit ignores the value at $a$ itself.

### Tiny numeric explorations

**Example 1 (a hole).** $f(x)=\dfrac{x^2-1}{x-1}$ is undefined at $x=1$ (it would be $0/0$). Tabulate:

| $x$ | 0.9 | 0.99 | 0.999 | 1.001 | 1.01 | 1.1 |
|---|---|---|---|---|---|---|
| $f(x)$ | 1.9 | 1.99 | 1.999 | 2.001 | 2.01 | 2.1 |

Both sides head to 2. Algebra confirms: for $x\neq1$, $\frac{x^2-1}{x-1}=\frac{(x-1)(x+1)}{x-1}=x+1$, which tends to 2. So $\lim_{x\to1}f(x)=2$ although $f(1)$ does not exist.

**Example 2 (an important one).** $\dfrac{\sin x}{x}$ at $0$ (radians). At $x=0.1$: $0.99833$; at $0.01$: $0.9999833$; at $0.001$: $0.99999983$. The limit is $1$.

**Example 3 (a cliff).** $f(x)=|x|/x$ equals $-1$ for $x<0$ and $+1$ for $x>0$.

### One-sided limits

$\lim_{x\to a^-}f(x)$ approaches from the left ($x<a$), $\lim_{x\to a^+}f(x)$ from the right. The ordinary limit exists **exactly when both one-sided limits exist and are equal**. For $|x|/x$ at 0: left limit $-1$, right limit $+1$, so no limit. For $x^2$ at 2 both sides give 4.

### Limits at infinity and infinite limits

$\lim_{x\to\infty}\frac1x=0$: the bigger $x$ is, the smaller $\frac1x$. A graph that rises without bound near a point (like $\frac1{x^2}$ near 0) is said to have limit $\infty$, which is a way of describing *how* the limit fails to exist. Similarly, the partial sums of a convergent geometric series (previous lesson) are a sequence whose limit is the sum, and $e=\lim_{n\to\infty}(1+\frac1n)^n$.

## Limit laws

If $\lim_{x\to a}f=L$ and $\lim_{x\to a}g=M$ both exist, then

$$
\lim(f+g)=L+M,\quad \lim(f-g)=L-M,\quad \lim(cf)=cL,\quad \lim(fg)=LM,\quad \lim\frac fg=\frac LM\ (M\neq0).
$$

In words: you can compute limits piece by piece, then combine. Example: $\lim_{x\to2}(3x^2-x)$: the limit of $x$ is 2, so the limit of $x^2$ is 4, so the limit is $3\cdot4-2=10$, which is just plugging in. For continuous functions that is always allowed. The laws fail to help when you get $0/0$: that is the case needing algebra (cancel, factor) or a clever argument. Typical technique: factor and cancel, as in Example 1; or the **squeeze principle**: if $g\le f\le h$ near $a$ and $g$ and $h$ both tend to $L$, then $f$ tends to $L$ (this is how $\sin x/x\to1$ is proved; we only test it numerically here).

## Continuity

A function $f$ is **continuous at $a$** if three things hold:

1. $f(a)$ is defined;
2. $\lim_{x\to a}f(x)$ exists;
3. the two are equal: $\lim_{x\to a}f(x)=f(a)$.

Three ways to fail, one per condition:

| failure | example at the trouble point | name |
|---|---|---|
| value undefined, limit exists | $\frac{x^2-1}{x-1}$ at 1 | removable (a hole; fixing $f(1)=2$ repairs it) |
| sides disagree | $\lvert x\rvert/x$ at 0, or a step function | jump |
| blows up | $\frac1x$ at 0 | infinite |

A **corner** like $|x|$ at 0 is still continuous (no break), though it will turn out not to be smooth enough to have a derivative there; this is why ReLU, $\max(0,x)$, which is continuous with a corner, is fine as a neural-network activation. Familiar functions (polynomials, $e^x$, $\sin$, $\cos$; $\ln x$ for $x>0$) are continuous wherever they are defined, and sums, products and compositions of continuous functions are continuous. A continuous function also cannot skip values: if $f(0)<0<f(1)$, it crosses zero somewhere between (the intermediate value property), and that is why the bisection method of `mf.03` works.

## The epsilon-delta idea, gently

"As close as we please" needs to be turned into a precise challenge game.

- Someone picks a tolerance $\varepsilon>0$ (epsilon) for how close the output must be to $L$: the *target band*.
- You must answer with a $\delta>0$ (delta): how close the input must be to $a$ to guarantee the output lands in the band.

The limit exists with value $L$ if you can win **for every** $\varepsilon$:

$$
\forall\varepsilon>0\ \exists\delta>0:\quad 0<|x-a|<\delta\ \Longrightarrow\ |f(x)-L|<\varepsilon .
$$

(Read with `mf.02`: "for all epsilon there exists delta such that...".)

**Worked example.** $f(x)=2x+1$, $a=3$, $L=7$. Then $|f(x)-7|=|2x-6|=2|x-3|$. We need $2|x-3|<\varepsilon$, i.e. $|x-3|<\varepsilon/2$. So choose $\delta=\varepsilon/2$ and the game is won for any $\varepsilon$: for $\varepsilon=0.1$ any $\delta\le0.05$ works; for $\varepsilon=0.01$ any $\delta\le0.005$ works. Notice: a smaller band needs a smaller $\delta$, but there is always one that works. For a curve that bends violently (like $\sin(1/x)$ near 0) no $\delta$ works for small $\varepsilon$: the function keeps oscillating between $-1$ and $1$ and has no limit at 0.

You will rarely need to run this game yourself. Its value is that it makes "approaches" a statement that can be checked, which is why every later theorem about derivatives and integrals can be proved. The exercise lets a program play the game by sampling.

## Why limits are needed for derivatives

The slope of the straight line through $(x,f(x))$ and $(x+h,f(x+h))$ is the **difference quotient**

$$
\frac{f(x+h)-f(x)}{h}.
$$

Setting $h=0$ directly gives $\frac00$, which tells you nothing. Instead let $h$ shrink and watch the quotient. The derivative is its limit:

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
$$

For $f(x)=x^2$ at $x=3$: $\frac{(3+h)^2-9}{h}=\frac{6h+h^2}{h}=6+h$ (for $h\neq0$), and $6+h\to6$. Numerically: $h=1$ gives 7, $h=0.1$ gives 6.1, $h=0.001$ gives 6.001. The textbook performs the same step for $x^n$: expand $(x+h)^n$, cancel the $x^n$, divide by $h$, and every leftover term still containing $h$ vanishes in the limit, leaving $nx^{n-1}$. So: derivative $=$ limit of slopes, and the secant line becomes the tangent line.

Continuity is needed too: a function with a jump at $a$ has no sensible slope there. (Continuity is necessary, but not sufficient: $|x|$ is continuous at 0 yet has a corner.)

## Numeric limit exploration, and its trap

In code we cannot let $h$ reach zero, so we use "small enough" $h$ and look at the trend:

```python
import math

f = lambda x: (x * x - 1) / (x - 1)
print([f(1 + h) for h in (0.1, 0.01, 0.001)])   # [2.1, 2.009999999999999, 2.0009999999999177]
print([math.sin(h) / h for h in (0.1, 0.01, 0.001)])
# [0.9983341664682815, 0.9999833334166665, 0.9999998333333416]
print([((3 + h) ** 2 - 9) / h for h in (0.1, 0.01, 0.001)])
# [6.100000000000012, 6.009999999999849, 6.000999999999479]
```

The trap: do not make $h$ absurdly small. Floating-point numbers hold about 16 digits, and subtracting two nearly equal numbers destroys them (catastrophic cancellation):

```python
for h in (1e-4, 1e-6, 1e-8):
    print(h, (1 - math.cos(h)) / h ** 2)
# 0.0001 0.4999999969612645
# 1e-06 0.5000444502911705
# 1e-08 0.0
```

The true limit is $1/2$. At $h=10^{-8}$, `cos(h)` rounds to exactly 1.0 and the numerator becomes 0. Good practice: look at several moderate values of $h$, check that the trend is steady, and confirm with algebra when you can. This is also why numerical gradients in ML use $h$ around $10^{-5}$ to $10^{-7}$, not $10^{-15}$.

## Worked example

Follow all the pieces for $f(x)=\dfrac{x^2-1}{x-1}$ at $a=1$:

1. $f(1)$ is undefined, so $f$ is not continuous at 1.
2. One-sided values at $h=10^{-6}$: $f(1-h)\approx1.999999$, $f(1+h)\approx2.000001$; they agree within $10^{-4}$, so the limit is 2.
3. Define a repaired $\tilde f(x)=f(x)$ for $x\neq1$ and $\tilde f(1)=2$. Now value $=$ limit, so $\tilde f$ is continuous at 1.
4. If instead we defined $\tilde f(1)=5$, the value 5 would disagree with the limit 2: still discontinuous (a removable break).

The exercise implements `one_sided`, `limit_estimate`, `is_continuous_at`, the epsilon-delta search and the difference quotient, and tests every claim above.

## Common misconceptions

- "The limit is $f(a)$." Only when $f$ is continuous at $a$. The limit ignores the value at $a$.
- "If $f(a)$ is undefined, the limit cannot exist." Counter-example: the hole above.
- "A limit means the function never reaches the value." It may reach it (constants do); a limit is about the trend.
- "If the two one-sided values are close, the limit exists." They must be equal in the limit, not merely close at one step size.
- "Smaller $h$ is always better numerically." Beyond a point, rounding errors take over.
- "Continuous means differentiable." $|x|$ is a counter-example. (The reverse holds: differentiable implies continuous.)
- "$\infty$ is a number you can plug in." It describes unbounded behaviour; $\infty-\infty$ and $\infty/\infty$ are undefined.

## Key takeaways

- A limit is where the function is heading near $a$, ignoring the value at $a$; it exists when left and right agree.
- Limit laws let you compute limits piece by piece; for $0/0$ forms, simplify algebraically.
- Continuous at $a$: value defined, limit exists, and they are equal. Discontinuities are removable, jump or infinite.
- Epsilon-delta turns "close" into a game: for every tolerance $\varepsilon$ there must be a $\delta$.
- The derivative is the limit of the difference quotient: the slope of secants becoming a tangent.
- In code, explore limits with moderately small $h$; extremely small $h$ suffers rounding errors.

## Explain it yourself

1. Explain in your own words why the limit of $(x^2-1)/(x-1)$ at 1 exists even though the function is undefined there.
2. Why can we not compute the slope of a curve at one point simply by setting $h=0$ in the difference quotient, and what do we do instead?

## Further reading

- The reference textbook, Sec. 5.1: difference quotient, derivative, and the derivation of the derivative of $x^n$.
- Next stages of the academy: derivatives and gradients (calculus), then everything that trains a model. You now have the notation, the algebra, the functions, the exponentials and logarithms, the trigonometry, the sums and the limits they rest on.

## Mastery criteria

You can: estimate a limit from a table of values and confirm it algebraically; compute left and right limits and decide if the limit exists; use the limit laws; test the three conditions of continuity and classify a discontinuity; carry out an epsilon-delta argument for a linear function; write the derivative as a limit of difference quotients and approximate it with a small $h$ while explaining why $h$ should not be too small.
