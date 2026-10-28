---
id: math.15
title: Floating point and numerical stability
summary: Why a computer's real numbers are only approximations, which formulas break because of it, and the standard rewrites (log-sum-exp, Welford, compensated sums) that fix them.
minutes: 75
teaches: [num.floatingpoint, num.stability]
requires: [alg.exp-log, calc.numerical-diff, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md: IEEE 754 layout and format table (float64/float32/float16/bfloat16), 0.1+0.2, catastrophic cancellation, overflow/underflow limits of exp and log, log-sum-exp, mixed precision and loss scaling, bfloat16 vs float16. READ."
  - "math/math for ML.pdf, Sec. 6.4.3 (PDF p.199, book p.193): the raw-score formula V[x]=E[x^2]-(E[x])^2, computable in one pass but numerically unstable when the two terms are huge and nearly equal; pairwise-difference identity (6.45). READ."
  - "academy/content/authored/math/01-functions-and-notation/lesson.md (frontmatter and opening): the exponential/log notation and why ML works in log space; cross-referenced. READ."
  - "academy/content/authored/math/04-derivatives/lesson.md (frontmatter and opening): finite differences as a numerical derivative; cross-referenced. READ."
---

## Where this fits

Your training run prints `loss = nan` at step 40,000. Or a softmax returns `[nan, nan, nan]` the moment a logit reaches 1000. Or a variance comes out *negative*. None of these is a bug in the mathematics: each is the mathematics being executed on a machine that stores real numbers with a fixed, finite number of significant digits. Algebra says two formulas are equal; floating point says one of them works and one explodes.

This lesson explains how real numbers are stored, where the errors come from, and the handful of rewrites (the **log-sum-exp trick**, **Welford's algorithm**, **compensated summation**) that every serious ML codebase relies on. It connects backwards to lesson 1 (we compute in log space because probabilities multiply to tiny numbers), to lesson 4 (finite-difference derivatives have a sweet-spot step size, which we can now explain), and to lesson 13 (the Deisenroth text warns that the raw-score variance is unstable; we measure just how badly). It also explains modern hardware choices: why large models train in bfloat16, and what loss scaling is.

## The idea in plain words

- **Scientific notation with a budget.** A float is like writing a number in scientific notation but with a fixed number of digits for the leading part (the *mantissa* or *significand*) and a fixed range for the power of ten (the *exponent*). A float64 has about 16 significant digits, float32 about 7, float16 about 3. Numbers do not have uniform spacing: near 1 the grid is fine, near $10^{15}$ the neighbouring floats are 0.125 or more apart. Like a ruler whose marks get further apart the larger the number you measure.
- **Most decimals cannot be stored.** In base 2, $0.1$ is a repeating fraction (just as $1/3$ repeats in base 10), so the computer stores the closest representable neighbour. That is why `0.1 + 0.2 == 0.3` is false.
- **Subtracting nearly equal numbers destroys digits.** If you know two lengths to 7 digits, $1.234567$ and $1.234561$, their difference $0.000006$ has only one correct digit left. The leading digits cancel and what remains is mostly the earlier rounding noise promoted to the front. This is **catastrophic cancellation**.
- **Two cliffs: too big and too small.** Past the largest representable number you fall off into `inf`; below the smallest you fall to `0`. Then `inf/inf` and `0*inf` give `nan`, which poisons everything it touches.
- **The cure is rearrangement, not more precision.** Subtract the maximum before exponentiating. Work with deviations from the mean instead of squares of large numbers. Add small numbers together before adding them to a big one. The mathematics stays the same; only the order of operations changes.

## How floats are stored

An IEEE-754 binary float has a sign bit, an exponent field and a mantissa field. For normal numbers

$$
x=(-1)^{s}\times 2^{\,e-\text{bias}}\times(1.m)_2 .
$$

The mantissa gives 1 implicit leading bit plus its stored bits, so the *relative* spacing between neighbouring floats is between $2^{-p}$ and $2^{-p+1}$ where $p$ is the number of stored mantissa bits.

| format | bits | exponent | mantissa | machine epsilon | largest finite | smallest normal |
|---|---|---|---|---|---|---|
| float64 | 64 | 11 | 52 | $2.2\times10^{-16}$ | $1.8\times10^{308}$ | $2.2\times10^{-308}$ |
| float32 | 32 | 8 | 23 | $1.19\times10^{-7}$ | $3.40\times10^{38}$ | $1.18\times10^{-38}$ |
| float16 | 16 | 5 | 10 | $9.77\times10^{-4}$ | $65504$ | $6.1\times10^{-5}$ |
| bfloat16 | 16 | 8 | 7 | $7.8\times10^{-3}$ ($2^{-7}$) | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |

(The first three rows of the numeric columns are printed by `np.finfo`; bfloat16 is not a NumPy type, and its values follow from its 8-bit exponent like float32's and its 7 mantissa bits, as the repo lesson states.)

**Machine epsilon** $\varepsilon$ is the gap between 1 and the next float above it ($2^{-52}$ in float64). It bounds the relative rounding error of a single stored number or arithmetic operation: $\text{fl}(x)=x(1+\delta)$ with $|\delta|\le\varepsilon/2$ for round-to-nearest. Concretely `1 + 1e-16 == 1` is `True` in float64, while `1 + 2e-16 == 1` is `False`.

**Which format when?** float16 has more precision but a tiny range (max 65,504): logits and activations overflow. bfloat16 is "float32 with the bottom 16 mantissa bits dropped": the same range, much coarser precision. For training, range matters more than precision, so bfloat16 is the default on TPUs and recent GPUs. With float16 one uses **loss scaling**: multiply the loss by e.g. 1024 before the backward pass so small gradients (below float16's smallest values, about $6\times10^{-8}$) do not underflow to zero, and divide the gradients by the same factor before the update. Accumulations and master copies of weights are kept in float32.

Also remember **denormals**: below the smallest normal number floats lose precision gradually (float32 reaches $1.4\times10^{-45}$), then flush to 0.

## Rounding error and cancellation (with a derivation)

Suppose $a$ and $b$ are stored with relative errors up to $\delta$, so $\tilde a=a(1+\delta_a)$, $\tilde b=b(1+\delta_b)$. Then

$$
\tilde a-\tilde b=(a-b)+(a\delta_a-b\delta_b),\qquad
\frac{|\text{error}|}{|a-b|}\ \le\ \frac{|a|+|b|}{|a-b|}\,\delta .
$$

When $a\approx b$ the amplification factor $\frac{|a|+|b|}{|a-b|}$ is huge: the absolute error stays the size of $\varepsilon\,|a|$ but the true difference is much smaller, so the *relative* error explodes. Addition of numbers of the same sign never amplifies like this; subtraction of near-equals does.

### The variance trap (the Deisenroth warning, measured)

The "raw-score" formula $\text{Var}[x]=\mathbb E[x^2]-(\mathbb E[x])^2$ needs only one pass over the data, but Deisenroth et al. warn that it is unstable when the two terms are huge and nearly equal. Take $x=10^9+\{4,7,13,16\}$. The true mean deviation values are $-6,-3,3,6$, so the population variance is $(36+9+9+36)/4=22.5$ (sample variance $90/3=30$). Both terms in the raw-score formula are about $10^{18}$, where float64 spacing is 128:

- `np.mean(x**2) - np.mean(x)**2` returns **$-128.0$**, a negative variance.
- Subtracting the mean first, `x.var()`, returns $22.5$.

**Welford's algorithm** does the centering online in a single pass. Keep a running count $n$, mean $m$ and sum of squared deviations $M_2$. For each new value $x$:

$$
\delta=x-m,\qquad m\leftarrow m+\frac{\delta}{n},\qquad M_2\leftarrow M_2+\delta\,(x-m_{\text{new}}),
$$

and at the end $\text{Var}=M_2/n$ (or $M_2/(n-1)$). *Why it is right:* after $n$ points $M_2=\sum(x_i-m_n)^2$; the update adds the term $\delta(x-m_{new})=\frac{n-1}{n}\delta^2$, which is the exact increment of that sum when one point is added (a short algebra exercise: expand $\sum(x_i-m_n)^2$ in terms of $m_{n-1}$). All quantities handled are deviations, never huge squares. On the example it returns $22.5$ and $30.0$ exactly.

## Overflow, underflow and the stable softmax

$\text{softmax}(z)_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$. In float32 $e^{88.7}\approx3.3\times10^{38}$ is the last finite value and $e^{89}$ is `inf`; in float64 the limit is about $e^{709}$. With logits $z=(1000,1001,1002)$ the naive formula gives `inf/inf = nan`.

**Key identity.** Softmax is unchanged if you subtract the same constant $c$ from every logit:

$$
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}=\frac{e^{-c}e^{z_i}}{e^{-c}\sum_j e^{z_j}}=\frac{e^{z_i}}{\sum_je^{z_j}} .
$$

Choose $c=\max_j z_j$. Then every exponent is $\le0$, so no term overflows, and at least one term equals $e^0=1$, so the denominator is at least 1 and cannot underflow to zero. For $z=(1000,1001,1002)$ this gives $(0.0900,\,0.2447,\,0.6652)$.

### Log-sum-exp

Losses need $\log\sum_j e^{z_j}$ (the log-normaliser). The same shift gives

$$
\operatorname{LSE}(z)=c+\log\sum_j e^{z_j-c},\qquad c=\max_j z_j .
$$

For $z=(1000,1001,1002)$: $c=1002$, the sum is $e^{-2}+e^{-1}+1=1.5032$, and $\operatorname{LSE}=1002.4076$. The naive version returns `inf`. Cross-entropy for a correct class $y$ is then $-z_y+\operatorname{LSE}(z)$: compute it from logits directly rather than taking $\log$ of a softmax output, because a probability that has underflowed to $0$ makes $\log 0=-\infty$. (This is why library losses take logits, not probabilities.)

## Condition number: the problem itself may be unstable

Even a perfect algorithm cannot beat an ill-conditioned problem. The **condition number** is the factor by which relative input errors can be amplified in the output. For solving $Ax=b$:

$$
\frac{\|\delta x\|}{\|x\|}\ \le\ \kappa(A)\,\frac{\|\delta b\|}{\|b\|},\qquad \kappa(A)=\|A\|\,\|A^{-1}\|=\frac{\sigma_{\max}}{\sigma_{\min}} \ \ (\text{2-norm}),
$$

the ratio of the largest to the smallest singular value (lesson 7). Example: $A=\begin{pmatrix}1&1\\1&1.0001\end{pmatrix}$ has $\kappa\approx4\times10^{4}$. For $b=(2,\,2.0001)$ the solution is $(1,1)$; changing $b_2$ by only $10^{-4}$, to $2.0002$, moves the solution to $(0,2)$. A tiny change in the data completely changes the answer. In ML this is why the normal equations $X^\top Xw=X^\top y$ are risky: $\kappa(X^\top X)=\kappa(X)^2$, so squaring the matrix squares the amplification, and why ridge regularisation (adding $\lambda I$) stabilises it.

## Summation order and finite differences

Floating-point addition is **not associative**: $(a+b)+c\ne a+(b+c)$ in general. Float32 cannot add $1$ to $16{,}777{,}216=2^{24}$ (the result is rounded back to $2^{24}$): once a running total is large, small terms vanish. Adding $10^6$ copies of $0.1$ in float32 one by one (as `np.cumsum` does) ends at $100958.34$, an error of about 1%, while `np.sum` (which adds in pairs/blocks) gives $100000.01$. Two standard remedies:

- **Pairwise (tree) summation**, used by `np.sum`: error grows like $\log N$ rather than $N$.
- **Kahan compensated summation**: keep a correction $c$ holding the low-order bits lost at each addition.

```text
y = x - c ; t = s + y ; c = (t - s) - y ; s = t
```

On the $10^6$ float32 values $0.1$ it ends at exactly $100000.0$. (Here floating-point order also explains why GPU reductions can give slightly different results from run to run.)

**Finite differences revisited (lesson 4).** For $f(x)=e^x$ at $x=1$ the central difference $\frac{f(x+h)-f(x-h)}{2h}$ has absolute error $5.9\times10^{-11}$ at $h=10^{-5}$, but $6.6\times10^{-9}$ at $h=10^{-8}$, $2.1\times10^{-4}$ at $h=10^{-12}$ and $0.17$ at $h=10^{-15}$. Two errors compete: the formula's truncation error shrinks like $h^2$, while the rounding error from subtracting two nearly equal values $f(x+h)-f(x-h)$ and dividing by tiny $2h$ grows like $\varepsilon/h$. The balance is at $h\approx\varepsilon^{1/3}\approx6\times10^{-6}$ for float64. Smaller $h$ is not better; it is catastrophic cancellation.

## Worked example

Four checks, all reproduced by the exercise:

1. `softmax([1000, 1001, 1002])`: naive `[nan, nan, nan]`, stable `[0.09003, 0.24473, 0.66524]` (sums to 1).
2. `logsumexp([1000, 1001, 1002]) = 1002.4076`; and `logsumexp([-1000, -1000]) = -1000 + log 2 = -999.3069`, where the naive formula gives `-inf` because $e^{-1000}$ underflows to 0.
3. Variance of $10^9+\{4,7,13,16\}$: raw-score $-128.0$, Welford $22.5$ (population) and $30.0$ (sample).
4. Float32 sum of $10^5$ copies of $0.1$: sequential $9998.557$, Kahan $10000.0$.

## Implementation

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def logsumexp(z):
    c = z.max()
    return c + np.log(np.exp(z - c).sum())

def welford(xs):
    n, m, M2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        d = x - m
        m += d / n
        M2 += d * (x - m)
    return m, M2 / n
```

## Shapes

For a batch of logits `z: (B, K)`, take the max over the class axis with `keepdims=True` so the shift `(B, 1)` broadcasts against `(B, K)` (lesson 16); `softmax(z): (B, K)`, `logsumexp(z): (B,)`. Welford over a stream of `N` vectors in `R^D` keeps `m: (D,)` and `M2: (D,)`.

## Common misconceptions

- **"Use float64 and the problem disappears."** It only moves the cliff: the raw-score variance with $10^{9}$ offsets already fails in float64, and ML hardware deliberately uses 16-bit formats.
- **"`nan` means my data is bad."** Very often it is `inf - inf`, `0 * inf` or `log(0)` produced by an unstable formula.
- **"`a == b` is a fine float test."** Compare with a tolerance, `np.isclose(a, b)`.
- **"Adding is always safe."** Adding a tiny number to a huge running total silently drops it.
- **"A stable algorithm fixes an ill-conditioned problem."** No: a bad $\kappa$ amplifies even the unavoidable input rounding.

## Key takeaways

- Floats have relative precision $\varepsilon$ (float64 $2.2\times10^{-16}$, float32 $1.2\times10^{-7}$, float16 $10^{-3}$, bfloat16 $8\times10^{-3}$) and a finite range; bfloat16 trades precision for float32's range.
- Subtracting nearly equal numbers loses digits; rearrange (centre first, use Welford) rather than subtract large squares.
- Subtract the max before exponentiating: stable softmax and $\operatorname{LSE}(z)=c+\log\sum e^{z-c}$; feed losses logits, not probabilities.
- $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ measures how much errors are amplified; summation order matters (pairwise, Kahan).

## Explain it yourself

1. Why does subtracting the maximum logit not change softmax, yet prevent `nan`? Cover both the overflow and the underflow side.
2. Explain why the variance formula "mean of squares minus square of the mean" can give a negative number on a computer, and what Welford does differently.

## Further reading

- Repo phase 01, lesson 13: mixed precision, gradient clipping, NaN/Inf detection, gradient checking.
- Deisenroth et al., Sec. 6.4.3 and the numerical-stability remarks in Chapter 7 on optimisation.
- Goldberg, "What every computer scientist should know about floating-point arithmetic" (a classic reference; not one of the sources read for this lesson).

## Mastery criteria

- Give machine epsilon and range for float64, float32, float16 and bfloat16 and say why bfloat16 is used for training.
- Show with numbers why the raw-score variance fails and implement Welford.
- Implement stable softmax and log-sum-exp and prove the shift identity.
- Explain what the condition number measures and why $\kappa(X^\top X)=\kappa(X)^2$ matters.
