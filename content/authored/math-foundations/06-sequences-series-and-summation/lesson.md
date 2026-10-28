---
id: mf.06
title: "Sequences, series and summation"
summary: Sequences, arithmetic and geometric series with their formulas derived, sigma and product notation, convergence of the geometric series, the divergent harmonic series, and loops versus closed forms.
minutes: 95
teaches: [mf.series]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (the limit definition of the derivative and gradient-descent discussion): sums over many terms and limits as the context for convergence. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md (log-sum-exp section; floating-point accumulation): why sums of many terms need care in code. READ."
  - "math/math for ML.pdf, Sec. 2.1 (PDF p.26): the general sum a_i1 x_1 + ... + a_in x_n written in sigma-style shorthand for a system of equations. READ."
---

## Where this fits

Machine learning is full of sums. A loss is an average of errors over $N$ examples: $\frac1N\sum_{i=1}^N\ell_i$. A dot product is $\sum_ja_jb_j$. A probability distribution sums to 1. A discounted reward in reinforcement learning is $\sum_k\gamma^kr_k$, a geometric series. Learning rates decay geometrically; the number of steps a process needs often follows a harmonic series. Before you can read these formulas you must be at ease with the notation, and before you can trust them you need to know when an infinite sum even has an answer.

No earlier lesson is required. We use only arithmetic and exponents (`mf.02`, `mf.04`) and write short Python loops (`mf.01`). The final section prepares the limit idea of `mf.07`.

## The idea in plain words

A **sequence** is a list with a rule: 2, 4, 6, 8, ... (add 2 each time) or 3, 6, 12, 24, ... (double each time). A **series** is what you get when you *add up* the list.

Two special lists are worth a lot:

- **Arithmetic**: each step adds the same amount, like climbing stairs of equal height. To add up 1 to 100, a famous trick is to write the list forward and backward and add column by column. Every column sums to the same number, so the total is easy.
- **Geometric**: each step multiplies by the same amount, like a bouncing ball that rises to half its previous height. If the multiplier is smaller than 1 in size, the terms shrink so fast that even *infinitely many* of them add to a finite number: a ball that bounces forever still travels a finite distance. If the multiplier is 1 or larger, the total is unbounded.

And a warning picture: you might think "if the pieces you add get smaller and smaller, the total must settle down". The **harmonic series** 1 + 1/2 + 1/3 + 1/4 + ... shows this is false: it creeps upward without limit, extremely slowly.

## Sequences

A sequence is a function of the position: $a_1,a_2,a_3,\dots$, with a **general term** $a_n$. Two ways to describe one: a direct formula ($a_n=3n-1$ gives 2, 5, 8, ...), or a recursion ($a_1=2$, $a_{n+1}=a_n+3$).

**Arithmetic sequence:** a constant difference $d$.

$$
a_n=a_1+(n-1)d .
$$

Example: first term 5, $d=3$: 5, 8, 11, ..., and the 10th term is $5+9\cdot3=32$.

**Geometric sequence:** a constant ratio $r$.

$$
a_n=a_1r^{n-1} .
$$

Example: 3, 6, 12, 24, ... has $a_1=3$, $r=2$, and $a_{10}=3\cdot2^9=1536$.

## Sigma and product notation

$$
\sum_{k=m}^{n}a_k=a_m+a_{m+1}+\dots+a_n
$$

reads "the sum of $a_k$ as $k$ runs from $m$ to $n$". The letter $k$ is the **index** (a dummy name: renaming it to $i$ changes nothing), $m$ the lower limit, $n$ the upper limit. Examples:

- $\sum_{k=1}^{4}k^2=1+4+9+16=30$.
- $\sum_{k=0}^{3}2^k=1+2+4+8=15$.
- $\sum_{k=1}^{3}5=5+5+5=15$ (a constant summand just counts terms).

In code the sigma is a loop with an accumulator:

```python
total = 0
for k in range(1, 5):      # k = 1, 2, 3, 4
    total += k ** 2
print(total)               # 30
```

or one NumPy line `np.sum(np.arange(1, 5) ** 2)`.

Rules (they follow from ordinary addition rules):

$$
\sum(a_k+b_k)=\sum a_k+\sum b_k,\qquad \sum c\,a_k=c\sum a_k .
$$

But $\sum a_kb_k\neq(\sum a_k)(\sum b_k)$ in general.

**Product notation** $\prod_{k=1}^{n}a_k=a_1a_2\cdots a_n$. The most famous: $n!=\prod_{k=1}^{n}k$, so $5!=120$. Taking the log (from `mf.04`) turns a product into a sum: $\ln\prod a_k=\sum\ln a_k$, the log-likelihood trick.

## Deriving the arithmetic series

Let $S=1+2+\dots+n$. Write it twice, the second time reversed, and add column by column:

$$
\begin{aligned}
S&=1+2+\dots+(n-1)+n\\
S&=n+(n-1)+\dots+2+1\\
2S&=(n+1)+(n+1)+\dots+(n+1)\quad(n\text{ columns})
\end{aligned}
$$

So $2S=n(n+1)$ and

$$
1+2+\dots+n=\frac{n(n+1)}2 .
$$

For $n=100$: $100\cdot101/2=5050$. The same pairing works for any arithmetic sequence: the first term plus the last term is the same in every column, giving

$$
S_n=\frac n2\,(a_1+a_n)=\frac n2\big(2a_1+(n-1)d\big).
$$

Check: 5, 8, ..., 32 (ten terms): $\frac{10}{2}(5+32)=185$.

## Deriving the geometric series

Let $S=a+ar+ar^2+\dots+ar^{n-1}$. Multiply by $r$ and subtract:

$$
\begin{aligned}
S&=a+ar+ar^2+\dots+ar^{n-1}\\
rS&=\phantom{a+{}}ar+ar^2+\dots+ar^{n-1}+ar^{n}\\
S-rS&=a-ar^{n}
\end{aligned}
$$

All middle terms cancel. Hence $S(1-r)=a(1-r^n)$, and for $r\neq1$

$$
S_n=a\,\frac{1-r^n}{1-r} .
$$

(If $r=1$ every term equals $a$ and $S_n=na$.) Checks: $1+\frac12+\dots+\frac1{2^9}$ (10 terms) $=\frac{1-2^{-10}}{1-\frac12}=1.998046875$; and $3+6+\dots+3\cdot2^9=3\frac{1-2^{10}}{1-2}=3069$.

## Convergence of the geometric series

Now let the number of terms grow without end. Look at the term $r^n$ in the formula.

- If $|r|<1$, then $r^n\to0$ (powers of a number smaller than 1 in size shrink towards zero: $0.9^n$, $(-0.5)^n$). So $S_n\to\frac{a}{1-r}$:

$$
\sum_{k=0}^{\infty}ar^k=\frac{a}{1-r},\qquad|r|<1 .
$$

- If $|r|\ge1$, then $r^n$ does not shrink, and the partial sums do not settle: for $r=1$ they grow without bound, for $r=-1$ they alternate $a,0,a,0,\dots$, for $|r|>1$ they explode.

Examples: $1+\frac12+\frac14+\dots=2$. $1-\frac12+\frac14-\dots=\frac1{1+\frac12}=\frac23$. $\sum_k0.9^k=10$ (with 200 terms the code gives $9.99999999$, the gap being $0.9^{200}\cdot10\approx7\times10^{-9}$). In reinforcement learning with discount $\gamma=0.9$ and reward 1 at every step, the total discounted reward is therefore 10: this is exactly why discounting is used, to keep infinite sums finite.

A picture of why $\sum 2^{-k}$ is 2: start with a 2-metre stick, remove 1 m, then 0.5 m, then 0.25 m... you approach but never pass the end of the stick.

## The harmonic series diverges

$$
H_n=1+\frac12+\frac13+\dots+\frac1n .
$$

The terms tend to zero, yet the sum is unbounded. Group the terms: $\frac13+\frac14\ge\frac14+\frac14=\frac12$; $\frac15+\dots+\frac18\ge4\cdot\frac18=\frac12$; the next 8 terms are each at least $\frac1{16}$, giving at least $\frac12$, and so on. Every doubling of the number of terms adds at least $\frac12$, so $H_n$ eventually passes any number. But slowly: $H_{10}=2.929$, $H_{1000}=7.485$, $H_{10^6}=14.39$, and 12 367 terms are needed just to exceed 10 (it grows like $\ln n$, `mf.04`). Lesson: **terms going to zero is necessary for convergence but not sufficient.**

## Loops versus closed forms

A closed form replaces $n$ additions by a formula. Compare for $\sum_{k=1}^nk^2$:

```python
def loop_sum_squares(n):
    total = 0
    for k in range(1, n + 1):
        total += k * k
    return total

def closed_sum_squares(n):
    return n * (n + 1) * (2 * n + 1) // 6

print(loop_sum_squares(10), closed_sum_squares(10))   # 385 385
```

Where does the formula come from? Use the identity $(k+1)^3-k^3=3k^2+3k+1$ and add it for $k=1,\dots,n$. The left side telescopes (most terms cancel) to $(n+1)^3-1$. The right side is $3\sum k^2+3\sum k+n$. So $3\sum k^2=(n+1)^3-1-\frac{3n(n+1)}2-n$, which simplifies to $\frac{n(n+1)(2n+1)}2$, giving $\sum k^2=\frac{n(n+1)(2n+1)}6$. Numeric test: $n=10$ gives $10\cdot11\cdot21/6=385$.

Practical rules: if a closed form exists, use it (cost does not grow with $n$) and test it against the loop for small $n$. If it does not (harmonic numbers), vectorise: `np.sum(1 / np.arange(1, n + 1))`, or running totals with `np.cumsum`. For very long sums of floats, accumulate carefully: adding many tiny terms to a big total loses precision, which is one reason numerical code groups sums or works with logs.

## Worked example in code

```python
import numpy as np

print(sum(range(1, 101)))                         # 5050
print(sum(1 / 2 ** k for k in range(10)))         # 1.998046875
print(3 * (1 - 2 ** 10) / (1 - 2))                # 3069.0
H = lambda n: sum(1 / k for k in range(1, n + 1))
print(H(10), H(1000))                             # 2.9289682539682538 7.485470860550345
n, s = 0, 0.0
while s <= 10:
    n += 1
    s += 1 / n
print(n)                                          # 12367
print(np.cumsum(1 / 2 ** np.arange(5)))           # [1. 1.5 1.75 1.875 1.9375]
```

## Common misconceptions

- "If the terms go to zero, the series converges." The harmonic series is the counterexample.
- "$\sum_{k=1}^nc=c$." It is $nc$: there are $n$ terms.
- "$\sum a_kb_k=\sum a_k\sum b_k$." Wrong; that is only true for a different object (a double sum).
- Off-by-one errors: $\sum_{k=1}^{n}$ has $n$ terms, $\sum_{k=0}^{n}$ has $n+1$; Python's `range(1, n+1)` includes $n$.
- Using the infinite-geometric formula when $|r|\ge1$.
- Thinking an infinite sum of positive numbers must be infinite. It need not be.

## Key takeaways

- Arithmetic: $a_n=a_1+(n-1)d$, $S_n=\frac n2(a_1+a_n)$ by forward-and-backward pairing.
- Geometric: $a_n=a_1r^{n-1}$, $S_n=a\frac{1-r^n}{1-r}$ by multiply-and-subtract; infinite sum $\frac a{1-r}$ only when $|r|<1$.
- Sigma is a loop with an accumulator; product notation multiplies; logs convert products to sums.
- The harmonic series diverges slowly like $\ln n$.
- Prefer closed forms or vectorised sums over Python loops, and verify one against the other.

## Explain it yourself

1. Explain why the derivation of the geometric sum multiplies by $r$ before subtracting.
2. Why does a discount factor $\gamma<1$ make an infinite stream of rewards finite, while $\gamma=1$ does not?

## Further reading

- The next lesson, `mf.07`, makes precise what "the partial sums approach a number" means (limits).
- Later: expectation (a weighted sum), series expansions and Taylor series, discounted returns.

## Mastery criteria

You can: write the $n$-th term of an arithmetic or geometric sequence; expand and evaluate sigma and product expressions; derive both finite formulas from scratch; decide whether an infinite geometric series converges and find its sum; explain why the harmonic series diverges; check a closed form against a loop in code.
