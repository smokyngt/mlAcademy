---
id: mf.04
title: "Exponentials and logarithms from zero"
summary: From repeated multiplication to the number e, then logarithms as the undo button, their laws, change of base, compound growth, and why every probabilistic model in ML is trained through logs.
minutes: 100
teaches: [mf.explog]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (opening to the derivative/partial-derivative sections): where exp and log appear (sigmoid, cross-entropy) and the limit idea behind rates of change. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md (The Problem, the exp/log overflow and underflow section, Log-Sum-Exp trick and its derivation, stable cross-entropy): ML uses of exp and log and the max-subtraction trick. READ."
---

## Where this fits

Probabilities multiply, growth multiplies, and learning rates decay by multiplication. Wherever quantities are multiplied over and over, exponentials appear; wherever you need to undo that, or to turn a long product into a manageable sum, logarithms appear. In machine learning you will constantly see $\exp$ (inside softmax and sigmoid) and $\ln$ (inside cross-entropy, log-likelihood, KL divergence, perplexity). A classifier's training loss is literally "minus the log of the probability you gave to the right answer".

This lesson builds both ideas from nothing but repeated multiplication and requires no earlier lesson (we use the power rules revisited in `mf.02`; if you feel shaky, skim its "Powers and roots" section first, but everything needed is restated here).

## The idea in plain words

**Exponentials** answer "what happens if I keep multiplying by the same number?" Fold a sheet of paper: each fold doubles the thickness. Ten folds is not 10 times the thickness but 1024 times. A rumour that doubles its audience each day, money with interest, a virus without control, a learning rate cut in half each epoch: all repeated multiplication. The key feeling: growth is *slow at first and then explosive*.

**Logarithms** answer the reverse question: "how many multiplications did it take?" If something doubled until it reached 1024 times its start, the log (base 2) says 10. A logarithm counts *how many times* you multiplied; it measures things on a scale of **orders of magnitude**. That is why earthquake and sound scales use logs: they compress huge ranges (1 to a billion) into small numbers (0 to 9).

A third picture: a **slide rule**. Because "counting multiplications" turns *multiplying* two numbers into *adding* their counts, logs turn products into sums. Everything useful about logs follows from that single trick.

## Exponents: repeated multiplication

$a^n$ means $n$ copies of $a$ multiplied: $2^5=2\cdot2\cdot2\cdot2\cdot2=32$.

**Exponent laws, derived by counting copies** (for $a>0$):

| law | why |
|---|---|
| $a^m\cdot a^n=a^{m+n}$ | $m$ copies then $n$ more copies is $m+n$ copies |
| $a^m/a^n=a^{m-n}$ | cancel $n$ copies from the top |
| $(a^m)^n=a^{mn}$ | $n$ groups of $m$ copies |
| $(ab)^n=a^nb^n$ | regroup the factors |
| $a^0=1$ | forced: $a^m\cdot a^0$ must equal $a^{m}$ |
| $a^{-n}=1/a^n$ | forced: $a^n\cdot a^{-n}=a^0=1$ |
| $a^{1/n}=\sqrt[n]{a}$ | forced: $(a^{1/n})^n=a^1$ |

Tiny examples: $2^3\cdot2^4=8\cdot16=128=2^7$. $(2^3)^2=64=2^6$. $2^{-2}=\tfrac14$. $8^{1/3}=2$. $9^{3/2}=(\sqrt9)^3=27$. With a base above 1, the larger the exponent the larger the result; with a base between 0 and 1 (such as $\tfrac12$), repeated multiplication **shrinks**: $(\tfrac12)^n$ is exponential decay.

## Compound growth

If an amount $P$ grows by a fraction $r$ each period (0.05 for 5%), one period multiplies it by $(1+r)$. After $t$ periods:

$$
P(1+r)^t .
$$

Example: 1000 at 5% for 10 years: $1000\cdot1.05^{10}=1628.89$. Not $1500$: the growth compounds because interest earns interest.

**Doubling time.** When is $(1+r)^t=2$? We need the exponent $t$, an unknown *in the exponent*. Ordinary algebra cannot isolate it; that is exactly the gap logarithms fill. Answer (derived below): $t=\ln2/\ln(1+r)$. At 7%: $10.24$ periods. The rule of thumb "divide 70 by the percentage" gives 10, close.

## The number e

Suppose a 100% annual rate is split into $n$ compounding steps per year, each at $1/n$. After a year, 1 becomes $(1+\tfrac1n)^n$.

| $n$ | $(1+1/n)^n$ |
|---|---|
| 1 | 2.0000 |
| 10 | 2.5937 |
| 1000 | 2.7169 |
| 1 000 000 | 2.71828 |

The values do not blow up; they creep toward a fixed number, $e\approx2.718281828$. We define $e=\lim_{n\to\infty}(1+1/n)^n$ ("limit" is made precise in `mf.07`). Continuous compounding at rate $r$ for time $t$ gives $Pe^{rt}$. For example, 5% for 10 years ($rt=0.5$) gives $1000\,e^{0.5}=1648.72$, slightly more than the $1628.89$ from yearly compounding. Another formula for $e$ is the sum $1+1+\tfrac12+\tfrac16+\tfrac1{24}+\dots=\sum_k1/k!$; eleven terms already give $2.718281801$ (the series idea comes back in `mf.06`).

Why is $e$ special? $e^x$ is the unique exponential whose slope at every point equals its own height. That makes it the natural language of anything whose growth rate is proportional to its size, and it is why derivatives of $e^x$ are so simple (a later lesson). In ML, $e^x$ turns any real number, positive or negative, into a positive number: perfect for building probabilities (softmax) and positive scales.

## Logarithms: the undo button

**Definition.** For a base $b>0$, $b\neq1$ and $x>0$:

$$
\log_b x = y \quad\Longleftrightarrow\quad b^y=x .
$$

Read: "$\log_b x$ is the exponent you put on $b$ to get $x$." So $\log_2 32=5$ because $2^5=32$; $\log_{10}1000=3$; $\log_b1=0$ (since $b^0=1$); $\log_bb=1$. The **natural logarithm** is base $e$: $\ln x=\log_ex$. Because logs are exponent-finders, they undo exponentials:

$$
\ln(e^x)=x,\qquad e^{\ln x}=x\ (x>0),\qquad \log_b(b^x)=x .
$$

**Domain.** $b^y$ is always positive, so $\log_bx$ exists only for $x>0$. $\ln0$ is not a number (it plunges to $-\infty$ as $x\to0^+$) and $\ln$ of a negative number is undefined over the reals. This is why code that computes `log(0.0)` returns `-inf` and `log(-1.0)` returns `nan`; ML code guards with a tiny constant like `log(x + 1e-8)` or uses stable formulas.

**Shape.** $\ln x$ rises, but ever more slowly: $\ln1=0$, $\ln2=0.693$, $\ln10=2.303$, $\ln100=4.605$. Multiplying the input by 10 only adds 2.303. Probabilities between 0 and 1 map to *negative* numbers ($\ln0.5=-0.693$), and the smaller the probability, the more negative.

## Log laws (derivations)

Let $\log_ba=m$ and $\log_bc=n$, meaning $a=b^m$ and $c=b^n$.

1. **Product.** $ac=b^mb^n=b^{m+n}$, so $\log_b(ac)=m+n$:

$$
\log_b(ac)=\log_ba+\log_bc .
$$

2. **Quotient.** $a/c=b^{m-n}$, so $\log_b(a/c)=\log_ba-\log_bc$.
3. **Power.** $a^k=(b^m)^k=b^{mk}$, so $\log_b(a^k)=k\log_ba$.
4. **Change of base.** Let $y=\log_bx$, i.e. $b^y=x$. Take $\ln$ of both sides and use law 3: $y\ln b=\ln x$, so

$$
\log_bx=\frac{\ln x}{\ln b} .
$$

Examples: $\log_28+\log_24=3+2=5=\log_232$ (since $8\cdot4=32$). $\log_2 10=\ln10/\ln2=3.3219$. Now the doubling time: $(1+r)^t=2\Rightarrow t\ln(1+r)=\ln2\Rightarrow t=\ln2/\ln(1+r)$, using the power law.

Beware: $\log(a+c)$ has **no** simplification. $\log(a+c)\ne\log a+\log c$.

## Why logs appear in ML

**1. Products of probabilities become sums.** If a model assigns probabilities $p_1,\dots,p_N$ to $N$ independent training examples, the *likelihood* is $\prod_ip_i$. With 2000 coin flips each of probability $0.5$, the product is $0.5^{2000}$, which is smaller than the smallest positive float and prints as `0.0`. The **log-likelihood** is

$$
\ln\prod_ip_i=\sum_i\ln p_i ,
$$

here $2000\ln0.5=-1386.29$, a perfectly ordinary number. Since $\ln$ is increasing, the parameters that maximise the likelihood also maximise the log-likelihood, and sums are far easier to differentiate than products.

**2. Cross-entropy.** For a classifier with predicted probabilities $q$ over classes and true distribution $p$, the loss is

$$
H(p,q)=-\sum_ip_i\ln q_i .
$$

For a one-hot truth the only surviving term is $-\ln q_{\text{true}}$. Giving the right class probability $0.7$ costs $-\ln0.7=0.357$; giving it $0.1$ costs $2.303$; giving it $0.001$ costs $6.9$. The log punishes confident mistakes severely but gently rewards improvement near 1, which is exactly the behaviour you want.

**3. Softmax and the log-sum-exp trick.** Softmax uses $e^{z_i}$, and its log needs $\ln\sum_ie^{z_i}$. For $z=(1000,1001,1002)$, $e^{1000}$ overflows to infinity in 64-bit floats. The fix uses the product law: with $m=\max z$,

$$
\ln\sum_ie^{z_i}=m+\ln\sum_ie^{z_i-m},
$$

and now the largest exponent is $e^0=1$, so nothing overflows. Check: $z=(1,2,3)$ gives $3.4076$ both ways, and $z=(1000,1001,1002)$ gives $1002.4076$.

## Worked example in code

All of these numbers are reproduced by the exercise.

```python
import math
import numpy as np

print(1000 * 1.05 ** 10)                 # 1628.894626777442
print(1000 * math.exp(0.5))              # 1648.7212707001281
print([(1 + 1/n) ** n for n in (1, 10, 1000, 10**6)])
print(math.log(2) / math.log(1.07))      # 10.244768351058712
print(math.log(10) / math.log(2))        # 3.3219280948873626  (log base 2 of 10)
print(0.5 ** 2000, 2000 * math.log(0.5)) # 0.0   -1386.2943611198905
print(-math.log(0.7))                    # 0.35667494393873245

z = np.array([1000., 1001., 1002.])
m = z.max()
print(m + np.log(np.exp(z - m).sum()))   # 1002.4076059644444
```

Notice line six: the product underflows to exactly zero, but the log version is fine.

## Common misconceptions

- "$\log(a+b)=\log a+\log b$." No: $\log(ab)=\log a+\log b$.
- "$(a+b)^n=a^n+b^n$." No.
- "$\ln$ and $\log$ are different functions." Both are logarithms with different bases; in most ML and math texts $\log$ means $\ln$. Check the convention (some engineering texts use base 10, and information theory often uses base 2).
- "Exponential growth is fast from the start." It looks flat for a long while, then explodes.
- "$e^{x}$ with negative $x$ is negative." It is positive, just small: $e^{-3}=0.0498$.
- Forgetting that logs of numbers in $(0,1)$ are negative, so loss formulas carry a minus sign in front.

## Key takeaways

- Exponent laws follow from counting copies; $a^0=1$, $a^{-n}=1/a^n$, $a^{1/n}=\sqrt[n]a$ keep them consistent.
- $e\approx2.71828$ is the limit of $(1+1/n)^n$ and the base of continuous growth.
- $\log_bx$ is the exponent giving $x$ from $b$; $\ln$ undoes $\exp$; the domain is $x>0$.
- Log laws: products to sums, powers to multipliers; change of base $\log_bx=\ln x/\ln b$.
- ML uses logs for log-likelihood, cross-entropy and numerically stable softmax (log-sum-exp).

## Explain it yourself

1. Why do logarithms turn multiplication into addition? Explain with the "counting how many multiplications" picture.
2. Why does a model trained on a million examples work with the sum of log-probabilities and not the product of probabilities?

## Further reading

- Later lessons on information theory (entropy, KL divergence), on softmax and sigmoid, and on numerical stability use everything here.

## Mastery criteria

You can: use all exponent laws with negative and fractional exponents; compute compound growth and doubling time; state what $e$ is and approximate it numerically; convert between exponential and log forms; apply log laws and change of base; explain in two sentences why ML trains with log-probabilities; compute a log-sum-exp without overflow.
