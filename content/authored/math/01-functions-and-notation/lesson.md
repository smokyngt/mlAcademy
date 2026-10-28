---
id: f2n.01
title: Functions, summation and logarithms
summary: The notation every later lesson uses, and why machine learning computes in log space.
minutes: 40
teaches: [alg.functions, alg.summation, alg.exp-log]
requires: []
sources:
  - math/math for ML.pdf (Ch. 1 Introduction and Motivation, PDF p.17; Ch. 2 opening, PDF p.23)
  - ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md
---

## Learning objectives

- Read and write function signatures such as $f:\mathbb{R}^n \to \mathbb{R}^m$ and say what the input and output sizes are.
- Expand and rewrite sums written with $\sum$, and move constants in and out of them.
- Use $\exp$ and $\ln$ as inverse functions and apply $\ln(ab)=\ln a+\ln b$.
- Explain why a product of many probabilities is computed as a sum of logs.

## Prerequisites

Arithmetic and the idea of a variable. Nothing else.

## Where this fits

Every machine learning model, from a spam filter to a language model, is a **function**: something that takes numbers in and gives numbers out. "Training" means adjusting the numbers inside that function until its outputs are good. To follow anything that comes later you need to be able to read how such a function is written, how several of them are chained, and how to add up an error over thousands of examples.

This first lesson is the alphabet. It has no difficult idea in it, but every later page uses its symbols without explaining them again, so it is worth ten patient minutes. It also contains one practical surprise (why programs add logarithms instead of multiplying probabilities) that explains a trick you will meet in nearly every ML codebase.

## The idea in plain words

- A **function** is a vending machine. You press one button (the input), and exactly one snack falls out (the output). Press the same button twice and you get the same snack. Writing `f: R^3 -> R^2` just says "this machine has three number slots on the way in and two on the way out".
- **Chaining** functions is putting machines in a row: the snack from the first machine is dropped into the second. A neural network is a row of simple machines.
- A **sum** (the Greek letter sigma) is a shopping receipt: go down the list and add each line to a running total. It is a `for` loop that adds.
- A **logarithm** answers "how many digits long is this number?" (roughly). That is why it turns multiplication into addition: multiplying a 3-digit number by a 4-digit number gives about a 7-digit number, and 3 + 4 = 7. Computers like this because multiplying hundreds of tiny numbers makes them vanish, while adding their logarithms stays comfortable.

Hold on to those four pictures; the rest of the lesson is just making them precise.

## Intuition

A **function** is a machine: put one input in, get exactly one output out. Machine learning is almost entirely about functions that are too complicated to write down by hand, so we describe them as *compositions of simple functions* and learn their parameters from data.

A **sum** $\sum$ is just a `for` loop that adds. A **logarithm** turns multiplication into addition, which is exactly what a computer needs when it multiplies hundreds of tiny numbers.

## Formal definitions

**Function.** A function $f$ from a set $A$ (the *domain*) to a set $B$ (the *codomain*) assigns to every $a\in A$ exactly one $f(a)\in B$. We write $f: A \to B$.

In machine learning the sets are nearly always spaces of real vectors:

| Signature | Read it as | Example |
|---|---|---|
| $f:\mathbb{R}\to\mathbb{R}$ | one number in, one number out | $f(x)=x^2$ |
| $L:\mathbb{R}^d\to\mathbb{R}$ | $d$ parameters in, one scalar loss out | mean squared error |
| $g:\mathbb{R}^n\to\mathbb{R}^m$ | a layer: $n$ features in, $m$ activations out | $g(x)=\mathrm{ReLU}(Wx+b)$ |

**Composition.** $(g\circ f)(x)=g(f(x))$. Apply $f$ first, then $g$. This requires that the output of $f$ lives in the domain of $g$. A neural network with three layers is $h = f_3\circ f_2\circ f_1$; the shapes must chain: $\mathbb{R}^{n}\to\mathbb{R}^{m_1}\to\mathbb{R}^{m_2}\to\mathbb{R}^{m_3}$.

**Summation.** For numbers $a_1,\dots,a_n$,

$$
\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n .
$$

The letter $i$ is a *dummy index*: renaming it to $k$ changes nothing. Three rules cover almost everything:

1. **Linearity:** $\sum_i (\alpha a_i + \beta b_i) = \alpha\sum_i a_i + \beta\sum_i b_i$.
2. **A constant summed $n$ times:** $\sum_{i=1}^{n} c = nc$.
3. **Order does not matter** for a finite sum (but see floating-point notes below).

The product $\prod_{i=1}^n a_i = a_1 a_2\cdots a_n$ works the same way.

**Exponential and logarithm.** $\exp(x)=e^x$ with $e\approx 2.71828$. The natural logarithm $\ln$ is its inverse: $\ln(\exp(x))=x$ and $\exp(\ln y)=y$ for $y>0$. The one property that matters most:

$$
\exp(a+b)=\exp(a)\exp(b)\qquad\Longleftrightarrow\qquad \ln(ab)=\ln a+\ln b .
$$

## Derivation: why products become sums

Let $y=\exp(a)$ and $z=\exp(b)$. Then $yz=\exp(a)\exp(b)=\exp(a+b)$ by the exponent rule. Take $\ln$ of both sides: $\ln(yz)=a+b=\ln y+\ln z$. Repeating the argument gives, for positive $p_i$,

$$
\ln\Big(\prod_{i=1}^n p_i\Big)=\sum_{i=1}^n \ln p_i .
$$

## Worked example 1: a sum

$$
\sum_{i=1}^{4} i^2 = 1^2+2^2+3^2+4^2 = 1+4+9+16 = 30 .
$$

With linearity: $\sum_{i=1}^{4}(3i^2+2) = 3\cdot 30 + 2\cdot 4 = 98$.

## Worked example 2: why we need logs

Suppose a model assigns probability $0.1$ to each of 400 independent observations. The likelihood is $0.1^{400}=10^{-400}$. A 64-bit float cannot represent anything smaller than about $2.2\times10^{-308}$, so the computer stores **0**. But

$$
\ln(0.1^{400}) = 400\ln 0.1 \approx 400\times(-2.3026) = -921.03,
$$

which is a perfectly ordinary number. This is why every likelihood in machine learning is handled as a **log-likelihood**.

## Implementation

```python
import numpy as np

p = np.full(400, 0.1)
print(np.prod(p))          # 0.0  (underflow)
print(np.sum(np.log(p)))   # -921.03...
```

A closely related trap: $\log\sum_i e^{x_i}$ overflows if some $x_i$ is large (e.g. $e^{1000}=\infty$). Subtract the maximum $m=\max_i x_i$ first:

$$
\ln\sum_i e^{x_i} = m + \ln\sum_i e^{x_i-m}.
$$

After the shift every exponent is $\le 0$, so no term can overflow, and at least one term equals $1$, so the sum cannot underflow to zero. You will implement this ("log-sum-exp") in the exercises; it is the core of the stable softmax in lesson 8.

## Shapes

For a signature $f:\mathbb{R}^n\to\mathbb{R}^m$, always ask: *what shape goes in, what shape comes out?* When you later see `W @ x`, you will check $W\in\mathbb{R}^{m\times n}$, $x\in\mathbb{R}^n$, result $\in\mathbb{R}^m$.

## Common misconceptions

- **"$f(x+y)=f(x)+f(y)$ for any function."** Only *linear* functions satisfy this. $(x+y)^2\neq x^2+y^2$, and $\ln(x+y)\neq\ln x+\ln y$.
- **"$\ln(a+b)=\ln a+\ln b$."** No: the logarithm converts *products* to sums, not sums to sums.
- **"$\sum_i a_ib_i = (\sum_i a_i)(\sum_i b_i)$."** The product of sums has $n^2$ cross terms; the sum of products has $n$.
- **Floating point:** a sum of floats depends slightly on the order of addition. Mathematically it does not.

## Key takeaways

- A function is defined by its input and output spaces: always read `f: R^n -> R^m` before anything else.
- Sums are loops; linearity lets you pull constants out and split sums.
- `ln(ab) = ln a + ln b` is why likelihoods are handled as sums of logs.
- Subtract the maximum before exponentiating: that is the whole trick behind a stable log-sum-exp.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Why does a product of 400 probabilities become exactly 0 in floating point, while the sum of their logs is fine?
2. Compute `ln(1000)` given only `ln 2` and `ln 5`. Which rule did you use?

## Further reading

- *Mathematics for Machine Learning* (Deisenroth et al.), Ch. 1 for the motivation, PDF p.17.
- Source library: `ai-engineering-from-scratch` phase 01, lesson 01.

## Mastery criteria

You can (a) state the input and output dimension of any function in this course, (b) rewrite a sum using linearity without a calculator, and (c) explain in two sentences why log-likelihoods exist.
