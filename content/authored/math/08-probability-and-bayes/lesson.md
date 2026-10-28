---
id: math.08
title: Probability, conditioning and Bayes' theorem
summary: The two rules that all of probability reduces to, and how Bayes' theorem turns evidence into updated beliefs.
minutes: 60
teaches: [prob.basics]
requires: [alg.summation]
sources:
  - "math/math for ML.pdf, Sec. 6.3 Sum Rule, Product Rule, and Bayes' Theorem (PDF p.189-192): sum rule (marginalisation), product rule, Bayes' theorem with prior / likelihood / posterior / evidence naming. READ."
  - "math/math for ML.pdf, Table 6.1 nomenclature (PDF p.189): pmf vs pdf vs cdf. READ."
---

## Where this fits

Machine learning is reasoning under uncertainty: the data are noisy, the model is imperfect, and the future is unseen. Probability is the language for saying how sure we are. Three things in this course lean on it directly: the **loss functions** of lesson 8 are negative log-probabilities; **classification** outputs are probabilities (logistic regression, softmax); and **Bayes' theorem** is how you update a belief after seeing data, which is the heart of naive Bayes, Bayesian inference and, in disguise, of every "train on data" procedure. The book makes the same point: once the distributions describing your data and problem are written down, "there are only two fundamental rules, the sum rule and the product rule", and everything else follows.

## The idea in plain words

- Probability is a number between 0 and 1 that says how plausible something is; 1 is certain, 0 impossible, and probabilities of all the exclusive possibilities add up to 1.
- Think of **a large crowd of 1000 people**, each labelled with two properties (say, has the disease or not, tested positive or not). A probability is then just a *fraction of the crowd*. A **conditional** probability is the fraction *within a sub-crowd*: "among the people who tested positive, what fraction have the disease?".
- The **sum rule** says: to get the chance of one property regardless of the other, add up (marginalise over) the other property. The **product rule** says: the chance of both is the chance of one times the chance of the other *given* the first.
- **Bayes' theorem** is the product rule read in both directions. It tells you how to flip a conditional: you know "if diseased, the test is positive 90% of the time" but you want "if the test is positive, what is the chance of disease?". These are different questions, and confusing them is one of the most common errors in reasoning with data.

## Formal rules

For random variables `x`, `y` with joint distribution `p(x, y)`:

$$
\textbf{Sum rule (marginalisation):}\quad p(x)=\sum_{y}p(x,y)\ \ \big(\text{or }\textstyle\int p(x,y)\,dy\big).
$$

$$
\textbf{Product rule:}\quad p(x,y)=p(y\mid x)\,p(x)=p(x\mid y)\,p(y).
$$

The **conditional** probability follows: `p(y | x) = p(x, y) / p(x)` for `p(x) > 0`. Two variables are **independent** if `p(x, y) = p(x) p(y)`, equivalently `p(y | x) = p(y)`: learning `x` tells you nothing about `y`.

### Bayes' theorem (derivation)

Write the product rule both ways, `p(x | y) p(y) = p(y | x) p(x)`, and divide by `p(y)`:

$$
\boxed{\underbrace{p(x\mid y)}_{\text{posterior}}=\frac{\overbrace{p(y\mid x)}^{\text{likelihood}}\ \overbrace{p(x)}^{\text{prior}}}{\underbrace{p(y)}_{\text{evidence}}}},\qquad p(y)=\sum_x p(y\mid x)\,p(x).
$$

Reading the terms as the book does: the **prior** `p(x)` is what you believe about the unobserved quantity `x` before seeing data; the **likelihood** `p(y | x)` says how likely the observation `y` is if `x` were true (note it is a distribution over `y`, not over `x`); the **evidence** `p(y)` just normalises so the posterior sums to 1; the **posterior** `p(x | y)` is what you know about `x` after seeing `y`. The prior must give non-zero probability to everything plausible, because a hypothesis with prior 0 can never be revived by any evidence.

## Worked example: the medical test

A disease affects 1% of people. A test detects it 90% of the time when present (sensitivity) and gives a false alarm 5% of the time when absent. Someone tests positive. What is the chance they have the disease?

Take 10,000 people:

- 100 have the disease; 90 of them test positive.
- 9,900 do not; 5% of them, `495`, also test positive.
- Positives in total: `90 + 495 = 585`. Of these, only 90 are sick.

$$
P(D\mid +)=\frac{0.9\cdot0.01}{0.9\cdot0.01+0.05\cdot0.99}=\frac{0.009}{0.0585}\approx0.154 .
$$

Only about **15%**, although the test is "90% accurate". The reason is the low prior: most positives come from the huge healthy group. This is the *base-rate fallacy*, and the crowd picture makes it obvious.

**Updating twice.** Posteriors become the next prior. A second independent positive test turns `0.154` into `0.9·0.154 / (0.9·0.154 + 0.05·0.846) = 0.1386/(0.1386+0.0423) ≈ 0.766`. Doing both updates at once, with likelihood `0.9²` vs `0.05²`, gives the same `0.766`: the order does not matter.

## Implementation

```python
import numpy as np

def posterior(prior, likelihood):
    """prior[h] = P(h); likelihood[h] = P(data | h). Returns P(h | data)."""
    joint = prior * likelihood           # product rule
    return joint / joint.sum()           # divide by the evidence (sum rule)

print(posterior(np.array([0.01, 0.99]), np.array([0.9, 0.05])))   # [0.1538, 0.8462]
```

## Shapes

For a discrete joint distribution over `x` (n states) and `y` (m states), store `P` with shape `(n, m)`, entries `≥ 0`, total `1`. Then `p(x) = P.sum(axis=1)` shape `(n,)`, `p(y) = P.sum(axis=0)` shape `(m,)`, `p(y | x) = P / P.sum(axis=1, keepdims=True)` (each row sums to 1).

## Common misconceptions

- **`P(A | B) = P(B | A)`.** No; they differ by the ratio of priors (Bayes). "Most positives are false" and "the test is accurate" can both be true.
- **Independence versus exclusivity.** Mutually exclusive events (cannot both happen) are strongly *dependent*: knowing one happened tells you the other did not.
- **A density is not a probability.** For a continuous variable `p(x)` can exceed 1; only integrals of it over intervals are probabilities (the book notes this explicitly).
- **The likelihood is not a distribution over the hypothesis.** It sums to 1 over data, not over `x`.
- **Priors do not disappear.** With little data the prior dominates; with a lot of data it is overwhelmed by the likelihood.

## Key takeaways

- Two rules: sum (marginalise) and product (chain). Everything else is derived.
- Bayes: `posterior ∝ likelihood × prior`, normalised by the evidence.
- Think in crowds of people: conditional probability is a fraction of a sub-crowd.
- Sequential Bayesian updating equals one batch update.

## Explain it yourself

1. A friend says "the test is 90% accurate, so a positive means 90% chance you are sick." Explain, using the crowd, why that is wrong in this example.
2. Why must a prior never be exactly 0 for a hypothesis that is merely unlikely?

## Further reading

- *Mathematics for Machine Learning*, Sec. 6.3 (PDF p.189-192); Bishop, *PRML*, Sec. 1.2.

## Mastery criteria

Given any small contingency table you can compute marginals, conditionals and the Bayes posterior, and explain the base-rate fallacy.
