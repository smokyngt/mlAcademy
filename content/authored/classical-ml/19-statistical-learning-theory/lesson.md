---
id: cml.19
title: "Statistical learning theory: generalisation bounds, VC dimension and Rademacher complexity"
summary: Turn "does the training error tell me anything about the future error?" into theorems you can compute, from Hoeffding and the union bound to VC dimension and Rademacher complexity, and see where the theory is tight, loose and silent.
minutes: 100
teaches: [ml.slt, ml.vc]
requires: [ml.generalization, prob.rv, stat.estimation]
sources:
  - "math/math for ML.pdf (Sec. 8.1.4 and Sec. 8.2 Empirical Risk Minimization, PDF p.263-269): empirical risk (8.6), expected/true risk (8.10), overfitting as under-estimated true risk, regularisation as a penalty on complexity. READ."
  - "math/math for ML.pdf (Sec. 8.3 Parameter estimation, PDF p.273-275): negative log-likelihood as a loss, MAP as a regularised estimate. READ."
---

Library coverage note: the library I read covers empirical risk minimisation and regularisation in words. It does **not** cover Hoeffding's inequality, the union-bound generalisation theorem, VC dimension, Rademacher complexity or no-free-lunch; those parts are written from general knowledge (standard textbook results) and every number is reproduced by the exercise code. Theorem constants differ between textbooks; treat the exact constants in the VC and Rademacher bounds as indicative and check a reference before quoting them.

## Where this fits

[[cml.03]] showed experimentally that a flexible model can fit the training data perfectly and still fail on new data, and that validation sets estimate the future error. A natural next question: **can we know, before looking at a test set, how large the gap between training and future error can be, and what controls it?** Statistical learning theory answers with inequalities of the form "with probability at least $1-\delta$, true error $\le$ training error $+$ a penalty that grows with model complexity and shrinks with data". That is the mathematical reason behind regularisation ([[cml.16]]), early stopping, margin methods ([[cml.08]]) and the rule of thumb "more parameters need more data". It also explains where the theory fails to describe modern deep networks. You need only expectation and variance ([[math.09]]) and the idea of an estimator ([[math.13]]).

## The idea in plain words

Imagine judging a restaurant by tasting one dish. If you taste ten dishes and all are great, you are fairly sure the kitchen is good. But what if you were allowed to choose, *after tasting*, among a million restaurants the one whose ten dishes looked best? With so many to choose from, some mediocre place will look perfect by luck. Learning is exactly this: the training set is the tasting, the model class is the set of restaurants you may pick from, and "empirical risk minimisation" is picking the one that looked best.

Therefore the honest worry has two factors: **how many tastings** (the sample size $n$) and **how many restaurants you could have chosen from** (the richness of the model class). With few, effectively distinct choices, a good tasting is trustworthy. With a limitless supply, nothing a tasting says can be trusted. The theory's job is to count "how many effectively different choices" in a clever way: by size for a finite class, by shattering (VC dimension) for an infinite class, or by the ability to fit random noise (Rademacher complexity).

## Definitions

Data $(x_i,y_i)$, $i=1..n$, are drawn i.i.d. from an unknown distribution $D$. A **hypothesis** $h$ maps inputs to predictions; $\mathcal H$ is the **hypothesis class**. The loss $\ell(h(x),y)\in[0,1]$; for classification the 0-1 loss $\mathbf 1[h(x)\ne y]$.

- **True (expected) risk** $R(h)=\mathbb E_{(x,y)\sim D}\,\ell(h(x),y)$. Cannot be computed; it is the quantity we care about (math for ML, eq. 8.10).
- **Empirical risk** $\hat R_n(h)=\frac1n\sum_{i=1}^n\ell(h(x_i),y_i)$ (eq. 8.6).
- **Generalisation gap** $R(h)-\hat R_n(h)$.
- **Empirical risk minimisation (ERM)**: $\hat h=\arg\min_{h\in\mathcal H}\hat R_n(h)$.

For a *fixed* $h$ chosen before seeing the data, $\hat R_n(h)$ is an average of $n$ i.i.d. variables with mean $R(h)$, so it is an unbiased estimate and the gap is small for large $n$. The trouble is that $\hat h$ is chosen *using* the data: $\hat R_n(\hat h)$ is a biased (optimistic) estimate of $R(\hat h)$. That optimism is overfitting (math for ML Sec. 8.2.3 describes it as the risk estimate from the training data under-estimating the true risk).

## Hoeffding's inequality

For independent random variables $Z_1,\dots,Z_n\in[0,1]$ with mean $\mu$,

$$
P\Big(\Big|\tfrac1n\sum_i Z_i-\mu\Big|\ge\varepsilon\Big)\le 2e^{-2n\varepsilon^2}.
$$

Apply it with $Z_i=\ell(h(x_i),y_i)$ for a fixed $h$: the chance that the empirical risk is off by $\varepsilon$ or more decays exponentially in $n\varepsilon^2$. (I state this standard result without proof; the proof bounds the moment generating function of a bounded variable, Hoeffding's lemma, then uses Markov's inequality.) For $n=100$ and $\varepsilon=0.1$ the bound is $2e^{-2}=0.271$: even 100 points leave a 27 % worst-case chance of being off by 10 points.

## Derivation: the union bound for a finite class

Let $|\mathcal H|=M$. We want a statement that holds for **all** $h\in\mathcal H$ simultaneously, in particular for the data-chosen $\hat h$.

1. For each fixed $h_j$, Hoeffding gives $P(|\hat R_n(h_j)-R(h_j)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2}$.
2. The **union bound**: the probability that at least one of $M$ events happens is at most the sum of their probabilities (events may overlap, which only helps). So

$$
P\big(\exists h\in\mathcal H:\ |\hat R_n(h)-R(h)|\ge\varepsilon\big)\le 2M e^{-2n\varepsilon^2}.
$$

3. Set the right-hand side equal to $\delta$ and solve for $\varepsilon$: $2Me^{-2n\varepsilon^2}=\delta \Rightarrow e^{-2n\varepsilon^2}=\frac{\delta}{2M}\Rightarrow 2n\varepsilon^2=\ln\frac{2M}{\delta}$:

$$
\boxed{\ \varepsilon=\sqrt{\frac{\ln(2M/\delta)}{2n}}\ }
$$

So with probability at least $1-\delta$, **every** $h\in\mathcal H$ satisfies $R(h)\le\hat R_n(h)+\varepsilon$, including $\hat h$. Solving for $n$ instead: $n\ge\frac{\ln(2M/\delta)}{2\varepsilon^2}$.

Consequences.

- The penalty grows only like $\sqrt{\ln M}$: squaring the number of hypotheses multiplies the penalty by $\sqrt2$, not by $M$. A class of $2^{b}$ hypotheses (a model with $b$ stored bits) costs $\sqrt{b\ln2/(2n)}$: *the number of bits needed to describe the model plays the role of complexity*.
- Excess risk: if $h^\star$ is the best in the class, then $R(\hat h)\le\hat R_n(\hat h)+\varepsilon\le\hat R_n(h^\star)+\varepsilon\le R(h^\star)+2\varepsilon$. ERM is within $2\varepsilon$ of the best hypothesis in the class.
- The bound is **uniform** over the class: it holds whatever the learning algorithm does, which is why it is loose for a specific algorithm.

## Infinite classes: shattering and VC dimension

Linear classifiers have infinitely many members ($M=\infty$), yet they cannot do everything. What matters is how many *different labelings of the sample* the class can produce.

- A set of $m$ points is **shattered** by $\mathcal H$ if for each of the $2^m$ labelings some $h\in\mathcal H$ realises it.
- The **VC dimension** $d_{VC}(\mathcal H)$ is the size of the largest set that *some* arrangement of points can be shattered by (infinite if there is no largest). To prove $d_{VC}=d$ you must exhibit one set of $d$ points that is shattered *and* show that no set of $d+1$ is.

Two worked cases, which the exercise checks by brute force.

- **Intervals on a line** (label 1 inside $(a,b)$). Any 2 points are shattered. For 3 points $x_1<x_2<x_3$ the labeling $(1,0,1)$ is impossible, since an interval containing $x_1$ and $x_3$ contains $x_2$. So $d_{VC}=2$. Brute force finds exactly $7$ of the $8$ labelings.
- **Half-planes in the plane** (linear classifiers with bias in $\mathbb R^2$). A triangle's 3 vertices are shattered (all $8$ labelings). For 4 points, brute force over the square gives $14$ of $16$ labelings: the "exclusive-or" labelings are missing. For 4 random points it never succeeds; the general fact (Radon's theorem: any 4 points in the plane can be split into two groups whose convex hulls intersect, so that labeling is impossible) gives $d_{VC}=3$. In $\mathbb R^p$ the VC dimension of linear classifiers with bias is $p+1$. Note that three *collinear* points are not shattered: the definition asks for the best arrangement, not every arrangement. The brute force in code is evidence for small cases, not a proof.

**Growth function and Sauer's lemma.** $\Pi_{\mathcal H}(n)$ = the maximum number of labelings of $n$ points. It equals $2^n$ for $n\le d$ and, by Sauer's lemma, is at most $(en/d)^d$ for $n\ge d$: it switches from exponential to polynomial. That polynomial plays the role of $M$ in the union bound.

**VC generalisation bound (statement).** For a class of VC dimension $d$ and 0-1 loss, with probability at least $1-\delta$, for all $h\in\mathcal H$,

$$
R(h)\le\hat R_n(h)+\sqrt{\frac{d\,\big(\ln(2n/d)+1\big)+\ln(4/\delta)}{n}}
$$

(one common form; constants vary). The structure is what matters: the penalty is roughly $\sqrt{d\ln n/n}$. **A rule of thumb: you want $n$ much larger than $d$.** I do not derive it here: the proof replaces $M$ by the growth function using a symmetrisation argument with a ghost sample.

## Rademacher complexity

VC dimension ignores the data distribution and is worst-case. Rademacher complexity measures how well the class can fit **pure noise** on the actual sample. Draw random signs $\sigma_i\in\{-1,+1\}$ with equal probability (Rademacher variables) and treat them as random labels. The **empirical Rademacher complexity** of $\mathcal H$ (functions with values $\pm1$) on points $x_1..x_n$ is

$$
\hat{\mathfrak R}_n(\mathcal H)=\mathbb E_\sigma\Big[\max_{h\in\mathcal H}\frac1n\sum_{i=1}^n\sigma_i\,h(x_i)\Big]\in[0,1].
$$

If $\mathcal H$ contains a hypothesis for every labeling, it matches every noise vector, the max is $1$, and the complexity is maximal. A single hypothesis gives about $0$, since random signs average to zero. For a finite class, Massart's lemma gives $\hat{\mathfrak R}_n\le\sqrt{2\ln M/n}$.

A typical generalisation bound (standard form; constants differ by source): with probability at least $1-\delta$, for all $h$,

$$
R(h)\le\hat R_n(h)+\mathfrak R_n(\mathcal H)+\sqrt{\frac{\ln(1/\delta)}{2n}}
$$

for $\pm1$-valued classes and 0-1 loss, where $\mathfrak R_n$ is the expectation of the empirical quantity over samples. Textbooks that define the complexity on the loss class instead write $2\mathfrak R_n$ with a different $\mathfrak R_n$, so always check the definition before comparing numbers. Rademacher complexity is *computable from data* (so it can be estimated, as below) and applies to real-valued classes such as norm-bounded linear models, where it scales like $\|w\|\cdot\max\|x\|/\sqrt n$. That norm-based control is the theoretical reason regularisation works even when the number of parameters is large.

## Worked example

All numbers are reproduced by the exercise code (seeded).

1. **Hoeffding.** $n=100,\ \varepsilon=0.1$: $2e^{-2}=0.2707$. To make $\varepsilon=0.05$ hold uniformly for $M=1000$ hypotheses at $\delta=0.05$ you need $n\ge\ln(40000)/(2\cdot0.05^2)=2120$.
2. **Union bound versus reality.** $M=200$ random linear classifiers in $\mathbb R^5$, labels from a hidden linear rule with noise, $\delta=0.05$, 200 training sets for each $n$:

| $n$ | union bound $\varepsilon$ | mean max gap over $\mathcal H$ | 95th percentile of the max gap | mean ERM gap |
|---|---|---|---|---|
| 50 | 0.300 | 0.175 | 0.235 | 0.072 |
| 200 | 0.150 | 0.087 | 0.123 | 0.026 |
| 1000 | 0.067 | 0.040 | 0.056 | 0.007 |

   The bound holds in every row (the 95th percentile is below it) and shrinks like $1/\sqrt n$ as the real gap does, but it is conservative: it covers the worst of $M$ hypotheses at 95 % confidence. The gap of the chosen $\hat h$ is several times smaller than the worst case, because ERM does not pick the hypothesis with the biggest gap.
3. **VC dimension by brute force.** Intervals: any 2 points shattered, 3 points give 7 labelings of 8, so $d_{VC}=2$. Half-planes: triangle gives 8 of 8; the square gives 14 of 16; so $d_{VC}=3$ in the plane.
4. **Rademacher complexity.** Thresholds on $n$ sorted points (both orientations, $2n+2$ hypotheses). Estimated $\hat{\mathfrak R}_n$ with 2000 sign vectors: $n=20$: $0.391$; $n=100$: $0.186$; $n=400$: $0.095$; Massart's bound gives $0.611,\ 0.326,\ 0.183$. The complexity falls like $1/\sqrt n$ (a class of VC dimension 2 behaves as $\sqrt{\ln n/n}$). The class of all $2^6$ labelings of 6 points has complexity exactly $1.0$.

## Implementation

```python
import numpy as np

def union_eps(n, M, delta):
    return np.sqrt(np.log(2 * M / delta) / (2 * n))        # uniform deviation width

def labelings_halfplanes(P, n_angles=720):
    labs = set()
    for th in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        w = np.array([np.cos(th), np.sin(th)])
        pr = P @ w                                         # (n,)
        s = np.sort(pr)
        for t in np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]]):
            labs.add(tuple((pr > t).astype(int)))
    return labs                                            # shattered iff len == 2**n

def rademacher(H, reps, seed):                             # H: (|H|, n) of +-1
    sig = np.random.default_rng(seed).choice([-1, 1], size=(reps, H.shape[1]))
    return (sig @ H.T).max(axis=1).mean() / H.shape[1]
```

## Shapes

- Predictions of $M$ hypotheses on $n$ points: $(n,M)$; empirical risks `mean(axis=0)`: $(M,)$.
- Point set $P$: $(n,2)$; labelings are tuples of length $n$, at most $2^n$ of them.
- Hypothesis matrix $H$: $(|\mathcal H|,n)$ of $\pm1$; sign matrix $\sigma$: $(\text{reps},n)$; product $\sigma H^\top$: $(\text{reps},|\mathcal H|)$.

## No free lunch, deep nets and structural risk minimisation

**No free lunch (statement).** Averaged over all possible target functions on a finite domain, no learning algorithm beats any other: for any algorithm, there are distributions on which it does badly. Learning therefore needs *assumptions* (a restricted class, a prior, smoothness, simplicity). The bounds above are the quantitative form of that: a small class with a good fit generalises, a class that can fit anything promises nothing.

**Structural risk minimisation (SRM).** Take nested classes $\mathcal H_1\subset\mathcal H_2\subset\dots$ of increasing capacity (polynomials of degree 1, 2, 3, ...; trees of growing depth). The training error falls with the index, the complexity penalty rises. Choose the class minimising *training error plus penalty*. This is what the bounds suggest, and it is the U-shape of [[cml.03]] in formulas. **Regularisation** is the continuous version: minimise $\hat R_n(h)+\lambda\,\Omega(h)$, e.g. $\|w\|^2$ (math for ML Sec. 8.2.3 presents this as a penalty on the complexity of the solution). It restricts the effective class to a ball $\|w\|\le B$, whose Rademacher complexity scales with $B$.

**Why deep nets challenge the bounds.** A modern network can have far more parameters than training points; its VC dimension grows at least with the number of weights, so the VC bound is vacuous (greater than 1). Yet such networks can generalise, and they can also fit random labels perfectly (shown in the literature, not in my sources), which means the class has high capacity and the *uniform* bound over the whole class cannot explain good behaviour. The bound is about the worst case over the class, while training with SGD from a small initialisation reaches particular, well-behaved solutions. Candidate explanations (implicit bias of SGD, norm- and margin-based bounds, flat minima, PAC-Bayes bounds, the "double descent" curve of test error as capacity passes the interpolation point) are active research; none is settled and I am not claiming any as the answer here. Practically: theory tells you which *knobs* matter (data size, capacity, norm, margin); validation data still decides.

## Common misconceptions

- "More parameters means a higher VC dimension." Not always: $\sin(ax)$ has one parameter and infinite VC dimension; conversely many strongly constrained parameters may add little.
- "VC dimension $d$ means any $d$ points can be shattered." Only some arrangement of $d$ points; three collinear points are not shattered by half-planes.
- "A bound that holds must be tight." Worst-case bounds are loose; the table above shows a factor of only about 1.2-1.3 between the bound and the 95th percentile even in this friendly finite case, the mean gap is about half the bound, and VC bounds are usually far looser.
- "The union bound says $M$ hypotheses cost a factor $M$." The cost is only $\sqrt{\ln M}$ in the width.
- "Low training error plus a small test set proves the model is good." The bounds assume the test points are fresh. Reusing a test set to choose among models re-introduces selection, the same mechanism as the million-restaurant story.
- "Theory says deep learning cannot work." The classical *uniform* bounds are vacuous there; that does not mean the models do not generalise.

## Key takeaways

- True risk $=$ empirical risk $+$ gap; ERM makes the gap optimistic because the hypothesis was chosen on the data.
- Finite class: gap $\le\sqrt{\ln(2M/\delta)/(2n)}$ for all $h$ at once (Hoeffding plus union bound).
- Infinite class: replace $\ln M$ by the VC dimension times $\ln n$; VC dimension is about shattering, and finding it needs a construction and an impossibility argument.
- Rademacher complexity measures the ability to fit noise and can be estimated from the sample.
- No free lunch: assumptions are unavoidable. SRM and regularisation trade fit against capacity. Deep networks sit outside what uniform bounds explain.

## Explain it yourself

1. Why is the empirical risk of the *selected* hypothesis an optimistic estimate of its true risk, while that of a hypothesis fixed in advance is unbiased?
2. Explain in your own words why the VC dimension of intervals on a line is 2 and not 3.

## Further reading

- math for ML, Ch. 8 (read here for empirical risk and regularisation).
- Written from general knowledge, not from the library: Shalev-Shwartz and Ben-David, *Understanding Machine Learning* (finite classes, VC, Rademacher, no free lunch); Mohri et al., *Foundations of Machine Learning*; Zhang et al., "Understanding deep learning requires rethinking generalization" (2017). Check these for exact theorem statements and constants.

## Mastery criteria

- Derive $\varepsilon=\sqrt{\ln(2M/\delta)/(2n)}$ from Hoeffding and the union bound, and compute it.
- Compare the bound with an experimental gap and explain why the bound is loose.
- Prove $d_{VC}=2$ for intervals and demonstrate $d_{VC}\ge3$, $<4$ for half-planes in the plane.
- Estimate a Rademacher complexity by Monte Carlo and interpret the values $0$ and $1$.
- State no-free-lunch and explain what SRM and regularisation do about it.
