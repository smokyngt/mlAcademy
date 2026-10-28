---
id: math.10
title: Information theory, entropy, cross-entropy and KL divergence
summary: Measure surprise in bits, see entropy as the shortest average message length, and derive why the cross-entropy loss is maximum likelihood and KL divergence is never negative.
minutes: 80
teaches: [info.entropy]
requires: [prob.rv, alg.exp-log]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.6 Information Theory (PDF p.68-78): information content h(x) = -log p(x), entropy, the 8-state example with code lengths, noiseless coding theorem, maximum entropy and the Gaussian, conditional entropy, relative entropy / KL, Jensen's inequality and KL >= 0, KL minimisation = maximum likelihood, mutual information. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/09-information-theory/docs/en.md (first 11000 characters): surprise table, entropy, cross-entropy as classification loss, cross-entropy = entropy + KL, mutual information, label smoothing, bits vs nats, perplexity. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 2.1-2.2 (PDF p.88-98): the Bernoulli/multinomial likelihood and the Lagrange-multiplier derivation mu_k = m_k / N that links frequencies to maximum likelihood. READ."
---

## Where this fits

Every time you train a classifier you call a function named `CrossEntropyLoss`. Every language-model paper reports *perplexity*. Variational autoencoders, knowledge distillation and RLHF all minimise a *KL divergence*. These look like three different ideas. They are one idea seen from three angles: **how many bits do you need to describe what really happened, if you built your code for a different belief about the world?**

This lesson comes right after probability (lesson 8) and the Gaussian (previous lesson): there you learned that the log-likelihood of independent data is a sum of $\ln p$ terms. Here we give those $-\ln p$ terms a meaning (surprise), average them (entropy), and compare two distributions (KL). The payoff is a clean answer to "why this loss?": cross-entropy is not a heuristic, it is maximum likelihood in disguise.

## The idea in plain words

- **Surprise.** Someone tells you "the sun rose today". You learn nothing. Someone tells you "you won the lottery". You learn a lot. Information is the *degree of surprise*: rare events carry a lot, certain events carry none. Two independent surprises should simply add up (being told two unrelated news items is worth the sum of each). The only function that turns "multiply probabilities" into "add surprises" is a **logarithm**, and the minus sign makes rare events big: surprise $=-\log p$.
- **Entropy = average surprise = shortest average message.** Imagine you must send the outcome of a repeated experiment over a wire made of 0s and 1s. If all eight outcomes are equally likely, you need 3 bits for each. But if one outcome happens half the time, give it a 1-bit name; give the next one a 2-bit name, and so on, and keep the long names for the rare cases. The *average* message gets shorter. Entropy is that unavoidable minimum average length: you cannot compress below it.
- **Cross-entropy = what you pay when your belief is wrong.** You design the code book for what you *believe* ($q$), but nature produces outcomes according to the truth ($p$). Your code names are the wrong lengths for the real frequencies, so the average message is longer. That average is the cross-entropy.
- **KL divergence = the price of being wrong.** The extra length compared with the best possible code. It is zero only when your belief is exactly the truth, never negative, and not symmetric: being confident about the wrong thing and being vague about the right thing are different mistakes.
- **Perplexity** turns an average surprise back into a count: "as confused as if choosing uniformly among this many options".

## Definitions

### Information content and entropy

For a discrete variable $x$ with probability $p(x)$ (Bishop 1.92, 1.93):

$$
h(x)=-\log p(x),\qquad H[p]=\mathbb E_{x\sim p}[h(x)]=-\sum_x p(x)\log p(x).
$$

Base-2 logarithms give **bits**, natural logarithms give **nats**; $1\text{ nat}=1/\ln 2\approx1.443$ bits. Machine-learning libraries use nats. We use the convention $0\log0=0$.

The first derivation Bishop gives for the logarithm: if $x$ and $y$ are independent then $p(x,y)=p(x)p(y)$ and information should add, $h(x,y)=h(x)+h(y)$. A function turning products into sums is a log.

Properties: $H\ge0$, with $H=0$ exactly when one outcome has probability 1. For $M$ states $H\le\ln M$, with equality for the uniform distribution (Bishop derives this with a Lagrange multiplier, a technique you will meet in lesson 12). Among continuous densities with a given mean and variance, the Gaussian has the largest (differential) entropy, $\tfrac12(1+\ln 2\pi\sigma^2)$, which is one more reason it is the default noise model.

### Example: the 8-state variable (Bishop, PDF p.70)

Eight states with probabilities $(\tfrac12,\tfrac14,\tfrac18,\tfrac1{16},\tfrac1{64},\tfrac1{64},\tfrac1{64},\tfrac1{64})$. Entropy:

$$
H=\tfrac12\cdot1+\tfrac14\cdot2+\tfrac18\cdot3+\tfrac1{16}\cdot4+4\cdot\tfrac1{64}\cdot6=0.5+0.5+0.375+0.25+0.375=2\text{ bits.}
$$

A flat 3-bit code costs 3 bits per symbol. Use the code words $0,\,10,\,110,\,1110,\,111100,\,111101,\,111110,\,111111$ of lengths $1,2,3,4,6,6,6,6$: the average length is again $2$ bits. Each length equals $-\log_2 p$ exactly. No shorter average code can exist (Shannon's noiseless coding theorem), and the code is *prefix-free*, so a bit stream like `11001110` decodes uniquely (c, a, d).

### Cross-entropy

If the truth is $p$ but we design the code for $q$, a state $x$ gets a word of length $-\log q(x)$, and since states occur with frequency $p(x)$ the average length is

$$
H(p,q)=-\sum_x p(x)\log q(x).
$$

### KL divergence and its decomposition

The extra cost over the optimal code (Bishop 1.113):

$$
\mathrm{KL}(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}=H(p,q)-H(p).
$$

So **cross-entropy = entropy + KL**. Because $H(p)$ does not depend on the model $q$, minimising cross-entropy over $q$ is the same as minimising the KL divergence.

### Gibbs' inequality: $\mathrm{KL}\ge0$

Claim: $\mathrm{KL}(p\|q)\ge0$, with equality iff $p=q$.

*Step 1, a tangent-line fact.* For all $x>0$, $\ln x\le x-1$, with equality only at $x=1$. Reason: $g(x)=x-1-\ln x$ has $g'(x)=1-1/x$, which is negative for $x<1$ and positive for $x>1$, so $g$ has its unique minimum at $x=1$ where $g(1)=0$.

*Step 2.* Apply it to $x=q(i)/p(i)$ for every state with $p(i)>0$:

$$
-\mathrm{KL}(p\|q)=\sum_i p(i)\ln\frac{q(i)}{p(i)}\ \le\ \sum_i p(i)\Big(\frac{q(i)}{p(i)}-1\Big)=\sum_i q(i)-\sum_i p(i)\ \le\ 1-1=0 .
$$

(The sum of $q$ over the states where $p>0$ is at most 1.) Hence $\mathrm{KL}\ge0$. Equality needs $q(i)/p(i)=1$ for every $i$, that is $p=q$. Bishop proves the same fact with Jensen's inequality for the convex function $-\ln x$; the two proofs are the same idea (a convex curve lies above its tangent). Since $\mathrm{KL}\ge0$, $H(p,q)\ge H(p)$: no code built for the wrong distribution can beat the right one.

### Asymmetry

In general $\mathrm{KL}(p\|q)\ne\mathrm{KL}(q\|p)$, so KL is not a distance. It is large when $q$ gives (almost) zero probability to something that $p$ says is likely: you are then infinitely surprised. This is why the direction matters in practice (forward versus reverse KL in variational inference).

### From cross-entropy to the classification loss, and to maximum likelihood

Let the label of one example be a one-hot vector $p$ (probability 1 on the true class $c$) and the model output $q=\operatorname{softmax}(\mathbf z)$. Then

$$
H(p,q)=-\sum_k p_k\ln q_k=-\ln q_c .
$$

Over $N$ independent examples with true classes $c_n$, the likelihood is $\prod_n q^{(n)}_{c_n}$ and the negative log-likelihood is $-\sum_n\ln q^{(n)}_{c_n}$: exactly $N$ times the average cross-entropy. So **minimising the cross-entropy loss = maximising the likelihood of the labels**.

The same conclusion from the other direction (Bishop, PDF p.77): let $p$ be the unknown data distribution and $q(x\mid\theta)$ the model. Replace the expectation under $p$ by an average over samples:

$$
\mathrm{KL}(p\|q)\approx\frac1N\sum_{n=1}^N\big\{-\ln q(x_n\mid\theta)+\ln p(x_n)\big\}.
$$

The second term does not contain $\theta$, and the first is the negative log-likelihood. **Minimising KL = maximum likelihood.** The Bernoulli/multinomial case closes the loop: maximising $\sum_k m_k\ln\mu_k$ with $\sum_k\mu_k=1$ (Lagrange multiplier) gives $\mu_k=m_k/N$, the observed frequencies (Bishop 2.33), the unique minimiser of $\mathrm{KL}(\hat p\|q)$.

### Mutual information (briefly)

For two variables, how far is the joint $p(x,y)$ from the product of marginals $p(x)p(y)$ (what independence would give)? Bishop 1.120:

$$
I[x,y]=\mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)=H[x]-H[x\mid y]=H[x]+H[y]-H[x,y]\ \ge 0 .
$$

It is zero iff $x$ and $y$ are independent, and it is the reduction in uncertainty about $x$ once you are told $y$. Unlike correlation it detects any kind of dependence, which is why it is used to rank features.

### Perplexity

Perplexity is the exponential of the average cross-entropy per token: $\mathrm{PPL}=\exp\big(-\frac1N\sum_n\ln q(x_n)\big)$ in nats (or $2^{\text{bits}}$). A model that spreads probability uniformly over $K$ options has perplexity exactly $K$, so a perplexity of $50$ means "as uncertain as a fair choice among 50 tokens". Lower is better.

## Worked example

Every number is reproduced by the exercise code.

**1. Surprise.** A fair coin side: $1$ bit. A die face: $-\log_2(1/6)=2.585$ bits. A $1/1000$ event: $9.966$ bits.

**2. Entropy and code length.** The 8-state distribution above: $H=2$ bits. The uniform one over 8 states: $3$ bits. A coin with $P(\text{heads})=0.99$: $0.0808$ bits.

**3. Cross-entropy and KL.** $p=(\tfrac12,\tfrac14,\tfrac18,\tfrac18)$ and $q$ uniform over 4 outcomes. $H(p)=1.75$ bits. A code built for $q$ uses 2 bits for every outcome, so $H(p,q)=2$ bits and $\mathrm{KL}(p\|q)=2-1.75=0.25$ bits. With the sharper wrong guess $q'=(0.7,0.1,0.1,0.1)$: $H(p,q')=1.918$ bits, $\mathrm{KL}(p\|q')=0.168$ bits.

**4. Asymmetry.** $p=(0.9,0.1)$, $q=(0.5,0.5)$. $\mathrm{KL}(p\|q)=0.531$ bits but $\mathrm{KL}(q\|p)=0.737$ bits.

**5. Softmax cross-entropy.** Logits $\mathbf z=(2,1,0.1)$ give probabilities $(0.659,0.242,0.099)$ (shift by $\max z$ before exponentiating). If the true class is the first, the loss is $-\ln0.659=0.417$ nats $=0.602$ bits.

**6. Perplexity.** A model assigns probabilities $(0.5,0.25,0.125,0.125)$ to the four correct tokens. Average NLL $=1.5596$ nats, perplexity $=e^{1.5596}=4.757$. A uniform model over 50 tokens has perplexity exactly $50$.

**7. Mutual information.** Joint over two binary variables: rows $(0.40,0.10)$ and $(0.05,0.45)$. Marginals $(0.5,0.5)$ and $(0.45,0.55)$. $H[x]=1$, $H[y]=0.9928$, $H[x,y]=1.5955$, so $I=1+0.9928-1.5955=0.3973$ bits, which equals $\mathrm{KL}(\text{joint}\,\|\,\text{product})$.

## Implementation

```python
import numpy as np

def entropy(p, base=2):
    p = np.asarray(p, float); p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(base)

def cross_entropy(p, q, base=2):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = p > 0
    return -(p[m] * np.log(q[m])).sum() / np.log(base)

def kl(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

def softmax_xent(z, c):                       # z: (K,), c: true class index
    z = z - z.max()                           # numerical stability
    return -(z[c] - np.log(np.exp(z).sum()))   # -log softmax(z)[c], in nats

p = np.array([.5, .25, .125, .125]); q = np.ones(4) / 4
print(entropy(p), cross_entropy(p, q), kl(p, q))   # 1.75 2.0 0.25
```

## Shapes

A distribution over $K$ classes is a vector `(K,)` summing to 1. A batch of model outputs is `(B, K)` (rows sum to 1 after softmax), integer labels are `(B,)`, and the batch cross-entropy is `-log(q[arange(B), labels]).mean()`, a scalar. A joint distribution of two variables is a matrix `(Kx, Ky)`; marginals are its row and column sums.

## Common misconceptions

- **KL is a distance.** It is not symmetric and violates the triangle inequality.
- **Cross-entropy is below entropy for a good model.** Never: $H(p,q)\ge H(p)$, with equality only for $q=p$. A loss of zero for one-hot labels is reachable only by pushing logits to infinity.
- **Bits and nats are different quantities.** Same quantity, different unit (factor $\ln2$).
- **$\log0$ is harmless.** If $q(x)=0$ where $p(x)>0$, cross-entropy and KL are infinite; clip probabilities or work in log space.
- **Mutual information is correlation.** Correlation sees only linear dependence; mutual information sees all.

## Key takeaways

- Surprise is $-\log p$; entropy $H=-\sum p\log p$ is the average surprise and the optimal average code length.
- $H(p,q)=H(p)+\mathrm{KL}(p\|q)$, and $\mathrm{KL}\ge0$ (proof from $\ln x\le x-1$).
- With one-hot labels the cross-entropy is $-\ln$ of the probability given to the true class; minimising it is maximum likelihood, and equivalently minimising KL to the data distribution.
- Mutual information is the KL between the joint and the product of marginals; perplexity is $\exp$ of the cross-entropy.

## Explain it yourself

1. Explain to a friend, with a message-length argument, why a skewed distribution has lower entropy than a uniform one.
2. Why does minimising cross-entropy for a model give the same answer as minimising KL to the data distribution, even though we cannot compute the data distribution?

## Further reading

- Bishop, *PRML*, Sec. 1.6 (PDF p.68-78), including the entropy-as-disorder argument on PDF p.70-72 and the differential entropy of the Gaussian.
- *Mathematics for Machine Learning*, remark on divergences and information geometry at the end of Sec. 6.4 (PDF p.203).

## Mastery criteria

You can compute entropy, cross-entropy and KL for small distributions by hand, explain them as code lengths, prove $\mathrm{KL}\ge0$ from $\ln x\le x-1$, show that softmax cross-entropy is a negative log-likelihood, and convert a loss in nats to bits and to perplexity.
