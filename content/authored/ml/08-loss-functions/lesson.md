---
id: f2n.08
title: "Loss functions: MSE, cross-entropy and softmax"
summary: Where losses come from (maximum likelihood), and the clean gradients p - y that make classification training stable.
minutes: 65
teaches: [ml.loss]
requires: [alg.exp-log, alg.summation]
sources:
  - math/math for ML.pdf (Ch. 8 When Models Meet Data, Sec. 8.2 Empirical risk minimisation and 8.3 Parameter estimation, PDF p.257)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 1.2 Probability theory and Sec. 4.3.2 Logistic regression, PDF p.32 and p.223
  - ai-engineering-from-scratch/phases/03-deep-learning-core/05-loss-functions/docs/en.md
---

## Learning objectives

- Derive mean squared error from a Gaussian noise model and binary cross-entropy from a Bernoulli model.
- Define softmax and categorical cross-entropy, and compute them stably with log-sum-exp.
- Derive $\partial L/\partial z = p - y$ for sigmoid+BCE and softmax+CE.
- State the shapes for a batch of $B$ examples and $K$ classes.

## Prerequisites

Logs and sums ([[f2n.01]]); gradient descent motivation ([[f2n.07]]).

## Intuition

A loss is a number measuring "how wrong". We choose it so that the best parameters are the ones that make the observed data *most probable* (maximum likelihood). Different assumptions about how the data is generated give different losses: continuous targets with additive noise give squared error; yes/no outcomes give cross-entropy.

## Maximum likelihood in one paragraph

Pick a model $p_\theta(y\mid x)$. For independent examples the likelihood is $\prod_ip_\theta(y_i\mid x_i)$. Maximising it is equivalent to minimising the **negative log-likelihood**

$$
\mathrm{NLL}(\theta)=-\sum_i\ln p_\theta(y_i\mid x_i)
$$

(taking logs turns the product into a sum, lesson 1, and $-\ln$ is decreasing).

## Derivation 1: MSE from a Gaussian

Assume $y=f_\theta(x)+\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. Then $p(y\mid x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\big(-\frac{(y-f_\theta(x))^2}{2\sigma^2}\big)$ and

$$
-\ln p(y_i\mid x_i)=\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}+\ln(\sigma\sqrt{2\pi}).
$$

The second term does not depend on $\theta$, so minimising NLL over $\theta$ is exactly minimising $\sum_i(y_i-f_\theta(x_i))^2$. Mean squared error is therefore "maximum likelihood under Gaussian noise". It penalises large errors quadratically, so it is sensitive to outliers.

## Derivation 2: binary cross-entropy from a Bernoulli

Targets $y\in\{0,1\}$, model output $p=\sigma(z)$ interpreted as $P(y=1)$. The Bernoulli likelihood is $p^y(1-p)^{1-y}$, so

$$
L=-\big[y\ln p+(1-y)\ln(1-p)\big].
$$

**Gradient with respect to the logit $z$.** By the chain rule (lesson 6), $\frac{\partial L}{\partial z}=\frac{\partial L}{\partial p}\frac{dp}{dz}$ with

$$
\frac{\partial L}{\partial p}=-\frac yp+\frac{1-y}{1-p},\qquad\frac{dp}{dz}=p(1-p).
$$

$$
\frac{\partial L}{\partial z}=\Big(-\frac yp+\frac{1-y}{1-p}\Big)p(1-p)=-y(1-p)+(1-y)p=p-y .
$$

The sigmoid's derivative cancels the denominators, leaving the beautifully simple **prediction minus target**. Unlike MSE+sigmoid (whose gradient carries a $p(1-p)$ factor that vanishes when the model is confidently wrong), cross-entropy keeps a strong gradient exactly when the error is large.

## Derivation 3: softmax and categorical cross-entropy

For $K$ classes with logits $z\in\mathbb{R}^K$,

$$
p_k=\operatorname{softmax}(z)_k=\frac{e^{z_k}}{\sum_je^{z_j}}.
$$

Softmax outputs are positive and sum to 1, and are invariant to adding a constant $c$ to every logit (the $e^c$ cancels). That invariance is the key to numerical stability: subtract $\max_jz_j$ first. With true class $y$, the loss is $L=-\ln p_y$. Substituting,

$$
L=-z_y+\ln\sum_je^{z_j}=-z_y+\operatorname{logsumexp}(z).
$$

Differentiating with respect to $z_k$ (using $\partial\,\mathrm{logsumexp}/\partial z_k=p_k$):

$$
\frac{\partial L}{\partial z_k}=p_k-\mathbb{1}[k=y]\quad\Longleftrightarrow\quad\nabla_zL=p-\text{onehot}(y).
$$

Same shape of answer as the binary case. Never compute $\ln(\operatorname{softmax}(z))$ by taking `log` of a softmax: use $z_y-\operatorname{logsumexp}(z)$ (this is what `log_softmax` does).

## Worked example

Logits $z=(2,1,0.1)$, true class $y=0$. Exponentials: $e^2=7.389$, $e^1=2.718$, $e^{0.1}=1.105$, sum $=11.212$. So $p=(0.659,0.242,0.099)$ and

$$
L=-\ln0.659=0.417,\qquad\nabla_zL=p-(1,0,0)=(-0.341,\,0.242,\,0.099).
$$

The gradient sums to zero (adding a constant to all logits cannot change the loss), and it pushes the true logit up and the others down.

## Shapes for a batch

Logits $Z\in\mathbb{R}^{B\times K}$, integer labels $y\in\{0..K-1\}^B$.

| Quantity | Shape |
|---|---|
| $P=\operatorname{softmax}(Z)$ (row-wise) | $B\times K$ |
| $L=-\frac1B\sum_b\ln P_{b,y_b}$ | scalar |
| $\nabla_ZL=\frac1B(P-Y_{\text{onehot}})$ | $B\times K$ |

The $\frac1B$ comes from the mean over the batch; forget it and your effective learning rate scales with the batch size.

## Implementation

```python
import numpy as np

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)      # stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Z, y):
    m = Z.max(axis=1, keepdims=True)
    logsumexp = (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True))).ravel()
    return np.mean(logsumexp - Z[np.arange(len(y)), y])
```

## Common misconceptions

- **Cross-entropy is not "just another distance".** It is the NLL of a categorical model, and it equals $H(q,p)=-\sum_kq_k\ln p_k$ for a one-hot target $q$.
- **Do not apply softmax and then cross-entropy in two numerically naive steps.** `log(softmax(z))` underflows to `-inf`. Use the fused log-sum-exp form.
- **MSE is not wrong for classification because it is "less accurate"**; it is a poor fit because of the vanishing-gradient behaviour above.
- **A loss of 0.693 on a binary task** ($=\ln2$) means you are predicting 50/50: the model has learned nothing yet. Likewise $\ln K$ for $K$ classes. Memorise this sanity check.

## Key takeaways

- A loss is a negative log-likelihood under a chosen noise model: Gaussian gives MSE, Bernoulli/categorical gives cross-entropy.
- Sigmoid + BCE and softmax + CE both give the clean gradient `p - y`.
- Compute with log-sum-exp: never `log(softmax(z))`.
- A fresh `K`-class model should start near loss `ln K`.

## Explain it yourself

Answer these out loud or in writing, without looking back. If you get stuck, that is the exact spot to re-read.

1. Derive `dL/dz = p - y` for sigmoid + binary cross-entropy without looking.
2. Why does adding a constant to every logit leave softmax unchanged, and why does that make it stable?

## Further reading

- *Mathematics for Machine Learning*, Ch. 8 (empirical risk, maximum likelihood), PDF p.257.
- Bishop, *PRML*, Sec. 4.3.2-4.3.3 (logistic regression and cross-entropy), PDF p.223.
- Source library: `ai-engineering-from-scratch` phase 03 lesson 05.

## Mastery criteria

You can derive $p-y$ for both sigmoid+BCE and softmax+CE on paper, name the expected initial loss ($\ln K$), and implement a stable cross-entropy that survives logits of 1000.
