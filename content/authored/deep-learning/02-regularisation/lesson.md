---
id: dl.02
title: "Regularisation: weight decay, dropout, early stopping, augmentation"
summary: Why a network that fits the training set perfectly generalises badly, and the five standard ways to close the gap, with the Gaussian-prior view of weight decay and the derivation of inverted dropout.
minutes: 80
teaches: [dl.weight-decay, dl.dropout, dl.early-stopping]
requires: [nn.training, ml.ridge, ml.generalization, prob.rv]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 5.5 Regularization in Neural Networks, PDF p.276-280 (weight decay as negative log of a zero-mean Gaussian prior, eq. 5.112; consistent priors; early stopping, Sec. 5.5.2) and PDF p.285-286 (Sec. 5.5.5 training with transformed data equals a regulariser, eq. 5.131). READ; tangent propagation and soft weight sharing (p.281-284, 289-292) were not read."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/docs/en.md (dropout with inverted scaling, weight decay values, early stopping with patience, data augmentation, when to apply what). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/code/main.py (Dropout class with forward/backward and the 1/(1-p) scaling, l2_regularization signature). READ (first 35 lines)."
---

## Where this fits

A network with enough parameters can fit any training set, including pure noise. In [[f2n.11]] you saw a small network separate the half-moons; give the same network a tiny, noisy training set and it will cheerfully reach 100% training accuracy while doing noticeably worse on new data. That train/test gap is overfitting, and it is the central practical problem of deep learning: the loss you can minimise (training) is not the loss you care about (new data).

Regularisation is everything that narrows the gap. It comes in five families, and a real model usually stacks several of them:

1. penalise large weights (weight decay, L2; L1 for sparsity),
2. inject noise into the network (dropout),
3. stop before the model has memorised (early stopping),
4. enlarge the data with label-preserving transformations (augmentation),
5. constrain the architecture (normalisation in [[dl.03]], weight sharing in convolutional networks, covered later).

This lesson derives the first four and measures them on a deliberately small problem.

## The idea in plain words

Think of a student preparing for an exam with only 40 practice questions. One strategy is to memorise the 40 answers; the student then scores 100% on practice and poorly on the real exam. Each regulariser takes away a way of memorising:

- **Weight decay** is a rule that says "explanations must be simple". A memorising network needs very large, finely tuned weights to carve a boundary around each individual point; shrinking the weights forces smoother boundaries.
- **Dropout** is studying with randomly missing notes. If any single note can vanish, no answer can depend on one fragile memory, so the student spreads knowledge over many redundant routes.
- **Early stopping** is putting the book down when the practice-exam score (on questions not used for studying) stops improving.
- **Data augmentation** is inventing more practice questions by rewording the old ones (nudge a point, flip an image) so that memorising exact inputs does not pay.

All of them trade a little training accuracy for better behaviour on data the model has not seen. Too much regularisation is underfitting: the student who is only allowed one note per subject.

## Formal definitions

Let $L(\theta)$ be the training loss and $\theta$ the weights (biases are usually excluded).

**L2 / weight decay.** Minimise $L(\theta)+\tfrac{\lambda}{2}\|\theta\|_2^2$. The gradient gains the term $\lambda\theta$.

**L1.** Minimise $L(\theta)+\lambda\|\theta\|_1$. The gradient gains $\lambda\,\mathrm{sign}(\theta)$.

**Dropout (inverted).** During training, for a layer output $a\in\mathbb{R}^n$ draw an independent mask $m_i\sim\mathrm{Bernoulli}(1-p)$ and output

$$
\tilde a=\frac{m\odot a}{1-p}.
$$

At evaluation time output $a$ unchanged.

**Early stopping.** Track validation loss each epoch; keep the parameters from the epoch with the lowest validation loss; stop after it has not improved for a "patience" window.

**Augmentation.** Replace each training pair $(x,y)$ by $(s(x,\xi),y)$ where $s$ is a random label-preserving transformation with parameter $\xi$.

## Derivation

### Weight decay is a Gaussian prior

Model the targets as $y=f_\theta(x)+\epsilon$ with Gaussian noise of variance $\sigma^2$, and put a zero-mean Gaussian prior with variance $\tau^2$ on every weight, $p(\theta)=\mathcal N(0,\tau^2 I)$. The maximum a posteriori (MAP) estimate maximises $\log p(\mathcal D\mid\theta)+\log p(\theta)$, i.e. minimises

$$
\frac{1}{2\sigma^2}\sum_n\big(y_n-f_\theta(x_n)\big)^2+\frac{1}{2\tau^2}\|\theta\|_2^2+\text{const}.
$$

Multiply through by $\sigma^2$: this is the sum-of-squares error plus $\tfrac{\lambda}{2}\|\theta\|^2$ with

$$
\lambda=\frac{\sigma^2}{\tau^2}.
$$

Bishop states exactly this interpretation (Sec. 5.5, eq. 5.112): the quadratic regulariser is the negative log of a zero-mean Gaussian prior. Reading $\lambda$ as a noise-to-prior-width ratio is useful: a tight prior (small $\tau$) means strong decay. Replace the Gaussian by a Laplace density $\propto e^{-|\theta|/b}$ and the same calculation gives the L1 penalty with $\lambda\propto1/b$; its sharp peak at zero is why L1 produces exact zeros.

### How decay acts in the update rule

With gradient descent on $L+\tfrac\lambda2\|\theta\|^2$:

$$
\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)=(1-\eta\lambda)\,\theta-\eta\nabla L .
$$

Every step first shrinks the weights by the factor $1-\eta\lambda$, then applies the ordinary gradient step; hence the name weight decay. (With Adam this equivalence breaks, see [[dl.01]].)

A one-dimensional example makes the effect concrete. Minimise $\tfrac12(w-3)^2$, whose unconstrained optimum is $w=3$. With L2 strength $\lambda=0.5$ the stationary condition $(w-3)+0.5\,w=0$ gives $w^\star=3/1.5=2$. With L1 strength $0.5$ it is $(w-3)+0.5\,\mathrm{sign}(w)=0$, giving $w^\star=2.5$; and for any $\lambda\ge3$ the L1 solution is exactly $0$ (soft thresholding: $w^\star=\mathrm{sign}(3)\max(3-\lambda,0)$). L2 shrinks proportionally and never reaches zero; L1 subtracts a constant and can kill a weight. Both values were confirmed by running gradient descent.

### Inverted dropout: why divide by $1-p$

Take one unit with activation $a$ and mask $m\sim\mathrm{Bernoulli}(1-p)$, so $\mathbb{E}[m]=1-p$.

Without scaling, the training-time output $ma$ has expectation $(1-p)a$. If at test time we feed the full $a$, the next layer sees inputs $1/(1-p)$ times larger than anything it saw in training. The old fix multiplied the test-time weights by $1-p$. The inverted version moves the correction to training: divide by $1-p$,

$$
\mathbb{E}[\tilde a]=\mathbb{E}\!\left[\frac{m\,a}{1-p}\right]=\frac{a}{1-p}\,\mathbb{E}[m]=a .
$$

The expected activation is unchanged, so the evaluation network is simply the network with no dropout, with nothing to rescale. The price is extra variance:

$$
\operatorname{Var}(\tilde a)=\frac{a^2}{(1-p)^2}\operatorname{Var}(m)=\frac{a^2}{(1-p)^2}\,p(1-p)=a^2\,\frac{p}{1-p}.
$$

For $p=0.5$ the variance equals $a^2$ (noise as large as the signal); for $p=0.1$ it is $a^2/9$. This noise is the regularising effect.

**Backward pass.** The forward map is elementwise, $\tilde a_i=c_i\,a_i$ with the fixed (per-batch) coefficient $c_i=m_i/(1-p)$. Hence

$$
\frac{\partial L}{\partial a_i}=c_i\,\frac{\partial L}{\partial\tilde a_i}=\frac{m_i}{1-p}\frac{\partial L}{\partial\tilde a_i}.
$$

The same mask and scale must be reused in backward: dropped units receive zero gradient, kept units receive the incoming gradient multiplied by $1/(1-p)$. Resampling the mask in backward is a classic bug.

An interpretation, with a caveat: a network with $n$ droppable units contains $2^n$ sub-networks that share weights, and dropout trains a random one per step; evaluating without dropout approximately averages them. This is a heuristic picture (the source lesson presents it as an ensemble interpretation), not an exact equality for non-linear networks.

### Early stopping

Training reduces training error monotonically, but error on a held-out validation set typically falls and then rises as the network starts fitting noise (Bishop, Sec. 5.5.2, Fig. 5.12). Stopping at the validation minimum gives a network of smaller effective complexity: the number of effectively free parameters starts small and grows during training. In practice you checkpoint the best model and keep training for a patience window before giving up. The validation set must be separate from the final test set, otherwise the stopping epoch itself leaks test information.

### Data augmentation is a regulariser

Bishop's Sec. 5.5.5 shows why training on randomly perturbed copies is not just "more data". Take a transformation $s(x,\xi)$ with $s(x,0)=x$ and a zero-mean small parameter $\xi$ with variance $\lambda$. Expand $y(s(x,\xi))$ to second order in $\xi$ and average the squared error: the first-order term vanishes because $\mathbb{E}[\xi]=0$, leaving

$$
\widetilde E=E+\lambda\,\Omega ,
$$

(eq. 5.131 in the book), where $\Omega$ penalises the sensitivity of the model output to the transformation. Augmentation is therefore an explicit smoothness penalty in disguise: the network is pushed to be (nearly) invariant along the transformation direction. Adding Gaussian noise to inputs, as in the experiment below, is the simplest instance.

## Worked example

Set-up: noisy half-moons (noise 0.35) with only **40 training points**; a $2\to128\to128\to2$ ReLU network, He initialisation, Adam with $\eta=0.01$, 1500 full-batch epochs; 4000 fresh test points. All numbers are averages over 5 seeds (a seed changes the training set and the initialisation). Accuracy is measured with dropout switched off.

| Method | Train acc | Test acc | Gap | Test loss |
|---|---|---|---|---|
| none | 1.000 | 0.831 | 0.169 | 4.91 |
| L2, $\lambda=0.01$ | 0.960 | 0.855 | 0.105 | 0.469 |
| L2, $\lambda=0.1$ | 0.875 | 0.830 | 0.045 | 0.391 |
| L1, $\lambda=0.01$ | 0.920 | 0.847 | 0.074 | 0.502 |
| dropout 0.3 | 1.000 | 0.838 | 0.162 | 5.50 |
| dropout 0.5 | 1.000 | 0.838 | 0.162 | 5.22 |
| input noise 0.2 (augmentation) | 0.975 | 0.850 | 0.125 | 0.464 |
| early stopping (patience on a 200-point validation set) | 0.870 | 0.825 | 0.045 | 0.463 |
| dropout 0.3 + noise 0.2 | 0.960 | 0.860 | 0.100 | 0.472 |

What the numbers say, and do not say:

- The unregularised network is overconfident: train loss $\approx0$ yet test loss $4.9$. The test *accuracy* of 83% hides how wrong its confident mistakes are.
- L2 and noise augmentation cut the test loss by a factor of about 10 and raise accuracy by 2 to 3 points. A stronger penalty ($\lambda=0.1$) closes the gap the most (0.045) but by lowering the training accuracy, not by raising the test accuracy: it is slightly underfitting.
- **Dropout alone did almost nothing here.** Gap $0.162$ against $0.169$, test loss slightly worse. With 40 points and 128 units per layer the network still memorises through the units that survive each mask, and 1500 full-batch steps are enough to average the noise out. Dropout is known to help most in wide networks trained on large data with many epochs; it is not a universal cure, and on this toy it is the weakest tool. Combined with augmentation it gave the best test accuracy (0.860).
- Early stopping stopped after about 8 epochs on average and gave the smallest gap but not the best accuracy.
- The standard deviation of test accuracy across the 5 seeds is between 0.011 and 0.022 for every row. Gains of about 2 points over the baseline (0.831 to 0.855 or 0.860) are therefore only 1 to 2 standard deviations, and rows that differ by one point (L2 0.01 versus L1 0.01) cannot be ranked. The large and consistent effect is on test loss, not accuracy.

The mechanics are checkable in the exercise: the dropout mask has the right mean and variance, the backward pass reuses it, and a ridge solution is a MAP estimate whose gradient vanishes.

## Implementation

```python
import numpy as np

class Dropout:
    def __init__(self, p, seed=0):
        self.p, self.rng = p, np.random.default_rng(seed)
    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)   # 0 or 1/(1-p)
        return x * self.mask
    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask

def decay_step(W, g, lr, lam):
    return (1 - lr * lam) * W - lr * g          # weight decay + gradient step
```

## Shapes

For a hidden layer with batch $B$ and width $n$: activations $(B,n)$, mask $(B,n)$ (independent per example and per unit, resampled every forward pass), gradient $(B,n)$. In a convolutional network the mask is often shared over spatial positions; in a transformer it is applied to attention weights and to residual-branch outputs. Weight decay acts on weight matrices $(n_{in},n_{out})$ and is normally skipped for biases and normalisation scales.

## Common misconceptions

- **"Dropout is applied at test time too."** No: evaluation uses the full network ($\texttt{model.eval()}$ in PyTorch). Forgetting this makes predictions noisy and non-deterministic.
- **"Dropping with probability $p$ keeps a fraction $p$."** $p$ is the drop probability; the keep probability is $1-p$.
- **"Weight decay and L2 penalty are always the same."** They coincide for SGD, not for Adam ([[dl.01]]).
- **"More regularisation is always safer."** Beyond the sweet spot the model underfits; in the table, $\lambda=0.1$ lowered training accuracy to 0.875.
- **"Early stopping on the test set is fine."** It is not: pick the stopping epoch on a validation split.
- **"L1 and L2 only differ in strength."** L1 gives exact zeros (sparsity); L2 shrinks proportionally.

## Key takeaways

- Overfitting is the gap between train and test loss; regularisers narrow it by removing ways to memorise.
- L2 is the negative log of a Gaussian prior, with $\lambda=\sigma^2/\tau^2$; its update is $\theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla L$.
- Inverted dropout multiplies by $m/(1-p)$ in training so $\mathbb{E}[\tilde a]=a$ and inference needs no change; backward reuses the same mask.
- Early stopping and augmentation regularise through the procedure and the data; augmentation is equivalent to a smoothness penalty for small transformations.
- Always measure on held-out data, and expect effects of 1-3 accuracy points on small problems.

## Explain it yourself

1. Derive why dropout needs the factor $1/(1-p)$ and what would go wrong at test time without it.
2. Explain why L1 drives weights to exactly zero while L2 only shrinks them, using the one-dimensional example.

## Further reading

- Bishop, *Pattern Recognition and Machine Learning*, Sec. 5.5 (priors, early stopping, transformed data, and the tangent-propagation and soft-weight-sharing sections that we did not read).
- Source library: phase 03 lesson 07.

## Mastery criteria

You can state the Gaussian-prior reading of weight decay, derive inverted dropout including its variance and backward pass, implement both with tests, and explain from measurements why a regulariser can leave the test accuracy unchanged yet change the test loss.
