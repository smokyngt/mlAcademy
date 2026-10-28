---
id: dl.01
title: Optimisers and learning-rate schedules
summary: Momentum, Nesterov, RMSProp, Adam and AdamW derived and implemented in NumPy, plus warmup and cosine schedules, compared on an ill-conditioned quadratic and on the moons network.
minutes: 80
teaches: [dl.momentum, dl.adam, dl.lr-schedule]
requires: [opt.gradient-descent, nn.training]
sources:
  - "math/math for ML.pdf (Sec. 7.1 Optimization Using Gradient Descent, PDF p.233-239: Example 7.1 quadratic, poor conditioning and zig-zag, momentum eq. 7.11-7.12, SGD and mini-batches). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/06-optimizers/docs/en.md (SGD, momentum, RMSProp, Adam, bias correction, AdamW, default hyperparameters). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/09-learning-rate-schedules/docs/en.md (constant, step, cosine, warmup+cosine, 1cycle). READ."
  - "LLMs-from-scratch/appendix-D/01_main-chapter-code/appendix-D.ipynb (D.1 warmup, D.2 cosine decay, D.3 gradient clipping). READ."
---

## Where this fits

In [[f2n.11]] you trained a network with plain gradient descent plus momentum and saw that it works, but also that it is touchy: the learning rate that is perfect for one problem diverges on the next. Every large model you will meet (a language model, a vision transformer, a diffusion model) is trained with a more elaborate update rule, almost always Adam or AdamW, together with a learning-rate schedule that starts near zero, climbs, and then decays. This lesson explains each ingredient, derives the one non-obvious formula (Adam's bias correction), and measures what each ingredient buys you on two small problems you can run in your browser.

Backpropagation ([[f2n.10]]) gives the gradient. The optimiser decides what to do with it. Same gradient, different update rule, very different training run.

## The idea in plain words

Picture a marble in a long, narrow valley with steep walls and a gently sloping floor. Plain gradient descent looks only at the slope under its feet. The slope points mostly at the nearest wall, so the marble bounces from wall to wall while creeping along the floor. Turning the step size down stops the bouncing but makes the creep along the floor painfully slow. This is the picture Deisenroth et al. draw for poorly conditioned problems: the valley is very curved in one direction and almost flat in the other.

Three fixes, each an upgrade of the marble:

- **Momentum**: give the marble weight. It remembers where it was heading. The bounce across the valley flips sign every step and cancels out in the memory; the push along the floor always points the same way and builds up.
- **Nesterov**: before deciding where to push, look ahead to where the memory is about to carry you, and read the slope there. You brake earlier when you are about to overshoot.
- **Adaptive step sizes (RMSProp, Adam)**: give each parameter its own step size. A parameter whose gradients have been huge recently gets a small step; one whose gradients have been tiny gets a big step. Adam combines this with momentum.

A **schedule** then changes the overall step size during training: gentle at the start (the running statistics are not trustworthy yet), large in the middle, small at the end so that you settle in the valley instead of bouncing around its floor.

## Formal definitions

Let $\theta$ be the parameters, $g_t=\nabla L(\theta_{t})$ the (mini-batch) gradient at step $t$, and $\eta$ the learning rate. All products and powers of vectors below are elementwise.

**SGD.** $\theta\leftarrow\theta-\eta g$. Mini-batches are fine because an unbiased gradient estimate is all that convergence needs (Deisenroth Sec. 7.1.3).

**Momentum** (heavy ball). Keep a velocity and step along it:

$$
v_t=\mu\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t .
$$

This is the same idea as eq. 7.11-7.12 in the book (a step plus $\alpha$ times the previous step); the form above, used by most libraries, keeps $\eta$ outside the velocity.

**Nesterov momentum.** Same velocity, but step along the lookahead direction:

$$
v_t=\mu\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,(g_t+\mu\,v_t).
$$

**RMSProp.** Track a running mean of squared gradients and divide by its root:

$$
s_t=\beta\,s_{t-1}+(1-\beta)\,g_t^2,\qquad \theta_{t+1}=\theta_t-\eta\,\frac{g_t}{\sqrt{s_t}+\varepsilon}.
$$

**Adam.** Two running means (of the gradient and of its square), each bias-corrected:

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}},\qquad
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$.

**AdamW** adds weight decay outside the adaptive scaling:

$$
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}-\eta\lambda\,\theta_t .
$$

## Derivation

### Why momentum accelerates consistent directions

Suppose the gradient is a constant $g$ along some direction. Then $v_t=\mu v_{t-1}+g$ has the fixed point $v=g/(1-\mu)$. The effective step is therefore $\eta/(1-\mu)$: ten times larger for $\mu=0.9$. Now suppose the gradient alternates $+g,-g,+g,\dots$ (the bounce across the valley). Writing $v_t=(-1)^t a$ in the recursion gives $a=-\mu a+g$, so $a=g/(1+\mu)$. The ratio between the consistent and the alternating case is $(1+\mu)/(1-\mu)=19$ for $\mu=0.9$. Momentum amplifies exactly the directions that agree and suppresses those that flip.

### How much it helps on a quadratic

For $f(\theta)=\tfrac12\theta^\top A\theta-b^\top\theta$ with eigenvalues between $\lambda_{\min}$ and $\lambda_{\max}$ and condition number $\kappa=\lambda_{\max}/\lambda_{\min}$, plain gradient descent with the best constant step $\eta=2/(\lambda_{\max}+\lambda_{\min})$ shrinks the error by a factor $(\kappa-1)/(\kappa+1)$ per step. Stability requires $\eta<2/\lambda_{\max}$; above that the iteration diverges along the steepest direction. Heavy-ball momentum with $\eta=4/(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2$ and $\mu=\big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$ improves the per-step factor to $(\sqrt\kappa-1)/(\sqrt\kappa+1)$. For $\kappa=100$ that is $0.980$ versus $0.818$ per step: the iteration count drops from order $\kappa$ to order $\sqrt\kappa$. We state these classical rates and verify them numerically below rather than prove them here.

### Adam's bias correction

Start both moments at zero. Unroll the recursion for the second moment:

$$
v_t=(1-\beta_2)\sum_{i=1}^{t}\beta_2^{\,t-i}\,g_i^2 .
$$

Assume for a moment that the gradients are stationary, $\mathbb{E}[g_i^2]=\sigma^2$ for all $i$. Then

$$
\mathbb{E}[v_t]=(1-\beta_2)\,\sigma^2\sum_{i=1}^{t}\beta_2^{\,t-i}=(1-\beta_2)\,\sigma^2\,\frac{1-\beta_2^{\,t}}{1-\beta_2}=\sigma^2\,(1-\beta_2^{\,t}).
$$

The estimate is too small by exactly the factor $1-\beta_2^{\,t}$ because the zero initial value is being averaged in. Dividing by that factor gives an unbiased estimate, $\mathbb{E}[\hat v_t]=\sigma^2$. The same argument gives $\hat m_t=m_t/(1-\beta_1^{\,t})$. At $t=1$ the correction factors are $1/(1-\beta_1)=10$ for $m$ and $1/(1-\beta_2)=1000$ for $v$. By $t=50$ the denominator for $m$ is $1-0.9^{50}=0.995$, essentially gone, but for $v$ it is still $1-0.999^{50}\approx0.049$, so the correction for the second moment matters for hundreds of steps.

A consequence worth remembering: at $t=1$, $\hat m_1=g_1$ and $\hat v_1=g_1^2$, so the first update is $-\eta\,g_1/(|g_1|+\varepsilon)\approx-\eta\,\mathrm{sign}(g_1)$. Adam's first step has size about $\eta$ in every coordinate, whatever the gradient scale.

### Why AdamW, not Adam plus L2

Adding $\tfrac{\lambda}{2}\|\theta\|^2$ to the loss adds $\lambda\theta$ to the gradient. In plain SGD this is the same as shrinking the weights by $\eta\lambda\theta$ each step. In Adam, however, the gradient $g+\lambda\theta$ is divided by $\sqrt{\hat v}$, so weights with a large gradient history are decayed less and weights with small gradients are decayed more: the strength of the regulariser now depends on the gradient statistics. AdamW applies $-\eta\lambda\theta$ directly, so every weight shrinks in proportion to its size. The source lesson reports that AdamW generalises better than Adam with L2 and is the default for transformers; treat the size of that improvement as problem-dependent, we do not measure it here.

### Schedules

Warmup plus cosine decay, with peak $\eta_{\max}$, minimum $\eta_{\min}$, $W$ warmup steps and $T$ total steps:

$$
\eta(t)=\begin{cases}\eta_{\max}\,t/W & t<W\\[2pt] \eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\Big(1+\cos\big(\pi\,\tfrac{t-W}{T-W}\big)\Big) & t\ge W.\end{cases}
$$

Why warm up? With $v_0=0$ the early second-moment estimate rests on a handful of gradients, so it can be far from the truth, and a large $\eta$ turns that error into a big, badly scaled step. Starting small lets the statistics settle (the bias correction removes the mean error, not the noise). Step decay (multiply by 0.1 at fixed epochs) and 1cycle (ramp up for half of training, then down) are the other common shapes. The appendix-D notebook warms up linearly from a small initial rate to a peak, then applies cosine decay, and adds gradient clipping: rescale the gradient so that its L2 norm is at most a threshold (a gradient matrix with $\|G\|_2=5$ and max-norm 1 is multiplied by $1/5$).

## Worked example

### An ill-conditioned quadratic

Take $A=R\,\mathrm{diag}(1,100)\,R^\top$ with $R$ a rotation by $30^\circ$, so $\kappa=100$ and the valley is not aligned with the axes. Minimise $f(x)=\tfrac12x^\top Ax-b^\top x$ with $b=(1,1)$, from $x_0=(-3,-1)$, until $f(x)-f(x^*)<10^{-6}$ (initial gap $29.83$). For each optimiser we searched the learning rates $\{0.003,0.005,0.01,0.015,0.02,0.03,0.05,0.1,0.2,0.3,0.5,1\}$ and kept the best (momentum $\mu=0.9$ for both momentum variants; Adam and RMSProp at default betas):

| Optimiser | Best learning rate | Steps to gap below $10^{-6}$ |
|---|---|---|
| SGD | 0.015 | 534 |
| Momentum | 0.03 | 138 |
| Nesterov | 0.005 | 50 |
| Adam | 0.5 | 151 |
| RMSProp | none in the grid | never reached |

Read the table carefully, not just the ranking.

- SGD at the theoretical optimum $\eta=2/101$ needs 431 steps, matching the rate $0.980^{2t}$; at $\eta=0.02=2/\lambda_{\max}$ it diverges.
- Heavy ball with the textbook values ($\eta=4/121$, $\mu=(9/11)^2$) needs 66 steps. The generic $\mu=0.9$ is not optimal for this problem, which is why it took 138.
- The Nesterov number comes from a favourable cell of the grid (at $\eta=0.01$ it takes 97 steps and at $0.015$ it diverges); do not over-read it. The honest conclusion is that both momentum variants beat SGD by a factor of 4 to 10 here.
- Adam needs a large learning rate (0.5) on this smooth, deterministic problem. Its step is about $\eta$ per coordinate regardless of gradient size, so near the optimum, where gradients vanish, the normalised step stays large and it tends to hop; it works only when $\eta$ is big enough to converge before that matters. Adam's strength is not smooth quadratics; it is noisy, badly scaled, high-dimensional losses.
- RMSProp with a constant $\eta$ never gets below the tolerance in this grid for the same reason (it has no momentum averaging and hops at scale $\eta$). That is why schedules that decay $\eta$ matter.

The first-step claim is checkable: Adam with $\eta=0.001$ and gradient $(5,-0.02)$ moves the parameters from $(0,0)$ to $(-0.001,+0.001)$. Both coordinates move by exactly $\eta$ although the gradients differ by a factor of 250.

### The moons network

Same network as [[f2n.11]] ($2\to16\to2$, ReLU, He initialisation, seed 0, 200 full-batch steps on 200 points). Final training loss by learning rate:

| Optimiser | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| SGD | 0.456 | 0.348 | 0.282 | 0.247 | 0.191 | 0.042 | 0.009 |
| Momentum | 0.282 | 0.245 | 0.183 | 0.041 | 0.014 | 0.008 | 0.0005 |
| Nesterov | 0.282 | 0.245 | 0.186 | 0.042 | 0.014 | 0.002 | 0.266 |
| RMSProp | 0.210 | 0.043 | 0.010 | 0.0011 | 0.0010 | 0.352 | 1.157 |
| Adam | 0.246 | 0.107 | 0.011 | 0.0014 | 0.0007 | 0.009 | 0.308 |

The pattern is the lesson. SGD only reaches a small loss at learning rates of 1 to 3. Adam and RMSProp reach a tiny loss across a 10-fold range of learning rates (0.03 to 0.3 for Adam), so they are far less sensitive to the exact value, at the cost of blowing up at the high end (RMSProp at 1, Adam at 3). These are training losses on separable data after 200 steps; they say how fast each rule optimises, not how well it generalises.

Schedule check: Adam with $\eta=0.3$ constant ends at loss $0.00075$; with 20 warmup steps and cosine decay to $0.003$ it ends at $0.0029$ (both reach accuracy 1.0). On this tiny problem warmup buys nothing: the early loss is slightly worse ($0.317$ versus $0.230$ at step 10) because warmup throws away steps. Warmup earns its keep in large transformers trained with big batches and high peak rates; do not expect to see it on a toy.

## Implementation

```python
import numpy as np

class Momentum:
    def __init__(self, lr, mu=0.9, nesterov=False):
        self.lr, self.mu, self.nes, self.v = lr, mu, nesterov, None
    def step(self, p, g):
        if self.v is None: self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g
        d = g + self.mu * self.v if self.nes else self.v
        return p - self.lr * d

class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = self.v = None; self.t = 0
    def step(self, p, g):
        if self.m is None: self.m = np.zeros_like(p); self.v = np.zeros_like(p)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps) - self.lr * self.wd * p
```

The exercise asks you to write these and a warmup-cosine schedule.

## Shapes

Every state tensor has exactly the shape of its parameter: for a weight $W\in\mathbb{R}^{n_{in}\times n_{out}}$, the arrays $v$, $m$ and $s$ are also $n_{in}\times n_{out}$. Adam therefore stores two extra copies of the model on top of the gradient: optimiser state is about three times the parameter count in float32, a real constraint for large models.

## Common misconceptions

- **"Adam makes the learning rate irrelevant."** It makes the scale of the gradient irrelevant. The step is still about $\eta$ per coordinate, and the table above shows Adam diverging at $\eta=3$.
- **"Bias correction is a detail."** For $\beta_2=0.999$ the second-moment estimate is badly biased for hundreds of steps without it.
- **"Weight decay and L2 are the same."** Only for SGD. With Adam they differ; use AdamW.
- **"Momentum 0.9 is always best."** The optimal $\mu$ depends on the curvature; for the quadratic above the textbook value is $0.67$.
- **"Warmup never hurts."** It costs early progress; it is a stability tool, not a speed tool.

## Key takeaways

- Momentum accumulates agreeing gradient components (effective step $\eta/(1-\mu)$) and cancels flipping ones; on quadratics it cuts the iteration count from order $\kappa$ to order $\sqrt\kappa$.
- Adam = momentum + per-parameter scaling by $1/\sqrt{\hat v}$; its first step is about $\eta\,\mathrm{sign}(g)$.
- Bias correction divides by $1-\beta^t$ because a zero-initialised moving average underestimates by exactly that factor.
- AdamW decays weights outside the adaptive scaling; Adam with L2 does not.
- A schedule (warmup, then cosine decay) controls the overall step size; decaying it lets the iterate settle.

## Explain it yourself

1. Why does an optimiser with a constant per-coordinate step of about $\eta$ (RMSProp, Adam) fail to settle on a smooth quadratic, and what fixes it?
2. Derive, in your own words, why $\hat v_t=v_t/(1-\beta_2^t)$ is unbiased under stationary gradients.

## Further reading

- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*, Sec. 7.1.
- Source library: phase 03 lessons 06 (optimisers) and 09 (schedules); *Build a Large Language Model From Scratch*, appendix D.

## Mastery criteria

You can write Momentum, Nesterov, Adam and AdamW from memory, derive the bias-correction factor, predict the size of Adam's first step, and explain from the numbers above why SGD is far more learning-rate-sensitive than Adam on the moons network.
