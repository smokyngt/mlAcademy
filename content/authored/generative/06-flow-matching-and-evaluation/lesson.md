---
id: gen.06
title: "Flow matching and evaluating generators"
summary: Learn a velocity field along straight noise-to-data paths and sample with an ODE (flow matching, rectified flow), then measure generators honestly - the Frechet distance behind FID, k-NN precision and recall, and pairwise human preference.
minutes: 85
teaches: [gen.flow-matching, gen.eval]
requires: [gen.diffusion, gen.taxonomy, la.eigen, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/13-flow-matching-rectified-flows/docs/en.md: straight-line interpolation, velocity target x1 - x0, conditional flow matching loss, Euler sampling backwards in time, rectified flow and reflow, why it displaced DDPM. READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai/14-evaluation-fid-clip-score/docs/en.md: FID definition and failure modes, CLIP score, CMMD, human preference with Elo, evaluation checklist. READ."
---

## Where this fits

Diffusion ([[gen.04]]) produces excellent samples but needs many small stochastic steps because the reverse path is curved and noisy. **Flow matching** asks a cleaner question: since we are free to choose how noise is connected to data, connect them with *straight lines* and learn the velocity along them. Sampling becomes deterministic ODE integration with few steps, and the training loss is as simple as the DDPM one. The second half of the lesson tackles the question every generative project eventually faces: how do you know the model is good? You will implement the Frechet distance behind **FID**, a precision/recall pair that separates quality from coverage ([[gen.01]]), and the rating arithmetic of human preference studies. It uses eigen-decompositions from [[math.07]] and Gaussian facts from [[math.09]].

## The idea in plain words

Imagine two clouds of points: a round cloud of "noise" and a cloud shaped like your data. Pair each noise point with a data point and draw a straight rope between them. A flow is a traffic rule that says, for every place and every moment, which way to move so that noise points slide along ropes to the data. Training means learning that rule from many random pairings. The catch: at a given place and time, ropes from many different pairings cross, so the learned rule is the *average* direction of all the ropes through that point. Following the averaged field still carries the whole noise cloud onto the data cloud, but the individual trajectories bend a little. The fewer the crossings, the straighter the paths and the fewer the steps you need. "Reflow" re-pairs points to reduce crossings.

For evaluation, the analogy is a wine competition. A *quality* judge asks "does each sample taste like real wine?" A *coverage* judge asks "do your samples span all the kinds of real wine?" FID blends both into one number; precision measures the first, recall the second; and humans are the final taster.

## Flow matching

Let $x_0\sim p_\text{data}$ and $x_1\sim\mathcal N(0,I)$ (note: this lesson puts data at $t=0$ and noise at $t=1$, following the source). The straight path is

$$
x_t=(1-t)\,x_0+t\,x_1,\qquad \frac{dx_t}{dt}=x_1-x_0 .
$$

A network $v_\theta(x,t)$ is trained with the **conditional flow matching** loss

$$
L_\text{CFM}=\mathbb E_{t\sim\mathrm U[0,1],\,x_0,\,x_1}\big\|v_\theta(x_t,t)-(x_1-x_0)\big\|^2 .
$$

It is *simulation-free*: sample a pair and a time, form $x_t$ with one line, regress. There are no noise schedules to tune and no ODE solve inside training.

### Why regressing on a per-pair target learns the right field

The target $u=x_1-x_0$ depends on the pair, but the network sees only $(x_t,t)$. Squared-error regression is minimised by the conditional expectation, so the optimal field is the **marginal velocity**

$$
v^*(x,t)=\mathbb E\big[x_1-x_0\ \big|\ x_t=x\big].
$$

Formally, $\mathbb E\|v_\theta-u\|^2=\mathbb E\|v_\theta-v^*\|^2+\mathbb E\|u-v^*\|^2$ (the cross term vanishes because $u-v^*$ has zero conditional mean given $x_t$), and the last term does not depend on $\theta$. So minimising CFM is the same as matching $v^*$ in least squares. By the continuity equation, a flow of $v^*$ transports the marginal distribution of $x_t$ for every $t$, hence moves $\mathcal N(0,I)$ at $t=1$ to $p_\text{data}$ at $t=0$ (we assume $v^*$ is well behaved).

### Sampling

Draw $x_1\sim\mathcal N(0,I)$ and integrate $dx/dt=v_\theta(x,t)$ **backwards** from $t=1$ to $0$. The Euler method with $N$ steps of size $\Delta t=1/N$ is

$$
x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t).
$$

Higher-order solvers (Heun, Runge-Kutta) get the same accuracy with fewer steps. A straight field would need only one step.

### Rectified flow and reflow

Even the optimal $v^*$ has curved trajectories wherever ropes cross. **Reflow** (Liu et al.): train $v^{(1)}$, then generate pairs $(x_1,\hat x_0)$ by integrating it, and train a new model $v^{(2)}$ on the straight lines between these *ODE-matched* pairs. Matched pairs cross much less, so the new field is much straighter; two rounds typically suffice for 2-4 step sampling. Distilled one-step models in the source lesson's list follow the same direction. This is a high-level description; I did not derive the straightening guarantee here.

## Worked example: exact flow for Gaussian data

For $x_0\sim\mathcal N(m,s^2)$, $x_1\sim\mathcal N(0,1)$ independent, $x_t$ is Gaussian with mean $(1-t)m$ and variance $(1-t)^2s^2+t^2$. The velocity $u=x_1-x_0$ has mean $-m$ and $\mathrm{Cov}(u,x_t)=t-(1-t)s^2$. Gaussian regression gives the optimal field in closed form:

$$
v^*(x,t)=-m+\frac{t-(1-t)s^2}{(1-t)^2s^2+t^2}\,\big(x-(1-t)m\big).
$$

Check by Monte Carlo: at $t=0.4$, $x=1.5$, $m=3$, $s=0.5$ the formula gives $-3.3$; averaging $x_1-x_0$ over two million simulated pairs with $x_t$ within $0.01$ of $1.5$ gives $-3.304$.

Now sample 20,000 points with Euler and $N$ steps, $m=3$, $s=0.5$ (seed 0):

| steps $N$ | sample mean | sample std |
|---|---|---|
| 1 | 3.000 | 0.000 |
| 2 | 3.001 | 0.199 |
| 4 | 3.002 | 0.339 |
| 20 | 3.002 | 0.462 |
| 100 | 3.002 | 0.491 |

The true std is $0.5$. With one step *every* noise point lands on the mean $3$: the first Euler step uses $v^*(x,1)=x-m$, which is the average over all the crossing ropes, so it cannot separate individual data points. With more steps the finer integration follows the curved paths and recovers the spread. This is the whole motivation for straightening the paths. It is a model case: real data are multi-modal and the curvature is far worse.

## Evaluating generators

### The Frechet distance and FID

Take a feature extractor (for images FID uses the activations of an Inception-v3 network, 2048 numbers per image). Fit a Gaussian $\mathcal N(\mu_r,\Sigma_r)$ to the features of real images and $\mathcal N(\mu_g,\Sigma_g)$ to generated ones. The **Frechet distance** (2-Wasserstein distance between Gaussians) is

$$
d^2=\|\mu_r-\mu_g\|^2+\mathrm{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big).
$$

Two sanity checks. In 1-D with variances $\sigma_r^2,\sigma_g^2$ this is $(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2$: the means and the spreads must both agree (example: $\mathcal N(0,1)$ vs $\mathcal N(2,4)$ gives $4+1=5$). With equal Gaussians it is $0$. The matrix square root of a non-symmetric product is awkward; the trace is computed stably as the sum of square roots of the eigenvalues of $\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}$, which is symmetric positive semi-definite (square roots of PSD matrices come from `eigh`, see [[math.07]]). That is what the exercise does. Example in 2-D, $\mu_r=0$, $\mu_g=(1,2)$, $\Sigma_r=\begin{pmatrix}2&0.3\\0.3&1\end{pmatrix}$, $\Sigma_g=\mathrm{diag}(1,1.5)$: $d^2=5.252$.

**FID's failure modes** (several are in the source; one I corrected). The Gaussian assumption ignores multimodality; the features come from a network trained on ImageNet, so domains far from ImageNet give a questionable number; the score is sensitive to preprocessing. **Sample-size bias:** the source lesson says small $N$ gives a falsely *low* FID. My experiment shows the opposite: two samples from the *same* 8-D Gaussian give FID $2.64$ for $N=20$, $0.554$ for $N=100$, $0.0497$ for $N=1000$, $0.0052$ for $N=10000$ (seeded, reproduced in the exercise). Noisy covariance estimates make the distance larger, and it decreases like $1/N$ toward 0. Either way, the practical rule is the same: compare models only at the same, large $N$ (the source recommends 10,000 or more). Also note that a single scalar hides *why* a model scores badly.

### Precision and recall for generators

Separate the two aspects with sets of feature vectors. For each point $x$ in a set $S$ let $r_k(x)$ be the distance to its $k$-th nearest neighbour in $S$ (excluding itself); the ball of that radius is an estimate of the local support. Then

- **precision** = fraction of generated points inside the estimated support of the real data (are samples realistic?),
- **recall** = fraction of real points inside the estimated support of the generated data (are all real modes covered?).

Experiment (400 real points $\sim\mathcal N(0,I_2)$, $k=3$, seeded):

| generated set | precision | recall |
|---|---|---|
| same distribution | 0.9825 | 0.975 |
| collapsed ($\mathcal N(0,0.1^2)$) | 1.000 | 0.035 |
| shifted far away | 0.0625 | 0.0875 |

A collapsed generator has perfect precision and terrible recall, exactly the mode-collapse signature of [[gen.03]], which FID alone would report only as "bad". I wrote this method from general knowledge of the k-NN support-estimation idea; the source lesson does not cover precision/recall.

### Other automatic metrics

*CLIP score*: cosine similarity between image and prompt embeddings, a measure of prompt adherence; it inherits the weaknesses of CLIP (poor compositional reasoning, bias by prompt length). *CMMD*: a maximum mean discrepancy in CLIP feature space, with no Gaussian assumption. Learned preference scorers (HPSv2, ImageReward, PickScore) are proxies trained on human choices. The source's advice is to report several (sample quality, prompt adherence, preference) plus a qualitative review of random samples.

### Human evaluation with Elo

Show raters pairs of images from two models for the same prompt and record the winner. Ratings follow the Bradley-Terry logic of Elo: the expected score of $A$ against $B$ is

$$
E_A=\frac1{1+10^{(R_B-R_A)/400}},\qquad R_A\leftarrow R_A+K\,(S_A-E_A),
$$

with $S_A=1$ for a win, $0$ for a loss and typically $K=32$. Equal ratings give $E_A=0.5$; if A (1000) beats B (1000) the new ratings are $1016$ and $984$; a 200-point favourite has $E=0.760$ and gains only $7.7$ for winning. Rules of good practice: blind the pairs, use held-out prompts (not seen in training), mix annotator pools, and treat an LLM judge as a proxy that can be fooled by pretty-but-wrong outputs.

## Implementation

```python
import numpy as np

def sqrtm_psd(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

def frechet_distance(mu1, S1, mu2, S2):
    r = sqrtm_psd(S1)
    M = r @ S2 @ r
    tr = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr)
```

In practice use `torchmetrics`/`pytorch-fid` with a fixed Inception checkpoint and cached real-data statistics; never compare FIDs computed with different extractors, resolutions or resizing code.

## Shapes

Features `(N, d)`; mean `(d,)`; covariance `np.cov(F, rowvar=False)` of shape `(d, d)` (needs $N>d$, else singular). Flow matching: `x0, x1, x_t: (B, ...)`, `t: (B,)` broadcast to the data shape, `v_theta(x_t, t)` same shape as `x_t`. Pairwise distances in k-NN: `(N_fake, N_real)`.

## Common misconceptions

- **"Flow matching and diffusion are unrelated."** For Gaussian paths they are two parameterisations of closely related models; the practical differences are the simulation-free regression target, the straighter paths and ODE sampling.
- **"One step is enough once trained."** Only if the learned field is straight; otherwise one step returns the conditional average (the blurry mean).
- **"Lower FID always means better."** It is biased by sample size, extractor and domain; it cannot tell quality from coverage.
- **"Precision and recall are classifier metrics here."** They are set-coverage measures in feature space and need a good feature extractor and a choice of $k$.
- **"Human preference is ground truth."** It depends on prompts, rater pool and pairing; it needs the same discipline as any experiment.

## Key takeaways

- Flow matching regresses the velocity $x_1-x_0$ along straight paths; the minimiser is $\mathbb E[x_1-x_0|x_t]$, and sampling integrates the ODE from noise to data.
- Fewer steps need straighter paths (reflow, distillation).
- FID is the Frechet distance between Gaussian fits in a feature space: $\|\Delta\mu\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$.
- Report quality and coverage separately (precision, recall), plus prompt adherence and blinded human preference.

## Explain it yourself

1. Why does the optimal flow-matching field equal $\mathbb E[x_1-x_0\mid x_t]$, and why does a single Euler step then return the mean of the data in the Gaussian example?
2. A generator has high precision and low recall. What does that say about the samples, and which failure mode of GAN training does it resemble?

## Further reading

- Source lessons: phase 08 lessons 13 (flow matching, rectified flows) and 14 (FID, CLIP score, preferences). Original papers (Lipman et al., Liu et al., Heusel et al., Kynkaanniemi et al.) were not read for this lesson.
- Back to the overview: [[gen.01]].

## Mastery criteria

You can state and justify the flow-matching loss and its optimal field, write an Euler sampler, compute the Frechet distance from feature statistics, measure precision and recall with k-NN balls, and update Elo ratings from preference outcomes.
