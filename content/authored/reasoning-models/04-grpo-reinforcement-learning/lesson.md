---
id: rsn.04
title: "Reinforcement learning for reasoning: GRPO"
summary: Train a model to reason by rewarding verified-correct answers, derive the group-relative advantage that replaces a learned critic, assemble the clipped and KL-regularised GRPO loss, and check its gradient numerically on a toy policy.
minutes: 100
teaches: [rsn.grpo, rsn.reward-design]
requires: [rl.policy-gradient, rsn.verifier]
sources:
  - "reasoning-from-scratch/ch06/01_main-chapter-code/ch06_main.ipynb (Sec. 6.1-6.13): RL for LLMs, RLHF versus RLVR, GRPO procedure, rollouts, reward_rlvr, advantages, sequence log-probabilities (sum versus mean), the policy-gradient loss, compute_grpo_loss, the training loop, results table. READ."
  - "reasoning-from-scratch/ch07/01_main-chapter-code/ch07_main.ipynb (Sec. 7.2-7.6: metrics to track, clipped policy ratio, KL term, format reward and conditional format reward): READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (Sec. 3.4-3.7: the verifier that defines the reward): READ."
  - "reasoning-from-scratch/chE/chE_main.ipynb (Sec. E.6, E.8: batched GRPO rollouts and timings): READ."
---

## Where this fits

So far the model was fixed and we spent compute at answer time ([[rsn.02]], [[rsn.03]]). Now we change the model itself: **train it, with reinforcement learning (RL), to produce chains of thought that end in verified-correct answers.** This is the technique behind recent reasoning models (the book follows the recipe publicly described for DeepSeek-R1). It needs three things you already have: a policy that generates text ([[llm.02]]), a verifier that scores it ([[rsn.01]]) and the policy-gradient idea from [[rl.05]]. The new piece is how to estimate "was this rollout better than usual?" cheaply, without training a second neural network. That is **Group Relative Policy Optimisation (GRPO)**.

Two lessons follow this one: [[rsn.05]] fixes GRPO's weak points and shows the alternative of distilling a bigger model's reasoning.

## The idea in plain words

A teacher gives a student one hard problem and asks for **four** different attempts. A grader marks each right or wrong. Two attempts were right, two wrong. The teacher does not need an expert "value model" to tell the student what a typical attempt scores: the four attempts *are* the reference. Praise the two that were better than the group average, discourage the two that were worse, and repeat on the next problem. If all four were right (or all wrong), nobody stands out, and there is nothing to learn from this problem today.

"Praise" means: make the exact words of that attempt a bit more likely next time. "Discourage" means: make them a bit less likely. Do this over thousands of problems and the model drifts toward the habits that lead to correct answers, such as writing out steps, double-checking and boxing the answer.

Chapter 6 of the book contrasts this with RLHF ([[llm.05]]), where a **learned reward model** scores answers by imitating human preferences. With **verifiable rewards (RLVR)** the reward comes from a program (the checker of [[rsn.01]], a unit test, a compiler), so it is cheap, objective and cannot be sweet-talked.

## From rewards to a gradient

### Setup

- Prompt $x$ (a problem). Policy $\pi_\theta(y\mid x)$: the LLM with weights $\theta$, which generates a token sequence $y=(y_1,\dots,y_T)$ with probability $\pi_\theta(y\mid x)=\prod_{t}\pi_\theta(y_t\mid x,y_{<t})$. A sampled $y$ is called a **rollout** or completion.
- Reward $r(x,y)$: for example $1$ if the boxed answer is verified correct and $0$ otherwise.
- Objective: $J(\theta)=\mathbb{E}_{y\sim\pi_\theta}[r(x,y)]$, the expected reward. We want to climb it.

### The policy gradient (recap of [[rl.05]])

$J$ depends on $\theta$ only through the sampling distribution, so we use the **log-derivative trick** $\nabla\pi=\pi\,\nabla\log\pi$:

$$
\nabla_\theta J=\sum_y r(y)\,\nabla_\theta\pi_\theta(y)=\sum_y \pi_\theta(y)\,r(y)\,\nabla_\theta\log\pi_\theta(y)=\mathbb{E}_{y\sim\pi_\theta}\bigl[r(y)\,\nabla_\theta\log\pi_\theta(y)\bigr].
$$

Sample rollouts, weight each one's log-probability gradient by its reward: that is REINFORCE. The log-probability of a whole sequence is the **sum** of its token log-probabilities, so every token of a rollout is pushed with the same weight.

### Why a baseline, and why a group

Raw rewards are all non-negative: even a bad rollout would be pushed *up*. The fix is to subtract a **baseline** $b$ that does not depend on the sampled $y$:

$$
\mathbb{E}_y\bigl[b\,\nabla\log\pi(y)\bigr]=b\sum_y\nabla\pi(y)=b\,\nabla\sum_y\pi(y)=b\,\nabla 1=0,
$$

so $\mathbb{E}[(r-b)\nabla\log\pi]=\nabla J$: subtracting a baseline leaves the gradient unchanged on average and typically reduces its variance. PPO (see [[rl.05]]) learns a value network $V_\phi(x)$ as the baseline: a second LLM-sized model to train and keep in memory. GRPO's trick: **sample a group of $G$ rollouts for the same prompt and use the group's mean reward as the baseline.**

Subtle point (derived here, not in the book): the group mean includes $r_i$ itself. Write $b=\frac1G\sum_jr_j$. For $j\ne i$, $r_j$ is independent of $y_i$ so $\mathbb{E}[r_j\nabla\log\pi(y_i)]=\mathbb{E}[r_j]\cdot0=0$. Only the $j=i$ term survives, and
$$
\mathbb{E}\bigl[(r_i-b)\nabla\log\pi(y_i)\bigr]=\Bigl(1-\frac1G\Bigr)\nabla J .
$$
So the group baseline leaves the gradient direction exact and merely scales it by $(G-1)/G$.

### Group-relative advantage

The book additionally divides by the group's standard deviation. For rewards $r_1,\dots,r_G$ of the rollouts to one prompt:

$$
A_i=\frac{r_i-\mu_r}{\sigma_r+\varepsilon},\qquad \mu_r=\frac1G\sum_j r_j,\quad \sigma_r=\sqrt{\frac1{G-1}\sum_j(r_j-\mu_r)^2},
$$

with a small $\varepsilon$ ($10^{-4}$) to avoid dividing by zero. Properties:

- The advantages sum to zero, so there are always some "up" and some "down" pushes. The average advantage is a sanity check that the code is right (it should stay at 0).
- If all rewards are equal, $r_i-\mu_r=0$ for every $i$: **no learning signal.**
- With binary rewards and $k$ of $G$ correct, the sample variance is $\frac{k(G-k)}{G(G-1)}$, so a correct rollout has $A=\frac{1-k/G}{\sigma}$ and a wrong one $A=-\frac{k/G}{\sigma}$. For $[1,1,0,0]$: $\sigma=\sqrt{1/3}=0.5774$, $A=\pm0.866$ (the code gives $0.8659$ because of $\varepsilon$). A rare success among eight ($k=1$) earns $A=+2.474$ while each failure gets only $-0.354$: **rare successes are strongly amplified.**

### The loss

Policy-gradient ascent on $\frac1G\sum_iA_i\log\pi_\theta(y_i\mid x)$ is gradient descent on

$$
\mathcal{L}_{\text{PG}}=-\frac1G\sum_{i=1}^{G}A_i\,\log\pi_\theta(y_i\mid x)=-\frac1G\sum_{i=1}^{G}A_i\sum_{t=1}^{T_i}\log\pi_\theta\bigl(y^{(i)}_t\mid x,y^{(i)}_{<t}\bigr).
$$

The advantages are treated as constants (`detach` in PyTorch): you differentiate through the log-probabilities only. The minus sign is there because optimisers minimise.

*Sum or mean over tokens?* The book sums (a sequence-level objective). Averaging would divide each rollout's push by its length, an unintended length-dependent rescaling. The summed log-probability is also more negative for long outputs, which, as the book notes, mildly favours shorter equally-good answers.

### PPO-style clipping and a KL anchor

Real training takes several optimisation steps on the same batch of rollouts, so after the first step the policy that generated them ($\pi_{\text{old}}$) is no longer the one being updated. Define the **ratio**
$\rho_i=\pi_\theta(y_i\mid x)/\pi_{\text{old}}(y_i\mid x)=\exp(\log\pi_\theta-\log\pi_{\text{old}})$, and replace $A_i\log\pi_\theta$ by the clipped surrogate used in PPO (see [[rl.05]]):

$$
\mathcal{L}=-\frac1G\sum_i\min\bigl(\rho_iA_i,\ \operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\,A_i\bigr)+\beta\,\frac1G\sum_i\bigl(\log\pi_\theta(y_i\mid x)-\log\pi_{\text{ref}}(y_i\mid x)\bigr).
$$

Why this form:

- At $\rho=1$ the gradient of $\rho A$ with respect to $\theta$ is $A\nabla\log\pi_\theta$, so the unclipped objective reduces to the policy gradient above. **At $\rho=1$ the loss value is 0** (the advantages sum to zero) **but the gradient is not**: do not use the logged loss as a progress meter.
- The $\min$ makes the update *pessimistic*. If $A>0$ and $\rho>1+\epsilon$ the clipped term is the smaller one and is constant in $\theta$: the gradient is zero, because this rollout has already become likelier than allowed. If $A<0$ and $\rho<1-\epsilon$ likewise. But if $A<0$ and $\rho>1+\epsilon$ the unclipped term is the smaller one and the gradient flows (the policy made a bad rollout likelier and should be corrected).
- The book notes DeepSeek-R1 used a very weak clip ($\epsilon=10$), whereas $0.1$ to $0.2$ is typical elsewhere.
- The last term is a **KL-style penalty** against a frozen reference model (the starting model). Its sample average estimates $\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ when rollouts come from $\pi_\theta$; it discourages drifting far from the original behaviour. It costs a second copy of the model. Chapter 7 reports that this simple form, computed on **summed** log-probabilities, destabilised training in the authors' run ([[rsn.05]]); the common recommendation for math is no KL at all.

### Everything in one picture

```mermaid
flowchart LR
  P[prompt x] --> S[sample G rollouts]
  S --> R[verifier: rewards r_i]
  R --> A[advantages A_i = (r_i - mean) / std]
  S --> L[sequence log-probs]
  A --> G[GRPO loss]
  L --> G
  G --> U[backward, clip grad norm, AdamW step]
  U --> P
```

## Reward design

The reward is the whole task specification. From the book:

- **Correctness reward.** $1$ if the last boxed answer is verified, else $0$. Requiring the box also acts as an implicit **format reward**, because an unboxed correct answer earns nothing (for the four sample rollouts "boxed 83", "the correct answer is boxed 83", "the final answer is 83" and "we get boxed 38" to a problem whose answer is 83, the rewards are $[1,1,0,0]$: the third is right but unboxed).
- **No process reward.** The book states that the DeepSeek-R1 team tried process reward models (scoring intermediate steps), found it unsuccessful, and used only final-answer rewards.
- **Explicit format reward** (chapter 7): add $+1$ if `<think>` appears before `</think>`. Unconditional, it can dominate: a wrong but well-formatted answer earns $1$, the same as a right answer without tags. The book saw the model improve for about a hundred steps, then degrade, with shorter responses, and suggests a smaller weight or a **conditional** format reward (format counts only if the answer is correct). In the exercise: right + tags $=2$; wrong + tags $=1$ unconditional but $0$ conditional.
- **Zero-signal problems.** If a problem is solved by *every* rollout or by *none*, its group has zero advantages. With success probability $p$ and $G$ rollouts the chance of that is $p^G+(1-p)^G$: for $p=0.2$ it is $0.41$ at $G=4$ and $0.17$ at $G=8$; for $p=0.05$ it is $0.81$ at $G=4$. Hard problems and a weak model give mostly empty groups, so more rollouts per prompt and a curriculum of moderate difficulty help. The book notes that an average reward of $1.0$ means no signal left: stop or use harder problems.

## What happened in the book's run

Qwen3 0.6B base, 50 GRPO steps, 8 rollouts of up to 512 tokens, learning rate $10^{-5}$ with AdamW, gradient-norm clip 1.0, MATH training problems that exclude MATH-500:

| model | MATH-500 accuracy | average generated tokens |
|---|---|---|
| base | 15.2% | 78.85 |
| reasoning variant (reference) | 48.2% | 1369.79 |
| base + 50 GRPO steps (no KL) | 47.4% | 586.11 |

Fifty steps of RL moved the base model close to the official reasoning variant, with responses 7.4 times longer than the base (writing out reasoning) but less than half as long as the reference. The book cautions that longer training is unstable and may even hurt, which motivates [[rsn.05]]. (This is one run on one model; differences of a point or two are inside the noise discussed in [[rsn.01]].)

The compute is large: the book's table in appendix E shows 68 minutes (H100) for the non-batched script and 19 minutes with 4 rollouts batched, at 43 to 45 GB of memory.

## Worked example

Four rollouts to a problem whose answer is 83 have rewards $[1,1,0,0]$.

1. $\mu=0.5$, $\sigma=\sqrt{\frac13}=0.5774$ (sample std).
2. Advantages: $\bigl(\tfrac{0.5}{0.5774+10^{-4}}\bigr)=0.8659$ for the first two, $-0.8659$ for the others.
3. Suppose the sequence log-probabilities are $\ell=[-7.9243,-20.1546,-16.6130,-23.3677]$ (the values the book uses to illustrate the loss). Then $\mathcal{L}_{\text{PG}}=-\frac14\sum_iA_i\ell_i=-\frac14\bigl(0.8659\cdot(-28.0789)-0.8659\cdot(-39.9807)\bigr)=-2.5764$. The sign is not meaningful by itself; what matters is the gradient, which raises $\ell_1,\ell_2$ and lowers $\ell_3,\ell_4$.
4. In the exercise's toy policy ($T=3$, $V=4$) the same four advantages produce an analytic gradient that agrees with central finite differences to better than $10^{-6}$, including when two of the four rollouts are clipped and a KL term is present.

## Implementation

```python
import numpy as np

def group_advantages(r, eps=1e-4):
    r = np.asarray(r, float)
    return (r - r.mean()) / (r.std(ddof=1) + eps)

def grpo_loss(logp, old_logp, ref_logp, adv, clip_eps=0.2, beta=0.0):
    ratio = np.exp(logp - old_logp)
    obj = np.minimum(ratio * adv, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    return -obj.mean() + beta * np.mean(logp - ref_logp)
```

In PyTorch: `logp = selected_logprobs.sum()` per rollout with gradients enabled, `adv` and `old_logp` detached, `loss.backward()`, `clip_grad_norm_(params, 1.0)`, `optimizer.step()`. Rollouts are sampled under `torch.no_grad` (not `inference_mode`, which blocks later autograd use). The production version also tracks reward, response length, advantage mean and std, and entropy ([[rsn.05]]), and batches rollouts. The tabular exercise differs from a real LLM only in how $\log\pi_\theta$ is computed, not in the loss.

## Common misconceptions

- "GRPO needs a reward model." Not for verifiable tasks; the checker is the reward model.
- "The group mean is an unbiased baseline, so there is no problem." It is only scaled by $(G-1)/G$; and the extra division by the standard deviation changes weights across problems ([[rsn.05]]).
- "A loss near 0 means training is stuck." With one update per batch the ratio is 1, the loss value is 0 by construction, yet the gradient is not.
- "If the average reward is 1.0, great, keep training." No signal remains; use harder problems.
- "A format reward is free." It can become the thing being optimised.

## Key takeaways

- RLVR: reward = a program's verdict; GRPO samples a group per prompt and uses the group as the baseline.
- $A_i=(r_i-\mu)/(\sigma+\varepsilon)$; zero when all rewards agree; rare successes get big positive advantages.
- Loss: advantage-weighted summed log-probabilities, with PPO-style clipping against $\pi_{\text{old}}$ and an optional KL term against $\pi_{\text{ref}}$.
- Check your gradient numerically; watch reward, length, advantage statistics, not the raw loss.
- Reward design is specification: be careful with format terms and zero-signal groups.

## Explain it yourself

1. Why can GRPO train without a value network, and what is the price?
2. Explain in your own words why the loss is exactly zero at the first step but the gradient is not.

## Further reading

- Book chapter 6 (algorithm), chapter 7.2-7.6 (monitoring, clipping, KL, format reward); policy gradients in [[rl.05]]; reward models and RLHF in [[llm.05]].
- DeepSeekMath and DeepSeek-R1 papers (GRPO's origin), cited by the notebooks but not opened for this lesson.

## Mastery criteria

- I can derive $\nabla J=\mathbb{E}[r\nabla\log\pi]$ and show a baseline does not bias it.
- I can compute group-relative advantages by hand and say when they vanish.
- I can write the clipped, KL-regularised loss and say which rollouts get zero gradient from clipping.
- I can verify an RL loss gradient numerically.
- I can explain two pitfalls of format rewards and zero-signal groups.
