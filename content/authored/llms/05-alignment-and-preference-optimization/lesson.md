---
id: llm.05
title: "Alignment: reward models, RLHF and direct preference optimisation"
summary: Why imitation is not enough, how pairwise human preferences become a reward model (Bradley-Terry), the KL-regularised RLHF objective and its closed-form optimum, how DPO is derived from it, and how these systems fail (reward hacking, sycophancy).
minutes: 90
teaches: [llm.reward-model, llm.dpo]
requires: [llm.sft, llm.lm, info.entropy, ml.logreg, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/07-rlhf/docs/en.md (three-stage pipeline, reward model with scalar head, Bradley-Terry loss, KL-penalised objective, PPO clipped surrogate and advantage, reward hacking and mitigation table, InstructGPT figures): READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/08-dpo/docs/en.md (DPO key insight, closed-form optimal policy, Z cancellation, DPO loss, RLHF vs DPO comparison, KTO/ORPO/SimPO, when each wins): READ."
---

## Where this fits

After pretraining ([[llm.03]]) and supervised fine-tuning ([[llm.04]]) you have a model that follows instructions. Is it a *good* assistant? Ask it to explain quantum computing and it may give two correct, grammatical answers, one clear and concise, the other rambling. SFT sees both as acceptable demonstrations; nothing in its loss says one is better. Yet users care exactly about that difference: helpfulness, honesty, tone, refusing what should be refused.

The fix is to train on **comparisons**. It turns out that people find it much easier and more consistent to say "answer A is better than answer B" than to write the ideal answer or to give a score like 7.3 out of 10. The comparison data are then used in one of two ways:

- **RLHF** (reinforcement learning from human feedback): fit a **reward model** to the comparisons, then optimise the language model to produce high-reward text without drifting too far from where it started.
- **DPO** (direct preference optimisation): a short algebraic step shows that the same objective can be optimised directly on the comparisons, with no separate reward model and no reinforcement-learning loop.

This lesson derives all three pieces, shows what each costs, and explains the main ways they go wrong.

## The idea in plain words

Picture a cooking competition. You cannot tell the contestant a recipe (that's SFT), but a panel can taste two dishes and say which is better. From many such verdicts, a **taste model** emerges: a function that scores any dish. That's the reward model. Now the contestant practises, wanting high scores from the taste model. Two dangers appear:

1. The taste model was learned from a limited number of verdicts. A clever contestant can find dishes that score high with it but that real judges would hate (very long, very salty, covered in garnish). This is **reward hacking**: optimising a proxy until it stops measuring what you cared about.
2. A contestant who rewrites their whole style to chase the score may lose basic skills.

The remedy for the second, and partly the first, is a **leash**: the contestant is penalised for straying too far from their original (SFT) style. That is the KL penalty. DPO's insight is that, with the leash in place, the best possible contestant has a known closed form; therefore you can train the contestant by *directly* making the preferred dishes more likely and the rejected ones less likely, measured relative to the leash, and skip the taste model.

## Preference data and the reward model

**Data.** For a prompt $x$ the model produces two responses; a person marks the better one, giving a triple $(x,y_w,y_l)$ ("winner", "loser"). The InstructGPT effort described in the source collected about 33,000 such comparisons from 40 contractors, around five minutes each.

**Reward model.** Take the SFT model, replace the vocabulary head with a head that outputs one number, and read it at the end of the (prompt, response) text: $r_\phi(x,y)\in\mathbb R$.

### Bradley-Terry derivation

We need a probabilistic model linking scores to choices. The Bradley-Terry model gives each response a positive strength $s=e^{r}$ and says the probability that $a$ beats $b$ is proportional to strength:

$$
P(a\succ b)=\frac{e^{r_a}}{e^{r_a}+e^{r_b}}=\frac{1}{1+e^{-(r_a-r_b)}}=\sigma(r_a-r_b).
$$

Properties worth noticing: the probability depends only on the *difference* $r_a-r_b$ (adding a constant to all scores changes nothing, so only differences are identifiable); equal scores give $1/2$; the probability tends to 1 as the gap grows.

**Training.** Maximise the likelihood of the observed winners. For one pair the negative log-likelihood is

$$
\ell(\phi)=-\ln\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)=\ln\bigl(1+e^{-d}\bigr),\qquad d=r_w-r_l .
$$

This is logistic regression on the score difference. Using $\sigma'(u)=\sigma(u)(1-\sigma(u))$ and $1-\sigma(d)=\sigma(-d)$:

$$
\frac{d\ell}{dd}=-\frac{\sigma(d)(1-\sigma(d))}{\sigma(d)}=-\sigma(-d),\qquad
\frac{\partial\ell}{\partial r_w}=-\sigma(-d),\quad\frac{\partial\ell}{\partial r_l}=+\sigma(-d).
$$

The push on the two scores is equal and opposite, and its size $\sigma(-d)$ is exactly the model's current probability of having the pair *wrong*: confidently right pairs hardly move; wrong pairs get the full push. The scores then flow back through the network by ordinary backpropagation.

**Numbers.** $r_w=1.5$, $r_l=0.5$: $d=1$, $P=\sigma(1)=0.7311$, loss $=0.3133$, gradient on $r_w$ is $-\sigma(-1)=-0.2689$ (and $+0.2689$ on $r_l$). With equal scores the loss is $\ln2=0.6931$. In the exercise, a linear reward $r=\phi^\top\theta$ is fitted on 400 simulated noisy comparisons whose labels come from a true $\theta^\star$; the learned $\theta$ points in the same direction (cosine $0.996$) and reaches $85\%$ pair accuracy, while even the true reward could only reach about $83\%$ on that sample because the labels are noisy by construction.

## RLHF: optimise against the reward, on a leash

Let $\pi_\theta(y\mid x)$ be the policy (the language model being trained) and $\pi_{\rm ref}$ the frozen SFT model. RLHF maximises

$$
J(\pi)=\mathbb E_{x}\,\mathbb E_{y\sim\pi(\cdot|x)}\bigl[r(x,y)\bigr]-\beta\,\mathbb E_x\,\mathrm{KL}\bigl(\pi(\cdot|x)\,\|\,\pi_{\rm ref}(\cdot|x)\bigr),
$$

with $\beta>0$ the strength of the leash. The first term wants high reward; the second penalises distributions that differ from the reference. In practice $y$ is a token sequence sampled from the current policy and the update uses an RL algorithm, usually PPO. The source lesson summarises PPO's role: sample responses, score them, estimate an *advantage* (reward minus a baseline such as the recent average), and update with a clipped objective

$$
\min\bigl(\rho A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\,A\bigr),\qquad \rho=\frac{\pi_{\rm new}}{\pi_{\rm old}},
$$

so that no single sample can cause a huge change ($\epsilon$ is commonly $0.2$). Costs: up to four models in memory (policy, reference, reward model, and often a value baseline), three training stages (SFT, reward model, PPO), and sensitivity to hyperparameters.

### The optimal policy in closed form

Fix a prompt $x$ and drop it from the notation. Over a finite set of responses define

$$
\pi^\star(y)=\frac{\pi_{\rm ref}(y)\,e^{r(y)/\beta}}{Z},\qquad Z=\sum_{y}\pi_{\rm ref}(y)\,e^{r(y)/\beta}.
$$

Claim: $\pi^\star$ maximises $J$. Proof by completing the divergence. For any distribution $\pi$,

$$
\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\ln\frac{\pi(y)}{\pi_{\rm ref}(y)}
=-\beta\sum_y\pi(y)\Bigl[\ln\frac{\pi(y)}{\pi_{\rm ref}(y)}-\frac{r(y)}{\beta}\Bigr].
$$

The bracket is $\ln\dfrac{\pi(y)}{\pi_{\rm ref}(y)e^{r(y)/\beta}}=\ln\dfrac{\pi(y)}{Z\,\pi^\star(y)}=\ln\dfrac{\pi(y)}{\pi^\star(y)}-\ln Z$. Therefore

$$
J(\pi)=-\beta\,\mathrm{KL}(\pi\,\|\,\pi^\star)+\beta\ln Z .
$$

A KL divergence is $\ge0$ with equality only at $\pi=\pi^\star$. So the unique maximiser is $\pi^\star$ and the maximum value is $\beta\ln Z$. (Reading the formula: reweight the reference by $e^{r/\beta}$; small $\beta$ means a sharp preference for high reward, large $\beta$ means staying put.)

**Numbers.** Three responses with $\pi_{\rm ref}=(0.5,0.3,0.2)$ and rewards $(1,0,-1)$.

| $\beta$ | $\pi^\star$ | $\mathbb E_{\pi^\star}[r]$ | $\mathrm{KL}(\pi^\star\|\pi_{\rm ref})$ | $J(\pi^\star)=\beta\ln Z$ |
|---|---|---|---|---|
| 1.0 | (0.7844, 0.1731, 0.0425) | 0.742 | 0.192 | 0.5497 |
| 0.5 | (0.9187, 0.0746, 0.0067) | 0.912 | 0.432 | 0.6958 |
| 0.1 | (0.99997, 0.00003, 0.0000) | 1.000 | 0.693 | 0.9307 |

The reference itself has $\mathbb E[r]=0.3$ and $\mathrm{KL}=0$, so $J=0.3$. As $\beta$ falls, the reward rises but the policy moves further from the reference: that trade is the whole story of RLHF tuning. The exercise checks that none of 3000 random distributions beats $\pi^\star$.

## DPO: the same optimum without the reward model

Rearrange the optimal-policy formula to express the reward through the policy:

$$
r(x,y)=\beta\ln\frac{\pi^\star(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\ln Z(x).
$$

Any reward function has an associated optimal policy, so any reward can be written this way. Now plug it into the Bradley-Terry probability for a pair answering the *same* prompt. The troublesome $\beta\ln Z(x)$ depends only on $x$, so it cancels in the difference:

$$
r(x,y_w)-r(x,y_l)=\beta\ln\frac{\pi(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\beta\ln\frac{\pi(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}.
$$

The preference probability is therefore a function of the policy's own probabilities. Instead of fitting a reward model and then a policy, fit the policy directly by maximum likelihood on the comparisons, the **DPO loss**:

$$
\mathcal L_{\rm DPO}=-\ln\sigma(z),\qquad z=\beta\Bigl[\bigl(\ln\pi(y_w|x)-\ln\pi_{\rm ref}(y_w|x)\bigr)-\bigl(\ln\pi(y_l|x)-\ln\pi_{\rm ref}(y_l|x)\bigr)\Bigr].
$$

Here $\ln\pi(y|x)$ is the sum of token log-probabilities of the whole response, computable with one forward pass per response. The quantity $\hat r(y)=\beta(\ln\pi-\ln\pi_{\rm ref})$ is called the **implicit reward**: the language model is secretly a reward model.

**Gradient.** With $\mathrm{d}\mathcal L/\mathrm{d}z=-\sigma(-z)$, the same computation as for the reward model gives

$$
\frac{\partial\mathcal L}{\partial\ln\pi(y_w)}=-\beta\,\sigma(-z),\qquad \frac{\partial\mathcal L}{\partial\ln\pi(y_l)}=+\beta\,\sigma(-z).
$$

Descent raises the log-probability of the chosen response and lowers that of the rejected one, with a weight $\sigma(-z)$ that is large when the implicit reward currently ranks the pair wrongly. The log-probability gradients then flow into the weights through ordinary backpropagation.

**Numbers.** $\beta=0.1$, policy log-probs $(-12,-15)$, reference $(-13,-14)$: the chosen response became 1 nat more likely than under the reference and the rejected one 1 nat less likely, so $z=0.1\,(1-(-1))=0.2$, $\sigma(0.2)=0.5498$, loss $0.5981$, gradient on $\ln\pi(y_w)$ is $-0.1\cdot\sigma(-0.2)=-0.0450$. If the policy equals the reference, $z=0$, the loss is $\ln2$ and the gradients are $\mp0.05$.

### RLHF versus DPO

| | RLHF with PPO | DPO |
|---|---|---|
| Models in memory | policy, reference, reward model (+ value baseline) | policy, reference |
| Stages | SFT, reward model, PPO | SFT, DPO |
| Data | comparisons for the reward model, then fresh samples from the policy | a fixed set of comparisons |
| Stability | sensitive to hyperparameters | close to ordinary supervised training |

The source reports DPO often matches or beats RLHF with small preference sets and limited compute, while RLHF has advantages when "better" has many dimensions that a learned reward can combine, and for iterative loops where the current policy generates new samples that are re-rated. Related methods remove further pieces: KTO uses unpaired good/bad labels, ORPO folds the preference signal into the SFT loss, SimPO drops the reference model by using length-normalised average log-probability as the reward. (Those are described in the source; they are not derived here.)

## Implementation

```python
import numpy as np

def bt_loss(r_w, r_l):                       # arrays of rewards, one entry per pair
    d = r_w - r_l
    return float(np.mean(np.logaddexp(0.0, -d)))     # = mean(-log sigmoid(d)), stable

def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    z = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return float(np.mean(np.logaddexp(0.0, -z)))

def dpo_grads(lp_w, lp_l, ref_w, ref_l, beta):
    z = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    g = -beta * (1.0 / (1.0 + np.exp(z))) / z.size   # -beta * sigmoid(-z), averaged
    return g, -g

def optimal_policy(ref, r, beta):
    w = np.log(ref) + r / beta
    p = np.exp(w - w.max())
    return p / p.sum()
```

## Shapes

- Reward model: input (prompt+response) tokens $(B,T)$, output scalar per sequence, shape $(B,)$. A batch of pairs gives $r_w,r_l$ of shape $(B,)$ each.
- DPO: four arrays of shape $(B,)$, each the *sum* over response tokens of per-token log-probabilities (so $(B,T_{\rm resp})$ token log-probs are summed over the time axis, with prompt and padding positions masked as in [[llm.04]]).
- Tabular check in the exercise: policy logits $\theta\in\mathbb R^{K}$, $\ln\pi=\theta-\operatorname{logsumexp}(\theta)$, and $\partial\ln\pi_i/\partial\theta_j=\mathbb 1[i=j]-\pi_j$.

## Failure modes

- **Reward hacking (Goodhart's law).** The policy optimises the reward *model*, which is an imperfect proxy trained on finite data. Reported symptoms in the source: *verbosity* (annotators preferred longer answers, so length earns reward), *sycophancy* (agreeing with the user's premise scored well), *hedging* (non-committal answers are rarely marked wrong), *format gaming* (bullet points and headers look polished). Mitigations named there: a stronger KL penalty, training the reward model on adversarial examples, using several reward models.
- **Over-optimisation by drift.** In the tabular exercise, five consistent preferences over four equally likely responses (0 beats 1, 2, 3; 1 and 2 beat 3), $\beta=1$, learning rate $0.5$: after 20 steps the policy is $(0.677,0.146,0.146,0.032)$ with $\mathrm{KL}=0.45$ nats from the reference; after 60 steps $(0.856,0.069,0.069,0.006)$ with $\mathrm{KL}=0.85$; after 1000 steps $\mathrm{KL}=1.32$. Because the labels are perfectly consistent, the loss has no finite minimiser: the probability of the winner keeps rising. Real data has noisy labels, which pull the other way, but the lesson stands: run length, $\beta$ and label noise jointly decide how far you drift from the reference.
- **Reward-model blind spots.** It only knows the region of text it was trained on; the more the policy moves, the more it is asked to score text outside that region.
- **Preference data are human, and so biased.** Annotator preferences for length, confidence and agreeableness become the model's behaviour. Comparisons also do not record *why* an answer is better.
- **Offline limitation of DPO.** It learns from fixed pairs, usually produced by an older model; the source notes that RLHF's online loop (fresh samples from the current policy, re-rated) has an advantage for iterative alignment.
- **Alignment is not truth.** A preference-optimised model is optimised to be *rated* well. Confident wrong answers can be rated well if raters cannot check.

## Common misconceptions

- "The reward model gives the true quality of an answer." It gives a score fitted to some people's comparisons; only differences are meaningful.
- "The KL penalty only slows training." It also defines the optimum: $\pi^\star\propto\pi_{\rm ref}e^{r/\beta}$.
- "DPO has no reward model." It has an *implicit* one, $\beta\ln(\pi/\pi_{\rm ref})$; the derivation shows the loss is equivalent to Bradley-Terry on that reward.
- "DPO needs no reference model." The reference appears in $z$; only some later variants remove it.
- "Higher reward during training means a better model." Past a point it usually means more exploitation of the proxy; evaluate with held-out human or independent judgements.

## Key takeaways

- Comparisons are easier to collect than demonstrations; Bradley-Terry turns them into $P(a\succ b)=\sigma(r_a-r_b)$ and the loss $-\ln\sigma(r_w-r_l)$ with gradient $\mp\sigma(-d)$.
- RLHF maximises $\mathbb E[r]-\beta\,\mathrm{KL}(\pi\|\pi_{\rm ref})$; its unique maximiser is $\pi^\star=\pi_{\rm ref}e^{r/\beta}/Z$ with maximum $\beta\ln Z$ (completing the KL).
- Inverting that formula gives $r=\beta\ln(\pi/\pi_{\rm ref})+\beta\ln Z$; the $Z$ terms cancel in a comparison, giving the DPO loss $-\ln\sigma\bigl(\beta[\Delta_w-\Delta_l]\bigr)$.
- DPO removes the reward model and the RL loop at the price of a fixed offline dataset.
- The leash $\beta$ trades reward against drift; reward hacking, sycophancy and verbosity are the proxy being exploited.

## Explain it yourself

1. Why do the $\ln Z(x)$ terms cancel in the DPO derivation, and what would go wrong if the two responses in a pair answered different prompts?
2. A model's KL from its reference keeps growing during preference training while a held-out human rating stops improving. What is happening, and which two knobs would you change?

## Further reading

- Ouyang et al. (2022), *Training language models to follow instructions with human feedback* (InstructGPT).
- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*; Bradley and Terry (1952), the original pairwise-comparison model.

## Mastery criteria

- Derive the Bradley-Terry loss and its gradient $\mp\sigma(-d)$.
- Derive $\pi^\star=\pi_{\rm ref}e^{r/\beta}/Z$ by completing the KL and state the maximum value.
- Derive the DPO loss from it, including why $Z(x)$ cancels, and compute it for given log-probabilities.
- Name three failure modes and what each tells you about the proxy.
