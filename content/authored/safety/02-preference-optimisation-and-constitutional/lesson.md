---
id: saf.02
title: "Preference optimisation and constitutional methods"
summary: Derive DPO from the KL-regularised RLHF optimum, compare what DPO, IPO, KTO, SimPO and ORPO each optimise, and see how replacing human labellers with principles (RLAIF) changes the kind of error.
minutes: 75
teaches: [saf.pref-optimisation, saf.constitutional-ai]
requires: [saf.goodhart, info.entropy, ml.logreg, llm.dpo]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/03-direct-preference-optimization-family/docs/en.md (whole lesson): DPO derivation outline, IPO, KTO, SimPO, ORPO, BPO, over-optimisation of direct alignment algorithms. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif/docs/en.md (whole lesson): critique-and-revise phase, RLAIF, the 2026 constitution rewrite, constitutional classifiers. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal/docs/en.md (loss definitions only). READ."
---

## Where this fits

[[saf.01]] showed the standard recipe for using human preferences: fit a reward model, then optimise a policy against it with a KL leash. That recipe has two awkward parts. It needs a separate reward model and an unstable RL loop, and it needs *people* to produce every comparison. This lesson covers the two main responses. **Direct alignment algorithms** (DPO and relatives) skip the reward model and train the policy directly on comparisons. **Constitutional AI / RLAIF** replaces some or all of the human labelling with a model that reads written principles. The alignment lesson [[llm.05]] introduces these methods from the engineering side; here we ask what each one *optimises* and which failure modes each one keeps. Both inherit the Goodhart problem of [[saf.01]].

## The idea in plain words

Think of training a new employee by showing them pairs of finished reports and saying "this one is better than that one".

- **The classic recipe** first trains a separate *critic* to imitate your pairwise verdicts, then lets the employee practise against the critic, with a rule: "do not stray too far from how you used to write".
- **The direct recipe** skips the critic. It notices that, with that "do not stray" rule, the best employee can be described *in terms of how much more often they write the better report than before*. So you can adjust the employee directly: raise the odds of the preferred report relative to before, lower the odds of the other.
- **The constitutional recipe** replaces the person who judges with an assistant who reads a short list of principles ("prefer the answer that is honest and not harmful") and judges according to those. You trade a messy, individual, noisy judge for a consistent, legible, but possibly *systematically skewed* one.

Every variant is a different choice about *what the judge sees* (pairs, single thumbs up/down, principles), *how the leash works* (a reference model, length normalisation, a supervised term), and *what to do when the signal is extreme*.

## Derivation: DPO from the RLHF optimum

Start from the closed form of [[saf.01]]. For a prompt $x$, the policy that maximises $\mathbb{E}[r]-\beta\,\mathrm{KL}(\pi\|\pi_{ref})$ is

$$
\pi^*(y|x)=\frac{\pi_{ref}(y|x)\,e^{r(x,y)/\beta}}{Z(x)} .
$$

Take logs and rearrange to express the reward through the policies:

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x).
$$

The Bradley-Terry model says a rater prefers $y_w$ to $y_l$ with probability $\sigma\big(r(x,y_w)-r(x,y_l)\big)$. Substitute the expression above. The term $\beta\log Z(x)$ depends only on $x$ and appears in both rewards, so it **cancels** in the difference:

$$
r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} .
$$

Now replace the unknown optimum $\pi^*$ with the policy $\pi_\theta$ being trained and maximise the likelihood of the observed preferences. The DPO loss is

$$
\mathcal{L}_{DPO}=-\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\log\tfrac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\big]\Big).
$$

No reward model, no sampling loop: only log-probabilities of the two responses under the trained and the reference policy. Define the **implicit reward** $\hat r=\beta\log(\pi_\theta/\pi_{ref})$; DPO is a Bradley-Terry fit of $\hat r$.

**Gradient.** Let $m=\beta[\Delta_w-\Delta_l]$ be the margin, where $\Delta=\log(\pi_\theta/\pi_{ref})$. Since $\frac{d}{dm}[-\log\sigma(m)]=-\sigma(-m)$,

$$
\frac{\partial\mathcal{L}}{\partial\log\pi_\theta(y_w)}=-\beta\,\sigma(-m),\qquad
\frac{\partial\mathcal{L}}{\partial\log\pi_\theta(y_l)}=+\beta\,\sigma(-m).
$$

Examples are weighted by $\sigma(-m)$: pairs the model currently gets *wrong* (small or negative margin) matter most. Notice the gradient only constrains the *difference*; nothing in the loss says the chosen log-probability must rise.

### The assumptions behind the derivation

The derivation assumes the optimum is reachable, the preference data resemble the policy's own samples, and the reference model is a good anchor. The source lesson stresses that none of these hold exactly, and that each family member repairs a different violation.

## The family: what each one optimises

Notation: $\Delta_w,\Delta_l$ are log-ratios to the reference for the winner and loser; $p$ denotes a (length-normalised) sequence probability.

| Method | Input data | Loss (per example) | What it changes versus DPO |
|---|---|---|---|
| DPO | pairs | $-\log\sigma\big(\beta(\Delta_w-\Delta_l)\big)$ | baseline |
| IPO | pairs | $\big(\Delta_w-\Delta_l-\tfrac1{2\beta}\big)^2$ | squared error to a bounded target; margin cannot blow up |
| KTO | single outputs with a good/bad flag | $1-\sigma(\beta\Delta-z_{ref})$ if desirable, $1-\sigma(z_{ref}-\beta\Delta)$ if not (simplified; real KTO also weights gains and losses differently) | no pairs needed |
| SimPO | pairs | $-\log\sigma\big(\tfrac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w)-\tfrac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l)-\gamma\big)$ | no reference model; length-normalised; margin $\gamma$ |
| ORPO | pairs | NLL on $y_w$ plus $\lambda\cdot[-\log\sigma(\log\tfrac{\text{odds}(y_w)}{\text{odds}(y_l)})]$ | single stage from the base model; the NLL term is the regulariser |
| BPO | pairs | DPO plus a correction that discourages the chosen log-probability from falling | targets the "degraded chosen response" effect |

Reading the table as failure modes (this is the source lesson's framing):

- **Unbounded implicit reward gap (DPO).** A tiny preference can still push the margin arbitrarily high. IPO's target $1/(2\beta)$ ties the gap to the preference strength.
- **Degraded chosen responses (DPO).** The loss is satisfied if the rejected log-probability falls faster than the chosen one, so the chosen response can become *less* likely in absolute terms. BPO is the lesson's example of a repair; the lesson reports a gain on a maths benchmark from a pre-print under review, which we cannot check here.
- **Length bias (DPO).** The log-probability of a sequence is a sum over tokens, so a longer response has a larger-magnitude log-ratio. SimPO normalises by length.
- **Pair availability.** KTO needs only thumbs-up/down data, which is far more plentiful.
- **Pipeline simplicity.** ORPO folds supervised fine-tuning and preference learning into one stage.

**Over-optimisation remains.** The source reports a study that trained DPO, IPO and a third direct method over a range of KL budgets and again found the rise-peak-fall gold curve. Direct methods have no explicit reward model, but the implicit reward $\beta\log(\pi_\theta/\pi_{ref})$ is queried on sequences far from the data, with the same consequence. Goodhart's law moved; it did not leave.

### Choosing

The source offers a rule of thumb (not a law): paired data and moderate $\beta$ for DPO; SimPO if length bias is visible; KTO for unpaired feedback; ORPO for a one-stage pipeline; BPO if chosen log-probabilities fall; IPO if preference strengths vary widely. Labs reportedly try several and keep the best per task.

## Constitutional AI and RLAIF

**Phase 1 (supervised).** Sample an answer to a risky prompt, ask a model to critique it against a randomly chosen principle, then to revise it. Fine-tune on the revised answers. The 2022 paper that introduced the method used a small list of principles; the source lesson says 16.

**Phase 2 (RL from AI feedback).** For pairs of answers, a feedback model chooses the better one according to sampled principles. The resulting AI preferences train a reward model (or feed a DPO-style loss), and the rest of the pipeline is RLHF-shaped. "RLAIF" only means that the preference signal is generated by a model.

**Why it is not just "cheaper RLHF".** The error changes character. Human labellers disagree and make independent mistakes; with enough of them the noise averages out. An AI labeller is consistent, so it applies *the same lean* to every item. Its mistakes are systematic: a bias (for example towards longer or more confident answers, or a particular reading of "be honest") does not shrink as you label more data. The upside is that the signal is **legible**: you can read the principle, the critique and the revision, which opaque human scores are not. The source also notes that sycophancy toward a user may drop, since the AI judge has no user to please, while Goodhart's law persists because the proxy is now "a model's interpretation of the principles".

**The 2026 constitution rewrite (as the source describes it).** The source reports a January 2026 revision of Anthropic's constitution that favours explanations over bare rules, orders four priorities (avoiding catastrophic outcomes, following the developer's guidelines, being broadly ethical, being helpful), expresses uncertainty about model moral status, and is released under a CC0 licence. We read this only as a summary inside the repo lesson, not the primary document, so verify against the original before relying on details.

**Constitutional classifiers.** A parallel approach trains small classifiers on principles to gate inputs and outputs rather than changing the model. The source reports compute overhead falling from about 24% in the first version to about 1% in the second, and no universal jailbreak reported as of early 2026. These are developer-reported figures. The design lesson is **layering**: post-training shapes tendencies, classifiers enforce invariants, and neither is sufficient alone.

## Worked example

Numbers are from the exercises.

**Losses on one pair.** Sequence log-probabilities: policy $(-10,-12)$ for (chosen, rejected), reference $(-11,-11)$. Log-ratios $\Delta_w=1$, $\Delta_l=-1$, so $\Delta_w-\Delta_l=2$.

- DPO with $\beta=0.1$: margin $0.2$, loss $=-\log\sigma(0.2)=0.598$. With $\beta=1$: margin 2, loss $0.127$.
- IPO: target $1/(2\beta)$. With $\beta=0.1$ the target is 5, loss $(2-5)^2=9$; with $\beta=1$ the target is 0.5, loss $(2-0.5)^2=2.25$. IPO punishes *overshooting* the target, DPO keeps rewarding larger margins.
- SimPO with lengths 5 and 9, $\log\pi$ = $-10$ and $-18$, $\beta=2$, $\gamma=0.5$: per-token average log-probabilities are $-2$ and $-2$, so the margin is $0-0.5$ and the loss is $0.974$. The raw sum would have favoured the shorter sequence; the average treats them equally.
- ORPO term with $p_w=0.6,p_l=0.3$: log odds ratio $1.253$, loss $0.251$.

**Does DPO recover the true reward?** Take the toy distribution of [[saf.01]] with $\beta=0.5$. Compute the exact optimum $\pi^*$ and the implicit reward $\beta\log(\pi^*/\pi_{ref})$ per answer: $(-0.036,\ 0.264,\ -1.036)$. Differences match the true reward differences exactly (the correct answer minus the wrong one is $1.0$ in both). The constant $\beta\log Z$ is the only thing the loss cannot see, which is why rewards are identified only up to a per-prompt shift.

**Random versus systematic labeller error** (second exercise). Items have a true quality gap $d$ and an independent length gap $l$. One human annotator with unit noise gets the sign of $d$ right about 75% of the time; majorities of 5, 15 and 51 reach about 84%, 90% and 95%. A deterministic AI labeller that adds $0.8\,l$ to $d$ is right about 78.5% of the time -- better than one human -- and stays there however often you ask, because its error is the same every time.

## Implementation

```python
import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    m = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -np.log(sigmoid(m))

def dpo_grad(lp_w, lp_l, ref_w, ref_l, beta):
    m = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -beta * sigmoid(-m), beta * sigmoid(-m)
```

In production (for example the TRL library's trainers) the same losses are computed on batches of per-token log-probabilities summed over response tokens, with the reference model's log-probabilities cached.

## Shapes

For a batch of $B$ pairs, the model produces per-token log-probabilities of shape `(B, T)` for each of chosen and rejected; mask padding, sum over $T$ to get `(B,)` sequence log-probabilities, and the loss is a vector of shape `(B,)` averaged over the batch. SimPO divides each sum by the response lengths `(B,)`.

## Common misconceptions

- **"DPO has no reward model, so it cannot reward-hack."** It has an implicit one; the source reports the same over-optimisation curves.
- **"Higher DPO margin means a better model."** The margin is unbounded and can grow by lowering the rejected probability while the chosen one also falls.
- **"AI feedback is unbiased because it is consistent."** Consistent means *systematically* biased; more labels do not average it away.
- **"A constitution makes the model follow the principles."** It changes the training signal. Whether behaviour follows is an empirical question to be tested (see [[saf.04]]).
- **"The newest variant is best."** The source's own advice is to run several and pick per task.

## Key takeaways

- DPO follows from the closed-form RLHF optimum: the partition function cancels in the Bradley-Terry difference, leaving a loss in policy log-ratios.
- Each variant repairs one assumption: IPO (unbounded margin), KTO (pair availability), SimPO (length bias, reference), ORPO (two-stage pipeline), BPO (degraded chosen responses).
- Direct methods still over-optimise; the proxy is implicit rather than absent.
- RLAIF swaps random labeller noise for systematic, legible bias; layering with classifiers is the practical response.
- Several recent figures in this area come from pre-prints or lab reports summarised second-hand; treat them as claims to check.

## Explain it yourself

1. Explain why the partition function disappears in the DPO derivation and what that implies about what the implicit reward can and cannot identify.
2. A team replaces 50,000 human comparisons with 500,000 AI comparisons from one judge model. What kind of error do they gain, and what kind do they lose?

## Further reading

- The papers named inside the source lessons: the DPO paper, IPO (Azar et al.), KTO (Ethayarajh et al.), SimPO (Meng et al.), ORPO (Hong et al.), the Constitutional AI paper (arXiv:2212.08073) and the over-optimisation study for direct alignment algorithms (arXiv:2406.02900). The alignment-engineering view is in [[llm.05]].

## Mastery criteria

- Derive the DPO loss from the KL-regularised optimum and say where $Z(x)$ cancels.
- Compute DPO, IPO and SimPO losses by hand for a pair.
- Name the failure each of IPO, KTO, SimPO, ORPO and BPO targets.
- Explain the difference between random and systematic labeller error with a number.
