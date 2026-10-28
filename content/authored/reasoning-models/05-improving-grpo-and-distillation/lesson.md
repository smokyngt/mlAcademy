---
id: rsn.05
title: "Improving GRPO and distilling reasoning models"
summary: Monitor a GRPO run, understand the biases hidden in its normalisations, the fixes the community adopted, then compare RL with the cheaper alternative of fine-tuning a small model on a big model's reasoning traces, and see how batching makes all of it affordable.
minutes: 100
teaches: [rsn.grpo-fixes, rsn.distill, rsn.batching]
requires: [rsn.grpo, llm.sft, info.entropy]
sources:
  - "reasoning-from-scratch/ch07/01_main-chapter-code/ch07_main.ipynb (Sec. 7.1-7.8): GRPO metrics (loss, reward, response length, advantage mean/std, entropy), clipped ratio and the token/sequence-level variants table, KL experiment and its failure analysis, <think> tokens and format reward, list of fifteen GRPO modifications. READ."
  - "reasoning-from-scratch/ch08/01_main-chapter-code/ch08_main.ipynb (Sec. 8.1-8.10): hard versus soft distillation, dataset of teacher traces, example building (chat template, <think> format, EOS), length filtering and split, cross-entropy as average log-probability, training loop, results table. READ."
  - "reasoning-from-scratch/chE/chE_main.ipynb (Sec. E.1-E.8): latency versus throughput, padding and attention masks, batched evaluation, sampling, GRPO rollouts and distillation, timing and memory table. READ."
  - "reasoning-from-scratch/ch06/01_main-chapter-code/ch06_main.ipynb (Sec. 6.8-6.12): the baseline GRPO loss and run that chapter 7 improves. READ."
---

## Where this fits

[[rsn.04]] gave you a working GRPO loop. The book's own run shows its weakness: fast early gains, then wobble, and sometimes collapse. The first half of this lesson is a tour of what to watch and of the fixes proposed by the community; the second half is the *other* way to build a small reasoning model, which needs no RL at all: copy the behaviour of a strong one (**distillation**). A short last section covers batching, which is what makes both affordable in practice.

You need the GRPO loss and advantages from [[rsn.04]], the verifier from [[rsn.01]], supervised fine-tuning of a language model ([[llm.04]]) and entropy ([[math.10]]).

## The idea in plain words

Teaching a student by trial and error (RL) is slow, noisy and sometimes the student develops bad habits (rambling, gaming the grader). Three practical questions follow: *How do I tell the training is going well while it is running?* (dashboards: reward, answer length, how varied the attempts are). *How do I stop one loud lesson from shouting over the rest?* (normalisation and clipping). And *is there a shortcut?*

There is: let a very strong teacher write out model solutions, and have the small student copy them word by word, as in a handwriting class. That is **distillation**. It is supervised learning, no sampling by the student, no reward. It is cheap per step, but its ceiling is the teacher, and the quality of the copied text matters a lot. Finally, whichever route you take, the compute bill is dominated by generating long answers, and **batching** (doing many at once) is the main lever to shrink it.

## Part A. Watching a GRPO run

The book tracks the following quantities during a run (it plots them for a 500-step run):

| signal | what you want to see | warning signs |
|---|---|---|
| **average reward** | rising | stuck near 0 (no signal) or reaching 1.0 (nothing left to learn from) |
| **response length** | first rising as the model learns to reason, then stable | collapse to very short outputs, or runaway growth |
| **loss** | roughly flat with small noise | not a progress meter (see [[rsn.04]]); spikes are a concern |
| **advantage std** | stable and non-zero (about 1 is ideal; far below 1 means vanishing signal; far above 1 spiky updates) | going to 0 |
| **entropy** | moderate | collapse to near 0 (every rollout the same) or explosion toward $\log V$ |

The advantage **mean** must be 0 by construction; if it drifts, you have a bug.

**Entropy** of one step's next-token distribution $p=\operatorname{softmax}(z)$ is $H=-\sum_vp_v\log p_v$, between $0$ (one token certain) and $\log V$ (uniform). For the book's seven-token toy logits $[0.6667,-2,1.3333,0,-0.6667,2,-1.3333]$ the probabilities are $0.13,0.01,0.25,0.07,0.03,0.49,0.02$ and $H=1.370$ against a maximum $\log7=1.946$. The tracked number is the average over the generated tokens of a rollout. Rule of thumb from the notebook: below about 0.5 the model is nearly deterministic, 1 to 2 is typical for stable training, far above that the model is close to random. Raising the sampling temperature increases rollout diversity but does *not* change the model's entropy.

The book's baseline run (no clipping, no KL) showed fast early gains in evaluation accuracy then a decline after a few hundred steps, with loss spikes: diminishing returns plus instability.

## Part B. Improving GRPO

### 1. Clipped ratios

The original [[rsn.04]] loss uses a ratio $\rho=\pi_\theta/\pi_{\text{old}}$ that is exactly 1 for a single update. With several updates per batch you need the clipped surrogate $\min(\rho A,\operatorname{clip}(\rho,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})A)$. Variants differ in *where* the ratio is formed:

| method | ratio level |
|---|---|
| DeepSeekMath GRPO | token: $\pi_\theta(o_t\mid q,o_{<t})/\pi_{\text{old}}(\cdot)$ |
| DeepSeek-R1 GRPO | sequence: $\pi_\theta(o\mid q)/\pi_{\text{old}}(o\mid q)$ |
| DAPO | token |
| GSPO | sequence, **length-normalised**: $\bigl(\pi_\theta(y)/\pi_{\text{old}}(y)\bigr)^{1/\lvert y\rvert}$ |

Why length-normalise? A sequence ratio is the exponential of the *sum* of $L$ token log-ratios, so its logarithm grows with length. If each token's probability changed by only $0.005$ nats, a 100-token rollout has $\rho=e^{0.5}=1.65$ (far outside $[0.8,1.2]$ and therefore clipped away), while the length-normalised ratio is $e^{0.5/100}=1.005$. In the book's experiment, adding clipping removed the drop that the baseline suffered around step 400.

**Clip higher** (DAPO): use $\epsilon_{\text{high}}>\epsilon_{\text{low}}$, e.g. 0.28 versus 0.2 (my illustration). With $\epsilon_{\text{high}}=0.2$ a positive-advantage token whose ratio is 1.25 gets no more push; with $0.28$ it keeps its full ratio, leaving room for low-probability tokens to grow.

### 2. The KL term and why the book's run fell apart

Adding $\beta\cdot\operatorname{mean}(\log\pi_\theta-\log\pi_{\text{ref}})$ ([[rsn.04]]), the authors' run collapsed shortly after 50 steps: evaluation accuracy went toward 0 and outputs became gibberish. Their diagnosis:

1. The KL is computed on **summed** sequence log-probabilities, so its magnitude grows with length. With a per-token drift of $0.05$ nats and $\beta=0.02$, the penalty is $0.1$ at 100 tokens and $1.0$ at 1000 tokens, versus a constant $0.001$ if you average. That encourages the optimiser to push length, which the logs showed growing.
2. When rewards collapse (all zero), advantages are zero and the policy-gradient term has **no gradient**, yet the KL term is still there and becomes the only gradient, which pushed the model toward high-entropy output.

Remedies listed: length-normalise the log-probabilities/KL, lower $\beta$ (the author found 0.001 did not help here), reweight the KL by an importance weight, or add a small format reward so advantages do not all vanish. The common recommendation from several papers listed in the notebook (Dr. GRPO, Olmo 3, DeepSeek V3.2 for math) is **no KL term** for math tasks, which also removes the reference model's memory cost.

### 3. Rewards: the format reward

Using `<think>`...`</think>` to separate reasoning from the answer needs the tokens to exist. The base model's tokenizer splits `<think>` into several subword pieces, so the book adds them as special tokens (the reasoning variant's tokenizer already has them) and rewards models that emit them in order. Training the *base* model on this directly works poorly, because it has never seen the tags: first fine-tune on data that contains them (Part C). An unconditional format reward of weight 1 let the model collect reward for tags and degrade after about 100 steps; weight 0.1 or a conditional format reward ([[rsn.04]]) are the fixes proposed.

### 4. Normalisation: what the divisions do

Two divisions in the original recipe change *who gets heard*. The analysis below is my own derivation from the formulas (the notebook lists the corresponding fixes by name but does not derive them), so treat the interpretation as reasoned rather than from the source.

**Per-token weights.** With rollouts $i=1..G$ of lengths $L_i$ and advantages $A_i$, three ways to aggregate the loss $-\sum(\text{weight})\cdot\log p_{\text{token}}$:

| aggregation | weight of each token | total push on rollout $i$ |
|---|---|---|
| sequence sum (the book's) | $A_i/G$ | $A_iL_i/G$ |
| sequence mean (1/length per rollout) | $A_i/(G\,L_i)$ | $A_i/G$ |
| token mean (DAPO "token-level loss") | $A_i/\sum_jL_j$ | $A_iL_i/\sum_jL_j$ |

Example, rewards $[1,1,0,0]$ ($A=\pm0.866$), lengths $[100,100,50,400]$. Sequence sum: the wrong 400-token rollout is pushed down by a total of $86.59$, the wrong 50-token one by $10.82$. Sequence mean: both wrong rollouts get the same total $0.2165$, so **each token of the long wrong rollout is penalised 8 times less** than each token of the short wrong one. Taken to its conclusion this under-punishes rambling mistakes (and under-rewards long correct derivations), which is the bias the length-fix (token-level loss) addresses. Token mean treats every token in the group alike, so the total push is proportional to length.

**Standard-deviation normalisation.** Dividing by $\sigma$ rescales each prompt's advantages. For $G=8$:

| correct $k$ | $\sum\lvert A\rvert$ with $\sigma$ | $\sum\lvert A\rvert$ without |
|---|---|---|
| 1 | 4.948 | 1.750 |
| 4 | 7.482 | 4.000 |
| 7 | 4.948 | 1.750 |

Without the division, a prompt solved by half the rollouts carries $2.3$ times the weight of a prompt solved by one; with it, $1.5$ times. So the division flattens the weight across difficulty, giving very hard and very easy prompts relatively *more* say than raw reward differences would. Dr. GRPO argues this is a bias and removes it; the notebook's list also includes DeepSeek V3.2, which *keeps* the original normalisation. The sources disagree, so treat it as a tunable choice and watch the metrics.

### 5. The rest of the list

The notebook's section 7.7 lists fifteen modifications; the ones implemented in the exercise are:

- **Zero-gradient filtering / active sampling** (DAPO): drop prompts whose group has identical rewards and sample new ones, so every batch carries signal.
- **Token-level loss** (DAPO), **no KL** (DAPO, Dr. GRPO), **clip higher** (DAPO), **no std normalisation** (Dr. GRPO), **sequence-level importance sampling and clipping** (GSPO).

Others in the list that I only name (not derived or tested here): truncated importance sampling, domain-specific KL strengths, reweighted KL, off-policy sequence masking, keeping top-p/top-k sampling masks, per-reward group-wise normalisation (GDPO), clipping importance weights instead of token updates (CISPO).

## Part C. Distilling reasoning traces into a small model

### Hard and soft distillation

Distillation trains a **student** on a **teacher**'s outputs.

- **Hard distillation**: train on the teacher's generated *text* with ordinary next-token cross-entropy, $L=\mathrm{CE}(y_{\text{teacher}},\,\text{student})$. This is plain SFT ([[llm.04]]) on synthetic data and needs no logits. The DeepSeek-R1 smaller models were produced this way from the 671B model's traces.
- **Soft distillation**: also match the teacher's distribution, $L=\mathrm{KL}(p_{\text{teacher}}\Vert p_{\text{student}})$ per token, which needs the teacher's logits and a shared tokenizer.
- **Combined**: $L=\mathrm{CE}+\lambda\,\mathrm{KL}$ (classic knowledge distillation).

Hard distillation is the common choice: proprietary APIs do not expose full-vocabulary logits, soft distillation needs identical tokenization, and storing full distributions for long sequences is expensive. (The notebook also notes that using a provider's outputs for training may violate its terms.) It adds that no recent large-scale head-to-head compares the two fairly, and that data-generation strategy can matter more than hard versus soft.

### The pipeline in the book

1. **Generate** teacher traces for the 12,000 MATH training problems (the same ones used for RL, excluding MATH-500) with DeepSeek-R1 671B through an API; the dataset cost about 50 US dollars in API fees. Each entry holds the problem, the reference answer, the thinking text and the final message.
2. **Format** each answer as `<think>thinking</think>` + blank line + final content.
3. **Tokenise** prompt (with the chat template) and answer, append the end-of-sequence token, remember `prompt_len`.
4. **Filter** by length (the book drops examples over 2048 tokens to fit 15 GB of memory) and split off a small validation set (25 examples).
5. **Train** with AdamW (learning rate $5\times10^{-6}$ in the demo, $10^{-5}$ in the full run), gradient-norm clip 1.0, three epochs, tracking validation loss.

### The loss: cross-entropy on the answer only

For tokens $x_1..x_L$ with the answer starting at index $m$ (`prompt_len`), the model's row $t$ predicts $x_{t+1}$, so

$$
\mathcal{L}_{\text{SFT}}=-\frac1{L-m}\sum_{t=m-1}^{L-2}\log p_\theta\bigl(x_{t+1}\mid x_{\le t}\bigr).
$$

Only answer tokens count (the prompt is given, not learned). This is *exactly* the negative **average** log-probability of the answer from [[rsn.03]], which the notebook verifies numerically against `torch.nn.functional.cross_entropy` (the same number). That equality is why SFT needs no special code: the standard classification loss does it.

### Filtering traces with the verifier

The book's pipeline filters by *length* only and uses the verifier just to measure how accurate the teacher was. A natural extension, which the exercise implements, is **rejection sampling**: keep a trace only if its final answer is verified correct ([[rsn.01]]), its content is non-empty, and it fits the length budget. Dropping wrong traces avoids teaching the student confident mistakes; the cost is less data and a bias toward problems the teacher solves. I did not find an experiment in the sources comparing filtered and unfiltered training, so the benefit is a plausible expectation, not a measured result here.

### Results (Qwen3 0.6B student, MATH-500, 500 problems)

| row | training | accuracy |
|---|---|---|
| 1 | base model | 15.2% |
| 2 | official reasoning variant | 48.2% |
| 3 | GRPO, 50 steps ([[rsn.04]]) | 47.4% |
| 4-6 | distilled on DeepSeek-R1 traces, epochs 1, 2, 3 | 30.6%, 32.4%, 33.6% |
| 7-9 | distilled on Qwen3-235B-A22B traces, epochs 1, 2, 3 | 45.0%, 43.8%, 44.2% |

Final validation loss was about 0.53 to 0.54 for the R1 traces and 0.39 to 0.40 for the Qwen3-235B traces. Observations: (1) distillation lifts a 15% model to 34% to 45% with supervised training only. (2) The **choice of teacher** moved accuracy by more than 10 points: the same-family teacher's traces gave a much lower validation loss, meaning its text is easier for the student to predict. The notebook reports the numbers but not the cause; plausible factors are shared vocabulary and writing style. Treat this as a hypothesis. (3) Extra epochs gave small, possibly noisy changes (standard error about 2 points on 500 problems, [[rsn.01]]). (4) The best distilled model is close to, but not above, GRPO from the same base and the official model; one run each.

The notebook's training run took about 3 hours 5 minutes and 15 GB on a DGX Spark; the GRPO run in [[rsn.04]] took 68 minutes on an H100 for its non-batched script. These are different machines and settings, so do not read them as a head-to-head cost comparison.

## Part D. Batching and throughput

Appendix E separates two goals: **latency** (how fast one prompt is answered) and **throughput** (how many prompts per hour). Batching serves throughput.

**Padding and masks.** Sequences in a batch must form a rectangle, so shorter prompts are padded. For generation the book pads on the **left**, so that every sequence's last real token sits at the same final position where generation continues, and keeps an attention mask (True marks padding) so that pad positions are ignored both as keys and as queries. The exercise's `left_pad` returns exactly such an array and mask.

**Waste.** If lengths in a batch differ a lot, much of the rectangle is padding. Lengths $[10,200,12,190,11,210]$ batched in consecutive pairs need $2\cdot200+2\cdot190+2\cdot210=1200$ cells to hold $633$ real tokens, a waste of $0.4725$. Sorted by length first (pairs $(10,11),(12,190),(200,210)$) the cells are $822$ and the waste $0.2299$. Bucketing by length is the standard remedy.

**What gains from batching** (book's table, wall-clock in minutes):

| task | batch | H100 single to batched | speed-up | DGX Spark single to batched | speed-up |
|---|---|---|---|---|---|
| MATH-500 evaluation | 64 | 90.0 to 16.0 | 5.6x | 174.7 to 108.4 | 1.6x |
| self-consistency sampling | 3 | 252.0 to 129.0 | 2.0x | 340.8 to 243.3 | 1.4x |
| GRPO rollouts | 4 | 68.0 to 19.0 | 3.6x | 63.7 to 23.1 | 2.8x |
| distillation | 4 | 10.9 to 9.1 | 1.2x | 32.8 to 28.2 | 1.2x |

Memory rises with batch size: 1.8 GB to 23.4 GB for evaluation with batch 64, while the other tasks barely change (2.45 GB for self-consistency, 43 to 45 GB for GRPO, 8.3 GB for distillation). Notes from the appendix: batching is not guaranteed to help on small models or weak hardware (padding overhead can eat the gain); self-consistency and GRPO rollouts use the *same* prompt repeated, so they need no padding; self-refinement ([[rsn.03]]) is sequential and is not provided in batched form; distillation batches have different prompt and answer lengths and need padding-aware code. A "stop-aware" variant removes finished rows from the active batch so that early finishers stop costing compute.

## Worked example

Everything numeric above is reproduced by the two exercises: token weights and pushes ($86.5875$, $10.8234$, $0.2165$), mass table ($4.948$, $7.482$, $1.75$, $4.0$), sequence ratio ($1.6487$ versus $1.0050$), KL scaling ($0.1$ versus $1.0$), entropy ($1.3700$), SFT loss as mean negative log-prob, the KL between $(0.5,0.5)$ and $(0.9,0.1)$ ($0.5108$), padding waste ($0.4725$ and $0.2299$) and the speed-ups.

## Implementation

```python
import numpy as np

def sft_loss(logits, token_ids, prompt_len):
    z = logits - logits.max(axis=-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))      # (L-1, V)
    rows = np.arange(prompt_len - 1, len(token_ids) - 1)       # rows predicting answer tokens
    return -np.mean(lp[rows, np.asarray(token_ids)[rows + 1]])

def left_pad(seqs, pad_id):
    m = max(map(len, seqs))
    ids = np.array([[pad_id] * (m - len(s)) + list(s) for s in seqs])
    mask = np.array([[True] * (m - len(s)) + [False] * len(s) for s in seqs])
    return ids, mask
```

In PyTorch the loss is `torch.nn.functional.cross_entropy(answer_logits, answer_targets)`; production RL libraries (the book points to scripts in its supplementary material, not reproduced here) provide the advanced GRPO variants and batched generation.

## Common misconceptions

- "A lower training loss in the KL-regularised run means a better model." The authors' KL run had loss behaving, yet accuracy fell to zero.
- "Distilled models can beat their teacher." Hard distillation copies the teacher's behaviour; the ceiling is the teacher's quality (and the student's capacity).
- "Any teacher is as good as any other." In the book the teacher choice moved accuracy by more than 10 points.
- "Batching always speeds things up." Gains range from 1.2x to 5.6x here and can vanish on small models or CPUs.
- "Remove the std division and the bias is gone." You trade one weighting for another; monitor and compare.

## Key takeaways

- Watch reward, length, advantage std and entropy; the loss value is not a progress meter.
- Clip the ratio (length-normalised for sequence-level ratios), keep the KL term off for math unless you have a reason, avoid unconditional format rewards.
- Aggregation (sum, per-sequence mean, token mean) and the std division decide the per-token weights; they change the length and difficulty biases.
- Hard distillation is SFT on teacher traces with the cross-entropy on answer tokens only; filter by length, and (as an extension) by verified correctness.
- Batching with left padding, masks and length bucketing is the throughput lever; know when it helps.

## Explain it yourself

1. Explain why summing token log-ratios in the importance ratio makes long rollouts get clipped more often, and how length normalisation changes that.
2. Why is the SFT cross-entropy on answer tokens the same quantity as the negative average log-probability of [[rsn.03]]?

## Further reading

- Book chapters 7 and 8, appendices E (batching) and the supplementary scripts listed in the notebooks (not opened).
- DAPO, Dr. GRPO, GSPO, DeepSeek-R1 and DeepSeek V3.2 papers named in the chapter 7 notebook (not opened for this lesson).
- Back to evaluation hygiene in [[rsn.01]]: every comparison above deserves an error bar.

## Mastery criteria

- I can list the metrics to track in a GRPO run and say what each failure looks like.
- I can compute per-token weights under sum, mean and token-mean aggregation, and explain the length bias of the second.
- I can explain why a KL term on summed log-probabilities scales with length and what the book did about it.
- I can build SFT examples from teacher traces, compute the answer-only cross-entropy, and filter traces with a verifier.
- I can estimate padding waste and decide when batching and sorting help.
