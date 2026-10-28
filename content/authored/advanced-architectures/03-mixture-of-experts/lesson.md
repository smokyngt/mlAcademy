---
id: adv.03
title: "Mixture of experts: routing, balance and expert parallelism"
summary: Replace the feed-forward block by many experts and a top-k router, derive and implement the load-balancing loss, and understand capacity, token dropping, shared experts and expert parallelism.
minutes: 75
teaches: [adv.moe, adv.moe-balance]
requires: [trf.block, prob.basics, calc.gradient]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md: FFN swap, router, load-balancing options, shared and fine-grained experts, memory catch, when to use MoE. READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/docs/en.md: knob 5 (dense vs MoE), Mixtral and DeepSeek-V3 parameter figures. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/docs/en.md: DeepSeek-V3 config (256 routed + 1 shared, top-8), auxiliary-loss-free bias balancing, 3 dense + 58 MoE layers. READ."
---

## Where this fits

In a transformer block ([[trf.04]]) the feed-forward network (FFN) holds about two thirds of the parameters, and every token pays for all of them. Scaling quality by scaling a dense model therefore scales cost per token. A **mixture of experts (MoE)** breaks that link: it keeps many FFNs ("experts") but sends each token through only a few. Total parameters, which store knowledge, grow with the number of experts $E$; compute per token grows only with the number $k$ chosen. Most frontier open models are MoE (DeepSeek-V3: 671 B parameters in total, about 37 B active per token). The price is a harder training problem (experts must be used evenly), and a systems problem (all experts must live somewhere and tokens must travel to them). This lesson derives the balancing loss that addresses the first and explains the second. The cost of keeping parameters resident is the memory side of [[sys.02]]; training-time sharding comes in [[adv.06]].

## The idea in plain words

A hospital does not send every patient to every doctor. A receptionist (the **router**) glances at each patient and sends them to the two most relevant specialists (the **experts**), then combines their opinions, perhaps weighting the more relevant one more. The hospital can employ hundreds of specialists, yet each patient only costs two consultations.

Three things can go wrong. If the receptionist sends everyone to the same star doctor, that doctor is swamped and the others never learn anything (**load imbalance**, which also makes training collapse towards one expert). If every doctor has a fixed number of slots per day, overflow patients go untreated (**token dropping**). And a hospital split over many buildings means patients walk between buildings (**expert parallelism** and its all-to-all communication). A few generalists see everyone (**shared experts**) so that common knowledge need not be duplicated in every specialist.

## The MoE layer

Input: $N$ token vectors $x_t\in\mathbb{R}^d$. Router weights $W_r\in\mathbb{R}^{d\times E}$ give logits $z_t=x_tW_r\in\mathbb{R}^E$ and probabilities $p_t=\operatorname{softmax}(z_t)$. Let $S_t$ be the indices of the $k$ largest scores. The layer output is

$$
y_t=\sum_{e\in S_t} g_{t,e}\;\mathrm{Expert}_e(x_t),\qquad g_{t,e}=\frac{\exp z_{t,e}}{\sum_{e'\in S_t}\exp z_{t,e'}} .
$$

The gates are a softmax over the selected logits only, so they sum to one. Each expert is an ordinary FFN (typically SwiGLU with three $d\times d_{ff}$ matrices). With **shared experts**, a few extra FFNs process every token and their outputs are simply added.

Parameter accounting (feed-forward only, SwiGLU): total $=(E+s)\cdot 3 d\, d_{ff}$, active $=(k+s)\cdot 3 d\, d_{ff}$, where $s$ is the number of shared experts. Fine-grained designs (DeepSeek, Qwen-MoE) use many small experts, for example 256 routed experts, top 8, plus 1 shared: the number of possible expert subsets per token, $\binom{256}{8}\approx 4.1\times10^{14}$, dwarfs the 28 of an 8-choose-2 layout, at similar active cost. In DeepSeek-V3 the first three layers are dense and the other 58 are MoE.

## The load-balancing problem and its loss

Left alone, a router tends to **rich-get-richer**: a slightly better expert gets more tokens, trains more, becomes even better. The classic fix adds an auxiliary loss. Define, over a batch of $N$ tokens with $k$ choices each,

- $f_i=\dfrac{1}{Nk}\#\{(t,j): \text{token } t\text{'s } j\text{th choice is expert } i\}$, the **fraction of assignments** to expert $i$ (hard, from top-$k$),
- $P_i=\dfrac1N\sum_t p_{t,i}$, the **mean router probability** of expert $i$ (soft).

$$
\mathcal{L}_{\text{bal}}=\alpha\,E\sum_{i=1}^E f_iP_i,\qquad \alpha\approx10^{-2}\text{ to }10^{-3}.
$$

**Why this form (derivation).**

1. Both $f$ and $P$ are probability vectors: $\sum_if_i=\sum_iP_i=1$.
2. Suppose routing is consistent, $f_i\approx P_i$. Then $\mathcal{L}_{\text{bal}}=\alpha E\sum_iP_i^2$. By Cauchy-Schwarz, $1=(\sum_i P_i)^2\le E\sum_iP_i^2$, so $\mathcal{L}_{\text{bal}}\ge\alpha$, with equality exactly when $P_i=1/E$ for all $i$ (uniform). If everything collapses onto one expert, $\sum P_i^2=1$ and $\mathcal{L}_{\text{bal}}=\alpha E$. The factor $E$ is chosen so that the perfectly balanced value is $\alpha$ regardless of $E$.
3. Why both $f$ and $P$? $f$ is what actually matters (load) but is piecewise constant in the router weights (argmax has zero gradient). $P$ is differentiable. The product gives a differentiable surrogate whose gradient is weighted by true load.

**Gradient.** Treat $f$ as a constant. Since $\partial\mathcal{L}/\partial P_i=\alpha E f_i$ and $P_i=\frac1N\sum_tp_{t,i}$, with $\partial p_{t,i}/\partial z_{t,j}=p_{t,i}(\delta_{ij}-p_{t,j})$ (the softmax Jacobian),

$$
\frac{\partial\mathcal{L}_{\text{bal}}}{\partial z_{t,j}}=\frac{\alpha E}{N}\sum_i f_i\,p_{t,i}(\delta_{ij}-p_{t,j})=\frac{\alpha E}{N}\,p_{t,j}\Big(f_j-\sum_if_ip_{t,i}\Big).
$$

Read it: the logit of expert $j$ is pushed **down** (gradient descent subtracts the gradient) when $f_j$ exceeds the token's probability-weighted average load $\sum_if_ip_{t,i}$, and pushed up when expert $j$ is below it. Over-loaded experts become less attractive, in proportion to how over-loaded they are. This formula was checked against finite differences for the worked example below.

**Alternative: auxiliary-loss-free balancing (DeepSeek-V3).** Add a per-expert bias $b_e$ to the scores used only for **selection**, $S_t=\operatorname{top-}k(z_t+b)$, while gates still use the raw $z$. After each step set $b_e\leftarrow b_e-\gamma\,\operatorname{sign}(c_e-\bar c)$ where $c_e$ is the expert's token count: overloaded experts get a lower bias, underloaded a higher one. No extra loss term competes with the language-modelling objective, and no $\alpha$ to tune; balance is steered outside the gradient.

## Capacity, dropping and parallelism

On accelerators, each expert's batch is a fixed-size buffer. Define the **capacity**

$$
C=\Big\lceil \text{cf}\cdot\frac{N k}{E}\Big\rceil ,
$$

where $Nk/E$ is the perfectly balanced load and the **capacity factor** cf (1.0 to 2.0) is slack. Tokens are processed in order; an assignment arriving at a full expert is **dropped**: that token skips the expert and passes through the residual connection only. Larger cf drops fewer tokens but wastes buffer memory and compute on padding; smaller cf is cheaper and hurts quality. Many modern systems avoid dropping by using dropless kernels (grouped matrix multiplies of variable size) plus balancing.

**Expert parallelism.** With $G$ GPUs, each holds $E/G$ experts. Per MoE layer: an **all-to-all** sends each token's vector to the GPUs holding its chosen experts, experts compute, and a second all-to-all returns the outputs. Each token moves about $2k\,d\,b$ bytes across the network per layer ($k$ destinations, there and back). At large scale this communication, not the matrix multiplications, dominates latency, which is why DeepSeek's DualPipe (see [[adv.06]]) overlaps it with computation. Also, **all experts stay in memory** whether or not they fire: a 671 B model needs about 1.3 TB for 16-bit weights.

## Worked example

Six tokens, $E=3$ experts, top-1, logits

| token | $z_0$ | $z_1$ | $z_2$ |
|---|---|---|---|
| 0 | 2.0 | 0 | 0 |
| 1 | 1.5 | 0 | 0.5 |
| 2 | 2.0 | 1 | 0 |
| 3 | 0 | 1 | 0 |
| 4 | 0 | 0.2 | 1.0 |
| 5 | 1.0 | 0 | 0 |

- Routing: tokens 0, 1, 2, 5 go to expert 0; token 3 to expert 1; token 4 to expert 2. So $f=(4/6,1/6,1/6)=(0.667,0.167,0.167)$.
- Mean probabilities (softmax rows averaged): $P=(0.512,0.255,0.234)$.
- $\mathcal{L}_{\text{bal}}/\alpha=E\sum f_iP_i=3\,(0.667\cdot0.512+0.167\cdot0.255+0.167\cdot0.234)=1.268$, above the balanced value 1.
- Capacity with cf $=1.0$: $C=\lceil6\cdot1/3\rceil=2$. Expert 0 receives four tokens in order 0, 1, 2, 5: it keeps 0 and 1, **drops 2 and 5**. With cf $=1.25$: $C=3$, only token 5 is dropped.
- Gradient on token 0: $\frac{\alpha E}{N}p_{0,j}(f_j-\sum_if_ip_{0,i})$ is positive for expert 0 and negative for experts 1 and 2, so descent lowers $z_{0,0}$ and raises the others for this token.

## Implementation

```python
import numpy as np

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def topk_route(logits, k, bias=None):
    scores = logits if bias is None else logits + bias
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return idx, softmax(np.take_along_axis(logits, idx, axis=1))

def balance_loss(logits, idx, alpha=1.0):
    N, E = logits.shape
    f = np.bincount(idx.ravel(), minlength=E) / idx.size
    return alpha * E * np.sum(f * softmax(logits).mean(0))
```

Production: Hugging Face `transformers` Mixtral and DeepSeek classes, with vLLM and SGLang providing expert-parallel inference.

## Shapes

| Tensor | Shape |
|---|---|
| tokens $x$ | (N, d) |
| router weights | (d, E) |
| logits, probabilities | (N, E) |
| selected indices, gates | (N, k) |
| expert weights (linear toy) | (E, d, d); SwiGLU expert: three (d, d_ff) |
| per-expert buffer | (C, d) with $C=\lceil\text{cf}\,Nk/E\rceil$ |

## Common misconceptions

- **"Active parameters are all that matter for memory."** Memory holds all $E$ experts; only compute is sparse.
- **"The balance loss makes experts equally good."** It equalises load; specialisation is learned separately.
- **"Dropped tokens are lost."** They still pass through the residual stream and attention, but that layer contributes nothing for them.
- **"A bigger $\alpha$ is always safer."** Too strong a penalty forces uniform routing against what the data wants and hurts quality; that tension motivated bias-based balancing.
- **"MoE always beats dense."** At small scale (a few billion active parameters), at the edge, or for latency-bound single-user serving, dense is often better.

## Key takeaways

- Router picks $k$ of $E$ experts per token; gates are a softmax over the selected logits; total parameters scale with $E$, compute with $k$.
- $\mathcal{L}_{\text{bal}}=\alpha E\sum f_iP_i$ is minimised ($=\alpha$) at uniform load; its gradient pushes down experts whose load exceeds the token-weighted average.
- Capacity $\lceil\text{cf}\,Nk/E\rceil$ bounds expert work at the price of dropped tokens; bias-based balancing avoids the auxiliary loss.
- Expert parallelism costs two all-to-alls per layer; shared experts hold common knowledge.

## Explain it yourself

1. Derive why $\alpha E\sum f_iP_i\ge\alpha$ when $f=P$, and when equality holds.
2. A team doubles the number of experts but keeps $k$. State what happens to total parameters, active parameters and memory per GPU on 8 GPUs.

## Further reading

- Switch Transformer (Fedus et al.), Mixtral, DeepSeek-V2/V3 reports, as summarised in the source lessons.
- [[adv.06]] for sharding and pipelines, [[adv.04]] for the attention side of DeepSeek-V3 (MLA).

## Mastery criteria

You can implement top-$k$ routing with a bias-for-selection and raw-logit gates, compute the balance loss and explain its gradient, size expert capacity and count dropped tokens, and compute total versus active parameters including shared experts.
