---
id: sys.02
title: "Quantisation, speculative decoding and mixture of experts"
summary: Three ways to make big models cheaper to serve - fewer bits per weight, several tokens per expensive pass, and activating only part of the network - with the maths of scale and zero point, the accept/reject rule that keeps the output distribution exact, and the speed-up formula.
minutes: 90
teaches: [llm.quantization, llm.speculative]
requires: [llm.kv-cache, prob.rv, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/11-quantization/docs/en.md (Number Formats, How Quantization Works, Sensitivity Hierarchy, Quality Measurement, Real Numbers): formats, scale formulas, per-channel, asymmetric, sensitivity ranking, PTQ vs QAT. READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/16-speculative-decoding/docs/en.md: draft/verify algorithm, residual distribution, speed-up rule of thumb. READ (Problem, Core algorithm, What determines speedup)."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/16-speculative-decoding/code/main.py: accept test min(1,q/p), residual, chi-square check of equivalence, expected_tokens_per_verify. READ (first half)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/12-inference-optimization/docs/en.md (Prefill vs Decode): why decode is memory-bound. READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md (Problem, FFN swap, load balancing, shared and fine-grained experts): router, top-k, total vs active parameters. READ."
---

## Where this fits

A 70-billion-parameter model stored in 16-bit numbers occupies 140 GB. No single consumer card holds that, and even on a data-centre GPU the cost of generating a token is dominated by reading those 140 GB from memory. The previous lesson showed that decode is memory-bound. This lesson gives three families of answers, each attacking a different term:

1. **Quantisation**: store the numbers in 8 or 4 bits. Fewer bytes to read, fewer bytes to store.
2. **Speculative decoding**: let a small model guess several tokens, and let the big model check all the guesses in a single pass. Fewer *passes* of the big model.
3. **Mixture of experts (MoE)**: build the network so that each token only uses a fraction of its parameters. Less *work per token*.

Quantisation changes the weights slightly, so you must measure the damage. Speculative decoding looks like an approximation but, done with the right rule, produces exactly the same distribution of outputs as the big model. That "exactly" is the delicate part, and we prove it.

## The idea in plain words

**Quantisation.** A ruler marked in millimetres measures more finely than one marked in centimetres, but needs ten times more marks. Quantisation chooses a ruler with fewer marks (256 of them for 8 bits) and places it carefully over the range where the numbers live. Every number is rounded to the nearest mark; the error is at most half a gap. The only decisions are *where to put the ruler* (the range) and *how many rulers* to use (one for the whole matrix, or one per row). If one huge number stretches the ruler, the marks become coarse for everyone else: that is the problem of **outliers**.

**Speculative decoding.** A senior editor (the big model) is slow; an intern (the small model) is fast but sometimes wrong. The intern drafts the next five words. The editor reads all five at once, which costs about the same as writing one word because reading in parallel is cheap, and approves words from the left until the first one they dislike, writing a replacement there. If the intern is usually right, the team produces several words per editor-turn. The surprise is that, with a carefully chosen approval rule, the final text is statistically *exactly* what the editor would have written alone.

**Mixture of experts.** A hospital does not send every patient to every specialist. A receptionist (the router) sends each patient to one or two relevant specialists. The hospital employs many specialists (many parameters) yet each patient uses only a few (little compute).

## Number formats in one table

| Format | Bits | Layout (sign/exponent/mantissa) | Note from the source |
|---|---|---|---|
| FP32 | 32 | 1/8/23 | accumulation, classic training |
| FP16 | 16 | 1/5/10 | small range; loss scaling needed in training |
| BF16 | 16 | 1/8/7 | FP32 range, less precision; common training format |
| FP8 | 8 | E4M3 or E5M2 | inference (E4M3), gradients (E5M2) |
| INT8 | 8 | 256 evenly spaced levels + a scale | integer arithmetic is cheap |
| INT4 | 4 | 16 levels + a scale | needs careful methods |

Floating-point formats spend bits on an exponent so that spacing between representable numbers grows with magnitude. Integer formats have uniform spacing and rely on a **scale** that maps integers back to real values.

## Quantisation: formal definition

Choose a bit width $b$, a real **scale** $s>0$ and an integer **zero point** $z$. Quantise and dequantise:

$$
q = \operatorname{clip}\big(\operatorname{round}(x/s) + z,\; q_{\min},\; q_{\max}\big),\qquad \hat{x} = s\,(q - z).
$$

**Symmetric** scheme: $z=0$, integers in $[-q_{\max}, q_{\max}]$ with $q_{\max}=2^{b-1}-1$ (127 for int8). Pick $s$ so that the largest magnitude lands on the largest integer:

$$
s = \frac{\max_i |x_i|}{q_{\max}} .
$$

**Asymmetric** scheme: unsigned integers $0\ldots 2^b-1$ and a range $[\ell, u]$ that must contain the real number 0 (so that zero, which is everywhere - padding, ReLU outputs - is represented exactly). Derivation: we want an affine map $x\mapsto q$ with $\ell\mapsto 0$ and $u\mapsto 2^b-1$. The slope is $1/s$ with

$$
s=\frac{u-\ell}{2^b-1},\qquad q=\frac{x-\ell}{s}=\frac{x}{s}+\underbrace{\Big(-\frac{\ell}{s}\Big)}_{z},
$$

and since $z$ must be an integer, $z=\operatorname{round}(-\ell/s)$. Then $x=0$ gives $q=z$ and dequantises to $s(z-z)=0$ exactly.

## Derivation: the rounding error

Rounding to the nearest integer moves $x/s$ by at most $1/2$, so $|\hat x - x|\le s/2$ for every element that is not clipped. If the rounding error is spread roughly uniformly over $[-s/2, s/2]$, its variance is

$$
\mathbb{E}\big[(\hat x - x)^2\big]=\frac1s\int_{-s/2}^{s/2} e^2\,de=\frac{s^2}{12}.
$$

Two consequences. Error depends on the *scale*, not on the values themselves: a wide range (large $\max|x|$) means a large $s$ and larger error for every element. And each bit removed doubles $s$, so the MSE is multiplied by 4 per bit.

**Per-tensor vs per-channel.** One scale for a whole weight matrix is dictated by the single largest weight. With one scale per output row (per channel) each row gets its own ruler, so an outlier only hurts its own row. The price is storing one scale per row.

## Worked example

All numbers come from the exercise code.

**Symmetric int8.** $x=[-1,\,-0.3,\,0,\,0.4,\,2.54]$. $\max|x|=2.54$, so $s=2.54/127=0.02$ and $q=[-50,-15,0,20,127]$. Every reconstruction error is at most $s/2=0.01$ and in this case is exactly 0.

**Asymmetric int8.** $y=[-1,\,0,\,0.5,\,3]$: range $[-1,3]$, $s=4/255\approx0.01569$, $z=\operatorname{round}(63.75)=64$, $q=[0,64,96,255]$. The real 0 maps to $q=64$ and back to exactly $0$.

**One-sided data (like a ReLU output).** $[0,0.5,1,2,6]$. Symmetric spends half of its integers on negative values that never occur: $s=6/127=0.0472$, MSE $1.40\times10^{-4}$. Asymmetric uses all 256 levels: $s=6/255=0.0235$, MSE $3.46\times10^{-5}$, four times smaller.

**Outliers.** A $64\times32$ Gaussian matrix with standard deviation 0.05, in which row 5 is multiplied by 40. Per-tensor scale $s=0.0306$ is set by the outlier row: MSE $7.74\times10^{-5}$. Per-channel scales: MSE $1.19\times10^{-6}$, a $65\times$ improvement. On the *other* 63 rows, the error drops from $7.7\times10^{-5}$ to $7.2\times10^{-8}$: the outlier row ruined every other row under per-tensor scaling.

**Bits sweep** (2000 standard normal values): MSE $5.5\times10^{-5}$ at 8 bits, $9.1\times10^{-4}$ at 6, $1.9\times10^{-2}$ at 4, $0.73$ at 2. The formula $s^2/12$ predicts $5.5\times10^{-5}$, $9.2\times10^{-4}$, $1.8\times10^{-2}$ and $0.88$. At 2 bits clipping and coarse levels break the uniform-error assumption.

**Memory:** 70 billion parameters take 140 GB in 16-bit, 70 GB in 8-bit and 35 GB in 4-bit (ignoring scale storage and layers kept in higher precision).

## What is safe to quantise

According to the sensitivity hierarchy in the quantisation material: **weights** (roughly Gaussian near zero) are the most robust; **activations** have outliers (a single channel can be 100 times larger than the mean) and need per-channel/per-token scales or keeping outlier channels in higher precision; the **KV cache** is more sensitive because errors accumulate over all later tokens; **attention logits** before the softmax are the most sensitive and are usually kept in 16-bit. *Post-training quantisation* (PTQ) rounds an existing model, fast and cheap; *quantisation-aware training* (QAT) trains with the rounding in the loop. Methods such as GPTQ and AWQ use calibration data to choose roundings that minimise the output error; they are mentioned here, not derived. The source's table of figures for a 70B model (perplexity and benchmark changes per format) comes from that material and should be read as indicative, since results differ across models, tools and benchmarks. Always measure perplexity and your own task metric before and after.

## Speculative decoding

### The algorithm

Let $q(\cdot\mid \text{ctx})$ be the target (big, slow) model and $p(\cdot\mid\text{ctx})$ the draft (small, fast). One round with $N$ drafts:

1. Draft $d_1,\dots,d_N$ autoregressively from $p$.
2. Run $q$ **once** on the context plus all drafts; this yields $q$'s distributions at all $N+1$ positions simultaneously (this is a prefill-like, parallel pass).
3. For $i=1,2,\dots$: accept $d_i$ with probability $\min\!\big(1,\ q_i(d_i)/p_i(d_i)\big)$.
4. At the first rejection, discard the remaining drafts and draw the replacement token from the **residual** distribution
$$
r(x)=\frac{\max\big(0,\;q(x)-p(x)\big)}{\sum_y \max\big(0,\;q(y)-p(y)\big)} .
$$
5. If all $N$ drafts were accepted, draw one bonus token from $q_{N+1}$.

### Why the output has exactly the distribution $q$

Look at a single position. The probability that the final token equals $x$ is

$$
\Pr[x]=\underbrace{p(x)\min\!\Big(1,\tfrac{q(x)}{p(x)}\Big)}_{\text{drafted as }x\text{ and accepted}}+\underbrace{\Pr[\text{reject}]\; r(x)}_{\text{replacement}} .
$$

The first term is $\min(p(x),q(x))$. The probability of rejection is $1-\sum_y\min(p(y),q(y))=:1-\alpha$, where $\alpha=\sum_y\min(p(y),q(y))$ is the acceptance rate. Since $q-\min(p,q)=\max(0,q-p)$, the normaliser of the residual is $\sum_y\max(0,q(y)-p(y))=1-\alpha$. Therefore the second term is $(1-\alpha)\cdot\frac{\max(0,q(x)-p(x))}{1-\alpha}=\max(0,q(x)-p(x))$, and

$$
\Pr[x]=\min(p(x),q(x))+\max(0,q(x)-p(x))=q(x).
$$

Applying this position after position (each position conditioned on the accepted prefix) shows the whole generated sequence follows $q$. The draft only influences *speed*, never the distribution. If $p$ is a poor guess, $\alpha$ is small and you waste work, but the text is still distributed as if the target had written it alone.

### Expected speed-up

Suppose each draft token is accepted independently with the same probability $\alpha$ (a modelling assumption: in real models $\alpha$ varies with context). A round produces the accepted prefix plus exactly one more token (the correction, or the bonus). The expected number of tokens is

$$
\mathbb{E}[\text{tokens}]=1+\alpha+\alpha^2+\dots+\alpha^N=\frac{1-\alpha^{N+1}}{1-\alpha}.
$$

Let one draft step cost a fraction $c$ of a target pass (for example $c=0.1$ if the draft is ten times cheaper). A round costs $Nc+1$ target-pass equivalents, so

$$
\text{speed-up}\;\approx\;\frac{1-\alpha^{N+1}}{(1-\alpha)(Nc+1)} .
$$

For $\alpha=0.8$, $N=4$: $3.36$ tokens per round; with $c=0.1$ the cost is $1.4$, giving $2.40\times$. If the draft is expensive ($c=0.5$) it falls to $1.12\times$; if the draft is poor ($\alpha=0.5$, $c=0.1$) it is $1.38\times$. Choosing $N$ is a balance: more drafts raise the numerator with diminishing returns ($\alpha^{N+1}$) while the cost grows linearly. The source mentions typical practical speed-ups of 2 to 4 times and variants (Medusa heads, EAGLE, lookahead) that avoid a separate draft model; those figures are from the course material and depend on hardware and workload.

### Worked single-position example

Target $q=[0.5,0.25,0.15,0.07,0.03]$, draft $p=[0.3,0.3,0.2,0.1,0.1]$. Then $\min(p,q)=[0.3,0.25,0.15,0.07,0.03]$, so $\alpha=0.8$. The residual numerator is $[0.2,0,0,0,0]$, normalised to $[1,0,0,0,0]$. Output probabilities: token 0 gets $0.3+0.2\cdot1=0.5$, token 1 gets $0.25$, ... exactly $q$. The exercise checks this empirically with a chi-square test on a two-token bigram model: the speculative sampler passes, and sampling from the draft alone fails by a wide margin.

## Mixture of experts in a nutshell

A dense transformer applies its feed-forward network (FFN) to every token, so compute per token is proportional to the parameter count. In an MoE layer the FFN is replaced by $E$ independent expert FFNs and a **router**, a linear layer $x\mapsto \text{scores}\in\mathbb{R}^E$. For each token take the top-$k$ scores, apply a softmax over those, and output the weighted sum of the $k$ selected experts' outputs:

$$
y=\sum_{e\in\text{top-}k}g_e(x)\,\text{Expert}_e(x).
$$

Total parameters grow with $E$; compute per token grows with $k$. The source gives examples such as 671B total parameters with 37B active per token, and describes typical modern settings of many small experts with a larger $k$ plus some always-on shared experts; these figures are quoted from the material, not verified here. The central difficulty is **load balancing**: if the router sends most tokens to a few experts the rest never learn and the busy ones become a bottleneck. Remedies in the sources: an auxiliary loss on usage variance, capacity limits that drop overflow tokens, or a per-expert bias that nudges selection without touching the training loss. MoE saves compute but not memory: all experts must still be stored, so serving needs lots of memory (and communication across devices).

## Implementation

```python
def quantize_sym(x, bits=8, axis=None):
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(x), axis=axis, keepdims=axis is not None)
    scale = np.where(amax == 0, 1.0, amax / qmax)
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int32)
    return q, scale

def accept_or_replace(q, p, d, u, u2):
    """d drafted from p; u, u2 uniform. Returns the emitted token."""
    if u < min(1.0, q[d] / p[d]):
        return d
    r = np.maximum(q - p, 0.0)
    return int(np.searchsorted(np.cumsum(r / r.sum()), u2, side="right"))
```

## Shapes

| Object | Shape |
|---|---|
| weight matrix $W$ | $(d_\text{out}, d_\text{in})$ |
| per-tensor scale | scalar |
| per-channel (per-row) scale | $(d_\text{out}, 1)$ so it broadcasts against $W$ |
| quantised weights | same shape as $W$, int8 |
| draft/target distributions at one position | $(V,)$ |
| verification pass over $N$ drafts | target logits $(N+1, V)$ in one call |
| MoE router scores | $(\text{tokens}, E)$; top-$k$ indices $(\text{tokens}, k)$ |

## Common misconceptions

1. "Speculative decoding trades quality for speed." With the accept/reject rule above it does not: the distribution is exactly the target's. (Hardware non-determinism and numerical differences between batched and unbatched passes can still change individual samples.)
2. "Accepting a draft only if it equals the target's argmax is the same thing." That is a different, greedy-only rule. For sampling, the $\min(1,q/p)$ rule with the residual is what preserves the distribution.
3. "4-bit means 4 times worse." Error depends on the scale; careful per-channel/group scaling and calibration keep the loss modest, and naive per-tensor int4 can fail badly.
4. "MoE models are small because few parameters are active." They are cheap to run per token but large to store.
5. "More draft tokens is always better." Past the point where $\alpha^{N+1}$ is small, extra drafts only add cost.

## Key takeaways

- Quantisation maps reals to integers with a scale (and zero point if asymmetric); rounding error is at most $s/2$ and about $s^2/12$ in MSE.
- Outliers inflate the scale; per-channel (or finer) scales isolate them.
- Zero must be exactly representable; asymmetric ranges handle one-sided data.
- Speculative sampling: accept with $\min(1,q/p)$, otherwise sample from $(q-p)_+$ normalised; the output law equals $q$ exactly.
- Expected tokens per round $=(1-\alpha^{N+1})/(1-\alpha)$; speed-up divides by the round cost $Nc+1$.
- MoE decouples total parameters from per-token compute, at the price of memory and routing balance.

## Explain it yourself

1. Why does one outlier weight in a matrix make the other weights less precise under per-tensor quantisation, and what does per-channel quantisation change?
2. In speculative decoding, why must the replacement token be sampled from the residual rather than from $q$ directly? (Compute what happens to token 0 in the worked example if you sampled from $q$ on rejection.)

## Further reading

- Leviathan et al., "Fast inference from transformers via speculative decoding" (2023), and Chen et al. (2023): the sources cite both for the accept/reject rule.
- Dettmers et al., LLM.int8() (outlier handling), GPTQ and AWQ papers: named in the quantisation material.
- Switch Transformer, Mixtral and DeepSeek-V3 reports for MoE routing and balancing.

## Mastery criteria

- [ ] I can derive scale and zero point for symmetric and asymmetric int8, and bound the error.
- [ ] I can explain why per-channel quantisation helps with outliers and verify it numerically.
- [ ] I can prove that the accept/reject-plus-residual rule samples from $q$.
- [ ] I can compute expected tokens per round and the speed-up for given $\alpha$, $N$, $c$.
- [ ] I can state what MoE saves (compute per token) and what it does not (memory).
