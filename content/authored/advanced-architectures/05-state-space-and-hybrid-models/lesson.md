---
id: adv.05
title: "State-space models, Mamba and hybrid architectures"
summary: Linear recurrences with constant memory, the selective (Mamba) idea as code you can run, the parallel scan that makes them trainable, and the Jamba-style attention/SSM hybrid.
minutes: 80
teaches: [adv.ssm, adv.scan]
requires: [la.matrix, alg.exp-log, trf.architectures, adv.kv-sharing]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/docs/en.md: SSM recurrence in brief, S4 vs Mamba vs Mamba-2/3, Jamba 1:7 layout with MoE every other layer, memory budget at 256k. READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/docs/en.md: only searched (grep) for Mamba mentions; not used for content."
---

## Where this fits

Attention ([[trf.01]]) remembers every past token exactly, at the price of a KV cache that grows with the sequence ([[adv.04]] showed how big that gets) and $O(N^2)$ training compute. A **recurrent** model instead squeezes the past into a state of fixed size: constant memory, constant time per generated token. Classic RNNs did this but were slow to train (strictly sequential) and unstable. **State-space models (SSMs)** are linear recurrences that can be trained in parallel; **Mamba** made them *selective* (what to remember depends on the input) and became competitive with transformers. In practice the best long-context systems mix both: a few attention layers for exact recall, many SSM layers for cheap bulk processing. This lesson implements the recurrence, its convolution and scan views, a selective SSM, and the memory arithmetic of a hybrid.

## The idea in plain words

An attention layer is a filing cabinet: every page ever read stays in a drawer and can be retrieved verbatim, but the cabinet grows with every page. An SSM is a **notebook with fixed page count**: after each new sentence you rewrite the notebook by fading old notes slightly and adding a summary of the new sentence. Memory never grows, but old details fade and the notebook may have no room for the one detail you later need.

**Selective** means the writer decides, sentence by sentence, how much to fade the old notes and how much to write. For filler text, write nothing and keep the notes untouched. At a topic change, fade a lot and start fresh. A fixed fading rate (a non-selective SSM) cannot do this.

A **hybrid** keeps a few filing-cabinet layers where exact lookup matters and uses notebooks everywhere else.

## The linear recurrence

For one input channel $x_t\in\mathbb{R}$, a state $h_t\in\mathbb{R}^n$ and diagonal dynamics,

$$
h_t=a\odot h_{t-1}+b\,x_t,\qquad y_t=c^\top h_t ,\qquad a,b,c\in\mathbb{R}^n,\ h_{-1}=0 .
$$

$a$ (elementwise, $|a_i|<1$) is how much of each state coordinate survives, $b$ writes the input, $c$ reads out. Computing $y_t$ needs only $h_{t-1}$ and $x_t$: **$O(1)$ memory and time per token** at inference, independent of history length.

**Where does it come from?** A continuous-time system $\dot h=Ah+Bx$ sampled with step $\Delta$ and the input held constant (zero-order hold) gives

$$
h_t=\bar A\,h_{t-1}+\bar B\,x_t,\qquad \bar A=e^{\Delta A},\quad \bar B=A^{-1}(e^{\Delta A}-I)B .
$$

For small $\Delta A$, $\bar B\approx\Delta B$. For diagonal negative $A$, $\bar A=e^{\Delta A}\in(0,1)$ is the decay, and $\Delta$ is how long the system "listens": a larger $\Delta$ means more forgetting of the past and a stronger write of the present. (The discretisation formulas are standard; the source lesson states the recurrence only at a high level, so this paragraph is general knowledge.)

**Convolution view.** Unrolling, $y_t=\sum_{k=0}^{t}c^\top\operatorname{diag}(a)^k b\;x_{t-k}=\sum_k K_k x_{t-k}$ with kernel $K_k=\sum_i c_ia_i^kb_i$. If $a,b,c$ do not depend on $t$ the model is a **linear time-invariant** convolution, trainable in parallel for all $t$ (S4 evaluates it with FFTs). The exercise checks that the recurrent and convolutional computations agree.

## The selective idea (Mamba)

Make the parameters depend on the input: $\Delta_t,B_t,C_t$ are computed from $x_t$ by small linear layers ($\Delta_t=\operatorname{softplus}(\cdot)>0$). For $D$ channels, each with its own $n$-dimensional state,

$$
h_t=\exp(\Delta_t A)\odot h_{t-1}+(\Delta_t\otimes B_t)\,x_t ,\qquad y_t=h_tC_t ,
$$

with shapes $h_t\in\mathbb{R}^{D\times n}$, $A\in\mathbb{R}^{D\times n}$ (negative), $\Delta_t\in\mathbb{R}^{D}$, $B_t,C_t\in\mathbb{R}^{n}$, $x_t\in\mathbb{R}^D$, $y_t\in\mathbb{R}^D$. Check the behaviour the exercise tests:

- $\Delta_t\to0$: $\bar A\to1$ and $\bar B\to0$: the token is **ignored** and the state carried over unchanged.
- $\Delta_t$ large: $\bar A\to0$: the state is **reset** and the current token dominates.

So $\Delta_t$ acts as an input-dependent forgetting gate; $B_t$ chooses what to write and $C_t$ what to read. This lets the model do things a fixed filter cannot, such as copying certain tokens while skipping filler. The price: with input-dependent parameters the model is no longer a fixed convolution, so we cannot use the FFT. We need another way to parallelise.

## Parallel scan

Each step is an **affine map** $h\mapsto a_th+b_t$, the pair $e_t=(a_t,b_t)$. Composing two maps, apply $e_1$ then $e_2$:

$$
e_2\circ e_1:\; h\mapsto a_2(a_1h+b_1)+b_2=(a_1a_2)\,h+(a_2b_1+b_2),
$$

so $\operatorname{combine}(e_1,e_2)=(a_1a_2,\;a_2b_1+b_2)$. This operator is **associative** (composition of functions always is), and $h_t$ is the $b$-component of $e_t\circ\dots\circ e_0$ applied to $h_{-1}=0$. An associative operator means we can bracket the product any way we like, so a *prefix scan* can be done in $\lceil\log_2T\rceil$ rounds: in round $r$ each element $i\ge2^r$ combines with element $i-2^r$ (Hillis-Steele). Sequential: $T$ dependent steps. Parallel: $\lceil\log_2 T\rceil$ dependent rounds, each fully parallel. The simple version does $O(T\log T)$ total work; work-efficient variants (Blelloch) reach $O(T)$. Real Mamba implementations use a fused GPU kernel for this scan that keeps the expanded state in fast on-chip memory (a systems detail beyond this lesson).

## Worked example

Scalar state, $a=0.5$, $b=1$, $c=1$, input $x=(1,2,0,1)$.

- Recurrence: $h=(1,\;0.5\cdot1+2=2.5,\;0.5\cdot2.5+0=1.25,\;0.5\cdot1.25+1=1.625)$, so $y=h$.
- Impulse $(1,0,0,0)$ gives $1,0.5,0.25,0.125$: the kernel $K_k=0.5^k$.
- Scan elements $e_t=(0.5,\,b_t)$ with $b=(1,2,0,1)$. Round 1 (offset 1): $e_1'=(0.25,\,0.5\cdot1+2)=(0.25,2.5)$, $e_2'=(0.25,\,0.5\cdot2+0)=(0.25,1)$, $e_3'=(0.25,\,0.5\cdot0+1)=(0.25,1)$. Round 2 (offset 2): $e_2''=(0.0625,\,0.25\cdot1+1)=(0.0625,1.25)$, $e_3''=(0.0625,\,0.25\cdot2.5+1)=(0.0625,1.625)$. The $b$-components $(1,2.5,1.25,1.625)$ match. Two rounds for four elements.

## Hybrids and when each wins

**Jamba** (AI21) interleaves layers in blocks of eight: 7 Mamba + 1 attention, with a mixture-of-experts feed-forward ([[adv.03]]) on every other layer. The source reports 52 B total / 12 B active parameters with a 256k window on one 80 GB GPU. Reasoning: attention layers provide exact recall and state tracking; SSM layers supply cheap processing. Ablations reported there found the best trade-off near 1:7 to 1:8; too little attention hurts in-context retrieval, too much costs memory. Jamba 1.5 adds rotary positions to attention layers.

**Memory at 256k tokens (BF16), 32 layers, 32 heads, head dim 128**, computed by the exercise:

| Design | KV / state | Total |
|---|---|---|
| pure transformer, MHA | 32 attention layers | 128 GiB |
| pure transformer, GQA (8 KV heads) | 32 layers | 32 GiB |
| Jamba-shaped 28 SSM + 4 attention (MHA) | KV for 4 layers + tiny states | 16 GiB |

(The source's text prints the hybrid KV as "8.4 GB" next to the formula $2\cdot4\cdot32\cdot128\cdot256\text{k}\cdot2$, which evaluates to 16 GiB, and later says 16 GB; I follow the formula. The SSM state per layer is a few tens of kilobytes to megabytes and does not grow with length.)

| | Transformer | SSM (Mamba) | Hybrid |
|---|---|---|---|
| training cost | $O(N^2)$, parallel | $O(N)$, scan | in between |
| memory per generated token | grows with $N$ | constant | grows slowly (few attention layers) |
| exact recall / copying from long context | strong | weak (fixed-size state) | strong enough |
| short contexts, maturity of tooling | best | good | complex |

SSM-only models match transformers in perplexity at small scale but lag on in-context retrieval and state tracking; hence hybrids. Mamba-3 (as reported in the source) adds a more accurate discretisation, complex-valued state updates and multi-input multi-output projections; I did not study its details.

## Implementation

```python
import numpy as np

def scan_combine(e1, e2):                       # apply e1 first, then e2
    return (e1[0] * e2[0], e2[0] * e1[1] + e2[1])

def parallel_scan(a, b):
    A, B = np.array(a, float), np.array(b, float)
    T, offset = len(A), 1
    while offset < T:
        nA, nB = A.copy(), B.copy()
        for i in range(offset, T):
            nA[i], nB[i] = scan_combine((A[i-offset], B[i-offset]), (A[i], B[i]))
        A, B, offset = nA, nB, offset * 2
    return B

def selective_scan(x, delta, A, B, C):          # x:(T,D) A:(D,n) B,C:(T,n)
    h = np.zeros(A.shape); y = np.zeros_like(x)
    for t in range(len(x)):
        h = np.exp(delta[t][:, None] * A) * h + delta[t][:, None] * B[t] * x[t][:, None]
        y[t] = h @ C[t]
    return y
```

The inner loop over $i$ is the part a GPU runs in parallel. Production: the `mamba-ssm` package (fused CUDA kernels) and Hugging Face Mamba/Jamba classes.

## Shapes

| Tensor | Shape |
|---|---|
| input sequence | (T, D) |
| state $h$ | (D, n), constant in T |
| $A$ | (D, n), negative |
| $\Delta_t$ | (T, D) |
| $B_t,C_t$ | (T, n) |
| scan elements $(a_t,b_t)$ | each (T, ...) |

## Common misconceptions

- **"SSMs have no memory limit."** The state is fixed-size; long histories are compressed lossily.
- **"Selective SSMs are convolutions."** Only the time-invariant ones are; selectivity breaks that and requires a scan.
- **"A parallel scan is faster in FLOPs."** It does more total work than the sequential loop; it wins by depth and hardware utilisation.
- **"Hybrids are just two models glued together."** Layers are interleaved in one residual stack, and the ratio is a tuned hyperparameter.
- **"Constant-time per token means always faster."** Short contexts are dominated by weights, where a good transformer kernel is hard to beat.

## Key takeaways

- A linear recurrence $h_t=a\odot h_{t-1}+bx_t$ gives $O(1)$ inference per token; unrolled it is a convolution.
- Mamba makes $\Delta,B,C$ input-dependent: small $\Delta$ ignores a token, large $\Delta$ resets the state.
- Affine maps compose associatively, $(a_1a_2,\,a_2b_1+b_2)$, so the recurrence can be scanned in $\lceil\log_2T\rceil$ parallel rounds.
- Hybrids use a few attention layers for exact recall and many SSM layers for cheap context; Jamba uses about 1 attention to 7 Mamba layers.

## Explain it yourself

1. Why does selectivity prevent the FFT convolution trick, and what replaces it?
2. A 256k-token chat needs exact quotes from the first message. Which layer type would you rely on, and what would you put in the rest of the model?

## Further reading

- S4 (Gu et al., 2021), Mamba (Gu and Dao, 2023), Mamba-2, Jamba and Jamba 1.5, as summarised in the source lesson.
- [[adv.04]] for KV-cache variants; [[adv.03]] for the MoE part of Jamba.

## Mastery criteria

You can implement the recurrence, convolution and selective scan, prove the scan operator is associative and use it to compute the recurrence in logarithmic depth, and estimate hybrid versus transformer memory.
