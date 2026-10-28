---
id: adv.04
title: "Attention variants for long context"
summary: Sliding-window, sparse and global-local masks, MQA/GQA/MLA key-value sharing and differential attention, with code that builds the masks and counts KV bytes for each variant.
minutes: 75
teaches: [adv.attn-variants, adv.kv-sharing]
requires: [trf.attention, trf.multihead, trf.mask]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/15-attention-variants/docs/en.md: sliding-window, sparse and differential attention, masks, comparison table, Gemma-style 5:1 layout. READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/16-differential-attention-v2/docs/en.md: softmax noise floor, differential formula, V1 vs V2. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/17-native-sparse-attention/docs/en.md: three NSA branches (compressed, selected, window), natively trainable. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/docs/en.md: MHA, MQA, GQA, MLA comparison. READ (knob 4)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/docs/en.md: MLA latent cache 7.6 GB vs 30.5 GB GQA at 128k. READ."
---

## Where this fits

Full attention ([[trf.01]]) compares every token with every earlier token: $O(N^2)$ compute, and a key-value (KV) cache ([[sys.01]]) that grows linearly with context in every layer. At 128 000 tokens both become the bottleneck: the cache of a 32-layer, 32-head model alone is 64 GiB. Two families of ideas fight this. **Attend to less** (windows, sparse patterns, summaries) cuts compute and sometimes cache. **Store less per token** (share or compress keys and values) cuts only the cache. A third idea, **differential attention**, improves the quality of what is attended to. Modern models mix all of them; this lesson gives you the masks as code and the KV arithmetic as a calculator.

## The idea in plain words

Reading a long report, you do not re-read every earlier page for each new sentence. You keep the **last few pages** in mind in detail (sliding window), you keep a few **bookmarks** to important places such as the title and table of contents (global tokens), and occasionally you jump to a **chapter summary** (compressed blocks). Together this is far cheaper than rereading everything, and each page you do read was itself written with knowledge of earlier pages, so information still spreads gradually.

For storage: if 32 reviewers each keep their own full copy of the notes (multi-head), memory explodes. Letting groups of reviewers share one copy (grouped-query), everyone share one (multi-query), or keeping a compact shorthand from which each reviewer can reconstruct their copy (latent attention) reduces what you must carry around.

Finally, softmax attention is like a room with a faint hiss: even irrelevant tokens receive a little weight. Take two noisy recordings and subtract one from the other, and the shared hiss cancels while the voice remains: differential attention.

## Masks: attend to less

Attention scores are $S=QK^\top/\sqrt{d}\in\mathbb{R}^{N\times N}$; a mask $M$ sets disallowed entries to $-\infty$ before the softmax ([[trf.02]]), so those keys get exactly zero weight. Boolean mask $M_{ij}=1$ means query $i$ may see key $j$.

| Variant | Allowed set for query $i$ | Entries (causal) | KV kept per layer |
|---|---|---|---|
| Full causal | $j\le i$ | $N(N+1)/2$ | $N$ |
| Sliding window $W$ | $i-W<j\le i$ | about $NW$ | $W$ |
| Local + strided | window, plus $j\le i$ with $j \bmod s=0$ | about $N(W+N/s)$ | all, or only strided |
| Global + local | window, plus a few global positions | about $N(W+g)$ | $W+g$ |

Examples verified by the exercise: for $N=8$, $W=4$: 26 of 36 causal entries. For $N=8192$, $W=1024$: 7 864 832 of 33 558 528, a factor of 4.3. (The source quotes "8x" for this case; that compares against the full square $N^2$, not the causal half that full attention really computes. Both views are informative, but compute against compute.)

**Stacking recovers range.** A token in layer $\ell$ sees only the previous $W$ positions of layer $\ell-1$, whose values themselves summarised $W$ earlier positions. After $L$ layers information can travel $L(W-1)+1$ positions: 131 041 for $W=4096$, $L=32$. In practice recall at that distance is weak, so models interleave a **full-attention layer every few layers**. A Gemma-style ratio of 5 sliding-window layers to 1 global is the textbook default.

**Sparse patterns** (sparse transformer, Longformer, BigBird) pick the pattern beforehand: local window, plus strided or global positions, plus sometimes random links. The mask is easy; the gain only appears when kernels skip the zero blocks. **Native sparse attention (DeepSeek NSA)** instead learns the pattern during pre-training with three parallel branches: attention over *compressed* block summaries (coarse global view), attention over the *fine-grained tokens of the top-scoring blocks* (the compressed scores choose which blocks to load), and a sliding window (local), combined by a learned gate. Because the compressed branch is differentiable and drives the selection, sparse routing is trainable end to end, unlike inference-time pruning applied to a densely trained model.

## KV memory: store less per token

The cache size is

$$
\text{KV}=2\cdot L\cdot h_{kv}\cdot d_h\cdot N_{\text{kept}}\cdot b,
$$

with $L$ layers, $h_{kv}$ KV heads, head dimension $d_h$, tokens kept $N_{\text{kept}}$ and $b$ bytes per value (2 for BF16).

- **MHA**: $h_{kv}=h$ (one KV pair per query head).
- **MQA** (multi-query): $h_{kv}=1$; cache shrinks by $h$, with a small accuracy cost.
- **GQA** (grouped-query): $G$ KV heads, each shared by $h/G$ query heads. Query head $j$ uses KV head $\lfloor j/(h/G)\rfloor$. Llama-3-8B-style: $h=32$, $G=8$.
- **Sliding window**: $N_{\text{kept}}=\min(N,W)$.
- **MLA** (multi-head latent attention, DeepSeek): compress the hidden state $x_t\in\mathbb{R}^d$ to a latent $c_t=x_tW_{DKV}\in\mathbb{R}^{r}$ with $r\ll h\,d_h$, and **cache only $c_t$**. Keys and values are rebuilt on the fly: $k_t=c_tW_{UK}$, $v_t=c_tW_{UV}$, each producing all heads. Because $q_t k_s^\top=q_t W_{UK}^\top c_s^\top$, the up-projection $W_{UK}$ can be folded (absorbed) into the query side, so the full keys need never be materialised. Cache per token per layer is $r$ numbers, not $2h_{kv}d_h$. DeepSeek-V3 has $r=512$ (the source's count). Real implementations also cache a small separate rotary-position key, about 64 extra numbers by my recollection of the DeepSeek-V2/V3 papers, not something I verified in the sources; the exercise has a `rope_dim` parameter so you can include it.

## Differential attention

Softmax never gives exactly zero, and an irrelevant key's score is a small random number around 0, so with $N$ keys the irrelevant mass adds up to $O(1)$ however long the context. Differential attention splits each head's queries and keys into two groups and subtracts the two attention maps:

$$
\operatorname{DiffAttn}(X)=\Big(\operatorname{softmax}\!\big(\tfrac{Q_1K_1^\top}{\sqrt d}\big)-\lambda\,\operatorname{softmax}\!\big(\tfrac{Q_2K_2^\top}{\sqrt d}\big)\Big)V ,
$$

with $\lambda$ a learned scalar per head. Both maps share the diffuse noise floor, which cancels; the map that learns the real dependency is not cancelled. Two checks you can run: if the two maps are identical and $\lambda=1$ the output is exactly zero, and if $V$ is all ones the output is $1-\lambda$ since each softmax row sums to one. Reported effects (from the source, not verified by me): lower perplexity and sharper needle-in-haystack retrieval. DIFF V2 reorganises the heads (more query heads, same KV heads) so that decoding speed and FlashAttention compatibility match the baseline.

## Worked example (all reproduced by the exercise)

Llama-3-8B-shaped model: 32 layers, $d_h=128$, 128K tokens (131 072), BF16.

| Variant | Config | KV cache |
|---|---|---|
| MHA | 32 KV heads | 64 GiB |
| GQA | 8 KV heads | 16 GiB |
| MQA | 1 KV head | 2 GiB |
| GQA + sliding window 4096 on every layer | 8 KV heads, 4096 tokens | 0.5 GiB |
| Hybrid 5:1 (30 layers: 5 global, 25 window 1024), GQA | 8 KV heads | 2.60 GiB, vs 15 GiB all-global |

DeepSeek-V3 shape (61 layers, 128K): MLA with $r=512$ stores $61\cdot512\cdot131072\cdot2$ bytes $=7.625$ GiB; a Llama-70B-style GQA (8 KV heads, $d_h=128$) stores 30.5 GiB, four times more; adding the 64-dim rotary key to MLA gives 8.58 GiB.

## Implementation

```python
import numpy as np

def sliding_mask(n, window):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return (j <= i) & (i - j < window)

def gqa_attention(Q, K, V, mask):            # Q:(H,N,d)  K,V:(G,N,d)
    H, N, d = Q.shape; G = K.shape[0]
    out = np.zeros_like(Q)
    for h in range(H):
        g = h // (H // G)
        s = np.where(mask, Q[h] @ K[g].T / np.sqrt(d), -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        out[h] = (s / s.sum(-1, keepdims=True)) @ V[g]
    return out
```

Production: FlashAttention and PyTorch FlexAttention take mask functions and skip empty blocks; Hugging Face exposes `num_key_value_heads` in model configs for GQA, and sliding-window sizes per layer for Gemma and Mistral.

## Shapes

| Tensor | Shape |
|---|---|
| queries | (H, N, d) |
| keys, values (GQA) | (G, N, d), with H divisible by G |
| mask | (N, N) boolean |
| MLA cache | (N, r) per layer (+ rope key) |
| differential attention | two score maps (N, N), subtracted before multiplying V |

## Common misconceptions

- **"Sliding window = the model forgets everything older than W."** Information propagates through depth, and global layers refresh it; recall is still weaker.
- **"GQA changes the number of query heads."** Only the number of distinct key/value heads.
- **"MLA is a free lunch."** It trades memory for a little extra compute (up-projection or absorbed matrices) and extra design complexity.
- **"A mask saves compute automatically."** Only if the kernel skips masked blocks; a dense mask on a dense kernel saves nothing.
- **"Differential attention subtracts the attention map from values."** It subtracts two attention *maps* before multiplying V.

## Key takeaways

- Masks cut compute (window $W$: about $NW$ instead of $N^2/2$ pairs); global layers or tokens restore long-range access.
- KV cache $=2Lh_{kv}d_hN_{\text{kept}}b$: GQA/MQA reduce $h_{kv}$, windows reduce $N_{\text{kept}}$, MLA stores a latent of size $r$.
- Differential attention cancels common-mode softmax noise: $(A_1-\lambda A_2)V$.
- Always count bytes for your own model before choosing a variant.

## Explain it yourself

1. Derive the KV cache of GQA for $h=32,G=8$ relative to MHA, and explain why query quality is only mildly affected.
2. Why does MLA's absorbed up-projection let you avoid ever forming full per-head keys?

## Further reading

- Longformer, BigBird, Gemma 2/3, GQA (Ainslie et al.), DeepSeek-V2/V3, Differential Transformer, Native Sparse Attention, as summarised in the source lessons.
- [[adv.02]] for video KV costs, [[adv.05]] for replacing the cache by a state, [[adv.03]] for DeepSeek-V3's MoE.

## Mastery criteria

You can write causal, sliding, strided and global-local masks, implement GQA and differential attention, and compute KV-cache size for MHA, MQA, GQA, sliding-window, hybrid and MLA layouts.
