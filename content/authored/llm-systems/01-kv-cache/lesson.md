---
id: sys.01
title: "KV cache and efficient autoregressive generation"
summary: Why naive token-by-token generation redoes the same work over and over, how caching keys and values removes it, what the cache costs in memory, and how grouped-query attention shrinks it.
minutes: 70
teaches: [llm.kv-cache]
requires: [la.matrix, nn.layer, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/12-kv-cache-flash-attention/docs/en.md: problem statement, per-token cache formula, FlashAttention and continuous batching overview. READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/12-inference-optimization/docs/en.md (Problem, Prefill vs Decode): compute-bound prefill, memory-bound decode, ops:byte ratio. READ (first part)."
  - "LLMs-from-scratch/ch04/03_kv-cache/README.md: the 'Time flies' picture of reusable key/value vectors. READ."
  - "LLMs-from-scratch/ch04/04_gqa/README.md: grouped-query attention, heads sharing K/V. READ."
  - "LLMs-from-scratch/pkg/llms_from_scratch/kv_cache/generate.py: structure of generation with and without a cache (prime the cache with the prompt, then feed one new token per step). READ."
---

## Where this fits

When you chat with a language model, the words appear one after another. That is not a UI trick: the model really does produce one token per forward pass, and each token depends on all the previous ones. A reply of 500 tokens therefore needs 500 sequential passes. If every pass re-read and re-processed the whole conversation from scratch, long chats would become unusably slow and expensive. Every serious serving system (vLLM, llama.cpp, TensorRT-LLM) avoids this with a **KV cache**, and the memory this cache needs is, in practice, what limits how many users one GPU can serve and how long a context can be.

Before this lesson you built attention (queries, keys, values, softmax) and a decoder block. After it you can reason about the cost of generation in FLOPs and in gigabytes, which is the entry ticket to the next lesson (quantisation and speculative decoding are two more ways to attack the same bottleneck).

## The idea in plain words

Imagine writing a story with a friend who must re-read the whole story from page one every time you add a word before they can suggest the next word. The friend's notes about page one never change, but they rewrite them each time. A smarter friend keeps a notebook: for every word already written, one short note ("what this word offers to later words" and "what this word is looking for"). When a new word arrives they write one new note and then consult the whole notebook.

In attention, each token produces three vectors: a **query** (what I am looking for), a **key** (what I offer for matching), and a **value** (what I hand over if I am matched). Crucially, in a decoder with a causal mask the key and value of token $i$ depend only on tokens up to $i$. Once computed, they never change when later tokens are appended. So they can be stored, and each new step only has to compute the new token's three vectors and compare its query against all stored keys.

Two phases follow from this:

- **Prefill**: the prompt is known in advance, so all its tokens are processed in parallel in one big matrix multiplication (fast per token, limited by compute) and their keys and values fill the cache.
- **Decode**: after that, one token per step. Each step does very little arithmetic but has to stream the model weights and the whole cache out of memory, so it is limited by memory bandwidth rather than compute.

The price of the notebook is memory, and it grows with every token.

## Formal setting and the cost of the naive method

Single head, model width $d$ (call it $d_\text{model}$), head width $d_h$. Let $X \in \mathbb{R}^{t\times d}$ hold the $t$ token embeddings so far and $W_q, W_k, W_v \in \mathbb{R}^{d\times d_h}$.

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v,\qquad
O = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}} + M\Big)V
$$

with $M$ the causal mask ($-\infty$ above the diagonal). Only the last row of $O$ is needed to choose the next token.

**FLOP convention.** A product of an $(a,b)$ and a $(b,c)$ matrix costs $2abc$ FLOPs (one multiply and one add per term). We ignore softmax, layer norm, MLP and the output projection; they are smaller or identical for both methods.

## Derivation: counting the work

**Naive step with $t$ tokens present.**

1. The three projections of all $t$ rows: $3\times 2\,t\,d\,d_h = 6\,t\,d\,d_h$.
2. Scores $QK^\top$: $2\,t^2 d_h$ if you compute the full square (a causal kernel needs only the lower triangle, $t(t+1)/2$ entries, which gives $t(t+1)d_h$).
3. Weighted sum $PV$: again $t(t+1)d_h$ with the triangle.

Total with the triangular count:

$$
F_\text{naive}(t) = 6\,t\,d\,d_h + 2\,t(t+1)\,d_h .
$$

Notice this is $O(t^2)$ per step, hence $O(T^3)$ for generating a length-$T$ sequence if you sum over steps (the projections give $O(T^2)$, the attention $O(T^3)$).

**Cached step with $t$ tokens present.**

1. Projections of **one** token: $6\,d\,d_h$.
2. One query against $t$ keys: $2\,t\,d_h$; weights times $t$ values: $2\,t\,d_h$.

$$
F_\text{cached}(t) = 6\,d\,d_h + 4\,t\,d_h .
$$

This is $O(t)$ per step: linear instead of quadratic. The ratio $F_\text{naive}/F_\text{cached}$ is roughly $t$ for large $t$, which is exactly "we stopped redoing $t-1$ tokens".

The cache does not change the *result*: the entries of $K$ and $V$ for earlier tokens are bit for bit the same numbers either way. That is why the test of a correct cache is exact equality of the outputs.

## Memory formula

Per layer, per token, per KV head, the cache stores one key vector and one value vector of length $d_h$:

$$
\text{bytes} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot T \cdot b \cdot B
$$

where $L$ is the number of layers, $n_{kv}$ the number of key/value heads, $T$ the sequence length, $b$ bytes per number (2 for fp16/bf16, 1 for int8) and $B$ the batch size. The factor 2 is "K and V".

## Worked example

All numbers below are reproduced by the exercise code.

**A 7B-class model** ($L=32$, $n_{kv}=32$, $d_h=128$, fp16, $T=4096$, $B=1$):

$2\cdot 32\cdot 32\cdot 128\cdot 4096\cdot 2 = 2{,}147{,}483{,}648$ bytes $= 2$ GiB for **one** conversation. Sixteen concurrent users need 32 GiB of cache alone, before counting the 14 GB of weights.

**Grouped-query attention (GQA).** In multi-head attention every query head has its own K and V. GQA keeps the full number of query heads but lets $g$ of them share one K/V head, so $n_{kv} = n_\text{heads}/g$. With $n_{kv}=8$ the same model needs $0.5$ GiB, four times less. Multi-query attention is the extreme $n_{kv}=1$. The book material reports that GQA models perform comparably to full multi-head attention in published ablations, so it is mostly a free memory saving, although it must be decided at training time.

A larger case: $L=80$, $n_{kv}=8$, $d_h=128$, $T=32768$, fp16 gives $10.0$ GiB per sequence; with 64 K/V heads it would be $80$ GiB.

**FLOPs for a toy model** ($d=16$, $d_h=8$, prompt of 5 tokens, 20 new tokens): the naive method uses $305{,}280$ FLOPs in total, the cached one $24{,}640$: a $12.4\times$ saving, and the outputs agree to $2\times10^{-16}$. The gap widens with length. For a realistic head ($d=4096$, $d_h=128$) at a single step:

| tokens present $t$ | naive FLOPs | cached FLOPs | ratio |
|---|---|---|---|
| 5 | 15,736,320 | 3,148,288 | 5.0 |
| 100 | 317,158,400 | 3,196,928 | 99.2 |
| 1000 | 3,401,984,000 | 3,657,728 | 930.1 |

## Implementation

```python
def generate_cached(X0, steps, W):
    d_head = W["Wq"].shape[1]
    K, V = X0 @ W["Wk"], X0 @ W["Wv"]          # prefill: all prompt tokens at once
    out = softmax((X0[-1] @ W["Wq"]) @ K.T / np.sqrt(d_head)) @ V
    rows = list(X0)
    for _ in range(steps):                      # decode: one token per step
        nxt = np.tanh(out @ W["Wo"])
        rows.append(nxt)
        K = np.vstack([K, nxt @ W["Wk"]])       # append, never recompute
        V = np.vstack([V, nxt @ W["Wv"]])
        out = softmax((nxt @ W["Wq"]) @ K.T / np.sqrt(d_head)) @ V
    return np.array(rows)
```

In real implementations the cache is a preallocated tensor and tokens are written in place; `np.vstack` copies and is only for clarity. The book's `kv_cache/generate.py` has the same two-branch shape: with a cache it first runs the prompt through the model with the cache attached, then feeds only the newest token each step.

## Shapes

| Object | Shape |
|---|---|
| prompt embeddings $X_0$ | $(T_0, d)$ |
| $W_q, W_k, W_v$ | $(d, d_h)$ |
| cache $K$, $V$ after $t$ tokens | $(t, d_h)$ per head, per layer |
| query of the new token | $(d_h,)$ |
| scores | $(t,)$ (cached) instead of $(t,t)$ (naive) |
| full cache tensor (all layers, heads, batch) | $(L, 2, B, n_{kv}, T, d_h)$ |

## Limits and trade-offs

- **Memory, not FLOPs, dominates.** Cached decode does little arithmetic per byte moved (low ops:byte ratio), so the GPU waits for memory. Making the cache smaller (GQA, quantised cache, sliding windows, paged allocation) speeds decoding as well as saving space. The sources estimate that reading all weights of a 70B fp16 model once takes about 70 ms on a 2 TB/s GPU, which is a floor for one decode step regardless of cache tricks.
- **Not for training.** In training the whole sequence is known and processed in parallel, so there is nothing to reuse across steps.
- **Quantising the cache** (int8/fp8) halves it again but errors in keys and values affect every later token, so quality loss grows with context length.
- **Related ideas, not derived here.** FlashAttention tiles the attention computation so the $t\times t$ score matrix never lives in slow memory; continuous batching and paged KV memory raise utilisation. They change speed and memory, never the mathematical result.
- Exact numbers for specific production models change between versions; use the formula with the model's own configuration.

## Common misconceptions

1. "The cache makes generation approximate." No: the stored $K$, $V$ are exactly the values a full recomputation would give. Only changes such as quantising the cache introduce error.
2. "Prefill and decode cost the same per token." Prefill is parallel and compute-bound, decode is sequential and bandwidth-bound; time-to-first-token and tokens-per-second are different metrics.
3. "GQA reduces the number of query heads." It reduces only the number of K/V heads; queries stay.
4. "We should also cache the queries." A query is used once, at its own step.

## Key takeaways

- Without a cache, a decode step costs $O(t^2)$ (sum over a generation: $O(T^3)$); with it, $O(t)$ per step.
- Cache size $=2\,L\,n_{kv}\,d_h\,T\,b\,B$ bytes; it grows linearly in context and in batch size.
- Prefill fills the cache in parallel; decode appends one entry per step and is memory-bound.
- GQA divides the cache by the group size $g=n_\text{heads}/n_{kv}$.
- A correct cache is verified by exact agreement with the uncached computation.

## Explain it yourself

1. Why can the key and value of token 3 be stored and reused when token 10 is generated, even though the model is rerun? What property of the causal mask guarantees this?
2. A friend says "just buy more compute, the cache is only an optimisation". Using the memory formula, explain why a long context and many users are limited by memory.

## Further reading

- Raschka, *Build a Large Language Model From Scratch*, bonus material `ch04/03_kv-cache` and `ch04/04_gqa`.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023), the paper the GQA README points to.
- Dao et al., FlashAttention (2022), for the memory-traffic side of attention.

## Mastery criteria

- [ ] I can derive $F_\text{naive}(t)$ and $F_\text{cached}(t)$ and say which is quadratic.
- [ ] I can compute the KV cache size for a configuration and see what GQA changes.
- [ ] I can explain prefill versus decode and why decode is memory-bound.
- [ ] I can test a cache implementation for exact equality with the naive one.
