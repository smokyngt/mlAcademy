---
id: trf.02
title: Multi-head attention and masking
summary: Why one attention pattern is not enough, how heads are subspaces obtained by reshaping, the exact split/merge shapes, the output projection, causal and padding masks, and the parameter count, with a NumPy implementation that is tested for future-token leakage.
minutes: 80
teaches: [trf.multihead, trf.mask]
requires: [trf.attention, la.matrix]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/03-multi-head-attention/docs/en.md (split/combine heads, W_o, MQA/GQA/MLA variants, typical d_head). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md (the causal mask as lower-triangular row-stochastic matrix, -inf before softmax, parallel training). READ."
  - "LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb (Sec. 3.5 causal attention: mask before vs after softmax; Sec. 3.6 MultiHeadAttention with view/transpose and out_proj). READ."
---

## Learning objectives

- Explain why several smaller attention heads can beat one large head at the same cost.
- Write the split and merge reshapes with the exact shape after each operation.
- Build a causal mask and a padding mask, and say why they go in before the softmax as $-\infty$.
- Count the parameters of a multi-head attention layer.
- Implement multi-head attention in NumPy and test that future tokens cannot change earlier outputs.

## Where this fits

Lesson 1 produced one weighted average per token: one $T\times T$ attention pattern. But a sentence has several kinds of relations at once: "who is the subject of this verb", "which noun does this pronoun refer to", "what was the previous word". One softmax distribution has to smear all of these together. **Multi-head attention** runs several attentions in parallel, each in its own smaller space, and lets a final matrix mix the results.

The second half of the lesson is about **what a token is allowed to see**. A language model is trained to predict the next token, so during training, token $t$ must not be able to look at token $t+1$ (that would be reading the answer). And sequences in a batch have different lengths, so padding tokens must be ignored. Both are handled by the same trick: masks.

This is the exact attention layer inside GPT-style models (decoder-only, lesson 4) and, without the causal mask, inside BERT-style models.

## The idea in plain words

**Heads.** Imagine a newsroom covering one event. One reporter tracks who did what, another tracks dates, a third tracks quotes. Each works from the same raw material, but with a different focus, and each writes a short note. An editor (the output projection) reads all notes and writes the final article. Eight narrow reporters usually capture more than one reporter with eight times the notebook space, because a single reporter can only follow one storyline per paragraph.

Concretely, the model's feature vector (say 768 numbers) is **cut into $h$ equal slices**; each head only looks at its own slice's projections, computes its own attention weights, and returns a short vector. The short vectors are glued back together to 768 numbers and passed through one learned mixing matrix. The total cost is the same as one big head, because the slices are smaller.

**Masks.** A mask is a stencil: a sheet of paper with holes. The score matrix has an entry for every (query, key) pair; the stencil blocks the pairs the model must not use. For a language model the forbidden pairs are the "future" ones (key position later than query position), which form the upper triangle of the matrix. For padding, the forbidden pairs are all those whose key is a dummy token.

## Definitions

Let $d=d_{\text{model}}$, $h$ the number of heads, and $d_h=d/h$ (we require $h\mid d$). Learned matrices are $W_Q,W_K,W_V,W_O\in\mathbb{R}^{d\times d}$. For input $X\in\mathbb{R}^{B\times T\times d}$:

1. Project: $Q=XW_Q$, $K=XW_K$, $V=XW_V$, each $(B,T,d)$.
2. **Split** each into $h$ heads: $(B,T,d)\to(B,T,h,d_h)\to(B,h,T,d_h)$.
3. Attend per head: $\text{scores}=QK^\top/\sqrt{d_h}$ in $(B,h,T,T)$, add the mask, softmax, multiply by $V$: $(B,h,T,d_h)$.
4. **Merge**: $(B,h,T,d_h)\to(B,T,h,d_h)\to(B,T,d)$.
5. Output projection: $\text{out}=\text{merged}\,W_O$, $(B,T,d)$.

Note the scale uses $\sqrt{d_h}$, not $\sqrt{d}$: the dot products now run over $d_h$ terms, so by the variance argument of lesson 1 that is the right factor.

### Why this is "heads as subspaces"

Write $W_Q=[W_Q^{(1)}\,|\,\dots\,|\,W_Q^{(h)}]$ with each block $d\times d_h$ (columns grouped). Then head $i$'s query is $XW_Q^{(i)}$: a learned projection of $X$ onto a $d_h$-dimensional subspace, and the same for keys and values. The reshape in step 2 is just this column grouping done by one big matrix multiplication. Head $i$'s score is $\sum_{m\in\text{slice }i}q_mk_m$, a partial dot product over its own coordinates, which can express a different similarity notion from the other heads.

The merge followed by $W_O$ is equivalent to summing per-head contributions: split $W_O$ into row blocks $W_O^{(i)}\in\mathbb{R}^{d_h\times d}$, then $\text{out}=\sum_i Z^{(i)}W_O^{(i)}$. Heads never interact *inside* attention; they interact only through $W_O$ (and later layers).

## Derivation: the masks

**Causal mask.** We want output $i$ to depend only on inputs $j\le i$. Output $i$ is $\sum_ja_{ij}v_j$, so it suffices that $a_{ij}=0$ for $j>i$. Softmax never returns an exact zero for finite scores, but $e^{-\infty}=0$. So add a matrix $M$ to the scores before the softmax:

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}
\qquad
a_{ij}=\frac{e^{S_{ij}+M_{ij}}}{\sum_{j'}e^{S_{ij'}+M_{ij'}}}.
$$

The numerator is $0$ for forbidden keys, and the denominator sums only over allowed keys, so each row still sums to 1. Shape of $M$: $(T,T)$, broadcast over $(B,h)$.

**Why before the softmax and not after.** If you zero the weights *after* softmax, the row no longer sums to 1 and you must renormalise; masking scores before softmax gives the same final result in one step and is numerically cleaner (the book's chapter 3 shows both and notes they coincide after renormalisation).

**Padding mask.** Batch sequences are padded to a common length $T$. Let $p\in\{0,1\}^{B\times T}$ mark padding keys. The key-padding mask is $M^{\text{pad}}_{b,i,j}=-\infty$ if $p_{b,j}=1$, i.e. shape $(B,1,1,T)$ broadcast over heads and queries. Padding queries still produce (ignored) outputs, so the loss must also skip those positions. The two masks combine by taking the logical OR of the "forbidden" flags (or adding the $-\infty$ matrices).

**A dangerous corner.** If *every* key of a row is masked, the row is $\operatorname{softmax}(-\infty,\dots,-\infty)=0/0=\mathrm{NaN}$. With a causal mask alone this cannot happen (the diagonal is allowed); with left-padding it can for padding queries. Implementations either keep the diagonal allowed or replace NaN rows by zeros.

## Shapes

| Step | Shape | Note |
|---|---|---|
| $X$ | $(B,T,d)$ | |
| $Q,K,V$ after projection | $(B,T,d)$ | $W_{Q,K,V}:(d,d)$ |
| after `reshape` | $(B,T,h,d_h)$ | consecutive $d_h$ features form one head |
| after `transpose(0,2,1,3)` | $(B,h,T,d_h)$ | heads become a batch-like axis |
| $K^\top$ (last two axes swapped) | $(B,h,d_h,T)$ | |
| scores / weights | $(B,h,T,T)$ | mask $(T,T)$ or $(B,1,1,T)$ broadcasts |
| weights @ V | $(B,h,T,d_h)$ | |
| `transpose(0,2,1,3)` | $(B,T,h,d_h)$ | |
| `reshape` merge | $(B,T,d)$ | inverse of the split |
| after $W_O$ | $(B,T,d)$ | $W_O:(d,d)$ |

The only subtle point is the order **reshape first, then transpose**. Reshaping `(B,T,d)` directly to `(B,h,T,d_h)` would mix tokens into heads. Tiny check from the code: for one sequence with $T=3$, $d=4$, $h=2$ and $X=\begin{pmatrix}0&1&2&3\\4&5&6&7\\8&9&10&11\end{pmatrix}$, head 0 gets the columns $\{0,1\}$: rows $(0,1),(4,5),(8,9)$ and head 1 gets the columns $\{2,3\}$: rows $(2,3),(6,7),(10,11)$. Merging gives back exactly $X$.

## Parameter count

Each of the four matrices has $d^2$ weights:

$$
N_{\text{attn}}=4d^2\quad(\text{+ }4d\text{ if the projections have biases}).
$$

For $d=768$, $h=12$ ($d_h=64$): $4\cdot768^2=2\,359\,296$ parameters (2 362 368 with biases). **The count does not depend on $h$**: splitting into heads re-uses the same $d\times d$ matrices. The attention matrices are *not* parameters, they are computed activations: $B\cdot h\cdot T^2$ numbers, which is where memory grows with the context length.

Variants (from the source lesson): multi-query attention shares one K/V head across all query heads, and grouped-query attention uses $G<h$ K/V heads. Both shrink $W_K,W_V$ to $d\times (G\,d_h)$ and, more importantly, shrink the key/value cache used when generating text, at slightly lower quality. In this lesson we implement plain multi-head attention.

## Worked example (checked by running code)

Take the three-token example of lesson 1 (same $Q,K,V$, one head) and apply the causal mask. The scaled scores were

$$
S=\begin{pmatrix}0.7071&1.4142&2.1213\\0&0.7071&0.7071\\0.7071&2.1213&2.8284\end{pmatrix}.
$$

Setting the entries above the diagonal to $-\infty$ gives the weights

$$
A_{\text{causal}}=\begin{pmatrix}1&0&0\\0.3302&0.6698&0\\0.0743&0.3057&0.6200\end{pmatrix}.
$$

Row 1 can only see itself: weight 1, so its output is exactly $v_1=(1,0)$. Row 2 has scores $(0,0.7071)$: $e^0=1$, $e^{0.7071}=2.028$, so weights $1/3.028=0.3302$ and $0.6698$. Row 3 is unchanged from the unmasked case: it is the last token, so the keys it is allowed to see (all three) are the same as in lesson 1. The outputs are

$$
Z_{\text{causal}}=\begin{pmatrix}1&0\\0.3302&1.3395\\0.6943&1.8514\end{pmatrix}.
$$

Row 3 equals row 3 of the unmasked output of lesson 1, which is exactly what we want: *the last token has no future, so masking does not change it*.

**Padding.** Suppose a fourth key is padding with score $5.0$ for the first query. Masking it gives the weights $(0.1400,\,0.2840,\,0.5760,\,0)$, identical to the unpadded sequence. Without the mask the padding key would soak up most of the probability mass because its score is the largest. (Running the softmax on a row of three $-\infty$ values indeed returns NaN, as claimed above.)

## Implementation

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def split_heads(x, h):                      # (B, T, d) -> (B, h, T, d_h)
    B, T, d = x.shape
    return x.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)

def merge_heads(x):                         # (B, h, T, d_h) -> (B, T, d)
    B, h, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, h * dh)

def mha(X, Wq, Wk, Wv, Wo, h, causal=True, pad=None):
    """X: (B,T,d). pad: (B,T) bool, True = padding key. Returns (out, weights)."""
    B, T, d = X.shape
    Q, K, V = (split_heads(X @ W, h) for W in (Wq, Wk, Wv))   # (B,h,T,d_h)
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d // h)          # (B,h,T,T)
    forbid = np.zeros((B, 1, T, T), dtype=bool)
    if causal:
        forbid |= np.triu(np.ones((T, T), dtype=bool), k=1)
    if pad is not None:
        forbid |= pad[:, None, None, :]
    A = softmax(np.where(forbid, -np.inf, S))                  # (B,h,T,T)
    return merge_heads(A @ V) @ Wo, A                           # (B,T,d)
```

## Common misconceptions

- **"More heads means more parameters."** The count is $4d^2$ for any $h$ that divides $d$. More heads means smaller heads.
- **"Heads automatically learn different things."** They are free to, and probing studies show specialised heads, but nothing forces diversity; many heads can be pruned with little loss.
- **"The causal mask is applied to the output."** It is applied to the scores before softmax; applying it to the output would not stop information flowing through $V$.
- **"Reshape `(B,T,d)` straight to `(B,h,T,d_h)`."** That silently scrambles tokens and features. Reshape to `(B,T,h,d_h)`, then transpose.
- **"Masking makes the model faster."** The full $T\times T$ matrix is still computed in the simple implementation; efficient kernels skip the masked half.

## Key takeaways

- Heads are $h$ parallel attentions in $d_h=d/h$-dimensional subspaces; split by reshape+transpose, merge by the inverse, then mix with $W_O$.
- Scores use $\sqrt{d_h}$; shapes are $(B,h,T,T)$ for weights and $(B,T,d)$ for the layer output.
- Causal mask: $-\infty$ above the diagonal added before the softmax; padding mask: $-\infty$ on padded keys.
- Parameters: $4d^2$ (independent of $h$); attention matrices are activations costing $B\,h\,T^2$.

## Explain it yourself

1. Starting from `(B, T, d)`, write the sequence of reshapes and transposes of one multi-head attention call, with the shape after each, and say what would go wrong if you swapped the first reshape and transpose.
2. Why does the output of token $i$ not change when you edit token $i+1$ under a causal mask? Trace it through the softmax and the weighted sum.

## Further reading

- Raschka, *Build a Large Language Model from Scratch*, ch. 3, Sec. 3.5-3.6 (causal attention, `MultiHeadAttention` with `view`/`transpose`).
- Source library phase 07, lesson 03 (head variants MQA, GQA, MLA) and lesson 07 (the triangle mask as a refined prefix average).
- Vaswani et al., 2017, Sec. 3.2.2 (cited by the source lesson).

## Mastery criteria

You can write the split and merge functions with shapes from memory, build a causal and a padding mask, compute $4d^2$ for a given $d$, and your implementation passes a test showing that changing tokens after position $t$ leaves outputs at positions $\le t$ unchanged.
