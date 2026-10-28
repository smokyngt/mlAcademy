---
id: trf.01
title: Self-attention from scratch
summary: Queries, keys and values, the scaled dot-product attention formula derived (including why we divide by the square root of d_k), tensor shapes at every step, and a NumPy implementation with a numeric variance test.
minutes: 80
teaches: [trf.attention]
requires: [la.matrix, la.dot, nn.layer, calc.chain]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/docs/en.md (why recurrence was the bottleneck, O(N^2) attention memory). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md (Q/K/V projections, attention matrix, softmax, formula, scaling motivation). READ."
  - "LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb (Sec. 3.3 simple attention with the 6-token 'Your journey starts with one step' example, Sec. 3.4 trainable Q/K/V with scaling by sqrt(d_k)). READ."
---

## Learning objectives

- Explain what queries, keys and values are, using the soft-lookup picture, and why one input produces three different vectors.
- Derive $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ and prove that the $1/\sqrt{d_k}$ factor keeps the score variance at 1.
- Give the shape of every tensor, with and without a batch dimension.
- State the time and memory cost of attention as a function of the sequence length $T$.
- Implement batched attention in NumPy and test the variance claim by simulation.

## Where this fits

Language, code, audio and DNA all arrive as **sequences**: an order of tokens where the meaning of one token depends on others, sometimes far away ("The trophy did not fit in the suitcase because **it** was too big": what is "it"?). Before 2017 the standard tool was a recurrent network, which reads tokens one at a time and squeezes everything seen so far into a single hidden vector. Two problems follow: the computation cannot be parallelised along the sequence (step $t+1$ needs step $t$), and information from token 1 has to survive many compressions to reach token 50.

**Self-attention** removes both problems with a different idea: let every token look directly at every other token and decide, by itself, how much each one matters. All those lookups can be written as a few matrix multiplications, which is exactly what GPUs are good at. This lesson builds that one operation. The next lessons extend it (several heads and masks, then position information, then the full block) and together they form the architecture behind every modern language model.

You will use what you already know: matrix products (la.matrix), dot products as similarity (la.dot), and the layer notation `X @ W` of lesson 9.

## The idea in plain words

Imagine a library with a very forgiving librarian. You walk in with a question written on a card (the **query**). Every book on the shelves has a label on its spine (the **key**), and inside it contains actual content (the **value**). A normal library does an exact match: your card matches one label, you get one book. The forgiving librarian instead compares your card with *every* label, gives each book a relevance score, and hands you a **blend** of all the books, with the most relevant ones contributing the most.

Self-attention is that, performed by every token at the same time, where the "books" are the other tokens of the same sentence:

- The **query** of token $i$ says "what kind of information am I looking for?".
- The **key** of token $j$ says "what kind of information do I hold?".
- The **value** of token $j$ says "this is what I will hand over if you pick me".

Why three different vectors from one embedding? Because *finding* a token and *using* a token are different jobs. The word "bank" may advertise (key) that it is a noun about places, while what it passes on (value) is a richer description. Three learned linear maps let the model decide independently what to look for, what to advertise and what to deliver.

Two more pictures:

- The relevance score is a **dot product**: large when the query and the key point in the same direction. It is a learned similarity.
- The scores are turned into **percentages** that sum to 100% (softmax), and the output is the percentage-weighted average of the values. A token's output is therefore a new vector that already contains information collected from the whole sentence.

## Definitions

Let a sequence of $T$ tokens be represented by an embedding matrix, one row per token:

$$
X\in\mathbb{R}^{T\times d_{\text{model}}}.
$$

Three learned weight matrices project it:

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V,
$$

with $W_Q,W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}$ and $W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}$, so $Q,K\in\mathbb{R}^{T\times d_k}$ and $V\in\mathbb{R}^{T\times d_v}$. Scaled dot-product attention is

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V,
$$

where softmax is applied **to each row** of the $T\times T$ score matrix.

## Derivation: building the formula in five small steps

**Step 1: a relevance score for one pair.** Token $i$ has query $q_i\in\mathbb{R}^{d_k}$ (row $i$ of $Q$) and token $j$ has key $k_j$. Their raw score is the dot product $s_{ij}=q_i\cdot k_j=\sum_{m=1}^{d_k}q_{im}k_{jm}$.

**Step 2: all pairs at once.** Entry $(i,j)$ of $QK^\top$ is exactly $q_i\cdot k_j$, because $(QK^\top)_{ij}=\sum_m Q_{im}K_{jm}$. Shapes: $(T\times d_k)(d_k\times T)=(T\times T)$. Row $i$ is "how much token $i$ cares about each token".

**Step 3: scaling (justified below).** $S=QK^\top/\sqrt{d_k}$, still $T\times T$.

**Step 4: normalise each row.** The weights of token $i$ are $a_{ij}=\dfrac{e^{S_{ij}}}{\sum_{j'}e^{S_{ij'}}}$. Then $a_{ij}\ge0$ and $\sum_j a_{ij}=1$. In matrix form $A=\operatorname{softmax}_{\text{rows}}(S)$, shape $(T\times T)$.

**Step 5: blend the values.** The output of token $i$ is $z_i=\sum_j a_{ij}v_j$. Stacking rows: $Z=AV$, shape $(T\times T)(T\times d_v)=(T\times d_v)$. Each row of $Z$ is a convex combination (a weighted average) of rows of $V$.

### Why divide by $\sqrt{d_k}$: the variance argument

Assume, as a model of a freshly initialised network, that the entries of $q$ and $k$ are independent, with mean 0 and variance 1. Consider one score $s=\sum_{m=1}^{d_k}q_mk_m$.

- Each product has mean $\mathbb{E}[q_mk_m]=\mathbb{E}[q_m]\,\mathbb{E}[k_m]=0$ (independence).
- Each product has variance $\operatorname{Var}(q_mk_m)=\mathbb{E}[q_m^2]\,\mathbb{E}[k_m^2]-0=1\cdot1=1$.
- The $d_k$ products are independent, and the variance of a sum of independent terms is the sum of the variances. Hence

$$
\operatorname{Var}(s)=d_k.
$$

The standard deviation of a raw score therefore grows like $\sqrt{d_k}$: with $d_k=64$ typical scores are about $\pm8$, with $d_k=512$ about $\pm23$. Dividing by $\sqrt{d_k}$ uses $\operatorname{Var}(cs)=c^2\operatorname{Var}(s)$ with $c=1/\sqrt{d_k}$:

$$
\operatorname{Var}\!\Big(\frac{s}{\sqrt{d_k}}\Big)=\frac{d_k}{d_k}=1.
$$

**Why large scores are a problem.** Softmax of widely spread numbers is almost one-hot: one weight is $\approx1$, the rest $\approx0$. The derivative of softmax $p=\operatorname{softmax}(s)$ is $\partial p_i/\partial s_j=p_i(\delta_{ij}-p_j)$. If $p$ is one-hot, then $p_i\in\{0,1\}$ and every entry of that Jacobian is $0$: the gradient vanishes, and the layer stops learning exactly where it has made a (maybe wrong) hard choice. Keeping the variance at 1 keeps softmax in its sensitive region at initialisation.

### Cost

The score matrix has $T^2$ entries, each costing $d_k$ multiply-adds: $O(T^2d_k)$ time. Storing it costs $O(T^2)$ memory per head. The product $AV$ is another $O(T^2d_v)$. Doubling the context length therefore **quadruples** the attention cost. This is the price of letting every token see every other token, and the reason long-context research (sliding windows, memory-efficient kernels) exists. In exchange the *depth* of the computation is constant in $T$: all tokens are processed in parallel, unlike the $T$ sequential steps of a recurrent network.

## Shapes

| Tensor | Single sequence | With batch $B$ |
|---|---|---|
| input $X$ | $(T,d_{\text{model}})$ | $(B,T,d_{\text{model}})$ |
| $W_Q,W_K$ | $(d_{\text{model}},d_k)$ | same (shared by the batch) |
| $W_V$ | $(d_{\text{model}},d_v)$ | same |
| $Q,K$ | $(T,d_k)$ | $(B,T,d_k)$ |
| $V$ | $(T,d_v)$ | $(B,T,d_v)$ |
| scores $S=QK^\top/\sqrt{d_k}$ | $(T,T)$ | $(B,T,T)$ |
| weights $A$ | $(T,T)$, rows sum to 1 | $(B,T,T)$ |
| output $Z=AV$ | $(T,d_v)$ | $(B,T,d_v)$ |

In NumPy the batched transpose of the last two axes is `K.transpose(0, 2, 1)` (or `np.swapaxes(K, -1, -2)`), and `@` broadcasts over the batch axis.

## Worked example (checked by running code)

Three tokens, $d_{\text{model}}=d_k=d_v=2$:

$$
X=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad
W_Q=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad
W_K=\begin{pmatrix}1&0\\1&1\end{pmatrix},\quad
W_V=\begin{pmatrix}1&0\\0&2\end{pmatrix}.
$$

Projections (each $3\times2$):

$$
Q=\begin{pmatrix}1&1\\0&1\\1&2\end{pmatrix},\quad
K=\begin{pmatrix}1&0\\1&1\\2&1\end{pmatrix},\quad
V=\begin{pmatrix}1&0\\0&2\\1&2\end{pmatrix}.
$$

Raw scores $QK^\top$ (shape $3\times3$) and the scaled scores with $\sqrt{d_k}=\sqrt2\approx1.4142$:

$$
QK^\top=\begin{pmatrix}1&2&3\\0&1&1\\1&3&4\end{pmatrix},\qquad
S=\begin{pmatrix}0.7071&1.4142&2.1213\\0&0.7071&0.7071\\0.7071&2.1213&2.8284\end{pmatrix}.
$$

Row-wise softmax gives the attention weights (each row sums to 1):

$$
A=\begin{pmatrix}0.1400&0.2840&0.5760\\0.1978&0.4011&0.4011\\0.0743&0.3057&0.6200\end{pmatrix}.
$$

Check row 1 by hand: $e^{0.7071}=2.028$, $e^{1.4142}=4.113$, $e^{2.1213}=8.341$, sum $=14.482$, so the weights are $0.140,\,0.284,\,0.576$. Output $Z=AV$ (shape $3\times2$):

$$
Z=\begin{pmatrix}0.7160&1.7199\\0.5989&1.6044\\0.6943&1.8514\end{pmatrix}.
$$

For example, $z_1=0.14\,(1,0)+0.284\,(0,2)+0.576\,(1,2)=(0.716,\,1.720)$. Token 3 puts 62% of its attention on itself (largest key alignment); token 1 puts only 14% on itself and mostly listens to token 3. The exercise reproduces these numbers.

**Variance simulation.** Drawing 20 000 pairs of standard-normal vectors, the variance of the raw score is $4.0,\ 64.5,\ 508.2$ for $d_k=4,64,512$ (theory: $d_k$), and after dividing by $\sqrt{d_k}$ it is $1.00,\ 1.01,\ 0.99$. The effect on softmax with 8 random keys: the largest attention weight is $0.63,\ 1.00,\ 1.00$ without scaling versus $0.35,\ 0.48,\ 0.50$ with it (for $d_k=4,64,512$, single random draw each; the trend, not the exact digits, is the point).

## Implementation

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))   # subtract max: stability
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """Q,K: (..., T, d_k)   V: (..., T, d_v)   mask: (T, T) bool, True = forbidden."""
    d_k = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)      # (..., T, T)
    if mask is not None:
        S = np.where(mask, -np.inf, S)                  # forbidden pairs get weight 0
    A = softmax(S, axis=-1)                             # (..., T, T)
    return A @ V, A                                     # (..., T, d_v), weights
```

The max subtraction does not change the result (softmax is invariant to adding a constant to a row) but prevents `exp` overflow.

## Common misconceptions

- **"Attention weights are what the model 'pays attention to', full stop."** They are one intermediate quantity; the output also depends on $V$ and on everything after. Treat attention maps as a diagnostic, not a proof of explanation.
- **"Queries, keys and values are different data."** In self-attention all three are linear projections of the *same* $X$. (In cross-attention, lesson 4, queries come from one sequence and keys and values from another.)
- **"Softmax is over the columns / over the batch."** It is over the key axis (the last axis of the $T\times T$ matrix) so that each query's weights sum to 1.
- **"The scale factor is cosmetic."** It is what keeps gradients alive at initialisation when $d_k$ is 64 or 128.
- **"Attention is $O(T)$ because it is parallel."** Parallel in depth, but $O(T^2)$ in work and memory.

## Key takeaways

- $\operatorname{Attn}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V$: score every query-key pair, normalise per query, average the values.
- Shapes: $Q,K:(T,d_k)$, $V:(T,d_v)$, scores and weights $(T,T)$, output $(T,d_v)$.
- Raw dot products have variance $d_k$ for unit-variance inputs; dividing by $\sqrt{d_k}$ restores variance 1 and prevents saturated softmax with vanishing gradients.
- Cost is $O(T^2d_k)$ time and $O(T^2)$ memory, but constant sequential depth.

## Explain it yourself

1. Without formulas, explain the roles of query, key and value to a friend, using the library analogy but a different example of your own.
2. A colleague removes the $1/\sqrt{d_k}$ and sees training loss stuck for the first few hundred steps with $d_k=128$. Using the variance argument and the softmax Jacobian, explain a plausible cause.

## Further reading

- Raschka, *Build a Large Language Model from Scratch*, ch. 3 (Sec. 3.3 to 3.4): the six-token example and trainable projections.
- Vaswani et al., 2017, *Attention Is All You Need*, Sec. 3.2.1 (listed in the phase 07 lesson's references).
- The source library's lesson 02 on self-attention (Build It section) for an alternative class-based implementation.

## Mastery criteria

You can, from a blank page, write the attention formula, derive the variance of a raw score, give all shapes for a batched call, reproduce the worked example's weight matrix to three decimals, and implement `attention` so that the exercise tests pass.
