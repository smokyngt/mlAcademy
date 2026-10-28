---
id: trf.03
title: Positional information, sinusoidal encodings and RoPE
summary: Why attention cannot tell word order by itself, how sinusoidal, learned and rotary position encodings fix it, and why RoPE makes the attention score depend only on the relative distance, derived with 2-D rotations and verified numerically.
minutes: 75
teaches: [trf.position, trf.rope]
requires: [trf.attention, la.matrix, la.dot]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/04-positional-encoding/docs/en.md (permutation invariance, sinusoidal formula, RoPE and the relative-distance property, ALiBi, comparison table). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/05-full-transformer/docs/en.md (where the positional signal enters the model; RoPE vs sinusoidal in the 2026 block table). READ."
  - "LLMs-from-scratch/ch04/01_main-chapter-code/ch04.ipynb (Sec. 4.1: GPT-2 uses learned token and positional embedding tables tok_emb and pos_emb). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/docs/en.md (transformers assume no locality, unlike RNNs). READ."
---

## Learning objectives

- Prove that attention without position information is permutation-equivariant, so it sees a bag of words.
- Write the sinusoidal encoding, state its shape, and show that a shift by $k$ positions is a fixed rotation of each sine/cosine pair.
- Compare learned absolute positions with sinusoidal ones.
- Derive rotary position embeddings (RoPE) as 2-D rotations and prove that $\langle R_mq,R_nk\rangle$ depends only on $m-n$.
- Implement sinusoidal encodings and RoPE and test the relative-shift property numerically.

## Where this fits

A recurrent network knows the order of words for free: it reads them in order. A transformer reads all tokens at once, so unless we do something, "dog bites man" and "man bites dog" are the *same set of tokens* and give the same set of outputs. For language, code and audio, order carries meaning, so we must inject **where each token is**.

The choice has real consequences. It decides whether a model trained on 2 000-token documents can read 20 000 tokens, whether attention scores can express "the word two to my left", and how large the first layer is. This lesson covers the three main answers: fixed sinusoids (the 2017 original), learned tables (GPT-2) and rotary embeddings (RoPE, used by most recent open models according to the source lesson).

## The idea in plain words

**The problem.** Picture a classroom where students raise their hands to answer, and the teacher collects the answers without noting who sits where. All she knows is the set of answers. Attention treats tokens the same way: it measures pairwise similarity between content vectors and averages; seat numbers never enter.

**Fix 1: stick a seat number on every student (absolute position).** Add to each token's embedding a vector that says "I am token number 7". Either designed in advance from sine and cosine waves of many different wavelengths (like a set of clock hands: the fast hand tells the exact minute, the slow hand tells roughly the hour) or learned from data, like a lookup table with one row per seat.

**Fix 2: measure the angle between students (relative position, RoPE).** Instead of adding information to the token, *rotate* its query and key vectors by an angle that grows with position. Two vectors that are rotated by the same amount keep their mutual angle; two vectors rotated by different amounts have an angle that changed by exactly the difference. So when query and key meet in a dot product, only the difference of the rotations survives: the **distance** between the tokens. Absolute seat numbers cancel, and "two seats to my left" is something attention can express directly.

## Why attention is permutation-equivariant (proof)

Let $P$ be a permutation matrix ($T\times T$, one 1 per row and column, $P^\top P=I$) and let $X'=PX$ reorder the rows (tokens). Projections act per row, so $Q'=PQ$, $K'=PK$, $V'=PV$. Then the scores are

$$
S'=\frac{Q'K'^\top}{\sqrt{d_k}}=\frac{PQK^\top P^\top}{\sqrt{d_k}}=PSP^\top .
$$

Row-wise softmax commutes with simultaneous row and column permutation: $\operatorname{softmax}(PSP^\top)=P\operatorname{softmax}(S)P^\top$ (each row is a set of numbers, softmax does not care in which order they are listed). Hence

$$
Z'=A'V'=PAP^\top\,PV=PA\,(P^\top P)\,V=PAV=PZ .
$$

Shuffling the input rows shuffles the output rows the same way and changes nothing else. A pure attention layer cannot distinguish orderings. (With a causal mask this is no longer exactly true, since the mask itself breaks the symmetry: a decoder gets some order information from the mask alone, but not enough to know *how far* back a token is.)

## Sinusoidal encodings

For position $p\in\{0,\dots,T-1\}$ and an even model dimension $d$, define $\omega_i=10000^{-2i/d}$ for $i=0,\dots,d/2-1$ and

$$
PE_{p,2i}=\sin(p\,\omega_i),\qquad PE_{p,2i+1}=\cos(p\,\omega_i).
$$

The matrix $PE$ has shape $(T,d)$ and is **added** to the token embeddings: $X\leftarrow X+PE$, both $(T,d)$ (broadcast over the batch). Pair $i$ is a sinusoid of wavelength $2\pi/\omega_i$, from $2\pi$ tokens for $i=0$ up to $2\pi\cdot10000$ tokens for the last pair. Fast pairs resolve neighbouring positions, slow pairs encode coarse location.

**A shift is a linear map.** Fix a pair $i$ and write $\omega=\omega_i$. Using $\sin(a+b)=\sin a\cos b+\cos a\sin b$ and $\cos(a+b)=\cos a\cos b-\sin a\sin b$,

$$
\begin{pmatrix}\sin((p+k)\omega)\\\cos((p+k)\omega)\end{pmatrix}
=\begin{pmatrix}\cos k\omega&\sin k\omega\\-\sin k\omega&\cos k\omega\end{pmatrix}
\begin{pmatrix}\sin p\omega\\\cos p\omega\end{pmatrix}.
$$

The matrix depends only on the offset $k$, not on $p$. So a model *can* learn linear maps that read relative offsets from sinusoidal inputs. But it must do it through the projections $W_Q,W_K$ and the content mixes in; nothing forces it, and the sinusoid is added to the content rather than being a separate channel.

**Learned absolute positions.** GPT-2, as built in the book's chapter 4, uses an embedding table `pos_emb` of shape $(T_{\max},d)$ that is looked up by position index and added to the token embeddings (the same operation as sinusoids, but the table is trained). Cost: $T_{\max}d$ parameters (for $T_{\max}=1024$, $d=768$: $786\,432$). Weakness: rows beyond $T_{\max}$ do not exist, and rows never seen in training are untrained. The source lesson also notes that fixed sinusoids extrapolate poorly in practice.

## Derivation of RoPE

**Goal.** Find maps $f_q(x,m)$, $f_k(x,n)$ (position $m$ for the query, $n$ for the key) such that the score is a function of $x_q,x_k$ and $m-n$ only:

$$
\langle f_q(x_q,m),\,f_k(x_k,n)\rangle=g(x_q,x_k,m-n).
$$

**Start in two dimensions.** Let $R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}$ be rotation by $\phi$. It is orthogonal: $R(\phi)^\top=R(-\phi)$ and $R(a)R(b)=R(a+b)$. Define $f(x,m)=R(m\theta)x$, i.e. rotate by an angle proportional to the position, with a fixed frequency $\theta$. Then

$$
\langle R(m\theta)q,\,R(n\theta)k\rangle=q^\top R(m\theta)^\top R(n\theta)\,k=q^\top R(-m\theta)R(n\theta)\,k=q^\top R\big((n-m)\theta\big)\,k .
$$

Only $n-m$ appears. Expanding with $q=(q_1,q_2)$, $k=(k_1,k_2)$ and $\phi=(m-n)\theta$ gives the concrete form

$$
\langle R(m\theta)q,R(n\theta)k\rangle=\cos\phi\,(q\cdot k)+\sin\phi\,(q_1k_2-q_2k_1).
$$

(Equivalently, in complex notation with $q\leftrightarrow q_1+iq_2$, the score is $\operatorname{Re}(q\,\bar k\,e^{i\phi})$.) The attention score is the plain dot product, "turned" by the relative angle.

**Many dimensions.** For head dimension $d_h$ (even), group coordinates in consecutive pairs $(2i,2i+1)$, $i=0,\dots,d_h/2-1$, and rotate pair $i$ with its own frequency $\theta_i=\text{base}^{-2i/d_h}$ (base $=10000$ by default, as in the source lesson). The full operator is block diagonal,

$$
R_m=\operatorname{diag}\big(R(m\theta_0),\dots,R(m\theta_{d_h/2-1})\big)\in\mathbb{R}^{d_h\times d_h},
$$

still orthogonal, and $R_m^\top R_n=R_{n-m}$ because each block obeys the 2-D rule. Therefore

$$
\langle R_mq,R_nk\rangle=q^\top R_{n-m}k .
$$

Properties, all consequences of the above:

- **Relative**: the score depends on the offset, so shifting the whole sentence by $s$ positions changes no score.
- **Norm preserving**: $\|R_mq\|=\|q\|$ (rotations are isometries). Content magnitude is untouched; only direction encodes position.
- **No extra parameters** and nothing is added to the embedding: RoPE is applied to $Q$ and $K$ (not $V$) inside every attention layer, per head.
- Different pairs rotate at different speeds: fast pairs resolve small offsets, slow pairs large ones (same multi-scale idea as the sinusoids).

In implementation you never build the $d_h\times d_h$ matrix. For pair $i$ with $c=\cos(m\theta_i)$, $s=\sin(m\theta_i)$:

$$
x'_{2i}=x_{2i}c-x_{2i+1}s,\qquad x'_{2i+1}=x_{2i}s+x_{2i+1}c .
$$

## Shapes

| Object | Shape |
|---|---|
| positions | $(T,)$ integers |
| sinusoidal table $PE$ | $(T,d)$, added to embeddings $(B,T,d)$ |
| frequencies $\theta_i$ | $(d_h/2,)$ |
| angles $m\theta_i$ | $(T,d_h/2)$ |
| $Q$ or $K$ per head, before and after RoPE | $(B,h,T,d_h)$ (unchanged shape) |
| $\cos,\sin$ tables | $(T,d_h/2)$, broadcast over $(B,h)$ |

## Worked example (checked by running code)

**Sinusoidal table**, $T=4$, $d=4$ (so $\omega_0=1$ and $\omega_1=10000^{-1/2}=0.01$):

$$
PE=\begin{pmatrix}0&1&0&1\\0.8415&0.5403&0.0100&1.0000\\0.9093&-0.4161&0.0200&0.9998\\0.1411&-0.9900&0.0300&0.9996\end{pmatrix}.
$$

Columns 0-1 oscillate with period $2\pi$; columns 2-3 hardly move over four positions (slow clock hand). Row 1, column 0: $\sin1=0.8415$.

**RoPE in 2-D**, $q=(1,2)$, $k=(3,1)$, $\theta=0.5$ (so $q\cdot k=5$ and $q_1k_2-q_2k_1=1-6=-5$). The score $\langle R(m\theta)q,R(n\theta)k\rangle$ is:

| $(m,n)$ | $m-n$ | score |
|---|---|---|
| $(0,0)$ | 0 | $5.0000$ |
| $(5,5)$ | 0 | $5.0000$ |
| $(2,1)$ | 1 | $1.9908$ |
| $(7,6)$ | 1 | $1.9908$ |
| $(3,0)$ | 3 | $-4.6338$ |
| $(10,7)$ | 3 | $-4.6338$ |

Same offset, same score, regardless of absolute position. Check $m-n=1$ with the closed form: $\phi=0.5$, $\cos\phi=0.8776$, $\sin\phi=0.4794$, so $0.8776\cdot5+0.4794\cdot(-5)=1.9908$ ✓. For $m-n=3$, $\phi=1.5$: $0.0707\cdot5+0.9975\cdot(-5)=-4.6338$ ✓.

**Full head**, $d_h=8$ with random $q,k$ (seed 0): $\langle R_5q,R_2k\rangle=-1.5865$ and $\langle R_{105}q,R_{102}k\rangle=-1.5865$ (same offset 3), and $\|R_9q\|=\|q\|=1.8627$.

## Implementation

```python
import numpy as np

def sinusoidal(T, d, base=10000.0):
    p = np.arange(T)[:, None]                       # (T, 1)
    i = np.arange(d // 2)[None, :]                  # (1, d/2)
    ang = p / base ** (2 * i / d)                   # (T, d/2)
    pe = np.zeros((T, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe                                       # (T, d)

def rope(x, pos, base=10000.0):
    """x: (..., T, d_h) with d_h even; pos: (T,) integer positions."""
    d = x.shape[-1]
    th = base ** (-2 * np.arange(d // 2) / d)       # (d/2,)
    ang = pos[:, None] * th[None, :]                # (T, d/2)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    a, b = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out
```

## Limits and trade-offs

- Sinusoids and learned tables add position to the *content*; the model must disentangle them. RoPE keeps content and position separate in the score.
- RoPE encodes relative distance in the score but gives no direct absolute position; for tasks needing an absolute index the model has to infer it (for example from a causal mask).
- Long-context behaviour beyond the training length depends on the frequency base and on scaling tricks; the source lesson lists several methods (rescaling the base, interpolation, ALiBi's linear distance penalty $-m_h|i-j|$). We did not implement them here.
- The rotation costs $O(Td_h)$ per head, negligible compared with attention.

## Common misconceptions

- **"Positional encoding is added to $V$ too."** Sinusoids are added to the embeddings (so they reach $Q,K,V$), but RoPE rotates only $Q$ and $K$.
- **"RoPE encodes absolute positions."** It uses absolute indices to build rotations, but the score depends only on the difference.
- **"Sine and cosine are redundant."** The pair $(\sin,\cos)$ is a point on a circle; it makes the shift-by-$k$ a rotation (linear) and keeps the pair at constant norm.
- **"Rotation changes how large the vector is."** It does not: $\|R_mq\|=\|q\|$.

## Key takeaways

- Attention alone is permutation-equivariant: $Z(PX)=PZ(X)$.
- Sinusoidal: $PE_{p,2i}=\sin(p\omega_i)$, $PE_{p,2i+1}=\cos(p\omega_i)$, shape $(T,d)$, added to embeddings; a shift is a pair-wise rotation.
- Learned: a trainable $(T_{\max},d)$ table, no values beyond $T_{\max}$.
- RoPE: rotate each coordinate pair of $Q,K$ by $m\theta_i$; $\langle R_mq,R_nk\rangle=q^\top R_{n-m}k$, a function of the offset only.

## Explain it yourself

1. In your own words, prove that a pure attention layer cannot tell "dog bites man" from "man bites dog".
2. Why do the absolute positions $m$ and $n$ disappear from $\langle R_mq,R_nk\rangle$? Use $R(a)^\top R(b)=R(b-a)$.

## Further reading

- Source library phase 07, lesson 04 (sinusoidal, RoPE, ALiBi, scaling methods) and its references (Vaswani 2017 Sec. 3.5; Su et al. 2021, RoFormer).
- Raschka, *LLMs from scratch*, ch. 4, Sec. 4.1 for the learned positional embedding in GPT-2.

## Mastery criteria

You can prove permutation equivariance, write the sinusoidal formula, derive $\langle R_mq,R_nk\rangle=q^\top R_{n-m}k$, and your implementation passes the relative-shift and norm-preservation tests.
