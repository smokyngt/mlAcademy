---
id: trf.04
title: The transformer block and architectures
summary: Residual connections, LayerNorm vs RMSNorm and pre-norm, the feed-forward layer and its gated SwiGLU variant, stacking blocks, and the three families (encoder-only, decoder-only, encoder-decoder), assembled into a decoder-only forward pass with every shape and a parameter count.
minutes: 90
teaches: [trf.block, trf.architectures]
requires: [trf.attention, trf.multihead, trf.position, nn.layer]
sources:
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/05-full-transformer/docs/en.md (six pieces of the block, pre-norm vs post-norm, 2017 vs 2026 block table, parameter count, encoder/decoder/encoder-decoder choice). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/06-bert-masked-language-modeling/docs/en.md (masked language modelling, 15% / 80-10-10 rule, bidirectional encoder, use cases). READ."
  - "ai-engineering-from-scratch/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md (next-token prediction, causal mask, parallel training vs serial generation). READ."
  - "LLMs-from-scratch/ch04/01_main-chapter-code/ch04.ipynb (Sec. 4.1 GPT-2 124M config, 4.2 LayerNorm with scale and shift, 4.3 GELU feed-forward with 4x expansion, 4.4 shortcut connections and gradients, 4.5 TransformerBlock, 4.6 GPTModel, parameter counts 163,009,536 and 124,412,160 with weight tying). READ."
---

## Learning objectives

- Draw one pre-norm transformer block and write its two update equations.
- Explain what residual connections and normalisation each contribute, and compute LayerNorm and RMSNorm on a vector.
- Write the standard feed-forward layer and the SwiGLU variant with their shapes, and explain the $8d/3$ hidden size.
- Compare encoder-only, decoder-only and encoder-decoder models by mask, training objective and use.
- Assemble a complete decoder-only forward pass, count its parameters in closed form, and check every output shape.

## Where this fits

Lessons 1 to 3 gave three ingredients: attention (mix information across positions), multi-head and masks (several views, with control over who sees whom), and positions (so that order matters). One attention layer is still a shallow feature extractor. Language needs depth, and deep stacks are hard to train without the right plumbing.

This lesson is the plumbing. It explains the repeating unit, the **transformer block**, and how stacking blocks and choosing masks gives three architecture families: BERT-like encoders (embeddings, search, classification), GPT-like decoders (chat, code, text generation) and encoder-decoder models (translation, speech recognition). After this lesson you can read the architecture section of almost any language-model paper.

## The idea in plain words

Think of a document passing through an editorial office with $L$ desks in a row. At each desk, two people work on the *same copy* and write notes in the margin; they never erase what is there.

1. The **attention** clerk reads the whole document and adds margin notes of the form "this word relates to that earlier word".
2. The **feed-forward** clerk rereads each word on its own, thinks harder about it, and adds a margin note.

The margin-note habit is the **residual connection**: each sublayer *adds* to a running vector (the "residual stream") rather than replacing it. A **normalisation** step, applied before each clerk reads, keeps the handwriting at a readable size so that notes from 40 desks do not blow up or fade out.

Stack $L$ desks and you have a transformer. The only decisions left are who may read what (the mask) and what the model is asked to predict (the objective), which give the three families.

## The block, formally

Let $x\in\mathbb{R}^{B\times T\times d}$ be the residual stream. A **pre-norm** block computes

$$
x' = x+\operatorname{Attn}\big(\operatorname{Norm}_1(x)\big),\qquad
x'' = x'+\operatorname{FFN}\big(\operatorname{Norm}_2(x')\big).
$$

Both sublayers map $(B,T,d)\to(B,T,d)$, so the shapes allow the sum and any number of repetitions. The model is

$$
\text{tokens}\;(B,T)\;\xrightarrow{\text{embed}}\;(B,T,d)\;\xrightarrow{L\text{ blocks}}\;(B,T,d)\;\xrightarrow{\text{Norm}}\;(B,T,d)\;\xrightarrow{\text{head}}\;(B,T,V)
$$

where $V$ is the vocabulary size and the last tensor holds one logit vector per position.

### Residual connections

With $x_{l+1}=x_l+F_l(x_l)$, the Jacobian is $\partial x_{l+1}/\partial x_l=I+\partial F_l/\partial x_l$. Unrolling $L$ blocks, the derivative of the final stream with respect to the first contains a term equal to the identity, a path along which gradients travel without being multiplied by any weight matrix. The book's chapter 4 shows this on a 5-layer toy network: without shortcuts the gradients of the first layers are much smaller than those of the last ones; with shortcuts they stay comparable. The source lesson makes the same point and says that without residuals gradients vanish beyond a handful of layers (it states about 6; treat that as a rough figure).

### Normalisation: LayerNorm and RMSNorm

For one token's vector $x\in\mathbb{R}^d$ (normalisation acts independently on each of the $B\cdot T$ rows, along the last axis):

$$
\operatorname{LayerNorm}(x)=g\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+b,\quad
\mu=\tfrac1d\textstyle\sum_ix_i,\ \sigma^2=\tfrac1d\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\tfrac1d\sum_ix_i^2+\varepsilon}}.
$$

$g,b\in\mathbb{R}^d$ are learned (initialised to 1 and 0). RMSNorm drops the mean subtraction and the shift $b$: slightly cheaper, with $d$ instead of $2d$ parameters per norm, and, according to the source lesson, at least as stable; it is the modern default there. Both are *per token* statistics, so the result does not depend on the batch or on other positions (unlike batch normalisation).

**Pre-norm vs post-norm.** The original 2017 layout was $\operatorname{Norm}(x+\operatorname{sublayer}(x))$ (post-norm); the layout above normalises the *input* of each sublayer and leaves the residual path completely untouched. The source lesson reports that post-norm is harder to train deeply and that pre-norm is the 2026 default. A reason you can verify yourself: in pre-norm the identity path from input to output has no norm on it, so the gradient identity term above survives intact. One consequence is that the stream's magnitude grows with depth, hence the extra final norm before the head.

### The feed-forward network

A position-wise MLP applied identically to each token (no mixing between positions):

$$
\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2,\qquad W_1:(d,4d),\ W_2:(4d,d),
$$

with $\phi=$ GELU in GPT-2 (the book's chapter 4 uses the tanh approximation of $x\Phi(x)$ and an expansion factor 4). Shapes: $(B,T,d)\to(B,T,4d)\to(B,T,d)$. Roughly, attention moves information *between* tokens and the FFN processes it *within* each token; the FFN holds about two thirds of a block's parameters.

**SwiGLU (gated).** Modern blocks use a gate:

$$
\operatorname{SwiGLU}(x)=\big(\operatorname{SiLU}(xW_1)\odot xW_3\big)W_2,\quad
\operatorname{SiLU}(z)=z\,\sigma(z),
$$

with $W_1,W_3:(d,f)$ and $W_2:(f,d)$. The branch $xW_3$ carries content and $\operatorname{SiLU}(xW_1)$ is a soft, learned gate that scales each hidden unit. Three matrices instead of two, so to keep the parameter budget equal to the $4d$ design we need $3df=2\cdot d\cdot4d$, i.e.

$$
f=\tfrac{8}{3}d\approx2.67\,d .
$$

(The source lesson quotes 2.6; real models round $f$ to a convenient multiple. In our test model $d=64$ gives $f=170.7$ and we use $f=172$.) The source lesson reports a small perplexity gain for gated variants in published models; I did not verify this independently.

### Modern vs classic block

| Piece | Vaswani 2017 | Source lesson's "2026" block |
|---|---|---|
| Norm placement | post-norm | pre-norm |
| Norm | LayerNorm | RMSNorm |
| FFN | ReLU, 4x | SwiGLU, about 2.6x |
| Position | sinusoidal added | RoPE on $Q,K$ |
| Attention | full MHA | GQA or MLA |
| Biases | yes | no |

## Architecture families

All three use the same block; they differ in masks, inputs and training objective.

**Encoder-only (BERT).** No causal mask: every token sees every other token (bidirectional). Trained by **masked language modelling**: hide about 15% of tokens and predict them. Of the selected tokens, 80% are replaced by `[MASK]`, 10% by a random token, 10% left unchanged (so the model cannot rely on `[MASK]` appearing, which never happens at inference). Output: one contextual vector per token, $(B,T,d)$, which feeds a classifier, a tagger, or is pooled into an embedding for search. It cannot generate text token by token naturally. The source lesson says encoders remain the tool for classification and retrieval and are much faster per token than decoders.

**Decoder-only (GPT).** Causal mask; trained by **next-token prediction**: logits at position $t$ are scored against token $t+1$, with the softmax cross-entropy loss averaged over all $B\cdot T$ positions, so one forward pass trains $T$ predictions in parallel. Generation is serial: sample a token, append it, rerun (a key/value cache avoids recomputation). No separate encoder, no cross-attention. This family dominates text generation.

**Encoder-decoder (T5, Whisper-style).** An encoder reads the source bidirectionally, giving $(B,T_{src},d)$. The decoder has causal self-attention plus a **cross-attention** sublayer whose queries come from the decoder stream $(B,T_{tgt},d)$ and whose keys and values come from the encoder output $(B,T_{src},d)$. Scores have shape $(B,h,T_{tgt},T_{src})$, rectangular instead of square. It fits tasks with a clear source and target (translation, speech recognition, summarisation).

| | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| Self-attention mask | none (full) | causal | encoder full, decoder causal |
| Cross-attention | no | no | yes (decoder) |
| Objective | masked tokens | next token | next target token given source |
| Typical use | classify, embed, tag | generate | translate, transcribe |

## Shapes of a decoder-only model

Notation: vocabulary $V$, width $d$, layers $L$, heads $h$, FFN width $f$, batch $B$, length $T$.

| Stage | Shape | Learned parameters |
|---|---|---|
| token ids | $(B,T)$ | |
| embedding lookup | $(B,T,d)$ | $Vd$ |
| per block: Norm1 | $(B,T,d)$ | $d$ (RMSNorm) |
| Q, K, V projections, split | $(B,h,T,d/h)$ | $3d^2$ |
| scores, weights | $(B,h,T,T)$ | none |
| merge, $W_O$ | $(B,T,d)$ | $d^2$ |
| residual add | $(B,T,d)$ | |
| Norm2 | $(B,T,d)$ | $d$ |
| SwiGLU hidden | $(B,T,f)$ | $3df$ |
| back to stream | $(B,T,d)$ | |
| final norm | $(B,T,d)$ | $d$ |
| head | $(B,T,V)$ | $dV$ |

**Parameter count of the Llama-style model** (RMSNorm, SwiGLU, no biases, untied head):

$$
N=\underbrace{Vd}_{\text{embed}}+L\big(\underbrace{4d^2}_{\text{attn}}+\underbrace{3df}_{\text{FFN}}+\underbrace{2d}_{\text{norms}}\big)+\underbrace{d}_{\text{final norm}}+\underbrace{dV}_{\text{head}} .
$$

Example $V=1000$, $d=64$, $L=4$, $h=4$, $f=172$: one block has $4\cdot4096+3\cdot64\cdot172+128=16\,384+33\,024+128=49\,536$ parameters; the total is $64\,000+4\cdot49\,536+64+64\,000=326\,208$. The code below reproduces this count and the logits shape $(2,5,1000)$ for a batch of 2 sequences of 5 tokens.

**The book's GPT-2 124M model** (chapter 4: $V=50257$, context 1024, $d=768$, $L=12$, $h=12$, no qkv bias, learned positions, LayerNorm, GELU FFN with bias): token embeddings $38\,597\,376$, positions $786\,432$, each block $7\,085\,568$, final LayerNorm $1\,536$ and an untied head $38\,597\,376$ give $163\,009\,536$, matching the notebook's printed total. The original GPT-2 *ties* the output head to the token embedding, which removes the head's $38\,597\,376$ parameters: $124\,412\,160$, again the number printed in the notebook.

Two lessons from these counts. First, per block, attention takes $4d^2$ and the FFN $8d^2$: the FFN dominates. Second, for small models the embedding and head are a large fraction of everything (here $128\,000$ of $326\,208$).

## Implementation

```python
import numpy as np

def rms_norm(x, g, eps=1e-6):                       # (B,T,d) -> (B,T,d)
    return g * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

def swiglu(x, W1, W3, W2):                          # (B,T,d) -> (B,T,f) -> (B,T,d)
    a = x @ W1
    return ((a / (1 + np.exp(-a))) * (x @ W3)) @ W2

def block(x, p, h):                                 # pre-norm residual block
    x = x + attn_rope(rms_norm(x, p["n1"]), p, h)   # attention from lessons 2 and 3
    x = x + swiglu(rms_norm(x, p["n2"]), p["W1"], p["W3"], p["W2"])
    return x

def forward(params, tokens, h):                     # tokens: (B,T) ints
    x = params["tok"][tokens]                       # (B,T,d)
    for p in params["blocks"]:
        x = block(x, p, h)                          # (B,T,d)
    return rms_norm(x, params["nf"]) @ params["head"]   # (B,T,V)
```

The exercise provides `attn_rope` (causal multi-head attention with RoPE) and asks you to write the pieces above, plus the parameter counter and tests for leakage and for the residual identity.

## Worked example (checked by running code)

Normalising $x=(1,2,3,4)$ (with $\varepsilon=0$):

- LayerNorm: $\mu=2.5$, $\sigma^2=1.25$, output $(-1.3416,-0.4472,0.4472,1.3416)$: mean 0, variance 1.
- RMSNorm: $\operatorname{mean}(x^2)=7.5$, RMS $=2.7386$, output $(0.3651,0.7303,1.0954,1.4606)$: root-mean-square 1, but the mean is not 0 and the ordering and the sign pattern of $x$ are preserved.

The first does re-centre, the second only rescales, so RMSNorm output stays positive when the input is positive.

## Common misconceptions

- **"The FFN mixes tokens."** It acts on each position independently; all cross-token mixing is in attention.
- **"Residual connections are an optimisation detail."** They define the architecture: each layer is a small edit of a shared stream, not a new representation.
- **"LayerNorm normalises across the batch."** It normalises across the features of one token; the batch does not interact.
- **"Decoder-only models cannot understand text, only generate."** The same causal stack is trained on prediction and is used for comprehension tasks too; the cost is that each token sees only its past.
- **"Parameters scale with sequence length."** Only a learned position table does; RoPE and sinusoids add none. Attention *compute and memory* scale with $T^2$ regardless.
- **"BERT predicts the next word."** It predicts masked words from both sides.

## Key takeaways

- Block: $x\leftarrow x+\operatorname{Attn}(\operatorname{Norm}(x))$, then $x\leftarrow x+\operatorname{FFN}(\operatorname{Norm}(x))$; every tensor in the stream is $(B,T,d)$.
- Residuals give gradients an identity path; pre-norm leaves that path clean.
- RMSNorm = LayerNorm without mean subtraction and shift. SwiGLU = gated FFN with $f\approx8d/3$ to match the $4d$ parameter budget.
- Encoder-only: bidirectional + masked tokens. Decoder-only: causal + next token. Encoder-decoder: adds cross-attention $(B,h,T_{tgt},T_{src})$.
- Parameters: $4d^2+3df+2d$ per block plus embeddings; the count is independent of $T$ and $h$.

## Explain it yourself

1. Starting from token ids of shape $(B,T)$, narrate a decoder-only forward pass, naming the shape after every operation, until the logits.
2. Why does the SwiGLU hidden width come out near $\tfrac83d$ and not $4d$? And why is the FFN, not attention, the largest part of a block?

## Further reading

- Raschka, *LLMs from scratch*, ch. 4 (Sec. 4.2 to 4.6), including the weight-tying discussion.
- Source library phase 07 lessons 05 to 07 (full transformer, BERT, GPT) and the later lessons in that phase on encoder-decoder models and the key/value cache.

## Mastery criteria

You can draw the pre-norm block, write both norms and SwiGLU, state the three families with their masks and objectives, and produce a decoder-only forward pass whose logits have shape $(B,T,V)$, whose parameter count matches $N$ above, and which passes the no-future-leakage test.
