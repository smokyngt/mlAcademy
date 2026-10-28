---
id: llm.04
title: "Fine-tuning: supervised instruction tuning and LoRA"
summary: How a base model that only continues text becomes an instruction follower (data format, loss masking on the prompt), why updating every weight is expensive, and how LoRA trains a low-rank update W + (alpha/r)BA with a few percent or less of the parameters, with a NumPy layer and gradient check.
minutes: 80
teaches: [llm.sft, llm.lora]
requires: [llm.lm, nn.backprop, la.svd, la.matrix]
sources:
  - "LLMs-from-scratch/ch07/01_main-chapter-code/ch07.ipynb (Sec. on instruction data formatting: Alpaca-style prompt template, padding collate function, ignore_index -100, remark that masking instruction targets is common practice): READ (the formatting, collate and masking cells)."
  - "LLMs-from-scratch/appendix-E/01_main-chapter-code/appendix-E.ipynb (Sec. E.1 LoRA idea W+AB, E.4 LoRALayer with zero-initialised B, alpha/rank scaling, LinearWithLoRA, replacing Linear layers): READ."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/06-instruction-tuning-sft/docs/en.md (what SFT does, data formats Alpaca/ShareGPT/ChatML, masked loss, SFT hyperparameters, catastrophic forgetting): READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/08-fine-tuning-lora/docs/en.md and code/lora.py (LoRA parameter arithmetic, alpha, which layers, rank choice, QLoRA NF4/double quantisation/paged optimisers, merging, memory table): READ."
---

## Where this fits

Pretraining ([[llm.03]]) produces a **base model**: an excellent continuer of text and a poor assistant. Ask it "What is the capital of France?" and it may continue with more quiz questions, because lists of questions are a common pattern in its data. Turning it into something that *answers* is the job of **supervised fine-tuning** (SFT), also called instruction tuning. The same machinery adapts a general model to a domain, a company's tone or a classification task.

Two practical walls appear immediately:

1. **What to train on, and on which tokens?** The answer is demonstration pairs (instruction, good response), with the loss counted on the response only.
2. **What to update?** Updating every weight of a 7-billion-parameter model needs gradients and optimiser state for all of them (see the memory bill of the previous lesson) and produces a full copy of the model per task. **LoRA** instead freezes the model and learns a tiny correction.

This lesson derives both pieces and implements them. Alignment from preference data, which refines the SFT model further, is the next lesson ([[llm.05]]).

## The idea in plain words

**SFT.** Pretraining gave the model knowledge; SFT teaches it manners. It does not need to learn that Paris is the capital of France, it already knows. It needs to learn that when a message looks like an instruction, the next thing to write is an *answer*. That is a small behavioural change, which is why tens of thousands of good examples, not billions, are enough. One subtle point: in each training example the instruction is context, the response is what we want to produce. If we also scored the model on predicting the instruction, we would spend learning signal teaching it to *ask* questions. So the loss is switched off on the prompt tokens.

**LoRA.** Imagine the pretrained weight matrix is a huge finished painting and fine-tuning is a retouch. A full retouch repaints every pixel and needs a canvas as big as the original. But a retouch is usually simple: a few broad strokes, a colour shift. LoRA says: don't store the retouch pixel by pixel; store it as the product of two thin matrices, a "tall and thin" times a "short and wide", whose product can only change the picture in a few directions (its **rank** is small). You keep the original painting untouched and add the thin correction on top. After training you can fuse the correction into the painting, or keep it as a removable overlay and swap overlays for different tasks.

## SFT in detail

**Data formats.** An example is a conversation turned into one token sequence. Common layouts: Alpaca (`instruction`, optional `input`, `output`, rendered into a fixed template with headers), ShareGPT-style multi-turn lists of (role, text), and ChatML, which delimits roles with special tokens such as `<|im_start|>` and `<|im_end|>` ([[llm.01]]). The template used at fine-tuning time must be reproduced exactly at inference time.

**The sequence.** For a prompt $p=(p_1..p_m)$ and response $r=(r_1..r_n)$ build $s=p\,\|\,r\,\|\,\text{eos}$, length $m+n+1$. The inputs are $s_{1:-1}$ and the targets $s_{2:}$. Labels at prompt positions are replaced by a sentinel (`-100`, the value that PyTorch's cross-entropy skips; the book uses it for padding, too).

**Masked loss.** Let $M_t\in\{0,1\}$ be 1 where the target is a response or end-of-text token. The loss is

$$
\mathcal L=-\frac{1}{\sum_t M_t}\sum_t M_t\,\ln p_\theta(y_t\mid s_{\le t}).
$$

The gradient with respect to the logits at position $t$ is $M_t\,(p_t-\text{onehot}(y_t))/\sum M$, so masked positions send *no* signal at all, but the prompt tokens still influence the output through attention because they are in the context. In a batch, sequences are padded to the longest; padded targets are masked as well. The same mechanism handles both.

**Hyperparameters differ from pretraining.** The source lesson contrasts a peak learning rate of about $3\times10^{-4}$ in pretraining with about $2\times10^{-5}$ in SFT, 1 to 3 epochs instead of one pass over trillions of tokens, and a few tens of thousands of high-quality examples (it cites around 27,000 curated pairs for one chat model and 52,000 generated pairs for Alpaca). Quality dominates quantity.

**Catastrophic forgetting.** Big steps on a small dataset overwrite general capabilities. Mitigations: small learning rate, few epochs, mixing a few percent of pretraining-style data back in, or restricting what can change (LoRA below).

## LoRA, derived

### The parametrisation

Fine-tuning a layer with weight $W\in\mathbb R^{d_{out}\times d_{in}}$ learns an update $\Delta W$, so the adapted layer is $W+\Delta W$. LoRA restricts the update to the form

$$
\Delta W=\frac{\alpha}{r}\,B A,\qquad B\in\mathbb R^{d_{out}\times r},\quad A\in\mathbb R^{r\times d_{in}},\quad r\ll\min(d_{in},d_{out}),
$$

and trains only $A$ and $B$, with $W$ frozen. (The book writes the product as $AB$ with the matrices transposed, to match a convention in which inputs multiply from the left; it is the same construction.) The scalar $\alpha$ is a hyperparameter; the factor $s=\alpha/r$ keeps the size of the update comparable across ranks, which the book notes lets one compare runs of different rank without retuning the learning rate.

**Rank.** The rank of a product is at most the rank of each factor, so $\operatorname{rank}(\Delta W)\le r$. The bet (Hu et al.) is that the *useful* update during adaptation has low "intrinsic rank", so a small $r$ loses little. A rank-$r$ matrix is exactly what truncating an SVD to $r$ singular values gives; LoRA learns such a factorisation directly instead of computing a full update and truncating it.

**Parameter count.** A full update has $d_{out}d_{in}$ numbers; LoRA has $r(d_{out}+d_{in})$. For $d_{out}=d_{in}=4096$ and $r=16$: $16\cdot8192=131{,}072$ against $16{,}777{,}216$, i.e. $0.78\%$. Applied to the four attention projections of every layer of GPT-2 small ($d=768$, 12 layers, $r=8$) the adapter has $4\cdot12\cdot8\cdot(768+768)=589{,}824$ parameters, $0.47\%$ of the 124,439,808 weights.

### Forward pass without forming $\Delta W$

For a batch $x\in\mathbb R^{B\times d_{in}}$ (rows are examples):

$$
y=xW^\top+s\,(xA^\top)B^\top,\qquad h=xA^\top\in\mathbb R^{B\times r}.
$$

By the distributive law $x(W+sBA)^\top=xW^\top+s\,xA^\top B^\top$, so we never build the $d_{out}\times d_{in}$ update; the extra cost is two skinny matrix products through the width-$r$ bottleneck.

### Backward pass

Let $G=\partial\mathcal L/\partial y\in\mathbb R^{B\times d_{out}}$. Entrywise, $y_{bk}=\sum_j x_{bj}W_{kj}+s\sum_{i}h_{bi}B_{ki}$ and $h_{bi}=\sum_j x_{bj}A_{ij}$. Therefore:

$$
\frac{\partial\mathcal L}{\partial B_{ki}}=s\sum_b G_{bk}h_{bi}\;\Longrightarrow\;\bar B=s\,G^\top h\quad(d_{out}\times r),
$$

$$
\frac{\partial\mathcal L}{\partial h_{bi}}=s\sum_k G_{bk}B_{ki}\;\Longrightarrow\;\bar h=s\,GB\quad(B\times r),
$$

$$
\frac{\partial\mathcal L}{\partial A_{ij}}=\sum_b\bar h_{bi}x_{bj}\;\Longrightarrow\;\bar A=\bar h^\top x=s\,(GB)^\top x\quad(r\times d_{in}),
$$

$$
\bar x=GW+\bar hA=GW+s\,(GB)A\quad(B\times d_{in}).
$$

(The last line is what must be passed to earlier layers; the frozen $W$ needs no gradient of its own.)

### Initialisation: why $B=0$ and $A$ random

With $B=0$ the adapter contributes exactly zero, so the model starts as the pretrained one: $\Delta W=0$. The formulas above show why not to zero *both*: $\bar A\propto B$ and $\bar B\propto h=xA^\top$. If $A=0$ too, then $h=0$ and $\bar B=0$; if $B=0$ then $\bar A=0$. With $B=0$ and random $A$, $\bar A=0$ at step one but $\bar B\neq0$, so $B$ moves off zero and from then on both factors learn. Random $A$ with $B=0$ is the standard choice (Gaussian in the paper, a Kaiming-uniform in the book's `LoRALayer`).

### Merging and swapping

After training, $W_{\text{merged}}=W+\frac{\alpha}{r}BA$ gives a plain layer with no extra latency. Alternatively keep $A,B$ separate: one frozen base model in memory and a small file per task, switchable at run time.

### Where to apply it, which rank

The source lesson reports that adapting the query and value projections is a good default, all four attention projections slightly better, and adding the MLP matrices roughly doubles the parameters for small gains; ranks 4 to 16 are common, with diminishing returns beyond 64. Treat these as rules of thumb from that source, to be checked on your task. The book replaces every `Linear` layer in the model by a `LinearWithLoRA` after freezing the originals.

### QLoRA

QLoRA keeps the *frozen* base weights in 4-bit precision (the source lesson names the NF4 format, whose 16 levels are placed at quantiles of a normal distribution because weights are roughly normal), stores the quantisation constants compactly ("double quantisation"), and trains the LoRA matrices in higher precision; paged optimiser state avoids out-of-memory spikes. Gradients flow through the dequantised weights into $A$ and $B$. The memory table in the source lesson for a 7B model (approximate, as reported there): full fine-tune about 56 GB, LoRA on a 16-bit base about 18 GB, QLoRA about 6 GB. The saving comes from not storing gradients and optimiser state for the frozen weights, and from shrinking the weights themselves.

## Worked example

A single LoRA layer small enough for pencil. $d_{in}=3$, $d_{out}=2$, $r=1$, $\alpha=2$ so $s=2$. $A=(1,0,-1)$, $B=(0.5,-1)^\top$, input $x=(1,2,3)$.

- $h=xA^\top=1\cdot1+2\cdot0+3\cdot(-1)=-2$.
- Adapter output $s\,h\,B^\top=2\cdot(-2)\cdot(0.5,-1)=(-2,\,4)$.
- Equivalent merged update $\Delta W=sBA=\begin{pmatrix}1&0&-1\\-2&0&2\end{pmatrix}$, and $\Delta W x=(1-3,\,-2+6)=(-2,4)$. Same result, as the distributive law promises. Rank 1: the two rows are proportional.
- Upstream gradient $G=(1,-1)$: $\bar B=s\,G^\top h=2\cdot(-2)\cdot(1,-1)=(-4,4)$; $GB=1\cdot0.5+(-1)(-1)=1.5$; $\bar A=s\,(GB)\,x=2\cdot1.5\cdot(1,2,3)=(3,6,9)$.

I checked these by running the exercise functions. Parameter count: 5 trainable numbers versus 6 for the full $2\times3$ matrix; the saving only becomes dramatic at large width.

**Capacity test.** In the exercise, the pretrained $W$ is $8\times10$ and the target differs from it by a rank-2 matrix. Gradient descent on $A,B$ with $r=2$ drives the squared error from $13.17$ to about $10^{-30}$ (the update is recovered exactly), while $r=1$ stalls at about $2.9$: a rank-1 update cannot represent a rank-2 change. This is the trade-off in miniature.

For the SFT side, `build_example([5,6,7],[8,9],eos=1)` gives ids `5 6 7 8 9 1` and labels `-100 -100 -100 8 9 1`; after the shift, inputs `5 6 7 8 9` and targets `-100 -100 8 9 1`, so three of five positions carry loss: the last prompt token `7` is the first input whose target is supervised.

## Implementation

```python
import numpy as np

def lora_forward(x, W, A, B, alpha):          # x:(Bt,d_in) W:(d_out,d_in) A:(r,d_in) B:(d_out,r)
    s = alpha / A.shape[0]
    h = x @ A.T                                # (Bt, r)  -- cache for backward
    return x @ W.T + s * (h @ B.T), h

def lora_backward(G, x, h, W, A, B, alpha):   # G = dL/dy : (Bt, d_out)
    s = alpha / A.shape[0]
    GB = G @ B                                 # (Bt, r)
    dB = s * G.T @ h                           # (d_out, r)
    dA = s * GB.T @ x                          # (r, d_in)
    dx = G @ W + s * GB @ A                    # (Bt, d_in)
    return dx, dA, dB

def merge(W, A, B, alpha):
    return W + (alpha / A.shape[0]) * B @ A
```

```python
IGNORE = -100
def masked_ce_grad(p, y):                      # p: (B,T,V) probabilities, y: (B,T) with IGNORE
    valid = y != IGNORE
    g = np.zeros_like(p)
    b, t = np.nonzero(valid)
    g[b, t] = p[b, t]; g[b, t, y[b, t]] -= 1
    return g / max(1, valid.sum())
```

## Shapes

- $W:(d_{out},d_{in})$ frozen; $A:(r,d_{in})$; $B:(d_{out},r)$; $x:(B,d_{in})$; $h:(B,r)$; $y,G:(B,d_{out})$.
- SFT batch: inputs $X:(B,T)$ int, targets $Y:(B,T)$ int with `-100`; logits $(B,T,V)$; the loss divides by the number of non-masked targets, not by $B\cdot T$.
- Adapter size: $r(d_{out}+d_{in})$ per adapted matrix; a model with $n$ adapted matrices stores $n$ such pairs.

## Limits and trade-offs

- **Capacity.** A low-rank update cannot represent changes that need high rank (the rank-1 failure above). Tasks that require a lot of new knowledge, not just a new behaviour, may prefer higher rank, more layers, or full fine-tuning.
- **Hyperparameters.** $r$, $\alpha$, target layers and learning rate interact; $\alpha/r$ scaling reduces but does not remove this.
- **Quantisation error** in QLoRA is a small additional approximation on the frozen weights; reported quality is close to 16-bit LoRA on benchmarks in the source tables, but verify on your own task.
- **SFT teaches format, not truth.** If the demonstrations contain errors or invent facts, the model learns to produce confident text in that style. SFT also cannot express *preference between two acceptable answers*, which motivates the next lesson.
- **Forgetting persists** at high learning rates even with LoRA; the frozen base only limits how far the function can move.

## Common misconceptions

- "Fine-tuning teaches the model new facts reliably." It mostly shapes behaviour and style; facts are mostly set in pretraining.
- "LoRA is an approximation of the fine-tuned model that is always worse." It is a constrained fine-tune; when the needed update is low rank it can match full fine-tuning, otherwise not.
- "Masking the prompt means the model ignores it." Masked tokens carry no loss, but they remain in the context and the response attends to them.
- "Initialise A and B to zero for safety." Then neither factor ever receives gradient.
- "Merged and unmerged LoRA give different outputs." They are algebraically identical, up to floating-point rounding.

## Key takeaways

- SFT = next-token training on (prompt, response) sequences with the loss restricted to response and end tokens; low learning rate, few epochs, quality over quantity.
- LoRA freezes $W$ and learns $\Delta W=\frac{\alpha}{r}BA$ of rank $\le r$ with $r(d_{in}+d_{out})$ parameters; $B=0$ at start so the model is unchanged, and $A$ random so that $B$ gets gradient.
- Gradients: $\bar B=sG^\top h$, $\bar A=s(GB)^\top x$, $\bar x=GW+s(GB)A$.
- Merge by $W+sBA$ for zero-latency inference, or keep adapters separate per task.
- QLoRA stores the frozen base in 4 bits and trains adapters on top.

## Explain it yourself

1. Why does zeroing *both* LoRA matrices freeze training forever, and why does zeroing only $B$ not?
2. A colleague masks nothing and fine-tunes on prompt plus response. What changes in the loss, and what unwanted behaviour might it encourage?

## Further reading

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*; Dettmers et al. (2023), *QLoRA*.
- Raschka, *Build a Large Language Model From Scratch*, chapter 7 and appendix E.
- Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, for SFT as the first stage of the pipeline.

## Mastery criteria

- Build an SFT example with correctly shifted inputs and masked labels by hand.
- Derive $\bar A,\bar B,\bar x$ for a LoRA layer and pass a finite-difference check.
- Compute the LoRA parameter count and fraction for a given matrix and rank.
- Explain the initialisation, the merge identity, and what limits the method.
