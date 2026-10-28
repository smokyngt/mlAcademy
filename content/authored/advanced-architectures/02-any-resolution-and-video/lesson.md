---
id: adv.02
title: "Any-resolution and video-language models"
summary: Pack images of any shape into one sequence, give patches 2D and 3D positions, and compute the token and KV-memory cost of hours of video.
minutes: 65
teaches: [adv.dynamic-res, adv.video-tokens]
requires: [adv.vlm, trf.mask, trf.rope, adv.visual-tokens]
sources:
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/06-any-resolution-patch-n-pack/docs/en.md: NaViT patch-n-pack, block-diagonal mask, AnyRes, M-RoPE, token budgets, min/max pixels. READ."
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps/docs/en.md: native dynamic resolution (multiples of 28, 960 tokens for 1120x672), M-RoPE, dynamic FPS, absolute time. READ (first part)."
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/17-video-language-temporal-grounding/docs/en.md: video token reduction strategies, per-frame pooling, TMRoPE, frame sampling. READ (first part)."
  - "ai-engineering-from-scratch/phases/12-multimodal-ai/18-long-video-million-token/docs/en.md: token counts of long video, brute context vs ring attention vs compression vs agentic retrieval. READ (first part)."
---

## Where this fits

[[adv.01]] gave every image a fixed size and therefore a fixed token count. Real inputs are not square: a receipt is tall, a chart is wide, a phone screenshot is 9:19.5, a scan may be 4096 pixels across, and a video is hundreds of images in a row. Squashing everything to a 336x336 square distorts text and wastes tokens, and a naive "every frame at full resolution" video blows through any context window. This lesson covers the three ways models avoid those problems (tiling, packing, native resolution with generalised rotary positions) and builds the calculator you should run before committing to a design: how many tokens, and how many gigabytes of KV cache.

## The idea in plain words

Picture a photo lab printing everyone's photos on identical square paper. A long panorama gets crushed, a tall receipt gets padded with blank paper, and a huge map gets shrunk until the labels vanish. There are three fixes.

1. **Cut the big picture into standard-size prints** and also print one small overview (AnyRes tiling).
2. **Print every photo at its own natural size**, and stack the prints into one pile, taking care that a viewer reading one photo never looks at its neighbour (patch-n-pack with a block mask).
3. **Describe where each patch is by coordinates** (row, column, and for video, time) rather than by a numbered slot in a fixed table, so any size is legal (M-RoPE).

For video, add one more idea: you do not need thirty photographs per second of a person sitting still. Sample fewer frames, shrink each by averaging neighbouring patches, and label every frame with its true time.

## Native resolution and packing

**Square resize, crop, pad** each fail: resize distorts text, crop throws content away, padding wastes more than half the tokens on a tall page and attention pays for pad tokens quadratically.

**Patch-n-pack (NaViT).** For each image compute its own patch grid; flatten each into a sequence of length $n_b$; concatenate all $B$ sequences into one of length $S=\sum_b n_b$ with no padding. To keep images separate, use a **block-diagonal** attention mask $M\in\{0,1\}^{S\times S}$, with $M_{ij}=1$ iff positions $i$ and $j$ lie in the same image (masks were introduced in [[trf.02]]). Score entries computed: $\sum_b n_b^2$, instead of $B\cdot(\max_b n_b)^2$ for padding. For four images of 256, 576, 729 and 1024 tokens: $S=2585$, packed entries $1{,}977{,}329$, padded entries $4\cdot1024^2=4{,}194{,}304$, and one undivided sequence would score $2585^2=6{,}682{,}225$ pairs. (FlashAttention's variable-length path takes cumulative lengths instead of a dense mask and skips the zeros.)

**Native dynamic resolution (Qwen2-VL style).** The vision transformer accepts any height and width divisible by $u=p\cdot m$ (patch $p=14$, an $m\times m=2\times2$ spatial merge of neighbouring patches into one token, so $u=28$):

$$
N=\frac{H'}{u}\cdot\frac{W'}{u},\qquad H',W' \text{ the sides rounded to multiples of } u .
$$

A 1120x672 image gives $40\times24=960$ tokens. A cap (Qwen's `max_pixels`; the exercise uses `max_tokens`) protects you from the 4000x3000 photo, and a floor (`min_pixels`) from the 5x5 icon.

**Position without a table: M-RoPE.** Rotary embeddings ([[trf.03]]) rotate query and key coordinates by an angle proportional to position, so scores depend on relative offset. Multimodal RoPE gives each token a **triple** $(t, r, c)$ and assigns different slices of the rotated dimensions to time, row and column. For an image $t=0$. There is no learned table to run out of, so any size works without resizing. `mrope_positions(2,2,3)` produces twelve triples running from $(0,0,0)$ to $(1,1,2)$.

**AnyRes tiling (LLaVA-NeXT)** is the alternative when the encoder is frozen at one resolution (336): choose a grid layout (1x2, 2x2, 3x1...) that fits the aspect ratio, cut tiles, add a downscaled thumbnail for global context. Cost is multiplicative: $(r c+1)\times 576$ tokens. A 2x2 grid gives 2880; a 4x4 grid on a 1344x1344 image gives 9792.

| Strategy | Tokens for a large image | Needs an encoder trained for it? |
|---|---|---|
| Square resize 336 | 576 (detail lost) | no |
| AnyRes tiles + thumbnail | $(rc+1)\cdot576$ | no |
| Native + M-RoPE | about $HW/u^2$ up to the cap | yes |

Rule of thumb from the source: pick a per-task token cap (OCR and documents 1000-4000, charts and UI 700-1000, natural photos 250-600) and encode at native aspect ratio up to that cap.

## Video: tokens and time

A video of $D$ seconds sampled at $f$ frames per second with $\tau$ tokens per frame has

$$
N_{\text{video}} = D\,f\,\tau
$$

tokens. Three levers reduce it: **fewer frames** (lower $f$, or dynamic sampling that spends frames where motion happens), **fewer tokens per frame** (average-pool the token grid by a factor $s$ per side: $\tau\to\lceil g_h/s\rceil\lceil g_w/s\rceil$, so a $24\times24=576$ grid becomes $8\times8=64$ for $s=3$), and **a learned compressor** (a Q-Former per clip, or one summary token per clip).

**Time positions.** Telling the model that frame 5 precedes frame 6 needs a temporal position. Learned frame indices are simplest. With dynamic FPS the index is misleading (frames are unevenly spaced), so Qwen2.5-VL ties the temporal rotary position to **absolute timestamps**; the model can then answer "at what second does the cat jump?" rather than "early in the clip".

**Memory.** The KV cache stores keys and values for every token in every layer:

$$
\text{KV bytes}=2\cdot L\cdot h_{kv}\cdot d_h\cdot N\cdot b ,
$$

with $L$ layers, $h_{kv}$ key-value heads, head dimension $d_h$, $N$ tokens and $b$ bytes per value (2 for BF16). It grows **linearly** in $N$; the prefill attention compute grows **quadratically**: causal attention scores $N(N+1)/2$ pairs. (The cache is introduced in [[sys.01]]; [[adv.04]] shows how to shrink it.)

## Worked example

LLM: 32 layers, 8 KV heads, $d_h=128$, BF16 (a Llama-3-8B-shaped model, used as one illustrative choice).

| Input | Tokens | KV cache |
|---|---|---|
| 5 min at 1 fps, 576 tokens/frame | $300\cdot576=172{,}800$ | 21.1 GiB |
| same, 3x3 pooled (64 tokens/frame) | $300\cdot64=19{,}200$ | 2.3 GiB |
| 30 min at 1 fps, 81 tokens/frame | $1800\cdot81=145{,}800$ | 17.80 GiB |
| 2 h at 1 fps, 81 tokens/frame | $583{,}200$ | 71.2 GiB |

Per token the cache costs $2\cdot32\cdot8\cdot128\cdot2=131{,}072$ bytes = 128 KiB. So the 2-hour video leaves almost nothing of an 80 GB card for weights, which is why long-video systems use ring attention across devices, aggressive token compression, or **agentic retrieval** (an LLM queries the video as a database and encodes only the relevant clips). Brute context (Gemini 1.5 style, 1M+ tokens) hides the engineering but not the arithmetic.

Receipt example: a 600x1500 receipt at patch 14 without merging gives $43\times107=4601$ tokens at native resolution versus 576 after a square resize to 336. The eightfold gap is where OCR accuracy lives.

## Implementation

```python
import numpy as np

def packed_mask(lengths):
    S = sum(lengths); M = np.zeros((S, S), bool); o = 0
    for l in lengths:
        M[o:o+l, o:o+l] = True; o += l
    return M

def kv_gib(layers, kv_heads, head_dim, n_tokens, b=2):
    return 2 * layers * kv_heads * head_dim * n_tokens * b / 2**30
```

Production code: Hugging Face processors expose `min_pixels` and `max_pixels` for Qwen-VL, and FlashAttention's varlen interface (`cu_seqlens`) implements packing without a dense mask.

## Shapes

| Tensor | Shape |
|---|---|
| packed patch sequence | (S, d), $S=\sum_b n_b$ |
| block mask | (S, S) boolean |
| M-RoPE positions | (N, 3) |
| video tokens | (frames x tokens_per_frame, d_llm) |
| KV cache (per sequence) | (L, 2, h_kv, N, d_h) |

## Common misconceptions

- **"Native resolution removes the token problem."** It makes tokens proportional to pixels; a large scan still costs thousands without a cap.
- **"Packing changes the math of attention."** Inside each block it is identical; the mask only forbids cross-image scores.
- **"More FPS means better video understanding."** Past a point you add redundant tokens; dynamic or event-driven sampling and temporal timestamps often matter more.
- **"Pooling is free."** It loses fine spatial detail (reading small text); it suits action questions better than OCR.
- **"KV memory is quadratic in length."** The cache is linear; the prefill compute is quadratic.

## Key takeaways

- Tokens are about pixels over $u^2$ at native resolution; AnyRes multiplies by the number of tiles plus a thumbnail.
- Packing uses a block-diagonal mask so images do not mix and padding is never paid for.
- M-RoPE gives patches (time, row, column) coordinates, so sizes need no table.
- Video cost is $Df\tau$ tokens and $2Lh_{kv}d_hNb$ bytes; use sampling, pooling, compression or retrieval to stay inside the budget.

## Explain it yourself

1. Why does a block-diagonal mask make a packed batch equivalent to separately processed images?
2. A user wants to analyse a 90-minute lecture video. List three strategies, their costs and the information each loses.

## Further reading

- NaViT (Dehghani et al., 2023), Qwen2-VL and Qwen2.5-VL technical reports, LLaVA-NeXT, as summarised in the source lessons.
- [[adv.01]] for connectors; [[sys.01]] for KV caches; [[adv.04]] for ways to shrink them.

## Mastery criteria

You can compute token counts for native-resolution, AnyRes and video inputs, build a packing mask, state what M-RoPE adds, and estimate KV-cache GiB for a given model shape and input length.
