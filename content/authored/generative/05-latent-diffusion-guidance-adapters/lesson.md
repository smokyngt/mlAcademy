---
id: gen.05
title: "Latent diffusion, conditioning and fine-tuning"
summary: Run diffusion in a compressed latent space, steer it with text through cross-attention, derive and implement classifier-free guidance, and see how ControlNet and LoRA adapt a frozen model.
minutes: 80
teaches: [gen.latent-diffusion, gen.cfg]
requires: [gen.diffusion, gen.vae, trf.attention]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/07-latent-diffusion-stable-diffusion/docs/en.md: two-stage design (VAE then diffusion on latents), text conditioning by cross-attention, model table, classifier-free guidance formula and typical scales, common pitfalls. READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai/08-controlnet-lora-conditioning/docs/en.md: ControlNet (trainable encoder copy, zero-convolutions), LoRA (low-rank delta, parameter counts, alpha scaling, stacking). READ."
---

## Where this fits

[[gen.04]] gave a diffusion model that works but is expensive: every one of ~1000 network calls processes a full-resolution image. Production text-to-image systems fix this in three ways that this lesson covers. **Latent diffusion** runs the process in a compressed space made by a VAE ([[gen.02]]). **Conditioning** injects a text prompt, using the attention mechanism of [[trf.01]]. **Classifier-free guidance** trades diversity for prompt adherence with one cheap formula. Finally, **ControlNet** and **LoRA** let you adapt a huge frozen model with a tiny number of new parameters; LoRA is the same idea you meet again for language models ([[llm.04]]).

## The idea in plain words

*Latent diffusion.* Most pixels in a photograph carry fine texture that a good compressor can throw away and re-invent later. So first train a compressor (an autoencoder) that turns a large picture into a small grid of numbers and back. Then do all the expensive denoising on the small grid, and decompress once at the end. It is like editing a thumbnail's structure and then letting a skilled retoucher add the pixel-level detail.

*Conditioning.* A prompt is a list of word vectors. At each denoising step every location of the image asks "which words are relevant to me?" and mixes in their information. That question-and-answer is cross-attention.

*Guidance.* Run the denoiser twice: once told the prompt, once not. The difference between the two answers is the direction that "means the prompt". Exaggerate that direction and the image follows the prompt more strongly, at the cost of variety.

*Adapters.* To teach a giant model a new trick, do not rewrite it. Clamp a small, cheap add-on next to the frozen model: a side network that reads an extra signal (ControlNet), or a thin low-rank correction to some weight matrices (LoRA).

## Latent diffusion

**Stage 1.** Train an autoencoder with encoder $E$ and decoder $D$ (reconstruction loss, perceptual loss, a small KL weight, often an adversarial loss) so that $z=E(x)$ is a compact latent and $D(z)\approx x$. A typical setting downsamples 8x in each spatial direction. For a $512\times512\times3$ image the latent is $64\times64\times4$: $786{,}432$ numbers become $16{,}384$, a **48x** reduction in size (the source text quotes about 16x fewer pixels; the 8x downsampling per side gives $8^2=64$ fewer spatial positions and 4 instead of 3 channels, which is the 48 above).

**Stage 2.** Treat $z_0=E(x_0)$ as the data and train the DDPM of [[gen.04]] on it: the noise-prediction loss is *identical*, only the tensor is smaller. Sampling: draw $z_T\sim\mathcal N(0,I)$, denoise to $z_0$, output $x=D(z_0)$.

Because the autoencoder is frozen after stage 1, diffusion only needs to model high-level structure; the decoder supplies the high-frequency detail. Compute drops roughly in proportion to the size reduction.

## Conditioning with cross-attention

A frozen text encoder turns the prompt into token embeddings $C\in\mathbb R^{n_\text{tok}\times d_c}$. Inside the denoising network, image features $F\in\mathbb R^{n_\text{pix}\times d}$ attend to the text:

$$
\mathrm{CrossAttn}(F,C)=\mathrm{softmax}\!\Big(\frac{(FW_Q)(CW_K)^\top}{\sqrt{d_k}}\Big)(CW_V),
$$

with $Q=FW_Q\in\mathbb R^{n_\text{pix}\times d_k}$, $K=CW_K,\ V=CW_V\in\mathbb R^{n_\text{tok}\times d_k}$. It is the attention of [[trf.01]] except that queries come from the image and keys and values from the text. Each pixel position gets a convex combination of text values, so text influences the image only through these layers (in newer diffusion transformers text and image tokens are processed jointly).

## Classifier-free guidance: derivation

We would like samples that are not just plausible but *strongly consistent with* a condition $c$. Bayes' rule on the conditional density gives $p(x\mid c)\propto p(x)\,p(c\mid x)$, hence for the scores (gradients of the log-density)

$$
\nabla_x\log p(x\mid c)=\nabla_x\log p(x)+\nabla_x\log p(c\mid x).
$$

The second term is the gradient of a classifier. Sharpen the classifier by raising it to a power $w$: $p_w(x\mid c)\propto p(x)\,p(c\mid x)^{w}$, with score

$$
\nabla_x\log p_w(x\mid c)=\nabla_x\log p(x)+w\,\nabla_x\log p(c\mid x)
=\nabla_x\log p(x)+w\big(\nabla_x\log p(x\mid c)-\nabla_x\log p(x)\big),
$$

where the second equality again uses the Bayes identity. No classifier is needed: the two scores are exactly what a conditional and an unconditional denoiser provide. Since a noise predictor is a rescaled score ($\varepsilon=-\sqrt{1-\bar\alpha_t}\,\nabla\log p_t$, [[gen.04]]), multiplying through by $-\sqrt{1-\bar\alpha_t}$ gives the **classifier-free guidance** rule

$$
\boxed{\tilde\varepsilon=\varepsilon_\theta(x_t,\varnothing)+w\big(\varepsilon_\theta(x_t,c)-\varepsilon_\theta(x_t,\varnothing)\big)}
$$

Here $\varnothing$ is the "empty" condition. One network is trained for both: with probability around 10% the condition is dropped during training. $w=0$ ignores the prompt, $w=1$ is the plain conditional model, $w>1$ extrapolates beyond it. The source lesson writes the same rule as $(1+w')\varepsilon_c-w'\varepsilon_u$, i.e. its scale $w'$ equals $w-1$ here; always check which convention a library uses (diffusion libraries usually call our $w$ the "guidance scale"). Typical values are about 3-8 in those conventions; much larger values give saturated, over-sharp, low-diversity images. The cost: two network evaluations per step.

## Worked example: guidance on a two-class 1-D mixture

Data: two classes with equal prior, class means $-2$ and $+2$, common standard deviation $0.5$. Both the conditional noise predictor (the Gaussian denoiser of [[gen.04]]) and the unconditional one (a responsibility-weighted mixture of the two posterior means) are exact. Sample 20,000 points with $T=500$ steps, condition $c=$ "class +2", seed 0:

| $w$ | sample mean | sample std |
|---|---|---|
| 0 (unconditional) | $-0.026$ | $2.058$ (both modes) |
| 1 (conditional) | $1.992$ | $0.503$ |
| 2 | $2.078$ | $0.486$ |
| 4 | $2.138$ | $0.483$ |
| 8 | $2.190$ | $0.483$ |

So $w=1$ recovers the class (mean 2, std 0.5); $w>1$ pushes samples past the class mean, towards the extreme of the class direction, away from the other class. In real images that overshoot shows up as over-saturated, exaggerated features. Also verified in the exercise: the Gaussian conditional noise predictor equals $-\sqrt{1-\bar\alpha_t}$ times the score of $\mathcal N(\sqrt{\bar\alpha_t}m,\bar\alpha_ts^2+1-\bar\alpha_t)$ to machine precision.

## ControlNet and LoRA

**ControlNet.** Text cannot say where the dog stands. ControlNet keeps the pretrained denoiser frozen, makes a *trainable copy* of its encoder that also receives a spatial condition (edge map, depth, pose), and adds the copy's outputs to the frozen network's decoder features through **zero-initialised convolutions**:

$$
\text{features}\leftarrow\text{features}+\mathrm{zeroconv}\big(\text{encoder\_copy}(x_t,\text{condition})\big).
$$

Because the zero-convolutions start at 0, the model initially behaves exactly like the original, and training cannot damage it; it learns a delta. Several ControlNets can be summed with weights.

**LoRA** (low-rank adaptation). For a frozen weight $W\in\mathbb R^{d_\text{out}\times d_\text{in}}$ learn a low-rank update $\Delta W=\frac{\alpha}{r}BA$ with $A\in\mathbb R^{r\times d_\text{in}}$, $B\in\mathbb R^{d_\text{out}\times r}$, $r\ll d$. Initialise $B=0$ (so $\Delta W=0$ at the start) and $A$ randomly. Trainable parameters: $r(d_\text{out}+d_\text{in})$ instead of $d_\text{out}d_\text{in}$. For $d=640$ and $r=16$ that is $20{,}480$ against $409{,}600$, a $20\times$ reduction (5% of the weights). A rank-$r$ matrix is a good fit when the update lives in a low-dimensional subspace, which fine-tuning updates empirically tend to do (see [[cml.05]] for the SVD view of low rank). At inference you can merge $W'=W+\Delta W$ (no extra latency) or scale the adapter, and stack several adapters additively. LoRA adapters teach concepts, styles and characters; ControlNet teaches new *conditioning signals*.

## Implementation

```python
import numpy as np

def cfg(eps_u, eps_c, w):
    return eps_u + w * (eps_c - eps_u)

def cross_attention(F, C, Wq, Wk, Wv):
    Q, K, V = F @ Wq, C @ Wk, C @ Wv
    S = Q @ K.T / np.sqrt(K.shape[1])
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return P @ V, P

def lora_delta(A, B, alpha=1.0):
    r = A.shape[0]
    return (alpha / r) * (B @ A)
```

In practice: `diffusers` pipelines take `guidance_scale`, run the unconditional and conditional passes as one batch of two, and `peft`/`load_lora_weights` attach adapters.

## Shapes

Latent: `(B, 4, 64, 64)`. Prompt embeddings `C: (B, n_tok, d_c)` with `n_tok` around 77 for CLIP-style encoders. Cross-attention: `F: (B, n_pix, d)`, scores `(B, n_pix, n_tok)`, output `(B, n_pix, d_k)`. LoRA: `A: (r, d_in)`, `B: (d_out, r)`, `ΔW: (d_out, d_in)`.

## Common misconceptions

- **"A higher guidance scale means a better image."** Past a moderate value, diversity collapses and colours saturate.
- **"Guidance needs a separate classifier."** That was the earlier method; classifier-free guidance uses the same network with the condition dropped.
- **"The VAE in Stable Diffusion is trained jointly with the diffusion model."** It is trained first and frozen.
- **"LoRA is a small model."** It is a small *delta* to a large model; it is useless without the base weights.
- **"ControlNet fine-tunes the base model."** The base stays frozen; a copy is trained.

## Key takeaways

- Diffusion in the latent space of a frozen autoencoder cuts compute by roughly the compression factor; the loss is unchanged.
- Text enters through cross-attention: queries from the image, keys/values from the prompt.
- $\tilde\varepsilon=\varepsilon_u+w(\varepsilon_c-\varepsilon_u)$ comes from sharpening $p(c\mid x)$; $w=1$ is the plain conditional model.
- ControlNet adds spatial control with zero-init side connections; LoRA adds $\frac{\alpha}{r}BA$ with $B=0$ initially.

## Explain it yourself

1. Derive the guidance formula from $p_w(x|c)\propto p(x)p(c|x)^w$.
2. Why must LoRA's $B$ (or one of its factors) start at zero, and why does ControlNet use zero-initialised convolutions?

## Further reading

- Source lessons: phase 08 lessons 07 (latent diffusion) and 08 (ControlNet, LoRA). Original papers (Rombach et al.; Ho and Salimans; Zhang et al.; Hu et al.) were not read for this lesson.
- Next: [[gen.06]].

## Mastery criteria

You can explain the two stages of latent diffusion, derive and implement classifier-free guidance, write cross-attention with explicit shapes, and compute the parameter saving and initial behaviour of a LoRA adapter.
