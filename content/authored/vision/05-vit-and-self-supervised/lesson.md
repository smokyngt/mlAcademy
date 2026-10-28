---
id: vis.05
title: "Vision transformers and self-supervised vision"
summary: Turn an image into a sequence of patch tokens for a transformer, compare its inductive biases with a CNN's, and learn features without labels with the contrastive NT-Xent loss.
minutes: 80
teaches: [vis.patch-embed, vis.contrastive]
requires: [trf.attention, trf.position, vis.image-tensor, vis.augment, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/14-vision-transformers/docs/en.md (Sec. 'The Problem' and 'The Concept': pipeline, patch embedding, class token, positional embedding, encoder block, pre-LN, patch-size trade-off, DeiT recipe, Swin vs ConvNeXt, MAE; the Build It code was not read). READ."
  - "ai-engineering-from-scratch/phases/04-computer-vision/17-self-supervised-vision/docs/en.md (Sec. 'The Problem' and 'The Concept': three families, SimCLR/InfoNCE, DINO teacher-student, MAE masking ratio, linear-probe evaluation; the Build It code was not read). READ."
---

## Where this fits

Until about 2020, "vision model" meant "convolutional network". The transformer ([[trf.01]], [[trf.04]]) then took over language, and the question was whether it could also read images. The answer, the **vision transformer (ViT)**, is the backbone of most current vision models and of the image side of CLIP in [[vis.06]]. The second half of this lesson addresses the other bottleneck: labels are expensive, images are cheap. **Self-supervised learning** produces useful features from unlabelled images, which then feed the linear-probe and fine-tuning recipes of [[vis.02]]. The contrastive loss you implement here is the direct ancestor of CLIP's loss.

## The idea in plain words

A transformer reads a sequence of tokens. A sentence is naturally a sequence of words; an image is not. The ViT trick is blunt: cut the picture into a grid of small squares (say $16\times16$ pixels), flatten each square into a long vector, and treat the squares as words. Everything after that is a standard transformer. Because attention has no notion of "left" or "above", a position label is added to each patch, and a special learned "summary" token is prepended whose final state is read out as the image description.

A convolutional network comes with assumptions built in: nearby pixels matter together (locality) and the same pattern means the same thing anywhere (translation equivariance). A ViT has almost none of these; it must learn them. That is a weakness with little data and a strength with a lot, which explains why early ViTs lost to CNNs on ImageNet alone but won when pre-trained on far larger datasets, and why better training recipes later closed the gap.

For self-supervised learning, use this thought experiment. Take one photo, make two different edited versions of it (cropped and recoloured differently). A good feature extractor should say "same thing" for those two, and "different thing" for edited versions of other photos. Nobody needs to label the photo "cat"; the supervision comes from knowing which two edits came from one original. Training with this rule forces the network to keep what survives the edits (content) and discard what does not (exact crop, colour tint).

## Patch embedding

An image of size $H\times W$ with $C$ channels and patch size $p$ (with $p$ dividing $H$ and $W$) gives $N=(H/p)(W/p)$ patches. Flatten each patch to a vector of length $C p^2$:

$$
x_{\text{patches}}\in\mathbb{R}^{N\times Cp^2}.
$$

A learned linear projection $W_E\in\mathbb{R}^{Cp^2\times D}$ with bias $b\in\mathbb{R}^D$ maps each to a token of width $D$:

$$
z_0^{\text{patch}}=x_{\text{patches}}W_E+b\in\mathbb{R}^{N\times D}.
$$

In frameworks this is a single convolution with kernel size $p$ and stride $p$ (so patches do not overlap) and $D$ output channels, followed by flattening the spatial grid; it is exactly the same linear map applied to each patch. See [[seq.01]] for convolution shapes.

Then:

1. **Class token.** A learned vector $x_{\text{cls}}\in\mathbb{R}^D$ is prepended, giving $N+1$ tokens. Its output after the last block is the image representation fed to the classifier.
2. **Positional embedding.** A learned matrix $E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$ is added: $z_0=[x_{\text{cls}};z_0^{\text{patch}}]+E_{\text{pos}}$. Without it attention is permutation-equivariant and cannot tell the top-left patch from the bottom-right one (see [[trf.03]]).
3. **Encoder blocks.** $L$ blocks of pre-LayerNorm transformer: $z\leftarrow z+\mathrm{MSA}(\mathrm{LN}(z))$, then $z\leftarrow z+\mathrm{MLP}(\mathrm{LN}(z))$, where the MLP is $D\to4D\to D$ with GELU. Pre-norm trains deep stacks more stably than post-norm.
4. **Head.** A linear layer or small MLP on the class-token output.

For ViT-B/16 at $224\times224$: $N=(224/16)^2=196$ patches, $197$ tokens with the class token, width $D=768$, $12$ blocks and $12$ heads (about 86M parameters). Note the coincidence $Cp^2=3\cdot16\cdot16=768=D$ there, so the patch projection is square; in general it is not.

### Patch size and cost

Attention compares every token with every other, so one attention matrix has $(N+1)^2$ entries per head per layer and cost grows quadratically in the token count. Halving the patch side quadruples the tokens and multiplies the attention matrix size by $16$. Larger patches are faster but lose spatial detail; $16$ is the standard compromise, and hierarchical designs (Swin) use small patches with local windows.

### Inductive bias: CNN versus ViT

| property | CNN | ViT |
|---|---|---|
| locality | built in (small kernels) | learned (attention can attend anywhere) |
| translation equivariance | built in (weight sharing) | only through patch embedding and learned positions |
| receptive field | grows with depth | global from the first block |
| data appetite | lower | higher, unless regularised or pre-trained |
| scaling | saturates earlier | keeps improving with data and size |

Practical consequences. A plain ViT trained on ImageNet-1k alone underperformed ResNets in the original work; with pre-training on a much larger dataset it won. DeiT then reached strong ImageNet-1k results by training recipe alone: heavy augmentation (RandAugment, mixup, CutMix, random erasing), stochastic depth, repeated augmentation, and optional distillation from a CNN teacher. Swin adds shifted local windows (a locality prior), and ConvNeXt shows a modernised CNN can match them, so the lasting difference lies as much in the training recipe as in attention versus convolution. On edge hardware CNNs remain competitive.

## Self-supervised learning

Three families:

- **Contrastive** (SimCLR, MoCo, CLIP): pull together embeddings of positive pairs, push apart negatives.
- **Teacher-student** (DINO, BYOL): a student predicts the output of a slowly updated teacher; no explicit negatives.
- **Masked reconstruction** (MAE, BEiT): hide most of the image and reconstruct it.

### Derivation: the NT-Xent loss (SimCLR)

Take a batch of $N$ images. Make two random augmentations of each, giving $2N$ views. Pass them through an encoder and a small projection head to get embeddings $z_1,\dots,z_{2N}\in\mathbb{R}^d$, and L2-normalise each. The similarity of two views is the cosine

$$
s_{ij}=\frac{z_i^\top z_j}{\lVert z_i\rVert\,\lVert z_j\rVert}=z_i^\top z_j\quad(\text{after normalising}).
$$

For a view $i$ let $i^+$ be its positive partner (the other view of the same image). The task is: among the $2N-1$ other views, identify $i^+$. That is a $(2N-1)$-way classification, so use softmax cross-entropy with logits $s_{ik}/\tau$ for all $k\ne i$, where $\tau>0$ is a **temperature**:

$$
\ell_i=-\log\frac{\exp(s_{i,i^+}/\tau)}{\sum_{k\ne i}\exp(s_{ik}/\tau)}
=-\frac{s_{i,i^+}}{\tau}+\log\sum_{k\ne i}\exp\!\Big(\frac{s_{ik}}{\tau}\Big),
$$

$$
L=\frac1{2N}\sum_{i=1}^{2N}\ell_i .
$$

This is the "normalised temperature-scaled cross-entropy", NT-Xent. It is the cross-entropy of [[vis.02]] where the "classes" are the other samples in the batch. Properties worth knowing:

- **Self-similarity is excluded** ($k\ne i$); otherwise the trivial match $s_{ii}=1$ would dominate.
- **Temperature.** Dividing by a small $\tau$ sharpens the softmax, concentrating the gradient on the hardest negatives (the most similar wrong views); a large $\tau$ treats all negatives alike. Typical values are around $0.1$ (the source quotes $0.1$ as standard).
- **Negatives.** The loss has $2N-2$ negatives per anchor, so more samples in the batch means a harder, more informative task. The source says SimCLR needs batches of 512 to 8192, and MoCo decouples the number of negatives from the batch size by keeping a queue of past embeddings.
- **Sanity value.** If all similarities are equal the loss is $\log(2N-1)$, the chance level. As $\tau\to\infty$ the logits flatten, so the loss approaches this value for any embeddings.
- **Scale invariance.** Because of normalisation, multiplying an embedding by any positive number changes nothing.

### Teacher-student (DINO) in brief

Two networks share an architecture: the **student**, trained by gradient descent, and the **teacher**, an exponential moving average of the student's weights ($\theta_t\leftarrow m\theta_t+(1-m)\theta_s$ with $m\approx0.996$). Both see different augmented views; the student is trained so that its output distribution matches the teacher's on the other view (a cross-entropy between the two). There are no negatives, so a collapse to a constant output is the danger. DINO avoids it by **centring** the teacher's outputs (subtracting a running mean, which stops one dimension from dominating) and **sharpening** them (a low temperature, which stops collapse to uniform). DINOv2 scales this up and its features are a current default for retrieval and dense prediction, according to the source.

### Masked autoencoding (MAE) in brief

Mask about 75% of the patches, encode only the visible 25% with a large ViT, and train a small decoder to reconstruct the pixels of the masked patches (mean-squared error on masked patches only). The ratio is high because neighbouring pixels are strongly correlated: hiding only 15% (as BERT does for text) lets the model fill in holes by simple interpolation, while 75% forces it to represent the content. After pre-training, the decoder is discarded.

### Evaluating self-supervised features

The standard measure is the **linear probe** ([[vis.02]]): freeze the encoder, train one linear layer with labels, report top-1 accuracy. The source quotes ImageNet linear-probe accuracies of roughly 71% (SimCLR ResNet-50), 77% (DINO ViT-S/16), 76% (MAE ViT-L/16) and 86% (DINOv2 ViT-g/14); treat these as indicative, since training setups differ. Fine-tuning typically adds a few points but also mixes in the effect of retraining.

## Worked example

All numbers are reproduced by the exercise code.

**Patchify a tiny image.** The $4\times4$ single-channel image with pixels $0..15$ (row-major) and $p=2$ gives $4$ patches of $4$ numbers: patch 0 (top-left) $=[0,1,4,5]$, patch 1 (top-right) $=[2,3,6,7]$, patch 2 (bottom-left) $=[8,9,12,13]$, patch 3 $=[10,11,14,15]$. Unpatchifying restores the image exactly.

**ViT-B/16 bookkeeping.** $3\times224\times224$ with $p=16$: $196$ patches of $768$ numbers; the patch projection has $768\cdot768+768=590{,}592$ parameters; $197$ tokens (with class token); one attention matrix per head has $197^2=38{,}809$ entries. With $p=8$: $784$ patches and $784^2=614{,}656$ entries without the class token, $16$ times the $196^2=38{,}416$ of the $p=16$ case.

**NT-Xent.** $N=2$ images with orthogonal embeddings $(1,0)$ and $(0,1)$, both views identical, $\tau=0.5$. For the anchor $(1,0)$ the logits against the other three views are: partner $s=1\to2$, the two views of the other image $s=0\to0$ each. So $\ell=-\log\frac{e^2}{e^2+1+1}=\log(1+2e^{-2})=0.2395$. By symmetry all four anchors give the same value, so $L=0.2395$. With unrelated random embeddings and a very large $\tau$ the loss is about $\log(2N-1)$; for $N=8$ that is $\log15=2.708$.

## Implementation

```python
import numpy as np

def patchify(img, p):                 # (C, H, W) -> (N, C*p*p)
    C, H, W = img.shape
    x = img.reshape(C, H // p, p, W // p, p).transpose(1, 3, 0, 2, 4)
    return x.reshape((H // p) * (W // p), C * p * p)

def nt_xent(z1, z2, tau):             # (N, d), (N, d) -> scalar
    N = len(z1)
    Z = np.concatenate([z1, z2], axis=0)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T / tau
    np.fill_diagonal(S, -np.inf)                 # exclude self-similarity
    pos = (np.arange(2 * N) + N) % (2 * N)       # partner of view i
    m = S.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(S - m).sum(axis=1, keepdims=True))).ravel()
    return float(np.mean(lse - S[np.arange(2 * N), pos]))
```

Production: `timm` and `torchvision.models.vision_transformer` provide ViT; `transformers` provides ViT, DeiT, DINOv2 and MAE checkpoints with the same loading API as text models; `lightly` and the official DINO/MAE repositories implement the self-supervised recipes at scale.

## Shapes

| object | shape |
|---|---|
| image | $(C,H,W)$, batch $(B,C,H,W)$ |
| patches | $(N,Cp^2)$, $N=HW/p^2$ |
| tokens after embedding and class token | $(N+1,D)$, batch $(B,N+1,D)$ |
| attention matrix per head | $(N+1,N+1)$ |
| projection-head embeddings for NT-Xent | $(2N_{\text{batch}},d)$ |
| similarity matrix | $(2N_{\text{batch}},2N_{\text{batch}})$ |

## Common misconceptions

- **"ViTs cannot work without huge data."** They need strong priors or recipes (augmentation, pre-training, self-supervision), not necessarily JFT-scale data.
- **"A ViT looks at the image globally, so patch size does not matter."** Patch size sets the token count, hence resolution of detail and the quadratic attention cost.
- **"The class token holds a label."** It is a learned query-like vector; it gathers information from the patches through attention.
- **"Contrastive learning needs labels."** It needs only the knowledge of which views come from the same image.
- **"Negatives are other classes."** In a random batch some negatives may share the class of the anchor ("false negatives"); this noise is tolerated at scale.
- **"Temperature is a detail."** It controls how much the loss focuses on hard negatives and strongly affects results.
- **"Self-supervised pre-training makes fine-tuning unnecessary."** The probe is a measurement of features; fine-tuning usually adds accuracy.

## Key takeaways

- ViT = patchify, linear projection, class token, positional embedding, standard transformer; $N=HW/p^2$ and attention cost scales as $(N+1)^2$.
- CNNs bake in locality and translation equivariance; ViTs learn them, so they depend on data and training recipe.
- NT-Xent is softmax cross-entropy over the other views in the batch, with cosine similarity and temperature.
- DINO avoids collapse with an EMA teacher, centring and sharpening; MAE masks 75% of patches.
- Evaluate with a linear probe on frozen features.

## Explain it yourself

1. Explain why a ViT needs positional embeddings and how the number of tokens changes when the patch size is halved.
2. Derive the NT-Xent loss as a classification problem and explain the roles of normalisation, temperature and the number of negatives.

## Further reading

- The two source lessons (ViT, DeiT, Swin, ConvNeXt, MAE, SimCLR, DINO references).
- [[trf.04]] for the transformer block, [[vis.06]] for the contrastive objective extended to image-text pairs.

## Mastery criteria

You can compute token counts and parameter counts for a given patch size, write patchify and its inverse, explain the CNN/ViT inductive-bias difference, and implement and sanity-check NT-Xent (value at chance level, scale invariance, aligned versus unrelated views).
