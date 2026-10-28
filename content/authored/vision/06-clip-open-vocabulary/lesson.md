---
id: vis.06
title: "CLIP and open-vocabulary vision"
summary: Align an image encoder and a text encoder with a symmetric contrastive loss, derive and implement it with a temperature, and use the shared space for zero-shot classification and retrieval.
minutes: 75
teaches: [vis.clip]
requires: [vis.contrastive, vis.patch-embed, trf.architectures, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/18-open-vocab-clip/docs/en.md (Sec. 'The Problem' and 'The Concept': two towers, symmetric objective, SigLIP, zero-shot recipe, prompt templates, uses; the Build It code was not read). READ."
  - "ai-engineering-from-scratch/phases/12-multimodal-learning/02-clip-contrastive-pretraining/docs/en.md (Sec. 'The Problem' and 'The Concept': dual encoder, InfoNCE, temperature, sigmoid loss, zero-shot, linear probe vs fine-tuning, zero-shot ceiling; the code and exercises were not read). READ."
---

## Where this fits

A classifier trained on ImageNet can name only its 1000 classes; a new category means new labels and retraining. CLIP changed the setting: train on hundreds of millions of image-caption pairs scraped from the web, and the resulting model can classify into any list of categories you can write in words. Almost every modern vision-language system, from text-to-image generators to multimodal chat models, sits on a CLIP-style embedding space. This lesson uses the ViT of [[vis.05]] as the image tower, the transformer text encoder of [[trf.04]] as the text tower, and the contrastive loss of [[vis.05]], now across two modalities. The transfer vocabulary (linear probe, fine-tuning) is from [[vis.02]].

## The idea in plain words

Imagine two translators: one reads photographs, one reads sentences, and each writes a short "address" (a list of numbers) for what they read. Training teaches them to agree: a photo of a dog on a beach and the sentence "a dog running on the beach" must get nearby addresses, while the same photo and the sentence "a plate of pasta" must get distant ones. The only supervision is that each pair in the training data belongs together; every other pairing in the same batch counts as a mismatch for free.

After that, there is no fixed list of classes any more. To classify a new photo into {cat, dog, bicycle}, write the sentences "a photo of a cat", "a photo of a dog", "a photo of a bicycle", get an address for each sentence, get an address for the photo, and choose the sentence whose address is nearest. A new class costs one sentence, not a dataset. The same address book also works for search: type a sentence, find the nearest photos.

## Setup and the loss

**Two towers.** An image encoder $f$ (a ViT or ResNet) and a text encoder $g$ (a transformer) each end in a linear projection to the same dimension $d$. For a batch of $N$ pairs $(x_i,y_i)$ compute and L2-normalise

$$
u_i=\frac{f(x_i)}{\lVert f(x_i)\rVert},\qquad v_i=\frac{g(y_i)}{\lVert g(y_i)\rVert}\qquad(u_i,v_i\in\mathbb{R}^d).
$$

With $U,V\in\mathbb{R}^{N\times d}$ stacking these rows, the **logit matrix** is

$$
S=\frac{UV^\top}{\tau}\in\mathbb{R}^{N\times N},\qquad S_{ij}=\frac{\cos(f(x_i),g(y_j))}{\tau},
$$

where $\tau>0$ is the **temperature**. In CLIP it is a learned scalar, parametrised through its logarithm and initialised at $0.07$ according to the sources (the learned value is clipped so it cannot collapse towards zero).

### Derivation: symmetric InfoNCE

Fix image $i$. The task: among the $N$ captions in the batch, find its own. That is an $N$-way classification with logits $S_{i1},\dots,S_{iN}$ and true class $i$. The cross-entropy of [[vis.02]] gives

$$
\ell^{\,i\to t}_i=-\log\frac{\exp(S_{ii})}{\sum_{j=1}^N\exp(S_{ij})}=-S_{ii}+\log\sum_{j}\exp(S_{ij}).
$$

Doing the same from the text side (fix caption $j$, find its image among $N$ images) uses the column $S_{\cdot j}$:

$$
\ell^{\,t\to i}_j=-S_{jj}+\log\sum_{i}\exp(S_{ij}).
$$

The CLIP objective averages both directions and all items:

$$
L=\frac12\left(\frac1N\sum_{i=1}^N\ell_i^{\,i\to t}+\frac1N\sum_{j=1}^N\ell_j^{\,t\to i}\right).
$$

In matrix terms: image-to-text is the row-wise softmax cross-entropy of $S$ with the identity as target, text-to-image is the same for $S^\top$. Compared with NT-Xent in [[vis.05]], the positives live in two different modalities so the similarity matrix is $N\times N$ with no diagonal to mask; the negatives of image $i$ are the $N-1$ other captions.

**Gradient with respect to the logits.** From the single-direction formula (the same $p-e_y$ rule as in [[vis.02]]), with row-softmax $P=\mathrm{softmax}_{\text{row}}(S)$ and column-softmax $Q=\mathrm{softmax}_{\text{row}}(S^\top)^\top$ (so each column of $Q$ sums to one) and $E$ the identity:

$$
\frac{\partial L}{\partial S}=\frac1{2N}\big[(P-E)+(Q-E)\big]\in\mathbb{R}^{N\times N}.
$$

Interpretation: each wrong pairing $(i,j)$ is pushed down in proportion to how much softmax mass it currently gets, in both directions; each correct pairing is pushed up. Gradients with respect to the embeddings follow by the chain rule through $S=UV^\top/\tau$ and the normalisation (a factor $1/\tau$ appears, so small $\tau$ makes the update larger and sharper).

**Temperature.** Since $|\cos|\le1$, the logits lie in $[-1/\tau,\,1/\tau]$. A large $\tau$ squeezes the logits together, so even a perfectly aligned model keeps a high loss, and all negatives look alike. A small $\tau$ gives wide logits and a sharp softmax that concentrates on the hardest negatives; too small, and training becomes unstable or over-focused on noise. This is why CLIP learns $\tau$. As $\tau\to\infty$ every row becomes uniform and the loss tends to $\log N$.

**Batch size.** Each pair has $N-1$ negatives, so larger batches give a stronger signal; CLIP trained at a batch size around 32k according to the sources. In distributed training the softmax needs all embeddings on every device (an all-gather), which is a communication cost.

**Sigmoid alternative (SigLIP).** Replace the softmax with an independent binary loss for every pair: with $y_{ij}=+1$ on the diagonal and $-1$ elsewhere, $L=\frac1N\sum_{i,j}\log\big(1+e^{-y_{ij}S_{ij}}\big)$ (the sources also add a learned bias). There is no row or column normalisation, so no all-gather is needed and it behaves well at small and very large batches. The sources report it matches or beats CLIP at equal data.

*A known result that the sources only mention:* the InfoNCE loss with $N$ candidates gives a lower bound on the mutual information between the two views, $I\ge\log N-L$, which is why the loss can be read as "how many bits of the pairing did the model recover".

## Zero-shot classification, prompts and retrieval

**Zero-shot recipe.** With class names $c=1..K$:

1. Build prompts, for example "a photo of a {class}".
2. Encode them: class embeddings $T\in\mathbb{R}^{K\times d}$, L2-normalised.
3. Encode the test image: $u\in\mathbb{R}^{d}$, normalised.
4. Score $s=Tu\in\mathbb{R}^K$ (cosine similarities) and predict $\arg\max_c s_c$.

No parameter is trained on the target classes. The temperature does not change the argmax; it matters only if you want probabilities (a softmax over $s/\tau$).

**Prompt engineering.** The wording changes accuracy by a few points. The sources note that the original work used about 80 templates per class ("a photo of a {}", "a sketch of a {}", ...). The embeddings of all templates for a class are averaged, then the mean is re-normalised, for an improvement of roughly one to three ImageNet points. With $K$ classes and $M$ templates the text side costs $KM$ encoder passes once, and the class embeddings are cached.

**Retrieval.** Embed all images once. For a text query, rank images by cosine similarity (text-to-image); for an image query, rank captions (image-to-text). **Recall@k** is the fraction of queries whose correct item is among the top $k$.

**Probes and fine-tuning.** Zero-shot is a baseline. A linear probe on frozen CLIP image features usually beats zero-shot on a specific in-domain task, and full fine-tuning can beat the probe in-domain but may hurt the zero-shot transfer to other categories (three regimes, three trade-offs).

**What it is used for.** Direct zero-shot classification; image search; text-conditioned detection and segmentation (open-vocabulary detectors wrap a CLIP-style text tower around a detector, see [[vis.03]] and [[vis.04]]); the vision encoder of multimodal chat models; conditioning for text-to-image generation.

## Limits

- **Closed by the training data.** The embedding space only covers what the web-scraped data described; the sources cite a zero-shot ImageNet ceiling of about 76% for the largest CLIP-class models of that era, with newer SigLIP 2 models above 80%. Treat these numbers as indicative.
- **Prompt sensitivity.** Class names are ambiguous words ("crane", "mouse"), and results depend on wording.
- **Noisy supervision.** Captions are loose descriptions; fine-grained distinctions, counting, spatial relations and text in images are common weak spots (general experience with these models, not claimed from the two source lessons).
- **Data bias.** The model reflects the biases of its web data, so it must be audited before decisions about people.
- **Cosine similarity is not calibrated probability.** Do not read a raw similarity of 0.3 as "30% sure".
- **Global embedding.** A single vector per image has no location; dense tasks need extra machinery.

## Worked example

All numbers are reproduced by the exercise code.

**Perfect alignment, $N=3$, $\tau=0.5$.** $U=V=I_3$. Then $S=2I$ (diagonal $2$, off-diagonal $0$). Each row's softmax gives the correct caption probability $e^2/(e^2+1+1)=0.787$, so $\ell=-\log0.7869=\log(1+2e^{-2})=0.2395$. The columns are identical by symmetry, so $L=0.2395$.

**Two pairs, imperfect text.** $U=I_2$, text embeddings $V=\begin{pmatrix}0.8&0.6\\0.6&0.8\end{pmatrix}$ (unit rows). The cosines are $0.8$ on the diagonal and $0.6$ off it.

- At $\tau=0.1$: $S=\begin{pmatrix}8&6\\6&8\end{pmatrix}$. Row softmax is $(0.881,0.119)$, so $\ell=-\log0.881=\log(1+e^{-2})=0.1269$ and $L=0.1269$ (both directions equal).
- At $\tau=1$: $S=\begin{pmatrix}0.8&0.6\\0.6&0.8\end{pmatrix}$ and $L=\log(1+e^{-0.2})=0.5981$. A larger temperature leaves a much higher loss for the same embeddings.
- Gradient at $\tau=0.1$: $P-E=\begin{pmatrix}-0.119&0.119\\0.119&-0.119\end{pmatrix}$, and with $Q=P$ here, $\partial L/\partial S=\frac1{2N}(2(P-E))=\begin{pmatrix}-0.0596&0.0596\\0.0596&-0.0596\end{pmatrix}$: raise the matching logits, lower the mismatched ones, equally.

**Zero-shot.** Class embeddings are the unit vectors $e_1,e_2,e_3$ ("cat", "dog", "car"); images $(0.9,0.1,0)$, $(0.1,0.2,0.9)$, $(0,1,0.1)$, $(2,0,0.1)$ are classified as $0,2,1,0$. Rescaling a class embedding by $5$ or $0.1$ changes nothing, because vectors are normalised first.

**Prompt ensembling.** Two classes with two template embeddings each: class 0 has $(1,0)$ and $(0,1)$, class 1 has $(1,0)$ twice. The mean-then-normalise embeddings are $(0.707,0.707)$ and $(1,0)$.

**Recall@k.** Four images against the identity text matrix, where image 0 is $(1,2,0,0)$: its cosine to caption 1 is $0.894$ and to its own caption $0.447$. The correct caption ranks second, so recall@1 $=3/4$ and recall@2 $=1$.

## Implementation

```python
import numpy as np

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def softmax_rows(L):
    Z = L - L.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)

def clip_loss(I, T, tau):                      # I, T: (N, d)
    L = l2_normalize(I) @ l2_normalize(T).T / tau   # (N, N)
    N = len(L)
    i2t = -np.log(softmax_rows(L)[np.arange(N), np.arange(N)]).mean()
    t2i = -np.log(softmax_rows(L.T)[np.arange(N), np.arange(N)]).mean()
    return 0.5 * (i2t + t2i)

def zero_shot_predict(image_emb, class_emb):   # (M, d), (K, d) -> (M,)
    return np.argmax(l2_normalize(image_emb) @ l2_normalize(class_emb).T, axis=1)
```

In practice use `open_clip` (OpenCLIP checkpoints, including LAION-trained ones), Hugging Face `transformers` (`CLIPModel`, `SiglipModel`) and the original OpenAI release. The training uses a learned `logit_scale = 1/tau` stored as a log parameter, mixed precision, and large-scale data pipelines; the loss itself is exactly the few lines above.

## Shapes

| object | shape |
|---|---|
| batch of images | $(N,3,H,W)$ |
| batch of tokenised captions | $(N,T_{\max})$ |
| image and text embeddings | $(N,d)$ each, unit norm after normalising |
| logit matrix | $(N,N)$ |
| class-prompt embeddings | $(K,d)$ (or $(K,M,d)$ before ensembling) |
| zero-shot scores for $M$ images | $(M,K)$ |

## Common misconceptions

- **"CLIP was trained on class labels."** It was trained on image-text pairs only; classes appear at test time as sentences.
- **"Zero-shot means no prompt design."** The prompt is the classifier; wording matters.
- **"Temperature only rescales."** In the loss it sets how sharply hard negatives are emphasised and changes the optimum; at inference argmax is unaffected.
- **"The two towers share weights."** They are separate networks joined only by the loss in the common space.
- **"A bigger batch is just faster."** It supplies more negatives and changes what is learned.
- **"CLIP embeddings tell you where things are."** One vector per image carries no location.
- **"Fine-tuning always helps."** It can erode the zero-shot ability on other categories.

## Key takeaways

- CLIP trains an image tower and a text tower so that matching pairs have high cosine similarity: a symmetric cross-entropy over the $N\times N$ matrix $UV^\top/\tau$.
- The gradient with respect to the logits is $\frac1{2N}[(P-E)+(Q-E)]$.
- Zero-shot classification = nearest class-prompt embedding; prompt ensembling averages templates then re-normalises.
- Retrieval, detection with text queries and multimodal models reuse the same space.
- Temperature, batch size, data quality and prompts are the practical levers; limits include bias, fine-grained detail and calibration.

## Explain it yourself

1. Starting from "find the right caption among $N$", derive the CLIP loss and its logit gradient, and explain why both directions are used.
2. You want a classifier for 30 types of industrial defect and have no labels. Describe the zero-shot recipe, how you would test the prompts, and when you would move to a linear probe.

## Further reading

- The two source lessons (CLIP, SigLIP, SigLIP 2, ALIGN, OpenCLIP scaling-law references).
- [[vis.05]] for the contrastive family, [[trf.04]] for the text encoder block, [[vis.02]] for probes and fine-tuning.

## Mastery criteria

You can implement the symmetric CLIP loss with a temperature and check its value at chance and at perfect alignment, derive its logit gradient, build a zero-shot classifier with prompt ensembling, compute recall@k for retrieval, and name the main limits of the approach.
