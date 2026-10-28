---
id: vis.07
title: "Image and text together: captioning and visual question answering"
summary: Encoder-decoder models that read an image with a CNN and write text with an RNN, soft attention over image regions (Show, Attend and Tell), temperature sampling, caption metrics such as BLEU-1, and visual question answering with its dataset-bias problem.
minutes: 75
teaches: [tal.captioning, tal.vqa]
requires: [seq.attention, seq.rnn, seq.cnn-arch, vis.transfer, info.entropy]
sources:
  - "cours-tal-05.pdf (RCP217, 'Vision et Texte', after a course by N. Thome, CNAM, French slides, PDF p.1-22, all pages read): CNN plus RNN encoder-decoder (p.2-3), RNN recap and task types (p.4-6), Show and Tell and the practical-session model (p.7-9), training and evaluation on Flickr8k (p.10-14), temperature sampling (p.12), Show, Attend and Tell with soft and hard attention (p.15-17), visual question answering and its biases (p.18-21), bibliography (p.22). READ."
---

## Learning objectives

- Draw the CNN-encoder, RNN-decoder captioning architecture and say what is fed to the decoder at each step.
- Write one decoding step with soft attention over $R$ image regions, with every shape.
- Explain temperature sampling and compute it on a distribution.
- Compute BLEU-1 with clipping and the brevity penalty, and say what it misses.
- Describe visual question answering (VQA) and why its datasets are biased.

## Where this fits

Earlier lessons gave you two encoders: CNNs that turn images into features ([[seq.01]], [[vis.02]]) and recurrent networks plus attention that turn sequences into sequences ([[seq.03]], [[seq.05]]). The CNAM deck "Vision et Texte" (cours-tal-05) glues them together: read the image with a CNN, write words with an RNN. That one idea gives image captioning, and, with a question as extra input, visual question answering. It is the ancestor of the vision-language models of [[vis.06]] and [[adv.01]], which replace both halves with transformers ([[trf.04]]), but the interface (image features in, tokens out, cross-entropy training) is unchanged.

## The idea in plain words

Imagine a tour guide who glances at a photo and then talks. The glance is the encoder: it squeezes the photo into a summary of "what is in it". The talking is the decoder: it says one word at a time, and each word depends on the words said so far and on the summary.

The deck stresses (p.3) that some tasks need more than naming objects: you must find the relations between objects and put them into natural language ("a bird flying over a body of water").

A single summary of the photo is a bottleneck, just like the single final state in translation ([[seq.05]]). The fix is the same: keep a grid of region summaries and let the guide look at different parts of the photo while talking about different words: when saying "bird" look at the bird, when saying "water" look at the lower half. That is the attention version, "Show, Attend and Tell".

Visual question answering turns the guide into a quiz master: you give the photo and a question ("what sport is this?") and the system must answer. The deck calls it very complex (p.19): it needs a precise model of the image and of the text, a model of how they interact, a global understanding of the scene, and some reasoning.

## Captioning with CNN plus RNN

The deck (p.2) states the recipe: a convolutional network for the vision side and a recurrent network for the text side. Page 7 describes "Show and Tell" (Vinyals et al., 2015): the image is encoded by deep features extracted from a CNN, the text is encoded one-hot over the vocabulary, and several architectural variants are possible. Pages 8 and 9 give the model used in the practical session:

- Image: deep features of a VGG network, reduced by PCA ([[cml.05]]) to dimension 100.
- Words: GloVe embeddings ([[seq.02]]) augmented with start and end tokens `<debut>` and `<fin>`; neither the GloVe embedding nor the image encoder is fine-tuned.
- At every step, the embedding of the previous word is concatenated with the image vector and fed to the RNN, whose output goes through a softmax over the vocabulary.

With a vocabulary of $V$ words, image vector $v\in\mathbb{R}^{100}$, word embedding $e_t\in\mathbb{R}^{E}$:

$$
h_t=\operatorname{RNN}\big([e_{t-1};v],\,h_{t-1}\big),\qquad p(w_t\mid w_{<t},\text{image})=\operatorname{softmax}(W_o h_t).
$$

Shapes: concatenated input $(E+100)$, hidden $(H)$, logits $(V)$.

### Training

As in language modelling and machine translation (p.10), the loss is cross-entropy on the next word, with teacher forcing, padding and masking to batch captions of unequal lengths ([[llm.02]] gives the link between this loss and perplexity). For a caption $w_1..w_T$:

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p(w_t\mid w_{<t},\text{image}).
$$

The practical session uses Flickr8k (p.11): 6,000 training images and 1,000 each for validation and test, with the vocabulary cut to 100 or 1,000 words to speed things up.

### Generation and temperature

At test time (p.11) the caption is produced by random sampling with a temperature; beam search is the suggested improvement. Temperature $t$ raises each probability to the power $1/t$ and renormalises:

$$
p'_i=\frac{p_i^{1/t}}{\sum_j p_j^{1/t}}.
$$

Equivalently $\operatorname{softmax}(\log p/t)$. Small $t$ sharpens toward the most likely word, large $t$ flattens. On $p=(0.3,0.5,0.1,0.1)$ (a distribution that sums to 1) I get, by running the code in the exercise: $t=0.25$ gives $(0.114,0.883,0.001,0.001)$ and $t=2$ gives $(0.290,0.375,0.168,0.168)$. The slide (p.12) prints $(0.11,0.887,0.001,0.001)$ and $(0.29,0.38,0.165,0.165)$, which matches closely, but its starting distribution is typed as $(0.3,0.5,0.01,0.01)$, which does not sum to 1 and does not reproduce the $t=2$ numbers; I read that as a typo for $0.1$. See [[llm.02]] for the same operation on language models.

## Attention over image regions

Pages 15 and 16 present Show, Attend and Tell (Xu et al., 2015), which uses spatial information. The CNN is not pooled into one vector; its last convolutional map gives $R$ region descriptors $a_i\in\mathbb{R}^{D}$ (for a $14\times14$ grid, $R=196$; that grid size is general knowledge, the deck does not give it). At step $t$:

1. The previous decoder state $h_{t-1}\in\mathbb{R}^H$ is compared to each region. The deck says the score $e_{t,i}$ is basically a dot product between $h_{t-1}$ and each $a_i$, possibly after simple transformations. With a matrix $W_{att}\in\mathbb{R}^{D\times H}$:
$$
e_{t,i}=a_i^\top W_{att}\,h_{t-1}.
$$
2. Weights: $\alpha_{t}=\operatorname{softmax}(e_t)\in\mathbb{R}^R$.
3. Context: $\hat z_t=\sum_i\alpha_{t,i}\,a_i\in\mathbb{R}^D$.
4. The context is combined with the previous word and state to give the new state, $h_t=\tanh(W_x e_{t-1}+W_z\hat z_t+W_h h_{t-1}+b)$ in our simplified cell (the paper uses an LSTM), and logits $W_o h_t\in\mathbb{R}^V$.

**Soft vs hard attention (p.15).** Soft attention is the weighted average above: differentiable, trained by ordinary backpropagation. Hard attention picks one region (a binary choice); it is not differentiable and the deck says it is trained with REINFORCE ([[rl.05]]). Page 17 shows the attention maps over an image for the caption "a bird flying over a body of water".

### Worked example (reproduced in the exercise)

Four regions with features $a_1=(1,0,0)$, $a_2=(0,1,0)$, $a_3=(0,0,1)$, $a_4=(1,1,0)$, $W_{att}=I_{3\times2}$ (identity padded) and $h_{t-1}=(2,1)$. Then $W_{att}h=(2,1,0)$ and the scores are $(2,1,0,3)$. Softmax gives $\alpha=(0.237,0.087,0.032,0.644)$ and the context $\hat z=(0.881,0.731,0.032)$. Region 4 dominates because it aligns best with the query. Shapes: $A:(R,D)=(4,3)$, $\alpha:(4,)$, $\hat z:(3,)$.

## Evaluating captions

The deck evaluates by showing generated captions (p.13-14) and gives no metric. So the following is general knowledge. Several valid captions exist for one image, so we compare to several human references with n-gram overlap.

**BLEU-1.** With candidate $c$ of length $|c|$ and references:

$$
\text{BLEU-1}=\mathrm{BP}\cdot\frac{\sum_{w}\min\big(\mathrm{count}_c(w),\,\max_r\mathrm{count}_r(w)\big)}{|c|},\qquad
\mathrm{BP}=\begin{cases}1&|c|>\rho\\ e^{1-\rho/|c|}&\text{otherwise}\end{cases}
$$

where $\rho$ is the length of the reference closest to $|c|$. Clipping stops a candidate from repeating a matching word. Example: candidate "a bird flies over water" against "a bird flying over a body of water" and "the bird is over the sea". Matches: a, bird, over, water give $4/5$; closest reference length is 6, so $\mathrm{BP}=e^{1-6/5}=0.819$ and BLEU-1 $=0.655$. Repeating "the" six times against "the cat sat" scores $1/6$ thanks to clipping.

BLEU-1 ignores word order, synonyms ("sea" vs "water") and correctness of relations; higher-order n-gram variants and consensus metrics such as CIDEr (tf-idf weighted n-gram similarity) correct some of this, and human judgement remains the reference. See [[nlp.06]] for BLEU and ROUGE in translation and summarisation.

## Visual question answering

Page 20 states the task: input is a question and an image, output an answer. A standard architecture (general knowledge; the deck only gives the task and a reference to Malinowski et al., "Ask Your Neurons", where an LSTM reads the question and CNN features of the image are fed to it) is:

- encode the image $v\in\mathbb{R}^{D}$ with a CNN, the question $q\in\mathbb{R}^{H}$ with an RNN;
- fuse them (concatenate, or elementwise product of projected vectors);
- classify over a fixed set of frequent answers, or generate the answer word by word.

Slide p.2 shows exactly this picture: a CNN on the image side and an RNN on the text side. Counting "types of RNN task" (p.5): captioning is one-to-many (one image, many words), VQA is many-to-one (question words and an image, one answer).

### Dataset bias

Page 21 warns that data are often biased, especially in VQA. For the question "what sport is it?" the answer "tennis" occurs 41% of the time. A model can score decently by ignoring the image. The deck lists the fixes: VQA 2.0 adds images so that each question has different answers on different images, and VQA-CP deliberately makes the answer priors differ between train and test. The exercise implements the *prior baseline*: always answer the most frequent training answer. If your model barely beats it, it is not looking at the image. [[saf.05]] discusses data bias more broadly.

## Implementation

```python
import numpy as np
def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attend(h_prev, A, W_att):            # A: (R, D), W_att: (D, H)
    alpha = softmax(A @ (W_att @ h_prev))   # (R,)
    return alpha, alpha @ A                 # context: (D,)
```

In practice you would use PyTorch (`nn.LSTM`/`nn.MultiheadAttention`) or a pre-trained vision-language model; evaluation uses the COCO caption toolkit for BLEU, CIDEr and others.

## Common misconceptions

- **"The CNN is trained with the captions."** In the deck's practical session the CNN features are frozen; only the decoder is trained. Fine-tuning the CNN is a separate choice.
- **"The attention map shows what the model understood."** It shows where it looked; faithfulness is not guaranteed (the deck says similar of transformer attention, p.15).
- **"High BLEU means a good caption."** BLEU rewards word overlap; a wrong but fluent caption can overlap a lot, and a correct paraphrase can score low.
- **"VQA accuracy proves visual reasoning."** With priors like tennis at 41%, a text-only model can look good; compare with the prior baseline.
- **"Sampling temperature changes the model."** It only reshapes the output distribution at generation time.

## Key takeaways

- Captioning is a one-to-many RNN task: CNN features in, one word out per step, trained with cross-entropy.
- Soft attention replaces a single summary by a weighted average of region features: scores, softmax, weighted sum.
- Hard attention is non-differentiable and needs REINFORCE.
- Temperature $t$: $p_i^{1/t}$ renormalised; below 1 sharpens, above 1 flattens.
- BLEU-1 is clipped unigram precision times a brevity penalty; it is a crude proxy.
- VQA data are biased; always compare to an answer-prior baseline.

## Explain it yourself

1. Why does attention over $R$ regions help for a long caption, and what is the shape of the context vector it produces?
2. Your VQA model reaches 45% accuracy and "most frequent answer" reaches 41%. What do you conclude and what do you test next?

## Further reading

- Vinyals et al., "Show and Tell" (arXiv 1411.4555); Xu et al., "Show, Attend and Tell" (arXiv 1502.03044); Malinowski et al., "Ask Your Neurons" (arXiv 1505.01121); Karpathy and Fei-Fei (arXiv 1412.2306): all listed in the deck's bibliography.

## Mastery criteria

- [ ] I can write one attention-augmented decoding step and give each shape.
- [ ] I can apply a temperature to a distribution by hand.
- [ ] I can compute BLEU-1 with clipping and brevity penalty.
- [ ] I can explain why VQA needs a language-prior baseline.
