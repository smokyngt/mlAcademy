---
id: trf.05
title: "BERT, BART, T5 and GPT: pre-training objectives and when to use each"
summary: Masked language modelling with the 80/10/10 rule, BERT fine-tuning heads (classification, tagging, span QA), denoising encoder-decoders (BART, T5 span corruption), how GPT differs, and a decision guide.
minutes: 80
teaches: [tal.mlm, tal.bert-finetune, tal.denoising-seq2seq]
requires: [trf.architectures, trf.block, nn.layer, info.entropy]
sources:
  - "cours-tal-04.pdf (RCP217, S. Rosmorduc, CNAM, French slides, PDF p.1-41, all pages read): motivation for transfer learning (p.2), recap of seq2seq and attention (p.3-5), Transformer (p.6-19), BERT objectives, use and head layouts (p.20-26), the other models XLNet/BART/T5 (p.27), GPT and ChatGPT (p.28-29), HuggingFace and LLM remarks (p.30-38), bibliography (p.39-41). READ."
  - "content/authored/transformers/04-transformer-block-and-architectures/lesson.md (what is already covered about the three families, to avoid duplication). READ."
---

## Learning objectives

- Explain why transformers made transfer learning work for text, and what BERT is pre-trained on.
- Apply the 80/10/10 masking rule and compute the masked-LM loss, averaged over selected positions only.
- State the shapes of the BERT outputs and how a classification head, a tagging head and a span-QA head read them.
- Describe denoising encoder-decoders (BART, T5 span corruption) and how GPT's causal objective differs.
- Choose between encoder-only, decoder-only and encoder-decoder for a given task.

## Where this fits

[[trf.04]] built the transformer block and listed the three families in a paragraph. This lesson is the "how are they actually trained and used" chapter. It follows the CNAM deck on BERT, BART and Transformers (cours-tal-04), whose storyline is: images got reusable pre-trained networks (the AlexNet era) and words got reusable embeddings, but recurrent networks transferred poorly to new tasks; from the Transformer onwards, big networks pre-trained on huge corpora transfer efficiently. The deck's pages 3 to 19 recap the encoder-decoder with attention ([[seq.05]]) and the Transformer ([[trf.01]] to [[trf.03]]), so we do not repeat them. Related lessons: [[llm.06]] shows how to load these models in practice, and [[nlp.02]] and [[nlp.03]] are the classical baselines they replace.

## The idea in plain words

Imagine teaching a child to read with games played on millions of books, no human labels needed.

- **Game 1, fill in the blanks (BERT).** Hide some words and ask the child to guess them using everything around the blank, left and right. To win, the child must learn grammar, facts and word senses. Afterwards you give a short course on one task (spam or not spam), and it learns quickly because the reading skills are already there.
- **Game 2, finish the sentence (GPT).** Show only the beginning and ask for the next word. The child can never peek ahead, which makes it a natural writer.
- **Game 3, repair the damaged page (BART, T5).** Tear out chunks of a page and ask the child to rewrite the original. A reader (encoder) looks at the damaged page and a writer (decoder) produces the repair.

All three games use unlabelled text, so the supply of exercises is practically unlimited. What changes between families is who may look at what, and what the model must write.

## Masked language modelling

BERT is, per the deck (p.20), the encoder half of a Transformer pre-trained on two tasks at once. The first is **masked language modelling (MLM)**: replace about 15% of the words by a mask symbol and ask the network to predict them at the output.

Let the input be token ids $x_1,\dots,x_T$ and let $S$ be the set of selected positions. The model reads the corrupted sequence and outputs logits $Z\in\mathbb{R}^{T\times V}$ ($V$ is the vocabulary size). The loss is

$$
\mathcal{L}_{\text{MLM}}=-\frac{1}{|S|}\sum_{i\in S}\log\operatorname{softmax}(Z_i)_{x_i}.
$$

Only selected positions contribute; the other $T-|S|$ positions have no label. A common slip is to average over all $T$ positions, which divides by the wrong number.

### The 80/10/10 rule

The deck says words are replaced by a mask token. The original BERT recipe (general knowledge, not on the slides) is slightly subtler. Among the selected positions:

- 80% become the mask token,
- 10% become a random token,
- 10% stay unchanged.

Why? The mask token never appears when the model is used on real text. If every selected token were masked, the network could learn "only work where I see the mask", and its representations of ordinary tokens would be neglected. Random replacements and unchanged tokens force it to build a good representation for every position, because any token might be wrong or might be a target. The loss is still computed at all selected positions, including the 10% left unchanged.

### The second task

The deck (p.20) describes a second simultaneous task: given a pair of sentences, say whether the second can follow the first, where examples are either two consecutive sentences of the corpus or two independent ones. This is usually called next-sentence prediction. The slides do not discuss how useful it is; later work reportedly found it adds little and several successors dropped it (general knowledge, treat as uncertain). The deck lists other models with different objectives (p.27): XLNet extends language modelling to all permutations of word order and, like BERT, is an encoder; BART couples an encoder with a decoder; T5 is a sequence-to-sequence network trained on very many tasks; Transformer-XL handles very long texts with relative positions.

## Using BERT: shapes of the outputs

BERT maps token ids $(B,T)$ to hidden states $H\in\mathbb{R}^{B\times T\times d}$. Two special tokens matter (general knowledge): a classification token `[CLS]` put first in the input, and a separator `[SEP]` between sentences. The deck (p.21) says a pre-trained network can be used directly or after fine-tuning, and pages 22 to 24 show three head layouts taken from the BERT paper:

| Task | Head reads | Head parameters | Output |
|---|---|---|---|
| Classification (p.23) | $H_{[:,0,:]}$, the `[CLS]` vector, shape $(B,d)$ | $W\in\mathbb{R}^{d\times K}$ | logits $(B,K)$ |
| Named-entity recognition (p.22) | every token vector, $(B,T,d)$ | $W\in\mathbb{R}^{d\times K}$, shared across positions | logits $(B,T,K)$, one tag per token |
| Question answering (p.24) | every passage token vector, $(B,T,d)$ | a start vector $s\in\mathbb{R}^d$ and an end vector $e\in\mathbb{R}^d$ | start scores $(B,T)$, end scores $(B,T)$ |

For QA the answer is a span of the passage: softmax the start scores and the end scores over positions and pick $(i,j)$ with $i\le j$ maximising the sum. This agrees with the HuggingFace remark (p.34) that extractive QA invents no text and quotes part of the document. A head adds very few parameters: with $d=768$ (a common base size) and $K=2$ classes, a classification head has $768\cdot 2+2=1538$ numbers, tiny next to the encoder.

**Fine-tuning** means: load pre-trained weights, add the head, train everything (or only the top) on a small labelled set with a small learning rate, so that the encoder is nudged rather than overwritten. The deck points to a practical session for this (p.35).

The deck also notes (p.21) that BERT, being an encoder, has not given very conclusive results in machine translation: it has no mechanism to write output text.

## What the attention heads do

Pages 25 and 26 show visualisations from Clark et al. (2019): some heads model simple relations, others syntactic relations. Treat this as a hint, not an explanation. The deck itself (p.15) says attention visualisation gives only a little information about what the model does.

## Denoising encoder-decoders: BART and T5

The deck only names BART and T5 in one line each (p.27). The details below are general knowledge and are labelled as such.

An encoder-decoder (see [[seq.05]], [[trf.04]]) reads a source with a bidirectional encoder and writes a target with a causal decoder that cross-attends to the encoder. To pre-train it without labels, damage the input and use the clean text as the target:

- **BART** corrupts text in several ways (masking, deleting, shuffling sentences, filling spans) and the decoder reconstructs the full original.
- **T5** uses **span corruption**: pick contiguous spans, replace each by one *sentinel* token in the input, and make the target the sentinels each followed by the removed span. Everything is cast as text-to-text, so translation, summarisation and classification all become "input string to output string" with a task prefix.

Worked example (the exercise reproduces it). Tokens $1..10$, spans (start 2, length 2) and (start 6, length 3), sentinel ids starting at 100:

- encoder input: `[1, 2, 100, 5, 6, 101, 10]`
- decoder target: `[100, 3, 4, 101, 7, 8, 9, 102]`

The input shrinks from 10 to 7 tokens and the target is short, which makes this cheaper than rewriting the whole text. The closing sentinel `102` tells the decoder where to stop.

## GPT: the causal family

The deck (p.28) presents GPT, GPT-2 and GPT-3 as models trained essentially as language models on ever larger data and sizes (GPT-3: 175 billion parameters, 300 billion training tokens), efficient at producing fairly coherent text. It describes them as "based on an encoder in input"; modern descriptions, and [[trf.04]], call GPT a decoder-only transformer: the same block, but with a causal mask so that position $i$ attends only to positions $\le i$. The objective is

$$
\mathcal{L}_{\text{LM}}=-\frac{1}{T}\sum_{i=1}^{T}\log p(x_i\mid x_{<i}).
$$

Every position yields a training signal (versus about 15% for MLM), but each sees only its left context. See [[llm.02]] for generation.

Page 29 describes how ChatGPT was obtained from a GPT-3.5 model: fine-tuning on question-answer data, human scores (1 to 10) on the model's answers, learning to predict those scores, then reinforcement learning with PPO to maximise predicted scores. [[llm.05]] and [[rl.06]] cover that pipeline in depth. Page 37 illustrates few-shot learning: put a few translation examples in the prompt and the language model continues the pattern, with no weight update.

## Comparison and choosing

| | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-decoder (BART, T5) |
|---|---|---|---|
| Attention | bidirectional | causal | bidirectional encoder, causal decoder plus cross-attention |
| Pre-training | masked tokens | next token | reconstruct damaged input |
| Output | a vector per token | next-token distribution | generated sequence |
| Natural uses | classification, tagging, extractive QA, embeddings | open-ended generation, chat, few-shot | translation, summarisation, rewriting |
| Weak at | generating text | seeing only the left context | two stacks to maintain |

Rules of thumb: fixed label set, labelled data and tight latency, try an encoder; free-form text or instructions with few examples, use a decoder (the deck, p.38, notes that large language models bring zero-shot and few-shot learning, new embeddings, APIs and retrieval-augmented generation); input and output are different sequences and you can fine-tune, use an encoder-decoder. A prompted LLM is often the most convenient and the most expensive and slowest: measure it against a fine-tuned small model ([[sys.04]]).

## Worked example: masking and loss

Tokens `[5,12,7,30,8,19,22,41]`, mask id 0, selection draws `[0.9,0.05,0.5,0.1,0.14,0.8,0.01,0.6]` with $p=0.15$ select positions 1, 3, 4 and 6. Action draws at those positions `0.3, 0.85, 0.95, 0.5` give: mask, random token (33), unchanged, mask. The corrupted input is `[5,0,7,33,8,19,0,41]` and the labels are `[-1,12,-1,30,8,-1,22,-1]`.

For the loss take two selected rows over a vocabulary of 4, logits `[2,0,0,0]` (label 0) and `[0,1,0,0]` (label 2). Row 1: $p=e^2/(e^2+3)=0.711$, loss $0.3408$. Row 2: $p=1/(e+3)=0.175$, loss $1.7437$. Mean $=1.0422$. A model guessing uniformly over $V$ tokens has loss $\log V$, a useful sanity check.

## Implementation

```python
import numpy as np
def mask_tokens(tokens, select_u, action_u, random_tokens, mask_id, p=0.15):
    inputs, labels = tokens.copy(), np.full(tokens.shape, -1)
    sel = select_u < p
    labels[sel] = tokens[sel]
    for i in np.where(sel)[0]:
        if action_u[i] < 0.8: inputs[i] = mask_id
        elif action_u[i] < 0.9: inputs[i] = random_tokens[i]
    return inputs, labels
```

Libraries (for example the HuggingFace masked-LM data collator) do this on batches and also exclude special tokens from selection. The exercise has the same logic with the random draws passed in, so tests are exact.

## Common misconceptions

- **"BERT predicts the next word."** It predicts hidden words using both sides; it is not a text generator.
- **"15% masked means 15% of inputs show the mask."** Only 80% of the selected positions show the mask; the rest are random or unchanged, yet all are in the loss.
- **"The `[CLS]` vector is a good sentence embedding out of the box."** It is trained to serve a fine-tuned head; without fine-tuning, pooled token vectors or a dedicated embedding model usually work better (general knowledge, check on your data).
- **"Encoder-decoder models are obsolete."** They remain strong for input-to-output rewriting; decoder-only models just cover more with one design.
- **"Next-sentence prediction is essential."** The deck lists it as one of two tasks; later practice often drops it.

## Key takeaways

- Self-supervised objectives turn raw text into endless training signal; the families differ in mask and target.
- MLM: select about 15%, apply 80/10/10, average cross-entropy over selected positions only.
- BERT heads: `[CLS]` vector for sentence labels, per-token vectors for tags, start and end scores for spans.
- Denoising encoder-decoders reconstruct damaged inputs; T5 span corruption uses sentinels.
- GPT is causal: every position gives a training signal and generation is its natural use.

## Explain it yourself

1. Why are 10% of the selected tokens left unchanged, and what would go wrong if all were replaced by the mask token?
2. A colleague wants to classify support tickets into 12 categories with 5,000 labelled examples. Which family would you try first, and what are the shapes of the input and of the output of its head?

## Further reading

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (arXiv 1810.04805), in the deck's bibliography.
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (arXiv 1910.10683).
- Brown et al. (GPT-3, arXiv 2005.14165); Ouyang et al. (instruction following, arXiv 2203.02155).

## Mastery criteria

- [ ] I can mask a token sequence by the 80/10/10 rule and compute the masked-LM loss by hand.
- [ ] I can give the shapes of the input and output of the classification, tagging and QA heads.
- [ ] I can write the input and target of a T5-style span-corrupted example.
- [ ] I can explain when to prefer an encoder, a decoder, or an encoder-decoder.
