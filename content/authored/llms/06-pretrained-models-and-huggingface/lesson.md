---
id: llm.06
title: "Pretrained models in practice: hubs, tokenizers, pipelines and model cards"
summary: The pretrained-model ecosystem as a working engineer sees it - model hubs, auto classes, tokenizer plus model plus decoder pipelines, zero-shot and few-shot use, the fine-tuning workflow, model cards and responsible use - rebuilt offline with fakes.
minutes: 60
teaches: [tal.hub-ecosystem]
requires: [llm.tokenization, trf.architectures, nn.layer]
sources:
  - "cours-tal-04.pdf (RCP217, S. Rosmorduc, CNAM, French slides, PDF p.21 use of pre-trained BERT, p.28-29 GPT and ChatGPT, p.30-38 HuggingFace: libraries, AutoClasses, pipelines, zero-shot, question answering, classification by fine-tuning, model choice, few-shot, large language models and self-hosting, p.39-41 bibliography). READ."
  - "content/authored/llms/01-tokenization/lesson.md (what is already taught about tokenisation, to avoid duplication). READ."
---

## Learning objectives

- Explain what a model hub provides and what the three objects (tokenizer, model, configuration) are.
- Describe a pipeline as tokenizer, then model, then decoder, with the shape of each intermediate value.
- Distinguish direct use, zero-shot, few-shot and fine-tuning, and choose between them.
- List what a model card should contain and why each field matters.
- Build an offline pipeline with fakes and a model-card checker.

## Where this fits

By now you know how to train a language model from scratch ([[llm.03]]) and what the architectures are ([[trf.04]], [[trf.05]]). In practice almost nobody trains from scratch: they download a model somebody else paid to pre-train. The CNAM deck opens its section on this (cours-tal-04, p.30) by calling the HuggingFace transformers library one of several libraries that provide implementations of all modern networks together with pre-trained versions, usable from TensorFlow or PyTorch, easy to run as they are and fine-tunable. This lesson is the mental model for that ecosystem, so you can read any "from_pretrained" snippet and know what happens. Fine-tuning itself is [[llm.04]]; retrieval around such models is [[sys.03]].

## The idea in plain words

A model hub is a library with a catalogue. For each book (model) it keeps three things on the shelf:

1. the **recipe card** (configuration): how many layers, how wide, which family;
2. the **dictionary** (tokenizer): how text is cut into pieces and numbered, which must be exactly the one used when the model was taught;
3. the **brain** (weights).

Borrowing a model means fetching all three and keeping them together. If you swap the dictionary, the numbers mean different words and the output is nonsense without any error message.

A **pipeline** is a vending machine around these parts: you put in text, a tokenizer turns it into numbers, the model computes scores, and a decoder turns scores into something readable ("POSITIVE, 0.95"). A **model card** is the leaflet in the box: what it is for, what it was taught on, how well it did, and what it should not be used for.

## The pieces, as the deck presents them

**Auto classes (p.31).** The idea: the name or path of the model says which class to build. `AutoModel.from_pretrained('bert-base-cased')` creates a BERT model; `AutoConfig` loads a configuration and lets you override attributes (the slide enables attention outputs); `AutoTokenizer` loads the matching tokenizer, for example for a German BERT. Mechanically, the hub stores a configuration file with a model type; a registry maps that type to a Python class. You will rebuild this registry in the exercise.

**Pipelines (p.32).** Ready-to-use models for tasks: feature extraction, text classification, token classification, question answering, table question answering, fill-mask, summarisation, translation, text-to-text generation, text generation, zero-shot classification, conversational. The slide's example is a sentiment pipeline whose result is a label and a score, for instance `POSITIVE` with 0.9998.

**Choosing a model (p.36).** The library does not always find a suitable model by itself; you may have to name one, for example a French-to-English translation model published on the hub. The deck points to the hub's catalogue of models as a practical resource.

**Zero-shot (p.33).** Classify a document against a freely chosen list of labels without fine-tuning on them: the deck's example gives the labels "social event, computing, science, fun, sport, theater, art" and the sentence about a Java program that failed to compile. (How it works internally is not on the slide; one common implementation, general knowledge, poses each label as a hypothesis to a natural-language-inference model.)

**Question answering (p.34).** Extracts the answer from a given text: "where was I before London?" with the text "I went to Nice, Paris and then London" yields a quoted part of the document. The deck stresses it does not invent text.

**Classification by fine-tuning (p.35).** Take a model suited to the language (BERT for English), fine-tune it; there is a practical session for it. See [[trf.05]] for the head and the shapes.

**Few-shot (p.37).** Show a few examples to a language model in the prompt and it extrapolates; the deck demonstrates with French-to-English and French-to-Spanish translation examples.

**Large language models (p.38).** ChatGPT and its peers (Claude, Gemini, Mistral and others) offer new zero-shot and few-shot capabilities, new embeddings, APIs, retrieval-augmented generation, usually via a big remote server; self-hosting is possible, in particular with Ollama. Prompting techniques are in [[app.01]] and retrieval in [[sys.03]].

## The pipeline as three functions

Let a batch of $B$ strings be tokenised, padded to length $T$:

$$
\text{texts}\xrightarrow{\text{tokenizer}}(\text{ids},\text{mask})\in\mathbb{Z}^{B\times T}\times\{0,1\}^{B\times T}\xrightarrow{\text{model}}\text{logits}\in\mathbb{R}^{B\times K}\xrightarrow{\text{decoder}}B\text{ results}.
$$

Two details commonly cause bugs. **Padding**: shorter texts are filled with a pad id, and the *attention mask* tells the model to ignore those positions; forgetting the mask changes predictions depending on what else is in the batch. **Decoding**: for classification, softmax then argmax gives label and score; for generation, the decoder loop is sampling or beam search ([[llm.02]]) and then detokenisation.

### Worked example (reproduced by the exercise)

Vocabulary: `[PAD]=0, [UNK]=1, good=2, bad=3, movie=4, not=5`. A fake bag-of-words sentiment model sums per-token score vectors (negative, positive): good $=(-1,2)$, bad $=(2,-1)$, not $=(0.5,-0.5)$, movie $=(0,0)$.

- "good movie": logits $(-1,2)$, softmax positive probability $1/(1+e^{-3})=0.953$.
- "not bad": logits $(2.5,-1.5)$, probability of the winning label (NEGATIVE) $1/(1+e^{-4})=0.982$. The fake model is naive: it treats "not bad" as negative, a reminder that the quality of the weights matters and that the plumbing says nothing about it.
- Batch `["good movie","not good at all","bad"]` encodes to ids of shape $(3,4)$; row 3 is `[3,0,0,0]` with mask `[1,0,0,0]`.

## Ways of using a pretrained model

| Mode | Weights change? | Needs labels? | When |
|---|---|---|---|
| Direct (pipeline default) | no | no | the hub already has a model for your task and language |
| Zero-shot | no | no | labels change often or you have no data |
| Few-shot (prompting) | no | a handful of examples in the prompt | LLM available, quick prototype |
| Fine-tuning | yes | yes (hundreds to thousands) | stable task, need accuracy, latency or cost control |

The deck frames it so: pre-trained networks can be used "directly on the learning already done" or after fine-tuning (p.21). Whichever you choose, evaluate on your own held-out data ([[sys.04]]): a hub's headline number is measured on someone else's data.

## Model cards and responsible use

The deck lists a catalogue but says nothing on documentation; this section is general knowledge and should be read as engineering practice rather than a summary of the slides. A model card typically records:

- **license**: what you are allowed to do (commercial use, redistribution);
- **language(s)**: a French user should check that the model was trained on French;
- **intended use** and **limitations** (out-of-scope uses, known failure modes);
- **training data**: provenance, and the deck notes (p.28) that GPT-3's data are not freely accessible, so you cannot audit them;
- **metrics**: with the evaluation data named; values for a classifier are usually in $[0,1]$;
- **base model**, if it is a fine-tune of another.

Checklist for a downloaded model: read the license; check language and domain fit; run it on 50 of your own examples; look for known biases ([[saf.05]]); note the exact model identifier and revision so results are reproducible; and be careful with the weights file itself (some formats can execute code on load; prefer formats designed for plain tensors, general knowledge). For hosted LLM APIs add data-privacy review, since your text leaves your machine; self-hosting is the alternative named by the deck (p.38).

## Implementation

```python
import numpy as np
class Pipeline:
    def __init__(self, tokenizer, model, decoder):
        self.tokenizer, self.model, self.decoder = tokenizer, model, decoder
    def __call__(self, inputs):
        single = isinstance(inputs, str)
        ids, mask = self.tokenizer.batch_encode([inputs] if single else list(inputs))
        out = [self.decoder(row) for row in self.model(ids, mask)]
        return out[0] if single else out
```

In HuggingFace the same shape is `pipeline("sentiment-analysis")` (slide p.32); `from_pretrained` downloads and caches files, which the exercise replaces by plain Python objects so it runs offline.

## Common misconceptions

- **"The tokenizer is interchangeable."** It must match the model; ids from another vocabulary silently mean other words.
- **"A pipeline result is the truth."** The score is a softmax probability, not a calibrated confidence.
- **"Zero-shot means no work."** You still need to evaluate; label wording changes results.
- **"A high score on the model card transfers to my data."** Only if the data are alike.
- **"Open weights mean open data or open license."** They are separate; read each.

## Key takeaways

- A pretrained model is a bundle: configuration, tokenizer, weights; keep them matched.
- Auto classes map a config's model type to a class; pipelines chain tokenizer, model, decoder.
- Choose between direct use, zero-shot, few-shot and fine-tuning by data, cost and accuracy needs, then evaluate on your own data.
- Model cards record license, languages, intended use, limitations, training data and metrics; check them before use.

## Explain it yourself

1. A colleague swaps the tokenizer of a fine-tuned classifier for a similar-looking one and accuracy falls to chance with no error. Explain why.
2. You must tag customer emails in French into 8 categories; you have 300 labelled examples and a strict latency budget. Which mode of use do you pick and what do you check on the model card?

## Further reading

- The HuggingFace documentation for `transformers` (pipelines, auto classes) and model cards.
- Mitchell et al., "Model Cards for Model Reporting" (2019), general reading, not consulted for this lesson.
- Brown et al., "Language Models are Few-Shot Learners" (arXiv 2005.14165), listed in the deck's bibliography.

## Mastery criteria

- [ ] I can say what is in a model repository and why the tokenizer must match.
- [ ] I can write the tokenizer, model, decoder chain with shapes, including the attention mask.
- [ ] I can pick between direct use, zero-shot, few-shot and fine-tuning and justify it.
- [ ] I can review a model card and list what is missing.
