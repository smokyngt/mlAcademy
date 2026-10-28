---
id: nlp.06
title: Translation, summarisation and question answering
summary: Understand how seq2seq systems translate, summarise and answer questions, and implement and critique the metrics used to score them (BLEU, ROUGE, exact match and token F1) plus extractive span selection.
minutes: 80
teaches: [nlp.seq-eval]
requires: [nlp.ngram-lm, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/11-machine-translation/docs/en.md: MT pipeline (tokeniser, encoder-decoder, beam search), BLEU and chrF evaluation. Opening sections read. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/12-text-summarization/docs/en.md: extractive (TextRank) versus abstractive summarisation, ROUGE-1/2/L. Opening sections read. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/13-question-answering/docs/en.md: extractive, open-domain/retrieval-augmented and generative QA, start/end span heads. Opening sections read. READ."
  - "math/cours-tal-03.pdf (PDF p.1-6, French slides): language-model view of translation, encoder-decoder architecture of Cho et al. and Sutskever et al. (2014). READ."
---

## Where this fits

Three of the most visible language applications share one structure: read some text, write other text. **Translation** maps a sentence in one language to another; **summarisation** compresses a document; **question answering** (QA) turns a question and some context into an answer. All three used to be separate research areas, and today they are handled by the same encoder-decoder or decoder-only transformers ([[llm.02]], [[trf.04]]). What remains different, and what this lesson focuses on, is how to *evaluate* them: there is no single correct output, so we score overlap with human references, and those scores have well-known blind spots. You need the n-grams of [[nlp.01]], the notion of precision/recall of [[nlp.02]], and the perplexity thinking of [[nlp.04]]. Retrieval for open-domain QA is the subject of [[sys.03]]; evaluating whole LLM systems continues in [[sys.04]].

## The idea in plain words

A **translator** reads the whole source sentence, forms an internal understanding, then writes the target sentence word by word, each time looking at what has already been written. Neural translation does exactly this with an *encoder* that reads and a *decoder* that writes. The first widely used version (2014) compressed the whole source into one vector and let a decoder unroll from it; attention ([[seq.05]]) later allowed the decoder to look back at every source word, which helped long sentences.

A **summary** can be built in two ways. An *extractive* summary is a highlighter: pick the most important sentences and copy them verbatim. It is always grammatical, never invents facts, but cannot rephrase or merge. An *abstractive* summary is a rewrite: the model writes new sentences. It can be fluent and compact, but may state something the document never said (a "hallucination").

**Extractive question answering** is a highlighter for a single answer: given a passage, mark where the answer starts and where it ends. **Generative** QA writes the answer, from memory or from retrieved passages.

And how do we grade? We cannot expect the machine to match a human word for word, so we count how much of its text overlaps with one or more human references. That is cheap and reproducible, but a perfectly good paraphrase scores low and a fluent wrong answer can score high.

## Seq2seq translation in one paragraph

The slides present translation as predicting the next item of a sequence, then as an encoder-decoder pair: an encoder network reads the source and its final state initialises a decoder network that emits the target token by token, starting from a start symbol and stopping at an end symbol. Training is the cross-entropy of [[nlp.04]]: the decoder is a conditional language model $P(y_t\mid y_{<t},x)$ over target tokens given the source $x$. At inference we search for a high-probability output: greedy decoding takes the arg max at each step, **beam search** keeps the $B$ best partial outputs (typically $B=4$ or $5$ for translation). Vocabularies are subwords ([[llm.01]]) so rare words are never unknown. Today's strong systems are multilingual transformer encoder-decoders; the sources mention models of the NLLB family, but their details are not needed here.

## BLEU

BLEU compares a candidate translation $c$ with one or more references. Define the **clipped n-gram precision**: count each candidate n-gram, but never more times than its maximum count in any single reference.

$$
p_n=\frac{\sum_{g\in n\text{-grams}(c)}\min\big(\mathrm{count}_c(g),\ \max_{r}\mathrm{count}_r(g)\big)}{\sum_{g\in n\text{-grams}(c)}\mathrm{count}_c(g)} .
$$

Clipping stops a degenerate candidate that repeats a common word ("the the the the the the the" would otherwise have precision 1). Precision alone rewards very short outputs, so multiply by a **brevity penalty** with $|c|$ the candidate length and $\rho$ the length of the reference closest in length:

$$
\mathrm{BP}=\begin{cases}1&|c|>\rho\\ e^{1-\rho/|c|}&|c|\le\rho\end{cases}
\qquad
\mathrm{BLEU}=\mathrm{BP}\cdot\exp\Big(\sum_{n=1}^{N}\tfrac1N\ln p_n\Big).
$$

The inside is the geometric mean of the precisions, usually with $N=4$. Properties: if any $p_n=0$ the sentence-level BLEU is $0$ (one reason BLEU is normally computed over a whole corpus, summing the counts before dividing, or smoothed at sentence level, for example by adding 1 to numerator and denominator for $n>1$). There is no recall term; the brevity penalty is its stand-in.

## ROUGE

ROUGE is the recall-oriented counterpart, standard for summaries. **ROUGE-$n$** counts the overlap of n-grams between candidate and reference:

$$
R_n=\frac{\text{overlap}}{\#\text{n-grams in reference}},\quad P_n=\frac{\text{overlap}}{\#\text{n-grams in candidate}},\quad F_n=\frac{2P_nR_n}{P_n+R_n},
$$

where overlap is the sum over n-grams of the minimum of the two counts (the same clipping). **ROUGE-L** uses the **longest common subsequence** (LCS): the longest sequence of tokens appearing in both texts in the same order, not necessarily adjacent. With $\ell=\mathrm{LCS}(c,r)$: $P=\ell/|c|$, $R=\ell/|r|$, and $F$ as above. The LCS is computed by the dynamic programme $D_{i+1,j+1}=D_{ij}+1$ if $c_i=r_j$, else $\max(D_{i,j+1},D_{i+1,j})$, in $O(|c||r|)$.

## Extractive QA: span prediction

Encode the question and the passage together with a transformer, so each passage token $j$ gets a vector $h_j$. Two small learned vectors $u_s$ and $u_e$ turn these into a **start score** $s_j=u_s\cdot h_j$ and an **end score** $e_j=u_e\cdot h_j$. Training uses the cross-entropy of a softmax over positions, once for the true start and once for the true end. At inference choose the span $(i,j)$ with $i\le j$ and $j-i+1\le L_{\max}$ maximising $s_i+e_j$; the length cap keeps the model from selecting huge spans. Models trained with unanswerable questions also compare the best span with a "no answer" score at a special position. QA is scored by **exact match** (EM) and **token F1** after normalisation (lowercase, strip punctuation and articles), where F1 treats prediction and gold as bags of tokens.

## Evaluation pitfalls

- **Paraphrase blindness.** `a feline rested on the rug` is a fair translation of `the cat sat on the mat`, yet shares almost no n-grams. Overlap metrics measure surface similarity, not meaning. Learned metrics (embedding or model-based scores) and human evaluation are better correlated with judgement at the system level; these are not covered here.
- **Reference dependence.** One reference penalises valid alternatives; BLEU is only comparable between systems scored on the same references, the same tokenisation and the same implementation (small implementation choices change the score by whole points).
- **Sentence-level noise.** BLEU is designed for corpora; per-sentence values are unstable and zero whenever a 4-gram is missing.
- **Length and copy effects.** ROUGE recall can be gamed by long outputs; extractive baselines such as "first three sentences" are strong on news, so always compare to them. For abstractive summaries, high ROUGE does not guarantee factual consistency; check faithfulness separately.
- **QA leaks.** Memorised answers in generative models, and lenient normalisation, inflate EM/F1.

## Worked example

Candidate $c$ = `the cat sat on the mat`, reference $r$ = `the cat is on the mat`, both of length 6.

| $n$ | candidate n-grams | matched (clipped) | $p_n$ |
|---|---|---|---|
| 1 | 6 | 5 (all but "sat") | 0.8333 |
| 2 | 5 | 3 (the cat, on the, the mat) | 0.6 |
| 3 | 4 | 1 (on the mat) | 0.25 |
| 4 | 3 | 0 | 0 |

The brevity penalty is 1 (equal length). BLEU-2 $=\sqrt{0.8333\cdot0.6}=0.7071$. BLEU-4 is $0$ because $p_4=0$; with add-one smoothing on $n\ge2$ the precisions become $0.8333,\ 0.6667,\ 0.4,\ 0.25$ and BLEU-4 $=0.4855$. The degenerate candidate `the the the the the the the` against the same reference has clipped unigram precision $2/7=0.2857$ (the reference contains "the" twice). A candidate `the cat sat` has $p_1=2/3$ but a brevity penalty $e^{1-6/3}=0.3679$, so BLEU-1 $=0.2453$. If a second reference `a cat sat on the mat` is added, the candidate matches it exactly and BLEU-2 becomes $1$.

ROUGE for the same pair: ROUGE-1 F1 $=0.8333$ (5 of 6 on each side), ROUGE-2 F1 $=0.6$, and the LCS is `the cat on the mat`, of length 5, so ROUGE-L F1 $=0.8333$. The paraphrase `a feline rested on the rug` gets BLEU-2 $=0.2582$ and ROUGE-1 F1 $=0.3333$ against the same reference although it means the same thing.

Span selection: tokens `the iphone launched on june 29 , 2007 .` with start scores peaking at `june` (3.0) and end scores peaking at `2007` (2.5) give the span `june 29 , 2007` (score 5.5) when $L_{\max}=5$ and only `june` when $L_{\max}=3$. QA scores: prediction `June 29` against gold `June 29, 2007` has EM $0$ and token F1 $=0.8$ ($P=1$, $R=2/3$). All of these are reproduced by the exercises.

## Implementation

```python
from collections import Counter
import numpy as np

def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

def clipped(c, refs, n):
    cc, mx = ngrams(c, n), Counter()
    for r in refs:
        for g, v in ngrams(r, n).items():
            mx[g] = max(mx[g], v)
    return sum(min(v, mx[g]) for g, v in cc.items()), sum(cc.values())

def lcs(a, b):
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            D[i + 1][j + 1] = D[i][j] + 1 if a[i] == b[j] else max(D[i][j + 1], D[i + 1][j])
    return D[-1][-1]
```

The complete BLEU and ROUGE functions are the exercise. In practice use `sacrebleu` (standardised tokenisation and signature, the reproducible choice), `rouge-score`, and the SQuAD evaluation script.

## Shapes

- Candidate and references: lists of tokens, lengths $|c|$ and $|r|$. LCS table: $(|c|+1,|r|+1)$.
- QA: `start_scores`, `end_scores` of shape $(T,)$ for $T$ passage tokens; the pair score matrix $s_i+e_j$ has shape $(T,T)$ and is masked to $i\le j<i+L_{\max}$.
- Decoder output: $(T_{\text{target}},|V|)$ logits, one distribution per generated token.

## Common misconceptions

- **"BLEU measures translation quality."** It measures n-gram overlap with references; high BLEU is neither necessary nor sufficient for a good translation.
- **"Compare BLEU numbers across papers."** Only valid with identical references, tokenisation and scoring tool.
- **"ROUGE rewards good summaries."** It rewards word overlap; faithfulness and coherence need separate checks.
- **"Abstractive is strictly better than extractive."** Extractive never hallucinates and is a strong baseline.
- **"EM is a fair QA metric by itself."** Equivalent answers with different wording fail EM; report F1 too.

## Key takeaways

- Translation, summarisation and generative QA are conditional language models trained with cross-entropy and decoded by greedy or beam search.
- BLEU = brevity penalty times the geometric mean of clipped n-gram precisions; ROUGE-$n$ and ROUGE-L are overlap F-scores (n-grams, LCS).
- Extractive QA picks the span maximising start score plus end score under a length cap; score with EM and token F1.
- Overlap metrics are cheap proxies: treat them with baselines, multiple references and human checks.

## Explain it yourself

1. Why is clipping needed in BLEU and why is there a brevity penalty but no recall term?
2. Give a pair of sentences with high meaning overlap but low BLEU, and explain what other evaluation you would add.

## Further reading

- Papineni et al., "BLEU: a method for automatic evaluation of machine translation" (2002); Lin, "ROUGE: a package for automatic evaluation of summaries" (2004); Post, "A call for clarity in reporting BLEU scores" (2018).
- Rajpurkar et al., SQuAD (2016) for span-based QA.

## Mastery criteria

- You can compute clipped precisions, brevity penalty and BLEU for a small pair by hand.
- You can compute ROUGE-1, ROUGE-2 and ROUGE-L, including an LCS.
- You can decode a best span under a length cap and compute EM/F1.
- You can name at least three ways overlap metrics mislead.
