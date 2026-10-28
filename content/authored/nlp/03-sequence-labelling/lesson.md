---
id: nlp.03
title: Sequence labelling with HMMs and Viterbi
summary: Tag every word with a part of speech or an entity label using a hidden Markov model, derive the Viterbi dynamic program, implement it, and see why CRFs came next.
minutes: 80
teaches: [nlp.hmm-viterbi]
requires: [prob.basics, alg.exp-log]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/docs/en.md: BIO tagging, progression from rules to HMM to CRF to neural taggers. READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/docs/en.md: POS tags, most-frequent-tag baseline, constituency versus dependency parsing. READ."
  - "math/cours-tal-01.pdf (PDF p.12-26, French slides): Markov chains, hidden Markov models for POS tagging, naive search over tag sequences, Viterbi dynamic programming, use of logs and the smoothing problem. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 13.2 Hidden Markov Models (PDF p.630): the joint distribution of an HMM and the conditional-independence argument. Only the first page was read. READ."
---

## Where this fits

Classification ([[nlp.02]]) gives one label per document. Many language tasks need one label per **word**: which words are nouns or verbs (part-of-speech tagging, needed by the lemmatiser of [[nlp.01]]), which words belong to a person, place or organisation (named-entity recognition, NER, the first step of information extraction). The labels of neighbouring words depend on each other: after "the" a noun is likely, after "to" a verb. A model that labels words independently ignores this. The **hidden Markov model** (HMM) captures it with a simple probabilistic story, and the **Viterbi algorithm** finds the best label sequence among exponentially many in linear time. Neural taggers ([[seq.03]] on recurrent networks, and transformers) replace the HMM today, but BIO tags, dynamic-programming decoding and the CRF idea all survive in them.

## The idea in plain words

Imagine a person who writes sentences by walking through a small number of rooms, one per grammatical category: DET, NOUN, VERB. In each room they say one word, then walk to the next room, choosing the room according to habit: from DET one nearly always goes to NOUN. You only hear the words, not the rooms. Given the words "the fish bark", you want the most plausible walk. "Fish" could be said in the NOUN room or in the VERB room, so you cannot decide word by word; you must weigh the whole walk, where the transition habits help ("DET -> NOUN" is likely).

Checking every possible walk is hopeless for long sentences (3 rooms and 20 words already means over three billion walks). The trick behind Viterbi: if two partial walks arrive in the same room at the same time, only the better one can ever be part of the best full walk, so you keep one champion per room per step.

## Model

A first-order HMM has hidden states $t_1,\dots,t_n$ (the tags) and observed words $x_1,\dots,x_n$. Three tables of parameters, for $K$ tags and vocabulary size $V$:

- initial probabilities $\pi_k=P(t_1=k)$, shape $(K,)$;
- transition probabilities $A_{jk}=P(t_i=k\mid t_{i-1}=j)$, shape $(K,K)$, each row sums to 1;
- emission probabilities $B_{kw}=P(x_i=w\mid t_i=k)$, shape $(K,V)$, each row sums to 1.

Two assumptions: each tag depends only on the previous tag (Markov property), and each word depends only on its own tag. The joint probability of a tagging is therefore

$$
P(x,t)=\pi_{t_1}B_{t_1x_1}\prod_{i=2}^{n}A_{t_{i-1}t_i}B_{t_ix_i}.
$$

This is the factorisation given for the HMM in Bishop Sec. 13.2 and in the slides. Tagging means finding $\arg\max_t P(t\mid x)$; since $P(t\mid x)=P(x,t)/P(x)$ and $P(x)$ does not depend on $t$, we maximise the joint.

**Learning from tagged text.** With a labelled corpus, maximum likelihood is counting: $\pi_k$ = fraction of sentences starting with tag $k$; $A_{jk}=\frac{c(j\to k)}{c(j)}$; $B_{kw}=\frac{c(k,w)}{c(k)}$. As in [[nlp.02]], add a pseudo-count $\alpha$ so unseen events do not get probability zero: the slides flag exactly this smoothing issue, together with the problem of unknown words (a usual remedy is an `<UNK>` tag-conditional word or suffix features).

## Derivation: Viterbi

Naively there are $K^n$ tag sequences. Define

$$
\delta_i(k)=\max_{t_1,\dots,t_{i-1}}P(x_{1:i},\,t_{1:i-1},\,t_i=k),
$$

the probability of the best partial tagging of the first $i$ words that ends in tag $k$.

**Base case.** $\delta_1(k)=\pi_kB_{kx_1}$.

**Recursion.** A partial tagging ending in $k$ at position $i$ consists of a partial tagging ending in some $j$ at position $i-1$, then the factor $A_{jk}B_{kx_i}$. The new factor depends on the past only through $j$ (Markov property), so for a fixed $j$ the best choice of the earlier tags is the one that maximises $\delta_{i-1}(j)$. Therefore

$$
\delta_i(k)=\Big[\max_{j}\ \delta_{i-1}(j)\,A_{jk}\Big]\,B_{kx_i},\qquad \psi_i(k)=\arg\max_j\ \delta_{i-1}(j)A_{jk},
$$

where the back-pointer $\psi_i(k)$ records which previous tag won.

**Termination and backtracking.** The best final tag is $t_n^*=\arg\max_k\delta_n(k)$; then $t_{i-1}^*=\psi_i(t_i^*)$ for $i=n,\dots,2$.

**Cost.** Each of the $n\cdot K$ cells looks at $K$ predecessors: $O(nK^2)$ instead of $O(K^n)$.

**Logs.** Multiplying many probabilities underflows, so store $\log\delta$ and replace products by sums: $\log\delta_i(k)=\max_j[\log\delta_{i-1}(j)+\log A_{jk}]+\log B_{kx_i}$. A zero probability becomes $-\infty$, which is harmless for the maximum.

Replacing the max by a sum gives the *forward algorithm*, which computes $P(x)$; It is not needed here; we only need decoding. (When tags are not observed, HMMs are trained with EM, the Baum-Welch algorithm; the first page of Bishop Sec. 13.2 introduces the model, and we do not derive that here.)

## From POS tagging to NER: BIO tags

An entity may span several words, so the label set encodes boundaries: `B-X` begins an entity of type X, `I-X` continues it, `O` is outside any entity. "Maria Lopez visited New York" is `B-PER I-PER O B-LOC I-LOC`. NER is then ordinary sequence labelling with $K=2\cdot\#\text{types}+1$ tags. After decoding, a small function turns the tag sequence into spans. Some tag sequences are illegal (`O I-PER`, or `B-PER I-LOC`); a plain HMM can learn that these transitions are rare (low $A_{jk}$), and a stray `I-X` is usually read as the start of an entity.

## The CRF idea

An HMM is *generative*: it models $P(x,t)$ and each word depends only on its own tag, so you cannot easily use features such as "capitalised", "ends in -ing", "previous word is Mr." or the word to the right. A **conditional random field** (linear-chain CRF) models $P(t\mid x)$ directly:

$$
P(t\mid x)=\frac{1}{Z(x)}\exp\Big(\sum_{i}\big[\mathbf{w}\cdot f(t_{i-1},t_i,x,i)\big]\Big),
$$

with arbitrary feature functions $f$ and a normaliser $Z(x)$ computed by a forward recursion. Decoding is again Viterbi, with the scores $\mathbf w\cdot f$ in place of $\log A+\log B$. Neural taggers keep this skeleton: a BiLSTM or transformer produces per-word scores, a CRF layer adds the transition scores, and Viterbi decodes. Many modern transformer taggers drop the CRF and take an arg max per word, trading a little consistency for simplicity.

A baseline worth knowing: tag every word with its most frequent training tag. It is strong because most words are unambiguous; the difficulty lies in ambiguous and unknown words, which is exactly where context helps.

## Worked example

Tags $\{\mathrm{DET},\mathrm{NOUN},\mathrm{VERB}\}$, words $\{\text{the},\text{a},\text{dog},\text{fish},\text{bark},\text{swim}\}$ and

$$
\pi=(0.6,\,0.3,\,0.1),\qquad A=\begin{pmatrix}0.05&0.90&0.05\\0.10&0.30&0.60\\0.50&0.40&0.10\end{pmatrix}.
$$

Emissions: DET says *the* with 0.7 and *a* with 0.3; NOUN says *dog* 0.4, *fish* 0.3, *bark* 0.3; VERB says *fish* 0.3, *bark* 0.4, *swim* 0.3. Tag "the fish bark". Both *fish* and *bark* are ambiguous.

- Step 1: only DET can say *the*: $\delta_1=(0.42,\,0,\,0)$ since $0.6\times0.7=0.42$.
- Step 2 (*fish*): NOUN: $0.42\times0.9\times0.3=0.1134$; VERB: $0.42\times0.05\times0.3=0.0063$; DET cannot say *fish*.
- Step 3 (*bark*): NOUN: best predecessor is NOUN ($0.1134\times0.3=0.03402$ against $0.0063\times0.4=0.00252$), so $\delta=0.03402\times0.3=0.010206$. VERB: best predecessor NOUN, $0.1134\times0.6=0.06804$ against $0.0063\times0.1$, so $\delta=0.06804\times0.4=0.027216$.
- The maximum is VERB with $0.027216$; backtracking gives DET, NOUN, VERB.

Note how the last choice combines the emission (VERB says *bark* with 0.4, NOUN with 0.3) and the strong NOUN to VERB transition (0.6): what matters is the whole path, not any single factor. The exercise checks Viterbi against brute force enumeration of all $3^n$ paths on this model.

## Implementation

```python
import numpy as np

def viterbi(obs, log_pi, log_A, log_B):
    n, K = len(obs), len(log_pi)
    delta = np.full((n, K), -np.inf)       # best log-prob ending in tag k at step i
    back = np.zeros((n, K), dtype=int)     # back-pointers
    delta[0] = log_pi + log_B[:, obs[0]]
    for i in range(1, n):
        cand = delta[i - 1][:, None] + log_A          # (K_prev, K_next)
        back[i] = cand.argmax(axis=0)
        delta[i] = cand.max(axis=0) + log_B[:, obs[i]]
    path = [int(delta[-1].argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1], float(delta[-1].max())
```

## Shapes

- `obs`: $(n,)$ integer word ids. `log_pi`: $(K,)$. `log_A`: $(K,K)$, rows are the previous tag. `log_B`: $(K,V)$.
- `delta`, `back`: $(n,K)$. `cand`: $(K,K)$, built by broadcasting a column $(K,1)$ against $(K,K)$.

Libraries: NLTK and spaCy ship pretrained taggers; `sklearn-crfsuite` provides CRFs; transformer taggers via Hugging Face `token-classification`. Evaluate NER with entity-level (span) precision, recall and F1, not per-token accuracy: most tokens are `O`, so token accuracy is flattering ([[nlp.02]] on imbalance).

## Common misconceptions

- **"Viterbi picks the best tag for each word."** It picks the best *sequence*; the tag of one word may differ from its individually most likely tag.
- **"Viterbi and the forward algorithm are the same."** Viterbi takes a max and gives a path; forward takes a sum and gives $P(x)$.
- **"HMM states are the words."** The hidden states are tags; the words are the observations.
- **"A Markov assumption of order one is fine for language."** It is a strong simplification, adequate for tagging, which is why richer models use whole-sentence context.
- **"Token accuracy is a fine NER metric."** Use entity-level F1.

## Key takeaways

- An HMM factorises $P(x,t)$ into initial, transition and emission tables, learnable by counting plus smoothing.
- Viterbi is dynamic programming over $\delta_i(k)$ with back-pointers, cost $O(nK^2)$, in log space.
- BIO tags turn entity extraction into sequence labelling; decode, then convert to spans.
- CRFs model $P(t\mid x)$ with rich features; modern taggers keep the structure but learn the scores.

## Explain it yourself

1. Why can the best path not be found by choosing the best tag of each word independently? Use an ambiguous word in your answer.
2. Why is it enough to keep one champion path per tag per position?

## Further reading

- Jurafsky and Martin, *Speech and Language Processing*, chapters on POS tagging, HMMs and NER.
- Bishop, *Pattern Recognition and Machine Learning*, Sec. 13.2 (forward-backward, Viterbi, EM for HMMs).

## Mastery criteria

- You can write the three HMM tables, their shapes, and the joint probability of a tagging.
- You can fill a Viterbi table by hand and backtrack.
- You can convert BIO tags to entity spans and explain why span-level F1 is used.
