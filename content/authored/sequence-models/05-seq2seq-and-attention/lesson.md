---
id: seq.05
title: Sequence-to-sequence and the road to attention
summary: Encoder-decoder models, teacher forcing, greedy and beam decoding, the fixed-vector bottleneck, Bahdanau additive attention derived with its backward pass, and how it turns into scaled dot-product self-attention.
minutes: 110
teaches: [seq.seq2seq, seq.attention]
requires: [seq.rnn, seq.lstm, seq.gru, nn.backprop, prob.basics]
sources:
  - "math/cours-tal-03.pdf (CNAM RCP217 'Sequence vers sequence', PDF p.2-23, 36: Sutskever and Cho encoder-decoder architecture, encoder/decoder inputs and outputs, greedy decoding, the objective argmax P(y|x), chain rule, Markov/Viterbi remark, log-probabilities, beam search and its pseudocode). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/docs/en.md (encoder, decoder, teacher forcing and exposure bias, greedy decode, bottleneck demonstration, scheduled sampling). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/10-attention-mechanism/docs/en.md (Bahdanau additive score and shapes table, Luong variants, s_{t-1} vs s_t convention, worked weights, Q/K/V reading, masking, attention-is-not-explanation caveat). READ."
---

## Learning objectives

- Draw an encoder-decoder, say what each part receives and produces, and explain teacher forcing and exposure bias.
- Explain greedy decoding versus beam search as approximations of $\arg\max_yP(y\mid x)$.
- Explain the bottleneck of a fixed-size context vector.
- Derive additive (Bahdanau) attention with shapes, including the backward pass.
- Show how dot-product attention and self-attention generalise it.
- Implement the attention computation and a tiny attention seq2seq forward pass in NumPy.

## Prerequisites

Recurrent networks and gated cells ([[seq.03]], [[seq.04]]), cross-entropy and softmax ([[f2n.08]]), backprop with shapes ([[f2n.10]]).

## Where this fits

Translation, summarisation, question rewriting and speech recognition all map a sequence to a *different* sequence, whose length is not known in advance and need not equal the input length (the "rewriting" category of the CNAM course). The 2014 papers of Sutskever et al. and Cho et al. showed that two recurrent networks glued together can do this. The CNAM slides report Sutskever's system as a stack of 4 LSTM layers of 1000 cells each with about 380 million parameters, almost as good as the best (even non-neural) translation systems of the time.

This lesson is also the hinge of the whole course: the encoder-decoder's weakness (a single vector must carry the whole input) is exactly what **attention** removes, and attention, with the recurrence stripped away, is the transformer.

## The idea in plain words

An interpreter listens to a whole sentence in French, forms an understanding, and then speaks it in English word by word. Two roles: a **listener** (encoder) and a **speaker** (decoder). In the original design the listener hands the speaker one small card, the final memory state, and nothing else. The speaker must produce the entire translation from that single card. For a four-word sentence this works. For a forty-word sentence it fails, because all the detail must be squeezed onto a card of fixed size.

**Attention** changes the rule: the speaker keeps the *whole transcript* of the listener's notes (one note per source word). Before speaking each word, the speaker glances back, decides which notes matter *right now* (when saying "cat" look at "chat"; when saying "black" look at "noir", even though the order is swapped), mixes those notes into a fresh summary, and speaks. The "how much to look at each note" are the **attention weights**; they are also a soft alignment between the two languages.

## Encoder-decoder

Source $x_1,\dots,x_{T_s}$, target $y_1,\dots,y_{T_t}$ (with special tokens `<start>`/`<end>`).

- **Encoder**: an RNN (LSTM/GRU/tanh) over the embedded source, producing states $h_1,\dots,h_{T_s}$, $h_i\in\mathbb{R}^{H}$. The classic model keeps only $h_{T_s}$. It may be bidirectional because the whole source is known (slides, p.6).
- **Decoder**: an RNN whose first state is initialised with the encoder's last state. At each step it takes the previous target token (and previous state) and outputs a softmax over the target vocabulary plus a new state, fed to the next step (slides, p.3 and 6).

The model defines, by the chain rule of probability,

$$
P(y\mid x)=\prod_{k=1}^{T_t}P(y_k\mid x,\;y_{<k}).
$$

Each factor is one softmax output of the decoder. Note that $y_k$ depends on the *previously chosen* words: translating "I like potatoes" after having written "j'aime les pommes" is not the same as after "j'aime les patates" (slides, p.7).

### Training: teacher forcing

Loss: $-\log P(y\mid x)=-\sum_k\log P(y_k\mid x,y_{<k})$, the sum of the per-step cross-entropies ([[f2n.08]]), by ordinary BPTT through decoder **and** encoder. During training, the input at step $k$ is the **gold** previous token $y_{k-1}$, not the model's own prediction: this is *teacher forcing*. It stabilises learning (an early mistake does not corrupt all later inputs) and lets all steps' losses be computed from the gold sequence. The price is **exposure bias**: at inference the model must consume its own possibly-wrong outputs, a situation it never practised. Scheduled sampling (mix gold and predicted tokens, annealing toward predictions) is one mitigation (source lesson).

### Inference: greedy vs beam search

We want $y^\star=\arg\max_yP(y\mid x)$, an exponential search over $|V|^{T_t}$ sequences. The slides stress that Viterbi cannot be used because the network has no Markov assumption: each step depends on the whole prefix through its hidden state.

- **Greedy**: at each step take the $\arg\max$ and feed it back. Cheap; a locally good first word can lead to a poor sentence.
- **Beam search** with width $b$: keep the $b$ best partial hypotheses, extend each with the $b$ best next words, keep the $b$ best of the $b\cdot b$ candidates, repeat; hypotheses ending in `<end>` are finished. Scores are **sums of log-probabilities**, $\log P(y_{\le k}\mid x)=\log P(y_k\mid x,y_{<k})+\log P(y_{<k}\mid x)$ (multiplying probabilities underflows; note that one sums the log-*softmax*, not the raw logits). $b=1$ is greedy. It is a heuristic with no optimality guarantee.

## The bottleneck

With the final encoder state as the only link, the decoder's access to the source is a single vector of size $H$:

$$
c=h_{T_s}\in\mathbb{R}^{H},\qquad\text{the same for every output step.}
$$

Two problems follow. (1) **Capacity**: the information needed grows with the source length but $c$ does not. The source lesson shows a copy task in which accuracy collapses as the sequence grows, even though the information is available at every encoder step. (2) **Credit assignment**: gradients from late target words must travel back through the long chain of recurrent steps to reach early source words ([[seq.03]]). The slides note that reversing the source sentence can improve results depending on the language (p.4); a plausible reading, which I state as an interpretation, is that it shortens the path between the first source and first target words.

## Bahdanau attention, derived

Keep **all** encoder states $H=(h_1;\dots;h_{T_s})\in\mathbb{R}^{T_s\times H}$ and let the decoder form a *different* context $c_t$ at each step. At step $t$, with previous decoder state $s_{t-1}\in\mathbb{R}^{d_s}$ as the **query**:

1. **Score** each source position with a small feed-forward network (additive attention):

$$
e_{t,i}=v^\top\tanh\big(W_as_{t-1}+U_ah_i\big),\qquad W_a:(A,d_s),\;U_a:(A,H),\;v:(A).
$$

   $W_as_{t-1}+U_ah_i$ has shape $(A)$; the vector $v$ collapses it to a scalar. For all positions at once: $Z=s_{t-1}W_a^\top+HU_a^\top:(T_s,A)$, $u=\tanh Z$, $e=uv:(T_s)$.

2. **Normalise** into weights that sum to 1 (a softmax over the source positions):

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T_s}\exp(e_{t,j})}.
$$

3. **Context**: the weighted average of the encoder states (this is the "alignment" at work):

$$
c_t=\sum_{i=1}^{T_s}\alpha_{t,i}\,h_i\in\mathbb{R}^{H}.
$$

4. **Decode**: $s_t=\tanh(W_d[\,\mathrm{emb}(y_{t-1});c_t\,]+U_ds_{t-1}+b_d)$ and $\text{logits}_t=W_o[\,s_t;c_t\,]+b_o$.

Stacking the weights over all output steps gives the **alignment matrix** $\alpha\in\mathbb{R}^{T_t\times T_s}$, whose rows each sum to 1; plotted, it shows which source words each output word looked at.

**Convention warning.** Bahdanau scores with the *previous* decoder state $s_{t-1}$; Luong's variants use the *current* $s_t$. Mixing them gives subtly wrong results (source lesson). The scoring function can also be multiplicative: dot $s^\top h_i$ (needs $d_s=H$), or general $s^\top Wh_i$.

### Backward pass of the attention block

Let $\bar c$ be the gradient arriving at the context. Everything is an application of rules seen before.

- Context $c=\sum_i\alpha_ih_i$: $\ \bar\alpha_i=\bar c^\top h_i$ and the **value path** $\bar h_i\mathrel{+}=\alpha_i\bar c$.
- Softmax (Jacobian $\operatorname{diag}(\alpha)-\alpha\alpha^\top$): $\ \bar e_i=\alpha_i\big(\bar\alpha_i-\sum_j\alpha_j\bar\alpha_j\big)$.
- Score $e_i=v^\top u_i$: $\ \bar v=\sum_i\bar e_iu_i$ and $\bar u_i=\bar e_iv$.
- tanh: $\bar z_i=\bar u_i\odot(1-u_i^2)$ where $z_i=W_as+U_ah_i$.
- Linear parts: $\bar W_a=\big(\sum_i\bar z_i\big)s^\top$, $\bar s=W_a^\top\sum_i\bar z_i$, $\bar U_a=\sum_i\bar z_ih_i^\top$, and the **key path** $\bar h_i\mathrel{+}=U_a^\top\bar z_i$.

So each encoder state receives gradient from two places: directly through the weighted average (value path) and through the score (key path). Both are checked against finite differences in the first exercise. Note the gradient from every output step reaches every source position through $\alpha$ in one hop: no long recurrent chain is needed. That is why attention also repairs problem (2).

## Worked example

Three encoder states in $\mathbb{R}^2$: $H=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$ (think "chat", "noir", "le"), query $s=(0.5,-0.5)$, and for transparency $W_a=U_a=I$, $v=(1,1)$.

- $u_1=\tanh(1.5,-0.5)=(0.9051,-0.4621)$, $e_1=0.4430$
- $u_2=\tanh(0.5,0.5)=(0.4621,0.4621)$, $e_2=0.9242$
- $u_3=\tanh(1.5,0.5)=(0.9051,0.4621)$, $e_3=1.3672$
- $\alpha=\operatorname{softmax}(0.4430,0.9242,1.3672)=(0.1946,\;0.3149,\;0.4905)$
- $c=0.1946(1,0)+0.3149(0,1)+0.4905(1,1)=(0.6851,\;0.8054)$

The third state, whose key fits the query best, gets 49% of the weight. If the decoder state moves, the weights move with it, and so does the context.

## Implementation

```python
import numpy as np

def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def additive_attention(s, H, Wa, Ua, v, mask=None):
    u = np.tanh(s @ Wa.T + H @ Ua.T)      # (T, A)
    e = u @ v                             # (T,)
    if mask is not None:                  # True = real token
        e = np.where(mask, e, -1e9)       # padded positions get weight ~0
    alpha = softmax(e)                    # (T,)
    return alpha @ H, alpha               # (d_h,), (T,)
```

The second exercise wires this into a small seq2seq: embedding + tanh-RNN encoder, a decoder cell fed with `[embedding of previous token; context]`, teacher forcing with a cross-entropy loss (about $\ln V$ at random initialisation: 2.30 for $V=10$), the `use_attention=False` baseline whose context is always the last encoder state, and greedy decoding. Tests check that the alignment matrix has shape $(T_t,T_s)$ with unit row sums, that teacher forcing is causal (changing the last gold input does not alter earlier logits), and that teacher-forcing the greedy output reproduces the greedy arg-maxes.

### Masking and shapes

| Quantity | Shape |
|---|---|
| encoder states $H$ | $(T_s,H)$ (batched $(B,T_s,H)$) |
| query $s_{t-1}$ | $(d_s)$ |
| scores $e$, weights $\alpha_t$ | $(T_s)$ |
| context $c_t$ | $(H)$ |
| alignment matrix | $(T_t,T_s)$, rows sum to 1 |

In a batch of unequal lengths, padded source positions must be excluded: set their scores to a very negative number *before* the softmax so their weight is 0 and the rest still sum to 1 (tested in the exercise).

## From attention to self-attention

Rename the pieces: the decoder state is the **query** $q$, the encoder states are the **keys** $k_i$ (what we match against) and the **values** $v_i$ (what we return, here the same $h_i$). Attention is "softmax of query-key scores, then average the values". Replace the additive score by a dot product and run all queries at once:

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big)V,\qquad Q:(T_q,d),\;K:(T_k,d),\;V:(T_k,d_v).
$$

The factor $1/\sqrt d$: if the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q\cdot k=\sum_{j=1}^dq_jk_j$ has variance $d$, so scores would grow like $\sqrt d$ and saturate the softmax; dividing by $\sqrt d$ restores unit variance.

Two further steps give the transformer:

1. **Self-attention**: take $Q,K,V$ from the **same** sequence, via learned projections $XW_Q,XW_K,XW_V$. Every position now relates directly to every other one in a single step, regardless of distance (path length 1 instead of up to $T$ recurrent steps), and the computation over positions is parallel instead of sequential.
2. **Drop the recurrence.** Since attention alone is permutation-equivariant (the exercise tests this: shuffling the inputs shuffles the outputs), order must be injected separately by position information. Stacking such layers with residual connections ([[seq.01]]) and feed-forward layers is the transformer, covered in the next course.

## Common misconceptions

- **"Attention replaces the encoder."** The encoder still produces the states that attention reads.
- **"Attention weights explain the model."** They are intuitive but not a faithful explanation in general; the source lesson cites work showing weights can be altered without changing predictions on some tasks. Check with ablations before claiming reasoning.
- **"Teacher forcing is cheating at test time."** It is only a training technique; at test time the model uses its own outputs (greedy or beam).
- **"Beam search finds the best translation."** It is a heuristic; wider beams are not guaranteed to be better.
- **"The bottleneck is a training problem."** It is structural: more training does not enlarge a fixed-size vector.
- **"Softmax weights must be sharp."** They are a soft mixture; sharp or diffuse depends on the scores.

## Limits and trade-offs

- Attention costs $O(T_tT_s)$ scores per sequence; for self-attention $O(T^2)$ memory and compute, which limits context length.
- RNN decoders remain sequential at inference (one token at a time); attention parallelises only over source positions and, in training with teacher forcing, over the target if the decoder is not recurrent.
- Exposure bias and the train/inference mismatch persist in all autoregressive generators, including LLMs.
- Additive attention has extra parameters ($W_a,U_a,v$) but works with $d_s\neq H$; dot-product attention is cheaper and maps to fast matrix multiplies.

## Key takeaways

- Encoder-decoder: $P(y\mid x)=\prod_kP(y_k\mid x,y_{<k})$; train with teacher forcing, decode with greedy or beam search on summed log-probabilities.
- A single fixed vector is a capacity and gradient-path bottleneck.
- Bahdanau attention: $e_{t,i}=v^\top\tanh(W_as_{t-1}+U_ah_i)$, $\alpha_t=\operatorname{softmax}(e_t)$, $c_t=\sum_i\alpha_{t,i}h_i$.
- Backward: $\bar\alpha_i=\bar c^\top h_i$, $\bar e=\alpha\odot(\bar\alpha-\alpha^\top\bar\alpha)$, with value and key paths into $\bar h_i$.
- Query/key/value plus scaling by $\sqrt d$ gives scaled dot-product attention; same-sequence $Q,K,V$ is self-attention, the core of transformers.

## Explain it yourself

1. Why does the bottleneck hurt long sentences more than short ones, and how does attention change the gradient path from a late target word to an early source word?
2. Derive $\bar e_i=\alpha_i(\bar\alpha_i-\sum_j\alpha_j\bar\alpha_j)$ from the softmax Jacobian.

## Further reading

- CNAM RCP217 slides `cours-tal-03.pdf` (seq2seq, decoding, beam search).
- Source library: `ai-engineering-from-scratch` phase 05 lessons 09 (sequence-to-sequence) and 10 (attention mechanism), which cite Bahdanau et al. (2014) and Luong et al. (2015).
- Next course: transformers, where self-attention replaces recurrence.

## Mastery criteria

You can derive additive attention and its backward pass with shapes, explain teacher forcing, exposure bias and beam search, and your NumPy attention and tiny seq2seq pass the exercise tests. You can explain in a few sentences how scaled dot-product self-attention follows from Bahdanau attention.
