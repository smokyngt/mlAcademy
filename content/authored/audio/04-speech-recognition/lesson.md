---
id: aud.04
title: "Automatic speech recognition: CTC, RNN-T, attention and Whisper"
summary: Solve the alignment problem between many audio frames and few characters: derive the CTC forward algorithm and implement it with greedy and prefix-beam decoding, then compare RNN-T and attention encoder-decoders (Whisper) and measure everything with word error rate.
minutes: 110
teaches: [aud.ctc, aud.asr, aud.wer]
requires: [aud.mel, aud.classify, prob.basics, info.entropy, nn.layer, trf.architectures, alg.summation]
sources:
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/04-speech-recognition-asr/docs/en.md (three formulations CTC / RNN-T / attention, conditional independence of CTC, greedy collapse, beam-search skeleton, WER dynamic programme, deployment table, pitfalls, exercises). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/05-whisper-architecture-finetuning/docs/en.md (30 s log-mel window, 3000 frames, conv stride 2, encoder/decoder sizes, special-token prompt, timestamps tokens, chunking, hallucination pitfalls). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/en.md (WER, CER, RTFx, first-token latency; normalisation before scoring). READ."
  - "math/coursAvanceApprentissageAudio.pdf, PDF p.15-19 (speech processing specifics, segmentation with CTC loss and greedy/beam decoding, transcription, Wav2Letter) and p.25-27 (autoregressive factorisation of waveform probability, WaveNet). READ."
---

## Where this fits

You say "turn on the kitchen lights" and a device turns text into action. Behind it sits automatic speech recognition (ASR): ten seconds of audio, about a thousand feature frames from [[aud.02]], must become a short string of about thirty characters. Classification ([[aud.03]]) gave one label per clip; here the output is a *sequence* whose length is unknown and whose parts are not marked in time. This lesson derives the loss that solved that problem (CTC), shows its two successors (the transducer RNN-T and attention encoder-decoders such as Whisper), and builds the metric (word error rate) used to compare them. It uses the self-attention and encoder-decoder ideas of the transformer lessons (concepts `trf.attention` and `trf.architectures`) only in the last part.

## The idea in plain words

Imagine you are given a long strip of film and a one-line caption, and asked to say which frames show which words. Nobody marked it. The word "okay" can last 200 ms or 1200 ms; there are silences, hesitations and breaths in between. If you had to train a network, a naive plan would be to have people label every frame with its letter, which is hugely expensive.

**CTC** (Connectionist Temporal Classification) avoids the labelling. The network looks at each frame and writes down a letter or a special "nothing here" mark (the *blank*). Reading the strip back, you merge neighbouring equal letters and erase the blanks. Many different frame-by-frame scribbles collapse to the same caption, for instance `h h _ e e l _ l o` and `_ h e _ l l _ l o o` both read "hello". CTC trains the network to maximise the **total probability of all scribbles that collapse to the right caption**, without ever choosing one of them. The blank is not decoration: without it, "hello" with its double "l" could not be written, because two equal letters in a row would merge.

**RNN-T** keeps this idea but adds a small language model inside: after writing a letter the network knows what it has already written and decides what comes next. **Attention encoder-decoders** (Whisper) drop the frame-by-frame view altogether: an encoder digests the audio, and a decoder writes the text word piece by word piece, each time *looking back* at the whole audio with attention.

## CTC: the model

Let the acoustic encoder produce, for each of $T$ frames, a probability distribution over $V+1$ symbols: $V$ real tokens (characters or word pieces) plus the blank $\varnothing$ (index 0 in code). Write $p_t(v)$ for the probability of symbol $v$ at frame $t$, rows of a matrix $\mathbf P\in\mathbb R^{T\times(V+1)}$ (softmax of the network outputs, so each row sums to 1).

An **alignment** (path) is a sequence $\pi=(\pi_1,\dots,\pi_T)$ of symbols, one per frame. Assuming the frames are conditionally independent given the audio (CTC's central assumption), its probability is

$$
P(\pi\mid x)=\prod_{t=1}^{T}p_t(\pi_t).
$$

The **collapse map** $\mathcal B$ first merges consecutive repeated symbols and then deletes blanks: $\mathcal B(a\,a\,\varnothing\,a\,b\,b\,\varnothing\,c)=(a\,a\,b\,c)$. The probability of a label sequence $y$ (the transcript) is the sum over all paths that collapse to it:

$$
P(y\mid x)=\sum_{\pi\in\mathcal B^{-1}(y)}\ \prod_{t=1}^{T}p_t(\pi_t),\qquad
\mathcal L_{\text{CTC}}=-\log P(y\mid x).
$$

There are exponentially many paths (up to $(V+1)^T$), so we need a recursion.

## Derivation: the forward algorithm

**Extended label sequence.** Insert a blank before, between and after the $U$ target tokens $y_1,\dots,y_U$:

$$
l'=(\varnothing,\,y_1,\,\varnothing,\,y_2,\,\dots,\,\varnothing,\,y_U,\,\varnothing),\qquad |l'|=S=2U+1 .
$$

Number positions $s=1,\dots,S$; odd $s$ are blanks, even $s$ are labels. Every valid path walks through $l'$ from left to right: at each frame it stays at the current position or advances, and its symbol at frame $t$ is $l'_s$ for its current position $s$.

**Definition.** $\alpha_t(s)$ is the total probability of all length-$t$ path prefixes that (i) collapse to the first $\lfloor s/2\rfloor$ target tokens and (ii) are at position $s$ (symbol $l'_s$) at frame $t$.

**Initialisation.** A path must start in the first blank or in the first label:

$$
\alpha_1(1)=p_1(\varnothing),\qquad \alpha_1(2)=p_1(y_1),\qquad \alpha_1(s)=0\ \ (s>2).
$$

**Recursion.** Where could a path at position $s$ at frame $t$ have been at frame $t-1$?

1. At $s$ itself (the same symbol repeated: repeats merge).
2. At $s-1$ (advance by one position: from a blank to the next label, or from a label to the next blank).
3. At $s-2$ (jump over a blank): allowed **only if** $l'_s$ is a label (not a blank) and $l'_s\ne l'_{s-2}$. If the two labels were equal ($y_u=y_{u+1}$, e.g. the double "l"), skipping the blank would merge them into one; a blank between them is mandatory. Jumping into a blank over a label would erase that label.

Each case multiplies by the probability of the new symbol at frame $t$:

$$
\alpha_t(s)=p_t(l'_s)\Big[\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\mathbb 1\{l'_s\neq\varnothing,\ l'_s\ne l'_{s-2}\}\;\alpha_{t-1}(s-2)\Big]
$$

(with $\alpha_{t-1}(0)=\alpha_{t-1}(-1)=0$).

**Termination.** A complete path ends in the last label or in the final blank:

$$
P(y\mid x)=\alpha_T(S)+\alpha_T(S-1).
$$

Cost: $O(T\cdot U)$ instead of $O((V+1)^T)$. If $T<U+r$ where $r$ is the number of adjacent equal target pairs (each needs a blank between), no path exists and the probability is exactly 0 (the loss is infinite). For example, the target $(1,1,1)$ needs at least $T=5$ frames: `1 _ 1 _ 1`.

**Numerical stability.** The products underflow for long utterances, so implementations work in log space: replace sums by `logaddexp` and products by sums, $\log\alpha_t(s)=\log p_t(l'_s)+\operatorname{logaddexp}(\dots)$.

**Gradients.** A symmetric backward recursion $\beta_t(s)$ gives the posterior probability that frame $t$ is at position $s$, $\gamma_t(s)=\alpha_t(s)\beta_t(s)/P(y\mid x)$, and the gradient of the loss with respect to the logits is the network output minus the posterior mass of each symbol. If you implement the forward pass with differentiable operations an autograd system produces the same gradient. Production code (`torch.nn.CTCLoss`, warp-ctc) fuses all this in a kernel.

## Worked example 1: three frames, one letter

Vocabulary: blank and one letter $a$. Every frame has $p(\varnothing)=0.6$, $p(a)=0.4$. Target $y=(a)$, so $l'=(\varnothing,a,\varnothing)$, $S=3$, $T=3$.

| $t$ | $\alpha_t(1)$ ($\varnothing$) | $\alpha_t(2)$ ($a$) | $\alpha_t(3)$ ($\varnothing$) |
|---|---|---|---|
| 1 | 0.6 | 0.4 | 0 |
| 2 | 0.36 | 0.40 | 0.24 |
| 3 | 0.216 | 0.304 | 0.384 |

Check one cell: $\alpha_2(2)=p(a)[\alpha_1(2)+\alpha_1(1)]=0.4\,(0.4+0.6)=0.4$. Termination: $P(a)=0.384+0.304=0.688$.

All three possible transcripts: $P(\text{""})=0.6^3=0.216$ (all blanks), $P(a)=0.688$, $P(aa)=0.4\cdot0.6\cdot0.4=0.096$ (only the path $a\,\varnothing\,a$). They sum to $1$.

## Decoding

**Greedy (best path).** Take the most probable symbol at each frame, then collapse. For the example above the best symbol per frame is the blank, so greedy decoding returns the **empty string**, although the transcript $a$ has probability $0.688$ versus $0.216$ for the empty one. The best *path* is not the best *transcript*, because the transcript's probability is a sum over many paths. Greedy is fast and often good enough when the network is confident.

**Prefix beam search.** Keep the $B$ best *prefixes* (partial transcripts), not paths. For each prefix store two log-probabilities: $p_b$ (alignments ending in a blank) and $p_{nb}$ (ending in a label). At each frame, for each prefix $g$ with total $p=p_b+p_{nb}$:

- blank: $g$ stays, adds $p\cdot p_t(\varnothing)$ to its $p_b$;
- label $c$ equal to the last label of $g$: either stay at $g$ (no blank in between, merged) adding $p_{nb}\,p_t(c)$ to $g$'s $p_{nb}$, or extend to $g{+}c$ (a blank separated them) adding $p_b\,p_t(c)$ to the new prefix's $p_{nb}$;
- any other label $c$: extend to $g{+}c$, adding $p\,p_t(c)$ to its $p_{nb}$.

Merging alignments that reach the same prefix is what the "naive" beam search in the repo lesson misses (its skeleton does not merge, as its own exercise 2 acknowledges). With a beam wide enough to keep everything, prefix beam search is **exact**: our test compares it with brute force. An external language model can be added to the score ("shallow fusion").

Because each frame is predicted independently, CTC has no internal language model: it may write phonetically plausible but ungrammatical text. That is why practical systems combine it with an external LM or choose RNN-T or attention.

## RNN-T: the idea

The transducer has three parts: an **encoder** over audio (frames $f_t$), a **prediction network** over the labels written so far (state $g_u$ after $u$ labels, effectively a language model), and a **joiner** that combines $f_t$ and $g_u$ into a distribution over $V+1$ symbols. Picture a grid with time $t=0..T$ along one axis and labels $u=0..U$ along the other. At a node the joiner either emits label $y_{u+1}$ (move $u\to u+1$ without consuming a frame) or emits a blank (move to the next frame). A transcript's probability is the sum over all monotone paths from $(0,0)$ to $(T,U)$ (their number is $\binom{T+U}{U}$ ending with a blank), computed by the same kind of forward recursion as CTC. Differences: (i) the next label depends on previous labels, fixing CTC's independence assumption; (ii) several labels per frame are possible; (iii) the loss lattice has $T\times(U+1)\times(V+1)$ entries per utterance instead of $T\times(V+1)$.

**Worked example 2: why RNN-T training is memory-hungry.** Take $T=1500$ frames, $U=100$ tokens, $V+1=1000$ symbols, 4-byte floats: the transducer lattice has $1500\cdot101\cdot1000=151.5\times10^6$ values (about 0.6 GB per utterance, before gradients), the CTC matrix $1.5\times10^6$ values (6 MB). These numbers are our illustration; actual sizes depend on the encoder frame rate and vocabulary. RNN-T remains the standard for streaming on-device recognition because it emits tokens as audio arrives.

## Attention encoder-decoders and Whisper

An attention encoder-decoder models the transcript autoregressively,

$$
P(y\mid x)=\prod_{u=1}^{U}P\big(y_u\mid y_{<u},\ \mathrm{Enc}(x)\big),
$$

(the same chain-rule factorisation the French course writes for waveform synthesis in its WaveNet section). The encoder turns the log-mel frames into hidden vectors `(T', d)`; the decoder is a causal transformer that, at every step, **cross-attends** to all encoder vectors (shapes as in the transformer lessons: queries from the decoder `(u, d)`, keys and values from the encoder `(T', d)`). No alignment is imposed: attention can look anywhere. Training is plain cross-entropy with teacher forcing. Costs: output tokens are produced one at a time, and the model needs the whole input, so it is not naturally streaming.

**Whisper** (as described in the repo lesson) is such a model:

- Input: 30 s of audio as an 80-channel log-mel spectrogram with 10 ms hop: $30/0.01=3000$ frames `(80, 3000)`. Shorter clips are padded, longer ones are chunked.
- Encoder: a convolutional stem with stride 2 halves the length to $3000/2=1500$ positions (one per 20 ms), then transformer blocks. Self-attention cost per head per layer is $1500^2=2.25\times10^6$ scores; for a 32-layer, 20-head encoder (the large configuration quoted in the repo lesson) that is $1.44\times10^9$ attention scores per 30-second window.
- Decoder: a transformer decoder over a vocabulary of about 51 865 byte-pair tokens.
- **Multitask by prompt:** special tokens such as `<|startoftranscript|>`, a language tag, `<|transcribe|>` or `<|translate|>`, and `<|notimestamps|>` steer one model into several tasks; timestamp tokens mark 20 ms offsets. Changing the language token changes the language.
- Log-mel statistics must be Whisper's own (not librosa defaults), or the output degrades.

Known failure modes in the repo lesson: hallucinated text ("thanks for watching") on silence, which is why a voice-activity detector gates the audio; runaway repetition when each 30 s window conditions on the previous text; and padded short clips. Reported 2026 benchmark numbers (about 1.4-1.8% WER on LibriSpeech test-clean for the best systems) come from the repo lesson and were not independently checked here; the ranking among top systems is within a few tenths of a percent, and deployment properties (streaming, latency, language coverage) matter more.

| Formulation | Alignment | Internal LM | Streaming | Typical use |
|---|---|---|---|---|
| CTC | marginalised over paths | no | yes | fast, simple, wav2vec-style models |
| RNN-T | marginalised over lattice | yes (prediction net) | yes | on-device, low latency |
| Attention enc-dec | learned by attention | yes (decoder) | not natively | offline quality (Whisper) |

## Word error rate and edit distance

Compare a reference $r_1\dots r_N$ with a hypothesis $h_1\dots h_M$. The **edit distance** (Levenshtein) is the minimum number of substitutions (S), deletions (D) and insertions (I) turning one into the other. Define $D[i][j]$ as the distance between the first $i$ reference words and the first $j$ hypothesis words:

$$
D[i][0]=i,\qquad D[0][j]=j,
$$

$$
D[i][j]=\min\Big(D[i-1][j]+1,\ D[i][j-1]+1,\ D[i-1][j-1]+\mathbb 1\{r_i\ne h_j\}\Big).
$$

The three terms are: delete $r_i$; insert $h_j$; match or substitute. Then

$$
\text{WER}=\frac{S+D+I}{N}=\frac{D[N][M]}{N}.
$$

Walking back through the table from $(N,M)$ recovers an alignment and the counts $S,D,I$ (equally cheap alignments can split the total differently; the total is unique). WER is **not bounded by 1**: a hypothesis with many insertions can exceed 100%. The **character error rate** (CER) applies the same formula to characters and is preferred for languages without word boundaries. Always normalise before scoring (lower-case, strip punctuation, unify numbers), otherwise formatting differences count as errors.

**Worked example 3.** Reference `the cat sat on mat`, hypothesis `the cat on a mat`. The table (rows: reference prefixes, columns: hypothesis prefixes, starting with the empty prefix) ends with $D[5][5]=2$:

$$
\begin{matrix}
 & \varnothing & \text{the} & \text{cat} & \text{on} & \text{a} & \text{mat}\\
\varnothing & 0&1&2&3&4&5\\
\text{the}&1&0&1&2&3&4\\
\text{cat}&2&1&0&1&2&3\\
\text{sat}&3&2&1&1&2&3\\
\text{on}&4&3&2&1&2&3\\
\text{mat}&5&4&3&2&2&2
\end{matrix}
$$

So WER $=2/5=0.4$ (delete "sat", insert "a"). A second example: reference `please turn on the kitchen lights`, hypothesis `please turn the kitchen light on now` gives distance 4, so WER $=4/6=0.667$ (one deletion, one substitution, two insertions). Note a correction to the repo lesson: it annotates the score for `Please turn on the lights.` against `please turn on the light` as "~0.17", but after normalisation there are 5 reference words and 1 error, i.e. **0.2**.

WER weights all words equally (a dropped "not" counts like a dropped "the") and says nothing about meaning; for production, also track latency (RTFx, first-token latency) and per-domain error.

## Implementation

```python
import numpy as np
from collections import defaultdict

NEG = -np.inf

def collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out

def ctc_forward(logp, target, blank=0):
    """logp: (T, V) log-probabilities (rows normalised). Returns log P(target | x)."""
    T = logp.shape[0]
    ext = [blank]
    for c in target:
        ext += [c, blank]                       # length S = 2U + 1
    S = len(ext)
    alpha = np.full((T, S), NEG)
    alpha[0, 0] = logp[0, blank]
    if S > 1:
        alpha[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [alpha[t - 1, s]]
            if s >= 1:
                terms.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[t - 1, s - 2])
            alpha[t, s] = np.logaddexp.reduce(terms) + logp[t, ext[s]]
    end = alpha[T - 1, S - 1]
    if S > 1:
        end = np.logaddexp(end, alpha[T - 1, S - 2])
    return end

def ctc_greedy(logp, blank=0):
    return collapse(logp.argmax(axis=1), blank)

def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h)[len(r), len(h)] / max(1, len(r))
```

The exercises add `ctc_prefix_beam_search`, the S/D/I back-trace and tests against brute-force enumeration.

### Shapes

| Object | Shape |
|---|---|
| encoder output / `logp` | `(T, V+1)` (CTC: log-softmax over symbols) |
| extended labels $l'$ and $\alpha$ | $S=2U+1$; `alpha` is `(T, S)` |
| transducer lattice | `(T, U+1, V+1)` |
| Whisper input / encoder output | `(80, 3000)` / `(1500, d)` |
| edit-distance table | `(N+1, M+1)` |

**What libraries do.** `torch.nn.CTCLoss` and `torchaudio.functional.rnnt_loss` implement the losses with fused kernels; `pyctcdecode`/`torchaudio` offer prefix beam search with KenLM fusion; `jiwer` computes WER with normalisation transforms; `openai-whisper`, `faster-whisper` and WhisperX provide inference, runtime acceleration and long-form handling (VAD gating, word alignment).

## Common misconceptions

- **"CTC picks the best alignment."** It sums over all alignments for training; the greedy decoder picks the best single path, which may differ from the best transcript (worked example 1).
- **"The blank means silence."** It means "no new label emitted at this frame"; networks often put blanks on most frames, even during speech.
- **"CTC output is length-preserving."** The path has $T$ symbols; the transcript is shorter after collapsing.
- **"Lower WER means better transcripts for every purpose."** WER ignores meaning, punctuation and speaker; compare on your domain, with identical normalisation.
- **"Whisper hears 30-second chunks natively, so streaming is easy."** The window is fixed; streaming needs extra engineering or a different architecture.
- **"Prefix beam search = ordinary beam search over frames."** Without merging paths that collapse to the same prefix, the beam fills with duplicates.

## Key takeaways

- CTC marginalises over blank-augmented alignments: $P(y|x)=\sum_{\pi\in\mathcal B^{-1}(y)}\prod_tp_t(\pi_t)$, computed by the $O(TU)$ forward recursion on $l'$ with the skip rule $l'_s\ne\varnothing,\ l'_s\ne l'_{s-2}$.
- Best path $\ne$ best transcript: prefix beam search (exact with a wide beam) tracks blank and non-blank probabilities per prefix.
- RNN-T adds a label-history network (a built-in LM) at the cost of a 3-D lattice; attention encoder-decoders drop alignment constraints and decode autoregressively (Whisper: 30 s, 3000 frames, 1500 encoder positions, task tokens).
- WER $=D[N][M]/N$ from the Levenshtein recursion; it can exceed 1; normalise first.

## Explain it yourself

1. Why must a blank be inserted between two equal letters in the extended sequence, and how does this appear in the skip rule of the forward recursion?
2. Describe a situation where greedy CTC decoding and the most probable transcript differ, and say which one prefix beam search finds.

## Further reading

- Repo phase 06, lessons 04, 05 and 17; the lessons cite Graves et al. (2006) for CTC, Graves (2012) for RNN-T and Radford et al. (2022) for Whisper (not read directly).
- [[aud.05]] runs the pipeline in the opposite direction: text to speech.
- Hannun, "Sequence Modeling with CTC" (Distill, 2017) for pictures of the lattice; we did not read it, and the prefix-search description above follows our own derivation which the exercise tests against brute force.

## Mastery criteria

You can (a) state the CTC collapse map and loss, (b) derive the forward recursion including the skip rule and the termination, (c) implement it in log space and verify it against brute-force enumeration, (d) explain why greedy decoding can be suboptimal and implement prefix beam search, (e) contrast CTC, RNN-T and attention encoder-decoders, and (f) compute WER and CER with an edit-distance table.
