---
id: aud.02
title: "Spectrograms, the mel scale and MFCCs"
summary: Turn an STFT into the features speech and audio models actually use: log-power spectrograms, mel filterbanks built as a matrix, MFCCs via the DCT, and delta coefficients, all implemented from scratch.
minutes: 80
teaches: [aud.mel, aud.mfcc]
requires: [aud.stft, aud.waveform, la.matrix, alg.exp-log]
sources:
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/02-spectrograms-mel-features/docs/en.md (log-magnitude, mel formula, triangular filterbank as one matmul, log-mel, MFCC via DCT type II, 80 mels default, pitfalls). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/01-audio-fundamentals/docs/en.md (STFT recap used as the input of this lesson). READ."
  - "math/coursIntroApprentissageAudio.pdf, PDF p.30-35 (spectrogram vs sonagramme, mel scale formulas m = 2595 log10(1+f/700) and its inverse, the four MFCC steps). READ."
  - "math/coursIntroApprentissageAudio.pdf, PDF p.40, 49-53 (model families: mel-spectrogram, raw-window and raw-sample models; spectrograms as sequences or as images; summary). READ."
---

## Where this fits

[[aud.01]] ended with the STFT: a complex matrix of shape `(frames, bins)`. For a 16 kHz clip with `n_fft=512` that is 257 numbers per 10 ms, and most of them are irrelevant for the task: neighbouring bins carry nearly the same information, high-frequency bins are tiny compared with low ones, and the exact pitch of a voice is rarely what you want to recognise. Speech recognisers ([[aud.04]]), sound classifiers ([[aud.03]]) and text-to-speech models ([[aud.05]]) therefore compress the STFT into a *log-mel spectrogram*, typically 80 numbers per frame, and sometimes compress further into about 13 *MFCCs*. This lesson builds that compression step by step and implements each stage.

## The idea in plain words

Imagine you could paint the sound as a picture: time left to right, pitch bottom to top, brightness for loudness. The raw STFT picture is too detailed and too lopsided, for three reasons, each of which has a simple remedy.

1. **Loudness spans enormous ranges.** A whisper and a shout differ by a factor of thousands in energy, and the ear judges loudness roughly on a logarithmic scale. So we show the *logarithm* of the energy.
2. **The ear is not a ruler.** We tell apart 300 Hz from 600 Hz easily, but 10 000 Hz from 10 300 Hz hardly at all: equal *ratios* of frequency sound like equal steps, equal *differences* do not. So we squeeze the pitch axis so that low frequencies get fine detail and high frequencies are lumped together. This warped axis is the **mel scale**.
3. **Neighbouring numbers are redundant.** Adjacent channels of a smooth spectrum move together. A final transform rewrites a spectrum as "a smooth outline plus ripples" and lets us keep only the outline. The result are the **MFCCs**, a short vector that describes the *shape* of the spectrum in each frame, roughly the shape of the mouth and throat producing the sound, while ignoring the fine ripples caused by the pitch of the voice.

Finally, because speech is about *change*, we append to each frame a number telling how fast each feature is moving: the **delta** features.

## Power and log spectrograms

Given the STFT $X[t,k]$ from [[aud.01]]:

- magnitude spectrogram $|X[t,k]|$,
- **power spectrogram** $P[t,k]=|X[t,k]|^2$ (energy),
- **log spectrogram** $\log(P[t,k]+\varepsilon)$ or in decibels $10\log_{10}(P+\varepsilon)$.

The small constant $\varepsilon$ (e.g. $10^{-10}$) avoids $\log 0$ for silent bins. Taking the logarithm compresses a dynamic range of five or six orders of magnitude into a range a network can handle, and it turns "twice as loud" (a multiplication) into "plus a constant" (an addition). Production systems differ in conventions (natural log, dB, clipping, normalisation); the repo lesson warns that mixing them up between training and inference is a silent bug, so write the convention down.

## The mel scale

The mel scale maps a frequency $f$ in hertz to a perceptual pitch coordinate $m$:

$$
m(f)=2595\,\log_{10}\!\left(1+\frac{f}{700}\right),\qquad f(m)=700\left(10^{m/2595}-1\right).
$$

The inverse follows by solving for $f$: $m/2595=\log_{10}(1+f/700)$, so $1+f/700 = 10^{m/2595}$. (The same curve is also written $1127\ln(1+f/700)$, because $2595/\ln 10\approx1127$.) The constant 700 is where the curve changes character: for $f\ll700$ we have $\ln(1+f/700)\approx f/700$, nearly linear; for $f\gg700$ it grows like $\ln f$, logarithmic.

How much does the axis compress? The slope is

$$
\frac{dm}{df}=\frac{2595}{700\ln 10}\cdot\frac{1}{1+f/700}\approx\frac{1.61}{1+f/700}\ \text{mel/Hz}.
$$

At $f=0$ one hertz is 1.61 mel; at 1000 Hz it is $0.663$ mel; at 8000 Hz only $0.130$ mel. Low frequencies receive roughly twelve times more of the new axis per hertz than the top of the 0-8 kHz range.

Reference values (reproduced in the exercise): $m(700)=781.17$, $m(1000)=999.99$ (the scale is calibrated so that 1000 Hz is about 1000 mel), $m(4000)=2146.07$, $m(8000)=2840.02$. This is one of several mel-scale variants; librosa also offers the "Slaney" version (linear below 1 kHz); features computed with different variants are not interchangeable.

## The mel filterbank as a matrix

We want $M$ numbers per frame, each summarising the energy in one pitch band, with bands equally wide *on the mel axis*.

**Construction** (for $M$ filters, FFT size $N$, sampling rate $f_s$, range $[f_{\min},f_{\max}]$):

1. Take $M+2$ points equally spaced in mel between $m(f_{\min})$ and $m(f_{\max})$, convert them back to hertz: $f_0<f_1<\dots<f_{M+1}$.
2. Filter $j$ ($j=1..M$) is a triangle that rises from $0$ at $f_{j-1}$ to $1$ at $f_j$ and falls back to $0$ at $f_{j+1}$. Evaluate it at the centre frequency $\nu_k=k f_s/N$ of each DFT bin $k=0..N/2$:

$$
H_j(\nu_k)=\max\!\left(0,\ \min\!\left(\frac{\nu_k-f_{j-1}}{f_j-f_{j-1}},\ \frac{f_{j+1}-\nu_k}{f_{j+1}-f_j}\right)\right).
$$

Neighbouring triangles overlap by half, so every frequency between $f_1$ and $f_M$ is shared between two filters.

3. Stack the $H_j(\nu_k)$ into a matrix $\mathbf H\in\mathbb R^{M\times(N/2+1)}$. The mel spectrogram is then a single matrix product:

$$
\mathbf M_{\text{spec}}=\mathbf P\,\mathbf H^\top,\qquad
\mathbf P\in\mathbb R^{T\times(N/2+1)},\ \ \mathbf H^\top\in\mathbb R^{(N/2+1)\times M},\ \ \mathbf M_{\text{spec}}\in\mathbb R^{T\times M}.
$$

and the **log-mel spectrogram** is $\log(\mathbf M_{\text{spec}}+\varepsilon)$, the input of Whisper-style recognisers (80 mels) and many audio classifiers (the repo lesson quotes 128 mels for classification; these choices are conventions, not laws).

**Worked example 1: a tiny filterbank.** $M=8$ filters, $N=512$, $f_s=16$ kHz, range 0-8 kHz. The $M+2=10$ boundary frequencies (Hz) are

$$
0,\ 226.2,\ 525.5,\ 921.5,\ 1445.4,\ 2138.6,\ 3055.9,\ 4269.5,\ 5875.3,\ 8000 .
$$

The distances between consecutive boundaries grow from $226$ Hz to $2125$ Hz: the filters are narrow at low pitch and wide at high pitch. The matrix `H` has shape `(8, 257)`. Filter 3 (index 2 counting from 0) spans 525.5 to 1445.4 Hz and peaks at 921.5 Hz. A pure 1000 Hz tone ($N=512$, Hann-windowed power) puts its energy mostly into this filter: the 8 mel energies are `[0, 0, 20891.8, 3684.2, 0, 0, 0, 0]` (rounded), so the arg-max is index 2.

**Worked example 2: a chirp.** For a sine whose frequency rises linearly from 200 Hz to 4000 Hz in 1 s ($x(t)=\sin(2\pi(200t+1900t^2))$) sampled at 16 kHz, with `n_fft=512, hop=160` and a 40-filter bank, the arg-max mel index at frames 0, 10, 20, ..., 90 is `[4, 10, 14, 17, 20, 22, 24, 26, 27, 29]`. The frequency rose by a constant amount per frame, but the mel index rises quickly at first and then slows: a straight line in hertz becomes a curve that flattens in mel.

## Cepstral coefficients: MFCCs

The classical recipe (also given in the French course, which lists the same four steps): window and Fourier transform; project the power spectrum on the mel scale; take the log of each mel energy; then apply a **discrete cosine transform (DCT)** to the vector of log-mel energies and keep its first coefficients. The coefficients are the **mel-frequency cepstral coefficients (MFCCs)**.

### The DCT, derived

Let $y\in\mathbb R^{M}$ be the log-mel vector of one frame. The DCT-II expresses it as a sum of cosines of increasing "wiggliness":

$$
c_k=\sum_{m=0}^{M-1} y_m\cos\!\left(\frac{\pi k\,(2m+1)}{2M}\right),\qquad k=0,\dots,M-1 .
$$

Cosine $k$ completes $k/2$ full periods across the $M$ channels: $k=0$ is a constant (the average level), $k=1$ is one slow half-wave (tilt), higher $k$ oscillate faster. With scaling factors $\sqrt{2/M}$ for $k\ge1$ and $\sqrt{1/M}$ for $k=0$ the rows of the transform matrix $\mathbf D\in\mathbb R^{M\times M}$ become orthonormal:

$$
D_{km}=\alpha_k\cos\!\left(\frac{\pi k(2m+1)}{2M}\right),\quad \alpha_0=\sqrt{\tfrac1M},\ \alpha_{k>0}=\sqrt{\tfrac2M},\qquad \mathbf D\mathbf D^\top=\mathbf I .
$$

Orthonormality means $\mathbf D$ is a rotation of the vector $y$ (it preserves lengths, and the inverse is the transpose), so keeping all $M$ coefficients loses nothing, and keeping the first $K$ is the best *smooth* approximation in this basis:

$$
\hat y=\mathbf D_{[:K]}^\top\,\mathbf D_{[:K]}\,y,\qquad \mathbf D_{[:K]}\in\mathbb R^{K\times M}.
$$

**Why cosines and not complex exponentials?** A DCT is the DFT of the signal mirrored around its edge, so there is no artificial jump where the vector "wraps around"; energy is packed into the first few coefficients and they are real numbers.

**Why this helps.** The log-mel vector is the sum of a slow component (the spectral envelope, shaped by the vocal tract) and a fast ripple (the harmonics of the voice pitch, plus noise). Because a product of two things becomes a sum after the log, a transform that separates slow from fast separates the two causes. Keeping the first 13 coefficients keeps the envelope and discards most of the ripple, and the coefficients are much less correlated with each other than neighbouring mel channels. (This source/filter argument is the classical explanation; modern networks fed raw log-mels often match MFCCs, so MFCCs survive mainly in small models, speaker systems and baselines.)

**Worked example 3: DCT and truncation.** $\mathbf D$ for $y=(1,2,3,4)$ gives $c=(5,\,-2.2304,\,0,\,-0.1585)$: a straight ramp is almost entirely a constant (5 = sum/2) plus one slow half-wave. On frame 40 of the chirp above (26 mel channels), reconstructing the log-mel vector from the first $K$ coefficients gives an RMS error of $4.96$ (K=3), $3.32$ (K=6), $0.70$ (K=13) and exactly $0$ (K=26). Thirteen coefficients preserve nearly all of the outline.

### Delta features

One frame is a snapshot; speech is defined by transitions (a plosive is a burst *followed by* a vowel). Append to each coefficient its local slope in time. Fit a straight line through the $2N+1$ values $c_{t-N},\dots,c_{t+N}$ by least squares; the slope is

$$
\Delta c_t=\frac{\sum_{n=-N}^{N} n\,c_{t+n}}{\sum_{n=-N}^{N} n^2}
=\frac{\sum_{n=1}^{N} n\,(c_{t+n}-c_{t-n})}{2\sum_{n=1}^{N} n^2}.
$$

(The numerator pairs $+n$ and $-n$; the denominator uses $\sum_{n=-N}^{N}n^2=2\sum_{n=1}^N n^2$.) Applying the same formula to $\Delta c$ gives the $\Delta\Delta$ (acceleration) features. At the sequence edges, repeat the first and last frame (`mode='edge'` padding). A classic feature vector is 13 MFCC + 13 $\Delta$ + 13 $\Delta\Delta$ = 39 numbers per frame, shape `(T, 39)`.

**Check.** For a ramp $c_t=t$ (slope 1) the delta is exactly 1 at interior frames, and for $c_t=3t$ it is 3; for $c_t=t^2$ with $N=2$ it equals $2t$: e.g. $t=2,3$ give $4.0$ and $6.0$ in the exercise (where `[0,1,4,9,16,25]` yields deltas `[0.9, 2.2, 4.0, 6.0, 5.8, 4.1]`; the first and last values are distorted by the edge padding).

## Implementation

```python
import numpy as np

def hz_to_mel(f):  return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)
def mel_to_hz(m):  return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)

def mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None):
    fmax = sr / 2 if fmax is None else fmax
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, len(freqs)))
    for j in range(n_mels):
        lo, mid, hi = pts[j], pts[j + 1], pts[j + 2]
        fb[j] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    return fb                                              # (n_mels, n_fft//2 + 1)

def dct_matrix(n_in, n_out):
    k = np.arange(n_out)[:, None]; m = np.arange(n_in)[None, :]
    D = np.sqrt(2.0 / n_in) * np.cos(np.pi * k * (2 * m + 1) / (2 * n_in))
    D[0] /= np.sqrt(2.0)                                   # alpha_0 = sqrt(1/n_in)
    return D                                               # (n_out, n_in)

def mfcc_from_power(P, sr, n_fft, n_mels=26, n_mfcc=13):
    logmel = np.log(P @ mel_filterbank(n_mels, n_fft, sr).T + 1e-10)   # (T, n_mels)
    return logmel @ dct_matrix(n_mels, n_mfcc).T                       # (T, n_mfcc)

def deltas(c, N=2):
    T = len(c)
    pad = np.pad(c, ((N, N), (0, 0)), mode="edge")
    out = np.zeros_like(c, dtype=float)
    for n in range(1, N + 1):
        out += n * (pad[N + n:N + n + T] - pad[N - n:N - n + T])
    return out / (2 * sum(n * n for n in range(1, N + 1)))
```

### Shapes

| Step | Shape |
|---|---|
| waveform | `(L,)` |
| power spectrogram `P` | `(T, N/2+1)` |
| filterbank `H` | `(M, N/2+1)` |
| log-mel | `(T, M)` |
| DCT matrix `D` | `(K, M)` |
| MFCC | `(T, K)` |
| MFCC + $\Delta$ + $\Delta\Delta$ | `(T, 3K)` |

**What libraries do differently.** `librosa.feature.melspectrogram` and `torchaudio.transforms.MelSpectrogram` use the same pipeline but typically *area-normalise* each triangle (so wide high-frequency filters do not receive more energy), offer the Slaney or HTK mel variants, and take frame/hop in samples. Whisper ships its own filterbank and normalisation. Reusing someone else's weights means reusing their exact features.

## Common misconceptions

- **"Mel means the filters are equally wide."** They are equally wide in mel, which is increasingly wide in hertz; our unnormalised triangles all have height 1, so wider filters sum more energy.
- **"MFCCs are just compressed spectrograms."** They are spectra of a *spectrum*: coefficient index relates to ripple speed along frequency, not to a frequency.
- **"More mel bins is always better."** With `n_fft=400` there are only 201 bins; asking for 128 mels leaves the lowest filters narrower than one bin, producing empty or duplicate channels.
- **"I can compare my features to a library's directly."** Window variant, mel variant, filter normalisation, log base and clipping all differ.
- **"c0 is a pitch feature."** It is the overall (log-)energy level; many systems drop it or keep it separately.

## Key takeaways

- Log power compresses dynamic range; the mel scale $m=2595\log_{10}(1+f/700)$ spends resolution where hearing does.
- The mel filterbank is a fixed `(M, N/2+1)` matrix, so the mel spectrogram is one matmul $\mathbf P\mathbf H^\top$.
- The orthonormal DCT-II is a rotation of the log-mel vector; truncating it keeps the smooth spectral outline. Check $\mathbf D\mathbf D^\top=\mathbf I$.
- Deltas are least-squares local slopes: $\sum n(c_{t+n}-c_{t-n})/(2\sum n^2)$.
- Always record `sr`, `n_fft`, hop, window, `n_mels`, log convention and normalisation: a mismatch between training and inference fails silently.

## Explain it yourself

1. Why does a flat straight line of frequency (a chirp) look curved on a mel axis? Use the derivative $dm/df$.
2. Explain why keeping only the first 13 DCT coefficients of a log-mel vector removes the ripple caused by the voice pitch but keeps the vocal-tract outline.

## Further reading

- Repo phase 06, lesson 02 (feature choices per task and pitfalls).
- Davis and Mermelstein (1980) introduced MFCCs for speech recognition; the repo lesson cites it (we did not read the paper itself).
- [[aud.03]] uses the MFCC statistics built here as a classifier input.

## Mastery criteria

You can convert between hertz and mel, build a triangular mel filterbank matrix without a library, show that the DCT matrix is orthonormal, compute MFCCs and deltas with explicit shapes, and explain which choices (window, mel variant, `n_mels`, log) must match between training and deployment.
