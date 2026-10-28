---
id: mx.02
title: The Fourier transform
summary: Write any sampled signal as a sum of rotating waves, derive the discrete Fourier transform from orthogonality, see why the FFT is fast, and learn what resolution, aliasing and convolution mean in the frequency domain.
minutes: 80
teaches: [mx.dft, mx.sampling, mx.conv-theorem]
requires: [mx.roots-unity, la.matrix]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/docs/en.md (DFT definition, coefficient meaning, FFT idea, frequency resolution, convolution theorem, spectrograms, aliasing, zero padding sections). READ."
  - "math/coursIntroApprentissageAudio.pdf, French slides (PDF p.13-15 waveform and continuous Fourier transform, p.19-25 sampling rate, Shannon-Nyquist statement, DFT definition; p.26 start of PCM encoding). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/19-complex-numbers/docs/en.md (roots of unity link to the DFT). READ."
---

## Where this fits

A second of audio recorded at 16,000 samples per second is a list of 16,000 numbers. By eye, that list tells you almost nothing: is it a single note, a chord, speech? The same data looks completely different if you ask *how much of each pitch is present*. That question is answered by the **Fourier transform**, and the answer is what speech recognisers, music models and many time-series methods actually consume (as spectrograms). It also explains why convolutions in CNNs can be computed quickly, and gives meaning to the sine and cosine waves of sinusoidal positional encodings.

You need [[mx.01]]: complex exponentials and roots of unity. If matrices as change of basis feel fuzzy, revisit [[math.07]], because the DFT is nothing but a change of basis.

## The idea in plain words

- A musical chord is several pure tones played together. Your ear splits it back into notes. The Fourier transform is a mathematical ear: it answers "which pure tones, how loud, and starting at which point of their cycle?".
- Think of a stack of **spinning wheels** with different speeds: wheel 0 does not spin (it carries the average level), wheel 1 does one turn over the recording, wheel 2 does two turns, and so on. To find out how much of wheel $k$ is in your signal, you ride along with that wheel and average the signal against it. If the signal contains that rotation it adds up coherently and gives a big number; if it does not, positive and negative parts cancel and you get close to zero.
- Since the recording has a finite number of samples, there are only as many distinct wheels as samples. Nothing is lost: it is the same information in a new coordinate system, like describing a point by (east, north) or by (distance, angle).
- Sampling has a price. Wheels that spin too fast between two sample instants *look* like slow wheels (the wagon-wheel effect in films). That is **aliasing**, and the rule that avoids it is the Nyquist rule.
- Computing the averages naively costs "number of wheels times number of samples". A clever reuse of work (the **FFT**) brings that down dramatically.

## Definitions

Take $N$ samples $x[0],\dots,x[N-1]$ (real or complex), and let

$$
\omega=e^{2\pi i/N}
$$

be the primitive $N$-th root of unity from [[mx.01]]. The **discrete Fourier transform** (DFT) is

$$
X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-2\pi i kn/N}=\sum_{n=0}^{N-1}x[n]\,\omega^{-kn},\qquad k=0,\dots,N-1 .
$$

Each $X[k]$ is a complex number; $|X[k]|$ is how much of frequency $k$ is present and $\arg X[k]$ is its phase. The French course defines the same sum for a series of $T$ values and presents it as the discrete counterpart of the continuous transform $\hat s(\nu)=\int s(t)e^{-i2\pi\nu t}\,dt$ (the integral idea is developed in [[mx.05]]).

### Why this formula? A derivation from orthogonality

Goal: write $x$ as a combination of $N$ rotating waves $e_k[n]=\omega^{kn}$ (wave $k$ completes $k$ turns across the recording):

$$
x[n]=\sum_{k=0}^{N-1}c_k\,\omega^{kn}.
$$

To find $c_l$ we need the key fact from [[mx.01]]: for integers $j$,

$$
\sum_{n=0}^{N-1}\omega^{jn}=\begin{cases}N& j\equiv0\pmod N\\0&\text{otherwise,}\end{cases}
$$

(the roots of unity sum to zero unless they are all 1). Multiply both sides of the expansion by $\omega^{-ln}$ and sum over $n$:

$$
\sum_n x[n]\omega^{-ln}=\sum_k c_k\sum_n\omega^{(k-l)n}=\sum_k c_k\,N\,\delta_{kl}=N c_l .
$$

So $c_l=\frac1N X[l]$ with $X$ defined as above. Substituting back gives the **inverse DFT**

$$
x[n]=\frac1N\sum_{k=0}^{N-1}X[k]\,e^{+2\pi i kn/N}.
$$

In matrix form let $F\in\mathbb{C}^{N\times N}$ with $F_{kn}=\omega^{-kn}$. Then $X=Fx$ (shapes $N\times N$ times $N$) and the orthogonality fact says $F^{H}F=N I$, i.e. $F/\sqrt N$ is a **unitary** matrix: a rotation in complex space. This is why the transform loses nothing, and why Parseval's identity $\sum|x[n]|^2=\frac1N\sum|X[k]|^2$ holds.

### What the coefficients mean

- $X[0]=\sum_n x[n]$: the sum, proportional to the mean ("DC" component).
- For a **real** signal, $X[N-k]=\overline{X[k]}$, because $\omega^{-(N-k)n}=\omega^{kn}=\overline{\omega^{-kn}}$. So only bins $0\ldots N/2$ carry independent information.
- A real cosine $A\cos(2\pi k n/N+\varphi)$ with $0<k<N/2$ produces $X[k]=\frac{NA}{2}e^{i\varphi}$ (and its conjugate at $N-k$). Hence its amplitude is $A=2|X[k]|/N$. A sine has phase $-\pi/2$, so it shows up as the imaginary number $X[k]=-i\,NA/2$.

## The FFT: why it is fast

Computing every $X[k]$ directly costs $N$ multiplications for each of $N$ outputs, $O(N^2)$ operations: $N=1024$ gives $1{,}048{,}576$. The **fast Fourier transform** gets the same numbers in $O(N\log N)$, about $10{,}240$ for $N=1024$.

The Cooley-Tukey idea, for even $N$: split the sum into even-indexed and odd-indexed samples, $n=2m$ and $n=2m+1$:

$$
X[k]=\underbrace{\sum_{m}x[2m]\,e^{-2\pi i km/(N/2)}}_{E[k]}+e^{-2\pi ik/N}\underbrace{\sum_m x[2m+1]\,e^{-2\pi ikm/(N/2)}}_{O[k]} .
$$

$E$ and $O$ are DFTs of length $N/2$, and they are periodic in $k$ with period $N/2$. Moreover $e^{-2\pi i(k+N/2)/N}=-e^{-2\pi ik/N}$. Therefore for $k=0,\dots,N/2-1$:

$$
X[k]=E[k]+t_kO[k],\qquad X[k+N/2]=E[k]-t_kO[k],\qquad t_k=e^{-2\pi ik/N}\ (\text{the "twiddle factor"}).
$$

One DFT of size $N$ becomes two of size $N/2$ plus $O(N)$ work. Repeating down to size 1 takes $\log_2N$ levels, each costing $O(N)$, hence $O(N\log N)$. Libraries (`np.fft`, `torch.fft`, FFTW, cuFFT) handle any $N$ with more elaborate factorisations, but powers of two are the clean case.

## Frequency resolution, Nyquist and aliasing

Say the samples are taken at a **sampling rate** $f_s$ (samples per second), so the recording lasts $T=N/f_s$ seconds. Bin $k$ completes $k$ turns in $T$ seconds, so its physical frequency is

$$
f_k=\frac{k\,f_s}{N},\qquad\text{bin spacing }\Delta f=\frac{f_s}{N}=\frac1T .
$$

**Resolution depends on the recording length, not on the number of bins.** To separate tones $1$ Hz apart you need at least about one second of signal. Padding the signal with zeros before the transform gives a smoother-looking spectrum (more bins) but does not add information.

The highest frequency a real signal can carry in $N$ samples is bin $N/2$, i.e. $f_s/2$, the **Nyquist frequency**. A frequency $f$ above it is not lost; it is *folded* back and appears at the **alias**. For $f$ reduced modulo $f_s$ into $[0,f_s)$, the apparent frequency is $\min(f,\ f_s-f)$.

Proof sketch: at sample times $t=n/f_s$, $\cos(2\pi(f+mf_s)t)=\cos(2\pi ft+2\pi mn)=\cos(2\pi ft)$. Frequencies that differ by a multiple of $f_s$ produce identical samples, so the samples cannot tell them apart.

The slides state the Shannon-Nyquist theorem for a signal whose spectrum lies in a band $[f_{\min},f_{\max}]$: sampling at $f_s\ge2(f_{\max}-f_{\min})$ allows perfect reconstruction of the continuous signal. For ordinary audio starting near $0$ Hz this is the familiar rule $f_s\ge2f_{\max}$. (The band-limited form is the more general one; the slides add that the band-limited assumption is reasonable for audible frequencies.) Typical consequence: music at $44.1$ kHz covers frequencies up to $22.05$ kHz, a bit above human hearing; speech models often use $16$ kHz, so everything above $8$ kHz is discarded or must be filtered *before* sampling.

A related effect is **spectral leakage**: if the signal does not contain a whole number of periods in the window, its energy spreads over neighbouring bins. Multiplying by a smooth window (Hann, etc.) before the transform reduces the leakage.

## The convolution theorem

The **circular convolution** of two length-$N$ signals is $(a\circledast b)[n]=\sum_{m}a[m]\,b[(n-m)\bmod N]$. Taking the DFT of both sides:

$$
\operatorname{DFT}(a\circledast b)[k]=\sum_n\sum_m a[m]b[(n-m)\bmod N]\omega^{-kn}.
$$

Substitute $n=m+p$ (indices mod $N$): $\omega^{-kn}=\omega^{-km}\omega^{-kp}$, so the double sum factorises into $A[k]\,B[k]$:

$$
\boxed{\operatorname{DFT}(a\circledast b)=\operatorname{DFT}(a)\cdot\operatorname{DFT}(b)\quad\text{(pointwise product).}}
$$

So convolution, an $O(N^2)$ operation, costs $O(N\log N)$ via FFT, multiply, inverse FFT. A CNN "convolution" layer with a large kernel can be implemented this way (frameworks use direct or Winograd methods for small kernels where it is faster). Also, the frequency view of a filter is simply its DFT: low-pass filters keep small $k$.

## Spectrograms (preview)

A single DFT of a whole recording says *which* frequencies occur but not *when*. The **short-time Fourier transform** cuts the signal into overlapping windows of, say, $25$ ms, multiplies each by a window function and takes the DFT of each. Stacking the magnitudes gives a matrix of shape (frames x frequency bins), the **spectrogram**, an image that audio networks treat like a picture. The window length trades time resolution against frequency resolution ($\Delta f=1/T_{\text{window}}$), an uncertainty trade-off, not a bug.

## Worked example

Checked in the exercise code.

1. Signal $x=(1,2,3,4,0,0,0,0)$, $N=8$. Then $X[0]=10$ (sum), and $|X|=(10,\ 7.254,\ 2.828,\ 2.715,\ 2,\ 2.715,\ 2.828,\ 7.254)$. Real input, so it is symmetric: $|X[N-k]|=|X[k]|$.
2. $N=16$ samples of $x[n]=\sin(2\pi\cdot3n/16)+0.5\cos(2\pi\cdot5n/16)$. The transform has exactly four non-zero entries: $X[3]=-8i$, $X[13]=+8i$ (the sine: $N\cdot1/2=8$, imaginary), $X[5]=X[11]=4$ (the cosine: $16\cdot0.5/2=4$, real).
3. Resolution: $f_s=8000$ Hz, $N=1024$ gives $\Delta f=7.8125$ Hz.
4. Aliasing: sample $\sin(2\pi\cdot7t)$ at $f_s=10$ Hz. The samples equal those of $-\sin(2\pi\cdot3t)$ exactly: the 7 Hz tone is above the Nyquist limit 5 Hz and appears at $10-7=3$ Hz (with a sign flip).
5. Convolution: $a=(1,2,3,0)$, $b=(1,1,0,0)$: circular convolution is $(1,3,5,3)$, equal to the inverse FFT of the product of the FFTs.

## Implementation

```python
import numpy as np

def dft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N)   # shape (N, N)
    return F @ x

def fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 1:
        return x
    E, O = fft(x[0::2]), fft(x[1::2])
    t = np.exp(-2j * np.pi * np.arange(N // 2) / N) * O
    return np.concatenate([E + t, E - t])

x = np.random.default_rng(0).normal(size=64)
print(np.allclose(dft(x), np.fft.fft(x)), np.allclose(fft(x), np.fft.fft(x)))
```

## Shapes

- Input $x$: $(N,)$; output $X$: $(N,)$ complex. For real input `np.fft.rfft` returns only the $N/2+1$ non-redundant bins.
- DFT matrix $F$: $(N,N)$, dense and complex: 16 bytes per entry, which is why nobody builds it for large $N$.
- Spectrogram: (number of frames, $N_{\text{win}}/2+1$) after taking magnitudes. Two-dimensional images use the 2D FFT: apply the 1D transform along each axis.

## Common misconceptions

- "More zero padding means better resolution." It only interpolates. Resolution is $1/T$ of the real signal.
- "The FFT is a different transform." It is the same output as the DFT, computed faster.
- "Negative frequencies are an error." For real signals they mirror the positive ones; for complex signals (I/Q data, complex-valued networks) they carry new information.
- "Aliasing can be removed afterwards." Once sampled, a 7 Hz and a 3 Hz tone are identical in the samples (as shown above). Filter before sampling.
- "The magnitude spectrum tells the whole story." Phase carries timing/shape information; discarding it (as spectrograms of magnitudes do) is a modelling choice with costs.

## Key takeaways

- The DFT expresses $N$ samples in the basis of $N$ rotating waves; orthogonality of roots of unity gives both transform and inverse.
- $F/\sqrt N$ is unitary: nothing is lost, energy is preserved.
- FFT splits into even and odd samples and reuses work: $O(N\log N)$ instead of $O(N^2)$.
- Bin spacing is $f_s/N=1/T$; frequencies above $f_s/2$ alias.
- Convolution in time is pointwise multiplication in frequency.

## Explain it yourself

1. Why can two different tones give exactly the same samples, and what does a musician-friendly anti-aliasing filter do?
2. Use the even/odd split to explain, in words, why the FFT needs about $N\log_2N$ operations.

## Further reading

- Repo lesson `phases/01-math-foundations/20-fourier-transform` (windowing, DFT properties, links to positional encodings and CNNs).
- Audio slides `coursIntroApprentissageAudio.pdf`, p.19-26 (sampling) and the continuation on spectrogram-based networks.
- Next in this course: [[mx.03]] applies eigenvectors to graphs; its Laplacian eigenvectors on a ring graph are exactly these Fourier waves.

## Mastery criteria

- [ ] I can derive the inverse DFT from the orthogonality of roots of unity.
- [ ] I can compute the bin spacing and the Nyquist frequency of a recording and predict the alias of a given tone.
- [ ] I can write the recursive FFT and explain the twiddle-factor step.
- [ ] I can state the convolution theorem and say when FFT-based convolution is worthwhile.
