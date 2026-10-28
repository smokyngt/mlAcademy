---
id: aud.03
title: "Audio classification: from MFCC k-NN to CNNs and SpecAugment"
summary: Turn a clip into a fixed-size vector or an image, classify it with k-NN or a CNN, augment the data with time and frequency masks, and evaluate honestly with confusion matrices, per-class recall and macro F1.
minutes: 85
teaches: [aud.classify, aud.specaugment]
requires: [aud.mfcc, aud.mel, ml.generalization, nn.layer, dl.early-stopping]
sources:
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/03-audio-classification/docs/en.md (k-NN on MFCC mean+variance, 2D CNN on log-mels, AST/BEATs/Whisper-encoder backbones, class imbalance, mixup, SpecAugment, metrics, the CNN code and its parameter-count claim). READ."
  - "math/coursAvanceApprentissageAudio.pdf, PDF p.3-8 (principle of data augmentation: label-preserving and realistic; time shift, noise, volume, time stretching, pitch shifting, compressor) and p.14 (quiz on which augmentation to avoid for music identification). READ."
  - "math/coursIntroApprentissageAudio.pdf, PDF p.40-47 (three families of audio networks, long raw sequences, variable durations, adaptive/global pooling, zero-padding batches) and p.49-53 (spectrogram as an image, rectangular kernels, CNN comparison, summary). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/en.md (metric per task: top-1, mAP, macro F1, per-class recall). READ."
---

## Where this fits

"Is that a dog or a siren?", "Which language is this?", "Which command was spoken: yes, no, stop?", "Is the speaker angry?": all are *audio classification*, one label (or several) for a clip. It is the first place where the features of [[aud.02]] meet a learning algorithm, and it is the simplest way to see the whole recipe of audio machine learning: **waveform, spectrogram, features, model, metric**. The same recipe, with a different output layer, powers speech recognition ([[aud.04]]).

You need the log-mel and MFCC features of [[aud.02]], the idea of generalisation and validation (concept `ml.generalization`), and the notion of a neural-network layer (concept `nn.layer`).

## The idea in plain words

A clip has a variable length, but a classifier wants a fixed-size input. There are two ways out.

- **Summarise.** Compute features frame by frame, then replace the whole clip by a few numbers: the average of each feature over time and how much it varies. "This clip is, on average, bright and high-pitched, and fairly steady" is a perfectly good description of a siren or a whistle. A nearest-neighbour rule then says "the clip is most like these labelled clips".
- **Look at the picture.** Treat the log-mel spectrogram as an image with time and frequency axes, and let a convolutional network find local patterns (a rising sweep, a vertical click, a horizontal hum), then pool over time to ignore *when* they happened. The network can learn what to look for, but it needs many examples.

Audio datasets are small, imbalanced and noisy. Two habits do most of the work: **augmentation**, showing the network slightly corrupted versions of each clip so it cannot rely on accidents; and **honest evaluation**, looking at errors class by class instead of one accuracy number.

## Clip-level features and k-NN

Let $C\in\mathbb R^{T\times K}$ be the MFCC (or log-mel) matrix of a clip: $T$ frames, $K$ coefficients. The summary vector is

$$
\mathbf z=\big[\ \mu_1,\dots,\mu_K,\ \sigma_1,\dots,\sigma_K\ \big]\in\mathbb R^{2K},\qquad
\mu_j=\frac1T\sum_t C_{tj},\quad \sigma_j=\sqrt{\frac1T\sum_t (C_{tj}-\mu_j)^2 } .
$$

Variable $T$ disappears. (The French course states the same issue for networks: sounds have different durations, and global pooling turns a `(channels, length)` feature map into a `(channels)` vector independent of the length.)

**k-nearest neighbours.** Store the labelled training vectors $\mathbf z_i$. To classify a new clip, find the $k$ training clips with the largest cosine similarity

$$
\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\,\|\mathbf b\|}
$$

and return the most common label among them. There is no training; the model *is* the data. Cosine similarity ignores the overall scale of a vector (which depends on loudness), Euclidean distance does not; either can be used, and the right choice is an empirical matter.

**Shapes.** `Xtr: (N_train, 2K)`, `Xte: (N_test, 2K)`, similarity `Xte @ Xtr.T` after row normalisation: `(N_test, N_train)`; for each row take the indices of the $k$ largest entries and vote.

### Worked example 1: a noisy four-class problem

We synthesise 0.25 s clips at 8 kHz in four classes (the exact generator is in the exercise): a 440 Hz tone, a 520 Hz tone, a rising sweep and white noise, each with additive Gaussian noise at standard deviation 3 (so the signal is buried: each tone has amplitude 1). Features: log power in 16 equal-width frequency bands over frames of 200 samples, summarised by mean and standard deviation over time (32 numbers per clip). With 30 clips per class and seed 0 we split 80 training / 40 test clips (test class counts 13, 8, 9, 10).

| $k$ | test accuracy |
|---|---|
| 1 | 0.775 |
| 3 | 0.900 |
| 5 | 0.875 |

For $k=3$ the confusion matrix (rows = true class, columns = predicted) is

$$
\begin{pmatrix}12&0&1&0\\1&6&1&0\\1&0&8&0\\0&0&0&10\end{pmatrix}
$$

The two nearby tones (440 and 520 Hz) are the hardest classes: the 520 Hz class has recall $6/8=0.75$. Noise is never confused with a tone. The aggregate accuracy $36/40=0.9$ hides this; the confusion matrix shows where the model fails.

## CNNs on log-mel images

A log-mel spectrogram `(T, M)` becomes a one-channel image `(1, T, M)`; a batch is `(B, 1, T, M)`. A 2-D convolution with $C_{in}$ input channels, $C_{out}$ output channels and a $k\times k$ kernel has $C_{in}C_{out}k^2+C_{out}$ parameters and shrinks each axis according to

$$
o=\left\lfloor\frac{i+2p-k}{s}\right\rfloor+1 ,
$$

(input length $i$, padding $p$, stride $s$; the same formula appears in the French course for 1-D convolutions). With $k=3$, $p=1$, $s=1$ the length is preserved; a $2\times2$ max-pool halves each axis.

The repo lesson gives a small network: three `Conv(3x3, pad 1) + ReLU` blocks with 32, 64, 128 channels, two $2\times2$ max-pools after the first two, global average pooling, and a linear layer to the classes. For an input `(1, 100, 80)` (100 frames, 80 mels) the shapes are

| stage | shape (channels, time, mel) |
|---|---|
| input | (1, 100, 80) |
| conv1 | (32, 100, 80) |
| pool | (32, 50, 40) |
| conv2 | (64, 50, 40) |
| pool | (64, 25, 20) |
| conv3 | (128, 25, 20) |
| global average pool | (128,) |
| linear | (50,) for 50 classes |

Parameters: $320+18\,496+73\,856+6\,450=99\,122$ for 50 classes (checked by the exercise). **The repo lesson describes this network as "3M parameters"; our count is about 0.1M**, so that figure in the source is inconsistent with its own code. Counting parameters yourself is a good habit.

**Design notes.** (1) The two axes of a spectrogram are *not* interchangeable, unlike the axes of a photograph: a pattern shifted in time is the same event, a pattern shifted in frequency is a different sound. The French course therefore suggests rectangular kernels in some architectures. (2) Global pooling over time makes the network independent of clip length. (3) The French course lists three input families: mel-spectrogram models, models on windows of raw waveform (the first layer learns its own filters), and models on individual raw samples; the spectrogram family is cheap and classic, the raw-sample models are expensive but often strongest. (4) A comparison cited in the same course of image CNNs on audio spectrograms found deeper image architectures (ResNet-50 and Inception) clearly better than a fully connected network on that video-tagging task.

**The 2026 default, per the repo lesson:** do not train from scratch. Take a pretrained audio backbone (the Audio Spectrogram Transformer, a ViT on spectrogram patches; BEATs; or a frozen Whisper encoder) and train a small head. The repo lesson reports ESC-50 accuracies around 97% for the best pretrained models; we did not verify these numbers.

## Data augmentation

Augmentation generates new training examples by transforming existing ones. The French course states two requirements: the transformation must **not change the information used for supervision**, and it should correspond to **realistic distortions**. Its audio list: time shift, additive noise, volume change and time stretching (changes duration, not pitch), plus pitch shifting (changes pitch, not duration) and a compressor (dynamic range compression).

*Which one to avoid?* The course asks which augmentation is dangerous for identifying a song from a recording (Shazam-style): pitch shift, because the pitch *is* the identifying information. Likewise pitch shifting would be harmful for speaker gender or for note classification.

### SpecAugment: masking the spectrogram

SpecAugment (Park et al.; summarised by the repo lesson, original paper not read) works on the log-mel image `S` of shape `(T, M)`:

- **Frequency mask.** Choose a width $f$ uniformly in $\{0,\dots,F\}$ and a start $f_0$ uniformly in $\{0,\dots,M-f\}$; set $S[:,f_0:f_0+f]$ to a constant (zero, or the mean).
- **Time mask.** Choose $t$ uniformly in $\{0,\dots,T_{\max}\}$ and a start $t_0\in\{0,\dots,T-t\}$; set $S[t_0:t_0+t,:]$ to a constant.

Repeat several times. The network learns to classify with parts of the picture missing, like a listener who still recognises a song through a bad phone line or a cough. The repo lesson suggests time mask 20 and frequency mask 10 as a starting point for the exercise. Masks only erase, so the label is preserved as long as the masked patch is not the whole evidence.

**Worked example 2: masks on a toy array.** On a $10\times6$ array of ones, a time mask of width 3 starting at row 7 and a frequency mask of width 1 at column 3 zero out $3\cdot6+1\cdot10-3\cdot1=25$ of the 60 entries (the overlap, 3 cells, is counted once); the sum drops from 60 to 35. This is exactly what the exercise `time_mask`/`freq_mask` functions produce.

### Mixup

Mixup blends two examples and their labels, $x=\lambda x_1+(1-\lambda)x_2$ and $y=\lambda y_1+(1-\lambda)y_2$ (labels one-hot). With $\lambda=0.7$, $y_1=(1,0)$, $y_2=(0,1)$ the new target is $(0.7,0.3)$. Train with cross-entropy against the soft target. Mixed audio is also physically plausible: two sounds playing together.

### Class imbalance

Real datasets are imbalanced (UrbanSound-style sets about 10:1; AudioSet has a very long tail, per the repo lesson). Remedies: balanced sampling *in training only*, augmentation of rare classes, and per-class metrics.

## Evaluation

For $K$ classes build the confusion matrix $C_{ab}$ = number of clips of true class $a$ predicted as $b$. Then

$$
\text{recall}_a=\frac{C_{aa}}{\sum_b C_{ab}},\qquad
\text{precision}_a=\frac{C_{aa}}{\sum_b C_{ba}},\qquad
F1_a=\frac{2\,\text{precision}_a\,\text{recall}_a}{\text{precision}_a+\text{recall}_a},
$$

accuracy $=\sum_a C_{aa}/\sum_{ab}C_{ab}$, and **macro F1** is the unweighted mean of $F1_a$ over classes. Use top-1 / top-5 accuracy for exclusive classes with balanced data, **mAP** (mean average precision) for multi-label tagging like AudioSet, and macro F1 plus per-class recall when classes are imbalanced (repo evaluation lesson).

**Worked example 3.** $C=\begin{pmatrix}8&1&1\\2&6&2\\0&1&9\end{pmatrix}$ (30 clips). Recalls $0.8, 0.6, 0.9$; precisions $0.8, 0.75, 0.75$; $F1=0.8,\ 0.667,\ 0.818$; macro F1 $=0.7616$; accuracy $=23/30=0.767$.

**Why macro F1 beats accuracy under imbalance.** With 95 negatives and 5 positives, a model that always answers "negative" has accuracy 0.95, but F1 is $0.974$ for the negative class and $0$ for the positive one: macro F1 $=0.487$.

**Avoid leakage.** Clips cut from the same long recording are not independent. If some are in training and others in test, k-NN and CNNs can match the recording rather than the sound class and report inflated scores. Split by source recording or speaker (benchmarks such as ESC-50 ship predefined folds for this reason). Choose hyper-parameters ($k$, augmentation strength) on a validation split, never on the test set (concept `ml.generalization`).

## Implementation

```python
import numpy as np

def knn_predict(Xtr, ytr, Xte, k=3):
    a = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    b = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
    sim = b @ a.T                                   # (N_test, N_train) cosine similarities
    out = []
    for row in sim:
        nn = np.argsort(-row)[:k]
        out.append(np.bincount(ytr[nn]).argmax())  # majority vote
    return np.array(out)

def confusion_matrix(y, p, K):
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C

def macro_f1(C):
    tp = np.diag(C).astype(float)
    prec = tp / np.maximum(C.sum(0), 1)
    rec = tp / np.maximum(C.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return f1.mean()

def spec_augment(spec, rng, n_time=2, T_max=20, n_freq=2, F_max=10, fill=0.0):
    out = spec.copy()
    T, F = spec.shape
    for _ in range(n_freq):
        f = int(rng.integers(0, F_max + 1)); f0 = int(rng.integers(0, F - f + 1))
        out[:, f0:f0 + f] = fill
    for _ in range(n_time):
        t = int(rng.integers(0, T_max + 1)); t0 = int(rng.integers(0, T - t + 1))
        out[t0:t0 + t, :] = fill
    return out
```

Libraries: `torchaudio.transforms.TimeMasking`/`FrequencyMasking`, `torch_audiomentations` for waveform augmentation, `sklearn.neighbors.KNeighborsClassifier`, `sklearn.metrics.classification_report`, and Hugging Face `transformers` (`ASTForAudioClassification`) for pretrained backbones.

## Common misconceptions

- **"Accuracy is enough."** Under imbalance it hides failing classes; always report per-class recall.
- **"Augment everything."** A transformation that changes the label (pitch shift for song identification) is noise injection in the wrong place.
- **"A CNN on spectrograms is an image classifier, so ImageNet tricks transfer unchanged."** Random crops in frequency, rotations and flips change meaning; time shifts and masking are the safe ones.
- **"Higher $k$ is always smoother and better."** It trades variance for bias; in worked example 1 $k=5$ was no better than $k=3$ on 40 test clips (and 40 clips give a standard error of several percent, so differences of one or two clips are noise).
- **"Test accuracy of a random split of clips estimates deployment accuracy."** Not when clips from one recording are on both sides.

## Key takeaways

- Fixed-size summaries (mean and std over time of MFCCs) plus cosine k-NN are a strong baseline for small datasets; CNNs on `(1,T,M)` log-mels with global pooling handle variable length.
- Output size of a convolution: $\lfloor(i+2p-k)/s\rfloor+1$; parameter count $C_{in}C_{out}k^2+C_{out}$.
- Augmentations must preserve the label and mimic real distortions; SpecAugment masks random time and frequency bands; mixup blends inputs and labels.
- Evaluate with the confusion matrix, per-class recall and macro F1; avoid leakage between train and test.
- In practice start from a pretrained backbone and train a head.

## Explain it yourself

1. A colleague augments a bird-song classifier with a pitch shift of up to 6 semitones. Argue for and against.
2. Your 10-class model has 96% accuracy but two classes with 20% recall. Explain how that can happen and what you would report.

## Further reading

- Repo phase 06 lesson 03 (backbone choices and the 2026 stack); the lesson cites Gong et al. (AST), Chen et al. (BEATs), Park et al. (SpecAugment) and Piczak (ESC-50), none of which we read.
- Hershey et al. (2016) is cited in the French course for the image-CNN comparison on audio; not read directly.
- [[aud.04]] shows how the same encoders handle variable-length outputs.

## Mastery criteria

You can build clip-level MFCC statistics, implement cosine k-NN, compute a confusion matrix, per-class recall and macro F1 by hand and in NumPy, derive the output size and parameter count of a small CNN, implement SpecAugment-style masking and mixup, and name an augmentation that is unsafe for a given task.
