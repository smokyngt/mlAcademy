---
id: aud.02
title: "Spectrogrammes, échelle mel et MFCC"
summary: "Transformer une STFT en caractéristiques réellement utilisées par les modèles de parole et d'audio : spectrogrammes de puissance logarithmiques, bancs de filtres mel construits comme une matrice, MFCC via la DCT et coefficients delta, le tout implémenté à partir de zéro."
minutes: 80
teaches: [aud.mel, aud.mfcc]
requires: [aud.stft, aud.waveform, la.matrix, alg.exp-log]
sources:
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/02-spectrograms-mel-features/docs/en.md (log-magnitude, mel formula, triangular filterbank as one matmul, log-mel, MFCC via DCT type II, 80 mels default, pitfalls). READ."
  - "ai-engineering-from-scratch/phases/06-speech-and-audio/01-audio-fundamentals/docs/en.md (STFT recap used as the input of this lesson). READ."
  - "math/coursIntroApprentissageAudio.pdf, PDF p.30-35 (spectrogram vs sonagramme, mel scale formulas m = 2595 log10(1+f/700) and its inverse, the four MFCC steps). READ."
  - "math/coursIntroApprentissageAudio.pdf, PDF p.40, 49-53 (model families: mel-spectrogram, raw-window and raw-sample models; spectrograms as sequences or as images; summary). READ."
---

## Où cela s'inscrit

[[aud.01]] s'est terminée sur la STFT : une matrice complexe de forme `(trames, bins)`. Pour un extrait à 16 kHz avec `n_fft=512`, cela fait 257 nombres toutes les 10 ms, et la plupart sont inutiles pour la tâche : les bins voisins portent presque la même information, les bins de haute fréquence sont minuscules comparés aux bas, et la hauteur exacte d'une voix est rarement ce que l'on veut reconnaître. Les reconnaisseurs de parole ([[aud.04]]), les classifieurs de sons ([[aud.03]]) et les modèles de synthèse vocale ([[aud.05]]) compressent donc la STFT en un *spectrogramme log-mel*, typiquement 80 nombres par trame, et parfois compressent encore davantage en une treizaine de *MFCC*. Cette leçon construit cette compression pas à pas et implémente chaque étape.

## L'idée en termes simples

Imagine que tu puisses peindre le son comme une image : le temps de gauche à droite, la hauteur de bas en haut, la luminosité pour l'intensité. L'image brute de la STFT est trop détaillée et trop déséquilibrée, pour trois raisons, chacune ayant un remède simple.

1. **L'intensité couvre des plages énormes.** Un chuchotement et un cri diffèrent d'un facteur de plusieurs milliers en énergie, et l'oreille juge l'intensité à peu près sur une échelle logarithmique. Nous montrons donc le *logarithme* de l'énergie.
2. **L'oreille n'est pas une règle.** Nous distinguons facilement 300 Hz de 600 Hz, mais à peine 10 000 Hz de 10 300 Hz : des *rapports* égaux de fréquence sonnent comme des pas égaux, des *différences* égales non. Nous comprimons donc l'axe des hauteurs pour que les basses fréquences reçoivent un détail fin et que les hautes fréquences soient regroupées. Cet axe déformé est l'**échelle mel**.
3. **Les nombres voisins sont redondants.** Les canaux adjacents d'un spectre lisse évoluent ensemble. Une transformée finale réécrit un spectre comme « un contour lisse plus des ondulations » et nous permet de ne garder que le contour. Le résultat, ce sont les **MFCC**, un vecteur court qui décrit la *forme* du spectre dans chaque trame, en gros la forme de la bouche et de la gorge qui produisent le son, tout en ignorant les fines ondulations dues à la hauteur de la voix.

Enfin, comme la parole est affaire de *changement*, nous ajoutons à chaque trame un nombre indiquant la vitesse à laquelle chaque caractéristique évolue : les caractéristiques **delta**.

## Spectrogrammes de puissance et logarithmiques

Étant donnée la STFT $X[t,k]$ de [[aud.01]] :

- spectrogramme de module $|X[t,k]|$,
- **spectrogramme de puissance** $P[t,k]=|X[t,k]|^2$ (énergie),
- **spectrogramme logarithmique** $\log(P[t,k]+\varepsilon)$ ou en décibels $10\log_{10}(P+\varepsilon)$.

La petite constante $\varepsilon$ (par ex. $10^{-10}$) évite $\log 0$ pour les bins silencieux. Prendre le logarithme comprime une dynamique de cinq ou six ordres de grandeur dans une plage qu'un réseau peut gérer, et transforme « deux fois plus fort » (une multiplication) en « plus une constante » (une addition). Les systèmes de production diffèrent par leurs conventions (log naturel, dB, écrêtage, normalisation) ; la leçon du dépôt avertit que les confondre entre l'entraînement et l'inférence est un bug silencieux, donc note la convention par écrit.

## L'échelle mel

L'échelle mel associe une fréquence $f$ en hertz à une coordonnée de hauteur perceptive $m$ :

$$
m(f)=2595\,\log_{10}\!\left(1+\frac{f}{700}\right),\qquad f(m)=700\left(10^{m/2595}-1\right).
$$

L'inverse s'obtient en résolvant pour $f$ : $m/2595=\log_{10}(1+f/700)$, donc $1+f/700 = 10^{m/2595}$. (La même courbe s'écrit aussi $1127\ln(1+f/700)$, car $2595/\ln 10\approx1127$.) La constante 700 est l'endroit où la courbe change de nature : pour $f\ll700$ on a $\ln(1+f/700)\approx f/700$, presque linéaire ; pour $f\gg700$ elle croît comme $\ln f$, logarithmique.

De combien l'axe est-il comprimé ? La pente est

$$
\frac{dm}{df}=\frac{2595}{700\ln 10}\cdot\frac{1}{1+f/700}\approx\frac{1.61}{1+f/700}\ \text{mel/Hz}.
$$

À $f=0$, un hertz vaut 1,61 mel ; à 1000 Hz, $0.663$ mel ; à 8000 Hz, seulement $0.130$ mel. Les basses fréquences reçoivent environ douze fois plus du nouvel axe par hertz que le haut de la plage 0-8 kHz.

Valeurs de référence (reproduites dans l'exercice) : $m(700)=781.17$, $m(1000)=999.99$ (l'échelle est calibrée pour que 1000 Hz vaille environ 1000 mel), $m(4000)=2146.07$, $m(8000)=2840.02$. C'est l'une des nombreuses variantes de l'échelle mel ; librosa propose aussi la version « Slaney » (linéaire en dessous de 1 kHz) ; les caractéristiques calculées avec des variantes différentes ne sont pas interchangeables.

## Le banc de filtres mel comme matrice

Nous voulons $M$ nombres par trame, chacun résumant l'énergie d'une bande de hauteur, avec des bandes de largeur égale *sur l'axe mel*.

**Construction** (pour $M$ filtres, taille de FFT $N$, fréquence d'échantillonnage $f_s$, plage $[f_{\min},f_{\max}]$) :

1. Prends $M+2$ points équidistants en mel entre $m(f_{\min})$ et $m(f_{\max})$, reconvertis-les en hertz : $f_0<f_1<\dots<f_{M+1}$.
2. Le filtre $j$ ($j=1..M$) est un triangle qui monte de $0$ en $f_{j-1}$ à $1$ en $f_j$ et redescend à $0$ en $f_{j+1}$. Évalue-le à la fréquence centrale $\nu_k=k f_s/N$ de chaque bin de DFT $k=0..N/2$ :

$$
H_j(\nu_k)=\max\!\left(0,\ \min\!\left(\frac{\nu_k-f_{j-1}}{f_j-f_{j-1}},\ \frac{f_{j+1}-\nu_k}{f_{j+1}-f_j}\right)\right).
$$

Les triangles voisins se recouvrent à moitié, donc chaque fréquence entre $f_1$ et $f_M$ est partagée entre deux filtres.

3. Empile les $H_j(\nu_k)$ dans une matrice $\mathbf H\in\mathbb R^{M\times(N/2+1)}$. Le spectrogramme mel est alors un unique produit matriciel :

$$
\mathbf M_{\text{spec}}=\mathbf P\,\mathbf H^\top,\qquad
\mathbf P\in\mathbb R^{T\times(N/2+1)},\ \ \mathbf H^\top\in\mathbb R^{(N/2+1)\times M},\ \ \mathbf M_{\text{spec}}\in\mathbb R^{T\times M}.
$$

et le **spectrogramme log-mel** est $\log(\mathbf M_{\text{spec}}+\varepsilon)$, l'entrée des reconnaisseurs de type Whisper (80 mels) et de nombreux classifieurs audio (la leçon du dépôt cite 128 mels pour la classification ; ces choix sont des conventions, pas des lois).

**Exemple détaillé 1 : un minuscule banc de filtres.** $M=8$ filtres, $N=512$, $f_s=16$ kHz, plage 0-8 kHz. Les $M+2=10$ fréquences-frontières (Hz) sont

$$
0,\ 226.2,\ 525.5,\ 921.5,\ 1445.4,\ 2138.6,\ 3055.9,\ 4269.5,\ 5875.3,\ 8000 .
$$

Les distances entre frontières consécutives passent de $226$ Hz à $2125$ Hz : les filtres sont étroits dans les graves et larges dans les aigus. La matrice `H` a la forme `(8, 257)`. Le filtre 3 (indice 2 en comptant depuis 0) s'étend de 525,5 à 1445,4 Hz et culmine à 921,5 Hz. Un son pur à 1000 Hz ($N=512$, puissance fenêtrée de Hann) place l'essentiel de son énergie dans ce filtre : les 8 énergies mel sont `[0, 0, 20891.8, 3684.2, 0, 0, 0, 0]` (arrondies), donc l'argmax est l'indice 2.

**Exemple détaillé 2 : un chirp.** Pour une sinusoïde dont la fréquence monte linéairement de 200 Hz à 4000 Hz en 1 s ($x(t)=\sin(2\pi(200t+1900t^2))$) échantillonnée à 16 kHz, avec `n_fft=512, hop=160` et un banc de 40 filtres, l'indice mel de l'argmax aux trames 0, 10, 20, ..., 90 est `[4, 10, 14, 17, 20, 22, 24, 26, 27, 29]`. La fréquence a augmenté d'une quantité constante par trame, mais l'indice mel monte vite au début puis ralentit : une droite en hertz devient une courbe qui s'aplatit en mel.

## Coefficients cepstraux : les MFCC

La recette classique (donnée aussi dans le cours en français, qui liste les quatre mêmes étapes) : fenêtrer et appliquer la transformée de Fourier ; projeter le spectre de puissance sur l'échelle mel ; prendre le log de chaque énergie mel ; puis appliquer une **transformée en cosinus discrète (DCT)** au vecteur des énergies log-mel et garder ses premiers coefficients. Les coefficients sont les **coefficients cepstraux en fréquences mel (MFCC)**.

### La DCT, démontrée

Soit $y\in\mathbb R^{M}$ le vecteur log-mel d'une trame. La DCT-II l'exprime comme une somme de cosinus de « sinuosité » croissante :

$$
c_k=\sum_{m=0}^{M-1} y_m\cos\!\left(\frac{\pi k\,(2m+1)}{2M}\right),\qquad k=0,\dots,M-1 .
$$

Le cosinus $k$ effectue $k/2$ périodes complètes sur les $M$ canaux : $k=0$ est une constante (le niveau moyen), $k=1$ est une demi-onde lente (une pente), les $k$ plus grands oscillent plus vite. Avec les facteurs d'échelle $\sqrt{2/M}$ pour $k\ge1$ et $\sqrt{1/M}$ pour $k=0$, les lignes de la matrice de transformation $\mathbf D\in\mathbb R^{M\times M}$ deviennent orthonormées :

$$
D_{km}=\alpha_k\cos\!\left(\frac{\pi k(2m+1)}{2M}\right),\quad \alpha_0=\sqrt{\tfrac1M},\ \alpha_{k>0}=\sqrt{\tfrac2M},\qquad \mathbf D\mathbf D^\top=\mathbf I .
$$

L'orthonormalité signifie que $\mathbf D$ est une rotation du vecteur $y$ (elle conserve les longueurs, et l'inverse est la transposée), donc garder les $M$ coefficients ne perd rien, et garder les $K$ premiers est la meilleure approximation *lisse* dans cette base :

$$
\hat y=\mathbf D_{[:K]}^\top\,\mathbf D_{[:K]}\,y,\qquad \mathbf D_{[:K]}\in\mathbb R^{K\times M}.
$$

**Pourquoi des cosinus et non des exponentielles complexes ?** Une DCT est la DFT du signal reflété autour de son bord, donc il n'y a pas de saut artificiel là où le vecteur « se referme sur lui-même » ; l'énergie se concentre dans les premiers coefficients et ce sont des nombres réels.

**Pourquoi cela aide.** Le vecteur log-mel est la somme d'une composante lente (l'enveloppe spectrale, façonnée par le conduit vocal) et d'une ondulation rapide (les harmoniques de la hauteur de la voix, plus du bruit). Comme un produit de deux choses devient une somme après le log, une transformée qui sépare le lent du rapide sépare les deux causes. Garder les 13 premiers coefficients conserve l'enveloppe et écarte l'essentiel de l'ondulation, et les coefficients sont bien moins corrélés entre eux que les canaux mel voisins. (Cet argument source/filtre est l'explication classique ; les réseaux modernes alimentés par des log-mels bruts égalent souvent les MFCC, donc les MFCC survivent surtout dans les petits modèles, les systèmes de locuteurs et les références de comparaison.)

**Exemple détaillé 3 : DCT et troncature.** $\mathbf D$ pour $y=(1,2,3,4)$ donne $c=(5,\,-2.2304,\,0,\,-0.1585)$ : une rampe rectiligne est presque entièrement une constante (5 = somme/2) plus une demi-onde lente. Sur la trame 40 du chirp ci-dessus (26 canaux mel), reconstruire le vecteur log-mel à partir des $K$ premiers coefficients donne une erreur RMS de $4.96$ (K=3), $3.32$ (K=6), $0.70$ (K=13) et exactement $0$ (K=26). Treize coefficients préservent presque tout le contour.

### Caractéristiques delta

Une trame est un instantané ; la parole se définit par des transitions (une occlusive est une explosion *suivie d'*une voyelle). Ajoute à chaque coefficient sa pente locale dans le temps. Ajuste une droite par moindres carrés sur les $2N+1$ valeurs $c_{t-N},\dots,c_{t+N}$ ; la pente est

$$
\Delta c_t=\frac{\sum_{n=-N}^{N} n\,c_{t+n}}{\sum_{n=-N}^{N} n^2}
=\frac{\sum_{n=1}^{N} n\,(c_{t+n}-c_{t-n})}{2\sum_{n=1}^{N} n^2}.
$$

(Le numérateur apparie $+n$ et $-n$ ; le dénominateur utilise $\sum_{n=-N}^{N}n^2=2\sum_{n=1}^N n^2$.) Appliquer la même formule à $\Delta c$ donne les caractéristiques $\Delta\Delta$ (accélération). Aux bords de la séquence, répète la première et la dernière trame (remplissage `mode='edge'`). Un vecteur de caractéristiques classique est 13 MFCC + 13 $\Delta$ + 13 $\Delta\Delta$ = 39 nombres par trame, de forme `(T, 39)`.

**Vérification.** Pour une rampe $c_t=t$ (pente 1), le delta vaut exactement 1 aux trames intérieures, et pour $c_t=3t$ il vaut 3 ; pour $c_t=t^2$ avec $N=2$ il vaut $2t$ : par ex. $t=2,3$ donnent $4.0$ et $6.0$ dans l'exercice (où `[0,1,4,9,16,25]` donne des deltas `[0.9, 2.2, 4.0, 6.0, 5.8, 4.1]` ; la première et la dernière valeur sont déformées par le remplissage des bords).

## Implémentation

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

### Formes

| Étape | Forme |
|---|---|
| forme d'onde | `(L,)` |
| spectrogramme de puissance `P` | `(T, N/2+1)` |
| banc de filtres `H` | `(M, N/2+1)` |
| log-mel | `(T, M)` |
| matrice DCT `D` | `(K, M)` |
| MFCC | `(T, K)` |
| MFCC + $\Delta$ + $\Delta\Delta$ | `(T, 3K)` |

**Ce que les bibliothèques font différemment.** `librosa.feature.melspectrogram` et `torchaudio.transforms.MelSpectrogram` utilisent le même pipeline mais *normalisent généralement l'aire* de chaque triangle (pour que les larges filtres de haute fréquence ne reçoivent pas plus d'énergie), proposent les variantes mel Slaney ou HTK, et prennent trame/saut en échantillons. Whisper fournit son propre banc de filtres et sa propre normalisation. Réutiliser les poids de quelqu'un d'autre, c'est réutiliser exactement ses caractéristiques.

## Idées reçues courantes

- **« Mel signifie que les filtres ont la même largeur. »** Ils ont la même largeur en mel, ce qui est de plus en plus large en hertz ; nos triangles non normalisés ont tous une hauteur de 1, donc les filtres plus larges somment plus d'énergie.
- **« Les MFCC sont juste des spectrogrammes compressés. »** Ce sont des spectres d'un *spectre* : l'indice du coefficient est lié à la vitesse d'ondulation le long de la fréquence, pas à une fréquence.
- **« Plus de bins mel, c'est toujours mieux. »** Avec `n_fft=400`, il n'y a que 201 bins ; demander 128 mels laisse les filtres les plus bas plus étroits qu'un bin, produisant des canaux vides ou dupliqués.
- **« Je peux comparer directement mes caractéristiques à celles d'une bibliothèque. »** La variante de fenêtre, la variante mel, la normalisation des filtres, la base du log et l'écrêtage diffèrent tous.
- **« c0 est une caractéristique de hauteur. »** C'est le niveau global d'énergie (logarithmique) ; beaucoup de systèmes l'écartent ou le gardent séparément.

## Points clés à retenir

- La puissance logarithmique comprime la dynamique ; l'échelle mel $m=2595\log_{10}(1+f/700)$ dépense la résolution là où l'audition le fait.
- Le banc de filtres mel est une matrice fixe `(M, N/2+1)`, donc le spectrogramme mel est un seul produit matriciel $\mathbf P\mathbf H^\top$.
- La DCT-II orthonormée est une rotation du vecteur log-mel ; la tronquer conserve le contour spectral lisse. Vérifie $\mathbf D\mathbf D^\top=\mathbf I$.
- Les deltas sont des pentes locales par moindres carrés : $\sum n(c_{t+n}-c_{t-n})/(2\sum n^2)$.
- Note toujours `sr`, `n_fft`, le saut, la fenêtre, `n_mels`, la convention de log et la normalisation : un décalage entre entraînement et inférence échoue silencieusement.

## Explique-le toi-même

1. Pourquoi une droite parfaite de fréquence (un chirp) paraît-elle courbe sur un axe mel ? Utilise la dérivée $dm/df$.
2. Explique pourquoi garder seulement les 13 premiers coefficients DCT d'un vecteur log-mel supprime l'ondulation causée par la hauteur de la voix mais conserve le contour du conduit vocal.

## Pour aller plus loin

- Dépôt, phase 06, leçon 02 (choix de caractéristiques par tâche et pièges).
- Davis et Mermelstein (1980) ont introduit les MFCC pour la reconnaissance de la parole ; la leçon du dépôt le cite (nous n'avons pas lu l'article lui-même).
- [[aud.03]] utilise les statistiques de MFCC construites ici comme entrée d'un classifieur.

## Critères de maîtrise

Tu sais convertir entre hertz et mel, construire sans bibliothèque une matrice de banc de filtres mel triangulaires, montrer que la matrice DCT est orthonormée, calculer MFCC et deltas avec des formes explicites, et expliquer quels choix (fenêtre, variante mel, `n_mels`, log) doivent coïncider entre l'entraînement et le déploiement.
