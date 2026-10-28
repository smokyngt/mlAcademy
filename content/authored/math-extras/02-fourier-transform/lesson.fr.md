---
id: mx.02
title: La transformée de Fourier
summary: "Écris n'importe quel signal échantillonné comme une somme d'ondes tournantes, dérive la transformée de Fourier discrète à partir de l'orthogonalité, vois pourquoi la FFT est rapide, et apprends ce que signifient résolution, repliement spectral et convolution dans le domaine fréquentiel."
minutes: 80
teaches: [mx.dft, mx.sampling, mx.conv-theorem]
requires: [mx.roots-unity, la.matrix]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/docs/en.md (DFT definition, coefficient meaning, FFT idea, frequency resolution, convolution theorem, spectrograms, aliasing, zero padding sections). READ."
  - "math/coursIntroApprentissageAudio.pdf, French slides (PDF p.13-15 waveform and continuous Fourier transform, p.19-25 sampling rate, Shannon-Nyquist statement, DFT definition; p.26 start of PCM encoding). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/19-complex-numbers/docs/en.md (roots of unity link to the DFT). READ."
---

## Où cela s'inscrit

Une seconde d'audio enregistrée à 16 000 échantillons par seconde est une liste de 16 000 nombres. À l'œil, cette liste ne te dit presque rien : est-ce une seule note, un accord, de la parole ? Les mêmes données paraissent complètement différentes si tu demandes *quelle quantité de chaque hauteur de son est présente*. Cette question reçoit sa réponse de la **transformée de Fourier**, et cette réponse est ce que consomment réellement les reconnaisseurs de parole, les modèles de musique et de nombreuses méthodes de séries temporelles (sous forme de spectrogrammes). Elle explique aussi pourquoi les convolutions des CNN peuvent être calculées rapidement, et donne un sens aux ondes sinus et cosinus des encodages positionnels sinusoïdaux.

Tu as besoin de [[mx.01]] : exponentielles complexes et racines de l'unité. Si les matrices comme changement de base te semblent floues, revois [[math.07]], car la DFT n'est rien d'autre qu'un changement de base.

## L'idée en mots simples

- Un accord de musique est fait de plusieurs sons purs joués ensemble. Ton oreille le redécoupe en notes. La transformée de Fourier est une oreille mathématique : elle répond à « quels sons purs, avec quelle intensité, et en partant de quel point de leur cycle ? ».
- Pense à une pile de **roues qui tournent** à des vitesses différentes : la roue 0 ne tourne pas (elle porte le niveau moyen), la roue 1 fait un tour sur l'enregistrement, la roue 2 en fait deux, et ainsi de suite. Pour savoir combien de la roue $k$ se trouve dans ton signal, tu voyages avec cette roue et tu fais la moyenne du signal contre elle. Si le signal contient cette rotation, elle s'additionne de façon cohérente et donne un grand nombre ; sinon, les parties positives et négatives s'annulent et tu obtiens presque zéro.
- Comme l'enregistrement a un nombre fini d'échantillons, il n'y a qu'autant de roues distinctes que d'échantillons. Rien n'est perdu : c'est la même information dans un nouveau système de coordonnées, comme décrire un point par (est, nord) ou par (distance, angle).
- L'échantillonnage a un prix. Les roues qui tournent trop vite entre deux instants d'échantillonnage *ressemblent* à des roues lentes (l'effet roue de chariot dans les films). C'est le **repliement spectral** (aliasing), et la règle qui l'évite est la règle de Nyquist.
- Calculer les moyennes naïvement coûte « nombre de roues fois nombre d'échantillons ». Une réutilisation astucieuse du travail (la **FFT**) fait baisser ce coût de façon spectaculaire.

## Définitions

Prends $N$ échantillons $x[0],\dots,x[N-1]$ (réels ou complexes), et soit

$$
\omega=e^{2\pi i/N}
$$

la racine primitive $N$-ième de l'unité de [[mx.01]]. La **transformée de Fourier discrète** (DFT) est

$$
X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-2\pi i kn/N}=\sum_{n=0}^{N-1}x[n]\,\omega^{-kn},\qquad k=0,\dots,N-1 .
$$

Chaque $X[k]$ est un nombre complexe ; $|X[k]|$ est la quantité de fréquence $k$ présente et $\arg X[k]$ est sa phase. Le cours français définit la même somme pour une série de $T$ valeurs et la présente comme la contrepartie discrète de la transformée continue $\hat s(\nu)=\int s(t)e^{-i2\pi\nu t}\,dt$ (l'idée de l'intégrale est développée dans [[mx.05]]).

### Pourquoi cette formule ? Une dérivation à partir de l'orthogonalité

Objectif : écrire $x$ comme une combinaison de $N$ ondes tournantes $e_k[n]=\omega^{kn}$ (l'onde $k$ accomplit $k$ tours sur l'enregistrement) :

$$
x[n]=\sum_{k=0}^{N-1}c_k\,\omega^{kn}.
$$

Pour trouver $c_l$, il nous faut le fait clé de [[mx.01]] : pour les entiers $j$,

$$
\sum_{n=0}^{N-1}\omega^{jn}=\begin{cases}N& j\equiv0\pmod N\\0&\text{otherwise,}\end{cases}
$$

(les racines de l'unité somment à zéro sauf si elles valent toutes 1). Multiplie les deux membres du développement par $\omega^{-ln}$ et somme sur $n$ :

$$
\sum_n x[n]\omega^{-ln}=\sum_k c_k\sum_n\omega^{(k-l)n}=\sum_k c_k\,N\,\delta_{kl}=N c_l .
$$

Donc $c_l=\frac1N X[l]$ avec $X$ défini comme ci-dessus. En réinjectant, on obtient la **DFT inverse**

$$
x[n]=\frac1N\sum_{k=0}^{N-1}X[k]\,e^{+2\pi i kn/N}.
$$

Sous forme matricielle, soit $F\in\mathbb{C}^{N\times N}$ avec $F_{kn}=\omega^{-kn}$. Alors $X=Fx$ (formes $N\times N$ fois $N$) et le fait d'orthogonalité dit que $F^{H}F=N I$, c'est-à-dire que $F/\sqrt N$ est une matrice **unitaire** : une rotation dans l'espace complexe. C'est pourquoi la transformée ne perd rien, et pourquoi l'identité de Parseval $\sum|x[n]|^2=\frac1N\sum|X[k]|^2$ est vraie.

### Ce que signifient les coefficients

- $X[0]=\sum_n x[n]$ : la somme, proportionnelle à la moyenne (la composante « DC »).
- Pour un signal **réel**, $X[N-k]=\overline{X[k]}$, car $\omega^{-(N-k)n}=\omega^{kn}=\overline{\omega^{-kn}}$. Donc seules les cases $0\ldots N/2$ portent une information indépendante.
- Un cosinus réel $A\cos(2\pi k n/N+\varphi)$ avec $0<k<N/2$ produit $X[k]=\frac{NA}{2}e^{i\varphi}$ (et son conjugué en $N-k$). Son amplitude est donc $A=2|X[k]|/N$. Un sinus a pour phase $-\pi/2$, il apparaît donc comme le nombre imaginaire $X[k]=-i\,NA/2$.

## La FFT : pourquoi elle est rapide

Calculer directement chaque $X[k]$ coûte $N$ multiplications pour chacune des $N$ sorties, soit $O(N^2)$ opérations : $N=1024$ donne $1{,}048{,}576$. La **transformée de Fourier rapide** obtient les mêmes nombres en $O(N\log N)$, environ $10{,}240$ pour $N=1024$.

L'idée de Cooley-Tukey, pour $N$ pair : sépare la somme en échantillons d'indice pair et d'indice impair, $n=2m$ et $n=2m+1$ :

$$
X[k]=\underbrace{\sum_{m}x[2m]\,e^{-2\pi i km/(N/2)}}_{E[k]}+e^{-2\pi ik/N}\underbrace{\sum_m x[2m+1]\,e^{-2\pi ikm/(N/2)}}_{O[k]} .
$$

$E$ et $O$ sont des DFT de longueur $N/2$, et elles sont périodiques en $k$ de période $N/2$. De plus $e^{-2\pi i(k+N/2)/N}=-e^{-2\pi ik/N}$. Donc, pour $k=0,\dots,N/2-1$ :

$$
X[k]=E[k]+t_kO[k],\qquad X[k+N/2]=E[k]-t_kO[k],\qquad t_k=e^{-2\pi ik/N}\ (\text{the "twiddle factor"}).
$$

Une DFT de taille $N$ devient deux DFT de taille $N/2$ plus un travail en $O(N)$. En répétant jusqu'à la taille 1, il y a $\log_2N$ niveaux, chacun coûtant $O(N)$, d'où $O(N\log N)$. Les bibliothèques (`np.fft`, `torch.fft`, FFTW, cuFFT) gèrent n'importe quel $N$ avec des factorisations plus élaborées, mais les puissances de deux sont le cas propre.

## Résolution fréquentielle, Nyquist et repliement spectral

Disons que les échantillons sont pris à une **fréquence d'échantillonnage** $f_s$ (échantillons par seconde), donc l'enregistrement dure $T=N/f_s$ secondes. La case $k$ accomplit $k$ tours en $T$ secondes, donc sa fréquence physique est

$$
f_k=\frac{k\,f_s}{N},\qquad\text{bin spacing }\Delta f=\frac{f_s}{N}=\frac1T .
$$

**La résolution dépend de la durée de l'enregistrement, pas du nombre de cases.** Pour séparer des sons espacés de $1$ Hz, il faut au moins environ une seconde de signal. Compléter le signal par des zéros avant la transformée donne un spectre d'allure plus lisse (plus de cases) mais n'ajoute pas d'information.

La fréquence la plus haute qu'un signal réel peut porter dans $N$ échantillons est la case $N/2$, c'est-à-dire $f_s/2$, la **fréquence de Nyquist**. Une fréquence $f$ au-dessus n'est pas perdue ; elle est *repliée* et apparaît à l'**alias**. Pour $f$ réduit modulo $f_s$ dans $[0,f_s)$, la fréquence apparente est $\min(f,\ f_s-f)$.

Esquisse de preuve : aux instants d'échantillonnage $t=n/f_s$, $\cos(2\pi(f+mf_s)t)=\cos(2\pi ft+2\pi mn)=\cos(2\pi ft)$. Des fréquences qui diffèrent d'un multiple de $f_s$ produisent des échantillons identiques, donc les échantillons ne peuvent pas les distinguer.

Les diapositives énoncent le théorème de Shannon-Nyquist pour un signal dont le spectre se trouve dans une bande $[f_{\min},f_{\max}]$ : échantillonner à $f_s\ge2(f_{\max}-f_{\min})$ permet une reconstruction parfaite du signal continu. Pour un audio ordinaire commençant près de $0$ Hz, c'est la règle familière $f_s\ge2f_{\max}$. (La forme à bande limitée est la plus générale ; les diapositives ajoutent que l'hypothèse de bande limitée est raisonnable pour les fréquences audibles.) Conséquence typique : la musique à $44.1$ kHz couvre les fréquences jusqu'à $22.05$ kHz, un peu au-dessus de l'audition humaine ; les modèles de parole utilisent souvent $16$ kHz, donc tout ce qui est au-dessus de $8$ kHz est écarté ou doit être filtré *avant* l'échantillonnage.

Un effet voisin est la **fuite spectrale** : si le signal ne contient pas un nombre entier de périodes dans la fenêtre, son énergie se répand sur les cases voisines. Multiplier par une fenêtre lisse (Hann, etc.) avant la transformée réduit la fuite.

## Le théorème de convolution

La **convolution circulaire** de deux signaux de longueur $N$ est $(a\circledast b)[n]=\sum_{m}a[m]\,b[(n-m)\bmod N]$. Prenons la DFT des deux membres :

$$
\operatorname{DFT}(a\circledast b)[k]=\sum_n\sum_m a[m]b[(n-m)\bmod N]\omega^{-kn}.
$$

Substitue $n=m+p$ (indices modulo $N$) : $\omega^{-kn}=\omega^{-km}\omega^{-kp}$, donc la double somme se factorise en $A[k]\,B[k]$ :

$$
\boxed{\operatorname{DFT}(a\circledast b)=\operatorname{DFT}(a)\cdot\operatorname{DFT}(b)\quad\text{(pointwise product).}}
$$

Ainsi la convolution, une opération en $O(N^2)$, coûte $O(N\log N)$ via FFT, multiplication, FFT inverse. Une couche de « convolution » de CNN avec un grand noyau peut être implémentée ainsi (les frameworks utilisent des méthodes directes ou de Winograd pour les petits noyaux, où c'est plus rapide). De plus, la vue fréquentielle d'un filtre n'est que sa DFT : les filtres passe-bas gardent les petits $k$.

## Spectrogrammes (aperçu)

Une seule DFT d'un enregistrement entier dit *quelles* fréquences apparaissent mais pas *quand*. La **transformée de Fourier à court terme** découpe le signal en fenêtres qui se chevauchent, disons de $25$ ms, multiplie chacune par une fonction de fenêtre et prend la DFT de chacune. Empiler les modules donne une matrice de forme (trames x cases de fréquence), le **spectrogramme**, une image que les réseaux audio traitent comme une photo. La longueur de la fenêtre échange résolution temporelle contre résolution fréquentielle ($\Delta f=1/T_{\text{window}}$), un compromis d'incertitude, pas un bug.

## Exemple travaillé

Vérifié dans le code de l'exercice.

1. Signal $x=(1,2,3,4,0,0,0,0)$, $N=8$. Alors $X[0]=10$ (la somme), et $|X|=(10,\ 7.254,\ 2.828,\ 2.715,\ 2,\ 2.715,\ 2.828,\ 7.254)$. L'entrée est réelle, donc c'est symétrique : $|X[N-k]|=|X[k]|$.
2. $N=16$ échantillons de $x[n]=\sin(2\pi\cdot3n/16)+0.5\cos(2\pi\cdot5n/16)$. La transformée a exactement quatre entrées non nulles : $X[3]=-8i$, $X[13]=+8i$ (le sinus : $N\cdot1/2=8$, imaginaire), $X[5]=X[11]=4$ (le cosinus : $16\cdot0.5/2=4$, réel).
3. Résolution : $f_s=8000$ Hz, $N=1024$ donne $\Delta f=7.8125$ Hz.
4. Repliement : échantillonne $\sin(2\pi\cdot7t)$ à $f_s=10$ Hz. Les échantillons sont exactement ceux de $-\sin(2\pi\cdot3t)$ : le son à 7 Hz est au-dessus de la limite de Nyquist de 5 Hz et apparaît à $10-7=3$ Hz (avec un changement de signe).
5. Convolution : $a=(1,2,3,0)$, $b=(1,1,0,0)$ : la convolution circulaire est $(1,3,5,3)$, égale à la FFT inverse du produit des FFT.

## Implémentation

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

## Formes (shapes)

- Entrée $x$ : $(N,)$ ; sortie $X$ : $(N,)$ complexe. Pour une entrée réelle, `np.fft.rfft` ne renvoie que les $N/2+1$ cases non redondantes.
- Matrice DFT $F$ : $(N,N)$, dense et complexe : 16 octets par entrée, c'est pourquoi personne ne la construit pour de grands $N$.
- Spectrogramme : (nombre de trames, $N_{\text{win}}/2+1$) après avoir pris les modules. Les images en deux dimensions utilisent la FFT 2D : applique la transformée 1D le long de chaque axe.

## Idées fausses courantes

- « Plus de zéros de remplissage signifie une meilleure résolution. » Cela ne fait qu'interpoler. La résolution vaut $1/T$ du signal réel.
- « La FFT est une transformée différente. » C'est la même sortie que la DFT, calculée plus vite.
- « Les fréquences négatives sont une erreur. » Pour les signaux réels, elles reflètent les positives ; pour les signaux complexes (données I/Q, réseaux à valeurs complexes), elles portent de l'information nouvelle.
- « Le repliement peut être supprimé après coup. » Une fois échantillonnés, un son à 7 Hz et un son à 3 Hz sont identiques dans les échantillons (comme montré plus haut). Filtre avant d'échantillonner.
- « Le spectre de modules dit toute l'histoire. » La phase porte l'information de timing/forme ; l'écarter (comme le font les spectrogrammes de modules) est un choix de modélisation qui a des coûts.

## Points clés à retenir

- La DFT exprime $N$ échantillons dans la base de $N$ ondes tournantes ; l'orthogonalité des racines de l'unité donne à la fois la transformée et son inverse.
- $F/\sqrt N$ est unitaire : rien n'est perdu, l'énergie est conservée.
- La FFT sépare échantillons pairs et impairs et réutilise le travail : $O(N\log N)$ au lieu de $O(N^2)$.
- L'espacement des cases est $f_s/N=1/T$ ; les fréquences au-dessus de $f_s/2$ se replient.
- La convolution dans le temps est un produit terme à terme dans le domaine fréquentiel.

## Explique-le toi-même

1. Pourquoi deux sons différents peuvent-ils donner exactement les mêmes échantillons, et que fait un filtre anti-repliement vu par un musicien ?
2. Utilise la séparation pair/impair pour expliquer, avec des mots, pourquoi la FFT demande environ $N\log_2N$ opérations.

## Pour aller plus loin

- Leçon du dépôt `phases/01-math-foundations/20-fourier-transform` (fenêtrage, propriétés de la DFT, liens avec les encodages positionnels et les CNN).
- Diapositives audio `coursIntroApprentissageAudio.pdf`, p.19-26 (échantillonnage) et la suite sur les réseaux à base de spectrogrammes.
- Ensuite dans ce cours : [[mx.03]] applique les vecteurs propres aux graphes ; ses vecteurs propres du laplacien sur un graphe en anneau sont exactement ces ondes de Fourier.

## Critères de maîtrise

- [ ] Je sais dériver la DFT inverse à partir de l'orthogonalité des racines de l'unité.
- [ ] Je sais calculer l'espacement des cases et la fréquence de Nyquist d'un enregistrement et prédire l'alias d'un son donné.
- [ ] Je sais écrire la FFT récursive et expliquer l'étape du facteur de rotation (twiddle).
- [ ] Je sais énoncer le théorème de convolution et dire quand la convolution par FFT vaut le coup.
