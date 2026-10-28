---
id: mx.01
title: Les nombres complexes pour le machine learning
summary: "Des nombres dotés d'une rotation intégrée, la formule d'Euler démontrée à partir des séries de Taylor, les racines de l'unité, et pourquoi une matrice de rotation a des valeurs propres complexes."
minutes: 60
teaches: [mx.complex, mx.euler, mx.roots-unity]
requires: [la.linear-map, la.eigen, alg.exp-log, calc.derivative]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/19-complex-numbers/docs/en.md (concept section: arithmetic, polar form, Euler's formula, roots of unity, links to DFT and RoPE; section list of Build It). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/docs/en.md (DFT definition, used only to see where complex numbers lead next). READ."
---

## Où cela s'inscrit

Tôt ou tard, un article te montre un `i`. La transformée de Fourier ([[mx.02]]) s'écrit avec lui, les plongements de position rotatifs (RoPE) des transformers sont décrits comme des rotations de nombres complexes, et les valeurs propres de certaines matrices parfaitement réelles se révèlent complexes. Dans [[math.07]], tu as vu qu'une rotation de 90 degrés du plan n'a pas de vecteur propre réel. Cette leçon explique quelles sont ses valeurs propres et pourquoi.

Tu n'as pas besoin des nombres complexes pour faire du ML au quotidien, mais ils sont le langage le plus propre pour **tout ce qui tourne ou oscille**. Après cette leçon, tu pourras lire `e^{i\theta}` comme « le point d'angle theta sur le cercle unité » et voir la transformée de Fourier comme un changement de base plutôt que comme un tour de magie.

## L'idée en mots simples

- Les nombres réels vivent sur une droite. Multiplier par $-1$ retourne la droite autour de zéro : c'est un demi-tour. Demandons maintenant : existe-t-il un nombre dont le carré vaut $-1$, c'est-à-dire un « quart de tour » qui, fait deux fois, donne un demi-tour ? Sur une droite, il n'y en a pas, alors les mathématiciens ajoutent une seconde direction. La nouvelle unité pointant « vers le haut » s'appelle $i$, et les nombres deviennent des **points d'un plan**.
- Le nom « imaginaire » est un accident historique. Rien dans $i$ n'est moins réel qu'un nombre négatif ; c'est simplement « tourne d'un quart de cercle ».
- Un point du plan peut être décrit par deux déplacements (à droite et en haut) ou par un seul déplacement dans une direction : « va aussi loin, avec cet angle ». Le premier est la description cartésienne, le second la description polaire. Additionner est facile avec la première ; multiplier est facile avec la seconde, parce que **multiplier par un nombre complexe signifie tourner et étirer**.
- Un nombre de longueur 1 ne fait que tourner. Les nombres de longueur 1 forment le cercle unité, et la formule d'Euler dit que la fonction exponentielle, alimentée par un pur angle de rotation, parcourt ce cercle.
- Si tu découpes un cercle en $n$ parts égales à partir du point 1, les frontières des parts sont les **racines de l'unité**. Ce sont les fréquences d'échantillonnage de la transformée de Fourier.

## Définitions

L'**unité imaginaire** $i$ est définie par $i^2=-1$. Un **nombre complexe** est $z=a+bi$ avec $a$ réel (partie réelle) et $b$ (partie imaginaire). L'ensemble de tous ces nombres est $\mathbb{C}$, identifié au plan $\mathbb{R}^2$ via $z\leftrightarrow(a,b)$.

L'arithmétique suit l'algèbre ordinaire plus $i^2=-1$ :

$$
(a+bi)+(c+di)=(a+c)+(b+d)i,
$$

$$
(a+bi)(c+di)=ac+adi+bci+bd\,i^2=(ac-bd)+(ad+bc)i .
$$

Le **conjugué** de $z=a+bi$ est $\bar z=a-bi$ (image miroir par rapport à l'axe réel). Le produit $z\bar z=a^2+b^2$ est réel et non négatif, ce qui définit le **module** $|z|=\sqrt{a^2+b^2}$ (distance à l'origine). La division s'en sert : $\dfrac{1}{z}=\dfrac{\bar z}{|z|^2}$.

L'**argument** (angle) est $\theta=\operatorname{atan2}(b,a)$, donc $z=r(\cos\theta+i\sin\theta)$ avec $r=|z|$. C'est la **forme polaire**.

### Une matrice cachée à l'intérieur

Écris $z=a+bi$ sous la forme de la matrice

$$
M_z=\begin{pmatrix}a&-b\\ b&a\end{pmatrix}\qquad(\text{shape }2\times2).
$$

Multiplier $M_z$ par la colonne $(c,d)^\top$ donne $(ac-bd,\ bc+ad)^\top$, exactement le produit $z\cdot(c+di)$. Donc la multiplication complexe **est** une famille particulière de matrices $2\times2$ : des matrices de rotation fois un facteur d'échelle. Avec $r=|z|$ et $a=r\cos\theta$, $b=r\sin\theta$ :

$$
M_z=r\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}.
$$

## Dérivation : la formule d'Euler à partir des séries de Taylor

Nous voulons un sens pour $e^{i\theta}$. L'exponentielle a pour série (qui converge pour tout $x$ réel ou complexe)

$$
e^{x}=\sum_{k=0}^{\infty}\frac{x^k}{k!}=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots
$$

Utilise cette série comme **définition** pour $x=i\theta$. Les puissances de $i$ se répètent avec une période de 4 : $i^0=1,\ i^1=i,\ i^2=-1,\ i^3=-i,\ i^4=1,\dots$ Donc

$$
e^{i\theta}=1+i\theta-\frac{\theta^2}{2!}-i\frac{\theta^3}{3!}+\frac{\theta^4}{4!}+i\frac{\theta^5}{5!}-\cdots
$$

Regroupe les termes sans $i$ (puissances paires) et avec un $i$ (puissances impaires) :

$$
e^{i\theta}=\underbrace{\Big(1-\frac{\theta^2}{2!}+\frac{\theta^4}{4!}-\cdots\Big)}_{\cos\theta}+i\underbrace{\Big(\theta-\frac{\theta^3}{3!}+\frac{\theta^5}{5!}-\cdots\Big)}_{\sin\theta}.
$$

Les deux parenthèses sont les séries de Taylor du cosinus et du sinus (la série du cosinus n'a que des puissances paires aux signes alternés ; celle du sinus seulement des puissances impaires). Donc

$$
\boxed{e^{i\theta}=\cos\theta+i\sin\theta.}
$$

Conséquences :

- $|e^{i\theta}|^2=\cos^2\theta+\sin^2\theta=1$ : il se trouve toujours sur le cercle unité.
- $e^{i\pi}=-1$, c'est-à-dire $e^{i\pi}+1=0$.
- Un nombre complexe général est $z=re^{i\theta}$, et la règle du produit n'est que la règle des exposants $e^{x}e^{y}=e^{x+y}$ :

$$
r_1e^{i\theta_1}\cdot r_2e^{i\theta_2}=r_1r_2\,e^{i(\theta_1+\theta_2)}.
$$

**On multiplie les longueurs, on additionne les angles.** Multiplier par $e^{i\varphi}$ fait tourner chaque point de $\varphi$ ; multiplier par un réel $r>0$ le met à l'échelle. Combiné avec la vue matricielle, cela explique pourquoi $M_{e^{i\varphi}}$ est une matrice de rotation.

Lire les parties réelle et imaginaire du produit démontre aussi gratuitement les identités d'addition d'angles : $\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$ et ainsi de suite.

## Racines de l'unité

Les solutions de $z^n=1$ sont les **racines $n$-ièmes de l'unité**. Écris $z=e^{i\theta}$ ; alors $e^{in\theta}=1$ exige $n\theta=2\pi k$, donc

$$
\omega_k=e^{2\pi i k/n},\qquad k=0,1,\dots,n-1.
$$

Ce sont les sommets d'un $n$-gone régulier sur le cercle unité. En notant $\omega=\omega_1$, on a $\omega_k=\omega^k$. Deux faits que tu utiliseras dans [[mx.02]] :

1. **Leur somme est nulle** pour $n\ge2$ : $\sum_{k=0}^{n-1}\omega^k=\dfrac{1-\omega^n}{1-\omega}=0$ (série géométrique, avec $\omega^n=1$ et $\omega\ne1$). Géométriquement, le polygone est équilibré autour de l'origine.
2. **Orthogonalité.** Pour $\omega^j$, la somme $\sum_{n}\omega^{jn}$ vaut $n$ si $j\equiv0 \pmod n$ et $0$ sinon. Les vecteurs $(\omega^{jm})_{m}$ pour différents $j$ sont donc orthogonaux (pour le produit scalaire complexe, qui conjugue un facteur).

## Valeurs propres complexes des rotations

Prends la matrice de rotation $R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}$. D'après [[math.07]], les valeurs propres résolvent $\lambda^2-\operatorname{tr}(R)\lambda+\det R=0$. Ici $\operatorname{tr}=2\cos\theta$ et $\det=\cos^2\theta+\sin^2\theta=1$ :

$$
\lambda^2-2\cos\theta\,\lambda+1=0\ \Longrightarrow\ \lambda=\cos\theta\pm\sqrt{\cos^2\theta-1}=\cos\theta\pm i\sin\theta=e^{\pm i\theta}.
$$

Pour $\theta=90^\circ$, les valeurs propres sont $\pm i$. C'est cohérent avec l'histoire : aucune direction réelle ne survit à une rotation (donc pas de vecteur propre réel), mais sur les nombres complexes il y en a un, avec une valeur propre qui est elle-même « une rotation d'angle $\theta$ ». Les vecteurs propres correspondants sont $(1,\mp i)^\top$ (à un facteur près). Deux leçons pour le ML :

- **Des valeurs propres de module 1 signifient une rotation pure ; un module supérieur ou inférieur à 1 signifie une spirale vers l'extérieur ou l'intérieur.** Pour un réseau récurrent $h_{t+1}=Wh_t$, les valeurs propres de $W$ avec $|\lambda|>1$ explosent et celles avec $|\lambda|<1$ s'évanouissent. Une paire complexe avec $|\lambda|=1$ oscille éternellement.
- Les matrices réelles ayant des valeurs propres complexes les ont toujours par paires conjuguées $\lambda,\bar\lambda$ (le polynôme caractéristique a des coefficients réels).

## Exemple travaillé

Tous ces nombres proviennent du code de l'exercice.

1. $z=3+4i$ : $|z|=5$, $\theta=\operatorname{atan2}(4,3)=0.9273$ rad $=53.13^\circ$.
2. $(1+2i)(3-i)=3-i+6i-2i^2=3+5i+2=5+5i$. Modules : $|1+2i|\,|3-i|=\sqrt5\sqrt{10}=\sqrt{50}=7.0711$, et $|5+5i|=\sqrt{50}$ aussi. Les angles s'additionnent : $\operatorname{atan2}(2,1)+\operatorname{atan2}(-1,3)=1.1071-0.3217=0.7854=45^\circ$, l'angle de $5+5i$.
3. Fais tourner $1+i$ de $90^\circ$ : multiplie par $i$, ce qui donne $i+i^2=-1+i$. Un quart de tour du point $(1,1)$ est $(-1,1)$. ✓
4. Sommes partielles de Taylor de $e^{i\pi/3}=0.5+0.8660i$. Avec $T$ termes, l'erreur est $0.5318$ ($T=2$), $0.0494$ ($T=4$), $0.0018$ ($T=6$) et $4\times10^{-7}$ ($T=10$). La factorielle au dénominateur fait converger la série très vite.
5. Racines quatrièmes de l'unité : $1,\ i,\ -1,\ -i$. Racines sixièmes : $1,\ 0.5\pm0.866i,\ -0.5\pm0.866i,\ -1$.
6. $R_{60^\circ}$ a pour valeurs propres $0.5\pm0.866i=e^{\pm i\pi/3}$ ; $R_{90^\circ}$ a $\pm i$.

## Implémentation

```python
import numpy as np

def to_polar(z):
    return abs(z), np.arctan2(z.imag, z.real)

def euler_series(theta, terms):
    # partial sum of the Taylor series of exp(i*theta)
    total, term = 0j, 1 + 0j
    for k in range(terms):
        total += term
        term = term * (1j * theta) / (k + 1)
    return total

def roots_of_unity(n):
    return np.exp(2j * np.pi * np.arange(n) / n)

def rotation_eigenvalues(theta):
    tr, det = 2 * np.cos(theta), 1.0
    disc = np.sqrt(complex(tr * tr - 4 * det))
    return (tr + disc) / 2, (tr - disc) / 2
```

En pratique, NumPy a des types complexes natifs (`complex128`), `np.abs`, `np.angle`, `np.exp`, et `np.linalg.eig` renvoie des valeurs propres complexes quand c'est nécessaire. PyTorch gère les tenseurs complexes (`torch.view_as_real` convertit en une dimension finale de taille 2), ce qui est la façon dont les implémentations de RoPE font souvent tourner des paires de coordonnées.

## Formes (shapes)

- Un scalaire complexe, c'est deux réels. Un vecteur complexe de longueur $n$, c'est $2n$ réels.
- Les vecteurs de $\mathbb{C}^n$ utilisent la **transposée conjuguée** $x^{H}=\bar x^\top$ pour les produits scalaires : $\langle x,y\rangle=x^{H}y$. Les transposées simples donnent de mauvaises longueurs ($z^2$ peut être négatif, $\bar zz$ ne le peut pas).
- Une rotation d'angle $\theta$ agissant sur $n/2$ paires de coordonnées d'un vecteur réel de longueur $n$ est une multiplication complexe d'un $n/2$-vecteur par $e^{i\theta_j}$ composante par composante.

## Idées fausses courantes

- « Les nombres imaginaires ne sont pas réels. » Ce sont une coordonnée sur un plan. Tout ce qui est ici peut se faire avec des matrices $2\times2$.
- « $\sqrt{ab}=\sqrt a\sqrt b$ toujours. » Cela échoue pour les nombres négatifs : $\sqrt{(-1)(-1)}=1$ mais $\sqrt{-1}\sqrt{-1}=-1$.
- « L'angle est unique. » L'argument n'est défini qu'à $2\pi$ près ; $\operatorname{atan2}$ en choisit un dans $(-\pi,\pi]$.
- « $|z|^2=z^2$. » L'identité correcte est $|z|^2=z\bar z$.
- « Une matrice réelle a des valeurs propres réelles. » Garanti seulement pour les matrices symétriques ([[math.07]]).

## Points clés à retenir

- $z=a+bi=re^{i\theta}$ : cartésien pour additionner, polaire pour multiplier.
- La multiplication additionne les angles et multiplie les longueurs, donc les nombres de module 1 sont des rotations.
- $e^{i\theta}=\cos\theta+i\sin\theta$ découle de la série de l'exponentielle en séparant puissances paires et impaires.
- Les racines $n$-ièmes de l'unité sont $e^{2\pi ik/n}$ ; leur somme est nulle et elles sont mutuellement orthogonales en tant que vecteurs.
- Une rotation réelle d'angle $\theta$ a pour valeurs propres $e^{\pm i\theta}$.

## Explique-le toi-même

1. Explique à un ami pourquoi multiplier par $i$ est un quart de tour, et pourquoi cela rend $i^2=-1$ inévitable.
2. Explique pourquoi la série de Taylor de $e^{i\theta}$ se sépare en cosinus et sinus, et ce que cela dit des cercles.

## Pour aller plus loin

- Leçon du dépôt `phases/01-math-foundations/19-complex-numbers` (liens avec RoPE et les encodages positionnels sinusoïdaux).
- Ensuite : [[mx.02]] utilise les racines de l'unité comme base de la transformée de Fourier.

## Critères de maîtrise

- [ ] Je sais convertir entre $a+bi$ et $re^{i\theta}$ et multiplier dans l'une ou l'autre forme.
- [ ] Je sais démontrer la formule d'Euler sans notes.
- [ ] Je sais lister les racines $n$-ièmes de l'unité et expliquer pourquoi leur somme est nulle.
- [ ] Je sais calculer les valeurs propres d'une rotation $2\times2$ et interpréter leur module.
