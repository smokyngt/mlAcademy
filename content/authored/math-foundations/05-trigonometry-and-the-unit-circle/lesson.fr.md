---
id: mf.05
title: "Trigonométrie et cercle unité"
summary: "Angles et radians, sinus, cosinus et tangente à partir du cercle unité, les identités que tu réutiliseras (dont les matrices de rotation), et les ondes périodiques, porte d'entrée vers l'analyse de Fourier."
minutes: 95
teaches: [mf.trig]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/03-matrix-transformations/docs/en.md (The Problem, 'Transformations as matrices', and the Rotation section): rotation matrix R(theta) and its role in ML. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/docs/en.md (The Problem, DFT definition and coefficient meanings, window formulas, connection to positional encodings): frequency, amplitude, phase and the sinusoidal position encoding formula. READ."
---

## Où cela s'inscrit

Tout ce qui tourne, se répète ou oscille utilise la trigonométrie. En apprentissage automatique, cela inclut : la rotation d'images et de nuages de points pour l'augmentation de données, les matrices de rotation dans les modèles géométriques, les **encodages positionnels sinusoïdaux** qui indiquent à un transformer où se trouve chaque mot, et la **transformée de Fourier** qui analyse l'audio, les images et les séries temporelles en ondes. À l'école, la trigonométrie ressemble à un tas de formules de triangles ; ici, nous la réorganisons autour d'une seule image, le cercle unité, et alors chaque formule devient quelque chose que tu peux voir.

Cette leçon ne demande aucune leçon antérieure. Tu utilises les idées sur les fonctions de `mf.03` (période, translation et changement d'échelle d'un graphe) et les habitudes de code de `mf.01`, mais chacune est réexpliquée quand c'est nécessaire.

## L'idée en mots simples

Imagine une grande roue de rayon 1 qui tourne à vitesse constante, et une lampe très haut au-dessus qui projette une ombre. Pendant que la roue tourne, la **hauteur** d'un passager monte, atteint le sommet, redescend, passe sous le centre, et revient : une onde lisse. La **position horizontale** du passager fait la même onde, mais un quart de tour en retard. Ces deux ondes sont le *sinus* et le *cosinus*. Tout le reste est de la comptabilité sur l'endroit où se trouve le passager.

- Un **angle** dit jusqu'où le passager a parcouru la roue. Nous le mesurons soit en degrés (un tour complet vaut 360), soit en **radians** (un tour complet vaut $2\pi$) : le radian est la *distance parcourue le long de la jante*, ce qui est l'unité naturelle car sur un cercle de rayon 1 la distance est égale à l'angle.
- **Cosinus** de l'angle : à quelle distance à droite se trouve le passager. **Sinus** : à quelle hauteur.
- **Tangente** : à quel point est raide la droite qui va du centre au passager (hauteur divisée par la distance à droite).
- **Périodique** : après un tour complet, le passager est revenu là où il avait commencé, donc les ondes se répètent indéfiniment.
- **Rotation** : si tu tournes toute l'image d'un certain angle, un point de coordonnées $(x,y)$ se déplace vers un point dont les nouvelles coordonnées sont des mélanges de sinus et de cosinus de cet angle.

## Angles et radians

Un cercle complet vaut $360^\circ$ ou $2\pi$ radians. La circonférence d'un cercle de rayon $r$ est $2\pi r$ ; sur le **cercle unité** ($r=1$), un trajet de longueur d'arc $\theta$ correspond à un angle de $\theta$ radians. Conversions :

$$
\text{radians}=\text{degrees}\times\frac{\pi}{180},\qquad \text{degrees}=\text{radians}\times\frac{180}{\pi}.
$$

Repères : $30^\circ=\pi/6$, $45^\circ=\pi/4$, $60^\circ=\pi/3$, $90^\circ=\pi/2$, $180^\circ=\pi$, $270^\circ=3\pi/2$, $360^\circ=2\pi$. Les angles se mesurent dans le sens inverse des aiguilles d'une montre à partir de l'axe horizontal positif ; les angles négatifs vont dans le sens des aiguilles. Des angles qui diffèrent de $2\pi$ décrivent la même direction.

Pourquoi des radians en mathématiques et en code ? Des formules comme la pente de $\sin$ ne deviennent propres qu'en radians, et chaque bibliothèque (`np.sin`, `math.cos`) attend des radians. L'oublier est le bug de trigonométrie le plus courant : `np.sin(90)` ne vaut pas 1, car cela signifie 90 radians.

## Sinus, cosinus, tangente

**Définition par le cercle unité.** Pour un angle $\theta$, soit $(x,y)$ le point du cercle unité à cet angle. Alors

$$
\cos\theta=x,\qquad \sin\theta=y,\qquad \tan\theta=\frac{y}{x}=\frac{\sin\theta}{\cos\theta}.
$$

**Vue par le triangle rectangle.** Pour un angle aigu dans un triangle rectangle : $\sin=\frac{\text{opposite}}{\text{hypotenuse}}$, $\cos=\frac{\text{adjacent}}{\text{hypotenuse}}$, $\tan=\frac{\text{opposite}}{\text{adjacent}}$ (côté opposé, côté adjacent, hypoténuse). Cela concorde avec le cercle car un triangle d'hypoténuse 1 inscrit dans le cercle a pour côté adjacent $x$ et pour côté opposé $y$.

**Valeurs à connaître.**

| $\theta$ | 0 | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ |
|---|---|---|---|---|---|
| $\sin\theta$ | 0 | $1/2$ | $\sqrt2/2\approx0.707$ | $\sqrt3/2\approx0.866$ | 1 |
| $\cos\theta$ | 1 | $\sqrt3/2$ | $\sqrt2/2$ | $1/2$ | 0 |

Pourquoi $\sin(\pi/6)=\frac12$ ? Prends un triangle équilatéral de côté 1 et coupe-le en deux : tu obtiens un triangle rectangle d'angles $30^\circ,60^\circ,90^\circ$, d'hypoténuse 1, et de petit côté $\frac12$, opposé à l'angle de $30^\circ$. Les autres valeurs viennent du triangle rectangle isocèle à $45^\circ$ (les deux côtés sont égaux, donc $2x^2=1$, $x=\sqrt2/2$).

**Signes par quadrant.** Droite/haut est positif. Dans le deuxième quadrant ($\pi/2$ à $\pi$), le point est à gauche et en haut : $\cos<0,\sin>0$. Dans le troisième, les deux sont négatifs ; dans le quatrième, $\cos>0,\sin<0$. Les symétries découlent des miroirs : $\sin(-\theta)=-\sin\theta$ (impaire), $\cos(-\theta)=\cos\theta$ (paire), $\sin(\theta+\pi)=-\sin\theta$.

**La tangente** n'est pas définie là où $\cos\theta=0$ ($\theta=\pi/2+k\pi$), où le rayon est vertical et sa pente infinie.

## L'identité de Pythagore

Le point $(\cos\theta,\sin\theta)$ se trouve sur le cercle unité $x^2+y^2=1$. Donc, pour tout angle,

$$
\sin^2\theta+\cos^2\theta=1 .
$$

Usage : si tu sais que $\cos\theta=0.6$ et que $\theta$ est dans le premier quadrant, alors $\sin\theta=\sqrt{1-0.36}=0.8$. Vérifie numériquement avec le code plus bas.

## Addition d'angles et matrices de rotation

Faisons tourner un point de $\theta$ dans le sens inverse des aiguilles d'une montre autour de l'origine. Commence avec les deux points de base $(1,0)$ et $(0,1)$ :

- $(1,0)$ est à l'angle 0 ; tourné de $\theta$, il est à l'angle $\theta$ sur le cercle unité : $(\cos\theta,\sin\theta)$.
- $(0,1)$ est à l'angle $\pi/2$ ; tourné de $\theta$, il se trouve à l'angle $\theta+\pi/2$, qui est le point $(-\sin\theta,\cos\theta)$ (le point précédent tourné d'un quart de tour de plus).

Une rotation traite un point général $(x,y)=x(1,0)+y(0,1)$ de façon linéaire, donc son image est $x(\cos\theta,\sin\theta)+y(-\sin\theta,\cos\theta)$. En notation matricielle :

$$
R(\theta)\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}x\cos\theta-y\sin\theta\\ x\sin\theta+y\cos\theta\end{pmatrix}.
$$

Vérification : $\theta=\pi/2$ envoie $(1,0)$ sur $(0,1)$ et $(0,2)$ sur $(-2,0)$. La rotation conserve la distance à l'origine (en élevant au carré : $(x\cos\theta-y\sin\theta)^2+(x\sin\theta+y\cos\theta)^2=x^2+y^2$ grâce à l'identité de Pythagore).

**Addition d'angles, dérivée.** Tourner de $b$ puis de $a$ doit être égal à tourner de $a+b$. Applique les matrices à $(1,0)$ : tourner de $b$ donne $(\cos b,\sin b)$, puis tourner de $a$ donne pour première coordonnée $\cos a\cos b-\sin a\sin b$ et pour seconde coordonnée $\sin a\cos b+\cos a\sin b$. Mais tourner $(1,0)$ de $a+b$ donne $(\cos(a+b),\sin(a+b))$. Égalise :

$$
\cos(a+b)=\cos a\cos b-\sin a\sin b,\qquad \sin(a+b)=\sin a\cos b+\cos a\sin b .
$$

Vérification numérique avec $a=\pi/4$, $b=\pi/6$ : $\sin(75^\circ)=0.9659258$ des deux côtés. La vue matricielle est celle qui compte pour le ML : les rotations se composent en multipliant des matrices. Dans certains modèles de langage, les positions sont encodées en faisant tourner des paires de coordonnées, et l'identité d'addition d'angles est la raison pour laquelle les décalages relatifs en ressortent proprement.

## Fonctions périodiques et ondes

$\sin$ et $\cos$ se répètent tous les $2\pi$ : $\sin(\theta+2\pi)=\sin\theta$. Une onde générale est

$$
y(t)=A\sin(2\pi ft+\varphi).
$$

- **Amplitude** $A$ : la hauteur des crêtes.
- **Fréquence** $f$ : cycles par unité de temps. La **période** est $T=1/f$, puisque l'argument du sinus avance de $2\pi$ quand $t$ avance de $1/f$.
- **Phase** $\varphi$ : décale l'onde latéralement (une translation du graphe, comme dans `mf.03` ; l'onde est retardée de $\varphi/(2\pi f)$ unités de temps).

Ainsi $\sin(3x)$ a pour période $2\pi/3$, et $\sin(2\pi\cdot5t)$ accomplit cinq cycles par unité de temps. De plus, $\cos\theta=\sin(\theta+\pi/2)$ : le cosinus est un sinus décalé d'un quart de période.

## Pourquoi l'analyse de Fourier existe

La grande idée (développée dans une leçon ultérieure) : tout signal périodique raisonnable est une somme d'ondes de ce type, de fréquences différentes. Pour trouver combien de la fréquence $k$ est cachée dans $N$ échantillons $x_n$ sur une unité de temps, corrèle le signal avec un cosinus et un sinus de cette fréquence : si le signal contient une onde de cette fréquence, les produits s'alignent et les sommes sont grandes, sinon les parties positives et négatives s'annulent. Pour une onde pure $A\sin(2\pi kn/N)$, l'estimation d'amplitude $\frac2N\sqrt{(\sum x_n\cos)^2+(\sum x_n\sin)^2}$ vaut $A$. C'est la transformée de Fourier discrète déguisée (le matériel du dépôt l'exprime avec des nombres complexes $e^{-2\pi ikn/N}$, qui regroupent un cosinus et un sinus ensemble).

Un usage voisin : le transformer original encode la position $pos$ avec les paires $\sin(pos/10000^{2i/d})$ et $\cos(pos/10000^{2i/d})$ : de nombreuses paires sinus/cosinus à des fréquences différentes, de sorte que chaque position a une empreinte distincte et que des positions proches ont des empreintes semblables.

## Exemple travaillé en code

Chaque nombre est reproduit par les fonctions de l'exercice.

```python
import numpy as np

print(np.sin(np.pi / 6), np.cos(np.pi / 6), np.tan(np.pi / 6))
# 0.49999999999999994 0.8660254037844387 0.5773502691896257   (float rounding: 0.5, sqrt(3)/2, 1/sqrt(3))

def R(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

print(R(np.pi / 2) @ np.array([1, 0]))     # [6.1e-17 1.0]  i.e. (0, 1) up to rounding
print(R(np.pi / 4) @ np.array([1, 0]))     # [0.7071 0.7071]
print(np.hypot(3, 4), np.arctan2(4, 3))    # 5.0 0.9272952180016122

N = 256
t = np.arange(N) / N
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
amp = lambda k: 2 / N * np.hypot(x @ np.cos(2*np.pi*k*t), x @ np.sin(2*np.pi*k*t))
print(round(amp(5), 6), round(amp(3), 6), round(amp(4), 6))   # 1.0 0.5 0.0
```

Le « 6.1e-17 » est du bruit de virgule flottante pour un vrai 0 ; compare avec `np.allclose`, jamais avec `==`.

## Idées fausses courantes

- « `np.sin(90)` vaut 1. » Les entrées sont en radians. Convertis : `np.sin(np.deg2rad(90))`.
- « $\sin(a+b)=\sin a+\sin b$. » Faux ; utilise la formule d'addition.
- « $\sin^2\theta$ signifie $\sin(\theta^2)$. » Cela signifie $(\sin\theta)^2$.
- « $\sin^{-1}$ est $1/\sin$. » La notation $\sin^{-1}$ (aussi appelée arcsin) est la fonction réciproque, qui renvoie un angle ; $1/\sin\theta$ s'appelle la cosécante.
- « $\tan$ est bornée. » Elle explose en $\pi/2+k\pi$.
- « Une rotation de $\theta$ change les longueurs. » Non ; c'est une isométrie.
- Mélanger l'ordre des entrées de la matrice de rotation : le signe moins est en haut à droite.

## Points clés à retenir

- Les radians mesurent la longueur d'arc sur le cercle unité ; $2\pi$ est un tour complet.
- $(\cos\theta,\sin\theta)$ est le point du cercle unité ; d'où $\sin^2+\cos^2=1$ et $\tan=\sin/\cos$.
- Les matrices de rotation conservent la longueur et se composent en additionnant les angles ; les formules d'addition sont cet énoncé écrit en toutes lettres.
- $A\sin(2\pi ft+\varphi)$ a pour amplitude $A$, fréquence $f$, période $1/f$ et phase $\varphi$.
- L'analyse de Fourier demande combien de chaque fréquence un signal contient ; les encodages positionnels sinusoïdaux utilisent de nombreuses fréquences à la fois.

## Explique-le toi-même

1. Pourquoi a-t-on forcément $\sin^2\theta+\cos^2\theta=1$ ? Argumente uniquement à partir du cercle unité.
2. Explique pourquoi tourner de $b$ puis de $a$ revient à tourner de $a+b$, et comment cela donne la formule de $\sin(a+b)$.

## Pour aller plus loin

- Leçons ultérieures : transformations matricielles (rotations, homothéties), nombres complexes et transformée de Fourier, et encodages positionnels des transformers.

## Critères de maîtrise

Tu sais : convertir entre degrés et radians ; lire $\sin$, $\cos$, $\tan$ sur le cercle unité pour les angles usuels, signes compris ; utiliser $\sin^2+\cos^2=1$ ; construire une matrice de rotation $2\times2$ et l'appliquer ; énoncer la période, l'amplitude, la fréquence et la phase de $A\sin(2\pi ft+\varphi)$ ; expliquer avec des mots ce que mesure un coefficient de Fourier.
