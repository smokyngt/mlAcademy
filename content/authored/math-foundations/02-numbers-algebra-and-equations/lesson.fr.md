---
id: mf.02
title: "Nombres, algèbre et équations"
summary: "Reconstruis l'algèbre du collège avec soin : types de nombres, ordre des opérations, puissances et racines, fractions et pourcentages, équations et inéquations linéaires, systèmes de deux équations, et lecture de la notation mathématique."
minutes: 100
teaches: [mf.algebra]
requires: []
sources:
  - "math/math for ML.pdf, Ch. 1 Introduction and Motivation (PDF p.17): why the mathematics is needed (data, model, learning). READ."
  - "math/math for ML.pdf, Ch. 2 intro (PDF p.24): vectors as tuples in R^n, scalars, and how that notation corresponds to arrays on a computer. READ."
  - "math/math for ML.pdf, Sec. 2.1 Systems of Linear Equations (PDF p.26-27): each equation is a line, the solution set is an intersection, and the possible outcomes are none, one or infinitely many. READ."
  - "math/math for ML.pdf, Sec. 5 intro (PDF p.145-146): the f: R^D -> R, x |-> f(x) notation for functions, domain and image. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md (opening sections): vectors as lists of numbers with meaning, and why notation matters. READ."
---

## Où cela s'inscrit

Chaque formule de l'apprentissage automatique est construite à partir de quelques gestes que tu connais déjà à moitié : additionner, multiplier, élever à une puissance, trouver une inconnue. La prédiction d'un modèle comme $\hat y = w x + b$ n'est qu'une équation linéaire. « Entraîner le modèle » signifie souvent « résoudre une équation (ou un couple d'équations) pour trouver les inconnues $w$ et $b$ ». Le problème, c'est que l'algèbre du collège est généralement enseignée comme des recettes sans compréhension, et qu'ensuite une notation comme $\mathbb R$, $\sum$ ou $f:\mathbb R\to\mathbb R$ ressemble à une autre langue.

Cette leçon est la reconstruction. Nous avançons lentement et vérifions chaque affirmation avec un petit morceau de code. Aucune leçon antérieure n'est nécessaire, sauf la capacité de `mf.01` à lire du court Python (si elle est fragile, parcours-la d'abord).

## L'idée en mots simples

Une équation est une **balance à deux plateaux**. Le plateau de gauche et celui de droite pèsent la même chose. Le seul geste légal est de faire *la même chose aux deux plateaux* : ajouter 5 aux deux, couper les deux en deux. Tant que la balance reste équilibrée, tu as le droit de simplifier jusqu'à ce que l'inconnue se retrouve seule d'un côté. Résoudre une équation, c'est *défaire*, dans l'ordre inverse, tout ce qui a été fait à l'inconnue. Si l'histoire est « j'ai multiplié par 3, puis ajouté 5, et j'ai obtenu 20 », le défaire est « soustrais 5, puis divise par 3 ».

Une inéquation est la même balance, mais penchée : le plateau de gauche est *plus lourd* que celui de droite. Faire la même chose aux deux plateaux la garde penchée du même côté, avec une exception célèbre que nous expliquerons : si tu inverses le signe des deux plateaux (multiplication par un négatif), « gauche plus lourd » devient « gauche plus léger ».

Deux inconnues demandent deux balances. Chaque balance donne un indice ; un seul indice ne suffit pas pour déterminer deux nombres, deux indices indépendants si.

## Types de nombres

| symbole | nom | exemples |
|---|---|---|
| $\mathbb N$ | nombres naturels | 0, 1, 2, 3, ... (certains livres commencent à 1) |
| $\mathbb Z$ | entiers relatifs | ..., -2, -1, 0, 1, 2, ... |
| $\mathbb Q$ | nombres rationnels (fractions $p/q$) | 1/2, -7/3, 0.25 |
| $\mathbb R$ | nombres réels (toute la droite numérique) | $\sqrt 2$, $\pi$, -0.5 |

Chaque famille contient la précédente : $\mathbb N\subset\mathbb Z\subset\mathbb Q\subset\mathbb R$. Pourquoi s'embêter à étendre ? Parce que chaque étape permet de résoudre quelque chose qu'on ne pouvait pas avant : $x+3=1$ demande $\mathbb Z$ (réponse $-2$), $2x=1$ demande $\mathbb Q$, et $x^2=2$ demande $\mathbb R$ ($\sqrt2$ n'est pas une fraction). Dans cette académie, « un nombre » sans précision signifie un nombre réel. (Plus tard, les nombres `complexes` apparaîtront pour $x^2=-1$.)

En code, `int` contient exactement les nombres entiers, et `float` contient les nombres réels de façon approchée (environ 16 chiffres). Souviens-toi : `0.1 + 0.2 != 0.3` exactement en virgule flottante.

## Ordre des opérations

Des conventions existent pour que tout le monde lise $2+3\times4$ de la même façon. L'ordre est :

1. les parenthèses, les plus internes d'abord ;
2. les puissances et les racines ;
3. la multiplication et la division, de gauche à droite ;
4. l'addition et la soustraction, de gauche à droite.

De petits exemples travaillés (tous vérifiables en Python avec `**` pour la puissance) :

- $2+3\times4 = 2+12 = 14$ (et non 20).
- $(2+3)\times4 = 20$.
- $2+3\times4^2$ : la puissance d'abord, $4^2=16$ ; puis $3\times16=48$ ; puis $2+48=50$.
- $-3^2$ vaut $-(3^2)=-9$, mais $(-3)^2=9$. Les parenthèses décident.
- $12/3\times2$ : de gauche à droite, $4\times2=8$ (et non $12/6=2$).

```python
print(2 + 3 * 4 ** 2)      # 50
print(-3 ** 2, (-3) ** 2)  # -9 9
print(12 / 3 * 2)          # 8.0
```

## Puissances et racines

Une puissance est une multiplication répétée : $a^n = a\times a\times\dots\times a$ ($n$ copies), donc $2^5=32$. Trois règles que tu peux dériver en comptant les copies :

- $a^m\cdot a^n = a^{m+n}$ : ($m$ copies) fois ($n$ copies) font $m+n$ copies. Exemple : $2^3\cdot2^2=8\cdot4=32=2^5$.
- $(a^m)^n = a^{mn}$ : $n$ groupes de $m$ copies. $(2^3)^2=64=2^6$.
- $a^0=1$ et $a^{-n}=1/a^n$ : ces définitions sont *choisies* pour que la première règle continue de fonctionner. Puisque $a^{3}\cdot a^{0}$ doit valoir $a^{3+0}=a^3$, le nombre $a^0$ doit être 1. Puisque $a^3\cdot a^{-3}=a^0=1$, il faut $a^{-3}=1/a^3$.

Une **racine** défait une puissance : $\sqrt{9}=3$ parce que $3^2=9$ (et par convention $\sqrt{\ }$ désigne la racine non négative). Les exposants fractionnaires sont des racines : $a^{1/2}=\sqrt a$, car $(a^{1/2})^2=a^{1}$ d'après la deuxième règle. Donc $49^{1/2}=7$ et $8^{1/3}=2$. Les exposants et les logarithmes ont leur propre leçon (`mf.04`).

## Fractions et pourcentages

Une fraction $\frac ab$ signifie « $a$ parts de taille $\frac1b$ ». Pour additionner des fractions, les parts doivent avoir la même taille : trouve un **dénominateur commun**.

$$
\frac ab+\frac cd=\frac{ad+bc}{bd}.
$$

Dérivation : $\frac ab=\frac{ad}{bd}$ et $\frac cd=\frac{bc}{bd}$ (multiplier le haut et le bas par le même nombre ne change pas la valeur), et maintenant les deux sont comptées en parts de taille $\frac1{bd}$. Exemple : $\frac12+\frac13=\frac36+\frac26=\frac56$. Faux (mais tentant) : $\frac{1+1}{2+3}=\frac25$.

La multiplication est plus facile : $\frac ab\cdot\frac cd=\frac{ac}{bd}$ (« une part d'une part »). La division inverse : $\frac ab\div\frac cd=\frac ab\cdot\frac dc$. Réduis en divisant le haut et le bas par leur plus grand diviseur commun (`math.gcd`).

Un **pourcentage** est une fraction sur 100. « $p\%$ de $x$ » vaut $\frac p{100}x$. Changer une quantité de $p\%$ la multiplie par $1+\frac p{100}$ (pour une hausse) ou $1-\frac p{100}$ (pour une baisse).

Exemple travaillé (vérifié plus bas) : un prix de 100 augmente de 20 %, puis baisse de 20 %. Après la hausse : $100\times1.2=120$. Après la baisse : $120\times0.8=96$. Tu ne **reviens pas** à 100, car la baisse de 20 % s'applique au nombre plus grand, 120. La variation en pourcentage de `old` à `new` est $\frac{\text{new}-\text{old}}{\text{old}}\times100$ ; de 120 à 96, elle vaut $-24/120\times100=-20\%$.

## Résoudre des équations linéaires

Une équation est **linéaire** en $x$ si $x$ n'apparaît que multiplié par des constantes et additionné : $ax+b=c$.

Dérivation de la solution générale, un geste équilibré par ligne :

$$
\begin{aligned}
ax + b &= c\\
ax &= c - b \qquad &&\text{(subtract } b \text{ from both sides)}\\
x &= \frac{c-b}{a} \qquad &&\text{(divide both sides by } a\neq 0\text{)}
\end{aligned}
$$

Exemple : $3x+5=20\Rightarrow3x=15\Rightarrow x=5$. Vérifie par substitution : $3\cdot5+5=20$. Vérifie toujours : remplacer par la solution est le détecteur de bugs le moins cher des mathématiques.

Cas particuliers à reconnaître : $2x+4=2x+7$ se simplifie en $4=7$, faux pour tout $x$ (**aucune solution**) ; $2x+4=2(x+2)$ se simplifie en $4=4$, vrai pour tout $x$ (**une infinité de solutions**). Si $a=0$, la formule diviserait par zéro, et c'est exactement là que surviennent ces cas dégénérés.

**Inconnues des deux côtés :** $5x-2=3x+8$. Rassemblons : soustrais $3x$ des deux côtés, $2x-2=8$ ; ajoute 2, $2x=10$ ; donc $x=5$.

## Inéquations

Les symboles $<,>,\le,\ge$ obéissent à la même règle des gestes équilibrés, avec un piège. Résolvons $-2x+3<9$ :

1. Soustrais 3 des deux côtés : $-2x<6$.
2. Divise par $-2$. Diviser par un nombre négatif inverse l'ordre, donc le signe **se retourne** : $x>-3$.

Pourquoi ? Prends $1<2$. Multiplie les deux par $-1$ : on obtient $-1$ et $-2$, et maintenant $-1>-2$. Sur la droite numérique, prendre l'opposé met tout en miroir. Vérifie avec un nombre : $x=0$ (qui est $>-3$) donne $-0+3=3<9$, vrai ; $x=-5$ donne $10<9$, faux. L'ensemble des solutions est un intervalle : $(-3,\infty)$.

## Deux équations, deux inconnues

Exemple : $x+y=10$ et $x-y=2$. **Élimination :** additionner les équations annule $y$ : $2x=12$, donc $x=6$, puis $y=10-6=4$. Vérifie les deux équations : $6+4=10$ et $6-4=2$.

Système général $a_1x+b_1y=c_1$, $a_2x+b_2y=c_2$. Multiplie la première par $b_2$ et la seconde par $b_1$ :

$$
a_1b_2x+b_1b_2y=c_1b_2,\qquad a_2b_1x+b_1b_2y=c_2b_1 .
$$

Soustrais : $(a_1b_2-a_2b_1)x=c_1b_2-c_2b_1$. Donc

$$
x=\frac{c_1b_2-c_2b_1}{a_1b_2-a_2b_1},\qquad y=\frac{a_1c_2-a_2c_1}{a_1b_2-a_2b_1}.
$$

(La formule de $y$ s'obtient de la même façon, en éliminant $x$.) Le dénominateur $a_1b_2-a_2b_1$ ne doit pas valoir 0. Géométriquement, chaque équation est une droite, et la solution est l'endroit où les droites se croisent ; le dénominateur vaut 0 exactement quand les droites sont parallèles (aucune solution) ou confondues (une infinité de solutions). Le manuel énonce la même trichotomie (aucune, exactement une, ou une infinité de solutions) et utilise comme exemple le couple $4x_1+4x_2=5$, $2x_1-4x_2=1$, dont l'unique solution est $(1,\tfrac14)$ : vérifie $4+1=5$ et $2-1=1$. Cette idée, résoudre plusieurs équations linéaires ensemble, est le germe de l'algèbre linéaire, où elle devient l'équation matricielle que tu rencontreras dans les étapes suivantes.

## Lire et écrire la notation mathématique

La notation est de l'anglais comprimé (ici, du français comprimé). Apprends à la décompresser.

| écrit | lu à voix haute |
|---|---|
| $x\in\mathbb R$ | « $x$ est un élément des nombres réels » |
| $\forall x$ | « pour tout $x$ » |
| $\exists x$ | « il existe un $x$ » |
| $x_1,x_2,\dots,x_n$ | $n$ nombres différents nommés par un indice |
| $\sum_{i=1}^{n}x_i$ | « la somme des $x_i$ pour $i$ allant de 1 à $n$ » (étudiée dans `mf.06`) |
| $f:\mathbb R\to\mathbb R,\ x\mapsto x^2$ | « $f$ prend un nombre réel et donne un nombre réel, envoyant $x$ sur $x^2$ » |
| $\approx$, $\propto$, $:=$ | « approximativement », « proportionnel à », « est défini comme » |
| $\mathbb R^n$ | toutes les listes de $n$ nombres réels |

Le manuel *Mathematics for Machine Learning* écrit les fonctions exactement ainsi, par exemple $f:\mathbb R^D\to\mathbb R$, $x\mapsto f(x)$, où l'ensemble des entrées permises s'appelle le **domaine** et l'ensemble des sorties l'**image** ou codomaine. Il décrit aussi un vecteur de $\mathbb R^3$ comme un triplet de nombres et note que cela correspond à un tableau dans un ordinateur, ce qui explique pourquoi le NumPy de `mf.01` s'y prête si bien. Une lettre en gras comme $\mathbf x$ désigne généralement une liste entière de nombres, pas un seul.

Conseil de lecture : quand une formule fait peur, (1) nomme chaque symbole, (2) remplace-les par de tout petits nombres, (3) code-la.

## Exemple travaillé, avec du code

Chaque nombre ci-dessous est reproduit par les fonctions de l'exercice.

```python
import math
print(2 + 3 * 4 ** 2)                      # 50
print((20 - 5) / 3)                        # 5.0       (3x + 5 = 20)
print(100 * 1.2 * 0.8)                     # 96.0      (up 20%, down 20%)
print((1*3 + 1*2, 2*3), math.gcd(5, 6))    # (5, 6) 1   -> 1/2 + 1/3 = 5/6

# system 2x + 3y = 13 and x - y = -1
a1, b1, c1, a2, b2, c2 = 2, 3, 13, 1, -1, -1
det = a1*b2 - a2*b1                        # -5
print((c1*b2 - c2*b1) / det, (a1*c2 - a2*c1) / det)   # 2.0 3.0
```

Vérifie à la main : $2\cdot2+3\cdot3=13$ et $2-3=-1$. Les deux sont vraies.

## Idées fausses courantes

- $\frac ab+\frac cd=\frac{a+c}{b+d}$. Faux ; utilise un dénominateur commun.
- « +20 % puis -20 % ramène au départ. » Cela donne 96 % du départ.
- Oublier de retourner une inéquation quand on divise par un nombre négatif.
- $(a+b)^2=a^2+b^2$. Faux : $(a+b)^2=a^2+2ab+b^2$ (essaie $a=b=1$ : 4, pas 2).
- $\sqrt{a+b}=\sqrt a+\sqrt b$. Faux (essaie $9+16$ : $\sqrt{25}=5\neq3+4$).
- Faire une opération sur un seul côté d'une équation.
- Faire confiance à la réponse sans la remplacer dans l'équation.

## Points clés à retenir

- Une équation est une balance ; fais la même chose des deux côtés et défais les opérations dans l'ordre inverse.
- Ordre des opérations : parenthèses, puissances, fois/divisé, plus/moins ; un signe négatif et un exposant demandent des parenthèses pour être explicites.
- Les puissances sont des multiplications répétées ; $a^0=1$, $a^{-n}=1/a^n$, $a^{1/2}=\sqrt a$ gardent les règles des exposants cohérentes.
- Les fractions s'additionnent via des dénominateurs communs ; les variations en pourcentage se multiplient.
- Équation linéaire : $x=(c-b)/a$. Diviser une inéquation par un négatif la retourne. Deux droites se coupent en un point, sauf si elles sont parallèles ou identiques.
- La notation est du langage comprimé ; décode-la symbole par symbole.

## Explique-le toi-même

1. Explique à un ami pourquoi une hausse de 20 % suivie d'une baisse de 20 % te laisse en dessous de ton point de départ.
2. Pourquoi multiplier une inéquation par $-1$ l'inverse-t-il ? Utilise une droite numérique.

## Pour aller plus loin

- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* (les conventions de notation des chapitres 1-2 ; nous y revenons à partir de `mf.03`).
- N'importe quel manuel d'algèbre de collège pour s'exercer davantage à factoriser et à développer des parenthèses.

## Critères de maîtrise

Tu sais : évaluer une expression dans le bon ordre ; additionner et réduire des fractions ; calculer des variations en pourcentage et les composer ; résoudre $ax+b=c$ et une inéquation linéaire (avec le retournement) ; résoudre un système 2x2 par élimination et reconnaître les cas sans solution ou à une infinité de solutions ; lire à voix haute $f:\mathbb R\to\mathbb R$, $\in$, $\forall$, $\exists$.
