---
id: mf.03
title: "Fonctions et graphes"
summary: "Considère une fonction comme une machine entrée-sortie, lis son graphe, rencontre les fonctions linéaires, quadratiques et polynomiales, et apprends à translater, étirer, réfléchir, composer et inverser des fonctions, en vérifiant tout numériquement."
minutes: 100
teaches: [mf.functions]
requires: []
sources:
  - "math/math for ML.pdf, Sec. 2.1 (PDF p.26-28): each linear equation is a line, intersections of lines; used for the graph picture of linear functions. READ."
  - "math/math for ML.pdf, Sec. 2.7.1 area (PDF p.62-63): composition of mappings written with the circle symbol (the notation for composing maps). READ."
  - "math/math for ML.pdf, Sec. 5 intro (PDF p.145-146): function notation f: R^D -> R, domain and image/codomain, one output per input. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (opening to the derivative section): functions like x^2 tabulated, slopes, 'bowl' shape and where its minimum is. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md (opening): why mapping inputs to outputs is the language of models. READ."
---

## Où cela s'inscrit

Un réseau de neurones est une fonction. Une perte est une fonction. Un algorithme d'apprentissage marche le long du graphe d'une fonction à la recherche de son point le plus bas. Avant tout cela, tu dois être à l'aise avec l'idée elle-même : ce qu'est une fonction, à quoi ressemble son graphe, ce que de simples familles (droites, paraboles, polynômes) peuvent et ne peuvent pas faire, et comment les fonctions peuvent être déplacées, combinées et défaites.

L'apprentissage automatique, c'est presque entièrement « construire des fonctions compliquées en composant des fonctions simples ». Quand tu rencontreras un réseau profond, tu le reconnaîtras comme `f3(f2(f1(x)))`, exactement la composition que nous pratiquons ici. Et « minimiser la perte » voudra dire trouver le fond d'un graphe en forme de parabole, c'est pourquoi nous étudions la parabole avec tant de soin.

Aucune leçon n'est requise avant celle-ci. Tu utilises l'arithmétique et la résolution d'équations de `mf.02` (relis sa section sur les équations linéaires si besoin) et les habitudes de code de `mf.01`.

## L'idée en mots simples

Une fonction est une **machine avec une fente d'entrée et une fente de sortie**. Tu y glisses un nombre, la machine suit à chaque fois la même procédure fixe, et un nombre en ressort. La règle de la sortie unique est toute la définition : donne deux fois la même entrée, tu obtiens deux fois la même sortie. Un distributeur automatique en est une bonne image : appuie sur B4, tu obtiens toujours la même friandise.

Le **domaine** est la liste des boutons qui existent (les entrées que la machine accepte). L'**image** (range) est l'ensemble des friandises qui sortent réellement. Certains boutons peuvent être bloqués (la machine ne sait pas prendre la racine carrée d'un négatif, ni diviser par zéro), c'est pourquoi un domaine n'est pas toujours « tout ».

Un **graphe** est le comportement de la machine dessiné sous forme d'image : la position horizontale est l'entrée, la hauteur est la sortie. Une fonction est le même objet que son graphe, mais l'image te permet de voir la forme : qui monte, qui descend, une cuvette, une bosse.

Ensuite, nous jouons avec les machines :

- **Translater** le comportement de la machine vers la gauche, la droite, le haut, le bas.
- Le **étirer** ou le **retourner**.
- **Enchaîner** deux machines : la sortie de la première est déposée dans la seconde (composition).
- **La faire tourner à l'envers** (inverse) : étant donnée la sortie, retrouver l'entrée, ce qui ne marche que si deux entrées différentes ne donnent jamais la même sortie.

## Définitions

Une **fonction** $f$ d'un ensemble $A$ (le domaine) vers un ensemble $B$ associe à chaque $x\in A$ exactement un $f(x)\in B$. On écrit $f:A\to B$, $x\mapsto f(x)$. C'est la même notation fléchée que celle utilisée pour des fonctions comme $f:\mathbb R^D\to\mathbb R$ dans le manuel de référence, où $\mathbb R^D$ est le domaine et où les valeurs vivent dans l'image/codomaine.

Vocabulaire : **entrée / argument** $x$ (variable indépendante), **sortie / valeur** $f(x)$, souvent appelée $y$. Le **domaine naturel** est l'ensemble des entrées réelles pour lesquelles la formule a un sens. Exemples :

| formule | domaine naturel | pourquoi |
|---|---|---|
| $x^2$ | tous les réels $x$ | rien ne casse |
| $1/x$ | $x\neq0$ | pas de division par zéro |
| $\sqrt{x-1}$ | $x\ge1$ | pas de racine carrée d'un négatif |
| image de $\sqrt{x}$ | sorties $\ge0$ | une racine carrée est non négative |

L'**image** de $x^2$ est $[0,\infty)$ : elle ne donne jamais de nombre négatif.

**Est-ce une fonction ?** Test de la droite verticale : une droite verticale coupe le graphe d'une fonction au plus une fois. Le cercle $x^2+y^2=1$ échoue au test (en $x=0$, on a à la fois $y=1$ et $y=-1$), donc $y$ n'est pas une fonction de $x$ ici.

### Évaluer (petits exemples travaillés)

Pour $f(x)=2x^2-3$ : $f(0)=-3$, $f(1)=-1$, $f(-2)=2\cdot4-3=5$, $f(a+1)=2(a+1)^2-3=2a^2+4a-1$. Le dernier est un endroit classique d'erreur : $f(a+1)$ n'est *pas* $f(a)+1$. Remplace toute l'entrée.

```python
f = lambda x: 2 * x ** 2 - 3
print(f(0), f(1), f(-2))   # -3 -1 5
```

## Fonctions linéaires

$f(x)=mx+b$. Deux nombres décrivent tout :

- la **pente** $m$ = variation de la sortie pour 1 unité d'entrée ; étant donnés deux points,

$$
m=\frac{y_2-y_1}{x_2-x_1};
$$

- l'**ordonnée à l'origine** $b=f(0)$, là où la droite rencontre l'axe vertical.

La droite passant par $(0,1)$ et $(2,5)$ : $m=(5-1)/(2-0)=2$, $b=1$, donc $f(x)=2x+1$. Vérification : $f(2)=5$. La propriété qui la définit est le **taux constant** : chaque pas de 1 unité en $x$ ajoute exactement $m$ à la sortie, pour toujours. C'est le modèle $\hat y=wx+b$ de la régression linéaire, avec $w$ la pente.

## Fonctions quadratiques

$f(x)=ax^2+bx+c$, $a\ne0$. Le graphe est une **parabole** : une cuvette ouverte vers le haut si $a>0$, vers le bas si $a<0$.

**Où est le fond (le sommet) ?** Complète le carré. Prenons $a>0$ :

$$
\begin{aligned}
ax^2+bx+c &= a\left(x^2+\tfrac{b}{a}x\right)+c\\
&= a\left(x+\tfrac{b}{2a}\right)^2 - \tfrac{b^2}{4a}+c .
\end{aligned}
$$

Le crochet au carré n'est jamais négatif et vaut 0 seulement quand $x=-\frac b{2a}$. Donc le point le plus bas est en $x=-\frac b{2a}$, et la valeur la plus basse est $c-\frac{b^2}{4a}$. Exemple : $x^2-4x+3$ a son sommet en $x=4/2=2$ avec la valeur $4-8+3=-1$.

**Où coupe-t-elle zéro (les racines) ?** Pose la forme à carré complété égale à 0 et résous :

$$
x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}.
$$

Le **discriminant** $b^2-4ac$ tranche : positif donne deux racines réelles, nul en donne une, négatif n'en donne aucune (la cuvette flotte au-dessus de l'axe). Pour $x^2-4x+3$ : $16-12=4$, racines $\frac{4\pm2}{2}=3$ et $1$ ; elles sont placées symétriquement autour du sommet en 2. Pour $x^2+1$ : discriminant $-4<0$, pas de racine réelle.

La parabole compte énormément : près de n'importe quel minimum lisse, une fonction de perte ressemble à une cuvette, donc « trouver le fond » se ramène à la formule du sommet. L'erreur quadratique $(y-wx)^2$ est un polynôme du second degré en $w$.

## Polynômes

$p(x)=a_nx^n+\dots+a_1x+a_0$ avec $a_n\ne0$ ; $n$ est le **degré** (droite : 1, parabole : 2, cubique : 3). Un polynôme de degré $n$ a au plus $n$ racines réelles et au plus $n-1$ points de rebroussement (extrema locaux). Évalue-le efficacement avec la **règle de Horner**, qui imbrique les multiplications :

$$
a_3x^3+a_2x^2+a_1x+a_0=((a_3x+a_2)x+a_1)x+a_0 .
$$

```python
def poly_eval(coeffs, x):          # coeffs highest power first
    result = 0
    for c in coeffs:
        result = result * x + c
    return result

print(poly_eval([1, 0, -1, 0], 3))  # x^3 - x at 3 -> 24
```

Suivi pour $x^3-x$ en $x=3$ (coefficients 1,0,-1,0) : result passe à 1, puis $1\cdot3+0=3$, puis $3\cdot3-1=8$, puis $8\cdot3+0=24$. Vérification : $27-3=24$. Les polynômes sont la première classe de modèles au-delà des droites ; avec un degré élevé, ils peuvent onduler à travers n'importe quels points, ce qui est l'origine du surapprentissage (overfitting).

## Transformations de graphes

Pars d'une fonction de base $f$. Ces règles déplacent son graphe ; vérifie chacune avec des nombres.

| nouvelle fonction | ce qui arrive au graphe |
|---|---|
| $f(x)+k$ | translation **vers le haut** de $k$ |
| $f(x-h)$ | translation **vers la droite** de $h$ |
| $c\,f(x)$, $c>1$ | étirement vertical de facteur $c$ |
| $f(x/c)$, $c>1$ | étirement horizontal de facteur $c$ |
| $-f(x)$ | réflexion par rapport à l'axe des $x$ |
| $f(-x)$ | réflexion par rapport à l'axe des $y$ |

**Pourquoi $f(x-h)$ se déplace-t-elle vers la droite ?** La nouvelle machine, à l'entrée $x$, fait ce que l'ancienne machine faisait en $x-h$. L'ancien point spécial (disons le sommet en $x=0$ pour $x^2$) apparaît maintenant là où $x-h=0$, c'est-à-dire en $x=h$. Exemple : $g(x)=(x-3)^2$ a $g(3)=0$, $g(5)=4$, $g(1)=4$ : la cuvette de $x^2$ est maintenant en 3. Le signe semble à l'envers au début ; la question « quelle entrée l'ancienne machine voit-elle ? » donne toujours la bonne réponse.

En combinant : $a\,f(x-h)+k$ translate d'abord vers la droite de $h$, étire de $a$, puis soulève de $k$. Le complètement du carré ci-dessus dit que toute parabole n'est que $x^2$ translatée, étirée et peut-être retournée.

## Composition

$(f\circ g)(x)=f(g(x))$ signifie : **applique d'abord $g$, puis $f$**. Avec $f(x)=x+1$ et $g(x)=2x$ :

- $(f\circ g)(3)=f(6)=7$
- $(g\circ f)(3)=g(4)=8$

L'ordre compte. La notation avec un cercle pour composer des applications est utilisée dans tout le manuel de référence (par ex. une application écrite comme composition de trois applications plus simples). En apprentissage profond, une couche de réseau est « une fonction linéaire, puis une fonction d'écrasement non linéaire », et un réseau profond est une longue composition de telles couches ; son entraînement utilise plus tard la règle de dérivation en chaîne, qui est précisément la règle du taux de variation d'une composition.

Précaution sur le domaine : $f\circ g$ n'est définie que là où la sortie de $g$ tombe dans le domaine de $f$. Si $f=\sqrt{\ }$ et $g(x)=x-1$, alors $f(g(x))=\sqrt{x-1}$ exige $x\ge1$.

## Fonctions inverses

L'inverse $f^{-1}$ défait $f$ : $f^{-1}(f(x))=x$ et $f(f^{-1}(y))=y$. (Le $-1$ est une étiquette, pas une puissance : $f^{-1}(y)\neq1/f(y)$.)

**Recette :** écris $y=f(x)$, résous pour $x$, puis renomme. Pour $f(x)=3x-6$ : $y=3x-6\Rightarrow x=\frac{y+6}{3}$, donc $f^{-1}(y)=\frac{y+6}{3}$. Vérification : $f^{-1}(9)=5$ et $f(5)=9$.

**Quand existe-t-il ?** Seulement si chaque sortie vient d'exactement une entrée (le **test de la droite horizontale** : aucune droite horizontale ne coupe le graphe deux fois). $f(x)=x^2$ échoue sur tout $\mathbb R$ : 2 et $-2$ vont tous deux vers 4, donc, étant donné 4, tu ne peux pas savoir lequel renvoyer. Restreindre le domaine à $x\ge0$ règle le problème, et l'inverse est alors $\sqrt{\ }$. Géométriquement, le graphe de $f^{-1}$ est le graphe de $f$ en miroir par rapport à la droite diagonale $y=x$.

Pour une fonction qui ne fait que monter, il existe une méthode numérique universelle, la **bissection** : pour trouver $x$ avec $f(x)=y$, garde un intervalle $[lo,hi]$ qui encadre la réponse, teste le milieu, et écarte la moitié qui ne peut pas la contenir. Soixante divisions par deux réduisent un intervalle de largeur 200 à environ $2\times10^{-16}$. C'est ainsi que les calculatrices inversent des fonctions compliquées.

## Exemple travaillé : vérifier par évaluation

```python
import numpy as np
f = lambda x: x**2 - 4*x + 3
xs = np.array([0., 1., 2., 3., 4.])
print(f(xs))                       # [ 3.  0. -1.  0.  3.]  roots 1 and 3, vertex (2,-1)

g = lambda x: f(x - 3)             # shifted right by 3
print(g(xs + 3))                   # same list: g at x+3 equals f at x

h = lambda x: 3*x - 6
h_inv = lambda y: (y + 6) / 3
print(h_inv(h(xs)))                # [0. 1. 2. 3. 4.]  the machine is undone
```

Chaque liste affichée peut être vérifiée à la main avec les règles ci-dessus.

## Idées fausses courantes

- « $f(a+b)=f(a)+f(b)$. » Seulement vrai pour des fonctions spéciales (linéaires passant par l'origine). Pour $f(x)=x^2$ : $f(1+1)=4\neq2$.
- « $f(x-3)$ se déplace vers la gauche. » Elle se déplace vers la droite.
- « $f^{-1}(x)=1/f(x)$. » Non, c'est l'inverse multiplicatif (réciproque), une chose différente.
- « Toute fonction a un inverse. » Seulement si des entrées différentes donnent des sorties différentes (sur le domaine choisi).
- « $f\circ g=g\circ f$. » Rarement vrai.
- « Une courbe est une fonction. » Seulement si chaque $x$ a un seul $y$.
- Lire un graphe loin en dehors de la fenêtre tracée comme si la tendance continuait. Un polynôme peut se comporter sauvagement en dehors de la région où tu l'as testé.

## Points clés à retenir

- Une fonction donne exactement une sortie par entrée ; connais son domaine et son image.
- Les droites ont une pente constante ; les paraboles ont un sommet en $-b/2a$ et des racines données par la formule quadratique ; le degré d'un polynôme borne les racines et les points de rebroussement.
- Les translations, étirements et réflexions sont systématiques ; $f(x-h)$ translate vers la droite.
- La composition applique d'abord la fonction intérieure, et l'ordre compte.
- Un inverse défait une fonction et n'existe que si la fonction est injective ; vérifie tout inverse annoncé en évaluant les deux compositions en plusieurs points.

## Explique-le toi-même

1. Pourquoi remplacer $x$ par $x-3$ déplace-t-il un graphe vers la droite ? Argumente avec le sommet de $x^2$.
2. Pourquoi $x^2$ n'a-t-elle pas d'inverse sur tous les nombres réels, et comment la restriction du domaine y remédie-t-elle ?

## Pour aller plus loin

- Le manuel de référence (Sec. 2.1 et le début du chap. 5) pour l'image de l'intersection de droites et la notation des fonctions.
- Plus tard dans l'académie : les dérivées mesurent la pente d'une courbe en un point (`mf.07` le prépare), et la règle de dérivation en chaîne dérive les compositions.

## Critères de maîtrise

Tu sais : évaluer et tabuler une fonction ; trouver un domaine naturel ; lire la pente et l'ordonnée à l'origine ; trouver le sommet et les racines d'une parabole (et utiliser le discriminant) ; prédire l'effet de $f(x-h)+k$, $cf(x)$, $-f(x)$, $f(-x)$ ; calculer $f\circ g$ et $g\circ f$ et montrer qu'elles diffèrent ; trouver un inverse algébriquement et le tester numériquement.
