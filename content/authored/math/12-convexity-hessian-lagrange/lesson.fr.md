---
id: math.12
title: "Convexité, hessienne et multiplicateurs de Lagrange"
summary: "Utilise les dérivées secondes pour lire la forme d'une surface de perte, vois pourquoi les problèmes convexes n'ont pas de mauvais minima locaux, fais des pas de Newton, et gère les contraintes avec les multiplicateurs de Lagrange et un premier aperçu de la dualité."
minutes: 90
teaches: [opt.convex]
requires: [calc.gradient, opt.gradient-descent, la.eigen]
sources:
  - "math/math for ML.pdf, Sec. 5.7 Higher-Order Derivatives (PDF p.170-171): second partial derivatives, symmetry of mixed partials, the Hessian as curvature. READ."
  - "math/math for ML.pdf, Sec. 5.8 Linearization and Multivariate Taylor Series (PDF p.171-172, start only): first-order Taylor expansion and the definition of the multivariate Taylor series. READ (p.171-172 only)."
  - "math/math for ML.pdf, Sec. 7.2 Constrained Optimization and Lagrange Multipliers (PDF p.239-242): constrained problem, indicator function, Lagrangian, primal and dual problems, minimax inequality, weak duality, equality constraints. READ."
  - "math/math for ML.pdf, Sec. 7.3 Convex Optimization (PDF p.242-245): convex sets and functions, first- and second-order conditions, negative entropy example, closure under non-negative sums, Jensen, linear program and its dual. READ (p.242-245 only)."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Appendix E Lagrange Multipliers (PDF p.727-729): gradient of g is normal to the surface, grad f parallel to grad g, the Lagrangian, worked example, inequality constraints. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/18-convex-optimization/docs/en.md (first 12000 characters): convex sets/functions, tests, Hessian eigenvalues, Newton's method, Lagrange example, KKT conditions. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/08-optimization/docs/en.md (Convex vs non-convex section and headings): non-convex landscapes and saddle points. READ."
---

## Où cela s'inscrit

La descente de gradient (leçon 7) ne regarde que la **pente** : de quel côté se trouve la descente. Elle ne sait pas si la vallée devant toi est large ou étroite, si tu te tiens au fond d'une cuvette ou sur un col de montagne. L'information sur la *forme* se trouve dans les dérivées secondes, rassemblées dans la matrice **hessienne**. Avec elle, tu peux (1) distinguer un minimum d'un maximum et d'un point selle, (2) choisir des pas plus malins que « une fraction fixe de la pente » (méthode de Newton), et (3) expliquer une profonde dichotomie du machine learning : la régression linéaire et logistique, les SVM, ridge et lasso sont **convexes**, donc sans mauvais minima locaux ; les réseaux de neurones ne le sont pas.

La seconde moitié traite des **contraintes** : minimiser une perte *tout en* gardant les poids dans un budget, des probabilités de somme un, ou des marges positives. L'outil est le multiplicateur de Lagrange, qui ouvre aussi la porte de la dualité (l'idée derrière les machines à vecteurs de support). Tu as déjà rencontré un multiplicateur de Lagrange dans la leçon de théorie de l'information (entropie maximale et fréquences $\mu_k=m_k/N$) ; voici la méthode elle-même.

## L'idée en mots simples

- **Courbure.** Marche à travers un paysage. La pente te dit la raideur ; la *courbure* te dit à quelle vitesse la raideur change. Sur une cuvette, le sol monte dans toutes les directions. Au sommet d'une colline, il descend dans toutes les directions. Sur un **col de montagne** (une selle), il monte le long du chemin et descend en travers. Sur un point plat, la courbure dans chaque direction décide de laquelle de ces trois situations tu te trouves.
- **La hessienne est le tableau des courbures.** Pour une fonction de $n$ variables, c'est un tableau symétrique $n\times n$. Ses vecteurs propres sont les directions principales du paysage et ses valeurs propres les courbures le long d'elles (leçon 7) : toutes positives, c'est une cuvette ; toutes négatives, un sommet ; signes mélangés, une selle.
- **Convexe = une seule cuvette.** Une fonction est convexe si le segment de droite entre deux points quelconques de son graphe ne passe jamais sous le graphe. Alors il est impossible de construire une deuxième vallée : un endroit qui ressemble localement à un fond *est* le fond global. C'est pourquoi les problèmes convexes sont des « problèmes résolus » alors que les pertes de réseaux de neurones ne le sont pas.
- **Méthode de Newton = « ajuster une cuvette et sauter à son fond ».** Au lieu de faire un petit pas dans le sens de la pente, approxime le paysage autour de toi par la meilleure parabole (avec la pente et la courbure) et saute directement au minimum de cette parabole. Pour une vraie cuvette, elle y arrive en un saut. Le prix est le coût du tableau de courbures.
- **Contraintes = tu dois rester sur un chemin.** Randonne le long d'un sentier fixe : le meilleur endroit du sentier est là où le sentier est *tangent* à une courbe de niveau de la carte des altitudes. Si le sentier traversait la courbe de niveau, tu pourrais glisser le long du sentier et descendre plus bas. À l'optimum, la direction de descente est perpendiculaire au sentier. Le **multiplicateur de Lagrange** est le nombre qui dit avec quelle force la contrainte repousse.

## Définitions et démonstrations

### Hessienne et développement de Taylor d'ordre deux

Pour $f:\mathbb R^n\to\mathbb R$ deux fois continûment dérivable, la **hessienne** est

$$
H(\mathbf x)=\nabla^2f(\mathbf x),\qquad H_{ij}=\frac{\partial^2f}{\partial x_i\,\partial x_j}\in\mathbb R^{n\times n}.
$$

Comme les dérivées partielles mixtes commutent ($\partial^2f/\partial x\partial y=\partial^2f/\partial y\partial x$, la (5.146) du livre), $H$ est **symétrique**, donc elle a des valeurs propres réelles et des vecteurs propres orthogonaux (leçon 7). La hessienne est la dérivée du gradient : chaque ligne est le gradient d'une composante de $\nabla f$.

Le développement de Taylor d'ordre un (livre (5.148)) est $f(\mathbf x)\approx f(\mathbf x_0)+\nabla f(\mathbf x_0)^\top(\mathbf x-\mathbf x_0)$ : un plan incliné. Garder le terme suivant (le terme d'ordre deux de la série de Taylor, dont le coefficient est la hessienne ; le livre définit la série générale dans la Déf. 5.7) donne une **quadratique** :

$$
f(\mathbf x_0+\boldsymbol\delta)\ \approx\ f(\mathbf x_0)+\nabla f(\mathbf x_0)^\top\boldsymbol\delta+\tfrac12\,\boldsymbol\delta^\top H(\mathbf x_0)\,\boldsymbol\delta .
$$

Formes : $\boldsymbol\delta:(n,)$, $\nabla f:(n,)$, $H:(n,n)$ ; chaque terme est un scalaire. En dimension un, c'est $f(x_0)+f'(x_0)\delta+\tfrac12f''(x_0)\delta^2$.

### Classer les points stationnaires

En un point stationnaire $\nabla f(\mathbf x^\star)=\mathbf 0$, le terme linéaire s'annule, donc la forme locale est décidée par $\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$. Écris $\boldsymbol\delta=\sum_ic_i\mathbf u_i$ dans les vecteurs propres orthonormés de $H$ ; alors $\boldsymbol\delta^\top H\boldsymbol\delta=\sum_i\lambda_ic_i^2$. Donc :

| Valeurs propres de $H(\mathbf x^\star)$ | Type |
|---|---|
| toutes $>0$ (définie positive) | minimum local |
| toutes $<0$ | maximum local |
| des deux signes | point selle |
| certaines $=0$, les autres d'un seul signe | non concluant (direction plate, regarde les termes d'ordre supérieur) |

Une selle est l'obstacle typique de l'entraînement en grande dimension : un point de gradient nul qui n'est pas un minimum. (La leçon d'optimisation du dépôt note que, dans les réseaux de grande dimension, les selles sont un obstacle pratique plus important que les mauvais minima locaux ; considère cela comme une règle empirique, pas un théorème.)

### Ensembles convexes et fonctions convexes

Un ensemble $C$ est **convexe** si pour tous $\mathbf x,\mathbf y\in C$ et $\theta\in[0,1]$ : $\theta\mathbf x+(1-\theta)\mathbf y\in C$ (tout le segment reste à l'intérieur). Une fonction $f$ sur un domaine convexe est **convexe** si

$$
f(\theta\mathbf x+(1-\theta)\mathbf y)\ \le\ \theta f(\mathbf x)+(1-\theta)f(\mathbf y)\qquad\forall\ \theta\in[0,1] \quad(\text{the chord lies above the graph}).
$$

Tests équivalents pour $f$ dérivable (Deisenroth Sec. 7.3) :

- **Ordre un :** $f(\mathbf y)\ge f(\mathbf x)+\nabla f(\mathbf x)^\top(\mathbf y-\mathbf x)$ pour tous $\mathbf x,\mathbf y$ : le graphe se situe au-dessus de chacun de ses plans tangents.
- **Ordre deux :** $H(\mathbf x)$ est semi-définie positive pour **tout** $\mathbf x$ du domaine. En 1-D, $f''\ge0$ partout.

Stabilité : une somme pondérée positive $\alpha f_1+\beta f_2$ ($\alpha,\beta\ge0$) de fonctions convexes est convexe (additionne les deux inégalités de définition, comme dans l'exemple 7.4 du livre). C'est pourquoi une perte convexe plus un régulariseur convexe (ridge $\lambda\|\mathbf w\|^2$, lasso $\lambda\|\mathbf w\|_1$) est convexe. Exemples : $x^2$, $e^x$, $|x|$, $-\ln x$, $\max(0,x)$, les fonctions linéaires, $x\log x$ pour $x>0$ (la vérification du livre en $x=2,4$ : valeur au milieu $3\log_23\approx4.755\le$ valeur de la corde $5$), et la quadratique $\tfrac12\mathbf x^\top Q\mathbf x-\mathbf c^\top\mathbf x$ avec $Q$ semi-définie positive, dont la hessienne est la constante $Q$.

### Pourquoi les problèmes convexes n'ont pas de mauvais minima locaux

**Affirmation.** Si $f$ est convexe, tout minimum local est un minimum global.

*Démonstration.* Supposons que $\mathbf x$ soit un minimum local mais qu'un $\mathbf y$ vérifie $f(\mathbf y)<f(\mathbf x)$. Pour $\theta\in(0,1]$ petit, le point $\mathbf z_\theta=\theta\mathbf y+(1-\theta)\mathbf x$ est aussi proche de $\mathbf x$ que nous voulons, et par convexité
$$
f(\mathbf z_\theta)\le\theta f(\mathbf y)+(1-\theta)f(\mathbf x)=f(\mathbf x)-\theta\big(f(\mathbf x)-f(\mathbf y)\big)<f(\mathbf x).
$$
Il existe donc des points arbitrairement proches de $\mathbf x$ de valeur strictement plus basse, ce qui contredit la minimalité locale. $\square$

De façon équivalente avec la condition d'ordre un : si $\nabla f(\mathbf x^\star)=\mathbf 0$ alors $f(\mathbf y)\ge f(\mathbf x^\star)+0$ pour tout $\mathbf y$, donc un point stationnaire d'une fonction convexe est automatiquement un minimum global. La descente de gradient avec un pas raisonnable ne peut donc pas être piégée. À l'inverse, $f(x)=x^4-2x^2$ a deux minima en $x=\pm1$ (valeur $-1$) et un maximum local entre eux ; l'inégalité de convexité échoue : la corde entre $x=-1$ et $x=1$ se situe à la hauteur $\frac12(f(1)+f(-1))=-1$, sous le graphe au milieu, où $f(0)=0$. La descente de gradient démarrée en $+0.5$ finit en $+1$, démarrée en $-0.5$ finit en $-1$.

### Méthode de Newton

Minimise le modèle d'ordre deux $m(\boldsymbol\delta)=f+\mathbf g^\top\boldsymbol\delta+\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$ ($\mathbf g=\nabla f(\mathbf x_k)$, $H=H(\mathbf x_k)$). Son gradient par rapport à $\boldsymbol\delta$ est $\mathbf g+H\boldsymbol\delta$ (avec les règles de gradient des formes linéaires et quadratiques ; $H$ symétrique). L'annuler donne :

$$
H\,\boldsymbol\delta=-\mathbf g\quad\Longrightarrow\quad\mathbf x_{k+1}=\mathbf x_k-H(\mathbf x_k)^{-1}\nabla f(\mathbf x_k).
$$

Compare avec la descente de gradient $\mathbf x_{k+1}=\mathbf x_k-\eta\nabla f(\mathbf x_k)$ : Newton remplace le taux d'apprentissage scalaire par l'inverse de la hessienne, qui fait de petits pas le long des directions de forte courbure et de grands pas le long des directions plates. En pratique, on n'inverse pas : résous le système linéaire $H\boldsymbol\delta=-\mathbf g$ (leçon 11). Pour une $f$ quadratique, un pas atterrit exactement sur le minimum. Près d'un minimum d'une fonction lisse générale, l'erreur est à peu près **élevée au carré** à chaque pas (convergence quadratique). Le coût : une mémoire en $O(n^2)$ pour $H$ et $O(n^3)$ pour résoudre, impossible pour des réseaux de millions de poids ; et là où $H$ n'est pas définie positive, le pas de Newton peut pointer vers le haut (vers une selle ou un maximum), donc les variantes pratiques régularisent $H$.

### Optimisation sous contrainte et multiplicateurs de Lagrange

**Problème (contrainte d'égalité).** Minimiser $f(\mathbf x)$ sous la contrainte $g(\mathbf x)=0$.

**Géométrie** (Bishop, Annexe E). En tout point de la surface de contrainte, $\nabla g$ est perpendiculaire à la surface : si $\mathbf x$ et $\mathbf x+\boldsymbol\epsilon$ y sont tous deux, alors $g(\mathbf x+\boldsymbol\epsilon)\approx g(\mathbf x)+\boldsymbol\epsilon^\top\nabla g$ et $g$ est identique aux deux points, donc $\boldsymbol\epsilon^\top\nabla g\approx0$. À un optimum sur la surface, $\nabla f$ doit aussi être perpendiculaire à la surface (sinon, glisser le long de la surface changerait $f$ au premier ordre). Deux vecteurs perpendiculaires à la même surface sont parallèles, donc il existe un nombre $\lambda$ tel que

$$
\nabla f+\lambda\nabla g=\mathbf 0 \qquad(\lambda\text{ may have either sign}).
$$

Regroupe cela dans le **lagrangien** $\mathcal L(\mathbf x,\lambda)=f(\mathbf x)+\lambda g(\mathbf x)$. Annuler $\nabla_{\mathbf x}\mathcal L=\mathbf 0$ donne la condition ci-dessus, et annuler $\partial\mathcal L/\partial\lambda=0$ redonne la contrainte $g=0$. Donc **l'optimisation sous contrainte devient la recherche d'un point stationnaire de $\mathcal L$** : $n+1$ équations à $n+1$ inconnues. Tu peux résoudre pour $\mathbf x$ sans jamais avoir besoin de $\lambda$ (d'où « multiplicateur indéterminé »).

**Exemple détaillé.** Minimiser $f(x,y)=x^2+y^2$ sous $x+y-1=0$.

$$
\mathcal L=x^2+y^2+\lambda(x+y-1),\qquad \frac{\partial\mathcal L}{\partial x}=2x+\lambda=0,\quad \frac{\partial\mathcal L}{\partial y}=2y+\lambda=0,\quad \frac{\partial\mathcal L}{\partial\lambda}=x+y-1=0 .
$$

Les deux premières donnent $x=y=-\lambda/2$ ; la troisième donne $-\lambda=1$, donc $\lambda=-1$ et $x=y=\tfrac12$ avec $f=\tfrac12$ : le point de la droite le plus proche de l'origine. Vérification par élimination : substitue $y=1-x$ : $x^2+(1-x)^2$ a pour dérivée $4x-2=0$, $x=\tfrac12$. La même méthode appliquée à l'exemple de Bishop $1-x_1^2-x_2^2$ (à maximiser) avec $x_1+x_2-1=0$ donne $(\tfrac12,\tfrac12)$ et $\lambda=1$ ; le signe de $\lambda$ dépend de si tu minimises ou maximises et de la façon dont tu écris $g$.

**La régularisation comme contrainte.** « Minimise la perte sous $\|\mathbf w\|^2\le t$ » et « minimise perte $+\lambda\|\mathbf w\|^2$ » sont deux visions du même problème : $\lambda$ est le multiplicateur de la contrainte de budget (leçon 18 du dépôt).

### Contraintes d'inégalité et premier aperçu de la dualité

Pour $\min f(\mathbf x)$ sous $g_i(\mathbf x)\le0$, le livre associe un multiplicateur $\lambda_i\ge0$ à chaque contrainte :

$$
\mathcal L(\mathbf x,\boldsymbol\lambda)=f(\mathbf x)+\boldsymbol\lambda^\top\mathbf g(\mathbf x),\qquad D(\boldsymbol\lambda)=\min_{\mathbf x}\mathcal L(\mathbf x,\boldsymbol\lambda).
$$

Pourquoi : la contrainte dure peut s'écrire comme une pénalité infinie (une fonction indicatrice). Pour $\lambda\ge0$, la pénalité linéaire $\lambda g$ n'est jamais plus grande que cette pénalité infinie (là où $g\le0$ elle est $\le0$ ; là où $g>0$ l'indicatrice vaut $+\infty$), et $\max_{\boldsymbol\lambda\ge0}\mathcal L$ égale $f(\mathbf x)$ quand $\mathbf x$ est admissible et $+\infty$ sinon, ce qui est exactement l'objectif pénalisé. Le problème **primal** est $\min_{\mathbf x}\max_{\boldsymbol\lambda\ge0}\mathcal L$, le problème **dual** est $\max_{\boldsymbol\lambda\ge0}D(\boldsymbol\lambda)=\max_{\boldsymbol\lambda\ge0}\min_{\mathbf x}\mathcal L$. Échanger min et max ne peut que diminuer la valeur (l'inégalité minimax, livre (7.23)), ce qui donne la **dualité faible** : chaque valeur duale $D(\boldsymbol\lambda)$ est une borne inférieure de l'optimum primal. $D$ est toujours concave, même quand $f$ ne l'est pas (c'est un minimum ponctuel de fonctions affines en $\boldsymbol\lambda$). Pour les problèmes convexes ($f$, $g_i$ convexes), le livre affirme que la **dualité forte** tient : les deux valeurs optimales coïncident.

*Exemple.* Minimiser $x^2$ sous $1-x\le0$ (c'est-à-dire $x\ge1$). Alors $\mathcal L=x^2+\lambda(1-x)$ ; minimiser sur $x$ donne $x=\lambda/2$ et $D(\lambda)=\lambda-\lambda^2/4$. Maximiser sur $\lambda\ge0$ : $D'=1-\lambda/2=0$, donc $\lambda=2$ et $D=1$. L'optimum primal est $x=1$ de valeur $1$ : égal. Pour d'autres $\lambda$, la valeur duale est plus basse : $D(1)=0.75$, $D(3)=0.75$, $D(0)=0$. À la solution, la contrainte est *active* ($1-x=0$) et $\lambda>0$ ; si le minimum sans contrainte $x=0$ avait déjà satisfait la contrainte, le multiplicateur aurait été $0$ (**complémentarité** : $\lambda_ig_i(\mathbf x^\star)=0$, comme dans la discussion de Bishop sur les contraintes actives et inactives et dans les conditions KKT de la leçon du dépôt). Les machines à vecteurs de support sont exactement cette construction, avec $\lambda_i>0$ pour les vecteurs de support.

## Exemple détaillé

Tous les nombres sont reproduits par le code de l'exercice.

**1. Classer des points stationnaires.** $f(x,y)=x^3-3x+y^2$. Le gradient $(3x^2-3,\ 2y)$ s'annule en $(1,0)$ et $(-1,0)$. Hessienne $\operatorname{diag}(6x,\,2)$. En $(1,0)$ : $\operatorname{diag}(6,2)$, définie positive, un minimum local avec $f=-2$. En $(-1,0)$ : $\operatorname{diag}(-6,2)$, signes mélangés, une selle avec $f=2$.

**2. Une selle due aux termes croisés.** $f(x,y)=x^2+3xy+y^2$ a $H=\begin{psmallmatrix}2&3\\3&2\end{psmallmatrix}$ de valeurs propres $5$ et $-1$ : une selle à l'origine, bien que les deux coefficients diagonaux soient positifs. Le long du vecteur propre $(1,-1)$ : $f(t,-t)=t^2-3t^2+t^2=-t^2<0$.

**3. Newton contre descente de gradient sur une quadratique.** $f(\mathbf x)=\tfrac12\mathbf x^\top Q\mathbf x-\mathbf c^\top\mathbf x$ avec $Q$ de lignes $(4,1)$ et $(1,3)$, $\mathbf c=(1,2)$ (valeurs propres $2.38$ et $4.62$, donc $f$ est convexe). Le minimiseur résout $Q\mathbf x=\mathbf c$ : $\mathbf x^\star=(1/11,\,7/11)=(0.0909,\,0.6364)$. Depuis $(5,5)$, un pas de Newton donne exactement $(0.0909,0.6364)$. La descente de gradient avec $\eta=0.2$ a des erreurs de $0.771$ après 1 pas, $0.0445$ après 5, $0.00175$ après 10, et $4\times10^{-9}$ après 30.

**4. Convergence quadratique.** Minimise $f(x)=e^x-2x$ (convexe, minimum en $\ln2=0.693147$). Newton : $x_{k+1}=x_k-(e^{x_k}-2)/e^{x_k}$ depuis $x_0=0$ donne les erreurs $0.307,\ 0.0426,\ 8.95\times10^{-4},\ 4.0\times10^{-7},\ 8.0\times10^{-14}$ : le nombre de chiffres corrects double à peu près à chaque pas. La descente de gradient avec $\eta=0.3$ a les erreurs $0.393,\ 0.198,\ 0.090,\ 0.038,\ 0.016,\ 0.0064$ : un facteur constant par pas.

**5. Exemple non convexe.** $h(x)=x^4-2x^2$ : $h(1)=h(-1)=-1$, $h(0)=0>\frac12(h(1)+h(-1))$, donc elle n'est pas convexe. La descente de gradient de pas $0.05$ depuis $\pm0.5$ finit en $\pm1$.

**6. Lagrange.** L'exemple ci-dessus : $(0.5,0.5)$, $\lambda=-1$, $f=0.5$. Exemple dual : $D(\lambda)=\lambda-\lambda^2/4$, maximum $1$ en $\lambda=2$.

## Implémentation

```python
import numpy as np

def grad(x):  return np.array([3*x[0]**2 - 3, 2*x[1]])      # f = x^3 - 3x + y^2
def hess(x):  return np.array([[6*x[0], 0.], [0., 2.]])

x_star = np.array([1., 0.])
eig = np.linalg.eigvalsh(hess(x_star))      # ascending, real because H is symmetric
kind = "min" if (eig > 0).all() else "max" if (eig < 0).all() else "saddle"

# Newton step for a quadratic: solve, never invert
Q = np.array([[4., 1], [1, 3]]); c = np.array([1., 2]); x = np.array([5., 5])
x = x - np.linalg.solve(Q, Q @ x - c)       # lands on Q^{-1} c = [1/11, 7/11]
```

## Formes

$\mathbf x,\nabla f:(n,)$ ; hessienne $(n,n)$ symétrique, `np.linalg.eigvalsh` renvoie `(n,)`. Pas de Newton : résous $H\boldsymbol\delta=-\mathbf g$ avec `np.linalg.solve(H, -g)`, résultat `(n,)`. Pour $m$ contraintes, $\mathbf g(\mathbf x)\in\mathbb R^m$ et les multiplicateurs $\boldsymbol\lambda\in\mathbb R^m$. Calculer la hessienne d'un réseau de $n=10^6$ poids demanderait $10^{12}$ nombres ; c'est pourquoi la méthode de Newton n'y est pas utilisée.

## Idées fausses courantes

- **Des coefficients diagonaux de $H$ positifs signifient un minimum.** Ce sont les valeurs propres qui décident, pas la diagonale ($x^2+3xy+y^2$ est une selle).
- **Un gradient nul signifie un minimum.** Ce peut être un maximum ou une selle ; vérifie la hessienne.
- **Convexe signifie « a un minimum unique ».** Convexe garantit que tout minimum local est global ; le minimiseur peut être tout un ensemble plat (par ex. $f=0$). La stricte convexité donne l'unicité.
- **La méthode de Newton est toujours meilleure.** Elle exige une hessienne définie positive et coûte $O(n^3)$ ; loin du minimum, elle peut diverger.
- **Les réseaux de neurones sont convexes en leurs poids.** Des couches cachées à activations non linéaires brisent la convexité ; la SGD marche bien malgré tout, pour des raisons en partie empiriques.
- **Le multiplicateur de Lagrange n'est qu'une astuce.** Il a un sens : la sensibilité de la valeur optimale à un relâchement de la contrainte (non démontré ici), et dans les SVM il repère les vecteurs de support.

## Points clés à retenir

- $f(\mathbf x_0+\boldsymbol\delta)\approx f+\mathbf g^\top\boldsymbol\delta+\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$ ; en un point stationnaire, les valeurs propres de $H$ disent minimum, maximum ou selle.
- Convexe : corde au-dessus du graphe, $H\succeq0$ partout, et tout minimum local est global.
- Newton : $\mathbf x\leftarrow\mathbf x-H^{-1}\mathbf g$ (résous, n'inverse pas) ; exact sur les quadratiques, à convergence quadratique près d'un minimum, trop coûteux pour les énormes modèles.
- Contraintes : $\nabla f=-\lambda\nabla g$ ; cherche les points stationnaires de $\mathcal L=f+\lambda g$. Les inégalités exigent $\lambda\ge0$ et la complémentarité ; le dual donne des bornes inférieures, égales pour les problèmes convexes.

## Explique-le avec tes mots

1. Pourquoi un point stationnaire d'une fonction convexe est-il toujours un minimum global ? Utilise l'image de la corde.
2. Dessine (ou décris) pourquoi, à l'optimum d'un problème contraint à une courbe, le gradient de l'objectif doit être perpendiculaire à la courbe.

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 7.1 (descente de gradient, PDF p.233, hors des sources de cette leçon), 7.3.2 programmation quadratique (PDF p.246) et Sec. 7.3.3 dualité de Legendre-Fenchel, qui continuent là où cette leçon s'arrête.
- Boyd et Vandenberghe, *Convex Optimization* (cité par le livre pour les définitions utilisées ici).

## Critères de maîtrise

Tu sais calculer une hessienne et classer des points stationnaires par ses valeurs propres, vérifier la convexité par la définition ou par la hessienne, expliquer pourquoi la convexité exclut les mauvais minima locaux, effectuer un pas de Newton, et résoudre un petit problème sous contrainte d'égalité avec un lagrangien en interprétant le multiplicateur et la valeur duale.
