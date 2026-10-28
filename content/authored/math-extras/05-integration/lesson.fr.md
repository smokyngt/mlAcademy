---
id: mx.05
title: L'intégration pour le machine learning
summary: "L'aire comme limite de sommes, le théorème fondamental de l'analyse, substitution et intégration par parties, pourquoi les espérances sont des intégrales, l'intégrale gaussienne par les coordonnées polaires, et trois méthodes numériques comparées."
minutes: 80
teaches: [mx.integral, mx.expect-integral, mx.numint]
requires: [calc.derivative, alg.summation, prob.rv, calc.numerical-diff]
sources:
  - "math/math for ML.pdf, Sec. 6.2 Discrete and Continuous Probabilities (PDF p.184-190): pdf definition (non-negative, integrates to 1), P(a<=X<=b) as an integral, cdf as an integral, densities may exceed 1 (uniform on [0.9,1.6]). READ."
  - "math/math for ML.pdf, Sec. 6.5 Gaussian Distribution (PDF p.203-205 and p.209): univariate and multivariate Gaussian density, mixture second moment E[x^2] as integrals and the variance of a mixture. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Appendix B Probability Distributions (PDF p.705-706, p.710): Bernoulli and Beta summaries (E[mu]=a/(a+b)), Gaussian marginal formula (B.51). Read selectively. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md (MCMC section, for the link from integrals to sampling). READ."
---

## Où cela s'inscrit

Le mot **espérance** cache une intégrale. La perte d'entropie croisée, la divergence KL, l'évidence en inférence bayésienne, l'ELBO d'un autoencodeur variationnel, la constante de normalisation d'une distribution, la récompense moyenne d'une politique : tout cela, ce sont des intégrales d'une fonction contre une densité de probabilité. La plupart du temps, elles ne peuvent pas être résolues au crayon, donc le ML les remplace par des sommes sur les données (la perte d'entraînement est une estimation de Monte-Carlo d'une perte espérée) ou par de l'échantillonnage ([[mx.04]] a montré comment le MCMC le fait).

Cette leçon te donne les outils pour lire ces formules et les évaluer numériquement. Elle s'appuie sur les dérivées ([[f2n.04]]) et l'idée de densité de [[math.09]]. L'intégrale gaussienne calculée ici justifie la constante devant la densité gaussienne, que tu utilises depuis [[math.09]].

## L'idée en mots simples

- Pour trouver l'aire sous une courbe, découpe-la en fines bandes verticales, traite chaque bande comme un rectangle, et additionne les rectangles. Prends des bandes plus fines et la somme s'approche d'un unique nombre, l'**intégrale**.
- Le **théorème fondamental de l'analyse** dit quelque chose de surprenant : si tu connais une fonction dont la pente est ta courbe (une *primitive*), alors l'aire entre deux points est simplement la différence de cette fonction aux deux extrémités. Intégrer défait dériver.
- Une **densité de probabilité** est une courbe d'aire totale 1. La probabilité de tomber dans un intervalle est l'aire au-dessus. Un point isolé n'a pas d'aire, donc il a probabilité zéro. La *hauteur* de la courbe n'est pas une probabilité, donc elle peut dépasser 1 (écrase une distribution dans une plage étroite et elle doit devenir haute).
- La **valeur moyenne** d'une quantité quand l'entrée est aléatoire est l'aire sous « quantité fois densité ». C'est l'espérance. Pour plusieurs variables, l'aire devient un volume, et tu intègres une variable à la fois.
- Quand il n'existe pas de formule propre, évalue l'aire numériquement : relie les points de la courbe par des segments de droite (trapèze), ou par des paraboles (Simpson), ou lance simplement des fléchettes au hasard et compte (Monte-Carlo). Les fléchettes sont lentes en dimension un, mais leur précision ne se soucie pas du nombre de dimensions.

## Intégrales définies et théorème fondamental

Divise $[a,b]$ en $n$ morceaux de largeur $h=(b-a)/n$ avec les points $x_k=a+kh$. La **somme de Riemann** est

$$
S_n=\sum_{k=0}^{n-1}f(x_k)\,h ,
$$

et l'**intégrale définie** est sa limite quand $n\to\infty$ : $\int_a^bf(x)\,dx=\lim S_n$ (pour $f$ continue). Les aires sous l'axe comptent négativement.

**Théorème fondamental de l'analyse.** Définis $F(x)=\int_a^xf(t)\,dt$. Alors $F'(x)=f(x)$, et par conséquent, pour toute primitive $G$ (toute $G$ avec $G'=f$)

$$
\int_a^bf(x)\,dx=G(b)-G(a).
$$

*Pourquoi $F'=f$.* $F(x+h)-F(x)=\int_x^{x+h}f(t)\,dt\approx f(x)\,h$ pour $h$ petit, puisque $f$ varie à peine sur un court intervalle. Diviser par $h$ et laisser $h\to0$ donne $F'(x)=f(x)$. Deux primitives quelconques diffèrent d'une constante, qui s'annule dans $G(b)-G(a)$.

### Deux techniques

La **substitution** est la règle de dérivation en chaîne lue à l'envers. Si $u=g(x)$ alors $du=g'(x)dx$ et

$$
\int_a^bf(g(x))\,g'(x)\,dx=\int_{g(a)}^{g(b)}f(u)\,du .
$$

Exemple : $\int_0^12x\,e^{x^2}dx$ ; avec $u=x^2$, $du=2x\,dx$, cela devient $\int_0^1e^u\,du=e-1=1.71828$. La substitution est aussi la façon dont une densité se transforme par changement de variable : un facteur jacobien apparaît (voir [[f2n.06]]).

L'**intégration par parties** est la règle du produit lue à l'envers. À partir de $(uv)'=u'v+uv'$, en intégrant de $a$ à $b$ :

$$
\int_a^bu\,v'\,dx=\big[uv\big]_a^b-\int_a^bu'\,v\,dx .
$$

Exemple : $\int_0^1x\,e^x\,dx$ avec $u=x$, $v'=e^x$ : $[xe^x]_0^1-\int_0^1e^x\,dx=e-(e-1)=1$.

## Les espérances sont des intégrales

Une variable aléatoire continue $X$ a une densité $p(x)\ge0$ avec $\int_{-\infty}^{\infty}p(x)\,dx=1$ (Deisenroth Def. 6.1), et $P(a\le X\le b)=\int_a^bp(x)\,dx$. La fonction de répartition $F(x)=P(X\le x)=\int_{-\infty}^xp(t)\,dt$ est l'aire cumulée. Deux conséquences que le livre souligne : $P(X=x)=0$ pour tout $x$ isolé (un intervalle de largeur nulle), et $p(x)$ peut dépasser 1 tant que l'aire vaut 1. Exemple du livre : la densité uniforme sur $[0.9,1.6]$ a pour hauteur $1/0.7\approx1.43$.

L'espérance d'une fonction $g(X)$ remplace la somme pondérée discrète $\sum_ig(x_i)p_i$ par

$$
\mathbb{E}[g(X)]=\int g(x)\,p(x)\,dx,
$$

donc la moyenne correspond à $g(x)=x$, et la variance est $\mathbb{E}[X^2]-(\mathbb{E}X)^2$ avec $g(x)=x^2$. Deisenroth s'en sert exactement pour calculer la moyenne et la variance d'un mélange de deux densités, $\alpha p_1+(1-\alpha)p_2$ : l'intégrale d'une somme est la somme des intégrales, donc $\mathbb{E}[X^2]=\alpha(\mu_1^2+\sigma_1^2)+(1-\alpha)(\mu_2^2+\sigma_2^2)$.

**Plusieurs variables.** Pour une densité jointe $p(x,y)$ : $\iint p\,dx\,dy=1$, et la **marginale** s'obtient en intégrant l'autre variable, $p(x)=\int p(x,y)\,dy$ (l'analogue intégral de la somme d'une ligne d'un tableau). Les intégrales itérées peuvent se faire dans n'importe quel ordre pour des fonctions bien élevées (théorème de Fubini, cité). L'Annexe B de Bishop liste des résultats sous forme close de telles intégrales pour les distributions usuelles : pour la densité Beta sur $[0,1]$, la moyenne est $a/(a+b)$ et la marginale gaussienne d'un sous-ensemble de variables est la gaussienne avec le bloc correspondant de la moyenne et de la covariance (B.51).

## L'intégrale gaussienne

La densité gaussienne univariée (Deisenroth Éq. 6.62) est $p(x)=\frac1{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$. Pourquoi cette constante ? Parce que la densité doit s'intégrer à 1, ce qui exige

$$
I=\int_{-\infty}^{\infty}e^{-x^2/2}\,dx=\sqrt{2\pi}.
$$

La fonction $e^{-x^2/2}$ n'a pas de primitive élémentaire, donc l'astuce consiste à élever l'intégrale au carré et à passer en deux dimensions :

$$
I^2=\int e^{-x^2/2}dx\int e^{-y^2/2}dy=\iint_{\mathbb{R}^2}e^{-(x^2+y^2)/2}\,dx\,dy .
$$

Passe aux coordonnées polaires $x=r\cos\theta$, $y=r\sin\theta$, pour lesquelles $x^2+y^2=r^2$ et l'élément d'aire est $dx\,dy=r\,dr\,d\theta$ (le jacobien de [[f2n.06]]) :

$$
I^2=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta=2\pi\int_0^\infty e^{-r^2/2}\,r\,dr .
$$

L'intégrale intérieure admet la substitution $u=r^2/2$, $du=r\,dr$ : $\int_0^\infty e^{-u}du=1$. Donc $I^2=2\pi$ et $I=\sqrt{2\pi}$. La substitution $z=(x-\mu)/\sigma$ donne la constante générale $\sqrt{2\pi\sigma^2}$. La version multivariée (Éq. 6.63) a la constante $(2\pi)^{-D/2}|\Sigma|^{-1/2}$.

**Le moment d'ordre deux.** Pour $X\sim\mathcal N(0,1)$, intègre par parties avec $u=x$ et $v'=xe^{-x^2/2}$ ($v=-e^{-x^2/2}$) :

$$
\int x\cdot xe^{-x^2/2}dx=\big[-xe^{-x^2/2}\big]_{-\infty}^{\infty}+\int e^{-x^2/2}dx=0+\sqrt{2\pi},
$$

donc $\mathbb{E}[X^2]=\sqrt{2\pi}/\sqrt{2\pi}=1$ : la variance de la loi normale centrée réduite est 1, comme il se doit.

## Intégration numérique

Soit $h=(b-a)/n$ et $y_k=f(a+kh)$.

- **Somme de Riemann à gauche :** $h\sum_{k=0}^{n-1}y_k$. Erreur $O(h)$.
- **Méthode des trapèzes :** remplace la courbe sur chaque bande par un segment de droite : $h\big(\tfrac12y_0+y_1+\dots+y_{n-1}+\tfrac12y_n\big)$. Erreur $O(h^2)$ : doubler $n$ divise l'erreur par environ 4.
- **Méthode de Simpson** ($n$ pair) : remplace la courbe par une parabole passant par chaque triplet de points : $\frac h3\big(y_0+4y_1+2y_2+4y_3+\dots+4y_{n-1}+y_n\big)$. Erreur $O(h^4)$ : doubler $n$ divise l'erreur par environ 16 ; elle est exacte pour les polynômes jusqu'au degré 3.
- **Monte-Carlo :** tire $u_1,\dots,u_n$ uniformément dans $[a,b]$ et estime $\int_a^bf\approx\frac{b-a}n\sum f(u_i)$. C'est une moyenne de termes indépendants, donc l'erreur est aléatoire d'écart type $\sigma/\sqrt n$, où $\sigma$ est l'écart type de $(b-a)f(u)$. Erreur $O(n^{-1/2})$ : 100 fois plus d'échantillons donnent 10 fois moins d'erreur. Lent en 1D... mais voir le paragraphe suivant.

**Le fléau de la dimension.** Une règle de grille avec $m$ points par axe en dimension $d$ demande $m^d$ évaluations : $m=100$ et $d=5$ signifient déjà $10^{10}$ points. Monte-Carlo garde le taux $n^{-1/2}$ quel que soit $d$ (la constante dépend de la variance de l'intégrande, pas directement de $d$). C'est pourquoi toute espérance en ML de grande dimension (lois a posteriori bayésiennes, ELBO, gradients de politique) est estimée par échantillonnage. L'entraînement par mini-lots est de l'intégration de Monte-Carlo de la perte espérée. La réduction de variance (variables de contrôle, échantillonnage préférentiel) améliore la constante.

Les gaussiennes sont un cas particulier : la méthode des trapèzes sur une fonction lisse à décroissance rapide converge bien plus vite que $O(h^2)$, comme le montre l'exemple.

## Exemple travaillé

Toutes les valeurs proviennent du code de l'exercice.

1. Somme de Riemann à gauche de $f(x)=x$ sur $[0,1]$ avec $n=100$ : $0.495$ (l'aire exacte est $0.5$ ; l'erreur est $h/2=0.005$, d'ordre $h$).
2. $\int_0^\pi\sin x\,dx=2$ exactement. Erreurs par rapport à la valeur exacte :

| $n$ | erreur des trapèzes | erreur de Simpson |
|---|---|---|
| 4 | $-0.10388$ | $+0.004560$ |
| 8 | $-0.025768$ | $+0.000269$ |
| 16 | $-0.006430$ | $+0.0000166$ |
| 32 | $-0.001607$ | $+0.00000103$ |

L'erreur des trapèzes est divisée par 4 chaque fois que $n$ double ; celle de Simpson par environ 16.

3. Monte-Carlo pour la même intégrale (échantillons uniformes, graine 0) : $n=100$ donne $1.866\pm0.105$ ; $n=10^4$ donne $1.9970\pm0.0097$ ; $n=10^6$ donne $1.99914\pm0.00097$. L'erreur standard diminue d'un facteur 10 pour chaque multiplication de $n$ par 100.
4. Intégrale gaussienne : la méthode des trapèzes pour $\int e^{-x^2/2}dx$ sur $[-8,8]$ a une erreur de $0.036$ avec $8$ intervalles, $1.3\times10^{-8}$ avec $16$, et environ $10^{-15}$ avec $32$. Sur $[-10,10]$ avec $2000$ intervalles, elle renvoie $2.5066283=\sqrt{2\pi}$.
5. Espérances comme intégrales (Simpson sur $[-8,8]$) : $\int p=1$, $\mathbb{E}[X]=0$, $\mathbb{E}[X^2]=1$ pour la loi normale centrée réduite, et $P(|X|\le1.96)=0.95000$.
6. La densité Beta$(2,5)$ est $30x(1-x)^4$ ; sa moyenne calculée par Simpson est $0.285714=2/7=a/(a+b)$, en accord avec la formule de Bishop (B.7).
7. Par parties : $\int_0^1xe^x\,dx=1$ (Simpson avec $n=20$ : $1.0000003$). Par substitution : $\int_0^12xe^{x^2}dx=e-1$ (Simpson $n=100$ : $1.7182818$).
8. Volume de la boule unité : Monte-Carlo avec $2\times10^5$ points dans le cube $[-1,1]^d$ donne $3.142$ pour $d=2$ (exact $\pi$) et $5.225$ pour $d=5$ (exact $5.264$). Une grille de 100 points par axe demanderait $10^{10}$ évaluations pour $d=5$.

## Implémentation

```python
import numpy as np

def trapezoid(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2)

def simpson(f, a, b, n):                      # n must be even
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])

def mc_integrate(f, a, b, n, rng):
    vals = (b - a) * f(rng.uniform(a, b, n))
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n)

def expectation(pdf, g, a, b, n=400):
    return simpson(lambda x: g(x) * pdf(x), a, b, n)
```

Bibliothèques : `scipy.integrate.quad` (Gauss-Kronrod adaptatif), `scipy.integrate.simpson`, `np.trapz`/`np.trapezoid` ; pour la grande dimension, quasi-Monte-Carlo (`scipy.stats.qmc`) et MCMC.

## Formes (shapes)

- Une fonction échantillonnée sur $n+1$ points de grille : tableau de forme $(n+1,)$. Pour une grille de dimension $d$, le tableau a $(n+1)^d$ entrées.
- Points de Monte-Carlo en dimension $d$ : forme $(n,d)$ ; valeurs de la fonction : $(n,)$ ; l'estimation est un scalaire et son erreur varie en $1/\sqrt n$ quel que soit $d$.

## Idées fausses courantes

- « La densité est une probabilité. » Seules les aires sont des probabilités ; la densité peut être supérieure à 1.
- « Probabilité zéro signifie impossible. » Chaque valeur individuelle a probabilité zéro pour une variable continue, et pourtant l'une d'elles se réalise.
- « La primitive existe sous forme close pour toute fonction raisonnable. » $e^{-x^2/2}$ n'en a pas ; l'intégrale gaussienne demande une astuce, et la plupart des intégrales en ML se font numériquement.
- « Monte-Carlo est une mauvaise méthode parce qu'elle converge lentement. » En grande dimension, c'est souvent la seule praticable.
- « Une règle d'ordre supérieur est toujours meilleure. » Simpson demande de la régularité ; pour des fonctions avec des coins ou des sauts (comme un indicateur), elle perd son avantage.

## Points clés à retenir

- L'intégrale est la limite des sommes de Riemann ; le théorème fondamental dit que $\int_a^bf=G(b)-G(a)$ pour une primitive $G$.
- La substitution inverse la règle en chaîne ; les parties inversent la règle du produit.
- Une densité est non négative d'aire totale 1 ; $\mathbb{E}[g(X)]=\int g\,p$ ; les marginales intègrent des variables.
- $\int e^{-x^2/2}dx=\sqrt{2\pi}$ par les coordonnées polaires, ce qui fixe la normalisation gaussienne.
- Les trapèzes sont $O(h^2)$, Simpson $O(h^4)$, Monte-Carlo $O(n^{-1/2})$ indépendamment de la dimension.

## Explique-le toi-même

1. Explique, sans formules, pourquoi la moyenne d'une fonction sur une entrée aléatoire est une aire sous « fonction fois densité ».
2. Explique pourquoi l'astuce des coordonnées polaires calcule l'intégrale gaussienne, et d'où vient le facteur supplémentaire $r$.

## Pour aller plus loin

- Deisenroth et al., chap. 6.2 et 6.5 ; Bishop Annexe B (table des distributions et de leurs moments).
- Leçon du dépôt `phases/01-math-foundations/22-stochastic-processes` pour le MCMC comme intégration par échantillonnage.
- Retourne à [[mx.04]] pour Metropolis-Hastings et à [[mx.02]] où la transformée de Fourier continue est définie par une intégrale.

## Critères de maîtrise

- [ ] Je sais évaluer des intégrales définies simples par le théorème fondamental, la substitution et les parties.
- [ ] Je sais écrire $P(a\le X\le b)$, une marginale et une espérance sous forme d'intégrales.
- [ ] Je sais dériver $\int e^{-x^2/2}dx=\sqrt{2\pi}$.
- [ ] Je sais implémenter les trapèzes, Simpson et Monte-Carlo et dire comment chaque erreur varie.
