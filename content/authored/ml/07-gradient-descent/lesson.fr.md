---
id: f2n.07
title: Descente de gradient
summary: Pourquoi avancer contre le gradient fait baisser la perte, quand l'algorithme converge, et comment le taux d'apprentissage et le conditionnement décident de tout.
minutes: 60
teaches: [opt.gradient-descent]
requires: [calc.gradient, calc.numerical-diff]
sources:
  - math/math for ML.pdf (Ch. 7 Continuous Optimization, Sec. 7.1 Optimization using gradient descent, PDF p.231)
  - ai-engineering-from-scratch/phases/01-math-foundations/08-optimization/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/06-optimizers/docs/en.md
---

## Objectifs d'apprentissage

- Écrire la mise à jour de la descente de gradient et la justifier par un argument du premier ordre.
- Dériver la condition de convergence exacte $0<\eta<2/\lambda$ pour une quadratique et l'effet du conditionnement.
- Prédire la divergence, la convergence lente et les oscillations à partir du taux d'apprentissage.
- Implémenter la descente de gradient et l'utiliser pour ajuster un modèle linéaire.

## Prérequis

Gradients ([[f2n.05]]) et vérification numérique ([[f2n.04]]).

## Intuition

Tu es sur un versant embrumé et tu veux atteindre la vallée. Tu ne sens que la pente sous tes pieds. Fais un petit pas vers le bas, ressens à nouveau, recommence. Le gradient t'indique « vers le haut » ; tu avances dans le sens opposé. La taille du pas $\eta$ (le *taux d'apprentissage*) concentre toute la difficulté : trop petit et tu rampes, trop grand et tu dépasses la vallée et rebondis hors de celle-ci.

Commence par expérimenter. Choisis un taux d'apprentissage et un nombre de conditionnement $\kappa$ (à quel point un axe est plus raide que l'autre) et observe la trajectoire.

```viz
gradient-descent
```

## L'algorithme

Minimiser $L:\mathbb{R}^n\to\mathbb{R}$. En partant de $\theta_0$,

$$
\theta_{t+1}=\theta_t-\eta\,\nabla L(\theta_t).
$$

**Pourquoi la perte diminue (pour $\eta$ petit).** Le développement du premier ordre de la leçon 5 avec le pas $\Delta=-\eta g$, où $g=\nabla L(\theta)$ :

$$
L(\theta-\eta g)\approx L(\theta)+\nabla L(\theta)\cdot(-\eta g)=L(\theta)-\eta\|g\|^2 .
$$

Comme $\|g\|^2\ge0$, la perte baisse dès que $\eta>0$ est assez petit pour que l'approximation soit valide, et elle baisse le plus vite par unité de pas le long de $-g$ (leçon 5). L'algorithme cesse de bouger quand $g=0$.

## Dérivation : quand converge-t-elle ? (la quadratique)

En une dimension : $L(\theta)=\tfrac12\lambda\theta^2$ avec une courbure $\lambda>0$ et un minimum en 0. Alors $L'(\theta)=\lambda\theta$ et

$$
\theta_{t+1}=\theta_t-\eta\lambda\theta_t=(1-\eta\lambda)\,\theta_t\quad\Rightarrow\quad\theta_t=(1-\eta\lambda)^t\theta_0 .
$$

Cela converge vers 0 si et seulement si $|1-\eta\lambda|<1$, c'est-à-dire $0<\eta<\dfrac{2}{\lambda}$. Trois régimes :

- $0<\eta<1/\lambda$ : convergence monotone (facteur dans $(0,1)$).
- $1/\lambda<\eta<2/\lambda$ : converge **avec oscillations** (facteur dans $(-1,0)$).
- $\eta>2/\lambda$ : $|1-\eta\lambda|>1$, **divergence**.

En $n$ dimensions, avec $L(\theta)=\tfrac12\theta^\top H\theta$ et les valeurs propres $0<\lambda_{\min}\le\dots\le\lambda_{\max}$ de $H$, chaque direction propre se comporte comme le cas 1-D avec son propre $\lambda_i$. Le pas doit vérifier $\eta<2/\lambda_{\max}$ pour éviter de faire exploser la direction la plus raide, mais alors la direction la plus plate ne se contracte que de $1-\eta\lambda_{\min}$ par pas, ce qui est lent. Le meilleur choix $\eta=\dfrac{2}{\lambda_{\max}+\lambda_{\min}}$ donne la contraction par pas

$$
\rho=\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}=\frac{\kappa-1}{\kappa+1},\qquad\kappa=\frac{\lambda_{\max}}{\lambda_{\min}} .
$$

Avec $\kappa=1$, elle converge en un pas ; avec $\kappa=100$, $\rho\approx0.98$ et il faut environ 100 pas pour réduire l'erreur d'un facteur $e^{-2}$. **Le mauvais conditionnement, et pas seulement la dimension, est ce qui rend l'optimisation lente.** Cela motive le momentum, Adam, les couches de normalisation et une bonne initialisation.

## Exemples résolus

1. $L(\theta)=\theta^2$ ($\lambda=2$), $\theta_0=3$, $\eta=0.1$ : $\theta_1=3-0.1\cdot6=2.4$, $\theta_2=2.4-0.1\cdot4.8=1.92$. Le rapport vaut $0.8=1-\eta\lambda$. ✓ Avec $\eta=1.1$ : rapport $1-2.2=-1.2$, donc $3\to-3.6\to4.32\to\cdots$ diverge.
2. $L(x,y)=\tfrac12(x^2+10y^2)$, donc $\lambda_{\max}=10,\lambda_{\min}=1$, $\kappa=10$. La stabilité exige $\eta<0.2$. Le meilleur $\eta=2/11\approx0.182$ avec $\rho=9/11\approx0.818$. Avec $\eta=0.19$, le facteur de la direction $y$ est $1-1.9=-0.9$ : convergence lente avec oscillations.

## Descente de gradient stochastique

Pour une perte qui est une moyenne sur les données, $L=\frac1N\sum_iL_i$, le calcul du gradient coûte $O(N)$. Utiliser un mini-batch aléatoire $B$ de taille $b$ donne $\hat g=\frac1b\sum_{i\in B}\nabla L_i$ avec $\mathbb{E}[\hat g]=\nabla L$ (non biaisé) et une variance qui décroît en $1/b$. La SGD échange des gradients exacts contre de nombreux pas peu coûteux ; le bruit aide aussi à quitter les minima aigus et les points selle. Le momentum, RMSProp et Adam (phase 03 de la bibliothèque source) changent la *direction/l'échelle* du pas, mais la forme de la mise à jour reste la même.

## Implémentation

```python
import numpy as np

def gradient_descent(grad, x0, lr, steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad(x)
        path.append(x.copy())
    return np.array(path)        # shape (steps+1, n)
```

## Idées reçues

- **« La descente de gradient trouve le minimum global. »** Seulement pour des pertes convexes (comme l'EQM de la régression linéaire). Les réseaux de neurones sont non convexes ; tu atteins une bonne solution *locale*.
- **« Un taux d'apprentissage plus bas est toujours plus sûr. »** C'est plus sûr contre la divergence, mais tu peux rester bloqué sur un plateau ou épuiser tout ton budget de calcul.
- **« La perte doit diminuer à chaque pas. »** Vrai pour la descente de gradient exacte avec $\eta$ petit ; faux pour la SGD, où elle est bruitée mais décroissante en moyenne.
- **Une perte qui diverge (NaN/inf) vient généralement du taux d'apprentissage**, pas de « mauvaises données ». Divise par 10 et réessaie.

## Points clés à retenir

- `theta <- theta - eta grad L` ; pour `eta` petit, la perte varie d'environ `-eta |grad|^2`.
- Sur une quadratique de courbure `lambda`, elle converge si et seulement si `0 < eta < 2/lambda`.
- Le nombre de conditionnement `kappa = lambda_max / lambda_min` fixe la lenteur de la convergence : `(kappa-1)/(kappa+1)` par pas au mieux.
- Un NaN en début d'entraînement signifie généralement que le taux d'apprentissage est trop grand.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Prédis ce qui se passe pour `eta` = 0.5/lambda, 1.5/lambda, 2.5/lambda, puis vérifie avec la figure.
2. Pourquoi une perte mal conditionnée ralentit-elle la descente de gradient, même en deux dimensions ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 7.1 (descente de gradient, pas, momentum, SGD), PDF p.231.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01 leçon 08 et phase 03 leçon 06.

## Critères de maîtrise

Tu sais prédire, à partir de la courbure, la plage de taux d'apprentissage qui convergent, calculer le taux optimal pour une quadratique 2-D, et diagnostiquer une divergence à partir d'une courbe de perte.
