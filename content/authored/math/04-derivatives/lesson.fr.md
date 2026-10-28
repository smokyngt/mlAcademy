---
id: f2n.04
title: "Dérivées et dérivation numérique"
summary: "La dérivée comme taux de variation instantané, les règles qui la calculent, et les différences finies qui les vérifient."
minutes: 55
teaches: [calc.derivative, calc.numerical-diff]
requires: [alg.functions, alg.exp-log]
sources:
  - math/math for ML.pdf (Ch. 5 Vector Calculus, Sec. 5.1 Differentiation of univariate functions, PDF p.145)
  - ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md
---

## Objectifs d'apprentissage

- Énoncer la définition de la dérivée par une limite et l'utiliser pour dériver $x^2$.
- Appliquer les règles de la puissance, de la somme, du produit, de l'exponentielle et du logarithme.
- Démontrer $\sigma'(x)=\sigma(x)(1-\sigma(x))$ pour la sigmoïde.
- Calculer des différences finies avant et centrées, et expliquer leurs ordres d'erreur et le meilleur pas.

## Prérequis

Fonctions, exponentielles et logarithmes ([[f2n.01]]).

## Où cela s'inscrit

Apprendre est une boucle : essayer le modèle, mesurer à quel point il se trompe, ajuster, recommencer. La seconde moitié, ajuster, demande une information par bouton de réglage : **si je tourne ce bouton un tout petit peu vers le haut, l'erreur monte-t-elle ou descend-elle, et de combien ?** Cette quantité est la dérivée.

Tout ce qui suit (gradients, règle de la chaîne, descente de gradient, rétropropagation) est cette unique idée appliquée à de plus en plus de boutons. Cette leçon sert donc à comprendre parfaitement le cas d'un seul bouton, plus un moyen pratique de vérifier tes réponses avec un ordinateur.

## L'idée en mots simples

- Pense au **robinet d'une douche**. Tourne-le un peu et la température de l'eau change. « Un petit tour donne tel changement de température, par unité de rotation » est la dérivée : une *sensibilité*. Raide signifie que l'eau réagit fortement ; plat, qu'elle réagit à peine ; négatif, qu'ouvrir davantage la refroidit.
- Sur un graphique, c'est la **pente** de la courbe en un point. Pour une droite, la pente est la même partout. Pour une courbe elle change, mais si tu **zoomes** assez sur n'importe quelle courbe lisse, elle ressemble à une droite, et la pente de cette droite est la dérivée.
- Compteur de vitesse contre compteur kilométrique : le kilométrique dit où tu en es, le compteur de vitesse dit à quelle vitesse la position change. La dérivée est le compteur de vitesse d'une fonction.
- Tu peux aussi *estimer* la dérivée sans algèbre : pousse l'entrée d'une toute petite quantité `h`, regarde de combien la sortie a bougé, et divise. C'est ce que fait une vérification par ordinateur, et sa seule subtilité (prendre `h` petit, mais pas trop petit) est expliquée plus bas.

Déplace `h` dans la figure et regarde la droite oblique se transformer en tangente.

## Intuition

La dérivée $f'(x)$ répond à : *si je pousse l'entrée d'une toute petite quantité, de combien la sortie bouge-t-elle, par unité de poussée ?* C'est la pente de la tangente. Entraîner un modèle ne consiste qu'à répéter « pousse chaque paramètre dans la direction qui diminue la perte », donc les dérivées sont le moteur de l'apprentissage.

Fais tendre $h$ vers zéro : la sécante passant par $(x,f(x))$ et $(x+h,f(x+h))$ devient la tangente.

```viz
derivative-explorer
```

## Définition formelle

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
$$

Si la limite existe, $f$ est dérivable en $x$. Énoncé équivalent au premier ordre : $f(x+h)\approx f(x)+f'(x)\,h$ pour $h$ petit. Cette **approximation linéaire locale** est l'idée qui se généralisera plus tard aux gradients et aux jacobiennes.

## Démonstration 1 : la règle de la puissance pour $x^2$

$$
\frac{(x+h)^2-x^2}{h}=\frac{x^2+2xh+h^2-x^2}{h}=\frac{2xh+h^2}{h}=2x+h\;\xrightarrow{h\to0}\;2x .
$$

La règle générale est $\frac{d}{dx}x^n=nx^{n-1}$ (le théorème du binôme donne $(x+h)^n=x^n+nx^{n-1}h+O(h^2)$).

## Règles que tu utiliseras constamment

| Règle | Énoncé |
|---|---|
| Multiple d'une constante | $(cf)'=cf'$ |
| Somme | $(f+g)'=f'+g'$ |
| Produit | $(fg)'=f'g+fg'$ |
| Puissance | $(x^n)'=nx^{n-1}$ |
| Exponentielle | $(e^x)'=e^x$ |
| Logarithme | $(\ln x)'=1/x$ pour $x>0$ |

(La règle de la chaîne est pour l'avant-dernière leçon d'après, après les gradients.)

## Démonstration 2 : la sigmoïde

$\sigma(x)=\dfrac{1}{1+e^{-x}}=(1+e^{-x})^{-1}$. Posons $u=1+e^{-x}$, donc $\sigma=u^{-1}$, $\sigma'=-u^{-2}u'$ et $u'=-e^{-x}$. Par conséquent

$$
\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\sigma(x)\,\big(1-\sigma(x)\big),
$$

puisque $\frac{e^{-x}}{1+e^{-x}}=1-\frac{1}{1+e^{-x}}$. En $x=0$ : $\sigma(0)=\tfrac12$ et $\sigma'(0)=\tfrac14$, la plus grande valeur qu'elle prend, ce qui explique pourquoi les réseaux profonds à sigmoïdes souffrent d'évanouissement du gradient (chaque couche multiplie par au plus $0.25$).

## Dérivation numérique

On peut approcher $f'$ sans algèbre :

- **Différence avant :** $D_+ f(x)=\dfrac{f(x+h)-f(x)}{h}$
- **Différence centrée :** $D_0 f(x)=\dfrac{f(x+h)-f(x-h)}{2h}$

**Analyse de l'erreur.** Théorème de Taylor : $f(x\pm h)=f(x)\pm hf'(x)+\tfrac{h^2}{2}f''(x)\pm\tfrac{h^3}{6}f'''(x)+\cdots$.

- Avant : $D_+f=f'+\tfrac h2f''+\cdots$, donc l'*erreur de troncature* est $O(h)$.
- Centrée : en soustrayant les deux développements, les termes pairs s'annulent, $f(x+h)-f(x-h)=2hf'+\tfrac{h^3}{3}f'''+\cdots$, donc $D_0f=f'+\tfrac{h^2}{6}f'''+\cdots$, erreur $O(h^2)$.

Mais la virgule flottante ajoute une *erreur d'arrondi*. Le numérateur soustrait deux nombres presque égaux, perdant environ $\varepsilon|f|$ de précision absolue, et diviser par $h$ donne une erreur $\sim\varepsilon/h$. L'erreur totale vaut $\approx Ah^p+\varepsilon/h$ ; elle diminue quand $h$ rétrécit, puis *remonte*. La minimisation donne $h\approx\varepsilon^{1/2}\approx10^{-8}$ pour les différences avant et $h\approx\varepsilon^{1/3}\approx10^{-5}$ pour les différences centrées en float64 ($\varepsilon\approx2.2\times10^{-16}$).

## Exemple traité

$f(x)=x^3$ en $x=2$ ; vraie dérivée $3x^2=12$.

- Avant, $h=0.1$ : $(2.1^3-2^3)/0.1=(9.261-8)/0.1=12.61$ (erreur $0.61$).
- Centrée, $h=0.1$ : $(2.1^3-1.9^3)/0.2=(9.261-6.859)/0.2=12.01$ (erreur $0.01$, en accord avec $\tfrac{h^2}{6}f'''=\tfrac{0.01}{6}\cdot6$).

Les différences centrées sont environ 60 fois plus précises ici pour le même $h$.

## Implémentation

```python
import numpy as np

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
print(central_diff(sigmoid, 0.0))   # 0.25
```

## Idées fausses courantes

- **« Plus $h$ est petit, mieux c'est. »** Non : en dessous de l'optimum, l'erreur d'arrondi domine et la réponse devient *pire*. L'exercice te le fait voir.
- **Dérivable est plus fort que continu.** $|x|$ est continue en 0 mais n'y a pas de dérivée ; ReLU a ce point anguleux, et les frameworks choisissent simplement une valeur (0) pour la dérivée en 0.
- **$(fg)'\neq f'g'$.**
- **Les dérivées numériques servent à vérifier, pas à entraîner** : elles coûtent une évaluation de fonction par paramètre. La rétropropagation calcule toutes les dérivées en une seule passe arrière.

## Points clés à retenir

- La dérivée est la pente de la tangente : `f(x+h) ~ f(x) + f'(x) h`.
- Apprends les six règles (constante, somme, produit, puissance, exp, ln) et démontre toi-même une fois la dérivée de la sigmoïde.
- Les différences centrées sont en `O(h^2)` ; les différences avant en `O(h)`.
- Un `h` plus petit n'est pas toujours meilleur : l'arrondi croît comme `eps/h`, donc le meilleur `h` est autour de `1e-5`.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Démontre `sigma'(x) = sigma(x)(1 - sigma(x))` de mémoire. Pourquoi son maximum est-il 0,25, et qu'est-ce que cela fait aux réseaux profonds à sigmoïdes ?
2. Pourquoi une dérivée numérique sert-elle à tester du code mais pas à entraîner un réseau ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Sec. 5.1, PDF p.145.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01, leçon 04.

## Critères de maîtrise

Tu sais dériver à la main une expression polynomiale, exponentielle ou logarithmique, démontrer $\sigma'$, et expliquer pourquoi une vérification de gradient utilise des différences centrées avec $h\approx10^{-5}$.
