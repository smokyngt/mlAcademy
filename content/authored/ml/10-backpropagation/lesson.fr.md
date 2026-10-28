---
id: f2n.10
title: La rétropropagation, dérivée avec les formes
summary: La règle de la chaîne à travers un réseau à deux couches, une formule à la fois, avec un exemple numérique pas à pas et une implémentation par batch.
minutes: 75
teaches: [nn.backprop]
requires: [calc.jacobian, nn.activation, nn.layer]
sources:
  - Bishop, Pattern Recognition and Machine Learning, Sec. 5.3 Error Backpropagation, PDF p.261-269
  - math/math for ML.pdf (Sec. 5.6 Backpropagation and automatic differentiation, PDF p.145-176)
  - ai-engineering-from-scratch/phases/03-deep-learning-core/03-backpropagation/docs/en.md
  - LLMs-from-scratch/appendix-A/01_main-chapter-code (automatic differentiation in PyTorch)
---

## Objectifs d'apprentissage

- Dériver les gradients de chaque paramètre d'un MLP à deux couches avec entropie croisée softmax.
- Donner la forme de chaque quantité intermédiaire et de chaque gradient.
- Expliquer ce qu'il faut mettre en cache pendant la passe avant et pourquoi.
- Dérouler à la main un petit exemple numérique en passe avant puis arrière.
- Implémenter la rétropropagation par batch et la vérifier avec un test par différences finies.

## Prérequis

Les trois formules arrière de la leçon 6 ([[f2n.06]]), le gradient de perte $p-y$ de la leçon 8 ([[f2n.08]]) et la notation des couches de la leçon 9 ([[f2n.09]]).

## Intuition

La passe avant calcule une chaîne de valeurs $X\to Z_1\to A_1\to Z_2\to P\to L$. Nous voulons $\partial L/\partial(\text{every weight})$. On part de la fin, où la dérivée de $L$ par rapport aux dernières valeurs est facile ($P-Y$), et on **remonte en arrière**, en convertissant « comment $L$ varie avec la sortie de cette étape » en « comment $L$ varie avec l'entrée de cette étape et avec les paramètres de cette étape ». Chaque étape fait une multiplication locale, donc le coût total de tous les gradients est d'environ celui de **deux** passes avant.

## Le réseau

Batch $X\in\mathbb{R}^{B\times n}$, largeur cachée $h$, $K$ classes, étiquettes $y\in\{0,\dots,K-1\}^B$ :

$$
\begin{aligned}
Z_1&=XW_1+b_1&&(B\times h),\ W_1:(n\times h),\ b_1:(h)\\
A_1&=\mathrm{ReLU}(Z_1)&&(B\times h)\\
Z_2&=A_1W_2+b_2&&(B\times K),\ W_2:(h\times K),\ b_2:(K)\\
P&=\operatorname{softmax}(Z_2)&&(B\times K)\\
L&=-\tfrac1B\textstyle\sum_{b}\ln P_{b,y_b}&&\text{scalar}
\end{aligned}
$$

## Dérivation arrière, pas à pas

Notation : $\bar{Q}:=\partial L/\partial Q$ a la même forme que $Q$.

**Étape 1 : perte et softmax** (leçon 8). $\;\bar Z_2=\dfrac1B\,(P-Y)\;\;(B\times K)$ où $Y$ est la matrice des étiquettes one-hot.

**Étape 2 : la seconde couche linéaire $Z_2=A_1W_2+b_2$.** Composante par composante, $(Z_2)_{bk}=\sum_j(A_1)_{bj}(W_2)_{jk}+(b_2)_k$.

- Poids : $\dfrac{\partial L}{\partial(W_2)_{jk}}=\sum_b\dfrac{\partial L}{\partial(Z_2)_{bk}}\dfrac{\partial(Z_2)_{bk}}{\partial(W_2)_{jk}}=\sum_b(A_1)_{bj}(\bar Z_2)_{bk}$, c'est-à-dire
$$
\bar W_2=A_1^\top\bar Z_2\qquad(h\times B)(B\times K)=(h\times K).
$$
- Biais : chaque $(b_2)_k$ apparaît dans toutes les lignes, donc $\bar b_2=\sum_b(\bar Z_2)_{b,:}$, de forme $(K)$.
- Entrée : $\dfrac{\partial L}{\partial(A_1)_{bj}}=\sum_k(\bar Z_2)_{bk}(W_2)_{jk}$, c'est-à-dire
$$
\bar A_1=\bar Z_2W_2^\top\qquad(B\times K)(K\times h)=(B\times h).
$$

Ce sont les $\nabla_xL=W^\top g$ et $\nabla_WL=gx^\top$ de la leçon 6 avec le batch empilé en lignes : les transposées passent de l'autre côté, et la somme sur le batch dans $A_1^\top\bar Z_2$ est la somme des produits externes $g_bx_b^\top$ de chaque exemple.

**Étape 3 : ReLU.** Terme à terme (leçon 6) : $\bar Z_1=\bar A_1\odot\mathbb{1}[Z_1>0]\ \ (B\times h)$.

**Étape 4 : la première couche linéaire** (même schéma qu'à l'étape 2) :
$$
\bar W_1=X^\top\bar Z_1\ (n\times h),\qquad\bar b_1=\textstyle\sum_b(\bar Z_1)_{b,:}\ (h).
$$
(On obtiendrait aussi $\bar X=\bar Z_1W_1^\top$, nécessaire seulement si $X$ provient lui-même d'une autre couche.)

**Résumé de ce qu'il faut mettre en cache pendant la passe avant :** $X$ (pour $\bar W_1$), $Z_1$ (le masque ReLU), $A_1$ (pour $\bar W_2$), $P$ (pour l'étape 1). C'est pourquoi la mémoire d'entraînement croît avec profondeur × taille du batch.

## Déroulé numérique (un exemple, sortie scalaire)

Pour garder des nombres minuscules : $x=(1,2)$, une unité cachée ReLU, une sortie sigmoïde avec entropie croisée binaire, vraie étiquette $y=1$. Paramètres $w_1=(0.5,-0.25)$, $b_1=0.1$, $w_2=2$, $b_2=-0.3$.

Passe avant :

- $z_1=0.5\cdot1+(-0.25)\cdot2+0.1=0.1$, $a_1=\mathrm{ReLU}(0.1)=0.1$
- $z_2=2\cdot0.1-0.3=-0.1$, $p=\sigma(-0.1)=0.47502$
- $L=-\ln0.47502=0.7444$

Passe arrière (chaque ligne est une formule) :

- $\bar z_2=p-y=-0.52498$
- $\bar w_2=\bar z_2\,a_1=-0.052498$, $\bar b_2=\bar z_2=-0.52498$
- $\bar a_1=\bar z_2\,w_2=-1.04996$
- $\bar z_1=\bar a_1\cdot\mathbb{1}[z_1>0]=-1.04996$ (l'unité est active)
- $\bar w_1=\bar z_1\,x=(-1.04996,\,-2.09992)$, $\bar b_1=-1.04996$

Une vérification par différences finies sur $w_{1,1}$ avec $h=10^{-6}$ donne $-1.04996$ ✓. Tous les gradients sont négatifs, donc la descente de gradient *augmente* $w_1,w_2,b$ : cela élève $z_2$ et donc $p$ vers l'étiquette 1. Le modèle apprend dans la bonne direction.

## Implémentation

```python
import numpy as np

def forward_backward(params, X, y):
    W1, b1, W2, b2 = params
    B = len(y)
    Z1 = X @ W1 + b1;  A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z = Z2 - Z2.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(B), y]))

    dZ2 = P.copy(); dZ2[np.arange(B), y] -= 1; dZ2 /= B
    dW2 = A1.T @ dZ2;  db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1;   db1 = dZ1.sum(axis=0)
    return loss, (dW1, db1, dW2, db2)
```

## Liste de contrôle pour le débogage (les formes attrapent la plupart des bugs)

1. Chaque gradient doit avoir exactement la forme de son paramètre.
2. Compare avec un gradient par différences finies sur un minuscule réseau (exercice ci-dessous). Une erreur relative inférieure à $10^{-6}$ est bonne, supérieure à $10^{-3}$ c'est un bug.
3. La perte initiale doit être proche de $\ln K$ (leçon 8).
4. Un réseau incapable de sur-apprendre 8 exemples a un bug, pas un problème de capacité.

## Idées reçues

- **« La rétropropagation ne sert que pour les réseaux de neurones. »** C'est de la différentiation automatique en mode inverse (leçon 12) et elle fonctionne sur n'importe quel graphe de calcul.
- **« Les poids sont mis à jour couche par couche pendant la passe arrière. »** Non : calcule d'abord tous les gradients, puis mets à jour. Mettre à jour en cours de route change les valeurs dont dépendent les formules suivantes.
- **« Les gradients traversent tout le réseau de façon égale. »** Ils sont multipliés par chaque dérivée locale rencontrée ; des facteurs inférieurs à 1 (sigmoïde) les réduisent (évanescence), des facteurs supérieurs à 1 les font croître (explosion).
- **Oublier la somme sur le batch** dans $\bar b$ ou le $1/B$ dans $\bar Z_2$ produit des gradients faux d'exactement un facteur $B$ : un bug subtil que les différences finies révèlent.

## Points clés à retenir

- La rétropropagation est la règle de la chaîne appliquée de la perte vers l'arrière, en réutilisant les gradients intermédiaires.
- Formules par batch : `dW = X^T dZ`, `db = sum_b dZ`, `dX = dZ W^T`.
- Mets en cache `X`, `Z1` (masque ReLU), `A1`, `P` pendant la passe avant.
- Chaque gradient doit avoir la forme de son paramètre ; les différences finies tranchent les désaccords.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. À partir des seules équations de la passe avant, dérive les quatre gradients du réseau à deux couches avec leurs formes.
2. Pourquoi oublier le `1/B` produit-il des gradients faux d'exactement un facteur `B` ?

## Pour aller plus loin

- Bishop, *PRML*, Sec. 5.3.1-5.3.2, PDF p.261-265.
- *Mathematics for Machine Learning*, Sec. 5.6, PDF p.145-176.
- Bibliothèque source : `ai-engineering-from-scratch` phase 03 leçon 03 ; Raschka, *LLMs from scratch*, annexe A.

## Critères de maîtrise

Sur papier, à partir des seules équations de la passe avant, tu sais produire les quatre formules de gradient avec leurs formes, et ton implémentation réussit un test de gradient avec une erreur relative $<10^{-6}$.
