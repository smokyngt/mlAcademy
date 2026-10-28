---
id: f2n.08
title: "Fonctions de perte : EQM, entropie croisée et softmax"
summary: "D'où viennent les pertes (maximum de vraisemblance), et les gradients propres p - y qui stabilisent l'entraînement en classification."
minutes: 65
teaches: [ml.loss]
requires: [alg.exp-log, alg.summation]
sources:
  - math/math for ML.pdf (Ch. 8 When Models Meet Data, Sec. 8.2 Empirical risk minimisation and 8.3 Parameter estimation, PDF p.257)
  - Bishop, Pattern Recognition and Machine Learning, Sec. 1.2 Probability theory and Sec. 4.3.2 Logistic regression, PDF p.32 and p.223
  - ai-engineering-from-scratch/phases/03-deep-learning-core/05-loss-functions/docs/en.md
---

## Objectifs d'apprentissage

- Dériver l'erreur quadratique moyenne à partir d'un modèle de bruit gaussien et l'entropie croisée binaire à partir d'un modèle de Bernoulli.
- Définir le softmax et l'entropie croisée catégorielle, et les calculer de façon stable avec le log-sum-exp.
- Dériver $\partial L/\partial z = p - y$ pour sigmoïde+BCE et softmax+CE.
- Donner les formes (shapes) pour un batch de $B$ exemples et $K$ classes.

## Prérequis

Logarithmes et sommes ([[f2n.01]]) ; motivation de la descente de gradient ([[f2n.07]]).

## Intuition

Une perte est un nombre qui mesure « à quel point on se trompe ». On la choisit pour que les meilleurs paramètres soient ceux qui rendent les données observées *les plus probables* (maximum de vraisemblance). Différentes hypothèses sur la façon dont les données sont générées donnent différentes pertes : des cibles continues avec un bruit additif donnent l'erreur quadratique ; des résultats oui/non donnent l'entropie croisée.

## Le maximum de vraisemblance en un paragraphe

Choisis un modèle $p_\theta(y\mid x)$. Pour des exemples indépendants, la vraisemblance est $\prod_ip_\theta(y_i\mid x_i)$. La maximiser équivaut à minimiser la **log-vraisemblance négative**

$$
\mathrm{NLL}(\theta)=-\sum_i\ln p_\theta(y_i\mid x_i)
$$

(passer aux logarithmes transforme le produit en somme, leçon 1, et $-\ln$ est décroissante).

## Dérivation 1 : l'EQM à partir d'une gaussienne

Suppose $y=f_\theta(x)+\varepsilon$ avec $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. Alors $p(y\mid x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\big(-\frac{(y-f_\theta(x))^2}{2\sigma^2}\big)$ et

$$
-\ln p(y_i\mid x_i)=\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}+\ln(\sigma\sqrt{2\pi}).
$$

Le second terme ne dépend pas de $\theta$, donc minimiser la NLL sur $\theta$ revient exactement à minimiser $\sum_i(y_i-f_\theta(x_i))^2$. L'erreur quadratique moyenne est donc le « maximum de vraisemblance sous bruit gaussien ». Elle pénalise les grosses erreurs de façon quadratique, donc elle est sensible aux valeurs aberrantes.

## Dérivation 2 : l'entropie croisée binaire à partir d'une Bernoulli

Cibles $y\in\{0,1\}$, sortie du modèle $p=\sigma(z)$ interprétée comme $P(y=1)$. La vraisemblance de Bernoulli est $p^y(1-p)^{1-y}$, donc

$$
L=-\big[y\ln p+(1-y)\ln(1-p)\big].
$$

**Gradient par rapport au logit $z$.** Par la règle de la chaîne (leçon 6), $\frac{\partial L}{\partial z}=\frac{\partial L}{\partial p}\frac{dp}{dz}$ avec

$$
\frac{\partial L}{\partial p}=-\frac yp+\frac{1-y}{1-p},\qquad\frac{dp}{dz}=p(1-p).
$$

$$
\frac{\partial L}{\partial z}=\Big(-\frac yp+\frac{1-y}{1-p}\Big)p(1-p)=-y(1-p)+(1-y)p=p-y .
$$

La dérivée de la sigmoïde annule les dénominateurs, ce qui laisse le très simple **prédiction moins cible**. Contrairement à MSE+sigmoïde (dont le gradient porte un facteur $p(1-p)$ qui s'annule quand le modèle se trompe avec assurance), l'entropie croisée conserve un gradient fort précisément quand l'erreur est grande.

## Dérivation 3 : softmax et entropie croisée catégorielle

Pour $K$ classes avec des logits $z\in\mathbb{R}^K$,

$$
p_k=\operatorname{softmax}(z)_k=\frac{e^{z_k}}{\sum_je^{z_j}}.
$$

Les sorties du softmax sont positives, de somme 1, et invariantes quand on ajoute une constante $c$ à chaque logit (le $e^c$ se simplifie). Cette invariance est la clé de la stabilité numérique : soustrais d'abord $\max_jz_j$. Avec la vraie classe $y$, la perte est $L=-\ln p_y$. En substituant,

$$
L=-z_y+\ln\sum_je^{z_j}=-z_y+\operatorname{logsumexp}(z).
$$

En dérivant par rapport à $z_k$ (avec $\partial\,\mathrm{logsumexp}/\partial z_k=p_k$) :

$$
\frac{\partial L}{\partial z_k}=p_k-\mathbb{1}[k=y]\quad\Longleftrightarrow\quad\nabla_zL=p-\text{onehot}(y).
$$

On obtient la même forme de réponse que dans le cas binaire. Ne calcule jamais $\ln(\operatorname{softmax}(z))$ en prenant le `log` d'un softmax : utilise $z_y-\operatorname{logsumexp}(z)$ (c'est ce que fait `log_softmax`).

## Exemple résolu

Logits $z=(2,1,0.1)$, vraie classe $y=0$. Exponentielles : $e^2=7.389$, $e^1=2.718$, $e^{0.1}=1.105$, somme $=11.212$. Donc $p=(0.659,0.242,0.099)$ et

$$
L=-\ln0.659=0.417,\qquad\nabla_zL=p-(1,0,0)=(-0.341,\,0.242,\,0.099).
$$

Le gradient a une somme nulle (ajouter une constante à tous les logits ne peut pas changer la perte), et il pousse le logit de la vraie classe vers le haut et les autres vers le bas.

## Formes pour un batch

Logits $Z\in\mathbb{R}^{B\times K}$, étiquettes entières $y\in\{0..K-1\}^B$.

| Grandeur | Forme |
|---|---|
| $P=\operatorname{softmax}(Z)$ (ligne par ligne) | $B\times K$ |
| $L=-\frac1B\sum_b\ln P_{b,y_b}$ | scalaire |
| $\nabla_ZL=\frac1B(P-Y_{\text{onehot}})$ | $B\times K$ |

Le facteur $\frac1B$ vient de la moyenne sur le batch ; si tu l'oublies, ton taux d'apprentissage effectif varie avec la taille du batch.

## Implémentation

```python
import numpy as np

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)      # stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Z, y):
    m = Z.max(axis=1, keepdims=True)
    logsumexp = (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True))).ravel()
    return np.mean(logsumexp - Z[np.arange(len(y)), y])
```

## Idées reçues

- **L'entropie croisée n'est pas « juste une autre distance ».** C'est la NLL d'un modèle catégoriel, et elle vaut $H(q,p)=-\sum_kq_k\ln p_k$ pour une cible one-hot $q$.
- **N'applique pas le softmax puis l'entropie croisée en deux étapes numériquement naïves.** `log(softmax(z))` dépasse par le bas vers `-inf`. Utilise la forme fusionnée avec log-sum-exp.
- **L'EQM n'est pas mauvaise en classification parce qu'elle serait « moins précise »** ; elle s'ajuste mal à cause du comportement de gradient évanescent décrit plus haut.
- **Une perte de 0.693 sur une tâche binaire** ($=\ln2$) signifie que tu prédis 50/50 : le modèle n'a encore rien appris. De même $\ln K$ pour $K$ classes. Retiens ce test de cohérence.

## Points clés à retenir

- Une perte est une log-vraisemblance négative sous un modèle de bruit choisi : gaussien donne l'EQM, Bernoulli/catégoriel donne l'entropie croisée.
- Sigmoïde + BCE et softmax + CE donnent tous deux le gradient propre `p - y`.
- Calcule avec le log-sum-exp : jamais `log(softmax(z))`.
- Un modèle neuf à `K` classes doit démarrer près d'une perte `ln K`.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Dérive `dL/dz = p - y` pour sigmoïde + entropie croisée binaire sans regarder.
2. Pourquoi ajouter une constante à chaque logit laisse-t-il le softmax inchangé, et pourquoi cela le rend-il stable ?

## Pour aller plus loin

- *Mathematics for Machine Learning*, Ch. 8 (risque empirique, maximum de vraisemblance), PDF p.257.
- Bishop, *PRML*, Sec. 4.3.2-4.3.3 (régression logistique et entropie croisée), PDF p.223.
- Bibliothèque source : `ai-engineering-from-scratch` phase 03 leçon 05.

## Critères de maîtrise

Tu sais dériver $p-y$ sur papier pour sigmoïde+BCE et softmax+CE, nommer la perte initiale attendue ($\ln K$), et implémenter une entropie croisée stable qui survit à des logits de 1000.
