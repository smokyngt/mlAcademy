---
id: f2n.11
title: Entraîner un réseau à partir de zéro
summary: L'initialisation, la boucle d'entraînement, les mini-batches, le momentum, et une méthode systématique pour déboguer un réseau qui n'apprend pas.
minutes: 70
teaches: [nn.training]
requires: [nn.backprop, opt.gradient-descent]
sources:
  - ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/10-mini-framework/docs/en.md
  - ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/docs/en.md
  - math/math for ML.pdf (Sec. 7.1 gradient descent with momentum and SGD, PDF p.231)
---

## Objectifs d'apprentissage

- Assembler la passe avant, la perte, la passe arrière et la mise à jour en une boucle d'entraînement.
- Initialiser les poids avec l'échelle de He et dire pourquoi.
- Expliquer la SGD par mini-batches et le momentum avec leurs équations de mise à jour.
- Appliquer l'échelle de débogage standard : perte initiale, test de gradient, sur-apprentissage d'un batch, balayage du taux d'apprentissage.
- Entraîner un réseau à deux couches pour séparer deux demi-lunes entrelacées.

## Prérequis

Rétropropagation ([[f2n.10]]) et descente de gradient ([[f2n.07]]).

## Intuition

L'entraînement, c'est simplement la descente de gradient (leçon 7) sur la perte du réseau (leçon 8) avec des gradients issus de la rétropropagation (leçon 10). Le savoir-faire consiste à bien *démarrer* (initialisation), *tourner à peu de frais* (mini-batches), *avancer efficacement* (momentum, taux d'apprentissage) et *te signaler quand quelque chose ne va pas* (diagnostics).

## La boucle

Pour chaque epoch : mélange les données ; pour chaque mini-batch de taille $b$ :

1. **Passe avant :** calcule les prédictions et la perte du batch.
2. **Passe arrière :** calcule $\nabla_\theta L$ (leçon 10).
3. **Mise à jour :** $\theta\leftarrow\theta-\eta\,\nabla_\theta L$.

Ne mets jamais à jour avant que tous les gradients soient calculés, et remets toujours à zéro (ou recalcule) les gradients à chaque batch : accumuler des gradients périmés est le bug classique de PyTorch.

## Initialisation (rappel de la dérivation)

La leçon 9 a montré $\operatorname{Var}(z)=n\sigma_w^2\sigma_x^2$. Pour les réseaux ReLU, utilise $W_{ij}\sim\mathcal{N}(0,2/n_{\text{in}})$ (He) et $b=0$. Vérifie l'effet empiriquement : fais passer des données aléatoires à travers 20 couches ReLU avec $\sigma_w^2=1/n$ (les activations décroissent de $2^{-20}$ en moment d'ordre deux) contre $2/n$ (stable).

## Momentum

La SGD simple fait des zigzags dans une vallée étroite (leçon 7 : grand $\kappa$). Le **momentum** conserve une vitesse courante :

$$
v\leftarrow\mu v-\eta\,g,\qquad\theta\leftarrow\theta+v,\qquad\mu\approx0.9 .
$$

Les composantes du gradient qui changent sans cesse de signe s'annulent dans $v$ ; celles qui s'accordent s'accumulent, donc le pas effectif le long d'une direction constante vaut environ $\eta/(1-\mu)$ ($10\times$ plus grand pour $\mu=0.9$). Adam rééchelonne en plus chaque paramètre par une estimation courante de la magnitude de son gradient ; tu le rencontreras dans la bibliothèque source (phase 03, leçon 06).

## Une échelle de débogage systématique

Descends cette liste ; chaque étape isole une classe de bugs différente.

1. **Perte initiale $\approx\ln K$ ?** Sinon, l'échelle de sortie ou la perte est erronée.
2. **Test de gradient** sur un minuscule réseau et batch (leçon 10). Attrape les bugs de la passe arrière.
3. **Sur-apprendre 8 exemples.** Avec assez de capacité, la perte doit tendre vers ~0 et la précision vers 100 %. Sinon : bug dans les données/étiquettes, le signe de la mise à jour, ou le taux d'apprentissage.
4. **Balayage du taux d'apprentissage** : essaie $\eta\in\{10^{-3},10^{-2},10^{-1},1\}$ pendant quelques centaines de pas et choisis le plus grand qui ne diverge pas.
5. **Seulement ensuite** entraîne sur toutes les données et surveille la perte de *validation*. Une divergence entre les courbes d'entraînement et de validation signifie du sur-apprentissage, pas un bug.

Symptôme → cause typique : perte bloquée à $\ln K$ → $\eta$ beaucoup trop petit, ReLU mortes, ou graphe détaché ; `nan` → $\eta$ trop grand ou log de 0 ; la perte d'entraînement baisse mais la précision reste plate → désalignement étiquettes/ordre entre $X$ et $y$ après le mélange.

## Exemple résolu : les demi-lunes

Deux arcs entrelacés de points 2-D ne sont pas linéairement séparables, donc la régression logistique (une seule frontière linéaire) plafonne autour de 89 % sur les données de l'exercice. Un réseau ReLU $2\to16\to2$ avec initialisation de He et $\eta=0.5$ en descente de gradient sur batch complet pendant 400 pas atteint environ 98 % de précision d'entraînement (mesuré : 98.5 %, perte de 0.60 à 0.04) ; l'exercice te fournit un banc d'essai pour vérifier cette affirmation et pour expérimenter avec d'autres largeurs, taux d'apprentissage et momentum.

## Esquisse d'implémentation

```python
params = init_params(2, 16, 2, seed=0)
for epoch in range(400):
    loss, grads = forward_backward(params, X, y)
    params = sgd_step(params, grads, lr=0.5)
```

## Idées reçues

- **« Si la perte baisse, le modèle fonctionne. »** Vérifie sur un jeu mis de côté ; vérifie aussi que les prédictions ne se réduisent pas à la classe majoritaire.
- **« Un plus grand taux d'apprentissage est plus rapide. »** Seulement jusqu'à la limite de stabilité ; au-delà, tu divergeas.
- **« Les graines aléatoires n'ont pas d'importance. »** Sur de petits réseaux, elles peuvent changer la précision finale de plusieurs points ; rapporte la moyenne et la dispersion sur plusieurs graines (voir statistics-for-ML dans la bibliothèque source, phase 01 leçon 15).
- **La taille du mini-batch change le bruit effectif et le meilleur taux d'apprentissage** ; quand tu modifies le batch, réajuste $\eta$.

## Points clés à retenir

- L'entraînement, c'est passe avant, passe arrière, mise à jour ; calcule tous les gradients avant de mettre à jour.
- Débogue dans l'ordre : perte initiale, test de gradient, sur-apprentissage de 8 exemples, balayage du taux d'apprentissage, puis entraînement complet.
- Le momentum accumule les directions cohérentes : pas effectif d'environ `eta/(1-mu)`.
- Un réseau incapable de sur-apprendre 8 exemples a un bug, pas un problème de capacité.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Ta perte est bloquée à `ln K`. Énumère trois causes possibles et le test qui les distingue.
2. Que t'attends-tu à voir dans la courbe de perte pour un `eta` trop petit, correct et trop grand ?

## Pour aller plus loin

- Bibliothèque source : `ai-engineering-from-scratch` phase 03 leçons 08, 10 et 13.
- *Mathematics for Machine Learning*, Sec. 7.1.

## Critères de maîtrise

Tu sais partir d'un fichier vide et écrire une boucle d'entraînement fonctionnelle pour un MLP, énoncer l'échelle de débogage en cinq échelons, et prédire ce qui arrive à la perte pour $\eta$ trop petit, correct et trop grand.
