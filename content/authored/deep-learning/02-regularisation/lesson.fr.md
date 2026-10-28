---
id: dl.02
title: "Régularisation : décroissance des poids, dropout, arrêt précoce, augmentation de données"
summary: Pourquoi un réseau qui ajuste parfaitement l'ensemble d'entraînement généralise mal, et les cinq façons standard de réduire l'écart, avec la vision a priori gaussien de la décroissance des poids et la dérivation du dropout inversé.
minutes: 80
teaches: [dl.weight-decay, dl.dropout, dl.early-stopping]
requires: [nn.training, ml.ridge, ml.generalization, prob.rv]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 5.5 Regularization in Neural Networks, PDF p.276-280 (weight decay as negative log of a zero-mean Gaussian prior, eq. 5.112; consistent priors; early stopping, Sec. 5.5.2) and PDF p.285-286 (Sec. 5.5.5 training with transformed data equals a regulariser, eq. 5.131). READ; tangent propagation and soft weight sharing (p.281-284, 289-292) were not read."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/docs/en.md (dropout with inverted scaling, weight decay values, early stopping with patience, data augmentation, when to apply what). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/code/main.py (Dropout class with forward/backward and the 1/(1-p) scaling, l2_regularization signature). READ (first 35 lines)."
---

## Où cela s'insère

Un réseau avec assez de paramètres peut ajuster n'importe quel ensemble d'entraînement, y compris du pur bruit. Dans [[f2n.11]], tu as vu un petit réseau séparer les demi-lunes ; donne au même réseau un minuscule ensemble d'entraînement bruité et il atteindra joyeusement 100 % de précision d'entraînement tout en faisant nettement moins bien sur de nouvelles données. Cet écart entraînement/test est le sur-apprentissage, et c'est le problème pratique central de l'apprentissage profond : la perte que tu peux minimiser (entraînement) n'est pas celle qui t'intéresse (nouvelles données).

La régularisation, c'est tout ce qui réduit cet écart. Elle se décline en cinq familles, et un vrai modèle en empile généralement plusieurs :

1. pénaliser les grands poids (décroissance des poids, L2 ; L1 pour la parcimonie),
2. injecter du bruit dans le réseau (dropout),
3. s'arrêter avant que le modèle ait mémorisé (arrêt précoce),
4. agrandir les données par des transformations qui préservent l'étiquette (augmentation),
5. contraindre l'architecture (normalisation dans [[dl.03]], partage de poids dans les réseaux convolutifs, vus plus tard).

Cette leçon dérive les quatre premières et les mesure sur un problème volontairement petit.

## L'idée en mots simples

Pense à un étudiant qui prépare un examen avec seulement 40 questions d'entraînement. Une stratégie est de mémoriser les 40 réponses ; l'étudiant obtient alors 100 % à l'entraînement et un mauvais score à l'examen réel. Chaque régulariseur retire une façon de mémoriser :

- **La décroissance des poids** est une règle qui dit « les explications doivent être simples ». Un réseau qui mémorise a besoin de poids très grands et finement réglés pour tracer une frontière autour de chaque point individuel ; rétrécir les poids force des frontières plus lisses.
- **Le dropout** consiste à étudier avec des notes qui manquent au hasard. Si n'importe quelle note peut disparaître, aucune réponse ne peut dépendre d'un seul souvenir fragile, donc l'étudiant répartit ses connaissances sur de nombreux chemins redondants.
- **L'arrêt précoce** consiste à poser le livre quand le score à l'examen blanc (sur des questions non utilisées pour réviser) cesse de s'améliorer.
- **L'augmentation de données** consiste à inventer davantage de questions d'entraînement en reformulant les anciennes (décaler un point, retourner une image) pour que mémoriser les entrées exactes ne rapporte plus rien.

Tous échangent un peu de précision d'entraînement contre un meilleur comportement sur des données que le modèle n'a pas vues. Trop de régularisation, c'est du sous-apprentissage : l'étudiant à qui l'on n'autorise qu'une note par matière.

## Définitions formelles

Soit $L(\theta)$ la perte d'entraînement et $\theta$ les poids (les biais sont généralement exclus).

**L2 / décroissance des poids.** Minimiser $L(\theta)+\tfrac{\lambda}{2}\|\theta\|_2^2$. Le gradient gagne le terme $\lambda\theta$.

**L1.** Minimiser $L(\theta)+\lambda\|\theta\|_1$. Le gradient gagne $\lambda\,\mathrm{sign}(\theta)$.

**Dropout (inversé).** Pendant l'entraînement, pour une sortie de couche $a\in\mathbb{R}^n$, tire un masque indépendant $m_i\sim\mathrm{Bernoulli}(1-p)$ et renvoie

$$
\tilde a=\frac{m\odot a}{1-p}.
$$

Au moment de l'évaluation, renvoie $a$ inchangé.

**Arrêt précoce.** Suis la perte de validation à chaque epoch ; conserve les paramètres de l'epoch ayant la plus faible perte de validation ; arrête-toi après qu'elle ne s'est pas améliorée pendant une fenêtre de « patience ».

**Augmentation.** Remplace chaque paire d'entraînement $(x,y)$ par $(s(x,\xi),y)$ où $s$ est une transformation aléatoire qui préserve l'étiquette, de paramètre $\xi$.

## Dérivation

### La décroissance des poids est un a priori gaussien

Modélise les cibles par $y=f_\theta(x)+\epsilon$ avec un bruit gaussien de variance $\sigma^2$, et place un a priori gaussien centré de variance $\tau^2$ sur chaque poids, $p(\theta)=\mathcal N(0,\tau^2 I)$. L'estimation du maximum a posteriori (MAP) maximise $\log p(\mathcal D\mid\theta)+\log p(\theta)$, c'est-à-dire minimise

$$
\frac{1}{2\sigma^2}\sum_n\big(y_n-f_\theta(x_n)\big)^2+\frac{1}{2\tau^2}\|\theta\|_2^2+\text{const}.
$$

Multiplie par $\sigma^2$ : c'est l'erreur en somme des carrés plus $\tfrac{\lambda}{2}\|\theta\|^2$ avec

$$
\lambda=\frac{\sigma^2}{\tau^2}.
$$

Bishop énonce exactement cette interprétation (Sec. 5.5, éq. 5.112) : le régulariseur quadratique est l'opposé du logarithme d'un a priori gaussien centré. Lire $\lambda$ comme un rapport entre le bruit et la largeur de l'a priori est utile : un a priori serré (petit $\tau$) signifie une forte décroissance. Remplace la gaussienne par une densité de Laplace $\propto e^{-|\theta|/b}$ et le même calcul donne la pénalité L1 avec $\lambda\propto1/b$ ; son pic aigu en zéro explique pourquoi L1 produit des zéros exacts.

### Comment la décroissance agit dans la règle de mise à jour

Avec la descente de gradient sur $L+\tfrac\lambda2\|\theta\|^2$ :

$$
\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)=(1-\eta\lambda)\,\theta-\eta\nabla L .
$$

Chaque pas rétrécit d'abord les poids du facteur $1-\eta\lambda$, puis applique le pas de gradient ordinaire ; d'où le nom de décroissance des poids. (Avec Adam, cette équivalence se brise, voir [[dl.01]].)

Un exemple unidimensionnel rend l'effet concret. Minimise $\tfrac12(w-3)^2$, dont l'optimum sans contrainte est $w=3$. Avec une force L2 $\lambda=0.5$, la condition stationnaire $(w-3)+0.5\,w=0$ donne $w^\star=3/1.5=2$. Avec une force L1 de $0.5$, c'est $(w-3)+0.5\,\mathrm{sign}(w)=0$, ce qui donne $w^\star=2.5$ ; et pour tout $\lambda\ge3$, la solution L1 est exactement $0$ (seuillage doux : $w^\star=\mathrm{sign}(3)\max(3-\lambda,0)$). L2 rétrécit proportionnellement et n'atteint jamais zéro ; L1 soustrait une constante et peut tuer un poids. Les deux valeurs ont été confirmées en exécutant la descente de gradient.

### Dropout inversé : pourquoi diviser par $1-p$

Prends une unité d'activation $a$ et un masque $m\sim\mathrm{Bernoulli}(1-p)$, donc $\mathbb{E}[m]=1-p$.

Sans mise à l'échelle, la sortie $ma$ à l'entraînement a pour espérance $(1-p)a$. Si au moment du test on fournit $a$ en entier, la couche suivante voit des entrées $1/(1-p)$ fois plus grandes que tout ce qu'elle a vu à l'entraînement. L'ancienne correction multipliait les poids au test par $1-p$. La version inversée déplace la correction à l'entraînement : divise par $1-p$,

$$
\mathbb{E}[\tilde a]=\mathbb{E}\!\left[\frac{m\,a}{1-p}\right]=\frac{a}{1-p}\,\mathbb{E}[m]=a .
$$

L'activation attendue est inchangée, donc le réseau d'évaluation est simplement le réseau sans dropout, sans rien à rééchelonner. Le prix est une variance supplémentaire :

$$
\operatorname{Var}(\tilde a)=\frac{a^2}{(1-p)^2}\operatorname{Var}(m)=\frac{a^2}{(1-p)^2}\,p(1-p)=a^2\,\frac{p}{1-p}.
$$

Pour $p=0.5$, la variance vaut $a^2$ (un bruit aussi grand que le signal) ; pour $p=0.1$, elle vaut $a^2/9$. Ce bruit est l'effet régularisant.

**Passe arrière.** L'application avant est terme à terme, $\tilde a_i=c_i\,a_i$ avec le coefficient fixe (par batch) $c_i=m_i/(1-p)$. Donc

$$
\frac{\partial L}{\partial a_i}=c_i\,\frac{\partial L}{\partial\tilde a_i}=\frac{m_i}{1-p}\frac{\partial L}{\partial\tilde a_i}.
$$

Le même masque et la même échelle doivent être réutilisés dans la passe arrière : les unités éliminées reçoivent un gradient nul, les unités conservées reçoivent le gradient entrant multiplié par $1/(1-p)$. Retirer le masque dans la passe arrière est un bug classique.

Une interprétation, avec une réserve : un réseau avec $n$ unités éliminables contient $2^n$ sous-réseaux qui partagent leurs poids, et le dropout en entraîne un au hasard à chaque pas ; évaluer sans dropout en fait approximativement la moyenne. C'est une image heuristique (la leçon source la présente comme une interprétation par ensemble), pas une égalité exacte pour des réseaux non linéaires.

### Arrêt précoce

L'entraînement réduit l'erreur d'entraînement de façon monotone, mais l'erreur sur un ensemble de validation mis de côté baisse typiquement puis remonte quand le réseau commence à ajuster le bruit (Bishop, Sec. 5.5.2, Fig. 5.12). S'arrêter au minimum de validation donne un réseau de complexité effective plus faible : le nombre de paramètres effectivement libres commence petit et croît pendant l'entraînement. En pratique, tu sauvegardes le meilleur modèle et continues d'entraîner pendant une fenêtre de patience avant d'abandonner. L'ensemble de validation doit être distinct de l'ensemble de test final, sinon l'epoch d'arrêt lui-même fait fuiter de l'information du test.

### L'augmentation de données est un régulariseur

La Sec. 5.5.5 de Bishop montre pourquoi entraîner sur des copies perturbées aléatoirement n'est pas seulement « plus de données ». Prends une transformation $s(x,\xi)$ avec $s(x,0)=x$ et un petit paramètre $\xi$ de moyenne nulle et de variance $\lambda$. Développe $y(s(x,\xi))$ au second ordre en $\xi$ et moyenne l'erreur quadratique : le terme du premier ordre disparaît car $\mathbb{E}[\xi]=0$, ce qui laisse

$$
\widetilde E=E+\lambda\,\Omega ,
$$

(éq. 5.131 du livre), où $\Omega$ pénalise la sensibilité de la sortie du modèle à la transformation. L'augmentation est donc une pénalité de régularité explicite déguisée : le réseau est poussé à être (presque) invariant le long de la direction de la transformation. Ajouter du bruit gaussien aux entrées, comme dans l'expérience ci-dessous, en est l'exemple le plus simple.

## Exemple résolu

Mise en place : demi-lunes bruitées (bruit 0.35) avec seulement **40 points d'entraînement** ; un réseau ReLU $2\to128\to128\to2$, initialisation de He, Adam avec $\eta=0.01$, 1500 epochs sur batch complet ; 4000 nouveaux points de test. Tous les nombres sont des moyennes sur 5 graines (une graine change l'ensemble d'entraînement et l'initialisation). La précision est mesurée avec le dropout désactivé.

| Méthode | Préc. entraîn. | Préc. test | Écart | Perte test |
|---|---|---|---|---|
| aucune | 1.000 | 0.831 | 0.169 | 4.91 |
| L2, $\lambda=0.01$ | 0.960 | 0.855 | 0.105 | 0.469 |
| L2, $\lambda=0.1$ | 0.875 | 0.830 | 0.045 | 0.391 |
| L1, $\lambda=0.01$ | 0.920 | 0.847 | 0.074 | 0.502 |
| dropout 0.3 | 1.000 | 0.838 | 0.162 | 5.50 |
| dropout 0.5 | 1.000 | 0.838 | 0.162 | 5.22 |
| bruit d'entrée 0.2 (augmentation) | 0.975 | 0.850 | 0.125 | 0.464 |
| arrêt précoce (patience sur un ensemble de validation de 200 points) | 0.870 | 0.825 | 0.045 | 0.463 |
| dropout 0.3 + bruit 0.2 | 0.960 | 0.860 | 0.100 | 0.472 |

Ce que disent les chiffres, et ce qu'ils ne disent pas :

- Le réseau non régularisé est trop sûr de lui : perte d'entraînement $\approx0$ mais perte de test $4.9$. La *précision* de test de 83 % cache à quel point ses erreurs assurées sont fausses.
- L2 et l'augmentation par bruit réduisent la perte de test d'un facteur d'environ 10 et relèvent la précision de 2 à 3 points. Une pénalité plus forte ($\lambda=0.1$) ferme le plus l'écart (0.045), mais en abaissant la précision d'entraînement, pas en relevant la précision de test : elle sous-apprend légèrement.
- **Le dropout seul n'a presque rien fait ici.** Écart $0.162$ contre $0.169$, perte de test légèrement pire. Avec 40 points et 128 unités par couche, le réseau mémorise encore via les unités qui survivent à chaque masque, et 1500 pas sur batch complet suffisent à moyenner le bruit. On sait que le dropout aide surtout dans les réseaux larges entraînés sur de grandes données avec de nombreuses epochs ; ce n'est pas un remède universel, et sur ce jouet c'est l'outil le plus faible. Combiné à l'augmentation, il a donné la meilleure précision de test (0.860).
- L'arrêt précoce s'est arrêté après environ 8 epochs en moyenne et a donné le plus petit écart mais pas la meilleure précision.
- L'écart-type de la précision de test sur les 5 graines est compris entre 0.011 et 0.022 pour chaque ligne. Des gains d'environ 2 points sur la référence (0.831 à 0.855 ou 0.860) ne représentent donc que 1 à 2 écarts-types, et des lignes qui diffèrent d'un point (L2 0.01 contre L1 0.01) ne peuvent pas être classées. L'effet important et cohérent porte sur la perte de test, pas sur la précision.

La mécanique est vérifiable dans l'exercice : le masque de dropout a la bonne moyenne et la bonne variance, la passe arrière le réutilise, et une solution ridge est une estimation MAP dont le gradient s'annule.

## Implémentation

```python
import numpy as np

class Dropout:
    def __init__(self, p, seed=0):
        self.p, self.rng = p, np.random.default_rng(seed)
    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)   # 0 or 1/(1-p)
        return x * self.mask
    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask

def decay_step(W, g, lr, lam):
    return (1 - lr * lam) * W - lr * g          # weight decay + gradient step
```

## Formes

Pour une couche cachée avec un batch $B$ et une largeur $n$ : activations $(B,n)$, masque $(B,n)$ (indépendant par exemple et par unité, retiré à chaque passe avant), gradient $(B,n)$. Dans un réseau convolutif, le masque est souvent partagé sur les positions spatiales ; dans un transformeur, il est appliqué aux poids d'attention et aux sorties des branches résiduelles. La décroissance des poids agit sur les matrices de poids $(n_{in},n_{out})$ et est normalement omise pour les biais et les échelles de normalisation.

## Idées reçues

- **« Le dropout est aussi appliqué au test. »** Non : l'évaluation utilise le réseau complet ($\texttt{model.eval()}$ dans PyTorch). L'oublier rend les prédictions bruitées et non déterministes.
- **« Éliminer avec la probabilité $p$ conserve une fraction $p$. »** $p$ est la probabilité d'élimination ; la probabilité de conservation est $1-p$.
- **« Décroissance des poids et pénalité L2 sont toujours identiques. »** Elles coïncident pour la SGD, pas pour Adam ([[dl.01]]).
- **« Plus de régularisation est toujours plus sûr. »** Au-delà de l'optimum, le modèle sous-apprend ; dans le tableau, $\lambda=0.1$ a abaissé la précision d'entraînement à 0.875.
- **« L'arrêt précoce sur l'ensemble de test, ça va. »** Non : choisis l'epoch d'arrêt sur une partition de validation.
- **« L1 et L2 ne diffèrent que par la force. »** L1 donne des zéros exacts (parcimonie) ; L2 rétrécit proportionnellement.

## Points clés à retenir

- Le sur-apprentissage est l'écart entre les pertes d'entraînement et de test ; les régulariseurs le réduisent en retirant des façons de mémoriser.
- L2 est l'opposé du logarithme d'un a priori gaussien, avec $\lambda=\sigma^2/\tau^2$ ; sa mise à jour est $\theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla L$.
- Le dropout inversé multiplie par $m/(1-p)$ à l'entraînement pour que $\mathbb{E}[\tilde a]=a$ et que l'inférence n'ait rien à changer ; la passe arrière réutilise le même masque.
- L'arrêt précoce et l'augmentation régularisent via la procédure et les données ; l'augmentation équivaut à une pénalité de régularité pour de petites transformations.
- Mesure toujours sur des données mises de côté, et attends-toi à des effets de 1 à 3 points de précision sur de petits problèmes.

## Explique-le toi-même

1. Dérive pourquoi le dropout a besoin du facteur $1/(1-p)$ et ce qui irait mal au test sans lui.
2. Explique pourquoi L1 mène les poids exactement à zéro alors que L2 ne fait que les rétrécir, en utilisant l'exemple unidimensionnel.

## Pour aller plus loin

- Bishop, *Pattern Recognition and Machine Learning*, Sec. 5.5 (a priori, arrêt précoce, données transformées, et les sections sur la propagation tangente et le partage souple de poids que nous n'avons pas lues).
- Bibliothèque source : phase 03 leçon 07.

## Critères de maîtrise

Tu sais énoncer la lecture a priori gaussien de la décroissance des poids, dériver le dropout inversé y compris sa variance et sa passe arrière, implémenter les deux avec des tests, et expliquer à partir de mesures pourquoi un régulariseur peut laisser la précision de test inchangée tout en modifiant la perte de test.
