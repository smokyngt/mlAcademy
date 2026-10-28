---
id: dl.03
title: Normalisation et initialisation
summary: Une initialisation qui préserve la variance dérivée à partir de zéro, la disparition et l'explosion des gradients mesurées, et la normalisation par batch, par couche et RMSNorm dérivées, implémentées et vérifiées par test de gradient.
minutes: 85
teaches: [dl.init, dl.batchnorm, dl.layernorm]
requires: [nn.backprop, nn.training, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/docs/en.md (symmetry problem, variance propagation, Xavier and He, transformer residual scaling). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/code/main.py (function list: zero/random/Xavier/Kaiming init, 50-layer forward experiment). READ (outline)."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/docs/en.md (batch norm, layer norm, RMSNorm sections: formulas, train/inference split, comparison). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/07-regularization/code/main.py (BatchNorm class with running statistics). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/docs/en.md (dead ReLUs, vanishing and exploding gradients, activation statistics table, gradient flow per layer, gradient checking, overfit-one-batch). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/13-debugging-neural-networks/code/debug_neural_nets.py (NetworkDebugger hooks, gradient_check; function list). READ (outline)."
---

## Où cela s'insère

Prends la boucle d'entraînement de [[f2n.11]] et donne 50 couches de profondeur au réseau. Sans précaution, rien ne s'entraîne : le signal qui circule vers l'avant soit rétrécit jusqu'à zéro, soit explose jusqu'à $10^{50}$, et le gradient qui circule vers l'arrière fait de même. Pendant deux décennies, cela a limité les réseaux à quelques couches. Deux familles de remèdes ont fait de la profondeur non plus un handicap mais la norme :

- **L'initialisation** choisit les poids de départ pour que les signaux ne rétrécissent ni ne croissent d'une couche à l'autre (Xavier, He).
- **Les couches de normalisation** restandardisent les activations à l'intérieur du réseau, en continu, pendant l'entraînement (batch norm pour la vision, layer norm et RMSNorm pour les transformeurs).

Cette leçon dérive les deux, mesure les défaillances qu'elles guérissent, et te fait implémenter la passe avant et arrière de la batch norm et les prouver correctes par un test de gradient.

## L'idée en mots simples

Imagine de faire passer un message le long d'une file de 50 personnes où chacune le répète à son propre volume. Si chacune est un peu plus silencieuse que la précédente, le message est inaudible à la fin ; un peu plus fort, et les dernières personnes hurlent. Ce n'est que si chaque personne préserve le volume que le message arrive intact. L'initialisation règle le bouton de volume de chaque personne au départ, selon le nombre de personnes qu'elle écoute.

La normalisation est différente : c'est un ingénieur du son qui, après quelques intervenants, mesure l'intensité et la ramène à un niveau standard. L'ingénieur peut remesurer en continu, donc la file reste audible même pendant que les intervenants continuent de changer leurs boutons pendant l'entraînement. Il existe trois sortes d'ingénieurs :

- **La batch norm** mesure chaque canal sur tout le groupe de messages du batch.
- **La layer norm** mesure chaque message sur ses propres canaux (indépendamment des autres messages).
- **RMSNorm** est la layer norm sans se soucier de retirer la moyenne, en se contentant de redimensionner à une taille standard.

## Définitions formelles et dérivations

### Propagation de la variance

Considérons une couche $z=\sum_{j=1}^{n_{in}}w_jx_j$ avec des poids indépendants de moyenne nulle et de variance $\mathrm{Var}(w)$, indépendants des entrées. Alors $\mathbb{E}[z]=0$ et

$$
\mathrm{Var}(z)=n_{in}\,\mathrm{Var}(w)\,\mathbb{E}[x^2].
$$

Si les entrées sont de moyenne nulle, $\mathbb{E}[x^2]=\mathrm{Var}(x)$ et chaque couche multiplie la variance par $n_{in}\mathrm{Var}(w)$. Après $L$ couches, le facteur est $(n_{in}\mathrm{Var}(w))^L$ : toute valeur différente de 1 explose ou s'évanouit exponentiellement. Le but est donc $n_{in}\mathrm{Var}(w)=1$ dans le cas linéaire.

**ReLU.** Pour $x=\max(0,z_{prev})$ avec $z_{prev}$ symétrique autour de zéro, la moitié des unités est mise à zéro et $\mathbb{E}[x^2]=\tfrac12\mathrm{Var}(z_{prev})$. Par conséquent

$$
\mathrm{Var}(z_l)=\tfrac12\,n_{in}\mathrm{Var}(w)\,\mathrm{Var}(z_{l-1}),
$$

et la condition pour une échelle constante est

$$
\mathrm{Var}(w)=\frac{2}{n_{in}}\qquad\text{(He / Kaiming).}
$$

**Direction arrière.** Appliquer le même raisonnement au gradient à travers $W^\top$ remplace $n_{in}$ par $n_{out}$. On ne peut satisfaire à la fois $n_{in}\mathrm{Var}(w)=1$ et $n_{out}\mathrm{Var}(w)=1$ que si $n_{in}=n_{out}$ ; Xavier (Glorot) prend un compromis,

$$
\mathrm{Var}(w)=\frac{2}{n_{in}+n_{out}},
$$

dérivé pour des activations à peu près linéaires (tanh et sigmoïde près de zéro). Pour ReLU, le facteur 2 supplémentaire (He) est nécessaire, sinon le moment d'ordre deux est divisé par deux à chaque couche. Pour les transformeurs avec des branches résiduelles $x\leftarrow x+f(x)$, les variances des branches s'additionnent, donc la leçon source indique que GPT-2 met à l'échelle les poids de sortie de chaque couche résiduelle par $1/\sqrt{2N}$ pour $N$ couches ; nous ne dérivons pas cela ici.

**Symétrie.** Des poids tous nuls (ou tous égaux) font calculer à chaque unité d'une couche la même fonction et recevoir le même gradient, donc elles restent identiques pour toujours ; l'initialisation aléatoire est ce qui brise la symétrie, et son échelle est ce que fixent les formules ci-dessus.

### Mesuré : l'échelle des activations à travers 50 couches

Largeur 256, batch 512, entrées normales centrées réduites, couches carrées, sans biais, un seul tirage aléatoire. Le tableau montre l'écart-type des activations après les couches 1, 5, 10, 25, 50 :

| Init et activation | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| ReLU, $\mathcal N(0,1)$ | 9.4 | $1.6\times10^{5}$ | $3.1\times10^{10}$ | $2.2\times10^{26}$ | $5.9\times10^{52}$ |
| ReLU, $\mathcal N(0,0.01^2)$ | 0.093 | $1.6\times10^{-5}$ | $3.1\times10^{-10}$ | $2.1\times10^{-24}$ | $2.9\times10^{-48}$ |
| ReLU, Xavier | 0.59 | 0.16 | 0.026 | $1.4\times10^{-4}$ | $1.4\times10^{-8}$ |
| ReLU, He | 0.83 | 0.92 | 0.89 | 1.07 | 0.92 |
| tanh, $\mathcal N(0,0.01^2)$ | 0.16 | $1.0\times10^{-4}$ | $1.1\times10^{-8}$ | $1.3\times10^{-20}$ | $1.4\times10^{-40}$ |
| tanh, $\mathcal N(0,1)$ | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 |
| tanh, Xavier | 0.63 | 0.32 | 0.23 | 0.14 | 0.093 |

Observations conformes à la dérivation : He maintient les activations ReLU à l'ordre 1 ; Xavier sur ReLU perd environ la moitié du moment d'ordre deux par couche ; les deux échelles naïves explosent ou s'évanouissent exponentiellement. Deux détails à lire attentivement : tanh avec $\mathcal N(0,1)$ n'« explose » pas (tanh est bornée) mais chaque unité sature près de $\pm1$, où la dérivée est à peu près nulle, donc les gradients meurent ; et tanh avec Xavier décroît lentement au lieu de rester exactement constante, parce que la dérivation suppose le régime linéaire et que tanh n'est qu'approximativement linéaire.

### Mesuré : gradients évanescents

Un réseau sigmoïde de 20 couches (largeur 128, poids à l'échelle de Xavier, batch 256, une perte factice avec un gradient amont unitaire) a les normes de Frobenius suivantes du gradient des poids aux couches 1, 5, 10, 15, 20 (en comptant depuis l'entrée) :

- sans normalisation : $3\times10^{-12}$, $7\times10^{-9}$, $9\times10^{-6}$, $1\times10^{-2}$, $15$,
- avec batch norm avant chaque sigmoïde : $0.58$, $0.57$, $0.52$, $0.46$, $0.37$.

La dérivée de la sigmoïde vaut au plus $0.25$, donc chaque couche multiplie le signal arrière par un facteur typiquement bien inférieur à un ; vingt couches se composent en un rapport d'environ $10^{12}$ entre la dernière et la première couche. La batch norm maintient les pré-activations dans la plage sensible de la sigmoïde et le profil des gradients plat. (Un réseau ReLU avec initialisation de He n'a montré aucune telle décroissance dans la même mesure, mais note que les normes de gradient des poids incluent aussi l'échelle avant de l'entrée de la couche, donc compare les profils, pas les valeurs absolues.)

### Diagnostics à toujours journaliser

1. **Statistiques d'activation par couche** (moyenne, écart-type, fraction de sorties ReLU exactement nulles). Sain : borné, moyenne proche de 0 et écart-type qui ne s'effondre ni ne croît ; « morte » signifie tout à zéro ; « saturée » signifie collée aux extrêmes.
2. **Norme du gradient par couche.** Les couches dont la norme est $10^3$ fois plus petite que celle de la dernière couche s'évanouissent ; $\texttt{nan}$ ou une croissance sans borne signifie explosion. Remèdes : meilleure initialisation, normalisation, connexions résiduelles, écrêtage du gradient (voir [[dl.01]]), taux d'apprentissage plus bas.
3. **ReLU mortes.** Une unité avec une grande pré-activation négative a un gradient nul pour toujours. Si plus d'environ la moitié des activations sont exactement nulles, change l'initialisation, le taux d'apprentissage ou l'activation.
4. **Test de gradient** de toute couche écrite à la main ([[f2n.10]]), et le test de **sur-apprentissage d'un batch** de [[f2n.11]].

## Normalisation par batch

### Passe avant

Pour un mini-batch $x\in\mathbb{R}^{N\times D}$ (les lignes sont les exemples, les colonnes les caractéristiques), par caractéristique $j$ :

$$
\mu_j=\frac1N\sum_{i}x_{ij},\qquad \sigma_j^2=\frac1N\sum_i(x_{ij}-\mu_j)^2,
$$

$$
\hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}},\qquad y_{ij}=\gamma_j\hat x_{ij}+\beta_j .
$$

Les $\gamma,\beta\in\mathbb{R}^D$ appris permettent au réseau d'annuler la standardisation si c'est ce qu'il veut. Au moment de l'évaluation, les statistiques du batch sont remplacées par des moyennes mobiles recueillies pendant l'entraînement, $\mu_{run}\leftarrow(1-m)\mu_{run}+m\mu$, de même pour la variance, avec $m=0.1$ typique. Formes : entrée $(N,D)$, statistiques $(D)$, sortie $(N,D)$. Pour des caractéristiques convolutives $(N,C,H,W)$, les statistiques sont prises par canal sur $N,H,W$.

Comme l'entraînement utilise les statistiques du batch, la sortie de chaque exemple dépend des autres exemples du batch. Avec de minuscules batches, les statistiques sont bruitées (la leçon source met en garde contre les batches inférieurs à environ 32), et avec $N=1$ elles n'ont aucun sens. C'est la raison pour laquelle les transformeurs ne l'utilisent pas.

### Passe arrière

Soit $\bar y$ le gradient amont $(N,D)$, et écrivons $\mathrm{inv}=1/\sqrt{\sigma^2+\varepsilon}$ et $\bar{\hat x}=\bar y\odot\gamma$. Les gradients des paramètres sont immédiats :

$$
\bar\gamma_j=\sum_i\bar y_{ij}\hat x_{ij},\qquad \bar\beta_j=\sum_i\bar y_{ij}.
$$

Pour l'entrée, suis les trois chemins par lesquels $x_{ij}$ affecte $\hat x$ : directement, via $\mu$ et via $\sigma^2$ (on omet l'indice de caractéristique $j$) :

$$
\bar{\sigma^2}=\sum_i\bar{\hat x}_i\,(x_i-\mu)\cdot\big(-\tfrac12\big)(\sigma^2+\varepsilon)^{-3/2},
$$

$$
\bar\mu=-\,\mathrm{inv}\sum_i\bar{\hat x}_i\;+\;\bar{\sigma^2}\cdot\frac{-2}{N}\sum_i(x_i-\mu),
$$

$$
\bar x_i=\bar{\hat x}_i\,\mathrm{inv}+\bar{\sigma^2}\,\frac{2(x_i-\mu)}{N}+\frac{\bar\mu}{N}.
$$

Le dernier terme de $\bar\mu$ s'annule car $\sum_i(x_i-\mu)=0$. En substituant $\bar{\sigma^2}\cdot\frac{2(x_i-\mu)}{N}=-\frac{\mathrm{inv}}{N}\hat x_i\sum_k\bar{\hat x}_k\hat x_k$, on obtient la forme compacte utilisée dans le code :

$$
\boxed{\;\bar x_i=\frac{\mathrm{inv}}{N}\Big(N\,\bar{\hat x}_i-\sum_k\bar{\hat x}_k-\hat x_i\sum_k\bar{\hat x}_k\hat x_k\Big)\;}
$$

Lis-la ainsi : prends le gradient entrant, retire sa moyenne sur le batch, retire sa composante le long de $\hat x$, puis rééchelonne. Deux conséquences méritent vérification : le gradient a une somme nulle sur le batch (décaler toutes les entrées d'une constante ne change rien), et il est orthogonal à $\hat x$ (multiplier toutes les entrées par une constante ne change rien).

### Layer norm et RMSNorm

La layer norm standardise chaque exemple sur ses propres $D$ caractéristiques :

$$
\mu_i=\frac1D\sum_jx_{ij},\quad \sigma_i^2=\frac1D\sum_j(x_{ij}-\mu_i)^2,\quad y_{ij}=\gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}+\beta_j .
$$

Elle se comporte de façon identique à l'entraînement et à l'évaluation et pour toute taille de batch, y compris 1. La passe arrière est la même formule encadrée avec les sommes prises sur $j$ (l'axe des caractéristiques) et $N$ remplacé par $D$. Dans les transformeurs, elle est appliquée avant (pre-LN, rapportée comme plus stable à entraîner) ou après (post-LN) chaque bloc d'attention et de réseau feed-forward.

RMSNorm supprime le centrage et le décalage :

$$
y_{ij}=\gamma_j\,\frac{x_{ij}}{\sqrt{\tfrac1D\sum_kx_{ik}^2+\varepsilon}} .
$$

Avec $r=(\tfrac1D\sum x^2+\varepsilon)^{-1/2}$, on a $\partial r/\partial x_j=-x_jr^3/D$, donc la passe arrière est $\bar x=r\,\bar{\hat x}-x\,\frac{r^3}{D}\sum_k\bar{\hat x}_kx_k$ avec $\bar{\hat x}=\bar y\odot\gamma$. La leçon source indique que les modèles de type LLaMA utilisent RMSNorm et qu'elle économise environ 10 % du coût de la normalisation pour une précision comparable ; nous ne l'avons pas mesuré.

Pourquoi la normalisation aide-t-elle ? L'explication d'origine (réduction du « décalage de covariable interne ») est, selon la leçon source, contredite par des travaux ultérieurs qui pointent un paysage de perte plus lisse et des gradients plus prédictifs. Considère le mécanisme comme en partie ouvert ; le bénéfice empirique (profil de gradient stable ci-dessus, taux d'apprentissage utilisables plus grands) ne fait aucun doute.

## Exemple résolu

Batch d'une caractéristique, $x=(1,2,3,4)^\top$, $\gamma=1$, $\beta=0$, $\varepsilon=0$. Alors $\mu=2.5$, $\sigma^2=1.25$, $\mathrm{inv}=1/\sqrt{1.25}=0.8944$, et

$$
\hat x=(-1.342,\,-0.447,\,0.447,\,1.342)^\top .
$$

La sortie a une moyenne $0$ et une variance $1$. Soit maintenant le gradient amont $\bar y=(1,0,0,0)^\top$. Alors $\sum\bar{\hat x}=1$ et $\sum\bar{\hat x}\hat x=-1.342$, donc pour $i=1$ :

$$
\bar x_1=\frac{0.8944}{4}\big(4\cdot1-1-(-1.342)(-1.342)\big)=0.2236\cdot(4-1-1.8)=0.2683 ,
$$

et le gradient complet est $(0.268,-0.358,-0.089,0.179)^\top$, dont la somme est nulle comme prévu. $\bar\gamma=\sum\bar y\hat x=-1.342$ et $\bar\beta=1$. Ces valeurs sont reproduites par le code ci-dessous.

Test de gradient sur des données aléatoires ($N=8$, $D=5$, entrées de moyenne 2 et d'écart-type 3) : des différences centrées avec $h=10^{-6}$ comparées aux formules de la passe arrière donnent des erreurs relatives maximales de $1.7\times10^{-9}$ (entrée de la batch norm), $7.6\times10^{-10}$ ($\gamma$) et $1.7\times10^{-10}$ ($\beta$) ; la layer norm donne $1.8\times10^{-9}$, $2.3\times10^{-10}$, $1.0\times10^{-10}$ ; RMSNorm donne des erreurs absolues inférieures à $1.5\times10^{-9}$.

## Implémentation

```python
import numpy as np

def bn_forward(x, gamma, beta, eps=1e-5):           # x: (N, D)
    mu, var = x.mean(0), x.var(0)                    # (D,)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return gamma * xh + beta, (xh, inv, gamma)

def bn_backward(dy, cache):
    xh, inv, gamma = cache
    N = dy.shape[0]
    dxh = dy * gamma
    dx = (inv / N) * (N * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0))
    return dx, (dy * xh).sum(0), dy.sum(0)           # dx, dgamma, dbeta

def layer_norm(x, gamma, beta, eps=1e-5):            # normalise each row
    mu, var = x.mean(1, keepdims=True), x.var(1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
```

Un utilitaire de débogage dans l'esprit du `NetworkDebugger` de la leçon source enregistre, par couche, `x.std()`, `(x == 0).mean()` et la norme du gradient ; journalise-les toutes les quelques centaines de pas.

## Idées reçues

- **« La batch norm est un régulariseur, donc supprime le dropout. »** La batch norm peut régulariser légèrement grâce au bruit du batch, mais son rôle principal est l'optimisation ; teste la combinaison.
- **« Le mode éval ne compte que pour le dropout. »** La batch norm passe aussi aux statistiques mobiles ; oublier $\texttt{model.eval()}$ donne des prédictions fausses et dépendantes du batch.
- **« L'initialisation ne compte qu'au pas 0. »** Elle décide si les premiers pas ont des gradients utilisables ; un mauvais départ peut être irrécupérable (ReLU mortes, tanh saturée).
- **« Xavier convient pour ReLU. »** Mesuré ci-dessus : le moment d'ordre deux est divisé par deux à chaque couche.
- **« La layer norm, c'est la batch norm sur l'autre axe, pareil. »** Statistiques différentes, comportement différent avec une taille de batch de 1 et à l'inférence.
- **« Le test de gradient peut utiliser les mêmes données que l'entraînement. »** Utilise un minuscule batch et la double précision ; un test en float32 est trop bruité pour être fiable.

## Points clés à retenir

- $\mathrm{Var}(z)=n_{in}\mathrm{Var}(w)\mathbb{E}[x^2]$ est toute l'histoire de l'initialisation ; ReLU divise $\mathbb{E}[x^2]$ par deux, ce qui donne $\mathrm{Var}(w)=2/n_{in}$.
- Une mauvaise échelle signifie une explosion ou une disparition exponentielle avec la profondeur ; le tableau montre $10^{52}$ et $10^{-48}$ après 50 couches.
- La batch norm standardise par caractéristique sur le batch avec $\gamma,\beta$ appris, et sa passe arrière est $\frac{\mathrm{inv}}{N}(N\bar{\hat x}-\sum\bar{\hat x}-\hat x\sum\bar{\hat x}\hat x)$.
- La layer norm normalise par exemple, RMSNorm abandonne la moyenne ; toutes deux sont indépendantes du batch, c'est pourquoi les transformeurs les utilisent.
- Journalise les statistiques d'activation et les normes de gradient par couche ; elles révèlent l'évanescence, l'explosion et les unités mortes avant que la perte ne le fasse.

## Explique-le toi-même

1. En partant de $z=\sum w_jx_j$, dérive l'initialisation de He et dis d'où vient le facteur 2.
2. Pourquoi le gradient d'entrée de la batch norm a-t-il une somme nulle sur le batch, et quelle invariance cela reflète-t-il ?

## Pour aller plus loin

- Bibliothèque source : phase 03 leçons 07 (sections sur la normalisation), 08 et 13.
- Glorot et Bengio (Xavier), He et al. (Kaiming), Ioffe et Szegedy (batch norm) : cités par les leçons source ; non ouverts ici.

## Critères de maîtrise

Tu sais dériver l'initialisation qui préserve la variance pour tanh et ReLU, implémenter la passe avant et arrière de la batch norm et les confirmer par un test de gradient, dire quand utiliser la normalisation par batch, par couche et RMS, et lire un tableau de statistiques d'activation/gradient pour diagnostiquer des signaux évanescents ou explosifs.
