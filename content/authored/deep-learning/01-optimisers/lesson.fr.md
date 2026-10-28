---
id: dl.01
title: Optimiseurs et plannings de taux d'apprentissage
summary: Momentum, Nesterov, RMSProp, Adam et AdamW dérivés et implémentés en NumPy, plus les plannings warmup et cosinus, comparés sur une quadratique mal conditionnée et sur le réseau des demi-lunes.
minutes: 80
teaches: [dl.momentum, dl.adam, dl.lr-schedule]
requires: [opt.gradient-descent, nn.training]
sources:
  - "math/math for ML.pdf (Sec. 7.1 Optimization Using Gradient Descent, PDF p.233-239: Example 7.1 quadratic, poor conditioning and zig-zag, momentum eq. 7.11-7.12, SGD and mini-batches). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/06-optimizers/docs/en.md (SGD, momentum, RMSProp, Adam, bias correction, AdamW, default hyperparameters). READ."
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/09-learning-rate-schedules/docs/en.md (constant, step, cosine, warmup+cosine, 1cycle). READ."
  - "LLMs-from-scratch/appendix-D/01_main-chapter-code/appendix-D.ipynb (D.1 warmup, D.2 cosine decay, D.3 gradient clipping). READ."
---

## Où cela s'insère

Dans [[f2n.11]], tu as entraîné un réseau avec la descente de gradient simple plus le momentum et tu as vu que cela fonctionne, mais aussi que c'est capricieux : le taux d'apprentissage parfait pour un problème diverge sur le suivant. Tous les grands modèles que tu rencontreras (un modèle de langage, un transformeur de vision, un modèle de diffusion) sont entraînés avec une règle de mise à jour plus élaborée, presque toujours Adam ou AdamW, associée à un planning de taux d'apprentissage qui démarre près de zéro, monte, puis décroît. Cette leçon explique chaque ingrédient, dérive la seule formule non évidente (la correction de biais d'Adam), et mesure ce que chaque ingrédient t'apporte sur deux petits problèmes que tu peux exécuter dans ton navigateur.

La rétropropagation ([[f2n.10]]) fournit le gradient. L'optimiseur décide quoi en faire. Même gradient, règle de mise à jour différente, entraînement très différent.

## L'idée en mots simples

Imagine une bille dans une longue vallée étroite aux parois raides et au fond en pente douce. La descente de gradient simple ne regarde que la pente sous ses pieds. Cette pente pointe surtout vers la paroi la plus proche, donc la bille rebondit d'une paroi à l'autre tout en avançant lentement le long du fond. Réduire la taille du pas arrête les rebonds, mais rend la progression le long du fond désespérément lente. C'est l'image que dessinent Deisenroth et al. pour les problèmes mal conditionnés : la vallée est très courbée dans une direction et presque plate dans l'autre.

Trois remèdes, chacun une amélioration de la bille :

- **Momentum** : donne du poids à la bille. Elle se souvient de la direction où elle allait. Le rebond en travers de la vallée change de signe à chaque pas et s'annule dans la mémoire ; la poussée le long du fond pointe toujours dans le même sens et s'accumule.
- **Nesterov** : avant de décider où pousser, regarde où la mémoire est sur le point de t'emporter, et lis la pente à cet endroit. Tu freines plus tôt quand tu vas dépasser la cible.
- **Tailles de pas adaptatives (RMSProp, Adam)** : donne à chaque paramètre sa propre taille de pas. Un paramètre dont les gradients ont été énormes récemment reçoit un petit pas ; un dont les gradients ont été minuscules reçoit un grand pas. Adam combine cela avec le momentum.

Un **planning** modifie ensuite la taille globale du pas pendant l'entraînement : doux au début (les statistiques courantes ne sont pas encore fiables), grand au milieu, petit à la fin pour que tu te stabilises dans la vallée au lieu de rebondir sur son fond.

## Définitions formelles

Soit $\theta$ les paramètres, $g_t=\nabla L(\theta_{t})$ le gradient (de mini-batch) au pas $t$, et $\eta$ le taux d'apprentissage. Tous les produits et puissances de vecteurs ci-dessous sont terme à terme.

**SGD.** $\theta\leftarrow\theta-\eta g$. Les mini-batches conviennent car une estimation non biaisée du gradient est tout ce dont la convergence a besoin (Deisenroth Sec. 7.1.3).

**Momentum** (heavy ball, balle lourde). Garde une vitesse et avance le long de celle-ci :

$$
v_t=\mu\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t .
$$

C'est la même idée que les éq. 7.11-7.12 du livre (un pas plus $\alpha$ fois le pas précédent) ; la forme ci-dessus, utilisée par la plupart des bibliothèques, garde $\eta$ en dehors de la vitesse.

**Momentum de Nesterov.** Même vitesse, mais on avance selon la direction d'anticipation :

$$
v_t=\mu\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,(g_t+\mu\,v_t).
$$

**RMSProp.** Suit une moyenne mobile des gradients au carré et divise par sa racine :

$$
s_t=\beta\,s_{t-1}+(1-\beta)\,g_t^2,\qquad \theta_{t+1}=\theta_t-\eta\,\frac{g_t}{\sqrt{s_t}+\varepsilon}.
$$

**Adam.** Deux moyennes mobiles (du gradient et de son carré), chacune corrigée du biais :

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}},\qquad
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

Valeurs par défaut : $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$.

**AdamW** ajoute la décroissance des poids (weight decay) en dehors de l'échelonnage adaptatif :

$$
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}-\eta\lambda\,\theta_t .
$$

## Dérivation

### Pourquoi le momentum accélère les directions cohérentes

Supposons que le gradient soit une constante $g$ le long d'une direction. Alors $v_t=\mu v_{t-1}+g$ a pour point fixe $v=g/(1-\mu)$. Le pas effectif est donc $\eta/(1-\mu)$ : dix fois plus grand pour $\mu=0.9$. Supposons maintenant que le gradient alterne $+g,-g,+g,\dots$ (le rebond en travers de la vallée). En écrivant $v_t=(-1)^t a$ dans la récurrence, on obtient $a=-\mu a+g$, donc $a=g/(1+\mu)$. Le rapport entre le cas cohérent et le cas alterné est $(1+\mu)/(1-\mu)=19$ pour $\mu=0.9$. Le momentum amplifie exactement les directions qui concordent et supprime celles qui s'inversent.

### De combien cela aide sur une quadratique

Pour $f(\theta)=\tfrac12\theta^\top A\theta-b^\top\theta$ avec des valeurs propres entre $\lambda_{\min}$ et $\lambda_{\max}$ et un nombre de conditionnement $\kappa=\lambda_{\max}/\lambda_{\min}$, la descente de gradient simple avec le meilleur pas constant $\eta=2/(\lambda_{\max}+\lambda_{\min})$ réduit l'erreur d'un facteur $(\kappa-1)/(\kappa+1)$ par pas. La stabilité exige $\eta<2/\lambda_{\max}$ ; au-delà, l'itération diverge le long de la direction la plus raide. Le momentum heavy-ball avec $\eta=4/(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2$ et $\mu=\big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$ améliore le facteur par pas à $(\sqrt\kappa-1)/(\sqrt\kappa+1)$. Pour $\kappa=100$, cela donne $0.980$ contre $0.818$ par pas : le nombre d'itérations passe de l'ordre de $\kappa$ à l'ordre de $\sqrt\kappa$. Nous énonçons ces taux classiques et les vérifions numériquement plus bas plutôt que de les démontrer ici.

### La correction de biais d'Adam

Démarre les deux moments à zéro. Déroule la récurrence du second moment :

$$
v_t=(1-\beta_2)\sum_{i=1}^{t}\beta_2^{\,t-i}\,g_i^2 .
$$

Suppose un instant que les gradients sont stationnaires, $\mathbb{E}[g_i^2]=\sigma^2$ pour tout $i$. Alors

$$
\mathbb{E}[v_t]=(1-\beta_2)\,\sigma^2\sum_{i=1}^{t}\beta_2^{\,t-i}=(1-\beta_2)\,\sigma^2\,\frac{1-\beta_2^{\,t}}{1-\beta_2}=\sigma^2\,(1-\beta_2^{\,t}).
$$

L'estimation est trop petite d'exactement le facteur $1-\beta_2^{\,t}$ parce que la valeur initiale nulle est intégrée dans la moyenne. Diviser par ce facteur donne une estimation non biaisée, $\mathbb{E}[\hat v_t]=\sigma^2$. Le même raisonnement donne $\hat m_t=m_t/(1-\beta_1^{\,t})$. À $t=1$, les facteurs de correction sont $1/(1-\beta_1)=10$ pour $m$ et $1/(1-\beta_2)=1000$ pour $v$. À $t=50$, le dénominateur pour $m$ est $1-0.9^{50}=0.995$, pratiquement disparu, mais pour $v$ il vaut encore $1-0.999^{50}\approx0.049$, donc la correction du second moment compte pendant des centaines de pas.

Une conséquence à retenir : à $t=1$, $\hat m_1=g_1$ et $\hat v_1=g_1^2$, donc la première mise à jour est $-\eta\,g_1/(|g_1|+\varepsilon)\approx-\eta\,\mathrm{sign}(g_1)$. Le premier pas d'Adam a une taille d'environ $\eta$ sur chaque coordonnée, quelle que soit l'échelle du gradient.

### Pourquoi AdamW, et non Adam plus L2

Ajouter $\tfrac{\lambda}{2}\|\theta\|^2$ à la perte ajoute $\lambda\theta$ au gradient. En SGD simple, cela revient à rétrécir les poids de $\eta\lambda\theta$ à chaque pas. Dans Adam, en revanche, le gradient $g+\lambda\theta$ est divisé par $\sqrt{\hat v}$, donc les poids ayant un grand historique de gradients sont moins décroissants et ceux aux petits gradients le sont davantage : la force du régulariseur dépend maintenant des statistiques du gradient. AdamW applique $-\eta\lambda\theta$ directement, donc chaque poids rétrécit proportionnellement à sa taille. La leçon source indique qu'AdamW généralise mieux qu'Adam avec L2 et qu'il est le choix par défaut pour les transformeurs ; considère l'ampleur de cette amélioration comme dépendante du problème, nous ne la mesurons pas ici.

### Plannings

Warmup plus décroissance cosinus, avec un pic $\eta_{\max}$, un minimum $\eta_{\min}$, $W$ pas de warmup et $T$ pas au total :

$$
\eta(t)=\begin{cases}\eta_{\max}\,t/W & t<W\\[2pt] \eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\Big(1+\cos\big(\pi\,\tfrac{t-W}{T-W}\big)\Big) & t\ge W.\end{cases}
$$

Pourquoi un warmup ? Avec $v_0=0$, l'estimation précoce du second moment repose sur une poignée de gradients, donc elle peut être très éloignée de la vérité, et un grand $\eta$ transforme cette erreur en un pas grand et mal échelonné. Commencer petit laisse les statistiques se stabiliser (la correction de biais supprime l'erreur moyenne, pas le bruit). La décroissance par paliers (multiplier par 0.1 à des epochs fixes) et le 1cycle (monter pendant la moitié de l'entraînement, puis descendre) sont les autres formes courantes. Le notebook de l'annexe D fait un warmup linéaire d'un petit taux initial jusqu'à un pic, puis applique une décroissance cosinus, et ajoute l'écrêtage de gradient (gradient clipping) : redimensionner le gradient pour que sa norme L2 soit au plus un seuil (une matrice de gradient avec $\|G\|_2=5$ et une norme max de 1 est multipliée par $1/5$).

## Exemple résolu

### Une quadratique mal conditionnée

Prenons $A=R\,\mathrm{diag}(1,100)\,R^\top$ avec $R$ une rotation de $30^\circ$, donc $\kappa=100$ et la vallée n'est pas alignée avec les axes. Minimisons $f(x)=\tfrac12x^\top Ax-b^\top x$ avec $b=(1,1)$, à partir de $x_0=(-3,-1)$, jusqu'à $f(x)-f(x^*)<10^{-6}$ (écart initial $29.83$). Pour chaque optimiseur, nous avons balayé les taux d'apprentissage $\{0.003,0.005,0.01,0.015,0.02,0.03,0.05,0.1,0.2,0.3,0.5,1\}$ et gardé le meilleur (momentum $\mu=0.9$ pour les deux variantes de momentum ; Adam et RMSProp avec les bêtas par défaut) :

| Optimiseur | Meilleur taux d'apprentissage | Pas pour un écart inférieur à $10^{-6}$ |
|---|---|---|
| SGD | 0.015 | 534 |
| Momentum | 0.03 | 138 |
| Nesterov | 0.005 | 50 |
| Adam | 0.5 | 151 |
| RMSProp | aucun dans la grille | jamais atteint |

Lis le tableau avec attention, pas seulement le classement.

- La SGD à l'optimum théorique $\eta=2/101$ nécessite 431 pas, ce qui correspond au taux $0.980^{2t}$ ; à $\eta=0.02=2/\lambda_{\max}$, elle diverge.
- Le heavy ball avec les valeurs du manuel ($\eta=4/121$, $\mu=(9/11)^2$) nécessite 66 pas. Le $\mu=0.9$ générique n'est pas optimal pour ce problème, d'où les 138.
- Le chiffre de Nesterov provient d'une case favorable de la grille (à $\eta=0.01$ il faut 97 pas et à $0.015$ il diverge) ; ne l'interprète pas trop. La conclusion honnête est que les deux variantes de momentum battent la SGD d'un facteur 4 à 10 ici.
- Adam a besoin d'un grand taux d'apprentissage (0.5) sur ce problème lisse et déterministe. Son pas vaut environ $\eta$ par coordonnée quelle que soit la taille du gradient, donc près de l'optimum, où les gradients s'annulent, le pas normalisé reste grand et il a tendance à sautiller ; cela ne fonctionne que si $\eta$ est assez grand pour converger avant que cela ne compte. La force d'Adam n'est pas les quadratiques lisses ; ce sont les pertes bruitées, mal échelonnées et de grande dimension.
- RMSProp avec un $\eta$ constant n'atteint jamais la tolérance dans cette grille pour la même raison (il n'a pas de moyennage par momentum et sautille à l'échelle $\eta$). C'est pourquoi les plannings qui font décroître $\eta$ comptent.

L'affirmation sur le premier pas est vérifiable : Adam avec $\eta=0.001$ et un gradient $(5,-0.02)$ déplace les paramètres de $(0,0)$ à $(-0.001,+0.001)$. Les deux coordonnées bougent d'exactement $\eta$ bien que les gradients diffèrent d'un facteur 250.

### Le réseau des demi-lunes

Même réseau que [[f2n.11]] ($2\to16\to2$, ReLU, initialisation de He, graine 0, 200 pas sur batch complet de 200 points). Perte d'entraînement finale par taux d'apprentissage :

| Optimiseur | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| SGD | 0.456 | 0.348 | 0.282 | 0.247 | 0.191 | 0.042 | 0.009 |
| Momentum | 0.282 | 0.245 | 0.183 | 0.041 | 0.014 | 0.008 | 0.0005 |
| Nesterov | 0.282 | 0.245 | 0.186 | 0.042 | 0.014 | 0.002 | 0.266 |
| RMSProp | 0.210 | 0.043 | 0.010 | 0.0011 | 0.0010 | 0.352 | 1.157 |
| Adam | 0.246 | 0.107 | 0.011 | 0.0014 | 0.0007 | 0.009 | 0.308 |

Le motif est la leçon. La SGD n'atteint une petite perte qu'avec des taux d'apprentissage de 1 à 3. Adam et RMSProp atteignent une perte minuscule sur une plage de taux d'apprentissage d'un facteur 10 (0.03 à 0.3 pour Adam), donc ils sont bien moins sensibles à la valeur exacte, au prix d'une explosion à l'extrémité haute (RMSProp à 1, Adam à 3). Ce sont des pertes d'entraînement sur des données séparables après 200 pas ; elles disent à quelle vitesse chaque règle optimise, pas à quel point elle généralise.

Vérification du planning : Adam avec $\eta=0.3$ constant termine à une perte de $0.00075$ ; avec 20 pas de warmup et une décroissance cosinus jusqu'à $0.003$, il termine à $0.0029$ (les deux atteignent une précision de 1.0). Sur ce minuscule problème, le warmup n'apporte rien : la perte précoce est légèrement pire ($0.317$ contre $0.230$ au pas 10) parce que le warmup gaspille des pas. Le warmup se justifie dans les grands transformeurs entraînés avec de gros batches et des taux de pic élevés ; ne t'attends pas à le voir sur un jouet.

## Implémentation

```python
import numpy as np

class Momentum:
    def __init__(self, lr, mu=0.9, nesterov=False):
        self.lr, self.mu, self.nes, self.v = lr, mu, nesterov, None
    def step(self, p, g):
        if self.v is None: self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g
        d = g + self.mu * self.v if self.nes else self.v
        return p - self.lr * d

class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = self.v = None; self.t = 0
    def step(self, p, g):
        if self.m is None: self.m = np.zeros_like(p); self.v = np.zeros_like(p)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps) - self.lr * self.wd * p
```

L'exercice te demande de les écrire ainsi qu'un planning warmup-cosinus.

## Formes

Chaque tenseur d'état a exactement la forme de son paramètre : pour un poids $W\in\mathbb{R}^{n_{in}\times n_{out}}$, les tableaux $v$, $m$ et $s$ sont aussi de taille $n_{in}\times n_{out}$. Adam stocke donc deux copies supplémentaires du modèle en plus du gradient : l'état de l'optimiseur représente environ trois fois le nombre de paramètres en float32, une vraie contrainte pour les grands modèles.

## Idées reçues

- **« Adam rend le taux d'apprentissage sans importance. »** Il rend l'échelle du gradient sans importance. Le pas reste d'environ $\eta$ par coordonnée, et le tableau ci-dessus montre Adam divergeant à $\eta=3$.
- **« La correction de biais est un détail. »** Pour $\beta_2=0.999$, l'estimation du second moment est fortement biaisée pendant des centaines de pas sans elle.
- **« Weight decay et L2, c'est pareil. »** Seulement pour la SGD. Avec Adam, ils diffèrent ; utilise AdamW.
- **« Un momentum de 0.9 est toujours le meilleur. »** Le $\mu$ optimal dépend de la courbure ; pour la quadratique ci-dessus, la valeur du manuel est $0.67$.
- **« Le warmup ne nuit jamais. »** Il coûte de la progression initiale ; c'est un outil de stabilité, pas de vitesse.

## Points clés à retenir

- Le momentum accumule les composantes de gradient qui concordent (pas effectif $\eta/(1-\mu)$) et annule celles qui s'inversent ; sur les quadratiques, il fait passer le nombre d'itérations de l'ordre de $\kappa$ à l'ordre de $\sqrt\kappa$.
- Adam = momentum + échelonnage par paramètre par $1/\sqrt{\hat v}$ ; son premier pas vaut environ $\eta\,\mathrm{sign}(g)$.
- La correction de biais divise par $1-\beta^t$ car une moyenne mobile initialisée à zéro sous-estime exactement de ce facteur.
- AdamW fait décroître les poids en dehors de l'échelonnage adaptatif ; Adam avec L2 ne le fait pas.
- Un planning (warmup, puis décroissance cosinus) contrôle la taille globale du pas ; le faire décroître laisse l'itéré se stabiliser.

## Explique-le toi-même

1. Pourquoi un optimiseur à pas constant d'environ $\eta$ par coordonnée (RMSProp, Adam) ne parvient-il pas à se stabiliser sur une quadratique lisse, et qu'est-ce qui y remédie ?
2. Dérive, avec tes propres mots, pourquoi $\hat v_t=v_t/(1-\beta_2^t)$ est non biaisé sous des gradients stationnaires.

## Pour aller plus loin

- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*, Sec. 7.1.
- Bibliothèque source : phase 03 leçons 06 (optimiseurs) et 09 (plannings) ; *Build a Large Language Model From Scratch*, annexe D.

## Critères de maîtrise

Tu sais écrire de mémoire Momentum, Nesterov, Adam et AdamW, dériver le facteur de correction de biais, prédire la taille du premier pas d'Adam, et expliquer à partir des chiffres ci-dessus pourquoi la SGD est bien plus sensible au taux d'apprentissage qu'Adam sur le réseau des demi-lunes.
