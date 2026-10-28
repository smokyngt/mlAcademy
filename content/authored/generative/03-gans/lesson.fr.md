---
id: gen.03
title: "Réseaux antagonistes génératifs (GAN)"
summary: "Le jeu minimax entre générateur et discriminateur, le discriminateur optimal et la divergence de Jensen-Shannon dérivés, pourquoi l'entraînement est instable, l'effondrement de modes et le remède de Wasserstein, avec une analyse de gradient sur un jouet 1-D."
minutes: 75
teaches: [gen.gan]
requires: [gen.taxonomy, ml.logreg, nn.training, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/03-gans-generator-discriminator/docs/en.md: generator/discriminator losses, non-saturating generator loss, failure modes, table of variants (DCGAN, WGAN, spectral norm, StyleGAN). READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai/04-conditional-gans-pix2pix/docs/en.md: conditional GAN, Pix2Pix (U-Net generator, PatchGAN, adversarial + L1 loss), CycleGAN cycle-consistency. READ."
---

## Où cela s'inscrit

Les VAE ([[gen.02]]) ont besoin d'une borne sur la vraisemblance et donnent des échantillons flous. Les réseaux antagonistes génératifs (GAN) font le pari inverse : oublier les densités, n'apprendre qu'un **échantillonneur**, et l'entraîner en rendant un second réseau incapable de distinguer sa sortie des données réelles. Ils ont produit les premiers visages photoréalistes (2017-2019) et restent une option rapide en une seule passe ; leurs discriminateurs apparaissent encore comme pertes perceptuelles et dans les modèles de diffusion distillés en une étape. Ils sont aussi la meilleure étude de cas sur l'importance de la dynamique d'entraînement : un objectif mathématiquement magnifique mais difficile à optimiser. Il te faut la régression logistique ([[cml.02]]) et la divergence KL ([[math.10]]).

## L'idée en mots simples

Un faussaire peint de faux billets. Un inspecteur examine les billets et dit vrai ou faux. Chaque fois que l'inspecteur détecte un faux, le faussaire apprend ce qui l'a trahi et s'améliore ; chaque fois que le faussaire s'améliore, l'inspecteur doit chercher des défauts plus subtils. Si cette course se passe bien, le faussaire finit par être si bon que l'inspecteur ne peut que deviner, et les billets du faussaire sont indiscernables des vrais. Le faussaire ne voit jamais directement un vrai billet ; tout ce qu'il apprend passe par les verdicts de l'inspecteur. C'est à la fois la force de l'idée (aucune densité nécessaire) et son danger : si l'inspecteur a un angle mort, le faussaire le trouvera et ne produira que le seul type de faux qui passe (**effondrement de modes**, *mode collapse*).

## Le jeu

Le générateur $G_\theta:\mathbb R^{k}\to\mathbb R^d$ transforme un bruit $z\sim p_z$ (en général $\mathcal N(0,I)$) en un échantillon ; sa distribution de sortie est $p_g$. Le discriminateur $D_\psi:\mathbb R^d\to(0,1)$ sort la probabilité que son entrée soit réelle. L'objectif **minimax** est

$$
\min_G\max_D V(D,G)=\mathbb E_{x\sim p_\text{data}}[\log D(x)]+\mathbb E_{x\sim p_g}[\log(1-D(x))].
$$

Pour $G$ fixé, maximiser sur $D$ revient à l'entropie croisée binaire d'un classifieur qui sépare les points réels (étiquette 1) des points générés (étiquette 0), c'est-à-dire une régression logistique avec un réseau de neurones.

## Dérivation : le discriminateur optimal et Jensen-Shannon

Fixons $G$. Écrivons $V$ comme une intégrale sur $x$ (en utilisant que tirer $x=G(z)$ donne $x\sim p_g$) :

$$
V(D,G)=\int\big[p_\text{data}(x)\log D(x)+p_g(x)\log(1-D(x))\big]\,dx .
$$

L'intégrande peut être maximisé séparément en chaque $x$. Pour des nombres $a=p_\text{data}(x)$, $b=p_g(x)$, la fonction $f(y)=a\log y+b\log(1-y)$ a $f'(y)=a/y-b/(1-y)=0$, c'est-à-dire $y=a/(a+b)$, et $f''<0$ donc c'est un maximum. D'où

$$
D^*(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)} .
$$

Réinjectons. En utilisant $1-D^*=\dfrac{p_g}{p_\text{data}+p_g}$ et le mélange $m=\tfrac12(p_\text{data}+p_g)$ :

$$
V(D^*,G)=\mathbb E_{p_\text{data}}\!\left[\log\frac{p_\text{data}}{2m}\right]+\mathbb E_{p_g}\!\left[\log\frac{p_g}{2m}\right]
=-\log4+\mathrm{KL}(p_\text{data}\|m)+\mathrm{KL}(p_g\|m).
$$

La **divergence de Jensen-Shannon** est $\mathrm{JS}(p\|q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$ (symétrique, comprise entre $0$ et $\log2$). Donc

$$
\boxed{V(D^*,G)=-\log4+2\,\mathrm{JS}(p_\text{data}\|p_g)} .
$$

Conséquences. La meilleure réponse du générateur à un discriminateur parfait est de minimiser la divergence JS, et le minimum global $-\log4$ est atteint exactement quand $p_g=p_\text{data}$, où $D^*\equiv\tfrac12$ (l'inspecteur ne peut que deviner). L'exercice vérifie cette identité numériquement sur un petit exemple discret : avec $p_\text{data}=(0.1,0.4,0.4,0.1)$ et $p_g=(0.4,0.1,0.1,0.4)$ on obtient $D^*=(0.2,0.8,0.8,0.2)$, $\mathrm{JS}=0.19274$ et $V(D^*,G)=-1.00080=-\log4+2\cdot0.19274$.

## La perte pratique du générateur

Au début de l'entraînement, le générateur est mauvais et le discriminateur confiant : $D(G(z))\approx0$. Avec la perte minimax du générateur $\log(1-D(G(z)))$, écrivons $D=\sigma(s)$ pour un logit $s$. Alors $\partial\log(1-\sigma(s))/\partial s=-\sigma(s)$, qui vaut $\approx0$ quand $D\approx0$. Le gradient **s'évanouit exactement quand le générateur en a le plus besoin**. La perte **non saturante** $-\log D(G(z))$ a pour gradient $-(1-\sigma(s))\to-1$ dans ce régime. Elle a le même point fixe mais des gradients forts dès le début, et c'est celle que tout le monde utilise. Chiffres de l'exercice : à $s=-6$, $D=0.0025$, le gradient saturant vaut $-0.0025$ et le non saturant $-0.9975$.

L'entraînement alterne : une (ou quelques) étape(s) du discriminateur, puis une étape du générateur. Le générateur ne voit jamais les données réelles : son gradient passe par $D$ pour arriver dans $G$.

## Pourquoi l'entraînement est instable

- **Une cible mouvante.** $G$ optimise contre un $D$ qui change, et inversement. La descente de gradient sur un jeu minimax peut tourner autour de l'équilibre au lieu de converger (dans le jeu bilinéaire le plus simple $\min_a\max_b ab$, des pas de gradient simultanés décrivent une spirale vers l'extérieur).
- **Supports disjoints.** Si $p_\text{data}$ et $p_g$ vivent sur des ensembles différents de faible dimension (les images réelles se trouvent sur une variété mince), un discriminateur peut les séparer parfaitement et $\mathrm{JS}=\log 2$ quelle que soit la distance qui les sépare. Le gradient ne porte aucune information sur *comment s'améliorer*. L'exercice le montre : pour deux masses ponctuelles à distance $\theta=1,2,5$, la JS vaut $0.6931$ pour les trois.
- **Effondrement de modes.** Le générateur envoie de nombreux $z$ vers l'unique sortie que le discriminateur préfère actuellement ; le discriminateur pénalise alors ce mode, le générateur saute vers un autre mode, et ainsi de suite. La diversité n'est jamais retrouvée car rien dans la perte ne récompense la couverture de tous les modes.

## Wasserstein GAN

La distance de Wasserstein-1 mesure le coût du déplacement de la masse d'une distribution sur l'autre. Pour deux masses ponctuelles en $0$ et $\theta$, elle vaut $|\theta|$ : elle *croît progressivement avec la distance*, contrairement à la JS, de sorte que le générateur reçoit un gradient utile même avec des supports disjoints. En 1-D, c'est simplement la différence absolue moyenne des échantillons triés (l'exercice le fait). Le WGAN remplace le discriminateur à sortie probabiliste par un **critique** à valeurs réelles $f$, contraint à être 1-lipschitzien, et maximise $\mathbb E_{p_\text{data}}f(x)-\mathbb E_{p_g}f(x)$ (dualité de Kantorovich-Rubinstein). La contrainte de Lipschitz est imposée par écrêtage des poids (grossier), par pénalité de gradient (WGAN-GP) ou par normalisation spectrale. La leçon source range ces techniques parmi les stabilisateurs qui ont rendu les GAN praticables, avec l'architecture convolutive de DCGAN et la croissance progressive.

## GAN conditionnels et Pix2Pix

On fournit une condition $c$ (une classe, un croquis) aux deux réseaux : $G(c,z)$ et $D(c,x)$. Le discriminateur juge alors si $x$ est *cohérent avec* $c$. Pix2Pix (traduction image-à-image appariée) utilise un générateur U-Net, un discriminateur **PatchGAN** qui note des patchs locaux se chevauchant (le réalisme est local, ce qui demande moins de paramètres), et la perte $-\log D(x,G(x))+\lambda\|y-G(x)\|_1$. Le terme L1 stabilise l'entraînement et préfère des bords nets au flou que donne la L2. CycleGAN s'entraîne sans paires en ajoutant la cohérence cyclique : traduire dans un sens puis dans l'autre doit reproduire l'entrée.

## Exemple résolu : un pas de gradient d'un GAN jouet 1-D

Données réelles $x\sim\mathcal N(3,1)$. Générateur $G_\theta(z)=\theta+z$ avec $z\sim\mathcal N(0,1)$ (il ne peut que décaler). Discriminateur $D(x)=\sigma(ax+b)$. Perte du discriminateur $L_D=-\overline{\log\sigma(ax_r+b)}-\overline{\log(1-\sigma(ax_f+b))}$ avec les gradients

$$
\frac{\partial L_D}{\partial a}=\overline{-(1-D_r)x_r}+\overline{D_fx_f},\qquad
\frac{\partial L_D}{\partial b}=\overline{-(1-D_r)}+\overline{D_f}.
$$

Perte non saturante du générateur $L_G=-\overline{\log\sigma(a(\theta+z)+b)}$, donc $\partial L_G/\partial\theta=\overline{-(1-D_f)\,a}$. Avec 64 échantillons initialisés, $\theta=0$, $a=0.5$, $b=-0.5$ : $\partial L_D/\partial a=-0.62107$, $\partial L_D/\partial b=0.11934$, $\partial L_G/\partial\theta=-0.30459$, tous en accord avec les différences finies. Le signe du dernier est le point essentiel : puisque $a>0$, le discriminateur pense que les $x$ plus grands sont plus réels, donc descendre $L_G$ augmente $\theta$, déplaçant les faux échantillons vers la moyenne réelle 3. Exécuter 300 pas alternés (taux d'apprentissage 0.1, batch 128) depuis $a=b=\theta=0$ amène $\theta$ à environ $3.2$ dans l'exécution avec la graine de l'exercice (il oscille ensuite autour de la moyenne réelle au lieu de se stabiliser, ce qui est l'instabilité en miniature).

## Implémentation

```python
import numpy as np
sigmoid = lambda s: 1 / (1 + np.exp(-s))

def d_loss(a, b, xr, xf):
    return -np.mean(np.log(sigmoid(a*xr + b))) - np.mean(np.log(1 - sigmoid(a*xf + b)))

def g_grad(theta, a, b, z):                    # non-saturating loss
    return np.mean(-(1 - sigmoid(a*(theta + z) + b)) * a)
```

En PyTorch, tu écris les deux mêmes pertes avec `BCEWithLogitsLoss`, tu appelles `d_opt.step()` et `g_opt.step()` en alternance, et tu fais `detach()` du batch de faux lors de la mise à jour de $D$ pour qu'aucun gradient n'atteigne $G$.

## Formes (shapes)

`z: (B, k)` bruit, `G(z): (B, d)`, `D(x): (B,)` ou `(B,1)`. Pour PatchGAN, `D(x)` est `(B, 1, H', W')`, un score par patch, moyenné. Les deux optimiseurs détiennent des ensembles de paramètres disjoints.

## Idées reçues

- **« D doit être entraîné jusqu'à l'optimalité à chaque pas. »** Un $D$ parfait sature et donne des gradients évanescents ; en pratique, les deux sont gardés à peu près équilibrés.
- **« La courbe de perte montre la progression. »** Les pertes de GAN oscillent et ne suivent pas la qualité des échantillons ; inspecte les échantillons et utilise des métriques (FID, [[gen.06]]).
- **« WGAN supprime toute instabilité. »** Il améliore les gradients mais la contrainte de Lipschitz doit être bien imposée.
- **« L'effondrement de modes signifie une perte faible. »** Le générateur peut atteindre une perte faible sur un mode en ignorant tous les autres ; la perte ne mesure pas la couverture.

## Points clés à retenir

- $D^*=p_\text{data}/(p_\text{data}+p_g)$ et $V(D^*,G)=-\log4+2\,\mathrm{JS}(p_\text{data}\|p_g)$.
- Utilise la perte non saturante du générateur ; l'originale a des gradients évanescents.
- La JS est plate pour des supports disjoints ; la distance de Wasserstein ne l'est pas, ce qui motive WGAN.
- L'effondrement de modes et l'oscillation sont des propriétés du jeu, pas des bogues d'une implémentation.

## Explique-le toi-même

1. Dérive le discriminateur optimal et montre que la fonction de valeur égale $-\log4+2\,\mathrm{JS}$.
2. Pourquoi un discriminateur parfait ne donne-t-il aucun gradient utile quand les supports sont disjoints, et en quoi la distance de Wasserstein diffère-t-elle ?

## Pour aller plus loin

- Leçons sources : phase 08, leçons 03 et 04 (GAN ; GAN conditionnels et Pix2Pix). L'article original sur les GAN de Goodfellow et al. (non lu ici) contient la même dérivation.
- Suite : [[gen.04]], un modèle génératif avec une unique perte de régression stable.

## Critères de maîtrise

Tu sais dériver $D^*$ et le lien avec la JS, écrire la perte non saturante et expliquer pourquoi elle est nécessaire, décrire l'effondrement de modes et le remède WGAN, et calculer les gradients du discriminateur et du générateur pour un GAN jouet 1-D.
