---
id: gen.02
title: "Autoencodeurs et VAE"
summary: "De l'autoencodeur à goulot d'étranglement à l'autoencodeur variationnel : dériver l'ELBO et le terme KL gaussien, rendre l'échantillonnage différentiable grâce à l'astuce de reparamétrisation (vérifiée numériquement), et comprendre l'effondrement a posteriori."
minutes: 75
teaches: [gen.vae]
requires: [gen.taxonomy, info.entropy, nn.backprop, calc.chain]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/02-autoencoders-vae/docs/en.md: autoencoder vs VAE, reparameterisation, loss with the closed-form Gaussian KL, beta-VAE. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 10.1 Variational Inference (PDF p.482-483 read in detail): functionals and the idea of approximating a posterior by an analytic family. READ."
---

## Où cela s'inscrit

[[gen.01]] s'est terminée sur un problème : pour un modèle à variables latentes $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$, l'intégrale ne peut pas être calculée, donc nous ne pouvons pas maximiser directement la vraisemblance. L'**autoencodeur variationnel** (VAE) résout cela en optimisant une borne inférieure calculable, en utilisant un réseau de neurones pour approcher la loi a posteriori difficile. C'est aussi la première étape de tout modèle de diffusion latente ([[gen.05]]) : le décodeur du VAE est ce qui retransforme un latent débruité en image. Cette leçon s'appuie sur la rétropropagation ([[f2n.10]]), la règle de la chaîne et la divergence KL ([[math.10]]).

## L'idée en mots simples

Un **autoencodeur** est une paire de machines. L'encodeur comprime une photo en une courte liste de nombres (le code) ; le décodeur essaie de reconstruire la photo à partir de ce code. Comme le code est beaucoup plus court que la photo, le réseau doit ne garder que l'essentiel (c'est un cousin non linéaire de l'ACP). Un autoencodeur **débruiteur** est entraîné à reconstruire la photo propre à partir d'une photo corrompue, ce qui l'oblige à apprendre la structure plutôt que recopier les pixels.

Mais un autoencodeur ordinaire est un mauvais *générateur*. Son espace de codes est plein de trous : les codes de deux vrais chats peuvent être très éloignés avec du n'importe quoi entre les deux, et tu ne sais pas quels codes sont légaux. Le VAE corrige cela en rendant le code **flou et ordonné**. L'encodeur ne sort pas un code mais un petit nuage (un centre et une dispersion), le décodeur doit travailler à partir d'un point aléatoire de ce nuage, et une pénalité maintient tous les nuages proches d'une boule ronde standard. L'espace de codes devient alors une région lisse et remplie, et choisir un point aléatoire dans la boule ronde puis le décoder donne une nouvelle image plausible.

## Les autoencodeurs en un paragraphe

Encodeur $z=f_\phi(x)$ avec $x\in\mathbb R^d$, $z\in\mathbb R^k$, $k\ll d$. Décodeur $\hat x=g_\theta(z)$. Perte $\|x-\hat x\|^2$ moyennée sur les données. Un autoencodeur linéaire avec perte quadratique retrouve le sous-espace de l'ACP (engendré par les $k$ premiers vecteurs propres ; voir [[cml.05]]). Les autoencodeurs non linéaires le courbent. Un autoencodeur **débruiteur** reçoit $\tilde x=x+\text{noise}$ et demande toujours $x$. Souviens-toi que cette idée revient, fortement amplifiée, dans les modèles de diffusion ([[gen.04]]).

## Du modèle à variables latentes à l'ELBO

Modèle : a priori $p(z)=\mathcal N(0,I)$, décodeur $p_\theta(x\mid z)$. Nous voulons $\log p_\theta(x)=\log\int p_\theta(x\mid z)p(z)\,dz$. Introduisons une distribution quelconque $q_\phi(z\mid x)$ (l'encodeur). Multiplions et divisons à l'intérieur de l'intégrale :

$$
\log p_\theta(x)=\log\int q_\phi(z\mid x)\,\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\,dz
=\log\mathbb E_{q_\phi}\!\left[\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\right].
$$

**Inégalité de Jensen** : pour une fonction concave comme $\log$, $\log\mathbb E[Y]\ge\mathbb E[\log Y]$. Par conséquent

$$
\log p_\theta(x)\;\ge\;\mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)=:\mathrm{ELBO}(x).
$$

C'est la **borne inférieure de l'évidence** (ELBO, *evidence lower bound*). Pour voir à quel point elle est lâche, développons la KL entre l'encodeur et la *vraie* loi a posteriori $p_\theta(z\mid x)$. En utilisant $p_\theta(z\mid x)=p_\theta(x\mid z)p(z)/p_\theta(x)$ :

$$
\mathrm{KL}\big(q\,\|\,p_\theta(z\mid x)\big)=\mathbb E_q[\log q-\log p_\theta(x\mid z)-\log p(z)]+\log p_\theta(x)
=\log p_\theta(x)-\mathrm{ELBO}(x).
$$

Donc $\log p_\theta(x)=\mathrm{ELBO}(x)+\mathrm{KL}(q\|p_\theta(z\mid x))$. Une KL étant positive ou nulle, la borne est valide, et elle est **serrée exactement quand l'encodeur égale la vraie loi a posteriori**. Maximiser l'ELBO sur $\phi$ fait de l'encodeur une meilleure approximation de la loi a posteriori ; sur $\theta$, cela augmente la vraisemblance. C'est l'idée variationnelle de Bishop, Sec. 10.1 : remplacer une distribution intractable par le membre le plus proche d'une famille tractable, où « le plus proche » est mesuré par une divergence KL (Bishop travaille avec la même décomposition du log de la marginale en une borne plus une KL).

### Les deux termes

- **Reconstruction** $\mathbb E_q[\log p_\theta(x\mid z)]$ : si le décodeur est gaussien de variance fixée, $\log p_\theta(x\mid z)=-\frac1{2\sigma^2}\|x-g_\theta(z)\|^2+\text{const}$, c'est-à-dire l'erreur quadratique (à un facteur d'échelle près).
- **Régularisateur** $\mathrm{KL}(q_\phi(z\mid x)\|p(z))$ : tire chaque nuage de codes vers l'a priori.

La perte à *minimiser* est $-\mathrm{ELBO}=\text{reconstruction error}+\beta\,\mathrm{KL}$ avec $\beta=1$ pour la vraie borne. Choisir $\beta\ne1$ donne le $\beta$-VAE : $\beta<1$ favorise des reconstructions plus nettes, $\beta>1$ favorise un espace de codes plus régulier et une sortie plus floue (le compromis énoncé dans la leçon source).

## Le terme KL entre gaussiennes

Prenons $q=\mathcal N(\mu,\operatorname{diag}(\sigma^2))$ et $p=\mathcal N(0,I)$. Les deux se factorisent sur les $k$ coordonnées latentes, donc la KL est une somme de KL 1-D. Pour une coordonnée :

$$
\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1)\big)=\mathbb E_q\!\left[\log q(z)-\log p(z)\right].
$$

Écrivons $\log q(z)=-\tfrac12\log(2\pi\sigma^2)-\frac{(z-\mu)^2}{2\sigma^2}$ et $\log p(z)=-\tfrac12\log(2\pi)-\frac{z^2}{2}$. Sous $q$ : $\mathbb E[(z-\mu)^2]=\sigma^2$ et $\mathbb E[z^2]=\mu^2+\sigma^2$. Par conséquent

$$
\mathrm{KL}=-\tfrac12\log\sigma^2-\tfrac12+\tfrac12(\mu^2+\sigma^2)=\tfrac12\big(\mu^2+\sigma^2-\log\sigma^2-1\big).
$$

Sommée sur les coordonnées : $\mathrm{KL}=\tfrac12\sum_{i=1}^k(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1)$. Elle ne vaut $0$ qu'en $\mu=0,\sigma=1$ et croît quand un nuage s'éloigne de l'origine ou devient trop étroit ou trop large. La formule générale entre deux gaussiennes 1-D est $\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12$, qui se réduit à ce qui précède pour $(\mu_2,\sigma_2)=(0,1)$.

En pratique, l'encodeur sort $\log\sigma^2$ (appelé `logvar`), pas $\sigma$ : il n'est pas contraint et $\sigma=e^{\mathrm{logvar}/2}>0$ automatiquement.

## L'astuce de reparamétrisation

Le terme de reconstruction est une espérance sur $z\sim q_\phi(z\mid x)$ et nous l'estimons avec un seul échantillon. Mais *échantillonner* n'est pas une opération différentiable : si $z$ est tiré au hasard, que vaut $\partial z/\partial\phi$ ? L'astuce sort l'aléa des paramètres :

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I).
$$

Désormais $z$ est une fonction déterministe et différentiable de $(\mu,\sigma)$ pour un bruit $\varepsilon$ fixé, et la règle de la chaîne fonctionne. Pour un échantillon, avec la perte du décodeur $L$ et $\delta_z=\partial L/\partial z$ :

$$
\frac{\partial L}{\partial\mu}=\delta_z,\qquad
\frac{\partial L}{\partial\,\mathrm{logvar}}=\delta_z\odot\tfrac12\sigma\odot\varepsilon,
$$

car $\partial z/\partial\mathrm{logvar}=\varepsilon\cdot\partial\sigma/\partial\mathrm{logvar}=\varepsilon\cdot\tfrac12 e^{\mathrm{logvar}/2}$. Le terme KL a ses propres gradients : $\partial\mathrm{KL}/\partial\mu=\mu$ et $\partial\mathrm{KL}/\partial\mathrm{logvar}=\tfrac12(e^{\mathrm{logvar}}-1)$. L'exercice vérifie chacun d'eux par différences finies.

## L'effondrement a posteriori

Si le décodeur est très puissant (par exemple un décodeur de texte autorégressif), il peut bien modéliser $x$ *sans utiliser $z$*. Le moyen le moins coûteux de réduire la perte est alors de poser $q_\phi(z\mid x)=p(z)$ pour tout $x$ : la KL tombe à $0$ et le code ne porte aucune information. C'est l'**effondrement a posteriori** (*posterior collapse*). Remèdes typiques : recuit de la KL (partir de $\beta\approx0$ et monter jusqu'à 1), « free bits » (ne pas pénaliser la KL en dessous d'un plancher par dimension), un décodeur plus faible, ou du dropout de mots dans le texte. Un nombre croissant de dimensions « mortes » avec une KL $\approx0$ est le symptôme à surveiller.

## Exemple résolu

Prenons $x=(1,-0.5,2)$, $\mu=(0.3,-0.2)$, $\mathrm{logvar}=(-0.5,0.4)$, un bruit $\varepsilon=(0.7,-1.1)$, et un décodeur linéaire $\hat x=Wz+b$ avec $W\in\mathbb R^{3\times2}$ aléatoire initialisé et $b=(0.1,0,-0.1)$. (L'exercice fixe ces valeurs.) Le code échantillonné est $z=\mu+e^{\mathrm{logvar}/2}\varepsilon$ ; l'erreur de reconstruction quadratique vaut $10.7986$, la KL vaut $0.1642$ et la perte totale $10.9628$. Le gradient analytique de la perte par rapport à $\mu_1$ vaut $4.61077$ et la différence finie centrée concorde à neuf chiffres ; pour $\mathrm{logvar}_2$, la valeur analytique est $1.52861$, en accord avec la valeur numérique.

Vérification de la valeur de la KL : $\mu=(1,0)$, $\sigma^2=(1,4)$ donne $\tfrac12(1+1-0-1)+\tfrac12(0+4-\log4-1)=0.5+0.8069=1.3069$.

## Implémentation

```python
import numpy as np

def kl_to_std_normal(mu, logvar):
    return 0.5 * np.sum(np.exp(logvar) + mu**2 - logvar - 1, axis=-1)

def reparameterize(mu, logvar, eps):
    return mu + np.exp(0.5 * logvar) * eps

def vae_loss(x, mu, logvar, eps, W, b, beta=1.0):
    z = reparameterize(mu, logvar, eps)          # (k,)
    xh = W @ z + b                               # (d,)
    rec = np.sum((x - xh) ** 2)
    kl = kl_to_std_normal(mu, logvar)
    return rec + beta * kl, rec, kl
```

Le code de production (PyTorch) calcule la même chose sur des batchs, par exemple `mu, logvar = encoder(x); z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)`, et laisse l'autograd produire tous les gradients.

## Formes (shapes)

Pour un batch de $B$ points : `x: (B, d)`, `mu, logvar, eps, z: (B, k)`, `x_hat: (B, d)`. La KL est sommée sur $k$ ce qui donne `(B,)`, puis moyennée sur le batch. L'erreur de reconstruction est sommée sur $d$ puis moyennée. Si tu moyennes sur $d$ au lieu de sommer, tu modifies silencieusement $\beta$ d'un facteur $d$.

## Idées reçues

- **« Le VAE maximise la vraisemblance. »** Il maximise une borne inférieure ; l'écart est $\mathrm{KL}(q\|p_\theta(z\mid x))$.
- **« Le terme KL est une nuisance. »** Sans lui, l'encodeur réduit $\sigma\to0$ et le modèle dégénère en autoencodeur simple avec un espace de codes plein de trous.
- **« Des échantillons flous signifient un bogue. »** Un décodeur gaussien avec erreur quadratique prédit la *moyenne* des images plausibles, qui est floue. C'est intrinsèque à la perte, pas un bogue.
- **« Utilise $\sigma$ comme sortie de l'encodeur. »** Sortir $\log\sigma^2$ évite les contraintes de signe et les problèmes numériques près de $\sigma=0$.
- **« L'étape d'échantillonnage bloque les gradients. »** Seulement si tu tires $z$ directement ; la reparamétrisation expose le chemin passant par $\mu$ et $\sigma$.

## Points clés à retenir

- $\log p_\theta(x)=\mathrm{ELBO}+\mathrm{KL}(q\|p_\theta(z\mid x))\ge\mathrm{ELBO}$ ; la borne est serrée quand l'encodeur est la vraie loi a posteriori.
- ELBO = reconstruction $-$ KL vers l'a priori ; pour des gaussiennes diagonales, KL $=\tfrac12\sum(\mu^2+\sigma^2-\log\sigma^2-1)$.
- $z=\mu+\sigma\varepsilon$ rend l'échantillon différentiable ; gradients : $\delta_z$ pour $\mu$ et $\delta_z\cdot\tfrac12\sigma\varepsilon$ pour logvar.
- Les décodeurs puissants peuvent ignorer $z$ (effondrement a posteriori) ; recuis $\beta$ ou affaiblis le décodeur.

## Explique-le toi-même

1. Dérive l'ELBO avec l'inégalité de Jensen et explique ce que mesure l'écart.
2. Pourquoi ne peut-on pas rétropropager à travers « tirer $z\sim\mathcal N(\mu,\sigma^2)$ », et comment $z=\mu+\sigma\varepsilon$ résout-il le problème ?

## Pour aller plus loin

- Bishop, PRML, Sec. 10.1 (PDF p.482) pour le cadre variationnel ; l'article original du VAE de Kingma et Welling (non lu pour cette leçon).
- Suite : [[gen.03]] abandonne complètement la densité et l'encodeur.

## Critères de maîtrise

Tu sais dériver à la main l'ELBO et la KL gaussienne, implémenter une passe avant de VAE et sa perte, vérifier numériquement les gradients reparamétrés, et reconnaître et atténuer l'effondrement a posteriori.
