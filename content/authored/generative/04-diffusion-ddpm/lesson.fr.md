---
id: gen.04
title: "Modèles de diffusion (DDPM)"
summary: "Détruire les données par du bruit gaussien dans un processus direct en forme close, apprendre à le défaire pas à pas en prédisant le bruit, et échantillonner en exécutant la chaîne inverse apprise, le tout dérivé étape par étape et testé contre un débruiteur analytique exact sur des données gaussiennes 1-D."
minutes: 90
teaches: [gen.diffusion]
requires: [gen.vae, prob.rv, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/06-diffusion-ddpm-from-scratch/docs/en.md: forward process with closed form, noise-prediction loss L_simple, reverse step formula, linear beta schedule 1e-4 to 0.02 over T=1000, intuitions (denoising is easy, score matching). READ."
---

## Où cela s'inscrit

Les GAN ([[gen.03]]) sont rapides mais instables ; les VAE ([[gen.02]]) sont stables mais flous. Les **modèles probabilistes de diffusion par débruitage (DDPM)** gardent la stabilité d'une simple perte de régression et atteignent la meilleure qualité d'échantillons, ce qui explique qu'ils alimentent les générateurs actuels d'images, d'audio et de vidéo. L'astuce centrale consiste à découper une tâche impossible (« dessiner une image à partir de rien ») en mille toutes petites (« retirer un peu de bruit »). Cette leçon dérive toute la méthode avec la seule algèbre gaussienne, puis la teste de bout en bout sur des données assez simples pour que le débruiteur parfait puisse être écrit. La leçon suivante ([[gen.05]]) la fait passer à l'échelle ; [[gen.06]] montre une alternative plus directe.

## L'idée en mots simples

Prends une photographie et ajoutes-y une pincée de neige (bruit). Recommence, encore et encore, mille fois. L'image se dissout en pur bruit, selon un chemin que tu contrôles entièrement. Maintenant, passe le film à l'envers : en partant du bruit, si tu avais un assistant capable de regarder une image légèrement bruitée et de dire « voilà la part de bruit », tu pourrais retirer ce bruit pas à pas, et le bruit se transformerait en image.

Pourquoi le travail de l'assistant est-il faisable ? Quand l'image est presque du pur bruit, deviner une minuscule correction est facile : l'image moyenne est une bonne estimation. Quand l'image est presque propre, la correction est petite et locale. Le milieu difficile est géré par un seul réseau à qui l'on dit à quel point l'entrée est bruitée. Point crucial : on peut *fabriquer* gratuitement les données d'entraînement de l'assistant : prendre une image réelle, ajouter une quantité connue de bruit, et demander au réseau de nommer le bruit. Pas de compétition, pas de codes latents, juste de la régression.

## Processus direct

Choisis un **calendrier de bruit** (*noise schedule*) $\beta_1,\dots,\beta_T\in(0,1)$ (petits nombres). En partant des données $x_0$, chaque étape réduit un peu le signal et ajoute du bruit gaussien frais :

$$
q(x_t\mid x_{t-1})=\mathcal N\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\ \beta_tI\big),\qquad x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon_t .
$$

Définissons $\alpha_t=1-\beta_t$ et $\bar\alpha_t=\prod_{s=1}^t\alpha_s$.

### Forme close pour tout $t$

Déroulons deux étapes : $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_t$ et $x_{t-1}=\sqrt{\alpha_{t-1}}\,x_{t-2}+\sqrt{1-\alpha_{t-1}}\,\varepsilon_{t-1}$. Alors

$$
x_t=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\,\varepsilon_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_t .
$$

Les deux termes de bruit sont des gaussiennes indépendantes centrées, donc leur somme est gaussienne de variance la somme des variances : $\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$. Le résultat a la même forme qu'une seule étape avec $\alpha_t\alpha_{t-1}$ à la place de $\alpha_t$. Par récurrence

$$
\boxed{q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ \ \varepsilon\sim\mathcal N(0,I).}
$$

C'est ce qui rend l'entraînement bon marché : pour obtenir un exemple bruité à *n'importe quel* instant $t$, tu ne simules pas $t$ étapes, tu sautes directement. Avec le calendrier linéaire standard $\beta_t$ de $10^{-4}$ à $0.02$ sur $T=1000$ étapes, $\bar\alpha_{10}=0.998$, $\bar\alpha_{100}=0.897$, $\bar\alpha_{300}=0.396$, et $\bar\alpha_{1000}=4.0\times10^{-5}$ : le $x_T$ final est essentiellement $\mathcal N(0,I)$ (poids du signal $\sqrt{\bar\alpha_T}=0.006$). Les nombres sont calculés par l'exercice.

## Le processus inverse et sa cible

Nous voulons échantillonner en partant de $x_T\sim\mathcal N(0,I)$ et en tirant $x_{t-1}\sim p_\theta(x_{t-1}\mid x_t)$, modélisé par $\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I)$. Que doit valoir $\mu_\theta$ ? La règle de Bayes donne la **vraie étape inverse quand $x_0$ est connu** : $q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$, un produit de deux gaussiennes en $x_{t-1}$, donc gaussien. En complétant le carré dans l'exposant, on obtient

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_tI),\quad
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\quad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .
$$

Au moment du test, $x_0$ est inconnu. Mais d'après la forme close, $x_0=\dfrac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}$, donc $x_0$ est déterminé par le bruit $\varepsilon$ contenu dans $x_t$. Substituons et simplifions (avec $1-\bar\alpha_t=\beta_t+\alpha_t(1-\bar\alpha_{t-1})$) :

$$
\tilde\mu_t=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\Big).
$$

La seule inconnue de la moyenne inverse est donc le bruit $\varepsilon$. Laissons un réseau $\varepsilon_\theta(x_t,t)$ le prédire et posons $\mu_\theta=\frac1{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\big)$.

## La perte

L'entraînement maximise une borne inférieure variationnelle de $\log p_\theta(x_0)$, exactement comme pour le VAE ([[gen.02]]) mais avec les $T$ variables bruitées comme variables latentes. La borne se décompose en une KL par étape, chacune entre deux gaussiennes de même variance, $\mathrm{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)=\frac1{2\sigma_t^2}\|\tilde\mu_t-\mu_\theta\|^2+\text{const}$. En substituant les deux moyennes, la différence vaut $\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\varepsilon-\varepsilon_\theta)$, donc chaque terme est

$$
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\ \|\varepsilon-\varepsilon_\theta(x_t,t)\|^2 .
$$

En abandonnant les poids dépendant de $t$ (ce que Ho et al. ont trouvé plus efficace), on obtient la **perte simple**

$$
\boxed{L_\text{simple}=\mathbb E_{x_0,\,t\sim\mathrm{U}\{1..T\},\,\varepsilon}\Big[\big\|\varepsilon-\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\big)\big\|^2\Big]}
$$

Boucle d'entraînement : tirer un point de données, un $t$ aléatoire, un bruit $\varepsilon$ ; former $x_t$ en une ligne ; régresser le bruit. Une seule perte stable, pas de minimax. Dans les systèmes réels, $\varepsilon_\theta$ est un U-Net ou un transformer prenant $(x_t,t)$ avec $t$ plongé (embedding) ; la perte est la même.

## Échantillonnage

Partir de $x_T\sim\mathcal N(0,I)$. Pour $t=T,\dots,1$ :

$$
x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\Big)+\sigma_t z,\qquad z\sim\mathcal N(0,I)\ (z=0\text{ at the last step}),
$$

avec $\sigma_t^2=\beta_t$ (ou $\tilde\beta_t$). C'est l'**échantillonnage ancestral**. Il demande $T$ évaluations du réseau, ce qui explique que la diffusion soit lente comparée à un GAN ; des échantillonneurs plus rapides (DDIM, moins d'étapes, distillation) et le flow matching ([[gen.06]]) réduisent le coût. Autre point de vue : $\varepsilon_\theta$ est, à un facteur $-\sqrt{1-\bar\alpha_t}$ près, le **score** $\nabla_{x_t}\log q(x_t)$, de sorte que chaque étape déplace $x_t$ vers le haut dans la log-densité des données bruitées, plus une injection de bruit (dynamique de type Langevin).

## Exemple résolu avec un débruiteur exact

Si les données sont $x_0\sim\mathcal N(m_0,s_0^2)$ en 1-D, tout est gaussien et le prédicteur de bruit parfait est connu en forme close. La marginale est $x_t\sim\mathcal N(\sqrt{\bar\alpha_t}m_0,\ \bar\alpha_ts_0^2+1-\bar\alpha_t)$. La moyenne conditionnelle de $x_0$ sachant $x_t$ est la formule de régression gaussienne

$$
\mathbb E[x_0\mid x_t]=m_0+\frac{\sqrt{\bar\alpha_t}\,s_0^2}{\bar\alpha_ts_0^2+1-\bar\alpha_t}\big(x_t-\sqrt{\bar\alpha_t}m_0\big),\qquad
\varepsilon^*(x_t,t)=\frac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]}{\sqrt{1-\bar\alpha_t}} .
$$

($\varepsilon^*$ est le minimiseur de $L_\text{simple}$ : la moyenne conditionnelle de $\varepsilon$ sachant $x_t$.) Prenons $m_0=3,\ s_0=0.5$, le calendrier linéaire avec $T=1000$, et la boucle d'échantillonnage ci-dessus avec ce $\varepsilon^*$ tenant lieu de réseau entraîné.

- À $t=300$ avec $x_t=2.0$ : $\mathbb E[x_0|x_t]=3.0249$, $\varepsilon^*=0.1229$.
- Vérification directe (200k tirages à $t=500$) : moyenne empirique $0.8401$ et écart-type $0.9704$ contre la théorie $\sqrt{\bar\alpha}m_0=0.8410$ et $\sqrt{\bar\alpha s_0^2+1-\bar\alpha}=0.9701$.
- La perte $\mathbb E\|\varepsilon-\hat\varepsilon\|^2$ à $t=300$ vaut $0.139$ pour $\varepsilon^*$, contre $1.0$ pour la référence « prédire zéro » et $0.144$ pour la prédiction de $x_t$ standardisé sans utiliser la structure des données.
- En échantillonnant 20 000 points à partir d'un bruit $\mathcal N(0,1)$ : moyenne $3.004$, écart-type $0.4986$, soit une récupération de $\mathcal N(3,0.5^2)$ à deux décimales. Avec ce minuscule modèle des données, la mécanique de la diffusion fonctionne exactement.

## Implémentation

```python
import numpy as np

def linear_betas(T, b0=1e-4, b1=0.02): return np.linspace(b0, b1, T)
def alpha_bars(betas): return np.cumprod(1 - betas)

def q_sample(x0, t, abar, eps):                       # jump straight to step t
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps

def sample(n, betas, eps_fn, rng):
    al = 1 - betas; ab = alpha_bars(betas)
    x = rng.normal(size=n)                            # x_T
    for t in range(len(betas) - 1, -1, -1):
        e = eps_fn(x, t)
        x = (x - betas[t] / np.sqrt(1 - ab[t]) * e) / np.sqrt(al[t])
        if t > 0: x = x + np.sqrt(betas[t]) * rng.normal(size=n)
    return x
```

(Les tableaux sont indexés à partir de zéro : l'indice $t-1$ dans le code correspond à l'étape $t$ dans les maths.) Avec PyTorch, le réseau est `eps_model(x_t, t)` et l'étape d'entraînement est `loss = ((eps - eps_model(x_t, t))**2).mean()`.

## Formes (shapes)

Pour des images : `x0, eps, x_t: (B, C, H, W)` ; `t: (B,)` étape entière par exemple ; `abar[t]` est collecté en `(B, 1, 1, 1)` pour se diffuser (broadcast). La sortie du réseau a la même forme que `x_t`.

## Idées reçues

- **« Le réseau retire le bruit en une seule fois. »** Il prédit le bruit à chaque niveau ; l'échantillonnage demande toujours de nombreuses petites étapes.
- **« Le processus direct est appris. »** Il est fixe ; seul le modèle inverse est entraîné.
- **« Plus d'étapes donnent toujours de meilleurs échantillons. »** Au-delà d'un certain point, elles ne coûtent que du temps ; de nombreux échantillonneurs obtiennent de bons résultats en 20-50 étapes.
- **« Prédire le bruit est arbitraire. »** C'est une reparamétrisation équivalente de la prédiction de $x_0$ ou du score ; elle donne simplement une cible bien mise à l'échelle à tous les niveaux de bruit.
- **« Le bruit d'échantillonnage $\sigma_tz$ est optionnel. »** Le supprimer change le processus (un échantillonneur déterministe est une méthode différente et valide, p. ex. DDIM, mais cette formule a besoin du bruit).

## Points clés à retenir

- $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$, donc les exemples d'entraînement à n'importe quel niveau de bruit sont gratuits.
- La moyenne inverse dépend du bruit inconnu : $\tilde\mu_t=\frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon)$.
- L'entraînement est une régression sur le bruit ; l'échantillonnage consiste en $T$ petites étapes de débruitage à partir du pur bruit.
- Avec une distribution de données gaussienne connue, un débruiteur exact existe, et l'échantillonneur reproduit la distribution des données.

## Explique-le toi-même

1. Dérive $q(x_t\mid x_0)$ en additionnant des gaussiennes indépendantes et explique pourquoi cela compte pour la vitesse d'entraînement.
2. Explique pourquoi le prédicteur de bruit optimal est une moyenne conditionnelle et pourquoi son erreur n'est jamais nulle.

## Pour aller plus loin

- Leçon source : phase 08, leçon 06 (DDPM from scratch). Article DDPM original de Ho, Jain et Abbeel (non lu ici).
- Suite : [[gen.05]] (diffusion latente, guidage) et [[gen.06]] (flow matching).

## Critères de maîtrise

Tu sais dériver la marginale directe en forme close et la moyenne inverse paramétrée par le bruit, écrire l'étape d'entraînement et la boucle d'échantillonnage, et vérifier un échantillonneur contre un débruiteur analytique.
