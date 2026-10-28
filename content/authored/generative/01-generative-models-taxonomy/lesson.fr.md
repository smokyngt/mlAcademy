---
id: gen.01
title: "Modèles génératifs : taxonomie et les mathématiques qu'ils partagent"
summary: "Ce que signifie modéliser une distribution de données, pourquoi les modèles discriminatifs et génératifs diffèrent, les familles de modèles génératifs, et le petit ensemble de quantités (vraisemblance, divergence KL, échantillonnage) qui servent à les juger tous."
minutes: 60
teaches: [gen.taxonomy]
requires: [prob.rv, info.entropy, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/01-generative-models-taxonomy-history/docs/en.md: the five-family taxonomy (explicit tractable, explicit approximate, implicit, score/continuous-time, token-based) and the history table. READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai (directory listing of lessons 01-19): course map of the phase. READ."
---

## Où cela s'inscrit

Jusqu'ici, presque tous les modèles de cette académie répondaient à une question de la forme « étant donnée une entrée $x$, quelle est l'étiquette $y$ ? ». C'est de la modélisation **discriminative**. Une seconde famille de problèmes demande les données elles-mêmes : écrire une phrase, dessiner un visage, composer une mélodie, proposer une molécule. Un modèle capable de cela est **génératif**. Cette leçon dessine la carte de tout le cours génératif : autoencodeurs et VAE ([[gen.02]]), GAN ([[gen.03]]), diffusion ([[gen.04]], [[gen.05]]) et flow matching avec évaluation ([[gen.06]]). Elle met aussi en place les quelques objets mathématiques qu'ils partagent tous, de sorte que chaque leçon suivante soit une variation sur un thème que tu connais déjà. Tu t'appuies directement sur les variables aléatoires et la gaussienne ([[math.09]]) ainsi que sur l'entropie et la divergence KL ([[math.10]]).

## L'idée en mots simples

Imagine un énorme tas de photographies de chats. Un modèle **discriminatif** est un videur qui regarde une photo et dit « chat » ou « pas chat ». Un modèle **génératif** est une artiste qui a si bien étudié le tas qu'elle peut peindre un nouveau chat qui n'y figurait pas. Le videur a seulement besoin de savoir où passe la frontière entre chats et non-chats. L'artiste, elle, doit savoir à quoi ressemblent les chats *partout* : il lui faut une image complète du tas, pas seulement de son bord. C'est pourquoi générer est le travail le plus difficile.

Les artistes travaillent de différentes façons :

- L'une **écrit l'image coup de pinceau après coup de pinceau**, chaque coup dépendant des précédents (modèles autorégressifs ; les modèles de langage font exactement cela avec les mots).
- L'une **compresse chaque image en un court code secret et apprend à le décompresser**, puis peint de nouvelles images en inventant de nouveaux codes (autoencodeurs, VAE).
- L'une **rivalise avec un critique** : elle peint, le critique dit « faux », elle s'améliore, le critique s'aiguise (GAN). Elle n'écrit jamais à quoi ressemble le tas, elle apprend seulement à tromper.
- L'une **part d'une feuille de neige (bruit) et la nettoie petit à petit** jusqu'à ce qu'une image apparaisse (modèles de diffusion et de flux).

Chaque artiste est une réponse différente à la même question : comment représenter une distribution en des millions de dimensions, et comment y tirer des échantillons ?

## Cadre : une distribution sur les données

Soit $x\in\mathbb{R}^d$ un point de données (une image est un vecteur de $d$ valeurs de pixels). On suppose que l'ensemble d'entraînement $x_1,\dots,x_N$ a été tiré indépendamment d'une distribution inconnue $p_\text{data}(x)$. Un **modèle** est une distribution $p_\theta(x)$ de paramètres $\theta$. Un modèle génératif a deux tâches :

1. **Estimation de densité** : rendre $p_\theta$ proche de $p_\text{data}$ et (idéalement) pouvoir évaluer $p_\theta(x)$ pour un $x$ donné.
2. **Échantillonnage** : produire de nouveaux $x\sim p_\theta$.

Les modèles discriminatifs apprennent $p(y\mid x)$. Les modèles génératifs apprennent $p(x)$, ou la loi jointe $p(x,y)$, qui donne les deux par la règle de Bayes :

$$
p(y\mid x)=\frac{p(x\mid y)\,p(y)}{\sum_{y'}p(x\mid y')\,p(y')} .
$$

Un modèle génératif doté de densités conditionnelles aux classes $p(x\mid y)$ peut donc *aussi* classer (c'est le bayésien naïf et l'analyse discriminante gaussienne). La réciproque est fausse : connaître seulement la frontière $p(y\mid x)$ ne te donne aucun moyen de tirer un $x$.

## Le maximum de vraisemblance revient à minimiser une divergence KL

Comment mesurer « proche » ? La réponse usuelle est la **vraisemblance**. Sous indépendance, la probabilité que le modèle assigne aux données est $\prod_i p_\theta(x_i)$, et nous maximisons son logarithme (une somme, plus facile à optimiser) :

$$
\hat\theta=\arg\max_\theta \frac1N\sum_{i=1}^N\log p_\theta(x_i).
$$

Voyons maintenant le lien avec la théorie de l'information. La divergence KL de $p_\text{data}$ vers $p_\theta$ vaut

$$
\mathrm{KL}(p_\text{data}\,\|\,p_\theta)=\mathbb{E}_{x\sim p_\text{data}}\big[\log p_\text{data}(x)-\log p_\theta(x)\big]
=-H(p_\text{data})-\mathbb{E}_{p_\text{data}}[\log p_\theta(x)] .
$$

L'entropie $H(p_\text{data})$ ne dépend pas de $\theta$. Minimiser la KL revient donc à maximiser $\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$, et la moyenne sur l'échantillon d'entraînement est une estimation de Monte Carlo de cette espérance (loi des grands nombres). Par conséquent :

$$
\text{maximum likelihood}\;\Longleftrightarrow\;\text{minimise }\mathrm{KL}(p_\text{data}\|p_\theta)\;\Longleftrightarrow\;\text{minimise cross-entropy } H(p_\text{data},p_\theta).
$$

La quantité $-\frac1N\sum_i\log p_\theta(x_i)$ est la **log-vraisemblance négative moyenne (NLL)**. Exprimée en bits par dimension, c'est le score standard pour les modèles à vraisemblance. Deux avertissements pratiques. La NLL sur les données d'entraînement peut toujours être abaissée en mémorisant : rapporte-la donc sur des données **de validation (mises de côté)**. Et les valeurs de NLL ne comparent que des modèles portant sur le même type de données (les densités continues dépendent des unités de $x$).

## La taxonomie

La leçon source range les modèles en cinq familles selon ce qu'ils modélisent et la façon dont ils paient la tractabilité. Voici la même carte organisée autour des deux questions auxquelles un modèle peut répondre : *puis-je calculer $p_\theta(x)$ ?* et *puis-je échantillonner à bas coût ?*

| Famille | Idée | Densité ? | Échantillonnage | Exemples |
|---|---|---|---|---|
| Explicite, tractable | écrire $p_\theta(x)$ comme un produit calculable ou une transformation inversible | exacte | autorégressif : une étape par composante (lent) | modèles autorégressifs, modèles de langage, flux normalisants |
| Explicite, approchée | minorer $\log p_\theta(x)$ et optimiser la borne | borne inférieure (ELBO) | décodeur à variable latente, ou débruitage itératif | VAE ([[gen.02]]), diffusion ([[gen.04]]) |
| Implicite | apprendre un échantillonneur $x=G(z)$ sans densité, entraîné contre un critique | aucune | un seul passage dans le réseau (rapide) | GAN ([[gen.03]]) |
| Score / temps continu | apprendre le champ de vecteurs qui déplace le bruit vers les données | via une EDO, coûteuse | intégrer une EDO ou une EDS | modèles de score, flow matching ([[gen.06]]) |
| Autorégressif à tokens | compresser les données en tokens discrets, puis modéliser la séquence de tokens | exacte sur les tokens | un token à la fois | modèles de tokens d'images/audio |

Deux idées traversent tout le tableau.

La **factorisation autorégressive** utilise la règle de la chaîne des probabilités, qui est exacte :

$$
p_\theta(x)=\prod_{j=1}^{d}p_\theta(x_j\mid x_{<j}).
$$

Chaque facteur est un petit classifieur ou une petite régression, donc l'entraînement est de la simple entropie croisée et $\log p_\theta(x)$ est exact. Le prix à payer : l'échantillonnage demande $d$ étapes séquentielles.

Les **variables latentes** supposent une cause cachée $z$ (un court code) qui génère $x$ : $p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz$. Intégrer sur $z$ est en général intractable, ce qui impose les approximations des leçons suivantes (VAE) ou évite complètement la densité (GAN).

Un bref historique, tiré du tableau de la source : VAE (2013), GAN (2014), flux tels que RealNVP et Glow (2017), diffusion DDPM (2020), systèmes texte-vers-image à diffusion latente et adaptateurs ControlNet/LoRA (2022), flow matching dans les grands modèles d'images (à partir de 2023). L'ordre de ce cours suit cet historique, car chaque méthode a été inventée pour corriger une faiblesse de la précédente : les VAE sont flous, les GAN instables et ils oublient des modes, la diffusion est lente, le flow matching raccourcit le chemin.

## Aperçu de l'évaluation

Un modèle à densité tractable peut être noté par sa NLL sur données de validation. Les modèles implicites et de nombreux modèles modernes ne le peuvent pas ; le domaine utilise donc aussi des scores **fondés sur les échantillons** : comparer des statistiques des échantillons générés à celles d'échantillons réels (FID, précision/rappel), mesurer l'accord avec un prompt, ou demander à des humains. [[gen.06]] dérive et implémente les principaux. Retiens la tension de base : la *qualité* (les échantillons sont-ils réalistes ?) et la *diversité/couverture* (le modèle atteint-il toutes les données ?) sont des propriétés différentes, et un seul nombre cache souvent laquelle a échoué.

## Exemple résolu : densité, échantillonnage et Bayes sur un mélange 1-D

Données : un mélange de deux gaussiennes, de poids $(0.5,0.5)$, de moyennes $(-2,2)$, d'écarts-types $(0.5,0.5)$. On tire 2000 points d'entraînement et 2000 points de test avec un générateur initialisé (l'exercice reproduit tous les nombres).

- **Modèle A**, une gaussienne ajustée par maximum de vraisemblance : l'EMV est la moyenne empirique et l'écart-type (biaisé) empirique, ce qui donne environ $\hat\mu\approx-0.015$, $\hat\sigma\approx2.05$. Il place sa masse en $0$, où il n'y a presque aucune donnée, et ses échantillons tomberaient surtout entre les deux vrais modes.
- **NLL moyenne de validation** : modèle A $\approx2.148$ nats ; le vrai mélange $\approx1.437$ nats. L'écart de $\approx0.71$ nat est le prix de la mauvaise famille de modèles, une estimation de $\mathrm{KL}(p_\text{data}\|p_A)$ puisque le terme d'entropie est commun.
- **Classification générative** : avec des densités conditionnelles $\mathcal N(\mp2,0.5^2)$ et des a priori égaux, $p(y=2\mid x=0)=0.5$ exactement (par symétrie). Avec des a priori $(0.9,0.1)$ et $x=0.5$, la probabilité a posteriori de la seconde classe est $0.997$ : le rapport de vraisemblance $e^{8}\approx2981$ écrase l'a priori de $9{:}1$.

## Implémentation

```python
import numpy as np

def gauss_logpdf(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - (x - mu)**2/(2*sigma**2)

def gmm_logpdf(x, w, mu, sigma):
    x = np.asarray(x, float)[:, None]                    # (N,1)
    lp = np.log(w) + gauss_logpdf(x, np.asarray(mu), np.asarray(sigma))   # (N,K)
    m = lp.max(1, keepdims=True)                          # log-sum-exp for stability
    return (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))[:, 0]

def gmm_sample(n, w, mu, sigma, rng):
    k = rng.choice(len(w), size=n, p=w)                   # ancestral sampling: pick a component...
    return rng.normal(np.asarray(mu)[k], np.asarray(sigma)[k])   # ...then draw from it
```

`gmm_sample` est le cas le plus simple d'**échantillonnage ancestral** : on tire la variable latente (la composante), puis on tire $x$ sachant celle-ci. Tous les modèles à variables latentes des leçons suivantes échantillonnent ainsi.

## Formes (shapes)

`x: (N,)` pour des données 1-D, ou `(N, d)` pour des données de dimension $d$. Les log-densités de mélange sont construites dans un tableau `(N, K)` (une colonne par composante) et réduites sur `K` par log-sum-exp. Les probabilités a posteriori de classe ont la forme `(N, n_classes)` et leurs lignes somment à 1.

## Idées reçues

- **« Un modèle génératif doit être un réseau de neurones. »** Un histogramme, une gaussienne et un mélange sont des modèles génératifs. Les modèles profonds ne font que passer la même idée à l'échelle.
- **« Une vraisemblance plus élevée signifie toujours de meilleurs échantillons. »** La vraisemblance et la qualité des échantillons peuvent diverger : un modèle peut avoir une bonne NLL et des échantillons flous, ou des échantillons nets et une NLL terrible. C'est pourquoi on rapporte plusieurs métriques.
- **« Les GAN et les modèles de diffusion estiment une densité. »** Un GAN n'a aucune densité. La diffusion optimise une borne sur celle-ci.
- **« Les valeurs de densité sont des probabilités. »** Une densité peut dépasser 1 (une gaussienne étroite). Seules les intégrales sur des régions sont des probabilités.
- **« Les classifieurs génératifs sont toujours moins bons. »** Ils demandent plus d'hypothèses mais apprennent souvent plus vite avec peu de données ; avec de mauvaises hypothèses, ils perdent face aux modèles discriminatifs.

## Points clés à retenir

- Les modèles génératifs apprennent $p(x)$ (ou $p(x,y)$) et peuvent échantillonner ; les modèles discriminatifs apprennent $p(y\mid x)$.
- Le maximum de vraisemblance équivaut à minimiser $\mathrm{KL}(p_\text{data}\|p_\theta)$, ce qui équivaut à minimiser l'entropie croisée.
- Les familles diffèrent par ce qu'elles sacrifient : vraisemblance exacte, échantillonnage peu coûteux ou stabilité de l'entraînement.
- Évalue toujours sur des données de validation et regarde à la fois la qualité et la couverture.

## Explique-le toi-même

1. Pourquoi minimiser la divergence KL de la distribution des données vers le modèle se réduit-il à maximiser la log-vraisemblance moyenne, alors que nous ne connaissons jamais $p_\text{data}$ ?
2. Donne un avantage et un inconvénient des modèles autorégressifs par rapport aux GAN.

## Pour aller plus loin

- Leçon source : phase 08, leçon 01 de la bibliothèque de cours (taxonomie et historique).
- Suite : [[gen.02]], où l'intégrale latente intractable est minorée.

## Critères de maîtrise

Tu sais ranger un modèle dans une famille, dériver « EMV = minimiser la KL », calculer une NLL de validation et échantillonner un mélange par échantillonnage ancestral, et expliquer pourquoi un classifieur ne peut pas être transformé en générateur.
