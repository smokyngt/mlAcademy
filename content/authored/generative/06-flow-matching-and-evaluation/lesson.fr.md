---
id: gen.06
title: "Flow matching et évaluation des générateurs"
summary: "Apprendre un champ de vitesse le long de chemins rectilignes du bruit vers les données et échantillonner avec une EDO (flow matching, rectified flow), puis mesurer honnêtement les générateurs : la distance de Fréchet derrière le FID, précision et rappel par k plus proches voisins, et préférence humaine par paires."
minutes: 85
teaches: [gen.flow-matching, gen.eval]
requires: [gen.diffusion, gen.taxonomy, la.eigen, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/08-generative-ai/13-flow-matching-rectified-flows/docs/en.md: straight-line interpolation, velocity target x1 - x0, conditional flow matching loss, Euler sampling backwards in time, rectified flow and reflow, why it displaced DDPM. READ."
  - "ai-engineering-from-scratch/phases/08-generative-ai/14-evaluation-fid-clip-score/docs/en.md: FID definition and failure modes, CLIP score, CMMD, human preference with Elo, evaluation checklist. READ."
---

## Où cela s'inscrit

La diffusion ([[gen.04]]) produit d'excellents échantillons mais demande de nombreuses petites étapes stochastiques parce que le chemin inverse est courbe et bruité. Le **flow matching** pose une question plus propre : puisque nous sommes libres de choisir comment relier le bruit aux données, relions-les par des *lignes droites* et apprenons la vitesse le long de celles-ci. L'échantillonnage devient une intégration d'EDO déterministe en peu d'étapes, et la perte d'entraînement est aussi simple que celle du DDPM. La seconde moitié de la leçon aborde la question que tout projet génératif finit par se poser : comment savoir que le modèle est bon ? Tu implémenteras la distance de Fréchet derrière le **FID**, une paire précision/rappel qui sépare la qualité de la couverture ([[gen.01]]), et l'arithmétique de notation des études de préférence humaine. Elle utilise les décompositions en valeurs propres de [[math.07]] et des faits gaussiens de [[math.09]].

## L'idée en mots simples

Imagine deux nuages de points : un nuage rond de « bruit » et un nuage en forme de tes données. Apparie chaque point de bruit à un point de données et tends une corde droite entre eux. Un flot est une règle de circulation qui dit, pour chaque endroit et chaque instant, dans quelle direction aller pour que les points de bruit glissent le long des cordes jusqu'aux données. Entraîner, c'est apprendre cette règle à partir de nombreux appariements aléatoires. Le hic : à un endroit et un instant donnés, des cordes issues de nombreux appariements différents se croisent, donc la règle apprise est la direction *moyenne* de toutes les cordes passant par ce point. Suivre le champ moyenné porte quand même tout le nuage de bruit sur le nuage de données, mais les trajectoires individuelles s'incurvent un peu. Moins il y a de croisements, plus les chemins sont droits et moins il faut d'étapes. Le « reflow » réapparie les points pour réduire les croisements.

Pour l'évaluation, l'analogie est un concours de vins. Un juge de la *qualité* demande « chaque échantillon a-t-il le goût d'un vrai vin ? ». Un juge de la *couverture* demande « tes échantillons couvrent-ils toutes les sortes de vrais vins ? ». Le FID mélange les deux en un seul nombre ; la précision mesure le premier, le rappel le second ; et les humains sont le dégustateur final.

## Flow matching

Soit $x_0\sim p_\text{data}$ et $x_1\sim\mathcal N(0,I)$ (note : cette leçon place les données en $t=0$ et le bruit en $t=1$, suivant la source). Le chemin rectiligne est

$$
x_t=(1-t)\,x_0+t\,x_1,\qquad \frac{dx_t}{dt}=x_1-x_0 .
$$

Un réseau $v_\theta(x,t)$ est entraîné avec la perte de **flow matching conditionnel**

$$
L_\text{CFM}=\mathbb E_{t\sim\mathrm U[0,1],\,x_0,\,x_1}\big\|v_\theta(x_t,t)-(x_1-x_0)\big\|^2 .
$$

Elle est *sans simulation* : on échantillonne une paire et un instant, on forme $x_t$ en une ligne, on régresse. Il n'y a aucun calendrier de bruit à régler et aucune résolution d'EDO à l'intérieur de l'entraînement.

### Pourquoi régresser sur une cible propre à chaque paire apprend le bon champ

La cible $u=x_1-x_0$ dépend de la paire, mais le réseau ne voit que $(x_t,t)$. La régression par erreur quadratique est minimisée par l'espérance conditionnelle, donc le champ optimal est la **vitesse marginale**

$$
v^*(x,t)=\mathbb E\big[x_1-x_0\ \big|\ x_t=x\big].
$$

Formellement, $\mathbb E\|v_\theta-u\|^2=\mathbb E\|v_\theta-v^*\|^2+\mathbb E\|u-v^*\|^2$ (le terme croisé s'annule car $u-v^*$ est de moyenne conditionnelle nulle sachant $x_t$), et le dernier terme ne dépend pas de $\theta$. Minimiser la CFM revient donc à approcher $v^*$ au sens des moindres carrés. Par l'équation de continuité, un flot de $v^*$ transporte la distribution marginale de $x_t$ pour tout $t$, et donc amène $\mathcal N(0,I)$ en $t=1$ vers $p_\text{data}$ en $t=0$ (on suppose $v^*$ bien comportée).

### Échantillonnage

Tirer $x_1\sim\mathcal N(0,I)$ et intégrer $dx/dt=v_\theta(x,t)$ **à rebours** de $t=1$ à $0$. La méthode d'Euler avec $N$ pas de taille $\Delta t=1/N$ est

$$
x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t).
$$

Des solveurs d'ordre supérieur (Heun, Runge-Kutta) atteignent la même précision avec moins de pas. Un champ rectiligne n'aurait besoin que d'un seul pas.

### Rectified flow et reflow

Même le $v^*$ optimal a des trajectoires courbes partout où des cordes se croisent. **Reflow** (Liu et al.) : entraîner $v^{(1)}$, puis générer des paires $(x_1,\hat x_0)$ en l'intégrant, et entraîner un nouveau modèle $v^{(2)}$ sur les lignes droites entre ces paires *appariées par l'EDO*. Les paires appariées se croisent beaucoup moins, donc le nouveau champ est bien plus droit ; deux tours suffisent typiquement pour un échantillonnage en 2-4 pas. Les modèles distillés en une étape de la liste de la leçon source vont dans la même direction. C'est une description de haut niveau ; je n'ai pas dérivé ici la garantie de redressement.

## Exemple résolu : flot exact pour des données gaussiennes

Pour $x_0\sim\mathcal N(m,s^2)$, $x_1\sim\mathcal N(0,1)$ indépendants, $x_t$ est gaussien de moyenne $(1-t)m$ et de variance $(1-t)^2s^2+t^2$. La vitesse $u=x_1-x_0$ a pour moyenne $-m$ et $\mathrm{Cov}(u,x_t)=t-(1-t)s^2$. La régression gaussienne donne le champ optimal en forme close :

$$
v^*(x,t)=-m+\frac{t-(1-t)s^2}{(1-t)^2s^2+t^2}\,\big(x-(1-t)m\big).
$$

Vérification par Monte Carlo : à $t=0.4$, $x=1.5$, $m=3$, $s=0.5$, la formule donne $-3.3$ ; en moyennant $x_1-x_0$ sur deux millions de paires simulées avec $x_t$ à moins de $0.01$ de $1.5$, on obtient $-3.304$.

Échantillonnons maintenant 20 000 points avec Euler et $N$ pas, $m=3$, $s=0.5$ (graine 0) :

| pas $N$ | moyenne des échantillons | écart-type des échantillons |
|---|---|---|
| 1 | 3.000 | 0.000 |
| 2 | 3.001 | 0.199 |
| 4 | 3.002 | 0.339 |
| 20 | 3.002 | 0.462 |
| 100 | 3.002 | 0.491 |

Le vrai écart-type est $0.5$. Avec un pas, *chaque* point de bruit atterrit sur la moyenne $3$ : le premier pas d'Euler utilise $v^*(x,1)=x-m$, qui est la moyenne sur toutes les cordes qui se croisent, donc il ne peut pas séparer les points de données individuels. Avec plus de pas, l'intégration plus fine suit les chemins courbes et retrouve la dispersion. C'est toute la motivation du redressement des chemins. C'est un cas d'école : les vraies données sont multimodales et la courbure est bien pire.

## Évaluer les générateurs

### La distance de Fréchet et le FID

Prenons un extracteur de caractéristiques (pour les images, le FID utilise les activations d'un réseau Inception-v3, 2048 nombres par image). Ajustons une gaussienne $\mathcal N(\mu_r,\Sigma_r)$ aux caractéristiques des images réelles et $\mathcal N(\mu_g,\Sigma_g)$ à celles des images générées. La **distance de Fréchet** (distance de Wasserstein-2 entre gaussiennes) est

$$
d^2=\|\mu_r-\mu_g\|^2+\mathrm{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big).
$$

Deux vérifications de cohérence. En 1-D, avec les variances $\sigma_r^2,\sigma_g^2$, c'est $(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2$ : les moyennes et les dispersions doivent toutes deux concorder (exemple : $\mathcal N(0,1)$ contre $\mathcal N(2,4)$ donne $4+1=5$). Avec des gaussiennes égales, c'est $0$. La racine carrée matricielle d'un produit non symétrique est délicate ; la trace se calcule de façon stable comme la somme des racines carrées des valeurs propres de $\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}$, qui est symétrique semi-définie positive (les racines carrées de matrices SDP s'obtiennent par `eigh`, voir [[math.07]]). C'est ce que fait l'exercice. Exemple en 2-D, $\mu_r=0$, $\mu_g=(1,2)$, $\Sigma_r=\begin{pmatrix}2&0.3\\0.3&1\end{pmatrix}$, $\Sigma_g=\mathrm{diag}(1,1.5)$ : $d^2=5.252$.

**Les modes de défaillance du FID** (plusieurs figurent dans la source ; un que j'ai corrigé). L'hypothèse gaussienne ignore la multimodalité ; les caractéristiques viennent d'un réseau entraîné sur ImageNet, donc les domaines éloignés d'ImageNet donnent un nombre douteux ; le score est sensible au prétraitement. **Biais de taille d'échantillon :** la leçon source dit qu'un petit $N$ donne un FID faussement *bas*. Mon expérience montre le contraire : deux échantillons de la *même* gaussienne 8-D donnent un FID de $2.64$ pour $N=20$, $0.554$ pour $N=100$, $0.0497$ pour $N=1000$, $0.0052$ pour $N=10000$ (graine fixée, reproduit dans l'exercice). Des estimations bruitées de covariance augmentent la distance, qui décroît en $1/N$ vers 0. Dans les deux cas, la règle pratique est la même : ne comparer les modèles qu'au même grand $N$ (la source recommande 10 000 ou plus). Note aussi qu'un seul scalaire cache *pourquoi* un modèle a un mauvais score.

### Précision et rappel pour les générateurs

On sépare les deux aspects avec des ensembles de vecteurs de caractéristiques. Pour chaque point $x$ d'un ensemble $S$, soit $r_k(x)$ la distance à son $k$-ième plus proche voisin dans $S$ (lui-même exclu) ; la boule de ce rayon est une estimation du support local. Alors

- **précision** = fraction des points générés situés dans le support estimé des données réelles (les échantillons sont-ils réalistes ?),
- **rappel** = fraction des points réels situés dans le support estimé des données générées (tous les modes réels sont-ils couverts ?).

Expérience (400 points réels $\sim\mathcal N(0,I_2)$, $k=3$, graine fixée) :

| ensemble généré | précision | rappel |
|---|---|---|
| même distribution | 0.9825 | 0.975 |
| effondré ($\mathcal N(0,0.1^2)$) | 1.000 | 0.035 |
| décalé très loin | 0.0625 | 0.0875 |

Un générateur effondré a une précision parfaite et un rappel désastreux, exactement la signature d'effondrement de modes de [[gen.03]], que le FID seul rapporterait simplement comme « mauvais ». J'ai écrit cette méthode à partir de ma connaissance générale de l'idée d'estimation de support par k plus proches voisins ; la leçon source ne couvre pas la précision/rappel.

### Autres métriques automatiques

*Score CLIP* : similarité cosinus entre les embeddings d'image et de prompt, mesure de la fidélité au prompt ; il hérite des faiblesses de CLIP (raisonnement compositionnel médiocre, biais lié à la longueur du prompt). *CMMD* : une discrépance maximale moyenne (MMD) dans l'espace de caractéristiques CLIP, sans hypothèse gaussienne. Les évaluateurs de préférence appris (HPSv2, ImageReward, PickScore) sont des substituts entraînés sur des choix humains. Le conseil de la source est d'en rapporter plusieurs (qualité des échantillons, fidélité au prompt, préférence) plus une revue qualitative d'échantillons aléatoires.

### Évaluation humaine avec Elo

On montre aux évaluateurs des paires d'images de deux modèles pour le même prompt et on note le gagnant. Les classements suivent la logique de Bradley-Terry d'Elo : le score espéré de $A$ contre $B$ est

$$
E_A=\frac1{1+10^{(R_B-R_A)/400}},\qquad R_A\leftarrow R_A+K\,(S_A-E_A),
$$

avec $S_A=1$ pour une victoire, $0$ pour une défaite et typiquement $K=32$. Des classements égaux donnent $E_A=0.5$ ; si A (1000) bat B (1000), les nouveaux classements sont $1016$ et $984$ ; un favori à 200 points a $E=0.760$ et ne gagne que $7.7$ en gagnant. Règles de bonne pratique : aveugler les paires, utiliser des prompts mis de côté (non vus à l'entraînement), mélanger les groupes d'annotateurs, et traiter un juge LLM comme un substitut qui peut être dupé par des sorties jolies mais fausses.

## Implémentation

```python
import numpy as np

def sqrtm_psd(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

def frechet_distance(mu1, S1, mu2, S2):
    r = sqrtm_psd(S1)
    M = r @ S2 @ r
    tr = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr)
```

En pratique, utilise `torchmetrics`/`pytorch-fid` avec un point de contrôle Inception fixe et des statistiques de données réelles mises en cache ; ne compare jamais des FID calculés avec des extracteurs, résolutions ou codes de redimensionnement différents.

## Formes (shapes)

Caractéristiques `(N, d)` ; moyenne `(d,)` ; covariance `np.cov(F, rowvar=False)` de forme `(d, d)` (demande $N>d$, sinon singulière). Flow matching : `x0, x1, x_t: (B, ...)`, `t: (B,)` diffusé (broadcast) à la forme des données, `v_theta(x_t, t)` de même forme que `x_t`. Distances par paires pour les k plus proches voisins : `(N_fake, N_real)`.

## Idées reçues

- **« Le flow matching et la diffusion n'ont aucun rapport. »** Pour des chemins gaussiens, ce sont deux paramétrisations de modèles étroitement liés ; les différences pratiques sont la cible de régression sans simulation, les chemins plus droits et l'échantillonnage par EDO.
- **« Un pas suffit une fois entraîné. »** Seulement si le champ appris est rectiligne ; sinon un pas renvoie la moyenne conditionnelle (la moyenne floue).
- **« Un FID plus bas est toujours meilleur. »** Il est biaisé par la taille d'échantillon, l'extracteur et le domaine ; il ne sait pas distinguer la qualité de la couverture.
- **« La précision et le rappel sont ici des métriques de classifieur. »** Ce sont des mesures de couverture d'ensemble dans l'espace de caractéristiques et elles demandent un bon extracteur et un choix de $k$.
- **« La préférence humaine est la vérité terrain. »** Elle dépend des prompts, du groupe d'évaluateurs et de l'appariement ; elle demande la même discipline que toute expérience.

## Points clés à retenir

- Le flow matching régresse la vitesse $x_1-x_0$ le long de chemins rectilignes ; le minimiseur est $\mathbb E[x_1-x_0|x_t]$, et l'échantillonnage intègre l'EDO du bruit vers les données.
- Moins de pas demandent des chemins plus droits (reflow, distillation).
- Le FID est la distance de Fréchet entre ajustements gaussiens dans un espace de caractéristiques : $\|\Delta\mu\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$.
- Rapporte séparément la qualité et la couverture (précision, rappel), plus la fidélité au prompt et une préférence humaine en aveugle.

## Explique-le toi-même

1. Pourquoi le champ optimal du flow matching est-il égal à $\mathbb E[x_1-x_0\mid x_t]$, et pourquoi un unique pas d'Euler renvoie-t-il alors la moyenne des données dans l'exemple gaussien ?
2. Un générateur a une précision élevée et un rappel faible. Qu'est-ce que cela dit des échantillons, et à quel mode de défaillance de l'entraînement des GAN cela ressemble-t-il ?

## Pour aller plus loin

- Leçons sources : phase 08, leçons 13 (flow matching, rectified flows) et 14 (FID, score CLIP, préférences). Les articles originaux (Lipman et al., Liu et al., Heusel et al., Kynkaanniemi et al.) n'ont pas été lus pour cette leçon.
- Retour à la vue d'ensemble : [[gen.01]].

## Critères de maîtrise

Tu sais énoncer et justifier la perte de flow matching et son champ optimal, écrire un échantillonneur d'Euler, calculer la distance de Fréchet à partir de statistiques de caractéristiques, mesurer précision et rappel avec des boules de k plus proches voisins, et mettre à jour des classements Elo à partir de résultats de préférence.
