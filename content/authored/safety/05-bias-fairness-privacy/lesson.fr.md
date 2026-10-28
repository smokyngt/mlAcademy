---
id: saf.05
title: "Biais, équité et vie privée"
summary: "Mesurer les préjudices de représentation et d'allocation, calculer les critères d'équité de groupe et voir pourquoi ils ne peuvent pas tous tenir en même temps, puis construire les mécanismes de Laplace et gaussien et l'écrêtage de DP-SGD et vérifier empiriquement leurs garanties de confidentialité."
minutes: 90
teaches: [saf.fairness-criteria, saf.differential-privacy]
requires: [prob.rv, prob.basics, ml.logreg, stat.testing]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/20-bias-representational-harm/docs/en.md (whole lesson): representational vs allocational harm, metric families, intersectionality, mechanistic approaches, meta-critique. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual/docs/en.md (whole lesson): group, individual and counterfactual fairness, impossibility, backtracking counterfactuals. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/docs/en.md (whole lesson): (epsilon, delta)-DP, DP-SGD recipe, LoRA+DP, MIA vs extraction tension, alternatives. READ."
---

## Où cela s'insère

Les leçons de sécurité précédentes portaient sur des adversaires et des optimiseurs. Cette leçon porte sur un préjudice qui n'a besoin d'aucun méchant : un modèle précis en moyenne mais qui traite différemment des groupes de personnes, ou qui mémorise et révèle des données sur les individus sur lesquels il a été entraîné. Ce sont des propriétés mesurables, donc la leçon est surtout faite de *définitions que tu peux calculer* et de *théorèmes qui disent ce qu'on ne peut pas avoir ensemble*. Elle dépend des probabilités conditionnelles et des variables aléatoires (`prob.basics`, `prob.rv`) et des bases de la classification (`ml.logreg`), et les habitudes d'évaluation de [[saf.04]] (rapporter l'incertitude) s'appliquent ici aussi. Le contexte de déploiement est dans [[saf.06]].

## L'idée en termes simples

**Biais.** Un système peut être injuste de deux manières différentes. Il peut *représenter* des personnes injustement (toujours imaginer les infirmiers comme des femmes : un préjudice **de représentation**), ou il peut *décider* injustement (noter différemment des candidats également qualifiés : un préjudice **d'allocation**). Un modèle peut réussir les tests de l'un et échouer à ceux de l'autre, donc on mesure les deux.

**Équité.** « Traiter les gens de manière égale » a plusieurs sens précis qui se ressemblent et entrent en conflit :

- *Résultats égaux pour les groupes* : la même part de chaque groupe obtient le prêt.
- *Erreurs égales pour les groupes* : les personnes qualifiées de chaque groupe ont autant de chances d'être approuvées, et les personnes non qualifiées autant de chances d'être approuvées à tort.
- *Même signification d'un score* : quand le système dit « 70 % de risque », il a raison 70 % du temps dans chaque groupe.
- *Personnes semblables, traitement semblable* (équité individuelle).
- *Même décision si seul l'attribut protégé avait été différent* (équité contrefactuelle).

Quand deux groupes diffèrent par la fréquence réelle de l'événement, on ne peut pas satisfaire les trois premiers ensemble sauf dans des cas dégénérés. C'est un fait mathématique, pas un échec de modélisation, donc choisir un critère est une décision de politique qui doit être prise explicitement.

**Vie privée.** Publier une statistique sur un jeu de données révèle quelque chose sur chaque personne qui s'y trouve. La confidentialité différentielle promet que la sortie aurait presque le même aspect que l'enregistrement d'une personne donnée soit présent ou non, en ajoutant un bruit aléatoire soigneusement calibré. Le prix est la précision ; la garantie est mathématique, pas une question de confiance.

## Biais et préjudice de représentation

D'après la leçon source (un résumé fondé sur une revue de littérature ; nous n'avons pas ouvert les articles sous-jacents) :

- **Familles de métriques.** Les tests *fondés sur les embeddings* mesurent des associations entre des mots d'identité et des mots d'attributs dans les représentations ; ceux *fondés sur les probabilités* comparent la vraisemblance de complétions conformes ou non aux stéréotypes ; ceux *fondés sur le texte généré* évaluent les sorties en aval (comme la notation de CV). Le dernier est le plus réaliste et le plus difficile à reproduire.
- **Intersectionnalité.** Un biais le long d'un seul axe (disons le genre) peut cacher un biais qui n'apparaît que pour des combinaisons (genre et origine ensemble). La leçon rapporte une étude de 2025 sur la notation de CV dans laquelle une combinaison était plus pénalisée que chaque facteur seul, et un benchmark qui mesure les différences dans l'*incertitude* du modèle, pas seulement ses prédictions ponctuelles, entre groupes intersectionnels.
- **Approches mécanistes** (rapportées) : localiser des neurones, des caractéristiques d'autoencodeur parcimonieux ou des têtes d'attention corrélés à un comportement sensible à l'identité et les ablater ou les repondérer, avec un coût de capacité limité dans les études citées.
- **Une méta-critique** (rapportée) : la littérature se concentre trop sur le genre binaire, donc un modèle peut sembler débiaisé sur l'axe que tout le monde teste tout en restant biaisé sur des axes que personne ne vérifie (handicap, religion, statut migratoire, langue).

## Critères d'équité de groupe

Soit $A$ le groupe, $Y\in\{0,1\}$ le vrai résultat, $\hat Y\in\{0,1\}$ la décision, et $S$ un score dont $\hat Y$ est obtenu par seuillage.

| Critère | Exigence |
|---|---|
| Parité démographique | $P(\hat Y=1\mid A=a)$ égale entre les groupes |
| Cotes égalisées (equalised odds) | $P(\hat Y=1\mid Y=y,A=a)$ égale entre les groupes pour $y=0$ et $y=1$ (FPR et TPR égaux) |
| Parité prédictive / exactitude d'usage conditionnelle | $P(Y=y\mid \hat Y=y,A=a)$ égale entre les groupes (précision, PPV égale) |
| Calibration au sein des groupes | $P(Y=1\mid S=s,A=a)=s$ pour chaque groupe |

Les cotes égalisées expriment la capacité du système à traiter de la même façon les personnes *qualifiées* et *non qualifiées* entre groupes ; la parité démographique ignore complètement $Y$ et peut forcer des taux d'erreur différents quand les taux de base diffèrent.

### Pourquoi ils entrent en conflit (dérivation)

Fixe un groupe de taux de base $p=P(Y=1)$, de taux de vrais positifs $\mathrm{TPR}$ et de taux de faux positifs $\mathrm{FPR}$. Par la règle de Bayes, la précision est

$$
\mathrm{PPV}=P(Y=1\mid\hat Y=1)=\frac{\mathrm{TPR}\cdot p}{\mathrm{TPR}\cdot p+\mathrm{FPR}\cdot(1-p)} .
$$

Supposons que deux groupes aient le même TPR et le même FPR (cotes égalisées), mais des taux de base différents $p_1\ne p_2$. L'expression ci-dessus est strictement croissante en $p$ dès que $\mathrm{FPR}>0$ et $\mathrm{TPR}>0$, donc les PPV doivent différer. Par conséquent les cotes égalisées et la précision égale ne peuvent pas tenir ensemble, sauf si le classifieur est parfait ($\mathrm{FPR}=0$ ou $\mathrm{TPR}=1$ avec $\mathrm{FPR}=0$) ou si les taux de base coïncident. C'est le contenu des résultats d'impossibilité (Chouldechova ; Kleinberg, Mullainathan et Raghavan, tels que cités par la source). Un score calibré au sein de chaque groupe, avec des TPR et FPR égaux, est de même impossible sous des taux de base différents.

### Équité individuelle et contrefactuelle

- **Équité individuelle** (Dwork et al., selon la source) : une application de décision $f$ est équitable si $|f(x)-f(x')|\le L\,d(x,x')$ pour une métrique de similarité propre à la tâche $d$. Le contenu statistique est une condition de Lipschitz ; la partie difficile est la politique, parce que quelqu'un doit définir « semblable ».
- **Équité contrefactuelle** (Kusner et al.) : une décision est équitable envers un individu si elle n'aurait pas changé si l'attribut protégé avait été différent dans un modèle causal du monde. Elle exige un graphe causal, qui est une hypothèse de modélisation ; la leçon rapporte un compromis théorique de précision pour l'imposer.
- **Contrefactuels rétroactifs** (rapportés, janvier 2024) : plutôt que d'intervenir sur l'attribut protégé (juridiquement délicat), demander quels autres changements dans les caractéristiques de la personne auraient produit le résultat contrefactuel.
- La source note qu'avec un graphe causal certains critères de groupe impliquent l'équité contrefactuelle, donc les familles sont moins opposées qu'il n'y paraît, mais le résultat d'impossibilité sous des taux de base inégaux demeure.

## Confidentialité différentielle

### Définition

Un algorithme aléatoire $M$ est $(\varepsilon,\delta)$-différentiellement privé si, pour tous jeux de données $D,D'$ qui diffèrent par l'enregistrement d'une personne, et tous ensembles de sorties $S$,

$$
P(M(D)\in S)\le e^{\varepsilon}\,P(M(D')\in S)+\delta .
$$

Un $\varepsilon$ plus petit signifie que les distributions de sortie sont plus proches, donc une meilleure confidentialité ; $\delta$ est une petite probabilité avec laquelle la garantie peut échouer. Il n'existe pas de $\varepsilon$ « sûr » universel : la source affirme clairement que les valeurs publiées varient selon le modèle de menace et l'objectif d'utilité.

### Le mécanisme de Laplace

Pour une requête numérique $f$ dont la valeur change d'au plus $\Delta$ (sa **sensibilité**) quand un enregistrement change, publie $f(D)+\text{Lap}(\Delta/\varepsilon)$ où le bruit de Laplace d'échelle $b$ a pour densité $\frac{1}{2b}e^{-|x|/b}$.

**Pourquoi la $\varepsilon$-DP tient.** Pour toute sortie $x$, le rapport de densités est

$$
\frac{p_D(x)}{p_{D'}(x)}=\exp\!\Big(\frac{|x-f(D')|-|x-f(D)|}{b}\Big)\le\exp\!\Big(\frac{|f(D)-f(D')|}{b}\Big)\le e^{\Delta/b}=e^{\varepsilon}.
$$

La première inégalité est l'inégalité triangulaire, la deuxième utilise la définition de la sensibilité, et la dernière fixe $b=\Delta/\varepsilon$. Intégrer le rapport de densités sur n'importe quel ensemble $S$ donne la garantie avec $\delta=0$.

### Le mécanisme gaussien

Publie $f(D)+\mathcal{N}(0,\sigma^2)$. La **perte de confidentialité** à la sortie $x$ est $L(x)=\log\frac{p_D(x)}{p_{D'}(x)}$. Avec $f(D')=f(D)+\Delta$ et $x=f(D)+\sigma Z$ pour $Z\sim\mathcal N(0,1)$ :

$$
L=\frac{(x-f(D)-\Delta)^2-(x-f(D))^2}{2\sigma^2}=\frac{\Delta^2-2\Delta\sigma Z}{2\sigma^2}=\frac{\Delta^2}{2\sigma^2}-\frac{\Delta}{\sigma}Z,
$$

donc $L\sim\mathcal N\big(\tfrac{\Delta^2}{2\sigma^2},\tfrac{\Delta^2}{\sigma^2}\big)$. Le mécanisme est $(\varepsilon,\delta)$-DP quand $P(L>\varepsilon)\le\delta$. La calibration suffisante classique (un résultat standard que nous énonçons sans preuve) est

$$
\sigma\ge\frac{\Delta\sqrt{2\ln(1.25/\delta)}}{\varepsilon}\qquad(\varepsilon<1).
$$

Elle est conservatrice : pour $\Delta=1,\varepsilon=0.9,\delta=0.01$ elle donne $\sigma=3.453$, où la perte a pour moyenne $0.042$ et pour écart-type $0.290$, donc $P(L>0.9)=P(Z>2.96)\approx0.0015$, bien en dessous de $\delta$.

### DP-SGD

Pour entraîner un modèle avec une garantie DP, la recette standard (Abadi et al., telle que la décrit la source) change la façon dont les gradients sont utilisés :

1. Échantillonner un mini-batch ; calculer les gradients **par exemple** $g_i\in\mathbb R^d$.
2. **Écrêter** chacun à une norme L2 d'au plus $C$ : $\bar g_i=g_i\cdot\min(1,C/\|g_i\|)$. Maintenant chaque exemple peut déplacer la *somme* d'au plus $C$ : la sensibilité est bornée par construction.
3. Ajouter du bruit gaussien : $\tilde g=\sum_i\bar g_i+\mathcal N(0,\sigma^2C^2I_d)$, puis diviser par la taille du batch.
4. Faire le pas de l'optimiseur avec $\tilde g$.

Le $(\varepsilon,\delta)$ total sur de nombreux pas est suivi par un *comptable de confidentialité* (la source nomme les comptables des moments et de Rényi), qui utilise l'amplification de confidentialité par sous-échantillonnage aléatoire des batchs et une composition plus fine que la simple addition des $\varepsilon$ par pas (la composition de base les additionne et est lâche). L'exercice implémente l'écrêtage, la somme bruitée et la composition de base, pas un comptable. En pratique, pour les modèles de langage, la source rapporte que la configuration courante de 2025 consiste à appliquer DP-SGD aux seuls paramètres des adaptateurs de bas rang (LoRA), de sorte que la garantie couvre l'adaptateur tandis que le modèle de base est figé.

### Ce que la garantie couvre, et la tension dans les preuves

La source rapporte deux corpus de preuves qui semblent contradictoires : les études d'inférence d'appartenance fondées sur des canaris rapportent un succès d'attaque limité contre les modèles de langage, tandis que les études d'extraction de données d'entraînement retrouvent une mémorisation mot à mot substantielle. Une analyse de 2025 les a réconciliées : elles mesurent des choses différentes, et ce sont les exemples *les plus extractibles* qui portent le risque de confidentialité, que les canaris insérés au hasard sous-représentent. Des alternatives à l'entraînement DP ont aussi été décrites : la prédiction privée à l'inférence en agrégeant des experts entraînés sur des fragments (PMixED), et les données synthétiques DP. Une attaque émergente utilise les *scores de confiance* d'un modèle entraîné en DP comme oracle, donc un déploiement ne devrait pas exposer les confiances brutes. Traite tout cela comme des résultats rapportés.

## Exemple détaillé : équité

Données synthétiques (20 000 personnes, deux groupes), score $S$ calibré par construction dans les deux groupes (étiquette tirée avec la probabilité $S$), le groupe 1 ayant des scores plus élevés. Taux de base : 0.286 et 0.500.

À un seuil commun de 0.5 :

| | groupe 0 | groupe 1 |
|---|---|---|
| taux de sélection | 0.109 | 0.493 |
| TPR | 0.228 | 0.629 |
| FPR | 0.061 | 0.357 |
| PPV | 0.598 | 0.638 |
| erreur de calibration (10 intervalles) | 0.009 | 0.014 |

Écart de parité démographique $0.384$ ; écart de cotes égalisées $0.401$ (l'écart de TPR), bien que le score soit calibré dans les deux groupes (ECE inférieure à 0.02). Abaisse ensuite le seuil du groupe 0 à $0.312$ pour que son TPR égale celui du groupe 1 ($0.629$). Maintenant le TPR est égal, mais le PPV est de $0.455$ contre $0.638$ et le FPR de $0.302$ contre $0.357$. Si les deux groupes avaient exactement un TPR de $0.629$ et un FPR de $0.357$, la règle de Bayes imposerait un PPV de $0.414$ pour le groupe 0 contre $0.638$ pour le groupe 1 : les nombres se comportent comme la dérivation le prédit.

## Exemple détaillé : vie privée

Requête de comptage, sensibilité 1, comptes voisins 100 et 101.

- Gaussien : $\varepsilon=0.9,\delta=10^{-2}\Rightarrow\sigma=3.453$ ; $\varepsilon=0.5,\delta=10^{-5}\Rightarrow\sigma=9.690$ (un $\varepsilon$ et un $\delta$ plus petits coûtent beaucoup plus de bruit).
- Vérification empirique avec 200 000 échantillons par jeu de données : pour les événements $\{x>t\}$, le plus grand excès de $P(M(D'))-e^\varepsilon P(M(D))-\delta$ est nul à l'erreur d'échantillonnage près (environ 0.001), pour le gaussien calibré et pour Laplace avec $\varepsilon=1$. Avec un quart du bruit, l'excès dépasse 0.1 dans les deux cas (gaussien, et Laplace avec $\varepsilon=1$ annoncé mais une échelle de 0.25 ; dans une exécution environ 0.19 et 0.75) : le test empirique détecte une affirmation invalide.
- Écrêtage DP-SGD : avec des gradients par exemple mis à l'échelle à une norme de 3 et $C=1$, chaque ligne écrêtée a une norme d'au plus 1, et retirer n'importe quel exemple change la somme écrêtée d'au plus 1.
- La composition de base de $(0.5,10^{-6})$, $(0.25,10^{-6})$, $(0.25,0)$ donne $(1.0,\,2\times10^{-6})$.

Une vérification empirique ne peut que *réfuter* une affirmation de confidentialité pour les événements testés ; elle ne peut jamais en prouver une. La preuve est la dérivation.

## Implémentation

```python
import math
import numpy as np

def gaussian_sigma(sensitivity, eps, delta):
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps

def clip_rows(G, C):
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    return G * np.minimum(1.0, C / np.maximum(norms, 1e-12))

def dp_sgd_sum(G, C, sigma, rng):
    return clip_rows(G, C).sum(axis=0) + sigma * C * rng.normal(size=G.shape[1])
```

En production, des bibliothèques comme Opacus (PyTorch) ou TensorFlow Privacy calculent efficacement les gradients par exemple et incluent les comptables.

## Dimensions

- Gradients par exemple `G` : `(B, d)` ; écrêtés `(B, d)` ; agrégat bruité `(d,)`.
- Équité : `g`, `s`, `y`, `pred` : `(n,)` ; les masques sont des booléens `(n,)`.

## Idées reçues

- **« Un score calibré est un score équitable. »** L'exemple est calibré dans les deux groupes et présente pourtant un écart de 0.40 dans les taux d'erreur.
- **« On peut choisir le modèle le plus équitable en optimisant toutes les métriques. »** Sous des taux de base inégaux, les critères sont prouvablement incompatibles ; le choix est une décision de politique, prise en pensant aux personnes concernées.
- **« Retirer l'attribut protégé retire le biais. »** D'autres variables lui sont corrélées ; les métriques ci-dessus sont calculées à partir des résultats et ont besoin de l'attribut.
- **« DP signifie que les données sont anonymisées. »** La DP est une propriété de l'*algorithme*, bornant ce qu'une sortie révèle d'un enregistrement ; des valeurs élevées de $\varepsilon$ offrent peu de protection.
- **« Un test empirique qui ne trouve aucune violation prouve la confidentialité. »** Il échoue seulement à réfuter l'affirmation pour les événements testés.
- **« L'écrêtage ne fait que stabiliser l'entraînement. »** Dans DP-SGD, c'est ce qui borne l'influence de chaque personne, de sorte que le bruit peut être calibré.

## Points clés à retenir

- Distingue le préjudice de représentation du préjudice d'allocation et teste les deux, y compris aux intersections.
- La parité démographique, les cotes égalisées et la précision/calibration égales entrent en conflit quand les taux de base diffèrent ; dérive-le à partir de $\mathrm{PPV}=\mathrm{TPR}\,p/(\mathrm{TPR}\,p+\mathrm{FPR}(1-p))$.
- L'équité individuelle et contrefactuelle ont besoin respectivement d'une métrique de similarité et d'un modèle causal ; les deux sont des hypothèses.
- Un bruit de Laplace d'échelle $\Delta/\varepsilon$ donne la $\varepsilon$-DP ; un bruit gaussien donne la $(\varepsilon,\delta)$-DP ; l'écrêtage rend connue la sensibilité d'une somme de gradients.
- La comptabilité de confidentialité, la mémorisation hors distribution et les confiances exposées sont les difficultés pratiques.

## Explique-le avec tes mots

1. Explique à un non-spécialiste pourquoi un modèle de prêt peut être « également précis pour tous » et pourtant injuste selon une autre définition.
2. Pourquoi écrêtons-nous les gradients par exemple avant d'ajouter du bruit dans DP-SGD, au lieu d'ajouter seulement du bruit ?

## Pour aller plus loin

- Nommés dans les leçons sources : la revue sur le biais de représentation (arXiv:2309.00770), l'article de Dwork et al. sur l'équité individuelle, Kusner et al. sur l'équité contrefactuelle, les articles d'impossibilité de Chouldechova et de Kleinberg, Mullainathan et Raghavan, Abadi et al. sur DP-SGD, et l'analyse de 2025 réconciliant MIA et extraction (arXiv:2503.06808). Seuls les résumés du dépôt ont été lus.

## Critères de maîtrise

- Calculer la parité démographique, les cotes égalisées, le PPV et l'ECE pour un ensemble de prédictions.
- Dériver l'expression du PPV et l'utiliser pour montrer l'impossibilité sous des taux de base inégaux.
- Prouver la $\varepsilon$-DP du mécanisme de Laplace et dériver la distribution de perte de confidentialité gaussienne.
- Décrire DP-SGD et ce que l'écrêtage et le bruit apportent chacun.
