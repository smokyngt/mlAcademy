---
id: ops.01
title: "Cadrer un problème de ML et concevoir des systèmes de ML"
summary: "Transformer un souhait métier vague en un problème de ML avec des entrées, des sorties et un objectif, et raisonner sur les exigences (percentiles de latence, passage à l'échelle, maintenance) qu'un système en production doit satisfaire au-delà de la précision du modèle."
minutes: 55
teaches: [ops.framing, ops.requirements]
requires: [ml.generalization, ml.eval-metrics, prob.rv]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.1 'Understanding Machine Learning Systems' - research versus production, stakeholders, latency versus throughput, percentiles (PDF p.32-41). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.2 Business and ML Objectives, Requirements for ML Systems, Framing ML Problems (task types, multiple framings, decoupling objectives), Mind Versus Data (PDF p.51-75, read in selected parts). READ."
---

## Où cela s'insère

Tous les cours précédents t'ont appris à ajuster un modèle une fois que quelqu'un te tend une table propre `X` et une cible `y`. Dans une vraie entreprise, personne ne te tend cela. Quelqu'un dit « notre support client est trop lent » ou « peut-on utiliser l'IA pour augmenter les ventes ? », et ton premier travail est de décider **ce que le modèle doit prédire, à partir de quoi, et comment le succès sera jugé**. Tout ce qui suit dans ce cours (données, features, évaluation, déploiement, monitoring) découle de cette première décision. Cette leçon introduit aussi le vocabulaire des exigences (latence, débit, passage à l'échelle) que [[ops.05]] et [[ops.06]] utilisent en permanence.

## L'idée en termes simples

Pense à un restaurant. Un cuisinier (le data scientist) peut s'acharner à rendre un plat 1 % meilleur. Mais le restaurant a aussi besoin que la nourriture arrive chaude (latence), qu'il survive au coup de feu du samedi (passage à l'échelle), que l'équipe de nuit puisse la cuisiner (maintenabilité), et qu'il suive l'évolution des goûts des clients (adaptabilité). Un plat plus savoureux qui arrive froid et qui demande trois chefs pour être reproduit fait un moins bon restaurant. De même, la patronne ne se soucie pas d'un « score de goût de 94,2 au lieu de 94,0 » : elle se soucie du chiffre d'affaires, et le cuisinier doit montrer comment le goût fait bouger ce chiffre d'affaires.

Cadrer un problème est l'étape de traduction : « les clients attendent trop longtemps » n'est pas encore un problème de ML. Cela le devient quand tu découvres que le vrai retard vient de l'aiguillage de chaque demande vers l'un de quatre services, ce qui te permet de dire : *entrée = le texte de la demande, sortie = l'un des quatre services, objectif = faire coïncider le service prédit avec le bon*. Un problème de ML, ce sont exactement ces trois éléments : **des entrées, des sorties et la fonction objectif**.

## Objectifs métier contre objectifs de ML

Les métriques de ML (précision, F1, perplexité, latence d'inférence) sont ce que tu peux mesurer sur un modèle. Les métriques métier (chiffre d'affaires, rétention, coûts économisés, heures visionnées) sont ce qui importe à l'organisation. Un modèle dont l'accuracy passe de 94 % à 94,2 % n'a d'importance que si une métrique métier bouge. Deux conséquences pratiques :

- Choisis, avec les parties prenantes, la métrique métier que le projet est censé influencer, et trouve un moyen de la relier à une métrique de ML. Certaines tâches rendent cela facile : chaque clic publicitaire supplémentaire est du revenu, chaque paiement frauduleux bloqué est de l'argent économisé. D'autres non, et l'effet ne peut être mesuré que par une expérience comme un test A/B ([[ops.06]]) ; dans ce cas, tu choisis le modèle qui fait bouger la métrique métier, même si un autre a de meilleures métriques de ML.
- Sois réaliste sur les retours. Les gains du ML arrivent généralement une fois qu'un pipeline mature existe (le livre cite une enquête dans laquelle les entreprises ayant plus d'années de ML en production déploient de nouveaux modèles beaucoup plus vite), pas du jour au lendemain.

**Recherche contre production.** Le livre oppose cinq différences. En recherche, il y a généralement un seul objectif (un score de benchmark) ; en production, il y a de nombreuses parties prenantes aux souhaits contradictoires. En recherche, la vitesse d'entraînement et le débit dominent ; en production, ce sont la vitesse d'inférence et la latence. Les données de recherche sont statiques et propres ; les données de production dérivent et sont désordonnées. L'équité et l'interprétabilité sont optionnelles dans les classements et obligatoires dans de nombreux produits. Un exemple concret : un ensemble de modèles qui gagne une compétition peut être trop lent et trop difficile à maintenir pour être livré.

## Exigences d'un système de ML

Quatre propriétés apparaissent dans la plupart des conceptions.

1. **Fiabilité** : continuer à faire ce qu'il faut malgré les pannes matérielles, les bogues logiciels et les erreurs humaines. Le ML ajoute une subtilité : le système peut **échouer silencieusement**. `model.predict()` retourne sans erreur alors que les prédictions sont fausses, et un utilisateur qui traduit vers une langue qu'il ne lit pas peut ne jamais s'en apercevoir.
2. **Passage à l'échelle** : croissance de la complexité du modèle (un modèle de 1 Go devient 16 Go), du trafic (de 10 mille à 10 millions de requêtes par jour), ou du **nombre de modèles** (un par client : 8 000 clients entreprise signifient 8 000 modèles à surveiller et à réentraîner). Passer à l'échelle inclut de réduire l'échelle pour économiser de l'argent, pas seulement de l'augmenter, et de gérer des artefacts (code, données, versions de modèles), pas seulement des machines.
3. **Maintenabilité** : d'autres personnes, y compris toi dans le futur, doivent pouvoir comprendre, déboguer et étendre le système. Documente, versionne, rends les résultats reproductibles.
4. **Adaptabilité** : les données et les exigences changent, donc le système doit pouvoir être mis à jour rapidement (voir [[ops.06]] pour le décalage de distribution et l'apprentissage continu).

### La latence est une distribution, pas un nombre

La **latence** est le temps entre la réception d'une requête et le renvoi de la réponse ; le **débit** (throughput) est le nombre de requêtes traitées par unité de temps. Si un serveur traite une requête à la fois, le débit vaut `1/latence` : 10 ms par requête font 100 requêtes par seconde. Mais les systèmes réels font du **batching** : traiter `B` requêtes ensemble en `t` ms donne

$$
\text{throughput}=\frac{B\cdot 1000}{t}\ \text{queries/s}.
$$

L'exemple du livre : 10 requêtes par lot de 10 ms font 1 000 qps ; 50 requêtes par lot de 20 ms font 2 500 qps. La latence a doublé **et** le débit a été multiplié par 2,5. Donc « latence plus haute signifie débit plus bas » n'est vrai que sans batching, et le batching a un coût : les requêtes doivent attendre que le lot se remplisse ([[ops.05]] le simule).

Ne résume jamais la latence par sa moyenne. Prends dix requêtes qui ont pris 100, 102, 100, 100, 99, 104, 110, 90, 3000 et 95 ms. La moyenne est de 390 ms, ce qui ne décrit aucune d'elles ; la médiane (**p50**) est de 100 ms ; le 99e percentile (**p99**) est de 3000 ms et révèle l'unique requête terrible. Nous utilisons la définition du **rang le plus proche** (nearest-rank) : trie les `n` valeurs et prends celle de rang 1-indexé `ceil(q/100 * n)`. (Le livre affirme que le p90 de cette liste est de 3000 ms ; avec la définition du rang le plus proche, le p90 est de 110 ms, et les définitions par interpolation donnent environ 399 ms. Les définitions de percentile diffèrent selon les bibliothèques, ce qui est en soi une raison de noter par écrit celle qu'utilise ton SLA. La leçon - regarder la queue, pas la moyenne - reste inchangée.) Les exigences sont donc formulées sur des percentiles, par exemple « médiane sous 100 ms et p99 sous 1 s ». Les percentiles élevés comptent parce que les requêtes lentes viennent souvent des utilisateurs les plus intensifs.

### Le processus itératif

Un projet de ML est une boucle, pas une ligne : cadrer le projet et ses métriques, collecter et préparer les données, développer le modèle, déployer, surveiller et continuer à apprendre, et analyser l'impact métier, qui alimente le cadrage du tour suivant. Chaque boucle t'apprend quelque chose qui modifie les étapes précédentes (de mauvaises prédictions te renvoient aux données). Prévois la boucle et l'itération rapide ; c'est pourquoi le reste du cours consacre autant de temps aux pipelines et au monitoring qu'aux modèles.

## Cadrer un problème

Le **type de tâche** est dicté par la sortie.

- La **classification** produit une catégorie ; la **régression** produit un nombre continu. Elles sont dans une certaine mesure interchangeables : regroupe les prix de maisons en tranches de prix et la régression devient une classification ; applique un seuil de 0,5 à un score de spam et un score de type régression devient une classe.
- La classification **binaire** est la plus facile. Les problèmes **multiclasses** deviennent plus difficiles quand les classes se multiplient (beaucoup de classes avec peu d'exemples chacune). Les problèmes **multilabels** (un article peut avoir plusieurs thèmes) demandent une règle pour transformer des probabilités comme `[0.45, 0.2, 0.02, 0.33]` en un ensemble : prendre le premier ? les deux premiers ? un seuil ? Et les annotateurs sont davantage en désaccord sur le nombre d'étiquettes applicables.
- **Un même problème peut être cadré de plusieurs façons**, et le cadrage change la difficulté. « Quelle application l'utilisateur ouvrira-t-il ensuite ? » en multiclasse a une couche de sortie qui doit changer à chaque nouvelle application installée ; cadré comme une régression « donner un score à chaque paire (utilisateur, application) », le modèle gère n'importe quel ensemble d'applications.

### Plusieurs objectifs : les découpler

Supposons qu'un fil d'actualité doive montrer des publications engageantes mais ni extrêmes ni de mauvaise qualité. Tu as deux pertes, `quality_loss` et `engagement_loss`, et tu veux un compromis. Une option est un modèle unique entraîné sur un mélange :

$$
\text{loss}=\alpha\,\text{quality\_loss}+\beta\,\text{engagement\_loss}.
$$

Changer `alpha` et `beta` implique de réentraîner. L'alternative est de **découpler** : entraîner un modèle par objectif et mélanger les *scores* au moment du classement,

$$
\text{score}(x)=\alpha\,\text{quality}(x)+\beta\,\text{engagement}(x).
$$

Le compromis devient alors un bouton que tu peux tourner sans réentraîner, et chaque modèle peut être mis à jour selon son propre calendrier (les tactiques de spam changent vite, les goûts de qualité lentement). Quand les objectifs sont réellement en conflit, aucun élément n'est le meilleur sur les deux ; les candidats utiles forment le **front de Pareto** : les éléments non **dominés** par un autre élément au moins aussi bon sur chaque score et strictement meilleur sur un. Des poids `alpha, beta >= 0` choisissent alors un point sur ce front.

## L'esprit contre les données

Le progrès est-il porté par une conception astucieuse du modèle (biais inductif, structure) ou par davantage de données et de calcul ? Des chercheurs réputés défendent les deux camps, et le livre les présente côte à côte sans trancher. Pour un praticien, la lecture prudente est pratique : il te faut des données de qualité et de quantité suffisantes avant toute chose (le livre appelle cela le fondement de la « hiérarchie des besoins de la data science »), mais davantage de données n'est pas automatiquement mieux, et la structure t'aide à apprendre plus avec moins. Les prochaines leçons partent des données.

## Exemple détaillé

Le code de l'exercice reproduit ces nombres. Pour les dix latences ci-dessus : moyenne 390 ms, p50 100 ms, p90 au rang le plus proche 110 ms, p99 3000 ms. Pour le batching : `throughput_qps(1,10)=100`, `throughput_qps(10,10)=1000`, `throughput_qps(50,20)=2500`.

Pour le découplage, prends cinq publications avec `quality = [0.9, 0.5, 0.7, 0.2, 0.9]` et `engagement = [0.1, 0.5, 0.3, 0.2, 0.4]`. La publication 0 est dominée par la 4 (même qualité, plus d'engagement) ; la 2 est elle aussi dominée par la 4 ; la 3 est dominée par plusieurs. Le front de Pareto est `{1, 4}`. Classer par la qualité seule donne `[0, 4, 2, 1, 3]` ; par l'engagement seul `[1, 4, 2, 3, 0]` ; avec `alpha=2, beta=1` les scores sont `1.9, 1.5, 1.7, 0.6, 2.2` et l'ordre est `[4, 0, 2, 1, 3]`. Remarque que tourner le bouton n'a jamais touché un modèle.

## Implémentation

```python
import numpy as np

def latency_stats(lat):
    s = np.sort(np.asarray(lat, dtype=float))
    pct = lambda q: float(s[max(int(np.ceil(q / 100 * len(s))), 1) - 1])
    return float(s.mean()), pct(50), pct(90), pct(99)

def blended_ranking(quality, engagement, alpha, beta):
    score = alpha * np.asarray(quality) + beta * np.asarray(engagement)
    return [int(i) for i in np.argsort(-score, kind="stable")]
```

En production, les percentiles de latence viennent d'un système de monitoring (histogrammes dans Prometheus ou un tableau de bord cloud), et le classement multi-objectif d'un service de features/scores ; la logique est la même que ci-dessus.

## Idées reçues

- « Une métrique hors ligne plus élevée signifie un meilleur produit. » Seulement si la métrique est liée à un résultat métier ; sinon tu optimises un proxy.
- « Latence en hausse signifie débit en baisse. » Pas avec le batching.
- « La latence moyenne suffit. » La moyenne cache les valeurs aberrantes et les utilisateurs qui souffrent le plus.
- « Un problème a un seul cadrage correct. » Classification contre régression, un modèle contre des modèles découplés : le cadrage est un choix de conception qui a des conséquences sur les données, les étiquettes et la maintenance.
- « Une fois déployé, c'est terminé. » Le déploiement relance la boucle.

## Points clés à retenir

- Un problème de ML est défini par des entrées, des sorties et une fonction objectif ; les problèmes métier doivent être traduits sous cette forme.
- Relie les métriques de ML aux métriques métier ; utilise des expériences quand le lien n'est pas clair.
- La fiabilité, le passage à l'échelle, la maintenabilité et l'adaptabilité sont des exigences aussi importantes que la précision ; les systèmes de ML peuvent échouer silencieusement.
- La latence est une distribution : utilise les percentiles ; le batching peut augmenter à la fois la latence et le débit.
- Avec plusieurs objectifs, préfère des modèles découplés mélangés au moment du score.

## Explique-le avec tes mots

1. Un collègue dit « notre modèle a 94,2 % de précision, donc le projet est un succès ». Quelles questions poses-tu avant d'être d'accord ?
2. Explique pourquoi le batching peut augmenter la latence et le débit en même temps, et quand cet arbitrage est une mauvaise idée.

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, Ch.1-2 (PDF p.18-78). Pour la terminologie de la latence, Kleppmann, *Designing Data-Intensive Applications* (référencé par le livre, non lu pour cette leçon).

## Critères de maîtrise

Tu sais cadrer une demande vague en entrées/sorties/objectif, nommer les quatre exigences et donner un risque propre au ML pour chacune, calculer des latences en percentiles et un débit par lot, et expliquer pourquoi des objectifs découplés sont plus faciles à maintenir qu'une perte mélangée.
