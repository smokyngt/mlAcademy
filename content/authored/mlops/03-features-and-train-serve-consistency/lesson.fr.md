---
id: ops.03
title: "Pipelines de features, fuite de données et cohérence entraînement/serving"
summary: "Construire des features dont les versions à l'entraînement et au serving concordent, reconnaître les schémas de fuite qui s'infiltrent dans les pipelines, et écrire un vérificateur de décalage (skew) qui compare automatiquement les deux mondes."
minutes: 65
teaches: [ops.feature-pipelines, ops.skew]
requires: [ml.features, ml.leakage, ops.sampling]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.5 Feature Engineering: learned versus engineered features, scaling, discretization, categorical encoding and the hashing trick, missing values, feature crossing (PDF p.157-172, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.5: Data Leakage (causes and detection) and Engineering Good Features (importance, generalization, coverage) (PDF p.172-178). READ."
---

## Où cela s'insère

[[cml.11]] a enseigné ce que sont les features et à quoi ressemble une fuite sur le jeu de données d'un seul notebook. La production change le problème. La *même* feature est maintenant calculée deux fois : une fois par un job hors ligne qui construit le jeu d'entraînement, et de nouveau par un service qui doit répondre à une requête en quelques millisecondes. Si les deux calculs diffèrent ne serait-ce qu'un peu, le modèle qui paraissait excellent hors ligne se comporte différemment en ligne, et rien ne plante. Cette leçon porte sur le pipeline : des transformations cohérentes, des séparations sûres et des contrôles automatisés. Elle utilise la distinction batch/stream de [[ops.02]] et prépare le monitoring de [[ops.06]], où exactement les mêmes contrôles tournent en continu sur le trafic réel.

## L'idée en termes simples

Une fiche de recette dit « ajoutez une pincée de sel ». Dans la cuisine à la maison (entraînement), tu utilises tes doigts ; au restaurant (serving), le nouveau cuisinier utilise une cuillère doseuse et une pincée plus petite. Chaque cuisinier a suivi la fiche, pourtant le plat n'a pas le même goût. Le modèle est le convive qui s'est habitué à la version maison.

La fuite est un échec différent : le cuisinier à la maison a jeté un œil au verdict final des convives pendant qu'il cuisinait. Le plat obtient une note brillante à la dégustation à la maison et est médiocre au restaurant, où personne ne peut te donner le verdict à l'avance.

Les deux échecs ont un seul remède : écrire la recette une seule fois, utiliser le même code dans les deux cuisines, ne jamais regarder d'information qui n'existera pas au moment du serving, et peser le plat aux deux endroits pour voir s'ils concordent.

## Features apprises contre features conçues

Les réseaux profonds peuvent apprendre des features à partir de pixels, d'audio ou de texte bruts, ce qui supprime beaucoup de travail manuel. Mais la plupart du ML en production n'est pas du deep learning sur des signaux bruts, et même les modèles profonds reçoivent des entrées conçues (historique utilisateur, comptages, ratios). Un classifieur de texte classique nécessitait de mettre en minuscules, de raciniser et de compter les mots à la main ; un classifieur moderne apprend sa représentation, mais consomme encore des choses comme « nombre d'achats dans les 7 derniers jours » que quelqu'un doit définir. Le feature engineering reste l'une des activités à plus fort effet de levier : une fois qu'un modèle viable existe, de meilleures features battent souvent des algorithmes plus astucieux.

## Opérations courantes, avec le piège de production

| Opération | Ce qu'elle fait | Piège dans le pipeline |
|---|---|---|
| Valeurs manquantes | Imputer ou supprimer. Les valeurs peuvent manquer de façon non aléatoire (la valeur elle-même a causé le trou), au hasard (une autre variable l'explique) ou complètement au hasard (rare). | Les valeurs de remplissage doivent provenir de la partition d'entraînement uniquement. L'absence elle-même peut être informative et mériter une feature indicatrice. |
| Mise à l'échelle | Ramener une feature à une plage comparable, p. ex. standardiser $(x-\mu)/\sigma$ ou min-max. | $\mu,\sigma$ sont des statistiques globales : les ajuster sur l'entraînement, les stocker, les réutiliser au serving. Si les nouvelles données dérivent, les statistiques stockées deviennent périmées. |
| Discrétisation | Regrouper une valeur continue en tranches. | Crée des falaises aux bords des tranches ; choisir les bornes à partir des quantiles d'entraînement et les garder fixes. |
| Encodage catégoriel | Associer les catégories à des entiers ou des vecteurs. | Les catégories changent en production (de nouvelles marques apparaissent chaque jour). Une tranche `UNKNOWN` évite les plantages mais un modèle jamais entraîné dessus l'ignorera. Le **hashing trick** attribue à chaque catégorie le hash de son nom modulo un espace fixe (par exemple $2^{18}$ emplacements), si bien que les catégories jamais vues sont gérées sans vocabulaire ; les collisions sont aléatoires et, selon l'étude citée par le livre, même 50 % de features en collision coûtent moins de 0,5 % de log loss. |
| Croisement de features | Combiner des features (statut marital x enfants) pour donner un pouvoir non linéaire à des modèles simples. | Le croisement multiplie la cardinalité et la parcimonie. |

## Fuite de données

Une **fuite** (leakage) signifie que des informations sur l'étiquette (ou sur le futur) sont présentes dans les features à l'entraînement alors qu'elles ne seront pas disponibles au moment du serving. Elle est dangereuse parce que l'évaluation a l'air excellente. Les causes cataloguées par le livre :

1. **Séparer aléatoirement des données corrélées dans le temps.** Les lignes d'un même jour partagent des tendances. Une séparation aléatoire met des lignes du jour 7 dans l'entraînement et des lignes du jour 7 dans le test. Sépare par le temps : entraîne sur les quatre premières semaines, teste sur la semaine cinq.
2. **Mettre à l'échelle (ou ajuster quoi que ce soit) avant de séparer.** Calculer des moyennes ou des variances sur tout le jeu de données laisse les statistiques du test entrer dans l'entraînement. Sépare d'abord, ajuste sur l'entraînement, transforme tout avec ces statistiques. Il en va de même pour les valeurs d'imputation, les vocabulaires, l'ACP et les encodages de la cible.
3. **Remplir les valeurs manquantes avec des statistiques de la partition de test**, même mécanisme.
4. **Doublons et quasi-doublons entre les partitions.** Typique après la fusion de sources ou après un suréchantillonnage ; le livre rapporte que 3,3 % des images de test de CIFAR-10 et 10 % de celles de CIFAR-100 ont des doublons dans le jeu d'entraînement. Déduplique avant de séparer, revérifie après, et ne suréchantillonne qu'après la séparation.
5. **Fuite de groupe.** Les lignes qui partagent une entité déterminant l'étiquette (deux scanners d'un même patient, des images d'un même objet) doivent être dans la même partition.
6. **Fuite issue du processus de génération des données.** Le modèle apprend l'artefact de la manière dont les données ont été collectées (l'exemple du livre : un type de scanner corrélé au fait que les médecins soupçonnent un cancer), qui n'existe pas dans le second hôpital. Seule la connaissance de la manière dont les données sont collectées permet de le trouver ; normalise entre les sources et implique des experts du domaine.

**Détecter une fuite.** Mesure le pouvoir prédictif de chaque feature et examine celles qui sont suspectement fortes (deux features peuvent ne fuir qu'ensemble, comme date de début et date de fin pour l'ancienneté). Fais des études d'ablation : retire une feature et regarde de combien la performance baisse. Traite avec méfiance toute nouvelle feature qui fait soudainement monter la métrique. Et ne regarde le jeu de test que pour rapporter le nombre final : l'utiliser pour concevoir des features ou régler des hyperparamètres le fait fuir.

## Concevoir de bonnes features

Davantage de features n'est pas automatiquement mieux : plus d'occasions de fuite, plus de surapprentissage, plus de mémoire et de latence au serving, et chaque feature inutilisée est une dette technique à maintenir chaque fois que les données en amont changent. Juge une feature selon deux axes.

- **Importance** : de combien la performance baisse-t-elle sans elle (importance des features, valeurs SHAP, permutation ou ablation) ? Souvent, quelques features portent l'essentiel de l'importance.
- **Généralisation** : se comportera-t-elle de la même façon sur des données jamais vues ? Vérifie la **couverture** (la part des lignes qui ont une valeur ; une feature présente dans 1 % des lignes est rarement utile, à moins que l'absence soit elle-même informative) et la **distribution des valeurs**. Si la couverture est de 90 % à l'entraînement et de 20 % au test, les partitions diffèrent. Si les valeurs vues à l'entraînement ne recouvrent pas celles du moment du test (l'exemple du livre : `DAY_OF_THE_WEEK` a lundi-samedi à l'entraînement et dimanche au moment du test), la feature peut nuire.

## Feature stores et code partagé

Un **feature store** est une infrastructure qui définit chaque feature une seule fois et la sert à la fois aux jobs d'entraînement hors ligne (valeurs historiques à la date de chaque exemple d'entraînement) et aux services en ligne (valeurs courantes à faible latence). Le livre reporte les feature stores à son chapitre sur l'infrastructure, qui ne fait pas partie de la lecture de cette leçon ; le point à retenir ici est le principe qu'ils implémentent : **une définition, deux consommateurs**. Sans store, la même discipline peut être obtenue en partageant une bibliothèque de transformations entre le job d'entraînement et le service, en la versionnant avec le modèle, et en testant que les deux chemins donnent les mêmes sorties pour la même entrée brute.

## Décalage entraînement/serving et comment le vérifier

Le **skew** est toute différence systématique entre les features que le modèle a vues à l'entraînement et celles qu'il reçoit au serving. Causes : les deux pipelines sont du code séparé (job batch Python contre un service dans un autre langage), des unités différentes (âge en années contre en mois), des statistiques périmées, des données amont manquantes, de nouvelles catégories, et des sémantiques temporelles différentes (une feature calculée « à l'instant » au serving mais « à la date de l'étiquette » à l'entraînement).

Un **vérificateur de skew** pratique compare une matrice d'entraînement avec un échantillon de features servies, feature par feature :

- **missing** : le taux de NaN diffère de plus d'une tolérance.
- **shift** : la moyenne bouge de plus de `mean_z` écarts-types d'entraînement, $|\bar x_{\text{serve}}-\bar x_{\text{train}}|/\sigma_{\text{train}}>z$.
- **range** : plus d'une tolérance de valeurs servies tombent hors du minimum et du maximum d'entraînement.
- pour les catégorielles, le **taux de catégories inconnues**, la part des valeurs servies jamais vues à l'entraînement.

Ces contrôles n'ont besoin d'aucune étiquette, ils peuvent donc tourner sur le trafic réel. Ils sont peu coûteux, rudimentaires et efficaces. Ils ne prouvent pas l'absence de skew (deux distributions peuvent partager une moyenne et différer par la forme) ; [[ops.06]] ajoute le test KS et le PSI pour cela.

## Exemple détaillé

Tous les nombres viennent de l'exercice. Entraîne sur 2 000 lignes de trois features : `age ~ N(30, 5)`, `income ~ N(100, 20)`, `z ~ N(0, 1)`. Prends 1 500 lignes de serving propres issues des mêmes distributions : le vérificateur ne signale aucun problème. Corromps maintenant les données de serving de trois façons :

- l'âge multiplié par 12 (mois au lieu d'années) place la moyenne de serving à environ 66 écarts-types d'entraînement (360 contre 30, avec un écart-type de 5) et presque toutes les valeurs au-dessus du maximum d'entraînement, donc les problèmes sont `["range", "shift"]` ;
- 300 des 1 500 valeurs de revenu remplacées par NaN (20 % manquantes, contre 0 % à l'entraînement) donne `["missing"]` seulement : la moyenne des valeurs non manquantes est inchangée ;
- `z` décalé de +3 donne `["range", "shift"]`.

Le vérificateur retourne `{"age": ["range","shift"], "income": ["missing"], "z": ["range","shift"]}`. Dans le même exercice, une séparation temporelle de dix lignes avec les horodatages 5,1,4,2,3,10,9,8,7,6 et `test_frac=0.3` envoie les horodatages 8, 9, 10 au test et 1 à 7 à l'entraînement, de sorte que chaque ligne d'entraînement est antérieure à chaque ligne de test.

## Implémentation

```python
import numpy as np

def fit_scaler(X):
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0)     # TRAIN split only

def apply_scaler(X, mean, std):
    return (X - mean) / std                                 # reuse at serving

def time_split(timestamps, test_frac):
    order = np.argsort(timestamps, kind="stable")
    n_train = int(round(len(order) * (1 - test_frac)))
    return order[:n_train], order[n_train:]
```

Bibliothèques : les objets `Pipeline` de scikit-learn regroupent les transformateurs ajustés avec le modèle pour que les statistiques ajustées voyagent ensemble ; `TimeSeriesSplit` pour les séparations ordonnées dans le temps ; `GroupKFold` pour la fuite de groupe ; des feature stores comme Feast ; des outils de validation de données qui comparent les statistiques d'entraînement et de serving.

## Dimensions

`X: (n, d)` features ; `mean, std: (d,)` ; un rapport de skew associe un nom de feature à une liste de problèmes ; `time_split` retourne deux vecteurs d'indices dans les `n` lignes d'origine.

## Idées reçues

- « Normalise tout le jeu de données une fois, puis sépare. » Cela fait fuir les statistiques du test. Ajuste sur l'entraînement.
- « Si la métrique hors ligne est élevée, le pipeline est correct. » La fuite rend les métriques hors ligne trop bonnes ; le skew rend le en-ligne moins bon que le hors-ligne.
- « Une tranche de catégorie inconnue résout les nouvelles catégories. » Elle évite le plantage, mais un modèle qui n'a jamais vu la tranche à l'entraînement l'ignore.
- « Même moyenne signifie pas de skew. » Les moyennes cachent les changements de variance, de forme et de plage.
- « Les feature stores suppriment le besoin de tests. » Ils aident, mais les deux chemins de code doivent quand même être vérifiés pour leur concordance.

## Points clés à retenir

- Définis chaque transformation une seule fois et réutilise les paramètres ajustés au serving.
- Sépare par le temps pour les données corrélées dans le temps ; ajuste les scalers, imputeurs et encodeurs sur la partition d'entraînement uniquement ; déduplique avant de séparer ; garde les groupes ensemble.
- Méfie-toi de toute feature trop bonne ; fais des ablations ; ne touche au jeu de test qu'une fois.
- Juge les features sur l'importance et la généralisation (couverture et distribution des valeurs).
- Un vérificateur de skew sans étiquettes (taux de manquants, décalage de moyenne, violations de plage, catégories inconnues) détecte les décalages entraînement/serving les plus courants.

## Explique-le avec tes mots

1. Donne trois façons différentes dont un jeu de test peut fuir dans l'entraînement via le prétraitement, et le remède pour chacune.
2. Pourquoi une catégorie `UNKNOWN` peut-elle échouer silencieusement, et en quoi le hachage change-t-il cela ?

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, Ch.5 (PDF p.157-179) et Ch.10 pour les feature stores (non lu pour cette leçon).
- [[cml.11]] pour le traitement au niveau notebook des features, du déséquilibre et de la fuite.

## Critères de maîtrise

Tu sais énumérer les six schémas de fuite avec un remède pour chacun, expliquer pourquoi toute statistique ajustée doit provenir de la partition d'entraînement, concevoir les chemins hors ligne/en ligne d'une feature pour qu'ils partagent du code, et implémenter et interpréter un vérificateur de skew.
