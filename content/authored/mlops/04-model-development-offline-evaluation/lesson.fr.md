---
id: ops.04
title: "Développement de modèles et évaluation hors ligne"
summary: "Choisir des modèles sans se tromper soi-même, les combiner avec des ensembles, suivre et versionner les expériences, comprendre l'entraînement distribué en un coup d'œil, et évaluer hors ligne avec des baselines, des tests comportementaux et des métriques par tranche."
minutes: 70
teaches: [ops.offline-eval, ops.slice-eval]
requires: [ml.eval-metrics, ml.generalization, ops.skew]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.6 Model Development and Training: six tips for model selection, ensembles (bagging, boosting, stacking), experiment tracking and versioning, distributed training (data and model parallelism) (PDF p.180-200, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.6 Model Offline Evaluation: baselines, perturbation tests, invariance and directional expectation tests, slice-based evaluation, Simpson's paradox (PDF p.201-214). READ."
---

## Où cela s'insère

À ce stade, tu as cadré le problème ([[ops.01]]), construit un jeu d'entraînement ([[ops.02]]) et des features cohérentes ([[ops.03]]). Cette leçon est la partie que la plupart des gens associent au « machine learning » : choisir et entraîner des modèles et les juger. La subtilité de la production est que juger est plus difficile qu'entraîner. Un seul chiffre d'accuracy calculé sur une partition de test dit peu de choses sur la capacité du modèle à survivre au contact des vrais utilisateurs. Tu apprendras à comparer avec des baselines, à sonder le comportement avec des tests de perturbation et d'invariance, et à découper la métrique en tranches pour qu'un modèle bon en moyenne mais mauvais pour un groupe important ne puisse pas se cacher. Le décalage de distribution après déploiement est le sujet de [[ops.06]] ; les métriques d'évaluation elles-mêmes sont traitées dans [[cml.10]].

## L'idée en termes simples

Choisir un modèle ressemble à recruter. Le CV (un score de classement) a été gagné sur les tâches de quelqu'un d'autre. Un processus de recrutement équitable donne à chaque candidat une période d'essai comparable, part d'une référence simple (que réussirait un débutant raisonnable ?), teste la réaction du candidat face à des situations réelles désordonnées, et vérifie la performance pour chaque type de client plutôt que seulement la moyenne générale. Un ensemble est un comité : des avis indépendants qui moyennent les erreurs des uns et des autres, au prix d'une masse salariale plus lourde. Le suivi d'expériences est le cahier de laboratoire qui te permet de répéter un bon résultat ou de comprendre pourquoi un mauvais est survenu.

## Sélection de modèle : six habitudes

1. **Évite le piège de l'état de l'art.** Un modèle qui domine un benchmark sur un jeu de données statique n'est pas nécessairement assez rapide, assez économique ou meilleur sur tes données. Utilise la solution la plus simple si elle résout ton problème.
2. **Commence par le modèle le plus simple.** Les modèles simples se déploient plus tôt (ce qui valide que ton pipeline de serving correspond à l'entraînement, voir [[ops.03]]), sont plus faciles à déboguer, et te donnent une baseline. Le plus simple ne veut pas dire le moins d'effort : un modèle pré-entraîné peut être facile pour démarrer, mais difficile à améliorer.
3. **Évite les biais humains dans les comparaisons.** Si tu essaies 100 configurations de l'architecture que tu aimes et 3 de celle que tu n'aimes pas, la première gagne par l'effort. Donne aux architectures des budgets de réglage comparables, et souviens-toi qu'une affirmation sur une architecture n'est vraie que dans le contexte où elle a été testée.
4. **Compare la bonne performance maintenant et plus tard.** Un modèle à base d'arbres peut gagner avec peu de données tandis qu'un réseau de neurones gagne quand tu les doubles ; une courbe d'apprentissage (performance en fonction de la taille du jeu d'entraînement) donne un indice.
5. **Évalue les compromis**, comme faux positifs contre faux négatifs, et précision contre latence ou interprétabilité.
6. **Comprends les hypothèses du modèle** (par exemple que les features sont indépendantes, ou que les données sont lisses), car elles te disent quand il échouera.

## Ensembles

Un **ensemble** combine plusieurs **apprenants de base**. Pourquoi cela aide-t-il ? Supposons que trois classifieurs soient chacun corrects avec la probabilité $p=0.7$ sur chaque exemple, et que leurs erreurs soient indépendantes. Le vote majoritaire est correct quand au moins deux sont corrects :

$$
P(\text{correct})=\sum_{k=2}^{3}\binom{3}{k}p^k(1-p)^{3-k}=3(0.7)^2(0.3)+(0.7)^3=0.441+0.343=0.784 .
$$

Donc trois modèles à 70 % donnent 78,4 %. Pour $n$ votants indépendants, c'est $\sum_{k>n/2}\binom{n}{k}p^k(1-p)^{n-k}$, qui tend vers 1 quand $n$ croît si $p>0.5$. L'hypothèse qui compte est l'**indépendance** : si les modèles sont toujours d'accord, l'ensemble est exactement aussi bon que l'un d'eux. C'est pourquoi les praticiens combinent différents types de modèles. Trois façons de construire des ensembles :

- **Bagging** (bootstrap aggregating) : entraîner chaque modèle sur un échantillon bootstrap (tiré avec remise), puis voter ou moyenner. Il réduit la variance et aide les apprenants instables comme les arbres profonds. Une forêt aléatoire est du bagging plus un sous-ensemble aléatoire de features à chaque séparation.
- **Boosting** : entraîner des apprenants faibles l'un après l'autre sur les mêmes données, en repondérant les exemples pour que les apprenants suivants se concentrent sur les erreurs précédentes ; les combiner avec des poids favorisant les apprenants précis. Le gradient boosting (XGBoost, LightGBM) est la famille la plus connue.
- **Stacking** : un méta-apprenant (même une simple moyenne, ou une régression logistique) combine les sorties des apprenants de base.

Les ensembles dominent les compétitions mais sont moins courants en production, parce qu'ils coûtent plus cher à servir et à maintenir et sont plus difficiles à interpréter. Ils survivent là où un petit gain vaut beaucoup d'argent, comme la prédiction du taux de clic publicitaire.

## Suivi d'expériences et versionnage

Des modèles qui diffèrent par des taux d'apprentissage de 0,003 contre 0,002 peuvent se comporter très différemment, donc tu dois pouvoir recréer n'importe quelle expérience. Le **suivi d'expériences** enregistre, pour chaque exécution : les courbes de perte pour l'entraînement et chaque partition d'évaluation, les métriques d'évaluation, les prédictions ou échantillons, la vitesse (itérations par seconde) et l'utilisation du système (mémoire, CPU/GPU), et les hyperparamètres. Le **versionnage** enregistre tout ce qu'il faut pour reproduire l'exécution : code, données (ou un instantané des données), configuration et environnement. Des outils comme MLflow et Weights & Biases sont nés du suivi, DVC du versionnage, et ils se recoupent maintenant. Les données sont les plus difficiles à versionner parce qu'elles sont volumineuses ; au minimum, garde des instantanés immuables et des hashes. Un test pratique : un collègue, dans six mois, peut-il reconstruire le modèle que tu as livré et obtenir les mêmes métriques ?

## L'entraînement distribué en un paragraphe

Le **parallélisme de données** donne à chaque worker une copie du modèle et une part de chaque batch, puis accumule leurs gradients. Le SGD synchrone attend tous les workers, donc un seul worker lent (un *straggler*) retarde tout le monde, et la probabilité que l'un d'eux soit lent croît avec le nombre de workers. Le SGD asynchrone met à jour au fur et à mesure que les gradients arrivent et souffre de gradients *périmés* calculés sur d'anciens poids. Le **parallélisme de modèle** répartit le modèle lui-même sur plusieurs machines quand il ne tient pas sur une seule ; naïvement, la machine 2 attend la machine 1, ce que le **parallélisme en pipeline** réduit en découpant chaque batch en parties plus petites qui traversent les étapes comme une chaîne de montage. Les frameworks (PyTorch DistributedDataParallel et similaires) cachent l'essentiel de cela.

## Évaluation hors ligne

### Baselines

Une métrique ne veut rien dire seule. Un modèle avec $F_1=0.90$ sur une tâche où 90 % des étiquettes sont positives peut ne pas faire mieux qu'un tirage aléatoire. Compare toujours avec des baselines :

- **Baseline aléatoire** : prédire au hasard, selon la distribution uniforme ou la distribution des étiquettes.
- **Heuristique simple** (oublier le ML) : classer un fil par récence, aiguiller par mots-clés.
- **Baseline à règle zéro** : toujours prédire la classe la plus courante (ou l'application la plus utilisée).
- **Baseline humaine** : comment les experts s'en sortent-ils ? C'est la barre pertinente quand on automatise un travail humain.
- **Solution existante** : le système actuellement déployé, parce que l'entreprise te compare à lui.

Dérivation de la baseline aléatoire. Soit une classe positive de prévalence $p$ et un modèle qui prédit positif avec la probabilité $q$, indépendamment de l'entrée. Alors la précision (precision) $=p$ (une prédiction aléatoire est de la classe positive avec la probabilité $p$), le rappel (recall) $=q$, donc

$$
F_1=\frac{2\,p\,q}{p+q},\qquad \text{accuracy}=p\,q+(1-p)(1-q).
$$

Avec $p=0.1$ et un tirage uniforme $q=0.5$ : $F_1=0.1/0.6\approx0.167$ et accuracy $=0.5$. En tirant selon la distribution des étiquettes $q=0.1$ : $F_1=0.1$ et accuracy $=0.01+0.81=0.82$. Un modèle à 82 % d'accuracy sur cette tâche est donc au niveau du hasard.

### Tests de perturbation

Les entrées de production sont plus bruitées que ton jeu de test soigné (des toux enregistrées sur des téléphones dans une pièce bondée contre dans un hôpital). Perturbe les entrées de test avec de petits changements réalistes (bruit, écrêtage, fautes de frappe) et mesure de combien la performance chute. Préfère le modèle qui est le meilleur sur les données perturbées, car la sensibilité au bruit rend un modèle difficile à maintenir : quand le comportement des utilisateurs change légèrement, un modèle fragile se dégrade.

### Tests d'invariance et d'attente directionnelle

Certains changements d'entrée **ne devraient pas** changer la sortie : changer un nom ou un attribut protégé dans un CV ou une demande de prêt. Garde tout le reste fixe, change le champ sensible, et compte combien de prédictions basculent (mieux, n'utilise pas du tout ce champ). D'autres changements devraient déplacer la sortie dans une **direction connue** : augmenter la taille du terrain d'une maison, toutes choses égales par ailleurs, ne doit pas faire baisser le prix prédit. Compter les lignes où la direction est violée expose un modèle qui a appris quelque chose d'étrange. Ces tests n'ont besoin d'aucune étiquette.

### Évaluation par tranche

Une métrique agrégée est une moyenne, et les moyennes cachent la structure. Les **tranches** (slices) sont des sous-ensembles des données (mobile contre ordinateur, région, classes rares, nouveaux utilisateurs). Un modèle peut être excellent dans l'ensemble et mauvais sur une tranche qui compte : une équipe a trouvé d'excellentes performances globales et de terribles résultats sur le trafic mobile parce qu'un bouton était à moitié caché sur les petits écrans, un bogue hors ML que le découpage en tranches a révélé.

Le **paradoxe de Simpson** montre que comparer des agrégats peut même inverser la bonne conclusion. Avec les comptes de la Table 6-4 du livre (tirés d'une étude sur le traitement des calculs rénaux), le modèle A est correct sur 81 de 87 dans le groupe A et 192 de 263 dans le groupe B ; le modèle B est correct sur 234 de 270 et 55 de 80. Par groupe, A gagne (93,1 % contre 86,7 %, et 73,0 % contre 68,8 %). Au global, A obtient 273/350 = 78,0 % tandis que B obtient 289/350 = 82,6 %, donc B gagne. L'inversion se produit parce que les groupes sont pondérés différemment dans les deux totaux : la plupart des exemples d'évaluation de B proviennent du groupe A, le plus facile. Morale : compare les modèles sur les mêmes tranches avec les mêmes données, et rapporte toujours des chiffres par tranche.

**Trouver des tranches.** (1) Des heuristiques issues de la connaissance du domaine ; (2) l'analyse d'erreurs, en lisant les exemples mal classés et en cherchant des motifs ; (3) des chercheurs automatiques de tranches qui génèrent des sous-ensembles candidats (recherche en faisceau, clustering, arbres de décision), les élaguent et les classent. Chaque tranche critique a besoin d'assez d'exemples correctement étiquetés pour donner une estimation stable, donc la qualité de ton évaluation est plafonnée par tes données d'évaluation.

## Exemple détaillé

Chaque nombre ci-dessous est calculé dans les exercices.

- Ensemble : `majority_accuracy(0.7, 3)=0.784` ; avec 11 votants indépendants, elle monte encore.
- Baselines aléatoires avec 10 % de positifs : un tirage uniforme donne $F_1\approx0.167$, accuracy $0.5$ ; un tirage selon la distribution des étiquettes donne $F_1=0.1$, accuracy $0.82$.
- Simpson : les accuracies par tranche du modèle A sont $81/87\approx0.931$ et $192/263\approx0.730$, celles du modèle B $234/270\approx0.867$ et $55/80\approx0.688$ ; au global A $=0.780$, B $=0.826$.
- Perturbation : un modèle qui prédit `x0 > 0` sur un jeu de données dont la vraie étiquette est `x0 + 0.1 x1 > 0` obtient 96,9 % sur données propres. Ajouter un bruit gaussien d'écart-type 0,1, 1 et 2 à chaque feature fait baisser l'accuracy de 0,7, 22,3 et 33,0 points respectivement (graine fixe).
- Invariance : un modèle qui ajoute `0.5 * (sensitive - 0.5)` à son score change sa prédiction pour 9,55 % des lignes quand la colonne sensible est forcée à 1 ; le modèle propre ne change pour aucune.
- Directionnel : le score $\sin(3x_0)$ va dans le mauvais sens pour 50,4 % des lignes quand $x_0$ augmente de 0,5, alors que $2x_0+x_1$ ne le fait jamais.

## Implémentation

```python
import numpy as np
from math import comb

def slice_metrics(y_true, y_pred, slice_ids):
    out = {}
    for s in np.unique(slice_ids):
        m = slice_ids == s
        out[s.item()] = (int(m.sum()), float(np.mean(y_true[m] == y_pred[m])))
    return out

def majority_accuracy(p, n):
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))

def perturbation_drop(predict, X, y, noise_std, rng):
    clean = np.mean(predict(X) == y)
    noisy = np.mean(predict(X + rng.normal(0, noise_std, X.shape)) == y)
    return float(clean - noisy)
```

Bibliothèques : `RandomForestClassifier`, `BaggingClassifier`, `StackingClassifier` de scikit-learn ; `xgboost`/`lightgbm` ; MLflow ou Weights & Biases pour le suivi ; les idées de tranches et de tests comportementaux apparaissent dans des boîtes à outils comme les suites de tests NLP de style checklist et dans des tableaux de bord d'évaluation.

## Dimensions

`y_true, y_pred, slice_ids: (n,)` ; dictionnaire de métriques : identifiant de tranche vers `(count, accuracy)` ; `predict: (n, d) -> (n,)` ; le bruit a la forme de `X`.

## Idées reçues

- « Le meilleur modèle du classement est le meilleur pour nous. » Données, latence et coût différents.
- « L'ensemblage aide toujours d'après la formule. » Seulement si les apprenants de base ne sont pas fortement corrélés.
- « L'accuracy globale monte, donc chaque groupe s'est amélioré. » Le paradoxe de Simpson et les régressions par tranche disent le contraire.
- « Un F1 élevé est bon. » Compare d'abord avec les baselines aléatoire et à règle zéro.
- « Retirer la feature sensible retire le biais. » D'autres features peuvent servir de proxys ; les tests d'invariance vérifient le comportement du modèle, pas seulement ses entrées.

## Points clés à retenir

- Commence simple, compare les architectures avec un effort comparable, et demande-toi quel modèle sera le meilleur plus tard quand les données croîtront.
- Des apprenants de base indépendants se combinent en de meilleures prédictions (trois votants à 70 % donnent 78,4 %) ; le bagging réduit la variance, le boosting se concentre sur les erreurs, le stacking apprend la combinaison.
- Suis et versionne tout ce qu'il faut pour reproduire une exécution.
- Le parallélisme de données est limité par les stragglers (sync) ou les gradients périmés (async) ; le parallélisme de modèle demande du pipelining.
- Évalue par rapport à des baselines, perturbe les entrées, teste les invariances et les directions, et rapporte les métriques par tranche.

## Explique-le avec tes mots

1. Explique, chiffres à l'appui, pourquoi 3 classifieurs indépendants à 70 % se combinent en 78,4 % et pourquoi des classifieurs corrélés ne le font pas.
2. Construis (ou décris) un scénario dans lequel le modèle A est meilleur dans chaque tranche mais pire au global.

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, Ch.6 (PDF p.180-216), y compris les parties non détaillées ici (autres méthodes d'évaluation comme la calibration).
- [[cml.10]] pour les métriques elles-mêmes, et [[cml.03]] pour la généralisation.

## Critères de maîtrise

Tu sais calculer les scores de baseline d'un modèle aléatoire, dériver l'accuracy d'un ensemble à vote majoritaire, dire quand l'ensemblage échoue, écrire des tests de perturbation, d'invariance et directionnels sous forme de courtes fonctions, calculer des métriques par tranche, et expliquer le paradoxe de Simpson avec un exemple.
