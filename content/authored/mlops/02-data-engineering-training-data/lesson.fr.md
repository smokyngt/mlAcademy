---
id: ops.02
title: "Ingénierie des données et données d'entraînement"
summary: "D'où viennent les données de production, comment elles sont stockées et traitées, et comment construire un jeu d'entraînement avec un échantillonnage correct, des étiquettes bon marché, de la supervision faible et de l'augmentation, avec des échantillonneurs par réservoir et stratifié implémentés from scratch."
minutes: 70
teaches: [ops.sampling, ops.labelling]
requires: [prob.sampling, ml.imbalance, prob.basics, ops.framing]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.3 Data Engineering Fundamentals: data sources, formats (JSON/CSV/Parquet, row- versus column-major), transactional versus analytical processing, batch versus stream processing (PDF p.79-110, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.4 Training Data: sampling (non-probability, stratified, reservoir, importance), labelling (hand labels, natural labels, weak supervision, semi-supervision, transfer learning, active learning), class imbalance overview, data augmentation and synthesis (PDF p.114-154, read in selected parts). READ."
---

## Où cela s'insère

Un modèle ne vaut que par les exemples qu'il a vus. Dans [[ops.01]], tu as cadré le problème ; il te faut maintenant des lignes sur lesquelles entraîner. Dans une entreprise, ces lignes sont dispersées entre des logs, des bases de données et des fichiers téléversés par les utilisateurs, dans plusieurs formats, arrivent en continu, et la plupart n'ont **aucune étiquette**. Cette leçon couvre la plomberie (où vivent les données et comment elles sont traitées), puis l'art de construire un jeu d'entraînement : quels exemples échantillonner, d'où viennent les étiquettes, et comment étirer un petit jeu étiqueté. Le déséquilibre de classes est traité en profondeur dans [[cml.11]] ; ici, nous le situons seulement dans le pipeline. La construction de features et les fuites dans les pipelines viennent ensuite dans [[ops.03]].

## L'idée en termes simples

Imagine une bibliothèque qui reçoit des milliers de nouveaux livres à l'heure (le **flux**, ou stream) et garde les anciens au sous-sol (l'**entrepôt**, ou warehouse). Tu veux une liste de lecture représentative de 100 livres pour un cours, et tu veux aussi que chaque livre soit annoté par un expert, mais les experts sont lents et chers. Plusieurs questions se posent :

- Quels livres choisis-tu ? Si tu prends ce qui se trouve sur l'étagère la plus proche, ta liste est biaisée. Si la bibliothèque ne compte que 1 % de poésie et que tu veux être sûr que la poésie figure dans la liste, tu prends 1 % de chaque étagère séparément.
- Si les livres continuent d'arriver et que tu ne peux pas les compter d'abord, comment garder à tout moment une liste équitable de 100 ? Tu gardes un petit sac et, pour chaque nouvel arrivant, tu lances une pièce pondérée pour décider s'il remplace un livre au hasard dans le sac.
- Qui annote ? Peut-être que des lecteurs en ont déjà annoté certains par leur comportement (ils ont fini le livre : une *étiquette naturelle*), peut-être que tu écris des règles simples (« mentionne un dragon signifie fantasy ») et laisses de nombreuses règles bruitées voter, peut-être que tu ne demandes aux experts que les livres sur lesquels tes suppositions actuelles sont les plus incertaines.
- Peux-tu fabriquer plus d'exemples d'entraînement à partir de ceux que tu as ? Retourne-les, recadre-les ou paraphrase-les, tant que l'étiquette ne change pas.

## Sources et formats de données

**Sources.** Les *entrées utilisateur* (formulaires, téléversements) sont abondantes mais désordonnées : les utilisateurs saisissent du texte là où des nombres sont attendus, donc il faut une validation, et les utilisateurs attendent des réponses instantanées. Les *données générées par le système* (logs, prédictions de modèles, métriques) sont plus propres et sont essentielles pour le débogage et, plus tard, le monitoring ([[ops.06]]). Il y a aussi les bases de données internes de l'entreprise et les données tierces.

**Formats.** JSON est flexible et lisible par un humain mais coûte de l'espace et rend les changements de schéma pénibles. CSV et Parquet sont les deux formats tabulaires canoniques et représentent deux paradigmes de stockage :

- **Row-major** (CSV) : les valeurs d'une ligne sont côte à côte. Lire des exemples entiers et ajouter de nouvelles lignes est rapide.
- **Column-major** (Parquet) : les valeurs d'une colonne sont côte à côte. Lire quelques colonnes parmi des milliers est rapide, ce qui est courant en analytique ; ajouter des lignes une à une est plus lent.

Si une table a 1 000 colonnes et que tu en as besoin de 4, le column-major lit 4 colonnes et le row-major doit parcourir chaque ligne. Si tu dois journaliser un nouvel événement à la fois, le row-major gagne. Des bibliothèques comme pandas sont construites autour d'un DataFrame orienté colonnes, c'est pourquoi itérer sur les lignes est lent et les opérations sur les colonnes sont rapides. Les formats texte sont lisibles par un humain mais plus volumineux que les formats binaires.

**Traitement.** Les systèmes *transactionnels* (OLTP) insèrent et mettent à jour des événements individuels avec une faible latence et une haute disponibilité ; les systèmes *analytiques* (OLAP) répondent à des requêtes sur de grandes quantités d'historique. Les pipelines de données de ML passent généralement par un **ETL** : extraire des sources, transformer (nettoyer, valider, joindre), charger dans un entrepôt ou un lac de données.

**Batch contre stream.** Le *traitement par lots* (batch) s'exécute périodiquement sur des données historiques (un job de nuit qui calcule la note moyenne de chaque conducteur). Le *traitement en flux* (stream) calcule sur les données au fil de leur passage dans un bus de messages, avec un court délai, et est particulièrement bon pour les mises à jour *avec état* : au lieu de recalculer chaque jour un chiffre d'engagement sur 30 jours, tu le mets à jour avec les événements du nouveau jour. Les quantités qui changent lentement deviennent des **features batch (statiques)** ; celles qui changent vite (courses demandées dans la dernière minute) deviennent des **features en flux (dynamiques)**. De nombreux modèles ont besoin des deux, c'est pourquoi les pipelines d'entraînement et de serving doivent s'accorder sur la manière de calculer chacune ([[ops.03]]).

## Échantillonnage

Tu ne t'entraînes presque jamais sur toutes les données. L'échantillonnage décide de ce que le modèle voit, donc de ce qu'il peut apprendre.

**L'échantillonnage non probabiliste** sélectionne par commodité, par effet boule de neige à partir d'échantillons existants, par jugement d'expert, ou par quotas. Il est bon marché et biaisé. Beaucoup de données d'entraînement réelles sont de ce type : des modèles de langage entraînés sur du texte facile à crawler, ou des modèles de sentiment entraînés sur des avis en ligne écrits par des gens prêts à donner leur avis. Connais le biais que tu acceptes.

**L'échantillonnage aléatoire (probabiliste)** donne à chaque élément une chance connue d'être sélectionné.

- *Échantillonnage aléatoire simple* : chaque élément a la même probabilité. Simple, mais une classe rare peut être entièrement manquée.
- *Échantillonnage stratifié* : diviser la population en groupes (**strates**) et échantillonner chaque groupe séparément, de sorte qu'un groupe rare soit garanti d'apparaître. Il échoue quand les éléments appartiennent à plusieurs groupes à la fois (multilabel).
- *Échantillonnage pondéré* : les éléments portent des poids qui fixent leur probabilité, par exemple des données récentes pondérées plus fortement.
- *Échantillonnage préférentiel* (importance sampling) : tu veux des échantillons d'une distribution $P$ difficile à échantillonner, donc tu échantillonnes une distribution facile $Q$ (avec $Q(x)>0$ partout où $P(x)>0$) et tu pondères chaque échantillon par $P(x)/Q(x)$. C'est la même idée que dans les leçons de Monte Carlo ; en ML, elle réapparaît quand tu corriges une distribution d'entraînement qui diffère de celle de production.
- *Échantillonnage par réservoir* pour les flux, dérivé ci-après.

### Échantillonnage par réservoir

**Objectif.** Maintenir, en un seul passage et avec une mémoire de seulement $k$ éléments, un échantillon dans lequel chacun des $n$ éléments vus jusqu'ici est présent avec la probabilité $k/n$, pour tout $n$ (de sorte que tu peux t'arrêter à tout moment).

**Algorithme.** Garde les $k$ premiers éléments. Pour l'élément d'indice 1-indexé $n>k$, tire un entier $j$ uniformément dans $\{1,\dots,n\}$ ; si $j\le k$, mets le nouvel élément dans l'emplacement $j$, sinon écarte-le.

**Pourquoi cela marche (récurrence sur $n$).** Supposons qu'après $n$ éléments chacun d'eux soit dans le réservoir avec la probabilité $k/n$. Le nouvel élément $n+1$ entre avec la probabilité $k/(n+1)$ par construction. Un ancien élément actuellement dans le réservoir n'est évincé que si le nouvel élément entre (probabilité $k/(n+1)$) **et** tombe dans l'emplacement précis de cet élément (probabilité $1/k$), c'est-à-dire avec la probabilité $1/(n+1)$. Il survit donc avec la probabilité $1-\frac{1}{n+1}=\frac{n}{n+1}$, et sa probabilité globale est

$$
\frac{k}{n}\cdot\frac{n}{n+1}=\frac{k}{n+1}.
$$

Le cas de base ($n=k$, probabilité 1) est vérifié, donc l'affirmation vaut pour tout $n$.

## Étiquetage

Les **étiquettes naturelles** viennent du système lui-même : le temps de trajet réel d'une course que tu avais prédit, le fait qu'un utilisateur ait cliqué sur une recommandation dans les dix minutes (une *étiquette comportementale*). Elles arrivent avec une **longueur de boucle de rétroaction** (feedback loop length), le délai entre la prédiction et son étiquette, qui va de quelques minutes (clics) à des mois (défaut de prêt). Un système peut aussi être conçu pour collecter du retour (boutons pouce haut/pouce bas). Les tâches avec étiquettes naturelles sont courantes dans l'industrie, en partie parce que les équipes les choisissent.

Les **étiquettes manuelles** sont le choix par défaut sinon, et elles sont coûteuses (des experts pour des images médicales), un risque pour la vie privée (quelqu'un doit lire les données), lentes (le livre cite la transcription phonétique de la parole prenant des centaines de fois la durée de l'audio) et incohérentes (les annotateurs sont en désaccord). Des étiquettes lentes rendent un modèle lent à s'adapter quand les exigences changent.

La **supervision faible** remplace une partie de l'étiquetage manuel par des *fonctions d'étiquetage* (labelling functions, LFs) : de petits programmes encodant des heuristiques, comme « si la note mentionne pneumonie alors URGENT », une expression régulière, une recherche dans une liste, ou la sortie d'un autre modèle. Une LF soit vote pour une classe, soit **s'abstient**. Comme les heuristiques sont bruitées, se recoupent et se contredisent, on les agrège. Un petit échantillon étiqueté à la main reste nécessaire pour mesurer la précision de chaque LF. La supervision faible convient aux données soumises à des contraintes de vie privée (les experts examinent un petit échantillon autorisé, puis les LFs s'exécutent sur tout) et rend l'expertise réutilisable et versionnable : change les exigences, relance les fonctions.

*Agrégation.* L'agrégateur le plus simple est le vote majoritaire parmi les LFs qui ne s'abstiennent pas. Un meilleur agrégateur pondère chaque LF par sa fiabilité. Si la LF $j$ est correcte avec la probabilité $a_j$ et que, étant donnée la vraie étiquette, les LFs se trompent indépendamment (une hypothèse de Bayes naïf), alors le log du rapport de cotes a posteriori de la classe 1 contre la classe 0 est le log du rapport de cotes a priori plus une somme, sur les LFs qui votent, de $\pm\log\frac{a_j}{1-a_j}$ (plus pour un vote en faveur de 1, moins pour un vote en faveur de 0), parce qu'un vote pour 1 multiplie les cotes par $a_j/(1-a_j)$ et qu'un vote pour 0 les divise par ce facteur. Prédire 1 quand cette somme est positive (a priori égaux) est le **vote pondéré** implémenté dans l'exercice. Les LFs précises reçoivent un grand poids, une LF de précision 0,5 reçoit un poids 0. Les vrais modèles d'étiquettes (par exemple celui de Snorkel) estiment les précisions sans étiquettes de référence et modélisent les dépendances entre LFs ; l'hypothèse d'indépendance est ici la simplification.

La **semi-supervision** utilise des hypothèses structurelles pour étendre un petit jeu étiqueté : l'*auto-entraînement* (self-training : entraîner, étiqueter les points non étiquetés sur lesquels le modèle est confiant, réentraîner), des hypothèses de regroupement/similarité, ou la cohérence par perturbation. Le **transfer learning** part d'un modèle pré-entraîné sur une tâche abondante (la modélisation du langage n'a besoin d'aucune étiquette) et l'affine avec peu d'étiquettes. L'**apprentissage actif** choisit quels exemples non étiquetés envoyer aux annotateurs, typiquement ceux dont le modèle actuel est le moins sûr.

Quand les données étiquetées sont rares, fais attention au jeu d'évaluation : trop petit, la sélection de modèle le sur-ajuste ; trop grand, tu gaspilles des étiquettes. Un compromis courant est de sélectionner le champion sur le jeu d'évaluation puis de continuer l'entraînement dessus.

## Déséquilibre de classes et augmentation

Le **déséquilibre de classes** (une classe positive rare, comme 0,01 % de scanners cancéreux) est la norme. Il pose problème parce que la classe rare porte peu de signal, parce que « toujours prédire la majorité » est un raccourci difficile à battre avec 99,99 % d'accuracy, et parce que les erreurs sur la classe rare sont généralement les coûteuses. Les remèdes sont des métriques qui respectent la classe rare, le rééchantillonnage et des fonctions de perte sensibles au coût ; ces techniques, avec du code, sont dans [[cml.11]]. Du point de vue du pipeline de données : stratifie quand tu sépares et échantillonnes, n'évalue jamais sur un jeu de test rééchantillonné, et rappelle-toi que le rééchantillonnage modifie les probabilités a priori des classes que ton modèle produira.

L'**augmentation de données** crée de nouveaux exemples à partir d'anciens.

- *Transformations préservant l'étiquette* : recadrer, retourner, faire pivoter une image (un chien tourné reste un chien) ; remplacer un mot par un quasi-synonyme dans un texte. C'est presque gratuit et cela multiplie les données.
- *Perturbation* : ajouter du bruit. Les réseaux peuvent être trompés par de minuscules changements, ce qui fait aussi de l'augmentation par bruit une défense et un test de robustesse ([[ops.04]]).
- *Synthèse de données* : remplir des modèles (« Trouve-moi un restaurant [CUISINE] à moins de [NOMBRE] miles de [LIEU] ») pour amorcer les requêtes d'entraînement d'un chatbot, ou mélanger deux images et leurs étiquettes pour obtenir un exemple avec une étiquette douce.

L'augmentation s'applique à la partition d'**entraînement** uniquement ; les partitions de validation et de test doivent rester intactes.

## Exemple détaillé

Tous les nombres viennent des deux exercices.

*Réservoir.* Flux de 20 éléments, $k=5$, 4 000 exécutions indépendantes avec une graine fixe. Chaque élément devrait apparaître dans $5/20=0.25$ des exécutions. Les fréquences empiriques se situent entre 0,236 et 0,264 pour les 20 éléments, sans tendance entre les premiers et les derniers éléments ; un naïf « garder les cinq premiers » donnerait 1,0 et 0,0.

*Stratifié.* Étiquettes avec 900 de la classe 0, 90 de la classe 1 et 10 de la classe 2 ; échantillon de 10 %. Le résultat a exactement 90, 9 et 1 éléments par classe. Un échantillon aléatoire simple de 10 % aurait, en moyenne, un élément de la classe 2 mais pourrait en avoir zéro ou trois.

*Supervision faible.* Prends la matrice de votes (les lignes sont des exemples, `-1` est une abstention) :

```text
[[ 1,  1, -1],
 [ 0,  1, -1],
 [-1, -1, -1],
 [ 0,  0,  1],
 [ 1, -1,  0]]
```

La couverture par LF est $[0.8, 0.6, 0.4]$. Les lignes 1, 3 et 4 contiennent à la fois un 0 et un 1, donc le taux de conflit est $0.6$. Le vote majoritaire donne $[1,-1,-1,0,-1]$ : la ligne 1 est une égalité 1-1, la ligne 2 n'a aucun vote, la ligne 4 est une égalité 1-1. Avec les étiquettes de référence $[1,1,0,0,1]$, les précisions des LFs (sur les lignes où elles votent) sont $[0.75, 1.0, 0.0]$. Dans une simulation avec 4 000 exemples et quatre LFs de précision réelle $0.9, 0.6, 0.6, 0.6$, les précisions estimées sortent proches de $0.896, 0.594, 0.593, 0.596$ ; le vote pondéré a raison dans 89,6 % des cas, tandis que le simple vote majoritaire (en comptant les égalités comme fausses) a raison dans 60,0 % des cas, parce que trois LFs médiocres l'emportent sur l'excellente.

## Implémentation

```python
import numpy as np

def reservoir_sample(stream, k, rng):
    res = []
    for n, x in enumerate(stream):          # n is 0-based
        if n < k:
            res.append(x)
        else:
            j = int(rng.integers(0, n + 1))  # uniform on {0..n}
            if j < k:
                res[j] = x
    return res

def weighted_vote(L, acc):
    w = np.log(acc / (1 - acc))
    sign = np.where(L == 1, 1.0, np.where(L == 0, -1.0, 0.0))
    return (sign @ w > 0).astype(int)
```

Outils de production : Spark/Flink pour les jobs batch et stream, Parquet pour le stockage en colonnes, Snorkel pour les modèles d'étiquettes, `torchvision.transforms` ou `albumentations` pour l'augmentation d'images, `StratifiedShuffleSplit` de scikit-learn pour les séparations stratifiées.

## Dimensions

Matrice de votes `L: (n, m)` avec `n` exemples et `m` fonctions d'étiquetage ; précisions `acc: (m,)` ; score pondéré `sign @ w: (n,)`. Réservoir : une liste de longueur $k$. Échantillon stratifié : un vecteur d'indices triés.

## Idées reçues

- « Plus de données, c'est toujours mieux. » Des données biaisées ou mal étiquetées amplifient le biais. La qualité et la représentativité passent d'abord.
- « L'échantillonnage aléatoire est toujours sûr. » L'échantillonnage aléatoire simple peut manquer entièrement une classe rare ; stratifie.
- « Les étiquettes faibles sont juste de mauvaises étiquettes. » Combinées et pondérées, de nombreuses LFs bruitées mais diverses peuvent battre une seule LF médiocre, et on peut les relancer quand les exigences changent ; mais elles ont tout de même besoin d'un petit jeu de référence pour être validées.
- « Le traitement en flux est le petit frère moins efficace du batch. » Les moteurs de flux modernes sont distribués, et les mises à jour avec état évitent le recalcul.
- « Augmenter avant de séparer. » Cela met des quasi-doublons des éléments de test dans l'entraînement : une fuite ([[ops.03]]).

## Points clés à retenir

- Choisis le stockage et le mode de traitement selon le schéma d'accès : row-major pour les écritures et les lignes entières, column-major pour les parcours de colonnes ; batch pour les features lentes, stream pour les rapides.
- L'échantillonnage par réservoir conserve un échantillon uniforme d'un flux de longueur inconnue avec une mémoire $O(k)$, prouvé par récurrence.
- L'échantillonnage stratifié garantit que les groupes rares apparaissent ; échantillonne et sépare par strate.
- Les étiquettes naturelles sont peu coûteuses mais retardées ; les étiquettes manuelles sont lentes et coûteuses ; la supervision faible, la semi-supervision, le transfert et l'apprentissage actif étirent des étiquettes rares.
- N'augmente que la partition d'entraînement, et uniquement avec des changements préservant l'étiquette.

## Explique-le avec tes mots

1. Prouve que, dans l'échantillonnage par réservoir, la probabilité qu'un élément précoce reste dans le réservoir après $n$ éléments est $k/n$.
2. Tu as trois fonctions d'étiquetage de précisions 0,55, 0,6 et 0,95. Pourquoi ne faut-il pas se fier à un simple vote majoritaire, et quels poids donnerais-tu à la place ?

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, Ch.3-4 (PDF p.79-156). Kleppmann, *Designing Data-Intensive Applications* pour les systèmes de données en profondeur (référencé par le livre, non lu pour cette leçon).
- L'idée du modèle d'étiquettes derrière Snorkel (Ratner et al.), citée dans le livre, non lue pour cette leçon.

## Critères de maîtrise

Tu sais choisir un stockage row- ou column-major et un traitement batch ou stream pour une charge donnée, implémenter et prouver l'échantillonnage par réservoir, construire un échantillon stratifié, agréger des fonctions d'étiquetage bruitées avec un vote pondéré, et expliquer où dans un pipeline l'augmentation et le rééchantillonnage sont permis.
