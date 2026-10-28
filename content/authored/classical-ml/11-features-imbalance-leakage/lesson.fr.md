---
id: cml.11
title: "Variables, déséquilibre de classes et fuite de données"
summary: "Préparer des entrées que le modèle peut exploiter (mise à l'échelle, encodage, croisements, valeurs manquantes, sélection), gérer les classes rares, et éviter le bug silencieux qui fait mentir les scores de validation : la fuite de données."
minutes: 75
teaches: [ml.features, ml.imbalance, ml.leakage]
requires: [ml.generalization, ml.logreg, ml.eval-metrics]
sources:
  - "Designing Machine Learning Systems.pdf, Ch. 5 Feature Engineering (PDF p.157-178): missing values (MNAR/MAR/MCAR), scaling, discretization, encoding categories and the hashing trick, feature crossing, causes and detection of data leakage, feature importance and generalization. READ."
  - "Designing Machine Learning Systems.pdf, Ch. 4 Class Imbalance (PDF p.132-148): why imbalance is hard, metrics, resampling (undersample/oversample, Tomek links, SMOTE), cost-sensitive and class-balanced loss, focal loss. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 1.5.4 'Compensating for class priors' (PDF p.65): correcting posteriors after rebalancing. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/08-feature-engineering/docs/en.md: numerical transforms, one-hot/label/target encoding, missing-value strategies, interactions. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/17-imbalanced-data/docs/en.md: class weights formula, SMOTE, threshold tuning, cost-sensitive learning, MCC. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/docs/en.md: filter, wrapper and embedded selection, variance threshold, mutual information, RFE, L1, permutation importance. READ."
---

## Où cela s'inscrit

La plupart des projets réels passent plus de temps sur les entrées que sur le modèle. Les données brutes ont des colonnes d'échelles très différentes (âge de 20 à 40 à côté d'un revenu de 10 000 à 150 000), des catégories de type texte (« marque = Acme »), des trous (revenu non renseigné), et une cible rarement équilibrée (0,01 % des radios montrent un cancer, une fraction de pour cent des transactions sont frauduleuses). Un bon modèle sur de mauvaises entrées perd contre un modèle ordinaire sur de bonnes entrées.

Par-dessus cela se trouve l'erreur la plus coûteuse de l'apprentissage automatique appliqué : la **fuite de données** (data leakage), où une information qui n'existera pas au moment de la prédiction s'infiltre dans l'entraînement ou dans le choix des variables. Le symptôme est un score de validation magnifique et un résultat de production qui s'effondre. Les leçons 3 et 10 t'ont dit comment mesurer honnêtement ; cette leçon te dit comment préparer les données pour que la mesure reste honnête, et comment traiter les classes rares où l'accuracy est inutile.

## L'idée en mots simples

**Les variables sont les yeux du modèle.** Le modèle ne voit que des nombres. Si une colonne s'étend de 10 000 à 150 000 et une autre de 20 à 40, un algorithme qui utilise des distances ou des gradients traitera la colonne aux grands nombres comme plus importante simplement parce que ses nombres sont grands. Nous mettons donc les colonnes sur une échelle comparable. Les catégories (« rouge », « bleu ») ne sont pas du tout des nombres ; il faut choisir une représentation numérique qui n'invente pas un ordre qui n'existe pas.

**Les trous portent de l'information.** La raison pour laquelle une valeur manque est en soi un indice. Si les gens riches ont tendance à sauter la question du revenu, alors « revenu manquant » prédit quelque chose. Jeter chaque ligne avec un trou peut écarter justement le motif dont tu as besoin, donc on remplit en général les trous et on se souvient aussi d'où ils étaient.

**Les classes rares se noient dans la foule.** Si 999 exemples sur 1000 sont normaux, la façon la moins coûteuse pour un modèle d'avoir « surtout raison » est d'ignorer la classe rare. On peut lutter contre cela en montrant plus souvent la classe rare (rééchantillonnage), en faisant compter davantage ses erreurs (poids de classes), ou en changeant ensuite le seuil de décision (déplacement du seuil). Il faut aussi utiliser des métriques qui regardent la classe rare.

**La fuite, c'est tricher en regardant.** Imagine un élève qui étudie à partir du sujet d'examen. Son score à cet examen ne dit rien du suivant. Chaque fois que le modèle (ou *toi*, en choisissant des variables, en mettant à l'échelle ou en remplissant des trous) peut regarder les données de test, ou le futur, ou un proxy de la réponse, ton score est celui d'un élève qui a vu le sujet. Le remède est une discipline : **découpe d'abord ; apprends chaque statistique sur le découpage d'entraînement uniquement ; applique-la telle quelle au reste.**

## Mise à l'échelle

Soit $x$ une variable numérique. Transformations courantes :

- **Min-max :** $x' = (x-\min x)/(\max x-\min x)\in[0,1]$.
- **Standardisation (score z) :** $x'=(x-\bar x)/\sigma$, donnant une moyenne 0 et un écart-type 1.
- **Transformation logarithmique :** $x'=\log(1+x)$ pour dompter les variables asymétriques à droite (revenu, comptages).

La mise à l'échelle compte pour tout ce qui utilise des distances ou des pas de gradient (k-means de la leçon 4, régression logistique, SVM, réseaux de neurones) et est presque sans importance pour les arbres de décision. Deux avertissements du chapitre de DMLS : la mise à l'échelle demande des *statistiques globales* (moyenne, écart-type, min, max), et celles-ci doivent être calculées sur le découpage d'entraînement et **réutilisées** à l'inférence ; et si les données de production dérivent des statistiques d'entraînement, la transformation se dégrade, donc les modèles doivent être réentraînés.

### Exemple détaillé 1 : la même valeur aberrante, deux normaliseurs

Valeurs d'entraînement $10,12,14$ et une valeur de test $30$.

- Ajusté sur l'entraînement uniquement : $\mu=12$, $\sigma=\sqrt{8/3}=1.633$. Le point de test a $z=(30-12)/1.633=11.02$ : clairement une valeur aberrante, comme il se doit.
- Ajusté sur l'entraînement et le test ensemble (la façon qui fuit) : $\mu=16.5$, $\sigma=\sqrt{62.75}=7.92$, donc $z=(30-16.5)/7.92=1.70$. La valeur aberrante paraît ordinaire parce qu'elle a influencé son propre étalon.

C'est bénin, mais le principe est exact : la ligne de test a changé les nombres donnés au modèle.

## Encoder les catégories

L'**encodage one-hot** transforme une catégorie à $m$ valeurs en $m$ colonnes binaires, dont exactement une vaut 1. Il n'invente aucun ordre. Coût : $m$ colonnes, ce qui explose pour les variables à forte cardinalité (un million de marques). L'**encodage par étiquette** (rouge=0, bleu=1, vert=2) est compact mais suggère que vert > bleu > rouge ; seuls les modèles à base d'arbres peuvent l'ignorer sans risque. L'**encodage par la cible** remplace une catégorie par la moyenne de la cible dans cette catégorie ; il est puissant mais constitue une fuite classique, car l'étiquette d'une ligne est dans son propre encodage sauf si tu le calcules sur les données d'entraînement uniquement (et de préférence hors-bloc).

La production ajoute une difficulté (DMLS) : les catégories changent avec le temps. Un modèle qui n'a jamais vu une nouvelle marque doit soit planter, soit la regrouper dans une catégorie « INCONNU » sur laquelle il ne s'est jamais entraîné. Dans le code one-hot plus loin, une valeur jamais vue produit simplement une ligne de zéros. Pour les catégories ouvertes, l'**astuce du hachage** (hashing trick) envoie chaque chaîne dans l'un d'un nombre fixe de paniers, $h(s)=\bigl(\sum_i \mathrm{ord}(s_i)\,31^i\bigr)\bmod B$. Avec un espace de hachage de 18 bits il y a $2^{18}=262{,}144$ paniers, les nouvelles catégories obtiennent toujours un indice valide, et les collisions sont aléatoires plutôt que systématiques. Le prix est les collisions et la perte d'interprétabilité.

## Croisements de variables et autres constructions

Un **croisement de variables** (feature cross) combine des variables en une seule : « état civil » et « nombre d'enfants » deviennent « Célibataire, 0 », « Marié, 2 », etc. Il permet à un modèle linéaire de représenter une interaction qu'il ne pourrait pas exprimer autrement, comme « marié avec deux enfants est différent de la somme de “marié” et “deux enfants” ». La réserve est la taille : croiser une variable à 100 valeurs avec une autre à 100 valeurs donne 10 000 valeurs, ce qui demande bien plus de données et invite au surapprentissage. Autres constructions classiques : termes polynomiaux ($x^2$, $x_1x_2$), rapports comme l'IMC $=$ poids / taille$^2$, **discrétisation** (transformer une valeur continue en paniers, au prix de discontinuités aux frontières), et variables métier comme « est-ce l'heure de pointe ». Il y a un compromis entre **généralisation** et **spécificité** : `is_rush_hour` généralise mieux que `hour_of_day` mais jette de l'information.

## Valeurs manquantes

Le chapitre de DMLS distingue trois mécanismes :

- **MNAR** (manquant non aléatoire) : la valeur manque *à cause de sa propre valeur* (les hauts revenus cachent leur revenu).
- **MAR** (manquant aléatoire) : le caractère manquant dépend d'*une autre variable observée* (les gens d'un groupe sautent la question de l'âge).
- **MCAR** (complètement aléatoire) : aucun motif ; rare en pratique.

Options : la **suppression** d'une colonne ou de lignes (raisonnable seulement pour MCAR et une toute petite fraction ; sinon elle perd de l'information et peut biaiser l'échantillon), ou l'**imputation** avec une valeur par défaut, la moyenne, la médiane ou le mode. Évite de remplir avec une valeur elle-même plausible (remplir « nombre d'enfants » par 0 confond « manquant » et « aucun »). Une recette robuste : impute avec la médiane d'entraînement et ajoute un **indicateur de valeur manquante** 0/1 par colonne pour que le modèle puisse exploiter le motif.

## Sélection de variables

Plus de variables n'est pas toujours mieux : elles ajoutent du bruit, de la mémoire, de la latence et des occasions de fuite. La leçon du dépôt regroupe les méthodes en trois familles :

- **Filtre** (regarder chaque variable, sans modèle) : supprimer les variables quasi constantes (seuil de variance), supprimer l'une de deux variables fortement corrélées, classer par corrélation ou par **information mutuelle** $I(X;Y)=\sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$, qui détecte aussi les relations non linéaires.
- **Enveloppe** (wrapper, utiliser un modèle pour évaluer des sous-ensembles) : l'élimination récursive de variables entraîne, supprime la variable la plus faible, et recommence ; précise mais coûteuse.
- **Intégrée** (la sélection se fait pendant l'entraînement) : la régularisation L1 pousse des poids à exactement zéro ; importances des arbres ; **importance par permutation** (mélanger une variable sur les données de validation et voir de combien le score chute) évite le biais de l'importance des arbres envers les variables à forte cardinalité.

DMLS ajoute un critère pratique pour une bonne variable : *l'importance* (à quel point le modèle s'appuie dessus) et la *généralisation* (existe-t-elle au moment de la prédiction avec une couverture et des valeurs similaires ?). Une colonne d'identifiant n'a aucune valeur de généralisation ; `day_of_week` est inutile si la fenêtre d'entraînement ne contient que du lundi au samedi et que tu dois prédire le dimanche. Une variable dont la couverture diffère fortement entre les découpages d'entraînement et de test signale que les découpages ne sont pas comparables, ou que la variable fuit.

**La sélection, c'est de l'entraînement.** Choisir des variables en regardant les étiquettes fait partie de l'ajustement du modèle, donc cela doit se faire à l'intérieur du découpage d'entraînement. Ce fait est le cœur de la section suivante.

## Fuite de données

**Définition.** Il y a fuite quand de l'information extérieure aux données d'entraînement (le jeu de test, le futur, ou l'étiquette elle-même) sert à construire le modèle, de sorte que l'évaluation surestime la performance réelle. Le chapitre de DMLS énumère des causes courantes :

1. **Découpage aléatoire de données corrélées dans le temps.** Prédire le jour 7 à partir des jours 1 à 6 est un problème différent de prédire un jour au hasard. Avec un découpage aléatoire, le marché du jour 7 fuit dans l'entraînement. Découpe par le temps.
2. **Mise à l'échelle avant le découpage** (exemple détaillé 1) : les statistiques des lignes de test entrent dans le modèle.
3. **Imputer avec des statistiques de tout le jeu de données** : même mécanisme.
4. **Doublons entre découpages.** Des quasi-copies dans l'entraînement et le test permettent au modèle de mémoriser. Dédoublonne avant de découper, et si tu sur-échantillonnes, fais-le **après** le découpage (le sur-échantillonnage crée des doublons !).
5. **Fuite par groupe.** Deux examens du même patient, l'un dans l'entraînement et l'autre dans le test.
6. **Fuite par le processus de génération des données.** Un modèle « détecte » une maladie grâce au modèle de scanner utilisé à l'hôpital où les patients sont plus malades.
7. **Sélection de variables ou encodage par la cible utilisant les étiquettes des lignes de test.**
8. **Une variable qui est une conséquence de l'étiquette** (une colonne « sinistre payé » dans un modèle qui prédit si un sinistre sera déclaré), ou deux variables qui ensemble la révèlent (date de début et date de fin prédisent l'ancienneté).

**Détection.** Un score suspicieusement élevé ; une seule variable au pouvoir prédictif anormalement élevé ; des études d'ablation (retirer une variable, voir une grosse chute) ; un grand saut après l'ajout d'une variable ; et toucher le jeu de test plus d'une fois (chaque coup d'œil en fait un second jeu de validation).

**Prévention**, en une règle : *le jeu de test est scellé ; tout ce qui s'apprend (moyennes, médianes, vocabulaires, colonnes sélectionnées, encodages, seuils) s'apprend sur l'entraînement et s'applique au reste ; découpe par le temps ou par groupe quand les données l'exigent.* La vue en pipeline est : `split -> fit transformers on train -> transform train/val/test -> fit model -> evaluate`.

### Exemple détaillé 2 : une fuite sur du pur bruit

Prends $n=40$ exemples avec $d=500$ variables gaussiennes standard et des étiquettes qui sont des tirages à pile ou face (équilibrées) : il n'y a rien à apprendre. Choisis les $k=10$ variables les plus corrélées à l'étiquette et entraîne un classifieur à moyenne de classe la plus proche sur une moitié aléatoire, testé sur l'autre moitié.

- **Avec fuite :** sélectionne les 10 variables en utilisant les 40 lignes, puis découpe. Les colonnes sélectionnées se trouvent aussi corrélées aux étiquettes des lignes de test, parce que ces lignes ont participé au choix. Précision moyenne de test sur 20 graines : environ **0.91**.
- **Honnête :** découpe d'abord, sélectionne en utilisant les 20 lignes d'entraînement uniquement. Précision moyenne de test sur 20 graines : environ **0.47**, c'est-à-dire le hasard.

Mêmes données, même modèle, même code pour le classifieur : 44 points d'écart créés uniquement par *le moment* où les étiquettes ont été consultées. Avec 500 variables candidates, certaines paraissent toujours informatives sur n'importe quel petit échantillon ; les choisir avec les lignes de test incluses ne fait que masquer l'effet des tests multiples. (L'exercice reproduit cela.)

## Déséquilibre de classes

Le déséquilibre est la norme : fraude, attrition, dépistage de maladies, détection de défauts, tri de CV. DMLS donne trois raisons pour lesquelles c'est difficile : trop peu de signal sur la classe rare (un problème à peu d'exemples), une solution triviale « prédire la majorité » difficile à battre, et des **coûts asymétriques** (un cancer raté est bien pire qu'une fausse alarme). Le déséquilibre peut aussi être un artefact d'échantillonnage (le spam qui n'a jamais atteint ta base) ou d'erreurs d'étiquetage, donc regarde d'où il vient.

### Étape 0 : utiliser les bonnes métriques

Évalue avec la précision, le rappel, le F1, la courbe précision-rappel, ou le coefficient de corrélation de Matthews $\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ (leçon 10). Un modèle qui dit « pas de fraude » 1000 fois dans un ensemble de 990 transactions légitimes et 10 frauduleuses a une accuracy de 0.99 et un rappel de 0. **N'évalue jamais sur des données rééchantillonnées** ; évalue sur des données au vrai rapport de classes.

### Méthodes au niveau des données : le rééchantillonnage

- Le **sur-échantillonnage aléatoire** copie des lignes minoritaires jusqu'à équilibrer les classes. Simple, mais les doublons exacts invitent au surapprentissage.
- Le **sous-échantillonnage aléatoire** supprime des lignes majoritaires. Rapide, mais jette de l'information.
- **SMOTE** crée de nouveaux points minoritaires sur le segment entre un point minoritaire $x$ et l'un de ses voisins minoritaires $x'$ : $x_{\text{new}}=x+u\,(x'-x)$ avec $u\sim U(0,1)$. Il fonctionne dans des espaces de variables de faible dimension ; DMLS note que SMOTE et les liens de Tomek n'ont été montrés efficaces qu'en faible dimension.
- Le rééchantillonnage fait partie de l'entraînement : fais-le sur le découpage d'entraînement **après** le découpage.

### Méthodes au niveau de l'algorithme : poids de classes et perte sensible au coût

Au lieu de changer les données, change la perte. Donne à un exemple d'entraînement de classe $c$ le poids

$$
w_c=\frac{N}{C\,n_c},
$$

avec $N$ exemples, $C$ classes et $n_c$ exemples dans la classe $c$. Dans l'exemple du dépôt, 950 négatifs et 50 positifs donnent $w_0=1000/(2\cdot950)=0.526$ et $w_1=1000/(2\cdot50)=10$ : un positif raté compte autant que 19 négatifs ratés. Pour la régression logistique, la perte devient $-\sum_i w_{y_i}\bigl[y_i\log p_i+(1-y_i)\log(1-p_i)\bigr]$. En espérance, cela se comporte comme du sur-échantillonnage sans dupliquer de lignes. Une **matrice de coûts** complète (Elkan) généralise cela à des coûts arbitraires $C_{ij}$ ; la **focal loss** diminue plutôt le poids des exemples que le modèle prédit déjà bien.

### Déplacement du seuil

La plupart des modèles produisent une probabilité. Plutôt que de fixer 0.5, balaie le seuil sur un jeu de validation et choisis celui qui optimise ta métrique ou minimise le coût espéré (leçon 10 : $c_{FP}/(c_{FP}+c_{FN})$ pour des probabilités calibrées).

### Exemple détaillé 3 : un meilleur seuil

Dix cas de validation, trois positifs. Probabilités des positifs : $0.30, 0.25, 0.08$ ; des sept négatifs : $0.20, 0.15, 0.07, 0.05, 0.04, 0.03, 0.02$. Au seuil 0.5 rien n'est signalé, donc $F_1=0$. En balayant les valeurs distinctes :

| seuil | signalés | TP | FP | $F_1$ |
|---|---|---|---|---|
| 0.30 | 1 | 1 | 0 | 0.50 |
| 0.25 | 2 | 2 | 0 | **0.80** |
| 0.20 | 3 | 2 | 1 | 0.67 |
| 0.08 | 5 | 3 | 2 | 0.75 |

Le meilleur seuil est $0.25$. Le classement du modèle était bon ; le seuil par défaut était faux.

### Corriger les probabilités après rééquilibrage

Si tu entraînes sur un jeu équilibré mais déploies sur une population où l'a priori positif est $\pi_{\text{true}}$ (a priori d'entraînement $\pi_{\text{bal}}$), les probabilités a posteriori du modèle équilibré sont proportionnelles aux a priori d'entraînement. Selon la recette de Bishop : divise par les fractions de classes d'entraînement, multiplie par les vraies, renormalise :

$$
p_{\text{true}}(1\mid x)=\frac{p_{\text{bal}}(1\mid x)\,\pi_{\text{true}}/\pi_{\text{bal}}}{p_{\text{bal}}(1\mid x)\,\pi_{\text{true}}/\pi_{\text{bal}}+p_{\text{bal}}(0\mid x)\,(1-\pi_{\text{true}})/(1-\pi_{\text{bal}})}.
$$

Pour un modèle entraîné à 50/50 qui produit $0.8$ dans une population à 1 % de positifs : numérateur $0.8\cdot0.02=0.016$, autre terme $0.2\cdot1.98=0.396$, résultat $0.016/0.412=0.039$. Un signalement « sûr à 80 % » du modèle équilibré ne signifie qu'environ 4 % dans le monde réel, un nombre dont tu as besoin pour budgéter les revues manuelles.

## Implémentation

```python
import numpy as np

class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        self.std_ = np.where(s == 0, 1.0, s)      # constant column: avoid 0/0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_

# the only correct order
# scaler = Standardizer().fit(X_train)
# X_train, X_val, X_test = (scaler.transform(A) for A in (X_train, X_val, X_test))
```

## Formes

`X: (N, d)`, `mean_, std_: (d,)`, one-hot de `n` valeurs et `m` catégories : `(n, m)`, sortie de l'imputeur `(n, 2d)` (valeurs remplies plus indicateurs), croisement de deux colonnes de longueur `n` : une liste de `n` chaînes, poids de classes : un scalaire par classe.

## Idées reçues

- **« Mettre à l'échelle avant le découpage est inoffensif, ce n'est que du prétraitement. »** Cela fait fuiter les statistiques de test (et toute autre statistique apprise aussi).
- **« Rééchantillonner tout le jeu de données puis découper convient. »** Les doublons et les points synthétiques se retrouvent des deux côtés et gonflent le score de test.
- **« Le déséquilibre doit toujours être corrigé. »** Si la métrique et le seuil sont bons, un modèle peut apprendre le vrai déséquilibre ; le rééchantillonnage déforme les probabilités.
- **« Supprime les lignes avec valeurs manquantes, c'est plus propre. »** Cela biaise l'échantillon sauf si les valeurs sont MCAR et rares, et le caractère manquant peut être informatif.
- **« Plus de variables ne peut pas nuire. »** Elles augmentent le surapprentissage, le coût et le risque de fuite.
- **« Un excellent score de validation est une bonne nouvelle. »** Un score étonnamment excellent est le premier signe qu'il faut chercher une fuite.

## Points clés à retenir

- Découpe d'abord ; apprends normaliseurs, imputeurs, vocabulaires, encodages et variables sélectionnées sur le découpage d'entraînement uniquement.
- Encode les catégories sans inventer d'ordre ; prévois les catégories jamais vues ; les croisements ajoutent des interactions au prix de la taille.
- Les valeurs manquantes sont de l'information : impute et ajoute des indicateurs plutôt que de supprimer.
- Sous déséquilibre, utilise PR/F1/MCC, des poids de classes ou du rééchantillonnage sur le découpage d'entraînement uniquement, et déplace le seuil.
- Soupçonne une fuite dès qu'un score est trop bon ; découpe par le temps ou par groupe quand les données l'exigent.

## Explique-le toi-même

1. Pourquoi sélectionner les 10 variables les mieux corrélées sur tout le jeu de données gonfle-t-il le score de test même quand les étiquettes sont aléatoires ?
2. Un modèle entraîné sur un jeu rééchantillonné 50/50 annonce 0.8. Que dois-tu faire avant d'y lire une probabilité pour le trafic réel, et pourquoi ?

## Pour aller plus loin

- *Designing Machine Learning Systems*, Ch. 4 (déséquilibre de classes) et Ch. 5 (ingénierie des variables, fuite de données).
- Bibliothèque source : `ai-engineering-from-scratch` phase 02, leçons 08, 17, 18.

## Critères de maîtrise

Tu sais construire des transformateurs de prétraitement ajustés uniquement sur l'entraînement, expliquer MNAR/MAR/MCAR, choisir entre rééchantillonnage, poids de classes et déplacement du seuil, corriger des probabilités a posteriori pour un a priori changé, et concevoir un pipeline qui ne peut pas fuir.
