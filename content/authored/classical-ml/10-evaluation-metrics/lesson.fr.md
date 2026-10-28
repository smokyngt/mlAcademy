---
id: cml.10
title: "Métriques d'évaluation pour la classification et la régression"
summary: "Transformer « ce modèle est-il bon ? » en nombres qui correspondent au coût réel des erreurs : matrice de confusion, précision/rappel, ROC et AUC (dérivée comme une probabilité), calibration, et erreurs de régression."
minutes: 70
teaches: [ml.eval-metrics, ml.roc-auc, ml.calibration]
requires: [prob.basics, ml.logreg, ml.generalization]
sources:
  - "Designing Machine Learning Systems.pdf, Ch. 6 'Model Offline Evaluation' (PDF p.201-214): baselines (Table 6-2), perturbation/invariance tests, calibration, slice-based evaluation and Simpson's paradox. READ."
  - "Designing Machine Learning Systems.pdf, Ch. 4 'Using the right evaluation metrics' (PDF p.134-145): the cancer models A and B (Tables 4-4 to 4-7), asymmetry of F1, ROC and precision-recall curves. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 1.5 Decision Theory (PDF p.58-66): minimum misclassification rate, loss matrix and expected loss (1.81), reject option, compensating for class priors. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/09-model-evaluation/docs/en.md: confusion matrix, precision/recall/F1/AUC, MSE/RMSE/MAE/R^2 definitions, common evaluation mistakes. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/17-imbalanced-data/docs/en.md (metrics section): F-beta, AUPRC baseline equals the positive rate. READ."
---

## Où cela s'inscrit

Un hôpital déploie un modèle qui lit des radiographies thoraciques. Dans le jeu de validation, 1 patient sur 100 a un cancer. Le modèle a 99 % de précision (accuracy). Est-il bon ? Ce pourrait être un modèle qui répond « sain » pour tout le monde. Une équipe antifraude compare deux modèles : l'un est légèrement meilleur sur l'erreur moyenne, l'autre classe plus haut les transactions frauduleuses. Lequel mettre en production ? Une équipe publicitaire veut prévoir combien de clics rapportera la semaine prochaine, pas seulement quelle annonce arrive en tête.

Chaque question demande un étalon différent. Entraîner un modèle (leçons 1 à 5) n'est que la moitié du travail ; l'autre moitié est de le **mesurer** d'une façon qui reflète ce que coûtent les erreurs. Une mauvaise métrique ne donne pas seulement un mauvais nombre : elle te fait choisir le mauvais modèle, régler la mauvaise chose, et déclarer un succès sur un système qui ne fonctionne pas. Cette leçon construit la boîte à outils standard et, plus important, l'habitude d'y choisir en se demandant « que coûte chaque type d'erreur ? ».

Elle se relie à la régression logistique (leçon 2), dont la sortie est une probabilité, et à la généralisation (leçon 3), car chaque nombre ici doit être calculé sur des données que le modèle n'a pas utilisées pour s'entraîner. Elle se prolonge vers le déséquilibre de classes et la fuite de données (leçon 11).

## L'idée en mots simples

**Compter correctement les erreurs.** Un classifieur peut se tromper de deux manières différentes. Il peut déclencher une alarme qui n'aurait pas dû l'être (une fausse alarme), ou il peut rester silencieux alors qu'il aurait dû parler (un oubli). Détecteurs de fumée, filtres anti-spam, dépistage du cancer : dans chacun, les deux erreurs ne font pas le même mal. Un tableau qui compte les quatre issues (alarme/pas d'alarme croisé avec vrai feu/pas de feu) est la **matrice de confusion**, et chaque métrique de classification est une façon différente de la résumer.

**La précision (accuracy) est une moyenne sur tout le monde.** Elle est dominée par le groupe le plus grand. Si 99 % des gens sont en bonne santé, « tout le monde est sain » obtient 99 %. La question qui nous importe vraiment porte en général sur le groupe rare : parmi les gens malades, combien en avons-nous détecté (**rappel**) ? Parmi les gens que nous avons signalés, combien étaient vraiment malades (**précision**, precision) ?

**Un seuil est un cadran, pas un fait.** La plupart des modèles produisent un score ou une probabilité, et nous choisissons un seuil pour en faire un oui/non. Baisse le seuil et tu détectes plus de vrais cas mais tu déclenches plus de fausses alarmes. La **courbe ROC** dessine tout ce compromis en une image, et l'**AUC** le résume par un seul nombre à la signification charmante : *prends un positif au hasard et un négatif au hasard ; l'AUC est la probabilité que le modèle donne au positif le score le plus élevé.* Elle ne mesure que le classement.

**Classer n'est pas être honnête.** Un prévisionniste qui annonce « 70 % de pluie » devrait voir de la pluie environ 70 % de ces jours-là. Cette propriété est la **calibration**. Un modèle peut classer parfaitement (AUC = 1) et pourtant annoncer « 99 % » quand la vérité est 60 %. Si tu ne fais que trier des éléments, la calibration n'a pas d'importance ; si tu multiplies des probabilités par de l'argent ou comptes des événements attendus, elle compte énormément.

**La régression raconte la même histoire.** Les erreurs peuvent se mesurer dans les unités de la cible (MAE, RMSE) ou relativement à une prédiction triviale (R carré). Élever les erreurs au carré punit lourdement les grosses erreurs ; les valeurs absolues les traitent linéairement.

**Un nombre seul n'a aucun sens.** « F1 = 0.90 » semble formidable jusqu'à ce que tu apprennes que prédire la classe majoritaire au hasard donne aussi environ 0.90 sur ces données. Compare toujours à une référence (baseline).

## La matrice de confusion et ses métriques

Pour un problème binaire avec des étiquettes $y\in\{0,1\}$ (1 = positif) et des prédictions $\hat y$ :

| | prédit 1 | prédit 0 |
|---|---|---|
| réel 1 | TP (succès) | FN (oubli) |
| réel 0 | FP (fausse alarme) | TN |

$$
\text{accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad \text{precision}=\frac{TP}{TP+FP},\qquad \text{recall}=\frac{TP}{TP+FN}.
$$

Le **taux de faux positifs** est $FPR = FP/(FP+TN)$, et le rappel s'appelle aussi taux de vrais positifs, $TPR$. Le score **F1** est la moyenne harmonique de la précision $P$ et du rappel $R$ :

$$
F_1=\frac{2PR}{P+R}.
$$

La moyenne harmonique est attirée vers le plus petit des deux, donc tu ne peux pas obtenir un $F_1$ élevé en maximisant l'un et en ignorant l'autre. La version pondérée $F_\beta=(1+\beta^2)PR/(\beta^2P+R)$ traite le rappel comme $\beta$ fois plus important que la précision.

La précision (precision), le rappel et le F1 sont **asymétriques** : ils changent quand on échange la classe appelée positive. L'accuracy, non.

## Exemple détaillé 1 : deux modèles, même accuracy

Prends l'exemple du livre de 1000 patients, 100 avec un cancer (positifs) et 900 normaux.

| | TP | FP | FN | TN |
|---|---|---|---|---|
| Modèle A | 10 | 10 | 90 | 890 |
| Modèle B | 90 | 90 | 10 | 810 |

Les deux ont une accuracy de $(10+890)/1000=(90+810)/1000=0.9$. Mais :

- Modèle A : précision $10/20=0.5$, rappel $10/100=0.1$, $F_1=2(0.5)(0.1)/0.6=0.167$.
- Modèle B : précision $90/180=0.5$, rappel $90/100=0.9$, $F_1=2(0.5)(0.9)/1.4=0.643$.

Le modèle A rate 90 des 100 personnes malades. Même accuracy, utilité complètement différente. (L'exercice reproduit ces nombres.)

**Références (baselines).** Avec 10 % de positifs, un modèle qui prédit positif avec probabilité 0.5 au hasard a une accuracy de $0.5$ et un $F_1=0.167$ (précision 0.1, rappel 0.5). Un modèle qui prédit positif 10 % du temps au hasard (en suivant la distribution des étiquettes) a une accuracy de $0.9\cdot0.9+0.1\cdot0.1=0.82$ et un $F_1=0.1$. Un modèle « règle zéro » qui prédit toujours la classe la plus fréquente est la référence la plus simple. Tout modèle que tu construis doit les battre d'une marge qui justifie son coût. Autres références utiles : une heuristique simple, un expert humain, le système existant.

## Théorie de la décision : d'où vient vraiment le seuil

Un classifieur qui produit $p(C_k\mid x)$ a fait l'*inférence*. Choisir l'action est une *décision* distincte. La Sec. 1.5 de Bishop montre que la règle qui commet le moins d'erreurs affecte $x$ à la classe de plus grande probabilité a posteriori. Mais les erreurs coûtent rarement la même chose, donc on introduit une **matrice de perte** $L_{kj}$ = perte quand la vraie classe est $k$ et que l'on décide $j$. Pour chaque $x$, la perte espérée de décider $j$ est $\sum_k L_{kj}\,p(C_k\mid x)$, et on choisit le $j$ qui la minimise.

### Dérivation : le seuil de coût

Deux classes : positive (1) et négative (0). Soit une décision correcte coûtant 0, un faux positif coûtant $c_{FP}$ et un faux négatif coûtant $c_{FN}$. Avec $p=p(y=1\mid x)$ :

$$
\mathbb E[\text{loss of predicting }1]=(1-p)\,c_{FP},\qquad
\mathbb E[\text{loss of predicting }0]=p\,c_{FN}.
$$

Prédis 1 exactement quand la première est plus petite :

$$
(1-p)\,c_{FP}<p\,c_{FN}\iff p>\frac{c_{FP}}{c_{FP}+c_{FN}}.
$$

Avec des coûts égaux, le seuil est $0.5$. Dans la matrice du cancer de Bishop, un cancer raté coûte 1000 et une fausse alarme coûte 1, donc le seuil est $1/1001\approx0.001$ : signale quiconque a ne serait-ce que 0,1 % de chances. Remarque ce que cela dit : **le défaut de 0.5 est un cas particulier, pas une loi.** Cela dit aussi pourquoi les probabilités (et pas seulement les étiquettes) ont de la valeur : si les coûts changent le trimestre prochain, tu déplaces le seuil sans réentraîner. Bishop décrit aussi une **option de rejet** : si la plus grande probabilité a posteriori est inférieure à un certain $\theta$, confie le cas à un humain.

## Courbe ROC et AUC

Étant donné des scores $s_i$ (des réels quelconques, plus grand signifie « plus positif »), chaque seuil $t$ donne un point $(\mathrm{FPR}(t),\mathrm{TPR}(t))$ avec les prédictions $\hat y=\mathbb 1[s\ge t]$. Balayer $t$ de $+\infty$ jusqu'à $-\infty$ trace la **courbe ROC** de $(0,0)$ (rien n'est signalé) à $(1,1)$ (tout est signalé). Un modèle parfait épouse le coin supérieur gauche ; des scores aléatoires suivent la diagonale.

### Dérivation : l'AUC est une probabilité

Soit $S^+$ le score d'un positif tiré au hasard, de densité $f_+$, et $S^-$ le score d'un négatif au hasard, de densité $f_-$. Alors

$$
\mathrm{TPR}(t)=P(S^+>t),\qquad \mathrm{FPR}(t)=P(S^->t).
$$

Quand $t$ diminue, la courbe se déplace vers la droite ; le pas infinitésimal dans la direction horizontale est $d\,\mathrm{FPR}=-f_-(t)\,dt$. L'aire sous la courbe est donc

$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}\;d\,\mathrm{FPR}
=\int_{-\infty}^{\infty} P(S^+>t)\,f_-(t)\,dt .
$$

Le membre de droite se lit : « tire un score négatif $t$ selon $f_-$, et demande-toi avec quelle probabilité un positif le bat ». Moyenner sur $t$ donne

$$
\mathrm{AUC}=P(S^+>S^-)\quad(\text{plus }\tfrac12P(S^+=S^-)\text{ for ties}).
$$

Pour un échantillon fini, la même affirmation est un comptage : parmi les $n_+n_-$ paires (positif, négatif), compte les paires où le positif a le score le plus élevé (les égalités comptent pour moitié) et divise. Cela égale aussi l'aire des trapèzes sous l'escalier ROC empirique. Conséquences :

- $\mathrm{AUC}=0.5$ est un classement aléatoire ; $\mathrm{AUC}<0.5$ signifie que le classement est à l'envers (inverse le signe).
- Elle ne dépend que de l'*ordre* des scores, donc toute transformation croissante de ceux-ci la laisse inchangée. C'est exactement pourquoi l'AUC ne dit rien de la calibration.
- Elle ne dépend pas du rapport entre classes, ce qui est à la fois une force et une faiblesse (section suivante).

### Exemple détaillé 2 : une ROC à six points

Étiquettes $y=(1,1,0,1,0,0)$ avec scores $s=(0.9,0.8,0.7,0.6,0.4,0.2)$. Il y a 3 positifs et 3 négatifs, donc le TPR avance par pas de $1/3$ et le FPR par pas de $1/3$. En descendant les scores triés, on obtient les points ROC

$$
(\mathrm{FPR},\mathrm{TPR}):\ (0,0),(0,\tfrac13),(0,\tfrac23),(\tfrac13,\tfrac23),(\tfrac13,1),(\tfrac23,1),(1,1).
$$

L'aire des trapèzes est $\tfrac13\cdot\tfrac23+\tfrac13\cdot1+\tfrac13\cdot 1=\tfrac89$. Comptage de paires : positifs $\{0.9,0.8,0.6\}$ contre négatifs $\{0.7,0.4,0.2\}$ ; le positif gagne $3+3+2=8$ des $9$ paires, donc $8/9\approx0.889$. Même réponse, comme la dérivation le promettait.

## Courbes précision-rappel et déséquilibre

La courbe ROC utilise $\mathrm{FPR}=FP/(FP+TN)$. Quand les négatifs sont bien plus nombreux que les positifs, un grand nombre de fausses alarmes déplace à peine le FPR, donc la ROC/AUC peut sembler excellente alors que la précision est mauvaise. Davis et Goadrich (tels que rapportés dans le chapitre de DMLS sur le déséquilibre) ont proposé la **courbe précision-rappel**, qui trace la précision en fonction du rappel quand le seuil bouge et ignore complètement les vrais négatifs. Un classifieur aléatoire a une courbe PR au niveau du taux de positifs (par exemple 0.01) plutôt que 0.5, donc les améliorations sont plus faciles à voir. Pour l'exemple à six points, les seuils $0.85, 0.65, 0.5, 0.3$ donnent (précision, rappel) de $(1,\tfrac13)$, $(\tfrac23,\tfrac23)$, $(0.75,1)$, $(0.6,1)$.

## Calibration

Un modèle probabiliste est **calibré** si, parmi tous les cas où il annonce $p$, l'événement se produit une fraction $p$ du temps. Le chapitre de DMLS utilise l'exemple d'une équipe prédite gagnante avec une probabilité de 70 % mais ne gagnant que 60 % des matchs, et d'une annonce dont la chance de clic prédite de 10 % est en réalité de 5 % : le classement est correct mais la prévision de clics est fausse d'un facteur deux.

**Diagramme de fiabilité.** Découpe $[0,1]$ en intervalles ; dans chacun trace la probabilité prédite moyenne (x) contre la fraction observée de positifs (y). Un modèle calibré se situe sur la diagonale. L'**erreur de calibration espérée** résume les écarts :

$$
\mathrm{ECE}=\sum_{b}\frac{n_b}{N}\,\bigl|\mathrm{freq}_b-\mathrm{conf}_b\bigr| .
$$

**Score de Brier.** L'erreur quadratique moyenne entre la probabilité prédite et le résultat 0/1, $\frac1N\sum_i(p_i-y_i)^2$. Il récompense à la fois la netteté et l'honnêteté ; plus bas est meilleur ; toujours prédire le taux de base est la référence.

### Exemple détaillé 3 : intervalles de calibration

Quinze prédictions : cinq à $0.1$ (dont une positive), cinq à $0.9$ (quatre positives), cinq à $0.6$ (deux positives). Les fréquences par intervalle sont $0.2, 0.4, 0.8$ contre des confiances $0.1, 0.6, 0.9$. Chaque intervalle a un poids $1/3$ :

$$
\mathrm{ECE}=\tfrac13(0.1+0.2+0.1)=0.1333.
$$

Le score de Brier est $\frac1{15}(0.85+1.40+0.85)=0.2067$ (le groupe à 0.1 contribue $4(0.01)+0.81=0.85$, le groupe à 0.6 $2(0.16)+3(0.36)=1.40$, le groupe à 0.9 $4(0.01)+0.81=0.85$).

Si un modèle classe bien mais est mal calibré, tu peux corriger les probabilités après coup en ajustant une application monotone (par exemple une régression logistique sur le score) sur un jeu mis de côté ; comme l'application est croissante, l'AUC est inchangée. Note aussi une conséquence subtile de Bishop : si tu rééquilibres le jeu d'entraînement (disons 50/50 au lieu de 1/1000), les probabilités a posteriori sont décalées par les a priori de classe, et tu dois diviser par les fractions de classes d'entraînement et multiplier par les vraies, puis renormaliser, pour obtenir des probabilités calibrées pour la vraie population.

## Métriques de régression

Pour des cibles $y_i$ et des prédictions $\hat y_i$, avec $e_i=\hat y_i-y_i$ :

$$
\mathrm{MAE}=\frac1N\sum|e_i|,\qquad \mathrm{RMSE}=\sqrt{\frac1N\sum e_i^2},\qquad R^2=1-\frac{\sum e_i^2}{\sum(y_i-\bar y)^2}.
$$

La MAE est dans les unités de la cible et traite chaque unité d'erreur de la même façon (c'est la perte dont l'optimum est la médiane). La RMSE est aussi dans les unités de la cible mais élève d'abord les erreurs au carré, donc quelques grosses erreurs dominent (son optimum est la moyenne). Le $R^2$ compare ton erreur quadratique à celle du prédicteur trivial « toujours $\bar y$ » : $R^2=1$ est parfait, $0$ n'est pas meilleur que la moyenne, négatif est pire.

### Exemple détaillé 4

$y=(3,5,2,7)$, $\hat y=(2.5,5,4,8)$. Les erreurs sont $(-0.5,0,2,1)$. MAE $=3.5/4=0.875$. MSE $=(0.25+0+4+1)/4=1.3125$, RMSE $=1.1456$. $\bar y=4.25$, $\sum(y-\bar y)^2=1.5625+0.5625+5.0625+7.5625=14.75$, $\sum e^2=5.25$, donc $R^2=1-5.25/14.75=0.644$. Une seule grosse erreur (le 2 au troisième point) contribue pour l'essentiel à l'erreur quadratique, ce qui explique que la RMSE (1.15) soit nettement plus grande que la MAE (0.875).

## Au-delà d'un seul nombre : ce qu'ajoute DMLS

- **Évaluation par tranches.** Un nombre global cache des sous-groupes. Dans l'exemple du livre, le modèle A a 98 % de précision sur la majorité (90 % des données) et 80 % sur la minorité, donc 96.2 % au global ; le modèle B a 95 % sur les deux, donc 95 % au global. La métrique globale préfère A, mais A est bien pire pour la minorité. Le paradoxe de Simpson montre que l'agrégation peut même inverser une comparaison. Regarde toujours les tranches qui comptent (utilisateurs payants, trafic mobile, groupes protégés).
- **Tests de perturbation** : ajoute un bruit réaliste aux entrées de test et regarde comment la métrique se dégrade. **Tests d'invariance** : changer seul un attribut sensible ne devrait pas changer la sortie. **Tests directionnels** : augmenter la taille du terrain ne devrait pas faire baisser le prix prédit d'une maison.
- Idéalement l'évaluation en développement et en production devrait coïncider ; quand les étiquettes arrivent tard, il faut surveiller à la place.

## Choisir la métrique à partir du coût métier

1. Note le coût (ou le bénéfice) de chaque case de la matrice de confusion en argent, en dommage ou en temps.
2. Si la décision est une action oui/non avec des coûts connus, utilise le seuil de coût espéré ci-dessus et rapporte le **coût espéré total** à ce seuil.
3. Si l'usage en aval est un classement (liste top-k, file de revue), utilise l'AUC ou la précision à k.
4. Si la sortie est une probabilité qui sera multipliée ou sommée (revenu, risque, nombre d'événements), exige la **calibration** (ECE, Brier).
5. Si les positifs sont rares, rapporte précision-rappel, pas l'accuracy ni la ROC seule.
6. En régression, choisis la MAE quand toutes les erreurs comptent proportionnellement et que les valeurs aberrantes sont du bruit, la RMSE quand les grosses erreurs sont disproportionnellement graves, une erreur relative quand une erreur de 100 % sur une grande valeur compte davantage.
7. Compare à une référence et rapporte l'incertitude ; ne règle pas sur le jeu de test.

## Implémentation

```python
import numpy as np

def roc_curve(y, s):
    order = np.argsort(-s, kind="stable")
    y_s, s_s = y[order], s[order]
    last = np.r_[np.where(np.diff(s_s) != 0)[0], len(s_s) - 1]   # one point per distinct score
    tpr = np.r_[0, np.cumsum(y_s)[last] / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y_s)[last] / (1 - y).sum()]
    return fpr, tpr

def auc_pairs(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
```

## Formes

`y: (N,)` de 0/1, `scores: (N,)`, `fpr, tpr: (T+1,)` où $T$ est le nombre de scores distincts, matrice de comparaison deux à deux `(n_pos, n_neg)`. Les intervalles de calibration renvoient trois tableaux de longueur (nombre d'intervalles non vides).

## Idées reçues

- **« Une accuracy élevée signifie un bon modèle. »** Pas quand les classes sont déséquilibrées ; compare avec la référence de la classe majoritaire.
- **« AUC = 0.9 signifie 90 % d'accuracy. »** C'est une probabilité sur des *paires*, indépendante de tout seuil et de la calibration.
- **« Un modèle bien calibré est précis. »** Un modèle qui produit toujours le taux de base est parfaitement calibré et inutile pour classer.
- **« Le seuil devrait être 0.5. »** Le seuil optimal dépend des coûts (et du taux de base) ; c'est une décision, pas une propriété du modèle.
- **« Le F1 est symétrique. »** Il change si tu échanges la classe positive ; de plus il ignore les vrais négatifs.
- **« RMSE, MAE et R carré classent les modèles de la même façon. »** Ils peuvent diverger, surtout en présence de valeurs aberrantes.

## Points clés à retenir

- Pars de la matrice de confusion et du coût de chaque type d'erreur ; l'accuracy est rarement le bon étalon.
- L'AUC est la probabilité qu'un positif au hasard batte un négatif au hasard ; elle ne mesure que le classement. Utilise précision-rappel quand les positifs sont rares.
- La calibration (diagramme de fiabilité, ECE, Brier) est distincte du classement et essentielle quand les probabilités sont utilisées comme des nombres.
- Le seuil optimal en coût est $c_{FP}/(c_{FP}+c_{FN})$ sur une probabilité calibrée.
- Rapporte des tranches et des références, pas seulement un nombre global.

## Explique-le toi-même

1. Pourquoi deux modèles d'accuracy identique peuvent-ils être très différents, et quels nombres montrerais-tu à un médecin ?
2. Explique avec tes propres mots pourquoi l'aire sous la courbe ROC égale la probabilité qu'un positif batte un négatif.

## Pour aller plus loin

- *Designing Machine Learning Systems*, Ch. 6 (évaluation hors ligne) et Ch. 4 (métriques sous déséquilibre de classes).
- Bishop, *PRML*, Sec. 1.5 (théorie de la décision, matrice de perte, option de rejet).
- Bibliothèque source : `ai-engineering-from-scratch` phase 02, leçon 09.

## Critères de maîtrise

Tu sais calculer à la main n'importe quelle métrique de la matrice de confusion, tracer une courbe ROC à partir de scores, démontrer et utiliser la lecture par paires de l'AUC, construire un tableau de fiabilité, dériver le seuil de coût, et justifier la métrique pour un problème métier donné.
