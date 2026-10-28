---
id: ops.06
title: "Monitoring, dérive et apprentissage continu"
summary: "Pourquoi les modèles déployés échouent, comment nommer et détecter le décalage de covariables, d'étiquettes et de concept (test KS et PSI from scratch), quoi surveiller, comment l'apprentissage continu met à jour les modèles, et comment tester un nouveau modèle en production sans se tromper soi-même avec le peeking."
minutes: 80
teaches: [ops.drift, ops.monitoring, ops.continual, ops.abtest]
requires: [stat.testing, prob.rv, ops.requirements, ops.serving]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.8 Data Distribution Shifts and Monitoring: causes of ML system failures, covariate/label/concept shift, detecting shifts (summary statistics, two-sample and KS tests, time-scale windows), monitoring and observability, alerts (PDF p.264-287, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.9 Continual Learning and Test in Production: stateless retraining versus stateful training, why continual learning, the four stages, shadow deployment, A/B testing, canary release, interleaving, bandits (PDF p.290-325, read in selected parts). READ."
---

## Où cela s'insère

Le déploiement ([[ops.05]]) n'est pas la ligne d'arrivée. Le monde change : les utilisateurs changent, les concurrents changent, les systèmes en amont modifient une colonne. Un modèle qui obtenait de bons scores hors ligne ([[ops.04]]) devient lentement ou soudainement moins bon, généralement sans aucun message d'erreur. Cette leçon boucle la boucle du cours : nommer les façons dont un système échoue, détecter les décalages dans les données avec des tests statistiques, surveiller les bons signaux, mettre à jour le modèle en continu, et comparer un nouveau modèle à l'ancien en production par des expériences. Les contrôles de skew de [[ops.03]] reviennent ici sous forme de moniteurs continus, et le test z et les idées d'échantillonnage viennent des leçons de statistiques ([[ops.02]] pour l'échantillonnage, et le matériel antérieur sur les probabilités et les tests).

## L'idée en termes simples

Un prévisionniste météo entraîné sur vingt ans de données est embauché dans une ville dont le climat change. Trois choses différentes peuvent mal tourner. Les types de journées qui se produisent peuvent changer (plus de canicules que pendant les années d'entraînement), alors qu'un ensemble donné de relevés signifie toujours la même chose : c'est le **décalage de covariables** (covariate shift). Le taux de base de ce que tu prédis peut changer (la pluie est maintenant fréquente), alors que les jours pluvieux se ressemblent toujours : le **décalage d'étiquettes** (label shift). Ou bien les mêmes relevés peuvent maintenant signifier autre chose (une chute de pression qui amenait la pluie ne l'amène plus) : la **dérive de concept** (concept drift). Chacun appelle une réponse différente.

Un détecteur de fumée est un bon modèle pour le monitoring. Il ne peut pas attendre la déclaration de sinistre de l'incendie (la vraie étiquette arrive tard, voire jamais), alors il surveille des signaux indirects qui changent en premier : température, particules dans l'air, état de la pile. Pour un système de ML, les signaux indirects sont la distribution des entrées, la distribution des prédictions et la santé du service. Tester un nouveau modèle en production, c'est comme essayer une nouvelle recette : tu peux demander au cuisinier de la préparer sans la servir (shadow), la servir à quelques tables (canary), séparer la salle au hasard et comparer (A/B), ou déplacer des tables vers le meilleur plat au fur et à mesure que les preuves s'accumulent (bandit).

## Pourquoi les systèmes de ML échouent

Un échec est une attente violée du système. Deux familles :

1. **Les échecs logiciels**, que tout système connaît : dépendances cassées, mauvais binaire déployé, panne matérielle, indisponibilité, bogues de systèmes distribués et de pipelines de données. Le livre rapporte une étude de 96 échecs d'un grand pipeline de ML en production dans laquelle 60 n'étaient pas directement liés au ML.
2. **Les échecs propres au ML** :
   - **Les données de production diffèrent des données d'entraînement.** Le jeu de test est censé représenter des données jamais vues, mais le monde n'est pas stationnaire et un jeu d'entraînement fini ne peut pas le couvrir.
   - **Les cas limites.** Des entrées rares qui causent des erreurs catastrophiques (une voiture autonome qui a raison 99,99 % du temps peut rester inutilisable) ; une augmentation soudaine de cas mal gérés peut aussi signaler un décalage de distribution.
   - **Les boucles de rétroaction dégénérées.** Les prédictions du système influencent les données dont il apprendra plus tard. Un recommandeur montre d'abord la chanson A, donc A est plus cliquée, donc le système classe A encore plus haut ; le résultat est un biais de popularité et des bulles de filtres, et les nouveaux éléments ne parviennent pas à percer. L'exploration (bandits, ci-dessous) et la randomisation l'atténuent.

## Décalage de distribution des données

Soit $X$ les entrées et $Y$ les étiquettes. La loi jointe se factorise de deux façons :

$$
P(X,Y)=P(Y\mid X)\,P(X)=P(X\mid Y)\,P(Y).
$$

Chaque facteur peut changer, ce qui donne trois décalages nommés.

| Décalage | Ce qui change | Ce qui reste identique | Exemple |
|---|---|---|---|
| **Décalage de covariables** | $P(X)$ | $P(Y\mid X)$ | Plus de patients de plus de 40 ans en production qu'à l'entraînement ; la relation entre l'âge et le risque est inchangée. |
| **Décalage d'étiquettes** | $P(Y)$ | $P(X\mid Y)$ | La maladie devient plus répandue ; étant donné la maladie, les patients se ressemblent toujours. |
| **Dérive de concept** | $P(Y\mid X)$ | $P(X)$ | Les mêmes caractéristiques de maison se vendent maintenant moins cher après un changement de marché (« même entrée, sortie différente »). Souvent cyclique ou saisonnière. |

D'autres changements nuisent aussi : les unités d'une feature changent, des colonnes sont ajoutées ou retirées, ou l'ensemble des valeurs possibles change (la feature d'âge passe d'années à mois). Tous les décalages ne comptent pas ; ceux qui comptent réduisent la performance.

### Détecter les décalages

Quand les étiquettes sont disponibles rapidement, surveille directement les métriques liées à la précision. Quand elles ne le sont pas (retardées ou absentes), surveille les distributions que tu peux observer sans étiquettes : surtout $P(X)$, et la distribution des prédictions du modèle.

*Statistiques de synthèse.* Compare la moyenne, la médiane, la variance, les quantiles entre une fenêtre de référence et la fenêtre courante. Peu coûteux et répandu, mais nécessaire sans être suffisant : des moyennes égales n'impliquent pas des distributions égales.

*Tests à deux échantillons.* Traite la fenêtre de référence et la fenêtre courante comme deux échantillons et demande-toi s'ils pourraient provenir de la même distribution. Une réserve du livre : une différence statistiquement significative n'est pas forcément importante en pratique, mais une différence détectable à partir d'un petit échantillon est probablement grande.

**Test de Kolmogorov-Smirnov (KS).** Soit $F_a$ et $F_b$ les fonctions de répartition empiriques de deux échantillons de tailles $n$ et $m$. La statistique KS est le plus grand écart vertical

$$
D=\max_x\left|F_a(x)-F_b(x)\right| .
$$

Elle est non paramétrique (aucune hypothèse sur la forme) et prend des valeurs dans $[0,1]$ : 0 pour des échantillons identiques, 1 pour des échantillons qui ne se recouvrent pas. Sous l'hypothèse nulle que les deux échantillons proviennent de la même distribution continue, la statistique mise à l'échelle $\sqrt{nm/(n+m)}\,D$ a une distribution asymptotique dont la queue est

$$
P(D\text{ at least this large})\approx 2\sum_{k=1}^{\infty}(-1)^{k-1}e^{-2k^2\lambda^2},\qquad
\lambda=\Big(\sqrt{n_e}+0.12+\tfrac{0.11}{\sqrt{n_e}}\Big)D,\quad n_e=\tfrac{nm}{n+m},
$$

une correction standard pour les petits échantillons. Limites pratiques (indiquées dans le livre) : le KS fonctionne pour des données unidimensionnelles, donc tu l'appliques par feature ou sur les prédictions et les étiquettes ; il peut produire de nombreuses fausses alertes quand tu testes beaucoup de features et de grands échantillons ; pour des données de grande dimension, réduis d'abord la dimension ou utilise des tests multivariés comme la MMD.

**Indice de stabilité de la population (PSI).** Non couvert dans le livre ; c'est un score très utilisé dans l'industrie (présenté ici d'après la pratique générale). Découpe l'échantillon de référence en intervalles de masse égale à l'aide de ses quantiles (disons des déciles), calcule la proportion $p_j$ de la référence et $q_j$ du nouvel échantillon dans chaque intervalle $j$, et

$$
\mathrm{PSI}=\sum_{j}(q_j-p_j)\ln\frac{q_j}{p_j}.
$$

Chaque terme est positif ou nul (le numérateur et le logarithme ont le même signe), le PSI vaut 0 seulement si les histogrammes sont égaux, et il est égal à la somme des deux sens de la divergence de KL (une mesure symétrique de la différence entre les histogrammes). Les proportions sont écrêtées à une petite valeur pour que les intervalles vides ne donnent pas l'infini. Les règles empiriques en pratique utilisent des seuils autour de 0,1 (examiner) et 0,2-0,25 (alerte) ; ce sont des conventions, pas des lois, donc calibre-les sur tes propres données.

*Fenêtres d'échelle de temps.* Le fait que les données aient « dérivé » dépend de la fenêtre de référence. Avec un cycle hebdomadaire, comparer aujourd'hui aux six derniers jours peut ressembler à un décalage, alors que comparer aux deux dernières semaines montre que c'est de la saisonnalité normale. Les décalages peuvent être spatiaux (un nouveau groupe d'utilisateurs ou d'appareils) ou temporels, et les décalages soudains sont plus faciles à attraper que les graduels. Utilise des statistiques glissantes plutôt que cumulatives pour la récence.

### Traiter les décalages

Trois approches : entraîner sur un jeu de données large et varié en espérant couvrir la production ; adapter un modèle entraîné à la distribution cible sans nouvelles étiquettes (surtout de la recherche) ; ou réentraîner régulièrement sur des données fraîches, ce que font la plupart des entreprises. La fréquence de réentraînement est généralement une décision au pif ; les sections suivantes montrent comment en faire une décision mesurée.

## Monitoring et observabilité

Le **monitoring** consiste à suivre, mesurer et journaliser des métriques pour savoir quand quelque chose tourne mal ; l'**observabilité** (instrumentation) consiste à configurer le système pour pouvoir enquêter sur le pourquoi : minuteurs sur les fonctions, comptage des NaN dans les features, journalisation des entrées inhabituelles, traçage des entrées à travers les transformations. Sans observabilité, le monitoring n'a rien à mesurer.

Deux classes de métriques :

- **Métriques opérationnelles** : latence, débit, nombre de requêtes, part de réponses 2xx, utilisation CPU/GPU et mémoire, disponibilité (le pourcentage de temps où le service respecte son objectif de niveau de service). Quatre neuf (99,99 %) autorisent environ quatre minutes d'indisponibilité par mois.
- **Métriques propres au ML**, issues de quatre artefacts à des distances croissantes du modèle : les *entrées brutes*, les *features*, les *prédictions* et les *métriques liées à la précision*. Plus l'artefact est profond dans le pipeline, plus il est proche de ce qui t'importe et plus il est facile à interpréter, mais plus il est probable qu'un changement provienne de l'une des nombreuses transformations antérieures ; surveille donc plusieurs niveaux. Utilise tout signal de retour que tu peux capter (clics, pouces vers le bas, taux d'achèvement) ; un taux de clic qui reste plat alors que le taux d'achèvement chute suggère que les recommandations sont devenues moins bonnes.

Les *alertes* ont trois parties : une **politique d'alerte** (la condition et la durée, par exemple accuracy sous 90 %, ou latence au-dessus de 1 s pendant 10 minutes), des **canaux de notification** (qui est prévenu, où), et une **description** (ce qui s'est passé et quoi vérifier). Trop d'alertes causent de la fatigue d'alerte ; les seuils des scores de dérive ci-dessus doivent être réglés selon ta tolérance aux fausses alertes.

## Apprentissage continu

L'**apprentissage continu** (continual learning) signifie mettre en place l'infrastructure pour que les modèles puissent être mis à jour aussi vite que nécessaire, et non réentraîner constamment pour le plaisir. Deux façons de mettre à jour :

- **Réentraînement sans état** (stateless) : entraîner à partir de zéro sur une fenêtre de données (par exemple les trois derniers mois) à chaque fois.
- **Entraînement avec état** (stateful, fine-tuning) : charger le checkpoint d'hier et continuer sur les seules nouvelles données. Il demande moins de données et de calcul (le livre cite une entreprise qui a réduit le coût d'entraînement de 45 fois et augmenté son taux de conversion en achat de 20 % en passant d'un entraînement quotidien sans état à un entraînement quotidien avec état), et en principe chaque échantillon n'est utilisé qu'une fois, donc les données n'ont pas à être stockées (un avantage pour la vie privée). Un entraînement occasionnel à partir de zéro reste utilisé pour recalibrer.

*Pourquoi :* répondre aux décalages soudains (un pic de demande de courses lors d'un événement), s'adapter aux événements rares (le Black Friday a trop peu d'historique).

*Quatre stades* de maturité : (1) **réentraînement manuel sans état** chaque fois que le modèle est clairement mauvais et que quelqu'un a le temps ; (2) **réentraînement automatisé** par un script planifié, avec une fréquence choisie au feeling ; (3) **entraînement avec état automatisé**, qui nécessite le suivi de la lignée des données et des modèles ; (4) **apprentissage continu** déclenché par des décalages ou des baisses de performance plutôt que par un calendrier. Le stade 4 est une marche haute : il exige un bon monitoring, une évaluation sûre de chaque candidat (ci-dessous) et une infrastructure fiable.

## Tester en production

L'évaluation hors ligne ([[ops.04]]) ne peut pas dire comment un modèle se comporte sur du trafic réel avec un vrai retour, donc les nouveaux modèles sont testés sur le trafic de production de manière contrôlée.

- **Déploiement fantôme** (shadow) : le candidat reçoit chaque requête en parallèle mais seule la prédiction du titulaire est servie ; tu journalises et compares. Le plus sûr, mais il double le coût d'inférence.
- **Test A/B** : répartir aléatoirement les utilisateurs entre le titulaire (A) et le candidat (B), servir à chacun son modèle, et comparer une métrique avec un test statistique. La randomisation est essentielle, et la taille d'échantillon et la métrique doivent être fixées à l'avance.
- **Canary release** : envoyer une petite part du trafic au candidat, l'augmenter si ses métriques tiennent, abandonner et revenir en arrière si elles se dégradent. Il n'a pas besoin de randomisation (tu pourrais commencer par un marché).
- **Interleaving** : montrer à chaque utilisateur une liste *mélangée* contenant des éléments des deux classeurs et compter les éléments de quel modèle reçoivent des clics ; chaque utilisateur éprouve les deux modèles, ce qui fait que la comparaison nécessite moins d'utilisateurs que l'A/B (Netflix rapporte une identification fiable avec des échantillons considérablement plus petits). Convient au classement et à la recommandation.
- **Bandits** : traiter chaque modèle comme une machine à sous dont le gain est inconnu et router le trafic de façon adaptative, en équilibrant **exploration** (essayer les autres) et **exploitation** (utiliser le meilleur jusqu'ici). Ils nécessitent la prédiction en ligne et des boucles de rétroaction courtes, et ils sont avec état (il faut suivre le gain de chaque modèle). Stratégies simples : $\varepsilon$-greedy (une fraction du trafic va à un modèle au hasard) ; **échantillonnage de Thompson** (tirer un taux de succès plausible pour chaque modèle à partir de sa loi a posteriori et jouer le plus grand) ; **UCB** (jouer le modèle avec la plus haute borne supérieure de confiance optimiste). Dans une expérience citée par le livre, l'échantillonnage de Thompson a tranché une différence de 5 % avec moins de 12 000 échantillons là où un test A/B fixe en a nécessité plus de 630 000. Les bandits sont plus difficiles à implémenter et sont surtout utilisés dans les grandes entreprises.

### Le test A/B et le problème du peeking

Pour un résultat binaire, avec $x_A$ succès sur $n_A$ et $x_B$ sur $n_B$, la proportion groupée est $\hat p=(x_A+x_B)/(n_A+n_B)$, et la statistique $z$ à deux proportions pour $B-A$ est

$$
z=\frac{x_B/n_B-x_A/n_A}{\sqrt{\hat p(1-\hat p)\left(\tfrac1{n_A}+\tfrac1{n_B}\right)}} .
$$

Sous l'hypothèse nulle (taux égaux), $z$ est approximativement normale centrée réduite, et rejeter quand $|z|>1.96$ donne un taux de faux positifs de 5 % **si tu regardes exactement une fois, à une taille d'échantillon fixée à l'avance**. La taille requise par bras pour détecter une différence $\delta$ au seuil de 5 % avec une puissance de 80 % est d'environ $n\approx 2\,(1.96+0.84)^2\,p(1-p)/\delta^2$. Pour $p=0.10$ et $\delta=0.03$, cela fait $2\cdot7.84\cdot0.09/0.0009=1{,}568$ utilisateurs par bras.

Imagine maintenant que tu consultes le tableau de bord chaque jour et t'arrêtes au premier $|z|>1.96$. Chaque consultation est une nouvelle chance pour le bruit de franchir le seuil, et les tests sont corrélés mais pas identiques, donc la probabilité qu'*une* consultation le franchisse est bien plus que 5 %. C'est le **problème du peeking**. Dans un **test A/A** (les deux bras identiques, donc tout « gagnant » est un faux positif), avec 1 000 utilisateurs par bras et dix consultations également espacées, notre simulation déclare un gagnant dans environ 20 % des expériences, contre 5 % pour une seule consultation. Remèdes : fixer la taille d'échantillon et ne regarder qu'une fois ; utiliser une correction (un seuil plus strict par consultation, ou l'alpha-spending) ; utiliser de vraies méthodes séquentielles conçues pour la surveillance continue ; ou utiliser un bandit quand ton but est la récompense plutôt qu'un test d'hypothèse propre.

Au-delà du peeking : choisis une métrique liée à l'objectif métier ([[ops.01]]), vérifie que la randomisation n'a pas de bogue (contrôle du ratio d'échantillons), exécute sur des cycles hebdomadaires entiers, et méfie-toi de tester de nombreuses métriques ou variantes sans correction.

## Exemple détaillé

Toutes les valeurs viennent du code de l'exercice avec des graines fixes.

*Détection de dérive.* Référence : 1 000 tirages de $N(0,1)$. Nouveaux échantillons de 1 000 :

| Nouvel échantillon | KS $D$ | p-valeur KS | PSI | Verdict |
|---|---|---|---|---|
| même $N(0,1)$ | 0.052 | 0.13 | 0.020 | pas de dérive |
| $N(0.5,1)$ (décalage de moyenne) | 0.210 | $\approx9\times10^{-20}$ | 0.240 | dérive |
| $N(0,2)$ (plus large) | 0.191 | $\approx2\times10^{-16}$ | 0.591 | dérive ; moyenne inchangée |

La dernière ligne est la raison pour laquelle on regarde les distributions et pas seulement les moyennes : la moyenne diffère de moins de 0,2, pourtant le KS comme le PSI signalent le changement de variance.

*Test A/B.* 100 sur 1 000 contre 130 sur 1 000 donne une proportion groupée $\hat p=0.115$, une erreur standard $0.01427$, $z=2.10$, juste au-delà de 1,96. C'est seulement limite, et la taille requise était de 1 568 par bras, donc une équipe prudente collecterait davantage avant de conclure. *Peeking* : A/A, $p=0.1$, 1 000 utilisateurs par bras, 2 000 expériences simulées : une consultation déclare un faux gagnant dans 5,05 % des exécutions, dix consultations dans 20,5 % (environ quatre fois plus). *Bandit* : l'échantillonnage de Thompson avec des taux réels de 0,10 et 0,20 sur 3 000 tours ne tire le moins bon bras que 29 fois et le meilleur 2 971 fois, alors qu'un test A/B 50/50 enverrait 1 500 utilisateurs vers le moins bon bras.

## Implémentation

```python
import numpy as np

def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))

def psi(reference, new, bins=10, eps=1e-4):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])
    p = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=bins) / len(reference)
    q = np.bincount(np.searchsorted(edges, new, side="right"), minlength=bins) / len(new)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))

def z_test(xa, na, xb, nb):
    p = (xa + xb) / (na + nb)
    return float((xb / nb - xa / na) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb)))
```

Bibliothèques : `scipy.stats.ks_2samp` (p-valeurs exactes et asymptotiques), Alibi Detect et Evidently pour la détection de dérive, Prometheus/Grafana ou le monitoring cloud pour les métriques opérationnelles, et des plateformes d'expérimentation pour les tests A/B et les bandits.

## Dimensions

Les échantillons de référence et nouveaux sont des tableaux 1-D de longueurs $n$ et $m$ (une feature à la fois) ; `edges: (bins-1,)` ; proportions par intervalle `(bins,)` ; les comptes d'un tableau A/B 2x2 sont des scalaires. Dans les simulations, les matrices de résultats ont la forme `(n_sims, n_per_arm)`.

## Idées reçues

- « Si l'accuracy était bonne au lancement, elle le reste. » Les distributions bougent ; surveille en continu.
- « Pas d'erreur signifie pas d'échec. » Les systèmes de ML échouent silencieusement.
- « Toute dérive significative nécessite un réentraînement. » La significativité n'est pas l'importance ; vérifie l'impact sur la performance.
- « Les moyennes suffisent à détecter la dérive. » Un changement de variance ou de forme peut laisser la moyenne intacte.
- « Jeter un œil à un test A/B est inoffensif si je ne m'arrête que lorsque c'est clairement significatif. » Les consultations répétées multiplient les faux positifs plusieurs fois.
- « L'apprentissage continu signifie réentraîner constamment. » Cela signifie pouvoir mettre à jour aussi vite que nécessaire, en toute sécurité.
- « Les bandits battent toujours les tests A/B. » Ils nécessitent un retour rapide et la prédiction en ligne, et donnent des conclusions statistiques moins nettes.

## Points clés à retenir

- Les échecs sont soit logiciels, soit propres au ML (décalage des données, cas limites, boucles de rétroaction dégénérées).
- Le décalage de covariables change $P(X)$, le décalage d'étiquettes $P(Y)$, la dérive de concept $P(Y\mid X)$.
- Le KS (plus grand écart de fonctions de répartition) et le PSI (divergence par intervalles) détectent des décalages dans des features unidimensionnelles sans étiquettes ; compare toujours les distributions, pas seulement les moyennes.
- Surveille les métriques opérationnelles et les métriques de ML à plusieurs étapes du pipeline ; les alertes ont besoin d'une politique, d'un canal et d'une description.
- L'entraînement avec état et l'automatisation mènent vers l'apprentissage continu ; teste les candidats par shadow, canary, A/B, interleaving ou bandits.
- Fixe la taille d'échantillon avant un test A/B ; les consultations répétées gonflent le taux de faux positifs (d'environ 5 % à environ 20 % avec dix consultations).

## Explique-le avec tes mots

1. La moyenne d'une feature est inchangée mais sa variance a doublé. Quels contrôles la détectent, et lesquels non ?
2. Pourquoi consulter un test A/B chaque jour et s'arrêter au premier $p<0.05$ donne-t-il bien plus de 5 % de faux positifs, et quels sont deux remèdes ?

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, Ch.8-9 (PDF p.264-327). Documentation d'Alibi Detect pour la détection de dérive multivariée (non lue pour cette leçon).
- [[ops.04]] pour l'évaluation hors ligne et les tranches qui décident quoi surveiller.

## Critères de maîtrise

Tu sais énoncer les trois décalages en termes de $P(X)$, $P(Y)$ et $P(Y\mid X)$, implémenter et interpréter le KS et le PSI, concevoir un plan de monitoring avec des métriques opérationnelles et de ML et des alertes, expliquer les mises à jour avec et sans état, choisir entre shadow, canary, A/B, interleaving et bandits, et démontrer le problème du peeking par simulation.
