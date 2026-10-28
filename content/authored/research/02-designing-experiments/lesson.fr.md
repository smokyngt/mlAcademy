---
id: res.02
title: "Concevoir et évaluer des expériences originales"
summary: "De la question de recherche à l'hypothèse, aux variables, aux contrôles et aux lignes de base ; taille d'échantillon et puissance (établies et simulées) ; randomisation ; comparaisons multiples avec Bonferroni et Benjamini-Hochberg ; conception d'ablations et interactions ; budgets de calcul ; préenregistrement, p-hacking et rapports honnêtes."
minutes: 95
teaches: [res.design]
requires: [res.reproduction, res.ablation, stat.testing]
sources:
  - "content/authored/llm-systems/05-research-practice/lesson.md (Designing a controlled experiment, Reporting honestly, Spotting weaknesses): the one-factor-at-a-time ablation, paired seeds and the checklist of weak papers that this lesson turns into a planning procedure. READ."
  - "content/authored/math/13-estimation-and-hypothesis-testing/lesson.md (Confidence intervals, Hypothesis tests and p-values, Pitfalls on multiple comparisons, Comparing two models fairly): the t interval, p-value definition and the 1-(1-alpha)^m calculation, used and extended here. READ."
  - "content/authored/classical-ml/03-generalization/lesson.md (measuring protocol: train/validation/test, cross-validation): why the test set is touched once, the root of the forking-paths problem. READ."
  - "content/authored/research/01-reproducing-a-result/lesson.md (the previous lesson, res.01: report template, moons dropout reproduction with 10 seeds and the resampled-data variant): the experiment whose variance is used for the sample-size example. READ."
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (markdown results table on MATH-500, 500 problems; accuracies and run times of 12 configurations, e.g. self-consistency n=3: 29.6 percent / 97.6 min, n=5: 27.8 percent / 116.8 min, n=10: 31.6 percent / 300.4 min, and the combination with CoT up to 862.6 min): used for the sampling error of an accuracy, the minimum detectable difference on 500 problems and compute budgeting. READ."
  - "Designing Machine Learning Systems.pdf, Ch.6 (PDF p.201-205 Model Offline Evaluation, Baselines: random, simple heuristic, zero-rule, human, existing solutions; p.208 slice-based evaluation, majority/minority example): baselines that give a number meaning and the danger of coarse aggregate metrics. READ."
---

## Où cela s'insère

[[res.01]] a vérifié l'affirmation de quelqu'un d'autre. Maintenant tu veux poser ta *propre* question : un nouveau calendrier de taux d'apprentissage aide-t-il ? Mon changement de récupération vaut-il la latence supplémentaire ? La difficulté est que tu contrôles désormais chaque choix, et que chaque choix est une occasion de te tromper toi-même. Tu pourrais lancer vingt variantes, regarder vingt métriques sur trois sous-ensembles de données, et trouver presque à coup sûr quelque chose de « significatif ». La conception d'expériences est la discipline qui consiste à décider de la question, de la comparaison, du nombre d'exécutions et de l'analyse **avant** de voir les données, pour que le résultat signifie ce qu'il semble signifier.

Cette leçon s'appuie sur les statistiques de [[math.13]] (intervalles, p-valeurs, mise en garde sur les comparaisons multiples), sur le protocole de généralisation de [[cml.03]] et sur l'ablation à graines appariées de [[sys.05]]. Elle va plus loin dans quatre directions : combien d'exécutions il te faut (puissance), que faire quand tu testes beaucoup de choses (Bonferroni, Benjamini-Hochberg), comment organiser une ablation (grilles et interactions), et comment rester honnête quand la réponse est « il ne s'est rien passé ».

## L'idée en mots simples

Imagine un test de dégustation entre deux marques de café. Un mauvais plan : tu goûtes chacune une fois, des matins différents, et tu décides d'après ton humeur. Un bon plan : tu décides à l'avance ce que « meilleur » veut dire (note de goût sur dix), un ami verse les tasses dans un ordre aléatoire sans te dire laquelle est laquelle, tu goûtes assez de paires pour qu'une vraie différence apparaisse, et tu écris la règle du verdict avant la première gorgée. Puis il se passe quelque chose de plus subtil quand tu as le droit de jeter un œil. Si après chaque tasse tu peux décider de t'arrêter, ou de changer la question en « lequel sent meilleur ? », ou « lequel est meilleur pour les grands seulement ? », tu finiras par trouver une différence même entre deux tasses de la *même* cafetière. Ce n'est pas de la malhonnêteté envers le café, c'est l'arithmétique des nombreuses chances.

Cinq idées portent la leçon :

- **Une question à laquelle on peut répondre.** Une hypothèse dit ce qui compterait comme réponse et ce qui compterait contre.
- **Contrôles et lignes de base.** Change une chose, garde tout le reste identique, et compare à une ligne de base qui a eu une chance égale.
- **Assez d'échantillons.** Trop peu d'exécutions et tu manqueras de vrais effets (faible puissance). Calcule le nombre qu'il te faut *avant* de lancer.
- **Compter tes chances.** Chaque comparaison supplémentaire est un billet de loterie. Soit tu achètes moins de billets (préenregistrement), soit tu relèves la barre pour chacun (corrections de comparaisons multiples).
- **Rapporter toute l'histoire.** Y compris ce que tu as essayé, ce qui a échoué, et l'ampleur de l'effet, pas seulement s'il a franchi une ligne.

## De la question à l'hypothèse

Écris la chaîne explicitement :

1. **Question de recherche** (ouverte) : « La décroissance cosinus du taux d'apprentissage aide-t-elle ce modèle ? »
2. **Hypothèse** (précise, directionnelle si tu peux la défendre) : « Une décroissance cosinus jusqu'à zéro sur toute l'exécution donne une précision de validation moyenne plus élevée qu'un taux constant de même valeur de pic, pour ce modèle et ce jeu de données. »
3. **Définitions opérationnelles** : métrique (précision de validation à l'étape finale), unité de réplication (une exécution d'entraînement complète avec sa graine), ce qui compte comme « plus élevé » (une différence appariée dont l'intervalle à 95 % exclut 0), et la plus petite différence qui t'importerait, l'**effet minimal d'intérêt** $\delta$ (disons 0.5 point de précision).
4. **Règle de décision** écrite avant l'exécution : « Adopter la décroissance cosinus si l'intervalle apparié exclut 0 *et* que le gain moyen est d'au moins $\delta$. »

L'effet minimal d'intérêt n'est pas décoratif. Il te dit combien d'exécutions planifier (sections suivantes) et empêche la fin classique « statistiquement significatif, pratiquement sans importance ».

## Variables, contrôles et lignes de base

- **Variable indépendante (facteur)** : ce que tu manipules (le calendrier).
- **Variable dépendante (résultat)** : ce que tu mesures (précision de validation).
- **Variables de contrôle** : tout ce qui est maintenu fixe (modèle, données, étapes, taille de lot, taux d'apprentissage de pic, jeu d'évaluation, précision). Écris-les ; c'est le tableau de dispositif de [[res.01]].
- **Variables de nuisance que tu ne peux pas fixer** : initialisation aléatoire, ordre des lots, masques de dropout, découpe des données. Tu ne les supprimes pas ; tu **randomises** dessus avec des graines et tu fais la moyenne.
- **Facteurs de confusion** : une variable qui change en même temps que le facteur et pourrait expliquer le résultat. Si les exécutions cosinus ont aussi eu plus d'étapes, les étapes sont un facteur de confusion.

**Lignes de base.** Un nombre n'a de sens qu'à côté d'une référence. Le chapitre d'évaluation hors ligne de Huyen (Ch.6, PDF p.201-205) liste celles à envisager : une ligne de base aléatoire (étant donnée la distribution des étiquettes, un classifieur qui devine obtient une précision de 0.82 et un F1 de 0.1 sur un problème 90/10, contre une précision de 0.5 et un F1 de 0.167 en devinant uniformément), une heuristique simple, la ligne de base zero-rule qui prédit toujours la classe majoritaire, une ligne de base humaine et la solution existante. Ajoute l'**alternative simple la plus forte** et le **meilleur résultat précédent** pour ta propre comparaison. La ligne de base reçoit le même effort de réglage que la nouvelle méthode ; une ligne de base sous-réglée est la façon la plus courante de fabriquer une victoire.

**Tranches.** Le même chapitre (p.208) met en garde contre le fait de vivre sur un seul agrégat : dans son exemple, un modèle à 98 % de précision sur un groupe majoritaire (90 % des données) et 80 % sur une minorité a 96.2 % au total, devant un modèle à 95 % sur les deux groupes (95 % au total). Décide avant l'exécution quelles tranches tu rapporteras ; ajouter des tranches après coup est une bifurcation (ci-dessous).

## Randomisation et graines

Randomise ce que tu ne peux pas contrôler, et utilise les mêmes tirages aléatoires entre bras quand tu le peux.

- **Les graines comme réplicats.** Une exécution complète avec une graine est une observation. Les exécutions, pas les mini-lots ni les exemples d'évaluation, sont les unités indépendantes pour une affirmation sur la *procédure d'entraînement*.
- **Nombres aléatoires communs.** Donne aux deux bras les mêmes graines (mêmes poids initiaux, même ordre des lots). La différence par graine $d_i=a_i-b_i$ annule le bruit partagé, ce qui réduit la variance de la comparaison (la statistique appariée de [[sys.05]]).
- **Randomise l'ordre d'exécution** entre bras quand des effets de matériel ou d'heure de la journée pourraient compter (une machine partagée qui ralentit l'après-midi ne doit pas toujours exécuter le bras B en dernier).
- **Décide ce qu'une graine fait varier** et dis-le (l'initialisation seulement, ou aussi la découpe des données). Dans [[res.01]], élargir la graine pour rééchantillonner le jeu de données a changé le verdict ; le bon choix dépend de la population dont tu veux parler.
- **Le bruit d'évaluation est séparé.** Sur un jeu de test fixe de $N$ exemples, la précision $\hat p$ a une erreur d'échantillonnage $\sqrt{\hat p(1-\hat p)/N}$ ([[math.13]]). Elle s'ajoute au bruit d'une exécution à l'autre plutôt que de le remplacer.

## Taille d'échantillon et puissance

**Les deux erreurs.** Avec une hypothèse nulle $H_0$ (pas de différence) et une alternative $H_1$ (une différence $\delta$) :

- Erreur de type I : rejeter $H_0$ alors qu'elle est vraie ; sa probabilité est le niveau de signification $\alpha$ (typiquement 0.05).
- Erreur de type II : ne pas rejeter $H_0$ alors que $H_1$ est vraie ; sa probabilité est $\beta$.
- La **puissance** est $1-\beta$ : la probabilité de détecter une vraie différence de taille $\delta$. Une cible courante est 0.8.

### Établissement pour une comparaison à deux échantillons

Prenons deux groupes indépendants de $n$ exécutions chacun. Le groupe A a des résultats $X\sim\mathcal N(\mu_A,\sigma^2)$, le groupe B a $Y\sim\mathcal N(\mu_B,\sigma^2)$ avec le même écart type $\sigma$, et $\delta=\mu_B-\mu_A$.

1. La différence des moyennes d'échantillon $D=\bar Y-\bar X$ a pour moyenne $\delta$ et pour variance $\sigma^2/n+\sigma^2/n=2\sigma^2/n$. Note l'erreur standard $\mathrm{se}=\sigma\sqrt{2/n}$.
2. Standardise : $Z=D/\mathrm{se}$. Sous $H_0$ ($\delta=0$) c'est $\mathcal N(0,1)$, donc le test bilatéral rejette quand $|Z|>z_{1-\alpha/2}$, où $z_q$ est le quantile d'ordre $q$ de la loi normale standard ($z_{0.975}=1.960$).
3. Sous $H_1$, $Z=(D-\delta)/\mathrm{se}+\delta/\mathrm{se}$ est $\mathcal N(\delta/\mathrm{se},\,1)$ : la courbe en cloche nulle décalée vers la droite de $\delta/\mathrm{se}$.
4. La puissance est la probabilité que cette variable décalée dépasse le seuil (en ignorant la chance négligeable de tomber sous $-z_{1-\alpha/2}$) :

$$
1-\beta \;\approx\; P\big(Z>z_{1-\alpha/2}\big)=\Phi\!\Big(\frac{\delta}{\mathrm{se}}-z_{1-\alpha/2}\Big).
$$

5. Applique $\Phi^{-1}$ aux deux membres : $z_{1-\beta}=\delta/\mathrm{se}-z_{1-\alpha/2}$, donc $\delta/\mathrm{se}=z_{1-\alpha/2}+z_{1-\beta}$.
6. Substitue $\mathrm{se}=\sigma\sqrt{2/n}$ et résous pour $n$ :

$$
n \;=\; \frac{2\,\sigma^{2}\,\big(z_{1-\alpha/2}+z_{1-\beta}\big)^{2}}{\delta^{2}}\quad\text{per group.}
$$

Avec $\alpha=0.05$ et une puissance de 0.8, $z_{1-\alpha/2}=1.960$ et $z_{0.8}=0.842$, donc $(1.960+0.842)^2=7.85$ et $n\approx 15.7\,(\sigma/\delta)^2$. Lis la formule : diviser par deux l'effet à détecter **quadruple** les exécutions ; doubler le bruit les quadruple aussi. Pour un plan **apparié** avec des différences d'écart type $\sigma_d$, les mêmes étapes donnent $n=(z_{1-\alpha/2}+z_{1-\beta})^2\sigma_d^2/\delta^2$ paires.

La formule utilise la loi normale ; avec un petit $n$ le test utilise une loi $t$ à queues plus lourdes, donc la vraie puissance est un peu plus basse. C'est pourquoi nous **simulons** aussi.

### Puissance par simulation

Quand le test ou le modèle de données est délicat, estime la puissance directement : simule l'expérience de nombreuses fois sous $H_1$ et compte combien de fois elle rejette.

```python
def simulated_power(delta, sigma, n, n_sims=2000, seed=0):
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sims):
        a = rng.normal(0.0, sigma, n)
        b = rng.normal(delta, sigma, n)
        t, df = welch_t(a, b)                      # Welch statistic and degrees of freedom
        rejections += abs(t) > t_crit(int(round(df)))
    return rejections / n_sims
```

Poser `delta=0` estime à la place l'erreur de type I ; elle devrait sortir proche de $\alpha$. C'est un bon test du simulateur lui-même.

### Chiffres (reproduits par le premier exercice)

Pour $\delta/\sigma=0.5$ (un effet « moyen») la formule donne $n=63$ par groupe ; pour $\delta/\sigma=0.2$ elle donne $393$, et pour $0.8$ elle donne $25$. Puissance simulée avec 4 000 simulations :

| $n$ par groupe | 10 | 32 | 63 | 100 |
|---|---|---|---|---|
| puissance à $\delta/\sigma=0.5$ | 0.167 | 0.499 | 0.798 | 0.936 |

Et avec $\delta=0$ et $n=10$ le taux de rejet est de 0.046, proche du 0.05 nominal. Le message : dix exécutions par bras détectent un effet d'un demi-sigma environ une fois sur six. Un résultat « non significatif » avec 10 graines est donc une preuve faible d'absence d'effet, un point soulevé dans [[sys.05]] qui a maintenant un chiffre.

**En utilisant l'expérience de la leçon précédente.** Quand le jeu de données était rééchantillonné pour chaque graine, les différences appariées de l'écart avaient pour moyenne 0.0082 et pour écart type $\sigma_d=0.0259$. La formule appariée donne $n=7.85\times0.0259^2/0.0082^2\approx 80$ paires pour une puissance de 80 % ; dix graines étaient bien trop peu pour trancher cette question.

**Les jeux d'évaluation sont aussi de petits échantillons.** Dans le tableau d'évaluation de reasoning-from-scratch (500 problèmes MATH-500), les précisions se regroupent autour de 0.3. L'erreur d'échantillonnage d'une précision est d'environ $\sqrt{0.3\times0.7/500}=0.020$ (2 points). Pour deux configurations indépendantes, l'erreur standard de la différence est $\sqrt{2\times0.21/500}=0.029$, donc à 80 % de puissance la *différence minimale détectable* est d'environ $2.80\times0.029\approx 8$ points. Vu ainsi, les lignes d'auto-cohérence du tableau (29.6 % pour $n=3$, 27.8 % pour $n=5$, 31.6 % pour $n=10$, chacune issue d'une seule évaluation) diffèrent de 2 à 4 points, bien à l'intérieur de ce que 500 problèmes et une graine d'échantillonnage peuvent résoudre, donc aucun de ces ordres n'est établi. Pour détecter une différence de 1.8 point avec la même formule il faudrait environ 10 000 problèmes par bras. (Évaluer sur les mêmes problèmes et apparier aide ; le calcul ici est le pire cas non apparié.)

## Comparaisons multiples

Si tu testes $m$ hypothèses au niveau $\alpha$ et que toutes les nulles sont vraies, la chance d'au moins un faux positif (le **taux d'erreur par famille**, FWER) est $1-(1-\alpha)^m$ quand les tests sont indépendants ([[math.13]]) : 0.226 pour $m=5$, 0.642 pour $m=20$.

### Bonferroni

Teste chaque hypothèse au niveau $\alpha/m$. Par la borne de l'union $P(\bigcup_i A_i)\le\sum_i P(A_i)$, le FWER est au plus $m\cdot\alpha/m=\alpha$, sans hypothèse d'indépendance. C'est simple et sûr, et conservateur quand $m$ est grand.

### Benjamini-Hochberg (BH)

Parfois tu n'as pas besoin de zéro faux positif, seulement d'une *fraction* maîtrisée. Le **taux de fausses découvertes** (FDR) est la proportion attendue de faux positifs parmi les rejets. La procédure ascendante de Benjamini-Hochberg au niveau $q$ :

1. Trie les p-valeurs par ordre croissant : $p_{(1)}\le p_{(2)}\le\dots\le p_{(m)}$.
2. Trouve le plus grand rang $k$ tel que $p_{(k)}\le q\,k/m$.
3. Rejette les hypothèses des $k$ plus petites p-valeurs (aucune si aucun rang ne passe).

Sa garantie (pour des p-valeurs indépendantes, et sous certaines dépendances positives) : FDR $\le q$. Les seuils $qk/m$ croissent avec le rang, donc la procédure est plus souple que Bonferroni pour la queue ordonnée. « Ascendante » signifie qu'une p-valeur plus grande qui passe son seuil sauve toutes les plus petites, même si certaines échouent à leurs propres seuils.

**Exemple travaillé.** Huit p-valeurs issues de huit ablations : 0.001, 0.008, 0.012, 0.021, 0.034, 0.041, 0.30, 0.62.

| Rang | $p_{(k)}$ | Seuil BH $0.05k/8$ | Passe ? |
|---|---|---|---|
| 1 | 0.001 | 0.00625 | oui |
| 2 | 0.008 | 0.01250 | oui |
| 3 | 0.012 | 0.01875 | oui |
| 4 | 0.021 | 0.02500 | oui |
| 5 | 0.034 | 0.03125 | non |
| 6 | 0.041 | 0.03750 | non |
| 7 | 0.30 | 0.04375 | non |
| 8 | 0.62 | 0.05000 | non |

Le plus grand rang qui passe est 4, donc BH rejette les quatre plus petites p-valeurs. Bonferroni utilise le seuil unique $0.05/8=0.00625$ et ne rejette que $p=0.001$. Un test non corrigé à 0.05 « trouverait » six effets. Lequel utiliser ? Bonferroni quand un faux positif coûte cher (une affirmation dans un article qui doit tenir seule) ; BH quand tu fais un criblage de nombreux candidats et peux tolérer une fraction de fausses pistes que tu confirmeras plus tard. (La procédure descendante de Holm conserve la garantie de Bonferroni tout en rejetant au moins autant d'hypothèses ; nous ne l'implémentons pas ici.)

## Conception d'ablations

Une **ablation** retire ou change un composant pour savoir combien il contribue ([[sys.05]]). Deux dispositions :

- **Un facteur à la fois (OFAT)** : partir de la ligne de base et changer un facteur par exécution. Pour des facteurs à $l_1,\dots,l_k$ niveaux cela demande $1+\sum_i (l_i-1)$ configurations. Bon marché, facile à lire, mais aveugle aux interactions.
- **Grille complète (factorielle)** : toutes les combinaisons, $\prod_i l_i$ configurations. Elle révèle les interactions et coûte exponentiellement plus.

**Interactions.** Deux facteurs interagissent quand l'effet de l'un dépend du niveau de l'autre. Dans un tableau $2\times2$ de résultats moyens $y_{ij}$ (facteur A au niveau $i$, facteur B au niveau $j$, le niveau 0 est « désactivé »), l'interaction est

$$
I=(y_{11}-y_{10})-(y_{01}-y_{00}).
$$

Exemple : ligne de base $y_{00}=0.80$, A seul $y_{10}=0.80$, B seul $y_{01}=0.80$, les deux $y_{11}=0.90$. OFAT rapporterait que ni A ni B ne fait rien et les écarterait tous deux ; la grille montre un gain de $+0.10$ seulement quand les deux sont activés, $I=0.10$. Si les effets étaient additifs (0.70, 0.75, 0.80, 0.85) alors $I=0$.

**Tableau de planification.** Quatre facteurs à 2, 2, 3 et 2 niveaux, 5 graines, 3 minutes par exécution :

| Disposition | Configurations | Exécutions | Minutes |
|---|---|---|---|
| OFAT | $1+1+1+2+1=6$ | 30 | 90 |
| Grille complète | $2\cdot2\cdot3\cdot2=24$ | 120 | 360 |

Avec un budget de 240 minutes la grille ne rentre pas ; elle rentrerait avec 3 graines ($24\times3\times3=216$ minutes), mais 3 graines donnent un intervalle très large. Un compromis pratique est OFAT d'abord pour trouver les facteurs qui comptent, puis une grille complète seulement sur ceux-là ; les plans factoriels fractionnaires (exécuter un sous-ensemble soigneusement choisi de la grille) sont l'étape suivante standard si tu en as besoin (non implémentés ici). Rappelle-toi que chaque configuration a besoin du *même* nombre de graines, et que chaque comparaison supplémentaire entre dans le décompte des comparaisons multiples.

## Budgétisation du calcul

Coût total $=\text{configurations}\times\text{seeds}\times\text{minutes per run}$. Alloue par étapes :

1. **Pilote** (environ 10 % du budget) : quelques graines sur la plus petite configuration pour estimer $\sigma$, chercher des bugs et vérifier la métrique.
2. **Calcul de puissance** avec le $\sigma$ du pilote : combien de graines ton effet minimal d'intérêt demande-t-il ? Si la réponse dépasse le budget, change le plan : apparie les bras, augmente $\delta$ (pose une plus grande question), réduis le bruit (plus de données d'évaluation, métrique à variance plus faible), ou accepte que l'expérience ne puisse exclure que de grands effets et dis-le.
3. **Expérience principale**, avec le script d'analyse écrit et testé sur les données du pilote.
4. **Réserve** pour les suites (confirmation d'un résultat surprenant).

Le tableau d'évaluation du chapitre reasoning-from-scratch montre à quelle vitesse le coût croît quand on achète de la qualité avec des échantillons : l'auto-cohérence avec 3, 5 et 10 échantillons plus la chaîne de pensée a pris 211.6, 452.9 et 862.6 minutes pour 42.2, 48.0 et 52.0 % de précision, en évaluations uniques sur 500 problèmes. Même une comparaison honnête de ces lignes avec plusieurs graines chacune multiplierait le coût par le nombre de graines, c'est pourquoi le plan doit fixer la question avant que le budget soit dépensé.

## Préenregistrement

**Préenregistrer** signifie écrire, avec un horodatage (un commit suffit pour un travail solo), l'hypothèse, la métrique, les données et les règles d'exclusion, la taille d'échantillon, les comparaisons, la correction des comparaisons multiples et la règle de décision, *avant* de regarder les résultats. Les analyses exploratoires sont permises et précieuses, mais étiquetées comme exploratoires. L'avantage n'est pas bureaucratique ; il supprime ta liberté de choisir l'analyse après avoir vu les données, ce qui est le moteur de la section suivante.

Un préenregistrement minimal pour une expérience de ML en solo :

```text
question        Does cosine decay raise final validation accuracy vs constant lr?
hypothesis      mean paired gain >= 0.5 points (minimum effect of interest)
design          2 arms, same peak lr, same steps; seeds 0..N-1 shared by both arms
sample size     N from power analysis (sigma_d from pilot): N = ...
primary metric  validation accuracy at final step (single, pre-declared)
secondary       training loss; reported as exploratory
analysis        paired t interval; adopt if interval excludes 0 and mean >= 0.5
multiplicity    1 primary comparison; secondary p-values via BH at q = 0.05
stop rule       fixed N; no interim looks
deviations      logged in the report
```

## Le p-hacking et le jardin des chemins qui bifurquent

Le **p-hacking** est toute analyse flexible qui augmente la chance d'atteindre « significatif ». Le **jardin des chemins qui bifurquent** désigne la version plus subtile : tu ne lances jamais vingt tests délibérément ; à chaque étape tu fais un choix d'apparence raisonnable (quelle métrique, quel sous-ensemble, écarter cette graine aberrante, essayer un taux d'apprentissage plus grand pour la nouvelle méthode, s'arrêter tôt parce que ça a l'air prometteur), et le choix dépend des données. Ce qui compte est le nombre d'analyses que tu *aurais pu* lancer si les données avaient eu un autre aspect, pas le nombre que tu as lancées.

**Simulation.** Rends le monde ennuyeux : aucun effet réel, deux groupes de 10 tirages de $\mathcal N(0,1)$. Un analyste discipliné teste un résultat préenregistré. Un analyste flexible a $k$ résultats indépendants (« bifurcations ») et rapporte l'étude si l'un est significatif à 5 %. La simulation (3 000 études simulées, test de Welch) donne :

| bifurcations $k$ | bifurcation préenregistrée significative | une bifurcation quelconque significative | $1-0.95^k$ |
|---|---|---|---|
| 1 | 0.047 | 0.047 | 0.050 |
| 5 | 0.046 | 0.219 | 0.226 |
| 20 | 0.043 | 0.621 | 0.642 |

L'analyse préenregistrée garde son taux d'erreur de 5 % ; la flexible rapporte une « découverte » dans environ trois études sur cinq alors que rien n'existe. La concordance avec $1-(1-\alpha)^k$ confirme l'arithmétique (le petit déficit vient du bruit de simulation et de la discrétisation des seuils de $t$). Les vraies bifurcations sont corrélées (les mêmes données sous-tendent différentes métriques), ce qui rend généralement l'inflation plus faible que pour des bifurcations indépendantes mais encore bien au-dessus de 5 %.

Bifurcations typiques en travail de ML : choisir la meilleure de plusieurs graines, régler les hyperparamètres de la nouvelle méthode mais pas ceux de la ligne de base, évaluer sur de nombreux checkpoints et rapporter le meilleur sur le jeu de test, choisir le sous-ensemble du benchmark après avoir vu les résultats, changer de métrique parce que l'originale était « bruitée ». Défenses : préenregistrement, une seule évaluation sur jeu de test mis de côté ([[cml.03]]), rapporter chaque variante essayée et appliquer une correction de multiplicité à toute la famille, et confirmation hors échantillon de toute surprise.

## Résultats négatifs et rapports honnêtes

- **Un résultat nul est un résultat.** « Aucune différence détectée avec 10 graines, intervalle $[-0.8,+0.6]$ points » dit que des effets supérieurs à environ un point sont improbables ; cela ne dit pas que l'effet est nul. Cite l'intervalle et l'**effet minimal détectable**.
- **Rapporte le dénominateur.** Combien de variantes, métriques, sous-ensembles et graines as-tu essayés ? Le lecteur a besoin du compte pour juger la multiplicité.
- **Sépare les résultats confirmatoires des exploratoires**, et marque les interprétations que tu n'as pas testées.
- **Rapporte les tailles d'effet avec des intervalles,** pas seulement des p-valeurs ou des étoiles.
- **Échoue bruyamment.** Si l'expérience prévue manquait de puissance, dis-le plutôt que de passer à une analyse plus flatteuse.
- **Publie le code, les graines et les configurations** pour que l'expérience puisse être reproduite comme dans [[res.01]].

Faire cela n'est pas seulement de l'hygiène éthique ; c'est ce qui rend ton propre travail ultérieur digne de confiance à tes yeux.

## Implémentation : les fonctions de planification

```python
def bonferroni(p, alpha=0.05):
    p = np.asarray(p, float)
    return p <= alpha / len(p)

def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    passing = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    reject = np.zeros(m, bool)
    if len(passing):
        reject[order[: passing.max() + 1]] = True
    return reject

def ablation_plan(levels, seeds, minutes_per_run, budget_minutes):
    ofat = 1 + sum(l - 1 for l in levels); grid = int(np.prod(levels))
    return {"ofat_configs": ofat, "grid_configs": grid,
            "ofat_minutes": ofat * seeds * minutes_per_run,
            "grid_minutes": grid * seeds * minutes_per_run,
            "grid_fits": grid * seeds * minutes_per_run <= budget_minutes,
            "max_seeds_grid": int(budget_minutes // (grid * minutes_per_run))}
```

Outils de production : `scipy.stats` et `statsmodels` fournissent des calculateurs de puissance, des tests t et des corrections de tests multiples (`multipletests` avec la méthode `fdr_bh`) ; les outils de suivi d'expériences stockent configurations et graines. Écrire toi-même les quelques lignes, comme le font les deux exercices, est la façon de comprendre ce que ces fonctions promettent.

## Formes

| Objet | Forme |
|---|---|
| résultats d'un bras | $(n,)$ |
| p-valeurs d'une famille | $(m,)$ ; BH renvoie un tableau booléen $(m,)$ dans l'ordre d'origine |
| tableau $2\times2$ de moyennes | $(2,2)$, indexé `[level of A][level of B]` |
| grille de résultats | $(l_1,\dots,l_k,\ n_\text{seeds})$ |

## Idées reçues

1. « Non significatif signifie pas d'effet. » Avec 10 exécutions par bras un effet d'un demi-sigma n'est détecté qu'environ une fois sur six ; l'absence de signification est surtout une absence de puissance.
2. « Une p-valeur plus petite signifie un effet plus grand. » Elle signifie que les données sont plus surprenantes sous $H_0$ ; avec assez d'exécutions un effet négligeable obtient une minuscule p-valeur.
3. « Bonferroni et BH contrôlent la même chose. » Bonferroni contrôle la probabilité d'un faux positif quelconque ; BH contrôle la fraction attendue de faux positifs parmi les rejets.
4. « Les ablations OFAT te disent la contribution de chaque composant. » Seulement si les composants n'interagissent pas ; l'exemple $2\times2$ ci-dessus a deux composants qui paraissent chacun inutile seul.
5. « Je n'ai lancé qu'un test, donc pas de multiplicité. » Si le choix de ce test dépendait des données que tu as vues, le décompte pertinent est le nombre que tu aurais pu lancer.
6. « Le préenregistrement interdit l'exploration. » Il sépare l'exploration de la confirmation ; il n'empêche ni l'une ni l'autre.
7. « L'analyse de puissance demande le vrai effet, que je ne connais pas. » Tu utilises le *plus petit effet qui t'importerait*, plus une estimation de variance d'un pilote ; c'est une décision, pas une conjecture sur la nature.

## Points clés à retenir

- Énonce l'hypothèse, la métrique, l'effet minimal d'intérêt et la règle de décision avant de lancer.
- Contrôle tout sauf le facteur ; randomise les variables de nuisance avec des graines ; utilise des nombres aléatoires communs.
- $n=2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2/\delta^2$ par groupe ; diviser $\delta$ par deux quadruple $n$. Simule la puissance en cas de doute, et vérifie que le simulateur donne $\alpha$ sous $H_0$.
- De nombreuses comparaisons demandent une correction : Bonferroni (FWER) ou Benjamini-Hochberg (FDR).
- OFAT est bon marché mais rate les interactions ; les grilles les révèlent à un coût multiplicatif ; budgète les exécutions comme configurations x graines x minutes.
- Une analyse flexible gonfle les faux positifs vers $1-(1-\alpha)^k$ ; le préenregistrement et un seul test confirmatoire l'empêchent.
- Rapporte les nuls avec des intervalles et des effets minimaux détectables, et dis combien de choses tu as essayées.

## Explique-le toi-même

1. Ton pilote donne $\sigma_d=0.02$ pour la différence de précision appariée entre deux recettes d'entraînement et tu tiens à $\delta=0.005$. Calcule le nombre de graines pour 80 % de puissance et explique ce que tu ferais si cela dépasse ton budget.
2. Explique, avec la simulation des chemins qui bifurquent, pourquoi un analyste qui rapporte honnêtement « j'ai essayé plusieurs métriques et l'une était significative à $p=0.03$ » n'a pas fourni de preuve d'un effet, et ce que l'article aurait dû dire.

## Pour aller plus loin

- Benjamini et Hochberg, « Controlling the False Discovery Rate » (1995), l'article original de BH.
- Gelman et Loken, « The garden of forking paths » (2013).
- Ioannidis, « Why Most Published Research Findings Are False » (2005).
- Cohen, *Statistical Power Analysis for the Behavioral Sciences*, sur les tailles d'effet et la puissance.
- Bouthillier et al., « Accounting for Variance in Machine Learning Benchmarks » (2021).

## Critères de maîtrise

- [ ] Je sais passer d'une question de recherche à une hypothèse, une métrique opérationnelle, un effet minimal d'intérêt et une règle de décision.
- [ ] Je sais établir $n=2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2/\delta^2$ et dire pourquoi il évolue comme $1/\delta^2$.
- [ ] Je sais estimer la puissance par simulation et vérifier le simulateur à $\delta=0$.
- [ ] Je sais appliquer Bonferroni et Benjamini-Hochberg à la main et en code, et expliquer quelle garantie chacun donne.
- [ ] Je sais choisir entre OFAT et une grille, détecter une interaction dans un tableau $2\times2$, et budgéter les exécutions.
- [ ] Je sais montrer par simulation comment l'analyse flexible gonfle les faux positifs et écrire un préenregistrement.
- [ ] Je sais rapporter un résultat nul avec un intervalle et un effet minimal détectable.
