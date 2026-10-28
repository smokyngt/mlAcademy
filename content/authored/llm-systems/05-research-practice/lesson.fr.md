---
id: sys.05
title: "Pratique de la recherche : lire, reproduire et faire des ablations"
summary: "Comment lire un article en trois passes, reproduire un résultat, concevoir une expérience contrôlée (graines, références, variance), mener une ablation appariée avec des statistiques honnêtes, et repérer les affirmations fragiles, avec une ablation détaillée sur le MLP des deux lunes."
minutes: 80
teaches: [res.ablation]
requires: [ml.generalization, nn.training, llm.eval]
sources:
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (markdown cells, Sec. 4.4-4.6 and the results table): self-consistency recipe and the table of accuracies for combinations of CoT, top-p and voting on 500 problems. READ."
  - "reasoning-from-scratch/ch07/01_main-chapter-code/ch07_main.ipynb (markdown cells, Sec. 7.2-7.3): which training metrics to track, sanity metrics that should stay at zero, interpretation of curves. READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (results tables) and chF/01_main-chapter-code/chF_main.ipynb: accuracy on 10 vs 500 problems, devices and runs. READ."
  - "Designing Machine Learning Systems.pdf, Ch.6 (PDF p.201-214) baselines and evaluation methods, and Ch.9 'Test in Production' (PDF p.314-317): shadow deployment, A/B testing requirements (random assignment, enough samples), canary release. READ."
  - "content/authored/classical-ml/03-generalization/lesson.md (measuring protocol: train/validation/test, cross-validation; the experiment table): the discipline this lesson extends from one model to a comparison of two. READ (first half)."
---

## Où cela s'inscrit

Tôt ou tard, tu voudras savoir si une idée « marche » : un échauffement du taux d'apprentissage, une initialisation différente, une astuce tirée d'un nouvel article. La réponse honnête exige une petite expérience, pas une impression. La pratique de la recherche est l'ensemble des habitudes qui transforment « ça avait l'air mieux » en une preuve que tu pourrais défendre devant un collègue sceptique (ou devant toi-même dans trois mois). Elle réutilise les statistiques de la leçon précédente (intervalles, comparaisons appariées) et la discipline de la leçon sur la généralisation (ne jamais juger sur des données sur lesquelles on a réglé), et les applique à la question à laquelle tout ingénieur ML fait face : *mon changement a-t-il causé l'amélioration, ou est-ce du bruit ?*

## L'idée en mots simples

Imagine deux boulangers qui comparent des recettes. L'un cuit la recette A le lundi dans un four froid et la recette B le mardi dans un four chaud, une fois chacune, et déclare B meilleure. Tout est confondu : four, jour, farine. Un test équitable cuit les deux recettes de nombreuses fois, dans le même four, en alternance, avec la même farine, et compare les *différences* fournée par fournée. Une **ablation** est la même idée appliquée à un système : prendre le système complet, retirer (ou modifier) exactement un composant, et voir ce qui change. Si tu changes deux choses à la fois, tu ne peux pas dire laquelle a compté.

Trois autres idées :

- **La variance est réelle.** Un réseau de neurones entraîné deux fois avec des graines aléatoires différentes donne des résultats différents. Toute affirmation issue d'une seule exécution porte sur un seul tirage aléatoire.
- **Les références donnent du sens.** Un nombre n'est interprétable qu'à côté d'une référence : la méthode simple, le meilleur résultat précédent, le hasard.
- **L'honnêteté est une méthode.** Rapporte ce que tu as essayé, y compris ce qui n'a pas marché ; dis ce que la preuve soutient et ce qu'elle ne soutient pas.

## Lire un article en trois passes

Tout lire en détail est lent et la plupart des articles ne le méritent pas. Une approche par étapes (la méthode « en trois passes » que de nombreux chercheurs utilisent) fait gagner du temps :

1. **Première passe, 5 à 10 minutes.** Titre, résumé, introduction, titres de sections, figures, conclusion. Objectif : *quel problème, quelle affirmation, est-ce pertinent pour moi ?* Décide s'il faut continuer.
2. **Deuxième passe, environ une heure.** Lis le texte en portant attention aux figures, tableaux et au dispositif expérimental ; saute les preuves. Objectif : comprendre la méthode assez bien pour l'expliquer, et lister les *affirmations* avec la *preuve* offerte pour chacune (quel tableau ou quelle figure).
3. **Troisième passe, plusieurs heures.** Redériver ou réimplémenter mentalement : quelles sont les hypothèses, qu'aurais-je fait différemment, où cela pourrait-il échouer ? C'est la passe où tu reproduis une partie du résultat.

À chaque passe, garde une note de quatre lignes : l'affirmation, la preuve, la référence, ce qui manque.

## Reproduire un résultat

La reproduction est la meilleure façon de comprendre une méthode et de calibrer ta confiance.

1. Commence par la **plus petite** version censée montrer l'effet (un modèle minuscule, un sous-ensemble de données). Les notebooks reasoning-from-scratch le font : ils évaluent 10 problèmes pour une démonstration rapide et rapportent séparément les nombres complets sur 500 problèmes ; les deux diffèrent par leur fiabilité.
2. Fixe tout ce qui peut l'être : graines, découpage des données, versions des bibliothèques, notes sur le matériel. Écris la commande qui reproduit l'exécution.
3. Reproduis d'abord la **référence** (baseline). Si tu ne peux pas retrouver le nombre publié pour la référence, tu ne peux pas faire confiance à une comparaison avec elle.
4. Attends-toi à de petits écarts. Compare avec la variance propre à l'article si elle est donnée ; sinon, lance toi-même plusieurs graines.
5. **Contrôle de cohérence des métriques.** Le chapitre sur le suivi des exécutions d'entraînement en donne un exemple : avec l'avantage relatif au groupe utilisé là-bas, l'avantage moyen doit rester à zéro par construction ; s'il dérive, cela indique un bug. Trouve de tels invariants dans tes propres expériences (une perte qui doit démarrer près de $\ln K$ pour $K$ classes, des gradients qui doivent correspondre aux différences finies) et vérifie-les avant d'interpréter les résultats.
6. Enregistre les courbes, pas seulement les nombres finaux : le même chapitre note que la précision d'évaluation peut monter puis redescendre pendant l'entraînement, et que certaines métriques (comme la perte) sont moins significatives que d'autres.

## Concevoir une expérience contrôlée

**Hypothèse.** Énonce-la de façon qu'elle puisse échouer : « Un échauffement linéaire du taux d'apprentissage sur 50 pas change la précision de test de ce MLP », et non « l'échauffement aide ».

**Variables.** Une variable *manipulée* par ablation (échauffement activé/désactivé), tout le reste maintenu fixe (données, architecture, budget de pas, taux d'apprentissage, jeu d'évaluation). Pré-enregistre la métrique principale.

**Référence.** La configuration sans le changement, réglée *avec autant de soin* que la nouvelle. Une référence solide et bien réglée l'emporte sur une méthode sophistiquée comparée à une référence négligée ; une comparaison inéquitable est l'une des faiblesses les plus courantes des articles.

**Graines et variance.** Lance $n$ graines par configuration (5 est un minimum pour une vérification rapide, davantage pour de petits effets). Rapporte la moyenne $\pm$ l'écart-type (écart-type empirique, $\text{ddof}=1$) et $n$. Dis ce que contrôle la graine (initialisation, ordre des batchs, découpage des données).

**Nombres aléatoires communs (appariement).** Utilise la *même* graine pour les deux configurations, afin qu'elles partagent les poids initiaux et l'ordre des batchs. La différence entre configurations est alors mesurée *au sein* des graines, où une grande partie du bruit s'annule. Pour des différences appariées $d_i=a_i-b_i$ sur $n$ graines :

$$
\bar d=\frac1n\sum_i d_i,\quad s_d=\sqrt{\tfrac{1}{n-1}\sum_i(d_i-\bar d)^2},\quad t=\frac{\bar d}{s_d/\sqrt n}.
$$

Compare $|t|$ à la valeur critique de Student pour $n-1$ degrés de liberté (2.776 pour 4 d.d.l. au niveau bilatéral de 5 pour cent). La statistique non appariée (Welch) utilise la variance propre à chaque groupe, $t=\frac{\bar a-\bar b}{\sqrt{s_a^2/n+s_b^2/n}}$ ; quand le bruit d'une graine à l'autre est grand comparé à l'effet, elle est bien moins puissante.

**Significativité statistique contre pratique.** Un effet minuscule peut être statistiquement solide et pratiquement sans intérêt ; un grand effet avec 3 graines peut être un coup de chance. Rapporte toujours la taille d'effet (ici, des points de précision) à côté du test.

**Discipline du jeu de test.** Choisis les réglages sur les données de validation et ne touche au jeu de test qu'une fois (leçon sur la généralisation). Si tu lances 30 ablations et rapportes la meilleure sur le jeu de test, le jeu de test est devenu un jeu de validation.

**Tester en production** (Huyen, chap. 9) : le protocole hors ligne ci-dessus ne peut pas tout capturer, donc les systèmes déployés ajoutent le *déploiement fantôme* (shadow deployment : exécuter le nouveau modèle en parallèle, journaliser ses prédictions sans les servir, au prix d'un doublement de l'inférence), les *tests A/B* (affectation aléatoire du trafic ; le livre insiste sur le fait que l'affectation doit être réellement aléatoire et l'échantillon assez grand, et que la significativité n'est pas infaillible), et les *déploiements canari* (déployer d'abord sur une petite fraction et abandonner si les métriques clés se dégradent).

## Exemple détaillé : l'échauffement ou l'échelle d'initialisation comptent-ils pour le MLP des deux lunes ?

**Dispositif.** Données : 200 points d'entraînement et 1000 points de test issus de deux lunes bruitées (bruit 0.2, données fixes). Modèle : $2\to16\to1$, unités cachées tanh, sortie logistique, SGD par minibatch de taille 32, 300 pas, taux d'apprentissage de base 0.5. La graine contrôle les poids initiaux et l'ordre des minibatchs. Les poids initiaux sont tirés selon $W\sim\mathcal N(0,\sigma^2/\text{fan\_in})$ avec $\sigma$ l'« échelle d'initialisation ». Cinq graines, de 0 à 4. Tous les nombres ci-dessous proviennent du harnais de l'exercice.

**Ablation 1 : échauffement (0 contre 50 pas).**

| graine | sans échauffement | échauffement 50 | différence |
|---|---|---|---|
| 0 | 0.911 | 0.904 | +0.007 |
| 1 | 0.873 | 0.873 | 0.000 |
| 2 | 0.900 | 0.893 | +0.007 |
| 3 | 0.935 | 0.929 | +0.006 |
| 4 | 0.918 | 0.908 | +0.010 |

Moyennes : $0.907\pm0.023$ et $0.901\pm0.021$ (moyenne $\pm$ écart-type empirique). Regarde la dispersion : l'écart-type d'une graine à l'autre (environ 0.02) est bien plus grand que l'écart (0.006), donc la statistique de Welch non appariée n'est que $t=0.43$ et les deux moyennes semblent indiscernables. Mais les différences *appariées* sont systématiquement non négatives : $\bar d=0.006$, $s_d=0.0037$, $t=3.65>2.776$, avec « sans échauffement » meilleur sur 4 graines et à égalité sur 1. Avec 20 graines, la différence appariée est $0.0053$ avec $t=9.1$ (moyennes 0.906 contre 0.901).

**Comment lire cela honnêtement.** Sans échauffement est *légèrement* meilleur ici, d'environ un demi-point de précision, et le test apparié dit que ce n'est pas de la chance. Mais l'effet est minuscule et a une explication banale : l'échauffement passe les 50 premiers des 300 pas à un taux d'apprentissage réduit, donc l'exécution avec échauffement a simplement un peu moins entraîné. Ce réseau est peu profond, à base de tanh et utilise un taux d'apprentissage doux, donc il n'y a aucune instabilité initiale que l'échauffement pourrait guérir. L'affirmation correcte est « à ce taux d'apprentissage et ce budget, l'échauffement n'aide pas et coûte environ 0.5 point », et non « l'échauffement est inutile ». L'échauffement se justifie surtout quand de grands taux d'apprentissage ou des modèles profonds/à attention sont instables aux premiers pas, un régime que cette expérience ne teste pas. Comme sonde supplémentaire rapide, avec un taux d'apprentissage de 5.0, le harnais donne $0.941\pm0.013$ sans et $0.914\pm0.034$ avec échauffement : toujours aucun bénéfice, et une variance plus grande. Nous n'avons pas réglé séparément les taux d'apprentissage pour chaque bras, ce qui est une limite.

**Ablation 2 : échelle d'initialisation (0.1 contre 1.0).** Apparié par graine, la précision de test est $0.864\pm0.005$ avec $\sigma=0.1$ et $0.907\pm0.023$ avec $\sigma=1$ : un gain de $0.044$, $s_d=0.019$, $t=5.2$. C'est un effet important comparé à l'échauffement. Un soupçon naturel est qu'une petite initialisation entraîne simplement plus lentement. Nous l'avons testé en entraînant 10 fois plus longtemps (3000 pas) : $\sigma=0.1$ atteint $0.867$ tandis que $\sigma=1$ atteint $0.953$, donc l'écart ne se comble pas. À $\sigma=0.1$, le réseau reste à peu près à la précision d'un classifieur linéaire sur ces données. Un mécanisme plausible est que des poids minuscules maintiennent les unités tanh dans leur région quasi linéaire et que le gradient atteignant la première couche est multiplié par de minuscules poids de la seconde couche, mais nous n'avons pas testé ce mécanisme directement, nous l'énonçons donc comme une hypothèse. Pour être complet, $\sigma=3$ donne $0.943$ à 300 pas et $0.948$ à 3000 pas : plus grand n'est pas mieux au-delà d'un certain point (et la dispersion entre graines est de quelques points, donc $0.953$ contre $0.948$ n'est pas une différence que nous pouvons revendiquer).

**Une mise en garde du monde réel tirée du matériel du cours.** Dans le chapitre sur les modèles de raisonnement consacré à la mise à l'échelle à l'inférence, un tableau de précisions sur 500 problèmes liste une configuration, « top-p avec auto-cohérence », avec $n=3$ échantillons à 29.6 pour cent, $n=5$ à 27.8 pour cent et $n=10$ à 31.6 pour cent. Plus de votes n'ont pas donné une amélioration monotone. Avec une précision proche de 0.3 sur 500 problèmes, l'erreur standard est $\sqrt{0.3\cdot0.7/500}\approx0.02$, donc la différence de 1.8 point entre $n=3$ et $n=5$ est dans le bruit (et chaque ligne est une exécution unique ; le notebook ne rapporte pas de graines répétées). C'est exactement le genre de tableau où un lecteur attentif demande la variance.

## Rapporter honnêtement

- Énonce le protocole complet : données, découpage, graines, pas, chaque hyperparamètre, et ce qui a été réglé pour quel bras.
- Rapporte la moyenne $\pm$ écart-type sur $n$ graines, la taille d'effet et un intervalle ou un test, ainsi que les résultats *négatifs* et *nuls*.
- Montre la référence et l'alternative simple la plus forte.
- Sépare les observations (« la précision a augmenté de 0.044 ») des interprétations (« parce que tanh sature »), et marque les interprétations que tu n'as pas testées.
- Publie le code et les graines pour que quelqu'un d'autre puisse relancer ton tableau.
- Ne cache pas le nombre de choses essayées. Si tu as essayé 20 variantes et rapportes la meilleure, le lecteur doit le savoir (les comparaisons multiples gonflent les faux positifs).

## Repérer les faiblesses d'un article (ou de ton propre travail)

1. Un gain plus petit que le bruit entre graines, issu d'une seule exécution.
2. Des références non réglées, obsolètes, ou dotées de moins de calcul.
3. Un réglage sur le jeu de test, ou un jeu d'évaluation qui recoupe les données d'entraînement (contamination).
4. Des comparaisons à budgets inégaux (paramètres, pas, tokens, temps réel) sans le dire.
5. Des moyennes sans tranches, ou des métriques choisies après avoir vu les résultats.
6. Aucune ablation : plusieurs changements introduits ensemble et le crédit donné au plus à la mode.
7. Des affirmations qui dépassent les réglages testés (un jeu de données, une taille de modèle).
8. Des détails manquants qui empêcheraient la reproduction.

## Implémentation

```python
def run_seeds(cfg, seeds):
    return np.array([train_mlp(s, **cfg) for s in seeds])

def paired_stats(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return {"mean_diff": d.mean(), "sd_diff": sd,
            "t": d.mean() / (sd / np.sqrt(len(d))), "wins": int((d > 0).sum())}

# usage: same seeds for both arms = common random numbers
a = run_seeds({"warmup": 0},  range(5))
b = run_seeds({"warmup": 50}, range(5))
print(summarize(a), summarize(b), paired_stats(a, b))
```

## Formes

| Objet | Forme |
|---|---|
| données d'entraînement | $X\in\mathbb{R}^{200\times2}$, $y\in\{0,1\}^{200}$ |
| $W_1,\ b_1$ | $(2,16)$, $(16,)$ |
| $W_2,\ b_2$ | $(16,1)$, $(1,)$ |
| minibatch | $x\in\mathbb{R}^{32\times2}$ |
| précisions d'une configuration | $(n_\text{seeds},)$ |
| différences appariées | $(n_\text{seeds},)$ |

## Idées reçues

1. « Une exécution suffit si le nombre est bon. » C'est un tirage d'une distribution ; la dispersion peut dépasser l'effet.
2. « Statistiquement significatif signifie important. » Ici un effet de 0.5 point est significatif avec appariement et pourtant négligeable.
3. « Pas de différence significative signifie que les deux sont égaux. » Avec 5 graines, tu manques peut-être simplement de puissance ; rapporte un intervalle et dis quelles tailles d'effet tu aurais pu détecter.
4. « Ablation signifie retirer une fonctionnalité et regarder la précision chuter. » Cela signifie un changement contrôlé d'un facteur avec une référence équitable et réglée autant.
5. « L'appariement est une astuce. » Il est légitime précisément parce que les bras partagent leurs graines ; il est invalide si les paires ne sont pas vraiment appariées.

## Points clés à retenir

- Lis en plusieurs passes ; note l'affirmation, la preuve, la référence et ce qui manque.
- Reproduis la référence avant de tester un changement.
- Change une chose à la fois, garde le budget fixe, règle les références équitablement.
- Utilise plusieurs graines, rapporte la moyenne $\pm$ écart-type, et compare les bras avec des nombres aléatoires communs et une statistique appariée.
- Sépare la taille d'effet de la significativité ; sépare l'observation de l'interprétation.
- Rapporte les nuls et les limites ; la production ajoute des tests fantôme, A/B et canari par-dessus l'évaluation hors ligne.

## Explique-le toi-même

1. Dans le tableau de l'échauffement, le $t$ non apparié vaut 0.43 et le $t$ apparié 3.65 pour les mêmes données. Explique pourquoi l'appariement fait une telle différence et quelle condition doit être remplie pour que l'appariement soit valide.
2. Pourquoi l'affirmation « l'échauffement est inutile » n'est-elle pas soutenue par l'expérience, alors que le test apparié est significatif en faveur de l'absence d'échauffement ?

## Pour aller plus loin

- Keshav, "How to Read a Paper" (2007), la méthode en trois passes.
- Huyen, *Designing Machine Learning Systems*, chap. 6 et chap. 9.
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021), sur les graines et les sources de variance.
- Henderson et al., "Deep Reinforcement Learning that Matters" (2018), sur les échecs de reproductibilité.

## Critères de maîtrise

- [ ] Je sais lire un article en trois passes et lister ses affirmations avec la preuve de chacune.
- [ ] Je sais concevoir une ablation avec une variable manipulée, un budget fixe et une référence équitable.
- [ ] Je sais lancer plusieurs graines et rapporter la moyenne $\pm$ écart-type et une comparaison appariée.
- [ ] Je sais expliquer les nombres aléatoires communs et calculer la statistique $t$ appariée.
- [ ] Je sais distinguer ce qu'une expérience montre de ce que je soupçonne seulement.
