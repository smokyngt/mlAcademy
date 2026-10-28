---
id: sys.04
title: "Évaluer les LLM et les systèmes RAG"
summary: "Pourquoi les scores de benchmarks trompent, comment construire et noter son propre jeu d'évaluation, les métriques de récupération (recall@k, MRR, nDCG), la fidélité et les LLM juges, les tranches, et les intervalles de confiance par bootstrap pour décider si le système B est vraiment meilleur que A."
minutes: 90
teaches: [llm.eval]
requires: [ml.generalization, prob.rv, llm.rag]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/10-evaluation/docs/en.md (Problem, Eval landscape, Why Benchmarks Break, Perplexity, LLM-as-Judge, ELO): saturation, contamination, judge biases. READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/10-evaluation/docs/en.md (Eval Datasets, Sample Size and Confidence, Regression Testing, Anti-Patterns): golden/adversarial/distribution sets, CI width table, paired comparison workflow. READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/07-advanced-rag/docs/en.md (Evaluation section): Recall@k, faithfulness, answer correctness. READ."
  - "reasoning-from-scratch/chF/01_main-chapter-code/chF_main.ipynb (Appendix F, markdown cells F.1-F.5): answer-choice accuracy, verifiers, Elo from pairwise preferences, LLM judges. READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (Sec. 3.1-3.9 markdown, results tables): verifier pipeline (extract, normalise, compare), tiny-sample accuracy (10 vs 500 problems). READ."
  - "reasoning-from-scratch/chF/02_mmlu/README.md and random_guessing_baseline.py, chF/04_llm-judge/README.md: three MMLU scoring methods, random baseline, judge by local model. READ (heads)."
  - "Designing Machine Learning Systems.pdf, Ch.6 Model Offline Evaluation (PDF p.201-214): baselines, perturbation/invariance/directional tests, calibration, slice-based evaluation, Simpson's paradox (Table 6-4). READ."
---

## Où cela s'inscrit

Tu as un chatbot RAG et un collègue a un nouveau prompt, un nouveau modèle d'embedding et une taille de chunk différente. Lequel est le meilleur ? « J'ai essayé cinq questions et ça avait l'air bien » est la façon dont la plupart des équipes répondent, et c'est ainsi que les régressions atteignent les utilisateurs. Les leçons précédentes ont rendu les systèmes plus rapides (cache, quantification, décodage spéculatif) et construit un pipeline de récupération. Chacun de ces changements exige une *évaluation* qui détecte la perte de qualité, sans quoi l'accélération n'est qu'un espoir. Cette leçon est la boîte à outils de mesure : quoi mesurer, sur quelles données, avec quelles statistiques, et comment distinguer une vraie amélioration du bruit.

Elle s'appuie directement sur la leçon de généralisation : le protocole du jeu de test (ne jamais régler dessus) s'applique toujours, tout comme l'idée qu'un nombre honnête est toujours mesuré sur des données que le système n'a jamais vues.

## L'idée en mots simples

Un examen ne renseigne sur l'étudiant que si (a) les questions n'ont pas fuité à l'avance, (b) il y a assez de questions, (c) la notation est équitable et (d) les questions ressemblent au vrai travail. L'évaluation des modèles de langage échoue précisément sur ces quatre points :

- **Fuite (contamination).** Les benchmarks vivent sur le web, les données d'entraînement viennent du web. Un modèle peut bien scorer en ayant vu les questions. Même sans intention, un mélange d'entraînement façonné sur les formats de benchmarks aide (la source note que MMLU est un QCM à 4 choix, donc un modèle peut apprendre la distribution des lettres de réponse).
- **Trop peu de questions.** Un score de 90 pour cent sur 50 questions est compatible avec une vraie précision allant d'environ 79 à 96 pour cent. Des différences de quelques points sur de petits jeux sont du bruit.
- **Notation inéquitable.** Les réponses libres exigent un correcteur : un script (correspondance exacte, vérificateur mathématique), un humain, ou un autre LLM, chacun avec ses propres erreurs.
- **Mauvaises questions.** Un classement dit peu de chose sur *ta* tâche : la source dit clairement que pour un bot de support client, un benchmark de connaissances générales n'est pas pertinent.

Le remède n'est pas « ne jamais utiliser de benchmarks » ; ils sont utiles pour une sélection grossière de modèle. Le remède est un petit jeu d'évaluation à toi, bien construit, plus des statistiques honnêtes.

## Quoi mesurer

**D'abord des références (baselines).** Le livre sur les systèmes de ML raconte une histoire : un classifieur avec un F1 de 0.90 sonnait bien jusqu'à ce que quelqu'un demande ce que score un devineur aléatoire sur une tâche où 90 pour cent des étiquettes sont positives ; le hasard obtient aussi un F1 d'environ 0.90. Compare avec une référence aléatoire, une règle triviale (toujours la classe majoritaire), la solution existante et une référence humaine. Le matériel MMLU du cours de raisonnement fournit de même un script de référence par devinette aléatoire : avec quatre options, 25 pour cent est le plancher. Dans les petites exécutions du notebook, un modèle de base a obtenu environ 21 pour cent sur un sous-ensemble de MMLU, c'est-à-dire à peu près le hasard.

**Métriques de tâche pour la génération.** Quand les réponses peuvent être vérifiées (maths, code, QCM), utilise un *vérificateur* : extraire la réponse finale, la normaliser, tester l'équivalence. Le chapitre 3 du cours de raisonnement construit exactement ce pipeline (extraire la réponse encadrée, normaliser, vérifier l'équivalence, noter). Le notebook rapporte 30 pour cent contre 90 pour cent de précision sur 10 problèmes, et 15.6 pour cent contre 48.2 pour cent sur 500 ; les chiffres sur 10 problèmes montrent à quel point les très petits échantillons sont grossiers (un problème déplace le score de dix points).

**Métriques de récupération pour le RAG** (le récupérateur est évalué séparément du générateur, avec des requêtes étiquetées) :

- **Recall@k** : fraction des documents pertinents qui apparaissent dans les $k$ premiers.
$$
\text{recall@}k=\frac{|\text{top-}k\cap \text{Rel}|}{|\text{Rel}|}.
$$
- **MRR** (rang réciproque moyen) : pour chaque requête, prendre $1/\text{rank}$ du premier résultat pertinent (0 s'il n'y en a pas), moyenner sur les requêtes. Elle récompense le fait de placer un bon document en premier.
- **nDCG@k** : pour une pertinence graduée. Gain cumulé actualisé avec l'actualisation usuelle :
$$
\text{DCG@}k=\sum_{i=1}^{k}\frac{g_i}{\log_2(i+1)},\qquad \text{nDCG@}k=\frac{\text{DCG@}k}{\text{DCG@}k\text{ of the ideal order}}.
$$
L'actualisation logarithmique dit qu'un bon document au rang 1 vaut plus qu'au rang 5, mais pas de façon brutale, en falaise. Diviser par l'idéal place le score dans $[0,1]$. (Le gain linéaire $g_i$ est utilisé ici ; certaines bibliothèques utilisent $2^{g_i}-1$.)

**Métriques de génération pour le RAG.**

- **Fidélité** (groundedness) : chaque affirmation de la réponse est-elle soutenue par les passages récupérés ? Si le chunk dit « délai de remboursement de 60 jours » et que le modèle dit 90, c'est un échec de fidélité même si la récupération était parfaite. Un vérificateur simple découpe la réponse en affirmations et teste chacune contre le texte récupéré ; un plus robuste utilise un LLM juge.
- **Exactitude de la réponse** : la réponse correspond-elle à la référence ? Elle mélange erreurs de récupération et de génération, garde donc les deux autres pour diagnostiquer.

**Tranches et robustesse.** Le livre sur les systèmes insiste sur le découpage en tranches : rapporter la précision par sous-groupe (langue, type de document, longueur de requête, utilisateurs payants contre gratuits), car les moyennes cachent les échecs. Son tableau 6-4 (une étude sur les calculs rénaux) montre le **paradoxe de Simpson** : le modèle A bat le modèle B dans les deux groupes (93 pour cent contre 87 pour cent, 73 pour cent contre 69 pour cent) et pourtant B a la meilleure précision globale (83 pour cent contre 78 pour cent), parce que les groupes ont des tailles très différentes pour les deux modèles. Les tests de perturbation (ajouter du bruit, changer la formulation), d'invariance (changer un nom ne doit pas changer la réponse) et les vérifications de calibration complètent la précision.

## LLM juge et ses pièges

Pour des sorties ouvertes, un modèle fort note les réponses d'un autre modèle selon une grille. C'est peu coûteux et assez corrélé aux humains (les sources citent environ 80 pour cent d'accord), mais :

- **Biais de position** : dans une comparaison par paires, la première réponse est favorisée. Randomise l'ordre, ou note les deux ordres et fais la moyenne.
- **Biais de verbosité** : les réponses plus longues paraissent meilleures. Contrôle la longueur.
- **Auto-préférence** : un modèle favorise le style de sa propre famille. Utilise un juge différent du système testé.
- **Les grilles vagues** donnent des notes bruitées ; une grille à niveaux explicites en donne de reproductibles.
- **Un juge faible** est bruité. Le juge devrait être au moins aussi capable que le système jugé.
- **Valide toujours le juge** sur quelques dizaines de cas étiquetés par des humains avant de lui faire confiance, et rapporte le modèle juge et le prompt.

**Préférences par paires et Elo.** L'évaluation de type arène de chatbots montre deux réponses et enregistre laquelle est préférée. Le score attendu d'un modèle de classement $R_w$ contre $R_l$ est
$$
E_w=\frac{1}{1+10^{(R_l-R_w)/400}},
$$
des classements égaux donnent $0.5$ et un écart de 400 points donne environ $0.91$. Le notebook de l'annexe F note que le vrai classement est passé à un ajustement de Bradley-Terry avec la même idée de paires.

## Construire un petit jeu d'évaluation

D'après les sources :

1. **Jeu d'or** (50 à 100 cas triés sur le volet) : les cas d'usage principaux et tes tests de régression.
2. **Jeu adversarial** (20 à 50) : requêtes ambiguës, questions hors domaine, injections de prompt, requêtes dont la réponse n'est *pas* dans le corpus (le système doit dire qu'il ne sait pas).
3. **Échantillon de distribution** (100 à 200) : requêtes réelles tirées au hasard, car les jeux triés manquent ce que les utilisateurs demandent vraiment.
4. Pour la récupération, stocke pour chaque requête les identifiants des chunks pertinents ; pour les réponses, une réponse de référence ou une propriété vérifiable.
5. Garde les données d'évaluation séparées des prompts, des exemples few-shot et de toute donnée de réglage ; ne règle jamais sur les éléments du test final.
6. Versionne le jeu et tiens un journal des changements ; un jeu modifié est un autre benchmark.

## Dérivation : intervalles de confiance par bootstrap

Tu as mesuré un vecteur de scores par élément $x_1,\dots,x_n$ (exactitude 0/1, ou rang réciproque par requête). La moyenne empirique $\bar x$ est une estimation ; nous voulons savoir de combien elle fluctuerait sur un autre échantillon de $n$ éléments issu de la même population. Le **bootstrap** remplace « un autre échantillon de la population » par « un échantillon tiré avec remise parmi nos propres $n$ éléments » :

1. Tire $B$ rééchantillons de taille $n$ avec remise.
2. Calcule la moyenne de chacun, ce qui donne $\bar x^{*}_1,\dots,\bar x^{*}_B$.
3. L'intervalle percentile est formé des quantiles $\alpha/2$ et $1-\alpha/2$ de ces moyennes.

Pourquoi c'est sensé : la distribution empirique des données est la meilleure image de la population dont nous disposions, et rééchantillonner depuis elle imite l'échantillonnage de la population. Pour une moyenne, la dispersion des moyennes rééchantillonnées est proche de $s/\sqrt n$, l'erreur standard classique, donc les intervalles rétrécissent comme $1/\sqrt n$ : quatre fois plus d'éléments divisent la largeur par deux. Pour des proportions proches de 0 ou 1 ou un $n$ minuscule, l'intervalle de Wilson est une meilleure alternative en forme fermée :

$$
\frac{p+\frac{z^2}{2n}\pm z\sqrt{\frac{p(1-p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}} .
$$

**Comparer deux systèmes sur les mêmes éléments** doit se faire par *appariement*. Soit $d_i=a_i-b_i$ sur l'élément $i$. Applique le bootstrap aux différences $d_i$ : les éléments que les deux systèmes réussissent ou ratent s'annulent, et seuls comptent les éléments où ils divergent. Regarder deux intervalles séparés pour $a$ et $b$ et vérifier s'ils se chevauchent jette l'appariement et est trop conservateur. À l'inverse, avec très peu de désaccords, même un bootstrap apparié est optimiste (il peut rééchantillonner « aucun élément en désaccord en faveur de $a$ ») ; rapporte le nombre de désaccords.

## Exemple détaillé

Tous les nombres ci-dessous sont reproduits par le code de l'exercice.

**Métriques de récupération.** Quatre requêtes, listes classées d'identifiants, ensembles pertinents :

| requête | classement | pertinents | RR | recall@3 | recall@5 |
|---|---|---|---|---|---|
| 1 | d3 d1 d7 d2 d9 | {d1} | 1/2 | 1 | 1 |
| 2 | d4 d5 d6 d1 d2 | {d1, d2} | 1/4 | 0 | 1 |
| 3 | d8 d2 d1 d3 d4 | {d8} | 1 | 1 | 1 |
| 4 | d5 d6 d7 d8 d9 | {d1} | 0 | 0 | 0 |

MRR $=(0.5+0.25+1+0)/4=0.4375$ ; recall@3 moyen $=0.5$ ; recall@5 moyen $=0.75$. La requête 2 montre la différence entre métriques : un document pertinent est récupéré avec $k=5$ mais seulement en position 4, donc le recall@5 est parfait tandis que son rang réciproque vaut 0.25.

**nDCG.** Pertinence graduée $a{:}3,\ b{:}2,\ c{:}3,\ d{:}0,\ e{:}1,\ f{:}2$ et le classement $a,b,c,d,e,f$ a $\text{DCG}=6.861$ ; l'ordre idéal ($3,3,2,2,1,0$) a $7.141$ ; donc nDCG@6 $=0.961$. Déplacer le document inutile $d$ tout en haut ($d,a,c,b,e,f$) le fait chuter à $0.750$. Le nDCG@3 du premier classement est $0.978$.

**Taille d'échantillon.** Intervalles de Wilson à 95 pour cent : 45/50 corrects donne $[0.786, 0.957]$ (une étendue de 17 points) ; 180/200 donne $[0.851, 0.934]$ ; 900/1000 donne $[0.880, 0.917]$. La même précision de 90 pour cent, trois éléments de preuve très différents. La règle empirique du matériel d'évaluation est au moins 200 cas pour les décisions de déploiement et 500 ou plus pour comparer des systèmes proches, en accord avec ces nombres.

**Bootstrap apparié.** Exactitude simulée par élément sur 100 éléments pour les systèmes A et B (les éléments partagent une difficulté latente, donc les erreurs sont corrélées) : A a raison sur 74, B sur 67. Intervalles bootstrap à 95 pour cent : A $[0.65,0.82]$, B $[0.58,0.76]$, qui se chevauchent fortement. A et B divergent sur 17 éléments (A seul a raison sur 12, B seul sur 5). Le bootstrap apparié des différences donne une moyenne $+0.07$, intervalle $[-0.01, +0.15]$ ; la version non appariée (rééchantillonnant les deux systèmes indépendamment) donne l'intervalle bien plus large $[-0.05, +0.20]$. Environ 4.8 pour cent des rééchantillons appariés ont une différence moyenne $\le 0$. Conclusion honnête : A est probablement meilleur, la preuve est suggestive mais non concluante à 95 pour cent, et 100 éléments est trop peu pour trancher une différence de 7 points. (Un test du signe exact sur les 12 contre 5 désaccords donne un p bilatéral d'environ 0.14, en accord avec cette lecture.)

## Implémentation

```python
def ndcg_at_k(ranked, rel, k):
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    return dcg(gains) / dcg(ideal) if ideal and dcg(ideal) > 0 else 0.0

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
```

## Formes

| Objet | Forme |
|---|---|
| liste classée pour une requête | liste de $k$ identifiants |
| vecteur de métrique par requête | $(n_\text{queries},)$ |
| matrice d'indices bootstrap | $(B, n)$ |
| moyennes bootstrap | $(B,)$ |
| différences appariées | $(n,)$ |

## Idées reçues

1. « Il faut des intervalles non chevauchants pour déclarer une différence réelle, et un chevauchement signifie aucune différence. » Aucun des deux n'est juste ; utilise un intervalle apparié pour la *différence*.
2. « Un recall@k élevé signifie que la réponse RAG sera juste. » Le générateur peut encore ignorer ou déformer les passages (fidélité), donc mesure les deux.
3. « Un score élevé sur un benchmark signifie que le modèle est bon pour ma tâche. » Seule une évaluation correspondante peut le dire.
4. « Les LLM juges sont objectifs. » Ce sont d'autres modèles avec des biais ; calibre-les contre des humains.
5. « Plus de métriques, c'est mieux. » Un tas de métriques invite à la sélection opportuniste ; décide de la métrique principale et de la règle de décision avant de regarder les résultats.
6. « La précision a augmenté, donc on livre. » Vérifie aussi les tranches et le coût (latence, mémoire) ; un modèle plus rapide et quantifié qui perd un point sur une tranche critique peut être inacceptable.

## Points clés à retenir

- Évalue sur des données qui n'ont pas pu fuiter, avec assez d'éléments, notés équitablement, ressemblant à l'usage réel.
- Récupération : recall@k (l'a-t-on obtenu), MRR (à quel point tôt), nDCG (gradué et sensible au rang). Génération : fidélité et exactitude.
- Rapporte des intervalles, pas des points : bootstrap ou Wilson ; la largeur rétrécit comme $1/\sqrt n$.
- Compare les systèmes avec une analyse appariée sur les mêmes éléments ; rapporte les désaccords.
- Découpe tes résultats en tranches ; le paradoxe de Simpson est réel.
- Les juges ont des biais (position, verbosité, auto-préférence) : randomise, contrôle la longueur, valide sur des étiquettes humaines.

## Explique-le toi-même

1. Le recall@5 de ton système RAG est de 0.95 mais les utilisateurs se plaignent de réponses fausses. Cite trois endroits où l'erreur peut se trouver et une métrique pour chacun.
2. Explique à un collègue pourquoi « A a obtenu 74 pour cent et B 67 pour cent sur 100 questions » ne suffit pas pour affirmer que A est meilleur, et ce que tu calculerais.

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, chap. 6 (Évaluation hors ligne) et chap. 9 (Tester en production).
- Efron et Tibshirani, *An Introduction to the Bootstrap*.
- Järvelin et Kekäläinen, "Cumulated gain-based evaluation of IR techniques" (2002), l'origine du nDCG.
- Les sources MMLU et MATH-500 citées dans le cours de raisonnement.

## Critères de maîtrise

- [ ] Je sais calculer recall@k, MRR et nDCG à la main et en code.
- [ ] Je sais expliquer la contamination, la saturation et pourquoi des évaluations sur mesure sont nécessaires.
- [ ] Je sais citer les biais des juges et leurs atténuations.
- [ ] Je sais construire un IC bootstrap et une comparaison appariée, et interpréter un intervalle qui contient zéro.
- [ ] Je sais expliquer le paradoxe de Simpson avec le tableau 6-4 et pourquoi les tranches comptent.
