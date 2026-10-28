---
id: app.04
title: "Évaluer et tester les applications LLM"
summary: "Construire un jeu d'évaluation, le noter avec des vérifications et des juges calibrés, et décider avec un intervalle bootstrap et un test de permutation exact si le prompt B bat vraiment le prompt A."
minutes: 75
teaches: [app.eval]
requires: [app.prompting, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/11-llm-engineering/10-evaluation/docs/en.md (opening sections: the silent-regression story, eval taxonomy, LLM-as-judge criteria and rubric design; first ~6.5k characters only). READ."
---

## Où cela s'inscrit

Chaque modification de prompt, changement de modèle ou ajustement de température change la distribution des sorties d'une façon que tu ne peux pas juger en lisant dix exemples. Les ingénieurs logiciels ne livrent pas sans tests ; pour les applications LLM, l'équivalent est un **harnais d'évaluation** exécuté à chaque changement. Cette leçon porte sur sa construction et, surtout, sur la *lecture honnête de ses chiffres* : les sorties sont aléatoires, les jeux de test sont petits et les juges sont imparfaits. Les leçons précédentes t'ont donné des choses à évaluer (prompts dans [[app.01]], schémas et outils dans [[app.02]], dispositions de contexte dans [[app.03]]). L'évaluation du modèle lui-même et des pipelines RAG (fidélité, métriques de récupération, contamination des benchmarks) est couverte dans la leçon `sys.04` ; ici nous nous concentrons sur les tests au niveau applicatif et les statistiques. Les statistiques utilisent l'idée de variable aléatoire et de sa variance issue de `prob.rv`.

## L'idée en termes simples

Suppose qu'un boulanger modifie une recette et demande à trois amis si le nouveau pain est meilleur. Deux disent oui. Est-il meilleur ? Tu ne peux pas le dire : trois amis, c'est un minuscule échantillon, ils peuvent être polis, et le boulanger a pu leur donner les bonnes tranches. Un test équitable utilise de nombreux dégustateurs, les mêmes tranches pour les deux recettes, une grille de notation claire, et une règle décidée *avant* la dégustation pour ce qui compte comme une vraie différence.

Évaluer une application LLM, c'est pareil. Le **jeu d'évaluation** est les dégustateurs. La **grille de notation** est un barème ou une vérification. Apparier les deux recettes sur les mêmes tranches est une **comparaison appariée**, qui annule la difficulté de chaque élément. Et la règle de la « vraie différence » est un **test de significativité** ou un **intervalle de confiance**, qui te disent quelle part de l'écart peut être due à la chance.

## Construire un jeu d'évaluation

- **Tire-le de la réalité.** Échantillonne de vraies entrées d'utilisateurs (anonymisées) ; ajoute les échecs que tu as vus ; ajoute des cas limites (entrée vide, très longue, autres langues, texte adversarial).
- **Étiquette ce qui compte.** Pour chaque élément, stocke l'entrée, une réponse de référence ou une règle d'acceptation, et des étiquettes (sujet, difficulté) pour pouvoir découper les résultats.
- **Garde deux ensembles.** Un ensemble de développement que tu regardes pendant le réglage, et un ensemble de réserve que tu touches rarement ; sinon tu sur-ajustes le prompt à l'ensemble, exactement comme on sur-ajuste un modèle.
- **Dimensionne-le pour l'effet qui t'intéresse.** Les petits ensembles ne détectent que de grandes différences (voir la formule de taille d'échantillon ci-dessous).
- **Versionne-le** avec la version du prompt et du modèle.

## Notation : trois types de notateurs

1. **Vérifications déterministes** : est-ce que cela s'analyse selon le schéma ([[app.02]]), contient un champ requis, reste sous une longueur, évite des chaînes interdites ? Rapide, gratuit, sans ambiguïté. Utilise-les d'abord et comme barrières.
2. **Métriques de référence** (correspondance exacte, recouvrement de n-grammes, similarité d'embeddings) : peu coûteuses mais aveugles au sens ; deux réponses correctes peuvent ne partager aucun mot.
3. **LLM-as-judge** : un modèle fort note une sortie par rapport à un barème. Il capture la pertinence, la justesse et le ton, coûte de l'argent, et peut être biaisé. La source rapporte des corrélations avec le jugement humain dans les 80 pour cent pour des barèmes bien conçus (des chiffres que nous ne pouvons pas vérifier et qui dépendent de la tâche).

**Les barèmes doivent être ancrés.** « Note de 1 à 5 la qualité » donne des scores bruités. Ancre chaque niveau à un comportement observable (« 5 : factuellement correct, répond directement à la question, cite la source ; 3 : correct mais manque un élément requis ; 1 : contient une erreur factuelle »).

### Pièges des juges

Ils sont largement rapportés en pratique (ils ne sont pas tous dans la source que nous avons lue, donc teste-les sur ton propre juge) :

- **Biais de position** : dans une comparaison par paires, le juge peut favoriser la première (ou la seconde) réponse. Fais les deux ordres et ne compte que les verdicts cohérents.
- **Biais de verbosité** : les réponses plus longues paraissent meilleures. Contrôle la longueur ou dis-le dans le barème.
- **Auto-préférence** : un modèle peut favoriser le texte dans son propre style ; utilise si possible un autre modèle juge que le générateur.
- **Dérive** : le juge est un modèle aussi ; fixe sa version.
- **Fuite** : ne montre pas au juge l'étiquette que tu veux qu'il confirme.

**Calibre contre des humains.** Étiquette 50 à 100 éléments à la main, fais tourner le juge, et mesure l'accord au-delà du hasard avec le **kappa de Cohen** :

$$
\kappa=\frac{p_o-p_e}{1-p_e},\qquad p_e=\sum_c P_{\text{human}}(c)\,P_{\text{judge}}(c).
$$

Ici $p_o$ est la fraction d'accord observée et $p_e$ l'accord attendu si les deux étiquetaient indépendamment avec leurs propres taux de base. Exemple : les étiquettes humaines `1,1,1,0,0,1,0,1,1,0` contre le juge `1,1,0,0,0,1,1,1,1,0` s'accordent sur 8 sur 10 ($p_o=0.8$) ; les deux disent 1 pour 6 sur 10, donc $p_e=0.6\cdot0.6+0.4\cdot0.4=0.52$ et $\kappa=0.28/0.48=0.583$. Un juge qui dit toujours « pass » a $\kappa=0$ quel que soit le taux de base, même s'il a raison la plupart du temps, c'est pourquoi l'exactitude brute est un mauvais étalon.

## Comparer deux prompts

Tu fais tourner les prompts A et B sur les mêmes $n$ éléments et tu notes chacun 0/1 (ou un score de barème). Pour chaque élément $i$, définis la différence appariée $d_i=b_i-a_i$. Sa moyenne $\bar d$ estime de combien B est meilleur. La question est de savoir de combien $\bar d$ varierait par hasard.

### Bootstrap par percentiles

Traite les éléments comme un échantillon de la population des entrées futures. Rééchantillonne $n$ éléments avec remise, recalcule $\bar d$, répète $B$ fois (disons 2 000), et prends les 2,5e et 97,5e percentiles des $B$ moyennes comme **intervalle de confiance** à 95 pour cent de la vraie amélioration. Aucune hypothèse de normalité n'est nécessaire. Si l'intervalle exclut confortablement zéro, l'amélioration est crédible ; s'il chevauche zéro, elle ne l'est pas.

### Test de permutation exact (inversion de signes)

Hypothèse nulle : les deux prompts sont interchangeables, donc pour chaque élément les étiquettes A et B auraient pu être échangées. Échanger inverse le signe de $d_i$ ; les éléments avec $d_i=0$ ne changent pas. Donc sous l'hypothèse nulle, chaque attribution de signes aux $m$ différences non nulles est également probable : $2^m$ motifs équiprobables. Soit $S=\sum_i d_i$ la somme observée. La p-valeur bilatérale est

$$
p=\frac{\#\{\text{sign patterns with }|S'|\ge |S|\}}{2^m}.
$$

Si $j$ des $m$ différences sont négatives, alors $|S'|=|m-2j|$, donc le décompte est une somme de coefficients binomiaux et le test est exactement le test des signes.

### Exemple détaillé

Quarante éléments. B réussit là où A échoue sur 9 éléments, A réussit là où B échoue sur 3, les deux réussissent sur 16, les deux échouent sur 12. A obtient $19/40=0.475$, B $25/40=0.625$ ; $\bar d=+0.15$. Seuls $m=12$ éléments diffèrent, $S=9-3=6$ observé. Les motifs de signes avec $|S'|\ge6$ sont ceux avec $j\le3$ ou $j\ge9$ négatifs : $2\,(1+12+66+220)=598$ sur $2^{12}=4096$, donc $p=598/4096=0.146$. L'intervalle bootstrap (graine 0, $B=2000$) est $(0.000,\,0.325)$. Les deux disent la même chose : un gain de 15 points sur 40 éléments n'est **pas** significatif au seuil de 5 pour cent. Il peut être réel ; le test ne peut simplement pas le distinguer de la chance. Avec 12 victoires contre 2 défaites, la p-valeur passe sous 0,05 (vérification de l'exercice).

### Combien d'éléments ?

Suppose que les différences par élément ont un écart type $\sigma$ et que tu veuilles détecter une vraie différence moyenne $\delta$ avec une significativité bilatérale de 5 pour cent et une puissance de 80 pour cent. La moyenne de $n$ différences est approximativement normale avec une erreur standard $\sigma/\sqrt n$. Tu rejettes quand $\bar d$ dépasse $z_{\alpha/2}\,\sigma/\sqrt n$ ; pour le faire avec une probabilité de 0,8 quand la vérité est $\delta$, le décalage $\delta$ doit se situer à $z_{\text{power}}$ erreurs standard au-delà de ce seuil :

$$
\delta=(z_{\alpha/2}+z_{\text{power}})\frac{\sigma}{\sqrt n}\;\Longrightarrow\; n=\left(\frac{(z_{\alpha/2}+z_{\text{power}})\,\sigma}{\delta}\right)^{2},
$$

avec $z_{\alpha/2}=1.96$ et $z_{\text{power}}=0.8416$. Pour $\delta=0.05$ et $\sigma=0.4$ (typique pour des scores 0/1 où environ 16 pour cent des éléments diffèrent), $n=\lceil(2.8016\cdot0.4/0.05)^2\rceil=503$. Diviser $\delta$ par deux quadruple $n$. De petites améliorations exigent de gros jeux d'évaluation, ou une métrique moins bruitée.

## Tests de régression et surveillance

- **Barrière en CI.** Lance l'évaluation à chaque changement de prompt ou de modèle. Bloque quand une vérification déterministe échoue, ou quand la borne basse de l'intervalle de confiance de la différence passe sous une marge que tu acceptes.
- **Regarde les tranches.** Une moyenne peut monter pendant qu'une tranche s'effondre. La source raconte un changement de prompt qui a réduit les hallucinations mais rendu les réponses 34 pour cent moins complètes, passé inaperçu pendant 11 jours parce qu'une seule métrique était suivie. Suis plusieurs métriques (justesse, complétude, format, latence, coût) et des résultats par étiquette.
- **Mets une barrière de coût sur l'évaluation elle-même.** Les appels au juge coûtent de l'argent ; lance un petit jeu de fumée à chaque commit et le jeu complet chaque nuit.
- **Surveille la production.** Journalise les entrées, sorties, résultats de récupération, appels d'outils, latence, coût en jetons et déclenchements de garde-fous ([[app.05]]) ; échantillonne une fraction pour une revue par juge ou par humain ; surveille la dérive de la distribution des entrées et des taux de refus ou de réessais ; réinjecte les nouveaux échecs dans le jeu d'évaluation.

## Implémentation

```python
import numpy as np

def bootstrap_ci(d, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def sign_flip_pvalue(d):
    nz = d[d != 0]; m = len(nz)
    bits = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1) * 2 - 1
    return np.mean(np.abs((bits * nz).sum(axis=1)) >= abs(nz.sum()) - 1e-12)
```

Le code de production utilise `scipy.stats` (`bootstrap`, `permutation_test`), des cadres d'évaluation comme RAGAS ou DeepEval pour des prompts de juge prêts à l'emploi, et des outils de suivi d'expériences pour stocker les exécutions. Pour de nombreux éléments, énumérer $2^m$ motifs est impossible ; échantillonne plutôt des motifs de signes aléatoires.

## Idées reçues courantes

- « Le nouveau prompt a un meilleur score, donc il est meilleur. » Seulement si la différence dépasse ce que le bruit produit à la taille de ton ensemble.
- « Un pourcentage élevé d'accord juge-humain signifie que le juge est bon. » Avec des étiquettes déséquilibrées, l'accord dû au hasard est élevé ; utilise le kappa.
- « Plus de métriques, c'est toujours mieux. » Lancer de nombreuses comparaisons gonfle les faux positifs ; décide de la métrique principale à l'avance.
- « La température 0 rend l'évaluation déterministe. » Souvent pas exactement (non-déterminisme du batching et du matériel), et elle cache la variance que tu verras en production.
- « Le jeu d'évaluation est terminé. » Il doit croître avec chaque échec en production.

## Points clés à retenir

- Évalue avec un jeu d'évaluation versionné et découpé en tranches, une séparation dev/réserve et des barèmes ancrés.
- Utilise d'abord des vérifications déterministes, puis des métriques de référence, puis des juges calibrés.
- Compare les prompts comme des différences appariées ; rapporte un IC bootstrap et, pour les petits ensembles, une p-valeur exacte par inversion de signes.
- La taille d'échantillon évolue comme $(\sigma/\delta)^2$ ; un gain de 15 points sur 40 éléments n'était pas significatif.
- Mets des barrières sur les changements en CI et surveille les tranches et les métriques de production.

## Explique-le toi-même

1. Pourquoi apparier les deux prompts sur les mêmes éléments donne-t-il une comparaison plus sensible que de calculer deux exactitudes séparées ?
2. Ton juge s'accorde avec les humains sur 90 pour cent des éléments, mais 90 pour cent des éléments sont « bons ». Que calcules-tu ensuite et pourquoi ?

## Pour aller plus loin

- Leçon `sys.04` pour l'évaluation des RAG et des benchmarks de LLM ; documentation de RAGAS et DeepEval pour les métriques à base de juges.
- Efron et Tibshirani sur le bootstrap, pour la théorie derrière le rééchantillonnage.

## Critères de maîtrise

- Tu sais construire un plan de jeu d'évaluation avec séparations et étiquettes.
- Tu sais calculer un intervalle bootstrap apparié et une p-valeur exacte par inversion de signes à partir de scores par élément.
- Tu sais calculer la taille d'échantillon pour un effet cible.
- Tu sais calculer le kappa et expliquer pourquoi l'exactitude induit en erreur.
