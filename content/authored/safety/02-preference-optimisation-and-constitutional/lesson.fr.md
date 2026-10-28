---
id: saf.02
title: "Optimisation des préférences et méthodes constitutionnelles"
summary: "Dériver le DPO à partir de l'optimum du RLHF régularisé par KL, comparer ce que DPO, IPO, KTO, SimPO et ORPO optimisent chacun, et voir comment remplacer les annotateurs humains par des principes (RLAIF) change la nature de l'erreur."
minutes: 75
teaches: [saf.pref-optimisation, saf.constitutional-ai]
requires: [saf.goodhart, info.entropy, ml.logreg, llm.dpo]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/03-direct-preference-optimization-family/docs/en.md (whole lesson): DPO derivation outline, IPO, KTO, SimPO, ORPO, BPO, over-optimisation of direct alignment algorithms. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif/docs/en.md (whole lesson): critique-and-revise phase, RLAIF, the 2026 constitution rewrite, constitutional classifiers. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal/docs/en.md (loss definitions only). READ."
---

## Où cela s'insère

[[saf.01]] a montré la recette standard pour utiliser les préférences humaines : ajuster un modèle de récompense, puis optimiser une politique contre lui avec une laisse KL. Cette recette a deux parties gênantes. Elle nécessite un modèle de récompense séparé et une boucle de RL instable, et elle nécessite que des *personnes* produisent chaque comparaison. Cette leçon couvre les deux principales réponses. Les **algorithmes d'alignement direct** (DPO et apparentés) sautent le modèle de récompense et entraînent la politique directement sur les comparaisons. L'**IA constitutionnelle / RLAIF** remplace une partie ou la totalité de l'annotation humaine par un modèle qui lit des principes écrits. La leçon d'alignement [[llm.05]] introduit ces méthodes du côté ingénierie ; ici, nous demandons ce que chacune *optimise* et quels modes d'échec chacune conserve. Les deux héritent du problème de Goodhart de [[saf.01]].

## L'idée en termes simples

Pense à former un nouvel employé en lui montrant des paires de rapports terminés et en disant « celui-ci est meilleur que celui-là ».

- **La recette classique** entraîne d'abord un *critique* séparé à imiter tes verdicts par paires, puis laisse l'employé s'exercer contre le critique, avec une règle : « ne t'éloigne pas trop de ta façon d'écrire d'avant ».
- **La recette directe** saute le critique. Elle remarque que, avec cette règle « ne t'éloigne pas », le meilleur employé peut être décrit *en termes de combien plus souvent il écrit le meilleur rapport qu'avant*. Tu peux donc ajuster l'employé directement : augmenter les chances du rapport préféré par rapport à avant, diminuer celles de l'autre.
- **La recette constitutionnelle** remplace la personne qui juge par un assistant qui lit une courte liste de principes (« préfère la réponse honnête et non nuisible ») et juge d'après ceux-ci. Tu échanges un juge désordonné, individuel, bruité contre un juge cohérent, lisible, mais peut-être *systématiquement biaisé*.

Chaque variante est un choix différent sur *ce que voit le juge* (des paires, un simple pouce haut/bas, des principes), *comment fonctionne la laisse* (un modèle de référence, une normalisation par la longueur, un terme supervisé), et *quoi faire quand le signal est extrême*.

## Dérivation : le DPO à partir de l'optimum du RLHF

Pars de la forme close de [[saf.01]]. Pour un prompt $x$, la politique qui maximise $\mathbb{E}[r]-\beta\,\mathrm{KL}(\pi\|\pi_{ref})$ est

$$
\pi^*(y|x)=\frac{\pi_{ref}(y|x)\,e^{r(x,y)/\beta}}{Z(x)} .
$$

Prends les logarithmes et réarrange pour exprimer la récompense à travers les politiques :

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x).
$$

Le modèle de Bradley-Terry dit qu'un évaluateur préfère $y_w$ à $y_l$ avec la probabilité $\sigma\big(r(x,y_w)-r(x,y_l)\big)$. Substitue l'expression ci-dessus. Le terme $\beta\log Z(x)$ ne dépend que de $x$ et apparaît dans les deux récompenses, donc il **s'annule** dans la différence :

$$
r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} .
$$

Remplace maintenant l'optimum inconnu $\pi^*$ par la politique $\pi_\theta$ en cours d'entraînement et maximise la vraisemblance des préférences observées. La perte DPO est

$$
\mathcal{L}_{DPO}=-\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\log\tfrac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\big]\Big).
$$

Pas de modèle de récompense, pas de boucle d'échantillonnage : seulement les log-probabilités des deux réponses sous la politique entraînée et la politique de référence. Définis la **récompense implicite** $\hat r=\beta\log(\pi_\theta/\pi_{ref})$ ; le DPO est un ajustement de Bradley-Terry de $\hat r$.

**Gradient.** Soit $m=\beta[\Delta_w-\Delta_l]$ la marge, où $\Delta=\log(\pi_\theta/\pi_{ref})$. Comme $\frac{d}{dm}[-\log\sigma(m)]=-\sigma(-m)$,

$$
\frac{\partial\mathcal{L}}{\partial\log\pi_\theta(y_w)}=-\beta\,\sigma(-m),\qquad
\frac{\partial\mathcal{L}}{\partial\log\pi_\theta(y_l)}=+\beta\,\sigma(-m).
$$

Les exemples sont pondérés par $\sigma(-m)$ : les paires que le modèle classe actuellement *mal* (marge petite ou négative) comptent le plus. Remarque que le gradient ne contraint que la *différence* ; rien dans la perte ne dit que la log-probabilité de la réponse choisie doit monter.

### Les hypothèses derrière la dérivation

La dérivation suppose que l'optimum est atteignable, que les données de préférence ressemblent aux propres échantillons de la politique, et que le modèle de référence est un bon ancrage. La leçon source insiste sur le fait qu'aucune de ces hypothèses ne tient exactement, et que chaque membre de la famille répare une violation différente.

## La famille : ce que chacun optimise

Notation : $\Delta_w,\Delta_l$ sont les log-rapports à la référence pour la gagnante et la perdante ; $p$ désigne une probabilité de séquence (normalisée par la longueur).

| Méthode | Données d'entrée | Perte (par exemple) | Ce qui change par rapport au DPO |
|---|---|---|---|
| DPO | paires | $-\log\sigma\big(\beta(\Delta_w-\Delta_l)\big)$ | référence |
| IPO | paires | $\big(\Delta_w-\Delta_l-\tfrac1{2\beta}\big)^2$ | erreur quadratique vers une cible bornée ; la marge ne peut pas exploser |
| KTO | sorties isolées avec un indicateur bon/mauvais | $1-\sigma(\beta\Delta-z_{ref})$ si désirable, $1-\sigma(z_{ref}-\beta\Delta)$ sinon (simplifié ; le vrai KTO pondère aussi différemment les gains et les pertes) | pas de paires nécessaires |
| SimPO | paires | $-\log\sigma\big(\tfrac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w)-\tfrac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l)-\gamma\big)$ | pas de modèle de référence ; normalisé par la longueur ; marge $\gamma$ |
| ORPO | paires | NLL sur $y_w$ plus $\lambda\cdot[-\log\sigma(\log\tfrac{\text{odds}(y_w)}{\text{odds}(y_l)})]$ | une seule étape à partir du modèle de base ; le terme NLL est le régularisateur |
| BPO | paires | DPO plus une correction qui empêche la log-probabilité de la réponse choisie de baisser | cible l'effet de « réponse choisie dégradée » |

Lecture du tableau comme modes d'échec (c'est le cadrage de la leçon source) :

- **Écart de récompense implicite non borné (DPO).** Une minuscule préférence peut quand même pousser la marge arbitrairement haut. La cible $1/(2\beta)$ de l'IPO lie l'écart à la force de la préférence.
- **Réponses choisies dégradées (DPO).** La perte est satisfaite si la log-probabilité de la rejetée chute plus vite que celle de la choisie, donc la réponse choisie peut devenir *moins* probable en valeur absolue. Le BPO est l'exemple de réparation de la leçon ; la leçon rapporte un gain sur un benchmark de mathématiques issu d'un pré-print en cours d'évaluation, que nous ne pouvons pas vérifier ici.
- **Biais de longueur (DPO).** La log-probabilité d'une séquence est une somme sur les tokens, donc une réponse plus longue a un log-rapport de plus grande magnitude. SimPO normalise par la longueur.
- **Disponibilité des paires.** KTO n'a besoin que de données pouce haut/bas, bien plus abondantes.
- **Simplicité du pipeline.** ORPO fusionne le fine-tuning supervisé et l'apprentissage des préférences en une seule étape.

**La sur-optimisation demeure.** La source rapporte une étude qui a entraîné DPO, IPO et une troisième méthode directe sur une gamme de budgets KL et a de nouveau trouvé la courbe or montée-sommet-descente. Les méthodes directes n'ont pas de modèle de récompense explicite, mais la récompense implicite $\beta\log(\pi_\theta/\pi_{ref})$ est interrogée sur des séquences éloignées des données, avec la même conséquence. La loi de Goodhart s'est déplacée ; elle n'est pas partie.

### Choisir

La source propose une règle empirique (pas une loi) : des données appariées et un $\beta$ modéré pour le DPO ; SimPO si le biais de longueur est visible ; KTO pour un retour non apparié ; ORPO pour un pipeline en une étape ; BPO si les log-probabilités des réponses choisies baissent ; IPO si les forces de préférence varient largement. Les laboratoires en essaieraient plusieurs et garderaient le meilleur par tâche.

## IA constitutionnelle et RLAIF

**Phase 1 (supervisée).** Échantillonner une réponse à un prompt risqué, demander à un modèle de la critiquer par rapport à un principe choisi au hasard, puis de la réviser. Faire un fine-tuning sur les réponses révisées. L'article de 2022 qui a introduit la méthode utilisait une petite liste de principes ; la leçon source dit 16.

**Phase 2 (RL à partir de retour d'IA).** Pour des paires de réponses, un modèle de retour choisit la meilleure d'après des principes échantillonnés. Les préférences d'IA qui en résultent entraînent un modèle de récompense (ou alimentent une perte de type DPO), et le reste du pipeline est de forme RLHF. « RLAIF » signifie seulement que le signal de préférence est généré par un modèle.

**Pourquoi ce n'est pas simplement du « RLHF moins cher ».** L'erreur change de nature. Les annotateurs humains sont en désaccord et font des erreurs indépendantes ; avec assez d'entre eux, le bruit se moyenne. Un annotateur IA est cohérent, donc il applique *la même inclinaison* à chaque élément. Ses erreurs sont systématiques : un biais (par exemple vers des réponses plus longues ou plus assurées, ou une lecture particulière de « sois honnête ») ne diminue pas quand tu annotes plus de données. L'avantage est que le signal est **lisible** : tu peux lire le principe, la critique et la révision, ce que ne permettent pas des notes humaines opaques. La source note aussi que la complaisance envers l'utilisateur peut baisser, puisque le juge IA n'a pas d'utilisateur à contenter, tandis que la loi de Goodhart persiste parce que le proxy est maintenant « l'interprétation des principes par un modèle ».

**La réécriture de la constitution de 2026 (telle que la décrit la source).** La source rapporte une révision de janvier 2026 de la constitution d'Anthropic qui privilégie les explications par rapport aux règles nues, ordonne quatre priorités (éviter les issues catastrophiques, suivre les directives du développeur, être globalement éthique, être utile), exprime une incertitude sur le statut moral des modèles, et est publiée sous licence CC0. Nous ne la lisons que comme un résumé à l'intérieur de la leçon du dépôt, pas le document primaire, donc vérifie avec l'original avant de te fier aux détails.

**Classifieurs constitutionnels.** Une approche parallèle entraîne de petits classifieurs sur des principes pour filtrer les entrées et sorties plutôt que de modifier le modèle. La source rapporte un surcoût de calcul passant d'environ 24 % dans la première version à environ 1 % dans la seconde, et aucun jailbreak universel signalé début 2026. Ce sont des chiffres rapportés par les développeurs. La leçon de conception est la **superposition de couches** : le post-entraînement façonne les tendances, les classifieurs imposent des invariants, et aucun n'est suffisant seul.

## Exemple détaillé

Les nombres viennent des exercices.

**Pertes sur une paire.** Log-probabilités de séquence : politique $(-10,-12)$ pour (choisie, rejetée), référence $(-11,-11)$. Log-rapports $\Delta_w=1$, $\Delta_l=-1$, donc $\Delta_w-\Delta_l=2$.

- DPO avec $\beta=0.1$ : marge $0.2$, perte $=-\log\sigma(0.2)=0.598$. Avec $\beta=1$ : marge 2, perte $0.127$.
- IPO : cible $1/(2\beta)$. Avec $\beta=0.1$ la cible est 5, perte $(2-5)^2=9$ ; avec $\beta=1$ la cible est 0.5, perte $(2-0.5)^2=2.25$. L'IPO punit le *dépassement* de la cible, le DPO continue de récompenser des marges plus grandes.
- SimPO avec des longueurs 5 et 9, $\log\pi$ = $-10$ et $-18$, $\beta=2$, $\gamma=0.5$ : les log-probabilités moyennes par token sont $-2$ et $-2$, donc la marge est $0-0.5$ et la perte est $0.974$. La somme brute aurait favorisé la séquence la plus courte ; la moyenne les traite à égalité.
- Terme ORPO avec $p_w=0.6,p_l=0.3$ : log du rapport de cotes $1.253$, perte $0.251$.

**Le DPO retrouve-t-il la vraie récompense ?** Prends la distribution jouet de [[saf.01]] avec $\beta=0.5$. Calcule l'optimum exact $\pi^*$ et la récompense implicite $\beta\log(\pi^*/\pi_{ref})$ par réponse : $(-0.036,\ 0.264,\ -1.036)$. Les différences coïncident exactement avec les différences de vraie récompense (la réponse correcte moins la fausse vaut $1.0$ dans les deux cas). La constante $\beta\log Z$ est la seule chose que la perte ne peut pas voir, c'est pourquoi les récompenses ne sont identifiées qu'à un décalage près par prompt.

**Erreur aléatoire contre erreur systématique de l'annotateur** (second exercice). Les éléments ont un écart de qualité réel $d$ et un écart de longueur indépendant $l$. Un annotateur humain avec un bruit unitaire obtient le signe de $d$ correct environ 75 % du temps ; des majorités de 5, 15 et 51 atteignent environ 84 %, 90 % et 95 %. Un annotateur IA déterministe qui ajoute $0.8\,l$ à $d$ a raison environ 78,5 % du temps -- mieux qu'un seul humain -- et y reste quel que soit le nombre de fois qu'on lui demande, parce que son erreur est la même à chaque fois.

## Implémentation

```python
import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    m = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -np.log(sigmoid(m))

def dpo_grad(lp_w, lp_l, ref_w, ref_l, beta):
    m = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -beta * sigmoid(-m), beta * sigmoid(-m)
```

En production (par exemple les entraîneurs de la bibliothèque TRL), les mêmes pertes sont calculées sur des batchs de log-probabilités par token sommées sur les tokens de la réponse, avec les log-probabilités du modèle de référence mises en cache.

## Dimensions

Pour un batch de $B$ paires, le modèle produit des log-probabilités par token de forme `(B, T)` pour chacune des réponses choisie et rejetée ; masque le padding, somme sur $T$ pour obtenir des log-probabilités de séquence `(B,)`, et la perte est un vecteur de forme `(B,)` moyenné sur le batch. SimPO divise chaque somme par les longueurs de réponse `(B,)`.

## Idées reçues

- **« Le DPO n'a pas de modèle de récompense, donc il ne peut pas faire de reward hacking. »** Il en a un implicite ; la source rapporte les mêmes courbes de sur-optimisation.
- **« Une marge DPO plus élevée signifie un meilleur modèle. »** La marge n'est pas bornée et peut croître en abaissant la probabilité de la rejetée tandis que celle de la choisie baisse aussi.
- **« Le retour d'IA est sans biais parce qu'il est cohérent. »** Cohérent signifie biaisé *systématiquement* ; davantage d'étiquettes ne le moyennent pas.
- **« Une constitution fait suivre les principes au modèle. »** Elle change le signal d'entraînement. Que le comportement suive est une question empirique à tester (voir [[saf.04]]).
- **« La variante la plus récente est la meilleure. »** Le propre conseil de la source est d'en exécuter plusieurs et de choisir par tâche.

## Points clés à retenir

- Le DPO découle de l'optimum RLHF à forme close : la fonction de partition s'annule dans la différence de Bradley-Terry, laissant une perte en log-rapports de politiques.
- Chaque variante répare une hypothèse : IPO (marge non bornée), KTO (disponibilité des paires), SimPO (biais de longueur, référence), ORPO (pipeline en deux étapes), BPO (réponses choisies dégradées).
- Les méthodes directes sur-optimisent encore ; le proxy est implicite plutôt qu'absent.
- Le RLAIF échange le bruit aléatoire des annotateurs contre un biais systématique et lisible ; la superposition avec des classifieurs est la réponse pratique.
- Plusieurs chiffres récents dans ce domaine proviennent de pré-prints ou de rapports de laboratoires résumés de seconde main ; traite-les comme des affirmations à vérifier.

## Explique-le avec tes mots

1. Explique pourquoi la fonction de partition disparaît dans la dérivation du DPO et ce que cela implique sur ce que la récompense implicite peut et ne peut pas identifier.
2. Une équipe remplace 50 000 comparaisons humaines par 500 000 comparaisons d'IA issues d'un seul modèle juge. Quel type d'erreur gagne-t-elle, et lequel perd-elle ?

## Pour aller plus loin

- Les articles nommés dans les leçons sources : l'article DPO, IPO (Azar et al.), KTO (Ethayarajh et al.), SimPO (Meng et al.), ORPO (Hong et al.), l'article sur l'IA constitutionnelle (arXiv:2212.08073) et l'étude de sur-optimisation pour les algorithmes d'alignement direct (arXiv:2406.02900). La vue ingénierie de l'alignement est dans [[llm.05]].

## Critères de maîtrise

- Dériver la perte DPO à partir de l'optimum régularisé par KL et dire où $Z(x)$ s'annule.
- Calculer à la main les pertes DPO, IPO et SimPO pour une paire.
- Nommer l'échec que cible chacun de IPO, KTO, SimPO, ORPO et BPO.
- Expliquer la différence entre erreur d'annotateur aléatoire et systématique avec un nombre.
