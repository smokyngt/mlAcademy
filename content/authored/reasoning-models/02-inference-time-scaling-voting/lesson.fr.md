---
id: rsn.02
title: "Mise à l'échelle à l'inférence : échantillonnage, vote et best-of-n"
summary: "Dépenser plus de calcul au moment de répondre, en échantillonnant de nombreuses chaînes de pensée et en votant, en choisissant celle qui a le meilleur score, ou en plafonnant le budget de réflexion, et apprendre à prédire combien de précision rapporte chaque échantillon supplémentaire grâce à un modèle binomial."
minutes: 80
teaches: [rsn.vote]
requires: [rsn.verifier, llm.sampling, app.self-consistency, prob.rv]
sources:
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (Sec. 4.1-4.6): chain-of-thought prompting, temperature scaling, top-p filter, self-consistency vote, MATH-500 results table with accuracy and run time. READ."
  - "reasoning-from-scratch/ch05/01_main-chapter-code/ch05_main.ipynb (Sec. 5.3 heuristic score used for best-of-N and as tie-breaker): READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (Sec. 3.4, 3.9): extraction used to compare answers; max_new_tokens in evaluation. READ."
  - "reasoning-from-scratch/chE/chE_main.ipynb (Sec. E.5 batched self-consistency sampling, E.8 table): READ."
---

## Où cela s'inscrit

[[rsn.01]] t'a donné un moyen fiable de savoir si une réponse est juste et a montré qu'une chaîne de pensée plus longue représente plus de calcul. Cette leçon pose la question suivante : **pour un modèle fixé, comment échange-t-on du calcul d'inférence contre de la précision ?** Rien n'est entraîné ici. Les poids du modèle restent gelés ; on ne change que la façon de l'utiliser. Les idées (échantillonner plusieurs fois et voter, ou laisser un évaluateur choisir) sont les outils les moins chers de la boîte à outils du raisonnement et la référence que les méthodes d'entraînement ([[rsn.04]], [[rsn.05]]) doivent battre. La leçon suivante ([[rsn.03]]) examine une alternative séquentielle, le raffinement.

Elle s'appuie sur le décodage et l'échantillonnage ([[llm.02]]) et sur l'idée de vote que tu connais peut-être des ensembles.

## L'idée en mots simples

Pose une énigme difficile à une personne et elle peut se tromper. Pose-la à neuf personnes indépendantes et prends la réponse que la plupart donnent : les erreurs sont surtout différentes les unes des autres, donc elles s'annulent, tandis que la bonne réponse, atteinte par de nombreux chemins différents, se répète. C'est l'**auto-cohérence** (*self-consistency*) : échantillonner plusieurs fois le même modèle et garder la réponse finale la plus fréquente.

Deux raffinements. Si les personnes peuvent aussi dire *à quel point* elles sont sûres, ou si tu disposes d'un arbitre capable de juger une réponse, tu peux prendre la seule réponse que l'arbitre préfère au lieu de compter les têtes : c'est le **best-of-n**. Et tu peux contrôler *combien de temps* chaque personne a le droit de réfléchir : fixer un plafond strict (pour que personne ne divague) ou l'inciter à continuer quand elle s'arrête trop tôt. C'est le **budget de réflexion**.

Les trois coûtent plus de calcul. Combien de précision obtient-on par unité de coût, et quand cesse-t-on d'y gagner ? Un petit modèle probabiliste répond à cela.

## Rendre les réponses différentes : température et top-p

Le décodage glouton donne la même réponse à chaque fois, donc voter sur des exécutions gloutonnes n'a aucun sens. Il nous faut des chaînes de pensée *diverses* mais toujours *sensées*. Deux boutons issus de [[llm.02]] :

- La **température** $T>0$ divise les logits avant le softmax : $p_i=\exp(z_i/T)/\sum_j\exp(z_j/T)$. $T<1$ affine la distribution, $T>1$ l'aplatit.
- Le **top-p (noyau)** garde le plus petit ensemble des tokens les plus probables dont la masse de probabilité atteint $p$, met le reste à zéro et renormalise. La variante du livre garde un token tant que la masse cumulée *avant* lui est inférieure à $p$ (et garde toujours le premier), ce qui inclut le token qui franchit le seuil.

Sur les logits jouets à dix tokens `[-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]` avec $T=1$, la plus grande probabilité est $0.4538$ ; avec $T=0.5$ elle est $0.6947$ et avec $T=5$ seulement $0.1567$. Triées, les masses cumulées sont $0.4538,\ 0.7290,\ 0.8119,\dots$ ; avec un top-p $=0.8$, les deux premiers tokens ont une masse avant de $0$ et $0.4538$, le troisième a une masse avant de $0.7290<0.8$, donc trois tokens survivent, renormalisés en $0.5589,\ 0.3390,\ 0.1021$ (indices de tokens 4, 9, 8).

L'échantillonnage coûte de la précision pour une réponse unique : dans la table du livre, le modèle de base obtient 15,2 % en glouton et 17,8 % avec température et top-p, alors qu'avec le prompting par chaîne de pensée le glouton obtient 40,6 % mais l'échantillonné 33,4 %. L'échantillonnage ne rapporte que quand on peut combiner de nombreux échantillons.

## Auto-cohérence : le vote à la pluralité

Algorithme (livre, Sec. 4.6) : échantillonner $n$ réponses avec, par exemple, $T=0.8$ et un top-p $=0.9$ ; extraire chaque réponse finale ([[rsn.01]]) ; compter ; renvoyer la plus fréquente. Le livre note qu'il s'agit strictement d'un vote à la **pluralité** (une majorité demande plus de la moitié). Si le meilleur décompte est partagé, le livre ne renvoie aucune réponse ; tu peux départager avec un évaluateur, par exemple la log-probabilité moyenne ou le score heuristique de [[rsn.03]].

Deux conseils pratiques du notebook : si toutes les longues réponses se ressemblent presque, augmente un peu la température ; si elles semblent aberrantes, baisse-la.

## Best-of-n avec un évaluateur

Au lieu de compter, on **note** chacune des $n$ réponses et on garde la meilleure. L'évaluateur peut être une règle (l'heuristique du livre : bonus pour la présence d'une boîte, bonus de brièveté), la propre confiance d'un modèle ([[rsn.03]]), ou un LLM jouant le juge. Avec un vérificateur *parfait*, le best-of-$n$ est égal à pass@$n$ ([[rsn.01]]) car on garde simplement un échantillon correct s'il en existe un. Le piège est qu'au moment de répondre on n'a généralement **aucune** réponse de référence, donc l'évaluateur n'est qu'un substitut, et un substitut peut être trompé : optimiser fortement contre un évaluateur imparfait choisit des réponses qui *ont l'air* bonnes pour lui.

## Un modèle probabiliste du vote

Supposons que chaque échantillon soit indépendant et correct avec la probabilité $p$ (pour ce problème), et que tout échantillon faux donne la même unique mauvaise réponse. Avec $n$ impair, la bonne réponse gagne si et seulement si plus de la moitié des échantillons sont corrects :

$$
A(n,p)=\sum_{c>n/2}\binom{n}{c}p^{c}(1-p)^{n-c}.
$$

(Pour $n$ pair, une égalité $c=n/2$ vaut $\tfrac12$ si elle est départagée au hasard, $0$ si on ne renvoie rien.)

Conséquences que tu peux vérifier avec l'exercice :

1. **Seuil à un demi.** Pour $p>\tfrac12$, $A(n,p)\to1$ quand $n$ croît (loi des grands nombres : la fraction de corrects converge vers $p$). Pour $p<\tfrac12$, $A(n,p)\to0$ : le vote aggrave un *mauvais* problème. Exactement à $p=\tfrac12$ et $n$ impair, $A=\tfrac12$.
2. **Gains lents près du seuil.** Pour $p=0.6$ : $A(1)=0.6$, $A(3)=0.648$, $A(5)=0.6826$. Deux échantillons avec égalités aléatoires redonnent exactement $0.6$ ; la première étape utile est trois.
3. **Une borne exponentielle.** L'inégalité de Hoeffding donne $1-A(n,p)\le\exp\bigl(-2n(p-\tfrac12)^2\bigr)$. Elle est lâche (pour $n=21,p=0.55$ elle dit $\le0.90$ alors que le vrai taux d'échec est $0.32$) mais elle montre le taux exponentiel en $n$ et que l'écart $p-\tfrac12$ contrôle la vitesse.

### Quand les mauvaises réponses divergent

Les vraies mauvaises réponses sont rarement identiques ; un problème difficile donne souvent de nombreux nombres faux différents. Modélise cela avec $m$ mauvaises réponses également probables. La bonne réponse n'a alors besoin que du *plus grand* décompte, pas de plus de la moitié. Le vote à la pluralité peut alors aider même quand $p<\tfrac12$ : avec $p=0.3$ et $n=9$, le modèle à une seule mauvaise réponse donne $A=0.0988$, mais avec $m=4$ mauvaises réponses également probables la bonne réponse gagne avec la probabilité $0.4339$.

La vérité se situe entre les deux : les modèles tendent à répéter *la même* erreur caractéristique, donc les mauvais échantillons sont corrélés et se regroupent. Un gain important du vote est l'indice que les mauvaises réponses sont dispersées ; un gain faible suggère des erreurs systématiques qu'aucune quantité d'échantillonnage n'annulera.

### Les problèmes diffèrent en difficulté

La précision est une moyenne sur des problèmes de $p$ différents. Prenons six problèmes avec $p=0.95,0.8,0.6,0.4,0.2,0.05$ (moyenne 0,5). Sous le modèle binaire, la précision moyenne du vote reste exactement $0.5$ pour tout $n$ impair : les problèmes faciles gagnent ce que les difficiles perdent. Sous le modèle dispersé avec $m=3$, elle monte : $0.5\to0.5497\ (n=3)\to0.5791\ (n=5)\to0.6113\ (n=9)$. Si on ne renvoie rien en cas d'égalité (la règle du livre), la valeur pour $n=9$ est $0.5827$.

## Le compromis calcul-précision dans les données du livre

MATH-500, 500 problèmes, temps d'exécution sur un GPU DGX Spark, modèle de base avec prompt de chaîne de pensée (« CoT ») et échantillonnage top-p :

| réglage | précision | temps d'exécution |
|---|---|---|
| CoT + top-p, un échantillon | 33.4% | 129.2 min |
| plus auto-cohérence $n=3$ | 42.2% | 211.6 min |
| $n=5$ | 48.0% | 452.9 min |
| $n=10$ | 52.0% | 862.6 min |
| (CoT, glouton, pour référence) | 40.6% | 84.5 min |

La précision marginale par minute supplémentaire diminue : $0.107$ points/min de 1 à 3 échantillons, $0.024$ de 3 à 5, $0.0098$ de 5 à 10. Trois échantillons battent déjà la référence CoT gloutonne ; dix échantillons coûtent environ 6,7 fois le temps d'un échantillon pour $+18.6$ points. Avec le modèle de raisonnement, $n=3$ atteint $55.2\%$ contre $48.2\%$ en glouton.

Deux réserves honnêtes. D'abord, le même notebook montre que sans CoT les résultats du vote ne sont pas monotones ($29.6\%$ à $n=3$, $27.8\%$ à $n=5$, $31.6\%$ à $n=10$) : avec 500 problèmes l'erreur standard est d'environ 2 points ([[rsn.01]]), donc ces différences sont surtout du bruit. Ensuite, le temps n'est pas exactement proportionnel à $n$ (il dépend des longueurs et du matériel), donc considère « coût $\propto n\times$ longueur moyenne » comme un modèle, pas comme une loi.

## Budgets de réflexion

Un **budget** limite le nombre de tokens que le modèle peut dépenser. Le côté plafond est déjà dans le code d'évaluation : `max_new_tokens`. Si la chaîne de pensée est coupée avant la boîte finale, aucune réponse finale n'est produite et, avec un extracteur strict, la tentative compte comme fausse. Dans l'exercice, six tentatives préparées de longueurs $120,350,900,1500,2600,4000$ (et de justesse $1,1,0,1,1,1$) donnent une précision de $2/6$ pour un budget de 1000 tokens, $3/6$ pour 2048, et $5/6$ pour 5000.

Le *budget forcing* de la littérature va aussi dans l'autre sens : quand le modèle essaie de s'arrêter trop tôt, on ajoute un mot comme « Wait » pour qu'il continue à raisonner, et quand il dépasse le budget, on le coupe et on force la réponse finale. Les chapitres que j'ai lus n'implémentent pas cette extension, donc considère ce paragraphe comme du contexte général et non comme quelque chose de vérifié ici.

## Exemple détaillé

Cinq échantillons répondent à un problème dont la bonne réponse est 83 : `83, 83, 38, 83, 5`. Décomptes : 83 en a 3, les autres 1. Vainqueur à la pluralité : `83`. Avec deux échantillons `83, 38`, le meilleur décompte est partagé : pas de vainqueur ; si leurs scores moyens sont $-0.2$ pour 38 et $-1.5$ pour 83, le départage par score moyen renvoie 38 (c'est pourquoi un mauvais évaluateur peut transformer une égalité en erreur). Le best-of-3 sur `[a, a, b]` avec les scores $[-9,-8,-0.1]$ renvoie `b` même si `a` a plus de voix.

Pour la courbe de précision attendue, utilise `plurality_acc` et `mixture_curve` dans l'exercice ; les nombres cités plus haut (0.648, 0.4339, 0.6113, 0.5827) en proviennent.

## Implémentation

```python
from collections import Counter

def plurality_vote(answers):
    counts = Counter(answers)
    top = max(counts.values())
    winners = [a for a, k in counts.items() if k == top]
    return (winners[0] if len(winners) == 1 else None), counts

def majority_acc_binary(n, p):          # odd n
    from math import comb
    return sum(comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n // 2 + 1, n + 1))
```

Les piles de production échantillonnent en parallèle les $n$ complétions d'un même prompt (même longueur de prompt, donc pas de remplissage nécessaire ; voir [[rsn.05]] pour le batching) et partagent le cache KV du prompt ([[sys.01]]).

## Idées reçues

- « Plus d'échantillons aident toujours. » Pas si le modèle se trompe plus souvent qu'il ne réussit sur un problème et que ses mauvaises réponses concordent ; alors le vote amplifie l'erreur.
- « Voter, c'est le vote à la majorité. » C'est la pluralité ; le vainqueur peut avoir bien moins de la moitié des voix.
- « Le best-of-n avec n'importe quel évaluateur est au moins aussi bon qu'un échantillon. » Seulement si l'évaluateur est corrélé à la justesse ; sinon tu paies plus cher pour un choix aléatoire, ou systématiquement faux.
- « Une température plus élevée est toujours plus diverse et donc meilleure. » Elle abaisse aussi la qualité de chaque chaîne individuelle ; le bon réglage équilibre les deux.

## Points clés

- La mise à l'échelle à l'inférence échange du calcul contre de la précision sans changer les poids : échantillonner, puis agréger.
- Le vote à la pluralité sur les réponses extraites est la référence ; les égalités demandent une règle.
- Sous le modèle binomial, le vote aide si et seulement si la bonne réponse est la réponse modale : $p>\tfrac12$ pour une seule mauvaise réponse, bien plus faible si les mauvaises réponses sont dispersées.
- Les gains montrent des rendements décroissants par unité de coût ; mesure avec des barres d'erreur.
- Un plafond de budget fait partie du pipeline : un raisonnement tronqué ne rapporte aucune réponse.

## Explique-le toi-même

1. Un problème a un taux de réussite par échantillon de $0.35$ et les mauvaises réponses sont presque toujours le même nombre. Qu'arrive-t-il à la précision quand $n$ croît, et pourquoi ?
2. Pourquoi le CoT glouton du livre (40,6 %) bat-il le CoT échantillonné unique (33,4 %), et que doit faire le vote avant que son coût en vaille la peine ?

## Pour aller plus loin

- Le chapitre 4 du livre (en entier) et l'annexe E.5 ; l'idée originale d'auto-cohérence est référencée dans le notebook (Wang et al., 2022, non ouvert ici).
- [[rsn.03]] pour la notation et le raffinement ; [[rsn.04]] pour entraîner le modèle de sorte qu'un seul échantillon soit déjà meilleur.

## Critères de maîtrise

- Je sais implémenter le vote à la pluralité avec une règle d'égalité explicite et un départage fondé sur un score.
- Je sais dériver et calculer $A(n,p)$ et expliquer le seuil à $p=\tfrac12$.
- Je sais dire comment des mauvaises réponses corrélées changent le bénéfice du vote.
- Je sais calculer la précision marginale par unité de calcul à partir d'une table de résultats et décider quand cesser d'ajouter des échantillons.
