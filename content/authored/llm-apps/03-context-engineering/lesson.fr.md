---
id: app.03
title: "Ingénierie du contexte : budgets, ordre, compaction et cache de préfixes"
summary: "Décider ce qui entre dans la fenêtre de contexte, dans quel ordre et à quel prix, et simuler comment un cache de préfixes transforme une disposition de prompt stable en économies mesurables."
minutes: 75
teaches: [app.context, app.prompt-cache]
requires: [app.prompting, trf.attention, llm.lm]
sources:
  - "ai-engineering-from-scratch/phases/11-llm-engineering/05-context-engineering/docs/en.md (opening sections: window as scarce resource, token-budget example, lost-in-the-middle, context components; first ~6.5k characters only). READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/11-caching-cost/docs/en.md (opening sections: cost anatomy, provider caching table, semantic and exact caches, token bucket; first ~6.5k characters only). READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/15-prompt-caching/docs/en.md (opening sections: the prefix invariant, cache-friendly layout, write premium and break-even; first ~6.5k characters only). READ."
---

## Où cela s'inscrit

[[app.01]] portait sur les mots d'un prompt. Dans une vraie application, l'entrée du modèle est bien plus qu'un prompt : règles système, définitions d'outils, documents récupérés, historique de conversation, exemples et message de l'utilisateur se disputent tous une seule fenêtre finie, et chaque jeton coûte de l'argent et de l'attention. L'**ingénierie du contexte** consiste à décider ce qui entre, ce qui reste dehors, dans quel ordre, et comment éviter de payer deux fois pour ce qui n'a pas changé. Elle utilise ce que tu sais de l'attention (`trf.attention` : le coût croît avec la longueur de séquence et l'attention n'est pas uniforme), se relie à la recherche d'information (la leçon `sys.03` explique les embeddings et les rouages du RAG ; ici nous décidons seulement *quand* récupérer), et au cache KV qui rend possible le cache de préfixes côté fournisseur (leçon `sys.01`). Elle se termine par un simulateur hors ligne qui chiffre une disposition.

## L'idée en termes simples

Imagine la fenêtre de contexte comme le bureau placé devant un expert. Le bureau est grand mais pas infini, l'expert regarde plus attentivement ce qui est en haut et en bas que le tas du milieu, et chaque feuille que tu ajoutes doit être lue (et tu es facturé par feuille). Un bon assistant ne vide pas tout le classeur sur le bureau. Il garde en haut une courte fiche de règles permanentes, apporte uniquement les documents pertinents pour cette question, remplace la longue correspondance du mois dernier par un résumé d'un paragraphe, et place la vraie question en bas, où elle ne peut pas être manquée.

La deuxième idée concerne la répétition. Si chaque requête commence par le même livre de règles de 4 000 jetons, un fournisseur malin se souvient de l'avoir lu et te facture une petite redevance pour réutiliser ses notes. Mais la mémoire ne fonctionne que si le début est *exactement* le même : change un mot près du haut et tout le reste doit être relu.

## Le budget

Tout compte contre la fenêtre de taille $W$ :

$$
W \;\ge\; T_{\text{system}}+T_{\text{tools}}+T_{\text{retrieved}}+T_{\text{history}}+T_{\text{examples}}+T_{\text{query}}+T_{\text{output}}
$$

La marge de sortie $T_{\text{output}}$ compte : une fenêtre pleine à la fin de l'entrée ne laisse pas de place pour répondre. Une répartition typique d'un assistant de code, tirée de la source que nous avons lue : système 500, définitions d'outils 8 000 (50 outils à environ 150 jetons chacun font 7 500), documentation récupérée 4 000, historique 6 000, requête 200, budget de sortie 4 000. Le total est de $22{,}700$ jetons, soit $22700/128000=17.7\%$ d'une fenêtre de 128K. La fenêtre n'était pas le problème ; c'est la **densité de signal**. Les définitions d'outils non pertinentes et les tours périmés rendent le modèle un peu moins bon et la facture plus grosse.

Une politique d'allocation simple : fixe les budgets des parties que tu ne peux pas réduire (système, requête, sortie), puis donne le reste aux parties par ordre de priorité (preuves récupérées, historique récent, exemples), en élaguant d'abord le matériel de plus faible priorité. Si la liste d'outils est grande, n'inclus que les outils pertinents pour la requête (un sélecteur dynamique d'outils peut en retirer la plupart).

## Ordre et récence

Les modèles n'utilisent pas toutes les positions de façon égale. La source cite une étude (Liu et al., 2023 ; nous n'avons pas lu l'article) dans laquelle un passage pertinent placé parmi une vingtaine de passages non pertinents était mieux utilisé au début ou à la fin du contexte et moins bien au milieu, avec une exactitude chutant de plusieurs dizaines de points dans le pire cas pour certains modèles. L'ampleur varie selon le modèle et la génération ; la forme de la leçon est assez robuste pour agir dessus :

- place les instructions permanentes en premier (ce sont aussi le préfixe stable, voir plus bas) ;
- place la question courante et les preuves les plus pertinentes en dernier ;
- considère le milieu comme la zone la plus faible ; si un fait doit y résider, répète-le près de la fin ;
- moins de passages mieux classés valent mieux que beaucoup de passages médiocres.

Teste cela sur ton propre modèle et tes propres données (une sonde de type aiguille dans une botte de foin est facile à écrire) ; ne suppose pas que les chiffres publiés se transfèrent.

## Compaction : garder l'historique abordable

Les longues conversations et exécutions d'agents dépassent n'importe quelle fenêtre. Options :

| Stratégie | Idée | Faiblesse |
|---|---|---|
| Tronquer | supprimer les tours les plus anciens | perd silencieusement les faits du début |
| Fenêtre glissante | garder les $k$ derniers tours | idem, mais prévisible |
| Résumer | remplacer les anciens tours par un résumé écrit par un modèle | les résumés perdent des détails et peuvent halluciner |
| Mémoire structurée | extraire les faits (préférences, décisions, tâches ouvertes) dans un petit magasin | exige un schéma et une règle de mise à jour |
| Récupérer | tout stocker à l'extérieur, ramener ce qui est pertinent | la récupération peut rater ; ajoute de la latence |

Un défaut robuste est hybride : garde les derniers tours textuellement, un résumé courant des plus anciens, et un magasin de faits durables ; déclenche la compaction quand l'historique dépasse un budget (pas à chaque tour, car réécrire l'historique change le préfixe et casse le cache, voir plus bas).

## Récupération ou long contexte ?

Les longues fenêtres tentent de tout coller. La récupération (leçon `sys.03`) sélectionne quelques passages. Compromis : le long contexte évite les ratés de récupération et la maintenance d'un index mais multiplie le coût par requête et pousse les preuves dans le milieu faible ; la récupération est moins chère et garde le prompt ciblé mais peut manquer le passage nécessaire et ajoute un composant à maintenir. Règle pratique : utilise le long contexte pour un ensemble de documents borné et connu sur lequel tu poses de nombreuses questions (combine avec le cache pour ne le payer qu'une fois), et la récupération pour des corpus grands ou changeants. Mesure les deux sur tes questions ([[app.04]]).

## Cache de préfixes

Les fournisseurs conservent l'état interne du modèle pour un préfixe de prompt vu récemment (c'est le cache KV de la leçon `sys.01`, conservé entre les requêtes). Si ta requête suivante commence par les mêmes jetons, ils n'ont pas besoin d'être retraités, et tu es facturé au tarif réduit de **lecture**. La première fois, tu peux payer une **prime d'écriture**. La comparaison de la source (les prix changent souvent ; consulte la page actuelle du fournisseur) décrit : un schéma à marqueurs explicites avec une grande remise en lecture et une prime d'écriture de 25 pour cent ; un schéma automatique avec une remise plus faible et sans prime ; et un objet de cache que tu crées et stockes moyennant des frais. Tous partagent un invariant :

> Seul un **préfixe** est mis en cache. Le premier jeton qui diffère met fin à la correspondance ; tout ce qui suit est un raté.

La disposition favorable au cache est donc : prompt système, définitions d'outils, exemples few-shot, documents récupérés (s'ils sont réutilisés), historique de conversation jusqu'au dernier tour, et en dernier le nouveau message de l'utilisateur. Un contenu dynamique placé tôt, des listes d'outils réordonnées, des horodatages ou des identifiants de requête dans le prompt système détruisent tous les succès. Les entrées de cache expirent aussi (par défaut quelques minutes chez certains fournisseurs).

### Le calcul du seuil de rentabilité

Soit un bloc qui coûte 1 unité sans cache, $w$ en écriture et $r$ en lecture. Une écriture suivie de $n$ lectures coûte en moyenne, relativement à l'absence de cache,

$$
\frac{w+r\,n}{n+1}.
$$

Avec $w=1.25$ et $r=0.1$ : $n=1$ donne $1.35/2=0.675$, $n=10$ donne $2.25/11=0.2045$. Le cache est rentabilisé dès que $w+rn<n+1$, c'est-à-dire $n>(w-1)/(1-r)=0.278$, donc une lecture suffit ; avec une prime d'écriture de $2\times$ et des lectures à moitié prix, il faudrait $n>2$, soit trois lectures.

## Exemple détaillé : une conversation de 10 tours

L'exercice simule une conversation où chaque requête porte un prompt système de 2 000 jetons et 1 500 jetons de définitions d'outils (tous deux pouvant être mis en cache), les tours précédents de 500 jetons chacun (également pouvant être mis en cache) et un nouveau message utilisateur de 200 jetons (non mis en cache). À un tarif illustratif de 3 dollars par million de jetons d'entrée, avec un multiplicateur d'écriture de 1,25, un multiplicateur de lecture de 0,1 et un TTL de 300 secondes :

- Entrée totale sur 10 tours : $10\cdot3700+500\cdot(0+1+\dots+9)=37{,}000+22{,}500=59{,}500$ jetons.
- Le tour $i\ge1$ lit tout le préfixe du tour précédent : $3500+500(i-1)$ jetons, total $49{,}500$. Taux de succès $49500/59500=83.2\%$.
- Coût avec cache : 0,05085 dollar. Sans : 0,1785 dollar. Économies $71.5\%$.

Modes de défaillance que révèle le même simulateur :

- Une **seule** requête coûte $23.6\%$ de plus que sans cache (les 3 500 jetons pouvant être mis en cache paient la prime d'écriture et rien n'est jamais lu).
- Des tours espacés de 400 s (TTL 300 s) : aucun succès, coût $24.2\%$ **au-dessus** de la référence, parce que tu continues de payer des primes d'écriture.
- **Texte dynamique en premier** : place le message utilisateur au début et le préfixe ne correspond jamais ; coût $25\%$ au-dessus de la référence.

Le cache est une optimisation qu'il faut vérifier : contrôle les compteurs de jetons lus depuis le cache dans les champs d'utilisation du fournisseur, et déclenche une alerte quand ils tombent à zéro.

## Implémentation

Le cœur du simulateur est un dictionnaire d'un préfixe (tuple de clés de segments) vers l'heure de dernière utilisation :

```python
def lookup(cache, keys, t, ttl, ends):
    for k in sorted(ends, reverse=True):          # longest cacheable prefix first
        last = cache.get(tuple(keys[:k]))
        if last is not None and t - last <= ttl:
            return k
    return 0
```

Des caches au niveau applicatif viennent s'y superposer : un **cache exact** (hachage du prompt complet, utile à la température 0) et un **cache sémantique** (une similarité d'embeddings au-dessus d'un seuil renvoie une réponse stockée). Un cache sémantique peut renvoyer une mauvaise réponse pour une question proche mais différente, donc fixe le seuil avec un jeu de test et exclus les requêtes personnalisées ou sensibles au temps.

## Idées reçues courantes

- « Une fenêtre plus grande signifie que je peux cesser de trier. » La qualité et le coût se dégradent tous deux avec le bruit.
- « Résumer est sans perte. » C'est avec perte ; garde un pointeur vers la source.
- « Le cache fait toujours économiser de l'argent. » Un cache qui n'est jamais lu coûte une prime ; vérifie les taux de succès.
- « Je peux mettre la date en haut du prompt système. » Un premier jeton changeant invalide tous les jetons mis en cache qui suivent.
- « La position dans la fenêtre n'a pas d'importance. » Les preuves au milieu sont utilisées de façon moins fiable par de nombreux modèles.

## Points clés à retenir

- Budgétise la fenêtre comme une somme, marge de sortie comprise ; mesure les nombres de jetons, ne devine pas.
- Place le matériel stable en premier, la question et les preuves clés en dernier, répète les faits critiques près de la fin.
- Compacte l'historique avec un mélange hybride de tours récents, de résumé et de faits durables, et fais-le par lots.
- Le cache de préfixes fait économiser $1-\frac{w+rn}{n+1}$ par bloc réutilisé, mais seulement pour des préfixes exacts dans le TTL.
- Choisis récupération ou long contexte selon la taille du corpus, son rythme de changement et l'exactitude mesurée.

## Explique-le toi-même

1. Pourquoi réécrire un résumé de conversation à chaque tour met-il en échec le cache de préfixes, et comment le restructurerais-tu ?
2. Ton taux de succès est de 0 alors que le prompt système est identique d'une requête à l'autre. Liste trois endroits où regarder.

## Pour aller plus loin

- Leçon `sys.01` sur le cache KV ; leçon `sys.03` sur la génération augmentée par récupération.
- La documentation des fournisseurs sur le cache de prompt (les noms de champs et les prix changent).

## Critères de maîtrise

- Tu sais calculer un budget de jetons et dire quoi couper en premier.
- Tu sais ordonner un prompt pour la récence et pour la mise en cache.
- Tu sais calculer le nombre de réutilisations de seuil de rentabilité pour des multiplicateurs d'écriture et de lecture donnés.
- Tu sais expliquer trois manières dont un cache cesse silencieusement de fonctionner.
