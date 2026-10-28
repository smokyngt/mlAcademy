---
id: ops.05
title: "Déploiement et services de prédiction"
summary: "Choisir entre prédiction par lots et en ligne, et entre cloud et edge, réduire les modèles par quantification, distillation et élagage, démonter les mythes du déploiement, et simuler comment les files d'attente et le batching dynamique façonnent la latence et le débit."
minutes: 70
teaches: [ops.serving, ops.queueing, ops.compression]
requires: [ops.requirements, ops.skew, prob.rv, num.floatingpoint]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.7 Model Deployment and Prediction Service: deployment myths 1-4, batch versus online prediction, unifying batch and streaming pipelines (PDF p.221-236, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.7: Model Compression (low-rank factorisation, knowledge distillation, pruning, quantisation, Roblox case study) and ML on the Cloud and on the Edge (compilation, browsers) (PDF p.237-257, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.1 latency versus throughput and batching (PDF p.38-40), re-read for the simulation. READ."
---

## Où cela s'insère

Un modèle qui vit dans un notebook n'aide personne. Le **déploiement** rend les prédictions disponibles aux personnes ou aux systèmes qui en ont besoin, et c'est là que les exigences de [[ops.01]] (percentiles de latence, échelle, coût) deviennent des contraintes dures. Cette leçon répond à trois questions de conception. *Quand* calcule-t-on les prédictions : avant qu'on les demande (batch) ou quand on les demande (en ligne) ? *Où* le modèle s'exécute-t-il : dans le cloud ou sur l'appareil de l'utilisateur ? *Quelle taille* de modèle peut-on se permettre, et comment réduire un modèle trop gros ? Tu construiras aussi un petit simulateur qui montre pourquoi un serveur à 80 % d'utilisation peut avoir une latence au 99e percentile terrible, et comment regrouper les requêtes en lots sauve un modèle surchargé. La cohérence entre les pipelines d'entraînement et de serving ([[ops.03]]) compte le plus ici, et ce qui se passe après le déploiement est le sujet de [[ops.06]].

## L'idée en termes simples

Une boulangerie peut cuire le pain pendant la nuit et le vendre depuis l'étagère (**batch** : peu coûteux par pain, remis instantanément, mais tu dois deviner à l'avance ce que les clients voudront et le pain peut être rassis) ou cuire à la commande (**en ligne** : toujours frais et sur mesure, mais le client attend et il te faut des fours prêts pour chaque coup de feu). Une boulangerie avec un four et des clients qui arrivent au hasard aura une file d'attente même si le four est inactif 20 % du temps : les arrivées viennent par grappes. Si les clients arrivent plus vite que le four ne peut servir, la file grandit sans limite. Cuire plusieurs pains à la fois rend chaque pain légèrement plus lent mais permet au four de servir bien plus de monde par heure, donc une file qui explosait peut disparaître. C'est le batching dynamique.

Mettre le four dans la cuisine de chaque client (**edge**) supprime le délai de livraison et garde ses ingrédients privés, mais leurs fours sont faibles et tu ne peux pas voir comment ils se comportent. Réduire la recette (**compression**) la fait tenir dans les petits fours.

## Mythes du déploiement

Le livre énumère plusieurs croyances de personnes qui n'ont déployé que dans le monde académique ; quatre d'entre elles valent la peine d'être retenues :

1. **« On déploie un ou deux modèles. »** Les vraies entreprises en ont beaucoup : une application de VTC a besoin de modèles pour la demande, l'heure d'arrivée estimée, la tarification, la fraude et le churn, peut-être par pays, donc 10 modèles dans 20 pays font déjà 200. L'infrastructure doit gérer de nombreux modèles.
2. **« Si on ne fait rien, la performance reste la même. »** Les logiciels vieillissent (« bit rot ») et les données dérivent ([[ops.06]]), donc un modèle est à son meilleur juste après l'entraînement et se dégrade.
3. **« On n'aura pas besoin de mettre à jour souvent les modèles. »** La bonne question n'est pas « à quelle fréquence dois-je mettre à jour » mais « à quelle fréquence *puis-je* mettre à jour ».
4. **« La plupart des ingénieurs n'ont pas à se soucier du passage à l'échelle. »** La plupart des ingénieurs travaillent dans des entreprises d'au moins 100 personnes, dont les systèmes servent probablement beaucoup d'utilisateurs.

## Prédiction par lots contre prédiction en ligne

- **La prédiction par lots** (batch) précalcule les prédictions selon un calendrier et les stocke (par exemple la propension d'achat de tous les clients), puis les consulte. Le débit est élevé, la latence au moment de la requête n'est qu'une consultation, et le modèle peut être arbitrairement lent. Le coût : les résultats ne sont aussi frais que le dernier lot (un utilisateur qui vient de passer de l'horreur à la comédie voit encore de l'horreur), et tu dois savoir à l'avance quelles entrées seront interrogées, ce qui est impossible pour des entrées ouvertes comme un texte arbitraire à traduire.
- **La prédiction en ligne** calcule une prédiction à l'arrivée de la requête. Elle peut utiliser l'information la plus fraîche et gérer n'importe quelle entrée, mais la latence est sur le chemin critique, donc le modèle, l'extraction de features et le réseau comptent tous.

De nombreux systèmes de production mélangent les deux : des *features batch* (qui changent lentement, comme la note à long terme d'un conducteur) jointes à des *features en flux* (courses demandées dans la dernière minute). Le bogue classique en découle : l'entraînement calcule une feature en batch sur un dataframe tandis que le serving calcule la même feature sur une fenêtre glissante dans un flux. **Deux pipelines séparés dérivent l'un de l'autre**, surtout quand deux équipes en sont propriétaires. Le remède est la définition partagée et les contrôles de skew de [[ops.03]], ou une infrastructure unifiée stream/batch.

Une règle empirique issue de la discussion du livre : le batch est en partie un héritage des systèmes batch existants ; à mesure que l'infrastructure de streaming mûrit, davantage de systèmes migrent vers la prédiction en ligne parce que la fraîcheur compte.

## Latence et débit comme problème de file d'attente

Les requêtes arrivent à des instants aléatoires et un serveur les traite une par une (ou par lots). Deux nombres décrivent une file à un serveur : le taux d'arrivée $\lambda$ (requêtes par seconde) et le taux de service $\mu=1/\text{(service time)}$. L'**utilisation** est $\rho=\lambda/\mu$.

La **récurrence de Lindley** donne une simulation exacte et simple d'un serveur unique premier arrivé, premier servi. Soit une requête $i$ qui arrive à l'instant $a_i$ et nécessite un temps de service $s_i$. Elle démarre quand elle est arrivée *et* que le serveur a fini la requête précédente, finit $s_i$ plus tard, et sa latence est le temps entre l'arrivée et la fin :

$$
\text{start}_i=\max(a_i,\ \text{finish}_{i-1}),\qquad
\text{finish}_i=\text{start}_i+s_i,\qquad
\text{latency}_i=\text{finish}_i-a_i .
$$

Si les arrivées sont de Poisson (aléatoires, sans mémoire) et les temps de service exponentiels, le système est une **file M/M/1**, et la théorie donne une latence moyenne (attente plus service) de

$$
W=\frac{1}{\mu-\lambda},\qquad \text{stable only if }\lambda<\mu .
$$

Regarde ce que cela dit : avec $\mu=10$ requêtes par seconde, $\lambda=8$ donne $W=0.5$ s, $\lambda=9$ donne $1.0$ s, $\lambda=9.5$ donne $2.0$ s, et quand $\lambda\to\mu$ la latence explose. Le service lui-même ne prend que 0,1 s en moyenne ; presque toute la latence est de l'attente. Planifier la capacité à 100 % d'utilisation est donc une erreur, et **les latences de queue sont bien pires que les moyennes**.

### Batching dynamique

Un GPU ou un CPU vectorisé met souvent presque le même temps pour traiter 16 entrées que 1. Modélise le temps de lot par $a+b\cdot\text{size}$ avec un surcoût fixe $a$ et un coût par élément $b\ll a$. Alors la capacité pour une taille de lot $B$ est $B/(a+bB)$ requêtes par seconde, bien supérieure à $1/(a+b)$. Le serveur doit décider quand démarrer un lot. Une règle courante : démarrer quand le lot est plein (`max_batch` requêtes) ou quand la plus ancienne requête a attendu `max_wait`, selon ce qui arrive en premier (et jamais avant que le serveur soit libre). Des lots plus grands augmentent le débit mais ajoutent de l'attente ; `max_wait` borne cette attente.

## Compression de modèles

Si le modèle est trop gros ou trop lent pour l'objectif de latence ou l'appareil, réduis-le.

- **La quantification** stocke les poids (et les activations) avec moins de bits. Un modèle de 100 millions de paramètres en 32 bits occupe $100\text{M}\times4$ octets $=400$ Mo ; en 16 bits, 200 Mo ; en entiers 8 bits, 100 Mo. Des nombres plus petits se calculent aussi plus vite et permettent des lots plus grands. Un schéma courant associe un poids $w$ à un entier 8 bits $q=\mathrm{round}(w/s)$ avec l'échelle $s=\max|w|/127$ ; l'erreur d'arrondi par poids est d'au plus $s/2$. Les coûts : une plage de valeurs représentables plus petite (les valeurs aberrantes doivent être écrêtées ou mises à l'échelle) et des erreurs d'arrondi qui peuvent parfois changer l'accuracy plus que prévu. Elle peut se faire pendant l'entraînement (quantification consciente de l'entraînement) ou après (quantification post-entraînement, offerte par les frameworks embarqués). Dans l'étude de cas Roblox du livre, passer de flottants 32 bits à des entiers 8 bits a donné la plus grande accélération de tous leurs changements (environ 7x moins de latence et 8x plus de débit), mais le livre note aussi que la source ne rapporte pas l'effet sur la qualité des sorties.
- **La distillation de connaissances** entraîne un petit *élève* à imiter un grand *professeur* (ou ensemble). DistilBERT conserve environ 97 % du score de compréhension du langage de BERT avec 40 % de paramètres en moins et une inférence 60 % plus rapide, selon le livre. Elle fonctionne entre architectures mais nécessite un professeur.
- **L'élagage** (pruning) retire les paramètres qui comptent le moins. Soit on retire des unités entières, soit (plus souvent) on met de nombreux poids à zéro, ce qui donne un réseau creux qui se stocke dans moins d'espace ; des rapports dépassent 90 % de paramètres non nuls en moins sans perte, bien que l'intérêt de l'élagage ait été débattu.
- **La factorisation de bas rang** remplace de grands tenseurs par des produits de plus petits (par exemple, les MobileNets scindent une convolution standard $K \times K$ en une convolution depthwise et une pointwise ; le livre rapporte une réduction des paramètres de huit à neuf fois pour des noyaux $3 \times 3$). Cette famille est liée à des architectures particulières.

En pratique, la quantification est la première chose à essayer parce qu'elle est générale et peu coûteuse ; les autres sont plus spécialisées.

## Cloud contre edge

Le déploiement **cloud** est le point de départ facile (services managés) mais les coûts croissent avec le trafic (les grandes entreprises dépensent des sommes énormes), chaque requête nécessite un aller-retour réseau, et les données de l'utilisateur doivent voyager jusqu'au serveur. Le déploiement **edge** exécute le modèle sur des téléphones, navigateurs, voitures, caméras ou puces embarquées. Avantages : il fonctionne hors ligne, évite la latence réseau (souvent plus grande que la latence d'inférence), réduit les factures du cloud, et garde les données sensibles sur l'appareil. Défis : un matériel faible (donc de la compression), de nombreux backends matériels avec des primitives de calcul et des organisations mémoire différentes (donc il faut des compilateurs qui traduisent les modèles pour chaque puce), et des mises à jour et un monitoring plus difficiles. Une façon d'être portable est d'exécuter dans le navigateur, en compilant les modèles vers WebAssembly plutôt que vers le JavaScript lent. De nombreux systèmes sont hybrides : un petit modèle sur l'appareil avec un plus gros dans le cloud.

## Exemple détaillé

Tous les nombres viennent des simulations de l'exercice (graines fixes).

*M/M/1.* Arrivées de Poisson à $\lambda=8$ par seconde, service exponentiel avec $\mu=10$, 50 000 requêtes. Théorie : latence moyenne 0,5 s. Simulation : 0,501 s. Le 99e percentile est de 2,22 s, plus de quatre fois la moyenne, et le serveur est inactif 20 % du temps.

*Batching.* Arrivées à environ 61 requêtes par seconde (6 000 requêtes), temps de lot $0.02+0.002\cdot\text{size}$ secondes. Un service une requête à la fois prend 0,022 s, donc la capacité est de 45,5 requêtes par seconde : la file est instable, latence moyenne 17,1 s et p99 33,6 s, croissant avec le nombre de requêtes. Avec `max_batch=16, max_wait=0.02`, le serveur forme des lots de 2,27 requêtes en moyenne, la latence moyenne est de 0,0405 s et le p99 de 0,0535 s. Autres réglages : `max_batch=4, max_wait=0.01` donne une moyenne de 0,0366 s, p99 0,056 s ; `max_batch=32, max_wait=0.05` donne une moyenne de 0,0606 s, p99 0,086 s (plus d'attente pour un lot plus rempli). Le batching peut faire la différence entre un service inutilisable et un service sain, et le meilleur réglage est un compromis trouvé par simulation ou test de charge.

## Implémentation

```python
import numpy as np

def fifo_latencies(arrivals, service):
    lat = np.empty(len(arrivals)); free = 0.0
    for i in range(len(arrivals)):
        start = max(arrivals[i], free)
        free = start + service[i]
        lat[i] = free - arrivals[i]
    return lat

def mm1_mean_latency(lam, mu):
    return 1.0 / (mu - lam)      # requires lam < mu
```

Le simulateur de batching de l'exercice démarre un lot à `max(server_free, min(arrival_of_oldest + max_wait, arrival_of_last_slot))` et inclut chaque requête, parmi les `max_batch` suivantes, qui est arrivée à ce moment-là.

Outils de production : frameworks de serving avec batching dynamique (TorchServe, NVIDIA Triton, TensorFlow Serving), `torch.quantization`, TensorFlow Lite, ONNX Runtime, TensorRT, et des compilateurs comme TVM ou XLA pour du code spécifique au matériel.

## Dimensions

`arrivals: (n,)` instants triés ; `service: (n,)` ; latences `(n,)` ; tailles de lots `(n_batches,)`. La quantification associe un tenseur de flottants de forme quelconque à un tenseur d'entiers de même forme plus une échelle scalaire (par tenseur) ou une par canal.

## Idées reçues

- « 80 % d'utilisation, c'est confortable. » Le délai de file domine et la queue est bien au-dessus de la moyenne.
- « Le batching ne fait qu'ajouter de la latence. » Il ajoute de l'attente mais augmente la capacité ; sous charge, il baisse la latence.
- « L'en ligne est toujours meilleur que le batch. » Le batch est plus simple et moins coûteux quand la fraîcheur n'est pas essentielle.
- « La quantification est gratuite. » Elle peut changer l'accuracy ; relance toujours l'évaluation hors ligne, y compris par tranches ([[ops.04]]).
- « Edge signifie pas de cloud. » Les conceptions hybrides sont courantes.

## Points clés à retenir

- Choisis le batch pour des entrées peu coûteuses et prévisibles et une fraîcheur relâchée ; l'en ligne pour des entrées fraîches et ouvertes ; méfie-toi des pipelines d'entraînement et de serving séparés.
- La latence sous charge est dominée par la file d'attente ; la latence moyenne en M/M/1 est $1/(\mu-\lambda)$ et explose quand $\lambda\to\mu$.
- Le batching dynamique échange une attente bornée contre une capacité bien plus élevée.
- La quantification est la technique de compression la plus générale ; la distillation, l'élagage et la factorisation de bas rang la complètent.
- L'edge supprime la latence réseau et garde les données en local au prix d'un matériel plus faible et d'opérations plus difficiles.

## Explique-le avec tes mots

1. Un service avec un temps de service moyen de 0,1 s reçoit 9,5 requêtes par seconde. Estime la latence moyenne et explique pourquoi le p99 est pire.
2. Pourquoi le batching peut-il réduire la latence alors que chaque requête attend maintenant les autres ?

## Pour aller plus loin

- Huyen, *Designing Machine Learning Systems*, Ch.7 (PDF p.217-263) ; Kleppmann, *Designing Data-Intensive Applications* sur le temps de réponse et les files d'attente (référencé par le livre, non lu pour cette leçon).
- [[ops.06]] pour ce qu'il faut surveiller après le déploiement.

## Critères de maîtrise

Tu sais choisir entre prédiction par lots et en ligne et entre cloud et edge pour un produit donné, calculer la latence M/M/1, expliquer le comportement de la queue, simuler le batching dynamique et choisir ses paramètres, et appliquer et critiquer la quantification, la distillation et l'élagage.
