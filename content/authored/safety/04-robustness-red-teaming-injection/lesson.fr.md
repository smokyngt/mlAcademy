---
id: saf.04
title: "Robustesse : red-teaming et injection de prompt"
summary: "Le point de vue du défenseur sur les tests adversariaux - les classes d'attaque contre les systèmes à base de modèles de langage, pourquoi l'injection de prompt indirecte est un problème de conception de système, comment s'articulent les outils de red-teaming, et comment rapporter le taux de réussite d'attaque avec une incertitude honnête."
minutes: 75
teaches: [saf.red-teaming, saf.prompt-injection]
requires: [saf.goodhart, prob.basics, stat.testing]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/docs/en.md (whole lesson): automated attack families at a conceptual level, JailbreakBench/HarmBench, ASR and budgets. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/13-many-shot-jailbreaking/docs/en.md (whole lesson): power-law scaling with number of in-context examples, shared mechanism with in-context learning, classifier-based defence. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks/docs/en.md (whole lesson): encoding-level attacks and why filters miss them. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/15-indirect-prompt-injection/docs/en.md (whole lesson): delivery vectors, information-flow control, adaptive-attack evaluation of defences. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit/docs/en.md (whole lesson): classifier, scanner and campaign tools and evaluation pitfalls. READ."
---

## Où cela s'insère

[[saf.01]]-[[saf.03]] portaient sur ce que l'entraînement fait à un modèle. Cette leçon porte sur ce qu'une *partie extérieure* peut faire à un modèle ou à une application déployés, et sur la manière dont un défenseur doit tester cela. Elle est écrite du côté du défenseur : nous décrivons des **classes** d'attaque pour que tu puisses les reconnaître et évaluer des défenses, mais nous ne donnons ni prompts, ni modèles, ni méthodes pas à pas, et les exercices utilisent des données abstraites (comptages, étiquettes) plutôt que du contenu nuisible. Elle s'appuie sur les probabilités et les intervalles de confiance (`prob.basics`, `stat.testing`) et mène à l'évaluation des capacités et à la gouvernance dans [[saf.06]].

## L'idée en termes simples

Une porte verrouillée est testée par des gens qui essaient de l'ouvrir ; le résultat que tu veux n'est pas « personne n'est entré » mais « voici combien d'essais, par qui, avec quels outils, et avec quel degré de confiance ». Tester un système à base de modèle de langage, c'est pareil, avec deux complications :

1. **L'attaquant peut apprendre.** Contrairement à une serrure physique, un attaquant qui voit un refus peut reformuler et réessayer, et les attaquants automatisés le font des milliers de fois. Une défense qui arrête les attaques d'*hier* dit peu de chose sur celles de demain.
2. **Le système mélange instructions et données.** Un programme classique sépare le code des données. Un modèle de langage lit tout comme du texte dans une seule fenêtre ; du texte récupéré sur une page web peut ressembler exactement à une instruction de l'utilisateur. C'est le cœur de l'**injection de prompt**, et cela fait de l'*architecture* de l'application, et pas seulement du modèle, une partie de la défense.

D'où deux questions qui traversent la leçon : *comment mesurer honnêtement la robustesse ?* et *comment concevoir des systèmes qui restent sûrs même quand une entrée est hostile ?*

## Une taxonomie des classes d'attaque (conceptuelle)

Les leçons sources regroupent les attaques publiées comme suit. Les descriptions sont délibérément de haut niveau.

| Classe | Idée | Nécessite | Notes issues des sources |
|---|---|---|---|
| Raffinement itératif en boîte noire | Un modèle attaquant propose des prompts, voit les réponses de la cible et le score d'un juge, et révise | Accès par requêtes seulement | Rapporté comme réussissant en quelques dizaines de requêtes sur certains modèles ; la boucle est une baseline standard de red-team (PAIR) |
| Recherche au niveau des tokens | Recherche guidée par le gradient de chaînes adversariales | Poids du modèle | Produit des chaînes illisibles ; se transfère à d'autres modèles dans certaines études (GCG) |
| Évolutionnaire / recherche arborescente / modèles de persuasion | Recherche parmi des prompts, tentatives ramifiées, ou schémas de persuasion issus des sciences sociales | Accès par requêtes | AutoDAN, TAP et PAP sont les exemples nommés |
| Bourrage d'exemples en contexte long (« many-shot ») | Remplir un long contexte d'exemples du comportement indésirable, puis demander | Longue fenêtre de contexte | L'ASR augmente selon une loi de puissance du nombre d'exemples ; partage un mécanisme avec l'apprentissage en contexte bénin |
| Encodage / obfuscation | Cacher les mots pertinents pour la sécurité sous une forme que le filtre ne reconnaît pas mais que le modèle peut lire (p. ex. du texte dessiné avec des caractères) | Un lecteur capable | Fonctionne parce que les filtres opèrent au niveau des tokens ou du sens tandis que le modèle lit la forme ; la capacité de reconnaissance et la vulnérabilité sont corrélées |
| Injection indirecte | Instructions placées dans un contenu externe que le système lira (une page, un e-mail, un résultat d'outil) | Pouvoir publier un contenu que l'agent verra | La menace dominante en production selon la source ; contourne les filtres d'entrée utilisateur |

Trois observations que les sources soulignent :

- **Capacité et vulnérabilité sont couplées.** Plus un modèle lit bien les encodages inhabituels ou apprend du contexte long, plus ces caractéristiques peuvent être détournées.
- **Un chiffre de benchmark n'a aucun sens sans budget ni juge.** « 90 % d'ASR à 200 requêtes » n'est pas comparable à « 85 % à 20 ».
- **La défense contre le many-shot pose un dilemme.** Le mécanisme est de l'apprentissage en contexte ordinaire, donc supprimer le suivi de motifs depuis les contextes longs casserait le prompting few-shot bénin. La défense rapportée dans la source inspecte le contexte avec un classifieur et modifie ou tronque la partie fautive ; elle rapporte une réussite d'attaque passant de 61 % à 2 % dans les cadres testés (rapporté par les développeurs).

## L'injection de prompt comme problème de systèmes

L'injection **directe** : l'utilisateur tape le texte hostile. L'injection **indirecte** (IPI) : le texte hostile se trouve dans un contenu tiers qu'une application récupère. La source énumère trois voies de livraison : la récupération (un document empoisonné est amené dans le prompt), les flux de travail de boîte de réception/documents (un message que l'agent lit), et les sorties d'outils (un résultat d'outil porte le texte).

Pourquoi les filtres d'entrée la ratent : le filtre surveille le message de l'utilisateur, mais la charge utile arrive par le canal de récupération ou d'outil. Filtrer *tout* le contenu qui atteint le modèle signalerait des pages légitimes qui utilisent simplement un langage impératif.

**Contrôle de flux d'information (IFC).** La source rapporte que c'est l'idée de défense dominante de 2026, empruntée à la sécurité des systèmes d'exploitation : étiqueter chaque contenu par sa fiabilité, propager les étiquettes dans le calcul, et exiger qu'une action déclenchée par un contenu étiqueté non fiable soit ratifiée par une partie de confiance avant d'avoir des effets de bord. Le principe dans la source est *l'endiguement, pas la prévention* : tant que instructions et données partagent une fenêtre de contexte, suppose que certaines injections réussiront et limite ce qu'elles peuvent faire. Les systèmes nommés dans la leçon comprennent CaMeL (2025), ConfAIde (2024) et un article de défense de 2026 ; nous n'avons pas lu ces articles.

**Le risque à trois branches.** Un schéma issu d'écrits de praticiens (pas dans les leçons du dépôt) aide à raisonner sur la conception d'agents : un agent est dans une position particulièrement risquée quand il a à la fois (1) l'accès à des données privées, (2) l'exposition à du contenu non fiable, et (3) la capacité de communiquer vers l'extérieur. Avec les trois, une instruction injectée peut lire des données privées et les envoyer à l'extérieur. Retirer *n'importe laquelle* des branches brise la chaîne, ce qui transforme une inquiétude vague en liste de contrôle de conception. L'exercice implémente à la fois ce contrôle et un petit vérificateur de propagation d'étiquettes.

**Contexte réel.** La source énumère plusieurs identifiants de vulnérabilités publiquement attribués pour de l'injection dans des assistants commerciaux et indique que l'injection de prompt est classée première dans la liste OWASP des risques des applications LLM (édition 2025). Nous n'avons pas vérifié ces références.

## Outils de red-team et pile de défense

Trois outils illustrent différentes couches (selon la source) :

- **Classifieurs de sécurité** (par exemple la famille Llama Guard) : un modèle affiné pour étiqueter les entrées et/ou sorties selon une taxonomie de 14 catégories de dangers. À placer avant le modèle, après, ou les deux. Coût : latence et faux positifs sur du contenu bénin, à mesurer pour chaque déploiement.
- **Scanners** (par exemple Garak) : une *sonde* (probe) génère des entrées de test, un *détecteur* note les sorties, un *harnais* les exécute et rapporte. Les sondes statiques vieillissent à mesure que les modèles sont corrigés ; les sondes adaptatives vieillissent plus lentement. Un score par paliers remplace un simple réussi/échoué.
- **Orchestrateurs de campagnes** (par exemple PyRIT) : campagnes multi-tours avec des convertisseurs (transforment une graine), des orchestrateurs (la stratégie) et des scoreurs.

Une pile typique dans la source : un classifieur de chaque côté du modèle, un scanner à chaque build nocturne pour la régression, des campagnes avant les versions majeures. Non couverts : le processus organisationnel, la divulgation, et les contraintes légales sur les tests.

## Mesurer la robustesse : l'ASR avec incertitude

Le **taux de réussite d'attaque** (attack success rate, ASR) est la fraction des comportements de test pour lesquels une attaque a produit une sortie nuisible ou contraire à la politique selon un **juge** (un modèle ou un classifieur). Sa valeur dépend de : l'ensemble de comportements, l'attaque, le budget de requêtes et le juge. Rapporte les quatre.

Comme l'ASR est une proportion issue de $n$ essais, elle porte une erreur d'échantillonnage. L'intervalle simple par approximation normale est mauvais pour un petit $n$ ou pour des taux proches de 0 ou de 1. Utilise l'**intervalle de Wilson**. Avec $\hat p=k/n$ et $z=1.96$ pour 95 % :

$$
\text{centre}=\frac{\hat p+\frac{z^2}{2n}}{1+\frac{z^2}{n}},\qquad
\text{half-width}=\frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}} .
$$

Il ne sort jamais de $[0,1]$ et, point important, **zéro succès ne signifie pas zéro risque** : avec 0 sur 50, l'intervalle à 95 % s'étend encore jusqu'à environ 7,1 %.

Autres outils :

- **ASR budgétée** : l'ASR à $k$ requêtes est la fraction des comportements dont le premier succès est survenu en au plus $k$ requêtes ; elle est croissante au sens large en $k$.
- **Comparer deux défenses** : une statistique z à deux proportions groupées, $z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}$, avec $\hat p$ le taux groupé.
- **Taille d'échantillon** : pour une demi-largeur $h$ autour du taux $p$, $n\approx z^2p(1-p)/h^2$ (385 pour $p=0.5$, $h=0.05$).
- **Évaluation adaptative contre statique** : un benchmark statique rapporte une attaque ; un attaquant réaliste utilise *n'importe quelle* attaque qui marche, donc la bonne métrique est l'union sur les attaques. La source rapporte une étude dans laquelle chacune de douze défenses publiées contre l'injection, qui avaient montré une ASR quasi nulle contre des attaques fixes, a été brisée à plus de 90 % par des attaques adaptatives (recherche, apprentissage par renforcement, une longue session de red-team humaine). Leçon méthodologique : une défense évaluée uniquement contre des attaques qui n'ont pas été conçues contre elle n'a pas été montrée robuste.
- **Loi de puissance** : pour des courbes comme l'erreur en fonction du nombre d'exemples en contexte, une droite sur des axes log-log indique $y\propto n^{-\alpha}$ ; la pente estime $\alpha$.

## Exemple détaillé

Tout vient des exercices.

- 30 succès sur 100 essais : $\hat p=0.30$, intervalle de Wilson à 95 % $[0.219,\ 0.396]$. Avec 3 sur 10, l'intervalle est $[0.108,\ 0.603]$ - estimation ponctuelle similaire, bien moins d'information. Avec 300 sur 1000 : $[0.272,\ 0.329]$.
- Comparer des défenses à 30/100 contre 15/100 donne $z=2.54$ (environ une $p$-valeur bilatérale de 1 %), donc la différence est peu probablement due au seul hasard - en supposant les mêmes comportements, le même juge et le même budget.
- Une population simulée de 200 comportements dont environ la moitié sont brisables donne une ASR de $0.04,\ 0.14,\ 0.25,\ 0.44$ pour des budgets de $1,5,10,20$. Le budget doit être rapporté.
- Cinq variantes d'attaque aux taux de réussite individuels $(0.01,0.10,0.00,0.07,0.06)$ sur 100 comportements : la meilleure attaque seule rapporte 10 %, mais l'union est de 24 %. Le chiffre statique sous-estime l'adaptatif de plus d'un facteur deux.
- Mise à l'échelle de type many-shot : une courbe d'erreur $0.9\,n^{-0.5}$ pour $n=1,\dots,256$ donne une pente de $-0.5$ sur des axes log-log ; l'ajustement retrouve exactement $\alpha=0.5$.
- Flux d'information : lire une page non fiable et envoyer un résumé à l'extérieur donne *confirmer* ; la combiner avec une boîte de réception privée et envoyer à l'extérieur donne *bloquer* ; envoyer des données privées qu'aucun contenu non fiable n'a influencées donne *autoriser* (la propre demande de l'utilisateur).

## Implémentation

```python
import math
import numpy as np

def wilson_interval(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def union_asr(M):           # M: (behaviours, attacks) boolean matrix
    return float(np.asarray(M).any(axis=1).mean())
```

Pratique en production : utilise un harnais maintenu (les scanners et benchmarks ci-dessus), fixe le juge, publie l'ensemble de comportements et le budget, et conserve les résultats bruts pour pouvoir recalculer les intervalles.

## Idées reçues

- **« Le filtre arrête l'injection. »** Un filtre sur l'entrée utilisateur ne voit pas le contenu qui arrive par la récupération ou les outils.
- **« Zéro échec observé signifie robuste. »** Cela signifie que la borne supérieure de confiance est d'environ $3/n$ (la « règle de trois » ; Wilson donne 7,1 % à $n=50$).
- **« Une défense qui bat les attaques publiées est robuste. »** Seule l'évaluation adaptative, où l'attaquant connaît la défense, compte comme preuve.
- **« De meilleures capacités rendent le modèle plus sûr face aux attaques. »** Lire des encodages inhabituels ou de longs contextes est précisément ce que certaines attaques exploitent.
- **« Le red-teaming est ponctuel. »** Les sondes vieillissent, les modèles changent et les attaquants s'adaptent ; c'est une suite de régression plus des campagnes périodiques.

## Points clés à retenir

- Les classes d'attaque (raffinement itératif, recherche de tokens, contexte long, encodage, injection indirecte) diffèrent par l'accès qu'elles exigent ; la bonne défense diffère aussi.
- L'injection indirecte est un problème d'architecture : endigue ce que le contenu non fiable peut déclencher (étiquettes, confirmation, moindre privilège), car le filtrage ne peut pas être complet.
- Rapporte l'ASR avec son budget, son juge, son ensemble de comportements et un intervalle de Wilson ; utilise l'union sur les attaques pour l'évaluation adaptative.
- L'absence de succès observé est une preuve bornée ; la taille d'échantillon détermine quel petit taux tu peux exclure.
- Superpose les défenses (classifieurs, scanners, campagnes, limites architecturales) ; aucune n'est suffisante seule.

## Explique-le avec tes mots

1. Pourquoi un filtre sur le message de l'utilisateur échoue-t-il contre l'injection indirecte, et que fait différemment le contrôle de flux d'information ?
2. Un article rapporte 0 % d'ASR sur 40 comportements contre sa propre défense. Que demanderais-tu avant de croire que la défense est robuste ?

## Pour aller plus loin

- Articles nommés dans les leçons sources : PAIR (arXiv:2310.08419), GCG (arXiv:2307.15043), JailbreakBench (arXiv:2404.01318), ArtPrompt (arXiv:2402.11753), l'article sur le jailbreaking many-shot (NeurIPS 2024) et l'étude d'attaques adaptatives sur les défenses contre l'injection (octobre 2025). L'OWASP LLM Top 10 est la référence des praticiens pour les risques de la couche applicative. Seuls les résumés du dépôt ont été lus.

## Critères de maîtrise

- Nommer six classes d'attaque et l'accès que chacune exige, sans donner de détail opérationnel.
- Expliquer pourquoi l'injection indirecte contourne les filtres d'entrée utilisateur et ce que signifie l'endiguement.
- Calculer un intervalle de Wilson, une ASR budgétée et une ASR en union, et dire pourquoi chacune compte.
- Appliquer le contrôle de risque à trois branches à une conception d'agent et proposer quelle branche retirer.
