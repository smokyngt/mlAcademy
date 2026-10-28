---
id: app.01
title: "Fondamentaux du prompting : instructions, exemples, raisonnement et vote"
summary: "Traiter un prompt comme un programme testable qui conditionne un prédicteur du jeton suivant, et apprendre les quatre leviers (instructions, rôles, exemples, raisonnement) ainsi que le vote par auto-cohérence avec son gain binomial exact."
minutes: 70
teaches: [app.prompting, app.self-consistency]
requires: [llm.lm, llm.sampling, trf.architectures, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/11-llm-engineering/01-prompt-engineering/docs/en.md (opening sections: the vague-versus-engineered prompt example, anatomy of a prompt, system/user/prefill, role prompting; first ~5.5k characters only). READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/02-few-shot-cot/docs/en.md (opening sections: zero-shot versus few-shot, example selection, chain-of-thought motivation; first ~5.5k characters only). READ."
---

## Où cela s'inscrit

Tu as rencontré un modèle de langage comme une fonction qui associe une séquence de jetons à une distribution de probabilité sur le jeton suivant (la leçon `trf.04` a construit l'architecture, et les leçons de décodage ont montré comment y échantillonner). Un **prompt** est simplement la séquence de jetons que tu places devant cette fonction. Tout ce qu'une application fait avec un modèle hébergé, d'un bot de support à un relecteur de code, commence comme du texte dans un prompt. Cette leçon est la première du cours sur les applications : elle t'apprend à *contrôler* le modèle par son entrée, à acheter de la précision supplémentaire avec plus de calcul à l'inférence (raisonnement et vote), et à traiter les prompts comme des artefacts que l'on peut tester plutôt que comme du folklore. Les leçons suivantes ajoutent de la structure à la sortie ([[app.02]]), gèrent toute la fenêtre ([[app.03]]) et mesurent la qualité ([[app.04]]).

## L'idée en termes simples

Imagine que tu embauches un intérimaire très cultivé qui a lu la plus grande partie de la bibliothèque mais ne sait rien de ton entreprise et prend chaque phrase au pied de la lettre. Si tu dis « écris-moi un e-mail », tu obtiens un e-mail moyen, parce que le « moyen » est ce que contient la bibliothèque. Si tu précises qui est le lecteur, quelle doit être la longueur, quel ton employer et si tu montres un e-mail que tu as aimé, tu obtiens quelque chose de proche de ce que tu voulais. Un prompt fait exactement cela : il **restreint l'ensemble des continuations** que le modèle juge plausibles.

Quatre leviers font l'essentiel du travail :

1. **Instructions** : dis précisément ce que tu veux, dans quel format, avec quelles contraintes, et quoi faire en cas de doute.
2. **Rôle ou contexte** : dis au modèle en tant que qui il écrit et pour qui. Cela n'ajoute pas de connaissance ; cela déplace la partie de ce que le modèle a vu qu'il imite.
3. **Exemples** (few-shot) : montre deux à cinq paires entrée-sortie. Un exemple est une instruction compressée ; le modèle copie le motif, y compris ses bizarreries accidentelles.
4. **Place pour réfléchir** (chaîne de pensée) : laisse le modèle écrire des étapes intermédiaires avant la réponse. Le modèle n'a pas d'espace de brouillon privé entre les jetons ; tout ce qu'il écrit devient entrée pour le jeton suivant. Les étapes écrites sont du calcul effectué.

Un modèle échantillonné avec un peu d'aléa ressemble aussi à un élève qui se trompe parfois. Si tu poses cinq fois la même question difficile et que tu retiens la réponse sur laquelle la plupart des élèves s'accordent, tu as généralement plus souvent raison que n'importe quel élève seul. C'est l'**auto-cohérence** (self-consistency).

## Anatomie d'un prompt

Un appel d'API de chat comporte jusqu'à trois parties :

- **Message système** : règles et identité persistantes, écrites une fois, placées en premier.
- **Message utilisateur** : la tâche ou la question de ce tour.
- **Préremplissage de l'assistant** (proposé par certains fournisseurs) : les premiers mots de la réponse, écrits par toi, que le modèle doit poursuivre. Commencer la réponse par une accolade ouvrante pousse le modèle vers du JSON sans bavardage.

Règles pratiques qui découlent de « le modèle lit littéralement » :

- Place le matériel stable (règles, définitions) avant le matériel variable (l'entrée de l'utilisateur). Cela compte à nouveau dans [[app.03]] pour le cache.
- Énonce le format de sortie et que faire en cas d'information manquante (« réponds `unknown` si le texte ne le dit pas »).
- Préfère des énoncés positifs et concrets (« réponds en 3 phrases au plus ») aux énoncés vagues (« sois bref »).
- Sépare les instructions des données par des délimiteurs clairs, pour que le texte à l'intérieur des données ne soit pas pris pour une instruction (cela devient une question de sécurité dans [[app.05]]).

**Choix des exemples few-shot.** Trois à cinq exemples sont typiques ; moins donnent trop peu de signal, plus coûtent du contexte pour des rendements décroissants. Choisis des exemples similaires à l'entrée, couvre chaque étiquette, et fais correspondre la difficulté. Les exemples l'emportent sur les tâches sensibles au format ; pour de simples questions factuelles ou du travail créatif, ils peuvent contraindre plus qu'ils n'aident. Une source que nous avons lue cite des pourcentages d'exactitude précis pour ces effets ; considère tout chiffre de ce type comme dépendant du modèle et de la tâche, et mesure sur tes propres données.

## La chaîne de pensée et son coût

Le prompting par chaîne de pensée (CoT) demande les étapes de raisonnement d'abord et la réponse en dernier, soit par une instruction (« réfléchis étape par étape »), soit en montrant des exemples résolus. Mécaniquement, un transformer dépense une quantité fixe de calcul par jeton généré, donc un problème exigeant de nombreuses étapes dépendantes ne peut pas toujours être résolu « en un jeton », mais peut l'être sur plusieurs. Le prix est davantage de jetons de sortie (coût et latence, voir [[app.06]]) et un nouveau mode de défaillance : une chaîne fluide mais fausse.

Deux conséquences pratiques :

- **Place le raisonnement avant la réponse** dans tout format structuré. Un objet JSON avec `"answer"` avant `"reasoning"` s'engage d'abord sur la réponse, et le « raisonnement » qui suit n'est qu'une justification.
- **Analyse la réponse finale de façon robuste.** Un raisonnement libre se termine de nombreuses manières ; il te faut une règle comme « l'entier après la dernière occurrence du mot answer ».

## Auto-cohérence : voter sur des raisonnements échantillonnés

Échantillonne $n$ chaînes de pensée indépendantes à une température supérieure à zéro, extrais chaque réponse finale, et renvoie la **plus fréquente** (vote à la majorité). Des chaînes différentes font des erreurs différentes, mais les chaînes correctes s'accordent.

### Dérivation : exactitude d'un vote à la majorité

Suppose que chaque échantillon est correct avec une probabilité $p$, indépendamment, et que les échantillons faux donnent des réponses fausses *différentes* (de sorte que les mauvaises réponses ne se liguent jamais contre la bonne). Alors le vote est correct exactement quand plus de la moitié des $n$ échantillons ($n$ impair) sont corrects. Le nombre d'échantillons corrects $K$ est binomial :

$$
P(K=k)=\binom{n}{k}p^{k}(1-p)^{n-k}.
$$

La majorité a donc raison avec la probabilité

$$
A(p,n)=\sum_{k=\lceil n/2\rceil}^{n}\binom{n}{k}p^{k}(1-p)^{n-k}.
$$

Lecture : $A(p,1)=p$. Si $p>1/2$, $A(p,n)$ augmente avec $n$ et tend vers 1 ; si $p=1/2$ elle reste à $1/2$ ; si $p<1/2$, voter aggrave les choses et tend vers 0. Le vote amplifie le camp qui mène déjà.

## Exemple détaillé

Avec $p=0.6$ et $n=5$ : $A=\binom53 0.6^3 0.4^2+\binom54 0.6^4 0.4+0.6^5=0.3456+0.2592+0.07776=0.68256$. Plus d'échantillons : $n=3$ donne $0.648$, $n=11$ donne $0.7535$, $n=21$ donne $0.8256$. Avec $p=0.4$ et $n=5$, le vote n'a raison que dans $0.3174$ des cas.

L'exercice stocke 8 questions avec 5 chaînes préenregistrées chacune. En analysant la réponse finale de chaque texte, l'exactitude par échantillon est $25/40=0.625$ (deux des 40 textes n'ont pas de réponse analysable), et le vote à la majorité a raison sur 7 questions sur 8, $0.875$. La formule d'indépendance à $p=0.625,n=5$ prédit $0.7248$ ; l'observé $0.875$ est plus élevé parce qu'il n'y a que 8 questions (grand bruit d'échantillonnage) et parce que les questions diffèrent en difficulté. La seule question que le vote manque (la troisième) est une question où la mauvaise réponse, 9, a été produite par trois chaînes : des erreurs qui **s'accordent** battent le vote, ce qui est la vraie limite de la méthode. Les vrais échantillons d'un même modèle partagent des angles morts, donc le gain est plus faible que ne le suggère la formule à échantillons indépendants.

## Tree-of-thought, brièvement

L'auto-cohérence échantillonne des chaînes entières indépendamment. Le **tree-of-thought** construit au contraire la solution pas à pas comme une recherche : à chaque nœud, le modèle *propose* plusieurs pensées suivantes, le modèle (ou une règle) *évalue* chaque solution partielle, les branches faibles sont élaguées, et la recherche continue depuis les meilleures (en largeur d'abord ou meilleur d'abord). Il aide quand les premiers choix comptent et peuvent être jugés tôt (casse-têtes, planification). Le coût est à peu près (facteur de branchement) × (profondeur) × (appels de proposition plus d'évaluation), donc ne l'utilise que là où une chaîne unique plus le vote échoue.

## Implémentation

L'exercice construit le harnais sans aucun appel de modèle :

```python
import re
from collections import Counter

def extract_answer(text):
    m = re.findall(r"answer\s*(?:is|:)?\s*(-?\d+)", text, flags=re.I)
    return int(m[-1]) if m else None

def majority_vote(answers):
    c = Counter(a for a in answers if a is not None)
    if not c:
        return None
    best = max(c.values())
    return next(a for a in answers if a is not None and c[a] == best)
```

Les bibliothèques de production encapsulent la même idée : tu appelles l'API avec `n` échantillons (ou `n` appels), tu analyses chacun, et tu votes. Pour des réponses libres, tu ne peux pas voter sur des chaînes, donc tu regroupes d'abord les réponses équivalentes (normalise les nombres, ou demande à un modèle juge si deux réponses concordent).

## Les prompts sont testables

Un prompt est du code dont tu ne peux pas lire le comportement sur le texte. Garde un petit ensemble d'entrées avec les sorties attendues (ou une règle de notation), fais passer chaque modification de prompt dessus, et regarde les échecs, pas seulement la moyenne. Comparer deux prompts équitablement demande des statistiques parce que les sorties sont aléatoires ; [[app.04]] l'implémente. Rappelle-toi aussi que les résultats dépendent de la version du modèle : un prompt réglé pour un modèle peut se dégrader sur le suivant.

## Idées reçues courantes

- « Des formules magiques rendent le modèle plus intelligent. » Un rôle ou « réfléchis étape par étape » change la continuation probable ; cela n'ajoute aucune connaissance. Les gains varient selon le modèle et peuvent être proches de zéro sur les modèles de raisonnement puissants.
- « Plus d'exemples, c'est toujours mieux. » Le contexte est fini et les exemples peuvent biaiser vers leurs bizarreries. Teste 0, 3 et 5.
- « L'auto-cohérence est une exactitude gratuite. » Elle multiplie le coût de sortie par $n$ et n'aide que si l'exactitude par échantillon dépasse 50 pour cent et que les erreurs ne sont pas systématiques.
- « Le modèle a suivi mon instruction une fois, donc le prompt fonctionne. » Un succès est une anecdote ; il te faut un jeu de test.
- « Le raisonnement écrit est la vraie raison de la réponse. » C'est du texte généré qui conditionne les jetons suivants ; il peut être faux ou infidèle même quand la réponse est juste.

## Points clés à retenir

- Un prompt restreint la distribution de sortie du modèle ; instructions, rôle, exemples et place pour raisonner sont les quatre leviers.
- L'ordre compte : instructions stables d'abord, données délimitées ensuite, raisonnement avant la réponse.
- Le vote à la majorité sur $n$ échantillons a une exactitude $A(p,n)$, qui croît pour $p>1/2$ et décroît pour $p<1/2$.
- Le vote échoue sur les erreurs concordantes et coûte $n$ fois la génération.
- Le tree-of-thought est une recherche sur des solutions partielles ; n'y recours qu'en dernier.
- Traite les prompts comme des artefacts avec des tests.

## Explique-le toi-même

1. Pourquoi écrire des étapes intermédiaires aide-t-il un modèle qui n'a pas de mémoire cachée entre les jetons, et quand cela nuit-il ?
2. Un collègue dit « nous échantillonnons 9 réponses et nous votons, donc l'exactitude doit être proche de 100 pour cent. » Quelles hypothèses cette phrase cache-t-elle, et comment les vérifierais-tu avec des données ?

## Pour aller plus loin

- Wei et al. (2022) sur le prompting par chaîne de pensée et Wang et al. (2022) sur l'auto-cohérence (cités par les leçons que nous avons lues ; nous n'avons pas ouvert les articles).
- Leçons 01 et 02 de la phase 11 de la bibliothèque du dépôt pour plus de code.

## Critères de maîtrise

- Tu sais réécrire un prompt vague avec des instructions, un format et des délimiteurs, et expliquer chaque changement.
- Tu sais dériver $A(p,n)$ et le calculer pour des $p,n$ donnés.
- Tu sais énoncer deux conditions dans lesquelles le vote n'aide pas.
- Tu sais écrire un analyseur robuste de réponse finale et un vote à la majorité avec une règle de départage définie.
