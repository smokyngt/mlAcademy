---
id: llm.06
title: "Modèles pré-entraînés en pratique : hubs, tokeniseurs, pipelines et fiches de modèle"
summary: "L'écosystème des modèles pré-entraînés vu par un ingénieur en activité - hubs de modèles, classes automatiques, pipelines tokeniseur plus modèle plus décodeur, usage zero-shot et few-shot, flux de travail de fine-tuning, fiches de modèle et usage responsable - reconstruit hors ligne avec des simulacres."
minutes: 60
teaches: [tal.hub-ecosystem]
requires: [llm.tokenization, trf.architectures, nn.layer]
sources:
  - "cours-tal-04.pdf (RCP217, S. Rosmorduc, CNAM, French slides, PDF p.21 use of pre-trained BERT, p.28-29 GPT and ChatGPT, p.30-38 HuggingFace: libraries, AutoClasses, pipelines, zero-shot, question answering, classification by fine-tuning, model choice, few-shot, large language models and self-hosting, p.39-41 bibliography). READ."
  - "content/authored/llms/01-tokenization/lesson.md (what is already taught about tokenisation, to avoid duplication). READ."
---

## Objectifs d'apprentissage

- Expliquer ce qu'un hub de modèles fournit et ce que sont les trois objets (tokeniseur, modèle, configuration).
- Décrire un pipeline comme tokeniseur, puis modèle, puis décodeur, avec la forme de chaque valeur intermédiaire.
- Distinguer l'usage direct, le zero-shot, le few-shot et le fine-tuning, et choisir entre eux.
- Lister ce qu'une fiche de modèle doit contenir et pourquoi chaque champ compte.
- Construire un pipeline hors ligne avec des simulacres et un vérificateur de fiche de modèle.

## Où cela s'inscrit

À présent, tu sais entraîner un modèle de langage à partir de zéro ([[llm.03]]) et tu connais les architectures ([[trf.04]], [[trf.05]]). En pratique, presque personne n'entraîne depuis zéro : on télécharge un modèle que quelqu'un d'autre a payé pour pré-entraîner. Le support du CNAM ouvre sa section à ce sujet (cours-tal-04, p.30) en présentant la bibliothèque transformers de HuggingFace comme l'une des bibliothèques qui fournissent des implémentations de tous les réseaux modernes avec leurs versions pré-entraînées, utilisables depuis TensorFlow ou PyTorch, faciles à exécuter telles quelles et fine-tunables. Cette leçon est le modèle mental de cet écosystème, pour que tu puisses lire n'importe quel extrait « from_pretrained » et savoir ce qui se passe. Le fine-tuning lui-même est [[llm.04]] ; la récupération d'information autour de tels modèles est [[sys.03]].

## L'idée en mots simples

Un hub de modèles est une bibliothèque avec un catalogue. Pour chaque livre (modèle), elle garde trois choses sur l'étagère :

1. la **fiche de recette** (configuration) : combien de couches, quelle largeur, quelle famille ;
2. le **dictionnaire** (tokeniseur) : comment le texte est découpé en morceaux et numéroté, qui doit être exactement celui utilisé lors de l'apprentissage du modèle ;
3. le **cerveau** (poids).

Emprunter un modèle, c'est récupérer les trois et les garder ensemble. Si tu échanges le dictionnaire, les numéros désignent d'autres mots et la sortie est absurde sans aucun message d'erreur.

Un **pipeline** est un distributeur automatique autour de ces pièces : tu y mets du texte, un tokeniseur le transforme en nombres, le modèle calcule des scores, et un décodeur transforme les scores en quelque chose de lisible (« POSITIVE, 0.95 »). Une **fiche de modèle** (model card) est la notice dans la boîte : à quoi il sert, sur quoi il a été entraîné, quelles sont ses performances, et pour quoi il ne faut pas l'utiliser.

## Les éléments, tels que le support les présente

**Classes automatiques (p.31).** L'idée : le nom ou le chemin du modèle dit quelle classe construire. `AutoModel.from_pretrained('bert-base-cased')` crée un modèle BERT ; `AutoConfig` charge une configuration et te laisse remplacer des attributs (la diapositive active les sorties d'attention) ; `AutoTokenizer` charge le tokeniseur correspondant, par exemple pour un BERT allemand. Mécaniquement, le hub stocke un fichier de configuration avec un type de modèle ; un registre associe ce type à une classe Python. Tu reconstruiras ce registre dans l'exercice.

**Pipelines (p.32).** Des modèles prêts à l'emploi pour des tâches : extraction de caractéristiques, classification de texte, classification de tokens, question-réponse, question-réponse sur tableaux, fill-mask, résumé, traduction, génération texte-vers-texte, génération de texte, classification zero-shot, conversationnel. L'exemple de la diapositive est un pipeline de sentiment dont le résultat est une étiquette et un score, par exemple `POSITIVE` avec 0.9998.

**Choisir un modèle (p.36).** La bibliothèque ne trouve pas toujours seule un modèle adapté ; tu devras peut-être en nommer un, par exemple un modèle de traduction français-anglais publié sur le hub. Le support indique le catalogue de modèles du hub comme ressource pratique.

**Zero-shot (p.33).** Classer un document face à une liste d'étiquettes choisie librement sans fine-tuning sur elles : l'exemple du support donne les étiquettes « social event, computing, science, fun, sport, theater, art » et la phrase sur un programme Java qui n'a pas compilé. (Son fonctionnement interne ne figure pas sur la diapositive ; une implémentation courante, connaissance générale, pose chaque étiquette comme une hypothèse à un modèle d'inférence en langage naturel.)

**Question-réponse (p.34).** Extrait la réponse d'un texte donné : « où étais-je avant Londres ? » avec le texte « I went to Nice, Paris and then London » donne une partie citée du document. Le support insiste sur le fait qu'il n'invente pas de texte.

**Classification par fine-tuning (p.35).** Prendre un modèle adapté à la langue (BERT pour l'anglais), le fine-tuner ; il existe une séance pratique pour cela. Voir [[trf.05]] pour la tête et les formes.

**Few-shot (p.37).** Montrer quelques exemples à un modèle de langage dans le prompt et il extrapole ; le support le démontre avec des exemples de traduction français-anglais et français-espagnol.

**Grands modèles de langage (p.38).** ChatGPT et ses pairs (Claude, Gemini, Mistral et autres) offrent de nouvelles capacités zero-shot et few-shot, de nouveaux embeddings, des API, la génération augmentée par récupération, généralement via un gros serveur distant ; l'auto-hébergement est possible, notamment avec Ollama. Les techniques de prompting sont dans [[app.01]] et la récupération dans [[sys.03]].

## Le pipeline comme trois fonctions

Soit un batch de $B$ chaînes tokenisées, complétées jusqu'à la longueur $T$ :

$$
\text{texts}\xrightarrow{\text{tokenizer}}(\text{ids},\text{mask})\in\mathbb{Z}^{B\times T}\times\{0,1\}^{B\times T}\xrightarrow{\text{model}}\text{logits}\in\mathbb{R}^{B\times K}\xrightarrow{\text{decoder}}B\text{ results}.
$$

Deux détails causent couramment des bugs. **Le padding** : les textes plus courts sont complétés avec un id de padding, et le *masque d'attention* indique au modèle d'ignorer ces positions ; oublier le masque change les prédictions selon ce qui se trouve d'autre dans le batch. **Le décodage** : pour la classification, softmax puis argmax donne l'étiquette et le score ; pour la génération, la boucle du décodeur est de l'échantillonnage ou de la recherche en faisceau ([[llm.02]]) puis la détokenisation.

### Exemple détaillé (reproduit par l'exercice)

Vocabulaire : `[PAD]=0, [UNK]=1, good=2, bad=3, movie=4, not=5`. Un faux modèle de sentiment en sac de mots somme des vecteurs de scores par token (négatif, positif) : good $=(-1,2)$, bad $=(2,-1)$, not $=(0.5,-0.5)$, movie $=(0,0)$.

- « good movie » : logits $(-1,2)$, probabilité softmax du positif $1/(1+e^{-3})=0.953$.
- « not bad » : logits $(2.5,-1.5)$, probabilité de l'étiquette gagnante (NEGATIVE) $1/(1+e^{-4})=0.982$. Le faux modèle est naïf : il traite « not bad » comme négatif, ce qui rappelle que la qualité des poids compte et que la plomberie n'en dit rien.
- Le batch `["good movie","not good at all","bad"]` s'encode en ids de forme $(3,4)$ ; la ligne 3 est `[3,0,0,0]` avec le masque `[1,0,0,0]`.

## Façons d'utiliser un modèle pré-entraîné

| Mode | Les poids changent ? | Besoin d'étiquettes ? | Quand |
|---|---|---|---|
| Direct (par défaut du pipeline) | non | non | le hub a déjà un modèle pour ta tâche et ta langue |
| Zero-shot | non | non | les étiquettes changent souvent ou tu n'as pas de données |
| Few-shot (prompting) | non | une poignée d'exemples dans le prompt | LLM disponible, prototype rapide |
| Fine-tuning | oui | oui (centaines à milliers) | tâche stable, besoin de précision, de latence ou de maîtrise des coûts |

Le support le formule ainsi : les réseaux pré-entraînés peuvent être utilisés « directement sur l'apprentissage déjà réalisé » ou après fine-tuning (p.21). Quel que soit ton choix, évalue sur tes propres données mises de côté ([[sys.04]]) : le chiffre phare d'un hub est mesuré sur les données de quelqu'un d'autre.

## Fiches de modèle et usage responsable

Le support liste un catalogue mais ne dit rien de la documentation ; cette section relève de la connaissance générale et doit se lire comme une pratique d'ingénierie plutôt que comme un résumé des diapositives. Une fiche de modèle consigne typiquement :

- la **licence** : ce que tu as le droit de faire (usage commercial, redistribution) ;
- la ou les **langues** : un utilisateur francophone doit vérifier que le modèle a été entraîné sur du français ;
- l'**usage prévu** et les **limites** (usages hors périmètre, modes d'échec connus) ;
- les **données d'entraînement** : provenance, et le support note (p.28) que les données de GPT-3 ne sont pas librement accessibles, donc tu ne peux pas les auditer ;
- les **métriques** : avec les données d'évaluation nommées ; les valeurs d'un classifieur sont généralement dans $[0,1]$ ;
- le **modèle de base**, s'il s'agit d'un fine-tuning d'un autre.

Liste de contrôle pour un modèle téléchargé : lire la licence ; vérifier l'adéquation de la langue et du domaine ; l'exécuter sur 50 de tes propres exemples ; rechercher les biais connus ([[saf.05]]) ; noter l'identifiant exact du modèle et sa révision pour que les résultats soient reproductibles ; et se méfier du fichier de poids lui-même (certains formats peuvent exécuter du code au chargement ; préférer les formats conçus pour de simples tenseurs, connaissance générale). Pour les API de LLM hébergées, ajouter une revue de confidentialité des données, puisque ton texte quitte ta machine ; l'auto-hébergement est l'alternative nommée par le support (p.38).

## Implémentation

```python
import numpy as np
class Pipeline:
    def __init__(self, tokenizer, model, decoder):
        self.tokenizer, self.model, self.decoder = tokenizer, model, decoder
    def __call__(self, inputs):
        single = isinstance(inputs, str)
        ids, mask = self.tokenizer.batch_encode([inputs] if single else list(inputs))
        out = [self.decoder(row) for row in self.model(ids, mask)]
        return out[0] if single else out
```

Dans HuggingFace, la même forme est `pipeline("sentiment-analysis")` (diapositive p.32) ; `from_pretrained` télécharge et met en cache des fichiers, ce que l'exercice remplace par de simples objets Python pour qu'il s'exécute hors ligne.

## Idées reçues courantes

- **« Le tokeniseur est interchangeable. »** Il doit correspondre au modèle ; des ids d'un autre vocabulaire désignent silencieusement d'autres mots.
- **« Le résultat d'un pipeline est la vérité. »** Le score est une probabilité softmax, pas une confiance calibrée.
- **« Zero-shot signifie aucun travail. »** Il faut toujours évaluer ; la formulation des étiquettes change les résultats.
- **« Un score élevé sur la fiche de modèle se transfère à mes données. »** Seulement si les données sont semblables.
- **« Des poids ouverts signifient des données ouvertes ou une licence ouverte. »** Ce sont des choses distinctes ; lis chacune.

## Points clés à retenir

- Un modèle pré-entraîné est un lot : configuration, tokeniseur, poids ; garde-les appariés.
- Les classes automatiques associent le type de modèle d'une configuration à une classe ; les pipelines enchaînent tokeniseur, modèle, décodeur.
- Choisis entre usage direct, zero-shot, few-shot et fine-tuning selon les données, le coût et le besoin de précision, puis évalue sur tes propres données.
- Les fiches de modèle consignent licence, langues, usage prévu, limites, données d'entraînement et métriques ; vérifie-les avant usage.

## Explique-le toi-même

1. Un collègue remplace le tokeniseur d'un classifieur fine-tuné par un autre d'apparence similaire et la précision tombe au niveau du hasard sans erreur. Explique pourquoi.
2. Tu dois étiqueter des e-mails de clients en français selon 8 catégories ; tu as 300 exemples étiquetés et un budget de latence strict. Quel mode d'usage choisis-tu et que vérifies-tu sur la fiche de modèle ?

## Pour aller plus loin

- La documentation HuggingFace de `transformers` (pipelines, classes automatiques) et les fiches de modèle.
- Mitchell et al., « Model Cards for Model Reporting » (2019), lecture générale, non consultée pour cette leçon.
- Brown et al., « Language Models are Few-Shot Learners » (arXiv 2005.14165), cité dans la bibliographie du support.

## Critères de maîtrise

- [ ] Je sais dire ce que contient un dépôt de modèle et pourquoi le tokeniseur doit correspondre.
- [ ] Je sais écrire la chaîne tokeniseur, modèle, décodeur avec les formes, y compris le masque d'attention.
- [ ] Je sais choisir entre usage direct, zero-shot, few-shot et fine-tuning et le justifier.
- [ ] Je sais relire une fiche de modèle et lister ce qui manque.
