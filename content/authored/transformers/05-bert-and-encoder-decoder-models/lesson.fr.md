---
id: trf.05
title: "BERT, BART, T5 et GPT : objectifs de pré-entraînement et quand utiliser chacun"
summary: "La modélisation de langage masquée avec la règle 80/10/10, les têtes de fine-tuning de BERT (classification, étiquetage, QA par segment), les encodeurs-décodeurs débruiteurs (BART, corruption de segments de T5), en quoi GPT diffère, et un guide de décision."
minutes: 80
teaches: [tal.mlm, tal.bert-finetune, tal.denoising-seq2seq]
requires: [trf.architectures, trf.block, nn.layer, info.entropy]
sources:
  - "cours-tal-04.pdf (RCP217, S. Rosmorduc, CNAM, French slides, PDF p.1-41, all pages read): motivation for transfer learning (p.2), recap of seq2seq and attention (p.3-5), Transformer (p.6-19), BERT objectives, use and head layouts (p.20-26), the other models XLNet/BART/T5 (p.27), GPT and ChatGPT (p.28-29), HuggingFace and LLM remarks (p.30-38), bibliography (p.39-41). READ."
  - "content/authored/transformers/04-transformer-block-and-architectures/lesson.md (what is already covered about the three families, to avoid duplication). READ."
---

## Objectifs d'apprentissage

- Expliquer pourquoi les transformers ont fait fonctionner l'apprentissage par transfert pour le texte, et sur quoi BERT est pré-entraîné.
- Appliquer la règle de masquage 80/10/10 et calculer la perte de modélisation de langage masquée, moyennée sur les seules positions sélectionnées.
- Donner les formes des sorties de BERT et comment une tête de classification, une tête d'étiquetage et une tête de QA par segment les lisent.
- Décrire les encodeurs-décodeurs débruiteurs (BART, corruption de segments de T5) et en quoi l'objectif causal de GPT diffère.
- Choisir entre encodeur seul, décodeur seul et encodeur-décodeur pour une tâche donnée.

## Où cela s'inscrit

[[trf.04]] a construit le bloc transformer et a listé les trois familles en un paragraphe. Cette leçon est le chapitre « comment sont-elles réellement entraînées et utilisées ». Elle suit le support du CNAM sur BERT, BART et les Transformers (cours-tal-04), dont le fil conducteur est : les images ont eu des réseaux pré-entraînés réutilisables (l'ère d'AlexNet) et les mots des embeddings réutilisables, mais les réseaux récurrents se transféraient mal vers de nouvelles tâches ; à partir du Transformer, de grands réseaux pré-entraînés sur d'énormes corpus se transfèrent efficacement. Les pages 3 à 19 du support récapitulent l'encodeur-décodeur avec attention ([[seq.05]]) et le Transformer ([[trf.01]] à [[trf.03]]), nous ne les répétons donc pas. Leçons liées : [[llm.06]] montre comment charger ces modèles en pratique, et [[nlp.02]] et [[nlp.03]] sont les références classiques qu'ils remplacent.

## L'idée en mots simples

Imagine que tu apprennes à lire à un enfant avec des jeux joués sur des millions de livres, sans aucune étiquette humaine.

- **Jeu 1, remplir les blancs (BERT).** On cache quelques mots et on demande à l'enfant de les deviner à partir de tout ce qui entoure le blanc, à gauche et à droite. Pour gagner, l'enfant doit apprendre la grammaire, des faits et les sens des mots. Ensuite, tu lui donnes un court cours sur une tâche précise (spam ou non spam), et il apprend vite parce que les compétences de lecture sont déjà là.
- **Jeu 2, finir la phrase (GPT).** On ne montre que le début et on demande le mot suivant. L'enfant ne peut jamais jeter un œil en avant, ce qui en fait un écrivain naturel.
- **Jeu 3, réparer la page abîmée (BART, T5).** On arrache des morceaux d'une page et on demande à l'enfant de réécrire l'original. Un lecteur (encodeur) regarde la page abîmée et un rédacteur (décodeur) produit la réparation.

Les trois jeux utilisent du texte non étiqueté, donc l'offre d'exercices est pratiquement illimitée. Ce qui change d'une famille à l'autre, c'est qui peut regarder quoi, et ce que le modèle doit écrire.

## Modélisation de langage masquée

BERT est, d'après le support (p.20), la moitié encodeur d'un Transformer pré-entraîné sur deux tâches à la fois. La première est la **modélisation de langage masquée (MLM)** : remplacer environ 15 % des mots par un symbole de masque et demander au réseau de les prédire en sortie.

Soit l'entrée constituée des ids de tokens $x_1,\dots,x_T$ et soit $S$ l'ensemble des positions sélectionnées. Le modèle lit la séquence corrompue et produit des logits $Z\in\mathbb{R}^{T\times V}$ ($V$ est la taille du vocabulaire). La perte est

$$
\mathcal{L}_{\text{MLM}}=-\frac{1}{|S|}\sum_{i\in S}\log\operatorname{softmax}(Z_i)_{x_i}.
$$

Seules les positions sélectionnées contribuent ; les $T-|S|$ autres positions n'ont pas d'étiquette. Une erreur courante consiste à moyenner sur les $T$ positions, ce qui divise par le mauvais nombre.

### La règle 80/10/10

Le support dit que les mots sont remplacés par un token de masque. La recette originale de BERT (connaissance générale, pas dans les diapositives) est un peu plus subtile. Parmi les positions sélectionnées :

- 80 % deviennent le token de masque,
- 10 % deviennent un token aléatoire,
- 10 % restent inchangées.

Pourquoi ? Le token de masque n'apparaît jamais quand le modèle est utilisé sur du vrai texte. Si chaque token sélectionné était masqué, le réseau pourrait apprendre « ne travaille que là où je vois le masque », et ses représentations des tokens ordinaires seraient négligées. Les remplacements aléatoires et les tokens inchangés le forcent à construire une bonne représentation de chaque position, car n'importe quel token peut être faux ou peut être une cible. La perte est toujours calculée à toutes les positions sélectionnées, y compris les 10 % laissées inchangées.

### La seconde tâche

Le support (p.20) décrit une seconde tâche simultanée : étant donné une paire de phrases, dire si la seconde peut suivre la première, les exemples étant soit deux phrases consécutives du corpus, soit deux phrases indépendantes. On l'appelle généralement prédiction de la phrase suivante. Les diapositives ne discutent pas de son utilité ; des travaux ultérieurs auraient trouvé qu'elle apporte peu et plusieurs successeurs l'ont abandonnée (connaissance générale, à considérer comme incertain). Le support liste d'autres modèles aux objectifs différents (p.27) : XLNet étend la modélisation de langage à toutes les permutations de l'ordre des mots et, comme BERT, est un encodeur ; BART couple un encodeur à un décodeur ; T5 est un réseau séquence-à-séquence entraîné sur un très grand nombre de tâches ; Transformer-XL gère de très longs textes avec des positions relatives.

## Utiliser BERT : formes des sorties

BERT envoie des ids de tokens $(B,T)$ vers des états cachés $H\in\mathbb{R}^{B\times T\times d}$. Deux tokens spéciaux comptent (connaissance générale) : un token de classification `[CLS]` placé en premier dans l'entrée, et un séparateur `[SEP]` entre les phrases. Le support (p.21) dit qu'un réseau pré-entraîné peut être utilisé directement ou après fine-tuning, et les pages 22 à 24 montrent trois dispositions de têtes tirées de l'article BERT :

| Tâche | La tête lit | Paramètres de la tête | Sortie |
|---|---|---|---|
| Classification (p.23) | $H_{[:,0,:]}$, le vecteur `[CLS]`, de forme $(B,d)$ | $W\in\mathbb{R}^{d\times K}$ | logits $(B,K)$ |
| Reconnaissance d'entités nommées (p.22) | chaque vecteur de token, $(B,T,d)$ | $W\in\mathbb{R}^{d\times K}$, partagé entre les positions | logits $(B,T,K)$, une étiquette par token |
| Question-réponse (p.24) | chaque vecteur de token du passage, $(B,T,d)$ | un vecteur de début $s\in\mathbb{R}^d$ et un vecteur de fin $e\in\mathbb{R}^d$ | scores de début $(B,T)$, scores de fin $(B,T)$ |

Pour la QA, la réponse est un segment du passage : on applique un softmax aux scores de début et aux scores de fin sur les positions et on choisit $(i,j)$ avec $i\le j$ maximisant la somme. Cela concorde avec la remarque HuggingFace (p.34) selon laquelle la QA extractive n'invente aucun texte et cite une partie du document. Une tête ajoute très peu de paramètres : avec $d=768$ (une taille de base courante) et $K=2$ classes, une tête de classification a $768\cdot 2+2=1538$ nombres, minuscule à côté de l'encodeur.

**Le fine-tuning** consiste à : charger les poids pré-entraînés, ajouter la tête, entraîner tout (ou seulement le haut) sur un petit jeu étiqueté avec un faible taux d'apprentissage, de sorte que l'encodeur soit infléchi plutôt qu'écrasé. Le support renvoie à une séance pratique pour cela (p.35).

Le support note aussi (p.21) que BERT, étant un encodeur, n'a pas donné de résultats très concluants en traduction automatique : il n'a aucun mécanisme pour écrire du texte en sortie.

## Ce que font les têtes d'attention

Les pages 25 et 26 montrent des visualisations de Clark et al. (2019) : certaines têtes modélisent des relations simples, d'autres des relations syntaxiques. Considère cela comme un indice, pas comme une explication. Le support lui-même (p.15) dit que la visualisation de l'attention ne donne que peu d'information sur ce que fait le modèle.

## Encodeurs-décodeurs débruiteurs : BART et T5

Le support ne nomme BART et T5 qu'en une ligne chacun (p.27). Les détails ci-dessous relèvent de la connaissance générale et sont signalés comme tels.

Un encodeur-décodeur (voir [[seq.05]], [[trf.04]]) lit une source avec un encodeur bidirectionnel et écrit une cible avec un décodeur causal qui fait de l'attention croisée sur l'encodeur. Pour le pré-entraîner sans étiquettes, on abîme l'entrée et on utilise le texte propre comme cible :

- **BART** corrompt le texte de plusieurs façons (masquage, suppression, mélange de phrases, remplissage de segments) et le décodeur reconstruit l'original complet.
- **T5** utilise la **corruption de segments** : on choisit des segments contigus, on remplace chacun par un seul token *sentinelle* dans l'entrée, et la cible est composée des sentinelles, chacune suivie du segment retiré. Tout est ramené à du texte-vers-texte, de sorte que la traduction, le résumé et la classification deviennent tous « chaîne d'entrée vers chaîne de sortie » avec un préfixe de tâche.

Exemple détaillé (l'exercice le reproduit). Tokens $1..10$, segments (début 2, longueur 2) et (début 6, longueur 3), ids de sentinelles commençant à 100 :

- entrée de l'encodeur : `[1, 2, 100, 5, 6, 101, 10]`
- cible du décodeur : `[100, 3, 4, 101, 7, 8, 9, 102]`

L'entrée passe de 10 à 7 tokens et la cible est courte, ce qui est moins coûteux que de réécrire tout le texte. La sentinelle de fermeture `102` indique au décodeur où s'arrêter.

## GPT : la famille causale

Le support (p.28) présente GPT, GPT-2 et GPT-3 comme des modèles entraînés essentiellement comme modèles de langage sur des données et des tailles toujours plus grandes (GPT-3 : 175 milliards de paramètres, 300 milliards de tokens d'entraînement), efficaces pour produire du texte assez cohérent. Il les décrit comme « basés sur un encodeur en entrée » ; les descriptions modernes, et [[trf.04]], appellent GPT un transformer décodeur seul : le même bloc, mais avec un masque causal de sorte que la position $i$ ne porte son attention que sur les positions $\le i$. L'objectif est

$$
\mathcal{L}_{\text{LM}}=-\frac{1}{T}\sum_{i=1}^{T}\log p(x_i\mid x_{<i}).
$$

Chaque position fournit un signal d'entraînement (contre environ 15 % pour la MLM), mais chacune ne voit que son contexte gauche. Voir [[llm.02]] pour la génération.

La page 29 décrit comment ChatGPT a été obtenu à partir d'un modèle GPT-3.5 : fine-tuning sur des données question-réponse, notes humaines (de 1 à 10) sur les réponses du modèle, apprentissage de la prédiction de ces notes, puis apprentissage par renforcement avec PPO pour maximiser les notes prédites. [[llm.05]] et [[rl.06]] couvrent ce pipeline en profondeur. La page 37 illustre l'apprentissage few-shot : mets quelques exemples de traduction dans le prompt et le modèle de langage prolonge le motif, sans mise à jour des poids.

## Comparaison et choix

| | Encodeur seul (BERT) | Décodeur seul (GPT) | Encodeur-décodeur (BART, T5) |
|---|---|---|---|
| Attention | bidirectionnelle | causale | encodeur bidirectionnel, décodeur causal plus attention croisée |
| Pré-entraînement | tokens masqués | token suivant | reconstruire l'entrée abîmée |
| Sortie | un vecteur par token | distribution du token suivant | séquence générée |
| Usages naturels | classification, étiquetage, QA extractive, embeddings | génération libre, chat, few-shot | traduction, résumé, réécriture |
| Faible en | génération de texte | ne voit que le contexte gauche | deux piles à maintenir |

Règles empiriques : jeu d'étiquettes fixe, données étiquetées et latence serrée, essaie un encodeur ; texte libre ou instructions avec peu d'exemples, utilise un décodeur (le support, p.38, note que les grands modèles de langage apportent l'apprentissage zero-shot et few-shot, de nouveaux embeddings, des API et la génération augmentée par récupération) ; entrée et sortie sont des séquences différentes et tu peux faire du fine-tuning, utilise un encodeur-décodeur. Un LLM sollicité par prompt est souvent le plus pratique, mais aussi le plus coûteux et le plus lent : mesure-le face à un petit modèle fine-tuné ([[sys.04]]).

## Exemple détaillé : masquage et perte

Tokens `[5,12,7,30,8,19,22,41]`, id de masque 0, les tirages de sélection `[0.9,0.05,0.5,0.1,0.14,0.8,0.01,0.6]` avec $p=0.15$ sélectionnent les positions 1, 3, 4 et 6. Les tirages d'action à ces positions `0.3, 0.85, 0.95, 0.5` donnent : masque, token aléatoire (33), inchangé, masque. L'entrée corrompue est `[5,0,7,33,8,19,0,41]` et les étiquettes sont `[-1,12,-1,30,8,-1,22,-1]`.

Pour la perte, prends deux lignes sélectionnées sur un vocabulaire de 4, logits `[2,0,0,0]` (étiquette 0) et `[0,1,0,0]` (étiquette 2). Ligne 1 : $p=e^2/(e^2+3)=0.711$, perte $0.3408$. Ligne 2 : $p=1/(e+3)=0.175$, perte $1.7437$. Moyenne $=1.0422$. Un modèle qui devine uniformément parmi $V$ tokens a une perte $\log V$, une vérification de cohérence utile.

## Implémentation

```python
import numpy as np
def mask_tokens(tokens, select_u, action_u, random_tokens, mask_id, p=0.15):
    inputs, labels = tokens.copy(), np.full(tokens.shape, -1)
    sel = select_u < p
    labels[sel] = tokens[sel]
    for i in np.where(sel)[0]:
        if action_u[i] < 0.8: inputs[i] = mask_id
        elif action_u[i] < 0.9: inputs[i] = random_tokens[i]
    return inputs, labels
```

Les bibliothèques (par exemple le collateur de données de masked-LM de HuggingFace) font cela sur des batchs et excluent aussi les tokens spéciaux de la sélection. L'exercice reprend la même logique avec les tirages aléatoires fournis en paramètre, de sorte que les tests sont exacts.

## Idées reçues courantes

- **« BERT prédit le mot suivant. »** Il prédit des mots cachés en utilisant les deux côtés ; ce n'est pas un générateur de texte.
- **« 15 % masqués signifie que 15 % des entrées montrent le masque. »** Seules 80 % des positions sélectionnées montrent le masque ; les autres sont aléatoires ou inchangées, et pourtant toutes sont dans la perte.
- **« Le vecteur `[CLS]` est un bon embedding de phrase tel quel. »** Il est entraîné pour servir une tête fine-tunée ; sans fine-tuning, des vecteurs de tokens agrégés ou un modèle d'embedding dédié fonctionnent généralement mieux (connaissance générale, à vérifier sur tes données).
- **« Les modèles encodeur-décodeur sont obsolètes. »** Ils restent solides pour la réécriture entrée-vers-sortie ; les modèles décodeur seul couvrent simplement plus de cas avec une seule conception.
- **« La prédiction de la phrase suivante est essentielle. »** Le support la liste comme l'une des deux tâches ; la pratique ultérieure l'abandonne souvent.

## Points clés à retenir

- Les objectifs auto-supervisés transforment le texte brut en signal d'entraînement sans fin ; les familles diffèrent par le masque et la cible.
- MLM : sélectionner environ 15 %, appliquer 80/10/10, moyenner l'entropie croisée sur les seules positions sélectionnées.
- Têtes de BERT : vecteur `[CLS]` pour les étiquettes de phrase, vecteurs par token pour les étiquettes de tokens, scores de début et de fin pour les segments.
- Les encodeurs-décodeurs débruiteurs reconstruisent des entrées abîmées ; la corruption de segments de T5 utilise des sentinelles.
- GPT est causal : chaque position donne un signal d'entraînement et la génération est son usage naturel.

## Explique-le toi-même

1. Pourquoi 10 % des tokens sélectionnés sont-ils laissés inchangés, et que se passerait-il si tous étaient remplacés par le token de masque ?
2. Un collègue veut classer des tickets de support en 12 catégories avec 5 000 exemples étiquetés. Quelle famille essaierais-tu en premier, et quelles sont les formes de l'entrée et de la sortie de sa tête ?

## Pour aller plus loin

- Devlin et al., « BERT: Pre-training of Deep Bidirectional Transformers » (arXiv 1810.04805), dans la bibliographie du support.
- Raffel et al., « Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer » (arXiv 1910.10683).
- Brown et al. (GPT-3, arXiv 2005.14165) ; Ouyang et al. (suivi d'instructions, arXiv 2203.02155).

## Critères de maîtrise

- [ ] Je sais masquer une séquence de tokens selon la règle 80/10/10 et calculer à la main la perte de masked-LM.
- [ ] Je sais donner les formes de l'entrée et de la sortie des têtes de classification, d'étiquetage et de QA.
- [ ] Je sais écrire l'entrée et la cible d'un exemple corrompu par segments à la manière de T5.
- [ ] Je sais expliquer quand préférer un encodeur, un décodeur ou un encodeur-décodeur.
