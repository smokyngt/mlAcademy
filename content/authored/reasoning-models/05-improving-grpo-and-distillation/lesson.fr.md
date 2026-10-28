---
id: rsn.05
title: "Améliorer GRPO et distiller les modèles de raisonnement"
summary: Surveiller un entraînement GRPO, comprendre les biais cachés dans ses normalisations et les correctifs adoptés par la communauté, puis comparer le RL à l'alternative moins coûteuse qui consiste à affiner un petit modèle sur les traces de raisonnement d'un grand, et voir comment le batching rend tout cela abordable.
minutes: 100
teaches: [rsn.grpo-fixes, rsn.distill, rsn.batching]
requires: [rsn.grpo, llm.sft, info.entropy]
sources:
  - "reasoning-from-scratch/ch07/01_main-chapter-code/ch07_main.ipynb (Sec. 7.1-7.8): GRPO metrics (loss, reward, response length, advantage mean/std, entropy), clipped ratio and the token/sequence-level variants table, KL experiment and its failure analysis, <think> tokens and format reward, list of fifteen GRPO modifications. READ."
  - "reasoning-from-scratch/ch08/01_main-chapter-code/ch08_main.ipynb (Sec. 8.1-8.10): hard versus soft distillation, dataset of teacher traces, example building (chat template, <think> format, EOS), length filtering and split, cross-entropy as average log-probability, training loop, results table. READ."
  - "reasoning-from-scratch/chE/chE_main.ipynb (Sec. E.1-E.8): latency versus throughput, padding and attention masks, batched evaluation, sampling, GRPO rollouts and distillation, timing and memory table. READ."
  - "reasoning-from-scratch/ch06/01_main-chapter-code/ch06_main.ipynb (Sec. 6.8-6.12): the baseline GRPO loss and run that chapter 7 improves. READ."
---

## Où cela s'inscrit

[[rsn.04]] t'a donné une boucle GRPO fonctionnelle. L'expérience du livre en montre la faiblesse : des gains rapides au début, puis des oscillations, et parfois un effondrement. La première moitié de cette leçon est un tour de ce qu'il faut surveiller et des correctifs proposés par la communauté ; la seconde moitié présente l'*autre* façon de construire un petit modèle de raisonnement, qui n'a pas besoin de RL du tout : copier le comportement d'un modèle fort (**distillation**). Une courte dernière section traite du batching, ce qui rend les deux approches abordables en pratique.

Tu as besoin de la perte et des avantages GRPO de [[rsn.04]], du vérificateur de [[rsn.01]], du fine-tuning supervisé d'un modèle de langage ([[llm.04]]) et de l'entropie ([[math.10]]).

## L'idée en termes simples

Enseigner à un élève par essai-erreur (RL) est lent, bruité, et l'élève prend parfois de mauvaises habitudes (divaguer, tromper le correcteur). Trois questions pratiques en découlent : *Comment savoir que l'entraînement se passe bien pendant qu'il tourne ?* (tableaux de bord : récompense, longueur des réponses, variété des tentatives). *Comment éviter qu'une leçon bruyante couvre les autres ?* (normalisation et clipping). Et *existe-t-il un raccourci ?*

Il y en a un : laisser un enseignant très fort écrire des solutions modèles, et demander au petit élève de les copier mot à mot, comme dans un cours d'écriture. C'est la **distillation**. C'est de l'apprentissage supervisé, sans échantillonnage par l'élève, sans récompense. C'est peu coûteux par pas, mais son plafond est l'enseignant, et la qualité du texte copié compte beaucoup. Enfin, quelle que soit la voie choisie, la facture de calcul est dominée par la génération de longues réponses, et le **batching** (en traiter beaucoup à la fois) est le principal levier pour la réduire.

## Partie A. Surveiller un entraînement GRPO

Le livre suit les quantités suivantes pendant un entraînement (il les trace pour 500 pas) :

| signal | ce que tu veux voir | signes d'alerte |
|---|---|---|
| **récompense moyenne** | en hausse | bloquée près de 0 (pas de signal) ou atteignant 1.0 (plus rien à apprendre) |
| **longueur des réponses** | d'abord en hausse quand le modèle apprend à raisonner, puis stable | effondrement vers des sorties très courtes, ou croissance incontrôlée |
| **perte** | à peu près plate avec un petit bruit | pas un indicateur de progrès (voir [[rsn.04]]) ; les pics sont préoccupants |
| **écart type des avantages** | stable et non nul (environ 1 est idéal ; très en dessous de 1 signifie un signal qui s'évanouit ; très au-dessus, des mises à jour en dents de scie) | tend vers 0 |
| **entropie** | modérée | effondrement vers presque 0 (tous les rollouts identiques) ou explosion vers $\log V$ |

La **moyenne** des avantages doit valoir 0 par construction ; si elle dérive, tu as un bug.

L'**entropie** de la distribution du token suivant à un pas, $p=\operatorname{softmax}(z)$, est $H=-\sum_vp_v\log p_v$, comprise entre $0$ (un token certain) et $\log V$ (uniforme). Pour les logits jouets à sept tokens du livre $[0.6667,-2,1.3333,0,-0.6667,2,-1.3333]$, les probabilités sont $0.13,0.01,0.25,0.07,0.03,0.49,0.02$ et $H=1.370$ pour un maximum $\log7=1.946$. Le nombre suivi est la moyenne sur les tokens générés d'un rollout. Règle empirique du notebook : en dessous d'environ 0.5, le modèle est quasi déterministe, 1 à 2 est typique d'un entraînement stable, bien au-delà le modèle est proche du hasard. Augmenter la température d'échantillonnage accroît la diversité des rollouts mais ne change *pas* l'entropie du modèle.

L'expérience de référence du livre (sans clipping ni KL) a montré des gains rapides de précision en évaluation au début, puis un déclin après quelques centaines de pas, avec des pics de perte : rendements décroissants plus instabilité.

## Partie B. Améliorer GRPO

### 1. Ratios avec clipping

La perte originale de [[rsn.04]] utilise un ratio $\rho=\pi_\theta/\pi_{\text{old}}$ qui vaut exactement 1 pour une seule mise à jour. Avec plusieurs mises à jour par lot, il faut le surrogat avec clipping $\min(\rho A,\operatorname{clip}(\rho,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})A)$. Les variantes diffèrent par *l'endroit* où le ratio est formé :

| méthode | niveau du ratio |
|---|---|
| GRPO de DeepSeekMath | token : $\pi_\theta(o_t\mid q,o_{<t})/\pi_{\text{old}}(\cdot)$ |
| GRPO de DeepSeek-R1 | séquence : $\pi_\theta(o\mid q)/\pi_{\text{old}}(o\mid q)$ |
| DAPO | token |
| GSPO | séquence, **normalisé par la longueur** : $\bigl(\pi_\theta(y)/\pi_{\text{old}}(y)\bigr)^{1/\lvert y\rvert}$ |

Pourquoi normaliser par la longueur ? Un ratio de séquence est l'exponentielle de la *somme* de $L$ log-ratios de tokens, donc son logarithme croît avec la longueur. Si la probabilité de chaque token n'a changé que de $0.005$ nat, un rollout de 100 tokens a $\rho=e^{0.5}=1.65$ (bien en dehors de $[0.8,1.2]$ et donc clippé), tandis que le ratio normalisé par la longueur vaut $e^{0.5/100}=1.005$. Dans l'expérience du livre, ajouter le clipping a supprimé la chute subie par la référence autour du pas 400.

**Clip higher** (DAPO) : utiliser $\epsilon_{\text{high}}>\epsilon_{\text{low}}$, par exemple 0.28 contre 0.2 (mon illustration). Avec $\epsilon_{\text{high}}=0.2$, un token d'avantage positif dont le ratio vaut 1.25 ne reçoit plus de poussée ; avec $0.28$ il garde son plein ratio, laissant de la place aux tokens de faible probabilité pour croître.

### 2. Le terme KL et pourquoi l'expérience du livre s'est effondrée

En ajoutant $\beta\cdot\operatorname{mean}(\log\pi_\theta-\log\pi_{\text{ref}})$ ([[rsn.04]]), l'expérience des auteurs s'est effondrée peu après 50 pas : la précision en évaluation est tombée vers 0 et les sorties sont devenues du charabia. Leur diagnostic :

1. Le KL est calculé sur des log-probabilités de séquence **sommées**, donc son amplitude croît avec la longueur. Avec une dérive de $0.05$ nat par token et $\beta=0.02$, la pénalité vaut $0.1$ à 100 tokens et $1.0$ à 1000 tokens, contre une constante $0.001$ si l'on moyenne. Cela incite l'optimiseur à pousser la longueur, que les journaux montraient en croissance.
2. Quand les récompenses s'effondrent (toutes à zéro), les avantages sont nuls et le terme de gradient de politique n'a **aucun gradient**, alors que le terme KL est toujours là et devient le seul gradient, ce qui a poussé le modèle vers des sorties à haute entropie.

Remèdes listés : normaliser par la longueur les log-probabilités/le KL, baisser $\beta$ (l'auteur a trouvé que 0.001 n'aidait pas ici), repondérer le KL par un poids d'importance, ou ajouter une petite récompense de format pour que les avantages ne s'annulent pas tous. La recommandation courante de plusieurs articles listés dans le notebook (Dr. GRPO, Olmo 3, DeepSeek V3.2 pour les maths) est de **ne pas utiliser de terme KL** pour les tâches de maths, ce qui supprime aussi le coût mémoire du modèle de référence.

### 3. Récompenses : la récompense de format

Utiliser `<think>`...`</think>` pour séparer le raisonnement de la réponse exige que ces tokens existent. Le tokenizer du modèle de base découpe `<think>` en plusieurs morceaux de sous-mots, donc le livre les ajoute comme tokens spéciaux (le tokenizer de la variante de raisonnement les a déjà) et récompense les modèles qui les émettent dans l'ordre. Entraîner directement le modèle *de base* ainsi marche mal, car il n'a jamais vu les balises : il faut d'abord l'affiner sur des données qui les contiennent (Partie C). Une récompense de format inconditionnelle de poids 1 a laissé le modèle collecter de la récompense avec les balises et se dégrader après environ 100 pas ; un poids de 0.1 ou une récompense de format conditionnelle ([[rsn.04]]) sont les correctifs proposés.

### 4. Normalisation : ce que font les divisions

Deux divisions de la recette originale changent *qui est entendu*. L'analyse ci-dessous est ma propre dérivation à partir des formules (le notebook liste les correctifs correspondants par leur nom mais ne les dérive pas), donc considère l'interprétation comme raisonnée plutôt que issue de la source.

**Poids par token.** Avec des rollouts $i=1..G$ de longueurs $L_i$ et d'avantages $A_i$, trois façons d'agréger la perte $-\sum(\text{weight})\cdot\log p_{\text{token}}$ :

| agrégation | poids de chaque token | poussée totale sur le rollout $i$ |
|---|---|---|
| somme de séquence (celle du livre) | $A_i/G$ | $A_iL_i/G$ |
| moyenne de séquence (1/longueur par rollout) | $A_i/(G\,L_i)$ | $A_i/G$ |
| moyenne par token (DAPO « token-level loss ») | $A_i/\sum_jL_j$ | $A_iL_i/\sum_jL_j$ |

Exemple, récompenses $[1,1,0,0]$ ($A=\pm0.866$), longueurs $[100,100,50,400]$. Somme de séquence : le rollout faux de 400 tokens est poussé vers le bas d'un total de $86.59$, celui de 50 tokens de $10.82$. Moyenne de séquence : les deux rollouts faux reçoivent le même total $0.2165$, donc **chaque token du long rollout faux est pénalisé 8 fois moins** que chaque token du court rollout faux. Poussé à sa conclusion, cela sous-punit les erreurs bavardes (et sous-récompense les longues dérivations correctes), ce qui est le biais que corrige le correctif de longueur (token-level loss). La moyenne par token traite tous les tokens du groupe de la même façon, donc la poussée totale est proportionnelle à la longueur.

**Normalisation par l'écart type.** Diviser par $\sigma$ rééchelonne les avantages de chaque prompt. Pour $G=8$ :

| corrects $k$ | $\sum\lvert A\rvert$ avec $\sigma$ | $\sum\lvert A\rvert$ sans |
|---|---|---|
| 1 | 4.948 | 1.750 |
| 4 | 7.482 | 4.000 |
| 7 | 4.948 | 1.750 |

Sans la division, un prompt résolu par la moitié des rollouts pèse $2.3$ fois plus qu'un prompt résolu par un seul ; avec elle, $1.5$ fois. La division aplatit donc le poids selon la difficulté, donnant aux prompts très difficiles et très faciles relativement *plus* de voix que ne le feraient les écarts bruts de récompense. Dr. GRPO soutient que c'est un biais et la supprime ; la liste du notebook inclut aussi DeepSeek V3.2, qui *garde* la normalisation d'origine. Les sources divergent, considère donc cela comme un choix à régler et surveille les métriques.

### 5. Le reste de la liste

La section 7.7 du notebook liste quinze modifications ; celles implémentées dans l'exercice sont :

- **Filtrage des gradients nuls / échantillonnage actif** (DAPO) : écarter les prompts dont le groupe a des récompenses identiques et en échantillonner de nouveaux, pour que chaque lot porte du signal.
- **Perte au niveau du token** (DAPO), **pas de KL** (DAPO, Dr. GRPO), **clip higher** (DAPO), **pas de normalisation par l'écart type** (Dr. GRPO), **échantillonnage d'importance et clipping au niveau de la séquence** (GSPO).

Autres éléments de la liste que je ne fais que nommer (ni dérivés ni testés ici) : échantillonnage d'importance tronqué, forces de KL propres au domaine, KL repondéré, masquage de séquences hors-politique, conservation des masques top-p/top-k de l'échantillonnage, normalisation par groupe et par récompense (GDPO), clipping des poids d'importance au lieu des mises à jour de tokens (CISPO).

## Partie C. Distiller des traces de raisonnement dans un petit modèle

### Distillation dure et douce

La distillation entraîne un **élève** sur les sorties d'un **enseignant**.

- **Distillation dure** : entraîner sur le *texte* généré par l'enseignant avec l'entropie croisée usuelle du token suivant, $L=\mathrm{CE}(y_{\text{teacher}},\,\text{student})$. C'est du SFT ordinaire ([[llm.04]]) sur des données synthétiques, sans besoin de logits. Les petits modèles DeepSeek-R1 ont été produits ainsi à partir des traces du modèle 671B.
- **Distillation douce** : faire aussi correspondre la distribution de l'enseignant, $L=\mathrm{KL}(p_{\text{teacher}}\Vert p_{\text{student}})$ par token, ce qui exige les logits de l'enseignant et un tokenizer commun.
- **Combinée** : $L=\mathrm{CE}+\lambda\,\mathrm{KL}$ (distillation de connaissances classique).

La distillation dure est le choix courant : les API propriétaires n'exposent pas les logits sur tout le vocabulaire, la distillation douce exige une tokenisation identique, et stocker les distributions complètes pour de longues séquences est coûteux. (Le notebook note aussi qu'utiliser les sorties d'un fournisseur pour entraîner peut violer ses conditions d'utilisation.) Il ajoute qu'aucune comparaison directe récente à grande échelle ne compare équitablement les deux, et que la stratégie de génération des données peut compter plus que dur contre doux.

### Le pipeline du livre

1. **Générer** des traces de l'enseignant pour les 12 000 problèmes d'entraînement MATH (les mêmes que pour le RL, en excluant MATH-500) avec DeepSeek-R1 671B via une API ; le jeu de données a coûté environ 50 dollars américains de frais d'API. Chaque entrée contient le problème, la réponse de référence, le texte de réflexion et le message final.
2. **Formater** chaque réponse en `<think>thinking</think>` + ligne vide + contenu final.
3. **Tokeniser** le prompt (avec le template de chat) et la réponse, ajouter le token de fin de séquence, mémoriser `prompt_len`.
4. **Filtrer** par longueur (le livre écarte les exemples de plus de 2048 tokens pour tenir dans 15 Go de mémoire) et isoler un petit jeu de validation (25 exemples).
5. **Entraîner** avec AdamW (taux d'apprentissage $5\times10^{-6}$ dans la démo, $10^{-5}$ dans l'exécution complète), clip de la norme du gradient à 1.0, trois époques, en suivant la perte de validation.

### La perte : entropie croisée sur la réponse seulement

Pour des tokens $x_1..x_L$ avec la réponse commençant à l'indice $m$ (`prompt_len`), la ligne $t$ du modèle prédit $x_{t+1}$, donc

$$
\mathcal{L}_{\text{SFT}}=-\frac1{L-m}\sum_{t=m-1}^{L-2}\log p_\theta\bigl(x_{t+1}\mid x_{\le t}\bigr).
$$

Seuls les tokens de la réponse comptent (le prompt est donné, pas appris). C'est *exactement* l'opposé de la log-probabilité **moyenne** de la réponse vue en [[rsn.03]], que le notebook vérifie numériquement contre `torch.nn.functional.cross_entropy` (le même nombre). Cette égalité explique pourquoi le SFT n'a besoin d'aucun code spécial : la perte de classification standard suffit.

### Filtrer les traces avec le vérificateur

Le pipeline du livre filtre uniquement par *longueur* et n'utilise le vérificateur que pour mesurer la précision de l'enseignant. Une extension naturelle, que l'exercice implémente, est l'**échantillonnage par rejet** : ne garder une trace que si sa réponse finale est vérifiée comme correcte ([[rsn.01]]), si son contenu n'est pas vide et si elle respecte le budget de longueur. Écarter les mauvaises traces évite d'enseigner à l'élève des erreurs assurées ; le coût est moins de données et un biais vers les problèmes que l'enseignant résout. Je n'ai trouvé dans les sources aucune expérience comparant un entraînement filtré et non filtré, donc le bénéfice est une attente plausible, pas un résultat mesuré ici.

### Résultats (élève Qwen3 0.6B, MATH-500, 500 problèmes)

| ligne | entraînement | précision |
|---|---|---|
| 1 | modèle de base | 15.2% |
| 2 | variante de raisonnement officielle | 48.2% |
| 3 | GRPO, 50 pas ([[rsn.04]]) | 47.4% |
| 4-6 | distillé sur des traces DeepSeek-R1, époques 1, 2, 3 | 30.6%, 32.4%, 33.6% |
| 7-9 | distillé sur des traces Qwen3-235B-A22B, époques 1, 2, 3 | 45.0%, 43.8%, 44.2% |

La perte de validation finale était d'environ 0.53 à 0.54 pour les traces R1 et 0.39 à 0.40 pour les traces Qwen3-235B. Observations : (1) la distillation fait passer un modèle de 15% à 34% à 45% par le seul entraînement supervisé. (2) Le **choix de l'enseignant** a déplacé la précision de plus de 10 points : les traces de l'enseignant de la même famille ont donné une perte de validation bien plus basse, ce qui signifie que son texte est plus facile à prédire pour l'élève. Le notebook rapporte les chiffres mais pas la cause ; des facteurs plausibles sont le vocabulaire partagé et le style d'écriture. Considère cela comme une hypothèse. (3) Des époques supplémentaires ont donné des changements faibles, peut-être bruités (erreur standard d'environ 2 points sur 500 problèmes, [[rsn.01]]). (4) Le meilleur modèle distillé est proche, sans le dépasser, du GRPO à partir de la même base et du modèle officiel ; une exécution chacun.

L'entraînement du notebook a pris environ 3 heures 5 minutes et 15 Go sur un DGX Spark ; l'exécution GRPO de [[rsn.04]] a pris 68 minutes sur un H100 pour son script sans batch. Ce sont des machines et des réglages différents, ne les lis donc pas comme une comparaison directe des coûts.

## Partie D. Batching et débit

L'annexe E sépare deux objectifs : la **latence** (la rapidité de réponse à un prompt) et le **débit** (le nombre de prompts par heure). Le batching sert le débit.

**Padding et masques.** Les séquences d'un lot doivent former un rectangle, donc les prompts plus courts sont complétés. Pour la génération, le livre complète à **gauche**, de sorte que le dernier vrai token de chaque séquence se trouve à la même position finale où la génération continue, et conserve un masque d'attention (True marque le padding) pour que les positions de padding soient ignorées à la fois comme clés et comme requêtes. La fonction `left_pad` de l'exercice renvoie exactement un tel tableau et son masque.

**Gaspillage.** Si les longueurs d'un lot diffèrent beaucoup, une grande partie du rectangle est du padding. Les longueurs $[10,200,12,190,11,210]$ groupées par paires consécutives demandent $2\cdot200+2\cdot190+2\cdot210=1200$ cellules pour contenir $633$ vrais tokens, soit un gaspillage de $0.4725$. Triées d'abord par longueur (paires $(10,11),(12,190),(200,210)$), les cellules sont $822$ et le gaspillage $0.2299$. Le regroupement par longueur (bucketing) est le remède standard.

**Ce que le batching apporte** (tableau du livre, temps réel en minutes) :

| tâche | lot | H100 simple vers batché | accélération | DGX Spark simple vers batché | accélération |
|---|---|---|---|---|---|
| évaluation MATH-500 | 64 | 90.0 à 16.0 | 5.6x | 174.7 à 108.4 | 1.6x |
| échantillonnage d'auto-cohérence | 3 | 252.0 à 129.0 | 2.0x | 340.8 à 243.3 | 1.4x |
| rollouts GRPO | 4 | 68.0 à 19.0 | 3.6x | 63.7 à 23.1 | 2.8x |
| distillation | 4 | 10.9 à 9.1 | 1.2x | 32.8 à 28.2 | 1.2x |

La mémoire augmente avec la taille du lot : de 1.8 Go à 23.4 Go pour l'évaluation avec un lot de 64, tandis que les autres tâches varient à peine (2.45 Go pour l'auto-cohérence, 43 à 45 Go pour GRPO, 8.3 Go pour la distillation). Remarques de l'annexe : le batching n'est pas garanti d'aider sur de petits modèles ou du matériel faible (le surcoût du padding peut manger le gain) ; l'auto-cohérence et les rollouts GRPO utilisent le *même* prompt répété, donc n'ont pas besoin de padding ; l'auto-raffinement ([[rsn.03]]) est séquentiel et n'est pas fourni sous forme batchée ; les lots de distillation ont des longueurs de prompt et de réponse différentes et demandent un code conscient du padding. Une variante « stop-aware » retire les lignes terminées du lot actif pour que celles qui finissent tôt cessent de coûter du calcul.

## Exemple détaillé

Tout ce qui est numérique plus haut est reproduit par les deux exercices : poids et poussées par token ($86.5875$, $10.8234$, $0.2165$), tableau de masse ($4.948$, $7.482$, $1.75$, $4.0$), ratio de séquence ($1.6487$ contre $1.0050$), mise à l'échelle du KL ($0.1$ contre $1.0$), entropie ($1.3700$), perte SFT comme moyenne des log-probabilités négatives, KL entre $(0.5,0.5)$ et $(0.9,0.1)$ ($0.5108$), gaspillage de padding ($0.4725$ et $0.2299$) et accélérations.

## Implémentation

```python
import numpy as np

def sft_loss(logits, token_ids, prompt_len):
    z = logits - logits.max(axis=-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))      # (L-1, V)
    rows = np.arange(prompt_len - 1, len(token_ids) - 1)       # rows predicting answer tokens
    return -np.mean(lp[rows, np.asarray(token_ids)[rows + 1]])

def left_pad(seqs, pad_id):
    m = max(map(len, seqs))
    ids = np.array([[pad_id] * (m - len(s)) + list(s) for s in seqs])
    mask = np.array([[True] * (m - len(s)) + [False] * len(s) for s in seqs])
    return ids, mask
```

En PyTorch, la perte est `torch.nn.functional.cross_entropy(answer_logits, answer_targets)` ; les bibliothèques de RL de production (le livre renvoie à des scripts de son matériel complémentaire, non reproduits ici) fournissent les variantes avancées de GRPO et la génération batchée.

## Idées reçues

- « Une perte d'entraînement plus basse dans l'exécution régularisée par KL signifie un meilleur modèle. » Dans l'exécution KL des auteurs, la perte se comportait bien, pourtant la précision est tombée à zéro.
- « Les modèles distillés peuvent dépasser leur enseignant. » La distillation dure copie le comportement de l'enseignant ; le plafond est la qualité de l'enseignant (et la capacité de l'élève).
- « N'importe quel enseignant en vaut un autre. » Dans le livre, le choix de l'enseignant a déplacé la précision de plus de 10 points.
- « Le batching accélère toujours. » Les gains vont de 1.2x à 5.6x ici et peuvent disparaître sur de petits modèles ou des CPU.
- « Supprime la division par l'écart type et le biais disparaît. » Tu échanges une pondération contre une autre ; surveille et compare.

## Points clés

- Surveille la récompense, la longueur, l'écart type des avantages et l'entropie ; la valeur de la perte n'est pas un indicateur de progrès.
- Applique le clipping au ratio (normalisé par la longueur pour les ratios de séquence), garde le terme KL désactivé pour les maths sauf raison contraire, évite les récompenses de format inconditionnelles.
- L'agrégation (somme, moyenne par séquence, moyenne par token) et la division par l'écart type déterminent les poids par token ; elles changent les biais de longueur et de difficulté.
- La distillation dure est du SFT sur des traces d'enseignant avec l'entropie croisée sur les seuls tokens de la réponse ; filtre par longueur et (en extension) par correction vérifiée.
- Le batching avec padding à gauche, masques et regroupement par longueur est le levier de débit ; sache quand il aide.

## Explique-le toi-même

1. Explique pourquoi sommer les log-ratios de tokens dans le ratio d'importance fait que les longs rollouts sont plus souvent clippés, et comment la normalisation par la longueur change cela.
2. Pourquoi l'entropie croisée du SFT sur les tokens de la réponse est-elle la même quantité que l'opposé de la log-probabilité moyenne de [[rsn.03]] ?

## Pour aller plus loin

- Chapitres 7 et 8 du livre, annexe E (batching) et les scripts complémentaires listés dans les notebooks (non ouverts).
- Articles DAPO, Dr. GRPO, GSPO, DeepSeek-R1 et DeepSeek V3.2 nommés dans le notebook du chapitre 7 (non ouverts pour cette leçon).
- Retour à l'hygiène d'évaluation dans [[rsn.01]] : chaque comparaison ci-dessus mérite une barre d'erreur.

## Critères de maîtrise

- Je sais lister les métriques à suivre dans un entraînement GRPO et dire à quoi ressemble chaque défaillance.
- Je sais calculer les poids par token sous les agrégations somme, moyenne et moyenne par token, et expliquer le biais de longueur de la deuxième.
- Je sais expliquer pourquoi un terme KL sur des log-probabilités sommées croît avec la longueur et ce qu'a fait le livre à ce sujet.
- Je sais construire des exemples de SFT à partir de traces d'enseignant, calculer l'entropie croisée sur la seule réponse, et filtrer les traces avec un vérificateur.
- Je sais estimer le gaspillage de padding et décider quand le batching et le tri aident.
