---
id: vis.07
title: "Image et texte ensemble : légendage et question-réponse visuelle"
summary: "Des modèles encodeur-décodeur qui lisent une image avec un CNN et écrivent du texte avec un RNN, l'attention douce sur des régions d'image (Show, Attend and Tell), l'échantillonnage à température, des métriques de légendes comme BLEU-1, et la question-réponse visuelle avec son problème de biais des jeux de données."
minutes: 75
teaches: [tal.captioning, tal.vqa]
requires: [seq.attention, seq.rnn, seq.cnn-arch, vis.transfer, info.entropy]
sources:
  - "cours-tal-05.pdf (RCP217, 'Vision et Texte', after a course by N. Thome, CNAM, French slides, PDF p.1-22, all pages read): CNN plus RNN encoder-decoder (p.2-3), RNN recap and task types (p.4-6), Show and Tell and the practical-session model (p.7-9), training and evaluation on Flickr8k (p.10-14), temperature sampling (p.12), Show, Attend and Tell with soft and hard attention (p.15-17), visual question answering and its biases (p.18-21), bibliography (p.22). READ."
---

## Objectifs d'apprentissage

- Dessiner l'architecture de légendage à encodeur CNN et décodeur RNN et dire ce qui est fourni au décodeur à chaque pas.
- Écrire un pas de décodage avec attention douce sur $R$ régions d'image, avec toutes les formes.
- Expliquer l'échantillonnage à température et le calculer sur une distribution.
- Calculer BLEU-1 avec écrêtage et pénalité de brièveté, et dire ce qu'il ne voit pas.
- Décrire la question-réponse visuelle (VQA) et expliquer pourquoi ses jeux de données sont biaisés.

## Où ça s'inscrit

Les leçons précédentes t'ont donné deux encodeurs : les CNN qui transforment des images en caractéristiques ([[seq.01]], [[vis.02]]) et les réseaux récurrents plus l'attention qui transforment des séquences en séquences ([[seq.03]], [[seq.05]]). Le cours du CNAM « Vision et Texte » (cours-tal-05) les colle ensemble : lire l'image avec un CNN, écrire des mots avec un RNN. Cette seule idée donne le légendage d'images et, avec une question en entrée supplémentaire, la question-réponse visuelle. C'est l'ancêtre des modèles vision-langage de [[vis.06]] et [[adv.01]], qui remplacent les deux moitiés par des transformers ([[trf.04]]), mais l'interface (caractéristiques d'image en entrée, tokens en sortie, entraînement par entropie croisée) est inchangée.

## L'idée en termes simples

Imagine un guide touristique qui jette un coup d'œil à une photo puis se met à parler. Le coup d'œil, c'est l'encodeur : il comprime la photo en un résumé de « ce qu'il y a dedans ». La parole, c'est le décodeur : il dit un mot à la fois, et chaque mot dépend des mots déjà dits et du résumé.

Le cours insiste (p.3) sur le fait que certaines tâches demandent plus que nommer des objets : il faut trouver les relations entre les objets et les mettre en langage naturel (« un oiseau volant au-dessus d'une étendue d'eau »).

Un seul résumé de la photo est un goulot d'étranglement, tout comme l'unique état final en traduction ([[seq.05]]). Le remède est le même : garder une grille de résumés de régions et laisser le guide regarder différentes parties de la photo tout en parlant de différents mots : en disant « oiseau », regarder l'oiseau, en disant « eau », regarder la moitié basse. C'est la version avec attention, « Show, Attend and Tell ».

La question-réponse visuelle transforme le guide en maître de quiz : tu donnes la photo et une question (« quel sport est-ce ? ») et le système doit répondre. Le cours la qualifie de très complexe (p.19) : elle exige un modèle précis de l'image et du texte, un modèle de leur interaction, une compréhension globale de la scène et un peu de raisonnement.

## Légendage avec CNN plus RNN

Le cours (p.2) énonce la recette : un réseau convolutif pour le côté vision et un réseau récurrent pour le côté texte. La page 7 décrit « Show and Tell » (Vinyals et al., 2015) : l'image est encodée par des caractéristiques profondes extraites d'un CNN, le texte est encodé en one-hot sur le vocabulaire, et plusieurs variantes d'architecture sont possibles. Les pages 8 et 9 donnent le modèle utilisé en travaux pratiques :

- Image : caractéristiques profondes d'un réseau VGG, réduites par ACP ([[cml.05]]) à la dimension 100.
- Mots : plongements GloVe ([[seq.02]]) augmentés des tokens de début et de fin `<debut>` et `<fin>` ; ni le plongement GloVe ni l'encodeur d'image ne sont affinés.
- À chaque pas, le plongement du mot précédent est concaténé avec le vecteur d'image et fourni au RNN, dont la sortie passe par un softmax sur le vocabulaire.

Avec un vocabulaire de $V$ mots, un vecteur d'image $v\in\mathbb{R}^{100}$, un plongement de mot $e_t\in\mathbb{R}^{E}$ :

$$
h_t=\operatorname{RNN}\big([e_{t-1};v],\,h_{t-1}\big),\qquad p(w_t\mid w_{<t},\text{image})=\operatorname{softmax}(W_o h_t).
$$

Formes : entrée concaténée $(E+100)$, état caché $(H)$, logits $(V)$.

### Entraînement

Comme pour la modélisation du langage et la traduction automatique (p.10), la perte est l'entropie croisée sur le mot suivant, avec forçage de l'enseignant (teacher forcing), remplissage et masquage pour mettre en lots des légendes de longueurs inégales ([[llm.02]] donne le lien entre cette perte et la perplexité). Pour une légende $w_1..w_T$ :

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p(w_t\mid w_{<t},\text{image}).
$$

La session pratique utilise Flickr8k (p.11) : 6 000 images d'entraînement et 1 000 chacune pour la validation et le test, avec un vocabulaire réduit à 100 ou 1 000 mots pour accélérer les choses.

### Génération et température

Au moment du test (p.11), la légende est produite par échantillonnage aléatoire avec une température ; la recherche en faisceau (beam search) est l'amélioration suggérée. La température $t$ élève chaque probabilité à la puissance $1/t$ et renormalise :

$$
p'_i=\frac{p_i^{1/t}}{\sum_j p_j^{1/t}}.
$$

De façon équivalente, $\operatorname{softmax}(\log p/t)$. Un petit $t$ affûte vers le mot le plus probable, un grand $t$ aplatit. Sur $p=(0.3,0.5,0.1,0.1)$ (une distribution de somme 1), j'obtiens, en exécutant le code de l'exercice : $t=0.25$ donne $(0.114,0.883,0.001,0.001)$ et $t=2$ donne $(0.290,0.375,0.168,0.168)$. La diapositive (p.12) imprime $(0.11,0.887,0.001,0.001)$ et $(0.29,0.38,0.165,0.165)$, ce qui correspond de près, mais sa distribution de départ est saisie comme $(0.3,0.5,0.01,0.01)$, qui n'a pas pour somme 1 et ne reproduit pas les nombres pour $t=2$ ; je lis cela comme une faute de frappe pour $0.1$. Voir [[llm.02]] pour la même opération sur les modèles de langage.

## Attention sur des régions d'image

Les pages 15 et 16 présentent Show, Attend and Tell (Xu et al., 2015), qui utilise l'information spatiale. Le CNN n'est pas agrégé en un seul vecteur ; sa dernière carte convolutive donne $R$ descripteurs de régions $a_i\in\mathbb{R}^{D}$ (pour une grille $14\times14$, $R=196$ ; cette taille de grille est une connaissance générale, le cours ne la donne pas). Au pas $t$ :

1. L'état précédent du décodeur $h_{t-1}\in\mathbb{R}^H$ est comparé à chaque région. Le cours dit que le score $e_{t,i}$ est essentiellement un produit scalaire entre $h_{t-1}$ et chaque $a_i$, éventuellement après de simples transformations. Avec une matrice $W_{att}\in\mathbb{R}^{D\times H}$ :
$$
e_{t,i}=a_i^\top W_{att}\,h_{t-1}.
$$
2. Poids : $\alpha_{t}=\operatorname{softmax}(e_t)\in\mathbb{R}^R$.
3. Contexte : $\hat z_t=\sum_i\alpha_{t,i}\,a_i\in\mathbb{R}^D$.
4. Le contexte est combiné avec le mot précédent et l'état précédent pour donner le nouvel état, $h_t=\tanh(W_x e_{t-1}+W_z\hat z_t+W_h h_{t-1}+b)$ dans notre cellule simplifiée (l'article utilise un LSTM), et les logits $W_o h_t\in\mathbb{R}^V$.

**Attention douce vs dure (p.15).** L'attention douce est la moyenne pondérée ci-dessus : dérivable, entraînée par la rétropropagation ordinaire. L'attention dure choisit une région (un choix binaire) ; elle n'est pas dérivable et le cours dit qu'elle est entraînée avec REINFORCE ([[rl.05]]). La page 17 montre les cartes d'attention sur une image pour la légende « a bird flying over a body of water ».

### Exemple travaillé (reproduit dans l'exercice)

Quatre régions de caractéristiques $a_1=(1,0,0)$, $a_2=(0,1,0)$, $a_3=(0,0,1)$, $a_4=(1,1,0)$, $W_{att}=I_{3\times2}$ (identité complétée) et $h_{t-1}=(2,1)$. Alors $W_{att}h=(2,1,0)$ et les scores sont $(2,1,0,3)$. Le softmax donne $\alpha=(0.237,0.087,0.032,0.644)$ et le contexte $\hat z=(0.881,0.731,0.032)$. La région 4 domine parce qu'elle s'aligne le mieux avec la requête. Formes : $A:(R,D)=(4,3)$, $\alpha:(4,)$, $\hat z:(3,)$.

## Évaluer des légendes

Le cours évalue en montrant des légendes générées (p.13-14) et ne donne aucune métrique. Ce qui suit est donc de la connaissance générale. Plusieurs légendes valides existent pour une même image, nous comparons donc à plusieurs références humaines par recouvrement de n-grammes.

**BLEU-1.** Avec un candidat $c$ de longueur $|c|$ et des références :

$$
\text{BLEU-1}=\mathrm{BP}\cdot\frac{\sum_{w}\min\big(\mathrm{count}_c(w),\,\max_r\mathrm{count}_r(w)\big)}{|c|},\qquad
\mathrm{BP}=\begin{cases}1&|c|>\rho\\ e^{1-\rho/|c|}&\text{otherwise}\end{cases}
$$

où $\rho$ est la longueur de la référence la plus proche de $|c|$. L'écrêtage empêche un candidat de répéter un mot qui correspond. Exemple : le candidat « a bird flies over water » contre « a bird flying over a body of water » et « the bird is over the sea ». Correspondances : a, bird, over, water donnent $4/5$ ; la longueur de la référence la plus proche est 6, donc $\mathrm{BP}=e^{1-6/5}=0.819$ et BLEU-1 $=0.655$. Répéter « the » six fois contre « the cat sat » donne $1/6$ grâce à l'écrêtage.

BLEU-1 ignore l'ordre des mots, les synonymes (« sea » vs « water ») et l'exactitude des relations ; les variantes à n-grammes d'ordre supérieur et les métriques de consensus comme CIDEr (similarité de n-grammes pondérée par tf-idf) en corrigent une partie, et le jugement humain reste la référence. Voir [[nlp.06]] pour BLEU et ROUGE en traduction et en résumé.

## Question-réponse visuelle

La page 20 énonce la tâche : l'entrée est une question et une image, la sortie une réponse. Une architecture standard (connaissance générale ; le cours ne donne que la tâche et une référence à Malinowski et al., « Ask Your Neurons », où un LSTM lit la question et des caractéristiques CNN de l'image lui sont fournies) est :

- encoder l'image $v\in\mathbb{R}^{D}$ avec un CNN, la question $q\in\mathbb{R}^{H}$ avec un RNN ;
- les fusionner (concaténation, ou produit terme à terme de vecteurs projetés) ;
- classer sur un ensemble fixe de réponses fréquentes, ou générer la réponse mot par mot.

La diapositive p.2 montre exactement cette image : un CNN côté image et un RNN côté texte. En comptant les « types de tâches RNN » (p.5) : le légendage est un-vers-plusieurs (une image, plusieurs mots), le VQA est plusieurs-vers-un (mots de la question et une image, une réponse).

### Biais du jeu de données

La page 21 prévient que les données sont souvent biaisées, surtout en VQA. Pour la question « quel sport est-ce ? », la réponse « tennis » apparaît 41 % du temps. Un modèle peut obtenir un score correct en ignorant l'image. Le cours liste les remèdes : VQA 2.0 ajoute des images pour que chaque question ait des réponses différentes sur des images différentes, et VQA-CP fait délibérément différer les a priori de réponses entre l'entraînement et le test. L'exercice implémente la *référence a priori* : répondre toujours la réponse d'entraînement la plus fréquente. Si ton modèle la bat à peine, il ne regarde pas l'image. [[saf.05]] discute plus largement des biais de données.

## Implémentation

```python
import numpy as np
def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attend(h_prev, A, W_att):            # A: (R, D), W_att: (D, H)
    alpha = softmax(A @ (W_att @ h_prev))   # (R,)
    return alpha, alpha @ A                 # context: (D,)
```

En pratique, tu utiliserais PyTorch (`nn.LSTM`/`nn.MultiheadAttention`) ou un modèle vision-langage pré-entraîné ; l'évaluation utilise la boîte à outils de légendes COCO pour BLEU, CIDEr et d'autres.

## Idées reçues

- **« Le CNN est entraîné avec les légendes. »** Dans la session pratique du cours, les caractéristiques CNN sont gelées ; seul le décodeur est entraîné. Affiner le CNN est un choix distinct.
- **« La carte d'attention montre ce que le modèle a compris. »** Elle montre où il a regardé ; la fidélité n'est pas garantie (le cours dit quelque chose de semblable de l'attention des transformers, p.15).
- **« Un BLEU élevé signifie une bonne légende. »** BLEU récompense le recouvrement de mots ; une légende fausse mais fluide peut beaucoup recouvrir, et une paraphrase correcte peut avoir un score bas.
- **« La précision en VQA prouve un raisonnement visuel. »** Avec des a priori comme tennis à 41 %, un modèle uniquement textuel peut sembler bon ; compare avec la référence a priori.
- **« La température d'échantillonnage change le modèle. »** Elle ne fait que remodeler la distribution de sortie au moment de la génération.

## Points clés à retenir

- Le légendage est une tâche RNN un-vers-plusieurs : caractéristiques CNN en entrée, un mot en sortie par pas, entraînement par entropie croisée.
- L'attention douce remplace un résumé unique par une moyenne pondérée de caractéristiques de régions : scores, softmax, somme pondérée.
- L'attention dure n'est pas dérivable et nécessite REINFORCE.
- Température $t$ : $p_i^{1/t}$ renormalisé ; sous 1 il affûte, au-dessus de 1 il aplatit.
- BLEU-1 est la précision de unigrammes écrêtée multipliée par une pénalité de brièveté ; c'est un indicateur grossier.
- Les données de VQA sont biaisées ; compare toujours à une référence a priori de réponses.

## Explique-le toi-même

1. Pourquoi l'attention sur $R$ régions aide-t-elle pour une longue légende, et quelle est la forme du vecteur de contexte qu'elle produit ?
2. Ton modèle de VQA atteint 45 % de précision et « la réponse la plus fréquente » atteint 41 %. Que conclus-tu et que testes-tu ensuite ?

## Pour aller plus loin

- Vinyals et al., « Show and Tell » (arXiv 1411.4555) ; Xu et al., « Show, Attend and Tell » (arXiv 1502.03044) ; Malinowski et al., « Ask Your Neurons » (arXiv 1505.01121) ; Karpathy et Fei-Fei (arXiv 1412.2306) : tous listés dans la bibliographie du cours.

## Critères de maîtrise

- [ ] Je sais écrire un pas de décodage augmenté d'attention et donner chaque forme.
- [ ] Je sais appliquer une température à une distribution à la main.
- [ ] Je sais calculer BLEU-1 avec écrêtage et pénalité de brièveté.
- [ ] Je sais expliquer pourquoi le VQA a besoin d'une référence a priori de langage.
