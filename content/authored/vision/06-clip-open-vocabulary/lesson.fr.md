---
id: vis.06
title: "CLIP et la vision à vocabulaire ouvert"
summary: "Aligne un encodeur d'images et un encodeur de texte avec une perte contrastive symétrique, dérive-la et implémente-la avec une température, et utilise l'espace partagé pour la classification zero-shot et la recherche."
minutes: 75
teaches: [vis.clip]
requires: [vis.contrastive, vis.patch-embed, trf.architectures, info.entropy]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/18-open-vocab-clip/docs/en.md (Sec. 'The Problem' and 'The Concept': two towers, symmetric objective, SigLIP, zero-shot recipe, prompt templates, uses; the Build It code was not read). READ."
  - "ai-engineering-from-scratch/phases/12-multimodal-learning/02-clip-contrastive-pretraining/docs/en.md (Sec. 'The Problem' and 'The Concept': dual encoder, InfoNCE, temperature, sigmoid loss, zero-shot, linear probe vs fine-tuning, zero-shot ceiling; the code and exercises were not read). READ."
---

## Où ça s'inscrit

Un classifieur entraîné sur ImageNet ne sait nommer que ses 1000 classes ; une nouvelle catégorie signifie de nouvelles étiquettes et un ré-entraînement. CLIP a changé le cadre : entraîné sur des centaines de millions de paires image-légende récupérées sur le web, le modèle obtenu peut classer selon n'importe quelle liste de catégories que tu peux écrire avec des mots. Presque tous les systèmes vision-langage modernes, des générateurs texte-vers-image aux modèles de chat multimodaux, reposent sur un espace de plongement de style CLIP. Cette leçon utilise le ViT de [[vis.05]] comme tour d'images, l'encodeur de texte transformer de [[trf.04]] comme tour de texte, et la perte contrastive de [[vis.05]], maintenant entre deux modalités. Le vocabulaire du transfert (sonde linéaire, fine-tuning) vient de [[vis.02]].

## L'idée en termes simples

Imagine deux traducteurs : l'un lit des photographies, l'autre lit des phrases, et chacun écrit une courte « adresse » (une liste de nombres) pour ce qu'il a lu. L'entraînement leur apprend à s'accorder : une photo d'un chien sur une plage et la phrase « un chien qui court sur la plage » doivent recevoir des adresses voisines, alors que la même photo et la phrase « une assiette de pâtes » doivent en recevoir de distantes. La seule supervision est que chaque paire des données d'entraînement va ensemble ; tout autre appariement du même lot compte gratuitement comme un désaccord.

Ensuite, il n'y a plus de liste fixe de classes. Pour classer une nouvelle photo dans {chat, chien, vélo}, écris les phrases « une photo d'un chat », « une photo d'un chien », « une photo d'un vélo », obtiens une adresse pour chaque phrase, obtiens une adresse pour la photo, et choisis la phrase dont l'adresse est la plus proche. Une nouvelle classe coûte une phrase, pas un jeu de données. Le même carnet d'adresses fonctionne aussi pour la recherche : tape une phrase, trouve les photos les plus proches.

## Mise en place et perte

**Deux tours.** Un encodeur d'images $f$ (un ViT ou un ResNet) et un encodeur de texte $g$ (un transformer) se terminent chacun par une projection linéaire vers la même dimension $d$. Pour un lot de $N$ paires $(x_i,y_i)$, calcule et normalise en L2

$$
u_i=\frac{f(x_i)}{\lVert f(x_i)\rVert},\qquad v_i=\frac{g(y_i)}{\lVert g(y_i)\rVert}\qquad(u_i,v_i\in\mathbb{R}^d).
$$

Avec $U,V\in\mathbb{R}^{N\times d}$ empilant ces lignes, la **matrice de logits** est

$$
S=\frac{UV^\top}{\tau}\in\mathbb{R}^{N\times N},\qquad S_{ij}=\frac{\cos(f(x_i),g(y_j))}{\tau},
$$

où $\tau>0$ est la **température**. Dans CLIP c'est un scalaire appris, paramétré par son logarithme et initialisé à $0.07$ d'après les sources (la valeur apprise est bornée pour qu'elle ne puisse pas s'effondrer vers zéro).

### Dérivation : InfoNCE symétrique

Fixe l'image $i$. La tâche : parmi les $N$ légendes du lot, trouver la sienne. C'est une classification à $N$ classes avec les logits $S_{i1},\dots,S_{iN}$ et la vraie classe $i$. L'entropie croisée de [[vis.02]] donne

$$
\ell^{\,i\to t}_i=-\log\frac{\exp(S_{ii})}{\sum_{j=1}^N\exp(S_{ij})}=-S_{ii}+\log\sum_{j}\exp(S_{ij}).
$$

Faire de même du côté texte (fixer la légende $j$, trouver son image parmi $N$ images) utilise la colonne $S_{\cdot j}$ :

$$
\ell^{\,t\to i}_j=-S_{jj}+\log\sum_{i}\exp(S_{ij}).
$$

L'objectif de CLIP fait la moyenne des deux directions et de tous les éléments :

$$
L=\frac12\left(\frac1N\sum_{i=1}^N\ell_i^{\,i\to t}+\frac1N\sum_{j=1}^N\ell_j^{\,t\to i}\right).
$$

En termes matriciels : image-vers-texte est l'entropie croisée softmax ligne par ligne de $S$ avec l'identité comme cible, texte-vers-image est la même chose pour $S^\top$. Comparé à NT-Xent dans [[vis.05]], les positifs vivent dans deux modalités différentes, donc la matrice de similarité est $N\times N$ sans diagonale à masquer ; les négatifs de l'image $i$ sont les $N-1$ autres légendes.

**Gradient par rapport aux logits.** À partir de la formule à une seule direction (la même règle $p-e_y$ que dans [[vis.02]]), avec le softmax par lignes $P=\mathrm{softmax}_{\text{row}}(S)$ et le softmax par colonnes $Q=\mathrm{softmax}_{\text{row}}(S^\top)^\top$ (chaque colonne de $Q$ a donc pour somme un) et $E$ l'identité :

$$
\frac{\partial L}{\partial S}=\frac1{2N}\big[(P-E)+(Q-E)\big]\in\mathbb{R}^{N\times N}.
$$

Interprétation : chaque mauvais appariement $(i,j)$ est repoussé vers le bas proportionnellement à la masse de softmax qu'il reçoit actuellement, dans les deux directions ; chaque bon appariement est poussé vers le haut. Les gradients par rapport aux plongements suivent par la règle de la chaîne à travers $S=UV^\top/\tau$ et la normalisation (un facteur $1/\tau$ apparaît, donc un petit $\tau$ rend la mise à jour plus grande et plus nette).

**Température.** Comme $|\cos|\le1$, les logits se situent dans $[-1/\tau,\,1/\tau]$. Un grand $\tau$ resserre les logits, donc même un modèle parfaitement aligné garde une perte élevée, et tous les négatifs se ressemblent. Un petit $\tau$ donne des logits étalés et un softmax net qui se concentre sur les négatifs les plus difficiles ; trop petit, l'entraînement devient instable ou trop focalisé sur le bruit. C'est pourquoi CLIP apprend $\tau$. Quand $\tau\to\infty$, chaque ligne devient uniforme et la perte tend vers $\log N$.

**Taille de lot.** Chaque paire a $N-1$ négatifs, donc de plus grands lots donnent un signal plus fort ; CLIP a été entraîné avec une taille de lot d'environ 32k d'après les sources. En entraînement distribué, le softmax a besoin de tous les plongements sur chaque périphérique (un all-gather), ce qui est un coût de communication.

**Alternative sigmoïde (SigLIP).** Remplace le softmax par une perte binaire indépendante pour chaque paire : avec $y_{ij}=+1$ sur la diagonale et $-1$ ailleurs, $L=\frac1N\sum_{i,j}\log\big(1+e^{-y_{ij}S_{ij}}\big)$ (les sources ajoutent aussi un biais appris). Il n'y a pas de normalisation par ligne ou par colonne, donc pas besoin d'all-gather et elle se comporte bien avec des lots petits comme très grands. Les sources rapportent qu'elle égale ou dépasse CLIP à données égales.

*Un résultat connu que les sources ne font que mentionner :* la perte InfoNCE avec $N$ candidats donne une borne inférieure de l'information mutuelle entre les deux vues, $I\ge\log N-L$, ce qui explique pourquoi la perte peut se lire comme « combien de bits de l'appariement le modèle a récupérés ».

## Classification zero-shot, prompts et recherche

**Recette zero-shot.** Avec les noms de classes $c=1..K$ :

1. Construis des prompts, par exemple « a photo of a {class} ».
2. Encode-les : plongements de classes $T\in\mathbb{R}^{K\times d}$, normalisés en L2.
3. Encode l'image de test : $u\in\mathbb{R}^{d}$, normalisée.
4. Calcule le score $s=Tu\in\mathbb{R}^K$ (similarités cosinus) et prédis $\arg\max_c s_c$.

Aucun paramètre n'est entraîné sur les classes cibles. La température ne change pas l'argmax ; elle ne compte que si tu veux des probabilités (un softmax sur $s/\tau$).

**Ingénierie des prompts.** La formulation change la précision de quelques points. Les sources notent que le travail original utilisait environ 80 modèles par classe (« a photo of a {} », « a sketch of a {} », ...). Les plongements de tous les modèles d'une classe sont moyennés, puis la moyenne est re-normalisée, pour un gain d'environ un à trois points sur ImageNet. Avec $K$ classes et $M$ modèles, le côté texte coûte $KM$ passages d'encodeur une seule fois, et les plongements de classes sont mis en cache.

**Recherche.** Plonge toutes les images une fois. Pour une requête texte, classe les images par similarité cosinus (texte-vers-image) ; pour une requête image, classe les légendes (image-vers-texte). Le **Recall@k** est la fraction des requêtes dont l'élément correct figure parmi les $k$ premiers.

**Sondes et fine-tuning.** Le zero-shot est une référence. Une sonde linéaire sur des caractéristiques d'image CLIP gelées bat généralement le zero-shot sur une tâche spécifique du domaine, et le fine-tuning complet peut battre la sonde dans le domaine mais peut nuire au transfert zero-shot vers d'autres catégories (trois régimes, trois compromis).

**À quoi ça sert.** Classification zero-shot directe ; recherche d'images ; détection et segmentation conditionnées par du texte (les détecteurs à vocabulaire ouvert enveloppent une tour de texte de style CLIP autour d'un détecteur, voir [[vis.03]] et [[vis.04]]) ; l'encodeur de vision des modèles de chat multimodaux ; le conditionnement de la génération texte-vers-image.

## Limites

- **Fermé par les données d'entraînement.** L'espace de plongement ne couvre que ce que les données récupérées sur le web décrivaient ; les sources citent un plafond zero-shot sur ImageNet d'environ 76 % pour les plus grands modèles de classe CLIP de cette époque, les modèles SigLIP 2 plus récents dépassant 80 %. Considère ces nombres comme indicatifs.
- **Sensibilité aux prompts.** Les noms de classes sont des mots ambigus (« crane », « mouse »), et les résultats dépendent de la formulation.
- **Supervision bruitée.** Les légendes sont des descriptions approximatives ; les distinctions fines, le comptage, les relations spatiales et le texte dans les images sont des points faibles courants (expérience générale avec ces modèles, non affirmée par les deux leçons sources).
- **Biais des données.** Le modèle reflète les biais de ses données web, il doit donc être audité avant toute décision concernant des personnes.
- **La similarité cosinus n'est pas une probabilité calibrée.** Ne lis pas une similarité brute de 0.3 comme « sûr à 30 % ».
- **Plongement global.** Un seul vecteur par image n'a pas de localisation ; les tâches denses demandent un mécanisme supplémentaire.

## Exemple travaillé

Tous les nombres sont reproduits par le code de l'exercice.

**Alignement parfait, $N=3$, $\tau=0.5$.** $U=V=I_3$. Alors $S=2I$ (diagonale $2$, hors-diagonale $0$). Le softmax de chaque ligne donne à la bonne légende la probabilité $e^2/(e^2+1+1)=0.787$, donc $\ell=-\log0.7869=\log(1+2e^{-2})=0.2395$. Les colonnes sont identiques par symétrie, donc $L=0.2395$.

**Deux paires, texte imparfait.** $U=I_2$, plongements de texte $V=\begin{pmatrix}0.8&0.6\\0.6&0.8\end{pmatrix}$ (lignes unitaires). Les cosinus valent $0.8$ sur la diagonale et $0.6$ en dehors.

- À $\tau=0.1$ : $S=\begin{pmatrix}8&6\\6&8\end{pmatrix}$. Le softmax par ligne est $(0.881,0.119)$, donc $\ell=-\log0.881=\log(1+e^{-2})=0.1269$ et $L=0.1269$ (les deux directions sont égales).
- À $\tau=1$ : $S=\begin{pmatrix}0.8&0.6\\0.6&0.8\end{pmatrix}$ et $L=\log(1+e^{-0.2})=0.5981$. Une température plus grande laisse une perte bien plus élevée pour les mêmes plongements.
- Gradient à $\tau=0.1$ : $P-E=\begin{pmatrix}-0.119&0.119\\0.119&-0.119\end{pmatrix}$, et avec $Q=P$ ici, $\partial L/\partial S=\frac1{2N}(2(P-E))=\begin{pmatrix}-0.0596&0.0596\\0.0596&-0.0596\end{pmatrix}$ : augmenter les logits qui correspondent, diminuer ceux qui ne correspondent pas, de façon égale.

**Zero-shot.** Les plongements de classes sont les vecteurs unitaires $e_1,e_2,e_3$ (« chat », « chien », « voiture ») ; les images $(0.9,0.1,0)$, $(0.1,0.2,0.9)$, $(0,1,0.1)$, $(2,0,0.1)$ sont classées $0,2,1,0$. Remettre à l'échelle un plongement de classe par $5$ ou $0.1$ ne change rien, car les vecteurs sont d'abord normalisés.

**Assemblage de prompts.** Deux classes avec deux plongements de modèles chacune : la classe 0 a $(1,0)$ et $(0,1)$, la classe 1 a $(1,0)$ deux fois. Les plongements moyenne-puis-normalisation sont $(0.707,0.707)$ et $(1,0)$.

**Recall@k.** Quatre images contre la matrice de texte identité, où l'image 0 vaut $(1,2,0,0)$ : son cosinus avec la légende 1 est $0.894$ et avec sa propre légende $0.447$. La bonne légende arrive en deuxième position, donc recall@1 $=3/4$ et recall@2 $=1$.

## Implémentation

```python
import numpy as np

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def softmax_rows(L):
    Z = L - L.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)

def clip_loss(I, T, tau):                      # I, T: (N, d)
    L = l2_normalize(I) @ l2_normalize(T).T / tau   # (N, N)
    N = len(L)
    i2t = -np.log(softmax_rows(L)[np.arange(N), np.arange(N)]).mean()
    t2i = -np.log(softmax_rows(L.T)[np.arange(N), np.arange(N)]).mean()
    return 0.5 * (i2t + t2i)

def zero_shot_predict(image_emb, class_emb):   # (M, d), (K, d) -> (M,)
    return np.argmax(l2_normalize(image_emb) @ l2_normalize(class_emb).T, axis=1)
```

En pratique, utilise `open_clip` (checkpoints OpenCLIP, y compris ceux entraînés sur LAION), `transformers` de Hugging Face (`CLIPModel`, `SiglipModel`) et la publication originale d'OpenAI. L'entraînement utilise un `logit_scale = 1/tau` appris, stocké comme paramètre logarithmique, la précision mixte et des pipelines de données à grande échelle ; la perte elle-même est exactement les quelques lignes ci-dessus.

## Formes

| objet | forme |
|---|---|
| lot d'images | $(N,3,H,W)$ |
| lot de légendes tokenisées | $(N,T_{\max})$ |
| plongements d'images et de texte | $(N,d)$ chacun, de norme unité après normalisation |
| matrice de logits | $(N,N)$ |
| plongements de prompts de classes | $(K,d)$ (ou $(K,M,d)$ avant assemblage) |
| scores zero-shot pour $M$ images | $(M,K)$ |

## Idées reçues

- **« CLIP a été entraîné sur des étiquettes de classes. »** Il a été entraîné uniquement sur des paires image-texte ; les classes apparaissent au moment du test sous forme de phrases.
- **« Zero-shot veut dire pas de conception de prompt. »** Le prompt est le classifieur ; la formulation compte.
- **« La température ne fait que remettre à l'échelle. »** Dans la perte, elle fixe à quel point les négatifs difficiles sont accentués et change l'optimum ; à l'inférence l'argmax n'est pas affecté.
- **« Les deux tours partagent leurs poids. »** Ce sont des réseaux séparés, reliés uniquement par la perte dans l'espace commun.
- **« Un plus grand lot, c'est juste plus rapide. »** Il fournit plus de négatifs et change ce qui est appris.
- **« Les plongements CLIP disent où sont les choses. »** Un vecteur par image ne porte aucune localisation.
- **« Le fine-tuning aide toujours. »** Il peut éroder la capacité zero-shot sur d'autres catégories.

## Points clés à retenir

- CLIP entraîne une tour d'images et une tour de texte pour que les paires correspondantes aient une forte similarité cosinus : une entropie croisée symétrique sur la matrice $N\times N$ $UV^\top/\tau$.
- Le gradient par rapport aux logits est $\frac1{2N}[(P-E)+(Q-E)]$.
- Classification zero-shot = plongement de prompt de classe le plus proche ; l'assemblage de prompts moyenne les modèles puis re-normalise.
- La recherche, la détection avec requêtes textuelles et les modèles multimodaux réutilisent le même espace.
- La température, la taille de lot, la qualité des données et les prompts sont les leviers pratiques ; les limites comprennent le biais, le détail fin et la calibration.

## Explique-le toi-même

1. En partant de « trouver la bonne légende parmi $N$ », dérive la perte de CLIP et son gradient par rapport aux logits, et explique pourquoi les deux directions sont utilisées.
2. Tu veux un classifieur pour 30 types de défauts industriels et tu n'as aucune étiquette. Décris la recette zero-shot, comment tu testerais les prompts, et quand tu passerais à une sonde linéaire.

## Pour aller plus loin

- Les deux leçons sources (références CLIP, SigLIP, SigLIP 2, ALIGN, lois d'échelle d'OpenCLIP).
- [[vis.05]] pour la famille contrastive, [[trf.04]] pour le bloc d'encodeur de texte, [[vis.02]] pour les sondes et le fine-tuning.

## Critères de maîtrise

Tu sais implémenter la perte CLIP symétrique avec une température et vérifier sa valeur au niveau du hasard et à l'alignement parfait, dériver son gradient par rapport aux logits, construire un classifieur zero-shot avec assemblage de prompts, calculer le recall@k pour la recherche, et nommer les principales limites de l'approche.
