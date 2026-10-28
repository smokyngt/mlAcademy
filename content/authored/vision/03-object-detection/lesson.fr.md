---
id: vis.03
title: "Détection d'objets : boîtes, IoU, NMS, YOLO et mAP"
summary: "Prédis un nombre variable de boîtes étiquetées par image, dédoublonne-les avec la suppression des non-maxima, et note un détecteur avec la précision moyenne, le tout implémenté à partir de zéro."
minutes: 80
teaches: [vis.iou, vis.detection, vis.map]
requires: [vis.image-tensor, vis.classification, vis.transfer]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/06-object-detection-yolo/docs/en.md (Sec. 'The Problem' and 'The Concept': dense prediction, grid and anchors, decoding, IoU, NMS, loss structure, detection metrics; the Build It code was not read). READ."
---

## Où ça s'inscrit

La classification dit « c'est un chien ». La détection dit « un chien en pixels (112, 40)-(280, 210), un chat en (400, 180)-(560, 310), et rien d'autre ». La conduite autonome, l'analyse du commerce de détail, le dépistage médical et le contrôle industriel ont tous besoin de cela, car le nombre d'objets varie et leurs positions comptent. La détection combine tout ce qui précède : les tenseurs d'images ([[vis.01]]), la classification et l'apprentissage par transfert ([[vis.02]]), et un backbone CNN ([[seq.01]]). Elle mène à la segmentation dans [[vis.04]], où la boîte devient un masque par pixel.

## L'idée en termes simples

Imagine que tu poses une feuille de papier quadrillé sur la photo. On demande à chaque case : « le centre d'un objet est-il en toi ? Si oui, à quelle distance de ton milieu, quelle taille a-t-il, et qu'est-ce que c'est ? » Un seul coup d'œil à l'image répond à toutes les cases en parallèle, c'est pourquoi les détecteurs en un seul passage (YOLO, « You Only Look Once ») sont rapides.

Deux problèmes pratiques apparaissent. D'abord, une case ne doit pas inventer des tailles de boîte à partir de rien ; on donne donc à chaque case quelques formes modèles (**ancres** : une haute, une large, une grande) et elle ne prédit qu'un petit ajustement par rapport au modèle. Ensuite, des cases voisines vont souvent toutes crier à propos du même chien, produisant un tas de boîtes qui se chevauchent. La **suppression des non-maxima** (NMS) nettoie cela : garder la boîte la plus confiante et écarter tout ce qui la chevauche trop, puis recommencer.

Pour noter le résultat, tu ne peux pas compter des « images justes et fausses ». Tu demandes, sur toutes les détections classées de la plus à la moins confiante, combien sont de vrais objets (précision) et combien des vrais objets ont été trouvés (rappel), et tu résumes ce compromis en un nombre, la **précision moyenne**.

## Boîtes et intersection sur union

Une boîte est `(x1, y1, x2, y2)` avec `x1 < x2`, `y1 < y2` en pixels (d'autres conventions stockent le centre et la taille, `(cx, cy, w, h)` ; sache laquelle ta bibliothèque utilise). L'aire d'une boîte est $(x_2-x_1)(y_2-y_1)$.

La similarité entre deux boîtes $A$ et $B$ est l'**intersection sur union** (IoU) :

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

L'intersection de deux boîtes alignées sur les axes est elle-même une boîte de largeur $\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B))$ et de hauteur calculée de la même façon. L'IoU vaut $1$ pour des boîtes identiques et $0$ pour des boîtes disjointes (ou simplement tangentes). Une prédiction est correcte quand son IoU avec une boîte de vérité terrain dépasse un seuil, couramment $0.5$ ; la NMS utilise l'IoU entre deux prédictions.

## Détection en un seul passage : le tenseur de sortie

Un backbone associe à l'image une carte de caractéristiques avec une grille $S\times S$ (pas de $s$ pixels par cellule, donc $S=\text{input size}/s$). Une petite tête de convolutions $1\times1$ prédit, pour chacune des $B$ ancres de chaque cellule, $5+C$ nombres :

- $t_x,t_y,t_w,t_h$ : décalages de la boîte,
- un score d'objectité : « un objet est-il centré ici ? »,
- $C$ scores de classe.

Pour le cadre classique $S=13$, $B=2$, $C=20$ (20 classes), cela fait $2\cdot25=50$ nombres par cellule et $13\cdot13\cdot50=8450$ par image. La sortie est un tenseur de forme $(N,\,S,\,S,\,B(5+C))$ ou, de façon équivalente, $(N,B(5+C),S,S)$.

### Pourquoi des ancres, et comment les boîtes sont décodées

Une convolution $3\times3$ sur un petit champ réceptif ne peut pas facilement régresser à partir de rien une boîte large de plusieurs centaines de pixels, et le même objet apparaissant ailleurs dans l'image devrait produire la même sortie *relative*. Le réseau prédit donc des décalages par rapport à des formes a priori fixes $(a_w,a_h)$ et relativement à la position de la cellule $(c_x,c_y)$ (coordonnées entières de la grille) :

$$
\begin{aligned}
b_x&=(\sigma(t_x)+c_x)\,s, & b_y&=(\sigma(t_y)+c_y)\,s,\\
b_w&=a_w\,e^{t_w}, & b_h&=a_h\,e^{t_h}.
\end{aligned}
$$

Pourquoi ces fonctions : la sigmoïde $\sigma$ garde le centre à l'intérieur de sa propre cellule (chaque cellule n'est donc responsable que des objets centrés en elle) ; l'exponentielle fait de la largeur un multiple positif de l'ancre, $t_w=0$ signifiant « l'ancre convient » et $t_w=\ln2$ signifiant « deux fois plus large ». Les coins de la boîte sont $(b_x\mp b_w/2,\ b_y\mp b_h/2)$. Les détecteurs modernes utilisent des réseaux à pyramide de caractéristiques avec de petites ancres sur les cartes haute résolution et de grandes ancres sur les cartes grossières ; le principe est inchangé.

### La perte

Le signal d'entraînement a trois parties, chacune avec un poids $\lambda$ :

$$
L=\lambda_{\text{box}}\!\!\sum_{\text{obj cells}}\!\!L_{\text{box}}
+\lambda_{\text{obj}}\!\!\sum_{\text{obj cells}}\!\!L_{\text{obj}}(\hat o,1)
+\lambda_{\text{noobj}}\!\!\sum_{\text{empty cells}}\!\!L_{\text{obj}}(\hat o,0)
+\lambda_{\text{cls}}\!\!\sum_{\text{obj cells}}\!\!L_{\text{cls}} .
$$

Chaque boîte de vérité terrain est affectée à la cellule (et à l'ancre la mieux adaptée) qui contient son centre. Les pertes de boîte et de classe ne sont calculées que pour ces emplacements affectés. Tous les autres emplacements sont entraînés uniquement à produire un faible score d'objectité. Les emplacements vides sont de loin plus nombreux que les emplacements d'objets, donc $\lambda_{\text{noobj}}$ est petit (la source mentionne environ $0.5$) ou la perte est rééquilibrée (par exemple avec la focal loss), sinon « ne rien prédire » domine. La régression de boîte par erreur quadratique est souvent remplacée par des pertes basées sur l'IoU (GIoU, DIoU, CIoU) qui optimisent directement le recouvrement.

## Suppression des non-maxima

Entrée : des boîtes avec des scores. Sortie : le sous-ensemble à conserver.

```text
sort by score, descending
keep = []
while boxes remain:
    take the best box b, append it to keep
    delete every remaining box with IoU(b, box) > threshold
```

Le seuil arbitre entre deux erreurs. Trop bas (disons 0.2), deux objets réellement différents et proches (une foule) sont fusionnés, donc tu perds de vraies détections. Trop haut (disons 0.9), les doublons survivent. Une valeur typique est d'environ 0.45 à 0.5. Il existe des variantes (soft-NMS, DIoU-NMS) et des détecteurs de bout en bout qui apprennent la suppression, mais le but est le même : une boîte par objet.

## Métriques de détection

Pour une classe, prends toutes les détections de l'ensemble d'évaluation et trie-les par confiance, la plus haute d'abord. Parcours la liste.

**Appariement.** Une détection est un **vrai positif** (TP) si sa boîte de vérité terrain d'IoU maximal a $\mathrm{IoU}\ge\tau$ et que cette vérité terrain n'a pas déjà été revendiquée par une détection de score plus élevé ; sinon c'est un **faux positif** (FP). Une seconde détection d'un objet déjà trouvé est donc un faux positif, ce qui explique l'importance de la NMS. Les vérités terrain jamais appariées sont des faux négatifs.

**Précision et rappel au rang $r$.** Avec $\mathrm{TP}_r$ vrais positifs parmi les $r$ premières détections et $G$ boîtes de vérité terrain :

$$
P_r=\frac{\mathrm{TP}_r}{r},\qquad R_r=\frac{\mathrm{TP}_r}{G}.
$$

**La précision moyenne** est l'aire sous la courbe précision-rappel. Pour supprimer le zigzag, rends la précision non croissante à partir de la droite (l'*enveloppe*) : $\tilde P_r=\max_{r'\ge r}P_{r'}$. Alors

$$
\mathrm{AP}=\sum_r (R_r-R_{r-1})\,\tilde P_r,\qquad R_0=0 .
$$

Chaque incrément de rappel est pondéré par la meilleure précision atteignable à ce rappel ou au-delà.

**mAP.** Moyenne des AP sur les classes. La **mAP@[.5:.95]** de style COCO fait aussi la moyenne sur les seuils d'IoU $\tau\in\{0.5,0.55,\dots,0.95\}$, ce qui récompense les boîtes serrées. Lecture des chiffres : une AP@0.5 élevée mais une AP@[.5:.95] faible signifie que les objets sont trouvés mais localisés de façon approximative (améliore la perte de boîte) ; une précision élevée mais un rappel faible signifie que le détecteur est trop timide (baisse le seuil de score, augmente le poids d'objectité).

## Exemple travaillé

Tous les nombres sont reproduits par le code de l'exercice.

**IoU.** Boîtes $A=(0,0,10,10)$ et $B=(5,5,15,15)$. Intersection $5\times5=25$, union $100+100-25=175$, IoU $=25/175=0.143$. Pour $A=(0,0,10,10)$ contre $(1,1,11,11)$ l'IoU vaut $81/119=0.681$, et contre $(0,0,10,9)$ elle vaut $0.9$.

**NMS.** Boîtes : $b_0=(0,0,10,10)$ score 0.9 ; $b_1=(1,1,11,11)$ score 0.8 ; $b_2=(50,50,60,60)$ score 0.7 ; $b_3=(0,0,10,9)$ score 0.6. Seuil 0.5. On garde $b_0$. IoU($b_0,b_1$)$=0.681>0.5$ donc on écarte $b_1$ ; IoU($b_0,b_2$)$=0$ on continue ; IoU($b_0,b_3$)$=0.9$ donc on écarte $b_3$. La meilleure suivante est $b_2$ ; il ne reste plus rien. Résultat : $[0,2]$.

**Décodage.** Cellule $(c_x,c_y)=(3,2)$, pas 32, ancre $(60,40)$, tous les décalages nuls. La sigmoïde de 0 vaut 0.5, donc le centre est $((0.5+3)\cdot32,\ (0.5+2)\cdot32)=(112,80)$, la taille égale l'ancre, et la boîte est $(82,60,142,100)$. Poser $t_w=\ln2$ double la largeur.

**Précision moyenne.** Avec 4 boîtes de vérité terrain et les indicateurs classés $[1,1,0,1,0]$ (TP, TP, FP, TP, FP) : précision $=[1,1,0.667,0.75,0.6]$, rappel $=[0.25,0.5,0.5,0.75,0.75]$, enveloppe $=[1,1,0.75,0.75,0.6]$. $\mathrm{AP}=0.25\cdot1+0.25\cdot1+0\cdot0.75+0.25\cdot0.75+0\cdot0.6=0.6875$.

**Appariement de bout en bout.** Vérités terrain `(0,0,10,10)`, `(20,20,30,30)`, `(50,50,60,60)`, `(70,70,80,80)` ; détections avec les scores 0.95 `(0,0,10,10)` (IoU 1), 0.9 `(21,21,31,31)` (IoU 0.68), 0.8 `(100,100,110,110)` (très loin), 0.7 `(0,1,10,11)` (IoU 0.82 avec la première vérité terrain, déjà revendiquée), 0.6 `(50,50,59,59)` (IoU 0.81). Indicateurs à $\tau=0.5$ : $[1,1,0,0,1]$, donnant AP $=0.25+0.25+0.25\cdot0.6=0.65$. À $\tau=0.9$ seule la première détection survit. Remarque que le doublon (score 0.7) est un faux positif même s'il chevauche bien, ce qui est la raison pour laquelle les détecteurs appliquent la NMS.

## Implémentation

```python
import numpy as np

def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def nms(boxes, scores, thr):
    order = list(np.argsort(-np.asarray(scores, dtype=float), kind="stable"))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep

def average_precision(tp_flags, n_gt):
    tp = np.cumsum(tp_flags)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / n_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * envelope))
```

Le code de production fait cela sur des tenseurs : `torchvision.ops.nms` et `batched_nms` (par classe), et l'évaluation avec les outils COCO (`pycocotools`) ou la bibliothèque fournie avec ton détecteur (par exemple Ultralytics, les références de détection de torchvision). Note que les benchmarks officiels diffèrent par des détails comme l'interpolation (tous les points contre 101 points) et le traitement des régions de foule, donc une AP écrite à la main ne leur correspondra pas exactement.

## Formes

| objet | forme |
|---|---|
| carte de caractéristiques du backbone | $(N,C_f,S,S)$ |
| sortie brute de la tête | $(N,B(5+C),S,S)$ |
| prédictions décodées | $(N,\,S\cdot S\cdot B,\,5+C)$ : boîte, objectité, classes |
| boîtes, scores | $(M,4)$, $(M,)$ |
| matrice d'IoU par paires | $(M_A,M_B)$ |
| indices conservés après NMS | $(K,)$ avec $K\le M$ |

## Idées reçues

- **« La précision de détection est la fraction de boîtes correctes. »** Il faut une précision et un rappel classés, car il n'y a pas d'ensemble fixe de négatifs et le seuil de score est un paramètre libre.
- **« La NMS fait partie du réseau. »** C'est un post-traitement, généralement exécuté par classe ; il n'est pas dérivable et n'est appliqué qu'à l'inférence.
- **« Un seuil d'IoU plus élevé dans la NMS supprime davantage. »** Plus élevé conserve davantage ; cela ne supprime que les boîtes qui se chevauchent plus que le seuil.
- **« L'IoU est une distance. »** C'est une similarité, et pour des boîtes disjointes elle vaut 0 quelle que soit leur distance (raison d'être des pertes GIoU et DIoU).
- **« mAP@0.5 suffit. »** Elle cache une localisation approximative ; regarde la métrique moyennée plus stricte.
- **« Le réseau prédit directement des coordonnées en pixels. »** Il prédit des décalages par rapport à une cellule et une ancre, décodés avec sigmoïde et exponentielle.

## Points clés à retenir

- Détection = classification + régression de boîte à chaque position de la grille et pour chaque ancre, plus l'objectité, suivies de la NMS.
- $\mathrm{IoU}=|A\cap B|/|A\cup B|$ ; elle définit à la fois « correct » et « doublon ».
- Décoder avec $b_x=(\sigma(t_x)+c_x)s$ et $b_w=a_we^{t_w}$.
- L'AP est l'aire, corrigée par l'enveloppe, sous la courbe précision-rappel des détections classées ; la mAP fait la moyenne sur les classes (et sur les seuils d'IoU pour COCO).
- Le terme « pas d'objet » doit être sous-pondéré ou rééquilibré.

## Explique-le toi-même

1. Pourquoi une seconde détection bien placée du même objet compte-t-elle comme un faux positif, et quel composant du système la supprime ?
2. Dérive l'IoU de deux boîtes à partir de leurs coordonnées de coins, et dis ce qui arrive à sa valeur quand elles ne se touchent que le long d'un bord.

## Pour aller plus loin

- La leçon source, et les articles YOLO (Redmon et al.) pour la conception d'origine.
- [[vis.04]] pour les sorties par pixel, [[vis.05]] pour les backbones à transformers qui remplacent le CNN, et [[cml.10]] pour la précision et le rappel en général.

## Critères de maîtrise

Tu sais calculer à la main l'IoU et la NMS, décoder une cellule de type YOLO, expliquer chaque terme de la perte de détection, et calculer l'AP à partir d'une liste classée de détections et de vérités terrain.
