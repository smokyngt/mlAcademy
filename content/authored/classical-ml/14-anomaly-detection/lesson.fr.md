---
id: cml.14
title: "Détection d'anomalies : z-scores, distance de Mahalanobis et forêts d'isolation"
summary: "Modéliser ce qui est normal plutôt que classifier ce qui est rare, du z-score unidimensionnel à la distance de Mahalanobis puis à une forêt d'isolation que tu construis toi-même, et apprendre à juger des détecteurs quand on n'a presque pas d'étiquettes."
minutes: 70
teaches: [ml.anomaly, ml.iforest]
requires: [prob.rv, la.norms, ml.pca, ml.eval-metrics]
sources:
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/16-anomaly-detection/docs/en.md (anomaly types, unsupervised framing, z-score/IQR, isolation forest, LOF, evaluation challenges, pipeline). READ."
---

## Où cela s'inscrit

Une carte utilisée dans deux pays à cinq minutes d'intervalle, une mesure de capteur très en dehors de sa plage habituelle, un serveur qui répond soudain à mille fois plus de requêtes. Ces événements sont rares, coûteux, et différents à chaque fois. Tu ne peux pas rassembler des milliers d'exemples étiquetés de « la fraude de demain », donc les classifieurs de [[cml.02]] et [[cml.10]] ont peu de choses à apprendre. La réponse standard consiste à inverser le problème : **apprendre à quoi ressemble le normal et signaler tout ce qui ne colle pas**.

Cette leçon construit une échelle de détecteurs. Un z-score travaille sur un seul nombre ; la distance de Mahalanobis gère des variables corrélées (elle utilise la matrice de covariance de [[cml.05]]) ; une forêt d'isolation ne fait aucune hypothèse de distribution ; le facteur local d'anomalie (LOF) s'adapte à des régions de densités différentes. Vient ensuite la partie que la plupart des tutoriels sautent : comment évaluer un détecteur quand les étiquettes sont rares ou absentes, et ce que le mot « contamination » signifie vraiment.

## L'idée en mots simples

Imagine une foule dans une gare. Si tu dois trouver une personne à qui remettre un message, et que tu poses des questions oui/non au hasard (« la personne est-elle à gauche de ce pilier ? »), alors quelqu'un qui se tient loin de tout le monde est isolé après quelques questions, tandis que quelqu'un au milieu de la foule en demande bien plus. La **forêt d'isolation** est exactement ce jeu : les points les plus faciles à isoler par des coupes aléatoires sont les suspects.

Les autres détecteurs formulent la même intuition autrement :

- **z-score** : « combien d'écarts typiques cette valeur est-elle éloignée de la moyenne ? »
- **Distance de Mahalanobis** : pareil, mais la notion d'« écart typique » est une ellipse inclinée qui suit la façon dont les variables évoluent ensemble. Une personne de taille moyenne et de poids moyen mais avec une *combinaison* impossible est détectée ici et manquée par les z-scores variable par variable.
- **Facteur local d'anomalie (LOF)** : « ce point est-il plus isolé que ses propres voisins ? » Un point ordinaire dans une région clairsemée est normal ; la même distance par rapport à un amas dense ne l'est pas.

Trois types d'anomalies comptent : les anomalies **ponctuelles** (une valeur est étrange), **contextuelles** (90 degrés est étrange en hiver, pas en été : il faut des variables de temps ou de lieu) et **collectives** (cinquante échecs de connexion d'affilée, chaque événement pris isolément étant inoffensif). La plupart des détecteurs ci-dessous trouvent des anomalies ponctuelles ; les contextuelles et collectives demandent des variables qui portent le contexte (statistiques glissantes, décalages comme dans [[cml.13]]).

## Le z-score et son mode de défaillance

Pour un échantillon unidimensionnel de moyenne $\mu$ et d'écart-type $\sigma$, le **z-score** de $x$ est $z=(x-\mu)/\sigma$, et une règle courante signale $|z|>3$. La faiblesse est le **masquage** : la valeur aberrante elle-même gonfle $\mu$ et $\sigma$ et se cache du test censé l'attraper. Le **z-score robuste** remplace la moyenne par la médiane et l'écart-type par la **MAD** (écart absolu médian, $\operatorname{MAD}=\operatorname{median}|x_i-\tilde x|$) :

$$
z^{\text{rob}}_i=0.6745\,\frac{x_i-\tilde x}{\operatorname{MAD}} .
$$

La constante 0.6745 rend la MAD comparable à $\sigma$ pour des données gaussiennes (c'est le 75e centile de la loi normale standard). La médiane et la MAD bougent à peine quand quelques points extrêmes sont présents.

## Distance de Mahalanobis

Pour $X\in\mathbb R^{N\times d}$ de moyenne $\mu$ et de covariance $\Sigma\in\mathbb R^{d\times d}$ (symétrique définie positive), la **distance de Mahalanobis** d'un point $x$ est

$$
D(x)=\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)} .
$$

### Pourquoi c'est la bonne règle

Écris $\Sigma=LL^\top$ (Cholesky, ou via la décomposition en valeurs propres $\Sigma=Q\Lambda Q^\top$, voir [[cml.05]]). Définis $u=L^{-1}(x-\mu)$. Cela **blanchit** les données : $\operatorname{Cov}(u)=L^{-1}\Sigma L^{-\top}=I$, donc $u$ a des coordonnées non corrélées et de variance unité. Alors

$$
\|u\|^2=(x-\mu)^\top L^{-\top}L^{-1}(x-\mu)=(x-\mu)^\top\Sigma^{-1}(x-\mu)=D(x)^2 .
$$

$D$ est donc simplement la distance euclidienne ordinaire après blanchiment, c'est-à-dire un z-score généralisé aux variables corrélées. Si les données sont gaussiennes, $D^2$ suit une loi du khi-deux à $d$ degrés de liberté et de moyenne $d$, ce qui donne un seuil fondé en principe (par exemple le 99e centile de $\chi^2_d$). Sur l'échantillon lui-même, la moyenne de $D^2$ vaut exactement $d\,(N-1)/N$ quand $\Sigma$ est estimée avec le diviseur $N-1$, un contrôle de cohérence pratique.

Faiblesse : la moyenne et la covariance sont elles-mêmes contaminées par les valeurs aberrantes (même problème de masquage), la covariance ne peut pas être inversée quand les variables sont colinéaires ou que $d>N$, et tout suppose un seul amas elliptique.

## Forêt d'isolation

Construis de nombreux **arbres d'isolation**. Chaque arbre utilise un petit sous-échantillon aléatoire de $\psi$ lignes (256 est une valeur par défaut typique). Un nœud choisit une variable au hasard et un seuil au hasard, uniformément entre le minimum et le maximum de cette variable dans le nœud, puis coupe. On s'arrête à la profondeur $\lceil\log_2\psi\rceil$ ou quand il ne reste qu'un point. La **longueur de chemin** $h(x)$ est le nombre de coupes nécessaires pour atteindre la feuille contenant $x$.

Les anomalies sont rares et différentes, donc des coupes aléatoires les séparent tôt : chemin moyen court. Pour transformer la longueur de chemin en score comparable, il faut savoir ce qui est « typique ». Un arbre construit sur $n$ points ressemble à un arbre binaire de recherche, dont la longueur moyenne de chemin pour une recherche infructueuse est

$$
c(n)=2\big(\ln(n-1)+\gamma\big)-\frac{2(n-1)}{n},\qquad \gamma\approx0.5772,
$$

avec $c(2)=1$ et $c(n)=0$ pour $n\le1$. Quand une feuille contient encore $s>1$ points (limite de profondeur atteinte), on ajoute $c(s)$ comme estimation des coupes qui auraient été nécessaires. Le **score d'anomalie** est

$$
s(x)=2^{-\,\mathbb E[h(x)]/c(\psi)} ,
$$

donc $E[h]=c(\psi)$ donne $0.5$ (rien de particulier), des chemins très courts donnent des scores proches de $1$, des chemins très longs donnent des scores bien inférieurs à $0.5$. Points forts : aucune hypothèse de distribution, un coût qui croît avec la taille du sous-échantillon et non du jeu de données, fonctionne avec beaucoup de variables. Points faibles : les coupes parallèles aux axes peinent avec les structures diagonales, et un amas d'anomalies peut se masquer mutuellement.

## Facteur local d'anomalie (idée seulement)

Pour chaque point, trouve ses $k$ plus proches voisins et définis une densité locale (à peu près l'inverse de la distance moyenne à ces voisins). Le LOF est le rapport des densités des voisins à celle du point : environ $1$ signifie « aussi dense que mon voisinage », nettement au-dessus de $1$ signifie « beaucoup plus clairsemé que les points autour de moi ». Il trouve des anomalies situées à côté d'un amas dense qu'un seuil global manquerait, au prix de calculs de distances en $O(N^2)$, d'une sensibilité à $k$ et du fléau de la dimension pour les distances ([[cml.09]]).

## Évaluation sans (beaucoup d')étiquettes

- **La précision (accuracy) est inutile** avec 0.1 % d'anomalies : prédire « normal » donne 99.9 %.
- Avec quelques étiquettes, classe par score et rapporte la **precision@k** (parmi les $k$ éléments les mieux notés, combien sont de vraies anomalies) et l'aire sous la courbe précision-rappel ([[cml.10]]) ; l'AUC-ROC peut paraître trop belle sous fort déséquilibre.
- Sans étiquettes : **injecte des anomalies synthétiques** (valeurs décalées ou permutées) et vérifie qu'elles sont retrouvées ; vérifie la **stabilité** (deux graines, ou deux sous-échantillons, signalent-ils les mêmes points ?) ; compare les détecteurs par **accord de rang** ; et surtout, envoie le haut de la liste à un humain et mesure le taux de réussite. Les étiquettes issues de cette boucle de retour sont ton futur jeu d'évaluation.
- Évalue sur des données **postérieures** à la période d'entraînement (walk-forward, [[cml.13]]) : la définition du normal dérive.

## Contamination

Beaucoup de bibliothèques prennent un paramètre `contamination` : la fraction supposée d'anomalies. Il ne **change pas** les scores. Il ne fait que placer le seuil : signaler les $\lceil \text{contamination}\cdot N\rceil$ meilleurs scores. C'est donc en réalité une décision sur le nombre d'alertes que tu peux te permettre de vérifier. Choisis-le d'après le coût d'une fausse alarme par rapport à un événement manqué, pas d'après une connaissance illusoire du vrai taux. Ajuster un détecteur sur des données qui contiennent déjà des anomalies (la situation habituelle) rend moins fiables les méthodes fondées sur les moyennes et covariances, ce qui motive les statistiques robustes et le sous-échantillonnage utilisé par les forêts d'isolation.

## Exemple détaillé

Les dix nombres $(10,11,9,10,12,10,11,9,10,50)$ ont pour moyenne $14.2$ et pour écart-type $11.96$, donc la valeur aberrante évidente 50 a $z=2.99<3$ : **masquée**. La médiane est $10$ et la MAD est $1$, donc son z-score robuste est $0.6745\cdot40=26.98$.

Pour un nuage de variances unité et de corrélation $0.9$, le point $(1,1)$ a une distance de Mahalanobis de $1.03$ (il suit la corrélation), tandis que $(1,-1)$ a une distance de $4.47$ bien que chaque coordonnée prise seule soit parfaitement ordinaire ($|z|=1$).

Pour $\psi=256$, $c(256)=10.2448$. Sur 200 points normaux standards en 2D plus trois points plantés $(6,6),(-5,5),(0,7)$, une forêt de 100 arbres avec $\psi=64$ (graine 0) donne aux trois points plantés les trois scores les plus élevés (environ 0.78, 0.75, 0.69) alors que le score médian est d'environ 0.43. L'exercice reproduit ces valeurs.

## Implémentation

```python
import numpy as np

def mahalanobis(X):
    Xc = X - X.mean(axis=0)
    Sinv = np.linalg.inv(np.cov(X, rowvar=False))
    return np.sqrt(np.einsum("ij,jk,ik->i", Xc, Sinv, Xc))

def c_factor(n):
    if n <= 1: return 0.0
    if n == 2: return 1.0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
```

L'exercice ajoute une `IsolationForest` complète (arbres aléatoires sur des sous-échantillons, score $2^{-E[h]/c(\psi)}$). Le code de production (`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`, `sklearn.covariance.MinCovDet` pour une covariance robuste) ajoute vitesse et parallélisme.

## Dimensions

`X: (N, d)`, `Xc: (N, d)`, `Sigma: (d, d)`, distances `(N,)`. Une forêt contient `n_trees` arbres, chacun entraîné sur `(psi, d)` lignes ; `score(X)` renvoie `(N,)`.

## Idées reçues

- **« Anomalie signifie erreur. »** Une anomalie est simplement rare ; ce peut être l'événement le plus précieux des données.
- **« La contamination change le modèle. »** Elle ne fixe que le seuil.
- **« |z|>3 est une règle sûre. »** Elle est masquée par les valeurs aberrantes elles-mêmes et suppose une forme à peu près gaussienne.
- **« Un détecteur convient à tout. »** Choisis selon les données : Mahalanobis pour un amas elliptique unique, forêt d'isolation pour des données tabulaires génériques, LOF pour des densités variables.
- **« Une AUC élevée prouve que ça marche. »** Avec 0.1 % d'anomalies, regarde la précision au nombre d'alertes que tu peux réellement vérifier.

## Points clés

- Modélise la normalité, classe par un score, puis choisis un seuil comme une décision métier.
- Les statistiques robustes (médiane, MAD) déjouent le masquage ; la distance de Mahalanobis est un z-score blanchi.
- Les forêts d'isolation notent par la longueur moyenne de chemin de coupes aléatoires, normalisée par $c(\psi)$.
- Évalue avec la precision@k, des injections synthétiques, la stabilité et le retour humain, sur des données plus tardives.

## Explique-le toi-même

1. Pourquoi les points faciles à isoler par des coupes aléatoires méritent-ils un score d'anomalie élevé ?
2. Montre, par le blanchiment, pourquoi la distance de Mahalanobis au carré de données gaussiennes a pour moyenne $d$.

## Pour aller plus loin

- La leçon source `phases/02-ml-fundamentals/16-anomaly-detection` pour les barrières IQR, le SVM à une classe et les détecteurs d'ensemble.
- Le guide utilisateur de scikit-learn, « Novelty and outlier detection ».

## Critères de maîtrise

Tu sais expliquer le masquage et le corriger avec des statistiques robustes, dériver la distance de Mahalanobis par blanchiment, implémenter une forêt d'isolation et la normalisation de son score, et concevoir une évaluation pour un détecteur disposant de très peu d'étiquettes.
