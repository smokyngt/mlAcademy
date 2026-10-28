---
id: cml.09
title: "k plus proches voisins et Bayes naïf"
summary: "Deux classifieurs opposés, l'un qui mémorise les données et vote parmi les voisins, l'autre qui modélise comment chaque classe engendre ses variables sous une hypothèse d'indépendance audacieuse ; implémenter les deux et apprendre quand chacun échoue."
minutes: 70
teaches: [ml.knn, ml.naive-bayes]
requires: [prob.basics, prob.rv, ml.generalization, la.dot]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 2.5.2 Nearest-neighbour methods (PDF p.144-147): K-NN density estimate, derivation of the K-NN classifier via Bayes' theorem, smoothing role of K, Cover-Hart bound, storage and search cost. READ."
  - "Bishop, PRML, Sec. 4.2 Probabilistic Generative Models, incl. 4.2.1 and 4.2.3 (PDF p.216-223): Bayes' theorem for classification, sigmoid/softmax of log-odds, Gaussian class-conditionals, naive Bayes assumption for discrete features. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/06-knn-and-distances/docs/en.md (choice of K, curse of dimensionality). READ (those sections)."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/14-naive-bayes/docs/en.md (independence assumption, variants, Laplace smoothing, log-space computation). READ (concept sections)."
---

## Où cela s'inscrit

Deux réponses très différentes à « comment classer un nouvel exemple à partir d'exemples étiquetés ? »

- **k plus proches voisins (kNN)** : ne fais aucun entraînement. Garde les données et, pour un nouveau point, regarde ses voisins étiquetés les plus proches et laisse-les voter. C'est la référence pour les recommandations (« des utilisateurs comme toi »), les contrôles d'anomalies et la recherche d'éléments similaires (la même idée que la recherche vectorielle sur des embeddings).
- **Bayes naïf** : construis une minuscule histoire probabiliste de la façon dont chaque classe produit ses variables, puis inverse-la avec le théorème de Bayes. Il a propulsé les premiers filtres anti-spam pratiques car il s'entraîne en un seul passage, a besoin de peu de données, gère des milliers de variables de mots et reste une référence honorable pour le texte.

Ensemble, ils montrent les deux familles de base : **non paramétrique / à base d'instances** contre **générative / paramétrique**, et le compromis entre flexibilité (kNN) et hypothèses fortes (Bayes naïf) que tu as rencontré dans [[cml.03]].

## L'idée en mots simples

**kNN.** Tu emménages dans une nouvelle ville et tu te demandes si un restaurant est bon. Tu interroges les cinq personnes qui habitent le plus près. Ces voisins sont ta prédiction. Deux choix comptent : *ce qui compte comme proche* (la distance, et donc les unités !) et *combien de voisins* interroger. Demande à une seule personne et un avis excentrique décide ; demande à toute la ville et tu entends toujours l'avis majoritaire. Le nombre $k$ est donc un cadran de lissage. Et « proche » s'effondre quand les points vivent en de nombreuses dimensions : avec des centaines de variables, presque tout le monde est à peu près à la même distance.

**Bayes naïf.** Un employé du courrier a appris que le spam contient souvent « gratuit » et « gagne » et rarement « réunion ». Pour un nouvel e-mail, il évalue à quel point chaque mot est typique du spam par rapport au courrier ordinaire et combine les indices, comme si chaque mot était un témoin indépendant. Cette indépendance est évidemment fausse (les mots viennent en expressions), d'où le « naïf », mais elle rend la comptabilité triviale : une petite table de fréquences de mots par classe au lieu d'une table impossible sur toutes les combinaisons de mots. Il se trompe souvent sur les *probabilités* mais a raison sur *quelle classe l'emporte*.

## kNN : définition et dérivation

Jeu d'entraînement $\{(x_n,y_n)\}$, une distance $d$. Pour une requête $x$ : trouve les $k$ points d'entraînement de plus petite $d(x,x_n)$ et prédis la classe majoritaire (ou, en régression, la moyenne de leurs cibles). Distance courante : euclidienne $d(x,z)=\|x-z\|_2$.

**D'où vient le « vote majoritaire »** (Bishop Sec. 2.5.2). Fais grandir une sphère autour de $x$ jusqu'à ce qu'elle contienne exactement $K$ points d'entraînement ; appelle $V$ son volume, et soit $K_c$ le nombre de ces points appartenant à la classe $c$ (la classe $c$ a $N_c$ points sur $N$ au total). Une estimation locale de densité est « le nombre divisé par (nombre de points fois volume) » :

$$
p(x\mid c)\approx\frac{K_c}{N_cV},\qquad p(x)\approx\frac{K}{NV},\qquad p(c)=\frac{N_c}{N}.
$$

Théorème de Bayes :

$$
p(c\mid x)=\frac{p(x\mid c)p(c)}{p(x)}=\frac{\frac{K_c}{N_cV}\cdot\frac{N_c}{N}}{\frac{K}{NV}}=\frac{K_c}{K}.
$$

Tout se simplifie : la probabilité a posteriori est la fraction des $K$ voisins qui appartiennent à la classe $c$, et choisir la plus grande probabilité a posteriori est le vote majoritaire. Bishop ajoute un résultat classique : quand $N\to\infty$, la règle du plus proche voisin (1-NN) a une erreur au plus double de la meilleure erreur atteignable (de Bayes) (Cover et Hart).

**Choisir $k$.** $k=1$ : la frontière suit chaque point, erreur d'entraînement nulle, variance élevée. Grand $k$ : plus lisse, plus de biais ; $k=N$ prédit la majorité globale. Choisis $k$ (et la distance) par l'erreur de validation ([[cml.03]]) ; un $k$ impair évite les égalités dans les problèmes binaires.

**La mise à l'échelle des variables est obligatoire.** Les distances additionnent les différences au carré des colonnes brutes, donc une variable mesurée en dollars écrase une variable mesurée en années. Standardise chaque colonne ($z=(x-\mu)/\sigma$) avec les statistiques du *jeu d'entraînement*, et applique la même transformation aux requêtes.

**Le fléau de la dimension.** En grande dimension, les points les plus proches et les plus éloignés deviennent presque équidistants, donc « le plus proche » porte peu d'information. Une mesure rapide est le contraste relatif $(d_{\max}-d_{\min})/d_{\min}$ d'une requête à un nuage aléatoire. De plus, pour capturer une part fixe des données, ton voisinage doit couvrir l'essentiel de la plage de chaque variable, il n'est donc plus « local ». Remèdes : moins de variables mais meilleures (ACP, [[cml.05]]), sélection de variables, ou embeddings appris. Coût : le kNN stocke toutes les données et, naïvement, calcule $N$ distances par requête ($O(Nd)$), donc la prédiction est lente pour un grand $N$ ; les index à base d'arbres n'aident qu'en dimension faible à modérée.

## Bayes naïf : définition et dérivation

Classification générative (Bishop Sec. 4.2) : modélise $p(x\mid C_k)$ et $p(C_k)$, puis

$$
p(C_k\mid x)=\frac{p(x\mid C_k)\,p(C_k)}{\sum_jp(x\mid C_j)p(C_j)}\ \propto\ p(C_k)\,p(x\mid C_k).
$$

Pour $d$ variables, la loi jointe $p(x\mid C_k)$ est une énorme table (pour $d$ variables binaires, $2^d-1$ nombres par classe ; Bishop p.202). L'**hypothèse de Bayes naïf** est l'indépendance conditionnelle des variables sachant la classe :

$$
p(x\mid C_k)=\prod_{i=1}^dp(x_i\mid C_k),
$$

qui ne demande que $d$ lois unidimensionnelles par classe. Pour éviter de multiplier des centaines de tout petits nombres (dépassement par le bas, underflow), travaille avec des logarithmes :

$$
\log p(C_k\mid x)=\text{const}+\log p(C_k)+\sum_{i=1}^d\log p(x_i\mid C_k),\qquad \hat y=\arg\max_k\ [\ldots].
$$

L'entraînement est du pur comptage ou de la moyenne ; la prédiction est un produit matriciel.

**Variante multinomiale (comptages de mots).** Soit $n_{ki}$ le comptage total du mot $i$ sur tous les documents d'entraînement de la classe $k$, $V$ la taille du vocabulaire. Estimation du maximum de vraisemblance $\theta_{ki}=n_{ki}/\sum_in_{ki}$, et un document de comptages $x_i$ a pour log-score $\log p(C_k)+\sum_ix_i\log\theta_{ki}$. C'est linéaire en le vecteur de comptages $x$ (compare avec l'Éq. 4.82 de Bishop pour des variables binaires, également linéaire en $x$ : le Bayes naïf donne un classifieur linéaire, comme la régression logistique, mais ajusté par comptage plutôt que par optimisation).

**Lissage de Laplace.** Si un mot n'est jamais apparu dans une classe, $\theta_{ki}=0$, et un seul mot de ce type annule tout le produit (ou donne un log-score de $-\infty$) quelle que soit la quantité d'autres indices. Ajoute un pseudo-comptage $\alpha$ (en général 1) à chaque mot :

$$
\theta_{ki}=\frac{n_{ki}+\alpha}{\sum_in_{ki}+\alpha V}.
$$

C'est équivalent à un a priori de Dirichlet uniforme sur les probabilités des mots, donc $\alpha$ est un cadran de force d'a priori : $\alpha\to0$ fait confiance aux données, un grand $\alpha$ aplatit vers l'uniforme.

**Variante gaussienne (variables continues).** Par classe et par variable, estime la moyenne $\mu_{ki}$ et la variance $\sigma_{ki}^2$ et utilise

$$
\log p(x_i\mid C_k)=-\tfrac12\log(2\pi\sigma_{ki}^2)-\frac{(x_i-\mu_{ki})^2}{2\sigma_{ki}^2}.
$$

Le modèle génératif gaussien de Bishop avec une covariance pleine *partagée* donne une frontière linéaire ; le gaussien naïf avec des variances propres à chaque classe donne des frontières quadratiques mais ignore les corrélations entre variables. Un minuscule plancher de variance évite la division par zéro.

**Pourquoi cela marche-t-il quand l'hypothèse est fausse ?** La classification n'a besoin que de l'argmax. Des variables corrélées comptent deux fois l'information, ce qui rend la probabilité a posteriori trop confiante (probabilités proches de 0 ou 1), mais le classement des classes reste souvent correct. L'hypothèse forte implique aussi peu de paramètres, donc une faible variance : avec peu de données il bat souvent des modèles flexibles. Il échoue quand des variables corrélées portent le signal (par ex. des interactions de type XOR) et ses probabilités sont mal calibrées.

## Exemple détaillé 1 : la mise à l'échelle décide du voisin

Requête : âge 30, revenu $50{,}000$. Candidat A : âge 31, revenu $60{,}000$ (classe 0). Candidat B : âge 60, revenu $50{,}500$ (classe 1).

- Distances euclidiennes brutes : $d(q,A)=\sqrt{1+10000^2}=10000.00005$, $d(q,B)=\sqrt{30^2+500^2}=500.90$. B est « le plus proche » : l'écart d'âge de 30 ans est invisible à côté des dollars.
- Standardise avec $\sigma_{\text{age}}=10$, $\sigma_{\text{income}}=10{,}000$ : $d(q,A)=\sqrt{0.1^2+1^2}=1.005$, $d(q,B)=\sqrt{3^2+0.05^2}=3.000$. Maintenant A est le plus proche.

## Exemple détaillé 2 : un minuscule filtre anti-spam

Vocabulaire (free, win, meeting, report). Trois e-mails de spam avec les lignes de comptages $(3,2,0,0),(2,1,0,1),(1,2,1,0)$ ; trois e-mails légitimes (ham) $(0,0,2,3),(1,0,3,1),(0,1,2,2)$. Les a priori valent $\tfrac12,\tfrac12$.

- Totaux de mots du spam : $(6,5,1,1)$, somme $13$. Avec $\alpha=1$, $V=4$ : $\theta_{\text{spam}}=(7,6,2,2)/17=(0.412,0.353,0.118,0.118)$.
- Totaux du ham : $(1,1,7,6)$, somme $15$ : $\theta_{\text{ham}}=(2,2,8,7)/19=(0.105,0.105,0.421,0.368)$.
- Nouvel e-mail « free free meeting » : comptages $x=(2,0,1,0)$.
  - $\log$ score spam $=\ln0.5+2\ln0.412+\ln0.118=-4.608$.
  - $\log$ score ham $=\ln0.5+2\ln0.105+\ln0.421=-6.061$.
  - A posteriori spam $=\sigma(-4.608-(-6.061))=\sigma(1.453)=0.810$, où $\sigma(a)=1/(1+e^{-a})$.

L'e-mail est classé comme spam avec une probabilité a posteriori de $0.81$. Si « free » n'était jamais apparu dans le ham, le score ham non lissé vaudrait $-\infty$ quel que soit « meeting » ; le lissage l'évite.

## Implémentation

```python
import numpy as np

def knn_predict(Xtr, ytr, Xte, k):
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)    # (M, N)
    nn = np.argsort(d2, axis=1)[:, :k]                         # (M, k)
    return np.array([np.bincount(ytr[r]).argmax() for r in nn])

def mnb_fit(X, y, alpha=1.0):                                   # X: (N, V) counts
    classes = np.unique(y)
    log_prior = np.log(np.array([(y == c).mean() for c in classes]))
    counts = np.array([X[y == c].sum(0) for c in classes]) + alpha
    log_theta = np.log(counts / counts.sum(1, keepdims=True))
    return classes, log_prior, log_theta

def mnb_scores(X, log_prior, log_theta):
    return X @ log_theta.T + log_prior                          # (N, classes)
```

## Formes

kNN : `Xtr (N, d)`, `Xte (M, d)`, distances deux à deux `(M, N)` (mémoire $MNd$ avec le broadcast naïf ci-dessus ; utilise l'identité $\|a-b\|^2=\|a\|^2+\|b\|^2-2a\cdot b$ pour n'avoir que du $(M,N)$). NB multinomial : comptages `(N, V)`, `log_theta (C, V)`, `log_prior (C,)`, scores `(N, C)`. NB gaussien : moyennes et variances `(C, d)`.

## Idées reçues

- **« Le kNN n'a pas d'hyperparamètres. »** $k$, la distance et la mise à l'échelle sont tous des hyperparamètres ; ajuste-les uniquement sur les blocs d'entraînement.
- **Mettre à l'échelle sur tout le jeu de données.** Calcule les moyennes et écarts-types sur le découpage d'entraînement uniquement ; sinon de l'information du test fuit.
- **« Un temps d'entraînement nul signifie pas cher. »** Le kNN reporte le coût sur le temps de prédiction et la mémoire.
- **« Le Bayes naïf produit des probabilités fiables. »** Elles sont en général trop extrêmes à cause du double comptage ; utilise-les pour classer, ou calibre-les.
- **« L'indépendance est nécessaire pour que le Bayes naïf marche. »** C'est une hypothèse du modèle, pas des données ; il marche souvent malgré les violations, mais échoue sous de fortes dépendances pertinentes pour l'étiquette.
- **Oublier de lisser, ou lisser l'a priori.** Le lissage de Laplace s'applique aux probabilités des variables $\theta_{ki}$.
- **Multiplier directement les probabilités** au lieu d'additionner des logs : dépassement par le bas pour de longs documents.

## Points clés à retenir

- A posteriori du kNN $=K_c/K$ découle d'estimations locales de densité plus le théorème de Bayes ; $k$ contrôle le lissage ; mets les variables à l'échelle ; méfie-toi des grandes dimensions.
- Bayes naïf : $\arg\max_k\ \log p(C_k)+\sum_i\log p(x_i\mid C_k)$, entraîné par comptage (multinomial) ou par moyenne et variance par classe (gaussien).
- Le lissage de Laplace $(n+\alpha)/(\text{total}+\alpha V)$ empêche les probabilités nulles ; calcule en espace logarithmique.
- Le kNN est flexible mais lent et fragile en grande dimension ; le Bayes naïf est rapide et stable avec peu de données mais biaisé par son hypothèse d'indépendance.

## Explique-le toi-même

1. Dérive $p(c\mid x)=K_c/K$ et dis quelles quantités se simplifient.
2. Explique pourquoi le Bayes naïf peut bien classer alors que ses estimations de probabilité sont mal calibrées.

## Pour aller plus loin

- Bishop, *PRML*, Sec. 2.5.2 (PDF p.144-147) et Sec. 4.2 (PDF p.216-223).
- Leçons du dépôt `phases/02-ml-fundamentals/06-knn-and-distances` et `14-naive-bayes`.
- Lié : [[cml.02]] (un classifieur linéaire discriminatif pour la même tâche), [[cml.05]] (réduire la dimension avant le kNN).

## Critères de maîtrise

Tu sais prédire avec le kNN à la main pour quelques points, montrer comment la mise à l'échelle des variables change les voisins, mesurer la concentration des distances quand la dimension croît, et entraîner et appliquer le Bayes naïf multinomial et gaussien avec lissage en espace logarithmique.
