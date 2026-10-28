---
id: nlp.02
title: "Classification de texte et analyse de sentiment"
summary: "Dérive le Naive Bayes multinomial avec lissage, compare-le à la régression logistique sur TF-IDF, et apprends à évaluer et déboguer des classifieurs de texte, y compris avec des classes déséquilibrées."
minutes: 75
teaches: [nlp.text-classification]
requires: [nlp.tfidf, prob.basics, ml.logreg, ml.eval-metrics]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/docs/en.md: Naive Bayes and logistic regression baselines, negation handling, failure modes. READ."
  - "math/cours-tal-01.pdf (PDF p.5): list of NLP tasks including text classification (spam, sentiment). READ."
---

## Où ça s'inscrit

Classer du texte est la tâche de NLP la plus courante en production : router un ticket de support, signaler du spam, détecter la langue, étiqueter le sentiment d'un avis. C'est aussi le meilleur laboratoire pour tout le domaine, car chaque échec d'une méthode simple (négation, sarcasme, mots rares, classes déséquilibrées) motive un modèle plus riche par la suite. Tu as besoin des vecteurs creux de [[nlp.01]], de la régression logistique de [[cml.02]], et des métriques de [[cml.10]] et [[cml.11]] (précision, rappel, F1, déséquilibre) ; cette leçon les applique au texte et ajoute un classifieur conçu pour les comptes, Naive Bayes.

## L'idée en termes simples

Supposons que tu aies un tas de critiques de films que des humains ont étiquetées « bonne » ou « mauvaise ». Naive Bayes se comporte comme un comptable : pour chaque classe il tient un pointage de la fréquence de chaque mot, et quand une nouvelle critique arrive il demande : « si c'était une bonne critique, à quel point ces mots seraient-ils surprenants ? Et si c'était une mauvaise ? » La classe qui trouve les mots moins surprenants gagne. On l'appelle *naïf* parce qu'il fait comme si les mots étaient indépendants une fois la classe connue, comme si le critique tirait chaque mot d'un sac propre à la classe sans tenir compte du mot précédent. C'est évidemment faux (« not » change le sens de « good »), pourtant le classement des classes est souvent juste.

La régression logistique apprend à la place un poids par mot : les poids positifs poussent vers « bonne », les négatifs vers « mauvaise », et elle peut apprendre que certains mots sont forts et d'autres sans importance, en s'ajustant pour les mots qui tendent à apparaître ensemble. Le prix est une boucle d'optimisation au lieu d'un seul passage de comptage.

## Naive Bayes : dérivation

Soit un document une séquence de tokens $x=(w_1,\dots,w_m)$ et $c$ une classe. La règle de Bayes donne

$$
P(c\mid x)=\frac{P(x\mid c)\,P(c)}{P(x)}\ \propto\ P(c)\,P(x\mid c).
$$

Le dénominateur est le même pour toutes les classes, on peut donc l'omettre pour la classification. L'hypothèse **naïve**, selon laquelle les tokens sont indépendants sachant la classe, et le modèle **multinomial**, où chaque token est tiré d'une distribution propre à la classe $\theta_{c}$ sur le vocabulaire, donnent

$$
P(x\mid c)=\prod_{i=1}^{m}\theta_{c,w_i}\qquad\Longrightarrow\qquad \log P(c\mid x)=\log P(c)+\sum_{i=1}^{m}\log\theta_{c,w_i}+\text{const}.
$$

Nous travaillons en logarithmes parce que multiplier des dizaines de probabilités inférieures à 1 provoque un dépassement vers le bas (underflow).

**Estimer les paramètres.** L'a priori est la fraction de classe, $P(c)=N_c/N$. Pour $\theta$ nous maximisons la vraisemblance des tokens d'entraînement de la classe $c$ sous la contrainte $\sum_j\theta_{c,j}=1$ ; un multiplicateur de Lagrange donne la fréquence relative $\theta_{c,j}=\frac{n_{c,j}}{\sum_k n_{c,k}}$, où $n_{c,j}$ est le nombre d'occurrences du token $j$ dans les documents de la classe $c$. Le hic : un token jamais vu dans la classe $c$ a $\theta=0$ et un seul de ces tokens envoie tout le produit (et la classe) à zéro.

**Lissage.** Ajoute un pseudo-compte $\alpha>0$ à chaque token dans chaque classe :

$$
\theta_{c,j}=\frac{n_{c,j}+\alpha}{\sum_k n_{c,k}+\alpha|V|},\qquad |V|=\text{vocabulary size}.
$$

Le dénominateur croît de $\alpha|V|$ pour que les probabilités aient toujours pour somme 1. Avec $\alpha=1$ c'est le lissage de Laplace ; il est égal à la moyenne a posteriori sous un a priori de Dirichlet symétrique (la distribution de Dirichlet réapparaît dans [[nlp.05]]). De plus petites valeurs comme $0.1$ sont souvent meilleures sur de grands vocabulaires, et $\alpha$ est un hyperparamètre à régler sur les données de validation.

**Règle de décision.** $\hat c=\arg\max_c\big[\log P(c)+\sum_i\log\theta_{c,w_i}\big]$. Les tokens hors du vocabulaire d'entraînement sont ignorés. Pour transformer les scores en probabilités, soustrais le score maximal et normalise avec un softmax (log-sum-exp), mais note que ces probabilités sont généralement trop confiantes à cause de l'hypothèse d'indépendance.

## Régression logistique sur TF-IDF

Avec la matrice TF-IDF $W\in\mathbb{R}^{N\times|V|}$ ([[nlp.01]]), le modèle est $P(y=1\mid w)=\sigma(w^\top\beta+b)$ avec $\sigma(z)=1/(1+e^{-z})$, entraîné en minimisant l'entropie croisée plus une pénalité L2. Le gradient est le familier $\frac{1}{N}W^\top(\hat p-y)+\lambda\beta$, un poids par colonne, donc $\beta$ est directement lisible : trie les poids pour voir les mots qui poussent le plus une décision. Comparée à Naive Bayes, elle gère mieux les caractéristiques corrélées et gagne généralement avec assez de données ; Naive Bayes gagne quand les données sont minuscules, s'entraîne bien plus vite et est difficile à surapprendre. Les deux sont des classifieurs linéaires : la frontière de décision dans l'espace des caractéristiques est un hyperplan. La régularisation L2 compte ici parce que le modèle a bien plus de caractéristiques que d'exemples.

## Évaluer les classifieurs de texte

Pour une classe positive, avec les vrais positifs $TP$, les faux positifs $FP$ et les faux négatifs $FN$ (détails dans [[cml.10]]) :

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}.
$$

Avec plusieurs classes, le **macro-F1** moyenne les F1 par classe (chaque classe compte de la même façon), tandis que le **micro-F1** met en commun les comptes (les classes fréquentes dominent). Rapporte le macro quand les classes rares comptent.

Le **déséquilibre des classes** est la norme en texte : 1 % des e-mails sont du phishing, 3 % des avis mentionnent la sécurité. La précision (accuracy) induit alors en erreur, car prédire la classe majoritaire donne 97 % gratuitement. Remèdes, du moins cher au plus coûteux : rapporter précision, rappel et F1 de la classe minoritaire ; déplacer le seuil de décision (pour Naive Bayes, biaiser le log de l'a priori) ; utiliser des poids de classes dans la perte ; rééchantillonner le jeu d'entraînement (jamais le jeu de test) ; collecter plus d'exemples minoritaires. Voir [[cml.11]] pour savoir pourquoi le rééchantillonnage doit se faire à l'intérieur du pli d'entraînement uniquement.

**L'analyse d'erreurs** est l'endroit d'où vient l'essentiel de l'amélioration. Affiche les documents de validation mal classés, regroupe-les par cause (négation, sarcasme, mots de domaine, bruit d'étiquettes, textes très courts), et corrige le plus gros groupe. Causes typiques propres au texte : la négation (`not good`), que les bigrammes ou l'astuce de marquage de la négation (préfixe « NOT_ » sur les tokens après un négateur, jusqu'à la ponctuation suivante) gèrent en partie ; le sarcasme et l'ironie, que le sac de mots ne peut pas voir ; le décalage de domaine (`unpredictable` est un éloge pour un thriller et une insulte pour un lave-linge) ; et les données mal étiquetées.

## Exemple travaillé

Jeu d'entraînement de six minuscules avis (tokens séparés par des espaces) : positifs : `great movie loved it`, `loved the great acting`, `fun and great` ; négatifs : `boring movie`, `terrible boring plot`, `bad acting boring`. Le vocabulaire compte $|V|=12$ mots. Les a priori sont $0.5/0.5$. La classe positive a $11$ tokens et la classe négative $8$.

Classe `great boring movie` avec $\alpha=1$. Comptes : dans la classe positive great = 3, boring = 0, movie = 1 ; dans la classe négative great = 0, boring = 3, movie = 1.

- Positive : $\ln0.5+\ln\frac{4}{23}+\ln\frac{1}{23}+\ln\frac{2}{23}=-0.6931-1.7492-3.1355-2.4423=-8.0202$.
- Négative : $\ln0.5+\ln\frac{1}{20}+\ln\frac{4}{20}+\ln\frac{2}{20}=-0.6931-2.9957-1.6094-2.3026=-7.6009$.

La classe négative gagne ; le softmax des deux scores donne $P(\text{neg})=0.6033$. Le seul mot `boring` est une preuve plus forte dans ce minuscule corpus que `great`, parce que `boring` apparaît trois fois dans la classe négative. En vérifiant un autre avis, `great fun movie` donne $P(\text{pos})=0.8403$.

Métriques sous déséquilibre : supposons qu'un jeu de test ait 10 avis dont 3 positifs, et qu'un classifieur prédise 4 positifs, dont 2 corrects. Alors $TP=2$, $FP=2$, $FN=1$, $TN=5$ : $P=0.5$, $R=0.667$, $F_1=0.571$, précision (accuracy) $=0.7$. Un classifieur qui dit toujours « négatif » a aussi une précision de $0.7$ mais $R=0$. L'accuracy seule ne peut pas les distinguer. L'exercice reproduit chaque nombre.

## Implémentation

```python
import numpy as np

def train_nb(docs, labels, alpha=1.0):
    vocab = sorted({w for d in docs for w in d})
    ix = {w: i for i, w in enumerate(vocab)}
    classes = sorted(set(labels))
    log_prior = np.zeros(len(classes))            # (C,)
    log_theta = np.zeros((len(classes), len(vocab)))  # (C, V)
    for c, cls in enumerate(classes):
        cls_docs = [d for d, l in zip(docs, labels) if l == cls]
        log_prior[c] = np.log(len(cls_docs) / len(docs))
        counts = np.zeros(len(vocab))
        for d in cls_docs:
            for w in d:
                counts[ix[w]] += 1
        log_theta[c] = np.log((counts + alpha) / (counts.sum() + alpha * len(vocab)))
    return classes, vocab, log_prior, log_theta

def nb_scores(doc, vocab, log_prior, log_theta):
    ix = {w: i for i, w in enumerate(vocab)}
    return log_prior + sum(log_theta[:, ix[w]] for w in doc if w in ix)   # (C,)
```

En pratique, utilise `sklearn.naive_bayes.MultinomialNB` (qui accepte aussi des valeurs TF-IDF comme comptes fractionnaires) et `sklearn.linear_model.LogisticRegression` avec `class_weight="balanced"` pour le déséquilibre, et compare toujours à une référence de classe majoritaire.

## Formes

- `log_prior` : $(C,)$. `log_theta` : $(C,V)$, les exponentielles de chaque ligne ont pour somme 1.
- Scores pour un document : $(C,)$ ; pour une matrice de comptes $X$ de forme $(N,V)$, tous les scores sont `X @ log_theta.T + log_prior`, de forme $(N,C)$. C'est pourquoi Naive Bayes est un modèle linéaire dans les comptes.

## Idées reçues

- **« L'indépendance est fausse, donc Naive Bayes ne peut pas fonctionner. »** Les probabilités sont médiocres, mais l'arg max est souvent juste ; c'est un outil de classement, pas un estimateur de probabilité calibré.
- **« Une haute précision (accuracy) signifie un bon classifieur. »** Sur des données déséquilibrées elle signifie souvent « prédit la classe majoritaire ».
- **« Le lissage est un détail. »** Sans lui, un seul mot non vu annule une classe. Et $\alpha$ change sensiblement les résultats, donc règle-le.
- **« Équilibre aussi le jeu de test. »** Rééchantillonne uniquement les données d'entraînement ; le jeu de test doit refléter la vraie distribution des classes.
- **« Plus de caractéristiques aident toujours. »** Avec peu de documents et un énorme vocabulaire, une régression logistique surapprend ; régularise.

## Points clés à retenir

- Le Naive Bayes multinomial est $\arg\max_c\,[\log P(c)+\sum_i\log\theta_{c,w_i}]$ avec des fréquences relatives lissées, entraîné par comptage.
- La régression logistique sur TF-IDF apprend un poids lisible par mot et bat généralement Naive Bayes avec assez de données.
- Évalue avec précision, rappel et F1 par classe (macro quand les classes comptent également), jamais l'accuracy seule sur des données déséquilibrées.
- L'analyse d'erreurs des fautes de validation est l'activité au plus fort effet de levier.

## Explique-le toi-même

1. Explique pourquoi Naive Bayes multiplie (additionne en logs) les probabilités des mots, et sur quelle fausse hypothèse cela repose.
2. Un filtre anti-spam a 99 % d'accuracy sur un jeu de test où 1 % des mails sont du spam. Quels autres nombres demandes-tu avant de lui faire confiance ?

## Pour aller plus loin

- Jurafsky et Martin, *Speech and Language Processing*, chapitre sur Naive Bayes et la classification de sentiment.
- Manning, Raghavan et Schütze, *Introduction to Information Retrieval*, chapitre sur la classification de texte.

## Critères de maîtrise

- Tu sais calculer à la main un score de Naive Bayes, lissage compris, et dire pourquoi les logs sont utilisés.
- Tu sais calculer précision, rappel, F1 à partir d'une matrice de confusion et expliquer macro vs micro.
- Tu sais citer trois remèdes au déséquilibre des classes et la seule chose à ne jamais faire au jeu de test.
