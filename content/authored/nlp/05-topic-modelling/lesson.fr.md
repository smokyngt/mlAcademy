---
id: nlp.05
title: "Modélisation de sujets avec LDA"
summary: "Comprends l'histoire générative de LDA et les a priori de Dirichlet, dérive la mise à jour de l'échantillonnage de Gibbs collapsé, implémente un minuscule LDA avec NumPy, et apprends comment les sujets sont évalués et en quoi les sujets à base de plongements diffèrent."
minutes: 85
teaches: [nlp.topic-models]
requires: [prob.basics, nlp.tfidf]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/15-topic-modeling/docs/en.md: LDA generative story, BERTopic pipeline, coherence and diversity, when to pick which. READ."
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 11.3 Gibbs Sampling (PDF p.564-565): why sampling one variable from its conditional leaves the joint invariant, the ergodicity condition, and the acceptance probability of 1. READ."
---

## Où ça s'inscrit

Tu as 50 000 tickets de support ou articles de presse et aucune étiquette. De quoi parlent-ils ? La **modélisation de sujets** répond sans supervision : elle renvoie une poignée de listes de mots (des « sujets ») et, pour chaque document, la part de chaque sujet qu'il contient. C'est de l'analyse exploratoire de données pour le texte, utilisée pour organiser des archives, résumer des tendances et créer des caractéristiques. Elle s'appuie sur les comptes de sacs de mots de [[nlp.01]] (LDA utilise des comptes bruts, pas TF-IDF) et sur des idées bayésiennes : une histoire générative, un a priori et de l'échantillonnage. L'échantillonnage de Gibbs ici est le même outil que Bishop présente pour les modèles probabilistes généraux, et le lissage de Dirichlet est le même que l'add-$\alpha$ de [[nlp.02]].

## L'idée en termes simples

Imagine un journal écrit par un comité. Le comité a quelques thèmes dans un tiroir : sport, cuisine, politique. Chaque thème est un sac de mots avec ses propres fréquences (le sport contient souvent « goal », jamais « oven »). Pour écrire un article, l'auteur décide d'abord d'un *mélange* (disons 70 % sport, 30 % politique). Puis, pour chaque position de mot, il tire à pile ou face selon ce mélange pour choisir un thème, et sort un mot au hasard du sac de ce thème. Nous ne voyons que les articles terminés. L'**allocation de Dirichlet latente** (LDA) fait tourner cette histoire à l'envers : étant donnés tous les articles, trouver des sacs de mots et des mélanges qui les rendraient probables.

Deux conséquences pratiques. Les documents sont des mélanges, pas des étiquettes uniques : un article sur un concours de cuisine peut être à 50 % du sport. Et le nombre de thèmes est un choix que tu fais.

## Le modèle

Soit $K$ le nombre de sujets, $V$ la taille du vocabulaire, $D$ le nombre de documents. Pour chaque sujet $k$ il y a une distribution de mots $\phi_k$ sur le vocabulaire (un vecteur de longueur $V$ de somme 1), et pour chaque document $d$ un mélange de sujets $\theta_d$ (de longueur $K$, de somme 1). L'histoire générative :

1. Pour chaque sujet $k$ : tirer $\phi_k\sim\mathrm{Dirichlet}(\beta,\dots,\beta)$.
2. Pour chaque document $d$ : tirer $\theta_d\sim\mathrm{Dirichlet}(\alpha,\dots,\alpha)$.
3. Pour chaque position de mot $i$ du document $d$ : tirer un sujet $z_{di}\sim\mathrm{Categorical}(\theta_d)$, puis un mot $w_{di}\sim\mathrm{Categorical}(\phi_{z_{di}})$.

**Intuition de Dirichlet.** Une distribution de Dirichlet est une distribution sur des vecteurs de probabilité. Avec un paramètre de concentration symétrique $\alpha$ : si $\alpha=1$ tous les mélanges sont équiprobables ; si $\alpha<1$ elle préfère les vecteurs creux, quelques grandes entrées et le reste proche de zéro (documents avec un ou deux sujets dominants, sujets avec un petit ensemble de mots caractéristiques) ; si $\alpha>1$ elle préfère les vecteurs presque uniformes. Choix typiques : $\alpha\approx0.1$ à $50/K$, $\beta\approx0.01$ à $0.1$. C'est l'a priori conjugué de la distribution catégorielle : un a priori de Dirichlet plus des observations catégorielles donne un a posteriori de Dirichlet dont les paramètres sont augmentés des comptes, ce qui explique pourquoi les formules ci-dessous sont « comptes plus $\alpha$ ».

Nous observons les mots $w$. Inconnues : $z$, $\theta$, $\phi$. L'inférence a posteriori exacte est intraitable, nous utilisons donc l'échantillonnage (ici) ou des méthodes variationnelles (le défaut de scikit-learn).

## Dérivation : échantillonnage de Gibbs collapsé

L'**échantillonnage de Gibbs** (Bishop Sec. 11.3) tire des échantillons d'une distribution jointe en ré-échantillonnant répétitivement une variable à partir de sa distribution conditionnelle sachant toutes les autres. Bishop montre deux choses : la distribution jointe est invariante par chacune de ces étapes (les variables de conditionnement ne sont pas touchées et celle ré-échantillonnée provient de sa conditionnelle correcte), et la chaîne est ergodique si aucune conditionnelle n'est nulle nulle part. L'échantillonnage de Gibbs est aussi du Metropolis-Hastings avec une probabilité d'acceptation toujours égale à 1, donc aucune proposition n'est rejetée.

Pour LDA nous **collapsons** (intégrons) $\theta$ et $\phi$ et n'échantillonnons que les affectations de sujets $z$. Notation : $n_{dk}$ = nombre de tokens du document $d$ affectés au sujet $k$ ; $n_{kw}$ = nombre de fois où le mot $w$ est affecté au sujet $k$ ; $n_k=\sum_w n_{kw}$.

Pour un document, intégrer un Dirichlet contre les comptes multinomiaux donne

$$
\int \mathrm{Dir}(\theta\mid\alpha)\prod_k\theta_k^{n_{dk}}\,d\theta=\frac{\Gamma(K\alpha)}{\Gamma(K\alpha+N_d)}\prod_{k}\frac{\Gamma(\alpha+n_{dk})}{\Gamma(\alpha)},
$$

et de même pour les comptes de mots de chaque sujet avec $\beta$ et $V$. La jointe $P(w,z)$ est le produit de ces termes. Pour ré-échantillonner le sujet d'un token, nous avons besoin de $P(z_{di}=k\mid z_{-di},w)\propto P(w,z)$ vue comme une fonction de $k$ avec tout le reste fixé. Retire le token $i$ des comptes (appelle $n^{-di}$ les comptes réduits). Le remettre dans le sujet $k$ ne change que $n_{dk}$ et $n_{kw}$ (avec $w=w_{di}$) de $+1$. Chaque rapport de Gamma est de la forme $\Gamma(a+1)/\Gamma(a)=a$, et les normaliseurs ne dépendent que de totaux qui sont les mêmes pour tout $k$ sauf $n_k$, donc

$$
P(z_{di}=k\mid z_{-di},w)\ \propto\ \big(n^{-di}_{dk}+\alpha\big)\cdot\frac{n^{-di}_{kw}+\beta}{n^{-di}_{k}+V\beta}.
$$

Lis cela ainsi : *à quel point ce document aime le sujet $k$* fois *à quel point le sujet $k$ aime ce mot*. Normalise sur les $K$ valeurs et échantillonne.

**Algorithme.** Initialise chaque $z_{di}$ au hasard et remplis les comptes. Répète pendant de nombreux balayages : pour chaque token, décrémente ses comptes, calcule les $K$ poids ci-dessus, échantillonne un nouveau sujet, incrémente les comptes. Après la période de chauffe (burn-in), estime

$$
\hat\theta_{dk}=\frac{n_{dk}+\alpha}{N_d+K\alpha},\qquad \hat\phi_{kw}=\frac{n_{kw}+\beta}{n_k+V\beta}.
$$

Coût : $O(K)$ par token et par balayage.

## Évaluation

Il n'y a pas de vérité terrain, donc l'évaluation combine plusieurs signaux.

- **Perplexité ou log-vraisemblance sur données mises de côté** de documents non vus : fondée en principe mais faiblement corrélée au jugement humain.
- **Cohérence** (par exemple $c_v$, construite sur l'information mutuelle ponctuelle normalisée des mots principaux d'un sujet sur une fenêtre de texte) : les mots principaux co-occurrent-ils dans du vrai texte ? Plus élevé, c'est mieux. Bibliothèques : `gensim.models.CoherenceModel`.
- **Diversité des sujets** : fraction de mots distincts parmi les $n$ mots principaux de tous les sujets. De faibles valeurs signifient des sujets répétés et redondants.
- **Lecture humaine** : regarde les dix mots principaux et des échantillons de documents de chaque sujet. Cela reste la dernière ligne de défense.

Choix pratiques : retire les mots vides et les mots très rares ou très fréquents avant de compter ; essaie plusieurs $K$ et compare la cohérence et ta lecture, plutôt que de te fier à un seul nombre ; lance avec différentes graines (l'échantillonneur trouve des optima locaux) et vérifie la stabilité.

## Sujets à base de plongements : l'idée de BERTopic

LDA ne voit que des comptes de mots, ce qui échoue sur du texte court (tweets, titres) où chaque document a trop peu de mots. Le pipeline BERTopic décrit dans la source fonctionne différemment : plonger chaque document avec un encodeur de phrases (vecteurs denses, voir [[seq.02]]), réduire les dimensions (UMAP), regrouper (HDBSCAN, qui étiquette aussi les valeurs aberrantes), puis nommer chaque groupe par ses mots les plus caractéristiques avec un TF-IDF par classe (le TF-IDF de [[nlp.01]] avec chaque groupe traité comme un gros document). Sortie : un sujet par document plus un groupe de valeurs aberrantes, avec la sémantique capturée par les plongements. Compromis : LDA donne des mélanges et gère les longs documents avec peu de calcul ; BERTopic est plus fort sur le texte court et la cohérence sémantique mais donne des affectations dures et tronque les longues entrées. Les deux peuvent être inspectés avec les mêmes vérifications de cohérence et de diversité.

## Exemple travaillé

D'abord un pas de mise à jour. Un token de mot $w$ a, après son retrait, les comptes de document $n_{d\cdot}=(3,1)$, les comptes de mot $n_{\cdot w}=(4,0)$, les totaux de sujets $n_\cdot=(10,8)$, avec $V=8$, $\alpha=0.1$, $\beta=0.01$. Poids non normalisés : sujet 0 : $(3+0.1)\cdot\frac{4+0.01}{10+0.08}=1.2333$ ; sujet 1 : $(1+0.1)\cdot\frac{0+0.01}{8+0.08}=0.001361$. Normalisés : $(0.99890,\ 0.00110)$. Le document est surtout du sujet 0 et le sujet 0 possède déjà ce mot, donc le token reste presque sûrement dans le sujet 0.

Maintenant tout l'échantillonneur. Vocabulaire de 8 identifiants de mots : les identifiants 0-3 sont des mots de « sport », les identifiants 4-7 des mots de « cuisine ». Neuf documents de 20 tokens : quatre tirés uniquement des identifiants 0-3, quatre uniquement des identifiants 4-7, et un tiré des huit. Avec $K=2$, $\alpha=0.1$, $\beta=0.01$ et 60 balayages, chacune des dix graines essayées retrouve deux sujets dont les 4 mots principaux sont exactement $\{0,1,2,3\}$ et $\{4,5,6,7\}$ (quel indice de sujet reçoit quel groupe dépend de la graine), les documents de sport et de cuisine obtiennent un $\hat\theta$ essentiellement égal à $(1,0)$ ou $(0,1)$, et le document mixte obtient un mélange partagé (environ $(0.5,0.5)$ pour la dernière graine). L'exercice vérifie ces propriétés.

## Implémentation

```python
import numpy as np

def conditional(ndk_d, nkw_w, nk, alpha, beta, V):
    p = (ndk_d + alpha) * (nkw_w + beta) / (nk + V * beta)   # shape (K,)
    return p / p.sum()

def lda_gibbs(docs, K, V, alpha, beta, iters, seed):
    rng = np.random.default_rng(seed)
    ndk = np.zeros((len(docs), K)); nkw = np.zeros((K, V)); nk = np.zeros(K)
    z = [rng.integers(K, size=len(d)) for d in docs]
    for d, doc in enumerate(docs):                       # fill the counts
        for i, w in enumerate(doc):
            k = z[d][i]; ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1
    for _ in range(iters):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                k = z[d][i]; ndk[d, k] -= 1; nkw[k, w] -= 1; nk[k] -= 1
                k = rng.choice(K, p=conditional(ndk[d], nkw[:, w], nk, alpha, beta, V))
                z[d][i] = k; ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1
    return ndk, nkw
```

En production, utilise `gensim.models.LdaModel` (variationnel, en flux) ou `sklearn.decomposition.LatentDirichletAllocation` ; pour BERTopic, utilise le paquet `bertopic`.

## Formes

- `docs` : liste de $D$ listes d'identifiants de mots. `z[d]` : $(N_d,)$.
- `ndk` : $(D,K)$, les lignes ont pour somme $N_d$. `nkw` : $(K,V)$, la somme totale est le nombre de tokens. `nk` : $(K,)$.
- $\hat\theta$ : $(D,K)$, $\hat\phi$ : $(K,V)$, chaque ligne ayant pour somme 1.

## Idées reçues

- **« Les sujets sont des catégories étiquetées. »** Un sujet est une distribution sur des mots ; le nom est ton interprétation.
- **« LDA trouve le bon nombre de sujets. »** $K$ est un hyperparamètre ; compare plusieurs valeurs.
- **« Utilise une entrée TF-IDF pour LDA. »** Le modèle génératif porte sur des comptes ; utilise des comptes bruts.
- **« Une exécution de l'échantillonneur donne la réponse. »** C'est une recherche aléatoire d'un optimum local ; vérifie la stabilité sur plusieurs graines.
- **« Les comptes ne doivent être décrémentés qu'à la fin du balayage. »** Le token ré-échantillonné doit être retiré des comptes avant de calculer sa conditionnelle, sinon il vote pour son propre sujet actuel.

## Points clés à retenir

- LDA = les documents sont des mélanges de sujets, les sujets sont des distributions sur des mots, avec des a priori de Dirichlet.
- Gibbs collapsé : $P(z=k)\propto(n_{dk}+\alpha)\frac{n_{kw}+\beta}{n_k+V\beta}$, avec le token courant retiré.
- Évalue avec la cohérence, la diversité, la vraisemblance sur données mises de côté et la lecture des sujets.
- BERTopic regroupe plutôt des plongements ; il aide sur le texte court mais perd les mélanges.

## Explique-le toi-même

1. Décris avec des mots ce que mesure chacun des deux facteurs de la mise à jour de Gibbs, et ce qui se passe quand $\alpha$ est très petit.
2. Quand préférerais-tu BERTopic à LDA, et à quoi renonces-tu ?

## Pour aller plus loin

- Blei, Ng et Jordan, « Latent Dirichlet Allocation » (2003) ; Griffiths et Steyvers, « Finding scientific topics » (2004) pour l'échantillonneur collapsé.
- Bishop, *Pattern Recognition and Machine Learning*, Sec. 11.3.

## Critères de maîtrise

- Tu sais écrire l'histoire générative de LDA et nommer $\theta$, $\phi$, $z$.
- Tu sais dériver (ou au moins justifier chaque facteur de) la conditionnelle de Gibbs collapsée.
- Tu sais exécuter un minuscule LDA, lire sa sortie et dire comment tu le jugerais.
