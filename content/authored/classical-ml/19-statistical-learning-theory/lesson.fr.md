---
id: cml.19
title: "Théorie de l'apprentissage statistique : bornes de généralisation, dimension VC et complexité de Rademacher"
summary: Transforme « l'erreur d'entraînement me dit-elle quelque chose sur l'erreur future ? » en théorèmes que tu peux calculer, de Hoeffding et la borne de l'union jusqu'à la dimension VC et la complexité de Rademacher, et vois où la théorie est serrée, lâche ou muette.
minutes: 100
teaches: [ml.slt, ml.vc]
requires: [ml.generalization, prob.rv, stat.estimation]
sources:
  - "math/math for ML.pdf (Sec. 8.1.4 and Sec. 8.2 Empirical Risk Minimization, PDF p.263-269): empirical risk (8.6), expected/true risk (8.10), overfitting as under-estimated true risk, regularisation as a penalty on complexity. READ."
  - "math/math for ML.pdf (Sec. 8.3 Parameter estimation, PDF p.273-275): negative log-likelihood as a loss, MAP as a regularised estimate. READ."
---

Note sur la couverture de la bibliothèque : la bibliothèque que j'ai lue couvre en mots la minimisation du risque empirique et la régularisation. Elle ne couvre **pas** l'inégalité de Hoeffding, le théorème de généralisation par la borne de l'union, la dimension VC, la complexité de Rademacher ni le « no free lunch » ; ces parties sont écrites à partir de connaissances générales (résultats classiques de manuels) et chaque nombre est reproduit par le code des exercices. Les constantes des théorèmes diffèrent d'un manuel à l'autre ; considère les constantes exactes des bornes VC et de Rademacher comme indicatives et consulte une référence avant de les citer.

## Où cela s'inscrit

[[cml.03]] a montré expérimentalement qu'un modèle flexible peut ajuster parfaitement les données d'entraînement et pourtant échouer sur de nouvelles données, et que les ensembles de validation estiment l'erreur future. Une question naturelle suit : **peut-on savoir, avant de regarder un ensemble de test, quelle peut être la taille de l'écart entre l'erreur d'entraînement et l'erreur future, et qu'est-ce qui le contrôle ?** La théorie de l'apprentissage statistique répond par des inégalités de la forme « avec une probabilité d'au moins $1-\delta$, erreur vraie $\le$ erreur d'entraînement $+$ une pénalité qui croît avec la complexité du modèle et diminue avec les données ». C'est la raison mathématique derrière la régularisation ([[cml.16]]), l'arrêt précoce, les méthodes à marge ([[cml.08]]) et la règle empirique « plus de paramètres demandent plus de données ». Elle explique aussi où la théorie échoue à décrire les réseaux profonds modernes. Tu n'as besoin que de l'espérance et de la variance ([[math.09]]) et de l'idée d'estimateur ([[math.13]]).

## L'idée en mots simples

Imagine que tu juges un restaurant en goûtant un seul plat. Si tu goûtes dix plats et que tous sont excellents, tu es assez sûr que la cuisine est bonne. Mais que se passe-t-il si tu avais le droit de choisir, *après dégustation*, parmi un million de restaurants celui dont les dix plats avaient l'air les meilleurs ? Avec autant de choix, un établissement médiocre paraîtra parfait par chance. Apprendre, c'est exactement cela : l'ensemble d'entraînement est la dégustation, la classe de modèles est l'ensemble des restaurants parmi lesquels tu peux choisir, et la « minimisation du risque empirique » consiste à choisir celui qui avait l'air le meilleur.

L'inquiétude légitime a donc deux facteurs : **combien de dégustations** (la taille d'échantillon $n$) et **parmi combien de restaurants tu aurais pu choisir** (la richesse de la classe de modèles). Avec peu de choix effectivement distincts, une bonne dégustation est digne de confiance. Avec une offre illimitée, rien de ce que dit une dégustation ne mérite confiance. Le rôle de la théorie est de compter « combien de choix effectivement différents » de façon astucieuse : par la taille pour une classe finie, par la pulvérisation (dimension VC) pour une classe infinie, ou par la capacité à ajuster du bruit aléatoire (complexité de Rademacher).

## Définitions

Les données $(x_i,y_i)$, $i=1..n$, sont tirées i.i.d. d'une distribution inconnue $D$. Une **hypothèse** $h$ associe des prédictions aux entrées ; $\mathcal H$ est la **classe d'hypothèses**. La perte $\ell(h(x),y)\in[0,1]$ ; pour la classification, la perte 0-1 $\mathbf 1[h(x)\ne y]$.

- **Risque vrai (espéré)** $R(h)=\mathbb E_{(x,y)\sim D}\,\ell(h(x),y)$. Il ne peut pas être calculé ; c'est la quantité qui nous intéresse (math for ML, éq. 8.10).
- **Risque empirique** $\hat R_n(h)=\frac1n\sum_{i=1}^n\ell(h(x_i),y_i)$ (éq. 8.6).
- **Écart de généralisation** $R(h)-\hat R_n(h)$.
- **Minimisation du risque empirique (ERM)** : $\hat h=\arg\min_{h\in\mathcal H}\hat R_n(h)$.

Pour un $h$ *fixé* avant de voir les données, $\hat R_n(h)$ est une moyenne de $n$ variables i.i.d. d'espérance $R(h)$, c'est donc une estimation sans biais et l'écart est petit pour un grand $n$. Le problème est que $\hat h$ est choisi *en utilisant* les données : $\hat R_n(\hat h)$ est une estimation biaisée (optimiste) de $R(\hat h)$. Cet optimisme est le surapprentissage (math for ML Sec. 8.2.3 le décrit comme l'estimation du risque à partir des données d'entraînement qui sous-estime le risque vrai).

## Inégalité de Hoeffding

Pour des variables aléatoires indépendantes $Z_1,\dots,Z_n\in[0,1]$ d'espérance $\mu$,

$$
P\Big(\Big|\tfrac1n\sum_i Z_i-\mu\Big|\ge\varepsilon\Big)\le 2e^{-2n\varepsilon^2}.
$$

Applique-la avec $Z_i=\ell(h(x_i),y_i)$ pour un $h$ fixé : la probabilité que le risque empirique s'écarte de $\varepsilon$ ou plus décroît exponentiellement en $n\varepsilon^2$. (J'énonce ce résultat classique sans preuve ; la preuve borne la fonction génératrice des moments d'une variable bornée, le lemme de Hoeffding, puis utilise l'inégalité de Markov.) Pour $n=100$ et $\varepsilon=0.1$, la borne vaut $2e^{-2}=0.271$ : même 100 points laissent 27 % de chances, dans le pire cas, d'être à côté de 10 points.

## Dérivation : la borne de l'union pour une classe finie

Soit $|\mathcal H|=M$. Nous voulons un énoncé valable pour **tous** les $h\in\mathcal H$ simultanément, en particulier pour le $\hat h$ choisi d'après les données.

1. Pour chaque $h_j$ fixé, Hoeffding donne $P(|\hat R_n(h_j)-R(h_j)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2}$.
2. La **borne de l'union** : la probabilité qu'au moins un de $M$ événements se produise est au plus la somme de leurs probabilités (les événements peuvent se chevaucher, ce qui ne fait qu'aider). Donc

$$
P\big(\exists h\in\mathcal H:\ |\hat R_n(h)-R(h)|\ge\varepsilon\big)\le 2M e^{-2n\varepsilon^2}.
$$

3. Pose le membre de droite égal à $\delta$ et résous pour $\varepsilon$ : $2Me^{-2n\varepsilon^2}=\delta \Rightarrow e^{-2n\varepsilon^2}=\frac{\delta}{2M}\Rightarrow 2n\varepsilon^2=\ln\frac{2M}{\delta}$ :

$$
\boxed{\ \varepsilon=\sqrt{\frac{\ln(2M/\delta)}{2n}}\ }
$$

Donc, avec une probabilité d'au moins $1-\delta$, **chaque** $h\in\mathcal H$ vérifie $R(h)\le\hat R_n(h)+\varepsilon$, y compris $\hat h$. En résolvant pour $n$ : $n\ge\frac{\ln(2M/\delta)}{2\varepsilon^2}$.

Conséquences.

- La pénalité ne croît que comme $\sqrt{\ln M}$ : élever au carré le nombre d'hypothèses multiplie la pénalité par $\sqrt2$, pas par $M$. Une classe de $2^{b}$ hypothèses (un modèle de $b$ bits stockés) coûte $\sqrt{b\ln2/(2n)}$ : *le nombre de bits nécessaires pour décrire le modèle joue le rôle de la complexité*.
- Risque en excès : si $h^\star$ est le meilleur de la classe, alors $R(\hat h)\le\hat R_n(\hat h)+\varepsilon\le\hat R_n(h^\star)+\varepsilon\le R(h^\star)+2\varepsilon$. L'ERM est à moins de $2\varepsilon$ de la meilleure hypothèse de la classe.
- La borne est **uniforme** sur la classe : elle vaut quoi que fasse l'algorithme d'apprentissage, ce qui explique pourquoi elle est lâche pour un algorithme précis.

## Classes infinies : pulvérisation et dimension VC

Les classifieurs linéaires ont une infinité de membres ($M=\infty$), et pourtant ils ne peuvent pas tout faire. Ce qui compte, c'est combien d'*étiquetages différents de l'échantillon* la classe peut produire.

- Un ensemble de $m$ points est **pulvérisé** (shattered) par $\mathcal H$ si, pour chacun des $2^m$ étiquetages, un $h\in\mathcal H$ le réalise.
- La **dimension VC** $d_{VC}(\mathcal H)$ est la taille du plus grand ensemble que *certaine* disposition de points peut pulvériser (infinie s'il n'y a pas de plus grand). Pour prouver $d_{VC}=d$, tu dois exhiber un ensemble de $d$ points pulvérisé *et* montrer qu'aucun ensemble de $d+1$ ne l'est.

Deux cas détaillés, que l'exercice vérifie par force brute.

- **Intervalles sur une droite** (étiquette 1 dans $(a,b)$). Deux points quelconques sont pulvérisés. Pour 3 points $x_1<x_2<x_3$, l'étiquetage $(1,0,1)$ est impossible, car un intervalle contenant $x_1$ et $x_3$ contient $x_2$. Donc $d_{VC}=2$. La force brute trouve exactement $7$ des $8$ étiquetages.
- **Demi-plans dans le plan** (classifieurs linéaires avec biais dans $\mathbb R^2$). Les 3 sommets d'un triangle sont pulvérisés (les $8$ étiquetages). Pour 4 points, la force brute sur le carré donne $14$ des $16$ étiquetages : les étiquetages de type « ou exclusif » manquent. Pour 4 points aléatoires, cela ne réussit jamais ; le fait général (théorème de Radon : 4 points quelconques du plan peuvent être séparés en deux groupes dont les enveloppes convexes se coupent, donc cet étiquetage est impossible) donne $d_{VC}=3$. Dans $\mathbb R^p$, la dimension VC des classifieurs linéaires avec biais est $p+1$. Note que trois points *alignés* ne sont pas pulvérisés : la définition demande la meilleure disposition, pas toutes les dispositions. La force brute dans le code est un indice pour de petits cas, pas une preuve.

**Fonction de croissance et lemme de Sauer.** $\Pi_{\mathcal H}(n)$ = le nombre maximal d'étiquetages de $n$ points. Elle vaut $2^n$ pour $n\le d$ et, par le lemme de Sauer, est au plus $(en/d)^d$ pour $n\ge d$ : elle passe de l'exponentiel au polynomial. Ce polynôme joue le rôle de $M$ dans la borne de l'union.

**Borne de généralisation VC (énoncé).** Pour une classe de dimension VC $d$ et la perte 0-1, avec une probabilité d'au moins $1-\delta$, pour tout $h\in\mathcal H$,

$$
R(h)\le\hat R_n(h)+\sqrt{\frac{d\,\big(\ln(2n/d)+1\big)+\ln(4/\delta)}{n}}
$$

(une forme courante ; les constantes varient). C'est la structure qui compte : la pénalité vaut à peu près $\sqrt{d\ln n/n}$. **Une règle empirique : on veut $n$ très supérieur à $d$.** Je ne la dérive pas ici : la preuve remplace $M$ par la fonction de croissance grâce à un argument de symétrisation avec un échantillon fantôme.

## Complexité de Rademacher

La dimension VC ignore la distribution des données et raisonne dans le pire cas. La complexité de Rademacher mesure à quel point la classe peut ajuster du **bruit pur** sur l'échantillon réel. Tire des signes aléatoires $\sigma_i\in\{-1,+1\}$ avec probabilités égales (variables de Rademacher) et traite-les comme des étiquettes aléatoires. La **complexité de Rademacher empirique** de $\mathcal H$ (fonctions à valeurs $\pm1$) sur les points $x_1..x_n$ est

$$
\hat{\mathfrak R}_n(\mathcal H)=\mathbb E_\sigma\Big[\max_{h\in\mathcal H}\frac1n\sum_{i=1}^n\sigma_i\,h(x_i)\Big]\in[0,1].
$$

Si $\mathcal H$ contient une hypothèse pour chaque étiquetage, elle s'accorde à chaque vecteur de bruit, le max vaut $1$, et la complexité est maximale. Une seule hypothèse donne environ $0$, puisque des signes aléatoires se moyennent à zéro. Pour une classe finie, le lemme de Massart donne $\hat{\mathfrak R}_n\le\sqrt{2\ln M/n}$.

Une borne de généralisation typique (forme classique ; les constantes diffèrent selon la source) : avec une probabilité d'au moins $1-\delta$, pour tout $h$,

$$
R(h)\le\hat R_n(h)+\mathfrak R_n(\mathcal H)+\sqrt{\frac{\ln(1/\delta)}{2n}}
$$

pour des classes à valeurs $\pm1$ et la perte 0-1, où $\mathfrak R_n$ est l'espérance de la quantité empirique sur les échantillons. Les manuels qui définissent la complexité sur la classe de pertes écrivent plutôt $2\mathfrak R_n$ avec un $\mathfrak R_n$ différent, donc vérifie toujours la définition avant de comparer des nombres. La complexité de Rademacher est *calculable à partir des données* (on peut donc l'estimer, comme ci-dessous) et s'applique aux classes à valeurs réelles comme les modèles linéaires à norme bornée, où elle varie comme $\|w\|\cdot\max\|x\|/\sqrt n$. Ce contrôle par la norme est la raison théorique pour laquelle la régularisation fonctionne même quand le nombre de paramètres est grand.

## Exemple détaillé

Tous les nombres sont reproduits par le code des exercices (avec graine).

1. **Hoeffding.** $n=100,\ \varepsilon=0.1$ : $2e^{-2}=0.2707$. Pour que $\varepsilon=0.05$ tienne uniformément pour $M=1000$ hypothèses à $\delta=0.05$, il faut $n\ge\ln(40000)/(2\cdot0.05^2)=2120$.
2. **Borne de l'union contre la réalité.** $M=200$ classifieurs linéaires aléatoires dans $\mathbb R^5$, étiquettes issues d'une règle linéaire cachée avec bruit, $\delta=0.05$, 200 ensembles d'entraînement pour chaque $n$ :

| $n$ | borne de l'union $\varepsilon$ | écart max moyen sur $\mathcal H$ | 95e percentile de l'écart max | écart moyen de l'ERM |
|---|---|---|---|---|
| 50 | 0.300 | 0.175 | 0.235 | 0.072 |
| 200 | 0.150 | 0.087 | 0.123 | 0.026 |
| 1000 | 0.067 | 0.040 | 0.056 | 0.007 |

   La borne tient dans chaque ligne (le 95e percentile est en dessous) et diminue comme $1/\sqrt n$ tout comme l'écart réel, mais elle est prudente : elle couvre la pire des $M$ hypothèses avec 95 % de confiance. L'écart du $\hat h$ choisi est plusieurs fois plus petit que le pire cas, car l'ERM ne choisit pas l'hypothèse au plus grand écart.
3. **Dimension VC par force brute.** Intervalles : deux points quelconques pulvérisés, 3 points donnent 7 étiquetages sur 8, donc $d_{VC}=2$. Demi-plans : le triangle donne 8 sur 8 ; le carré donne 14 sur 16 ; donc $d_{VC}=3$ dans le plan.
4. **Complexité de Rademacher.** Seuils sur $n$ points triés (les deux orientations, $2n+2$ hypothèses). $\hat{\mathfrak R}_n$ estimée avec 2000 vecteurs de signes : $n=20$ : $0.391$ ; $n=100$ : $0.186$ ; $n=400$ : $0.095$ ; la borne de Massart donne $0.611,\ 0.326,\ 0.183$. La complexité diminue comme $1/\sqrt n$ (une classe de dimension VC 2 se comporte comme $\sqrt{\ln n/n}$). La classe des $2^6$ étiquetages de 6 points a exactement la complexité $1.0$.

## Implémentation

```python
import numpy as np

def union_eps(n, M, delta):
    return np.sqrt(np.log(2 * M / delta) / (2 * n))        # uniform deviation width

def labelings_halfplanes(P, n_angles=720):
    labs = set()
    for th in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        w = np.array([np.cos(th), np.sin(th)])
        pr = P @ w                                         # (n,)
        s = np.sort(pr)
        for t in np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]]):
            labs.add(tuple((pr > t).astype(int)))
    return labs                                            # shattered iff len == 2**n

def rademacher(H, reps, seed):                             # H: (|H|, n) of +-1
    sig = np.random.default_rng(seed).choice([-1, 1], size=(reps, H.shape[1]))
    return (sig @ H.T).max(axis=1).mean() / H.shape[1]
```

## Formes

- Prédictions de $M$ hypothèses sur $n$ points : $(n,M)$ ; risques empiriques `mean(axis=0)` : $(M,)$.
- Ensemble de points $P$ : $(n,2)$ ; les étiquetages sont des tuples de longueur $n$, au plus $2^n$ d'entre eux.
- Matrice d'hypothèses $H$ : $(|\mathcal H|,n)$ de $\pm1$ ; matrice de signes $\sigma$ : $(\text{reps},n)$ ; produit $\sigma H^\top$ : $(\text{reps},|\mathcal H|)$.

## No free lunch, réseaux profonds et minimisation du risque structurel

**No free lunch (énoncé).** En moyenne sur toutes les fonctions cibles possibles d'un domaine fini, aucun algorithme d'apprentissage n'en bat un autre : pour tout algorithme, il existe des distributions sur lesquelles il se comporte mal. Apprendre exige donc des *hypothèses* (une classe restreinte, un a priori, de la régularité, de la simplicité). Les bornes ci-dessus en sont la forme quantitative : une petite classe avec un bon ajustement généralise, une classe qui peut tout ajuster ne promet rien.

**Minimisation du risque structurel (SRM).** Prends des classes emboîtées $\mathcal H_1\subset\mathcal H_2\subset\dots$ de capacité croissante (polynômes de degré 1, 2, 3, ... ; arbres de profondeur croissante). L'erreur d'entraînement diminue avec l'indice, la pénalité de complexité augmente. Choisis la classe qui minimise *l'erreur d'entraînement plus la pénalité*. C'est ce que suggèrent les bornes, et c'est la courbe en U de [[cml.03]] en formules. La **régularisation** en est la version continue : minimiser $\hat R_n(h)+\lambda\,\Omega(h)$, par ex. $\|w\|^2$ (math for ML Sec. 8.2.3 la présente comme une pénalité sur la complexité de la solution). Elle restreint la classe effective à une boule $\|w\|\le B$, dont la complexité de Rademacher varie avec $B$.

**Pourquoi les réseaux profonds défient les bornes.** Un réseau moderne peut avoir bien plus de paramètres que de points d'entraînement ; sa dimension VC croît au moins avec le nombre de poids, donc la borne VC est vide (supérieure à 1). Pourtant ces réseaux peuvent généraliser, et ils peuvent aussi ajuster parfaitement des étiquettes aléatoires (montré dans la littérature, pas dans mes sources), ce qui signifie que la classe a une grande capacité et que la borne *uniforme* sur toute la classe ne peut pas expliquer le bon comportement. La borne porte sur le pire cas de la classe, alors que l'entraînement par SGD depuis une petite initialisation atteint des solutions particulières et bien élevées. Les explications candidates (biais implicite de la SGD, bornes par la norme et la marge, minima plats, bornes PAC-Bayes, la courbe de « double descente » de l'erreur de test quand la capacité dépasse le point d'interpolation) font l'objet de recherches actives ; aucune n'est tranchée et je n'en présente aucune comme la réponse. En pratique : la théorie te dit quels *boutons* comptent (taille des données, capacité, norme, marge) ; les données de validation décident toujours.

## Idées reçues

- « Plus de paramètres veut dire une dimension VC plus élevée. » Pas toujours : $\sin(ax)$ a un paramètre et une dimension VC infinie ; inversement, beaucoup de paramètres fortement contraints peuvent ajouter peu.
- « Une dimension VC $d$ veut dire que $d$ points quelconques peuvent être pulvérisés. » Seulement une certaine disposition de $d$ points ; trois points alignés ne sont pas pulvérisés par les demi-plans.
- « Une borne qui tient doit être serrée. » Les bornes du pire cas sont lâches ; le tableau ci-dessus montre un facteur de seulement 1,2 à 1,3 environ entre la borne et le 95e percentile même dans ce cas fini favorable, l'écart moyen vaut environ la moitié de la borne, et les bornes VC sont en général bien plus lâches.
- « La borne de l'union dit que $M$ hypothèses coûtent un facteur $M$. » Le coût n'est que de $\sqrt{\ln M}$ dans la largeur.
- « Une faible erreur d'entraînement plus un petit ensemble de test prouve que le modèle est bon. » Les bornes supposent que les points de test sont neufs. Réutiliser un ensemble de test pour choisir entre des modèles réintroduit une sélection, le même mécanisme que l'histoire du million de restaurants.
- « La théorie dit que l'apprentissage profond ne peut pas marcher. » Les bornes *uniformes* classiques y sont vides ; cela ne veut pas dire que les modèles ne généralisent pas.

## Points clés

- Risque vrai $=$ risque empirique $+$ écart ; l'ERM rend l'écart optimiste parce que l'hypothèse a été choisie sur les données.
- Classe finie : écart $\le\sqrt{\ln(2M/\delta)/(2n)}$ pour tous les $h$ à la fois (Hoeffding plus borne de l'union).
- Classe infinie : remplace $\ln M$ par la dimension VC fois $\ln n$ ; la dimension VC concerne la pulvérisation, et la trouver demande une construction et un argument d'impossibilité.
- La complexité de Rademacher mesure la capacité à ajuster du bruit et peut s'estimer à partir de l'échantillon.
- No free lunch : les hypothèses sont inévitables. La SRM et la régularisation arbitrent entre ajustement et capacité. Les réseaux profonds sortent de ce qu'expliquent les bornes uniformes.

## Explique-le toi-même

1. Pourquoi le risque empirique de l'hypothèse *sélectionnée* est-il une estimation optimiste de son risque vrai, alors que celui d'une hypothèse fixée à l'avance est sans biais ?
2. Explique avec tes mots pourquoi la dimension VC des intervalles sur une droite vaut 2 et non 3.

## Pour aller plus loin

- math for ML, chap. 8 (lu ici pour le risque empirique et la régularisation).
- Écrits à partir de connaissances générales, pas de la bibliothèque : Shalev-Shwartz et Ben-David, *Understanding Machine Learning* (classes finies, VC, Rademacher, no free lunch) ; Mohri et al., *Foundations of Machine Learning* ; Zhang et al., « Understanding deep learning requires rethinking generalization » (2017). Consulte-les pour les énoncés exacts et les constantes.

## Critères de maîtrise

- Dériver $\varepsilon=\sqrt{\ln(2M/\delta)/(2n)}$ à partir de Hoeffding et de la borne de l'union, et le calculer.
- Comparer la borne à un écart expérimental et expliquer pourquoi la borne est lâche.
- Prouver $d_{VC}=2$ pour les intervalles et démontrer $d_{VC}\ge3$, $<4$ pour les demi-plans dans le plan.
- Estimer une complexité de Rademacher par Monte-Carlo et interpréter les valeurs $0$ et $1$.
- Énoncer le no-free-lunch et expliquer ce que font la SRM et la régularisation à ce sujet.
