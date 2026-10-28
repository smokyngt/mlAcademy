---
id: math.10
title: "Théorie de l'information, entropie, entropie croisée et divergence KL"
summary: "Mesure la surprise en bits, vois l'entropie comme la longueur moyenne minimale d'un message, et démontre pourquoi la perte d'entropie croisée est du maximum de vraisemblance et pourquoi la divergence KL n'est jamais négative."
minutes: 80
teaches: [info.entropy]
requires: [prob.rv, alg.exp-log]
sources:
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 1.6 Information Theory (PDF p.68-78): information content h(x) = -log p(x), entropy, the 8-state example with code lengths, noiseless coding theorem, maximum entropy and the Gaussian, conditional entropy, relative entropy / KL, Jensen's inequality and KL >= 0, KL minimisation = maximum likelihood, mutual information. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/09-information-theory/docs/en.md (first 11000 characters): surprise table, entropy, cross-entropy as classification loss, cross-entropy = entropy + KL, mutual information, label smoothing, bits vs nats, perplexity. READ."
  - "math/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, Sec. 2.1-2.2 (PDF p.88-98): the Bernoulli/multinomial likelihood and the Lagrange-multiplier derivation mu_k = m_k / N that links frequencies to maximum likelihood. READ."
---

## Où cela s'inscrit

Chaque fois que tu entraînes un classifieur, tu appelles une fonction nommée `CrossEntropyLoss`. Chaque article sur les modèles de langage rapporte la *perplexité*. Les autoencodeurs variationnels, la distillation de connaissances et le RLHF minimisent tous une *divergence KL*. Cela ressemble à trois idées différentes. C'est une seule idée vue sous trois angles : **combien de bits te faut-il pour décrire ce qui s'est réellement passé, si tu as construit ton code pour une autre croyance sur le monde ?**

Cette leçon vient juste après les probabilités (leçon 8) et la gaussienne (leçon précédente) : tu y as appris que la log-vraisemblance de données indépendantes est une somme de termes $\ln p$. Ici nous donnons un sens à ces termes $-\ln p$ (la surprise), nous en faisons la moyenne (l'entropie), et nous comparons deux distributions (la KL). Le bénéfice est une réponse nette à « pourquoi cette perte ? » : l'entropie croisée n'est pas une heuristique, c'est du maximum de vraisemblance déguisé.

## L'idée en mots simples

- **La surprise.** Quelqu'un te dit « le soleil s'est levé aujourd'hui ». Tu n'apprends rien. Quelqu'un te dit « tu as gagné à la loterie ». Tu apprends beaucoup. L'information est le *degré de surprise* : les événements rares en apportent beaucoup, les événements certains n'en apportent aucune. Deux surprises indépendantes doivent simplement s'additionner (apprendre deux nouvelles sans rapport vaut la somme de chacune). La seule fonction qui transforme « multiplier des probabilités » en « additionner des surprises » est un **logarithme**, et le signe moins rend les événements rares grands : surprise $=-\log p$.
- **Entropie = surprise moyenne = message moyen le plus court.** Imagine que tu doives envoyer le résultat d'une expérience répétée sur un fil fait de 0 et de 1. Si les huit résultats sont équiprobables, il te faut 3 bits pour chacun. Mais si un résultat arrive la moitié du temps, donne-lui un nom d'1 bit ; donne au suivant un nom de 2 bits, et ainsi de suite, en gardant les noms longs pour les cas rares. Le message *moyen* devient plus court. L'entropie est cette longueur moyenne minimale incompressible : tu ne peux pas compresser en dessous.
- **Entropie croisée = ce que tu paies quand ta croyance est fausse.** Tu conçois le livre de codes d'après ce que tu *crois* ($q$), mais la nature produit des résultats selon la vérité ($p$). Tes mots de code ont les mauvaises longueurs pour les fréquences réelles, donc le message moyen est plus long. Cette moyenne est l'entropie croisée.
- **Divergence KL = le prix de l'erreur.** La longueur supplémentaire par rapport au meilleur code possible. Elle est nulle seulement quand ta croyance est exactement la vérité, jamais négative, et non symétrique : être sûr de la mauvaise chose et être vague sur la bonne sont deux erreurs différentes.
- La **perplexité** retransforme une surprise moyenne en un nombre : « aussi perdu que si tu choisissais uniformément parmi ce nombre d'options ».

## Définitions

### Contenu informationnel et entropie

Pour une variable discrète $x$ de probabilité $p(x)$ (Bishop 1.92, 1.93) :

$$
h(x)=-\log p(x),\qquad H[p]=\mathbb E_{x\sim p}[h(x)]=-\sum_x p(x)\log p(x).
$$

Les logarithmes en base 2 donnent des **bits**, les logarithmes naturels donnent des **nats** ; $1\text{ nat}=1/\ln 2\approx1.443$ bits. Les bibliothèques de machine learning utilisent les nats. Nous adoptons la convention $0\log0=0$.

La première justification que donne Bishop pour le logarithme : si $x$ et $y$ sont indépendantes, alors $p(x,y)=p(x)p(y)$ et l'information doit s'additionner, $h(x,y)=h(x)+h(y)$. Une fonction qui transforme les produits en sommes est un log.

Propriétés : $H\ge0$, avec $H=0$ exactement quand un résultat a une probabilité 1. Pour $M$ états, $H\le\ln M$, avec égalité pour la distribution uniforme (Bishop le démontre avec un multiplicateur de Lagrange, une technique que tu rencontreras dans la leçon 12). Parmi les densités continues de moyenne et de variance données, la gaussienne a la plus grande entropie (différentielle), $\tfrac12(1+\ln 2\pi\sigma^2)$, ce qui est une raison de plus d'en faire le modèle de bruit par défaut.

### Exemple : la variable à 8 états (Bishop, PDF p.70)

Huit états de probabilités $(\tfrac12,\tfrac14,\tfrac18,\tfrac1{16},\tfrac1{64},\tfrac1{64},\tfrac1{64},\tfrac1{64})$. Entropie :

$$
H=\tfrac12\cdot1+\tfrac14\cdot2+\tfrac18\cdot3+\tfrac1{16}\cdot4+4\cdot\tfrac1{64}\cdot6=0.5+0.5+0.375+0.25+0.375=2\text{ bits.}
$$

Un code plat de 3 bits coûte 3 bits par symbole. Utilise les mots de code $0,\,10,\,110,\,1110,\,111100,\,111101,\,111110,\,111111$ de longueurs $1,2,3,4,6,6,6,6$ : la longueur moyenne est à nouveau de $2$ bits. Chaque longueur est exactement égale à $-\log_2 p$. Aucun code de longueur moyenne plus courte ne peut exister (théorème de codage sans bruit de Shannon), et le code est *sans préfixe*, donc un flux de bits comme `11001110` se décode de manière unique (c, a, d).

### Entropie croisée

Si la vérité est $p$ mais que nous concevons le code pour $q$, un état $x$ reçoit un mot de longueur $-\log q(x)$, et puisque les états apparaissent avec la fréquence $p(x)$, la longueur moyenne est

$$
H(p,q)=-\sum_x p(x)\log q(x).
$$

### Divergence KL et sa décomposition

Le surcoût par rapport au code optimal (Bishop 1.113) :

$$
\mathrm{KL}(p\,\|\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}=H(p,q)-H(p).
$$

Donc **entropie croisée = entropie + KL**. Comme $H(p)$ ne dépend pas du modèle $q$, minimiser l'entropie croisée sur $q$ revient à minimiser la divergence KL.

### Inégalité de Gibbs : $\mathrm{KL}\ge0$

Affirmation : $\mathrm{KL}(p\|q)\ge0$, avec égalité si et seulement si $p=q$.

*Étape 1, un fait sur la tangente.* Pour tout $x>0$, $\ln x\le x-1$, avec égalité seulement en $x=1$. Raison : $g(x)=x-1-\ln x$ a pour dérivée $g'(x)=1-1/x$, négative pour $x<1$ et positive pour $x>1$, donc $g$ atteint son unique minimum en $x=1$ où $g(1)=0$.

*Étape 2.* Applique-la à $x=q(i)/p(i)$ pour chaque état tel que $p(i)>0$ :

$$
-\mathrm{KL}(p\|q)=\sum_i p(i)\ln\frac{q(i)}{p(i)}\ \le\ \sum_i p(i)\Big(\frac{q(i)}{p(i)}-1\Big)=\sum_i q(i)-\sum_i p(i)\ \le\ 1-1=0 .
$$

(La somme de $q$ sur les états où $p>0$ vaut au plus 1.) Donc $\mathrm{KL}\ge0$. L'égalité exige $q(i)/p(i)=1$ pour tout $i$, c'est-à-dire $p=q$. Bishop démontre le même fait avec l'inégalité de Jensen pour la fonction convexe $-\ln x$ ; les deux preuves reposent sur la même idée (une courbe convexe se situe au-dessus de sa tangente). Puisque $\mathrm{KL}\ge0$, $H(p,q)\ge H(p)$ : aucun code construit pour la mauvaise distribution ne peut battre le bon.

### Asymétrie

En général $\mathrm{KL}(p\|q)\ne\mathrm{KL}(q\|p)$, donc la KL n'est pas une distance. Elle est grande quand $q$ donne une probabilité (presque) nulle à quelque chose que $p$ juge probable : tu es alors infiniment surpris. C'est pourquoi le sens compte en pratique (KL directe contre KL inverse en inférence variationnelle).

### De l'entropie croisée à la perte de classification, puis au maximum de vraisemblance

Soit l'étiquette d'un exemple un vecteur one-hot $p$ (probabilité 1 sur la vraie classe $c$) et la sortie du modèle $q=\operatorname{softmax}(\mathbf z)$. Alors

$$
H(p,q)=-\sum_k p_k\ln q_k=-\ln q_c .
$$

Sur $N$ exemples indépendants de vraies classes $c_n$, la vraisemblance est $\prod_n q^{(n)}_{c_n}$ et la log-vraisemblance négative est $-\sum_n\ln q^{(n)}_{c_n}$ : exactement $N$ fois l'entropie croisée moyenne. Donc **minimiser la perte d'entropie croisée = maximiser la vraisemblance des étiquettes**.

La même conclusion dans l'autre sens (Bishop, PDF p.77) : soit $p$ la distribution inconnue des données et $q(x\mid\theta)$ le modèle. Remplace l'espérance sous $p$ par une moyenne sur des échantillons :

$$
\mathrm{KL}(p\|q)\approx\frac1N\sum_{n=1}^N\big\{-\ln q(x_n\mid\theta)+\ln p(x_n)\big\}.
$$

Le second terme ne contient pas $\theta$, et le premier est la log-vraisemblance négative. **Minimiser la KL = maximum de vraisemblance.** Le cas Bernoulli/multinomial boucle la boucle : maximiser $\sum_k m_k\ln\mu_k$ avec $\sum_k\mu_k=1$ (multiplicateur de Lagrange) donne $\mu_k=m_k/N$, les fréquences observées (Bishop 2.33), l'unique minimiseur de $\mathrm{KL}(\hat p\|q)$.

### Information mutuelle (brièvement)

Pour deux variables, à quelle distance se trouve la loi jointe $p(x,y)$ du produit des marginales $p(x)p(y)$ (ce que donnerait l'indépendance) ? Bishop 1.120 :

$$
I[x,y]=\mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)=H[x]-H[x\mid y]=H[x]+H[y]-H[x,y]\ \ge 0 .
$$

Elle est nulle si et seulement si $x$ et $y$ sont indépendantes, et c'est la réduction de l'incertitude sur $x$ une fois que l'on te donne $y$. Contrairement à la corrélation, elle détecte n'importe quel type de dépendance, ce qui explique son usage pour classer des variables explicatives.

### Perplexité

La perplexité est l'exponentielle de l'entropie croisée moyenne par token : $\mathrm{PPL}=\exp\big(-\frac1N\sum_n\ln q(x_n)\big)$ en nats (ou $2^{\text{bits}}$). Un modèle qui répartit uniformément sa probabilité sur $K$ options a une perplexité exactement égale à $K$, donc une perplexité de $50$ signifie « aussi incertain qu'un choix équitable parmi 50 tokens ». Plus bas est mieux.

## Exemple détaillé

Chaque nombre est reproduit par le code de l'exercice.

**1. Surprise.** Une face de pièce équilibrée : $1$ bit. Une face de dé : $-\log_2(1/6)=2.585$ bits. Un événement de probabilité $1/1000$ : $9.966$ bits.

**2. Entropie et longueur de code.** La distribution à 8 états ci-dessus : $H=2$ bits. L'uniforme sur 8 états : $3$ bits. Une pièce avec $P(\text{heads})=0.99$ : $0.0808$ bits.

**3. Entropie croisée et KL.** $p=(\tfrac12,\tfrac14,\tfrac18,\tfrac18)$ et $q$ uniforme sur 4 résultats. $H(p)=1.75$ bits. Un code construit pour $q$ utilise 2 bits pour chaque résultat, donc $H(p,q)=2$ bits et $\mathrm{KL}(p\|q)=2-1.75=0.25$ bits. Avec la mauvaise hypothèse plus tranchée $q'=(0.7,0.1,0.1,0.1)$ : $H(p,q')=1.918$ bits, $\mathrm{KL}(p\|q')=0.168$ bits.

**4. Asymétrie.** $p=(0.9,0.1)$, $q=(0.5,0.5)$. $\mathrm{KL}(p\|q)=0.531$ bits mais $\mathrm{KL}(q\|p)=0.737$ bits.

**5. Entropie croisée softmax.** Les logits $\mathbf z=(2,1,0.1)$ donnent les probabilités $(0.659,0.242,0.099)$ (décale par $\max z$ avant d'exponentier). Si la vraie classe est la première, la perte vaut $-\ln0.659=0.417$ nats $=0.602$ bits.

**6. Perplexité.** Un modèle attribue les probabilités $(0.5,0.25,0.125,0.125)$ aux quatre tokens corrects. NLL moyenne $=1.5596$ nats, perplexité $=e^{1.5596}=4.757$. Un modèle uniforme sur 50 tokens a une perplexité exactement égale à $50$.

**7. Information mutuelle.** Loi jointe de deux variables binaires : lignes $(0.40,0.10)$ et $(0.05,0.45)$. Marginales $(0.5,0.5)$ et $(0.45,0.55)$. $H[x]=1$, $H[y]=0.9928$, $H[x,y]=1.5955$, donc $I=1+0.9928-1.5955=0.3973$ bits, ce qui égale $\mathrm{KL}(\text{joint}\,\|\,\text{product})$.

## Implémentation

```python
import numpy as np

def entropy(p, base=2):
    p = np.asarray(p, float); p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(base)

def cross_entropy(p, q, base=2):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = p > 0
    return -(p[m] * np.log(q[m])).sum() / np.log(base)

def kl(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

def softmax_xent(z, c):                       # z: (K,), c: true class index
    z = z - z.max()                           # numerical stability
    return -(z[c] - np.log(np.exp(z).sum()))   # -log softmax(z)[c], in nats

p = np.array([.5, .25, .125, .125]); q = np.ones(4) / 4
print(entropy(p), cross_entropy(p, q), kl(p, q))   # 1.75 2.0 0.25
```

## Formes

Une distribution sur $K$ classes est un vecteur `(K,)` de somme 1. Un lot de sorties du modèle est `(B, K)` (les lignes somment à 1 après softmax), les étiquettes entières sont `(B,)`, et l'entropie croisée du lot est `-log(q[arange(B), labels]).mean()`, un scalaire. Une loi jointe de deux variables est une matrice `(Kx, Ky)` ; les marginales sont ses sommes par lignes et par colonnes.

## Idées fausses courantes

- **La KL est une distance.** Non : elle n'est pas symétrique et viole l'inégalité triangulaire.
- **L'entropie croisée est inférieure à l'entropie pour un bon modèle.** Jamais : $H(p,q)\ge H(p)$, avec égalité seulement pour $q=p$. Une perte nulle pour des étiquettes one-hot n'est atteignable qu'en poussant les logits vers l'infini.
- **Les bits et les nats sont des grandeurs différentes.** C'est la même grandeur, dans une unité différente (facteur $\ln2$).
- **$\log0$ est inoffensif.** Si $q(x)=0$ là où $p(x)>0$, l'entropie croisée et la KL sont infinies ; borne les probabilités ou travaille en espace logarithmique.
- **L'information mutuelle est la corrélation.** La corrélation ne voit que la dépendance linéaire ; l'information mutuelle les voit toutes.

## Points clés à retenir

- La surprise est $-\log p$ ; l'entropie $H=-\sum p\log p$ est la surprise moyenne et la longueur de code moyenne optimale.
- $H(p,q)=H(p)+\mathrm{KL}(p\|q)$, et $\mathrm{KL}\ge0$ (preuve à partir de $\ln x\le x-1$).
- Avec des étiquettes one-hot, l'entropie croisée vaut $-\ln$ de la probabilité donnée à la vraie classe ; la minimiser, c'est du maximum de vraisemblance, et de façon équivalente minimiser la KL vers la distribution des données.
- L'information mutuelle est la KL entre la loi jointe et le produit des marginales ; la perplexité est l'$\exp$ de l'entropie croisée.

## Explique-le avec tes mots

1. Explique à un ami, avec un argument de longueur de message, pourquoi une distribution asymétrique a une entropie plus faible qu'une uniforme.
2. Pourquoi minimiser l'entropie croisée d'un modèle donne-t-il la même réponse que minimiser la KL vers la distribution des données, alors que nous ne pouvons pas calculer la distribution des données ?

## Pour aller plus loin

- Bishop, *PRML*, Sec. 1.6 (PDF p.68-78), y compris l'argument de l'entropie comme désordre en PDF p.70-72 et l'entropie différentielle de la gaussienne.
- *Mathematics for Machine Learning*, remarque sur les divergences et la géométrie de l'information à la fin de la Sec. 6.4 (PDF p.203).

## Critères de maîtrise

Tu sais calculer à la main l'entropie, l'entropie croisée et la KL de petites distributions, les expliquer comme des longueurs de code, démontrer $\mathrm{KL}\ge0$ à partir de $\ln x\le x-1$, montrer que l'entropie croisée softmax est une log-vraisemblance négative, et convertir une perte en nats en bits et en perplexité.
