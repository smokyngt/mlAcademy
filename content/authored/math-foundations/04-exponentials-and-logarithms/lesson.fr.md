---
id: mf.04
title: "Exponentielles et logarithmes depuis zéro"
summary: "De la multiplication répétée au nombre e, puis les logarithmes comme bouton « annuler », leurs lois, le changement de base, la croissance composée, et pourquoi tout modèle probabiliste en ML s'entraîne à travers des logs."
minutes: 100
teaches: [mf.explog]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (opening to the derivative/partial-derivative sections): where exp and log appear (sigmoid, cross-entropy) and the limit idea behind rates of change. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md (The Problem, the exp/log overflow and underflow section, Log-Sum-Exp trick and its derivation, stable cross-entropy): ML uses of exp and log and the max-subtraction trick. READ."
---

## Où cela s'inscrit

Les probabilités se multiplient, la croissance se multiplie, et les taux d'apprentissage décroissent par multiplication. Partout où des quantités sont multipliées encore et encore, des exponentielles apparaissent ; partout où il faut défaire cela, ou transformer un long produit en une somme maniable, des logarithmes apparaissent. En apprentissage automatique, tu verras constamment $\exp$ (dans softmax et sigmoïde) et $\ln$ (dans l'entropie croisée, la log-vraisemblance, la divergence KL, la perplexité). La perte d'entraînement d'un classifieur est littéralement « moins le log de la probabilité que tu as donnée à la bonne réponse ».

Cette leçon construit les deux idées à partir de rien d'autre que la multiplication répétée et ne demande aucune leçon antérieure (nous utilisons les règles de puissances revues dans `mf.02` ; si tu te sens fragile, parcours d'abord sa section « Puissances et racines », mais tout le nécessaire est réénoncé ici).

## L'idée en mots simples

Les **exponentielles** répondent à « que se passe-t-il si je continue de multiplier par le même nombre ? ». Plie une feuille de papier : chaque pli double l'épaisseur. Dix plis ne donnent pas 10 fois l'épaisseur mais 1024 fois. Une rumeur qui double son audience chaque jour, de l'argent avec des intérêts, un virus sans contrôle, un taux d'apprentissage divisé par deux à chaque epoch : tout cela est de la multiplication répétée. Le sentiment clé : la croissance est *lente au début, puis explosive*.

Les **logarithmes** répondent à la question inverse : « combien de multiplications cela a-t-il pris ? ». Si quelque chose a doublé jusqu'à atteindre 1024 fois son départ, le log (en base 2) dit 10. Un logarithme compte *combien de fois* tu as multiplié ; il mesure les choses sur une échelle d'**ordres de grandeur**. C'est pourquoi les échelles de séismes et de son utilisent des logs : elles compressent d'énormes plages (de 1 à un milliard) en petits nombres (de 0 à 9).

Une troisième image : une **règle à calcul**. Comme « compter les multiplications » transforme la *multiplication* de deux nombres en l'*addition* de leurs comptes, les logs transforment les produits en sommes. Tout ce qui est utile à propos des logs découle de cette seule astuce.

## Exposants : multiplication répétée

$a^n$ signifie $n$ copies de $a$ multipliées : $2^5=2\cdot2\cdot2\cdot2\cdot2=32$.

**Lois des exposants, dérivées en comptant les copies** (pour $a>0$) :

| loi | pourquoi |
|---|---|
| $a^m\cdot a^n=a^{m+n}$ | $m$ copies puis $n$ copies de plus font $m+n$ copies |
| $a^m/a^n=a^{m-n}$ | on simplifie $n$ copies du haut |
| $(a^m)^n=a^{mn}$ | $n$ groupes de $m$ copies |
| $(ab)^n=a^nb^n$ | on regroupe les facteurs |
| $a^0=1$ | imposé : $a^m\cdot a^0$ doit valoir $a^{m}$ |
| $a^{-n}=1/a^n$ | imposé : $a^n\cdot a^{-n}=a^0=1$ |
| $a^{1/n}=\sqrt[n]{a}$ | imposé : $(a^{1/n})^n=a^1$ |

De petits exemples : $2^3\cdot2^4=8\cdot16=128=2^7$. $(2^3)^2=64=2^6$. $2^{-2}=\tfrac14$. $8^{1/3}=2$. $9^{3/2}=(\sqrt9)^3=27$. Avec une base supérieure à 1, plus l'exposant est grand, plus le résultat est grand ; avec une base comprise entre 0 et 1 (comme $\tfrac12$), la multiplication répétée **rétrécit** : $(\tfrac12)^n$ est une décroissance exponentielle.

## Croissance composée

Si une quantité $P$ augmente d'une fraction $r$ à chaque période (0.05 pour 5 %), une période la multiplie par $(1+r)$. Après $t$ périodes :

$$
P(1+r)^t .
$$

Exemple : 1000 à 5 % pendant 10 ans : $1000\cdot1.05^{10}=1628.89$. Pas $1500$ : la croissance se compose parce que les intérêts rapportent des intérêts.

**Temps de doublement.** Quand a-t-on $(1+r)^t=2$ ? Il nous faut l'exposant $t$, une inconnue *dans l'exposant*. L'algèbre ordinaire ne peut pas l'isoler ; c'est exactement le trou que comblent les logarithmes. Réponse (dérivée plus bas) : $t=\ln2/\ln(1+r)$. À 7 % : $10.24$ périodes. La règle empirique « divise 70 par le pourcentage » donne 10, c'est proche.

## Le nombre e

Supposons qu'un taux annuel de 100 % soit découpé en $n$ étapes de capitalisation par an, chacune à $1/n$. Après un an, 1 devient $(1+\tfrac1n)^n$.

| $n$ | $(1+1/n)^n$ |
|---|---|
| 1 | 2.0000 |
| 10 | 2.5937 |
| 1000 | 2.7169 |
| 1 000 000 | 2.71828 |

Les valeurs n'explosent pas ; elles se rapprochent lentement d'un nombre fixe, $e\approx2.718281828$. Nous définissons $e=\lim_{n\to\infty}(1+1/n)^n$ (la « limite » est précisée dans `mf.07`). La capitalisation continue au taux $r$ pendant le temps $t$ donne $Pe^{rt}$. Par exemple, 5 % pendant 10 ans ($rt=0.5$) donne $1000\,e^{0.5}=1648.72$, légèrement plus que les $1628.89$ de la capitalisation annuelle. Une autre formule pour $e$ est la somme $1+1+\tfrac12+\tfrac16+\tfrac1{24}+\dots=\sum_k1/k!$ ; onze termes donnent déjà $2.718281801$ (l'idée de série revient dans `mf.06`).

Pourquoi $e$ est-il spécial ? $e^x$ est l'unique exponentielle dont la pente en chaque point est égale à sa propre hauteur. Cela en fait le langage naturel de tout ce dont le taux de croissance est proportionnel à la taille, et c'est pourquoi les dérivées de $e^x$ sont si simples (une leçon ultérieure). En ML, $e^x$ transforme n'importe quel nombre réel, positif ou négatif, en un nombre positif : parfait pour construire des probabilités (softmax) et des échelles positives.

## Logarithmes : le bouton « annuler »

**Définition.** Pour une base $b>0$, $b\neq1$ et $x>0$ :

$$
\log_b x = y \quad\Longleftrightarrow\quad b^y=x .
$$

Lis : « $\log_b x$ est l'exposant que tu mets sur $b$ pour obtenir $x$ ». Donc $\log_2 32=5$ parce que $2^5=32$ ; $\log_{10}1000=3$ ; $\log_b1=0$ (puisque $b^0=1$) ; $\log_bb=1$. Le **logarithme naturel** est en base $e$ : $\ln x=\log_ex$. Comme les logs sont des chercheurs d'exposants, ils défont les exponentielles :

$$
\ln(e^x)=x,\qquad e^{\ln x}=x\ (x>0),\qquad \log_b(b^x)=x .
$$

**Domaine.** $b^y$ est toujours positif, donc $\log_bx$ n'existe que pour $x>0$. $\ln0$ n'est pas un nombre (il plonge vers $-\infty$ quand $x\to0^+$) et $\ln$ d'un nombre négatif n'est pas défini sur les réels. C'est pourquoi le code qui calcule `log(0.0)` renvoie `-inf` et `log(-1.0)` renvoie `nan` ; le code de ML se protège avec une minuscule constante comme `log(x + 1e-8)` ou utilise des formules stables.

**Forme.** $\ln x$ monte, mais de plus en plus lentement : $\ln1=0$, $\ln2=0.693$, $\ln10=2.303$, $\ln100=4.605$. Multiplier l'entrée par 10 n'ajoute que 2.303. Les probabilités entre 0 et 1 sont envoyées sur des nombres *négatifs* ($\ln0.5=-0.693$), et plus la probabilité est petite, plus c'est négatif.

## Lois des logs (dérivations)

Soit $\log_ba=m$ et $\log_bc=n$, c'est-à-dire $a=b^m$ et $c=b^n$.

1. **Produit.** $ac=b^mb^n=b^{m+n}$, donc $\log_b(ac)=m+n$ :

$$
\log_b(ac)=\log_ba+\log_bc .
$$

2. **Quotient.** $a/c=b^{m-n}$, donc $\log_b(a/c)=\log_ba-\log_bc$.
3. **Puissance.** $a^k=(b^m)^k=b^{mk}$, donc $\log_b(a^k)=k\log_ba$.
4. **Changement de base.** Soit $y=\log_bx$, c'est-à-dire $b^y=x$. Prends $\ln$ des deux côtés et utilise la loi 3 : $y\ln b=\ln x$, donc

$$
\log_bx=\frac{\ln x}{\ln b} .
$$

Exemples : $\log_28+\log_24=3+2=5=\log_232$ (puisque $8\cdot4=32$). $\log_2 10=\ln10/\ln2=3.3219$. Maintenant le temps de doublement : $(1+r)^t=2\Rightarrow t\ln(1+r)=\ln2\Rightarrow t=\ln2/\ln(1+r)$, en utilisant la loi de la puissance.

Attention : $\log(a+c)$ n'a **aucune** simplification. $\log(a+c)\ne\log a+\log c$.

## Pourquoi les logs apparaissent en ML

**1. Les produits de probabilités deviennent des sommes.** Si un modèle attribue des probabilités $p_1,\dots,p_N$ à $N$ exemples d'entraînement indépendants, la *vraisemblance* est $\prod_ip_i$. Avec 2000 lancers de pièce de probabilité $0.5$ chacun, le produit vaut $0.5^{2000}$, ce qui est plus petit que le plus petit flottant positif et s'affiche `0.0`. La **log-vraisemblance** est

$$
\ln\prod_ip_i=\sum_i\ln p_i ,
$$

ici $2000\ln0.5=-1386.29$, un nombre parfaitement ordinaire. Comme $\ln$ est croissante, les paramètres qui maximisent la vraisemblance maximisent aussi la log-vraisemblance, et les sommes sont bien plus faciles à dériver que les produits.

**2. Entropie croisée.** Pour un classifieur avec des probabilités prédites $q$ sur les classes et une distribution vraie $p$, la perte est

$$
H(p,q)=-\sum_ip_i\ln q_i .
$$

Pour une vérité one-hot, le seul terme qui survit est $-\ln q_{\text{true}}$. Donner à la bonne classe la probabilité $0.7$ coûte $-\ln0.7=0.357$ ; lui donner $0.1$ coûte $2.303$ ; lui donner $0.001$ coûte $6.9$. Le log punit sévèrement les erreurs faites avec confiance mais récompense doucement l'amélioration près de 1, ce qui est exactement le comportement voulu.

**3. Softmax et l'astuce log-sum-exp.** Softmax utilise $e^{z_i}$, et son log demande $\ln\sum_ie^{z_i}$. Pour $z=(1000,1001,1002)$, $e^{1000}$ déborde vers l'infini en flottants 64 bits. La correction utilise la loi du produit : avec $m=\max z$,

$$
\ln\sum_ie^{z_i}=m+\ln\sum_ie^{z_i-m},
$$

et maintenant le plus grand exposant est $e^0=1$, donc rien ne déborde. Vérification : $z=(1,2,3)$ donne $3.4076$ des deux façons, et $z=(1000,1001,1002)$ donne $1002.4076$.

## Exemple travaillé en code

Tous ces nombres sont reproduits par l'exercice.

```python
import math
import numpy as np

print(1000 * 1.05 ** 10)                 # 1628.894626777442
print(1000 * math.exp(0.5))              # 1648.7212707001281
print([(1 + 1/n) ** n for n in (1, 10, 1000, 10**6)])
print(math.log(2) / math.log(1.07))      # 10.244768351058712
print(math.log(10) / math.log(2))        # 3.3219280948873626  (log base 2 of 10)
print(0.5 ** 2000, 2000 * math.log(0.5)) # 0.0   -1386.2943611198905
print(-math.log(0.7))                    # 0.35667494393873245

z = np.array([1000., 1001., 1002.])
m = z.max()
print(m + np.log(np.exp(z - m).sum()))   # 1002.4076059644444
```

Remarque la sixième ligne : le produit sous-déborde jusqu'à exactement zéro, mais la version avec le log est correcte.

## Idées fausses courantes

- « $\log(a+b)=\log a+\log b$. » Non : $\log(ab)=\log a+\log b$.
- « $(a+b)^n=a^n+b^n$. » Non.
- « $\ln$ et $\log$ sont des fonctions différentes. » Ce sont deux logarithmes de bases différentes ; dans la plupart des textes de ML et de maths, $\log$ signifie $\ln$. Vérifie la convention (certains textes d'ingénierie utilisent la base 10, et la théorie de l'information utilise souvent la base 2).
- « La croissance exponentielle est rapide dès le départ. » Elle paraît plate longtemps, puis explose.
- « $e^{x}$ avec $x$ négatif est négatif. » Il est positif, simplement petit : $e^{-3}=0.0498$.
- Oublier que les logs de nombres dans $(0,1)$ sont négatifs, d'où le signe moins devant les formules de perte.

## Points clés à retenir

- Les lois des exposants découlent du comptage des copies ; $a^0=1$, $a^{-n}=1/a^n$, $a^{1/n}=\sqrt[n]a$ les gardent cohérentes.
- $e\approx2.71828$ est la limite de $(1+1/n)^n$ et la base de la croissance continue.
- $\log_bx$ est l'exposant qui donne $x$ à partir de $b$ ; $\ln$ défait $\exp$ ; le domaine est $x>0$.
- Lois des logs : produits en sommes, puissances en multiplicateurs ; changement de base $\log_bx=\ln x/\ln b$.
- Le ML utilise les logs pour la log-vraisemblance, l'entropie croisée et le softmax numériquement stable (log-sum-exp).

## Explique-le toi-même

1. Pourquoi les logarithmes transforment-ils la multiplication en addition ? Explique avec l'image du « comptage du nombre de multiplications ».
2. Pourquoi un modèle entraîné sur un million d'exemples travaille-t-il avec la somme des log-probabilités et non avec le produit des probabilités ?

## Pour aller plus loin

- Les leçons ultérieures sur la théorie de l'information (entropie, divergence KL), sur softmax et sigmoïde, et sur la stabilité numérique utilisent tout ce qui est ici.

## Critères de maîtrise

Tu sais : utiliser toutes les lois des exposants avec des exposants négatifs et fractionnaires ; calculer une croissance composée et un temps de doublement ; dire ce qu'est $e$ et l'approcher numériquement ; passer de la forme exponentielle à la forme logarithmique ; appliquer les lois des logs et le changement de base ; expliquer en deux phrases pourquoi le ML s'entraîne avec des log-probabilités ; calculer un log-sum-exp sans débordement.
