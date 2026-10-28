---
id: mf.06
title: "Suites, séries et sommation"
summary: "Suites, séries arithmétiques et géométriques avec leurs formules démontrées, notations sigma et produit, convergence de la série géométrique, série harmonique divergente, et boucles contre formes closes."
minutes: 95
teaches: [mf.series]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (the limit definition of the derivative and gradient-descent discussion): sums over many terms and limits as the context for convergence. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md (log-sum-exp section; floating-point accumulation): why sums of many terms need care in code. READ."
  - "math/math for ML.pdf, Sec. 2.1 (PDF p.26): the general sum a_i1 x_1 + ... + a_in x_n written in sigma-style shorthand for a system of equations. READ."
---

## Où cela s'inscrit

L'apprentissage automatique est plein de sommes. Une perte est une moyenne d'erreurs sur $N$ exemples : $\frac1N\sum_{i=1}^N\ell_i$. Un produit scalaire est $\sum_ja_jb_j$. Une distribution de probabilité a pour somme 1. Une récompense actualisée en apprentissage par renforcement est $\sum_k\gamma^kr_k$, une série géométrique. Les taux d'apprentissage décroissent géométriquement ; le nombre d'étapes dont un processus a besoin suit souvent une série harmonique. Avant de pouvoir lire ces formules, tu dois être à l'aise avec la notation, et avant de pouvoir leur faire confiance, tu dois savoir quand une somme infinie a seulement une réponse.

Aucune leçon antérieure n'est requise. Nous n'utilisons que l'arithmétique et les exposants (`mf.02`, `mf.04`) et nous écrivons de courtes boucles Python (`mf.01`). La dernière section prépare l'idée de limite de `mf.07`.

## L'idée en mots simples

Une **suite** est une liste avec une règle : 2, 4, 6, 8, ... (ajouter 2 à chaque fois) ou 3, 6, 12, 24, ... (doubler à chaque fois). Une **série** est ce que tu obtiens quand tu *additionnes* la liste.

Deux listes spéciales valent beaucoup :

- **Arithmétique** : chaque pas ajoute la même quantité, comme monter des marches de même hauteur. Pour additionner de 1 à 100, une astuce célèbre consiste à écrire la liste à l'endroit et à l'envers et à additionner colonne par colonne. Chaque colonne a la même somme, donc le total est facile.
- **Géométrique** : chaque pas multiplie par la même quantité, comme une balle qui rebondit en remontant à la moitié de sa hauteur précédente. Si le multiplicateur est de taille inférieure à 1, les termes rétrécissent si vite que même une *infinité* d'entre eux s'additionnent en un nombre fini : une balle qui rebondit éternellement parcourt pourtant une distance finie. Si le multiplicateur vaut 1 ou plus, le total n'est pas borné.

Et une image d'avertissement : tu pourrais penser que « si les morceaux que tu additionnes deviennent de plus en plus petits, le total doit se stabiliser ». La **série harmonique** 1 + 1/2 + 1/3 + 1/4 + ... montre que c'est faux : elle monte sans limite, extrêmement lentement.

## Suites

Une suite est une fonction de la position : $a_1,a_2,a_3,\dots$, avec un **terme général** $a_n$. Deux façons d'en décrire une : une formule directe ($a_n=3n-1$ donne 2, 5, 8, ...), ou une récurrence ($a_1=2$, $a_{n+1}=a_n+3$).

**Suite arithmétique :** une différence constante $d$.

$$
a_n=a_1+(n-1)d .
$$

Exemple : premier terme 5, $d=3$ : 5, 8, 11, ..., et le 10e terme est $5+9\cdot3=32$.

**Suite géométrique :** un rapport constant $r$.

$$
a_n=a_1r^{n-1} .
$$

Exemple : 3, 6, 12, 24, ... a $a_1=3$, $r=2$, et $a_{10}=3\cdot2^9=1536$.

## Notations sigma et produit

$$
\sum_{k=m}^{n}a_k=a_m+a_{m+1}+\dots+a_n
$$

se lit « la somme des $a_k$ quand $k$ va de $m$ à $n$ ». La lettre $k$ est l'**indice** (un nom muet : le renommer en $i$ ne change rien), $m$ la borne inférieure, $n$ la borne supérieure. Exemples :

- $\sum_{k=1}^{4}k^2=1+4+9+16=30$.
- $\sum_{k=0}^{3}2^k=1+2+4+8=15$.
- $\sum_{k=1}^{3}5=5+5+5=15$ (un terme constant ne fait que compter les termes).

En code, le sigma est une boucle avec un accumulateur :

```python
total = 0
for k in range(1, 5):      # k = 1, 2, 3, 4
    total += k ** 2
print(total)               # 30
```

ou une ligne NumPy `np.sum(np.arange(1, 5) ** 2)`.

Règles (elles découlent des règles ordinaires de l'addition) :

$$
\sum(a_k+b_k)=\sum a_k+\sum b_k,\qquad \sum c\,a_k=c\sum a_k .
$$

Mais $\sum a_kb_k\neq(\sum a_k)(\sum b_k)$ en général.

**Notation produit** $\prod_{k=1}^{n}a_k=a_1a_2\cdots a_n$. La plus célèbre : $n!=\prod_{k=1}^{n}k$, donc $5!=120$. Prendre le log (de `mf.04`) transforme un produit en somme : $\ln\prod a_k=\sum\ln a_k$, l'astuce de la log-vraisemblance.

## Démontrer la série arithmétique

Soit $S=1+2+\dots+n$. Écris-la deux fois, la seconde à l'envers, et additionne colonne par colonne :

$$
\begin{aligned}
S&=1+2+\dots+(n-1)+n\\
S&=n+(n-1)+\dots+2+1\\
2S&=(n+1)+(n+1)+\dots+(n+1)\quad(n\text{ columns})
\end{aligned}
$$

Donc $2S=n(n+1)$ et

$$
1+2+\dots+n=\frac{n(n+1)}2 .
$$

Pour $n=100$ : $100\cdot101/2=5050$. Le même appariement marche pour toute suite arithmétique : le premier terme plus le dernier est le même dans chaque colonne, ce qui donne

$$
S_n=\frac n2\,(a_1+a_n)=\frac n2\big(2a_1+(n-1)d\big).
$$

Vérification : 5, 8, ..., 32 (dix termes) : $\frac{10}{2}(5+32)=185$.

## Démontrer la série géométrique

Soit $S=a+ar+ar^2+\dots+ar^{n-1}$. Multiplie par $r$ et soustrais :

$$
\begin{aligned}
S&=a+ar+ar^2+\dots+ar^{n-1}\\
rS&=\phantom{a+{}}ar+ar^2+\dots+ar^{n-1}+ar^{n}\\
S-rS&=a-ar^{n}
\end{aligned}
$$

Tous les termes du milieu s'annulent. Donc $S(1-r)=a(1-r^n)$, et pour $r\neq1$

$$
S_n=a\,\frac{1-r^n}{1-r} .
$$

(Si $r=1$, chaque terme vaut $a$ et $S_n=na$.) Vérifications : $1+\frac12+\dots+\frac1{2^9}$ (10 termes) $=\frac{1-2^{-10}}{1-\frac12}=1.998046875$ ; et $3+6+\dots+3\cdot2^9=3\frac{1-2^{10}}{1-2}=3069$.

## Convergence de la série géométrique

Laissons maintenant le nombre de termes croître sans fin. Regarde le terme $r^n$ dans la formule.

- Si $|r|<1$, alors $r^n\to0$ (les puissances d'un nombre de taille inférieure à 1 rétrécissent vers zéro : $0.9^n$, $(-0.5)^n$). Donc $S_n\to\frac{a}{1-r}$ :

$$
\sum_{k=0}^{\infty}ar^k=\frac{a}{1-r},\qquad|r|<1 .
$$

- Si $|r|\ge1$, alors $r^n$ ne rétrécit pas, et les sommes partielles ne se stabilisent pas : pour $r=1$ elles croissent sans borne, pour $r=-1$ elles alternent $a,0,a,0,\dots$, pour $|r|>1$ elles explosent.

Exemples : $1+\frac12+\frac14+\dots=2$. $1-\frac12+\frac14-\dots=\frac1{1+\frac12}=\frac23$. $\sum_k0.9^k=10$ (avec 200 termes le code donne $9.99999999$, l'écart étant $0.9^{200}\cdot10\approx7\times10^{-9}$). En apprentissage par renforcement avec un facteur d'actualisation $\gamma=0.9$ et une récompense de 1 à chaque étape, la récompense actualisée totale vaut donc 10 : c'est exactement pourquoi on utilise l'actualisation, pour garder finies les sommes infinies.

Une image pour comprendre pourquoi $\sum 2^{-k}$ vaut 2 : pars d'un bâton de 2 mètres, retire 1 m, puis 0.5 m, puis 0.25 m... tu t'approches du bout du bâton sans jamais le dépasser.

## La série harmonique diverge

$$
H_n=1+\frac12+\frac13+\dots+\frac1n .
$$

Les termes tendent vers zéro, et pourtant la somme n'est pas bornée. Regroupe les termes : $\frac13+\frac14\ge\frac14+\frac14=\frac12$ ; $\frac15+\dots+\frac18\ge4\cdot\frac18=\frac12$ ; les 8 termes suivants valent chacun au moins $\frac1{16}$, ce qui donne au moins $\frac12$, et ainsi de suite. Chaque doublement du nombre de termes ajoute au moins $\frac12$, donc $H_n$ finit par dépasser n'importe quel nombre. Mais lentement : $H_{10}=2.929$, $H_{1000}=7.485$, $H_{10^6}=14.39$, et il faut 12 367 termes rien que pour dépasser 10 (elle croît comme $\ln n$, `mf.04`). Leçon : **que les termes tendent vers zéro est nécessaire pour la convergence, mais non suffisant.**

## Boucles contre formes closes

Une forme close remplace $n$ additions par une formule. Compare pour $\sum_{k=1}^nk^2$ :

```python
def loop_sum_squares(n):
    total = 0
    for k in range(1, n + 1):
        total += k * k
    return total

def closed_sum_squares(n):
    return n * (n + 1) * (2 * n + 1) // 6

print(loop_sum_squares(10), closed_sum_squares(10))   # 385 385
```

D'où vient la formule ? Utilise l'identité $(k+1)^3-k^3=3k^2+3k+1$ et additionne-la pour $k=1,\dots,n$. Le membre de gauche télescope (la plupart des termes s'annulent) en $(n+1)^3-1$. Le membre de droite est $3\sum k^2+3\sum k+n$. Donc $3\sum k^2=(n+1)^3-1-\frac{3n(n+1)}2-n$, ce qui se simplifie en $\frac{n(n+1)(2n+1)}2$, donnant $\sum k^2=\frac{n(n+1)(2n+1)}6$. Test numérique : $n=10$ donne $10\cdot11\cdot21/6=385$.

Règles pratiques : si une forme close existe, utilise-la (son coût ne croît pas avec $n$) et teste-la contre la boucle pour de petits $n$. Si elle n'existe pas (nombres harmoniques), vectorise : `np.sum(1 / np.arange(1, n + 1))`, ou des totaux cumulés avec `np.cumsum`. Pour de très longues sommes de flottants, accumule avec soin : ajouter beaucoup de minuscules termes à un gros total fait perdre de la précision, ce qui est l'une des raisons pour lesquelles le code numérique regroupe les sommes ou travaille avec des logs.

## Exemple travaillé en code

```python
import numpy as np

print(sum(range(1, 101)))                         # 5050
print(sum(1 / 2 ** k for k in range(10)))         # 1.998046875
print(3 * (1 - 2 ** 10) / (1 - 2))                # 3069.0
H = lambda n: sum(1 / k for k in range(1, n + 1))
print(H(10), H(1000))                             # 2.9289682539682538 7.485470860550345
n, s = 0, 0.0
while s <= 10:
    n += 1
    s += 1 / n
print(n)                                          # 12367
print(np.cumsum(1 / 2 ** np.arange(5)))           # [1. 1.5 1.75 1.875 1.9375]
```

## Idées fausses courantes

- « Si les termes tendent vers zéro, la série converge. » La série harmonique est le contre-exemple.
- « $\sum_{k=1}^nc=c$. » Elle vaut $nc$ : il y a $n$ termes.
- « $\sum a_kb_k=\sum a_k\sum b_k$. » Faux ; cela n'est vrai que pour un autre objet (une somme double).
- Erreurs d'un cran (off-by-one) : $\sum_{k=1}^{n}$ a $n$ termes, $\sum_{k=0}^{n}$ en a $n+1$ ; le `range(1, n+1)` de Python inclut $n$.
- Utiliser la formule géométrique infinie quand $|r|\ge1$.
- Penser qu'une somme infinie de nombres positifs doit être infinie. Ce n'est pas obligatoire.

## Points clés à retenir

- Arithmétique : $a_n=a_1+(n-1)d$, $S_n=\frac n2(a_1+a_n)$ par appariement à l'endroit et à l'envers.
- Géométrique : $a_n=a_1r^{n-1}$, $S_n=a\frac{1-r^n}{1-r}$ par multiplication et soustraction ; somme infinie $\frac a{1-r}$ seulement quand $|r|<1$.
- Sigma est une boucle avec un accumulateur ; la notation produit multiplie ; les logs convertissent les produits en sommes.
- La série harmonique diverge lentement, comme $\ln n$.
- Préfère les formes closes ou les sommes vectorisées aux boucles Python, et vérifie l'une contre l'autre.

## Explique-le toi-même

1. Explique pourquoi la démonstration de la somme géométrique multiplie par $r$ avant de soustraire.
2. Pourquoi un facteur d'actualisation $\gamma<1$ rend-il fini un flux infini de récompenses, alors que $\gamma=1$ ne le fait pas ?

## Pour aller plus loin

- La leçon suivante, `mf.07`, précise ce que signifie « les sommes partielles s'approchent d'un nombre » (limites).
- Plus tard : l'espérance (une somme pondérée), les développements en série et les séries de Taylor, les retours actualisés.

## Critères de maîtrise

Tu sais : écrire le $n$-ième terme d'une suite arithmétique ou géométrique ; développer et évaluer des expressions sigma et produit ; démontrer de zéro les deux formules finies ; décider si une série géométrique infinie converge et trouver sa somme ; expliquer pourquoi la série harmonique diverge ; vérifier une forme close contre une boucle en code.
