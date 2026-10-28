---
id: mf.07
title: "Limites et continuité"
summary: "Construis l'idée de limite à partir de nombres qui s'approchent d'une cible, ajoute les limites à gauche et à droite, les lois des limites et la continuité, rencontre doucement la définition epsilon-delta, et vois pourquoi les dérivées sont des limites de pentes."
minutes: 100
teaches: [mf.limits]
requires: []
sources:
  - "math/math for ML.pdf, Ch. 5 intro and Sec. 5.1 (PDF p.145-149): function notation, the difference quotient as the slope of a secant, the derivative defined as the limit as h -> 0, and the worked derivation of the derivative of x^n by shrinking terms. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (The Problem, 'What is a derivative?' and its limit definition; numerical derivative with a small h): why derivatives matter for learning and how a small h replaces the limit in code. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/docs/en.md (floating-point precision and catastrophic cancellation discussion): why numerical limits cannot take h too small. READ."
---

## Où cela s'inscrit

Entraîner un modèle, c'est pousser les paramètres dans la direction qui fait baisser une perte. « Quelle direction ? » reçoit sa réponse de la **dérivée**, la pente de la perte au point courant. Mais la pente d'une courbe en un *seul* point semble impossible : une pente demande deux points, et un point n'est qu'un point. La solution est l'idée de **limite** : prends deux points, rapproche-les, et demande vers quoi la pente *tend*. Les limites sont le fondement sous les dérivées, les intégrales, les séries convergentes (`mf.06`), et la définition de $e$ (`mf.04`).

C'est la dernière leçon de l'étape des fondations. Elle ne demande que les idées de fonctions de `mf.03` et les habitudes de code de `mf.01` ; l'exemple de la série géométrique renvoie à `mf.06`, mais rien d'autre n'est supposé.

## L'idée en mots simples

Une limite parle de **la direction que prend un voyage, pas de l'endroit où il s'arrête**. Marche vers une porte par des pas qui divisent toujours par deux la distance restante. Tu ne « touches » jamais la porte en un nombre fini de pas, et pourtant il est parfaitement clair vers où tu vas. La limite, c'est la porte.

Trois images :

1. **Un trou dans une route.** Il manque une planche à la position 1. Tu ne peux pas t'y tenir, mais chaque planche des deux côtés y mène tout droit, à la même hauteur. La limite existe (c'est cette hauteur) même si la valeur en ce point n'existe pas.
2. **Une falaise.** La route à gauche de la position 0 est à la hauteur $-1$, et à droite à la hauteur $+1$. En approchant par la gauche, tu te diriges vers $-1$, par la droite vers $+1$. Les deux directions sont en désaccord, donc il n'y a pas de destination unique : la limite n'existe pas.
3. **Une route continue.** Pas de planche manquante, pas de falaise : là où tu vas est là où tu es. C'est la **continuité** : tu peux dessiner le graphe sans lever le crayon.

## Limites, intuitivement

Nous écrivons

$$
\lim_{x\to a}f(x)=L
$$

et nous disons : *quand $x$ se rapproche de $a$ (des deux côtés, mais sans être égal à $a$), $f(x)$ se rapproche de $L$, aussi près que nous voulons.* L'expression « sans être égal à $a$ » est délibérée : une limite ignore la valeur en $a$ lui-même.

### De petites explorations numériques

**Exemple 1 (un trou).** $f(x)=\dfrac{x^2-1}{x-1}$ n'est pas définie en $x=1$ (elle vaudrait $0/0$). Tabulons :

| $x$ | 0.9 | 0.99 | 0.999 | 1.001 | 1.01 | 1.1 |
|---|---|---|---|---|---|---|
| $f(x)$ | 1.9 | 1.99 | 1.999 | 2.001 | 2.01 | 2.1 |

Les deux côtés tendent vers 2. L'algèbre le confirme : pour $x\neq1$, $\frac{x^2-1}{x-1}=\frac{(x-1)(x+1)}{x-1}=x+1$, qui tend vers 2. Donc $\lim_{x\to1}f(x)=2$ bien que $f(1)$ n'existe pas.

**Exemple 2 (un exemple important).** $\dfrac{\sin x}{x}$ en $0$ (radians). En $x=0.1$ : $0.99833$ ; en $0.01$ : $0.9999833$ ; en $0.001$ : $0.99999983$. La limite est $1$.

**Exemple 3 (une falaise).** $f(x)=|x|/x$ vaut $-1$ pour $x<0$ et $+1$ pour $x>0$.

### Limites à gauche et à droite

$\lim_{x\to a^-}f(x)$ approche par la gauche ($x<a$), $\lim_{x\to a^+}f(x)$ par la droite. La limite ordinaire existe **exactement quand les deux limites unilatérales existent et sont égales**. Pour $|x|/x$ en 0 : limite à gauche $-1$, limite à droite $+1$, donc pas de limite. Pour $x^2$ en 2, les deux côtés donnent 4.

### Limites à l'infini et limites infinies

$\lim_{x\to\infty}\frac1x=0$ : plus $x$ est grand, plus $\frac1x$ est petit. Un graphe qui monte sans borne près d'un point (comme $\frac1{x^2}$ près de 0) est dit avoir pour limite $\infty$, ce qui est une façon de décrire *comment* la limite échoue à exister. De même, les sommes partielles d'une série géométrique convergente (leçon précédente) forment une suite dont la limite est la somme, et $e=\lim_{n\to\infty}(1+\frac1n)^n$.

## Lois des limites

Si $\lim_{x\to a}f=L$ et $\lim_{x\to a}g=M$ existent toutes les deux, alors

$$
\lim(f+g)=L+M,\quad \lim(f-g)=L-M,\quad \lim(cf)=cL,\quad \lim(fg)=LM,\quad \lim\frac fg=\frac LM\ (M\neq0).
$$

En mots : tu peux calculer les limites morceau par morceau, puis combiner. Exemple : $\lim_{x\to2}(3x^2-x)$ : la limite de $x$ est 2, donc la limite de $x^2$ est 4, donc la limite est $3\cdot4-2=10$, ce qui revient à remplacer simplement. Pour les fonctions continues, c'est toujours permis. Les lois n'aident plus quand tu obtiens $0/0$ : c'est le cas qui demande de l'algèbre (simplifier, factoriser) ou un argument astucieux. Technique typique : factoriser et simplifier, comme dans l'exemple 1 ; ou le **principe d'encadrement** (théorème des gendarmes) : si $g\le f\le h$ près de $a$ et que $g$ et $h$ tendent toutes deux vers $L$, alors $f$ tend vers $L$ (c'est ainsi que se prouve $\sin x/x\to1$ ; ici nous ne le testons que numériquement).

## Continuité

Une fonction $f$ est **continue en $a$** si trois choses sont vraies :

1. $f(a)$ est défini ;
2. $\lim_{x\to a}f(x)$ existe ;
3. les deux sont égaux : $\lim_{x\to a}f(x)=f(a)$.

Trois façons d'échouer, une par condition :

| échec | exemple au point problématique | nom |
|---|---|---|
| valeur non définie, limite existante | $\frac{x^2-1}{x-1}$ en 1 | amovible (un trou ; fixer $f(1)=2$ le répare) |
| les côtés sont en désaccord | $\lvert x\rvert/x$ en 0, ou une fonction en escalier | saut |
| explose | $\frac1x$ en 0 | infinie |

Un **coin** comme $|x|$ en 0 est encore continu (pas de rupture), même s'il s'avérera qu'il n'est pas assez lisse pour avoir une dérivée là ; c'est pourquoi ReLU, $\max(0,x)$, qui est continue avec un coin, convient très bien comme activation de réseau de neurones. Les fonctions familières (polynômes, $e^x$, $\sin$, $\cos$ ; $\ln x$ pour $x>0$) sont continues partout où elles sont définies, et les sommes, produits et compositions de fonctions continues sont continus. Une fonction continue ne peut pas non plus sauter des valeurs : si $f(0)<0<f(1)$, elle s'annule quelque part entre les deux (la propriété des valeurs intermédiaires), et c'est pourquoi la méthode de bissection de `mf.03` fonctionne.

## L'idée epsilon-delta, doucement

« Aussi près que nous voulons » doit être transformé en un jeu de défi précis.

- Quelqu'un choisit une tolérance $\varepsilon>0$ (epsilon) pour la proximité de la sortie à $L$ : la *bande cible*.
- Tu dois répondre par un $\delta>0$ (delta) : à quelle distance de $a$ l'entrée doit être pour garantir que la sortie tombe dans la bande.

La limite existe avec la valeur $L$ si tu peux gagner **pour tout** $\varepsilon$ :

$$
\forall\varepsilon>0\ \exists\delta>0:\quad 0<|x-a|<\delta\ \Longrightarrow\ |f(x)-L|<\varepsilon .
$$

(À lire avec `mf.02` : « pour tout epsilon il existe un delta tel que... ».)

**Exemple travaillé.** $f(x)=2x+1$, $a=3$, $L=7$. Alors $|f(x)-7|=|2x-6|=2|x-3|$. Il nous faut $2|x-3|<\varepsilon$, c'est-à-dire $|x-3|<\varepsilon/2$. Choisissons donc $\delta=\varepsilon/2$ et le jeu est gagné pour tout $\varepsilon$ : pour $\varepsilon=0.1$ tout $\delta\le0.05$ convient ; pour $\varepsilon=0.01$ tout $\delta\le0.005$ convient. Remarque : une bande plus petite demande un $\delta$ plus petit, mais il y en a toujours un qui marche. Pour une courbe qui se courbe violemment (comme $\sin(1/x)$ près de 0), aucun $\delta$ ne marche pour un petit $\varepsilon$ : la fonction continue d'osciller entre $-1$ et $1$ et n'a pas de limite en 0.

Tu auras rarement besoin de jouer ce jeu toi-même. Sa valeur est de faire de « tend vers » un énoncé vérifiable, c'est pourquoi chaque théorème ultérieur sur les dérivées et les intégrales peut être démontré. L'exercice laisse un programme jouer le jeu par échantillonnage.

## Pourquoi les dérivées ont besoin des limites

La pente de la droite passant par $(x,f(x))$ et $(x+h,f(x+h))$ est le **taux d'accroissement** (quotient aux différences)

$$
\frac{f(x+h)-f(x)}{h}.
$$

Poser directement $h=0$ donne $\frac00$, ce qui ne dit rien. À la place, laisse $h$ rétrécir et observe le quotient. La dérivée est sa limite :

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
$$

Pour $f(x)=x^2$ en $x=3$ : $\frac{(3+h)^2-9}{h}=\frac{6h+h^2}{h}=6+h$ (pour $h\neq0$), et $6+h\to6$. Numériquement : $h=1$ donne 7, $h=0.1$ donne 6.1, $h=0.001$ donne 6.001. Le manuel effectue la même étape pour $x^n$ : développer $(x+h)^n$, simplifier le $x^n$, diviser par $h$, et chaque terme restant qui contient encore $h$ disparaît à la limite, laissant $nx^{n-1}$. Donc : dérivée $=$ limite de pentes, et la sécante devient la tangente.

La continuité est nécessaire aussi : une fonction avec un saut en $a$ n'a pas de pente sensée là. (La continuité est nécessaire, mais non suffisante : $|x|$ est continue en 0 mais a un coin.)

## Exploration numérique des limites, et son piège

En code, nous ne pouvons pas laisser $h$ atteindre zéro, donc nous utilisons un $h$ « assez petit » et regardons la tendance :

```python
import math

f = lambda x: (x * x - 1) / (x - 1)
print([f(1 + h) for h in (0.1, 0.01, 0.001)])   # [2.1, 2.009999999999999, 2.0009999999999177]
print([math.sin(h) / h for h in (0.1, 0.01, 0.001)])
# [0.9983341664682815, 0.9999833334166665, 0.9999998333333416]
print([((3 + h) ** 2 - 9) / h for h in (0.1, 0.01, 0.001)])
# [6.100000000000012, 6.009999999999849, 6.000999999999479]
```

Le piège : ne rends pas $h$ absurdement petit. Les nombres à virgule flottante contiennent environ 16 chiffres, et soustraire deux nombres presque égaux les détruit (annulation catastrophique) :

```python
for h in (1e-4, 1e-6, 1e-8):
    print(h, (1 - math.cos(h)) / h ** 2)
# 0.0001 0.4999999969612645
# 1e-06 0.5000444502911705
# 1e-08 0.0
```

La vraie limite est $1/2$. À $h=10^{-8}$, `cos(h)` s'arrondit à exactement 1.0 et le numérateur devient 0. Bonne pratique : regarde plusieurs valeurs modérées de $h$, vérifie que la tendance est stable, et confirme avec l'algèbre quand tu le peux. C'est aussi pourquoi les gradients numériques en ML utilisent un $h$ autour de $10^{-5}$ à $10^{-7}$, non $10^{-15}$.

## Exemple travaillé

Suis toutes les pièces pour $f(x)=\dfrac{x^2-1}{x-1}$ en $a=1$ :

1. $f(1)$ n'est pas défini, donc $f$ n'est pas continue en 1.
2. Valeurs à gauche et à droite en $h=10^{-6}$ : $f(1-h)\approx1.999999$, $f(1+h)\approx2.000001$ ; elles concordent à $10^{-4}$ près, donc la limite est 2.
3. Définis une version réparée $\tilde f(x)=f(x)$ pour $x\neq1$ et $\tilde f(1)=2$. Maintenant valeur $=$ limite, donc $\tilde f$ est continue en 1.
4. Si au contraire nous définissions $\tilde f(1)=5$, la valeur 5 serait en désaccord avec la limite 2 : toujours discontinue (une rupture amovible).

L'exercice implémente `one_sided`, `limit_estimate`, `is_continuous_at`, la recherche epsilon-delta et le taux d'accroissement, et teste chaque affirmation ci-dessus.

## Idées fausses courantes

- « La limite est $f(a)$. » Seulement quand $f$ est continue en $a$. La limite ignore la valeur en $a$.
- « Si $f(a)$ n'est pas défini, la limite ne peut pas exister. » Contre-exemple : le trou ci-dessus.
- « Une limite signifie que la fonction n'atteint jamais la valeur. » Elle peut l'atteindre (les constantes le font) ; une limite parle de la tendance.
- « Si les deux valeurs de chaque côté sont proches, la limite existe. » Elles doivent être égales à la limite, pas seulement proches pour un pas donné.
- « Un $h$ plus petit est toujours meilleur numériquement. » Passé un certain point, les erreurs d'arrondi prennent le dessus.
- « Continue signifie dérivable. » $|x|$ est un contre-exemple. (La réciproque tient : dérivable implique continue.)
- « $\infty$ est un nombre qu'on peut remplacer. » Il décrit un comportement non borné ; $\infty-\infty$ et $\infty/\infty$ ne sont pas définis.

## Points clés à retenir

- Une limite est la direction que prend la fonction près de $a$, en ignorant la valeur en $a$ ; elle existe quand la gauche et la droite concordent.
- Les lois des limites permettent de calculer les limites morceau par morceau ; pour les formes $0/0$, simplifie algébriquement.
- Continue en $a$ : valeur définie, limite existante, et égales. Les discontinuités sont amovibles, à saut ou infinies.
- Epsilon-delta transforme « proche » en jeu : pour chaque tolérance $\varepsilon$ il doit exister un $\delta$.
- La dérivée est la limite du taux d'accroissement : la pente des sécantes qui devient une tangente.
- En code, explore les limites avec un $h$ modérément petit ; un $h$ extrêmement petit souffre d'erreurs d'arrondi.

## Explique-le toi-même

1. Explique avec tes propres mots pourquoi la limite de $(x^2-1)/(x-1)$ en 1 existe même si la fonction n'y est pas définie.
2. Pourquoi ne peut-on pas calculer la pente d'une courbe en un point en posant simplement $h=0$ dans le taux d'accroissement, et que fait-on à la place ?

## Pour aller plus loin

- Le manuel de référence, Sec. 5.1 : taux d'accroissement, dérivée, et dérivation de la dérivée de $x^n$.
- Prochaines étapes de l'académie : dérivées et gradients (calcul différentiel), puis tout ce qui entraîne un modèle. Tu possèdes maintenant la notation, l'algèbre, les fonctions, les exponentielles et logarithmes, la trigonométrie, les sommes et les limites sur lesquelles tout cela repose.

## Critères de maîtrise

Tu sais : estimer une limite à partir d'un tableau de valeurs et la confirmer algébriquement ; calculer les limites à gauche et à droite et décider si la limite existe ; utiliser les lois des limites ; tester les trois conditions de la continuité et classer une discontinuité ; mener un argument epsilon-delta pour une fonction linéaire ; écrire la dérivée comme limite de taux d'accroissement et l'approcher avec un petit $h$ en expliquant pourquoi $h$ ne doit pas être trop petit.
