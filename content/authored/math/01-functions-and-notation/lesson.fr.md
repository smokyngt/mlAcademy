---
id: f2n.01
title: "Fonctions, sommes et logarithmes"
summary: "La notation que toutes les leçons suivantes utilisent, et pourquoi le machine learning calcule dans l'espace des logarithmes."
minutes: 40
teaches: [alg.functions, alg.summation, alg.exp-log]
requires: []
sources:
  - math/math for ML.pdf (Ch. 1 Introduction and Motivation, PDF p.17; Ch. 2 opening, PDF p.23)
  - ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md
---

## Objectifs d'apprentissage

- Lire et écrire des signatures de fonctions comme $f:\mathbb{R}^n \to \mathbb{R}^m$ et dire quelles sont les tailles de l'entrée et de la sortie.
- Développer et réécrire des sommes écrites avec $\sum$, et faire entrer ou sortir des constantes.
- Utiliser $\exp$ et $\ln$ comme fonctions inverses et appliquer $\ln(ab)=\ln a+\ln b$.
- Expliquer pourquoi un produit de nombreuses probabilités se calcule comme une somme de logarithmes.

## Prérequis

L'arithmétique et la notion de variable. Rien d'autre.

## Où cela s'inscrit

Chaque modèle de machine learning, d'un filtre anti-spam à un modèle de langage, est une **fonction** : quelque chose qui prend des nombres en entrée et renvoie des nombres en sortie. « Entraîner » signifie ajuster les nombres à l'intérieur de cette fonction jusqu'à ce que ses sorties soient bonnes. Pour suivre ce qui vient ensuite, tu dois savoir lire la façon dont une telle fonction est écrite, comment plusieurs fonctions s'enchaînent, et comment additionner une erreur sur des milliers d'exemples.

Cette première leçon est l'alphabet. Elle ne contient aucune idée difficile, mais chaque page suivante utilise ses symboles sans les réexpliquer : dix minutes de patience valent donc la peine. Elle contient aussi une surprise pratique (pourquoi les programmes additionnent des logarithmes au lieu de multiplier des probabilités) qui explique une astuce que tu rencontreras dans presque tous les codes de ML.

## L'idée en mots simples

- Une **fonction** est un distributeur automatique. Tu appuies sur un bouton (l'entrée), et exactement une friandise tombe (la sortie). Appuie deux fois sur le même bouton et tu obtiens la même friandise. Écrire `f: R^3 -> R^2` dit simplement « cette machine a trois fentes pour les nombres à l'entrée et deux à la sortie ».
- **Enchaîner** des fonctions, c'est mettre des machines à la suite : la friandise de la première machine est déposée dans la seconde. Un réseau de neurones est une rangée de machines simples.
- Une **somme** (la lettre grecque sigma) est un ticket de caisse : on descend la liste et on ajoute chaque ligne à un total courant. C'est une boucle `for` qui additionne.
- Un **logarithme** répond à la question « combien de chiffres a ce nombre ? » (à peu près). C'est pourquoi il transforme la multiplication en addition : multiplier un nombre à 3 chiffres par un nombre à 4 chiffres donne un nombre d'environ 7 chiffres, et 3 + 4 = 7. Les ordinateurs apprécient cela, car multiplier des centaines de très petits nombres les fait disparaître, alors qu'additionner leurs logarithmes reste confortable.

Garde en tête ces quatre images ; le reste de la leçon consiste simplement à les rendre précises.

## Intuition

Une **fonction** est une machine : on met une entrée, on obtient exactement une sortie. Le machine learning s'intéresse presque exclusivement à des fonctions trop compliquées pour être écrites à la main, donc on les décrit comme des *compositions de fonctions simples* et on apprend leurs paramètres à partir des données.

Une **somme** $\sum$ est simplement une boucle `for` qui additionne. Un **logarithme** transforme la multiplication en addition, ce qui est exactement ce dont un ordinateur a besoin quand il multiplie des centaines de très petits nombres.

## Définitions formelles

**Fonction.** Une fonction $f$ d'un ensemble $A$ (le *domaine de départ*) vers un ensemble $B$ (le *codomaine*) associe à chaque $a\in A$ exactement un $f(a)\in B$. On écrit $f: A \to B$.

En machine learning, les ensembles sont presque toujours des espaces de vecteurs réels :

| Signature | Se lit | Exemple |
|---|---|---|
| $f:\mathbb{R}\to\mathbb{R}$ | un nombre en entrée, un nombre en sortie | $f(x)=x^2$ |
| $L:\mathbb{R}^d\to\mathbb{R}$ | $d$ paramètres en entrée, une perte scalaire en sortie | erreur quadratique moyenne |
| $g:\mathbb{R}^n\to\mathbb{R}^m$ | une couche : $n$ caractéristiques en entrée, $m$ activations en sortie | $g(x)=\mathrm{ReLU}(Wx+b)$ |

**Composition.** $(g\circ f)(x)=g(f(x))$. On applique d'abord $f$, puis $g$. Cela exige que la sortie de $f$ vive dans le domaine de $g$. Un réseau de neurones à trois couches est $h = f_3\circ f_2\circ f_1$ ; les formes doivent s'enchaîner : $\mathbb{R}^{n}\to\mathbb{R}^{m_1}\to\mathbb{R}^{m_2}\to\mathbb{R}^{m_3}$.

**Sommation.** Pour des nombres $a_1,\dots,a_n$,

$$
\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n .
$$

La lettre $i$ est un *indice muet* : la renommer en $k$ ne change rien. Trois règles couvrent presque tout :

1. **Linéarité :** $\sum_i (\alpha a_i + \beta b_i) = \alpha\sum_i a_i + \beta\sum_i b_i$.
2. **Une constante sommée $n$ fois :** $\sum_{i=1}^{n} c = nc$.
3. **L'ordre n'a pas d'importance** pour une somme finie (mais voir les remarques sur les flottants plus bas).

Le produit $\prod_{i=1}^n a_i = a_1 a_2\cdots a_n$ fonctionne de la même manière.

**Exponentielle et logarithme.** $\exp(x)=e^x$ avec $e\approx 2.71828$. Le logarithme népérien $\ln$ est sa réciproque : $\ln(\exp(x))=x$ et $\exp(\ln y)=y$ pour $y>0$. La propriété qui compte le plus :

$$
\exp(a+b)=\exp(a)\exp(b)\qquad\Longleftrightarrow\qquad \ln(ab)=\ln a+\ln b .
$$

## Démonstration : pourquoi les produits deviennent des sommes

Soit $y=\exp(a)$ et $z=\exp(b)$. Alors $yz=\exp(a)\exp(b)=\exp(a+b)$ d'après la règle des exposants. Prenons $\ln$ des deux côtés : $\ln(yz)=a+b=\ln y+\ln z$. En répétant l'argument, on obtient, pour des $p_i$ positifs,

$$
\ln\Big(\prod_{i=1}^n p_i\Big)=\sum_{i=1}^n \ln p_i .
$$

## Exemple traité 1 : une somme

$$
\sum_{i=1}^{4} i^2 = 1^2+2^2+3^2+4^2 = 1+4+9+16 = 30 .
$$

Avec la linéarité : $\sum_{i=1}^{4}(3i^2+2) = 3\cdot 30 + 2\cdot 4 = 98$.

## Exemple traité 2 : pourquoi on a besoin des logarithmes

Supposons qu'un modèle attribue la probabilité $0.1$ à chacune de 400 observations indépendantes. La vraisemblance est $0.1^{400}=10^{-400}$. Un flottant 64 bits ne peut rien représenter de plus petit qu'environ $2.2\times10^{-308}$, donc l'ordinateur stocke **0**. Mais

$$
\ln(0.1^{400}) = 400\ln 0.1 \approx 400\times(-2.3026) = -921.03,
$$

ce qui est un nombre parfaitement ordinaire. C'est pourquoi toute vraisemblance en machine learning est manipulée sous forme de **log-vraisemblance**.

## Implémentation

```python
import numpy as np

p = np.full(400, 0.1)
print(np.prod(p))          # 0.0  (underflow)
print(np.sum(np.log(p)))   # -921.03...
```

Un piège très proche : $\log\sum_i e^{x_i}$ déborde si un $x_i$ est grand (par exemple $e^{1000}=\infty$). Soustrais d'abord le maximum $m=\max_i x_i$ :

$$
\ln\sum_i e^{x_i} = m + \ln\sum_i e^{x_i-m}.
$$

Après le décalage, tous les exposants sont $\le 0$, donc aucun terme ne peut déborder, et au moins un terme vaut $1$, donc la somme ne peut pas passer sous le seuil et devenir zéro. Tu implémenteras cela (« log-sum-exp ») dans les exercices ; c'est le cœur du softmax stable de la leçon 8.

## Formes

Pour une signature $f:\mathbb{R}^n\to\mathbb{R}^m$, demande-toi toujours : *quelle forme entre, quelle forme sort ?* Quand tu verras plus tard `W @ x`, tu vérifieras $W\in\mathbb{R}^{m\times n}$, $x\in\mathbb{R}^n$, résultat $\in\mathbb{R}^m$.

## Idées fausses courantes

- **« $f(x+y)=f(x)+f(y)$ pour n'importe quelle fonction. »** Seules les fonctions *linéaires* satisfont cela. $(x+y)^2\neq x^2+y^2$, et $\ln(x+y)\neq\ln x+\ln y$.
- **« $\ln(a+b)=\ln a+\ln b$. »** Non : le logarithme convertit les *produits* en sommes, pas les sommes en sommes.
- **« $\sum_i a_ib_i = (\sum_i a_i)(\sum_i b_i)$. »** Le produit de sommes a $n^2$ termes croisés ; la somme de produits en a $n$.
- **Virgule flottante :** une somme de flottants dépend légèrement de l'ordre d'addition. Mathématiquement, ce n'est pas le cas.

## Points clés à retenir

- Une fonction est définie par ses espaces d'entrée et de sortie : lis toujours `f: R^n -> R^m` avant toute autre chose.
- Les sommes sont des boucles ; la linéarité permet de sortir des constantes et de scinder des sommes.
- `ln(ab) = ln a + ln b` explique pourquoi les vraisemblances sont manipulées comme des sommes de logarithmes.
- Soustrais le maximum avant d'exponentier : c'est toute l'astuce d'un log-sum-exp stable.

## Explique-le toi-même

Réponds à voix haute ou par écrit, sans regarder en arrière. Si tu bloques, c'est exactement l'endroit à relire.

1. Pourquoi un produit de 400 probabilités devient-il exactement 0 en virgule flottante, alors que la somme de leurs logarithmes ne pose aucun problème ?
2. Calcule `ln(1000)` en connaissant seulement `ln 2` et `ln 5`. Quelle règle as-tu utilisée ?

## Pour aller plus loin

- *Mathematics for Machine Learning* (Deisenroth et al.), Ch. 1 pour la motivation, PDF p.17.
- Bibliothèque source : `ai-engineering-from-scratch` phase 01, leçon 01.

## Critères de maîtrise

Tu sais (a) indiquer la dimension d'entrée et de sortie de n'importe quelle fonction de ce cours, (b) réécrire une somme à l'aide de la linéarité sans calculatrice, et (c) expliquer en deux phrases pourquoi les log-vraisemblances existent.
