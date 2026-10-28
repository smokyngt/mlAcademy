---
id: mf.01
title: "Programmer pour ce cours : Python et NumPy"
summary: "Apprends juste assez de Python et de NumPy, depuis zéro, pour exécuter tous les exemples et exercices de l'académie."
minutes: 90
teaches: [mf.python, mf.numpy]
requires: []
sources:
  - "ai-engineering-from-scratch/phases/00-setup-and-tooling/01-dev-environment/docs/en.md: the 'layers' picture of an AI environment and why setup matters. READ."
  - "ai-engineering-from-scratch/phases/00-setup-and-tooling/05-jupyter-notebooks/docs/en.md: cells, kernel, the out-of-order-execution trap. READ."
  - "ai-engineering-from-scratch/phases/00-setup-and-tooling/06-python-environments/docs/en.md: why each project gets its own isolated environment. READ."
  - "ai-engineering-from-scratch/phases/00-setup-and-tooling/12-debugging-and-profiling/docs/en.md: print-debugging of shapes and dtypes. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/12-tensor-operations/docs/en.md: tensor, rank, axis, shape, broadcasting vocabulary. READ."
---

## Où cela s'inscrit

C'est la toute première leçon de l'académie. Chaque leçon suivante te demande d'exécuter un petit programme : pour vérifier un nombre, pour voir une formule se comporter, pour construire un modèle à partir de zéro. Tu n'as pas besoin d'être programmeur pour commencer. Il te faut environ **vingt petites idées** en Python et **dix** en NumPy (la bibliothèque pour travailler avec des grilles de nombres). C'est ce que cette leçon te donne, un tout petit pas à la fois.

Pourquoi coder dans un cours de maths ? Parce que le code est une calculatrice qui ne se fatigue jamais et ne ment jamais : si tu affirmes « la somme des 10 premiers carrés vaut 385 », l'ordinateur te dira si c'est vrai. Dans toute l'académie, chaque nombre d'un exemple travaillé vient d'un code que tu peux exécuter.

Tu n'as besoin de rien avant cette leçon, à part l'arithmétique du collège. Les leçons suivantes de cette étape (`mf.02` et la suite) supposent que tu sais lire les programmes présentés ici.

## L'idée en mots simples

Pense à un programme comme à une **recette**. Une recette comporte :

- des **ingrédients étiquetés** (en Python : les *variables*, un nom collé sur une valeur),
- des **étapes répétées** (« mélanger 10 fois » : une *boucle*),
- des **décisions** (« si la sauce est trop épaisse, ajouter de l'eau » : une *conditionnelle*),
- des **sous-recettes nommées** que l'on peut réutiliser (« faire la sauce » : une *fonction*).

NumPy ajoute une grande idée : au lieu de manipuler les nombres un par un, tu manipules **un plateau entier de nombres d'un coup**. Si une colonne de tableur contient 1 000 000 de prix et que tu veux ajouter la taxe à tous, tu n'écris pas un million d'étapes ; tu dis « multiplie toute la colonne par 1.2 ». Cette habitude d'agir sur des plateaux entiers (la pensée « vectorisée ») est l'idée de programmation la plus importante de l'apprentissage automatique.

Où tape-t-on tout cela ? Dans un **notebook** : un document fait de *cellules*, certaines contenant du texte et d'autres du code. Tu exécutes une cellule de code et la réponse apparaît juste en dessous. Un processus Python en arrière-plan (le *noyau*, ou *kernel*) se souvient de chaque variable créée dans les cellules précédentes. C'est pratique, et c'est aussi un piège, comme tu le verras plus bas. Sur cette plateforme, les exercices exécutent Python directement dans ton navigateur, donc tu peux commencer sans rien installer. Plus tard, sur ton propre ordinateur, tu voudras un **environnement virtuel** (un dossier isolé de paquets installés par projet, pour que deux projets qui exigent des versions différentes d'une bibliothèque ne se battent pas).

## Partie 1 : Python, une idée à la fois

### Nombres, variables et affichage

Une **variable** est un nom attaché à une valeur. Le signe `=` signifie ici *affecter*, pas « égale ».

```python
x = 5
y = 2
print(x + y)     # 7
print(x - y)     # 3
print(x * y)     # 10
print(x / y)     # 2.5   (ordinary division, always a decimal)
print(x // y)    # 2     (integer division: how many whole times y fits in x)
print(x % y)     # 1     (remainder)
print(x ** y)    # 25    (x to the power y)
```

Les lignes qui commencent par `#` sont des commentaires : Python les ignore, ce sont des notes pour les humains.

Python respecte l'ordre habituel des opérations (parenthèses, puissances, multiplication/division, addition/soustraction) : `2 + 3 * 4` vaut 14, `(2 + 3) * 4` vaut 20. Nous l'étudions proprement dans `mf.02`.

Deux types de nombres comptent pour l'instant : `int` (nombres entiers, 7) et `float` (décimaux, 7.0 ou 2.5). Les ordinateurs stockent les flottants avec une précision limitée, donc `0.1 + 0.2` donne `0.30000000000000004`. Ce n'est pas un bug de Python ; c'est ainsi que fonctionnent les décimaux binaires. On ne teste jamais des flottants avec `==` ; on demande s'ils sont *proches*.

### Chaînes et booléens

Du texte entre guillemets est une *chaîne de caractères* : `name = "Ada"`. Un **booléen** vaut `True` ou `False`. Les comparaisons donnent des booléens : `3 < 5` vaut `True`, `3 == 5` vaut `False` (le double `==` demande « égal ? »), `3 != 5` vaut `True`.

### Listes et indexation

Une **liste** est une rangée ordonnée de valeurs.

```python
xs = [10, 20, 30, 40]
print(xs[0])        # 10   counting starts at 0
print(xs[2])        # 30
print(xs[-1])       # 40   -1 means the last item
print(len(xs))      # 4
xs.append(50)       # add to the end
print(xs[1:3])      # [20, 30]  slice: start included, stop EXCLUDED
```

Pourquoi compte-t-on à partir de 0 ? Pense à l'indice comme à « combien de pas depuis le début ». Le premier élément est à 0 pas. La tranche `xs[1:3]` prend les éléments aux positions 1 et 2, donc elle contient `3 - 1 = 2` éléments.

Un avertissement qui piège tout le monde une fois : `b = a` pour une liste ne la **copie pas**. Les deux noms pointent vers la même liste. Utilise `b = a.copy()` quand tu veux une copie indépendante.

### Boucles

Une boucle `for` répète un bloc de code pour chaque élément. Le bloc est marqué par l'**indentation** (quatre espaces) ; c'est la façon qu'a Python de dire « ces lignes appartiennent à la boucle ».

```python
total = 0
for v in [3, 5, 8]:
    total = total + v
print(total)        # 16
```

Suis-la à la main, comme tu devrais suivre n'importe quelle boucle :

| étape | v | total après |
|---|---|---|
| début | | 0 |
| 1 | 3 | 3 |
| 2 | 5 | 8 |
| 3 | 8 | 16 |

`range(n)` produit `0, 1, ..., n-1`, et `range(a, b)` produit `a, ..., b-1`. Ainsi `sum_sq = 0; for i in range(1, 4): sum_sq += i*i` calcule $1+4+9 = 14$. (`+=` signifie « ajoute à la variable elle-même ».)

Une boucle d'une ligne qui construit une liste s'appelle une **liste en compréhension** : `[i*i for i in range(1, 4)]` donne `[1, 4, 9]`.

### Conditionnelles

```python
def sign_word(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
```

Python vérifie les conditions de haut en bas et n'exécute que la première branche qui est vraie.

### Fonctions

Une **fonction** prend des entrées (les *arguments*), fait un travail, et `return` un résultat. Écris-la une fois, utilise-la plusieurs fois.

```python
def area_of_circle(r):
    pi = 3.141592653589793
    return pi * r ** 2

print(area_of_circle(1))   # 3.141592653589793
print(area_of_circle(2))   # 12.566370614359172
```

Si une fonction n'a pas de `return`, l'appeler donne `None`. Un bug fréquent chez les débutants est de faire un `print` dans une fonction au lieu d'un `return`, puis de se demander pourquoi le résultat ne peut pas être réutilisé.

### Lire les messages d'erreur

Quand Python échoue, il affiche une *trace d'appels* (traceback). Lis **la dernière ligne en premier** : elle nomme l'erreur (`NameError` : un nom que tu n'as jamais défini, `TypeError`, `IndexError` : tu as demandé un élément qui n'existe pas, `ZeroDivisionError`). Regarde ensuite le numéro de ligne indiqué. Les erreurs sont de l'information, pas des insultes.

## Partie 2 : NumPy, le plateau de nombres

### Pourquoi pas simplement des listes ?

Une liste Python `[1, 2, 3]` fois 2 donne `[1, 2, 3, 1, 2, 3]` : elle se répète. Un tableau NumPy fait le calcul auquel tu t'attends.

```python
import numpy as np          # np is the usual nickname

a = np.array([1, 2, 3])
print(a * 2)                # [2 4 6]
print(a + 10)               # [11 12 13]
print(a ** 2)               # [1 4 9]
print(a + np.array([10, 20, 30]))   # [11 22 33]
```

Chaque opération agit **élément par élément** (« vectorisée »). C'est plus court, et cela s'exécute dans du code compilé rapide plutôt que dans une lente boucle Python.

### Créer des tableaux

```python
np.zeros(3)                 # [0. 0. 0.]
np.ones((2, 3))             # 2 rows, 3 columns of ones
np.arange(5)                # [0 1 2 3 4]
np.arange(2, 10, 3)         # [2 5 8]   start, stop (excluded), step
np.linspace(0, 1, 5)        # [0. 0.25 0.5 0.75 1.]  5 evenly spaced points, both ends included
np.array([[1, 2, 3], [4, 5, 6]])   # a 2x3 table
```

### La forme (shape) : l'attribut le plus important

La **forme** (shape) d'un tableau est un tuple donnant la taille le long de chaque **axe** (direction). Un tableau de 2 lignes et 3 colonnes a pour forme `(2, 3)`. Une simple liste de 3 nombres a pour forme `(3,)` (la virgule en fait un tuple). `a.shape`, `a.ndim` (nombre d'axes) et `a.size` (nombre total d'éléments) te disent tout de sa disposition.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)    # (2, 3)
print(A.ndim)     # 2
print(A.size)     # 6
```

Des mots que tu reverras : un tableau à 1 axe est un **vecteur**, un tableau à 2 axes est une **matrice**, et les tableaux à un nombre quelconque d'axes sont des **tenseurs**. Le code de deep learning, en première approximation, c'est de la comptabilité de formes. Quand quelque chose casse, affiche d'abord les formes.

### Indexation et tranches

Prenons `B = np.arange(12).reshape(3, 4)` :

```
B = [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
```

```python
B[1, 2]        # 6          row 1, column 2
B[0]           # [0 1 2 3]  whole first row
B[:, 0]        # [0 4 8]    ":" means "all"; this is the first column
B[1:, 2:]      # [[6 7] [10 11]]   rows 1.., columns 2..
B[B > 8]       # [ 9 10 11]  boolean mask: keep only where the condition is True
```

`reshape` ré-étiquette les mêmes nombres avec une nouvelle forme (le nombre total doit correspondre) ; `-1` signifie « calcule cette taille toi-même » : `np.arange(12).reshape(3, -1)` a pour forme `(3, 4)`.

### Réductions et l'argument `axis`

`a.sum()`, `a.mean()`, `a.max()`, `a.std()` réduisent un tableau à un seul nombre. Sur un tableau à deux dimensions, tu peux choisir la direction : `axis=0` écrase les lignes (donnant un résultat par colonne), `axis=1` écrase les colonnes (un résultat par ligne). La règle : **l'axe que tu nommes disparaît de la forme**.

```python
B.sum(axis=0)   # [12 15 18 21]   shape (4,)
B.sum(axis=1)   # [ 6 22 38]      shape (3,)
```

### Le broadcasting par l'exemple

Et si les formes diffèrent ? NumPy essaie d'*étirer* les axes de taille 1. La règle : aligne les formes **à partir de la droite** ; deux tailles sont compatibles si elles sont égales, ou si l'une d'elles vaut 1 (ou est absente).

Exemple 1 : une ligne ajoutée à chaque ligne d'un tableau.

```python
B + np.array([100, 200, 300, 400])
```

Formes `(3, 4)` et `(4,)` : à partir de la droite, 4 correspond à 4, et l'axe de tête manquant est étiré, donc la ligne est ajoutée à chacune des 3 lignes. Forme du résultat : `(3, 4)`.

Exemple 2 : une colonne face à une ligne construit un tableau entier (ici la table de multiplication) :

```python
c = np.arange(1, 4)[:, None]   # shape (3, 1)  [:, None] adds an axis
r = np.arange(1, 5)[None, :]   # shape (1, 4)
print(c * r)                   # shape (3, 4): entry [i, j] = (i+1)*(j+1)
```

Exemple 3, une erreur : formes `(3,)` et `(2,)` : 3 et 2 ne sont ni égaux ni égaux à 1, donc NumPy lève `ValueError: operands could not be broadcast together`.

Un usage très courant : **centrer** les colonnes d'un tableau de données `X` de forme `(n, d)` : `X - X.mean(axis=0)`. La moyenne a pour forme `(d,)`, qui se diffuse (broadcast) sur chaque ligne.

## Exemple travaillé

Standardisons la liste 1, 2, 3, 4 : soustrais la moyenne et divise par l'écart type.

1. moyenne = (1+2+3+4)/4 = 2.5
2. écarts : -1.5, -0.5, 0.5, 1.5
3. carrés : 2.25, 0.25, 0.25, 2.25 ; leur moyenne vaut 5/4 = 1.25, donc écart type = $\sqrt{1.25} \approx 1.1180$
4. division : -1.3416, -0.4472, 0.4472, 1.3416

```python
a = np.array([1., 2., 3., 4.])
z = (a - a.mean()) / a.std()
print(z)               # [-1.3416 -0.4472  0.4472  1.3416]
print(z.mean(), z.std())   # ~0 and 1
```

Quatre lignes d'arithmétique condensées en une seule expression vectorisée, et aucune boucle nulle part.

## Idées fausses courantes

- « `=` signifie égal. » Cela signifie affecter. Le test d'égalité s'écrit `==`.
- « On compte à partir de 1. » En Python et NumPy on commence à 0, et les tranches excluent l'indice d'arrêt.
- « `b = a` copie. » Pour les listes et les tableaux, cela ne donne qu'un second nom ; utilise `.copy()`.
- « Les cellules d'un notebook s'exécutent de haut en bas. » Elles s'exécutent dans l'ordre où *tu cliques*, et le noyau garde en vie les anciennes variables. Si les résultats semblent étranges, redémarre le noyau et exécute toutes les cellules dans l'ordre. C'est le piège de notebook le plus courant.
- « Compare les flottants avec `==`. » Utilise `np.isclose(a, b)` ou `abs(a - b) < 1e-9`.
- « Des réponses fausses signifient un bug dans NumPy. » Presque toujours, la forme n'est pas celle que tu crois ; affiche `x.shape` et `x.dtype`.

## Habitudes de débogage (petites mais puissantes)

1. Rends le cas qui échoue minuscule (3 nombres, pas 3 millions).
2. Affiche les formes, les types et quelques valeurs après chaque ligne suspecte.
3. Ne change qu'une chose à la fois.
4. Énonce le problème à voix haute (le « canard en plastique ») : cela révèle souvent l'hypothèse erronée.

## Points clés à retenir

- Python = variables, listes, boucles, conditionnelles, fonctions. NumPy = arithmétique sur tableaux entiers.
- Un tableau a une **forme** ; connais-la toujours. L'indexation et les tranches commencent à 0 et excluent l'arrêt.
- `axis` nomme la direction qui disparaît quand tu réduis.
- Le broadcasting étire les axes de taille 1, en faisant correspondre les formes à partir de la droite.
- Préfère les expressions vectorisées aux boucles, et teste les flottants par proximité, non par égalité.

## Explique-le toi-même

1. Explique à un ami pourquoi `A.sum(axis=0)` sur un tableau `(3, 4)` donne 4 nombres, et non 3.
2. Avec tes propres mots, pourquoi `np.arange(1,4)[:, None] * np.arange(1,5)[None, :]` est-il un tableau à deux dimensions, et quelle forme a-t-il ?

## Pour aller plus loin

- Le guide « absolute beginners » de NumPy et le tutoriel officiel de Python.
- Plus tard dans l'académie, le matériel sur les opérations sur tenseurs étend ces idées aux lots (batches) et aux formes de type attention (`einsum`, redimensionnement).

## Critères de maîtrise

Tu sais : (1) écrire une fonction avec une boucle et une conditionnelle ; (2) créer des tableaux avec `arange`, `linspace`, `zeros` ; (3) prédire la forme du résultat d'une indexation, d'une réduction ou d'un broadcasting avant de l'exécuter ; (4) réécrire une boucle sur des nombres en une seule ligne vectorisée ; (5) lire une trace d'erreur et trouver la ligne fautive.
