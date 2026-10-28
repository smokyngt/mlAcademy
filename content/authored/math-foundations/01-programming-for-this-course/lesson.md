---
id: mf.01
title: "Programming for this course: Python and NumPy"
summary: Learn just enough Python and NumPy, from scratch, to run every example and exercise in the academy.
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

## Where this fits

This is the very first lesson of the academy. Every later lesson asks you to run a small program: to check a number, to see a formula behave, to build a model from scratch. You do not need to be a programmer to start. You need roughly **twenty small ideas** in Python and **ten** in NumPy (the library for working with grids of numbers). That is what this lesson gives you, one tiny step at a time.

Why code in a math course? Because code is a calculator that never gets tired and never lies: if you claim "the sum of the first 10 squares is 385", the computer will tell you whether that is true. Throughout the academy, every number in a worked example comes from code you can run.

You need nothing before this lesson except school arithmetic. Later lessons in this stage (`mf.02` onward) assume you can read the programs shown here.

## The idea in plain words

Think of a program as a **recipe**. A recipe has:

- **ingredients with labels** (in Python: *variables*, a name stuck on a value),
- **steps repeated** ("stir 10 times": a *loop*),
- **decisions** ("if the sauce is too thick, add water": a *conditional*),
- **named sub-recipes** you can reuse ("make the sauce": a *function*).

NumPy adds one big idea: instead of handling numbers one at a time, you handle a **whole tray of numbers at once**. If a spreadsheet column holds 1 000 000 prices and you want to add tax to all of them, you do not write a million steps; you say "multiply the whole column by 1.2". This habit of acting on whole trays ("vectorised" thinking) is the single most important programming idea of machine learning.

Where do you type all this? In a **notebook**: a document made of *cells*, some text and some code. You run a code cell and the answer appears right below it. A background Python process (the *kernel*) remembers every variable you created in earlier cells. That is convenient and also a trap, as you will see below. On this platform, the exercises run Python directly in your browser, so you can start without installing anything. Later, on your own computer, you will want a **virtual environment** (an isolated folder of installed packages per project, so two projects needing different library versions do not fight).

## Part 1: Python, one idea at a time

### Numbers, variables, and printing

A **variable** is a name attached to a value. The sign `=` here means *assign*, not "equals".

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

Lines starting with `#` are comments: Python ignores them, and they are notes for humans.

Python respects the usual order of operations (brackets, powers, multiplication/division, addition/subtraction): `2 + 3 * 4` is 14, `(2 + 3) * 4` is 20. We study this properly in `mf.02`.

Two number types matter now: `int` (whole numbers, 7) and `float` (decimals, 7.0 or 2.5). Computers store floats with limited precision, so `0.1 + 0.2` gives `0.30000000000000004`. That is not a bug in Python; it is how binary decimals work. We never test floats with `==`; we ask whether they are *close*.

### Strings and booleans

Text in quotes is a *string*: `name = "Ada"`. A **boolean** is `True` or `False`. Comparisons give booleans: `3 < 5` is `True`, `3 == 5` is `False` (double `==` asks "equal?"), `3 != 5` is `True`.

### Lists and indexing

A **list** is an ordered row of values.

```python
xs = [10, 20, 30, 40]
print(xs[0])        # 10   counting starts at 0
print(xs[2])        # 30
print(xs[-1])       # 40   -1 means the last item
print(len(xs))      # 4
xs.append(50)       # add to the end
print(xs[1:3])      # [20, 30]  slice: start included, stop EXCLUDED
```

Why does counting start at 0? Think of the index as "how many steps from the start". The first item is 0 steps away. The slice `xs[1:3]` takes the items at positions 1 and 2, so it has `3 - 1 = 2` items.

A warning that bites everyone once: `b = a` for a list does **not** copy it. Both names point to the same list. Use `b = a.copy()` when you want an independent copy.

### Loops

A `for` loop repeats a block of code for each item. The block is marked by **indentation** (four spaces); that is Python's way of saying "these lines belong to the loop".

```python
total = 0
for v in [3, 5, 8]:
    total = total + v
print(total)        # 16
```

Trace it by hand, the way you should trace any loop:

| step | v | total after |
|---|---|---|
| start | | 0 |
| 1 | 3 | 3 |
| 2 | 5 | 8 |
| 3 | 8 | 16 |

`range(n)` produces `0, 1, ..., n-1`, and `range(a, b)` produces `a, ..., b-1`. So `sum_sq = 0; for i in range(1, 4): sum_sq += i*i` computes $1+4+9 = 14$. (`+=` means "add to the variable itself".)

A one-line loop that builds a list is a **list comprehension**: `[i*i for i in range(1, 4)]` is `[1, 4, 9]`.

### Conditionals

```python
def sign_word(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
```

Python checks the conditions from top to bottom and runs only the first branch that is true.

### Functions

A **function** takes inputs (*arguments*), does work, and `return`s a result. Write it once, use it many times.

```python
def area_of_circle(r):
    pi = 3.141592653589793
    return pi * r ** 2

print(area_of_circle(1))   # 3.141592653589793
print(area_of_circle(2))   # 12.566370614359172
```

If a function has no `return`, calling it gives `None`. A frequent beginner bug is to `print` inside a function instead of `return`ing, and then wonder why the result cannot be reused.

### Reading error messages

When Python fails it prints a *traceback*. Read the **last line first**: it names the error (`NameError`: a name you never defined, `TypeError`, `IndexError`: you asked for an item that does not exist, `ZeroDivisionError`). Then look at the line number it points to. Errors are information, not insults.

## Part 2: NumPy, the tray of numbers

### Why not just lists?

A Python list `[1, 2, 3]` times 2 gives `[1, 2, 3, 1, 2, 3]`: it repeats. A NumPy array does the maths you expect.

```python
import numpy as np          # np is the usual nickname

a = np.array([1, 2, 3])
print(a * 2)                # [2 4 6]
print(a + 10)               # [11 12 13]
print(a ** 2)               # [1 4 9]
print(a + np.array([10, 20, 30]))   # [11 22 33]
```

Every operation acts **element by element** ("vectorised"). It is shorter, and it runs in fast compiled code instead of a slow Python loop.

### Creating arrays

```python
np.zeros(3)                 # [0. 0. 0.]
np.ones((2, 3))             # 2 rows, 3 columns of ones
np.arange(5)                # [0 1 2 3 4]
np.arange(2, 10, 3)         # [2 5 8]   start, stop (excluded), step
np.linspace(0, 1, 5)        # [0. 0.25 0.5 0.75 1.]  5 evenly spaced points, both ends included
np.array([[1, 2, 3], [4, 5, 6]])   # a 2x3 table
```

### Shape: the most important attribute

The **shape** of an array is a tuple with the size along each **axis** (direction). A table with 2 rows and 3 columns has shape `(2, 3)`. A plain list of 3 numbers has shape `(3,)` (the comma makes it a tuple). `a.shape`, `a.ndim` (number of axes) and `a.size` (total count) tell you everything about its layout.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)    # (2, 3)
print(A.ndim)     # 2
print(A.size)     # 6
```

Words you will meet again: a 1-axis array is a **vector**, a 2-axis array is a **matrix**, and arrays with any number of axes are **tensors**. Deep learning code is, to a first approximation, shape bookkeeping. When something breaks, print the shapes first.

### Indexing and slicing

Take `B = np.arange(12).reshape(3, 4)`:

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

`reshape` re-labels the same numbers with a new shape (the total count must match); `-1` means "work this size out": `np.arange(12).reshape(3, -1)` has shape `(3, 4)`.

### Reductions and the `axis` argument

`a.sum()`, `a.mean()`, `a.max()`, `a.std()` boil an array down to one number. On a table you can choose the direction: `axis=0` collapses the rows (giving one result per column), `axis=1` collapses the columns (one result per row). The rule: **the axis you name disappears from the shape**.

```python
B.sum(axis=0)   # [12 15 18 21]   shape (4,)
B.sum(axis=1)   # [ 6 22 38]      shape (3,)
```

### Broadcasting by example

What if the shapes differ? NumPy tries to *stretch* size-1 axes. The rule: line the shapes up **from the right**; two sizes are compatible if they are equal, or if one of them is 1 (or missing).

Example 1: a row added to every row of a table.

```python
B + np.array([100, 200, 300, 400])
```

Shapes `(3, 4)` and `(4,)`: from the right, 4 matches 4, and the missing leading axis is stretched, so the row is added to each of the 3 rows. Result shape `(3, 4)`.

Example 2: a column against a row builds a whole table (here the multiplication table):

```python
c = np.arange(1, 4)[:, None]   # shape (3, 1)  [:, None] adds an axis
r = np.arange(1, 5)[None, :]   # shape (1, 4)
print(c * r)                   # shape (3, 4): entry [i, j] = (i+1)*(j+1)
```

Example 3, an error: shapes `(3,)` and `(2,)`: 3 and 2 are neither equal nor 1, so NumPy raises `ValueError: operands could not be broadcast together`.

A very common use: **centering** the columns of a data table `X` of shape `(n, d)`: `X - X.mean(axis=0)`. The mean has shape `(d,)`, which broadcasts over every row.

## Worked example

Standardise the list 1, 2, 3, 4: subtract the mean and divide by the standard deviation.

1. mean = (1+2+3+4)/4 = 2.5
2. deviations: -1.5, -0.5, 0.5, 1.5
3. squares: 2.25, 0.25, 0.25, 2.25; their average is 5/4 = 1.25, so std = $\sqrt{1.25} \approx 1.1180$
4. divide: -1.3416, -0.4472, 0.4472, 1.3416

```python
a = np.array([1., 2., 3., 4.])
z = (a - a.mean()) / a.std()
print(z)               # [-1.3416 -0.4472  0.4472  1.3416]
print(z.mean(), z.std())   # ~0 and 1
```

Four lines of arithmetic collapsed into one vectorised expression, and no loop appears anywhere.

## Common misconceptions

- "`=` means equal." It means assign. Equality testing is `==`.
- "Counting starts at 1." In Python and NumPy it starts at 0, and slices exclude the stop index.
- "`b = a` copies." For lists and arrays it only gives a second name; use `.copy()`.
- "Notebook cells run top to bottom." They run in the order *you click*, and the kernel keeps old variables alive. If results look strange, restart the kernel and run all cells in order. This is the most common notebook trap.
- "Compare floats with `==`." Use `np.isclose(a, b)` or `abs(a - b) < 1e-9`.
- "Wrong answers mean a bug in NumPy." Almost always the shape is not what you think; print `x.shape` and `x.dtype`.

## Debugging habits (small but powerful)

1. Make the failing case tiny (3 numbers, not 3 million).
2. Print shapes, types and a few values after each suspicious line.
3. Change one thing at a time.
4. Say the problem aloud ("rubber duck"): it often exposes the wrong assumption.

## Key takeaways

- Python = variables, lists, loops, conditionals, functions. NumPy = whole-array arithmetic.
- An array has a **shape**; always know it. Indexing and slicing start at 0 and exclude the stop.
- `axis` names the direction that disappears when you reduce.
- Broadcasting stretches size-1 axes, matching shapes from the right.
- Prefer vectorised expressions over loops, and test floats with closeness, not equality.

## Explain it yourself

1. Explain to a friend why `A.sum(axis=0)` on a `(3, 4)` table gives 4 numbers, not 3.
2. In your own words, why is `np.arange(1,4)[:, None] * np.arange(1,5)[None, :]` a table, and what shape does it have?

## Further reading

- The NumPy "absolute beginners" guide and the official Python tutorial.
- Later in the academy, the tensor-operations material extends these ideas to batches and attention-style shapes (`einsum`, reshaping).

## Mastery criteria

You can: (1) write a function with a loop and a conditional; (2) create arrays with `arange`, `linspace`, `zeros`; (3) predict the shape of an indexing, reduction or broadcasting result before running it; (4) rewrite a loop over numbers as one vectorised line; (5) read a traceback and find the failing line.
