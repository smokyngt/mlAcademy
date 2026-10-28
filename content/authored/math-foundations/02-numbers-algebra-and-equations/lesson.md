---
id: mf.02
title: "Numbers, algebra and equations"
summary: Rebuild school algebra carefully: number types, order of operations, powers and roots, fractions and percentages, linear equations and inequalities, two-equation systems, and how to read mathematical notation.
minutes: 100
teaches: [mf.algebra]
requires: []
sources:
  - "math/math for ML.pdf, Ch. 1 Introduction and Motivation (PDF p.17): why the mathematics is needed (data, model, learning). READ."
  - "math/math for ML.pdf, Ch. 2 intro (PDF p.24): vectors as tuples in R^n, scalars, and how that notation corresponds to arrays on a computer. READ."
  - "math/math for ML.pdf, Sec. 2.1 Systems of Linear Equations (PDF p.26-27): each equation is a line, the solution set is an intersection, and the possible outcomes are none, one or infinitely many. READ."
  - "math/math for ML.pdf, Sec. 5 intro (PDF p.145-146): the f: R^D -> R, x |-> f(x) notation for functions, domain and image. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md (opening sections): vectors as lists of numbers with meaning, and why notation matters. READ."
---

## Where this fits

Every formula in machine learning is built from a few moves you already half-know: adding, multiplying, raising to a power, solving for an unknown. A model's prediction such as $\hat y = w x + b$ is just a linear equation. "Train the model" often means "solve an equation (or a pair of equations) for the unknowns $w$ and $b$". The problem is that school algebra is usually taught as recipes without understanding, and then notation like $\mathbb R$, $\sum$ or $f:\mathbb R\to\mathbb R$ looks like another language.

This lesson is the rebuild. We go slowly and check every claim with a tiny piece of code. No earlier lesson is needed except `mf.01`'s ability to read short Python (if that is shaky, skim it first).

## The idea in plain words

An equation is a **balance scale**. The left pan and the right pan weigh the same. The only legal move is to do the *same thing to both pans*: add 5 to both, halve both. As long as the scale stays balanced, you are allowed to simplify until the unknown sits alone on one side. Solving an equation is *undoing*, in reverse order, whatever was done to the unknown. If the story is "I multiplied by 3, then added 5, and got 20", the undo is "subtract 5, then divide by 3".

An inequality is the same scale, but tipped: the left pan is *heavier* than the right. Doing the same thing to both pans keeps it tipped the same way, with one famous exception we will explain: if you flip the sign of both pans (multiply by a negative), left-heavy becomes left-light.

Two unknowns need two scales. Each scale gives a clue; one clue is not enough to pin down two numbers, two independent clues are.

## Number types

| symbol | name | examples |
|---|---|---|
| $\mathbb N$ | natural numbers | 0, 1, 2, 3, ... (some books start at 1) |
| $\mathbb Z$ | integers | ..., -2, -1, 0, 1, 2, ... |
| $\mathbb Q$ | rational numbers (fractions $p/q$) | 1/2, -7/3, 0.25 |
| $\mathbb R$ | real numbers (the whole number line) | $\sqrt 2$, $\pi$, -0.5 |

Each family contains the previous one: $\mathbb N\subset\mathbb Z\subset\mathbb Q\subset\mathbb R$. Why bother to extend? Because each step lets you solve something you could not before: $x+3=1$ needs $\mathbb Z$ (answer $-2$), $2x=1$ needs $\mathbb Q$, and $x^2=2$ needs $\mathbb R$ ($\sqrt2$ is not a fraction). In this academy "a number" without qualification means a real number. (Later, `complex` numbers appear for $x^2=-1$.)

In code, `int` holds whole numbers exactly, and `float` holds real numbers approximately (about 16 digits). Remember: `0.1 + 0.2 != 0.3` exactly in floating point.

## Order of operations

Conventions exist so that everyone reads $2+3\times4$ the same way. The order is:

1. brackets, innermost first;
2. powers and roots;
3. multiplication and division, left to right;
4. addition and subtraction, left to right.

Tiny worked examples (all checkable in Python with `**` for power):

- $2+3\times4 = 2+12 = 14$ (not 20).
- $(2+3)\times4 = 20$.
- $2+3\times4^2$: power first, $4^2=16$; then $3\times16=48$; then $2+48=50$.
- $-3^2$ is $-(3^2)=-9$, but $(-3)^2=9$. The brackets decide.
- $12/3\times2$: left to right, $4\times2=8$ (not $12/6=2$).

```python
print(2 + 3 * 4 ** 2)      # 50
print(-3 ** 2, (-3) ** 2)  # -9 9
print(12 / 3 * 2)          # 8.0
```

## Powers and roots

A power is repeated multiplication: $a^n = a\times a\times\dots\times a$ ($n$ copies), so $2^5=32$. Three rules you can derive by counting copies:

- $a^m\cdot a^n = a^{m+n}$: ($m$ copies) times ($n$ copies) is $m+n$ copies. Example: $2^3\cdot2^2=8\cdot4=32=2^5$.
- $(a^m)^n = a^{mn}$: $n$ groups of $m$ copies. $(2^3)^2=64=2^6$.
- $a^0=1$ and $a^{-n}=1/a^n$: these definitions are *chosen* so the first rule keeps working. Since $a^{3}\cdot a^{0}$ should be $a^{3+0}=a^3$, the number $a^0$ must be 1. Since $a^3\cdot a^{-3}=a^0=1$, we need $a^{-3}=1/a^3$.

A **root** undoes a power: $\sqrt{9}=3$ because $3^2=9$ (and by convention $\sqrt{\ }$ means the non-negative root). Fractional exponents are roots: $a^{1/2}=\sqrt a$, because $(a^{1/2})^2=a^{1}$ by the second rule. So $49^{1/2}=7$ and $8^{1/3}=2$. Exponents and logarithms get their own lesson (`mf.04`).

## Fractions and percentages

A fraction $\frac ab$ means "$a$ parts of size $\frac1b$". To add fractions the parts must be the same size: find a **common denominator**.

$$
\frac ab+\frac cd=\frac{ad+bc}{bd}.
$$

Derivation: $\frac ab=\frac{ad}{bd}$ and $\frac cd=\frac{bc}{bd}$ (multiplying top and bottom by the same number does not change the value), and now both are counted in parts of size $\frac1{bd}$. Example: $\frac12+\frac13=\frac36+\frac26=\frac56$. Wrong (but tempting): $\frac{1+1}{2+3}=\frac25$.

Multiplication is easier: $\frac ab\cdot\frac cd=\frac{ac}{bd}$ ("a part of a part"). Division flips: $\frac ab\div\frac cd=\frac ab\cdot\frac dc$. Reduce by dividing top and bottom by their greatest common divisor (`math.gcd`).

A **percentage** is a fraction out of 100. "$p\%$ of $x$" is $\frac p{100}x$. Changing a quantity by $p\%$ multiplies it by $1+\frac p{100}$ (for a rise) or $1-\frac p{100}$ (for a fall).

Worked example (checked below): a price of 100 rises 20%, then falls 20%. After the rise: $100\times1.2=120$. After the fall: $120\times0.8=96$. You do **not** get back to 100, because the 20% fall applies to the bigger number 120. Percent change from `old` to `new` is $\frac{\text{new}-\text{old}}{\text{old}}\times100$; from 120 to 96 it is $-24/120\times100=-20\%$.

## Solving linear equations

An equation is **linear** in $x$ if $x$ appears only multiplied by constants and added: $ax+b=c$.

Derivation of the general solution, one balanced move per line:

$$
\begin{aligned}
ax + b &= c\\
ax &= c - b \qquad &&\text{(subtract } b \text{ from both sides)}\\
x &= \frac{c-b}{a} \qquad &&\text{(divide both sides by } a\neq 0\text{)}
\end{aligned}
$$

Example: $3x+5=20\Rightarrow3x=15\Rightarrow x=5$. Check by substitution: $3\cdot5+5=20$. Always check: substituting back is the cheapest bug detector in mathematics.

Special cases to recognise: $2x+4=2x+7$ simplifies to $4=7$, false for every $x$ (**no solution**); $2x+4=2(x+2)$ simplifies to $4=4$, true for every $x$ (**infinitely many solutions**). If $a=0$ the formula would divide by zero, which is exactly when these degenerate cases occur.

**Unknowns on both sides:** $5x-2=3x+8$. Collect: subtract $3x$ from both sides, $2x-2=8$; add 2, $2x=10$; so $x=5$.

## Inequalities

The symbols $<,>,\le,\ge$ obey the same balanced-moves rule, with one trap. Solve $-2x+3<9$:

1. Subtract 3 from both sides: $-2x<6$.
2. Divide by $-2$. Dividing by a negative number reverses the order, so the sign **flips**: $x>-3$.

Why? Take $1<2$. Multiply both by $-1$: $-1$ and $-2$, and now $-1>-2$. On the number line, negating mirrors everything. Check with a number: $x=0$ (which is $>-3$) gives $-0+3=3<9$ true; $x=-5$ gives $10<9$ false. The solution set is an interval: $(-3,\infty)$.

## Two equations, two unknowns

Example: $x+y=10$ and $x-y=2$. **Elimination:** adding the equations cancels $y$: $2x=12$, so $x=6$, then $y=10-6=4$. Check both equations: $6+4=10$ and $6-4=2$.

General system $a_1x+b_1y=c_1$, $a_2x+b_2y=c_2$. Multiply the first by $b_2$ and the second by $b_1$:

$$
a_1b_2x+b_1b_2y=c_1b_2,\qquad a_2b_1x+b_1b_2y=c_2b_1 .
$$

Subtract: $(a_1b_2-a_2b_1)x=c_1b_2-c_2b_1$. So

$$
x=\frac{c_1b_2-c_2b_1}{a_1b_2-a_2b_1},\qquad y=\frac{a_1c_2-a_2c_1}{a_1b_2-a_2b_1}.
$$

(The formula for $y$ is found the same way, eliminating $x$.) The denominator $a_1b_2-a_2b_1$ must not be 0. Geometrically each equation is a straight line, and the solution is where the lines cross; the denominator is 0 exactly when the lines are parallel (no solution) or the same line (infinitely many). The textbook states the same trichotomy (no, exactly one, or infinitely many solutions) and uses the pair $4x_1+4x_2=5$, $2x_1-4x_2=1$ as an example, whose single solution is $(1,\tfrac14)$: check $4+1=5$ and $2-1=1$. This idea, solving several linear equations together, is the seed of linear algebra, where it becomes the matrix equation you will meet in later stages.

## Reading and writing mathematical notation

Notation is compressed English. Learn to decompress.

| written | read aloud |
|---|---|
| $x\in\mathbb R$ | "$x$ is an element of the real numbers" |
| $\forall x$ | "for all $x$" |
| $\exists x$ | "there exists an $x$" |
| $x_1,x_2,\dots,x_n$ | $n$ different numbers named by subscript |
| $\sum_{i=1}^{n}x_i$ | "the sum of $x_i$ for $i$ from 1 to $n$" (studied in `mf.06`) |
| $f:\mathbb R\to\mathbb R,\ x\mapsto x^2$ | "$f$ takes a real number and gives a real number, sending $x$ to $x^2$" |
| $\approx$, $\propto$, $:=$ | "approximately", "proportional to", "is defined as" |
| $\mathbb R^n$ | all lists of $n$ real numbers |

The textbook *Mathematics for Machine Learning* writes functions exactly like this, for example $f:\mathbb R^D\to\mathbb R$, $x\mapsto f(x)$, where the set of allowed inputs is called the **domain** and the set of outputs the **image** or codomain. It also describes a vector in $\mathbb R^3$ as a triple of numbers and notes this corresponds to an array on a computer, which is why the NumPy of `mf.01` fits so well. A bold letter such as $\mathbf x$ usually means a whole list of numbers, not a single one.

Reading tip: when a formula looks scary, (1) name every symbol, (2) plug in tiny numbers, (3) code it.

## Worked example, with code

Every number below is reproduced by the exercise functions.

```python
import math
print(2 + 3 * 4 ** 2)                      # 50
print((20 - 5) / 3)                        # 5.0       (3x + 5 = 20)
print(100 * 1.2 * 0.8)                     # 96.0      (up 20%, down 20%)
print((1*3 + 1*2, 2*3), math.gcd(5, 6))    # (5, 6) 1   -> 1/2 + 1/3 = 5/6

# system 2x + 3y = 13 and x - y = -1
a1, b1, c1, a2, b2, c2 = 2, 3, 13, 1, -1, -1
det = a1*b2 - a2*b1                        # -5
print((c1*b2 - c2*b1) / det, (a1*c2 - a2*c1) / det)   # 2.0 3.0
```

Check by hand: $2\cdot2+3\cdot3=13$ and $2-3=-1$. Both hold.

## Common misconceptions

- $\frac ab+\frac cd=\frac{a+c}{b+d}$. False; use a common denominator.
- "+20% then -20% returns to the start." It gives 96% of the start.
- Forgetting to flip an inequality when dividing by a negative number.
- $(a+b)^2=a^2+b^2$. False: $(a+b)^2=a^2+2ab+b^2$ (try $a=b=1$: 4, not 2).
- $\sqrt{a+b}=\sqrt a+\sqrt b$. False (try $9+16$: $\sqrt{25}=5\neq3+4$).
- Doing an operation to only one side of an equation.
- Trusting the answer without substituting back.

## Key takeaways

- An equation is a balance; do the same to both sides and undo operations in reverse order.
- Order of operations: brackets, powers, times/divide, plus/minus; a negative sign and an exponent need brackets to be explicit.
- Powers are repeated multiplication; $a^0=1$, $a^{-n}=1/a^n$, $a^{1/2}=\sqrt a$ keep the exponent rules consistent.
- Fractions add through common denominators; percent changes multiply.
- Linear equation: $x=(c-b)/a$. Dividing an inequality by a negative flips it. Two lines meet at one point unless they are parallel or identical.
- Notation is compressed English; decode symbol by symbol.

## Explain it yourself

1. Explain to a friend why a 20% rise followed by a 20% fall leaves you below where you started.
2. Why does multiplying an inequality by $-1$ reverse it? Use a number line.

## Further reading

- Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* (the notation conventions in Chapters 1-2; we return to it from `mf.03` onwards).
- Any school algebra text for extra practice with factorising and expanding brackets.

## Mastery criteria

You can: evaluate an expression in the correct order; add and reduce fractions; compute percentage changes and compose them; solve $ax+b=c$ and a linear inequality (with the flip); solve a 2x2 system by elimination and recognise the no/infinite-solution cases; read $f:\mathbb R\to\mathbb R$, $\in$, $\forall$, $\exists$ aloud.
