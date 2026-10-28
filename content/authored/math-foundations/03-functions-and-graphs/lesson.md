---
id: mf.03
title: "Functions and graphs"
summary: Treat a function as an input-output machine, read its graph, meet linear, quadratic and polynomial functions, and learn to shift, scale, reflect, compose and invert functions, checking everything numerically.
minutes: 100
teaches: [mf.functions]
requires: []
sources:
  - "math/math for ML.pdf, Sec. 2.1 (PDF p.26-28): each linear equation is a line, intersections of lines; used for the graph picture of linear functions. READ."
  - "math/math for ML.pdf, Sec. 2.7.1 area (PDF p.62-63): composition of mappings written with the circle symbol (the notation for composing maps). READ."
  - "math/math for ML.pdf, Sec. 5 intro (PDF p.145-146): function notation f: R^D -> R, domain and image/codomain, one output per input. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/docs/en.md (opening to the derivative section): functions like x^2 tabulated, slopes, 'bowl' shape and where its minimum is. READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md (opening): why mapping inputs to outputs is the language of models. READ."
---

## Where this fits

A neural network is a function. A loss is a function. A learning algorithm walks along the graph of a function looking for its lowest point. Before any of that, you need to be fluent with the idea itself: what a function is, how its graph looks, what simple families (lines, parabolas, polynomials) can and cannot do, and how functions can be moved, combined and undone.

Machine learning is almost entirely "build complicated functions by composing simple ones". When you meet a deep network, you will recognise it as `f3(f2(f1(x)))`, exactly the composition we practise here. And "minimise the loss" will mean finding the bottom of a parabola-like graph, which is why we study the parabola so carefully.

No lesson is required before this one. You use the arithmetic and equation-solving of `mf.02` (re-read its linear-equation section if needed) and the code habits of `mf.01`.

## The idea in plain words

A function is a **machine with an input slot and an output slot**. You drop in a number, the machine does the same fixed procedure every time, and one number drops out. The one-output rule is the whole definition: feed in the same input twice, get the same output twice. A vending machine is a good picture: press B4, always the same snack.

The **domain** is the list of buttons that exist (the inputs the machine accepts). The **range** is the snacks that actually come out. Some buttons may be jammed (the machine cannot take square roots of negative numbers, or divide by zero), which is why a domain is not always "everything".

A **graph** is the machine's behaviour drawn as a picture: horizontal position is the input, height is the output. A function is the same object as its graph, but the picture lets you see shape: rising, falling, a bowl, a bump.

Then we play with machines:

- **Shift** the machine's behaviour left, right, up, down.
- **Stretch** or **flip** it.
- **Chain** two machines: the output of the first is dropped into the second (composition).
- **Run it backwards** (inverse): given the output, recover the input, which only works if no two inputs give the same output.

## Definitions

A **function** $f$ from a set $A$ (the domain) to a set $B$ assigns to each $x\in A$ exactly one $f(x)\in B$. We write $f:A\to B$, $x\mapsto f(x)$. This is the same arrow notation used for functions such as $f:\mathbb R^D\to\mathbb R$ in the reference textbook, where $\mathbb R^D$ is the domain and the values live in the image/codomain.

Vocabulary: **input / argument** $x$ (independent variable), **output / value** $f(x)$, often called $y$. The **natural domain** is every real input for which the formula makes sense. Examples:

| formula | natural domain | why |
|---|---|---|
| $x^2$ | all real $x$ | nothing breaks |
| $1/x$ | $x\neq0$ | no division by zero |
| $\sqrt{x-1}$ | $x\ge1$ | no square root of a negative |
| $\sqrt{x}$ range | outputs $\ge0$ | a square root is non-negative |

The **range** of $x^2$ is $[0,\infty)$: it never gives a negative number.

**Is it a function?** Vertical-line test: a vertical line meets a function's graph at most once. The circle $x^2+y^2=1$ fails it (at $x=0$, both $y=1$ and $y=-1$), so $y$ is not a function of $x$ there.

### Evaluating (tiny worked examples)

For $f(x)=2x^2-3$: $f(0)=-3$, $f(1)=-1$, $f(-2)=2\cdot4-3=5$, $f(a+1)=2(a+1)^2-3=2a^2+4a-1$. The last one is a classic mistake spot: $f(a+1)$ is *not* $f(a)+1$. Substitute the whole input.

```python
f = lambda x: 2 * x ** 2 - 3
print(f(0), f(1), f(-2))   # -3 -1 5
```

## Linear functions

$f(x)=mx+b$. Two numbers describe everything:

- the **slope** $m$ = change in output per 1 unit of input; given two points,

$$
m=\frac{y_2-y_1}{x_2-x_1};
$$

- the **intercept** $b=f(0)$, where the line meets the vertical axis.

The line through $(0,1)$ and $(2,5)$: $m=(5-1)/(2-0)=2$, $b=1$, so $f(x)=2x+1$. Check: $f(2)=5$. The defining property is **constant rate**: every 1-unit step in $x$ adds exactly $m$ to the output, forever. This is the model $\hat y=wx+b$ of linear regression, with $w$ the slope.

## Quadratic functions

$f(x)=ax^2+bx+c$, $a\ne0$. The graph is a **parabola**: a bowl opening up if $a>0$, down if $a<0$.

**Where is the bottom (vertex)?** Complete the square. Take $a>0$:

$$
\begin{aligned}
ax^2+bx+c &= a\left(x^2+\tfrac{b}{a}x\right)+c\\
&= a\left(x+\tfrac{b}{2a}\right)^2 - \tfrac{b^2}{4a}+c .
\end{aligned}
$$

The squared bracket is never negative and is 0 only when $x=-\frac b{2a}$. So the lowest point is at $x=-\frac b{2a}$, and the lowest value is $c-\frac{b^2}{4a}$. Example: $x^2-4x+3$ has vertex at $x=4/2=2$ with value $4-8+3=-1$.

**Where does it cross zero (roots)?** Set the completed-square form to 0 and solve:

$$
x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}.
$$

The **discriminant** $b^2-4ac$ decides: positive gives two real roots, zero gives one, negative gives none (the bowl floats above the axis). For $x^2-4x+3$: $16-12=4$, roots $\frac{4\pm2}{2}=3$ and $1$; they sit symmetrically around the vertex at 2. For $x^2+1$: discriminant $-4<0$, no real roots.

The parabola matters hugely: near any smooth minimum, a loss function looks like a bowl, so "find the bottom" reduces to the vertex formula. Squared error $(y-wx)^2$ is a quadratic in $w$.

## Polynomials

$p(x)=a_nx^n+\dots+a_1x+a_0$ with $a_n\ne0$; $n$ is the **degree** (line: 1, parabola: 2, cubic: 3). A degree-$n$ polynomial has at most $n$ real roots and at most $n-1$ turning points. Evaluate efficiently with **Horner's rule**, which nests the multiplications:

$$
a_3x^3+a_2x^2+a_1x+a_0=((a_3x+a_2)x+a_1)x+a_0 .
$$

```python
def poly_eval(coeffs, x):          # coeffs highest power first
    result = 0
    for c in coeffs:
        result = result * x + c
    return result

print(poly_eval([1, 0, -1, 0], 3))  # x^3 - x at 3 -> 24
```

Trace for $x^3-x$ at $x=3$ (coeffs 1,0,-1,0): result goes 1, then $1\cdot3+0=3$, then $3\cdot3-1=8$, then $8\cdot3+0=24$. Check: $27-3=24$. Polynomials are the first model class beyond lines; with high degree they can wiggle through any points, which is the origin of overfitting.

## Transformations of graphs

Start with a base function $f$. These rules move its graph; verify each with numbers.

| new function | what happens to the graph |
|---|---|
| $f(x)+k$ | shift **up** by $k$ |
| $f(x-h)$ | shift **right** by $h$ |
| $c\,f(x)$, $c>1$ | stretch vertically by $c$ |
| $f(x/c)$, $c>1$ | stretch horizontally by $c$ |
| $-f(x)$ | reflect over the $x$-axis |
| $f(-x)$ | reflect over the $y$-axis |

**Why does $f(x-h)$ move right?** The new machine at input $x$ does what the old machine did at $x-h$. The old special point (say the vertex at $x=0$ for $x^2$) now appears where $x-h=0$, i.e. at $x=h$. Example: $g(x)=(x-3)^2$ has $g(3)=0$, $g(5)=4$, $g(1)=4$: the bowl of $x^2$ now sits at 3. The sign feels backwards at first; the "what input does the old machine see?" question always gets it right.

Combining: $a\,f(x-h)+k$ first shifts right by $h$, stretches by $a$, then lifts by $k$. Completing the square above says every parabola is just $x^2$ shifted, stretched and maybe flipped.

## Composition

$(f\circ g)(x)=f(g(x))$ means: **apply $g$ first, then $f$**. With $f(x)=x+1$ and $g(x)=2x$:

- $(f\circ g)(3)=f(6)=7$
- $(g\circ f)(3)=g(4)=8$

Order matters. The circle notation for composing maps is used throughout the reference textbook (e.g. a map written as a composition of three simpler maps). In deep learning a network layer is "linear function, then a nonlinear squashing function", and a deep network is a long composition of such layers; training it later uses the chain rule, which is precisely the rule for the rate of change of a composition.

Domain caution: $f\circ g$ is defined only where $g$'s output lands inside $f$'s domain. If $f=\sqrt{\ }$ and $g(x)=x-1$ then $f(g(x))=\sqrt{x-1}$ needs $x\ge1$.

## Inverse functions

The inverse $f^{-1}$ undoes $f$: $f^{-1}(f(x))=x$ and $f(f^{-1}(y))=y$. (The $-1$ is a label, not a power: $f^{-1}(y)\neq1/f(y)$.)

**Recipe:** write $y=f(x)$, solve for $x$, then rename. For $f(x)=3x-6$: $y=3x-6\Rightarrow x=\frac{y+6}{3}$, so $f^{-1}(y)=\frac{y+6}{3}$. Check: $f^{-1}(9)=5$ and $f(5)=9$.

**When does it exist?** Only if each output comes from exactly one input (the **horizontal-line test**: no horizontal line meets the graph twice). $f(x)=x^2$ fails on all of $\mathbb R$: 2 and $-2$ both map to 4, so given 4 you cannot tell which to return. Restricting the domain to $x\ge0$ fixes it, and the inverse is then $\sqrt{\ }$. Geometrically the graph of $f^{-1}$ is the graph of $f$ mirrored in the diagonal line $y=x$.

For a function that keeps rising there is a universal numeric method, **bisection**: to find $x$ with $f(x)=y$, keep an interval $[lo,hi]$ that brackets the answer, test the midpoint, and discard the half that cannot contain it. Sixty halvings shrink an interval of width 200 to about $2\times10^{-16}$. This is how calculators invert complicated functions.

## Worked example: checking by evaluation

```python
import numpy as np
f = lambda x: x**2 - 4*x + 3
xs = np.array([0., 1., 2., 3., 4.])
print(f(xs))                       # [ 3.  0. -1.  0.  3.]  roots 1 and 3, vertex (2,-1)

g = lambda x: f(x - 3)             # shifted right by 3
print(g(xs + 3))                   # same list: g at x+3 equals f at x

h = lambda x: 3*x - 6
h_inv = lambda y: (y + 6) / 3
print(h_inv(h(xs)))                # [0. 1. 2. 3. 4.]  the machine is undone
```

Every printed list can be verified by hand with the rules above.

## Common misconceptions

- "$f(a+b)=f(a)+f(b)$." Only true for special (linear through the origin) functions. $f(x)=x^2$: $f(1+1)=4\neq2$.
- "$f(x-3)$ moves left." It moves right.
- "$f^{-1}(x)=1/f(x)$." No, that is the reciprocal, a different thing.
- "Every function has an inverse." Only if different inputs give different outputs (on the chosen domain).
- "$f\circ g=g\circ f$." Rarely true.
- "A curve is a function." Only if each $x$ has one $y$.
- Reading a graph far outside the plotted window as if the trend continues. A polynomial can behave wildly outside the region where you tested it.

## Key takeaways

- A function gives exactly one output per input; know its domain and range.
- Lines have constant slope; parabolas have a vertex at $-b/2a$ and roots from the quadratic formula; polynomial degree bounds roots and turning points.
- Shifts, stretches and reflections are systematic; $f(x-h)$ shifts right.
- Composition applies the inner function first, and order matters.
- An inverse undoes a function and exists only when the function is one-to-one; check any claimed inverse by evaluating both compositions at several points.

## Explain it yourself

1. Why does replacing $x$ by $x-3$ move a graph to the right? Argue with the vertex of $x^2$.
2. Why does $x^2$ have no inverse on all real numbers, and how does restricting the domain repair it?

## Further reading

- The reference textbook (Sec. 2.1 and the start of Ch. 5) for the line-intersection picture and function notation.
- Later in the academy: derivatives measure the slope of a curve at a point (`mf.07` prepares that), and the chain rule differentiates compositions.

## Mastery criteria

You can: evaluate and tabulate a function; find a natural domain; read slope and intercept; find a parabola's vertex and roots (and use the discriminant); predict the effect of $f(x-h)+k$, $cf(x)$, $-f(x)$, $f(-x)$; compute $f\circ g$ and $g\circ f$ and show they differ; find an inverse algebraically and test it numerically.
