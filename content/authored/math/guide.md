## What this part is for

Machine learning looks like magic until you see that it is built from a handful of mathematical tools used over and over. This part teaches exactly those tools, in the order they depend on each other, and *only* those. When you finish you can read the equation `∇_W L = Xᵀ(P − Y)/B` and say what every symbol is, what shape it has, and where it comes from.

You start from zero: if you can add, multiply and read a function like `f(x) = x²`, you can start lesson 1.

## The route

| # | Lesson | The question it answers | Needs |
|---|---|---|---|
| 1 | Functions, summation, logarithms | What do the symbols mean, and why do we compute in log space? | nothing |
| 2 | Vectors and the dot product | How do we measure how alike two lists of numbers are? | 1 |
| 3 | Matrices and linear maps | What is a layer, geometrically, and what are its shapes? | 2 |
| 4 | Derivatives | How does the output react to a tiny nudge of the input? | 1 |
| 5 | Gradients | With many inputs, which direction is uphill? | 2, 4 |
| 6 | Chain rule, Jacobians | How do rates of change flow through a composition? | 3, 4, 5 |

Lessons 1-3 are algebra and geometry (the *language* of data); 4-6 are calculus (the *language of change*). Lesson 6 is the bridge: its three backward formulas are the whole of backpropagation, which the Machine Learning part then builds on.

## How to study a lesson (the loop that works)

1. **Predict (2 min).** Read only the objectives and write one sentence guessing the main idea. A written wrong guess teaches more than a right feeling.
2. **Read for intuition.** Skip the derivation on the first pass; say the idea aloud without symbols.
3. **Derive on paper.** Close the page and reproduce each derivation yourself. When you cannot, find the single step you missed, then close it and try again. This is the highest-value habit.
4. **Run the numbers.** Redo the worked example by hand, then play with the interactive figure until you can predict what a slider does *before* moving it.
5. **Code it.** Do the exercise without hints first. Tests tell you *what* fails; shapes ("what must the shape of this be?") tell you *why*.
6. **Self-test**, then **explain it yourself** with the prompts at the end. If you cannot, you do not own it yet.
7. **Come back.** Flashcards resurface it on a schedule so it lasts months, not days.

## How progress is measured

Reading earns nothing by itself. The app records **mastery per concept** from evidence: quiz answers, exercises solved, flashcard recall. A concept is mastered at 80% with at least three pieces of evidence, so luck cannot get you there. Missing the same concept twice sends you back to its weakest prerequisites. Finish the part with the cumulative assessment, then move on to Machine Learning.
