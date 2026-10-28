## What this part is for

You will take a neural network from a blank file to a trained, **verified** model without a framework: forward pass, loss, backpropagation with every shape, a training loop, and a proof (against finite differences and PyTorch autograd) that your gradients are right. The goal is not to use libraries but to understand what they do, so that when training fails you know where to look.

**Prerequisite:** the Mathematics part, especially lessons 5 and 6 (gradients, Jacobians). Each lesson's "Requires" panel names the exact earlier idea it needs, and the app warns you when your quiz results say a prerequisite is shaky.

## The route

| # | Lesson | The question it answers | Needs (math / ml) |
|---|---|---|---|
| 7 | Gradient descent | How do we use the gradient to improve, and when does it fail? | M5 |
| 8 | Loss functions | What is "wrong", and why these formulas? | M1 |
| 9 | Layers and the forward pass | What does a network compute? | M3, 8 |
| 10 | Backpropagation | How do we get every gradient at once? | M6, 9 |
| 11 | Training from scratch | How do we make it actually learn, and debug it? | 7, 10 |
| 12 | Autodiff and PyTorch | Is my implementation provably right? | 11 |

## How to study a lesson

The same loop as the Mathematics part, with one addition: **always write the shape of every tensor before you write any code.** In this part almost every bug is a shape bug, and a gradient that does not have the shape of its parameter is wrong before you run anything.

When training misbehaves, follow the ladder from lesson 11 in order: initial loss ≈ ln K, gradient check, overfit 8 examples, learning-rate sweep, only then full training.

## Evidence of mastery

Quizzes and exercises track concept mastery as usual. For this part the real test is **Capstone 1**: you submit your gradient-check numbers, your own derivation of the four gradients, a learning-rate experiment with several seeds, a failure you diagnosed, and the limits of your implementation. Nothing grades it for you; the point is that you could defend it to a reviewer.

## What comes after

The source library holds the next steps already imported from your folders: classical ML, convolutional and recurrent networks, attention and the transformer, and building a GPT. They are source text with quizzes, not yet authored lessons with exercises; the Roadmap page shows exactly what is covered and what is missing.
