---
id: app.01
title: "Prompting fundamentals: instructions, examples, reasoning and voting"
summary: Treat a prompt as a testable program that conditions a next-token predictor, and learn the four levers (instructions, roles, examples, reasoning) plus self-consistency voting with its exact binomial gain.
minutes: 70
teaches: [app.prompting, app.self-consistency]
requires: [llm.lm, llm.sampling, trf.architectures, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/11-llm-engineering/01-prompt-engineering/docs/en.md (opening sections: the vague-versus-engineered prompt example, anatomy of a prompt, system/user/prefill, role prompting; first ~5.5k characters only). READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/02-few-shot-cot/docs/en.md (opening sections: zero-shot versus few-shot, example selection, chain-of-thought motivation; first ~5.5k characters only). READ."
---

## Where this fits

You have met a language model as a function that maps a token sequence to a probability distribution over the next token (lesson `trf.04` built the architecture, and the decoding lessons showed how to sample from it). A **prompt** is simply the token sequence you put in front of that function. Everything an application does with a hosted model, from a support bot to a code reviewer, starts as text in a prompt. This lesson is the first of the application course: it teaches how to *control* the model through its input, how to buy extra accuracy with more computation at inference time (reasoning and voting), and how to treat prompts as artifacts you can test rather than as folklore. Later lessons add structure to the output ([[app.02]]), manage the whole window ([[app.03]]) and measure quality ([[app.04]]).

## The idea in plain words

Imagine hiring a very well-read temp worker who has read most of the library but knows nothing about your company and takes every sentence literally. If you say "write me an email", you get an average email, because "average" is what the library contains. If you say who the reader is, how long it should be, what tone to use and show one email you liked, you get something close to what you wanted. A prompt does exactly this: it **narrows the set of continuations** the model considers plausible.

Four levers do most of the work:

1. **Instructions**: say precisely what you want, in what format, with what constraints, and what to do when unsure.
2. **Role or context**: tell the model who it is writing as and for whom. This does not add knowledge; it shifts which part of what the model has seen it imitates.
3. **Examples** (few-shot): show two to five input-output pairs. An example is a compressed instruction; the model copies the pattern, including accidental quirks.
4. **Room to think** (chain-of-thought): let the model write intermediate steps before the answer. The model has no private scratch space between tokens; whatever it writes becomes input for the next token. Steps written down are computation done.

A model sampled with some randomness is also like a student who sometimes slips. If you ask the same hard question five times and take the answer most students agree on, you are usually more right than any single student. That is **self-consistency**.

## Anatomy of a prompt

A chat API call has up to three parts:

- **System message**: persistent rules and identity, written once, placed first.
- **User message**: this turn's task or question.
- **Assistant prefill** (offered by some providers): the first words of the answer, written by you, which the model must continue. Starting the reply with an opening brace pushes the model into JSON without chatter.

Practical rules that follow from "the model reads literally":

- Put stable material (rules, definitions) before variable material (the user's input). This matters again in [[app.03]] for caching.
- State the output format and what to do on missing information ("answer `unknown` if the text does not say").
- Prefer positive, concrete statements ("answer in at most 3 sentences") to vague ones ("be brief").
- Separate instructions from data with clear delimiters, so text inside the data is not mistaken for an instruction (this becomes a security matter in [[app.05]]).

**Few-shot choices.** Three to five examples are typical; fewer give too little signal, more cost context for diminishing returns. Pick examples similar to the input, cover every label, and match difficulty. Examples win on format-sensitive tasks; for simple factual questions or creative work they can constrain more than they help. A source we read quotes specific accuracy percentages for these effects; treat any such figure as dependent on model and task, and measure on your own data.

## Chain-of-thought and its cost

Chain-of-thought (CoT) prompting asks for the reasoning steps first and the answer last, either by instruction ("think step by step") or by showing worked examples. Mechanically, a transformer spends a fixed amount of computation per generated token, so a problem needing many dependent steps cannot always be solved "in one token", but can be solved across many. The price is more output tokens (cost and latency, see [[app.06]]) and a new failure mode: a fluent but wrong chain.

Two practical consequences:

- **Put the reasoning before the answer** in any structured format. A JSON object with `"answer"` before `"reasoning"` commits to the answer first, and the "reasoning" that follows is just a justification.
- **Parse the final answer robustly.** Free-form reasoning ends in many shapes; you need a rule such as "the integer after the last occurrence of the word answer".

## Self-consistency: voting over sampled reasoning

Sample $n$ independent chains of thought at temperature above zero, extract each final answer, and return the **most frequent** one (majority vote). Different chains make different mistakes, but correct chains agree.

### Derivation: accuracy of a majority vote

Assume each sample is correct with probability $p$, independently, and that wrong samples give *different* wrong answers (so wrong answers never gang up on the right one). Then the vote is correct exactly when more than half of the $n$ samples ($n$ odd) are correct. The number of correct samples $K$ is binomial:

$$
P(K=k)=\binom{n}{k}p^{k}(1-p)^{n-k}.
$$

So the majority is right with probability

$$
A(p,n)=\sum_{k=\lceil n/2\rceil}^{n}\binom{n}{k}p^{k}(1-p)^{n-k}.
$$

Reading it: $A(p,1)=p$. If $p>1/2$, $A(p,n)$ increases with $n$ and tends to 1; if $p=1/2$ it stays at $1/2$; if $p<1/2$ voting makes things **worse** and tends to 0. Voting amplifies whichever side already leads.

## Worked example

With $p=0.6$ and $n=5$: $A=\binom53 0.6^3 0.4^2+\binom54 0.6^4 0.4+0.6^5=0.3456+0.2592+0.07776=0.68256$. More samples: $n=3$ gives $0.648$, $n=11$ gives $0.7535$, $n=21$ gives $0.8256$. With $p=0.4$ and $n=5$ the vote is right only $0.3174$ of the time.

The exercise stores 8 questions with 5 canned chains each. Parsing the final answer from each text, single-sample accuracy is $25/40=0.625$ (two of the 40 texts have no parsable answer), and the majority vote is right on 7 of 8 questions, $0.875$. The independence formula at $p=0.625,n=5$ predicts $0.7248$; the observed $0.875$ is higher because there are only 8 questions (large sampling noise) and because the questions differ in difficulty. The one question the vote fails (the third) is one where the wrong answer, 9, was produced by three chains: errors that **agree** defeat voting, which is the real limit of the method. Real samples from one model share blind spots, so the gain is smaller than the independent-sample formula suggests.

## Tree-of-thought, briefly

Self-consistency samples whole chains independently. **Tree-of-thought** instead builds the solution step by step as a search: at each node the model *proposes* several next thoughts, the model (or a rule) *evaluates* each partial solution, weak branches are pruned, and the search continues from the best ones (breadth-first or best-first). It helps when early choices matter and can be judged early (puzzles, planning). The cost is roughly (branching factor) × (depth) × (proposal plus evaluation calls), so use it only where a single chain plus voting fails.

## Implementation

The exercise builds the harness without any model call:

```python
import re
from collections import Counter

def extract_answer(text):
    m = re.findall(r"answer\s*(?:is|:)?\s*(-?\d+)", text, flags=re.I)
    return int(m[-1]) if m else None

def majority_vote(answers):
    c = Counter(a for a in answers if a is not None)
    if not c:
        return None
    best = max(c.values())
    return next(a for a in answers if a is not None and c[a] == best)
```

Production libraries wrap the same idea: you call the API with `n` samples (or `n` calls), parse each, and vote. For free-form answers you cannot vote on strings, so you cluster equivalent answers first (normalise numbers, or ask a judge model whether two answers agree).

## Prompts are testable

A prompt is code whose behaviour you cannot read off the text. Keep a small set of inputs with expected outputs (or a scoring rule), run every prompt change against it, and look at the failures, not only the average. Comparing two prompts fairly needs statistics because outputs are random; [[app.04]] implements that. Remember also that results depend on the model version: a prompt tuned for one model can degrade on the next.

## Common misconceptions

- "Magic phrases make the model smarter." A role or "think step by step" changes which continuation is likely; it adds no knowledge. Gains vary by model and may be near zero on strong reasoning models.
- "More examples are always better." Context is finite and examples can bias toward their quirks. Test 0, 3 and 5.
- "Self-consistency is free accuracy." It multiplies output cost by $n$ and only helps when the single-sample accuracy is above 50 percent and errors are not systematic.
- "The model followed my instruction once, so the prompt works." One success is an anecdote; you need a test set.
- "The written reasoning is the real reason for the answer." It is generated text that conditions later tokens; it can be wrong or unfaithful even when the answer is right.

## Key takeaways

- A prompt narrows the model's output distribution; instructions, role, examples and reasoning room are the four levers.
- Order matters: stable instructions first, delimited data after, reasoning before answer.
- Majority voting over $n$ samples has accuracy $A(p,n)$, which grows for $p>1/2$ and shrinks for $p<1/2$.
- Voting fails on agreeing errors and costs $n$ times the generation.
- Tree-of-thought is search over partial solutions; reach for it last.
- Treat prompts as artifacts with tests.

## Explain it yourself

1. Why does writing intermediate steps help a model that has no hidden memory between tokens, and when does it hurt?
2. A colleague says "we sample 9 answers and vote, so accuracy must be near 100 percent." What assumptions are hidden in that sentence, and how would you check them with data?

## Further reading

- Wei et al. (2022) on chain-of-thought prompting and Wang et al. (2022) on self-consistency (cited by the lessons we read; we did not open the papers).
- Phase 11 lessons 01 and 02 of the repo library for more code.

## Mastery criteria

- You can rewrite a vague prompt using instructions, format and delimiters and explain each change.
- You can derive $A(p,n)$ and compute it for given $p,n$.
- You can state two conditions under which voting does not help.
- You can write a robust final-answer parser and a majority vote with a defined tie rule.
