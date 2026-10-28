---
id: rsn.02
title: "Inference-time scaling: sampling, voting and best-of-n"
summary: Spend more compute at answer time, by sampling many chains of thought and voting, picking the best-scored one, or capping the thinking budget, and learn to predict how much accuracy each extra sample buys with a binomial model.
minutes: 80
teaches: [rsn.vote]
requires: [rsn.verifier, llm.sampling, app.self-consistency, prob.rv]
sources:
  - "reasoning-from-scratch/ch04/01_main-chapter-code/ch04_main.ipynb (Sec. 4.1-4.6): chain-of-thought prompting, temperature scaling, top-p filter, self-consistency vote, MATH-500 results table with accuracy and run time. READ."
  - "reasoning-from-scratch/ch05/01_main-chapter-code/ch05_main.ipynb (Sec. 5.3 heuristic score used for best-of-N and as tie-breaker): READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (Sec. 3.4, 3.9): extraction used to compare answers; max_new_tokens in evaluation. READ."
  - "reasoning-from-scratch/chE/chE_main.ipynb (Sec. E.5 batched self-consistency sampling, E.8 table): READ."
---

## Where this fits

[[rsn.01]] gave you a reliable way to tell whether an answer is right and showed that a longer chain of thought is more compute. This lesson asks the next question: **given a fixed model, how do we trade extra inference compute for accuracy?** Nothing is trained here. The model's weights stay frozen; we change only how we use it. The ideas (sample several times and vote, or let a scorer pick) are the cheapest tools in the reasoning toolbox and the baseline that training methods ([[rsn.04]], [[rsn.05]]) must beat. The next lesson ([[rsn.03]]) looks at a sequential alternative, refinement.

It builds on decoding and sampling ([[llm.02]]) and on the idea of a vote that you may know from ensembles.

## The idea in plain words

Ask one person a hard riddle and they may slip. Ask nine independent people and take the answer most of them give: the slips are mostly different from each other, so they cancel, while the correct answer, reached by many different routes, repeats. That is **self-consistency**: sample the same model several times and keep the most common final answer.

Two refinements. If the people can also say *how sure* they are, or you have a referee who can judge an answer, you can take the single answer the referee likes best instead of counting heads: **best-of-n**. And you can control *how long* each person may think: give a hard cap (so nobody rambles) or nudge them to keep going when they stop too early. That is the **thinking budget**.

All three cost more compute. How much accuracy do you get per unit cost, and when does it stop paying? A small probability model answers that.

## Making answers different: temperature and top-p

Greedy decoding gives the same answer every time, so voting over greedy runs is pointless. We need *diverse* chains of thought that are still *sensible*. Two knobs from [[llm.02]]:

- **Temperature** $T>0$ divides the logits before the softmax: $p_i=\exp(z_i/T)/\sum_j\exp(z_j/T)$. $T<1$ sharpens the distribution, $T>1$ flattens it.
- **Top-p (nucleus)** keeps the smallest set of most likely tokens whose probability mass reaches $p$, zeroes the rest and renormalises. The book's variant keeps a token while the cumulative mass *before* it is below $p$ (and always keeps the first), which includes the token that crosses the threshold.

On the ten-token toy logits `[-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]` at $T=1$ the largest probability is $0.4538$; at $T=0.5$ it is $0.6947$ and at $T=5$ only $0.1567$. Sorted, the cumulative masses are $0.4538,\ 0.7290,\ 0.8119,\dots$; with top-p $=0.8$ the first two tokens have mass-before $0$ and $0.4538$, the third has mass-before $0.7290<0.8$, so three tokens survive, renormalised to $0.5589,\ 0.3390,\ 0.1021$ (token indices 4, 9, 8).

Sampling costs accuracy for a single answer: in the book's table the base model gets 15.2% greedy and 17.8% with temperature and top-p, while with chain-of-thought prompting greedy gets 40.6% but sampled gets 33.4%. Sampling pays only when you can combine many samples.

## Self-consistency: plurality voting

Algorithm (book Sec. 4.6): sample $n$ responses at, for example, $T=0.8$ and top-p $=0.9$; extract each final answer ([[rsn.01]]); count; return the most frequent. The book notes this is strictly a **plurality** vote (a majority needs more than half). If the top count is shared, the book returns no answer; you may break ties with a scorer, for example the average log-probability or the heuristic score of [[rsn.03]].

Two practical tips from the notebook: if all long answers look nearly identical, raise the temperature a little; if they look off, lower it.

## Best-of-n with a scorer

Instead of counting, **score** each of the $n$ responses and keep the best. The scorer can be a rule (the book's heuristic: bonuses for having a box, a brevity bonus), a model's own confidence ([[rsn.03]]), or an LLM acting as judge. With a *perfect* verifier, best-of-$n$ equals pass@$n$ ([[rsn.01]]) because you simply keep a correct sample if one exists. The catch is that at answer time you usually have **no** reference answer, so the scorer is only a proxy, and a proxy can be fooled: optimising hard against an imperfect scorer picks answers that *look* good to it.

## A probability model of voting

Suppose each sample is independent and correct with probability $p$ (for this problem), and every wrong sample gives the same single wrong answer. With $n$ odd the correct answer wins iff more than half the samples are correct:

$$
A(n,p)=\sum_{c>n/2}\binom{n}{c}p^{c}(1-p)^{n-c}.
$$

(For even $n$, a tie $c=n/2$ is worth $\tfrac12$ if broken at random, $0$ if you return nothing.)

Consequences you can verify with the exercise:

1. **Threshold at one half.** For $p>\tfrac12$, $A(n,p)\to1$ as $n$ grows (law of large numbers: the fraction correct converges to $p$). For $p<\tfrac12$, $A(n,p)\to0$: voting makes a *bad* problem worse. At exactly $p=\tfrac12$ and odd $n$, $A=\tfrac12$.
2. **Slow gains near the threshold.** For $p=0.6$: $A(1)=0.6$, $A(3)=0.648$, $A(5)=0.6826$. Two samples with random ties give back exactly $0.6$; the first useful step is three.
3. **An exponential bound.** Hoeffding's inequality gives $1-A(n,p)\le\exp\bigl(-2n(p-\tfrac12)^2\bigr)$. It is loose (for $n=21,p=0.55$ it says $\le0.90$ while the true failure rate is $0.32$) but it shows the exponential rate in $n$ and that the gap $p-\tfrac12$ controls the speed.

### When wrong answers disagree

Real wrong answers are rarely identical; a hard problem often yields many different wrong numbers. Model it with $m$ equally likely wrong answers. The correct answer now only needs the *largest* count, not more than half. Then plurality voting can help even when $p<\tfrac12$: with $p=0.3$ and $n=9$ the single-wrong-answer model gives $A=0.0988$, but with $m=4$ equally likely wrong answers the correct answer wins with probability $0.4339$.

Truth lies in between: models tend to repeat *the same* characteristic mistake, so wrong samples are correlated and cluster. A large gain from voting is evidence that wrong answers are scattered; a small gain suggests systematic errors that no amount of sampling will cancel.

### Problems differ in difficulty

Accuracy is an average over problems with different $p$. Take six problems with $p=0.95,0.8,0.6,0.4,0.2,0.05$ (average 0.5). Under the binary model the mean voting accuracy stays exactly $0.5$ for every odd $n$: the easy problems gain what the hard ones lose. Under the spread model with $m=3$ it rises: $0.5\to0.5497\ (n=3)\to0.5791\ (n=5)\to0.6113\ (n=9)$. If you return nothing on ties (the book's rule) the $n=9$ value is $0.5827$.

## The compute-accuracy trade-off in the book's data

MATH-500, 500 problems, run times on one DGX Spark GPU, base model with chain-of-thought prompt ("CoT") and top-p sampling:

| setting | accuracy | run time |
|---|---|---|
| CoT + top-p, one sample | 33.4% | 129.2 min |
| plus self-consistency $n=3$ | 42.2% | 211.6 min |
| $n=5$ | 48.0% | 452.9 min |
| $n=10$ | 52.0% | 862.6 min |
| (CoT, greedy, for reference) | 40.6% | 84.5 min |

The marginal accuracy per extra minute shrinks: $0.107$ points/min from 1 to 3 samples, $0.024$ from 3 to 5, $0.0098$ from 5 to 10. Three samples already beat the greedy CoT baseline; ten samples cost about 6.7 times the single-sample time for $+18.6$ points. With the reasoning model, $n=3$ reaches $55.2\%$ against its greedy $48.2\%$.

Two honest caveats. First, the same notebook shows that without CoT the voting results are not monotone ($29.6\%$ at $n=3$, $27.8\%$ at $n=5$, $31.6\%$ at $n=10$): with 500 problems the standard error is about 2 points ([[rsn.01]]), so those differences are mostly noise. Second, time is not exactly proportional to $n$ (it depends on lengths and on the hardware), so treat "cost $\propto n\times$ mean length" as a model, not a law.

## Thinking budgets

A **budget** limits how many tokens the model may spend. The cap side is already in the evaluation code: `max_new_tokens`. If the chain of thought is cut before the final box, no final answer is produced and, with a strict extractor, the attempt counts as wrong. In the exercise, six canned attempts with lengths $120,350,900,1500,2600,4000$ (and correctness $1,1,0,1,1,1$) give accuracy $2/6$ at a 1000-token budget, $3/6$ at 2048, and $5/6$ at 5000.

*Budget forcing* in the literature goes the other way too: when the model tries to stop early, append a word such as "Wait" so it keeps reasoning, and when it exceeds the budget, cut it off and force the final answer. The chapters I read do not implement that extension, so treat this paragraph as general background rather than something verified here.

## Worked example

Five samples answer a problem whose correct answer is 83: `83, 83, 38, 83, 5`. Counts: 83 has 3, the others 1. Plurality winner: `83`. With two samples `83, 38` the top count is shared: no winner; if their mean scores are $-0.2$ for 38 and $-1.5$ for 83, tie-breaking by mean score returns 38 (which is why a bad scorer can turn a tie into an error). Best-of-3 on `[a, a, b]` with scores $[-9,-8,-0.1]$ returns `b` even though `a` has more votes.

For the expected-accuracy curve use `plurality_acc` and `mixture_curve` in the exercise; the numbers quoted above (0.648, 0.4339, 0.6113, 0.5827) come from them.

## Implementation

```python
from collections import Counter

def plurality_vote(answers):
    counts = Counter(answers)
    top = max(counts.values())
    winners = [a for a, k in counts.items() if k == top]
    return (winners[0] if len(winners) == 1 else None), counts

def majority_acc_binary(n, p):          # odd n
    from math import comb
    return sum(comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n // 2 + 1, n + 1))
```

Production stacks sample the $n$ completions of one prompt in parallel (same prompt length, so no padding is needed; see [[rsn.05]] for batching) and share the KV cache of the prompt ([[sys.01]]).

## Common misconceptions

- "More samples always help." Not if the model is more often wrong than right on a problem and its wrong answers agree; then voting amplifies the error.
- "Voting is majority vote." It is plurality; the winner may have far less than half the votes.
- "Best-of-n with any scorer is at least as good as one sample." Only if the scorer correlates with correctness; otherwise you are paying more for a random pick, or a systematically wrong one.
- "Higher temperature is always more diverse and therefore better." It also lowers the quality of each individual chain; the right setting balances the two.

## Key takeaways

- Inference-time scaling trades compute for accuracy without changing the weights: sample, then aggregate.
- Plurality voting over extracted answers is the baseline; ties need a rule.
- Under the binomial model voting helps iff the correct answer is the modal one: $p>\tfrac12$ for a single wrong answer, much weaker if wrong answers are scattered.
- Gains show diminishing returns per unit cost; measure with error bars.
- A budget cap is part of the pipeline: truncated reasoning earns no answer.

## Explain it yourself

1. A problem has per-sample success rate $0.35$ and the wrong answers are almost always the same number. What happens to accuracy as $n$ grows, and why?
2. Why does the book's greedy CoT (40.6%) beat the single sampled CoT (33.4%), and what must voting do before it is worth its cost?

## Further reading

- Book chapter 4 (all of it) and appendix E.5; the original self-consistency idea is referenced in the notebook (Wang et al., 2022, not opened here).
- [[rsn.03]] for scoring and refinement; [[rsn.04]] for training the model so that a single sample is already better.

## Mastery criteria

- I can implement plurality voting with an explicit tie rule and a score-based tie-breaker.
- I can derive and compute $A(n,p)$ and explain the $p=\tfrac12$ threshold.
- I can say how correlated wrong answers change the benefit of voting.
- I can compute marginal accuracy per unit of compute from a results table and decide when to stop adding samples.
