---
id: sys.04
title: "Evaluating LLMs and RAG systems"
summary: Why benchmark scores mislead, how to build and score your own eval set, retrieval metrics (recall@k, MRR, nDCG), faithfulness and LLM judges, slices, and bootstrap confidence intervals for deciding whether system B is really better than A.
minutes: 90
teaches: [llm.eval]
requires: [ml.generalization, prob.rv, llm.rag]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/10-evaluation/docs/en.md (Problem, Eval landscape, Why Benchmarks Break, Perplexity, LLM-as-Judge, ELO): saturation, contamination, judge biases. READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/10-evaluation/docs/en.md (Eval Datasets, Sample Size and Confidence, Regression Testing, Anti-Patterns): golden/adversarial/distribution sets, CI width table, paired comparison workflow. READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/07-advanced-rag/docs/en.md (Evaluation section): Recall@k, faithfulness, answer correctness. READ."
  - "reasoning-from-scratch/chF/01_main-chapter-code/chF_main.ipynb (Appendix F, markdown cells F.1-F.5): answer-choice accuracy, verifiers, Elo from pairwise preferences, LLM judges. READ."
  - "reasoning-from-scratch/ch03/01_main-chapter-code/ch03_main.ipynb (Sec. 3.1-3.9 markdown, results tables): verifier pipeline (extract, normalise, compare), tiny-sample accuracy (10 vs 500 problems). READ."
  - "reasoning-from-scratch/chF/02_mmlu/README.md and random_guessing_baseline.py, chF/04_llm-judge/README.md: three MMLU scoring methods, random baseline, judge by local model. READ (heads)."
  - "Designing Machine Learning Systems.pdf, Ch.6 Model Offline Evaluation (PDF p.201-214): baselines, perturbation/invariance/directional tests, calibration, slice-based evaluation, Simpson's paradox (Table 6-4). READ."
---

## Where this fits

You have a RAG chatbot and a colleague has a new prompt, a new embedding model and a different chunk size. Which one is better? "I tried five questions and it looked fine" is how most teams answer, and it is how regressions reach users. The previous lessons made systems faster (cache, quantisation, speculative decoding) and built a retrieval pipeline. Every one of those changes needs an *evaluation* that detects quality loss, or the speed-up is only a hope. This lesson is the measurement toolkit: what to measure, on what data, with which statistics, and how to tell a real improvement from noise.

It builds directly on the generalisation lesson: the test set protocol (never tune on it) carries over, and so does the idea that an honest number is always measured on data the system never saw.

## The idea in plain words

An exam only tells you about the student if (a) the questions were not leaked in advance, (b) there are enough questions, (c) the grading is fair and (d) the questions look like real work. Evaluating language models fails on exactly these four points:

- **Leakage (contamination).** Benchmarks live on the web, training data comes from the web. A model can score well by having seen the questions. Even without intent, a training mix shaped to benchmark formats helps (the source notes that MMLU is 4-way multiple choice, so a model can learn the answer-letter distribution).
- **Too few questions.** A 90 percent score on 50 questions is compatible with a true accuracy anywhere from about 79 to 96 percent. Differences of a few points on small sets are noise.
- **Unfair grading.** Free-form answers need a grader: a script (exact match, a math verifier), a human, or another LLM, each with its own errors.
- **Wrong questions.** A leaderboard says little about *your* task: the source says plainly that for a customer-support bot, a general knowledge benchmark is irrelevant.

The cure is not "never use benchmarks"; they are useful for rough model selection. The cure is a small, well-built evaluation set of your own plus honest statistics.

## What to measure

**Baselines first.** The ML systems book gives a story: a classifier with F1 0.90 sounded great until someone asked what a random guesser scores on a task where 90 percent of labels are positive; random guessing also gets F1 about 0.90. Compare with a random baseline, a trivial rule (always the majority class), the existing solution and a human baseline. The MMLU material in the reasoning course likewise ships a random-guessing baseline script: with four options, 25 percent is the floor. In the notebook's tiny runs, a base model scored around 21 percent on one MMLU subset, i.e. about chance.

**Task metrics for generation.** When answers can be checked (maths, code, multiple choice), use a *verifier*: extract the final answer, normalise it, compare for equivalence. Chapter 3 of the reasoning course builds exactly this pipeline (extract the boxed answer, normalise, check equivalence, grade). The notebook reports 30 percent vs 90 percent accuracy on 10 problems, and 15.6 percent vs 48.2 percent on 500; the 10-problem figures show how coarse tiny samples are (one problem moves the score ten points).

**Retrieval metrics for RAG** (the retriever is evaluated separately from the generator, with labelled queries):

- **Recall@k**: fraction of the relevant documents that appear in the top $k$.
$$
\text{recall@}k=\frac{|\text{top-}k\cap \text{Rel}|}{|\text{Rel}|}.
$$
- **MRR** (mean reciprocal rank): for each query take $1/\text{rank}$ of the first relevant hit (0 if none), average over queries. It rewards putting a right document first.
- **nDCG@k**: for graded relevance. Discounted cumulative gain with the usual discount:
$$
\text{DCG@}k=\sum_{i=1}^{k}\frac{g_i}{\log_2(i+1)},\qquad \text{nDCG@}k=\frac{\text{DCG@}k}{\text{DCG@}k\text{ of the ideal order}}.
$$
The logarithmic discount says that a good document at rank 1 is worth more than at rank 5, but not in a harsh, cliff-like way. Dividing by the ideal puts the score in $[0,1]$. (Linear gain $g_i$ is used here; some libraries use $2^{g_i}-1$.)

**Generation metrics for RAG.**

- **Faithfulness** (groundedness): is every claim in the answer supported by the retrieved passages? If the chunk says "60-day refund window" and the model says 90, that is a faithfulness failure even if retrieval was perfect. A simple checker splits the answer into claims and checks each against the retrieved text; a stronger one uses an LLM judge.
- **Answer correctness**: does the answer match the reference? It mixes retrieval and generation errors, so keep the other two to diagnose.

**Slices and robustness.** The systems book stresses slicing: report accuracy per subgroup (language, document type, query length, paying vs free users), because averages hide failures. Its Table 6-4 (a kidney-stone study) shows **Simpson's paradox**: model A beats model B in both groups (93 percent vs 87 percent, 73 percent vs 69 percent) yet B has the higher overall accuracy (83 percent vs 78 percent) because the groups are of very different sizes for the two models. Perturbation tests (add noise, change phrasing), invariance tests (changing a name must not change the answer) and calibration checks complement accuracy.

## LLM-as-judge and its pitfalls

For open-ended outputs, a strong model grades another model's answers against a rubric. It is cheap and correlates reasonably with humans (the sources quote around 80 percent agreement), but:

- **Position bias**: in pairwise comparison, the first answer is favoured. Randomise order, or score both orders and average.
- **Verbosity bias**: longer answers look better. Control for length.
- **Self-preference**: a model favours its own family's style. Use a different judge from the system under test.
- **Vague rubrics** give noisy scores; a rubric with explicit levels gives reproducible ones.
- **A weak judge** is noisy. The judge should be at least as capable as the system being judged.
- Always **validate the judge** on a few dozen human-labelled cases before trusting it, and report judge model and prompt.

**Pairwise preferences and Elo.** Chatbot-arena style evaluation shows two answers and records which is preferred. The expected score of a model with rating $R_w$ against $R_l$ is
$$
E_w=\frac{1}{1+10^{(R_l-R_w)/400}},
$$
equal ratings give $0.5$ and a 400-point gap gives about $0.91$. The Appendix F notebook notes that the real leaderboard moved to a Bradley-Terry fit with the same pairwise idea.

## Building a small eval set

Following the sources:

1. **Golden set** (50 to 100 curated cases): the core use cases and your regression tests.
2. **Adversarial set** (20 to 50): ambiguous queries, out-of-domain questions, prompt injections, queries whose answer is *not* in the corpus (the system should say it does not know).
3. **Distribution sample** (100 to 200): random real queries, because curated sets miss what users actually ask.
4. For retrieval, store for every query the ids of the relevant chunks; for answers, a reference answer or a checkable property.
5. Keep eval data separate from prompts, few-shot examples and any tuning data; never tune on the final test items.
6. Version the set and keep a changelog; a changed set is a different benchmark.

## Derivation: confidence intervals by bootstrap

You measured a per-item score vector $x_1,\dots,x_n$ (0/1 correctness, or per-query reciprocal rank). The sample mean $\bar x$ is an estimate; we want to know how much it would wobble over another sample of $n$ items from the same population. The **bootstrap** replaces "another sample from the population" by "a sample drawn with replacement from our own $n$ items":

1. Draw $B$ resamples of size $n$ with replacement.
2. Compute the mean of each, giving $\bar x^{*}_1,\dots,\bar x^{*}_B$.
3. The percentile interval is the $\alpha/2$ and $1-\alpha/2$ quantiles of these means.

Why it is sensible: the empirical distribution of the data is our best available picture of the population, and resampling from it mimics sampling from the population. For a mean, the spread of the resampled means is close to $s/\sqrt n$, the textbook standard error, so intervals shrink like $1/\sqrt n$: four times more items halve the width. For proportions near 0 or 1 or tiny $n$, the Wilson interval is a better closed-form alternative:

$$
\frac{p+\frac{z^2}{2n}\pm z\sqrt{\frac{p(1-p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}} .
$$

**Comparing two systems on the same items** must be *paired*. Let $d_i=a_i-b_i$ on item $i$. Bootstrap the differences $d_i$: items that both systems get right or wrong cancel, and only items where they disagree matter. Looking at two separate intervals for $a$ and $b$ and checking whether they overlap throws away the pairing and is too conservative. Conversely, with very few disagreements even a paired bootstrap is optimistic (it can resample "no disagreeing items in favour of $a$"); report the number of disagreements.

## Worked example

All numbers below are reproduced by the exercise code.

**Retrieval metrics.** Four queries, ranked lists of ids, relevant sets:

| query | ranking | relevant | RR | recall@3 | recall@5 |
|---|---|---|---|---|---|
| 1 | d3 d1 d7 d2 d9 | {d1} | 1/2 | 1 | 1 |
| 2 | d4 d5 d6 d1 d2 | {d1, d2} | 1/4 | 0 | 1 |
| 3 | d8 d2 d1 d3 d4 | {d8} | 1 | 1 | 1 |
| 4 | d5 d6 d7 d8 d9 | {d1} | 0 | 0 | 0 |

MRR $=(0.5+0.25+1+0)/4=0.4375$; mean recall@3 $=0.5$; mean recall@5 $=0.75$. Query 2 shows the difference between metrics: a relevant document is retrieved by $k=5$ but only in position 4, so recall@5 is perfect while its reciprocal rank is 0.25.

**nDCG.** Graded relevance $a{:}3,\ b{:}2,\ c{:}3,\ d{:}0,\ e{:}1,\ f{:}2$ and the ranking $a,b,c,d,e,f$ has $\text{DCG}=6.861$; the ideal order ($3,3,2,2,1,0$) has $7.141$; so nDCG@6 $=0.961$. Moving the useless document $d$ to the top ($d,a,c,b,e,f$) drops it to $0.750$. nDCG@3 of the first ranking is $0.978$.

**Sample size.** Wilson 95 percent intervals: 45/50 correct gives $[0.786, 0.957]$ (a spread of 17 points); 180/200 gives $[0.851, 0.934]$; 900/1000 gives $[0.880, 0.917]$. The same 90 percent accuracy, three very different pieces of evidence. The evaluation material's rule of thumb is at least 200 cases for deployment decisions and 500 or more when comparing close systems, consistent with these numbers.

**Paired bootstrap.** Simulated per-item correctness on 100 items for systems A and B (items share a latent difficulty, so errors correlate): A is right on 74, B on 67. Bootstrap 95 percent intervals: A $[0.65,0.82]$, B $[0.58,0.76]$, which overlap heavily. A and B disagree on 17 items (A-only right 12, B-only 5). The paired bootstrap of the differences gives mean $+0.07$, interval $[-0.01, +0.15]$; the unpaired version (resampling the two systems independently) gives the much wider $[-0.05, +0.20]$. About 4.8 percent of the paired resamples have a mean difference $\le 0$. Honest conclusion: A is probably better, the evidence is suggestive but not conclusive at 95 percent, and 100 items is too few to settle a 7-point difference. (An exact sign test on the 12 vs 5 disagreements gives a two-sided p of roughly 0.14, in line with that reading.)

## Implementation

```python
def ndcg_at_k(ranked, rel, k):
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    return dcg(gains) / dcg(ideal) if ideal and dcg(ideal) > 0 else 0.0

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
```

## Shapes

| Object | Shape |
|---|---|
| ranked list for one query | list of $k$ ids |
| per-query metric vector | $(n_\text{queries},)$ |
| bootstrap index matrix | $(B, n)$ |
| bootstrap means | $(B,)$ |
| paired differences | $(n,)$ |

## Common misconceptions

1. "Non-overlapping intervals are needed to call a difference real, and overlapping means no difference." Neither is right; use a paired interval for the *difference*.
2. "High recall@k means the RAG answer will be right." The generator can still ignore or distort the passages (faithfulness), so measure both.
3. "A high benchmark score means the model is good for my task." Only a matching evaluation can say that.
4. "LLM judges are objective." They are another model with biases; calibrate against humans.
5. "More metrics is better." A pile of metrics invites cherry-picking; decide the primary metric and the decision rule before looking at the results.
6. "Accuracy rose, so ship." Check slices and the cost (latency, memory) as well; a faster, quantised model that loses a point on a critical slice may not be acceptable.

## Key takeaways

- Evaluate on data that could not have leaked, with enough items, graded fairly, resembling real use.
- Retrieval: recall@k (did we get it), MRR (how early), nDCG (graded and rank-aware). Generation: faithfulness and correctness.
- Report intervals, not points: bootstrap or Wilson; width shrinks like $1/\sqrt n$.
- Compare systems with a paired analysis on the same items; report disagreements.
- Slice your results; Simpson's paradox is real.
- Judges have biases (position, verbosity, self-preference): randomise, control length, validate on human labels.

## Explain it yourself

1. Your RAG system's recall@5 is 0.95 but users complain of wrong answers. List three places the error can be and one metric for each.
2. Explain to a colleague why "A scored 74 percent and B scored 67 percent on 100 questions" is not enough to claim A is better, and what you would compute.

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch. 6 (Offline evaluation) and Ch. 9 (Test in production).
- Efron and Tibshirani, *An Introduction to the Bootstrap*.
- Järvelin and Kekäläinen, "Cumulated gain-based evaluation of IR techniques" (2002), the origin of nDCG.
- The MMLU and MATH-500 sources named in the reasoning course.

## Mastery criteria

- [ ] I can compute recall@k, MRR and nDCG by hand and in code.
- [ ] I can explain contamination, saturation and why custom evals are needed.
- [ ] I can list judge biases and mitigations.
- [ ] I can build a bootstrap CI and a paired comparison, and interpret an interval that contains zero.
- [ ] I can explain Simpson's paradox with Table 6-4 and why slices matter.
