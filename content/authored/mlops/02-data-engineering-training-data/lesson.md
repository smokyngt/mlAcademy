---
id: ops.02
title: "Data engineering and training data"
summary: Where production data comes from, how it is stored and processed, and how to build a training set with correct sampling, cheap labels, weak supervision and augmentation, with reservoir and stratified samplers implemented from scratch.
minutes: 70
teaches: [ops.sampling, ops.labelling]
requires: [prob.sampling, ml.imbalance, prob.basics, ops.framing]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.3 Data Engineering Fundamentals: data sources, formats (JSON/CSV/Parquet, row- versus column-major), transactional versus analytical processing, batch versus stream processing (PDF p.79-110, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.4 Training Data: sampling (non-probability, stratified, reservoir, importance), labelling (hand labels, natural labels, weak supervision, semi-supervision, transfer learning, active learning), class imbalance overview, data augmentation and synthesis (PDF p.114-154, read in selected parts). READ."
---

## Where this fits

A model is only as good as the examples it saw. In [[ops.01]] you framed the problem; now you need rows to train on. In a company those rows are scattered across logs, databases and user uploads, in several formats, arriving continuously, and most of them have **no label**. This lesson covers the plumbing (where data lives and how it is processed), then the craft of building a training set: which examples to sample, where labels come from, and how to stretch a small labelled set. Class imbalance is treated in depth in [[cml.11]]; here we only place it in the pipeline. Feature construction and leakage inside pipelines come next in [[ops.03]].

## The idea in plain words

Imagine a library that receives thousands of new books every hour (the **stream**) and keeps the old ones in the basement (the **warehouse**). You want a representative reading list of 100 books for a class, and you also want each book annotated by an expert, but experts are slow and expensive. Several questions arise:

- Which books do you pick? If you grab whatever is on the nearest shelf, your list is biased. If the library has only 1% poetry and you want to be sure poetry is in the list, you pick 1% of each shelf separately.
- If the books keep arriving and you cannot count them first, how do you keep a fair list of 100 at all times? You keep a small bag and, for each new arrival, flip a weighted coin to decide whether it replaces a random book in the bag.
- Who annotates? Perhaps readers already annotated some by their behaviour (they finished the book: a *natural label*), perhaps you write simple rules ("mentions a dragon means fantasy") and let many noisy rules vote, perhaps you ask experts only for the books your current guesses are most unsure about.
- Can you make more training examples from the ones you have? Flip, crop or paraphrase them, as long as the label does not change.

## Data sources and formats

**Sources.** *User input* (forms, uploads) is plentiful but messy: users enter text where numbers are expected, so it needs validation, and users expect instant responses. *System-generated data* (logs, model predictions, metrics) is cleaner and is essential for debugging and, later, monitoring ([[ops.06]]). There are also internal databases of the company and third-party data.

**Formats.** JSON is flexible and human-readable but costs space and makes schema changes painful. CSV and Parquet are the two canonical tabular formats and represent two storage paradigms:

- **Row-major** (CSV): the values of one row sit next to each other. Reading whole examples and appending new rows is fast.
- **Column-major** (Parquet): the values of one column sit next to each other. Reading a few columns out of thousands is fast, which is common in analytics; appending single rows is slower.

If a table has 1,000 columns and you need 4 of them, column-major reads 4 columns and row-major must scan every row. If you must log one new event at a time, row-major wins. Libraries such as pandas are built around a column-oriented DataFrame, which is why iterating over rows is slow and column operations are fast. Text formats are human-readable but larger than binary formats.

**Processing.** *Transactional* systems (OLTP) insert and update individual events with low latency and high availability; *analytical* systems (OLAP) answer queries over large amounts of history. ML data pipelines usually pass through **ETL**: extract from sources, transform (clean, validate, join), load into a warehouse or lake.

**Batch versus stream.** *Batch processing* runs periodically on historical data (a nightly job computing each driver's average rating). *Stream processing* computes on data as it flows through a message bus, with short delay, and is especially good at *stateful* updates: instead of recomputing a 30-day engagement figure every day, you update it with the new day's events. Slow-changing quantities become **batch (static) features**; fast-changing ones (rides requested in the last minute) become **streaming (dynamic) features**. Many models need both, which is why training and serving pipelines must agree on how each is computed ([[ops.03]]).

## Sampling

You almost never train on all the data. Sampling decides what the model sees, so it decides what the model can learn.

**Non-probability sampling** selects by convenience, snowballing from existing samples, expert judgement, or quotas. It is cheap and biased. Much real training data is of this kind: language models trained on text that is easy to crawl, or sentiment models trained on online reviews written by people willing to review. Know the bias you accept.

**Random (probability) sampling** gives every item a known chance of selection.

- *Simple random sampling*: each item has equal probability. Simple, but a rare class may be missed entirely.
- *Stratified sampling*: divide the population into groups (**strata**) and sample each group separately, so a rare group is guaranteed to appear. It fails when items belong to several groups at once (multilabel).
- *Weighted sampling*: items carry weights that set their probability, for example recent data weighted higher.
- *Importance sampling*: you want samples from a distribution $P$ that is hard to sample from, so you sample from an easy distribution $Q$ (with $Q(x)>0$ wherever $P(x)>0$) and weight each sample by $P(x)/Q(x)$. It is the same idea as in the Monte Carlo lessons; in ML it reappears when you correct for a training distribution that differs from the production one.
- *Reservoir sampling* for streams, derived next.

### Reservoir sampling

**Goal.** Maintain, in one pass and with memory for only $k$ items, a sample in which every one of the $n$ items seen so far is present with probability $k/n$, for every $n$ (so you can stop at any time).

**Algorithm.** Keep the first $k$ items. For the item with 1-based index $n>k$, draw an integer $j$ uniformly from $\{1,\dots,n\}$; if $j\le k$, put the new item in slot $j$, otherwise discard it.

**Why it works (induction on $n$).** Suppose after $n$ items each of them is in the reservoir with probability $k/n$. The new item $n+1$ enters with probability $k/(n+1)$ by construction. An older item currently in the reservoir is evicted only if the new item enters (probability $k/(n+1)$) **and** lands in that item's specific slot (probability $1/k$), i.e. with probability $1/(n+1)$. So it survives with probability $1-\frac{1}{n+1}=\frac{n}{n+1}$, and its overall probability is

$$
\frac{k}{n}\cdot\frac{n}{n+1}=\frac{k}{n+1}.
$$

The base case ($n=k$, probability 1) holds, so the claim holds for every $n$.

## Labelling

**Natural labels** come from the system itself: the true travel time of a trip you predicted, whether a user clicked a recommendation within ten minutes (a *behavioural label*). They arrive with a **feedback loop length**, the delay between the prediction and its label, which ranges from minutes (clicks) to months (loan default). A system can also be designed to collect feedback (upvote/downvote buttons). Tasks with natural labels are common in industry, partly because teams choose them.

**Hand labels** are the default otherwise, and they are expensive (experts for medical images), a privacy risk (someone must read the data), slow (the book quotes phonetic speech transcription taking hundreds of times the audio length) and inconsistent (annotators disagree). Slow labels make a model slow to adapt when requirements change.

**Weak supervision** replaces some hand labelling by *labelling functions* (LFs): small programs encoding heuristics, such as "if the note mentions pneumonia then URGENT", a regular expression, a lookup in a list, or the output of another model. An LF either votes for a class or **abstains**. Because heuristics are noisy, overlap and conflict, you aggregate them. A small hand-labelled sample is still needed to measure each LF's accuracy. Weak supervision suits privacy-constrained data (experts look at a small cleared sample, the LFs then run on everything) and makes expertise reusable and versionable: change the requirements, re-run the functions.

*Aggregation.* The simplest aggregator is majority vote among non-abstaining LFs. A better one weights each LF by its reliability. If LF $j$ is correct with probability $a_j$ and, given the true label, the LFs err independently (a naive-Bayes assumption), then the log posterior odds of class 1 over class 0 are the prior log odds plus a sum over voting LFs of $\pm\log\frac{a_j}{1-a_j}$ (plus for a vote for 1, minus for a vote for 0), because a vote for 1 multiplies the odds by $a_j/(1-a_j)$ and a vote for 0 divides them by it. Predicting 1 when that sum is positive (equal priors) is the **weighted vote** implemented in the exercise. Accurate LFs get a large weight, a 0.5-accuracy LF gets weight 0. Real label models (for instance Snorkel's) estimate the accuracies without gold labels and model dependencies between LFs; the independence assumption here is the simplification.

**Semi-supervision** uses structural assumptions to extend a small labelled set: *self-training* (train, label the unlabelled points the model is confident about, retrain), clustering/similarity assumptions, or perturbation-based consistency. **Transfer learning** starts from a model pretrained on an abundant task (language modelling needs no labels) and fine-tunes it with few labels. **Active learning** chooses which unlabelled examples to send to annotators, typically those the current model is least sure about.

When labelled data is scarce, mind the evaluation set: too small and model selection overfits it, too large and you waste labels. A common compromise is to select the champion on the evaluation set and then continue training on it.

## Class imbalance and augmentation

**Class imbalance** (a rare positive class, such as 0.01% cancer scans) is the norm. It hurts because the rare class has little signal, because "always predict the majority" is a hard-to-beat shortcut with 99.99% accuracy, and because errors on the rare class are usually the costly ones. Remedies are metrics that respect the rare class, resampling, and cost-sensitive losses; those techniques, with code, are in [[cml.11]]. From the data-pipeline viewpoint: stratify when you split and sample, never evaluate on a resampled test set, and remember that resampling changes the class priors your model will output.

**Data augmentation** creates new examples from old ones.

- *Label-preserving transformations*: crop, flip, rotate an image (a rotated dog is still a dog); swap a word for a near-synonym in text. This is nearly free and multiplies the data.
- *Perturbation*: add noise. Networks can be fooled by tiny changes, which also makes noise augmentation a defence and a robustness test ([[ops.04]]).
- *Data synthesis*: fill templates ("Find me a [CUISINE] restaurant within [NUMBER] miles of [LOCATION]") to bootstrap a chatbot's training queries, or blend two images and their labels to get an example with a soft label.

Augmentation applies to the **training** split only; the validation and test splits must stay untouched.

## Worked example

All numbers come from the two exercises.

*Reservoir.* Stream of 20 items, $k=5$, 4,000 independent runs with a fixed seed. Each item should appear in $5/20=0.25$ of runs. The empirical frequencies fall between 0.236 and 0.264 for all 20 items, with no trend between early and late items; a naive "keep the first five" would give 1.0 and 0.0.

*Stratified.* Labels with 900 of class 0, 90 of class 1 and 10 of class 2; sample 10%. The result has exactly 90, 9 and 1 items per class. A simple random 10% sample would have, on average, one item of class 2 but could have zero or three.

*Weak supervision.* Take the vote matrix (rows are examples, `-1` is abstain):

```text
[[ 1,  1, -1],
 [ 0,  1, -1],
 [-1, -1, -1],
 [ 0,  0,  1],
 [ 1, -1,  0]]
```

Coverage per LF is $[0.8, 0.6, 0.4]$. Rows 1, 3 and 4 contain both a 0 and a 1, so the conflict rate is $0.6$. Majority vote gives $[1,-1,-1,0,-1]$: row 1 is a 1-1 tie, row 2 has no votes, row 4 is a 1-1 tie. With gold labels $[1,1,0,0,1]$ the LF accuracies (on rows where they vote) are $[0.75, 1.0, 0.0]$. In a simulation with 4,000 examples and four LFs of true accuracy $0.9, 0.6, 0.6, 0.6$, the estimated accuracies come out near $0.896, 0.594, 0.593, 0.596$; the weighted vote is right 89.6% of the time, while plain majority vote (counting ties as wrong) is right 60.0% of the time, because three mediocre LFs outvote the excellent one.

## Implementation

```python
import numpy as np

def reservoir_sample(stream, k, rng):
    res = []
    for n, x in enumerate(stream):          # n is 0-based
        if n < k:
            res.append(x)
        else:
            j = int(rng.integers(0, n + 1))  # uniform on {0..n}
            if j < k:
                res[j] = x
    return res

def weighted_vote(L, acc):
    w = np.log(acc / (1 - acc))
    sign = np.where(L == 1, 1.0, np.where(L == 0, -1.0, 0.0))
    return (sign @ w > 0).astype(int)
```

Production tools: Spark/Flink for batch and stream jobs, Parquet for columnar storage, Snorkel for label models, `torchvision.transforms` or `albumentations` for image augmentation, scikit-learn's `StratifiedShuffleSplit` for stratified splits.

## Shapes

Vote matrix `L: (n, m)` with `n` examples and `m` labelling functions; accuracies `acc: (m,)`; weighted score `sign @ w: (n,)`. Reservoir: a list of length $k$. Stratified sample: a sorted index vector.

## Common misconceptions

- "More data is always better." Biased or mislabelled data scales the bias. Quality and representativeness come first.
- "Random sampling is always safe." Simple random sampling can miss a rare class entirely; stratify.
- "Weak labels are just bad labels." Combined and weighted, many noisy but diverse LFs can beat a single mediocre one, and they can be re-run when requirements change; but they still need a small gold set to validate.
- "Stream processing is the less efficient sibling of batch." Modern stream engines are distributed, and stateful updates avoid recomputation.
- "Augment before splitting." That puts near-duplicates of test items in training: a leak ([[ops.03]]).

## Key takeaways

- Choose storage and processing mode by access pattern: row-major for writes and whole rows, column-major for column scans; batch for slow features, stream for fast ones.
- Reservoir sampling keeps a uniform sample of a stream of unknown length using $O(k)$ memory, proved by induction.
- Stratified sampling guarantees that rare groups appear; sample and split by stratum.
- Natural labels are cheap but delayed; hand labels are slow and costly; weak supervision, semi-supervision, transfer and active learning stretch scarce labels.
- Augment only the training split, and only with label-preserving changes.

## Explain it yourself

1. Prove that in reservoir sampling the probability of an early item staying in the reservoir after $n$ items is $k/n$.
2. You have three labelling functions with accuracies 0.55, 0.6 and 0.95. Why should a plain majority vote not be trusted, and what weights would you give instead?

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch.3-4 (PDF p.79-156). Kleppmann, *Designing Data-Intensive Applications* for data systems in depth (referenced by the book, not read for this lesson).
- The label-model idea behind Snorkel (Ratner et al.), cited in the book, not read for this lesson.

## Mastery criteria

You can pick row- or column-major storage and batch or stream processing for a given workload, implement and prove reservoir sampling, build a stratified sample, aggregate noisy labelling functions with a weighted vote, and explain where in a pipeline augmentation and resampling are allowed.
