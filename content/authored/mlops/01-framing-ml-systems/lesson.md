---
id: ops.01
title: "Framing ML problems and designing ML systems"
summary: Turn a vague business wish into an ML problem with inputs, outputs and an objective, and reason about the requirements (latency percentiles, scale, upkeep) that a production system must meet beyond model accuracy.
minutes: 55
teaches: [ops.framing, ops.requirements]
requires: [ml.generalization, ml.eval-metrics, prob.rv]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.1 'Understanding Machine Learning Systems' - research versus production, stakeholders, latency versus throughput, percentiles (PDF p.32-41). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.2 Business and ML Objectives, Requirements for ML Systems, Framing ML Problems (task types, multiple framings, decoupling objectives), Mind Versus Data (PDF p.51-75, read in selected parts). READ."
---

## Where this fits

Every earlier course taught you how to fit a model once someone hands you a clean table `X` and a target `y`. In a real company nobody hands you that. Somebody says "our customer support is too slow" or "can we use AI to increase sales?", and your first job is to decide **what the model should predict, from what, and how success will be judged**. Everything later in this course (data, features, evaluation, deployment, monitoring) hangs off that first decision. This lesson also introduces the vocabulary of requirements (latency, throughput, scalability) that [[ops.05]] and [[ops.06]] use constantly.

## The idea in plain words

Think of a restaurant. A cook (the data scientist) can obsess over making one dish 1% tastier. But the restaurant also needs the food to arrive hot (latency), to survive a Saturday rush (scalability), to be cookable by the night shift (maintainability), and to follow the customers' changing tastes (adaptability). A tastier dish that arrives cold and takes three chefs to reproduce is a worse restaurant. Likewise the owner does not care about "taste score 94.2 instead of 94.0": she cares about revenue, and the cook must show how taste moves revenue.

Framing a problem is the translation step: "customers wait too long" is not yet an ML problem. It becomes one when you discover that the real delay is routing each request to one of four departments, so you can say: *input = the request text, output = one of four departments, objective = make the predicted department match the correct one*. An ML problem is exactly those three things: **inputs, outputs, and the objective function**.

## Business objectives versus ML objectives

ML metrics (accuracy, F1, perplexity, inference latency) are what you can measure on a model. Business metrics (revenue, retention, cost saved, hours watched) are what the organisation cares about. A model whose accuracy goes from 94% to 94.2% matters only if some business metric moves. Two practical consequences:

- Pick, together with stakeholders, the business metric the project is supposed to influence, and find a way to connect it to an ML metric. Some tasks make this easy: every extra ad click is revenue, every blocked fraudulent payment is money saved. Others do not, and the effect can only be measured by an experiment such as an A/B test ([[ops.06]]); in that case you choose the model that moves the business metric, even if another has better ML metrics.
- Be realistic about returns. Gains from ML usually arrive after a mature pipeline exists (the book cites a survey in which companies with more years of ML in production deploy new models much faster), not overnight.

**Research versus production.** The book contrasts five differences. In research there is usually one objective (a benchmark score); in production there are many stakeholders with conflicting wishes. In research training speed and throughput dominate; in production inference speed and latency dominate. Research data is static and clean; production data shifts and is messy. Fairness and interpretability are optional in leaderboards and mandatory in many products. A concrete example: an ensemble that wins a competition may be too slow and too hard to maintain to ship.

## Requirements of an ML system

Four properties appear in most designs.

1. **Reliability**: keep doing the right thing despite hardware faults, software bugs and human error. ML adds a twist: the system can **fail silently**. `model.predict()` returns without an error while the predictions are wrong, and a user translating into a language they do not read may never notice.
2. **Scalability**: growth in model complexity (a 1 GB model becomes 16 GB), in traffic (10 thousand to 10 million requests a day), or in the **number of models** (one per customer: 8,000 enterprise customers means 8,000 models to monitor and retrain). Scaling includes scaling down to save money, not only up, and managing artifacts (code, data, model versions), not just machines.
3. **Maintainability**: other people, including future you, must be able to understand, debug and extend the system. Document, version, make results reproducible.
4. **Adaptability**: data and requirements change, so the system must be updatable quickly (see [[ops.06]] for distribution shift and continual learning).

### Latency is a distribution, not a number

**Latency** is the time from receiving a request to returning the answer; **throughput** is the number of requests processed per unit time. If a server handles one request at a time, throughput is `1/latency`: 10 ms per query is 100 queries per second. But real systems **batch** requests: processing `B` queries together in `t` ms gives

$$
\text{throughput}=\frac{B\cdot 1000}{t}\ \text{queries/s}.
$$

The book's example: 10 queries per 10 ms batch is 1,000 qps; 50 queries per 20 ms batch is 2,500 qps. Latency doubled **and** throughput rose 2.5x. So "higher latency means lower throughput" is only true without batching, and batching has a cost: queries must wait for the batch to fill ([[ops.05]] simulates this).

Never summarise latency by its mean. Take ten requests that took 100, 102, 100, 100, 99, 104, 110, 90, 3000 and 95 ms. The mean is 390 ms, which describes none of them; the median (**p50**) is 100 ms; the 99th percentile (**p99**) is 3000 ms and exposes the one terrible request. We use the **nearest-rank** definition: sort the `n` values and take the one at 1-based rank `ceil(q/100 * n)`. (The book states that p90 of this list is 3000 ms; with the nearest-rank definition p90 is 110 ms, and interpolating definitions give about 399 ms. Percentile definitions differ between libraries, which is itself a reason to write down which one your SLA uses. The lesson - look at the tail, not the mean - is unchanged.) Requirements are therefore stated on percentiles, for example "median under 100 ms and p99 under 1 s". High percentiles matter because slow requests are often the heaviest users.

### The iterative process

An ML project is a loop, not a line: scope the project and its metrics, collect and prepare data, develop the model, deploy, monitor and keep learning, and analyse the business impact, which feeds the next round of scoping. Each loop teaches you something that changes the previous steps (bad predictions send you back to the data). Plan for the loop and for rapid iteration; this is why the rest of the course spends as much time on pipelines and monitoring as on models.

## Framing a problem

The **task type** is dictated by the output.

- **Classification** outputs a category; **regression** outputs a continuous number. They are interchangeable to a degree: bucket house prices into price ranges and regression becomes classification; threshold a spam score at 0.5 and a regression-like score becomes a class.
- **Binary** classification is easiest. **Multiclass** problems get harder as classes grow (many classes with few examples each). **Multilabel** problems (an article can have several topics) need a rule for turning probabilities such as `[0.45, 0.2, 0.02, 0.33]` into a set: take the top 1? top 2? a threshold? And annotators disagree more on how many labels apply.
- **The same problem can be framed several ways**, and the framing changes difficulty. "Which app will the user open next?" as a multiclass problem has an output layer that must change whenever a new app is installed; framed as a regression "score each (user, app) pair" the model handles any set of apps.

### Several objectives: decouple them

Suppose a news feed should show engaging posts but not extreme or low-quality ones. You have two losses, `quality_loss` and `engagement_loss`, and want a trade-off. One option is a single model trained on a blend:

$$
\text{loss}=\alpha\,\text{quality\_loss}+\beta\,\text{engagement\_loss}.
$$

Changing `alpha` and `beta` means retraining. The alternative is to **decouple**: train one model per objective and blend the *scores* at ranking time,

$$
\text{score}(x)=\alpha\,\text{quality}(x)+\beta\,\text{engagement}(x).
$$

Now the trade-off is a dial you can turn without retraining, and each model can be updated on its own schedule (spam tactics change fast, quality tastes slowly). When objectives truly conflict, no single item is best at both; the useful candidates are the **Pareto front**: items not **dominated** by another item that is at least as good on every score and strictly better on one. Weights `alpha, beta >= 0` then choose a point on that front.

## Mind versus data

Is progress driven by clever model design (inductive bias, structure) or by more data and compute? Well-known researchers take both sides, and the book sets them side by side without a verdict. For a practitioner the safe reading is practical: you need data of sufficient quality and quantity before anything else (the book calls this the foundation of the "data science hierarchy of needs"), but more data is not automatically better, and structure helps you learn more from less. The next lessons start from the data.

## Worked example

The exercise code reproduces these numbers. For the ten latencies above: mean 390 ms, p50 100 ms, nearest-rank p90 110 ms, p99 3000 ms. For batching: `throughput_qps(1,10)=100`, `throughput_qps(10,10)=1000`, `throughput_qps(50,20)=2500`.

For decoupling, take five posts with `quality = [0.9, 0.5, 0.7, 0.2, 0.9]` and `engagement = [0.1, 0.5, 0.3, 0.2, 0.4]`. Post 0 is dominated by post 4 (same quality, more engagement); post 2 is dominated by post 4 as well; post 3 is dominated by several. The Pareto front is `{1, 4}`. Ranking by quality alone gives `[0, 4, 2, 1, 3]`; by engagement alone `[1, 4, 2, 3, 0]`; with `alpha=2, beta=1` the scores are `1.9, 1.5, 1.7, 0.6, 2.2` and the order is `[4, 0, 2, 1, 3]`. Note how changing the dial never touched a model.

## Implementation

```python
import numpy as np

def latency_stats(lat):
    s = np.sort(np.asarray(lat, dtype=float))
    pct = lambda q: float(s[max(int(np.ceil(q / 100 * len(s))), 1) - 1])
    return float(s.mean()), pct(50), pct(90), pct(99)

def blended_ranking(quality, engagement, alpha, beta):
    score = alpha * np.asarray(quality) + beta * np.asarray(engagement)
    return [int(i) for i in np.argsort(-score, kind="stable")]
```

In production, latency percentiles come from a monitoring system (histograms in Prometheus or a cloud dashboard), and multi-objective ranking from a feature/score service; the logic is the same as above.

## Common misconceptions

- "A higher offline metric means a better product." Only if the metric is tied to a business outcome; otherwise you are optimising a proxy.
- "Latency up means throughput down." Not with batching.
- "Average latency is enough." The mean hides outliers and the users who suffer most.
- "A problem has one correct framing." Classification versus regression, one model versus decoupled models: framing is a design choice with consequences for data, labels and maintenance.
- "Once deployed, we are done." Deployment starts the loop again.

## Key takeaways

- An ML problem is defined by inputs, outputs and an objective function; business problems must be translated into that form.
- Connect ML metrics to business metrics; use experiments when the link is unclear.
- Reliability, scalability, maintainability and adaptability are requirements as important as accuracy; ML systems can fail silently.
- Latency is a distribution: use percentiles; batching can raise both latency and throughput.
- With several objectives, prefer decoupled models blended at score time.

## Explain it yourself

1. A colleague says "our model has 94.2% accuracy, so the project is a success". What questions do you ask before agreeing?
2. Explain why batching can increase latency and throughput at the same time, and when that trade is a bad idea.

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch.1-2 (PDF p.18-78). For latency terminology, Kleppmann, *Designing Data-Intensive Applications* (referenced by the book, not read for this lesson).

## Mastery criteria

You can frame a vague request as inputs/outputs/objective, name the four requirements and give an ML-specific risk for each, compute percentile latencies and batch throughput, and explain why decoupled objectives are easier to maintain than a blended loss.
