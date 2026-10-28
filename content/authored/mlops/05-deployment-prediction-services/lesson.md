---
id: ops.05
title: "Deployment and prediction services"
summary: Decide between batch and online prediction and between cloud and edge, shrink models with quantisation, distillation and pruning, debunk deployment myths, and simulate how queueing and dynamic batching shape latency and throughput.
minutes: 70
teaches: [ops.serving, ops.queueing, ops.compression]
requires: [ops.requirements, ops.skew, prob.rv, num.floatingpoint]
sources:
  - "math/Designing Machine Learning Systems.pdf (Huyen), Ch.7 Model Deployment and Prediction Service: deployment myths 1-4, batch versus online prediction, unifying batch and streaming pipelines (PDF p.221-236, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.7: Model Compression (low-rank factorisation, knowledge distillation, pruning, quantisation, Roblox case study) and ML on the Cloud and on the Edge (compilation, browsers) (PDF p.237-257, read in selected parts). READ."
  - "math/Designing Machine Learning Systems.pdf, Ch.1 latency versus throughput and batching (PDF p.38-40), re-read for the simulation. READ."
---

## Where this fits

A model that lives in a notebook helps nobody. **Deployment** makes predictions available to the people or systems that need them, and it is where the requirements of [[ops.01]] (latency percentiles, scale, cost) become hard constraints. This lesson answers three design questions. *When* do we compute predictions: before they are asked for (batch) or when they are asked for (online)? *Where* does the model run: in the cloud or on the user's device? *How big* a model can we afford, and how do we shrink one that is too big? You will also build a small simulator that shows why a server at 80% utilisation can have a terrible 99th-percentile latency, and how batching requests rescues an overloaded model. Consistency between training and serving pipelines ([[ops.03]]) matters most here, and what happens after deployment is the subject of [[ops.06]].

## The idea in plain words

A bakery can bake bread overnight and sell it from the shelf (**batch**: cheap per loaf, instant to hand over, but you must guess in advance what customers will want and the bread may be stale) or bake to order (**online**: always fresh and tailored, but the customer waits and you need ovens ready for every rush). A bakery with one oven and customers arriving randomly will have a queue even if the oven is idle 20% of the time: arrivals come in clumps. If customers arrive faster than the oven can serve, the queue grows without bound. Baking several loaves at once makes each loaf slightly slower but lets the oven serve many more people per hour, so a queue that was exploding can disappear. That is dynamic batching.

Moving the oven into every customer's kitchen (**edge**) removes the delivery delay and keeps their ingredients private, but their ovens are weak and you cannot see how they are doing. Shrinking the recipe (**compression**) makes it fit in the small ovens.

## Deployment myths

The book lists several beliefs of people who have only deployed in academia; four of them are worth carrying:

1. **"You deploy one or two models."** Real companies have many: a ride-hailing app needs models for demand, ETA, pricing, fraud and churn, perhaps per country, so 10 models in 20 countries is already 200. Infrastructure must handle many models.
2. **"If we do nothing, performance stays the same."** Software ages ("bit rot") and data drifts ([[ops.06]]), so a model is best right after training and decays.
3. **"We will not need to update models often."** The right question is not "how often should I update" but "how often *can* I update".
4. **"Most engineers need not worry about scale."** Most engineers work at companies with at least 100 people, whose systems likely serve many users.

## Batch versus online prediction

- **Batch prediction** precomputes predictions on a schedule and stores them (for example all customers' purchase propensities), then looks them up. Throughput is high, latency at request time is just a lookup, and the model can be arbitrarily slow. The cost: results are only as fresh as the last batch (a user who just switched from horror to comedy still sees horror), and you must know in advance which inputs will be queried, which is impossible for open-ended inputs such as arbitrary text to translate.
- **Online prediction** computes a prediction when the request arrives. It can use the freshest information and handle any input, but latency is on the critical path, so the model, the feature extraction and the network all count.

Many production systems blend them: *batch features* (slowly-changing, such as a driver's long-term rating) joined with *streaming features* (rides requested in the last minute). The classic bug follows from this: training computes a feature in batch on a dataframe while serving computes the same feature on a sliding window in a stream. **Two separate pipelines drift apart**, especially when two teams own them. The remedy is the shared definition and skew checks of [[ops.03]], or a unified stream/batch infrastructure.

A rule of thumb from the book's discussion: batch is partly a legacy of existing batch systems; as streaming infrastructure matures, more systems move toward online prediction because freshness matters.

## Latency and throughput as a queueing problem

Requests arrive at random times and a server processes them one by one (or in batches). Two numbers describe a one-server queue: the arrival rate $\lambda$ (requests per second) and the service rate $\mu=1/\text{(service time)}$. The **utilisation** is $\rho=\lambda/\mu$.

**Lindley's recursion** gives an exact, simple simulation of a first-come-first-served single server. Let request $i$ arrive at time $a_i$ and need service time $s_i$. It starts when it has arrived *and* the server has finished the previous request, finishes $s_i$ later, and its latency is the time from arrival to finish:

$$
\text{start}_i=\max(a_i,\ \text{finish}_{i-1}),\qquad
\text{finish}_i=\text{start}_i+s_i,\qquad
\text{latency}_i=\text{finish}_i-a_i .
$$

If arrivals are Poisson (random, memoryless) and service times are exponential, the system is an **M/M/1 queue**, and theory gives a mean latency (waiting plus service) of

$$
W=\frac{1}{\mu-\lambda},\qquad \text{stable only if }\lambda<\mu .
$$

Look at what this says: with $\mu=10$ requests per second, $\lambda=8$ gives $W=0.5$ s, $\lambda=9$ gives $1.0$ s, $\lambda=9.5$ gives $2.0$ s, and as $\lambda\to\mu$ the latency explodes. The service itself only takes 0.1 s on average; almost all of the latency is waiting. Capacity planning at 100% utilisation is therefore a mistake, and **tail latencies are far worse than means**.

### Dynamic batching

A GPU or vectorised CPU often takes nearly the same time to process 16 inputs as 1. Model the batch time as $a+b\cdot\text{size}$ with a fixed overhead $a$ and a per-item cost $b\ll a$. Then the capacity at batch size $B$ is $B/(a+bB)$ requests per second, much higher than $1/(a+b)$. The server must decide when to start a batch. A common rule: start when the batch is full (`max_batch` requests) or when the oldest request has waited `max_wait`, whichever comes first (and never before the server is free). Larger batches raise throughput but add waiting; `max_wait` bounds that waiting.

## Model compression

If the model is too large or slow for the latency target or the device, shrink it.

- **Quantisation** stores weights (and activations) with fewer bits. A 100 million parameter model at 32 bits takes $100\text{M}\times4$ bytes $=400$ MB; at 16 bits, 200 MB; at 8-bit integers, 100 MB. Smaller numbers also compute faster and allow bigger batches. A common scheme maps a weight $w$ to an 8-bit integer $q=\mathrm{round}(w/s)$ with scale $s=\max|w|/127$; the rounding error per weight is at most $s/2$. The costs: a smaller range of representable values (outliers must be clipped or scaled) and rounding errors that can sometimes change accuracy more than expected. It can be done during training (quantisation-aware training) or after training (post-training quantisation, offered by on-device frameworks). In the book's Roblox case study, moving from 32-bit floats to 8-bit integers gave the largest speed-up of all their changes (roughly 7x lower latency and 8x higher throughput), but the book also notes that the source does not report the effect on output quality.
- **Knowledge distillation** trains a small *student* to imitate a large *teacher* (or ensemble). DistilBERT keeps about 97% of BERT's language-understanding score with 40% fewer parameters and 60% faster inference, according to the book. It works across architectures but needs a teacher.
- **Pruning** removes parameters that matter least. Either remove whole units, or (more commonly) set many weights to zero, which gives a sparse network that stores in less space; reports exceed 90% fewer non-zero parameters without loss, though the value of pruning has been debated.
- **Low-rank factorisation** replaces large tensors with products of smaller ones (for example, MobileNets split a standard $K \times K$ convolution into a depthwise and a pointwise one; the book reports an eight to nine times parameter reduction for $3 \times 3$ kernels). This family is tied to particular architectures.

In practice, quantisation is the first thing to try because it is general and cheap; the others are more specialised.

## Cloud versus edge

**Cloud** deployment is the easy starting point (managed services) but costs grow with traffic (large companies spend huge sums), every request needs a network round trip, and user data must travel to the server. **Edge** deployment runs the model on phones, browsers, cars, cameras or embedded chips. Advantages: it works offline, avoids network latency (often larger than inference latency), reduces cloud bills, and keeps sensitive data on the device. Challenges: weak hardware (so compression), many hardware backends with different compute primitives and memory layouts (so compilers that translate models to each chip are needed), and harder updates and monitoring. One way to be portable is to run in the browser, compiling models to WebAssembly instead of slow JavaScript. Many systems are hybrid: a small model on the device with a larger one in the cloud.

## Worked example

All numbers come from the simulations in the exercise (fixed seeds).

*M/M/1.* Poisson arrivals at $\lambda=8$ per second, exponential service with $\mu=10$, 50,000 requests. Theory: mean latency 0.5 s. Simulation: 0.501 s. The 99th percentile is 2.22 s, over four times the mean, and the server is idle 20% of the time.

*Batching.* Arrivals at about 61 requests per second (6,000 requests), batch time $0.02+0.002\cdot\text{size}$ seconds. One-at-a-time service takes 0.022 s, so capacity is 45.5 requests per second: the queue is unstable, mean latency 17.1 s and p99 33.6 s, growing with the number of requests. With `max_batch=16, max_wait=0.02`, the server forms batches of 2.27 requests on average, the mean latency is 0.0405 s and the p99 is 0.0535 s. Other settings: `max_batch=4, max_wait=0.01` gives mean 0.0366 s, p99 0.056 s; `max_batch=32, max_wait=0.05` gives mean 0.0606 s, p99 0.086 s (more waiting for a fuller batch). Batching can be the difference between an unusable and a healthy service, and the best setting is a trade-off found by simulation or load testing.

## Implementation

```python
import numpy as np

def fifo_latencies(arrivals, service):
    lat = np.empty(len(arrivals)); free = 0.0
    for i in range(len(arrivals)):
        start = max(arrivals[i], free)
        free = start + service[i]
        lat[i] = free - arrivals[i]
    return lat

def mm1_mean_latency(lam, mu):
    return 1.0 / (mu - lam)      # requires lam < mu
```

The batching simulator in the exercise starts a batch at `max(server_free, min(arrival_of_oldest + max_wait, arrival_of_last_slot))` and includes every request among the next `max_batch` that has arrived by then.

Production tools: serving frameworks with dynamic batching (TorchServe, NVIDIA Triton, TensorFlow Serving), `torch.quantization`, TensorFlow Lite, ONNX Runtime, TensorRT, and compilers such as TVM or XLA for hardware-specific code.

## Shapes

`arrivals: (n,)` sorted times; `service: (n,)`; latencies `(n,)`; batch sizes `(n_batches,)`. Quantisation maps a float tensor of any shape to an integer tensor of the same shape plus one scalar scale (per tensor) or one per channel.

## Common misconceptions

- "Utilisation 80% means we are comfortable." Queueing delay dominates and the tail is far above the mean.
- "Batching only adds latency." It adds waiting but increases capacity; under load it lowers latency.
- "Online is always better than batch." Batch is simpler and cheaper when freshness is not essential.
- "Quantisation is free." It can change accuracy; always re-run offline evaluation, including slices ([[ops.04]]).
- "Edge means no cloud." Hybrid designs are common.

## Key takeaways

- Choose batch for cheap, predictable inputs and relaxed freshness; online for fresh, open-ended inputs; watch for separate training and serving pipelines.
- Latency under load is dominated by queueing; mean latency in M/M/1 is $1/(\mu-\lambda)$ and explodes as $\lambda\to\mu$.
- Dynamic batching trades a bounded wait for much higher capacity.
- Quantisation is the most general compression technique; distillation, pruning and low-rank factorisation complement it.
- Edge removes network latency and keeps data local at the price of weaker hardware and harder operations.

## Explain it yourself

1. A service with mean service time 0.1 s receives 9.5 requests per second. Estimate the mean latency and explain why p99 is worse.
2. Why can batching reduce latency even though each request now waits for others?

## Further reading

- Huyen, *Designing Machine Learning Systems*, Ch.7 (PDF p.217-263); Kleppmann, *Designing Data-Intensive Applications* on response time and queueing (referenced by the book, not read for this lesson).
- [[ops.06]] for what to watch after deployment.

## Mastery criteria

You can choose between batch and online prediction and between cloud and edge for a given product, compute M/M/1 latency, explain tail behaviour, simulate dynamic batching and choose its parameters, and apply and critique quantisation, distillation and pruning.
