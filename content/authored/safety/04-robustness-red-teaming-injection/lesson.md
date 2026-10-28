---
id: saf.04
title: "Robustness: red-teaming and prompt injection"
summary: A defender's view of adversarial testing - the classes of attack on language-model systems, why indirect prompt injection is a system-design problem, how red-team tooling fits together, and how to report attack success rate with honest uncertainty.
minutes: 75
teaches: [saf.red-teaming, saf.prompt-injection]
requires: [saf.goodhart, prob.basics, stat.testing]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/docs/en.md (whole lesson): automated attack families at a conceptual level, JailbreakBench/HarmBench, ASR and budgets. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/13-many-shot-jailbreaking/docs/en.md (whole lesson): power-law scaling with number of in-context examples, shared mechanism with in-context learning, classifier-based defence. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks/docs/en.md (whole lesson): encoding-level attacks and why filters miss them. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/15-indirect-prompt-injection/docs/en.md (whole lesson): delivery vectors, information-flow control, adaptive-attack evaluation of defences. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit/docs/en.md (whole lesson): classifier, scanner and campaign tools and evaluation pitfalls. READ."
---

## Where this fits

[[saf.01]]-[[saf.03]] were about what training does to a model. This lesson is about what an *outside party* can do to a deployed model or application, and how a defender should test for it. It is written from the defender's side: we describe attack **classes** so that you can recognise them and evaluate defences, but we do not give prompts, templates or step-by-step methods, and the exercises use abstract data (counts, labels) rather than harmful content. It builds on probability and confidence intervals (`prob.basics`, `stat.testing`) and leads into capability evaluation and governance in [[saf.06]].

## The idea in plain words

A locked door is tested by people who try to open it; the result you want is not "nobody got in" but "here is how many tries, by whom, with what tools, and how confident we are". Testing a language-model system is the same, with two complications:

1. **The attacker can learn.** Unlike a physical lock, an attacker who sees a refusal can rephrase and try again, and automated attackers do this thousands of times. A defence that stops *yesterday's* attacks says little about tomorrow's.
2. **The system mixes instructions and data.** A conventional program keeps code and data apart. A language model reads everything as text in one window; text fetched from a web page can look just like an instruction from the user. This is the core of **prompt injection**, and it makes the *architecture* of the application, not only the model, part of the defence.

Hence two questions run through the lesson: *how do we measure robustness honestly?* and *how do we design systems that remain safe even when some input is hostile?*

## A taxonomy of attack classes (conceptual)

The source lessons group published attacks as follows. Descriptions are deliberately high level.

| Class | Idea | Needs | Notes from the sources |
|---|---|---|---|
| Iterative black-box refinement | An attacker model proposes prompts, sees the target's replies and a judge's score, and revises | Query access only | Reported to succeed within a few tens of queries on some models; the loop is a standard red-team baseline (PAIR) |
| Token-level search | Gradient-guided search over adversarial strings | Model weights | Produces unreadable strings; transfers to other models in some studies (GCG) |
| Evolutionary / tree search / persuasion templates | Search over prompts, branching attempts, or social-science persuasion patterns | Query access | AutoDAN, TAP and PAP are the named examples |
| Long-context example stuffing ("many-shot") | Fill a long context with examples of the unwanted behaviour, then ask | Long context window | ASR rises as a power law in the number of examples; shares a mechanism with benign in-context learning |
| Encoding / obfuscation | Hide the safety-relevant words in a form the filter does not recognise but the model can read (e.g. text drawn as characters) | A capable reader | Works because filters are token- or meaning-level while the model reads the shape; recognition ability and vulnerability correlate |
| Indirect injection | Instructions placed in external content the system will read (a page, an email, a tool result) | Ability to publish content the agent will see | The dominant production threat per the source; bypasses user-input filters |

Three observations the sources stress:

- **Capability and vulnerability are coupled.** The better a model reads unusual encodings or learns from long context, the more those features can be abused.
- **A benchmark number is meaningless without a budget and a judge.** "90% ASR at 200 queries" is not comparable to "85% at 20".
- **Many-shot defence has a dilemma.** The mechanism is ordinary in-context learning, so suppressing pattern-following from long contexts would break benign few-shot prompting. The defence reported in the source inspects the context with a classifier and modifies or truncates the offending part; it reports attack success falling from 61% to 2% in the settings tested (developer-reported).

## Prompt injection as a systems problem

**Direct** injection: the user types the hostile text. **Indirect** injection (IPI): the hostile text sits in third-party content that an application retrieves. The source lists three delivery routes: retrieval (a poisoned document is fetched into the prompt), inbox/document workflows (a message the agent reads), and tool outputs (a tool result carries the text).

Why input filters miss it: the filter watches the user's message, but the payload arrives through the retrieval or tool channel. Filtering *all* content that reaches the model would flag legitimate pages that merely use imperative language.

**Information-flow control (IFC).** The source reports this as the dominant 2026 defence idea, borrowed from operating-system security: label every piece of content by trust, propagate labels through the computation, and require that an action triggered by untrusted-labelled content is ratified by a trusted party before it has side effects. The principle in the source is *containment, not prevention*: as long as instructions and data share a context window, assume some injection will succeed and limit what it can do. Systems named in the lesson include CaMeL (2025), ConfAIde (2024) and a 2026 defence paper; we did not read those papers.

**The three-way risk.** A pattern from practitioner writing (not in the repo lessons) helps reasoning about agent design: an agent is in a particularly risky position when it has all of (1) access to private data, (2) exposure to untrusted content, and (3) the ability to communicate externally. With all three, an injected instruction can read private data and send it out. Removing *any one* leg breaks the chain, which turns a vague worry into a design checklist. The exercise implements both this check and a tiny label-propagation checker.

**Real-world context.** The source lists several publicly assigned vulnerability identifiers for injection in commercial assistants and states that prompt injection is ranked first in the OWASP list of LLM application risks (2025 edition). We did not check those references.

## Red-team tooling and the defence stack

Three tools illustrate different layers (per the source):

- **Safety classifiers** (for example the Llama Guard family): a model fine-tuned to label inputs and/or outputs against a taxonomy of 14 hazard categories. Place it before the model, after it, or both. Cost: latency and false positives on benign content that must be measured per deployment.
- **Scanners** (for example Garak): a *probe* generates test inputs, a *detector* scores outputs, a *harness* runs them and reports. Static probes age as models are patched; adaptive probes age more slowly. Tiered scoring replaces a single pass/fail.
- **Campaign orchestrators** (for example PyRIT): multi-turn campaigns with converters (transform a seed), orchestrators (the strategy) and scorers.

A typical stack in the source: classifier on both sides of the model, a scanner on every nightly build for regression, campaigns before major releases. Not covered: organisational process, disclosure, and legal constraints on testing.

## Measuring robustness: ASR with uncertainty

**Attack success rate (ASR)** is the fraction of test behaviours for which an attack produced a harmful or policy-violating output according to a **judge** (a model or classifier). Its value depends on: the behaviour set, the attack, the query budget, and the judge. Report all four.

Because ASR is a proportion from $n$ trials, it carries sampling error. The simple normal-approximation interval is poor for small $n$ or for rates near 0 or 1. Use the **Wilson score interval**. With $\hat p=k/n$ and $z=1.96$ for 95%:

$$
\text{centre}=\frac{\hat p+\frac{z^2}{2n}}{1+\frac{z^2}{n}},\qquad
\text{half-width}=\frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}} .
$$

It never leaves $[0,1]$ and, importantly, **zero successes does not mean zero risk**: with 0 of 50 the 95% interval still extends to about 7.1%.

Further tools:

- **Budgeted ASR**: ASR at $k$ queries is the fraction of behaviours whose first success came within $k$ queries; it is non-decreasing in $k$.
- **Comparing two defences**: a pooled two-proportion z statistic, $z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}$, with $\hat p$ the pooled rate.
- **Sample size**: for half-width $h$ around rate $p$, $n\approx z^2p(1-p)/h^2$ (385 for $p=0.5$, $h=0.05$).
- **Adaptive versus static evaluation**: a static benchmark reports one attack; a realistic attacker uses *any* attack that works, so the right metric is the union over attacks. The source reports a study in which every one of twelve published injection defences, which had shown near-zero ASR against fixed attacks, was broken above 90% by adaptive attacks (search, reinforcement learning, a long human red-team session). Methodological lesson: a defence evaluated only against attacks that were not designed against it has not been shown robust.
- **Power-law scaling**: for curves such as error against number of in-context examples, a straight line on log-log axes indicates $y\propto n^{-\alpha}$; the slope estimates $\alpha$.

## Worked example

All from the exercises.

- 30 successes in 100 trials: $\hat p=0.30$, Wilson 95% interval $[0.219,\ 0.396]$. With 3 of 10 the interval is $[0.108,\ 0.603]$ - similar point estimate, far less information. With 300 of 1000: $[0.272,\ 0.329]$.
- Comparing defences at 30/100 versus 15/100 gives $z=2.54$ (roughly a 1% two-sided $p$-value), so the difference is unlikely to be chance alone - assuming the same behaviours, judge and budget.
- A simulated population of 200 behaviours where about half are breakable gives ASR $0.04,\ 0.14,\ 0.25,\ 0.44$ at budgets $1,5,10,20$. The budget must be reported.
- Five attack variants with individual success rates $(0.01,0.10,0.00,0.07,0.06)$ over 100 behaviours: the best single attack reports 10%, but the union is 24%. The static number understates the adaptive one by more than a factor of two.
- Many-shot-style scaling: an error curve $0.9\,n^{-0.5}$ for $n=1,\dots,256$ gives slope $-0.5$ on log-log axes; the fit recovers $\alpha=0.5$ exactly.
- Information flow: reading an untrusted page and sending a summary externally is *confirm*; combining it with a private inbox and sending externally is *block*; sending private data that no untrusted content influenced is *allow* (the user's own request).

## Implementation

```python
import math
import numpy as np

def wilson_interval(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def union_asr(M):           # M: (behaviours, attacks) boolean matrix
    return float(np.asarray(M).any(axis=1).mean())
```

Production practice: use a maintained harness (the scanners and benchmarks above), fix the judge, publish the behaviour set and budget, and store raw outcomes so intervals can be recomputed.

## Common misconceptions

- **"The filter stops injection."** A filter on user input does not see content arriving through retrieval or tools.
- **"Zero observed failures means robust."** It means the upper confidence bound is about $3/n$ (the 'rule of three'; Wilson gives 7.1% at $n=50$).
- **"A defence that beats published attacks is robust."** Only adaptive evaluation, where the attacker knows the defence, counts as evidence.
- **"Better capabilities make the model safer against attacks."** Reading unusual encodings or long contexts is exactly what some attacks exploit.
- **"Red-teaming is a one-off."** Probes age, models change and attackers adapt; it is a regression suite plus periodic campaigns.

## Key takeaways

- Attack classes (iterative refinement, token search, long-context, encoding, indirect injection) differ in what access they need; the right defence differs too.
- Indirect injection is an architectural problem: contain what untrusted content can trigger (labels, confirmation, least privilege), because filtering cannot be complete.
- Report ASR with its budget, judge, behaviour set and a Wilson interval; use the union over attacks for adaptive evaluation.
- Absence of observed success is bounded evidence; sample size determines how small a rate you can rule out.
- Layer defences (classifiers, scanners, campaigns, architectural limits); none is sufficient alone.

## Explain it yourself

1. Why does a filter on the user's message fail against indirect injection, and what does information-flow control do differently?
2. A paper reports 0% ASR on 40 behaviours against its own defence. What would you ask before believing the defence is robust?

## Further reading

- Papers named in the source lessons: PAIR (arXiv:2310.08419), GCG (arXiv:2307.15043), JailbreakBench (arXiv:2404.01318), ArtPrompt (arXiv:2402.11753), the many-shot jailbreaking paper (NeurIPS 2024) and the adaptive-attack study of injection defences (October 2025). The OWASP LLM Top 10 is the practitioner reference for application-layer risks. Only the repo summaries were read.

## Mastery criteria

- Name six attack classes and the access each needs, without giving operational detail.
- Explain why indirect injection bypasses user-input filters and what containment means.
- Compute a Wilson interval, a budgeted ASR and a union ASR, and say why each matters.
- Apply the three-leg risk check to an agent design and propose which leg to remove.
