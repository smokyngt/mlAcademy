---
id: saf.06
title: "Capability evaluations, governance and provenance"
summary: How dual-use capability is measured without producing harm, how frontier safety frameworks turn measurements into commitments, how a green-list watermark and a z-test work, and a dated overview of the regulatory landscape.
minutes: 75
teaches: [saf.capability-evals, saf.watermarking, saf.governance]
requires: [saf.red-teaming, prob.rv, stat.testing]
sources:
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation/docs/en.md (whole lesson): benchmark composition, yellow-zone criterion, unlearning method, novice-relative vs expert-absolute uplift, measurement pitfalls. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf/docs/en.md (whole lesson): three lab frameworks as summarised, safety-case pillars, race-dynamic clauses. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/docs/en.md (whole lesson): green-list text watermark, image watermark and removal result, C2PA, limitations. READ."
  - "ai-engineering-from-scratch/phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea/docs/en.md (whole lesson): EU AI Act timeline, GPAI Code of Practice, UK, US and Korean developments. READ."
---

## Where this fits

Everything so far asked how a model behaves. This lesson asks how an *organisation* decides whether a model is safe enough to release, how the public can tell where content came from, and what rules apply. It is mostly measurement and institutions. The measurement parts reuse the statistics of [[saf.04]] (intervals, honest uncertainty); the control and oversight ideas of [[saf.03]] reappear as evidence inside "safety cases". The watermark section is a hands-on detector you can build and attack.

**Date and provenance of the regulatory facts.** The governance and regulation sections summarise the repo's lesson notes 18 and 24 (and 23 for the labelling code). Those notes state that they reflect the situation in roughly the first half of 2026 (one passage says a code of practice "remains in draft as of April 2026"). Today is later than that. Dates, thresholds and document versions below may have changed; treat every item as "reported in the course notes, to be verified against the official source" and do not use this lesson for legal advice. We did not open any primary regulatory text.

## The idea in plain words

**Capability evaluation.** Before a new aircraft is certified, engineers do not test whether it can crash; they test components, margins and failure modes. For AI, the worry is *dual-use* knowledge: information that has legitimate uses but could also help someone cause mass harm. We cannot test by asking the model to help cause harm. Instead we measure a *proxy*: questions whose answers are stepping stones, written so that the benchmark itself is not a hazardous publication; and we measure *uplift*: how much better a person does with the model than without it.

**Safety frameworks.** A lab commits in advance: "if our measurements show the model crosses capability level X, we will not deploy it without safeguards Y". It is like a building code with tiers, plus a document (a **safety case**) arguing why a particular release is acceptable.

**Provenance.** Two complementary ideas: hide a statistical signal inside generated content (a **watermark**), and attach tamper-evident signed metadata saying who made it and how (**content credentials**). Neither proves a negative: lack of a signal does not mean content is human-made.

## Dual-use capability evaluation

As summarised in the source:

- **WMDP** is a multiple-choice benchmark of 4,157 questions: biosecurity 1,520, cybersecurity 2,225, chemistry 412. Questions sit in a "yellow zone": knowledge that is proximate to harmful processes but not a recipe, reviewed by multiple experts and screened for export-control compliance. The model answers questions; it is never asked to produce anything harmful, so capability can be measured without eliciting harm.
- **Unlearning.** The same benchmark serves to evaluate methods that remove hazardous knowledge. The reported baseline, RMU, pushed a 7B model's score on the benchmark towards random while keeping general benchmark scores within a few points. The measurable claims are therefore two numbers: *target score near chance* and *small general loss*.
- **Novice-relative versus expert-absolute capability.** Uplift for a novice can be a large multiple even if the absolute information is modest; an expert's ceiling concerns what can be extracted at maximum effort. A safety argument must address both.
- **The reported narrative** (as the source frames it) moved from "mild uplift over a web search" in 2024 to a lab saying models were "on the cusp" of meaningfully helping novices (April 2025), and a controlled novice trial reported at about 2.5 times the success of the control arm, described as insufficient to rule out the second tier of a lab's safety levels. We cannot check these claims; read them as what the notes report.
- **Pitfalls.** A benchmark score is a capability proxy, not a deployment measurement. Real-world risk also depends on how hard the capability is to *elicit* past safeguards, on **tacit knowledge** that cannot be learned from text, and on **execution barriers** such as procurement and equipment. Conversely, a low score may reflect weak elicitation rather than absence of capability, so evaluations aim for best-effort elicitation and report the result as a lower bound.

### Statistics of a threshold decision

Suppose a framework says "take extra precautions if the model's success rate on task suite $X$ reaches 10%". You observe $k$ successes in $n$ trials. The observed rate is a noisy estimate. A conservative rule treats the capability as possibly present when the **upper confidence bound** reaches the threshold:

$$
\text{act if }\ \mathrm{Wilson}_{\text{upper}}(k,n)\ge\text{threshold}.
$$

Example: 3 successes in 100 trials has upper bound $0.085$, below $0.10$; 8 in 100 has upper bound $0.150$, above it; 0 in 40 still allows $0.088$. For uplift studies the effect is a **risk ratio** $\mathrm{RR}=\frac{k_1/n_1}{k_0/n_0}$; on the log scale its standard error is $\sqrt{\frac1{k_1}-\frac1{n_1}+\frac1{k_0}-\frac1{n_0}}$, giving the interval $\exp(\log\mathrm{RR}\pm z\,\mathrm{se})$. With 30/50 versus 12/50: $\mathrm{RR}=2.5$, 95% interval $[1.45,\,4.30]$. Small trials give wide intervals, so "uplift of about 2.5" can be consistent with 1.5 or with 4.

## Frontier safety frameworks

The course notes summarise three published frameworks (versions and dates as stated in the notes):

- **Anthropic Responsible Scaling Policy** (v3.0 per the notes, February 2026): tiered AI Safety Levels, modelled on biosafety levels. The notes say the third level was activated in May 2025 for models with chemical/biological/radiological/nuclear relevance, and higher levels concern automating AI research; once a higher AI-R&D capability level is crossed an affirmative safety case is required.
- **OpenAI Preparedness Framework** (v2, April 2025 per the notes): a capability is tracked if it is *plausible, measurable, severe, net-new* and *instantaneous or irremediable*; capabilities reports are separated from safeguards reports; a safety advisory group reviews.
- **Google DeepMind Frontier Safety Framework** (v3.0, September 2025 per the notes): *critical capability levels* by domain (biological uplift, cyber uplift, machine-learning R&D acceleration, and harmful manipulation), with a deceptive-alignment section added earlier.

Common structure: published capability thresholds, evaluations before scaling or deployment, graded safeguards, and a clause allowing a lab to adjust requirements if a competitor ships without comparable safeguards. That last clause is contested: critics say it creates a race to the bottom, defenders say unilateral restraint can be worse. Terminology is not standardised across labs. These frameworks are voluntary.

**Safety cases.** A safety case is a written argument that a deployment is acceptable under stated assumptions. The notes give three pillars: *monitoring* (we would detect bad behaviour), *illegibility* (the model cannot carry out a coherent harmful plan within its action budget) and *incapability* (the model lacks the capability), with different pillars carrying different cases. A control argument ([[saf.03]]) is one type of evidence; a dual-use capability evaluation is another.

## Watermarking and provenance

### The green-list text watermark

At each generation step, hash the previous token (and a secret key) to split the vocabulary into a **green** set (fraction $\gamma$, for example one half) and the rest. Add $\delta$ to the logits of green tokens before sampling, so green tokens are over-represented but the text stays fluent.

**Detection.** Given a text of $n$ tokens, recompute for each of the $T=n-1$ consecutive pairs whether the token is green given its predecessor. Count $g$ green tokens.

*Null hypothesis:* the text was not generated with this key. Then each position is green with probability $\gamma$, independently (approximately; see below), so $g$ has mean $\gamma T$ and variance $T\gamma(1-\gamma)$, and by the central limit theorem

$$
z=\frac{g-\gamma T}{\sqrt{T\gamma(1-\gamma)}}\ \approx\ \mathcal N(0,1).
$$

Reject the null if $z$ exceeds a threshold. One-sided $p=\tfrac12\operatorname{erfc}(z/\sqrt2)$; at $z=4$, $p=3.2\times10^{-5}$.

*Why watermarked text scores high.* If the sampled token is green with probability $f>\gamma$, then $\mathbb E[z]=\frac{(f-\gamma)\sqrt T}{\sqrt{\gamma(1-\gamma)}}$, which grows like $\sqrt T$: longer text gives stronger evidence. If the model's probability mass were spread evenly, adding $\delta$ to green logits would give $f=\frac{\gamma e^\delta}{1+\gamma(e^\delta-1)}$; for $\gamma=0.5,\delta=2$ that is $0.881$. A text of $T$ tokens reaches $z=4$ when $T\approx\big(\frac{4\sqrt{\gamma(1-\gamma)}}{f-\gamma}\big)^2\approx28$ tokens in that idealised case.

*Limits (standard, and echoed in the source):*

- **Low-entropy text** (code, quotes, one-answer questions) offers little room to bias sampling, so detection is weak.
- **Edits.** Each replaced token spoils two pairs (the one into it and the one out of it). Paraphrase or translation can erase the signal, and the notes cite meaning-preserving attacks that defeat many text and image watermarks.
- **Model-specific.** Only generations from watermarked models carry the signal; "no watermark found" is not evidence of human authorship.
- **Key control.** Whoever has the key can detect; publishing it lets others forge or scrub.
- **Repeated n-grams** break the independence assumption, so z-scores can be miscalibrated on repetitive text; check the null distribution on real human text.

### Other provenance layers

- **Production text watermark**: the notes describe a Google DeepMind system whose text version adjusts next-token sampling imperceptibly, open-sourced in October 2024.
- **Image watermarks** embed a message in the generator (the notes cite Stable Signature) and report high detection after heavy cropping, but also a 2024 result that fine-tuning the decoder removes the mark while preserving quality.
- **Content credentials (C2PA)**: cryptographically signed metadata recording who created content and what edits followed. It is rich but can be stripped; a watermark carries only bits but is embedded. The notes recommend using both.

## The regulatory landscape (as reported in the course notes, early 2026)

| Jurisdiction | What the notes report |
|---|---|
| European Union | AI Act in force 1 Aug 2024. Prohibited practices and AI-literacy duties from 2 Feb 2025. Obligations for general-purpose AI models and governance from 2 Aug 2025; models above a stated training-compute level (the notes give $10^{25}$ FLOP) are treated as systemic-risk. Wider application and transparency duties (labelling of AI-generated content) from 2 Aug 2026; some embedded high-risk systems and legacy models from 2 Aug 2027. Fines up to 15 million EUR or 3% of turnover are quoted for the relevant tier. The notes mention a Commission proposal to adjust the high-risk timeline. |
| EU code of practice for GPAI | Published 10 July 2025 with chapters on transparency, copyright, and safety and security (the last for systemic-risk models), 12 commitments in total; enforcement from August 2026 per the notes. A separate code on labelling AI-generated content was in draft (first draft December 2025). |
| United Kingdom | The AI Safety Institute was renamed the AI Security Institute (February 2025) with a narrower focus on security-relevant frontier capability; it released an open-source evaluation framework (Inspect, May 2024). |
| United States | The national AI safety institute at NIST was reorganised as the Center for AI Standards and Innovation (June 2025), with an emphasis on standards and innovation support; the notes describe federal policy as innovation-favouring and decentralised with states filling gaps. |
| South Korea | An AI Framework Act passed December 2024, effective January 2026, consolidating earlier bills; the notes mention a safety institute under the science ministry, local representatives for foreign providers and risk assessment for "high-impact" AI. |

The notes' practical remark: a deployer in several jurisdictions usually has to meet the strictest, which in 2026 is typically the EU regime. The mapping from a technical control (an evaluation, a watermark, a data-governance record) to a legal obligation depends on the exact legal text, which this lesson does not provide.

## Worked example

From the exercises.

- **Watermark.** Vocabulary 500, $\gamma=0.5$, $\delta=2$, $T=200$ tokens. Unwatermarked sampling gives $z$ values near 0 (over 600 random 100-token texts: mean about 0, standard deviation about 1; false-positive rate at $z\ge2$ about 2%, at $z\ge4$ zero). Watermarked generations give $z\approx10$ to $11$ and a green fraction around 0.88, matching the formula above. With a wrong key the same text gives $z$ near 0. In one run, replacing 10%, 30%, 50% and 80% of the tokens of a text with $z=11.6$ at random gave $z=9.3,\ 6.0,\ 2.3$ and $-0.9$: the signal degrades smoothly and disappears when most tokens are replaced.
- **Thresholds.** 3/100 against a 10% threshold: upper bound 0.085, can be ruled out. 8/100: 0.150, cannot. A 300/1000 score on four-way multiple choice is $3.65$ standard deviations above guessing; 520/1000 is $19.7$.
- **Unlearning report.** Target score from 0.62 to 0.27 with general score from 0.64 to 0.61: a drop of 0.35, near chance (0.25), and a 0.03 general loss.

## Implementation

```python
import math
import numpy as np

def watermark_z(tokens, is_green, gamma=0.5):
    tokens = np.asarray(tokens)
    gr = is_green(tokens[:-1], tokens[1:])
    T = len(gr)
    return (gr.sum() - gamma * T) / math.sqrt(T * gamma * (1 - gamma))

def p_value(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))
```

Production systems (for example the open-sourced text watermark in Google's responsible-generative-AI toolkit, per the notes) use more careful sampling schemes and keyed hashing of longer contexts, and evaluate robustness to paraphrase.

## Common misconceptions

- **"A watermark proves text is human if it is missing."** It only identifies text from a cooperating, watermarked generator.
- **"A benchmark score is the deployment risk."** It is a proxy; elicitation, tacit knowledge and execution barriers change real-world risk.
- **"Zero successes means the capability is absent."** Use an upper bound; weak elicitation also yields low scores.
- **"Frameworks are laws."** The lab frameworks are voluntary commitments; regulation is separate.
- **"The regulatory dates in this lesson are current."** They are dated early 2026 and may have moved; check official sources.

## Key takeaways

- Dual-use capability is measured through proxies and uplift studies, never by eliciting harm; scores are lower bounds and must be paired with elicitation effort.
- Threshold decisions should use confidence bounds: act when the upper bound reaches the threshold.
- Safety frameworks translate measurements into pre-committed safeguards; safety cases argue via monitoring, illegibility or incapability.
- A green-list watermark is detected by a z-test whose strength grows as $\sqrt T$; it is fragile to rewriting, weak on low-entropy text and silent on unwatermarked sources.
- Regulation is moving and jurisdiction-specific; anchor claims to dated primary sources.

## Explain it yourself

1. Why does a watermark detector need a *threshold* on $z$, and what are the two ways of being wrong?
2. A lab's threshold is "10% task success". Their model scores 8/100. What do you recommend and why?

## Further reading

- Named in the source lessons: the WMDP paper (arXiv:2403.03218), the three frameworks themselves (each lab's own site), the green-list watermark paper by Kirchenbauer et al. (2023), Stable Signature (arXiv:2303.15435) and its removal result (arXiv:2405.07145), the C2PA specification, and the official EU AI Act and AI Office pages. Only the repo summaries were read.

## Mastery criteria

- Explain why dual-use evaluation uses proxy benchmarks and what makes a score a lower bound.
- Apply the upper-confidence-bound rule to a threshold decision and compute a risk-ratio interval.
- Derive the watermark z-test from the null hypothesis and explain why detection strengthens with $\sqrt T$.
- State, with dates and caveats, what you read about the regulatory landscape and what you would verify.
