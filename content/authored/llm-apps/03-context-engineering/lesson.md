---
id: app.03
title: "Context engineering: budgets, ordering, compaction and prefix caching"
summary: Decide what goes into the context window, in what order and at what price, and simulate how a prefix cache turns a stable prompt layout into measurable savings.
minutes: 75
teaches: [app.context, app.prompt-cache]
requires: [app.prompting, trf.attention, llm.lm]
sources:
  - "ai-engineering-from-scratch/phases/11-llm-engineering/05-context-engineering/docs/en.md (opening sections: window as scarce resource, token-budget example, lost-in-the-middle, context components; first ~6.5k characters only). READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/11-caching-cost/docs/en.md (opening sections: cost anatomy, provider caching table, semantic and exact caches, token bucket; first ~6.5k characters only). READ."
  - "ai-engineering-from-scratch/phases/11-llm-engineering/15-prompt-caching/docs/en.md (opening sections: the prefix invariant, cache-friendly layout, write premium and break-even; first ~6.5k characters only). READ."
---

## Where this fits

[[app.01]] was about the words of a prompt. In a real application the model's input is much more than a prompt: system rules, tool definitions, retrieved documents, conversation history, examples and the user's message all compete for one finite window, and every token costs money and attention. **Context engineering** is deciding what goes in, what stays out, in what order, and how to avoid paying twice for what has not changed. It uses what you know about attention (`trf.attention`: cost grows with sequence length and attention is not uniform), connects to retrieval (lesson `sys.03` explains embeddings and RAG internals; here we only decide *when* to retrieve), and to the KV cache that makes provider-side prefix caching possible (lesson `sys.01`). It ends with an offline simulator that prices a layout.

## The idea in plain words

Picture the context window as the desk in front of an expert. The desk is big but not infinite, the expert glances more carefully at what is at the top and at the bottom than at the pile in the middle, and every sheet you add must be read (and you are billed per sheet). A good assistant does not dump the whole filing cabinet on the desk. It keeps a short card of standing rules at the top, brings only the documents relevant to this question, replaces last month's long correspondence by a one-paragraph summary, and puts the actual question at the bottom, where it cannot be missed.

The second idea is about repetition. If every request starts with the same 4,000-token rule book, a smart provider remembers having read it and charges you a small fee to reuse its notes. But the memory works only if the beginning is *exactly* the same: change one word near the top and the whole rest must be re-read.

## The budget

Everything counts against the window of size $W$:

$$
W \;\ge\; T_{\text{system}}+T_{\text{tools}}+T_{\text{retrieved}}+T_{\text{history}}+T_{\text{examples}}+T_{\text{query}}+T_{\text{output}}
$$

The output headroom $T_{\text{output}}$ counts: a window that is full at the end of the input leaves no room to answer. A typical coding-assistant breakdown from the source we read: system 500, tool definitions 8,000 (50 tools at about 150 tokens each is 7,500), retrieved documentation 4,000, history 6,000, query 200, output budget 4,000. The total is $22{,}700$ tokens, which is $22700/128000=17.7\%$ of a 128K window. The window was not the problem; **signal density** is. Irrelevant tool definitions and stale turns make the model slightly worse and the bill larger.

A simple allocation policy: fix budgets for the parts you cannot shrink (system, query, output), then give the rest to the parts in priority order (retrieved evidence, recent history, examples), trimming the lowest-priority material first. If the tool list is large, include only tools relevant to the request (a dynamic tool selector can remove most of them).

## Ordering and recency

Models do not use all positions equally. The source cites a study (Liu et al., 2023; we did not read the paper) in which a relevant passage placed among about 20 irrelevant ones was used best when at the start or end of the context and worse in the middle, with accuracy dropping by tens of points in the worst case for some models. The size varies by model and generation; the shape of the lesson is robust enough to act on:

- put standing instructions first (they are also the stable prefix, see below);
- put the current question and the most relevant evidence last;
- treat the middle as the weakest zone; if a fact must live there, repeat it near the end;
- fewer, better-ranked passages beat many mediocre ones.

Test this on your own model and data (a needle-in-a-haystack style probe is easy to write); do not assume the published numbers transfer.

## Compaction: keeping history affordable

Long conversations and agent runs outgrow any window. Options:

| Strategy | Idea | Weakness |
|---|---|---|
| Truncate | drop the oldest turns | loses early facts silently |
| Sliding window | keep the last $k$ turns | same, but predictable |
| Summarise | replace old turns by a model-written summary | summaries drop details and can hallucinate |
| Structured memory | extract facts (preferences, decisions, open tasks) into a small store | needs a schema and an update rule |
| Retrieve | store everything outside, bring back what is relevant | retrieval can miss; adds latency |

A robust default is hybrid: keep the last few turns verbatim, a running summary of the earlier ones, and a store of durable facts; trigger compaction when the history exceeds a budget (not every turn, because rewriting the history changes the prefix and breaks caching, see below).

## Retrieval or long context?

Long windows tempt you to paste everything. Retrieval (lesson `sys.03`) selects a few passages. Trade-offs: long context avoids retrieval misses and index maintenance but multiplies cost per request and pushes evidence into the weak middle; retrieval is cheaper and keeps the prompt focused but can miss the needed passage and adds a component to maintain. Practical rule: use long context for a bounded, known document set you ask many questions about (combine with caching so you pay for it once), and retrieval for large or changing corpora. Measure both on your questions ([[app.04]]).

## Prefix caching

Providers keep the model's internal state for a recently seen prompt prefix (this is the KV cache of lesson `sys.01`, retained between requests). If your next request starts with the same tokens, they do not need re-processing, and you are billed a reduced **read** price. The first time, you may pay a **write** premium. The source's comparison (prices change often; check the provider's current page) describes: an explicit-marker scheme with a large read discount and a 25 percent write premium; an automatic scheme with a smaller discount and no premium; and a cache object you create and store with a fee. All share one invariant:

> Only a **prefix** is cached. The first token that differs ends the match; everything after it is a miss.

So the cache-friendly layout is: system prompt, tool definitions, few-shot examples, retrieved documents (if reused), conversation history up to the last turn, and last the new user message. Dynamic content early, reordered tool lists, timestamps or request ids inside the system prompt all destroy hits. Cache entries also expire (minutes by default on some providers).

### The break-even calculation

Let a block cost 1 unit uncached, $w$ on a write and $r$ on a read. One write followed by $n$ reads costs on average, relative to never caching,

$$
\frac{w+r\,n}{n+1}.
$$

With $w=1.25$ and $r=0.1$: $n=1$ gives $1.35/2=0.675$, $n=10$ gives $2.25/11=0.2045$. Caching pays back as soon as $w+rn<n+1$, i.e. $n>(w-1)/(1-r)=0.278$, so one read suffices; with a $2\times$ write premium and half-price reads you would need $n>2$, three reads.

## Worked example: a 10-turn chat

The exercise simulates a chat where every request carries a 2,000-token system prompt and 1,500 tokens of tool definitions (both cacheable), the earlier turns of 500 tokens each (also cacheable) and a fresh 200-token user message (not cached). At an illustrative 3 dollars per million input tokens, with write multiplier 1.25, read multiplier 0.1 and a 300-second TTL:

- Total input over 10 turns: $10\cdot3700+500\cdot(0+1+\dots+9)=37{,}000+22{,}500=59{,}500$ tokens.
- Turn $i\ge1$ reads the previous turn's whole prefix: $3500+500(i-1)$ tokens, total $49{,}500$. Hit rate $49500/59500=83.2\%$.
- Cost with cache: 0.05085 dollars. Without: 0.1785 dollars. Savings $71.5\%$.

Failure modes the same simulator exposes:

- A **single** request costs $23.6\%$ more than no caching (all 3,500 cacheable tokens pay the write premium and nothing is ever read).
- Turns spaced 400 s apart (TTL 300 s): no hits, cost $24.2\%$ **above** baseline, because you keep paying write premiums.
- **Dynamic text first**: put the user message at the front and the prefix never matches; cost is $25\%$ above baseline.

Caching is an optimisation you must verify: check the cache-read token counts in the provider's usage fields, and alert when they drop to zero.

## Implementation

The simulator's core is a dictionary from prefix (tuple of segment keys) to last-used time:

```python
def lookup(cache, keys, t, ttl, ends):
    for k in sorted(ends, reverse=True):          # longest cacheable prefix first
        last = cache.get(tuple(keys[:k]))
        if last is not None and t - last <= ttl:
            return k
    return 0
```

Application-level caches sit on top: an **exact cache** (hash of the full prompt, useful at temperature 0) and a **semantic cache** (embedding similarity above a threshold returns a stored answer). A semantic cache can return a wrong answer for a question that is similar but different, so set the threshold with a test set and exclude personalised or time-sensitive queries.

## Common misconceptions

- "A bigger window means I can stop curating." Quality and cost both degrade with noise.
- "Summarising is lossless." It is lossy; keep a pointer to the source.
- "Caching always saves money." A cache that is never read costs a premium; verify hit rates.
- "I can put the date at the top of the system prompt." A changing first token invalidates every later cached token.
- "Position in the window does not matter." Evidence in the middle is used less reliably by many models.

## Key takeaways

- Budget the window as a sum, including output headroom; measure token counts, do not guess.
- Put stable material first, the question and key evidence last, repeat critical facts near the end.
- Compact history with a hybrid of recent turns, summary and durable facts, and do it in batches.
- Prefix caching saves $1-\frac{w+rn}{n+1}$ per reused block but only for exact prefixes within the TTL.
- Choose retrieval or long context by corpus size, change rate and measured accuracy.

## Explain it yourself

1. Why does rewriting a conversation summary on every turn defeat prefix caching, and how would you restructure it?
2. Your hit rate is 0 although the system prompt is identical across requests. List three places to look.

## Further reading

- Lesson `sys.01` on the KV cache; lesson `sys.03` on retrieval-augmented generation.
- Provider documentation on prompt caching (field names and prices change).

## Mastery criteria

- You can compute a token budget and say what to cut first.
- You can order a prompt for recency and for cacheability.
- You can compute the break-even number of reuses for given write and read multipliers.
- You can explain three ways a cache silently stops working.
