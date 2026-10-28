# %% meta
{"id": "proj.rag", "title": "A retrieval-augmented QA system from scratch", "summary": "Build, evaluate and dissect a question-answering system over a synthetic document collection with no external model: sentence-window chunking, TF-IDF and BM25 retrieval, an SVD embedder, reciprocal-rank fusion, a feature reranker, an extractive answer selector, recall/MRR/nDCG with bootstrap intervals, a chunk-size ablation and a failure analysis.", "level": "advanced", "minutes": 150, "requires": ["sys.03", "sys.04", "cml.10"], "outcomes": ["Chunk documents with overlapping sentence windows and reason about the chunk-size trade-off", "Implement TF-IDF, BM25 and an SVD (LSA) embedder, and fuse rankings with reciprocal rank fusion", "Rerank candidates and extract answers with a type-aware sentence selector", "Report recall@k, MRR and nDCG with bootstrap confidence intervals and paired comparisons", "Run an ablation over chunk size and k with several corpus seeds and classify why queries fail"], "stages": ["Corpus and eval set", "Chunking and measurement", "Sparse retrieval", "Dense and hybrid", "Rerank and answer", "Ablation", "Failure analysis", "Report"]}
# %% md Briefing
A retrieval-augmented generation (RAG) system answers a question in two steps: **retrieve** the passages most likely to contain the answer, then **read** them to produce the answer. The reading step is usually a language model. Here there is no model at all: the "reader" is a small extractive selector, so every number you see is caused by code you can inspect. That is the point: before putting an LLM on top, you should know how good your retrieval is, how to measure it, and where it breaks.

You will build the whole stack over a synthetic collection of 60 technical notes, each describing one invented machine (its inventor, launch year and home city), plus a labelled evaluation set of 80 questions of four kinds (direct, paraphrased, misspelled and indirect). Because we generate the data, every question has a known gold answer sentence, so retrieval and answering can both be scored exactly.

Rules for the project:
- retrieval quality is measured with **answer recall@k, MRR and nDCG@k**, always with a bootstrap interval;
- no claim about "better" without a paired comparison;
- every failure you report must be assigned a cause by code, not by intuition.
# %% md Stage 1: Corpus and eval set
Each note has eight sentences in a fixed order. Sentence 0 introduces the machine, sentence 1 names the inventor, sentence 2 gives the launch year (with the pronoun *It*, not the name!), sentence 3 gives the city, and the rest are filler or a distractor person who *improved* the design. The structure matters: the answer to a "year" question lives in a sentence that does not contain the machine's name, so a chunk made of that single sentence is hopeless. Chunking will have to deal with this.

Question types:
- **direct**: names the machine and reuses the note's own words;
- **paraphrase**: names the machine but rewords the relation (made / come out / produced);
- **typo**: like direct but one letter of the machine's name is dropped;
- **indirect**: does not name the machine; identifies it by category and launch year ("Who invented the pump launched in 1987?").
# %% code
import re, zlib
import numpy as np

SYL = ["zor", "vex", "kal", "mir", "tan", "quo", "bel", "dru", "fen", "lix", "nor", "pae", "sul", "gro", "wen", "jat"]
FIRST = ["Mira", "Ola", "Tomas", "Ines", "Karl", "Nadia", "Piotr", "Lena", "Hugo", "Sana", "Ivo", "Rita", "Emil", "Dara", "Boris", "Yuna"]
LAST = ["Lund", "Berg", "Voss", "Haro", "Kane", "Doyle", "Mori", "Falk", "Rask", "Ibsen", "Cole", "Nagy", "Olsen", "Pike", "Soto", "Wirth"]
CITIES = ["Karsk", "Belmar", "Tovik", "Ostra", "Dunmere", "Halvik", "Pertoa", "Quillan", "Rovan", "Selbury", "Tarsus", "Ulmen"]
CATS = {
    "pump": ["Pumps of this family move liquid through long pipes.", "A pump needs a steady seal to avoid leaks.", "Engineers test every pump under heavy pressure.",
             "The pump housing is cast from a single block.", "Operators check the pump flow once per shift."],
    "bridge": ["Bridges of this family cross wide rivers on slender arches.", "A bridge must carry traffic through storms and floods.", "Inspectors measure every bridge span each spring.",
               "The bridge deck rests on tall stone piers.", "Builders raise the bridge in modular sections."],
    "lens": ["Lenses of this family bend light onto a small sensor.", "A lens needs clean glass to stay sharp.", "Technicians polish each lens by hand.",
             "The lens barrel is machined from light metal.", "Users swap the lens depending on the scene."],
    "engine": ["Engines of this family turn fuel into steady motion.", "An engine must stay cool under a long load.", "Mechanics tune each engine before delivery.",
               "The engine block is forged in one piece.", "Drivers listen to the engine for early signs of wear."],
    "turbine": ["Turbines of this family spin in a fast stream of steam.", "A turbine needs balanced blades to run quietly.", "Crews inspect each turbine blade after a season.",
                "The turbine shaft turns on oiled bearings.", "Plants run the turbine around the clock."],
    "valve": ["Valves of this family open and close a narrow channel.", "A valve must seal tightly after thousands of cycles.", "Workers grease every valve stem before shipping.",
              "The valve seat is ground to a mirror finish.", "Technicians replace the valve gasket when it hardens."],
}
USES = {"pump": "moving liquid", "bridge": "crossing rivers", "lens": "focusing light", "engine": "producing motion", "turbine": "turning steam into power", "valve": "controlling flow"}
VERBS = ["invented", "devised", "created"]

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def make_corpus(seed, n_docs=60):
    rng = np.random.default_rng(seed)
    names = []
    while len(names) < n_docs:
        nm = "".join(rng.choice(SYL, 2)).capitalize()
        if nm not in names:
            names.append(nm)
    people = []
    while len(people) < n_docs * 2:
        p = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if p not in people:
            people.append(p)
    cats = list(CATS)
    docs = []
    for i in range(n_docs):
        cat = cats[i % len(cats)]
        E, P, P2 = names[i], people[2 * i], people[2 * i + 1]
        year, city, verb = int(rng.integers(1900, 2020)), str(rng.choice(CITIES)), str(rng.choice(VERBS))
        adj = str(rng.choice(["compact", "heavy", "reliable", "early"]))
        fill = [CATS[cat][j] for j in rng.permutation(5)[:3]]
        sents = [f"The {E} {cat} is a {adj} device used for {USES[cat]}.",
                 f"The {E} {cat} was {verb} by {P}.",
                 f"It was launched in {year}.",
                 f"Its main workshop is located in the city of {city}.",
                 fill[0], fill[1],
                 f"Later, {P2} improved the design.",
                 fill[2]]
        docs.append({"id": i, "entity": E, "cat": cat, "person": P, "year": year, "city": city, "verb": verb, "sents": sents})
    return docs

def make_queries(docs, seed, per_type=20):
    rng = np.random.default_rng(seed + 1000)
    queries = []
    for typ in ["direct", "paraphrase", "typo", "indirect"]:
        for _ in range(per_type):
            d = docs[int(rng.integers(len(docs)))]
            E, cat = d["entity"], d["cat"]
            if typ == "indirect":
                rel = str(rng.choice(["who", "where"]))
                if rel == "who":
                    q, ans, sent = f"Who {d['verb']} the {cat} launched in {d['year']}?", d["person"], 1
                else:
                    q, ans, sent = f"Which city hosts the main workshop of the {cat} launched in {d['year']}?", d["city"], 3
            else:
                rel = str(rng.choice(["who", "when", "where"]))
                name = E
                if typ == "typo":
                    cut = int(rng.integers(1, len(E) - 1))
                    name = E[:cut] + E[cut + 1:]
                if rel == "who":
                    q = f"Who {d['verb']} the {name} {cat}?" if typ != "paraphrase" else f"Who made the {name} {cat}?"
                    ans, sent = d["person"], 1
                elif rel == "when":
                    q = f"In what year was the {name} {cat} launched?" if typ != "paraphrase" else f"When did the {name} {cat} come out?"
                    ans, sent = str(d["year"]), 2
                else:
                    q = f"In which city is the main workshop of the {name} {cat} located?" if typ != "paraphrase" else f"In which city is the {name} {cat} produced?"
                    ans, sent = d["city"], 3
            queries.append({"q": q, "type": typ, "rel": rel, "doc": d["id"], "sent": sent, "answer": ans})
    return queries

DOCS = make_corpus(seed=0)
QUERIES = make_queries(DOCS, seed=0)
print(f"{len(DOCS)} documents, {sum(len(d['sents']) for d in DOCS)} sentences, {len(QUERIES)} labelled questions")
print("\nExample document:")
for i, s in enumerate(DOCS[3]["sents"]):
    print(f"  s{i}: {s}")
print("\nExample questions (type | question | gold answer):")
for k in (0, 20, 40, 60, 25, 65):
    q = QUERIES[k]
    print(f"  {q['type']:10s} | {q['q']:75s} | {q['answer']}")
# %% md Stage 1 summary
You have a corpus with known structure, a labelled question set with gold sentence indices, and a feel for how the four question types differ. Everything in this project is reproducible from `seed=0` and, later, two more corpus seeds.
# %% report
Before building anything: which question type do you expect to be hardest for a purely lexical (word-matching) retriever, and which for an embedding-based one? Write your guess; you will compare it with the data in Stage 7.
# %% md Stage 2: Chunking and measurement
**Chunking.** We cannot embed or score a whole corpus as one piece, and a long document dilutes the match with its rare words. A chunk is a window of `size` consecutive sentences; the window slides by `stride` sentences, so neighbouring chunks overlap and an answer is less likely to be cut in half. The last window must reach the final sentence.

Windows are half-open ranges `(start, end)` of sentence indices.
# %% task Task 1: sentence-window chunking
def chunk_sentences(sentences, size, stride):
    # return a list of (start, end) pairs: start = 0, stride, 2*stride, ...; end = min(start + size, n)
    # stop right after the window whose end == n. An empty input gives [].
    raise NotImplementedError   # TODO
# %% check
_s = [f"s{i}" for i in range(8)]
check("size 3 stride 1 on 8 sentences gives 6 windows ending at (5, 8)", lambda: chunk_sentences(_s, 3, 1) == [(i, i + 3) for i in range(6)], "(0,3),(1,4),...,(5,8)")
check("size 4 stride 2", lambda: chunk_sentences(_s, 4, 2) == [(0, 4), (2, 6), (4, 8)])
check("a window larger than the document gives a single chunk", lambda: chunk_sentences(_s, 20, 5) == [(0, 8)])
check("stride = size gives a partition", lambda: chunk_sentences(_s, 3, 3) == [(0, 3), (3, 6), (6, 8)])
check("every sentence is covered by at least one window", lambda: set(i for a, b in chunk_sentences(_s, 2, 1) for i in range(a, b)) == set(range(8)))
check("empty document", lambda: chunk_sentences([], 3, 1) == [])
check("size 1 gives one window per sentence", lambda: chunk_sentences(_s, 1, 1) == [(i, i + 1) for i in range(8)])
# %% hint
Loop: start = 0; while True: end = min(start + size, n); append (start, end); if end == n: break; start += stride.
# %% solution
def chunk_sentences(sentences, size, stride):
    n = len(sentences)
    out = []
    if n == 0:
        return out
    start = 0
    while True:
        end = min(start + size, n)
        out.append((start, end))
        if end == n:
            break
        start += stride
    return out
# %% md Retrieval metrics
For one question we have a ranked list of chunk ids and a set of **relevant** chunks (those that contain the gold answer sentence). Three numbers:
- **answer recall@k**: 1 if at least one relevant chunk is in the top $k$, else 0. Averaged over questions it is the share of questions whose answer was reachable by the reader. (This is the "hit rate" definition used for RAG; it makes different chunk sizes comparable because several overlapping chunks may be relevant.)
- **MRR**: reciprocal rank $1/r$ of the first relevant chunk ($0$ if absent from the ranking).
- **nDCG@k** with binary gains: $\mathrm{DCG@k}=\sum_{i=1}^{k} \mathrm{rel}_i/\log_2(i+1)$, divided by the ideal DCG (all relevant chunks first, at most $k$ of them).
# %% task Task 2: recall@k, MRR and nDCG@k
def recall_at_k(ranked, relevant, k):
    raise NotImplementedError   # TODO
def mrr(ranked, relevant):
    raise NotImplementedError   # TODO
def ndcg_at_k(ranked, relevant, k):
    raise NotImplementedError   # TODO
# %% check
check("recall@k is a hit indicator", lambda: recall_at_k([5, 2, 9], {9}, 3) == 1.0 and recall_at_k([5, 2, 9], {9}, 2) == 0.0)
check("recall@k with several relevant chunks needs only one", lambda: recall_at_k([5, 2, 9], {2, 9, 11}, 2) == 1.0)
check("MRR of first relevant at rank 2", lambda: mrr([5, 2, 9], {2, 9}) == 0.5)
check("MRR is 0 when nothing relevant is ranked", lambda: mrr([5, 2, 9], {7}) == 0.0)
check("nDCG hand example 0.6934", lambda: abs(ndcg_at_k([5, 2, 9], {2, 9}, 3) - (1 / np.log2(3) + 0.5) / (1 + 1 / np.log2(3))) < 1e-9)
check("nDCG is 1 for an ideal ranking and 0 for none found", lambda: ndcg_at_k([2, 9, 5], {2, 9}, 3) == 1.0 and ndcg_at_k([5, 6], {2}, 2) == 0.0)
check("nDCG cuts off at k", lambda: ndcg_at_k([5, 6, 2], {2}, 2) == 0.0 and ndcg_at_k([5, 6, 2], {2}, 3) > 0)
# %% hint
nDCG: gains = [1 if r in relevant else 0 for r in ranked[:k]]; dcg = sum(g / np.log2(i + 2)); ideal has min(len(relevant), k) ones.
# %% solution
def recall_at_k(ranked, relevant, k):
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0

def mrr(ranked, relevant):
    for i, r in enumerate(ranked):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(ranked, relevant, k):
    gains = np.array([1.0 if r in relevant else 0.0 for r in ranked[:k]])
    dcg = float((gains / np.log2(np.arange(2, len(gains) + 2))).sum())
    m = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, m + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0
# %% md Uncertainty: the bootstrap
With 80 questions, a difference of 0.03 in MRR may be noise. The **percentile bootstrap** resamples the per-question scores with replacement $B$ times, recomputes the mean each time, and takes the 2.5% and 97.5% quantiles. For comparing two systems on the *same* questions use a **paired** bootstrap: resample question indices once and apply them to both systems, then look at the distribution of the mean difference. Pairing cancels the question-to-question difficulty and gives much tighter intervals.
# %% task Task 3: bootstrap intervals
def bootstrap_ci(values, B=1000, seed=0):
    # returns (mean, lo, hi): lo/hi are the 2.5 and 97.5 percentiles of the B resampled means
    raise NotImplementedError   # TODO

def paired_bootstrap(a, b, B=1000, seed=0):
    # returns (mean(a - b), lo, hi) using the SAME resampled indices for a and b
    raise NotImplementedError   # TODO
# %% check
_v = np.random.default_rng(0).random(80)
_m, _lo, _hi = bootstrap_ci(_v)
check("mean is the sample mean and lies in its interval", lambda: abs(_m - _v.mean()) < 1e-12 and _lo <= _m <= _hi)
check("a constant sample has a zero-width interval", lambda: bootstrap_ci(np.ones(30))[1:] == (1.0, 1.0))
check("more data gives a narrower interval", lambda: (lambda w1, w2: w2 < w1)(np.subtract(*bootstrap_ci(_v[:20])[2:0:-1]), np.subtract(*bootstrap_ci(np.random.default_rng(1).random(400))[2:0:-1])))
check("deterministic for a seed", lambda: bootstrap_ci(_v, seed=3) == bootstrap_ci(_v, seed=3))
_b = np.random.default_rng(2).random(80) * 5
check("identical systems give a zero difference", lambda: paired_bootstrap(_b, _b) == (0.0, 0.0, 0.0))
_d, _dlo, _dhi = paired_bootstrap(_b + 0.1, _b)
check("pairing makes a constant gain certain, however noisy the questions", lambda: abs(_d - 0.1) < 1e-9 and _dhi - _dlo < 1e-9, "resample one index vector, use it for both")
check("a clearly better system has an interval above 0", lambda: paired_bootstrap(_b + np.random.default_rng(3).normal(0.5, 0.2, 80), _b)[1] > 0)
# %% hint
rng = np.random.default_rng(seed); idx = rng.integers(0, n, (B, n)); means = values[idx].mean(axis=1); np.percentile(means, [2.5, 97.5]).
# %% solution
def bootstrap_ci(values, B=1000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (B, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)

def paired_bootstrap(a, b, B=1000, seed=0):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
# %% md Stage 2 summary
You can cut documents into overlapping windows, score a ranking three ways, and attach an honest interval to any mean. The sentence-window structure of our corpus will decide which chunk sizes work.
# %% report
Why can't we use plain "recall = share of relevant chunks retrieved" when comparing chunk sizes? Explain with the overlapping windows.
# %% md Stage 3: Sparse retrieval
Both retrievers below score every chunk against a query. They share a tokenizer (`tokenize`: lowercase, letters and digits).

**TF-IDF with cosine.** With $N$ chunks and document frequency $\mathrm{df}_t$ (number of chunks containing term $t$):

$$
\mathrm{idf}_t = \ln\frac{1+N}{1+\mathrm{df}_t} + 1,\qquad w_{d,t} = (1+\ln \mathrm{tf}_{d,t})\,\mathrm{idf}_t \ \text{ if } \mathrm{tf}>0 \text{ else } 0.
$$

Each chunk row $w_d$ and each query vector is scaled to unit L2 norm, so the score is the cosine similarity. Query terms not in the vocabulary are ignored.
# %% task Task 4: TF-IDF index
class TfidfIndex:
    def __init__(self, docs_tokens):
        # build: self.vocab (token -> column), self.idf (V,), self.X (N, V) unit-norm rows, self.N
        raise NotImplementedError   # TODO

    def qvec(self, q_tokens):
        # (V,) unit-norm query vector with the same weighting (zero vector if no query term is known)
        raise NotImplementedError   # TODO

    def scores(self, q_tokens):
        # (N,) cosine similarities
        raise NotImplementedError   # TODO
# %% check
_T = TfidfIndex([["a", "b", "b"], ["b", "c"], ["c", "d"]])
check("vocabulary and idf formula", lambda: set(_T.vocab) == {"a", "b", "c", "d"} and abs(_T.idf[_T.vocab["a"]] - (np.log(4 / 2) + 1)) < 1e-9 and abs(_T.idf[_T.vocab["b"]] - (np.log(4 / 3) + 1)) < 1e-9)
check("matrix shape and unit-norm rows", lambda: _T.X.shape == (3, 4) and np.allclose(np.linalg.norm(_T.X, axis=1), 1))
check("sublinear tf: the first document weights b as (1 + ln 2) * idf", lambda: abs(_T.X[0, _T.vocab["b"]] / _T.X[0, _T.vocab["a"]] - (1 + np.log(2)) * _T.idf[_T.vocab["b"]] / _T.idf[_T.vocab["a"]]) < 1e-9)
check("a rare term ranks its document first", lambda: int(np.argmax(_T.scores(["a"]))) == 0 and int(np.argmax(_T.scores(["d"]))) == 2)
check("unknown terms are ignored and an all-unknown query scores 0", lambda: _T.scores(["zzz"]).shape == (3,) and (_T.scores(["zzz"]) == 0).all() and np.allclose(_T.scores(["a", "zzz"]), _T.scores(["a"])))
check("scores are cosines in [0, 1]", lambda: (_T.scores(["b", "c"]) <= 1 + 1e-9).all() and (_T.scores(["b", "c"]) >= 0).all())
# %% hint
vocab = {t: j for j, t in enumerate(sorted(set(all tokens)))}; counts matrix by looping over docs; df = (counts > 0).sum(0).
# %% hint
X = np.where(counts > 0, (1 + np.log(np.maximum(counts, 1))) * idf, 0); divide rows by np.maximum(norm, 1e-12). Use the same code for the query counts.
# %% solution
class TfidfIndex:
    def __init__(self, docs_tokens):
        vocab = sorted(set(t for d in docs_tokens for t in d))
        self.vocab = {t: j for j, t in enumerate(vocab)}
        self.N = len(docs_tokens)
        counts = np.zeros((self.N, len(vocab)))
        for i, d in enumerate(docs_tokens):
            for t in d:
                counts[i, self.vocab[t]] += 1
        self.counts = counts
        df = (counts > 0).sum(axis=0)
        self.idf = np.log((1 + self.N) / (1 + df)) + 1
        W = np.where(counts > 0, (1 + np.log(np.maximum(counts, 1))) * self.idf, 0.0)
        self.X = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)

    def qvec(self, q_tokens):
        c = np.zeros(len(self.vocab))
        for t in q_tokens:
            if t in self.vocab:
                c[self.vocab[t]] += 1
        w = np.where(c > 0, (1 + np.log(np.maximum(c, 1))) * self.idf, 0.0)
        return w / max(np.linalg.norm(w), 1e-12)

    def scores(self, q_tokens):
        return self.X @ self.qvec(q_tokens)
# %% md BM25
**BM25** is TF-IDF's sturdier cousin. Each query term contributes

$$
\mathrm{idf}_t\cdot\frac{\mathrm{tf}\,(k_1+1)}{\mathrm{tf} + k_1\,(1-b+b\,\mathrm{dl}/\mathrm{avgdl})},\qquad \mathrm{idf}_t=\ln\!\Big(1+\frac{N-\mathrm{df}_t+0.5}{\mathrm{df}_t+0.5}\Big)
$$

where $\mathrm{dl}$ is the chunk length in tokens and $\mathrm{avgdl}$ the mean length. The fraction **saturates** in tf (the tenth occurrence of a word adds little), and $b$ controls how strongly long chunks are penalised. Each distinct query term counts once. Use $k_1=1.5$, $b=0.75$ as defaults.
# %% task Task 5: BM25 index
class Bm25Index:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        # build: self.vocab, self.counts (N, V), self.dl (N,), self.avgdl, self.idf (V,)
        raise NotImplementedError   # TODO

    def scores(self, q_tokens):
        # (N,) BM25 scores; unknown terms contribute 0
        raise NotImplementedError   # TODO
# %% check
_docs = [["a", "b", "b"], ["b", "c"], ["c", "d", "d", "d"]]
_B = Bm25Index(_docs)
def _hand(dix, term):
    N = 3; df = sum(term in d for d in _docs); idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    tf = _docs[dix].count(term); dl = len(_docs[dix]); avg = np.mean([len(d) for d in _docs])
    return idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
check("matches the formula for a single term", lambda: np.allclose(_B.scores(["b"]), [_hand(0, "b"), _hand(1, "b"), 0.0]))
check("scores add over distinct query terms and duplicates count once", lambda: np.allclose(_B.scores(["b", "c", "b"]), _B.scores(["b", "c"])) and np.allclose(_B.scores(["b", "c"])[1], _hand(1, "b") + _hand(1, "c")))
check("term-frequency saturation: doubling tf less than doubles the score", lambda: (lambda s: s[0] < s[1] < 2 * s[0] and s[1] < s[2] < 2 * s[1])(Bm25Index([["x", "f", "f", "f"], ["x", "x", "f", "f"], ["x", "x", "x", "x"], ["g"] * 4]).scores(["x"])))
_long = Bm25Index([["x"] + ["f"] * 2, ["x"] + ["f"] * 40, ["g"] * 5])
check("with b = 0.75 a longer chunk scores lower for the same tf", lambda: _long.scores(["x"])[0] > _long.scores(["x"])[1])
check("with b = 0 length does not matter", lambda: abs(Bm25Index([["x"] + ["f"] * 2, ["x"] + ["f"] * 40, ["g"] * 5], b=0.0).scores(["x"])[0] - Bm25Index([["x"] + ["f"] * 2, ["x"] + ["f"] * 40, ["g"] * 5], b=0.0).scores(["x"])[1]) < 1e-12)
check("unknown terms give zero scores", lambda: (_B.scores(["zzz"]) == 0).all() and _B.scores(["zzz"]).shape == (3,))
check("rare term outranks common term for the same document", lambda: _B.scores(["a"])[0] > _B.scores(["b"])[0] * 0.5)
# %% hint
tf_part = counts[:, cols] * (k1 + 1) / (counts[:, cols] + k1 * (1 - b + b * dl[:, None] / avgdl)), then multiply by idf[cols] and sum over columns. cols = distinct known query terms.
# %% solution
class Bm25Index:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        vocab = sorted(set(t for d in docs_tokens for t in d))
        self.vocab = {t: j for j, t in enumerate(vocab)}
        N = len(docs_tokens)
        counts = np.zeros((N, len(vocab)))
        for i, d in enumerate(docs_tokens):
            for t in d:
                counts[i, self.vocab[t]] += 1
        self.counts = counts
        self.dl = counts.sum(axis=1)
        self.avgdl = self.dl.mean()
        df = (counts > 0).sum(axis=0)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        self.N = N

    def scores(self, q_tokens):
        cols = sorted(set(self.vocab[t] for t in q_tokens if t in self.vocab))
        if not cols:
            return np.zeros(self.N)
        tf = self.counts[:, cols]
        norm = self.k1 * (1 - self.b + self.b * self.dl[:, None] / self.avgdl)
        return (tf * (self.k1 + 1) / (tf + norm) * self.idf[cols]).sum(axis=1)
# %% code
HASH_D = 512
_tok_cache = {}

def trigrams(tok):
    w = "#" + tok + "#"
    return [w[i:i + 3] for i in range(len(w) - 2)]

def hash_token_vec(tok):
    """Signed hashing of character trigrams into HASH_D buckets. Works for ANY token, even unseen ones."""
    if tok not in _tok_cache:
        v = np.zeros(HASH_D)
        for g in trigrams(tok):
            h = zlib.crc32(g.encode())
            v[h % HASH_D] += 1.0 if (h >> 16) & 1 else -1.0
        _tok_cache[tok] = v
    return _tok_cache[tok]

def unit(M):
    M = np.atleast_2d(M)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

def build_system(docs, size, dim=24, lsa=True):
    stride = max(1, size // 2)
    chunks = []
    for d in docs:
        for a, b in chunk_sentences(d["sents"], size, stride):
            text = " ".join(d["sents"][a:b])
            chunks.append({"doc": d["id"], "a": a, "b": b, "tok": tokenize(text)})
    toks = [c["tok"] for c in chunks]
    S = {"size": size, "chunks": chunks, "toks": toks, "docs": {d["id"]: d for d in docs}}
    S["tfidf"], S["bm25"] = TfidfIndex(toks), Bm25Index(toks)
    S["idf"] = {t: float(S["tfidf"].idf[j]) for t, j in S["tfidf"].vocab.items()}
    S["max_idf"] = float(S["tfidf"].idf.max())
    tv = np.stack([hash_token_vec(t) for t in sorted(S["tfidf"].vocab, key=S["tfidf"].vocab.get)])
    S["TokVec"] = tv
    S["Zh"] = unit(S["tfidf"].X @ tv)               # hashed trigram "embedding" of every chunk
    S["lsa"] = LsaEmbedder(dim) if (lsa and "LsaEmbedder" in globals()) else None
    if S["lsa"] is not None:
        S["lsa"].fit(S["tfidf"].X)
        S["Zl"] = S["lsa"].transform(S["tfidf"].X)
    return S

def relevant_chunks(S, q):
    return {i for i, c in enumerate(S["chunks"]) if c["doc"] == q["doc"] and c["a"] <= q["sent"] < c["b"]}
# %% code
# Sparse systems only for now (the dense part is added in Stage 4)
def sparse_rank(S, method, q_tokens):
    s = S["bm25"].scores(q_tokens) if method == "bm25" else S["tfidf"].scores(q_tokens)
    return list(np.argsort(-s, kind="stable"))

def evaluate(S, queries, rank_fn, k=5):
    out = {"hit1": [], "hitk": [], "mrr": [], "ndcg": []}
    for q in queries:
        ranked = rank_fn(S, tokenize(q["q"]))
        rel = relevant_chunks(S, q)
        out["hit1"].append(recall_at_k(ranked, rel, 1)); out["hitk"].append(recall_at_k(ranked, rel, k))
        out["mrr"].append(mrr(ranked, rel)); out["ndcg"].append(ndcg_at_k(ranked, rel, 10))
    return {m: np.array(v) for m, v in out.items()}

def fmt(vals):
    m, lo, hi = bootstrap_ci(vals)
    return f"{m:.2f} [{lo:.2f},{hi:.2f}]"

S3 = build_system(DOCS, size=3)
print(f"chunks: {len(S3['chunks'])} (size 3, stride 1), vocabulary {len(S3['tfidf'].vocab)}")
print("method   recall@1          recall@5          MRR               nDCG@10")
R_sparse = {}
for m in ["tfidf", "bm25"]:
    R_sparse[m] = evaluate(S3, QUERIES, lambda S, qt, m=m: sparse_rank(S, m, qt))
    r = R_sparse[m]
    print(f"{m:8s} {fmt(r['hit1'])}  {fmt(r['hitk'])}  {fmt(r['mrr'])}  {fmt(r['ndcg'])}")
d, lo, hi = paired_bootstrap(R_sparse["bm25"]["mrr"], R_sparse["tfidf"]["mrr"])
print(f"paired MRR difference bm25 - tfidf: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
# %% md Stage 3 summary
Two lexical retrievers, scored on the same labelled questions with intervals. Look at the paired difference, not just the two means: if its interval contains zero, the data does not separate the retrievers.
# %% report
Which of the four question types do you think BM25 loses most on, and why? Check by splitting the table by type in Stage 7.
# %% md Stage 4: Dense and hybrid retrieval
**An embedder without a neural network: LSA.** Take the TF-IDF matrix $X\in\mathbb R^{N\times V}$ and its SVD $X=U\Sigma V^\top$. Keep the top $r$ right singular vectors $V_r\in\mathbb R^{V\times r}$. A chunk (or query) with TF-IDF vector $x\in\mathbb R^V$ is embedded as $z=\mathrm{unit}(xV_r)\in\mathbb R^{r}$. Terms that occur together in the same chunks end up with similar loadings, so a chunk about "pump flow" and a query about "liquid pipe" can become neighbours even without sharing a word. The price: rare words (like our machine names) are discarded as noise, because they carry little variance.
# %% task Task 6: the LSA embedder
class LsaEmbedder:
    def __init__(self, dim):
        self.dim = dim

    def fit(self, X):
        # SVD of X (N, V); keep self.Vr (V, dim) = first dim rows of Vt, transposed
        raise NotImplementedError   # TODO
        return self

    def transform(self, M):
        # M (n, V) -> (n, dim), rows scaled to unit L2 norm (zero rows stay zero, no NaN)
        raise NotImplementedError   # TODO
# %% check
_Xe = np.random.default_rng(0).random((30, 12)); _Xe[:, 0] = 0.0
_E = LsaEmbedder(5).fit(_Xe)
_Z = _E.transform(_Xe)
check("shapes: Vr is (V, dim), embeddings (n, dim)", lambda: _E.Vr.shape == (12, 5) and _Z.shape == (30, 5))
check("rows have unit norm", lambda: np.allclose(np.linalg.norm(_Z, axis=1), 1))
check("zero input row stays zero (no NaN)", lambda: not np.isnan(_E.transform(np.zeros((2, 12)))).any() and (_E.transform(np.zeros((2, 12))) == 0).all())
_Xu = unit(_Xe)
check("at full rank cosine similarities are preserved", lambda: np.allclose(LsaEmbedder(12).fit(_Xu).transform(_Xu) @ LsaEmbedder(12).fit(_Xu).transform(_Xu).T, _Xu @ _Xu.T, atol=1e-8))
check("Vr columns are orthonormal", lambda: np.allclose(_E.Vr.T @ _E.Vr, np.eye(5), atol=1e-8))
_Xd = np.vstack([_Xe, _Xe[:1] + 1e-3])
check("near-duplicates stay near each other", lambda: (lambda Z: float(Z[0] @ Z[-1]) > 0.999)(LsaEmbedder(5).fit(_Xd).transform(_Xd)))
# %% hint
U, s, Vt = np.linalg.svd(X, full_matrices=False); self.Vr = Vt[:dim].T
# %% hint
transform: Z = M @ self.Vr; norms = np.linalg.norm(Z, axis=1, keepdims=True); Z / np.maximum(norms, 1e-12).
# %% solution
class LsaEmbedder:
    def __init__(self, dim):
        self.dim = dim

    def fit(self, X):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.Vr = Vt[:self.dim].T
        return self

    def transform(self, M):
        Z = np.atleast_2d(M) @ self.Vr
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
# %% md Reciprocal rank fusion
Different retrievers produce scores on different scales (BM25 is unbounded, cosines live in $[-1,1]$), so averaging scores is fragile. **Reciprocal rank fusion** uses only ranks:

$$
\mathrm{RRF}(d)=\sum_{r\in\text{rankings}}\frac{1}{k+\mathrm{rank}_r(d)},\qquad \mathrm{rank}\ \text{starting at }1,\ k=60.
$$

A document absent from a ranking contributes nothing for that ranking.
# %% task Task 7: reciprocal rank fusion
def rrf(rankings, k=60):
    # rankings: list of lists of ids (best first). Return a list of (id, score) sorted by score descending;
    # ties are broken by smaller id first so the result is deterministic.
    raise NotImplementedError   # TODO
# %% check
_f = rrf([[1, 2, 3], [3, 1, 4]])
check("hand-computed scores", lambda: abs(dict(_f)[1] - (1 / 61 + 1 / 62)) < 1e-12 and abs(dict(_f)[3] - (1 / 63 + 1 / 61)) < 1e-12 and abs(dict(_f)[4] - 1 / 63) < 1e-12)
check("sorted by fused score", lambda: [i for i, _ in _f] == [1, 3, 2, 4] and all(_f[i][1] >= _f[i + 1][1] for i in range(len(_f) - 1)))
check("a document only in one list still appears", lambda: {i for i, _ in _f} == {1, 2, 3, 4})
check("a single ranking keeps its order", lambda: [i for i, _ in rrf([[7, 5, 9]])] == [7, 5, 9])
check("ties are broken by smaller id", lambda: [i for i, _ in rrf([[2, 1], [1, 2]])] == [1, 2])
check("a small k rewards top ranks more", lambda: (lambda r1, r2: r1[0][1] / r1[1][1] > r2[0][1] / r2[1][1])(rrf([[1, 2]], k=1), rrf([[1, 2]], k=100)))
# %% hint
score = {}; for each ranking, for pos, id in enumerate(ranking): score[id] = score.get(id, 0) + 1 / (k + pos + 1). Sort with key (-score, id).
# %% solution
def rrf(rankings, k=60):
    score = {}
    for ranking in rankings:
        for pos, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score.items(), key=lambda t: (-t[1], t[0]))
# %% code
def dense_scores(S, method, q_tokens):
    if method == "lsa":
        return S["Zl"] @ S["lsa"].transform(S["tfidf"].qvec(q_tokens))[0]
    # hash: OOV query tokens (typos!) are embedded too, with the maximum idf (they are rare by definition)
    w = np.array([S["idf"].get(t, S["max_idf"]) for t in q_tokens])
    qv = unit((w[:, None] * np.stack([hash_token_vec(t) for t in q_tokens])).sum(axis=0))[0]
    return S["Zh"] @ qv

def full_rank(S, method, q_tokens, depth=40):
    if method in ("bm25", "tfidf"):
        return sparse_rank(S, method, q_tokens)
    if method in ("lsa", "hash"):
        return list(np.argsort(-dense_scores(S, method, q_tokens), kind="stable"))
    if method == "hybrid":      # RRF of BM25 + hashed trigrams + LSA
        lists = [full_rank(S, m, q_tokens)[:depth] for m in ("bm25", "hash", "lsa")]
        return [i for i, _ in rrf(lists)]
    if method == "hybrid2":     # RRF of BM25 + hashed trigrams only
        lists = [full_rank(S, m, q_tokens)[:depth] for m in ("bm25", "hash")]
        return [i for i, _ in rrf(lists)]
    raise ValueError(method)

S3 = build_system(DOCS, size=3)
print("method   recall@1          recall@5          MRR               nDCG@10")
R4 = {}
for m in ["bm25", "lsa", "hash", "hybrid2", "hybrid"]:
    R4[m] = evaluate(S3, QUERIES, lambda S, qt, m=m: full_rank(S, m, qt))
    r = R4[m]
    print(f"{m:8s} {fmt(r['hit1'])}  {fmt(r['hitk'])}  {fmt(r['mrr'])}  {fmt(r['ndcg'])}")
for m in ["hybrid2", "hybrid"]:
    d, lo, hi = paired_bootstrap(R4[m]["mrr"], R4["bm25"]["mrr"])
    print(f"paired MRR {m} - bm25: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
# %% md Stage 4 summary
Hashed character trigrams survive typos that BM25 cannot see, LSA is cheap and soft but loses rare names, and RRF combines their rankings without tuning score scales. Whether the hybrid is better than BM25 alone is a question for the paired interval, not for the eye.
# %% report
Compare `hybrid` with `hybrid2` (the only difference: LSA is left out). What does the comparison say about the usefulness of the LSA retriever on this corpus, and why might that be?
# %% md Stage 5: Rerank and answer
**Reranking.** The first stage should be cheap and recall-oriented. A second stage re-scores only the top few candidates with richer evidence. Our reranker uses three features computed on *content words* (stopwords removed):
- the normalised first-stage score (min-max over the candidates; 0 if they are all equal),
- **coverage**: fraction of distinct query content words present in the chunk,
- **bigram match**: fraction of adjacent query content-word pairs that appear adjacent in the chunk's content words.

The final score is `w[0]*base + w[1]*coverage + w[2]*bigram`.
# %% code
STOP = set("a an the of in is was it its to by and for on at as be this that which what who when where how did does do year city".split())

def content(tokens):
    return [t for t in tokens if t not in STOP]
# %% task Task 8: feature reranker
def rerank(q_tokens, cand, base, chunk_tokens, w=(1.0, 0.5, 0.5)):
    # cand: list of chunk ids; base: array of first-stage scores aligned with cand; chunk_tokens: list of token lists indexed by chunk id
    # return the cand ids reordered by decreasing final score (stable for ties)
    raise NotImplementedError   # TODO
# %% check
_ct = {0: ["pump", "valve", "flow"], 1: ["valve", "x", "pump", "flow"], 2: ["flow", "pump", "valve"], 3: ["pump", "valve", "flow", "extra"]}
_q = ["the", "valve", "pump", "flow"]
check("returns a permutation of the candidates", lambda: sorted(rerank(_q, [0, 1, 2], np.array([1.0, 1.0, 1.0]), _ct)) == [0, 1, 2])
check("with equal base scores the chunk containing the query bigrams wins", lambda: rerank(["valve", "pump", "flow"], [0, 1, 2], np.array([1.0, 1.0, 1.0]), _ct)[0] == 1, "bigram valve-pump and pump-flow occur only in chunk 1")
check("w = (1, 0, 0) preserves the first-stage order", lambda: rerank(_q, [2, 0, 1], np.array([3.0, 2.0, 1.0]), _ct, w=(1.0, 0.0, 0.0)) == [2, 0, 1])
check("coverage rewards chunks that contain more query terms", lambda: rerank(["valve", "pump", "gasket"], [4, 5], np.array([1.0, 1.0]), {4: ["valve", "x"], 5: ["pump", "valve"]}, w=(0.0, 1.0, 0.0))[0] == 5)
check("constant base scores do not produce NaN", lambda: len(rerank(_q, [0, 1], np.array([2.0, 2.0]), _ct)) == 2)
check("a strong first-stage lead can survive the features", lambda: rerank(["pump"], [0, 1], np.array([10.0, 0.0]), _ct, w=(1.0, 0.1, 0.1))[0] == 0)
# %% hint
cq = content(q_tokens); coverage = len(set(cq) & set(content(chunk))) / max(len(set(cq)), 1); bigrams are zip(cq, cq[1:]) vs zip(cchunk, cchunk[1:]).
# %% solution
def rerank(q_tokens, cand, base, chunk_tokens, w=(1.0, 0.5, 0.5)):
    cq = content(q_tokens)
    qset = set(cq)
    qbi = set(zip(cq, cq[1:]))
    base = np.asarray(base, dtype=float)
    span = base.max() - base.min() if len(base) else 0.0
    bn = (base - base.min()) / span if span > 0 else np.zeros_like(base)
    final = []
    for j, c in enumerate(cand):
        cc = content(chunk_tokens[c])
        cov = len(qset & set(cc)) / max(len(qset), 1)
        bi = len(qbi & set(zip(cc, cc[1:]))) / max(len(qbi), 1)
        final.append(w[0] * bn[j] + w[1] * cov + w[2] * bi)
    order = np.argsort(-np.array(final), kind="stable")
    return [cand[j] for j in order]
# %% md Extractive answers
The reader is a rule-based extractor. First the question's **type** is read from its words (`who`, `when`, `where`). Then:
- `extract_span(wh, sentence)` pulls the answer out of one sentence: for `who` the first pair of consecutive capitalised words **after the first word of the sentence** (the first word is capitalised anyway); for `when` the first 4-digit year from 1800 to 2099; for `where` the capitalised word following `city of`. It returns `None` if the sentence has no span of that type.
- `pick_sentence(...)` chooses the sentence among the candidates.
# %% task Task 9: span extraction
def extract_span(wh, sentence):
    raise NotImplementedError   # TODO
# %% check
check("who: skips the capitalised first word", lambda: extract_span("who", "The Zorvex pump was invented by Mira Lund.") == "Mira Lund")
check("who: first name pair even when a distractor follows", lambda: extract_span("who", "Later, Ola Berg improved the Kalvex pump.") == "Ola Berg")
check("who: none when there are no two consecutive capitalised words", lambda: extract_span("who", "The Zorvex pump is a compact device used for moving liquid.") is None)
check("when: finds a 4-digit year", lambda: extract_span("when", "It was launched in 1987.") == "1987" and extract_span("when", "It has 12345 parts.") is None)
check("where: capitalised word after 'city of'", lambda: extract_span("where", "Its main workshop is located in the city of Karsk.") == "Karsk" and extract_span("where", "It is in a city of karsk") is None)
check("unknown question type gives None", lambda: extract_span("why", "It was launched in 1987.") is None)
# %% hint
import re. who: re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", sentence.split(" ", 1)[1]); when: re.search(r"\b(1[89]\d\d|20\d\d)\b", ...); where: re.search(r"city of ([A-Z][a-z]+)", ...)
# %% solution
def extract_span(wh, sentence):
    if wh == "who":
        parts = sentence.split(" ", 1)
        if len(parts) < 2:
            return None
        m = re.search(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", parts[1])
        return m.group(1) if m else None
    if wh == "when":
        m = re.search(r"\b(1[89]\d\d|20\d\d)\b", sentence)
        return m.group(1) if m else None
    if wh == "where":
        m = re.search(r"city of ([A-Z][a-z]+)", sentence)
        return m.group(1) if m else None
    return None
# %% task Task 10: pick the answer sentence
def pick_sentence(q_tokens, wh, sentences, idf):
    # sentences: candidate sentences in retrieval order (best chunk first). idf: dict token -> weight.
    # Keep only sentences for which extract_span(wh, s) is not None. Among them choose the one with the largest
    # idf-weighted overlap: sum of idf.get(t, 0) over the DISTINCT content words (see `content`) shared by the question and the sentence.
    # Ties go to the earliest sentence. Return its index in `sentences`, or None if no sentence qualifies.
    raise NotImplementedError   # TODO
# %% check
_sent = ["The Zorvex pump is a compact device used for moving liquid.", "The Zorvex pump was invented by Mira Lund.", "Later, Ola Berg improved the design.", "The Kalvex pump was devised by Ines Haro."]
_idf = {"zorvex": 5.0, "kalvex": 5.0, "pump": 1.0, "invented": 2.0, "devised": 2.0, "design": 2.0}
_qt = ["who", "invented", "the", "zorvex", "pump"]
check("picks the sentence with the right type and the best overlap", lambda: pick_sentence(_qt, "who", _sent, _idf) == 1)
check("sentences without an answer of the right type are skipped", lambda: pick_sentence(_qt, "when", _sent, _idf) is None)
check("ties go to the earliest sentence", lambda: pick_sentence(["who", "pump"], "who", ["Later, Ola Berg improved the pump.", "Later, Ines Haro improved the pump."], _idf) == 0)
check("the distractor loses to the matching entity", lambda: pick_sentence(["who", "devised", "kalvex", "pump"], "who", _sent, _idf) == 3)
check("idf weights decide: a rare shared word beats several common ones", lambda: pick_sentence(["who", "zorvex", "pump", "design"], "who", ["Later, Ola Berg improved the design.", "The Zorvex pump was invented by Mira Lund."], _idf) == 1)
check("empty candidate list", lambda: pick_sentence(_qt, "who", [], _idf) is None)
# %% hint
overlap = sum(idf.get(t, 0.0) for t in set(content(q_tokens)) & set(content(tokenize(s)))). Use `>` (strict) when updating the best so that ties keep the earliest.
# %% solution
def pick_sentence(q_tokens, wh, sentences, idf):
    qc = set(content(q_tokens))
    best, best_s = None, -1.0
    for i, s in enumerate(sentences):
        if extract_span(wh, s) is None:
            continue
        ov = sum(idf.get(t, 0.0) for t in qc & set(content(tokenize(s))))
        if ov > best_s:
            best, best_s = i, ov
    return best
# %% code
def wh_of(q_tokens):
    if "who" in q_tokens: return "who"
    if "year" in q_tokens or "when" in q_tokens: return "when"
    if "city" in q_tokens or "where" in q_tokens: return "where"
    return None

def gather_sentences(S, ranked, k):
    seen, out = set(), []
    for ci in ranked[:k]:
        c = S["chunks"][ci]
        for si in range(c["a"], c["b"]):
            if (c["doc"], si) not in seen:
                seen.add((c["doc"], si)); out.append((c["doc"], si, S["docs"][c["doc"]]["sents"][si]))
    return out

def answer(S, q_text, ranked, k=3):
    qt = tokenize(q_text)
    cands = gather_sentences(S, ranked, k)
    j = pick_sentence(qt, wh_of(qt), [c[2] for c in cands], S["idf"])
    return None if j is None else extract_span(wh_of(qt), cands[j][2])

def retrieve(S, q_tokens, method="hybrid", depth=20, use_rerank=True):
    ranked = full_rank(S, method, q_tokens)
    if use_rerank:
        top = ranked[:depth]
        base = S["bm25"].scores(q_tokens)[top]
        ranked = rerank(q_tokens, top, base, S["toks"]) + ranked[depth:]
    return ranked

def evaluate_qa(S, queries, method, use_rerank, k=3):
    em, ret = [], {"mrr": [], "hit1": []}
    for q in queries:
        ranked = retrieve(S, tokenize(q["q"]), method, use_rerank=use_rerank)
        rel = relevant_chunks(S, q)
        ret["mrr"].append(mrr(ranked, rel)); ret["hit1"].append(recall_at_k(ranked, rel, 1))
        em.append(float(answer(S, q["q"], ranked, k) == q["answer"]))
    return np.array(em), {m: np.array(v) for m, v in ret.items()}

print("pipeline                  MRR               recall@1          exact match (reader sees top-3 chunks)")
QA = {}
for name, method, rr in [("bm25", "bm25", False), ("bm25 + rerank", "bm25", True), ("hybrid", "hybrid", False), ("hybrid + rerank", "hybrid", True)]:
    em, ret = evaluate_qa(S3, QUERIES, method, rr)
    QA[name] = (em, ret)
    print(f"{name:24s}  {fmt(ret['mrr'])}  {fmt(ret['hit1'])}  {fmt(em)}")
d, lo, hi = paired_bootstrap(QA["hybrid + rerank"][0], QA["bm25"][0])
print(f"paired exact-match difference hybrid+rerank - bm25: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
# %% md Stage 5 summary
You now have the full pipeline: chunk, retrieve (sparse, dense, fused), rerank, select, extract. The end-to-end exact-match number is bounded by retrieval quality: a perfect extractor cannot answer from chunks that do not contain the answer.
# %% report
Is the answer-level improvement (if any) from reranking about the same size as the retrieval-level improvement? Explain what the extractor does with chunks it receives that contain several sentences from the right document.
# %% md Stage 6: Ablation over chunk size and k
Two knobs we have not tuned: the chunk size (sentences per window) and the number of chunks `k` handed to the reader. We sweep both, on **three different corpora** (seeds 0, 1, 2; each with its own 80 questions) and report mean ± std over seeds. Use the hybrid of BM25 and hashed trigrams (`hybrid2`) plus the reranker; the SVD embedder is left out because it was weak in Stage 4 and is the most expensive part to rebuild for every configuration.

Expected trade-off: tiny chunks cannot contain both the name and the answer (the year sentence says *It*); huge chunks contain everything but dilute rare words and force the reader to scan more sentences.
# %% code
SIZES = [1, 2, 3, 4, 6]
SEEDS = [0, 1, 2]
grid_hit, grid_em = {}, {}   # (size, k) -> list over seeds

def run_seed(seed):
    docs = make_corpus(seed); qs = make_queries(docs, seed)
    for size in SIZES:
        S = build_system(docs, size, lsa=False)      # LSA did not earn its SVD in Stage 4: fuse BM25 + hashed trigrams
        rk = [retrieve(S, tokenize(q["q"]), "hybrid2", use_rerank=True) for q in qs]
        rels = [relevant_chunks(S, q) for q in qs]
        for k in (1, 3, 5):
            grid_hit.setdefault((size, k), []).append(np.mean([recall_at_k(r, rel, k) for r, rel in zip(rk, rels)]))
        for k in (1, 3):
            grid_em.setdefault((size, k), []).append(np.mean([answer(S, q["q"], r, k) == q["answer"] for q, r in zip(qs, rk)]))
# %% code
run_seed(0)
# %% code
run_seed(1)
# %% code
run_seed(2)
print("answer recall@k, mean +- std over 3 corpus seeds")
print("size  stride | k=1            k=3            k=5")
for size in SIZES:
    cells = "   ".join(f"{np.mean(grid_hit[(size, k)]):.2f}+-{np.std(grid_hit[(size, k)]):.2f}" for k in (1, 3, 5))
    print(f"{size:4d}  {max(1, size // 2):6d} | {cells}")
print("\nexact match when the reader sees the top-k chunks")
print("size         | k=1            k=3")
for size in SIZES:
    cells = "   ".join(f"{np.mean(grid_em[(size, k)]):.2f}+-{np.std(grid_em[(size, k)]):.2f}" for k in (1, 3))
    print(f"{size:4d}         | {cells}")
# %% md Stage 6 summary
Chunk size and k interact: larger k compensates for small chunks but gives the reader more distractor sentences. The std across corpus seeds tells you which differences in the table you can trust.
# %% report
Pick the best (size, k) cell for exact match. Is it also best for recall@k? Is it clearly better than its neighbours, given the seed-to-seed std?
# %% md Stage 7: Failure analysis
For every question that was answered wrongly, we assign **one cause** from a fixed list:
- `ok`: the predicted answer equals the gold answer;
- `not_retrieved`: no relevant chunk anywhere in the ranking;
- `below_k`: a relevant chunk exists in the ranking, but beyond rank $k$, so the reader never saw it;
- `extraction`: a relevant chunk was within the top $k$ but the reader still returned a wrong answer (or `None`).
# %% task Task 11: failure mode classifier
def failure_mode(ranked, relevant, pred, gold, k):
    # ranked: list of chunk ids, relevant: set of chunk ids, pred/gold: answer strings (pred may be None)
    raise NotImplementedError   # TODO
# %% check
check("correct answers are ok", lambda: failure_mode([1, 2, 3], {2}, "Karsk", "Karsk", 3) == "ok")
check("relevant chunk beyond the ranking is not_retrieved", lambda: failure_mode([1, 2, 3], {9}, "Karsk", "Tovik", 3) == "not_retrieved")
check("relevant chunk ranked below k", lambda: failure_mode([1, 2, 3, 9], {9}, None, "Tovik", 3) == "below_k")
check("relevant chunk inside the top k but wrong prediction", lambda: failure_mode([1, 9, 3], {9}, "Karsk", "Tovik", 3) == "extraction" and failure_mode([9, 1], {9}, None, "Tovik", 1) == "extraction")
check("ok even if retrieval was imperfect (lucky answer)", lambda: failure_mode([1, 2], {9}, "1987", "1987", 1) == "ok")
check("boundary: rank exactly k counts as seen", lambda: failure_mode([1, 2, 9], {9}, "x", "y", 3) == "extraction" and failure_mode([1, 2, 9], {9}, "x", "y", 2) == "below_k")
# %% hint
First compare pred == gold. Then find the 1-based rank of the first relevant chunk; none -> not_retrieved; rank > k -> below_k; else extraction.
# %% solution
def failure_mode(ranked, relevant, pred, gold, k):
    if pred == gold:
        return "ok"
    rank = next((i + 1 for i, r in enumerate(ranked) if r in relevant), None)
    if rank is None:
        return "not_retrieved"
    if rank > k:
        return "below_k"
    return "extraction"
# %% code
S = build_system(DOCS, size=3, lsa=False)
MODES = ["ok", "not_retrieved", "below_k", "extraction"]
tab = {t: {m: 0 for m in MODES} for t in ["direct", "paraphrase", "typo", "indirect"]}
fails = []
for q in QUERIES:
    ranked = retrieve(S, tokenize(q["q"]), "hybrid2", use_rerank=True)
    pred = answer(S, q["q"], ranked, 3)
    mode = failure_mode(ranked, relevant_chunks(S, q), pred, q["answer"], 3)
    tab[q["type"]][mode] += 1
    if mode != "ok":
        fails.append((mode, q, pred))
print("type        " + "  ".join(f"{m:>13s}" for m in MODES))
for t, row in tab.items():
    print(f"{t:11s} " + "  ".join(f"{row[m]:13d}" for m in MODES))
print("\nsample of failed questions")
for mode, q, pred in fails[:8]:
    print(f"  [{mode}] {q['q']}  gold={q['answer']}  predicted={pred}")
# %% code
# Which relation is hard? Split exact match by relation and question type, then check the 'when' + direct case at chunk size 1 vs 3.
rows = {}
for q in QUERIES:
    ranked = retrieve(S, tokenize(q["q"]), "hybrid2", use_rerank=True)
    rows.setdefault((q["type"], q["rel"]), []).append(answer(S, q["q"], ranked, 3) == q["answer"])
print("type x relation exact match (n)")
for key in sorted(rows):
    print(f"  {key[0]:11s} {key[1]:6s} {np.mean(rows[key]):.2f} ({len(rows[key])})")
S1 = build_system(DOCS, size=1, lsa=False)
when_q = [q for q in QUERIES if q["type"] == "direct" and q["rel"] == "when"]
for name, SS in [("size 1", S1), ("size 3", S)]:
    h = np.mean([recall_at_k(retrieve(SS, tokenize(q["q"]), "hybrid2"), relevant_chunks(SS, q), 3) for q in when_q])
    print(f"direct 'when' questions, recall@3 with {name}: {h:.2f}   (n={len(when_q)})")
# %% md Stage 7 summary
Failures are not one thing. Some questions are lost before the reader (retrieval), some inside it (extraction), and which one dominates depends on the question type. The chunk-size experiment at the end isolates the coreference problem of the year sentence: a one-sentence chunk "It was launched in 1987." has no name to match.
# %% report
Take the two most frequent failure categories in your table. For each, name the cause in the data (not just the label), and say which single change (a different chunking, a different retriever, a fix to the reader, or more data) you would try first and why.
# %% md Stage 8: Report
Write the final report as if for a colleague deciding whether to put an LLM reader on top of this retriever. It should cover: the best configuration and its numbers with bootstrap intervals; the evidence that hybrid and reranking help (or do not) from paired differences; the chunk-size and k choice with the seed-to-seed variability; the dominant failure causes by question type; and what this synthetic benchmark cannot tell you about real documents (for example, real text has far more paraphrase and no templates).
# %% report
Final report (about 200 words). Every number must come from a table printed above.
# %% md Sources
Only material that was actually opened and read while writing this project:
- `content/authored/llm-systems/03-embeddings-rag` and `04-evaluating-llms` (section outlines and one exercise skimmed: retrieval pipeline, failure modes, building an eval set, bootstrap intervals). READ.
- `content/authored/classical-ml/10-evaluation-metrics` (section outline skimmed: ranking and threshold-free metrics). READ.
- `docs/PROJECTS.md` and `docs/AGENT_BRIEF.md` (grammar and rules). READ.
BM25, TF-IDF, LSA and reciprocal rank fusion are standard methods implemented here from their usual definitions; the corpus and questions are synthetic and generated by the code above.
