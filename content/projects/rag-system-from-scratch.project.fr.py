# %% meta
{"id": "proj.rag", "title": "Un système de questions-réponses augmenté par retrieval, from scratch", "summary": "Construis, évalue et dissèque un système de questions-réponses sur une collection de documents synthétiques, sans modèle externe : découpage en fenêtres de phrases, retrieval TF-IDF et BM25, un embedder SVD, reciprocal rank fusion, un reranker à features, un sélecteur de réponse extractif, recall/MRR/nDCG avec intervalles bootstrap, une ablation sur la taille des chunks et une analyse des échecs.", "level": "advanced", "minutes": 150, "requires": ["sys.03", "sys.04", "cml.10"], "outcomes": ["Découper des documents en fenêtres de phrases qui se chevauchent et raisonner sur le compromis lié à la taille des chunks", "Implémenter TF-IDF, BM25 et un embedder SVD (LSA), et fusionner des classements par reciprocal rank fusion", "Reranker des candidats et extraire des réponses avec un sélecteur de phrases sensible au type de question", "Rapporter recall@k, MRR et nDCG avec des intervalles de confiance bootstrap et des comparaisons appariées", "Mener une ablation sur la taille des chunks et k avec plusieurs graines de corpus et classer les raisons pour lesquelles des requêtes échouent"], "stages": ["Corpus et jeu d'évaluation", "Découpage en chunks et mesure", "Retrieval sparse", "Dense et hybride", "Reranking et réponse", "Ablation", "Analyse des échecs", "Rapport"]}
# %% md Présentation
Un système de génération augmentée par retrieval (RAG) répond à une question en deux temps : il **retrouve** (retrieve) les passages les plus susceptibles de contenir la réponse, puis il les **lit** pour produire la réponse. L'étape de lecture est généralement confiée à un modèle de langage. Ici, il n'y a aucun modèle : le « lecteur » est un petit sélecteur extractif, de sorte que chaque nombre que tu vois est causé par du code que tu peux inspecter. C'est tout l'intérêt : avant de poser un LLM par-dessus, tu dois savoir quelle est la qualité de ton retrieval, comment la mesurer et où il casse.

Tu vas construire toute la pile sur une collection synthétique de 60 notes techniques, chacune décrivant une machine inventée (son inventeur, son année de lancement et sa ville d'origine), ainsi qu'un jeu d'évaluation étiqueté de 80 questions de quatre types (directes, paraphrasées, avec faute de frappe et indirectes). Comme nous générons les données, chaque question a une phrase de réponse de référence connue, ce qui permet d'évaluer exactement le retrieval comme la réponse.

Règles du projet :
- la qualité du retrieval se mesure avec **answer recall@k, MRR et nDCG@k**, toujours accompagnés d'un intervalle bootstrap ;
- aucune affirmation du type « meilleur » sans comparaison appariée ;
- chaque échec que tu rapportes doit se voir attribuer une cause par le code, pas par intuition.
# %% md Étape 1 : corpus et jeu d'évaluation
Chaque note compte huit phrases dans un ordre fixe. La phrase 0 présente la machine, la phrase 1 nomme l'inventeur, la phrase 2 donne l'année de lancement (avec le pronom *It*, et non le nom !), la phrase 3 donne la ville, et le reste est du remplissage ou une personne distractrice qui a *amélioré* la conception. Cette structure compte : la réponse à une question sur l'« année » se trouve dans une phrase qui ne contient pas le nom de la machine, donc un chunk constitué de cette seule phrase est sans espoir. Le découpage devra gérer ce problème.

Types de questions :
- **direct** : nomme la machine et reprend les mots de la note ;
- **paraphrase** : nomme la machine mais reformule la relation (made / come out / produced) ;
- **typo** : comme direct, mais une lettre du nom de la machine est supprimée ;
- **indirect** : ne nomme pas la machine ; l'identifie par sa catégorie et son année de lancement (« Who invented the pump launched in 1987? »).
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
# %% md Bilan de l'étape 1
Tu disposes d'un corpus à la structure connue, d'un jeu de questions étiquetées avec les indices des phrases de référence, et d'une idée de la façon dont les quatre types de questions diffèrent. Tout dans ce projet est reproductible à partir de `seed=0` et, plus tard, de deux autres graines de corpus.
# %% report
Avant de construire quoi que ce soit : quel type de question t'attends-tu à voir le plus difficile pour un retriever purement lexical (par correspondance de mots), et lequel pour un retriever fondé sur les embeddings ? Écris ton pronostic ; tu le compareras aux données à l'étape 7.
# %% md Étape 2 : découpage en chunks et mesure
**Découpage en chunks.** On ne peut pas encoder ni évaluer un corpus entier d'un seul bloc, et un long document dilue la correspondance avec ses mots rares. Un chunk est une fenêtre de `size` phrases consécutives ; la fenêtre glisse de `stride` phrases, de sorte que les chunks voisins se chevauchent et qu'une réponse a moins de chances d'être coupée en deux. La dernière fenêtre doit atteindre la dernière phrase.

Les fenêtres sont des intervalles semi-ouverts `(start, end)` d'indices de phrases.
# %% task Tâche 1 : découpage par fenêtres de phrases
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
Boucle : start = 0 ; tant que vrai : end = min(start + size, n) ; ajoute (start, end) ; si end == n : arrête ; start += stride.
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
# %% md Métriques de retrieval
Pour une question, on dispose d'une liste ordonnée d'identifiants de chunks et d'un ensemble de chunks **pertinents** (ceux qui contiennent la phrase de réponse de référence). Trois nombres :
- **answer recall@k** : 1 si au moins un chunk pertinent figure parmi les $k$ premiers, sinon 0. Moyenné sur les questions, c'est la proportion de questions dont la réponse était accessible au lecteur. (C'est la définition en « hit rate » utilisée pour le RAG ; elle rend comparables des tailles de chunks différentes, car plusieurs chunks qui se chevauchent peuvent être pertinents.)
- **MRR** : rang réciproque $1/r$ du premier chunk pertinent ($0$ s'il est absent du classement).
- **nDCG@k** avec gains binaires : $\mathrm{DCG@k}=\sum_{i=1}^{k} \mathrm{rel}_i/\log_2(i+1)$, divisé par le DCG idéal (tous les chunks pertinents en premier, au plus $k$ d'entre eux).
# %% task Tâche 2 : recall@k, MRR et nDCG@k
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
nDCG : gains = [1 if r in relevant else 0 for r in ranked[:k]] ; dcg = sum(g / np.log2(i + 2)) ; l'idéal a min(len(relevant), k) uns.
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
# %% md Incertitude : le bootstrap
Avec 80 questions, un écart de 0,03 en MRR peut n'être que du bruit. Le **bootstrap par percentiles** rééchantillonne les scores par question avec remise $B$ fois, recalcule la moyenne à chaque fois, et prend les quantiles à 2,5 % et 97,5 %. Pour comparer deux systèmes sur les *mêmes* questions, utilise un bootstrap **apparié** : tire les indices de questions une seule fois et applique-les aux deux systèmes, puis regarde la distribution de la différence des moyennes. L'appariement annule la difficulté propre à chaque question et donne des intervalles beaucoup plus serrés.
# %% task Tâche 3 : intervalles bootstrap
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
rng = np.random.default_rng(seed) ; idx = rng.integers(0, n, (B, n)) ; means = values[idx].mean(axis=1) ; np.percentile(means, [2.5, 97.5]).
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
# %% md Bilan de l'étape 2
Tu sais découper des documents en fenêtres qui se chevauchent, évaluer un classement de trois façons, et associer un intervalle honnête à n'importe quelle moyenne. La structure en fenêtres de phrases de notre corpus décidera des tailles de chunks qui fonctionnent.
# %% report
Pourquoi ne peut-on pas utiliser un simple « recall = proportion de chunks pertinents retrouvés » pour comparer des tailles de chunks ? Explique-le avec les fenêtres qui se chevauchent.
# %% md Étape 3 : retrieval sparse
Les deux retrievers ci-dessous évaluent chaque chunk par rapport à une requête. Ils partagent un tokenizer (`tokenize` : minuscules, lettres et chiffres).

**TF-IDF avec cosinus.** Avec $N$ chunks et la fréquence documentaire $\mathrm{df}_t$ (nombre de chunks contenant le terme $t$) :

$$
\mathrm{idf}_t = \ln\frac{1+N}{1+\mathrm{df}_t} + 1,\qquad w_{d,t} = (1+\ln \mathrm{tf}_{d,t})\,\mathrm{idf}_t \ \text{ if } \mathrm{tf}>0 \text{ else } 0.
$$

Chaque ligne de chunk $w_d$ et chaque vecteur de requête est mis à l'échelle pour avoir une norme L2 unitaire, si bien que le score est la similarité cosinus. Les termes de la requête absents du vocabulaire sont ignorés.
# %% task Tâche 4 : index TF-IDF
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
vocab = {t: j for j, t in enumerate(sorted(set(all tokens)))} ; matrice de comptages en bouclant sur les documents ; df = (counts > 0).sum(0).
# %% hint
X = np.where(counts > 0, (1 + np.log(np.maximum(counts, 1))) * idf, 0) ; divise les lignes par np.maximum(norm, 1e-12). Utilise le même code pour les comptages de la requête.
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
**BM25** est le cousin plus robuste de TF-IDF. Chaque terme de la requête contribue

$$
\mathrm{idf}_t\cdot\frac{\mathrm{tf}\,(k_1+1)}{\mathrm{tf} + k_1\,(1-b+b\,\mathrm{dl}/\mathrm{avgdl})},\qquad \mathrm{idf}_t=\ln\!\Big(1+\frac{N-\mathrm{df}_t+0.5}{\mathrm{df}_t+0.5}\Big)
$$

où $\mathrm{dl}$ est la longueur du chunk en tokens et $\mathrm{avgdl}$ la longueur moyenne. La fraction **sature** en tf (la dixième occurrence d'un mot n'apporte presque rien), et $b$ contrôle la force de la pénalisation des longs chunks. Chaque terme distinct de la requête compte une seule fois. Utilise $k_1=1.5$, $b=0.75$ par défaut.
# %% task Tâche 5 : index BM25
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
tf_part = counts[:, cols] * (k1 + 1) / (counts[:, cols] + k1 * (1 - b + b * dl[:, None] / avgdl)), puis multiplie par idf[cols] et somme sur les colonnes. cols = termes de requête connus et distincts.
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
# %% md Bilan de l'étape 3
Deux retrievers lexicaux, évalués sur les mêmes questions étiquetées avec des intervalles. Regarde la différence appariée, pas seulement les deux moyennes : si son intervalle contient zéro, les données ne séparent pas les retrievers.
# %% report
Sur lequel des quatre types de questions penses-tu que BM25 perd le plus, et pourquoi ? Vérifie-le en découpant le tableau par type à l'étape 7.
# %% md Étape 4 : retrieval dense et hybride
**Un embedder sans réseau de neurones : LSA.** Prends la matrice TF-IDF $X\in\mathbb R^{N\times V}$ et sa SVD $X=U\Sigma V^\top$. Garde les $r$ premiers vecteurs singuliers à droite $V_r\in\mathbb R^{V\times r}$. Un chunk (ou une requête) de vecteur TF-IDF $x\in\mathbb R^V$ est encodé par $z=\mathrm{unit}(xV_r)\in\mathbb R^{r}$. Les termes qui apparaissent ensemble dans les mêmes chunks se retrouvent avec des coefficients similaires, si bien qu'un chunk sur « pump flow » et une requête sur « liquid pipe » peuvent devenir voisins même sans partager un seul mot. Le prix à payer : les mots rares (comme nos noms de machines) sont écartés comme du bruit, car ils portent peu de variance.
# %% task Tâche 6 : l'embedder LSA
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
U, s, Vt = np.linalg.svd(X, full_matrices=False) ; self.Vr = Vt[:dim].T
# %% hint
transform : Z = M @ self.Vr ; norms = np.linalg.norm(Z, axis=1, keepdims=True) ; Z / np.maximum(norms, 1e-12).
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
# %% md Reciprocal rank fusion (fusion par rangs réciproques)
Des retrievers différents produisent des scores à des échelles différentes (BM25 n'est pas borné, les cosinus vivent dans $[-1,1]$), donc moyenner les scores est fragile. La **reciprocal rank fusion** n'utilise que les rangs :

$$
\mathrm{RRF}(d)=\sum_{r\in\text{rankings}}\frac{1}{k+\mathrm{rank}_r(d)},\qquad \mathrm{rank}\ \text{starting at }1,\ k=60.
$$

Un document absent d'un classement ne contribue à rien pour ce classement.
# %% task Tâche 7 : reciprocal rank fusion
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
score = {} ; pour chaque classement, pour pos, id in enumerate(ranking) : score[id] = score.get(id, 0) + 1 / (k + pos + 1). Trie avec la clé (-score, id).
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
# %% md Bilan de l'étape 4
Les trigrammes de caractères hachés survivent aux fautes de frappe que BM25 ne peut pas voir, LSA est peu coûteux et souple mais perd les noms rares, et RRF combine leurs classements sans qu'il faille régler les échelles de scores. Savoir si l'hybride est meilleur que BM25 seul est une question pour l'intervalle apparié, pas pour l'œil.
# %% report
Compare `hybrid` et `hybrid2` (seule différence : LSA est laissé de côté). Que dit cette comparaison de l'utilité du retriever LSA sur ce corpus, et pourquoi cela pourrait-il être le cas ?
# %% md Étape 5 : reranking et réponse
**Reranking.** La première étape doit être peu coûteuse et orientée recall. Une deuxième étape ne re-score que les quelques meilleurs candidats avec des indices plus riches. Notre reranker utilise trois features calculées sur les *mots de contenu* (stopwords retirés) :
- le score de première étape normalisé (min-max sur les candidats ; 0 s'ils sont tous égaux),
- la **couverture** (coverage) : fraction des mots de contenu distincts de la requête présents dans le chunk,
- la **correspondance de bigrammes** : fraction des paires adjacentes de mots de contenu de la requête qui apparaissent adjacentes dans les mots de contenu du chunk.

Le score final est `w[0]*base + w[1]*coverage + w[2]*bigram`.
# %% code
STOP = set("a an the of in is was it its to by and for on at as be this that which what who when where how did does do year city".split())

def content(tokens):
    return [t for t in tokens if t not in STOP]
# %% task Tâche 8 : reranker à features
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
cq = content(q_tokens) ; coverage = len(set(cq) & set(content(chunk))) / max(len(set(cq)), 1) ; les bigrammes sont zip(cq, cq[1:]) comparés à zip(cchunk, cchunk[1:]).
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
# %% md Réponses extractives
Le lecteur est un extracteur à base de règles. On lit d'abord le **type** de la question à partir de ses mots (`who`, `when`, `where`). Ensuite :
- `extract_span(wh, sentence)` extrait la réponse d'une seule phrase : pour `who`, la première paire de mots capitalisés consécutifs **après le premier mot de la phrase** (le premier mot est de toute façon capitalisé) ; pour `when`, la première année à 4 chiffres de 1800 à 2099 ; pour `where`, le mot capitalisé qui suit `city of`. Elle renvoie `None` si la phrase n'a pas de segment de ce type.
- `pick_sentence(...)` choisit la phrase parmi les candidates.
# %% task Tâche 9 : extraction de segment
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
import re. who : re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", sentence.split(" ", 1)[1]) ; when : re.search(r"\b(1[89]\d\d|20\d\d)\b", ...) ; where : re.search(r"city of ([A-Z][a-z]+)", ...)
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
# %% task Tâche 10 : choisir la phrase de réponse
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
overlap = sum(idf.get(t, 0.0) for t in set(content(q_tokens)) & set(content(tokenize(s)))). Utilise `>` (strict) pour mettre à jour le meilleur, afin que les égalités gardent la phrase la plus précoce.
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
# %% md Bilan de l'étape 5
Tu as maintenant le pipeline complet : chunk, retrieval (sparse, dense, fusionné), reranking, sélection, extraction. Le score d'exact match de bout en bout est borné par la qualité du retrieval : un extracteur parfait ne peut pas répondre à partir de chunks qui ne contiennent pas la réponse.
# %% report
L'amélioration au niveau de la réponse (s'il y en a une) due au reranking est-elle de la même taille que l'amélioration au niveau du retrieval ? Explique ce que fait l'extracteur des chunks qu'il reçoit lorsqu'ils contiennent plusieurs phrases du bon document.
# %% md Étape 6 : ablation sur la taille des chunks et k
Deux paramètres que nous n'avons pas réglés : la taille des chunks (phrases par fenêtre) et le nombre `k` de chunks transmis au lecteur. Nous balayons les deux, sur **trois corpus différents** (graines 0, 1, 2 ; chacun avec ses 80 questions) et nous rapportons la moyenne ± l'écart type sur les graines. Utilise l'hybride de BM25 et de trigrammes hachés (`hybrid2`) plus le reranker ; l'embedder SVD est laissé de côté parce qu'il était faible à l'étape 4 et qu'il est la partie la plus coûteuse à reconstruire pour chaque configuration.

Compromis attendu : de minuscules chunks ne peuvent pas contenir à la fois le nom et la réponse (la phrase de l'année dit *It*) ; d'énormes chunks contiennent tout mais diluent les mots rares et obligent le lecteur à parcourir davantage de phrases.
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
# %% md Bilan de l'étape 6
Taille des chunks et k interagissent : un k plus grand compense de petits chunks mais donne au lecteur davantage de phrases distractrices. L'écart type entre les graines de corpus te dit quelles différences du tableau tu peux croire.
# %% report
Choisis la meilleure cellule (size, k) pour l'exact match. Est-elle aussi la meilleure pour recall@k ? Est-elle nettement meilleure que ses voisines, compte tenu de l'écart type d'une graine à l'autre ?
# %% md Étape 7 : analyse des échecs
Pour chaque question à laquelle on a mal répondu, nous attribuons **une cause** parmi une liste fixe :
- `ok` : la réponse prédite est égale à la réponse de référence ;
- `not_retrieved` : aucun chunk pertinent nulle part dans le classement ;
- `below_k` : un chunk pertinent existe dans le classement, mais au-delà du rang $k$, donc le lecteur ne l'a jamais vu ;
- `extraction` : un chunk pertinent figurait dans les $k$ premiers mais le lecteur a quand même renvoyé une mauvaise réponse (ou `None`).
# %% task Tâche 11 : classifieur de modes d'échec
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
Compare d'abord pred == gold. Trouve ensuite le rang (à partir de 1) du premier chunk pertinent ; aucun -> not_retrieved ; rang > k -> below_k ; sinon extraction.
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
# %% md Bilan de l'étape 7
Les échecs ne sont pas une seule chose. Certaines questions sont perdues avant le lecteur (retrieval), d'autres à l'intérieur (extraction), et ce qui domine dépend du type de question. L'expérience sur la taille des chunks, à la fin, isole le problème de coréférence de la phrase de l'année : un chunk d'une seule phrase « It was launched in 1987. » n'a aucun nom à faire correspondre.
# %% report
Prends les deux catégories d'échec les plus fréquentes de ton tableau. Pour chacune, nomme la cause dans les données (pas seulement l'étiquette), et dis quel changement unique (un autre découpage, un autre retriever, une correction du lecteur ou davantage de données) tu essaierais en premier et pourquoi.
# %% md Étape 8 : rapport
Rédige le rapport final comme pour un collègue qui doit décider s'il faut poser un lecteur LLM sur ce retriever. Il doit couvrir : la meilleure configuration et ses chiffres avec des intervalles bootstrap ; les preuves que l'hybride et le reranking aident (ou non) à partir des différences appariées ; le choix de la taille de chunk et de k avec la variabilité d'une graine à l'autre ; les causes d'échec dominantes par type de question ; et ce que ce benchmark synthétique ne peut pas te dire sur de vrais documents (par exemple, un vrai texte contient bien plus de paraphrases et aucun gabarit).
# %% report
Rapport final (environ 200 mots). Chaque nombre doit provenir d'un tableau imprimé plus haut.
# %% md Sources
Uniquement le matériel qui a réellement été ouvert et lu pendant la rédaction de ce projet :
- `content/authored/llm-systems/03-embeddings-rag` et `04-evaluating-llms` (plans de sections et un exercice parcouru : pipeline de retrieval, modes d'échec, construction d'un jeu d'évaluation, intervalles bootstrap). LU.
- `content/authored/classical-ml/10-evaluation-metrics` (plan de section parcouru : métriques de classement et métriques sans seuil). LU.
- `docs/PROJECTS.md` et `docs/AGENT_BRIEF.md` (grammaire et règles). LU.
BM25, TF-IDF, LSA et la reciprocal rank fusion sont des méthodes standard implémentées ici à partir de leurs définitions usuelles ; le corpus et les questions sont synthétiques et générés par le code ci-dessus.
