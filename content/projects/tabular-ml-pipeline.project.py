# %% meta
{"id": "proj.tabular", "title": "A leakage-proof tabular ML pipeline", "summary": "Take a messy synthetic churn table from raw columns to an honest, cost-aware, once-only test evaluation: fold-safe preprocessing, baselines, logistic regression and boosted stumps written by you, PR-AUC and calibration, a planted leak you must catch, budgeted search, slice analysis and a written report.", "level": "intermediate", "minutes": 150, "requires": ["cml.02", "cml.03", "cml.07", "cml.10", "cml.11", "cml.15"], "outcomes": ["Build a preprocessing pipeline that is fitted inside every cross-validation fold", "Compare baselines, a linear model and a boosted model with PR-AUC averaged over seeds", "Detect and remove a planted leaky feature with a single-feature scan", "Check and repair calibration, then choose a decision threshold from costs", "Spend a fixed hyper-parameter budget, analyse slices, and touch the test set exactly once"], "stages": ["Data and EDA", "Pipeline and validation", "Metrics and baselines", "Models", "Leakage hunt", "Calibration and decision", "Tuning and slices", "Final test and report"]}
# %% md Briefing
You work for a subscription company. Every month some customers cancel (**churn**). The retention team can phone customers and offer a discount, but each call costs money, and losing a customer costs ten times more than one call. Your job: build a model that ranks customers by churn risk, pick a calling threshold, and tell the team honestly how well it will work on **new** customers.

The data is synthetic (we generate it with NumPy so the answer is known) but it has every wart of a real table: numeric and categorical columns, missing values, only about one customer in five churns, one group of customers that is intrinsically hard to predict, and **one feature that secretly leaks the label**. You will not be told which one until you find it.

The rules of the project are the rules of the lessons you have done:
- every statistic used to transform data (medians, means, categories) is learned on training rows only, inside each fold;
- the test set is opened once, at the very end;
- every claim in the report must be backed by a number that the notebook printed.

Work through the cells in order. Cells marked *task* are yours to complete; the checks run when you press Check.
# %% md Stage 1: Data and EDA
We generate 6000 customers. Columns:

| column | type | meaning |
|---|---|---|
| `tenure` | numeric | months as a customer |
| `monthly_charge` | numeric | euros per month |
| `tickets` | numeric | support tickets in the last 90 days |
| `usage_hours` | numeric | hours of use per week |
| `retention_call_min` | numeric | minutes of retention-team call logged for this customer |
| `plan`, `region`, `channel` | categorical | contract plan, region, sign-up channel (`region` has blanks) |

Missing numeric values are `NaN`; a missing category is the empty string `""`.
# %% code
import numpy as np

NUM = ["tenure", "monthly_charge", "tickets", "usage_hours", "retention_call_min"]
CAT = ["plan", "region", "channel"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def make_churn(n, seed):
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.gamma(2.0, 12.0, n), 1, 72).round(1)
    monthly = np.clip(rng.normal(65, 20, n), 15, 150).round(2)
    tickets = rng.poisson(1.5, n).astype(float)
    usage = np.clip(rng.normal(25, 10, n), 0, None).round(1)
    plan = rng.choice(np.array(["basic", "plus", "pro"]), n, p=[0.5, 0.35, 0.15])
    region = rng.choice(np.array(["north", "south", "east", "west"]), n)
    channel = rng.choice(np.array(["web", "store", "phone"]), n, p=[0.5, 0.3, 0.2])
    logit = (-1.9 - 0.05 * tenure + 0.03 * (monthly - 65) + 0.45 * tickets - 0.045 * (usage - 25)
             + np.where(plan == "basic", 0.7, np.where(plan == "pro", -0.7, 0.0))
             + np.where(region == "south", 0.6, 0.0))
    # phone sign-ups carry a large hidden factor we never observe: this slice is hard
    logit = logit + np.where(channel == "phone", rng.normal(0, 2.5, n), 0.0)
    y = (rng.random(n) < sigmoid(logit)).astype(int)
    # PLANTED LEAK: this is logged during the retention call, i.e. AFTER the churn decision
    call = np.where(y == 1, rng.normal(12, 4, n), rng.normal(2, 2, n))
    call = np.clip(call, 0, None).round(1)
    num = np.column_stack([tenure, monthly, tickets, usage, call])
    for j, rate in enumerate([0.04, 0.08, 0.0, 0.10, 0.03]):
        num[rng.random(n) < rate, j] = np.nan
    cat = np.column_stack([plan, region, channel]).astype("<U8")
    cat[rng.random(n) < 0.05, 1] = ""
    return {"num": num, "cat": cat, "y": y}

def take(D, idx):
    return {"num": D["num"][idx], "cat": D["cat"][idx], "y": D["y"][idx]}

def rankdata(a):
    """Average ranks (1-based), ties share the mean rank."""
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    return (csum - (counts - 1) / 2.0)[inv]

D = make_churn(6000, seed=7)
print("rows", len(D["y"]), "| churn rate", D["y"].mean().round(3))
# %% task Task 1: two EDA helpers
# missing_rate(Xnum)  -> array with the fraction of NaN in each column
# rate_by_group(g, y) -> dict: group value -> (number of rows, mean of y) for every distinct value in g
def missing_rate(Xnum):
    raise NotImplementedError   # TODO

def rate_by_group(g, y):
    raise NotImplementedError   # TODO
# %% check
_X = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, 4.0], [5.0, np.nan]])
check("missing_rate per column", lambda: np.allclose(missing_rate(_X), [0.25, 0.75]), "mean of np.isnan along axis 0")
_g = np.array(["a", "b", "a", "a", "b"]); _y = np.array([1, 0, 0, 1, 0])
_r = rate_by_group(_g, _y)
check("rate_by_group counts and rates", lambda: _r["a"][0] == 3 and abs(_r["a"][1] - 2 / 3) < 1e-9 and _r["b"] == (2, 0.0), "return (count, mean of y)")
check("rate_by_group keeps the empty-string group", lambda: set(rate_by_group(np.array(["x", "", "x"]), np.array([1, 1, 0]))) == {"x", ""}, "do not drop '' groups")
check("counts add up on real data", lambda: sum(c for c, _ in rate_by_group(D["cat"][:, 0], D["y"]).values()) == len(D["y"]))
check("planted missing rate on monthly_charge is near 8%", lambda: abs(missing_rate(D["num"])[1] - 0.08) < 0.02)
# %% hint
np.isnan(Xnum).mean(axis=0) gives the fraction of missing values per column.
# %% hint
For the groups, loop over sorted(set(g.tolist())) and use the boolean mask g == v.
# %% solution
def missing_rate(Xnum):
    return np.isnan(Xnum).mean(axis=0)

def rate_by_group(g, y):
    return {v: (int((g == v).sum()), float(y[g == v].mean())) for v in sorted(set(g.tolist()))}
# %% code
print("column               missing")
for name, r in zip(NUM, missing_rate(D["num"])):
    print(f"{name:20s} {r:6.1%}")
for j, name in enumerate(CAT):
    print(f"\n{name}: group -> (rows, churn rate)")
    for v, (c, r) in rate_by_group(D["cat"][:, j], D["y"]).items():
        print(f"  {v or '<blank>':8s} {c:5d}  {r:6.1%}")
print("\nnumeric columns: mean for churners vs stayers (NaN ignored)")
for j, name in enumerate(NUM):
    a = np.nanmean(D["num"][D["y"] == 1, j]); b = np.nanmean(D["num"][D["y"] == 0, j])
    print(f"{name:20s} churn {a:7.2f}   stay {b:7.2f}")
# %% report
Look at the printed tables. Which columns look informative, and which differences worry you or surprise you? Write down one thing that looks "too good" and say what you would check about it.
# %% md Stage 1 summary
You know the base rate (about one customer in five churns, so accuracy is a poor metric), where values are missing, and that categories differ in churn rate. Any column whose group means differ by a huge amount deserves suspicion before it deserves praise.
# %% md Stage 2: Pipeline and validation
Preprocessing is **part of the model**. If you compute a median or a mean on the whole table and then cross-validate, the validation rows have influenced the transformation: a mild form of leakage. The cure is a pipeline object that has `fit` (learn statistics from the rows it is given) and `transform` (apply them to any rows), and that is re-fitted inside each fold.

Your `Preprocessor` will:
1. replace each missing numeric value with the training **median** of that column;
2. standardise each numeric column with the training mean and standard deviation (after imputation);
3. one-hot encode each categorical column using the categories **seen in training**; an unseen category becomes an all-zero block. The blank `""` counts as an ordinary category.

Output shape: `(n, n_numeric + total number of training categories)`.
# %% task Task 2: the Preprocessor
class Preprocessor:
    def fit(self, Xnum, Xcat):
        # learn: self.median (k,), self.mean (k,), self.std (k,) [add 1e-9], self.cats (list of sorted category lists)
        raise NotImplementedError   # TODO
        return self

    def transform(self, Xnum, Xcat):
        raise NotImplementedError   # TODO
# %% check
_rng = np.random.default_rng(0)
_A = _rng.normal(10, 3, (200, 2)); _A[_rng.random((200, 2)) < 0.2] = np.nan
_Ac = np.array([["u", "p"], ["v", "q"]] * 100)
_pp = Preprocessor().fit(_A, _Ac)
_ZA = _pp.transform(_A, _Ac)
check("output shape = numeric + one-hot columns", lambda: _ZA.shape == (200, 2 + 2 + 2), "2 numeric + 2 + 2 categories")
check("no NaN left and train columns standardised", lambda: not np.isnan(_ZA).any() and np.allclose(_ZA[:, :2].mean(0), 0, atol=1e-6) and np.allclose(_ZA[:, :2].std(0), 1, atol=1e-3))
check("median comes from the fitted data", lambda: np.allclose(_pp.median, np.nanmedian(_A, axis=0)))
_B = _A + 100.0
check("transform does not refit (shifted data is not centred)", lambda: _pp.transform(_B, _Ac)[:, :2].mean() > 10, "transform must reuse the training mean/std")
_Cc = np.array([["zzz", "p"], ["u", "new"]])
_ZC = _pp.transform(np.array([[1.0, 2.0], [np.nan, 3.0]]), _Cc)
check("unseen categories give an all-zero block", lambda: _ZC[0, 2:4].sum() == 0 and _ZC[1, 4:6].sum() == 0 and _ZC[0, 4:6].sum() == 1)
check("missing value is imputed with the train median", lambda: abs(_ZC[1, 0] - (_pp.median[0] - _pp.mean[0]) / _pp.std[0]) < 1e-9)
# %% hint
Impute first: filled = np.where(np.isnan(X), median, X). Mean and std are those of the *filled* training matrix.
# %% hint
One-hot for column j with categories cs: (Xcat[:, j][:, None] == np.array(cs)[None, :]).astype(float). Then np.hstack all blocks.
# %% solution
class Preprocessor:
    def fit(self, Xnum, Xcat):
        self.median = np.nanmedian(Xnum, axis=0)
        filled = np.where(np.isnan(Xnum), self.median, Xnum)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0) + 1e-9
        self.cats = [sorted(set(Xcat[:, j].tolist())) for j in range(Xcat.shape[1])]
        return self

    def transform(self, Xnum, Xcat):
        filled = np.where(np.isnan(Xnum), self.median, Xnum)
        blocks = [(filled - self.mean) / self.std]
        for j, cs in enumerate(self.cats):
            blocks.append((Xcat[:, j][:, None] == np.array(cs)[None, :]).astype(float))
        return np.hstack(blocks)
# %% md Stratified folds
With about 20% positives, a random fold could contain too few churners to score. **Stratified** K-fold deals each class out separately so every fold has almost the same class ratio.
# %% task Task 3: stratified K-fold
# stratified_kfold(y, k, seed) -> list of k pairs (train_idx, val_idx)
# shuffle each class's indices with np.random.default_rng(seed), then deal them to folds 0..k-1 in turn
def stratified_kfold(y, k, seed):
    raise NotImplementedError   # TODO
# %% check
_y = (np.random.default_rng(1).random(500) < 0.15).astype(int)
_F = stratified_kfold(_y, 5, 0)
check("k folds returned", lambda: len(_F) == 5)
check("validation folds partition the data", lambda: sorted(np.concatenate([v for _, v in _F]).tolist()) == list(range(500)), "every row is validated exactly once")
check("train and validation are disjoint", lambda: all(len(np.intersect1d(t, v)) == 0 and len(t) + len(v) == 500 for t, v in _F))
check("class ratio is balanced across folds", lambda: max(_y[v].sum() for _, v in _F) - min(_y[v].sum() for _, v in _F) <= 1, "deal each class round-robin")
check("deterministic per seed, different across seeds", lambda: all((a[1] == b[1]).all() for a, b in zip(_F, stratified_kfold(_y, 5, 0))) and not all((a[1] == b[1]).all() for a, b in zip(_F, stratified_kfold(_y, 5, 1))))
# %% hint
For each class c: idx = rng.permutation(np.where(y == c)[0]); fold[idx] = np.arange(len(idx)) % k.
# %% solution
def stratified_kfold(y, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]
# %% code
# Hold out 20% as the TEST set. It will not be touched until Stage 8.
_tr, _te = stratified_kfold(D["y"], 5, seed=2024)[0]
dev, test = take(D, _tr), take(D, _te)
print("dev rows", len(dev["y"]), "positives", dev["y"].sum(), "| test rows", len(test["y"]), "positives", test["y"].sum())
# %% md Stage 2 summary
Preprocessing is now a fitted object and folds are stratified. The test set is locked away in the variable `test`; from here on all decisions use `dev` only.
# %% report
In your own words: why is "impute with the global median, then cross-validate" subtly wrong even though no label is used?
# %% md Stage 3: Metrics and baselines
With about 20% churners, a model that says "nobody churns" has about 80% accuracy and is worthless. We use two threshold-free metrics:
- **ROC-AUC**: the probability that a random churner gets a higher score than a random stayer. Computed from ranks: $\mathrm{AUC} = \frac{\sum_{i\in P} r_i - P(P+1)/2}{P\,N}$ with average ranks for ties.
- **PR-AUC** (average precision): $\mathrm{AP} = \sum_k (R_k - R_{k-1})\,P_k$ over the distinct score thresholds, from high to low. A random scorer gets AP equal to the base rate, so it is much more informative than ROC-AUC when positives are rare.

Ties must be handled by grouping equal scores into a single threshold; otherwise a constant scorer would get an arbitrary AP that depends on row order.
# %% task Task 4: ROC-AUC and average precision
# helper rankdata(a) is already defined (average ranks, 1-based)
def roc_auc(y, s):
    raise NotImplementedError   # TODO: rank formula
def average_precision(y, s):
    raise NotImplementedError   # TODO: sort by score descending, cumulative tp/fp, keep the last row of each distinct score
# %% check
_y4 = np.array([1, 0, 1, 0]); _s4 = np.array([0.9, 0.8, 0.7, 0.1])
check("roc_auc on a hand example is 0.75", lambda: abs(roc_auc(_y4, _s4) - 0.75) < 1e-9)
check("average_precision on a hand example is 0.5*1 + 0.5*2/3", lambda: abs(average_precision(_y4, _s4) - (0.5 + 0.5 * 2 / 3)) < 1e-9)
check("perfect and reversed rankings", lambda: roc_auc(_y4, _y4 * 1.0) == 1.0 and roc_auc(_y4, -_y4 * 1.0) == 0.0 and average_precision(_y4, _y4 * 1.0) == 1.0)
_c = np.zeros(10); _yc = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
check("constant scores: AUC 0.5 and AP = base rate", lambda: roc_auc(_yc, _c) == 0.5 and abs(average_precision(_yc, _c) - 0.2) < 1e-9, "group ties!")
_r = np.random.default_rng(0)
_yr = (_r.random(4000) < 0.15).astype(int); _sr = _r.random(4000)
check("random scores: AUC near 0.5, AP near base rate", lambda: abs(roc_auc(_yr, _sr) - 0.5) < 0.04 and abs(average_precision(_yr, _sr) - _yr.mean()) < 0.04)
check("AUC is invariant to a monotone transform", lambda: abs(roc_auc(_yr, _sr) - roc_auc(_yr, _sr ** 3)) < 1e-12)
# %% hint
AUC: P = y.sum(), N = len(y) - P, ranks = rankdata(s); (ranks[y == 1].sum() - P*(P+1)/2) / (P*N).
# %% hint
AP: order = np.argsort(-s, kind="stable"); tp = cumsum(y[order]); fp = cumsum(1 - y[order]); keep = np.r_[np.diff(s[order]) != 0, True]; precision = tp/(tp+fp); recall = tp/P at the kept rows; AP = sum(diff([0, recall]) * precision).
# %% solution
def roc_auc(y, s):
    P = y.sum(); N = len(y) - P
    r = rankdata(s)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))

def average_precision(y, s):
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    keep = np.r_[np.diff(ss) != 0, True]
    tp, fp = tp[keep], fp[keep]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
# %% md Leak-proof cross-validation
Now the function that ties the stage together. `cv_scores` loops over folds; **inside** each fold it fits a fresh `Preprocessor` on the training rows, fits a fresh model, and predicts the validation rows. It returns the out-of-fold (OOF) predictions for every dev row plus the per-fold PR-AUC.

A model here is any object built by `make_model()` with `fit(Z, y)` and `predict_proba(Z)` (a vector of probabilities of churn).
# %% task Task 5: fold-safe cross-validation
# Use the global name Preprocessor (the checks swap it for a spy to verify you fit inside the fold).
def cv_scores(make_model, data, folds):
    # returns (oof, scores): oof has one probability per row of data, scores is the list of per-fold average_precision
    raise NotImplementedError   # TODO
# %% check
class _Mean:
    def fit(self, Z, y):
        self.p = y.mean(); return self
    def predict_proba(self, Z):
        return np.full(len(Z), self.p)
_Fd = stratified_kfold(dev["y"], 4, 0)
_oof, _sc = cv_scores(_Mean, dev, _Fd)
check("one OOF prediction per row, none left unset", lambda: _oof.shape == (len(dev["y"]),) and not np.isnan(_oof).any())
check("one score per fold", lambda: len(_sc) == 4)
_log = []
_Orig = Preprocessor
class _Spy(_Orig):
    def fit(self, Xnum, Xcat):
        _log.append(len(Xnum)); return super().fit(Xnum, Xcat)
Preprocessor = _Spy
try:
    cv_scores(_Mean, dev, _Fd)
finally:
    Preprocessor = _Orig
check("a Preprocessor is fitted on exactly the training rows of every fold", lambda: _log == [len(t) for t, _ in _Fd], "fit inside the loop, on data[train]")
class _Cap:
    seen = []
    def fit(self, Z, y):
        _Cap.seen.append((len(Z), Z[:, 0].mean())); return self
    def predict_proba(self, Z):
        return np.full(len(Z), 0.5)
cv_scores(_Cap, dev, _Fd)
check("training matrix of each fold is standardised on its own rows", lambda: all(abs(m) < 1e-6 for _, m in _Cap.seen[-4:]))
check("OOF of the prior model equals each fold's training base rate", lambda: all(abs(_oof[v][0] - dev["y"][t].mean()) < 1e-12 for t, v in _Fd))
# %% hint
for tr, va in folds: pre = Preprocessor().fit(data["num"][tr], data["cat"][tr]); Ztr = pre.transform(...); model = make_model().fit(Ztr, y[tr]); oof[va] = model.predict_proba(pre.transform(val rows)).
# %% solution
def cv_scores(make_model, data, folds):
    y = data["y"]
    oof = np.full(len(y), np.nan)
    scores = []
    for tr, va in folds:
        pre = Preprocessor().fit(data["num"][tr], data["cat"][tr])
        Ztr = pre.transform(data["num"][tr], data["cat"][tr])
        Zva = pre.transform(data["num"][va], data["cat"][va])
        model = make_model().fit(Ztr, y[tr])
        oof[va] = model.predict_proba(Zva)
        scores.append(average_precision(y[va], oof[va]))
    return oof, scores
# %% md Baselines
Two baselines every model must beat: the **prior** (predict the training base rate for everyone: PR-AUC equals the base rate) and a **one-feature heuristic** (rank by number of support tickets, standardised column 2 of the matrix).
# %% code
class PriorModel:
    def fit(self, Z, y):
        self.p = float(y.mean()); return self
    def predict_proba(self, Z):
        return np.full(len(Z), self.p)

class Heuristic:
    """Score = standardised value of one column (here: tickets, column 2), squashed to (0, 1)."""
    def __init__(self, col=2, sign=1.0):
        self.col, self.sign = col, sign
    def fit(self, Z, y):
        return self
    def predict_proba(self, Z):
        return sigmoid(self.sign * Z[:, self.col])

FOLDS = stratified_kfold(dev["y"], 4, seed=11)
print("model            PR-AUC per fold (4-fold)                mean   ROC-AUC")
for name, mk in [("prior", PriorModel), ("tickets rule", Heuristic)]:
    oof, sc = cv_scores(mk, dev, FOLDS)
    print(f"{name:16s} {np.round(sc, 3)}  {np.mean(sc):.3f}  {roc_auc(dev['y'], oof):.3f}")
print("base rate", dev["y"].mean().round(3))
# %% md Stage 3 summary
PR-AUC of the prior equals the base rate: that is the floor. Any real model should clear the tickets rule by a clear margin. Both baselines were scored with the same fold-safe loop you will use for everything else.
# %% report
Why do we print PR-AUC next to the base rate rather than quoting PR-AUC alone? What would ROC-AUC hide here?
# %% md Stage 4: Models
Two models, both written by you.

**Logistic regression** with L2 regularisation and an optional positive-class weight. Objective (weights $s_i = $ `pos_weight` for churners and 1 otherwise, bias not penalised):

$$
J(w,b) = \frac{\sum_i s_i\,\ell_i}{\sum_i s_i} + \frac{\lambda}{2}\lVert w\rVert^2,\qquad \ell_i = -y_i\log p_i - (1-y_i)\log(1-p_i)
$$

with gradient $\nabla_w J = Z^\top\!\big(s\odot(p-y)\big)/\sum s + \lambda w$ and $\partial_b J = \sum s\odot(p-y)/\sum s$. Full-batch gradient descent is enough on standardised inputs.
# %% task Task 6: logistic regression
class LogReg:
    def __init__(self, l2=1.0, iters=300, lr=0.5, pos_weight=1.0):
        self.l2, self.iters, self.lr, self.pos_weight = l2, iters, lr, pos_weight

    def fit(self, Z, y):
        # gradient descent from w = 0, b = 0 using the gradients above
        raise NotImplementedError   # TODO
        return self

    def predict_proba(self, Z):
        raise NotImplementedError   # TODO
# %% check
_r6 = np.random.default_rng(3)
_Z6 = _r6.normal(size=(3000, 2)); _w6 = np.array([2.0, -1.0])
_y6 = (_r6.random(3000) < sigmoid(_Z6 @ _w6 - 1.0)).astype(int)
_m6 = LogReg(l2=1e-3, iters=500).fit(_Z6, _y6)
check("learned weights point the right way", lambda: _m6.w[0] > 1.0 and _m6.w[1] < -0.4 and _m6.b < -0.4, "store self.w (d,) and self.b")
check("probabilities lie in (0, 1) with right shape", lambda: _m6.predict_proba(_Z6).shape == (3000,) and 0 < _m6.predict_proba(_Z6).min() and _m6.predict_proba(_Z6).max() < 1)
check("larger l2 shrinks the weights", lambda: np.linalg.norm(LogReg(l2=1.0).fit(_Z6, _y6).w) < np.linalg.norm(LogReg(l2=0.01).fit(_Z6, _y6).w))
check("pos_weight raises the average predicted probability", lambda: LogReg(pos_weight=5.0).fit(_Z6, _y6).predict_proba(_Z6).mean() > LogReg(pos_weight=1.0).fit(_Z6, _y6).predict_proba(_Z6).mean() + 0.05)
check("beats chance on held-out rows", lambda: roc_auc(_y6[2000:], _m6.predict_proba(_Z6[2000:])) > 0.8)
check("unregularised fit is calibrated on average", lambda: abs(LogReg(l2=1e-4, iters=600).fit(_Z6, _y6).predict_proba(_Z6).mean() - _y6.mean()) < 0.02)
# %% hint
Inside the loop: p = sigmoid(Z @ w + b); sw = np.where(y == 1, pos_weight, 1.0); r = sw * (p - y); gw = Z.T @ r / sw.sum() + l2 * w; gb = r.sum() / sw.sum().
# %% solution
class LogReg:
    def __init__(self, l2=1.0, iters=300, lr=0.5, pos_weight=1.0):
        self.l2, self.iters, self.lr, self.pos_weight = l2, iters, lr, pos_weight

    def fit(self, Z, y):
        n, d = Z.shape
        w = np.zeros(d); b = 0.0
        sw = np.where(y == 1, self.pos_weight, 1.0)
        for _ in range(self.iters):
            p = sigmoid(Z @ w + b)
            r = sw * (p - y)
            w = w - self.lr * (Z.T @ r / sw.sum() + self.l2 * w)
            b = b - self.lr * r.sum() / sw.sum()
        self.w, self.b = w, b
        return self

    def predict_proba(self, Z):
        return sigmoid(Z @ self.w + self.b)
# %% md Boosted stumps
**Gradient boosting** adds small trees one at a time, each fitted to the current errors. For logistic loss, with current raw score $F_i$ and $p_i=\sigma(F_i)$, each row has gradient $g_i = p_i - y_i$ and Hessian $h_i = p_i(1-p_i)$. A leaf holding rows $S$ gets the Newton value $w = -\sum_S g/(\sum_S h+\lambda)$, and a split into left/right is scored by

$$
\text{gain} = \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}.
$$

We use depth-1 trees (**stumps**) on pre-binned features: each column is cut into `nb` quantile bins, "bin $\le b$" goes left. Your job is the core of the algorithm, `best_stump`: one pass with `np.bincount` over all features at once, then cumulative sums to evaluate every split at once.

`B` has shape `(n, d)` with integer bins in `0..nb-1`. Return `(j, b, w_left, w_right)`.
# %% task Task 7: the best stump
def best_stump(B, g, h, nb, lam):
    # sums of g and h per (feature, bin): index = B + j*nb  ->  bincount with weights, reshape (d, nb)
    # left side of split "bin <= b" = cumulative sum over bins; only b = 0..nb-2 are valid splits
    raise NotImplementedError   # TODO
# %% check
def _brute(B, g, h, nb, lam):
    best = (-1e18, None)
    for j in range(B.shape[1]):
        for b in range(nb - 1):
            L = B[:, j] <= b
            GL, HL, GR, HR = g[L].sum(), h[L].sum(), g[~L].sum(), h[~L].sum()
            gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam)
            if gain > best[0] + 1e-12:
                best = (gain, (j, b, -GL / (HL + lam), -GR / (HR + lam)))
    return best[1]
_r7 = np.random.default_rng(5)
_B7 = _r7.integers(0, 6, (300, 4)); _g7 = _r7.normal(size=300); _h7 = _r7.uniform(0.05, 0.25, 300)
_got = best_stump(_B7, _g7, _h7, 6, 1.0); _exp = _brute(_B7, _g7, _h7, 6, 1.0)
check("matches a brute-force search (feature and bin)", lambda: tuple(_got[:2]) == tuple(_exp[:2]), "gain per (feature, split)")
check("matches brute-force leaf values", lambda: np.allclose(_got[2:], _exp[2:]))
_g8 = np.where(_B7[:, 2] <= 2, 0.4, -0.4) + 0.0 * _g7
_s8 = best_stump(_B7, _g8, _h7, 6, 1.0)
check("finds the planted feature and cut", lambda: _s8[0] == 2 and _s8[1] == 2)
check("leaf values oppose the gradient sign", lambda: _s8[2] < 0 < _s8[3], "w = -G/(H+lam)")
_big = best_stump(_B7, _g8, _h7, 6, 1e6)
check("a huge lambda shrinks the leaf values", lambda: abs(_big[2]) < 0.01 * abs(_s8[2]))
# %% hint
off = B + np.arange(d)*nb; G = np.bincount(off.ravel(), weights=np.broadcast_to(g[:, None], B.shape).ravel(), minlength=d*nb).reshape(d, nb); same for H.
# %% hint
GL = np.cumsum(G, 1)[:, :-1]; HL likewise; GR = G.sum(1, keepdims=True) - GL (same for H). gain array has shape (d, nb-1); np.unravel_index(argmax) gives (j, b).
# %% solution
def best_stump(B, g, h, nb, lam):
    n, d = B.shape
    off = (B + np.arange(d) * nb).ravel()
    G = np.bincount(off, weights=np.broadcast_to(g[:, None], B.shape).ravel(), minlength=d * nb).reshape(d, nb)
    H = np.bincount(off, weights=np.broadcast_to(h[:, None], B.shape).ravel(), minlength=d * nb).reshape(d, nb)
    GL = np.cumsum(G, axis=1)[:, :-1]; HL = np.cumsum(H, axis=1)[:, :-1]
    Gt = G.sum(axis=1, keepdims=True); Ht = H.sum(axis=1, keepdims=True)
    GR, HR = Gt - GL, Ht - HL
    gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - Gt ** 2 / (Ht + lam)
    j, b = np.unravel_index(np.argmax(gain), gain.shape)
    return int(j), int(b), float(-GL[j, b] / (HL[j, b] + lam)), float(-GR[j, b] / (HR[j, b] + lam))
# %% code
class GBStumps:
    """Gradient-boosted logistic stumps built on your best_stump."""
    def __init__(self, rounds=40, lr=0.3, lam=1.0, nb=16):
        self.rounds, self.lr, self.lam, self.nb = rounds, lr, lam, nb

    def _bin(self, Z):
        return (Z[:, :, None] > self.edges[None, :, :]).sum(axis=2)

    def fit(self, Z, y):
        qs = np.linspace(0, 1, self.nb + 1)[1:-1]
        self.edges = np.quantile(Z, qs, axis=0).T            # (d, nb-1)
        B = self._bin(Z)
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.f0 = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.f0)
        self.stumps = []
        for _ in range(self.rounds):
            p = sigmoid(F)
            g, h = p - y, p * (1 - p)
            j, b, wl, wr = best_stump(B, g, h, self.nb, self.lam)
            F = F + self.lr * np.where(B[:, j] <= b, wl, wr)
            self.stumps.append((j, b, wl, wr))
        return self

    def predict_proba(self, Z):
        B = self._bin(Z)
        F = np.full(len(Z), self.f0)
        for j, b, wl, wr in self.stumps:
            F = F + self.lr * np.where(B[:, j] <= b, wl, wr)
        return sigmoid(F)

print("model            PR-AUC per fold (4-fold)                mean   ROC-AUC   [ALL 5 numeric columns]")
for name, mk in [("logreg", lambda: LogReg(l2=0.01)), ("boosted stumps", lambda: GBStumps(rounds=40))]:
    oof, sc = cv_scores(mk, dev, FOLDS)
    print(f"{name:16s} {np.round(sc, 3)}  {np.mean(sc):.3f}  {roc_auc(dev['y'], oof):.3f}")
# %% md Stage 4 summary
Both models crush the baselines. Pause: the PR-AUC you just saw is far higher than anything in a real churn problem has a right to be. Great numbers on a first try are a symptom to investigate, not a result to report.
# %% report
Before reading on, write down two hypotheses for why the scores are so high, and one experiment that would distinguish them.
# %% md Stage 5: Leakage hunt
A cheap first test for a leak is to score **each feature alone**. A legitimate feature is usually weak by itself (AUC 0.55 to 0.75). A feature that alone reaches 0.95 is either a miracle or recorded after the outcome. For each numeric column: fill NaN with the column median, compute `roc_auc(y, x)`, and report $\max(\mathrm{AUC},1-\mathrm{AUC})$ so the direction of the relation does not matter.
# %% task Task 8: single-feature leak scan
def leak_scan(Xnum, y):
    # returns one number per column: max(auc, 1 - auc) using roc_auc(y, filled column)
    raise NotImplementedError   # TODO
# %% check
_r8 = np.random.default_rng(8)
_y8 = (_r8.random(2000) < 0.2).astype(int)
_X8 = np.column_stack([_r8.normal(size=2000), np.where(_y8 == 1, 5.0, 0.0) + _r8.normal(size=2000), -_y8 + 0.1 * _r8.normal(size=2000)])
_X8[_r8.random(_X8.shape) < 0.1] = np.nan
_sc8 = leak_scan(_X8, _y8)
check("one score per column in [0.5, 1]", lambda: _sc8.shape == (3,) and (_sc8 >= 0.5).all() and (_sc8 <= 1).all())
check("pure noise scores near 0.5", lambda: _sc8[0] < 0.57)
check("direction does not matter (negative relation still scores high)", lambda: _sc8[2] > 0.95, "use max(auc, 1 - auc)")
check("handles NaN without crashing and ranks the leak high", lambda: _sc8[1] > 0.95 and np.isfinite(_sc8).all())
check("does not modify its input", lambda: np.isnan(_X8).sum() > 0)
# %% hint
filled = np.where(np.isnan(col), np.nanmedian(col), col); a = roc_auc(y, filled); score = max(a, 1 - a).
# %% solution
def leak_scan(Xnum, y):
    out = []
    for j in range(Xnum.shape[1]):
        col = Xnum[:, j]
        filled = np.where(np.isnan(col), np.nanmedian(col), col)
        a = roc_auc(y, filled)
        out.append(max(a, 1 - a))
    return np.array(out)
# %% code
scan = leak_scan(dev["num"], dev["y"])
print("column               solo AUC")
for name, s in sorted(zip(NUM, scan), key=lambda t: -t[1]):
    print(f"{name:20s} {s:6.3f}  {'<-- SUSPICIOUS' if s > 0.85 else ''}")
LEAK = NUM[int(np.argmax(scan))]
j_leak = NUM.index(LEAK)
print("\nDropping:", LEAK)

def drop_num(data, j):
    keep = [c for c in range(data["num"].shape[1]) if c != j]
    return {"num": data["num"][:, keep], "cat": data["cat"], "y": data["y"]}

dev_c, test_c = drop_num(dev, j_leak), drop_num(test, j_leak)
NUM_C = [c for c in NUM if c != LEAK]
print("clean numeric columns:", NUM_C)
# %% code
print("model            PR-AUC per fold (4-fold)                mean   ROC-AUC   [LEAK REMOVED]")
RES = {}
for name, mk in [("prior", PriorModel), ("tickets rule", Heuristic), ("logreg", lambda: LogReg(l2=0.01)), ("boosted stumps", lambda: GBStumps(rounds=40))]:
    oof, sc = cv_scores(mk, dev_c, FOLDS)
    RES[name] = oof
    print(f"{name:16s} {np.round(sc, 3)}  {np.mean(sc):.3f}  {roc_auc(dev_c['y'], oof):.3f}")
# %% md Experiment: several seeds
One split can flatter a model by luck. We repeat the whole 4-fold cross-validation for 4 different fold seeds and report mean and standard deviation of the PR-AUC (computed on the pooled OOF predictions of each repeat).
# %% code
def repeated_ap(make_model, data, seeds, k=4):
    out = []
    for s in seeds:
        oof, _ = cv_scores(make_model, data, stratified_kfold(data["y"], k, s))
        out.append(average_precision(data["y"], oof))
    return np.array(out)

SEEDS = [101, 102, 103, 104]
print("model            PR-AUC mean +- std over 4 fold seeds")
for name, mk in [("tickets rule", Heuristic), ("logreg", lambda: LogReg(l2=0.01)), ("boosted stumps", lambda: GBStumps(rounds=40))]:
    r = repeated_ap(mk, dev_c, SEEDS)
    print(f"{name:16s} {r.mean():.3f} +- {r.std():.3f}")
# %% md Stage 5 summary
The scan flagged one column with a solo AUC near 1: it is logged during the retention call, after the customer already decided. Removing it dropped the scores to realistic values, and the multi-seed table shows whether the gap between models is larger than the seed-to-seed noise.
# %% report
Explain why `retention_call_min` is leakage even though it is a perfectly honest measurement. What question would you ask a data engineer to catch this kind of feature *before* modelling?
# %% md Stage 6: Calibration and decision
A ranking is not yet a decision. The retention team needs (1) probabilities they can trust, because the expected cost of calling depends on them, and (2) a threshold.

**Calibration.** Bin the predictions into equal-width probability bins; in each bin compare the mean predicted probability with the observed fraction of churners. The **expected calibration error** is $\mathrm{ECE} = \sum_b \frac{n_b}{n}\,|\overline{y}_b - \overline{p}_b|$ over non-empty bins.
# %% task Task 9: reliability table and ECE
def reliability_table(y, p, nbins=10):
    # rows (mean_p, frac_pos, count) for each NON-EMPTY equal-width bin of [0, 1]; the value p = 1.0 belongs in the last bin
    raise NotImplementedError   # TODO

def ece(y, p, nbins=10):
    raise NotImplementedError   # TODO
# %% check
_p9 = np.array([0.1, 0.1, 0.9, 0.9]); _y9 = np.array([0, 0, 1, 1])
check("ECE hand example = 0.1", lambda: abs(ece(_y9, _p9) - 0.1) < 1e-9)
_t = reliability_table(_y9, _p9)
check("table has one row per non-empty bin with (mean_p, frac, count)", lambda: _t.shape == (2, 3) and _t[:, 2].sum() == 4 and np.allclose(_t[:, 0], [0.1, 0.9]) and np.allclose(_t[:, 1], [0, 1]))
_r9 = np.random.default_rng(9); _pp9 = _r9.random(20000); _yy9 = (_r9.random(20000) < _pp9).astype(int)
check("a calibrated model has tiny ECE", lambda: ece(_yy9, _pp9) < 0.02)
check("an over-confident model has a large ECE", lambda: ece(_yy9, np.clip(_pp9 * 1.6 - 0.3, 0, 1)) > 0.08)
check("p = 1.0 does not crash and lands in the last bin", lambda: reliability_table(np.array([1, 0]), np.array([1.0, 1.0]))[-1, 2] == 2)
# %% hint
bins = np.minimum((p * nbins).astype(int), nbins - 1); loop over b in range(nbins), skip empty ones.
# %% solution
def reliability_table(y, p, nbins=10):
    bins = np.minimum((p * nbins).astype(int), nbins - 1)
    rows = []
    for b in range(nbins):
        m = bins == b
        if m.any():
            rows.append((p[m].mean(), y[m].mean(), m.sum()))
    return np.array(rows, dtype=float)

def ece(y, p, nbins=10):
    t = reliability_table(y, p, nbins)
    return float(np.sum(t[:, 2] / len(y) * np.abs(t[:, 1] - t[:, 0])))
# %% code
yd = dev_c["y"]
oof_w, _ = cv_scores(lambda: LogReg(l2=0.01, pos_weight=4.0), dev_c, FOLDS)
print("model                       ECE     Brier   PR-AUC")
for name, o in [("logreg", RES["logreg"]), ("boosted stumps", RES["boosted stumps"]), ("logreg pos_weight=4", oof_w)]:
    print(f"{name:26s} {ece(yd, o):.3f}   {np.mean((o - yd) ** 2):.3f}   {average_precision(yd, o):.3f}")

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

# Platt scaling: a 1-D logistic regression on the OOF logit (for honesty it should be cross-fitted; here we keep it simple)
platt = LogReg(l2=0.0, iters=400).fit(((logit(oof_w) - logit(oof_w).mean()) / logit(oof_w).std())[:, None], yd)
z = ((logit(oof_w) - logit(oof_w).mean()) / logit(oof_w).std())[:, None]
oof_w_cal = platt.predict_proba(z)
print(f"{'pos_weight=4 + Platt':26s} {ece(yd, oof_w_cal):.3f}   {np.mean((oof_w_cal - yd) ** 2):.3f}   {average_precision(yd, oof_w_cal):.3f}")
# %% md Costs and the threshold
A call costs `c_fp = 1` when the customer would have stayed (wasted offer) and a lost customer costs `c_fn = 10`. For a calibrated probability $p$, calling is worth it when $p\,c_{fn} > (1-p)\,c_{fp}$, i.e. $p > c_{fp}/(c_{fp}+c_{fn}) \approx 0.09$. With real, imperfect scores we do not trust the formula blindly: we scan every threshold on out-of-fold predictions and take the cheapest.
# %% task Task 10: cost-optimal threshold
def best_threshold(y, p, c_fp, c_fn):
    # predict churn when p >= t. Return (t, total_cost) minimising c_fp*FP + c_fn*FN over all cut points
    # (including "flag nobody"). Only cut between distinct scores.
    raise NotImplementedError   # TODO
# %% check
def _cost(y, p, t, a, b):
    pred = p >= t
    return a * ((pred == 1) & (y == 0)).sum() + b * ((pred == 0) & (y == 1)).sum()
_y10 = np.array([0, 0, 0, 1, 1, 0, 1, 0]); _p10 = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.5, 0.4, 0.05])
_t10, _c10 = best_threshold(_y10, _p10, 1.0, 1.0)
check("returned cost equals recomputed cost at the returned threshold", lambda: abs(_cost(_y10, _p10, _t10, 1.0, 1.0) - _c10) < 1e-9)
check("equal costs: best is 1 error here", lambda: _c10 == 1.0, "cut between 0.5 and 0.4 or similar")
_t, _c = best_threshold(_y10, _p10, 1.0, 100.0)
check("huge miss cost flags every positive", lambda: _cost(_y10, _p10, _t, 1.0, 100.0) == _c and _p10[_y10 == 1].min() >= _t)
_t, _c = best_threshold(_y10, _p10, 100.0, 1.0)
check("huge false-alarm cost: no false alarms", lambda: ((_p10 >= _t) & (_y10 == 0)).sum() == 0)
_r10 = np.random.default_rng(10); _pp10 = _r10.random(3000); _yy10 = (_r10.random(3000) < _pp10).astype(int)
_tt, _cc = best_threshold(_yy10, _pp10, 1.0, 10.0)
check("with calibrated p the optimum is near c_fp/(c_fp+c_fn) = 0.09", lambda: 0.03 < _tt < 0.2 and _cc <= min(_cost(_yy10, _pp10, 0.5, 1, 10), _cost(_yy10, _pp10, 0.0, 1, 10)))
# %% hint
Sort scores descending; flagging the top k rows costs c_fp*(k - tp_k) + c_fn*(P - tp_k). Only consider k = 0 and k at the end of a run of equal scores.
# %% solution
def best_threshold(y, p, c_fp, c_fn):
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys); k = np.arange(1, len(y) + 1)
    cost = c_fp * (k - tp) + c_fn * (y.sum() - tp)
    valid = np.r_[np.diff(ps) != 0, True]
    best_k = int(np.argmin(np.where(valid, cost, np.inf)))
    cost0 = c_fn * y.sum()                      # flag nobody
    if cost0 <= cost[best_k]:
        return float(ps[0] + 1e-9), float(cost0)
    return float(ps[best_k]), float(cost[best_k])
# %% code
oof_g = RES["boosted stumps"]
thr, cost_best = best_threshold(yd, oof_g, 1.0, 10.0)
n_pos = yd.sum()
print(f"flag nobody cost : {10.0 * n_pos:8.0f}")
print(f"flag everybody   : {1.0 * (len(yd) - n_pos):8.0f}")
print(f"threshold {thr:.3f}  cost {cost_best:8.0f}  (calls {int((oof_g >= thr).sum())}, recall {((oof_g >= thr) & (yd == 1)).sum() / n_pos:.2f})")
print(f"textbook threshold 0.091 would cost {1.0 * ((oof_g >= 0.0909) & (yd == 0)).sum() + 10.0 * ((oof_g < 0.0909) & (yd == 1)).sum():.0f}")
# %% md Stage 6 summary
Class weighting (`pos_weight=4`) left the ranking essentially untouched but wrecked calibration (ECE jumped by an order of magnitude); a monotone recalibration (Platt) repaired the probabilities without changing the ranking. The best threshold came from minimising cost on OOF predictions, not from 0.5, and it beat both trivial policies.
# %% report
Your threshold was tuned on the same OOF predictions it is judged on. What bias does that create, and how large do you think it is compared with the cost saving?
# %% md Stage 7: Tuning and slices
**Search with a budget.** Random search samples configurations independently, evaluates each with cross-validation, and stops after exactly `budget` evaluations. The sampler receives the generator, so reruns with the same seed are identical.
# %% task Task 11: budgeted random search
def random_search(sample, evaluate, budget, seed):
    # sample(rng) -> params dict; evaluate(params) -> score (higher is better)
    # returns (best_params, best_score, history) where history is a list of (params, score) in evaluation order
    raise NotImplementedError   # TODO
# %% check
_calls = []
def _ev(p):
    _calls.append(p); return -(p["x"] - 0.3) ** 2
_smp = lambda rng: {"x": float(rng.random())}
_bp, _bs, _h = random_search(_smp, _ev, 12, 0)
check("evaluates exactly `budget` configurations", lambda: len(_calls) == 12 and len(_h) == 12)
check("best score is the max of the history", lambda: _bs == max(s for _, s in _h) and _bp in [p for p, _ in _h])
_bp2, _bs2, _h2 = random_search(_smp, _ev, 12, 0)
check("same seed gives the same search", lambda: _bp2 == _bp and [p for p, _ in _h2] == [p for p, _ in _h])
_bp3, _, _ = random_search(_smp, _ev, 12, 1)
check("different seed explores different configs", lambda: _bp3 != _bp)
check("more budget never hurts when the first configs are shared", lambda: random_search(_smp, _ev, 40, 0)[1] >= _bs)
check("ties keep the earliest config", lambda: random_search(lambda r: {"x": float(r.integers(0, 3))}, lambda p: 1.0, 5, 0)[2][0][0] == random_search(lambda r: {"x": float(r.integers(0, 3))}, lambda p: 1.0, 5, 0)[0])
# %% hint
rng = np.random.default_rng(seed); loop budget times: params = sample(rng); score = evaluate(params); append; keep strict improvements only.
# %% solution
def random_search(sample, evaluate, budget, seed):
    rng = np.random.default_rng(seed)
    history, best_p, best_s = [], None, -np.inf
    for _ in range(budget):
        params = sample(rng)
        score = evaluate(params)
        history.append((params, score))
        if score > best_s:
            best_p, best_s = params, score
    return best_p, best_s, history
# %% code
def sample_gb(rng):
    return {"rounds": int(rng.choice([20, 40, 60])), "lr": float(np.exp(rng.uniform(np.log(0.05), np.log(0.5)))), "lam": float(rng.choice([0.1, 1.0, 10.0])), "nb": int(rng.choice([8, 16]))}

def eval_gb(params):
    oof, _ = cv_scores(lambda: GBStumps(**params), dev_c, FOLDS)
    return average_precision(dev_c["y"], oof)

BUDGET = 6
best_p, best_s, hist = random_search(sample_gb, eval_gb, BUDGET, seed=0)
print(f"{BUDGET} configurations evaluated (budget is fixed in advance)")
for p, s in sorted(hist, key=lambda t: -t[1]):
    print(f"  PR-AUC {s:.3f}  rounds={p['rounds']:3d} lr={p['lr']:.3f} lam={p['lam']:5.1f} nb={p['nb']}")
print("default GBStumps  :", round(average_precision(yd, RES['boosted stumps']), 3))
print("best of search    :", round(best_s, 3), "(optimistic: it was picked because it scored best)")
# %% md Slices
An average hides who the model fails. A **slice report** computes, for each group of a categorical column, the number of rows, the base rate, and recall and precision at the chosen threshold.
# %% task Task 12: slice report
def slice_report(y, p, groups, thr):
    # returns dict: group value -> {"n": int, "base_rate": float, "recall": float, "precision": float}
    # recall = TP/(TP+FN), precision = TP/(TP+FP); use float("nan") when the denominator is 0
    raise NotImplementedError   # TODO
# %% check
_y12 = np.array([1, 1, 0, 0, 1, 0]); _p12 = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3]); _g12 = np.array(["a", "a", "a", "b", "b", "b"])
_s12 = slice_report(_y12, _p12, _g12, 0.5)
check("per-group counts and base rates", lambda: _s12["a"]["n"] == 3 and abs(_s12["a"]["base_rate"] - 2 / 3) < 1e-9 and abs(_s12["b"]["base_rate"] - 1 / 3) < 1e-9)
check("recall and precision for group a", lambda: abs(_s12["a"]["recall"] - 0.5) < 1e-9 and abs(_s12["a"]["precision"] - 0.5) < 1e-9)
check("recall and precision for group b", lambda: _s12["b"]["recall"] == 1.0 and _s12["b"]["precision"] == 1.0)
_s13 = slice_report(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]), np.array(["x", "x", "y"]), 0.5)
check("zero denominators give NaN", lambda: np.isnan(_s13["x"]["recall"]) and np.isnan(_s13["x"]["precision"]))
check("single group equals the overall numbers", lambda: slice_report(_y12, _p12, np.zeros(6, int), 0.5)[0]["n"] == 6)
# %% hint
Loop over np.unique(groups); mask m = groups == v; pred = p[m] >= thr; tp = (pred & (y[m] == 1)).sum().
# %% solution
def slice_report(y, p, groups, thr):
    out = {}
    for v in np.unique(groups):
        m = groups == v
        yy, pred = y[m], p[m] >= thr
        tp = int((pred & (yy == 1)).sum()); fn = int((~pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
        out[v.item() if hasattr(v, "item") else v] = {"n": int(m.sum()), "base_rate": float(yy.mean()),
                                                       "recall": tp / (tp + fn) if tp + fn else float("nan"),
                                                       "precision": tp / (tp + fp) if tp + fp else float("nan")}
    return out
# %% code
oof_best, _ = cv_scores(lambda: GBStumps(**best_p), dev_c, FOLDS)
thr2, cost2 = best_threshold(yd, oof_best, 1.0, 10.0)
print(f"tuned model: PR-AUC {average_precision(yd, oof_best):.3f}, threshold {thr2:.3f}, cost {cost2:.0f}")
for j, name in enumerate(CAT):
    print(f"\nslice by {name}")
    print("  group     n    base   recall  precision  ROC-AUC")
    rep = slice_report(yd, oof_best, dev_c["cat"][:, j], thr2)
    for v, r in rep.items():
        m = dev_c["cat"][:, j] == v
        a = roc_auc(yd[m], oof_best[m]) if 0 < yd[m].sum() < m.sum() else float("nan")
        print(f"  {v or '<blank>':8s} {r['n']:5d} {r['base_rate']:6.3f} {r['recall']:7.2f} {r['precision']:9.2f} {a:8.3f}")
# %% md Failure analysis
The `channel` slice should stand out. The generator gave phone sign-ups a large hidden factor that no column records, so the best possible AUC for that slice is lower. Check: did the model fail because it is too simple, or because the information is simply not in the table? Compare the phone slice's ROC-AUC under the simple logistic regression and under the boosted model. If both are equally poor, a bigger model will not help; the fix is new data.
# %% code
m_phone = dev_c["cat"][:, 2] == "phone"
print("phone slice ROC-AUC  logreg: %.3f   boosted: %.3f" % (roc_auc(yd[m_phone], RES["logreg"][m_phone]), roc_auc(yd[m_phone], RES["boosted stumps"][m_phone])))
print("other slices ROC-AUC logreg: %.3f   boosted: %.3f" % (roc_auc(yd[~m_phone], RES["logreg"][~m_phone]), roc_auc(yd[~m_phone], RES["boosted stumps"][~m_phone])))
# %% md Stage 7 summary
A fixed budget of 6 random configurations did not beat the default model (the best of the search and the default are within about 0.002 PR-AUC of each other, smaller than the seed noise measured in Stage 5), so the honest conclusion is that tuning bought nothing here. The slice table shows a group where the model is weak for reasons a bigger model cannot fix.
# %% report
Which slice is worst, and is that a modelling problem or a data problem? What would you tell the retention team to do for that slice?
# %% md Stage 8: Final test and report
Only now do we open the test set, **once**. To make "once" mechanical, you will write a wrapper that raises an error on a second call. The final procedure: refit the preprocessor and the tuned model on all of `dev_c`, use the threshold chosen on dev OOF predictions, and compute PR-AUC, cost and a bootstrap 95% interval on the test set.
# %% task Task 13: a one-shot guard
def one_shot(fn):
    # return a wrapper that calls fn on the first call and raises RuntimeError on every later call
    raise NotImplementedError   # TODO
# %% check
_n = []
def _f(a, b=1):
    _n.append(1); return a + b
_w = one_shot(_f)
check("first call returns the function's value, with arguments", lambda: _w(2, b=5) == 7)
def _second():
    try:
        _w(1); return False
    except RuntimeError:
        return True
check("second call raises RuntimeError", _second)
check("the wrapped function ran exactly once", lambda: len(_n) == 1)
_w2 = one_shot(lambda: 42)
check("each wrapper has its own counter", lambda: _w2() == 42)
def _third():
    try:
        _w2(); return False
    except RuntimeError:
        return True
check("still raises on the third call", lambda: _third() and _third())
# %% hint
Keep a mutable flag in the closure, e.g. a one-element list, and set it before returning.
# %% solution
def one_shot(fn):
    used = []
    def wrapper(*args, **kwargs):
        if used:
            raise RuntimeError("the test set was already used")
        used.append(1)
        return fn(*args, **kwargs)
    return wrapper
# %% code
def bootstrap_ci(y, s, metric, B=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            vals.append(metric(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])

def final_evaluation():
    pre = Preprocessor().fit(dev_c["num"], dev_c["cat"])
    model = GBStumps(**best_p).fit(pre.transform(dev_c["num"], dev_c["cat"]), dev_c["y"])
    p = model.predict_proba(pre.transform(test_c["num"], test_c["cat"]))
    yt = test_c["y"]
    ap = average_precision(yt, p)
    lo, hi = bootstrap_ci(yt, p, average_precision)
    flag = p >= thr2
    cost = 1.0 * (flag & (yt == 0)).sum() + 10.0 * (~flag & (yt == 1)).sum()
    return {"p": p, "ap": ap, "ci": (lo, hi), "auc": roc_auc(yt, p), "cost": float(cost), "ece": ece(yt, p),
            "cost_nobody": 10.0 * yt.sum(), "cost_all": 1.0 * (yt == 0).sum(), "recall": float((flag & (yt == 1)).sum() / yt.sum())}

evaluate_once = one_shot(final_evaluation)
R = evaluate_once()
print(f"TEST PR-AUC {R['ap']:.3f}  95% CI [{R['ci'][0]:.3f}, {R['ci'][1]:.3f}]   ROC-AUC {R['auc']:.3f}   ECE {R['ece']:.3f}")
print(f"TEST cost at dev threshold {thr2:.3f}: {R['cost']:.0f}   (flag nobody {R['cost_nobody']:.0f}, flag all {R['cost_all']:.0f}), recall {R['recall']:.2f}")
print("dev OOF PR-AUC of the same configuration:", round(average_precision(yd, oof_best), 3))
try:
    evaluate_once()
except RuntimeError as e:
    print("second evaluation blocked:", e)
# %% md Stage 8 summary
You opened the test set once, got a number with an interval, and compared it with the dev estimate. A test result near the dev estimate (within the interval) is evidence the pipeline is leak-free; a much worse one would send you back to look for leakage, not to re-tune on the test set.
# %% report
Write the final report (about 150 words) for the retention team: the claim, the numbers with intervals, the threshold and its cost savings versus the trivial policies, the weak slice, what you removed and why, and what you still do not know.
# %% md Sources
Only material that was actually opened and read while writing this project:
- `content/authored/classical-ml/03-generalization`, `10-evaluation-metrics`, `11-features-imbalance-leakage`, `15-tuning-and-pipelines` (lesson section structure skimmed: protocols, metrics and calibration, leakage, pipelines that cannot leak, budgeted search). READ.
- `content/authored/llm-systems/03-embeddings-rag` and `04-evaluating-llms` (section outlines skimmed; not used directly in this project). READ.
- `docs/PROJECTS.md` and `docs/AGENT_BRIEF.md` (grammar and rules). READ.
Formulas for AUC by ranks, average precision, Newton boosting gain and ECE are standard results re-derived in the text; the data is synthetic and generated by the code above.
