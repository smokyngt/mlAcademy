# %% meta
{"id": "proj.reproduce", "title": "Reproduce and ablate: do warmup and normalisation make training robust to the learning rate?", "summary": "A miniature research project: turn a folk claim into a testable hypothesis, build a seeded experiment harness with a result store and confidence intervals, run a learning-rate sweep, a one-factor ablation and a 2x2 factorial over 8 seeds, test significance with paired tests and a multiple-comparison correction, analyse failures and write an honest reproduction report.", "level": "expert", "minutes": 150,
 "requires": ["sys.05", "trf.04"],
 "outcomes": ["Turn a vague claim into a falsifiable hypothesis with a pre-registered metric", "Build a deterministic harness: seeding, canonical result keys, a cache, aggregation with t and bootstrap confidence intervals", "Run a learning-rate sweep and quantify robustness instead of eyeballing a plot", "Separate main effects from interaction in a 2x2 factorial design", "Use paired tests and Holm correction, and say what a null or reversed result means", "Write a reproduction report with deviations, limitations and a verdict that the data can support"],
 "stages": ["Claim and plan", "Model and training loop", "Experiment harness", "Learning-rate sweep", "Ablation and factorial", "Significance", "Failure analysis", "Report"]}
# %% md Stage 1: the claim and the plan
## The claim

> *"Learning-rate warmup and normalisation layers make training robust to the choice of learning rate."*

This sentence is repeated in many papers and talks about transformers. It is also **not yet a hypothesis**: robust in what sense, compared with what, on which model? A reproduction starts by making the claim precise enough that data can contradict it.

**Our operational version.** Small network, small data, scaled-down in every direction (see "Deviations" at the end):

- Model: an MLP `2 -> 16 -> 16 -> 1` with ReLU, trained with Adam on the two-moons classification task.
- Manipulated factors: `norm` (parameter-free LayerNorm after each hidden linear layer: yes/no) and `warmup` (linear learning-rate warmup over 20 of 100 steps: yes/no).
- Learning rates: 6 values on a log grid from 0.01 to 3.
- **Primary metric (fixed before looking at any result): final test accuracy.** "Robust" means two things we can compute: the *worst* mean accuracy over the learning-rate grid, and the *width of the window* of learning rates (in decades) at which mean accuracy stays above 0.88.
- 8 seeds per cell. The seed controls the initial weights and the order of mini-batches; the dataset is fixed. Using the same seed for two configurations gives them the same initial weights and batch order, which makes **paired** comparisons possible.
- Predictions written before running: (1) normalisation widens the window; (2) warmup also helps at high learning rates; (3) their effects add up. We will see which survive.

Everything is NumPy. Each stage ends with a summary and a free-text report prompt.
# %% report
Before running anything: write what result would make you say the claim is *false* in this setting. If no result could, rewrite the claim.
# %% md Stage 2: model and training loop
## Normalisation, written by hand

For a batch `z` of shape `(N, D)`, parameter-free LayerNorm normalises each row: $y = (z-\mu)/\sigma$ with $\mu,\sigma^2$ the row mean and variance (plus $\varepsilon=10^{-5}$ inside the root). With upstream gradient `dy` the backward pass is
$$dz = \frac{1}{\sigma}\Big(dy - \overline{dy} - y\,\overline{dy\odot y}\Big),$$
where the overlines are row means. (Derivation: $y$ depends on $z_j$ directly, through $\mu$ and through $\sigma$; the last two terms project out the directions along the all-ones vector and along $y$ itself, which are the two things normalisation removes.)
# %% code
import json, math, time
import numpy as np
EPS_LN = 1e-5
# %% task LayerNorm forward and backward
# layernorm_forward(z) -> (y, cache);  layernorm_backward(dy, cache) -> dz
def layernorm_forward(z):
    raise NotImplementedError   # TODO

def layernorm_backward(dy, cache):
    raise NotImplementedError   # TODO
# %% check
_r = np.random.default_rng(0)
_z = _r.normal(2.0, 3.0, size=(6, 10)); _dy = _r.normal(size=(6, 10))
_y, _c = layernorm_forward(_z)
check("each row has mean 0 and variance ~1", lambda: np.allclose(_y.mean(1), 0, atol=1e-10) and np.allclose(_y.var(1), 1, atol=1e-4), "subtract the row mean, divide by sqrt(var + eps)")
def _fd():
    f = lambda zz: float((layernorm_forward(zz)[0] * _dy).sum())
    num = np.zeros_like(_z)
    for i in range(_z.shape[0]):
        for j in range(_z.shape[1]):
            e = np.zeros_like(_z); e[i, j] = 1e-6
            num[i, j] = (f(_z + e) - f(_z - e)) / 2e-6
    return np.allclose(num, layernorm_backward(_dy, _c), atol=1e-6)
check("backward matches finite differences", _fd, "dz = (dy - mean(dy) - y*mean(dy*y)) / sigma")
check("gradient is orthogonal to the ones vector and to y", lambda: np.allclose(layernorm_backward(_dy, _c).sum(1), 0, atol=1e-9) and np.allclose((layernorm_backward(_dy, _c) * _y).sum(1), 0, atol=1e-3), "normalisation removes those two directions")
check("invariant to shifting and scaling the input", lambda: np.allclose(layernorm_forward(5 * _z + 7)[0], _y, atol=1e-4), "that is why the layer tolerates large activations")
# %% hint
Keep `y` and `sigma = sqrt(var + eps)` in the cache; the backward needs both.
# %% solution
def layernorm_forward(z):
    mu = z.mean(1, keepdims=True)
    sigma = np.sqrt(z.var(1, keepdims=True) + EPS_LN)
    y = (z - mu) / sigma
    return y, (y, sigma)

def layernorm_backward(dy, cache):
    y, sigma = cache
    return (dy - dy.mean(1, keepdims=True) - y * (dy * y).mean(1, keepdims=True)) / sigma
# %% md Optimiser and schedule
Adam keeps running averages of the gradient `m` and of its square `v` and moves each parameter by about $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, where $\hat m=m/(1-\beta_1^t)$ and $\hat v=v/(1-\beta_2^t)$ are bias-corrected. A consequence worth remembering: on the **first** step $\hat m/\sqrt{\hat v}=\pm1$, so every parameter moves by about $\eta$ no matter how small or large its gradient is. At a large $\eta$ that is a big, blind jump, which is the folk argument for warmup: start with a small $\eta_t$ and ramp it up linearly, $\eta_t=\eta\cdot\min(1,(t+1)/w)$ for $t=0,1,\dots$ (and $\eta_t=\eta$ when $w=0$).
# %% task Warmup schedule and Adam update
# lr_at(t, peak, warmup) -> float, t is the 0-based step
# adam_init(theta) -> state dict with "m", "v" (zeros) and "t" = 0
# adam_update(state, theta, grad, lr, b1=0.9, b2=0.999, eps=1e-8) -> theta   (updates theta IN PLACE, returns it)
def lr_at(t, peak, warmup):
    raise NotImplementedError   # TODO

def adam_init(theta):
    raise NotImplementedError   # TODO

def adam_update(state, theta, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
    raise NotImplementedError   # TODO
# %% check
check("warmup ramps linearly to the peak and stays there", lambda: np.isclose(lr_at(0, 1.0, 10), 0.1) and np.isclose(lr_at(4, 1.0, 10), 0.5) and np.isclose(lr_at(9, 1.0, 10), 1.0) and lr_at(500, 1.0, 10) == 1.0, "peak * min(1, (t+1)/warmup)")
check("warmup=0 means a constant learning rate", lambda: all(lr_at(t, 0.3, 0) == 0.3 for t in range(5)), "avoid dividing by zero")
def _first():
    th = np.zeros(3); st = adam_init(th)
    out = adam_update(st, th, np.array([1e-3, -5.0, 200.0]), 0.1)
    return out is th and np.allclose(th, [-0.1, 0.1, -0.1], atol=1e-6) and st["t"] == 1
check("first Adam step moves every coordinate by ~lr whatever the gradient size", _first, "bias correction makes m_hat/sqrt(v_hat) = +-1 at t=1")
def _two():
    th = np.array([1.0]); st = adam_init(th)
    adam_update(st, th, np.array([2.0]), 0.1); adam_update(st, th, np.array([1.0]), 0.1)
    m = 0.9 * (0.1 * 2.0) + 0.1 * 1.0; v = 0.999 * (0.001 * 4.0) + 0.001 * 1.0
    th2 = 1.0 - 0.1 * 2.0 / (2.0 + 1e-8)
    th2 -= 0.1 * (m / (1 - 0.9 ** 2)) / (np.sqrt(v / (1 - 0.999 ** 2)) + 1e-8)
    return np.isclose(th[0], th2, atol=1e-9)
check("two steps match a hand computation", _two, "update m, v, increment t, then correct with 1-b^t")
def _quad():
    th = np.zeros(4); st = adam_init(th)
    for _ in range(400):
        adam_update(st, th, th - 3.0, 0.05)
    return np.allclose(th, 3.0, atol=0.1)
check("minimises a quadratic", _quad, "gradient of 0.5*||th-3||^2 is th-3")
# %% hint
Update `st["m"][:] = ...` in place and increment `st["t"]` BEFORE computing the corrections.
# %% solution
def lr_at(t, peak, warmup):
    if warmup <= 0:
        return peak
    return peak * min(1.0, (t + 1) / warmup)

def adam_init(theta):
    return {"m": np.zeros_like(theta), "v": np.zeros_like(theta), "t": 0}

def adam_update(state, theta, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    state["m"][:] = b1 * state["m"] + (1 - b1) * grad
    state["v"][:] = b2 * state["v"] + (1 - b2) * grad * grad
    mh = state["m"] / (1 - b1 ** t)
    vh = state["v"] / (1 - b2 ** t)
    theta -= lr * mh / (np.sqrt(vh) + eps)
    return theta
# %% md The model, the data and one training run (given)
The code below uses your LayerNorm and Adam. Parameters live in **one flat vector** `theta` so that Adam is a handful of NumPy calls; `unpack` returns views of its pieces. The last cell of this stage checks the hand-written backward pass against finite differences, **before** any experiment is trusted: an unverified gradient would silently invalidate every number downstream.
# %% code
def make_moons(n, noise, rng):
    n1 = n // 2; n2 = n - n1
    t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, n2)
    X = np.r_[np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]]
    X = X + rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n1), np.ones(n2)]
    p = rng.permutation(n)
    return X[p], y[p]

_X, _y = make_moons(400, 0.25, np.random.default_rng(0))     # fixed dataset: the seed does NOT change the data
DATA = (_X[:250], _y[:250], _X[250:], _y[250:])
print("train", DATA[0].shape, "test", DATA[2].shape, "class balance", DATA[1].mean().round(2))

def sizes_for(hidden):
    return (2, hidden, hidden, 1)

def init_theta(sizes, rng):
    parts = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        parts += [rng.normal(0, np.sqrt(2.0 / a), a * b), np.zeros(b)]
    return np.concatenate(parts)

def unpack(theta, sizes):
    out, i = [], 0
    for a, b in zip(sizes[:-1], sizes[1:]):
        W = theta[i:i + a * b].reshape(a, b); i += a * b
        bias = theta[i:i + b]; i += b
        out.append((W, bias))
    return out

def forward_backward(theta, sizes, X, y, norm):
    layers = unpack(theta, sizes); h = X; cache = []
    for W, b in layers[:-1]:
        z = h @ W + b; c = None
        if norm:
            z, c = layernorm_forward(z)
        cache.append((h, c, z)); h = np.maximum(z, 0)
    W, b = layers[-1]
    z = (h @ W + b)[:, 0]
    loss = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))   # stable binary cross-entropy
    p = 0.5 * (1 + np.tanh(0.5 * z))
    grad = np.zeros_like(theta); gl = unpack(grad, sizes)
    d = ((p - y) / len(y))[:, None]
    gl[-1][0][:] = h.T @ d; gl[-1][1][:] = d.sum(0)
    dh = d @ W.T
    for i in range(len(layers) - 2, -1, -1):
        hin, c, zz = cache[i]; W, b = layers[i]
        dz = dh * (zz > 0)
        if norm:
            dz = layernorm_backward(dz, c)
        gl[i][0][:] = hin.T @ dz; gl[i][1][:] = dz.sum(0)
        if i > 0:
            dh = dz @ W.T
    return loss, grad, z

# gradient check (finite differences) for both architectures
for _norm in (False, True):
    _sz = sizes_for(16); _th = init_theta(_sz, np.random.default_rng(1))
    _l, _g, _ = forward_backward(_th, _sz, DATA[0][:32], DATA[1][:32], _norm)
    _idx = np.random.default_rng(2).choice(len(_th), 12, replace=False); _err = 0.0
    for _i in _idx:
        _e = np.zeros_like(_th); _e[_i] = 1e-6
        _num = (forward_backward(_th + _e, _sz, DATA[0][:32], DATA[1][:32], _norm)[0] - forward_backward(_th - _e, _sz, DATA[0][:32], DATA[1][:32], _norm)[0]) / 2e-6
        _err = max(_err, abs(_num - _g[_i]))
    print("norm =", _norm, " parameters", len(_th), " max gradient error", f"{_err:.1e}")
    assert _err < 1e-6, "backward pass is wrong"
# %% code
def train_run(config, seed):
    """One training run. All randomness comes from default_rng(seed): initial weights first, then mini-batch indices."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = DATA
    sizes = sizes_for(config["hidden"]); theta = init_theta(sizes, rng); st = adam_init(theta)
    losses, diverged = [], False
    with np.errstate(all="ignore"):
        for t in range(config["steps"]):
            idx = rng.integers(0, len(ytr), 32)
            loss, g, _ = forward_backward(theta, sizes, Xtr[idx], ytr[idx], config["norm"])
            if not np.isfinite(loss) or not np.all(np.isfinite(theta)):
                diverged = True; break
            adam_update(st, theta, g, lr_at(t, config["lr"], config["warmup"]))
            losses.append(float(loss))
        if not diverged:
            tl, _, z = forward_backward(theta, sizes, Xte, yte, config["norm"])
            if not np.isfinite(tl):
                diverged = True
    if diverged:
        return dict(test_acc=0.5, test_loss=float("inf"), diverged=True, losses=losses)
    return dict(test_acc=float(np.mean((z > 0) == yte)), test_loss=float(tl), diverged=False, losses=losses)

BASE = dict(norm=False, warmup=0, lr=0.1, steps=100, hidden=16)
def cfg(**kw):
    c = dict(BASE); c.update(kw); return c

t0 = time.time()
_a = train_run(cfg(lr=0.03), 0); _b = train_run(cfg(lr=0.03), 0)
print("one run: test acc", round(_a["test_acc"], 3), " identical when repeated:", _a == _b, f" ({time.time()-t0:.2f}s for two runs)")
# %% md Stage 2 summary
You have a verified model, an optimiser and a schedule, and a training function whose only source of randomness is the seed. Two sanity checks were passed before any science was done: finite-difference gradients, and bit-identical repetition of a run. Without them, a "result" could be a bug.
# %% report
Why is it important that the dataset is fixed and the seed only controls initialisation and batch order? What question would change if the seed also resampled the dataset?
# %% md Stage 3: the experiment harness
## Result store

An experiment is a function `(config, seed) -> result`. Running grids of them by hand in a notebook leads to lost results, accidentally different settings and re-runs that take hours. A **result store** gives each run a canonical key (the config serialised with sorted keys, plus the seed), runs it only if the key is absent and returns the cached result otherwise. Any later analysis calls `get_or_run` again and gets the same numbers instantly, so the notebook is a *query* over a deterministic function.
# %% task Result store with canonical keys
class ResultStore:
    def __init__(self, runner):
        self.runner = runner      # runner(config, seed) -> dict
        self.data = {}            # key -> result dict (with "config" and "seed" added)
        self.n_runs = 0           # how many times the runner was really called

    # key(config, seed) -> str : json with sort_keys=True, plus the seed. Dict order must not matter.
    def key(self, config, seed):
        raise NotImplementedError   # TODO

    # get_or_run(config, seed) -> result dict; stores a COPY with "config" (dict) and "seed" (int) added
    def get_or_run(self, config, seed):
        raise NotImplementedError   # TODO

    # select(**filters) -> list of stored results whose config matches every filter, sorted by (seed)
    def select(self, **filters):
        raise NotImplementedError   # TODO
# %% check
def _fake(config, seed):
    return dict(value=config["a"] * 10 + seed)
def _s1():
    s = ResultStore(_fake)
    r1 = s.get_or_run({"a": 1, "b": 2}, 3); r2 = s.get_or_run({"b": 2, "a": 1}, 3)
    return s.n_runs == 1 and r1 == r2 and r1["value"] == 13 and r1["seed"] == 3 and r1["config"] == {"a": 1, "b": 2}
check("same config in a different key order is one cached run", _s1, "json.dumps(config, sort_keys=True)")
def _s2():
    s = ResultStore(_fake)
    keys = {s.key({"a": 1, "b": 2}, 0), s.key({"a": 1, "b": 2}, 1), s.key({"a": 2, "b": 2}, 0)}
    return len(keys) == 3 and s.key({"a": 1, "b": 2}, 0) == s.key({"b": 2, "a": 1}, 0)
check("different seeds or configs give different keys", _s2, "the seed is part of the key")
def _s3():
    s = ResultStore(_fake)
    for seed in (4, 2, 3):
        s.get_or_run({"a": 1, "b": 0}, seed); s.get_or_run({"a": 2, "b": 0}, seed)
    got = s.select(a=1)
    return [r["seed"] for r in got] == [2, 3, 4] and all(r["config"]["a"] == 1 for r in got) and len(s.select(b=0)) == 6 and s.select(a=9) == []
check("select filters on config fields and sorts by seed", _s3, "compare r['config'][k] == v for every filter")
check("get_or_run does not call the runner again (counter stays 1)", lambda: (lambda s: (s.get_or_run({"a": 1}, 0), s.get_or_run({"a": 1}, 0), s.n_runs)[2] == 1)(ResultStore(_fake)), "check `key in self.data` first")
# %% hint
`res = dict(self.runner(config, seed)); res["config"] = dict(config); res["seed"] = seed`.
# %% solution
class ResultStore:
    def __init__(self, runner):
        self.runner = runner
        self.data = {}
        self.n_runs = 0

    def key(self, config, seed):
        return json.dumps(config, sort_keys=True) + f"|seed={seed}"

    def get_or_run(self, config, seed):
        k = self.key(config, seed)
        if k not in self.data:
            res = dict(self.runner(config, seed))
            res["config"] = dict(config); res["seed"] = seed
            self.data[k] = res
            self.n_runs += 1
        return self.data[k]

    def select(self, **filters):
        out = [r for r in self.data.values() if all(r["config"].get(k) == v for k, v in filters.items())]
        return sorted(out, key=lambda r: r["seed"])
# %% md Confidence intervals
Mean and standard deviation describe the seeds; a **confidence interval** describes our uncertainty about the true mean. For $n$ seeds with sample standard deviation $s$ (divide by $n-1$) the standard error is $s/\sqrt n$ and the 95% t-interval is $\bar x \pm t_{0.975,\,n-1}\, s/\sqrt n$. With $n=8$ the critical value is 2.365, not 1.96; using the normal value with a handful of seeds makes intervals too narrow. As a second opinion we also compute a **percentile bootstrap** interval: resample the $n$ values with replacement many times, take the mean each time and read the 2.5% and 97.5% quantiles.

The Student-t distribution function is given below (numerical integration of its density; no SciPy in the sandbox).
# %% code
def t_cdf(x, df):
    """CDF of Student's t via Simpson integration of the density on [0, |x|]."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    a = abs(float(x))
    if a == 0:
        return 0.5
    n = 2000
    u = np.linspace(0, a, n + 1)
    f = c * (1 + u * u / df) ** (-(df + 1) / 2)
    h = a / n
    area = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    return 0.5 + math.copysign(area, x)

def t_ppf(q, df):
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

print("t_{0.975, 7} =", round(t_ppf(0.975, 7), 4), "(textbook 2.3646)    t_{0.975, 4} =", round(t_ppf(0.975, 4), 4), "(textbook 2.7764)")
assert abs(t_ppf(0.975, 7) - 2.3646) < 1e-3 and abs(t_ppf(0.975, 4) - 2.7764) < 1e-3
# %% task Aggregation with confidence intervals
# summarize(values, level=0.95, n_boot=2000, seed=0) -> dict with
#   n, mean, std (ddof=1), sem, ci_t (lo, hi) using t_ppf, ci_boot (lo, hi) percentile bootstrap with default_rng(seed)
def summarize(values, level=0.95, n_boot=2000, seed=0):
    raise NotImplementedError   # TODO
# %% check
_s = summarize(np.arange(1.0, 9.0))
check("mean, sample std and standard error of 1..8", lambda: _s["n"] == 8 and np.isclose(_s["mean"], 4.5) and np.isclose(_s["std"], 2.4494897, atol=1e-6) and np.isclose(_s["sem"], 0.8660254, atol=1e-6), "std uses ddof=1; sem = std / sqrt(n)")
check("t interval of 1..8 is 4.5 +- 2.048", lambda: np.allclose(_s["ci_t"], (2.4518, 6.5482), atol=2e-3), "critical value t_{0.975,7} = 2.365")
def _boot():
    x = np.array([0.91, 0.93, 0.88, 0.95, 0.90, 0.87, 0.94, 0.92])
    a = summarize(x, seed=0)["ci_boot"]; b = summarize(x, seed=0)["ci_boot"]; c = summarize(x, seed=1)["ci_boot"]
    t = summarize(x)["ci_t"]
    return a == b and a != c and abs(a[0] - t[0]) < 0.01 and abs(a[1] - t[1]) < 0.01
check("bootstrap interval is reproducible for a seed, differs across seeds and is close to the t interval", _boot, "rng = default_rng(seed); resample n values n_boot times")
check("a constant sample has zero spread and a zero-width interval", lambda: summarize([0.9] * 8)["std"] == 0 and summarize([0.9] * 8)["ci_t"] == (0.9, 0.9) and np.allclose(summarize([0.9] * 8)["ci_boot"], (0.9, 0.9)), "no division by zero problems")
check("a 99% interval is wider than a 95% interval", lambda: np.diff(summarize(np.arange(8.0), level=0.99)["ci_t"])[0] > np.diff(summarize(np.arange(8.0), level=0.95)["ci_t"])[0], "level changes the t quantile")
# %% hint
`q = 1 - (1 - level) / 2`; `half = t_ppf(q, n - 1) * sem`. Bootstrap: `idx = rng.integers(0, n, (n_boot, n))`, `means = x[idx].mean(1)`, `np.percentile(means, [100*(1-level)/2, 100*q])`.
# %% solution
def summarize(values, level=0.95, n_boot=2000, seed=0):
    x = np.asarray(values, float)
    n = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    q = 1 - (1 - level) / 2
    half = t_ppf(q, n - 1) * sem if n > 1 else 0.0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, (n_boot, n))].mean(1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * q])
    return dict(n=n, mean=mean, std=std, sem=sem, ci_t=(mean - half, mean + half), ci_boot=(float(lo), float(hi)))
# %% md Running the harness
`STORE` wraps `train_run`. The helper `acc_table` collects per-seed accuracies for a list of configs; it always iterates the same seeds in the same order, which is what makes pairing by index valid.
# %% code
STORE = ResultStore(train_run)
SEEDS = list(range(8))
LRS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RECIPES = {"plain": dict(norm=False, warmup=0), "norm": dict(norm=True, warmup=0),
           "warmup": dict(norm=False, warmup=20), "norm+warmup": dict(norm=True, warmup=20)}

def accs(config, metric="test_acc"):
    """Per-seed values of a metric for one config, in SEEDS order."""
    return np.array([STORE.get_or_run(config, s)[metric] for s in SEEDS])

t0 = time.time()
a = accs(cfg(lr=0.1)); b = accs(cfg(lr=0.1))
print("8 seeds, plain, lr=0.1:", a.round(3), "| runner calls:", STORE.n_runs, "(second call was free)")
s = summarize(a)
print(f"mean {s['mean']:.3f}  std {s['std']:.3f}  95% t-CI [{s['ci_t'][0]:.3f}, {s['ci_t'][1]:.3f}]  bootstrap [{s['ci_boot'][0]:.3f}, {s['ci_boot'][1]:.3f}]  ({time.time()-t0:.1f}s)")
# %% md Stage 3 summary
The harness gives you determinism (seeded rng, canonical keys), memoisation (the store) and honest uncertainty (t and bootstrap intervals). Every number in the remaining stages is a query against `STORE`.
# %% report
Why is a standard deviation not the same thing as a confidence interval, and which one should error bars in a paper show? Which one shrinks as you add seeds?
# %% md Stage 4: the learning-rate sweep (reproduction)
## 4 recipes x 6 learning rates x 8 seeds = 192 runs

We run the whole grid first, in four cells (one per recipe) so each stays fast. Nothing is tuned per recipe: every arm gets the same step budget, batch size, seeds and learning-rate grid.
# %% code
def run_recipe(name):
    t0 = time.time()
    for lr in LRS:
        for s in SEEDS:
            STORE.get_or_run(cfg(lr=lr, **RECIPES[name]), s)
    print(f"{name:12s} done in {time.time()-t0:.1f}s, total runs so far {STORE.n_runs}")
run_recipe("plain")
# %% code
run_recipe("norm")
# %% code
run_recipe("warmup")
# %% code
run_recipe("norm+warmup")
# %% md Quantifying robustness
Eyeballing a table is not a protocol. We define, for one recipe, `acc_by_lr` (learning rate -> list of per-seed accuracies) and then:

- `means[lr]`: mean accuracy at each learning rate;
- `worst`: the minimum of those means (robustness in the worst case);
- `window_lrs`: the learning rates whose mean accuracy is at least `thresh`;
- `decades`: $\log_{10}(\max\text{ window lr} / \min\text{ window lr})$ (0 when the window has fewer than two entries).
# %% task Robustness metrics of a sweep
# robustness(acc_by_lr, thresh) -> dict(means={lr: mean}, best, worst, window_lrs=[sorted lrs], decades)
def robustness(acc_by_lr, thresh):
    raise NotImplementedError   # TODO
# %% check
_toy = {0.01: [0.90, 0.92], 0.1: [0.95, 0.95], 1.0: [0.50, 0.60], 10.0: [0.95, 0.97]}
_rt = robustness(_toy, 0.90)
check("means, best and worst over the grid", lambda: np.isclose(_rt["means"][0.01], 0.91) and np.isclose(_rt["best"], 0.96) and np.isclose(_rt["worst"], 0.55), "mean per lr first, then min/max over lrs")
check("window lists lrs with mean >= thresh in sorted order", lambda: _rt["window_lrs"] == [0.01, 0.1, 10.0], "use >=")
check("decades = log10(max/min) of the window", lambda: np.isclose(_rt["decades"], 3.0) and robustness(_toy, 0.99)["decades"] == 0 and robustness(_toy, 0.99)["window_lrs"] == [], "zero for fewer than two members")
def _real():
    by = {r: {lr: list(accs(cfg(lr=lr, **RECIPES[r]))) for lr in LRS} for r in ("plain", "norm")}
    a, b = robustness(by["plain"], 0.88), robustness(by["norm"], 0.88)
    return b["worst"] > a["worst"] and b["decades"] >= a["decades"]
check("on the real sweep, the normalised network has a better worst case and a window at least as wide", _real, "this is the core measurement of the claim")
# %% hint
`acc_by_lr` keys are floats; sort them before building `window_lrs`. `decades = np.log10(max(w) / min(w))`.
# %% solution
def robustness(acc_by_lr, thresh):
    means = {lr: float(np.mean(v)) for lr, v in acc_by_lr.items()}
    window = sorted(lr for lr, m in means.items() if m >= thresh)
    dec = float(np.log10(window[-1] / window[0])) if len(window) >= 2 else 0.0
    return dict(means=means, best=max(means.values()), worst=min(means.values()), window_lrs=window, decades=dec)
# %% code
def sweep_table(thresh=0.88):
    print(f"{'recipe':12s} | " + " ".join(f"lr={lr:<5g}" for lr in LRS) + " | worst  window(decades)")
    out = {}
    for name in RECIPES:
        by = {lr: list(accs(cfg(lr=lr, **RECIPES[name]))) for lr in LRS}
        rb = robustness(by, thresh); out[name] = rb
        cells = []
        for lr in LRS:
            s = summarize(by[lr])
            cells.append(f"{s['mean']:.3f}+-{s['std']:.2f}")
        print(f"{name:12s} | " + " ".join(f"{c:<9s}" for c in cells) + f" | {rb['worst']:.3f}  {len(rb['window_lrs'])} lrs, {rb['decades']:.1f}")
    return out
ROB = sweep_table()
# %% md Stage 4 summary
Read the table with the window and worst-case columns, not with your eyes alone. Mean +- std over 8 seeds: the std is the seed-to-seed spread of a single run, which is what a practitioner who trains once will experience. Do **not** yet claim which of `norm` and `warmup` is responsible for the changes: a sweep confounds nothing, but it also isolates nothing. The next stage does that.
# %% report
Summarise the sweep in three sentences: which recipes tolerate lr=1 and lr=3, which learning rates are best for each recipe, and did anything surprise you relative to your prediction in Stage 1?
# %% md Stage 5: ablation and factorial design
## One-factor ablation: how long should the warmup be?

Change one thing and keep the rest fixed: warmup length in {0, 5, 10, 20, 40} steps at the stress learning rate 1.0, once without and once with normalisation. Same seeds everywhere, so each row is paired with the 0-step row.
# %% code
STRESS = 1.0
WARMS = [0, 5, 10, 20, 40]
t0 = time.time()
abl = {}
print(f"{'norm':>5} {'warmup':>6} | {'test acc mean +- std':>22} | 95% t-CI            | mean diff vs warmup=0")
for norm in (False, True):
    base = accs(cfg(lr=STRESS, norm=norm, warmup=0))
    for w in WARMS:
        v = accs(cfg(lr=STRESS, norm=norm, warmup=w)); abl[(norm, w)] = v
        s = summarize(v)
        print(f"{str(norm):>5} {w:>6} | {s['mean']:>10.3f} +- {s['std']:.3f}      | [{s['ci_t'][0]:.3f}, {s['ci_t'][1]:.3f}] | {np.mean(v-base):+.3f}")
print(f"({time.time()-t0:.1f}s, runner calls {STORE.n_runs})")
# %% md 2x2 factorial
Ablating one factor at a time cannot tell whether the factors interact. A **factorial design** runs all four combinations. For per-seed results $c_{nw}$ with $n$ = norm (0/1) and $w$ = warmup (0/1) (all paired by seed):

- norm main effect: $\tfrac12[(c_{10}-c_{00}) + (c_{11}-c_{01})]$,
- warmup main effect: $\tfrac12[(c_{01}-c_{00}) + (c_{11}-c_{10})]$,
- interaction: $c_{11}-c_{10}-c_{01}+c_{00}$. If the factors simply add, it is 0. A positive interaction means each helps more when the other is present.

All three are computed **per seed** (arrays of length 8) so that they can be tested with a paired test later.
# %% task Factorial effects
# factorial_effects(cells) with cells = {(0,0): arr, (1,0): arr, (0,1): arr, (1,1): arr}, keys (norm, warmup)
# -> dict(norm=array, warmup=array, interaction=array)  (per-seed effects, same length as the inputs)
def factorial_effects(cells):
    raise NotImplementedError   # TODO
# %% check
_c = {(0, 0): np.array([0.5, 0.6]), (1, 0): np.array([0.7, 0.9]), (0, 1): np.array([0.6, 0.7]), (1, 1): np.array([0.9, 1.1])}
_fe = factorial_effects(_c)
check("norm main effect averages the effect of norm with and without warmup", lambda: np.allclose(_fe["norm"], [(0.2 + 0.3) / 2, (0.3 + 0.4) / 2]), "0.5 * ((c10-c00) + (c11-c01))")
check("warmup main effect", lambda: np.allclose(_fe["warmup"], [(0.1 + 0.2) / 2, (0.1 + 0.2) / 2]), "0.5 * ((c01-c00) + (c11-c10))")
check("interaction is c11 - c10 - c01 + c00 per seed", lambda: np.allclose(_fe["interaction"], [0.1, 0.1]), "difference of differences")
def _additive():
    r = np.random.default_rng(0).normal(size=5); cells = {(0, 0): r, (1, 0): r + 0.3, (0, 1): r + 0.1, (1, 1): r + 0.4}
    e = factorial_effects(cells)
    return np.allclose(e["interaction"], 0) and np.allclose(e["norm"], 0.3) and np.allclose(e["warmup"], 0.1) and e["norm"].shape == (5,)
check("an additive world has zero interaction and recovers the planted effects", _additive, "per-seed arrays, no averaging over seeds")
# %% hint
Unpack `c00 = cells[(0, 0)]` etc. and combine the arrays elementwise.
# %% solution
def factorial_effects(cells):
    c00, c10, c01, c11 = cells[(0, 0)], cells[(1, 0)], cells[(0, 1)], cells[(1, 1)]
    return dict(norm=0.5 * ((c10 - c00) + (c11 - c01)),
                warmup=0.5 * ((c01 - c00) + (c11 - c10)),
                interaction=c11 - c10 - c01 + c00)
# %% code
def factorial_at(lr):
    cells = {(n, w): accs(cfg(lr=lr, norm=bool(n), warmup=20 * w)) for n in (0, 1) for w in (0, 1)}
    return cells, factorial_effects(cells)

FACT = {}
print(f"{'lr':>5} | {'plain':>6} {'norm':>6} {'warmup':>6} {'both':>6} | effects (mean +- sem over seeds): norm, warmup, interaction")
for lr in (1.0, 3.0):
    cells, eff = factorial_at(lr); FACT[lr] = (cells, eff)
    m = [cells[k].mean() for k in ((0, 0), (1, 0), (0, 1), (1, 1))]
    e = [f"{eff[k].mean():+.3f}+-{eff[k].std(ddof=1)/np.sqrt(len(eff[k])):.3f}" for k in ("norm", "warmup", "interaction")]
    print(f"{lr:>5g} | {m[0]:>6.3f} {m[1]:>6.3f} {m[2]:>6.3f} {m[3]:>6.3f} | " + "  ".join(e))
# %% md Stage 5 summary
The ablation shows the dose-response of warmup, the factorial shows main effects and whether they interact. Look at the signs. A main effect that is positive in one learning rate and negative in another is a finding in itself; it means the claim has a regime of validity. No test has been run yet, so everything here is still descriptive.
# %% report
State, for lr=1 and lr=3, the sign and rough size of each of: normalisation main effect, warmup main effect, interaction. Which of your three Stage 1 predictions are supported descriptively?
# %% md Stage 6: statistical significance
## Paired tests

For a comparison of configuration A against B over the same seeds, take the per-seed differences $d_i=a_i-b_i$. The paired t statistic is $t=\bar d/(s_d/\sqrt n)$ with $n-1$ degrees of freedom, the two-sided p-value is $2\,(1-F_{t}(|t|))$, and the standardised effect size is $d_z=\bar d/s_d$. Pairing removes the seed-to-seed variation that is shared by both arms (same initial weights, same batches).

## Many comparisons, many chances
Testing 8 comparisons at 5% each gives a $1-0.95^8\approx 34\%$ chance of at least one false positive even if nothing is real. The **Bonferroni** correction multiplies each p by the number of tests $m$. The **Holm** step-down correction is uniformly more powerful: sort the p-values ascending, $p_{(1)}\le\dots\le p_{(m)}$, set $\tilde p_{(i)}=\max_{j\le i}\,(m-j+1)\,p_{(j)}$ (capped at 1), and map back to the original order. Both control the probability of any false positive (the family-wise error rate).
# %% task Paired t-test
# paired_t_test(a, b) -> dict(mean_diff, sd_diff, t, df, p, dz)   (a - b, two-sided p using t_cdf)
#   if sd_diff == 0: t = 0 and p = 1 when mean_diff == 0, otherwise t = +-inf and p = 0; dz likewise 0 / +-inf
def paired_t_test(a, b):
    raise NotImplementedError   # TODO
# %% check
_pt = paired_t_test(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
check("diffs 1..5: t = 4.243 with 4 degrees of freedom", lambda: _pt["df"] == 4 and np.isclose(_pt["mean_diff"], 3.0) and np.isclose(_pt["sd_diff"], 1.5811388, atol=1e-6) and np.isclose(_pt["t"], 4.2426, atol=1e-3), "t = mean / (sd / sqrt(n))")
check("two-sided p-value is 0.0132 and dz = 1.897", lambda: np.isclose(_pt["p"], 0.01324, atol=2e-4) and np.isclose(_pt["dz"], 1.8974, atol=1e-3), "p = 2 * (1 - t_cdf(|t|, df))")
check("swapping a and b flips the sign but keeps p", lambda: (lambda r: np.isclose(r["t"], -_pt["t"]) and np.isclose(r["p"], _pt["p"]))(paired_t_test(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 4.0, 6.0, 8.0, 10.0]))), "two-sided")
check("degenerate cases do not crash", lambda: paired_t_test([1.0, 2.0], [1.0, 2.0])["p"] == 1.0 and paired_t_test([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])["p"] == 0.0, "zero variance of the differences")
check("the pairing matters: noise shared by both arms cancels", lambda: paired_t_test(np.arange(8) * 5.0 + 1.0, np.arange(8) * 5.0)["p"] < 1e-6, "huge between-seed spread, constant within-seed difference")
# %% hint
`d = np.asarray(a) - np.asarray(b)`, `sd = d.std(ddof=1)`, `t = d.mean() / (sd / np.sqrt(n))`.
# %% solution
def paired_t_test(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d); md = float(d.mean()); sd = float(d.std(ddof=1))
    if sd == 0:
        if md == 0:
            return dict(mean_diff=md, sd_diff=0.0, t=0.0, df=n - 1, p=1.0, dz=0.0)
        s = math.copysign(math.inf, md)
        return dict(mean_diff=md, sd_diff=0.0, t=s, df=n - 1, p=0.0, dz=s)
    t = md / (sd / math.sqrt(n))
    return dict(mean_diff=md, sd_diff=sd, t=t, df=n - 1, p=2 * (1 - t_cdf(abs(t), n - 1)), dz=md / sd)
# %% task Multiple-comparison correction
# adjust_pvalues(p, method="holm") -> array of adjusted p-values in the ORIGINAL order, capped at 1
#   methods: "bonferroni" (p*m) and "holm" (step-down, see above)
def adjust_pvalues(p, method="holm"):
    raise NotImplementedError   # TODO
# %% check
check("bonferroni multiplies by m and caps at 1", lambda: np.allclose(adjust_pvalues([0.01, 0.04, 0.5], "bonferroni"), [0.03, 0.12, 1.0]), "min(1, p*m)")
check("holm example [0.01, 0.04, 0.03, 0.005] -> [0.03, 0.06, 0.06, 0.02]", lambda: np.allclose(adjust_pvalues([0.01, 0.04, 0.03, 0.005], "holm"), [0.03, 0.06, 0.06, 0.02]), "sort, multiply by (m-rank), running max, unsort")
def _prop():
    r = np.random.default_rng(3)
    for _ in range(20):
        p = r.uniform(0, 0.2, 7)
        h, b = adjust_pvalues(p, "holm"), adjust_pvalues(p, "bonferroni")
        if not (np.all(h <= b + 1e-12) and np.all(h >= p - 1e-12) and np.all(h <= 1)):
            return False
        o = np.argsort(p)
        if not np.all(np.diff(h[o]) >= -1e-12):
            return False
    return True
check("holm <= bonferroni, never below the raw p, monotone in p order", _prop, "running maximum guarantees monotonicity")
check("a single test is unchanged; ties and an unsorted input keep their positions", lambda: np.allclose(adjust_pvalues([0.04]), [0.04]) and np.allclose(adjust_pvalues([0.04, 0.01]), [0.04, 0.02]), "return in the original order")
# %% hint
`order = np.argsort(p)`; `adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])`; clip to 1; `out[order] = adj_sorted`.
# %% solution
def adjust_pvalues(p, method="holm"):
    p = np.asarray(p, float); m = len(p)
    if method == "bonferroni":
        return np.minimum(1.0, p * m)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(1.0, adj)
    return out
# %% md Applying the tests
Eight planned comparisons (the family): at each of the two stress learning rates, `norm vs plain`, `warmup vs plain`, `norm+warmup vs norm`, `norm+warmup vs plain`. Pairing is by seed. The family is declared in advance and corrected as a whole; we do not get to drop the ones that look boring.
# %% code
PAIRS = [("norm", "plain"), ("warmup", "plain"), ("norm+warmup", "norm"), ("norm+warmup", "plain")]
rows = []
for lr in (1.0, 3.0):
    for a_name, b_name in PAIRS:
        a = accs(cfg(lr=lr, **RECIPES[a_name])); b = accs(cfg(lr=lr, **RECIPES[b_name]))
        t = paired_t_test(a, b); rows.append((lr, a_name, b_name, t))
padj = adjust_pvalues([r[3]["p"] for r in rows], "holm")
print(f"{'lr':>4} {'A':>12} vs {'B':<12} | {'mean diff':>9} {'t':>7} {'dz':>6} {'p raw':>8} {'p holm':>8}  sig@5%")
for (lr, a_name, b_name, t), pa in zip(rows, padj):
    print(f"{lr:>4g} {a_name:>12} vs {b_name:<12} | {t['mean_diff']:>+9.3f} {t['t']:>7.2f} {t['dz']:>6.2f} {t['p']:>8.4f} {pa:>8.4f}  {'YES' if pa < 0.05 else 'no'}")
TESTS = {(lr, a, b): (t, pa) for (lr, a, b, t), pa in zip(rows, padj)}
# %% md Stage 6 summary
Significance is a statement about noise, not about importance: read `mean diff` (accuracy points) and `dz` next to the p-values. With 8 seeds the power is modest, so a non-significant result is *inconclusive* rather than evidence of no effect. The Holm column is the one to quote.
# %% report
Pick the most interesting row. Explain in plain words what its raw and corrected p-values mean, and what you would conclude if you had run only that one comparison (a form of the "forking paths" problem).
# %% md Stage 7: failure analysis
## What exactly goes wrong at a bad learning rate?

An accuracy of 0.67 hides different failure modes with different cures. We classify each run:

- `diverged`: the loss or the weights became non-finite;
- `chance`: finished, but test accuracy < 0.60 (the network predicts nearly one class);
- `degraded`: 0.60 <= accuracy < 0.85 (trained, but badly);
- `ok`: accuracy >= 0.85.
# %% task Classify runs and count failure modes
# classify_run(result) -> one of "diverged", "chance", "degraded", "ok"   (result has test_acc and diverged)
# failure_counts(results) -> dict with all four categories as keys (counts, zeros included)
def classify_run(result):
    raise NotImplementedError   # TODO

def failure_counts(results):
    raise NotImplementedError   # TODO
# %% check
check("each category on a synthetic result", lambda: classify_run(dict(test_acc=0.5, diverged=True)) == "diverged" and classify_run(dict(test_acc=0.55, diverged=False)) == "chance" and classify_run(dict(test_acc=0.7, diverged=False)) == "degraded" and classify_run(dict(test_acc=0.9, diverged=False)) == "ok", "diverged takes priority over accuracy")
check("boundaries: 0.60 is degraded, 0.85 is ok", lambda: classify_run(dict(test_acc=0.60, diverged=False)) == "degraded" and classify_run(dict(test_acc=0.85, diverged=False)) == "ok" and classify_run(dict(test_acc=0.599, diverged=False)) == "chance", ">= comparisons")
_rs = [dict(test_acc=0.9, diverged=False), dict(test_acc=0.9, diverged=False), dict(test_acc=0.7, diverged=False), dict(test_acc=0.5, diverged=True)]
check("counts include zero entries and sum to the number of runs", lambda: failure_counts(_rs) == {"diverged": 1, "chance": 0, "degraded": 1, "ok": 2}, "initialise every key to 0")
check("on the real grid, the plain network at lr=3 fails more often than the normalised one", lambda: (lambda a, b: sum(a[k] for k in ("diverged", "chance", "degraded")) > sum(b[k] for k in ("diverged", "chance", "degraded")))(failure_counts(STORE.select(lr=3.0, norm=False, warmup=0)), failure_counts(STORE.select(lr=3.0, norm=True, warmup=0))), "uses the stored sweep")
# %% hint
Use a dict `{k: 0 for k in ("diverged", "chance", "degraded", "ok")}` and increment `classify_run(r)`.
# %% solution
def classify_run(result):
    if result["diverged"]:
        return "diverged"
    a = result["test_acc"]
    if a < 0.60:
        return "chance"
    if a < 0.85:
        return "degraded"
    return "ok"

def failure_counts(results):
    out = {k: 0 for k in ("diverged", "chance", "degraded", "ok")}
    for r in results:
        out[classify_run(r)] += 1
    return out
# %% code
print(f"{'recipe':12s} {'lr':>5} | diverged chance degraded ok | mean steps before loss first exceeds 2x its first-10-step mean")
for name in RECIPES:
    for lr in (1.0, 3.0):
        res = STORE.select(lr=lr, **RECIPES[name])
        fc = failure_counts(res)
        spikes = []
        for r in res:
            L = np.array(r["losses"])
            if len(L) < 12:
                continue
            ref = L[:10].mean(); hit = np.flatnonzero(L > 2 * ref)
            spikes.append(hit[0] if len(hit) else len(L))
        print(f"{name:12s} {lr:>5g} | {fc['diverged']:>8} {fc['chance']:>6} {fc['degraded']:>8} {fc['ok']:>2} | {np.mean(spikes):6.1f}  (of {res[0]['config']['steps']})")
# %% md Reading the failure table
Compare rows: runs that end `degraded` rarely diverge to infinity. They train, but lose accuracy to a few catastrophic updates, typically dead ReLUs (units that output zero for every input and therefore get zero gradient) or a loss that spikes and never recovers. The last column tells when the first spike happens: if warmup really protected the network from the first-step jump, spikes should occur later or not at all for warmup recipes. Whether that happens is an empirical question about *this* table; do not assume it.
# %% report
Failure analysis: describe the dominant failure mode of the plain network at lr=3, and say whether warmup changes its timing or its frequency. State one experiment (not run here) that would separate "dead ReLU units" from "exploding weights" as the mechanism.
# %% md Stage 8: verdict and report
## A decision rule you can defend

The last function turns a measured effect into a sentence the data can support. Inputs: the mean paired difference (positive means the proposed recipe is better), its 95% t-interval, the Holm-adjusted p-value and the direction the claim predicts (+1: improvement). Rule:

- `supported`: adjusted p < alpha and the interval excludes 0 on the predicted side;
- `contradicted`: adjusted p < alpha and the interval excludes 0 on the opposite side;
- `inconclusive`: otherwise (the data do not discriminate; more seeds or a bigger effect are needed).
# %% task Verdict function
# verdict(mean_diff, ci, p_adj, alpha=0.05, expected_sign=+1) -> "supported" | "contradicted" | "inconclusive"
def verdict(mean_diff, ci, p_adj, alpha=0.05, expected_sign=+1):
    raise NotImplementedError   # TODO
# %% check
check("a clear improvement is supported", lambda: verdict(0.08, (0.04, 0.12), 0.01) == "supported", "significant and the interval is above 0")
check("a clear degradation is contradicted", lambda: verdict(-0.05, (-0.09, -0.01), 0.02) == "contradicted", "significant, opposite sign")
check("significant but the interval touches 0, or not significant: inconclusive", lambda: verdict(0.05, (-0.01, 0.11), 0.20) == "inconclusive" and verdict(0.05, (-0.01, 0.11), 0.01) == "inconclusive", "both conditions are needed")
check("expected_sign=-1 flips the roles (e.g. lower loss is better)", lambda: verdict(-0.3, (-0.5, -0.1), 0.01, expected_sign=-1) == "supported" and verdict(0.3, (0.1, 0.5), 0.01, expected_sign=-1) == "contradicted", "multiply by expected_sign")
check("alpha is respected", lambda: verdict(0.08, (0.04, 0.12), 0.04, alpha=0.01) == "inconclusive" and verdict(0.08, (0.04, 0.12), 0.04, alpha=0.05) == "supported", "p_adj < alpha")
# %% hint
Multiply the difference and the interval by `expected_sign` (and re-sort the interval ends) so that the claim always predicts a positive number.
# %% solution
def verdict(mean_diff, ci, p_adj, alpha=0.05, expected_sign=+1):
    lo, hi = sorted((expected_sign * ci[0], expected_sign * ci[1]))
    if p_adj < alpha and lo > 0:
        return "supported"
    if p_adj < alpha and hi < 0:
        return "contradicted"
    return "inconclusive"
# %% md The generated report table
The cell below assembles the verdicts from the numbers you computed. Edit the sentences in your own report after reading it; the code only prevents you from reporting a verdict the statistics do not support.
# %% code
print("REPRODUCTION REPORT (generated from STORE)")
print("Claim: warmup and normalisation make training robust to the learning rate.")
print(f"Setup: moons MLP 2-16-16-1, Adam, 100 steps, batch 32, {len(SEEDS)} seeds, {len(LRS)} learning rates, {STORE.n_runs} runs in total.")
print("Robustness (worst-case mean acc / window width in decades at acc>=0.88):")
for name, rb in ROB.items():
    print(f"  {name:12s} worst {rb['worst']:.3f}   window {len(rb['window_lrs'])} lrs / {rb['decades']:.1f} decades")
print("Paired comparisons at stress learning rates (Holm-adjusted, expected direction: first recipe is better):")
for (lr, a_name, b_name), (t, pa) in TESTS.items():
    d = np.array(accs(cfg(lr=lr, **RECIPES[a_name]))) - np.array(accs(cfg(lr=lr, **RECIPES[b_name])))
    ci = summarize(d)["ci_t"]
    print(f"  lr={lr:<3g} {a_name:>12} vs {b_name:<7} diff {t['mean_diff']:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p_holm {pa:.3f}  -> {verdict(t['mean_diff'], ci, pa)}")
print("Interaction (norm x warmup), per-seed mean +- sem:")
for lr in (1.0, 3.0):
    e = FACT[lr][1]["interaction"]
    print(f"  lr={lr:<3g} {e.mean():+.3f} +- {e.std(ddof=1)/np.sqrt(len(e)):.3f}")
# %% md Deviations from a real reproduction
- The original claim concerns large transformers trained for many thousands of steps. Ours is an MLP with two hidden layers of 16 units trained for 100 steps on a 250-example toy problem. We reproduce a *qualitative* statement only.
- The normalisation is parameter-free LayerNorm, without the learned gain and bias that most implementations have; and the ReLU network is post-norm (norm before the activation inside each hidden layer).
- Warmup is 20 steps of 100: a fifth of the training budget, far more than the roughly 1 percent typical for large runs, so the warmup arm trains with a smaller effective learning rate for a large share of the run. We did not tune the warmup length per arm beyond the dose-response table.
- The learning-rate grid is shared by every arm and not tuned per arm; the metric (test accuracy on 150 points) has a resolution of 0.7 points.
- Only Adam was tried; the folk argument for warmup is about Adam's early steps, and results for SGD can differ.
# %% md Limitations
Eight seeds give wide intervals and modest power; a non-significant row is inconclusive, not a null effect. One dataset, one architecture, one step budget. The seed controls initialisation and batch order but not the data, so variance from the sample of data is not in our error bars. Eight comparisons were corrected together; other plausible comparisons (for example other learning rates) were not part of the pre-declared family, and any additional look at the data would need a new correction. Mechanisms (dead ReLUs, first-step jumps) are hypotheses unless a dedicated experiment tests them.
# %% report
Final reproduction report (write it as you would for a colleague). Include: (1) the claim in operational terms; (2) the headline result with numbers from the generated report; (3) the verdict for each of the three Stage 1 predictions; (4) two deviations that most limit the conclusion; (5) the next experiment you would run, with its seeds and factors.
# %% md Sources
Only material that was actually opened and read while writing this project:

- Lesson `sys.05` ("Research practice: reading, reproducing and ablating"), `content/authored/llm-systems/05-research-practice/lesson.md`, sections "Reproducing a result", "Designing a controlled experiment", the worked moons/warmup example, "Reporting honestly" and "Spotting weaknesses": protocol, pairing by seed, paired t formula and reporting rules.
- `docs/PROJECTS.md` and `docs/AGENT_BRIEF.md` of this repository (format and rules).
- Everything else (Adam, LayerNorm backward, Holm correction, factorial effects, the t distribution) is standard textbook material written from general knowledge and verified numerically by the checks in this notebook (finite differences, a hand-computed example, known critical values). No paper on warmup or normalisation was opened for this project, so the claim is used as a statement to test, not as a cited result.
