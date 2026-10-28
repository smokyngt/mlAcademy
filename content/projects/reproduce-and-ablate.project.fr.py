# %% meta
{"id": "proj.reproduce", "title": "Reproduire et ablater : le warmup et la normalisation rendent-ils l'entraînement robuste au taux d'apprentissage ?", "summary": "Un projet de recherche en miniature : transformer une affirmation de comptoir en hypothèse testable, construire un banc d'expériences à graines fixées avec un stockage des résultats et des intervalles de confiance, lancer un balayage du taux d'apprentissage, une ablation à un facteur et un plan factoriel 2x2 sur 8 graines, tester la significativité avec des tests appariés et une correction pour comparaisons multiples, analyser les échecs et rédiger un rapport de reproduction honnête.", "level": "expert", "minutes": 150,
 "requires": ["sys.05", "trf.04"],
 "outcomes": ["Transformer une affirmation vague en hypothèse falsifiable avec une métrique pré-enregistrée", "Construire un banc d'essai déterministe : graines, clés de résultat canoniques, cache, agrégation avec intervalles de confiance t et bootstrap", "Lancer un balayage du taux d'apprentissage et quantifier la robustesse au lieu de regarder un graphique à l'œil", "Séparer les effets principaux de l'interaction dans un plan factoriel 2x2", "Utiliser des tests appariés et la correction de Holm, et dire ce que signifie un résultat nul ou inversé", "Rédiger un rapport de reproduction avec les écarts, les limites et un verdict que les données peuvent soutenir"],
 "stages": ["Affirmation et plan", "Modèle et boucle d'entraînement", "Banc d'expériences", "Balayage du taux d'apprentissage", "Ablation et plan factoriel", "Significativité", "Analyse des échecs", "Rapport"]}
# %% md Étape 1 : l'affirmation et le plan
## L'affirmation

> *« Le warmup du taux d'apprentissage et les couches de normalisation rendent l'entraînement robuste au choix du taux d'apprentissage. »*

Cette phrase est répétée dans de nombreux articles et exposés sur les transformers. Ce n'est pourtant **pas encore une hypothèse** : robuste en quel sens, par rapport à quoi, sur quel modèle ? Une reproduction commence par rendre l'affirmation assez précise pour que les données puissent la contredire.

**Notre version opérationnelle.** Petit réseau, petites données, réduit dans toutes les directions (voir « Écarts » à la fin) :

- Modèle : un MLP `2 -> 16 -> 16 -> 1` avec ReLU, entraîné avec Adam sur la tâche de classification des deux lunes (two-moons).
- Facteurs manipulés : `norm` (LayerNorm sans paramètres après chaque couche linéaire cachée : oui/non) et `warmup` (warmup linéaire du taux d'apprentissage sur 20 des 100 pas : oui/non).
- Taux d'apprentissage : 6 valeurs sur une grille logarithmique de 0,01 à 3.
- **Métrique principale (fixée avant de regarder le moindre résultat) : l'exactitude finale sur le jeu de test.** « Robuste » veut dire deux choses que nous savons calculer : la *pire* exactitude moyenne sur la grille de taux d'apprentissage, et la *largeur de la fenêtre* de taux d'apprentissage (en décades) dans laquelle l'exactitude moyenne reste au-dessus de 0,88.
- 8 graines par cellule. La graine contrôle les poids initiaux et l'ordre des mini-batchs ; le jeu de données est fixe. Utiliser la même graine pour deux configurations leur donne les mêmes poids initiaux et le même ordre de batchs, ce qui rend possibles des comparaisons **appariées**.
- Prédictions écrites avant de lancer quoi que ce soit : (1) la normalisation élargit la fenêtre ; (2) le warmup aide aussi aux taux d'apprentissage élevés ; (3) leurs effets s'additionnent. Nous verrons lesquelles survivent.

Tout est en NumPy. Chaque étape se termine par un résumé et une invite de rapport en texte libre.
# %% report
Avant de lancer quoi que ce soit : écris quel résultat te ferait dire que l'affirmation est *fausse* dans ce cadre. Si aucun résultat ne le pourrait, réécris l'affirmation.
# %% md Étape 2 : modèle et boucle d'entraînement
## La normalisation, écrite à la main

Pour un batch `z` de forme `(N, D)`, la LayerNorm sans paramètres normalise chaque ligne : $y = (z-\mu)/\sigma$ où $\mu,\sigma^2$ sont la moyenne et la variance de la ligne (avec $\varepsilon=10^{-5}$ ajouté sous la racine). Avec le gradient amont `dy`, la passe arrière est
$$dz = \frac{1}{\sigma}\Big(dy - \overline{dy} - y\,\overline{dy\odot y}\Big),$$
où les barres désignent des moyennes par ligne. (Dérivation : $y$ dépend de $z_j$ directement, à travers $\mu$ et à travers $\sigma$ ; les deux derniers termes projettent hors des directions du vecteur de uns et de $y$ lui-même, qui sont les deux choses que la normalisation supprime.)
# %% code
import json, math, time
import numpy as np
EPS_LN = 1e-5
# %% task LayerNorm : passe avant et passe arrière
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
Garde `y` et `sigma = sqrt(var + eps)` dans le cache ; la passe arrière a besoin des deux.
# %% solution
def layernorm_forward(z):
    mu = z.mean(1, keepdims=True)
    sigma = np.sqrt(z.var(1, keepdims=True) + EPS_LN)
    y = (z - mu) / sigma
    return y, (y, sigma)

def layernorm_backward(dy, cache):
    y, sigma = cache
    return (dy - dy.mean(1, keepdims=True) - y * (dy * y).mean(1, keepdims=True)) / sigma
# %% md Optimiseur et planning
Adam conserve des moyennes mobiles du gradient `m` et de son carré `v`, et déplace chaque paramètre d'environ $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, où $\hat m=m/(1-\beta_1^t)$ et $\hat v=v/(1-\beta_2^t)$ sont corrigés du biais. Une conséquence à retenir : au **premier** pas, $\hat m/\sqrt{\hat v}=\pm1$, donc chaque paramètre bouge d'environ $\eta$, quelle que soit la petitesse ou la grandeur de son gradient. Avec un $\eta$ élevé, c'est un grand saut aveugle, et c'est l'argument de comptoir en faveur du warmup : partir d'un $\eta_t$ petit et le faire monter linéairement, $\eta_t=\eta\cdot\min(1,(t+1)/w)$ pour $t=0,1,\dots$ (et $\eta_t=\eta$ quand $w=0$).
# %% task Planning de warmup et mise à jour d'Adam
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
Mets à jour `st["m"][:] = ...` sur place et incrémente `st["t"]` AVANT de calculer les corrections.
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
# %% md Le modèle, les données et un entraînement (fourni)
Le code ci-dessous utilise ta LayerNorm et ton Adam. Les paramètres vivent dans **un seul vecteur plat** `theta`, de sorte qu'Adam se réduit à une poignée d'appels NumPy ; `unpack` renvoie des vues de ses morceaux. La dernière cellule de cette étape vérifie la passe arrière écrite à la main contre les différences finies, **avant** de faire confiance à la moindre expérience : un gradient non vérifié invaliderait silencieusement tous les nombres qui suivent.
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
# %% md Résumé de l'étape 2
Tu as un modèle vérifié, un optimiseur et un planning, ainsi qu'une fonction d'entraînement dont la seule source d'aléa est la graine. Deux contrôles de cohérence ont été passés avant de faire de la science : les gradients par différences finies, et la répétition à l'identique, bit pour bit, d'un entraînement. Sans eux, un « résultat » pourrait n'être qu'un bug.
# %% report
Pourquoi est-il important que le jeu de données soit fixe et que la graine ne contrôle que l'initialisation et l'ordre des batchs ? Quelle question changerait si la graine rééchantillonnait aussi le jeu de données ?
# %% md Étape 3 : le banc d'expériences
## Stockage des résultats

Une expérience est une fonction `(config, seed) -> result`. Lancer à la main des grilles d'expériences dans un notebook mène à des résultats perdus, à des réglages accidentellement différents et à des relances qui prennent des heures. Un **stockage des résultats** donne à chaque entraînement une clé canonique (la config sérialisée avec les clés triées, plus la graine), ne le lance que si la clé est absente et renvoie sinon le résultat en cache. Toute analyse ultérieure rappelle `get_or_run` et obtient instantanément les mêmes nombres : le notebook devient une *requête* sur une fonction déterministe.
# %% task Stockage des résultats avec clés canoniques
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
# %% md Intervalles de confiance
La moyenne et l'écart-type décrivent les graines ; un **intervalle de confiance** décrit notre incertitude sur la vraie moyenne. Pour $n$ graines d'écart-type empirique $s$ (division par $n-1$), l'erreur standard vaut $s/\sqrt n$ et l'intervalle t à 95 % est $\bar x \pm t_{0.975,\,n-1}\, s/\sqrt n$. Avec $n=8$, la valeur critique est 2,365 et non 1,96 ; utiliser la valeur de la loi normale avec une poignée de graines rend les intervalles trop étroits. Pour avoir un second avis, nous calculons aussi un intervalle **bootstrap par percentiles** : rééchantillonner les $n$ valeurs avec remise un grand nombre de fois, prendre la moyenne à chaque fois et lire les quantiles à 2,5 % et 97,5 %.

La fonction de répartition de la loi de Student est donnée ci-dessous (intégration numérique de sa densité ; pas de SciPy dans le bac à sable).
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
# %% task Agrégation avec intervalles de confiance
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
`q = 1 - (1 - level) / 2` ; `half = t_ppf(q, n - 1) * sem`. Bootstrap : `idx = rng.integers(0, n, (n_boot, n))`, `means = x[idx].mean(1)`, `np.percentile(means, [100*(1-level)/2, 100*q])`.
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
# %% md Lancer le banc d'expériences
`STORE` enveloppe `train_run`. L'assistant `acc_table` rassemble les exactitudes par graine pour une liste de configs ; il parcourt toujours les mêmes graines dans le même ordre, ce qui rend valide l'appariement par indice.
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
# %% md Résumé de l'étape 3
Le banc d'expériences t'apporte le déterminisme (rng à graine fixée, clés canoniques), la mémoïsation (le stockage) et une incertitude honnête (intervalles t et bootstrap). Chaque nombre des étapes restantes est une requête sur `STORE`.
# %% report
Pourquoi un écart-type n'est-il pas la même chose qu'un intervalle de confiance, et lequel les barres d'erreur d'un article devraient-elles montrer ? Lequel diminue quand tu ajoutes des graines ?
# %% md Étape 4 : le balayage du taux d'apprentissage (reproduction)
## 4 recettes x 6 taux d'apprentissage x 8 graines = 192 entraînements

Nous lançons d'abord toute la grille, en quatre cellules (une par recette) pour que chacune reste rapide. Rien n'est réglé par recette : chaque bras reçoit le même budget de pas, la même taille de batch, les mêmes graines et la même grille de taux d'apprentissage.
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
# %% md Quantifier la robustesse
Regarder un tableau à l'œil n'est pas un protocole. Nous définissons, pour une recette, `acc_by_lr` (taux d'apprentissage -> liste des exactitudes par graine), puis :

- `means[lr]` : l'exactitude moyenne à chaque taux d'apprentissage ;
- `worst` : le minimum de ces moyennes (la robustesse dans le pire cas) ;
- `window_lrs` : les taux d'apprentissage dont l'exactitude moyenne vaut au moins `thresh` ;
- `decades` : $\log_{10}(\max\text{ window lr} / \min\text{ window lr})$ (0 quand la fenêtre compte moins de deux éléments).
# %% task Métriques de robustesse d'un balayage
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
Les clés de `acc_by_lr` sont des flottants ; trie-les avant de construire `window_lrs`. `decades = np.log10(max(w) / min(w))`.
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
# %% md Résumé de l'étape 4
Lis le tableau avec les colonnes de fenêtre et de pire cas, pas seulement avec tes yeux. Moyenne +- écart-type sur 8 graines : l'écart-type est la dispersion d'une graine à l'autre pour un entraînement unique, c'est-à-dire ce que vivra un praticien qui n'entraîne qu'une fois. Ne prétends **pas** encore savoir lequel de `norm` et `warmup` est responsable des changements : un balayage ne confond rien, mais il n'isole rien non plus. L'étape suivante s'en charge.
# %% report
Résume le balayage en trois phrases : quelles recettes tolèrent lr=1 et lr=3, quels taux d'apprentissage sont les meilleurs pour chaque recette, et quelque chose t'a-t-il surpris par rapport à ta prédiction de l'étape 1 ?
# %% md Étape 5 : ablation et plan factoriel
## Ablation à un facteur : quelle durée pour le warmup ?

On change une seule chose et on garde le reste fixe : durée du warmup dans {0, 5, 10, 20, 40} pas au taux d'apprentissage de stress 1,0, une fois sans et une fois avec normalisation. Mêmes graines partout, donc chaque ligne est appariée à la ligne à 0 pas.
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
# %% md Plan factoriel 2x2
Ablater un facteur à la fois ne permet pas de savoir si les facteurs interagissent. Un **plan factoriel** lance les quatre combinaisons. Pour des résultats par graine $c_{nw}$ avec $n$ = norm (0/1) et $w$ = warmup (0/1) (tous appariés par graine) :

- effet principal de norm : $\tfrac12[(c_{10}-c_{00}) + (c_{11}-c_{01})]$,
- effet principal du warmup : $\tfrac12[(c_{01}-c_{00}) + (c_{11}-c_{10})]$,
- interaction : $c_{11}-c_{10}-c_{01}+c_{00}$. Si les facteurs s'additionnent simplement, elle vaut 0. Une interaction positive signifie que chacun aide davantage quand l'autre est présent.

Les trois sont calculés **par graine** (tableaux de longueur 8) afin de pouvoir être testés plus tard avec un test apparié.
# %% task Effets factoriels
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
Extrais `c00 = cells[(0, 0)]` etc. et combine les tableaux élément par élément.
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
# %% md Résumé de l'étape 5
L'ablation montre la relation dose-réponse du warmup, le plan factoriel montre les effets principaux et leur éventuelle interaction. Regarde les signes. Un effet principal positif pour un taux d'apprentissage et négatif pour un autre est un résultat en soi : cela signifie que l'affirmation a un domaine de validité. Aucun test n'a encore été lancé, donc tout ce qui est ici reste descriptif.
# %% report
Indique, pour lr=1 et lr=3, le signe et l'ordre de grandeur de chacun de ces éléments : effet principal de la normalisation, effet principal du warmup, interaction. Lesquelles de tes trois prédictions de l'étape 1 sont soutenues de façon descriptive ?
# %% md Étape 6 : significativité statistique
## Tests appariés

Pour comparer une configuration A à une configuration B sur les mêmes graines, on prend les différences par graine $d_i=a_i-b_i$. La statistique t appariée est $t=\bar d/(s_d/\sqrt n)$ avec $n-1$ degrés de liberté, la p-valeur bilatérale est $2\,(1-F_{t}(|t|))$, et la taille d'effet standardisée est $d_z=\bar d/s_d$. L'appariement supprime la variation d'une graine à l'autre qui est partagée par les deux bras (mêmes poids initiaux, mêmes batchs).

## Beaucoup de comparaisons, beaucoup de chances
Tester 8 comparaisons à 5 % chacune donne une probabilité d'environ $1-0.95^8\approx 34\%$ d'obtenir au moins un faux positif, même si rien n'est réel. La correction de **Bonferroni** multiplie chaque p par le nombre de tests $m$. La correction pas à pas de **Holm** est uniformément plus puissante : trie les p-valeurs par ordre croissant, $p_{(1)}\le\dots\le p_{(m)}$, pose $\tilde p_{(i)}=\max_{j\le i}\,(m-j+1)\,p_{(j)}$ (plafonné à 1), puis remets-les dans l'ordre d'origine. Les deux contrôlent la probabilité d'avoir un quelconque faux positif (le taux d'erreur par famille).
# %% task Test t apparié
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
# %% task Correction pour comparaisons multiples
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
`order = np.argsort(p)` ; `adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])` ; plafonne à 1 ; `out[order] = adj_sorted`.
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
# %% md Appliquer les tests
Huit comparaisons planifiées (la famille) : pour chacun des deux taux d'apprentissage de stress, `norm vs plain`, `warmup vs plain`, `norm+warmup vs norm`, `norm+warmup vs plain`. L'appariement se fait par graine. La famille est déclarée à l'avance et corrigée dans son ensemble ; nous n'avons pas le droit d'écarter celles qui paraissent ennuyeuses.
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
# %% md Résumé de l'étape 6
La significativité est un énoncé sur le bruit, pas sur l'importance : lis `mean diff` (en points d'exactitude) et `dz` à côté des p-valeurs. Avec 8 graines, la puissance est modeste, donc un résultat non significatif est *non concluant* plutôt qu'une preuve d'absence d'effet. La colonne de Holm est celle qu'il faut citer.
# %% report
Choisis la ligne la plus intéressante. Explique avec des mots simples ce que signifient ses p-valeurs brute et corrigée, et ce que tu conclurais si tu n'avais lancé que cette comparaison (une forme du problème des « chemins qui bifurquent », ou forking paths).
# %% md Étape 7 : analyse des échecs
## Que se passe-t-il exactement avec un mauvais taux d'apprentissage ?

Une exactitude de 0,67 cache des modes d'échec différents, avec des remèdes différents. Nous classons chaque entraînement :

- `diverged` : la perte ou les poids sont devenus non finis ;
- `chance` : terminé, mais exactitude de test < 0,60 (le réseau prédit presque toujours une seule classe) ;
- `degraded` : 0,60 <= exactitude < 0,85 (entraîné, mais mal) ;
- `ok` : exactitude >= 0,85.
# %% task Classer les entraînements et compter les modes d'échec
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
Utilise un dict `{k: 0 for k in ("diverged", "chance", "degraded", "ok")}` et incrémente `classify_run(r)`.
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
# %% md Lire le tableau des échecs
Compare les lignes : les entraînements qui finissent `degraded` divergent rarement vers l'infini. Ils s'entraînent, mais perdent en exactitude à cause de quelques mises à jour catastrophiques, typiquement des ReLU morts (des unités qui sortent zéro pour toute entrée et reçoivent donc un gradient nul) ou une fonction de perte qui explose et ne revient jamais. La dernière colonne indique quand survient le premier pic : si le warmup protégeait vraiment le réseau du saut du premier pas, les pics devraient survenir plus tard, ou pas du tout, pour les recettes avec warmup. Que cela se produise ou non est une question empirique sur *ce* tableau ; ne le suppose pas.
# %% report
Analyse des échecs : décris le mode d'échec dominant du réseau simple (plain) à lr=3, et dis si le warmup change son moment d'apparition ou sa fréquence. Énonce une expérience (non lancée ici) qui permettrait de départager « unités ReLU mortes » et « poids qui explosent » comme mécanisme.
# %% md Étape 8 : verdict et rapport
## Une règle de décision que tu peux défendre

La dernière fonction transforme un effet mesuré en une phrase que les données peuvent soutenir. Entrées : la différence appariée moyenne (positive signifie que la recette proposée est meilleure), son intervalle t à 95 %, la p-valeur ajustée de Holm et la direction que prédit l'affirmation (+1 : amélioration). Règle :

- `supported` : p ajustée < alpha et l'intervalle exclut 0 du côté prédit ;
- `contradicted` : p ajustée < alpha et l'intervalle exclut 0 du côté opposé ;
- `inconclusive` : sinon (les données ne tranchent pas ; il faut plus de graines ou un effet plus grand).
# %% task Fonction de verdict
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
Multiplie la différence et l'intervalle par `expected_sign` (et retrie les bornes de l'intervalle) pour que l'affirmation prédise toujours un nombre positif.
# %% solution
def verdict(mean_diff, ci, p_adj, alpha=0.05, expected_sign=+1):
    lo, hi = sorted((expected_sign * ci[0], expected_sign * ci[1]))
    if p_adj < alpha and lo > 0:
        return "supported"
    if p_adj < alpha and hi < 0:
        return "contradicted"
    return "inconclusive"
# %% md Le tableau de rapport généré
La cellule ci-dessous assemble les verdicts à partir des nombres que tu as calculés. Modifie les phrases de ton propre rapport après l'avoir lue ; le code empêche seulement de rapporter un verdict que les statistiques ne soutiennent pas.
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
# %% md Écarts par rapport à une vraie reproduction
- L'affirmation d'origine concerne de grands transformers entraînés pendant plusieurs milliers de pas. Le nôtre est un MLP à deux couches cachées de 16 unités, entraîné pendant 100 pas sur un problème jouet de 250 exemples. Nous ne reproduisons qu'un énoncé *qualitatif*.
- La normalisation est une LayerNorm sans paramètres, sans le gain et le biais appris que la plupart des implémentations possèdent ; et le réseau ReLU est en post-norm (la norme avant l'activation à l'intérieur de chaque couche cachée).
- Le warmup dure 20 pas sur 100 : un cinquième du budget d'entraînement, bien plus que le 1 pour cent environ habituel dans les gros entraînements, si bien que le bras avec warmup s'entraîne avec un taux d'apprentissage effectif plus petit pendant une grande part de l'entraînement. Nous n'avons pas réglé la durée du warmup par bras au-delà du tableau dose-réponse.
- La grille de taux d'apprentissage est partagée par tous les bras et non réglée par bras ; la métrique (exactitude de test sur 150 points) a une résolution de 0,7 point.
- Seul Adam a été essayé ; l'argument de comptoir en faveur du warmup porte sur les premiers pas d'Adam, et les résultats avec SGD peuvent différer.
# %% md Limites
Huit graines donnent des intervalles larges et une puissance modeste ; une ligne non significative est non concluante, pas un effet nul. Un seul jeu de données, une seule architecture, un seul budget de pas. La graine contrôle l'initialisation et l'ordre des batchs mais pas les données, donc la variance due à l'échantillon de données n'est pas dans nos barres d'erreur. Huit comparaisons ont été corrigées ensemble ; d'autres comparaisons plausibles (par exemple d'autres taux d'apprentissage) ne faisaient pas partie de la famille déclarée à l'avance, et tout regard supplémentaire sur les données exigerait une nouvelle correction. Les mécanismes (ReLU morts, sauts du premier pas) restent des hypothèses tant qu'une expérience dédiée ne les teste pas.
# %% report
Rapport final de reproduction (écris-le comme pour un collègue). Inclus : (1) l'affirmation en termes opérationnels ; (2) le résultat principal avec les nombres du rapport généré ; (3) le verdict pour chacune des trois prédictions de l'étape 1 ; (4) les deux écarts qui limitent le plus la conclusion ; (5) la prochaine expérience que tu lancerais, avec ses graines et ses facteurs.
# %% md Sources
Uniquement du matériel qui a réellement été ouvert et lu pendant la rédaction de ce projet :

- Leçon `sys.05` (« Research practice: reading, reproducing and ablating »), `content/authored/llm-systems/05-research-practice/lesson.md`, sections « Reproducing a result », « Designing a controlled experiment », l'exemple travaillé moons/warmup, « Reporting honestly » et « Spotting weaknesses » : protocole, appariement par graine, formule du t apparié et règles de rapport.
- `docs/PROJECTS.md` et `docs/AGENT_BRIEF.md` de ce dépôt (format et règles).
- Tout le reste (Adam, rétropropagation de LayerNorm, correction de Holm, effets factoriels, loi de Student) est du matériel de manuel standard écrit de mémoire et vérifié numériquement par les contrôles de ce notebook (différences finies, exemple calculé à la main, valeurs critiques connues). Aucun article sur le warmup ou la normalisation n'a été ouvert pour ce projet : l'affirmation est donc utilisée comme un énoncé à tester, pas comme un résultat cité.
