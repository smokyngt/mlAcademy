# %% meta
{"id": "proj.spirals", "title": "Une étude de réseau de neurones sur des spirales à trois bras", "summary": "Génère un jeu de données en spirale, construis un MLP avec une rétropropagation écrite à la main et une vérification du gradient, implémente SGD, le momentum et Adam, compare-les sur cinq graines, mets en évidence le surapprentissage et ses remèdes sur un très petit jeu d'entraînement, puis diagnostique des entraînements défaillants à partir de leurs courbes de perte.", "level": "intermediate", "minutes": 150, "requires": ["f2n.10", "f2n.11", "dl.01"], "outcomes": ["Écrire les passes avant et arrière d'un MLP multicouche avec régularisation L2 et dropout inversé, et prouver leur exactitude par une vérification du gradient par différences finies", "Implémenter SGD, le momentum et Adam comme optimiseurs agissant sur place, ainsi qu'une boucle d'entraînement par mini-batchs", "Présenter une comparaison d'optimiseurs par la moyenne et l'écart-type sur plusieurs graines plutôt que sur une seule exécution", "Mettre en évidence le surapprentissage sur un petit jeu d'entraînement et mesurer comment L2 et le dropout modifient la précision en test", "Diagnostiquer un entraînement qui diverge, qui stagne ou dont l'initialisation est mauvaise, à partir de sa seule trace de perte"], "stages": ["Données", "Modèle et vérification du gradient", "Optimiseurs et boucle d'entraînement", "Expérience sur les optimiseurs", "Régularisation", "Analyse des échecs", "Rapport"]}
# %% md Bienvenue
# Une étude sur des spirales à trois bras

Trois bras de points s'enroulent autour de l'origine comme un moulinet. Aucune droite (ni aucune couche cachée unique de largeur modeste) ne les sépare bien : c'est donc le plus petit jeu de données où la profondeur, le choix de l'optimiseur et la régularisation comptent visiblement. Tu vas construire chaque élément toi-même avec NumPy : les données, un perceptron multicouche avec rétropropagation manuelle, trois optimiseurs, et les expériences qui les comparent.

Tu t'appuies sur [[f2n.10]] (la rétropropagation avec les formes : pour une couche, `Z = A @ W + b`, et le gradient de l'entropie croisée softmax au niveau des logits est `(P - onehot(y)) / B`), [[f2n.11]] (l'initialisation de He, la boucle d'entraînement, l'échelle de débogage) et [[dl.01]] (le momentum et Adam avec correction du biais).

Conventions utilisées partout :

- Les paramètres forment une seule liste Python à plat `[W0, b0, W1, b1, ..., W_{L-1}, b_{L-1}]`. `W_l` a pour forme `(n_in, n_out)`, `b_l` a pour forme `(n_out,)`.
- Les couches cachées utilisent ReLU, la dernière couche est linéaire et produit les logits, la fonction de perte est l'entropie croisée softmax moyenne.
- Tout est initialisé avec une graine. Chaque cellule s'exécute en quelques secondes.

Travaille de haut en bas. Chaque tâche a une vérification cachée que tu lances avec « Check » ; chaque étape se termine par un court résumé et une question de rapport à réponse libre.

# %% md Étape 1 : les données
## Étape 1 : Données

Le bras `k` (pour `k = 0, 1, 2`) est une courbe : le rayon `r` va de 0,05 à 1 par pas égaux, et l'angle vaut

$$
\theta = \frac{2\pi k}{3} + 7r + \varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma^2),
$$

de sorte que chaque bras fait environ 1,1 tour en s'éloignant du centre, et que les trois bras sont décalés de 120 degrés. Le point est `(r cos(theta), r sin(theta))`. Le bruit angulaire `sigma` (appelé `noise`) fait légèrement se chevaucher les bras voisins vers l'extérieur, ce qui rend le problème non trivial.

# %% task Génère la spirale
import numpy as np
K = 3

def make_spiral(n_per_class, noise=0.3, seed=0):
    """Return X of shape (3*n_per_class, 2) and integer labels y of shape (3*n_per_class,),
    class 0 first, then class 1, then class 2. Use np.random.default_rng(seed)."""
    rng = np.random.default_rng(seed)
    # TODO: for each class k build r = linspace(0.05, 1, n), theta as in the formula,
    #       then stack the points and the labels
    raise NotImplementedError
# %% check
_X, _y = make_spiral(20, 0.3, 1)
check("X has shape (60, 2) and y has shape (60,)", lambda: _X.shape == (60, 2) and _y.shape == (60,), "3 classes x 20 points, 2 coordinates")
check("labels are integers, 20 of each class, ordered by class", lambda: np.issubdtype(_y.dtype, np.integer) and list(np.bincount(_y)) == [20, 20, 20] and np.all(np.diff(_y) >= 0))
check("all radii lie in [0.05, 1]", lambda: np.all(np.hypot(_X[:, 0], _X[:, 1]) <= 1 + 1e-9) and np.all(np.hypot(_X[:, 0], _X[:, 1]) >= 0.05 - 1e-9), "The noise is added to the angle only, so the radius is exactly linspace(0.05, 1, n)")
check("same seed gives identical data, a different seed does not", lambda: np.array_equal(make_spiral(20, 0.3, 1)[0], _X) and not np.array_equal(make_spiral(20, 0.3, 2)[0], _X))
check("with zero noise the first point of arm 1 sits at angle 2*pi/3 + 7*0.05", lambda: np.allclose(make_spiral(20, 0.0, 0)[0][20], 0.05 * np.array([np.cos(2 * np.pi / 3 + 0.35), np.sin(2 * np.pi / 3 + 0.35)])))
# %% hint
Pour chaque classe : `r = np.linspace(0.05, 1, n)`, `t = 2*np.pi*k/K + 7*r + noise*rng.normal(size=n)`, les points `np.c_[r*np.cos(t), r*np.sin(t)]`.
# %% hint
Range les tableaux de chaque classe dans des listes, puis utilise `np.vstack(...)` pour X et `np.concatenate(...)` (ou `np.repeat(np.arange(K), n)`) pour y.
# %% solution
import numpy as np
K = 3

def make_spiral(n_per_class, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for k in range(K):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = 2 * np.pi * k / K + 7.0 * r + noise * rng.normal(size=n_per_class)
        Xs.append(np.c_[r * np.cos(t), r * np.sin(t)])
        ys.append(np.full(n_per_class, k))
    return np.vstack(Xs), np.concatenate(ys)
# %% md Standardisation
Les réseaux de neurones s'entraînent mieux quand les entrées sont à peu près de moyenne nulle et de variance unité pour chaque variable. La règle qui compte : **calcule les statistiques sur le jeu d'entraînement uniquement**, puis applique les mêmes nombres aux données de validation et de test. Les calculer sur toutes les données fait fuir de l'information sur le jeu de test vers l'entraînement.

# %% task Ajuste et applique un standardiseur
def fit_standardiser(X):
    """Return (mu, sd): per-column mean and standard deviation (population std, ddof=0).
    If a column has sd == 0 use 1.0 for that column so we never divide by zero."""
    raise NotImplementedError  # TODO

def apply_standardiser(X, mu, sd):
    """Return (X - mu) / sd."""
    raise NotImplementedError  # TODO
# %% check
_rng = np.random.default_rng(3)
_A = _rng.normal([5.0, -2.0], [2.0, 0.5], size=(500, 2))
_mu, _sd = fit_standardiser(_A)
_Z = apply_standardiser(_A, _mu, _sd)
check("mu and sd have shape (2,)", lambda: np.shape(_mu) == (2,) and np.shape(_sd) == (2,))
check("standardised training data has mean 0 and std 1 per column", lambda: np.allclose(_Z.mean(0), 0, atol=1e-9) and np.allclose(_Z.std(0), 1, atol=1e-9))
check("new data is transformed with the TRAINING statistics", lambda: np.allclose(apply_standardiser(_A + 10, _mu, _sd).mean(0), 10 / _sd, atol=1e-9), "Do not recompute mu and sd inside apply_standardiser")
check("a constant column does not produce nan or inf", lambda: np.all(np.isfinite(apply_standardiser(np.c_[_A[:, 0], np.ones(500)], *fit_standardiser(np.c_[_A[:, 0], np.ones(500)])))), "Replace a zero sd by 1")
# %% hint
`mu = X.mean(axis=0)`, `sd = X.std(axis=0)`, puis `sd = np.where(sd == 0, 1.0, sd)`.
# %% solution
def fit_standardiser(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def apply_standardiser(X, mu, sd):
    return (X - mu) / sd
# %% md Construction des découpages
Nous générons 600 points, nous les mélangeons, et nous en gardons 300 pour l'entraînement et 300 pour le test. Le grand jeu de test rend fiables les écarts de précision entre expériences (l'erreur standard d'une précision proche de 0,9 sur 300 points est d'environ 0,017). Plus loin, nous extrairons aussi de minuscules jeux d'entraînement à partir de nouveaux tirages.

# %% code
X_all, y_all = make_spiral(200, 0.3, 0)
perm = np.random.default_rng(0).permutation(len(y_all))
X_all, y_all = X_all[perm], y_all[perm]
Xtr_raw, ytr, Xte_raw, yte = X_all[:300], y_all[:300], X_all[300:], y_all[300:]
mu, sd = fit_standardiser(Xtr_raw)
Xtr = apply_standardiser(Xtr_raw, mu, sd)
Xte = apply_standardiser(Xte_raw, mu, sd)
print("train", Xtr.shape, np.bincount(ytr), " test", Xte.shape, np.bincount(yte))
print("train mean", Xtr.mean(0).round(3), "std", Xtr.std(0).round(3))
# %% md Résumé de l'étape 1
**Ce que tu as appris.** Un générateur de données est une fonction d'une graine, donc chaque nombre obtenu ensuite est reproductible. La standardisation n'utilise que les statistiques d'entraînement. La spirale est une géométrie volontairement difficile : les classes sont entremêlées, donc un modèle linéaire reste près du hasard (environ un tiers) et le réseau doit courber le plan.

# %% report
En deux ou trois phrases : pourquoi une spirale exige-t-elle un modèle non linéaire, et que pourrait-il mal se passer si tu standardisais avec la moyenne de l'entraînement et du test réunis ?
# %% md Étape 2 : le modèle
## Étape 2 : Le MLP et son gradient

Tu écris maintenant le réseau. Le squelette est la dérivation par batchs d'un réseau à deux couches de [[f2n.10]], généralisée à n'importe quelle profondeur, avec deux extras dont tu auras besoin plus tard : une pénalité L2 et le dropout inversé.

**Passe avant** pour la couche `l` avec l'entrée `A_prev` (de forme `(B, n_in)`) : `Z = A_prev @ W + b`. Les couches cachées appliquent ReLU, `H = max(0, Z)`. Avec une probabilité de dropout `p`, chaque activation cachée est mise à zéro avec la probabilité `p` et les survivantes sont multipliées par `1/(1-p)` (dropout « inversé »), de sorte que l'activation attendue est inchangée et qu'**aucune remise à l'échelle n'est nécessaire au moment du test**. Le dropout n'est actif que lorsque `p_drop > 0` *et* qu'un `rng` est fourni ; sinon le réseau est en mode évaluation.

**Passe arrière** pour l'entropie croisée softmax : `dZ_L = (P - onehot(y)) / B`. Ensuite, pour chaque couche en descendant : `dW = A_prev.T @ dZ`, `db = dZ.sum(0)`, `dA_prev = dZ @ W.T` ; si la couche du dessous avait du dropout, multiplie par son masque ; puis `dZ_prev = dA_prev * (Z_prev > 0)`.

**L2** ajoute `0.5 * l2 * sum(W**2)` sur les matrices de poids (jamais sur les biais) à la fonction de perte, de sorte que le gradient de chaque `W` reçoit `+ l2 * W`.

# %% task Initialisation de He
def init_params(sizes, seed=0):
    """sizes = [n_in, h1, ..., n_out]. Return the flat list [W0, b0, W1, b1, ...].
    W_l ~ Normal(0, 2/n_in) (He), b_l = zeros. Use np.random.default_rng(seed) and draw layers in order."""
    rng = np.random.default_rng(seed)
    # TODO
    raise NotImplementedError
# %% check
_p = init_params([2, 8, 3], 0)
check("returns 2 arrays per layer with shapes (n_in, n_out) and (n_out,)", lambda: len(_p) == 4 and [q.shape for q in _p] == [(2, 8), (8,), (8, 3), (3,)])
check("biases start at zero", lambda: np.all(_p[1] == 0) and np.all(_p[3] == 0))
_big = init_params([400, 500, 3], 1)
check("weight std is sqrt(2/n_in) (within 5%)", lambda: abs(_big[0].std() / np.sqrt(2 / 400) - 1) < 0.05, "He init: standard deviation sqrt(2/fan_in), not variance 2/fan_in")
check("same seed gives the same weights, different seed different weights", lambda: np.array_equal(init_params([2, 8, 3], 0)[0], _p[0]) and not np.array_equal(init_params([2, 8, 3], 1)[0], _p[0]))
# %% hint
`for a, b in zip(sizes[:-1], sizes[1:]): params += [rng.normal(0, np.sqrt(2/a), (a, b)), np.zeros(b)]`
# %% solution
def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    return params
# %% task La passe avant
def forward(params, X, p_drop=0.0, rng=None):
    """Return (logits, cache). cache is a list with one tuple (A_prev, Z, mask) per layer:
    A_prev is the input to the layer (after dropout of the layer below), Z the pre-activation,
    mask the inverted-dropout mask used on this layer's output (None if none / last layer)."""
    L = len(params) // 2
    A = X
    cache = []
    # TODO loop over layers; ReLU on hidden layers; dropout only if p_drop > 0 and rng is not None
    raise NotImplementedError
# %% check
_p = init_params([2, 6, 5, 3], 0)
_x = np.random.default_rng(1).normal(size=(7, 2))
_lg, _c = forward(_p, _x)
def _manual(p, x):
    a = x
    for l in range(len(p) // 2):
        z = a @ p[2 * l] + p[2 * l + 1]
        a = np.maximum(0, z) if l < len(p) // 2 - 1 else z
    return a
check("logits have shape (B, n_classes)", lambda: _lg.shape == (7, 3))
check("evaluation mode equals the plain ReLU network", lambda: np.allclose(_lg, _manual(_p, _x)))
check("cache has one (A_prev, Z, mask) entry per layer", lambda: len(_c) == 3 and all(len(t) == 3 for t in _c) and np.array_equal(_c[0][0], _x))
check("p_drop without an rng is still evaluation mode", lambda: np.allclose(forward(_p, _x, 0.5, None)[0], _lg))
_xb = np.random.default_rng(2).normal(size=(4000, 2)); _pb = init_params([2, 50, 3], 0)
_lgd, _cd = forward(_pb, _xb, 0.4, np.random.default_rng(0))
check("dropout masks only contain 0 and 1/(1-p) and average about 1", lambda: set(np.round(np.unique(_cd[0][2]), 6)) <= {0.0, round(1 / 0.6, 6)} and abs(_cd[0][2].mean() - 1) < 0.02, "mask = (rng.random(H.shape) >= p) / (1 - p)")
check("the last layer has no dropout mask", lambda: _cd[-1][2] is None)
check("training mode changes the logits", lambda: not np.allclose(_lgd, forward(_pb, _xb)[0]))
# %% hint
Pour une couche cachée : `H = np.maximum(0, Z)` ; si on applique le dropout : `mask = (rng.random(H.shape) >= p_drop) / (1 - p_drop); H = H * mask`. Stocke `(A, Z, mask)` et pose `A = H`.
# %% hint
Pour la dernière couche, stocke `(A, Z, None)` et retourne tout de suite `Z, cache`.
# %% solution
def forward(params, X, p_drop=0.0, rng=None):
    L = len(params) // 2
    A = X
    cache = []
    for l in range(L):
        W, b = params[2 * l], params[2 * l + 1]
        Z = A @ W + b
        if l == L - 1:
            cache.append((A, Z, None))
            return Z, cache
        H = np.maximum(0.0, Z)
        mask = None
        if p_drop > 0 and rng is not None:
            mask = (rng.random(H.shape) >= p_drop) / (1.0 - p_drop)
            H = H * mask
        cache.append((A, Z, mask))
        A = H
# %% task Fonction de perte et passe arrière
def loss_and_grads(params, X, y, l2=0.0, p_drop=0.0, rng=None):
    """Return (loss, grads). loss = mean softmax cross-entropy + 0.5*l2*sum of squared weights (not biases).
    grads is a list with the same shapes as params. Compute log-softmax stably (subtract the row max)."""
    logits, cache = forward(params, X, p_drop, rng)
    B = len(y)
    L = len(params) // 2
    # TODO: stable log-softmax, loss, then the backward loop from the last layer down
    raise NotImplementedError
# %% check
def _numgrad(f, p, h=1e-5):
    out = []
    for k, arr in enumerate(p):
        g = np.zeros_like(arr)
        for idx in np.ndindex(*arr.shape):
            q1 = [a.copy() for a in p]; q2 = [a.copy() for a in p]
            q1[k][idx] += h; q2[k][idx] -= h
            g[idx] = (f(q1) - f(q2)) / (2 * h)
        out.append(g)
    return out
_p = init_params([2, 5, 4, 3], 0)
_p = [q + 0.1 * np.random.default_rng(i).normal(size=q.shape) for i, q in enumerate(_p)]
_x = np.random.default_rng(1).normal(size=(6, 2)); _y = np.array([0, 1, 2, 0, 1, 2])
_l, _g = loss_and_grads(_p, _x, _y)
check("loss is a finite scalar and grads match the parameter shapes", lambda: np.ndim(_l) == 0 and np.isfinite(_l) and [g.shape for g in _g] == [q.shape for q in _p])
check("gradients match finite differences (no regularisation)", lambda: all(np.allclose(a, n, atol=1e-7, rtol=1e-5) for a, n in zip(_g, _numgrad(lambda q: loss_and_grads(q, _x, _y)[0], _p))), "Check the 1/B factor and the ReLU mask (Z > 0)")
check("gradients match finite differences with L2 = 0.3", lambda: all(np.allclose(a, n, atol=1e-7, rtol=1e-5) for a, n in zip(loss_and_grads(_p, _x, _y, l2=0.3)[1], _numgrad(lambda q: loss_and_grads(q, _x, _y, l2=0.3)[0], _p))), "L2 adds l2*W to the weight gradients only, not to the biases")
check("the L2 term equals 0.5*l2*sum(W^2)", lambda: abs(loss_and_grads(_p, _x, _y, l2=2.0)[0] - _l - 0.5 * 2.0 * sum((_p[i] ** 2).sum() for i in (0, 2, 4))) < 1e-9)
check("gradients match finite differences with dropout (fixed mask)", lambda: all(np.allclose(a, n, atol=1e-7, rtol=1e-5) for a, n in zip(loss_and_grads(_p, _x, _y, 0.0, 0.3, np.random.default_rng(9))[1], _numgrad(lambda q: loss_and_grads(q, _x, _y, 0.0, 0.3, np.random.default_rng(9))[0], _p))), "Multiply dA by the stored mask before applying the ReLU derivative")
_small = [q * 0.01 for q in _p]
check("tiny weights give loss about ln 3", lambda: abs(loss_and_grads(_small, _x, _y)[0] - np.log(3)) < 0.02)
check("no overflow with huge logits", lambda: np.isfinite(loss_and_grads([q * 50 for q in _p], _x * 20, _y)[0]))
# %% hint
`Zs = logits - logits.max(1, keepdims=True); S = np.exp(Zs).sum(1, keepdims=True); logp = Zs - np.log(S); P = np.exp(logp)`. Perte : `-logp[np.arange(B), y].mean()`.
# %% hint
`dZ = P.copy(); dZ[np.arange(B), y] -= 1; dZ /= B`. Boucle `for l in reversed(range(L))` : `A_prev, Z, mask = cache[l]`, `grads[2*l] = A_prev.T @ dZ + l2 * W`, `grads[2*l+1] = dZ.sum(0)`.
# %% hint
Si `l > 0` : `dA = dZ @ W.T` ; prends le masque de la couche `l-1` (`cache[l-1][2]`) et la pré-activation `cache[l-1][1]` ; `dA = dA * mask` quand un masque existe ; puis `dZ = dA * (Zprev > 0)`.
# %% solution
def loss_and_grads(params, X, y, l2=0.0, p_drop=0.0, rng=None):
    logits, cache = forward(params, X, p_drop, rng)
    B = len(y)
    L = len(params) // 2
    Zs = logits - logits.max(axis=1, keepdims=True)
    logp = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
    P = np.exp(logp)
    loss = -logp[np.arange(B), y].mean()
    loss += 0.5 * l2 * sum((params[2 * l] ** 2).sum() for l in range(L))
    dZ = P.copy()
    dZ[np.arange(B), y] -= 1.0
    dZ /= B
    grads = [None] * len(params)
    for l in reversed(range(L)):
        A_prev, Z, mask = cache[l]
        W = params[2 * l]
        grads[2 * l] = A_prev.T @ dZ + l2 * W
        grads[2 * l + 1] = dZ.sum(axis=0)
        if l > 0:
            dA = dZ @ W.T
            Zprev, mprev = cache[l - 1][1], cache[l - 1][2]
            if mprev is not None:
                dA = dA * mprev
            dZ = dA * (Zprev > 0)
    return float(loss), grads
# %% md Le jalon de la vérification du gradient
N'entraîne jamais une rétropropagation écrite à la main avant qu'elle ait passé une vérification du gradient. La différence centrée pour une entrée de paramètre est $(L(\theta+h)-L(\theta-h))/2h$. La comparer au gradient analytique avec une erreur *relative* $|a-n|/(|a|+|n|+10^{-12})$ rend le test indépendant de l'échelle. Avec `h = 1e-5` en float64, une implémentation correcte donne des erreurs autour de $10^{-8}$ ou mieux.

# %% task Écris le vérificateur de gradient
def grad_check(fn, params, X, y, h=1e-5):
    """fn(params, X, y) -> (loss, grads). Return the MAXIMUM relative error over every entry of every parameter.
    Must not modify `params` when it returns (restore each entry after perturbing it)."""
    # TODO: get analytic grads once, then perturb every entry by +h and -h
    raise NotImplementedError
# %% check
_p = [q + 0.1 * np.random.default_rng(i).normal(size=q.shape) for i, q in enumerate(init_params([2, 5, 3], 0))]
_x = np.random.default_rng(1).normal(size=(6, 2)); _y = np.array([0, 1, 2, 0, 1, 2])
_copy = [q.copy() for q in _p]
_e = grad_check(loss_and_grads, _p, _x, _y)
def _bug_scale(p, X, y):
    l, g = loss_and_grads(p, X, y); g = list(g); g[0] = g[0] * 1.1; return l, g
def _bug_bias(p, X, y):
    l, g = loss_and_grads(p, X, y); g = list(g); g[3] = np.zeros_like(g[3]); return l, g
check("returns a small float for a correct gradient", lambda: np.ndim(_e) == 0 and 0 <= _e < 1e-6, "Expect about 1e-8; the denominator needs |a|+|n|+1e-12")
check("detects a gradient that is 10% too large in one matrix", lambda: grad_check(_bug_scale, _p, _x, _y) > 1e-3)
check("detects a missing bias gradient", lambda: grad_check(_bug_bias, _p, _x, _y) > 1e-3)
check("leaves the parameters exactly as it found them", lambda: all(np.array_equal(a, b) for a, b in zip(_p, _copy)))
# %% hint
Boucle `for k, arr in enumerate(params): for idx in np.ndindex(*arr.shape):` sauvegarde `old = arr[idx]`, pose `arr[idx] = old + h`, évalue `fn(...)[0]`, puis `old - h`, puis restaure `arr[idx] = old`.
# %% solution
def grad_check(fn, params, X, y, h=1e-5):
    _, grads = fn(params, X, y)
    worst = 0.0
    for k, arr in enumerate(params):
        for idx in np.ndindex(*arr.shape):
            old = arr[idx]
            arr[idx] = old + h
            lp = fn(params, X, y)[0]
            arr[idx] = old - h
            lm = fn(params, X, y)[0]
            arr[idx] = old
            num = (lp - lm) / (2 * h)
            a = grads[k][idx]
            worst = max(worst, abs(a - num) / (abs(a) + abs(num) + 1e-12))
    return float(worst)
# %% code
_tiny = init_params([2, 8, 8, 3], 0)
# non-zero biases: with all biases at 0 a ReLU input can sit exactly on its kink, where finite differences are meaningless
_tiny = [q + 0.1 * np.random.default_rng(i).normal(size=q.shape) for i, q in enumerate(_tiny)]
print("gradient check on a 2-8-8-3 network:", grad_check(loss_and_grads, _tiny, Xtr[:10], ytr[:10]))
print("initial loss on the training set:", round(loss_and_grads(init_params([2, 32, 32, 3], 0), Xtr, ytr)[0], 3), " (ln 3 =", round(np.log(3), 3), ")")
# %% md Résumé de l'étape 2
**Ce que tu as appris.** Une passe arrière indépendante de la profondeur est une boucle qui fait descendre `dZ` ; c'est le cache qui décide de ce que tu peux calculer. Une vérification du gradient transforme « je crois que les maths sont justes » en un nombre que tu peux afficher. Avec une initialisation raisonnable, la perte initiale est proche de $\ln K$ (un premier barreau de l'échelle de débogage de [[f2n.11]]).

# %% report
Quel bug unique une vérification du gradient détecterait-elle alors qu'une courbe de perte qui descend ne le ferait pas ? Donne un exemple concret (par exemple un terme L2 mal dérivé).
# %% md Étape 3 : les optimiseurs
## Étape 3 : Les optimiseurs et la boucle d'entraînement

Chaque optimiseur est un objet avec `step(params, grads)` qui **met à jour les tableaux sur place** et garde son propre état par paramètre, créé paresseusement au premier appel (les tableaux d'état ont exactement la forme de chaque paramètre). Les règles de mise à jour (issues de [[dl.01]]) :

- SGD avec momentum : $v \leftarrow \mu v - \eta g$, $\theta \leftarrow \theta + v$. Avec $\mu = 0$, c'est du SGD simple.
- Adam : $m \leftarrow \beta_1 m + (1-\beta_1) g$, $v \leftarrow \beta_2 v + (1-\beta_2) g^2$, $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$, $\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\varepsilon)$ où $t$ compte les pas à partir de 1.

# %% task SGD avec momentum
class SGD:
    def __init__(self, lr, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, None
    def step(self, params, grads):
        """Update every array in `params` IN PLACE (use +=, -=, not rebinding)."""
        # TODO create self.v on first call; v = mu*v - lr*g ; p += v
        raise NotImplementedError
# %% check
_a = np.array([1.0, 2.0]); _b = np.array([[1.0]])
_o = SGD(0.1)
_o.step([_a, _b], [np.array([1.0, -1.0]), np.array([[2.0]])])
check("plain SGD: p <- p - lr*g", lambda: np.allclose(_a, [0.9, 2.1]) and np.allclose(_b, [[0.8]]))
_p1 = np.array([1.0]); _m = SGD(0.1, 0.9)
_m.step([_p1], [np.array([1.0])]); _first = _p1.copy(); _m.step([_p1], [np.array([1.0])])
check("momentum after one step", lambda: np.allclose(_first, [0.9]))
check("momentum after two steps (v2 = -0.19, p = 0.71)", lambda: np.allclose(_p1, [0.71]), "v = mu*v - lr*g, then p += v")
_arr = np.zeros(3); _ident = id(_arr); SGD(0.5).step([_arr], [np.ones(3)])
check("the parameter array is modified in place", lambda: id(_arr) == _ident and np.allclose(_arr, -0.5))
_m2 = SGD(0.1, 0.9); _pa, _pb = np.zeros(2), np.zeros((2, 2))
_m2.step([_pa, _pb], [np.ones(2), np.ones((2, 2))]); _m2.step([_pa, _pb], [np.ones(2), np.ones((2, 2))])
check("each parameter keeps its own velocity (shapes preserved)", lambda: np.allclose(_pa, -0.29) and np.allclose(_pb, -0.29) and _pb.shape == (2, 2))
# %% hint
`if self.v is None: self.v = [np.zeros_like(p) for p in params]` ; puis `for p, g, v in zip(params, grads, self.v): v *= self.mu; v -= self.lr * g; p += v`.
# %% solution
class SGD:
    def __init__(self, lr, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, None
    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.mu
            v -= self.lr * g
            p += v
# %% task Adam
class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t, self.m, self.v = 0, None, None
    def step(self, params, grads):
        """In-place Adam update with bias correction."""
        # TODO
        raise NotImplementedError
# %% check
def _adam_ref(g_list, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    p = np.array([0.0, 0.0]); m = np.zeros(2); v = np.zeros(2)
    for t, g in enumerate(g_list, 1):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return p
_gs = [np.array([5.0, -0.02]), np.array([1.0, 0.3]), np.array([-2.0, 0.1])]
_pp = np.zeros(2); _ad = Adam(0.01)
_ad.step([_pp], [_gs[0]]); _one = _pp.copy()
for _g in _gs[1:]:
    _ad.step([_pp], [_g])
check("the first step moves each coordinate by about lr, whatever the gradient scale", lambda: np.allclose(_one, [-0.01, 0.01], atol=1e-6), "At t=1, m_hat = g and v_hat = g^2, so the step is lr*g/(|g|+eps)")
check("three steps match an independent reference implementation", lambda: np.allclose(_pp, _adam_ref(_gs), atol=1e-10))
_q1 = np.zeros(2); _q2 = np.zeros(2); Adam(0.01).step([_q1], [np.array([1.0, 1.0])]); Adam(0.01).step([_q2], [np.array([1000.0, 1000.0])])
check("scaling the gradient by 1000 does not change the first step", lambda: np.allclose(_q1, _q2, atol=1e-6))
_z = np.zeros(2); _idz = id(_z); _a2 = Adam(0.1); _a2.step([_z], [np.ones(2)])
check("updates in place and counts steps", lambda: id(_z) == _idz and _a2.t == 1 and np.all(_z < 0))
_pm, _pv = np.zeros(2), np.zeros((3, 2)); _a3 = Adam(0.1); _a3.step([_pm, _pv], [np.ones(2), np.ones((3, 2))])
check("state arrays match the parameter shapes", lambda: _a3.m[1].shape == (3, 2) and _a3.v[0].shape == (2,))
# %% hint
Au premier appel, crée `self.m` et `self.v` comme des listes de `np.zeros_like(p)`. Incrémente `self.t` une fois par appel (pas une fois par paramètre).
# %% hint
Correction du biais : `mh = m / (1 - self.b1 ** self.t)`, `vh = v / (1 - self.b2 ** self.t)`, puis `p -= self.lr * mh / (np.sqrt(vh) + self.eps)`.
# %% solution
class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t, self.m, self.v = 0, None, None
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
# %% md La boucle d'entraînement
Nous évaluons avec `accuracy`, qui fait tourner le réseau en mode évaluation (sans dropout). La boucle d'entraînement mélange les données à chaque époque avec son propre générateur initialisé par une graine, parcourt des mini-batchs et enregistre un historique par époque. Le même générateur fournit les masques de dropout, si bien qu'une exécution est une fonction pure de `(params, seed, ...)`.

# %% code
def accuracy(params, X, y):
    return float((forward(params, X)[0].argmax(axis=1) == y).mean())
# %% task La boucle d'entraînement
def train(params, X, y, Xv, yv, opt, epochs, batch=32, seed=0, l2=0.0, p_drop=0.0):
    """Mini-batch training. Work on COPIES of the parameters (do not modify the caller's list).
    Each epoch: draw a fresh permutation from rng = np.random.default_rng(seed) (created once, before the loop),
    iterate over consecutive slices of `batch` indices (the last one may be shorter), call loss_and_grads with
    l2, p_drop and the same rng, then opt.step.
    Return (trained_params, hist) where hist = {"loss": [...], "train_acc": [...], "val_acc": [...]}
    with one entry per epoch: the mean mini-batch loss, and accuracies on (X, y) and (Xv, yv)."""
    # TODO
    raise NotImplementedError
# %% check
class _Counter:
    def __init__(self): self.n = 0
    def step(self, params, grads): self.n += 1
_p0 = init_params([2, 16, 3], 0); _keep = [q.copy() for q in _p0]
_cnt = _Counter(); train(_p0, Xtr[:100], ytr[:100], Xte, yte, _cnt, epochs=3, batch=32)
check("the optimiser is called ceil(N/batch) times per epoch", lambda: _cnt.n == 3 * 4, "100 points with batch 32 gives 4 batches (the last has 4 points)")
check("the caller's parameters are not modified", lambda: all(np.array_equal(a, b) for a, b in zip(_p0, _keep)))
_pt, _h = train(_p0, Xtr, ytr, Xte, yte, Adam(0.01), epochs=25, batch=32, seed=0)
check("history has one entry per epoch for loss, train_acc, val_acc", lambda: set(_h) == {"loss", "train_acc", "val_acc"} and all(len(_h[k]) == 25 for k in _h))
check("training loss goes down substantially", lambda: _h["loss"][-1] < 0.6 * _h["loss"][0], "Check that the optimiser really updates the copies you pass to loss_and_grads")
check("it beats chance clearly on the test set", lambda: _h["val_acc"][-1] > 0.6)
_pt2, _h2 = train(_p0, Xtr, ytr, Xte, yte, Adam(0.01), epochs=25, batch=32, seed=0)
check("the same seed reproduces the run exactly", lambda: _h["loss"] == _h2["loss"] and all(np.array_equal(a, b) for a, b in zip(_pt, _pt2)))
check("returned parameters have the original shapes", lambda: [q.shape for q in _pt] == [q.shape for q in _p0])
# %% hint
`params = [p.copy() for p in params]; rng = np.random.default_rng(seed)`. À chaque époque : `perm = rng.permutation(len(y))`, puis `for i in range(0, len(y), batch): idx = perm[i:i+batch]`.
# %% hint
`loss, grads = loss_and_grads(params, X[idx], y[idx], l2, p_drop, rng); opt.step(params, grads)`. Ajoute à la fin de chaque époque la moyenne des pertes des batchs et les deux précisions.
# %% solution
def train(params, X, y, Xv, yv, opt, epochs, batch=32, seed=0, l2=0.0, p_drop=0.0):
    params = [p.copy() for p in params]
    rng = np.random.default_rng(seed)
    N = len(y)
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        perm = rng.permutation(N)
        tot, nb = 0.0, 0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            loss, grads = loss_and_grads(params, X[idx], y[idx], l2, p_drop, rng)
            opt.step(params, grads)
            tot += loss
            nb += 1
        hist["loss"].append(tot / nb)
        hist["train_acc"].append(accuracy(params, X, y))
        hist["val_acc"].append(accuracy(params, Xv, yv))
    return params, hist
# %% md Résumé de l'étape 3
**Ce que tu as appris.** Un optimiseur est un petit objet à état ; cet état a la forme des paramètres ; les mises à jour sur place sont ce qui permet à `loss_and_grads` et à `step` de se composer. La boucle d'entraînement ne fait que 10 lignes, mais ses détails (copier les paramètres, un seul rng, un dernier batch plus court) décident si une exécution est reproductible.

# %% report
Le premier pas d'Adam a une taille d'environ le taux d'apprentissage dans chaque coordonnée, quelle que soit l'échelle du gradient. Pourquoi la correction du biais produit-elle cela, et quand pourrait-elle nuire ?
# %% md Étape 4 : l'expérience sur les optimiseurs
## Étape 4 : Comparer les optimiseurs équitablement

Une seule exécution ne prouve rien : une autre graine change à la fois les poids initiaux et l'ordre des batchs. Nous comparons le **SGD simple** (lr 0,1), le **SGD avec momentum** (lr 0,05, mu 0,9) et **Adam** (lr 0,01) sur un réseau 2-32-32-3 pendant 40 époques, chacun avec 5 graines, et nous présentons la précision finale en test sous la forme moyenne +- écart-type d'échantillon (ddof = 1). Les taux d'apprentissage ont été choisis par un balayage grossier pour que chaque méthode soit raisonnablement réglée, mais ils ne sont pas optimaux : considère le tableau comme une comparaison à budget fixé, et non comme un classement des algorithmes en général.

# %% task Résume les exécutions en moyenne +- écart-type
def summarise(results):
    """results: dict name -> list of final accuracies (one per seed).
    Return a dict name -> (mean, std) with std the SAMPLE standard deviation (ddof=1; 0.0 if only one value),
    ordered by decreasing mean."""
    # TODO
    raise NotImplementedError
# %% check
_r = {"a": [0.80, 0.90, 1.00], "c": [0.5], "b": [0.95, 0.95, 0.95]}
_s = summarise(_r)
check("keys are ordered by decreasing mean", lambda: list(_s) == ["b", "a", "c"], "Sort the items by mean, descending, and build a new dict")
check("means are correct", lambda: np.allclose([_s[k][0] for k in _s], [0.95, 0.90, 0.5]))
check("std uses ddof=1", lambda: abs(_s["a"][1] - 0.1) < 1e-12 and abs(_s["b"][1]) < 1e-12)
check("a single seed has std 0.0", lambda: _s["c"][1] == 0.0)
check("returns plain floats", lambda: all(isinstance(_s[k][0], float) and isinstance(_s[k][1], float) for k in _s))
# %% hint
`np.std(v, ddof=1)` vaut nan pour une seule valeur, donc traite à part le cas `len(v) == 1`.
# %% solution
def summarise(results):
    rows = []
    for name, vals in results.items():
        v = np.asarray(vals, dtype=float)
        std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append((name, float(v.mean()), std))
    rows.sort(key=lambda r: -r[1])
    return {n: (m, s) for n, m, s in rows}
# %% code
OPTS = {"SGD": lambda: SGD(0.1), "Momentum": lambda: SGD(0.05, 0.9), "Adam": lambda: Adam(0.01)}
SEEDS = [0, 1, 2, 3, 4]
runs = {name: [] for name in OPTS}
curves = {name: [] for name in OPTS}
for name, mk in OPTS.items():
    for s in SEEDS:
        _, h = train(init_params([2, 32, 32, 3], s), Xtr, ytr, Xte, yte, mk(), epochs=40, batch=32, seed=s)
        runs[name].append(h["val_acc"][-1]); curves[name].append(h["loss"])
table = summarise(runs)
print("optimiser    test accuracy (mean +- std over 5 seeds)")
for name, (m, s) in table.items():
    print(f"{name:<10}   {m:.3f} +- {s:.3f}   per seed: {np.round(runs[name], 3).tolist()}")
spark = " .:-=+*#%@"
for name in OPTS:
    c = np.mean(curves[name], axis=0)
    idx = np.clip(((c - c.min()) / (c.max() - c.min() + 1e-12) * 9).astype(int), 0, 9)
    print(f"{name:<10}   loss {c[0]:.2f} -> {c[-1]:.2f}  |{''.join(spark[9 - i] for i in idx)}|")
# %% md Résumé de l'étape 4
**Ce que tu as appris.** Un résultat est une distribution sur des graines, pas un nombre. Pour lire le tableau : regarde l'écart entre les moyennes par rapport aux écarts-types avant de déclarer un vainqueur. Le momentum et Adam réduisent en général la perte bien plus vite que le SGD simple pour le même budget d'époques ; le SGD simple avec un taux d'apprentissage modeste n'a tout simplement pas fini.

# %% report
À partir de ton tableau : quel optimiseur a gagné, de combien par rapport à la dispersion d'une graine à l'autre, et que changerais-tu pour rendre la comparaison plus équitable (balayages de taux d'apprentissage par méthode, plus d'époques, plus de graines) ?
# %% md Étape 5 : la régularisation
## Étape 5 : Surapprentissage et remède

Avec 300 points d'entraînement, le réseau voit une grande partie de la spirale. Donne-lui maintenant seulement **15 points par classe (45 au total)**, un réseau 2-64-64-3 (environ 4 600 paramètres, cent fois plus que de points de données), Adam en batch complet pendant 200 époques, et évalue sur le même jeu de test de 300 points. Il va mémoriser le jeu d'entraînement. Nous essayons ensuite la décroissance de poids L2 et le dropout. Les moyennes portent sur 3 graines ; chaque graine utilise un petit tirage d'entraînement différent.

# %% task Lance l'étude de régularisation
def small_train(seed):
    X, y = make_spiral(15, 0.3, 100 + seed)
    return apply_standardiser(X, mu, sd), y

# one entry per configuration. TODO: replace the None values with sensible numbers
# (l2 in [1e-3, 3e-2] is a good range for this problem; dropout probability between 0.05 and 0.5)
REG_CONFIGS = {
    "none":    {"l2": 0.0,  "p_drop": 0.0},
    "l2":      {"l2": None, "p_drop": 0.0},
    "dropout": {"l2": 0.0,  "p_drop": None},
}

def reg_study(configs, seeds=(0, 1, 2)):
    """For every config and seed: train a 2-64-64-3 network (init seed = seed) on small_train(seed) with
    Adam(0.01), 200 epochs, batch = whole training set, training seed = seed, passing l2 and p_drop.
    Return {name: {"train": [acc per seed], "test": [acc per seed]}} using the final epoch's accuracies,
    the test accuracy measured on the global Xte, yte."""
    # TODO
    raise NotImplementedError
# %% check
_res = reg_study(REG_CONFIGS)
_mean = lambda n, k: float(np.mean(_res[n][k]))
check("all three configurations were run on 3 seeds", lambda: set(_res) == {"none", "l2", "dropout"} and all(len(_res[n]["test"]) == 3 and len(_res[n]["train"]) == 3 for n in _res))
check("the unregularised network memorises the training set (train acc >= 0.97)", lambda: _mean("none", "train") >= 0.97)
check("and generalises worse than it fits (gap of at least 0.08)", lambda: _mean("none", "train") - _mean("none", "test") >= 0.08, "If the gap is small, check you trained on the 45 small points, not on Xtr")
check("L2 weight decay improves the test accuracy by at least 0.02", lambda: _mean("l2", "test") >= _mean("none", "test") + 0.02, "Try l2 between 1e-3 and 1e-2")
check("L2 and dropout strengths are in sensible ranges", lambda: 1e-4 <= REG_CONFIGS["l2"]["l2"] <= 0.05 and 0.0 < REG_CONFIGS["dropout"]["p_drop"] <= 0.5)
check("dropout does not wreck the model (test acc within 0.08 of unregularised)", lambda: _mean("dropout", "test") >= _mean("none", "test") - 0.08, "Dropout probability 0.5 is far too strong for such a small network")
# %% hint
Boucle `for name, cfg in configs.items(): for s in seeds:` récupère `Xs, ys = small_train(s)`, `params = init_params([2, 64, 64, 3], s)`, puis `train(params, Xs, ys, Xte, yte, Adam(0.01), 200, batch=len(ys), seed=s, l2=cfg["l2"], p_drop=cfg["p_drop"])`.
# %% hint
Dans le balayage qui a produit cette tâche, l2 = 6e-3 a donné la meilleure précision en test et un dropout de 0,1 a très peu aidé : le dropout n'est pas un remède universel, surtout dans un espace d'entrée de faible dimension.
# %% solution
def small_train(seed):
    X, y = make_spiral(15, 0.3, 100 + seed)
    return apply_standardiser(X, mu, sd), y

REG_CONFIGS = {
    "none":    {"l2": 0.0,   "p_drop": 0.0},
    "l2":      {"l2": 6e-3,  "p_drop": 0.0},
    "dropout": {"l2": 0.0,   "p_drop": 0.1},
}

def reg_study(configs, seeds=(0, 1, 2)):
    out = {}
    for name, cfg in configs.items():
        out[name] = {"train": [], "test": []}
        for s in seeds:
            Xs, ys = small_train(s)
            _, h = train(init_params([2, 64, 64, 3], s), Xs, ys, Xte, yte, Adam(0.01), 200,
                         batch=len(ys), seed=s, l2=cfg["l2"], p_drop=cfg["p_drop"])
            out[name]["train"].append(h["train_acc"][-1])
            out[name]["test"].append(h["val_acc"][-1])
    return out
# %% code
print("config      train acc          test acc")
for name, r in reg_study(REG_CONFIGS).items():
    print(f"{name:<10}  {np.mean(r['train']):.3f} +- {np.std(r['train'], ddof=1):.3f}   {np.mean(r['test']):.3f} +- {np.std(r['test'], ddof=1):.3f}")
# %% md Résumé de l'étape 5
**Ce que tu as appris.** La signature du surapprentissage est un écart entre l'entraînement et le test, pas une mauvaise perte d'entraînement. L2 réduit les poids pour que la frontière de décision reste plus lisse ; cela coûte un peu de précision en entraînement et rapporte de la précision en test. Le dropout, qui brille dans les réseaux larges et surparamétrés sur des entrées de grande dimension, était presque sans effet ici. Rapporte ce que tu as mesuré, pas ce que dit le manuel.

# %% report
Explique avec tes propres mots pourquoi L2 a légèrement diminué la précision en entraînement tout en augmentant celle en test. Le dropout s'est-il comporté comme tu l'attendais ? Propose une raison pour laquelle il pourrait aider moins sur un problème en 2 dimensions.
# %% md Étape 6 : analyse des échecs
## Étape 6 : Quand l'entraînement tourne mal

Quatre exécutions du même réseau 2-64-64-3 pendant 30 époques, en SGD, sur tout le jeu d'entraînement. Elles ne diffèrent que par un seul réglage chacune. Leurs pertes d'entraînement par époque sont stockées dans `traces`. Étudie l'affichage avant d'écrire du code.

# %% code
def _trace(lr=0.1, scale=1.0):
    p = init_params([2, 64, 64, 3], 0)
    p = [q * scale if q.ndim == 2 else q for q in p]
    with np.errstate(all="ignore"):
        _, h = train(p, Xtr, ytr, Xte, yte, SGD(lr), epochs=30, batch=32, seed=0)
    return h["loss"]
traces = {
    "run A (lr 10)":         _trace(lr=10.0),
    "run B (weights x0)":    _trace(scale=0.0),
    "run C (weights x10)":   _trace(scale=10.0),
    "run D (the reference)": _trace(),
}
for name, t in traces.items():
    print(f"{name:<24}", np.array2string(np.array(t)[[0, 1, 2, 5, 10, 20, 29]], precision=3, floatmode="maxprec"))
# %% task Diagnostique une trace de perte
def diagnose(losses, n_classes=3):
    """Classify a list of per-epoch training losses into exactly one of
    'diverged'  : any non-finite value, or the final loss is above the first one;
    'bad_init'  : (not diverged) the first loss is above 3*ln(n_classes), i.e. the network started absurdly confident and wrong;
    'stuck'     : (neither of the above) the final loss is still above 0.95 * the first loss;
    'healthy'   : otherwise. Apply the rules in this order."""
    # TODO
    raise NotImplementedError
# %% check
check("the real traces: lr 10 is 'diverged'", lambda: diagnose(traces["run A (lr 10)"]) == "diverged", "NaN appears when the loss overflows: use np.isfinite")
check("the real traces: zero weights are 'stuck' (loss stays at ln 3)", lambda: diagnose(traces["run B (weights x0)"]) == "stuck", "With all weights zero every hidden unit is dead and the weight gradients vanish")
check("the real traces: weights x10 is 'bad_init'", lambda: diagnose(traces["run C (weights x10)"]) == "bad_init")
check("the real traces: the reference is 'healthy'", lambda: diagnose(traces["run D (the reference)"]) == "healthy")
check("synthetic traces follow the rule order", lambda: [diagnose(t) for t in ([1.1, 0.9, 2.0], [1.1, float("inf")], [1.1, 1.09, 1.08], [50.0, 10.0, 2.0], [1.2, 0.8, 0.4], [float("nan")])] == ["diverged", "diverged", "stuck", "bad_init", "healthy", "diverged"])
check("n_classes changes the bad_init threshold", lambda: diagnose([4.0, 2.0, 1.0], n_classes=3) == "bad_init" and diagnose([4.0, 2.0, 1.0], n_classes=10) == "healthy")
# %% hint
Utilise `np.asarray(losses, float)` ; `if not np.all(np.isfinite(a)) or a[-1] > a[0]: return "diverged"`. Compare ensuite `a[0]` avec `3*np.log(n_classes)`, puis `a[-1]` avec `0.95*a[0]`.
# %% solution
def diagnose(losses, n_classes=3):
    a = np.asarray(losses, dtype=float)
    if not np.all(np.isfinite(a)) or a[-1] > a[0]:
        return "diverged"
    if a[0] > 3 * np.log(n_classes):
        return "bad_init"
    if a[-1] > 0.95 * a[0]:
        return "stuck"
    return "healthy"
# %% code
for name, t in traces.items():
    print(f"{name:<24} -> {diagnose(t)}")
# %% md Pourquoi chaque échec se produit
- **lr 10 (diverged).** Le pas $\eta g$ dépasse le minimum de plus que la distance qui l'en sépare, donc la perte augmente à chaque pas jusqu'à ce que les nombres débordent en `nan`. Le remède est de baisser le taux d'apprentissage (l'échelle de débogage : balaie $\{10^{-3},\dots,1\}$ et prends la plus grande valeur stable).
- **Poids tous nuls (stuck).** Chaque activation cachée vaut 0, donc le gradient qui atteint les matrices de poids est nul (seul le dernier biais bouge). La symétrie n'est jamais brisée. Remède : une initialisation aléatoire.
- **Poids x10 (bad init).** Les logits démarrent énormes, donc la première perte se chiffre en centaines. Le SGD s'en remet souvent, mais gaspille des époques ; la perte au pas 0 devrait être proche de $\ln K$, donc mets l'initialisation à l'échelle de $\sqrt{2/n_{in}}$.
- **Limite des règles.** Ton classifieur n'utilise que la trace de perte. Une exécution avec lr = 3 oscille ici autour de 1,0 sans diverger et serait étiquetée `bad_init` par les seuils, parce que sa première époque est élevée. Les traces de perte seules ne séparent pas tous les échecs ; ajoute les normes de gradient et une vérification du gradient.

# %% report
Choisis un échec et décris le symptôme exact que tu as vu dans la trace, la cause, et le seul changement que tu ferais en premier. Dis ensuite quel autre diagnostic (norme du gradient, statistiques des activations, surapprentissage sur 8 exemples) tu ajouterais à tes règles.
# %% md Étape 7 : le rapport
## Étape 7 : Rédige ton rapport

# %% report
Rédige un rapport d'une page (environ 250 mots) avec ces titres : **Question** (qu'as-tu cherché à comparer), **Dispositif** (données, réseau, optimiseurs, graines), **Résultats** (cite ton propre tableau : moyenne +- écart-type pour les trois optimiseurs, et les précisions d'entraînement et de test de l'étude de régularisation), **Modes d'échec** (ce que tu as appris des traces de perte), **Limites** (ce que les expériences ne peuvent pas te dire : taux d'apprentissage réglés grossièrement, un seul jeu de données, 3 et 5 graines). Termine par la seule expérience que tu lancerais ensuite.
# %% md Sources
## Sources

Uniquement le matériel qui a réellement été ouvert pendant la rédaction de ce projet :

- Leçon `[[f2n.10]]` (Rétropropagation, dérivée avec les formes) : `content/authored/ml/10-backpropagation/lesson.md` (liste des sections) et son exercice `exercises/mlp-backprop.exercise.txt`, pour la notation avant/arrière, le gradient `(P - onehot(y)) / B` et la vérification par différences finies. LU.
- Leçon `[[f2n.11]]` (Entraîner un réseau à partir de zéro) : `content/authored/ml/11-training-from-scratch/lesson.md` pour l'initialisation de He, la boucle d'entraînement, le momentum et l'échelle de débogage. LU.
- Leçon `[[dl.01]]` (Optimiseurs et plannings de taux d'apprentissage) : `content/authored/deep-learning/01-optimisers/lesson.md` pour la mise à jour d'Adam, la correction du biais et la propriété du premier pas. LU.
- `docs/PROJECTS.md`, `docs/AGENT_BRIEF.md`, `docs/AUTHORING.md` : format et règles. LU.

Tous les nombres affichés par le projet sont produits par le code qu'il contient ; aucun n'est cité d'une source.
