# %% meta
{"id": "proj.tinygpt", "title": "Construire un petit moteur d'inférence GPT", "summary": "Implémente tout le chemin d'inférence d'un transformer décodeur seul en NumPy : tokenizer d'octets, RoPE, attention multi-têtes causale, RMSNorm, SwiGLU, échantillonnage, un KV cache dont tu prouves qu'il est identique au chemin sans cache, comptabilité des FLOP et de la mémoire, batching avec padding et un benchmark de latence simulé.", "level": "expert", "minutes": 180, "requires": ["trf.01", "trf.02", "trf.03", "trf.04", "llm.01", "llm.02", "sys.01"], "outcomes": ["Écrire chaque opérateur d'un transformer décodeur seul moderne avec des formes de tenseurs exactes et tester chacun contre une propriété indépendante", "Compter les paramètres de façon analytique et les faire correspondre aux tableaux que tu as alloués", "Implémenter l'échantillonnage glouton, par température, top-k et top-p et vérifier leur masse de probabilité", "Construire un KV cache et prouver qu'il change la vitesse mais pas une seule sortie", "Comptabiliser les FLOP et les octets et expliquer pourquoi le décodage est limité par la mémoire", "Regrouper en batch des prompts de longueurs variables avec des masques de padding et des identifiants de position sans changer aucune réponse"], "stages": ["Tokenizer et configuration", "Briques de base", "La passe avant complète", "Échantillonnage et génération", "KV cache", "Comptabilité des coûts", "Batching", "Benchmark et rapport"]}
# %% md Étape 1 : tokenizer et configuration
## Ce que tu construis

Un moteur d'inférence prend une séquence d'entiers, la fait passer dans un ensemble fixe de poids et renvoie une distribution de probabilité sur l'entier suivant. L'entraînement est le travail de quelqu'un d'autre : ici, les poids sont **aléatoires mais fixés par une graine**, donc le modèle raconte n'importe quoi. C'est voulu. L'objectif est la **correction et l'ingénierie** : chaque fonction que tu écris possède une propriété testable sans savoir si le modèle est bon (causalité, équivalence entre deux chemins de code, nombre exact de paramètres).

L'architecture est le décodeur moderne « façon Llama » : pré-normalisation avec RMSNorm, embeddings de position rotatifs (RoPE), attention causale multi-têtes, un bloc feed-forward SwiGLU et une tête de sortie non liée (untied).

Tout vit dans un seul espace de noms partagé, comme dans un notebook. Les tâches que tu termines restent définies pour les cellules suivantes. Noms des dimensions utilisés partout : `B` batch, `T` longueur de séquence, `d` largeur du modèle, `H` têtes, `dh = d/H` largeur d'une tête, `V` taille du vocabulaire, `ff` largeur cachée du MLP.

Ta première étape est le tokenizer le plus simple capable de représenter n'importe quel texte : les octets UTF-8 bruts. 256 valeurs d'octet plus trois identifiants spéciaux.
# %% code
import numpy as np

CFG = dict(vocab=259, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_len=64, eps=1e-6, rope_base=10000.0)
PAD, BOS, EOS = 256, 257, 258          # ids 0..255 are raw bytes
assert CFG["d_model"] % CFG["n_heads"] == 0
print("head width dh =", CFG["d_model"] // CFG["n_heads"])
# %% task Tokenizer au niveau de l'octet : encode et decode
# encode(text, add_bos=True, add_eos=False) -> list[int]
#   UTF-8 bytes of `text` as ints 0..255, optionally preceded by BOS and/or followed by EOS.
# decode(ids) -> str
#   keep only ids < 256, rebuild the bytes, decode as UTF-8 with errors="replace".
def encode(text, add_bos=True, add_eos=False):
    raise NotImplementedError   # TODO

def decode(ids):
    raise NotImplementedError   # TODO
# %% check
check("round trip with multi-byte characters", lambda: decode(encode("héllo wörld ✓ 日本")) == "héllo wörld ✓ 日本", "decode(encode(s)) must give s back, including non-ASCII")
check("BOS is prepended, bytes are raw", lambda: encode("A") == [BOS, 65] and encode("A", add_bos=False) == [65], "'A' is byte 65")
check("one id per UTF-8 byte (é is two bytes)", lambda: len(encode("é", add_bos=False)) == 2 and len(encode("✓", add_bos=False)) == 3, "work on text.encode('utf-8'), not on characters")
check("EOS appended and specials are dropped by decode", lambda: encode("hi", add_bos=True, add_eos=True)[-1] == EOS and decode([BOS, 104, 105, PAD, EOS]) == "hi", "filter ids >= 256 before decoding")
check("a truncated multi-byte sequence does not crash", lambda: "�" in decode([0xE2, 0x9C]), "use errors='replace'")
# %% hint
Une chaîne est une séquence de caractères, mais le modèle voit des octets. `list(text.encode("utf-8"))` donne les valeurs des octets.
# %% hint
decode : `bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")`.
# %% solution
def encode(text, add_bos=True, add_eos=False):
    ids = list(text.encode("utf-8"))
    if add_bos:
        ids = [BOS] + ids
    if add_eos:
        ids = ids + [EOS]
    return ids

def decode(ids):
    return bytes(int(i) for i in ids if int(i) < 256).decode("utf-8", errors="replace")
# %% md Paramètres d'un décodeur
Chaque couche contient : un gain RMSNorm d'attention (`d`), quatre matrices d'attention `Wq, Wk, Wv, Wo` (chacune `d x d`), un gain RMSNorm du MLP (`d`), et trois matrices SwiGLU `W_gate, W_up` (`d x ff`) et `W_down` (`ff x d`). En dehors des couches : la table d'embedding `V x d`, un gain RMSNorm final `d`, et la tête de sortie `d x V` (non liée).

L'initialiseur ci-dessous est fourni. Ton travail est de prédire combien de nombres il crée **sans rien allouer** ; c'est la vérification que tu ferais sur un vrai checkpoint avant de le charger.
# %% code
def init_params(cfg, seed=0, std=0.05):
    rng = np.random.default_rng(seed)
    d, ff, V, L = cfg["d_model"], cfg["d_ff"], cfg["vocab"], cfg["n_layers"]
    w = lambda *s: rng.normal(0.0, std, s)
    layers = []
    for _ in range(L):
        layers.append(dict(attn_norm=np.ones(d), wq=w(d, d), wk=w(d, d), wv=w(d, d), wo=w(d, d),
                           mlp_norm=np.ones(d), w_gate=w(d, ff), w_up=w(d, ff), w_down=w(ff, d)))
    return dict(embed=w(V, d), layers=layers, final_norm=np.ones(d), lm_head=w(d, V))

def tree_size(p):
    if isinstance(p, dict):
        return sum(tree_size(v) for v in p.values())
    if isinstance(p, list):
        return sum(tree_size(v) for v in p)
    return p.size

PARAMS = init_params(CFG, seed=0)
print("allocated parameters:", tree_size(PARAMS))
# %% task Nombre analytique de paramètres
# count_params(cfg) -> int, using only the numbers in cfg (vocab, d_model, n_layers, d_ff)
def count_params(cfg):
    raise NotImplementedError   # TODO
# %% check
BIG = dict(vocab=32000, d_model=4096, n_heads=32, n_layers=32, d_ff=11008, max_len=4096, eps=1e-5, rope_base=10000.0)
check("matches the allocated arrays exactly", lambda: count_params(CFG) == tree_size(PARAMS), "recount: embed, per-layer norms + attention + MLP, final norm, head")
check("hand-computed value for the tiny config is 37216", lambda: count_params(CFG) == 37216, "8288 + 2*10304 + 32 + 8288")
check("matches a different random config", lambda: count_params(dict(CFG, d_model=16, n_heads=2, n_layers=3, d_ff=24, vocab=300)) == tree_size(init_params(dict(CFG, d_model=16, n_heads=2, n_layers=3, d_ff=24, vocab=300))), "formula must use cfg values, not constants")
check("a 7B-class config lands near 6.7 billion", lambda: abs(count_params(BIG) / 6.74e9 - 1) < 0.01, "32 layers, d=4096, ff=11008, V=32000")
check("zero layers leaves embedding + final norm + head", lambda: count_params(dict(CFG, n_layers=0)) == 259 * 32 + 32 + 32 * 259, "the layer term is multiplied by n_layers")
# %% hint
Par couche : `2*d + 4*d*d + 3*d*ff`. Total : `V*d + L*(per layer) + d + d*V`.
# %% solution
def count_params(cfg):
    d, ff, V, L = cfg["d_model"], cfg["d_ff"], cfg["vocab"], cfg["n_layers"]
    per_layer = 2 * d + 4 * d * d + 3 * d * ff
    return V * d + L * per_layer + d + d * V
# %% md Récapitulatif de l'étape 1
Tu as un tokenizer sans perte (n'importe quelle chaîne, n'importe quelle langue, n'importe quel emoji) et une comptabilité exacte de chaque poids. La plupart des 37 216 paramètres se trouvent dans l'embedding et la tête parce que le vocabulaire (259) est grand par rapport à la largeur (32) ; dans un modèle 7B, ce sont au contraire les couches qui dominent (environ 6,5 G sur 6,7 G). Retiens ce rapport : il décide de ce que tu peux quantifier ou répartir (sharding).
# %% report
Avec tes propres mots : pourquoi un tokenizer au niveau de l'octet n'est-il jamais « hors vocabulaire », et que coûte-t-il par rapport à BPE (pense à la longueur de séquence et au travail que les couches d'attention doivent fournir) ?
# %% md Étape 2 : briques de base
## Normalisation, MLP et positions rotatives

**RMSNorm** remet à l'échelle chaque vecteur de token à une moyenne quadratique (root-mean-square) unitaire et multiplie par un gain appris :
$$\text{rmsnorm}(x)_i = g_i \cdot \frac{x_i}{\sqrt{\tfrac1d\sum_j x_j^2 + \varepsilon}}.$$
Elle n'a ni soustraction de la moyenne ni biais, donc elle est moins coûteuse que LayerNorm.

**SwiGLU** remplace le MLP classique à deux couches par une version à porte : $\text{swiglu}(x) = \big(\text{silu}(xW_g)\odot xW_u\big)W_d$ avec $\text{silu}(z)=z\,\sigma(z)$. Formes : `x (.., d)`, `W_g, W_u (d, ff)`, `W_d (ff, d)`.

Les deux fonctions doivent accepter un nombre quelconque de dimensions de batch en tête.
# %% code
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
# %% task RMSNorm, SiLU et SwiGLU
# silu(x)=x*sigmoid(x); rmsnorm(x, g, eps) over the LAST axis; swiglu(x, w_gate, w_up, w_down)
def silu(x):
    raise NotImplementedError   # TODO

def rmsnorm(x, g, eps=1e-6):
    raise NotImplementedError   # TODO

def swiglu(x, w_gate, w_up, w_down):
    raise NotImplementedError   # TODO
# %% check
_r = np.random.default_rng(1)
_x = _r.normal(size=(2, 5, 32))
check("rmsnorm output has unit RMS when the gain is 1", lambda: np.allclose(np.sqrt((rmsnorm(_x, np.ones(32)) ** 2).mean(-1)), 1.0, atol=1e-4), "divide by sqrt(mean(x^2)+eps) on the last axis")
check("rmsnorm is scale invariant and the gain scales the output", lambda: np.allclose(rmsnorm(3 * _x, np.ones(32), 0.0), rmsnorm(_x, np.ones(32), 0.0)) and np.allclose(rmsnorm(_x, 2 * np.ones(32)), 2 * rmsnorm(_x, np.ones(32))), "g multiplies after normalising")
check("silu(0)=0, silu(large)~x, silu(-large)~0", lambda: abs(silu(np.array(0.0))) < 1e-12 and abs(silu(np.array(20.0)) - 20) < 1e-6 and abs(silu(np.array(-20.0))) < 1e-6, "x / (1 + exp(-x))")
check("swiglu equals the hand formula on a 2-d example", lambda: np.allclose(swiglu(np.array([1.0, 2.0]), np.eye(2), 2 * np.eye(2), np.eye(2)), silu(np.array([1.0, 2.0])) * np.array([2.0, 4.0])), "(silu(x Wg) * (x Wu)) Wd")
check("swiglu keeps leading dims and maps d -> d", lambda: swiglu(_x, _r.normal(size=(32, 64)), _r.normal(size=(32, 64)), _r.normal(size=(64, 32))).shape == (2, 5, 32), "ff is internal only")
# %% hint
`np.mean(x*x, axis=-1, keepdims=True)` conserve l'axe pour que le broadcasting fonctionne.
# %% solution
def silu(x):
    return x / (1.0 + np.exp(-x))

def rmsnorm(x, g, eps=1e-6):
    return g * x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

def swiglu(x, w_gate, w_up, w_down):
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down
# %% md Embeddings de position rotatifs (RoPE)
Au lieu d'ajouter un vecteur de position à l'entrée, RoPE **fait tourner** les vecteurs requête et clé d'un angle proportionnel à la position du token. Découpe chaque vecteur de tête de largeur `dh` en deux moitiés `(x1, x2)`, chacune de largeur `dh/2`. La paire `i` a la fréquence $\theta_i = \text{base}^{-2i/d_h}$ et, à la position `p`, elle est tournée de l'angle $p\,\theta_i$ :
$$x_1' = x_1\cos(p\theta) - x_2\sin(p\theta),\qquad x_2' = x_1\sin(p\theta) + x_2\cos(p\theta).$$
Les rotations conservent la longueur, et le produit scalaire d'une requête tournée à la position `m` avec une clé tournée à la position `n` ne dépend que de `m - n`. C'est tout l'intérêt : les scores d'attention deviennent sensibles à la position relative sans aucun paramètre supplémentaire.

`positions` peut être un vecteur `(T,)` partagé par le batch, ou un tableau d'entiers `(B, T)` (nécessaire plus tard pour les batchs avec padding). `x` a la forme `(B, H, T, dh)`.
# %% task Tables RoPE et application
# rope_tables(max_len, head_dim, base) -> (cos, sin), each (max_len, head_dim//2)
# apply_rope(x, cos, sin, positions): x (B,H,T,dh); positions (T,) or (B,T); half-split convention above
def rope_tables(max_len, head_dim, base=10000.0):
    raise NotImplementedError   # TODO

def apply_rope(x, cos, sin, positions):
    raise NotImplementedError   # TODO
# %% check
_dh = 8
_cos, _sin = rope_tables(64, _dh)
_q = np.random.default_rng(2).normal(size=(1, 1, 1, _dh))
_k = np.random.default_rng(3).normal(size=(1, 1, 1, _dh))
_score = lambda m, n: float((apply_rope(_q, _cos, _sin, np.array([m])) * apply_rope(_k, _cos, _sin, np.array([n]))).sum())
check("tables have shape (max_len, dh/2) and position 0 is the identity rotation", lambda: _cos.shape == (64, 4) and _sin.shape == (64, 4) and np.allclose(_cos[0], 1) and np.allclose(_sin[0], 0), "angle = position * base**(-2i/dh)")
check("rotation preserves vector norm and position 0 leaves x unchanged", lambda: np.isclose(np.linalg.norm(apply_rope(_q, _cos, _sin, np.array([37]))), np.linalg.norm(_q)) and np.allclose(apply_rope(_q, _cos, _sin, np.array([0])), _q), "x1' and x2' mix as a 2-d rotation")
check("score depends only on the relative offset", lambda: np.isclose(_score(5, 2), _score(13, 10)) and np.isclose(_score(20, 20), _score(0, 0)) and not np.isclose(_score(5, 2), _score(5, 3)), "q at m, k at n: only m-n matters")
check("per-sequence positions (B,T) are supported", lambda: np.allclose(apply_rope(np.tile(_q, (2, 2, 3, 1)), _cos, _sin, np.array([[0, 1, 2], [4, 5, 6]]))[1, 0, 0], apply_rope(_q, _cos, _sin, np.array([4]))[0, 0, 0]) and apply_rope(np.tile(_q, (2, 2, 3, 1)), _cos, _sin, np.array([[0, 1, 2], [4, 5, 6]])).shape == (2, 2, 3, 8), "index the tables with positions, then add a head axis")
# %% hint
`inv = base ** (-np.arange(0, dh, 2) / dh)` ; angles = `np.arange(max_len)[:, None] * inv[None, :]`.
# %% hint
`cos[positions]` a la forme (T, dh/2) ou (B, T, dh/2). Dans le second cas, insère l'axe des têtes avec `[:, None]` ; dans le premier, utilise `[None, None]`.
# %% solution
def rope_tables(max_len, head_dim, base=10000.0):
    inv = base ** (-np.arange(0, head_dim, 2) / head_dim)
    ang = np.arange(max_len)[:, None] * inv[None, :]
    return np.cos(ang), np.sin(ang)

def apply_rope(x, cos, sin, positions):
    positions = np.asarray(positions)
    c, s = cos[positions], sin[positions]
    if c.ndim == 2:
        c, s = c[None, None], s[None, None]
    else:
        c, s = c[:, None], s[:, None]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
# %% md Attention multi-têtes causale
Formes, pour une entrée `x (B, T, d)` :

1. projeter : `x @ Wq, Wk, Wv`, chacune `(B, T, d)` ;
2. séparer les têtes : remodeler en `(B, T, H, dh)` puis transposer en `(B, H, T, dh)` ;
3. faire tourner `q` et `k` avec RoPE (jamais `v`) ;
4. les scores `q @ k^T / sqrt(dh)` ont la forme `(B, H, T, T)` ;
5. masque : la position `i` peut regarder `j <= i` **et** les clés qui sont de vrais tokens (`pad_mask (B, T)` vaut True pour les vrais tokens, False pour le padding). Les scores masqués deviennent `-1e9` (et non `-inf`, pour qu'une ligne de padding entièrement masquée reste finie) ;
6. softmax sur les clés, multiplication par `v` ce qui donne `(B, H, T, dh)`, fusion des têtes en `(B, T, d)`, projection avec `Wo`.
# %% task Attention multi-têtes causale avec masque de padding
# mha(x, layer, cfg, cos, sin, positions, pad_mask=None) -> (B, T, d)
# layer holds "wq","wk","wv","wo". positions: (T,) or (B,T).
def mha(x, layer, cfg, cos, sin, positions, pad_mask=None):
    raise NotImplementedError   # TODO
# %% check
_cfg = CFG; _H = 4; _dh = 8
_lay = PARAMS["layers"][0]
_cos, _sin = rope_tables(64, _dh)
_x = np.random.default_rng(4).normal(size=(2, 6, 32))
_pos = np.arange(6)
def _naive(x, lay, H=4, dh=8):
    # explicit loops over batch, head and query position (single-vector RoPE through apply_rope)
    B, T, d = x.shape; out = np.zeros((B, T, d))
    for b in range(B):
        heads = []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            Q = (x[b] @ lay["wq"])[:, sl]; K = (x[b] @ lay["wk"])[:, sl]; Vv = (x[b] @ lay["wv"])[:, sl]
            Qr = apply_rope(Q[None, None], _cos, _sin, np.arange(T))[0, 0]
            Kr = apply_rope(K[None, None], _cos, _sin, np.arange(T))[0, 0]
            o = np.zeros((T, dh))
            for i in range(T):
                s = np.array([Qr[i] @ Kr[j] / np.sqrt(dh) for j in range(i + 1)])
                w = np.exp(s - s.max()); w /= w.sum()
                o[i] = sum(w[j] * Vv[j] for j in range(i + 1))
            heads.append(o)
        out[b] = np.concatenate(heads, axis=1) @ lay["wo"]
    return out
_y = mha(_x, _lay, _cfg, _cos, _sin, _pos)
check("output shape is (B, T, d)", lambda: _y.shape == (2, 6, 32), "merge heads back to d")
check("matches a naive triple-loop implementation", lambda: np.allclose(_y, _naive(_x, _lay), atol=1e-10), "check the 1/sqrt(dh) scale, RoPE on q and k only, head order")
def _causal():
    x2 = _x.copy(); x2[:, 4:] = np.random.default_rng(5).normal(size=(2, 2, 32))
    return np.allclose(mha(x2, _lay, _cfg, _cos, _sin, _pos)[:, :4], _y[:, :4])
check("causal: changing future tokens never changes earlier outputs", _causal, "mask strictly-upper-triangular scores before the softmax")
def _padmask():
    m = np.array([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], bool)
    p = np.maximum(np.cumsum(m, 1) - 1, 0)
    a = mha(_x, _lay, _cfg, _cos, _sin, p, m)
    x2 = _x.copy(); x2[0, :2] = 99.0
    b = mha(x2, _lay, _cfg, _cos, _sin, p, m)
    return np.allclose(a[0, 2:], b[0, 2:]) and np.all(np.isfinite(a))
check("padded keys are ignored and stay finite", _padmask, "combine causal mask with pad_mask[:, None, None, :]; use -1e9")
# %% hint
`q = (x @ wq).reshape(B, T, H, dh).transpose(0, 2, 1, 3)`.
# %% hint
`allow = np.tril(np.ones((T, T), bool))[None, None]` ; avec padding : `allow = allow & pad_mask[:, None, None, :]` ; `scores = np.where(allow, scores, -1e9)`.
# %% solution
def mha(x, layer, cfg, cos, sin, positions, pad_mask=None):
    B, T, d = x.shape
    H = cfg["n_heads"]; dh = d // H
    split = lambda z: z.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    q, k, v = split(x @ layer["wq"]), split(x @ layer["wk"]), split(x @ layer["wv"])
    q = apply_rope(q, cos, sin, positions)
    k = apply_rope(k, cos, sin, positions)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    allow = np.tril(np.ones((T, T), bool))[None, None]
    if pad_mask is not None:
        allow = allow & pad_mask[:, None, None, :]
    scores = np.where(allow, scores, -1e9)
    o = softmax(scores, -1) @ v
    return o.transpose(0, 2, 1, 3).reshape(B, T, d) @ layer["wo"]
# %% md Récapitulatif de l'étape 2
Quatre opérateurs, chacun verrouillé par une propriété qui ne demande pas de modèle entraîné : RMS unitaire, la formule à la main de SwiGLU, l'invariance par rotation des scores RoPE, et une attention qui concorde avec une triple boucle tout en ignorant le futur et le padding. Garde l'idée de la boucle naïve : quand un noyau optimisé remplacera ton code, la version lente devient l'oracle.
# %% report
Pourquoi est-il sans danger que le masque de padding utilise -1e9 plutôt que -inf, et que se passerait-il de mal avec -inf sur une ligne de requête entièrement en padding ?
# %% md Étape 3 : la passe avant complète
## Assembler un décodeur

Un bloc, avec connexions résiduelles et pré-normalisation :
$$h = x + \text{MHA}(\text{rmsnorm}(x)),\qquad y = h + \text{SwiGLU}(\text{rmsnorm}(h)).$$
Le modèle fait l'embedding des identifiants de tokens, applique `n_layers` blocs, applique un RMSNorm final et multiplie par la tête de sortie pour obtenir `logits (B, T, V)`. La ligne `t` des logits est la prédiction du modèle pour le token `t+1`.

Avec des poids d'écart type 0,05, les logits sont proches de zéro : un modèle non entraîné est presque uniforme et son entropie croisée doit être proche de $\ln V = \ln 259 \approx 5.557$. C'est une vérification de cohérence célèbre pour toute implémentation de modèle de langage : une fonction de perte initiale très différente signale un bug.
# %% task La passe avant complète
# forward(params, cfg, ids, pad_mask=None, positions=None) -> logits (B, T, V)
# ids: int array (B, T). positions default to arange(T) for every row.
def forward(params, cfg, ids, pad_mask=None, positions=None):
    raise NotImplementedError   # TODO
# %% check
def next_token_loss(logits, ids):
    lp = np.log(softmax(logits[:, :-1], -1))
    tgt = ids[:, 1:]
    return float(-np.mean(np.take_along_axis(lp, tgt[..., None], -1)))

_ids = np.random.default_rng(6).integers(0, 256, size=(4, 16))
_lg = forward(PARAMS, CFG, _ids)
check("logits have shape (B, T, V) and are finite", lambda: _lg.shape == (4, 16, 259) and np.all(np.isfinite(_lg)), "embed -> blocks -> final norm -> lm_head")
check("initial loss is close to ln(V)", lambda: abs(next_token_loss(_lg, _ids) - np.log(259)) < 0.3, "an untrained model should be near uniform; check residuals and norms")
def _prefix():
    return np.allclose(forward(PARAMS, CFG, _ids[:, :9]), _lg[:, :9], atol=1e-10)
check("prefix property: logits of a prefix equal the first rows of the full run", _prefix, "causality at the model level")
def _future():
    ids2 = _ids.copy(); ids2[:, -1] = (ids2[:, -1] + 7) % 256
    l2 = forward(PARAMS, CFG, ids2)
    return np.allclose(l2[:, :-1], _lg[:, :-1]) and not np.allclose(l2[:, -1], _lg[:, -1])
check("changing the last token only changes the last logits", _future, "every layer must be causal")
check("different weights give different logits (params are really used)", lambda: not np.allclose(forward(init_params(CFG, seed=1), CFG, _ids), _lg), "do not hard-code anything")
# %% hint
Résiduel : `x = x + mha(rmsnorm(x, attn_norm), ...)`, puis `x = x + swiglu(rmsnorm(x, mlp_norm), ...)`.
# %% solution
def forward(params, cfg, ids, pad_mask=None, positions=None):
    ids = np.asarray(ids)
    B, T = ids.shape
    dh = cfg["d_model"] // cfg["n_heads"]
    if positions is None:
        positions = np.broadcast_to(np.arange(T), (B, T))
    cos, sin = rope_tables(cfg["max_len"], dh, cfg["rope_base"])
    x = params["embed"][ids]
    for ly in params["layers"]:
        x = x + mha(rmsnorm(x, ly["attn_norm"], cfg["eps"]), ly, cfg, cos, sin, positions, pad_mask)
        x = x + swiglu(rmsnorm(x, ly["mlp_norm"], cfg["eps"]), ly["w_gate"], ly["w_up"], ly["w_down"])
    return rmsnorm(x, params["final_norm"], cfg["eps"]) @ params["lm_head"]
# %% md Récapitulatif de l'étape 3
Un transformer complet en une vingtaine de lignes, plus deux propriétés au niveau du modèle (préfixe et dernier token) qui te permettront plus tard de prouver que le KV cache est exact. Remarque le coût : la passe avant sans cache recalcule tout pour chaque préfixe.
# %% report
Prédis, avant de mesurer, comment la fonction de perte initiale changerait si tu oubliais l'échelle 1/sqrt(dh) ou si tu doublais `std`. Puis essaie et écris ce que tu as observé.
# %% md Étape 4 : échantillonnage et génération
## Des logits à un token

La dernière ligne des logits est un vecteur de `V` scores. Les stratégies de décodage ne diffèrent que par la façon de le transformer en token :

- **greedy** (glouton) : prendre l'argmax (température 0) ;
- **température** $\tau$ : diviser les logits par $\tau$ avant le softmax ; $\tau<1$ accentue, $\tau>1$ aplatit ;
- **top-k** : ne garder que les `k` logits les plus élevés et renormaliser ;
- **top-p** (nucleus) : trier les probabilités par ordre décroissant et garder le plus petit préfixe dont la masse cumulée atteint `p`, puis renormaliser.

Ordre utilisé ici : température, puis top-k, puis top-p. Ta fonction renvoie le **vecteur de probabilités final** pour que tu puisses le tester exactement.
# %% task Distribution d'échantillonnage : température, top-k, top-p
# next_token_probs(logits, temperature=1.0, top_k=0, top_p=1.0) -> probability vector (V,)
#   temperature <= 0 -> one-hot argmax. top_k=0 disables top-k. top_p=1.0 disables top-p.
#   For top-p keep the smallest set whose cumulative probability >= top_p (tolerance 1e-12).
def next_token_probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    raise NotImplementedError   # TODO
# %% check
_p0 = np.array([0.5, 0.3, 0.15, 0.05]); _lg0 = np.log(_p0)
_rl = np.random.default_rng(7).normal(size=50)
def _entropy(p):
    q = p[p > 0]; return float(-(q * np.log(q)).sum())
check("temperature 0 is greedy (one-hot on the argmax)", lambda: np.argmax(next_token_probs(_rl, 0.0)) == np.argmax(_rl) and np.isclose(next_token_probs(_rl, 0.0).sum(), 1) and (next_token_probs(_rl, 0.0) > 0).sum() == 1, "handle temperature <= 0 before dividing")
check("temperature 1 is the plain softmax; higher temperature raises entropy", lambda: np.allclose(next_token_probs(_rl), softmax(_rl)) and _entropy(next_token_probs(_rl, 2.0)) > _entropy(next_token_probs(_rl, 1.0)) > _entropy(next_token_probs(_rl, 0.5)), "logits / temperature")
check("top-k keeps exactly k tokens and renormalises", lambda: (next_token_probs(_rl, 1.0, top_k=3) > 0).sum() == 3 and np.isclose(next_token_probs(_rl, 1.0, top_k=3).sum(), 1) and set(np.flatnonzero(next_token_probs(_rl, 1.0, top_k=3))) == set(np.argsort(-_rl)[:3]), "mask all but the k largest logits with -inf")
check("top-p on [.5,.3,.15,.05]: p=0.7 keeps two tokens, p=0.5 keeps one", lambda: np.allclose(next_token_probs(_lg0, 1.0, top_p=0.7), [0.625, 0.375, 0, 0]) and np.allclose(next_token_probs(_lg0, 1.0, top_p=0.5), [1, 0, 0, 0]), "find the first index where the cumulative sum reaches p; keep that many")
def _freq():
    rng = np.random.default_rng(0); p = next_token_probs(_rl, 1.0, top_k=2)
    draws = rng.choice(50, size=4000, p=p)
    top2 = set(np.argsort(-_rl)[:2])
    return set(draws.tolist()) <= top2 and abs((draws == np.argmax(_rl)).mean() - p[np.argmax(_rl)]) < 0.03
check("empirical frequencies follow the returned distribution", _freq, "sample only from the surviving tokens")
# %% hint
Top-k : `kth = np.sort(z)[-k]`, puis `z = np.where(z >= kth, z, -np.inf)` avant le softmax.
# %% hint
Top-p : `order = np.argsort(-p)`, `cs = np.cumsum(p[order])`, `n = np.searchsorted(cs, top_p - 1e-12) + 1`.
# %% solution
def next_token_probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = np.asarray(logits, float)
    if temperature <= 0:
        p = np.zeros_like(logits); p[np.argmax(logits)] = 1.0
        return p
    z = logits / temperature
    if top_k and top_k < len(z):
        kth = np.sort(z)[-top_k]
        z = np.where(z >= kth, z, -np.inf)
    p = softmax(z)
    if top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        cs = np.cumsum(p[order])
        n = min(int(np.searchsorted(cs, top_p - 1e-12)) + 1, len(p))
        q = np.zeros_like(p); q[order[:n]] = p[order[:n]]
        p = q / q.sum()
    return p
# %% md La boucle de génération sans cache
Cette boucle est fournie. C'est la **référence lente et manifestement correcte** : à chaque étape, elle relance tout le modèle sur toute la séquence jusque-là et échantillonne à partir de la dernière ligne. Ta version avec KV cache doit reproduire sa sortie token pour token.
# %% code
def pick(logits, rng, **samp):
    p = next_token_probs(logits, **samp)
    return int(rng.choice(len(p), p=p))

def generate(params, cfg, prompt_ids, n_new, rng, eos_id=None, **samp):
    ids, out = list(prompt_ids), []
    for _ in range(n_new):
        logits = forward(params, cfg, np.array([ids]))[0, -1]
        t = pick(logits, rng, **samp)
        out.append(t); ids.append(t)
        if eos_id is not None and t == eos_id:
            break
    return out

_prompt = encode("Hello")
_g = generate(PARAMS, CFG, _prompt, 12, np.random.default_rng(0), temperature=0.0)
print("greedy ids:", _g)
print("decoded   :", repr(decode(_prompt + _g)))
# %% md Récapitulatif de l'étape 4
L'échantillonnage est une fonction pure d'un vecteur de logits vers un vecteur de probabilités, plus un tirage aléatoire ; c'est pourquoi il est facile à tester exactement. Un unique objet `rng` passé dans toute la boucle rend les exécutions échantillonnées reproductibles, et il nous permettra de comparer la génération avec et sans cache même quand l'échantillonnage est aléatoire.
# %% report
Quand choisirais-tu top-p plutôt que top-k ? Donne une distribution concrète du token suivant pour laquelle top-k=40 est clairement faux dans un sens.
# %% md Étape 5 : le KV cache
## Ne recalcule pas ce que tu sais déjà

Dans la boucle sans cache, l'étape `n` recalcule les clés et les valeurs de tous les `n-1` tokens précédents, alors qu'elles ne peuvent pas changer (le masque causal fait que les positions antérieures ne voient jamais les suivantes). Un **KV cache** stocke les **clés post-RoPE** et les valeurs de chaque couche, de forme `(B, H, S, dh)` où `S` est le nombre de tokens vus jusque-là. Une étape ne traite alors que les **nouveaux** tokens : elle calcule leurs `q, k, v`, ajoute `k, v` au cache et attend sur tout le cache.

Deux détails décident de la correction :

- les nouveaux tokens aux positions absolues `start .. start+T-1` ont besoin de RoPE avec **ces** positions ;
- la requête `i` du morceau (position absolue `start+i`) peut voir l'indice de cache `j <= start+i`. Avec `T=1`, ce sont toutes les clés en cache ; avec un morceau de préremplissage plus long, c'est toujours causal.
# %% task Attention avec un KV cache
# mha_cached(x, layer, cfg, cos, sin, cache, start) -> (B, T, d)
#   x holds the NEW tokens only, at absolute positions start..start+T-1.
#   cache is a dict {"k": None or (B,H,S,dh), "v": ...}; update it IN PLACE (append along axis 2).
def mha_cached(x, layer, cfg, cos, sin, cache, start):
    raise NotImplementedError   # TODO
# %% check
_x = np.random.default_rng(8).normal(size=(2, 7, 32)); _lay = PARAMS["layers"][1]
_cos, _sin = rope_tables(64, 8)
_full = mha(_x, _lay, CFG, _cos, _sin, np.arange(7))
def _prefill():
    c = {"k": None, "v": None}
    return np.allclose(mha_cached(_x, _lay, CFG, _cos, _sin, c, 0), _full, atol=1e-10) and c["k"].shape == (2, 4, 7, 8) and c["v"].shape == (2, 4, 7, 8)
check("prefill of the whole sequence equals uncached attention; cache is (B,H,S,dh)", _prefill, "with an empty cache this is plain causal attention")
def _stepwise():
    c = {"k": None, "v": None}; outs = []
    for t in range(7):
        outs.append(mha_cached(_x[:, t:t + 1], _lay, CFG, _cos, _sin, c, t))
    return np.allclose(np.concatenate(outs, 1), _full, atol=1e-10)
check("token-by-token decoding equals uncached attention", _stepwise, "rotate the new q and k with position start, attend to all cached keys")
def _chunks():
    c = {"k": None, "v": None}; outs = []; s = 0
    for n in (3, 1, 2, 1):
        outs.append(mha_cached(_x[:, s:s + n], _lay, CFG, _cos, _sin, c, s)); s += n
    return np.allclose(np.concatenate(outs, 1), _full, atol=1e-10) and c["k"].shape[2] == 7
check("mixed chunk sizes (3,1,2,1) work: the mask is causal inside a chunk", _chunks, "allow[i, j] = j <= start + i")
def _postrope():
    c = {"k": None, "v": None}; mha_cached(_x[:, :3], _lay, CFG, _cos, _sin, c, 0)
    k = (_x[:, :3] @ _lay["wk"]).reshape(2, 3, 4, 8).transpose(0, 2, 1, 3)
    return np.allclose(c["k"], apply_rope(k, _cos, _sin, np.arange(3)))
check("the cache stores keys AFTER RoPE", _postrope, "rotate once, store the rotated keys, never rotate again")
# %% hint
`K = np.concatenate([cache["k"], k], axis=2)` quand le cache n'est pas vide, puis range K dans le dictionnaire.
# %% hint
`allow = np.arange(S)[None, :] <= (start + np.arange(T))[:, None]` a la forme (T, S).
# %% solution
def mha_cached(x, layer, cfg, cos, sin, cache, start):
    B, T, d = x.shape
    H = cfg["n_heads"]; dh = d // H
    split = lambda z: z.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    q, k, v = split(x @ layer["wq"]), split(x @ layer["wk"]), split(x @ layer["wv"])
    pos = np.arange(start, start + T)
    q = apply_rope(q, cos, sin, pos)
    k = apply_rope(k, cos, sin, pos)
    if cache["k"] is not None:
        k = np.concatenate([cache["k"], k], axis=2)
        v = np.concatenate([cache["v"], v], axis=2)
    cache["k"], cache["v"] = k, v
    S = k.shape[2]
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    allow = np.arange(S)[None, :] <= (start + np.arange(T))[:, None]
    scores = np.where(allow[None, None], scores, -1e9)
    o = softmax(scores, -1) @ v
    return o.transpose(0, 2, 1, 3).reshape(B, T, d) @ layer["wo"]
# %% code
def new_cache(cfg):
    return [{"k": None, "v": None} for _ in range(cfg["n_layers"])]

def forward_cached(params, cfg, ids, cache, start):
    """Run NEW tokens `ids (B,T)` that sit at absolute positions start..start+T-1."""
    ids = np.asarray(ids)
    dh = cfg["d_model"] // cfg["n_heads"]
    cos, sin = rope_tables(cfg["max_len"], dh, cfg["rope_base"])
    x = params["embed"][ids]
    for ly, c in zip(params["layers"], cache):
        x = x + mha_cached(rmsnorm(x, ly["attn_norm"], cfg["eps"]), ly, cfg, cos, sin, c, start)
        x = x + swiglu(rmsnorm(x, ly["mlp_norm"], cfg["eps"]), ly["w_gate"], ly["w_up"], ly["w_down"])
    return rmsnorm(x, params["final_norm"], cfg["eps"]) @ params["lm_head"]

def cache_len(cache):
    return 0 if cache[0]["k"] is None else cache[0]["k"].shape[2]
# %% task Génération avec cache (préremplissage + décodage)
# generate_cached(params, cfg, prompt_ids, n_new, rng, eos_id=None, return_cache=False, **samp)
#   1) PREFILL: one forward_cached call on the whole prompt (start=0), sample from the last row.
#   2) DECODE: each further token is fed alone with start = number of tokens already in the cache.
#   Must produce exactly the same tokens as `generate` and process each token only once.
def generate_cached(params, cfg, prompt_ids, n_new, rng, eos_id=None, return_cache=False, **samp):
    raise NotImplementedError   # TODO
# %% check
_pr = encode("The cat")
_ref = generate(PARAMS, CFG, _pr, 20, np.random.default_rng(0), temperature=0.0)
check("greedy output is identical to the uncached loop", lambda: generate_cached(PARAMS, CFG, _pr, 20, np.random.default_rng(0), temperature=0.0) == _ref, "feed one token at a time after the prefill")
check("sampled output (temperature, top-k, top-p) is identical for the same seed", lambda: generate_cached(PARAMS, CFG, _pr, 25, np.random.default_rng(5), temperature=0.9, top_k=40, top_p=0.95) == generate(PARAMS, CFG, _pr, 25, np.random.default_rng(5), temperature=0.9, top_k=40, top_p=0.95), "use the same pick() and consume the rng once per token")
def _lens():
    toks, cache = generate_cached(PARAMS, CFG, _pr, 10, np.random.default_rng(0), temperature=0.0, return_cache=True)
    return len(toks) == 10 and cache_len(cache) == len(_pr) + 10 - 1
check("cache holds prompt + n_new - 1 tokens (the last sample is never fed back)", _lens, "the final sampled token has not been processed yet")
def _work():
    seen = []; orig = globals()["forward_cached"]
    def spy(params, cfg, ids, cache, start):
        seen.append(np.asarray(ids).shape[1]); return orig(params, cfg, ids, cache, start)
    globals()["forward_cached"] = spy
    try:
        generate_cached(PARAMS, CFG, _pr, 9, np.random.default_rng(0), temperature=0.0)
    finally:
        globals()["forward_cached"] = orig
    return seen == [len(_pr)] + [1] * 8
check("work done: one prefill chunk then single-token steps", _work, "total tokens processed = prompt + n_new - 1")
def _eos():
    first = generate(PARAMS, CFG, _pr, 1, np.random.default_rng(0), temperature=0.0)[0]
    return generate_cached(PARAMS, CFG, _pr, 20, np.random.default_rng(0), eos_id=first, temperature=0.0) == [first]
check("generation stops at eos_id", _eos, "break right after appending the eos token")
# %% hint
Garde `out` et un compteur. Après avoir échantillonné le token `t`, si tu as besoin d'un autre token, lance `forward_cached(params, cfg, np.array([[t]]), cache, len(prompt_ids) + len(out) - 1)`.
# %% solution
def generate_cached(params, cfg, prompt_ids, n_new, rng, eos_id=None, return_cache=False, **samp):
    cache = new_cache(cfg)
    logits = forward_cached(params, cfg, np.array([list(prompt_ids)]), cache, 0)[0, -1]
    out = []
    for step in range(n_new):
        t = pick(logits, rng, **samp)
        out.append(t)
        if eos_id is not None and t == eos_id:
            break
        if step < n_new - 1:
            logits = forward_cached(params, cfg, np.array([[t]]), cache, len(prompt_ids) + len(out) - 1)[0, -1]
    return (out, cache) if return_cache else out
# %% md Preuve par l'expérience : plusieurs graines
Des tokens égaux pourraient être de la chance sur une graine. L'affirmation que nous voulons est plus forte : **les logits concordent à l'arrondi flottant près pour n'importe quels poids**. Voici l'expérience sur 8 modèles aléatoires et 3 prompts chacun ; les nombres sont la moyenne ± l'écart type, sur les 24 exécutions, de la plus grande différence absolue de logit entre la passe avant sans cache et la passe avant avec cache pas à pas.
# %% code
def stepwise_cached_logits(params, cfg, ids, n_prefill):
    cache = new_cache(cfg)
    outs = [forward_cached(params, cfg, ids[:, :n_prefill], cache, 0)]
    for t in range(n_prefill, ids.shape[1]):
        outs.append(forward_cached(params, cfg, ids[:, t:t + 1], cache, t))
    return np.concatenate(outs, axis=1)

diffs = []
for seed in range(8):
    prm = init_params(CFG, seed=100 + seed)
    for r in range(3):
        ids = np.random.default_rng(1000 * seed + r).integers(0, 256, size=(1, 24))
        diffs.append(np.abs(stepwise_cached_logits(prm, CFG, ids, 5) - forward(prm, CFG, ids)).max())
diffs = np.array(diffs)
print(f"max |logit diff| over {len(diffs)} runs: mean {diffs.mean():.2e} +- {diffs.std():.2e}, worst {diffs.max():.2e}")
assert diffs.max() < 1e-9
# %% md Analyse de défaillance : où un mauvais cache diverge-t-il en premier ?
Un bug de cache plante rarement. Il rend le modèle *légèrement* différent, et le symptôme apparaît bien plus tard sous forme de texte dégradé. Le réflexe professionnel est de **localiser la première divergence** : comparer les logits incrémentaux à la référence ligne par ligne et signaler la première position en désaccord.
# %% task Première divergence entre deux tenseurs de logits
# first_divergence(ref, test, tol=1e-8) -> int index of the first position (axis 1) where
#   max over batch and vocab of |ref - test| exceeds tol, or None when they agree everywhere.
# stepwise_logits(params, cfg, ids, n_prefill, offset_fn) -> (B, T, V)
#   like stepwise_cached_logits, but the `start` passed for the decode token at index t is offset_fn(t).
def first_divergence(ref, test, tol=1e-8):
    raise NotImplementedError   # TODO

def stepwise_logits(params, cfg, ids, n_prefill, offset_fn):
    raise NotImplementedError   # TODO
# %% check
_ids = np.random.default_rng(9).integers(0, 256, size=(2, 14))
_ref = forward(PARAMS, CFG, _ids)
_a = _ref.copy(); _a[:, 6, 3] += 1e-3
check("first_divergence returns None for identical tensors and the right index otherwise", lambda: first_divergence(_ref, _ref.copy()) is None and first_divergence(_ref, _a) == 6, "scan positions in order; take the max over batch and vocab")
check("a correct offset function reproduces the reference everywhere", lambda: first_divergence(_ref, stepwise_logits(PARAMS, CFG, _ids, 5, lambda t: t)) is None, "start must equal the number of tokens already cached")
check("bug 'off by one position' first diverges at the first decoded token", lambda: first_divergence(_ref, stepwise_logits(PARAMS, CFG, _ids, 5, lambda t: t + 1)) == 5, "the prefill rows are still right")
check("bug 'forgot the offset' (start=0) also diverges at the first decoded token", lambda: first_divergence(_ref, stepwise_logits(PARAMS, CFG, _ids, 5, lambda t: 0)) == 5, "RoPE and the mask both use start")
check("a tiny tolerance matters: a 1e-3 bug is invisible at tol=1e-2", lambda: first_divergence(_ref, _a, tol=1e-2) is None, "compare with tol, not with ==")
# %% hint
`d = np.abs(ref - test).max(axis=(0, 2))` donne un nombre par position ; `np.flatnonzero(d > tol)`.
# %% solution
def first_divergence(ref, test, tol=1e-8):
    d = np.abs(np.asarray(ref) - np.asarray(test)).max(axis=(0, 2))
    bad = np.flatnonzero(d > tol)
    return int(bad[0]) if len(bad) else None

def stepwise_logits(params, cfg, ids, n_prefill, offset_fn):
    cache = new_cache(cfg)
    outs = [forward_cached(params, cfg, ids[:, :n_prefill], cache, 0)]
    for t in range(n_prefill, ids.shape[1]):
        outs.append(forward_cached(params, cfg, ids[:, t:t + 1], cache, offset_fn(t)))
    return np.concatenate(outs, axis=1)
# %% md Récapitulatif de l'étape 5
Le cache est exact : la plus grande différence de logit est du bruit d'arrondi (environ 1e-15), pas « à peu près pareil ». Tu disposes aussi d'un outil de débogage qui pointe la première ligne fausse. Deux règles à retenir : stocker les clés après RoPE, et le décalage de position/masque est la longueur du cache.
# %% report
Un collègue dit « la sortie avec cache diffère de 1e-4 de celle sans cache, c'est juste du bruit flottant ». En t'appuyant sur cette étape, argumente si c'est plausible et ce que tu testerais ensuite.
# %% md Étape 6 : comptabilité des coûts
## FLOP et octets

Nous ne comptons que les multiplications matricielles (normes, RoPE et softmax sont des termes d'ordre inférieur) et nous utilisons la convention **une multiplication-addition = 2 FLOP**.

Pour `t_new` nouveaux tokens quand `t_past` tokens sont déjà en cache :

- couches linéaires par token et par couche : $2(4d^2 + 3d\,ff)$ (les quatre projections d'attention et les trois matrices du MLP) ;
- attention par couche : le token `i` du morceau voit $t_{past}+i+1$ clés, et les scores plus la somme pondérée coûtent chacun $2\,d$ FLOP par clé visible, donc $4d$ par clé visible, avec $\sum_i (t_{past}+i+1) = t_{new}\,t_{past} + t_{new}(t_{new}+1)/2$ paires visibles ;
- tête de sortie : $2\,d\,V$ par token.

Une génération entière coûte alors la somme de ses étapes : la boucle sans cache retraite `L+k` tokens à l'étape `k`, la boucle avec cache traite le prompt une fois puis un seul token ensuite.
# %% task Comptabilité des FLOP
# forward_flops(cfg, t_new, t_past=0) -> int
# generate_flops(cfg, prompt_len, n_new, cached) -> int  (n_new sampled tokens; the last one is not fed back when cached)
def forward_flops(cfg, t_new, t_past=0):
    raise NotImplementedError   # TODO

def generate_flops(cfg, prompt_len, n_new, cached):
    raise NotImplementedError   # TODO
# %% check
check("single-token cost for the tiny model is 57792 FLOPs", lambda: forward_flops(CFG, 1, 0) == 57792, "2 layers * (20480 + 128) + head 16576")
check("chunk consistency: prefill(a+b) = prefill(a) + process b with a cached", lambda: forward_flops(CFG, 10, 0) == forward_flops(CFG, 6, 0) + forward_flops(CFG, 4, 6), "visible-pair count must include t_new*t_past")
check("cached and uncached agree for a single generated token", lambda: generate_flops(CFG, 8, 1, True) == generate_flops(CFG, 8, 1, False) == forward_flops(CFG, 8, 0), "no step after the first")
check("caching saves more and more as generation gets longer", lambda: generate_flops(CFG, 8, 8, False) / generate_flops(CFG, 8, 8, True) < generate_flops(CFG, 8, 64, False) / generate_flops(CFG, 8, 64, True) and generate_flops(CFG, 8, 64, False) / generate_flops(CFG, 8, 64, True) > 10, "uncached is quadratic, cached ~linear")
check("cached cost = prefill + sum of single-token steps", lambda: generate_flops(CFG, 5, 4, True) == forward_flops(CFG, 5, 0) + sum(forward_flops(CFG, 1, 5 + j) for j in range(3)), "step j sees 5+j cached tokens")
# %% hint
FLOP d'attention par couche : `4 * d * vis` avec `vis = t_new*t_past + t_new*(t_new+1)//2`.
# %% solution
def forward_flops(cfg, t_new, t_past=0):
    d, ff, L, V = cfg["d_model"], cfg["d_ff"], cfg["n_layers"], cfg["vocab"]
    lin = 2 * (4 * d * d + 3 * d * ff)
    vis = t_new * t_past + t_new * (t_new + 1) // 2
    return L * (t_new * lin + 4 * d * vis) + t_new * 2 * d * V

def generate_flops(cfg, prompt_len, n_new, cached):
    if cached:
        return forward_flops(cfg, prompt_len, 0) + sum(forward_flops(cfg, 1, prompt_len + j) for j in range(n_new - 1))
    return sum(forward_flops(cfg, prompt_len + k, 0) for k in range(n_new))
# %% md Mémoire
Les poids coûtent `params * bytes_per_param`. Le KV cache coûte, par token et par couche, un vecteur clé et un vecteur valeur pour chaque tête KV : `2 * n_kv_heads * dh * bytes`. Pour l'attention multi-têtes, `n_kv_heads = H` ; l'attention à requêtes groupées (GQA) partage chaque tête clé/valeur entre plusieurs têtes de requête, ce qui réduit le cache sans toucher au côté requête des poids.
# %% task Mémoire du KV cache et capacité de batch
# kv_cache_bytes(cfg, batch, seq_len, bytes_per=2, n_kv_heads=None) -> int
# param_bytes(cfg, bytes_per=2) -> int
# max_batch(cfg, seq_len, mem_bytes, bytes_per=2, n_kv_heads=None) -> int (sequences that fit next to the weights)
def kv_cache_bytes(cfg, batch, seq_len, bytes_per=2, n_kv_heads=None):
    raise NotImplementedError   # TODO

def param_bytes(cfg, bytes_per=2):
    raise NotImplementedError   # TODO

def max_batch(cfg, seq_len, mem_bytes, bytes_per=2, n_kv_heads=None):
    raise NotImplementedError   # TODO
# %% check
check("7B-class model, 4096 tokens, fp16: the KV cache is exactly 2 GiB per sequence", lambda: kv_cache_bytes(BIG, 1, 4096) == 2 ** 31, "2 * layers * seq * heads * dh * 2 bytes")
check("cache grows linearly in batch and sequence length", lambda: kv_cache_bytes(BIG, 8, 1000) == 8 * kv_cache_bytes(BIG, 1, 1000) == 1000 * kv_cache_bytes(BIG, 8, 1), "no quadratic term")
check("GQA with 8 KV heads divides the cache by 4", lambda: kv_cache_bytes(BIG, 1, 4096, n_kv_heads=8) * 4 == kv_cache_bytes(BIG, 1, 4096), "dh stays the same, only the head count changes")
check("weights of the 7B-class model in fp16 are about 13.5 GB", lambda: abs(param_bytes(BIG) / 13.48e9 - 1) < 0.01 and param_bytes(CFG, 4) == 4 * 37216, "bytes = params * bytes_per")
check("an 80 GB card holds 30 full-length sequences next to the weights", lambda: max_batch(BIG, 4096, 80e9) == 30 and max_batch(BIG, 4096, 1e9) == 0, "floor((mem - weights) / cache per sequence), never negative")
# %% hint
`dh = d_model // n_heads` ; le terme du cache est `2 * n_layers * batch * seq_len * n_kv_heads * dh * bytes_per`.
# %% solution
def kv_cache_bytes(cfg, batch, seq_len, bytes_per=2, n_kv_heads=None):
    H = n_kv_heads or cfg["n_heads"]
    dh = cfg["d_model"] // cfg["n_heads"]
    return 2 * cfg["n_layers"] * batch * seq_len * H * dh * bytes_per

def param_bytes(cfg, bytes_per=2):
    return count_params(cfg) * bytes_per

def max_batch(cfg, seq_len, mem_bytes, bytes_per=2, n_kv_heads=None):
    free = mem_bytes - param_bytes(cfg, bytes_per)
    if free <= 0:
        return 0
    return int(free // kv_cache_bytes(cfg, 1, seq_len, bytes_per, n_kv_heads))
# %% md Récapitulatif de l'étape 6
Deux faits en ressortent. (1) Sans cache, le travail croît comme le carré de la longueur de sortie ; avec cache, linéairement. (2) Le cache n'est pas gratuit : à 4096 tokens, il pèse 2 GiB par séquence pour un modèle 7B, ce qui plafonne la taille de batch à environ 30 sur un appareil de 80 GB et explique pourquoi la GQA, les caches quantifiés et l'allocation paginée existent.
# %% report
Calcule le rapport entre les FLOP sans cache et avec cache pour un prompt de 512 tokens et 512 tokens générés sur la configuration de classe 7B (utilise tes fonctions). Le rapport est-il celui que tu attendais ? Quelle hypothèse dans « FLOP » ignore le coût réel du cache ?
# %% md Étape 7 : batching avec padding
## Servir plusieurs prompts à la fois

Les prompts ont des longueurs différentes, mais un batch est un rectangle. Nous faisons un **padding à gauche** avec `PAD` pour que la dernière colonne de chaque ligne soit un vrai token (celui dont on lit les logits du token suivant). Le padding doit être invisible :

- `pad_mask (B, T)` : True là où le token est réel ; les clés de padding sont masquées dans l'attention ;
- `positions (B, T)` : la position de chaque vrai token comptée à partir du premier vrai token (0, 1, 2 ...) et 0 sur le padding, pour que RoPE voie les mêmes positions que dans une exécution sans padding.

Si tu oublies l'un des deux, la réponse pour un prompt dépend des autres prompts de son batch, ce qui est un bug de production bien réel.
# %% task Batchs avec padding à gauche
# make_batch(seqs, pad_id=PAD) -> (ids (B,T) int, pad_mask (B,T) bool, positions (B,T) int)
def make_batch(seqs, pad_id=PAD):
    raise NotImplementedError   # TODO
# %% check
_seqs = [encode("Hi"), encode("A longer prompt"), encode("mid size")]
_ids, _m, _pos = make_batch(_seqs)
check("shape, left alignment and padding value", lambda: _ids.shape == (3, len(_seqs[1])) and _ids[1, 0] == BOS and _ids[0, -1] == _seqs[0][-1] and _ids[0, 0] == PAD and _m.sum(1).tolist() == [len(s) for s in _seqs], "row i has len(seq_i) real tokens at the right end")
check("positions count from the first real token and are 0 on padding", lambda: _pos[0].tolist()[:len(_ids[0]) - 3] == [0] * (len(_ids[0]) - 3) and _pos[0, -3:].tolist() == [0, 1, 2] and _pos[1].tolist() == list(range(len(_seqs[1]))), "cumsum(mask) - 1, clipped at 0")
def _equiv():
    prm = PARAMS
    lb = forward(prm, CFG, _ids, _m, _pos)
    for i, s in enumerate(_seqs):
        one = forward(prm, CFG, np.array([s]))[0]
        if not np.allclose(lb[i, -len(s):], one, atol=1e-9):
            return False
    return True
check("batched logits at real positions equal the one-sequence runs", _equiv, "pad_mask and positions must both be passed to forward")
def _needed():
    naive = forward(PARAMS, CFG, _ids)  # ignoring mask and positions
    one = forward(PARAMS, CFG, np.array([_seqs[0]]))[0]
    return not np.allclose(naive[0, -len(_seqs[0]):], one, atol=1e-6)
check("ignoring the mask and positions changes the answer (so they are needed)", _needed, "padding leaks into attention and shifts RoPE")
def _next():
    lb = forward(PARAMS, CFG, _ids, _m, _pos)[:, -1]
    return [int(np.argmax(r)) for r in lb] == [int(np.argmax(forward(PARAMS, CFG, np.array([s]))[0, -1])) for s in _seqs]
check("greedy next token of each row is independent of its neighbours", _next, "read the last column")
# %% hint
`positions = np.maximum(np.cumsum(mask, axis=1) - 1, 0)`.
# %% solution
def make_batch(seqs, pad_id=PAD):
    B, T = len(seqs), max(len(s) for s in seqs)
    ids = np.full((B, T), pad_id, dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = s
        mask[i, T - len(s):] = True
    positions = np.maximum(np.cumsum(mask, axis=1) - 1, 0)
    return ids, mask, positions
# %% md Récapitulatif de l'étape 7
Le batching change le débit, jamais les réponses. Le test d'équivalence ci-dessus est le modèle de toute optimisation de serving : exécuter le chemin lent à une seule séquence comme oracle et comparer les logits aux positions réelles. Le prix est du travail gaspillé : avec le padding à gauche, les cellules de padding coûtent quand même des FLOP.
# %% report
Les cellules de padding sont calculées mais jetées. Comment le continuous batching ou le regroupement par longueur (length bucketing) réduirait-il ce gaspillage ? Estime le gaspillage pour des longueurs de prompt [3, 50, 51, 52].
# %% md Étape 8 : benchmark de latence simulé
## Un modèle roofline du décodage

Nous ne mesurons pas la machine (c'est un bac à sable Python). Nous **simulons** un appareil avec deux nombres : la puissance de calcul de crête `flops` (FLOP/s) et la bande passante mémoire `bw` (octets/s). Une étape de décodage pour `batch` séquences avec `ctx` tokens visibles nécessite

- un temps de calcul : `batch * forward_flops(cfg, 1, ctx-1) / flops`,
- un temps mémoire : `(param_bytes + kv_cache_bytes(cfg, batch, ctx)) / bw` (chaque poids et chaque clé/valeur en cache est lu une fois par étape),

et dure le `max` des deux. Le plus grand des deux désigne le facteur limitant. L'appareil ci-dessous ressemble à un GPU de centre de données (312 TFLOP/s en fp16, 2 TB/s).
# %% code
HW = dict(flops=312e12, bw=2.0e12)
# %% task Temps roofline d'une étape de décodage
# decode_step_time(cfg, batch, ctx, hw) -> dict(time=seconds, t_compute=..., t_mem=..., bound="compute"|"memory")
# hw = dict(flops=..., bw=...)   fp16 (2 bytes) for weights and cache
def decode_step_time(cfg, batch, ctx, hw):
    raise NotImplementedError   # TODO
# %% check
_s = decode_step_time(BIG, 1, 1024, HW)
check("batch 1 on the 7B-class model is memory-bound at about 7 ms per token", lambda: _s["bound"] == "memory" and 0.0065 < _s["time"] < 0.0075 and _s["time"] == max(_s["t_compute"], _s["t_mem"]), "weights are re-read every step: 13.5 GB / 2 TB/s")
check("a big batch with a short context flips to compute-bound", lambda: decode_step_time(BIG, 512, 16, HW)["bound"] == "compute", "compute grows with batch, weight traffic does not")
check("longer contexts never make a step faster; batching improves tokens/s", lambda: decode_step_time(BIG, 8, 4096, HW)["time"] > decode_step_time(BIG, 8, 128, HW)["time"] and 8 / decode_step_time(BIG, 8, 512, HW)["time"] > 6 / decode_step_time(BIG, 1, 512, HW)["time"], "monotone in ctx; per-token cost amortised over the batch")
check("memory time equals the byte count over bandwidth", lambda: np.isclose(decode_step_time(BIG, 4, 256, HW)["t_mem"], (param_bytes(BIG) + kv_cache_bytes(BIG, 4, 256)) / 2.0e12), "include the KV cache read")
# %% hint
`bound = "compute" if t_compute >= t_mem else "memory"`.
# %% solution
def decode_step_time(cfg, batch, ctx, hw):
    t_c = batch * forward_flops(cfg, 1, ctx - 1) / hw["flops"]
    t_m = (param_bytes(cfg) + kv_cache_bytes(cfg, batch, ctx)) / hw["bw"]
    return dict(time=max(t_c, t_m), t_compute=t_c, t_mem=t_m, bound="compute" if t_c >= t_m else "memory")
# %% md Le tableau de benchmark
Maintenant, nous combinons tout. La boucle sans cache refait tout le préfixe à chaque étape, soit un préremplissage gourmand en calcul à chaque fois ; la boucle avec cache fait un seul préremplissage puis des étapes de décodage limitées par le roofline. Tous les nombres sont simulés pour la configuration de classe 7B sur `HW`.
# %% code
def prefill_time(cfg, batch, n_tok, hw):
    return max(batch * forward_flops(cfg, n_tok, 0) / hw["flops"], param_bytes(cfg) / hw["bw"])

def request_time(cfg, prompt_len, n_new, batch, hw, cached):
    if cached:
        t = prefill_time(cfg, batch, prompt_len, hw)
        for j in range(n_new - 1):
            t += decode_step_time(cfg, batch, prompt_len + j + 1, hw)["time"]
        return t
    return sum(prefill_time(cfg, batch, prompt_len + k, hw) for k in range(n_new))

print(f"{'batch':>5} {'prompt':>6} {'new':>5} | {'uncached ms':>12} {'cached ms':>10} {'speedup':>8} | {'tokens/s (cached)':>18}")
for batch in (1, 8, 32):
    for prompt_len, n_new in ((128, 128), (512, 256)):
        tu = request_time(BIG, prompt_len, n_new, batch, HW, False)
        tc = request_time(BIG, prompt_len, n_new, batch, HW, True)
        print(f"{batch:>5} {prompt_len:>6} {n_new:>5} | {tu*1e3:>12.1f} {tc*1e3:>10.1f} {tu/tc:>7.1f}x | {batch*n_new/tc:>18.0f}")

print("\nFLOPs for the tiny model that we actually run (prompt 8):")
for n_new in (8, 32, 128):
    u, c = generate_flops(CFG, 8, n_new, False), generate_flops(CFG, 8, n_new, True)
    print(f"  new={n_new:>4}: uncached {u:>12,}  cached {c:>12,}  ratio {u/c:>6.1f}")
# %% md Conclusion et ce que dit le tableau
Lecture du tableau : le décodage avec cache est limité par le déplacement d'octets, donc une séquence seule atteint environ 145 tokens/s quelle que soit l'ingéniosité de l'arithmétique, et le débit croît presque linéairement avec la taille du batch jusqu'à ce que les lectures du cache ou le plafond de calcul apparaissent. C'est la vraie motivation du batching, de la quantification (moins d'octets par poids), de la GQA (un cache plus petit) et du décodage spéculatif (plus de tokens par lecture des poids).
# %% report
Rapport final. (1) Liste les trois tests qui t'ont donné le plus de confiance dans ton moteur et le bug que chacun détecterait. (2) Quelle partie du modèle roofline est la moins réaliste (surcoût de lancement des noyaux, interconnexion, attention comme noyau séparé limité par la mémoire) ? (3) Si tu devais ajouter la GQA, quelles fonctions changeraient et quels tests mettrais-tu à jour en premier ?
# %% md Sources
Uniquement du matériel réellement ouvert et lu pendant la rédaction de ce projet :

- `docs/PROJECTS.md` et `docs/AGENT_BRIEF.md` de ce dépôt (format et règles).
- Front matter (id, title, requires) des leçons `trf.01`-`trf.04`, `llm.01`, `llm.02`, `sys.01` et `sys.05` dans `content/authored/` pour choisir les prérequis. Les corps des leçons n'ont pas été étudiés ligne par ligne : les équations de ce projet sont standard (rotation RoPE par moitiés, RMSNorm, SwiGLU, les formules de FLOP et d'octets ci-dessus) et chaque affirmation numérique est vérifiée par les cellules d'ici.
# %% md Limites
Poids aléatoires : rien ici ne dit quoi que ce soit sur la qualité du texte. Les décomptes de FLOP ignorent les normes, le softmax, RoPE et la lecture de l'embedding ; le roofline traite l'appareil comme deux nombres et ignore les surcoûts des noyaux, la communication et la pagination du cache.