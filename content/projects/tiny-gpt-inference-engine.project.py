# %% meta
{"id": "proj.tinygpt", "title": "Build a tiny GPT inference engine", "summary": "Implement the whole inference path of a decoder-only transformer in NumPy: byte tokenizer, RoPE, causal multi-head attention, RMSNorm, SwiGLU, sampling, a KV cache proven identical to the uncached path, FLOP and memory accounting, padded batching and a simulated latency benchmark.", "level": "expert", "minutes": 180,
 "requires": ["trf.01", "trf.02", "trf.03", "trf.04", "llm.01", "llm.02", "sys.01"],
 "outcomes": ["Write every operator of a modern decoder-only transformer with exact tensor shapes and test each one against an independent property", "Count parameters analytically and match them to the arrays you allocated", "Implement greedy, temperature, top-k and top-p sampling and verify their probability mass", "Build a KV cache and prove that it changes speed but not a single output", "Account for FLOPs and bytes and explain why decoding is memory-bound", "Batch variable-length prompts with padding masks and position ids without changing any answer"],
 "stages": ["Tokenizer and config", "Building blocks", "The full forward pass", "Sampling and generation", "KV cache", "Cost accounting", "Batching", "Benchmark and report"]}
# %% md Stage 1: tokenizer and configuration
## What you are building

An inference engine takes a sequence of integers, runs them through a fixed set of weights and returns a probability distribution over the next integer. Training is somebody else's job: here the weights are **random but seeded**, so the model talks nonsense. That is deliberate. The goal is **correctness and engineering**: every function you write has a property that can be tested without knowing whether the model is any good (causality, equivalence between two code paths, an exact parameter count).

The architecture is the modern "Llama-style" decoder: pre-normalisation with RMSNorm, rotary position embeddings (RoPE), multi-head causal attention, a SwiGLU feed-forward block and an untied output head.

Everything lives in one shared namespace, like a notebook. Tasks you finish stay defined for later cells. Tensor shape names used throughout: `B` batch, `T` sequence length, `d` model width, `H` heads, `dh = d/H` head width, `V` vocabulary size, `ff` hidden width of the MLP.

Your first step is the simplest tokenizer that can represent any text: raw UTF-8 bytes. 256 byte values plus three special ids.
# %% code
import numpy as np

CFG = dict(vocab=259, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_len=64, eps=1e-6, rope_base=10000.0)
PAD, BOS, EOS = 256, 257, 258          # ids 0..255 are raw bytes
assert CFG["d_model"] % CFG["n_heads"] == 0
print("head width dh =", CFG["d_model"] // CFG["n_heads"])
# %% task Byte-level tokenizer: encode and decode
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
A string is a sequence of characters, but the model sees bytes. `list(text.encode("utf-8"))` gives the byte values.
# %% hint
decode: `bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")`.
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
# %% md Parameters of a decoder
Each layer holds: an attention RMSNorm gain (`d`), four attention matrices `Wq, Wk, Wv, Wo` (each `d x d`), an MLP RMSNorm gain (`d`), and three SwiGLU matrices `W_gate, W_up` (`d x ff`) and `W_down` (`ff x d`). Outside the layers: the embedding table `V x d`, a final RMSNorm gain `d`, and the output head `d x V` (untied).

The initialiser below is given. Your job is to predict how many numbers it creates **without allocating anything**; this is the check you would run on a real checkpoint before loading it.
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
# %% task Analytic parameter count
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
Per layer: `2*d + 4*d*d + 3*d*ff`. Total: `V*d + L*(per layer) + d + d*V`.
# %% solution
def count_params(cfg):
    d, ff, V, L = cfg["d_model"], cfg["d_ff"], cfg["vocab"], cfg["n_layers"]
    per_layer = 2 * d + 4 * d * d + 3 * d * ff
    return V * d + L * per_layer + d + d * V
# %% md Stage 1 summary
You have a lossless tokenizer (any string, any language, any emoji) and an exact accounting of every weight. Most of the 37 216 parameters live in the embedding and the head because the vocabulary (259) is large relative to the width (32); in a 7B model the layers dominate instead (about 6.5 B of 6.7 B). Remember this ratio: it decides what you can quantise or shard.
# %% report
In your own words: why is a byte-level tokenizer never "out of vocabulary", and what does it cost compared with BPE (think of sequence length and of the work the attention layers must do)?
# %% md Stage 2: building blocks
## Normalisation, the MLP and rotary positions

**RMSNorm** rescales each token vector to unit root-mean-square and multiplies by a learned gain:
$$\text{rmsnorm}(x)_i = g_i \cdot \frac{x_i}{\sqrt{\tfrac1d\sum_j x_j^2 + \varepsilon}}.$$
It has no mean subtraction and no bias, so it is cheaper than LayerNorm.

**SwiGLU** replaces the plain two-layer MLP by a gated one: $\text{swiglu}(x) = \big(\text{silu}(xW_g)\odot xW_u\big)W_d$ with $\text{silu}(z)=z\,\sigma(z)$. Shapes: `x (.., d)`, `W_g, W_u (d, ff)`, `W_d (ff, d)`.

Both functions must accept any number of leading batch dimensions.
# %% code
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
# %% task RMSNorm, SiLU and SwiGLU
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
`np.mean(x*x, axis=-1, keepdims=True)` keeps the axis so broadcasting works.
# %% solution
def silu(x):
    return x / (1.0 + np.exp(-x))

def rmsnorm(x, g, eps=1e-6):
    return g * x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

def swiglu(x, w_gate, w_up, w_down):
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down
# %% md Rotary position embeddings (RoPE)
Instead of adding a position vector to the input, RoPE **rotates** the query and key vectors by an angle proportional to the token position. Split each head vector of width `dh` into two halves `(x1, x2)`, each of width `dh/2`. Pair `i` gets frequency $\theta_i = \text{base}^{-2i/d_h}$ and at position `p` it is rotated by angle $p\,\theta_i$:
$$x_1' = x_1\cos(p\theta) - x_2\sin(p\theta),\qquad x_2' = x_1\sin(p\theta) + x_2\cos(p\theta).$$
Rotations preserve length, and the dot product of a rotated query at position `m` with a rotated key at position `n` depends only on `m - n`. That is the whole point: attention scores become relative-position aware with no extra parameters.

`positions` can be a vector `(T,)` shared by the batch, or an integer array `(B, T)` (needed for padded batches later). `x` has shape `(B, H, T, dh)`.
# %% task RoPE tables and application
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
`inv = base ** (-np.arange(0, dh, 2) / dh)`; angles = `np.arange(max_len)[:, None] * inv[None, :]`.
# %% hint
`cos[positions]` has shape (T, dh/2) or (B, T, dh/2). In the second case insert the head axis with `[:, None]`; in the first use `[None, None]`.
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
# %% md Causal multi-head attention
Shapes, for input `x (B, T, d)`:

1. project: `x @ Wq, Wk, Wv` each `(B, T, d)`;
2. split heads: reshape to `(B, T, H, dh)` and transpose to `(B, H, T, dh)`;
3. rotate `q` and `k` with RoPE (never `v`);
4. scores `q @ k^T / sqrt(dh)` has shape `(B, H, T, T)`;
5. mask: position `i` may look at `j <= i` **and** at keys that are real tokens (`pad_mask (B, T)` is True for real tokens, False for padding). Masked scores become `-1e9` (not `-inf`, so a fully masked padding row stays finite);
6. softmax over keys, multiply by `v` giving `(B, H, T, dh)`, merge heads to `(B, T, d)`, project with `Wo`.
# %% task Causal multi-head attention with padding mask
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
`allow = np.tril(np.ones((T, T), bool))[None, None]`; with padding: `allow = allow & pad_mask[:, None, None, :]`; `scores = np.where(allow, scores, -1e9)`.
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
# %% md Stage 2 summary
Four operators, each pinned by a property that does not need a trained model: unit RMS, the SwiGLU hand formula, rotation invariance of RoPE scores, and attention that agrees with a triple loop while ignoring the future and padding. Keep the naive loop idea: when an optimised kernel replaces your code, the slow version becomes the oracle.
# %% report
Why is it harmless that the pad mask uses -1e9 instead of -inf, and what would go wrong with -inf on a fully padded query row?
# %% md Stage 3: the full forward pass
## Putting a decoder together

One block, with residual connections and pre-norm:
$$h = x + \text{MHA}(\text{rmsnorm}(x)),\qquad y = h + \text{SwiGLU}(\text{rmsnorm}(h)).$$
The model embeds token ids, applies `n_layers` blocks, applies a final RMSNorm and multiplies by the output head to get `logits (B, T, V)`. Row `t` of the logits is the model's prediction for token `t+1`.

With std 0.05 weights the logits are close to zero, so an untrained model is almost uniform and its cross-entropy should be close to $\ln V = \ln 259 \approx 5.557$. That is a famous sanity check for any language-model implementation: a wildly different initial loss means a bug.
# %% task The full forward pass
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
Residual: `x = x + mha(rmsnorm(x, attn_norm), ...)`, then `x = x + swiglu(rmsnorm(x, mlp_norm), ...)`.
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
# %% md Stage 3 summary
A complete transformer in about twenty lines, plus two model-level properties (prefix and last-token) that will later let you prove that the KV cache is exact. Notice the cost: the uncached forward pass recomputes everything for every prefix.
# %% report
Predict, before measuring, how the initial loss would change if you forgot the 1/sqrt(dh) scale or doubled `std`. Then try it and write what you saw.
# %% md Stage 4: sampling and generation
## From logits to a token

The last row of the logits is a vector of `V` scores. Decoding strategies differ only in how they turn it into a token:

- **greedy**: take the argmax (temperature 0);
- **temperature** $\tau$: divide the logits by $\tau$ before the softmax; $\tau<1$ sharpens, $\tau>1$ flattens;
- **top-k**: keep only the `k` highest logits and renormalise;
- **top-p** (nucleus): sort probabilities in decreasing order and keep the smallest prefix whose cumulative mass reaches `p`, then renormalise.

Order used here: temperature, then top-k, then top-p. Your function returns the **final probability vector** so you can test it exactly.
# %% task Sampling distribution: temperature, top-k, top-p
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
Top-k: `kth = np.sort(z)[-k]`, then `z = np.where(z >= kth, z, -np.inf)` before the softmax.
# %% hint
Top-p: `order = np.argsort(-p)`, `cs = np.cumsum(p[order])`, `n = np.searchsorted(cs, top_p - 1e-12) + 1`.
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
# %% md The uncached generation loop
This loop is given. It is the **slow, obviously correct reference**: at every step it re-runs the whole model on the entire sequence so far and samples from the last row. Your KV-cache version must reproduce its output token for token.
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
# %% md Stage 4 summary
Sampling is a pure function from a logit vector to a probability vector plus one random draw, which is why it is easy to test exactly. A single `rng` object threaded through the loop makes sampled runs reproducible, and it will let us compare cached and uncached generation even when sampling is random.
# %% report
When would you pick top-p over top-k? Give a concrete next-token distribution where top-k=40 is clearly wrong in one direction.
# %% md Stage 5: the KV cache
## Do not recompute what you already know

In the uncached loop, step `n` recomputes keys and values for all `n-1` earlier tokens, although they cannot change (causal masking means earlier positions never see later ones). A **KV cache** stores the **post-RoPE keys** and the values of every layer, shaped `(B, H, S, dh)` where `S` is the number of tokens seen so far. A step then only processes the **new** tokens: it computes their `q, k, v`, appends `k, v` to the cache and attends over the whole cache.

Two details decide correctness:

- new tokens at absolute positions `start .. start+T-1` need RoPE with **those** positions;
- query `i` of the chunk (absolute position `start+i`) may see cache index `j <= start+i`. With `T=1` that is every cached key; with a longer prefill chunk it is still causal.
# %% task Attention with a KV cache
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
`K = np.concatenate([cache["k"], k], axis=2)` when the cache is not empty, then store K back into the dict.
# %% hint
`allow = np.arange(S)[None, :] <= (start + np.arange(T))[:, None]` has shape (T, S).
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
# %% task Cached generation (prefill + decode)
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
Keep `out` and a counter. After sampling token `t`, if you need another token, run `forward_cached(params, cfg, np.array([[t]]), cache, len(prompt_ids) + len(out) - 1)`.
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
# %% md Proof by experiment: several seeds
Equal tokens could be luck on one seed. The statement we want is stronger: **the logits agree to floating-point rounding for any weights**. Here is the experiment over 8 random models and 3 prompts each; the numbers are mean ± std over the 24 runs of the largest absolute logit difference between the uncached forward and the cached step-by-step forward.
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
# %% md Failure analysis: where does a wrong cache first diverge?
A cache bug rarely crashes. It makes the model *slightly* different, and the symptom shows up many tokens later as degraded text. The professional move is to **localise the first divergence**: compare the incremental logits with the reference row by row and report the first position that disagrees.
# %% task First divergence between two logit tensors
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
`d = np.abs(ref - test).max(axis=(0, 2))` gives one number per position; `np.flatnonzero(d > tol)`.
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
# %% md Stage 5 summary
The cache is exact: the largest logit difference is rounding noise (about 1e-15), not "approximately the same". You also have a debugging tool that points at the first wrong row. Two rules to remember: store keys after RoPE, and the position/mask offset is the cache length.
# %% report
Your teammate says "the cached output is 1e-4 different from the uncached one, that's just float noise". Using this stage, argue whether that is plausible and what you would test next.
# %% md Stage 6: cost accounting
## FLOPs and bytes

We count only matrix multiplications (norms, RoPE and softmax are lower-order terms) and use the convention **one multiply-add = 2 FLOPs**.

For `t_new` new tokens when `t_past` tokens are already cached:

- linear layers per token and per layer: $2(4d^2 + 3d\,ff)$ (the four attention projections and three MLP matrices);
- attention per layer: token `i` of the chunk sees $t_{past}+i+1$ keys, and scores plus weighted sum cost $2\,d$ FLOPs each per visible key, so $4d$ per visible key, with $\sum_i (t_{past}+i+1) = t_{new}\,t_{past} + t_{new}(t_{new}+1)/2$ visible pairs;
- output head: $2\,d\,V$ per token.

A whole generation then costs the sum of its steps: the uncached loop re-processes `L+k` tokens at step `k`, the cached loop processes the prompt once and one token afterwards.
# %% task FLOP accounting
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
Per-layer attention FLOPs: `4 * d * vis` with `vis = t_new*t_past + t_new*(t_new+1)//2`.
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
# %% md Memory
Weights cost `params * bytes_per_param`. The KV cache costs, per token and per layer, one key and one value vector for every KV head: `2 * n_kv_heads * dh * bytes`. For multi-head attention `n_kv_heads = H`; grouped-query attention (GQA) shares each key/value head among several query heads, shrinking the cache without touching the weights' query side.
# %% task KV-cache memory and batch capacity
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
`dh = d_model // n_heads`; the cache term is `2 * n_layers * batch * seq_len * n_kv_heads * dh * bytes_per`.
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
# %% md Stage 6 summary
Two facts fall out. (1) Without a cache the work grows with the square of the output length; with it, linearly. (2) The cache is not free: at 4096 tokens it is 2 GiB per sequence for a 7B model, which caps the batch size on a 80 GB device at about 30 and is why GQA, quantised caches and paged allocation exist.
# %% report
Compute the ratio between uncached and cached FLOPs for a 512-token prompt and 512 generated tokens on the 7B-class config (use your functions). Is the ratio what you expected? Which assumption in "FLOPs" ignores the real cost of the cache?
# %% md Stage 7: batching with padding
## Serving several prompts at once

Prompts have different lengths, but a batch is a rectangle. We **left-pad** with `PAD` so the last column of every row is a real token (the one we read the next-token logits from). Padding must be invisible:

- `pad_mask (B, T)`: True where the token is real; padded keys are masked out of attention;
- `positions (B, T)`: the position of each real token counted from the first real token (0, 1, 2 ...) and 0 on padding, so RoPE sees the same positions as in an unpadded run.

If you forget either, the answer for a prompt depends on who else is in its batch, which is a very real production bug.
# %% task Left-padded batches
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
# %% md Stage 7 summary
Batching changes throughput, never answers. The equivalence test above is the template for every serving optimisation: run the slow single-sequence path as the oracle and compare logits on real positions. Wasted work is the price: with left padding the padded cells still cost FLOPs.
# %% report
The padded cells are computed but thrown away. How would continuous batching or length bucketing reduce that waste? Estimate the waste for prompt lengths [3, 50, 51, 52].
# %% md Stage 8: simulated latency benchmark
## A roofline model of decoding

We do not time the machine (it is a Python sandbox). We **simulate** a device with two numbers: peak compute `flops` (FLOP/s) and memory bandwidth `bw` (bytes/s). One decoding step for `batch` sequences with `ctx` visible tokens needs

- compute time: `batch * forward_flops(cfg, 1, ctx-1) / flops`,
- memory time: `(param_bytes + kv_cache_bytes(cfg, batch, ctx)) / bw` (every weight and every cached key/value is read once per step),

and takes `max` of the two. Whichever is larger names the bound. The device below resembles a datacentre GPU (312 TFLOP/s fp16, 2 TB/s).
# %% code
HW = dict(flops=312e12, bw=2.0e12)
# %% task Roofline time of one decode step
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
# %% md The benchmark table
Now we combine everything. The uncached loop redoes the whole prefix every step, which is a compute-heavy prefill each time; the cached loop does one prefill and then roofline-limited decode steps. All numbers are simulated for the 7B-class config on `HW`.
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
# %% md Wrap-up and what the table says
Reading the table: decoding with a cache is bound by moving bytes, so a single sequence gets about 145 tokens/s regardless of how clever the arithmetic is, and throughput scales almost linearly with batch size until the cache reads or the compute roof appears. That is the real motivation for batching, quantisation (fewer bytes per weight), GQA (a smaller cache) and speculative decoding (more tokens per weight read).
# %% report
Final report. (1) List the three tests that gave you the most confidence in your engine and what bug each would catch. (2) Which part of the roofline model is least realistic (kernel launch overhead, interconnect, attention being a separate memory-bound kernel)? (3) If you had to add GQA, which functions would change and which tests would you update first?
# %% md Sources
Only material that was actually opened and read while writing this project:

- `docs/PROJECTS.md` and `docs/AGENT_BRIEF.md` of this repository (format and rules).
- Front matter (id, title, requires) of the lessons `trf.01`-`trf.04`, `llm.01`, `llm.02`, `sys.01` and `sys.05` in `content/authored/` to choose the prerequisites. The lesson bodies were not studied line by line: the equations in this project are standard (RoPE half-split rotation, RMSNorm, SwiGLU, the FLOP and byte formulas above) and every numeric claim is checked by the cells here.
# %% md Limitations
Random weights: nothing here says anything about text quality. FLOP counts ignore norms, softmax, RoPE and the embedding gather; the roofline treats the device as two numbers and ignores kernel overheads, communication and cache paging.
