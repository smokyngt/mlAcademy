# %% meta
{"id": "proj.autograd", "title": "Build a mini autograd framework in NumPy", "summary": "Write a tensor autodiff engine from the ground up: broadcasting-aware gradients, topological ordering and accumulation, ten differentiable ops each verified by finite differences, Linear/Sequential modules, SGD and Adam, then train an MLP on moons and show your framework reproduces the hand-derived backprop of the backpropagation lesson.", "level": "advanced", "minutes": 150, "requires": ["f2n.10", "f2n.12", "dl.01"], "outcomes": ["Implement reverse-mode automatic differentiation on NumPy arrays with a topological sort and gradient accumulation", "Explain and implement unbroadcasting, the reverse of NumPy broadcasting", "Derive and verify the backward rule of add, mul, matmul, sum, mean, exp, log, ReLU, tanh and softmax cross-entropy with numerical gradient checks", "Compose Linear layers, a Sequential container and the SGD and Adam optimisers into a trainable model", "Prove that autograd gradients equal manually derived backprop gradients to machine precision"], "stages": ["Graph machinery", "Differentiable ops", "Modules and optimisers", "Training on moons", "Matching hand-derived backprop", "Report"]}
# %% md Welcome
# A mini autograd framework

PyTorch's `loss.backward()` looks like magic. In this project you build the same mechanism in about 250 lines of NumPy, and then use it to train a network. The goal is not speed, it is understanding: every line of the reverse pass is something you will have written and tested.

You build on [[f2n.10]] (the hand-derived gradients of a two-layer MLP: `dZ2 = (P - onehot(y)) / B`, `dW2 = A1.T @ dZ2`, ...), [[f2n.12]] (the computational graph, reverse mode, topological order, accumulation with `+=`) and [[dl.01]] (the SGD-with-momentum and Adam updates).

## The design

- A `Tensor` wraps a float64 `data` array and (after a backward pass) a `grad` array of the same shape. It records its `_prev` parents and a `_backward` closure that pushes `out.grad` into the parents' `grad`.
- Every op is a plain function (`add`, `mul`, `matmul`, ...) that builds the output Tensor and attaches the closure. `Tensor` methods and operators (`a + b`, `a @ b`, `x.sum()`) just call those functions, so you can implement them in any order.
- `backward()` sorts the graph topologically, seeds the output gradient, and calls the closures in reverse order. Gradients **accumulate** (`+=`) because a tensor can feed several consumers.
- NumPy broadcasting makes shapes grow silently in the forward pass; the backward pass must **unbroadcast** (sum back) so that a gradient always has the shape of its tensor.

Each task is verified by a hidden check; the numerical gradient check you write in Stage 1 is later used on every op.

# %% md Stage 1: graph machinery
## Stage 1: Graph machinery

Four small pieces make reverse-mode work for any op: `unbroadcast`, `accumulate`, `topo_order` and `backprop`. You first write them and test them on hand-built graphs, before any real op exists.

First, the `Tensor` class itself. It is given to you; read it, since the rest of the project hangs from it. Note that operators dispatch to functions you will write later (Python looks the names up at call time).

# %% code
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _prev=(), _op=""):
        self.data = np.array(data, dtype=np.float64)       # always a float64 copy
        self.requires_grad = bool(requires_grad) or any(p.requires_grad for p in _prev)
        self.grad = None                                    # same shape as data once filled
        self._prev = tuple(_prev)                           # parent Tensors
        self._backward = lambda: None                       # pushes self.grad into the parents
        self._op = _op

    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def zero_grad(self): self.grad = None
    def backward(self, grad=None): backprop(self, grad)
    def __repr__(self): return f"Tensor({self.data!r}, op={self._op!r})"

    def __add__(self, o): return add(self, as_tensor(o))
    def __radd__(self, o): return add(as_tensor(o), self)
    def __mul__(self, o): return mul(self, as_tensor(o))
    def __rmul__(self, o): return mul(as_tensor(o), self)
    def __neg__(self): return mul(self, as_tensor(-1.0))
    def __sub__(self, o): return add(self, -as_tensor(o))
    def __rsub__(self, o): return add(as_tensor(o), -self)
    def __truediv__(self, o): return mul(self, pow_const(as_tensor(o), -1.0))
    def __rtruediv__(self, o): return mul(as_tensor(o), pow_const(self, -1.0))
    def __pow__(self, p): return pow_const(self, p)
    def __matmul__(self, o): return matmul(self, as_tensor(o))
    def sum(self, axis=None, keepdims=False): return tsum(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False): return tmean(self, axis, keepdims)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def relu(self): return relu(self)
    def tanh(self): return tanh(self)

def as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)
# %% md A worked example of an op
Here is one complete op, `pow_const` ($c = a^p$ for a constant exponent $p$, $\partial c/\partial a = p\,a^{p-1}$). Study the pattern, you will repeat it for every op: compute the forward value, create the output with its parents, define a closure that calls `accumulate(parent, local_gradient)`, attach it.

# %% code
def pow_const(a, p):
    out = Tensor(a.data ** p, _prev=(a,), _op=f"pow{p}")
    def _backward():
        accumulate(a, out.grad * p * a.data ** (p - 1))
    out._backward = _backward
    return out
# %% task Unbroadcast
def unbroadcast(grad, shape):
    """Reduce `grad` (an array that was produced by broadcasting something of `shape`) back to `shape`.
    1. Sum away the extra LEADING axes. 2. For every axis where `shape` has size 1 but grad has size > 1,
    sum over that axis keeping the dimension. Return an array with exactly `shape`."""
    g = np.asarray(grad, dtype=np.float64)
    # TODO
    raise NotImplementedError
# %% check
_g = np.arange(12.0).reshape(3, 4)
check("(3,4) -> (4,) sums over rows", lambda: np.allclose(unbroadcast(_g, (4,)), _g.sum(0)) and unbroadcast(_g, (4,)).shape == (4,))
check("(3,4) -> (3,1) sums over columns and keeps the axis", lambda: np.allclose(unbroadcast(_g, (3, 1)), _g.sum(1, keepdims=True)) and unbroadcast(_g, (3, 1)).shape == (3, 1))
check("(3,4) -> () sums everything", lambda: np.shape(unbroadcast(_g, ())) == () and np.isclose(unbroadcast(_g, ()), _g.sum()))
check("same shape is returned unchanged", lambda: np.array_equal(unbroadcast(_g, (3, 4)), _g))
_g3 = np.arange(24.0).reshape(2, 3, 4)
check("(2,3,4) -> (3,1) sums the leading axis AND the last axis", lambda: np.allclose(unbroadcast(_g3, (3, 1)), _g3.sum(0).sum(1, keepdims=True)) and unbroadcast(_g3, (3, 1)).shape == (3, 1))
check("the total of the gradient is preserved", lambda: np.isclose(unbroadcast(_g3, (1, 4)).sum(), _g3.sum()), "Unbroadcasting only adds numbers up, it never drops or averages them")
# %% hint
`while g.ndim > len(shape): g = g.sum(axis=0)` removes the leading axes.
# %% hint
Then `for i, s in enumerate(shape): if s == 1 and g.shape[i] != 1: g = g.sum(axis=i, keepdims=True)`.
# %% solution
def unbroadcast(grad, shape):
    g = np.asarray(grad, dtype=np.float64)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g
# %% task Accumulate a gradient
def accumulate(t, g):
    """Add the gradient contribution `g` into t.grad (the single place where gradients are written).
    - Do nothing if t does not require grad.
    - Unbroadcast g to t.data.shape first.
    - If t.grad is None create zeros of t.data's shape, then ADD (never overwrite, never alias g)."""
    # TODO
    raise NotImplementedError
# %% check
_t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
accumulate(_t, np.array([1.0, 1.0, 1.0])); accumulate(_t, np.array([0.5, 0.0, 2.0]))
check("two contributions add up", lambda: np.allclose(_t.grad, [1.5, 1.0, 3.0]), "Use +=, a second call must not overwrite the first")
_c = Tensor([1.0, 2.0])
accumulate(_c, np.ones(2))
check("a tensor that does not require grad keeps grad None", lambda: _c.grad is None)
_b = Tensor(np.zeros((2, 3)), requires_grad=True)
accumulate(_b, np.ones((4, 2, 3)))
check("contributions are unbroadcast to the tensor's shape", lambda: _b.grad.shape == (2, 3) and np.allclose(_b.grad, 4.0))
_s = Tensor(2.0, requires_grad=True)
accumulate(_s, np.ones((3, 3)))
check("a scalar tensor receives the sum of a broadcast gradient", lambda: np.shape(_s.grad) == () and np.isclose(_s.grad, 9.0))
_src = np.ones(3); _u = Tensor(np.zeros(3), requires_grad=True)
accumulate(_u, _src); _src[:] = 100.0
check("t.grad does not alias the array that was passed in", lambda: np.allclose(_u.grad, 1.0), "Start from np.zeros_like(t.data) and add into it")
# %% hint
`if not t.requires_grad: return`, then `g = unbroadcast(g, t.data.shape)`, `if t.grad is None: t.grad = np.zeros_like(t.data)`, `t.grad += g`.
# %% solution
def accumulate(t, g):
    if not t.requires_grad:
        return
    g = unbroadcast(g, t.data.shape)
    if t.grad is None:
        t.grad = np.zeros_like(t.data)
    t.grad += g
# %% task Topological order
def topo_order(root):
    """Return the list of all Tensors reachable from `root` through `_prev`, each exactly once, ordered so that
    every tensor appears AFTER all of its parents (the root is last). Must be iterative: graphs can be
    thousands of nodes deep and Python's recursion limit is about 1000. Identify nodes with id(node)."""
    # TODO
    raise NotImplementedError
# %% check
def _mk(data, parents, fn):
    out = Tensor(data, _prev=parents)
    out._backward = lambda: fn(out)
    return out
_a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
_b = _mk(2 * _a.data, (_a,), lambda o: accumulate(_a, 2 * o.grad))
_c = _mk(3 * _a.data, (_a,), lambda o: accumulate(_a, 3 * o.grad))
_d = _mk(_b.data + _c.data, (_b, _c), lambda o: (accumulate(_b, o.grad), accumulate(_c, o.grad)))
_ord = topo_order(_d)
check("a diamond graph has 4 distinct nodes, the root last, the leaf first", lambda: len(_ord) == 4 and _ord[-1] is _d and _ord[0] is _a)
check("a shared parent appears once", lambda: sum(1 for n in _ord if n is _a) == 1)
check("every node comes after its parents", lambda: all(_ord.index(p) < _ord.index(n) for n in _ord for p in n._prev))
_chain = [Tensor(1.0, requires_grad=True)]
for _i in range(3000):
    _chain.append(Tensor(1.0, _prev=(_chain[-1],)))
check("a chain of 3000 nodes does not hit the recursion limit and is ordered", lambda: topo_order(_chain[-1]) == _chain, "Use an explicit stack of (node, expanded) pairs instead of recursion")
_other = Tensor(5.0, requires_grad=True)
check("unreachable nodes are not included", lambda: all(n is not _other for n in topo_order(_d)))
_e = _mk(_b.data + _c.data + _a.data, (_d, _b, _a), lambda o: None)
_ord2 = topo_order(_e)
check("a node reachable by paths of different lengths still comes after all parents", lambda: len(_ord2) == 5 and all(_ord2.index(p) < _ord2.index(n) for n in _ord2 for p in n._prev))
# %% hint
Keep `order = []`, `seen = set()`, `stack = [(root, False)]`. Pop `(node, expanded)`. If `expanded`: append to order. Else if id(node) not in seen: mark seen, push `(node, True)`, then push `(p, False)` for each parent not yet seen.
# %% solution
def topo_order(root):
    order, seen = [], set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node._prev:
            if id(p) not in seen:
                stack.append((p, False))
    return order
# %% task The backward pass
def backprop(root, seed=None):
    """Reverse-mode differentiation from `root`.
    1. Compute the topological order.
    2. Reset the gradient of every NON-leaf node (a node with parents) to None: intermediate gradients belong
       to ONE backward pass. Leaves keep what they have, so calling backward twice doubles leaf gradients.
    3. Seed the root: accumulate(root, seed) with seed defaulting to ones of root's shape.
    4. Call node._backward() for the nodes in REVERSE topological order, but only for nodes that require grad
       (a node computed only from constants never receives a gradient, so its closure must not run)."""
    # TODO
    raise NotImplementedError
# %% check
_a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
_b = _mk(2 * _a.data, (_a,), lambda o: accumulate(_a, 2 * o.grad))
_c = _mk(3 * _a.data, (_a,), lambda o: accumulate(_a, 3 * o.grad))
_d = _mk(_b.data + _c.data, (_b, _c), lambda o: (accumulate(_b, o.grad), accumulate(_c, o.grad)))
_d.backward()
check("diamond: the two paths 2 + 3 add up to 5 for each entry", lambda: np.allclose(_a.grad, 5.0), "accumulate, do not overwrite; process the root first")
check("intermediate nodes received the seed (b.grad == 1)", lambda: np.allclose(_b.grad, 1.0) and np.allclose(_d.grad, 1.0))
_d.backward()
check("calling backward twice doubles a leaf gradient (and only that)", lambda: np.allclose(_a.grad, 10.0) and np.allclose(_b.grad, 1.0), "Reset non-leaf grads at the start of backprop")
_a2 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
_o = _mk(4 * _a2.data, (_a2,), lambda o: accumulate(_a2, 4 * o.grad))
_o.backward(np.array([1.0, 10.0]))
check("an explicit seed is used", lambda: np.allclose(_a2.grad, [4.0, 40.0]))
_k = Tensor(np.array([1.0]))
_kk = Tensor(np.array([7.0]), requires_grad=True)
_m = _mk(_k.data + _kk.data, (_k, _kk), lambda o: (accumulate(_k, o.grad), accumulate(_kk, o.grad)))
_m.backward()
check("a leaf that does not require grad stays None", lambda: _k.grad is None and np.allclose(_kk.grad, 1.0))
# %% hint
`order = topo_order(root)`, `for n in order: if n._prev: n.grad = None`, `accumulate(root, np.ones_like(root.data) if seed is None else seed)`, then `for n in reversed(order): if n.requires_grad: n._backward()`.
# %% solution
def backprop(root, seed=None):
    order = topo_order(root)
    for n in order:
        if n._prev:
            n.grad = None
    if seed is None:
        seed = np.ones_like(root.data)
    accumulate(root, seed)
    for n in reversed(order):
        if n.requires_grad:
            n._backward()
# %% md Numerical gradients
A backward pass is only trustworthy once it is tested against something independent: the central difference $\partial f/\partial x_i \approx (f(x+h e_i) - f(x - h e_i))/2h$. This is exactly the check from [[f2n.10]] and [[f2n.12]], now written as a reusable tool on plain NumPy functions (no Tensor involved, so it cannot share a bug with your engine).

# %% task Numerical gradient
def numeric_grad(f, x, h=1e-6):
    """f maps a float array to a Python float. Return the central-difference gradient, same shape as x.
    Do not modify the array the caller passed in."""
    x = np.array(x, dtype=np.float64)
    # TODO
    raise NotImplementedError
# %% check
_x = np.array([[1.0, -2.0], [0.5, 3.0]])
check("quadratic: grad of sum(x^2) is 2x", lambda: np.allclose(numeric_grad(lambda a: float((a ** 2).sum()), _x), 2 * _x, atol=1e-7))
_w = np.array([[1.0, 2.0], [3.0, 4.0]])
check("linear: grad of sum(x * w) is w", lambda: np.allclose(numeric_grad(lambda a: float((a * _w).sum()), _x), _w, atol=1e-7))
check("nonlinear: grad of sum(sin(x)) is cos(x)", lambda: np.allclose(numeric_grad(lambda a: float(np.sin(a).sum()), _x), np.cos(_x), atol=1e-7))
_keep = _x.copy(); numeric_grad(lambda a: float(a.sum()), _x)
check("the input array is not modified", lambda: np.array_equal(_x, _keep))
check("scalar input (0-d array) works and the shape is preserved", lambda: np.shape(numeric_grad(lambda a: float(a ** 3), np.array(2.0))) == () and abs(float(numeric_grad(lambda a: float(a ** 3), np.array(2.0))) - 12.0) < 1e-5)
# %% hint
`g = np.zeros_like(x)`; `for idx in np.ndindex(*x.shape):` save `old = x[idx]`, set `x[idx] = old + h`, evaluate, `x[idx] = old - h`, evaluate, restore.
# %% solution
def numeric_grad(f, x, h=1e-6):
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + h
        fp = f(x)
        x[idx] = old - h
        fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * h)
    return g
# %% md The op checker
`op_grad_error(fn, *arrays)` is the test harness for the whole project. It runs `fn` on Tensors, backpropagates a **random fixed upstream gradient** `W` (so that every output entry matters, not only their sum), and compares each input's `.grad` with `numeric_grad` of the scalar `sum(fn(...).data * W)`. It returns the worst error `max |a - n| / (1 + |a| + |n|)`. A correct op gives about $10^{-9}$ or less.

# %% code
def op_grad_error(fn, *arrays, seed=0):
    rng = np.random.default_rng(seed)
    ts = [Tensor(a, requires_grad=True) for a in arrays]
    out = fn(*ts)
    W = rng.normal(size=out.shape)
    out.backward(W)
    worst = 0.0
    for i, a in enumerate(arrays):
        def f(x, i=i):
            args = [Tensor(x if j == i else arrays[j]) for j in range(len(arrays))]
            return float((fn(*args).data * W).sum())
        num = numeric_grad(f, np.asarray(a, dtype=float))
        ana = ts[i].grad if ts[i].grad is not None else np.zeros_like(num)
        assert ana.shape == num.shape, f"input {i}: gradient shape {ana.shape} != {num.shape}"
        worst = max(worst, float(np.max(np.abs(ana - num) / (1 + np.abs(ana) + np.abs(num)))))
    return worst
# %% md Stage 1 summary
**What you learned.** Reverse mode needs four ingredients: an order in which every node comes after its parents, a seed, an accumulating write (`+=`), and shape repair after broadcasting. Non-leaf gradients are scratch space for one pass; leaf gradients persist, which is why training loops must zero them. A numerical check on plain NumPy is the independent referee for everything that follows.

# %% report
Draw (or describe) the graph of `f = a*b + a**2` and write the two contributions that reach `a.grad`. Then explain what would go wrong if `accumulate` used `=` instead of `+=`.
# %% md Stage 2: differentiable ops
## Stage 2: The ops

Each op follows the `pow_const` pattern. The backward closures call `accumulate`, which unbroadcasts, so you only write the *local* derivative multiplied by `out.grad`:

| op | forward | backward into inputs |
|---|---|---|
| `add(a, b)` | $a+b$ | $\bar a \mathrel{+}= \bar c,\ \bar b \mathrel{+}= \bar c$ |
| `mul(a, b)` | $a\odot b$ | $\bar a \mathrel{+}= \bar c\odot b,\ \bar b \mathrel{+}= \bar c\odot a$ |
| `matmul(a, b)` | $AB$ | $\bar A \mathrel{+}= \bar C B^\top,\ \bar B \mathrel{+}= A^\top \bar C$ |
| `tsum`, `tmean` | reduce over axes | broadcast $\bar c$ back (and divide by the count for the mean) |
| `exp`, `log`, `relu`, `tanh` | elementwise | $c\bar c,\ \bar c/a,\ \bar c\,[a>0],\ (1-c^2)\bar c$ |
| `softmax_cross_entropy(z, y)` | mean CE | $\frac{1}{B}(P-\mathrm{onehot}(y))\,\bar c$ |

The `matmul` rule is the same pair of formulas you derived for `dW2 = A1.T @ dZ2` and `dA1 = dZ2 @ W2.T` in [[f2n.10]].

# %% task add and mul
def add(a, b):
    """Elementwise sum with NumPy broadcasting. Return a Tensor with _prev=(a, b)."""
    # TODO
    raise NotImplementedError

def mul(a, b):
    """Elementwise product with NumPy broadcasting."""
    # TODO
    raise NotImplementedError
# %% check
_rng = np.random.default_rng(10)
_A = _rng.normal(size=(3, 4)); _B = _rng.normal(size=(4,)); _C = _rng.normal(size=(3, 1))
check("forward values equal NumPy (with broadcasting)", lambda: np.allclose(add(Tensor(_A), Tensor(_B)).data, _A + _B) and np.allclose(mul(Tensor(_A), Tensor(_C)).data, _A * _C))
check("add gradients: (3,4) + (4,) [row broadcast]", lambda: op_grad_error(add, _A, _B) < 1e-7, "The bias-like (4,) operand must receive the SUM over rows; accumulate/unbroadcast does that if you pass out.grad")
check("mul gradients: (3,4) * (3,1) [column broadcast]", lambda: op_grad_error(mul, _A, _C) < 1e-7)
check("mul gradients: (3,1) * (4,) -> (3,4) [both operands broadcast]", lambda: op_grad_error(mul, _C, _B) < 1e-7)
check("scalar (0-d) operands", lambda: op_grad_error(add, _A, np.array(1.5)) < 1e-7 and op_grad_error(mul, np.array(2.0), _A) < 1e-7)
_x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
(_x * _x).backward()
check("using the same tensor twice accumulates: d(x*x)/dx = 2x", lambda: np.allclose(_x.grad, 2 * _x.data))
check("operators and derived ops (-, /, constants) work end to end", lambda: op_grad_error(lambda a, b: (a * b + a - b / 2.0) * 3.0 - 1.0 / (b * b + 1.0), _A, _A[::-1].copy()) < 1e-7)
_k = Tensor(np.array([2.0, 2.0]))
_v = Tensor(np.array([1.0, 5.0]), requires_grad=True)
(_k * _v).backward()
check("a constant operand gets no gradient", lambda: _k.grad is None and np.allclose(_v.grad, 2.0))
# %% hint
```
out = Tensor(a.data + b.data, _prev=(a, b), _op="add")
def _backward():
    accumulate(a, out.grad)
    accumulate(b, out.grad)
out._backward = _backward
return out
```
# %% hint
For `mul` the gradient into `a` is `out.grad * b.data` and into `b` is `out.grad * a.data`. Do not unbroadcast yourself: `accumulate` does it.
# %% solution
def add(a, b):
    out = Tensor(a.data + b.data, _prev=(a, b), _op="add")
    def _backward():
        accumulate(a, out.grad)
        accumulate(b, out.grad)
    out._backward = _backward
    return out

def mul(a, b):
    out = Tensor(a.data * b.data, _prev=(a, b), _op="mul")
    def _backward():
        accumulate(a, out.grad * b.data)
        accumulate(b, out.grad * a.data)
    out._backward = _backward
    return out
# %% task matmul
def matmul(a, b):
    """Matrix product of two 2-D tensors, shapes (m, k) @ (k, n) -> (m, n)."""
    # TODO
    raise NotImplementedError
# %% check
_A = _rng.normal(size=(3, 4)); _B = _rng.normal(size=(4, 2)); _D = _rng.normal(size=(2, 5))
check("forward equals NumPy and has shape (3, 2)", lambda: matmul(Tensor(_A), Tensor(_B)).shape == (3, 2) and np.allclose(matmul(Tensor(_A), Tensor(_B)).data, _A @ _B))
check("gradients of both operands (3,4) @ (4,2)", lambda: op_grad_error(matmul, _A, _B) < 1e-7, "dA = dC @ B.T and dB = A.T @ dC")
check("the @ operator and a chain a @ b @ d", lambda: op_grad_error(lambda a, b, d: a @ b @ d, _A, _B, _D) < 1e-7)
_S = _rng.normal(size=(3, 3))
check("a square matrix multiplied with itself (shared operand)", lambda: op_grad_error(lambda s: s @ s, _S) < 1e-7, "Both uses of s contribute; this only works if gradients accumulate")
check("a single-row batch (1, 4) @ (4, 2)", lambda: op_grad_error(matmul, _A[:1], _B) < 1e-7)
_X = Tensor(_A); _Wt = Tensor(_B, requires_grad=True)
(_X @ _Wt).backward(np.ones((3, 2)))
check("a constant input matrix (data) gets no gradient but the weights do", lambda: _X.grad is None and np.allclose(_Wt.grad, _A.sum(0)[:, None] * np.ones((1, 2))))
# %% hint
`out = Tensor(a.data @ b.data, _prev=(a, b), _op="matmul")`. Backward: `accumulate(a, out.grad @ b.data.T)` and `accumulate(b, a.data.T @ out.grad)`. Check the shapes: (m,n)(n,k) = (m,k) and (k,m)(m,n) = (k,n).
# %% solution
def matmul(a, b):
    out = Tensor(a.data @ b.data, _prev=(a, b), _op="matmul")
    def _backward():
        accumulate(a, out.grad @ b.data.T)
        accumulate(b, a.data.T @ out.grad)
    out._backward = _backward
    return out
# %% task Reductions: sum and mean
def tsum(a, axis=None, keepdims=False):
    """Sum over `axis` (None, an int, or a tuple of ints), like np.sum. The backward pass must broadcast
    out.grad back to a.data.shape; when keepdims is False the reduced axes have to be re-inserted first
    (np.expand_dims) before broadcasting with np.broadcast_to."""
    # TODO
    raise NotImplementedError

def tmean(a, axis=None, keepdims=False):
    """Mean over `axis`: the sum divided by the number of reduced elements (backward: spread out.grad / count)."""
    # TODO
    raise NotImplementedError
# %% check
_A = _rng.normal(size=(3, 4, 2))
check("tsum values: all, axis, tuple, keepdims, negative axis", lambda: np.isclose(tsum(Tensor(_A)).data, _A.sum()) and np.allclose(tsum(Tensor(_A), 1).data, _A.sum(1)) and tsum(Tensor(_A), (0, 2), True).shape == (1, 4, 1) and np.allclose(tsum(Tensor(_A), -1).data, _A.sum(-1)))
check("tsum gradients for every axis choice", lambda: max(op_grad_error(lambda t, ax=ax, kd=kd: tsum(t, ax, kd), _A) for ax in (None, 0, 1, -1, (0, 2)) for kd in (False, True)) < 1e-7, "Without keepdims, re-insert the reduced axes with np.expand_dims before np.broadcast_to")
check("tmean values", lambda: np.isclose(tmean(Tensor(_A)).data, _A.mean()) and np.allclose(tmean(Tensor(_A), (0, 1)).data, _A.mean((0, 1))))
check("tmean gradients for every axis choice", lambda: max(op_grad_error(lambda t, ax=ax, kd=kd: tmean(t, ax, kd), _A) for ax in (None, 0, 2, (1, 2)) for kd in (False, True)) < 1e-7)
_z = Tensor(np.ones((4, 5)), requires_grad=True)
_z.mean().backward()
check("mean of 20 ones: every gradient entry is 1/20", lambda: np.allclose(_z.grad, 1 / 20))
_q = Tensor(np.ones(3), requires_grad=True)
(_q.sum() * 2.0).backward()
check("methods and scalars compose: d(2*sum(q))/dq = 2", lambda: np.allclose(_q.grad, 2.0) and tsum(Tensor(_A)).shape == ())
# %% hint
For `sum`: `out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), _prev=(a,))`. In the closure: `g = out.grad`; `if axis is not None and not keepdims: g = np.expand_dims(g, axis)`; then `accumulate(a, np.broadcast_to(g, a.data.shape))`.
# %% hint
`np.expand_dims` accepts an int or a tuple of axes, including negative ones, so the same code handles every case.
# %% hint
`tmean`: `count = a.data.size / out.data.size` (number of elements averaged into each output entry) and divide the spread gradient by it.
# %% solution
def tsum(a, axis=None, keepdims=False):
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), _prev=(a,), _op="sum")
    def _backward():
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        accumulate(a, np.broadcast_to(g, a.data.shape))
    out._backward = _backward
    return out

def tmean(a, axis=None, keepdims=False):
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), _prev=(a,), _op="mean")
    count = a.data.size / out.data.size
    def _backward():
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        accumulate(a, np.broadcast_to(g, a.data.shape) / count)
    out._backward = _backward
    return out
# %% task Elementwise: exp, log, relu, tanh
def exp(a):
    """c = e^a. Hint: the derivative is the output itself."""
    # TODO
    raise NotImplementedError

def log(a):
    """c = ln a (natural log, a > 0)."""
    # TODO
    raise NotImplementedError

def relu(a):
    """c = max(0, a). Convention: the derivative at exactly 0 is 0, i.e. the mask is (a > 0)."""
    # TODO
    raise NotImplementedError

def tanh(a):
    """c = tanh(a). Derivative 1 - c^2."""
    # TODO
    raise NotImplementedError
# %% check
_P = _rng.normal(size=(3, 4)); _Q = np.abs(_P) + 0.5
check("exp: values and gradient", lambda: np.allclose(exp(Tensor(_P)).data, np.exp(_P)) and op_grad_error(exp, _P) < 1e-7)
check("log: values and gradient (positive inputs)", lambda: np.allclose(log(Tensor(_Q)).data, np.log(_Q)) and op_grad_error(log, _Q) < 1e-7)
check("tanh: values and gradient", lambda: np.allclose(tanh(Tensor(_P)).data, np.tanh(_P)) and op_grad_error(tanh, _P) < 1e-7)
check("relu: values and gradient away from 0", lambda: np.allclose(relu(Tensor(_P)).data, np.maximum(0, _P)) and op_grad_error(relu, _P) < 1e-7)
_r = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
relu(_r).backward()
check("relu gradient is 0 for negatives, 0 at exactly 0, 1 for positives", lambda: np.allclose(_r.grad, [0.0, 0.0, 1.0]))
check("methods and pow_const compose: tanh(x).exp() and (x**3)", lambda: op_grad_error(lambda t: t.tanh().exp() + t ** 3, _P) < 1e-7)
check("a composition of all four (chain rule through several nodes)", lambda: op_grad_error(lambda t: log(exp(relu(tanh(t) * 2.0 + 1.0)) + 1.0), _P) < 1e-7)
# %% hint
`exp`: `out = Tensor(np.exp(a.data), _prev=(a,), _op="exp")`, backward `accumulate(a, out.grad * out.data)`.
# %% hint
`log`: `accumulate(a, out.grad / a.data)`. `relu`: `accumulate(a, out.grad * (a.data > 0))`. `tanh`: `accumulate(a, out.grad * (1 - out.data ** 2))`.
# %% solution
def exp(a):
    out = Tensor(np.exp(a.data), _prev=(a,), _op="exp")
    def _backward():
        accumulate(a, out.grad * out.data)
    out._backward = _backward
    return out

def log(a):
    out = Tensor(np.log(a.data), _prev=(a,), _op="log")
    def _backward():
        accumulate(a, out.grad / a.data)
    out._backward = _backward
    return out

def relu(a):
    out = Tensor(np.maximum(0.0, a.data), _prev=(a,), _op="relu")
    def _backward():
        accumulate(a, out.grad * (a.data > 0))
    out._backward = _backward
    return out

def tanh(a):
    out = Tensor(np.tanh(a.data), _prev=(a,), _op="tanh")
    def _backward():
        accumulate(a, out.grad * (1.0 - out.data ** 2))
    out._backward = _backward
    return out
# %% task Fused softmax cross-entropy
def softmax_cross_entropy(logits, y):
    """Mean cross-entropy of softmax(logits) against integer labels y. logits: Tensor (B, K); y: int array (B,).
    Return a SCALAR Tensor. Compute log-softmax stably (subtract the row max). This is one fused op because
    the combined gradient is simple: d loss / d logits = (P - onehot(y)) / B  (multiply by out.grad)."""
    # TODO
    raise NotImplementedError
# %% check
_Z = _rng.normal(size=(5, 4)); _y = np.array([0, 3, 1, 1, 2])
def _ref_ce(z, y):
    zs = z - z.max(1, keepdims=True); lp = zs - np.log(np.exp(zs).sum(1, keepdims=True))
    return -lp[np.arange(len(y)), y].mean()
check("the loss is a scalar Tensor equal to the NumPy reference", lambda: softmax_cross_entropy(Tensor(_Z), _y).shape == () and np.isclose(softmax_cross_entropy(Tensor(_Z), _y).data, _ref_ce(_Z, _y)))
check("finite-difference gradient check", lambda: op_grad_error(lambda z: softmax_cross_entropy(z, _y), _Z) < 1e-7)
def _manual_dz(z, y):
    zs = z - z.max(1, keepdims=True); P = np.exp(zs); P /= P.sum(1, keepdims=True)
    P[np.arange(len(y)), y] -= 1; return P / len(y)
_zt = Tensor(_Z, requires_grad=True); softmax_cross_entropy(_zt, _y).backward()
check("gradient equals the hand-derived (P - onehot(y)) / B of lesson f2n.10", lambda: np.allclose(_zt.grad, _manual_dz(_Z, _y), atol=1e-12))
check("stable for huge logits (no inf / nan)", lambda: np.isfinite(softmax_cross_entropy(Tensor(_Z * 500), _y).data) and np.isfinite(op_grad_error(lambda z: softmax_cross_entropy(z, _y), _Z * 50)))
check("uniform logits give loss ln K", lambda: np.isclose(softmax_cross_entropy(Tensor(np.zeros((3, 4))), np.array([0, 1, 2])).data, np.log(4)))
_zt2 = Tensor(_Z, requires_grad=True); (softmax_cross_entropy(_zt2, _y) * 3.0).backward()
check("the upstream gradient is respected (3 * loss gives 3 * gradient)", lambda: np.allclose(_zt2.grad, 3 * _manual_dz(_Z, _y), atol=1e-12))
check("every gradient row sums to 0", lambda: np.allclose(_zt.grad.sum(1), 0, atol=1e-12))
# %% hint
`Zs = z - z.max(1, keepdims=True)`; `logp = Zs - np.log(np.exp(Zs).sum(1, keepdims=True))`; loss `= -logp[np.arange(B), y].mean()`; keep `P = np.exp(logp)` for the backward pass.
# %% hint
Backward: `d = P.copy(); d[np.arange(B), y] -= 1; accumulate(logits, d / B * out.grad)`. `out.grad` is a 0-d array here; multiplying by it is fine.
# %% solution
def softmax_cross_entropy(logits, y):
    z = logits.data
    B = len(y)
    zs = z - z.max(axis=1, keepdims=True)
    logp = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True))
    P = np.exp(logp)
    out = Tensor(-logp[np.arange(B), y].mean(), _prev=(logits,), _op="softmax_ce")
    def _backward():
        d = P.copy()
        d[np.arange(B), y] -= 1.0
        accumulate(logits, d / B * out.grad)
    out._backward = _backward
    return out
# %% md The gradient-check table
Every op, one number each. This is the table you would paste into a pull request. All errors should be below $10^{-7}$.

# %% code
_r = np.random.default_rng(42)
_M, _N, _V = _r.normal(size=(4, 3)), _r.normal(size=(3, 5)), _r.normal(size=(5,))
_Zc, _yc = _r.normal(size=(6, 4)), np.array([0, 1, 2, 3, 0, 1])
OP_TABLE = [
    ("add (broadcast)",   lambda: op_grad_error(add, _M, _V[:3])),
    ("mul (broadcast)",   lambda: op_grad_error(mul, _M, _V[:3])),
    ("matmul",            lambda: op_grad_error(matmul, _M, _N)),
    ("sum(axis=0)",       lambda: op_grad_error(lambda t: tsum(t, 0), _M)),
    ("mean(axis=1, kd)",  lambda: op_grad_error(lambda t: tmean(t, 1, True), _M)),
    ("exp",               lambda: op_grad_error(exp, _M)),
    ("log",               lambda: op_grad_error(log, np.abs(_M) + 0.5)),
    ("relu",              lambda: op_grad_error(relu, _M)),
    ("tanh",              lambda: op_grad_error(tanh, _M)),
    ("softmax_xent",      lambda: op_grad_error(lambda z: softmax_cross_entropy(z, _yc), _Zc)),
]
print(f"{'op':<20}{'max error':>12}")
for _name, _fn in OP_TABLE:
    _e = _fn()
    print(f"{_name:<20}{_e:>12.2e}")
    assert _e < 1e-7, _name + " failed its gradient check"
# %% md Stage 2 summary
**What you learned.** Every op's backward is "local derivative times upstream gradient, then accumulate". Broadcasting hides in the forward pass and is undone in `accumulate`; reductions have the opposite problem (axes disappear) and need `expand_dims`. Fusing softmax with cross-entropy turns a numerically fragile product of Jacobians into `(P - onehot) / B`. And a fixed *random* upstream gradient makes the checker exercise every output entry, which a plain `sum()` would not.

# %% report
Why does `op_grad_error` backpropagate a random weight vector `W` instead of calling `.sum()` on the output? Give an example of a wrong backward rule that a plain sum check might miss.
# %% md Stage 3: modules and optimisers
## Stage 3: Modules and optimisers

Ops are enough to write any network by hand, but a `Module` bundles parameters with the computation that uses them, and an optimiser updates a list of parameter Tensors from their `.grad`. Given: the `Module` base class, `ReLU` and `Tanh` wrappers. You write `Linear` and `Sequential`.

# %% code
class Module:
    def __call__(self, x): return self.forward(x)
    def forward(self, x): raise NotImplementedError
    def parameters(self): return []
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

class ReLU(Module):
    def forward(self, x): return relu(x)

class Tanh(Module):
    def forward(self, x): return tanh(x)
# %% task Linear and Sequential
class Linear(Module):
    def __init__(self, n_in, n_out, rng):
        """W has shape (n_in, n_out), drawn from Normal(0, 2/n_in) with rng.normal (He initialisation);
        b has shape (n_out,) of zeros. Both are Tensors with requires_grad=True: self.W, self.b."""
        # TODO
        raise NotImplementedError
    def forward(self, x):
        # TODO y = x @ W + b  (the bias broadcasts over the batch)
        raise NotImplementedError
    def parameters(self):
        # TODO
        raise NotImplementedError

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    def forward(self, x):
        # TODO apply the layers in order
        raise NotImplementedError
    def parameters(self):
        # TODO the concatenated parameters of all layers, in layer order
        raise NotImplementedError
# %% check
_rg = np.random.default_rng(5)
_lin = Linear(400, 300, np.random.default_rng(0))
check("Linear parameters: shapes, zero bias, requires_grad", lambda: _lin.W.shape == (400, 300) and _lin.b.shape == (300,) and np.all(_lin.b.data == 0) and _lin.W.requires_grad and _lin.b.requires_grad and len(_lin.parameters()) == 2)
check("He initialisation: std of W is sqrt(2/n_in) within 5%", lambda: abs(_lin.W.data.std() / np.sqrt(2 / 400) - 1) < 0.05)
_l2 = Linear(3, 2, np.random.default_rng(1)); _xx = _rg.normal(size=(5, 3))
check("Linear forward equals x @ W + b and returns a Tensor", lambda: isinstance(_l2(Tensor(_xx)), Tensor) and np.allclose(_l2(Tensor(_xx)).data, _xx @ _l2.W.data + _l2.b.data))
_net = Sequential(Linear(3, 6, np.random.default_rng(2)), ReLU(), Linear(6, 4, np.random.default_rng(3)), Tanh(), Linear(4, 2, np.random.default_rng(4)))
def _manual_net(net, x):
    a = x
    for L in net.layers:
        if isinstance(L, Linear): a = a @ L.W.data + L.b.data
        elif isinstance(L, ReLU): a = np.maximum(0, a)
        else: a = np.tanh(a)
    return a
check("Sequential.parameters returns 6 tensors in layer order (W, b, W, b, ...)", lambda: len(_net.parameters()) == 6 and _net.parameters()[0] is _net.layers[0].W and _net.parameters()[3] is _net.layers[2].b and _net.parameters()[4] is _net.layers[4].W)
check("Sequential forward equals the layer-by-layer NumPy computation", lambda: np.allclose(_net(Tensor(_xx)).data, _manual_net(_net, _xx)))
_yy = np.array([0, 1, 1, 0, 1])
softmax_cross_entropy(_net(Tensor(_xx)), _yy).backward()
check("after backward every parameter has a gradient of its own shape (the bias gradient is summed over the batch)", lambda: all(p.grad is not None and p.grad.shape == p.data.shape for p in _net.parameters()))
_net.zero_grad()
check("zero_grad clears all parameter gradients", lambda: all(p.grad is None for p in _net.parameters()))
check("the whole network passes a finite-difference check on its input", lambda: op_grad_error(lambda t: _net(t), _xx) < 1e-7)
# %% hint
`self.W = Tensor(rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out)), requires_grad=True)` and `self.b = Tensor(np.zeros(n_out), requires_grad=True)`. `forward`: `return x @ self.W + self.b`.
# %% hint
`Sequential.parameters`: `return [p for layer in self.layers for p in layer.parameters()]`.
# %% solution
class Linear(Module):
    def __init__(self, n_in, n_out, rng):
        self.W = Tensor(rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out)), requires_grad=True)
        self.b = Tensor(np.zeros(n_out), requires_grad=True)
    def forward(self, x):
        return x @ self.W + self.b
    def parameters(self):
        return [self.W, self.b]

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# %% task SGD (with momentum) and Adam for Tensors
class SGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params); self.lr = lr; self.mu = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]
    def step(self):
        """For each parameter with grad not None: v = mu*v - lr*grad ; p.data += v  (in place on p.data and v)."""
        # TODO
        raise NotImplementedError
    def zero_grad(self):
        for p in self.params:
            p.grad = None

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = list(params); self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    def step(self):
        """t += 1 ONCE per call; for each parameter with grad not None: the bias-corrected Adam update of dl.01,
        applied in place on p.data."""
        # TODO
        raise NotImplementedError
    def zero_grad(self):
        for p in self.params:
            p.grad = None
# %% check
def _p(vals): return Tensor(np.array(vals, dtype=float), requires_grad=True)
_w = _p([1.0, 2.0]); _o = SGD([_w], 0.1); _w.grad = np.array([1.0, -1.0]); _arr = _w.data; _o.step()
check("SGD: p <- p - lr * g, in place on p.data", lambda: np.allclose(_w.data, [0.9, 2.1]) and _w.data is _arr)
_w1 = _p([1.0]); _om = SGD([_w1], 0.1, 0.9)
_w1.grad = np.array([1.0]); _om.step(); _w1.grad = np.array([1.0]); _om.step()
check("SGD momentum after two steps: 0.71", lambda: np.allclose(_w1.data, [0.71]))
_wa, _wb = _p([1.0]), _p([1.0]); _wa.grad = np.array([1.0]); SGD([_wa, _wb], 0.5).step()
check("parameters whose grad is None are skipped (SGD)", lambda: np.allclose(_wa.data, 0.5) and np.allclose(_wb.data, 1.0))
def _adam_ref(gs, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    p = np.zeros(2); m = np.zeros(2); v = np.zeros(2)
    for t, g in enumerate(gs, 1):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return p
_gs = [np.array([5.0, -0.02]), np.array([1.0, 0.3]), np.array([-2.0, 0.1])]
_wd = _p([0.0, 0.0]); _oa = Adam([_wd], 0.01)
_wd.grad = _gs[0].copy(); _oa.step(); _first = _wd.data.copy()
for _g in _gs[1:]:
    _wd.grad = _g.copy(); _oa.step()
check("Adam: first step is about +-lr per coordinate whatever the gradient scale", lambda: np.allclose(_first, [-0.01, 0.01], atol=1e-6))
check("Adam: three steps equal an independent reference", lambda: np.allclose(_wd.data, _adam_ref(_gs), atol=1e-10))
_wc, _wd2 = _p([0.0, 0.0]), _p([0.0, 0.0]); _wc.grad = np.array([1.0, 1.0]); _wd2.grad = None
_ob = Adam([_wc, _wd2], 0.1); _ob.step()
check("Adam: skips None gradients and counts one step per call", lambda: _ob.t == 1 and np.allclose(_wd2.data, 0.0) and np.all(_wc.data < 0))
_wz = _p([1.0]); _wz.grad = np.array([2.0]); _oz = SGD([_wz], 0.1); _oz.zero_grad()
check("zero_grad sets every gradient to None", lambda: _wz.grad is None)
# %% hint
SGD: `for p, v in zip(self.params, self.v): if p.grad is None: continue; v *= self.mu; v -= self.lr * p.grad; p.data += v`.
# %% hint
Adam: after `self.t += 1` compute `c1 = 1 - b1**t`, `c2 = 1 - b2**t`; for each parameter update `m` and `v` in place, then `p.data -= lr * (m/c1) / (np.sqrt(v/c2) + eps)`.
# %% solution
class SGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params); self.lr = lr; self.mu = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]
    def step(self):
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            v *= self.mu
            v -= self.lr * p.grad
            p.data += v
    def zero_grad(self):
        for p in self.params:
            p.grad = None

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = list(params); self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    def step(self):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m *= self.b1; m += (1 - self.b1) * p.grad
            v *= self.b2; v += (1 - self.b2) * p.grad * p.grad
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
    def zero_grad(self):
        for p in self.params:
            p.grad = None
# %% md Stage 3 summary
**What you learned.** A module is just "parameters plus a forward function"; the graph is rebuilt on every forward call, so nothing about the model needs to be declared in advance (define-by-run). Optimisers read `.grad` and mutate `.data` in place, and a gradient left from an earlier step is a bug that `zero_grad` exists to prevent.

# %% report
Where in a training step must `zero_grad` be called, and why is the answer different from "after `step`"? What would happen to the loss curve if you forgot it entirely (you will test your answer in the next stage)?
# %% md Stage 4: training on moons
## Stage 4: Train an MLP on moons

Time to use the framework. Two interleaved half-circles with noise, a 2-16-16-2 ReLU network, full-batch training. Everything below is given code, so any failure here points back to one of your components.

# %% code
def make_moons(n, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.vstack([a, b]) + noise * rng.normal(size=(2 * (n // 2), 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

Xm, ym = make_moons(400, 0.15, 0)
Xm_tr, ym_tr, Xm_te, ym_te = Xm[:200], ym[:200], Xm[200:], ym[200:]

def build_model(seed):
    rng = np.random.default_rng(seed)
    return Sequential(Linear(2, 16, rng), ReLU(), Linear(16, 16, rng), ReLU(), Linear(16, 2, rng))

def accuracy_of(model, X, y):
    return float((model(Tensor(X)).data.argmax(axis=1) == y).mean())

def fit(model, opt, X, y, epochs, zero=True):
    losses = []
    for _ in range(epochs):
        if zero:
            opt.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(X)), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    return losses

def sparkline(v, width=40):
    v = np.asarray(v, dtype=float)
    idx = np.linspace(0, len(v) - 1, width).astype(int)
    s = v[idx]; q = np.clip(((s - s.min()) / (s.max() - s.min() + 1e-12) * 9).astype(int), 0, 9)
    return "".join(" .:-=+*#%@"[9 - i] for i in q)

print(f"{'optimiser':<22}{'loss first -> last':<22}{'test acc':<10} loss curve")
for _name, _mk in [("SGD lr 0.1", lambda ps: SGD(ps, 0.1)), ("SGD+momentum 0.9", lambda ps: SGD(ps, 0.02, 0.9)), ("Adam lr 0.01", lambda ps: Adam(ps, 0.01))]:
    _m = build_model(0)
    _l = fit(_m, _mk(_m.parameters()), Xm_tr, ym_tr, 150)
    print(f"{_name:<22}{_l[0]:.3f} -> {_l[-1]:.3f}     {accuracy_of(_m, Xm_te, ym_te):<10.3f} |{sparkline(_l)}|")
# %% md What happens without zero_grad
If gradients are never cleared, every backward pass adds to the previous gradients (leaf accumulation, which you implemented in `backprop`). The update direction becomes the running sum of all past gradients: a hidden momentum with no decay and a steadily growing effective step. Whether that helps or hurts depends on the problem and the learning rate (on this easy problem it may even look fine for 60 steps), but it is never what you meant to write.

# %% code
_mz = build_model(0); _lz = fit(_mz, SGD(_mz.parameters(), 0.1), Xm_tr, ym_tr, 60, zero=False)
_mo = build_model(0); _lo = fit(_mo, SGD(_mo.parameters(), 0.1), Xm_tr, ym_tr, 60, zero=True)
print("with zero_grad    : loss", round(_lo[0], 3), "->", round(_lo[-1], 3), " test acc", round(accuracy_of(_mo, Xm_te, ym_te), 3))
print("without zero_grad : loss", round(_lz[0], 3), "->", round(_lz[-1], 3), " test acc", round(accuracy_of(_mz, Xm_te, ym_te), 3))
# %% md Stage 4 summary
**What you learned.** Your engine trains a real network end to end. SGD, momentum and Adam all reduce the loss at these settings, Adam fastest on this tiny problem. Forgetting `zero_grad` does not crash anything, it silently changes the optimisation into something different (compare the two printed runs, and try a larger learning rate), which is why it is one of the classic framework bugs.

# %% report
Look at the printout: describe the difference between the three loss curves and between the runs with and without `zero_grad`. Which result surprised you?
# %% md Stage 5: matching hand-derived backprop
## Stage 5: Does the framework equal hand-derived backprop?

The final exam. In [[f2n.10]] you derived, for $Z_1 = XW_1 + b_1,\ A_1 = \mathrm{ReLU}(Z_1),\ Z_2 = A_1W_2 + b_2$ and the mean softmax cross-entropy: $\bar Z_2 = (P - Y)/B$, $\bar W_2 = A_1^\top \bar Z_2$, $\bar b_2 = \sum_b \bar Z_{2,b}$, $\bar A_1 = \bar Z_2 W_2^\top$, $\bar Z_1 = \bar A_1 \odot [Z_1 > 0]$, $\bar W_1 = X^\top \bar Z_1$, $\bar b_1 = \sum_b \bar Z_{1,b}$. The function below is that derivation in NumPy. Your framework must reproduce it to machine precision, both for one gradient and over a whole training run.

# %% code
def manual_loss_and_grads(params, X, y):
    """Hand-derived backprop from f2n.10. params = (W1, b1, W2, b2). Returns (loss, [dW1, db1, dW2, db2])."""
    W1, b1, W2, b2 = params
    B = len(y)
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z = Z2 - Z2.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(P[np.arange(B), y])))
    dZ2 = P.copy(); dZ2[np.arange(B), y] -= 1; dZ2 /= B
    dW2 = A1.T @ dZ2; db2 = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = X.T @ dZ1; db1 = dZ1.sum(axis=0)
    return loss, [dW1, db1, dW2, db2]
# %% md Training run: manual versus framework
Now two full training runs from identical weights on the moons data: plain gradient descent with the manual gradients and `p -= lr * g` on one side; the framework's `Sequential`, `softmax_cross_entropy` and `SGD` on the other. If every component is right, the two loss trajectories are the same numbers up to floating-point rounding.

# %% code
def training_comparison(steps=100, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, np.sqrt(2 / 2), (2, 16)), np.zeros(16)
    W2, b2 = rng.normal(0, np.sqrt(2 / 16), (16, 2)), np.zeros(2)
    manual = [W1.copy(), b1.copy(), W2.copy(), b2.copy()]
    model = Sequential(Linear(2, 16, np.random.default_rng(99)), ReLU(), Linear(16, 2, np.random.default_rng(99)))
    for layer_p, arr in zip(model.parameters(), [W1, b1, W2, b2]):
        layer_p.data[...] = arr
    opt = SGD(model.parameters(), lr)
    lm, lf = [], []
    for _ in range(steps):
        loss, grads = manual_loss_and_grads(manual, Xm_tr, ym_tr)
        for p, g in zip(manual, grads):
            p -= lr * g
        lm.append(loss)
        opt.zero_grad()
        loss_t = softmax_cross_entropy(model(Tensor(Xm_tr)), ym_tr)
        loss_t.backward(); opt.step()
        lf.append(float(loss_t.data))
    acc_m = float((np.maximum(0, Xm_te @ manual[0] + manual[1]) @ manual[2] + manual[3]).argmax(1).__eq__(ym_te).mean())
    return {"manual_losses": lm, "framework_losses": lf, "manual_params": manual,
            "framework_params": [p.data.copy() for p in model.parameters()],
            "manual_acc": acc_m, "framework_acc": accuracy_of(model, Xm_te, ym_te)}
# %% task Gradients through your framework
def autograd_grads(params, X, y):
    """Compute the same network's loss and gradients with YOUR Tensor ops (matmul, add via +, relu,
    softmax_cross_entropy), without Modules. params = (W1, b1, W2, b2) are NumPy arrays; wrap each in a
    Tensor(requires_grad=True), build the graph from Tensor(X), call backward, and return
    (float loss, [dW1, db1, dW2, db2]) as NumPy arrays in the same order as manual_loss_and_grads."""
    # TODO
    raise NotImplementedError
# %% check
_rg = np.random.default_rng(7)
_X = _rg.normal(size=(30, 4)); _y = _rg.integers(0, 3, size=30)
_pr = (_rg.normal(0, 0.7, (4, 8)), _rg.normal(0, 0.2, 8), _rg.normal(0, 0.7, (8, 3)), _rg.normal(0, 0.2, 3))
_lm, _gm = manual_loss_and_grads(_pr, _X, _y)
_la, _ga = autograd_grads(_pr, _X, _y)
check("returns a float loss and four arrays with the parameter shapes", lambda: isinstance(_la, float) and [g.shape for g in _ga] == [q.shape for q in _pr])
check("the loss equals the manual loss to 1e-12", lambda: abs(_la - _lm) < 1e-12)
for _nm, _gg, _hh in zip(["dW1", "db1", "dW2", "db2"], _ga, _gm):
    check(f"{_nm} equals the hand-derived gradient to 1e-10", (lambda gg=_gg, hh=_hh: np.max(np.abs(gg - hh)) < 1e-10), "Compare with the formulas in f2n.10; also check that you build the graph from Tensor(X) with requires_grad=False")
_keep = [q.copy() for q in _pr]; autograd_grads(_pr, _X, _y)
_trc = training_comparison(60, 0.2, 1)
check("a full SGD training run through Sequential/SGD/softmax_cross_entropy reproduces the manual loss trajectory to 1e-9", lambda: np.max(np.abs(np.array(_trc["manual_losses"]) - np.array(_trc["framework_losses"]))) < 1e-9, "If one gradient matches but the run does not, look at the optimiser, zero_grad, or the Module wiring")
check("and ends in the same trained parameters (1e-9) with a good test accuracy", lambda: max(np.max(np.abs(a - b)) for a, b in zip(_trc["manual_params"], _trc["framework_params"])) < 1e-9 and _trc["framework_acc"] > 0.85 and _trc["framework_acc"] == _trc["manual_acc"])
check("the input arrays are not modified and no gradient state leaks between calls", lambda: all(np.array_equal(a, b) for a, b in zip(_pr, _keep)) and np.allclose(autograd_grads(_pr, _X, _y)[1][0], _gm[0], atol=1e-10))
# %% hint
`W1, b1, W2, b2 = [Tensor(p, requires_grad=True) for p in params]`, then `logits = relu(Tensor(X) @ W1 + b1) @ W2 + b2` and `loss = softmax_cross_entropy(logits, y)`.
# %% hint
After `loss.backward()` return `float(loss.data), [W1.grad, b1.grad, W2.grad, b2.grad]`.
# %% solution
def autograd_grads(params, X, y):
    W1, b1, W2, b2 = [Tensor(p, requires_grad=True) for p in params]
    logits = relu(Tensor(X) @ W1 + b1) @ W2 + b2
    loss = softmax_cross_entropy(logits, y)
    loss.backward()
    return float(loss.data), [W1.grad.copy(), b1.grad.copy(), W2.grad.copy(), b2.grad.copy()]
# %% md The final comparison table
The table collects the quantities that must agree. The function below builds its rows from the training runs and from a gradient comparison on fixed parameters.

# %% code
def comparison_rows(result, params, X, y):
    lm = np.array(result["manual_losses"]); lf = np.array(result["framework_losses"])
    rows = [("loss at step 0", float(lm[0]), float(lf[0]), float(abs(lm[0] - lf[0]))),
            ("loss at last step", float(lm[-1]), float(lf[-1]), float(abs(lm[-1] - lf[-1])))]
    d = float(np.max(np.abs(lm - lf)))
    rows.append(("max |loss diff| over the run", 0.0, d, d))
    _, gm = manual_loss_and_grads(params, X, y)
    _, ga = autograd_grads(params, X, y)
    for name, a, b in zip(["dW1", "db1", "dW2", "db2"], ga, gm):
        e = float(np.max(np.abs(a - b)))
        rows.append((f"max |grad diff| {name}", 0.0, e, e))
    pd = float(max(np.max(np.abs(a - b)) for a, b in zip(result["manual_params"], result["framework_params"])))
    rows.append(("max |param diff| after training", 0.0, pd, pd))
    rows.append(("test accuracy", float(result["manual_acc"]), float(result["framework_acc"]), float(abs(result["manual_acc"] - result["framework_acc"]))))
    return rows
# %% code
_res = training_comparison(100, 0.2, 0)
_pp = tuple(np.random.default_rng(3).normal(0, 0.5, s) for s in [(2, 16), (16,), (16, 2), (2,)])
print(f"{'quantity':<36}{'manual':>14}{'framework':>14}{'|diff|':>12}")
for _q, _a, _b, _d in comparison_rows(_res, _pp, Xm_tr, ym_tr):
    print(f"{_q:<36}{_a:>14.6g}{_b:>14.6g}{_d:>12.2e}")
# %% md Stage 5 summary
**What you learned.** The framework is not an approximation of hand-derived backprop, it *is* the same sequence of matrix products, discovered mechanically by walking the graph in reverse. The differences are rounding errors (around $10^{-16}$ to $10^{-13}$), several orders of magnitude below any learning-rate or seed effect. What the framework adds is generality: you changed no gradient code to try a different architecture.

# %% report
In your own words: what exactly does the framework automate compared with the hand derivation in [[f2n.10]], and what is the price (think of memory held by the graph, speed, and the time cost of debugging a bad op)? Name one thing this engine cannot do that PyTorch can (for example batched matmul, in-place ops, GPU, higher-order gradients).
# %% md Stage 6: the report
## Stage 6: Write the report

# %% report
Write a one-page engineering report (about 250 words) with these headings: **What was built** (the classes and ops, in two sentences), **How it was verified** (the op gradient-check table and the manual-versus-framework table, quote the error magnitudes), **Bugs you hit** (at least one, with its symptom and cause: for example a missing unbroadcast, `=` instead of `+=`, or a recursive topological sort), **Limits** (what the engine does not support), and **Next step** (one extension you would add: a new op, higher-order gradients, or a convolution).
# %% md Sources
## Sources

Only material that was actually opened while writing this project:

- Lesson `[[f2n.10]]` (Backpropagation, derived with shapes): `content/authored/ml/10-backpropagation/lesson.md` (the network definition and backward derivation) and its exercise `exercises/mlp-backprop.exercise.txt`. READ.
- Lesson `[[f2n.12]]` (Automatic differentiation and validating against PyTorch): `content/authored/ml/12-autograd-validation/lesson.md` (computational graph, reverse mode, the scalar-engine algorithm with topological order, accumulation, pitfalls). READ.
- Lesson `[[dl.01]]` (Optimisers and learning-rate schedules): `content/authored/deep-learning/01-optimisers/lesson.md` (momentum and Adam with bias correction). READ.
- `docs/PROJECTS.md`, `docs/AGENT_BRIEF.md`, `docs/AUTHORING.md`: format and rules. READ.

All numbers printed by the project are produced by the code in it; none are quoted from a source.
