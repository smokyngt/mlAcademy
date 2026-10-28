import numpy as np
def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)
def nll(T, z, y):
    p = softmax(z / T); return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
def golden(f, lo, hi, iters=60):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi; c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if f(c) < f(d): b = d
        else: a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return (a + b) / 2
rng = np.random.default_rng(0)
N, K = 2000, 3
tl = rng.normal(size=(N, K)) * 1.5
p = softmax(tl)
y = np.array([rng.choice(K, p=pi) for pi in p])
z = tl * 2.5
print(golden(lambda t: nll(t, z, y), 0.05, 10), nll(1, z, y))
# faster y generation
rng = np.random.default_rng(0)
tl = rng.normal(size=(N, K)) * 1.5
p = softmax(tl); u = rng.random(N)
y = (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, K-1)
z = tl * 2.5
Tst = golden(lambda t: nll(t, z, y), 0.05, 10)
print("T",Tst, nll(1,z,y), nll(Tst,z,y), "calibrated-model T", golden(lambda t: nll(t, tl, y), 0.05, 10))
print("grid", min(np.linspace(0.5,6,1101), key=lambda t: nll(t,z,y)))
def ece(P, y, bins=10):
    conf=P.max(1); acc=(P.argmax(1)==y); e=0; ed=np.linspace(0,1,bins+1)
    for lo,hi in zip(ed[:-1],ed[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): e+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    return e
print("ece",ece(softmax(z),y),ece(softmax(z/Tst),y),(softmax(z).argmax(1)==y).mean(), softmax(z).max(1).mean(), softmax(z/Tst).max(1).mean())
