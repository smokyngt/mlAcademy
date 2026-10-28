---
id: seq.04
title: LSTM and GRU
summary: Why an additive cell-state path repairs the vanishing gradient, all gate equations with shapes for LSTM and GRU, parameter counts, a hand-computed LSTM step with its backward pass, and NumPy implementations checked against finite differences.
minutes: 100
teaches: [seq.lstm, seq.gru]
requires: [seq.rnn, seq.rnn-gradients, nn.activation, nn.backprop]
sources:
  - "math/cours-tal-02.pdf (CNAM RCP217, PDF p.21-25: the LSTM idea of a second memory path, LSTM equations f/i/o/c/h, the LSTM and vanishing gradient slide with d c_t / d c_{t-1}, GRU as a simpler alternative, stacked layers). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md (LSTM and GRU described as gated cells, the cell-state highway, bidirectional RNNs, max-pooling over states). READ."
---

## Learning objectives

- Explain, with a derivative, why a gated additive cell state lets gradients travel far.
- Write all LSTM gate equations ($i,f,g,o$) and GRU equations ($r,z$) with shapes.
- Count parameters of RNN, LSTM and GRU layers and compare them.
- Backpropagate through one LSTM step by hand and in code.
- Verify properties of the gates (a large forget bias keeps the cell, an update gate of 1 keeps the state) and gradient-check a step.

## Prerequisites

The RNN recurrence, BPTT and the Jacobian-product argument of [[seq.03]]; activation functions and the sigmoid/tanh derivatives.

## Where this fits

Recurrent networks were the workhorse of language, speech and time-series modelling from roughly 2014 to 2017 (translation, speech recognition, handwriting), and the LSTM is the reason: it made it possible to train on sequences of hundreds of steps. The idea carries over to modern architectures, where the additive "highway" appears again as the residual connection ([[seq.01]]) in every transformer block. Understanding why the LSTM works is one of the best ways to understand why residual paths work.

## The idea in plain words

Return to the sticky note of [[seq.03]]. In a plain RNN you rewrite the *whole* note at every word. After 50 words the first thing you wrote has been overwritten and blurred 50 times.

An LSTM gives you **a notebook plus a secretary of gates**. The notebook ($c_t$, the cell state) is passed along unchanged unless something tells it otherwise. Three small decisions, each a number between 0 and 1 per memory slot, control it:

- **Forget gate**: how much of each old entry do I keep? (1: keep it all, 0: erase.)
- **Input gate**: how much of the new candidate content do I write?
- **Output gate**: how much of the notebook do I show to the rest of the network right now?

The key is that the update is **additive**: new notebook = (kept old notebook) + (newly written content). Because it is a sum, not a repeated squashing, both information and gradient can ride along the notebook for a long time when the forget gate stays near 1.

The **GRU** is a leaner version: no separate notebook, a single state, and two gates. An *update gate* decides how much of the old state to keep versus replace; a *reset gate* decides how much of the old state the proposed new content may look at.

## Formal definitions

Input $x_t\in\mathbb{R}^n$, hidden size $H$, previous $h_{t-1},c_{t-1}\in\mathbb{R}^H$. $\sigma$ is the logistic sigmoid and $\odot$ is the elementwise product.

### LSTM

$$
\begin{aligned}
&i_t=\sigma(W_ix_t+U_ih_{t-1}+b_i) &&\text{input gate}\\
&f_t=\sigma(W_fx_t+U_fh_{t-1}+b_f) &&\text{forget gate}\\
&g_t=\tanh(W_gx_t+U_gh_{t-1}+b_g) &&\text{candidate}\\
&o_t=\sigma(W_ox_t+U_oh_{t-1}+b_o) &&\text{output gate}
\end{aligned}
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).
$$

(The CNAM slides call the candidate $\tanh(W_cx_t+U_ch_{t-1}+b_c)$ and note that $f_t$ forgets selectively while $i_t$ lets new information in.) Shapes: each $W_\bullet:(H,n)$, each $U_\bullet:(H,H)$, each $b_\bullet:(H)$, all gates and states $(H)$. In code the four blocks are stacked: $W:(4H,n)$, $U:(4H,H)$, $b:(4H)$, a single matrix product gives $a=Wx+Uh_{t-1}+b\in\mathbb{R}^{4H}$ which is split into four. We use the order $i,f,g,o$ (PyTorch uses the same order).

### GRU

$$
\begin{aligned}
&r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r) &&\text{reset gate}\\
&z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z) &&\text{update gate}\\
&\tilde h_t=\tanh\big(W_nx_t+r_t\odot(U_nh_{t-1})+b_n\big) &&\text{candidate}\\
&h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}
\end{aligned}
$$

Stacked: $W:(3H,n)$, $U:(3H,H)$, $b:(3H)$ in the order $r,z,n$. Conventions differ between papers (which side gets $z$ versus $1-z$, whether $r$ multiplies $h_{t-1}$ before or after the matrix product). The version above is the one used by PyTorch's `GRUCell` (up to its second bias vector); the numerical check against PyTorch below confirms it.

### Parameter counts

Each gate (or candidate) block has $Hn+H^2+H$ parameters.

| Cell | Blocks | Formula | $n=128$, $H=256$ |
|---|---|---|---|
| RNN | 1 | $H(n+H+1)$ | 98,560 |
| GRU | 3 | $3H(n+H+1)$ | 295,680 |
| LSTM | 4 | $4H(n+H+1)$ | 394,240 |

So a GRU has 3/4 of the parameters of an LSTM of the same size, and the LSTM has 4 times those of the plain RNN. (PyTorch keeps two bias vectors per block, so its `LSTMCell(128,256)` reports 395,264 and `GRUCell` 296,448: $+4H$ and $+3H$.) The CNAM slides add that GRUs are faster and train faster, "maybe less efficient in some cases (the jury seems to be still out)".

## Derivation: why gating helps gradients

In a plain RNN the path from $h_{t-1}$ to $h_t$ goes through a matrix and a squashing function: $\partial h_t/\partial h_{t-1}=\operatorname{diag}(1-h_t^2)W_{hh}$, and the product over $T$ steps behaves like $\rho(W_{hh})^T$ ([[seq.03]]). Every step *must* pass through $W_{hh}$ and the nonlinearity.

Look at the cell path of the LSTM instead. $c_t=f_t\odot c_{t-1}+i_t\odot g_t$. The gates and the candidate depend on $c_{t-1}$ only indirectly through $h_{t-1}=o_{t-1}\odot\tanh(c_{t-1})$. Splitting the derivative into the **direct** and **indirect** parts (this is the sum structure shown in the slides, p.23):

$$
\frac{\partial c_t}{\partial c_{t-1}}=\underbrace{\operatorname{diag}(f_t)}_{\text{direct, elementwise}}+\underbrace{\operatorname{diag}(c_{t-1})\frac{\partial f_t}{\partial c_{t-1}}+\operatorname{diag}(i_t)\frac{\partial g_t}{\partial c_{t-1}}+\operatorname{diag}(g_t)\frac{\partial i_t}{\partial c_{t-1}}}_{\text{through }h_{t-1}} .
$$

The direct term contains **no weight matrix and no squashing derivative**: it is just the forget gate. If we follow only that term over $T$ steps we get

$$
\frac{\partial c_T}{\partial c_0}\;\approx\;\prod_{t=1}^{T}\operatorname{diag}(f_t)\quad\Longrightarrow\quad \text{component } j:\ \prod_tf_{t,j}.
$$

Each component is a product of numbers the network itself controls and can push close to 1. Compare:

| per-step factor | after $T=100$ steps |
|---|---|
| $0.9$ | $2.7\times10^{-5}$ |
| $0.99$ | $0.37$ |
| $\sigma(5)=0.9933$ | $0.51$ |

If $f_t\approx1$ and the input gate is $\approx0$, the cell is copied unchanged and the gradient flows through it unchanged. The gradient that reaches $c_0$ is not forced to pass through a squashing function and a repeated matrix. This is exactly the role of the **forget-gate bias**: initialising $b_f$ to a positive value (such as 1 to 5) starts training with $f\approx$ large, so memory is kept by default. (The indirect terms still exist and can matter; this is an argument for a favourable path, not a guarantee. And note that $\sigma(1)^{100}\approx2\times10^{-14}$: a bias of 1 alone does not preserve information over 100 steps unless training pushes $f$ higher.)

Same argument for the GRU. From $h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}$:

$$
\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(z_t)+(\text{terms through the gates and }\tilde h_t).
$$

An update gate near 1 copies the previous state. It is the same additive-identity idea as the residual block $y=x+F(x)$ of [[seq.01]], with a learnable, input-dependent coefficient.

## Derivation: backward through one LSTM step

Given $\bar h_t$ and $\bar c_t$ (gradients arriving from the future and from the output at time $t$), compute everything with the chain rule. Let $\tau=\tanh(c_t)$.

1. Through $h_t=o_t\odot\tau$: $\ \bar o=\bar h\odot\tau$ and the part of $\bar c$ coming from $h_t$ is $\bar h\odot o\odot(1-\tau^2)$. Total gradient on the cell:

$$
\bar c^{\,tot}=\bar c_t+\bar h_t\odot o_t\odot(1-\tau^2).
$$

2. Through $c_t=f\odot c_{t-1}+i\odot g$:

$$
\bar f=\bar c^{\,tot}\odot c_{t-1},\quad \bar i=\bar c^{\,tot}\odot g,\quad \bar g=\bar c^{\,tot}\odot i,\quad \bar c_{t-1}=\bar c^{\,tot}\odot f .
$$

   The last one is the additive path: the gradient to the previous cell is the incoming gradient times $f$, with nothing else in the way.

3. Through the gate nonlinearities (sigmoid derivative $s(1-s)$, tanh derivative $1-g^2$):

$$
\bar a_i=\bar i\odot i\odot(1-i),\ \bar a_f=\bar f\odot f\odot(1-f),\ \bar a_g=\bar g\odot(1-g^2),\ \bar a_o=\bar o\odot o\odot(1-o).
$$

4. Stack $\bar a=[\bar a_i;\bar a_f;\bar a_g;\bar a_o]\in\mathbb{R}^{4H}$. Since $a=Wx+Uh_{t-1}+b$:

$$
\bar W=\bar a\,x^\top\ (4H\times n),\quad \bar U=\bar a\,h_{t-1}^\top\ (4H\times H),\quad \bar b=\bar a,\quad \bar x=W^\top\bar a,\quad \bar h_{t-1}=U^\top\bar a .
$$

   Across time steps $\bar W,\bar U,\bar b$ accumulate (shared weights, as in [[seq.03]]), and $\bar h_{t-1}$, $\bar c_{t-1}$ are the two quantities passed back.

## Worked example (one LSTM step, $n=H=1$)

$W=(0.5,\,1.0,\,0.8,\,0.3)^\top$ for $(i,f,g,o)$, $U=(0.1,\,0.2,\,0.4,\,0.5)^\top$, $b=(0,\,1,\,0,\,0)^\top$; $x=1$, $h_{t-1}=0.5$, $c_{t-1}=1$.

- Pre-activations $a=Wx+Uh_{t-1}+b=(0.55,\ 2.10,\ 1.00,\ 0.55)$.
- Gates: $i=\sigma(0.55)=0.6341$, $f=\sigma(2.1)=0.8909$, $g=\tanh(1)=0.7616$, $o=\sigma(0.55)=0.6341$.
- Cell: $c_t=0.8909\cdot1+0.6341\cdot0.7616=1.3739$; $\tanh(c_t)=0.8796$; $h_t=0.6341\cdot0.8796=0.5578$.

Backward with $\bar h_t=1$, $\bar c_t=0$:

- $\bar c^{\,tot}=1\cdot0.6341\cdot(1-0.8796^2)=0.14354$
- $\bar c_{t-1}=\bar c^{\,tot}f=0.14354\cdot0.8909=0.12788$
- $\bar o=\tau=0.8796$, so $\bar a_o=0.8796\cdot0.6341\cdot0.3659=0.2041$
- $\bar i=\bar c^{\,tot}g=0.1093$, so $\bar a_i=0.1093\cdot0.6341\cdot0.3659=0.02536$

The exercise code returns $\bar c_{t-1}=0.12788$, $\bar b=(0.02536,\,0.01395,\,0.03823,\,0.20407)$, $\bar x=0.11844$, $\bar h_{t-1}=0.12265$.

## Implementation

```python
import numpy as np

def sigmoid(s):
    return 0.5 * (1.0 + np.tanh(0.5 * s))

def lstm_step(x, h_prev, c_prev, W, U, b):          # W:(4H,n) U:(4H,H) b:(4H)
    H = h_prev.shape[0]
    a = W @ x + U @ h_prev + b
    i, f = sigmoid(a[:H]), sigmoid(a[H:2*H])
    g, o = np.tanh(a[2*H:3*H]), sigmoid(a[3*H:])
    c = f * c_prev + i * g
    tc = np.tanh(c)
    return o * tc, c, (x, h_prev, c_prev, W, U, i, f, g, o, tc)

def gru_step(x, h_prev, W, U, b):                   # W:(3H,n) U:(3H,H) b:(3H)
    H = h_prev.shape[0]
    r = sigmoid(W[:H] @ x + U[:H] @ h_prev + b[:H])
    z = sigmoid(W[H:2*H] @ x + U[H:2*H] @ h_prev + b[H:2*H])
    n_ = np.tanh(W[2*H:] @ x + r * (U[2*H:] @ h_prev) + b[2*H:])
    return (1 - z) * n_ + z * h_prev
```

The backward step of the first exercise is the five lines of the derivation above.

### Verification

- **Finite differences**: with the scalar loss $A^\top h_t+B^\top c_t$ for random $A,B$, every gradient ($\bar x,\bar h_{t-1},\bar c_{t-1},\bar W,\bar U,\bar b$) matches central differences to about $10^{-7}$ in the exercise tests.
- **Against PyTorch** (a scratch check, not part of the exercises): copying $W,U,b$ into `torch.nn.LSTMCell` (second bias set to 0) reproduces $h_t,c_t$ and all gradients to $\sim10^{-8}$ (float32 weight rounding), and `torch.nn.GRUCell` reproduces our $h_t$ to machine precision. This confirms the gate order ($i,f,g,o$ and $r,z,n$) and that `GRUCell`'s convention is $h_t=(1-z)\odot\tilde h+z\odot h_{t-1}$.
- **Behavioural tests** (second and first exercise): with zero weights and $b_f=+20$, $b_i=-20$ the cell is unchanged and $\bar c_{t-1}=\bar c_t$; with $b_f=b_i=-20$ the cell is erased; for the GRU, $b_z=+30$ gives $h_t=h_{t-1}$ and $b_z=-30$ gives $h_t=\tilde h$, and with $b_r=b_z=-30$ the output does not depend on $h_{t-1}$ at all.

## Common misconceptions

- **"LSTMs solve vanishing gradients completely."** They make them much milder through the additive path; very long sequences (thousands of steps) are still hard, and exploding gradients still need clipping.
- **"The cell state is the output."** The output is $h_t=o_t\odot\tanh(c_t)$; the cell is internal memory.
- **"More gates means more power."** GRUs with 3/4 of the parameters often perform similarly; which is better is task-dependent.
- **"The gates are decided by a rule."** They are learned functions of $x_t$ and $h_{t-1}$, trained by backprop like everything else.
- **"Gating removes the sequential bottleneck."** Computation is still sequential in $t$; only the gradient path is improved.

## Limits and trade-offs

- Cost: 4x (LSTM) or 3x (GRU) the parameters and per-step compute of a plain RNN with the same $H$.
- Memory for BPTT stores all gates and states per step: about $7H$ values per step per sample ($i,f,g,o,c,h,\tanh c$).
- Still strictly sequential, so slower to train than convolutions or attention on modern hardware.
- A single fixed-size state (and $c$) must summarise the past: for translation this becomes a bottleneck ([[seq.05]]).

## Key takeaways

- LSTM: $c_t=f_t\odot c_{t-1}+i_t\odot g_t$, $h_t=o_t\odot\tanh(c_t)$ with four blocks $i,f,g,o$ of size $H$; stacked $W:(4H,n)$, $U:(4H,H)$.
- Cell-path derivative contains the direct term $\operatorname{diag}(f_t)$ with no weight matrix and no squashing: gradients can survive when $f\approx1$ (so initialise the forget bias positive).
- GRU: $h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}$, reset gate $r_t$ inside the candidate; 3 blocks, 3/4 of an LSTM's parameters.
- LSTM backward: $\bar c^{tot}=\bar c+\bar h\odot o\odot(1-\tanh^2c)$, $\bar c_{t-1}=\bar c^{tot}\odot f$, then gate nonlinearities and one stacked matrix backward.

## Explain it yourself

1. Why does $\bar c_{t-1}=\bar c^{tot}\odot f_t$ give a gradient path that a plain RNN lacks?
2. Which parameter would you initialise to a large positive value to make an LSTM remember by default, and what do you expect the cell to do on the first training steps?

## Further reading

- CNAM RCP217 slides `cours-tal-02.pdf`, PDF p.21-25.
- Source library: `ai-engineering-from-scratch` phase 05 lesson 08.
- Next: [[seq.05]], where an encoder and decoder of such cells meet the bottleneck that motivated attention.

## Mastery criteria

You can write the LSTM and GRU equations with all shapes from memory, derive the five backward formulas of an LSTM step, compute the parameter counts, and your NumPy cells pass a gradient check and the gate-property tests.
