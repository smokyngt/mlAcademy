---
id: dl.04
title: PyTorch in practice
summary: Tensors and dtypes, autograd, nn.Module, DataLoader, the standard training loop, devices, mixed precision and checkpointing, each mapped to the NumPy network you wrote by hand.
minutes: 80
teaches: [dl.pytorch]
requires: [nn.training, nn.autograd, nn.backprop]
sources:
  - "ai-engineering-from-scratch/phases/03-deep-learning-core/11-intro-to-pytorch/docs/en.md (tensors, dtype table, devices, autograd rules, nn.Module, losses and optimisers, the five-line loop, Dataset/DataLoader, GPU and mixed precision, common bugs). READ."
  - "LLMs-from-scratch/appendix-A/01_main-chapter-code/code-part1.ipynb (A.2 tensors and from_numpy memory sharing, A.3-A.4 computation graph and autograd example, A.5 NeuralNetwork module and parameter count, A.6 Dataset/DataLoader, A.7 training loop and compute_accuracy, A.8 save/load state_dict). READ."
  - "LLMs-from-scratch/appendix-A/01_main-chapter-code/code-part2.ipynb (A.9 moving tensors to a device and the device-aware compute_accuracy). READ. DDP-script.py (multi-GPU) was NOT opened."
  - "LLMs-from-scratch/appendix-D/01_main-chapter-code/appendix-D.ipynb (D.1 warmup, D.2 cosine decay, D.3 gradient clipping with clip_grad_norm_, D.4 the extended training function). READ."
---

## Where this fits

Across [[f2n.10]] and [[f2n.11]] you built a complete network by hand: forward pass, backward pass derived on paper, an optimiser, a training loop. That was the point of those lessons, and it is why PyTorch will not feel like magic. But nobody trains a real model that way. A framework gives you three things you could not reasonably write yourself: automatic differentiation for any composition of operations, fast kernels on CPU and GPU, and an ecosystem of tested layers, losses, optimisers and data tools. This lesson walks through the pieces in the order you meet them, and for each one shows the from-scratch version it replaces. The exercise finishes the loop by proving numerically that PyTorch computes the same loss and the same gradients as your NumPy network.

## The idea in plain words

Think of the from-scratch network as a hand-built car and PyTorch as the same car built in a factory.

- A **tensor** is a NumPy array that remembers three extra facts: what number format it uses (dtype), where it lives (CPU or a particular GPU), and whether anyone wants its derivative.
- **Autograd** is a tape recorder. While the forward pass runs, every operation is written onto the tape. Calling `loss.backward()` plays the tape backwards and applies the chain rule at each step: exactly the backward pass you derived, but generated instead of hand-written.
- An **nn.Module** is a labelled box of parameters plus a `forward` function. It registers its parameters so that the optimiser can find them without you keeping a list.
- A **DataLoader** is the person who shuffles the dataset, cuts it into mini-batches and hands them to you.
- The **training loop** is the same five lines you already wrote: clear old gradients, forward, loss, backward, update.
- **Devices and precision** are about speed and memory: put the tensors where the fast hardware is, and use fewer bits per number when you can afford to.
- A **checkpoint** is a save file, so that training can survive a crash and the model can be used later.

## Map from the NumPy version to PyTorch

| From scratch ([[f2n.10]], [[f2n.11]], [[dl.01]]) | PyTorch |
|---|---|
| `X @ W + b` with `W` of shape `(n_in, n_out)` | `nn.Linear(n_in, n_out)`, weight stored as `(n_out, n_in)` |
| `np.maximum(0, Z)` | `nn.ReLU()` or `torch.relu` |
| hand-written softmax + cross-entropy, backward `(P - Y)/B` | `F.cross_entropy(logits, labels)` on raw logits and integer labels, mean over the batch |
| your `forward_backward` | forward under autograd, then `loss.backward()` |
| list of arrays `params` | `model.parameters()`, `model.state_dict()` |
| `sgd_step`, `Adam` class | `torch.optim.SGD`, `Adam`, `AdamW` |
| manual shuffle and slicing | `Dataset` + `DataLoader(shuffle=True, batch_size=...)` |
| He initialisation by hand | `nn.init.kaiming_normal_(w, nonlinearity="relu")` |
| gradient check | `torch.autograd.gradcheck` in float64, or compare to your NumPy gradients as in the exercise |
| warmup/cosine function ([[dl.01]]) | set `param_group["lr"]` each step, or a `torch.optim.lr_scheduler` |

## Tensors and dtypes

A tensor has a shape, a dtype and a device. A scalar has shape $()$, a vector $(n)$, a matrix $(m,n)$, an image batch $(B,C,H,W)$. Key operations: `reshape`/`view` (change the shape without copying data; `view` requires contiguous memory), `.T`/`permute` (reorder axes), `@` (matrix product), `unsqueeze`/`squeeze`.

Two memory facts worth knowing from the book's appendix: `torch.tensor(array)` **copies** a NumPy array, whereas `torch.from_numpy(array)` **shares** memory (changing the NumPy array changes the tensor). Integer lists give integer tensors; mixing integer and floating tensors in a loss is a classic source of errors (labels for cross-entropy must be `long`).

The dtype decides precision and range. Values measured with `torch.finfo`:

| dtype | bits | machine epsilon | largest value | smallest normal |
|---|---|---|---|---|
| float32 | 32 | $1.19\times10^{-7}$ | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |
| float16 | 16 | $9.77\times10^{-4}$ | $65504$ | $6.1\times10^{-5}$ |
| bfloat16 | 16 | $7.8\times10^{-3}$ | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |

Consequences you can verify yourself: $70000$ in float16 becomes `inf`; $10^{-8}$ in float16 becomes $0$ (it underflows below the smallest representable subnormal); bfloat16 has float32's range but only about 2 to 3 significant decimal digits, so $1+10^{-3}$ rounds to $1$.

## Autograd

Every tensor created with `requires_grad=True` is a leaf of a graph that is recorded during the forward pass. `loss.backward()` traverses it in reverse and **adds** the gradients into each leaf's `.grad`. Three rules follow:

1. Only leaves with `requires_grad=True` receive `.grad`.
2. Gradients accumulate. Three calls of `(2*p).sum().backward()` on a parameter $p$ leave `p.grad` equal to 6 instead of 2 (measured). That is why the loop clears gradients every step; accumulation is also the trick behind gradient accumulation for large effective batches.
3. `torch.no_grad()` switches recording off; use it for evaluation and for anything you do not want differentiated.

The book's tiny example: $x=1.1$, $w=2.2$, $b=0$, target $y=1$, $a=\sigma(wx+b)$ and binary cross-entropy. PyTorch gives $a=0.9183$, loss $0.08519$, $\partial L/\partial w=-0.08983$, $\partial L/\partial b=-0.08166$. These agree with the hand derivation from [[f2n.10]]: for sigmoid plus BCE, $\partial L/\partial z=a-y=-0.08166$, then $\partial L/\partial w=x\,(a-y)=1.1\times(-0.08166)=-0.08983$.

## nn.Module

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 30), nn.ReLU(),
            nn.Linear(30, 20), nn.ReLU(),
            nn.Linear(20, n_out),
        )
    def forward(self, x):          # x: (B, n_in) -> logits (B, n_out)
        return self.layers(x)
```

Assigning a layer as an attribute registers its parameters, so `model.parameters()` finds them recursively. For `NeuralNetwork(50, 3)` the parameter count is $50\cdot30+30+30\cdot20+20+20\cdot3+3=2213$ (the notebook's code `sum(p.numel() ...)` gives this number; the formula is weights plus biases per layer, as for the NumPy network). The module also carries a mode: `model.train()` and `model.eval()` switch layers such as dropout ([[dl.02]]) and batch norm ([[dl.03]]) between their training and inference behaviours. Models output **logits**; the softmax lives inside `F.cross_entropy` (passing probabilities there silently gives wrong gradients).

## Dataset and DataLoader

A `Dataset` implements `__len__` and `__getitem__`; a `DataLoader` adds batching, shuffling and optional parallel workers. Options from the book's example: `shuffle=True` for training only, `drop_last=True` so a short final batch does not distort batch statistics, `num_workers` to load data in background processes (0 means in the main process). With 5 training points and batch size 2, the loader yields 3 batches (the last of size 1) unless `drop_last` is set, in which case it yields 2.

## The standard training loop

```python
model = NeuralNetwork(2, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:            # xb: (B, 2), yb: (B,) long
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
```

Line by line against the NumPy loop of [[f2n.11]]: `model(xb)` is the forward pass, `F.cross_entropy` the loss, `backward()` replaces your hand-derived `forward_backward`, `opt.step()` is `sgd_step`, and `zero_grad()` is the reset that your functional code never needed because it returned fresh gradients every call. Typical bugs (from the source lesson's table): forgetting `zero_grad`, forgetting `eval()` and `no_grad` at test time, CPU/GPU mismatch, wrong label dtype, keeping the graph alive by storing losses without `.item()` or `.detach()`, and in-place operations that autograd cannot differentiate.

Extras from appendix D, which belong in any serious loop: a learning-rate schedule updated every step ([[dl.01]]: linear warmup then cosine decay) and **gradient clipping**, `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`, called after `backward()` and before `step()`. We verified the book's arithmetic: for a gradient with entries $(1,2;2,4)$ the norm is $\sqrt{25}=5$, and clipping to 1 multiplies it by $1/5$, giving $(0.2,0.4;0.4,0.8)$.

## Devices and GPU placement

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                   # moves all parameters and buffers
xb, yb = xb.to(device), yb.to(device)      # move every batch
```

Everything that meets in one operation must live on the same device; the error "Expected all tensors to be on the same device" is the most common beginner failure. The model moves once, the data moves every batch. The book's `compute_accuracy(model, loader, device)` is exactly the CPU function with two extra `.to(device)` calls. Copying tensors between CPU and GPU is slow, so keep the whole loop on the device and bring back only scalars (with `.item()`). Multi-GPU training with DistributedDataParallel is covered by the book's `DDP-script.py`; we did not read it and do not cover it here.

On this machine `torch.cuda.is_available()` is false, and the exercise runs on CPU.

## Mixed precision

Running forward and backward in 16 bits halves memory traffic and, on recent GPUs, increases throughput, while the parameters and optimiser state stay in float32 (the "master weights"). Two flavours, directly from the dtype table:

- **float16** has a narrow range (largest $65504$, smallest normal $6\times10^{-5}$). Small gradients underflow to zero, so one multiplies the loss by a large factor before `backward` and divides afterwards: `GradScaler` automates this (the source lesson shows `autocast` plus `scaler.scale(loss).backward(); scaler.step(opt); scaler.update()`).
- **bfloat16** has float32's range but fewer mantissa bits, so it needs no loss scaling; this is the common choice for large language models.

Under `torch.autocast` matrix multiplications run in the low precision while numerically sensitive operations (softmax, reductions, losses) are kept in float32 by PyTorch. A quick CPU check with bfloat16 on the trained moons model: the output tensor has dtype bfloat16 and the logits differ from float32 by at most $0.085$ in absolute value. That is a real accuracy cost per number, acceptable for training, which is robust to noise, and the reason inference with low precision is evaluated carefully.

## Checkpointing

`state_dict()` is an ordered dictionary from parameter names to tensors (for the moons model the keys are `0.weight, 0.bias, 2.weight, 2.bias`; the indices are the layer positions in the `Sequential`). Saving and loading:

```python
torch.save(model.state_dict(), "model.pth")
model = NeuralNetwork(2, 2)                     # architecture must match exactly
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

`weights_only=True` restricts unpickling to tensors, which matters because a pickle file can execute code. To **resume** training you must also save the optimiser state (Adam's $m$ and $v$ and step count from [[dl.01]]), the epoch and the scheduler: `torch.save({"model":..., "opt":opt.state_dict(), "epoch":e}, path)`. We measured that a model reloaded from an in-memory checkpoint produces bit-identical outputs.

## Worked example

The exercise reuses the half-moons network and checks the claim at the heart of this lesson: PyTorch computes the same thing as your NumPy code.

- Copy the NumPy parameters (seed 0) into a float64 `nn.Sequential(Linear(2,16), ReLU, Linear(16,2))`, transposing the weights.
- Loss: $0.60329\ldots$ in both; the difference is exactly $0.0$ in float64.
- Gradients: maximum absolute difference over all four tensors is $1.2\times10^{-16}$.
- Training with the standard loop, Kaiming initialisation, Adam ($\eta=0.03$), batch size 32, 60 epochs on the 200 points: training accuracy 1.0, loss $0.0128$, in about 0.8 seconds on CPU.

If you had copied the weights into a float32 model first and converted afterwards, the agreement would only be about $10^{-9}$ (we saw $3\times10^{-9}$): precision is lost at the copy. Gradient checks belong in float64.

## Common misconceptions

- **"PyTorch computes something different from my NumPy code."** The same maths; the exercise shows agreement to $10^{-16}$ in float64.
- **"`nn.Linear` weights have shape $(n_{in},n_{out})$."** They are $(n_{out},n_{in})$; the layer computes $xW^\top+b$.
- **"Only dropout needs `eval()`."** Batch norm uses running statistics in eval mode.
- **"`no_grad` and `eval()` are the same."** One stops recording the graph, the other changes layer behaviour; evaluation needs both.
- **"Mixed precision is free."** It trades precision for speed and can overflow or underflow (float16), which is why loss scaling or bfloat16 exist.
- **"Saving the model means saving training."** `state_dict` holds only parameters (and buffers); resume needs the optimiser and scheduler too.

## Key takeaways

- Tensors carry shape, dtype and device; `from_numpy` shares memory, `tensor` copies.
- Autograd records the forward pass and replays it backwards; gradients accumulate, so call `zero_grad` every step.
- Module, loss, optimiser, DataLoader and the five-line loop are one-to-one replacements for the parts you wrote by hand.
- Move model once and data every batch to the same device; use `eval()` plus `no_grad()` for inference.
- float16 needs loss scaling, bfloat16 does not; keep master weights in float32.
- Checkpoint `state_dict` with `weights_only=True` on load, and save optimiser state to resume.

## Explain it yourself

1. Walk through one iteration of the PyTorch loop and name the NumPy function from [[f2n.11]] that each line replaces.
2. Why does float16 training need a loss scale but bfloat16 training does not? Use the dtype table.

## Further reading

- *Build a Large Language Model From Scratch*, appendices A and D.
- Source library: phase 03 lesson 11 (the plain-PyTorch MNIST build is described there; we did not run it here).

## Mastery criteria

You can port a NumPy network to `nn.Module`, write the training and evaluation loops with the correct mode and gradient handling, move model and data to a device, explain float16 versus bfloat16, save and restore a checkpoint including optimiser state, and verify the port by comparing gradients with your from-scratch implementation.
