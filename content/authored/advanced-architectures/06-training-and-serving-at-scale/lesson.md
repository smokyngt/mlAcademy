---
id: adv.06
title: "Training at scale: parallelism, ZeRO, mixed precision, checkpointing, MTP"
summary: Split a model and its batch across GPUs, count per-GPU bytes under ZeRO/FSDP, use fp16/bf16 with loss scaling, trade recomputation for activation memory, and add multi-token prediction.
minutes: 85
teaches: [adv.parallelism, adv.zero, adv.mixed-precision, adv.checkpointing, adv.mtp]
requires: [nn.training, nn.backprop, dl.adam, trf.block, info.entropy, adv.moe-balance]
sources:
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/05-scaling-distributed/docs/en.md: memory table, data/tensor/pipeline/3D parallelism, bubble fraction, ZeRO stages, mixed precision and loss scaling. READ (most sections; code skimmed)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/19-dualpipe-parallelism/docs/en.md: pipeline bubbles, 1F1B, zero bubble, bidirectional DualPipe overlap with all-to-all. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/34-gradient-checkpointing/docs/en.md: what backward needs, full vs selective checkpointing, recompute cost model, offload. READ (first part)."
  - "ai-engineering-from-scratch/phases/10-llms-from-scratch/18-multi-token-prediction/docs/en.md: sequential MTP modules, joint loss, parallel vs sequential, parameter overhead, speculative-decoding reuse. READ (first part)."
---

## Where this fits

Everything so far (long contexts, MoE, many experts) strains memory and communication. A 7-billion-parameter model already needs more than one GPU to **train** (not just to run), because the optimizer keeps several copies of every parameter. This lesson is the toolbox for fitting and speeding up training: three ways to split work (data, tensor, pipeline parallelism), how ZeRO/FSDP shard the redundant state, why low-precision arithmetic needs loss scaling, how gradient checkpointing trades computation for memory, and how multi-token prediction gives denser supervision. It builds on the optimizer of the training loop and Adam, and on the transformer block ([[trf.04]]). Inference-side memory (KV cache, quantisation) is covered in [[sys.01]] and [[sys.02]].

## The idea in plain words

Imagine writing a huge encyclopedia with a team.

- **Data parallelism**: every editor has a complete copy of the draft, each edits different new pages, and at the end of each day they average their changes so all copies stay identical. Fast, but everyone needs a full copy.
- **Tensor parallelism**: one very wide table is cut into column strips, each editor fills their strip, and strips are glued together. Needs constant chatting, so editors must sit in the same room (same node).
- **Pipeline parallelism**: an assembly line: editor 1 handles chapters 1-8, editor 2 chapters 9-16, and so on. At the start and end of each day some editors idle (the **bubble**), unless you feed the line with many small batches.
- **ZeRO / FSDP**: editors still work in the data-parallel style, but the bulky bookkeeping (the optimizer's notes) is split so each editor stores only a slice, and slices are borrowed when needed.
- **Low precision**: writing numbers with fewer digits is faster and smaller, but very small numbers vanish; multiply everything by a big constant before writing, and divide afterwards.
- **Checkpointing**: instead of keeping every intermediate note from the forward pass, keep a few bookmarks and re-derive the rest when you need them.

## Memory model: where do the bytes go?

With mixed-precision Adam, each parameter costs

| Item | Bytes per parameter |
|---|---|
| parameters (fp16/bf16) | 2 |
| gradients (fp16/bf16) | 2 |
| fp32 master copy of weights | 4 |
| Adam first moment $m$ (fp32) | 4 |
| Adam second moment $v$ (fp32) | 4 |
| **total model state** | **16** |

So $16\psi$ bytes for $\psi$ parameters, before any activation. A 7.5 B model needs $16\cdot7.5\times10^9/2^{30}=111.8$ GiB, more than an 80 GB GPU. (The source lesson's table counts 12 bytes per parameter: 2 for weights, 2 for gradients and 8 for Adam moments, leaving out the fp32 master copy; the 16-byte accounting is the common one for mixed-precision training with master weights, which I quote from general knowledge of the ZeRO paper rather than from the source.) Data parallelism replicates all 16 bytes on every GPU, so adding GPUs buys speed, not capacity.

**ZeRO / FSDP** removes the redundancy by sharding across $N$ GPUs:

| Stage | Sharded | Per-GPU bytes per parameter |
|---|---|---|
| 0 (plain DP) | nothing | $2+2+12=16$ |
| 1 | optimizer state | $2+2+12/N$ |
| 2 | + gradients | $2+(2+12)/N$ |
| 3 (= FSDP) | + parameters | $16/N$ |

For $\psi=7.5$ B and $N=8$: 111.76, 38.42, 26.19 and 13.97 GiB. Stage 3 must **all-gather** each layer's parameters just before it is used (and free them afterwards), costing extra communication that overlaps with compute. ZeRO-Offload / Infinity move optimizer state to CPU RAM or NVMe, trading speed for capacity. For 70 B on 64 GPUs: stage 0 is 1043 GiB per GPU (impossible), stage 3 is 16.3 GiB; with an 80 GiB budget stage 3 needs 14 GPUs ($1043.1/80=13.04$) even before counting activations.

## Parallelism strategies

**Data parallel (DP).** Split the batch into $N$ shards; each GPU computes gradients on its shard; an **all-reduce** averages them; every replica applies the same update. Effective batch $=$ per-GPU batch $\times N$.

**Tensor parallel (TP).** Split a weight matrix. *Column-parallel*: $W=[W_1\,|\,W_2]$, each GPU computes $xW_i$, results concatenated: $x[W_1|W_2]=[xW_1|xW_2]$. *Row-parallel*: $x=[x_1\,|\,x_2]$, $W=\begin{bmatrix}W_1\\W_2\end{bmatrix}$, so $xW=x_1W_1+x_2W_2$ and the partial sums are added by an all-reduce. In a transformer MLP the first matrix is column-parallel and the second row-parallel, so one all-reduce per MLP suffices. TP is exact (identical to the unsplit computation) but communicates inside every layer, so it is confined to GPUs with fast interconnect.

**Pipeline parallel (PP).** Assign layer groups to $p$ stages; split the batch into $m$ micro-batches so stages overlap. During fill and drain stages idle. With a GPipe-style schedule the idle fraction of the whole run is

$$
\frac{p-1}{m+p-1},
$$

for example $p=4$: $m=1$ gives 75% idle, $m=16$ gives $3/19=15.8\%$. The extra time relative to the ideal is $(p-1)/m=18.75\%$ for $m=16$ (the source quotes this latter figure). Schedules such as 1F1B (start backward early, less activation memory), zero-bubble (split backward into input-gradient and weight-gradient parts to fill gaps) and **DualPipe** reduce it further. DualPipe feeds micro-batches from both ends of the pipeline and overlaps the all-to-all token exchange of MoE layers ([[adv.03]]) with compute, at the cost of holding two copies of the layers of each stage; DeepSeek-V3 reports more than 95% GPU utilisation.

**3D parallelism** combines all three: for example 8-way TP inside a node, 16-way PP across nodes and 128-way DP gives 16 384 GPUs. Example from the source: Llama 3 405B on 16 384 H100s.

## Mixed precision and loss scaling

fp32 has 8 exponent and 23 mantissa bits; **fp16** has 5 and 10 (range up to 65 504, smallest subnormal about $6\times10^{-8}$); **bf16** has 8 and 7 (fp32's range, coarser precision). Forward and backward run in 16-bit for speed and memory, while the **master weights and Adam moments stay in fp32**, because tiny updates $\eta\hat m/\sqrt{\hat v}$ would otherwise be rounded away.

**Gradient underflow.** In fp16 a gradient of $10^{-8}$ becomes exactly 0. **Loss scaling** multiplies the loss (hence all gradients, by linearity of backprop) by $S$ before the backward pass and divides the gradients by $S$ in fp32 before the optimizer step. With $S=1024$, $10^{-8}\to1.025\times10^{-5}$, representable; dividing back gives $1.0012\times10^{-8}$ (0.1% error). Too large an $S$ overflows to $\infty$. **Dynamic loss scaling**: on a non-finite gradient skip the step and halve $S$; after a streak (for example 2000) of clean steps double $S$. bf16 rarely needs scaling because its range matches fp32, which is why it became the default.

## Gradient checkpointing

Backward needs the forward activations of every layer. Say each of $L$ layers stores $A$ units and the input of a layer costs $a$ units ($a\ll A$). **Checkpointing** keeps only the inputs of every $k$-th layer; during backward, each segment of $k$ layers is re-run forward to regenerate its activations. Memory:

$$
M(k)=\frac{L}{k}\,a+k\,A .
$$

Minimising over $k$: $dM/dk=-La/k^2+A=0\Rightarrow k^\star=\sqrt{La/A}$, giving $M^\star=2\sqrt{LaA}$. With $a=A=1$ and $L=64$: $k^\star=8$ and $M=16$ instead of 64, a 4x saving.

**Cost.** A normal step is forward (1 unit) + backward (2 units) $=3$ per layer. Recomputing everything adds one forward: $4/3$, a **33%** overhead. **Selective** checkpointing recomputes only the expensive-to-store, cheap-to-recompute part (the $O(L^2)$ attention scores), say 15% of a forward: $(3+0.15)/3=1.05$, 5% overhead, per the source's cost model. Alternative: **offload** activations to CPU RAM if PCIe bandwidth is spare.

## Multi-token prediction (MTP)

Standard training supervises each position on one target, the next token. MTP adds heads for tokens further ahead. DeepSeek-V3's design is **sequential**: module $d$ takes the previous depth's hidden state, combines it (after normalisation and a projection) with the embedding of the true token at that depth, runs one transformer block, and predicts the next offset using the *shared* embedding and output head; chaining keeps the causal structure (unlike Gloeckle et al.'s independent parallel heads). The loss is

$$
\mathcal{L}=\mathcal{L}_{\text{main}}+\frac{\lambda}{D}\sum_{d=1}^{D}\mathcal{L}_d,\qquad \mathcal{L}_d=\operatorname{CE}\big(\text{head}_d(\cdot),\;t_{i+1+d}\big),
$$

with $\lambda=0.3$ early in training and $0.1$ later (source). In the exercise I implement the loss with $\lambda/D$ weighting and independent heads for simplicity; the sequential chaining is described, not implemented. Benefits: denser training signal, and the trained module can serve as a speculative-decoding draft at inference (reported 80%+ acceptance and about 1.8x throughput; see [[sys.02]]). Overhead: 14 B parameters on 671 B (2%).

## Worked example

1. **Memory.** 7.5 B parameters on 8 GPUs: ZeRO-0 111.76, ZeRO-1 38.42, ZeRO-2 26.19, ZeRO-3 13.97 GiB per GPU (model state only).
2. **Bubble.** $p=4$: $m=1,4,16$ give idle fractions $75\%,\ 42.9\%,\ 15.8\%$.
3. **Checkpoint.** $L=64$: $k=8$ is optimal for equal sizes; if inputs are four times cheaper ($a=0.25$) the best is $k=4$ with $M=16\cdot0.25+4=8$.
4. **Loss scaling.** $g=10^{-8}$: unscaled fp16 gives 0; $S=1024$ recovers $1.001\times10^{-8}$.
5. **MTP.** With uniform logits every head has loss $\ln V$; for $V=5$ that is 1.609, and the extra term is $0.3\ln5=0.483$.

All numbers come from the exercise code.

## Implementation

```python
import numpy as np

def model_state_gib(psi, n, stage, bp=2, bg=2, bo=12):
    p, g, o = bp * psi, bg * psi, bo * psi
    if stage >= 1: o /= n
    if stage >= 2: g /= n
    if stage >= 3: p /= n
    return (p + g + o) / 2**30

def checkpoint_memory(L, k, a_in, a_layer):
    return -(-L // k) * a_in + k * a_layer      # ceil(L/k) inputs + one live segment
```

Production: PyTorch `DistributedDataParallel` and `FullyShardedDataParallel` (FSDP2), DeepSpeed ZeRO, Megatron-Core (TP, PP, selective recomputation), `torch.utils.checkpoint`, `torch.autocast` with `GradScaler`.

## Shapes

| Quantity | Shape / size |
|---|---|
| model state per GPU (ZeRO-3) | $16\psi/N$ bytes |
| column-parallel block | $W_i$: (d_in, d_out/n) |
| row-parallel block | $x_i$: (B, d_in/n), $W_i$: (d_in/n, d_out); partial outputs (B, d_out) summed |
| pipeline stage | L/p consecutive layers |
| MTP logits | one (T, V) tensor per head |

## Common misconceptions

- **"More GPUs with data parallelism lets me train a bigger model."** Each replica holds everything; use ZeRO/FSDP, TP or PP for capacity.
- **"Mixed precision stores everything in fp16."** Master weights and optimizer moments stay fp32.
- **"Loss scaling changes the training result."** It cancels exactly (up to rounding); it only preserves small gradients.
- **"Checkpointing makes backward free of forward activations."** It recomputes them, costing about a third more FLOPs for full checkpointing.
- **"The pipeline bubble disappears with enough stages."** More stages make the bubble larger for a given $m$.
- **"ZeRO-3 is free."** It adds all-gathers per layer; overlap hides much but not all.

## Key takeaways

- Mixed-precision Adam costs 16 bytes per parameter; ZeRO stages 1-3 shard optimizer state, gradients, then parameters across $N$ GPUs.
- DP scales batch, TP splits matrices (exact, communication-heavy), PP splits layers (bubble $\frac{p-1}{m+p-1}$); production combines all three.
- Loss scaling keeps small fp16 gradients alive; bf16 largely avoids the issue.
- Checkpoint every $\sqrt{La/A}$ layers: memory $O(\sqrt L)$ for about 33% more compute (about 5% with selective recomputation).
- MTP adds $\frac{\lambda}{D}\sum_d\mathcal{L}_d$ for denser supervision and a free draft model.

## Explain it yourself

1. Why does adding data-parallel replicas never help a model that does not fit on one GPU, and which ZeRO stage first does?
2. Derive the optimal checkpoint interval and explain why attention scores are the first thing selective checkpointing recomputes.

## Further reading

- ZeRO (Rajbhandari et al., 2020), Megatron-LM (Narayanan et al., 2021), Chen et al. (2016) on sublinear-memory training, Micikevicius et al. (2018) on mixed precision, DeepSeek-V3 report; summaries in the source lessons.
- [[adv.03]] for MoE and expert parallelism, [[adv.04]] for inference-side memory.

## Mastery criteria

You can compute per-GPU model state for any ZeRO stage, split a matmul by columns and rows, compute pipeline bubbles, choose a checkpoint interval and its overhead, simulate fp16 underflow with and without loss scaling, and write the MTP loss.
