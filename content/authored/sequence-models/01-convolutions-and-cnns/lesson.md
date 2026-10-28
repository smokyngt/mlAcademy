---
id: seq.01
title: Convolutions and CNNs
summary: Cross-correlation as a sliding dot product, output-size arithmetic, channels, pooling, the full conv backward pass derived and gradient-checked, receptive fields and the road from LeNet to ResNet.
minutes: 100
teaches: [seq.conv, seq.cnn-arch]
requires: [nn.backprop, la.matrix, calc.chain, nn.activation]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/02-convolutions-from-scratch/docs/en.md (whole file up to the Build It / Use It sections: sliding kernel, 5x5 worked input and Sobel kernel, output-size formula, padding, stride, channels, parameter count, im2col, receptive field). READ."
  - "ai-engineering-from-scratch/phases/04-computer-vision/03-cnns-lenet-to-resnet/docs/en.md (The Concept: LeNet-5 layout, AlexNet, VGG, Inception, degradation problem, ResNet; Build It steps 1-2 for the LeNet parameter count 61,706). READ."
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md (The Concept and TextCNN: 1D convolution over word embeddings as a learnable n-gram detector). READ."
---

## Learning objectives

- Explain a convolution layer as a small dense layer slid over the input with **shared weights**, and compute its output with a dot product by hand.
- Compute output sizes, parameter counts and receptive fields for any stack of conv and pool layers.
- Derive the backward pass of a conv layer (gradients for $W$, $b$ and the input) and express it with matrix products.
- Implement conv2d forward and backward in NumPy and verify them with finite differences.
- Say what LeNet, AlexNet, VGG and ResNet each added, and why a residual connection helps gradients.

## Prerequisites

Backpropagation with shapes ([[f2n.10]]) and the training loop ([[f2n.11]]). You should be comfortable with matrix products and the chain rule.

## Where this fits

Until now every input was a flat vector. Real data has **structure**: pixels have neighbours, audio samples have neighbours, words in a sentence have neighbours. A fully connected layer ignores that. Feeding a $224\times224$ RGB image to a dense layer with 1000 units needs $224\cdot224\cdot3\cdot1000+1000=150{,}529{,}000$ parameters, and a cat that moves three pixels to the right looks like a completely different input.

Convolutions fix both problems with one idea: look at a small neighbourhood, with the same detector everywhere. That makes images tractable and it is also the first of three ways to handle **sequences** in this course. A 1D convolution over word embeddings is a trainable n-gram detector (the TextCNN idea), and the same sliding-window thinking reappears when we meet recurrent networks ([[seq.03]]) and attention ([[seq.05]]).

## The idea in plain words

Imagine a small stamp with a pattern printed on it, say a pattern shaped like a vertical edge. You press the stamp on every possible spot of a photo and, at each spot, write down how well the stamp matches what is underneath. The sheet of scores you end up with is a **feature map**: bright where the photo contains that pattern, dark elsewhere. The stamp is the **kernel** (or filter).

Three things make this powerful.

- **One stamp, many places.** The same few numbers are reused at every position. This is *parameter sharing*, and it is why a convolution layer is tiny compared to a dense layer.
- **Local.** Each score looks only at a small patch, so detectors learn local patterns such as edges, corners and textures.
- **Many stamps.** A layer owns many kernels, each producing its own feature map. Stack layers and the later stamps match patterns of patterns: edges become corners, corners become eyes, eyes become faces.

Pooling is the other ingredient: after matching, shrink the sheet by keeping only the strongest score in each small block. You lose exact position, you keep "this pattern was somewhere around here".

## Formal definition

For one input channel, an image $x\in\mathbb{R}^{H\times W}$ and a kernel $w\in\mathbb{R}^{K\times K}$, the (valid, stride 1) output is

$$
y_{ij}=\sum_{u=0}^{K-1}\sum_{v=0}^{K-1} w_{uv}\,x_{i+u,\,j+v}+b .
$$

Strictly this is a **cross-correlation** (the kernel is not flipped as in the textbook convolution). Deep-learning libraries all do this; since the kernel is learned, the flip would make no difference except a relabelling of the weights.

With several channels, a layer has input $x\in\mathbb{R}^{C_{in}\times H\times W}$, weights $W\in\mathbb{R}^{C_{out}\times C_{in}\times K\times K}$ and bias $b\in\mathbb{R}^{C_{out}}$. With padding $P$ (zeros added on all four sides) and stride $S$:

$$
y_{o,i,j}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1} W_{o,c,u,v}\;x^{\mathrm{pad}}_{c,\;iS+u,\;jS+v}.
$$

Each output value is a dot product between the kernel $W_{o}$ (all input channels at once, a block of $C_{in}K^2$ numbers) and one window of the padded input.

### Output size

The window's top-left corner can be at rows $0,S,2S,\dots$ as long as the window fits inside the padded height $H+2P$. The last valid start $iS$ satisfies $iS+K\le H+2P$, hence $i\le (H+2P-K)/S$, and counting from 0:

$$
H_{out}=\Big\lfloor\frac{H-K+2P}{S}\Big\rfloor+1 .
$$

Same for the width. "Same" padding at stride 1 means $H_{out}=H$, i.e. $P=(K-1)/2$ for odd $K$.

### Parameters

$$
\#\text{params}=C_{out}\,C_{in}\,K^2+C_{out}.
$$

It does **not** depend on $H,W$. A $3\to64$ layer with $3\times3$ kernels has $64\cdot3\cdot9+64=1792$ parameters whether the image is $32\times32$ or $4000\times4000$. Doing the same map with a dense layer on a $32\times32$ image would need $(3\cdot32\cdot32)(64\cdot32\cdot32)\approx2.0\times10^{8}$ weights.

### Pooling

Max pooling with window $k$ and stride $k$ takes the maximum over each non-overlapping $k\times k$ block, independently per channel. It has **no parameters**, halves the resolution for $k=2$, and adds a little robustness to small shifts. Average pooling takes the mean instead (LeNet used it).

### Receptive field

The receptive field of a unit is the patch of the *input* that can influence it. Track two numbers while walking through the layers: the current field size $r$ (start at 1) and the "jump" $j$, the distance in input pixels between two neighbouring units (start at 1). A layer with kernel $K$ and stride $S$ updates

$$
r\leftarrow r+(K-1)\,j,\qquad j\leftarrow j\,S.
$$

Three stacked $3\times3$ stride-1 convs give $r=1+3\cdot2=7$. Two $3\times3$ convs give $r=5$, the same as one $5\times5$ conv, but use $2\cdot9C^2=18C^2$ weights instead of $25C^2$ (for $C$ channels in and out) and have an extra non-linearity in between. That observation is the whole design of VGG.

## Derivation: the backward pass

Let $L$ be a scalar loss and $\bar Y:=\partial L/\partial Y$ the upstream gradient, with the shape of $Y$: $(C_{out},H_{out},W_{out})$. Write $x^{\mathrm{pad}}$ as $\tilde x$. The forward formula again:

$$
Y_{o,i,j}=b_o+\sum_{c,u,v}W_{o,c,u,v}\,\tilde x_{c,\,iS+u,\,jS+v}.
$$

**Bias.** $b_o$ appears in every position of channel $o$, so by the chain rule $\partial Y_{o,i,j}/\partial b_o=1$ and

$$
\bar b_o=\sum_{i,j}\bar Y_{o,i,j}\qquad\text{(shape }C_{out}\text{).}
$$

**Weights.** $\partial Y_{o,i,j}/\partial W_{o,c,u,v}=\tilde x_{c,iS+u,jS+v}$ and the weight is used at *every* $(i,j)$ (that is exactly parameter sharing), so the contributions add up:

$$
\bar W_{o,c,u,v}=\sum_{i,j}\bar Y_{o,i,j}\;\tilde x_{c,\,iS+u,\,jS+v}.
$$

This is itself a cross-correlation of the padded input with the upstream gradient.

**Input.** $\partial Y_{o,i,j}/\partial\tilde x_{c,p,q}=W_{o,c,u,v}$ whenever $p=iS+u$ and $q=jS+v$, and 0 otherwise. A pixel $\tilde x_{c,p,q}$ is used by every window that covers it, so

$$
\bar{\tilde x}_{c,p,q}=\sum_{o}\sum_{\substack{i,j,u,v:\\ iS+u=p,\ jS+v=q}}\bar Y_{o,i,j}\,W_{o,c,u,v}.
$$

For $S=1$ this is a "full" correlation of $\bar Y$ with the spatially flipped kernel. Finally $\bar x$ is $\bar{\tilde x}$ with the padding cropped off (padding pixels are constants, not variables).

### The same thing as matrix products (im2col)

Stack every window of $\tilde x$ as a column of a matrix $X_{col}\in\mathbb{R}^{(C_{in}K^2)\times(H_{out}W_{out})}$ (this is `im2col`). Flatten the weights to $W_{mat}\in\mathbb{R}^{C_{out}\times(C_{in}K^2)}$ and $\bar Y$ to $\bar Y_{mat}\in\mathbb{R}^{C_{out}\times(H_{out}W_{out})}$. Then

$$
Y_{mat}=W_{mat}X_{col}+b\mathbf{1}^\top .
$$

That is an ordinary linear layer where the "batch" is the set of windows, so the formulas from [[f2n.10]] apply verbatim, with the transposes on the other side because here samples are columns:

$$
\bar W_{mat}=\bar Y_{mat}\,X_{col}^\top,\qquad \bar b=\bar Y_{mat}\mathbf 1,\qquad \bar X_{col}=W_{mat}^\top\,\bar Y_{mat}.
$$

Shapes: $(C_{out}\times HW)(HW\times C_{in}K^2)=C_{out}\times C_{in}K^2$, and $(C_{in}K^2\times C_{out})(C_{out}\times HW)=C_{in}K^2\times HW$. The last step, `col2im`, takes each column of $\bar X_{col}$ and **adds** it back into the window it was copied from. Windows overlap, so every input pixel receives the sum of its contributions: that is the sum over $(i,j,u,v)$ in the formula above.

## Worked example

Take the $5\times5$ input from the source lesson and a Sobel-like kernel (stride 1, no padding):

$$
x=\begin{pmatrix}1&2&0&1&2\\0&1&3&1&0\\2&1&0&2&1\\1&0&2&1&3\\2&1&1&0&1\end{pmatrix},\qquad
w=\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}.
$$

Output size: $(5-3+0)/1+1=3$. The top-left value is the dot product of $w$ with the top-left $3\times3$ window $\begin{pmatrix}1&2&0\\0&1&3\\2&1&0\end{pmatrix}$:

- row 0: $1\cdot1+2\cdot0+0\cdot(-1)=1$
- row 1: $0\cdot2+1\cdot0+3\cdot(-2)=-6$
- row 2: $2\cdot1+1\cdot0+0\cdot(-1)=2$

so $Y_{00}=1-6+2=-3$. Doing all nine windows gives (checked by the exercise code)

$$
Y=\begin{pmatrix}-3&0&3\\0&-3&0\\1&-2&-3\end{pmatrix}.
$$

With stride 2 and padding 1 the output size is $\lfloor(5-3+2)/2\rfloor+1=3$ again, with values $\begin{pmatrix}-5&2&3\\-3&-3&6\\-2&1&1\end{pmatrix}$.

**Backward**, with the simplest upstream gradient $\bar Y=\mathbf{1}$ (all ones, i.e. loss $=\sum Y$): then $\bar b=9$, and $\bar W_{uv}$ is the sum of $x_{i+u,j+v}$ over the nine windows, i.e. the sum of a $3\times3$ block of $x$ shifted by $(u,v)$:

$$
\bar W=\begin{pmatrix}10&11&10\\10&11&13\\10&8&11\end{pmatrix},\qquad
\bar x=\begin{pmatrix}1&1&0&-1&-1\\3&3&0&-3&-3\\4&4&0&-4&-4\\3&3&0&-3&-3\\1&1&0&-1&-1\end{pmatrix}.
$$

Check the corner: $x_{00}$ belongs to a single window (the top-left one) at kernel position $(0,0)$, so $\bar x_{00}=w_{00}=1$. The centre pixel $x_{22}$ belongs to all nine windows, each time at a different kernel position, so $\bar x_{22}=\sum_{uv}w_{uv}=0$ ✓.

## Implementation

```python
import numpy as np

def im2col(x, K, stride, pad):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - K) // stride + 1
    Wo = (W + 2 * pad - K) // stride + 1
    cols = np.zeros((C * K * K, Ho * Wo))
    for i in range(Ho):
        for j in range(Wo):
            cols[:, i * Wo + j] = xp[:, i*stride:i*stride+K, j*stride:j*stride+K].reshape(-1)
    return cols, Ho, Wo

def conv2d_forward(x, W, b, stride=1, pad=0):
    Co, Ci, K, _ = W.shape
    cols, Ho, Wo = im2col(x, K, stride, pad)
    out = W.reshape(Co, -1) @ cols + b[:, None]
    return out.reshape(Co, Ho, Wo), (cols, x.shape, W, stride, pad)

def conv2d_backward(dout, cache):
    cols, x_shape, W, stride, pad = cache
    Co = W.shape[0]
    dY = dout.reshape(Co, -1)
    dW = (dY @ cols.T).reshape(W.shape)
    db = dY.sum(axis=1)
    dcols = W.reshape(Co, -1).T @ dY
    dx = col2im(dcols, x_shape, W.shape[2], stride, pad)   # scatter-add, crop padding
    return dx, dW, db
```

`col2im` is the loop that does `xp[:, rows, cols] += column.reshape(C, K, K)` for each output position and crops the padding at the end; you write it in the exercise.

### Shapes

| Quantity | Shape |
|---|---|
| input $x$ | $(C_{in},H,W)$ |
| weights $W$, bias $b$ | $(C_{out},C_{in},K,K)$, $(C_{out})$ |
| $X_{col}$ | $(C_{in}K^2,\;H_{out}W_{out})$ |
| output $Y$ | $(C_{out},H_{out},W_{out})$ |
| $\bar W$, $\bar b$, $\bar x$ | same as $W$, $b$, $x$ |

With a batch you simply add a leading axis $B$ to $x$ and $Y$.

## Gradient check

Pick random tensors $R$ with the shape of $Y$ and use the scalar $L=\sum R\odot Y$, for which $\bar Y=R$. Compare `conv2d_backward(R, cache)` with central differences, as in [[f2n.10]]. The first exercise does this for four (stride, padding) combinations and expects agreement to $\sim10^{-6}$. The classic bug is writing `=` instead of `+=` in `col2im`: gradients are then too small in the overlapping regions and finite differences catch it at once.

## From LeNet to ResNet in brief

- **LeNet-5 (1998)**: conv $5\times5$ ($1\to6$) $\to$ pool $\to$ conv $5\times5$ ($6\to16$) $\to$ pool $\to$ dense $400\to120\to84\to10$. Feature maps go $32\to28\to14\to10\to5$; the whole network has $61{,}706$ parameters (check: $156+2416+48120+10164+850$). This conv-pool-dense template is still the skeleton of every CNN.
- **AlexNet (2012)**: ReLU instead of tanh, dropout, and far more depth and width on GPUs.
- **VGG (2014)**: only $3\times3$ convs, stacked, pooled; justified by the receptive-field arithmetic above.
- **Inception (2014)**: run several kernel sizes in parallel and concatenate along the channel axis; $1\times1$ convs reduce channels cheaply.
- **ResNet (2015)**: the residual block $y=x+F(x)$.

### Why the residual connection helps gradients

Very deep plain networks got *worse* training loss than shallower ones (the degradation problem): gradients are products of many local factors, each typically smaller than 1 in magnitude, so they shrink exponentially with depth. For a residual block $y=x+F(x)$ the Jacobian is

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x},
$$

so the upstream gradient becomes $\bar x=\bar y+\bar y\,\partial F/\partial x$: it always contains a **direct copy** of $\bar y$ that does not pass through $F$ at all. Across many blocks the gradient therefore has a path of pure identities from the loss to early layers. The block can also represent "do nothing" by setting $F\approx0$, so adding a block is never forced to hurt. The same trick reappears as the additive cell-state path of the LSTM ([[seq.04]]) and in every transformer block.

## Common misconceptions

- **"A convolution layer flips its kernel."** In practice it does not (cross-correlation); because the kernel is learned this changes nothing.
- **"Pooling has weights."** Max and average pooling have no parameters. Max pooling's backward pass just routes the gradient to the arg-max position.
- **"More pixels means more parameters."** A conv layer's parameter count is independent of the image size; its *compute* and *activation memory* grow with it.
- **"The receptive field equals the kernel size."** It grows with depth (and multiplicatively with strides): deep units see large parts of the image.
- **"Padding is learned or is data."** It is a constant (zeros), so no gradient is kept for padded positions.

## Limits and trade-offs

- Convolutions assume **locality** and **translation equivariance**. That is a strong prior: great for images and audio, wrong when far-apart elements must interact directly (long-range dependencies need many layers or a different mechanism such as attention).
- The prior also means CNNs are not invariant to rotation or scale by themselves; data augmentation or architecture changes are needed.
- im2col trades memory for speed: $X_{col}$ is about $K^2$ times larger than $x$. Production libraries use smarter kernels.
- Max pooling discards position information. Strided convs are often used instead.
- For text, a 1D convolution of width $k$ over word embeddings detects $k$-grams in parallel and cheaply, but cannot carry information beyond its window, unlike a recurrent network.

## Key takeaways

- A conv layer is a dense layer applied to every window with shared weights: $\#\text{params}=C_{out}C_{in}K^2+C_{out}$.
- $H_{out}=\lfloor (H-K+2P)/S\rfloor+1$; receptive field grows with $r\leftarrow r+(K-1)j$, $j\leftarrow jS$.
- Backward: $\bar W=\bar Y_{mat}X_{col}^\top$, $\bar b=\sum\bar Y$, $\bar X_{col}=W_{mat}^\top\bar Y_{mat}$ followed by a scatter-**add** (`col2im`).
- Residual blocks give gradients an identity path: $\partial y/\partial x=I+\partial F/\partial x$.

## Explain it yourself

1. Why does parameter sharing both cut the number of weights and make the weight gradient a *sum over positions*?
2. Derive $\bar x$ for a $3\times3$ kernel at stride 1 for one interior pixel and explain why `col2im` must add rather than assign.

## Further reading

- Source library: `ai-engineering-from-scratch` phase 04, lessons 02 (convolutions from scratch) and 03 (LeNet to ResNet).
- Same library, phase 05 lesson 08, for 1D convolutions on text.
- Next: [[seq.02]] (word embeddings), the representation that feeds both TextCNNs and recurrent networks.

## Mastery criteria

You can compute the output size, parameter count and receptive field of a small CNN on paper, write the three backward formulas of a conv layer with shapes, and your NumPy conv2d passes a finite-difference check for several strides and paddings.
