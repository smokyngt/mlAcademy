---
id: vis.01
title: "Images as data: pixels, channels, normalisation and filters"
summary: Treat an image as a tensor of numbers, prepare it exactly as a model expects, and detect edges with a sliding-window filter written from scratch.
minutes: 60
teaches: [vis.image-tensor, vis.filter, vis.augment]
requires: [la.matrix, prob.rv]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/01-image-fundamentals/docs/en.md (whole file: pixels as samples, channels, HWC vs CHW, byte ranges, colour spaces, grayscale weights, resizing and interpolation, ImageNet standardisation, preprocessing order). READ."
  - "ai-engineering-from-scratch/phases/04-computer-vision/04-image-classification/docs/en.md (Sec. 'Why augmentation works', flip, reflect-padded random crop, mixup; used for the augmentation section). READ."
---

## Where this fits

Every vision system, from a phone's face unlock to a tumour detector, begins the same way: a camera produces numbers, and a program turns those numbers into the exact tensor a model was trained on. Most "the model is bad" bugs in applied vision are not model bugs. They are preprocessing bugs: BGR instead of RGB, `uint8` instead of `float`, channels in the wrong place, or the wrong mean and standard deviation. This lesson fixes the foundation for the whole Vision course.

It also introduces the sliding-window filter. If you have already met convolutions and CNNs (see [[seq.01]]), the filters here are the hand-designed ancestors of the filters a CNN learns; if you have not, nothing here needs more than loops and multiplication. Later lessons build on this one: [[vis.02]] trains classifiers on these tensors, [[vis.03]] puts boxes on them, [[vis.04]] labels every pixel.

## The idea in plain words

A photograph is a grid of tiny light meters. Each meter reports "how much light reached me", and a pixel is just that one report, not a little coloured square with an area. A grayscale image has one report per location. A colour image has three: how much red-filtered, green-filtered and blue-filtered light arrived. So a colour image is three grayscale pictures stacked, one per channel.

A model is like a very fussy cook who learned a recipe with ingredients measured in grams: if you hand over the same ingredients in ounces the dish silently goes wrong. The "unit" for images is the scale and centring of the numbers (0 to 255 whole numbers versus 0 to 1 decimals versus centred and rescaled values) and the order of the axes. You must prepare your image the way the training images were prepared.

A filter is a small stencil you slide over the picture. At each position you multiply the stencil's numbers with the pixels underneath and add them up. A stencil of equal weights blurs (it averages a neighbourhood). A stencil that is negative on the left and positive on the right answers the question "is it brighter on the right than on the left here?", which is exactly an edge detector.

## Setup: images as tensors

A colour image of height $H$ and width $W$ is an array with three axes.

- **HWC** layout: shape $(H, W, 3)$, pixels interleaved. Files, PIL, OpenCV and matplotlib use it.
- **CHW** layout: shape $(3, H, W)$, three stacked planes. PyTorch convolution layers expect it, and a batch is $(N, 3, H, W)$ (NCHW).

Moving between them is only an axis permutation: `chw = hwc.transpose(2, 0, 1)`.

Three numeric conventions you meet in practice:

| name | dtype | range | where |
|---|---|---|---|
| raw | `uint8` | 0..255 | files, decoders |
| normalised | `float` | 0..1 | after dividing by 255 |
| standardised | `float` | roughly $-2$..$2$ | after subtracting the per-channel mean and dividing by the per-channel std |

Standardisation per channel $c$ is

$$
x'_{c} = \frac{x_{c}/255 - \mu_c}{\sigma_c}.
$$

The common ImageNet values are $\mu = (0.485, 0.456, 0.406)$ and $\sigma = (0.229, 0.224, 0.225)$, the mean and standard deviation of each channel over the ImageNet training set, computed on 0..1 pixels. A model pre-trained on ImageNet expects these; a model trained on another dataset expects that dataset's statistics.

The full order that production pipelines follow is: decode (`uint8` HWC), fix the colour order, resize (shorter side), crop to the model size, divide by 255, standardise, transpose to CHW, add the batch axis. Reordering the steps changes what the model sees.

### Colour spaces and grayscale

RGB is the capture format, but other coordinates are useful. HSV separates the colour (hue) from the brightness (value), which helps classical colour thresholding. YCbCr separates brightness (Y) from two colour-difference channels; JPEG and video codecs compress the colour channels harder because the eye is less sensitive to them. For a model you almost always feed RGB, or grayscale when colour is a nuisance (documents, OCR).

Grayscale is a weighted sum, not a plain average, because the eye is most sensitive to green:

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B .
$$

### Resizing

Models have a fixed input size, so images are resized. Nearest-neighbour copies the closest source pixel (blocky, but the only safe choice for label images holding class ids). Bilinear blends the four surrounding pixels by distance (smooth; the default for training). Bicubic and Lanczos are sharper and slower, used for images people will look at. Resize-then-centre-crop keeps the aspect ratio and drops a border; resize-and-pad keeps every pixel (detection, OCR); direct stretch is cheap but distorts shapes.

## Derivation: a filter as a sliding dot product

Take a grayscale image $I\in\mathbb{R}^{H\times W}$ and a kernel $K\in\mathbb{R}^{k_h\times k_w}$. The **valid** filtered image (no padding, stride 1) is

$$
O[i,j]=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1} K[u,v]\; I[i+u,\,j+v].
$$

This is the dot product between the kernel and the window of the image starting at $(i,j)$. Deep-learning libraries call this "convolution" although mathematically it is cross-correlation: true convolution flips the kernel first. For a learned kernel the flip is irrelevant, because the network can learn the flipped version.

**Output size.** The window must fit inside the image, so $i$ runs over $H-k_h+1$ values and $j$ over $W-k_w+1$:

$$
O\in\mathbb{R}^{(H-k_h+1)\times(W-k_w+1)}.
$$

A $3\times3$ kernel therefore loses one pixel on every border; padding the image with zeros or reflected values keeps the size.

**Blur.** $K=\tfrac19\mathbf 1_{3\times3}$ replaces each pixel with the mean of its neighbourhood, which averages out noise but also softens edges.

**Sobel edges.** The kernel

$$
K_x=\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}
$$

takes right-column pixels minus left-column pixels, weighting the middle row twice. Over a region of constant brightness the weights sum to zero, so the response is zero. Across a vertical step it is large. The transposed kernel $K_y=K_x^\top$ responds to horizontal edges. The edge strength combines both:

$$
G=\sqrt{G_x^2+G_y^2},\qquad G_x=I\star K_x,\ G_y=I\star K_x^\top .
$$

**Histogram.** Counting how many pixels fall in each brightness bin gives a one-dimensional summary of the image that ignores location. It is the quickest check that an image is under-exposed (mass piled near 0) or clipped (piled at 255).

## Worked example

All values below are produced by the exercise code.

**Standardising one pixel.** The pixel $(R,G,B)=(210,140,30)$ divided by 255 is $(0.8235, 0.5490, 0.1176)$. Subtract the ImageNet mean and divide by the std: $(1.478,\ 0.415,\ -1.282)$. Its grayscale value is $0.299\cdot210+0.587\cdot140+0.114\cdot30=148.39$.

**A vertical step edge.** The $5\times5$ image whose every row is $(0,0,0,10,10)$ has an edge between columns 2 and 3.

- Sobel $G_x$ with the valid $3\times3$ kernel has shape $3\times3$ and every row equal to $(0,\ 40,\ 40)$. For the window on columns 1..3 the right column minus left column is $10-0$ for each of three rows, weighted $1+2+1=4$, giving $40$. For the window on columns 0..2 everything is 0.
- $G_y=0$ everywhere, so the magnitude is also $(0,40,40)$ per row.
- The $3\times3$ box blur gives rows $(0,\ 3.33,\ 6.67)$: the sharp step has become a ramp.

**Histogram.** The 6 numbers `[0,10,200,255,128,64]` in 4 bins of width 64 produce counts $(2,1,1,2)$.

## Augmentation (preview)

A network has no built-in invariance to flips, crops or lighting changes. Showing it altered copies of each training image, with the same label, teaches it to ignore those changes. The rule: an augmentation must **preserve the label**. A horizontal flip is safe for "dog" but not for text or for a left/right-handed classification; a rotation can turn a 6 into a 9. Common choices:

- **Random flip**: mirror with probability $0.5$.
- **Random crop**: pad (reflection padding is better than black borders) and crop back to the original size at a random offset.
- **Colour jitter**: small random changes of brightness and contrast.
- **Mixup**: blend two images and their labels, $x=\lambda x_i+(1-\lambda)x_j$, $y=\lambda y_i+(1-\lambda)y_j$.

Augmentation is applied to training images only. Validation and test images get only the deterministic part of the pipeline (resize, crop, standardise). [[vis.02]] shows how to measure whether it helps.

## Implementation

```python
import numpy as np

def to_gray(rgb):                       # (H, W, 3) -> (H, W)
    return rgb.astype(float) @ np.array([0.299, 0.587, 0.114])

def standardize(img_uint8, mean, std):  # (H, W, 3) uint8 -> (3, H, W) float
    x = img_uint8.astype(float) / 255.0
    x = (x - np.asarray(mean)) / np.asarray(std)
    return x.transpose(2, 0, 1)

def conv2d_valid(img, kernel):          # (H, W), (kh, kw) -> (H-kh+1, W-kw+1)
    H, W = img.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out

Kx = np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
def sobel_magnitude(gray):
    return np.sqrt(conv2d_valid(gray, Kx) ** 2 + conv2d_valid(gray, Kx.T) ** 2)
```

Production libraries do the same with batched, device-aware tensors: `torchvision.transforms` for the pipeline (`Resize`, `CenterCrop`, `ToTensor`, `Normalize`) and `torch.nn.functional.conv2d` or OpenCV's `filter2D` for filters. The two nested Python loops above are slow by design; libraries vectorise them.

## Shapes

| object | shape |
|---|---|
| RGB image, HWC | $(H,W,3)$ |
| same image, CHW | $(3,H,W)$ |
| batch for a model | $(N,3,H,W)$ |
| per-channel mean / std | $(3,)$, broadcast against the last axis of HWC (so standardise **before** transposing) |
| valid filtering with a $k_h\times k_w$ kernel | $(H-k_h+1,\ W-k_w+1)$ |

## Common misconceptions

- **"A pixel is a coloured square."** It is one sample of light intensity per channel; resizing and rotating resample those measurements.
- **"Dividing by 255 is normalising enough."** It only puts values in 0..1. A model trained on standardised inputs still sees the wrong distribution.
- **"Convolution flips the kernel in deep learning."** Libraries implement cross-correlation; for learned kernels the difference does not matter, but copying a textbook kernel direction does.
- **"Grayscale is the average of R, G, B."** The eye weights green most; use the weighted sum.
- **"Any resize will do."** Nearest-neighbour on a segmentation mask is correct; bilinear on a mask invents class ids that do not exist.
- **"More augmentation is always better."** An augmentation that changes the label (rotating digits, flipping text) damages the data.

## Key takeaways

- An image is a tensor $(H,W,C)$ or $(C,H,W)$; check dtype, range, channel order and layout before anything else.
- Reproduce the training preprocessing exactly: colour order, resize, crop, divide by 255, standardise with the right statistics, transpose.
- A filter is a sliding dot product; the valid output has size $(H-k_h+1)\times(W-k_w+1)$.
- Blur averages, Sobel differences; a kernel whose weights sum to zero ignores flat regions.
- Augmentations must preserve the label and are used at training time only.

## Explain it yourself

1. A colleague feeds a `uint8` array straight into a pre-trained model and gets confident but wrong predictions. Explain what is wrong and list the exact steps that would fix it.
2. Why does the Sobel kernel have weights that sum to zero, and what would a response of exactly zero mean at a pixel?

## Further reading

- The source lesson's reading list (Poynton on colour spaces, the torchvision transforms and models documentation).
- [[seq.01]] for convolution as a learnable layer with channels, strides and pooling.

## Mastery criteria

You can convert between HWC and CHW and between `uint8`, 0..1 and standardised ranges without a reference, predict the output shape of a valid filter, implement and test a filter from scratch, and explain why each step of an ImageNet-style pipeline is there.
