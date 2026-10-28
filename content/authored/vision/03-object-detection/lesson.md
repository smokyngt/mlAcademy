---
id: vis.03
title: "Object detection: boxes, IoU, NMS, YOLO and mAP"
summary: Predict a variable number of labelled boxes per image, deduplicate them with non-maximum suppression, and score a detector with average precision, all implemented from scratch.
minutes: 80
teaches: [vis.iou, vis.detection, vis.map]
requires: [vis.image-tensor, vis.classification, vis.transfer]
sources:
  - "ai-engineering-from-scratch/phases/04-computer-vision/06-object-detection-yolo/docs/en.md (Sec. 'The Problem' and 'The Concept': dense prediction, grid and anchors, decoding, IoU, NMS, loss structure, detection metrics; the Build It code was not read). READ."
---

## Where this fits

Classification says "this is a dog". Detection says "a dog at pixels (112, 40)-(280, 210), a cat at (400, 180)-(560, 310), and nothing else". Self-driving, retail analytics, medical screening and industrial inspection all need this, because the number of objects varies and their positions matter. Detection combines everything so far: image tensors ([[vis.01]]), classification and transfer learning ([[vis.02]]), and a CNN backbone ([[seq.01]]). It leads to segmentation in [[vis.04]], where the box becomes a per-pixel mask.

## The idea in plain words

Imagine laying a sheet of graph paper over the photo. Every square is asked: "is the centre of an object in you? If so, how far from your middle is it, how big is it, and what is it?" One look at the picture answers all squares in parallel, which is why single-shot detectors (YOLO, "You Only Look Once") are fast.

Two practical problems appear. First, a square should not invent box sizes from nothing, so each square is given a few template shapes (**anchors**: a tall one, a wide one, a big one) and only predicts a small nudge from the template. Second, neighbouring squares will often all shout about the same dog, producing a pile of overlapping boxes. **Non-maximum suppression** cleans this up: keep the most confident box and discard anything that overlaps it too much, then repeat.

To score the result you cannot count "right and wrong images". You ask, over all detections ranked from most to least confident, how many are real objects (precision) and how many of the real objects were found (recall), and summarise that trade-off in one number, **average precision**.

## Boxes and intersection over union

A box is `(x1, y1, x2, y2)` with `x1 < x2`, `y1 < y2` in pixels (other conventions store centre and size, `(cx, cy, w, h)`; know which one your library uses). The area of a box is $(x_2-x_1)(y_2-y_1)$.

The similarity between two boxes $A$ and $B$ is **intersection over union**:

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

The intersection of two axis-aligned boxes is itself a box with width $\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B))$ and height computed the same way. IoU is $1$ for identical boxes and $0$ for disjoint (or merely touching) ones. A prediction counts as correct when its IoU with a ground-truth box is above a threshold, commonly $0.5$; NMS uses IoU between two predictions.

## Single-shot detection: the output tensor

A backbone maps the image to a feature map with an $S\times S$ grid (stride $s$ pixels per cell, so $S=\text{input size}/s$). A small head of $1\times1$ convolutions predicts, for each of $B$ anchors in each cell, $5+C$ numbers:

- $t_x,t_y,t_w,t_h$: box offsets,
- one objectness score: "is an object centred here?",
- $C$ class scores.

For the classic setting $S=13$, $B=2$, $C=20$ (20 classes) this is $2\cdot25=50$ numbers per cell and $13\cdot13\cdot50=8450$ per image. The output is a tensor of shape $(N,\,S,\,S,\,B(5+C))$ or equivalently $(N,B(5+C),S,S)$.

### Why anchors, and how boxes are decoded

A $3\times3$ convolution on a small receptive field cannot easily regress a box several hundred pixels wide from nothing, and the same object appearing elsewhere in the image should produce the same *relative* output. So the network predicts offsets from fixed prior shapes $(a_w,a_h)$ and relative to the cell position $(c_x,c_y)$ (integer grid coordinates):

$$
\begin{aligned}
b_x&=(\sigma(t_x)+c_x)\,s, & b_y&=(\sigma(t_y)+c_y)\,s,\\
b_w&=a_w\,e^{t_w}, & b_h&=a_h\,e^{t_h}.
\end{aligned}
$$

Why these functions: the sigmoid $\sigma$ keeps the centre inside its own cell (so each cell is responsible only for objects centred in it); the exponential makes the width a positive multiple of the anchor, with $t_w=0$ meaning "the anchor fits" and $t_w=\ln2$ meaning "twice as wide". The box corners are $(b_x\mp b_w/2,\ b_y\mp b_h/2)$. Modern detectors use feature-pyramid networks with small anchors on high-resolution maps and large anchors on coarse maps; the principle is unchanged.

### The loss

The training signal has three parts, each with a weight $\lambda$:

$$
L=\lambda_{\text{box}}\!\!\sum_{\text{obj cells}}\!\!L_{\text{box}}
+\lambda_{\text{obj}}\!\!\sum_{\text{obj cells}}\!\!L_{\text{obj}}(\hat o,1)
+\lambda_{\text{noobj}}\!\!\sum_{\text{empty cells}}\!\!L_{\text{obj}}(\hat o,0)
+\lambda_{\text{cls}}\!\!\sum_{\text{obj cells}}\!\!L_{\text{cls}} .
$$

Each ground-truth box is assigned to the cell (and best-matching anchor) containing its centre. Box and class losses are computed only for those assigned slots. All other slots are trained only to output a low objectness score. Empty slots outnumber object slots by far, so $\lambda_{\text{noobj}}$ is small (the source mentions about $0.5$) or the loss is re-balanced (for example with focal loss), otherwise "predict nothing" dominates. Box regression with squared error is often replaced by IoU-based losses (GIoU, DIoU, CIoU) that optimise overlap directly.

## Non-maximum suppression

Input: boxes with scores. Output: the subset to keep.

```text
sort by score, descending
keep = []
while boxes remain:
    take the best box b, append it to keep
    delete every remaining box with IoU(b, box) > threshold
```

The threshold trades two errors. Too low (say 0.2) and two genuinely different, nearby objects (a crowd) get merged, so you lose true detections. Too high (say 0.9) and duplicates survive. A typical value is about 0.45 to 0.5. Variants (soft-NMS, DIoU-NMS) and end-to-end detectors that learn the suppression exist, but the purpose is the same: one box per object.

## Detection metrics

For one class, take every detection in the evaluation set and sort by confidence, highest first. Walk down the list.

**Matching.** A detection is a **true positive** (TP) if its best-IoU ground-truth box has $\mathrm{IoU}\ge\tau$ and that ground truth has not already been claimed by a higher-scoring detection; otherwise it is a **false positive** (FP). A second detection of an already-found object is therefore a false positive, which is why NMS matters. Ground truths never matched are false negatives.

**Precision and recall at rank $r$.** With $\mathrm{TP}_r$ true positives among the first $r$ detections and $G$ ground-truth boxes:

$$
P_r=\frac{\mathrm{TP}_r}{r},\qquad R_r=\frac{\mathrm{TP}_r}{G}.
$$

**Average precision** is the area under the precision-recall curve. To remove the zig-zag, make precision non-increasing from the right (the *envelope*): $\tilde P_r=\max_{r'\ge r}P_{r'}$. Then

$$
\mathrm{AP}=\sum_r (R_r-R_{r-1})\,\tilde P_r,\qquad R_0=0 .
$$

Each recall increment is weighted by the best precision achievable at that recall or beyond.

**mAP.** Average AP over classes. COCO-style **mAP@[.5:.95]** also averages over IoU thresholds $\tau\in\{0.5,0.55,\dots,0.95\}$, so it rewards tight boxes. Reading the numbers: high AP@0.5 but low AP@[.5:.95] means objects are found but localised loosely (improve the box loss); high precision but low recall means the detector is too timid (lower the score threshold, raise the objectness weight).

## Worked example

All numbers are reproduced by the exercise code.

**IoU.** Boxes $A=(0,0,10,10)$ and $B=(5,5,15,15)$. Intersection $5\times5=25$, union $100+100-25=175$, IoU $=25/175=0.143$. For $A=(0,0,10,10)$ against $(1,1,11,11)$ the IoU is $81/119=0.681$, and against $(0,0,10,9)$ it is $0.9$.

**NMS.** Boxes: $b_0=(0,0,10,10)$ score 0.9; $b_1=(1,1,11,11)$ score 0.8; $b_2=(50,50,60,60)$ score 0.7; $b_3=(0,0,10,9)$ score 0.6. Threshold 0.5. Keep $b_0$. IoU($b_0,b_1$)$=0.681>0.5$ so drop $b_1$; IoU($b_0,b_2$)$=0$ keep going; IoU($b_0,b_3$)$=0.9$ so drop $b_3$. Next best is $b_2$; nothing is left. Result: $[0,2]$.

**Decoding.** Cell $(c_x,c_y)=(3,2)$, stride 32, anchor $(60,40)$, all offsets zero. The sigmoid of 0 is 0.5, so the centre is $((0.5+3)\cdot32,\ (0.5+2)\cdot32)=(112,80)$, the size equals the anchor, and the box is $(82,60,142,100)$. Setting $t_w=\ln2$ doubles the width.

**Average precision.** With 4 ground-truth boxes and ranked flags $[1,1,0,1,0]$ (TP, TP, FP, TP, FP): precision $=[1,1,0.667,0.75,0.6]$, recall $=[0.25,0.5,0.5,0.75,0.75]$, envelope $=[1,1,0.75,0.75,0.6]$. $\mathrm{AP}=0.25\cdot1+0.25\cdot1+0\cdot0.75+0.25\cdot0.75+0\cdot0.6=0.6875$.

**Matching end to end.** Ground truths `(0,0,10,10)`, `(20,20,30,30)`, `(50,50,60,60)`, `(70,70,80,80)`; detections with scores 0.95 `(0,0,10,10)` (IoU 1), 0.9 `(21,21,31,31)` (IoU 0.68), 0.8 `(100,100,110,110)` (far away), 0.7 `(0,1,10,11)` (IoU 0.82 with the first ground truth, which is already claimed), 0.6 `(50,50,59,59)` (IoU 0.81). Flags at $\tau=0.5$: $[1,1,0,0,1]$, giving AP $=0.25+0.25+0.25\cdot0.6=0.65$. At $\tau=0.9$ only the first detection survives. Note the duplicate (score 0.7) is a false positive even though it overlaps well, which is the reason detectors apply NMS.

## Implementation

```python
import numpy as np

def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def nms(boxes, scores, thr):
    order = list(np.argsort(-np.asarray(scores, dtype=float), kind="stable"))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep

def average_precision(tp_flags, n_gt):
    tp = np.cumsum(tp_flags)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / n_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * envelope))
```

Production code does this on tensors: `torchvision.ops.nms` and `batched_nms` (per class), and evaluation with the COCO tools (`pycocotools`) or the library that ships with your detector (for example Ultralytics, torchvision detection references). Note that official benchmarks differ in details such as interpolation (all-point versus 101-point) and how crowd regions are treated, so a hand-written AP will not match them exactly.

## Shapes

| object | shape |
|---|---|
| backbone feature map | $(N,C_f,S,S)$ |
| raw head output | $(N,B(5+C),S,S)$ |
| decoded predictions | $(N,\,S\cdot S\cdot B,\,5+C)$: box, objectness, classes |
| boxes, scores | $(M,4)$, $(M,)$ |
| pairwise IoU matrix | $(M_A,M_B)$ |
| kept indices after NMS | $(K,)$ with $K\le M$ |

## Common misconceptions

- **"Detection accuracy is the fraction of correct boxes."** Ranked precision and recall are needed, because there is no fixed set of negatives and the score threshold is a free parameter.
- **"NMS is part of the network."** It is post-processing, usually run per class; it is not differentiable and is applied only at inference.
- **"Higher IoU threshold in NMS suppresses more."** Higher keeps more; it only suppresses boxes that overlap more than the threshold.
- **"IoU is a distance."** It is a similarity, and for disjoint boxes it gives 0 regardless of how far apart they are (the reason GIoU and DIoU losses exist).
- **"mAP@0.5 is enough."** It hides loose localisation; look at the stricter averaged metric.
- **"The network predicts pixel coordinates directly."** It predicts offsets from a cell and an anchor, decoded with sigmoid and exponential.

## Key takeaways

- Detection = classification + box regression at every grid position and anchor, plus objectness, followed by NMS.
- $\mathrm{IoU}=|A\cap B|/|A\cup B|$; it defines both "correct" and "duplicate".
- Decode with $b_x=(\sigma(t_x)+c_x)s$ and $b_w=a_we^{t_w}$.
- AP is the envelope-corrected area under the precision-recall curve of ranked detections; mAP averages over classes (and IoU thresholds for COCO).
- The no-object term must be down-weighted or re-balanced.

## Explain it yourself

1. Why does a second, well-placed detection of the same object count as a false positive, and which component of the system removes it?
2. Derive IoU for two boxes from their corner coordinates, and say what happens to its value when they only touch along an edge.

## Further reading

- The source lesson, and the YOLO papers (Redmon et al.) for the original design.
- [[vis.04]] for per-pixel outputs, [[vis.05]] for transformer backbones that replace the CNN, and [[cml.10]] for precision and recall in general.

## Mastery criteria

You can compute IoU and NMS by hand, decode a YOLO-style cell, explain each term of the detection loss, and compute AP from a ranked list of detections and ground truths.
