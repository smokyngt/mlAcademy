---
id: cml.06
title: Decision trees and random forests
summary: Grow a tree by greedy impurity-reducing splits, see why it overfits, then average many decorrelated trees (bagging, random forests) and read the out-of-bag error for free.
minutes: 75
teaches: [ml.trees, ml.bagging]
requires: [ml.generalization, prob.basics, info.entropy]
sources:
  - "Bishop, Pattern Recognition and Machine Learning, Sec. 14.2 Committees (PDF p.675-677): bootstrap aggregation, error of an averaged committee. READ."
  - "Bishop, PRML, Sec. 14.4 Tree-based Models (PDF p.683-686): CART, greedy growth, cost-complexity pruning, Gini and cross-entropy, weaknesses of trees. READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/04-decision-trees/docs/en.md (split criteria, stopping rules, random forest, MDI vs permutation importance). READ."
  - "ai-engineering-from-scratch/phases/02-ml-fundamentals/11-ensemble-methods/docs/en.md (bagging, 63.2 percent in-bag, out-of-bag samples). READ (bagging and boosting sections)."
---

## Where this fits

Most business data is a table: customers, transactions, sensor readings, patient records. Columns have different units, some are categories, relationships are lumpy ("risk jumps once income drops below a threshold *and* the account is new"). Linear models (lessons 01-02) need you to engineer those interactions by hand. Trees discover them: they ask a sequence of yes/no questions about single columns, and each answer narrows the problem. On tabular data, trees and forests (and their boosted cousins, next lesson) are the default strong baseline and a regular winner of competitions. Their weakness is the theme of lesson 03: one tree has low bias and very high variance. Averaging many of them is the cure, and it is one of the cleanest applications of the bias-variance idea you will meet.

## The idea in plain words

Think of the game "twenty questions". You do not start by asking "is it a golden retriever?"; you start with a question that splits the possibilities as evenly as possible ("is it alive?"), then ask the best next question *given the answer*. A decision tree is that strategy, learned from data: at the top, pick the single question about one feature ("age < 30?") that makes the two resulting groups as *pure* as possible, meaning each group is mostly one class. Then repeat inside each group, independently, until the groups are pure or too small. To predict, drop a new example down the questions and report the majority class (or the average value, for regression) of the group where it lands.

Two things follow. First, a tree cuts the input space into boxes with sides parallel to the axes and predicts a constant in every box. Second, if you let it keep asking questions it ends up with one box per training point: it has memorised the data, noise included. Change a few training rows and the questions at the top can change completely, which is why a single tree is unstable.

A **forest** exploits that instability. Train many trees on slightly different random versions of the data and let them vote. Their individual mistakes point in different directions and largely cancel, like a jury of independent witnesses.

## Impurity: what "pure" means

Let a node contain $n$ examples and let $p_k$ be the fraction in class $k$. Two standard measures of disorder (Bishop writes them for a region $R_\tau$ as in his Eq. 14.32-14.33):

$$
G=\sum_k p_k(1-p_k)=1-\sum_k p_k^2 \qquad\text{(Gini)}
$$

$$
H=-\sum_k p_k\log_2 p_k \qquad\text{(entropy, in bits)}
$$

Both are $0$ for a pure node and largest when classes are evenly mixed ($G=0.5$, $H=1$ bit for two classes). Gini has a nice reading: it is the probability that a random example from the node is mislabelled if you label it by drawing a class from the node's own class distribution. Why not simply use the misclassification rate? Bishop points out that Gini and entropy react to changes in node proportions more smoothly and are differentiable; misclassification can stay flat across splits that are clearly useful. (He notes the misclassification rate is still the usual choice when pruning.)

For regression the "impurity" is the sum of squared errors around the node mean.

## Derivation: the greedy split

A split is a pair (feature $j$, threshold $\theta$). It sends $n_L$ examples left ($x_j\le\theta$) and $n_R$ right. Its **impurity decrease** (information gain when $H$ is used) is

$$
\Delta = I(\text{parent})-\frac{n_L}{n}\,I(\text{left})-\frac{n_R}{n}\,I(\text{right}).
$$

The weights $n_L/n$ and $n_R/n$ matter: a split that peels off one pure example is almost worthless, a split that sends half the data into a pure child is great.

**Why exhaustive search is cheap.** For feature $j$, only thresholds between consecutive *distinct sorted values* change the partition, so there are at most $n-1$ candidates per feature: test the midpoints. Searching all features and thresholds gives the best single split at the node. Finding the best *whole tree* is combinatorially infeasible (Bishop, p.664-665), so we are greedy: take the best split now, never revisit it, recurse on both children. Greedy can miss a poor first split that enables two brilliant ones (XOR is the classic case), but it works well enough in practice.

**Regression leaves.** If the partition is fixed and we minimise $\sum_n (t_n-y_{\tau(n)})^2$, setting the derivative with respect to each leaf's constant $y_\tau$ to zero gives $y_\tau=\frac1{N_\tau}\sum_{x_n\in R_\tau}t_n$, the **leaf mean** (Bishop Eq. 14.29). So for regression a split is scored by the total squared error after replacing each side with its mean; we choose the split that reduces it most.

**Stopping and pruning.** Growing until every leaf is pure gives zero training error and poor generalisation. Bishop observes that stopping as soon as the best split gives little gain is short-sighted, because a useless-looking split can enable a very useful one below it. So the common recipe is: grow a big tree $T_0$ (limited only by a minimum leaf size), then prune it back by minimising the **cost-complexity** criterion (Eq. 14.31)

$$
C(T)=\sum_{\tau=1}^{|T|}Q_\tau(T)+\lambda|T|,
$$

where $Q_\tau$ is the leaf's error (squared error, Gini or entropy), $|T|$ is the number of leaves and $\lambda\ge0$ is chosen by cross-validation. It is the same structure as ridge regression: data fit plus a complexity price. Simpler (pre-pruning) controls such as `max_depth`, `min_samples_leaf` do a similar job more crudely and faster.

## From one tree to a forest

**Why averaging helps.** Suppose $M$ models each predict $y_m(x)=h(x)+\epsilon_m(x)$, true function plus error. If the errors have zero mean and are *uncorrelated*, the error of the average is $1/M$ of the average individual error (Bishop Eq. 14.14). Real errors are correlated because all models see the same data. Let each have variance $\sigma^2$ and pairwise correlation $\rho$. The variance of their mean is

$$
\operatorname{Var}\Big(\frac1M\sum_m \epsilon_m\Big)=\frac{1}{M^2}\Big(M\sigma^2+M(M-1)\rho\sigma^2\Big)=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2 .
$$

Adding trees kills the second term but never the first. The lever is therefore to make trees **less correlated**, not just more numerous.

**Bagging** (bootstrap aggregation, Bishop Eq. 14.7): draw $M$ bootstrap samples (size $N$, with replacement) from the training set, fit one tree on each, average (regression) or vote (classification). Deep trees have low bias and high variance, so averaging them is the right trade.

**Random forest**: bagging plus a second randomisation. At *each split* only a random subset of $m$ features is eligible (a common default is about $\sqrt d$ for classification, roughly $d/3$ for regression). Without this, one dominant feature sits at the root of every tree and the trees are strongly correlated (large $\rho$). Restricting the choice decorrelates them, which lowers $\rho\sigma^2$ at the cost of making each tree slightly worse (a bit more bias, a bit more variance in each tree).

**Out-of-bag (OOB) error.** A bootstrap sample of size $N$ misses a given example with probability $(1-\tfrac1N)^N\to e^{-1}\approx0.368$, so each tree sees only about $63.2\%$ of the distinct examples. For example $i$, average the predictions of only the trees that did *not* train on it; the error of these predictions is an honest estimate of generalisation error with no separate validation set. It is cross-validation that comes free with bagging.

## Worked example

Ten points on a line, $x=1,\dots,10$, labels $y=(0,0,1,0,0,1,1,1,1,1)$ (mostly `0` for small $x$ and `1` for large $x$, with one stray `1` at $x=3$).

- Parent: 4 zeros and 6 ones, $G=1-(0.4^2+0.6^2)=0.48$, $H=0.971$ bits.
- Threshold $5.5$: left $=\{0,0,1,0,0\}$ has $G=1-(0.8^2+0.2^2)=0.32$; right $=\{1,1,1,1,1\}$ is pure. Gain $=0.48-\tfrac{5}{10}0.32-\tfrac{5}{10}0=0.32$. In entropy terms the gain is $0.610$ bits.
- Threshold $2.5$ (isolating the first two zeros): gain $0.18$ (Gini). Every other threshold is worse, so the greedy choice is $\theta=5.5$.
- Bootstrap: for $N=10$ the chance an example is missed is $0.9^{10}=0.349$, so the in-bag fraction is $0.651$; for $N=1000$ it is $0.632$, matching $1-e^{-1}$.

The exercise reproduces these numbers.

## Implementation

```python
import numpy as np

def gini(y):
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)

def best_split(X, y):
    n, d = X.shape
    best = (None, None, 0.0)                       # feature, threshold, gain
    parent = gini(y)
    for j in range(d):
        xs = np.unique(X[:, j])
        for t in (xs[:-1] + xs[1:]) / 2:           # midpoints
            L = X[:, j] <= t
            gain = parent - (L.sum() * gini(y[L]) + (~L).sum() * gini(y[~L])) / n
            if gain > best[2] + 1e-12:
                best = (j, t, gain)
    return best
```

A full tree calls `best_split`, stops on a depth or leaf-size limit or a pure node, and otherwise recurses on the left and right subsets. A forest is a loop: `idx = rng.integers(0, n, n)`; fit a tree on `X[idx]`; remember which rows were left out for the OOB estimate.

## Shapes

`X: (N, d)`, `y: (N,)` integer classes. A candidate split yields boolean mask `L: (N,)`. A bagged ensemble of $B$ trees holds `B` index arrays of shape `(N,)`; the OOB prediction for all rows is `(N,)`, each averaged over roughly $0.368B$ trees.

## Common misconceptions

- **"Trees need no tuning."** An unpruned tree overfits badly; depth and leaf size are the main knobs.
- **"More trees means overfitting."** For a forest, more trees only reduces the variance term $\frac{1-\rho}{M}\sigma^2$; accuracy plateaus. (Depth of the individual trees is what controls overfitting.)
- **"Feature importance from impurity is the truth."** Mean decrease in impurity favours features with many distinct values and is computed on training data; permutation importance on held-out data is more trustworthy but slower (repo lesson 04).
- **"Trees extrapolate."** A regression tree predicts a constant inside each box, so beyond the training range it just repeats the edge value.
- **Axis-aligned limits.** A diagonal class boundary needs many staircase splits (Bishop p.666), where a linear model needs one.
- **Interpretability caveat.** A single small tree is readable, but Bishop notes the learned structure is very sensitive to small data changes; a forest of 500 trees is not readable.

## Key takeaways

- A tree greedily picks the (feature, threshold) with the largest weighted impurity decrease; leaves predict the majority class or the mean.
- Gini and entropy behave similarly; regression uses squared error and leaf means.
- Grow big, then prune with $\sum Q_\tau+\lambda|T|$, or limit depth and leaf size.
- Averaging $M$ models with variance $\sigma^2$ and correlation $\rho$ leaves variance $\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$: decorrelate the trees.
- OOB error is a free validation estimate because about 36.8 percent of rows are left out of each bootstrap.

## Explain it yourself

1. Why does the impurity decrease weight the children by $n_L/n$ and $n_R/n$?
2. A colleague says "more trees in a forest will start to overfit". Use the variance formula to answer.

## Further reading

- Bishop, *PRML*, Sec. 14.2 and 14.4 (PDF p.675-686).
- Repo lessons `phases/02-ml-fundamentals/04-decision-trees` and `11-ensemble-methods`.
- Next: [[cml.07]] boosting uses trees in the opposite way, as high-bias learners corrected one after another.

## Mastery criteria

You can compute Gini and entropy by hand, find the best threshold for a toy dataset, write a recursive tree with depth and leaf-size limits, explain bagging via the $\rho\sigma^2$ formula, and compute an OOB error from index sets.
