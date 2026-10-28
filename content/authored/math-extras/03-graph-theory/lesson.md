---
id: mx.03
title: Graph theory for machine learning
summary: Turn networks into matrices, read connectivity off the Laplacian's eigenvalues, cut a graph with one eigenvector, and derive PageRank as the dominant eigenvector of a random walk.
minutes: 75
teaches: [mx.graph, mx.laplacian, mx.pagerank]
requires: [la.matrix, la.eigen, prob.basics]
sources:
  - "ai-engineering-from-scratch/phases/01-math-foundations/21-graph-theory/docs/en.md (graphs, adjacency matrix, degree, BFS/DFS, Laplacian, spectral properties, message passing, spectral clustering build steps). READ."
  - "ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md (Markov chain and transition-matrix section, read for the random-walk view of PageRank). READ."
---

## Where this fits

Rows in a table are independent; but molecules, road maps, citation networks, social networks and knowledge bases are defined by *connections*. Graph neural networks, recommender systems, search ranking and clustering of non-vector data all start from the same few objects: the **adjacency matrix**, the **degree matrix** and the **Laplacian**. The remarkable part is that the eigenvectors you met in [[math.07]] tell you about the shape of the graph: how many pieces it has, where to cut it, and which nodes matter.

This lesson builds on matrices ([[math.07]] for eigenvalues, symmetric matrices have real eigenvalues and orthogonal eigenvectors) and on basic probability. It leads to [[mx.04]], where the random walk on a graph becomes a general Markov chain.

## The idea in plain words

- A graph is dots and lines: people and friendships, cities and roads, web pages and links. A matrix can record who is connected to whom, and then *multiplying* matrices answers questions about walking around the graph (two steps, three steps...).
- Imagine each node holding a number, like a temperature. Heat flows along the edges until neighbours agree. The **Laplacian** is the machine that measures how much the numbers disagree across edges. If the graph is in two disconnected pieces, each piece can have its own constant temperature forever: that is a "flat" pattern the Laplacian cannot see.
- If the graph is *almost* in two pieces joined by a thin bridge, there is a pattern that is almost flat on each side with a jump in the middle. It is cheap in disagreement, and it reveals the two communities. The sign of that pattern tells you which side each node is on. This is **spectral clustering**.
- Now picture a surfer who clicks a random link on each page, forever. Pages that are linked from many important pages get visited more. The long-run fraction of time spent on each page is its **PageRank**, and it is an eigenvector of the surfing matrix.

## Definitions

A graph $G=(V,E)$ has $n$ nodes and edges between pairs of them. Undirected edges work both ways; directed edges (links, follows) do not. Number the nodes $0,\dots,n-1$.

- **Adjacency matrix** $A\in\{0,1\}^{n\times n}$ with $A_{ij}=1$ if there is an edge $i\to j$ (shape $n\times n$). Undirected graph: $A=A^\top$.
- **Degree** of node $i$: $d_i=\sum_jA_{ij}$. **Degree matrix** $D=\operatorname{diag}(d_0,\dots,d_{n-1})$.
- A **walk** of length $k$ from $i$ to $j$ is a sequence of $k$ edges. A **path** is a walk without repeated nodes. Two nodes are in the same **connected component** if some path joins them.

**Counting walks.** $(A^k)_{ij}$ equals the number of walks of length $k$ from $i$ to $j$. Proof by induction: $(A^{k+1})_{ij}=\sum_m(A^k)_{im}A_{mj}$, i.e. walks of length $k$ to some $m$, then one more edge to $j$. In particular $\operatorname{trace}(A^3)=6\times$(number of triangles) in an undirected simple graph, since each triangle is counted from 3 starting nodes in 2 directions.

## The graph Laplacian

$$
L=D-A\qquad(n\times n,\ \text{symmetric}).
$$

For a vector $x\in\mathbb{R}^n$ (one number per node) compute the quadratic form:

$$
x^\top Lx=x^\top Dx-x^\top Ax=\sum_id_ix_i^2-\sum_{i,j}A_{ij}x_ix_j .
$$

Because $d_i=\sum_jA_{ij}$, we can write $\sum_id_ix_i^2=\frac12\sum_{i,j}A_{ij}(x_i^2+x_j^2)$. Therefore

$$
\boxed{x^\top Lx=\frac12\sum_{i,j}A_{ij}(x_i-x_j)^2=\sum_{\{i,j\}\in E}(x_i-x_j)^2\ \ge 0 .}
$$

(the last form sums over each undirected edge once.) Consequences:

1. $L$ is **positive semi-definite**: all eigenvalues are $\ge0$.
2. $L\mathbf 1=0$ (the constant vector): all rows of $L$ sum to zero. So $0$ is always an eigenvalue.
3. $x^\top Lx=0$ exactly when $x_i=x_j$ across every edge, i.e. $x$ is constant on each connected component. Therefore **the multiplicity of the eigenvalue 0 equals the number of connected components**, and the eigenspace is spanned by the indicator vectors of the components.

### Spectral bisection

Cut the nodes into two groups with labels $s_i=\pm1$. A cut edge has $(s_i-s_j)^2=4$ and an uncut one has $0$, so by the boxed formula $s^\top Ls=4\cdot\text{cut}$, that is $\text{cut}=\tfrac14s^\top Ls$ (the number of edges between the groups). Minimising this over all $\pm1$ vectors is a hard combinatorial problem, and putting everything in one group trivially gives zero cut. The **relaxation** drops the $\pm1$ requirement: minimise $x^\top Lx$ subject to $\|x\|=1$ and $x\perp\mathbf 1$ (to exclude the trivial flat solution). By the Rayleigh-quotient argument from [[math.07]], the minimiser is the eigenvector of the **second smallest eigenvalue** $\lambda_2$, the **Fiedler vector**, and the value achieved is $\lambda_2$. Round it by sign: $s_i=\operatorname{sign}(x_i)$.

$\lambda_2>0$ iff the graph is connected, and a small $\lambda_2$ means the graph nearly splits (a bottleneck). Use more eigenvectors, then run $k$-means ([[cml.04]]: $k$-means applied to the rows of the eigenvector matrix) to get $k$ clusters. This is the heart of **spectral clustering**, used when clusters are connected shapes rather than round blobs, and of normalised variants where $L$ is rescaled by $D^{-1/2}$.

A side remark connecting to [[mx.02]]: on a ring of $n$ nodes the Laplacian is a circulant matrix and its eigenvectors are exactly the discrete Fourier waves. Laplacian eigenvectors are "Fourier analysis on a graph".

## Random walks and PageRank

A **random walk** on a directed graph repeatedly moves from the current node to one of its out-neighbours chosen uniformly. The transition matrix is the row-normalised adjacency matrix

$$
P=D_{\text{out}}^{-1}A,\qquad P_{ij}=\frac{A_{ij}}{d_i^{\text{out}}},\qquad\text{each row sums to }1 .
$$

If $r^{(t)}\in\mathbb{R}^n$ is the (row) vector of probabilities of being at each node at time $t$, then $r^{(t+1)}=r^{(t)}P$ (shapes $1\times n$ times $n\times n$). A distribution that does not change, $r=rP$, is a left eigenvector of $P$ with eigenvalue 1; we meet this properly in [[mx.04]].

Two problems for the web: a page without out-links (a **dangling** node) is a dead end, and the graph may have groups the walker can enter but never leave. PageRank fixes both with **teleportation**: with probability $d$ (the damping factor, commonly $0.85$) follow a link, with probability $1-d$ jump to a uniformly random page. The resulting **Google matrix** is

$$
G=dP+\frac{1-d}{n}\mathbf 1\mathbf 1^\top,\qquad\text{(dangling rows of }P\text{ replaced by }\tfrac1n\mathbf 1^\top).
$$

All entries of $G$ are strictly positive and rows sum to 1. By the Perron-Frobenius theorem (used here without proof) such a matrix has eigenvalue 1 which is simple, all other eigenvalues have modulus $<1$ (in fact $\le d$), and the left eigenvector for eigenvalue 1 can be chosen positive. **PageRank is that eigenvector**, normalised to sum to 1:

$$
r=rG,\qquad\sum_ir_i=1 .
$$

Equivalently, since $r\mathbf 1\mathbf 1^\top=\mathbf 1^\top$ when $r$ sums to 1,

$$
r=d\,rP+\frac{1-d}{n}\mathbf 1^\top .
$$

**Power iteration** computes it: start from the uniform vector and apply this update repeatedly. The error shrinks at least like $d^t$, so $0.85^{50}\approx3\times10^{-4}$. With $n$ in the billions the matrix $P$ is stored sparsely and one iteration is a sparse matrix-vector product.

## Worked example

All numbers are produced by the exercise code.

**Two triangles joined by a bridge.** Nodes $0,1,2$ form one triangle, $3,4,5$ another, and the edge $(2,3)$ joins them. Degrees: $(2,2,3,3,2,2)$.

- Laplacian eigenvalues: $0,\ 0.4384,\ 3,\ 3,\ 3,\ 4.5616$. Exactly one zero: connected. $\lambda_2=0.4384$ is small compared with the others: a bottleneck.
- Fiedler vector $\approx(-0.465,-0.465,-0.261,\ 0.261,\ 0.465,\ 0.465)$. The signs split $\{0,1,2\}$ from $\{3,4,5\}$, exactly the two triangles. The bridge endpoints have the smallest magnitudes.
- Remove the bridge: the eigenvalues become $0,0,3,3,3,3$. Two zeros = two components.
- Walks: $(A^2)_{03}=1$ (only via node 2) and $\operatorname{trace}(A^3)=12=6\times2$ triangles.
- Quadratic form check: $x=(1,2,0,3,-1,5)$ gives $x^\top Lx=\sum_{\text{edges}}(x_i-x_j)^2$.

**PageRank** on 4 pages with links $0\to1,\ 0\to2,\ 1\to2,\ 2\to0,\ 2\to3,\ 3\to2$ and $d=0.85$: 

$$
P=\begin{pmatrix}0&.5&.5&0\\0&0&1&0\\.5&0&0&.5\\0&0&1&0\end{pmatrix}.
$$

Power iteration converges to $r=(0.2199,\ 0.1310,\ 0.4292,\ 0.2199)$, summing to 1. Page 2 wins: three pages link to it. Page 0 and page 3 tie at $0.2199$. The same vector comes out of the eigenvector of $G^\top$ for eigenvalue 1 (computed with `np.linalg.eig`): the eigenvalue is $1.0$ up to rounding.

## Implementation

```python
import numpy as np

def laplacian(A):
    return np.diag(A.sum(axis=1)) - A

def spectral_bisection(A):
    w, V = np.linalg.eigh(laplacian(A))      # ascending eigenvalues
    lab = (V[:, 1] > 0).astype(int)           # sign of the Fiedler vector
    return 1 - lab if lab[0] == 1 else lab

def pagerank(M, d=0.85, tol=1e-12):
    n = M.shape[0]
    out = M.sum(axis=1, keepdims=True)
    P = np.where(out > 0, M / np.where(out > 0, out, 1), 1.0 / n)  # dangling -> uniform
    r = np.ones(n) / n
    while True:
        r_new = d * (r @ P) + (1 - d) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
```

Production: `scipy.sparse` and `scipy.sparse.linalg.eigsh` for large sparse Laplacians (compute only a few eigenvectors), `networkx.pagerank`, `sklearn.cluster.SpectralClustering`; GNN libraries (PyTorch Geometric, DGL) store the adjacency as an edge list, not a dense matrix.

## Shapes

- $A,D,L,P,G$: $(n,n)$. Node features $H$: $(n,f)$. One round of GNN-style message passing is $\hat A H$ with $\hat A=D^{-1}(A+I)$ or a symmetric normalisation: $(n,n)\times(n,f)\to(n,f)$, every node averaging itself and its neighbours.
- Rank vector: row $(1,n)$ when multiplied on the left of $P$; as a column you would write $P^\top r$.
- Laplacian eigenvectors: columns of $V\in\mathbb{R}^{n\times n}$ from `eigh`, ordered by ascending eigenvalue.

## Common misconceptions

- "Eigenvalue 0 appears once." It appears once *per connected component*.
- "The Fiedler vector gives the minimum cut." It gives a relaxed solution that is rounded; it can be off, especially for unbalanced groups. Normalised variants work better on graphs with very different degrees.
- "PageRank needs the graph to be connected." Teleportation removes that requirement.
- "$\sum_j P_{ij}=1$ means $P$ is symmetric." Rows sum to 1; columns need not.
- "The Laplacian eigenvectors with large eigenvalues are noise." They are the high-frequency patterns, alternating across edges; the small ones are the smooth ones. GNNs that smooth features act as a low-pass filter on this spectrum.

## Key takeaways

- $A$ stores edges, $D$ degrees, $L=D-A$; $(A^k)_{ij}$ counts walks.
- $x^\top Lx=\sum_{\text{edges}}(x_i-x_j)^2$: $L$ measures roughness over the graph and is positive semi-definite.
- The number of zero eigenvalues is the number of components; the Fiedler vector's sign splits a graph at its thinnest bottleneck.
- PageRank is the stationary distribution of a damped random walk, an eigenvector of the Google matrix, found by power iteration.

## Explain it yourself

1. Why does the quadratic form $x^\top Lx$ vanish exactly on vectors that are constant on every component?
2. Why is teleportation needed in PageRank, and what does the damping factor $d$ control?

## Further reading

- Repo lesson `phases/01-math-foundations/21-graph-theory` (BFS/DFS, message passing).
- Next: [[mx.04]] for the general theory of Markov chains behind PageRank.

## Mastery criteria

- [ ] I can build $A$, $D$, $L$ for a small graph and verify $x^\top Lx$ by hand.
- [ ] I can explain why the multiplicity of eigenvalue 0 counts components.
- [ ] I can run spectral bisection and PageRank from scratch.
- [ ] I can state what each of the damping and dangling-node fixes repairs.
