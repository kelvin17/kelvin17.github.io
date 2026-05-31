# Advanced Algorithms — Notes

> These algorithms are used in Modern Distributed System. Most of them are fundation of some middlewares.

---

## Table of Contents

1. [Bloom Filter](#1-bloom-filter)
2. [Distance Oracles](#2-distance-oracles)
3. [Approximate Nearest Neighbour & LSH](#3-approximate-nearest-neighbour--lsh)
4. [Distributed Algorithms — Breaking Symmetry (Coloring)](#4-distributed-algorithms--breaking-symmetry-coloring)
5. [Distributed Graph Algorithms — SSSP, BFS, Leader Election, APSP](#5-distributed-graph-algorithms--sssp-bfs-leader-election-apsp)
6. [Monte Carlo vs Las Vegas, and Coloring](#6-monte-carlo-vs-las-vegas-and-1-coloring)
7. [Streaming — Frequent Elements & Reservoir Sampling](#7-streaming--frequent-elements--reservoir-sampling)
8. [Streaming — Distinct Elements (AMS/Tidemark) & Counting (Morris)](#8-streaming--distinct-elements-amstidemark--counting-morris)
9. [Sketches — Count-Min & Range Queries](#9-sketches--count-min--range-queries)
10. [Graph Streaming — Connectivity, Bipartiteness, Spanner](#10-graph-streaming--connectivity-bipartiteness-spanner)
11. [Massively Parallel Computation (MPC) — Sorting & MST](#11-massively-parallel-computation-mpc--sorting--mst)
12. [Dynamic Graph Connectivity](#12-dynamic-graph-connectivity)

---

## 1. Bloom Filter

**Problem.** Maintain a set $X = \{x_1, \dots, x_n\} \subseteq U$ supporting two operations:

1. **Insert** $x$ into $X$.
2. **Query** whether $x \in X$.

**Structure.** One $m$-bit array $M$ and $k$ hash functions $h_i : U \to \{1, \dots, m\}$.

- **Insert($x$):** apply each hash, set $M[h_i(x)] = 1$ for all $i$.
- **Query($x$):** apply all $k$ hashes; if **all** $M[h_i(x)] = 1$ return *yes*, otherwise *no*.

**Key property.** A *no* answer is always correct (no false negatives). A *yes* may be a **false positive** — we want to bound its probability.

### False-positive probability

Let $p'$ be the probability that a *specific* bit is still $0$ after all insertions:

$$
p' = \left(1 - \frac{1}{m}\right)^{kn}
$$

Let $\rho = \mathbb{E}[\text{fraction of zeros}] \doteq p'$. Using $1 + x \approx e^{x}$ as $x \to 0$ (set $x = -\tfrac{1}{m}$):

$$
p' = \left(1 - \tfrac{1}{m}\right)^{kn} \approx e^{-kn/m}
$$

The false-positive rate is

$$
\text{FP} = (1 - p')^{k} \approx \left(1 - e^{-kn/m}\right)^{k}
$$

### Optimal number of hash functions

Write $f(k) = \exp\!\Big(k \cdot \ln\big(1 - e^{-kn/m}\big)\Big)$ and minimize over $k$. (Uses the identities $a^{k} = \exp(k \ln a)$ and $1 + x \approx e^x$.)

$$
\boxed{k_{\min} = \frac{m}{n}\ln 2}
$$

Substituting $k_{\min}$ gives a false-positive rate of

$$
\text{FP} = \left(\tfrac{1}{2}\right)^{k_{\min}} = \left(\tfrac{1}{2}\right)^{(m/n)\ln 2}
$$

### Space requirement

To achieve target FP rate $\varepsilon = (\tfrac{1}{2})^{k_{\min}}$, solve for $m$ (using $\ln 2 = 1/\log_2 e$):

$$
m = n \cdot \log_2\!\left(\tfrac{1}{\varepsilon}\right) \cdot \log_2 e
$$

### Optimality

The information-theoretic lower bound is $m \ge n \cdot \log_2\frac{1}{\varepsilon}$. The Bloom filter uses

$$
m = n \cdot \log_2\tfrac{1}{\varepsilon} \cdot \log_2 e,
$$

so it is within a factor of $\log_2 e \approx 1.44$ of optimal.

### Drawbacks

1. The analysis relies on hash functions with strong independence guarantees. In practice weaker hashes also work well.
2. $m$ is large — space $\approx n \cdot \log_2\frac{1}{\varepsilon} \cdot \log_2 e$.

---

## 2. Distance Oracles

**Problem.** Edge-weighted graph $G = (V, E)$. Let $\delta(s,t)$ be the shortest-path distance. Storing all exact distances costs $\Theta(n^2)$ space. We want **approximate** shortest-path distances in **smaller space** (trading accuracy for space).

A **$t$-approximate** oracle (stretch $t$) returns $\hat\delta$ with

$$
\delta(u,v) \;\le\; \hat\delta(u,v) \;\le\; t \cdot \delta(u,v)
$$

### Simple oracle: stretch 3, space $O(n^{3/2})$, time $O(1)$

1. **Sample** each vertex independently with probability $p$ → sampled set $S$.
2. For each vertex $u$ store:
   - **Set 1:** all vertices closer to $u$ than $u$'s nearest sampled vertex (with their distances).
   - **Set 2:** distances to **all** sampled vertices.
   - The nearest sampled vertex of $u$.

**Query($u,v$):**

- **Case 1 — $v$ is closer than the nearest sample:** look up directly in $u$'s Set 1.
- **Case 2 — $v$ is far:** route through $u$'s nearest sampled vertex. Get $\delta(u, \text{nearest sample})$, then look up Set 2 of $v$ to get the distance from $v$ to that sampled vertex.

The triple $v, s, u$ forms a triangle, giving **stretch 3**.

**Space analysis.**

- Set 2: $\mathbb{E}[|S|] = np$, stored for all $v$ ⟹ $\mathbb{E}[|V|\cdot|S|] = n^2 p$.
- Set 1: let $L$ be the rank of the nearest sampled vertex; $\mathbb{E}[L] = \tfrac{1}{p}$, so per vertex $O(1/p)$ ⟹ $O(n/p)$ total.

$$
\text{Total} = O\!\left(n^2 p + \tfrac{n}{p}\right)
\;\;\xrightarrow{\;n^2 p = n/p\;}\;\; p = \tfrac{1}{\sqrt n}
\;\;\Rightarrow\;\; O(n^{3/2})
$$

### Generalization — Thorup–Zwick

For any integer $k \in \mathbb{N}$:

$$
\boxed{O\!\left(k \cdot n^{1+1/k}\right) \text{ space}, \quad O(k)\text{ query time}, \quad \text{stretch } 2k-1}
$$

This trades off space against stretch; $k = 2$ is the special case above.

**Construction.** Nested sampled subsets $A_0 \supseteq A_1 \supseteq \dots \supseteq A_k$:

$$
A_0 = V, \qquad A_i \text{ samples from } A_{i-1} \text{ w.p. } p, \qquad A_k = \varnothing
$$

For each $u \in V$ store:
- a hash table $B(u)$ with $\delta(u,v)$ for all $v \in B(u)$;
- pivots $p_i(u)$ for $i = 0, \dots, k-1$ — the closest sampled vertex of $u$ in $A_i$, with $\delta(u, A_i) = \delta(u, p_i(u))$.

**Query($u,v$):**

```
w ← u;  i ← 0
while w ∉ B(v):
    i ← i + 1
    swap(u, v)
    w ← p_i(u)
return δ(w, v) + δ(w, u)
```

**Space.** $\mathbb{E}[|B(u)|] = O\!\big(\tfrac{k}{p} + n\,p^{\,k-1}\big)$. Minimize by balancing $\tfrac{k}{p} = n\,p^{\,k-1}$ (ignoring the $k$ factor), giving $\tfrac{1}{p} = n^{-1/k} \Rightarrow p = n^{-1/k}$, so $\mathbb{E}[|B(u)|] = O(k \cdot n^{1/k})$ and

$$
n \cdot \mathbb{E}[|B(u)|] = O\!\left(k \cdot n^{1 + 1/k}\right)
$$

---

## 3. Approximate Nearest Neighbour & LSH

**Problem.** Don't find the exact point — just find an **approximate** nearest neighbour.

**$c$-approximate $r$-near neighbour:** return a point $y$ with $d(x,y) \le c \cdot \min_{z} d(x,z)$. More precisely, if there exists $z$ with $d(x,z) \le r$, return a $y$ with $d(x,y) \le c \cdot r$. (Randomized version: return such a $y$ with constant probability.)

### Locality-Sensitive Hashing (LSH)

A hash family $H$ is **$(r, cr, p_1, p_2)$-sensitive** (with $p_1 > p_2$ and $c > 1$) if:

$$
d(x,y) \le r \;\Rightarrow\; \Pr[h(x) = h(y)] \ge p_1
$$
$$
d(x,y) \ge cr \;\Rightarrow\; \Pr[h(x) = h(y)] \le p_2
$$

### LSH for Hamming distance

Hamming distance: $d(x,y) = |\{ i : x_i \ne y_i \}|$.

Hash function: $h(x) = x_i$ — pick a random coordinate $i$. Then for $d$-dimensional strings,

$$
\Pr[h(x) = h(y)] = 1 - \frac{d(x,y)}{d}
$$

- **Insert($x$):** insert $x$ into list $L[h(x)]$.
- **Search (near-neighbour of $x$):**
  1. $h(x)$ → get bucket.
  2. Traverse items in the bucket, compute distance to each.
  3. If some item has distance $\le cr$, return it; else **FAIL**.

**Amplification (Solution 2).** Concatenate $k$ hashes:

$$
g(x) = h_1(x)\,h_2(x)\cdots h_k(x)
$$

to **decrease the wrong (false-positive) probability** to $\big(1 - \tfrac{cr}{d}\big)^{k}$. Build a hash table keyed by $h_T(g(x))$. To **increase the probability of a correct find**, use **multiple ($L$) tables**.

### Distance metrics → LSH families (summary)

| Data type | Distance | LSH technique |
|---|---|---|
| String | Hamming | random bit / coordinate |
| Set | Jaccard | MinHash |
| Vectors | Angular | SimHash (random projection) |

- **Jaccard:** $J_{\text{sim}}(A,B) = \dfrac{|A \cap B|}{|A \cup B|}$, distance $= 1 - J_{\text{sim}}(A,B)$.
  - **MinHash:** $h(A) = \min_{x \in A} h(x)$ — hash all elements of the set, take the minimum.
- **Vectors:** angular distance (angle between vectors) → **SimHash** via random projection.

### Bounding the number of hashes $k$ (for strings)

Let $F = \{ y : d(x,y) \ge cr \}$ be the "far" points. We want the collision probability bounded:

$$
\Pr[g(x) = g(y)] \le \tfrac{1}{n}
\;\;\Rightarrow\;\; p_2^{\,k} = \tfrac{1}{n}
\;\;\Rightarrow\;\; k = \frac{\log n}{\log \tfrac{1}{p_2}}
$$

Let $X_y = \mathbb{1}[\,y \text{ collides with } x\,]$ and $X = \sum_{y \in F} X_y$. Then

$$
\mathbb{E}[X] = \sum_{y \in F} \Pr[X_y = 1] = \sum_{y \in F} \tfrac{1}{n} \le 1
$$

By Markov's inequality, $\Pr[X \ge 6] \le \dfrac{\mathbb{E}[X]}{6} = \tfrac{1}{6}$.

---

## 4. Distributed Algorithms — Breaking Symmetry (Coloring)

**Model.** Two standard models: the **general (LOCAL)** model and the **CONGEST** model.

- Nodes have **unique identifiers** $1, 2, \dots, n^{c}$.
- Nodes exchange messages with neighbours (both sides), **synchronously, in parallel**: each round = *receive → process → send*.
- **Message size:** in CONGEST, at most $O(\log n)$ bits per message per round.

### P3C — 3-coloring a path

A naive increasing/decreasing-order coloring needs $O(n)$ rounds. We use unique identifiers to break symmetry.

```
c ← id            // initialize
repeat:
    1. send msg c to both neighbours
    2. receive msgs from neighbours → form set M
    3. if c ∉ {1,2,3} and c > all msgs received this round:
           c ← min( {1,2,3} \ M )
```

Runs in $O(\log^* n)$ rounds.

### Faster deterministic path coloring (Cole–Vishkin)

Each round reduces the number of colors from $2^{x}$ to $2x$. Repeat until $2x \le 6$, then apply P3C to go from 6 colors → 3 colors.

```
Round, for each node u with color C(u):
    1. send C(u) to its predecessor (so each node learns its successor's color)
    2. C0(u) = C(u)                  // itself
       C1(u) = C(u) from successor
    3. i(u) = index of the first bit where C0(u) differs from C1(u)
       b(u) = bit value of C0(u) at index i(u)
       set C(u) = 2·i(u) + b(u)
```

**Why colors shrink.** If a color uses $x$ bits, then $C(u) = 2 \cdot i(u) + b(u)$, where $i(u) < x$. So the color range shrinks $2^{x} \to 2x$, i.e. $x \to \log x$ per round ⟹ $O(\log^* n)$ rounds.

**Correctness.** Show $C(u) \ne C(v)$ for adjacent $u,v$ after a round. Two cases:
- $i(u) = i(v) = i$ : then $b(u) \ne b(v)$ ⟹ differ. ✓
- $i(u) \ne i(v)$ : differ regardless of the bit values. ✓

**Rounds.** Initially color = id; continue until at most 6 colors — needs $O(\log^* n)$ rounds; then P3C reduces 6 → 3 in 3 rounds.

### Randomized path coloring

Each node has a private source of randomness. Flag $s(u) \in \{0,1\}$: $0$ = unstopped, $1$ = stopped.

```
In each round, for each node u:
    if s(u) == 1: do nothing
    else:
        1. pick color c(u) ∈ {1,2,3} uniformly at random
        2. send c(u) to neighbours
        3. if c(u) differs from all neighbours: set s(u) = 1
```

**Analysis.** A node stops in a round with probability $\ge \tfrac{1}{3}$ (not stopped $\le \tfrac{2}{3}$). After $k$ rounds, a node is still unstopped with probability $\le (\tfrac{2}{3})^{k}$.

Set $k = (c+1)\log_{3/2} n$. Since $\tfrac{2}{3} = (\tfrac{3}{2})^{-1}$,

$$
\left(\tfrac{2}{3}\right)^{k} = \left(\tfrac{2}{3}\right)^{(c+1)\log_{3/2} n} = \left(\tfrac{1}{n}\right)^{c+1}
$$

There are $n$ nodes; by a union bound the probability that **some** node is still unstopped is

$$
n \cdot \frac{1}{n^{\,c+1}} = \frac{1}{n^{c}}.
$$

So all nodes stop within $k = O(\log n)$ rounds with probability $\ge 1 - \tfrac{1}{n^{c}}$ (w.h.p.).

---

## 5. Distributed Graph Algorithms — SSSP, BFS, Leader Election, APSP

**Setting.** CONGEST model. Graph problems:
1. Single-Source Shortest Path (**SSSP**)
2. All-Pairs Shortest Path (**APSP**)

### SSSP — Wave algorithm

Local output at $v$ is its distance to the source $s$: $\text{dist}_G(s,v) = \text{dist}_G(v,s)$. Solved in $O(\operatorname{diam}(G))$ rounds, where $\operatorname{diam}(G) = \max_{u,v} \text{dist}_G(u,v)$.

```
Init: state ← 0 (unstopped)
Round 0:  s sends a wave to all neighbours; dist(s) = 0
Round i>0: if unstopped and received a wave in round i-1:
               set state = i        // = distance
               send wave to all neighbours
```

### BFS tree construction

**Per-node data structure.**
- $d(v)$ : distance to root $s$
- $p(v)$ : parent in the BFS tree $T$
- $C(v)$ : set of children
- $a(v)$ : ack — equals $1$ when the subtree is finished, $0$ otherwise (still constructing / processing)

```
Init: d(v) = ⊥ for v ≠ s,  d(s) = 0;  p(v) = ⊥;  C(v) = ⊥;  a(v) = 0

Each round:
    every node u with d(u) ≠ ⊥ sends d(u) to neighbours
    every node v with d(v) = ⊥ that receives a msg this round:
        set p(v) = u,  d(v) = d(u) + 1
        (send "accept" to u in the next round)
    each node sets C(u) when it receives an accept
```

Total rounds: $O(\operatorname{diam}(G))$.

### Leader election (node with smallest ID)

Output: $s(\text{leader}) = 1$ for the node with the smallest ID, $0$ for all others. Runs in $O(\operatorname{diam}(G))$ rounds, **based on BFS**.

Each node tries to build its own BFS; $d(u)$ carries the source's id.

**Correctness.** $s$ is the smallest id. The BFS rooted at $s$ reaches every node $v$, which sets $l(v) = s$, and the acks return to $s$ so $s$ successfully builds its BFS. For any $s' \ne s$, $s'$ **cannot** build a BFS because nodes will not send accepts to it (they prefer the smaller id).

**Naive issue:** every node forwards its message ⟹ $O(n)$ messages.
**Fix:** each node forwards only the **smallest root id** it has received so far.

### APSP

```
1. Elect leader S.
2. Build a BFS tree rooted at S.
3. Pass a token in DFS order. The first time a node receives the token,
   it starts the wave algorithm using itself as the source (non-interfering waves).
   In the next round, it passes the token downward.
```

CONGEST per round: receive → process → send.

**Total rounds.**
1. Elect leader: $O(\operatorname{diam}(G))$.
2. Build BFS of the leader: $O(\operatorname{diam}(G))$.
3. BFS has $n-1$ edges, so the token moves $2(n-1)$ times — $O(n)$.
4. Each node starts a wave and takes $O(\operatorname{diam}(G))$ rounds to finish.

$$
\text{Total} = O\!\big(n + \operatorname{diam}(G)\big) = O(n)
$$

---

## 6. Monte Carlo vs Las Vegas, and Coloring

| | Monte Carlo | Las Vegas |
|---|---|---|
| **Time** | always stops within $T(n)$ | stops within $T(n)$ with high probability |
| **Correctness** | correct with probability $\ge p$ | always correct |

**With high probability (w.h.p.)** means $p$ is of the form $1 - \tfrac{1}{n^{c}}$ for any chosen constant $c > 0$.

### Refined Δ+1 coloring (general graphs)

**Naive.** Each node picks a color $c(u) \in \{1, \dots, \Delta+1\}$, sends it to all neighbours; if there's a conflict, repeat, else stop. When $\Delta$ is large this is slow.

**Refined.** Each node has its own palette $C(u) = \{1, 2, \dots, \deg(u)+1\} \subseteq \{1,\dots,\Delta+1\}$. Each node maintains a state $s(u) \in \{0,1\}$, with $c(u) \in \{\bot\} \cup C(u)$. At termination $c(u) \ne \bot$.

For node $u$:
```
Round 0: s(u) ← 1,  c(u) ← ⊥
Until u stops, it alternates between state 1 and state 0:
    odd rounds:  s(u) starts as 1, ends as 0
    even rounds: opposite
Stop state: s(u) = 1 and c(u) ≠ ⊥.
After stopping, u keeps sending its color c(u).
```

**Odd round — pick a color:**
```
1. send c(u) to neighbours (assume not stopped, so c(u) = ⊥, state = 1)
2. M(u) = colors received from all neighbours
3. F(u) = C(u) \ M(u);  with probability 1/2, pick c(u) from F(u)
4. set state = 0
```

**Even round — check conflict:**
```
1. send c(u) to all neighbours;  M(u) = received set
2. if c(u) ∈ M(u):  conflict → set c(u) = ⊥ (not stopped)
   else:            can stop → stop
3. set state = 1
```

**Correctness.** A node $u$ stops only when (1) it has a color in an odd round, and (2) there is no conflict in the next even round. Once stopped, $u$ keeps color $c(u)$ and keeps broadcasting it, so its neighbours will never claim the same color. Since $C(u) = \{1, \dots, \deg(u)+1\} \subseteq \{1, \dots, \Delta+1\}$ for all $u \in V$, the result is a valid **$(\Delta+1)$-coloring**.

**Showing $O(\log n)$ rounds w.h.p.** (concentration argument as in randomized path coloring).

---

## 7. Streaming — Frequent Elements & Reservoir Sampling

**Streaming model.** Elements $a_1, a_2, \dots, a_m$ from universe $[n] = \{1, \dots, n\}$ arrive one by one ($a_i$ before $a_{i+1}$). Space is measured in **bits**. Goal: **small space** (sublinear / polylogarithmic).

### Frequent elements — Heavy Hitters (Misra–Gries)

**Problem.** For fixed $k$, find all elements $i$ that occur more than $\tfrac{m}{k}$ times.

```
Stage 1 — initialize:
    use an array/map A (key = element, value = counter); size of A = k - 1
    while stream not empty (next element j):
        if j ∈ keys(A):              A[j] = A[j] + 1
        else if |keys(A)| < k - 1:   A[j] = 1
        else:
            1. decrement every value in A by 1
            2. remove all keys whose value reaches 0

Stage 2 — query frequency of i:
    if i ∈ keys(A):  return f̂(i) = A[i]
    else:            return 0
```

**Space.** $O\!\big(k \cdot (\log m + \log n)\big)$ — $k$ entries, each storing a key ($\log n$) and a counter ($\log m$).

**Guarantee.**

$$
f_i - \frac{m}{k} \;\le\; \hat f_i \;\le\; f_i
$$

- Upper bound $\hat f_i \le f_i$ is trivial.
- Lower bound comes from decrements: a single element can be discarded at most $\tfrac{m}{k}$ times (each decrement step touches $\ge k$ distinct elements at once, so each consumes $\ge k$ arrivals).

### Reservoir sampling

**Goal.** Sample $k$ elements **uniformly** from a stream that is huge and of unknown length.

```
1. put the first k elements into reservoir R = {R_1, ..., R_k}
2. for i > k until the stream ends:
       with probability k/i, decide to replace
       if so, pick a uniform random element of R and replace it with a_i
return R
```

**Claim.** For all $t \ge i$, $\;\Pr[a_i \in R] = \tfrac{k}{t}$.

**Proof.**
- $\Pr[a_i \text{ chosen at time } i] = \tfrac{k}{i}$.
- $\Pr[a_i \text{ replaced at time } j] = \tfrac{k}{j} \cdot \tfrac{1}{k} = \tfrac{1}{j}$, so $\Pr[a_i \text{ not replaced at } j] = 1 - \tfrac{1}{j} = \tfrac{j-1}{j}$.

$$
\Pr[a_i \in R_t] = \frac{k}{i} \cdot \frac{i}{i+1} \cdot \frac{i+1}{i+2} \cdots \frac{t-1}{t} = \frac{k}{t}. \qquad\blacksquare
$$

---

## 8. Streaming — Distinct Elements (AMS/Tidemark) & Counting (Morris)

### P1 — Count distinct elements

**Goal.** Given a stream $\sigma = \langle a_1, \dots, a_m \rangle$, $a_i \in [n]$, estimate the number of **distinct** elements $d$. Output an $(\varepsilon, \delta)$-estimate $\hat d$:

$$
\Pr\!\left[\,\left|\frac{\hat d}{d} - 1\right| > \varepsilon \,\right] \le \delta
$$

**Background — zeros of an integer.** For $p > 0$, $\operatorname{zeros}(p)$ = number of trailing zeros in the binary representation of $p$, i.e. $\operatorname{zeros}(p) = \max\{ i : 2^{i} \mid p \}$.

$$
\operatorname{zeros}(2) = 1 \;(10), \quad \operatorname{zeros}(7) = 0 \;(0111), \quad \operatorname{zeros}(16) = 4 \;(10000)
$$

**AMS / Tidemark algorithm.**
```
Init:    2-universal hash h : [n] → [n];  z ← 0
Process(token j):  z ← max{ z, zeros(h(j)) }
Output:  2^(z + 1/2)
```
where $z = \max_{i \in [m]} \operatorname{zeros}(h(a_i))$. Intuition: with $d$ distinct values, the largest number of trailing zeros is about $\log d$, so $2^{z+1/2}$ tracks $d$.

**Analysis.** Indicator $X_{r,j} = \mathbb{1}[\operatorname{zeros}(h(j)) \ge r]$.

$$
\mathbb{E}[X_{r,j}] = \Pr[\operatorname{zeros}(h(j)) \ge r] = \left(\tfrac{1}{2}\right)^{r} = \frac{1}{2^{r}}
$$

Let $Y_r = \sum_{j : f_j > 0} X_{r,j}$ (sum over **distinct** elements). Then

$$
\mathbb{E}[Y_r] = \sum_{j} \mathbb{E}[X_{r,j}] = \frac{d}{2^{r}}
$$

Note: $Y_r \ge 1 \iff z_{\text{out}} \ge r$ and $Y_r = 0 \iff z_{\text{out}} \le r-1$.

**Overestimate tail $\Pr[\hat d \ge 3d]$.** Let $a$ be the smallest integer with $2^{a+1/2} \ge 3d$. Then

$$
\Pr[\hat d \ge 3d] = \Pr[\,2^{z_{\text{out}} + 1/2} \ge 3d\,] = \Pr[z_{\text{out}} \ge a] = \Pr[Y_a \ge 1]
$$

By Markov, $\Pr[Y_a \ge 1] \le \mathbb{E}[Y_a] = \dfrac{d}{2^{a}}$. Since $2^{a+1/2} \ge 3d \Rightarrow 2^{a} \ge \tfrac{3d}{\sqrt2}$,

$$
\Pr[\hat d \ge 3d] \le \frac{d}{2^{a}} \le \frac{\sqrt2}{3} = \tfrac{1}{3}\sqrt2 \approx 0.471
$$

**Underestimate tail $\Pr[\hat d \le \tfrac{d}{3}]$.** Let $b$ be the largest integer with $2^{b+1/2} \le \tfrac{d}{3}$. Then

$$
\Pr[\hat d \le \tfrac{d}{3}] = \Pr[z_{\text{out}} \le b] = \Pr[Y_{b+1} = 0]
$$

Using the second moment ($\operatorname{Var}[Y_r] \le \mathbb{E}[Y_r]$ for sums of indicators) / Chebyshev:

$$
\Pr[Y_{b+1} = 0] \le \frac{1}{\mathbb{E}[Y_{b+1}]} = \frac{2^{b+1}}{d}
\;\le\; \frac{2^{b+1}}{3 \cdot 2^{b+1/2}} = \tfrac{1}{3}\sqrt2
$$

Both error probabilities are $\le \tfrac{1}{3}\sqrt2$. This is then amplified (median of $O(\log\frac1\delta)$ independent estimators) to obtain an $(\varepsilon,\delta)$ guarantee.

> **Variance bound used above.** $\operatorname{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2 \le \mathbb{E}[X^2]$. Since $X_{r,j}^2 = X_{r,j}$ (indicator), $\operatorname{Var}[Y_r] \le \sum_j \mathbb{E}[X_{r,j}] = \tfrac{d}{2^r} = \mathbb{E}[Y_r]$.

### P2 — Estimate the stream length (counting)

**Goal.** Count the length $n$ of the stream seen so far ($n \le m$). Trivial counter uses $O(\log m)$; goal is $O(\log\log m)$, with an $(\varepsilon, \delta)$-estimate that is an **unbiased estimator** of $m$.

**Morris counter.**
```
Init:    x ← 0
Process(token):  with probability 2^(-x), update x ← x + 1
Output:  2^x − 1
```

Equivalent form (more space, clearer), let $c = 2^x$:
```
Init:    c ← 1
Process(token):  with probability 1/c, update c ← 2c
Output:  c − 1
```

**Unbiasedness.** Let $C_i$ be the value of $c$ after processing $i$ tokens, $C_0 = 1$. Want $\mathbb{E}[C_n - 1] = n$.

Let $Z_i$ be the indicator $Z_i = 1$ if $C_{i+1} = 2C_i$, else $0$. So $C_{i+1} = (1 + Z_i) C_i$ and $\mathbb{E}[Z_i \mid C_i] = \Pr[Z_i = 1 \mid C_i] = \tfrac{1}{C_i}$ (from the algorithm).

By the law of total expectation ($\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$ with $X = C_{i+1}, Y = C_i$):

$$
\begin{aligned}
\mathbb{E}[C_{i+1}] &= \mathbb{E}\big[\mathbb{E}[C_{i+1} \mid C_i]\big]
= \mathbb{E}\big[\mathbb{E}[(1 + Z_i) C_i \mid C_i]\big] \\
&= \mathbb{E}\Big[ C_i \big(1 + \mathbb{E}[Z_i \mid C_i]\big) \Big]
= \mathbb{E}\Big[ C_i \big(1 + \tfrac{1}{C_i}\big) \Big] \\
&= \mathbb{E}[C_i + 1] = \mathbb{E}[C_i] + 1
\end{aligned}
$$

Hence $\mathbb{E}[C_n] = n + 1$, so $\mathbb{E}[C_n - 1] = n$. ✓

To finish the analysis one also shows $\operatorname{Var}[C_n] = \operatorname{Var}[C_n - 1]$ is small (high concentration).

---

## 9. Sketches — Count-Min & Range Queries

**Applications.**
1. Count frequency → **Count-Min** / **Count Sketch**.
2. **Range$(a,b)$** — estimate the number of elements between $a$ and $b$.

### Count-Min Sketch

Use $d$ hash functions $h_j : [n] \to [w]$. The sketch is a $d \times w$ table of counters $C$.

```
Insert(x):  for each row j ∈ [d]:  C_j[ h_j(x) ] += 1
Query(x):   f̂(x) = min_{j ∈ [d]}  C_j( h_j(x) )
```

**Guarantee.** $\hat f(i) \ge f(i)$ always; and with probability $\ge 1 - (\tfrac{1}{2})^{d}$,

$$
\hat f(i) \le f(i) + \frac{2}{w}\, m
$$

Set $w = \tfrac{2}{\varepsilon}$ and $d = \lg\tfrac{1}{\delta}$.

**Analysis.**
- **Lower bound** $\hat f_i \ge f_i$ holds because counters only add.
- For a fixed $i$ and row $j$, let $Z_j = C_j(h_j(i))$ and $b = h_j(i)$:

$$
\mathbb{E}[Z_j] = \mathbb{E}\!\Big[ \textstyle\sum_{s : h_j(s) = b} f_s \Big]
= f_i + \frac{1}{w} \sum_{s \ne i} f_s
\;\le\; f_i + \frac{m}{w}
$$

- By Markov (since $Z_j - f_i \ge 0$):

$$
\Pr\!\Big[ Z_j \ge f_i + \tfrac{2m}{w} \Big]
= \Pr\!\Big[ Z_j - f_i \ge \tfrac{2m}{w} \Big]
\le \frac{\mathbb{E}[Z_j - f_i]}{2m/w}
\le \frac{m/w}{2m/w} = \tfrac{1}{2}
$$

- For one row the failure probability is $\tfrac12$; taking the min over $d$ rows gives $(\tfrac{1}{2})^{d}$.

**Space.** $O(d \cdot w) = O\!\big(\tfrac{1}{\varepsilon} \log \tfrac{1}{\delta}\big)$.

### Dyadic intervals + Count-Min → Range queries

**Heavy hitter via dyadic intervals.**
1. Build $\log n$ levels (dyadic intervals); each level is an **independent** Count-Min sketch with the same $w$ and $d$.
2. **Insert:** insert into every level. At each level, compute the element's index (its bucket) and insert into that level's sketch.
3. **Query Top-$k$:** set a threshold $\theta = \tfrac{m}{k}$. Starting from the root, compute each bucket's frequency; keep those $> \theta$; for each kept bucket descend into all its children; continue to the leaf level; keep the top-$k$ buckets with frequency $> \theta$.

**Range$(a,b)$.**
- **Insert:** same as heavy hitter, but maintaining counters.
- **Query** (recursive, from the root):
  - if $[a,b]$ equals this interval block → return its counter;
  - else if $[a,b] \cap \text{interval} = \varnothing$ → return $0$;
  - else recurse on the left and right children and sum the two.

---

## 10. Graph Streaming — Connectivity, Bipartiteness, Spanner

Topics: (1) connectivity, (2) bipartiteness testing, (3) distance estimation / spanner.

### Connectivity

Space goes from $O(m+n)$ to $O(n \log n)$.

```
Init: F ← ∅,  connected ← false
Process(token (u,v)):
    if F ∪ {(u,v)} has no cycle:
        F ← F ∪ {(u,v)}
        if |F| = n - 1:  connected ← true
Output: connected
```

**Correctness.** $G$ connected $\Rightarrow$ any spanning forest is a tree (a spanning tree has $n$ vertices and $n-1$ edges) $\iff$ any spanning forest has $n-1$ edges.

**Space.** Store edge set $F$ and one flag. Each edge = two vertices = $2\log n$; at most $n-1$ pairs ⟹ $O(n \log n)$.

### Bipartiteness testing

```
Init: F ← ∅,  is_bipart ← true
Process(token (u,v)):
    if F ∪ {(u,v)} has no cycle:  F ← F ∪ {(u,v)}
    else if it closes an odd cycle: is_bipart ← false
Output: is_bipart
```

**Space.** Similar, $n \cdot 2\log n = O(n \log n)$.

**Correctness.**
- **False:** the graph has an odd cycle.
- **True:** for every $\{u,v\} \in E$, the spanning-forest 2-coloring is consistent; if $(u,v) \in F$ it's fine; if $(u,v) \notin F$ it closes an even cycle, so colors differ as required.

### Distance estimation / $t$-spanner

In a streaming graph, estimate distances in smaller space with a $t$-stretch guarantee:

$$
\delta_G(u,v) \le \hat\delta(u,v) \le t \cdot \delta_G(u,v)
$$

**$t$-spanner algorithm.**
```
Init: H ← ∅
Process(token (u,v)):
    if δ_H(u,v) ≥ t + 1:
        H ← H ∪ {(u,v)}
Output(x,y):  use BFS to query δ_H(x,y)
```

**Space.** $O\!\big(|E(H)| \cdot 2\log n\big) = O\!\big(|E(H)| \cdot \log n\big)$ bits.

---

## 11. Massively Parallel Computation (MPC) — Sorting & MST

### Sorting

Input $N$, space per machine $S = \tfrac{N}{\sqrt N} = \sqrt N$. Goal: sort, output sorted.

```
1. Machine 1 (first round) sorts √N sampled elements, m = O(N^{1-ε}) sampled.
   Find pivots in S_1, S_2, ... — a global division by rank that splits the data.
2. Within machine 1, sort and compute dividers based on the sizes |S_i|;
   produce results [l_i, u_i] in S and the indices of machines that will sort each S_i.
3. Machine 1 sends this routing info to every other machine (to help group).
4. Each machine loops over x in its data; if x ∈ [l_i, u_i], send x to a random
   machine in that group → a single machine collects all of S_i.
5. Terminate when |S_i| ≤ S (sortable on a single machine); otherwise repeat 1–4.
```

**Bounding.**
1. Size **received** by a machine — bound using a **converge-cast tree** (reduce message size received by machine 1).
2. Size **sent**.
3. Number of rounds: Part 1 is $O(\tfrac{1}{\varepsilon})$; Part 2 (recursion depth) is $O(\tfrac{1}{\varepsilon})$. Total $O(\tfrac{1}{\varepsilon}) \cdot O(\tfrac{1}{\varepsilon}) = O(1)$.

### Minimum Spanning Tree (MST)

Edge-weighted, undirected graph. $n = |V|$, $m = |E|$ ($N = m$). Input: list of edges. Output: $\text{MST}(E)$ on one machine.

Parameters: $S = n^{1+\varepsilon}$ ($\varepsilon > 0$), $M = \Theta(\tfrac{N}{S}) = \tfrac{m}{n^{1+\varepsilon}}$, and $m = \Omega(n^{1+\varepsilon})$. **Goal:** compute $\text{MST}(E)$ in $O(1)$ rounds.

**F1 — Shuffle** (for edge set $E'$):
1. Pick $K = \dfrac{2|E'|}{n^{1+\varepsilon}}$ active machines.
2. Distribute $E'$ randomly among the $K$ machines.

**F2 — Filter:**
1. Every machine builds an MST (MSF) based on the edges it holds.

**Whole process** (init $E' = E$):
```
1. Run F1 (shuffle).
2. Run F2 (filter) on each of the K machines; let M_1, ..., M_K be the outputs.
3. If K = 1: output M_1 (the result).
4. Otherwise: set E' = ⋃_{i=1}^{K} E(M_i),  jump back to 1.
```

**Bounds.**

1. **Termination in $O(\tfrac{1}{\varepsilon})$.** With $K_j = \dfrac{2|E'_j|}{n^{1+\varepsilon}}$ and $|E'_j| \le K_{j-1}\cdot(n-1)$,

$$
K_j \le \frac{2}{n^{\varepsilon}} \cdot K_{j-1},
\qquad K_1 = \frac{2|E|}{n^{1+\varepsilon}} \le \frac{2n^2}{n^{1+\varepsilon}} = 2 n^{1-\varepsilon}
$$

so $K_j < \big(\tfrac{2}{n^{\varepsilon}}\big)^{j-1} K_1 = n^{1-\varepsilon}\, 2^{j}$. Setting $K_j = 1$: $n^{1-\varepsilon j} 2^{j} = 1$ ⟹ taking logs, $j = O(\tfrac{1}{\varepsilon}) = O(1)$.

2. **Space & machine count.** $K = \tfrac{2|E'|}{n^{1+\varepsilon}}$ and $S = n^{1+\varepsilon}$, so each of the $K$ machines processes $\tfrac{E'}{K} = \tfrac{|E'|}{2|E'|/(nS)}\cdots \le S$. ✓

3. **Correctness.** $\text{MSF}(E) = \text{MSF}\!\big(\bigcup_{i=1}^{K} E(M_i)\big)$. Proof by the **cycle property**: for any $e = (u,v) \in E \setminus E'$ with $e \notin \text{MST}(E)$ — Step 1: $e$ has the maximum weight in some cycle $C$. Step 2: if we removed $e$ and reconnected the two endpoints via the rest of the cycle, the resulting tree would be no heavier (in fact lighter), contradicting $e$ being in the MST. So such $e$ is never needed, and filtering out non-MST edges per machine is safe.

---

## 12. Dynamic Graph Connectivity

**Operations** (initial graph $|V| = n$, $E = \varnothing$):
- `insert(u,v)` — add edge to $E$
- `delete(u,v)` — remove edge from $E$
- `query: connected(u,v)`

### Data structure

**Levels / clusters.** Each edge $e \in E$ has a level $0 \le l(e) \le \lfloor \log n \rfloor = L_{\max}$. Define $G_i = (V, E_i)$ where $e \in E_i \iff l(e) \ge i$, so

$$
E = E_0 \supseteq E_1 \supseteq \dots \supseteq E_{L_{\max}}
$$

An **$i$-cluster** is a connected component of $G_i$.

> **Invariant:** any $i$-cluster contains at most $\big\lfloor \tfrac{n}{2^{i}} \big\rfloor$ vertices.

So $0$-clusters are the components of $G$, and an $L_{\max}$-cluster is a single vertex.

**Cluster forest.** Each node corresponds to a cluster; tree-level $i$ = $i$-cluster; children are $(i+1)$-clusters. Size $n(u)$ = number of leaves of the subtree rooted at $u$.

### Operations

**Query($u,v$).** From the leaves $u$ and $v$, check whether they can meet — in $O(\log n)$.

**Insert($u,v$).**
```
l(u,v) = 0
find r_u and r_v (roots)
if r_u == r_v:  ok
else:           merge r_u into r_v
```

**Delete($u,v$).**
```
i = level(u,v)
find C_u and C_v at level i+1
  Case 1: C_u = C_v   → just delete it (a replacement edge exists)
  Case 2: C_u ≠ C_v   →
      (a) they may still meet through other level-i edges (a replacement exists), or
      (b) they cannot meet → go down to level i, split as needed
          (amortized cost via the level mechanism)
```

### Local tree (auxiliary structure)

Aim: help the search procedure.
1. Rank nodes, merge into a binary tree.
2. `edge[u][i] = 1`, group edges per level.

Maintain via **merge / split**. A **bitmap** supports clear / set / traverse efficiently.

---

## Quick Review Map

- **Bloom filter:** bound FP probability $(1-\rho)^k$; $\mathbb{E}[(1-\rho)^k] = (\tfrac12)^{k_{\min}}$, $k_{\min} = \tfrac{m}{n}\ln 2$; $m = n\log_2\tfrac1\varepsilon\log_2 e$, within $1.44\times$ of optimal.
- **Distance oracles:** shortest path → Thorup–Zwick oracle, $O(k\, n^{1+1/k})$ space, $O(k)$ query, stretch $2k-1$.
- **Approximate near neighbour:** $c$-approximate $r$-near neighbour — string → Hamming → LSH; set → Jaccard → MinHash; vector → angular → SimHash.
- **Distributed:** P3C (3-coloring a path); faster deterministic coloring in $O(\log^* n)$ rounds (P3C → 3 colors); randomized coloring fails w.p. $\le n^{-c}$.
- **Distributed shortest paths:** SSSP via wave; BFS; leader election → APSP via non-interfering waves + moving-token, $O(n)$ rounds.
- **Randomized algorithms:** Monte Carlo (always fast, sometimes wrong) vs Las Vegas (always correct, fast w.h.p.); refined $(\Delta+1)$-coloring in $O(\log n)$ rounds.
- **Streaming (frequent):** Misra–Gries heavy hitters, space $O(k(\log m + \log n))$; reservoir sampling, $\Pr[a_i \in R] = \tfrac{k}{t}$.
- **Streaming (AMS/Tidemark):** distinct-count estimate $2^{z+1/2}$, tails $\le \tfrac{1}{3}\sqrt2$; **Morris counter** unbiased length estimator.
- **Sketches:** Count-Min ($\hat f \ge f$, error $\le \tfrac{2m}{w}$ w.p. $\ge 1-(\tfrac12)^d$); dyadic intervals → range queries.
- **Graph streaming:** connectivity & bipartiteness in $O(n\log n)$; $t$-spanner for distance estimation.
- **MPC:** sorting in $O(1)$ rounds; MST via shuffle/filter (cycle property), $O(1)$ rounds.
- **Dynamic connectivity:** leveled clusters with invariant $\le \lfloor n/2^i \rfloor$ vertices per $i$-cluster; cluster forest + local tree.
