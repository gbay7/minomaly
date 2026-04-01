# Minomaly — Future Ideas

## 1. Contextual Anomaly Detection via Two-Space Conditional Frequency

### Motivation

Current Minomaly detects **structural anomalies** — subgraphs whose topology is globally rare. Node attributes are ignored entirely (the GNN uses only anchor indicators as features). However, many real-world anomalies are **contextual**: a subgraph whose structure is common but whose attribute pattern is unusual for that structure, or a structure+attribute combination that doesn't appear elsewhere.

Exact attribute matching for subgraph isomorphism is unsuitable for anomaly detection. Two subgraphs G1 (node labels 1,2,3) and G2 (node labels 5,6,7) with the same structure may belong to the same "context" — they follow the same relational pattern (X, X+1, X+2) with different base values. In social networks, different communities share similar structure and similar attribute generation patterns without matching exactly. The **context** is the learned generation function, not the instantiation.

### Core Idea

Keep the structural OES unchanged. Add a **parallel contextual embedding** and integrate via **conditional frequency**.

### Two Types of Contextual Anomaly

- **Type A**: Structure is common, but attribute pattern is rare for that structure. Example: a triangle where nodes usually have similar attributes, but one instance has wildly divergent attributes.
- **Type B**: Structure AND attribute pattern are jointly rare. Neither is anomalous alone, but the pairing is.

### Architecture: Two Parallel Embedding Spaces

1. **Structural embedding** `phi_struct` (existing OES, order-preserving, no attributes) — for subgraph containment checking
2. **Contextual embedding** `psi_context` (new, with real node attributes) — for context assignment via clustering

The structural OES cannot incorporate attributes directly because the order embedding property `G1 ⊆ G2 ⟺ phi(G1) ⪯ phi(G2)` is purely topological — structural containment does NOT imply attribute containment.

### Integration in Three Stages

**Stage 1: Context Discovery (parallel to OES construction)**

After sampling k neighborhoods and building the structural OES:

1. Compute a context embedding `psi(G'_i)` for each neighborhood using a GNN that takes **real node attributes** as input. This GNN doesn't need order-embedding properties — it produces representations where subgraphs with similar structure + similar attribute patterns are close.
2. Cluster `{psi(G'_1), ..., psi(G'_k)}` into C context clusters (k-means, DBSCAN, etc.) or using this https://arxiv.org/pdf/2006.16904. Each cluster = a discovered context (a class of subgraphs sharing structural template + attribute generation pattern).
3. Store cluster assignments: each `G'_i` gets a context label `ctx(G'_i) ∈ {1, ..., C}`.

The clustering naturally handles invariance: communities with labels (1,2,3) and (5,6,7) get similar `psi` embeddings because GNN message passing captures relative attribute relationships (e.g., average attribute difference between neighbors), not absolute values.

**Stage 2: Conditional Frequency**

Instead of counting frequency against all of RG, partition by context:

```
Freq_G(G' | c) = |{G'_i ∈ RG_c | phi(G') ⪯ phi(G'_i)}| / |RG_c|
```

where `RG_c = {G'_i ∈ RG : ctx(G'_i) = c}`.

The structural order-embedding check stays exactly the same — we just restrict which neighborhoods we count against.

**Key property**: Anti-monotonicity holds within each context partition. If `G1 ⊆ G2`, then `UP(G2) ∩ RG_c ⊆ UP(G1) ∩ RG_c` for any c. The greedy search remains valid.

**Stage 3: Context-Aware Search**

During greedy search, when growing a pattern G':

1. Compute `psi(G')` using the context GNN with real node attributes
2. Assign to nearest context cluster: `c* = argmin_c dist(psi(G'), center_c)`
3. Score using `Freq_G(G' | c*)` instead of `Freq_G(G')`

Anomaly detection becomes:

- **Structural anomaly** (current): `Freq_G(G') < T_F` — globally rare structure
- **Contextual anomaly** (new): `Freq_G(G') >= T_F` but `Freq_G(G' | c*) < T_F` — structure is common but rare within its context

Combined scoring: `score = beta * Freq_G(G') + (1-beta) * Freq_G(G' | c*)`

### Training the Context GNN (`psi`)

Options (ranked by rigor):

1. **Self-supervised contrastive**: Augment subgraphs, pull augmentations together in embedding space. No labels needed. Naturally encodes attribute patterns.
2. **Autoencoder**: Train `psi` to reconstruct node attributes from the subgraph embedding. Similar attribute patterns → similar embeddings.
3. **No training**: Raw attribute-aware message passing (even random init) with clustering. Fast baseline — WL-style aggregation already captures structural+attribute features.

### Pipeline Diagram

```
                    Structural OES (existing)
                    phi(G') — order embedding, no attributes
                    Used for: containment check phi(G') ⪯ phi(G'_i)
                         |
Sample k neighborhoods --+
with real attributes     |
                         |
                    Context Embedding (new)
                    psi(G') — attribute-aware GNN
                    Used for: clustering -> context assignment
                         |
                         v
              Context-Conditional Frequency
              Freq_G(G' | c) = count within context c
                         |
                         v
              Greedy Search (same algorithm)
              but scoring uses Freq_G(G' | c)
```

### Why This Works

- **Decoupling**: phi handles containment/anti-monotonicity/frequency. psi handles attribute pattern discovery. They operate in parallel.
- **Integration point**: The frequency computation — filter RG by context before counting.
- **Invariance**: Clustering captures the "generation functions" (f1, f2, f3...), not exact values.
- **Interpretability preserved**: Detected patterns still come with structural visualization + now also a context label explaining which behavioral class they deviate from.

---

## 2. Efficient Search Alternatives to the Greedy Expansion

### Problem Analysis

The greedy search solves: **find the smallest connected subgraph G' containing starting node u such that Freq_G(G') < T_F.**

Current cost per starting node: `O(T · c · k · d)` where T = max_steps, c = max_cands, k = OES size, d = embedding dim. The bottleneck is the frequency computation: for each of c candidates at each of T steps, the algorithm scans ALL k reference embeddings.

Three sources of waste:

1. **Redundant scanning across steps.** At step t, the pattern G'_t is a superset of G'_{t-1}. By anti-monotonicity, the supergraph set S_t ⊆ S_{t-1}. But the algorithm re-scans all k embeddings instead of only S_{t-1}.
2. **Full frequency for ranking.** All c candidates get exact O(k·d) frequency computation, but we only need to RANK them (pick the best). Exact frequency is only needed for the chosen candidate (to decide whether to stop).
3. **Sequential dependency.** Step t+1 depends on the choice at step t. No parallelism across steps.

### Proposed Approaches

#### 2a. Incremental Supergraph Set (same algorithm, faster frequency)

Maintain a cached set `S_t = {i : phi(G'_t) ⪯ phi(G'_i)}` — the indices of reference embeddings that are currently supergraphs of the pattern.

At step t+1, for candidate v, only check `i ∈ S_t` (not all k), because `S_{t+1} ⊆ S_t` by anti-monotonicity. The frequency is `|S_{t+1}| / k`.

As the search progresses and frequency drops from ~50% → ~1%, the working set shrinks from 5000 → 100. The later steps — which have the largest frontiers — become the cheapest.

- **Average cost**: `O(T · c · f̄ · k · d)` where f̄ is mean frequency across steps (~15% for a typical search going 50% → 1%). Gives ~6x speedup with identical results.
- **Limitation**: Still sequential, still evaluates all c candidates.

#### 2b. Embed-Rank-Verify (proxy scoring to eliminate c factor)

Key insight: exact frequency is needed to DECIDE (compare against T_F), not to RANK candidates.

At each step:
1. Batch-embed all c candidates — one GPU forward pass
2. **Rank cheaply** using a proxy: train a small MLP regressor `f(phi) ≈ Freq_G` from the (embedding, frequency) pairs already available from Phase 2 (thousands of them). The MLP scores each candidate in O(d) instead of O(k·d).
3. **Verify only the top-1** by computing exact frequency against the cached supergraph set S_{t-1}

Cost per step drops from `O(c · k · d)` to `O(c · d + |S_t| · d)`. The factor of c is removed from the dominant frequency term entirely.

Alternative proxies (no training needed): rank by OES norm `||phi(G' ∪ {v})||` or by max component increase. Larger embeddings correspond to larger subgraphs which have lower frequency by anti-monotonicity.

#### 2c. Batch Random Walks + Parallel Verification

Replace the sequential greedy with embarrassingly parallel sampling:

1. From starting node u, generate R random connected subgraphs of sizes 1 to max_steps (random walks with varying lengths)
2. Batch-embed ALL R subgraphs in one GPU forward pass
3. Batch-compute frequencies for all R in one pass over the OES (stack R embeddings into one (R, d) tensor, compute pairwise violations against all k in a single tensor operation)
4. Among those with Freq < T_F, return the smallest

**Cost**: `O(R · k · d)`, fully parallelizable on GPU in a single tensor operation.

**Comparison**: Greedy costs `O(T · c · k · d)` sequentially. With T=7, c=20: 140 sequential frequency computations. With R=50 parallel random walks: 50 frequency computations, all batched.

Wall-clock speedup is much larger than 140/50 because the 50 are done in a SINGLE GPU kernel while the 140 span 7 dependent steps.

**Quality trade-off**: Greedy finds the optimal (smallest) anomalous subgraph. Random sampling finds a near-optimal one with high probability. With R ≥ 100, the chance of missing the optimal is very low because there are typically many anomalous expansion paths, not just one unique path. Anti-monotonicity guarantees that any sufficiently long walk will cross T_F.

#### 2d. Learned Expansion Policy (amortized search)

Train a lightweight policy network `pi(v | G'_t, frontier)` that directly predicts which node to add, without any frequency computation during inference:

**Training**: Run greedy search on training graphs. Record (state, action, reward) tuples where state = current subgraph embedding, action = chosen node, reward = frequency decrease. Train via supervised learning (imitate greedy) or RL (maximize frequency decrease per step).

**Inference**: pi selects the next node in O(1). No GNN embedding, no frequency computation during search. Only compute frequency at the END to verify.

**Cost**: `O(S · T)` for search + `O(S · k · d)` for final verification. The `k · d` factor appears ONCE per starting node, not once per step per candidate.

This is the theoretical optimum: amortize the expensive frequency computation into an offline training phase.

**Drawback**: Requires training data (greedy runs), may not generalize across very different graph structures without retraining.

### Recommended Combination: 2a + 2b

```
For each starting node u:
    S_0 = {1, ..., k}                           # full supergraph set
    G' = {u}

    For step t = 1 to T:
        Batch-embed all c candidates             # one GPU forward pass
        Proxy-rank by learned f(phi) or norm     # O(c · d)
        Pick top-1 candidate v*

        Compute exact Freq via cached S_{t-1}    # O(|S_{t-1}| · d), shrinks each step
        Update S_t ⊆ S_{t-1}

        G' = G' ∪ {v*}
        If Freq < T_F: break
```

**Per step**: `O(c · GNN_batch + c · d + |S_t| · d)`
**Current**: `O(c · GNN_batch + c · k · d)`

This removes the `c × k` interaction. For typical parameters (c=20, k=10000, f̄=15%), this gives **10-20x improvement** with identical or near-identical search quality.

For maximum throughput over search quality, approach 2c (batch random walks) replaces the inherently sequential greedy with a single GPU-parallel operation.

---

## 3. Cross-Phase Frequency Caching (Phase 2 → Phase 3)

### Observation

Phase 2 (starting node detection) already computes `Freq_G(G'_i)` for every one of the k neighborhoods in RG. This is the O(k² · d) computation. These k frequency values are then discarded — Phase 3 (search) recomputes frequencies from scratch for every candidate pattern.

### The Insight

Cache the k frequencies `{Freq_G(G'_1), ..., Freq_G(G'_k)}` from Phase 2. During Phase 3, when computing `Freq_G(G'_t)` for a search pattern, the cached values provide **free bounds via anti-monotonicity** that can eliminate expensive exact computations.

When the search identifies that `G'_t ⊆ G'_i` (i.e., G'_i is a supergraph of the current pattern — which is exactly the containment check we're already doing), we know from the cached Phase 2 data:

```
Freq_G(G'_i) ≤ Freq_G(G'_t)
```

because G'_i is larger than G'_t, so fewer things contain G'_i. The cached `Freq_G(G'_i)` is a **lower bound** on our pattern's frequency.

Taking the maximum across all identified supergraphs:

```
Freq_G(G'_t) ≥ max_{i ∈ S_t} Freq_G(G'_i)
```

where `S_t = {i | G'_t ⊆ G'_i}` is the supergraph set.

### How This Helps

**Early candidate pruning**: When evaluating candidate v at step t+1, suppose we've already determined a few members of S_{t+1}(v) (partial scan). If `max Freq_G(G'_i)` among those already found exceeds T_F, we know `Freq_G(G'_{t+1}) > T_F` — this candidate is NOT anomalous yet. We can stop scanning the remaining embeddings for this candidate and move on. This avoids completing the O(|S_t|) scan for candidates that clearly won't cross the threshold.

**Early termination confirmation**: Combined with the upper bound from the previous step (`Freq_G(G'_{t+1}) ≤ Freq_G(G'_t)`), if the upper bound is already below T_F, the pattern is anomalous regardless of which candidate we pick. No frequency computation needed at all — just pick any candidate and stop.

**Tighter incremental bounds**: As the search progresses, the supergraph set S_t shrinks. At each step we refine both bounds:
- Upper: `Freq_G(G'_t)` from the exact computation at the previous step (decreasing)
- Lower: `max_{i ∈ S_t} Freq_G(G'_i)` from cached Phase 2 values (also adjusting as S_t changes)

When upper and lower bounds are both on the same side of T_F, exact computation is unnecessary.

### Combined with Idea 2a (Supergraph Set Caching)

These two ideas compose naturally:

```
Phase 2: compute and cache freq_cache[i] = Freq_G(G'_i) for all i = 1..k

Phase 3 search:
    S_0 = {1, ..., k}
    For step t:
        For each candidate v:
            # Incremental check against S_t only (Idea 2a)
            S_{t+1}(v) = {i ∈ S_t | phi(G'_t ∪ {v}) ⪯ phi(G'_i)}

            # Lower bound from cached frequencies (Idea 3)
            lower_bound = max(freq_cache[i] for i in S_{t+1}(v))
            if lower_bound > T_F:
                skip candidate    # provably not anomalous yet

        # Exact freq for chosen candidate
        Freq = |S_{t+1}| / k
        if Freq < T_F: break
```

### Cost

Zero additional computation — Phase 2 already does the work. The only cost is storing k float values (one per neighborhood), which is negligible. The savings come from skipping exact frequency scans for candidates where the cached bounds already determine the outcome.
