"""Test Insight 2 (freq_cache pre-filter) in sampled search.

Compares 4 variants on Cora:
  A: no filter (baseline)
  B: strict filter (freq_cache > f_prev)
  C: margin=0.02 (freq_cache > f_prev + 0.02)
  D: margin=0.05 (freq_cache > f_prev + 0.05)

For each: runs sampled search on 100 starting nodes, reports P/R/F1 + timing.
"""
import pickle, random, sys, time, torch
sys.path.insert(0, ".")

from minomaly.registry import EMBEDDERS, SCORING
from minomaly.data.graph import GraphData
from minomaly.data.loaders import load_organic_dataset, extract_anomaly_labels, pyg_data_to_nx
from minomaly.data.convert import batch_pyg_data
from minomaly.search.beam import Beam
from minomaly.search.beam_set import BeamSet
from minomaly.search.pattern_store import PatternStore
from minomaly.evaluation.metrics import get_stat_results


def run_sampled_search(
    starting_nodes, graph, model, embs, scorer, max_strength,
    sampled_fc, ref_sample, m,
    insight2_margin=None,  # None=disabled, 0=strict, >0=with margin
    max_steps=5, n_beams=3, max_cands=20, min_steps=1,
):
    """Minimal sampled search with configurable Insight 2."""
    beam_sets = []
    for node in starting_nodes:
        beam = Beam(node=int(node), graph=graph, total_weight=graph.num_nodes,
                    add_self_loop=True, node_anchored=True, input_dim=2)
        beam_sets.append(BeamSet([beam]))

    verified = BeamSet()
    pattern_store = PatternStore(node_anchored=True)
    node_votes = {}
    active_mask = torch.ones(m, dtype=torch.bool, device=ref_sample.device)

    steps = 1
    while beam_sets and steps < max_steps:
        steps += 1

        all_cands = []
        set_boundaries = []
        for beam_set in beam_sets:
            start = len(all_cands)
            for beam in beam_set:
                if not beam.frontier:
                    continue
                cands = beam.gen_candidates(total_weight=graph.num_nodes, max_cands=max_cands)
                all_cands.extend(cands)
            set_boundaries.append((start, len(all_cands)))

        if not all_cands:
            break

        # Insight 2: pre-filter refs by cached freq
        if insight2_margin is not None and sampled_fc is not None and steps > 2:
            max_parent_freq = 0.0
            for beam_set in beam_sets:
                for beam in beam_set:
                    if beam.freq is not None and beam.freq > max_parent_freq:
                        max_parent_freq = beam.freq
            if max_parent_freq > 0:
                threshold = max_parent_freq + insight2_margin
                freq_filter = sampled_fc <= threshold
                active_mask = active_mask & freq_filter

        # Embed
        data_list = [c.get_pyg_data() for c in all_cands]
        batch = batch_pyg_data(data_list, device=ref_sample.device)
        with torch.no_grad():
            if hasattr(model, "embed_and_project"):
                cand_embs = model.embed_and_project(batch)
            else:
                cand_embs = model.emb_model(batch)
        for c, emb in zip(all_cands, cand_embs):
            c.emb = emb.detach()

        # Score against active refs
        active_refs = ref_sample[active_mask]
        n_active = active_refs.shape[0]
        if n_active == 0:
            break

        all_emb_stack = torch.stack([c.emb for c in all_cands])
        with torch.no_grad():
            violations = model.batch_predict(active_refs, all_emb_stack)
            preds = model.clf_model(violations.unsqueeze(-1))
            is_super = torch.argmax(preds, dim=-1).bool()
            freq_counts = is_super.float().sum(dim=1)

        for i, c in enumerate(all_cands):
            c.freq = freq_counts[i].item() / m  # normalize by TOTAL m, not n_active
            c.freq_history.append((len(c.neigh), c.freq))
            c.score = scorer(c.freq, c.weight, 0.33, c.last_score)
            c.unchange = c.unchange + 1 if c.score >= c.last_score else 0

        new_beam_sets = []
        for s, e in set_boundaries:
            if s == e:
                continue
            new_beams = BeamSet(all_cands[s:e])
            new_beams.prune(0.0, max_strength, 5)
            new_beams.sort_and_keep(n_beams, node_votes)
            if steps >= min_steps:
                v = new_beams.extract_verified(0.0, max_strength)
            else:
                v = BeamSet()
            for beam in v:
                pattern_store.add(beam)
            verified += v
            for beam in new_beams:
                if beam.score is not None:
                    node_votes[beam.node] = node_votes.get(beam.node, 0) + 1
            if new_beams:
                new_beam_sets.append(new_beams)

        beam_sets = new_beam_sets

    return verified


def main():
    # Load model
    model = EMBEDDERS.build("hybrid", input_dim=2, hidden_dim=64, margin=0.1,
                            encoder_name="skip_last_gnn", n_layers=8, conv_type="SAGE",
                            skip="learnable", dropout=0.0, positive_only=False)
    sd = torch.load("ckpt/model.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to("cuda").eval()

    # Load cached embeddings
    with open("savings/embeddings/inj_cora_10000_1_30_True_1000_dim2_hybrid.p", "rb") as f:
        cached = pickle.load(f)
    embs = [e.to("cuda") for e in cached["embs"]]
    all_embs = torch.cat(embs, dim=0)
    K = all_embs.shape[0]

    # Load freq cache
    with open("savings/starting_nodes/inj_cora_10000_1_30_True_1000_dim2_hybrid_ofreq35.0_starting_nodes.p", "rb") as f:
        cached_start = pickle.load(f)
    starting_nodes = list(cached_start["starting_nodes"])

    # Load freq_cache from model-based detector
    # Re-derive: we need the freq_ratios tensor
    # For simplicity, compute it inline
    freq_cache = torch.zeros(K, device="cuda")
    chunk = 64
    for i in range(0, K, chunk):
        end = min(i + chunk, K)
        query = all_embs[i:end]
        for emb_batch in embs:
            violations = model.batch_predict(emb_batch, query)
            preds = model.clf_model(violations.unsqueeze(-1))
            supergraphs = torch.argmax(preds, dim=-1)
            freq_cache[i:end] += supergraphs.sum(dim=1).float()
    freq_cache /= K
    print(f"Freq cache: min={freq_cache.min():.4f}, med={freq_cache.median():.4f}, max={freq_cache.max():.4f}")

    # Load graph
    data, _ = load_organic_dataset("inj_cora")
    anomaly_labels = extract_anomaly_labels(data)
    nx_graph = pyg_data_to_nx(data)
    graph = GraphData.from_nx(nx_graph, device="cpu")
    anomalous_nodes = [i for i, a in enumerate(anomaly_labels.tolist()) if a]
    all_nodes = list(range(graph.num_nodes))

    scorer = SCORING.build("freq")
    max_strength = scorer(5.0 / graph.num_nodes, 5 / graph.num_nodes, 0.33)

    # Fixed sample
    m = 500
    torch.manual_seed(42)
    sample_idx = torch.randperm(K, device="cuda")[:m]
    ref_sample = all_embs[sample_idx]
    sampled_fc = freq_cache[sample_idx]

    # Fixed starting nodes
    random.seed(42)
    random.shuffle(starting_nodes)
    batch = starting_nodes[:200]

    print(f"\nRunning on {len(batch)} starting nodes, m={m}, max_strength={max_strength:.6f}")
    print(f"Sampled freq cache stats: min={sampled_fc.min():.4f}, med={sampled_fc.median():.4f}, max={sampled_fc.max():.4f}")
    print()

    variants = [
        ("A: no filter (baseline)", None),
        ("B: strict (margin=0)", 0.0),
        ("C: margin=0.02", 0.02),
        ("D: margin=0.05", 0.05),
        ("E: margin=0.10", 0.10),
    ]

    print(f"{'Variant':<30} {'P':>8} {'R':>8} {'F1':>8} {'AUROC':>8} {'TP':>5} {'FP':>5} {'Ver':>5} {'Time':>8}")
    print("-" * 100)

    for name, margin in variants:
        t0 = time.time()
        verified = run_sampled_search(
            batch, graph, model, embs, scorer, max_strength,
            sampled_fc, ref_sample, m,
            insight2_margin=margin,
        )
        elapsed = time.time() - t0

        stats = get_stat_results(anomalous_nodes, verified, all_nodes)
        print(f"{name:<30} {stats['precision']:>8.4f} {stats['recall']:>8.4f} {stats['f1']:>8.4f} "
              f"{stats['auroc']:>8.4f} {stats['tp']:>5} {stats['fp']:>5} {len(verified):>5} {elapsed:>7.1f}s")


if __name__ == "__main__":
    main()
