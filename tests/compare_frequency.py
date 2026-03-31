"""Compare frequency computation between original and new code.

For specific anchor nodes (true anomalies + true normals), compute:
1. Exact frequency (how many of K references predict "is subgraph")
2. Violation score distribution (mean, std, min, max)
3. Supergraph count per embedding batch
4. Step-by-step beam growth with scores

Runs with the original model on Cora.
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
import networkx as nx
import torch_geometric.utils as pyg_utils

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code-original'))

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def load_cora():
    """Load Cora and return (nx_graph, anomalous_nodes, data)."""
    data = torch.load(
        os.path.expanduser('~/.pygod/data/inj_cora.pt'),
        map_location='cpu', weights_only=False,
    )
    g = pyg_utils.to_networkx(data, to_undirected=True)
    anomalies = ((data.y >> 1) & 1)
    anom_nodes = set(i for i, a in enumerate(anomalies.tolist()) if a)
    return g, anom_nodes, data


def load_model(device):
    """Load original pretrained model."""
    from minomaly.registry import EMBEDDERS
    model = EMBEDDERS.build(
        'order', input_dim=1, hidden_dim=64, margin=0.1,
        encoder_name='skip_last_gnn', n_layers=8, conv_type='SAGE',
        skip='learnable', dropout=0.0,
    )
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(
        'code-original/ckpt/model.pt', map_location=device, weights_only=True,
    ))
    return model


def embed_subgraph_our_way(model, graph, neigh, anchor, device):
    """Embed using OUR code's approach."""
    from minomaly.search.beam import Beam
    from minomaly.data.graph import GraphData
    from minomaly.data.convert import batch_pyg_data

    gd = GraphData.from_nx(graph, device=torch.device('cpu'))
    beam = Beam(
        node=anchor, graph=gd, neigh=[n for n in neigh if n != anchor],
        frontier=[], visited=set(neigh), total_weight=graph.number_of_nodes(),
        add_self_loop=True, node_anchored=True, input_dim=1,
    )
    data = beam.get_pyg_data()
    batch = batch_pyg_data([data], device=device)
    with torch.no_grad():
        emb = model.emb_model(batch).squeeze(0)
    return emb, data


def embed_subgraph_original_way(model, graph, neigh, anchor, device):
    """Embed using ORIGINAL code's approach (DeepSNAP)."""
    try:
        from common import utils as orig_utils
        sub = graph.subgraph(neigh)
        batch = orig_utils.batch_nx_graphs([sub], anchors=[anchor])
        batch = batch.to(device)
        with torch.no_grad():
            emb = model.emb_model(batch).squeeze(0)
        return emb, batch
    except ImportError:
        return None, None


def compute_frequency(emb, ref_embs, model, device):
    """Compute exact frequency: count of supergraphs among ref_embs."""
    freq_count = 0
    n_total = 0
    violations_all = []

    for ref_batch in ref_embs:
        ref_batch = ref_batch.to(device)
        n_total += len(ref_batch)

        # Violation: max(0, beam - ref)^2
        violations = model.predict((ref_batch, emb))
        violations_all.append(violations.cpu())

        # Classify
        preds = model.clf_model(violations.unsqueeze(1))
        is_super = torch.argmax(preds, dim=1)
        freq_count += is_super.sum().item()

    all_v = torch.cat(violations_all)
    return {
        'freq': freq_count / n_total,
        'freq_count': freq_count,
        'n_total': n_total,
        'violation_mean': all_v.mean().item(),
        'violation_std': all_v.std().item(),
        'violation_min': all_v.min().item(),
        'violation_max': all_v.max().item(),
        'violation_median': all_v.median().item(),
        'pct_below_threshold': (all_v < 0.616).float().mean().item(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading Cora...", flush=True)
    g, anom_nodes, data = load_cora()
    print(f"  {g.number_of_nodes()} nodes, {len(anom_nodes)} anomalies")

    print("Loading model...", flush=True)
    model = load_model(device)

    # Sample reference neighborhoods (same as detection pipeline)
    print("Sampling 10000 reference neighborhoods...", flush=True)
    from minomaly.data.graph import GraphData
    from minomaly.data.convert import batch_pyg_data
    from minomaly.registry import SAMPLERS

    gd = GraphData.from_nx(g, device=torch.device('cpu'))
    sampler = SAMPLERS.build('tree_fast', n_neighborhoods=10000, min_size=1, max_size=30)
    tr = sampler.sample([gd])
    # Slice to 1D
    for n in tr.neighborhoods:
        if n.x.shape[1] > 1:
            n.x = n.x[:, :1]

    print("Embedding references...", flush=True)
    ref_embs = []
    for i in range(0, 10000, 1000):
        batch = batch_pyg_data(tr.neighborhoods[i:i+1000], device=device)
        with torch.no_grad():
            ref_embs.append(model.emb_model(batch))

    # Pick test nodes: 5 true anomalies + 5 true normals
    anom_list = sorted(anom_nodes)
    normal_candidates = [n for n in range(g.number_of_nodes()) if n not in anom_nodes]
    test_anomalies = anom_list[:5]
    test_normals = random.sample(normal_candidates, 5)

    print(f"\nTest anomalies: {test_anomalies}")
    print(f"Test normals:   {test_normals}")
    print()

    # For each test node: grow a 7-node neighborhood, embed it, compute frequency
    results = []
    freq_thresh = 35.0 / g.number_of_nodes()
    print(f"Frequency threshold (max_strength): {freq_thresh:.6f}")
    print(f"{'Node':>6} {'Type':>8} {'Neigh':>30} {'Freq_Ours':>10} {'ViolMean':>10} {'ViolMed':>10} {'<%τ':>6} {'Anomalous?':>10}")
    print("-" * 110)

    for node in test_anomalies + test_normals:
        is_anom = node in anom_nodes
        node_type = "ANOM" if is_anom else "NORMAL"

        # Grow 7-node neighborhood via BFS
        neigh = [node]
        frontier = list(set(g.neighbors(node)) - set(neigh))
        visited = {node}
        while len(neigh) < 7 and frontier:
            new = random.choice(frontier)
            neigh.append(new)
            visited.add(new)
            frontier += [n for n in g.neighbors(new) if n not in visited]
            frontier = [x for x in frontier if x not in visited]

        if len(neigh) < 3:
            print(f"{node:>6} {node_type:>8} {'(too small)':>30}")
            continue

        # Embed our way
        emb_ours, pyg_data = embed_subgraph_our_way(model, g, neigh, node, device)

        # Compute frequency
        freq_info = compute_frequency(emb_ours, ref_embs, model, device)

        neigh_str = str(neigh[:5]) + ('...' if len(neigh) > 5 else '')
        is_anomalous = freq_info['freq'] <= freq_thresh

        print(
            f"{node:>6} {node_type:>8} {neigh_str:>30} "
            f"{freq_info['freq']:>10.6f} "
            f"{freq_info['violation_mean']:>10.4f} "
            f"{freq_info['violation_median']:>10.4f} "
            f"{freq_info['pct_below_threshold']:>6.1%} "
            f"{'YES' if is_anomalous else 'no':>10}"
        )

        # Count how many neighbors are anomalous
        anom_in_neigh = len(set(neigh) & anom_nodes)

        results.append({
            'node': node,
            'is_true_anomaly': is_anom,
            'neigh': neigh,
            'neigh_size': len(neigh),
            'anom_in_neigh': anom_in_neigh,
            'detected_anomalous': is_anomalous,
            **freq_info,
        })

    # Summary
    print()
    print("=== Summary ===")
    anom_results = [r for r in results if r['is_true_anomaly']]
    norm_results = [r for r in results if not r['is_true_anomaly']]

    if anom_results:
        anom_freqs = [r['freq'] for r in anom_results]
        print(f"True anomalies: freq mean={np.mean(anom_freqs):.6f}, "
              f"detected={sum(1 for r in anom_results if r['detected_anomalous'])}/{len(anom_results)}")
    if norm_results:
        norm_freqs = [r['freq'] for r in norm_results]
        print(f"True normals:   freq mean={np.mean(norm_freqs):.6f}, "
              f"detected={sum(1 for r in norm_results if r['detected_anomalous'])}/{len(norm_results)}")

    # Save
    out_path = 'tests/compare_results.json'
    os.makedirs('tests', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
