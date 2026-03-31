"""CLI for grid search over max_freq and max_steps.

Usage:
    python -m minomaly.grid_search_cli --dataset inj_cora --model-path ckpt/model.pt
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from minomaly.data.convert import batch_pyg_data
from minomaly.data.graph import GraphData
from minomaly.data.loaders import extract_anomaly_labels, load_pygod_dataset, pyg_data_to_nx
from minomaly.registry import EMBEDDERS, OUTLIERS, SAMPLERS
from minomaly.search.grid_search import run_grid_search
from minomaly.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Grid search for optimal hyperparameters")
    parser.add_argument("--dataset", default="inj_cora")
    parser.add_argument("--model-path", default="ckpt/model.pt")
    parser.add_argument("--method-type", default="order")
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--search-strategy", default="fast")
    parser.add_argument("--n-neighborhoods", type=int, default=10000)
    parser.add_argument("--max-cands", type=int, default=None)
    parser.add_argument("--output", default="grid_search_results.json")
    args = parser.parse_args()

    device = get_device()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load dataset
    print(f"Loading {args.dataset}...", flush=True)
    data, _ = load_pygod_dataset(args.dataset)
    anomalies = extract_anomaly_labels(data)
    gd = GraphData.from_nx(pyg_data_to_nx(data), device=torch.device("cpu"))
    anomalous_nodes = [i for i, a in enumerate(anomalies.tolist()) if a]
    all_nodes = list(range(gd.num_nodes))
    print(f"  {gd.num_nodes} nodes, {len(anomalous_nodes)} anomalies", flush=True)

    # Load model
    print(f"Loading model ({args.method_type}, input_dim={args.input_dim})...", flush=True)
    if args.method_type == "hybrid":
        model = EMBEDDERS.build("hybrid", input_dim=args.input_dim, hidden_dim=64, margin=0.1,
                                 encoder_name="skip_last_gnn", n_layers=8, conv_type="SAGE",
                                 skip="learnable", dropout=0.0)
    else:
        model = EMBEDDERS.build("order", input_dim=args.input_dim, hidden_dim=64, margin=0.1,
                                 encoder_name="skip_last_gnn", n_layers=8, conv_type="SAGE",
                                 skip="learnable", dropout=0.0)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

    # Sample
    print("Sampling...", flush=True)
    sampler = SAMPLERS.build("tree_fast", n_neighborhoods=args.n_neighborhoods, min_size=1, max_size=30)
    tr = sampler.sample([gd])
    input_dim = args.input_dim
    for n in tr.neighborhoods:
        if n.x.shape[1] > input_dim:
            n.x = n.x[:, :input_dim]

    # Embed
    print("Embedding...", flush=True)
    embs = []
    bs = 1000
    for i in range(0, len(tr.neighborhoods), bs):
        batch = batch_pyg_data(tr.neighborhoods[i:i + bs], device=device)
        with torch.no_grad():
            if hasattr(model, "embed_and_project"):
                embs.append(model.embed_and_project(batch))
            else:
                embs.append(model.emb_model(batch))

    # Outlier detection
    print("Outlier detection...", flush=True)
    all_embs_t = torch.cat(embs, dim=0)
    N = all_embs_t.shape[0]
    # Use a middle-range freq_thresh for starting nodes (will be overridden per grid point)
    freq_thresh = 50.0 / gd.num_nodes
    freq_counts = torch.zeros(N, device=device)
    for start in range(0, N, 256):
        end = min(start + 256, N)
        for eb in embs:
            v = model.batch_predict(eb, all_embs_t[start:end])
            p = model.clf_model(v.unsqueeze(-1))
            freq_counts[start:end] += torch.argmax(p, dim=-1).sum(dim=1).float()
    torch.cuda.synchronize()
    freq_ratios = freq_counts / N
    starting_nodes = set(np.array(tr.real_anchors)[(freq_ratios <= freq_thresh).cpu().numpy()].tolist())
    print(f"  {len(starting_nodes)} starting nodes", flush=True)

    # Grid ranges (matching paper Table 5)
    if args.dataset == "inj_cora":
        max_freq_range = list(range(1, 41))  # 1 to 40
        max_steps_range = list(range(1, 12))  # 1 to 11
    elif args.dataset == "inj_amazon":
        max_freq_range = list(range(10, 201, 10))  # 10 to 200, step 10
        max_steps_range = list(range(1, 22))  # 1 to 21
    elif args.dataset == "inj_flickr":
        max_freq_range = list(range(100, 1401, 100))  # 100 to 1400, step 100
        max_steps_range = list(range(1, 12))  # 1 to 11
    else:
        max_freq_range = list(range(5, 51, 5))
        max_steps_range = list(range(3, 12))

    # Run grid search
    run_grid_search(
        model=model,
        graphs=[gd],
        embs=embs,
        anomalous_nodes=anomalous_nodes,
        all_nodes=all_nodes,
        starting_nodes=starting_nodes,
        max_freq_range=max_freq_range,
        max_steps_range=max_steps_range,
        search_strategy=args.search_strategy,
        max_cands=args.max_cands,
        input_dim=args.input_dim,
        num_nodes=gd.num_nodes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
