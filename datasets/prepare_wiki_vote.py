"""Prepare Wiki-Vote as a PyGOD-compatible anomaly detection dataset.

Downloads Wiki-Vote (7,115 nodes, 103K edges), converts to undirected,
injects structural anomalies (dense cliques) following the PyGOD protocol,
and saves as a .pt file with the same label encoding:
    (data.y >> 1) & 1 == 1 for structural anomalies.

Usage:
    python datasets/prepare_wiki_vote.py
"""

import os
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def load_wiki_vote(path: str = "datasets/wiki-Vote.txt") -> nx.Graph:
    """Load Wiki-Vote as an undirected NetworkX graph."""
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)

    # Relabel to contiguous 0..n-1
    G = nx.convert_node_labels_to_integers(G)
    print(f"Wiki-Vote: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def inject_structural_anomalies(
    G: nx.Graph,
    n_cliques: int = 15,
    clique_sizes: tuple[int, int] = (10, 20),
    seed: int = 42,
) -> tuple[nx.Graph, set[int]]:
    """Inject dense cliques as structural anomalies.

    Follows the PyGOD injection protocol:
    1. Select random nodes as clique centers
    2. Connect each center to (clique_size - 1) random neighbors
    3. Make the selected nodes fully connected (clique)

    Returns the modified graph and the set of anomalous node IDs.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    nodes = list(G.nodes())
    num_nodes = len(nodes)
    anomalous_nodes = set()

    for _ in range(n_cliques):
        # Random clique size
        size = rng.randint(clique_sizes[0], clique_sizes[1])

        # Pick random nodes for the clique
        clique_nodes = rng.sample(nodes, min(size, num_nodes))

        # Make them fully connected
        for i in range(len(clique_nodes)):
            for j in range(i + 1, len(clique_nodes)):
                G.add_edge(clique_nodes[i], clique_nodes[j])

        anomalous_nodes.update(clique_nodes)

    print(f"Injected {n_cliques} cliques: {len(anomalous_nodes)} anomalous nodes "
          f"({len(anomalous_nodes) / num_nodes:.1%})")
    print(f"After injection: {G.number_of_edges()} edges")
    return G, anomalous_nodes


def to_pyg_data(G: nx.Graph, anomalous_nodes: set[int]) -> Data:
    """Convert to PyG Data with PyGOD-compatible label encoding.

    Label encoding: y uses bit flags
        bit 0: contextual anomaly (not used here, set to 0)
        bit 1: structural anomaly (1 if anomalous)
    So: (y >> 1) & 1 == 1 for structural anomalies.
    """
    num_nodes = G.number_of_nodes()

    # Build edge_index
    edges = list(G.edges())
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node features: degree + clustering coefficient (simple structural features)
    degrees = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float)
    clustering = torch.tensor(
        [nx.clustering(G, i) for i in range(num_nodes)], dtype=torch.float
    )
    x = torch.stack([degrees, clustering], dim=1)

    # Labels: bit 1 = structural anomaly
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node in anomalous_nodes:
        y[node] = 0b10  # bit 1 set

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    n_anom = ((y >> 1) & 1).sum().item()
    print(f"PyG Data: {num_nodes} nodes, {edge_index.shape[1]} edges, "
          f"{n_anom} anomalies ({n_anom / num_nodes:.1%})")
    return data


def main():
    G = load_wiki_vote()

    # Inject ~5% anomalies (matching PyGOD convention)
    num_nodes = G.number_of_nodes()
    target_anomalies = int(num_nodes * 0.05)
    avg_clique_size = 15
    n_cliques = max(1, target_anomalies // avg_clique_size)

    G, anomalous_nodes = inject_structural_anomalies(
        G, n_cliques=n_cliques, clique_sizes=(10, 20),
    )

    data = to_pyg_data(G, anomalous_nodes)

    # Save
    cache_dir = os.path.join(os.path.expanduser("~"), ".pygod", "data")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "inj_wiki_vote.pt")
    torch.save(data, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
