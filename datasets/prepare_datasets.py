"""Prepare organic anomaly datasets in PyGOD-compatible format.

Creates .pt files identical in structure to inj_cora / inj_amazon / inj_flickr:
    - data.x:          (num_nodes, num_features)  node features
    - data.edge_index:  (2, num_edges)             undirected edges (COO)
    - data.y:           (num_nodes,)               bit-encoded anomaly labels
                                                   bit 1 = structural anomaly
    - data.labeled_mask: (num_nodes,)              True for nodes with ground-truth

Usage:
    python datasets/prepare_datasets.py --dataset elliptic
    python datasets/prepare_datasets.py --dataset mgtab
    python datasets/prepare_datasets.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce


# ---------------------------------------------------------------------------
# Elliptic Bitcoin
# ---------------------------------------------------------------------------

def prepare_elliptic(output_dir: str) -> Path:
    """Download Elliptic Bitcoin dataset via PyG and convert to PyGOD format.

    Source: Weber et al. (2019) — 203,769 Bitcoin transactions, 234,355 edges.
    Labels: 0 = licit, 1 = illicit, 2 = unknown.
    Only ~23% of nodes have ground-truth labels (2.2% illicit).

    The graph consists of 49 disjoint temporal components. We keep the full
    graph (all components) and treat it as a static undirected graph.
    """
    from torch_geometric.datasets import EllipticBitcoinDataset

    print("=" * 60)
    print("Preparing Elliptic Bitcoin dataset")
    print("=" * 60)

    # Download via PyG (auto-cached)
    raw_dir = os.path.join(output_dir, "_raw", "elliptic")
    print(f"Downloading to {raw_dir} ...")
    dataset = EllipticBitcoinDataset(root=raw_dir)
    data = dataset[0]

    print(f"  Nodes:    {data.num_nodes:,}")
    print(f"  Edges:    {data.edge_index.size(1):,} (directed)")
    print(f"  Features: {data.x.size(1)}")

    # Label distribution
    n_licit = (data.y == 0).sum().item()
    n_illicit = (data.y == 1).sum().item()
    n_unknown = (data.y == 2).sum().item()
    print(f"  Licit:    {n_licit:,} ({100*n_licit/data.num_nodes:.1f}%)")
    print(f"  Illicit:  {n_illicit:,} ({100*n_illicit/data.num_nodes:.1f}%)")
    print(f"  Unknown:  {n_unknown:,} ({100*n_unknown/data.num_nodes:.1f}%)")

    # Convert directed → undirected
    edge_index = to_undirected(data.edge_index)
    edge_index = coalesce(edge_index, num_nodes=data.num_nodes)
    print(f"  Edges:    {edge_index.size(1):,} (undirected)")

    # Convert labels to PyGOD bit-encoding:
    #   normal = 0 (0b00), structural anomaly = 2 (0b10)
    # Illicit nodes → structural anomaly (bit 1 set)
    # Licit + unknown → normal (value 0)
    y_pygod = torch.zeros(data.num_nodes, dtype=torch.long)
    y_pygod[data.y == 1] = 2  # bit 1 = structural anomaly

    # Labeled mask: True for nodes with known ground-truth (licit or illicit)
    labeled_mask = data.y != 2

    new_data = Data(
        x=data.x,
        edge_index=edge_index,
        y=y_pygod,
        num_nodes=data.num_nodes,
    )
    new_data.labeled_mask = labeled_mask

    out_path = Path(output_dir) / "elliptic.pt"
    torch.save(new_data, out_path)
    print(f"\nSaved to {out_path}")
    print(f"  Anomaly nodes (illicit): {(y_pygod == 2).sum().item():,}")
    print(f"  Normal nodes (licit+unknown): {(y_pygod == 0).sum().item():,}")
    print(f"  Labeled nodes: {labeled_mask.sum().item():,}")

    _print_verification(new_data)
    return out_path


# ---------------------------------------------------------------------------
# MGTAB (Multi-Relational Graph-Based Twitter Account Detection Benchmark)
# ---------------------------------------------------------------------------

_MGTAB_GDRIVE_ID = "1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe"


def _download_gdrive(file_id: str, output: str) -> None:
    """Download a file from Google Drive by ID using gdown."""
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading from Google Drive (id={file_id[:8]}...) ...")
    gdown.download(url, output, quiet=False)


def prepare_mgtab(output_dir: str) -> Path:
    """Download MGTAB dataset and convert to PyGOD format.

    Source: MGTAB (Multi-Relational Graph-Based Twitter Account Detection).
    10,199 expert-annotated Twitter users (7,451 humans / 2,748 bots).
    7 relation types: followers, friends, mention, reply, quoted, URL, hashtag.

    We merge all edge types into a single undirected graph.
    Bot accounts are marked as structural anomalies.
    """
    import zipfile

    print("=" * 60)
    print("Preparing MGTAB dataset")
    print("=" * 60)

    raw_dir = Path(output_dir) / "_raw" / "mgtab"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "MGTAB.zip"
    extract_dir = raw_dir / "MGTAB"

    if not extract_dir.exists():
        if not zip_path.exists():
            _download_gdrive(_MGTAB_GDRIVE_ID, str(zip_path))
            print(f"  Downloaded to {zip_path}")
        print("  Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    # Find the actual data directory (may be nested)
    data_dir = _find_mgtab_tensors(raw_dir)
    if data_dir is None:
        raise FileNotFoundError(
            f"Could not find MGTAB tensor files under {raw_dir}. "
            f"Contents: {list(raw_dir.rglob('*.pt'))}"
        )
    print(f"  Found tensors in {data_dir}")

    # Load tensors
    features = torch.load(data_dir / "features.pt", weights_only=False)
    edge_index = torch.load(data_dir / "edge_index.pt", weights_only=False)
    bot_labels = torch.load(data_dir / "labels_bot.pt", weights_only=False)

    # Load edge types if available (for info)
    edge_type_path = data_dir / "edge_type.pt"
    if edge_type_path.exists():
        edge_type = torch.load(edge_type_path, weights_only=False)
        unique_types = edge_type.unique()
        print(f"  Edge types: {len(unique_types)} ({unique_types.tolist()})")

    num_nodes = features.size(0)
    print(f"  Nodes:    {num_nodes:,}")
    print(f"  Features: {features.size(1)}")
    print(f"  Edges:    {edge_index.size(1):,} (raw)")

    # Ensure long type for edge_index
    edge_index = edge_index.long()

    # Make undirected and deduplicate
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    print(f"  Edges:    {edge_index.size(1):,} (undirected)")

    n_bots = (bot_labels == 1).sum().item()
    n_humans = (bot_labels == 0).sum().item()
    print(f"  Humans:   {n_humans:,} ({100*n_humans/num_nodes:.1f}%)")
    print(f"  Bots:     {n_bots:,} ({100*n_bots/num_nodes:.1f}%)")

    # Convert labels to PyGOD bit-encoding
    # Bot = structural anomaly (bit 1 set → value 2)
    y_pygod = torch.zeros(num_nodes, dtype=torch.long)
    y_pygod[bot_labels == 1] = 2

    # All nodes are labeled in MGTAB
    labeled_mask = torch.ones(num_nodes, dtype=torch.bool)

    new_data = Data(
        x=features.float(),
        edge_index=edge_index,
        y=y_pygod,
        num_nodes=num_nodes,
    )
    new_data.labeled_mask = labeled_mask

    out_path = Path(output_dir) / "mgtab.pt"
    torch.save(new_data, out_path)
    print(f"\nSaved to {out_path}")
    print(f"  Anomaly nodes (bots): {(y_pygod == 2).sum().item():,}")
    print(f"  Normal nodes (humans): {(y_pygod == 0).sum().item():,}")

    _print_verification(new_data)
    return out_path


def _find_mgtab_tensors(root: Path) -> Path | None:
    """Search for the directory containing MGTAB tensor files."""
    for p in root.rglob("features.pt"):
        d = p.parent
        if (d / "edge_index.pt").exists() and (d / "labels_bot.pt").exists():
            return d
    return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _print_verification(data: Data) -> None:
    """Print verification that the data matches PyGOD format."""
    print("\n--- Verification ---")
    print(f"  data.x.shape:          {tuple(data.x.shape)}")
    print(f"  data.edge_index.shape: {tuple(data.edge_index.shape)}")
    print(f"  data.y.shape:          {tuple(data.y.shape)}")
    print(f"  data.y unique values:  {data.y.unique().tolist()}")

    struct_anomalies = (data.y >> 1) & 1
    print(f"  Structural anomalies:  {struct_anomalies.sum().item():,}")
    print(f"  Normal nodes:          {(struct_anomalies == 0).sum().item():,}")

    if hasattr(data, "labeled_mask"):
        print(f"  Labeled nodes:         {data.labeled_mask.sum().item():,}")

    # Check edge_index validity
    assert data.edge_index.min() >= 0, "Negative node index in edge_index"
    assert data.edge_index.max() < data.num_nodes, "Node index exceeds num_nodes"
    print("  Format check:          PASSED")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare organic anomaly datasets in PyGOD-compatible format"
    )
    parser.add_argument(
        "--dataset",
        choices=["elliptic", "mgtab"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.expanduser("~"), ".pygod", "data"),
        help="Output directory (default: ~/.pygod/data/)",
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all or args.dataset == "elliptic":
        prepare_elliptic(args.output_dir)

    if args.all or args.dataset == "mgtab":
        prepare_mgtab(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
