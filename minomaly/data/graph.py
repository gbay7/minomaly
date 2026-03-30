"""GPU-friendly graph representation for Minomaly.

Stores the full graph as a PyG-style edge_index tensor with a precomputed
CSR adjacency for O(1) neighbor lookups. NetworkX is used only at
visualization time via :meth:`GraphData.to_nx`.
"""

from __future__ import annotations

from typing import Optional

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce


class GraphData:
    """Thin wrapper around a PyG ``edge_index`` with fast neighbor access.

    All heavy tensors live on *device*, enabling GPU-accelerated subgraph
    operations, neighbor lookups, and feature construction.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = edge_index.device
        self.device = device
        self.num_nodes = num_nodes
        self.edge_index = edge_index.to(device)
        self.x = x.to(device) if x is not None else None
        self.y = y.to(device) if y is not None else None

        # Pre-compute CSR for O(1) neighbor lookups
        self._row_ptr: Optional[torch.Tensor] = None
        self._col: Optional[torch.Tensor] = None
        self._build_csr()

    # ── CSR construction ──────────────────────────────────────────────

    def _build_csr(self) -> None:
        """Build compressed sparse row pointers from edge_index."""
        src, dst = self.edge_index
        # Remove self-loops for neighbor queries
        mask = src != dst
        src_clean = src[mask]
        dst_clean = dst[mask]
        # Sort by source
        perm = src_clean.argsort()
        src_sorted = src_clean[perm]
        dst_sorted = dst_clean[perm]
        # Build row_ptr
        self._row_ptr = torch.zeros(
            self.num_nodes + 1, dtype=torch.long, device=self.device
        )
        ones = torch.ones_like(src_sorted)
        self._row_ptr.scatter_add_(0, src_sorted + 1, ones)
        self._row_ptr = self._row_ptr.cumsum(0)
        self._col = dst_sorted

    def neighbors(self, node: int) -> torch.Tensor:
        """Return 1-D tensor of neighbor node indices (excluding self-loops)."""
        start = self._row_ptr[node].item()
        end = self._row_ptr[node + 1].item()
        return self._col[start:end]

    def degree(self, node: int) -> int:
        return self._row_ptr[node + 1].item() - self._row_ptr[node].item()

    # ── Subgraph extraction ───────────────────────────────────────────

    def subgraph(self, nodes: torch.Tensor | list[int]) -> SubgraphView:
        """Return a lightweight subgraph view over *nodes*.

        This does NOT copy data; it creates an index mapping for the
        subgraph and filters edges on demand.
        """
        if isinstance(nodes, list):
            nodes = torch.tensor(nodes, dtype=torch.long, device=self.device)
        return SubgraphView(self, nodes)

    # ── Conversion ────────────────────────────────────────────────────

    def to_pyg(
        self,
        node_subset: Optional[torch.Tensor] = None,
        anchor: Optional[int] = None,
        node_anchored: bool = True,
        add_self_loop: bool = True,
    ) -> Data:
        """Convert (a subset of) this graph to a PyG Data object.

        If *node_subset* is given, extracts the induced subgraph and
        relabels nodes to 0..len(node_subset)-1.  The *anchor* is
        specified in the **original** node id space.
        """
        if node_subset is not None:
            node_set = set(node_subset.tolist())
            mask_src = torch.tensor(
                [s in node_set for s in self.edge_index[0].tolist()],
                device=self.device,
            )
            mask_dst = torch.tensor(
                [d in node_set for d in self.edge_index[1].tolist()],
                device=self.device,
            )
            mask = mask_src & mask_dst
            ei = self.edge_index[:, mask]
            # Relabel
            mapping = {old.item(): new for new, old in enumerate(node_subset)}
            src = torch.tensor([mapping[s.item()] for s in ei[0]], dtype=torch.long)
            dst = torch.tensor([mapping[d.item()] for d in ei[1]], dtype=torch.long)
            ei = torch.stack([src, dst])
            num_nodes = len(node_subset)
            anchor_idx = mapping[anchor] if anchor is not None else None
        else:
            ei = self.edge_index.clone()
            num_nodes = self.num_nodes
            anchor_idx = anchor

        # Optionally add self-loop on anchor
        if add_self_loop and anchor_idx is not None:
            sl = torch.tensor([[anchor_idx], [anchor_idx]], dtype=torch.long)
            ei = torch.cat([ei, sl], dim=1)
            ei = coalesce(ei, num_nodes=num_nodes)

        # Node features: GLASS labeling trick (ICLR 2022)
        # Channel 0: anchor indicator (1 for anchor, 0 otherwise)
        # Channel 1: inside-subgraph indicator (1 for all nodes in subgraph)
        # This gives the GNN boundary information, not just anchor position.
        x = torch.zeros(num_nodes, 2)
        x[:, 1] = 1.0  # all nodes in this subgraph are "inside"
        if node_anchored and anchor_idx is not None:
            x[anchor_idx, 0] = 1.0  # anchor indicator

        return Data(x=x, edge_index=ei, num_nodes=num_nodes)

    def to_nx(self) -> nx.Graph:
        """Convert to NetworkX for visualization only."""
        ei = self.edge_index.cpu()
        g = nx.Graph()
        g.add_nodes_from(range(self.num_nodes))
        edges = set()
        for i in range(ei.shape[1]):
            u, v = ei[0, i].item(), ei[1, i].item()
            if u != v:
                edges.add((min(u, v), max(u, v)))
        g.add_edges_from(edges)
        return g

    @staticmethod
    def from_pyg(data: Data, device: Optional[torch.device] = None) -> GraphData:
        """Create a GraphData from a PyG Data object."""
        ei = data.edge_index
        num_nodes = data.num_nodes or int(ei.max().item()) + 1
        return GraphData(
            edge_index=to_undirected(ei),
            num_nodes=num_nodes,
            x=getattr(data, "x", None),
            y=getattr(data, "y", None),
            device=device,
        )

    @staticmethod
    def from_nx(g: nx.Graph, device: Optional[torch.device] = None) -> GraphData:
        """Create a GraphData from a NetworkX graph."""
        g = nx.convert_node_labels_to_integers(g)
        num_nodes = g.number_of_nodes()
        edges = list(g.edges())
        if edges:
            src = [u for u, v in edges] + [v for u, v in edges]
            dst = [v for u, v in edges] + [u for u, v in edges]
            ei = torch.tensor([src, dst], dtype=torch.long)
            ei = coalesce(ei, num_nodes=num_nodes)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        return GraphData(edge_index=ei, num_nodes=num_nodes, device=device)


class SubgraphView:
    """Lightweight view into a parent GraphData for a node subset.

    Does not copy the parent's edge_index; instead, filters on access.
    Maintains a mapping from original node ids to local ids (0..n-1).
    """

    def __init__(self, parent: GraphData, nodes: torch.Tensor) -> None:
        self.parent = parent
        self.nodes = nodes  # original node ids
        self.num_nodes = len(nodes)
        self.device = parent.device

        # Build old→new mapping
        self._global_to_local: dict[int, int] = {
            old.item(): new for new, old in enumerate(nodes)
        }

        # Filter and relabel edges
        node_set = set(nodes.tolist())
        src, dst = parent.edge_index
        mask = torch.tensor(
            [(s.item() in node_set and d.item() in node_set)
             for s, d in zip(src, dst)],
            device=self.device,
        )
        if mask.any():
            filtered = parent.edge_index[:, mask]
            new_src = torch.tensor(
                [self._global_to_local[s.item()] for s in filtered[0]],
                dtype=torch.long, device=self.device,
            )
            new_dst = torch.tensor(
                [self._global_to_local[d.item()] for d in filtered[1]],
                dtype=torch.long, device=self.device,
            )
            self.edge_index = torch.stack([new_src, new_dst])
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

    def anchor_local(self, anchor_global: int) -> int:
        """Map a global anchor id to the local subgraph id."""
        return self._global_to_local[anchor_global]

    def to_pyg(
        self,
        anchor_global: Optional[int] = None,
        node_anchored: bool = True,
        add_self_loop: bool = True,
    ) -> Data:
        """Convert this subgraph view to a PyG Data object."""
        ei = self.edge_index.clone()
        anchor_local = None
        if anchor_global is not None:
            anchor_local = self._global_to_local.get(anchor_global)

        if add_self_loop and anchor_local is not None:
            sl = torch.tensor(
                [[anchor_local], [anchor_local]], dtype=torch.long, device=self.device,
            )
            ei = torch.cat([ei, sl], dim=1)
            ei = coalesce(ei, num_nodes=self.num_nodes)

        # GLASS labeling: [anchor_indicator, inside_subgraph]
        x = torch.zeros(self.num_nodes, 2, device=self.device)
        x[:, 1] = 1.0  # all nodes are inside this subgraph
        if node_anchored and anchor_local is not None:
            x[anchor_local, 0] = 1.0

        return Data(x=x, edge_index=ei, num_nodes=self.num_nodes)

    def to_nx(self) -> nx.Graph:
        """Convert to NetworkX for visualization only."""
        ei = self.edge_index.cpu()
        g = nx.Graph()
        g.add_nodes_from(range(self.num_nodes))
        edges = set()
        for i in range(ei.shape[1]):
            u, v = ei[0, i].item(), ei[1, i].item()
            if u != v:
                edges.add((min(u, v), max(u, v)))
        g.add_edges_from(edges)
        return g

    def global_nodes(self) -> list[int]:
        """Return original (global) node ids."""
        return self.nodes.tolist()
