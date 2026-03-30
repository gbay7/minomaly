"""PatternStore — deduplication of verified patterns via WL hash."""

from __future__ import annotations

import random
from collections import defaultdict

from minomaly.hashing import wl_hash
from minomaly.search.beam import Beam


class PatternStore:
    """Groups verified patterns by (size, WL_hash) for deduplication.

    Replaces the inline ``defaultdict(lambda: defaultdict(list))`` pattern
    from the original ``decoder.py``.
    """

    def __init__(self, node_anchored: bool = True) -> None:
        self.node_anchored = node_anchored
        # counts[pattern_size][wl_hash_tuple] = list of (subgraph_view, anchor)
        self._counts: dict[int, dict[tuple, list]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add(self, beam: Beam) -> None:
        """Add a verified beam's subgraph pattern to the store."""
        import torch
        nodes_t = torch.tensor(beam.neigh, dtype=torch.long, device=beam.graph.device)
        view = beam.graph.subgraph(nodes_t)
        nx_g = view.to_nx()
        # Set anchor attribute for WL hash
        if self.node_anchored:
            anchor_local = view.anchor_local(beam.anchor())
            for v in nx_g.nodes:
                nx_g.nodes[v]["anchor"] = 1 if v == anchor_local else 0
        h = wl_hash(nx_g, node_anchored=self.node_anchored)
        size = view.num_nodes
        self._counts[size][h].append(
            (view, beam.anchor() if self.node_anchored else None)
        )

    def merge(self, other: PatternStore) -> None:
        """Merge another store into this one."""
        for size, hash_dict in other._counts.items():
            for h, entries in hash_dict.items():
                self._counts[size][h].extend(entries)

    def get_unique_patterns(self, max_per_size: int = 10) -> list[tuple]:
        """Return deduplicated patterns: one random example per WL hash.

        Returns list of (subgraph_view, anchor) tuples, at most
        *max_per_size* per pattern size.
        """
        results = []
        for size in sorted(self._counts.keys()):
            sorted_hashes = sorted(
                self._counts[size].items(),
                key=lambda kv: len(kv[1]),
                reverse=True,
            )
            for _, entries in sorted_hashes[:max_per_size]:
                results.append(random.choice(entries))
        return results

    @property
    def total_patterns(self) -> int:
        return sum(
            len(entries)
            for hash_dict in self._counts.values()
            for entries in hash_dict.values()
        )

    @property
    def unique_count(self) -> int:
        return sum(len(hash_dict) for hash_dict in self._counts.values())
