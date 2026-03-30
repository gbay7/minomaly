"""Weisfeiler-Lehman graph hashing and vector hashing utilities.

Ported from ``code-original/common/utils.py`` without any DeepSNAP
dependency.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import networkx as nx
import numpy as np

_cached_masks: List[int] | None = None


def vec_hash(v: np.ndarray | list) -> List[int]:
    """XOR-based hash of an integer vector using cached random masks.

    The random masks are generated once (seeded with ``2019``) and cached
    globally.  Their length is determined by the length of the first
    vector passed to this function.

    Parameters
    ----------
    v:
        A 1-D sequence of hashable values (typically integers).

    Returns
    -------
    list[int]
        A list of the same length as *v* with each element XOR-ed
        against its corresponding random mask.
    """
    global _cached_masks
    if _cached_masks is None:
        random.seed(2019)
        _cached_masks = [random.getrandbits(32) for _ in range(len(v))]
    return [hash(v[i]) ^ mask for i, mask in enumerate(_cached_masks)]


def wl_hash(
    g: nx.Graph,
    dim: int = 64,
    node_anchored: bool = False,
) -> Tuple[int, ...]:
    """Compute a Weisfeiler-Lehman style hash of a graph.

    The algorithm iterates ``len(g)`` rounds of neighbour-label
    aggregation (using :func:`vec_hash`) and returns the element-wise
    sum of all final node vectors as a tuple.

    Parameters
    ----------
    g:
        The input graph.  Node labels are re-indexed to consecutive
        integers internally.
    dim:
        Dimensionality of the hash vectors.
    node_anchored:
        If ``True``, the initial vector for the anchor node (identified
        by ``g.nodes[v]["anchor"] == 1``) is set to all ones; all other
        nodes start at zero.

    Returns
    -------
    tuple[int, ...]
        A *dim*-length tuple of integers serving as a canonical hash.
    """
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=int)

    if node_anchored:
        for v in g.nodes:
            if g.nodes[v].get("anchor") == 1:
                vecs[v] = 1
                break

    for _ in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=np.int64)
        for n in g.nodes:
            newvecs[n] = vec_hash(
                np.sum(vecs[list(g.neighbors(n)) + [n]], axis=0)
            )
        vecs = newvecs

    return tuple(np.sum(vecs, axis=0))
