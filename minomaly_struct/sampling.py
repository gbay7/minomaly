import random

import networkx as nx
from tqdm import tqdm

from common import utils

class SamplingMethod:
     
    def __init__(self, graphs):
        self.graphs = graphs

    def sample(self):
        pass

    def name():
        return "sampling"

# TODO: I think it is not stable
class RadialSampling(SamplingMethod):

    def __init__(self, graphs, radius, subgraph_sample_size=0, nodes=None):
        super().__init__(graphs)
        self.radius = radius
        self.subgraph_sample_size = subgraph_sample_size
        self.nodes = nodes

    def sample(self, node_anchored=True):
        neighs, anchors, real_anchors = [], [], []

        for i, graph in enumerate(self.graphs):
            print(f"Radial sampling of graph {i}")
            nodes = graph.nodes if not self.nodes else self.nodes[i]
            for j in tqdm(range(len(nodes)), desc="Radial Sampling neighborhoods"):
                node = nodes[j]
                neigh = list(nx.single_source_shortest_path_length(graph,
                    node, cutoff=self.radius).keys())
                
                if self.subgraph_sample_size != 0:
                    neigh = [node] + random.sample(neigh, min(len(neigh),
                        self.subgraph_sample_size))
                
                if len(neigh) > 1:
                    real_anchors.append(node)
                    neigh = graph.subgraph(neigh)
                    # print(node, len(neigh))

                    if self.subgraph_sample_size != 0:
                        connected = nx.connected_components(neigh)
                        for c in connected:
                            if node in c:
                                neigh = neigh.subgraph(c)
                                break

                    mapping = {node: 0}
                    mapping.update({n: i+1 for i, n in enumerate(set(neigh.nodes) - {node})})
                    neigh = nx.relabel_nodes(neigh, mapping)
                    neigh.add_edge(0, 0)
                    neighs.append(neigh)
                    if node_anchored:
                        anchors.append(0)   # after converting labels, 0 will be anchor
        return neighs, anchors, real_anchors
    
    def name():
        return "radial"

class TreeSampling(SamplingMethod):

    def __init__(self, graphs, n_neighborhoods, min_neighborhood_size, max_neighborhood_size, nodes=None):
        super().__init__(graphs)
        self.n_neighborhoods = n_neighborhoods
        self.min_neighborhood_size = min_neighborhood_size
        self.max_neighborhood_size = max_neighborhood_size
        self.nodes = nodes

    def sample(self, node_anchored=True):
        neighs, anchors, real_anchors = [], [], []
        for _ in tqdm(range(self.n_neighborhoods), desc="Tree Sampling neighborhoods"):
            if not self.nodes:
                graph, neigh = utils.sample_neigh(self.graphs,
                    random.randint(self.min_neighborhood_size,
                        self.max_neighborhood_size))
            else:
                graph, neigh = utils.sample_neigh_from_nodes(self.graphs,
                random.randint(self.min_neighborhood_size,
                    self.max_neighborhood_size), self.nodes)
            anchor = neigh[0]
            real_anchors.append(anchor)
            neigh = graph.subgraph(neigh)
            mapping = {anchor: 0}
            mapping.update({n: i+1 for i, n in enumerate(set(neigh.nodes) - {anchor})})
            neigh = nx.relabel_nodes(neigh, mapping)
            neigh.add_edge(0, 0)
            neighs.append(neigh)
            if node_anchored:
                anchors.append(0)   # after converting labels, 0 will be anchor
        return neighs, anchors, real_anchors

    def name():
        return "tree"


class ExactTreeSampling(SamplingMethod):

    def __init__(self, graphs, min_neighborhood_size, max_neighborhood_size, nodes):
        super().__init__(graphs)
        self.min_neighborhood_size = min_neighborhood_size
        self.max_neighborhood_size = max_neighborhood_size
        self.nodes = nodes

    def sample(self, node_anchored=True):
        neighs, anchors, real_anchors = [], [], []
        for i, graph in enumerate(self.graphs):
            print(f"Tree sampling of graph {i}")
            for j in tqdm(range(len(self.nodes[i])), desc="Tree Sampling neighborhoods"):
                neigh = utils.sample_neigh_from_node(self.nodes[i][j], graph,
                    random.randint(self.min_neighborhood_size, self.max_neighborhood_size))
                anchor = neigh[0]
                real_anchors.append(anchor)
                neigh = graph.subgraph(neigh)
                mapping = {anchor: 0}
                mapping.update({n: i+1 for i, n in enumerate(set(neigh.nodes) - {anchor})})
                neigh = nx.relabel_nodes(neigh, mapping)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if node_anchored:
                    anchors.append(0)   # after converting labels, 0 will be anchor
        return neighs, anchors, real_anchors

    def name():
        return "tree"
