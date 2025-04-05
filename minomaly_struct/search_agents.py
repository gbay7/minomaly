import json
import numpy as np
import torch
from tqdm import tqdm

from common import utils

import random
from collections import defaultdict
import networkx as nx

def harmonic_score(freq, weight):
    return (freq * weight) / (freq + weight)
def geometric_score(freq, weight):
        return (freq * weight) ** 0.5
def arithmetic_score(freq, weight):
        return (freq + weight) / 2
    
def freq_score(freq):
        return freq
def freq_verif_score(freq, last_freq, weight):
        return freq + weight * (freq - last_freq)

def old_score(freq, weight, alpha=0.5):
        return alpha * freq + (1 - alpha) * weight
# def drop_score(freq, weight, last_score):
#     delta_score = (freq - last_score) if last_score != float("inf") else 0
#     return freq_score(freq) + weight * delta_score

# TODO: instead of node: make nodes, because the more nodes in candidates, the better
class Beam:

    @staticmethod
    def calculate_score(freq, weight, alpha=0.5, last_score=float("inf")):
        return freq_score(freq)
        # return drop_score(freq, weight, last_score)
        # return old_score(freq, weight, alpha)
        
    @staticmethod
    def calculate_verif_score(freq, weight, alpha=0.5, last_score=float("inf")):
        return freq_verif_score(freq, last_score, weight)
        # return drop_score(freq, weight, last_score)
        # return old_score(freq, weight, alpha)

    def __init__(self, node, neigh:list=None, frontier=None, visited=None, graph:nx.Graph=None, total_weight=None, unchange=0, last_score=float("inf")):
        """Beam class for the Beam Search algorithm.

        Args:
            - node: the node to be added to the subgraph pattern.
            - neigh: the current subgraph pattern.
            - frontier: the set of nodes that are candidates to be added to the pattern.
            - visited: the set of nodes that have been visited.
            - graph: the graph from which the pattern is being mined.
            - total_weight: the total weight of the graph or dataset of graphs.
            - unchange: the number of steps the strength of the source pattern has not improved. 
            (TODO: calculate the gradient of the strength score to quantify the amount of changing)
            - last_score: the strength score of the source pattern in the last step.
        """

        self.node = node
        self.graph = graph
        self.neigh = neigh + [node] if neigh is not None else None
        self.anchored_neigh = None

        self.frontier = list(((set(frontier) | set(graph.neighbors(node))) - visited) - set([node])) if frontier is not None else None
        self.visited = visited | set([node]) if visited is not None else None
        self.emb = None

        self.score = None
        self.freq = None
        self.weight = len(self.neigh) / (len(graph) if total_weight is None else total_weight) if self.neigh is not None else None

        self.unchange = unchange
        self.last_score = last_score
        self.original = True

    def copy(self, node):
        beam = Beam(node)
        beam.node = node
        beam.graph = self.graph
        beam.neigh = [node] + self.neigh[:-1]
        beam.anchored_neigh = self.anchored_neigh
        beam.frontier = self.frontier
        beam.visited = self.visited
        beam.emb = self.emb
        beam.score = self.score
        beam.freq = self.freq
        beam.weight = self.weight
        beam.unchange = self.unchange
        beam.last_score = self.last_score
        beam.original = False
        return beam

    def to_dict(self):
        return {
            "anchor": self.anchor(),
            "added_node": self.node,
            "neigh": self.neigh,
            "neigh_len": len(self.neigh),
            "frontier": self.frontier,
            "frontier_len": len(self.frontier),
            "weight": self.weight,
            "freq": self.freq,
            "score": self.score,
            "last_score": self.last_score,
            "unchange": self.unchange,
            "original": self.original,
            "is_true": self.is_true if hasattr(self, "is_true") else None,
        }

    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return str(self.to_dict())

    def gen_cand_beams(self, total_weight=None, sample_random_cands=None, max_cands=None) -> list:
        frontier = self.frontier
        if sample_random_cands is not None:
            frontier = random.sample(frontier, round(sample_random_cands * len(frontier)))
        if max_cands is not None and len(frontier) > max_cands:
            frontier = random.sample(frontier, max_cands)
        return [Beam(cand_node, self.neigh, self.frontier, self.visited, self.graph, total_weight=total_weight, unchange=self.unchange, last_score=self.score if self.score is not None else self.last_score) 
                    for cand_node in frontier]
    
    def get_anchored_neigh(self):
        if self.anchored_neigh is not None:
            return self.anchored_neigh
        neigh_g: nx.Graph = self.graph.subgraph(self.neigh).copy()
        neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
        for v in neigh_g.nodes:
            neigh_g.nodes[v]["anchor"] = 1 if v == self.anchor() else 0
        # TODO: prefer time or space ?
        # self.anchored_neigh = neigh_g
        return neigh_g
    
    def get_neigh(self):
        return self.graph.subgraph(self.neigh)

    def anchor(self):
        return self.neigh[0]

    def __eq__(self, other) :
        if not isinstance(other, Beam):
            return False
        return self.anchor() == other.anchor()

    def __hash__(self):
        return hash(self.anchor())

    def strength(self, embs, model, alpha=0.5, unchange_direction=False):
        """Compute the strength score of the beam.
        Args:
            - embs: the embeddings of the sampled node neighborhoods.
            - model: the subgraph matching model.
            - alpha: the importance of the frequency score in the strength score.
            - unchange_direction: whether to consider the increasing/decreasing unchange of the strength score. False means increasing unchange, True otherwise.
        """
        freq, n_embs = 0, 0
        # check how many the beam is subgraph of (frequency) in the order embedding space
        for emb_batch in embs:
            n_embs += len(emb_batch)
            supergraphs = torch.argmax(
                model.clf_model(model.predict((
                emb_batch.to(utils.get_device()),
                self.emb)).unsqueeze(1)), axis=1)
            freq += torch.sum(supergraphs).item()
        self.freq = freq / n_embs
        self.score = Beam.calculate_score(self.freq, self.weight, alpha, self.last_score)

        if unchange_direction:
            if self.score <= self.last_score:
                self.unchange += 1
            else:
                self.unchange = 0
        else:
            if self.score >= self.last_score:
                self.unchange += 1
            else:
                self.unchange = 0
        
        return self.score
    
    def verif_score(self, alpha=0.5):
        return Beam.calculate_verif_score(self.freq, self.weight, alpha, last_score=self.last_score)
    
    def embed(self, emb_model, node_anchored=True):
        """Embed the beam.
        - Note: the embedding is not stored in the beam object.
        """
        if self.emb is not None:
            return self.emb
        return emb_model(utils.batch_nx_graphs([self.get_anchored_neigh()], anchors=[self.anchor()] if node_anchored
                        else None)).squeeze(0)

    def pruned(self, min_strength, max_strength, max_unchanged):
        if self.score > max_strength and self.unchange > max_unchanged:
            # print('Pruned (unchange > max_unchanged):', self)
            return True
        if self.score < min_strength:
            # print('Pruned (score < min_strength):', self)
            return True
        return False
    
    def is_verified(self, min_strength, max_strength):
        verified = min_strength <= self.score <= max_strength
        # if verified: print('Verified', self)
        return verified
    
    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

class BeamSet(list):

    def __init__(self, beams:list):
        """BeamSet class for the Beam Search algorithm.

        Args:
            - beams: the list of beams in the beam set.
        """
        super(BeamSet, self).__init__(beams)

    def sort(self, top_k=1, node_votes=None):
        if len(self) == 0:
            return self
        
        # Sort by score
        super().sort(key=lambda x: x.score)
        
        if node_votes is None:
            self[:] = self[:top_k]
            return self
        
        # Find the minimum score
        min_score = self[0].score
        
        # Collect all items with the minimum score
        min_score_items = [item for item in self if item.score == min_score]
        
        # Update node votes
        for item in min_score_items:
            node_votes[item.node] = node_votes.get(item.node, 0) + 1
        
        # Sort the items with the minimum score by their votes
        min_score_items.sort(key=lambda x: node_votes[x.node], reverse=True)
        
        # Get the top k items
        self[:] = min_score_items[:top_k]
        
        return self

    def embed(self, emb_model, node_anchored=True):
        """Embed all the beams in the beam set."""
        neighs = [beam.get_neigh() for beam in self]
        anchors = [beam.anchor() for beam in self]
        neighs_embs = emb_model(utils.batch_nx_graphs(
                    neighs, anchors=anchors if node_anchored else None))
        for beam, emb in zip(self, neighs_embs):
            beam.emb = emb.detach()
        return self

    def compute_scores(self, embs, model, alpha=0.5, unchange_direction=False):
        """Compute the strength scores of all the beams in the beam set."""
        beam: Beam
        for beam in self:
            beam.strength(embs, model, alpha, unchange_direction)
        return self
    
    def prune(self, min_strength, max_strength, max_unchanged):
        """Prune the beams that are hopeless."""
        self[:] = [beam for beam in self if not beam.pruned(min_strength, max_strength, max_unchanged)]
        return self

    def extract_verified(self, min_strength, max_strength):
        """Extract and prune the beams that are verified."""
        # print('Verified beams scores:', [beam.score for beam in self])
        verified_beams = BeamSet([beam for beam in self if beam.is_verified(min_strength, max_strength)])
        self[:] = [beam for beam in self if not beam in verified_beams]
        return verified_beams
    
    def get_verified_neighs(self, min_strength, max_strength):
        """Add the verified neighbors of the verified beams."""
        verified_beams = [beam for beam in self if beam.is_verified(min_strength, max_strength)]
        verified_neighs = set([beam.copy(node) for beam in verified_beams for node in beam.neigh if node != beam.anchor()])
        return BeamSet(verified_neighs)
    
    def add_pattern_results(self, counts, node_anchored=True):
        beam: Beam
        for beam in self:
            neigh_g: nx.Graph = beam.get_anchored_neigh()
            # cand_patterns[len(neigh_g)].append((beam.score, neigh_g))
            counts[len(neigh_g)][utils.wl_hash(neigh_g,
                node_anchored=node_anchored)].append((neigh_g, beam.anchor() if node_anchored else None))
            
    def analyze(self, emb_model, analyze_embs_cur, node_anchored=True):
        beam: Beam
        for beam in self:
            emb = beam.embed(emb_model, node_anchored)
            analyze_embs_cur.append(emb.detach().cpu().numpy())

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def defaultdict_list():
    return defaultdict(list)
class StrengthSearchAgent:
    def __init__(self, max_unchanged, max_strength, model, dataset,
        embs, node_anchored=False, analyze=False, n_beams=1, min_strength=0, alpha=0.5, sample_random_cands=None, max_cands=None, unchange_direction=False, min_steps=None, max_steps=None, add_verified_neighs=False, track_node=None, n_threads=1):
        """Strength computation implementation of the subgraph pattern search.
        At every step, the algorithm chooses greedily the next node to grow while the pattern
        remains predicted to be frequent. The criteria to choose the next action depends
        on the score predicted by the subgraph matching model.
        The algorithm terminates when the strength of the pattern is below a certain threshold
        or when the strength has not improved for a certain number of steps.

        Args:
            - model: the trained subgraph matching model (PyTorch nn.Module).
            - dataset: the DeepSNAP dataset for which to mine the frequent subgraph pattern.
            - embs: embeddings of sampled node neighborhoods (see paper).
            - node_anchored: an option to specify whether to identify node_anchored subgraph patterns.
                node_anchored search procedure has to use a node_anchored model (specified in subgraph
                matching config.py).
            - analyze: whether to enable analysis visualization.
            - max_unchanged: the maximum number of steps the strength of the pattern
                has not improved.
            - max_strength: the maximum strength score of the pattern.
            - min_strength: the minimum strength score of the pattern.
            - n_beams: the number of beams to consider at each step.
            - alpha: the importance of the frequency score in the strength score.
            - sample_random_cands: the ratio of random candidates to sample at each step (if None, all will be considered).
            - max_cands: the maximum number of candidates to consider at each step.
            - unchange_direction: whether to consider the increasing/decreasing unchange of the strength score. False means increasing unchange, True otherwise.
            - max_steps: the maximum number of steps to run the search.
            - min_steps: the minimum number of steps to consider a pattern verified.
            - add_verified_neighs: whether to add the verified neighbors of the verified beams.
        """
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.n_beams = n_beams
        self.max_unchanged = max_unchanged
        self.unchange_direction = unchange_direction
        self.max_strength = max_strength
        self.min_strength = min_strength
        self.alpha = alpha
        self.num_nodes = sum([len(g) for g in self.dataset])
        self.sample_random_cands = sample_random_cands
        self.max_cands = max_cands
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.add_verified_neighs = add_verified_neighs
        self.track_node = track_node
        self.node_votes = {}
        self.n_threads = n_threads
        self.lock = threading.Lock()

    def run_search(self, nodes: list, graph_idx=0):
        self.counts = defaultdict(defaultdict_list)
        self.verified = BeamSet([])
        self.copied_verified = BeamSet([])
        self.analyze_embs = []

        self.init_search(nodes, graph_idx=graph_idx)

        self.steps = 1
        with ThreadPoolExecutor(self.n_threads) as executor:
            while not self.is_search_done():
                self.steps += 1
                self.step(executor)
                if self.max_steps is not None and self.steps >= self.max_steps:
                    break
        return self.finish_search()

    def init_search(self, nodes: list, graph_idx=0):
        self.beam_sets = []
        graph = self.dataset[graph_idx]
        for node in nodes:
            beam_set: BeamSet = BeamSet([Beam(int(node), [], [], set(), graph, total_weight=self.num_nodes)])
            # NOTE: maybe we don't have to take the top_k in sorting
            # beam_set.embed(self.model.emb_model, self.node_anchored).compute_scores(self.embs, self.model, alpha=self.alpha).prune(min_strength=self.min_strength, max_strength=self.max_strength, max_unchanged=self.max_unchanged).sort(self.n_beams)
            self.beam_sets.append(beam_set)

            if self.track_node is not None:
                self.track_node(beam_set, 'init_search')

    def is_search_done(self):
        return len(self.beam_sets) == 0

    def process_beam_sets(self, beam_sets, step):
        results = []
        node_votes = {}
        for beam_set in beam_sets:
            new_beams = BeamSet([])
            for beam in beam_set:
                if not beam.frontier:
                    continue
                cands = beam.gen_cand_beams(total_weight=self.num_nodes, sample_random_cands=self.sample_random_cands, max_cands=self.max_cands)
                new_beams += BeamSet(cands)

            if len(new_beams) > 0:
                new_beams.embed(self.model.emb_model, self.node_anchored).compute_scores(self.embs, self.model, alpha=self.alpha, unchange_direction=self.unchange_direction).prune(min_strength=self.min_strength, max_strength=self.max_strength, max_unchanged=self.max_unchanged).sort(self.n_beams, node_votes=node_votes)
                verified = new_beams.extract_verified(self.min_strength, self.max_strength)
                if (self.min_steps is not None and step < self.min_steps):
                    verified = BeamSet([])
                results.append((new_beams, verified, node_votes))
            else:
                results.append((new_beams, BeamSet([]), node_votes))
        return results
    
    def step(self, executor):
        """Run a step of the search algorithm.
        A step means expanding all the beams of the beams sets by one candidate node"""
        new_beam_sets = []
        if self.analyze: analyze_embs_cur = []

        # Batch beam sets into groups of 3 for processing
        batch_size = 1
        beam_set_batches = [self.beam_sets[i:i + batch_size] for i in range(0, len(self.beam_sets), batch_size)]

        futures = [executor.submit(self.process_beam_sets, batch, self.steps) for batch in beam_set_batches]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Step {self.steps}"):
            results = future.result()
            for new_beams, verified, node_votes in results:
                with self.lock:
                    verified.add_pattern_results(self.counts, node_anchored=self.node_anchored)
                    self.verified += verified
                    for node, vote in node_votes.items():
                        self.node_votes[node] = self.node_votes.get(node, 0) + vote

                    if self.track_node is not None:
                        if len(verified) > 0:
                            self.track_node(verified, 'verified', {'step': self.steps})
                        else:
                            self.track_node(new_beams, 'unverified', {'step': self.steps})

                    if self.add_verified_neighs:
                        verified_neighs = verified.get_verified_neighs(self.min_strength, self.max_strength)
                        # self.verified += verified_neighs
                        self.copied_verified += verified_neighs

                if len(new_beams) > 0:
                    if self.analyze:
                        new_beams.analyze(self.model.emb_model, analyze_embs_cur, node_anchored=self.node_anchored)
                    new_beam_sets.append(new_beams)

        self.beam_sets = new_beam_sets
        if self.analyze: self.analyze_embs += analyze_embs_cur

    def finish_search(self):
        return self.verified