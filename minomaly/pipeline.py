"""MinomalyPipeline — end-to-end anomaly detection orchestrator."""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from minomaly.callbacks.base import Callback
from minomaly.callbacks.composite import CallbackList
from minomaly.config.schema import MinomalyConfig
from torch_geometric.data import Data
from minomaly.data.convert import batch_pyg_data
from minomaly.data.graph import GraphData
from minomaly.data.loaders import extract_anomaly_labels, load_fraud_dataset, load_molecules_dataset, load_organic_dataset, load_pyg_dataset, load_pygod_dataset, pyg_data_to_nx
from minomaly.evaluation.metrics import get_stat_results
from minomaly.registry import EMBEDDERS, ENCODERS, OUTLIERS, SAMPLERS, SCORING, SEARCH
from minomaly.search.beam import Beam
from minomaly.search.beam_set import BeamSet
from minomaly.utils.device import get_device, resolve_device
from minomaly.utils.seeding import set_deterministic
from minomaly.utils.serialization import save_json

CACHE_DIR = Path("/tmp/minomaly_savings")


class MinomalyPipeline:
    """End-to-end anomaly detection pipeline.

    Replaces the monolithic ``pattern_growth()`` function from ``decoder.py``.

    Usage::

        cfg = load_config("config.yaml")
        pipeline = MinomalyPipeline(cfg)
        pipeline.add_callback(LoggingCallback())
        pipeline.add_callback(EvaluationCallback(...))
        results = pipeline.run()
    """

    def __init__(self, config: MinomalyConfig, config_path: str | None = None) -> None:
        self.config = config
        self.config_path = config_path
        self.callbacks = CallbackList([])
        self.device = resolve_device(config.device.device)

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.callbacks.append(callback)

    def train_only(self):
        """Train the order embedding model and save checkpoint. No detection."""
        set_deterministic(self.config.seed)
        cfg = self.config
        cfg.training.enabled = True
        model = self._load_or_train_model()
        return model

    def calibrate_only(self) -> dict:
        """Run just the embedding step + max_freq calibration (no search)."""
        import random as _rnd
        _rnd.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        data, graphs, anomalies, all_nodes, anomalous_nodes = self._load_dataset()
        self._calib_data = data
        model = self._load_or_train_model()
        embs = self._load_or_generate_embeddings(model, graphs, anomalies)
        return self.calibrate_max_freq(model, graphs, embs)

    def _load_or_generate_embeddings(self, model, graphs, anomalies):
        """Load cached embeddings or generate + cache them."""
        cfg = self.config
        sc = cfg.sampling
        model_tag = Path(cfg.model.model_path).stem
        deg_tag = "_degnorm" if cfg.model.degree_normalize else ""
        inj_tag = ""
        if cfg.dataset.injection != "none":
            inj_parts = [cfg.dataset.injection, f"g{cfg.dataset.group_size}"]
            if cfg.dataset.n_outliers is not None:
                inj_parts.append(f"n{cfg.dataset.n_outliers}")
            inj_parts.append(f"s{cfg.seed}")
            inj_tag = "_" + "_".join(inj_parts)
        emb_key = (
            f"{cfg.dataset.name}_{sc.n_neighborhoods}_{sc.min_neighborhood_size}"
            f"_{sc.max_neighborhood_size}_{sc.node_anchored}_{cfg.batch_size}"
            f"_dim{cfg.model.input_dim}_{cfg.model.method_type}_{model_tag}{deg_tag}{inj_tag}"
        )
        emb_cache_pt = CACHE_DIR / "embeddings" / f"{emb_key}_embs.pt"

        if emb_cache_pt.exists():
            t0 = time.time()
            cached = torch.load(emb_cache_pt, map_location=self.device, weights_only=True)
            embs = list(cached['embs'].split(cfg.batch_size, dim=0))
            del cached
            print(f"[pre-search] Embeddings -> {self.device} in {time.time()-t0:.1f}s")
            return embs

        # Generate embeddings
        tree_sampler = SAMPLERS.build(
            "tree_csr",
            n_neighborhoods=sc.n_neighborhoods,
            min_size=sc.min_neighborhood_size,
            max_size=sc.max_neighborhood_size,
        )
        tree_result = tree_sampler.sample(
            graphs, node_anchored=sc.node_anchored, add_self_loop=sc.add_self_loop,
        )

        input_dim = cfg.model.input_dim
        for n in tree_result.neighborhoods:
            if n.x.shape[1] > input_dim:
                n.x = n.x[:, :input_dim]

        embs = self._embed_neighborhoods(
            model, tree_result.neighborhoods,
            tree_result.anchors, cfg.batch_size,
        )

        # Cache for future runs
        emb_cache_pt.parent.mkdir(parents=True, exist_ok=True)
        cat_embs_cpu = torch.cat(embs, dim=0).cpu()
        torch.save({'embs': cat_embs_cpu, 'anom_embs': torch.empty(0)}, emb_cache_pt)
        meta_cache = CACHE_DIR / "embeddings" / f"{emb_key}_meta.p"
        import pickle
        tree_result_data = {
            "neighborhoods": tree_result.neighborhoods,
            "anchors": tree_result.anchors,
            "real_anchors": tree_result.real_anchors,
            "node_lists": tree_result.node_lists,
        }
        with open(meta_cache, "wb") as f:
            pickle.dump(tree_result_data, f)
        del cat_embs_cpu
        print(f"[calibrate] Embedding cache saved to {emb_cache_pt}")
        return embs

    def run(self) -> dict:
        """Execute the full pipeline.

        1. Set deterministic seeds.
        2. Load dataset.
        3. Load or train model.
        4. Sample neighborhoods.
        5. Embed neighborhoods.
        6. Detect outlier starting nodes.
        7. Run beam search on batches of starting nodes.
        8. Deduplicate and export patterns.
        9. Compute and return metrics.
        """
        # Seed RNGs but do NOT enable torch.use_deterministic_algorithms —
        # it forces CPU fallbacks for scatter/index ops in GNN message
        # passing, making detection 100x slower.
        import random as _rnd
        _rnd.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        cfg = self.config

        # Experiment output directory: {config_name}_{timestamp}
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if self.config_path:
            experiment_name = Path(self.config_path).stem
        else:
            experiment_name = cfg.dataset.name
        plots_dir = Path(cfg.visualization.output_dir) / f"{experiment_name}_{timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "cluster" / "tracked_nodes").mkdir(parents=True, exist_ok=True)

        # ── 1. Load dataset ───────────────────────────────────────────
        data, graphs, anomalies, all_nodes, anomalous_nodes = self._load_dataset()
        self._calib_data = data
        num_nodes = len(all_nodes)

        # ── 2. Load or train model ────────────────────────────────────
        model = self._load_or_train_model()

        # ── 3. Sample neighborhoods (cached) ──────────────────────────
        sc = cfg.sampling
        model_tag = Path(cfg.model.model_path).stem
        deg_tag = "_degnorm" if cfg.model.degree_normalize else ""
        if cfg.dataset.injection != "none":
            inj_parts = [cfg.dataset.injection, f"g{cfg.dataset.group_size}"]
            if cfg.dataset.n_outliers is not None:
                inj_parts.append(f"n{cfg.dataset.n_outliers}")
            inj_parts.append(f"s{cfg.seed}")
            inj_tag = "_" + "_".join(inj_parts)
        else:
            inj_tag = ""
        emb_key = (
            f"{cfg.dataset.name}_{sc.n_neighborhoods}_{sc.min_neighborhood_size}"
            f"_{sc.max_neighborhood_size}_{sc.node_anchored}_{cfg.batch_size}"
            f"_dim{cfg.model.input_dim}_{cfg.model.method_type}_{model_tag}{deg_tag}{inj_tag}"
        )
        outlier_freq = cfg.search.outlier_max_freq if cfg.search.outlier_max_freq is not None else cfg.search.max_freq
        start_key = f"{emb_key}_ofreq{outlier_freq}"

        emb_cache_pt = CACHE_DIR / "embeddings" / f"{emb_key}_embs.pt"
        meta_cache = CACHE_DIR / "embeddings" / f"{emb_key}_meta.p"
        emb_cache_old = CACHE_DIR / "embeddings" / f"{emb_key}.p"
        start_cache = CACHE_DIR / "starting_nodes" / f"{start_key}_starting_nodes.p"
        need_meta = not start_cache.exists()

        tree_result_data = None

        if emb_cache_pt.exists():
            # V2 cache: pre-concatenated tensor, direct GPU load via map_location
            t0 = time.time()
            cached = torch.load(emb_cache_pt, map_location=self.device, weights_only=True)
            embs = list(cached['embs'].split(cfg.batch_size, dim=0))
            del cached
            print(f"[pre-search] Embeddings -> {self.device} in {time.time()-t0:.1f}s")

            if need_meta:
                if meta_cache.exists():
                    t0 = time.time()
                    with open(meta_cache, "rb") as f:
                        tree_result_data = pickle.load(f)
                    print(f"[pre-search] Metadata loaded in {time.time()-t0:.1f}s")
                elif emb_cache_old.exists():
                    t0 = time.time()
                    with open(emb_cache_old, "rb") as f:
                        cached_old = pickle.load(f)
                    tree_result_data = cached_old["tree"]
                    del cached_old
                    with open(meta_cache, "wb") as f:
                        pickle.dump(tree_result_data, f)
                    print(f"[pre-search] Metadata extracted from legacy in {time.time()-t0:.1f}s")

        elif emb_cache_old.exists():
            # Legacy single-file cache — load and migrate to v2
            t0 = time.time()
            with open(emb_cache_old, "rb") as f:
                cached = pickle.load(f)
            if need_meta:
                tree_result_data = cached["tree"]
            embs_cpu = cached["embs"]
            anom_embs_cpu = cached.get("anom_embs", [])
            del cached

            cat_embs = torch.cat(embs_cpu, dim=0)
            del embs_cpu
            embs = list(cat_embs.to(self.device).split(cfg.batch_size, dim=0))

            # Migrate to v2 for future runs
            emb_cache_pt.parent.mkdir(parents=True, exist_ok=True)
            cat_anom = torch.cat(anom_embs_cpu, dim=0) if anom_embs_cpu else torch.empty(0)
            torch.save({'embs': cat_embs, 'anom_embs': cat_anom}, emb_cache_pt)
            if need_meta and tree_result_data is not None:
                with open(meta_cache, "wb") as f:
                    pickle.dump(tree_result_data, f)
            del cat_embs, cat_anom, anom_embs_cpu
            print(f"[pre-search] Legacy cache loaded + migrated in {time.time()-t0:.1f}s")

        else:
            tree_sampler = SAMPLERS.build(
                "tree_csr",
                n_neighborhoods=sc.n_neighborhoods,
                min_size=sc.min_neighborhood_size,
                max_size=sc.max_neighborhood_size,
            )
            tree_result = tree_sampler.sample(
                graphs, node_anchored=sc.node_anchored, add_self_loop=sc.add_self_loop,
            )

            nx_graphs = [g.to_nx() for g in graphs]
            radial_sampler = SAMPLERS.build(
                "radial_fast",
                radius=sc.radial_radius,
                subsample_size=cfg.search.max_steps,
                nodes=[[i for i, a in enumerate(anom) if a] for anom in anomalies],
            )
            anom_result = radial_sampler.sample(
                graphs, node_anchored=sc.node_anchored, add_self_loop=sc.add_self_loop,
                nx_graphs=nx_graphs,
            )
            del nx_graphs

            input_dim = cfg.model.input_dim
            for n in tree_result.neighborhoods:
                if n.x.shape[1] > input_dim:
                    n.x = n.x[:, :input_dim]
            for n in anom_result.neighborhoods:
                if n.x.shape[1] > input_dim:
                    n.x = n.x[:, :input_dim]

            # ── 4. Embed neighborhoods ────────────────────────────
            embs = self._embed_neighborhoods(
                model, tree_result.neighborhoods,
                tree_result.anchors, cfg.batch_size,
            )
            anom_embs = self._embed_neighborhoods(
                model, anom_result.neighborhoods,
                anom_result.anchors, len(anom_result.anchors) or 1,
            )

            tree_result_data = {
                "neighborhoods": tree_result.neighborhoods,
                "anchors": tree_result.anchors,
                "real_anchors": tree_result.real_anchors,
                "node_lists": tree_result.node_lists,
            }

            # Save V2 cache: embeddings (.pt) separate from metadata (.p)
            emb_cache_pt.parent.mkdir(parents=True, exist_ok=True)
            cat_embs_cpu = torch.cat(embs, dim=0).cpu()
            cat_anom_cpu = torch.cat(anom_embs, dim=0).cpu()
            torch.save({'embs': cat_embs_cpu, 'anom_embs': cat_anom_cpu}, emb_cache_pt)
            with open(meta_cache, "wb") as f:
                pickle.dump(tree_result_data, f)
            del cat_embs_cpu, cat_anom_cpu, anom_embs
            print(f"[pre-search] Cache saved to {emb_cache_pt.parent}")

        # ── 4b. Calibrate max_freq ────────────────────────────────────
        if cfg.search.calibrate:
            self.calibrate_max_freq(model, graphs, embs)

        # ── 5. Detect starting nodes (cached) ────────────────────────
        outlier_freq = cfg.search.outlier_max_freq if cfg.search.outlier_max_freq is not None else cfg.search.max_freq
        outlier_freq_norm = outlier_freq / num_nodes
        max_freq_norm = cfg.search.max_freq / num_nodes

        if start_cache.exists():
            print(f"Loading cached starting nodes from {start_cache}")
            with open(start_cache, "rb") as f:
                cached_start = pickle.load(f)
            starting_nodes = cached_start["starting_nodes"]
            outlier_embs_np = cached_start["outlier_embs_np"]
            del cached_start
        else:
            if tree_result_data is None:
                raise RuntimeError(
                    f"Starting node cache missing and metadata not available. "
                    f"Delete {emb_cache_pt} to regenerate."
                )
            embs_np = torch.cat(embs, dim=0).cpu().numpy()
            starting_nodes, outlier_embs_np = self._detect_starting_nodes(
                embs, model, outlier_freq_norm,
                tree_result_data["real_anchors"],
                tree_result_data["neighborhoods"],
                cfg.search.max_steps, embs_np,
            )
            del embs_np, tree_result_data
            start_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving starting nodes cache to {start_cache}")
            with open(start_cache, "wb") as f:
                pickle.dump({
                    "starting_nodes": starting_nodes,
                    "outlier_embs_np": outlier_embs_np,
                }, f)

        self.callbacks.on_outliers_detected(
            starting_nodes=starting_nodes,
            method_counts={"total": len(starting_nodes)},
        )

        # ── 6. Beam search ────────────────────────────────────────────
        # Re-seed so search results are identical whether detection was
        # cached or computed fresh (IsolationForest consumes np.random).
        import random as _rnd
        _rnd.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        scorer = SCORING.build(cfg.search.scoring_function)
        max_strength = scorer(
            max_freq_norm,
            cfg.search.max_steps / num_nodes,
            cfg.search.alpha,
        )

        verified: set[Beam] = set()
        count_copied = {}
        total_time = timedelta(0)
        anomalous_nodes_flat = [n for sublist in anomalous_nodes for n in sublist]

        for graph_idx, graph in enumerate(graphs):
            graph_nodes = set(range(graph.num_nodes))
            starting_graph = list(starting_nodes & graph_nodes)

            for batch, batch_num, batch_start in _batch_nodes(
                starting_graph, cfg.search.nodes_batch_size,
            ):
                # Skip already-verified anchors
                batch = [n for n in batch if n not in {b.anchor() for b in verified}]
                if not batch:
                    continue

                agent = SEARCH.build(
                    cfg.search.search_strategy,
                    model=model,
                    graphs=graphs,
                    embs=embs,
                    scorer=scorer,
                    node_anchored=sc.node_anchored,
                    add_self_loop=sc.add_self_loop,
                    n_beams=cfg.search.n_beams,
                    min_strength=cfg.search.min_strength,
                    max_strength=max_strength,
                    alpha=cfg.search.alpha,
                    max_unchanged=cfg.search.max_unchanged,
                    unchange_direction=cfg.search.unchange_direction,
                    min_steps=cfg.search.min_steps,
                    max_steps=cfg.search.max_steps,
                    max_cands=cfg.search.max_cands,
                    sample_random_cands=cfg.search.sample_random_cands,
                    add_verified_neighs=cfg.search.add_verified_neighs,
                    min_neigh_repeat=cfg.search.min_neigh_repeat,
                    input_dim=cfg.model.input_dim,
                    freq_cache=getattr(self, "_freq_cache", None),
                    sample_size=cfg.search.sample_size,
                    min_subgraph_size=cfg.search.min_subgraph_size,
                    policy_model_path=cfg.search.policy_model_path,
                    rl_top_k=cfg.search.rl_top_k,
                    rl_temperature=cfg.search.rl_temperature,
                    vote_samples=cfg.search.vote_samples,
                    vote_sizes=cfg.search.vote_sizes,
                    vote_top_k=cfg.search.vote_top_k,
                    score_mode=cfg.search.score_mode,
                    valley_threshold=cfg.search.valley_threshold,
                    consensus=cfg.search.consensus,
                    cohesion_top_k=cfg.search.cohesion_top_k,
                    **({"anomalous_nodes": anomalous_nodes_flat}
                       if cfg.search.search_strategy == "diagnostic" else {}),
                )

                self.callbacks.on_search_start(batch, cfg.search)
                agent_verified = agent.run(
                    batch, graph_idx=graph_idx, callbacks=self.callbacks.callbacks,
                )

                verified.update(set(agent_verified))

                # Handle copied verified neighbors
                if cfg.search.add_verified_neighs:
                    for beam in agent.copied_verified:
                        entry = count_copied.get(beam.anchor(), (None, 0))
                        count_copied[beam.anchor()] = (beam, entry[1] + 1)
                    to_remove = []
                    for anchor, (beam, count) in count_copied.items():
                        if count > cfg.search.min_neigh_repeat:
                            verified.add(beam)
                            to_remove.append(anchor)
                    for anchor in to_remove:
                        del count_copied[anchor]

                batch_time = timedelta(seconds=time.time() - batch_start)
                total_time += batch_time

                # Per-batch logging (matches original decoder.py)
                batch_stats = get_stat_results(anomalous_nodes_flat, verified, all_nodes,
                                               structural_scoring=cfg.search.structural_scoring)
                batch_log = {
                    "stats": {
                        "stat_results": batch_stats,
                        "predicted_anomalies_len": len(agent_verified),
                        "batch_number": batch_num,
                        "batch_time": str(batch_time),
                        "total_time": str(total_time),
                        "cumulative_verified": len(verified),
                        "copied_pending": len(count_copied),
                    },
                    "anomalies": [
                        {**b.to_dict(), "is_true": b.anchor() in set(anomalous_nodes_flat)}
                        for b in agent_verified
                    ],
                }
                save_json(batch_log, plots_dir / f"batch_{batch_num}_anomalies.json")

                self.callbacks.on_search_batch_end(
                    batch_number=batch_num,
                    batch_verified=list(agent_verified),
                    cumulative_verified=verified,
                    elapsed_time=batch_time.total_seconds(),
                )

        # ── 7. Compute final metrics ──────────────────────────────────
        stat_results = get_stat_results(anomalous_nodes_flat, verified, all_nodes,
                                        structural_scoring=cfg.search.structural_scoring)

        results = {
            "dataset": cfg.dataset.name,
            "stat_results": stat_results,
            "verified_count": len(verified),
            "starting_nodes_count": len(starting_nodes),
            "true_anomalies": anomalous_nodes_flat,
            "true_anomalies_len": len(anomalous_nodes_flat),
            "search_params": {
                "search_strategy": cfg.search.search_strategy,
                "max_freq": cfg.search.max_freq,
                "max_steps": cfg.search.max_steps,
                "min_steps": cfg.search.min_steps,
                "max_unchanged": cfg.search.max_unchanged,
                "alpha": cfg.search.alpha,
                "n_beams": cfg.search.n_beams,
                "max_cands": cfg.search.max_cands,
                "add_verified_neighs": cfg.search.add_verified_neighs,
                "min_neigh_repeat": cfg.search.min_neigh_repeat,
                "nodes_batch_size": cfg.search.nodes_batch_size,
            },
            "total_time": str(total_time),
            "gpu_memory": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            "all_verified_beams": [
                {**b.to_dict(), "is_true": b.anchor() in set(anomalous_nodes_flat)}
                for b in verified
            ],
        }

        # Save results
        save_json(results, plots_dir / "anomalies.json")

        # Auto-analysis: generate diagnostic plots
        from minomaly.analysis.experiment import run_analysis
        run_analysis(results, plots_dir, anomalous_nodes_flat)

        # Interpretation: visualize detected anomalous structures
        from minomaly.analysis.interpretation import interpret_patterns
        interpret_patterns(
            list(verified), graphs[0], set(anomalous_nodes_flat),
            plots_dir / "interpretation",
        )

        self.callbacks.on_search_end(
            all_verified=verified,
            patterns=[],
            stats=results,
        )

        return results

    # ── Internal helpers ──────────────────────────────────────────────

    def _load_dataset(self):
        """Load dataset and convert to GraphData."""
        cfg = self.config.dataset
        _ORGANIC = {"elliptic", "mgtab"}
        _PYG_DATASETS = {
            "cora", "citeseer", "pubmed",
            "amazon_computers",
            "squirrel", "chameleon",
            "actor",
        }

        _MOLECULE_DATASETS = {"aids", "mutag", "nci1", "nci109", "ptc_mr",
                               "imdb-binary", "imdb-multi", "enzymes",
                               "proteins", "dd", "reddit-binary", "collab"}

        _FRAUD_DATASETS = {"fraud_amazon", "fraud_yelp"}

        _PYGOD_DATASETS = {"weibo", "reddit", "enron", "disney"}

        if cfg.name.startswith("inj_") or cfg.name in _PYGOD_DATASETS:
            data, _ = load_pygod_dataset(cfg.name, cache_dir=cfg.cache_dir)
        elif cfg.name in _FRAUD_DATASETS:
            data, _ = load_fraud_dataset(
                cfg.name, relation=cfg.relation, cache_dir=cfg.cache_dir,
            )
        elif cfg.name in _ORGANIC:
            data, _ = load_organic_dataset(cfg.name, cache_dir=cfg.cache_dir)
        elif cfg.name in _MOLECULE_DATASETS:
            data, _ = load_molecules_dataset(
                cfg.name,
                anomaly_class=cfg.anomaly_class,
                n_anomaly_molecules=cfg.n_outliers or 20,
                cache_dir=cfg.cache_dir,
                seed=self.config.seed,
            )
        elif cfg.name in _PYG_DATASETS:
            data, _ = load_pyg_dataset(
                cfg.name,
                cache_dir=cfg.cache_dir,
                injection=cfg.injection,
                injection_ratio=cfg.injection_ratio,
                dice_perturb=cfg.dice_perturb,
                group_size=cfg.group_size,
                drop_prob=cfg.drop_prob,
                seed=self.config.seed,
                n_outliers=cfg.n_outliers,
            )
        else:
            data, _ = load_molecules_dataset(
                cfg.name,
                anomaly_class=cfg.anomaly_class,
                n_anomaly_molecules=cfg.n_outliers or 20,
                cache_dir=cfg.cache_dir,
                seed=self.config.seed,
            )

        anomaly_labels = extract_anomaly_labels(data, task=cfg.task)
        nx_graph = pyg_data_to_nx(data)
        graph = GraphData.from_nx(nx_graph, device=self.device)
        graphs = [graph]
        anomalies = [anomaly_labels.tolist()]
        all_nodes = list(range(graph.num_nodes))
        anomalous_nodes = [
            [i for i, a in enumerate(anomaly_labels.tolist()) if a]
        ]
        return data, graphs, anomalies, all_nodes, anomalous_nodes

    def _load_or_train_model(self):
        """Load pre-trained model or train from scratch."""
        cfg = self.config
        method = cfg.model.method_type  # "order" or "poincare"

        # Build embedder-specific kwargs
        embedder_kwargs = dict(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            margin=cfg.model.margin,
        )
        encoder_name = cfg.model.encoder_name
        edge_kwargs = {"edge_n_layers": cfg.model.edge_n_layers} if encoder_name == "edge_centric" else {}
        if method == "poincare":
            embedder_kwargs.update(
                curvature=cfg.model.curvature,
                learnable_curvature=cfg.model.learnable_curvature,
                encoder_name=encoder_name,
                n_layers=cfg.model.n_layers,
                conv_type=cfg.model.conv_type,
                skip=cfg.model.skip,
                dropout=cfg.model.dropout,
                **edge_kwargs,
            )
        elif method == "contrastive":
            embedder_kwargs.update(
                temperature=cfg.model.temperature,
                encoder_name=encoder_name if encoder_name != "skip_last_gnn" else "gatv2",
                n_layers=cfg.model.n_layers,
                n_heads=cfg.model.n_heads,
                dropout=cfg.model.dropout,
                **edge_kwargs,
            )
        elif method == "hybrid":
            embedder_kwargs.update(
                positive_only=False,
                encoder_name=encoder_name,
                n_layers=cfg.model.n_layers,
                conv_type=cfg.model.conv_type,
                skip=cfg.model.skip,
                dropout=cfg.model.dropout,
                degree_normalize=cfg.model.degree_normalize,
                **edge_kwargs,
            )
        else:
            embedder_kwargs.update(
                encoder_name=encoder_name,
                n_layers=cfg.model.n_layers,
                conv_type=cfg.model.conv_type,
                skip=cfg.model.skip,
                dropout=cfg.model.dropout,
                degree_normalize=cfg.model.degree_normalize,
                **edge_kwargs,
            )

        model = EMBEDDERS.build(method, **embedder_kwargs)
        model.to(self.device)
        model.eval()

        if cfg.training.enabled:
            from minomaly.generators.ensemble import build_default_ensemble
            from minomaly.training.data_gen import TrainingPairGenerator
            from minomaly.training.trainer import OrderEmbeddingTrainer

            # Fine-tune: load existing weights if model_path exists
            model_path = cfg.model.model_path
            if os.path.exists(model_path):
                print(f"Fine-tuning from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)

            ensemble = build_default_ensemble(
                np.arange(cfg.training.min_graph_size + 1, cfg.training.max_graph_size + 1),
                include_structural=cfg.training.include_structural,
            )
            data_gen = TrainingPairGenerator(
                generator=ensemble,
                min_size=cfg.training.min_graph_size,
                max_size=cfg.training.max_graph_size,
                node_anchored=cfg.training.node_anchored,
            )
            trainer = OrderEmbeddingTrainer(
                model=model,
                train_config=cfg.training,
                data_generator=data_gen,
                callbacks=self.callbacks.callbacks,
            )
            trainer.train()
            return model

        # Load pre-trained
        model_path = cfg.model.model_path
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                f"Set training.enabled=True to train from scratch."
            )
        return model

    def _embed_neighborhoods(
        self, model, neighborhoods, anchors, batch_size,
    ) -> list[torch.Tensor]:
        """Batch-embed neighborhoods. Uses embed_and_project if available."""
        embs = []
        n = len(neighborhoods)
        has_project = hasattr(model, "embed_and_project")
        emb_fn = model.embed_and_project if has_project else model.emb_model
        for i in tqdm(range(0, n, batch_size), desc="Embedding neighborhoods"):
            top = min(n, i + batch_size)
            batch_data = neighborhoods[i:top]
            batch = batch_pyg_data(batch_data, device=self.device)
            with torch.no_grad():
                emb = emb_fn(batch)
            embs.append(emb)
            self.callbacks.on_embedding_batch(i // batch_size, n // batch_size)
        return embs

    def _detect_starting_nodes(
        self, embs, model, freq_thresh,
        real_anchors, neighborhoods, max_steps, embs_np,
    ) -> tuple[set[int], np.ndarray]:
        """Detect starting nodes using configured outlier detectors."""
        import time as _t

        methods = self.config.outlier.methods
        print(f"[detect] Building detectors... freq_thresh={freq_thresh:.6f}, methods={methods}")

        sets_and_embs: list[tuple[set, np.ndarray]] = []

        if "model_based" in methods:
            model_det = OUTLIERS.build("model_based", freq_thresh=freq_thresh)
            print(f"[detect] Running model-based detector...")
            _t0 = _t.time()
            s1, e1 = model_det.detect(
                embs, model, real_anchors, neighborhoods,
            )
            self._freq_cache = getattr(model_det, "freq_cache", None)
            if self._freq_cache is not None:
                print(f"[detect] Cached {len(self._freq_cache)} neighborhood frequencies for Phase 3")
            print(f"[detect] Model-based done in {_t.time()-_t0:.1f}s: {len(s1)} nodes, e1 shape={e1.shape}")
            sets_and_embs.append((s1, e1))

        if "isolation_forest" in methods:
            contam = self.config.outlier.isolation_contamination
            iforest_det = OUTLIERS.build(
                "isolation_forest",
                contamination=float(contam) if contam != "auto" else "auto",
                max_neigh_len=max_steps,
            )
            print(f"[detect] Running IsolationForest...")
            _t0 = _t.time()
            s2, e2 = iforest_det.detect(
                embs, model, real_anchors, neighborhoods,
                embs_np=embs_np,
            )
            print(f"[detect] IsolationForest done in {_t.time()-_t0:.1f}s: {len(s2)} nodes, e2 shape={e2.shape}")
            sets_and_embs.append((s2, e2))

        if not sets_and_embs:
            raise ValueError(f"No valid outlier methods in {methods}")

        if len(sets_and_embs) == 1:
            starting, all_embs = sets_and_embs[0]
        else:
            all_sets = [s for s, _ in sets_and_embs]
            if self.config.outlier.combine == "union":
                starting = all_sets[0] | all_sets[1]
            else:
                starting = all_sets[0] & all_sets[1]
            emb_list = [e for _, e in sets_and_embs if len(e) > 0]
            all_embs = np.concatenate(emb_list) if emb_list else np.empty((0, 0))

        print(f"[detect] Total starting nodes: {len(starting)}")
        return starting, all_embs


    def calibrate_max_freq(
        self,
        model: nn.Module,
        graphs: list[GraphData],
        embs: list[torch.Tensor],
    ) -> dict:
        """Calibrate max_freq from the OES containment-frequency distribution.

        Formula: max_freq = percentile(freq_cache, 40 - 2*max_steps) * num_nodes

        The freq_cache captures each OES neighborhood's containment frequency.
        Higher max_steps means larger beams that are naturally rarer, so the
        percentile threshold decreases linearly with search depth.
        """
        from minomaly.search.beam_set import _precompute_clf_threshold

        cfg = self.config
        num_nodes = graphs[0].num_nodes
        max_steps = cfg.search.max_steps

        all_embs = torch.cat(embs, dim=0).to(self.device)
        N = all_embs.shape[0]

        threshold = _precompute_clf_threshold(model.clf_model, self.device)
        freq_counts = torch.zeros(N, device=self.device)
        chunk_size = 512

        print(f"\n[calibrate] Computing freq_cache ({N} neighborhoods)...")
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            query_chunk = all_embs[start:end]
            for emb_batch in embs:
                eb = emb_batch.to(self.device)
                violations = model.batch_predict(eb, query_chunk)
                supergraphs = violations < threshold
                freq_counts[start:end] += supergraphs.sum(dim=1).float()

        freq_ratios = freq_counts / N
        freq_np = freq_ratios.cpu().numpy()

        P = max(1, 40 - 2 * max_steps)
        calib_norm = float(np.percentile(freq_np, P))
        calib_max_freq = calib_norm * num_nodes

        old_max_freq = cfg.search.max_freq
        old_norm = old_max_freq / num_nodes

        print(f"[calibrate] P = 40 - 2*{max_steps} = {P}")
        print(f"  Calibrated: max_freq = {calib_max_freq:.1f}  (norm={calib_norm:.6f})")
        if old_max_freq > 0:
            ratio = calib_max_freq / old_max_freq
            print(f"  Config:     max_freq = {old_max_freq:.1f}  (norm={old_norm:.6f})")
            print(f"  Ratio: {ratio:.2f}x")

        cfg.search.max_freq = calib_max_freq
        print(f"  → Applied: max_freq={calib_max_freq:.1f}")

        return {
            "calibrated_max_freq": calib_max_freq,
            "calibrated_norm": calib_norm,
            "formula_pct": P,
            "old_max_freq": old_max_freq,
            "n_total": N,
        }


def _batch_nodes(nodes: list[int], batch_size: int):
    """Yield (batch, batch_number, start_time) tuples."""
    import random
    random.shuffle(nodes)
    it = iter(nodes)
    batch_num = 0
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        batch_num += 1
        yield batch, batch_num, time.time()
