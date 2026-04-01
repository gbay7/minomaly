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
from tqdm import tqdm

from minomaly.callbacks.base import Callback
from minomaly.callbacks.composite import CallbackList
from minomaly.config.schema import MinomalyConfig
from minomaly.data.convert import batch_pyg_data
from minomaly.data.graph import GraphData
from minomaly.data.loaders import extract_anomaly_labels, load_organic_dataset, load_pygod_dataset, pyg_data_to_nx
from minomaly.evaluation.metrics import get_stat_results
from minomaly.registry import EMBEDDERS, ENCODERS, OUTLIERS, SAMPLERS, SCORING, SEARCH
from minomaly.search.beam import Beam
from minomaly.search.beam_set import BeamSet
from minomaly.utils.device import get_device, resolve_device
from minomaly.utils.seeding import set_deterministic
from minomaly.utils.serialization import save_json

CACHE_DIR = Path("savings")


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
        num_nodes = len(all_nodes)

        # ── 2. Load or train model ────────────────────────────────────
        model = self._load_or_train_model()

        # ── 3. Sample neighborhoods (cached) ──────────────────────────
        sc = cfg.sampling
        emb_key = (
            f"{cfg.dataset.name}_{sc.n_neighborhoods}_{sc.min_neighborhood_size}"
            f"_{sc.max_neighborhood_size}_{sc.node_anchored}_{cfg.batch_size}"
            f"_dim{cfg.model.input_dim}_{cfg.model.method_type}"
        )
        outlier_freq = cfg.search.outlier_max_freq if cfg.search.outlier_max_freq is not None else cfg.search.max_freq
        start_key = f"{emb_key}_ofreq{outlier_freq}"
        emb_cache = CACHE_DIR / "embeddings" / f"{emb_key}.p"
        start_cache = CACHE_DIR / "starting_nodes" / f"{start_key}_starting_nodes.p"

        if emb_cache.exists():
            print(f"Loading cached sampling + embeddings from {emb_cache}")
            with open(emb_cache, "rb") as f:
                cached = pickle.load(f)
            tree_result_data = cached["tree"]
            anom_result_data = cached["anom"]
            embs = cached["embs"]
            anom_embs = cached["anom_embs"]
            # Move tensors to device
            embs = [e.to(self.device) for e in embs]
            anom_embs = [e.to(self.device) for e in anom_embs]
        else:
            tree_sampler = SAMPLERS.build(
                "tree_fast",
                n_neighborhoods=sc.n_neighborhoods,
                min_size=sc.min_neighborhood_size,
                max_size=sc.max_neighborhood_size,
            )
            tree_result = tree_sampler.sample(
                graphs, node_anchored=sc.node_anchored, add_self_loop=sc.add_self_loop,
            )

            radial_sampler = SAMPLERS.build(
                "radial_fast",
                radius=sc.radial_radius,
                subsample_size=cfg.search.max_steps,
                nodes=[[i for i, a in enumerate(anom) if a] for anom in anomalies],
            )
            anom_result = radial_sampler.sample(
                graphs, node_anchored=sc.node_anchored, add_self_loop=sc.add_self_loop,
            )

            # Slice features to match model's input_dim
            input_dim = cfg.model.input_dim
            for n in tree_result.neighborhoods:
                if n.x.shape[1] > input_dim:
                    n.x = n.x[:, :input_dim]
            for n in anom_result.neighborhoods:
                if n.x.shape[1] > input_dim:
                    n.x = n.x[:, :input_dim]

            # ── 4. Embed neighborhoods ────────────────────────────────
            embs = self._embed_neighborhoods(
                model, tree_result.neighborhoods,
                tree_result.anchors, cfg.batch_size,
            )
            anom_embs = self._embed_neighborhoods(
                model, anom_result.neighborhoods,
                anom_result.anchors, len(anom_result.anchors) or 1,
            )

            # Cache to disk (move to CPU for pickling)
            tree_result_data = {
                "neighborhoods": tree_result.neighborhoods,
                "anchors": tree_result.anchors,
                "real_anchors": tree_result.real_anchors,
                "node_lists": tree_result.node_lists,
            }
            anom_result_data = {
                "neighborhoods": anom_result.neighborhoods,
                "anchors": anom_result.anchors,
                "real_anchors": anom_result.real_anchors,
                "node_lists": anom_result.node_lists,
            }
            emb_cache.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving sampling + embeddings cache to {emb_cache}")
            with open(emb_cache, "wb") as f:
                pickle.dump({
                    "tree": tree_result_data,
                    "anom": anom_result_data,
                    "embs": [e.cpu() for e in embs],
                    "anom_embs": [e.cpu() for e in anom_embs],
                }, f)

        # ── 5. Detect starting nodes (cached) ────────────────────────
        # Outlier detection uses a separate (wider) threshold to cast a wide net.
        # Verification uses max_freq (tighter) to only verify truly anomalous patterns.
        outlier_freq = cfg.search.outlier_max_freq if cfg.search.outlier_max_freq is not None else cfg.search.max_freq
        outlier_freq_norm = outlier_freq / num_nodes
        max_freq_norm = cfg.search.max_freq / num_nodes
        embs_np = torch.cat(embs, dim=0).cpu().numpy()

        if start_cache.exists():
            print(f"Loading cached starting nodes from {start_cache}")
            with open(start_cache, "rb") as f:
                cached_start = pickle.load(f)
            starting_nodes = cached_start["starting_nodes"]
            outlier_embs_np = cached_start["outlier_embs_np"]
        else:
            starting_nodes, outlier_embs_np = self._detect_starting_nodes(
                embs, model, outlier_freq_norm,
                tree_result_data["real_anchors"],
                tree_result_data["neighborhoods"],
                cfg.search.max_steps, embs_np,
            )
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

        # ── 5b. Contextual embedding (optional) ─────────────────────
        context_scorer = None
        if cfg.context.enabled:
            node_lists = tree_result_data.get("node_lists", [])
            if not node_lists:
                print("[context] WARNING: node_lists not in cache. "
                      "Delete cache and re-run to enable contextual detection.")
            else:
                from minomaly.context.embedder import ContextEmbedder
                from minomaly.context.clustering import ContextClustering
                from minomaly.context.scorer import ContextualScorer

                print("[context] Computing contextual embeddings...")
                ctx_embedder = ContextEmbedder(
                    input_dim=data.x.shape[1],
                    hidden_dim=cfg.context.hidden_dim,
                    encoder_name=cfg.context.encoder,
                    n_layers=cfg.context.n_layers,
                    conv_type=cfg.context.conv_type,
                    skip=cfg.context.skip,
                    dropout=cfg.context.dropout,
                ).to(self.device)

                ctx_embs = ctx_embedder.embed_neighborhoods(
                    tree_result_data["neighborhoods"],
                    node_lists,
                    data.x,
                    device=self.device,
                )
                print(f"[context] Context embeddings: {ctx_embs.shape}")

                print(f"[context] Clustering into {cfg.context.n_clusters} contexts...")
                clustering = ContextClustering(cfg.context.n_clusters, cfg.context.clustering)
                ctx_labels, ctx_centers = clustering.fit(ctx_embs)

                # Precompute threshold for contextual scorer
                from minomaly.search.adaptive_search import _precompute_threshold
                ctx_threshold = _precompute_threshold(model.clf_model, self.device)

                context_scorer = ContextualScorer(
                    structural_embs=torch.cat(embs, dim=0),
                    context_labels=ctx_labels,
                    n_clusters=cfg.context.n_clusters,
                    beta=cfg.context.beta,
                    threshold=ctx_threshold,
                )
                print(f"[context] Contextual scorer ready (beta={cfg.context.beta})")

        # ── 6. Beam search ────────────────────────────────────────────
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
                    context_scorer=context_scorer,
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
                batch_stats = get_stat_results(anomalous_nodes_flat, verified, all_nodes)
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
        stat_results = get_stat_results(anomalous_nodes_flat, verified, all_nodes)

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

        if cfg.name.startswith("inj_"):
            data, _ = load_pygod_dataset(cfg.name, cache_dir=cfg.cache_dir)
        elif cfg.name in _ORGANIC:
            data, _ = load_organic_dataset(cfg.name, cache_dir=cfg.cache_dir)
        else:
            raise ValueError(f"Unknown dataset: {cfg.name}")

        anomaly_labels = extract_anomaly_labels(data)
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
        if method == "poincare":
            embedder_kwargs.update(
                curvature=cfg.model.curvature,
                learnable_curvature=cfg.model.learnable_curvature,
                encoder_name="skip_last_gnn",
                n_layers=cfg.model.n_layers,
                conv_type=cfg.model.conv_type,
                skip=cfg.model.skip,
                dropout=cfg.model.dropout,
            )
        elif method == "contrastive":
            embedder_kwargs.update(
                temperature=cfg.model.temperature,
                encoder_name="gatv2",
                n_layers=cfg.model.n_layers,
                n_heads=cfg.model.n_heads,
                dropout=cfg.model.dropout,
            )
        elif method == "hybrid":
            # Use skip_last_gnn with configurable conv_type (GIN, SAGE, etc.)
            embedder_kwargs.update(
                positive_only=False,
                encoder_name="skip_last_gnn",
                n_layers=cfg.model.n_layers,
                conv_type=cfg.model.conv_type,
                skip=cfg.model.skip,
                dropout=cfg.model.dropout,
            )
        else:
            embedder_kwargs.update(
                encoder_name="skip_last_gnn",
                n_layers=cfg.model.n_layers,
                conv_type=cfg.model.conv_type,
                skip=cfg.model.skip,
                dropout=cfg.model.dropout,
            )

        model = EMBEDDERS.build(method, **embedder_kwargs)
        model.to(self.device)
        model.eval()

        if cfg.training.enabled:
            from minomaly.generators.ensemble import build_default_ensemble
            from minomaly.training.data_gen import TrainingPairGenerator
            from minomaly.training.trainer import OrderEmbeddingTrainer

            ensemble = build_default_ensemble(
                np.arange(cfg.training.min_graph_size + 1, cfg.training.max_graph_size + 1)
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

        print(f"[detect] Building detectors... freq_thresh={freq_thresh:.6f}")
        model_det = OUTLIERS.build("model_based", freq_thresh=freq_thresh)
        iforest_det = OUTLIERS.build(
            "isolation_forest",
            contamination=self.config.outlier.isolation_contamination,
            max_neigh_len=max_steps,
        )

        print(f"[detect] Running model-based detector...")
        _t0 = _t.time()
        s1, e1 = model_det.detect(
            embs, model, real_anchors, neighborhoods,
        )
        # Cross-phase frequency cache (Idea 3): reuse Phase 2 frequencies
        self._freq_cache = getattr(model_det, "freq_cache", None)
        if self._freq_cache is not None:
            print(f"[detect] Cached {len(self._freq_cache)} neighborhood frequencies for Phase 3")
        print(f"[detect] Model-based done in {_t.time()-_t0:.1f}s: {len(s1)} nodes, e1 shape={e1.shape}")

        print(f"[detect] Running IsolationForest...")
        _t0 = _t.time()
        s2, e2 = iforest_det.detect(
            embs, model, real_anchors, neighborhoods,
            embs_np=embs_np,
        )
        print(f"[detect] IsolationForest done in {_t.time()-_t0:.1f}s: {len(s2)} nodes, e2 shape={e2.shape}")

        if self.config.outlier.combine == "union":
            starting = s1 | s2
        else:
            starting = s1 & s2

        print(f"[detect] Combining: {len(starting)} total starting nodes")
        all_embs = np.concatenate([e1, e2]) if len(e1) > 0 and len(e2) > 0 else (
            e1 if len(e1) > 0 else e2
        )
        return starting, all_embs


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
