"""Dataclass-based configuration schema for Minomaly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceConfig:
    """Device selection configuration."""

    device: str = "auto"


@dataclass
class ModelConfig:
    """GNN model architecture configuration."""

    conv_type: str = "SAGE"
    method_type: str = "order"  # "order" (Euclidean) or "poincare" (hyperbolic)
    n_layers: int = 8
    hidden_dim: int = 64
    skip: str = "learnable"
    dropout: float = 0.0
    margin: float = 0.1
    model_path: str = "ckpt/model.pt"
    input_dim: int = 2  # GLASS labeling: [anchor_indicator, inside_subgraph]
    curvature: float = 1.0           # Poincaré ball curvature init
    learnable_curvature: bool = True  # whether curvature is a learnable param
    n_heads: int = 4                 # attention heads for GATv2
    temperature: float = 0.07        # InfoNCE temperature for contrastive
    proj_dim: int = 64               # projection dimension for hybrid/DSAN
    neg_weight: float = 0.05         # negative repulsion weight for hybrid
    degree_normalize: bool = False   # degree-scaled readout before sum pool
    encoder_name: str = "skip_last_gnn"  # encoder from ENCODERS registry
    edge_n_layers: int = 4               # conv layers for edge GNN (edge_centric encoder)


@dataclass
class SamplingConfig:
    """Neighborhood sampling configuration."""

    n_neighborhoods: int = 10_000
    min_neighborhood_size: int = 1
    max_neighborhood_size: int = 30
    radial_radius: int = 2
    node_anchored: bool = True
    add_self_loop: bool = True


@dataclass
class SearchConfig:
    """Beam-search pattern growth configuration."""

    min_steps: int = 1
    max_steps: int = 7
    max_freq: float = 35.0
    outlier_max_freq: Optional[float] = None  # separate threshold for starting node detection; defaults to max_freq
    min_strength: float = 0.0
    max_unchanged: int = 5
    n_beams: int = 1
    max_cands: Optional[int] = None
    sample_random_cands: Optional[float] = None
    alpha: float = 0.33
    unchange_direction: bool = False
    scoring_function: str = "freq"
    verif_scoring_function: str = "freq_verif"
    structural_scoring: bool = False  # use multi-feature IF for anomaly scores
    search_strategy: str = "strength"
    add_verified_neighs: bool = False
    min_neigh_repeat: int = 2
    nodes_batch_size: int = 16
    min_subgraph_size: int = 1  # skip uninformative early steps (start beam at this size)
    sample_size: int = 500  # reference sample size for sampled search
    policy_model_path: Optional[str] = None  # trained RL policy for rl search
    rl_top_k: Optional[int] = None             # frontier candidates per step (None = all)
    rl_temperature: float = 1.0               # policy sampling temperature (rl search)
    vote_samples: int = 3                      # neighborhoods per frontier node per size (vote search)
    vote_sizes: Optional[List[int]] = None     # lookahead neighborhood sizes (vote search)
    vote_top_k: int = 5                        # top-voted frontier nodes to expand (vote search)
    calibrate: bool = False                    # auto-calibrate max_freq from freq_cache distribution
    score_mode: str = "min_freq"               # rarity search: "min_freq", "spike", "vote"
    valley_threshold: Optional[float] = None   # rarity search: auto-verify below this freq
    consensus: bool = False                    # rarity search: group-consensus AP boost
    cohesion_top_k: int = 10                   # cohesion search: frontier candidates ranked by internal edges


@dataclass
class OutlierConfig:
    """Outlier / starting-node detection configuration."""

    methods: List[str] = field(default_factory=lambda: ["model_based", "isolation_forest"])
    combine: str = "union"
    isolation_contamination: str = "auto"


@dataclass
class TrainingConfig:
    """Order-embedding GNN training configuration."""

    enabled: bool = False
    n_batches: int = 1_000_000
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "none"
    eval_interval: int = 1000
    val_size: int = 4096
    min_graph_size: int = 5
    max_graph_size: int = 29
    data_source: str = "synthetic"
    checkpoint_dir: str = "ckpt"
    node_anchored: bool = True
    clip_grad: float = 1.0
    include_structural: bool = False
    num_data_workers: int = 0  # >0 → parallel process-pool batch generation


@dataclass
class VisualizationConfig:
    """Visualization and output configuration."""

    enabled: bool = True
    reduction_method: Optional[str] = None
    output_dir: str = "plots"
    out_batch_size: int = 10


@dataclass
class DatasetConfig:
    """Dataset loading configuration."""

    name: str = "inj_cora"
    task: str = "struct-anomaly"
    cache_dir: Optional[str] = None
    injection: str = "none"
    injection_ratio: float = 0.05
    n_outliers: Optional[int] = None
    dice_perturb: float = 0.5
    group_size: int = 10
    drop_prob: float = 0.0
    anomaly_class: int = 0
    relation: str = "upu"  # fraud datasets: which relation (upu, usu, uvu, homo)


@dataclass
class MinomalyConfig:
    """Top-level configuration aggregating all sub-configs."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int = 42
    batch_size: int = 1000
