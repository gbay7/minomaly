# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minomaly is an unsupervised graph anomaly detection framework. It discovers structurally anomalous subgraph patterns — connected subgraphs whose local topology is rare compared to the rest of the graph. It uses a pre-trained order-embedding GNN to measure subgraph containment in an embedding space, then runs beam search to grow candidate patterns until they reach an anomalous frequency band.

## Repository Layout

- `minomaly/` — **New modular package** (PyG-native, registry-based, GPU-accelerated)
- `code-original/` — Original source from github.com/gbay7/minomaly (reference only)
- `latex-project/` — Paper manuscript (Springer Nature template, `main.tex`)
- `configs/` — YAML configuration files (`default.yaml`, `train.yaml`)
- `Dockerfile` — Container for reproducible runs

## Running the New Package

```bash
# Run detection with default config
python -m minomaly --config configs/default.yaml

# Override parameters via CLI
python -m minomaly --config configs/default.yaml search.max_steps=9 search.max_freq=24

# Train a model from scratch
python -m minomaly --config configs/train.yaml --train

# Docker
docker build -t minomaly .
docker run --gpus all minomaly --config configs/default.yaml
```

Datasets (`inj_cora`, `inj_amazon`, `inj_flickr`) auto-download from PyGOD. Structural anomaly labels: `(data.y >> 1) & 1`.

### Dependencies

PyTorch 2.4+, PyG 2.6+, networkx, scikit-learn, scipy, matplotlib, tqdm, pyyaml, dacite. **No DeepSNAP dependency.**

## New Architecture (`minomaly/`)

### Registry Pattern

All swappable components register via decorators and are looked up by string key:

```python
from minomaly.registry import SAMPLERS, ENCODERS, EMBEDDERS, SCORING, SEARCH, OUTLIERS, GENERATORS, CALLBACKS
```

Ablation = change one string in YAML:
```yaml
search:
  scoring_function: harmonic  # swap "freq" → "harmonic"
model:
  conv_type: GIN              # swap "SAGE" → "GIN"
```

Available components:
- **SAMPLERS**: `tree`, `radial`
- **ENCODERS**: `skip_last_gnn`
- **EMBEDDERS**: `order`
- **SCORING**: `freq`, `weighted`, `harmonic`, `geometric`, `arithmetic`, `freq_verif`
- **SEARCH**: `strength`
- **GENERATORS**: `erdos_renyi`, `watts_strogatz`, `barabasi_albert`, `powerlaw_cluster`, `ensemble`
- **OUTLIERS**: `model_based`, `isolation_forest`
- **CALLBACKS**: `logging`, `visualization`, `evaluation`, `checkpoint`

### Module Map

| Module | Purpose |
|--------|---------|
| `registry.py` | Generic `Registry[T]` class with `@register`, `build()`, `list_available()` |
| `config/schema.py` | Dataclass configs: `MinomalyConfig`, `ModelConfig`, `SearchConfig`, etc. |
| `config/loader.py` | YAML loading, `dacite`-based dict→dataclass, CLI dot-notation overrides |
| `data/graph.py` | **`GraphData`** — GPU-native graph with CSR adjacency for O(1) neighbor lookups. **`SubgraphView`** — lightweight subgraph with edge filtering and relabeling. NX used only at visualization time via `.to_nx()` |
| `data/convert.py` | `nx_to_pyg()`, `batch_nx_graphs()`, `batch_pyg_data()` — pure PyG, replaces all 14 DeepSNAP touch points |
| `data/loaders.py` | `load_pygod_dataset()`, `extract_anomaly_labels()`, `pyg_data_to_nx()` |
| `models/` | `SkipLastGNN` (uses `data.x` not `data.node_feature`), `OrderEmbedder`, custom `SAGEConv`/`GINConv` |
| `generators/` | ER, WS, BA, PowerLaw, Ensemble — standalone (no DeepSNAP `dataset.Generator`) |
| `training/` | `TrainingPairGenerator`, `order_embedding_loss()`, `OrderEmbeddingTrainer` — full SPMiner-style training loop |
| `scoring/` | Pluggable scoring functions: `freq`, `weighted` (alpha-based), `harmonic`, `geometric`, `freq_verif` |
| `samplers/` | `TreeSampler`, `RadialSampler` — work with `GraphData`, produce PyG `Data` |
| `outliers/` | `ModelBasedDetector`, `IsolationForestDetector`, `CombinedDetector` |
| `search/` | `Beam`, `BeamSet` (GPU-batched scoring), `StrengthSearchAgent`, `PatternStore` (WL hash dedup) |
| `callbacks/` | `Callback` ABC → `LoggingCallback`, `VisualizationCallback`, `EvaluationCallback`, `CheckpointCallback` |
| `evaluation/metrics.py` | `get_stat_results()` — P/R/F1/AUC/AP computation |
| `visualization/` | `scatter_embs()`, `export_patterns()` |
| `pipeline.py` | `MinomalyPipeline` — end-to-end orchestrator replacing the monolithic `pattern_growth()` |
| `cli.py` | Entry point: `python -m minomaly` |

### GPU Acceleration

**Batch frequency computation** (biggest speedup vs original): `BeamSet.compute_all_scores()` stacks all beam embeddings `(N, D)` and computes pairwise violations against each embedding batch `(B, D)` in one tensor operation:

```python
diff = beam_embs.unsqueeze(1) - emb_batch.unsqueeze(0)  # (N, B, D)
violations = torch.sum(torch.clamp(diff, min=0) ** 2, dim=2)  # (N, B)
preds = model.clf_model(violations.unsqueeze(-1))  # (N, B, 2)
```

Replaces N sequential Python loops with a single GPU matmul.

### Graph Representation (`data/graph.py`)

The full graph is stored as `GraphData` with `edge_index` tensor and pre-computed CSR for O(1) neighbor lookups. NetworkX is used **only** in generators (for random graph creation) and visualization (`.to_nx()`). The beam search operates entirely on `GraphData` + tensor-based subgraph views.

### Callback System

Hooks fire at: search start/step/batch-end/end, training batch-end/epoch-end/end, embedding batch, outlier detection. Example:

```python
pipeline = MinomalyPipeline(config)
pipeline.add_callback(LoggingCallback())
pipeline.add_callback(EvaluationCallback(anomalous_nodes, all_nodes))
pipeline.add_callback(VisualizationCallback(output_dir="plots"))
results = pipeline.run()
```

### Bug Fixes vs Original

1. **Self-loop inconsistency**: Explicit `add_self_loop` parameter, consistent between OES construction and search.
2. **Dead alpha**: Scoring is a pluggable registry component — `freq` ignores alpha, `weighted` uses it.
3. **`beam.copy()` stale scores**: `copy()` creates fresh Beam with `score=None`, `freq=None`.
4. **`torch.load` without `weights_only`**: Fixed to use `weights_only=True` for model loading.
5. **Device management**: Tensors follow the model's device automatically.

## Adding a New Strategy

To add a new scoring function:
```python
# In minomaly/scoring/functions.py (or a new file)
from minomaly.registry import SCORING
from minomaly.scoring.base import ScoringFunction

@SCORING.register("my_score")
class MyScore(ScoringFunction):
    def __call__(self, freq, weight, alpha=0.5, last_score=float("inf")):
        return ...  # your formula
```

Then in YAML: `search.scoring_function: my_score`. Same pattern for samplers, encoders, search strategies, outlier detectors, etc.

## Building the Paper

Springer Nature class `sn-jnl.cls`. Compile `latex-project/main.tex` with `pdflatex` + `bibtex`.
