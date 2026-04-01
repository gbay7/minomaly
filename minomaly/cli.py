"""Command-line entry point for Minomaly.

Usage::

    python -m minomaly --config config.yaml
    python -m minomaly --config config.yaml search.max_steps=9 search.max_freq=24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minomaly.config.loader import load_config, merge_cli_overrides
from minomaly.config.schema import MinomalyConfig


def parse_args(argv: list[str] | None = None) -> tuple[MinomalyConfig, list[str]]:
    parser = argparse.ArgumentParser(
        description="Minomaly — Unsupervised Graph Anomaly Detection",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable model training before detection.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train the model and exit (no detection).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-notation config overrides, e.g. search.max_steps=9",
    )

    args = parser.parse_args(argv)

    if args.config:
        config = load_config(args.config)
    else:
        config = MinomalyConfig()

    if args.train or args.train_only:
        config.training.enabled = True

    # Parse dot-notation overrides
    if args.overrides:
        override_dict = {}
        for override in args.overrides:
            if "=" not in override:
                print(f"Warning: ignoring override without '=': {override}", file=sys.stderr)
                continue
            key, val = override.split("=", 1)
            override_dict[key] = val
        config = merge_cli_overrides(config, override_dict)

    return config, args.overrides, args.train_only, args.config


def main(argv: list[str] | None = None) -> None:
    config, _, train_only, config_path = parse_args(argv)

    # Import here to avoid circular imports and speed up --help
    from minomaly.callbacks.logging_cb import LoggingCallback
    from minomaly.pipeline import MinomalyPipeline

    if train_only:
        pipeline = MinomalyPipeline(config)
        pipeline.add_callback(LoggingCallback())
        model = pipeline.train_only()
        print("\n=== Training Complete ===")
        print(f"  Model saved to: {config.training.checkpoint_dir}/model.pt")
        return

    pipeline = MinomalyPipeline(config, config_path=config_path)
    pipeline.add_callback(LoggingCallback())

    results = pipeline.run()

    print("\n=== Final Results ===")
    if "stat_results" in results:
        sr = results["stat_results"]
        for key in ("precision", "recall", "f1", "auroc", "ap"):
            if key in sr:
                print(f"  {key}: {sr[key]:.4f}")
    print(f"  verified: {results.get('verified_count', 0)}")
    print(f"  time: {results.get('total_time', 'N/A')}")


if __name__ == "__main__":
    main()
