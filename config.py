"""Typed experiment configuration and JSON inheritance, independent of notebooks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CLIENTS = ("electricity", "steam", "cooling")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    source: str = "prepared"
    root: str = "data"
    raw_csv: str = "AUS Dataset2016-2019.csv"
    train_csv: str = "data_train_asu.csv"
    test_csv: str = "data_test_asu.csv"
    history: int = 168
    horizon: int = 24
    stride: int = 24
    train_years: list[int] = field(default_factory=lambda: [2016, 2017, 2018])
    test_years: list[int] = field(default_factory=lambda: [2019])
    weather: str = "target_observed"
    validation: str = "chronological"
    validation_fraction: float = 0.1
    split_seed: int = 42
    purge_hours: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None


@dataclass
class ModelConfig:
    encoder_hidden: list[int] = field(default_factory=lambda: [64])
    representation_dim: int = 64
    server_hidden_dim: int = 192
    server_tail_hidden: list[int] = field(default_factory=list)
    server_output_dim: int = 192
    decoder_hidden: list[int] = field(default_factory=lambda: [64])
    baseline_hidden: list[int] = field(default_factory=lambda: [64, 64, 64])


@dataclass
class TrainingConfig:
    mode: str = "vfl"
    seed: int = 42
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    device: str = "cpu"
    deterministic: bool = True
    num_threads: int = 1
    task_weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(CLIENTS, 1.0))
    dcl_alpha: dict[str, float] = field(default_factory=lambda: dict.fromkeys(CLIENTS, 0.05))
    dcor_epsilon: float = 1e-12
    mape_epsilon: float = 1e-8
    eval_dcor_max_samples: int = 1024


@dataclass
class CryptoConfig:
    backend: str = "plaintext"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        d, m, t = self.data, self.model, self.training
        if d.source not in {"raw", "prepared"}:
            raise ValueError("data.source must be raw or prepared")
        if d.weather not in {"target_observed", "history"}:
            raise ValueError("data.weather must be target_observed or history")
        if d.source == "prepared" and d.weather != "target_observed":
            raise ValueError("Historical weather requires data.source=raw")
        if d.source == "prepared" and (d.history > 168 or d.horizon > 24 or d.stride != 24):
            raise ValueError("Bundled prepared CSVs support history<=168, horizon<=24, stride=24")
        if not 0 < d.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if d.validation not in {"chronological", "random"} or d.purge_hours < 0:
            raise ValueError("Invalid validation strategy or purge_hours")
        if d.validation == "random" and d.purge_hours:
            raise ValueError("purge_hours is supported only for chronological validation")
        if not d.train_years or not d.test_years or max(d.train_years) >= min(d.test_years):
            raise ValueError("Training years must precede test years")
        sizes = [
            d.history,
            d.horizon,
            d.stride,
            m.representation_dim,
            m.server_hidden_dim,
            m.server_output_dim,
            t.epochs,
            t.batch_size,
            t.patience,
            t.num_threads,
            t.eval_dcor_max_samples,
            *m.encoder_hidden,
            *m.server_tail_hidden,
            *m.decoder_hidden,
            *m.baseline_hidden,
        ]
        if not m.baseline_hidden or any(type(x) is not int or x <= 0 for x in sizes):
            raise ValueError("Layer sizes and iteration counts must be positive integers")
        for cap in (d.max_train_samples, d.max_val_samples, d.max_test_samples):
            if cap is not None and (type(cap) is not int or cap <= 0):
                raise ValueError("Sample limits must be positive integers or null")
        if t.mode not in {"vfl", "single"}:
            raise ValueError("training.mode must be vfl or single")
        if t.mode == "single" and self.crypto.backend != "plaintext":
            raise ValueError("Single-energy baselines have no server FE layer; use plaintext")
        for mapping in (t.task_weights, t.dcl_alpha):
            if set(mapping) != set(CLIENTS):
                raise ValueError(f"Coefficients must name exactly {CLIENTS}")
            if any(not math.isfinite(v) or v < 0 for v in mapping.values()):
                raise ValueError("Coefficients must be finite and nonnegative")
        if not any(t.task_weights.values()):
            raise ValueError("At least one forecasting task weight must be positive")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (t.learning_rate, t.dcor_epsilon, t.mape_epsilon)
        ):
            raise ValueError("Learning rate and epsilons must be positive and finite")
        if not math.isfinite(t.min_delta) or t.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")


def merge(base: dict, override: dict) -> dict:
    """Recursively merge dictionaries; replace lists and scalar values."""
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def config_from_dict(value: dict) -> Config:
    extra = set(value) - {"data", "model", "training", "crypto"}
    if extra:
        raise ValueError(f"Unknown config sections: {sorted(extra)}")
    value = merge(Config().to_dict(), value)
    config = Config(
        DataConfig(**value["data"]),
        ModelConfig(**value["model"]),
        TrainingConfig(**value["training"]),
        CryptoConfig(**value["crypto"]),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> Config:
    def read(current: Path, visited: set[Path]) -> dict:
        current = current.resolve()
        if current in visited:
            raise ValueError(f"Configuration inheritance cycle: {current}")
        value = json.loads(current.read_text(encoding="utf-8-sig"))
        parent = value.pop("extends", None)
        return merge(read(current.parent / parent, visited | {current}), value) if parent else value

    return config_from_dict(read(Path(path), set()))
