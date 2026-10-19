"""Explicit split-learning messages and validation-based checkpoint selection."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import CLIENTS, Config
from .crypto import ForwardTransport
from .data import DataBundle, prepare_data
from .metrics import distance_correlation, forecast_metrics
from .models import ForecastSystem


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def set_seed(config: Config) -> torch.device:
    t = config.training
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t.seed)
    torch.set_num_threads(t.num_threads)
    torch.use_deterministic_algorithms(t.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = t.deterministic
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (t.device == "auto")
        else torch.device(t.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select cpu or auto")
    return device


def move_batch(x: dict, y: dict, device: torch.device) -> tuple[dict, dict]:
    return ({k: v.to(device) for k, v in x.items()}, {k: v.to(device) for k, v in y.items()})


def protected_shared(
    system: ForecastSystem, z: dict, transport: ForwardTransport, batch: str, version: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only linear outputs enter the server graph; FE has no autograd edge."""
    server = system.server
    contributions = []
    for client in CLIENTS:
        value = transport.linear(
            client,
            z[client].detach().cpu().numpy(),
            server.block(client).detach().cpu().numpy(),
            batch,
            version,
        )
        contributions.append(torch.as_tensor(value, device=z[client].device, dtype=z[client].dtype))
    activation = torch.stack(contributions).sum(0) + server.first.bias
    return server.tail(activation), activation


def backward_batch(
    system: ForecastSystem,
    x: dict,
    y: dict,
    config: Config,
    transport: ForwardTransport | None,
    batch: str,
    version: int,
) -> dict:
    """Compute gradients without updating parameters; matches Algorithm 1.

    Implementation uses row-major batches: G_k = Delta.T @ Z_k and H_k = Delta @ W_k.
    With fixed-point FE, float backward gradients are a straight-through surrogate
    for quantized forward evaluation, not a derivative through integer decryption.
    """
    t = config.training
    z = {k: system.clients[k].encoder(x[k]) for k in CLIENTS}
    penalties = {k: distance_correlation(x[k], z[k], t.dcor_epsilon) for k in CLIENTS}
    losses = {}
    if system.server is None:
        for k in CLIENTS:
            pred = system.clients[k].decoder(z[k])
            losses[k] = nn.functional.mse_loss(pred, y[k])
            (t.task_weights[k] * losses[k] + t.dcl_alpha[k] * penalties[k]).backward()
    else:
        shared, activation = protected_shared(system, z, transport, batch, version)
        activation.retain_grad()
        q = torch.zeros_like(shared)
        for k in CLIENTS:
            local_shared = shared.detach().requires_grad_(True)
            pred = system.clients[k].decoder(local_shared)
            losses[k] = nn.functional.mse_loss(pred, y[k])
            (t.task_weights[k] * losses[k]).backward()
            q += local_shared.grad  # Only Q_k returns from the decoder's local graph.
        shared.backward(q)
        delta = activation.grad.detach()
        weight_gradients = []
        for k in CLIENTS:
            weight_gradients.append(delta.T @ z[k].detach())  # Computed at client k.
            h = delta @ system.server.block(k).detach()  # Computed by server, before updates.
            torch.autograd.backward(
                (z[k], penalties[k]), (h, torch.full_like(penalties[k], t.dcl_alpha[k]))
            )
        system.server.first.weight.grad = torch.cat(weight_gradients, dim=1)
    prediction = sum(t.task_weights[k] * losses[k].detach().item() for k in CLIENTS)
    penalty = sum(t.dcl_alpha[k] * penalties[k].detach().item() for k in CLIENTS)
    return {"prediction_loss": prediction, "dcl_loss": penalty, "objective": prediction + penalty}


@torch.no_grad()
def evaluate(
    system: ForecastSystem,
    bundle: DataBundle,
    partition: str,
    config: Config,
    transport: ForwardTransport | None,
    device: torch.device,
    version: int,
    collect: bool = False,
) -> tuple[float, dict, dict]:
    system.eval()
    total, count = 0.0, 0
    predictions, representations = {k: [] for k in CLIENTS}, {k: [] for k in CLIENTS}
    for index, (x, y, _) in enumerate(bundle.loader(partition, config.training.batch_size)):
        x, y = move_batch(x, y, device)
        z = {k: system.clients[k].encoder(x[k]) for k in CLIENTS}
        shared = (
            protected_shared(system, z, transport, f"{partition}:{index}", version)[0]
            if (system.server is not None)
            else None
        )
        for k in CLIENTS:
            pred = system.clients[k].decoder(shared if shared is not None else z[k])
            total += (
                config.training.task_weights[k]
                * nn.functional.mse_loss(pred, y[k]).item()
                * len(y[k])
            )
            if collect:
                predictions[k].append(pred.cpu())
                representations[k].append(z[k].cpu())
        count += len(x[CLIENTS[0]])
    return total / count, predictions, representations


def save_test_results(
    system: ForecastSystem,
    bundle: DataBundle,
    config: Config,
    transport: ForwardTransport | None,
    device: torch.device,
    version: int,
    output: Path,
) -> dict:
    loss, predictions, representations = evaluate(
        system, bundle, "test", config, transport, device, version, True
    )
    metrics = {
        "normalized_weighted_mse": loss,
        "clients": {},
        "model_version": version,
        "mode": config.training.mode,
        "crypto_backend": config.crypto.backend,
        "weather": config.data.weather,
        "mape_unit": "fraction",
        "physical_units": "as supplied in source CSV; no rescaling applied",
    }
    dataset = bundle.datasets["test"]
    for k in CLIENTS:
        scale = bundle.preprocessing[k]
        predicted = torch.cat(predictions[k]).numpy() * scale["y_span"] + scale["y_min"]
        actual = dataset.y[k].numpy() * scale["y_span"] + scale["y_min"]
        m = forecast_metrics(actual, predicted, config.training.mape_epsilon)
        n = min(len(dataset), config.training.eval_dcor_max_samples)
        m["dcor"] = float(
            distance_correlation(
                dataset.x[k][:n], torch.cat(representations[k])[:n], config.training.dcor_epsilon
            )
        )
        m["dcor_samples"] = n
        m["dcor_selection"] = "first chronological test samples, full-set statistic on this subset"
        metrics["clients"][k] = m
        pd.DataFrame(
            {
                "sample_id": np.repeat(dataset.ids, config.data.horizon),
                "horizon_hour": np.tile(np.arange(config.data.horizon), len(dataset)),
                "actual": actual.ravel(),
                "predicted": predicted.ravel(),
            }
        ).to_csv(output / f"predictions_{k}.csv", index=False)
    write_json(output / "metrics.json", metrics)
    return metrics


def run_experiment(config: Config, output: str | Path) -> dict:
    """Train from fresh weights, restore this run's best validation checkpoint, test once."""
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = set_seed(config)
    bundle = prepare_data(config)
    input_dim = bundle.datasets["train"].x[CLIENTS[0]].shape[1]
    system = ForecastSystem(config, input_dim).to(device)
    transport = (
        ForwardTransport(config.crypto, config.model.representation_dim)
        if (system.server is not None)
        else None
    )
    optimizers = [
        torch.optim.Adam(system.clients[k].parameters(), lr=config.training.learning_rate)
        for k in CLIENTS
    ]
    if system.server is not None:
        optimizers.append(
            torch.optim.Adam(system.server.parameters(), lr=config.training.learning_rate)
        )
    write_json(output / "resolved_config.json", config.to_dict())
    write_json(output / "preprocessing.json", bundle.preprocessing)
    bundle.manifest.to_csv(output / "split_manifest.csv", index=False)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "packages": {
                name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas")
            },
            "cuda": torch.version.cuda,
            "deterministic": config.training.deterministic,
            "sample_counts": {p: len(d) for p, d in bundle.datasets.items()},
            "crypto_randomness": "fresh OS entropy, independent of training seed",
        },
    )
    loader = bundle.loader("train", config.training.batch_size, config.training.seed)
    best, stale, version, history = float("inf"), 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, config.training.epochs + 1):
        system.train()
        totals, count = dict.fromkeys(("prediction_loss", "dcl_loss", "objective"), 0.0), 0
        for index, (x, y, _) in enumerate(loader):
            x, y = move_batch(x, y, device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            values = backward_batch(
                system, x, y, config, transport, f"train:{epoch}:{index}", version
            )
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError("Nonfinite training loss")
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in system.parameters()
            ):
                raise FloatingPointError("Nonfinite gradient")
            for optimizer in optimizers:
                optimizer.step()
            version += 1
            n = len(x[CLIENTS[0]])
            count += n
            for name, value in values.items():
                totals[name] += value * n
        val, _, _ = evaluate(system, bundle, "val", config, transport, device, version)
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        history.append(
            {
                "epoch": epoch,
                **{k: v / count for k, v in totals.items()},
                "val_prediction_loss": val,
            }
        )
        if val < best - config.training.min_delta:
            best, stale = val, 0
            torch.save(
                {
                    "state_dict": system.state_dict(),
                    "config": config.to_dict(),
                    "preprocessing": bundle.preprocessing,
                    "input_dim": input_dim,
                    "best_epoch": epoch,
                    "model_version": version,
                    "best_val_loss": best,
                    "sample_ids": {p: d.ids for p, d in bundle.datasets.items()},
                },
                output / "best.pt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch} train={totals['objective'] / count:.6f} val_mse={val:.6f}", flush=True
        )
        if stale >= config.training.patience:
            break
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    system.load_state_dict(checkpoint["state_dict"])
    metrics = save_test_results(
        system, bundle, config, transport, device, checkpoint["model_version"], output
    )
    metrics.update(
        {
            "best_epoch": checkpoint["best_epoch"],
            "epochs_run": len(history),
            "training_and_validation_seconds": training_seconds,
            "best_val_loss": checkpoint["best_val_loss"],
        }
    )
    write_json(output / "metrics.json", metrics)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    write_json(output / "crypto_timing.json", dict(transport.timings) if transport else {})
    return metrics


def evaluate_checkpoint(
    checkpoint_path: str | Path, output: str | Path, device_override: str | None = None
) -> dict:
    from .config import config_from_dict

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = config_from_dict(checkpoint["config"])
    if device_override:
        config.training.device = device_override
    device = set_seed(config)
    bundle = prepare_data(config, checkpoint["preprocessing"])
    if {p: d.ids for p, d in bundle.datasets.items()} != checkpoint["sample_ids"]:
        raise ValueError("Forecast-origin IDs differ from the checkpoint's experiment")
    system = ForecastSystem(config, checkpoint["input_dim"]).to(device)
    system.load_state_dict(checkpoint["state_dict"])
    transport = (
        ForwardTransport(config.crypto, config.model.representation_dim)
        if (system.server is not None)
        else None
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metrics = save_test_results(
        system, bundle, config, transport, device, checkpoint["model_version"], output
    )
    write_json(output / "resolved_config.json", config.to_dict())
    write_json(output / "crypto_timing.json", dict(transport.timings) if transport else {})
    return metrics
