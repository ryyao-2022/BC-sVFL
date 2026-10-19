"""Synthetic module timings, explicitly separate from forecasting measurements."""

import time
from pathlib import Path

import numpy as np

from .config import Config
from .crypto import make_backend
from .crypto.base import Context
from .training import write_json


def benchmark_crypto(config: Config, output: Path, samples: int, repetitions: int) -> dict:
    if samples < 1 or repetitions < 1:
        raise ValueError("Benchmark samples and repetitions must be positive")
    backend = make_backend(config.crypto)
    rng = np.random.default_rng(config.training.seed)
    dimension, rows = config.model.representation_dim, config.model.server_hidden_dim
    z_bound = min(1, config.crypto.options.get("z_bound", 2) / 2)
    w_bound = min(1, config.crypto.options.get("w_bound", 2) / 2)
    z = rng.uniform(-z_bound, z_bound, (samples, dimension))
    w = rng.uniform(-w_bound, w_bound, (rows, dimension))
    start = time.perf_counter()
    public, authority = backend.setup(dimension, "benchmark")
    setup_seconds = time.perf_counter() - start
    expected = z @ w.T
    if config.crypto.backend == "reference_fe":
        zs, ws = public.z_scale, public.w_scale
        expected = (np.rint(z * zs) @ np.rint(w * ws).T) / (zs * ws)
    records, max_error = [], 0.0
    # First repetition includes any discrete-log table creation; no hidden warmup.
    for i in range(repetitions):
        context = Context("benchmark", f"benchmark:{i}", i)
        record = {}
        start = time.perf_counter()
        ciphertext = backend.encrypt(public, z, context)
        record["encrypt_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        keys = backend.keygen(authority, w, context)
        record["keygen_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        result = backend.evaluate(public, ciphertext, keys)
        record["evaluate_seconds"] = time.perf_counter() - start
        max_error = max(max_error, float(np.max(np.abs(result - expected))))
        records.append(record)
    report = {
        "config": config.to_dict(),
        "samples": samples,
        "dimension": dimension,
        "weight_rows": rows,
        "repetitions": repetitions,
        "warmup": 0,
        "setup_seconds": setup_seconds,
        "records": records,
        "max_abs_error_against_encoded_linear_if_reference_fe_else_float": max_error,
        "scope": "one synthetic client; excludes neural training, transport and ledger",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "benchmark.json", report)
    return report
