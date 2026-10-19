"""Backend registry and timing; no cryptographic operation is hidden in the model."""

from __future__ import annotations

import importlib
import time
from collections import defaultdict

import numpy as np

from ..config import CLIENTS, CryptoConfig
from .base import Context
from .fe import ReferenceFEBackend
from .plaintext import PlaintextBackend


def make_backend(config: CryptoConfig):
    if config.backend == "plaintext":
        factory = PlaintextBackend
    elif config.backend == "reference_fe":
        factory = ReferenceFEBackend
    else:
        module, separator, name = config.backend.partition(":")
        if not separator:
            raise ValueError("Custom backend must use module:Factory format")
        factory = getattr(importlib.import_module(module), name)
    backend = factory(**config.options)
    for method in ("setup", "keygen", "encrypt", "evaluate"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"Backend lacks required method {method}")
    return backend


class ForwardTransport:
    """Simulate separate role calls; record actual module costs by phase.

    Roles share a Python process. This object is an orchestrator, not a server
    deployment boundary. A production adapter must locate secrets at its authority.
    """

    def __init__(self, config: CryptoConfig, dimension: int):
        self.backend = make_backend(config)
        self.timings = defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        self.contexts = {k: self._call("setup", "setup", dimension, k) for k in CLIENTS}

    def _call(self, phase: str, method: str, *args):
        start = time.perf_counter()
        result = getattr(self.backend, method)(*args)
        entry = self.timings[f"{phase}.{method}"]
        entry["seconds"] += time.perf_counter() - start
        entry["calls"] += 1
        return result

    def linear(
        self, client: str, z: np.ndarray, weights: np.ndarray, batch: str, model_version: int
    ) -> np.ndarray:
        public, authority = self.contexts[client]
        context = Context(client, batch, model_version)
        phase = batch.split(":", 1)[0]
        ciphertext = self._call(phase, "encrypt", public, z, context)
        keys = self._call(phase, "keygen", authority, weights, context)
        output = np.asarray(self._call(phase, "evaluate", public, ciphertext, keys))
        if output.shape != (len(z), len(weights)) or not np.isfinite(output).all():
            raise ValueError("Backend returned invalid protected-linear output")
        return output
