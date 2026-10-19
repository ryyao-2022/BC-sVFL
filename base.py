"""Forward-only protected linear interface. Arrays use batch-first orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Context:
    """Identify a client, forward batch, and pre-update model version."""

    client: str
    batch: str
    model_version: int


class LinearBackend(Protocol):
    """Replacement boundary for FE or another protected-linear implementation.

    setup: public material and an authority-only secret/handle.
    encrypt: client input z [batch, dimension] -> opaque ciphertext.
    keygen: authority input weights [outputs, dimension] -> function keys.
    evaluate: server inputs contain no raw z or master secret -> U [batch, outputs].
    Backward messages are handled by training.py, outside this protocol.
    """

    def setup(self, dimension: int, client: str) -> tuple[Any, Any]: ...

    def encrypt(self, public: Any, z: np.ndarray, context: Context) -> Any: ...

    def keygen(self, authority: Any, weights: np.ndarray, context: Context) -> Any: ...

    def evaluate(self, public: Any, ciphertext: Any, keys: Any) -> np.ndarray: ...
