"""Unencrypted control backend for numerical equivalence and forecasting runs."""

from dataclasses import dataclass

import numpy as np

from .base import Context


@dataclass(frozen=True)
class PlainPacket:
    context: Context
    value: np.ndarray


class PlaintextBackend:
    """No privacy protection; preserve the same call graph as encrypted training."""

    def __init__(self, **options):
        if options:
            raise ValueError("plaintext backend accepts no options")

    def setup(self, dimension: int, client: str) -> tuple[tuple, tuple]:
        return (dimension, client), (dimension, client)

    def encrypt(self, public: tuple, z: np.ndarray, context: Context) -> PlainPacket:
        if z.ndim != 2 or z.shape[1] != public[0] or context.client != public[1]:
            raise ValueError("Client representation does not match its context")
        return PlainPacket(context, z.copy())

    def keygen(self, authority: tuple, weights: np.ndarray, context: Context) -> PlainPacket:
        if weights.ndim != 2 or weights.shape[1] != authority[0] or context.client != authority[1]:
            raise ValueError("Weight block does not match its context")
        return PlainPacket(context, weights.copy())

    def evaluate(self, public: tuple, ciphertext: PlainPacket, keys: PlainPacket) -> np.ndarray:
        if ciphertext.context != keys.context or ciphertext.context.client != public[1]:
            raise ValueError("Mismatched forward/key context")
        return ciphertext.value @ keys.value.T
