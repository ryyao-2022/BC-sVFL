"""Inner-product functional encryption with signed fixed-point encoding.

This backend implements protected linear evaluation for in-process experiments.
Python big-integer operations are not constant-time. toy31 is for tests only.
Group-14 modulus source: https://www.rfc-editor.org/rfc/rfc3526.html#section-3
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .base import Context

RFC3526_2048 = int(
    """
FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1
29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD
EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245
E485B576 625E7EC6 F44C42E9 A637ED6B 0BFF5CB6 F406B7ED
EE386BFB 5A899FA5 AE9F2411 7C4B1FE6 49286651 ECE45B3D
C2007CB8 A163BF05 98DA4836 1C55D39A 69163FA8 FD24CF5F
83655D23 DCA3AD96 1C62F356 208552BB 9ED52907 7096966D
670C354E 4ABC9804 F1746C08 CA18217C 32905E46 2E36CE3B
E39E772C 180E8603 9B2783A2 EC07A28F B5C55DF0 6F4C52C9
DE2BCBF6 95581718 3995497C EA956AE5 15D22618 98FA0510
15728E5A 8AACAA68 FFFFFFFF FFFFFFFF
""".replace(" ", "").replace("\n", ""),
    16,
)


@dataclass(frozen=True)
class PublicParameters:
    client: str
    setup_id: str
    p: int
    q: int
    g: int
    h: int
    h_vector: tuple[int, ...]
    z_scale: int
    w_scale: int
    z_bound: float
    w_bound: float
    output_bound: int


@dataclass(frozen=True)
class AuthoritySecret:
    public: PublicParameters
    s: tuple[int, ...]
    t: tuple[int, ...]


@dataclass(frozen=True)
class Ciphertext:
    setup_id: str
    context: Context
    # Each row is C, D, E_1, ..., E_dimension, all group elements.
    rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FunctionalKeys:
    setup_id: str
    context: Context
    weights: tuple[tuple[int, ...], ...]
    keys: tuple[tuple[int, int], ...]


def encode(values: np.ndarray, scale: int, bound: float, dimension: int) -> np.ndarray:
    """Round to nearest (ties to even), rejecting range violations without clipping."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != dimension or not len(values):
        raise ValueError(f"Expected nonempty [rows, {dimension}] array")
    if not np.isfinite(values).all() or np.any(np.abs(values) > bound):
        raise ValueError(f"FE input exceeds configured absolute bound {bound}; no clipping applied")
    return np.rint(values * scale).astype(np.int64)


@lru_cache(maxsize=16)
def _baby_steps(p: int, g: int, bound: int) -> tuple[dict[int, int], int, int]:
    size = math.isqrt(2 * bound) + 1
    table, current = {}, 1
    for j in range(size):
        table[current] = j
        current = current * g % p
    return table, pow(g, -size, p), size


def bounded_discrete_log(value: int, p: int, g: int, bound: int) -> int:
    """Recover a signed exponent in [-bound, bound] using baby-step/giant-step."""
    table, factor, size = _baby_steps(p, g, bound)
    shifted = value * pow(g, bound, p) % p
    for i in range(size + 1):
        if shifted in table:
            candidate = i * size + table[shifted]
            if candidate <= 2 * bound:
                return candidate - bound
        shifted = shifted * factor % p
    raise ValueError("Inner product outside the declared recovery interval")


class ReferenceFEBackend:
    """Implements Setup/KeyGen/Enc/Dec for the first server layer.

    group='rfc3526_2048' is the default; group='toy31' is deliberately insecure.
    All current weight rows are approved by this simulation's authority. It does
    not implement a policy preventing reconstruction from authorized outputs.
    """

    def __init__(
        self,
        group: str = "rfc3526_2048",
        z_scale: int = 100,
        w_scale: int = 100,
        z_bound: float = 4.0,
        w_bound: float = 2.0,
        max_baby_steps: int = 100_000,
    ):
        if group not in {"toy31", "rfc3526_2048"}:
            raise ValueError("Unknown FE group")
        for value in (z_scale, w_scale, max_baby_steps):
            if type(value) is not int or value <= 0:
                raise ValueError("FE scales and max_baby_steps must be positive integers")
        if any(not math.isfinite(x) or x <= 0 for x in (z_bound, w_bound)):
            raise ValueError("FE bounds must be positive and finite")
        if max(z_bound * z_scale, w_bound * w_scale) >= 2**52:
            raise ValueError("Fixed-point encoding exceeds exact float64 integer range")
        self.p = 2147483783 if group == "toy31" else RFC3526_2048
        self.q = (self.p - 1) // 2
        # Squaring a nonidentity element places it in the prime-order subgroup.
        self.g = 4
        self.z_scale, self.w_scale = z_scale, w_scale
        self.z_bound, self.w_bound = z_bound, w_bound
        self.max_baby_steps = max_baby_steps

    def setup(self, dimension: int, client: str) -> tuple[PublicParameters, AuthoritySecret]:
        if dimension < 1:
            raise ValueError("FE dimension must be positive")
        bound = (
            dimension
            * math.ceil(self.z_scale * self.z_bound)
            * math.ceil(self.w_scale * self.w_bound)
        )
        if 2 * bound >= self.q:
            raise ValueError("FE recovery interval wraps modulo q; reduce scales/bounds/dimension")
        if math.isqrt(2 * bound) + 1 > self.max_baby_steps:
            raise ValueError("Recovery table exceeds max_baby_steps; adjust the FE parameters")
        h = 1
        while h == 1:
            h = pow(secrets.randbelow(self.p - 1) + 1, 2, self.p)
        s = tuple(secrets.randbelow(self.q) for _ in range(dimension))
        t = tuple(secrets.randbelow(self.q) for _ in range(dimension))
        h_vector = tuple(
            pow(self.g, a, self.p) * pow(h, b, self.p) % self.p for a, b in zip(s, t, strict=True)
        )
        public = PublicParameters(
            client,
            secrets.token_hex(16),
            self.p,
            self.q,
            self.g,
            h,
            h_vector,
            self.z_scale,
            self.w_scale,
            self.z_bound,
            self.w_bound,
            bound,
        )
        return public, AuthoritySecret(public, s, t)

    def encrypt(self, public: PublicParameters, z: np.ndarray, context: Context) -> Ciphertext:
        if context.client != public.client:
            raise ValueError("Client mismatch")
        encoded = encode(z, public.z_scale, public.z_bound, len(public.h_vector))
        rows = []
        for row in encoded:
            r = secrets.randbelow(
                public.q
            )  # Fresh independent randomness for every representation.
            values = tuple(
                pow(public.g, int(v) % public.q, public.p) * pow(h_i, r, public.p) % public.p
                for v, h_i in zip(row, public.h_vector, strict=True)
            )
            rows.append((pow(public.g, r, public.p), pow(public.h, r, public.p), *values))
        return Ciphertext(public.setup_id, context, tuple(rows))

    def keygen(
        self, authority: AuthoritySecret, weights: np.ndarray, context: Context
    ) -> FunctionalKeys:
        public = authority.public
        if context.client != public.client:
            raise ValueError("Client mismatch")
        encoded = encode(weights, public.w_scale, public.w_bound, len(public.h_vector))
        rows = tuple(tuple(int(v) for v in row) for row in encoded)
        keys = tuple(
            (
                sum(a * w for a, w in zip(authority.s, row, strict=True)) % public.q,
                sum(b * w for b, w in zip(authority.t, row, strict=True)) % public.q,
            )
            for row in rows
        )
        return FunctionalKeys(public.setup_id, context, rows, keys)

    def evaluate(
        self, public: PublicParameters, ciphertext: Ciphertext, keys: FunctionalKeys
    ) -> np.ndarray:
        if not (public.setup_id == ciphertext.setup_id == keys.setup_id):
            raise ValueError("FE setup mismatch")
        if ciphertext.context != keys.context or ciphertext.context.client != public.client:
            raise ValueError("FE batch/model/client context mismatch")
        output = np.empty((len(ciphertext.rows), len(keys.keys)), dtype=np.float64)
        for i, (c, d, *elements) in enumerate(ciphertext.rows):
            for j, (weights, (s_w, t_w)) in enumerate(zip(keys.weights, keys.keys, strict=True)):
                numerator = 1
                for element, weight in zip(elements, weights, strict=True):
                    numerator = numerator * pow(element, weight, public.p) % public.p
                denominator = pow(c, s_w, public.p) * pow(d, t_w, public.p) % public.p
                value = numerator * pow(denominator, -1, public.p) % public.p
                integer = bounded_discrete_log(value, public.p, public.g, public.output_bound)
                output[i, j] = integer / (public.z_scale * public.w_scale)
        return output
