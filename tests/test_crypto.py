"""Protect the encoding and role contracts required by the training interface."""

from dataclasses import replace

import numpy as np
import pytest

from vfl_ies.config import CryptoConfig
from vfl_ies.crypto import make_backend
from vfl_ies.crypto.base import Context
from vfl_ies.crypto.fe import ReferenceFEBackend


@pytest.mark.parametrize("group", ["toy31", "rfc3526_2048"])
def test_signed_fixed_point_inner_products(group):
    backend = ReferenceFEBackend(group=group, z_scale=10, w_scale=20, z_bound=2, w_bound=2)
    public, authority = backend.setup(3, "electricity")
    context = Context("electricity", "train:1:0", 0)
    z = np.array([[1.25, -0.78, 0.0], [-1.0, 0.0, 2.0]])
    w = np.array([[-0.44, 1.0, -0.5], [0.0, 0.0, 0.0], [1.5, -1.0, 2.0]])
    ciphertext = backend.encrypt(public, z, context)
    keys = backend.keygen(authority, w, context)
    expected = np.rint(z * 10) @ np.rint(w * 20).T / 200
    np.testing.assert_allclose(backend.evaluate(public, ciphertext, keys), expected, atol=1e-12)
    assert not hasattr(ciphertext, "value")
    assert not hasattr(keys, "s")
    assert ciphertext != backend.encrypt(public, z, context)
    with pytest.raises(ValueError, match="context mismatch"):
        backend.evaluate(
            public, ciphertext, replace(keys, context=replace(context, model_version=1))
        )
    public2, _ = backend.setup(3, "electricity")
    with pytest.raises(ValueError, match="setup mismatch"):
        backend.evaluate(public2, ciphertext, keys)


def test_range_errors_do_not_silently_clip_or_wrap():
    backend = ReferenceFEBackend(group="toy31", z_scale=10, w_scale=10, z_bound=2, w_bound=2)
    public, authority = backend.setup(2, "steam")
    context = Context("steam", "test:0", 0)
    with pytest.raises(ValueError, match="exceeds"):
        backend.encrypt(public, np.array([[2.1, 0.0]]), context)
    with pytest.raises(ValueError, match="exceeds"):
        backend.keygen(authority, np.array([[0.0, np.nan]]), context)
    with pytest.raises(ValueError, match="wraps"):
        ReferenceFEBackend(group="toy31", z_scale=100000, w_scale=100000).setup(20, "steam")
    with pytest.raises(ValueError, match="max_baby_steps"):
        ReferenceFEBackend(max_baby_steps=1).setup(20, "steam")


def test_plugin_replacement_uses_the_same_contract():
    backend = make_backend(
        CryptoConfig(
            "examples.backend_plugin:Backend", {"group": "toy31", "z_scale": 10, "w_scale": 10}
        )
    )
    public, authority = backend.setup(2, "cooling")
    context = Context("cooling", "test:0", 7)
    ciphertext = backend.encrypt(public, np.array([[1.0, -1.0]]), context)
    keys = backend.keygen(authority, np.array([[0.5, 1.0]]), context)
    np.testing.assert_allclose(backend.evaluate(public, ciphertext, keys), [[-0.5]])
