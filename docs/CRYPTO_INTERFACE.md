# Encryption backend

Backends compute each client's contribution to the server's first linear layer. The interface is defined in [`LinearBackend`](../vfl_ies/crypto/base.py); a working adapter is available in [`examples/backend_plugin.py`](../examples/backend_plugin.py).

## Interface

```python
setup(dimension, client) -> (public, authority)
encrypt(public, z, context) -> ciphertext
keygen(authority, weights, context) -> keys
evaluate(public, ciphertext, keys) -> output
```

Inputs are CPU NumPy arrays: `z` has shape `[batch, dimension]` and `weights` has shape `[outputs, dimension]`. The result has shape `[batch, outputs]` and represents `z @ weights.T`.

`public` holds public parameters; `authority` holds the key authority's secret or service handle. `Context` identifies the client, batch and model version. Function keys must match the current weights. Gradient exchange is handled separately by the training loop.

## Replacing the backend

Implement the four methods in an importable class. Its constructor receives `crypto.options` as keyword arguments. For example, save this configuration under `configs/`:

```json
{
  "extends": "fe_smoke.json",
  "crypto": {
    "backend": "examples.backend_plugin:Backend",
    "options": {"group": "toy31", "z_scale": 100, "w_scale": 100}
  }
}
```

The example delegates to the Python FE implementation. Replace those calls with your library or service while preserving the array shapes and linear output semantics.

## Reference implementation

`reference_fe` uses fixed-point encoding and bounded discrete-log recovery. `z_scale` and `w_scale` set precision; `z_bound` and `w_bound` limit input magnitudes. Larger scales or bounds increase recovery cost.

`toy31` is for tests only. `configs/fe_2048.json` selects a 2048-bit group for a small experiment. This is a research implementation; its Python integer operations are not constant-time.

Encrypted forward passes use quantized values, while backward passes use floating-point states with a straight-through approximation. The protection covers the forward linear computation; its outputs and training gradients remain part of the exchanged messages.
