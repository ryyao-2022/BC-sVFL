"""Working plugin example: replace delegated methods with your Go/library calls.

Use backend='examples.backend_plugin:Backend' and the reference_fe options.
The default adapter delegates each operation to the Python FE backend.
"""

from vfl_ies.crypto.fe import ReferenceFEBackend


class Backend:
    """Replace this adapter while keeping the training loop and model unchanged."""

    def __init__(self, **options):
        self.implementation = ReferenceFEBackend(**options)

    def setup(self, dimension, client):
        return self.implementation.setup(dimension, client)

    def encrypt(self, public, z, context):
        return self.implementation.encrypt(public, z, context)

    def keygen(self, authority, weights, context):
        return self.implementation.keygen(authority, weights, context)

    def evaluate(self, public, ciphertext, keys):
        return self.implementation.evaluate(public, ciphertext, keys)
