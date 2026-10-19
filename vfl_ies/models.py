"""Local encoders/decoders and explicit server first-layer weight blocks."""

import torch
from torch import nn

from .config import CLIENTS, Config


def mlp(
    input_dim: int, hidden: list[int], output_dim: int, final_relu: bool = False
) -> nn.Sequential:
    layers = []
    dimensions = [input_dim, *hidden, output_dim]
    for i, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        layers.append(nn.Linear(left, right))
        if i < len(dimensions) - 2 or final_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ClientModel(nn.Module):
    def __init__(self, config: Config, input_dim: int):
        super().__init__()
        m = config.model
        if config.training.mode == "vfl":
            self.encoder = mlp(input_dim, m.encoder_hidden, m.representation_dim)
            self.decoder = mlp(m.server_output_dim, m.decoder_hidden, config.data.horizon)
        else:
            self.encoder = mlp(input_dim, m.baseline_hidden[:-1], m.baseline_hidden[-1], True)
            self.decoder = nn.Linear(m.baseline_hidden[-1], config.data.horizon)


class ServerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        m = config.model
        self.representation_dim = m.representation_dim
        self.first = nn.Linear(len(CLIENTS) * m.representation_dim, m.server_hidden_dim)
        self.tail = nn.Sequential(
            nn.ReLU(), mlp(m.server_hidden_dim, m.server_tail_hidden, m.server_output_dim)
        )

    def block(self, client: str) -> torch.Tensor:
        index = CLIENTS.index(client)
        d = self.representation_dim
        return self.first.weight[:, index * d : (index + 1) * d]


class ForecastSystem(nn.Module):
    def __init__(self, config: Config, input_dim: int):
        super().__init__()
        self.clients = nn.ModuleDict({k: ClientModel(config, input_dim) for k in CLIENTS})
        self.server = ServerModel(config) if config.training.mode == "vfl" else None

    def plaintext_forward(self, x: dict[str, torch.Tensor]) -> tuple[dict, dict]:
        """Ordinary autograd composition, used as a reference in gradient tests."""
        z = {k: self.clients[k].encoder(x[k]) for k in CLIENTS}
        shared = (
            self.server.tail(self.server.first(torch.cat(list(z.values()), dim=1)))
            if (self.server is not None)
            else None
        )
        return {
            k: self.clients[k].decoder(shared if shared is not None else z[k]) for k in CLIENTS
        }, z
