"""Numerical checks against ordinary autograd and complete reproducible runs."""

import copy
import json

import numpy as np
import pytest
import torch

from vfl_ies.config import CLIENTS, PROJECT_ROOT, load_config
from vfl_ies.crypto import ForwardTransport
from vfl_ies.metrics import distance_correlation, forecast_metrics
from vfl_ies.models import ForecastSystem
from vfl_ies.training import backward_batch, evaluate_checkpoint, run_experiment, set_seed


@pytest.mark.parametrize("alpha", [0.0, 0.05])
def test_split_messages_match_full_autograd_for_every_parameter(alpha):
    config = load_config(PROJECT_ROOT / "configs/smoke.json")
    config.training.task_weights = dict(zip(CLIENTS, (0.7, 1.1, 2.0), strict=True))
    config.training.dcl_alpha = dict.fromkeys(CLIENTS, alpha)
    set_seed(config)
    split = ForecastSystem(config, 5).double()
    direct = copy.deepcopy(split)
    x = {k: torch.randn(7, 5, dtype=torch.double) for k in CLIENTS}
    y = {k: torch.randn(7, 24, dtype=torch.double) for k in CLIENTS}
    transport = ForwardTransport(config.crypto, config.model.representation_dim)
    backward_batch(split, x, y, config, transport, "train:0", 0)
    pred, z = direct.plaintext_forward(x)
    loss = sum(
        config.training.task_weights[k] * (pred[k] - y[k]).square().mean()
        + alpha * distance_correlation(x[k], z[k])
        for k in CLIENTS
    )
    loss.backward()
    for (name, p), (other_name, q) in zip(
        split.named_parameters(), direct.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8, msg=name)


def test_unsquared_distance_correlation_matches_independent_numpy_formula():
    rng = np.random.default_rng(44)
    x, z = rng.normal(size=(9, 4)), rng.normal(size=(9, 3))

    def centered(value):
        distances = np.sqrt(((value[:, None] - value[None, :]) ** 2).sum(-1))
        return distances - distances.mean(0) - distances.mean(1)[:, None] + distances.mean()

    a, b = centered(x), centered(z)
    expected = np.sqrt((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))
    actual = distance_correlation(torch.tensor(x), torch.tensor(z))
    assert float(actual) == pytest.approx(expected)
    assert torch.autograd.gradcheck(
        lambda value: distance_correlation(torch.tensor(x), value),
        (torch.tensor(z, requires_grad=True),),
    )


@pytest.mark.parametrize("samples", [1, 5])
def test_constant_representation_has_finite_zero_penalty_gradient(samples):
    z = torch.ones(samples, 3, requires_grad=True)
    penalty = distance_correlation(torch.randn(samples, 4), z)
    penalty.backward()
    assert penalty.item() == 0
    assert torch.isfinite(z.grad).all()
    assert not z.grad.any()


def test_mape_zero_policy_and_units():
    result = forecast_metrics(np.array([0.0, 2.0]), np.array([1.0, 4.0]), epsilon=1)
    assert result["mape"] == 1
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["mape_small_denominator_count"] == 1


def test_same_seed_and_checkpoint_reevaluation_reproduce_predictions(tmp_path):
    config = load_config(PROJECT_ROOT / "configs/smoke.json")
    config.training.min_delta = 100.0  # Exercise restoring epoch 1 after later parameter updates.
    first = run_experiment(config, tmp_path / "first")
    second = run_experiment(config, tmp_path / "second")
    restored = evaluate_checkpoint(tmp_path / "first/best.pt", tmp_path / "restored")
    assert first["best_epoch"] == 1
    assert first["clients"] == second["clients"] == restored["clients"]
    for k in CLIENTS:
        name = f"predictions_{k}.csv"
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "restored" / name
        ).read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path / "first")


@pytest.mark.parametrize("mode,backend", [("single", "plaintext"), ("vfl", "reference_fe")])
def test_baseline_and_encrypted_end_to_end(tmp_path, mode, backend):
    config = load_config(PROJECT_ROOT / "configs/fe_smoke.json")
    config.training.mode = mode
    if backend == "plaintext":
        config.crypto.backend, config.crypto.options = backend, {}
    metrics = run_experiment(config, tmp_path / mode)
    assert all(np.isfinite(m["rmse"]) for m in metrics["clients"].values())
    timing = json.loads((tmp_path / mode / "crypto_timing.json").read_text())
    if mode == "vfl":
        assert timing["train.encrypt"]["calls"] == 12
        assert timing["test.evaluate"]["calls"] == 3


def test_unknown_config_does_not_silently_change_the_experiment(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{"training": {"epoch": 4}}')
    with pytest.raises(TypeError):
        load_config(path)
    path.write_text('{"extends": "wrong.json"}')
    with pytest.raises(ValueError, match="cycle"):
        load_config(path)
