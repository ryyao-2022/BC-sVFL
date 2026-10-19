"""Forecasting metrics and unsquared biased sample distance correlation."""

import numpy as np
import torch


def distance_correlation(x: torch.Tensor, z: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Compare samples (rows), not individual time steps.

    Zero/near-zero variance returns zero with a finite zero gradient. epsilon
    stabilizes optimization and is recorded in the experiment configuration.
    """
    if x.shape[0] != z.shape[0]:
        raise ValueError("Distance correlation requires paired samples")
    if len(x) < 2:
        return z.sum() * 0.0
    a = torch.cdist(x.flatten(1), x.flatten(1), compute_mode="donot_use_mm_for_euclid_dist")
    b = torch.cdist(z.flatten(1), z.flatten(1), compute_mode="donot_use_mm_for_euclid_dist")
    a = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    b = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
    variance = (a.square().mean() * b.square().mean()).clamp_min(0)
    if variance.detach().item() <= epsilon:
        return z.sum() * 0.0
    ratio = (a * b).mean().clamp_min(0) / variance.sqrt()
    if ratio.detach().item() <= epsilon:
        return z.sum() * 0.0
    return ratio.clamp(max=1).sqrt()


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> dict:
    """RMSE/MAE in the input unit, MAPE as a fraction, averaging samples and hours."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metric arrays must be nonempty and have identical shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric arrays contain nonfinite values")
    error = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), epsilon))),
        "mape_epsilon": epsilon,
        "mape_small_denominator_count": int((np.abs(actual) < epsilon).sum()),
        "forecast_values": int(actual.size),
    }
