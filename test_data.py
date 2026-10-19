"""Regression checks for alignment, train-only scaling, and temporal inputs."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vfl_ies.config import CLIENTS, Config
from vfl_ies.data import AlignedDataset, build_windows, feature_columns, prepare_data


def test_shared_shuffle_preserves_client_sample_alignment():
    x = {k: np.arange(30).reshape(10, 3) + index * 100 for index, k in enumerate(CLIENTS)}
    y = {k: value[:, :1] for k, value in x.items()}
    loader = DataLoader(
        AlignedDataset(x, y, list(map(str, range(10)))),
        batch_size=3,
        shuffle=True,
        generator=torch.Generator().manual_seed(8),
    )
    order = []
    for xb, yb, ids in loader:
        indices = np.array([int(i) for i in ids])
        order.extend(indices)
        for index, k in enumerate(CLIENTS):
            np.testing.assert_array_equal(xb[k][:, 0], indices * 3 + index * 100)
            np.testing.assert_array_equal(yb[k][:, 0], indices * 3 + index * 100)
    assert order != list(range(10))
    assert sorted(order) == list(range(10))


def test_scalers_ignore_extreme_validation_and_test_values(tmp_path):
    config = Config()
    config.data.root = str(tmp_path)
    config.data.history = 3
    config.data.horizon = 2
    config.data.validation_fraction = 0.2
    columns = set()
    for client in CLIENTS:
        x, y = feature_columns(config.data, client)
        columns.update(x + y)
    for filename, year, n in ((config.data.train_csv, 2016, 20), (config.data.test_csv, 2019, 5)):
        frame = pd.DataFrame({c: np.arange(n, dtype=float) for c in columns})
        frame["Year"], frame["Month"], frame["Day"], frame["Hour"] = year, 1, np.arange(1, n + 1), 0
        _, targets = feature_columns(config.data, "electricity")
        frame.loc[16 if year == 2016 else 0 :, targets] = 100000
        frame.to_csv(tmp_path / filename, index=False)
    bundle = prepare_data(config)
    scale = bundle.preprocessing["electricity"]
    assert scale["y_min"] == 0
    assert scale["y_span"] == 15
    assert bundle.datasets["val"].y["electricity"].min() > 1
    assert np.isfinite(bundle.datasets["test"].x["electricity"].numpy()).all()


def test_raw_history_weather_never_uses_forecast_period():
    config = Config()
    config.data.source = "raw"
    config.data.weather = "history"
    train, test = build_windows(config.data)
    assert (len(train), len(test)) == (1089, 365)
    from vfl_ies.config import PROJECT_ROOT

    raw = pd.read_csv(PROJECT_ROOT / config.data.root / config.data.raw_csv)
    assert train.iloc[0]["RelativeHumidity_lag0"] == raw.iloc[167]["Relative Humidity"]
    assert train.iloc[0]["Electricity_load_t-0"] == raw.iloc[167]["Electricity"]
    assert train.iloc[0]["Electricity_load_t0"] == raw.iloc[168]["Electricity"]
    assert train.iloc[0]["AirTemperature_lag23"] == raw.iloc[144]["Temperature"]


def test_default_manifest_partitions_are_disjoint_and_chronological():
    bundle = prepare_data(Config())
    ids = {name: set(data.ids) for name, data in bundle.datasets.items()}
    assert not ids["train"] & ids["val"]
    assert not (ids["train"] | ids["val"]) & ids["test"]
    assert max(ids["train"]) < min(ids["val"]) < min(ids["test"])
    assert [len(bundle.datasets[k]) for k in ("train", "val", "test")] == [980, 109, 365]
