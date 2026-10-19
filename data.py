"""Aligned forecasting windows and scalers fitted exclusively on training rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CLIENTS, PROJECT_ROOT, Config, DataConfig

LOAD_COLUMNS = {
    "electricity": "Electricity_load",
    "steam": "Steam_load",
    "cooling": "Chillwater_load",
}
RAW_COLUMNS = {
    "Electricity": "Electricity_load",
    "Heating": "Steam_load",
    "Cooling": "Chillwater_load",
    "Temperature": "AirTemperature",
    "Relative Humidity": "RelativeHumidity",
}


def data_root(config: DataConfig) -> Path:
    return PROJECT_ROOT / config.root


def origins(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Create forecast-origin IDs from hourly calendar columns."""
    date = pd.to_datetime(frame[["Year", "Month", "Day"]].rename(columns=str.lower))
    return pd.DatetimeIndex(date + pd.to_timedelta(frame["Hour"], unit="h"))


def feature_columns(config: DataConfig, client: str) -> tuple[list[str], list[str]]:
    load = LOAD_COLUMNS[client]
    weather_suffix = "t" if config.weather == "target_observed" else "lag"
    history = [f"{load}_t-{i}" for i in range(config.history)]
    weather = [
        f"{name}_{weather_suffix}{i}"
        for name in ("AirTemperature", "RelativeHumidity")
        for i in range(config.horizon)
    ]
    return history + weather + ["Month", "Year", "Week"], [
        f"{load}_t{i}" for i in range(config.horizon)
    ]


def build_windows(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build windows from hourly raw data; lag 0 is origin minus one hour.

    target_observed uses weather observations from the forecast period.
    history uses only the most recent horizon hours of observed weather.
    Values retain the source CSV units; this function performs no unit conversion.
    """
    raw = pd.read_csv(data_root(config) / config.raw_csv).rename(columns=RAW_COLUMNS)
    time = origins(raw)
    if time.has_duplicates or not time.is_monotonic_increasing:
        raise ValueError("Raw data must have unique, increasing hourly timestamps")
    if len(time) < 2 or not np.all(np.diff(time.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError("Raw data has missing hours; impute explicitly before windowing")
    required = [*LOAD_COLUMNS.values(), "AirTemperature", "RelativeHumidity"]
    if not np.isfinite(raw[required].to_numpy(dtype=float)).all():
        raise ValueError("Raw inputs contain missing/nonfinite values")
    columns = {name: raw[name] for name in ("Year", "Month", "Day", "Hour")}
    columns["Week"] = pd.Series(time.weekday, index=raw.index)
    for load in LOAD_COLUMNS.values():
        for i in range(config.history):
            columns[f"{load}_t-{i}"] = raw[load].shift(i + 1)
        for i in range(config.horizon):
            columns[f"{load}_t{i}"] = raw[load].shift(-i)
    suffix = "t" if config.weather == "target_observed" else "lag"
    for weather in ("AirTemperature", "RelativeHumidity"):
        for i in range(config.horizon):
            shift = -i if config.weather == "target_observed" else i + 1
            columns[f"{weather}_{suffix}{i}"] = raw[weather].shift(shift)
    windows = pd.DataFrame(columns).dropna().iloc[:: config.stride].copy()
    start = origins(windows)
    end = start + pd.Timedelta(hours=config.horizon - 1)
    train = windows.loc[start.year.isin(config.train_years) & end.year.isin(config.train_years)]
    test = windows.loc[start.year.isin(config.test_years) & end.year.isin(config.test_years)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def read_frames(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.source == "raw":
        frames = build_windows(config)
    else:
        frames = tuple(
            pd.read_csv(data_root(config) / name) for name in (config.train_csv, config.test_csv)
        )
    result = []
    for frame, years in zip(frames, (config.train_years, config.test_years), strict=True):
        # Map the prepared CSV humidity field to a consistent feature name.
        frame = frame.rename(
            columns={
                col: col.replace("DewTemperature", "RelativeHumidity")
                for col in frame
                if col.startswith("DewTemperature")
            }
        )
        frame = frame.copy()
        frame["sample_id"] = origins(frame).astype(str)
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        if frame.empty or frame["sample_id"].duplicated().any():
            raise ValueError("Each partition needs nonempty, unique forecast-origin IDs")
        time = origins(frame)
        end = time + pd.Timedelta(hours=config.horizon - 1)
        if not time.year.isin(years).all() or not end.year.isin(years).all():
            raise ValueError("CSV forecast windows do not match configured partition years")
        for client in CLIENTS:
            x_cols, y_cols = feature_columns(config, client)
            missing = set(x_cols + y_cols) - set(frame)
            if missing:
                raise ValueError(f"Missing columns for {client}: {sorted(missing)[:5]}")
            if not np.isfinite(frame[x_cols + y_cols].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite features or labels for {client}")
        result.append(frame)
    if origins(result[0]).max() + pd.Timedelta(hours=config.horizon) > origins(result[1]).min():
        raise ValueError("Training targets overlap the test period")
    return result[0], result[1]


class AlignedDataset(Dataset):
    """One index selects all clients together, including when batches are shuffled."""

    def __init__(self, x: dict[str, np.ndarray], y: dict[str, np.ndarray], ids: list[str]):
        self.x = {k: torch.from_numpy(v.astype(np.float32)) for k, v in x.items()}
        self.y = {k: torch.from_numpy(v.astype(np.float32)) for k, v in y.items()}
        self.ids = ids
        if any(len(v) != len(ids) for v in (*self.x.values(), *self.y.values())):
            raise ValueError("All clients must share exactly the same sample IDs")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[dict, dict, str]:
        return (
            {k: v[index] for k, v in self.x.items()},
            {k: v[index] for k, v in self.y.items()},
            self.ids[index],
        )


@dataclass
class DataBundle:
    datasets: dict[str, AlignedDataset]
    preprocessing: dict
    manifest: pd.DataFrame

    def loader(self, partition: str, batch_size: int, seed: int = 42) -> DataLoader:
        return DataLoader(
            self.datasets[partition],
            batch_size=batch_size,
            shuffle=partition == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )


def prepare_data(config: Config, preprocessing: dict | None = None) -> DataBundle:
    """Split first, then fit or reuse all preprocessing parameters."""
    d = config.data
    candidates, test = read_frames(d)
    n_val = math.ceil(len(candidates) * d.validation_fraction)
    if d.validation == "random":
        order = np.random.default_rng(d.split_seed).permutation(len(candidates))
        val = candidates.iloc[np.sort(order[:n_val])]
        train = candidates.iloc[np.sort(order[n_val:])]
    else:
        val = candidates.iloc[-n_val:]
        train = candidates.iloc[:-n_val]
        cutoff = origins(val).min() - pd.Timedelta(hours=d.purge_hours)
        train = train.loc[origins(train) + pd.Timedelta(hours=d.horizon) <= cutoff]
    partitions = {"train": train, "val": val, "test": test}
    caps = (d.max_train_samples, d.max_val_samples, d.max_test_samples)
    partitions = {
        k: frame.iloc[:cap].copy() for (k, frame), cap in zip(partitions.items(), caps, strict=True)
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("Empty train/validation/test partition after splitting and limits")
    manifest = pd.concat(
        [frame[["sample_id"]].assign(partition=name) for name, frame in partitions.items()],
        ignore_index=True,
    )
    fit = partitions["train"]
    state = {} if preprocessing is None else preprocessing
    all_x, all_y = {p: {} for p in partitions}, {p: {} for p in partitions}
    for client in CLIENTS:
        x_cols, y_cols = feature_columns(d, client)
        if preprocessing is None:
            minimum, span = np.zeros(len(x_cols)), np.ones(len(x_cols))
            # Loads/weather share a scale over their respective window, calendar per column.
            cuts = [0, d.history, d.history + d.horizon, d.history + 2 * d.horizon]
            groups = [list(range(a, b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
            groups += [[i] for i in range(cuts[-1], len(x_cols))]
            values = fit[x_cols].to_numpy(dtype=float)
            for group in groups:
                low, high = float(values[:, group].min()), float(values[:, group].max())
                minimum[group], span[group] = low, high - low if high > low else 1.0
            labels = fit[y_cols].to_numpy(dtype=float)
            low, high = float(labels.min()), float(labels.max())
            state[client] = {
                "x_columns": x_cols,
                "y_columns": y_cols,
                "x_min": minimum.tolist(),
                "x_span": span.tolist(),
                "y_min": low,
                "y_span": high - low if high > low else 1.0,
            }
        scale = state[client]
        if scale["x_columns"] != x_cols or scale["y_columns"] != y_cols:
            raise ValueError("Checkpoint feature schema differs from configured data")
        for name, frame in partitions.items():
            all_x[name][client] = (frame[x_cols].to_numpy(dtype=float) - scale["x_min"]) / scale[
                "x_span"
            ]
            all_y[name][client] = (frame[y_cols].to_numpy(dtype=float) - scale["y_min"]) / scale[
                "y_span"
            ]
    datasets = {
        name: AlignedDataset(all_x[name], all_y[name], frame.sample_id.tolist())
        for name, frame in partitions.items()
    }
    return DataBundle(datasets, state, manifest)
