# BC-sVFL

## Overview

**Blockchain-Assisted Vertical Federated Learning for Multi-Energy Load Forecasting**

BC-sVFL studies collaborative forecasting across electricity, heating and cooling loads. Clients keep their data locally and train through shared representations, with distance-correlation regularization and a replaceable encryption backend.

本项目面向冷、热、电多能源负荷预测，通过纵向联邦学习利用不同能源之间的关联，结合距离相关正则化与可替换的加密模块，在各方保留本地数据的前提下开展协同训练。

![BC-sVFL framework](docs/assets/bc_svfl_framework.png)

The code runs the training pipeline in a single process. Blockchain integration remains an extension of the framework.

## Getting started

Use Python 3.11+ in a virtual environment. From the repository root:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m vfl_ies train --config configs/smoke.json
```

For a full training run:

```bash
python -m vfl_ies train --config configs/default.json --output runs/vfl
```

Experiment settings live in `configs/`. Use `baseline.json` for independent single-energy models, `historical_weather.json` for past weather inputs, or `fe_2048.json` for a small encrypted run. Set `--alpha 0` to disable distance-correlation regularization.

Each run saves its configuration, data splits, preprocessing parameters, best checkpoint, metrics and predictions. Choose a new output directory for each run; omitting `--output` creates one automatically.

## Implementation

- **Data:** `data.py` aligns the three clients by forecast time and fits scaling parameters on training data only. The default uses 168 hours of load history to predict 24 hours ahead, with 2016–2018 for training and validation and 2019 for testing. Default weather inputs are observations from the forecast period; the historical-weather configuration uses past observations instead.
- **Training:** `models.py` defines local encoders, a shared server model and local prediction heads. `training.py` handles gradient exchange, prediction loss, distance-correlation regularization and validation-based early stopping.
- **Encryption:** `crypto/` provides a plaintext control and reference inner-product functional encryption for the server's first linear layer. A custom backend implements `setup`, `encrypt`, `keygen` and `evaluate`, and is selected through `crypto.backend = "your_module:YourBackend"`. See the [interface notes](docs/CRYPTO_INTERFACE.md) and [example adapter](examples/backend_plugin.py).

## Development

Install `requirements-dev.txt`, then run:

```bash
python -m pytest -q
python -m ruff check vfl_ies tests examples
python -m ruff format --check vfl_ies tests examples
```
