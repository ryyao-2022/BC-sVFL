"""Command-line entry points for preparation, runs, repeats, and checkpoint evaluation."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLIENTS, load_config
from .data import build_windows
from .training import evaluate_checkpoint, run_experiment, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible VFL-IES research workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "repeat", "prepare", "benchmark-crypto"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/default.json")
        command.add_argument("--output", type=Path)
        if name in {"train", "repeat"}:
            command.add_argument("--epochs", type=int)
            command.add_argument("--alpha", type=float, help="Set DCL coefficient for every client")
            command.add_argument("--device")
        if name == "train":
            command.add_argument("--seed", type=int)
        elif name == "repeat":
            command.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
        elif name == "benchmark-crypto":
            command.add_argument("--samples", type=int, default=2)
            command.add_argument("--repetitions", type=int, default=3)
    command = commands.add_parser("evaluate")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command.add_argument("--device")
    args = parser.parse_args()
    output = args.output or Path("runs") / (
        args.command + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if args.command == "evaluate":
        evaluate_checkpoint(args.checkpoint, output, args.device)
    else:
        config = load_config(args.config)
        for name in ("epochs", "seed", "device"):
            if getattr(args, name, None) is not None:
                setattr(config.training, name, getattr(args, name))
        if getattr(args, "alpha", None) is not None:
            config.training.dcl_alpha = dict.fromkeys(CLIENTS, args.alpha)
        config.validate()
        if args.command == "prepare":
            train, test = build_windows(config.data)
            output.mkdir(parents=True, exist_ok=False)
            train.to_csv(output / "data_train_asu.csv", index=False)
            test.to_csv(output / "data_test_asu.csv", index=False)
            write_json(
                output / "preparation.json",
                {
                    "config": config.to_dict(),
                    "train_candidates": len(train),
                    "test_samples": len(test),
                },
            )
        elif args.command == "train":
            run_experiment(config, output)
        elif args.command == "benchmark-crypto":
            from .benchmark import benchmark_crypto

            benchmark_crypto(config, output, args.samples, args.repetitions)
        else:
            if len(set(args.seeds)) != len(args.seeds):
                raise ValueError("Repeated seeds must be unique")
            output.mkdir(parents=True, exist_ok=False)
            rows = []
            for seed in args.seeds:
                current = copy.deepcopy(config)
                current.training.seed = seed
                metrics = run_experiment(current, output / f"seed_{seed}")
                rows += [
                    {
                        "seed": seed,
                        "client": k,
                        **{m: metrics["clients"][k][m] for m in ("rmse", "mae", "mape", "dcor")},
                    }
                    for k in CLIENTS
                ]
            pd.DataFrame(rows).to_csv(output / "all_seeds.csv", index=False)
            summary = {
                k: {
                    m: {
                        "mean": float(np.mean(values := [r[m] for r in rows if r["client"] == k])),
                        "std_ddof_1": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    }
                    for m in ("rmse", "mae", "mape", "dcor")
                }
                for k in CLIENTS
            }
            write_json(output / "summary.json", summary)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
