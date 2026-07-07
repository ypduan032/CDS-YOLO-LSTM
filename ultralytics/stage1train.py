from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev

from ultralytics import YOLO


TRAIN_CONFIG = {
    "model": "ultralytics/cfg/models/26/yolo26n-CDS.yaml",
    "data": "datasets/datas.yaml",
    "epochs": 200,
    "batch": 32,
    "workers": 16,
    "device": 0,
    "optimizer": "SGD",
    "project": "runs/train",
    "name": "yolo26_cds",
    "deterministic": True,
    # "repeats": False,
    # "seed": False,
}

SUMMARY_METRICS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
]


def format_mean_std(values: list[float]) -> str:
    avg = mean(values)
    std = stdev(values) if len(values) > 1 else 0.0
    return f"{avg:.4f} +/- {std:.4f}"


def safe_float(value) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def run_default_training(train_cfg: dict):
    current_cfg = dict(train_cfg)
    current_cfg.pop("repeats", None)

    print("\n===== Default YOLO Training Mode =====")
    model = YOLO(current_cfg["model"])
    return model.train(**current_cfg)


def run_repeated_experiments(train_cfg: dict) -> tuple[list[dict], Path]:
    repeats = int(train_cfg.get("repeats", 1))
    base_seed = int(train_cfg.get("seed", 0))
    project = Path(train_cfg.get("project", "runs/repeat_train"))
    experiment_name = str(train_cfg.get("name", "exp"))

    run_summaries: list[dict] = []
    summary_dir = project / experiment_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(repeats):
        run_seed = base_seed + run_idx
        run_name = f"{experiment_name}_seed{run_seed}"
        current_cfg = dict(train_cfg)
        current_cfg.pop("repeats", None)
        current_cfg["seed"] = run_seed
        current_cfg["name"] = run_name
        current_cfg["exist_ok"] = True

        print(f"\n===== Repeat {run_idx + 1}/{repeats} | seed={run_seed} | save_name={run_name} =====")
        model = YOLO(current_cfg["model"])
        metrics = model.train(**current_cfg)

        result_row = {
            "repeat_id": run_idx + 1,
            "seed": run_seed,
            "save_dir": str(project / run_name),
        }
        for metric_name in SUMMARY_METRICS:
            result_row[metric_name] = safe_float(metrics.results_dict.get(metric_name))
        run_summaries.append(result_row)

    return run_summaries, summary_dir


def save_summary(run_summaries: list[dict], summary_dir: Path) -> None:
    detail_csv = summary_dir / "repeat_results.csv"
    summary_csv = summary_dir / "repeat_summary_mean_std.csv"

    fieldnames = ["repeat_id", "seed", "save_dir", *SUMMARY_METRICS]
    with detail_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_summaries)

    summary_rows = []
    for metric_name in SUMMARY_METRICS:
        values = [row[metric_name] for row in run_summaries if not math.isnan(row[metric_name])]
        if not values:
            summary_rows.append({"metric": metric_name, "mean": "nan", "std": "nan", "mean_std": "nan +/- nan"})
            continue
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": f"{mean(values):.4f}",
                "std": f"{(stdev(values) if len(values) > 1 else 0.0):.4f}",
                "mean_std": format_mean_std(values),
            }
        )

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "mean_std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n===== Repeated Experiment Summary =====")
    for row in summary_rows:
        print(f"{row['metric']}: {row['mean_std']}")
    print(f"\nDetailed results saved to: {detail_csv}")
    print(f"Mean+/-std summary saved to: {summary_csv}")


if __name__ == "__main__":
    repeats = TRAIN_CONFIG.get("repeats")
    if repeats is None:
        run_default_training(TRAIN_CONFIG)
    else:
        repeated_results, output_dir = run_repeated_experiments(TRAIN_CONFIG)
        save_summary(repeated_results, output_dir)
