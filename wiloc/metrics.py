"""Localisation error statistics."""

from __future__ import annotations

import numpy as np


def errors(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(est)[:, :2] - np.asarray(truth)[:, :2], axis=1)


def summarize(err: np.ndarray) -> dict:
    err = np.asarray(err)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p75": float(np.percentile(err, 75)),
        "p90": float(np.percentile(err, 90)),
        "max": float(np.max(err)),
    }


def ecdf(err: np.ndarray):
    x = np.sort(np.asarray(err))
    return x, np.arange(1, len(x) + 1) / len(x)


def format_table(rows: dict[str, dict], keys=("mean", "median", "rmse", "p75", "p90")) -> str:
    """Markdown table of metric dicts keyed by method name."""
    head = "| Method | " + " | ".join(f"{k} [m]" for k in keys) + " |"
    sep = "|---|" + "---:|" * len(keys)
    lines = [head, sep]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.2f}" for k in keys) + " |")
    return "\n".join(lines)
