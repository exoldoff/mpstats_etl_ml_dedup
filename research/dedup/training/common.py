from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import inspect
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import pandas as pd


DEFAULT_SPLIT_DATA_PATH = Path("research/dedup/data/training/dedup_pairs_final_split.csv")
DEFAULT_MODEL_OUTPUT_ROOT = Path("artifacts/models/dedup")

LABEL2ID = {"different_product": 0, "same_base_product": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

REQUIRED_SPLIT_COLUMNS = frozenset({"sentence_A", "sentence_B", "same_base_product", "split"})


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "model"


def read_split_pairs(path: Path, *, smoke_limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"split dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_SPLIT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"split dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["same_base_product"] = frame["same_base_product"].astype(int)
    frame["labels"] = frame["same_base_product"].astype(int)
    frame["sentence_A"] = frame["sentence_A"].fillna("").astype(str)
    frame["sentence_B"] = frame["sentence_B"].fillna("").astype(str)

    if smoke_limit is not None and smoke_limit > 0 and len(frame) > smoke_limit:
        frame = _limit_by_split(frame, smoke_limit)
    return frame.reset_index(drop=True)


def _limit_by_split(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    split_order = [split for split in ("train", "dev", "test") if split in set(frame["split"])]
    extra_splits = sorted(set(frame["split"]) - set(split_order))
    split_order.extend(extra_splits)
    if not split_order:
        return frame.head(limit).copy()

    per_split = max(1, limit // len(split_order))
    remainder = max(0, limit - per_split * len(split_order))
    parts = []
    for idx, split_name in enumerate(split_order):
        take = per_split + (1 if idx < remainder else 0)
        parts.append(frame[frame["split"].eq(split_name)].head(take))
    limited = pd.concat(parts, ignore_index=True, sort=False)
    return limited.head(limit).copy()


def split_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return frame[frame["split"].eq(split_name)].reset_index(drop=True)


def frame_to_pair_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["sentence_A", "sentence_B", "labels"]].copy()


def split_label_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    if frame.empty:
        return {}
    counts = frame.groupby("split")["same_base_product"].value_counts().unstack(fill_value=0)
    return {
        str(split_name): {str(label): int(value) for label, value in row.items()}
        for split_name, row in counts.iterrows()
    }


def package_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return output.strip() or None


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    parameters = signature.parameters
    if "eval_strategy" in kwargs and "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs = dict(kwargs)
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return {key: value for key, value in kwargs.items() if key in parameters and value is not None}


def trainer_tokenizer_kwarg(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def binary_metrics_from_predictions(labels: Any, predictions: Any) -> dict[str, float]:
    import numpy as np

    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    tp = int(((predictions_array == 1) & (labels_array == 1)).sum())
    fp = int(((predictions_array == 1) & (labels_array == 0)).sum())
    fn = int(((predictions_array == 0) & (labels_array == 1)).sum())
    tn = int(((predictions_array == 0) & (labels_array == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels_array) if len(labels_array) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
