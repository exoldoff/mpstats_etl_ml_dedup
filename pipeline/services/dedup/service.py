from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import RLock, Thread
from typing import Any, Callable, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from mpstats_app.config import AppSettings
from mpstats_app.repositories.duckdb_repository import DuckDbAppRepository
from pipeline.services.dedup.retrieval_cache import (
    RETRIEVAL_CACHE_SCHEMA_VERSION,
    RETRIEVAL_NORMALIZE_EMBEDDINGS,
    RETRIEVAL_TEXT_BUILDER_VERSION,
    RetrievalEmbeddingCache,
)
from pipeline.services.dedup.graph_grouping import (
    GraphGroupingConfig,
    build_family_components,
    normalize_graph_algorithm,
)


DEDUP_SETTINGS_KEY = "dedup_settings_json"
DEDUP_MODEL_PATH_ENV = "DEDUP_FINE_TUNED_MODEL_PATH"
DEDUP_TOP_K = 30
DEDUP_THRESHOLD_STRATEGY = "threshold_weighted_cost"
DEDUP_THRESHOLD_SAME = 0.872321
DEDUP_GRAPH_GROUPING_ALGORITHM = "leiden"
DEDUP_GRAPH_COMMUNITY_RESOLUTION = 0.1
DEDUP_GRAPH_COMMUNITY_SEED = 42
DEDUP_GRAPH_EDGE_WEIGHT_COL = "score"
DEDUP_CATEGORY_THRESHOLDS = {
    "sauces": 0.917444,
    "coconut_oil": 0.872321,
    "soap": 0.930329,
}
DEDUP_CATEGORY_THRESHOLD_ALIASES = {
    "sauce": "sauces",
    "sauces": "sauces",
    "соус": "sauces",
    "соусы": "sauces",
    "coconut_oil": "coconut_oil",
    "coconut-oil": "coconut_oil",
    "coconut oil": "coconut_oil",
    "кокосовое масло": "coconut_oil",
    "soap": "soap",
    "мыло": "soap",
}
DEDUP_METHOD = "ft_bge_reranker_v2_m3"
DEDUP_HF_MODEL_ID = "exoldoff/bge-reranker-v2-m3-cross-encoder-marketplaces-rus"
DEDUP_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEDUP_MODEL_DEVICE = "auto"
DEDUP_ACTIVATION = "sigmoid"
DEDUP_PROFILE_PATH = Path(__file__).with_name("model_profile.json")
E5_TEXT_PREFIX = "query: "
ELIGIBLE_CATEGORY_NAMES = {"соус", "соусы", "кокосовое масло", "мыло"}
DEDUP_ACTIVE_STATUSES = {"queued", "running"}
DEDUP_REPLACED_STATUSES = ["failed"]
DEDUP_PRODUCT_LEVELS = {"expanded", "family", "canonical"}

EmbeddingFactory = Callable[[str], Any]
CrossEncoderFactory = Callable[[str], Any]


class DedupRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DedupProfile:
    model_method: str = DEDUP_METHOD
    model_path: str = ""
    hf_model_id: str = DEDUP_HF_MODEL_ID
    embedding_model_name: str = DEDUP_EMBEDDING_MODEL
    model_device: str = DEDUP_MODEL_DEVICE
    activation: str = DEDUP_ACTIVATION
    threshold_strategy: str = DEDUP_THRESHOLD_STRATEGY
    threshold_same: float = DEDUP_THRESHOLD_SAME
    category_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEDUP_CATEGORY_THRESHOLDS))
    faiss_top_k: int = DEDUP_TOP_K
    graph_grouping_algorithm: str = DEDUP_GRAPH_GROUPING_ALGORITHM
    graph_community_resolution: float = DEDUP_GRAPH_COMMUNITY_RESOLUTION
    graph_community_seed: int | None = DEDUP_GRAPH_COMMUNITY_SEED
    graph_edge_weight_col: str = DEDUP_GRAPH_EDGE_WEIGHT_COL
    embedding_batch_size: int = 64
    cross_encoder_batch_size: int = 32

    @classmethod
    def from_settings(cls, payload: dict[str, Any]) -> "DedupProfile":
        tracked = _tracked_profile_config()
        return cls(
            model_method=str(tracked.get("method") or DEDUP_METHOD).strip(),
            model_path=str(payload.get("model_path") or os.environ.get(DEDUP_MODEL_PATH_ENV) or "").strip(),
            hf_model_id=str(payload.get("hf_model_id") or tracked.get("hf_model_id") or DEDUP_HF_MODEL_ID).strip(),
            embedding_model_name=str(
                payload.get("embedding_model_name") or tracked.get("embedding_model_name") or DEDUP_EMBEDDING_MODEL
            ).strip(),
            model_device=_model_device(payload.get("model_device") or tracked.get("model_device") or DEDUP_MODEL_DEVICE),
            activation=str(tracked.get("activation") or DEDUP_ACTIVATION).strip(),
            threshold_strategy=str(tracked.get("threshold_strategy") or DEDUP_THRESHOLD_STRATEGY).strip(),
            threshold_same=float(tracked.get("threshold_same") or DEDUP_THRESHOLD_SAME),
            category_thresholds=_category_thresholds_from_profile(tracked),
            faiss_top_k=_bounded_int(
                payload.get("faiss_top_k") if "faiss_top_k" in payload else tracked.get("faiss_top_k"),
                default=DEDUP_TOP_K,
                minimum=1,
                maximum=100,
            ),
            graph_grouping_algorithm=normalize_graph_algorithm(
                payload.get("graph_grouping_algorithm")
                if "graph_grouping_algorithm" in payload
                else tracked.get("graph_grouping_algorithm") or DEDUP_GRAPH_GROUPING_ALGORITHM
            ),
            graph_community_resolution=_positive_float(
                payload.get("graph_community_resolution")
                if "graph_community_resolution" in payload
                else tracked.get("graph_community_resolution"),
                default=DEDUP_GRAPH_COMMUNITY_RESOLUTION,
            ),
            graph_community_seed=_optional_int(
                payload.get("graph_community_seed")
                if "graph_community_seed" in payload
                else tracked.get("graph_community_seed"),
                default=DEDUP_GRAPH_COMMUNITY_SEED,
            ),
            graph_edge_weight_col=str(
                payload.get("graph_edge_weight_col")
                if "graph_edge_weight_col" in payload
                else tracked.get("graph_edge_weight_col") or DEDUP_GRAPH_EDGE_WEIGHT_COL
            ).strip()
            or DEDUP_GRAPH_EDGE_WEIGHT_COL,
            embedding_batch_size=max(1, int(payload.get("embedding_batch_size") or 64)),
            cross_encoder_batch_size=max(1, int(payload.get("cross_encoder_batch_size") or 32)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_method": self.model_method,
            "model_path": self.model_path,
            "hf_model_id": self.hf_model_id,
            "embedding_model_name": self.embedding_model_name,
            "model_device": self.model_device,
            "activation": self.activation,
            "threshold_strategy": self.threshold_strategy,
            "threshold_same": self.threshold_same,
            "category_thresholds": dict(self.category_thresholds),
            "faiss_top_k": self.faiss_top_k,
            "graph_grouping_algorithm": self.graph_grouping_algorithm,
            "graph_community_resolution": self.graph_community_resolution,
            "graph_community_seed": self.graph_community_seed,
            "graph_edge_weight_col": self.graph_edge_weight_col,
            "embedding_batch_size": self.embedding_batch_size,
            "cross_encoder_batch_size": self.cross_encoder_batch_size,
        }

    def threshold_for_category(self, *, category_key: object, category_name: object | None = None) -> float:
        for value in (category_key, category_name):
            normalized = _norm_threshold_key(value)
            if not normalized:
                continue
            direct = self.category_thresholds.get(normalized)
            if direct is not None:
                return float(direct)
            canonical = DEDUP_CATEGORY_THRESHOLD_ALIASES.get(normalized)
            if canonical and canonical in self.category_thresholds:
                return float(self.category_thresholds[canonical])
        return self.threshold_same

    def for_category(self, *, category_key: object, category_name: object | None = None) -> "DedupProfile":
        return replace(
            self,
            threshold_same=self.threshold_for_category(category_key=category_key, category_name=category_name),
        )

    def graph_config(self) -> GraphGroupingConfig:
        return GraphGroupingConfig(
            algorithm=self.graph_grouping_algorithm,
            edge_weight_col=self.graph_edge_weight_col,
            resolution=self.graph_community_resolution,
            seed=self.graph_community_seed,
        )


@dataclass(frozen=True)
class _EmbeddingCacheResult:
    embeddings: np.ndarray
    hits: int
    misses: int
    status: str
    dimension: int


@dataclass(frozen=True)
class _ScoreCacheResult:
    edges: pd.DataFrame
    hits: int
    misses: int


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except TypeError:
        return ""
    return " ".join(str(value).strip().split())


def _tracked_profile_config() -> dict[str, Any]:
    try:
        payload = json.loads(DEDUP_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _norm_threshold_key(value: object) -> str:
    return _clean_text(value).casefold().replace("ё", "е").replace("-", "_")


def _category_thresholds_from_profile(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("category_thresholds")
    if not isinstance(raw, dict):
        return dict(DEDUP_CATEGORY_THRESHOLDS)

    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        normalized = _norm_threshold_key(key)
        if not normalized:
            continue
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(threshold) and threshold > 0:
            thresholds[normalized] = threshold
    return thresholds or dict(DEDUP_CATEGORY_THRESHOLDS)


def _norm_text(value: object) -> str:
    return _clean_text(value).casefold().replace("ё", "е")


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)
    return max(minimum, min(maximum, number))


def _positive_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return number if math.isfinite(number) and number > 0 else float(default)


def _optional_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _model_device(value: object) -> str:
    device = _clean_text(value).casefold()
    return device if device in {"auto", "cpu", "mps", "cuda"} else DEDUP_MODEL_DEVICE


def _product_level(value: object) -> str:
    level = _clean_text(value).casefold()
    return level if level in DEDUP_PRODUCT_LEVELS else "expanded"


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _hash_id(*parts: object, prefix: str = "") -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _embedding_input_text(value: object) -> str:
    return E5_TEXT_PREFIX + _clean_text(value)


def _embedding_text_hash(value: object) -> str:
    return _sha256_text(_embedding_input_text(value))


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _pair_text_hash(text_a: object, text_b: object) -> str:
    return _sha256_text(json.dumps([_clean_text(text_a), _clean_text(text_b)], ensure_ascii=False, separators=(",", ":")))


def _safe_segment(value: object) -> str:
    text = re.sub(r"[^\w_.-]+", "_", _clean_text(value), flags=re.UNICODE)
    return text.strip("._") or "all"


def _numbers_close(left: object, right: object, *, abs_tol: float, rel_tol: float) -> bool:
    left_num = _to_float(left)
    right_num = _to_float(right)
    if left_num is None or right_num is None:
        return False
    return abs(left_num - right_num) <= max(abs_tol, rel_tol * max(abs(left_num), abs(right_num)))


def _same_pack(left: pd.Series, right: pd.Series) -> bool:
    unit_same = _numbers_close(left.get("unit_amount"), right.get("unit_amount"), abs_tol=0.02, rel_tol=0.05)
    total_same = _numbers_close(left.get("total_amount"), right.get("total_amount"), abs_tol=0.02, rel_tol=0.05)
    pack_same = _numbers_close(left.get("multipack_count"), right.get("multipack_count"), abs_tol=0.25, rel_tol=0.0)
    return unit_same and (total_same or pack_same)


def _pack_signature(row: pd.Series) -> str:
    unit = _to_float(row.get("unit_amount"))
    total = _to_float(row.get("total_amount"))
    pack = _to_float(row.get("multipack_count"))
    return "|".join(
        [
            f"u={round(unit, 3) if unit is not None else 'na'}",
            f"t={round(total, 3) if total is not None else 'na'}",
            f"p={round(pack, 2) if pack is not None else 'na'}",
        ]
    )


def _format_model_text(row: pd.Series) -> str:
    parts = [
        f"бренд: {_clean_text(row.get('brand'))}" if _clean_text(row.get("brand")) else "",
        f"подкатегория: {_clean_text(row.get('subcategory'))}" if _clean_text(row.get("subcategory")) else "",
        f"название: {_clean_text(row.get('sku'))}",
        f"вес единицы кг: {row.get('unit_amount')}" if _to_float(row.get("unit_amount")) is not None else "",
        f"общий вес кг: {row.get('total_amount')}" if _to_float(row.get("total_amount")) is not None else "",
        f"штук в упаковке: {row.get('multipack_count')}" if _to_float(row.get("multipack_count")) is not None else "",
    ]
    return " | ".join(part for part in parts if part)


def _next_sequence_id(prefix: str, existing_ids: set[str]) -> str:
    max_number = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for item in existing_ids:
        match = pattern.match(str(item))
        if match:
            max_number = max(max_number, int(match.group(1)))
    while True:
        max_number += 1
        candidate = f"{prefix}_{max_number:06d}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate


class DedupService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        repository: DuckDbAppRepository,
        embedding_model_factory: EmbeddingFactory | None = None,
        cross_encoder_factory: CrossEncoderFactory | None = None,
        faiss_module: Any | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self._embedding_model_factory = embedding_model_factory
        self._cross_encoder_factory = cross_encoder_factory
        self._faiss_module = faiss_module
        self._retrieval_cache = RetrievalEmbeddingCache(project_root=settings.project_root)
        self._lock = RLock()
        self._threads: dict[str, Thread] = {}

    def run_startup_maintenance(self) -> dict[str, int]:
        """Mark interrupted runs failed and keep only the latest failed retry."""
        with self._lock:
            stale_failed = self.repository.fail_stale_dedup_runs()
            failed_pruned = self.repository.prune_failed_dedup_runs()
        return {"stale_failed": stale_failed, "failed_pruned": failed_pruned}

    def get_settings(self) -> dict[str, Any]:
        raw = self.repository.get_setting(DEDUP_SETTINGS_KEY)
        payload: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = {}
        return self._settings_payload(DedupProfile.from_settings(payload))

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        merged = {**current, **payload}
        profile = DedupProfile.from_settings(merged)
        data = profile.to_dict()
        self.repository.set_setting(DEDUP_SETTINGS_KEY, json.dumps(data, ensure_ascii=False))
        return self._settings_payload(profile)

    def eligible_categories(self, *, project_name: str) -> dict[str, Any]:
        rows = self.repository.list_dedup_eligible_categories(project_name=project_name)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            latest = self.repository.latest_successful_dedup_run(
                project_name=project_name,
                category_key=str(row["category_key"]),
            )
            enriched.append({**row, "latest_successful_run": self._normalize_run(latest)})
        return {"project_name": project_name, "categories": enriched, "settings": self.get_settings()}

    def start_runs(self, *, project_name: str, category_keys: list[str], wait: bool = False) -> dict[str, Any]:
        eligible_rows = self.repository.list_dedup_eligible_categories(project_name=project_name)
        eligible: dict[str, dict[str, Any]] = {}
        for row in eligible_rows:
            category_key = str(row["category_key"])
            eligible[category_key] = row
            for source_category_key in row.get("source_category_keys") or []:
                clean_source_key = str(source_category_key).strip()
                if clean_source_key:
                    eligible[clean_source_key] = row
        clean_keys = [str(key) for key in category_keys if str(key).strip()]
        if not clean_keys:
            raise ValueError("Выбери хотя бы одну категорию для ML-дедупа.")
        unknown = [key for key in clean_keys if key not in eligible]
        if unknown:
            raise ValueError("ML-дедуп v1 доступен только для Соус/Соусы, Кокосовое масло и Мыло: " + ", ".join(unknown))

        base_profile = DedupProfile.from_settings(self.get_settings())
        runs: list[dict[str, Any]] = []
        selected_categories: list[dict[str, Any]] = []
        seen_category_keys: set[str] = set()
        for requested_key in clean_keys:
            category = eligible[requested_key]
            category_key = str(category["category_key"])
            if category_key in seen_category_keys:
                continue
            seen_category_keys.add(category_key)
            selected_categories.append(category)

        for category in selected_categories:
            category_key = str(category["category_key"])
            source_category_keys = [
                str(key).strip()
                for key in category.get("source_category_keys") or [category_key]
                if str(key or "").strip()
            ]
            lookup_category_keys = list(dict.fromkeys([category_key, *source_category_keys]))
            existing_runs = [
                self._normalize_run(run) or run
                for lookup_category_key in lookup_category_keys
                for run in self.repository.list_dedup_runs(
                    project_name=project_name,
                    category_key=lookup_category_key,
                    limit=10,
                )
            ]
            active_run = next(
                (run for run in existing_runs if str(run.get("status") or "") in DEDUP_ACTIVE_STATUSES),
                None,
            )
            if active_run is not None:
                runs.append(active_run)
                continue
            self.repository.delete_dedup_runs(
                project_name=project_name,
                category_keys=lookup_category_keys,
                statuses=DEDUP_REPLACED_STATUSES,
            )
            profile = base_profile.for_category(
                category_key=category_key,
                category_name=category.get("category_name"),
            )
            run_id = uuid4().hex
            run = self.repository.create_dedup_run(
                {
                    "run_id": run_id,
                    "project_name": project_name,
                    "category_key": category_key,
                    "category_name": category.get("category_name"),
                    "status": "queued",
                    **profile.to_dict(),
                    "manifest": {
                        "requested_at": datetime.now().isoformat(timespec="seconds"),
                        "source_category_keys": source_category_keys,
                        "source_marketplaces": list(category.get("marketplaces") or []),
                        "runtime_profile": {
                            "model_device": profile.model_device,
                            "graph_grouping_algorithm": profile.graph_grouping_algorithm,
                            "graph_community_resolution": profile.graph_community_resolution,
                            "graph_community_seed": profile.graph_community_seed,
                            "graph_edge_weight_col": profile.graph_edge_weight_col,
                            "embedding_batch_size": profile.embedding_batch_size,
                            "cross_encoder_batch_size": profile.cross_encoder_batch_size,
                        },
                        "progress_percent": 0,
                        "progress_stage": "queued",
                        "progress_message": "Ожидает запуска",
                    },
                }
            )
            runs.append(run)
            if wait:
                self._execute_run(run_id)
            else:
                thread = Thread(target=self._execute_run, args=(run_id,), daemon=True)
                with self._lock:
                    self._threads[run_id] = thread
                thread.start()
        return {"runs": [self._normalize_run(self.repository.get_dedup_run(str(run["run_id"])) or run) for run in runs]}

    def rebuild_graph_runs(self, *, project_name: str, category_keys: list[str], wait: bool = False) -> dict[str, Any]:
        eligible_rows = self.repository.list_dedup_eligible_categories(project_name=project_name)
        eligible: dict[str, dict[str, Any]] = {}
        for row in eligible_rows:
            category_key = str(row["category_key"])
            eligible[category_key] = row
            for source_category_key in row.get("source_category_keys") or []:
                clean_source_key = str(source_category_key).strip()
                if clean_source_key:
                    eligible[clean_source_key] = row
        clean_keys = [str(key) for key in category_keys if str(key).strip()]
        if not clean_keys:
            raise ValueError("Выбери хотя бы одну категорию для пересборки графа.")
        unknown = [key for key in clean_keys if key not in eligible]
        if unknown:
            raise ValueError("ML-дедуп v1 доступен только для Соус/Соусы, Кокосовое масло и Мыло: " + ", ".join(unknown))

        base_profile = DedupProfile.from_settings(self.get_settings())
        runs: list[dict[str, Any]] = []
        seen_category_keys: set[str] = set()
        for requested_key in clean_keys:
            category = eligible[requested_key]
            category_key = str(category["category_key"])
            if category_key in seen_category_keys:
                continue
            seen_category_keys.add(category_key)
            existing_runs = [
                self._normalize_run(run) or run
                for run in self.repository.list_dedup_runs(
                    project_name=project_name,
                    category_key=category_key,
                    limit=10,
                )
            ]
            active_run = next(
                (run for run in existing_runs if str(run.get("status") or "") in DEDUP_ACTIVE_STATUSES),
                None,
            )
            if active_run is not None:
                runs.append(active_run)
                continue
            source_run = self._latest_graph_rebuild_source_run(project_name=project_name, category_key=category_key)
            if not source_run:
                raise ValueError(f"Для {category.get('category_name') or category_key} ещё нет успешного ML-дедуп run.")
            profile = self._graph_rebuild_profile(source_run, base_profile.for_category(
                category_key=category_key,
                category_name=category.get("category_name"),
            ))
            run_id = uuid4().hex
            run = self.repository.create_dedup_run(
                {
                    "run_id": run_id,
                    "project_name": project_name,
                    "category_key": category_key,
                    "category_name": category.get("category_name"),
                    "status": "queued",
                    **profile.to_dict(),
                    "manifest": {
                        "requested_at": datetime.now().isoformat(timespec="seconds"),
                        "run_mode": "graph_only",
                        "source_run_id": source_run["run_id"],
                        "source_category_keys": self._source_category_keys_from_run(source_run),
                        "source_marketplaces": list(category.get("marketplaces") or []),
                        "runtime_profile": {
                            "model_device": profile.model_device,
                            "graph_grouping_algorithm": profile.graph_grouping_algorithm,
                            "graph_community_resolution": profile.graph_community_resolution,
                            "graph_community_seed": profile.graph_community_seed,
                            "graph_edge_weight_col": profile.graph_edge_weight_col,
                            "embedding_batch_size": profile.embedding_batch_size,
                            "cross_encoder_batch_size": profile.cross_encoder_batch_size,
                        },
                        "progress_percent": 0,
                        "progress_stage": "queued",
                        "progress_message": "Ожидает пересборки графа",
                    },
                }
            )
            runs.append(run)
            if wait:
                self._execute_graph_rebuild_run(run_id)
            else:
                thread = Thread(target=self._execute_graph_rebuild_run, args=(run_id,), daemon=True)
                with self._lock:
                    self._threads[run_id] = thread
                thread.start()
        return {"runs": [self._normalize_run(self.repository.get_dedup_run(str(run["run_id"])) or run) for run in runs]}

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.repository.get_dedup_run(run_id)
        if not run:
            raise ValueError("Dedup run не найден.")
        return self._normalize_run(run)

    def list_runs(self, *, project_name: str | None = None, category_key: str | None = None) -> dict[str, Any]:
        return {
            "runs": [
                self._normalize_run(run)
                for run in self.repository.list_dedup_runs(project_name=project_name, category_key=category_key)
            ]
        }

    def export_artifact(self, *, run_id: str, artifact: str) -> dict[str, Any]:
        self.get_run(run_id)
        return {"run_id": run_id, "artifact": artifact, "rows": self.repository.fetch_dedup_artifact(run_id=run_id, artifact=artifact)}

    def graph_report(self, *, run_id: str) -> dict[str, Any]:
        self.get_run(run_id)
        return self.repository.fetch_dedup_graph_report(run_id=run_id)

    def _latest_graph_rebuild_source_run(self, *, project_name: str, category_key: str) -> dict[str, Any] | None:
        runs = self.repository.list_dedup_runs(project_name=project_name, category_key=category_key, limit=50)
        successful = [run for run in runs if str(run.get("status") or "") == "success"]
        return next((run for run in successful if int(run.get("edge_count") or 0) > 0), successful[0] if successful else None)

    def products_browser(
        self,
        *,
        project_name: str,
        category_key: str | None = None,
        level: str = "expanded",
        query_text: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        payload = self.repository.fetch_dedup_products(
            project_name=project_name,
            category_key=category_key,
            level=level,
            query_text=query_text,
            limit=limit,
        )
        payload["project_name"] = project_name
        payload["category_key"] = category_key
        payload["level"] = _product_level(level)
        return payload

    def split_product_from_family(self, *, run_id: str, node_id: str, note: str | None = None) -> dict[str, Any]:
        override = self.repository.upsert_dedup_manual_split_override(
            run_id=run_id,
            node_id=node_id,
            note=note,
        )
        materialized_rows = self.repository.apply_dedup_manual_overrides(run_id=run_id)
        return {
            "run_id": run_id,
            "node_id": node_id,
            "override": override,
            "materialized_rows": materialized_rows,
        }

    def export_products(self, *, project_name: str, category_key: str | None = None, level: str = "expanded") -> Path:
        clean_level = "canonical" if str(level or "").strip().casefold() == "canonical" else "expanded"
        project_dir = self.settings.project_root / "data" / "projects" / _safe_segment(project_name) / "exports"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_part = _safe_segment(category_key or "all")
        target = project_dir / f"dedup_products_{category_part}_{clean_level}_{timestamp}.csv"
        result = self.repository.export_dedup_products_csv(
            project_name=project_name,
            category_key=category_key,
            level=clean_level,
            target=target,
        )
        return result.output_path

    def _execute_run(self, run_id: str) -> None:
        run = self.repository.get_dedup_run(run_id)
        if not run:
            return
        started = datetime.now()
        self.repository.update_dedup_run(run_id, {"status": "running", "started_at": started, "error": None})
        self._update_progress(
            run_id,
            run,
            percent=5,
            stage="read_source",
            message="Читаю строки куба",
        )
        try:
            runtime_profile = self._runtime_profile_from_run(run)
            profile = DedupProfile(
                model_method=str(run.get("model_method") or DEDUP_METHOD),
                model_path=str(run.get("model_path") or ""),
                hf_model_id=str(run.get("hf_model_id") or DEDUP_HF_MODEL_ID),
                embedding_model_name=str(run.get("embedding_model_name") or DEDUP_EMBEDDING_MODEL),
                model_device=_model_device(runtime_profile.get("model_device") or DEDUP_MODEL_DEVICE),
                activation=str(run.get("activation") or DEDUP_ACTIVATION),
                threshold_strategy=str(run.get("threshold_strategy") or DEDUP_THRESHOLD_STRATEGY),
                threshold_same=float(run.get("threshold_same") or DEDUP_THRESHOLD_SAME),
                category_thresholds=DedupProfile.from_settings({}).category_thresholds,
                faiss_top_k=int(run.get("faiss_top_k") or DEDUP_TOP_K),
                graph_grouping_algorithm=normalize_graph_algorithm(
                    runtime_profile.get("graph_grouping_algorithm") or DEDUP_GRAPH_GROUPING_ALGORITHM
                ),
                graph_community_resolution=_positive_float(
                    runtime_profile.get("graph_community_resolution"),
                    default=DEDUP_GRAPH_COMMUNITY_RESOLUTION,
                ),
                graph_community_seed=_optional_int(
                    runtime_profile.get("graph_community_seed"),
                    default=DEDUP_GRAPH_COMMUNITY_SEED,
                ),
                graph_edge_weight_col=str(runtime_profile.get("graph_edge_weight_col") or DEDUP_GRAPH_EDGE_WEIGHT_COL).strip()
                or DEDUP_GRAPH_EDGE_WEIGHT_COL,
                embedding_batch_size=max(1, int(runtime_profile.get("embedding_batch_size") or 64)),
                cross_encoder_batch_size=max(1, int(runtime_profile.get("cross_encoder_batch_size") or 32)),
            )
            source = self.repository.fetch_dedup_source_dataframe(
                table_name=self.settings.products_table,
                project_name=str(run["project_name"]),
                category_keys=self._source_category_keys_from_run(run),
            )
            if source.empty:
                raise DedupRuntimeError("Нет строк куба для выбранной категории.")
            self._update_progress(
                run_id,
                run,
                percent=15,
                stage="build_nodes",
                message=f"Прочитано строк куба: {len(source):,}".replace(",", " "),
            )

            nodes = self._build_nodes(source, run_id=run_id)
            self._update_progress(
                run_id,
                run,
                percent=25,
                stage="load_embeddings",
                message=f"Собрано SKU-node: {len(nodes):,}".replace(",", " "),
                node_count=len(nodes),
            )
            node_ids = [str(item) for item in nodes["node_id"].tolist()]
            manual_overrides = self.repository.fetch_dedup_manual_overrides(
                project_name=str(run["project_name"]),
                category_key=str(run["category_key"]),
                node_ids=node_ids,
            )
            if len(nodes) < 2:
                cache_metadata = self._retrieval_cache.skipped("not_enough_nodes").to_manifest()
                self._update_progress(
                    run_id,
                    run,
                    percent=55,
                    stage="score_pairs",
                    message="SKU-node меньше двух, проверяю модель",
                    node_count=len(nodes),
                    candidate_count=0,
                )
                self._load_cross_encoder(profile)
                groups = self._build_groups(
                    nodes,
                    edges=pd.DataFrame(),
                    graph_config=profile.graph_config(),
                    manual_overrides=manual_overrides,
                )
                self._update_progress(
                    run_id,
                    run,
                    percent=85,
                    stage="materialize",
                    message="Записываю каноническую таблицу",
                    node_count=len(nodes),
                    candidate_count=0,
                    edge_count=0,
                    group_count=len(groups),
                )
                self.repository.replace_dedup_nodes(run_id, nodes.to_dict(orient="records"))
                self.repository.replace_dedup_edges(run_id, [])
                self.repository.replace_dedup_groups(run_id, groups.to_dict(orient="records"))
                materialized_rows = self.repository.refresh_dedup_products_table(run_id=run_id)
                manifest_path = self._write_manifest(
                    run_id,
                    run,
                    profile,
                    nodes,
                    pd.DataFrame(),
                    groups,
                    retrieval_cache=cache_metadata,
                    materialized_rows=materialized_rows,
                )
                self.repository.update_dedup_run(
                    run_id,
                    {
                        "status": "success",
                        "node_count": len(nodes),
                        "candidate_count": 0,
                        "edge_count": 0,
                        "group_count": len(groups),
                        "manifest_path": str(manifest_path),
                        "manifest_json": self._run_manifest_json(
                            manifest_path=manifest_path,
                            profile=profile,
                            retrieval_cache=cache_metadata,
                            materialized_rows=materialized_rows,
                            source_category_keys=self._source_category_keys_from_run(run),
                            progress_percent=100,
                            progress_stage="success",
                            progress_message=f"Готово: {materialized_rows:,} строк в mpstats_products_dedup".replace(",", " "),
                        ),
                        "finished_at": datetime.now(),
                    },
                )
                return

            previous_assignment_rows = self.repository.fetch_dedup_identity_assignments(
                project_name=str(run["project_name"]),
                category_key=str(run["category_key"]),
                model_method=profile.model_method,
                model_path=profile.model_path.strip(),
                hf_model_id=profile.hf_model_id.strip(),
                embedding_model_name=profile.embedding_model_name,
                faiss_top_k=profile.faiss_top_k,
                graph_grouping_algorithm=profile.graph_grouping_algorithm,
                graph_community_resolution=profile.graph_community_resolution,
                graph_community_seed=profile.graph_community_seed,
                graph_edge_weight_col=profile.graph_edge_weight_col,
                activation=profile.activation.strip(),
                threshold_strategy=profile.threshold_strategy,
                threshold_same=profile.threshold_same,
                node_ids=node_ids,
            )
            previous_assignments = {str(row["node_id"]): row for row in previous_assignment_rows}
            known_node_ids = set(previous_assignments)
            new_node_ids = sorted(set(node_ids) - known_node_ids)

            if not new_node_ids:
                cache_metadata = {
                    "status": "identity_hit",
                    "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "cache_key": None,
                    "cache_path": "duckdb:dedup_identity_assignments",
                    "embedding_shape": [0, 0],
                    "cache_rebuild_reason": None,
                    "identity_hits": len(known_node_ids),
                    "identity_misses": 0,
                    "embedding_cache_hits": 0,
                    "embedding_cache_misses": 0,
                    "score_cache_hits": 0,
                    "score_cache_misses": 0,
                }
                self._update_progress(
                    run_id,
                    run,
                    percent=78,
                    stage="build_groups",
                    message=f"Все SKU-node найдены в identity-cache: {len(known_node_ids):,}, ML scoring не нужен".replace(",", " "),
                    node_count=len(nodes),
                    candidate_count=0,
                    edge_count=0,
                )
                candidates = pd.DataFrame()
                edges = pd.DataFrame()
            else:
                cache_load = self._retrieval_cache.load(
                    nodes,
                    project_name=str(run["project_name"]),
                    category_key=str(run["category_key"]),
                    embedding_model_name=profile.embedding_model_name,
                )
                if cache_load.embeddings is not None:
                    embeddings = cache_load.embeddings
                    cache_metadata = cache_load.metadata.to_manifest()
                    cache_metadata.update(
                        {
                            "identity_hits": len(known_node_ids),
                            "identity_misses": len(new_node_ids),
                            "embedding_cache_hits": len(nodes),
                            "embedding_cache_misses": 0,
                        }
                    )
                    self._update_progress(
                        run_id,
                        run,
                        percent=45,
                        stage="build_candidates",
                        message=f"Embeddings взяты из snapshot-cache, новых SKU-node: {len(new_node_ids):,}".replace(",", " "),
                        node_count=len(nodes),
                    )
                else:
                    embedding_result = self._encode_nodes_with_cache(
                        nodes,
                        profile,
                        project_name=str(run["project_name"]),
                        category_key=str(run["category_key"]),
                    )
                    embeddings = embedding_result.embeddings
                    cache_metadata = self._retrieval_cache.write(
                        nodes,
                        embeddings,
                        project_name=str(run["project_name"]),
                        category_key=str(run["category_key"]),
                        embedding_model_name=profile.embedding_model_name,
                        rebuild_reason=str(cache_load.metadata.cache_rebuild_reason or "miss"),
                    ).to_manifest()
                    cache_metadata.update(
                        {
                            "status": embedding_result.status,
                            "identity_hits": len(known_node_ids),
                            "identity_misses": len(new_node_ids),
                            "embedding_cache_hits": embedding_result.hits,
                            "embedding_cache_misses": embedding_result.misses,
                            "embedding_shape": [len(nodes), embedding_result.dimension],
                        }
                    )
                    self._update_progress(
                        run_id,
                        run,
                        percent=45,
                        stage="build_candidates",
                        message=(
                            f"Embeddings cache: {embedding_result.hits:,} hit / {embedding_result.misses:,} новых, "
                            f"собираю FAISS-кандидаты"
                        ).replace(",", " "),
                        node_count=len(nodes),
                    )
                candidates = self._generate_candidates(nodes, embeddings, profile, query_node_ids=new_node_ids)
                if candidates.empty:
                    self._update_progress(
                        run_id,
                        run,
                        percent=78,
                        stage="build_groups",
                        message="FAISS не нашёл пар для новых SKU-node, добавляю их как singleton",
                        node_count=len(nodes),
                        candidate_count=0,
                        edge_count=0,
                    )
                    edges = pd.DataFrame()
                    cache_metadata.update({"score_cache_hits": 0, "score_cache_misses": 0})
                else:
                    self._update_progress(
                        run_id,
                        run,
                        percent=60,
                        stage="score_pairs",
                        message=f"FAISS-кандидатов для новых SKU: {len(candidates):,}, проверяю cross-encoder cache".replace(",", " "),
                        node_count=len(nodes),
                        candidate_count=len(candidates),
                    )
                    score_result = self._score_candidates(
                        nodes,
                        candidates,
                        profile,
                        run_id=run_id,
                        run=run,
                        project_name=str(run["project_name"]),
                        category_key=str(run["category_key"]),
                    )
                    edges = score_result.edges
                    cache_metadata.update(
                        {
                            "score_cache_hits": score_result.hits,
                            "score_cache_misses": score_result.misses,
                        }
                    )
            self._update_progress(
                run_id,
                run,
                percent=78,
                stage="build_groups",
                message=f"Модельных пар: {len(edges):,}, собираю группы".replace(",", " "),
                node_count=len(nodes),
                candidate_count=len(candidates),
                edge_count=len(edges),
            )
            groups = self._build_groups(
                nodes,
                edges,
                graph_config=profile.graph_config(),
                previous_assignments=previous_assignments,
                manual_overrides=manual_overrides,
            )
            self._update_progress(
                run_id,
                run,
                percent=88,
                stage="materialize",
                message=f"Групп: {len(groups):,}, записываю таблицу".replace(",", " "),
                node_count=len(nodes),
                candidate_count=len(candidates),
                edge_count=len(edges),
                group_count=len(groups),
            )

            self.repository.replace_dedup_nodes(run_id, nodes.to_dict(orient="records"))
            self.repository.replace_dedup_edges(run_id, edges.to_dict(orient="records"))
            self.repository.replace_dedup_groups(run_id, groups.to_dict(orient="records"))
            self.repository.upsert_dedup_identity_assignments(
                project_name=str(run["project_name"]),
                category_key=str(run["category_key"]),
                model_method=profile.model_method,
                model_path=profile.model_path.strip(),
                hf_model_id=profile.hf_model_id.strip(),
                embedding_model_name=profile.embedding_model_name,
                faiss_top_k=profile.faiss_top_k,
                graph_grouping_algorithm=profile.graph_grouping_algorithm,
                graph_community_resolution=profile.graph_community_resolution,
                graph_community_seed=profile.graph_community_seed,
                graph_edge_weight_col=profile.graph_edge_weight_col,
                activation=profile.activation.strip(),
                threshold_strategy=profile.threshold_strategy,
                threshold_same=profile.threshold_same,
                run_id=run_id,
                rows=groups.to_dict(orient="records"),
            )
            materialized_rows = self.repository.refresh_dedup_products_table(run_id=run_id)
            manifest_path = self._write_manifest(
                run_id,
                run,
                profile,
                nodes,
                edges,
                groups,
                retrieval_cache=cache_metadata,
                materialized_rows=materialized_rows,
            )
            self.repository.update_dedup_run(
                run_id,
                {
                    "status": "success",
                    "node_count": len(nodes),
                    "candidate_count": len(candidates),
                    "edge_count": len(edges),
                    "group_count": len(groups),
                    "manifest_path": str(manifest_path),
                    "manifest_json": self._run_manifest_json(
                        manifest_path=manifest_path,
                        profile=profile,
                        retrieval_cache=cache_metadata,
                        materialized_rows=materialized_rows,
                        source_category_keys=self._source_category_keys_from_run(run),
                        progress_percent=100,
                        progress_stage="success",
                        progress_message=f"Готово: {materialized_rows:,} строк в mpstats_products_dedup".replace(",", " "),
                    ),
                    "finished_at": datetime.now(),
                },
            )
        except Exception as exc:
            try:
                self._update_progress(
                    run_id,
                    run,
                    percent=100,
                    stage="failed",
                    message=f"Ошибка: {type(exc).__name__}",
                )
            except Exception:
                pass
            self.repository.update_dedup_run(
                run_id,
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": datetime.now(),
                },
            )

    @staticmethod
    def _graph_rebuild_profile(source_run: dict[str, Any], graph_profile: DedupProfile) -> DedupProfile:
        threshold_same = _to_float(source_run.get("threshold_same"))
        return DedupProfile(
            model_method=str(source_run.get("model_method") or graph_profile.model_method).strip(),
            model_path=str(source_run.get("model_path") or graph_profile.model_path).strip(),
            hf_model_id=str(source_run.get("hf_model_id") or graph_profile.hf_model_id).strip(),
            embedding_model_name=str(source_run.get("embedding_model_name") or graph_profile.embedding_model_name).strip(),
            model_device=graph_profile.model_device,
            activation=str(source_run.get("activation") or graph_profile.activation).strip(),
            threshold_strategy=str(source_run.get("threshold_strategy") or graph_profile.threshold_strategy).strip(),
            threshold_same=float(threshold_same if threshold_same is not None else graph_profile.threshold_same),
            category_thresholds=dict(graph_profile.category_thresholds),
            faiss_top_k=_bounded_int(
                source_run.get("faiss_top_k"),
                default=graph_profile.faiss_top_k,
                minimum=1,
                maximum=100,
            ),
            graph_grouping_algorithm=graph_profile.graph_grouping_algorithm,
            graph_community_resolution=graph_profile.graph_community_resolution,
            graph_community_seed=graph_profile.graph_community_seed,
            graph_edge_weight_col=graph_profile.graph_edge_weight_col,
            embedding_batch_size=graph_profile.embedding_batch_size,
            cross_encoder_batch_size=graph_profile.cross_encoder_batch_size,
        )

    def _execute_graph_rebuild_run(self, run_id: str) -> None:
        run = self.repository.get_dedup_run(run_id)
        if not run:
            return
        started = datetime.now()
        self.repository.update_dedup_run(run_id, {"status": "running", "started_at": started, "error": None})
        self._update_progress(
            run_id,
            run,
            percent=8,
            stage="copy_scoring",
            message="Копирую готовые SKU-node и модельные пары",
        )
        try:
            manifest = self._manifest_from_run(run)
            source_run_id = str(manifest.get("source_run_id") or "").strip()
            if not source_run_id:
                raise DedupRuntimeError("Для graph-only run не найден source_run_id.")
            source_run = self.repository.get_dedup_run(source_run_id)
            if not source_run:
                raise DedupRuntimeError("Исходный ML-дедуп run для пересборки графа не найден.")
            if str(source_run.get("status") or "") != "success":
                raise DedupRuntimeError("Граф можно пересобрать только из успешного ML-дедуп run.")

            runtime_profile = self._runtime_profile_from_run(run)
            profile = DedupProfile(
                model_method=str(run.get("model_method") or DEDUP_METHOD),
                model_path=str(run.get("model_path") or ""),
                hf_model_id=str(run.get("hf_model_id") or DEDUP_HF_MODEL_ID),
                embedding_model_name=str(run.get("embedding_model_name") or DEDUP_EMBEDDING_MODEL),
                model_device=_model_device(runtime_profile.get("model_device") or DEDUP_MODEL_DEVICE),
                activation=str(run.get("activation") or DEDUP_ACTIVATION),
                threshold_strategy=str(run.get("threshold_strategy") or DEDUP_THRESHOLD_STRATEGY),
                threshold_same=float(run.get("threshold_same") or DEDUP_THRESHOLD_SAME),
                category_thresholds=DedupProfile.from_settings({}).category_thresholds,
                faiss_top_k=int(run.get("faiss_top_k") or DEDUP_TOP_K),
                graph_grouping_algorithm=normalize_graph_algorithm(
                    runtime_profile.get("graph_grouping_algorithm") or DEDUP_GRAPH_GROUPING_ALGORITHM
                ),
                graph_community_resolution=_positive_float(
                    runtime_profile.get("graph_community_resolution"),
                    default=DEDUP_GRAPH_COMMUNITY_RESOLUTION,
                ),
                graph_community_seed=_optional_int(
                    runtime_profile.get("graph_community_seed"),
                    default=DEDUP_GRAPH_COMMUNITY_SEED,
                ),
                graph_edge_weight_col=str(runtime_profile.get("graph_edge_weight_col") or DEDUP_GRAPH_EDGE_WEIGHT_COL).strip()
                or DEDUP_GRAPH_EDGE_WEIGHT_COL,
                embedding_batch_size=max(1, int(runtime_profile.get("embedding_batch_size") or 64)),
                cross_encoder_batch_size=max(1, int(runtime_profile.get("cross_encoder_batch_size") or 32)),
            )

            copy_counts = self.repository.copy_dedup_scoring_artifacts(
                source_run_id=source_run_id,
                target_run_id=run_id,
            )
            nodes = pd.DataFrame(self.repository.fetch_dedup_nodes_for_run(run_id=run_id))
            if nodes.empty:
                raise DedupRuntimeError("В исходном run нет SKU-node для пересборки графа.")
            positive_edges = pd.DataFrame(self.repository.fetch_dedup_positive_edges_for_run(run_id=run_id))
            node_ids = [str(item) for item in nodes["node_id"].tolist()]
            manual_overrides = self.repository.fetch_dedup_manual_overrides(
                project_name=str(run["project_name"]),
                category_key=str(run["category_key"]),
                node_ids=node_ids,
            )
            self._update_progress(
                run_id,
                run,
                percent=60,
                stage="build_groups",
                message=f"Positive-рёбер: {len(positive_edges):,}, пересобираю граф".replace(",", " "),
                node_count=len(nodes),
                candidate_count=int(source_run.get("candidate_count") or copy_counts.get("edge_count") or 0),
                edge_count=copy_counts.get("edge_count", 0),
            )
            groups = self._build_groups(
                nodes,
                positive_edges,
                graph_config=profile.graph_config(),
                manual_overrides=manual_overrides,
            )
            self._update_progress(
                run_id,
                run,
                percent=86,
                stage="materialize",
                message=f"Групп: {len(groups):,}, записываю результат".replace(",", " "),
                node_count=len(nodes),
                candidate_count=int(source_run.get("candidate_count") or copy_counts.get("edge_count") or 0),
                edge_count=copy_counts.get("edge_count", 0),
                group_count=len(groups),
            )
            self.repository.replace_dedup_groups(run_id, groups.to_dict(orient="records"))
            self.repository.upsert_dedup_identity_assignments(
                project_name=str(run["project_name"]),
                category_key=str(run["category_key"]),
                model_method=profile.model_method,
                model_path=profile.model_path.strip(),
                hf_model_id=profile.hf_model_id.strip(),
                embedding_model_name=profile.embedding_model_name,
                faiss_top_k=profile.faiss_top_k,
                graph_grouping_algorithm=profile.graph_grouping_algorithm,
                graph_community_resolution=profile.graph_community_resolution,
                graph_community_seed=profile.graph_community_seed,
                graph_edge_weight_col=profile.graph_edge_weight_col,
                activation=profile.activation.strip(),
                threshold_strategy=profile.threshold_strategy,
                threshold_same=profile.threshold_same,
                run_id=run_id,
                rows=groups.to_dict(orient="records"),
            )
            materialized_rows = self.repository.refresh_dedup_products_table(run_id=run_id)
            edge_count_frame = pd.DataFrame(index=range(int(copy_counts.get("edge_count") or len(positive_edges))))
            cache_metadata = {
                "status": "graph_only",
                "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                "cache_key": None,
                "cache_path": f"duckdb:dedup_sku_edges:{source_run_id}",
                "embedding_shape": [0, 0],
                "cache_rebuild_reason": "graph_profile_changed",
                "identity_hits": 0,
                "identity_misses": len(node_ids),
                "embedding_cache_hits": 0,
                "embedding_cache_misses": 0,
                "score_cache_hits": copy_counts.get("edge_count", 0),
                "score_cache_misses": 0,
            }
            manifest_path = self._write_manifest(
                run_id,
                run,
                profile,
                nodes,
                edge_count_frame,
                groups,
                retrieval_cache=cache_metadata,
                materialized_rows=materialized_rows,
            )
            graph_report = self.repository.fetch_dedup_graph_report(run_id=run_id)
            manifest_json = self._run_manifest_json(
                manifest_path=manifest_path,
                profile=profile,
                retrieval_cache=cache_metadata,
                materialized_rows=materialized_rows,
                source_category_keys=self._source_category_keys_from_run(source_run),
                progress_percent=100,
                progress_stage="success",
                progress_message=f"Граф пересобран: {materialized_rows:,} строк в mpstats_products_dedup".replace(",", " "),
            )
            manifest_json.update(
                {
                    "run_mode": "graph_only",
                    "source_run_id": source_run_id,
                    "source_edge_count": copy_counts.get("edge_count", 0),
                    "positive_edge_count": graph_report["summary"].get("positive_edge_count"),
                    "graph_report": graph_report["summary"],
                }
            )
            self.repository.update_dedup_run(
                run_id,
                {
                    "status": "success",
                    "node_count": len(nodes),
                    "candidate_count": int(source_run.get("candidate_count") or copy_counts.get("edge_count") or 0),
                    "edge_count": copy_counts.get("edge_count", 0),
                    "group_count": len(groups),
                    "manifest_path": str(manifest_path),
                    "manifest_json": manifest_json,
                    "finished_at": datetime.now(),
                },
            )
        except Exception as exc:
            try:
                self._update_progress(
                    run_id,
                    run,
                    percent=100,
                    stage="failed",
                    message=f"Ошибка: {type(exc).__name__}",
                )
            except Exception:
                pass
            self.repository.update_dedup_run(
                run_id,
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": datetime.now(),
                },
            )

    def _build_nodes(self, source: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
        frame = source.copy()
        frame["article"] = frame["article"].map(_clean_text)
        frame = frame[frame["article"].ne("")]
        if frame.empty:
            raise DedupRuntimeError("В выбранном срезе нет артикула или business hash для построения SKU-node.")
        frame["marketplace_code"] = frame["marketplace_code"].map(_clean_text)
        group_cols = ["project_name", "category_key", "marketplace_code", "article"]
        rows: list[dict[str, Any]] = []
        for key, group in frame.groupby(group_cols, dropna=False):
            project_name, category_key, marketplace_code, article = [str(item) for item in key]
            unit_amount = group["unit_amount"].dropna().iloc[0] if group["unit_amount"].notna().any() else None
            total_amount = group["total_amount"].dropna().iloc[0] if group["total_amount"].notna().any() else None
            unit = _to_float(unit_amount)
            total = _to_float(total_amount)
            multipack = round(total / unit) if unit and total else None
            node = {
                "run_id": run_id,
                "node_id": _hash_id(project_name, category_key, marketplace_code, article, prefix="sku_"),
                "project_name": project_name,
                "category_key": category_key,
                "category_name": _first_non_empty(group["category_name"]),
                "marketplace_code": marketplace_code,
                "marketplace": _first_non_empty(group["marketplace"]),
                "article": article,
                "sku": _first_non_empty(group["sku"]),
                "brand": _first_non_empty(group["brand"]),
                "subcategory": _first_non_empty(group["subcategory"]),
                "unit_amount": unit,
                "total_amount": total,
                "multipack_count": float(multipack) if multipack else None,
                "sales_volume": float(pd.to_numeric(group["sales_volume"], errors="coerce").fillna(0).sum()),
                "revenue": float(pd.to_numeric(group["revenue"], errors="coerce").fillna(0).sum()),
                "row_count": int(len(group)),
                "source_row_hashes_json": json.dumps(sorted({_clean_text(item) for item in group["row_hash"] if _clean_text(item)}), ensure_ascii=False),
            }
            node["embedding_text"] = _format_model_text(pd.Series(node))
            rows.append(node)
        return pd.DataFrame(rows).sort_values(["category_key", "marketplace_code", "article"]).reset_index(drop=True)

    def _load_embedding_model(self, profile: DedupProfile) -> Any:
        if self._embedding_model_factory is not None:
            return self._embedding_model_factory(profile.embedding_model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise DedupRuntimeError("Для ML-дедупа нужен sentence-transformers. Установи зависимости из requirements.txt.") from exc
        device = self._model_device_arg(profile)
        if device:
            return SentenceTransformer(profile.embedding_model_name, device=device)
        return SentenceTransformer(profile.embedding_model_name)

    def _load_cross_encoder(self, profile: DedupProfile) -> Any:
        model_name = profile.model_path.strip() or profile.hf_model_id.strip()
        if not model_name:
            raise DedupRuntimeError("Fine-tuned BGE model_path не настроен и hf_model_id пуст.")
        if profile.model_path.strip() and not Path(profile.model_path).expanduser().exists():
            raise DedupRuntimeError(f"Fine-tuned BGE model_path не найден: {profile.model_path}")
        if self._cross_encoder_factory is not None:
            return self._cross_encoder_factory(model_name)
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:
            raise DedupRuntimeError(
                "Для fine-tuned BGE scoring нужен sentence-transformers. Установи зависимости из requirements.txt."
            ) from exc
        device = self._model_device_arg(profile)
        if device:
            return CrossEncoder(model_name, device=device)
        return CrossEncoder(model_name)

    @staticmethod
    def _model_device_arg(profile: DedupProfile) -> str | None:
        device = _model_device(profile.model_device)
        if device == "auto":
            return None
        if device == "cpu":
            return "cpu"
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise DedupRuntimeError(f"Для устройства {device} нужен torch.") from exc
        if device == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise DedupRuntimeError("Устройство MPS недоступно в текущем Python/torch окружении.")
            return "mps"
        if device == "cuda":
            if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
                raise DedupRuntimeError("Устройство CUDA недоступно в текущем Python/torch окружении.")
            return "cuda"
        return None

    def _encode_nodes(self, nodes: pd.DataFrame, profile: DedupProfile) -> np.ndarray:
        model = self._load_embedding_model(profile)
        texts = [_embedding_input_text(text) for text in nodes["embedding_text"].tolist()]
        vectors = model.encode(
            texts,
            batch_size=profile.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        output = np.asarray(vectors, dtype="float32")
        if output.ndim != 2 or len(output) != len(nodes):
            raise DedupRuntimeError("Embedding model вернула некорректную форму vectors.")
        return output

    def _encode_nodes_with_cache(
        self,
        nodes: pd.DataFrame,
        profile: DedupProfile,
        *,
        project_name: str,
        category_key: str,
    ) -> _EmbeddingCacheResult:
        entries = [
            {
                "node_id": str(row.node_id),
                "embedding_text_hash": _embedding_text_hash(row.embedding_text),
            }
            for row in nodes.itertuples(index=False)
        ]
        cached = self.repository.fetch_dedup_node_embeddings(
            project_name=project_name,
            category_key=category_key,
            embedding_model_name=profile.embedding_model_name,
            text_builder_version=RETRIEVAL_TEXT_BUILDER_VERSION,
            normalize_embeddings=RETRIEVAL_NORMALIZE_EMBEDDINGS,
            entries=entries,
        )
        vectors_by_node: dict[str, np.ndarray] = {}
        dimension: int | None = None
        missing_rows: list[tuple[int, Any]] = []
        for index, row in enumerate(nodes.itertuples(index=False)):
            node_id = str(row.node_id)
            cached_row = cached.get(node_id)
            if cached_row:
                vector = np.frombuffer(cached_row["embedding_blob"], dtype="float32")
                cached_dimension = int(cached_row["embedding_dimension"])
                if cached_dimension > 0 and len(vector) == cached_dimension:
                    vectors_by_node[node_id] = vector
                    dimension = dimension or cached_dimension
                    continue
            missing_rows.append((index, row))

        if missing_rows:
            model = self._load_embedding_model(profile)
            texts = [_embedding_input_text(row.embedding_text) for _, row in missing_rows]
            encoded = model.encode(
                texts,
                batch_size=profile.embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            missing_vectors = np.asarray(encoded, dtype="float32")
            if missing_vectors.ndim != 2 or len(missing_vectors) != len(missing_rows):
                raise DedupRuntimeError("Embedding model вернула некорректную форму vectors.")
            dimension = int(missing_vectors.shape[1])
            cache_rows: list[dict[str, Any]] = []
            for offset, (_, row) in enumerate(missing_rows):
                node_id = str(row.node_id)
                vector = np.ascontiguousarray(missing_vectors[offset], dtype="float32")
                vectors_by_node[node_id] = vector
                cache_rows.append(
                    {
                        "node_id": node_id,
                        "embedding_text_hash": _embedding_text_hash(row.embedding_text),
                        "embedding_dimension": dimension,
                        "embedding_blob": vector.tobytes(),
                    }
                )
            self.repository.upsert_dedup_node_embeddings(
                project_name=project_name,
                category_key=category_key,
                embedding_model_name=profile.embedding_model_name,
                text_builder_version=RETRIEVAL_TEXT_BUILDER_VERSION,
                normalize_embeddings=RETRIEVAL_NORMALIZE_EMBEDDINGS,
                rows=cache_rows,
            )

        if dimension is None:
            raise DedupRuntimeError("Embedding cache не вернул размерность vectors.")
        output_rows: list[np.ndarray] = []
        for row in nodes.itertuples(index=False):
            vector = vectors_by_node.get(str(row.node_id))
            if vector is None or len(vector) != dimension:
                raise DedupRuntimeError("Embedding cache вернул несовместимую размерность vectors.")
            output_rows.append(np.asarray(vector, dtype="float32"))
        output = np.ascontiguousarray(np.vstack(output_rows), dtype="float32")
        hits = len(nodes) - len(missing_rows)
        misses = len(missing_rows)
        status = "hit" if misses == 0 else "partial_hit" if hits else "rebuilt"
        return _EmbeddingCacheResult(
            embeddings=output,
            hits=hits,
            misses=misses,
            status=status,
            dimension=dimension,
        )

    def _load_faiss(self) -> Any:
        if self._faiss_module is not None:
            return self._faiss_module
        try:
            import faiss
        except ModuleNotFoundError as exc:
            raise DedupRuntimeError("Для FAISS retrieval нужен faiss-cpu. Установи зависимости из requirements.txt.") from exc
        return faiss

    def _generate_candidates(
        self,
        nodes: pd.DataFrame,
        embeddings: np.ndarray,
        profile: DedupProfile,
        *,
        query_node_ids: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        faiss = self._load_faiss()
        pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
        nodes_with_idx = nodes.reset_index(drop=True).copy()
        nodes_with_idx["_row_idx"] = nodes_with_idx.index
        nodes_with_idx["_subcategory_norm"] = nodes_with_idx["subcategory"].map(_norm_text)
        query_set = {str(item) for item in query_node_ids} if query_node_ids is not None else set(nodes_with_idx["node_id"].astype(str))
        if not query_set:
            return pd.DataFrame()

        known_groups = [
            group
            for _, group in nodes_with_idx[nodes_with_idx["_subcategory_norm"].ne("")].groupby("_subcategory_norm", dropna=False)
            if len(group) > 1 and group["node_id"].astype(str).isin(query_set).any()
        ]
        known_indexes = {int(idx) for group in known_groups for idx in group["_row_idx"].tolist()}
        for group in known_groups:
            query_indexes = group.loc[group["node_id"].astype(str).isin(query_set), "_row_idx"].astype(int).tolist()
            self._search_group(
                faiss,
                nodes,
                embeddings,
                group["_row_idx"].tolist(),
                query_indexes,
                pair_rows,
                top_k=profile.faiss_top_k,
                blocking_scope="same_subcategory",
            )
        fallback_group = nodes_with_idx[~nodes_with_idx["_row_idx"].isin(known_indexes)]
        fallback_query_indexes = fallback_group.loc[
            fallback_group["node_id"].astype(str).isin(query_set),
            "_row_idx",
        ].astype(int).tolist()
        if len(fallback_group) > 1 and fallback_query_indexes:
            self._search_group(
                faiss,
                nodes,
                embeddings,
                fallback_group["_row_idx"].tolist(),
                fallback_query_indexes,
                pair_rows,
                top_k=profile.faiss_top_k,
                blocking_scope="global",
            )
        if not pair_rows:
            return pd.DataFrame()
        return pd.DataFrame(pair_rows.values()).sort_values(["candidate_rank", "embedding_similarity_score"], ascending=[True, False]).reset_index(drop=True)

    def _search_group(
        self,
        faiss: Any,
        nodes: pd.DataFrame,
        embeddings: np.ndarray,
        indexes: list[int],
        query_indexes: list[int],
        pair_rows: dict[tuple[str, str], dict[str, Any]],
        *,
        top_k: int,
        blocking_scope: str,
    ) -> None:
        if len(indexes) < 2 or not query_indexes:
            return
        group_vectors = np.ascontiguousarray(embeddings[indexes], dtype="float32")
        query_vectors = np.ascontiguousarray(embeddings[query_indexes], dtype="float32")
        index = faiss.IndexFlatIP(group_vectors.shape[1])
        index.add(group_vectors)
        search_k = min(len(indexes), top_k + 1)
        scores, neighbors = index.search(query_vectors, search_k)
        for local_left, neighbor_row in enumerate(neighbors):
            left_idx = query_indexes[local_left]
            for rank, local_right in enumerate(neighbor_row.tolist(), start=0):
                if local_right < 0:
                    continue
                right_idx = indexes[int(local_right)]
                if left_idx == right_idx:
                    continue
                left_node = str(nodes.iloc[left_idx]["node_id"])
                right_node = str(nodes.iloc[right_idx]["node_id"])
                key = _pair_key(left_node, right_node)
                score = float(scores[local_left][rank])
                existing = pair_rows.get(key)
                if existing and float(existing["embedding_similarity_score"]) >= score:
                    continue
                left = nodes.iloc[left_idx]
                right = nodes.iloc[right_idx]
                pair_rows[key] = {
                    "node_id_a": key[0],
                    "node_id_b": key[1],
                    "embedding_similarity_score": round(score, 6),
                    "candidate_rank": max(1, int(rank)),
                    "candidate_source": "faiss_embedding_topk",
                    "blocking_scope": blocking_scope,
                    "sku_a": left["sku"] if str(left["node_id"]) == key[0] else right["sku"],
                    "sku_b": right["sku"] if str(left["node_id"]) == key[0] else left["sku"],
                }

    def _score_candidates(
        self,
        nodes: pd.DataFrame,
        candidates: pd.DataFrame,
        profile: DedupProfile,
        *,
        run_id: str,
        run: dict[str, Any] | None = None,
        project_name: str,
        category_key: str,
    ) -> _ScoreCacheResult:
        node_map = nodes.set_index("node_id")
        pair_entries: list[dict[str, Any]] = []
        text_pairs_by_key: dict[tuple[str, str], tuple[str, str]] = {}
        for row in candidates.itertuples(index=False):
            node_id_a, node_id_b = _pair_key(str(row.node_id_a), str(row.node_id_b))
            text_a = str(node_map.loc[node_id_a, "embedding_text"])
            text_b = str(node_map.loc[node_id_b, "embedding_text"])
            pair_hash = _pair_text_hash(text_a, text_b)
            pair_entries.append(
                {
                    "node_id_a": node_id_a,
                    "node_id_b": node_id_b,
                    "pair_text_hash": pair_hash,
                }
            )
            text_pairs_by_key[(node_id_a, node_id_b)] = (text_a, text_b)
        cached_scores = self.repository.fetch_dedup_pair_scores(
            project_name=project_name,
            category_key=category_key,
            model_method=profile.model_method,
            model_path=profile.model_path.strip(),
            hf_model_id=profile.hf_model_id.strip(),
            activation=profile.activation.strip(),
            entries=pair_entries,
        )
        scores_by_pair: dict[tuple[str, str], float] = dict(cached_scores)
        missing_entries = [
            entry
            for entry in pair_entries
            if (str(entry["node_id_a"]), str(entry["node_id_b"])) not in scores_by_pair
        ]
        hits = len(pair_entries) - len(missing_entries)
        misses = len(missing_entries)
        if misses == 0:
            if run is not None:
                self._update_progress(
                    run_id,
                    run,
                    percent=78,
                    stage="score_pairs",
                    message=f"Cross-encoder cache hit: {hits:,} / {len(pair_entries):,} пар".replace(",", " "),
                    node_count=len(nodes),
                    candidate_count=len(candidates),
                    edge_count=len(pair_entries),
                )
        predict_kwargs: dict[str, Any] = {"batch_size": profile.cross_encoder_batch_size}
        legacy_activation: Any | None = None
        if missing_entries and profile.activation == "sigmoid":
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise DedupRuntimeError("Для sigmoid activation fine-tuned BGE нужен torch.") from exc
            legacy_activation = torch.nn.Sigmoid()
            predict_kwargs["activation_fn"] = legacy_activation

        score_cache_rows: list[dict[str, Any]] = []
        model: Any | None = None
        total_pairs = len(pair_entries)
        chunk_size = max(128, int(profile.cross_encoder_batch_size) * 16)
        for start in range(0, len(missing_entries), chunk_size):
            if model is None:
                model = self._load_cross_encoder(profile)
            end = min(len(missing_entries), start + chunk_size)
            chunk_entries = missing_entries[start:end]
            text_pairs = [
                text_pairs_by_key[(str(entry["node_id_a"]), str(entry["node_id_b"]))]
                for entry in chunk_entries
            ]
            raw_scores = self._predict_cross_encoder_batch(
                model,
                text_pairs,
                predict_kwargs=predict_kwargs,
                legacy_activation=legacy_activation,
            )
            chunk_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
            if len(chunk_scores) != end - start:
                raise DedupRuntimeError("Fine-tuned BGE вернула число scores, не совпадающее с batch candidates.")
            for offset, entry in enumerate(chunk_entries):
                key = (str(entry["node_id_a"]), str(entry["node_id_b"]))
                score = round(float(chunk_scores[offset]), 6)
                scores_by_pair[key] = score
                score_cache_rows.append(
                    {
                        "node_id_a": key[0],
                        "node_id_b": key[1],
                        "pair_text_hash": entry["pair_text_hash"],
                        "score": score,
                    }
                )
            if run is not None:
                scored_count = hits + end
                percent = 60 + round(18 * scored_count / max(1, total_pairs))
                self._update_progress(
                    run_id,
                    run,
                    percent=percent,
                    stage="score_pairs",
                    message=(
                        f"Cross-encoder: {scored_count:,} / {total_pairs:,} пар "
                        f"(cache hit {hits:,}, новых {misses:,})"
                    ).replace(",", " "),
                    node_count=len(nodes),
                    candidate_count=len(candidates),
                    edge_count=scored_count,
                )
        if score_cache_rows:
            self.repository.upsert_dedup_pair_scores(
                project_name=project_name,
                category_key=category_key,
                model_method=profile.model_method,
                model_path=profile.model_path.strip(),
                hf_model_id=profile.hf_model_id.strip(),
                activation=profile.activation.strip(),
                rows=score_cache_rows,
            )

        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(candidates.itertuples(index=False), start=1):
            node_id_a, node_id_b = _pair_key(str(row.node_id_a), str(row.node_id_b))
            left = node_map.loc[node_id_a]
            right = node_map.loc[node_id_b]
            score = float(scores_by_pair[(node_id_a, node_id_b)])
            predicted = score >= profile.threshold_same
            rows.append(
                {
                    "run_id": run_id,
                    "edge_id": _hash_id(run_id, node_id_a, node_id_b, prefix="edge_"),
                    "node_id_a": node_id_a,
                    "node_id_b": node_id_b,
                    "score": round(score, 6),
                    "threshold_strategy": profile.threshold_strategy,
                    "threshold_same": profile.threshold_same,
                    "predicted_binary": bool(predicted),
                    "predicted_label": "exact_duplicate" if predicted else "different_product",
                    "candidate_rank": int(row.candidate_rank),
                    "candidate_source": row.candidate_source,
                    "blocking_scope": row.blocking_scope,
                    "same_pack_signature": bool(_same_pack(left, right)),
                }
            )
        return _ScoreCacheResult(edges=pd.DataFrame(rows), hits=hits, misses=misses)

    @staticmethod
    def _predict_cross_encoder_batch(
        model: Any,
        text_pairs: list[tuple[str, str]],
        *,
        predict_kwargs: dict[str, Any],
        legacy_activation: Any | None,
    ) -> Any:
        try:
            return model.predict(text_pairs, **predict_kwargs)
        except TypeError:
            if legacy_activation is not None:
                legacy_kwargs = {**predict_kwargs, "activation_fct": legacy_activation}
                legacy_kwargs.pop("activation_fn", None)
                return model.predict(text_pairs, **legacy_kwargs)
            return model.predict(text_pairs)

    def _build_groups(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        *,
        graph_config: GraphGroupingConfig,
        previous_assignments: dict[str, dict[str, Any]] | None = None,
        manual_overrides: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        previous_assignments = previous_assignments or {}
        manual_by_node = {
            str(row.get("node_id")): row
            for row in (manual_overrides or [])
            if str(row.get("action") or "") == "split_singleton" and str(row.get("node_id") or "").strip()
        }
        manual_singleton_nodes = set(manual_by_node)
        node_ids = [str(item) for item in nodes["node_id"].tolist()]
        roots = build_family_components(
            node_ids=node_ids,
            edges=edges,
            previous_assignments=previous_assignments,
            manual_singleton_nodes=manual_singleton_nodes,
            config=graph_config,
        )
        root_members: dict[str, list[str]] = {}
        for node_id, root in roots.items():
            root_members.setdefault(root, []).append(node_id)

        existing_family_ids = {
            _clean_text(row.get("ml_family_id"))
            for row in previous_assignments.values()
            if _clean_text(row.get("ml_family_id"))
        }
        existing_family_ids.update(
            _clean_text(row.get("target_family_id"))
            for row in manual_by_node.values()
            if _clean_text(row.get("target_family_id"))
        )
        root_to_family: dict[str, str] = {}
        for root in sorted(root_members):
            manual_ids = [
                _clean_text(manual_by_node[node_id].get("target_family_id"))
                for node_id in root_members[root]
                if node_id in manual_by_node and _clean_text(manual_by_node[node_id].get("target_family_id"))
            ]
            previous_ids = [
                _clean_text(previous_assignments[node_id].get("ml_family_id"))
                for node_id in root_members[root]
                if node_id in previous_assignments and _clean_text(previous_assignments[node_id].get("ml_family_id"))
            ]
            if len(root_members[root]) == 1 and manual_ids:
                root_to_family[root] = manual_ids[0]
            elif previous_ids:
                root_to_family[root] = sorted(set(previous_ids), key=lambda item: (-previous_ids.count(item), item))[0]
            else:
                root_to_family[root] = _next_sequence_id("mlfam", existing_family_ids)

        node_map = nodes.set_index("node_id")
        family_members: dict[str, list[str]] = {}
        for node_id, root in roots.items():
            family_members.setdefault(root_to_family[root], []).append(node_id)

        existing_pack_ids = {
            _clean_text(row.get("ml_pack_id"))
            for row in previous_assignments.values()
            if _clean_text(row.get("ml_pack_id"))
        }
        existing_pack_ids.update(
            _clean_text(row.get("target_pack_id"))
            for row in manual_by_node.values()
            if _clean_text(row.get("target_pack_id"))
        )
        pack_ids: dict[tuple[str, str], str] = {}
        pack_members: dict[tuple[str, str], list[str]] = {}
        pack_signatures = {node_id: _pack_signature(node_map.loc[node_id]) for node_id in node_ids}
        for node_id in node_ids:
            family_id = root_to_family[roots[node_id]]
            pack_key = (family_id, pack_signatures[node_id])
            pack_members.setdefault(pack_key, []).append(node_id)
        for pack_key, members in sorted(pack_members.items()):
            manual_pack_ids = [
                _clean_text(manual_by_node[node_id].get("target_pack_id"))
                for node_id in members
                if node_id in manual_by_node and _clean_text(manual_by_node[node_id].get("target_pack_id"))
            ]
            previous_ids = [
                _clean_text(previous_assignments[node_id].get("ml_pack_id"))
                for node_id in members
                if node_id in previous_assignments and _clean_text(previous_assignments[node_id].get("ml_pack_id"))
            ]
            if len(members) == 1 and manual_pack_ids:
                pack_ids[pack_key] = manual_pack_ids[0]
            elif previous_ids:
                pack_ids[pack_key] = sorted(set(previous_ids), key=lambda item: (-previous_ids.count(item), item))[0]
            else:
                pack_ids[pack_key] = _next_sequence_id("mlpack", existing_pack_ids)
        rows: list[dict[str, Any]] = []
        for node_id in node_ids:
            row = node_map.loc[node_id]
            family_id = root_to_family[roots[node_id]]
            pack_key = (family_id, pack_signatures[node_id])
            members = family_members[family_id]
            pack_group_members = pack_members[pack_key]
            manual_singleton = node_id in manual_singleton_nodes
            canonical_node = max(
                pack_group_members,
                key=lambda item: (
                    float(node_map.loc[item].get("sales_volume") or 0),
                    float(node_map.loc[item].get("revenue") or 0),
                    str(node_map.loc[item].get("sku") or ""),
                ),
            )
            rows.append(
                {
                    "run_id": str(row["run_id"]),
                    "node_id": node_id,
                    "ml_family_id": family_id,
                    "ml_pack_id": pack_ids[pack_key],
                    "canonical_node_id": canonical_node,
                    "canonical_sku": str(node_map.loc[canonical_node].get("sku") or ""),
                    "ml_dedup_status": "manual_singleton" if manual_singleton else "auto_grouped" if len(members) > 1 else "singleton",
                    "confidence_score": None
                    if manual_singleton
                    else self._node_confidence(node_id, edges, previous_assignments=previous_assignments),
                    "component_size": 1 if manual_singleton else len(members),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _node_confidence(
        node_id: str,
        edges: pd.DataFrame,
        *,
        previous_assignments: dict[str, dict[str, Any]] | None = None,
    ) -> float | None:
        scores: list[float] = []
        if not edges.empty:
            related = edges[
                edges["predicted_binary"].astype(bool)
                & (edges["node_id_a"].astype(str).eq(node_id) | edges["node_id_b"].astype(str).eq(node_id))
            ]
            if not related.empty:
                scores.extend(float(item) for item in pd.to_numeric(related["score"], errors="coerce").dropna().tolist())
        previous = (previous_assignments or {}).get(node_id)
        if previous and previous.get("confidence_score") is not None:
            previous_score = _to_float(previous.get("confidence_score"))
            if previous_score is not None:
                scores.append(previous_score)
        if not scores:
            return None
        return round(max(scores), 6)

    def _write_manifest(
        self,
        run_id: str,
        run: dict[str, Any],
        profile: DedupProfile,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        groups: pd.DataFrame,
        *,
        retrieval_cache: dict[str, object] | None = None,
        materialized_rows: int = 0,
    ) -> Path:
        output_dir = self.settings.project_root / "data" / "projects" / str(run["project_name"]) / "dedup" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest = {
            "run_id": run_id,
            "project_name": run["project_name"],
            "category_key": run["category_key"],
            "category_name": run.get("category_name"),
            "source_category_keys": self._source_category_keys_from_run(run),
            "profile": profile.to_dict(),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "group_count": int(len(groups)),
            "materialized_row_count": int(materialized_rows),
            "retrieval_cache": retrieval_cache or {},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _settings_payload(profile: DedupProfile) -> dict[str, Any]:
        payload = profile.to_dict()
        payload.update(
            {
                "retrieval_cache_enabled": True,
                "retrieval_cache_schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            }
        )
        return payload

    @staticmethod
    def _normalize_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
        if run is None:
            return None
        output = dict(run)
        manifest = output.get("manifest_json")
        if isinstance(manifest, str):
            try:
                output["manifest_json"] = json.loads(manifest) if manifest.strip() else {}
            except json.JSONDecodeError:
                output["manifest_json"] = {}
        elif manifest is None:
            output["manifest_json"] = {}
        return output

    def _update_progress(
        self,
        run_id: str,
        run: dict[str, Any],
        *,
        percent: int,
        stage: str,
        message: str,
        node_count: int | None = None,
        candidate_count: int | None = None,
        edge_count: int | None = None,
        group_count: int | None = None,
    ) -> None:
        manifest = self._manifest_from_run(run)
        manifest.update(
            {
                "progress_percent": max(0, min(100, int(percent))),
                "progress_stage": stage,
                "progress_message": message,
                "progress_updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        values: dict[str, Any] = {"manifest_json": manifest}
        if node_count is not None:
            values["node_count"] = int(node_count)
        if candidate_count is not None:
            values["candidate_count"] = int(candidate_count)
        if edge_count is not None:
            values["edge_count"] = int(edge_count)
        if group_count is not None:
            values["group_count"] = int(group_count)
        self.repository.update_dedup_run(run_id, values)

    @staticmethod
    def _manifest_from_run(run: dict[str, Any]) -> dict[str, Any]:
        manifest = run.get("manifest_json")
        if isinstance(manifest, str):
            try:
                parsed = json.loads(manifest) if manifest.strip() else {}
            except json.JSONDecodeError:
                parsed = {}
            return parsed if isinstance(parsed, dict) else {}
        if isinstance(manifest, dict):
            return dict(manifest)
        return {}

    @staticmethod
    def _runtime_profile_from_run(run: dict[str, Any]) -> dict[str, Any]:
        manifest = DedupService._manifest_from_run(run)
        runtime_profile = manifest.get("runtime_profile")
        return dict(runtime_profile) if isinstance(runtime_profile, dict) else {}

    @staticmethod
    def _source_category_keys_from_run(run: dict[str, Any]) -> list[str]:
        manifest = DedupService._manifest_from_run(run)
        raw_keys = manifest.get("source_category_keys")
        if isinstance(raw_keys, list):
            source_keys = [str(key).strip() for key in raw_keys if str(key or "").strip()]
            if source_keys:
                return source_keys
        return [str(run["category_key"])]

    @staticmethod
    def _run_manifest_json(
        *,
        manifest_path: Path,
        profile: DedupProfile,
        retrieval_cache: dict[str, object],
        materialized_rows: int = 0,
        source_category_keys: list[str] | None = None,
        progress_percent: int = 100,
        progress_stage: str = "success",
        progress_message: str = "Готово",
    ) -> dict[str, object]:
        return {
            "manifest_path": str(manifest_path),
            "model_method": profile.model_method,
            "model_device": profile.model_device,
            "threshold_strategy": profile.threshold_strategy,
            "threshold_same": profile.threshold_same,
            "faiss_top_k": profile.faiss_top_k,
            "runtime_profile": {
                "model_device": profile.model_device,
                "graph_grouping_algorithm": profile.graph_grouping_algorithm,
                "graph_community_resolution": profile.graph_community_resolution,
                "graph_community_seed": profile.graph_community_seed,
                "graph_edge_weight_col": profile.graph_edge_weight_col,
                "embedding_batch_size": profile.embedding_batch_size,
                "cross_encoder_batch_size": profile.cross_encoder_batch_size,
            },
            "source_category_keys": list(source_category_keys or []),
            "progress_percent": max(0, min(100, int(progress_percent))),
            "progress_stage": progress_stage,
            "progress_message": progress_message,
            "progress_updated_at": datetime.now().isoformat(timespec="seconds"),
            "materialized_row_count": int(materialized_rows),
            "retrieval_cache_status": retrieval_cache.get("status"),
            "retrieval_cache_key": retrieval_cache.get("cache_key"),
            "retrieval_cache_path": retrieval_cache.get("cache_path"),
            "retrieval_cache_schema_version": retrieval_cache.get("schema_version") or RETRIEVAL_CACHE_SCHEMA_VERSION,
            "embedding_shape": retrieval_cache.get("embedding_shape"),
            "cache_rebuild_reason": retrieval_cache.get("cache_rebuild_reason"),
            "identity_hits": retrieval_cache.get("identity_hits"),
            "identity_misses": retrieval_cache.get("identity_misses"),
            "embedding_cache_hits": retrieval_cache.get("embedding_cache_hits"),
            "embedding_cache_misses": retrieval_cache.get("embedding_cache_misses"),
            "score_cache_hits": retrieval_cache.get("score_cache_hits"),
            "score_cache_misses": retrieval_cache.get("score_cache_misses"),
        }
