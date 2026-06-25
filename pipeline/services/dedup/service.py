from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Callable, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from mpstats_app.config import AppSettings
from mpstats_app.repositories.duckdb_repository import DuckDbAppRepository


DEDUP_SETTINGS_KEY = "dedup_settings_json"
DEDUP_MODEL_PATH_ENV = "DEDUP_FINE_TUNED_MODEL_PATH"
DEDUP_TOP_K = 30
DEDUP_THRESHOLD_STRATEGY = "threshold_weighted_cost"
DEDUP_THRESHOLD_SAME = 0.872321
DEDUP_METHOD = "ft_bge_reranker_v2_m3"
DEDUP_HF_MODEL_ID = "exoldoff/bge-reranker-v2-m3-cross-encoder-marketplaces-rus"
DEDUP_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEDUP_ACTIVATION = "sigmoid"
DEDUP_PROFILE_PATH = Path(__file__).with_name("model_profile.json")
E5_TEXT_PREFIX = "query: "
ELIGIBLE_CATEGORY_NAMES = {"соус", "соусы", "кокосовое масло", "мыло"}

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
    activation: str = DEDUP_ACTIVATION
    threshold_strategy: str = DEDUP_THRESHOLD_STRATEGY
    threshold_same: float = DEDUP_THRESHOLD_SAME
    faiss_top_k: int = DEDUP_TOP_K
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
            activation=str(tracked.get("activation") or DEDUP_ACTIVATION).strip(),
            threshold_strategy=str(tracked.get("threshold_strategy") or DEDUP_THRESHOLD_STRATEGY).strip(),
            threshold_same=float(tracked.get("threshold_same") or DEDUP_THRESHOLD_SAME),
            faiss_top_k=int(tracked.get("faiss_top_k") or DEDUP_TOP_K),
            embedding_batch_size=max(1, int(payload.get("embedding_batch_size") or 64)),
            cross_encoder_batch_size=max(1, int(payload.get("cross_encoder_batch_size") or 32)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_method": self.model_method,
            "model_path": self.model_path,
            "hf_model_id": self.hf_model_id,
            "embedding_model_name": self.embedding_model_name,
            "activation": self.activation,
            "threshold_strategy": self.threshold_strategy,
            "threshold_same": self.threshold_same,
            "faiss_top_k": self.faiss_top_k,
            "embedding_batch_size": self.embedding_batch_size,
            "cross_encoder_batch_size": self.cross_encoder_batch_size,
        }


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


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _hash_id(*parts: object, prefix: str = "") -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


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


class _UnionFind:
    def __init__(self, nodes: Sequence[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent.setdefault(node, node)
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


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
        self._lock = RLock()
        self._threads: dict[str, Thread] = {}

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
        return DedupProfile.from_settings(payload).to_dict()

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        merged = {**current, **payload}
        profile = DedupProfile.from_settings(merged)
        data = profile.to_dict()
        self.repository.set_setting(DEDUP_SETTINGS_KEY, json.dumps(data, ensure_ascii=False))
        return data

    def eligible_categories(self, *, project_name: str) -> dict[str, Any]:
        rows = self.repository.list_dedup_eligible_categories(project_name=project_name)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            latest = self.repository.latest_successful_dedup_run(
                project_name=project_name,
                category_key=str(row["category_key"]),
            )
            enriched.append({**row, "latest_successful_run": latest})
        return {"project_name": project_name, "categories": enriched, "settings": self.get_settings()}

    def start_runs(self, *, project_name: str, category_keys: list[str], wait: bool = False) -> dict[str, Any]:
        eligible = {
            str(row["category_key"]): row
            for row in self.repository.list_dedup_eligible_categories(project_name=project_name)
        }
        clean_keys = [str(key) for key in category_keys if str(key).strip()]
        if not clean_keys:
            raise ValueError("Выбери хотя бы одну категорию для ML-дедупа.")
        unknown = [key for key in clean_keys if key not in eligible]
        if unknown:
            raise ValueError("ML-дедуп v1 доступен только для Соус/Соусы, Кокосовое масло и Мыло: " + ", ".join(unknown))

        profile = DedupProfile.from_settings(self.get_settings())
        runs: list[dict[str, Any]] = []
        for category_key in clean_keys:
            category = eligible[category_key]
            run_id = uuid4().hex
            run = self.repository.create_dedup_run(
                {
                    "run_id": run_id,
                    "project_name": project_name,
                    "category_key": category_key,
                    "category_name": category.get("category_name"),
                    "status": "queued",
                    **profile.to_dict(),
                    "manifest": {"requested_at": datetime.now().isoformat(timespec="seconds")},
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
        return {"runs": [self.repository.get_dedup_run(str(run["run_id"])) or run for run in runs]}

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.repository.get_dedup_run(run_id)
        if not run:
            raise ValueError("Dedup run не найден.")
        return run

    def list_runs(self, *, project_name: str | None = None, category_key: str | None = None) -> dict[str, Any]:
        return {"runs": self.repository.list_dedup_runs(project_name=project_name, category_key=category_key)}

    def export_artifact(self, *, run_id: str, artifact: str) -> dict[str, Any]:
        self.get_run(run_id)
        return {"run_id": run_id, "artifact": artifact, "rows": self.repository.fetch_dedup_artifact(run_id=run_id, artifact=artifact)}

    def _execute_run(self, run_id: str) -> None:
        run = self.repository.get_dedup_run(run_id)
        if not run:
            return
        started = datetime.now()
        self.repository.update_dedup_run(run_id, {"status": "running", "started_at": started, "error": None})
        try:
            profile = DedupProfile(
                model_path=str(run.get("model_path") or ""),
                hf_model_id=str(run.get("hf_model_id") or DEDUP_HF_MODEL_ID),
                embedding_model_name=str(run.get("embedding_model_name") or DEDUP_EMBEDDING_MODEL),
                embedding_batch_size=64,
                cross_encoder_batch_size=max(1, int(run.get("cross_encoder_batch_size") or 32)),
            )
            source = self.repository.fetch_dedup_source_dataframe(
                table_name=self.settings.products_table,
                project_name=str(run["project_name"]),
                category_key=str(run["category_key"]),
            )
            if source.empty:
                raise DedupRuntimeError("Нет строк куба для выбранной категории.")

            nodes = self._build_nodes(source, run_id=run_id)
            if len(nodes) < 2:
                self._load_cross_encoder(profile)
                groups = self._build_groups(nodes, edges=pd.DataFrame())
                self.repository.replace_dedup_nodes(run_id, nodes.to_dict(orient="records"))
                self.repository.replace_dedup_edges(run_id, [])
                self.repository.replace_dedup_groups(run_id, groups.to_dict(orient="records"))
                manifest_path = self._write_manifest(run_id, run, profile, nodes, pd.DataFrame(), groups)
                self.repository.update_dedup_run(
                    run_id,
                    {
                        "status": "success",
                        "node_count": len(nodes),
                        "candidate_count": 0,
                        "edge_count": 0,
                        "group_count": len(groups),
                        "manifest_path": str(manifest_path),
                        "finished_at": datetime.now(),
                    },
                )
                return

            embeddings = self._encode_nodes(nodes, profile)
            candidates = self._generate_candidates(nodes, embeddings, profile)
            if candidates.empty:
                raise DedupRuntimeError("FAISS не вернул ни одной пары-кандидата.")
            edges = self._score_candidates(nodes, candidates, profile, run_id=run_id)
            groups = self._build_groups(nodes, edges)

            self.repository.replace_dedup_nodes(run_id, nodes.to_dict(orient="records"))
            self.repository.replace_dedup_edges(run_id, edges.to_dict(orient="records"))
            self.repository.replace_dedup_groups(run_id, groups.to_dict(orient="records"))
            manifest_path = self._write_manifest(run_id, run, profile, nodes, edges, groups)
            self.repository.update_dedup_run(
                run_id,
                {
                    "status": "success",
                    "node_count": len(nodes),
                    "candidate_count": len(candidates),
                    "edge_count": len(edges),
                    "group_count": len(groups),
                    "manifest_path": str(manifest_path),
                    "manifest_json": {
                        "manifest_path": str(manifest_path),
                        "model_method": profile.model_method,
                        "threshold_strategy": profile.threshold_strategy,
                        "threshold_same": profile.threshold_same,
                        "faiss_top_k": profile.faiss_top_k,
                    },
                    "finished_at": datetime.now(),
                },
            )
        except Exception as exc:
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
        return CrossEncoder(model_name)

    def _encode_nodes(self, nodes: pd.DataFrame, profile: DedupProfile) -> np.ndarray:
        model = self._load_embedding_model(profile)
        texts = [E5_TEXT_PREFIX + _clean_text(text) for text in nodes["embedding_text"].tolist()]
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

    def _load_faiss(self) -> Any:
        if self._faiss_module is not None:
            return self._faiss_module
        try:
            import faiss
        except ModuleNotFoundError as exc:
            raise DedupRuntimeError("Для FAISS retrieval нужен faiss-cpu. Установи зависимости из requirements.txt.") from exc
        return faiss

    def _generate_candidates(self, nodes: pd.DataFrame, embeddings: np.ndarray, profile: DedupProfile) -> pd.DataFrame:
        faiss = self._load_faiss()
        pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
        nodes_with_idx = nodes.reset_index(drop=True).copy()
        nodes_with_idx["_row_idx"] = nodes_with_idx.index
        nodes_with_idx["_subcategory_norm"] = nodes_with_idx["subcategory"].map(_norm_text)

        for _, category_nodes in nodes_with_idx.groupby("category_key", dropna=False):
            known_groups = [
                group
                for _, group in category_nodes[category_nodes["_subcategory_norm"].ne("")].groupby("_subcategory_norm", dropna=False)
                if len(group) > 1
            ]
            known_indexes = {int(idx) for group in known_groups for idx in group["_row_idx"].tolist()}
            for group in known_groups:
                self._search_group(
                    faiss,
                    nodes,
                    embeddings,
                    group["_row_idx"].tolist(),
                    pair_rows,
                    top_k=profile.faiss_top_k,
                    blocking_scope="same_subcategory",
                )
            fallback_group = category_nodes[~category_nodes["_row_idx"].isin(known_indexes)]
            if len(fallback_group) > 1:
                self._search_group(
                    faiss,
                    nodes,
                    embeddings,
                    fallback_group["_row_idx"].tolist(),
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
        pair_rows: dict[tuple[str, str], dict[str, Any]],
        *,
        top_k: int,
        blocking_scope: str,
    ) -> None:
        if len(indexes) < 2:
            return
        group_vectors = np.ascontiguousarray(embeddings[indexes], dtype="float32")
        index = faiss.IndexFlatIP(group_vectors.shape[1])
        index.add(group_vectors)
        search_k = min(len(indexes), top_k + 1)
        scores, neighbors = index.search(group_vectors, search_k)
        for local_left, neighbor_row in enumerate(neighbors):
            left_idx = indexes[local_left]
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
    ) -> pd.DataFrame:
        model = self._load_cross_encoder(profile)
        node_map = nodes.set_index("node_id")
        text_pairs = [
            (
                str(node_map.loc[row.node_id_a, "embedding_text"]),
                str(node_map.loc[row.node_id_b, "embedding_text"]),
            )
            for row in candidates.itertuples(index=False)
        ]
        predict_kwargs: dict[str, Any] = {"batch_size": profile.cross_encoder_batch_size}
        legacy_activation: Any | None = None
        if profile.activation == "sigmoid":
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise DedupRuntimeError("Для sigmoid activation fine-tuned BGE нужен torch.") from exc
            legacy_activation = torch.nn.Sigmoid()
            predict_kwargs["activation_fn"] = legacy_activation
        try:
            raw_scores = model.predict(text_pairs, **predict_kwargs)
        except TypeError:
            if legacy_activation is not None:
                legacy_kwargs = {**predict_kwargs, "activation_fct": legacy_activation}
                legacy_kwargs.pop("activation_fn", None)
                raw_scores = model.predict(text_pairs, **legacy_kwargs)
            else:
                raw_scores = model.predict(text_pairs)
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if len(scores) != len(candidates):
            raise DedupRuntimeError("Fine-tuned BGE вернула число scores, не совпадающее с числом candidates.")

        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(candidates.itertuples(index=False), start=1):
            left = node_map.loc[row.node_id_a]
            right = node_map.loc[row.node_id_b]
            score = float(scores[idx - 1])
            predicted = score >= profile.threshold_same
            rows.append(
                {
                    "run_id": run_id,
                    "edge_id": _hash_id(run_id, row.node_id_a, row.node_id_b, prefix="edge_"),
                    "node_id_a": row.node_id_a,
                    "node_id_b": row.node_id_b,
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
        return pd.DataFrame(rows)

    def _build_groups(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
        node_ids = [str(item) for item in nodes["node_id"].tolist()]
        uf = _UnionFind(node_ids)
        if not edges.empty:
            for row in edges[edges["predicted_binary"].astype(bool)].itertuples(index=False):
                uf.union(str(row.node_id_a), str(row.node_id_b))
        roots = {node_id: uf.find(node_id) for node_id in node_ids}
        root_to_family: dict[str, str] = {}
        for root in sorted(set(roots.values())):
            root_to_family[root] = f"mlfam_{len(root_to_family) + 1:06d}"

        node_map = nodes.set_index("node_id")
        family_members: dict[str, list[str]] = {}
        for node_id, root in roots.items():
            family_members.setdefault(root_to_family[root], []).append(node_id)

        pack_ids: dict[tuple[str, str], str] = {}
        rows: list[dict[str, Any]] = []
        for node_id in node_ids:
            row = node_map.loc[node_id]
            family_id = root_to_family[roots[node_id]]
            pack_key = (family_id, _pack_signature(row))
            if pack_key not in pack_ids:
                pack_ids[pack_key] = f"mlpack_{len(pack_ids) + 1:06d}"
            members = family_members[family_id]
            canonical_node = max(
                members,
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
                    "ml_dedup_status": "auto_grouped" if len(members) > 1 else "singleton",
                    "confidence_score": self._node_confidence(node_id, edges),
                    "component_size": len(members),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _node_confidence(node_id: str, edges: pd.DataFrame) -> float | None:
        if edges.empty:
            return None
        related = edges[
            edges["predicted_binary"].astype(bool)
            & (edges["node_id_a"].astype(str).eq(node_id) | edges["node_id_b"].astype(str).eq(node_id))
        ]
        if related.empty:
            return None
        return round(float(pd.to_numeric(related["score"], errors="coerce").dropna().max()), 6)

    def _write_manifest(
        self,
        run_id: str,
        run: dict[str, Any],
        profile: DedupProfile,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        groups: pd.DataFrame,
    ) -> Path:
        output_dir = self.settings.project_root / "data" / "projects" / str(run["project_name"]) / "dedup" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest = {
            "run_id": run_id,
            "project_name": run["project_name"],
            "category_key": run["category_key"],
            "category_name": run.get("category_name"),
            "profile": profile.to_dict(),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "group_count": int(len(groups)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path
