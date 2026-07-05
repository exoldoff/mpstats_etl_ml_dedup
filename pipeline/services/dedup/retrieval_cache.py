from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd


RETRIEVAL_CACHE_SCHEMA_VERSION = "dedup_retrieval_cache_v1"
RETRIEVAL_TEXT_BUILDER_VERSION = "dedup_retrieval_text_v2"
RETRIEVAL_NORMALIZE_EMBEDDINGS = True


@dataclass(frozen=True)
class RetrievalCacheMetadata:
    status: str
    schema_version: str = RETRIEVAL_CACHE_SCHEMA_VERSION
    cache_key: str | None = None
    cache_path: str | None = None
    embedding_shape: list[int] | None = None
    cache_rebuild_reason: str | None = None

    def to_manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "embedding_shape": self.embedding_shape,
            "cache_rebuild_reason": self.cache_rebuild_reason,
        }


@dataclass(frozen=True)
class RetrievalCacheLoad:
    embeddings: np.ndarray | None
    metadata: RetrievalCacheMetadata


@dataclass(frozen=True)
class _CacheSpec:
    project_name: str
    category_key: str
    embedding_model_name: str
    source_fingerprint: str
    node_ids: list[str]
    node_ids_sha256: str
    node_count: int

    def cache_key(self, embedding_dimension: int) -> str:
        payload = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "source_fingerprint": self.source_fingerprint,
            "embedding_model_name": self.embedding_model_name,
            "text_builder_version": RETRIEVAL_TEXT_BUILDER_VERSION,
            "normalize_embeddings": RETRIEVAL_NORMALIZE_EMBEDDINGS,
            "embedding_dimension": int(embedding_dimension),
        }
        return _sha256_text(_canonical_json(payload))[:32]


class RetrievalEmbeddingCache:
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root

    def skipped(self, reason: str) -> RetrievalCacheMetadata:
        return RetrievalCacheMetadata(
            status="skipped",
            embedding_shape=[0, 0],
            cache_rebuild_reason=reason,
        )

    def load(
        self,
        nodes: pd.DataFrame,
        *,
        project_name: str,
        category_key: str,
        embedding_model_name: str,
    ) -> RetrievalCacheLoad:
        spec = self._spec(
            nodes,
            project_name=project_name,
            category_key=category_key,
            embedding_model_name=embedding_model_name,
        )
        category_dir = self._category_dir(project_name=project_name, category_key=category_key)
        if not category_dir.exists():
            return RetrievalCacheLoad(None, self._miss("miss"))

        corrupt_reason: str | None = None
        for manifest_path in sorted(category_dir.glob("*/manifest.json")):
            try:
                manifest = _read_json(manifest_path)
            except (OSError, json.JSONDecodeError) as exc:
                corrupt_reason = f"corrupt_manifest: {exc}"
                continue
            if not self._manifest_matches_spec(manifest, spec):
                continue
            try:
                embeddings = self._load_matching_cache(manifest_path.parent, manifest, spec)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                corrupt_reason = f"corrupt: {exc}"
                continue
            return RetrievalCacheLoad(
                embeddings,
                RetrievalCacheMetadata(
                    status="hit",
                    cache_key=str(manifest.get("cache_key") or ""),
                    cache_path=str(manifest_path.parent),
                    embedding_shape=[int(item) for item in embeddings.shape],
                ),
            )

        return RetrievalCacheLoad(None, self._miss(corrupt_reason or "miss"))

    def write(
        self,
        nodes: pd.DataFrame,
        embeddings: np.ndarray,
        *,
        project_name: str,
        category_key: str,
        embedding_model_name: str,
        rebuild_reason: str,
    ) -> RetrievalCacheMetadata:
        output = np.asarray(embeddings, dtype="float32")
        if output.ndim != 2 or len(output) != len(nodes):
            raise ValueError("embeddings shape does not match node catalog")
        spec = self._spec(
            nodes,
            project_name=project_name,
            category_key=category_key,
            embedding_model_name=embedding_model_name,
        )
        cache_key = spec.cache_key(int(output.shape[1]))
        category_dir = self._category_dir(project_name=project_name, category_key=category_key)
        final_dir = category_dir / cache_key
        tmp_dir = category_dir / f".{cache_key}.{uuid4().hex}.tmp"
        category_dir.mkdir(parents=True, exist_ok=True)

        try:
            tmp_dir.mkdir(parents=True, exist_ok=False)
            node_ids_path = tmp_dir / "node_ids.json"
            embeddings_path = tmp_dir / "embeddings.npy"
            manifest_path = tmp_dir / "manifest.json"

            node_ids_path.write_text(_canonical_json(spec.node_ids), encoding="utf-8")
            np.save(embeddings_path, np.ascontiguousarray(output, dtype="float32"))
            embeddings_sha256 = _file_sha256(embeddings_path)
            node_ids_sha256 = _file_sha256(node_ids_path)
            manifest = {
                "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "source_fingerprint": spec.source_fingerprint,
                "embedding_model_name": embedding_model_name,
                "text_builder_version": RETRIEVAL_TEXT_BUILDER_VERSION,
                "normalize_embeddings": RETRIEVAL_NORMALIZE_EMBEDDINGS,
                "embedding_dimension": int(output.shape[1]),
                "embedding_shape": [int(item) for item in output.shape],
                "node_count": spec.node_count,
                "node_ids_sha256": node_ids_sha256,
                "embeddings_sha256": embeddings_sha256,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "artifacts": {
                    "manifest": "manifest.json",
                    "node_ids": "node_ids.json",
                    "embeddings": "embeddings.npy",
                },
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return RetrievalCacheMetadata(
            status="rebuilt",
            cache_key=cache_key,
            cache_path=str(final_dir),
            embedding_shape=[int(item) for item in output.shape],
            cache_rebuild_reason=rebuild_reason,
        )

    def _miss(self, reason: str) -> RetrievalCacheMetadata:
        return RetrievalCacheMetadata(status="miss", cache_rebuild_reason=reason)

    def _category_dir(self, *, project_name: str, category_key: str) -> Path:
        return (
            self.project_root
            / "data"
            / "projects"
            / _safe_segment(project_name)
            / "dedup_cache"
            / _safe_segment(category_key)
        )

    def _spec(
        self,
        nodes: pd.DataFrame,
        *,
        project_name: str,
        category_key: str,
        embedding_model_name: str,
    ) -> _CacheSpec:
        node_ids = [_clean_text(item) for item in nodes["node_id"].tolist()]
        source_entries: list[dict[str, object]] = []
        for row in nodes.to_dict(orient="records"):
            hashes = _source_hashes(row.get("source_row_hashes_json"))
            if not hashes:
                hashes = [_sha256_text(_canonical_json(_fallback_node_source(row)))]
            source_entries.append(
                {
                    "node_id": _clean_text(row.get("node_id")),
                    "row_hashes": hashes,
                }
            )
        source_fingerprint = _sha256_text(
            _canonical_json(
                {
                    "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "nodes": source_entries,
                }
            )
        )
        return _CacheSpec(
            project_name=project_name,
            category_key=category_key,
            embedding_model_name=embedding_model_name,
            source_fingerprint=source_fingerprint,
            node_ids=node_ids,
            node_ids_sha256=_sha256_text(_canonical_json(node_ids)),
            node_count=len(node_ids),
        )

    def _manifest_matches_spec(self, manifest: dict[str, object], spec: _CacheSpec) -> bool:
        if not isinstance(manifest, dict):
            return False
        try:
            node_count = int(manifest.get("node_count") or -1)
        except (TypeError, ValueError):
            return False
        return (
            manifest.get("schema_version") == RETRIEVAL_CACHE_SCHEMA_VERSION
            and manifest.get("source_fingerprint") == spec.source_fingerprint
            and manifest.get("embedding_model_name") == spec.embedding_model_name
            and manifest.get("text_builder_version") == RETRIEVAL_TEXT_BUILDER_VERSION
            and bool(manifest.get("normalize_embeddings")) == RETRIEVAL_NORMALIZE_EMBEDDINGS
            and node_count == spec.node_count
        )

    def _load_matching_cache(self, cache_dir: Path, manifest: dict[str, object], spec: _CacheSpec) -> np.ndarray:
        node_ids_path = cache_dir / "node_ids.json"
        embeddings_path = cache_dir / "embeddings.npy"
        if not node_ids_path.is_file():
            raise ValueError("node_ids.json missing")
        if not embeddings_path.is_file():
            raise ValueError("embeddings.npy missing")
        if str(manifest.get("node_ids_sha256") or "") != _file_sha256(node_ids_path):
            raise ValueError("node_ids checksum mismatch")
        node_ids = _read_json(node_ids_path)
        if node_ids != spec.node_ids:
            raise ValueError("node_ids mismatch")
        if str(manifest.get("embeddings_sha256") or "") != _file_sha256(embeddings_path):
            raise ValueError("embeddings checksum mismatch")
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        expected_shape = manifest.get("embedding_shape")
        if not isinstance(expected_shape, list) or len(expected_shape) != 2:
            raise ValueError("embedding_shape missing")
        if tuple(int(item) for item in expected_shape) != tuple(int(item) for item in embeddings.shape):
            raise ValueError("embedding_shape mismatch")
        if embeddings.ndim != 2 or int(embeddings.shape[0]) != spec.node_count:
            raise ValueError("embedding matrix shape mismatch")
        if int(manifest.get("embedding_dimension") or -1) != int(embeddings.shape[1]):
            raise ValueError("embedding dimension mismatch")
        expected_cache_key = spec.cache_key(int(embeddings.shape[1]))
        if str(manifest.get("cache_key") or "") != expected_cache_key or cache_dir.name != expected_cache_key:
            raise ValueError("cache_key mismatch")
        if str(embeddings.dtype) != "float32":
            raise ValueError("embedding dtype mismatch")
        return embeddings


def _source_hashes(value: object) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return sorted({_clean_text(item) for item in parsed if _clean_text(item)})


def _fallback_node_source(row: dict[str, object]) -> dict[str, object]:
    return {
        "node_id": row.get("node_id"),
        "article": row.get("article"),
        "marketplace_code": row.get("marketplace_code"),
        "sku": row.get("sku"),
        "brand": row.get("brand"),
        "subcategory": row.get("subcategory"),
        "retrieval_text": row.get("retrieval_text"),
    }


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _safe_segment(value: str) -> str:
    segment = re.sub(r"[^\w_.-]+", "_", value.strip(), flags=re.UNICODE)
    return segment.strip("._") or "mpstats"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
