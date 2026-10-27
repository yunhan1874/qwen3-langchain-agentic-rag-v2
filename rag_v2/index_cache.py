"""Content-addressed index versions; old live indices are never auto-deleted."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import RAGConfig


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_identity(config: RAGConfig) -> dict[str, Any]:
    model = Path(config.embedding_model)
    assets = {}
    if model.is_dir():
        for path in sorted(model.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}:
                assets[path.relative_to(model).as_posix()] = file_digest(path)
    return {
        "schema_version": 2,
        "source_path": str(config.docs_path.resolve()),
        "source_sha256": file_digest(config.docs_path),
        "embedding_model": config.embedding_model,
        "embedding_revision": config.embedding_revision,
        "embedding_assets": assets,
        "chunking": {
            "breakpoint_percentile": config.semantic_breakpoint_percentile,
            "min_chars": config.min_chunk_chars,
            "max_chars": config.max_chunk_chars,
        },
    }


def version_directory(config: RAGConfig, identity: dict[str, Any] | None = None) -> Path:
    identity = index_identity(config) if identity is None else identity
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return config.index_dir / "versions" / hashlib.sha256(payload).hexdigest()[:24]


def manifest_matches(directory: Path, identity: dict[str, Any]) -> bool:
    try:
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8")) == identity
    except (OSError, ValueError):
        return False


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
