"""Ollama embeddings, with an on-disk cache.

The cache is not an optimisation detail, it is what makes the benchmark
practical. A grid over 4 chunkers x 3 retrievers re-embeds the same corpus
repeatedly; without caching, a single sweep spends most of its wall clock
recomputing vectors it already has. Keyed on model plus content hash, so
changing the model or the text invalidates the entry automatically.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from .base import Embedder


class OllamaEmbedder(Embedder):
    """Embeddings from a local Ollama daemon. No API key, no network egress."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        cache_dir: Path | None = None,
        batch_size: int = 64,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._cache_dir = cache_dir
        self._memory: dict[str, list[float]] = {}
        self.dimensions = 0  # discovered on first call
        self.api_calls = 0

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache --------------------------------------------------------------

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self._cache_dir / f"{key}.json" if self._cache_dir else None

    def _load(self, key: str) -> list[float] | None:
        if key in self._memory:
            return self._memory[key]
        path = self._cache_path(key)
        if path is not None and path.exists():
            vector: list[float] = json.loads(path.read_text(encoding="utf-8"))
            self._memory[key] = vector
            return vector
        return None

    def _store(self, key: str, vector: list[float]) -> None:
        self._memory[key] = vector
        path = self._cache_path(key)
        if path is not None:
            path.write_text(json.dumps(vector), encoding="utf-8")

    # -- embedding ----------------------------------------------------------

    def _call(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"model {self.model!r} is not available - run: ollama pull {self.model}"
            )
        response.raise_for_status()
        self.api_calls += 1

        embeddings: list[list[float]] = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(embeddings)}")
        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [self._load(self._key(t)) for t in texts]
        missing = [i for i, vector in enumerate(results) if vector is None]

        for start in range(0, len(missing), self.batch_size):
            indices = missing[start : start + self.batch_size]
            for index, vector in zip(indices, self._call([texts[i] for i in indices]), strict=True):
                self._store(self._key(texts[index]), vector)
                results[index] = vector

        final = [vector for vector in results if vector is not None]
        if len(final) != len(texts):
            raise RuntimeError("embedding produced fewer vectors than inputs")
        if final and not self.dimensions:
            self.dimensions = len(final[0])
        return final

    def health(self) -> bool:
        try:
            return self._client.get(f"{self.base_url}/api/tags", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def warm_up(self) -> float:
        """Embed once and return the elapsed seconds.

        Ollama loads a model on first use. Without this, that one-off cost lands
        on whichever configuration the benchmark happens to run first and shows
        up as that configuration being slow.
        """
        started = time.perf_counter()
        self.embed(["warm up"])
        return time.perf_counter() - started

    def close(self) -> None:
        self._client.close()
