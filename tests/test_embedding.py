"""Embedder behaviour, with the Ollama backend mocked at the HTTP layer."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from rag_eval.embedding import HashingEmbedder, OllamaEmbedder

BASE = "http://localhost:11434"


class TestHashingEmbedder:
    def test_dimensions_match_the_configuration(self) -> None:
        embedder = HashingEmbedder(dimensions=64)
        assert len(embedder.embed(["text"])[0]) == 64

    def test_is_deterministic(self) -> None:
        embedder = HashingEmbedder()
        assert embedder.embed(["same"]) == embedder.embed(["same"])

    def test_vectors_are_unit_length(self) -> None:
        vector = HashingEmbedder().embed(["some text here"])[0]
        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-5)

    def test_similar_surface_forms_are_close(self) -> None:
        """Character n-grams capture surface similarity - which is the point,
        and also the ceiling: it has no way to relate 2FA to two-factor auth."""
        embedder = HashingEmbedder()
        password, passwords, unrelated = embedder.embed(
            ["password reset", "passwords reset", "quantum chromodynamics"]
        )
        close = sum(a * b for a, b in zip(password, passwords, strict=True))
        far = sum(a * b for a, b in zip(password, unrelated, strict=True))
        assert close > far

    def test_empty_string_is_handled(self) -> None:
        assert len(HashingEmbedder(dimensions=32).embed([""])[0]) == 32

    def test_batch_order_is_preserved(self) -> None:
        embedder = HashingEmbedder()
        batch = embedder.embed(["one", "two"])
        assert batch[0] == embedder.embed(["one"])[0]
        assert batch[1] == embedder.embed(["two"])[0]


class TestOllamaEmbedder:
    @respx.mock
    def test_calls_the_embed_endpoint(self) -> None:
        route = respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        )
        embedder = OllamaEmbedder(model="test-model")
        assert embedder.embed(["hello"]) == [[0.1, 0.2]]

        sent = json.loads(route.calls[0].request.read())
        assert sent == {"model": "test-model", "input": ["hello"]}

    @respx.mock
    def test_discovers_dimensions_from_the_first_response(self) -> None:
        respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.0] * 768]})
        )
        embedder = OllamaEmbedder()
        embedder.embed(["hello"])
        assert embedder.dimensions == 768

    @respx.mock
    def test_missing_model_gives_an_actionable_error(self) -> None:
        respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(404))
        with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text"):
            OllamaEmbedder().embed(["hello"])

    @respx.mock
    def test_short_response_is_rejected(self) -> None:
        """Silently accepting fewer vectors than inputs would misalign every
        chunk with its embedding - corrupting results rather than failing."""
        respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1]]})
        )
        with pytest.raises(RuntimeError, match="asked for 2 embeddings, got 1"):
            OllamaEmbedder().embed(["a", "b"])

    @respx.mock
    def test_batches_large_inputs(self) -> None:
        route = respx.post(f"{BASE}/api/embed").mock(
            side_effect=lambda request: httpx.Response(
                200,
                json={"embeddings": [[0.5]] * len(json.loads(request.read())["input"])},
            )
        )
        embedder = OllamaEmbedder(batch_size=2)
        assert len(embedder.embed(["a", "b", "c", "d", "e"])) == 5
        assert route.call_count == 3  # 2 + 2 + 1

    @respx.mock
    def test_memory_cache_avoids_a_second_call(self) -> None:
        route = respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1]]})
        )
        embedder = OllamaEmbedder()
        embedder.embed(["repeat"])
        embedder.embed(["repeat"])
        assert route.call_count == 1

    @respx.mock
    def test_disk_cache_survives_a_new_instance(self, tmp_path: Path) -> None:
        route = respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.7]]})
        )
        OllamaEmbedder(cache_dir=tmp_path).embed(["persisted"])
        assert route.call_count == 1

        # A fresh instance reads the vector off disk. Asserting the call count
        # did not move is the check that matters; re-mocking the route to raise
        # would not work, since respx keeps the first matching route.
        fresh = OllamaEmbedder(cache_dir=tmp_path)
        assert fresh.embed(["persisted"]) == [[0.7]]
        assert route.call_count == 1

    @respx.mock
    def test_changing_model_invalidates_the_cache(self, tmp_path: Path) -> None:
        """The cache key includes the model, so switching models must not serve
        vectors from a different embedding space."""
        route = respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1]]})
        )
        OllamaEmbedder(model="model-a", cache_dir=tmp_path).embed(["text"])

        route.return_value = httpx.Response(200, json={"embeddings": [[0.9]]})
        assert OllamaEmbedder(model="model-b", cache_dir=tmp_path).embed(["text"]) == [[0.9]]
        assert route.call_count == 2

    @respx.mock
    def test_partial_cache_hit_only_fetches_the_rest(self, tmp_path: Path) -> None:
        route = respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1]]})
        )
        embedder = OllamaEmbedder(cache_dir=tmp_path)
        embedder.embed(["cached"])

        route.return_value = httpx.Response(200, json={"embeddings": [[0.2]]})
        assert embedder.embed(["cached", "fresh"]) == [[0.1], [0.2]]
        # Only the uncached text is sent upstream, in the right slot.
        assert json.loads(route.calls[-1].request.read())["input"] == ["fresh"]

    @respx.mock
    def test_health_true_when_daemon_answers(self) -> None:
        respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={}))
        assert OllamaEmbedder().health() is True

    @respx.mock
    def test_health_false_when_daemon_is_down(self) -> None:
        respx.get(f"{BASE}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        assert OllamaEmbedder().health() is False

    @respx.mock
    def test_warm_up_returns_elapsed_seconds(self) -> None:
        respx.post(f"{BASE}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1]]})
        )
        assert OllamaEmbedder().warm_up() >= 0.0
