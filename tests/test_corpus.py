"""Corpus loading, and the validation that keeps a benchmark honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.corpus import Document, load_corpus, load_queries


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "alpha.md").write_text(
        "# Alpha Topic\ntags: one, two\n\nBody of alpha.\n", encoding="utf-8"
    )
    (directory / "beta.md").write_text("# Beta Topic\n\nBody of beta.\n", encoding="utf-8")
    return directory


class TestLoadCorpus:
    def test_reads_every_markdown_file(self, corpus_dir: Path) -> None:
        assert {d.doc_id for d in load_corpus(corpus_dir)} == {"alpha", "beta"}

    def test_parses_title_and_tags(self, corpus_dir: Path) -> None:
        alpha = next(d for d in load_corpus(corpus_dir) if d.doc_id == "alpha")
        assert alpha.title == "Alpha Topic"
        assert alpha.tags == ("one", "two")
        assert alpha.text == "Body of alpha."

    def test_tags_are_optional(self, corpus_dir: Path) -> None:
        beta = next(d for d in load_corpus(corpus_dir) if d.doc_id == "beta")
        assert beta.tags == ()
        assert beta.text == "Body of beta."

    def test_ordering_is_stable(self, corpus_dir: Path) -> None:
        """Deterministic order keeps two benchmark runs comparable."""
        assert [d.doc_id for d in load_corpus(corpus_dir)] == ["alpha", "beta"]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match=r"no \.md documents"):
            load_corpus(tmp_path / "empty")

    def test_full_text_prepends_the_title(self) -> None:
        document = Document(doc_id="d", title="Title", text="Body")
        assert document.full_text == "Title\n\nBody"


class TestLoadQueries:
    def write(self, tmp_path: Path, records: list[dict[str, object]]) -> Path:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        return path

    def test_parses_queries_and_grades(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            [{"query_id": "q1", "text": "question?", "relevance": {"alpha": 2, "beta": 1}}],
        )
        query = load_queries(path)[0]
        assert query.query_id == "q1"
        assert query.relevance == {"alpha": 2.0, "beta": 1.0}

    def test_relevant_ids_excludes_zero_grades(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path, [{"query_id": "q1", "text": "q", "relevance": {"a": 2, "b": 0}}]
        )
        assert load_queries(path)[0].relevant_ids == {"a"}

    def test_unknown_document_reference_raises(self, tmp_path: Path) -> None:
        """A label pointing at a missing document silently caps recall below
        1.0 for that query, quietly invalidating the whole benchmark. It has to
        fail loudly."""
        path = self.write(tmp_path, [{"query_id": "q1", "text": "q", "relevance": {"ghost": 2}}])
        with pytest.raises(ValueError, match="labels unknown documents"):
            load_queries(path, known_doc_ids={"alpha"})

    def test_error_names_the_offending_line(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            [
                {"query_id": "q1", "text": "q", "relevance": {"alpha": 2}},
                {"query_id": "q2", "text": "q", "relevance": {"ghost": 2}},
            ],
        )
        with pytest.raises(ValueError, match=":2 labels unknown"):
            load_queries(path, known_doc_ids={"alpha"})

    def test_query_without_labels_raises(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, [{"query_id": "q1", "text": "q", "relevance": {}}])
        with pytest.raises(ValueError, match="no relevance labels"):
            load_queries(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        record = json.dumps({"query_id": "q1", "text": "q", "relevance": {"a": 1}})
        path.write_text(f"\n{record}\n\n", encoding="utf-8")
        assert len(load_queries(path)) == 1

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no queries found"):
            load_queries(path)


class TestShippedDataset:
    """The committed corpus is the benchmark's foundation; if it is malformed
    every number downstream is wrong."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def test_corpus_and_golden_set_agree(self, data_dir: Path) -> None:
        documents = load_corpus(data_dir / "corpus")
        queries = load_queries(
            data_dir / "golden.jsonl", known_doc_ids={d.doc_id for d in documents}
        )
        assert len(documents) >= 20
        assert len(queries) >= 30

    def test_every_document_is_reachable_by_some_query(self, data_dir: Path) -> None:
        """An unlabelled document is dead weight: it can only ever hurt
        precision, never help recall."""
        documents = load_corpus(data_dir / "corpus")
        queries = load_queries(data_dir / "golden.jsonl")
        labelled = {doc_id for q in queries for doc_id in q.relevance}
        assert {d.doc_id for d in documents} - labelled == set()
