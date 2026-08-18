"""Corpus and evaluation set.

A ``Document`` is the unit a golden label points at; a ``Chunk`` is the unit a
retriever actually returns. Keeping them distinct is what makes it possible to
compare chunking strategies at all: a chunk always knows which document it came
from, so a hit is scored against the document regardless of how the text was
split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()

    @property
    def full_text(self) -> str:
        """Title prepended to the body.

        The title carries a disproportionate share of the topical signal, so
        including it measurably helps both lexical and dense retrieval. It is
        done here rather than in each chunker so every strategy sees the same
        input.
        """
        return f"{self.title}\n\n{self.text}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int


@dataclass(frozen=True)
class Query:
    """One evaluation question and its labelled answers.

    ``relevance`` holds graded labels: 2 for a document that answers the
    question, 1 for one that is related but insufficient. nDCG uses the grades;
    the other metrics treat anything above 0 as relevant.
    """

    query_id: str
    text: str
    relevance: dict[str, float] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> set[str]:
        return {doc_id for doc_id, grade in self.relevance.items() if grade > 0}


def load_corpus(directory: Path) -> list[Document]:
    """Read every ``.md`` file in ``directory`` as one document.

    Front matter is a single optional ``tags:`` line; anything richer would mean
    a YAML dependency for no benefit at this size.
    """
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()

        title = lines[0].lstrip("# ").strip() if lines else path.stem
        tags: tuple[str, ...] = ()
        body_start = 1
        if len(lines) > 1 and lines[1].startswith("tags:"):
            tags = tuple(t.strip() for t in lines[1][5:].split(",") if t.strip())
            body_start = 2

        documents.append(
            Document(
                doc_id=path.stem,
                title=title,
                text="\n".join(lines[body_start:]).strip(),
                tags=tags,
            )
        )
    if not documents:
        raise FileNotFoundError(f"no .md documents found in {directory}")
    return documents


def load_queries(path: Path, *, known_doc_ids: set[str] | None = None) -> list[Query]:
    """Read the golden set from JSONL.

    Validates label references when ``known_doc_ids`` is supplied. A label
    pointing at a document that does not exist silently caps recall below 1.0
    for that query, which is the kind of bug that quietly invalidates a whole
    benchmark - so it fails loudly instead.
    """
    queries: list[Query] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        relevance = {str(k): float(v) for k, v in record["relevance"].items()}

        if known_doc_ids is not None:
            unknown = set(relevance) - known_doc_ids
            if unknown:
                raise ValueError(
                    f"{path}:{line_number} labels unknown documents: {sorted(unknown)}"
                )
        if not relevance:
            raise ValueError(f"{path}:{line_number} has no relevance labels")

        queries.append(Query(query_id=record["query_id"], text=record["text"], relevance=relevance))
    if not queries:
        raise ValueError(f"no queries found in {path}")
    return queries


__all__ = ["Chunk", "Document", "Query", "load_corpus", "load_queries"]
