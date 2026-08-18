"""Command line entry point.

rag-eval run                    # full grid, real embeddings if available
rag-eval run --embedder hashing # no model needed, runs anywhere
rag-eval inspect --query q01    # what a single query actually retrieved
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from .chunking import (
    Chunker,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
)
from .corpus import load_corpus, load_queries
from .embedding import Embedder, HashingEmbedder, OllamaEmbedder
from .experiment import Config, RunResult, run_grid
from .report import full_report, write_json
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

DEFAULT_DATA = Path("data")


def build_chunkers(size: int, overlap: int) -> list[Chunker]:
    return [
        FixedSizeChunker(size=size, overlap=overlap),
        SentenceChunker(size=size, overlap=overlap),
        RecursiveChunker(size=size, overlap=overlap),
        ParagraphChunker(),
    ]


def build_embedder(kind: str, *, model: str, cache_dir: Path) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    embedder = OllamaEmbedder(model=model, cache_dir=cache_dir)
    if not embedder.health():
        raise SystemExit(
            "Ollama is not reachable at http://localhost:11434.\n"
            "Start it and pull the embedding model:\n"
            "  ollama serve\n"
            f"  ollama pull {model}\n"
            "Or run without a model:  rag-eval run --embedder hashing"
        )
    return embedder


def build_hybrid(embedder: Embedder) -> Retriever:
    """Named rather than a lambda: partial binds the embedder immediately, and
    a named function is inferable where a default-argument lambda is not."""
    return HybridRetriever([BM25Retriever(), DenseRetriever(embedder)])


def build_configs(chunkers: list[Chunker], embedder: Embedder) -> list[Config]:
    """The grid: every chunker crossed with every retrieval strategy."""
    configs: list[Config] = []
    for chunker in chunkers:
        short = chunker.name.split("(")[0]
        configs.append(Config(chunker, BM25Retriever, f"{short}+bm25"))
        configs.append(Config(chunker, partial(DenseRetriever, embedder), f"{short}+dense"))
        configs.append(Config(chunker, partial(build_hybrid, embedder), f"{short}+hybrid"))
    return configs


def _run(args: argparse.Namespace) -> int:
    data = Path(args.data)
    documents = load_corpus(data / "corpus")
    queries = load_queries(data / "golden.jsonl", known_doc_ids={d.doc_id for d in documents})

    embedder = build_embedder(args.embedder, model=args.model, cache_dir=data / ".embedding-cache")
    if isinstance(embedder, OllamaEmbedder):
        elapsed = embedder.warm_up()
        # Pre-embed every query before timing starts. Without this, the first
        # dense configuration pays the embedding cost for all 40 queries and
        # every later one reads them from cache - which showed up as that one
        # configuration being ~500x slower to query than an identical one.
        # The measurement was of cache state, not of retrieval.
        embedder.embed([q.text for q in queries])
        print(f"warmed up {embedder.name} in {elapsed:.2f}s, pre-embedded {len(queries)} queries")

    configs = build_configs(build_chunkers(args.chunk_size, args.overlap), embedder)
    print(
        f"corpus: {len(documents)} documents, {len(queries)} queries, "
        f"{len(configs)} configurations, k={args.k}\n"
    )

    def progress(label: str, result: RunResult) -> None:
        print(
            f"  {label:<24} recall={result.recall:.3f}  nDCG={result.ndcg:.3f}  "
            f"chunks={result.chunk_count}"
        )

    results = run_grid(configs, documents, queries, k=args.k, on_progress=progress)

    report = full_report(results, queries, k=args.k)
    print("\n" + report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        write_json(results, Path(args.json))
        print(f"wrote {args.json}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    data = Path(args.data)
    documents = load_corpus(data / "corpus")
    queries = load_queries(data / "golden.jsonl", known_doc_ids={d.doc_id for d in documents})

    matches = [q for q in queries if q.query_id == args.query]
    if not matches:
        print(f"no query with id {args.query!r}", file=sys.stderr)
        return 1
    query = matches[0]

    embedder = build_embedder(args.embedder, model=args.model, cache_dir=data / ".embedding-cache")
    chunker = SentenceChunker(size=args.chunk_size, overlap=args.overlap)
    chunks = chunker.chunk_all(documents)

    retrievers = [
        BM25Retriever(),
        DenseRetriever(embedder),
        HybridRetriever([BM25Retriever(), DenseRetriever(embedder)]),
    ]

    print(f"query {query.query_id}: {query.text}")
    print(f"labelled relevant: {sorted(query.relevant_ids)}\n")

    for retriever in retrievers:
        retriever.index(chunks)
        retrieved = retriever.search_documents(query.text, args.k)
        marks = ["OK " if doc in query.relevant_ids else "   " for doc in retrieved]
        print(f"{retriever.name}")
        for mark, doc in zip(marks, retrieved, strict=True):
            print(f"  {mark} {doc}")
        print()
    return 0


def add_shared_options(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    """Attach the options every subcommand accepts.

    Called twice: once on the top-level parser with real defaults, and once per
    subparser with ``default=SUPPRESS``.

    The SUPPRESS half is not cosmetic. argparse runs the top-level parser first
    and the subparser second, into the *same* namespace - so a subparser default
    silently overwrites a value the user supplied before the subcommand.
    `rag-eval --embedder hashing run` would quietly fall back to the ollama
    default and try to reach a daemon that may not exist. SUPPRESS leaves the
    attribute unset when the option is absent, so the earlier value survives.
    """

    def value_or_suppress(value: object) -> object:
        return value if defaults else argparse.SUPPRESS

    parser.add_argument(
        "--data", default=value_or_suppress(str(DEFAULT_DATA)), help="data directory"
    )
    parser.add_argument(
        "--embedder", default=value_or_suppress("ollama"), choices=["ollama", "hashing"]
    )
    parser.add_argument("--model", default=value_or_suppress("nomic-embed-text"))
    parser.add_argument("--chunk-size", type=int, default=value_or_suppress(512))
    parser.add_argument("--overlap", type=int, default=value_or_suppress(64))
    parser.add_argument(
        "-k", type=int, default=value_or_suppress(5), help="documents retrieved per query"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    add_shared_options(parser, defaults=True)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate the full configuration grid")
    add_shared_options(run, defaults=False)
    run.add_argument("--out", help="write the markdown report here")
    run.add_argument("--json", help="write raw results here")
    run.set_defaults(func=_run)

    inspect = sub.add_parser("inspect", help="show what one query retrieved")
    add_shared_options(inspect, defaults=False)
    inspect.add_argument("--query", required=True, help="query id, e.g. q01")
    inspect.set_defaults(func=_inspect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
