"""CLI wiring and argument handling.

All of these run with ``--embedder hashing``, so the suite needs no daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval import cli
from rag_eval.chunking import ParagraphChunker
from rag_eval.embedding import HashingEmbedder


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


class TestArgumentParsing:
    def test_missing_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            cli.main([])

    def test_options_work_after_the_subcommand(self, data_dir: Path) -> None:
        """The reason the parent-parser exists: `run --embedder hashing` has to
        parse, not only `--embedder hashing run`."""
        assert cli.main(["run", "--data", str(data_dir), "--embedder", "hashing", "-k", "3"]) == 0

    def test_options_still_work_before_the_subcommand(self, data_dir: Path) -> None:
        assert cli.main(["--data", str(data_dir), "--embedder", "hashing", "run", "-k", "3"]) == 0

    def test_rejects_an_unknown_embedder(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["run", "--embedder", "magic"])


class TestBuilders:
    def test_builds_all_four_chunkers(self) -> None:
        chunkers = cli.build_chunkers(512, 64)
        assert len(chunkers) == 4
        assert len({c.name for c in chunkers}) == 4

    def test_grid_covers_every_chunker_and_retriever(self) -> None:
        configs = cli.build_configs([ParagraphChunker()], HashingEmbedder())
        assert [c.label for c in configs] == [
            "paragraph+bm25",
            "paragraph+dense",
            "paragraph+hybrid",
        ]

    def test_each_config_builds_a_distinct_retriever(self) -> None:
        configs = cli.build_configs([ParagraphChunker()], HashingEmbedder())
        first, second = configs[0].build_retriever(), configs[0].build_retriever()
        assert first is not second

    def test_hashing_embedder_needs_no_daemon(self) -> None:
        embedder = cli.build_embedder("hashing", model="unused", cache_dir=Path("."))
        assert isinstance(embedder, HashingEmbedder)


class TestRun:
    def test_writes_both_output_files(self, tmp_path: Path, data_dir: Path) -> None:
        report = tmp_path / "report.md"
        results = tmp_path / "results.json"

        code = cli.main(
            [
                "run",
                "--data",
                str(data_dir),
                "--embedder",
                "hashing",
                "-k",
                "5",
                "--out",
                str(report),
                "--json",
                str(results),
            ]
        )

        assert code == 0
        assert "Leaderboard" in report.read_text(encoding="utf-8")
        rows = json.loads(results.read_text(encoding="utf-8"))
        assert len(rows) == 12  # 4 chunkers x 3 retrievers
        assert {"config", "recall", "ndcg", "recall_ci"} <= set(rows[0])

    def test_reports_are_reproducible(self, tmp_path: Path, data_dir: Path) -> None:
        """Two runs of a deterministic pipeline must agree, or no comparison
        between configurations means anything."""
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        for path in (first, second):
            cli.main(["run", "--data", str(data_dir), "--embedder", "hashing", "--json", str(path)])

        left = json.loads(first.read_text(encoding="utf-8"))
        right = json.loads(second.read_text(encoding="utf-8"))
        assert [r["ndcg"] for r in left] == [r["ndcg"] for r in right]


class TestInspect:
    def test_shows_retrieval_for_a_known_query(
        self, capsys: pytest.CaptureFixture, data_dir: Path
    ) -> None:
        code = cli.main(
            ["inspect", "--data", str(data_dir), "--embedder", "hashing", "--query", "q01"]
        )
        out = capsys.readouterr().out

        assert code == 0
        assert "query q01" in out
        assert "labelled relevant" in out
        assert "bm25" in out and "dense" in out and "hybrid" in out

    def test_unknown_query_id_returns_one(
        self, capsys: pytest.CaptureFixture, data_dir: Path
    ) -> None:
        code = cli.main(
            ["inspect", "--data", str(data_dir), "--embedder", "hashing", "--query", "q999"]
        )
        assert code == 1
        assert "no query with id" in capsys.readouterr().err
