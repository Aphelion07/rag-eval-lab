"""Turn run results into something a human can act on.

Three outputs, in increasing order of usefulness:

1. A ranked table - which configuration scored best.
2. A significance note - whether the gap between the top two is larger than
   the noise, using the bootstrap intervals rather than the point estimates.
3. A failure list - which specific queries the winner still gets wrong. This
   is the only output that tells you what to fix next.
"""

from __future__ import annotations

import json
from pathlib import Path

from .corpus import Query
from .experiment import RunResult

BAR_WIDTH = 24


def bar(value: float, maximum: float = 1.0, width: int = BAR_WIDTH) -> str:
    """A text bar. Renders identically in a terminal, a README and a diff,
    which is more than can be said for an image."""
    filled = round(width * value / maximum) if maximum else 0
    return "#" * filled + "." * (width - filled)


def leaderboard(results: list[RunResult], *, sort_by: str = "ndcg") -> str:
    ranked = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
    headers = ["#", "configuration", "recall@k", "nDCG", "MRR", "hit", "chunks", "p50 ms"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

    for position, result in enumerate(ranked, start=1):
        low, high = result.recall_ci
        lines.append(
            "| "
            + " | ".join(
                [
                    str(position),
                    f"`{result.label}`",
                    f"{result.recall:.3f} ({low:.2f}-{high:.2f})",
                    f"{result.ndcg:.3f}",
                    f"{result.mrr:.3f}",
                    f"{result.hit_rate:.3f}",
                    str(result.chunk_count),
                    f"{result.latency_p50_ms:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def chart(results: list[RunResult], *, metric: str = "ndcg") -> str:
    ranked = sorted(results, key=lambda r: getattr(r, metric), reverse=True)
    width = max(len(r.label) for r in ranked) if ranked else 0
    lines = [f"{metric} by configuration", ""]
    for result in ranked:
        value = getattr(result, metric)
        lines.append(f"  {result.label:<{width}}  {bar(value)}  {value:.3f}")
    return "\n".join(lines)


def significance_note(results: list[RunResult], *, metric: str = "recall") -> str:
    """Compare the top two by overlapping bootstrap intervals.

    Non-overlapping intervals are strong evidence of a real difference;
    overlapping ones are not proof of no difference, only insufficient evidence
    at this sample size. The wording says so rather than implying more.
    """
    if len(results) < 2:
        return ""

    ranked = sorted(results, key=lambda r: getattr(r, metric), reverse=True)
    best, runner_up = ranked[0], ranked[1]
    best_low, best_high = best.recall_ci
    other_low, other_high = runner_up.recall_ci

    gap = getattr(best, metric) - getattr(runner_up, metric)
    if best_low > other_high:
        verdict = (
            f"`{best.label}` beats `{runner_up.label}` by {gap:.3f} {metric}, and the "
            f"95% intervals do not overlap ([{best_low:.3f}, {best_high:.3f}] vs "
            f"[{other_low:.3f}, {other_high:.3f}]). The difference is real at this sample size."
        )
    else:
        verdict = (
            f"`{best.label}` leads `{runner_up.label}` by {gap:.3f} {metric}, but the 95% "
            f"intervals overlap ([{best_low:.3f}, {best_high:.3f}] vs "
            f"[{other_low:.3f}, {other_high:.3f}]). With {len(best.queries)} queries that is "
            f"not enough evidence to call the top two apart - treat them as tied."
        )
    return verdict


def failure_analysis(result: RunResult, queries: list[Query], *, limit: int = 10) -> str:
    failures = result.failures(queries)
    if not failures:
        return f"`{result.label}` retrieved something relevant for all {len(queries)} queries."

    lines = [
        f"`{result.label}` found nothing relevant for {len(failures)} of {len(queries)} queries:",
        "",
    ]
    lines.extend(f"- **{query_id}** - {text}" for query_id, text in failures[:limit])
    if len(failures) > limit:
        lines.append(f"- ... and {len(failures) - limit} more")
    return "\n".join(lines)


def full_report(results: list[RunResult], queries: list[Query], *, k: int) -> str:
    ranked = sorted(results, key=lambda r: r.ndcg, reverse=True)
    sections = [
        f"# Retrieval evaluation (k={k}, {len(queries)} queries)",
        "",
        "## Leaderboard",
        "",
        leaderboard(results),
        "",
        "```",
        chart(results),
        "```",
        "",
        "## Is the winner actually winning?",
        "",
        significance_note(results),
        "",
        "## What the best configuration still misses",
        "",
        failure_analysis(ranked[0], queries) if ranked else "",
        "",
    ]
    return "\n".join(sections)


def write_json(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.row() for r in results], indent=2), encoding="utf-8")


__all__ = [
    "bar",
    "chart",
    "failure_analysis",
    "full_report",
    "leaderboard",
    "significance_note",
    "write_json",
]
