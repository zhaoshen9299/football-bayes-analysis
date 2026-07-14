#!/usr/bin/env python3
"""Evaluate frozen 90-minute corner forecasts from JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


class InputError(ValueError):
    pass


def parse_time(value: str, field: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError(f"{field} must be ISO 8601 with timezone") from exc
    if parsed.tzinfo is None:
        raise InputError(f"{field} must include timezone")
    return parsed


def number(value: Any, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InputError(f"{field} must be finite numeric")
    result = float(value)
    if result < minimum:
        raise InputError(f"{field} must be >= {minimum}")
    return result


def prediction(value: Any, field: str) -> Dict[str, float]:
    if not isinstance(value, dict):
        raise InputError(f"{field} must be an object")
    parsed = {key: number(value.get(key), f"{field}.{key}") for key in ("mean", "p05", "p95")}
    if not parsed["p05"] <= parsed["mean"] <= parsed["p95"]:
        raise InputError(f"{field} must satisfy p05 <= mean <= p95")
    return parsed


def load(path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"line {line_number}: invalid JSON") from exc
            kickoff = parse_time(str(raw.get("kickoff")), f"line {line_number}.kickoff")
            frozen = parse_time(str(raw.get("frozen_at")), f"line {line_number}.frozen_at")
            if frozen >= kickoff:
                raise InputError(f"line {line_number}: frozen_at must be before kickoff")
            actual = raw.get("actual")
            if not isinstance(actual, dict) or actual.get("period") != "90min_including_stoppage":
                raise InputError(
                    f"line {line_number}: actual.period must be 90min_including_stoppage; extra-time corners are invalid"
                )
            source = actual.get("source")
            if not isinstance(source, str) or not source.strip():
                raise InputError(f"line {line_number}.actual.source is required for traceability")
            home_actual = int(number(actual.get("home"), f"line {line_number}.actual.home"))
            away_actual = int(number(actual.get("away"), f"line {line_number}.actual.away"))
            if home_actual != actual.get("home") or away_actual != actual.get("away"):
                raise InputError(f"line {line_number}: actual corner counts must be integers")
            predicted = raw.get("prediction")
            if not isinstance(predicted, dict):
                raise InputError(f"line {line_number}.prediction must be an object")
            total_lines: Dict[str, float] = {}
            for line_key, item in predicted.get("total_lines", {}).items():
                try:
                    line_value = float(line_key)
                except ValueError as exc:
                    raise InputError(f"line {line_number}: total line keys must be numeric") from exc
                if line_value % 1 != 0.5 or not isinstance(item, dict):
                    raise InputError(f"line {line_number}: only half-corner total lines are supported")
                over = number(item.get("over"), f"line {line_number}.prediction.total_lines.{line_key}.over")
                if over > 1:
                    raise InputError(f"line {line_number}: over probability must be <= 1")
                total_lines[str(line_value)] = over
            records.append({
                "match_id": str(raw.get("match_id", f"line-{line_number}")),
                "home": prediction(predicted.get("home"), f"line {line_number}.prediction.home"),
                "away": prediction(predicted.get("away"), f"line {line_number}.prediction.away"),
                "total": prediction(predicted.get("total"), f"line {line_number}.prediction.total"),
                "total_lines": total_lines,
                "actual": {"home": home_actual, "away": away_actual, "total": home_actual + away_actual},
            })
    if not records:
        raise InputError("no corner forecast records found")
    if len(records) < 30:
        warnings.append("Fewer than 30 independent matches: corner calibration conclusions are premature.")
    return records, warnings


def count_metrics(records: Sequence[Dict[str, Any]], side: str) -> Dict[str, float]:
    errors = [record[side]["mean"] - record["actual"][side] for record in records]
    coverage = [record[side]["p05"] <= record["actual"][side] <= record[side]["p95"] for record in records]
    return {
        "mae": round(statistics.fmean(abs(error) for error in errors), 6),
        "rmse": round(math.sqrt(statistics.fmean(error * error for error in errors)), 6),
        "mean_error": round(statistics.fmean(errors), 6),
        "interval_90_coverage": round(statistics.fmean(coverage), 6),
    }


def evaluate(records: Sequence[Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
    line_scores: Dict[str, List[float]] = defaultdict(list)
    line_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        for line, probability_over in record["total_lines"].items():
            observed_over = 1.0 if record["actual"]["total"] > float(line) else 0.0
            line_scores[line].append((probability_over - observed_over) ** 2)
            line_counts[line] += 1
    return {
        "count": len(records),
        "period": "90min_including_stoppage",
        "count_metrics": {side: count_metrics(records, side) for side in ("home", "away", "total")},
        "total_line_brier": {
            line: {"count": line_counts[line], "brier": round(statistics.fmean(values), 6)}
            for line, values in sorted(line_scores.items(), key=lambda item: float(item[0]))
        },
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        records, warnings = load(args.input)
        rendered = json.dumps(evaluate(records, warnings), ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except (OSError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
