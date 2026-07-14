#!/usr/bin/env python3
"""Evaluate frozen football forecasts without mixing 90-minute and qualification targets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


OUTCOMES = ("home", "draw", "away")
TARGETS = ("result_90min",)
KNOCKOUT_RESULT_TYPES = ("aet", "penalties")


class InputError(ValueError):
    """Raised when a forecast record is invalid."""


def parse_time(value: str, field: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError(f"{field} must be ISO 8601 with a timezone") from exc
    if parsed.tzinfo is None:
        raise InputError(f"{field} must include a timezone")
    return parsed


def probabilities(value: Any, field: str) -> Dict[str, float]:
    if not isinstance(value, dict):
        raise InputError(f"{field} must be an object")
    parsed: Dict[str, float] = {}
    for key in OUTCOMES:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise InputError(f"{field}.{key} must be finite numeric")
        parsed[key] = float(item)
        if not 0.0 <= parsed[key] <= 1.0:
            raise InputError(f"{field}.{key} must be between 0 and 1")
    if abs(sum(parsed.values()) - 1.0) > 0.001:
        raise InputError(f"{field} must sum to 1 within 0.001")
    return parsed


def nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{field} must be a non-negative integer")
    return value


def outcome_from_score(home: int, away: int) -> str:
    if home > away:
        return "home"
    if home < away:
        return "away"
    return "draw"


def parse_actual(raw: Dict[str, Any], line_number: int, warnings: List[str]) -> Tuple[str, Dict[str, Any]]:
    target = raw.get("target", "result_90min")
    if target not in TARGETS:
        raise InputError(f"line {line_number}: target must be result_90min")
    actual = raw.get("actual")
    explicit_outcome = raw.get("outcome")
    if actual is None:
        if explicit_outcome not in OUTCOMES:
            raise InputError(f"line {line_number}: supply actual.home_goals_90/away_goals_90 or outcome")
        warnings.append(f"line {line_number} uses legacy outcome without a verifiable 90-minute score")
        return explicit_outcome, {"outcome_source": "legacy_outcome"}
    if not isinstance(actual, dict):
        raise InputError(f"line {line_number}.actual must be an object")
    home_90 = nonnegative_int(actual.get("home_goals_90"), f"line {line_number}.actual.home_goals_90")
    away_90 = nonnegative_int(actual.get("away_goals_90"), f"line {line_number}.actual.away_goals_90")
    derived = outcome_from_score(home_90, away_90)
    if explicit_outcome is not None and explicit_outcome != derived:
        raise InputError(f"line {line_number}: outcome conflicts with the supplied 90-minute score")
    result_type = str(actual.get("result_type", "regular")).lower()
    if result_type not in ("regular", "aet", "penalties"):
        raise InputError(f"line {line_number}.actual.result_type must be regular, aet, or penalties")
    if result_type in KNOCKOUT_RESULT_TYPES and derived != "draw":
        raise InputError(f"line {line_number}: an AET/penalties match must be level at 90 minutes")
    source = actual.get("source")
    if not isinstance(source, str) or not source.strip():
        raise InputError(f"line {line_number}.actual.source is required for traceability")
    parsed_actual = {
        "outcome_source": "score_90",
        "home_goals_90": home_90,
        "away_goals_90": away_90,
        "result_type": result_type,
        "source": source,
    }
    for field in ("home_goals_final", "away_goals_final"):
        if field in actual:
            parsed_actual[field] = nonnegative_int(actual[field], f"line {line_number}.actual.{field}")
    if ("home_goals_final" in parsed_actual) != ("away_goals_final" in parsed_actual):
        raise InputError(f"line {line_number}: final score fields must be supplied together")
    if "qualified" in actual:
        if actual["qualified"] not in ("home", "away"):
            raise InputError(f"line {line_number}.actual.qualified must be home or away")
        parsed_actual["qualified"] = actual["qualified"]
    return derived, parsed_actual


def scores(probability: Dict[str, float], outcome: str) -> Tuple[float, float, float]:
    observed = {key: 1.0 if key == outcome else 0.0 for key in OUTCOMES}
    brier = sum((probability[key] - observed[key]) ** 2 for key in OUTCOMES)
    log_loss = -math.log(max(probability[outcome], 1e-15))
    forecast_cumulative = (probability["home"], probability["home"] + probability["draw"])
    observed_cumulative = (observed["home"], observed["home"] + observed["draw"])
    rps = 0.5 * sum((forecast - actual) ** 2 for forecast, actual in zip(forecast_cumulative, observed_cumulative))
    return brier, log_loss, rps


def calibration(records: Sequence[Dict[str, Any]], bins: int) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for outcome in OUTCOMES:
        grouped: List[List[Tuple[float, float]]] = [[] for _ in range(bins)]
        for record in records:
            forecast = record["probabilities"][outcome]
            index = min(int(forecast * bins), bins - 1)
            grouped[index].append((forecast, 1.0 if record["outcome"] == outcome else 0.0))
        rows = []
        for index, items in enumerate(grouped):
            if items:
                rows.append({
                    "range": [round(index / bins, 3), round((index + 1) / bins, 3)],
                    "count": len(items),
                    "mean_probability": round(statistics.fmean(item[0] for item in items), 6),
                    "observed_frequency": round(statistics.fmean(item[1] for item in items), 6),
                })
        output[outcome] = rows
    return output


def aggregate(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, float]:
    triples = [scores(record[field], record["outcome"]) for record in records]
    top_pick_hits = [max(record[field], key=record[field].get) == record["outcome"] for record in records]
    return {
        "multiclass_brier": round(statistics.fmean(item[0] for item in triples), 6),
        "log_loss": round(statistics.fmean(item[1] for item in triples), 6),
        "rps_normalized": round(statistics.fmean(item[2] for item in triples), 6),
        "top_pick_accuracy": round(statistics.fmean(top_pick_hits), 6),
        "mean_probability_observed": round(
            statistics.fmean(record[field][record["outcome"]] for record in records), 6
        ),
    }


def load_records(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise InputError(f"line {line_number}: record must be an object")
            outcome, actual = parse_actual(raw, line_number, warnings)
            record: Dict[str, Any] = {
                "match_id": str(raw.get("match_id", f"line-{line_number}")),
                "target": raw.get("target", "result_90min"),
                "probabilities": probabilities(raw.get("probabilities"), f"line {line_number}.probabilities"),
                "outcome": outcome,
                "actual": actual,
                "stage": str(raw.get("stage", "unspecified")),
                "data_level": str(raw.get("data_level", "unspecified")),
                "snapshot_kind": str(raw.get("snapshot_kind", "unspecified")),
                "forecast_role": str(raw.get("forecast_role", "unspecified")),
            }
            if "benchmark" in raw:
                record["benchmark"] = probabilities(raw["benchmark"], f"line {line_number}.benchmark")
            kickoff = raw.get("kickoff")
            frozen_at = raw.get("frozen_at")
            if not isinstance(kickoff, str) or not isinstance(frozen_at, str):
                raise InputError(f"line {line_number}: kickoff and frozen_at are required strings")
            kickoff_time = parse_time(kickoff, f"line {line_number}.kickoff")
            frozen_time = parse_time(frozen_at, f"line {line_number}.frozen_at")
            if frozen_time >= kickoff_time:
                raise InputError(f"line {line_number}: frozen_at must be before kickoff")
            if "benchmark" in record:
                benchmark_meta = raw.get("benchmark_meta")
                if not isinstance(benchmark_meta, dict):
                    warnings.append(f"line {line_number} has benchmark probabilities without timestamp/source metadata")
                else:
                    source = benchmark_meta.get("source")
                    captured_max = benchmark_meta.get("captured_at_max")
                    if not isinstance(source, str) or not source.strip() or not isinstance(captured_max, str):
                        raise InputError(f"line {line_number}.benchmark_meta requires source and captured_at_max")
                    if parse_time(captured_max, f"line {line_number}.benchmark_meta.captured_at_max") > frozen_time:
                        raise InputError(f"line {line_number}: benchmark was captured after frozen_at")
                    record["benchmark_meta"] = benchmark_meta
            record["kickoff"] = kickoff
            record["frozen_at"] = frozen_at
            record["horizon_hours"] = round((kickoff_time - frozen_time).total_seconds() / 3600.0, 3)
            records.append(record)
    if not records:
        raise InputError("no forecast records found")
    duplicate_keys = [key for key, count in Counter(
        (r["match_id"], r["target"], r["snapshot_kind"]) for r in records
    ).items() if count > 1]
    if duplicate_keys:
        raise InputError(
            "duplicate match/target/snapshot_kind records are not allowed; choose one frozen record per snapshot type"
        )
    if len({record["match_id"] for record in records}) < len(records):
        warnings.append("Multiple snapshot types exist for some matches; record_count exceeds independent_match_count.")
    return records, warnings


def load_manifest(path: Path, forecast_ids: set[str]) -> Dict[str, Any]:
    allowed = {"FROZEN_FORECAST", "NO_FORECAST", "INVALID_FORECAST", "PENDING_RESULT"}
    statuses: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"manifest line {line_number}: invalid JSON") from exc
            match_id = str(raw.get("match_id", ""))
            status = raw.get("status")
            if not match_id or status not in allowed:
                raise InputError(f"manifest line {line_number}: match_id and valid status are required")
            if match_id in statuses:
                raise InputError(f"manifest line {line_number}: duplicate match_id {match_id}")
            statuses[match_id] = status
    if not statuses:
        raise InputError("manifest contains no matches")
    missing = forecast_ids - set(statuses)
    if missing:
        raise InputError("manifest is missing one or more forecast match_ids")
    wrong = [match_id for match_id in forecast_ids if statuses[match_id] != "FROZEN_FORECAST"]
    if wrong:
        raise InputError("forecast records must have FROZEN_FORECAST status in manifest")
    counts = Counter(statuses.values())
    completed_scope = counts["FROZEN_FORECAST"] + counts["NO_FORECAST"] + counts["INVALID_FORECAST"]
    return {
        "total_matches": len(statuses),
        "completed_scope": completed_scope,
        "status_counts": dict(counts),
        "valid_forecast_coverage": round(counts["FROZEN_FORECAST"] / completed_scope, 6) if completed_scope else None,
    }


def grouped_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for field in ("stage", "data_level", "snapshot_kind", "forecast_role"):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        output[field] = {
            name: {"count": len(items), "metrics": aggregate(items, "probabilities")}
            for name, items in sorted(groups.items())
        }
    return output


def evaluate(
    records: Sequence[Dict[str, Any]], bins: int, warnings: List[str], coverage: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    forecast_metrics = aggregate(records, "probabilities")
    benchmark_records = [record for record in records if "benchmark" in record]
    benchmark = None
    if benchmark_records:
        paired_forecast = aggregate(benchmark_records, "probabilities")
        benchmark_metrics = aggregate(benchmark_records, "benchmark")
        metric_keys = ("multiclass_brier", "log_loss", "rps_normalized")
        benchmark = {
            "count": len(benchmark_records),
            "paired_forecast_metrics": paired_forecast,
            "metrics": benchmark_metrics,
            "forecast_minus_benchmark": {
                key: round(paired_forecast[key] - benchmark_metrics[key], 6) for key in metric_keys
            },
        }
        if all(record["probabilities"] == record["benchmark"] for record in benchmark_records):
            warnings.append("Forecasts equal the market benchmark; this is an L3 market audit, not independent model validation.")
        if len(benchmark_records) != len(records):
            warnings.append("Benchmark coverage is incomplete; paired comparisons use only records with benchmarks.")
    else:
        warnings.append("No market benchmark was supplied.")
    independent_count = len({record["match_id"] for record in records})
    if independent_count < 30:
        warnings.append("Fewer than 30 independent matches: do not interpret calibration bins or subgroup rates.")
    elif independent_count < 100:
        warnings.append("Fewer than 100 independent matches: metric comparisons remain highly uncertain.")
    return {
        "record_count": len(records),
        "independent_match_count": independent_count,
        "coverage": coverage,
        "target": "result_90min",
        "outcome_counts": dict(Counter(record["outcome"] for record in records)),
        "metric_conventions": {
            "multiclass_brier": "mean sum of squared errors across home/draw/away; not divided by 3",
            "log_loss": "mean negative natural log probability assigned to the observed 90-minute outcome",
            "rps_normalized": "two cumulative squared errors divided by 2",
            "lower_is_better": True,
        },
        "forecast_metrics": forecast_metrics,
        "benchmark": benchmark,
        "strata": grouped_metrics(records),
        "calibration": calibration(records, bins) if independent_count >= 30 else None,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL forecast file")
    parser.add_argument("--bins", type=int, default=10, help="calibration bins (default: 10)")
    parser.add_argument("--manifest", type=Path, help="optional JSONL completeness manifest")
    parser.add_argument("--output", type=Path, help="write JSON output to this file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 2 <= args.bins <= 50:
        print("error: --bins must be between 2 and 50", file=sys.stderr)
        return 2
    try:
        records, warnings = load_records(args.input)
        coverage = load_manifest(args.manifest, {record["match_id"] for record in records}) if args.manifest else None
        if coverage is None:
            warnings.append("No completeness manifest supplied; forecast coverage is unknown.")
        rendered = json.dumps(evaluate(records, args.bins, warnings, coverage), ensure_ascii=False, indent=2)
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
