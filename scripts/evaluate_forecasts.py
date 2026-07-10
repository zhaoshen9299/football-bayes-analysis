#!/usr/bin/env python3
"""Evaluate frozen three-way football forecasts from JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


OUTCOMES = ("home", "draw", "away")


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
            if not items:
                continue
            rows.append(
                {
                    "range": [round(index / bins, 3), round((index + 1) / bins, 3)],
                    "count": len(items),
                    "mean_probability": round(statistics.fmean(item[0] for item in items), 6),
                    "observed_frequency": round(statistics.fmean(item[1] for item in items), 6),
                }
            )
        output[outcome] = rows
    return output


def aggregate(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, float]:
    brier_values = []
    log_values = []
    rps_values = []
    for record in records:
        brier, log_loss, rps = scores(record[field], record["outcome"])
        brier_values.append(brier)
        log_values.append(log_loss)
        rps_values.append(rps)
    return {
        "multiclass_brier": round(statistics.fmean(brier_values), 6),
        "log_loss": round(statistics.fmean(log_values), 6),
        "rps_normalized": round(statistics.fmean(rps_values), 6),
    }


def load_records(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    records = []
    warnings = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise InputError(f"line {line_number}: record must be an object")
            outcome = raw.get("outcome")
            if outcome not in OUTCOMES:
                raise InputError(f"line {line_number}: outcome must be home, draw, or away")
            record = {
                "match_id": str(raw.get("match_id", f"line-{line_number}")),
                "probabilities": probabilities(raw.get("probabilities"), f"line {line_number}.probabilities"),
                "outcome": outcome,
            }
            if "benchmark" in raw:
                record["benchmark"] = probabilities(raw["benchmark"], f"line {line_number}.benchmark")
            kickoff = raw.get("kickoff")
            frozen_at = raw.get("frozen_at")
            if kickoff is not None or frozen_at is not None:
                if not isinstance(kickoff, str) or not isinstance(frozen_at, str):
                    raise InputError(f"line {line_number}: kickoff and frozen_at must both be present strings")
                kickoff_time = parse_time(kickoff, f"line {line_number}.kickoff")
                frozen_time = parse_time(frozen_at, f"line {line_number}.frozen_at")
                if frozen_time >= kickoff_time:
                    raise InputError(f"line {line_number}: frozen_at must be before kickoff")
                record["kickoff"] = kickoff
                record["frozen_at"] = frozen_at
            else:
                warnings.append(f"line {line_number} has no kickoff/frozen_at leakage check")
            records.append(record)
    if not records:
        raise InputError("no forecast records found")
    return records, warnings


def evaluate(records: Sequence[Dict[str, Any]], bins: int, warnings: List[str]) -> Dict[str, Any]:
    model_metrics = aggregate(records, "probabilities")
    benchmark_records = [record for record in records if "benchmark" in record]
    benchmark = None
    if benchmark_records:
        paired_model_metrics = aggregate(benchmark_records, "probabilities")
        benchmark_metrics = aggregate(benchmark_records, "benchmark")
        benchmark = {
            "count": len(benchmark_records),
            "paired_model_metrics": paired_model_metrics,
            "metrics": benchmark_metrics,
            "model_minus_benchmark": {
                key: round(paired_model_metrics[key] - benchmark_metrics[key], 6) for key in paired_model_metrics
            },
        }
        if len(benchmark_records) != len(records):
            warnings.append("Benchmark coverage is incomplete; model and benchmark metrics use different record counts.")
    else:
        warnings.append("No market benchmark was supplied.")
    if len(records) < 100:
        warnings.append("Fewer than 100 forecasts: calibration and metric comparisons are highly uncertain.")
    return {
        "count": len(records),
        "metric_conventions": {
            "multiclass_brier": "mean sum of squared errors across home/draw/away; not divided by 3",
            "log_loss": "mean negative natural log probability assigned to the observed outcome",
            "rps_normalized": "two cumulative squared errors divided by 2",
            "lower_is_better": True,
        },
        "model_metrics": model_metrics,
        "benchmark": benchmark,
        "calibration": calibration(records, bins),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL forecast file")
    parser.add_argument("--bins", type=int, default=10, help="calibration bins (default: 10)")
    parser.add_argument("--output", type=Path, help="write JSON output to this file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 2 <= args.bins <= 50:
        print("error: --bins must be between 2 and 50", file=sys.stderr)
        return 2
    try:
        records, warnings = load_records(args.input)
        result = evaluate(records, args.bins, warnings)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
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
