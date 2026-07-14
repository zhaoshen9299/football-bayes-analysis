#!/usr/bin/env python3
"""Evaluate frozen two-way qualification forecasts from JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


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


def probs(value: Any, field: str) -> Dict[str, float]:
    if not isinstance(value, dict):
        raise InputError(f"{field} must be an object")
    output = {}
    for side in ("home", "away"):
        item = value.get(side)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise InputError(f"{field}.{side} must be finite numeric")
        output[side] = float(item)
    if any(not 0 <= item <= 1 for item in output.values()) or abs(sum(output.values()) - 1) > 0.001:
        raise InputError(f"{field} probabilities must be in [0,1] and sum to 1")
    return output


def load(path: Path) -> List[Dict[str, Any]]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"line {line_number}: invalid JSON") from exc
            if raw.get("target") != "to_qualify":
                raise InputError(f"line {line_number}: target must be to_qualify")
            kickoff = parse_time(str(raw.get("kickoff")), f"line {line_number}.kickoff")
            frozen = parse_time(str(raw.get("frozen_at")), f"line {line_number}.frozen_at")
            if frozen >= kickoff:
                raise InputError(f"line {line_number}: frozen_at must be before kickoff")
            actual = raw.get("actual")
            if not isinstance(actual, dict) or actual.get("qualified") not in ("home", "away"):
                raise InputError(f"line {line_number}.actual.qualified must be home or away")
            if not isinstance(actual.get("source"), str) or not actual["source"].strip():
                raise InputError(f"line {line_number}.actual.source is required")
            key = (str(raw.get("match_id", "")), str(raw.get("snapshot_kind", "unspecified")))
            if not key[0] or key in seen:
                raise InputError(f"line {line_number}: unique match_id/snapshot_kind is required")
            seen.add(key)
            records.append({"probabilities": probs(raw.get("probabilities"), f"line {line_number}.probabilities"), "actual": actual["qualified"]})
    if not records:
        raise InputError("no qualification records found")
    return records


def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    realized = [record["probabilities"][record["actual"]] for record in records]
    accuracy = [max(record["probabilities"], key=record["probabilities"].get) == record["actual"] for record in records]
    warnings = []
    if len(records) < 30:
        warnings.append("Fewer than 30 independent matches: qualification calibration is premature.")
    return {
        "count": len(records),
        "target": "to_qualify",
        "log_loss": round(statistics.fmean(-math.log(max(p, 1e-15)) for p in realized), 6),
        "binary_brier": round(statistics.fmean((1 - p) ** 2 for p in realized), 6),
        "top_pick_accuracy": round(statistics.fmean(accuracy), 6),
        "mean_probability_observed": round(statistics.fmean(realized), 6),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        rendered = json.dumps(evaluate(load(args.input)), ensure_ascii=False, indent=2)
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
