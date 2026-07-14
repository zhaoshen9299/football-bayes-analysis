#!/usr/bin/env python3
"""Leakage-safe rolling backtest for Football-Data style league CSV files."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


OUTCOMES = ("H", "D", "A")
ROLE_NAMES = ("home_for", "home_against", "away_for", "away_against")


class InputError(ValueError):
    pass


def parse_date(row: Dict[str, str]) -> dt.datetime:
    date = dt.datetime.strptime(row["Date"], "%d/%m/%Y")
    time = row.get("Time") or "12:00"
    try:
        hour, minute = (int(item) for item in time.split(":"))
    except (ValueError, AttributeError) as exc:
        raise InputError(f"invalid Time value: {time}") from exc
    return date.replace(hour=hour, minute=minute)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HC", "AC"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required.issubset(rows[0]):
        raise InputError(f"{path} is empty or missing required columns")
    parsed = []
    for index, row in enumerate(rows, 1):
        try:
            item = dict(row)
            item["kickoff"] = parse_date(row)
            for field in ("FTHG", "FTAG", "HC", "AC"):
                item[field] = int(row[field])
            if item["FTR"] not in OUTCOMES:
                raise ValueError("invalid FTR")
        except (ValueError, KeyError) as exc:
            raise InputError(f"{path} row {index} has invalid result/count data") from exc
        parsed.append(item)
    parsed.sort(key=lambda item: item["kickoff"])
    return parsed


def baseline(rows: Sequence[Dict[str, Any]], home_field: str, away_field: str) -> Tuple[float, float]:
    return (
        statistics.fmean(row[home_field] for row in rows),
        statistics.fmean(row[away_field] for row in rows),
    )


def add_history(history: Dict[str, Dict[str, List[Tuple[dt.datetime, int]]]], row: Dict[str, Any], home_field: str, away_field: str) -> None:
    home, away, when = row["HomeTeam"], row["AwayTeam"], row["kickoff"]
    home_value, away_value = row[home_field], row[away_field]
    history[home]["home_for"].append((when, home_value))
    history[home]["home_against"].append((when, away_value))
    history[away]["away_for"].append((when, away_value))
    history[away]["away_against"].append((when, home_value))


def initial_history(rows: Iterable[Dict[str, Any]], home_field: str, away_field: str):
    history: Dict[str, Dict[str, List[Tuple[dt.datetime, int]]]] = defaultdict(
        lambda: {role: [] for role in ROLE_NAMES}
    )
    for row in rows:
        add_history(history, row, home_field, away_field)
    return history


def posterior(
    observations: Sequence[Tuple[dt.datetime, int]], prior_mean: float, prior_weight: float,
    half_life_days: float, as_of: dt.datetime,
) -> Tuple[float, float]:
    alpha = prior_mean * prior_weight
    beta = prior_weight
    for when, value in observations:
        days = max(0.0, (as_of - when).total_seconds() / 86400.0)
        weight = 0.5 ** (days / half_life_days)
        alpha += weight * value
        beta += weight
    return alpha, beta


def role_posteriors(history, row, base_home, base_away, prior_weight, half_life):
    when, home, away = row["kickoff"], row["HomeTeam"], row["AwayTeam"]
    return {
        "home_for": posterior(history[home]["home_for"], base_home, prior_weight, half_life, when),
        "home_against": posterior(history[home]["home_against"], base_away, prior_weight, half_life, when),
        "away_for": posterior(history[away]["away_for"], base_away, prior_weight, half_life, when),
        "away_against": posterior(history[away]["away_against"], base_home, prior_weight, half_life, when),
    }


def combined_means(posts, base_home, base_away) -> Tuple[float, float]:
    means = {key: alpha / beta for key, (alpha, beta) in posts.items()}
    return (
        means["home_for"] * means["away_against"] / base_home,
        means["away_for"] * means["home_against"] / base_away,
    )


def poisson_probs(rate: float, maximum: int = 12) -> List[float]:
    output = [math.exp(-rate)]
    for count in range(1, maximum + 1):
        output.append(output[-1] * rate / count)
    return output


def match_probabilities(home_rate: float, away_rate: float, maximum: int = 12, rho: float = 0.0) -> Dict[str, float]:
    home, away = poisson_probs(home_rate, maximum), poisson_probs(away_rate, maximum)
    result = {key: 0.0 for key in OUTCOMES}
    over_2_5 = 0.0
    total = 0.0
    for hg, hp in enumerate(home):
        for ag, ap in enumerate(away):
            tau = 1.0
            if hg == 0 and ag == 0:
                tau = 1.0 - home_rate * away_rate * rho
            elif hg == 0 and ag == 1:
                tau = 1.0 + home_rate * rho
            elif hg == 1 and ag == 0:
                tau = 1.0 + away_rate * rho
            elif hg == 1 and ag == 1:
                tau = 1.0 - rho
            if tau <= 0:
                raise InputError("rho produces a non-positive Dixon-Coles correction")
            p = hp * ap * tau
            total += p
            result["H" if hg > ag else ("D" if hg == ag else "A")] += p
            if hg + ag >= 3:
                over_2_5 += p
    for key in result:
        result[key] /= total
    result["over_2_5"] = over_2_5 / total
    return result


def de_vig(row: Dict[str, Any], fields: Sequence[str]) -> Dict[str, float] | None:
    try:
        raw = [1.0 / float(row[field]) for field in fields]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    total = sum(raw)
    return {key: value / total for key, value in zip(OUTCOMES, raw)}


def binary_de_vig(row: Dict[str, Any], over_field: str, under_field: str) -> float | None:
    try:
        over, under = 1.0 / float(row[over_field]), 1.0 / float(row[under_field])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return over / (over + under)


def log_loss(probabilities: Dict[str, float], outcome: str) -> float:
    return -math.log(max(probabilities[outcome], 1e-15))


def brier(probabilities: Dict[str, float], outcome: str) -> float:
    return sum((probabilities[key] - (1.0 if key == outcome else 0.0)) ** 2 for key in OUTCOMES)


def rps(probabilities: Dict[str, float], outcome: str) -> float:
    observed = {key: 1.0 if key == outcome else 0.0 for key in OUTCOMES}
    return 0.5 * (
        (probabilities["H"] - observed["H"]) ** 2
        + (probabilities["H"] + probabilities["D"] - observed["H"] - observed["D"]) ** 2
    )


def deterministic_predictions(
    warm_rows, target_rows, home_field, away_field, base_home, base_away, prior_weight, half_life
) -> List[Tuple[float, float]]:
    history = initial_history(warm_rows, home_field, away_field)
    output = []
    pending_time = None
    pending_rows = []
    for row in target_rows:
        if pending_time is not None and row["kickoff"] != pending_time:
            for completed in pending_rows:
                add_history(history, completed, home_field, away_field)
            pending_rows = []
        posts = role_posteriors(history, row, base_home, base_away, prior_weight, half_life)
        output.append(combined_means(posts, base_home, base_away))
        pending_time = row["kickoff"]
        pending_rows.append(row)
    return output


def tune_goal(warm_rows, validation_rows, base_home, base_away):
    candidates = []
    for prior_weight in (5.0, 10.0, 20.0, 40.0):
        for half_life in (90.0, 180.0, 365.0):
            rates = deterministic_predictions(
                warm_rows, validation_rows, "FTHG", "FTAG", base_home, base_away, prior_weight, half_life
            )
            for rho in (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05):
                losses = [
                    log_loss(match_probabilities(home, away, rho=rho), row["FTR"])
                    for row, (home, away) in zip(validation_rows, rates)
                ]
                candidates.append((statistics.fmean(losses), prior_weight, half_life, rho))
    return min(candidates), sorted(candidates)


def tune_corners(warm_rows, validation_rows, base_home, base_away):
    candidates = []
    for prior_weight in (5.0, 10.0, 20.0, 40.0):
        for half_life in (90.0, 180.0, 365.0):
            rates = deterministic_predictions(
                warm_rows, validation_rows, "HC", "AC", base_home, base_away, prior_weight, half_life
            )
            errors = [abs(home + away - row["HC"] - row["AC"]) for row, (home, away) in zip(validation_rows, rates)]
            candidates.append((statistics.fmean(errors), prior_weight, half_life))
    return min(candidates), sorted(candidates)


def tune_market_weight(model_probs, market_probs, outcomes):
    candidates = []
    for step in range(11):
        weight = step / 10.0
        losses = []
        for model, market, outcome in zip(model_probs, market_probs, outcomes):
            blend = {key: (1 - weight) * model[key] + weight * market[key] for key in OUTCOMES}
            losses.append(log_loss(blend, outcome))
        candidates.append((statistics.fmean(losses), weight))
    return min(candidates), candidates


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def poisson_draw(rate: float, rng: random.Random) -> int:
    limit = math.exp(-rate)
    product, count = 1.0, 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def posterior_predictive(posts, base_home, base_away, draws, rng, goal_mode=False, rho=0.0):
    home_counts, away_counts = [], []
    probability_sums = {key: 0.0 for key in OUTCOMES}
    over_sum = 0.0
    for _ in range(draws):
        sampled = {key: rng.gammavariate(alpha, 1.0 / beta) for key, (alpha, beta) in posts.items()}
        home_rate = sampled["home_for"] * sampled["away_against"] / base_home
        away_rate = sampled["away_for"] * sampled["home_against"] / base_away
        if goal_mode:
            probabilities = match_probabilities(home_rate, away_rate, rho=rho)
            for key in OUTCOMES:
                probability_sums[key] += probabilities[key]
            over_sum += probabilities["over_2_5"]
        else:
            home_counts.append(poisson_draw(home_rate, rng))
            away_counts.append(poisson_draw(away_rate, rng))
    if goal_mode:
        return {**{key: probability_sums[key] / draws for key in OUTCOMES}, "over_2_5": over_sum / draws}
    totals = [home + away for home, away in zip(home_counts, away_counts)]
    return {
        "home_mean": statistics.fmean(home_counts), "away_mean": statistics.fmean(away_counts),
        "total_mean": statistics.fmean(totals),
        "home_p05": percentile(home_counts, 0.05), "home_p95": percentile(home_counts, 0.95),
        "away_p05": percentile(away_counts, 0.05), "away_p95": percentile(away_counts, 0.95),
        "total_p05": percentile(totals, 0.05), "total_p95": percentile(totals, 0.95),
        **{f"over_{line}": statistics.fmean(total > line for total in totals) for line in (8.5, 9.5, 10.5, 11.5)},
    }


def bootstrap_mean_difference(first, second, draws=5000, seed=7):
    rng = random.Random(seed)
    differences = [a - b for a, b in zip(first, second)]
    samples = [statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences) for _ in range(draws)]
    return {
        "mean": round(statistics.fmean(differences), 6),
        "p025": round(percentile(samples, 0.025), 6),
        "p975": round(percentile(samples, 0.975), 6),
    }


def summarize_three_way(records, prefix):
    return {
        "count": len(records),
        "log_loss": round(statistics.fmean(row[f"{prefix}_log_loss"] for row in records), 6),
        "multiclass_brier": round(statistics.fmean(row[f"{prefix}_brier"] for row in records), 6),
        "rps_normalized": round(statistics.fmean(row[f"{prefix}_rps"] for row in records), 6),
        "top_pick_accuracy": round(statistics.fmean(row[f"{prefix}_top_pick"] for row in records), 6),
    }


def calibration(records, prefix, bins=10):
    output = {}
    for outcome in OUTCOMES:
        groups = [[] for _ in range(bins)]
        for row in records:
            probability = row[f"{prefix}_{outcome.lower()}"]
            groups[min(int(probability * bins), bins - 1)].append((probability, row["outcome"] == outcome))
        output[outcome] = [
            {"range": [index / bins, (index + 1) / bins], "count": len(items),
             "mean_probability": round(statistics.fmean(x[0] for x in items), 6),
             "observed_frequency": round(statistics.fmean(x[1] for x in items), 6)}
            for index, items in enumerate(groups) if items
        ]
    return output


def run_backtest(train_rows, validation_rows, test_rows, draws, seed):
    if train_rows[-1]["kickoff"] >= validation_rows[0]["kickoff"] or validation_rows[-1]["kickoff"] >= test_rows[0]["kickoff"]:
        raise InputError("train, validation and test periods must be strictly ordered and non-overlapping")
    train_goal_base = baseline(train_rows, "FTHG", "FTAG")
    validation_goal_base = baseline(validation_rows, "FTHG", "FTAG")
    train_corner_base = baseline(train_rows, "HC", "AC")
    validation_corner_base = baseline(validation_rows, "HC", "AC")

    goal_best, goal_grid = tune_goal(train_rows, validation_rows, *train_goal_base)
    corner_best, corner_grid = tune_corners(train_rows, validation_rows, *train_corner_base)
    _, goal_prior, goal_half, goal_rho = goal_best
    _, corner_prior, corner_half = corner_best

    validation_rates = deterministic_predictions(
        train_rows, validation_rows, "FTHG", "FTAG", *train_goal_base, goal_prior, goal_half
    )
    validation_model = [match_probabilities(*rates, rho=goal_rho) for rates in validation_rates]
    validation_market = [de_vig(row, ("AvgCH", "AvgCD", "AvgCA")) for row in validation_rows]
    if any(item is None for item in validation_market):
        raise InputError("validation rows are missing AvgCH/AvgCD/AvgCA")
    blend_best, blend_grid = tune_market_weight(validation_model, validation_market, [row["FTR"] for row in validation_rows])
    _, market_weight = blend_best

    goal_history = initial_history([*train_rows, *validation_rows], "FTHG", "FTAG")
    corner_history = initial_history([*train_rows, *validation_rows], "HC", "AC")
    rng = random.Random(seed)
    records = []
    pending_time = None
    pending_rows = []
    for index, row in enumerate(test_rows, 1):
        if pending_time is not None and row["kickoff"] != pending_time:
            for completed in pending_rows:
                add_history(goal_history, completed, "FTHG", "FTAG")
                add_history(corner_history, completed, "HC", "AC")
            pending_rows = []
        goal_posts = role_posteriors(goal_history, row, *validation_goal_base, goal_prior, goal_half)
        model = posterior_predictive(goal_posts, *validation_goal_base, draws, rng, goal_mode=True, rho=goal_rho)
        market = de_vig(row, ("AvgCH", "AvgCD", "AvgCA"))
        opening = de_vig(row, ("AvgH", "AvgD", "AvgA"))
        if market is None or opening is None:
            raise InputError(f"test row {index} is missing market odds")
        blend = {key: (1 - market_weight) * model[key] + market_weight * market[key] for key in OUTCOMES}

        corner_posts = role_posteriors(corner_history, row, *validation_corner_base, corner_prior, corner_half)
        corners = posterior_predictive(corner_posts, *validation_corner_base, draws, rng, goal_mode=False)
        actual_total = row["HC"] + row["AC"]
        naive_home, naive_away = validation_corner_base
        outcome = row["FTR"]
        item = {
            "match_no": index, "date": row["Date"], "time": row.get("Time", ""),
            "home_team": row["HomeTeam"], "away_team": row["AwayTeam"],
            "home_goals": row["FTHG"], "away_goals": row["FTAG"], "outcome": outcome,
            "home_corners": row["HC"], "away_corners": row["AC"], "total_corners": actual_total,
        }
        for prefix, probabilities in (("model", model), ("market", market), ("opening", opening), ("blend", blend)):
            for key in OUTCOMES:
                item[f"{prefix}_{key.lower()}"] = probabilities[key]
            item[f"{prefix}_log_loss"] = log_loss(probabilities, outcome)
            item[f"{prefix}_brier"] = brier(probabilities, outcome)
            item[f"{prefix}_rps"] = rps(probabilities, outcome)
            item[f"{prefix}_top_pick"] = int(max(OUTCOMES, key=lambda key: probabilities[key]) == outcome)
        item["model_realized_probability"] = model[outcome]
        item["market_realized_probability"] = market[outcome]
        item["model_minus_market_realized_probability"] = model[outcome] - market[outcome]
        if market[outcome] < 0.20:
            item["result_review_label"] = "market_low_probability_result"
        elif market[outcome] < 0.35:
            item["result_review_label"] = "market_moderate_surprise"
        else:
            item["result_review_label"] = "market_regular_range"
        item["model_over_2_5"] = model["over_2_5"]
        item["market_over_2_5"] = binary_de_vig(row, "AvgC>2.5", "AvgC<2.5")
        item["actual_over_2_5"] = int(row["FTHG"] + row["FTAG"] >= 3)
        item.update(corners)
        item["corner_total_error"] = corners["total_mean"] - actual_total
        item["corner_total_abs_error"] = abs(item["corner_total_error"])
        item["corner_total_squared_error"] = item["corner_total_error"] ** 2
        item["corner_total_covered"] = int(corners["total_p05"] <= actual_total <= corners["total_p95"])
        item["naive_corner_total_mean"] = naive_home + naive_away
        item["naive_corner_total_abs_error"] = abs(naive_home + naive_away - actual_total)
        for line in (8.5, 9.5, 10.5, 11.5):
            actual_over = float(actual_total > line)
            item[f"corner_over_{line}_brier"] = (corners[f"over_{line}"] - actual_over) ** 2
        records.append(item)
        pending_time = row["kickoff"]
        pending_rows.append(row)

    corner_errors = [row["corner_total_error"] for row in records]
    team_corner_errors = defaultdict(list)
    for row in records:
        team_corner_errors[row["home_team"]].append(row["home_mean"] - row["home_corners"])
        team_corner_errors[row["away_team"]].append(row["away_mean"] - row["away_corners"])
    model_ou = [row for row in records if row["market_over_2_5"] is not None]
    metrics = {
        "data": {
            "train_count": len(train_rows), "validation_count": len(validation_rows), "test_count": len(test_rows),
            "test_date_range": [test_rows[0]["Date"], test_rows[-1]["Date"]],
        },
        "locked_hyperparameters": {
            "goals": {"prior_weight": goal_prior, "half_life_days": goal_half, "dixon_coles_rho": goal_rho, "validation_log_loss": round(goal_best[0], 6)},
            "corners": {"prior_weight": corner_prior, "half_life_days": corner_half, "validation_total_mae": round(corner_best[0], 6)},
            "market_blend_weight": market_weight,
        },
        "result_90min": {
            prefix: summarize_three_way(records, prefix) for prefix in ("model", "opening", "market", "blend")
        },
        "paired_log_loss_difference": {
            "model_minus_closing_market": bootstrap_mean_difference(
                [row["model_log_loss"] for row in records], [row["market_log_loss"] for row in records], seed=seed
            ),
            "opening_minus_closing_market": bootstrap_mean_difference(
                [row["opening_log_loss"] for row in records], [row["market_log_loss"] for row in records], seed=seed + 1
            ),
        },
        "over_2_5": {
            "count": len(model_ou),
            "model_brier": round(statistics.fmean((row["model_over_2_5"] - row["actual_over_2_5"]) ** 2 for row in model_ou), 6),
            "closing_market_brier": round(statistics.fmean((row["market_over_2_5"] - row["actual_over_2_5"]) ** 2 for row in model_ou), 6),
        },
        "corners": {
            "home_mae": round(statistics.fmean(abs(row["home_mean"] - row["home_corners"]) for row in records), 6),
            "away_mae": round(statistics.fmean(abs(row["away_mean"] - row["away_corners"]) for row in records), 6),
            "total_mae": round(statistics.fmean(row["corner_total_abs_error"] for row in records), 6),
            "total_rmse": round(math.sqrt(statistics.fmean(row["corner_total_squared_error"] for row in records)), 6),
            "total_mean_error": round(statistics.fmean(corner_errors), 6),
            "total_interval_90_coverage": round(statistics.fmean(row["corner_total_covered"] for row in records), 6),
            "naive_total_mae": round(statistics.fmean(row["naive_corner_total_abs_error"] for row in records), 6),
            "model_minus_naive_total_mae": bootstrap_mean_difference(
                [row["corner_total_abs_error"] for row in records],
                [row["naive_corner_total_abs_error"] for row in records], seed=seed + 2
            ),
            "line_brier": {
                str(line): round(statistics.fmean(row[f"corner_over_{line}_brier"] for row in records), 6)
                for line in (8.5, 9.5, 10.5, 11.5)
            },
            "team_own_corner_mean_error": {
                team: round(statistics.fmean(errors), 6)
                for team, errors in sorted(team_corner_errors.items())
            },
        },
        "calibration": {prefix: calibration(records, prefix) for prefix in ("model", "market", "blend")},
        "validation_grids": {
            "goals": [{"log_loss": round(x[0], 6), "prior_weight": x[1], "half_life_days": x[2], "dixon_coles_rho": x[3]} for x in goal_grid],
            "corners": [{"total_mae": round(x[0], 6), "prior_weight": x[1], "half_life_days": x[2]} for x in corner_grid],
            "blend": [{"log_loss": round(x[0], 6), "market_weight": x[1]} for x in blend_grid],
        },
    }
    return records, metrics


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.draws < 100:
        print("error: --draws must be >= 100", file=sys.stderr)
        return 2
    try:
        train, validation, test = load_csv(args.train), load_csv(args.validation), load_csv(args.test)
        records, metrics = run_backtest(train, validation, test, args.draws, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "match_review.csv", records)
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except (OSError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
