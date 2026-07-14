#!/usr/bin/env python3
"""Reproducible pre-match football probability calculator.

Supports an empirical-Bayes Gamma-Poisson approximation, externally estimated
goal rates, or a market-only baseline. Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


OUTCOMES = ("home", "draw", "away")


class InputError(ValueError):
    """Raised when the input contract is violated."""


def parse_time(value: str, field: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError(f"{field} must be ISO 8601 with a timezone") from exc
    if parsed.tzinfo is None:
        raise InputError(f"{field} must include a timezone")
    return parsed


def require_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise InputError(f"{name} must be >= {minimum}")
    return result


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise InputError("cannot summarize an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    fraction = pos - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rounded(value: float) -> float:
    return round(value, 6)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": rounded(statistics.fmean(values)),
        "p05": rounded(percentile(values, 0.05)),
        "p50": rounded(percentile(values, 0.50)),
        "p95": rounded(percentile(values, 0.95)),
    }


def poisson_probabilities(rate: float, max_goals: int) -> List[float]:
    probabilities = [math.exp(-rate)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * rate / goals)
    return probabilities


def dixon_coles_tau(home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(home_rate: float, away_rate: float, max_goals: int, rho: float) -> Tuple[List[List[float]], float]:
    home_probs = poisson_probabilities(home_rate, max_goals)
    away_probs = poisson_probabilities(away_rate, max_goals)
    tail = 1.0 - sum(home_probs) * sum(away_probs)
    matrix: List[List[float]] = []
    total = 0.0
    for home_goals, home_probability in enumerate(home_probs):
        row: List[float] = []
        for away_goals, away_probability in enumerate(away_probs):
            tau = dixon_coles_tau(home_goals, away_goals, home_rate, away_rate, rho)
            if tau <= 0.0:
                raise InputError(
                    "dixon_coles_rho produces a non-positive low-score correction; "
                    "use a validated rho closer to zero"
                )
            probability = home_probability * away_probability * tau
            row.append(probability)
            total += probability
        matrix.append(row)
    if total <= 0.0:
        raise InputError("score matrix has zero mass")
    return [[value / total for value in row] for row in matrix], max(0.0, tail)


def matrix_markets(matrix: Sequence[Sequence[float]]) -> Tuple[Dict[str, float], float, float]:
    result = {key: 0.0 for key in OUTCOMES}
    over_2_5 = 0.0
    btts_yes = 0.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if home_goals > away_goals:
                result["home"] += probability
            elif home_goals == away_goals:
                result["draw"] += probability
            else:
                result["away"] += probability
            if home_goals + away_goals >= 3:
                over_2_5 += probability
            if home_goals > 0 and away_goals > 0:
                btts_yes += probability
    return result, over_2_5, btts_yes


def gamma_posterior(
    prior_mean: float,
    prior_weight: float,
    observations: Iterable[Dict[str, Any]],
    half_life_days: float,
    series_name: str,
) -> Dict[str, float]:
    alpha = prior_mean * prior_weight
    beta = prior_weight
    effective_exposure = 0.0
    raw_count = 0
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise InputError(f"series.{series_name}[{index}] must be an object")
        value = require_number(observation.get("value"), f"series.{series_name}[{index}].value", minimum=0.0)
        days_ago = require_number(observation.get("days_ago"), f"series.{series_name}[{index}].days_ago", minimum=0.0)
        reliability = require_number(
            observation.get("reliability", 1.0),
            f"series.{series_name}[{index}].reliability",
            minimum=0.0,
        )
        if reliability > 1.0:
            raise InputError(f"series.{series_name}[{index}].reliability must be <= 1")
        weight = reliability * (0.5 ** (days_ago / half_life_days))
        alpha += weight * value
        beta += weight
        effective_exposure += weight
        raw_count += 1
    return {
        "alpha": alpha,
        "beta": beta,
        "effective_exposure": effective_exposure,
        "raw_count": float(raw_count),
    }


def parse_adjustments(payload: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    adjustments = payload.get("adjustments", {})
    if not isinstance(adjustments, dict):
        raise InputError("adjustments must be an object")
    items = adjustments.get(f"{side}_log_rate", [])
    if not isinstance(items, list):
        raise InputError(f"adjustments.{side}_log_rate must be a list")
    parsed = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise InputError(f"adjustments.{side}_log_rate[{index}] must be an object")
        label = item.get("label")
        source = item.get("source")
        if not isinstance(label, str) or not label.strip():
            raise InputError(f"adjustments.{side}_log_rate[{index}].label is required")
        if not isinstance(source, str) or not source.strip():
            raise InputError(f"adjustments.{side}_log_rate[{index}].source is required")
        mean = require_number(item.get("mean"), f"adjustments.{side}_log_rate[{index}].mean")
        sd = require_number(item.get("sd", 0.0), f"adjustments.{side}_log_rate[{index}].sd", minimum=0.0)
        parsed.append({"label": label, "source": source, "mean": mean, "sd": sd})
    return parsed


def adjustment_draw(items: Sequence[Dict[str, Any]], rng: random.Random) -> float:
    return sum(rng.gauss(item["mean"], item["sd"]) if item["sd"] > 0 else item["mean"] for item in items)


def lognormal_draw(mean: float, sd: float, rng: random.Random) -> float:
    if mean <= 0.0:
        raise InputError("goal-rate means must be > 0")
    if sd == 0.0:
        return mean
    sigma_sq = math.log1p((sd * sd) / (mean * mean))
    mu = math.log(mean) - 0.5 * sigma_sq
    return rng.lognormvariate(mu, math.sqrt(sigma_sq))


def market_baseline(payload: Dict[str, Any], as_of: str) -> Dict[str, Any] | None:
    market = payload.get("market")
    if market is None:
        return None
    if not isinstance(market, dict) or not isinstance(market.get("books"), list) or not market["books"]:
        raise InputError("market.books must be a non-empty list")
    book_results = []
    for index, book in enumerate(market["books"]):
        if not isinstance(book, dict):
            raise InputError(f"market.books[{index}] must be an object")
        name = book.get("name")
        captured_at = book.get("captured_at")
        if not isinstance(name, str) or not name.strip():
            raise InputError(f"market.books[{index}].name is required")
        if not isinstance(captured_at, str) or not captured_at.strip():
            raise InputError(f"market.books[{index}].captured_at is required")
        if parse_time(captured_at, f"market.books[{index}].captured_at") > parse_time(as_of, "match.as_of"):
            raise InputError(f"market.books[{index}].captured_at must not be after match.as_of")
        odds = {key: require_number(book.get(key), f"market.books[{index}].{key}") for key in OUTCOMES}
        if any(value <= 1.0 for value in odds.values()):
            raise InputError(f"market.books[{index}] decimal odds must all be > 1")
        raw = {key: 1.0 / odds[key] for key in OUTCOMES}
        total = sum(raw.values())
        fair = {key: raw[key] / total for key in OUTCOMES}
        book_results.append(
            {
                "name": name,
                "captured_at": captured_at,
                "odds": {key: rounded(odds[key]) for key in OUTCOMES},
                "overround": rounded(total - 1.0),
                "fair_probabilities": {key: rounded(fair[key]) for key in OUTCOMES},
                "_fair": fair,
            }
        )
    medians = {key: statistics.median(item["_fair"][key] for item in book_results) for key in OUTCOMES}
    median_total = sum(medians.values())
    consensus = {key: medians[key] / median_total for key in OUTCOMES}
    cleaned_books = []
    for item in book_results:
        cleaned = dict(item)
        cleaned.pop("_fair")
        cleaned_books.append(cleaned)
    return {
        "method": "per-book proportional de-vig, outcome median, renormalized",
        "books_count": len(cleaned_books),
        "median_overround": rounded(statistics.median(item["overround"] for item in cleaned_books)),
        "consensus_probabilities": {key: rounded(consensus[key]) for key in OUTCOMES},
        "fair_decimal_odds": {key: rounded(1.0 / consensus[key]) for key in OUTCOMES},
        "books": cleaned_books,
    }


def validate_match(payload: Dict[str, Any]) -> Dict[str, Any]:
    match = payload.get("match")
    if not isinstance(match, dict):
        raise InputError("match must be an object")
    required = ("id", "home_team", "away_team", "kickoff_utc", "as_of")
    for key in required:
        if not isinstance(match.get(key), str) or not match[key].strip():
            raise InputError(f"match.{key} is required")
    kickoff = parse_time(match["kickoff_utc"], "match.kickoff_utc")
    as_of = parse_time(match["as_of"], "match.as_of")
    if as_of >= kickoff:
        raise InputError("match.as_of must be before match.kickoff_utc")
    validated = {key: match[key] for key in required}
    if "stage" in match:
        if not isinstance(match["stage"], str) or not match["stage"].strip():
            raise InputError("match.stage must be a non-empty string when supplied")
        validated["stage"] = match["stage"]
    return validated


def build_rate_sampler(payload: Dict[str, Any], rng: random.Random, warnings: List[str]):
    mode = payload.get("mode")
    if mode == "direct_rates":
        goal_rates = payload.get("goal_rates")
        if not isinstance(goal_rates, dict):
            raise InputError("goal_rates must be an object in direct_rates mode")
        parsed: Dict[str, Dict[str, float]] = {}
        for side in ("home", "away"):
            if not isinstance(goal_rates.get(side), dict):
                raise InputError(f"goal_rates.{side} must be an object")
            mean = require_number(goal_rates[side].get("mean"), f"goal_rates.{side}.mean", minimum=0.0)
            sd = require_number(goal_rates[side].get("sd", 0.0), f"goal_rates.{side}.sd", minimum=0.0)
            if mean == 0.0:
                raise InputError(f"goal_rates.{side}.mean must be > 0")
            parsed[side] = {"mean": mean, "sd": sd}

        def sample_direct() -> Tuple[float, float]:
            return (
                lognormal_draw(parsed["home"]["mean"], parsed["home"]["sd"], rng),
                lognormal_draw(parsed["away"]["mean"], parsed["away"]["sd"], rng),
            )

        return sample_direct, {"goal_rates": parsed}

    if mode != "empirical_bayes":
        raise InputError("mode must be empirical_bayes, direct_rates, or market_only")

    data_type = payload.get("data_type")
    if data_type not in ("goals", "xg"):
        raise InputError("data_type must be goals or xg")
    if data_type == "xg":
        warnings.append("xG values are treated as fractional pseudo-counts; this is a quasi-likelihood approximation.")
    baseline = payload.get("league_baseline")
    if not isinstance(baseline, dict):
        raise InputError("league_baseline must be an object")
    home_base = require_number(baseline.get("home_rate"), "league_baseline.home_rate", minimum=0.0)
    away_base = require_number(baseline.get("away_rate"), "league_baseline.away_rate", minimum=0.0)
    prior_weight = require_number(baseline.get("prior_weight"), "league_baseline.prior_weight", minimum=0.0)
    if home_base == 0.0 or away_base == 0.0 or prior_weight == 0.0:
        raise InputError("league baseline rates and prior_weight must be > 0")
    half_life = require_number(payload.get("half_life_days"), "half_life_days", minimum=0.0)
    if half_life == 0.0:
        raise InputError("half_life_days must be > 0")
    series = payload.get("series")
    if not isinstance(series, dict):
        raise InputError("series must be an object")
    prior_means = {
        "home_attack": home_base,
        "home_defense": away_base,
        "away_attack": away_base,
        "away_defense": home_base,
    }
    posteriors: Dict[str, Dict[str, float]] = {}
    for name, prior_mean in prior_means.items():
        observations = series.get(name)
        if not isinstance(observations, list):
            raise InputError(f"series.{name} must be a list")
        if not observations:
            raise InputError(f"series.{name} must contain at least one observation")
        posteriors[name] = gamma_posterior(prior_mean, prior_weight, observations, half_life, name)
    home_adjustments = parse_adjustments(payload, "home")
    away_adjustments = parse_adjustments(payload, "away")

    def sample_empirical() -> Tuple[float, float]:
        sampled = {
            name: rng.gammavariate(values["alpha"], 1.0 / values["beta"])
            for name, values in posteriors.items()
        }
        home_rate = sampled["home_attack"] * sampled["away_defense"] / home_base
        away_rate = sampled["away_attack"] * sampled["home_defense"] / away_base
        home_rate *= math.exp(adjustment_draw(home_adjustments, rng))
        away_rate *= math.exp(adjustment_draw(away_adjustments, rng))
        return home_rate, away_rate

    metadata = {
        "data_type": data_type,
        "league_baseline": {"home_rate": home_base, "away_rate": away_base, "prior_weight": prior_weight},
        "half_life_days": half_life,
        "role_posteriors": {
            name: {
                "alpha": rounded(values["alpha"]),
                "beta": rounded(values["beta"]),
                "posterior_mean": rounded(values["alpha"] / values["beta"]),
                "effective_exposure": rounded(values["effective_exposure"]),
                "raw_count": int(values["raw_count"]),
            }
            for name, values in posteriors.items()
        },
        "adjustments": {"home_log_rate": home_adjustments, "away_log_rate": away_adjustments},
    }
    return sample_empirical, metadata


def run_model(payload: Dict[str, Any], warnings: List[str]) -> Dict[str, Any] | None:
    mode = payload.get("mode")
    if mode == "market_only":
        warnings.append("Market-only mode: no independent model or scoreline probabilities were produced.")
        return None
    draws_value = payload.get("draws", 5000)
    if isinstance(draws_value, bool) or not isinstance(draws_value, int) or draws_value < 100:
        raise InputError("draws must be an integer >= 100")
    seed_value = payload.get("seed", 0)
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise InputError("seed must be an integer")
    max_goals_value = payload.get("max_goals", 10)
    if isinstance(max_goals_value, bool) or not isinstance(max_goals_value, int) or max_goals_value < 5 or max_goals_value > 20:
        raise InputError("max_goals must be an integer between 5 and 20")
    rho = require_number(payload.get("dixon_coles_rho", 0.0), "dixon_coles_rho")
    rng = random.Random(seed_value)
    sampler, model_metadata = build_rate_sampler(payload, rng, warnings)

    probability_samples = {key: [] for key in OUTCOMES}
    home_rate_samples: List[float] = []
    away_rate_samples: List[float] = []
    over_samples: List[float] = []
    btts_samples: List[float] = []
    tails: List[float] = []
    matrix_accumulator = [[0.0 for _ in range(max_goals_value + 1)] for _ in range(max_goals_value + 1)]

    for _ in range(draws_value):
        home_rate, away_rate = sampler()
        if not (0.0 < home_rate < 20.0 and 0.0 < away_rate < 20.0):
            raise InputError("sampled goal rate is outside the supported range (0, 20)")
        matrix, tail = score_matrix(home_rate, away_rate, max_goals_value, rho)
        probabilities, over_2_5, btts_yes = matrix_markets(matrix)
        home_rate_samples.append(home_rate)
        away_rate_samples.append(away_rate)
        tails.append(tail)
        over_samples.append(over_2_5)
        btts_samples.append(btts_yes)
        for key in OUTCOMES:
            probability_samples[key].append(probabilities[key])
        for home_goals, row in enumerate(matrix):
            for away_goals, probability in enumerate(row):
                matrix_accumulator[home_goals][away_goals] += probability

    mean_matrix = [[value / draws_value for value in row] for row in matrix_accumulator]
    scorelines = [
        {
            "score": f"{home_goals}-{away_goals}",
            "probability": rounded(mean_matrix[home_goals][away_goals]),
        }
        for home_goals in range(max_goals_value + 1)
        for away_goals in range(max_goals_value + 1)
    ]
    scorelines.sort(key=lambda item: item["probability"], reverse=True)
    mean_tail = statistics.fmean(tails)
    if mean_tail > 0.001:
        warnings.append("The mean score-grid truncation tail exceeds 0.1%; increase max_goals.")
    if rho == 0.0:
        warnings.append("Dixon-Coles rho is zero; no low-score correlation correction was applied.")
    probabilities_summary = {key: summarize(probability_samples[key]) for key in OUTCOMES}
    means_total = sum(probabilities_summary[key]["mean"] for key in OUTCOMES)
    if abs(means_total - 1.0) > 0.00001:
        warnings.append("Rounded model probability means do not sum exactly to one; use unrounded internals for calculations.")
    return {
        "mode": mode,
        "draws": draws_value,
        "seed": seed_value,
        "max_goals": max_goals_value,
        "dixon_coles_rho": rounded(rho),
        "metadata": model_metadata,
        "posterior_goal_rates": {"home": summarize(home_rate_samples), "away": summarize(away_rate_samples)},
        "probabilities": probabilities_summary,
        "fair_decimal_odds_from_mean": {
            key: rounded(1.0 / probabilities_summary[key]["mean"]) for key in OUTCOMES
        },
        "derived_markets": {"over_2_5": summarize(over_samples), "btts_yes": summarize(btts_samples)},
        "top_scorelines": scorelines[:5],
        "mean_truncation_tail": rounded(mean_tail),
    }


def analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise InputError("top-level JSON must be an object")
    match = validate_match(payload)
    warnings: List[str] = []
    market = market_baseline(payload, match["as_of"])
    if payload.get("mode") == "market_only" and market is None:
        raise InputError("market_only mode requires market.books")
    if market and market["books_count"] == 1:
        warnings.append("Only one bookmaker was supplied; this is not a market consensus.")
    model = run_model(payload, warnings)
    comparison = None
    if model is not None and market is not None:
        comparison = {
            "edge_percentage_points": {
                key: rounded(100.0 * (model["probabilities"][key]["mean"] - market["consensus_probabilities"][key]))
                for key in OUTCOMES
            }
        }
    forecast_probabilities = (
        {key: model["probabilities"][key]["mean"] for key in OUTCOMES}
        if model is not None
        else dict(market["consensus_probabilities"])
    )
    forecast_record = {
        "match_id": match["id"],
        "target": "result_90min",
        "stage": match.get("stage", "unspecified"),
        "data_level": payload.get("data_level", "L3" if model is None else "unspecified"),
        "snapshot_kind": payload.get("snapshot_kind", "unspecified"),
        "forecast_role": "model" if model is not None else "market",
        "kickoff": match["kickoff_utc"],
        "frozen_at": match["as_of"],
        "probabilities": {key: rounded(forecast_probabilities[key]) for key in OUTCOMES},
    }
    if model is not None and market is not None:
        forecast_record["benchmark"] = dict(market["consensus_probabilities"])
        captured = [book["captured_at"] for book in market["books"]]
        forecast_record["benchmark_meta"] = {
            "source": "multi-book proportional de-vig consensus",
            "books_count": market["books_count"],
            "captured_at_min": min(captured),
            "captured_at_max": max(captured),
        }
    return {
        "match": match,
        "model": model,
        "market": market,
        "comparison": comparison,
        "forecast_record": forecast_record,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON input file")
    parser.add_argument("--output", type=Path, help="write JSON output to this file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.input.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = analyze(payload)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except (OSError, json.JSONDecodeError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
