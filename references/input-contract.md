# 脚本输入协议

两个脚本只使用 Python 标准库。所有概率输出为 0–1 小数，报告层再转成百分比。

## `predict_match.py`

运行：

```bash
python scripts/predict_match.py match.json --output prediction.json
```

### 通用字段

```json
{
  "match": {
    "id": "competition|season|kickoff_utc|home|away",
    "home_team": "Home FC",
    "away_team": "Away FC",
    "kickoff_utc": "2026-07-12T19:00:00Z",
    "as_of": "2026-07-10T12:00:00Z"
  },
  "mode": "empirical_bayes",
  "draws": 5000,
  "seed": 20260710,
  "max_goals": 10,
  "dixon_coles_rho": 0.0,
  "market": {
    "books": [
      {
        "name": "Book A",
        "captured_at": "2026-07-10T11:55:00Z",
        "home": 2.1,
        "draw": 3.4,
        "away": 3.6
      },
      {
        "name": "Book B",
        "captured_at": "2026-07-10T11:58:00Z",
        "home": 2.08,
        "draw": 3.45,
        "away": 3.7
      }
    ]
  }
}
```

- 时间使用带时区的 ISO 8601。
- `as_of` 必须早于开球；所有赔率 `captured_at` 必须不晚于 `as_of`，否则脚本拒绝计算。
- 赔率必须是十进制，每家机构的主/平/客必须来自同一快照。
- `draws` 建议至少 2000；固定 `seed` 保证复现。
- `max_goals` 是每队比分网格上限；检查输出的 `mean_truncation_tail`。
- `dixon_coles_rho` 默认 0；只填入经该赛事历史验证的值。

### 模式一：`empirical_bayes`

在通用字段上增加：

```json
{
  "data_type": "xg",
  "league_baseline": {
    "home_rate": 1.52,
    "away_rate": 1.18,
    "prior_weight": 8.0
  },
  "half_life_days": 120.0,
  "series": {
    "home_attack": [
      {"value": 1.7, "days_ago": 8, "reliability": 1.0}
    ],
    "home_defense": [
      {"value": 1.1, "days_ago": 8, "reliability": 1.0}
    ],
    "away_attack": [
      {"value": 1.3, "days_ago": 6, "reliability": 1.0}
    ],
    "away_defense": [
      {"value": 1.6, "days_ago": 6, "reliability": 1.0}
    ]
  },
  "adjustments": {
    "home_log_rate": [],
    "away_log_rate": []
  }
}
```

角色含义：

- `home_attack`：主队在可比主场比赛的进球或 xG for。
- `home_defense`：主队在可比主场比赛的失球或 xG against；其先验均值是联赛客队进球率。
- `away_attack`：客队在可比客场比赛的进球或 xG for。
- `away_defense`：客队在可比客场比赛的失球或 xG against；其先验均值是联赛主队进球率。

`data_type` 只能是 `goals` 或 `xg`。使用 xG 时脚本会提示这是小数伪计数的准似然近似。`reliability` 必须在 0–1；缺省为 1，只用于数据质量，不要借它手工迎合结论。

调整项格式：

```json
{
  "label": "已验证的主力中锋缺阵影响",
  "mean": -0.08,
  "sd": 0.03,
  "source": "player-model-v3"
}
```

它表示把进球率乘以 `exp(delta)`。只有经验证的影响映射可以进入基础后验；未确认或纯战术判断应复制输入文件另跑情景。

### 模式二：`direct_rates`

已有外部模型的后验进球率时使用：

```json
{
  "mode": "direct_rates",
  "goal_rates": {
    "home": {"mean": 1.55, "sd": 0.2},
    "away": {"mean": 1.08, "sd": 0.16}
  }
}
```

脚本用匹配均值和标准差的 Lognormal 分布传播正值不确定性。若外部模型已提供后验抽样，优先在外部直接计算，不要用均值/标准差压缩复杂后验。

### 模式三：`market_only`

只保留通用字段中的 `match`、`mode` 和 `market`。脚本只输出逐家去水后的市场共识，不生成比分与独立模型概率。

### 输出解释

- `model.probabilities.*.mean/p05/p95`：后验胜平负及 90% 参数区间。
- `model.posterior_goal_rates`：主客进球率后验摘要。
- `model.top_scorelines`：平均后验比分矩阵中概率最高的比分。
- `market.consensus_probabilities`：逐家比例去水、取中位数后再归一化的市场概率。
- `comparison.edge_percentage_points`：模型减市场，仅是差异，不是投注建议。
- `warnings`：必须带入最终报告中与本场有关的限制。

## `evaluate_forecasts.py`

运行：

```bash
python scripts/evaluate_forecasts.py forecasts.jsonl --manifest manifest.jsonl --bins 10 --output metrics.json
```

每行一个在开赛前冻结的预测。推荐使用带90分钟实际比分的新协议：

```json
{"match_id":"m1","target":"result_90min","stage":"Quarter-finals","data_level":"L3","snapshot_kind":"pre_match","forecast_role":"market","kickoff":"2026-07-01T19:00:00Z","frozen_at":"2026-07-01T12:00:00Z","probabilities":{"home":0.5,"draw":0.28,"away":0.22},"benchmark":{"home":0.48,"draw":0.29,"away":0.23},"actual":{"home_goals_90":1,"away_goals_90":1,"result_type":"aet","source":"official-match-report"}}
```

- `target` 当前必须为 `result_90min`；晋级概率不得混入这个文件。
- `actual.result_type` 为 `regular`、`aet` 或 `penalties`。后两者要求90分钟比分为平局。
- `actual.source` 必填，优先保存官方比赛报告链接或稳定标识。
- 旧协议仍接受 `outcome`，但会警告无法核验90分钟比分。
- 三项概率必须合计为 1，脚本不会静默归一化。
- `kickoff` 与 `frozen_at` 必填，脚本拒绝赛后冻结记录。
- `benchmark` 可选，但建议保存同一截止时间的去水市场概率。
- 有 `benchmark` 时同时保存 `benchmark_meta.source` 与 `benchmark_meta.captured_at_max`；脚本拒绝晚于 `frozen_at` 的基准。
- 用完整连续样本，不要只保留有利比赛；同场多个快照不会被视作更多独立比赛。

完整性清单每行格式为 `{"match_id":"m1","status":"FROZEN_FORECAST"}`；状态可为 `FROZEN_FORECAST`、`NO_FORECAST`、`INVALID_FORECAST` 或 `PENDING_RESULT`。脚本据此输出预测覆盖率；未提供清单时明确警告覆盖率未知。

## `evaluate_qualification.py`

晋级概率使用独立二项协议：

```bash
python scripts/evaluate_qualification.py qualification.jsonl --output qualification-metrics.json
```

```json
{"match_id":"m1","target":"to_qualify","snapshot_kind":"pre_match","kickoff":"2026-07-01T19:00:00Z","frozen_at":"2026-07-01T12:00:00Z","probabilities":{"home":0.58,"away":0.42},"actual":{"qualified":"home","source":"official-match-report"}}
```

脚本输出二项 Log Loss、Brier、最高概率命中率和实际结果平均获配概率；不得把三项90分钟概率直接归一化成晋级概率。

## `evaluate_corners.py`

运行：

```bash
python scripts/evaluate_corners.py corner-forecasts.jsonl --output corner-metrics.json
```

每行记录冻结的预测摘要、半球线概率和官方90分钟角球：

```json
{"match_id":"m1","kickoff":"2026-07-01T19:00:00Z","frozen_at":"2026-07-01T12:00:00Z","prediction":{"home":{"mean":5.7,"p05":2,"p95":10},"away":{"mean":4.5,"p05":1,"p95":9},"total":{"mean":10.2,"p05":5,"p95":16},"total_lines":{"9.5":{"over":0.57}}},"actual":{"home":6,"away":3,"period":"90min_including_stoppage","source":"official-match-report"}}
```

脚本拒绝未声明90分钟口径或实际值来源的角球记录，输出主客及总数MAE、RMSE、均值误差、90%区间覆盖率和各半球线Brier。

## `predict_corners.py`

运行：

```bash
python scripts/predict_corners.py corners.json --output corners-prediction.json
```

输入沿用 `match`，并提供：

```json
{
  "match": {"id":"m1","home_team":"Home","away_team":"Away","kickoff_utc":"2026-07-20T19:00:00Z","as_of":"2026-07-20T12:00:00Z"},
  "draws": 20000,
  "seed": 7,
  "half_life_days": 180,
  "corner_baseline": {"home_rate":5.5,"away_rate":4.6,"prior_weight":10},
  "series": {
    "home_for":[{"value":7,"days_ago":8,"reliability":1}],
    "home_against":[{"value":3,"days_ago":8,"reliability":1}],
    "away_for":[{"value":5,"days_ago":6,"reliability":1}],
    "away_against":[{"value":6,"days_ago":6,"reliability":1}]
  },
  "total_lines":[8.5,9.5,10.5,11.5]
}
```

四个序列必须是同供应商、同90分钟口径的非负整数角球。`reliability` 为0–1；先验权重和半衰期须由历史滚动验证。脚本输出主客及总角球预测摘要、各半球线大小概率、角球数领先概率和最可能总角球。

## `backtest_league.py`

对三个连续完整联赛 CSV 做训练／验证／测试：

```bash
python scripts/backtest_league.py --train E0_2324.csv --validation E0_2425.csv --test E0_2526.csv --output-dir backtest-output --draws 1000 --seed 20260714
```

CSV 至少需要 `Date`、`HomeTeam`、`AwayTeam`、`FTHG`、`FTAG`、`FTR`、`HC`、`AC`、`AvgH`、`AvgD`、`AvgA`、`AvgCH`、`AvgCD`、`AvgCA`、`AvgC>2.5`、`AvgC<2.5`；`Time` 可选。脚本按开球时间滚动，对同时开球比赛批量更新，验证集锁定超参数，输出逐场 `match_review.csv` 和汇总 `metrics.json`。所有脚本只依赖 Python 标准库。
