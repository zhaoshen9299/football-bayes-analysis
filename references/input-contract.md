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
python scripts/evaluate_forecasts.py forecasts.jsonl --bins 10 --output metrics.json
```

每行一个在开赛前冻结的预测：

```json
{"match_id":"m1","kickoff":"2026-07-01T19:00:00Z","frozen_at":"2026-07-01T12:00:00Z","probabilities":{"home":0.5,"draw":0.28,"away":0.22},"outcome":"home","benchmark":{"home":0.48,"draw":0.29,"away":0.23}}
```

- `outcome` 只能是 `home`、`draw`、`away`。
- 三项概率必须合计为 1，脚本不会静默归一化。
- 同时提供 `kickoff` 与 `frozen_at` 时，脚本拒绝赛后冻结记录。
- `benchmark` 可选，但建议保存同一截止时间的去水市场概率。
- 用完整连续样本，不要只保留有利比赛。

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
