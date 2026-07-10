# Football Bayes Analysis

一个面向 Codex 的足球赛前概率分析 Skill：使用可追溯的数据快照、贝叶斯/经验贝叶斯进球模型、市场赔率去水、战术情景与赛后校准，对足球比赛给出胜平负概率和不确定性说明。

它不会把搜索到的数字直接拼成“推荐”，也不会用主观百分点加减冒充贝叶斯更新。数据不足时会主动降级，必要时拒绝输出伪精确概率。

## 主要能力

- 区分外部事实、模型输入、模型派生结果和战术假设。
- 支持完整模型、经验贝叶斯、外部后验进球率和纯市场基准四级降级。
- 使用 Gamma-Poisson 收缩、时间衰减和可选 Dixon-Coles 低比分修正。
- 对多家机构的十进制赔率逐家去水，再形成市场概率基准。
- 输出90分钟胜/平/负、后验区间、比分分布和模型—市场差异。
- 支持伤停、预计首发、战术机制和不确定性情景分析。
- 支持重复查询刷新、多场分批、快照对比和断点续写。
- 提供 Log Loss、Multiclass Brier、RPS 和校准分箱的赛后评估脚本。

## 设计原则

1. **赛前截止**：只使用分析截止时间之前可获得的信息。
2. **概率而非断言**：最高概率结果不等于必然结果。
3. **证据分级**：区分官方确认、可靠报道、推断和假设。
4. **市场是基准**：先去水，再比较；不自动把市场与模型混合。
5. **战术必须可证伪**：没有量化映射时只做情景，不直接修改胜率。
6. **允许降级**：数据不足时宁可输出 `UNAVAILABLE`，也不编造输入。
7. **持续校准**：用冻结的赛前预测做时序评估，禁止赛后回填。

## 目录结构

```text
football-bayes-analysis/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── data-sources.md
│   ├── input-contract.md
│   ├── methodology.md
│   └── report-template.md
└── scripts/
    ├── predict_match.py
    └── evaluate_forecasts.py
```

## 安装

### Codex

克隆到用户级 Skills 目录：

```powershell
git clone https://github.com/zhaoshen9299/football-bayes-analysis.git "$env:USERPROFILE\.codex\skills\football-bayes-analysis"
```

macOS/Linux：

```bash
git clone https://github.com/zhaoshen9299/football-bayes-analysis.git ~/.codex/skills/football-bayes-analysis
```

安装后新开一个 Codex 任务，即可显式调用：

```text
$football-bayes-analysis 分析西班牙对比利时的世界杯比赛。
```

也可以直接描述比赛、开球时间和希望分析的市场，由 Codex 自动触发。

## 使用示例

```text
使用 $football-bayes-analysis 分析：
世界杯四分之一决赛，西班牙 vs 比利时。
请给出90分钟胜平负概率、市场去水概率、阵容影响、两个比赛情景和刷新条件。
```

```text
刷新上一版分析，重新抓取官方首发和即时赔率，并单独列出数据变化与方法变化。
```

```text
批量分析以下5场比赛。每场保存独立快照，按数据质量选择模型层级，不要共享阵容假设。
```

## 数据层级

| 层级 | 可用数据 | 输出方式 |
|---|---|---|
| L1 完整模型 | 同口径历史比赛/xG、联赛基线、阵容、赔率 | 动态分层 Poisson、双变量 Poisson 或已验证模型 |
| L2 收缩模型 | 近期进球或 xG、基线、赔率 | 经验贝叶斯 Gamma-Poisson 近似 |
| L3 市场基准 | 可靠的多机构赔率 | 去水市场概率＋定性情景 |
| L4 数据不足 | 无可靠历史数据或赔率 | 已确认事实、数据缺口，数值概率为 `UNAVAILABLE` |

## 命令行计算

脚本只依赖 Python 标准库。

根据 JSON 输入生成概率：

```bash
python scripts/predict_match.py match.json --output prediction.json
```

支持三种模式：

- `empirical_bayes`：对四个攻防角色序列做时间衰减和 Gamma-Poisson 收缩。
- `direct_rates`：传播外部模型给出的主客队后验进球率及其不确定性。
- `market_only`：只计算逐家赔率去水后的市场共识。

完整 JSON 协议见 [`references/input-contract.md`](references/input-contract.md)。

评估冻结的历史预测：

```bash
python scripts/evaluate_forecasts.py forecasts.jsonl --bins 10 --output metrics.json
```

## 输出边界

- 默认预测90分钟常规时间；晋级、加时和点球必须单独建模。
- 不同供应商的 xG、Field Tilt、PPDA 等指标不得直接拼接。
- 没有历史验证的伤停或战术判断不得直接换算成胜率百分点。
- 模型高于市场不自动等于“有价值”，还需考虑历史回测、误差、赔率分散和数据延迟。
- 本 Skill 不用于滚球分析，不承诺命中率或收益，不提供追损建议。

## 方法依据

- Dixon, M. J. & Coles, S. G. (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, DOI: [`10.1111/1467-9876.00065`](https://doi.org/10.1111/1467-9876.00065)
- Baio, G. & Blangiardo, M. (2010), *Bayesian Hierarchical Model for the Prediction of Football Results* ([PDF](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf))
- Wheatcroft, E. (2021), *Evaluating Probabilistic Forecasts of Football Matches* ([PDF](https://eprints.lse.ac.uk/111494/3/Wheatcroft_evaluating_probabilistic_forecasts_published.pdf))

## 许可

本仓库当前未声明开源许可证。公开可见不等于自动授予复制、修改或再分发权利；如需开放许可，可后续单独添加。

## 免责声明

本项目用于足球数据研究和概率分析。所有输出均依赖数据质量、模型假设和分析截止时间，不构成赛果保证、投资建议或投注指令。
