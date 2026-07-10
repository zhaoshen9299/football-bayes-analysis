# 方法参考

## 目录

1. 预测对象
2. 模型层级
3. 经验贝叶斯近似
4. 比分分布与低比分修正
5. 市场去水与比较
6. 阵容和战术证据
7. 不确定性
8. 回测和校准
9. 研究依据

## 1. 预测对象

默认预测 90 分钟常规时间主胜、平局、客胜。加时、点球和晋级是不同目标，必须另建模型。固定分析截止时间 `t0`，只允许使用 `t0` 之前可获得的信息。

使用以下分解：

- `P0`：联赛、球队长期实力、主客场和赛制构成的先验。
- `L(D|theta)`：截止时间之前可比比赛数据的似然。
- `P(theta|D)`：球队进攻、防守和主场效应的后验。
- `P(Y|D)`：从参数后验积分得到的比分与胜平负后验预测。

市场概率是外部基准，不自动等于先验，也不自动和模型混合。

## 2. 模型层级

### L1：动态分层模型

数据充分时使用：

```text
HomeGoals_i ~ Poisson(lambda_home_i)
AwayGoals_i ~ Poisson(lambda_away_i)

log(lambda_home_i) = intercept + home_advantage
                     + attack_home(team_h, t)
                     - defense_away(team_a, t) + X_i beta

log(lambda_away_i) = intercept
                     + attack_away(team_a, t)
                     - defense_home(team_h, t) + X_i beta
```

给攻击和防守参数设置联赛级层次先验，并允许随时间演化。按时间滚动验证超参数；不要用随机切分破坏时间顺序。若低比分相关性显著，可使用双变量 Poisson 或 Dixon-Coles 修正。

### L2：经验贝叶斯 Gamma-Poisson 近似

本 Skill 的脚本用于无法拟合完整层级模型、但有同口径比赛级进球或 xG 时。对每个角色序列设置 Gamma 先验：

```text
lambda ~ Gamma(alpha0, beta0)
alpha0 = league_role_mean * prior_weight
beta0  = prior_weight
```

对第 `i` 个观测使用时间和可靠性权重：

```text
w_i = reliability_i * 0.5 ** (days_ago_i / half_life_days)
alpha_post = alpha0 + sum(w_i * y_i)
beta_post  = beta0  + sum(w_i)
```

角色后验组合为：

```text
lambda_home = home_attack_rate * away_defense_rate / league_home_rate
lambda_away = away_attack_rate * home_defense_rate / league_away_rate
```

进球计数下这是加权的共轭近似；把 xG 当作小数伪计数时属于准似然近似，必须在报告中说明。`prior_weight` 与半衰期必须通过历史时序验证，不得为当前比赛调参。

### L3：市场基准

只有赔率时，输出去水概率。不要从结果赔率反推精细比分率，也不要把市场概率称作独立贝叶斯模型。

## 3. 先验与数据口径

- 使用同赛事、同赛季阶段或合理的跨赛季衰减建立联赛主客场基线。
- 升班马、换帅、跨联赛球队和长时间停赛后的球队应增加先验方差或使用更强收缩。
- 主客场角色序列必须匹配：主队主场进攻、主队主场防守、客队客场进攻、客队客场防守。
- 样本很少时优先收缩，不把“近 5 场”当作固定真理。
- xG 供应商改变时切断序列或做已验证的桥接；不得直接连接。

## 4. 比分分布与低比分修正

基础模型假定主客进球在给定参数后条件独立。Dixon-Coles 对低比分格点使用：

```text
tau(0,0) = 1 - lambda_home * lambda_away * rho
tau(0,1) = 1 + lambda_home * rho
tau(1,0) = 1 + lambda_away * rho
tau(1,1) = 1 - rho
tau(x,y) = 1 otherwise
```

将 `tau` 乘到独立 Poisson 概率后归一化。只有当 `rho` 在该赛事历史样本中估计并通过滚动验证时才使用；否则取 `rho=0`。任何 `tau <= 0` 都表示参数无效。

从每次后验参数抽样生成比分矩阵，汇总胜平负、大小球、双方进球和比分。比分网格截断尾部必须足够小并在输出中报告。

## 5. 市场去水与比较

对每家机构的十进制赔率 `o_k`：

```text
q_k = 1 / o_k
overround = sum(q_k) - 1
p_k = q_k / sum(q_k)
```

这是比例去水基线。先逐家去水，再对各结果的公平概率取中位数并归一化。不要先平均赔率。更复杂的 power 或 Shin 方法只能在历史验证优于比例法时替换。

模型差异定义为：

```text
edge_pp = 100 * (p_model - p_market)
```

它只是概率差，不等于期望收益或可下注结论。赔率来源、采集时间、机构数量和离散度必须一并报告。不要把收盘赔率用于模拟更早时点可执行的预测。

## 6. 阵容和战术证据

把信息分成三层：

1. **事实层**：球员是否可用、预计位置、阵型、休息天数。
2. **机制层**：信息通过何种机制影响射门数量、射门质量或失球风险。
3. **量化层**：是否存在经验证的球员 on/off、替代者、位置组或历史情景效应。

只有三层都足够时才改变无条件进球率。推荐在对数尺度表达：

```text
lambda_adjusted = lambda_base * exp(delta)
```

`delta` 必须有均值、标准差或范围、来源和状态。没有量化层时运行独立情景，不做直接胜率加减。

“控球陷阱”只能作为待验证假设。至少联合检查推进或领地、禁区进入、射门质量、比赛状态和对手策略；不存在跨联赛通用阈值。

## 7. 不确定性

至少区分：

- 参数不确定性：球队强度和进球率后验宽度；
- 数据不确定性：伤停状态、供应商缺失、赔率延迟；
- 模型不确定性：Poisson 假设、相关性、跨联赛迁移；
- 情景不确定性：首发和战术选择。

报告 90% 后验区间或明确标记“无可用区间”。区间不包含突发红牌等赛中随机性；不要把它解释成 90% 的赛果保证。

证据等级建议：

- A：官方或原始数据、时间戳完整、两个来源一致；
- B：可靠二手来源或单一高质量数据源；
- C：聚合、预计、样本口径不完整；
- U：不可验证，不进入模型。

## 8. 回测和校准

按开球时间做扩展窗口或滚动窗口验证。每条预测必须保存 `frozen_at`，并与同一截止时点可获得的市场基准比较。

主要指标：

- **Log Loss**：`-log(p_observed)`，越低越好，强烈惩罚过度自信。
- **Multiclass Brier**：三个结果上平方误差之和的均值，越低越好。本 Skill 不除以类别数。
- **校准分箱**：比较预测概率均值和实际发生率，同时报告样本数。
- **RPS**：可作为补充，不作为唯一指标。

同时检查覆盖率、赛事分层、时间稳定性和相对市场差值。CLV 可衡量价格移动，但不能替代预测准确性与校准。

`scripts/evaluate_forecasts.py` 使用冻结预测计算以上指标。任何参数调整都必须在下一段未见数据上验证。

## 9. 研究依据

- Dixon 与 Coles 的低比分修正及时间加权思路：[Lancaster University 论文记录](https://research.lancaster-university.uk/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-f/)，DOI `10.1111/1467-9876.00065`。
- Baio 与 Blangiardo 的足球分层贝叶斯进球模型及过度收缩讨论：[UCL 论文 PDF](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)。
- 足球概率评分规则的比较与 Log Loss 讨论：[LSE 论文 PDF](https://eprints.lse.ac.uk/111494/3/Wheatcroft_evaluating_probabilistic_forecasts_published.pdf)。
- xG 是供应商特定的射门得分概率模型，不同模型使用的上下文特征可能不同：[Hudl StatsBomb 说明](https://www.hudl.com/blog/expected-goals-xg-explained)。
