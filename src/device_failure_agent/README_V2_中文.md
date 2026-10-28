# v2：知识先验 + 原始事件解析 + 错例反馈 + 交叉验证采纳（2026-10-27）

v2 在原有三角色闭环（分析 / 特征 / 代码 agent）上改了五处。评分口径、时间划分、标签、选择性采样和测试隔离都没有变。

| 改动 | 解决什么问题 | 代码 |
|---|---|---|
| ① 知识先验 | 让 agent 在没见过数据结果时，先用器件失效机理提出可证伪的假设；每次特征请求都必须注明在验证哪条 | `agent.py: Loop.hypotheses`，`prompts/feature.txt` task=hypotheses |
| ② 原始事件解析 | 官方 99 个窗口特征丢失了错误比特形态和地址空间结构；解析器把 `burst_info` 和 DRAM 地址解码成 65 个因果特征 | `raw_events.py`，`raw_cache.py` |
| ③ 错例驱动反馈 | 给 agent 看"高分区真故障和误报在哪些量上不同"，而不是只给 F1 | `contrast.py`，诊断类型 `high_score_contrast` |
| ③' 特征预筛 | 新特征先在父模型高分区比较 TP 和 FP，区分不开就不训练 | `agent.py: features()`，`contrast.prescreen` |
| ④⑤ 交叉验证采纳 | 选择窗口只有约 51 台故障设备，比不出 trial 之间的差别。改为在 TRAIN 内按设备分组做 5 折，配对 bootstrap 区间下界大于 0 才采纳 | `crossval.py`，`agent.py: accept` |

## 原始特征（`raw_*`）

按每个官方观测时刻计算，只使用该时刻及之前的原始 CE 事件：

- 窗口（6h / 24h / 7d）：事件数、单次 CE 的最大错误比特数、最大 DQ 数、最大 beat 数、最小 DQ 间隔（只算多 DQ 的 CE）、最大 beat 间隔、相邻 DQ 事件数、多 DQ 事件数、多 beat 事件数、risky 事件数（同时跨 beat 两半和 DQ 两半），以及不同的 cell / row / column / bank / 芯片数。
- 全历史：累计事件数；不同 cell / row / column / bank / 芯片 / rank 数；历史最大比特、DQ、beat 数；risky 和相邻 DQ 事件累计数；最多 cell 的占比（=1 表示只有一个坏 cell）；单行最多列数、单列最多行数；距首次 CE 的天数；最近 7 天新出现的 row / column / 比特形态数；scrub 占比。

对应关系 `bit // 4 == deviceID`（芯片号）已在全部 56,403 个原始文件、265 万种比特形态上核对过，没有不一致。比特编号最大为 143，即 36 颗芯片。没有原始 CSV 的设备（8,391 台），raw 特征保持缺失，不填 0。

## 交叉验证口径

- 设备按"在 TRAIN 内是否故障"分层后随机分成 5 折，种子固定。每折用其余 4 折设备的 TRAIN 行训练，原选择性采样只保留这些设备对应的行。
- 留出设备在 TRAIN 末段连续 5 个 15 天时期上评分，cohort、预测时刻、聚合方式和提前量规则都与验证集完全相同。`tests/test_crossval.py` 按 5 种聚合方式、多个时期逐设备核对，结果与 `prediction.predict` 相等。
- 合并所有 (时期, 设备) 单元，选出步数和阈值，再与当前 best 做配对、按故障分层的 bootstrap（1000 次）。
- 采纳条件：区间下界 > `accept_lower_bound`（默认 0），并且新增命中 ≥ 丢失命中。选择窗口 F1 比 best 低超过 `screen_margin`（默认 0.02）的实验直接淘汰，不跑交叉验证。
- 确认窗口（验证集后半段）在搜索结束后只用一次，阈值冻结。TEST 不参与。

## 运行

```bash
# 1. 原始特征缓存（一次，约 15 分钟，32 进程）
python runtime/raw_cache.py --source-cache <官方 cache> --raw stage1_csv.zip --out raw_cache
# 2. 测试 + 建会话
python prepare_v2.py
# 3. 离线对照（可选，写报告用）：采样是否有效、原始特征是否有效
python compare_cv.py
# 4. 启动搜索（隐藏输入 DeepSeek Key，只存在内存里）
python key_gate_v2.py
```

## 测试

`tests/test_raw_events.py`、`test_pool_scores.py`、`test_crossval.py`、`test_v2_loop.py`，外加顶层 `test_search_v2.py`。原有的 `tests/test_workflow.py` 中有 5 个用例在原版 runtime 上同样失败（测试夹具缺少 cache），与 v2 无关，未修改。
