# v3：误报与漏报双向诊断的设备故障实验框架

本包基于本地 `autodl_v2/runtime` 的2026-10-28版本开发。它是独立交付版本；旧代码、会话、模型与服务器任务保持原状。改过程序或表示后必须建立新会话，不能重写旧会话的 code_seal。

## 本次实现

1. **误报与漏报分开诊断**：保留 `high_score_contrast`，增加 `missed_fault_contrast` 和 `error_atlas`。后者同时输出 TP/FP 与 FN/TN 的探索性特征对比，以及没有及时预测时刻、有时刻但没有分数、有分数但低于阈值的漏报数量。窗口外告警单列为重叠计数，不把这些类别错误相加。
2. **双向预筛与有限探索**：默认 balanced，任一对比发现单变量信号就可训练；没有信号时，分析 Agent 可提交带 hypothesis_id 的 features 请求，用 exploration_reason 解释组合效应、召回或数据覆盖问题，申请有限探索名额（默认整个会话2次）。名额持久化，恢复不会重复扣除。strict_high 保留为原规则对照。诊断不显著不等于证明无效。
3. **独立补充分支**：新增 `rerank=parallel`，单阶段父模型固定；第二分支对固定采样规则保留的所有训练行学习，不受父模型分数门限限制。第二分支采用设备权重总和为1、平方根类别权重、深度2的XGBoost；逐行分数取两分支最大值，再走原预测时间表、聚合与评分。它可以提起父模型低分，但也可能新增误报，仍须通过采纳检验。它是通用的组合模型实验，**不是完整M²-MFP，也不是论文的DIMM-centric规则树**；输入是现有数值观测行，可能已经过时间汇总。
4. **设备权重单因素对照**：scope默认开放 base_weight，仅改权重可单独检验设备日志数量偏差。parallel同时改变结构和权重，其收益不能全部归功于某一个因素。
5. **可选内存结构适配器**：`raw_spatial_v3.py` 保留v1原始特征，增加15分钟/1小时统计、事件形状摘要、按deviceID区分的芯片内位置数量、比特信息可用计数。原有逻辑地址统计不改名、不改义。通过 `--representation spatial_v3` 新建缓存才能启用；没有原始日志的设备保持缺失。它不是完整Multi-BSFE。其他领域使用自己的字段适配，通用控制器不假设硬盘有DQ/bank。
6. **重复设备的统计处理**：诊断区间与CV配对区间按设备整体重采样，将同一设备的各时间段放在一起。CV仍用于搜索和挑选阈值/步数，结果是探索证据，不能冒充独立测试显著性。
7. **提示真正进入Agent调用**：包内 skill、共享 experiment-method.md、analysis/feature提示词已同步；providers.py每次向三个角色注入同一份共享方法。分析Agent收到合法动作、双向诊断、剩余探索名额、误报/漏报计数；特征Agent收到目标与已有诊断证据。

## 如何与当前实验衔接

先只换框架，复用原TRAIN/VAL缓存和采样政策，单独比较预筛、权重或分支。启用空间适配器属于第二个独立数据表示实验，不与框架修改混为单因素。

依赖：Python及 requirements.txt；本次在Windows、Python 3.13、XGBoost 3.4.1上做CPU合成数据验证。没有调用付费DeepSeek API，也没有运行真实比赛全量训练或访问服务器。

在新目录解压，从已有服务器会话建立新会话（以下路径是占位符）：

```bash
python search_step.py --initialize-from /path/to/old/session --session /path/to/new/v3_session --active-cache /path/to/train_val_cache --arm arm_policy.json
python agent.py run --session /path/to/new/v3_session
```

运行前由使用者在进程环境中设置 `DEEPSEEK_API_KEY`。不要写入代码、配置、交付包或聊天。保留原来已部署的 `TYPEA_SEARCH_ISOLATION` 配置并将新会话列为允许输出；旧会话/原始数据/TEST的隔离范围须与服务器实际目录对应。这里没有包含机器专用密钥或调度器，也没有自动启动训练。

初始化默认读取来源配置的 active_cache（没有时使用 source/cache），保留来源sampling_reference；若主动提供不同缓存，要求保持来源的时间切分。缓存、采样引用必须在目标机器可用；本包不携带真实数据、已有模型或会话。新会话会重新建立自己的基线，不宣称导入了旧模型。

配置中的 max_trials、max_agent_calls 等仍是总预算；探索名额不是额外训练预算。独立分支和设备权重可通过 arm.config 的 parallel_allowed/device_weight_allowed 关闭，便于做消融。

## 可选空间缓存

从**不含raw_*列的官方缓存**出发，先做一个分片的资源检查：

```bash
python raw_cache.py --source-cache /path/to/official_train_val_cache --raw /path/to/type_a_csv_directory --out /path/to/spatial_smoke --smoke-parts 1 --representation spatial_v3
python raw_cache.py --source-cache /path/to/official_train_val_cache --raw /path/to/type_a_csv_directory --out /path/to/new_spatial_cache --workers 4 --representation spatial_v3
```

支持原有stage1_csv.zip格式。表示版本和解析器哈希写入manifest；版本/解析器不同的已有缓存拒绝复用，已有或partial输出不自动删除。构建器会读取CSV，再在计算前剔除分区结束之后的行；若要求连原始TEST字节也不能读取，应提供事先隔离的TRAIN/VAL文件。Agent搜索仍只读取已允许的缓存。

新增短窗口和结构描述会增加处理时间、缓存大小和内存。合成小数据验证不代表真实全量资源已经验证；先用smoke结果估算再决定并发。

## 判断收益

- 固定数据切分、评估队列、预警窗口、训练预算；只比较同 evaluation_id 的结果。
- 记录新找回的故障、丢失命中、新增误报、消除误报；不只看一项F1。
- 当前策略的CV有重复使用和选择偏差。确认窗口在模型冻结后使用，曾经被查看的时期如实标注；本版本训练入口继续拒绝TEST。
- 内存/硬盘合成数据跑通表示接口兼容；真实跨设备有效性仍需各自的独立实验。不能把合成数据分数写成比赛结果。

## 验证与交接

```bash
python -m unittest discover -s tests -v
```

交付中的“验证结果.txt”记录最终测试数、环境与局限；“修改清单.json”按文件列出新增/修改与哈希，“相对v2的修改.diff”便于其他AI审阅。先读本文件，再读包内skill及 experiment-method.md；不要把旧版README中的历史机器路径当成本版本启动配置。
