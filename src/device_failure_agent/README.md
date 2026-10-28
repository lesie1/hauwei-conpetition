# 截断处理修复版 api_fix2

本版将默认生成上限设为 32768，数据适配使用 low 思考强度；遇到 length 结束时最多提高一次，上限 65536。截断不再进入格式修复。已有失败会话按包顶层的运行说明升级，保留全部调用与 token 记录。

本版补全适配输出的精确键和类型枚举，要求紧凑输出并复用同义字段映射。已在原 AutoDL 会话通过真实 DeepSeek 数据适配，覆盖 366 个输入文件；完整预处理和训练未启动，详情见 FIX_VALIDATION.txt。

# API 修复版说明（2026-10-24）

如果已有因 DeepSeek HTTP 400 停止的会话，先按压缩包顶层的 AutoDL_运行说明.txt 升级并续跑；不要重新 init。

本版统一补充 JSON 输出要求，并记录脱敏后的接口错误详情。以下为完整程序的使用说明。

# 通用设备故障实验闭环 · 2026-10-23

**三个角色均使用 DeepSeek：分析 agent、特征 agent、代码 agent。**

参考 CAAFE 的“理解数据 → 生成特征代码 → 执行 → 用下游模型验证 → 反馈”思路，接入原来的 XGBoost 实验闭环。数据路径通过命令传入。每次 `init` 都创建一个独立会话，内存 TypeA、TypeB、硬盘或新器件之间不混用结果。

本包是可运行的工程实现，并非 CAAFE 论文复现。完整流程的自动测试使用模拟 DeepSeek 回复和小型测试数据；真实服务器的接口与映射验证记录见压缩包顶层 FIX_VALIDATION.txt。当前没有完整真实数据训练后的预测性能结论。

## 1. 已实现的流程

```text
用户提供数据目录 + 数据说明 + 任务类型
    ↓
扫描全部文件的表头/格式变体 → 特征 agent 生成字段映射
    ↓
本地分批读取、按设备与可用时间排序、固定 6:2:2 时间划分
    ↓
全新 XGBoost 基线 → 设备级验证结果 → 分析 agent
    ├─ 参数/标签/权重/采样/告警策略 → 代码 agent → 执行实验 ─┐
    ├─ 新特征问题 → 特征 agent → 代码校验与执行 → 特征对照 ─┤
    └─ 诊断请求 → 本地生成证据 ──────────────────────────┘
    ↓
保留最佳结果 → 冻结基线和赢家 → 显式运行最终测试
```

分析回复只描述一个动作，控制器负责补全配置。代码 agent 原样落实已接受的计划，不能悄悄换父实验。特征代码能使用过去滚动统计、累计统计、差分、缺失标记、数值变换和组合，不局限于固定的 26 组参数。

提示词、请求、回复、报错、修复、模型和数据摘要均有记录。格式错误只做有限次修复；分析仍不合格时，程序记录原因并选择一个未试过的单因素参数对照。没有候选时停止。磁盘/内存/超时错误保留结果，等待资源修复，不自动反复训练同一轮。

## 2. 换数据和继续训练的规则

| 情况 | 处理方式 |
|---|---|
| 新输入一类数据 | 新会话、新基线、新模型；特征映射和评分任务单独保存 |
| 同一工作区之前处理过同类数据 | 取最近会话的参数作候选，再在新数据上重新训练、对照；特征经验供参考 |
| 不同器件的数据 | 复用流程与特征生成方法；本版不直接搬用跨器件模型或自动搬用参数 |
| 同一会话只改告警汇总或预测频率 | 检查训练配置和特征一致后复用已保存树，重新预测/评分 |
| 同一会话改特征、训练标签、权重、样本单位或超参数 | 重新训练，不在旧树后盲目追加 |
| 断网、分析失败、终端退出 | 使用原会话 `run`，从保存的步骤继续 |

共享经验存于 `workspace/experience/`；每个会话位于 `workspace/sessions/独立编号/`。保留相同 workspace 才能读取以前的参数经验；数据或语义有变化时新建会话。输入数据不会被改写。

## 3. 评分和可修改范围

**内存**：故障发生在预测时刻后的 **15 分钟～7 天 15 分钟**，含端点，才算及时命中。这沿用 SmartMem 的提前量区间；本包的固定日历期设备评分是本地实现，不声称完整复刻主办方服务。

**硬盘/其他器件**：保留至少提前 **15 分钟**，没有提前天数上限，但只判断当前固定评价期内、已知随访的故障。没有上限不等于未来永远没故障就标健康。

共同采用按时间 6:2:2、同一设备队列、实际发出预测的时间、验证集设备 F1 选阈值。没有预测的已知故障设备仍计 FN；只有错误时机告警的故障设备同时计 FP 和 FN。未知随访单列，不冒充负样本。`precision` 是查准率，不是总体分类准确率。

搜索从小范围参数对照开始；累计完成当前层至少 3 个对照且连续 2 次没有刷新最佳 F1 后，逐层开放标签/权重、再开放采样/时间表示/告警策略。有诊断证据时可以提前请求新特征。具体范围：

- XGBoost 深度、学习率、正则、叶子样本约束、列采样、分箱、随机种子等。
- **训练**标签期限 7/15/30 天。评分窗口保持不变；扩大标签期限会同时改变成熟样本数量，报告会记录。
- 行等权或设备总权重相等；正样本倍率为比例、平方根比例或 1。
- 每条观测训练，或每设备每天一个截至当日的历史样本。
- 原始数值特征、累计历史统计、累计历史加近期统计，以及 agent 生成的新增特征。
- 1 小时/24 小时预测；最新值、窗口最大/均值/第二高、历史最高等告警汇总。

本版训练器仍是 XGBoost 二元交叉熵；“设备等权/设备日样本”不等于直接优化设备 F1，也没有实现 rank loss。不会让 agent 为了提高分数修改评估器。

## 4. AutoDL 安装

上传 `device_failure_agent_20261023_api_fix2.zip` 后，在上传目录解压。以下安装路径只是示例，**程序没有固定的数据路径或工作目录**。

```bash
cd /root/autodl-tmp
python -m zipfile -e device_failure_agent_20261023_api_fix2.zip .
cd device_failure_agent_20261023_api_fix2
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

也可先激活已有环境，再安装依赖。加载已有 DeepSeek 密钥文件：

```bash
set -a
source ~/.config/smartmem-agent-chatgpt/deepseek.env
set +a
```

或者在当前终端安全输入，输入内容不会显示，也不会写进命令历史：

```bash
read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
printf '\n'
```

三个角色默认 `deepseek-v4-pro`，均通过 DeepSeek 接口。可在 `examples/runtime.json` 修改角色模型名、显卡、内存预算和实验预算；初始化时传 `--config`。不需要 OpenAI API key 或 Codex 登录。磁盘应能容纳原始数据、归一化缓存、多个特征版本和模型；实际需要量随数据增长，程序有磁盘检查。

## 5. 接入 TypeB

数据下载并解压后，设置三个变量；路径由你填写。说明文件建议包含设备 ID、时间单位/时区、故障定义、字段含义、随访覆盖和需要选择的 TypeB 范围。

```bash
DATA_ROOT="/你的数据目录"
WORKSPACE="/你的大容量实验目录"
DESCRIPTION="/你的数据说明.txt"

python -u agent.py init \
  --data "$DATA_ROOT" --workspace "$WORKSPACE" --domain memory \
  --description-file "$DESCRIPTION" \
  --instructions "处理内存 TypeB；根据说明识别 TypeB，使用内存官方提前量区间。" \
  --config examples/runtime.json
```

屏幕先打印 `SESSION: ...`，复制这个路径：

```bash
SESSION="刚才打印的完整会话路径"
python -u agent.py doctor --session "$SESSION" --api-check
nohup python -u agent.py run --session "$SESSION" > "$SESSION/console.log" 2>&1 < /dev/null &
tail -n 30 -f "$SESSION/console.log"
```

`init` 会调用特征 agent 生成映射；`run` 先归一化数据，再训练。关键语义不明时保存会话并明确指出缺什么，不猜测标签。修正说明后可另建会话；也支持 `prepare --session "$SESSION" --mapping 自己确认的映射.json`。映射格式见 `examples/mapping_example.json`，手工映射是可选入口，正常流程由 agent 生成。

**内存 CE 日志稀疏，不等于日志结束之后设备就健康。** 默认使用末次观测作为保守随访。若数据说明明确保证整个声明期间的故障工单完整，可在初始化加 `--complete-failure-coverage`，并在 runtime 配置设置 `coverage_start` / `coverage_end`（带时区的 ISO 时间；结束边界不含）。特征 agent 才能据此选择完整覆盖；程序不会自行假定。

## 6. 再接入硬盘或新器件

把变量换成硬盘的真实数据路径和说明，复用相同 workspace：

```bash
python -u agent.py init \
  --data "$DATA_ROOT" --workspace "$WORKSPACE" --domain hdd \
  --description-file "$DESCRIPTION" \
  --instructions "硬盘故障预测；跨季度检查列名和字段含义，预测提前天数无上限。" \
  --config examples/runtime.json
```

这会生成新的 SESSION。之后的 `doctor/run/status/export` 操作相同。光放大器等新设备用 `--domain generic`，提供该数据自己的说明，不需要改内存训练代码。

格式支持 CSV、Parquet、Feather、单工作表 XLSX、记录数组 JSON、JSONL，以及有明确命名分组正则的 log。压缩包先解压；大 JSON 数组需要转 JSONL。会检查全部文件的表头，按格式变化分别映射；缺列保留 NaN，同名不同含义的字段应分开。复合设备 ID 可用列名列表，或每设备一个文件的文件名。类别列本版默认排除并说明原因；不是全自动理解任意二进制、任意自然语言日志或任意业务语义。

原始数值观测或已有官方数值特征均可接入。多张异步表的时间对齐、原始 CE 地址展开等复杂专用变换，需要先归一化为设备时序表；本版没有通用异步多表关联器。新增特征目前作用于该数值时序接口。

## 7. 看进度、断点续跑、导出

```bash
python agent.py status --session "$SESSION"
python agent.py export --session "$SESSION"
```

导出路径会打印为 `REPORT: ..._report.zip`。报告有分析、日志和指标，**不含大模型文件和特征缓存**，不能仅靠报告压缩包恢复训练。完整会话目录和数据要留在服务器。

| 会话内路径 | 内容 |
|---|---|
| `events.jsonl` / `console.log` | 总流程与进度 |
| `calls/000xx_analysis/response.json` | 每一轮分析、动作和原因 |
| `calls/*/usage.json` | API 返回的 token 数量 |
| `mapping.json` / `inventory.json` | 数据映射、各批次格式 |
| `feature_jobs/job_*/` | 提出问题、生成代码、修复、校验结果 |
| `feature_programs/版本号/transform.py` | 可复用的新特征代码 |
| `trials/trial_*/attempt_01/result.json` | 训练配置、设备指标、运行耗时 |
| `trials/trial_*/attempt_01/loss_history.json` | 训练 loss 和验证设备指标历史 |
| `trials/trial_*/attempt_01/execution.log` | 训练内部日志；Working 是心跳，不是训练轮数 |
| `diagnostics/` | 分析请求的证据 |
| `state.json` | 最佳实验、未完成动作、剩余预算依据 |

断网恢复后，激活环境、重新加载密钥，使用同一个 SESSION：

```bash
python -u agent.py run --session "$SESSION"
```

已经完成的模型不会因分析重试而重新训练；训练中被强制中断且没有完成凭据的那一轮会重新开始。避免同时启动同一会话两份进程（程序会加锁）。升级本包代码或改变数据语义请新建会话，旧版本留着续跑旧会话。

资源已增加或希望扩大预算时，在进程停止后执行，例如：

```bash
python agent.py limits --session "$SESSION" \
  --memory-limit-gb 48 --trial-timeout-seconds 14400 \
  --max-trials 30 --max-agent-calls 240
python -u agent.py run --session "$SESSION"
```

该命令只修改资源/调用预算并记录变更，不修改数据、模型参数或评分规则。

## 8. 最终测试

优化结束、配置确定之后：

```bash
python agent.py freeze --session "$SESSION"
python -u agent.py test --session "$SESSION"
python agent.py export --session "$SESSION"
```

冻结基线和赢家各自的模型、特征算法和验证阈值，在相同测试设备队列评分。测试结果在 `test_summary.json`，不会回传给分析 agent 继续调参。冻结后禁止继续优化该会话。

## 9. 已验证范围与工程边界

运行 `python -m unittest discover -s tests -v` 可做离线回归检查。测试使用小型合成内存、混合季度硬盘数据和真实 CPU XGBoost；agent 回复使用测试替身。它证明流程及关键约束可执行，不代表真实故障预测效果。

数据批处理本地执行。发给 DeepSeek 的有数据说明、列名、训练期少量数值样本、聚合指标及诊断。若企业数据不允许这些信息外发，应先采用合规的本地模型服务并修改接口配置。特征 AST 检查是数值接口约束，不是抵御恶意代码的完整安全沙箱；实际部署宜运行在专用容器内。

为建立数据范围，程序扫描全量文件的表头、设备和时间元数据；优化期不把测试特征值/标签送给 agent 或模型。特征只使用到预测时刻可用的历史。若原始字段本来包含未来信息、或上报时间晚于测量时间，需在数据说明中声明并使用正确的可用时间，代码不能自动证明业务语义无泄漏。

本版保留独立的旧实验文件，不把此前改动过评分口径的 TypeA 分数当作新任务基线。外部旧模型自动迁移没有开启。

## 10. 参考与 skill

- CAAFE：Hollmann 等，*Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering*，NeurIPS 2023。[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html)。借鉴语义生成特征—执行—评估反馈流程，本包采用时序设备数据约束与独立三个角色。
- [SmartMem 官方 Evaluation](https://hwcloud-ras.github.io/SmartMem.github.io/)，用于内存提前量区间。
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)，用于角色调用与 JSON 输出。
- 随包 skill：`skills/device-failure-experiment-loop/SKILL.md`。控制器直接加载对应工作流，已安装的旧 skill 不会被本包自动覆盖。
