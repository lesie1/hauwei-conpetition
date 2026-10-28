# 设备故障预测与自动化调优 Agent

本仓库用于审查 2026-10-28 v3 框架。项目利用分析、特征和代码三个 Agent，将设备故障数据分析、特征构建、模型训练、误报/漏报诊断与实验选择连接起来。

## 给 Claude 的当前任务

**先审查、再在仓库写出修改方案；本轮不实施程序修改。** 从 [CLAUDE.md](CLAUDE.md) 开始，执行 [审查任务](docs/CLAUDE_REVIEW_TASK.md)。完成后提交 `docs/review/FINDINGS.md`、`docs/review/CHANGE_PLAN.md` 和 `docs/review/REQUIREMENTS_MATRIX.md`；不要仅在聊天里回复。

当前任务未执行，仓库中的审查模板不是已有审查结论。只有能写仓库的 Claude Code 会话才能直接完成文件提交；普通聊天只读连接不会自动产生提交。

## 阅读顺序

1. [项目要求](docs/PROJECT_REQUIREMENTS.md)：匿名化技术要求，区分来源要求与本项目验收约束。
2. [版本与证据状态](docs/CURRENT_STATUS.md)：当前源码是什么、哪些结果尚未证明。
3. [v3 实现说明](src/device_failure_agent/README_V3_中文.md)。
4. [历史运行说明](src/device_failure_agent/README.md)：旧版本背景，涉及当前行为时以 v3 源码为准。
5. [原交付验证摘要](src/device_failure_agent/验证结果.txt)与[脱敏日志](src/device_failure_agent/最终测试日志.txt)。

## 源码与离线验证

程序保留原目录结构，位于 `src/device_failure_agent/`。Python 环境中执行：

```bash
cd src/device_failure_agent
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

测试需要安装依赖并有足够 CPU/内存；它们不是实际业务收益评测。审查时先检查测试代码，再运行合成数据、模拟接口的离线测试，记录实际执行结果。真实数据、训练模型、API 密钥和远程机器访问不随仓库提供。

## 交付边界

本仓库是匿名化技术审查工作区，不代表正式提交已经完成。项目之外的行政提交事项由所有者另行管理。修改方案经所有者确认后，再新建实现分支修复并验证。
