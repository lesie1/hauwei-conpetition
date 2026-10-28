---
name: device-failure-experiment-loop
description: Run evidence-driven device-failure experiments with separate false-alarm and missed-fault diagnostics, bounded feature exploration, and controlled model comparisons across device domains.
---

# Device failure experiment loop

Read [the experiment method](references/experiment-method.md) before proposing, implementing, or interpreting an experiment. The runtime injects that exact file into all three roles on every call; role-specific JSON interfaces live in prompts/.

Use the current session's scope, field descriptions, receipts and diagnostic budget. Identify whether the next experiment targets false alarms, missed faults, coverage or an interaction. Record the hypothesis, competing explanation, expected TP/FP/FN change and falsification condition.

Complete a change only when its accepted plan is executable, evidence is saved, old results remain traceable and independent evaluation boundaries remain intact. Distinguish structural compatibility across memory/HDD/generic inputs from demonstrated predictive improvement on actual datasets.

For v3 handoff and cache migration read [the package handoff](../../README_V3_中文.md). Changed program bytes require a new session; the old code seal is never rewritten to make an old run appear compatible.
