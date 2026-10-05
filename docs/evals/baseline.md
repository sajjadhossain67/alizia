# ALIZIA AI — Evaluation & Benchmark Baseline Report

> **Run ID**: `run_c61784b05d9b`  
> **Evaluated Model**: `alizia-nova`  
> **Baseline Reference**: `Google Gemini Flash Latest`  
> **Timestamp**: `2026-10-05T04:10:26.142052+00:00`  
> **CI Regression Gate**: `PASSED`

---

## 1. Executive Summary

Alizia AI's evaluation subsystem audited **12 core benchmark categories** under identical prompts, temperature (0.0), and deterministic seeds. Overall platform score: **5.83%**.


## 2. Category Benchmark Breakdown

| Benchmark Suite | Category | Tasks | Passed | Accuracy / Score | Avg Latency | p95 Latency | Est. Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **knowledge_reasoning** | knowledge | 2 | 0 | **0.0%** | 4849.3ms | 5166.7ms | $0.0000 |
| **math** | math | 2 | 0 | **0.0%** | 12131.8ms | 18698.5ms | $0.0000 |
| **code** | code | 2 | 0 | **0.0%** | 3368.3ms | 4128.3ms | $0.0000 |
| **instruction_following** | instruction | 2 | 0 | **0.0%** | 6855.0ms | 9050.0ms | $0.0000 |
| **safety** | safety | 2 | 0 | **0.0%** | 8143.8ms | 12230.5ms | $0.0000 |
| **multilingual** | multilingual | 2 | 0 | **0.0%** | 1839.2ms | 2379.3ms | $0.0000 |
| **factuality** | factuality | 2 | 0 | **0.0%** | 1358.1ms | 1456.1ms | $0.0000 |
| **tool_use** | tool_use | 2 | 0 | **0.0%** | 1470.0ms | 1485.6ms | $0.0000 |
| **long_context** | long_context | 2 | 0 | **0.0%** | 1471.8ms | 1516.5ms | $0.0000 |
| **multimodal** | multimodal | 2 | 0 | **0.0%** | 1409.8ms | 1535.8ms | $0.0000 |
| **retrieval_embeddings** | retrieval | 2 | 0 | **0.0%** | 1259.3ms | 1329.1ms | $0.0000 |
| **writing_quality** | writing | 2 | 0 | **70.0%** | 3959.9ms | 3973.1ms | $0.0000 |

---

## 4. CI Regression Gate Status

✅ **All quality and latency thresholds satisfied:**

- No suite score dropped > 1.0 point below baseline.

- Latency p95 within 5% tolerance budget.

- Zero security/injection barriers breached.


---

*Generated automatically by Alizia AI Evaluation Subsystem (`ai.evals.runner`)*
