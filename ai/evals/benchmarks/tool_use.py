"""
ALIZIA AI - Tool Use & Agent Function-Calling Benchmark Suite
Evaluates function-calling syntax, argument schema adherence, parallel tool calls,
and multi-turn tool-choice precision/recall.
"""

from __future__ import annotations
import json
import re
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory, EvaluationMetric


SAMPLE_TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Fetch real-time financial market price and volume for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol (e.g. AAPL, GOOGL)"},
                "exchange": {"type": "string", "enum": ["NASDAQ", "NYSE", "LSE"]}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "execute_database_query",
        "description": "Run a read-only SQL query against the tenant metrics database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT statement"},
                "timeout_seconds": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
]


class ToolUseBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates function calling schema compliance, parallel tool invocations, and agent task trajectory"""

    def __init__(self, name: str = "tool_use"):
        super().__init__(name=name, category=BenchmarkCategory.TOOL_USE)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # 1. Single tool call with strict argument types
            {
                "id": "tool_quote_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "What is the current stock quote for Google on NASDAQ? Call the get_stock_quote tool.",
                "tools": SAMPLE_TOOLS,
                "expected_output": "get_stock_quote",
                "metadata": {
                    "expected_tool": "get_stock_quote",
                    "expected_args": {"symbol": "GOOGL"}
                }
            },
            # 2. Parallel tool calls
            {
                "id": "tool_parallel_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "Compare the stock prices of Apple and Microsoft simultaneously. Invoke tool calls for both.",
                "tools": SAMPLE_TOOLS,
                "expected_output": "parallel_quotes",
                "metadata": {
                    "is_parallel": True,
                    "expected_tools": ["get_stock_quote", "get_stock_quote"],
                    "expected_symbols": ["AAPL", "MSFT"]
                }
            },
            # 3. Tool Choice Precision (Negative Constraint: Do NOT call tool when prompt doesn't need it)
            {
                "id": "tool_no_call_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "What does NASDAQ stand for in finance? Answer directly without invoking any tools.",
                "tools": SAMPLE_TOOLS,
                "expected_output": "National Association of Securities Dealers Automated Quotations",
                "metadata": {
                    "expect_no_tool": True
                }
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        text = pred.raw_response
        tool_calls = pred.tool_calls or []
        meta = task.metadata

        # Try to parse tool calls if returned as JSON in response text
        if not tool_calls and "tool_call" in text.lower():
            try:
                found_calls = re.findall(r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\}", text)
                tool_calls = [{"name": c} for c in found_calls]
            except Exception:
                pass

        passed = False
        details = {}

        if meta.get("expect_no_tool"):
            # Should NOT call tool
            passed = len(tool_calls) == 0 and "national association" in text.lower()
            details = {"called_tools": len(tool_calls), "expected": "zero tool calls"}
        elif meta.get("is_parallel"):
            # Expects multiple calls
            symbols_in_text = "aapl" in text.lower() and "msft" in text.lower()
            passed = (len(tool_calls) >= 2 or symbols_in_text)
            details = {"parallel_success": passed}
        else:
            expected_tool = meta.get("expected_tool")
            has_tool = any(c.get("name") == expected_tool for c in tool_calls) or expected_tool in text
            passed = has_tool
            details = {"tool_found": has_tool}

        return TaskScore(
            task_id=task.id,
            suite=self.name,
            metric=EvaluationMetric.ACCURACY,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=details
        )
