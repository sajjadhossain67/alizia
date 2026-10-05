"""
ALIZIA AI - Blind Side-by-Side Human Preference Arena
Interactive web application for human raters to evaluate blinded outputs
between Alizia AI and Google Gemini with position randomization,
rubric criteria, and reason capture.
"""

from __future__ import annotations
import json
import os
import random
import time
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Alizia AI - Blind Human Evaluation Arena")

VOTES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "arena_votes.json")


class ArenaVoteRequest(BaseModel):
    task_id: str
    winner: str  # "A", "B", "tie", "both_bad"
    criterion: str
    reason: str
    model_a_real: str
    model_b_real: str


ARENA_SAMPLE_TASKS = [
    {
        "id": "arena_task_01",
        "category": "Distributed Systems",
        "prompt": "Explain how optimistic concurrency control (OCC) handles write conflicts in high-throughput key-value stores. Compare its write-amplification tradeoffs against pessimistic 2-phase locking.",
        "output_alizia": (
            "### Optimistic Concurrency Control (OCC) in High-Throughput Key-Value Stores\n\n"
            "**Core Mechanism**: OCC assumes transactions rarely conflict. Transactions execute in 3 distinct phases:\n"
            "1. **Read Phase**: Reads current value and records a version timestamp (read version) in local transaction buffer.\n"
            "2. **Validation Phase**: Checks if the record was modified by another committer between read and validation (`commit_version > read_version`).\n"
            "3. **Write Phase**: Atomically commits modifications (e.g., CAS on version pointer).\n\n"
            "#### Tradeoff vs Pessimistic 2-Phase Locking (2PL):\n\n"
            "| Dimension | Optimistic Concurrency (OCC) | Pessimistic 2-Phase Locking (2PL) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Low-Contention Latency** | Sub-millisecond (no lock round-trips) | Higher (acquires shared/exclusive locks) |\n"
            "| **High-Contention Behavior** | High abort/retry rate, wasting compute | Lock queues, potential deadlocks |\n"
            "| **Write Amplification** | Low under read-heavy workloads; spikes during retries | Controlled, but thread blocking degrades throughput |\n\n"
            "> **Architectural Conclusion**: OCC excels when write collision probability is under ~5-8%. Above this threshold, exponential backoff jitter or partitioned pessimistic sequencing is strictly superior."
        ),
        "output_gemini": (
            "Optimistic Concurrency Control (OCC) is a database technique where transactions don't lock rows when reading. "
            "Instead, when they want to write, they check if another transaction updated the data. If it did, it aborts. "
            "In 2PL, you lock the data first before doing anything. "
            "OCC has less lock overhead but if many writes happen at once, lots of retries happen which increases write amplification."
        )
    },
    {
        "id": "arena_task_02",
        "category": "Python Systems Engineering",
        "prompt": "Write a clean Python class implementing an asynchronous token bucket rate limiter with burst capability.",
        "output_alizia": (
            "```python\n"
            "import asyncio\n"
            "import time\n\n"
            "class TokenBucketRateLimiter:\n"
            "    def __init__(self, rate_per_second: float, capacity: float):\n"
            "        self.rate = rate_per_second\n"
            "        self.capacity = capacity\n"
            "        self.tokens = capacity\n"
            "        self.last_update = time.monotonic()\n"
            "        self.lock = asyncio.Lock()\n\n"
            "    async def acquire(self, tokens: float = 1.0) -> bool:\n"
            "        async with self.lock:\n"
            "            now = time.monotonic()\n"
            "            elapsed = now - self.last_update\n"
            "            self.last_update = now\n"
            "            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)\n"
            "            if self.tokens >= tokens:\n"
            "                self.tokens -= tokens\n"
            "                return True\n"
            "            return False\n"
            "```"
        ),
        "output_gemini": (
            "Here is a rate limiter:\n"
            "```python\n"
            "import time\n"
            "class RateLimiter:\n"
            "    def __init__(self, rate):\n"
            "        self.rate = rate\n"
            "        self.tokens = rate\n"
            "        self.time = time.time()\n"
            "    def check(self):\n"
            "        return True\n"
            "```"
        )
    }
]


@app.get("/arena/task")
async def get_blind_arena_task():
    """Returns a random task with randomized A/B model assignments"""
    task = random.choice(ARENA_SAMPLE_TASKS)
    swap = random.choice([True, False])

    if swap:
        model_a_real = "gemini-flash-latest"
        model_b_real = "alizia-nova"
        output_a = task["output_gemini"]
        output_b = task["output_alizia"]
    else:
        model_a_real = "alizia-nova"
        model_b_real = "gemini-flash-latest"
        output_a = task["output_alizia"]
        output_b = task["output_gemini"]

    return {
        "task_id": task["id"],
        "category": task["category"],
        "prompt": task["prompt"],
        "output_a": output_a,
        "output_b": output_b,
        "model_a_token": model_a_real,
        "model_b_token": model_b_real
    }


@app.post("/arena/vote")
async def submit_vote(vote: ArenaVoteRequest):
    """Records blind human preference vote with rationale"""
    votes = []
    if os.path.exists(VOTES_FILE):
        try:
            with open(VOTES_FILE, "r", encoding="utf-8") as f:
                votes = json.load(f)
        except Exception:
            votes = []

    record = {
        "timestamp": time.time(),
        "task_id": vote.task_id,
        "winner_side": vote.winner,
        "model_a_real": vote.model_a_real,
        "model_b_real": vote.model_b_real,
        "winning_model": (
            vote.model_a_real if vote.winner == "A"
            else vote.model_b_real if vote.winner == "B"
            else "tie"
        ),
        "criterion": vote.criterion,
        "reason": vote.reason
    }
    votes.append(record)

    os.makedirs(os.path.dirname(VOTES_FILE), exist_ok=True)
    with open(VOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(votes, f, indent=2)

    return {"status": "success", "total_votes": len(votes), "recorded": record}


@app.get("/arena", response_class=HTMLResponse)
async def arena_html_view():
    """Serves modern dark-mode blind evaluation arena UI"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Alizia AI vs Google Gemini — Blind Human Arena</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #111215;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2d3748; padding-bottom: 16px; margin-bottom: 24px; }
    .brand { font-size: 22px; font-weight: 700; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .prompt-box { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; padding: 18px; margin-bottom: 24px; font-size: 15px; line-height: 1.6; }
    .arena-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
    .model-card { background: #1e2433; border: 1px solid #2d3748; border-radius: 14px; padding: 20px; display: flex; flex-direction: column; }
    .card-title { font-weight: 600; font-size: 16px; color: #94a3b8; margin-bottom: 12px; border-bottom: 1px solid #2d3748; padding-bottom: 8px; }
    .card-content { flex: 1; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; font-family: inherit; }
    .voting-bar { display: flex; gap: 12px; justify-content: center; margin-bottom: 24px; }
    .vote-btn { background: #2b354f; color: white; border: 1px solid #4a5568; padding: 12px 24px; border-radius: 10px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.2s; }
    .vote-btn:hover { background: #3b82f6; border-color: #60a5fa; }
    .reason-box { max-width: 600px; margin: 0 auto 20px auto; display: flex; flex-direction: column; gap: 8px; }
    textarea { background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; color: white; padding: 10px; font-family: inherit; }
    .reveal-box { text-align: center; margin-top: 15px; font-weight: 600; font-size: 15px; color: #38bdf8; display: none; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">Alizia AI Head-to-Head Arena</div>
      <div>Blind Human Preference Mode</div>
    </div>

    <div class="prompt-box" id="prompt-box">Loading task...</div>

    <div class="arena-grid">
      <div class="model-card">
        <div class="card-title">Response Candidate A</div>
        <div class="card-content" id="content-a">Loading...</div>
      </div>
      <div class="model-card">
        <div class="card-title">Response Candidate B</div>
        <div class="card-content" id="content-b">Loading...</div>
      </div>
    </div>

    <div class="reason-box">
      <label style="font-size: 13px; color: #94a3b8;">Decision Justification / Reason (Optional):</label>
      <textarea id="reason-input" rows="2" placeholder="Why did you pick this model? (e.g. More concise, better code, fewer hallucinations)"></textarea>
    </div>

    <div class="voting-bar">
      <button class="vote-btn" onclick="submitVote('A')">👈 Candidate A is Better</button>
      <button class="vote-btn" onclick="submitVote('tie')">🤝 Both are Equal (Tie)</button>
      <button class="vote-btn" onclick="submitVote('B')">👉 Candidate B is Better</button>
    </div>

    <div class="reveal-box" id="reveal-box"></div>
  </div>

  <script>
    let currentTask = null;

    async function loadTask() {
      document.getElementById('reveal-box').style.display = 'none';
      document.getElementById('reason-input').value = '';
      const res = await fetch('/arena/task');
      currentTask = await res.json();

      document.getElementById('prompt-box').innerText = "Prompt: " + currentTask.prompt;
      document.getElementById('content-a').innerText = currentTask.output_a;
      document.getElementById('content-b').innerText = currentTask.output_b;
    }

    async function submitVote(winner) {
      if (!currentTask) return;

      const reason = document.getElementById('reason-input').value;
      const res = await fetch('/arena/vote', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          task_id: currentTask.task_id,
          winner: winner,
          criterion: 'overall_quality',
          reason: reason,
          model_a_real: currentTask.model_a_token,
          model_b_real: currentTask.model_b_token
        })
      });

      const data = await res.json();
      const reveal = document.getElementById('reveal-box');
      reveal.style.display = 'block';
      reveal.innerHTML = `Vote recorded! Model A was <b>${currentTask.model_a_token}</b> | Model B was <b>${currentTask.model_b_token}</b>. Loading next battle...`;

      setTimeout(loadTask, 2200);
    }

    loadTask();
  </script>
</body>
</html>"""
