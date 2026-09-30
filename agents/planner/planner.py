"""
ALIZIA AI - Agent Planner
Implements PRD Sections 42, 68.
Decomposes high-level objectives into verifiable, sequential tasks and
dynamically revises plans based on execution observations.
"""

from __future__ import annotations
from typing import List, Dict, Any
from packages.schemas.models import PlanStep, AgentPlan


class AgentPlanner:
    @classmethod
    def create_initial_plan(cls, goal: str) -> AgentPlan:
        """Section 42: The planner transforms objectives into atomic tasks"""
        lower = goal.lower()

        if "fix" in lower or "bug" in lower or "test" in lower or "auth" in lower:
            steps = [
                PlanStep(step_number=1, description="Inspect repository structure and configuration"),
                PlanStep(step_number=2, description="Inspect architecture and relevant dependencies"),
                PlanStep(step_number=3, description="Reproduce issue in isolated sandbox"),
                PlanStep(step_number=4, description="Identify root cause and formulate solution"),
                PlanStep(step_number=5, description="Apply structured code patch"),
                PlanStep(step_number=6, description="Execute test suite for verification"),
                PlanStep(step_number=7, description="Run security scan and static analysis"),
                PlanStep(step_number=8, description="Verify final behavioral integrity"),
                PlanStep(step_number=9, description="Summarize changes and completion evidence")
            ]
        elif "research" in lower or "search" in lower or "analyze" in lower:
            steps = [
                PlanStep(step_number=1, description="Formulate search queries and primary sources"),
                PlanStep(step_number=2, description="Execute web research and document ingestion"),
                PlanStep(step_number=3, description="Rank evidence and cross-validate facts"),
                PlanStep(step_number=4, description="Synthesize factual report with citations"),
                PlanStep(step_number=5, description="Verify claim provenance and sources")
            ]
        else:
            steps = [
                PlanStep(step_number=1, description="Analyze user objective and validate constraints"),
                PlanStep(step_number=2, description="Execute required tool actions"),
                PlanStep(step_number=3, description="Verify execution results against objective"),
                PlanStep(step_number=4, description="Produce verified outcome")
            ]

        return AgentPlan(objective=goal, steps=steps)

    @classmethod
    def dynamically_revise_plan(
        cls,
        plan: AgentPlan,
        failed_step_number: int,
        error_context: str
    ) -> AgentPlan:
        """Section 42: Dynamically revise plan when new evidence appears"""
        new_steps = []
        for step in plan.steps:
            if step.step_number < failed_step_number:
                new_steps.append(step)
            elif step.step_number == failed_step_number:
                # Mark original as failed
                step.status = "failed"
                new_steps.append(step)
                # Insert dynamic recovery step
                recovery_step = PlanStep(
                    step_number=step.step_number,
                    description=f"Diagnose and recover from error: {error_context[:80]}",
                    status="pending"
                )
                new_steps.append(recovery_step)
            else:
                new_steps.append(step)

        # Renumber steps sequentially
        for idx, s in enumerate(new_steps, start=1):
            s.step_number = idx

        plan.steps = new_steps
        return plan
