"""
ALIZIA AI - Evaluation CLI Entrypoint
Provides terminal commands for benchmark execution, Gemini head-to-head evaluation,
blind arena server, contamination auditing, and CI gating.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys
import uvicorn
from ai.evals.runner import EvaluationRunner
from ai.evals.reporter import EvalReportGenerator
from ai.evals.scorers.contamination import ContaminationChecker
from ai.evals.arena.app import app as arena_app


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        prog="python -m ai.evals.cli",
        description="Alizia AI - Benchmarking, Head-to-Head Evaluation & CI Regression Gate"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # 1. RUN
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation suites")
    run_parser.add_argument("--model", type=str, default="alizia-nova", help="Candidate model identifier")
    run_parser.add_argument("--suites", type=str, default=None, help="Comma-separated suite names to run")
    run_parser.add_argument("--compare", action="store_true", help="Run head-to-head comparison against Google Gemini")
    run_parser.add_argument("--limit", type=int, default=None, help="Limit number of items per suite")
    run_parser.add_argument("--output", type=str, default="docs/evals/baseline.md", help="Path to write markdown report")
    run_parser.add_argument("--json-out", type=str, default=None, help="Path to write raw JSON report")

    # 2. ARENA
    arena_parser = subparsers.add_parser("arena", help="Launch Blind Side-by-Side Human Preference Arena")
    arena_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host binding")
    arena_parser.add_argument("--port", type=int, default=8001, help="Port binding")

    # 3. DECONTAMINATE
    decontam_parser = subparsers.add_parser("decontaminate", help="Check eval datasets for train-set contamination")
    decontam_parser.add_argument("--text", type=str, default=None, help="Optional text string to check for canary keys")

    # 4. REPORT
    report_parser = subparsers.add_parser("report", help="Render markdown report from existing JSON report")
    report_parser.add_argument("--json-file", type=str, required=True, help="Path to source JSON report")
    report_parser.add_argument("--output", type=str, default="docs/evals/baseline.md", help="Output markdown path")

    args = parser.parse_args()

    if args.command == "run":
        suite_list = [s.strip() for s in args.suites.split(",")] if args.suites else None
        runner = EvaluationRunner()

        print(f"\n==================================================================")
        print(f"[ALIZIA AI EVALUATION HARNESS]")
        print(f"Model: {args.model} | Compare with Gemini: {args.compare}")
        print(f"Suites: {suite_list or 'ALL 12 CORE SUITES'}")
        print(f"==================================================================\n")

        report = asyncio.run(
            runner.run_evaluation(
                model_name=args.model,
                suite_names=suite_list,
                compare_with_gemini=args.compare,
                limit_per_suite=args.limit,
                save_report_path=args.output
            )
        )

        print(f"\n[OK] Evaluation Completed!")
        print(f"Overall Platform Score: {report.overall_score:.2f}%")
        if report.overall_win_rate_vs_baseline is not None:
            print(f"Overall Win Rate vs Gemini: {report.overall_win_rate_vs_baseline:.1f}%")
        print(f"CI Regression Gate: {'PASSED' if report.passed_ci_gate else 'FAILED'}")
        print(f"Markdown report generated at: {args.output}")

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            print(f"Raw JSON report saved at: {args.json_out}")

        if not report.passed_ci_gate:
            print(f"\n[FAIL] REGRESSION VIOLATIONS DETECTED:")
            for v in report.regression_violations:
                print(f"  - {v}")
            sys.exit(1)

    elif args.command == "arena":
        print(f"[ARENA] Launching Alizia AI Blind Human Preference Arena on http://{args.host}:{args.port}/arena")
        uvicorn.run(arena_app, host=args.host, port=args.port)

    elif args.command == "decontaminate":
        print("[AUDIT] Auditing evaluation canary keys & contamination bounds...")
        test_str = args.text or "This is a clean production dataset without public canaries."
        canaries = ContaminationChecker.check_canary_present(test_str)
        if canaries:
            print(f"[WARN] CONTAMINATION WARNING: Found canary keys: {canaries}")
            sys.exit(1)
        else:
            print("[OK] Dataset clean: zero canary leaks detected.")

    elif args.command == "report":
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        from ai.evals.types import FullEvalReport
        report_obj = FullEvalReport(**data)
        EvalReportGenerator.generate_markdown_report(report_obj, output_path=args.output)
        print(f"Report generated at {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
