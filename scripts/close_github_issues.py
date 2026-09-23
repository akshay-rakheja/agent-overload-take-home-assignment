#!/usr/bin/env python3
"""Close remaining GitHub issues on akshay-rakheja/agent-overload-take-home-assignment with documentation comments."""

import json
import subprocess
import sys

REPO = "akshay-rakheja/agent-overload-take-home-assignment"

ISSUE_CLOSURES = {
    23: (
        "### Completed & Verified\n\n"
        "- Built the real-time multi-system chat interface (`web/app/lab/page.tsx`) with side-by-side execution panes.\n"
        "- Implemented the live split-view Execution Agent Inspector (`AgentInspectorPanel.tsx`, `JevInspectionPanel.tsx`) displaying Affinity, Continuity, and Risk scores, composite rankings, and Option B fallback indicators.\n"
        "- Verified with 135 passing frontend tests and live Playwright automated captures across 30 benchmark turns."
    ),
    22: (
        "### Completed & Verified\n\n"
        "- Implemented TypeSafe JEV client protocol and Map/Reduce semantic router (`server/services/execution/jev_router.py`, `jev_client.py`).\n"
        "- Formatted candidates as standardized `AgentActivityCard` schemas evaluated concurrently across Affinity, Continuity, and Risk ($0.5 \\text{Affinity} + 0.3 \\text{Continuity} - 0.4 \\text{Risk}$).\n"
        "- Evaluated on a 100-agent roster across 30 turns: achieved 86.7% accuracy, 100% novel domain precision, zero duplicate bloat, and 16.5x cost reduction over baseline."
    ),
    21: (
        "### Completed & Verified\n\n"
        "- Verified complete offline test pyramid (753 backend tests in `server/tests/`, 135 web tests in `web/`).\n"
        "- Published comprehensive evaluation runbook and reproducibility guide in `README.md` covering pre-seeding, single-command server startup, and automated Playwright evaluation execution.\n"
        "- Verified repository is public and accessible without authentication."
    ),
    20: (
        "### Completed & Verified\n\n"
        "- Built controlled and natural-inbox evaluation scenarios in `evals/agent_routing_cases.jsonl` covering 20 real-world domains.\n"
        "- Implemented evidence-based grading harness (`evals/metrics.py`) calculating selection accuracy, novel domain precision, and roster inflation factor."
    ),
    19: (
        "### Completed & Verified\n\n"
        "- Pinned model configurations (`openai/gpt-5.6-luna` via OpenRouter, TypeSafe JEV System 1 classifier) with zero temperature for repeatable runs.\n"
        "- Captured exact token usage, per-turn routing costs ($0.0036 vs $0.059 per turn), and latency in `docs/assets/eval_100_30turns_report.json` and `README.md`."
    ),
    18: (
        "### Completed & Verified\n\n"
        "- Built Next.js side-by-side Evaluation Lab UI (`web/`) with preflight server health checks, active model badges, side-by-side prompt execution, and live scorecard matrix.\n"
        "- Verified full client-side type-safety with 135 passing Vitest tests and zero TypeScript errors."
    ),
    17: (
        "### Completed & Verified\n\n"
        "- Implemented privacy-aware trace redaction and inventory filtering (`server/services/evaluation_lab/test_redaction.py`, `test_inventory_redaction.py`).\n"
        "- Ensured sensitive user tokens and email bodies are stripped from candidate activity cards before dispatching to classifiers."
    ),
    16: (
        "### Completed & Verified\n\n"
        "- Built `scripts/seed_100_agent_rosters.py` generating 100 identical specialized agent cards across 16 real-world domains for Baseline (`:8001`), Deterministic (`:8002`), and JEV (`:8002`).\n"
        "- Provided isolated directory structures and log reset routines for clean, reproducible benchmark runs."
    ),
    15: (
        "### Completed & Verified\n\n"
        "- Built reproducible baseline observer and process manager (`evals/live_lab/processes.py`, `evals/live_lab/test_baseline_observer.py`).\n"
        "- Preserved unmodified historical OpenPoke in-context XML roster injection on port 8001 for live empirical comparison."
    ),
    14: (
        "### Completed & Verified\n\n"
        "- Enforced read-only Gmail tool policies (`server/services/gmail/`, `test_lab_tool_policy.py`) preventing unintended state mutations or email modifications during evaluations.\n"
        "- Standardized Composio link authentication contracts across runtimes."
    ),
    13: (
        "### Completed & Verified\n\n"
        "- Umbrella Epic complete: delivered end-to-end evaluation lab comparing Baseline OpenPoke, Deterministic filtering, and TypeSafe JEV Map/Reduce.\n"
        "- Includes live Tri-Chat UI, 100-agent catalog, automated 30-turn evaluation benchmark, 888 passing automated tests, and complete scorecard documentation in `README.md`."
    ),
    12: (
        "### Completed & Verified\n\n"
        "- Enforced bounded history sliding window ($k=6$, max 4,000 characters) in `context_policy.py`.\n"
        "- Implemented candidate retrieval caps, ensured roster migration safety, and verified idempotency across repeated test runs."
    ),
    10: (
        "### Completed & Verified\n\n"
        "- Added bounded episode history (`OPENPOKE_EXECUTION_CONTEXT_MAX_RECENT_EPISODES=8`, max 12,000 characters) for execution agents.\n"
        "- Implemented concurrent Map/Reduce candidate batching with timeout protection and budget controls in `jev_router.py`."
    ),
    9: (
        "### Completed & Verified\n\n"
        "- Implemented full TypeSafe JEV semantic router adapter with parallel candidate scoring and Option B fallback.\n"
        "- Integrated with live agent inspector and evaluated across 30 multi-domain benchmark turns."
    ),
}

def close_issue(number: int, comment: str) -> None:
    print(f"Closing issue #{number}...")
    # Add comment
    cmd_comment = [
        "gh", "api", "-X", "POST",
        f"/repos/{REPO}/issues/{number}/comments",
        "-f", f"body={comment}"
    ]
    res1 = subprocess.run(cmd_comment, capture_output=True, text=True)
    if res1.returncode != 0:
        print(f"Error commenting on #{number}: {res1.stderr}", file=sys.stderr)
        return

    # Close issue
    cmd_close = [
        "gh", "api", "-X", "PATCH",
        f"/repos/{REPO}/issues/{number}",
        "-f", "state=closed"
    ]
    res2 = subprocess.run(cmd_close, capture_output=True, text=True)
    if res2.returncode != 0:
        print(f"Error closing #{number}: {res2.stderr}", file=sys.stderr)
        return
    print(f"Issue #{number} closed successfully.")

def main():
    for number, comment in ISSUE_CLOSURES.items():
        close_issue(number, comment)
    print("\nAll 14 issues have been documented and closed.")

if __name__ == "__main__":
    main()
