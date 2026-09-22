# OpenPoke Evaluation Lab: Interview Runbook and Operator Guide

## Overview

This runbook guides reviewers, interviewers, and operators through preflighting, starting, inspecting, and evaluating the **OpenPoke Evaluation Lab**. It covers both the **five-minute live walkthrough** and deeper technical explorations of the side-by-side evidence UI.

---

## 1. Prerequisites and Setup

### System Requirements
- **macOS**: Sonoma / Sequoia (Darwin arm64 or x86_64).
- **Python**: 3.11+ with `.venv` installed.
- **Node.js**: v20+ with npm dependencies installed in `web/`.
- **Ports**: `8001`, `8002`, and `3000` available on `127.0.0.1` (loopback only).
- **Keep-Awake**: macOS built-in `caffeinate` utility.

### Canonical Environment File
The Evaluation Lab mandates exactly one physical canonical `.env` file with mode `0600`:
- **Physical Path**: `/Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env`
- **Permissions**: `chmod 0600 /Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env`

Worktrees must contain only **symlinks** pointing strictly to this canonical file:
```bash
# In agent-overload-evaluation-lab:
ln -s /Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env .env

# In openpoke-evaluation-baseline:
ln -s /Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env .env
```

Verify that worktree `.env` files are ignored by git:
```bash
git check-ignore -v .env
```

> [!IMPORTANT]
> **Zero Credential Exposure Guarantee**:
> The CLI, logs, web proxy, reports, and UI never print or leak secret keys, tokens, or raw email addresses. Environment checks inspect key names and safe SHA-256 digests only.

---

## 2. One-Command Lifecycle Operations

The unified CLI `evals.live_lab.cli` controls all lab operations:

```bash
# Activate virtual environment
source .venv/bin/activate
```

### Preflight Check
Verify environment mode, required variables, symlinks, git-ignored status, and port readiness:
```bash
python -m evals.live_lab.cli preflight
```
Expected output:
```json
{
  "canonical_env_mode": "0600",
  "canonical_env_path": "/Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env",
  "canonical_env_sha256": "...",
  "errors": [],
  "missing_variables": [],
  "present_variables": ["OPENROUTER_API_KEY", "OPENPOKE_LAB_ENABLED", ...],
  "valid": true
}
```

### Start Services
Starts keep-awake (`caffeinate`), Baseline (port 8001), Enhanced (port 8002), and Next.js UI (port 3000):
```bash
python -m evals.live_lab.cli start
```

### Check Process Status
Inspect running PIDs, ports, and health statuses:
```bash
python -m evals.live_lab.cli status
```

### Stop Services
Terminates all lab processes in reverse shutdown order (`UI -> enhanced -> baseline -> keep-awake`):
```bash
python -m evals.live_lab.cli stop
```

### Offline Evaluation & Verification
Execute offline paired evaluation across all 14 scenarios and verify resulting artifacts:
```bash
LAB_DIR="$(mktemp -d)"
python -m evals.live_lab.cli evaluate --offline --output "$LAB_DIR/run-results"
python -m evals.live_lab.cli verify --artifacts "$LAB_DIR/run-results"
```

---

## 3. Five-Minute Interview Walkthrough

### Minute 0:00 – 1:00: Preflight and Launch
1. Open a terminal in `agent-overload-evaluation-lab`.
2. Run preflight to prove security invariants and environment isolation:
   ```bash
   python -m evals.live_lab.cli preflight
   ```
3. Start the lab processes:
   ```bash
   python -m evals.live_lab.cli start
   ```
4. Point out that all three servers (8001, 8002, 3000) and keep-awake are up on loopback `127.0.0.1`.

### Minute 1:00 – 2:00: Open Evidence Browser
1. In a browser, navigate to:
   ```
   http://127.0.0.1:3000/lab
   ```
2. Note the **Preflight Status Strip**:
   - Both Baseline and Enhanced systems report green checkmarks.
   - Shows active model (`openai/gpt-4.1-mini`), fixture count (100 agents), and hard budget cap ($10.00).

### Minute 2:00 – 3:30: Run Controlled Scenario Comparison
1. In the **Scenario Runner**, select `exact-instagram-security` or `duplicate-clipweaver-prevention`.
2. Click **Run Scenario**.
3. Watch the real-time execution transition through:
   - `queued` $\rightarrow$ `resetting` $\rightarrow$ `baseline_running` $\rightarrow$ `enhanced_running` $\rightarrow$ `grading` $\rightarrow$ `complete`.
4. Walk through the **Side-by-Side Evidence Columns**:
   - **Candidate Selection**:
     - *Baseline*: All 100 roster names injected into XML prompt (~18,400 characters).
     - *Enhanced*: Top candidate filtered to 1 relevant agent, strictly capped at 5 candidates (< 400 characters).
   - **Routing Decision**:
     - *Baseline*: Inferred from post-hoc journal append.
     - *Enhanced*: Deterministic confidence score, explicit authorization gate, and stable UUID dispatch.
   - **Context Bounding**:
     - *Baseline*: Unbounded lifetime log.
     - *Enhanced*: Bounded window (8 episodes, 4,000 characters) preserving raw history on disk.

### Minute 3:30 – 4:30: Inspect Layer Scorecards
1. Scroll to the **Layer Scorecard Matrix** at the bottom of each column:
   - Routing: **PASS**
   - Response: **PASS** (factual assertions verified against fixture manifest)
   - Gmail Safety: **PASS** (read-only enforcement verified)
   - Identity Continuity: **PASS** (stable UUID maintained)
   - Duplicate Prevention: **PASS**
   - Context Containment: **PASS**
2. Explain the **Availability Contract**:
   - Highlight that baseline metrics honestly report `not_applicable` or `inferred` rather than misleading zeroes.

### Minute 4:30 – 5:00: Clean Teardown
1. Return to the terminal and stop the lab:
   ```bash
   python -m evals.live_lab.cli stop
   ```
2. Confirm with `status`:
   ```bash
   python -m evals.live_lab.cli status
   ```
   All processes report `stopped`.

---

## 4. Deep-Dive Topics

### 1,000-Agent Scale Benchmark
To demonstrate performance under extreme roster sizes without running live API calls:
```bash
python -m evals.demo
```
- Baseline roster prompt size: **41,999 characters**.
- Enhanced candidate context: **1 candidate**, under **400 characters**.
- Hybrid directory benchmark: **< 50 ms p95** against a live 1,000-record production directory.

### 10,000-Entry History Depth Containment
- Lifetime execution history of 10,000 entries equates to **1,322,499 characters**.
- Enhanced context policy bounds prompt injection to **4,376 characters** (8 episodes).
- Raw logs remain byte-for-byte immutable on disk.

### Safe OAuth & Composio Integration
- When testing live Gmail capabilities, OAuth connections are initiated through the UI via a secure redirect handoff.
- Access tokens and authorization codes are handled strictly server-side; client cookies and headers are scrubbed by the Next.js proxy (`web/app/api/lab/_proxy.ts`).
- Disconnecting a Gmail account cleanly revokes local session tokens.

---

## 5. Troubleshooting & Recovery

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **Preflight fails: mode not 0600** | Canonical env permissions too open | Run `chmod 0600 /Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env` |
| **Port 8001 / 8002 / 3000 in use** | Stale processes from an earlier run | Run `python -m evals.live_lab.cli stop` or `kill -9 $(lsof -ti:8001,8002,3000)` |
| **BudgetExceeded error** | Cumulative spend reached $10 cap | Check `.lab/budget.json`; reset evaluation budget state or switch to `--offline` |
| **Symlink mismatch error** | Worktree `.env` points to wrong file | Re-create symlink: `ln -sf /Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env .env` |
