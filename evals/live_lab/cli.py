"""Explicit-path CLI for preparing, hashing, resetting, and orchestrating lab lifecycle and reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .consistency import verify_artifacts_consistency
from .contracts import StateFingerprint, StateSnapshot
from .environment import (
    DEFAULT_CANONICAL_ENV_PATH,
    check_environment,
)
from .fixtures import (
    build_fixture_manifest,
    materialize_baseline,
    materialize_enhanced,
    serialize_manifest,
)
from .lifecycle import create_default_lifecycle
from .offline_eval import run_offline_evaluation
from .report import (
    render_paired_run_json,
    render_paired_run_markdown,
    write_reports,
)
from .secret_scan import scan_directory
from .state import compare_logical_state, create_snapshot, fingerprint_state, restore_snapshot
from server.services.evaluation_lab.orchestrator import PairedRunResult


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _fingerprint_summary(fingerprint: StateFingerprint) -> dict[str, object]:
    return {
        "journal_bytes": fingerprint.journal_bytes,
        "journal_entries": fingerprint.journal_entries,
        "logical_identity_digest": fingerprint.logical_identity_digest,
        "raw_journal_digest": fingerprint.raw_journal_digest,
        "roster_count": fingerprint.roster_count,
        "roster_sha256": fingerprint.roster_sha256,
        "sentinel_checks": fingerprint.sentinel_checks,
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _prepare(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve(strict=False)
    if output_root in {Path("/"), Path.home().resolve(), _REPOSITORY_ROOT}:
        raise ValueError(f"unsafe output root: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_fixture_manifest(seed=args.seed, roster_size=args.size)
    baseline_dir = output_root / "baseline" / "server" / "data"
    enhanced_dir = output_root / "enhanced" / "server" / "data"
    baseline = materialize_baseline(manifest, baseline_dir)
    enhanced = materialize_enhanced(manifest, enhanced_dir)
    baseline_snapshot = create_snapshot(baseline_dir, output_root / "snapshots" / "baseline")
    enhanced_snapshot = create_snapshot(enhanced_dir, output_root / "snapshots" / "enhanced")
    equivalence = compare_logical_state(baseline, enhanced)

    import hashlib

    _print_json(
        {
            "baseline": {
                "data_dir": str(baseline_dir),
                "snapshot_dir": baseline_snapshot.snapshot_dir,
                **_fingerprint_summary(baseline),
            },
            "enhanced": {
                "data_dir": str(enhanced_dir),
                "snapshot_dir": enhanced_snapshot.snapshot_dir,
                **_fingerprint_summary(enhanced),
            },
            "equivalent": equivalence.equivalent,
            "manifest_sha256": hashlib.sha256(serialize_manifest(manifest)).hexdigest(),
            "output_root": str(output_root),
        }
    )
    return 0


def _fingerprint(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve(strict=True)
    fingerprint = fingerprint_state(data_dir)
    _print_json({"data_dir": str(data_dir), **_fingerprint_summary(fingerprint)})
    return 0


def _reset(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    snapshot_dir = Path(args.snapshot_dir).resolve(strict=True)
    before = fingerprint_state(data_dir)
    snapshot = StateSnapshot(
        snapshot_dir=str(snapshot_dir),
        fingerprint=fingerprint_state(snapshot_dir),
    )
    after = restore_snapshot(snapshot, data_dir, allowed_root=Path(args.allowed_root))
    _print_json(
        {
            "after": _fingerprint_summary(after),
            "allowed_root": str(Path(args.allowed_root).resolve(strict=True)),
            "before": _fingerprint_summary(before),
            "data_dir": str(data_dir.resolve(strict=True)),
            "snapshot_dir": str(snapshot_dir),
        }
    )
    return 0


def _preflight(args: argparse.Namespace) -> int:
    canonical = Path(args.canonical_env) if getattr(args, "canonical_env", None) else DEFAULT_CANONICAL_ENV_PATH
    enhanced_dir = _REPOSITORY_ROOT
    baseline_dir = _REPOSITORY_ROOT.parent / "openpoke-evaluation-baseline"
    worktrees = [enhanced_dir]
    if baseline_dir.exists():
        worktrees.append(baseline_dir)

    rep = check_environment(canonical_env=canonical, worktrees=worktrees)
    _print_json(rep.model_dump())
    return 0 if rep.valid else 1


def _start(args: argparse.Namespace) -> int:
    canonical = Path(args.canonical_env) if getattr(args, "canonical_env", None) else DEFAULT_CANONICAL_ENV_PATH
    enhanced_dir = _REPOSITORY_ROOT
    baseline_dir = _REPOSITORY_ROOT.parent / "openpoke-evaluation-baseline"
    worktrees = [enhanced_dir]
    if baseline_dir.exists():
        worktrees.append(baseline_dir)

    rep = check_environment(canonical_env=canonical, worktrees=worktrees)
    if not rep.valid:
        _print_json({"error": "Preflight failed", "details": rep.model_dump()})
        return 1

    runtime_dir = Path(args.runtime_dir).resolve() if getattr(args, "runtime_dir", None) else _REPOSITORY_ROOT / ".lab" / "runtime"
    lifecycle = create_default_lifecycle(base_dir=enhanced_dir, runtime_state_dir=runtime_dir)
    try:
        lifecycle.start(readiness_timeout=getattr(args, "timeout", 30.0))
        _print_json({"status": "started", "processes": lifecycle.status()})
        return 0
    except Exception as exc:
        _print_json({"error": str(exc), "status": "failed"})
        return 1


def _stop(args: argparse.Namespace) -> int:
    enhanced_dir = _REPOSITORY_ROOT
    runtime_dir = Path(args.runtime_dir).resolve() if getattr(args, "runtime_dir", None) else _REPOSITORY_ROOT / ".lab" / "runtime"
    lifecycle = create_default_lifecycle(base_dir=enhanced_dir, runtime_state_dir=runtime_dir)
    lifecycle.stop(stop_timeout=getattr(args, "timeout", 5.0))
    _print_json({"status": "stopped", "processes": lifecycle.status()})
    return 0


def _status(args: argparse.Namespace) -> int:
    enhanced_dir = _REPOSITORY_ROOT
    runtime_dir = Path(args.runtime_dir).resolve() if getattr(args, "runtime_dir", None) else _REPOSITORY_ROOT / ".lab" / "runtime"
    lifecycle = create_default_lifecycle(base_dir=enhanced_dir, runtime_state_dir=runtime_dir)
    _print_json({"processes": lifecycle.status()})
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = (args.scenario,) if getattr(args, "scenario", None) else None
    repetitions = getattr(args, "repetitions", 1) or 1
    seed = getattr(args, "seed", 42) or 42
    model = getattr(args, "model", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"

    run_result = run_offline_evaluation(
        scenario_ids=scenario_ids,
        repetitions=repetitions,
        seed=seed,
        model=model,
    )

    (output_dir / "run.json").write_text(render_paired_run_json(run_result), encoding="utf-8")
    write_reports(run_result, output_dir)

    _print_json(
        {
            "output_dir": str(output_dir),
            "pairs_count": len(run_result.pairs),
            "run_id": str(run_result.run_id),
            "status": run_result.status.value,
        }
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file).resolve(strict=True)
    output_dir = Path(args.output).resolve() if getattr(args, "output", None) else run_file.parent
    run_result = PairedRunResult.model_validate_json(run_file.read_text(encoding="utf-8"))
    json_path, md_path = write_reports(run_result, output_dir)
    _print_json(
        {
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "run_id": str(run_result.run_id),
        }
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts).resolve(strict=True)
    run_file = artifacts_dir / "run.json"
    if not run_file.exists():
        run_file = artifacts_dir / "report.json"

    consistency_rep = None
    if run_file.exists():
        run_result = PairedRunResult.model_validate_json(run_file.read_text(encoding="utf-8"))
        artifacts = {p.name: p for p in artifacts_dir.iterdir() if p.is_file()}
        consistency_rep = verify_artifacts_consistency(run=run_result, artifacts=artifacts)

    secret_failures = scan_directory(artifacts_dir, check_machine_paths=True)
    passed = (consistency_rep.valid if consistency_rep else True) and len(secret_failures) == 0

    _print_json(
        {
            "artifacts_dir": str(artifacts_dir),
            "consistency": consistency_rep.model_dump() if consistency_rep else None,
            "secret_scan_failures": len(secret_failures),
            "valid": passed,
        }
    )
    return 0 if passed else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="materialize independent runtime trees")
    prepare.add_argument("--seed", required=True, type=int)
    prepare.add_argument("--size", required=True, type=int, choices=(10, 100, 500, 1000))
    prepare.add_argument("--output-root", required=True)
    prepare.set_defaults(handler=_prepare)

    fingerprint = commands.add_parser("fingerprint", help="hash one explicit data tree")
    fingerprint.add_argument("--data-dir", required=True)
    fingerprint.set_defaults(handler=_fingerprint)

    reset = commands.add_parser("reset", help="restore one data tree from its snapshot")
    reset.add_argument("--snapshot-dir", required=True)
    reset.add_argument("--data-dir", required=True)
    reset.add_argument("--allowed-root", required=True)
    reset.set_defaults(handler=_reset)

    preflight = commands.add_parser("preflight", help="check environment and process requirements")
    preflight.add_argument("--canonical-env", required=False)
    preflight.set_defaults(handler=_preflight)

    start = commands.add_parser("start", help="start keep-awake, baseline, enhanced, and UI processes")
    start.add_argument("--canonical-env", required=False)
    start.add_argument("--runtime-dir", required=False)
    start.add_argument("--timeout", type=float, default=30.0)
    start.set_defaults(handler=_start)

    stop = commands.add_parser("stop", help="gracefully terminate all lab processes in reverse order")
    stop.add_argument("--runtime-dir", required=False)
    stop.add_argument("--timeout", type=float, default=5.0)
    stop.set_defaults(handler=_stop)

    status_cmd = commands.add_parser("status", help="probe running lab processes and ports")
    status_cmd.add_argument("--runtime-dir", required=False)
    status_cmd.set_defaults(handler=_status)

    evaluate = commands.add_parser("evaluate", help="run paired evaluation (offline or live)")
    evaluate.add_argument("--offline", action="store_true", default=True)
    evaluate.add_argument("--live", action="store_true", default=False)
    evaluate.add_argument("--scenario", required=False)
    evaluate.add_argument("--repetitions", type=int, default=1)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--model", default="openai/gpt-4.1-mini")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_evaluate)

    report_cmd = commands.add_parser("report", help="render JSON and Markdown reports from a run file")
    report_cmd.add_argument("--run-file", required=True)
    report_cmd.add_argument("--output", required=False)
    report_cmd.set_defaults(handler=_report)

    verify_cmd = commands.add_parser("verify", help="verify cross-artifact consistency and scan for secrets")
    verify_cmd.add_argument("--artifacts", required=True)
    verify_cmd.set_defaults(handler=_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
