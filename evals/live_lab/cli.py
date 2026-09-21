"""Explicit-path CLI for preparing, hashing, and resetting lab state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .contracts import StateFingerprint, StateSnapshot
from .fixtures import (
    build_fixture_manifest,
    materialize_baseline,
    materialize_enhanced,
    serialize_manifest,
)
from .state import compare_logical_state, create_snapshot, fingerprint_state, restore_snapshot


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
