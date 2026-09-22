"""Deterministic fabricated self-envelope Gmail fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


SUBJECT_PREFIX = "[OpenPoke Interview Fixture]"
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_BANNED = re.compile(
    r"[\w.+-]+@[\w.-]+|https?://|\boauth\b|\bbearer\b|"
    r"api[_ -]?key|access[_ -]?token|client[_ -]?secret|authorization[_ -]?code",
    re.IGNORECASE,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FixtureMessage(_FrozenModel):
    template_id: str
    fact_id: str
    envelope_from: Literal["self"] = "self"
    envelope_to: Literal["self"] = "self"
    subject: str
    body: str
    response_facts: tuple[str, ...]


class ManifestFact(_FrozenModel):
    template_id: str
    subject_sha256: str
    body_sha256: str
    envelope: Literal["self"] = "self"
    response_facts: tuple[str, ...]

    @field_validator("subject_sha256", "body_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        normalized = value.casefold()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("manifest hashes must be SHA-256")
        return normalized


class FixtureFactManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    run_id: str
    facts: Mapping[str, ManifestFact]
    manifest_sha256: str

    @field_validator("facts", mode="after")
    @classmethod
    def _freeze_facts(cls, value: Mapping[str, ManifestFact]) -> Mapping[str, ManifestFact]:
        return MappingProxyType(dict(value))

    @field_serializer("facts")
    def _serialize_facts(self, value: Mapping[str, ManifestFact]) -> dict[str, ManifestFact]:
        return dict(value)


class FixtureResponseField(_FrozenModel):
    key: str
    aliases: tuple[str, ...]
    value: str


class _Template(_FrozenModel):
    template_id: str
    fact_id: str
    subject_label: str
    body: str
    response_fields: tuple[FixtureResponseField, ...]

    @property
    def response_facts(self) -> tuple[str, ...]:
        return tuple(field.value for field in self.response_fields)


def _response_field(
    key: str, value: str, *aliases: str
) -> FixtureResponseField:
    return FixtureResponseField(
        key=key,
        aliases=(key.replace("_", " "), *aliases),
        value=value,
    )


_TEMPLATES = (
    _Template(
        template_id="instagram-security",
        fact_id="SEC-7419",
        subject_label="Instagram security notice SEC-7419",
        body=(
            "Fabricated security notice SEC-7419. A sign-in was recorded at "
            "2026-09-18 04:12 UTC from Lisbon on Pixel 10. Verification phrase: indigo-orbit."
        ),
        response_fields=(
            _response_field("reference", "SEC-7419"),
            _response_field("timestamp", "2026-09-18 04:12 UTC", "sign-in time", "dated"),
            _response_field("location", "Lisbon"),
            _response_field("device", "Pixel 10"),
            _response_field("verification_phrase", "indigo-orbit"),
        ),
    ),
    _Template(
        template_id="instagram-engagement",
        fact_id="ENG-2284",
        subject_label="Instagram engagement digest ENG-2284",
        body="Fabricated engagement digest ENG-2284: Aurora Loop received 183 likes and 27 comments.",
        response_fields=(
            _response_field("reference", "ENG-2284"),
            _response_field("creator", "Aurora Loop"),
            _response_field("likes", "183 likes"),
            _response_field("comments", "27 comments"),
        ),
    ),
    _Template(
        template_id="nebulaframe-newsletter",
        fact_id="NF-3207",
        subject_label="NebulaFrame newsletter NF-3207",
        body=(
            "Fabricated NebulaFrame bulletin NF-3207. Prism Cut 2.4 releases on "
            "2026-10-07 with the Storyboard Lock feature."
        ),
        response_fields=(
            _response_field("reference", "NF-3207"),
            _response_field("release", "Prism Cut 2.4", "version"),
            _response_field("release_date", "2026-10-07"),
            _response_field("feature", "Storyboard Lock"),
        ),
    ),
    _Template(
        template_id="vidforge-receipt",
        fact_id="VF-20481",
        subject_label="VidForge receipt VF-20481",
        body=(
            "Fabricated receipt VF-20481 for Pro Render Monthly. Total CAD 47.80 "
            "on 2026-09-19."
        ),
        response_fields=(
            _response_field("reference", "VF-20481"),
            _response_field("product", "Pro Render Monthly", "item"),
            _response_field("total", "CAD 47.80", "amount"),
            _response_field("purchase_date", "2026-09-19", "date"),
        ),
    ),
    _Template(
        template_id="motionsynth-newsletter",
        fact_id="MS-8820",
        subject_label="MotionSynth newsletter MS-8820",
        body=(
            "Fabricated MotionSynth bulletin MS-8820. Temporal Layers session at "
            "2026-10-11 17:30 UTC. Reference code GLASS-52."
        ),
        response_fields=(
            _response_field("reference", "MS-8820"),
            _response_field("session", "Temporal Layers"),
            _response_field("session_time", "2026-10-11 17:30 UTC", "scheduled time"),
            _response_field("reference_code", "GLASS-52", "code"),
        ),
    ),
    _Template(
        template_id="clipweaver-invoice",
        fact_id="CW-8117",
        subject_label="ClipWeaver invoice CW-8117",
        body=(
            "Fabricated ClipWeaver invoice CW-8117 for CAD 312.40, due 2026-10-15, "
            "purchase order PO-4406."
        ),
        response_fields=(
            _response_field("reference", "CW-8117"),
            _response_field("total", "CAD 312.40", "amount"),
            _response_field("due_date", "2026-10-15"),
            _response_field("purchase_order", "PO-4406"),
        ),
    ),
    _Template(
        template_id="long-history-anchor",
        fact_id="ARC-1042",
        subject_label="Archive anchor ARC-1042",
        body=(
            "Fabricated archive anchor ARC-1042 for Cedar Comet dated 2026-08-29. "
            "Checksum prefix 9f2c7a."
        ),
        response_fields=(
            _response_field("reference", "ARC-1042"),
            _response_field("archive", "Cedar Comet", "project"),
            _response_field("date", "2026-08-29"),
            _response_field("checksum_prefix", "9f2c7a"),
        ),
    ),
    _Template(
        template_id="ambiguous-creator-notice",
        fact_id="AMB-6063",
        subject_label="Ambiguous creator notice AMB-6063",
        body=(
            "Fabricated creator notice AMB-6063 combines account-security language "
            "with an engagement-performance update. Clarification is required before routing."
        ),
        response_fields=(
            _response_field("reference", "AMB-6063"),
            _response_field("security_category", "account-security"),
            _response_field("engagement_category", "engagement-performance"),
        ),
    ),
)


def _validate_safe(value: str, *, label: str) -> None:
    if _BANNED.search(value):
        raise ValueError(f"{label} contains banned address, secret, or authenticated URL content")


def fixture_fact_ids() -> set[str]:
    return {template.fact_id for template in _TEMPLATES}


def fixture_response_facts(fact_id: str) -> tuple[str, ...]:
    for template in _TEMPLATES:
        if template.fact_id == fact_id:
            return template.response_facts
    raise KeyError(fact_id)


def fixture_response_fields(fact_id: str) -> tuple[FixtureResponseField, ...]:
    for template in _TEMPLATES:
        if template.fact_id == fact_id:
            return template.response_fields
    raise KeyError(fact_id)


def render_fixture_messages(run_id: str) -> tuple[FixtureMessage, ...]:
    """Render the eight fixed fabricated templates for one opaque run ID."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be 8-64 opaque URL-safe characters")
    _validate_safe(run_id, label="run_id")
    messages = tuple(
        FixtureMessage(
            template_id=template.template_id,
            fact_id=template.fact_id,
            subject=f"{SUBJECT_PREFIX} {run_id} {template.subject_label}",
            body=template.body,
            response_facts=template.response_facts,
        )
        for template in _TEMPLATES
    )
    rendered = "\n".join(
        f"{item.subject}\n{item.body}\n{item.envelope_from}\n{item.envelope_to}"
        for item in messages
    )
    _validate_safe(rendered, label="fixture envelope")
    if len(messages) != 8 or len({item.subject for item in messages}) != 8:
        raise RuntimeError("fixture templates must render exactly eight unique subjects")
    return messages


def _canonical_manifest_payload(
    run_id: str,
    facts: Mapping[str, ManifestFact],
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "facts": {
            key: value.model_dump(mode="json")
            for key, value in sorted(facts.items())
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_fact_manifest(messages: tuple[FixtureMessage, ...]) -> FixtureFactManifest:
    """Build a body-free, address-free manifest for pre-send verification."""

    if len(messages) != 8 or {item.fact_id for item in messages} != fixture_fact_ids():
        raise ValueError("fact manifest requires the exact eight rendered fixture messages")
    run_ids = {
        item.subject[len(SUBJECT_PREFIX) + 1 :].split(" ", 1)[0]
        for item in messages
        if item.subject.startswith(f"{SUBJECT_PREFIX} ")
    }
    if len(run_ids) != 1:
        raise ValueError("fixture messages must share one opaque run_id")
    run_id = next(iter(run_ids))
    facts = {
        item.fact_id: ManifestFact(
            template_id=item.template_id,
            subject_sha256=hashlib.sha256(item.subject.encode("utf-8")).hexdigest(),
            body_sha256=hashlib.sha256(item.body.encode("utf-8")).hexdigest(),
            response_facts=item.response_facts,
        )
        for item in messages
    }
    digest = hashlib.sha256(_canonical_manifest_payload(run_id, facts)).hexdigest()
    return FixtureFactManifest(run_id=run_id, facts=facts, manifest_sha256=digest)


def _open_directory_without_symlinks(path: Path, flags: int) -> int:
    anchor = Path(path.anchor)
    descriptor = os.open(anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise ValueError(
                    "explicit ignored .lab root ancestors must not contain a symlink"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def write_pre_send_manifest(
    destination: Path,
    messages: tuple[FixtureMessage, ...],
    *,
    allowed_root: Path,
) -> FixtureFactManifest:
    """Atomically create a manifest inside one explicit, existing ``.lab`` root."""

    path = Path(destination)
    root = Path(allowed_root)
    if root.name != ".lab" or ".." in root.parts:
        raise ValueError("allowed root must be the explicit ignored .lab directory")
    if ".." in path.parts:
        raise ValueError("pre-send manifest path traversal is forbidden")
    root_absolute = Path(os.path.abspath(root))
    path_absolute = Path(os.path.abspath(path))
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("pre-send manifest must be inside the ignored .lab root") from exc
    if not relative.parts or relative.name in {"", ".", ".."}:
        raise ValueError("pre-send manifest destination must name a file")
    manifest = build_fact_manifest(messages)
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    _validate_safe(payload, label="fact manifest")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_descriptor = _open_directory_without_symlinks(
            root_absolute, directory_flags
        )
    except FileNotFoundError as exc:
        raise ValueError("explicit ignored .lab root must already exist") from exc
    directory_descriptor = root_descriptor
    temporary_name = f".{relative.name}.{secrets.token_hex(8)}.tmp"
    try:
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(component, directory_flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise ValueError("pre-send manifest parent must not contain a symlink") from exc
            if directory_descriptor != root_descriptor:
                os.close(directory_descriptor)
            directory_descriptor = child
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(
            temporary_name, file_flags, 0o600, dir_fd=directory_descriptor
        )
        try:
            encoded = payload.encode("utf-8")
            offset = 0
            while offset < len(encoded):
                offset += os.write(file_descriptor, encoded[offset:])
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        try:
            os.link(
                temporary_name,
                relative.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        if directory_descriptor != root_descriptor:
            os.close(directory_descriptor)
        os.close(root_descriptor)
    return manifest


def render_browser_fixture_contract() -> bytes:
    messages = render_fixture_messages("browser_contract_v1")
    unsigned = {
        "schema_version": 1,
        "facts": [
            {"fact_id": message.fact_id, "content": message.body}
            for message in sorted(messages, key=lambda item: item.fact_id)
        ],
    }
    canonical_unsigned = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected = {
        **unsigned,
        "contract_sha256": hashlib.sha256(canonical_unsigned).hexdigest(),
    }
    return (
        json.dumps(
            expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def write_browser_fixture_contract(destination: Path | None = None) -> Path:
    target = (
        destination
        if destination is not None
        else Path(__file__).resolve().parents[2]
        / "web"
        / "lib"
        / "lab"
        / "controlled-fixtures.generated.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(render_browser_fixture_contract())
    return target
