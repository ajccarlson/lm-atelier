"""Durable Media Library membership and fail-closed artifact retention authority."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn, cast

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .domain import ArtifactKind, utcnow
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    MessagePart,
    MessageReference,
    ReferenceAsset,
    ReferenceSubject,
    ResponseRevisionPart,
    Run,
    SetupVerification,
    WorkStep,
)

MAX_REFERENCE_ROWS = 100_000
MAX_REFERENCE_LIST = 4_096
MAX_REFERENCE_VALUES = 100_000
MAX_REFERENCE_DEPTH = 16
REFERENCE_CORRUPT = "Stored artifact reference data is invalid."
LIBRARY_CURSOR_INVALID = "The Media Library cursor is invalid."
LIBRARY_DATA_INVALID = "Stored Media Library data is invalid."
_CURSOR_CONTEXT = b"artifact-library-page-v1"
_ENTRY_ID = re.compile(r"^libentry:sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactReferenceDataError(RuntimeError):
    pass


class ArtifactLibraryConflict(ValueError):
    pass


class ArtifactLibraryCursorError(ValueError):
    pass


class ArtifactLibraryDataError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactLibraryPageRow:
    entry: ArtifactLibraryEntry
    artifact: Artifact


@dataclass(frozen=True)
class _LibraryCursor:
    anchor_created_at: datetime
    anchor_id: str
    after_created_at: datetime
    after_id: str


def _cursor_fail() -> NoReturn:
    raise ArtifactLibraryCursorError(LIBRARY_CURSOR_INVALID)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not value or len(value) > 1_600 or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        _cursor_fail()
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:
        raise ArtifactLibraryCursorError(LIBRARY_CURSOR_INVALID) from exc
    if not hmac.compare_digest(_b64encode(decoded), value):
        _cursor_fail()
    return decoded


def _cursor_filters(
    *, kind: str | None, state: str, favorite: bool | None, query: str, limit: int
) -> dict[str, object]:
    return {
        "favorite": favorite,
        "kind": kind,
        "limit": limit,
        "query": query,
        "state": state,
    }


def _encode_cursor(
    cursor: _LibraryCursor,
    *,
    signing_key: bytes,
    filters: dict[str, object],
) -> str:
    payload = json.dumps(
        {
            "after": [cursor.after_created_at.isoformat(timespec="microseconds"), cursor.after_id],
            "anchor": [
                cursor.anchor_created_at.isoformat(timespec="microseconds"),
                cursor.anchor_id,
            ],
            "filters": filters,
            "version": 1,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    signature = hmac.new(signing_key, _CURSOR_CONTEXT + payload, hashlib.sha256).digest()
    return f"{_b64encode(payload)}.{_b64encode(signature)}"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _cursor_fail()
        result[key] = value
    return result


def _decode_position(value: object) -> tuple[datetime, str]:
    if not isinstance(value, list) or len(value) != 2:
        _cursor_fail()
    timestamp, entry_id = value
    if not isinstance(timestamp, str) or len(timestamp) > 40:
        _cursor_fail()
    if not isinstance(entry_id, str) or not _ENTRY_ID.fullmatch(entry_id):
        _cursor_fail()
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ArtifactLibraryCursorError(LIBRARY_CURSOR_INVALID) from exc
    if parsed.isoformat(timespec="microseconds") != timestamp:
        _cursor_fail()
    return parsed, entry_id


def _decode_cursor(
    token: str,
    *,
    signing_key: bytes,
    filters: dict[str, object],
) -> _LibraryCursor:
    if not token or len(token) > 2_048 or token.count(".") != 1:
        _cursor_fail()
    if type(signing_key) is not bytes or len(signing_key) != hashlib.sha256().digest_size:
        _cursor_fail()
    encoded, encoded_signature = token.split(".", 1)
    payload = _b64decode(encoded)
    if len(payload) > 1_200:
        _cursor_fail()
    signature = _b64decode(encoded_signature)
    expected = hmac.new(signing_key, _CURSOR_CONTEXT + payload, hashlib.sha256).digest()
    if len(signature) != len(expected) or not hmac.compare_digest(signature, expected):
        _cursor_fail()
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except (ArtifactLibraryCursorError, RecursionError, UnicodeError, ValueError) as exc:
        raise ArtifactLibraryCursorError(LIBRARY_CURSOR_INVALID) from exc
    if not isinstance(raw, dict) or set(raw) != {"after", "anchor", "filters", "version"}:
        _cursor_fail()
    raw_filters = raw["filters"]
    if (
        type(raw["version"]) is not int
        or raw["version"] != 1
        or not isinstance(raw_filters, dict)
        or set(raw_filters) != {"favorite", "kind", "limit", "query", "state"}
        or type(raw_filters["limit"]) is not int
        or type(raw_filters["favorite"]) not in {bool, type(None)}
        or not isinstance(raw_filters["query"], str)
        or not isinstance(raw_filters["state"], str)
        or (raw_filters["kind"] is not None and not isinstance(raw_filters["kind"], str))
        or raw_filters != filters
    ):
        _cursor_fail()
    anchor_created_at, anchor_id = _decode_position(raw["anchor"])
    after_created_at, after_id = _decode_position(raw["after"])
    if (after_created_at, after_id) > (anchor_created_at, anchor_id):
        _cursor_fail()
    return _LibraryCursor(anchor_created_at, anchor_id, after_created_at, after_id)


def _begin_library_read_snapshot(session: Session) -> None:
    connection = session.connection()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN")


def _validate_library_row(entry: ArtifactLibraryEntry, artifact: Artifact) -> None:
    valid_recovery = (
        entry.state == "visible" and entry.deleted_at is None and entry.recovery_id is None
    ) or (
        entry.state == "trashed"
        and entry.deleted_at is not None
        and isinstance(entry.recovery_id, str)
        and 1 <= len(entry.recovery_id) <= 80
    )
    if (
        not isinstance(entry.id, str)
        or not _ENTRY_ID.fullmatch(entry.id)
        or not isinstance(entry.artifact_id, str)
        or not (1 <= len(entry.artifact_id) <= 80)
        or entry.artifact_id != artifact.id
        or not isinstance(artifact.sha256, str)
        or not _SHA256.fullmatch(artifact.sha256)
        or artifact.id != f"sha256:{artifact.sha256}"
        or artifact.relative_path
        != f"{artifact.sha256[:2]}/{artifact.sha256[2:4]}/{artifact.sha256}"
        or entry.id != f"libentry:sha256:{artifact.sha256}"
        or not isinstance(entry.display_name, str)
        or not (1 <= len(entry.display_name) <= 500)
        or "\0" in entry.display_name
        or type(entry.favorite) is not bool
        or type(entry.version) is not int
        or entry.version < 1
        or not valid_recovery
        or artifact.kind not in {"image", "video"}
        or not isinstance(artifact.media_type, str)
        or artifact.media_type != artifact.media_type.strip()
        or not (1 <= len(artifact.media_type) <= 120)
        or any(ord(character) < 32 for character in artifact.media_type)
        or type(artifact.size_bytes) is not int
        or artifact.size_bytes < 1
        or not isinstance(entry.created_at, datetime)
        or not isinstance(entry.updated_at, datetime)
    ):
        raise ArtifactLibraryDataError(LIBRARY_DATA_INVALID)


def list_library_entries(
    session: Session,
    *,
    signing_key: bytes,
    limit: int,
    cursor: str | None,
    kind: str | None,
    state: str,
    favorite: bool | None,
    query: str,
) -> tuple[list[ArtifactLibraryPageRow], str | None]:
    """Read one stable, Entry-backed page without loading files or reference graphs."""

    if not 1 <= limit <= 100 or kind not in {None, "image", "video"}:
        _cursor_fail()
    if state not in {"visible", "trashed"} or type(favorite) not in {bool, type(None)}:
        _cursor_fail()
    normalized_query = query.strip().lower()
    if len(normalized_query) > 200:
        _cursor_fail()
    filters = _cursor_filters(
        kind=kind, state=state, favorite=favorite, query=normalized_query, limit=limit
    )
    decoded = _decode_cursor(cursor, signing_key=signing_key, filters=filters) if cursor else None
    _begin_library_read_snapshot(session)
    conditions = [ArtifactLibraryEntry.state == state]
    if kind is not None:
        conditions.append(Artifact.kind == kind)
    if favorite is not None:
        conditions.append(ArtifactLibraryEntry.favorite.is_(favorite))
    if normalized_query:
        conditions.append(
            func.lower(ArtifactLibraryEntry.display_name).contains(
                normalized_query, autoescape=True
            )
        )
    statement = select(ArtifactLibraryEntry, Artifact).join(
        Artifact, Artifact.id == ArtifactLibraryEntry.artifact_id
    )
    statement = statement.where(*conditions)
    if decoded is not None:
        anchor_row = session.execute(
            select(ArtifactLibraryEntry, Artifact)
            .join(Artifact, Artifact.id == ArtifactLibraryEntry.artifact_id)
            .where(
                *conditions,
                ArtifactLibraryEntry.id == decoded.anchor_id,
                ArtifactLibraryEntry.created_at == decoded.anchor_created_at,
            )
        ).one_or_none()
        if anchor_row is None:
            _cursor_fail()
        _validate_library_row(*anchor_row)
        statement = statement.where(
            or_(
                ArtifactLibraryEntry.created_at < decoded.anchor_created_at,
                and_(
                    ArtifactLibraryEntry.created_at == decoded.anchor_created_at,
                    ArtifactLibraryEntry.id <= decoded.anchor_id,
                ),
            ),
            or_(
                ArtifactLibraryEntry.created_at < decoded.after_created_at,
                and_(
                    ArtifactLibraryEntry.created_at == decoded.after_created_at,
                    ArtifactLibraryEntry.id < decoded.after_id,
                ),
            ),
        )
    rows = session.execute(
        statement.order_by(
            ArtifactLibraryEntry.created_at.desc(), ArtifactLibraryEntry.id.desc()
        ).limit(limit + 1)
    ).all()
    for entry, artifact in rows:
        _validate_library_row(entry, artifact)
    page_rows = [ArtifactLibraryPageRow(entry, artifact) for entry, artifact in rows[:limit]]
    if len(rows) <= limit or not page_rows:
        return page_rows, None
    anchor_created_at = (
        decoded.anchor_created_at if decoded is not None else page_rows[0].entry.created_at
    )
    anchor_id = decoded.anchor_id if decoded is not None else page_rows[0].entry.id
    last = page_rows[-1].entry
    next_cursor = _encode_cursor(
        _LibraryCursor(anchor_created_at, anchor_id, last.created_at, last.id),
        signing_key=signing_key,
        filters=filters,
    )
    return page_rows, next_cursor


def begin_artifact_write_fence(session: Session) -> None:
    """Take SQLite's writer reservation before proving deletion authority."""

    connection = session.connection()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")


def library_entry_id(artifact: Artifact) -> str:
    return f"libentry:sha256:{artifact.sha256}"


def ensure_library_entry(session: Session, artifact: Artifact) -> ArtifactLibraryEntry | None:
    """Publish durable image/video membership; generic ingest deliberately does not call this."""

    if artifact.kind not in {ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value}:
        return None
    display_name = (artifact.original_name or "").strip() or artifact.sha256
    statement = (
        insert(ArtifactLibraryEntry)
        .values(
            id=library_entry_id(artifact),
            artifact_id=artifact.id,
            display_name=display_name,
            favorite=artifact.favorite,
            state="visible",
            deleted_at=None,
            recovery_id=None,
            version=1,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
        .on_conflict_do_nothing(index_elements=["artifact_id"])
    )
    session.execute(statement)
    entry = session.scalar(
        select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact.id)
    )
    if entry is None or entry.id != library_entry_id(artifact):
        raise RuntimeError("Artifact library membership is inconsistent.")
    return entry


def set_library_favorite(
    session: Session, artifact: Artifact, favorite: bool
) -> ArtifactLibraryEntry:
    entry = ensure_library_entry(session, artifact)
    if entry is None:
        raise ValueError("only image and video artifacts can enter the Media Library")
    desired = bool(favorite)
    observed_version = entry.version
    changed = cast(
        CursorResult[Any],
        session.execute(
            update(ArtifactLibraryEntry)
            .where(
                ArtifactLibraryEntry.id == entry.id,
                ArtifactLibraryEntry.version == observed_version,
                ArtifactLibraryEntry.favorite != desired,
            )
            .values(favorite=desired, version=observed_version + 1, updated_at=utcnow())
        ),
    )
    session.expire(entry)
    session.refresh(entry)
    if changed.rowcount != 1 and entry.favorite != desired:
        raise ArtifactLibraryConflict("Media Library entry changed; refresh and try again.")
    session.execute(update(Artifact).where(Artifact.id == artifact.id).values(favorite=desired))
    session.expire(artifact)
    session.refresh(artifact)
    return entry


def _fail() -> NoReturn:
    raise ArtifactReferenceDataError(REFERENCE_CORRUPT)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail()
    if len(value) > MAX_REFERENCE_LIST:
        _fail()
    return value


def _ids(value: object) -> set[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_LIST:
        _fail()
    if any(not isinstance(item, str) or not item or len(item) > 80 for item in value):
        _fail()
    return set(value)


def _optional_id(value: object) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, str) or not value or len(value) > 80:
        _fail()
    return {value}


def _run_ids(value: object) -> set[str]:
    row = _mapping(value)
    found: set[str] = set()
    for key in ("input_artifact_ids", "resolved_dependency_artifact_ids"):
        if key in row:
            found.update(_ids(row[key]))
    outputs = row.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, list) or len(outputs) > MAX_REFERENCE_LIST:
            _fail()
        for output in outputs:
            item = _mapping(output)
            for key in ("artifact_id", "poster_artifact_id", "browser_proxy_artifact_id"):
                if key in item:
                    found.update(_optional_id(item[key]))
    return found


def _work_step_ids(value: object) -> set[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_LIST:
        _fail()
    found: set[str] = set()
    for raw in value:
        item = _mapping(raw)
        if "artifact_id" in item:
            found.update(_optional_id(item["artifact_id"]))
    return found


def _settings_ids(value: object) -> set[str]:
    row = _mapping(value)
    mask = row.get("mask")
    if mask is None:
        return set()
    mask_row = _mapping(mask)
    if "artifact_id" not in mask_row:
        _fail()
    return _optional_id(mask_row["artifact_id"])


_SCALAR_ARTIFACT_KEYS = {
    "artifact_id",
    "source_artifact_id",
    "result_artifact_id",
    "input_artifact_id",
    "poster_artifact_id",
    "browser_proxy_artifact_id",
}
_LIST_ARTIFACT_KEYS = {"artifact_ids", "input_artifact_ids"}


def _job_ids(value: object, *, depth: int = 0, budget: list[int] | None = None) -> set[str]:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > MAX_REFERENCE_VALUES or depth > MAX_REFERENCE_DEPTH:
        _fail()
    found: set[str] = set()
    if isinstance(value, Mapping):
        if len(value) > MAX_REFERENCE_LIST:
            _fail()
        for key, child in value.items():
            if not isinstance(key, str):
                _fail()
            if key in _SCALAR_ARTIFACT_KEYS:
                found.update(_optional_id(child))
            elif key in _LIST_ARTIFACT_KEYS:
                found.update(_ids(child))
            else:
                found.update(_job_ids(child, depth=depth + 1, budget=budget))
    elif isinstance(value, list):
        if len(value) > MAX_REFERENCE_LIST:
            _fail()
        for child in value:
            found.update(_job_ids(child, depth=depth + 1, budget=budget))
    elif value is not None and not isinstance(value, str | int | float | bool):
        _fail()
    return found


def _pending_json_reference_ids(session: Session) -> set[str]:
    """Parse every new or changed JSON reference producer before it is written."""

    found: set[str] = set()

    def retain(values: set[str]) -> None:
        found.update(values)
        if len(found) > MAX_REFERENCE_VALUES:
            _fail()

    for value in session.new.union(session.dirty):
        if isinstance(value, MessageReference):
            retain(_ids(value.artifact_ids_json or []))
        elif isinstance(value, Run):
            retain(_run_ids(value.provenance_json or {}))
            retain(_settings_ids(value.settings_json or {}))
        elif isinstance(value, WorkStep):
            retain(_work_step_ids(value.input_bindings_json or []))
            retain(_settings_ids(value.settings_json or {}))
        elif isinstance(value, Chat) and value.scope == "studio":
            retain(_optional_id(_mapping(value.origin_json).get("source_artifact_id")))
        elif isinstance(value, Job):
            retain(_job_ids(value.payload_json or {}))
            retain(_job_ids(value.result_json or {}))
        elif isinstance(value, Artifact):
            metadata = _mapping(value.metadata_json or {})
            for key in ("poster_artifact_id", "browser_proxy_artifact_id"):
                if key in metadata:
                    retain(_optional_id(metadata[key]))
    return found


def guard_artifact_reference_flush(
    session: Session,
    _flush_context: object,
    _instances: object,
) -> None:
    """Serialize JSON reference publication with deletion and refuse dangling ids."""

    referenced = _pending_json_reference_ids(session)
    deleted = {value.id for value in session.deleted if isinstance(value, Artifact)}
    if not referenced and not deleted:
        return
    begin_artifact_write_fence(session)
    if deleted & deletion_restricted_artifact_ids(session):
        raise ArtifactReferenceDataError(REFERENCE_CORRUPT)
    available = {
        value.id for value in session.new if isinstance(value, Artifact) and value.id not in deleted
    }
    available.update(
        session.scalars(select(Artifact.id).where(Artifact.id.in_(sorted(referenced)))).all()
    )
    available.difference_update(deleted)
    if referenced - available:
        raise ArtifactReferenceDataError(REFERENCE_CORRUPT)


def referenced_artifact_ids(session: Session) -> set[str]:
    """Return the complete strong-reference graph or fail closed on corrupt JSON."""

    found: set[str] = set()
    counted_tables = (
        MessagePart,
        ResponseRevisionPart,
        ReferenceSubject,
        ReferenceAsset,
        SetupVerification,
        ArtifactLibraryEntry,
        MessageReference,
        Run,
        WorkStep,
        Chat,
        Job,
    )
    row_count = 0
    for table in counted_tables:
        row_count += session.scalar(select(func.count()).select_from(table)) or 0
        if row_count > MAX_REFERENCE_ROWS:
            _fail()

    def retain(values: set[str]) -> None:
        found.update(values)
        if len(found) > MAX_REFERENCE_VALUES:
            _fail()

    # A subject cover is a replaceable selector, not a retention edge. Its
    # foreign key is deliberately SET NULL when those bytes are removed.
    direct_columns = (
        MessagePart.artifact_id,
        ResponseRevisionPart.artifact_id,
        ReferenceAsset.artifact_id,
        SetupVerification.input_artifact_id,
        ArtifactLibraryEntry.artifact_id,
    )
    for column in direct_columns:
        retain({value for value in session.scalars(select(column)) if value})

    for value in session.scalars(select(MessageReference.artifact_ids_json)):
        retain(_ids(value))
    for value in session.scalars(select(Run.provenance_json)):
        retain(_run_ids(value))
    for value in session.scalars(select(Run.settings_json)):
        retain(_settings_ids(value))
    for value in session.scalars(select(WorkStep.input_bindings_json)):
        retain(_work_step_ids(value))
    for value in session.scalars(select(WorkStep.settings_json)):
        retain(_settings_ids(value))
    for scope, value in session.execute(select(Chat.scope, Chat.origin_json)):
        if scope == "studio":
            row = _mapping(value)
            retain(_optional_id(row.get("source_artifact_id")))
    for payload, result in session.execute(select(Job.payload_json, Job.result_json)):
        retain(_job_ids(payload))
        retain(_job_ids(result))

    pending = list(found)
    visited: set[str] = set()
    while pending:
        artifact_id = pending.pop()
        if artifact_id in visited:
            continue
        visited.add(artifact_id)
        if len(visited) > MAX_REFERENCE_VALUES:
            _fail()
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            continue
        metadata = _mapping(artifact.metadata_json)
        for key in ("poster_artifact_id", "browser_proxy_artifact_id"):
            if key in metadata:
                linked = _optional_id(metadata[key])
                for linked_id in linked - found:
                    retain({linked_id})
                    pending.append(linked_id)
    return found


def deletion_restricted_artifact_ids(session: Session) -> set[str]:
    """Ids deletion must refuse.

    SET NULL pointers are clearable only when no RESTRICT relationship also
    names the same artifact. Cover/part pointers do not excuse deleting a
    ReferenceAsset or library membership row.
    """

    blocked = referenced_artifact_ids(session)
    clearable: set[str] = set()
    for column in (
        MessagePart.artifact_id,
        ResponseRevisionPart.artifact_id,
        ReferenceSubject.cover_artifact_id,
    ):
        clearable.update(value for value in session.scalars(select(column)) if value)
    restricted: set[str] = set()
    for column in (
        ReferenceAsset.artifact_id,
        ArtifactLibraryEntry.artifact_id,
    ):
        restricted.update(value for value in session.scalars(select(column)) if value)
    return blocked - (clearable - restricted)
