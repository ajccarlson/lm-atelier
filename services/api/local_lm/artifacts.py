from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import re
import shutil
import stat
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import IO

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from .artifact_library import begin_artifact_write_fence, referenced_artifact_ids
from .config import Settings
from .domain import ArtifactKind, MessageRole, PartType
from .filesystem_links import is_link_or_reparse
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    Message,
    MessagePart,
    ResponseRevision,
    ResponseRevisionPart,
)
from .subprocess_env import subprocess_environment

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGED_DELETION = re.compile(r"^(?P<digest>[0-9a-f]{64})\.[0-9a-f]{32}$")
_MAX_VIDEO_POSTER_BYTES = 16 * 1024 * 1024


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_nlink,
    )


@dataclass(frozen=True)
class RetentionCleanupSummary:
    marked_count: int
    pending_count: int
    removed_count: int
    reclaimed_bytes: int


@dataclass(frozen=True)
class StagedArtifactFile:
    path: Path
    media_type: str
    original_name: str

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


class ArtifactStore:
    def __init__(self, settings: Settings, *, root: Path | None = None) -> None:
        self.root = (root or settings.artifact_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._verified_files: dict[Path, tuple[int, int]] = {}

    def _destination(self, digest: str) -> Path:
        if not _SHA256.fullmatch(digest):
            raise ValueError("invalid artifact digest")
        return self.root / digest[:2] / digest[2:4] / digest

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def resolve(self, artifact: Artifact) -> Path:
        if artifact.id != f"sha256:{artifact.sha256}" or not _SHA256.fullmatch(artifact.sha256):
            raise ValueError("artifact identity is invalid")
        expected_relative = PurePosixPath(
            artifact.sha256[:2],
            artifact.sha256[2:4],
            artifact.sha256,
        )
        if artifact.relative_path != expected_relative.as_posix():
            raise ValueError("artifact path is not canonical")
        candidate = self.root.joinpath(*expected_relative.parts)
        cursor = self.root
        for part in expected_relative.parts:
            cursor /= part
            if self._is_link(cursor):
                raise ValueError("artifact path uses a filesystem link")
        path = candidate.resolve()
        if self.root not in path.parents:
            raise ValueError("artifact path escapes store")
        return path

    def verified_path(self, artifact: Artifact) -> Path:
        path = self.resolve(artifact)
        try:
            stat_result = path.stat()
        except OSError as exc:
            raise FileNotFoundError(path) from exc
        if not path.is_file() or stat_result.st_size != artifact.size_bytes:
            raise ValueError("artifact file size does not match its record")
        cached = self._verified_files.get(path)
        fingerprint = (stat_result.st_size, stat_result.st_mtime_ns)
        if cached != fingerprint:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise ValueError("artifact file checksum does not match its record")
            self._verified_files[path] = fingerprint
        return path

    def read_verified_bytes(self, artifact: Artifact, *, maximum_bytes: int) -> bytes:
        """Read one exact ordinary CAS file through its retained descriptor.

        ``verified_path`` is appropriate for a downstream path consumer, but a
        caller that is about to make an authority decision over bytes must not
        verify one pathname object and then reopen whatever occupies that name.
        This method binds the row, path, retained descriptor, digest, and byte
        cap in one synchronous operation and refuses links or multiply-linked
        files.
        """

        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("artifact read limit is invalid")
        row_snapshot = (
            artifact.id,
            artifact.sha256,
            artifact.relative_path,
            artifact.size_bytes,
        )
        if (
            type(row_snapshot[0]) is not str
            or type(row_snapshot[1]) is not str
            or type(row_snapshot[2]) is not str
            or type(row_snapshot[3]) is not int
            or row_snapshot[3] < 0
        ):
            raise ValueError("artifact record is invalid")
        path = self.resolve(artifact)
        if row_snapshot != (
            artifact.id,
            artifact.sha256,
            artifact.relative_path,
            artifact.size_bytes,
        ):
            raise ValueError("artifact record changed during verification")
        if row_snapshot[3] > maximum_bytes:
            raise ValueError("artifact exceeds the verified read limit")
        if is_link_or_reparse(path, missing="raise", unreadable="raise"):
            raise ValueError("artifact path uses a filesystem link")

        try:
            before = path.lstat()
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise FileNotFoundError(path) from exc
        except OSError as exc:
            raise ValueError("artifact file could not be opened safely") from exc

        try:
            source = os.fdopen(descriptor, "rb", closefd=True)
        except Exception:
            os.close(descriptor)
            raise
        with source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ValueError("artifact file identity is invalid")
            if _file_identity(before) != _file_identity(opened):
                raise ValueError("artifact path changed before it was opened")
            if opened.st_size != row_snapshot[3]:
                raise ValueError("artifact file size does not match its record")
            digest = hashlib.sha256()
            content = bytearray()
            while chunk := source.read(1024 * 1024):
                if len(chunk) > maximum_bytes - len(content):
                    raise ValueError("artifact exceeds the verified read limit")
                digest.update(chunk)
                content.extend(chunk)
            after_descriptor = os.fstat(source.fileno())
        try:
            after_path = path.lstat()
        except OSError as exc:
            raise ValueError("artifact path changed during verification") from exc
        if _file_identity(opened) != _file_identity(after_descriptor) or _file_identity(
            after_descriptor
        ) != _file_identity(after_path):
            raise ValueError("artifact file changed during verification")
        if len(content) != row_snapshot[3] or digest.hexdigest() != row_snapshot[1]:
            raise ValueError("artifact file checksum does not match its record")
        if row_snapshot != (
            artifact.id,
            artifact.sha256,
            artifact.relative_path,
            artifact.size_bytes,
        ):
            raise ValueError("artifact record changed during verification")
        return bytes(content)

    def delivery_metadata(self, artifact: Artifact) -> tuple[Path, str, str]:
        path = self.verified_path(artifact)
        detected = self._detect_media_type(path)
        if detected is None:
            return path, "application/octet-stream", "attachment"
        return path, detected, "inline"

    def ingest_path(
        self,
        session: Session,
        source: Path,
        *,
        kind: ArtifactKind,
        media_type: str | None = None,
        original_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Artifact:
        source = source.resolve(strict=True)
        with source.open("rb") as handle:
            return self.ingest_stream(
                session,
                handle,
                kind=kind,
                media_type=media_type or mimetypes.guess_type(source.name)[0],
                original_name=original_name or source.name,
                metadata=metadata,
            )

    def ingest_bytes(
        self,
        session: Session,
        content: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        original_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Artifact:
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as handle:
            handle.write(content)
            handle.seek(0)
            return self.ingest_stream(
                session,
                handle,
                kind=kind,
                media_type=media_type,
                original_name=original_name,
                metadata=metadata,
            )

    def ingest_stream(
        self,
        session: Session,
        source: IO[bytes],
        *,
        kind: ArtifactKind,
        media_type: str | None,
        original_name: str | None,
        metadata: dict[str, object] | None,
    ) -> Artifact:
        digest = hashlib.sha256()
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="ingest-", dir=self.root)
        size = 0
        try:
            with os.fdopen(fd, "wb") as destination:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    destination.write(chunk)
                    size += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())

            sha256 = digest.hexdigest()
            existing = session.scalar(select(Artifact).where(Artifact.sha256 == sha256))
            if existing:
                existing_path = self.resolve(existing)
                changed = False
                if not self._matches_file(existing_path, sha256, size):
                    existing_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary_name, existing_path)
                    self._remember_verified(existing_path)
                if existing.size_bytes != size:
                    existing.size_bytes = size
                    changed = True
                sanitized_name = self._safe_original_name(existing.original_name)
                if existing.original_name != sanitized_name:
                    existing.original_name = sanitized_name
                    changed = True
                sanitized_media_type = self._safe_media_type(existing.media_type)
                if existing.media_type != sanitized_media_type:
                    existing.media_type = sanitized_media_type
                    changed = True
                if existing.metadata_json.get("temporary_preview") and not (metadata or {}).get(
                    "temporary_preview"
                ):
                    existing.kind = kind.value
                    existing.media_type = self._safe_media_type(media_type or existing.media_type)
                    existing.original_name = (
                        self._safe_original_name(original_name) or existing.original_name
                    )
                    existing.metadata_json = metadata or {}
                    changed = True
                if changed:
                    session.flush()
                return existing

            destination_path = self._destination(sha256)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if self._is_link(destination_path):
                raise ValueError("artifact destination uses a filesystem link")
            if not self._matches_file(destination_path, sha256, size):
                os.replace(temporary_name, destination_path)
            self._remember_verified(destination_path)
            artifact = Artifact(
                id=f"sha256:{sha256}",
                sha256=sha256,
                kind=kind.value,
                media_type=self._safe_media_type(media_type),
                size_bytes=size,
                relative_path=self._relative(destination_path),
                original_name=self._safe_original_name(original_name),
                metadata_json=metadata or {},
            )
            session.add(artifact)
            session.flush()
            return artifact
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def export_copy(self, artifact: Artifact, destination: Path) -> Path:
        source = self.resolve(artifact)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    async def video_poster(self, artifact: Artifact) -> bytes | None:
        executable = shutil.which("ffmpeg")
        if not executable or not artifact.media_type.startswith("video/"):
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.resolve(artifact)),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=subprocess_environment(),
            )
            stdout = await self._bounded_stdout(
                process,
                maximum_bytes=_MAX_VIDEO_POSTER_BYTES,
                timeout_seconds=30,
            )
        except asyncio.CancelledError:
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except (OSError, TimeoutError, ValueError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        return stdout if process.returncode == 0 and stdout else None

    async def browser_video_proxy(self, artifact: Artifact) -> StagedArtifactFile | None:
        if artifact.media_type in {"video/mp4", "video/webm"}:
            return None
        executable = shutil.which("ffmpeg")
        if not executable:
            return None
        fd, temporary_name = tempfile.mkstemp(prefix="video-proxy-", suffix=".mp4", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_name)
        retained = False
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(self.resolve(artifact)),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(temporary),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=subprocess_environment(),
            )
            await asyncio.wait_for(process.wait(), timeout=600)
            if process.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
                return None
            retained = True
            return StagedArtifactFile(
                path=temporary,
                media_type="video/mp4",
                original_name=f"{artifact.original_name or 'video'}.proxy.mp4",
            )
        except asyncio.CancelledError:
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except (OSError, TimeoutError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        finally:
            if not retained:
                temporary.unlink(missing_ok=True)

    @staticmethod
    async def _bounded_stdout(
        process: asyncio.subprocess.Process,
        *,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> bytes:
        stdout = process.stdout
        if stdout is None:
            raise ValueError("media process did not expose output")

        async def collect() -> bytes:
            content = bytearray()
            while chunk := await stdout.read(64 * 1024):
                content.extend(chunk)
                if len(content) > maximum_bytes:
                    raise ValueError("media process output exceeded its configured limit")
            await process.wait()
            return bytes(content)

        return await asyncio.wait_for(collect(), timeout=timeout_seconds)

    def delete_temporary_preview(self, session: Session, artifact_id: str) -> bool:
        artifact = session.get(Artifact, artifact_id)
        if not artifact or not artifact.metadata_json.get("temporary_preview"):
            return False
        references = (
            session.scalar(
                select(func.count(MessagePart.id)).where(MessagePart.artifact_id == artifact_id)
            )
            or 0
        )
        references += (
            session.scalar(
                select(func.count(ResponseRevisionPart.id)).where(
                    ResponseRevisionPart.artifact_id == artifact_id
                )
            )
            or 0
        )
        if references:
            return False
        try:
            self._delete_artifact(session, artifact)
        except OSError as exc:
            if getattr(exc, "winerror", None) in {32, 33}:
                return False
            raise
        return True

    @staticmethod
    def referenced_artifact_ids(session: Session) -> set[str]:
        return referenced_artifact_ids(session)

    def cleanup_retention(
        self,
        session: Session,
        *,
        retention_days: int,
        temporary_hours: int,
        dry_run: bool,
        now: datetime | None = None,
    ) -> RetentionCleanupSummary:
        current = now or datetime.now(UTC)
        if not dry_run:
            begin_artifact_write_fence(session)
            self._recover_staged_deletions(session)
        referenced = self.referenced_artifact_ids(session)
        marked_count = 0
        pending_count = 0
        removed_count = 0
        reclaimed_bytes = 0
        for artifact in session.scalars(select(Artifact).order_by(Artifact.created_at)).all():
            metadata = dict(artifact.metadata_json)
            # A favorite pins against the automatic sweep exactly like a live
            # reference: never marked, never removed here. Explicit deletion
            # is untouched - a user deleting a favorite means it.
            if artifact.id in referenced or artifact.favorite:
                if "unreferenced_at" in metadata and not dry_run:
                    metadata.pop("unreferenced_at", None)
                    artifact.metadata_json = metadata
                continue
            temporary = bool(metadata.get("temporary_preview") or metadata.get("intermediate"))
            age = current - self._aware(artifact.created_at)
            eligible = temporary and age >= timedelta(hours=temporary_hours)
            unreferenced_at = self._metadata_datetime(metadata.get("unreferenced_at"))
            if not temporary and unreferenced_at:
                eligible = current - unreferenced_at >= timedelta(days=retention_days)
            if eligible:
                removed_count += 1
                reclaimed_bytes += artifact.size_bytes
                if not dry_run:
                    self._delete_artifact(session, artifact)
                continue
            if not temporary:
                pending_count += 1
                if not unreferenced_at:
                    marked_count += 1
                    if not dry_run:
                        metadata["unreferenced_at"] = current.isoformat()
                        artifact.metadata_json = metadata
        if not dry_run:
            session.flush()
        orphan_count, orphan_bytes = self._cleanup_orphan_files(
            session,
            current=current,
            temporary_hours=temporary_hours,
            dry_run=dry_run,
        )
        return RetentionCleanupSummary(
            marked_count=marked_count,
            pending_count=pending_count,
            removed_count=removed_count + orphan_count,
            reclaimed_bytes=reclaimed_bytes + orphan_bytes,
        )

    def delete_library_artifact(
        self,
        session: Session,
        artifact: Artifact,
    ) -> tuple[int, int, int]:
        begin_artifact_write_fence(session)
        if artifact.kind not in {ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value}:
            raise ValueError("only image and video library artifacts can be deleted directly")

        entry_id = session.scalar(
            select(ArtifactLibraryEntry.id).where(ArtifactLibraryEntry.artifact_id == artifact.id)
        )
        if entry_id:
            raise ValueError("This media item is retained by its Media Library membership.")

        linked_ids = {
            linked_id
            for key in ("poster_artifact_id", "browser_proxy_artifact_id")
            if isinstance((linked_id := artifact.metadata_json.get(key)), str)
        }
        parts = session.scalars(
            select(MessagePart).where(MessagePart.artifact_id == artifact.id)
        ).all()
        revision_parts = session.scalars(
            select(ResponseRevisionPart).where(ResponseRevisionPart.artifact_id == artifact.id)
        ).all()
        for part in parts:
            part.artifact_id = None
        for revision_part in revision_parts:
            revision_part.artifact_id = None
        session.flush()

        removed_count = 1
        reclaimed_bytes = artifact.size_bytes
        self._delete_artifact(session, artifact)

        referenced = self.referenced_artifact_ids(session)
        for linked_id in linked_ids:
            linked = session.get(Artifact, linked_id)
            if not linked or linked.id in referenced:
                continue
            removed_count += 1
            reclaimed_bytes += linked.size_bytes
            self._delete_artifact(session, linked)
        # Revision parts are internal snapshots of the same user-visible message
        # reference. Clear them as well, but do not inflate the public reference
        # count with implementation details.
        return len(parts), removed_count, reclaimed_bytes

    def generated_media_artifact_ids_for_chat(
        self,
        session: Session,
        chat_id: str,
    ) -> tuple[str, ...]:
        artifacts_by_id = {
            artifact.id: artifact
            for artifact in session.scalars(
                select(Artifact)
                .join(MessagePart, MessagePart.artifact_id == Artifact.id)
                .join(Message, Message.id == MessagePart.message_id)
                .where(
                    Message.chat_id == chat_id,
                    Message.role == MessageRole.ASSISTANT.value,
                    MessagePart.type.in_((PartType.IMAGE.value, PartType.VIDEO.value)),
                    Artifact.kind.in_((ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value)),
                )
                .order_by(Artifact.created_at)
            )
            .unique()
            .all()
        }
        revision_artifacts = session.scalars(
            select(Artifact)
            .join(
                ResponseRevisionPart,
                ResponseRevisionPart.artifact_id == Artifact.id,
            )
            .join(
                ResponseRevision,
                ResponseRevision.id == ResponseRevisionPart.response_revision_id,
            )
            .join(Message, Message.id == ResponseRevision.message_id)
            .where(
                Message.chat_id == chat_id,
                ResponseRevisionPart.type.in_((PartType.IMAGE.value, PartType.VIDEO.value)),
                Artifact.kind.in_((ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value)),
            )
            .order_by(Artifact.created_at)
        ).unique()
        for artifact in revision_artifacts:
            artifacts_by_id[artifact.id] = artifact
        return tuple(
            artifact.id
            for artifact in sorted(artifacts_by_id.values(), key=lambda item: item.created_at)
        )

    def delete_generated_media_artifacts(
        self,
        session: Session,
        artifact_ids: tuple[str, ...],
    ) -> int:
        """Delete a removed chat's now-unreferenced generated media.

        Callers snapshot the ids before deleting the chat, then flush the chat
        deletion before entering here. The canonical reference graph therefore
        protects every surviving consumer without treating the deleted chat's
        own Run and Job rows as external retention.
        """

        removed = 0
        for artifact_id in artifact_ids:
            artifact = session.get(Artifact, artifact_id)
            if not artifact or artifact.kind not in {
                ArtifactKind.IMAGE.value,
                ArtifactKind.VIDEO.value,
            }:
                continue
            if session.scalar(
                select(ArtifactLibraryEntry.id).where(
                    ArtifactLibraryEntry.artifact_id == artifact.id
                )
            ):
                continue
            if artifact.id in self.referenced_artifact_ids(session):
                continue
            _references, removed_count, _reclaimed_bytes = self.delete_library_artifact(
                session, artifact
            )
            removed += removed_count
        return removed

    def _delete_artifact(self, session: Session, artifact: Artifact) -> None:
        begin_artifact_write_fence(session)
        if artifact.id in self.referenced_artifact_ids(session):
            raise ValueError("This artifact is still retained.")
        try:
            path = self.resolve(artifact)
        except ValueError:
            # Invalid metadata must never redirect deletion to another file.
            session.delete(artifact)
            session.flush()
            return
        staged: Path | None = None
        if path.exists():
            trash = self.root / ".delete-pending"
            if self._is_link(trash):
                raise ValueError("artifact deletion staging uses a filesystem link")
            trash.mkdir(parents=True, exist_ok=True)
            if not trash.is_dir() or trash.resolve().parent != self.root:
                raise ValueError("artifact deletion staging escapes the store")
            staged = trash / f"{artifact.sha256}.{uuid.uuid4().hex}"
            os.replace(path, staged)
            self._verified_files.pop(path, None)
        try:
            session.delete(artifact)
            session.flush()
        except Exception:
            if staged is not None:
                self._restore_staged_file(staged, path)
            raise
        if staged is not None:
            self._register_staged_deletion(session, staged, path)

    def _register_staged_deletion(
        self,
        session: Session,
        staged: Path,
        original: Path,
    ) -> None:
        def finalize(_session: Session) -> None:
            with suppress(OSError):
                staged.unlink(missing_ok=True)
            self._prune_empty_parents(original)
            with suppress(OSError):
                staged.parent.rmdir()

        def restore(_session: Session) -> None:
            self._restore_staged_file(staged, original)

        event.listen(session, "after_commit", finalize, once=True)
        event.listen(session, "after_rollback", restore, once=True)

    def _restore_staged_file(self, staged: Path, original: Path) -> None:
        if not staged.exists():
            return
        original.parent.mkdir(parents=True, exist_ok=True)
        if original.exists():
            staged.unlink(missing_ok=True)
        else:
            os.replace(staged, original)
            self._remember_verified(original)

    def _recover_staged_deletions(self, session: Session) -> None:
        trash = self.root / ".delete-pending"
        if not trash.is_dir() or self._is_link(trash):
            return
        artifacts_by_sha = {
            artifact.sha256: artifact for artifact in session.scalars(select(Artifact)).all()
        }
        for staged in trash.iterdir():
            match = _STAGED_DELETION.fullmatch(staged.name)
            if not match or (not staged.is_file() and not staged.is_symlink()):
                continue
            if self._is_link(staged):
                staged.unlink(missing_ok=True)
                continue
            artifact = artifacts_by_sha.get(match.group("digest"))
            if artifact is None:
                staged.unlink(missing_ok=True)
                continue
            try:
                original = self.resolve(artifact)
            except ValueError:
                continue
            self._restore_staged_file(staged, original)
        with suppress(OSError):
            trash.rmdir()

    def _cleanup_orphan_files(
        self,
        session: Session,
        *,
        current: datetime,
        temporary_hours: int,
        dry_run: bool,
    ) -> tuple[int, int]:
        indexed = {artifact.relative_path for artifact in session.scalars(select(Artifact)).all()}
        removed_count = 0
        reclaimed_bytes = 0
        cutoff = current - timedelta(hours=temporary_hours)
        for temporary in self.root.iterdir():
            if (
                not temporary.is_file()
                or self._is_link(temporary)
                or not (
                    temporary.name.startswith("ingest-")
                    or (temporary.name.startswith("video-proxy-") and temporary.suffix == ".mp4")
                )
            ):
                continue
            stat_result = temporary.stat()
            modified = datetime.fromtimestamp(stat_result.st_mtime, UTC)
            if modified > cutoff:
                continue
            removed_count += 1
            reclaimed_bytes += stat_result.st_size
            if not dry_run:
                temporary.unlink(missing_ok=True)
        for first in self.root.iterdir():
            if (
                not re.fullmatch(r"[0-9a-f]{2}", first.name)
                or not first.is_dir()
                or self._is_link(first)
            ):
                continue
            for second in first.iterdir():
                if (
                    not re.fullmatch(r"[0-9a-f]{2}", second.name)
                    or not second.is_dir()
                    or self._is_link(second)
                ):
                    continue
                for path in second.iterdir():
                    is_restore_partial = bool(
                        re.fullmatch(
                            r"(?:[0-9a-f]{64}|\.[0-9a-f]{64}\.[^.]+)\.restore-partial",
                            path.name,
                        )
                    )
                    if is_restore_partial and path.is_file() and not self._is_link(path):
                        stat_result = path.stat()
                        modified = datetime.fromtimestamp(stat_result.st_mtime, UTC)
                        if modified <= cutoff:
                            removed_count += 1
                            reclaimed_bytes += stat_result.st_size
                            if not dry_run:
                                path.unlink(missing_ok=True)
                                self._prune_empty_parents(path)
                        continue
                    if (
                        not _SHA256.fullmatch(path.name)
                        or path.name[:2] != first.name
                        or path.name[2:4] != second.name
                        or self._is_link(path)
                        or not path.is_file()
                    ):
                        continue
                    if path.relative_to(self.root).as_posix() in indexed:
                        continue
                    stat_result = path.stat()
                    modified = datetime.fromtimestamp(stat_result.st_mtime, UTC)
                    if modified > cutoff:
                        continue
                    removed_count += 1
                    reclaimed_bytes += stat_result.st_size
                    if not dry_run:
                        path.unlink(missing_ok=True)
                        self._prune_empty_parents(path)
        return removed_count, reclaimed_bytes

    @staticmethod
    def _safe_original_name(value: str | None) -> str | None:
        if not value:
            return None
        basename = value.replace("\\", "/").rsplit("/", 1)[-1]
        basename = "".join(character for character in basename if character.isprintable()).strip()
        if basename in {"", ".", ".."}:
            return None
        return basename[:500]

    @staticmethod
    def _safe_media_type(value: str | None) -> str:
        normalized = (value or "").split(";", 1)[0].strip().lower()
        if len(normalized) <= 120 and re.fullmatch(
            r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+",
            normalized,
        ):
            return normalized
        return "application/octet-stream"

    @staticmethod
    def _is_link(path: Path) -> bool:
        return is_link_or_reparse(
            path,
            missing="assume_regular",
            unreadable="assume_link",
        )

    @staticmethod
    def _matches_file(path: Path, digest_value: str, size: int) -> bool:
        try:
            if not path.is_file() or path.stat().st_size != size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest() == digest_value
        except OSError:
            return False

    def _remember_verified(self, path: Path) -> None:
        stat_result = path.stat()
        self._verified_files[path] = (stat_result.st_size, stat_result.st_mtime_ns)

    def _prune_empty_parents(self, path: Path) -> None:
        for parent in (path.parent, path.parent.parent):
            with suppress(OSError):
                parent.rmdir()

    @staticmethod
    def _detect_media_type(path: Path) -> str | None:
        with path.open("rb") as source:
            header = source.read(4096)
        stripped = header.lstrip(b"\xef\xbb\xbf \t\r\n")
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
        if header.startswith(b"BM"):
            return "image/bmp"
        if header.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        if stripped.startswith(b"<svg") or (
            stripped.startswith(b"<?xml") and b"<svg" in stripped[:2048]
        ):
            return "image/svg+xml"
        if len(header) >= 12 and header[4:8] == b"ftyp":
            brand = header[8:12]
            if brand in {b"avif", b"avis"}:
                return "image/avif"
            if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
                return "image/heic"
            if brand == b"qt  ":
                return "video/quicktime"
            return "video/mp4"
        if header.startswith(b"\x1aE\xdf\xa3"):
            return "video/webm"
        if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
            return "video/x-msvideo"
        if header.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
            return "video/mpeg"
        return None

    @staticmethod
    def _provenance_input_ids(provenance: object) -> set[str]:
        if not isinstance(provenance, dict):
            return set()
        artifact_ids = provenance.get("input_artifact_ids")
        if not isinstance(artifact_ids, list):
            return set()
        return {artifact_id for artifact_id in artifact_ids if isinstance(artifact_id, str)}

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @classmethod
    def _metadata_datetime(cls, value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        with suppress(ValueError):
            return cls._aware(datetime.fromisoformat(value))
        return None
