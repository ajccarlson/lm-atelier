from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from .config import Settings
from .filesystem_links import is_link_or_reparse
from .network import shared_tls_context
from .progress import reduce_progress
from .runtime_config import persist_runtime_values
from .schemas import ProgressV2, RuntimeStatus
from .subprocess_env import subprocess_environment

logger = logging.getLogger(__name__)

RuntimeName = Literal["llama.cpp", "vllm", "comfyui"]
RUNTIME_NAMES: tuple[RuntimeName, ...] = ("llama.cpp", "vllm", "comfyui")
_MANAGED_MARKER = ".lm-atelier-runtime.json"
_RUNTIME_PROBE_SENTINEL = "LM_ATELIER_RUNTIME_PROBE:"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RUNTIME_FILES = 250_000
_MAX_RUNTIME_BYTES = 32 * 1024**3
_REGISTRY_NODE_PREFIX = "lm-atelier-registry_"
_MANAGED_NODE_PREFIXES = ("lm-atelier-node_", _REGISTRY_NODE_PREFIX)
_MUTABLE_RUNTIME_DIRECTORIES = {
    "__pycache__",
    "custom_nodes",
    "input",
    "output",
    "temp",
    "user",
}


class RuntimeProvisioningError(RuntimeError):
    pass


class RuntimeVerificationCancelled(RuntimeError):
    pass


def default_engine_manifest_path() -> Path:
    candidates: list[Path] = []
    if bundle_root := getattr(sys, "_MEIPASS", None):
        candidates.append(Path(str(bundle_root)) / "engines.json")
    candidates.append(Path(__file__).resolve().parents[3] / "packaging" / "engines.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeProvisioningError("The inference runtime manifest is missing.")


class RuntimeProvisioner:
    """Install pinned external inference runtimes into the application data folder."""

    def __init__(
        self,
        settings: Settings,
        *,
        manifest_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
        environment: MutableMapping[str, str] | None = None,
        platform_key: str | None = None,
        allowed_download_hosts: set[str] | None = None,
    ) -> None:
        self.settings = settings
        self.manifest_path = manifest_path or default_engine_manifest_path()
        self.environment = environment if environment is not None else os.environ
        self.platform_key_override = platform_key
        if platform_key:
            self._platform_keys = {name: platform_key for name in RUNTIME_NAMES}
        else:
            nvidia = self._nvidia_runtime_info()
            self._platform_keys = {
                name: self._platform_key_for(name, nvidia) for name in RUNTIME_NAMES
            }
        self.allowed_download_hosts = allowed_download_hosts or {
            "github.com",
            "www.python.org",
        }
        self._manifest = self._read_manifest(self.manifest_path)
        self._client = client or httpx.AsyncClient(
            follow_redirects=True,
            verify=shared_tls_context(),
            timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),
        )
        self._owns_client = client is None
        self._locks = {name: asyncio.Lock() for name in RUNTIME_NAMES}
        self._tasks: dict[RuntimeName, asyncio.Task[RuntimeStatus]] = {}
        self._restore_task: asyncio.Task[None] | None = None
        self._restore_cancel = threading.Event()
        self._states: dict[RuntimeName, RuntimeStatus] = {}
        self._integrity_cache: dict[
            Path,
            tuple[str, tuple[tuple[str, int, int, int, int], ...]],
        ] = {}
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._managed_node_source_releases = self._configured_managed_comfy_releases()
        self._restore_candidates = self._managed_restoration_candidates()
        for engine in self._restore_candidates:
            definition = self._definition(engine)
            asset = self._asset(engine, definition)
            self._states[engine] = self._status(
                engine,
                definition,
                state="installing",
                supported=True,
                managed=True,
                size_bytes=int(asset["size_bytes"]) if asset else None,
                message="Checking managed runtime integrity.",
                asset=asset,
            )
        self._cleanup_completed_archives()

    def start_restore(self) -> asyncio.Task[None] | None:
        """Verify managed runtimes without delaying the API serving boundary."""

        if not self._restore_candidates:
            return None
        if self._restore_task is None:
            self._restore_cancel.clear()
            self._restore_task = asyncio.create_task(
                self._restore_managed_installations(),
                name="verify-managed-runtimes",
            )
        return self._restore_task

    @property
    def runtime_root(self) -> Path:
        return self.settings.data_dir / "runtimes"

    @property
    def archive_root(self) -> Path:
        return self.settings.download_dir / "runtimes"

    def statuses(self) -> list[RuntimeStatus]:
        return [self.status(name) for name in RUNTIME_NAMES]

    def status(self, engine: RuntimeName) -> RuntimeStatus:
        active = self._states.get(engine)
        if active is not None:
            return active
        return self.verify_status(engine)

    def verify_status(self, engine: RuntimeName) -> RuntimeStatus:
        """Refresh one runtime from disk.

        Managed runtimes are verified during installation and again when a new
        application process starts. Normal status polling must use the verified
        in-process state: a ComfyUI tree contains tens of thousands of immutable
        files, and walking it for every setup poll can take longer than the poll
        interval.
        """

        definition = self._definition(engine)
        configured = self._configured_status(engine, definition)
        if configured:
            self._states[engine] = configured
            return configured
        if self._security_blocked(definition):
            blocked = self._status(
                engine,
                definition,
                state="unsupported",
                supported=False,
                message=self._security_message(definition),
            )
            self._states[engine] = blocked
            return blocked
        asset = self._asset(engine, definition)
        if self._security_blocked(definition, asset):
            blocked = self._status(
                engine,
                definition,
                state="unsupported",
                supported=False,
                message=self._security_message(definition, asset),
                asset=asset,
            )
            self._states[engine] = blocked
            return blocked
        active = self._states.get(engine)
        if active and active.state in {"installing", "failed"}:
            return active
        if not asset:
            return self._status(
                engine,
                definition,
                state="unsupported",
                supported=False,
                message="Automatic setup is not available for this machine.",
            )
        return self._status(
            engine,
            definition,
            state="missing",
            supported=True,
            size_bytes=asset["size_bytes"],
            message="Installs automatically when first used.",
            asset=asset,
        )

    def start(self, engine: RuntimeName) -> RuntimeStatus:
        self._definition(engine)
        current = self.status(engine)
        if current.state in {"ready", "unsupported", "installing"}:
            return current
        self._states[engine] = current.model_copy(
            update={
                "state": "installing",
                "progress": 0,
                "progress_json": ProgressV2.model_validate(
                    reduce_progress(
                        current.progress_json.model_dump(mode="json")
                        if current.progress_json
                        else None,
                        stage="preparing download",
                        queue_resource="network_transfer",
                        indeterminate=True,
                    )
                ),
                "message": "Preparing download.",
            }
        )
        task = asyncio.create_task(
            self.provision(engine),
            name=f"provision-{engine.replace('.', '-')}",
        )
        self._tasks[engine] = task

        def consume_result(completed: asyncio.Task[RuntimeStatus]) -> None:
            self._tasks.pop(engine, None)
            with suppress(asyncio.CancelledError, RuntimeProvisioningError):
                completed.result()

        task.add_done_callback(consume_result)
        return self._states[engine]

    async def ensure(self, engine: RuntimeName) -> RuntimeStatus:
        current = self.status(engine)
        if current.state == "installing" and self._restore_task is not None:
            await asyncio.shield(self._restore_task)
            current = self.status(engine)
        if current.state == "ready":
            return current
        existing = self._tasks.get(engine)
        if existing and existing is not asyncio.current_task():
            return await existing
        return await self.provision(engine)

    async def provision(self, engine: RuntimeName) -> RuntimeStatus:
        definition = self._definition(engine)
        async with self._locks[engine]:
            configured = self._configured_status(engine, definition)
            if configured:
                self._states[engine] = configured
                return configured
            if self._security_blocked(definition):
                status = self._status(
                    engine,
                    definition,
                    state="unsupported",
                    supported=False,
                    message=self._security_message(definition),
                )
                self._states[engine] = status
                raise RuntimeProvisioningError(status.message)
            asset = self._asset(engine, definition)
            if self._security_blocked(definition, asset):
                status = self._status(
                    engine,
                    definition,
                    state="unsupported",
                    supported=False,
                    message=self._security_message(definition, asset),
                    asset=asset,
                )
                self._states[engine] = status
                raise RuntimeProvisioningError(status.message)
            if not asset:
                status = self._status(
                    engine,
                    definition,
                    state="unsupported",
                    supported=False,
                    message="Automatic setup is not available for this machine.",
                )
                self._states[engine] = status
                raise RuntimeProvisioningError(status.message)
            self._states[engine] = self._status(
                engine,
                definition,
                state="installing",
                supported=True,
                size_bytes=asset["size_bytes"],
                message="Preparing download.",
                asset=asset,
            )
            try:
                self._check_disk_space(asset)
                archive = await self._download(engine, definition, asset)
                overlays = [
                    (overlay, await self._download(engine, definition, overlay))
                    for overlay in self._security_overlays(asset)
                ]
                installed = await asyncio.to_thread(
                    self._install_archive,
                    engine,
                    definition,
                    asset,
                    archive,
                    overlays,
                )
                self._apply_configuration(engine, installed, persist=True)
                configured_status = self._status(
                    engine,
                    definition,
                    state="ready",
                    supported=True,
                    managed=True,
                    progress=1,
                    message="Ready.",
                    asset=asset,
                )
                self._remove_completed_archive(archive)
                for _overlay, overlay_archive in overlays:
                    self._remove_completed_archive(overlay_archive)
                self._states[engine] = configured_status
                return configured_status
            except asyncio.CancelledError:
                self._states[engine] = self._status(
                    engine,
                    definition,
                    state="missing",
                    supported=True,
                    size_bytes=asset["size_bytes"],
                    message="Download paused; retry to resume.",
                    asset=asset,
                )
                raise
            except Exception as exc:
                detail = str(exc).strip() or f"{engine} setup failed"
                self._states[engine] = self._status(
                    engine,
                    definition,
                    state="failed",
                    supported=True,
                    size_bytes=asset["size_bytes"],
                    message=detail,
                    asset=asset,
                )
                if isinstance(exc, RuntimeProvisioningError):
                    raise
                raise RuntimeProvisioningError(detail) from exc

    async def close(self) -> None:
        if self._restore_task is not None:
            if not self._restore_task.done():
                self._restore_cancel.set()
            await asyncio.gather(self._restore_task, return_exceptions=True)
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._owns_client:
            await self._client.aclose()

    async def _download(
        self,
        engine: RuntimeName,
        definition: dict[str, Any],
        asset: dict[str, Any],
    ) -> Path:
        url = str(asset["url"])
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.hostname not in self.allowed_download_hosts
        ):
            raise RuntimeProvisioningError("The runtime download URL is not trusted.")
        filename = PurePosixPath(parsed.path).name
        if not filename or filename in {".", ".."}:
            raise RuntimeProvisioningError("The runtime archive name is invalid.")
        archive = self._archive_path(engine, definition, asset)
        partial = archive.with_name(f"{archive.name}.part")
        expected_size = int(asset["size_bytes"])
        expected_hash = str(asset["sha256"])

        if archive.is_file():
            if (
                archive.stat().st_size == expected_size
                and await asyncio.to_thread(self._sha256_file, archive) == expected_hash
            ):
                return archive
            archive.unlink()

        starting_size = partial.stat().st_size if partial.is_file() else 0
        if starting_size > expected_size:
            partial.unlink()
            starting_size = 0
        headers = {"Range": f"bytes={starting_size}-"} if starting_size else {}
        async with self._client.stream("GET", url, headers=headers) as response:
            if starting_size and response.status_code == 206:
                content_range = response.headers.get("content-range", "")
                if not content_range.startswith(f"bytes {starting_size}-"):
                    raise RuntimeProvisioningError(
                        "The runtime server returned an invalid resume range."
                    )
                mode = "ab"
                downloaded = starting_size
            elif response.status_code == 200:
                mode = "wb"
                downloaded = 0
            elif response.status_code == 416 and starting_size == expected_size:
                mode = ""
                downloaded = starting_size
            else:
                response.raise_for_status()
                raise RuntimeProvisioningError(
                    f"The runtime server returned HTTP {response.status_code}."
                )

            if mode:
                with partial.open(mode) as destination:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        if not chunk:
                            continue
                        destination.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise RuntimeProvisioningError(
                                "The runtime download exceeded its pinned size."
                            )
                        self._set_download_progress(
                            engine,
                            definition,
                            expected_size,
                            downloaded,
                        )
                    destination.flush()
                    os.fsync(destination.fileno())

        if downloaded != expected_size:
            raise RuntimeProvisioningError(
                f"The runtime download stopped at {downloaded} of {expected_size} bytes; "
                "retry to resume."
            )
        actual_hash = await asyncio.to_thread(self._sha256_file, partial)
        if actual_hash != expected_hash:
            partial.unlink(missing_ok=True)
            raise RuntimeProvisioningError("The runtime archive failed SHA-256 verification.")
        os.replace(partial, archive)
        self._states[engine] = self._status(
            engine,
            definition,
            state="installing",
            supported=True,
            progress=1,
            progress_json=reduce_progress(
                self._runtime_progress(engine),
                stage="installing verified runtime",
                completed_units=expected_size,
                total_units=expected_size,
                unit="bytes",
                queue_resource="disk",
                indeterminate=True,
            ),
            downloaded_bytes=expected_size,
            size_bytes=expected_size,
            message="Installing verified runtime.",
        )
        return archive

    def _install_archive(
        self,
        engine: RuntimeName,
        definition: dict[str, Any],
        asset: dict[str, Any],
        archive: Path,
        overlays: list[tuple[dict[str, Any], Path]],
    ) -> dict[str, Path]:
        release = self._safe_component(str(definition["pinned_release"]))
        parent = self.runtime_root / self._safe_component(engine)
        final = parent / release
        staging = parent / f".{release}.partial-{uuid4().hex}"
        parent.mkdir(parents=True, exist_ok=True)
        self._prune_stale_staging(parent, engine)
        staging.mkdir()
        try:
            self._set_install_progress(
                engine,
                definition,
                asset,
                stage="extracting verified runtime",
                message="Extracting verified runtime.",
            )
            archive_type = str(asset["archive_type"])
            if archive_type == "zip":
                self._extract_zip(archive, staging)
            elif archive_type == "tar.gz":
                self._extract_tar(archive, staging)
            elif archive_type == "7z":
                self._extract_7z(
                    archive,
                    staging,
                    max_entries=int(asset.get("max_entries", 200_000)),
                    max_uncompressed_bytes=int(asset.get("max_uncompressed_bytes", 32 * 1024**3)),
                )
            else:
                raise RuntimeProvisioningError(f"Unsupported runtime archive type: {archive_type}")
            self._set_install_progress(
                engine,
                definition,
                asset,
                stage="applying runtime security updates",
                message="Applying runtime security updates.",
            )
            for overlay, overlay_archive in overlays:
                self._apply_security_overlay(staging, overlay, overlay_archive)
            if engine == "comfyui":
                self._restore_managed_registry_nodes(staging, asset, final)
            self._set_install_progress(
                engine,
                definition,
                asset,
                stage="validating runtime contract",
                message="Validating runtime contract.",
            )
            installed = self._resolve_installed_paths(staging, asset)
            self._verify_runtime_contract(staging, asset, installed)
            executable = installed["executable"]
            if os.name != "nt":
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            file_hashes = self._runtime_file_hashes(
                staging,
                asset,
                progress=lambda completed, total: self._set_install_progress(
                    engine,
                    definition,
                    asset,
                    stage="recording runtime integrity",
                    message="Recording runtime integrity.",
                    completed_units=completed,
                    total_units=total,
                ),
            )
            marker = {
                "schema_version": 3,
                "engine": engine,
                "release": definition["pinned_release"],
                "asset": PurePosixPath(urlparse(str(asset["url"])).path).name,
                "sha256": asset["sha256"],
                "distribution": definition["distribution"],
                "license": definition["license"],
                "overlays": self._overlay_marker(asset),
                "runtime_contract_sha256": self._runtime_contract_sha256(asset),
                "files": file_hashes,
            }
            (staging / _MANAGED_MARKER).write_text(
                json.dumps(marker, indent=2) + "\n",
                encoding="utf-8",
            )
            if final.exists():
                if not self._managed_marker_owned(final, engine, definition):
                    raise RuntimeProvisioningError(
                        f"Refusing to replace an unmanaged runtime directory: {final}"
                    )
                shutil.rmtree(final)
            os.replace(staging, final)
            return self._resolve_installed_paths(final, asset)
        finally:
            self._discard_staging(staging, engine)

    def _restore_managed_registry_nodes(
        self,
        staging: Path,
        asset: Mapping[str, Any],
        final: Path,
    ) -> None:
        """Carry application-owned Registry nodes into a replacement runtime."""

        raw_directory = asset.get("directory")
        if not isinstance(raw_directory, str):
            raise RuntimeProvisioningError("The ComfyUI runtime directory is invalid.")
        directory = self._safe_archive_path(raw_directory)
        target = staging.joinpath(*directory.parts, "custom_nodes")
        self._ensure_inside(staging, target.resolve())
        target.mkdir(parents=True, exist_ok=True)

        restored: set[str] = set()
        copied_entries = 0
        copied_bytes = 0
        for source in self._managed_registry_node_sources(directory, final):
            if not source.exists():
                continue
            if source.resolve() == target.resolve():
                continue
            if is_link_or_reparse(source, missing="assume_link", unreadable="assume_link"):
                raise RuntimeProvisioningError(
                    "A managed Registry node source is not an ordinary directory."
                )
            if not source.is_dir():
                raise RuntimeProvisioningError(
                    "A managed Registry node source is not an ordinary directory."
                )
            for candidate in sorted(source.iterdir(), key=lambda item: item.name.casefold()):
                if not candidate.name.startswith(_MANAGED_NODE_PREFIXES):
                    continue
                if candidate.name in restored:
                    continue
                destination = target / candidate.name
                if destination.exists():
                    added_entries, added_bytes, source_sha256 = self._managed_node_tree_identity(
                        candidate,
                        max_entries=_MAX_RUNTIME_FILES - copied_entries,
                        max_bytes=_MAX_RUNTIME_BYTES - copied_bytes,
                    )
                    destination_identity = self._managed_node_tree_identity(
                        destination,
                        max_entries=added_entries,
                        max_bytes=added_bytes,
                    )
                    if destination_identity != (
                        added_entries,
                        added_bytes,
                        source_sha256,
                    ):
                        raise RuntimeProvisioningError(
                            "A managed custom node differs between runtime releases."
                        )
                else:
                    restoring = target / f".lm-atelier-restoring-{uuid4().hex}"
                    try:
                        added_entries, added_bytes = self._copy_managed_registry_tree(
                            candidate,
                            restoring,
                            max_entries=_MAX_RUNTIME_FILES - copied_entries,
                            max_bytes=_MAX_RUNTIME_BYTES - copied_bytes,
                        )
                        if destination.exists():
                            raise RuntimeProvisioningError(
                                "The replacement runtime already contains a managed custom node."
                            )
                        os.replace(restoring, destination)
                    finally:
                        shutil.rmtree(restoring, ignore_errors=True)
                copied_entries += added_entries
                copied_bytes += added_bytes
                restored.add(candidate.name)

    def _managed_registry_node_sources(
        self,
        directory: PurePosixPath,
        final: Path,
    ) -> tuple[Path, ...]:
        releases = list(self._managed_node_source_releases)
        if final.exists():
            releases.append(final)

        sources: list[Path] = []
        seen: set[Path] = set()
        for release in releases:
            if release in seen or not self._managed_comfy_release_owned(release):
                continue
            seen.add(release)
            sources.append(release.joinpath(*directory.parts, "custom_nodes"))
        return tuple(sources)

    def _configured_managed_comfy_releases(self) -> tuple[Path, ...]:
        """Snapshot the configured managed release before background restore."""

        configured = self.settings.comfy_directory
        if configured is None:
            return ()
        engine_root = self.runtime_root / "comfyui"
        configured_path = Path(configured).expanduser()
        for parent in (configured_path, *configured_path.parents):
            if parent.parent == engine_root:
                return (parent,)
        return ()

    def _managed_comfy_release_owned(self, release: Path) -> bool:
        try:
            engine_root = (self.runtime_root / "comfyui").resolve(strict=True)
            if is_link_or_reparse(release, missing="assume_link", unreadable="assume_link"):
                return False
            owned = release.resolve(strict=True)
            if owned.parent != engine_root:
                return False
            marker = json.loads((owned / _MANAGED_MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(
            isinstance(marker, dict)
            and marker.get("schema_version") in {1, 2, 3}
            and marker.get("engine") == "comfyui"
            and isinstance(marker.get("release"), str)
            and owned.name == self._safe_component(str(marker["release"]))
        )

    @staticmethod
    def _copy_managed_registry_tree(
        source: Path,
        destination: Path,
        *,
        max_entries: int,
        max_bytes: int,
    ) -> tuple[int, int]:
        if is_link_or_reparse(source, missing="assume_link", unreadable="assume_link"):
            raise RuntimeProvisioningError(
                "A managed Registry node source contains an unsupported link."
            )
        if not source.is_dir():
            raise RuntimeProvisioningError(
                "A managed Registry node source is not an ordinary directory."
            )
        destination.mkdir()
        entries = 0
        total_size = 0
        stack = [(source, destination)]
        while stack:
            source_directory, destination_directory = stack.pop()
            with os.scandir(source_directory) as children:
                for child in children:
                    entries += 1
                    if entries > max_entries or RuntimeProvisioner._is_reparse_point(child):
                        raise RuntimeProvisioningError(
                            "A managed Registry node source has an unsupported shape."
                        )
                    output = destination_directory / child.name
                    if child.is_dir(follow_symlinks=False):
                        output.mkdir()
                        stack.append((Path(child.path), output))
                        continue
                    if not child.is_file(follow_symlinks=False):
                        raise RuntimeProvisioningError(
                            "A managed Registry node source has an unsupported file type."
                        )
                    total_size += child.stat(follow_symlinks=False).st_size
                    if total_size > max_bytes:
                        raise RuntimeProvisioningError(
                            "A managed Registry node source exceeds its allowed size."
                        )
                    shutil.copy2(child.path, output, follow_symlinks=False)
        return entries, total_size

    @classmethod
    def _managed_node_tree_identity(
        cls,
        root: Path,
        *,
        max_entries: int,
        max_bytes: int,
    ) -> tuple[int, int, str]:
        """Hash one bounded ordinary tree for idempotent runtime restoration."""

        if is_link_or_reparse(root, missing="assume_link", unreadable="assume_link"):
            raise RuntimeProvisioningError("A managed custom node contains an unsupported link.")
        if not root.is_dir():
            raise RuntimeProvisioningError("A managed custom node is not an ordinary directory.")
        digest = hashlib.sha256(b"lm-atelier-managed-custom-node-v1\0")
        entries = 0
        total_size = 0
        stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
        while stack:
            source_directory, relative_directory = stack.pop()
            with os.scandir(source_directory) as scanned:
                children = sorted(scanned, key=lambda child: child.name)
            for child in children:
                entries += 1
                if entries > max_entries or cls._is_reparse_point(child):
                    raise RuntimeProvisioningError(
                        "A managed custom node has an unsupported shape."
                    )
                relative = relative_directory / child.name
                encoded = relative.as_posix().encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "big"))
                digest.update(encoded)
                if child.is_dir(follow_symlinks=False):
                    digest.update(b"D")
                    stack.append((Path(child.path), relative))
                    continue
                if not child.is_file(follow_symlinks=False):
                    raise RuntimeProvisioningError(
                        "A managed custom node has an unsupported file type."
                    )
                size = child.stat(follow_symlinks=False).st_size
                total_size += size
                if total_size > max_bytes:
                    raise RuntimeProvisioningError(
                        "A managed custom node exceeds its allowed size."
                    )
                digest.update(b"F")
                digest.update(size.to_bytes(8, "big"))
                digest.update(bytes.fromhex(cls._sha256_file(Path(child.path))))
        return entries, total_size, digest.hexdigest()

    def _prune_stale_staging(self, parent: Path, engine: RuntimeName) -> None:
        """Reclaim staging trees abandoned by an interrupted or failed attempt.

        A partial extraction can be several gigabytes across tens of thousands of
        files, and Windows regularly refuses to delete it while a scanner still
        holds a handle. Nothing else reclaims these, so sweep them before staging
        a new attempt rather than accumulating one tree per failure.
        """
        for candidate in sorted(parent.glob(".*.partial-*")):
            if candidate.is_dir():
                self._discard_staging(candidate, engine, stale=True)

    @staticmethod
    def _discard_staging(staging: Path, engine: RuntimeName, *, stale: bool = False) -> None:
        if not staging.exists():
            return
        shutil.rmtree(staging, ignore_errors=True)
        if staging.exists():
            logger.warning(
                "Could not remove %s %s runtime staging directory at %s; "
                "it will be retried on the next installation.",
                "stale" if stale else "incomplete",
                engine,
                staging,
            )
        elif stale:
            logger.info("Reclaimed stale %s runtime staging directory at %s", engine, staging)

    @staticmethod
    def _extract_zip(archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = RuntimeProvisioner._safe_archive_path(member.filename)
                target = destination.joinpath(*relative.parts)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeProvisioningError("Runtime archives may not contain ZIP links.")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if mode:
                    target.chmod(mode & 0o777)

    @staticmethod
    def _extract_tar(archive_path: Path, destination: Path) -> None:
        pending_links: list[tarfile.TarInfo] = []
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                relative = RuntimeProvisioner._safe_archive_path(member.name)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeProvisioningError(
                            f"Could not read runtime archive member: {member.name}"
                        )
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(member.mode & 0o777)
                elif member.issym() or member.islnk():
                    pending_links.append(member)
                else:
                    raise RuntimeProvisioningError(
                        f"Unsupported runtime archive member: {member.name}"
                    )
        for member in pending_links:
            relative = RuntimeProvisioner._safe_archive_path(member.name)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            link_name = PurePosixPath(member.linkname.replace("\\", "/"))
            if member.issym():
                link_target = (target.parent / Path(*link_name.parts)).resolve()
            else:
                link_target = (destination / Path(*link_name.parts)).resolve()
            RuntimeProvisioner._ensure_inside(destination, link_target)
            if not link_target.exists():
                raise RuntimeProvisioningError(
                    f"Runtime archive link target is missing: {member.linkname}"
                )
            if member.issym():
                os.symlink(os.path.relpath(link_target, target.parent), target)
            else:
                os.link(link_target, target)

    @staticmethod
    def _extract_7z(
        archive_path: Path,
        destination: Path,
        *,
        max_entries: int,
        max_uncompressed_bytes: int,
    ) -> None:
        executable = shutil.which("tar")
        if not executable:
            raise RuntimeProvisioningError(
                "Windows archive support is unavailable; install the current Windows updates "
                "and retry."
            )
        try:
            listing = subprocess.run(
                [executable, "-tf", str(archive_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
                env=subprocess_environment(),
            )
            verbose_listing = subprocess.run(
                [executable, "-tvf", str(archive_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
                env=subprocess_environment(),
            )
            RuntimeProvisioner._validate_7z_listing(
                listing.stdout,
                verbose_listing.stdout,
                max_entries=max_entries,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            subprocess.run(
                [executable, "-xf", str(archive_path), "-C", str(destination)],
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,
                env=subprocess_environment(),
            )
            RuntimeProvisioner._validate_extracted_tree(
                destination,
                max_entries=max_entries,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeProvisioningError(
                "The verified ComfyUI archive could not be extracted."
            ) from exc

    @staticmethod
    def _validate_7z_listing(
        names_output: str,
        verbose_output: str,
        *,
        max_entries: int,
        max_uncompressed_bytes: int,
    ) -> None:
        names = [line.strip() for line in names_output.splitlines() if line.strip()]
        verbose = [line for line in verbose_output.splitlines() if line.strip()]
        if len(names) != len(verbose):
            raise RuntimeProvisioningError("The runtime archive listing is inconsistent.")
        if len(names) > max_entries:
            raise RuntimeProvisioningError("The runtime archive contains too many entries.")
        total_size = 0
        for name, details in zip(names, verbose, strict=True):
            RuntimeProvisioner._safe_archive_path(name)
            fields = details.split(maxsplit=8)
            if len(fields) < 8 or not fields[0]:
                raise RuntimeProvisioningError("The runtime archive listing is invalid.")
            if fields[0][0] not in {"-", "d"}:
                raise RuntimeProvisioningError(
                    "Runtime archives may contain only regular files and directories."
                )
            if fields[8] != name:
                raise RuntimeProvisioningError("The runtime archive listing is inconsistent.")
            try:
                size = int(fields[4])
            except ValueError as exc:
                raise RuntimeProvisioningError(
                    "The runtime archive entry size is invalid."
                ) from exc
            if size < 0 or size > max_uncompressed_bytes:
                raise RuntimeProvisioningError("A runtime archive entry is too large.")
            total_size += size
            if total_size > max_uncompressed_bytes:
                raise RuntimeProvisioningError(
                    "The runtime archive expands beyond its allowed size."
                )

    @staticmethod
    def _validate_extracted_tree(
        destination: Path,
        *,
        max_entries: int,
        max_uncompressed_bytes: int,
    ) -> None:
        entries = 0
        total_size = 0
        for candidate in destination.rglob("*"):
            entries += 1
            if entries > max_entries:
                raise RuntimeProvisioningError("The extracted runtime contains too many entries.")
            metadata = candidate.lstat()
            if is_link_or_reparse(
                candidate,
                missing="raise",
                unreadable="raise",
            ):
                raise RuntimeProvisioningError(
                    "The extracted runtime may not contain links or reparse points."
                )
            RuntimeProvisioner._ensure_inside(destination, candidate.resolve())
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise RuntimeProvisioningError(
                    "The extracted runtime contains an unsupported file type."
                )
            total_size += metadata.st_size
            if total_size > max_uncompressed_bytes:
                raise RuntimeProvisioningError("The extracted runtime exceeds its allowed size.")

    def _managed_restoration_candidates(self) -> tuple[RuntimeName, ...]:
        candidates: list[RuntimeName] = []
        for engine in RUNTIME_NAMES:
            definition = self._definition(engine)
            asset = self._asset(engine, definition)
            if not asset or self._security_blocked(definition, asset):
                continue
            ready, paths = self._configured_paths(engine)
            configured_managed = ready and any(
                self._is_inside_runtime_root(path.expanduser()) for path in paths
            )
            final = self._installation_path(engine, definition)
            if not configured_managed and not self._managed_marker_owned(final, engine, definition):
                continue
            candidates.append(engine)
        return tuple(candidates)

    async def _restore_managed_installations(self) -> None:
        for engine in self._restore_candidates:
            if self._restore_cancel.is_set():
                return
            definition = self._definition(engine)
            asset = self._asset(engine, definition)
            if not asset:
                continue
            try:
                final = self._installation_path(engine, definition)
                matched = await asyncio.to_thread(
                    self._managed_marker_matches,
                    final,
                    engine,
                    definition,
                    asset,
                    cancel_requested=self._restore_cancel.is_set,
                )
                if not matched:
                    self._states[engine] = self._status(
                        engine,
                        definition,
                        state="missing",
                        supported=True,
                        size_bytes=int(asset["size_bytes"]),
                        message="Managed runtime verification failed; reinstall it to repair.",
                        asset=asset,
                    )
                    logger.warning(
                        "Managed %s runtime failed startup integrity verification",
                        engine,
                    )
                    continue
                if engine == "comfyui":
                    # A newer managed release can already be present before a
                    # Registry package is renewed in the currently configured
                    # release. Carry those application-owned node bytes before
                    # switching the persisted runtime paths; afterwards the old
                    # configured release is no longer discoverable authority.
                    self._restore_managed_registry_nodes(final, asset, final)
                installed = self._resolve_installed_paths(final, asset)
                self._apply_configuration(engine, installed, persist=True)
                self._states[engine] = self._status(
                    engine,
                    definition,
                    state="ready",
                    supported=True,
                    managed=True,
                    progress=1,
                    message="Ready.",
                    asset=asset,
                )
                self._remove_completed_archive(self._archive_path(engine, definition, asset))
                for overlay in self._security_overlays(asset):
                    self._remove_completed_archive(self._archive_path(engine, definition, overlay))
            except RuntimeVerificationCancelled:
                return
            except (OSError, RuntimeProvisioningError):
                self._states[engine] = self._status(
                    engine,
                    definition,
                    state="missing",
                    supported=True,
                    size_bytes=int(asset["size_bytes"]),
                    message="Managed runtime verification failed; reinstall it to repair.",
                    asset=asset,
                )
                logger.warning("Ignored incomplete managed %s runtime at %s", engine, final)

    def _cleanup_completed_archives(self) -> None:
        for engine in RUNTIME_NAMES:
            definition = self._definition(engine)
            asset = self._asset(engine, definition)
            if not asset:
                continue
            status = self._states.get(engine)
            if status and status.state == "ready" and status.managed:
                self._remove_completed_archive(self._archive_path(engine, definition, asset))
                for overlay in self._security_overlays(asset):
                    self._remove_completed_archive(self._archive_path(engine, definition, overlay))

    @staticmethod
    def _remove_completed_archive(archive: Path) -> None:
        try:
            archive.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove extracted runtime archive at %s", archive)

    def _archive_path(
        self,
        engine: RuntimeName,
        definition: Mapping[str, Any],
        asset: Mapping[str, Any],
    ) -> Path:
        filename = PurePosixPath(urlparse(str(asset["url"])).path).name
        if not filename or filename in {".", ".."}:
            raise RuntimeProvisioningError("The runtime archive name is invalid.")
        prefix = self._safe_component(f"{engine}-{definition['pinned_release']}")
        return self.archive_root / f"{prefix}-{filename}"

    def _apply_configuration(
        self,
        engine: RuntimeName,
        installed: Mapping[str, Path],
        *,
        persist: bool,
    ) -> None:
        executable = installed["executable"].resolve()
        if engine == "llama.cpp":
            self.settings.chat_engine = "llama.cpp"
            self.settings.llama_executable = executable
            values = {
                "LOCAL_LM_CHAT_ENGINE": "llama.cpp",
                "LOCAL_LM_LLAMA_EXECUTABLE": str(executable),
            }
        elif engine == "vllm":
            self.settings.chat_engine = "vllm"
            self.settings.vllm_executable = executable
            values = {
                "LOCAL_LM_CHAT_ENGINE": "vllm",
                "LOCAL_LM_VLLM_EXECUTABLE": str(executable),
            }
        else:
            directory = installed["directory"].resolve()
            self.settings.media_engine = "comfyui"
            self.settings.comfy_executable = executable
            self.settings.comfy_directory = directory
            values = {
                "LOCAL_LM_MEDIA_ENGINE": "comfyui",
                "LOCAL_LM_COMFY_EXECUTABLE": str(executable),
                "LOCAL_LM_COMFY_DIRECTORY": str(directory),
            }
        if persist:
            persist_runtime_values(
                self.settings.data_dir,
                values,
                self.environment,
            )

    def _configured_status(
        self,
        engine: RuntimeName,
        definition: dict[str, Any],
    ) -> RuntimeStatus | None:
        ready, paths = self._configured_paths(engine)
        if not ready:
            return None
        managed = any(self._is_inside_runtime_root(item.expanduser()) for item in paths)
        asset: dict[str, Any] | None = None
        if managed:
            asset = self._asset(engine, definition)
            if self._security_blocked(definition, asset):
                return None
            final = self._installation_path(engine, definition)
            if (
                not asset
                or not all(self._is_inside(item.expanduser(), final) for item in paths)
                or not self._managed_marker_matches(final, engine, definition, asset)
            ):
                return None
        return self._status(
            engine,
            definition,
            state="ready",
            supported=True,
            managed=managed,
            progress=1,
            message="Ready.",
            asset=asset if managed else None,
        )

    def _configured_paths(self, engine: RuntimeName) -> tuple[bool, list[Path]]:
        if engine == "llama.cpp":
            executable = self.settings.llama_executable
            ready = bool(executable and executable.expanduser().is_file())
            paths = [executable] if executable else []
        elif engine == "vllm":
            executable = self.settings.vllm_executable
            ready = bool(executable and executable.expanduser().is_file())
            paths = [executable] if executable else []
        else:
            executable = self.settings.comfy_executable
            directory = self.settings.comfy_directory
            ready = bool(
                executable
                and executable.expanduser().is_file()
                and directory
                and directory.expanduser().is_dir()
                and (directory.expanduser() / "main.py").is_file()
            )
            paths = [item for item in (executable, directory) if item]
        return ready, paths

    def _asset(
        self,
        engine: RuntimeName,
        definition: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = self._platform_keys[engine]
        raw = definition.get("runtime_assets", {}).get(key)
        return raw if isinstance(raw, dict) else None

    @classmethod
    def _platform_key(cls, engine: RuntimeName) -> str:
        return cls._platform_key_for(engine, cls._nvidia_runtime_info())

    @staticmethod
    def _platform_key_for(
        engine: RuntimeName,
        nvidia: tuple[int, int] | None,
    ) -> str:
        machine = platform.machine().lower()
        x64 = machine in {"amd64", "x86_64"}
        system = platform.system().lower()
        if not x64:
            nvidia = None
        if system == "windows" and x64:
            if engine == "vllm":
                if nvidia:
                    driver_major, compute_major = nvidia
                    if driver_major >= 580 and compute_major >= 8:
                        return "windows-x86_64-nvidia-cu13"
                return "windows-x86_64"
            if engine == "comfyui":
                if nvidia:
                    driver_major, compute_major = nvidia
                    if driver_major >= 580 and compute_major >= 7:
                        return "windows-x86_64-nvidia-cu13"
                    if driver_major >= 560 and compute_major >= 6:
                        return "windows-x86_64-nvidia-cu126"
                return "windows-x86_64"
            return "windows-x86_64-nvidia" if nvidia else "windows-x86_64"
        if system == "linux" and x64:
            try:
                distribution = platform.freedesktop_os_release().get("ID", "").lower()
            except OSError:
                distribution = ""
            if distribution == "ubuntu":
                return "ubuntu-x86_64-nvidia" if nvidia else "ubuntu-x86_64"
            return "linux-x86_64"
        return f"{system}-{machine}"

    @staticmethod
    def _nvidia_runtime_info() -> tuple[int, int] | None:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=driver_version,compute_cap",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                env=subprocess_environment(),
            )
            first_gpu = result.stdout.splitlines()[0]
            driver, compute = (item.strip() for item in first_gpu.split(",", maxsplit=1))
            return int(driver.split(".", maxsplit=1)[0]), int(float(compute))
        except (IndexError, OSError, subprocess.SubprocessError, ValueError):
            return None

    def _resolve_installed_paths(
        self,
        root: Path,
        asset: Mapping[str, Any],
    ) -> dict[str, Path]:
        executable = self._safe_installed_path(root, str(asset["executable"]))
        if not executable.is_file():
            raise RuntimeProvisioningError("The runtime archive has no required executable.")
        installed = {"executable": executable}
        if raw_directory := asset.get("directory"):
            directory = self._safe_installed_path(root, str(raw_directory))
            if not directory.is_dir() or not (directory / "main.py").is_file():
                raise RuntimeProvisioningError("The ComfyUI runtime directory is incomplete.")
            installed["directory"] = directory
        return installed

    @staticmethod
    def _security_overlays(asset: Mapping[str, Any]) -> list[dict[str, Any]]:
        overlays = asset.get("security_overlays", [])
        if not isinstance(overlays, list):
            raise RuntimeProvisioningError("The runtime security overlay definition is invalid.")
        return [cast(dict[str, Any], overlay) for overlay in overlays]

    @classmethod
    def _overlay_marker(cls, asset: Mapping[str, Any]) -> list[dict[str, str]]:
        markers: list[dict[str, str]] = []
        for overlay in cls._security_overlays(asset):
            filename = PurePosixPath(urlparse(str(overlay["url"])).path).name
            markers.append(
                {
                    "name": str(overlay["name"]),
                    "asset": filename,
                    "sha256": str(overlay["sha256"]),
                    "target_directory": str(overlay["target_directory"]),
                    "compatibility_tag": str(overlay["compatibility_tag"]),
                    "contract_sha256": cls._json_sha256(overlay),
                }
            )
        return markers

    @classmethod
    def _runtime_contract_sha256(cls, asset: Mapping[str, Any]) -> str:
        contract = {
            "dependency_inventory_count": asset.get("dependency_inventory_count"),
            "dependency_inventory_sha256": asset.get("dependency_inventory_sha256"),
            "dependency_review": asset.get("dependency_review"),
            "runtime_probe": asset.get("runtime_probe"),
            "security_overlays": asset.get("security_overlays", []),
        }
        return cls._json_sha256(contract)

    @staticmethod
    def _json_sha256(value: Any) -> str:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _apply_security_overlay(
        self,
        root: Path,
        overlay: Mapping[str, Any],
        archive_path: Path,
    ) -> None:
        if overlay.get("archive_type") != "zip":
            raise RuntimeProvisioningError("Only pinned ZIP security overlays are supported.")
        target = self._safe_installed_path(root, str(overlay["target_directory"]))
        if not target.is_dir():
            raise RuntimeProvisioningError("The runtime security overlay target is missing.")
        expected_files = cast(Mapping[str, str], overlay["expected_files"])
        expected_names = set(expected_files)
        max_entries = int(overlay["max_entries"])
        max_uncompressed_bytes = int(overlay["max_uncompressed_bytes"])
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                if len(members) > max_entries:
                    raise RuntimeProvisioningError(
                        "The runtime security overlay contains too many entries."
                    )
                names: list[str] = []
                casefolded: set[str] = set()
                total_size = 0
                for member in members:
                    relative = self._safe_archive_path(member.filename)
                    normalized = str(relative)
                    mode = member.external_attr >> 16
                    if member.is_dir() or stat.S_ISLNK(mode):
                        raise RuntimeProvisioningError(
                            "The runtime security overlay may contain only regular files."
                        )
                    folded = normalized.casefold()
                    if folded in casefolded:
                        raise RuntimeProvisioningError(
                            "The runtime security overlay contains duplicate paths."
                        )
                    casefolded.add(folded)
                    names.append(normalized)
                    total_size += member.file_size
                    if member.file_size < 0 or total_size > max_uncompressed_bytes:
                        raise RuntimeProvisioningError(
                            "The runtime security overlay expands beyond its allowed size."
                        )
                if set(names) != expected_names or len(names) != len(expected_names):
                    raise RuntimeProvisioningError(
                        "The runtime security overlay inventory does not match its audit."
                    )
            self._extract_zip(archive_path, target)
        except (OSError, zipfile.BadZipFile) as exc:
            raise RuntimeProvisioningError(
                "The verified runtime security overlay could not be extracted."
            ) from exc

        for relative_value, expected_hash in expected_files.items():
            candidate = self._safe_installed_path(target, relative_value)
            if not candidate.is_file() or self._sha256_file(candidate) != expected_hash:
                raise RuntimeProvisioningError(
                    "The runtime security overlay failed its file-level audit."
                )

        rewrite_files = cast(Mapping[str, str], overlay.get("rewrite_files", {}))
        for relative_value, content in rewrite_files.items():
            candidate = self._safe_installed_path(target, relative_value)
            encoded = content.encode("utf-8")
            candidate.write_bytes(encoded)
            if candidate.read_bytes() != encoded:
                raise RuntimeProvisioningError(
                    "The runtime security overlay configuration could not be verified."
                )

    def _verify_runtime_contract(
        self,
        root: Path,
        asset: Mapping[str, Any],
        installed: Mapping[str, Path],
    ) -> None:
        inventory_count = asset.get("dependency_inventory_count")
        inventory_hash = asset.get("dependency_inventory_sha256")
        if inventory_count is not None or inventory_hash is not None:
            expected_count = self._positive_int(inventory_count)
            if not isinstance(inventory_hash, str) or not _SHA256.fullmatch(inventory_hash):
                raise RuntimeProvisioningError(
                    "The runtime dependency inventory contract is invalid."
                )
            executable = installed["executable"]
            site_packages = executable.parent / "Lib" / "site-packages"
            self._ensure_inside(root, site_packages.resolve())
            if not site_packages.is_dir():
                raise RuntimeProvisioningError("The runtime dependency inventory is missing.")
            identities = sorted(
                candidate.name
                for candidate in site_packages.iterdir()
                if candidate.is_dir() and candidate.name.endswith(".dist-info")
            )
            canonical = ("\n".join(identities) + "\n").encode("utf-8")
            if (
                len(identities) != expected_count
                or hashlib.sha256(canonical).hexdigest() != inventory_hash
            ):
                raise RuntimeProvisioningError(
                    "The runtime dependency inventory does not match its audit."
                )

        raw_probe = asset.get("runtime_probe")
        if raw_probe is None:
            return
        probe = cast(Mapping[str, Any], raw_probe)
        imports = cast(list[str], probe["imports"])
        packages = cast(Mapping[str, str], probe["packages"])
        script = (
            "import importlib, importlib.metadata, json, sys\n"
            f"imports = {json.dumps(imports)}\n"
            f"packages = {json.dumps(list(packages))}\n"
            "for module in imports:\n"
            "    importlib.import_module(module)\n"
            "import comfyui_version\n"
            "result = {\n"
            "    'python': '.'.join(str(part) for part in sys.version_info[:3]),\n"
            "    'comfyui': comfyui_version.__version__,\n"
            "    'packages': {\n"
            "        name: importlib.metadata.version(name) for name in packages\n"
            "    },\n"
            "}\n"
            f"print({_RUNTIME_PROBE_SENTINEL!r} + json.dumps(result, sort_keys=True))\n"
        )
        try:
            result = subprocess.run(
                [str(installed["executable"]), "-I", "-c", script],
                cwd=installed["directory"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                env=subprocess_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeProvisioningError(
                "The staged runtime failed its isolated compatibility probe."
            ) from exc
        payload_line = next(
            (
                line.removeprefix(_RUNTIME_PROBE_SENTINEL)
                for line in reversed(result.stdout.splitlines())
                if line.startswith(_RUNTIME_PROBE_SENTINEL)
            ),
            None,
        )
        try:
            actual = json.loads(payload_line) if payload_line is not None else None
        except ValueError as exc:
            raise RuntimeProvisioningError(
                "The staged runtime returned an invalid compatibility result."
            ) from exc
        expected = {
            "python": probe["python"],
            "comfyui": probe["comfyui"],
            "packages": dict(packages),
        }
        if actual != expected:
            raise RuntimeProvisioningError(
                "The staged runtime versions do not match the audited contract."
            )

    def _runtime_file_hashes(
        self,
        root: Path,
        asset: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, str]:
        files = self._integrity_file_map(root)
        required = set(self._runtime_file_paths(asset))
        if not required.issubset(files):
            raise RuntimeProvisioningError(
                "The runtime archive is missing a required integrity file."
            )
        ordered = sorted(files.items())
        total = len(ordered)
        hashes: dict[str, str] = {}
        last_update = 0.0
        for index, (relative, path) in enumerate(ordered, start=1):
            hashes[relative] = self._sha256_file(path)
            now = time.monotonic()
            if progress and (index == total or now - last_update >= 0.25):
                progress(index, total)
                last_update = now
        return hashes

    @staticmethod
    def _runtime_file_paths(asset: Mapping[str, Any]) -> list[str]:
        paths = [str(asset["executable"]).replace("\\", "/")]
        if raw_directory := asset.get("directory"):
            directory = str(raw_directory).replace("\\", "/").rstrip("/")
            paths.append(f"{directory}/main.py")
        return paths

    def _integrity_file_map(self, root: Path) -> dict[str, Path]:
        """Enumerate the files that make up a managed runtime.

        This runs on every status read, and status is read on every setup
        readiness poll, so it walks with `os.scandir` and reuses the metadata the
        directory read already returned. The previous `rglob` plus `resolve` plus
        `stat` cost three or more syscalls per entry and took about 7.8 seconds
        across the 55,000 files of the ComfyUI runtime - longer than the poll
        interval, which is what made setup appear to hang.

        Only a reparse point can refer somewhere else, so only a reparse point is
        resolved and containment-checked. An ordinary entry cannot escape the
        root because there is nothing to follow.
        """

        return self._integrity_file_map_cancellable(root)

    def _integrity_file_map_cancellable(
        self,
        root: Path,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, Path]:
        files: dict[str, Path] = {}
        total_size = 0
        stack = [root]
        while stack:
            if cancel_requested and cancel_requested():
                raise RuntimeVerificationCancelled
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if cancel_requested and cancel_requested():
                        raise RuntimeVerificationCancelled
                    candidate = Path(entry.path)
                    relative = candidate.relative_to(root)
                    relative_posix = str(relative).replace("\\", "/")
                    if relative_posix == _MANAGED_MARKER or self._mutable_runtime_path(relative):
                        continue
                    if self._is_reparse_point(entry):
                        resolved = candidate.resolve()
                        self._ensure_inside(root, resolved)
                        if resolved.is_dir():
                            raise RuntimeProvisioningError(
                                "The runtime contains an unsupported dependency directory link."
                            )
                        if not resolved.is_file():
                            raise RuntimeProvisioningError(
                                "The runtime contains an unsupported dependency file type."
                            )
                        size = resolved.stat().st_size
                    elif entry.is_dir(follow_symlinks=False):
                        stack.append(candidate)
                        continue
                    elif not entry.is_file(follow_symlinks=False):
                        raise RuntimeProvisioningError(
                            "The runtime contains an unsupported dependency file type."
                        )
                    else:
                        size = entry.stat(follow_symlinks=False).st_size
                    files[relative_posix] = candidate
                    total_size += size
                    if len(files) > _MAX_RUNTIME_FILES or total_size > _MAX_RUNTIME_BYTES:
                        raise RuntimeProvisioningError(
                            "The runtime dependency tree exceeds its integrity limits."
                        )
        return files

    @staticmethod
    def _is_reparse_point(entry: os.DirEntry[str]) -> bool:
        """Whether an entry can refer somewhere other than where it sits.

        `is_symlink` misses Windows junctions, which carry a different reparse
        tag, so the file attributes are consulted too. A junction escaping its
        tree has already caused damage in this project once.
        """

        if entry.is_symlink():
            return True
        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))

    @staticmethod
    def _mutable_runtime_path(relative: Path) -> bool:
        lowered_parts = {part.casefold() for part in relative.parts}
        return bool(
            lowered_parts & _MUTABLE_RUNTIME_DIRECTORIES
            or relative.suffix.casefold() in {".log", ".pyc", ".pyo"}
        )

    @staticmethod
    def _safe_installed_path(root: Path, relative_value: str) -> Path:
        relative = RuntimeProvisioner._safe_archive_path(relative_value)
        candidate = root.joinpath(*relative.parts).resolve()
        RuntimeProvisioner._ensure_inside(root, candidate)
        return candidate

    @staticmethod
    def _safe_archive_path(value: str) -> PurePosixPath:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and ":" in path.parts[0])
        ):
            raise RuntimeProvisioningError("The runtime archive contains an unsafe path.")
        return path

    @staticmethod
    def _ensure_inside(root: Path, candidate: Path) -> None:
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeProvisioningError(
                "The runtime archive attempted to escape its install directory."
            ) from exc

    def _managed_marker_owned(
        self,
        root: Path,
        engine: RuntimeName,
        definition: Mapping[str, Any],
    ) -> bool:
        try:
            marker = json.loads((root / _MANAGED_MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(
            isinstance(marker, dict)
            and marker.get("schema_version") in {1, 2, 3}
            and marker.get("engine") == engine
            and marker.get("release") == definition["pinned_release"]
        )

    def _managed_marker_matches(
        self,
        root: Path,
        engine: RuntimeName,
        definition: Mapping[str, Any],
        asset: Mapping[str, Any],
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> bool:
        try:
            marker_text = (root / _MANAGED_MARKER).read_text(encoding="utf-8")
            marker = json.loads(marker_text)
        except (OSError, ValueError):
            return False
        expected_asset = PurePosixPath(urlparse(str(asset["url"])).path).name
        if not (
            isinstance(marker, dict)
            and marker.get("schema_version") == 3
            and marker.get("engine") == engine
            and marker.get("release") == definition["pinned_release"]
            and marker.get("asset") == expected_asset
            and marker.get("sha256") == asset["sha256"]
            and marker.get("overlays") == self._overlay_marker(asset)
            and marker.get("runtime_contract_sha256") == self._runtime_contract_sha256(asset)
            and isinstance(marker.get("files"), dict)
        ):
            return False
        files = marker["files"]
        expected_files = set(self._runtime_file_paths(asset))
        if not expected_files.issubset(files):
            return False
        try:
            installed_files = self._integrity_file_map_cancellable(
                root,
                cancel_requested=cancel_requested,
            )
        except (OSError, RuntimeProvisioningError):
            return False
        if set(files) != set(installed_files):
            return False
        signatures: list[tuple[str, int, int, int, int]] = []
        for relative_value, expected_hash in sorted(files.items()):
            if cancel_requested and cancel_requested():
                raise RuntimeVerificationCancelled
            if not isinstance(relative_value, str) or not _SHA256.fullmatch(str(expected_hash)):
                return False
            candidate = installed_files[relative_value]
            link_metadata = candidate.lstat()
            target_metadata = candidate.stat()
            signatures.append(
                (
                    relative_value,
                    link_metadata.st_size,
                    link_metadata.st_mtime_ns,
                    target_metadata.st_size,
                    target_metadata.st_mtime_ns,
                )
            )
        cache_value = (
            hashlib.sha256(marker_text.encode()).hexdigest(),
            tuple(signatures),
        )
        if self._integrity_cache.get(root) == cache_value:
            return True
        for relative_value, expected_hash in files.items():
            if (
                self._sha256_file(
                    installed_files[relative_value],
                    cancel_requested=cancel_requested,
                )
                != expected_hash
            ):
                return False
        self._integrity_cache[root] = cache_value
        return True

    def _installation_path(
        self,
        engine: RuntimeName,
        definition: Mapping[str, Any],
    ) -> Path:
        return (
            self.runtime_root
            / self._safe_component(engine)
            / self._safe_component(str(definition["pinned_release"]))
        )

    def _is_inside_runtime_root(self, path: Path) -> bool:
        return self._is_inside(path, self.runtime_root)

    @staticmethod
    def _is_inside(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            return False

    def _check_disk_space(self, asset: Mapping[str, Any]) -> None:
        required = int(asset.get("required_free_bytes", asset["size_bytes"] * 2))
        available = shutil.disk_usage(self.runtime_root).free
        if available < required:
            raise RuntimeProvisioningError(
                f"Runtime setup needs {required} free bytes; {available} are available."
            )

    def _set_download_progress(
        self,
        engine: RuntimeName,
        definition: dict[str, Any],
        expected_size: int,
        downloaded: int,
    ) -> None:
        self._states[engine] = self._status(
            engine,
            definition,
            state="installing",
            supported=True,
            progress=downloaded / expected_size,
            progress_json=reduce_progress(
                self._runtime_progress(engine),
                stage="downloading runtime",
                completed_units=downloaded,
                total_units=expected_size,
                unit="bytes",
                queue_resource="network_transfer",
            ),
            downloaded_bytes=downloaded,
            size_bytes=expected_size,
            message="Downloading runtime.",
        )

    def _set_install_progress(
        self,
        engine: RuntimeName,
        definition: Mapping[str, Any],
        asset: Mapping[str, Any],
        *,
        stage: str,
        message: str,
        completed_units: int | None = None,
        total_units: int | None = None,
    ) -> None:
        current = self._states.get(engine)
        progress_json = reduce_progress(
            current.progress_json.model_dump(mode="json")
            if current and current.progress_json
            else None,
            stage=stage,
            completed_units=completed_units,
            total_units=total_units,
            unit="files" if total_units is not None else None,
            queue_resource="disk",
            indeterminate=total_units is None,
        )
        stage_progress = (
            completed_units / total_units
            if completed_units is not None and total_units
            else (current.progress if current else 0)
        )
        self._states[engine] = self._status(
            engine,
            definition,
            state="installing",
            supported=True,
            progress=stage_progress,
            progress_json=progress_json,
            downloaded_bytes=int(asset["size_bytes"]),
            size_bytes=int(asset["size_bytes"]),
            message=message,
            asset=asset,
        )

    def _status(
        self,
        engine: RuntimeName,
        definition: Mapping[str, Any],
        *,
        state: Literal["missing", "installing", "ready", "failed", "unsupported"],
        supported: bool,
        managed: bool = False,
        progress: float = 0,
        progress_json: dict[str, Any] | None = None,
        downloaded_bytes: int = 0,
        size_bytes: int | None = None,
        message: str,
        asset: Mapping[str, Any] | None = None,
    ) -> RuntimeStatus:
        security_source = (
            asset if asset and asset.get("security_status") is not None else definition
        )
        return RuntimeStatus(
            engine=engine,
            release=str(definition["pinned_release"]),
            state=state,
            supported=supported,
            managed=managed,
            progress=progress,
            progress_json=ProgressV2.model_validate(progress_json)
            if progress_json
            else (
                ProgressV2.model_validate(
                    reduce_progress(
                        self._runtime_progress(engine),
                        stage="complete",
                        stage_progress=1,
                        overall_progress=1,
                    )
                )
                if state == "ready"
                else ProgressV2.model_validate(
                    reduce_progress(
                        self._runtime_progress(engine),
                        stage=message.rstrip(".").lower(),
                        indeterminate=True,
                    )
                )
                if state == "installing"
                else None
            ),
            downloaded_bytes=downloaded_bytes,
            size_bytes=size_bytes,
            distribution=str(definition["distribution"]),
            license=str(definition["license"]),
            security_status=cast(
                Literal["checksum-pinned", "blocked"],
                str(security_source.get("security_status") or "checksum-pinned"),
            ),
            security_message=str(security_source.get("security_message") or ""),
            message=message,
        )

    def _runtime_progress(self, engine: RuntimeName) -> dict[str, Any] | None:
        current = self._states.get(engine)
        progress = current.progress_json if current else None
        if not progress:
            return None
        return ProgressV2.model_validate(progress).model_dump(mode="json")

    def _definition(self, engine: RuntimeName) -> dict[str, Any]:
        raw = self._manifest["engines"].get(engine)
        if not isinstance(raw, dict):
            raise ValueError(f"unknown runtime: {engine}")
        return raw

    @staticmethod
    def _security_blocked(
        definition: Mapping[str, Any],
        asset: Mapping[str, Any] | None = None,
    ) -> bool:
        return bool(
            definition.get("security_status") == "blocked"
            or (asset and asset.get("security_status") == "blocked")
        )

    @staticmethod
    def _security_message(
        definition: Mapping[str, Any],
        asset: Mapping[str, Any] | None = None,
    ) -> str:
        return str(
            (asset or {}).get("security_message")
            or definition.get("security_message")
            or "Automatic setup is paused pending a runtime security review."
        )

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeProvisioningError("The inference runtime manifest is invalid.") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 2
            or not isinstance(payload.get("engines"), dict)
        ):
            raise RuntimeProvisioningError("The inference runtime manifest schema is unsupported.")
        for engine in RUNTIME_NAMES:
            definition = payload["engines"].get(engine)
            if not isinstance(definition, dict):
                raise RuntimeProvisioningError(f"The {engine} runtime definition is missing.")
            for key in ("pinned_release", "distribution", "license"):
                if not definition.get(key):
                    raise RuntimeProvisioningError(f"The {engine} runtime definition has no {key}.")
            if "runtime_assets" not in definition:
                raise RuntimeProvisioningError(
                    f"The {engine} runtime definition has no runtime_assets."
                )
            if definition.get("security_status", "checksum-pinned") not in {
                "checksum-pinned",
                "blocked",
            }:
                raise RuntimeProvisioningError(f"The {engine} runtime security status is invalid.")
            if definition.get("security_status") == "blocked" and not definition.get(
                "security_message"
            ):
                raise RuntimeProvisioningError(
                    f"The {engine} blocked runtime has no security explanation."
                )
            runtime_assets = definition["runtime_assets"]
            if not isinstance(runtime_assets, dict) or (
                not runtime_assets and definition.get("security_status") != "blocked"
            ):
                raise RuntimeProvisioningError(f"The {engine} runtime assets are invalid.")
            if engine == "comfyui":
                RuntimeProvisioner._validate_security_review(definition)
            for asset_key, asset in runtime_assets.items():
                if not isinstance(asset_key, str) or not isinstance(asset, dict):
                    raise RuntimeProvisioningError(
                        f"The {engine} runtime asset definition is invalid."
                    )
                RuntimeProvisioner._validate_runtime_asset(engine, asset)
                if engine == "comfyui":
                    RuntimeProvisioner._validate_comfy_asset(
                        path.parent,
                        definition,
                        asset_key,
                        asset,
                    )
        return payload

    @staticmethod
    def _validate_security_review(definition: Mapping[str, Any]) -> None:
        review = definition.get("security_review")
        if not isinstance(review, Mapping):
            raise RuntimeProvisioningError("The ComfyUI runtime has no security review.")
        advisories = review.get("upstream_advisories")
        if (
            not isinstance(review.get("reviewed_at"), str)
            or not review["reviewed_at"]
            or review.get("release_is_immutable") is not True
            or not isinstance(review.get("package_audit"), str)
            or not review["package_audit"]
            or not isinstance(advisories, list)
            or not advisories
        ):
            raise RuntimeProvisioningError("The ComfyUI runtime security review is incomplete.")
        for url in advisories:
            RuntimeProvisioner._validate_https_url(url)

    @staticmethod
    def _validate_runtime_asset(
        engine: RuntimeName,
        asset: Mapping[str, Any],
    ) -> None:
        try:
            size_bytes = RuntimeProvisioner._positive_int(asset.get("size_bytes"))
        except RuntimeProvisioningError as exc:
            raise RuntimeProvisioningError(
                f"The {engine} runtime asset definition is invalid."
            ) from exc
        if (
            not _SHA256.fullmatch(str(asset.get("sha256", "")))
            or not asset.get("url")
            or asset.get("archive_type") not in {"zip", "tar.gz", "7z"}
            or not asset.get("executable")
            or size_bytes <= 0
        ):
            raise RuntimeProvisioningError(f"The {engine} runtime asset definition is invalid.")
        RuntimeProvisioner._validate_https_url(asset["url"])
        RuntimeProvisioner._safe_archive_path(str(asset["executable"]))
        if asset.get("directory"):
            RuntimeProvisioner._safe_archive_path(str(asset["directory"]))
        security_status = asset.get("security_status", "checksum-pinned")
        if security_status not in {"checksum-pinned", "blocked"}:
            raise RuntimeProvisioningError(
                f"The {engine} runtime asset security status is invalid."
            )
        if security_status == "blocked" and not asset.get("security_message"):
            raise RuntimeProvisioningError(
                f"The {engine} blocked runtime asset has no security explanation."
            )

    @staticmethod
    def _validate_comfy_asset(
        manifest_root: Path,
        definition: Mapping[str, Any],
        asset_key: str,
        asset: Mapping[str, Any],
    ) -> None:
        count = RuntimeProvisioner._positive_int(asset.get("dependency_inventory_count"))
        inventory_hash = str(asset.get("dependency_inventory_sha256", ""))
        if not _SHA256.fullmatch(inventory_hash):
            raise RuntimeProvisioningError("The ComfyUI runtime dependency inventory is invalid.")
        RuntimeProvisioner._validate_dependency_review(
            manifest_root,
            definition,
            asset_key,
            asset,
            count=count,
            inventory_hash=inventory_hash,
        )
        if RuntimeProvisioner._security_blocked(definition, asset):
            return
        overlays = asset.get("security_overlays")
        if not isinstance(overlays, list):
            raise RuntimeProvisioningError("The ComfyUI runtime security overlay list is missing.")
        for overlay in overlays:
            RuntimeProvisioner._validate_security_overlay(overlay)
        RuntimeProvisioner._validate_runtime_probe(asset.get("runtime_probe"))

    @staticmethod
    def _validate_dependency_review(
        manifest_root: Path,
        definition: Mapping[str, Any],
        asset_key: str,
        asset: Mapping[str, Any],
        *,
        count: int,
        inventory_hash: str,
    ) -> None:
        reference = asset.get("dependency_review")
        if not isinstance(reference, Mapping):
            raise RuntimeProvisioningError("The ComfyUI dependency review reference is missing.")
        relative_value = str(reference.get("file", ""))
        expected_hash = str(reference.get("sha256", ""))
        review_asset_key = str(reference.get("asset_key", ""))
        if (
            not relative_value
            or not _SHA256.fullmatch(expected_hash)
            or review_asset_key != asset_key
        ):
            raise RuntimeProvisioningError("The ComfyUI dependency review reference is invalid.")
        relative = RuntimeProvisioner._safe_archive_path(relative_value)
        review_path = manifest_root.joinpath(*relative.parts).resolve()
        RuntimeProvisioner._ensure_inside(manifest_root, review_path)
        try:
            review_bytes = review_path.read_bytes()
            review = json.loads(review_bytes)
        except (OSError, ValueError) as exc:
            raise RuntimeProvisioningError(
                "The ComfyUI dependency review is missing or invalid."
            ) from exc
        if hashlib.sha256(review_bytes).hexdigest() != expected_hash:
            raise RuntimeProvisioningError(
                "The ComfyUI dependency review failed SHA-256 verification."
            )
        if (
            not isinstance(review, dict)
            or review.get("schema_version") != 1
            or review.get("release") != definition["pinned_release"]
            or not isinstance(review.get("assets"), dict)
        ):
            raise RuntimeProvisioningError("The ComfyUI dependency review schema is invalid.")
        reviewed_asset = review["assets"].get(review_asset_key)
        if not isinstance(reviewed_asset, dict):
            raise RuntimeProvisioningError(
                "The ComfyUI dependency review omits the selected asset."
            )
        if (
            reviewed_asset.get("source_asset_url") != asset["url"]
            or reviewed_asset.get("source_asset_sha256") != asset["sha256"]
            or reviewed_asset.get("inventory_count") != count
            or reviewed_asset.get("inventory_sha256") != inventory_hash
        ):
            raise RuntimeProvisioningError(
                "The ComfyUI dependency review does not match the runtime asset."
            )
        distributions = reviewed_asset.get("distributions")
        if not isinstance(distributions, list) or len(distributions) != count:
            raise RuntimeProvisioningError(
                "The ComfyUI dependency review has an incomplete distribution list."
            )
        identities: list[str] = []
        folded_identities: set[str] = set()
        for distribution in distributions:
            if not isinstance(distribution, Mapping):
                raise RuntimeProvisioningError(
                    "The ComfyUI dependency review contains an invalid distribution."
                )
            identity = distribution.get("dist_info")
            license_source = distribution.get("license_source")
            if (
                not isinstance(identity, str)
                or not identity.endswith(".dist-info")
                or PurePosixPath(identity).name != identity
                or identity.casefold() in folded_identities
                or not all(
                    isinstance(distribution.get(field), str) and bool(distribution[field])
                    for field in ("name", "version", "license")
                )
            ):
                raise RuntimeProvisioningError(
                    "The ComfyUI dependency review contains an invalid distribution."
                )
            RuntimeProvisioner._validate_https_url(license_source)
            identities.append(identity)
            folded_identities.add(identity.casefold())
        canonical = ("\n".join(sorted(identities)) + "\n").encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != inventory_hash:
            raise RuntimeProvisioningError("The ComfyUI dependency review inventory has drifted.")

    @staticmethod
    def _validate_security_overlay(raw_overlay: Any) -> None:
        if not isinstance(raw_overlay, Mapping):
            raise RuntimeProvisioningError("The ComfyUI runtime security overlay is invalid.")
        overlay = raw_overlay
        if (
            not all(
                isinstance(overlay.get(field), str) and bool(overlay[field])
                for field in (
                    "name",
                    "url",
                    "sha256",
                    "archive_type",
                    "target_directory",
                    "compatibility_tag",
                    "license",
                    "license_file",
                )
            )
            or overlay["archive_type"] != "zip"
            or not _SHA256.fullmatch(str(overlay["sha256"]))
        ):
            raise RuntimeProvisioningError("The ComfyUI runtime security overlay is invalid.")
        RuntimeProvisioner._validate_https_url(overlay["url"])
        RuntimeProvisioner._positive_int(overlay.get("size_bytes"))
        RuntimeProvisioner._positive_int(overlay.get("max_entries"))
        RuntimeProvisioner._positive_int(overlay.get("max_uncompressed_bytes"))
        RuntimeProvisioner._safe_archive_path(str(overlay["target_directory"]))
        expected_files = overlay.get("expected_files")
        if not isinstance(expected_files, Mapping) or not expected_files:
            raise RuntimeProvisioningError(
                "The ComfyUI runtime security overlay file audit is missing."
            )
        folded_paths: set[str] = set()
        for relative_value, expected_hash in expected_files.items():
            if not isinstance(relative_value, str) or not _SHA256.fullmatch(str(expected_hash)):
                raise RuntimeProvisioningError(
                    "The ComfyUI runtime security overlay file audit is invalid."
                )
            relative = RuntimeProvisioner._safe_archive_path(relative_value)
            folded = str(relative).casefold()
            if folded in folded_paths:
                raise RuntimeProvisioningError(
                    "The ComfyUI runtime security overlay file audit has duplicate paths."
                )
            folded_paths.add(folded)
        license_file = str(RuntimeProvisioner._safe_archive_path(str(overlay["license_file"])))
        if license_file not in expected_files:
            raise RuntimeProvisioningError(
                "The ComfyUI runtime security overlay omits its license file."
            )
        rewrite_files = overlay.get("rewrite_files", {})
        if not isinstance(rewrite_files, Mapping):
            raise RuntimeProvisioningError(
                "The ComfyUI runtime security overlay rewrites are invalid."
            )
        for relative_value, content in rewrite_files.items():
            if (
                not isinstance(relative_value, str)
                or str(RuntimeProvisioner._safe_archive_path(relative_value)) not in expected_files
                or not isinstance(content, str)
                or len(content.encode("utf-8")) > 1024 * 1024
            ):
                raise RuntimeProvisioningError(
                    "The ComfyUI runtime security overlay rewrites are invalid."
                )

    @staticmethod
    def _validate_runtime_probe(raw_probe: Any) -> None:
        if not isinstance(raw_probe, Mapping):
            raise RuntimeProvisioningError("The ComfyUI runtime compatibility probe is missing.")
        probe = raw_probe
        version_pattern = re.compile(r"^\d+\.\d+\.\d+(?:[+._A-Za-z0-9-]+)?$")
        imports = probe.get("imports")
        packages = probe.get("packages")
        if (
            not isinstance(probe.get("python"), str)
            or not version_pattern.fullmatch(str(probe["python"]))
            or not isinstance(probe.get("comfyui"), str)
            or not version_pattern.fullmatch(str(probe["comfyui"]))
            or not isinstance(imports, list)
            or not imports
            or len(imports) != len(set(imports))
            or not all(isinstance(item, str) and bool(item) for item in imports)
            or not isinstance(packages, Mapping)
            or not packages
            or not all(
                isinstance(name, str) and bool(name) and isinstance(version, str) and bool(version)
                for name, version in packages.items()
            )
        ):
            raise RuntimeProvisioningError("The ComfyUI runtime compatibility probe is invalid.")

    @staticmethod
    def _validate_https_url(raw_url: Any) -> None:
        if not isinstance(raw_url, str):
            raise RuntimeProvisioningError("The runtime source URL is invalid.")
        parsed = urlparse(raw_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise RuntimeProvisioningError("The runtime source URL is invalid.")

    @staticmethod
    def _positive_int(raw_value: Any) -> int:
        if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value <= 0:
            raise RuntimeProvisioningError("The runtime manifest requires a positive integer.")
        return raw_value

    @staticmethod
    def _safe_component(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
        if not cleaned:
            raise RuntimeProvisioningError("The runtime identifier is invalid.")
        return cleaned

    @staticmethod
    def _sha256_file(
        path: Path,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                if cancel_requested and cancel_requested():
                    raise RuntimeVerificationCancelled
                digest.update(chunk)
        return digest.hexdigest()
