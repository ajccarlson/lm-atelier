from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import httpx
import pytest

import local_lm.runtime_provisioning as runtime_provisioning
from local_lm.config import Settings
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import (
    RuntimeProvisioner,
    RuntimeProvisioningError,
    RuntimeVerificationCancelled,
    default_engine_manifest_path,
)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _inventory_sha256(identities: list[str]) -> str:
    canonical = ("\n".join(sorted(identities)) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_manifest(
    path: Path,
    *,
    llama_content: bytes,
    llama_sha256: str | None = None,
    comfy_content: bytes | None = None,
    comfy_release: str = "v-test",
    comfy_version: str = "0.28.0",
) -> None:
    llama_hash = llama_sha256 or hashlib.sha256(llama_content).hexdigest()
    comfy_hash = hashlib.sha256(comfy_content).hexdigest() if comfy_content else "0" * 64
    comfy_size = len(comfy_content) if comfy_content else 1
    comfy_identity = "example-1.0.dist-info"
    review = {
        "schema_version": 1,
        "release": comfy_release,
        "assets": {
            "test-platform": {
                "source_asset_url": "https://runtime.test/comfy-runtime.zip",
                "source_asset_sha256": comfy_hash,
                "inventory_count": 1,
                "inventory_sha256": _inventory_sha256([comfy_identity]),
                "distributions": [
                    {
                        "dist_info": comfy_identity,
                        "name": "example",
                        "version": "1.0",
                        "license": "MIT",
                        "license_source": "https://runtime.test/example-license",
                    }
                ],
            }
        },
    }
    review_path = path.parent / "runtime-reviews" / f"comfyui-{comfy_release}.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_bytes = (json.dumps(review, indent=2) + "\n").encode()
    review_path.write_bytes(review_bytes)
    payload = {
        "schema_version": 2,
        "updated_at": "2026-07-25",
        "engines": {
            "llama.cpp": {
                "pinned_release": "b-test",
                "distribution": "external",
                "license": "MIT",
                "runtime_assets": {
                    "test-platform": {
                        "url": "https://runtime.test/llama-runtime.zip",
                        "sha256": llama_hash,
                        "size_bytes": len(llama_content),
                        "required_free_bytes": 1,
                        "archive_type": "zip",
                        "executable": "llama-server.exe",
                    }
                },
            },
            "vllm": {
                "pinned_release": "v-test",
                "distribution": "external",
                "license": "Apache-2.0",
                "security_status": "blocked",
                "security_message": "The test vLLM bundle is pending dependency review.",
                "runtime_assets": {},
            },
            "comfyui": {
                "pinned_release": comfy_release,
                "distribution": "external-gpl-3.0",
                "license": "GPL-3.0-only",
                "security_review": {
                    "reviewed_at": "2026-07-25",
                    "release_is_immutable": True,
                    "package_audit": "Test fixture dependency review.",
                    "upstream_advisories": ["https://runtime.test/security-advisories"],
                },
                "runtime_assets": {
                    "test-platform": {
                        "url": "https://runtime.test/comfy-runtime.zip",
                        "sha256": comfy_hash,
                        "size_bytes": comfy_size,
                        "required_free_bytes": 1,
                        "archive_type": "zip",
                        "executable": "python/python.exe",
                        "directory": "ComfyUI",
                        "dependency_inventory_count": 1,
                        "dependency_inventory_sha256": _inventory_sha256([comfy_identity]),
                        "dependency_review": {
                            "file": f"runtime-reviews/comfyui-{comfy_release}.json",
                            "sha256": hashlib.sha256(review_bytes).hexdigest(),
                            "asset_key": "test-platform",
                        },
                        "security_overlays": [],
                        "runtime_probe": {
                            "python": "3.13.14",
                            "comfyui": comfy_version,
                            "imports": ["example"],
                            "packages": {"example": "1.0"},
                        },
                    }
                },
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


async def test_runtime_download_resumes_verifies_and_persists(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"verified executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    partial = settings.download_dir / "runtimes" / "llama.cpp-b-test-llama-runtime.zip.part"
    partial.parent.mkdir(parents=True)
    split = len(content) // 3
    partial.write_bytes(content[:split])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["range"] == f"bytes={split}-"
        return httpx.Response(
            206,
            headers={"content-range": f"bytes {split}-{len(content) - 1}/{len(content)}"},
            content=content[split:],
        )

    environment: dict[str, str] = {}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment=environment,
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert len(requests) == 1
    assert status.state == "ready"
    assert status.managed is True
    assert settings.llama_executable
    assert settings.llama_executable.read_bytes() == b"verified executable"
    assert environment["LOCAL_LM_CHAT_ENGINE"] == "llama.cpp"
    assert environment["LOCAL_LM_LLAMA_EXECUTABLE"] == str(settings.llama_executable)
    saved = json.loads(runtime_config_path(settings.data_dir).read_text(encoding="utf-8"))
    assert saved["LOCAL_LM_LLAMA_EXECUTABLE"] == str(settings.llama_executable)
    assert not partial.exists()
    assert not list((settings.download_dir / "runtimes").glob("*.zip"))


async def test_runtime_start_keeps_background_progress_typed(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"runtime"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    async with httpx.AsyncClient() as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
        )

        async def observe_background_progress(
            engine: runtime_provisioning.RuntimeName,
        ) -> runtime_provisioning.RuntimeStatus:
            definition = provisioner._definition(engine)
            return provisioner._status(
                engine,
                definition,
                state="installing",
                supported=True,
                message="Downloading runtime.",
            )

        monkeypatch.setattr(provisioner, "provision", observe_background_progress)
        started = provisioner.start("llama.cpp")
        background = provisioner._tasks["llama.cpp"]

        assert started.progress_json is not None
        assert started.progress_json.stage == "preparing download"
        progressed = await background
        assert progressed.progress_json is not None
        assert progressed.progress_json.stage == "downloading runtime"
        await provisioner.close()


async def test_runtime_download_uses_shared_exact_byte_progress(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"runtime"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    async with httpx.AsyncClient() as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
        )
        provisioner._states["llama.cpp"] = provisioner.status("llama.cpp")
        definition = provisioner._definition("llama.cpp")

        provisioner._set_download_progress("llama.cpp", definition, 1_000, 250)
        status = provisioner.status("llama.cpp")

    assert status.progress == pytest.approx(0.25)
    assert status.progress_json
    assert status.progress_json.version == 2
    assert status.progress_json.stage_progress == pytest.approx(0.25)
    assert status.progress_json.completed_units == 250
    assert status.progress_json.total_units == 1_000
    assert status.progress_json.unit == "bytes"


async def test_runtime_checksum_failure_removes_untrusted_partial(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    expected = _zip_bytes({"llama-server.exe": b"expected"})
    corrupted = b"x" * len(expected)
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=expected)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=corrupted)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        with pytest.raises(RuntimeProvisioningError, match="SHA-256"):
            await provisioner.ensure("llama.cpp")
        assert provisioner.status("llama.cpp").state == "failed"
        await provisioner.close()

    assert not list((settings.download_dir / "runtimes").glob("*.part"))
    assert settings.llama_executable is None


async def test_runtime_archive_cannot_escape_install_root(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes(
        {
            "../outside.exe": b"unsafe",
            "llama-server.exe": b"otherwise valid",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        with pytest.raises(RuntimeProvisioningError, match="unsafe path"):
            await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert not (settings.data_dir / "outside.exe").exists()
    assert not (tmp_path / "outside.exe").exists()


def test_7z_listing_rejects_links_and_uncompressed_size_overflow() -> None:
    with pytest.raises(RuntimeProvisioningError, match="regular files and directories"):
        RuntimeProvisioner._validate_7z_listing(
            "runtime/link.dll\n",
            "lrwxrwxrwx  0 0  0  0 Jan 01  2026 runtime/link.dll -> outside.dll\n",
            max_entries=10,
            max_uncompressed_bytes=100,
        )

    with pytest.raises(RuntimeProvisioningError, match="allowed size"):
        RuntimeProvisioner._validate_7z_listing(
            "runtime/first.dll\nruntime/second.dll\n",
            (
                "-rw-r--r--  0 0  0  60 Jan 01  2026 runtime/first.dll\n"
                "-rw-r--r--  0 0  0  60 Jan 01  2026 runtime/second.dll\n"
            ),
            max_entries=10,
            max_uncompressed_bytes=100,
        )


async def test_explicit_external_runtime_is_never_replaced(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"managed"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    external = tmp_path / "external-llama-server.exe"
    external.write_bytes(b"external")
    settings.llama_executable = external
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("configured external runtimes must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert status.state == "ready"
    assert status.managed is False
    assert external.read_bytes() == b"external"


async def test_external_comfy_archive_is_provisioned_without_bundling_it(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    llama_content = _zip_bytes({"llama-server.exe": b"llama"})
    comfy_content = _zip_bytes(
        {
            "python/python.exe": b"python",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"print('comfy')",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(
        manifest,
        llama_content=llama_content,
        comfy_content=comfy_content,
    )
    settings.prepare()
    expected_probe = {
        "python": "3.13.14",
        "comfyui": "0.28.0",
        "packages": {"example": "1.0"},
    }
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
            args[0],
            0,
            stdout=(
                f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(expected_probe)}\n"
            ),
            stderr="",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/comfy-runtime.zip")
        return httpx.Response(200, content=comfy_content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("comfyui")
        assert provisioner.status("comfyui").state == "ready"
        managed_asset = provisioner._manifest["engines"]["comfyui"]["runtime_assets"][
            "test-platform"
        ]
        managed_asset["runtime_probe"]["python"] = "3.13.15"
        assert provisioner.status("comfyui").state == "ready"
        assert provisioner.verify_status("comfyui").state == "missing"
        managed_asset["runtime_probe"]["python"] = "3.13.14"
        assert provisioner.verify_status("comfyui").state == "ready"
        await provisioner.close()

    assert status.state == "ready"
    assert status.distribution == "external-gpl-3.0"
    assert settings.comfy_executable
    assert settings.comfy_executable.read_bytes() == b"python"
    assert settings.comfy_directory
    assert (settings.comfy_directory / "main.py").is_file()


async def test_comfy_upgrade_carries_only_managed_registry_nodes(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    llama_content = _zip_bytes({"llama-server.exe": b"llama"})
    comfy_content = _zip_bytes(
        {
            "python/python.exe": b"python",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"print('comfy')",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(
        manifest,
        llama_content=llama_content,
        comfy_content=comfy_content,
        comfy_release="v-old",
        comfy_version="0.28.0",
    )
    settings.prepare()
    probe = {"python": "3.13.14", "comfyui": "0.28.0", "packages": {"example": "1.0"}}
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
            args[0],
            0,
            stdout=f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(probe)}\n",
            stderr="",
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=comfy_content))
    ) as client:
        first = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await first.ensure("comfyui")).state == "ready"
        await first.close()

        assert settings.comfy_directory is not None
        old_directory = settings.comfy_directory
        managed = old_directory / "custom_nodes" / "lm-atelier-registry_example"
        managed_manual = old_directory / "custom_nodes" / "lm-atelier-node_example"
        unmanaged = old_directory / "custom_nodes" / "manually-installed-node"
        managed.mkdir(parents=True)
        managed_manual.mkdir()
        unmanaged.mkdir()
        (managed / "node.py").write_bytes(b"managed node")
        (managed_manual / "node.py").write_bytes(b"managed manual node")
        (unmanaged / "node.py").write_bytes(b"manual node")
        unrelated_release = settings.data_dir / "runtimes" / "comfyui" / "v-unrelated"
        unrelated = unrelated_release / "ComfyUI" / "custom_nodes" / "lm-atelier-registry_unrelated"
        unrelated.mkdir(parents=True)
        (unrelated / "node.py").write_bytes(b"unrelated node")
        (unrelated_release / ".lm-atelier-runtime.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "engine": "comfyui",
                    "release": "v-unrelated",
                }
            ),
            encoding="utf-8",
        )

        _write_manifest(
            manifest,
            llama_content=llama_content,
            comfy_content=comfy_content,
            comfy_release="v-new",
            comfy_version="0.30.0",
        )
        probe["comfyui"] = "0.30.0"
        second = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await second.ensure("comfyui")).state == "ready"
        await second.close()

    assert settings.comfy_directory is not None
    assert settings.comfy_directory != old_directory
    restored = settings.comfy_directory / "custom_nodes" / managed.name
    assert (restored / "node.py").read_bytes() == b"managed node"
    restored_manual = settings.comfy_directory / "custom_nodes" / managed_manual.name
    assert (restored_manual / "node.py").read_bytes() == b"managed manual node"
    assert not (settings.comfy_directory / "custom_nodes" / unmanaged.name).exists()
    assert not (settings.comfy_directory / "custom_nodes" / unrelated.name).exists()
    assert (managed / "node.py").read_bytes() == b"managed node"


async def test_startup_restore_carries_registry_nodes_into_a_preinstalled_release(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    llama_content = _zip_bytes({"llama-server.exe": b"llama"})
    comfy_content = _zip_bytes(
        {
            "python/python.exe": b"python",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"print('comfy')",
        }
    )
    manifest = tmp_path / "engines.json"
    settings.prepare()
    probe = {"python": "3.13.14", "comfyui": "0.28.0", "packages": {"example": "1.0"}}
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
            args[0],
            0,
            stdout=f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(probe)}\n",
            stderr="",
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=comfy_content))
    ) as client:
        _write_manifest(
            manifest,
            llama_content=llama_content,
            comfy_content=comfy_content,
            comfy_release="v-old",
            comfy_version="0.28.0",
        )
        old = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await old.ensure("comfyui")).state == "ready"
        await old.close()
        assert settings.comfy_directory is not None
        assert settings.comfy_executable is not None
        old_directory = settings.comfy_directory
        old_executable = settings.comfy_executable

        _write_manifest(
            manifest,
            llama_content=llama_content,
            comfy_content=comfy_content,
            comfy_release="v-new",
            comfy_version="0.30.0",
        )
        probe["comfyui"] = "0.30.0"
        preinstall = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await preinstall.ensure("comfyui")).state == "ready"
        await preinstall.close()
        assert settings.comfy_directory is not None
        new_directory = settings.comfy_directory

        managed = old_directory / "custom_nodes" / "lm-atelier-registry_late-renewal"
        managed_manual = old_directory / "custom_nodes" / "lm-atelier-node_late-review"
        unmanaged = old_directory / "custom_nodes" / "manual-node"
        managed.mkdir(parents=True)
        managed_manual.mkdir()
        unmanaged.mkdir()
        (managed / "node.py").write_bytes(b"renewed after preinstall")
        (managed_manual / "node.py").write_bytes(b"reviewed after preinstall")
        (unmanaged / "node.py").write_bytes(b"unmanaged")
        assert not (new_directory / "custom_nodes" / managed.name).exists()

        # Recreate the real restart boundary: startup constructs the
        # provisioner while the worker still uses the old release, then another
        # startup path advances the mutable settings object before background
        # verification reaches ComfyUI.
        settings.comfy_directory = old_directory
        settings.comfy_executable = old_executable
        environment: dict[str, str] = {}
        restored = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment=environment,
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        settings.comfy_directory = new_directory
        task = restored.start_restore()
        assert task is not None
        await task
        assert restored.status("comfyui").state == "ready"
        await restored.close()

    assert settings.comfy_directory == new_directory
    copied = new_directory / "custom_nodes" / managed.name
    assert (copied / "node.py").read_bytes() == b"renewed after preinstall"
    copied_manual = new_directory / "custom_nodes" / managed_manual.name
    assert (copied_manual / "node.py").read_bytes() == b"reviewed after preinstall"
    assert not (new_directory / "custom_nodes" / unmanaged.name).exists()
    assert environment["LOCAL_LM_COMFY_DIRECTORY"] == str(new_directory)

    # A repeated recovery with the same exact bytes is a no-op rather than a
    # false conflict; changed bytes still compare by the bounded tree digest.
    settings.comfy_directory = old_directory
    settings.comfy_executable = old_executable
    repeat = RuntimeProvisioner(
        settings,
        manifest_path=manifest,
        environment={},
        platform_key="test-platform",
        allowed_download_hosts={"runtime.test"},
    )
    settings.comfy_directory = new_directory
    task = repeat.start_restore()
    assert task is not None
    await task
    assert repeat.status("comfyui").state == "ready"
    await repeat.close()

    original = (copied / "node.py").read_bytes()
    changed = b"x" * len(original)
    assert changed != original
    (copied / "node.py").write_bytes(changed)
    settings.comfy_directory = old_directory
    settings.comfy_executable = old_executable
    conflict = RuntimeProvisioner(
        settings,
        manifest_path=manifest,
        environment={},
        platform_key="test-platform",
        allowed_download_hosts={"runtime.test"},
    )
    settings.comfy_directory = new_directory
    task = conflict.start_restore()
    assert task is not None
    await task
    assert conflict.status("comfyui").state == "missing"
    assert (copied / "node.py").read_bytes() == changed
    await conflict.close()


def test_managed_registry_copy_budget_is_shared_across_folders(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "node.py").write_bytes(b"a")
    (second / "node.py").write_bytes(b"b")

    entries, copied_bytes = RuntimeProvisioner._copy_managed_registry_tree(
        first,
        tmp_path / "first-copy",
        max_entries=1,
        max_bytes=2,
    )
    assert (entries, copied_bytes) == (1, 1)
    with pytest.raises(RuntimeProvisioningError, match="unsupported shape"):
        RuntimeProvisioner._copy_managed_registry_tree(
            second,
            tmp_path / "second-copy",
            max_entries=1 - entries,
            max_bytes=2 - copied_bytes,
        )


def _provisioner(settings, tmp_path: Path):  # type: ignore[no-untyped-def]
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=_zip_bytes({"llama-server.exe": b"llama"}))
    settings.prepare()
    return RuntimeProvisioner(
        settings,
        manifest_path=manifest,
        environment={},
        platform_key="test-platform",
        allowed_download_hosts={"runtime.test"},
    )


def test_integrity_walk_refuses_a_link_that_escapes_the_runtime(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    """Only a reparse point can point elsewhere, so only it is containment-checked."""
    provisioner = _provisioner(settings, tmp_path)
    root = tmp_path / "runtime-root"
    (root / "inner").mkdir(parents=True)
    (root / "inner" / "real.bin").write_bytes(b"payload")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"elsewhere")
    try:
        (root / "inner" / "escape.bin").symlink_to(outside)
    except OSError:
        pytest.skip("filesystem links are unavailable in this test environment")

    with pytest.raises(RuntimeProvisioningError, match="escape its install directory"):
        provisioner._integrity_file_map(root)


def test_integrity_walk_refuses_a_linked_directory(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    provisioner = _provisioner(settings, tmp_path)
    root = tmp_path / "runtime-root"
    (root / "inner").mkdir(parents=True)
    target = root / "inner" / "target"
    target.mkdir()
    try:
        (root / "inner" / "linked").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem links are unavailable in this test environment")

    with pytest.raises(RuntimeProvisioningError, match="unsupported dependency directory link"):
        provisioner._integrity_file_map(root)


def test_integrity_walk_lists_every_ordinary_file(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    provisioner = _provisioner(settings, tmp_path)
    root = tmp_path / "runtime-root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "one.bin").write_bytes(b"1")
    (root / "a" / "b" / "two.bin").write_bytes(b"22")
    # Mutable areas and the marker stay excluded, as before.
    (root / "temp").mkdir()
    (root / "temp" / "scratch.bin").write_bytes(b"ignored")
    (root / ".lm-atelier-runtime.json").write_text("{}", encoding="utf-8")

    found = provisioner._integrity_file_map(root)
    progress: list[tuple[int, int]] = []
    hashes = provisioner._runtime_file_hashes(
        root,
        {"executable": "a/one.bin"},
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert set(found) == {"a/one.bin", "a/b/two.bin"}
    assert set(hashes) == set(found)
    assert progress[-1] == (2, 2)


async def test_installation_reclaims_staging_left_by_a_failed_attempt(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    """A failed extraction can strand gigabytes; the next attempt must reclaim it."""
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    stranded = settings.data_dir / "runtimes" / "llama.cpp" / ".b0000.partial-deadbeef"
    stranded.mkdir(parents=True)
    (stranded / "half-extracted.bin").write_bytes(b"partial")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert status.state == "ready"
    assert not stranded.exists()
    assert list((settings.data_dir / "runtimes" / "llama.cpp").glob(".*.partial-*")) == []


async def test_managed_runtime_integrity_change_is_not_reported_ready(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes(
        {
            "llama-server.exe": b"verified executable",
            "backend.dll": b"verified dependency",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await provisioner.ensure("llama.cpp")).state == "ready"
        assert settings.llama_executable
        marker = json.loads(
            (settings.llama_executable.parent / ".lm-atelier-runtime.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(marker["files"]) == {"backend.dll", "llama-server.exe"}
        (settings.llama_executable.parent / "backend.dll").write_bytes(b"changed dependency")

        # Ordinary polling remains cheap for the lifetime of a process. A new
        # process performs the full integrity verification and rejects changes.
        assert provisioner.status("llama.cpp").state == "ready"
        await provisioner.close()
        restarted = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert restarted.status("llama.cpp").state == "installing"
        restore = restarted.start_restore()
        assert restore is not None
        await restore
        assert restarted.status("llama.cpp").state == "missing"
        await restarted.close()


async def test_same_release_asset_correction_replaces_only_the_owned_runtime(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    original = _zip_bytes({"llama-server.exe": b"original executable"})
    corrected = _zip_bytes({"llama-server.exe": b"corrected executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=original)
    settings.prepare()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=original))
    ) as client:
        first = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        await first.ensure("llama.cpp")
        await first.close()

    _write_manifest(manifest, llama_content=corrected)
    requests: list[httpx.Request] = []

    def corrected_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=corrected)

    async with httpx.AsyncClient(transport=httpx.MockTransport(corrected_handler)) as client:
        second = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        restore = second.start_restore()
        assert restore is not None
        await restore
        assert second.status("llama.cpp").state == "missing"
        status = await second.ensure("llama.cpp")
        await second.close()

    assert len(requests) == 1
    assert status.state == "ready"
    assert settings.llama_executable
    assert settings.llama_executable.read_bytes() == b"corrected executable"


async def test_managed_runtime_verification_starts_in_background(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"verified executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await first.ensure("llama.cpp")).state == "ready"
        await first.close()

        started = threading.Event()
        release = threading.Event()
        original = RuntimeProvisioner._managed_marker_matches

        def delayed_verification(*args, **kwargs):  # type: ignore[no-untyped-def]
            started.set()
            assert release.wait(timeout=5)
            return original(*args, **kwargs)

        monkeypatch.setattr(RuntimeProvisioner, "_managed_marker_matches", delayed_verification)
        restarted = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )

        assert not started.is_set()
        assert restarted.status("llama.cpp").state == "installing"
        restore = restarted.start_restore()
        assert restore is not None
        assert await asyncio.to_thread(started.wait, 1)
        assert restarted.status("llama.cpp").state == "installing"

        ensure = asyncio.create_task(restarted.ensure("llama.cpp"))
        await asyncio.sleep(0)
        assert not ensure.done()
        release.set()
        assert (await ensure).state == "ready"
        await restore
        await restarted.close()


async def test_managed_runtime_verification_stops_cleanly_on_close(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"verified executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=content))
    ) as client:
        first = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        await first.ensure("llama.cpp")
        await first.close()

        started = threading.Event()

        def cancellable_verification(*_args, **kwargs):  # type: ignore[no-untyped-def]
            cancel_requested = kwargs["cancel_requested"]
            started.set()
            while not cancel_requested():
                time.sleep(0.01)
            raise RuntimeVerificationCancelled

        monkeypatch.setattr(
            RuntimeProvisioner,
            "_managed_marker_matches",
            cancellable_verification,
        )
        restarted = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        restore = restarted.start_restore()
        assert restore is not None
        assert await asyncio.to_thread(started.wait, 1)

        await asyncio.wait_for(restarted.close(), timeout=1)
        assert restore.done()


def test_platform_selection_gates_nvidia_runtime_generations(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime_provisioning.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_provisioning.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (610, 12)),
    )
    assert RuntimeProvisioner._platform_key("llama.cpp") == "windows-x86_64-nvidia"
    assert RuntimeProvisioner._platform_key("comfyui") == "windows-x86_64-nvidia-cu13"

    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (570, 6)),
    )
    assert RuntimeProvisioner._platform_key("comfyui") == "windows-x86_64-nvidia-cu126"

    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (550, 8)),
    )
    assert RuntimeProvisioner._platform_key("comfyui") == "windows-x86_64"


def test_platform_selection_advertises_the_pinned_ubuntu_nvidia_chat_asset(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime_provisioning.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_provisioning.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        runtime_provisioning.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu"},
    )
    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (610, 12)),
    )

    assert RuntimeProvisioner._platform_key("llama.cpp") == "ubuntu-x86_64-nvidia"
    assert RuntimeProvisioner._platform_key("comfyui") == "ubuntu-x86_64-nvidia"


def test_nvidia_probe_parses_driver_and_compute_capability(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime_provisioning.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
            args[0],
            0,
            stdout="610.74, 12.0\n",
            stderr="",
        ),
    )

    assert RuntimeProvisioner._nvidia_runtime_info() == (610, 12)


async def test_blocked_runtime_security_status_prevents_automatic_download(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["engines"]["comfyui"]["security_status"] = "blocked"
    payload["engines"]["comfyui"]["security_message"] = (
        "Automatic setup is paused because dependency security review failed."
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked runtimes must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = provisioner.status("comfyui")
        assert status.state == "unsupported"
        assert status.security_status == "blocked"
        with pytest.raises(RuntimeProvisioningError, match="security review failed"):
            await provisioner.ensure("comfyui")
        await provisioner.close()


async def test_blocked_runtime_without_assets_is_reported_before_asset_selection(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked runtimes without assets must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = provisioner.status("vllm")
        assert status.state == "unsupported"
        assert status.supported is False
        assert status.security_status == "blocked"
        assert "pending dependency review" in status.message
        with pytest.raises(RuntimeProvisioningError, match="pending dependency review"):
            await provisioner.ensure("vllm")
        await provisioner.close()


async def test_asset_level_security_block_prevents_automatic_download(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    asset = payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]
    asset["security_status"] = "blocked"
    asset["security_message"] = "The selected compatibility tier is not audited."
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked runtime assets must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = provisioner.status("comfyui")
        assert status.state == "unsupported"
        assert status.security_status == "blocked"
        assert status.security_message == asset["security_message"]
        with pytest.raises(RuntimeProvisioningError, match="not audited"):
            await provisioner.ensure("comfyui")
        await provisioner.close()


async def test_security_overlay_requires_exact_files_and_rewrites_deterministically(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    overlay_files = {
        "LICENSE.txt": b"Python license",
        "python.exe": b"patched python",
        "python313._pth": b"python313.zip\n.\n",
    }
    overlay = {
        "name": "Test Python security overlay",
        "url": "https://runtime.test/python-overlay.zip",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "max_uncompressed_bytes": 1024,
        "max_entries": 8,
        "archive_type": "zip",
        "target_directory": "runtime/python",
        "compatibility_tag": "cp313-win_amd64-embeddable",
        "license": "Python-2.0",
        "license_file": "LICENSE.txt",
        "expected_files": {
            name: hashlib.sha256(value).hexdigest() for name, value in overlay_files.items()
        },
        "rewrite_files": {"python313._pth": "../ComfyUI\npython313.zip\n.\nimport site\n"},
    }
    archive = tmp_path / "overlay.zip"
    archive.write_bytes(_zip_bytes(overlay_files))
    install_root = tmp_path / "install"
    target = install_root / "runtime" / "python"
    target.mkdir(parents=True)

    async with httpx.AsyncClient() as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        provisioner._apply_security_overlay(install_root, overlay, archive)
        assert (target / "python.exe").read_bytes() == b"patched python"
        assert (target / "python313._pth").read_bytes() == (
            b"../ComfyUI\npython313.zip\n.\nimport site\n"
        )

        unexpected = tmp_path / "unexpected-overlay.zip"
        unexpected.write_bytes(_zip_bytes({**overlay_files, "unreviewed.dll": b"unexpected"}))
        with pytest.raises(RuntimeProvisioningError, match="inventory"):
            provisioner._apply_security_overlay(
                install_root,
                overlay,
                unexpected,
            )
        await provisioner.close()


async def test_runtime_contract_rejects_inventory_and_probe_drift(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    install_root = tmp_path / "staged"
    executable = install_root / "python" / "python.exe"
    site_packages = executable.parent / "Lib" / "site-packages"
    identity = "example-1.0.dist-info"
    (site_packages / identity).mkdir(parents=True)
    executable.write_bytes(b"python")
    comfy_directory = install_root / "ComfyUI"
    comfy_directory.mkdir()
    (comfy_directory / "main.py").write_text("", encoding="utf-8")
    probe = {
        "python": "3.13.14",
        "comfyui": "0.28.0",
        "imports": ["example"],
        "packages": {"example": "1.0"},
    }
    asset = {
        "dependency_inventory_count": 1,
        "dependency_inventory_sha256": _inventory_sha256([identity]),
        "runtime_probe": probe,
    }
    installed = {
        "executable": executable,
        "directory": comfy_directory,
    }
    expected_result = {
        "python": probe["python"],
        "comfyui": probe["comfyui"],
        "packages": probe["packages"],
    }
    probe_result = runtime_provisioning.subprocess.CompletedProcess(
        [str(executable)],
        0,
        stdout=(f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(expected_result)}\n"),
        stderr="",
    )
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: probe_result,  # noqa: ARG005
    )

    async with httpx.AsyncClient() as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        provisioner._verify_runtime_contract(install_root, asset, installed)

        (site_packages / "unreviewed-2.0.dist-info").mkdir()
        with pytest.raises(RuntimeProvisioningError, match="inventory"):
            provisioner._verify_runtime_contract(install_root, asset, installed)
        (site_packages / "unreviewed-2.0.dist-info").rmdir()

        drifted_result = {
            **expected_result,
            "python": "3.13.15",
        }
        monkeypatch.setattr(
            runtime_provisioning.subprocess,
            "run",
            lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
                args[0],
                0,
                stdout=(
                    f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(drifted_result)}\n"
                ),
                stderr="",
            ),
        )
        with pytest.raises(RuntimeProvisioningError, match="versions"):
            provisioner._verify_runtime_contract(install_root, asset, installed)
        await provisioner.close()


def test_comfy_manifest_fails_closed_when_audit_contract_is_omitted(
    tmp_path: Path,
) -> None:
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    asset = payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]

    del asset["dependency_review"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeProvisioningError, match="review reference"):
        RuntimeProvisioner._read_manifest(manifest)

    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]["runtime_probe"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeProvisioningError, match="compatibility probe"):
        RuntimeProvisioner._read_manifest(manifest)

    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]["security_overlays"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeProvisioningError, match="overlay list"):
        RuntimeProvisioner._read_manifest(manifest)


async def test_pinned_comfy_review_accounts_for_every_distribution_and_license(
    settings: Settings,
) -> None:
    manifest_path = default_engine_manifest_path()
    provisioner = RuntimeProvisioner(
        settings,
        manifest_path=manifest_path,
        platform_key="windows-x86_64-nvidia-cu13",
    )
    definition = provisioner._definition("comfyui")
    assets = definition["runtime_assets"]
    selected_asset = provisioner._asset("comfyui", definition)
    assert selected_asset is not None
    selected_review = selected_asset["dependency_review"]
    review_path = manifest_path.parent / selected_review["file"]
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)

    assert definition["pinned_release"] == review["release"] == "v0.30.0"
    assert selected_review["file"] == "runtime-reviews/comfyui-v0.30.0.json"
    assert selected_review["asset_key"] == "windows-x86_64-nvidia-cu13"
    assert hashlib.sha256(review_bytes).hexdigest() == selected_review["sha256"]
    reviewed_selected_asset = review["assets"][selected_review["asset_key"]]
    assert reviewed_selected_asset["source_asset_url"] == selected_asset["url"]
    assert reviewed_selected_asset["source_asset_sha256"] == selected_asset["sha256"]
    assert reviewed_selected_asset["source_asset_size_bytes"] == selected_asset["size_bytes"]
    assert review["vulnerability_audit"] == {
        "tool": "pip-audit 2.10.1",
        "service": "OSV",
        "dependency_count": 89,
        "known_vulnerabilities": 0,
        # Both archives were downloaded and hashed for this release, so the
        # inventory comes from the payloads rather than from their indexes.
        "requirements_source": (
            "Exact dist-info identities from both downloaded portable archives; "
            "CUDA local version suffixes were normalized only for advisory lookup."
        ),
        "advisory_sources": [
            "https://github.com/Comfy-Org/ComfyUI/security/advisories",
            "https://github.com/pytorch/pytorch/security/advisories",
            "https://docs.python.org/3.13/whatsnew/changelog.html",
            "https://docs.python.org/3.12/whatsnew/changelog.html",
        ],
    }
    expected_review_hash = hashlib.sha256(review_bytes).hexdigest()
    expected_inventory = {
        "windows-x86_64-nvidia-cu13": (
            "e71912637473513109c05d7cb0dee99d6f1864f13ce26ff2b468e7ad89025438"
        ),
        "windows-x86_64-nvidia-cu126": (
            "09d226f9097d598a256fef80b60b82b59933c5b3ee49197fd1ac5d6473e59cac"
        ),
    }
    for asset_key, inventory_hash in expected_inventory.items():
        asset = assets[asset_key]
        reviewed_asset = review["assets"][asset_key]
        distributions = reviewed_asset["distributions"]
        assert asset["dependency_inventory_count"] == len(distributions) == 89
        assert asset["dependency_inventory_sha256"] == inventory_hash
        assert asset["dependency_review"]["sha256"] == expected_review_hash
        assert all(
            distribution["dist_info"]
            and distribution["name"]
            and distribution["version"]
            and distribution["license"]
            and distribution["license_source"].startswith("https://")
            for distribution in distributions
        )
        canonical = (
            "\n".join(sorted(distribution["dist_info"] for distribution in distributions)) + "\n"
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == inventory_hash

    cu13 = assets["windows-x86_64-nvidia-cu13"]
    cu13_identities = {
        item["dist_info"]: item
        for item in review["assets"]["windows-x86_64-nvidia-cu13"]["distributions"]
    }
    assert cu13_identities["torch-2.13.0+cu130.dist-info"]["license"].startswith("Apache-2.0")
    assert cu13_identities["comfy_aimdo-0.4.11.dist-info"]["license"] == ("GPL-3.0-only")
    assert cu13["runtime_probe"]["python"] == "3.13.14"
    assert cu13["runtime_probe"]["packages"]["torch"] == "2.13.0+cu130"
    overlay = cu13["security_overlays"][0]
    assert overlay["url"] == (
        "https://www.python.org/ftp/python/3.13.14/python-3.13.14-embed-amd64.zip"
    )
    assert overlay["sha256"] == ("90b4e5b9898b72d744650524bff92377c367f44bd5fbd09e3148656c080ad907")
    assert overlay["compatibility_tag"] == "cp313-win_amd64-embeddable"
    assert overlay["license"] == "Python-2.0"
    assert len(overlay["expected_files"]) == 34
    assert assets["windows-x86_64-nvidia-cu126"]["security_status"] == "blocked"
    await provisioner.close()


async def test_comfy_v028_marker_is_invalid_after_production_pin_moves_to_v030(
    settings: Settings,
) -> None:
    provisioner = RuntimeProvisioner(
        settings,
        manifest_path=default_engine_manifest_path(),
        platform_key="windows-x86_64-nvidia-cu13",
    )
    definition = provisioner._definition("comfyui")
    asset = provisioner._asset("comfyui", definition)
    assert asset is not None
    install_root = provisioner._installation_path("comfyui", definition)
    install_root.mkdir(parents=True)
    (install_root / ".managed-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "engine": "comfyui",
                "release": "v0.28.0",
            }
        ),
        encoding="utf-8",
    )

    assert definition["pinned_release"] == "v0.30.0"
    assert install_root.name == "v0.30.0"
    assert not provisioner._managed_marker_owned(install_root, "comfyui", definition)
    assert not provisioner._managed_marker_matches(
        install_root,
        "comfyui",
        definition,
        asset,
    )
    await provisioner.close()
