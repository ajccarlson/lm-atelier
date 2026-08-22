from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from local_lm import __version__

ROOT = Path(__file__).resolve().parents[3]


def test_release_license_normalization_is_fail_closed() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/build-release-metadata.py"))
    normalized_license = namespace["normalized_license"]

    assert normalized_license("Apache 2.0") == "Apache-2.0"
    assert normalized_license("Dual License") == "UNKNOWN"
    assert normalized_license("LGPL") == "UNKNOWN"
    assert normalized_license("BSD License") == "UNKNOWN"
    assert (
        normalized_license(
            None,
            ["License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)"],
        )
        == "LGPL-2.1-or-later"
    )
    assert normalized_license("Proprietary custom terms") == "UNKNOWN"
    assert normalized_license("Apache-2.0 AND made-up-license") == "UNKNOWN"


def test_release_metadata_publish_retries_short_windows_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/build-release-metadata.py"))
    publish_directory = namespace["publish_directory"]
    staging = tmp_path / "metadata.partial"
    output = tmp_path / "metadata"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    original_replace = Path.replace
    replace_attempts = 0
    sleeps: list[float] = []

    def intermittently_locked(source: Path, target: Path) -> Path:
        nonlocal replace_attempts
        if source == staging:
            replace_attempts += 1
            if replace_attempts < 4:
                raise PermissionError("simulated scanner lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", intermittently_locked)
    monkeypatch.setattr(namespace["time"], "sleep", sleeps.append)

    publish_directory(staging, output)

    assert replace_attempts == 4
    assert len(sleeps) == 3
    assert (output / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert not staging.exists()


def test_sbom_serial_distinguishes_platform_builds() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/build-release-metadata.py"))
    sbom_serial = namespace["sbom_serial"]
    common = {
        "root_ref": "pkg:generic/lm-atelier@0.1.7",
        "commit": "a" * 40,
        "generated_at": datetime(2026, 7, 25, tzinfo=UTC),
        "lock_inputs": [{"path": "lock", "sha256": "b" * 64}],
    }

    windows = sbom_serial(
        **common,
        toolchain={
            "platform": {"system": "Windows", "machine": "AMD64"},
            "installer": {"name": "Inno Setup", "version": "6.7.1"},
        },
    )
    linux = sbom_serial(
        **common,
        toolchain={
            "platform": {"system": "Linux", "machine": "x86_64"},
            "installer": {"name": "GNU tar", "version": "1.35"},
        },
    )

    assert windows.startswith("urn:uuid:")
    assert windows != linux


def test_payload_sbom_reconciliation_marks_build_only_components_excluded() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/inventory-frozen-payload.py"))
    reconcile_sbom = namespace["reconcile_sbom"]
    root_ref = "pkg:generic/lm-atelier@0.1.7"
    sbom = {
        "metadata": {
            "component": {"bom-ref": root_ref},
            "properties": [],
        },
        "components": [
            {
                "bom-ref": "pkg:pypi/fastapi@1",
                "name": "FastAPI",
                "version": "1",
                "properties": [{"name": "lm-atelier:ecosystem", "value": "python"}],
            },
            {
                "bom-ref": "pkg:pypi/pytest@9",
                "name": "pytest",
                "version": "9",
                "properties": [{"name": "lm-atelier:ecosystem", "value": "python"}],
            },
            {
                "bom-ref": "pkg:npm/react@19",
                "name": "react",
                "version": "19",
                "properties": [{"name": "lm-atelier:ecosystem", "value": "npm"}],
            },
            {
                "bom-ref": "pkg:npm/unused@1",
                "name": "unused",
                "version": "1",
                "properties": [{"name": "lm-atelier:ecosystem", "value": "npm"}],
            },
        ],
        "dependencies": [],
    }

    counts = reconcile_sbom(sbom, {("fastapi", "1")}, {"react"})

    assert counts == (1, 1, 1, 1)
    assert sbom["components"][0]["scope"] == "required"
    assert sbom["components"][1]["scope"] == "excluded"
    assert sbom["components"][2]["scope"] == "required"
    assert sbom["components"][3]["scope"] == "excluded"
    assert sbom["dependencies"] == [
        {
            "ref": root_ref,
            "dependsOn": ["pkg:npm/react@19", "pkg:pypi/fastapi@1"],
        }
    ]


def test_payload_sbom_adds_frozen_vendored_distribution_and_license(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/inventory-frozen-payload.py"))
    payload = tmp_path / "payload"
    metadata_root = tmp_path / "metadata"
    dist_info = (
        payload / "_internal" / "setuptools" / "_vendor" / "importlib_metadata-8.7.1.dist-info"
    )
    license_path = dist_info / "licenses" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.4",
                "Name: importlib_metadata",
                "Version: 8.7.1",
                "License-Expression: Apache-2.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    license_path.write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    (metadata_root / "third-party-licenses").mkdir(parents=True)
    root_ref = "pkg:generic/lm-atelier@0.1.7"
    sbom = {
        "metadata": {
            "component": {"bom-ref": root_ref},
            "properties": [],
        },
        "components": [],
        "dependencies": [],
    }

    added = namespace["augment_sbom_with_frozen_metadata"](
        sbom,
        payload,
        metadata_root,
    )
    counts = namespace["reconcile_sbom"](
        sbom,
        {("importlib-metadata", "8.7.1")},
        set(),
    )

    assert added == 1
    assert counts == (1, 0, 0, 0)
    component = sbom["components"][0]
    assert component["name"] == "importlib_metadata"
    assert component["licenses"] == [{"expression": "Apache-2.0"}]
    properties = {property_["name"]: property_["value"] for property_ in component["properties"]}
    assert properties["lm-atelier:vendored-by"] == "setuptools"
    assert properties["lm-atelier:frozen-metadata-path"].endswith(
        "importlib_metadata-8.7.1.dist-info/METADATA"
    )
    copied_license_root = (
        metadata_root / "third-party-licenses" / "python" / "importlib_metadata@8.7.1"
    )
    assert len(list(copied_license_root.iterdir())) == 1
    notices = (metadata_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "| python | importlib_metadata | 8.7.1 | Apache-2.0 |" in notices


def test_payload_sbom_rejects_unreviewed_frozen_distribution_license(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/inventory-frozen-payload.py"))
    payload = tmp_path / "payload"
    metadata_root = tmp_path / "metadata"
    dist_info = payload / "_internal" / "vendor-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.4",
                "Name: vendor",
                "Version: 1.0",
                "License-Expression: Proprietary",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dist_info / "LICENSE").write_text("unknown terms", encoding="utf-8")
    (metadata_root / "third-party-licenses").mkdir(parents=True)
    sbom = {"components": []}

    with pytest.raises(RuntimeError, match="unreviewed license metadata"):
        namespace["augment_sbom_with_frozen_metadata"](
            sbom,
            payload,
            metadata_root,
        )


def test_payload_sbom_reads_bundled_npm_packages_from_vite_maps(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/inventory-frozen-payload.py"))
    bundled_node_packages = namespace["bundled_node_packages"]
    assets = tmp_path / "_internal" / "web" / "assets"
    assets.mkdir(parents=True)
    (assets / "index.js.map").write_text(
        json.dumps(
            {
                "version": 3,
                "sources": [
                    "../../../../node_modules/react/index.js",
                    "../../../../node_modules/@tanstack/react-query/build/index.js",
                    "../../src/App.tsx",
                ],
            }
        ),
        encoding="utf-8",
    )

    assert bundled_node_packages(tmp_path) == {
        "react",
        "@tanstack/react-query",
    }


def test_payload_manifest_detects_changes(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/inventory-frozen-payload.py"))
    payload_entries = namespace["payload_entries"]
    verify_inventory = namespace["verify_inventory"]
    payload_root = tmp_path / "application"
    internal = payload_root / "_internal"
    internal.mkdir(parents=True)
    application = payload_root / "lm-atelier"
    application.write_bytes(b"application")
    entries = payload_entries(payload_root)
    inventory = {
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "files": entries,
    }
    (internal / "payload-manifest.json").write_text(
        json.dumps(inventory),
        encoding="utf-8",
    )

    assert verify_inventory(payload_root)["file_count"] == 1
    application.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="does not match"):
        verify_inventory(payload_root)


def _write_test_release_bundle(root: Path, platform_name: str = "windows") -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/verify-release-bundle.py"))
    expected = namespace["_platform_files"](platform_name, "1.2.3")
    source_sha = "a" * 40
    source = {
        "sha": source_sha,
        "commit": source_sha,
        "tag": "v1.2.3",
        "dirty": False,
    }
    root.mkdir()
    for name in expected:
        if not name.startswith("SHA256SUMS-"):
            (root / name).write_bytes(name.encode())
    (root / f"release-manifest-{platform_name}.json").write_text(
        json.dumps(
            {
                "application": "LM Atelier",
                "version": "1.2.3",
                "source": source,
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (root / f"payload-manifest-{platform_name}.json").write_text(
        json.dumps(
            {
                "application": "LM Atelier",
                "version": "1.2.3",
                "source": source,
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (root / f"sbom-{platform_name}.cdx.json").write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "metadata": {
                    "component": {"name": "LM Atelier", "version": "1.2.3"},
                    "properties": [
                        {"name": "lm-atelier:source-commit", "value": source_sha},
                        {"name": "lm-atelier:source-tag", "value": "v1.2.3"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    checksum_name = f"SHA256SUMS-{platform_name}"
    lines = [
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}"
        for name in sorted(expected - {checksum_name})
    ]
    (root / checksum_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("platform_name", ["windows", "linux"])
def test_release_bundle_verification_is_fail_closed(
    tmp_path: Path,
    platform_name: str,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/verify-release-bundle.py"))
    verify_bundle = namespace["verify_bundle"]
    bundle = tmp_path / "bundle"
    _write_test_release_bundle(bundle, platform_name)

    verify_bundle(bundle, (platform_name,), "1.2.3", "a" * 40)

    (bundle / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_bundle(bundle, (platform_name,), "1.2.3", "a" * 40)
    (bundle / "unexpected.txt").unlink()

    checksum = bundle / f"SHA256SUMS-{platform_name}"
    checksum.write_text(
        checksum.read_text(encoding="utf-8") + f"{'0' * 64}  ../outside\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid or duplicate"):
        verify_bundle(bundle, (platform_name,), "1.2.3", "a" * 40)


def test_release_bundle_rejects_mismatched_source_identity(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/verify-release-bundle.py"))
    verify_bundle = namespace["verify_bundle"]
    bundle = tmp_path / "bundle"
    _write_test_release_bundle(bundle)
    manifest = bundle / "release-manifest-windows.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["source"]["sha"] = "b" * 40
    manifest.write_text(json.dumps(document), encoding="utf-8")
    checksum = bundle / "SHA256SUMS-windows"
    checksum_lines = checksum.read_text(encoding="utf-8").splitlines()
    checksum.write_text(
        "\n".join(
            (
                f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}"
                "  release-manifest-windows.json"
                if line.endswith("  release-manifest-windows.json")
                else line
            )
            for line in checksum_lines
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="source is inconsistent"):
        verify_bundle(bundle, ("windows",), "1.2.3", "a" * 40)


@pytest.mark.parametrize(
    ("path", "unsafe"),
    [
        (".env.example", False),
        (".env.production", True),
        ("docs/assets/application-preview.png", False),
        ("private/model.onnx", True),
        ("keys/signing.pem", True),
        ("keys/signing.der", True),
        ("keys/signing.ppk", True),
        ("keys/private.pkcs8", True),
        ("keys/release.asc", True),
        ("keys/keyring.gpg", True),
        ("keys/archive.age", True),
        ("keys/passwords.kdbx", True),
        ("profiles/client.ovpn", True),
        ("profiles/app.mobileprovision", True),
        ("state/chat.db", True),
        ("logs/service.log", True),
        ("release/application.exe", True),
        ("package-lock.json", False),
    ],
)
def test_repository_hygiene_rejects_force_added_artifacts(
    path: str,
    unsafe: bool,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/check-repository-hygiene.py"))
    assert namespace["unsafe_path"](path) is unsafe


def test_secret_scan_finds_keys_without_flagging_hyphenated_names() -> None:
    """A scanner that cries wolf gets disabled, so its edges matter.

    The `sk-` rule matched mid-word before: `mask-feather-out-of-range` and
    `task-scheduler-configuration` both looked like OpenAI keys, which
    failed the gate on ordinary identifiers.
    """

    namespace = runpy.run_path(str(ROOT / "scripts/check-repository-hygiene.py"))
    patterns = namespace["SECRET_PATTERNS"]

    def flagged(value: str) -> bool:
        return any(pattern.search(value) for pattern in patterns)

    # Real credentials are still caught, bare and in context.
    assert flagged("sk-" + "a1b2c3d4e5f6g7h8i9j0")
    assert flagged('OPENAI_KEY="sk-' + "a1b2c3d4e5f6g7h8i9j0" + '"')
    assert flagged("hf_" + "abcdefghijklmnopqrstuvwxyz")
    assert flagged("AKIA" + "ABCDEFGHIJKLMNOP")

    # Ordinary hyphenated identifiers are not credentials.
    assert not flagged("mask-feather-out-of-range")
    assert not flagged("task-scheduler-configuration-error")
    assert not flagged("workflow-asset-binding-plan-changed")


def test_release_metadata_timestamp_honors_source_date_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/build-release-metadata.py"))
    generated_time = namespace["generated_time"]

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1234567890")
    assert generated_time() == datetime.fromtimestamp(1234567890, tz=UTC)

    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    started_at = datetime.now(UTC)
    actual = generated_time()
    finished_at = datetime.now(UTC)
    assert started_at <= actual <= finished_at


def test_release_and_engine_manifests_are_pinned_and_versioned() -> None:
    release = json.loads((ROOT / "packaging/release-manifest.json").read_text())
    engines = json.loads((ROOT / "packaging/engines.json").read_text())

    assert release["version"] == __version__
    assert release["data_policy"]["rollback"]
    assert release["bundled_engines"] is False
    assert release["prerequisites"]["python"] == "Bundled in official installers"
    assert "llama.cpp" in release["runtime_scope"]["windows"]["chat"]
    assert "llama.cpp" in release["runtime_scope"]["linux"]["chat"]
    assert "reviewed compatible NVIDIA runtime" in release["runtime_scope"]["windows"]["media"]
    linux_media_scope = release["runtime_scope"]["linux"]["media"]
    assert "externally configured compatible media engine" in linux_media_scope
    assert "not certified" in release["runtime_scope"]["linux"]["media"]
    assert engines["schema_version"] == 2
    assert engines["engines"]["llama.cpp"]["pinned_release"] != "latest"
    assert engines["engines"]["comfyui"]["pinned_release"] != "latest"
    for engine in engines["engines"].values():
        assert engine["runtime_assets"] or engine["security_status"] == "blocked"
        for asset in engine["runtime_assets"].values():
            assert asset["url"].startswith("https://github.com/")
            assert len(asset["sha256"]) == 64
            assert asset["size_bytes"] > 0
    assert engines["engines"]["comfyui"]["distribution"] == "external-gpl-3.0"
    vllm = engines["engines"]["vllm"]
    assert vllm["security_status"] == "blocked"
    assert vllm["security_message"]
    assert vllm["runtime_assets"] == {}
    comfy = engines["engines"]["comfyui"]
    assert comfy["security_status"] == "checksum-pinned"
    assert comfy["runtime_assets"]["windows-x86_64-nvidia-cu13"]["dependency_inventory_count"] == 89
    assert comfy["runtime_assets"]["windows-x86_64-nvidia-cu126"]["security_status"] == "blocked"
    assert comfy["runtime_assets"]["windows-x86_64-nvidia-cu126"]["security_message"]
    assert "windows-x86_64-nvidia" in engines["engines"]["llama.cpp"]["runtime_assets"]
    assert "ubuntu-x86_64-nvidia" in engines["engines"]["llama.cpp"]["runtime_assets"]
    assert {
        "windows-x86_64-nvidia-cu13",
        "windows-x86_64-nvidia-cu126",
    }.issubset(engines["engines"]["comfyui"]["runtime_assets"])
    assert not any(
        key.startswith(("linux-", "ubuntu-"))
        for key in engines["engines"]["comfyui"]["runtime_assets"]
    )
    assert all(
        engine["certification"] == "hardware-pending" for engine in engines["engines"].values()
    )


def test_application_version_has_one_canonical_source() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/sync-version.py")],
        cwd=ROOT,
        check=True,
    )
    pyproject = (ROOT / "services/api/pyproject.toml").read_text()
    installer = (ROOT / "packaging/windows/LMAtelier.iss").read_text()
    setup = (ROOT / "scripts/setup.sh").read_text()

    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "local_lm/__init__.py"' in pyproject
    assert 'requires-python = ">=3.12,<3.13"' in pyproject
    assert '"pyyaml>=6.0.3,<7"' in pyproject
    assert "sys.version_info[:2] == (3, 12)" in setup
    assert '#define MyAppVersion "' not in installer
    assert "#error MyAppVersion must be supplied" in installer
    assert "#error MyFileVersion must be supplied" in installer


@pytest.mark.skipif(sys.platform == "win32", reason="Bash syntax is checked by Ubuntu CI")
def test_linux_release_scripts_pass_shell_syntax_check() -> None:
    scripts = [
        ROOT / "scripts/build-linux-installer.sh",
        *sorted((ROOT / "packaging/linux").glob("*.sh")),
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_installers_preserve_data_unless_purge_is_explicit() -> None:
    linux = (ROOT / "packaging/linux/frozen-uninstall.sh").read_text()
    windows = (ROOT / "packaging/windows/LMAtelier.iss").read_text()

    assert "--purge-data" in linux
    assert "XDG_DATA_HOME" not in linux
    assert "LOCAL_LM_DATA_DIR" not in linux
    assert 'managed_data_parent="$home_root/.local/share"' in linux
    assert "PurgeDataCheckBox" in windows
    assert r"{localappdata}\LMAtelier\data" in windows
    assert "PurgeDataCheckBox.Checked" in windows
    assert "RedirectionGuard=yes" in windows


def _linux_uninstall_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    home = tmp_path / "home"
    install_root = home / ".local" / "opt" / "lm-atelier"
    install_root.mkdir(parents=True)
    uninstall = install_root / "uninstall.sh"
    shutil.copy2(ROOT / "packaging/linux/frozen-uninstall.sh", uninstall)
    uninstall.chmod(0o755)
    (install_root / ".lm-atelier-install").write_text(
        "lm-atelier-managed-install-v1\n",
        encoding="utf-8",
    )
    data_root = home / ".local" / "share" / "lm-atelier"
    return home, install_root, data_root


@pytest.mark.skipif(sys.platform == "win32", reason="Linux uninstaller behavior")
def test_linux_purge_ignores_caller_selected_data_paths(tmp_path: Path) -> None:
    home, install_root, data_root = _linux_uninstall_fixture(tmp_path)
    data_root.mkdir(parents=True)
    (data_root / "managed").write_text("remove", encoding="utf-8")
    protected_xdg = tmp_path / "protected-xdg"
    protected_local = tmp_path / "protected-local"
    (protected_xdg / "lm-atelier").mkdir(parents=True)
    protected_local.mkdir()
    (protected_xdg / "lm-atelier" / "sentinel").write_text(
        "preserve",
        encoding="utf-8",
    )
    (protected_local / "sentinel").write_text("preserve", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "XDG_DATA_HOME": str(protected_xdg / ".." / "protected-xdg"),
            "LOCAL_LM_DATA_DIR": str(protected_local),
        }
    )
    environment.pop("LM_ATELIER_INSTALL_ROOT", None)

    subprocess.run(
        [str(install_root / "uninstall.sh"), "--purge-data"],
        check=True,
        env=environment,
    )

    assert not install_root.exists()
    assert not data_root.exists()
    assert (protected_xdg / "lm-atelier" / "sentinel").read_text() == "preserve"
    assert (protected_local / "sentinel").read_text() == "preserve"


@pytest.mark.skipif(sys.platform == "win32", reason="Linux uninstaller behavior")
@pytest.mark.parametrize("redirect", ["data-root", "data-parent"])
def test_linux_purge_refuses_symlink_redirects(
    tmp_path: Path,
    redirect: str,
) -> None:
    home, install_root, data_root = _linux_uninstall_fixture(tmp_path)
    protected = tmp_path / "protected"
    protected_data = protected / "lm-atelier"
    protected_data.mkdir(parents=True)
    sentinel = protected_data / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    if redirect == "data-root":
        data_root.parent.mkdir(parents=True)
        data_root.symlink_to(protected_data, target_is_directory=True)
    else:
        data_root.parent.parent.mkdir(parents=True, exist_ok=True)
        data_root.parent.symlink_to(protected, target_is_directory=True)
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment.pop("LM_ATELIER_INSTALL_ROOT", None)

    result = subprocess.run(
        [str(install_root / "uninstall.sh"), "--purge-data"],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to purge" in result.stderr
    assert install_root.exists()
    assert sentinel.read_text() == "preserve"


def test_windows_installer_creates_start_menu_and_application_launchers() -> None:
    installer = (ROOT / "packaging/windows/LMAtelier.iss").read_text()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert r"DefaultDirName={localappdata}\Programs\LM Atelier" in installer
    assert 'Name: "{group}\\LM Atelier"' in installer
    assert 'Filename: "{app}\\{#MyAppExeName}"' in installer
    assert "LM Atelier terminal" in readme


def test_locked_logo_geometry_and_colors_stay_consistent() -> None:
    mark = (ROOT / "docs/assets/lm-atelier-mark.svg").read_text()
    social = (ROOT / "docs/assets/social-preview.svg").read_text()
    app = (ROOT / "apps/web/src/AtelierMark.tsx").read_text()
    icon_builder = (ROOT / "scripts/build-icons.py").read_text()

    light_path = "M43 20h64v300h169v60H43z"
    blue_path = "M125 20l118 109L356 20v360h-58v-92h-85l46-45h39v-82L164 303h-39z"
    for source in (mark, social, app):
        assert light_path in source
        assert blue_path in source
    for color in ("#12110f", "#efe7d8", "#315a9a"):
        assert color in mark
        assert color in social
        assert color in icon_builder


def test_release_workflow_builds_self_contained_platform_installers() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    package = json.loads((ROOT / "package.json").read_text())
    release_template = (ROOT / ".github/RELEASE_TEMPLATE.md").read_text()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    troubleshooting = (ROOT / "docs/TROUBLESHOOTING.md").read_text(encoding="utf-8")
    release_template_text = " ".join(release_template.split())
    readme_text = " ".join(readme.split())
    troubleshooting_text = " ".join(troubleshooting.split())
    windows_build = workflow.split("  windows-installer:", 1)[1].split("  linux-installer:", 1)[0]
    linux_build = workflow.split("  linux-installer:", 1)[1].split("  windows-upgrade-smoke:", 1)[0]
    windows_smoke = workflow.split("  windows-upgrade-smoke:", 1)[1].split(
        "  linux-upgrade-smoke:", 1
    )[0]
    linux_smoke = workflow.split("  linux-upgrade-smoke:", 1)[1].split("  release-candidate:", 1)[0]

    assert "build-windows-installer.ps1" in workflow
    assert "build-linux-installer.sh" in workflow
    assert "smoke-windows-installer.ps1" not in windows_build
    assert "smoke-linux-installer.sh" not in linux_build
    assert "smoke-windows-installer.ps1" in windows_smoke
    assert "smoke-linux-installer.sh" in linux_smoke
    assert "actions/setup-python@" in windows_smoke
    assert "actions/setup-python@" in linux_smoke
    assert "python -m venv .venv" in windows_smoke
    assert "python -m venv .venv" in linux_smoke
    assert "release-manifest-windows.json" in workflow
    assert "release-manifest-linux.json" in workflow
    assert "payload-manifest-windows.json" in workflow
    assert "payload-manifest-linux.json" in workflow
    assert "Assemble release candidate" in workflow
    assert "actions/download-artifact@" in workflow
    assert "actions/attest@" in workflow
    assert "uvx" not in workflow
    assert r".\.venv\Scripts\python.exe -m pip_audit" in workflow
    assert ".venv/bin/python -m pip_audit" in workflow
    for platform in ("windows", "linux"):
        assert f"npm-audit-{platform}.json" in workflow
        assert f"pip-audit-{platform}.json" in workflow
        for scope in ("payload", "metadata", "installer"):
            assert f"gitleaks-{platform}-{scope}.json" in workflow
    assert 'Version = "8.30.1"' in workflow
    assert 'version="8.30.1"' in workflow
    assert "Microsoft Defender" in workflow
    assert "clamscan" in workflow
    assert "provenance-attestation.sigstore.json" in workflow
    assert "previous_tag" in workflow
    assert '"${tag_sha}:packaging/prior-release-checksums.json"' in workflow
    assert "prior-release-checksums.json" in windows_smoke
    assert "prior-release-checksums.json" in linux_smoke
    assert "--pattern SHA256SUMS" not in workflow
    assert "-PreviousInstaller" in windows_smoke
    assert "windows-upgrade-smoke.result == 'success'" in workflow
    assert "linux-upgrade-smoke.result == 'success'" in workflow
    assert "create_draft_release:" in workflow
    assert 'RELEASE_PLATFORM" != "all"' in workflow
    assert 'REPOSITORY_VISIBILITY" != "public"' in workflow
    assert "gh release create" in workflow
    assert "--draft" in workflow
    assert "refusing to overwrite it" in workflow
    assert "--clobber" not in workflow
    assert "scripts/render-release-notes.py" in workflow
    assert workflow.count("compression-level: 0") == 3
    assert '"$RUNNER_TEMP/gitleaks" git . --redact' in ci
    assert '"$RUNNER_TEMP/gitleaks" git . --redact' in workflow
    assert "--no-install-project" in workflow
    assert "npm run e2e:run" in ci
    assert package["scripts"]["e2e:run"] == "node scripts/run-browser-e2e.mjs"
    assert package["scripts"]["e2e"].endswith("npm run e2e:run")
    assert "gitleaks-<platform>-{payload,metadata,installer}.json" in release_template
    assert "npm-audit-<platform>.json" in release_template
    assert "pip-audit-<platform>.json" in release_template
    assert "LM-Atelier-Setup-<version>-windows-x86_64.exe" in readme
    assert "LM-Atelier-Setup-<version>-linux-x86_64.run" in readme
    assert "Managed llama.cpp chat setup is one-click on both installer targets" in readme_text
    assert "Linux image/video require an externally" in readme_text
    assert "reviewed compatible Windows NVIDIA runtime" in release_template_text
    assert "Linux image/video require an" in release_template_text
    assert "automatic engine setup currently covers compatible chat models" in troubleshooting_text
    assert "externally configured compatible media engine" in troubleshooting_text
    assert "windows-2025" in workflow
    assert "ubuntu-24.04" in workflow


def test_release_notes_are_complete_and_fail_closed() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/render-release-notes.py"))
    render = namespace["render"]
    template = (ROOT / ".github/RELEASE_TEMPLATE.md").read_text(encoding="utf-8")
    source_sha = "0123456789abcdef0123456789abcdef01234567"

    notes = render(template, "0.2.0-beta.1", source_sha)
    notes_text = " ".join(notes.split())

    assert "<!--" not in notes
    assert "0.2.0-beta.1" in notes
    assert source_sha in notes
    assert "not code-signed" in notes
    assert notes.count("automated build smoke passed; physical certification pending") == 2
    assert "reviewed compatible Windows NVIDIA runtime" in notes_text
    assert "Linux image/video require an externally configured" in notes_text
    with pytest.raises(ValueError, match="SemVer"):
        render(template, "latest", source_sha)
    with pytest.raises(ValueError, match="commit"):
        render(template, "0.2.0", "short")
    with pytest.raises(ValueError, match="unresolved"):
        render(template + "\n<!-- new review item -->", "0.2.0", source_sha)


def test_prior_release_installer_checksums_are_complete_and_exact() -> None:
    checksums = json.loads((ROOT / "packaging/prior-release-checksums.json").read_text())

    assert set(checksums) == {f"v0.1.{patch}" for patch in range(8)}
    for tag, platforms in checksums.items():
        assert set(platforms) == {"linux-x86_64", "windows-x86_64"}
        for platform, checksum in platforms.items():
            assert len(checksum) == 64, f"{tag} {platform}"
            assert checksum == checksum.lower()
            assert int(checksum, 16) >= 0


def test_workflow_policy_rejects_docker_actions() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/validate-workflows.py"))
    validate_action_pins = namespace["validate_action_pins"]
    workflow_path = Path(".github/workflows/example.yml")

    errors = validate_action_pins(
        workflow_path,
        "steps:\n  - uses: docker://example.invalid/tool@sha256:" + ("a" * 64),
    )

    assert errors == [
        f"{workflow_path}:2: docker actions are prohibited; use a reviewed SHA-pinned action"
    ]


def test_workflow_policy_limits_release_write_permission_to_the_draft_job() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/validate-workflows.py"))
    validate_permissions = namespace["validate_permissions"]
    workflow_path = Path("release.yml")
    base = {"permissions": {"contents": "read"}}

    allowed = {
        **base,
        "jobs": {"draft-release": {"permissions": {"actions": "read", "contents": "write"}}},
    }
    denied = {
        **base,
        "jobs": {"build": {"permissions": {"contents": "write"}}},
    }

    assert validate_permissions(workflow_path, allowed) == []
    assert validate_permissions(workflow_path, denied) == [
        "release.yml: contents permission must not grant 'write' access"
    ]


def test_workflow_policy_rejects_runner_context_in_job_environment() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/validate-workflows.py"))
    validate_environment_contexts = namespace["validate_environment_contexts"]
    workflow_path = Path(".github/workflows/example.yml")

    allowed = {
        "env": {"ROOT": "${{ github.workspace }}/temp"},
        "jobs": {
            "test": {
                "env": {"DATA": "${{ github.workspace }}/temp/data"},
                "steps": [{"env": {"SCRATCH": "${{ runner.temp }}"}}],
            }
        },
    }
    denied = {
        "jobs": {
            "test": {
                "env": {"DATA": "${{ runner.temp }}/data"},
            }
        }
    }

    assert validate_environment_contexts(workflow_path, allowed) == []
    assert validate_environment_contexts(workflow_path, denied) == [
        f"{workflow_path}: job test env DATA cannot use the runner context"
    ]


def test_ci_plan_is_fail_closed_and_audits_dependency_changes() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/ci-plan.py"))
    classify = namespace["classify_develop_changes"]
    requires_windows = namespace["requires_windows_verification"]

    assert classify(["docs/ARCHITECTURE.md", "CONTRIBUTING.md"]) == (
        "documentation",
        False,
    )
    assert classify(["README.md"]) == ("full", False)
    assert classify(["docs/TROUBLESHOOTING.md"]) == ("full", False)
    assert classify(["services/api/local_lm/api.py"]) == ("full", False)
    assert classify(["package-lock.json"]) == ("full", True)
    assert classify([]) == ("full", False)
    assert requires_windows(["services/api/local_lm/api.py"])
    assert requires_windows(["packaging/windows/LMAtelier.iss"])
    assert requires_windows(["packaging/LMAtelier.spec"])
    assert requires_windows(["scripts/verify.ps1"])
    assert requires_windows([".github/workflows/ci.yml"])
    assert not requires_windows(["apps/web/src/App.tsx"])
    assert not requires_windows(["packaging/linux/frozen-uninstall.sh"])
    assert not requires_windows(["docs/ARCHITECTURE.md"])


def test_ci_plan_rejects_malformed_event_shas() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/ci-plan.py"))
    require_sha = namespace["require_sha"]

    with pytest.raises(ValueError, match="full commit SHA"):
        require_sha("base SHA", "--output=unexpected")


def test_ci_plan_requires_exact_protected_develop_promotion(monkeypatch) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/ci-plan.py"))
    validate = namespace["validate_develop_promotion"]
    base = "a" * 40
    head = "b" * 40
    common = "c" * 40

    def exact_git(*arguments: str) -> str:
        queries = {
            ("rev-parse", "--verify", f"{base}^{{commit}}"): base,
            ("rev-parse", "--verify", f"{head}^{{commit}}"): head,
            ("rev-parse", "origin/main"): base,
            ("rev-parse", "origin/develop"): head,
            ("merge-base", base, head): common,
            ("rev-parse", f"{base}^{{tree}}"): "tree",
            ("rev-parse", f"{common}^{{tree}}"): "tree",
        }
        return queries[arguments]

    monkeypatch.setitem(validate.__globals__, "git", exact_git)
    validate(base_ref="main", head_ref="develop", base_sha=base, head_sha=head)

    with pytest.raises(ValueError, match="develop branch"):
        validate(base_ref="main", head_ref="other", base_sha=base, head_sha=head)

    def divergent_git(*arguments: str) -> str:
        if arguments == ("rev-parse", f"{common}^{{tree}}"):
            return "different-tree"
        return exact_git(*arguments)

    monkeypatch.setitem(validate.__globals__, "git", divergent_git)
    with pytest.raises(ValueError, match="not present in the develop lineage"):
        validate(base_ref="main", head_ref="develop", base_sha=base, head_sha=head)


def test_ci_workflow_retains_required_check_for_every_pr_scope() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    plan = workflow.split("  verification-plan:", 1)[1].split("  compatibility:", 1)[0]
    compatibility = workflow.split("  compatibility:", 1)[1].split("  windows-compatibility:", 1)[0]
    windows = workflow.split("  windows-compatibility:", 1)[1].split("  scheduled-audit:", 1)[0]

    assert "name: Ubuntu compatibility" in compatibility
    assert "scripts/ci-plan.py" in plan
    for variable in ("EVENT_NAME", "BASE_REF", "HEAD_REF", "BASE_SHA", "HEAD_SHA"):
        assert f'"${variable}"' in plan
    for output in ("mode", "dependency_audit", "windows"):
        assert f"{output}: ${{{{ steps.plan.outputs.{output} }}}}" in plan
    assert compatibility.count("needs.verification-plan.outputs.mode") >= 10
    assert "needs.verification-plan.result != 'success'" in compatibility
    assert "documentation" in compatibility
    assert "promotion" in compatibility
    assert "needs.verification-plan.outputs.dependency_audit" in compatibility
    assert "name: Windows compatibility" in windows
    assert "runs-on: windows-2025" in windows
    assert "needs.verification-plan.outputs.windows == 'true'" in windows
    assert "choco install ffmpeg --version 8.1.2" in windows
    assert r".\scripts\verify.ps1" in windows
    assert "23 9 * * 2" in workflow
    assert "name: Scheduled dependency audit" in workflow
    assert "npm audit --audit-level=high" in workflow
    assert ".venv/bin/python -m pip_audit" in workflow


def test_merge_gate_refuses_every_unverified_pull_request_shape() -> None:
    """Run the merge gate's own script, rather than reading the workflow text.

    A ruleset counts a skipped required check as satisfied, so this job is the
    only thing preventing a merge with no verification behind it. Asserting that
    it *exists* would assert the wrong thing: the first version of it existed,
    read correctly, and still exited zero on a draft, which handed the draft head
    a green required context.

    So the evaluator shared with `scripts/validate-workflows.py` executes the
    real script across the event and result matrix and this pins the outcomes.
    """
    namespace = runpy.run_path(str(ROOT / "scripts/validate-workflows.py"))
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    problems = namespace["validate_merge_gate"](Path(".github/workflows/ci.yml"), workflow)
    assert problems == [], "\n".join(problems)

    matrix = namespace["MERGE_GATE_MATRIX"]
    labels = {label for label, _environment, _expected in matrix}
    # The two shapes that were wrong in the first attempt, named so that
    # deleting either case fails here rather than silently narrowing coverage.
    assert "draft must fail closed" in labels
    assert "a plan-required Windows must not be skipped" in labels
    assert any(expected == 0 for _l, _e, expected in matrix), (
        "a matrix that only ever expects failure would pass against a gate that rejects everything"
    )


def test_public_repository_configuration_verifies_every_applied_control() -> None:
    script = (ROOT / "scripts/configure-public-repository.ps1").read_text()

    for endpoint in (
        "actions/permissions",
        "actions/permissions/selected-actions",
        "actions/permissions/workflow",
        "automated-security-fixes",
        "code-scanning/default-setup",
        "private-vulnerability-reporting",
        "topics",
        "vulnerability-alerts",
    ):
        assert f"repos/$Repository/{endpoint}" in script
    assert "Assert-JsonSubset" in script
    assert "-Expected $BranchRules" in script
    assert "-Expected $TagRules" in script
    assert "-Expected $TagCreationRules" in script
    assert "sha_pinning_required = $true" in script
    assert 'secret_scanning_push_protection = @{ status = "enabled" }' in script
    assert script.count("allow_auto_merge = $true") == 2
    assert "allow_auto_merge = $false" not in script


def test_browser_runners_do_not_execute_an_environment_selected_program() -> None:
    for runner in (
        ROOT / "scripts/run-browser-e2e.mjs",
        ROOT / "scripts/run-workflow-editor-e2e.mjs",
    ):
        source = runner.read_text()
        assert "LM_ATELIER_E2E_PYTHON" not in source
        assert "firstExistingPath([environmentPython, projectPython])" in source
        assert 'process.platform === "win32" ? "python.exe" : "python3"' in source


def test_frozen_installer_contracts_are_explicit() -> None:
    spec = (ROOT / "packaging/LMAtelier.spec").read_text()
    frozen_smoke = (ROOT / "scripts/smoke-frozen.py").read_text()
    windows_build = (ROOT / "scripts/build-windows-installer.ps1").read_text()
    linux_build = (ROOT / "scripts/build-linux-installer.sh").read_text()
    linux_smoke = (ROOT / "scripts/smoke-linux-installer.sh").read_text()
    windows_smoke = (ROOT / "scripts/smoke-windows-installer.ps1").read_text()
    windows = (ROOT / "packaging/windows/LMAtelier.iss").read_text()
    linux = (ROOT / "packaging/linux/self-extracting-installer.sh").read_text()
    linux_uninstall = (ROOT / "packaging/linux/frozen-uninstall.sh").read_text()

    assert '"apps" / "web" / "dist"' in spec
    assert '"packaging" / "engines.json"' in spec
    assert '"packaging" / "runtime-reviews"' in spec
    assert '"local_lm/migrations"' in spec
    assert '"local_lm" / "capability_packs"' in spec
    assert '"local_lm" / "comfy_editor_bridge_assets"' in spec
    assert "*editor_bridge_datas" in spec
    assert '"local_lm" / "workflow_editor_shell_assets"' in spec
    assert "*editor_shell_datas" in spec
    pyproject = (ROOT / "services/api/pyproject.toml").read_text()
    assert '"local_lm/comfy_editor_bridge_assets/**/*.js"' in pyproject
    assert '"local_lm/workflow_editor_shell_assets/*"' in pyproject
    assert '"__pycache__" not in source.parts' in spec
    assert 'source.suffix not in {".pyc", ".pyo"}' in spec
    assert '"PIL",' not in spec
    assert "upx=False" in spec
    assert "upx=True" not in spec
    assert '"release-metadata" / "LICENSE"' in spec
    assert '"release-metadata" / "THIRD_PARTY_NOTICES.md"' in spec
    assert '"release-metadata" / "sbom.cdx.json"' in spec
    assert '"release-metadata" / "third-party-licenses"' in spec
    assert '"LOCAL_LM_CHAT_ENGINE": "mock"' in frozen_smoke
    assert '"LOCAL_LM_MEDIA_ENGINE": "mock"' in frozen_smoke
    assert 'runtime_result.get("chat_engine") == "llama.cpp"' in frozen_smoke
    assert 'runtime_result.get("media_engine") == "comfyui"' in frozen_smoke
    assert 'runtime_result.get("engine_manifest_available") is True' in frozen_smoke
    assert "args.pre_manifest_release or (" in frozen_smoke
    assert "--installer-tool" in windows_build
    assert "--installer-tool-version" in windows_build
    assert "--installer-tool-sha256" in windows_build
    assert '$ExpectedInnoVersion = "6.7.1"' in windows_build
    assert "eb6f4410c8db367a5f74127e8025ad2ccacc0afabbe783959d237df3050f97fb" in windows_build
    assert "Get-AuthenticodeSignature" in windows_build
    assert "--require-release-tag" in windows_build
    assert '"/DMyFileVersion=$FileVersion"' in windows_build
    assert "--installer-tool" in linux_build
    assert "--installer-tool-version" in linux_build
    assert "--require-release-tag" in linux_build
    assert "inventory-frozen-payload.py" in windows_build
    assert "inventory-frozen-payload.py" in linux_build
    assert "installer-smoke-preserve" in windows_smoke
    assert "Start-Process" in windows_smoke
    assert "-WaitProcess" in windows_smoke
    assert '$InstallArguments += "/NOICONS"' in windows_smoke
    assert "PreviousInstaller" in windows_smoke
    assert "Version upgrade did not preserve local data" in windows_smoke
    assert "/PURGEDATA" in windows_smoke
    assert "installer-smoke-preserve" in linux_smoke
    assert "previous_installer" in linux_smoke
    assert "lm-atelier-legacy" in linux_smoke
    assert "--purge-data" in linux_smoke
    assert "PrivilegesRequired=lowest" in windows
    assert "VersionInfoVersion={#MyFileVersion}" in windows
    assert "VersionInfoProductTextVersion={#MyAppVersion}" in windows
    assert r"DefaultDirName={localappdata}\Programs\LM Atelier" in windows
    assert r'Filename: "{app}\{#MyAppExeName}"' in windows
    assert "__LM_ATELIER_PAYLOAD_BELOW__" in linux
    assert "$HOME/.local/bin/lm-atelier" in linux
    assert "Linux image/video require an externally configured compatible media engine" in linux
    assert "--purge-data" in linux_uninstall
    assert "is_managed_install" in linux_uninstall
    assert "is_managed_install" in linux


def test_release_metadata_contains_licenses_and_sbom() -> None:
    output = ROOT / "build" / f"test-release-metadata-{os.getpid()}"
    try:
        started_at = datetime.now(UTC)
        environment = os.environ.copy()
        environment.pop("SOURCE_DATE_EPOCH", None)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build-release-metadata.py"),
                "--output-dir",
                str(output),
                "--installer-tool",
                "Test Installer",
                "--installer-tool-version",
                " 1.2.3 ",
                "--installer-tool-sha256",
                "a" * 64,
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        finished_at = datetime.now(UTC)
        manifest = json.loads((output / "release-manifest.json").read_text())
        sbom = json.loads((output / "sbom.cdx.json").read_text())
        notices = (output / "THIRD_PARTY_NOTICES.md").read_text()

        assert (output / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
        assert manifest["version"] == __version__
        release_source = json.loads((ROOT / "packaging/release-manifest.json").read_text())
        assert manifest["runtime_scope"] == release_source["runtime_scope"]
        source_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_tag_result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        source_tag = source_tag_result.stdout.strip() if source_tag_result.returncode == 0 else None
        source_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        assert manifest["source"] == {
            "sha": source_sha,
            "commit": source_sha,
            "tag": source_tag,
            "dirty": source_dirty,
        }
        assert manifest["dependency_locks"] == ["package-lock.json", "services/api/uv.lock"]
        assert manifest["locked_build_inputs"] == [
            {
                "path": path,
                "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            }
            for path in manifest["dependency_locks"]
        ]
        generated_at = datetime.fromisoformat(manifest["generated_at"].replace("Z", "+00:00"))
        assert started_at <= generated_at <= finished_at
        assert manifest["toolchain"]["python"]
        assert manifest["toolchain"]["node"]
        assert manifest["toolchain"]["npm"]
        assert manifest["toolchain"]["uv"]
        assert manifest["toolchain"]["pyinstaller"]
        assert manifest["toolchain"]["platform"]["system"]
        assert manifest["toolchain"]["platform"]["release"]
        assert manifest["toolchain"]["platform"]["version"]
        assert manifest["toolchain"]["platform"]["machine"]
        assert manifest["toolchain"]["installer"] == {
            "name": "Test Installer",
            "version": "1.2.3",
            "sha256": "a" * 64,
        }
        assert manifest["signature_status"] == "unsigned-development-build"
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.6"
        assert sbom["components"]
        assert "UNKNOWN" not in notices
        assert any((output / "third-party-licenses").rglob("*"))
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_strict_mypy_gates_load_the_strict_api_config() -> None:
    """A check that promises more than it enforces is worse than one that
    promises less.

    Without an explicit config, mypy does not discover the nested API
    pyproject from the repository root. Both platform gates must name that
    file, and the named configuration must continue to enable strict mode.
    """

    config = tomllib.loads(
        (ROOT / "services" / "api" / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert config["tool"]["mypy"]["strict"] is True

    script = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")
    step = re.search(r'Invoke-Checked "Strict mypy".*?\)', script, re.S)
    assert step is not None
    assert "--config-file" in step.group(0)
    assert "services/api/pyproject.toml" in step.group(0)

    linux = (ROOT / "scripts" / "verify.sh").read_text(encoding="utf-8")
    linux_step = linux.split('run_checked "Strict mypy"', 1)[1].split(
        'run_checked "Bandit high-severity scan"', 1
    )[0]
    assert "--config-file services/api/pyproject.toml" in linux_step


def test_api_mypy_config_rejects_a_strict_only_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "strict_only_fixture.py"
    fixture.write_text(
        "def missing_annotations(value):" + chr(10) + "    return value" + chr(10),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(ROOT / "services" / "api" / "pyproject.toml"),
            str(fixture),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, result.stdout + result.stderr


def test_a_file_that_declares_it_must_not_ship_is_refused(tmp_path: Path) -> None:
    """Some files carry handling flags saying they must never be published, and
    nothing enforced them, so the rule held only as long as nobody copied the
    content somewhere tracked.

    Matching the declaration rather than a path means a rename, a copy, or an
    excerpt embedded in another document is still caught, which is how this kind
    of content actually escapes.
    """

    namespace = runpy.run_path(str(ROOT / "scripts/check-repository-hygiene.py"))
    refused = namespace["declares_it_must_not_ship"]

    # Assembled rather than written out, so this file does not trip the very
    # check it tests - the same discipline the secret scan test follows.
    committable = '"never' + '_commit"'
    publishable = '"never' + '_publish"'
    documentable = '"never' + '_include_in_public_documentation"'

    for ordinal, flag in enumerate((committable, publishable, documentable)):
        declared = tmp_path / f"declared-{ordinal}.json"
        declared.write_text("{" + flag + ": true}", encoding="utf-8")
        assert refused(str(declared)), flag

    excerpt = tmp_path / "notes.md"
    excerpt.write_text(
        "Pasted from elsewhere:\n\n    " + committable + ": true\n", encoding="utf-8"
    )
    assert refused(str(excerpt))

    ordinary = tmp_path / "settings.json"
    ordinary.write_text(
        '{"classification": "public", ' + committable + ": false}", encoding="utf-8"
    )
    assert not refused(str(ordinary))
