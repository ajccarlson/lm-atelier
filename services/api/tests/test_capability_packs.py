from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import local_lm.capability_packs as capability_packs
from local_lm.capability_packs import CapabilityPackError, architecture_family_contracts


def test_bundled_architecture_contracts_are_hash_pinned_data() -> None:
    capability_packs.architecture_family_contracts.cache_clear()
    contracts = architecture_family_contracts()

    assert any(
        contract.id == "qwen"
        and contract.roles == ("chat",)
        and "qwen" in contract.architecture_markers
        for contract in contracts
    )
    assert any(contract.id == "stable-diffusion-xl" for contract in contracts)
    assert any(
        contract.id == "minimax-h3"
        and contract.roles == ("video",)
        and contract.architecture_markers == ("minimaxh3", "minimax_h3")
        for contract in contracts
    )


def test_modified_capability_pack_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "architecture-families-v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "families": [
                    {
                        "id": "untrusted",
                        "roles": ["chat"],
                        "architecture_markers": ["untrusted"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "capability-packs.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "packs": {"architecture-families-v1.json": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capability_packs, "_PACK_DIRECTORY", tmp_path)
    capability_packs.architecture_family_contracts.cache_clear()
    try:
        with pytest.raises(CapabilityPackError, match="integrity"):
            architecture_family_contracts()
    finally:
        capability_packs.architecture_family_contracts.cache_clear()


def _install_hashed_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    filename = "architecture-families-v1.json"
    content = json.dumps(payload, separators=(",", ":")).encode()
    (tmp_path / filename).write_bytes(content)
    (tmp_path / "capability-packs.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "packs": {filename: hashlib.sha256(content).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capability_packs, "_PACK_DIRECTORY", tmp_path)
    capability_packs.architecture_family_contracts.cache_clear()


@pytest.mark.parametrize("invalid_role", [None, 1, True, {}, []])
def test_capability_pack_role_elements_fail_with_fixed_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_role: object,
) -> None:
    payload = {
        "version": 1,
        "families": [
            {
                "id": "one",
                "roles": [invalid_role],
                "architecture_markers": ["one"],
            }
        ],
    }
    _install_hashed_pack(tmp_path, monkeypatch, payload)
    try:
        with pytest.raises(CapabilityPackError, match="family entry"):
            architecture_family_contracts()
    finally:
        capability_packs.architecture_family_contracts.cache_clear()


@pytest.mark.parametrize(
    "content",
    [
        b'{"version":1,"version":1,"packs":{}}',
        b'{"version":1,"families":[],"families":[]}',
        (
            b'{"version":1,"families":[{"id":"one","id":"two",'
            b'"roles":["video"],"architecture_markers":["one"]}]}'
        ),
    ],
)
def test_json_decoder_refuses_duplicate_keys_at_every_schema_level(content: bytes) -> None:
    with pytest.raises(CapabilityPackError, match="duplicate keys"):
        capability_packs._parse_json_object(content)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra=True),
        lambda payload: payload.update(version=True),
        lambda payload: payload["families"][0].update(extra=True),
        lambda payload: payload["families"][0]["roles"].append("video"),
        lambda payload: payload["families"][0]["architecture_markers"].append("one"),
    ],
)
def test_capability_pack_schema_refuses_noncanonical_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    payload: dict[str, Any] = {
        "version": 1,
        "families": [
            {
                "id": "one",
                "roles": ["video"],
                "architecture_markers": ["one"],
            }
        ],
    }
    mutation(payload)
    _install_hashed_pack(tmp_path, monkeypatch, payload)
    try:
        with pytest.raises(CapabilityPackError):
            architecture_family_contracts()
    finally:
        capability_packs.architecture_family_contracts.cache_clear()
