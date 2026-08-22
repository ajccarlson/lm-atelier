from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MAX_CAPABILITY_PACK_BYTES = 256 * 1024
MAX_CAPABILITY_JSON_DEPTH = 16
MAX_CAPABILITY_JSON_NODES = 2_500
MAX_CAPABILITY_JSON_STRING_CHARACTERS = 65_536
MAX_CAPABILITY_JSON_OBJECT_MEMBERS = 1_024
MAX_CAPABILITY_JSON_LIST_ITEMS = 2_048
MAX_CAPABILITY_JSON_NUMBER_CHARACTERS = 128
MAX_CAPABILITY_FAMILIES = 256
MAX_MARKERS_PER_FAMILY = 64
_PACK_DIRECTORY = Path(__file__).with_name("capability_packs")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")


class CapabilityPackError(ValueError):
    """A data-only capability pack failed its integrity or schema checks."""


@dataclass(frozen=True)
class ArchitectureFamilyContract:
    id: str
    roles: tuple[str, ...]
    architecture_markers: tuple[str, ...]


def _read_bounded_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(MAX_CAPABILITY_PACK_BYTES + 1)
    if len(content) > MAX_CAPABILITY_PACK_BYTES:
        raise CapabilityPackError("capability pack exceeds the size limit")
    return content


def _parse_json_object(content: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CapabilityPackError("capability pack JSON has duplicate keys")
            result[key] = value
        return result

    def bounded_int(token: str) -> int:
        if len(token) > MAX_CAPABILITY_JSON_NUMBER_CHARACTERS:
            raise CapabilityPackError("capability pack JSON number is too long")
        try:
            return int(token)
        except ValueError as exc:
            raise CapabilityPackError("capability pack JSON number is invalid") from exc

    def bounded_float(token: str) -> float:
        if len(token) > MAX_CAPABILITY_JSON_NUMBER_CHARACTERS:
            raise CapabilityPackError("capability pack JSON number is too long")
        try:
            value = float(token)
        except (OverflowError, ValueError) as exc:
            raise CapabilityPackError("capability pack JSON number is invalid") from exc
        if not math.isfinite(value):
            raise CapabilityPackError("capability pack JSON number is invalid")
        return value

    def reject_constant(_token: str) -> float:
        raise CapabilityPackError("capability pack JSON number is invalid")

    try:
        value = json.loads(
            content,
            object_pairs_hook=unique_object,
            parse_int=bounded_int,
            parse_float=bounded_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CapabilityPackError("capability pack JSON is malformed") from exc
    if not isinstance(value, dict):
        raise CapabilityPackError("capability pack must contain an object")
    _validate_json_budget(value)
    return value


def _validate_json_budget(value: dict[str, Any]) -> None:
    nodes = 0
    string_characters = 0
    object_members = 0
    list_items = 0
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_CAPABILITY_JSON_NODES:
            raise CapabilityPackError("capability pack JSON has too many values")
        if depth > MAX_CAPABILITY_JSON_DEPTH:
            raise CapabilityPackError("capability pack JSON is nested too deeply")
        if isinstance(current, dict):
            object_members += len(current)
            if object_members > MAX_CAPABILITY_JSON_OBJECT_MEMBERS:
                raise CapabilityPackError("capability pack JSON has too many object members")
            for key, child in current.items():
                string_characters += len(key)
                pending.append((child, depth + 1))
        elif isinstance(current, list):
            list_items += len(current)
            if list_items > MAX_CAPABILITY_JSON_LIST_ITEMS:
                raise CapabilityPackError("capability pack JSON has too many list items")
            pending.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            string_characters += len(current)
        elif isinstance(current, float) and not math.isfinite(current):
            raise CapabilityPackError("capability pack JSON number is invalid")
        if string_characters > MAX_CAPABILITY_JSON_STRING_CHARACTERS:
            raise CapabilityPackError("capability pack JSON strings are too long")


@lru_cache(maxsize=1)
def architecture_family_contracts() -> tuple[ArchitectureFamilyContract, ...]:
    lock = _parse_json_object(_read_bounded_bytes(_PACK_DIRECTORY / "capability-packs.lock.json"))
    packs = lock.get("packs")
    if (
        set(lock) != {"version", "packs"}
        or type(lock.get("version")) is not int
        or lock["version"] != 1
        or not isinstance(packs, dict)
    ):
        raise CapabilityPackError("capability pack lock is invalid")
    filename = "architecture-families-v1.json"
    if set(packs) != {filename}:
        raise CapabilityPackError("capability pack lock is invalid")
    expected_hash = packs.get(filename)
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise CapabilityPackError("capability pack is not hash-pinned")
    path = _PACK_DIRECTORY / filename
    content = _read_bounded_bytes(path)
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise CapabilityPackError("capability pack integrity check failed")
    payload = _parse_json_object(content)
    raw_families = payload.get("families")
    if (
        set(payload) != {"version", "families"}
        or type(payload.get("version")) is not int
        or payload["version"] != 1
        or not isinstance(raw_families, list)
    ):
        raise CapabilityPackError("capability pack schema is unsupported")
    if len(raw_families) > MAX_CAPABILITY_FAMILIES:
        raise CapabilityPackError("capability pack declares too many families")
    contracts: list[ArchitectureFamilyContract] = []
    seen: set[str] = set()
    for item in raw_families:
        if not isinstance(item, dict):
            raise CapabilityPackError("capability family entry is invalid")
        family_id = item.get("id")
        roles = item.get("roles")
        markers = item.get("architecture_markers")
        if (
            set(item) != {"id", "roles", "architecture_markers"}
            or not isinstance(family_id, str)
            or not _SAFE_ID.fullmatch(family_id)
            or family_id in seen
            or not isinstance(roles, list)
            or not roles
            or any(
                not isinstance(role, str) or role not in {"chat", "image", "video"}
                for role in roles
            )
            or len(set(roles)) != len(roles)
            or not isinstance(markers, list)
            or not markers
            or len(markers) > MAX_MARKERS_PER_FAMILY
            or any(
                not isinstance(marker, str)
                or not marker
                or len(marker) > 200
                or marker != marker.casefold()
                for marker in markers
            )
            or len(set(markers)) != len(markers)
        ):
            raise CapabilityPackError("capability family entry is invalid")
        seen.add(family_id)
        contracts.append(
            ArchitectureFamilyContract(
                id=family_id,
                roles=tuple(dict.fromkeys(str(role) for role in roles)),
                architecture_markers=tuple(dict.fromkeys(str(marker) for marker in markers)),
            )
        )
    return tuple(contracts)
