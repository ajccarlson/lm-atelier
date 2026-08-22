from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .capability_packs import architecture_family_contracts

MAX_METADATA_BYTES = 1024 * 1024
MAX_WEIGHT_HEADER_BYTES = 16 * 1024 * 1024
MAX_METADATA_NODES = 100_000
MAX_GGUF_FIELDS = 4_096
MAX_GGUF_NESTING = 8

_GGUF_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class ModelManifestError(ValueError):
    """Declarative model metadata is malformed, unsafe, or exceeds a bound."""


@dataclass(frozen=True)
class InspectedComponent:
    path: str
    kind: str
    target_folder: str
    architecture: str | None = None
    family: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelManifestInspection:
    architecture: str | None
    family: str | None
    components: tuple[InspectedComponent, ...]
    metadata_files: tuple[str, ...]


def inspect_repository_metadata(
    files: Mapping[str, bytes],
    selected_paths: list[str],
    *,
    role: str,
    component_folders: Mapping[str, str] | None = None,
) -> ModelManifestInspection:
    """Inspect bounded data-only metadata without importing repository code.

    `component_folders` maps a repository path to the ComfyUI directory the
    compiled template installs it into. It is the authority when present:
    inferring the folder from the path only works when the repository happens to
    name its directories the way ComfyUI does, and falls back to `checkpoints`
    when it does not - which silently mislabels every component of a
    multi-component model.
    """

    metadata: dict[str, dict[str, Any]] = {}
    components: list[InspectedComponent] = []
    for raw_path, content in files.items():
        path = _safe_path(raw_path)
        name = path.name.casefold()
        if name.endswith(".json"):
            metadata[path.as_posix()] = _bounded_json_object(content)
        elif name.endswith(".safetensors"):
            components.append(_inspect_safetensors(path, content))
        elif name.endswith(".gguf"):
            components.append(_inspect_gguf(path, content))

    architecture, family = _repository_identity(metadata, components, role)
    selected = {_safe_path(value).as_posix() for value in selected_paths}
    by_path = {component.path: component for component in components}
    declared_folders = dict(component_folders or {})
    resolved: list[InspectedComponent] = []
    for selected_path in sorted(selected):
        component = by_path.get(selected_path)
        if component:
            target_folder = declared_folders.get(
                selected_path,
                _target_folder(selected_path, role)
                if component.kind == "unknown_safetensors"
                else component.target_folder,
            )
            kind = component.kind
            if kind == "unknown_safetensors":
                kind = _kind_from_comfy_folder(target_folder) or kind
            resolved.append(
                InspectedComponent(
                    path=component.path,
                    kind=kind,
                    target_folder=target_folder,
                    architecture=component.architecture or architecture,
                    family=component.family or family,
                    metadata=component.metadata,
                )
            )
            continue
        resolved.append(
            InspectedComponent(
                path=selected_path,
                kind=_component_kind_from_path(selected_path, role),
                target_folder=declared_folders.get(
                    selected_path, _target_folder(selected_path, role)
                ),
                architecture=architecture,
                family=family,
            )
        )
    return ModelManifestInspection(
        architecture=architecture,
        family=family,
        components=tuple(resolved),
        metadata_files=tuple(sorted(metadata)),
    )


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or len(value) > 1_000
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ModelManifestError("model metadata path is unsafe")
    return path


def _bounded_json_object(content: bytes) -> dict[str, Any]:
    if len(content) > MAX_METADATA_BYTES:
        raise ModelManifestError("model metadata JSON exceeds the size limit")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelManifestError("model metadata JSON is malformed") from exc
    if not isinstance(value, dict):
        raise ModelManifestError("model metadata JSON must contain an object")
    if _count_nodes(value) > MAX_METADATA_NODES:
        raise ModelManifestError("model metadata JSON is too complex")
    return value


def _count_nodes(value: Any) -> int:
    count = 0
    pending = [value]
    while pending:
        current = pending.pop()
        count += 1
        if count > MAX_METADATA_NODES:
            return count
        if isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return count


def _inspect_safetensors(path: PurePosixPath, content: bytes) -> InspectedComponent:
    if len(content) < 8:
        raise ModelManifestError("safetensors header is truncated")
    header_size = int.from_bytes(content[:8], "little")
    if header_size < 2 or header_size > MAX_WEIGHT_HEADER_BYTES:
        raise ModelManifestError("safetensors header exceeds the size limit")
    if len(content) < 8 + header_size:
        raise ModelManifestError("safetensors header is truncated")
    header = _bounded_json_object_with_limit(
        content[8 : 8 + header_size],
        MAX_WEIGHT_HEADER_BYTES,
    )
    tensor_names = sorted(str(key) for key in header if key != "__metadata__")
    if len(tensor_names) > MAX_METADATA_NODES:
        raise ModelManifestError("safetensors tensor index is too large")
    raw_metadata = header.get("__metadata__") or {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    kind = _safetensors_kind(tensor_names, metadata)
    family = _safetensors_family(tensor_names, metadata)
    component_metadata: dict[str, Any] = {
        "tensor_count": len(tensor_names),
        "metadata_keys": sorted(str(key)[:200] for key in metadata)[:128],
    }
    if kind == "lora":
        component_metadata.update(_lora_metadata(header, metadata))
    return InspectedComponent(
        path=path.as_posix(),
        kind=kind,
        target_folder=_target_folder_for_kind(kind),
        family=family,
        metadata=component_metadata,
    )


def _lora_metadata(
    header: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    network_type = next(
        (
            str(metadata[key])[:200]
            for key in ("ss_network_module", "modelspec.architecture", "network_type")
            if isinstance(metadata.get(key), str)
        ),
        "lora",
    )
    rank: int | None = None
    declared_rank = metadata.get("ss_network_dim")
    if isinstance(declared_rank, str) and declared_rank.isdigit():
        rank = int(declared_rank)
    elif isinstance(declared_rank, int) and not isinstance(declared_rank, bool):
        rank = declared_rank
    if rank is None:
        ranks = {
            int(shape[0])
            for name, tensor in header.items()
            if name != "__metadata__"
            and "lora_down" in str(name).casefold()
            and isinstance(tensor, dict)
            and isinstance((shape := tensor.get("shape")), list)
            and shape
            and isinstance(shape[0], int)
            and 0 < shape[0] <= 65_536
        }
        rank = min(ranks) if ranks else None
    trigger_words: list[str] = []
    for key in ("trigger_words", "modelspec.trigger_phrase"):
        raw = metadata.get(key)
        if isinstance(raw, str):
            trigger_words.extend(
                word.strip()[:100] for word in raw.replace("\n", ",").split(",") if word.strip()
            )
    return {
        "network_type": network_type,
        "rank": rank,
        "trigger_words": list(dict.fromkeys(trigger_words))[:32],
    }


def _bounded_json_object_with_limit(content: bytes, limit: int) -> dict[str, Any]:
    if len(content) > limit:
        raise ModelManifestError("weight metadata exceeds the size limit")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelManifestError("weight metadata is malformed") from exc
    if not isinstance(value, dict) or _count_nodes(value) > MAX_METADATA_NODES:
        raise ModelManifestError("weight metadata is too complex")
    return value


def _safetensors_kind(tensor_names: list[str], metadata: Mapping[str, Any]) -> str:
    lowered = [name.casefold() for name in tensor_names]
    metadata_text = " ".join(f"{key}={value}" for key, value in metadata.items()).casefold()
    # LyCORIS adapters are LoRAs in every way that matters here - ComfyUI loads
    # them through the same loader and providers list them as LoRAs - but none
    # of them spells "lora" in a tensor name. LoKr factorises into `lokr_w1`
    # and `lokr_w2`, LoHa into `hada_w1_a` and `hada_w2_b`, and DoRA adds a
    # `dora_scale`. A file full of those was classified as an unknown
    # safetensors blob, so a download that had already finished was thrown away
    # at the contract check for not being what it plainly is.
    #
    # These names carry no other meaning in a checkpoint, so matching them
    # cannot promote something that is not an adapter.
    adapter_markers = ("lokr_", "hada_w", "dora_scale", "oft_blocks", "oft_diag")
    if (
        "lora" in metadata_text
        or any(name.startswith(("lora_", "lycoris_")) or ".lora_" in name for name in lowered)
        or any(marker in name for name in lowered for marker in adapter_markers)
    ):
        return "lora"
    has_diffusion = any(
        name.startswith(("model.diffusion_model.", "diffusion_model.", "transformer."))
        or ".double_blocks." in name
        for name in lowered
    )
    has_conditioning = any(
        name.startswith(("conditioner.", "cond_stage_model.", "text_encoder.")) for name in lowered
    )
    has_vae = any(name.startswith(("first_stage_model.", "vae.")) for name in lowered)
    if has_diffusion and (has_conditioning or has_vae):
        return "checkpoint"
    if has_diffusion:
        return "diffusion_model"
    roots = {name.partition(".")[0] for name in lowered}
    # Krea 2 and related NextDiT-style diffusion weights do not use the
    # historical model.diffusion_model/transformer prefixes. Require the
    # distinctive collection of image and text-fusion roots so an arbitrary
    # blocks.* tensor set cannot be mistaken for a media model.
    if {
        "blocks",
        "first",
        "last",
        "tproj",
        "txtfusion",
    }.issubset(roots) and {"tmlp", "txtmlp"}.intersection(roots):
        return "diffusion_model"
    if has_vae:
        return "vae"
    if any("text_model." in name or name.startswith("text_encoder.") for name in lowered):
        return "text_encoder"
    if any("vision_model." in name for name in lowered):
        return "clip_vision"
    return "unknown_safetensors"


def _safetensors_family(
    tensor_names: list[str],
    metadata: Mapping[str, Any],
) -> str | None:
    lowered = [name.casefold() for name in tensor_names]
    metadata_text = " ".join(str(value) for value in metadata.values()).casefold()
    if "sdxl" in metadata_text or any("conditioner.embedders.1." in name for name in lowered):
        return "sdxl"
    if "flux" in metadata_text or any("double_blocks." in name for name in lowered):
        return "flux"
    if "stable-diffusion" in metadata_text or any(
        name.startswith("cond_stage_model.") for name in lowered
    ):
        return "stable-diffusion"
    return None


def _inspect_gguf(path: PurePosixPath, content: bytes) -> InspectedComponent:
    if len(content) > MAX_WEIGHT_HEADER_BYTES:
        content = content[:MAX_WEIGHT_HEADER_BYTES]
    cursor = _BinaryCursor(content)
    if cursor.read(4) != b"GGUF":
        raise ModelManifestError("GGUF header has an invalid magic value")
    version = cursor.scalar("<I")
    if version not in {2, 3}:
        raise ModelManifestError("GGUF version is unsupported")
    tensor_count = cursor.scalar("<Q")
    field_count = cursor.scalar("<Q")
    if field_count > MAX_GGUF_FIELDS:
        raise ModelManifestError("GGUF metadata has too many fields")
    fields: dict[str, Any] = {}
    retained_fields = {
        "general.architecture",
        "general.name",
        "general.type",
        "clip.projector_type",
    }
    for _ in range(field_count):
        key = cursor.string()
        value_type = cursor.scalar("<I")
        retain = key in retained_fields
        value = _read_gguf_value(cursor, value_type, retain=retain)
        if retain:
            fields[key] = value
    architecture = _printable_metadata(fields.get("general.architecture"))
    projector = "mmproj" in path.name.casefold() or "clip.projector_type" in fields
    return InspectedComponent(
        path=path.as_posix(),
        kind="projector" if projector else "gguf_model",
        target_folder="projectors" if projector else "models",
        architecture=architecture,
        family=architecture,
        metadata={
            "gguf_version": version,
            "tensor_count": tensor_count,
            "general_type": _printable_metadata(fields.get("general.type")),
        },
    )


class _BinaryCursor:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.position = 0

    def read(self, length: int) -> bytes:
        if length < 0 or self.position + length > len(self.content):
            raise ModelManifestError("weight header is truncated")
        result = self.content[self.position : self.position + length]
        self.position += length
        return result

    @property
    def remaining(self) -> int:
        return len(self.content) - self.position

    def skip(self, length: int) -> None:
        if length < 0 or self.position + length > len(self.content):
            raise ModelManifestError("weight header is truncated")
        self.position += length

    def scalar(self, format_string: str) -> Any:
        size = struct.calcsize(format_string)
        return struct.unpack(format_string, self.read(size))[0]

    def string(self) -> str:
        length = self.scalar("<Q")
        if length > MAX_METADATA_BYTES:
            raise ModelManifestError("weight metadata string exceeds the size limit")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelManifestError("weight metadata string is invalid") from exc

    def skip_string(self) -> None:
        length = self.scalar("<Q")
        if length > MAX_METADATA_BYTES:
            raise ModelManifestError("weight metadata string exceeds the size limit")
        self.skip(length)


def _read_gguf_value(
    cursor: _BinaryCursor,
    value_type: int,
    *,
    retain: bool,
    depth: int = 0,
) -> Any:
    if depth > MAX_GGUF_NESTING:
        raise ModelManifestError("GGUF metadata arrays are nested too deeply")
    if value_type in _GGUF_SCALAR_FORMATS:
        format_string = _GGUF_SCALAR_FORMATS[value_type]
        if retain:
            return cursor.scalar(format_string)
        cursor.skip(struct.calcsize(format_string))
        return None
    if value_type == 8:
        if retain:
            return cursor.string()
        cursor.skip_string()
        return None
    if value_type == 9:
        item_type = cursor.scalar("<I")
        item_count = cursor.scalar("<Q")
        if item_type in _GGUF_SCALAR_FORMATS:
            format_string = _GGUF_SCALAR_FORMATS[item_type]
            item_size = struct.calcsize(format_string)
            retained_count = min(item_count, 128) if retain else 0
            values = [cursor.scalar(format_string) for _ in range(retained_count)]
            cursor.skip((item_count - retained_count) * item_size)
            return values if retain else None
        if item_type not in {8, 9}:
            raise ModelManifestError("GGUF metadata uses an unknown array item type")
        minimum_item_size = 8 if item_type == 8 else 12
        if item_count > cursor.remaining // minimum_item_size:
            raise ModelManifestError("weight header is truncated")
        values = []
        for index in range(item_count):
            keep = retain and index < 128
            value = _read_gguf_value(
                cursor,
                item_type,
                retain=keep,
                depth=depth + 1,
            )
            if keep:
                values.append(value)
        return values if retain else None
    raise ModelManifestError("GGUF metadata uses an unknown value type")


def _repository_identity(
    metadata: Mapping[str, Mapping[str, Any]],
    components: list[InspectedComponent],
    role: str,
) -> tuple[str | None, str | None]:
    candidates: list[str] = []
    for document in metadata.values():
        for key in ("model_type", "_class_name", "architecture"):
            value = document.get(key)
            if isinstance(value, str):
                candidates.append(value)
        architectures = document.get("architectures")
        if isinstance(architectures, list):
            candidates.extend(str(value) for value in architectures if isinstance(value, str))
    candidates.extend(
        value
        for component in components
        for value in (component.architecture, component.family)
        if value
    )
    architecture = candidates[0][:200] if candidates else None
    family = _family_from_architecture(candidates, role)
    if not family:
        family = next((component.family for component in components if component.family), None)
    return architecture, family


def _family_from_architecture(candidates: list[str], role: str) -> str | None:
    normalized = " ".join(candidates).casefold()
    for contract in architecture_family_contracts():
        if contract.id == "minimax-h3":
            if role in contract.roles and any(
                len(candidate) <= 200 and candidate.casefold() in contract.architecture_markers
                for candidate in candidates
            ):
                return contract.id
            continue
        if role in contract.roles and any(
            marker in normalized for marker in contract.architecture_markers
        ):
            return contract.id
    return "gguf" if role == "chat" and "gguf" in normalized else None


def _component_kind_from_path(path: str, role: str) -> str:
    name = PurePosixPath(path).name.casefold()
    if name.endswith(".gguf"):
        return "projector" if "mmproj" in name else "gguf_model"
    if name.endswith(".safetensors"):
        return "checkpoint" if role in {"image", "video"} else "weights"
    return "metadata"


def _target_folder(path: str, role: str) -> str:
    parts = PurePosixPath(path).parts
    folder = next((part for part in parts[:-1] if part in COMFY_MODEL_FOLDERS), None)
    if folder:
        return folder
    return "models" if role == "chat" else "checkpoints"


_COMFY_FOLDER_BY_KIND = {
    "checkpoint": "checkpoints",
    "diffusion_model": "diffusion_models",
    "text_encoder": "text_encoders",
    "vae": "vae",
    "clip_vision": "clip_vision",
    "lora": "loras",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "embedding": "embeddings",
    "ip_adapter": "ipadapter",
}
COMFY_MODEL_ASSET_KINDS: frozenset[str] = frozenset(_COMFY_FOLDER_BY_KIND)
COMFY_MODEL_FOLDERS: frozenset[str] = frozenset(_COMFY_FOLDER_BY_KIND.values())


def comfy_folder_for_kind(kind: str) -> str | None:
    """Name the ComfyUI folder that serves a component kind.

    One table, because a second copy drifts: a kind missing from the copy that
    publishes model paths leaves the file downloaded, verified, recorded, and
    invisible to the runtime that needs it.
    """

    return _COMFY_FOLDER_BY_KIND.get(kind)


def _target_folder_for_kind(kind: str) -> str:
    return comfy_folder_for_kind(kind) or "checkpoints"


def _kind_from_comfy_folder(folder: str) -> str | None:
    """Recover a component role from a verified ComfyUI component folder."""

    return {
        "diffusion_models": "diffusion_model",
        "text_encoders": "text_encoder",
        "vae": "vae",
        "clip_vision": "clip_vision",
        "loras": "lora",
    }.get(folder.casefold())


def _printable_metadata(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 200:
        return None
    if any(character < " " for character in value):
        return None
    return value
