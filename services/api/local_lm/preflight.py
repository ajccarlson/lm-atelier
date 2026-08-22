from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Literal

from .config import Settings
from .gguf import (
    GGUFSelectionError,
    automatic_gguf_selection,
    automatic_mmproj_selection,
    validate_gguf_selection,
)
from .hardware_fit import (
    FitRequirements,
    HardwareFit,
    capacity_from_system_info,
    recommend_hardware_fit,
)
from .schemas import (
    CatalogDetail,
    CatalogFileSource,
    CatalogPreflight,
    CatalogPreflightCheck,
    CatalogPreflightRequest,
    SystemInfo,
)

_BLOCKED_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
_CIVITAI_ID = re.compile(r"^[1-9][0-9]{0,11}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


_VLLM_SNAPSHOT_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "hf_quant_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def catalog_file_index(
    files: list[dict[str, Any]],
    *,
    provider: str,
    prefer_duplicate_variants: bool,
) -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
    """Index catalog paths without silently discarding provider variants."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in files:
        filename = str(item.get("filename") or "")
        groups.setdefault(filename, []).append(item)

    indexed: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for filename, candidates in groups.items():
        identities = {
            (
                str(item.get("source_file_id") or ""),
                str(item.get("sha256") or "").casefold(),
                item.get("size"),
            )
            for item in candidates
        }
        if len(identities) > 1:
            ambiguous.add(filename)
        if provider == "civitai" and prefer_duplicate_variants:
            indexed[filename] = min(candidates, key=_civitai_variant_rank)
        else:
            # The caller refuses distinct duplicates before a plan can become
            # installable. Keeping the first item makes blocked-plan evidence
            # deterministic instead of preserving the old last-write-wins bug.
            indexed[filename] = candidates[0]
    return indexed, frozenset(ambiguous)


def _safe_civitai_file(item: dict[str, Any]) -> bool:
    size = item.get("size")
    return (
        PurePosixPath(str(item.get("filename") or "")).suffix.casefold() == ".safetensors"
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size > 0
        and bool(_SHA256.fullmatch(str(item.get("sha256") or "")))
        and bool(_CIVITAI_ID.fullmatch(str(item.get("source_file_id") or "")))
        and str(item.get("pickle_scan_result") or "").casefold() == "success"
        and str(item.get("virus_scan_result") or "").casefold() == "success"
    )


class ExactCivitaiFileSelectionError(ValueError):
    """An immutable provider file ID could not name one safe local artifact."""


def _exact_civitai_files_by_id(
    files: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index immutable identities, refusing a contradictory provider record."""

    by_id: dict[str, dict[str, Any]] = {}
    for item in files:
        source_file_id = str(item.get("source_file_id") or "")
        if not source_file_id:
            continue
        previous = by_id.get(source_file_id)
        if previous is not None and (
            str(previous.get("filename") or "") != str(item.get("filename") or "")
            or str(previous.get("sha256") or "").casefold()
            != str(item.get("sha256") or "").casefold()
            or previous.get("size") != item.get("size")
        ):
            raise ExactCivitaiFileSelectionError(
                "the provider returned one file ID for different artifacts"
            )
        by_id[source_file_id] = item
    return by_id


def safe_civitai_file_variants(
    files: list[dict[str, Any]],
    filename: str,
) -> list[dict[str, Any]]:
    """Safe exact choices for one ambiguous provider destination.

    The caller may render identity, size, and precision from these fresh rows,
    but must submit only the immutable ID. Reusing the resolver's index and
    safety predicate keeps the offered choices identical to what selection can
    actually accept.
    """

    if not filename:
        return []
    candidates = [
        item
        for item in _exact_civitai_files_by_id(files).values()
        if str(item.get("filename") or "") == filename and _safe_civitai_file(item)
    ]
    return sorted(candidates, key=_civitai_variant_rank)


def selected_catalog_file_metadata(
    files: list[dict[str, Any]],
    selected_filenames: list[str],
    *,
    provider: str,
    prefer_duplicate_variants: bool,
    selected_file_ids: list[str] | None = None,
    expected_sha256: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fresh metadata rows that must feed the immutable install planner.

    Preflight returns destination filenames for every provider, but a CivitAI
    destination is not an identity when several variants share it. If exact IDs
    were used, resolve them again against the same fresh detail and prove their
    destinations still equal the assessed result. This prevents the finalizer
    from silently falling back to the provider-primary variant by filename.
    """

    if selected_file_ids:
        if provider != "civitai":
            raise ExactCivitaiFileSelectionError(
                "exact provider file variants are available only for CivitAI"
            )
        exact_files = exact_civitai_file_selection(files, selected_file_ids)
        if [str(item.get("filename") or "") for item in exact_files] != selected_filenames:
            raise ExactCivitaiFileSelectionError(
                "the exact file variants do not match the assessed destinations"
            )
        return exact_files

    indexed, _ambiguous = catalog_file_index(
        files,
        provider=provider,
        prefer_duplicate_variants=prefer_duplicate_variants,
    )
    digests = expected_sha256 or {}
    return [
        indexed.get(
            filename,
            {
                "filename": filename,
                "size": None,
                "sha256": digests.get(filename),
            },
        )
        for filename in selected_filenames
    ]


def exact_civitai_file_selection(
    files: list[dict[str, Any]],
    selected_file_ids: list[str],
) -> list[dict[str, Any]]:
    """Resolve exact CivitAI IDs without trusting a client-supplied filename.

    A CivitAI version can publish several different objects under one display
    filename. The filename is therefore a destination, not an identity. IDs
    are matched only against the freshly fetched version detail, while hashes,
    sizes, scans, and destination names continue to come from that server-side
    record.
    """

    if not selected_file_ids:
        return []
    if len(selected_file_ids) > 512:
        raise ExactCivitaiFileSelectionError("too many exact file variants were selected")

    by_id = _exact_civitai_files_by_id(files)

    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    destinations: set[str] = set()
    for source_file_id in selected_file_ids:
        if not _CIVITAI_ID.fullmatch(source_file_id):
            raise ExactCivitaiFileSelectionError("an exact file variant ID is invalid")
        if source_file_id in seen_ids:
            raise ExactCivitaiFileSelectionError("an exact file variant was selected twice")
        seen_ids.add(source_file_id)
        selected_item = by_id.get(source_file_id)
        if selected_item is None:
            raise ExactCivitaiFileSelectionError(
                "an exact file variant is not part of this model version"
            )
        if not _safe_civitai_file(selected_item):
            raise ExactCivitaiFileSelectionError(
                "an exact file variant is not a scan-cleared, hash-pinned safetensors file"
            )
        destination = str(selected_item.get("filename") or "")
        destination_key = destination.casefold()
        if destination_key in destinations:
            raise ExactCivitaiFileSelectionError(
                "two exact file variants would overwrite the same destination"
            )
        destinations.add(destination_key)
        resolved.append(selected_item)
    return resolved


def _civitai_variant_rank(item: dict[str, Any]) -> tuple[int, int, int, str, str]:
    size = item.get("size")
    provider_primary = str(item.get("source_file_type") or "").casefold() == "model"
    return (
        0 if _safe_civitai_file(item) else 1,
        0 if provider_primary else 1,
        size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else 2**63 - 1,
        str(item.get("source_file_id") or ""),
        str(item.get("sha256") or ""),
    )


def _automatic_vllm_selection(files: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    names: set[str] = set()
    for item in files:
        raw_name = str(item.get("filename") or "")
        path = PurePosixPath(raw_name.replace("\\", "/"))
        if (
            not raw_name
            or path.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        ):
            continue
        base_name = path.name.casefold()
        if base_name.endswith(".safetensors") or base_name in _VLLM_SNAPSHOT_FILES:
            normalized = path.as_posix()
            if normalized.casefold() not in names:
                selected.append(normalized)
                names.add(normalized.casefold())
    selected.sort(key=lambda value: (not value.casefold().endswith(".safetensors"), value))
    selected_names = {PurePosixPath(value).name.casefold() for value in selected}
    if (
        not any(value.endswith(".safetensors") for value in selected_names)
        or "config.json" not in selected_names
        or "hf_quant_config.json" not in selected_names
    ):
        return []
    return selected


def _automatic_selection(
    files: dict[str, dict[str, Any]],
    role: str,
    system_memory_bytes: int,
    auxiliary_kind: str | None = None,
) -> list[str]:
    if role == "chat":
        try:
            chat_primary = automatic_gguf_selection(files.values(), system_memory_bytes)
            projector = automatic_mmproj_selection(files.values(), chat_primary)
            return [*chat_primary, *([projector] if projector else [])]
        except GGUFSelectionError:
            return []

    safe_weights = [
        item for name, item in files.items() if PurePosixPath(name).suffix.lower() == ".safetensors"
    ]
    if not safe_weights:
        return []
    if auxiliary_kind:
        ranked = sorted(
            safe_weights,
            key=lambda item: (
                auxiliary_kind not in str(item.get("filename") or "").casefold(),
                -int(item.get("size") or 0),
                str(item.get("filename") or ""),
            ),
        )
        return [str(ranked[0]["filename"])]
    primary_markers = ("diffusion", "checkpoint", "model", "unet", "t2v", "i2v")
    dependency_markers = ("vae", "text_encoder", "text_encoders", "clip_vision")

    def primary_rank(item: dict[str, Any]) -> tuple[int, int, int]:
        name = str(item.get("filename") or "").lower()
        dependency = any(marker in name for marker in dependency_markers)
        marked = any(marker in name for marker in primary_markers)
        depth = len(PurePosixPath(name).parts)
        return (0 if dependency else 1, 1 if marked else 0, -depth)

    primary = max(
        safe_weights,
        key=lambda item: (
            primary_rank(item),
            int(item.get("size") or 0),
            str(item.get("filename") or ""),
        ),
    )
    selected = [str(primary["filename"])]
    # Component-style media repositories need one companion from each declared
    # dependency folder. Single-checkpoint repositories naturally select only
    # the primary file.
    primary_name = str(primary.get("filename") or "")
    if len(PurePosixPath(primary_name).parts) > 1:
        for marker in dependency_markers:
            companions = [
                item for item in safe_weights if marker in str(item.get("filename") or "").lower()
            ]
            if companions:
                chosen = min(
                    companions,
                    key=lambda item: (
                        int(item.get("size") or 2**63 - 1),
                        str(item.get("filename") or ""),
                    ),
                )
                filename = str(chosen["filename"])
                if filename not in selected:
                    selected.append(filename)
    return selected


def assess_catalog_install(
    detail: CatalogDetail,
    request: CatalogPreflightRequest,
    settings: Settings,
    system: SystemInfo,
    *,
    selected_file_ids: list[str] | None = None,
) -> CatalogPreflight:
    resolved_revision = detail.revision
    provider = detail.model.provider or "huggingface"
    requested_selected = list(request.selected_files)
    files, ambiguous_files = catalog_file_index(
        detail.files,
        provider=provider,
        prefer_duplicate_variants=not requested_selected,
    )
    selected = list(dict.fromkeys(requested_selected))
    checks: list[CatalogPreflightCheck] = []
    selection_error: str | None = None
    exact_file_ids = list(selected_file_ids or [])

    normalized_requested = [name.casefold() for name in requested_selected]
    if exact_file_ids and provider != "civitai":
        selection_error = "Exact provider file variants are available only for CivitAI."
    elif exact_file_ids and requested_selected:
        selection_error = "Choose files by name or exact provider ID, not both."
    elif exact_file_ids:
        try:
            exact_files = exact_civitai_file_selection(detail.files, exact_file_ids)
        except ExactCivitaiFileSelectionError as exc:
            selection_error = str(exc)
        else:
            selected = [str(item["filename"]) for item in exact_files]
            files.update({str(item["filename"]): item for item in exact_files})
    elif len(normalized_requested) != len(set(normalized_requested)):
        selection_error = (
            "The file selection contains duplicate paths. Run the install check again."
        )
    elif requested_selected and any(name in ambiguous_files for name in selected):
        selection_error = (
            "Several provider files share the selected filename. "
            "Choose an exact file variant and run the install check again."
        )
    elif not requested_selected and provider != "civitai" and ambiguous_files:
        selection_error = (
            "Several catalog files share one path. The provider must return a unique file identity."
        )
    elif not selected:
        try:
            automatic_files = (
                {name: item for name, item in files.items() if _safe_civitai_file(item)}
                if provider == "civitai"
                else files
            )
            selected = (
                _automatic_vllm_selection(detail.files)
                if request.role == "chat" and request.engine == "vllm"
                else automatic_gguf_selection(detail.files, system.memory_total_bytes)
                if request.role == "chat"
                else _automatic_selection(
                    automatic_files,
                    request.role,
                    system.memory_total_bytes,
                    request.auxiliary_kind,
                )
            )
        except GGUFSelectionError as exc:
            selection_error = str(exc)
    if (
        selection_error is None
        and provider == "civitai"
        and any(name in files and not _safe_civitai_file(files[name]) for name in selected)
    ):
        selection_error = (
            "The selected CivitAI file is not a scan-cleared, hash-pinned safetensors file."
        )
    if (
        request.role == "chat"
        and request.engine == "llama.cpp"
        and selected
        and selection_error is None
    ):
        projector = automatic_mmproj_selection(detail.files, selected)
        if projector and projector not in selected:
            selected.append(projector)

    missing = [name for name in selected if name not in files]
    unsafe_paths = [
        name
        for name in selected
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    if (
        request.role == "chat"
        and request.engine == "llama.cpp"
        and selected
        and not missing
        and not unsafe_paths
        and selection_error is None
    ):
        selected_names = set(selected)
        selected_metadata = [
            item for item in detail.files if str(item.get("filename") or "") in selected_names
        ]
        projectors = [
            str(item.get("filename") or "")
            for item in selected_metadata
            if "mmproj" in PurePosixPath(str(item.get("filename") or "")).name.casefold()
        ]
        try:
            primary = validate_gguf_selection(
                selected_metadata,
                require_split_metadata=True,
            )
            if len(projectors) > 1:
                raise GGUFSelectionError(
                    "The chat model selection contains multiple multimodal projectors."
                )
            selected = [*primary, *projectors]
        except GGUFSelectionError as exc:
            selection_error = str(exc)

    if selection_error:
        checks.append(
            _check(
                "selection",
                "Automatic model selection",
                "block",
                selection_error,
            )
        )
    elif not selected:
        checks.append(
            _check(
                "selection",
                "Automatic model selection",
                "block",
                "No safe model file could be selected automatically.",
            )
        )
    elif missing or unsafe_paths:
        detail_text = "Selected files are not present in this revision."
        if unsafe_paths:
            detail_text = "Selected files contain unsafe paths."
        checks.append(_check("selection", "File selection", "block", detail_text))
    else:
        checks.append(
            _check(
                "selection",
                "Automatic model selection",
                "pass",
                (
                    f"LM Atelier selected {len(selected)} safe file"
                    f"{'s' if len(selected) != 1 else ''} for this runtime."
                ),
            )
        )

    unsafe_weights = [
        name for name in selected if PurePosixPath(name).suffix.lower() in _BLOCKED_SUFFIXES
    ]
    checks.append(
        _check(
            "weights",
            "Weight safety",
            "block" if unsafe_weights else "pass",
            (
                "Pickle-compatible weights are blocked: " + ", ".join(unsafe_weights[:3])
                if unsafe_weights
                else "No blocked pickle-compatible weights are selected."
            ),
        )
    )

    expected_sha256 = {
        name: str(files[name].get("sha256")).lower()
        for name in selected
        if name in files
        and isinstance(files[name].get("sha256"), str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(files[name]["sha256"]))
    }
    # CivitAI requests must carry no file sources: the frozen manager accepts
    # exact identity only from the immutable plan and rejects any echoed here.
    file_sources = (
        {}
        if (detail.model.provider or "huggingface") == "civitai"
        else {
            name: CatalogFileSource(
                remote_id=str(files[name]["source_remote_id"]),
                revision=str(files[name]["source_revision"]),
                filename=str(files[name]["source_filename"]),
                size_bytes=(
                    int(files[name]["size"])
                    if isinstance(files[name].get("size"), int)
                    and not isinstance(files[name]["size"], bool)
                    else None
                ),
                sha256=expected_sha256.get(name),
                source_version_id=(
                    str(files[name]["source_version_id"])
                    if files[name].get("source_version_id")
                    else None
                ),
                source_file_id=(
                    str(files[name]["source_file_id"])
                    if files[name].get("source_file_id")
                    else None
                ),
            )
            for name in selected
            if name in files
            and files[name].get("source_remote_id")
            and files[name].get("source_revision")
            and files[name].get("source_filename")
        }
    )
    checksum_complete = bool(selected) and len(expected_sha256) == len(selected)
    checks.append(
        _check(
            "checksum",
            "Checksum metadata",
            "pass" if checksum_complete else "warn",
            (
                "SHA-256 metadata is available for every selected file."
                if checksum_complete
                else "One or more selected files have no public SHA-256 metadata."
            ),
        )
    )

    # The download manager authenticates every CivitAI transfer with the
    # vaulted token, gated or not - a public card without one would pass here
    # and fail mid-queue. Hugging Face keeps its gated-only requirement.
    if provider == "civitai":
        access_blocked = not settings.civitai_token
        access_message = "CivitAI downloads need a CivitAI token supplied to the local service."
    else:
        access_blocked = bool(detail.model.gated and not settings.hf_token)
        access_message = (
            "This gated model needs a Hugging Face token supplied to the local service."
        )
    if access_blocked:
        checks.append(
            _check(
                "access",
                "Repository access",
                "block",
                access_message,
            )
        )
    else:
        checks.append(
            _check(
                "access",
                "Repository access",
                "pass",
                (
                    "A local CivitAI token is available."
                    if provider == "civitai"
                    else "Access is open or a local Hugging Face token is available."
                ),
            )
        )

    checks.append(
        _check(
            "license",
            "License metadata",
            "pass" if detail.model.license_id else "warn",
            (
                f"Hub metadata declares {detail.model.license_id}."
                if detail.model.license_id
                else "No license identifier was found; review the model card before installing."
            ),
        )
    )

    expected_engine = detail.model.required_runtime or (
        "llama.cpp" if request.role == "chat" else "comfyui"
    )
    compatibility_status: Literal["pass", "warn", "block"] = "pass"
    compatibility_detail = "Catalog metadata matches the selected local runtime."
    if request.engine != expected_engine:
        compatibility_status = "block"
        compatibility_detail = f"{request.role} installs require the {expected_engine} runtime."
    elif detail.model.compatibility == "unsupported":
        compatibility_status = "block"
        compatibility_detail = "Catalog metadata marks this model unsupported."
    elif detail.model.compatibility == "advanced_import" and not (
        detail.model.required_runtime == "vllm" and request.engine == "vllm"
    ):
        compatibility_status = "warn"
        compatibility_detail = "This model needs advanced setup or a verified workflow."
    checks.append(
        _check(
            "runtime",
            "Runtime compatibility",
            compatibility_status,
            compatibility_detail,
        )
    )

    known_sizes = [int(files[name].get("size") or 0) for name in selected if name in files]
    download_bytes = sum(known_sizes)
    unknown_sizes = [name for name in selected if name in files and not files[name].get("size")]
    if unknown_sizes:
        checks.append(
            _check(
                "disk",
                "Disk capacity",
                "warn",
                "Some selected file sizes are unknown; exact disk fit cannot be confirmed.",
            )
        )
    elif download_bytes > system.disk_free_bytes:
        checks.append(
            _check(
                "disk",
                "Disk capacity",
                "block",
                "The selected files exceed currently available local disk space.",
            )
        )
    else:
        checks.append(
            _check(
                "disk",
                "Disk capacity",
                "pass",
                "The selected files fit in currently available local disk space.",
            )
        )

    estimated_ram = int(download_bytes * 1.2) + 512 * 1024**2 if download_bytes else None
    estimated_vram = (
        int(download_bytes * 1.25) + 1024**3
        if download_bytes and (request.role != "chat" or request.engine == "vllm")
        else None
    )
    hardware_fit = assess_preflight_hardware_fit(
        request,
        system,
        estimated_ram_bytes=estimated_ram,
        estimated_vram_bytes=estimated_vram,
    )
    checks.append(_hardware_fit_check(hardware_fit))

    checks.append(
        _check(
            "revision",
            "Pinned revision",
            "pass" if resolved_revision != "main" else "warn",
            (
                f"Install is pinned to {resolved_revision}."
                if resolved_revision != "main"
                else "The catalog could not resolve main to an immutable commit."
            ),
        )
    )

    return CatalogPreflight(
        remote_id=detail.model.remote_id,
        revision=resolved_revision,
        selected_files=selected,
        expected_sha256=expected_sha256,
        file_sources=file_sources,
        download_bytes=download_bytes,
        available_disk_bytes=system.disk_free_bytes,
        estimated_ram_bytes=estimated_ram,
        estimated_vram_bytes=estimated_vram,
        can_install=not any(check.status == "block" for check in checks),
        checks=checks,
        auxiliary_kind=request.auxiliary_kind,
        content_rating=detail.model.content_rating,
    )


def assess_preflight_hardware_fit(
    request: CatalogPreflightRequest,
    system: SystemInfo,
    *,
    estimated_ram_bytes: int | None,
    estimated_vram_bytes: int | None,
) -> HardwareFit:
    """Assess calculated catalog fit without turning an estimate into a block."""

    return recommend_hardware_fit(
        capacity_from_system_info(system, runtime_backends=(request.engine,)),
        FitRequirements(
            estimated_system_memory_bytes=estimated_ram_bytes,
            estimated_accelerator_memory_bytes=estimated_vram_bytes,
        ),
    )


def _hardware_fit_check(fit: HardwareFit) -> CatalogPreflightCheck:
    status: Literal["pass", "warn", "block"]
    if fit.status == "unsupported":
        status = "block"
    elif fit.status in {"recommended", "likely"}:
        status = "pass"
    else:
        status = "warn"
    headline = {
        "recommended": "Recommended fit.",
        "likely": "Likely fit.",
        "tight": "Tight fit.",
        "unsupported": "Unsupported by declared hardware requirements.",
        "unknown": "Hardware fit is unknown.",
    }[fit.status]
    reason_text = " ".join(reason.message for reason in fit.reasons if reason.severity != "info")
    if not reason_text:
        reason_text = " ".join(reason.message for reason in fit.reasons)
    detail = f"{headline} {reason_text}".strip()
    return _check("memory", "Hardware fit", status, detail)


def _check(
    check_id: str,
    label: str,
    status: Literal["pass", "warn", "block"],
    detail: str,
) -> CatalogPreflightCheck:
    return CatalogPreflightCheck(id=check_id, label=label, status=status, detail=detail)
