from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .schemas import SystemInfo

FitStatus = Literal["recommended", "likely", "tight", "unsupported", "unknown"]
FitBasis = Literal["unknown", "calculated", "declared", "measured", "tested", "certified"]
ReasonSeverity = Literal["info", "warning", "block"]

_LIKELY_UTILIZATION_LIMIT = 0.90
_RECOMMENDED_MEASURED_UTILIZATION_LIMIT = 0.80


@dataclass(frozen=True, slots=True)
class AcceleratorCapacity:
    backend: str
    name: str
    total_memory_bytes: int
    available_memory_bytes: int | None


@dataclass(frozen=True, slots=True)
class HardwareCapacity:
    platform: str
    architecture: str
    cpu_model: str
    cpu_capabilities: tuple[str, ...]
    system_memory_bytes: int
    system_memory_available_bytes: int | None
    accelerators: tuple[AcceleratorCapacity, ...]
    runtime_backends: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundedSetting:
    key: str
    label: str
    unit: str
    minimum: int
    maximum: int
    preferred_minimum: int
    preferred_maximum: int
    tight_minimum: int
    tight_maximum: int
    user_override: int | None = None


@dataclass(frozen=True, slots=True)
class FitEvidence:
    exact_match: bool
    claim: Literal["measured", "tested", "certified"] = "measured"
    peak_system_memory_bytes: int | None = None
    peak_accelerator_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class FitRequirements:
    supported_platforms: tuple[str, ...] = ()
    supported_architectures: tuple[str, ...] = ()
    required_runtime_backends: tuple[str, ...] = ()
    required_accelerator_backends: tuple[str, ...] = ()
    required_cpu_capabilities: tuple[str, ...] = ()
    minimum_system_memory_bytes: int | None = None
    recommended_system_memory_bytes: int | None = None
    estimated_system_memory_bytes: int | None = None
    minimum_accelerator_memory_bytes: int | None = None
    recommended_accelerator_memory_bytes: int | None = None
    estimated_accelerator_memory_bytes: int | None = None
    concurrent_system_memory_bytes: int = 0
    concurrent_accelerator_memory_bytes: int = 0
    settings: tuple[BoundedSetting, ...] = ()


@dataclass(frozen=True, slots=True)
class FitReason:
    code: str
    severity: ReasonSeverity
    message: str


@dataclass(frozen=True, slots=True)
class FitAlternative:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SettingRecommendation:
    key: str
    label: str
    unit: str
    minimum: int
    maximum: int
    advisory_only: bool
    preserves_user_override: bool


@dataclass(frozen=True, slots=True)
class FitResource:
    kind: Literal["system", "accelerator"]
    capacity_bytes: int
    available_bytes: int | None
    required_bytes: int
    status: FitStatus
    basis: FitBasis
    immediate_pressure: bool


@dataclass(frozen=True, slots=True)
class HardwareFit:
    status: FitStatus
    basis: FitBasis
    evidence_label: Literal["tested", "certified"] | None
    reasons: tuple[FitReason, ...]
    alternatives: tuple[FitAlternative, ...]
    resources: tuple[FitResource, ...]
    settings: tuple[SettingRecommendation, ...]


@dataclass(frozen=True, slots=True)
class HardwareCandidate:
    key: str
    requirements: FitRequirements
    evidence: FitEvidence | None = None


@dataclass(frozen=True, slots=True)
class RankedHardwareCandidate:
    key: str
    fit: HardwareFit


@dataclass(frozen=True, slots=True)
class _MemoryAssessment:
    status: FitStatus
    basis: FitBasis
    reasons: tuple[FitReason, ...]
    immediate_pressure: bool
    capacity_bytes: int
    available_bytes: int | None
    required_bytes: int | None


def capacity_from_system_info(
    system: SystemInfo,
    *,
    runtime_backends: tuple[str, ...] = (),
) -> HardwareCapacity:
    """Translate public system inventory without combining unlike accelerators."""

    cpu_capabilities: list[str] = []
    accelerators: list[AcceleratorCapacity] = []
    for device in system.devices:
        if device.kind.casefold() == "cpu" or (device.backend or "").casefold() == "cpu":
            cpu_capabilities.extend(_declared_cpu_capabilities(device.details))
            continue
        accelerators.append(
            AcceleratorCapacity(
                backend=device.backend or "unknown",
                name=device.name,
                total_memory_bytes=device.total_memory_bytes or 0,
                available_memory_bytes=device.available_memory_bytes,
            )
        )
    return HardwareCapacity(
        platform=getattr(system, "platform", "unknown"),
        architecture=getattr(system, "architecture", "unknown"),
        cpu_model=getattr(system, "cpu_model", "CPU"),
        cpu_capabilities=tuple(dict.fromkeys(cpu_capabilities)),
        system_memory_bytes=system.memory_total_bytes,
        system_memory_available_bytes=getattr(system, "memory_available_bytes", None),
        accelerators=tuple(accelerators),
        runtime_backends=runtime_backends,
    )


def recommend_hardware_fit(
    capacity: HardwareCapacity,
    requirements: FitRequirements,
    *,
    evidence: FitEvidence | None = None,
) -> HardwareFit:
    """Return an advisory fit without weakening compatibility enforcement.

    The caller is responsible for deciding whether recorded evidence still
    matches the exact model, runtime, workflow, driver, and hardware identities.
    Evidence is used here only when ``exact_match`` is true.
    """

    _validate_capacity(capacity)
    _validate_requirements(requirements)
    reasons: list[FitReason] = []
    alternatives: list[FitAlternative] = []

    if requirements.supported_platforms and not _matches(
        capacity.platform, requirements.supported_platforms
    ):
        reasons.append(
            FitReason(
                "platform_unsupported",
                "block",
                "This runtime does not support the current operating system.",
            )
        )
    if requirements.supported_architectures and not _matches(
        capacity.architecture, requirements.supported_architectures
    ):
        reasons.append(
            FitReason(
                "architecture_unsupported",
                "block",
                "This runtime does not support the current processor architecture.",
            )
        )
    if requirements.required_cpu_capabilities and not capacity.cpu_capabilities:
        reasons.append(
            FitReason(
                "cpu_capabilities_unknown",
                "warning",
                "Required processor capabilities could not be confirmed.",
            )
        )
    else:
        missing_cpu_capabilities = _missing_values(
            capacity.cpu_capabilities,
            requirements.required_cpu_capabilities,
        )
        if missing_cpu_capabilities:
            reasons.append(
                FitReason(
                    "cpu_capability_missing",
                    "block",
                    "The processor lacks a capability required by this runtime or model.",
                )
            )
            alternatives.append(
                FitAlternative(
                    "choose_cpu_compatible_variant",
                    "Use a runtime or model variant compatible with this processor.",
                )
            )
    if requirements.required_runtime_backends and not _overlaps(
        capacity.runtime_backends, requirements.required_runtime_backends
    ):
        reasons.append(
            FitReason(
                "runtime_backend_missing",
                "block",
                "A required local runtime backend is not installed or available.",
            )
        )
        alternatives.append(
            FitAlternative(
                "install_supported_runtime",
                "Install a supported local runtime backend before using this model.",
            )
        )
    available_accelerator_backends = tuple(item.backend for item in capacity.accelerators)
    if requirements.required_accelerator_backends and not _overlaps(
        available_accelerator_backends,
        requirements.required_accelerator_backends,
    ):
        reasons.append(
            FitReason(
                "accelerator_backend_missing",
                "block",
                "No compatible accelerator backend was detected.",
            )
        )
        alternatives.append(
            FitAlternative(
                "choose_compatible_backend",
                "Use a build or model variant that supports an available accelerator.",
            )
        )

    matching_evidence = evidence if evidence and evidence.exact_match else None
    if evidence and not evidence.exact_match:
        reasons.append(
            FitReason(
                "evidence_stale",
                "info",
                "Earlier measurements do not match this exact setup and were not reused.",
            )
        )

    accelerator_capacity = _best_accelerator(
        capacity.accelerators,
        requirements.required_accelerator_backends,
    )
    if accelerator_capacity is None and requirements.minimum_accelerator_memory_bytes is not None:
        reasons.append(
            FitReason(
                "accelerator_missing",
                "block",
                "No accelerator was detected for the declared minimum memory requirement.",
            )
        )
        alternatives.append(
            FitAlternative(
                "choose_cpu_compatible_variant",
                "Use a model variant that supports CPU execution.",
            )
        )
    system = _assess_memory(
        kind="system",
        capacity_bytes=capacity.system_memory_bytes,
        available_bytes=capacity.system_memory_available_bytes,
        minimum_bytes=requirements.minimum_system_memory_bytes,
        recommended_bytes=requirements.recommended_system_memory_bytes,
        estimated_bytes=requirements.estimated_system_memory_bytes,
        concurrent_bytes=requirements.concurrent_system_memory_bytes,
        measured_peak_bytes=(
            matching_evidence.peak_system_memory_bytes if matching_evidence else None
        ),
        evidence_claim=matching_evidence.claim if matching_evidence else None,
    )
    accelerator = _assess_memory(
        kind="accelerator",
        capacity_bytes=(accelerator_capacity.total_memory_bytes if accelerator_capacity else 0),
        available_bytes=(
            accelerator_capacity.available_memory_bytes if accelerator_capacity else None
        ),
        minimum_bytes=requirements.minimum_accelerator_memory_bytes,
        recommended_bytes=requirements.recommended_accelerator_memory_bytes,
        estimated_bytes=requirements.estimated_accelerator_memory_bytes,
        concurrent_bytes=requirements.concurrent_accelerator_memory_bytes,
        measured_peak_bytes=(
            matching_evidence.peak_accelerator_memory_bytes if matching_evidence else None
        ),
        evidence_claim=matching_evidence.claim if matching_evidence else None,
    )
    reasons.extend(system.reasons)
    reasons.extend(accelerator.reasons)

    hard_block = any(reason.severity == "block" for reason in reasons)
    has_memory_requirements = any(
        value is not None
        for value in (
            requirements.minimum_system_memory_bytes,
            requirements.recommended_system_memory_bytes,
            requirements.estimated_system_memory_bytes,
            requirements.minimum_accelerator_memory_bytes,
            requirements.recommended_accelerator_memory_bytes,
            requirements.estimated_accelerator_memory_bytes,
            matching_evidence.peak_system_memory_bytes if matching_evidence else None,
            matching_evidence.peak_accelerator_memory_bytes if matching_evidence else None,
        )
    )
    test_claim: Literal["tested", "certified"] | None = None
    if matching_evidence is not None:
        if matching_evidence.claim == "tested":
            test_claim = "tested"
        elif matching_evidence.claim == "certified":
            test_claim = "certified"
    exact_test_claim = test_claim is not None
    compatibility_unknown = any(reason.code == "cpu_capabilities_unknown" for reason in reasons)
    if hard_block:
        status: FitStatus = "unsupported"
    elif compatibility_unknown and not exact_test_claim:
        status = "unknown"
    elif not has_memory_requirements:
        status = "recommended" if exact_test_claim else "unknown"
    else:
        status = _worst_status(system.status, accelerator.status)

    if system.immediate_pressure or accelerator.immediate_pressure:
        alternatives.append(
            FitAlternative(
                "free_current_memory",
                "Close other resource-heavy apps or unload idle models before starting.",
            )
        )
    if status in {"tight", "unsupported"}:
        alternatives.append(
            FitAlternative(
                "choose_smaller_variant",
                "Choose a smaller or more compressed model variant.",
            )
        )

    setting_recommendations = _setting_recommendations(requirements.settings, status)
    if status == "tight" and setting_recommendations:
        alternatives.append(
            FitAlternative(
                "use_safer_settings",
                "Use the suggested lower-cost settings for the first run.",
            )
        )

    basis = _strongest_basis(system.basis, accelerator.basis)
    evidence_label: Literal["tested", "certified"] | None = None
    if test_claim is not None and not hard_block:
        evidence_label = test_claim
        basis = _strongest_basis(basis, test_claim)
    return HardwareFit(
        status=status,
        basis=basis,
        evidence_label=evidence_label,
        reasons=tuple(reasons),
        alternatives=_deduplicate_alternatives(alternatives),
        resources=_fit_resources(system, accelerator),
        settings=setting_recommendations,
    )


def rank_hardware_candidates(
    capacity: HardwareCapacity,
    candidates: tuple[HardwareCandidate, ...],
) -> tuple[RankedHardwareCandidate, ...]:
    """Rank advisory candidates without replacing an explicit user choice.

    General fit is ordered before evidence strength. Volatile free-memory
    pressure deliberately does not affect this order; it remains a launch-time
    warning on each fit. Equal candidates preserve the caller's catalog order.
    """

    seen: set[str] = set()
    ranked: list[tuple[int, RankedHardwareCandidate]] = []
    for position, candidate in enumerate(candidates):
        key = candidate.key
        if not key or key != key.strip():
            raise ValueError(
                "Hardware candidate keys must not be blank or contain surrounding whitespace."
            )
        if key in seen:
            raise ValueError(f"Duplicate hardware candidate key: {key}")
        seen.add(key)
        ranked.append(
            (
                position,
                RankedHardwareCandidate(
                    key=key,
                    fit=recommend_hardware_fit(
                        capacity,
                        candidate.requirements,
                        evidence=candidate.evidence,
                    ),
                ),
            )
        )
    ranked.sort(key=lambda item: _candidate_rank_key(item[1].fit, item[0]))
    return tuple(item for _, item in ranked)


def _assess_memory(
    *,
    kind: Literal["system", "accelerator"],
    capacity_bytes: int,
    available_bytes: int | None,
    minimum_bytes: int | None,
    recommended_bytes: int | None,
    estimated_bytes: int | None,
    concurrent_bytes: int,
    measured_peak_bytes: int | None,
    evidence_claim: Literal["measured", "tested", "certified"] | None,
) -> _MemoryAssessment:
    label = "system memory" if kind == "system" else "accelerator memory"
    reasons: list[FitReason] = []
    basis: FitBasis = "unknown"
    demanded = [
        value
        for value in (minimum_bytes, recommended_bytes, estimated_bytes, measured_peak_bytes)
        if value is not None
    ]
    if not demanded:
        return _MemoryAssessment(
            "recommended",
            basis,
            (),
            False,
            capacity_bytes,
            available_bytes,
            None,
        )

    load_bytes = max(demanded) + concurrent_bytes
    if capacity_bytes <= 0:
        reasons.append(
            FitReason(
                f"{kind}_memory_unknown",
                "warning",
                f"The required {label} capacity could not be confirmed.",
            )
        )
        return _MemoryAssessment(
            "unknown",
            basis,
            tuple(reasons),
            False,
            capacity_bytes,
            available_bytes,
            load_bytes,
        )

    if minimum_bytes is not None and minimum_bytes + concurrent_bytes > capacity_bytes:
        reasons.append(
            FitReason(
                f"{kind}_memory_below_minimum",
                "block",
                f"Available hardware is below the declared minimum {label} requirement.",
            )
        )
        return _MemoryAssessment(
            "unsupported",
            "declared",
            tuple(reasons),
            False,
            capacity_bytes,
            available_bytes,
            load_bytes,
        )

    utilization = load_bytes / capacity_bytes
    if measured_peak_bytes is not None and evidence_claim is not None:
        basis = evidence_claim
        status: FitStatus = (
            "recommended" if utilization <= _RECOMMENDED_MEASURED_UTILIZATION_LIMIT else "tight"
        )
        reasons.append(
            FitReason(
                f"{kind}_memory_measured",
                "info" if status == "recommended" else "warning",
                f"An exact matching run measured this setup's {label} use.",
            )
        )
    elif recommended_bytes is not None:
        basis = "declared"
        declared_fit = recommended_bytes + concurrent_bytes <= capacity_bytes
        calculated_fit = (
            estimated_bytes is None or estimated_bytes + concurrent_bytes <= capacity_bytes
        )
        if declared_fit and calculated_fit:
            status = "recommended"
            message = f"Hardware meets the declared recommended {label} capacity."
        else:
            status = "tight"
            message = (
                f"Hardware meets the minimum but the recommended or calculated {label} "
                "need leaves too little headroom."
            )
        reasons.append(
            FitReason(
                f"{kind}_memory_declared",
                "info" if status == "recommended" else "warning",
                message,
            )
        )
    else:
        basis = "calculated"
        status = "likely" if utilization <= _LIKELY_UTILIZATION_LIMIT else "tight"
        reasons.append(
            FitReason(
                f"{kind}_memory_estimated",
                "info" if status == "likely" else "warning",
                (
                    f"The calculated {label} estimate leaves practical headroom."
                    if status == "likely"
                    else f"The calculated {label} estimate leaves little headroom."
                ),
            )
        )

    immediate_pressure = available_bytes is not None and load_bytes > available_bytes
    if immediate_pressure:
        reasons.append(
            FitReason(
                f"{kind}_memory_busy",
                "warning",
                f"The model fits total {label}, but currently free capacity is lower.",
            )
        )
    return _MemoryAssessment(
        status,
        basis,
        tuple(reasons),
        immediate_pressure,
        capacity_bytes,
        available_bytes,
        load_bytes,
    )


def _setting_recommendations(
    settings: tuple[BoundedSetting, ...],
    status: FitStatus,
) -> tuple[SettingRecommendation, ...]:
    recommendations: list[SettingRecommendation] = []
    for setting in settings:
        _validate_setting(setting)
        if status == "tight":
            lower = setting.tight_minimum
            upper = setting.tight_maximum
        else:
            lower = setting.preferred_minimum
            upper = setting.preferred_maximum
        recommendations.append(
            SettingRecommendation(
                key=setting.key,
                label=setting.label,
                unit=setting.unit,
                minimum=lower,
                maximum=upper,
                advisory_only=True,
                preserves_user_override=setting.user_override is not None,
            )
        )
    return tuple(recommendations)


def _declared_cpu_capabilities(details: dict[str, Any]) -> tuple[str, ...]:
    capabilities: list[str] = []
    for key in ("capabilities", "features", "flags"):
        value = details.get(key)
        if isinstance(value, str):
            capabilities.extend(value.split())
        elif isinstance(value, list):
            capabilities.extend(item for item in value if isinstance(item, str))
    return tuple(item.casefold() for item in capabilities if item.strip())


def _matches(value: str, expected: tuple[str, ...]) -> bool:
    return value.casefold() in {item.casefold() for item in expected}


def _overlaps(available: tuple[str, ...], required: tuple[str, ...]) -> bool:
    normalized = {item.casefold() for item in available}
    return bool(normalized & {item.casefold() for item in required})


def _missing_values(available: tuple[str, ...], required: tuple[str, ...]) -> set[str]:
    normalized = {item.casefold() for item in available}
    return {item.casefold() for item in required} - normalized


def _best_accelerator(
    accelerators: tuple[AcceleratorCapacity, ...],
    required_backends: tuple[str, ...],
) -> AcceleratorCapacity | None:
    required = {item.casefold() for item in required_backends}
    compatible = [
        item for item in accelerators if not required or item.backend.casefold() in required
    ]
    return max(
        compatible,
        key=lambda item: (item.total_memory_bytes, item.available_memory_bytes or 0),
        default=None,
    )


def _worst_status(first: FitStatus, second: FitStatus) -> FitStatus:
    order: dict[FitStatus, int] = {
        "recommended": 0,
        "likely": 1,
        "tight": 2,
        "unknown": 3,
        "unsupported": 4,
    }
    return first if order[first] >= order[second] else second


def _candidate_rank_key(fit: HardwareFit, position: int) -> tuple[int, int, float, int]:
    status_order: dict[FitStatus, int] = {
        "recommended": 0,
        "likely": 1,
        "tight": 2,
        "unknown": 3,
        "unsupported": 4,
    }
    basis_order: dict[FitBasis, int] = {
        "certified": 0,
        "tested": 1,
        "measured": 2,
        "declared": 3,
        "calculated": 4,
        "unknown": 5,
    }
    utilizations = [
        resource.required_bytes / resource.capacity_bytes
        for resource in fit.resources
        if resource.capacity_bytes > 0
    ]
    utilization = max(utilizations, default=float("inf") if fit.resources else 0.0)
    return (status_order[fit.status], basis_order[fit.basis], utilization, position)


def _fit_resources(
    system: _MemoryAssessment,
    accelerator: _MemoryAssessment,
) -> tuple[FitResource, ...]:
    resources: list[FitResource] = []
    assessments: tuple[tuple[Literal["system", "accelerator"], _MemoryAssessment], ...] = (
        ("system", system),
        ("accelerator", accelerator),
    )
    for kind, assessment in assessments:
        if assessment.required_bytes is None:
            continue
        resources.append(
            FitResource(
                kind=kind,
                capacity_bytes=assessment.capacity_bytes,
                available_bytes=assessment.available_bytes,
                required_bytes=assessment.required_bytes,
                status=assessment.status,
                basis=assessment.basis,
                immediate_pressure=assessment.immediate_pressure,
            )
        )
    return tuple(resources)


def _strongest_basis(first: FitBasis, second: FitBasis) -> FitBasis:
    order: dict[FitBasis, int] = {
        "unknown": 0,
        "calculated": 1,
        "declared": 2,
        "measured": 3,
        "tested": 4,
        "certified": 5,
    }
    return first if order[first] >= order[second] else second


def _deduplicate_alternatives(
    alternatives: list[FitAlternative],
) -> tuple[FitAlternative, ...]:
    unique: dict[str, FitAlternative] = {}
    for alternative in alternatives:
        unique.setdefault(alternative.code, alternative)
    return tuple(unique.values())


def _validate_capacity(capacity: HardwareCapacity) -> None:
    values = (
        capacity.system_memory_bytes,
        capacity.system_memory_available_bytes,
    )
    if any(value is not None and value < 0 for value in values):
        raise ValueError("Hardware memory values must not be negative.")
    if any(
        item.total_memory_bytes < 0
        or (item.available_memory_bytes is not None and item.available_memory_bytes < 0)
        for item in capacity.accelerators
    ):
        raise ValueError("Accelerator memory values must not be negative.")


def _validate_requirements(requirements: FitRequirements) -> None:
    values = (
        requirements.minimum_system_memory_bytes,
        requirements.recommended_system_memory_bytes,
        requirements.estimated_system_memory_bytes,
        requirements.minimum_accelerator_memory_bytes,
        requirements.recommended_accelerator_memory_bytes,
        requirements.estimated_accelerator_memory_bytes,
        requirements.concurrent_system_memory_bytes,
        requirements.concurrent_accelerator_memory_bytes,
    )
    if any(value is not None and value < 0 for value in values):
        raise ValueError("Hardware requirements must not be negative.")
    pairs = (
        (
            requirements.minimum_system_memory_bytes,
            requirements.recommended_system_memory_bytes,
        ),
        (
            requirements.minimum_accelerator_memory_bytes,
            requirements.recommended_accelerator_memory_bytes,
        ),
    )
    if any(
        minimum is not None and recommended is not None and recommended < minimum
        for minimum, recommended in pairs
    ):
        raise ValueError("Recommended memory must not be below the declared minimum.")


def _validate_setting(setting: BoundedSetting) -> None:
    ranges = (
        (setting.minimum, setting.maximum),
        (setting.preferred_minimum, setting.preferred_maximum),
        (setting.tight_minimum, setting.tight_maximum),
    )
    if any(lower > upper for lower, upper in ranges):
        raise ValueError(f"Invalid bounded setting range: {setting.key}")
    if not (
        setting.minimum <= setting.preferred_minimum <= setting.preferred_maximum <= setting.maximum
        and setting.minimum <= setting.tight_minimum <= setting.tight_maximum <= setting.maximum
    ):
        raise ValueError(f"Suggested ranges escape declared bounds: {setting.key}")
