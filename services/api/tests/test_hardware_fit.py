from __future__ import annotations

from dataclasses import replace

import pytest

from local_lm.hardware_fit import (
    AcceleratorCapacity,
    BoundedSetting,
    FitEvidence,
    FitRequirements,
    HardwareCandidate,
    HardwareCapacity,
    capacity_from_system_info,
    rank_hardware_candidates,
    recommend_hardware_fit,
)
from local_lm.preflight import _hardware_fit_check, assess_preflight_hardware_fit
from local_lm.schemas import (
    CatalogPreflightRequest,
    DeviceInfo,
    PlatformAssessment,
    SystemInfo,
)

_GIB = 1024**3


def _capacity(
    *,
    system: int = 32 * _GIB,
    system_free: int | None = 24 * _GIB,
    accelerator: int = 16 * _GIB,
    accelerator_free: int | None = 14 * _GIB,
    accelerator_backends: tuple[str, ...] = ("cuda",),
    runtime_backends: tuple[str, ...] = ("llama.cpp", "comfyui"),
) -> HardwareCapacity:
    return HardwareCapacity(
        platform="windows",
        architecture="amd64",
        cpu_model="Example CPU",
        cpu_capabilities=("avx2",),
        system_memory_bytes=system,
        system_memory_available_bytes=system_free,
        accelerators=tuple(
            AcceleratorCapacity(
                backend=backend,
                name=f"Example {backend} GPU",
                total_memory_bytes=accelerator,
                available_memory_bytes=accelerator_free,
            )
            for backend in accelerator_backends
        ),
        runtime_backends=runtime_backends,
    )


def test_system_inventory_conversion_preserves_each_device_and_cpu_capabilities() -> None:
    system = SystemInfo(
        platform="Windows",
        platform_release="11",
        distribution="Windows",
        distribution_version="11",
        architecture="AMD64",
        python_version="3.12",
        cpu_model="Example CPU",
        cpu_count=16,
        memory_total_bytes=32 * _GIB,
        memory_available_bytes=20 * _GIB,
        disk_total_bytes=100 * _GIB,
        disk_free_bytes=50 * _GIB,
        ffmpeg_available=False,
        devices=[
            DeviceInfo(
                id="cuda:0",
                name="CUDA GPU",
                kind="gpu",
                total_memory_bytes=16 * _GIB,
                available_memory_bytes=12 * _GIB,
                backend="cuda",
            ),
            DeviceInfo(
                id="directml:0",
                name="DirectML GPU",
                kind="gpu",
                total_memory_bytes=8 * _GIB,
                available_memory_bytes=7 * _GIB,
                backend="directml",
            ),
            DeviceInfo(
                id="cpu:0",
                name="Example CPU",
                kind="cpu",
                total_memory_bytes=32 * _GIB,
                available_memory_bytes=20 * _GIB,
                backend="cpu",
                details={"flags": ["AVX2", "FMA", "AVX2"]},
            ),
            DeviceInfo(
                id="llama:cpu",
                name="llama.cpp CPU",
                kind="accelerator",
                total_memory_bytes=32 * _GIB,
                available_memory_bytes=20 * _GIB,
                backend="CPU",
                details={"features": "AVX2"},
            ),
        ],
        support=PlatformAssessment(
            platform_status="target",
            platform_label="Windows",
            accelerator_status="primary",
            accelerator_label="NVIDIA",
            certification_status="hardware-pending",
            chat_ready=True,
            reference_media_ready=True,
        ),
    )

    capacity = capacity_from_system_info(system, runtime_backends=("comfyui",))

    assert capacity.cpu_model == "Example CPU"
    assert capacity.cpu_capabilities == ("avx2", "fma")
    assert [(item.backend, item.total_memory_bytes) for item in capacity.accelerators] == [
        ("cuda", 16 * _GIB),
        ("directml", 8 * _GIB),
    ]
    assert capacity.runtime_backends == ("comfyui",)


def test_declared_recommended_capacity_is_recommended() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(
            supported_platforms=("Windows",),
            required_runtime_backends=("COMFYUI",),
            required_accelerator_backends=("CUDA",),
            minimum_system_memory_bytes=16 * _GIB,
            recommended_system_memory_bytes=24 * _GIB,
            minimum_accelerator_memory_bytes=8 * _GIB,
            recommended_accelerator_memory_bytes=12 * _GIB,
        ),
    )

    assert result.status == "recommended"
    assert result.basis == "declared"
    assert result.evidence_label is None


def test_calculated_fit_is_likely_and_never_claims_tested() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(estimated_accelerator_memory_bytes=10 * _GIB),
    )

    assert result.status == "likely"
    assert result.basis == "calculated"
    assert result.evidence_label is None


def test_calculated_fit_with_little_headroom_is_tight() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(estimated_accelerator_memory_bytes=15 * _GIB),
    )

    assert result.status == "tight"
    assert any(reason.code == "accelerator_memory_estimated" for reason in result.reasons)
    assert any(item.code == "choose_smaller_variant" for item in result.alternatives)


def test_calculated_load_can_tighten_an_outdated_declared_recommendation() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(
            minimum_accelerator_memory_bytes=8 * _GIB,
            recommended_accelerator_memory_bytes=12 * _GIB,
            estimated_accelerator_memory_bytes=18 * _GIB,
        ),
    )

    assert result.status == "tight"
    assert result.basis == "declared"


def test_declared_minimum_that_cannot_fit_is_unsupported() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(minimum_accelerator_memory_bytes=20 * _GIB),
    )

    assert result.status == "unsupported"
    assert result.basis == "declared"
    assert any(reason.severity == "block" for reason in result.reasons)


def test_missing_estimated_accelerator_capacity_is_unknown_not_unsupported() -> None:
    result = recommend_hardware_fit(
        _capacity(
            accelerator=0,
            accelerator_free=None,
            accelerator_backends=(),
        ),
        FitRequirements(estimated_accelerator_memory_bytes=8 * _GIB),
    )

    assert result.status == "unknown"
    assert result.basis == "unknown"


def test_current_free_memory_is_a_warning_not_general_fit_downgrade() -> None:
    result = recommend_hardware_fit(
        _capacity(accelerator_free=4 * _GIB),
        FitRequirements(
            minimum_accelerator_memory_bytes=8 * _GIB,
            recommended_accelerator_memory_bytes=12 * _GIB,
        ),
    )

    assert result.status == "recommended"
    assert any(reason.code == "accelerator_memory_busy" for reason in result.reasons)
    assert any(item.code == "free_current_memory" for item in result.alternatives)


def test_stale_evidence_is_ignored() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(estimated_accelerator_memory_bytes=15 * _GIB),
        evidence=FitEvidence(
            exact_match=False,
            claim="certified",
            peak_accelerator_memory_bytes=6 * _GIB,
        ),
    )

    assert result.status == "tight"
    assert result.basis == "calculated"
    assert result.evidence_label is None
    assert any(reason.code == "evidence_stale" for reason in result.reasons)


def test_exact_test_evidence_can_be_labeled_tested() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(
            exact_match=True,
            claim="tested",
            peak_accelerator_memory_bytes=10 * _GIB,
        ),
    )

    assert result.status == "recommended"
    assert result.basis == "tested"
    assert result.evidence_label == "tested"


def test_exact_certification_label_requires_matching_evidence() -> None:
    exact = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(
            exact_match=True,
            claim="certified",
            peak_accelerator_memory_bytes=10 * _GIB,
        ),
    )
    stale = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(
            exact_match=False,
            claim="certified",
            peak_accelerator_memory_bytes=10 * _GIB,
        ),
    )

    assert exact.evidence_label == "certified"
    assert stale.evidence_label is None
    assert stale.status == "unknown"


def test_exact_recorded_matrix_can_be_labeled_without_a_memory_peak() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(exact_match=True, claim="certified"),
    )

    assert result.status == "recommended"
    assert result.basis == "certified"
    assert result.evidence_label == "certified"


def test_unmeasured_non_test_evidence_does_not_make_a_fit_claim() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(exact_match=True, claim="measured"),
    )

    assert result.status == "unknown"
    assert result.basis == "unknown"
    assert result.evidence_label is None


def test_unreported_cpu_capabilities_remain_unknown() -> None:
    result = recommend_hardware_fit(
        replace(_capacity(), cpu_capabilities=()),
        FitRequirements(
            required_cpu_capabilities=("avx2",),
            estimated_system_memory_bytes=8 * _GIB,
        ),
    )

    assert result.status == "unknown"
    assert any(reason.code == "cpu_capabilities_unknown" for reason in result.reasons)
    assert not any(reason.severity == "block" for reason in result.reasons)


def test_missing_required_cpu_capability_is_unsupported() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(required_cpu_capabilities=("avx512",)),
    )

    assert result.status == "unsupported"
    assert any(reason.code == "cpu_capability_missing" for reason in result.reasons)
    assert any(
        alternative.code == "choose_cpu_compatible_variant" for alternative in result.alternatives
    )


def test_required_backends_fail_closed_without_model_name_branches() -> None:
    result = recommend_hardware_fit(
        _capacity(accelerator_backends=("directml",), runtime_backends=("llama.cpp",)),
        FitRequirements(
            required_runtime_backends=("comfyui",),
            required_accelerator_backends=("cuda",),
        ),
    )

    assert result.status == "unsupported"
    assert {reason.code for reason in result.reasons} >= {
        "runtime_backend_missing",
        "accelerator_backend_missing",
    }


def test_memory_is_taken_from_the_compatible_device_not_another_backend() -> None:
    capacity = HardwareCapacity(
        platform="windows",
        architecture="amd64",
        cpu_model="Example CPU",
        cpu_capabilities=("avx2",),
        system_memory_bytes=32 * _GIB,
        system_memory_available_bytes=24 * _GIB,
        accelerators=(
            AcceleratorCapacity(
                backend="directml",
                name="Large DirectML GPU",
                total_memory_bytes=24 * _GIB,
                available_memory_bytes=22 * _GIB,
            ),
            AcceleratorCapacity(
                backend="cuda",
                name="Small CUDA GPU",
                total_memory_bytes=8 * _GIB,
                available_memory_bytes=8 * _GIB,
            ),
        ),
        runtime_backends=("comfyui",),
    )

    result = recommend_hardware_fit(
        capacity,
        FitRequirements(
            required_accelerator_backends=("cuda",),
            minimum_accelerator_memory_bytes=12 * _GIB,
        ),
    )

    assert result.status == "unsupported"
    assert any(reason.code == "accelerator_memory_below_minimum" for reason in result.reasons)


def test_tight_fit_uses_bounded_safer_ranges_without_overwriting_user_choice() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(
            estimated_accelerator_memory_bytes=15 * _GIB,
            settings=(
                BoundedSetting(
                    key="resolution",
                    label="Resolution",
                    unit="pixels",
                    minimum=512,
                    maximum=2048,
                    preferred_minimum=768,
                    preferred_maximum=1024,
                    tight_minimum=512,
                    tight_maximum=768,
                    user_override=1536,
                ),
            ),
        ),
    )

    assert result.status == "tight"
    assert result.settings[0].minimum == 512
    assert result.settings[0].maximum == 768
    assert result.settings[0].advisory_only is True
    assert result.settings[0].preserves_user_override is True


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="escape declared bounds"):
        recommend_hardware_fit(
            _capacity(),
            FitRequirements(
                settings=(
                    BoundedSetting(
                        key="steps",
                        label="Steps",
                        unit="steps",
                        minimum=1,
                        maximum=10,
                        preferred_minimum=4,
                        preferred_maximum=12,
                        tight_minimum=1,
                        tight_maximum=4,
                    ),
                )
            ),
        )


def test_candidates_rank_by_general_fit_then_strongest_evidence() -> None:
    ranked = rank_hardware_candidates(
        _capacity(),
        (
            HardwareCandidate("unknown", FitRequirements()),
            HardwareCandidate(
                "calculated-likely",
                FitRequirements(estimated_accelerator_memory_bytes=8 * _GIB),
            ),
            HardwareCandidate(
                "declared-recommended",
                FitRequirements(recommended_accelerator_memory_bytes=8 * _GIB),
            ),
            HardwareCandidate(
                "tested-recommended",
                FitRequirements(),
                FitEvidence(exact_match=True, claim="tested"),
            ),
            HardwareCandidate(
                "tight",
                FitRequirements(estimated_accelerator_memory_bytes=15 * _GIB),
            ),
            HardwareCandidate(
                "unsupported",
                FitRequirements(minimum_accelerator_memory_bytes=20 * _GIB),
            ),
        ),
    )

    assert [item.key for item in ranked] == [
        "tested-recommended",
        "declared-recommended",
        "calculated-likely",
        "tight",
        "unknown",
        "unsupported",
    ]


def test_candidate_ranking_keeps_free_memory_as_a_warning_not_a_model_reorder() -> None:
    ranked = rank_hardware_candidates(
        _capacity(accelerator_free=2 * _GIB),
        (
            HardwareCandidate(
                "first",
                FitRequirements(recommended_accelerator_memory_bytes=8 * _GIB),
            ),
            HardwareCandidate(
                "second",
                FitRequirements(recommended_accelerator_memory_bytes=8 * _GIB),
            ),
        ),
    )

    assert [item.key for item in ranked] == ["first", "second"]
    assert all(
        any(reason.code == "accelerator_memory_busy" for reason in item.fit.reasons)
        for item in ranked
    )


@pytest.mark.parametrize(
    "candidates, message",
    [
        ((HardwareCandidate(" ", FitRequirements()),), "must not be blank"),
        (
            (
                HardwareCandidate("same", FitRequirements()),
                HardwareCandidate("same", FitRequirements()),
            ),
            "Duplicate hardware candidate key",
        ),
    ],
)
def test_candidate_ranking_requires_stable_unique_keys(
    candidates: tuple[HardwareCandidate, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rank_hardware_candidates(_capacity(), candidates)


def test_unknown_estimated_preflight_fit_remains_advisory() -> None:
    system = SystemInfo.model_construct(
        platform="Windows",
        architecture="AMD64",
        cpu_model="Example CPU",
        memory_total_bytes=16 * _GIB,
        memory_available_bytes=12 * _GIB,
        devices=[],
    )
    fit = assess_preflight_hardware_fit(
        CatalogPreflightRequest(role="image", engine="comfyui"),
        system,
        estimated_ram_bytes=4 * _GIB,
        estimated_vram_bytes=8 * _GIB,
    )

    assert fit.status == "unknown"
    check = _hardware_fit_check(fit)
    assert check.status == "warn"
    assert check.detail.startswith("Hardware fit is unknown.")


def test_declared_unsupported_fit_blocks_preflight() -> None:
    fit = recommend_hardware_fit(
        _capacity(),
        FitRequirements(minimum_accelerator_memory_bytes=20 * _GIB),
    )

    check = _hardware_fit_check(fit)

    assert fit.status == "unsupported"
    assert check.status == "block"
    assert check.detail.startswith("Unsupported by declared hardware requirements.")


def test_fit_exposes_structured_total_capacity_headroom() -> None:
    result = recommend_hardware_fit(
        _capacity(system_free=6 * _GIB, accelerator_free=4 * _GIB),
        FitRequirements(
            estimated_system_memory_bytes=8 * _GIB,
            concurrent_system_memory_bytes=2 * _GIB,
            estimated_accelerator_memory_bytes=10 * _GIB,
            concurrent_accelerator_memory_bytes=1 * _GIB,
        ),
    )

    assert [(item.kind, item.required_bytes, item.capacity_bytes) for item in result.resources] == [
        ("system", 10 * _GIB, 32 * _GIB),
        ("accelerator", 11 * _GIB, 16 * _GIB),
    ]
    assert [item.available_bytes for item in result.resources] == [6 * _GIB, 4 * _GIB]
    assert all(item.immediate_pressure for item in result.resources)


def test_same_fit_and_basis_prefers_more_total_capacity_headroom() -> None:
    ranked = rank_hardware_candidates(
        _capacity(),
        (
            HardwareCandidate(
                "larger",
                FitRequirements(estimated_accelerator_memory_bytes=12 * _GIB),
            ),
            HardwareCandidate(
                "smaller",
                FitRequirements(estimated_accelerator_memory_bytes=8 * _GIB),
            ),
        ),
    )

    assert [item.key for item in ranked] == ["smaller", "larger"]


def test_present_accelerator_with_unknown_capacity_is_unknown() -> None:
    result = recommend_hardware_fit(
        _capacity(accelerator=0, accelerator_free=None),
        FitRequirements(
            required_accelerator_backends=("cuda",),
            minimum_accelerator_memory_bytes=8 * _GIB,
        ),
    )

    assert result.status == "unknown"
    assert any(reason.code == "accelerator_memory_unknown" for reason in result.reasons)
    assert not any(reason.severity == "block" for reason in result.reasons)


def test_declared_accelerator_minimum_without_a_device_is_unsupported() -> None:
    result = recommend_hardware_fit(
        _capacity(accelerator_backends=()),
        FitRequirements(minimum_accelerator_memory_bytes=8 * _GIB),
    )

    assert result.status == "unsupported"
    assert any(reason.code == "accelerator_missing" for reason in result.reasons)
