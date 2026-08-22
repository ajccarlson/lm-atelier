from __future__ import annotations

import json
import struct
from dataclasses import replace

import pytest

from local_lm.db import SessionLocal
from local_lm.model_manifests import (
    MAX_METADATA_BYTES,
    InspectedComponent,
    ModelManifestError,
    ModelManifestInspection,
    inspect_repository_metadata,
)
from local_lm.model_planner import persist_install_plan, resolve_install_plan


def _safetensors(tensor_names: list[str], metadata: dict[str, str] | None = None) -> bytes:
    header = {
        **{
            name: {"dtype": "F16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
            for index, name in enumerate(tensor_names)
        },
        "__metadata__": metadata or {},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return len(encoded).to_bytes(8, "little") + encoded


def _gguf(fields: dict[str, str], *, tensors: int = 12) -> bytes:
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, tensors, len(fields)))
    for key, value in fields.items():
        encoded_key = key.encode()
        encoded_value = value.encode()
        payload.extend(struct.pack("<Q", len(encoded_key)))
        payload.extend(encoded_key)
        payload.extend(struct.pack("<I", 8))
        payload.extend(struct.pack("<Q", len(encoded_value)))
        payload.extend(encoded_value)
    return bytes(payload)


def _gguf_with_large_string_array(*, item_count: int, architecture: str) -> bytes:
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, 12, 2))
    array_key = b"tokenizer.ggml.tokens"
    payload.extend(struct.pack("<Q", len(array_key)))
    payload.extend(array_key)
    payload.extend(struct.pack("<IIQ", 9, 8, item_count))
    payload.extend(struct.pack("<Q", 0) * item_count)
    architecture_key = b"general.architecture"
    encoded_architecture = architecture.encode()
    payload.extend(struct.pack("<Q", len(architecture_key)))
    payload.extend(architecture_key)
    payload.extend(struct.pack("<I", 8))
    payload.extend(struct.pack("<Q", len(encoded_architecture)))
    payload.extend(encoded_architecture)
    return bytes(payload)


def test_static_inspector_resolves_unknown_repository_by_config_and_headers() -> None:
    files = {
        "config.json": json.dumps(
            {
                "_class_name": "StableDiffusionXLPipeline",
                "architectures": ["UNet2DConditionModel"],
            }
        ).encode(),
        "weights/model.safetensors": _safetensors(
            [
                "model.diffusion_model.input_blocks.0.weight",
                "conditioner.embedders.1.model.text_projection",
                "first_stage_model.decoder.conv.weight",
            ]
        ),
    }

    inspection = inspect_repository_metadata(
        files,
        ["weights/model.safetensors"],
        role="image",
    )

    assert inspection.family == "stable-diffusion-xl"
    assert inspection.architecture == "StableDiffusionXLPipeline"
    assert inspection.components[0].kind == "checkpoint"
    assert inspection.components[0].target_folder == "checkpoints"


def test_minimax_h3_family_is_video_only_and_does_not_grant_a_workflow() -> None:
    files = {
        "config.json": json.dumps({"_class_name": "MiniMaxH3"}).encode(),
        "model.safetensors": _safetensors(["model.diffusion_model.block.weight"]),
    }

    inspection = inspect_repository_metadata(files, ["model.safetensors"], role="video")
    wrong_role = inspect_repository_metadata(files, ["model.safetensors"], role="image")
    unrelated = inspect_repository_metadata(
        {"config.json": json.dumps({"_class_name": "MiniMaxTextModel"}).encode()},
        ["model.safetensors"],
        role="video",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/minimax-h3-metadata",
        revision="3" * 40,
        role="video",
        engine="comfyui",
        selected_files=[{"filename": "model.safetensors", "size": 4_096, "sha256": "4" * 64}],
        inspection=inspection,
    )

    assert inspection.architecture == "MiniMaxH3"
    assert inspection.family == "minimax-h3"
    assert wrong_role.family is None
    assert unrelated.family is None
    assert plan.family == "minimax-h3"
    assert plan.compatibility == "trusted_extension_required"
    assert plan.failure_code == "workflow_contract_missing"
    assert plan.activation_probe["required"] is False
    assert replace(plan, family=None).plan_hash != plan.plan_hash


@pytest.mark.parametrize(
    "declaration",
    ["NotMiniMaxH3Custom", "MiniMaxH30", "prefix_minimax_h3_evil"],
)
def test_minimax_h3_family_refuses_substring_declarations(declaration: str) -> None:
    inspection = inspect_repository_metadata(
        {"config.json": json.dumps({"_class_name": declaration}).encode()},
        ["model.safetensors"],
        role="video",
    )

    assert inspection.architecture == declaration
    assert inspection.family is None


def test_static_inspector_distinguishes_lora_from_primary_checkpoint() -> None:
    inspection = inspect_repository_metadata(
        {
            "adapter.safetensors": _safetensors(
                ["lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight"],
                {"ss_network_module": "networks.lora"},
            )
        },
        ["adapter.safetensors"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/unknown-adapter",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "adapter.safetensors",
                "size": 1_024,
                "sha256": "b" * 64,
            }
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="c" * 64,
    )

    assert inspection.components[0].kind == "lora"
    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "auxiliary_asset_not_primary"


def test_install_plan_freezes_bounded_provider_trigger_words() -> None:
    inspection = inspect_repository_metadata(
        {"model.safetensors": _safetensors(["model.diffusion_model.input_blocks.0.weight"])},
        ["model.safetensors"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/trigger-model",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "model.safetensors",
                "size": 1_024,
                "sha256": "b" * 64,
                "metadata": {
                    "trained_words": [
                        " portrait-style ",
                        "Portrait-Style",
                        "soft light",
                    ],
                    "trigger_words": ["phone candid"],
                },
            }
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="c" * 64,
    )

    assert plan.runtime_contract["trigger_words"] == [
        "portrait-style",
        "soft light",
        "phone candid",
    ]


def test_static_inspector_recognizes_nextdit_diffusion_tensor_layout() -> None:
    inspection = inspect_repository_metadata(
        {
            "Krea2_Raw_convrot_int8mixed.safetensors": _safetensors(
                [
                    "blocks.0.attn.qkv.weight",
                    "first.weight",
                    "last.weight",
                    "tmlp.0.weight",
                    "tproj.weight",
                    "txtfusion.0.attn.qkv.weight",
                ],
                {"_quantization_metadata": "{}"},
            )
        },
        ["Krea2_Raw_convrot_int8mixed.safetensors"],
        role="image",
    )

    assert inspection.components[0].kind == "diffusion_model"
    assert inspection.components[0].target_folder == "diffusion_models"


def test_static_inspector_uses_verified_component_folder_for_unknown_weights() -> None:
    inspection = inspect_repository_metadata(
        {
            "text_encoders/qwen3vl_4b_fp8_scaled.safetensors": _safetensors(
                [
                    "model.embed_tokens.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.visual.blocks.0.attn.qkv.weight",
                ]
            )
        },
        ["text_encoders/qwen3vl_4b_fp8_scaled.safetensors"],
        role="image",
    )

    assert inspection.components[0].kind == "text_encoder"
    assert inspection.components[0].target_folder == "text_encoders"


def test_static_inspector_leaves_ambiguous_tensor_layout_unknown() -> None:
    inspection = inspect_repository_metadata(
        {"weights.safetensors": _safetensors(["blocks.0.attn.qkv.weight"])},
        ["weights.safetensors"],
        role="image",
    )

    assert inspection.components[0].kind == "unknown_safetensors"
    assert inspection.components[0].target_folder == "checkpoints"


def test_official_workflow_plan_accepts_a_required_lora_with_primary_weights() -> None:
    selected = ["model.safetensors", "lightning.safetensors"]
    inspection = inspect_repository_metadata(
        {
            "model.safetensors": _safetensors(["model.diffusion_model.input_blocks.0.weight"]),
            "lightning.safetensors": _safetensors(
                ["lora_unet_block.lora_down.weight"],
                {"ss_network_module": "networks.lora"},
            ),
        },
        selected,
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/complete-edit-workflow",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {"filename": selected[0], "size": 2_048, "sha256": "b" * 64},
            {
                "filename": selected[1],
                "size": 1_024,
                "sha256": "c" * 64,
                "source_remote_id": "synthetic/lightning",
                "source_revision": "d" * 40,
                "source_filename": selected[1],
            },
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="e" * 64,
    )

    assert plan.compatibility == "supported"
    assert {artifact.kind for artifact in plan.artifacts} == {"diffusion_model", "lora"}


def test_official_workflow_plan_uses_unambiguous_declared_component_paths() -> None:
    selected = [
        "split/diffusion/model.safetensors",
        "encoders/text.safetensors",
        "vae/model.safetensors",
        "lightning.safetensors",
    ]
    inspection = ModelManifestInspection(
        architecture=None,
        family=None,
        components=tuple(
            InspectedComponent(
                path=path,
                kind="unknown_safetensors",
                target_folder="checkpoints",
            )
            for path in selected
        ),
        metadata_files=(),
    )
    plan = resolve_install_plan(
        remote_id="synthetic/complete-edit-workflow",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {"filename": path, "size": 1_024, "sha256": str(index) * 64}
            for index, path in enumerate(selected, start=1)
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="e" * 64,
        comfy_paths={
            "diffusion_models": "split/diffusion",
            "text_encoders": "encoders",
            "vae": "vae",
            "loras": ".",
        },
    )

    assert plan.compatibility == "supported"
    assert [(item.kind, item.target_folder) for item in plan.artifacts] == [
        ("diffusion_model", "diffusion_models"),
        ("text_encoder", "text_encoders"),
        ("vae", "vae"),
        ("lora", "loras"),
    ]


def test_chat_install_plan_binds_external_projector_provenance() -> None:
    model_path = "model-Q4_K_M.gguf"
    projector_path = "companions/author/model/mmproj-model-f16.gguf"
    inspection = inspect_repository_metadata(
        {
            model_path: _gguf({"general.architecture": "vision"}),
            projector_path: _gguf(
                {
                    "general.architecture": "clip",
                    "clip.projector_type": "mlp",
                }
            ),
        },
        [model_path, projector_path],
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="converter/model-gguf",
        revision="a" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": model_path,
                "size": 10,
                "sha256": "b" * 64,
            },
            {
                "filename": projector_path,
                "size": 3,
                "sha256": "c" * 64,
                "source_remote_id": "author/model",
                "source_revision": "d" * 40,
                "source_filename": "mmproj-model-f16.gguf",
            },
        ],
        inspection=inspection,
    )

    projector = next(item for item in plan.artifacts if item.kind == "projector")
    assert projector.source_remote_id == "author/model"
    assert projector.source_revision == "d" * 40
    assert projector.source_path == "mmproj-model-f16.gguf"
    assert plan.compatibility == "supported"


def test_static_inspector_accepts_lora_only_as_a_typed_auxiliary_plan() -> None:
    inspection = inspect_repository_metadata(
        {
            "adapter.safetensors": _safetensors(
                ["lora_unet_block.lora_down.weight"],
                {
                    "ss_network_module": "networks.lora",
                    "ss_network_dim": "16",
                    "modelspec.trigger_phrase": "atelier ink, paper grain",
                },
            )
        },
        ["adapter.safetensors"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/unknown-adapter",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "adapter.safetensors",
                "size": 1_024,
                "sha256": "b" * 64,
            }
        ],
        inspection=inspection,
        comfy_paths={"loras": "."},
        auxiliary_kind="lora",
    )

    assert plan.compatibility == "supported"
    assert plan.runtime_contract["auxiliary_kind"] == "lora"
    assert plan.artifacts[0].target_folder == "loras"
    assert inspection.components[0].metadata["network_type"] == "networks.lora"
    assert inspection.components[0].metadata["rank"] == 16
    assert inspection.components[0].metadata["trigger_words"] == [
        "atelier ink",
        "paper grain",
    ]


def test_workflow_owned_encoder_stays_one_inert_exact_component() -> None:
    inspection = ModelManifestInspection(
        architecture=None,
        family="qwen",
        components=(
            InspectedComponent(
                path="text_encoders/qwen.safetensors",
                kind="text_encoder",
                target_folder="text_encoders",
            ),
        ),
        metadata_files=(),
    )

    plan = resolve_install_plan(
        remote_id="synthetic/workflow-components",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "text_encoders/qwen.safetensors",
                "size": 4_096,
                "sha256": "b" * 64,
            }
        ],
        inspection=inspection,
        workflow_reference_kind="checkpoint",
    )

    assert plan.compatibility == "supported"
    assert [(item.path, item.kind, item.target_folder) for item in plan.artifacts] == [
        ("text_encoders/qwen.safetensors", "text_encoder", "text_encoders")
    ]
    assert plan.runtime_contract["workflow_asset_kind"] == "text_encoder"
    assert plan.runtime_contract["workflow_reference_kind"] == "checkpoint"
    assert plan.runtime_contract["comfy_paths"] == {"text_encoders": "."}
    assert plan.runtime_contract["workflow_component_folders"] == {
        "text_encoders/qwen.safetensors": "text_encoders"
    }
    assert plan.activation_probe == {
        "version": "activation-probe-v2",
        "kind": "workflow_asset",
        "timeout_seconds": 300,
        "required": False,
    }


def test_workflow_owned_lora_does_not_become_a_standalone_auxiliary() -> None:
    inspection = ModelManifestInspection(
        architecture=None,
        family=None,
        components=(
            InspectedComponent(
                path="detail.safetensors",
                kind="lora",
                target_folder="loras",
            ),
        ),
        metadata_files=(),
    )

    plan = resolve_install_plan(
        remote_id="synthetic/workflow-lora",
        revision="c" * 40,
        role="image",
        engine="comfyui",
        selected_files=[{"filename": "detail.safetensors", "size": 1_024, "sha256": "d" * 64}],
        inspection=inspection,
        workflow_reference_kind="lora",
    )

    assert plan.compatibility == "supported"
    assert plan.runtime_contract["auxiliary_kind"] is None
    assert plan.runtime_contract["workflow_asset_kind"] == "lora"
    assert plan.activation_probe["required"] is False


@pytest.mark.parametrize(
    ("reference_kind", "size", "digest", "failure_code"),
    [
        ("lora", 1_024, "e" * 64, "workflow_asset_kind_mismatch"),
        ("checkpoint", 0, "e" * 64, "unverified_workflow_asset"),
        ("checkpoint", 1_024, "E" * 64, "unverified_workflow_asset"),
    ],
)
def test_workflow_owned_assets_fail_closed_on_kind_or_evidence(
    reference_kind: str,
    size: int,
    digest: str,
    failure_code: str,
) -> None:
    inspection = ModelManifestInspection(
        architecture=None,
        family=None,
        components=(
            InspectedComponent(
                path="encoder.safetensors",
                kind="text_encoder",
                target_folder="text_encoders",
            ),
        ),
        metadata_files=(),
    )

    plan = resolve_install_plan(
        remote_id="synthetic/workflow-components",
        revision="f" * 40,
        role="image",
        engine="comfyui",
        selected_files=[{"filename": "encoder.safetensors", "size": size, "sha256": digest}],
        inspection=inspection,
        workflow_reference_kind=reference_kind,
    )

    assert plan.compatibility == "unsupported"
    assert plan.failure_code == failure_code


def test_standalone_encoder_without_a_template_remains_unsupported() -> None:
    inspection = ModelManifestInspection(
        architecture=None,
        family=None,
        components=(
            InspectedComponent(
                path="encoder.safetensors",
                kind="text_encoder",
                target_folder="text_encoders",
            ),
        ),
        metadata_files=(),
    )

    plan = resolve_install_plan(
        remote_id="synthetic/workflow-components",
        revision="1" * 40,
        role="image",
        engine="comfyui",
        selected_files=[{"filename": "encoder.safetensors", "size": 1_024, "sha256": "2" * 64}],
        inspection=inspection,
    )

    assert plan.compatibility == "trusted_extension_required"
    assert plan.failure_code == "workflow_contract_missing"


def test_media_plan_rejects_pickle_compatible_weight_formats() -> None:
    inspection = inspect_repository_metadata(
        {},
        ["pytorch_model.bin"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/unsafe-media",
        revision="2" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "pytorch_model.bin",
                "size": 1_024,
                "sha256": "3" * 64,
            }
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="4" * 64,
    )

    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "unsafe_model_format"


def test_static_inspector_reads_gguf_architecture_without_filename_guessing() -> None:
    inspection = inspect_repository_metadata(
        {
            "weights.bin.gguf": _gguf(
                {
                    "general.architecture": "qwen3",
                    "general.type": "model",
                }
            )
        },
        ["weights.bin.gguf"],
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/future-chat-model",
        revision="d" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "weights.bin.gguf",
                "size": 2_048,
                "sha256": "e" * 64,
            }
        ],
        inspection=inspection,
    )

    assert inspection.architecture == "qwen3"
    assert inspection.family == "qwen"
    assert plan.compatibility == "supported"
    assert plan.activation_probe["kind"] == "chat_completion"


def test_static_inspector_skips_large_gguf_token_arrays_without_materializing_them() -> None:
    inspection = inspect_repository_metadata(
        {
            "qwen.gguf": _gguf_with_large_string_array(
                item_count=100_001,
                architecture="qwen35",
            )
        },
        ["qwen.gguf"],
        role="chat",
    )

    assert inspection.architecture == "qwen35"
    assert inspection.family == "qwen"


def test_modelopt_snapshot_produces_a_vllm_install_contract() -> None:
    selected = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "config.json",
        "hf_quant_config.json",
        "tokenizer.json",
    ]
    inspection = inspect_repository_metadata(
        {
            "config.json": json.dumps(
                {"architectures": ["Qwen3_5ForConditionalGeneration"]}
            ).encode(),
            "hf_quant_config.json": json.dumps({"quantization": {"quant_algo": "NVFP4"}}).encode(),
            "model-00001-of-00002.safetensors": _safetensors(
                ["model.layers.0.self_attn.q_proj.weight"]
            ),
            "model-00002-of-00002.safetensors": _safetensors(
                ["model.layers.1.self_attn.q_proj.weight"]
            ),
        },
        selected,
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="nvidia/Qwen3.6-27B-NVFP4",
        revision="a" * 40,
        role="chat",
        engine="vllm",
        selected_files=[
            {
                "filename": filename,
                "size": 1_024,
                "sha256": "b" * 64,
            }
            for filename in selected
        ],
        inspection=inspection,
    )

    assert plan.compatibility == "supported"
    assert plan.runtime_contract["engine"] == "vllm"
    assert plan.runtime_contract["quantization"] == "modelopt"
    assert plan.runtime_contract["model_layout"] == "transformers_snapshot"
    assert {artifact.kind for artifact in plan.artifacts} == {"weights", "metadata"}


def test_modelopt_snapshot_requires_quantization_metadata() -> None:
    inspection = inspect_repository_metadata(
        {
            "config.json": json.dumps({"model_type": "qwen3_5"}).encode(),
            "model.safetensors": _safetensors(["model.embed_tokens.weight"]),
        },
        ["model.safetensors", "config.json"],
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/incomplete-modelopt",
        revision="a" * 40,
        role="chat",
        engine="vllm",
        selected_files=[
            {"filename": "model.safetensors", "size": 1_024, "sha256": "b" * 64},
            {"filename": "config.json", "size": 128, "sha256": "c" * 64},
        ],
        inspection=inspection,
    )

    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "incomplete_modelopt_snapshot"


def test_static_inspector_rejects_oversized_and_unsafe_metadata() -> None:
    with pytest.raises(ModelManifestError, match="size limit"):
        inspect_repository_metadata(
            {"config.json": b"{" + b" " * MAX_METADATA_BYTES + b"}"},
            [],
            role="image",
        )
    with pytest.raises(ModelManifestError, match="unsafe"):
        inspect_repository_metadata(
            {"../config.json": b"{}"},
            [],
            role="image",
        )


async def test_install_plan_hash_is_stable_and_persistence_is_idempotent(client) -> None:  # type: ignore[no-untyped-def]
    inspection = inspect_repository_metadata(
        {"model.gguf": _gguf({"general.architecture": "llama"})},
        ["model.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/unknown-llama",
        revision="f" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "model.gguf",
                "size": 4_096,
                "sha256": "1" * 64,
            }
        ],
        inspection=inspection,
    )

    with SessionLocal() as session:
        first = persist_install_plan(session, resolved)
        session.commit()
        first_id = first.id
    with SessionLocal() as session:
        second = persist_install_plan(session, resolved)
        session.commit()
        assert second.id == first_id
        assert second.plan_hash == resolved.plan_hash


async def test_a_stored_plan_stops_quoting_a_reason_that_no_longer_applies(client) -> None:  # type: ignore[no-untyped-def]
    """A refusal must not outlive the attempt that produced it.

    Plans are reused by hash, and the failure fields are not part of that hash,
    so two resolves of the same install can disagree about why it failed. The
    stored row was never rewritten while it sat in "planned", so the first
    reason recorded was the reason reported forever - including to a caller
    whose own request had nothing to do with it.

    This is how a malformed request came back, unchanged, to well-formed ones
    made afterwards: same install, same hash, somebody else's error message.
    """
    inspection = inspect_repository_metadata(
        {"model.gguf": _gguf({"general.architecture": "llama"})},
        ["model.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/repairable",
        revision="c" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[{"filename": "model.gguf", "size": 4_096, "sha256": "2" * 64}],
        inspection=inspection,
    )
    # Only the failure fields differ, so both resolutions share a plan hash and
    # therefore the same stored row - which is the whole point.
    first = replace(
        resolved, failure_code="preflight_blocked", failure_reason="Choose one or the other."
    )
    second = replace(resolved, failure_code="disk_full", failure_reason="Not enough room for this.")
    assert first.plan_hash == second.plan_hash

    with SessionLocal() as session:
        persist_install_plan(session, first)
        session.commit()

    with SessionLocal() as session:
        stored = persist_install_plan(session, second)
        session.commit()

        assert stored.failure_code == "disk_full"
        assert stored.failure_reason == "Not enough room for this."

    with SessionLocal() as session:
        cleared = persist_install_plan(session, resolved)
        session.commit()

        assert cleared.failure_code is None
        assert cleared.failure_reason is None


async def test_a_plan_being_downloaded_is_not_rewritten_underneath_the_transfer(client) -> None:  # type: ignore[no-untyped-def]
    """The transfer reads these fields while it runs."""
    inspection = inspect_repository_metadata(
        {"model.gguf": _gguf({"general.architecture": "llama"})},
        ["model.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/in-flight",
        revision="d" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[{"filename": "model.gguf", "size": 4_096, "sha256": "3" * 64}],
        inspection=inspection,
    )

    with SessionLocal() as session:
        stored = persist_install_plan(session, resolved)
        stored.status = "downloading"
        session.commit()

    with SessionLocal() as session:
        again = persist_install_plan(
            session, replace(resolved, failure_code="late", failure_reason="arrived mid-transfer")
        )
        session.commit()

        assert again.status == "downloading"
        assert again.failure_code is None


async def test_supported_plan_can_be_retried_after_a_terminal_attempt(client) -> None:  # type: ignore[no-untyped-def]
    inspection = inspect_repository_metadata(
        {"model.gguf": _gguf({"general.architecture": "future"})},
        ["model.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/retryable",
        revision="9" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "model.gguf",
                "size": 4_096,
                "sha256": "8" * 64,
            }
        ],
        inspection=inspection,
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, resolved)
        plan.status = "failed"
        plan.failure_code = "activation_runtime_failed"
        plan.failure_reason = "Synthetic failure"
        session.commit()
        plan_id = plan.id

    with SessionLocal() as session:
        retry = persist_install_plan(session, resolved)
        session.commit()
        assert retry.id == plan_id
        assert retry.status == "planned"
        assert retry.failure_code is None
        assert retry.failure_reason is None
