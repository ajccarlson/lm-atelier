from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_serializer

from .domain import Operation, RoutingMode
from .references import MAX_REFERENCES_PER_TURN, MAX_ROLE, MentionSource
from .worker_failures import WorkerFailureCode


class ApiModel(BaseModel):
    # extra="forbid": a client typo in a request field must be a 422, not a
    # silently applied default. Response construction is unaffected - servers
    # build these from exact attributes.
    model_config = ConfigDict(from_attributes=True, extra="forbid")


ContentRating = Literal["general", "mature", "unknown"]

GenerationSettingsByRole = dict[
    Literal["chat", "image", "video"],
    dict[str, Any],
]
GenerationPresetIdsByRole = dict[
    Literal["chat", "image", "video"],
    str | None,
]


class VisionSettings(ApiModel):
    max_images: int = Field(default=4, ge=1, le=16)
    max_video_frames: int = Field(default=6, ge=3, le=16)
    include_prior_visual: bool = True
    verify_image_edits: bool = False
    compile_visual_prompts: bool = True


def new_chat_vision_settings() -> VisionSettings:
    """Enable edit review for new chats without changing legacy settings defaults."""
    return VisionSettings(verify_image_edits=True)


class WebSettings(ApiModel):
    """Whether this conversation may reach the internet.

    Off unless someone turned it on for this chat specifically. A new chat
    never inherits it, so permission cannot spread by being nearby.
    """

    allow_url_fetch: bool = False


class ProjectCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10_000)
    instructions: str = Field(default="", max_length=100_000)
    image_workflow_revision_id: str | None = None
    video_workflow_revision_id: str | None = None
    generation_settings_json: GenerationSettingsByRole = Field(default_factory=dict)
    generation_preset_ids_json: GenerationPresetIdsByRole = Field(default_factory=dict)


class ProjectUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=10_000)
    instructions: str | None = Field(default=None, max_length=100_000)
    archived: bool | None = None
    pinned: bool | None = None
    image_workflow_revision_id: str | None = None
    video_workflow_revision_id: str | None = None
    generation_settings_json: GenerationSettingsByRole | None = None
    generation_preset_ids_json: GenerationPresetIdsByRole | None = None


class ProjectOut(ApiModel):
    id: str
    name: str
    description: str
    instructions: str
    archived: bool
    pinned: bool
    image_workflow_revision_id: str | None
    video_workflow_revision_id: str | None
    generation_settings_json: GenerationSettingsByRole
    generation_preset_ids_json: GenerationPresetIdsByRole
    created_at: datetime
    updated_at: datetime


class ChatCreate(ApiModel):
    title: str = Field(default="New chat", min_length=1, max_length=240)
    project_id: str | None = None
    routing_mode: RoutingMode = RoutingMode.AUTO
    generation_settings_json: GenerationSettingsByRole = Field(default_factory=dict)
    generation_preset_ids_json: GenerationPresetIdsByRole = Field(default_factory=dict)
    vision_settings_json: VisionSettings = Field(default_factory=new_chat_vision_settings)


class ChatUpdate(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    project_id: str | None = None
    archived: bool | None = None
    pinned: bool | None = None
    routing_mode: RoutingMode | None = None
    confirm_uncertain_media: bool | None = None
    active_chat_profile_id: str | None = None
    active_vision_profile_id: str | None = None
    active_image_profile_id: str | None = None
    active_video_profile_id: str | None = None
    generation_settings_json: GenerationSettingsByRole | None = None
    generation_preset_ids_json: GenerationPresetIdsByRole | None = None
    vision_settings_json: VisionSettings | None = None
    web_settings_json: WebSettings | None = None


class GenerationIdentityOut(ApiModel):
    model_profile_name: str | None = None
    workflow_family_name: str | None = None
    workflow_definition_name: str | None = None
    workflow_version: int | None = None


class ArtifactOut(ApiModel):
    id: str
    sha256: str
    kind: str
    media_type: str
    size_bytes: int
    original_name: str | None
    metadata_json: dict[str, Any]
    favorite: bool = False
    created_at: datetime
    url: str | None = None
    generation_identity: GenerationIdentityOut | None = None


class ArtifactUpdate(ApiModel):
    favorite: bool


class ArtifactLibraryItem(ArtifactOut):
    reference_count: int = 0
    chat_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)


class ArtifactLibraryEntrySummary(ApiModel):
    id: str = Field(min_length=1, max_length=80)
    artifact_id: str = Field(min_length=1, max_length=80)
    version: int = Field(ge=1)
    state: Literal["visible", "trashed"]
    display_name: str = Field(min_length=1, max_length=500)
    favorite: bool
    kind: Literal["image", "video"]
    media_type: str = Field(min_length=1, max_length=120)
    size_bytes: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ArtifactLibraryPage(ApiModel):
    items: list[ArtifactLibraryEntrySummary]
    next_cursor: str | None = Field(default=None, min_length=1, max_length=2_048)


class ArtifactStorageInfo(ApiModel):
    total_bytes: int
    total_count: int
    referenced_bytes: int
    referenced_count: int
    unreferenced_bytes: int
    unreferenced_count: int
    temporary_bytes: int
    temporary_count: int
    eligible_bytes: int
    eligible_count: int
    retention_pending_count: int
    disk_free_bytes: int
    warning: bool
    retention_days: int
    temporary_retention_hours: int


class ArtifactCleanupRequest(ApiModel):
    dry_run: bool = True


class ArtifactCleanupResult(ApiModel):
    dry_run: bool
    marked_count: int
    retention_pending_count: int
    removed_count: int
    reclaimed_bytes: int


class ArtifactDeleteResult(ApiModel):
    artifact_id: str
    reference_count: int
    removed_count: int
    reclaimed_bytes: int


class MessagePartOut(ApiModel):
    id: str
    position: int
    type: str
    text: str | None
    artifact_id: str | None
    metadata_json: dict[str, Any]
    artifact: ArtifactOut | None = None


class ResponseRevisionOut(ApiModel):
    id: str
    message_id: str
    run_id: str | None
    sequence: int
    status: str
    parts: list[MessagePartOut]
    feedback: Literal["up", "down"] | None = None
    created_at: datetime
    updated_at: datetime


class MessageReferenceOut(ApiModel):
    """What one turn referred to, as it stood when the turn was accepted.

    The name and mention are the recorded ones, not the subject's current
    values, and the subject id carries no promise that the subject still
    exists. That is the point: a renamed subject must not rewrite an old
    message, and a deleted one must not erase the record that it was used.
    """

    reference_subject_id: str
    mention_slug: str
    subject_name: str
    subject_kind: str
    role: str | None = None
    strength: float | None = None
    source: str
    # The _json suffix matches the columns and the shape every other
    # reference field already takes in this API.
    reference_asset_ids_json: list[str] = Field(default_factory=list)
    artifact_ids_json: list[str] = Field(default_factory=list)


class MessageOut(ApiModel):
    id: str
    chat_id: str
    parent_id: str | None
    role: str
    status: str
    transcript_visible: bool
    active_response_revision_id: str | None
    parts: list[MessagePartOut]
    # Empty for every message that named nothing, which is almost all of them.
    references: list[MessageReferenceOut] = Field(default_factory=list)
    response_revisions: list[ResponseRevisionOut] = Field(default_factory=list)
    feedback: Literal["up", "down"] | None = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at", when_used="json")
    def serialize_timestamp_as_utc(self, value: datetime) -> str:
        """Keep SQLite-naive UTC instants explicit at the browser boundary."""
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")


class ResponseFeedbackUpdate(ApiModel):
    """Set or clear one verdict; null rating clears. A click stores a local
    preference for evaluation and reranking - it never trains weights."""

    rating: Literal["up", "down"] | None
    response_revision_id: str | None = None


class ResponseFeedbackOut(ApiModel):
    message_id: str
    response_revision_id: str | None
    rating: Literal["up", "down"] | None


class ChatOut(ApiModel):
    id: str
    project_id: str | None
    title: str
    archived: bool
    pinned: bool
    routing_mode: str
    confirm_uncertain_media: bool
    active_chat_profile_id: str | None
    active_vision_profile_id: str | None
    active_image_profile_id: str | None
    active_video_profile_id: str | None
    active_head_message_id: str | None
    generation_settings_json: GenerationSettingsByRole
    generation_preset_ids_json: GenerationPresetIdsByRole
    vision_settings_json: VisionSettings
    web_settings_json: WebSettings = Field(default_factory=WebSettings)
    # Empty for a chat created directly; carries the source chat and message
    # when this thread was forked from one.
    origin_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatOut):
    messages: list[MessageOut]


class ExchangeDeletionOut(ApiModel):
    chat_id: str
    user_message_id: str
    message_ids: list[str]
    run_ids: list[str]
    job_ids: list[str]
    work_plan_ids: list[str]
    released_artifact_ids: list[str]
    retained_artifact_ids: list[str]
    new_head_message_id: str | None = None


class StudioSessionCreate(ApiModel):
    """Open the studio over one image; the session is found or created."""

    source_artifact_id: str = Field(min_length=1, max_length=80)
    # When the studio is entered from a chat, its profile and settings
    # snapshot carry over so applies run with the same models.
    source_chat_id: str | None = Field(default=None, max_length=40)


class PromptHelperCreate(ApiModel):
    source_chat_id: str = Field(min_length=1, max_length=40)
    draft_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
    ]


class PromptHelperUpdate(ApiModel):
    draft_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
    ]


class PromptHelperDetail(ChatDetail):
    draft_prompt: str


class TurnReferenceIn(ApiModel):
    """One explicitly structured Reference attached to a turn."""

    reference_subject_id: str = Field(min_length=1, max_length=80)
    role: str | None = Field(default=None, min_length=1, max_length=MAX_ROLE)
    selected_asset_ids: list[str] = Field(default_factory=list, max_length=16)
    strength: float | None = Field(default=None, ge=0.0, le=2.0)
    source: MentionSource = MentionSource.MENTION


class TurnRequest(ApiModel):
    text: str = Field(min_length=1, max_length=200_000)
    mode: RoutingMode | None = None
    parent_message_id: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list, max_length=16)
    references: list[TurnReferenceIn] = Field(
        default_factory=list, max_length=MAX_REFERENCES_PER_TURN
    )
    settings: dict[str, Any] = Field(default_factory=dict)
    ordered_settings: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=3)
    output_count: int | None = Field(default=None, ge=1, le=16)
    # The workflow a recipe recorded. A recipe that stored which workflow made
    # a result and then ran against whichever one happens to be current is not
    # a recipe; it is the instruction with extra fields. An id that does not
    # match this operation, engine, or install is not honored - the turn
    # refuses rather than quietly substituting.
    workflow_revision_id: str | None = Field(default=None, max_length=40)
    confirm_media: bool = False
    idempotency_key: str | None = Field(default=None, max_length=200)


class DraftClassificationRequest(ApiModel):
    """An unsent composer draft, classified with the router the turn will use."""

    text: str = Field(default="", max_length=200_000)
    mode: RoutingMode | None = None
    parent_message_id: str | None = None


class DraftClassification(ApiModel):
    references_prior_visual: bool


class VerifiedSetup(ApiModel):
    """A working setup, described so another machine can resolve it."""

    version: int
    role: str
    engine: str
    model: dict[str, Any]
    workflow: dict[str, Any] | None
    settings: dict[str, Any]
    hardware: dict[str, Any] | None
    attestation: dict[str, Any]
    digest: str


class ResolvedSetupComponent(ApiModel):
    target_folder: str
    sha256: str
    present: bool


class ResolvedSetup(ApiModel):
    """What an imported setup finds on this machine, and what it still needs."""

    version: int
    digest: str | None
    components: list[ResolvedSetupComponent]
    missing_components: list[dict[str, str]]
    hardware_compatible: bool
    # Provenance from the artifact, kept separate from anything earned here.
    verified_elsewhere: bool
    verified_here: bool
    requires_approval: bool
    ready_to_verify: bool


class TrustDerivation(ApiModel):
    """Whether this machine could vouch for a workflow by rebuilding it."""

    version: int
    trusted: bool
    reason: str
    message: str


class RegenerateRequest(ApiModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class RoutingReasonCode(StrEnum):
    EXPLICIT_TEXT_MODE = "explicit_text_mode"
    EXPLICIT_IMAGE_MODE = "explicit_image_mode"
    EXPLICIT_VIDEO_MODE = "explicit_video_mode"
    REPEAT_LAST_GENERATION = "repeat_last_generation"
    ASSISTANT_SUGGESTION_SELECTED = "assistant_suggestion_selected"
    DISCUSSION = "discussion"
    TEXT_EDIT = "text_edit"
    TEXT_MEDIA_TASK = "text_media_task"
    VIDEO_CREATION = "video_creation"
    PRIOR_IMAGE_EDIT = "prior_image_edit"
    IMAGE_CREATION = "image_creation"
    TEXT_TASK = "text_task"
    DEFAULT_TEXT = "default_text"
    MODEL_PLANNER = "model_planner"
    GENERATION_OFFER_ACCEPTED = "generation_offer_accepted"


class RoutingPlan(ApiModel):
    operation: Operation
    standalone_prompt: str
    # The chat passage this request is asking to depict, when it is asking for
    # one. Carried apart from `standalone_prompt` so a media prompt can be
    # compiled from the request and its source rather than their concatenation.
    text_context: str | None = None
    negative_prompt: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    profile_id: str | None = None
    workflow_id: str | None = None
    # Cost projections computed at admission. These were previously stuffed
    # into a `parameter_overrides` dict under underscore-prefixed keys and
    # read by the browser through that internal marker; they are contract, so
    # they are named fields like the sibling ordered-plan 409 already used.
    generation_estimate: dict[str, Any] | None = None
    media_plan_estimate: dict[str, Any] | None = None
    output_count: int = Field(default=1, ge=1, le=16)
    confidence: float = Field(ge=0, le=1)
    reason_code: RoutingReasonCode
    reason: str


class GenerationOfferItem(ApiModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    mode: Literal["image", "video"]
    prompt: str = Field(min_length=1, max_length=20_000)


class GenerationOffer(ApiModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    message: str = Field(min_length=1, max_length=1_000)
    items: list[GenerationOfferItem] = Field(min_length=1, max_length=8)


class OrderedStepInput(ApiModel):
    source_step_id: str = Field(
        min_length=1,
        max_length=40,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    kind: Literal["text_context", "artifact"]


class OrderedStepIntent(ApiModel):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")
    mode: Literal["text", "image", "video"]
    prompt: str = Field(min_length=1, max_length=20_000)
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    inputs: list[OrderedStepInput] = Field(default_factory=list, max_length=8)


class OrderedWorkIntent(ApiModel):
    planner_version: Literal["ordered-work-v1"] = "ordered-work-v1"
    steps: list[OrderedStepIntent] = Field(min_length=2, max_length=8)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1_000)
    requires_confirmation: bool = False


class RunOut(ApiModel):
    id: str
    idempotency_key: str | None
    chat_id: str
    user_message_id: str
    assistant_message_id: str
    work_plan_id: str | None
    work_step_id: str | None
    operation: str
    status: str
    standalone_prompt: str
    profile_id: str | None
    vision_profile_id: str | None
    workflow_revision_id: str | None
    settings_json: dict[str, Any]
    provenance_json: dict[str, Any]
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: int | None
    created_at: datetime
    updated_at: datetime


class TurnAccepted(ApiModel):
    run: RunOut
    user_message: MessageOut
    assistant_message: MessageOut


class ProgressStageTiming(ApiModel):
    stage: str
    duration_ms: int = Field(ge=0)


class ProgressV2(ApiModel):
    version: Literal[2] = 2
    stage: str
    stage_progress: float | None = Field(default=None, ge=0, le=1)
    overall_progress: float | None = Field(default=None, ge=0, le=1)
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    unit: str | None = None
    bytes_reused: int = Field(default=0, ge=0)
    rate_bytes_per_second: float | None = Field(default=None, ge=0)
    eta_seconds: int | None = Field(default=None, ge=0)
    file_index: int | None = Field(default=None, ge=1)
    file_count: int | None = Field(default=None, ge=1)
    queue_resource: str | None = None
    queue_position: int | None = Field(default=None, ge=0)
    queue_length: int | None = Field(default=None, ge=0)
    blocked_by: list[str] = Field(default_factory=list)
    indeterminate: bool = False
    stage_started_at: datetime | None = None
    stage_elapsed_ms: int = Field(default=0, ge=0)
    completed_stages: list[ProgressStageTiming] = Field(default_factory=list)
    updated_at: datetime


class JobOut(ApiModel):
    id: str
    kind: str
    status: str
    run_id: str | None
    work_plan_id: str | None
    work_step_id: str | None
    progress: float
    phase: str
    progress_json: dict[str, Any]
    queue_resource: str | None
    queue_group: str | None
    queue_priority: int
    queue_ticket: str | None
    enqueued_at: datetime | None
    claim_expires_at: datetime | None
    heartbeat_at: datetime | None
    payload_json: dict[str, Any]
    result_json: dict[str, Any]
    error: str | None
    attempt: int
    cancellable: bool
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class WorkStepOut(ApiModel):
    id: str
    plan_id: str
    run_id: str | None
    ordinal: int
    display_group: str | None
    operation: str
    status: str
    prompt: str
    profile_id: str | None
    workflow_revision_id: str | None
    settings_json: dict[str, Any]
    input_bindings_json: list[dict[str, Any]]
    output_contract_json: list[dict[str, Any]]
    queue_class: str
    error: str | None
    created_at: datetime
    updated_at: datetime


class WorkPlanOut(ApiModel):
    id: str
    chat_id: str
    idempotency_key: str | None
    source_action: str
    persistence_scope: str
    status: str
    context_head_message_id: str | None
    transcript_sequence: int
    priority: int
    planner_version: str
    failure_policy: str
    summary_json: dict[str, Any]
    steps: list[WorkStepOut]
    created_at: datetime
    updated_at: datetime


class ModelSourceOut(ApiModel):
    id: str
    provider: str
    remote_id: str
    revision: str
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class InstallArtifact(ApiModel):
    path: str = Field(min_length=1, max_length=1_000)
    kind: str = Field(min_length=1, max_length=40)
    target_folder: str = Field(min_length=1, max_length=80)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required: bool = True
    reuse: Literal["download", "installed", "verified-cache"] = "download"


class InstallPlanOut(ApiModel):
    id: str
    provider: str
    remote_id: str
    revision: str
    role: str
    engine: str
    architecture: str | None
    family: str | None
    plan_hash: str
    resolver_version: str
    compatibility: str
    artifacts_json: list[dict[str, Any]]
    runtime_contract_json: dict[str, Any]
    activation_probe_json: dict[str, Any]
    status: str
    failure_code: str | None
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime


class ModelCapabilityEvidenceOut(ApiModel):
    id: str
    model_install_id: str
    evidence_key: str
    result: str
    component_hashes_json: dict[str, str]
    runtime_build: str
    adapter_contract_version: int
    launch_contract_version: str
    workflow_contract_version: str | None
    hardware_class: str
    probe_version: str
    failure_code: str | None
    failure_reason: str | None
    details_json: dict[str, Any]
    probed_at: datetime


class ModelInstallOut(ApiModel):
    id: str
    source_id: str | None
    name: str
    role: str
    engine: str
    local_path: str
    size_bytes: int
    compatibility: str
    manifest_json: dict[str, Any]
    active: bool
    readiness: Literal["ready", "unverified", "unsupported"] = "unverified"
    capability_evidence: ModelCapabilityEvidenceOut | None = None
    created_at: datetime
    updated_at: datetime


class ModelUpdateOut(ApiModel):
    """One installed asset's staleness verdict against its provider.

    `state` is "update_available", "current", or "unknown" - unknown means the
    provider could not answer, never a guess. Update fields are set only with
    "update_available"; installing the candidate goes through the normal
    verified catalog flow for its version id.
    """

    install_id: str
    name: str
    kind: str
    model_id: str
    installed_version_id: str
    installed_version_name: str | None
    state: Literal["update_available", "current", "unknown"]
    update_version_id: str | None = None
    update_version_name: str | None = None
    update_published_at: str | None = None
    update_base_model: str | None = None
    update_changelog: str | None = None


class ModelStorageInfo(ApiModel):
    installed_bytes: int
    partial_download_bytes: int
    catalog_cache_bytes: int
    installed_count: int
    partial_download_count: int


class StorageCleanupResult(ApiModel):
    removed_count: int
    reclaimed_bytes: int


class ModelImport(ApiModel):
    name: str = Field(min_length=1, max_length=300)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    local_path: str = Field(min_length=1, max_length=4_096)


class ModelProfileCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    use_case: str = Field(default="", max_length=1_000)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class ModelProfileUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    use_case: str | None = Field(default=None, max_length=1_000)
    load_settings: dict[str, Any] | None = None
    request_settings: dict[str, Any] | None = None
    is_default: bool | None = None


class ModelProfileClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class ModelProfileBundle(ApiModel):
    format: Literal["lm-atelier-profile"] = "lm-atelier-profile"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=200)
    use_case: str = Field(default="", max_length=1_000)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: dict[str, Any] = Field(default_factory=dict)


class ModelProfileOut(ApiModel):
    id: str
    model_install_id: str | None
    name: str
    use_case: str
    role: str
    engine: str
    load_settings_json: dict[str, Any]
    request_settings_json: dict[str, Any]
    is_default: bool
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    created_at: datetime
    updated_at: datetime


class PresetCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    settings: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class PresetUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    settings: dict[str, Any] | None = None
    is_default: bool | None = None


class PresetClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class PresetBundle(ApiModel):
    format: Literal["lm-atelier-preset"] = "lm-atelier-preset"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    settings: dict[str, Any] = Field(default_factory=dict)


class PresetOut(ApiModel):
    id: str
    name: str
    role: str
    settings_json: dict[str, Any]
    is_default: bool
    created_at: datetime
    updated_at: datetime


class WorkflowCreate(ApiModel):
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)
    engine: str = "comfyui"
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False


class WorkflowRevisionCreate(ApiModel):
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False


class WorkflowUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=10_000)


class WorkflowClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)


class WorkflowBundle(ApiModel):
    format: Literal["lm-atelier-workflow"] = "lm-atelier-workflow"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)
    engine: str = "comfyui"
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False
    source_revision: int | None = None


class StudioToolCapability(ApiModel):
    """Whether one studio tool can run here, and what would fix it."""

    kind: str
    workflow_class: str
    available: bool
    reason: str | None


class StudioCapabilityReport(ApiModel):
    tools: list[StudioToolCapability]


class WorkflowRevisionOut(ApiModel):
    id: str
    workflow_id: str
    version: int
    engine: str
    engine_version: str | None
    ui_graph_json: dict[str, Any]
    api_graph_json: dict[str, Any]
    input_schema_json: dict[str, Any]
    dependencies_json: dict[str, Any]
    trusted: bool
    created_at: datetime


class WorkflowOut(ApiModel):
    id: str
    name: str
    operation: str
    description: str
    current_revision_id: str | None
    revisions: list[WorkflowRevisionOut]
    created_at: datetime
    updated_at: datetime


WorkflowSelectorCapability = Literal["chat", "vision", "image", "video"]
WorkflowDependencyResourceKind = Literal[
    "model_profile",
    "model_install",
    "model_asset",
    "custom_node",
    "registry_package",
    "runtime",
]
WorkflowVariantReadiness = Literal[
    "ready",
    "setup_required",
    "review_required",
    "unavailable",
]
WorkflowSetupResolution = Literal[
    "reviewed_download_available",
    "attention_required",
]
WorkflowSelectionResponseMode = Literal[
    "default",
    "inherit",
    "automatic",
    "family",
    "revision",
    "legacy",
]


class WorkflowFamilyVariantOut(ApiModel):
    id: str
    variant_key: str
    name: str
    operation: Operation
    current_revision_id: str | None
    current_revision_version: int | None
    engine: str | None
    capabilities: list[str] = Field(default_factory=list)
    trusted: bool
    readiness: WorkflowVariantReadiness
    readiness_reason: str | None = None
    setup_resolution: WorkflowSetupResolution | None = None
    install_offer_id: str | None = Field(default=None, max_length=40)


class WorkflowFamilyPreferenceOut(ApiModel):
    selector_capability: WorkflowSelectorCapability
    enabled: bool
    is_default: bool
    sort_order: int


class WorkflowFamilyOut(ApiModel):
    id: str
    name: str
    description: str
    use_case: str
    tags: list[str] = Field(default_factory=list)
    enabled: bool
    archived: bool
    compatibility: bool
    variants: list[WorkflowFamilyVariantOut] = Field(default_factory=list)
    preferences: list[WorkflowFamilyPreferenceOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class WorkflowFamilyUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=10_000)
    use_case: str | None = Field(default=None, max_length=10_000)
    tags: list[str] | None = Field(default=None, max_length=100)
    enabled: bool | None = None
    archived: bool | None = None


class WorkflowFamilyPreferenceUpdate(ApiModel):
    enabled: bool = True
    is_default: bool = False
    sort_order: int = Field(default=0, ge=-1_000_000, le=1_000_000)


class WorkflowDependencyImpactOut(ApiModel):
    resource_kind: str
    resource_id: str
    resource_name: str
    binding_count: int
    revision_count: int
    current_revision: bool
    shared: bool
    other_workflow_count: int
    other_family_ids: list[str] = Field(default_factory=list)


class WorkflowFamilyRemovalImpactOut(ApiModel):
    family_id: str
    removal_strategy: Literal["archive"] = "archive"
    archive_blocked: bool
    revision_count: int
    current_revision_count: int
    chat_selection_count: int
    project_selection_count: int
    project_revision_pin_count: int
    active_run_count: int
    queued_step_count: int
    historical_run_count: int
    active_activation_count: int
    default_for: list[WorkflowSelectorCapability] = Field(default_factory=list)
    dependencies: list[WorkflowDependencyImpactOut] = Field(default_factory=list)


class WorkflowResourceConsumerOut(ApiModel):
    workflow_id: str
    workflow_name: str
    workflow_family_id: str | None = None
    workflow_family_name: str | None = None
    revision_ids: list[str] = Field(default_factory=list)
    binding_count: int
    current_revision: bool


class WorkflowResourceConsumersOut(ApiModel):
    resource_kind: WorkflowDependencyResourceKind
    resource_id: str
    resource_name: str
    consumers: list[WorkflowResourceConsumerOut] = Field(default_factory=list)


class ChatWorkflowDefaultSelectionIn(ApiModel):
    mode: Literal["default"]


class ChatWorkflowAutomaticSelectionIn(ApiModel):
    mode: Literal["automatic"]


class ChatWorkflowFamilySelectionIn(ApiModel):
    mode: Literal["family"]
    workflow_family_id: str = Field(min_length=1, max_length=64)


ChatWorkflowSelectionIn = Annotated[
    ChatWorkflowDefaultSelectionIn
    | ChatWorkflowAutomaticSelectionIn
    | ChatWorkflowFamilySelectionIn,
    Field(discriminator="mode"),
]


class ProjectWorkflowInheritSelectionIn(ApiModel):
    mode: Literal["inherit"]


class ProjectWorkflowAutomaticSelectionIn(ApiModel):
    mode: Literal["automatic"]


class ProjectWorkflowFamilySelectionIn(ApiModel):
    mode: Literal["family"]
    workflow_family_id: str = Field(min_length=1, max_length=64)


class ProjectWorkflowRevisionSelectionIn(ApiModel):
    mode: Literal["revision"]
    workflow_revision_id: str = Field(min_length=1, max_length=40)


ProjectWorkflowSelectionIn = Annotated[
    ProjectWorkflowInheritSelectionIn
    | ProjectWorkflowAutomaticSelectionIn
    | ProjectWorkflowFamilySelectionIn
    | ProjectWorkflowRevisionSelectionIn,
    Field(discriminator="mode"),
]


class WorkflowSelectionOut(ApiModel):
    selector_capability: WorkflowSelectorCapability
    mode: WorkflowSelectionResponseMode
    workflow_family_id: str | None = None
    workflow_revision_id: str | None = None
    legacy_profile_id: str | None = None


class WorkflowOpenTarget(ApiModel):
    url: str
    filename: str
    ui_graph: dict[str, Any]


class WorkflowEditorSessionOut(ApiModel):
    id: str
    protocol_version: int
    workflow_id: str
    base_revision_id: str
    base_graph_sha256: str
    base_prompt_sha256: str
    created_at: datetime
    expires_at: datetime
    ui_graph: dict[str, Any]
    nonce: str


class WorkflowEditorCancelIn(ApiModel):
    nonce: str = Field(min_length=1, max_length=200)


class WorkflowEditorConsumeIn(ApiModel):
    nonce: str = Field(min_length=1, max_length=200)
    base_revision_id: str = Field(min_length=1, max_length=40)
    ui_graph: dict[str, Any]
    api_prompt: dict[str, Any]


class WorkflowEditorGraphDeltaOut(ApiModel):
    node_count_delta: int
    link_count_delta: int
    added_node_types: list[str]
    removed_node_types: list[str]
    added_asset_filenames: list[str]
    removed_asset_filenames: list[str]


class WorkflowEditorReturnOut(ApiModel):
    validated_return_id: str
    session_id: str
    workflow_id: str
    base_revision_id: str
    current_revision_id: str
    base_graph_sha256: str
    returned_graph_sha256: str
    base_prompt_sha256: str
    returned_prompt_sha256: str
    changed: bool
    forked: bool
    delta: WorkflowEditorGraphDeltaOut
    expires_at: datetime


class WorkflowEditorDraftCreateIn(ApiModel):
    validated_return_id: str = Field(min_length=1, max_length=200)


class WorkflowEditorDraftOut(ApiModel):
    workflow_id: str
    base_revision_id: str
    draft_revision_id: str
    current_revision_id: str | None
    version: int
    created: bool
    forked: bool
    trusted: Literal[False]
    review_required: Literal[True]


class EditTemplateCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    instruction: str = Field(min_length=1, max_length=20_000)
    settings_json: dict[str, Any] = Field(default_factory=dict)
    # When given, the recipe is read from what this run actually did rather
    # than from whatever is current when Save is pressed.
    from_run_id: str | None = Field(default=None, max_length=40)


class EditTemplateOut(ApiModel):
    id: str
    name: str
    description: str
    instruction: str
    operation: str
    settings_json: dict[str, Any]
    workflow_revision_id: str | None
    model_profile_id: str | None
    mask_mode: str
    trigger_words_json: list[str]
    content_rating: ContentRating
    builtin: bool
    enabled: bool


class RegistryInstallReviewOut(ApiModel):
    """What staging found, so that trusting a package can be an informed act.

    Trust is what lets this code run. Asking someone to confirm they reviewed
    a package while showing them nothing to review makes the confirmation a
    formality, so these are the things that decide the answer: code that runs
    on install, code that runs at startup, compiled binaries, and the files
    that declare what else gets pulled in.
    """

    file_count: int
    expanded_bytes: int
    python_file_count: int
    install_scripts: list[str] = Field(default_factory=list, max_length=64)
    startup_hooks: list[str] = Field(default_factory=list, max_length=64)
    native_files: list[str] = Field(default_factory=list, max_length=64)
    dependency_manifests: list[str] = Field(default_factory=list, max_length=64)
    top_level_entries: list[str] = Field(default_factory=list, max_length=64)
    # Carried from the resolution: why this package needs looking at, in the
    # resolver's words rather than restated here.
    registry_warnings: list[str] = Field(default_factory=list, max_length=32)


class RegistryInstallOut(ApiModel):
    """One prepared package and the two explicit decisions it is waiting for."""

    id: str
    package_id: str
    package_version: str
    node_types: list[str]
    archive_sha256: str
    manifest_sha256: str
    wheel_closure_sha256: str | None
    wheel_environment_sha256: str | None
    disk_status: Literal[
        "ready",
        "node_files_missing",
        "wheel_environment_missing",
        "files_missing",
    ]
    node_files_present: bool
    wheel_environment_present: bool
    trusted: bool
    active: bool
    reviewed_at: str | None
    activated_at: str | None
    review: RegistryInstallReviewOut | None = None


class RegistryInstallReviewRequest(ApiModel):
    trusted: bool


class WorkflowAssetSelectionIn(ApiModel):
    """One explicit choice: this missing file comes from that plan artifact."""

    reference_filename: str = Field(min_length=1, max_length=1_000)
    install_plan_id: str = Field(min_length=1, max_length=40)
    artifact_path: str = Field(min_length=1, max_length=1_000)


class WorkflowAssetReviewRequest(ApiModel):
    """Review selections against a freshly re-analyzed graph.

    The browser sends the graph and its choices - never a report, digest,
    size, kind, or bound asset. Everything else is rebuilt server-side.
    """

    ui_graph: dict[str, Any]
    selections: list[WorkflowAssetSelectionIn] = Field(default_factory=list, max_length=64)


class WorkflowAssetQueueRequest(WorkflowAssetReviewRequest):
    """Queue exactly the reviewed binding, confirmed by its hash."""

    binding_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class BoundWorkflowAssetOut(ApiModel):
    reference_filename: str
    kind: str
    install_plan_id: str
    install_plan_hash: str
    provider: str
    remote_id: str
    revision: str
    artifact_path: str
    artifact_kind: str
    target_folder: str
    size_bytes: int
    sha256: str


class WorkflowAssetReviewOut(ApiModel):
    """What the browser may show: the binding, its hash, and the cost."""

    binding_plan_hash: str
    assets: list[BoundWorkflowAssetOut]
    download_count: int
    total_bytes: int


class WorkflowInstallOfferCreate(ApiModel):
    """Explicit plan choices for one persisted workflow revision."""

    selections: list[WorkflowAssetSelectionIn] = Field(
        min_length=1,
        max_length=64,
    )


class WorkflowInstallOfferOut(ApiModel):
    id: str
    workflow_revision_id: str
    workflow_artifact_sha256: str
    dependency_contract_sha256: str
    binding_plan_sha256: str
    offer_sha256: str
    assets: list[BoundWorkflowAssetOut]
    plan_count: int
    total_bytes: int
    status: Literal["ready", "queued", "invalidated", "completed", "expired"]
    queued_at: datetime | None
    completed_at: datetime | None
    invalidated_at: datetime | None
    invalidation_code: str | None
    invalidation_reason: str | None


class WorkflowPackageImportRequest(ApiModel):
    """Import a fully resolved ComfyUI package as an untrusted workflow.

    The user confirms the name and operation - the analyzer's guess prefills
    the form, but nothing is silently decided for them.
    """

    ui_graph: dict[str, Any]
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)
    # A package that needed preparation is first persisted as a deliberately
    # non-executable revision. Supplying both identities lets import finalize
    # that exact draft instead of creating a second, unrelated workflow.
    draft_workflow_id: str | None = Field(default=None, max_length=64)
    draft_revision_id: str | None = Field(default=None, max_length=40)


class WorkflowPackageDraftRequest(ApiModel):
    """Persist one exact source graph before resolving its dependencies."""

    ui_graph: dict[str, Any]
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)


class WorkflowPackagePrepareRequest(ApiModel):
    # Preparation binds to one exact identity; an unpinned request has nothing
    # to verify against and the resolver refuses it anyway.
    package_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    # The server re-analyzes the source graph and derives the exact node types.
    # A browser-provided node list would only be another unverified claim.
    ui_graph: dict[str, Any]
    # Required before an omitted source dependency can be recorded against the
    # install. Re-analyzing a submitted graph proves that graph is internally
    # consistent; it does not bind it to anything stored, and a proof about a
    # graph nobody saved is a proof about nothing.
    workflow_revision_id: str | None = Field(default=None, max_length=40)


class WorkflowPackageAnalyzeRequest(ApiModel):
    ui_graph: dict[str, Any]


class WorkflowPackageRequirementOut(ApiModel):
    package_id: str
    versions: list[str]
    node_types: list[str]
    locally_resolved: bool


class WorkflowSourceCandidateOut(ApiModel):
    """One source link the package author wrote down, already validated.

    A suggestion of what to preflight - never a download instruction. The
    normal immutable-plan path still resolves it, and the browser cannot
    substitute a URL of its own.
    """

    provider: str
    remote_id: str
    revision: str | None
    filename: str | None
    url: str


# What the workflow analyzer can say a referenced file is. One definition, so
# a caller naming an exact file cannot name a kind the analyzer never emits.
WorkflowAssetKind = Literal["checkpoint", "configuration", "embedding", "lora", "upscaler", "vae"]
AuxiliaryAssetKind = Literal[
    "lora",
    "vae",
    "controlnet",
    "upscaler",
    "embedding",
    "ip_adapter",
]


class WorkflowAssetReferenceOut(ApiModel):
    filename: str
    suffix: str
    policy: Literal["supported", "blocked", "unsupported"]
    kind: WorkflowAssetKind
    source_url: str | None
    present_locally: bool
    # Populated only when the author's own text names this exact file; a link
    # that names no file is reported once in the analysis instead of guessed
    # onto an asset here.
    source_candidates: list[WorkflowSourceCandidateOut] = []


class WorkflowPackageIssueOut(ApiModel):
    code: str
    count: int
    node_types: list[str]
    severity: Literal["blocking", "advisory"]


class WorkflowMissingNodeOut(ApiModel):
    node_type: str
    count: int
    package_id: str | None


class WorkflowPackageAnalysisOut(ApiModel):
    """The analyzer report, field names frozen with the analyzer.

    `ready` is the one trust/activation gate the browser obeys; it is computed
    by the analyzer, never re-derived client-side from list emptiness.
    `node_inventory_available` is this endpoint's own honesty flag: when the
    media runtime cannot enumerate its nodes, `missing_node_types` covers every
    runtime node and must be presented as "unknown", not as "missing".
    """

    format_version: str
    frontend_version: str | None
    node_count: int
    link_count: int
    subgraph_count: int
    operation_guess: Literal["image", "unknown", "video"]
    truncated: bool
    required_node_types: list[str]
    frontend_node_types: list[str]
    missing_node_types: list[str]
    missing_nodes: list[WorkflowMissingNodeOut]
    custom_packages: list[WorkflowPackageRequirementOut]
    asset_references: list[WorkflowAssetReferenceOut]
    issues: list[WorkflowPackageIssueOut]
    ready: bool
    runtime_nodes_available: bool
    dependencies_resolved: bool
    # Links the author recorded that name no particular file. Most authors
    # write display names rather than filenames, so this is the common case,
    # and it is offered for the user to assign rather than matched by guess.
    source_candidates: list[WorkflowSourceCandidateOut] = []
    node_inventory_available: bool


class CustomNodeInstallRequest(ApiModel):
    name: str = Field(min_length=1, max_length=240)
    source_url: str = Field(min_length=1, max_length=1000)
    revision: str = Field(min_length=40, max_length=40)


class CustomNodeUpdateRequest(ApiModel):
    revision: str = Field(min_length=40, max_length=40)


class CustomNodeTrustRequest(ApiModel):
    trusted: bool
    node_types: list[str] = Field(default_factory=list, max_length=4_096)


class CustomNodeOut(ApiModel):
    id: str
    name: str
    source_url: str
    revision: str
    previous_revision: str | None
    tree_hash: str
    trusted: bool
    active: bool
    security_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class CatalogModel(ApiModel):
    provider: str = "huggingface"
    remote_id: str
    name: str
    author: str | None = None
    pipeline_tag: str | None = None
    tags: list[str] = Field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
    trending_score: float | None = None
    created_at: datetime | None = None
    last_modified: datetime | None = None
    gated: bool | str | None = None
    private: bool = False
    library_name: str | None = None
    architecture: str | None = None
    formats: list[str] = Field(default_factory=list)
    quantizations: list[str] = Field(default_factory=list)
    parameter_count: int | None = None
    license_id: str | None = None
    total_size_bytes: int | None = None
    compatibility: str
    compatibility_reasons: list[str] = Field(default_factory=list)
    required_runtime: str | None = None
    # A CivitAI card is one *version*, because a version is what installs and
    # what the download path is bound to. These say which model it is a version
    # of, so the library can list versions under one parent without giving up
    # the version identity that install depends on. Absent for providers where
    # the repository is already the installable thing, and a card with no
    # parent renders exactly as it does now.
    parent_model_id: str | None = None
    parent_model_name: str | None = None
    # How many versions this card stands for. One means the card is the whole
    # story; more means it must open a chooser rather than install, because
    # the point of version identity is that the person picked one.
    version_count: int = 1
    # How many of them are already here, or `None` when that cannot be known.
    # Only kinds that record a provider version can answer; a checkpoint
    # install stores none, so "how many are installed" has no truthful number
    # and a guess of zero invites reinstalling what is already on disk.
    installed_version_count: int | None = None
    # Set only on workflow catalog cards: one repository can ship several
    # official workflows, and the card must say which one it is.
    workflow_template_id: str | None = None
    operation: str | None = None
    # Neutral provider-declared rating. The public app renders nothing from
    # it (general-only source); it exists so install provenance is honest and
    # downstream consumers inherit one labeling mechanism.
    content_rating: ContentRating = "unknown"


class CatalogVersionRow(ApiModel):
    """One installable version of a catalogue model, as the chooser sees it."""

    version_id: str
    version_name: str | None = None
    published_at: str | None = None
    base_model: str | None = None
    size_bytes: int = 0
    changelog: str | None = None
    # True, false, or unknown - and unknown is a real answer. Checkpoint
    # installs do not record a provider version, so for those we cannot tell
    # whether this exact version is on disk. Reporting `false` there would be
    # a claim we cannot support, and the one that would make someone install
    # a second copy of what they already have.
    installed: bool | None = None
    installed_as: str | None = None


class CatalogVersions(ApiModel):
    model_id: str
    model_name: str | None = None
    versions: list[CatalogVersionRow] = Field(default_factory=list)


class CatalogPage(ApiModel):
    items: list[CatalogModel]
    next_cursor: str | None = None
    stale: bool = False


class CatalogDetail(ApiModel):
    model: CatalogModel
    revision: str
    files: list[dict[str, Any]]


class CatalogFileVariant(ApiModel):
    """One immutable choice behind an ambiguous filename.

    Everything here is the server's: the identity, the name, and the size come
    from a freshly fetched version detail. No URL and no hash, because the
    browser has no business carrying either - it names a choice, and the
    planner re-resolves it.
    """

    source_file_id: str
    filename: str
    size_bytes: int | None
    precision: str | None


class CatalogPreflightRequest(ApiModel):
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    selected_files: list[str] = Field(default_factory=list, max_length=512)
    # Immutable provider file identities, for the case a filename cannot
    # settle: one CivitAI version can publish the same safetensors name five
    # times at different precisions, and preflight rightly refuses to guess.
    # It used to ask the caller to choose a variant and give it no way to say
    # which. Filename-only callers are unchanged.
    selected_file_ids: list[str] = Field(default_factory=list, max_length=512)
    # The exact workflow variant the user chose from the catalog. Absent for
    # repository-only callers, which keep the ranked fallback.
    workflow_template_id: str | None = Field(default=None, max_length=200)
    # A workflow named this exact file, so plan that file and nothing else.
    # Absent means an ordinary repository install, which keeps template
    # ranking. Present means the caller already knows what it needs, and
    # ranking a repository's official bundle over it would install several
    # gigabytes nobody asked for.
    workflow_reference_kind: WorkflowAssetKind | None = None
    auxiliary_kind: AuxiliaryAssetKind | None = None


class CatalogPreflightCheck(ApiModel):
    id: str
    label: str
    status: Literal["pass", "warn", "block"]
    detail: str


class CatalogFileSource(ApiModel):
    remote_id: str = Field(min_length=1, max_length=500)
    revision: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=1_000)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    # CivitAI identity; the download manager derives its URL from these
    # server-side and never consumes a catalog-supplied one.
    source_version_id: str | None = Field(default=None, pattern=r"^[1-9][0-9]{0,11}$")
    source_file_id: str | None = Field(default=None, pattern=r"^[1-9][0-9]{0,11}$")


class CatalogPreflight(ApiModel):
    remote_id: str
    source_remote_id: str | None = None
    revision: str
    selected_files: list[str]
    expected_sha256: dict[str, str] = Field(default_factory=dict)
    file_sources: dict[str, CatalogFileSource] = Field(default_factory=dict)
    comfy_paths: dict[str, str] = Field(default_factory=dict)
    workflow_template_id: str | None = None
    workflow_template_sha256: str | None = None
    download_bytes: int
    available_disk_bytes: int
    estimated_ram_bytes: int | None = None
    estimated_vram_bytes: int | None = None
    can_install: bool
    checks: list[CatalogPreflightCheck]
    install_plan: InstallPlanOut | None = None
    auxiliary_kind: str | None = None
    # The choices behind any filename this version could not settle, so a
    # refusal arrives with the answer to it. Asking someone to pick a variant
    # and then making them go and find the variants is not a choice, it is a
    # riddle.
    file_variants: dict[str, list[CatalogFileVariant]] = Field(default_factory=dict)
    # Copied server-side from the catalog detail, never client-supplied.
    content_rating: ContentRating = "unknown"


class DownloadRequest(ApiModel):
    install_plan_id: str | None = Field(default=None, max_length=40)
    remote_id: str = Field(min_length=1, max_length=500)
    source_remote_id: str | None = Field(default=None, min_length=1, max_length=500)
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    allow_patterns: list[str] = Field(default_factory=list)
    expected_sha256: dict[str, str] = Field(default_factory=dict)
    file_sources: dict[str, CatalogFileSource] = Field(default_factory=dict)
    recipe_id: str | None = None
    recipe_version: int | None = None
    comfy_paths: dict[str, str] = Field(default_factory=dict)
    workflow_path: str | None = None
    workflow_template_id: str | None = None
    workflow_template_sha256: str | None = None
    content_rating: ContentRating = "unknown"
    default_settings: dict[str, Any] = Field(default_factory=dict)
    auxiliary_kind: AuxiliaryAssetKind | None = None
    # A dependency owned by one reviewed workflow binding. Unlike an
    # auxiliary asset it is never offered for auto-application or activated as
    # a standalone profile.
    workflow_asset_kind: (
        Literal[
            "checkpoint",
            "clip_vision",
            "controlnet",
            "diffusion_model",
            "embedding",
            "gguf_model",
            "ip_adapter",
            "lora",
            "text_encoder",
            "upscaler",
            "vae",
        ]
        | None
    ) = None


class ModelAssetOut(ApiModel):
    id: str
    source_id: str | None
    name: str
    kind: str
    family: str | None
    size_bytes: int
    manifest_json: dict[str, Any]
    active: bool
    use_case: str
    auto_apply: bool
    default_model_strength: float
    default_clip_strength: float
    verified_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelAssetUpdate(ApiModel):
    active: bool | None = None
    use_case: str | None = Field(default=None, max_length=1_000)
    auto_apply: bool | None = None
    default_model_strength: float | None = Field(default=None, ge=-4, le=4)
    default_clip_strength: float | None = Field(default=None, ge=-4, le=4)


class AdapterPromptGrammarReview(ApiModel):
    """A reviewer recording how one adapter must be prompted.

    `verified_values` is the reviewer asserting what they have *seen work here*,
    not what the source document claims. That separation is the point of the
    whole record: one published vocabulary already turned out to contain a value
    the model does not implement, and prompting it degraded silently to the stem
    rather than failing.

    `approve_prose` carries exact text rather than a flag, and is accepted only
    when that text actually appears in `source_text`. Approving prose nobody
    read would mean nothing.
    """

    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: str = Field(max_length=1_000)
    source_text: str = Field(max_length=20_000)
    grammar: dict[str, Any]
    approve_prose: list[str] = Field(default_factory=list, max_length=16)
    verified_values: dict[str, list[str]] = Field(default_factory=dict)


class AdapterPromptGrammarOut(ApiModel):
    """What was recorded. Digests and decisions only - never the source text."""

    id: str
    model_asset_install_id: str
    asset_sha256: str
    source_identity: str
    source_sha256: str
    schema_version: int
    grammar_sha256: str
    approved_prose_json: list[str]
    verified_values_json: dict[str, list[str]]
    compiler_version: str
    compiler_ceiling: int
    fits: bool
    reviewed_at: datetime | None


class ReferenceSubjectCreate(ApiModel):
    """A new subject. The mention is derived unless one is asked for by name.

    Leaving `mention_slug` unset is the ordinary path and collides gracefully -
    two real people can share a name, so a derived mention is suffixed rather
    than refused. Asking for one explicitly is refused on collision instead,
    because the user asked for that exact mention and quietly giving them a
    different one would be worse than saying no.
    """

    name: str = Field(min_length=1, max_length=120)
    kind: str
    mention_slug: str | None = Field(default=None, max_length=64)
    description: str | None = Field(default=None, max_length=4_000)
    aliases: list[str] = Field(default_factory=list, max_length=32)
    tags: list[str] = Field(default_factory=list, max_length=32)


class ReferenceSubjectUpdate(ApiModel):
    """Rename, archive or favourite. Every field is optional and independent.

    `follow_mention` defaults to false: a rename does not move the mention,
    because a live chat draft may already hold the old one and silently
    breaking it is worse than a mention that no longer matches the name.
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    follow_mention: bool = False
    archived: bool | None = None
    favorite: bool | None = None
    # Omitting one leaves it alone; sending an empty string or an empty list
    # clears it. Those are different instructions and a single nullable value
    # could not carry both.
    description: str | None = Field(default=None, max_length=4_000)
    aliases: list[str] | None = Field(default=None, max_length=32)
    tags: list[str] | None = Field(default=None, max_length=32)


class ReferenceCoverIn(ApiModel):
    """Which of a reference's images stands for it.

    Its own request rather than a field on the update above, because "leave the
    cover alone" and "remove the cover" are different intentions and one
    optional field cannot say both. Clearing is the DELETE.
    """

    artifact_id: str = Field(min_length=1, max_length=80)


class ReferenceSubjectOut(ApiModel):
    id: str
    name: str
    mention_slug: str
    kind: str
    description: str | None
    aliases_json: list[str]
    tags_json: list[str]
    cover_artifact_id: str | None
    favorite: bool
    archived: bool


class ReferenceSubjectPage(ApiModel):
    items: list[ReferenceSubjectOut]
    total: int
    limit: int
    offset: int


class ReferenceDeletionImpact(ApiModel):
    """What deleting would destroy, offered before it is done.

    `exclusive_artifact_ids` counts only images nobody else references. A
    photograph showing two subjects belongs to both, and removing one of them
    is not permission to delete the picture.
    """

    reference_subject_id: str
    name: str
    asset_count: int
    exclusive_artifact_ids: list[str]


class ReferenceAssetAttach(ApiModel):
    artifact_id: str = Field(min_length=1, max_length=80)
    caption: str | None = Field(default=None, max_length=2_000)
    purpose: str = "other"
    view_label: str | None = Field(default=None, max_length=60)


class ReferenceAssetOut(ApiModel):
    id: str
    reference_subject_id: str
    artifact_id: str
    caption: str | None
    purpose: str
    view_label: str | None
    sort_order: int
    validation_state: str
    validation_reasons_json: list[str]
    width: int | None
    height: int | None
    review_version: int


ReferenceReviewReason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class ReferenceAssetReviewRequest(ApiModel):
    expected_state: Literal["unchecked"]
    expected_version: int = Field(ge=1)
    decision: Literal["usable", "weak", "rejected"]
    reasons: list[ReferenceReviewReason] = Field(default_factory=list, max_length=16)


class ReferenceAssetReviewEventOut(ApiModel):
    id: str
    reference_asset_id: str
    artifact_id: str
    artifact_sha256: str
    reviewer_kind: Literal["local-human"]
    expected_state: Literal["unchecked"]
    expected_version: int
    result_version: int
    decision: Literal["usable", "weak", "rejected"]
    reasons_json: list[str]
    width: int
    height: int
    decision_sha256: str
    reviewed_at: datetime


class ReferenceAssetReviewed(ApiModel):
    asset: ReferenceAssetOut
    review: ReferenceAssetReviewEventOut
    idempotent: bool


class ReferenceSimilarAsset(ApiModel):
    """An image already held that closely resembles the one just added."""

    reference_asset_id: str
    artifact_id: str
    mean_absolute_difference: float


class ReferenceAssetAttached(ApiModel):
    """The attachment, plus anything the caller should weigh up afterwards.

    `similar` is advice rather than a refusal: two close shots of one subject
    are often deliberate, so the person adding them decides. An empty list means
    nothing resembled it *or* the comparison could not run - never a claim that
    the image is definitely new.
    """

    asset: ReferenceAssetOut
    similar: list[ReferenceSimilarAsset]


class RecipeFile(ApiModel):
    path: str
    size_bytes: int | None = None
    sha256: str | None = None


class RecipeHardware(ApiModel):
    tier: Literal["cpu", "midrange-gpu", "high-end-gpu"]
    minimum_ram_gb: int
    recommended_ram_gb: int
    minimum_vram_gb: int | None = None
    recommended_vram_gb: int | None = None
    guidance: str


class ReferenceRecipe(ApiModel):
    id: str
    version: int
    name: str
    summary: str
    role: Literal["chat", "image", "video"]
    engine: Literal["llama.cpp", "vllm", "comfyui"]
    operations: list[str]
    license_id: str
    status: Literal["reference-candidate", "certified"]
    certified: bool
    remote_id: str
    revision: str
    files: list[RecipeFile]
    total_size_bytes: int | None
    hardware: RecipeHardware
    default_settings: dict[str, Any]
    workflow_path: str | None = None
    node_policy: str | None = None
    notes: list[str] = Field(default_factory=list)


class SettingField(ApiModel):
    key: str
    label: str
    type: Literal["boolean", "integer", "number", "string", "enum", "array", "object"]
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    multiple_of: float | None = None
    choices: list[Any] = Field(default_factory=list)
    scope: Literal["load", "request", "workflow"]
    visibility: Literal["basic", "advanced", "expert"] = "advanced"
    restart_required: bool = False
    available: bool = True
    unavailable_reason: str | None = None
    help: str = ""


class EngineCapabilities(ApiModel):
    engine: str
    version: str
    roles: list[str]
    operations: list[str]
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    formats: list[str]
    devices: list[str]
    streaming: bool
    tool_calling: bool
    settings: list[SettingField]
    settings_by_role: dict[str, list[SettingField]] = Field(default_factory=dict)
    healthy: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCapabilityProbe(ApiModel):
    engine: str
    version: str
    advertised: bool
    passed: bool
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    error: str | None = None


class DeviceInfo(ApiModel):
    id: str
    name: str
    kind: str
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    backend: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PlatformMatrixEntry(ApiModel):
    id: str
    name: str
    status: Literal["target", "experimental"]
    operating_systems: list[str]
    architectures: list[str]
    accelerator: str
    workloads: list[str]
    vram_tiers_gb: list[int] = Field(default_factory=list)
    evidence: str
    notes: list[str] = Field(default_factory=list)


class PlatformAssessment(ApiModel):
    platform_status: Literal["target", "experimental", "unsupported"]
    platform_label: str
    accelerator_status: Literal["primary", "experimental", "cpu-only"]
    accelerator_label: str
    certification_status: Literal["hardware-pending", "experimental", "unsupported"]
    chat_ready: bool
    reference_media_ready: bool
    vram_tier_gb: int | None = None
    messages: list[str] = Field(default_factory=list)


class SystemInfo(ApiModel):
    platform: str
    platform_release: str
    distribution: str
    distribution_version: str
    architecture: str
    python_version: str
    cpu_model: str
    cpu_count: int
    memory_total_bytes: int
    memory_available_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    ffmpeg_available: bool
    devices: list[DeviceInfo]
    support: PlatformAssessment


class ApplicationInfo(ApiModel):
    version: str
    data_directory: str
    log_directory: str
    max_media_outputs_per_plan: int = Field(ge=1, le=16)
    # The installation-wide gate. When this is false no chat can open its
    # own, and the UI says so rather than offering a switch that does
    # nothing.
    web_access_enabled: bool = False


class WorkerStatus(ApiModel):
    name: Literal["chat", "media"]
    state: Literal["stopped", "starting", "ready", "exited"] = "stopped"
    managed: bool
    running: bool
    pid: int | None = None
    profile_id: str | None = None
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    estimated_memory_bytes: int | None = None
    startup_duration_ms: int | None = Field(default=None, ge=0)
    current_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    active_jobs: int = 0
    queued_jobs: int = 0
    failure_detail: str | None = None
    # What kind of failure this was, and what the user can do about it. Both are
    # derived from the same output `stderr_tail` carries; neither replaces it.
    failure_code: WorkerFailureCode | None = None
    failure_remedy: str | None = None
    stderr_tail: str | None = None
    log_path: str | None = None


class WorkerSettings(ApiModel):
    # Bounds mirror Settings.worker_startup_seconds so a value accepted here is
    # never rejected when the process restarts and reads it back from disk.
    worker_startup_seconds: float = Field(ge=1, le=600)


class WorkerResetResult(ApiModel):
    worker: WorkerStatus
    cancelled_jobs: int


class WorkerLogTail(ApiModel):
    name: Literal["chat", "media"]
    text: str
    truncated: bool
    log_bytes: int


class WorkerLogLocation(ApiModel):
    path: str


class RuntimeStatus(ApiModel):
    engine: Literal["llama.cpp", "vllm", "comfyui"]
    release: str
    state: Literal["missing", "installing", "ready", "failed", "unsupported"]
    supported: bool
    managed: bool = False
    progress: float = 0
    progress_json: ProgressV2 | None = None
    downloaded_bytes: int = 0
    size_bytes: int | None = None
    distribution: str
    license: str
    security_status: Literal["checksum-pinned", "blocked"] = "checksum-pinned"
    security_message: str = ""
    message: str = ""


class SetupReadinessCheck(ApiModel):
    code: str = Field(min_length=1, max_length=80)
    status: Literal["pass", "pending", "fail"]
    message: str = Field(min_length=1, max_length=240)
    action: str | None = Field(default=None, min_length=1, max_length=80)


class SetupRoleReadiness(ApiModel):
    role: Literal["chat", "image", "video"]
    state: Literal["ready", "in_progress", "action_required"]
    verification_level: Literal["generation_probe"] = "generation_probe"
    engine: str | None = None
    job_id: str | None = None
    verification_id: str | None = None
    install_id: str | None = None
    profile_id: str | None = None
    workflow_revision_id: str | None = None
    next_action: str | None = None
    checks: list[SetupReadinessCheck] = Field(default_factory=list)


class SetupReadinessReport(ApiModel):
    version: Literal[2] = 2
    state: Literal["ready", "in_progress", "action_required"]
    roles: list[SetupRoleReadiness]


class SetupVerificationOut(ApiModel):
    id: str
    role: Literal["chat", "image", "video"]
    state: Literal["queued", "running", "ready", "failed"]
    job_id: str | None
    failure_code: str | None
    started_at: datetime | None
    completed_at: datetime | None


class BackupInfo(ApiModel):
    name: str
    size_bytes: int
    sha256: str
    created_at: datetime
    verified: bool = False
    restore_pending: bool = False
    media_included: bool = False
    media_size_bytes: int = 0


class EventOut(ApiModel):
    sequence: int
    type: str
    entity_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class HealthOut(ApiModel):
    status: Literal["ok", "degraded"]
    version: str
    database: bool
    engines: list[EngineCapabilities]


class CredentialStatus(ApiModel):
    provider: Literal["huggingface", "civitai"]
    configured: bool
    source: Literal["none", "environment", "credential_vault"]
    vault_available: bool


class CredentialSet(ApiModel):
    token: str = Field(min_length=1, max_length=10_000)
