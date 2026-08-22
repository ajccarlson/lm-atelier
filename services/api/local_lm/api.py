from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shutil
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy import Select, and_, func, or_, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload
from starlette.responses import FileResponse, HTMLResponse

from . import __version__
from .adapter_grammar_review import review_adapter_grammar
from .api_errors import api_error
from .artifact_library import (
    ArtifactLibraryConflict,
    ArtifactLibraryCursorError,
    ArtifactLibraryDataError,
    ensure_library_entry,
    list_library_entries,
    set_library_favorite,
)
from .auxiliary_assets import AUXILIARY_ASSET_KINDS, validate_lora_workflow_contract
from .capability_evidence import current_capability_evidence, evidence_input_modalities
from .capability_probe import probe_structured_tools
from .catalog_sources import CatalogSource, CatalogSourceNotFound
from .chat_deletion import (
    ExchangeBusy,
    ExchangeHasReplies,
    ExchangeNotFound,
    delete_exchange,
)
from .chat_forking import ForkSourceNotFound, fork_chat_from_message
from .civitai_catalog import CivitaiCatalog
from .comfy_editor_bridge import ComfyEditorBridgeError
from .comfy_registry import ComfyRegistryClient
from .comfy_registry_activation import (
    ComfyRegistryActivationError,
    activate_comfy_registry_install,
    deactivate_comfy_registry_install,
    review_comfy_registry_install,
)
from .comfy_registry_closure_driver import ComfyRegistryWheelMetadataClient
from .comfy_registry_downloads import ComfyRegistryArchiveDownloader
from .comfy_registry_installs import installed_comfy_registry_versions
from .comfy_registry_interpreter import probe_comfy_registry_runtime_target
from .comfy_registry_paths import registry_wheel_environment_root
from .comfy_registry_reconciliation import (
    ComfyRegistryReconciliationError,
    RegistryInstallDiskState,
    inspect_registry_install_disk_state,
)
from .comfy_registry_reconciliation import (
    remove_registry_install as remove_comfy_registry_install,
)
from .comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from .comfy_registry_wheel_projects import ComfyRegistryWheelProjectClient
from .comfy_templates import (
    COMFY_TEMPLATE_COMPILER_VERSION,
    ComfyTemplate,
    ComfyTemplateRegistry,
)
from .comfy_workflow_compiler import WorkflowCompilationError, compile_comfyui_ui_graph
from .comfy_workflow_packages import (
    ComfyWorkflowPackageAnalysis,
    WorkflowPackageError,
    analyze_comfyui_workflow_package,
)
from .config import Settings
from .credentials import (
    CredentialProvider,
    CredentialVaultUnavailable,
    credential_provider,
)
from .custom_nodes import reviewed_custom_node_types
from .db import SessionLocal, get_session
from .domain import (
    ArtifactKind,
    CompatibilityLevel,
    JobKind,
    JobStatus,
    MessageRole,
    MessageStatus,
    ModelRole,
    Operation,
    RoutingMode,
    new_id,
    utcnow,
)
from .downloads import DownloadManager
from .edit_recipes import capture_recipe
from .engines import (
    EngineNotConfiguredError,
    EngineRegistry,
    EngineSchemaUnavailableError,
)
from .filesystem_links import is_link_or_reparse
from .gguf import (
    GGUFSelectionError,
    automatic_gguf_selection,
    automatic_mmproj_selection,
)
from .hardware import collect_system_info
from .image_edit_strength import STRENGTH_MODE_PARAMETER
from .model_manifests import (
    COMFY_MODEL_FOLDERS,
    MAX_METADATA_BYTES,
    MAX_WEIGHT_HEADER_BYTES,
    ModelManifestError,
    comfy_folder_for_kind,
    inspect_repository_metadata,
)
from .model_planner import (
    INSTALL_RESOLVER_VERSION,
    ResolvedInstallPlan,
    persist_install_plan,
    resolve_install_plan,
    workflow_artifact_contract,
)
from .model_updates import installed_civitai_identities, newer_version
from .models import (
    AdapterPromptGrammar,
    AppSetting,
    Artifact,
    Chat,
    ChatWorkflowSelection,
    ComfyRegistryInstall,
    CustomNodeInstall,
    EditTemplate,
    GenerationPreset,
    InstallPlan,
    Job,
    Message,
    MessagePart,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    Project,
    ProjectWorkflowSelection,
    ReferenceAsset,
    ReferenceSubject,
    ResponseFeedback,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    SetupVerification,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowInstallOffer,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)
from .orchestrator import (
    ConversationOrchestrator,
    ProjectWorkflowPinInvalid,
    ResponseRevisionConflict,
)
from .ordered_planning import OrderedPlanConfirmationRequired
from .platforms import list_platform_matrix
from .preflight import (
    ExactCivitaiFileSelectionError,
    assess_catalog_install,
    catalog_file_index,
    safe_civitai_file_variants,
    selected_catalog_file_metadata,
)
from .profile_service import (
    AUTO_PROFILE_ID,
    LAST_CHAT_PROFILE_KEY,
    ensure_profile_for_install,
    validate_profile_binding,
    validate_profile_install,
)
from .progress import update_job_progress
from .prompt_grammar import PromptGrammarError
from .prompt_helpers import (
    PROMPT_HELPER_SCOPE,
    STANDARD_CHAT_SCOPE,
    prompt_preview_settings,
)
from .recipes import get_reference_recipe, list_reference_recipes
from .reference_library import (
    DEFAULT_PAGE,
    attach_asset,
    clear_cover,
    create_subject,
    deletion_impact,
    detach_asset,
    list_subjects,
    rename_subject,
    set_archived,
    set_cover,
    set_details,
    set_favorite,
)
from .reference_review import (
    ReferenceReviewConflict,
    ReferenceReviewInvalid,
    ReferenceReviewNotFound,
    review_reference_asset,
)
from .references import ReferenceError, ReferenceNotFoundError
from .routing import RouteConfirmationRequired
from .runtime_config import persist_runtime_values
from .schemas import (
    AdapterPromptGrammarOut,
    AdapterPromptGrammarReview,
    ApplicationInfo,
    ArtifactCleanupRequest,
    ArtifactCleanupResult,
    ArtifactDeleteResult,
    ArtifactLibraryEntrySummary,
    ArtifactLibraryItem,
    ArtifactLibraryPage,
    ArtifactOut,
    ArtifactStorageInfo,
    ArtifactUpdate,
    BackupInfo,
    BoundWorkflowAssetOut,
    CatalogDetail,
    CatalogFileVariant,
    CatalogModel,
    CatalogPage,
    CatalogPreflight,
    CatalogPreflightCheck,
    CatalogPreflightRequest,
    CatalogVersionRow,
    CatalogVersions,
    ChatCreate,
    ChatDetail,
    ChatOut,
    ChatUpdate,
    ChatWorkflowSelectionIn,
    CredentialSet,
    CredentialStatus,
    CustomNodeInstallRequest,
    CustomNodeOut,
    CustomNodeTrustRequest,
    CustomNodeUpdateRequest,
    DownloadRequest,
    DraftClassification,
    DraftClassificationRequest,
    EditTemplateCreate,
    EditTemplateOut,
    EngineCapabilities,
    ExchangeDeletionOut,
    GenerationIdentityOut,
    HealthOut,
    JobOut,
    MessageOut,
    ModelAssetOut,
    ModelAssetUpdate,
    ModelCapabilityEvidenceOut,
    ModelImport,
    ModelInstallOut,
    ModelProfileBundle,
    ModelProfileClone,
    ModelProfileCreate,
    ModelProfileOut,
    ModelProfileUpdate,
    ModelStorageInfo,
    ModelUpdateOut,
    PlatformMatrixEntry,
    PresetBundle,
    PresetClone,
    PresetCreate,
    PresetOut,
    PresetUpdate,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    ProjectWorkflowSelectionIn,
    PromptHelperCreate,
    PromptHelperDetail,
    PromptHelperUpdate,
    ReferenceAssetAttach,
    ReferenceAssetAttached,
    ReferenceAssetOut,
    ReferenceAssetReviewed,
    ReferenceAssetReviewEventOut,
    ReferenceAssetReviewRequest,
    ReferenceCoverIn,
    ReferenceDeletionImpact,
    ReferenceRecipe,
    ReferenceSimilarAsset,
    ReferenceSubjectCreate,
    ReferenceSubjectOut,
    ReferenceSubjectPage,
    ReferenceSubjectUpdate,
    RegenerateRequest,
    RegistryInstallOut,
    RegistryInstallReviewOut,
    RegistryInstallReviewRequest,
    ResolvedSetup,
    ResponseFeedbackOut,
    ResponseFeedbackUpdate,
    RunOut,
    RuntimeStatus,
    SettingField,
    SetupReadinessReport,
    SetupVerificationOut,
    StorageCleanupResult,
    StudioCapabilityReport,
    StudioSessionCreate,
    StudioToolCapability,
    SystemInfo,
    ToolCapabilityProbe,
    TrustDerivation,
    TurnAccepted,
    TurnRequest,
    VerifiedSetup,
    WorkerLogLocation,
    WorkerLogTail,
    WorkerResetResult,
    WorkerSettings,
    WorkerStatus,
    WorkflowAssetQueueRequest,
    WorkflowAssetReferenceOut,
    WorkflowAssetReviewOut,
    WorkflowAssetReviewRequest,
    WorkflowAssetSelectionIn,
    WorkflowBundle,
    WorkflowClone,
    WorkflowCreate,
    WorkflowDependencyImpactOut,
    WorkflowDependencyResourceKind,
    WorkflowEditorCancelIn,
    WorkflowEditorConsumeIn,
    WorkflowEditorDraftCreateIn,
    WorkflowEditorDraftOut,
    WorkflowEditorGraphDeltaOut,
    WorkflowEditorReturnOut,
    WorkflowEditorSessionOut,
    WorkflowFamilyOut,
    WorkflowFamilyPreferenceOut,
    WorkflowFamilyPreferenceUpdate,
    WorkflowFamilyRemovalImpactOut,
    WorkflowFamilyUpdate,
    WorkflowFamilyVariantOut,
    WorkflowInstallOfferCreate,
    WorkflowInstallOfferOut,
    WorkflowMissingNodeOut,
    WorkflowOpenTarget,
    WorkflowOut,
    WorkflowPackageAnalysisOut,
    WorkflowPackageAnalyzeRequest,
    WorkflowPackageDraftRequest,
    WorkflowPackageImportRequest,
    WorkflowPackageIssueOut,
    WorkflowPackagePrepareRequest,
    WorkflowPackageRequirementOut,
    WorkflowResourceConsumerOut,
    WorkflowResourceConsumersOut,
    WorkflowRevisionCreate,
    WorkflowRevisionOut,
    WorkflowSelectionOut,
    WorkflowSelectionResponseMode,
    WorkflowSelectorCapability,
    WorkflowSourceCandidateOut,
    WorkflowUpdate,
    WorkflowVariantReadiness,
    WorkPlanOut,
    WorkStepOut,
)
from .security import SessionSecurity
from .settings_registry import (
    defaults,
    validate_settings,
    workflow_settings,
)
from .setup_readiness import MEDIA_OPERATIONS_BY_ROLE, setup_readiness_report
from .setup_verification import (
    ACTIVE_VERIFICATION_STATES,
    SETUP_VERIFICATION_SCOPE,
    current_setup_verification,
    delete_setup_artifacts,
    ingest_synthetic_setup_image,
    setup_verification_prompt,
    setup_verification_settings,
    verification_evidence_key,
)
from .studio_capabilities import tool_capabilities
from .studio_sessions import (
    STUDIO_SCOPE,
    find_studio_session,
    studio_session_title,
)
from .verified_setup import build_verified_setup, resolve_verified_setup
from .workflow_asset_aliases import (
    WorkflowAssetAliasError,
    materialize_workflow_asset_aliases,
)
from .workflow_asset_bindings import (
    WorkflowAssetBindingError,
    WorkflowAssetBindingPlan,
    WorkflowAssetPlanSelection,
    bind_workflow_assets_to_install_plans,
)
from .workflow_asset_downloads import (
    WorkflowAssetDownloadError,
    compose_workflow_asset_download_requests,
)
from .workflow_compatibility import (
    copy_chat_workflow_selections,
    ensure_legacy_profile_workflow,
    mirror_legacy_chat_workflow_selections,
    mirror_legacy_project_workflow_selections,
    reconcile_legacy_workflow_compatibility,
    retire_legacy_profile_workflow,
)
from .workflow_edit_calibration import validate_workflow_edit_calibration
from .workflow_editor_sessions import (
    WorkflowEditorSessionError,
    workflow_api_graph_sha256,
    workflow_ui_graph_sha256,
)
from .workflow_editor_shell import (
    SHELL_SCRIPT_ROUTE,
    SHELL_STYLE_ROUTE,
    workflow_editor_origins_conflict,
    workflow_editor_shell_asset,
    workflow_editor_shell_csp,
    workflow_editor_shell_document,
)
from .workflow_install_offers import (
    WorkflowInstallOfferError,
    create_workflow_install_offer,
    invalidate_workflow_install_offer,
    mark_workflow_install_offer_queued,
    revalidate_workflow_install_offer,
)
from .workflow_library import (
    WorkflowFamilyRemovalImpact,
    workflow_family_removal_impact,
    workflow_family_selector_reference_count,
    workflow_resource_consumers,
    workflow_resource_name,
)
from .workflow_node_dependencies import node_dependency_errors
from .workflow_ownership import ensure_workflow_family_ownership
from .workflow_package_drafts import (
    is_workflow_package_draft,
    workflow_package_draft_dependencies,
)
from .workflow_package_inputs import (
    WorkflowPackageInputError,
    prepare_workflow_package_compilation,
    prepare_workflow_revision_compilation,
)
from .workflow_package_preparation import (
    PreparationContext,
    WorkflowPackagePreparationError,
    prepare_workflow_package,
)
from .workflow_source_candidates import collect_source_candidates
from .workflow_trust import (
    TRUST_DERIVATION_VERSION,
    TrustDecision,
    derive_trust,
    recorded_template_identity,
)

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]


async def get_conversation_session(request: Request) -> AsyncGenerator[Session, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


ConversationSessionDep = Annotated[Session, Depends(get_conversation_session)]


def _services(request: Request) -> Services:
    return cast("Services", request.app.state.services)


router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


async def _engine_role_fields(
    request: Request,
    role: str,
    *,
    engine: str | None = None,
    allow_inactive: bool = False,
) -> list[SettingField]:
    try:
        return await _services(request).engines.settings_for_role(
            role,
            engine=engine,
            allow_inactive=allow_inactive,
        )
    except EngineNotConfiguredError as exc:
        raise api_error(409, "engine-not-configured", str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise api_error(503, "engine-schema-unavailable", str(exc)) from exc


@router.post("/session")
async def create_session(request: Request, response: Response) -> dict[str, str | int]:
    services = _services(request)
    security: SessionSecurity = services.security
    return {
        "csrf_token": security.issue_session(response),
        "event_epoch": services.events.epoch,
        "event_sequence": services.events.sequence,
    }


@router.get("/health", response_model=HealthOut)
async def health(request: Request, session: SessionDep) -> HealthOut:
    engines: EngineRegistry = _services(request).engines
    database_healthy = True
    try:
        session.execute(text("SELECT 1")).scalar_one()
    except SQLAlchemyError:
        database_healthy = False
    try:
        capabilities = await engines.capabilities()
    except Exception:
        capabilities = []
    return HealthOut(
        status=(
            "ok"
            if database_healthy and capabilities and all(item.healthy for item in capabilities)
            else "degraded"
        ),
        version=__version__,
        database=database_healthy,
        engines=capabilities,
    )


@router.get("/ready")
async def ready() -> dict[str, str]:
    return {"version": __version__}


def _provider(value: str) -> CredentialProvider:
    try:
        return credential_provider(value)
    except ValueError as exc:
        raise api_error(404, "credential-provider-unknown", str(exc)) from exc


def _credential_status(provider: CredentialProvider, request: Request) -> CredentialStatus:
    state = _services(request).credentials.state(provider)
    return CredentialStatus(
        provider=provider,
        configured=state.configured,
        source=state.source,
        vault_available=state.vault_available,
    )


def _refresh_credential_clients(
    services: Services, provider: CredentialProvider, token: str | None
) -> None:
    if provider == "huggingface":
        services.settings.hf_token = token
        services.catalog.set_token(token)
        services.downloads.set_token(token)
    else:
        services.settings.civitai_token = token
        try:
            civitai_source = services.catalog_sources.get("civitai")
        except CatalogSourceNotFound:  # pragma: no cover - registered in build_services
            return
        if isinstance(civitai_source, CivitaiCatalog):
            civitai_source.set_token(token)


@router.get("/credentials/{provider}", response_model=CredentialStatus)
async def credential_status(provider: str, request: Request) -> CredentialStatus:
    return _credential_status(_provider(provider), request)


@router.put("/credentials/{provider}", response_model=CredentialStatus)
async def set_credential(
    provider: str, payload: CredentialSet, request: Request
) -> CredentialStatus:
    services = _services(request)
    selected = _provider(provider)
    try:
        services.credentials.set_token(payload.token, selected)
    except ValueError as exc:
        raise api_error(409, "credential-rejected", str(exc)) from exc
    except CredentialVaultUnavailable as exc:
        raise api_error(503, "credential-vault-unavailable", str(exc)) from exc
    _refresh_credential_clients(services, selected, services.credentials.token(selected))
    return _credential_status(selected, request)


@router.delete("/credentials/{provider}", response_model=CredentialStatus)
async def delete_credential(provider: str, request: Request) -> CredentialStatus:
    services = _services(request)
    selected = _provider(provider)
    try:
        services.credentials.delete_token(selected)
    except ValueError as exc:
        raise api_error(409, "credential-rejected", str(exc)) from exc
    except CredentialVaultUnavailable as exc:
        raise api_error(503, "credential-vault-unavailable", str(exc)) from exc
    _refresh_credential_clients(services, selected, None)
    return _credential_status(selected, request)


@router.get("/system", response_model=SystemInfo)
async def system_info(request: Request) -> SystemInfo:
    settings: Settings = _services(request).settings
    return collect_system_info(settings)


@router.get("/about", response_model=ApplicationInfo)
async def application_info(request: Request) -> ApplicationInfo:
    settings: Settings = _services(request).settings
    return ApplicationInfo(
        version=__version__,
        data_directory=str(settings.data_dir.resolve()),
        log_directory=str(settings.log_dir.resolve()),
        max_media_outputs_per_plan=settings.max_media_outputs_per_plan,
        web_access_enabled=settings.web_access_enabled,
    )


@router.get("/platforms", response_model=list[PlatformMatrixEntry])
async def platform_matrix() -> list[PlatformMatrixEntry]:
    return list_platform_matrix()


@router.post("/diagnostics", response_model=ArtifactOut, status_code=201)
async def create_diagnostics(request: Request, session: SessionDep) -> ArtifactOut:
    artifact = _services(request).diagnostics.create(session)
    session.commit()
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/backups", response_model=list[BackupInfo])
async def list_backups(request: Request) -> list[BackupInfo]:
    return await asyncio.to_thread(_services(request).backups.list)


@router.post("/backups", response_model=BackupInfo, status_code=201)
async def create_backup(request: Request, include_media: bool = False) -> BackupInfo:
    return await asyncio.to_thread(
        _services(request).backups.create,
        include_media=include_media,
    )


@router.post("/backups/{name}/verify", response_model=BackupInfo)
async def verify_backup(name: str, request: Request) -> BackupInfo:
    try:
        return await asyncio.to_thread(_services(request).backups.verify, name)
    except FileNotFoundError as exc:
        raise api_error(404, "backup-not-found", "That backup no longer exists.") from exc
    except ValueError as exc:
        raise api_error(422, "backup-invalid", str(exc)) from exc


@router.post("/backups/{name}/restore", response_model=BackupInfo)
async def restore_backup(name: str, request: Request) -> BackupInfo:
    try:
        return await asyncio.to_thread(_services(request).backups.request_restore, name)
    except FileNotFoundError as exc:
        raise api_error(404, "backup-not-found", "That backup no longer exists.") from exc
    except ValueError as exc:
        raise api_error(422, "backup-invalid", str(exc)) from exc


@router.delete("/backups/{name}", status_code=204)
async def delete_backup(name: str, request: Request) -> Response:
    try:
        await asyncio.to_thread(_services(request).backups.delete, name)
    except FileNotFoundError as exc:
        raise api_error(404, "backup-not-found", "That backup no longer exists.") from exc
    except ValueError as exc:
        raise api_error(422, "backup-invalid", str(exc)) from exc
    return Response(status_code=204)


@router.get("/engines", response_model=list[EngineCapabilities])
async def engine_capabilities(request: Request) -> list[EngineCapabilities]:
    try:
        return await _services(request).engines.capabilities()
    except EngineSchemaUnavailableError as exc:
        raise api_error(503, "engine-schema-unavailable", str(exc)) from exc


@router.post("/engines/chat/tool-probe", response_model=ToolCapabilityProbe)
async def probe_chat_tool_capability(request: Request) -> ToolCapabilityProbe:
    return await probe_structured_tools(_services(request).engines.chat)


@router.get("/workers", response_model=list[WorkerStatus])
async def worker_status(request: Request, session: SessionDep) -> list[WorkerStatus]:
    statuses = _services(request).processes.statuses()
    for index, status in enumerate(statuses):
        kinds = (
            [JobKind.CHAT.value]
            if status.name == "chat"
            else [JobKind.IMAGE.value, JobKind.VIDEO.value]
        )
        active_jobs = (
            session.scalar(
                select(func.count(Job.id)).where(Job.kind.in_(kinds), Job.status == "running")
            )
            or 0
        )
        queued_jobs = (
            session.scalar(
                select(func.count(Job.id)).where(
                    Job.kind.in_(kinds), Job.status.in_(["queued", "paused"])
                )
            )
            or 0
        )
        statuses[index] = status.model_copy(
            update={"active_jobs": active_jobs, "queued_jobs": queued_jobs}
        )
    return statuses


@router.get("/workers/settings", response_model=WorkerSettings)
async def get_worker_settings(request: Request) -> WorkerSettings:
    return WorkerSettings(worker_startup_seconds=_services(request).settings.worker_startup_seconds)


@router.put("/workers/settings", response_model=WorkerSettings)
async def update_worker_settings(payload: WorkerSettings, request: Request) -> WorkerSettings:
    services = _services(request)
    seconds = payload.worker_startup_seconds
    # Assigning the live Settings object takes effect on the next worker start;
    # persisting keeps the value across application restarts.
    services.settings.worker_startup_seconds = seconds
    persist_runtime_values(
        services.settings.data_dir,
        {"LOCAL_LM_WORKER_STARTUP_SECONDS": f"{seconds:g}"},
    )
    return WorkerSettings(worker_startup_seconds=services.settings.worker_startup_seconds)


@router.get("/runtimes", response_model=list[RuntimeStatus])
def runtime_status(request: Request) -> list[RuntimeStatus]:
    # Deliberately synchronous: reading runtime status touches the filesystem, so
    # the framework runs this in a worker thread rather than on the event loop.
    return _services(request).runtimes.statuses()


@router.get("/setup/readiness", response_model=SetupReadinessReport)
def get_setup_readiness(
    request: Request,
    session: SessionDep,
) -> SetupReadinessReport:
    # Deliberately synchronous, and polled every few seconds while setup is
    # incomplete: this reads runtime status from disk and queries the database,
    # so it belongs in a worker thread rather than blocking every other request.
    services = _services(request)
    return setup_readiness_report(
        session,
        services.settings,
        services.runtimes,
        services.processes.statuses(),
    )


@router.post("/setup/resolve-setup", response_model=ResolvedSetup)
def resolve_imported_setup(
    payload: VerifiedSetup, request: Request, session: SessionDep
) -> ResolvedSetup:
    """Report what an imported setup finds here, and what it would still need.

    Reports only. Nothing is downloaded, pinned, trusted or activated: an
    imported file must not be able to make this machine fetch several gigabytes
    because it says so. Missing components stay behind the existing
    approval-gated install path.

    Its attestation travels as provenance, never as permission. "Verified
    elsewhere on compatible hardware" is reported separately from anything this
    machine has earned, because the two must never be confused - a hardware
    envelope establishes compatibility, not identity of the local runtime,
    drivers, files, or result.
    """
    return ResolvedSetup.model_validate(
        resolve_verified_setup(
            session,
            payload.model_dump(mode="json"),
            _services(request).settings,
        )
    )


@router.get("/setup/verified-setup/{role}", response_model=VerifiedSetup)
def export_verified_setup(
    role: Literal["chat", "image", "video"],
    request: Request,
    session: SessionDep,
) -> VerifiedSetup:
    """Export a setup that is known to work, with nothing local left in it.

    Refuses unless a generation actually succeeded for this exact configuration.
    A record that only says "this ought to work" is what the user already has;
    the attestation is the part worth shipping.
    """
    services = _services(request)
    report = setup_readiness_report(
        session,
        services.settings,
        services.runtimes,
        services.processes.statuses(),
    )
    readiness = next(item for item in report.roles if item.role == role)
    install = session.get(ModelInstall, readiness.install_id) if readiness.install_id else None
    profile = session.get(ModelProfile, readiness.profile_id) if readiness.profile_id else None
    if not install or not profile:
        raise api_error(409, "setup-not-verified", "This role has no verified setup to export yet.")
    workflow = (
        session.get(WorkflowRevision, readiness.workflow_revision_id)
        if readiness.workflow_revision_id
        else None
    )
    evidence = current_capability_evidence(
        session,
        install,
        services.settings,
        services.runtimes,
    )
    if not evidence:
        raise api_error(
            409, "setup-evidence-missing", "This setup has no current activation evidence."
        )
    verification = current_setup_verification(session, role, install, profile, workflow, evidence)
    if not verification or verification.state != "verified":
        raise api_error(
            409,
            "setup-not-verified",
            "Run setup verification for this role first - an exported setup has to "
            "carry proof that a real generation succeeded.",
        )
    return VerifiedSetup.model_validate(
        build_verified_setup(
            session,
            verification=verification,
            install=install,
            profile=profile,
            revision=workflow,
            evidence=evidence,
        )
    )


@router.post(
    "/setup/verify/{role}",
    response_model=SetupVerificationOut,
    status_code=202,
)
async def start_setup_verification(
    role: Literal["chat", "image", "video"],
    request: Request,
    session: ConversationSessionDep,
) -> SetupVerification:
    services = _services(request)
    # Before deciding this role is blocked, vouch for anything this machine can
    # rebuild for itself. An imported workflow is otherwise untrusted, and
    # `workflow_untrusted` blocks verification with "review the workflow" as the
    # only offered remedy.
    await _vouch_for_role_workflows(services, session, role)
    report = setup_readiness_report(
        session,
        services.settings,
        services.runtimes,
        services.processes.statuses(),
    )
    readiness = next(item for item in report.roles if item.role == role)
    blocking = [
        check
        for check in readiness.checks
        if check.status != "pass" and not check.code.startswith("generation_verification")
    ]
    if blocking:
        raise api_error(409, "setup-incomplete", blocking[0].message)
    if not readiness.install_id or not readiness.profile_id:
        raise api_error(409, "setup-incomplete", "Finish model activation and profile setup first.")

    install = session.get(ModelInstall, readiness.install_id)
    profile = session.get(ModelProfile, readiness.profile_id)
    workflow = (
        session.get(WorkflowRevision, readiness.workflow_revision_id)
        if readiness.workflow_revision_id
        else None
    )
    if not install or not profile:
        raise api_error(409, "setup-changed", "The selected setup changed. Refresh and try again.")
    capability_evidence = current_capability_evidence(
        session,
        install,
        services.settings,
        services.runtimes,
    )
    if not capability_evidence:
        raise api_error(
            409,
            "setup-evidence-changed",
            "The model activation evidence changed. Refresh and try again.",
        )

    existing = current_setup_verification(
        session,
        role,
        install,
        profile,
        workflow,
        capability_evidence,
    )
    if existing and (existing.state in ACTIVE_VERIFICATION_STATES or existing.state == "ready"):
        return existing

    fields = workflow_settings(
        await _engine_role_fields(request, role),
        workflow.input_schema_json if workflow else None,
    )
    settings = setup_verification_settings(fields, role)
    verification = existing or SetupVerification(
        role=role,
        evidence_key=verification_evidence_key(
            role,
            install,
            profile,
            workflow,
            capability_evidence,
        ),
        model_install_id=install.id,
        profile_id=profile.id,
        workflow_revision_id=workflow.id if workflow else None,
    )
    verification.state = "queued"
    verification.failure_code = None
    verification.started_at = None
    verification.completed_at = None
    verification.run_id = None
    verification.job_id = None
    verification.input_artifact_id = None
    if not existing:
        session.add(verification)
    session.flush()

    routing_mode = RoutingMode.TEXT if role == "chat" else RoutingMode(role)
    chat = Chat(
        title="Setup verification",
        archived=True,
        scope=SETUP_VERIFICATION_SCOPE,
        routing_mode=routing_mode.value,
        confirm_uncertain_media=False,
        active_chat_profile_id=profile.id if role == "chat" else AUTO_PROFILE_ID,
        active_vision_profile_id=AUTO_PROFILE_ID,
        active_image_profile_id=profile.id if role == "image" else AUTO_PROFILE_ID,
        active_video_profile_id=profile.id if role == "video" else AUTO_PROFILE_ID,
        generation_settings_json={role: settings},
    )
    session.add(chat)
    session.flush()
    mirror_legacy_chat_workflow_selections(session, chat)
    verification.chat_id = chat.id

    input_artifact_ids: list[str] = []
    if workflow:
        definition = session.get(WorkflowDefinition, workflow.workflow_id)
        if definition and definition.operation in {"image_to_image", "image_to_video"}:
            artifact = ingest_synthetic_setup_image(
                session,
                services.artifacts,
                verification.id,
            )
            verification.input_artifact_id = artifact.id
            input_artifact_ids.append(artifact.id)
    session.commit()

    try:
        accepted = await _accept_turn(
            services.orchestrator,
            session,
            chat.id,
            TurnRequest(
                text=setup_verification_prompt(role),
                mode=routing_mode,
                input_artifact_ids=input_artifact_ids,
                settings=settings,
                confirm_media=True,
                idempotency_key=f"setup-verification:{verification.id}",
            ),
            source_action="setup_verification",
        )
    except Exception:
        session.expire_all()
        current = session.get(SetupVerification, verification.id)
        if current:
            current.state = "failed"
            current.failure_code = "generation_not_started"
            current.completed_at = utcnow()
            artifact_ids = {current.input_artifact_id} if current.input_artifact_id else set()
            if current.chat_id and (failed_chat := session.get(Chat, current.chat_id)):
                session.delete(failed_chat)
            current.chat_id = None
            current.input_artifact_id = None
            session.flush()
            delete_setup_artifacts(session, services.artifacts, artifact_ids)
            session.commit()
        raise

    session.expire_all()
    current = session.get(SetupVerification, verification.id)
    if not current:
        raise api_error(500, "setup-verification-lost", "Setup verification state was lost.")
    job = session.scalar(select(Job).where(Job.run_id == accepted.run.id))
    if current.state in ACTIVE_VERIFICATION_STATES:
        current.run_id = accepted.run.id
        current.job_id = job.id if job else None
        if job:
            job.payload_json = {
                **job.payload_json,
                "setup_verification_id": current.id,
            }
        session.commit()
        session.refresh(current)
    return current


@router.post("/runtimes/{engine}/install", response_model=RuntimeStatus, status_code=202)
async def install_runtime(engine: str, request: Request) -> RuntimeStatus:
    if engine not in {"llama.cpp", "vllm", "comfyui"}:
        raise api_error(422, "runtime-unknown", "runtime must be llama.cpp, vllm, or comfyui")
    status = _services(request).runtimes.start(
        cast(Literal["llama.cpp", "vllm", "comfyui"], engine)
    )
    if status.state == "unsupported":
        raise api_error(422, "runtime-unavailable", status.message)
    return status


def _worker_job_kinds(name: str) -> list[str]:
    return [JobKind.CHAT.value] if name == "chat" else [JobKind.IMAGE.value, JobKind.VIDEO.value]


def _ensure_worker_idle(session: Session, name: str) -> None:
    busy_jobs = (
        session.scalar(
            select(func.count(Job.id)).where(
                Job.kind.in_(_worker_job_kinds(name)),
                Job.status.in_(["queued", "running", "paused"]),
            )
        )
        or 0
    )
    if busy_jobs:
        raise api_error(
            409,
            "worker-busy",
            f"the {name} worker has {busy_jobs} active or queued "
            f"{'job' if busy_jobs == 1 else 'jobs'}; cancel or wait for them before "
            "changing the worker",
            busy_jobs=busy_jobs,
        )


def _validated_profile_install(
    session: Session,
    *,
    model_install_id: str | None,
    role: str,
    engine: str,
) -> ModelInstall | None:
    try:
        return validate_profile_install(
            session,
            model_install_id=model_install_id,
            role=role,
            engine=engine,
        )
    except LookupError as exc:
        raise api_error(404, "profile-install-missing", str(exc)) from exc
    except ValueError as exc:
        raise api_error(422, "profile-install-invalid", str(exc)) from exc


@router.post("/workers/chat/load/{profile_id}", response_model=WorkerStatus)
async def load_chat_worker(profile_id: str, request: Request, session: SessionDep) -> WorkerStatus:
    services = _services(request)
    # Check before taking the lease: a wedged job holds "primary" indefinitely,
    # and a request that hangs there cannot even report the 409 that explains it.
    _ensure_worker_idle(session, "chat")
    async with services.scheduler.lease("primary"):
        session.expire_all()
        _ensure_worker_idle(session, "chat")
        return await _load_chat_profile(services, session, profile_id)


async def _load_chat_profile(services: Services, session: Session, profile_id: str) -> WorkerStatus:
    profile = session.get(ModelProfile, profile_id)
    if not profile or not profile.model_install_id:
        raise api_error(404, "profile-install-missing", "profile with a model install not found")
    if profile.role != ModelRole.CHAT.value:
        raise api_error(422, "profile-role-mismatch", "chat worker requires a chat profile")
    install = _validated_profile_install(
        session,
        model_install_id=profile.model_install_id,
        role=profile.role,
        engine=profile.engine,
    )
    if not install:
        raise api_error(404, "profile-install-missing", "profile with a model install not found")
    try:
        status = await services.processes.load_chat(profile, install)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        raise api_error(422, "chat-worker-start-failed", str(exc)) from exc
    setting = session.get(AppSetting, LAST_CHAT_PROFILE_KEY)
    if setting:
        setting.value_json = profile.id
    else:
        session.add(AppSetting(key=LAST_CHAT_PROFILE_KEY, value_json=profile.id))
    session.commit()
    return status


@router.post("/workers/media/start", response_model=WorkerStatus)
async def start_media_worker(request: Request, session: SessionDep) -> WorkerStatus:
    services = _services(request)
    _ensure_worker_idle(session, "media")
    async with services.scheduler.lease("primary"):
        session.expire_all()
        _ensure_worker_idle(session, "media")
        if services.settings.media_engine != "comfyui":
            raise api_error(422, "media-engine-inactive", "The ComfyUI media engine is not active.")
        try:
            return await services.processes.start_media()
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            raise api_error(422, "media-worker-start-failed", str(exc)) from exc


@router.post("/workers/{name}/stop", response_model=WorkerStatus)
async def stop_worker(name: str, request: Request, session: SessionDep) -> WorkerStatus:
    if name not in {"chat", "media"}:
        raise api_error(422, "worker-unknown", "worker must be chat or media")
    services = _services(request)
    _ensure_worker_idle(session, name)
    async with services.scheduler.lease("primary"):
        session.expire_all()
        _ensure_worker_idle(session, name)
        try:
            return await services.processes.stop(name)
        except ValueError as exc:
            raise api_error(422, "worker-stop-failed", str(exc)) from exc


@router.post("/workers/{name}/restart", response_model=WorkerStatus)
async def restart_worker(name: str, request: Request, session: SessionDep) -> WorkerStatus:
    if name not in {"chat", "media"}:
        raise api_error(422, "worker-unknown", "worker must be chat or media")
    services = _services(request)
    _ensure_worker_idle(session, name)
    async with services.scheduler.lease("primary"):
        session.expire_all()
        _ensure_worker_idle(session, name)
        if name == "media":
            if services.settings.media_engine != "comfyui":
                raise api_error(
                    422, "media-engine-inactive", "The ComfyUI media engine is not active."
                )
            try:
                await services.processes.stop("media")
                return await services.processes.start_media()
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                raise api_error(422, "media-worker-restart-failed", str(exc)) from exc
        # The chat worker restarts with the model it ran last, whether or not it
        # is currently running - a crashed worker still knows what to reload.
        record = next((item for item in services.processes.statuses() if item.name == "chat"), None)
        profile_id = record.profile_id if record else None
        if not profile_id:
            setting = session.get(AppSetting, LAST_CHAT_PROFILE_KEY)
            profile_id = setting.value_json if setting else None
        if not isinstance(profile_id, str) or not profile_id:
            raise api_error(
                409,
                "chat-model-not-loaded",
                "no chat model has been loaded yet; load a profile first",
            )
        return await _load_chat_profile(services, session, profile_id)


@router.post("/workers/{name}/reset", response_model=WorkerResetResult)
async def reset_worker(name: str, request: Request, session: SessionDep) -> WorkerResetResult:
    """Cancel this worker's blocking jobs and stop it.

    The escape hatch for a wedged worker: every other worker control refuses to
    act while jobs are queued, so a job that never finishes would otherwise lock
    out exactly the control needed to clear it. Deliberately takes no scheduler
    lease - the wedged job may hold it forever.
    """

    if name not in {"chat", "media"}:
        raise api_error(422, "worker-unknown", "worker must be chat or media")
    services = _services(request)
    job_ids = list(
        session.scalars(
            select(Job.id).where(
                Job.kind.in_(_worker_job_kinds(name)),
                Job.status.in_(["queued", "running", "paused"]),
            )
        )
    )
    cancelled = 0
    for job_id in job_ids:
        if await services.orchestrator.cancel(job_id):
            cancelled += 1
    try:
        worker = await services.processes.stop(name)
    except ValueError as exc:
        raise api_error(422, "worker-stop-failed", str(exc)) from exc
    session.expire_all()
    return WorkerResetResult(worker=worker, cancelled_jobs=cancelled)


WORKER_LOG_TAIL_BYTES = 64 * 1024


@router.get("/workers/log-location", response_model=WorkerLogLocation)
def worker_log_location(request: Request) -> WorkerLogLocation:
    """The absolute log folder, for the user to open themselves.

    Deliberately a path and not an "open folder" action: the hardened local
    HTTP surface has no endpoint that executes anything, and this keeps it so.
    """

    return WorkerLogLocation(path=str(_services(request).settings.log_dir.resolve()))


@router.get("/workers/{name}/log-tail", response_model=WorkerLogTail)
def worker_log_tail(name: str, request: Request) -> WorkerLogTail:
    # Deliberately synchronous: this reads a file, so the framework runs it in
    # a worker thread rather than on the event loop.
    if name not in {"chat", "media"}:
        raise api_error(422, "worker-unknown", "worker must be chat or media")
    path = _services(request).settings.log_dir / f"{name}-worker.log"
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > WORKER_LOG_TAIL_BYTES:
                handle.seek(-WORKER_LOG_TAIL_BYTES, os.SEEK_END)
            payload = handle.read(WORKER_LOG_TAIL_BYTES)
    except OSError:
        size = 0
        payload = b""
    return WorkerLogTail(
        name=cast(Literal["chat", "media"], name),
        text=payload.decode("utf-8", errors="replace"),
        truncated=size > WORKER_LOG_TAIL_BYTES,
        log_bytes=size,
    )


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(
    session: SessionDep,
    include_archived: bool = False,
    query: str = Query(default="", max_length=500),
) -> list[Project]:
    statement = select(Project).order_by(Project.pinned.desc(), Project.updated_at.desc())
    if not include_archived:
        statement = statement.where(Project.archived.is_(False))
    if query.strip():
        statement = statement.where(Project.name.ilike(f"%{query.strip()}%"))
    return list(session.scalars(statement).all())


def _validate_project_workflow_pins(session: Session, values: dict[str, Any]) -> None:
    expected = {
        "image_workflow_revision_id": {
            Operation.TEXT_TO_IMAGE.value,
            Operation.IMAGE_TO_IMAGE.value,
        },
        "video_workflow_revision_id": {
            Operation.TEXT_TO_VIDEO.value,
            Operation.IMAGE_TO_VIDEO.value,
        },
    }
    for field, operations in expected.items():
        revision_id = values.get(field)
        if revision_id is None:
            continue
        revision = session.get(WorkflowRevision, revision_id)
        definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
        if not revision or not definition:
            raise api_error(
                422,
                "workflow-revision-unknown",
                f"{field} does not identify a workflow revision",
            )
        if definition.operation not in operations:
            raise api_error(
                422,
                "workflow-operation-mismatch",
                f"{field} has an incompatible workflow operation",
            )


async def _validate_generation_defaults(
    request: Request,
    session: Session,
    values: dict[str, Any],
) -> None:
    scoped = values.get("generation_settings_json")
    if scoped is None and "generation_settings_json" in values:
        values["generation_settings_json"] = {}
        scoped = {}
    if scoped:
        for role, settings in scoped.items():
            if len(settings) > 256 or any(len(key) > 200 for key in settings):
                raise api_error(
                    422,
                    "generation-defaults-too-large",
                    f"{role} generation defaults are too large",
                )
            request_settings = settings
            if STRENGTH_MODE_PARAMETER in settings:
                mode = settings[STRENGTH_MODE_PARAMETER]
                if role != ModelRole.IMAGE.value or mode not in {"auto", "manual"}:
                    raise api_error(
                        422,
                        "strength-mode-invalid",
                        "image edit strength mode must be auto or manual for image defaults",
                    )
                request_settings = {
                    key: value for key, value in settings.items() if key != STRENGTH_MODE_PARAMETER
                }
            fields = await _engine_role_fields(request, role)
            request_fields = [field for field in fields if field.scope != "load"]
            load_keys = {field.key for field in fields if field.scope == "load"}
            disallowed = sorted(load_keys & set(request_settings))
            if disallowed:
                raise api_error(
                    422,
                    "generation-defaults-load-settings",
                    f"{role} generation defaults cannot include load settings: "
                    f"{', '.join(disallowed)}",
                )
            try:
                validate_settings(request_settings, request_fields)
            except ValueError as exc:
                raise api_error(422, "generation-defaults-invalid", str(exc)) from exc

    bindings = values.get("generation_preset_ids_json")
    if bindings is None and "generation_preset_ids_json" in values:
        values["generation_preset_ids_json"] = {}
        bindings = {}
    for role, preset_id in (bindings or {}).items():
        if preset_id is None:
            continue
        if len(preset_id) > 40:
            raise api_error(
                422, "generation-preset-id-too-long", f"{role} generation preset id is too long"
            )
        preset = session.get(GenerationPreset, preset_id)
        if not preset:
            raise api_error(
                404, "generation-preset-not-found", f"{role} generation preset not found"
            )
        if preset.role != role:
            raise api_error(
                422,
                "generation-preset-role-mismatch",
                f"{role} generation preset has an incompatible role",
            )


@router.post("/projects", response_model=ProjectOut, status_code=201)
async def create_project(
    payload: ProjectCreate,
    request: Request,
    session: SessionDep,
) -> Project:
    values = payload.model_dump()
    _validate_project_workflow_pins(session, values)
    await _validate_generation_defaults(request, session, values)
    project = Project(**values)
    session.add(project)
    session.flush()
    mirror_legacy_project_workflow_selections(session, project)
    session.commit()
    session.refresh(project)
    return project


@router.post("/projects/import", response_model=ProjectOut, status_code=201)
async def import_project(
    request: Request,
    session: SessionDep,
    archive: Annotated[UploadFile, File()],
) -> Project:
    archive.file.seek(0, 2)
    size = archive.file.tell()
    archive.file.seek(0)
    if size > _services(request).settings.max_project_import_bytes:
        raise api_error(
            413, "project-archive-too-large", "project archive exceeds the configured limit"
        )
    # Resolve the live engine schema per role so imported settings are validated
    # the way the REST API validates them. Best effort: a role whose engine is
    # not configured, or whose schema cannot be read now, imports unvalidated
    # rather than failing the whole archive.
    known_fields: dict[str, list[SettingField]] = {}
    for settings_role in ("chat", "image", "video"):
        try:
            known_fields[settings_role] = await _engine_role_fields(
                request,
                settings_role,
                allow_inactive=True,
            )
        except HTTPException:
            continue
    try:
        project = _services(request).exports.import_archive(
            session,
            archive.file,
            known_fields=known_fields,
        )
    except ValueError as exc:
        session.rollback()
        raise api_error(422, "project-import-invalid", str(exc)) from exc
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    session.refresh(project)
    return project


@router.patch("/projects/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str,
    payload: ProjectUpdate,
    request: Request,
    session: SessionDep,
) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise api_error(404, "project-not-found", "project not found")
    values = payload.model_dump(exclude_unset=True)
    _validate_project_workflow_pins(session, values)
    await _validate_generation_defaults(request, session, values)
    for key, value in values.items():
        setattr(project, key, value)
    changed_capabilities = [
        capability
        for capability, field in {
            "image": "image_workflow_revision_id",
            "video": "video_workflow_revision_id",
        }.items()
        if field in values
    ]
    if changed_capabilities:
        mirror_legacy_project_workflow_selections(
            session,
            project,
            cast(list[Literal["image", "video"]], changed_capabilities),
        )
    session.commit()
    session.refresh(project)
    return project


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(project_id: str, session: SessionDep) -> Response:
    project = session.get(Project, project_id)
    if not project:
        raise api_error(404, "project-not-found", "project not found")
    for chat in project.chats:
        chat.project_id = None
    session.delete(project)
    session.commit()
    return Response(status_code=204)


@router.post("/projects/{project_id}/export", response_model=ArtifactOut, status_code=201)
async def export_project(
    project_id: str,
    request: Request,
    session: SessionDep,
    include_media: bool = True,
) -> ArtifactOut:
    try:
        artifact = _services(request).exports.export(
            session, project_id, include_media=include_media
        )
    except LookupError as exc:
        raise api_error(404, "project-not-found", str(exc)) from exc
    session.commit()
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/chats", response_model=list[ChatOut])
async def list_chats(
    session: ConversationSessionDep,
    project_id: str | None = None,
    include_archived: bool = False,
    query: str = Query(default="", max_length=500),
) -> list[Chat]:
    statement = (
        select(Chat)
        .where(Chat.scope == STANDARD_CHAT_SCOPE)
        .order_by(Chat.pinned.desc(), Chat.updated_at.desc())
    )
    if project_id:
        statement = statement.where(Chat.project_id == project_id)
    if not include_archived:
        statement = statement.where(Chat.archived.is_(False))
    if query.strip():
        statement = statement.where(Chat.title.ilike(f"%{query.strip()}%"))
    return list(session.scalars(statement).all())


@router.post("/chats", response_model=ChatOut, status_code=201)
async def create_chat(
    payload: ChatCreate,
    request: Request,
    session: ConversationSessionDep,
) -> Chat:
    if payload.project_id and not session.get(Project, payload.project_id):
        raise api_error(404, "project-not-found", "project not found")
    values = payload.model_dump(mode="json")
    await _validate_generation_defaults(request, session, values)
    chat = Chat(
        title=payload.title,
        project_id=payload.project_id,
        routing_mode=payload.routing_mode.value,
        generation_settings_json=values["generation_settings_json"],
        generation_preset_ids_json=values["generation_preset_ids_json"],
        vision_settings_json=values["vision_settings_json"],
        active_chat_profile_id=AUTO_PROFILE_ID,
        active_vision_profile_id=AUTO_PROFILE_ID,
        active_image_profile_id=AUTO_PROFILE_ID,
        active_video_profile_id=AUTO_PROFILE_ID,
    )
    session.add(chat)
    session.flush()
    mirror_legacy_chat_workflow_selections(session, chat)
    session.commit()
    session.refresh(chat)
    return chat


@router.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: str, session: ConversationSessionDep) -> Chat:
    chat = session.scalar(
        select(Chat)
        .options(
            selectinload(Chat.messages)
            .selectinload(Message.parts)
            .selectinload(MessagePart.artifact),
            selectinload(Chat.messages)
            .selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.parts)
            .selectinload(ResponseRevisionPart.artifact),
            selectinload(Chat.messages).selectinload(Message.feedback_rows),
            # One query for the whole chat rather than one per message. Almost
            # every message names nothing, so this is usually an empty result
            # rather than a cost. Every loader that returns a ChatDetail needs
            # it, or the transcript falls back to a query per message.
            selectinload(Chat.messages).selectinload(Message.references),
            selectinload(Chat.messages)
            .selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.feedback_rows),
        )
        .where(Chat.id == chat_id, Chat.scope == STANDARD_CHAT_SCOPE)
    )
    if not chat:
        raise api_error(404, "chat-not-found", "chat not found")
    return chat


@router.put("/messages/{message_id}/feedback", response_model=ResponseFeedbackOut)
async def set_response_feedback(
    message_id: str, payload: ResponseFeedbackUpdate, session: SessionDep
) -> ResponseFeedbackOut:
    """Record one local preference verdict; a null rating clears it.

    The run pointer is the provenance anchor - model, profile, and effective
    settings already live there - so nothing is copied and nothing trains.
    """

    message = session.get(Message, message_id)
    if not message or message.role != MessageRole.ASSISTANT.value:
        raise api_error(404, "message-not-found", "This response no longer exists")
    revision = None
    if payload.response_revision_id is not None:
        revision = session.get(ResponseRevision, payload.response_revision_id)
        if not revision or revision.message_id != message.id:
            raise api_error(404, "revision-not-found", "This response revision no longer exists")
    existing = session.scalar(
        select(ResponseFeedback).where(
            ResponseFeedback.message_id == message.id,
            ResponseFeedback.response_revision_id == payload.response_revision_id,
        )
    )
    if payload.rating is None:
        if existing:
            session.delete(existing)
            session.commit()
        return ResponseFeedbackOut(
            message_id=message.id,
            response_revision_id=payload.response_revision_id,
            rating=None,
        )
    if existing:
        existing.rating = payload.rating
    else:
        session.add(
            ResponseFeedback(
                message_id=message.id,
                response_revision_id=payload.response_revision_id,
                run_id=revision.run_id if revision else None,
                rating=payload.rating,
            )
        )
    session.commit()
    return ResponseFeedbackOut(
        message_id=message.id,
        response_revision_id=payload.response_revision_id,
        rating=payload.rating,
    )


def _prompt_helper_query(helper_id: str) -> Select[tuple[Chat]]:
    return (
        select(Chat)
        .options(
            selectinload(Chat.messages)
            .selectinload(Message.parts)
            .selectinload(MessagePart.artifact),
            selectinload(Chat.messages)
            .selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.parts)
            .selectinload(ResponseRevisionPart.artifact),
        )
        .where(Chat.id == helper_id, Chat.scope == PROMPT_HELPER_SCOPE)
    )


def _is_editable_image(artifact: Artifact) -> bool:
    """Whether the studio can open this, judged by what it is.

    `kind` records where an artifact came from, not what it holds: a generated
    picture is `image` and an uploaded one is `input`. Asking `kind == image`
    therefore asked "did we make this", and refused every picture a person
    brought in themselves - which is most of the reason to open the studio at
    all.

    The media type is the fact about content, so that is what decides. `kind`
    still rules out the things that are images only incidentally, like a
    thumbnail standing in for a video.
    """
    if artifact.kind not in {ArtifactKind.IMAGE.value, ArtifactKind.INPUT.value}:
        return False
    return (artifact.media_type or "").casefold().startswith("image/")


@router.get("/studio/capabilities", response_model=StudioCapabilityReport)
async def studio_capabilities(
    request: Request, session: ConversationSessionDep
) -> StudioCapabilityReport:
    """What the studio's tools can do on this machine, asked before the click.

    A tool that needs a workflow nobody has installed is answered here rather
    than at apply time, so a carefully drawn selection is never the thing that
    discovers the gap.
    """

    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    schemas = orchestrator.installed_edit_input_schemas(session)
    return StudioCapabilityReport(
        tools=[
            StudioToolCapability(
                kind=capability.kind,
                workflow_class=capability.workflow_class,
                available=capability.available,
                reason=capability.reason,
            )
            for capability in tool_capabilities(edit_input_schemas=schemas)
        ]
    )


@router.post("/studio/sessions", response_model=ChatDetail)
async def open_studio_session(
    payload: StudioSessionCreate, session: ConversationSessionDep
) -> Chat:
    """Find or create the hidden session behind the studio canvas.

    One session per source image: reopening resumes the same history, so
    the filmstrip is durable edit history rather than view state.
    """

    artifact = session.get(Artifact, payload.source_artifact_id)
    if not artifact:
        raise api_error(404, "artifact-not-found", "This media item no longer exists")
    if not _is_editable_image(artifact):
        raise api_error(422, "studio-image-only", "The studio edits images")
    existing = find_studio_session(session, artifact.id)
    if existing:
        return session.scalar(_studio_session_query(existing.id)) or existing
    source = session.get(Chat, payload.source_chat_id) if payload.source_chat_id else None
    if payload.source_chat_id and (not source or source.scope != STANDARD_CHAT_SCOPE):
        raise api_error(404, "chat-not-found", "The source chat no longer exists")
    studio = Chat(
        title=studio_session_title(artifact),
        archived=True,
        scope=STUDIO_SCOPE,
        routing_mode=RoutingMode.IMAGE.value,
        confirm_uncertain_media=False,
        origin_json={
            "source_artifact_id": artifact.id,
            **({"source_chat_id": source.id} if source else {}),
        },
        active_chat_profile_id=source.active_chat_profile_id if source else None,
        active_vision_profile_id=source.active_vision_profile_id if source else None,
        active_image_profile_id=source.active_image_profile_id if source else None,
        active_video_profile_id=source.active_video_profile_id if source else None,
        generation_settings_json=(copy.deepcopy(source.generation_settings_json) if source else {}),
        generation_preset_ids_json=(
            copy.deepcopy(source.generation_preset_ids_json) if source else {}
        ),
        vision_settings_json=copy.deepcopy(source.vision_settings_json) if source else {},
    )
    session.add(studio)
    session.flush()
    if source:
        copy_chat_workflow_selections(session, source, studio)
    else:
        mirror_legacy_chat_workflow_selections(session, studio)
    session.commit()
    return session.scalar(_studio_session_query(studio.id)) or studio


@router.get("/studio/sessions/{session_id}", response_model=ChatDetail)
async def get_studio_session(session_id: str, session: ConversationSessionDep) -> Chat:
    studio = session.scalar(_studio_session_query(session_id))
    if not studio:
        raise api_error(404, "studio-session-not-found", "This studio session no longer exists")
    return studio


def _studio_session_query(session_id: str) -> Select[tuple[Chat]]:
    return (
        select(Chat)
        .options(
            selectinload(Chat.messages)
            .selectinload(Message.parts)
            .selectinload(MessagePart.artifact),
            selectinload(Chat.messages)
            .selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.parts)
            .selectinload(ResponseRevisionPart.artifact),
            selectinload(Chat.messages).selectinload(Message.feedback_rows),
            # One query for the whole chat rather than one per message. Almost
            # every message names nothing, so this is usually an empty result
            # rather than a cost. Every loader that returns a ChatDetail needs
            # it, or the transcript falls back to a query per message.
            selectinload(Chat.messages).selectinload(Message.references),
            selectinload(Chat.messages)
            .selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.feedback_rows),
        )
        .where(Chat.id == session_id, Chat.scope == STUDIO_SCOPE)
    )


@router.post("/prompt-helpers", response_model=PromptHelperDetail, status_code=201)
async def create_prompt_helper(
    payload: PromptHelperCreate,
    request: Request,
    session: ConversationSessionDep,
) -> Chat:
    source = session.get(Chat, payload.source_chat_id)
    if not source or source.scope != STANDARD_CHAT_SCOPE:
        raise api_error(404, "chat-not-found", "source chat not found")
    generation_settings = copy.deepcopy(source.generation_settings_json)
    for role in (ModelRole.IMAGE.value, ModelRole.VIDEO.value):
        try:
            fields = await _engine_role_fields(request, role)
        except HTTPException as exc:
            if exc.status_code not in {409, 503}:
                raise
            continue
        preview_defaults = prompt_preview_settings(fields)
        if preview_defaults:
            generation_settings[role] = {
                **generation_settings.get(role, {}),
                **preview_defaults,
            }
    helper = Chat(
        title="Prompt workshop",
        archived=True,
        scope=PROMPT_HELPER_SCOPE,
        draft_prompt=payload.draft_prompt.strip(),
        routing_mode=RoutingMode.TEXT.value,
        confirm_uncertain_media=False,
        active_chat_profile_id=source.active_chat_profile_id,
        active_vision_profile_id=source.active_vision_profile_id,
        active_image_profile_id=source.active_image_profile_id,
        active_video_profile_id=source.active_video_profile_id,
        generation_settings_json=generation_settings,
        generation_preset_ids_json=copy.deepcopy(source.generation_preset_ids_json),
        vision_settings_json=copy.deepcopy(source.vision_settings_json),
    )
    session.add(helper)
    session.flush()
    copy_chat_workflow_selections(session, source, helper)
    session.commit()
    return session.scalar(_prompt_helper_query(helper.id)) or helper


@router.get("/prompt-helpers/{helper_id}", response_model=PromptHelperDetail)
async def get_prompt_helper(
    helper_id: str,
    session: ConversationSessionDep,
) -> Chat:
    helper = session.scalar(_prompt_helper_query(helper_id))
    if not helper:
        raise api_error(404, "prompt-helper-not-found", "prompt helper not found")
    return helper


@router.patch("/prompt-helpers/{helper_id}", response_model=PromptHelperDetail)
async def update_prompt_helper(
    helper_id: str,
    payload: PromptHelperUpdate,
    session: ConversationSessionDep,
) -> Chat:
    helper = session.scalar(_prompt_helper_query(helper_id))
    if not helper:
        raise api_error(404, "prompt-helper-not-found", "prompt helper not found")
    helper.draft_prompt = payload.draft_prompt.strip()
    session.commit()
    return session.scalar(_prompt_helper_query(helper_id)) or helper


@router.delete("/prompt-helpers/{helper_id}", status_code=204)
async def delete_prompt_helper(
    helper_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Response:
    helper = session.get(Chat, helper_id)
    if not helper or helper.scope != PROMPT_HELPER_SCOPE:
        raise api_error(404, "prompt-helper-not-found", "prompt helper not found")
    services = _services(request)
    async with services.orchestrator.prepare_chat_deletion(helper_id):
        session.expire_all()
        helper = session.get(Chat, helper_id)
        if not helper or helper.scope != PROMPT_HELPER_SCOPE:
            raise api_error(404, "prompt-helper-not-found", "prompt helper not found")
        artifact_ids = services.artifacts.generated_media_artifact_ids_for_chat(session, helper_id)
        helper_jobs = session.scalars(
            select(Job).join(Run, Job.run_id == Run.id).where(Run.chat_id == helper_id)
        ).all()
        for job in helper_jobs:
            session.delete(job)
        session.delete(helper)
        session.flush()
        services.artifacts.delete_generated_media_artifacts(session, artifact_ids)
        session.commit()
    return Response(status_code=204)


@router.patch("/chats/{chat_id}", response_model=ChatOut)
async def update_chat(
    chat_id: str,
    payload: ChatUpdate,
    request: Request,
    session: ConversationSessionDep,
) -> Chat:
    chat = session.get(Chat, chat_id)
    if not chat or chat.scope != STANDARD_CHAT_SCOPE:
        raise api_error(404, "chat-not-found", "chat not found")
    values = payload.model_dump(exclude_unset=True, mode="json")
    await _validate_generation_defaults(request, session, values)
    if (
        "project_id" in values
        and values["project_id"]
        and not session.get(Project, values["project_id"])
    ):
        raise api_error(404, "project-not-found", "project not found")
    profile_fields = {
        "active_chat_profile_id": ModelRole.CHAT.value,
        "active_vision_profile_id": ModelRole.CHAT.value,
        "active_image_profile_id": ModelRole.IMAGE.value,
        "active_video_profile_id": ModelRole.VIDEO.value,
    }
    for field, role in profile_fields.items():
        profile_id = values.get(field)
        if profile_id in (None, AUTO_PROFILE_ID):
            continue
        profile = session.get(ModelProfile, profile_id)
        if not profile:
            raise api_error(404, "profile-not-found", f"{role} profile not found")
        if profile.role != role:
            raise api_error(422, "profile-required", f"{field} requires a {role} profile")
        _validated_profile_install(
            session,
            model_install_id=profile.model_install_id,
            role=profile.role,
            engine=profile.engine,
        )
        if field == "active_vision_profile_id":
            install = session.get(ModelInstall, profile.model_install_id)
            evidence = (
                current_capability_evidence(
                    session,
                    install,
                    _services(request).settings,
                    _services(request).runtimes,
                )
                if install
                else None
            )
            if "image" not in evidence_input_modalities(evidence):
                raise api_error(
                    422,
                    "vision-profile-not-verified",
                    "active_vision_profile_id requires a runtime-verified vision profile",
                )
    for key, value in values.items():
        setattr(chat, key, value)
    changed_capabilities = [
        cast(
            Literal["chat", "vision", "image", "video"],
            field.removeprefix("active_").removesuffix("_profile_id"),
        )
        for field in profile_fields
        if field in values
    ]
    if changed_capabilities:
        mirror_legacy_chat_workflow_selections(session, chat, changed_capabilities)
    session.commit()
    session.refresh(chat)
    return chat


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: str,
    request: Request,
    session: ConversationSessionDep,
    delete_generated_media: bool = Query(False),
) -> Response:
    chat = session.get(Chat, chat_id)
    if not chat or chat.scope != STANDARD_CHAT_SCOPE:
        raise api_error(404, "chat-not-found", "chat not found")
    services = _services(request)
    async with services.orchestrator.prepare_chat_deletion(chat_id):
        session.expire_all()
        chat = session.get(Chat, chat_id)
        if not chat or chat.scope != STANDARD_CHAT_SCOPE:
            raise api_error(404, "chat-not-found", "chat not found")
        artifact_ids = (
            services.artifacts.generated_media_artifact_ids_for_chat(session, chat_id)
            if delete_generated_media
            else ()
        )
        session.delete(chat)
        session.flush()
        services.artifacts.delete_generated_media_artifacts(session, artifact_ids)
        session.commit()
    return Response(status_code=204)


@router.post("/chats/{chat_id}/turns", response_model=TurnAccepted, status_code=202)
async def create_turn(
    chat_id: str, payload: TurnRequest, request: Request, session: ConversationSessionDep
) -> TurnAccepted:
    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    return await _accept_turn(orchestrator, session, chat_id, payload)


async def _accept_turn(
    orchestrator: ConversationOrchestrator,
    session: Session,
    chat_id: str,
    payload: TurnRequest,
    *,
    use_explicit_parent: bool = False,
    replacement_message_id: str | None = None,
    source_action: str = "send",
    inherited_image_edit_strength: dict[str, Any] | None = None,
    reference_source_message_id: str | None = None,
) -> TurnAccepted:
    try:
        return await orchestrator.create_turn(
            session,
            chat_id,
            payload,
            use_explicit_parent=use_explicit_parent,
            replacement_message_id=replacement_message_id,
            source_action=source_action,
            inherited_image_edit_strength=inherited_image_edit_strength,
            reference_source_message_id=reference_source_message_id,
        )
    except LookupError as exc:
        raise api_error(404, "turn-subject-not-found", str(exc)) from exc
    except RouteConfirmationRequired as exc:
        raise HTTPException(
            409,
            detail={
                "code": "route_confirmation_required",
                "message": str(exc),
                "plan": exc.plan.model_dump(mode="json"),
                # Top level, matching the ordered-plan 409 below, so a client
                # reads one shape rather than digging into the plan.
                "estimate": exc.plan.generation_estimate,
            },
        ) from exc
    except OrderedPlanConfirmationRequired as exc:
        raise HTTPException(
            409,
            detail={
                "code": "ordered_plan_confirmation_required",
                "message": str(exc),
                "plan": exc.intent.model_dump(mode="json"),
                "estimate": exc.estimate,
            },
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            409,
            detail={
                "code": "idempotency_request_in_progress",
                "message": str(exc),
            },
        ) from exc
    except ProjectWorkflowPinInvalid as exc:
        # Named rather than silently replaced: the caller can update or remove
        # the pin, which is the only thing that actually resolves this.
        raise HTTPException(
            409,
            detail={
                "code": "project_workflow_pin_invalid",
                "message": str(exc),
                "project_id": exc.project_id,
                "workflow_revision_id": exc.revision_id,
                "role": exc.role,
                "reason": exc.reason,
                "actions": ["update_project_workflow_pin", "remove_project_workflow_pin"],
            },
        ) from exc
    except ResponseRevisionConflict as exc:
        raise api_error(409, "response-revision-conflict", str(exc)) from exc
    except EngineNotConfiguredError as exc:
        raise api_error(409, "engine-not-configured", str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise api_error(503, "engine-schema-unavailable", str(exc)) from exc
    except ReferenceNotFoundError as exc:
        raise api_error(404, "reference-not-found", str(exc)) from exc
    except ValueError as exc:
        raise api_error(422, "turn-invalid", str(exc)) from exc


@router.get("/messages/{message_id}", response_model=MessageOut)
async def get_message(message_id: str, session: ConversationSessionDep) -> Message:
    message = session.scalar(
        select(Message)
        .options(
            selectinload(Message.parts).selectinload(MessagePart.artifact),
            selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.parts)
            .selectinload(ResponseRevisionPart.artifact),
        )
        .where(Message.id == message_id)
    )
    if not message:
        raise api_error(404, "message-not-found", "message not found")
    return message


@router.post("/messages/{message_id}/fork", response_model=ChatOut, status_code=201)
async def fork_thread_from_message(message_id: str, session: ConversationSessionDep) -> Chat:
    """Start a new chat carrying the history up to this message."""

    try:
        fork = fork_chat_from_message(session, message_id)
    except ForkSourceNotFound as exc:
        raise api_error(404, "fork-source-not-found", str(exc)) from exc
    session.commit()
    created = session.get(Chat, fork.chat_id)
    if not created:  # pragma: no cover - the row was just committed
        raise api_error(500, "fork-unavailable", "the forked chat could not be read back")
    return created


@router.delete("/messages/{message_id}/exchange", response_model=ExchangeDeletionOut)
async def delete_message_exchange(
    message_id: str, session: ConversationSessionDep
) -> ExchangeDeletionOut:
    """Delete one user turn, its answer, and everything the exchange produced.

    Refusals mirror the service's two product decisions: a turn with later
    replies must have them deleted first (`exchange-has-replies`), and a turn
    with live jobs is history-in-progress, not history
    (`exchange-busy`).
    """

    try:
        result = delete_exchange(session, message_id)
    except ExchangeHasReplies as exc:
        raise api_error(409, "exchange-has-replies", str(exc), reply_count=exc.reply_count) from exc
    except ExchangeBusy as exc:
        raise api_error(409, "exchange-busy", str(exc), job_count=exc.job_count) from exc
    except ExchangeNotFound as exc:
        raise api_error(404, "exchange-not-found", str(exc)) from exc
    session.commit()
    return ExchangeDeletionOut.model_validate(result)


def _inherited_auto_image_edit_strength(run: Run) -> dict[str, Any] | None:
    image_edit = run.provenance_json.get("image_edit")
    if not isinstance(image_edit, dict):
        return None
    strength = image_edit.get("strength")
    if not isinstance(strength, dict) or strength.get("mode") != "auto":
        return None
    value = strength.get("value")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return strength


@router.post("/messages/{message_id}/regenerate", response_model=TurnAccepted, status_code=202)
async def regenerate_message(
    message_id: str,
    payload: RegenerateRequest,
    request: Request,
    session: ConversationSessionDep,
) -> TurnAccepted:
    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    source_assistant = session.get(Message, message_id)
    if (
        not source_assistant
        or not source_assistant.transcript_visible
        or source_assistant.status != MessageStatus.COMPLETE.value
    ):
        raise api_error(
            409, "response-not-regenerable", "only a completed visible response can be regenerated"
        )
    pending_revision = session.scalar(
        select(ResponseRevision.id).where(
            ResponseRevision.message_id == message_id,
            ResponseRevision.status == MessageStatus.PENDING.value,
        )
    )
    if pending_revision:
        raise api_error(
            409, "response-regeneration-in-progress", "this response is already being regenerated"
        )
    active_revision = (
        session.get(ResponseRevision, source_assistant.active_response_revision_id)
        if source_assistant.active_response_revision_id
        else None
    )
    prior_run = (
        session.get(Run, active_revision.run_id)
        if active_revision and active_revision.run_id
        else None
    )
    if not prior_run:
        prior_run = session.scalar(select(Run).where(Run.assistant_message_id == message_id))
    if not prior_run:
        raise api_error(404, "assistant-run-not-found", "assistant run not found")
    user_message = session.scalar(
        select(Message)
        .options(selectinload(Message.parts))
        .where(Message.id == prior_run.user_message_id)
    )
    if not user_message:
        raise api_error(404, "source-user-message-not-found", "source user message not found")
    text = "\n".join(part.text for part in user_message.parts if part.text).strip()
    mode = _mode_for_operation(Operation(prior_run.operation))
    prior_revision = (
        session.get(WorkflowRevision, prior_run.workflow_revision_id)
        if prior_run.workflow_revision_id
        else None
    )
    prior_profile = (
        session.get(ModelProfile, prior_run.profile_id) if prior_run.profile_id else None
    )
    try:
        prior_settings = await orchestrator.request_settings_for_operation(
            Operation(prior_run.operation),
            prior_run.settings_json,
            input_schema=prior_revision.input_schema_json if prior_revision else None,
            engine=prior_profile.engine if prior_profile else None,
        )
    except EngineNotConfiguredError as exc:
        raise api_error(409, "engine-not-configured", str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise api_error(503, "engine-schema-unavailable", str(exc)) from exc
    except ValueError as exc:
        raise api_error(422, "generation-settings-invalid", str(exc)) from exc
    turn = TurnRequest(
        text=text,
        mode=mode,
        parent_message_id=user_message.parent_id,
        input_artifact_ids=orchestrator.input_artifact_ids_for_run(session, prior_run),
        settings={**prior_settings, **payload.settings},
    )
    prior_strength = _inherited_auto_image_edit_strength(prior_run)
    inherited_parameter = (
        prior_strength.get("parameter")
        if prior_strength and isinstance(prior_strength.get("parameter"), str)
        else "denoise"
    )
    inherited_image_edit_strength = (
        None if inherited_parameter in payload.settings else prior_strength
    )
    return await _accept_turn(
        orchestrator,
        session,
        prior_run.chat_id,
        turn,
        use_explicit_parent=True,
        replacement_message_id=message_id,
        source_action="regenerate",
        inherited_image_edit_strength=inherited_image_edit_strength,
        reference_source_message_id=user_message.id,
    )


@router.post(
    "/messages/{message_id}/revisions/{revision_id}/select",
    response_model=MessageOut,
)
async def select_response_revision(
    message_id: str,
    revision_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Message:
    try:
        return _services(request).orchestrator.select_response_revision(
            session,
            message_id,
            revision_id,
        )
    except LookupError as exc:
        raise api_error(404, "response-revision-not-found", str(exc)) from exc
    except ValueError as exc:
        raise api_error(409, "response-revision-not-selectable", str(exc)) from exc


@router.post("/messages/{message_id}/branch", response_model=TurnAccepted, status_code=202)
async def edit_and_branch(
    message_id: str,
    payload: TurnRequest,
    request: Request,
    session: ConversationSessionDep,
) -> TurnAccepted:
    source = session.get(Message, message_id)
    if not source or source.role != MessageRole.USER.value:
        raise api_error(404, "user-message-not-found", "user message not found")
    prior_run = session.scalar(select(Run).where(Run.user_message_id == source.id))
    updates: dict[str, Any] = {"parent_message_id": source.parent_id}
    inherited_image_edit_strength: dict[str, Any] | None = None
    if prior_run:
        prior_mode = _mode_for_operation(Operation(prior_run.operation))
        mode_was_supplied = "mode" in payload.model_fields_set and payload.mode is not None
        target_mode = payload.mode if mode_was_supplied else prior_mode
        updates["mode"] = target_mode
        same_explicit_role = mode_was_supplied and target_mode == prior_mode
        legacy_inheritance = not mode_was_supplied
        inherit_inputs = (legacy_inheritance and not payload.input_artifact_ids) or (
            same_explicit_role and "input_artifact_ids" not in payload.model_fields_set
        )
        inherit_settings = (legacy_inheritance and not payload.settings) or (
            same_explicit_role and "settings" not in payload.model_fields_set
        )
        if inherit_inputs:
            updates["input_artifact_ids"] = _services(
                request
            ).orchestrator.input_artifact_ids_for_run(session, prior_run)
        if inherit_settings:
            prior_revision = (
                session.get(WorkflowRevision, prior_run.workflow_revision_id)
                if prior_run.workflow_revision_id
                else None
            )
            prior_profile = (
                session.get(ModelProfile, prior_run.profile_id) if prior_run.profile_id else None
            )
            try:
                updates["settings"] = await _services(
                    request
                ).orchestrator.request_settings_for_operation(
                    Operation(prior_run.operation),
                    prior_run.settings_json,
                    input_schema=(prior_revision.input_schema_json if prior_revision else None),
                    engine=prior_profile.engine if prior_profile else None,
                )
                inherited_image_edit_strength = _inherited_auto_image_edit_strength(prior_run)
            except EngineNotConfiguredError as exc:
                raise api_error(409, "engine-not-configured", str(exc)) from exc
            except EngineSchemaUnavailableError as exc:
                raise api_error(503, "engine-schema-unavailable", str(exc)) from exc
            except ValueError as exc:
                raise api_error(422, "generation-settings-invalid", str(exc)) from exc
    turn = payload.model_copy(update=updates)
    reference_source_message_id = (
        source.id if "references" not in payload.model_fields_set else None
    )
    return await _accept_turn(
        _services(request).orchestrator,
        session,
        source.chat_id,
        turn,
        use_explicit_parent=True,
        source_action="edit_and_branch",
        inherited_image_edit_strength=inherited_image_edit_strength,
        reference_source_message_id=reference_source_message_id,
    )


def _mode_for_operation(operation: Operation) -> RoutingMode:
    if operation == Operation.TEXT:
        return RoutingMode.TEXT
    if "video" in operation.value:
        return RoutingMode.VIDEO
    return RoutingMode.IMAGE


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, session: ConversationSessionDep) -> Run:
    run = session.get(Run, run_id)
    if not run:
        raise api_error(404, "run-not-found", "run not found")
    return run


@router.get("/work-plans", response_model=list[WorkPlanOut])
async def list_work_plans(
    session: ConversationSessionDep,
    chat_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[WorkPlan]:
    statement = (
        select(WorkPlan)
        .options(selectinload(WorkPlan.steps))
        .order_by(WorkPlan.created_at.desc(), WorkPlan.id.desc())
        .limit(limit)
    )
    if chat_id:
        statement = statement.where(WorkPlan.chat_id == chat_id)
    return list(session.scalars(statement).unique().all())


@router.get("/work-plans/{plan_id}", response_model=WorkPlanOut)
async def get_work_plan(plan_id: str, session: ConversationSessionDep) -> WorkPlan:
    plan = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not plan:
        raise api_error(404, "work-plan-not-found", "work plan not found")
    return plan


@router.get("/work-steps/{step_id}", response_model=WorkStepOut)
async def get_work_step(step_id: str, session: ConversationSessionDep) -> WorkStep:
    step = session.get(WorkStep, step_id)
    if not step:
        raise api_error(404, "work-step-not-found", "work step not found")
    return step


@router.post("/work-plans/{plan_id}/cancel", response_model=WorkPlanOut)
async def cancel_work_plan(
    plan_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> WorkPlan:
    plan = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not plan:
        raise api_error(404, "work-plan-not-found", "work plan not found")
    jobs = list(
        session.scalars(
            select(Job)
            .where(
                Job.work_plan_id == plan_id,
                Job.status.in_(["queued", "running", "paused"]),
            )
            .order_by(Job.created_at, Job.id)
        ).all()
    )
    if not jobs:
        raise api_error(409, "work-plan-not-cancellable", "work plan has no cancellable steps")
    for job in jobs:
        await _services(request).orchestrator.cancel(job.id)
    session.expire_all()
    refreshed = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not refreshed:
        raise api_error(404, "work-plan-not-found", "work plan not found")
    return refreshed


@router.post("/work-steps/{step_id}/cancel", response_model=JobOut)
async def cancel_work_step(
    step_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job | JobOut:
    job = session.scalar(select(Job).where(Job.work_step_id == step_id))
    if not job:
        raise api_error(404, "work-step-job-not-found", "work step job not found")
    return await cancel_job(job.id, request, session)


@router.post("/work-plans/{plan_id}/retry", response_model=WorkPlanOut)
async def retry_work_plan(
    plan_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> WorkPlan:
    plan = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not plan:
        raise api_error(404, "work-plan-not-found", "work plan not found")
    jobs = list(
        session.scalars(
            select(Job)
            .where(
                Job.work_plan_id == plan_id,
                Job.status.in_(["failed", "cancelled", "interrupted"]),
            )
            .order_by(Job.created_at, Job.id)
        ).all()
    )
    if not jobs:
        raise api_error(409, "work-plan-not-retryable", "work plan has no retryable steps")
    for job in jobs:
        await retry_job(job.id, request, session)
    session.expire_all()
    refreshed = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not refreshed:
        raise api_error(404, "work-plan-not-found", "work plan not found")
    return refreshed


@router.post("/work-steps/{step_id}/retry", response_model=JobOut)
async def retry_work_step(
    step_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    job = session.scalar(select(Job).where(Job.work_step_id == step_id))
    if not job:
        raise api_error(404, "work-step-job-not-found", "work step job not found")
    return await retry_job(job.id, request, session)


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(
    session: ConversationSessionDep,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[Job]:
    statement = (
        select(Job)
        .where(Job.kind != JobKind.EDIT_VERIFY.value)
        .order_by(Job.created_at.desc())
        .limit(limit)
    )
    if status:
        statement = statement.where(Job.status == status)
    return list(session.scalars(statement).all())


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
async def cancel_job(
    job_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job | JobOut:
    job = session.get(Job, job_id)
    if not job:
        raise api_error(404, "job-not-found", "job not found")
    verification_snapshot = (
        JobOut.model_validate(job)
        if isinstance(job.payload_json, dict)
        and isinstance(job.payload_json.get("setup_verification_id"), str)
        else None
    )
    if job.kind == JobKind.DOWNLOAD.value:
        changed = await _services(request).downloads.cancel(job_id)
    elif job.kind == JobKind.REGISTRY_PREPARE.value:
        changed = await _cancel_registry_preparation(job_id)
    else:
        changed = await _services(request).orchestrator.cancel(job_id)
    if not changed:
        raise api_error(
            409, "job-not-cancellable", "job is already terminal or cannot be cancelled"
        )
    session.expire_all()
    refreshed = session.get(Job, job_id)
    if not refreshed:
        if verification_snapshot:
            progress = {
                **verification_snapshot.progress_json,
                "stage": "cancelled",
                "indeterminate": True,
                "updated_at": utcnow().isoformat(),
            }
            return verification_snapshot.model_copy(
                update={
                    "status": "cancelled",
                    "phase": "cancelled",
                    "progress_json": progress,
                    "payload_json": {},
                    "result_json": {},
                    "error": None,
                    "cancellable": False,
                    "completed_at": utcnow(),
                }
            )
        raise api_error(404, "job-not-found", "job not found")
    return refreshed


@router.post("/chats/{chat_id}/classify-draft", response_model=DraftClassification)
async def classify_chat_draft(
    chat_id: str,
    payload: DraftClassificationRequest,
    request: Request,
    session: ConversationSessionDep,
) -> DraftClassification:
    """Classify an unsent draft so the composer does not have to guess.

    The browser previously kept its own copy of the router's patterns to decide
    which workflow schema to show and whether edit strength applied. That copy
    drifted every time the router learned a new phrasing, so the composer showed
    the wrong controls for exactly the wording the server handled correctly.
    """
    chat = session.get(Chat, chat_id)
    if not chat:
        raise api_error(404, "chat-not-found", "chat not found")
    return DraftClassification(
        references_prior_visual=_services(request).orchestrator.classify_draft(
            session,
            chat,
            text=payload.text,
            mode=payload.mode,
            parent_message_id=payload.parent_message_id,
        )
    )


@router.post("/chats/{chat_id}/cancel", response_model=JobOut)
async def cancel_active_chat_run(
    chat_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    if not session.get(Chat, chat_id):
        raise api_error(404, "chat-not-found", "chat not found")
    if not _current_chat_job(session, chat_id):
        raise api_error(409, "chat-run-absent", "chat has no cancellable run")
    refreshed = await _cancel_current_chat_work(request, session, chat_id)
    if not refreshed:
        raise api_error(
            409, "chat-run-not-cancellable", "chat run is already terminal or cannot be cancelled"
        )
    return refreshed


@router.post(
    "/chats/{chat_id}/stop-and-send",
    response_model=TurnAccepted,
    status_code=202,
)
async def stop_and_send_turn(
    chat_id: str,
    payload: TurnRequest,
    request: Request,
    session: ConversationSessionDep,
) -> TurnAccepted:
    if not session.get(Chat, chat_id):
        raise api_error(404, "chat-not-found", "chat not found")
    await _cancel_current_chat_work(request, session, chat_id)
    return await _accept_turn(
        _services(request).orchestrator,
        session,
        chat_id,
        payload,
        source_action="stop_and_send",
    )


async def _cancel_current_chat_work(
    request: Request,
    session: Session,
    chat_id: str,
) -> Job | None:
    """Cancel the chat's active work, including every sibling output.

    One turn can produce several outputs and they run one at a time, so
    cancelling only the running job lets the next one start immediately.
    Stopping has to stop the whole turn.
    """

    job = _current_chat_job(session, chat_id)
    if not job:
        return None
    targets = [job]
    if job.work_plan_id:
        siblings = list(
            session.scalars(
                select(Job)
                .where(
                    Job.work_plan_id == job.work_plan_id,
                    Job.status.in_(["queued", "running", "paused"]),
                    Job.cancellable.is_(True),
                )
                .order_by(Job.created_at, Job.id)
            ).all()
        )
        targets = siblings or [job]
    orchestrator = _services(request).orchestrator
    cancelled = False
    for target in targets:
        if await orchestrator.cancel(target.id):
            cancelled = True
    if not cancelled:
        return None
    session.expire_all()
    return session.get(Job, job.id)


def _current_chat_job(session: Session, chat_id: str) -> Job | None:
    running = session.scalar(
        select(Job)
        .join(Run, Job.run_id == Run.id)
        .where(
            Run.chat_id == chat_id,
            Job.status == "running",
            Job.cancellable.is_(True),
        )
        .order_by(Job.started_at, Job.created_at, Job.id)
        .limit(1)
    )
    if running:
        return running
    return session.scalar(
        select(Job)
        .join(Run, Job.run_id == Run.id)
        .where(
            Run.chat_id == chat_id,
            Job.status.in_(["queued", "paused"]),
            Job.cancellable.is_(True),
        )
        .order_by(Job.created_at, Job.id)
        .limit(1)
    )


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
async def retry_job(
    job_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise api_error(404, "job-not-found", "job not found")
    if job.status not in {"failed", "cancelled", "interrupted"}:
        raise api_error(
            409, "job-not-retryable-state", "only terminal unsuccessful jobs can be retried"
        )
    if job.kind == JobKind.DOWNLOAD.value:
        job.status = "queued"
        job.progress = 0
        job.error = None
        job.started_at = None
        job.completed_at = None
        job.enqueued_at = utcnow()
        job.claim_owner = None
        job.claim_expires_at = None
        job.heartbeat_at = None
        update_job_progress(
            job,
            stage="retry queued",
            queue_resource=job.queue_resource or "network",
            indeterminate=True,
        )
        session.commit()
        _services(request).downloads.start(job.id)
        session.refresh(job)
        return job
    if not job.run_id:
        raise api_error(422, "job-not-retryable", "job has no retryable operation")
    run = session.get(Run, job.run_id)
    if not run:
        raise api_error(422, "job-not-retryable", "job has no retryable operation")

    orchestrator = _services(request).orchestrator
    async with orchestrator.chat_guard(run.chat_id):
        session.expire_all()
        job = session.get(Job, job_id)
        if not job:
            raise api_error(404, "job-not-found", "job not found")
        if job.status not in {"failed", "cancelled", "interrupted"}:
            raise api_error(
                409, "job-not-retryable-state", "only terminal unsuccessful jobs can be retried"
            )
        if not job.run_id:
            raise api_error(422, "job-not-retryable", "job has no retryable operation")
        run = session.get(Run, job.run_id)
        if not run:
            raise api_error(422, "job-not-retryable", "job has no retryable operation")
        job.status = "queued"
        job.progress = 0
        job.error = None
        job.started_at = None
        job.completed_at = None
        job.enqueued_at = utcnow()
        job.claim_owner = None
        job.claim_expires_at = None
        job.heartbeat_at = None
        update_job_progress(
            job,
            stage="retry queued",
            queue_resource=job.queue_resource or "interactive_compute",
            indeterminate=True,
        )
        run.status = "queued"
        run.error = None
        run.completed_at = None
        try:
            orchestrator.prepare_retry(session, run)
        except LookupError as exc:
            raise api_error(422, "job-not-retryable", str(exc)) from exc
        session.commit()
        orchestrator.start(job.id, run.id)
        session.refresh(job)
        return job


@router.post("/artifacts", response_model=ArtifactOut, status_code=201)
async def upload_artifact(
    request: Request,
    session: ConversationSessionDep,
    file: Annotated[UploadFile, File()],
    kind: ArtifactKind = ArtifactKind.INPUT,
) -> ArtifactOut:
    services = _services(request)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > services.settings.max_upload_bytes:
        raise api_error(413, "upload-too-large", "upload exceeds configured limit")
    artifact = services.artifacts.ingest_stream(
        session,
        file.file,
        kind=kind,
        media_type=file.content_type,
        original_name=file.filename,
        metadata={"uploaded": True},
    )
    ensure_library_entry(session, artifact)
    session.commit()
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


def _captured_generation_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    return name if name and len(name) <= 500 else None


def _generation_identity(provenance: object) -> GenerationIdentityOut | None:
    if not isinstance(provenance, dict):
        return None
    model = provenance.get("model")
    workflow = provenance.get("workflow")
    model_name = (
        _captured_generation_name(model.get("profile_name")) if isinstance(model, dict) else None
    )
    family_name = (
        _captured_generation_name(workflow.get("family_name"))
        if isinstance(workflow, dict)
        else None
    )
    definition_name = (
        _captured_generation_name(workflow.get("definition_name"))
        if isinstance(workflow, dict)
        else None
    )
    raw_version = workflow.get("version") if isinstance(workflow, dict) else None
    version = (
        raw_version
        if isinstance(raw_version, int)
        and not isinstance(raw_version, bool)
        and 0 < raw_version <= 2_147_483_647
        and bool(family_name or definition_name)
        else None
    )
    if not any((model_name, family_name, definition_name)):
        return None
    return GenerationIdentityOut(
        model_profile_name=model_name,
        workflow_family_name=family_name,
        workflow_definition_name=definition_name,
        workflow_version=version,
    )


def _artifact_generation_identity(
    session: Session,
    artifact: Artifact,
) -> GenerationIdentityOut | None:
    run_id = artifact.metadata_json.get("run_id")
    run = session.get(Run, run_id) if isinstance(run_id, str) else None
    return _generation_identity(run.provenance_json) if run is not None else None


@router.get("/artifact-library", response_model=ArtifactLibraryPage)
async def list_artifact_library(
    request: Request,
    session: ConversationSessionDep,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, min_length=1, max_length=2_048),
    kind: Literal["image", "video"] | None = None,
    state: Literal["visible", "trashed"] = "visible",
    favorite: Literal["true", "false"] | None = None,
    query: str = Query(default="", max_length=200),
) -> ArtifactLibraryPage:
    """Return one bounded page of durable Media Library memberships."""

    favorite_value = None if favorite is None else favorite == "true"
    try:
        rows, next_cursor = list_library_entries(
            session,
            signing_key=_services(request).security.local_state_signing_key(b"artifact-library"),
            limit=limit,
            cursor=cursor,
            kind=kind,
            state=state,
            favorite=favorite_value,
            query=query,
        )
    except ArtifactLibraryCursorError as exc:
        raise api_error(
            422,
            "artifact-library-cursor-invalid",
            "The Media Library page request is invalid. Start again from the first page.",
        ) from exc
    except ArtifactLibraryDataError as exc:
        raise api_error(
            409,
            "artifact-library-conflict",
            "The Media Library could not be read safely. Refresh and try again.",
        ) from exc
    return ArtifactLibraryPage(
        items=[
            ArtifactLibraryEntrySummary(
                id=row.entry.id,
                artifact_id=row.artifact.id,
                version=row.entry.version,
                state=cast(Literal["visible", "trashed"], row.entry.state),
                display_name=row.entry.display_name,
                favorite=row.entry.favorite,
                kind=cast(Literal["image", "video"], row.artifact.kind),
                media_type=row.artifact.media_type,
                size_bytes=row.artifact.size_bytes,
                created_at=row.entry.created_at,
                updated_at=row.entry.updated_at,
            )
            for row in rows
        ],
        next_cursor=next_cursor,
    )


@router.get("/artifacts", response_model=list[ArtifactLibraryItem])
async def list_artifacts(
    session: ConversationSessionDep,
    kind: Literal["image", "video"] | None = None,
    chat_id: str | None = None,
    project_id: str | None = None,
    favorites: bool = False,
    query: str = Query(default="", max_length=200),
) -> list[ArtifactLibraryItem]:
    reference_rows = session.execute(
        select(MessagePart.artifact_id, Message.chat_id, Chat.project_id)
        .join(Message, Message.id == MessagePart.message_id)
        .join(Chat, Chat.id == Message.chat_id)
        .where(MessagePart.artifact_id.is_not(None))
    ).all()
    references: dict[str, list[tuple[str, str | None]]] = {}
    for artifact_id, referenced_chat_id, referenced_project_id in reference_rows:
        references.setdefault(artifact_id, []).append((referenced_chat_id, referenced_project_id))
    statement = select(Artifact).where(
        Artifact.kind.in_([ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value])
    )
    if kind:
        statement = statement.where(Artifact.kind == kind)
    if favorites:
        statement = statement.where(Artifact.favorite.is_(True))
    normalized_query = query.strip().lower()
    if normalized_query:
        statement = statement.where(
            func.lower(func.coalesce(Artifact.original_name, Artifact.sha256)).contains(
                normalized_query
            )
        )
    artifacts = session.scalars(statement.order_by(Artifact.created_at.desc())).all()
    run_ids = {
        run_id
        for artifact in artifacts
        if isinstance((run_id := artifact.metadata_json.get("run_id")), str)
    }
    runs_by_id: dict[str, Run] = {}
    ordered_run_ids = sorted(run_ids)
    for offset in range(0, len(ordered_run_ids), 400):
        batch = ordered_run_ids[offset : offset + 400]
        runs_by_id.update(
            (run.id, run) for run in session.scalars(select(Run).where(Run.id.in_(batch))).all()
        )
    results: list[ArtifactLibraryItem] = []
    for artifact in artifacts:
        artifact_references = references.get(artifact.id, [])
        chat_ids = sorted({item[0] for item in artifact_references})
        project_ids = sorted({item[1] for item in artifact_references if item[1]})
        if chat_id and chat_id not in chat_ids:
            continue
        if project_id and project_id not in project_ids:
            continue
        result = ArtifactLibraryItem.model_validate(artifact)
        result.reference_count = len(artifact_references)
        result.chat_ids = chat_ids
        result.project_ids = project_ids
        run_id = artifact.metadata_json.get("run_id")
        run = runs_by_id.get(run_id) if isinstance(run_id, str) else None
        result.generation_identity = (
            _generation_identity(run.provenance_json) if run is not None else None
        )
        result.url = f"/api/artifacts/{artifact.id}/content"
        results.append(result)
    return results


@router.patch("/artifacts/{artifact_id}", response_model=ArtifactOut)
async def update_artifact(
    artifact_id: str, payload: ArtifactUpdate, session: ConversationSessionDep
) -> ArtifactOut:
    """Set the favorite flag; it pins against automatic cleanup only."""

    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise api_error(404, "artifact-not-found", "This media item no longer exists")
    try:
        set_library_favorite(session, artifact, payload.favorite)
    except ArtifactLibraryConflict as exc:
        session.rollback()
        raise api_error(
            409,
            "artifact-library-conflict",
            "The Media Library item changed. Refresh and try again.",
        ) from exc
    session.commit()
    session.refresh(artifact)
    result = ArtifactOut.model_validate(artifact)
    result.generation_identity = _artifact_generation_identity(session, artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/artifacts/storage", response_model=ArtifactStorageInfo)
async def artifact_storage(
    request: Request,
    session: ConversationSessionDep,
) -> ArtifactStorageInfo:
    services = _services(request)
    artifacts = session.scalars(select(Artifact)).all()
    referenced = services.artifacts.referenced_artifact_ids(session)
    retention = services.artifacts.cleanup_retention(
        session,
        retention_days=services.settings.artifact_retention_days,
        temporary_hours=services.settings.temporary_retention_hours,
        dry_run=True,
    )
    temporary = [
        artifact
        for artifact in artifacts
        if artifact.metadata_json.get("temporary_preview")
        or artifact.metadata_json.get("intermediate")
    ]
    referenced_artifacts = [artifact for artifact in artifacts if artifact.id in referenced]
    disk_free = shutil.disk_usage(services.settings.data_dir).free
    total_bytes = sum(artifact.size_bytes for artifact in artifacts)
    referenced_bytes = sum(artifact.size_bytes for artifact in referenced_artifacts)
    return ArtifactStorageInfo(
        total_bytes=total_bytes,
        total_count=len(artifacts),
        referenced_bytes=referenced_bytes,
        referenced_count=len(referenced_artifacts),
        unreferenced_bytes=total_bytes - referenced_bytes,
        unreferenced_count=len(artifacts) - len(referenced_artifacts),
        temporary_bytes=sum(artifact.size_bytes for artifact in temporary),
        temporary_count=len(temporary),
        eligible_bytes=retention.reclaimed_bytes,
        eligible_count=retention.removed_count,
        retention_pending_count=retention.pending_count,
        disk_free_bytes=disk_free,
        warning=disk_free < services.settings.storage_warning_free_bytes,
        retention_days=services.settings.artifact_retention_days,
        temporary_retention_hours=services.settings.temporary_retention_hours,
    )


@router.post("/artifacts/cleanup", response_model=ArtifactCleanupResult)
async def cleanup_artifacts(
    payload: ArtifactCleanupRequest,
    request: Request,
    session: ConversationSessionDep,
) -> ArtifactCleanupResult:
    services = _services(request)
    cleanup = services.artifacts.cleanup_retention(
        session,
        retention_days=services.settings.artifact_retention_days,
        temporary_hours=services.settings.temporary_retention_hours,
        dry_run=payload.dry_run,
    )
    if not payload.dry_run:
        session.commit()
    return ArtifactCleanupResult(
        dry_run=payload.dry_run,
        marked_count=cleanup.marked_count,
        retention_pending_count=cleanup.pending_count,
        removed_count=cleanup.removed_count,
        reclaimed_bytes=cleanup.reclaimed_bytes,
    )


@router.delete("/artifacts/{artifact_id}", response_model=ArtifactDeleteResult)
async def delete_artifact(
    artifact_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> ArtifactDeleteResult:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise api_error(404, "artifact-not-found", "artifact not found")
    try:
        references, removed, reclaimed = _services(request).artifacts.delete_library_artifact(
            session, artifact
        )
    except ValueError as exc:
        raise api_error(409, "artifact-in-use", str(exc)) from exc
    session.commit()
    return ArtifactDeleteResult(
        artifact_id=artifact_id,
        reference_count=references,
        removed_count=removed,
        reclaimed_bytes=reclaimed,
    )


@router.get("/artifacts/{artifact_id}", response_model=ArtifactOut)
async def get_artifact(
    artifact_id: str,
    session: ConversationSessionDep,
) -> ArtifactOut:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise api_error(404, "artifact-not-found", "artifact not found")
    result = ArtifactOut.model_validate(artifact)
    result.generation_identity = _artifact_generation_identity(session, artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/artifacts/{artifact_id}/content")
async def artifact_content(
    artifact_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Response:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise api_error(404, "artifact-not-found", "artifact not found")
    try:
        path, media_type, disposition = _services(request).artifacts.delivery_metadata(artifact)
    except (FileNotFoundError, ValueError) as exc:
        raise api_error(
            410, "artifact-file-unreadable", "artifact file is missing or corrupt"
        ) from exc
    return FileResponse(
        path,
        media_type=media_type,
        filename=Path(artifact.original_name or "artifact").name,
        content_disposition_type=disposition,
        stat_result=path.stat(),
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "ETag": f'"{artifact.sha256}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


def _with_installed_counts(items: list[CatalogModel], counts: dict[str, int]) -> list[CatalogModel]:
    """Say how many versions are here, and say nothing when that is unknown.

    A model with no recorded identity gets `None` rather than `0`. The two
    look alike and mean opposite things: one is "none of these are installed",
    the other is "this kind does not record which version it is, so nothing on
    disk can be matched against these". Rendering the second as the first is
    how someone reinstalls a checkpoint they already have.
    """
    return [
        item
        if not item.parent_model_id
        else item.model_copy(update={"installed_version_count": counts.get(item.parent_model_id)})
        for item in items
    ]


def _installed_counts_by_parent(session: Session) -> dict[str, int]:
    """How many versions of each model are already here, where that is knowable.

    Built from the same manifest field update checks read. A model absent from
    this map has no recorded identity at all, which is not the same as having
    none installed - see `_grouped_by_parent`.
    """
    counts: dict[str, int] = {}
    for identity in installed_civitai_identities(session):
        counts[identity.model_id] = counts.get(identity.model_id, 0) + 1
    return counts


def _grouped_by_parent(items: list[CatalogModel]) -> list[CatalogModel]:
    """One card per model, where the provider says a card is a version of one.

    A CivitAI card is a version, because a version is what installs. Listing
    every version as its own card buries a model with twelve releases under
    twelve rows that differ only in a suffix. The card becomes the model, and
    keeps the newest version's identity so it still names something real.

    `version_count` is what stops this from silently installing the latest: a
    card offering more than one has to open the chooser rather than act, and
    the count is how the browser knows which card that is.

    Order is preserved - the first card seen for a parent keeps its place, so
    whatever the provider's ranking meant still holds.
    """
    grouped: dict[str, CatalogModel] = {}
    ordered: list[CatalogModel] = []
    for item in items:
        parent = item.parent_model_id
        if not parent:
            ordered.append(item)
            continue
        existing = grouped.get(parent)
        if existing is None:
            card = item.model_copy(
                update={"name": item.parent_model_name or item.name, "version_count": 1}
            )
            grouped[parent] = card
            ordered.append(card)
            continue
        merged = existing.model_copy(update={"version_count": existing.version_count + 1})
        grouped[parent] = merged
        ordered[ordered.index(existing)] = merged
    return ordered


@router.get("/catalog", response_model=CatalogPage)
async def catalog_search(
    request: Request,
    session: SessionDep,
    source: str = Query(default="huggingface", min_length=1, max_length=32),
    query: str = "",
    role: str | None = None,
    sort: str = "trending",
    limit: int = Query(default=30, ge=1, le=100),
    cursor: str | None = None,
    compatibility: str | None = None,
    file_format: str | None = None,
    quantization: str | None = None,
    license_id: str | None = None,
    gated: str | None = None,
    architecture: str | None = None,
    min_parameters: int | None = Query(default=None, ge=0),
    max_parameters: int | None = Query(default=None, ge=0),
    max_size_bytes: int | None = Query(default=None, ge=0),
    updated_within_days: int | None = Query(default=None, ge=1, le=3650),
) -> CatalogPage:
    services = _services(request)
    try:
        catalog: CatalogSource = services.catalog_sources.get(source)
    except CatalogSourceNotFound as exc:
        raise api_error(404, "catalog-source-not-found", str(exc)) from exc
    try:
        media_catalog = role in {"image", "video"} and services.settings.media_engine == "comfyui"
        page = await catalog.search(
            query=query,
            role=role,
            sort=sort,
            limit=limit,
            cursor=cursor,
            compatibility=None if media_catalog else compatibility,
            file_format=file_format,
            quantization=quantization,
            license_id=license_id,
            gated=gated,
            architecture=architecture,
            min_parameters=min_parameters,
            max_parameters=max_parameters,
            max_size_bytes=max_size_bytes,
            updated_within_days=updated_within_days,
        )
        if not media_catalog or role is None:
            return page.model_copy(
                update={
                    "items": _with_installed_counts(
                        _grouped_by_parent(page.items), _installed_counts_by_parent(session)
                    )
                }
            )
        registry = ComfyTemplateRegistry(services.settings)
        items = []
        for item in page.items:
            ready = bool(registry.matches(item.remote_id, role))
            adaptive_candidate = role == "image" and "safetensors" in {
                item_format.casefold() for item_format in item.formats
            }
            items.append(
                item.model_copy(
                    update={
                        "compatibility": (
                            "likely"
                            if ready
                            else ("advanced_import" if adaptive_candidate else "unsupported")
                        ),
                        "compatibility_reasons": (
                            ["Official ComfyUI workflow available"]
                            if ready
                            else (
                                ["One-click checkpoint compatibility is checked before download"]
                                if adaptive_candidate
                                else ["No safe automatic workflow contract is available"]
                            )
                        ),
                    }
                )
            )
        if compatibility:
            items = [item for item in items if item.compatibility == compatibility]
        if sort == "compatible":
            items.sort(
                key=lambda item: (
                    item.compatibility != "likely",
                    -(item.downloads or 0),
                )
            )
        return page.model_copy(
            update={
                "items": _with_installed_counts(
                    _grouped_by_parent(items), _installed_counts_by_parent(session)
                )
            }
        )
    except ValueError as exc:
        raise api_error(422, "catalog-request-invalid", f"invalid catalog request: {exc}") from exc
    except Exception as exc:
        raise api_error(
            503,
            "catalog-unavailable",
            f"{catalog.display_name} is temporarily unavailable. Check your connection and retry.",
        ) from exc


@router.get("/catalog/workflow-models", response_model=list[CatalogModel])
async def workflow_catalog_models(request: Request, role: str) -> list[CatalogModel]:
    if role not in {"image", "video"}:
        return []
    registry = ComfyTemplateRegistry(_services(request).settings)
    # One card per workflow template, not per repository: a repository can
    # ship several official workflows for different operations, and collapsing
    # them meant the alphabetically-first variant silently answered for all.
    cards: list[CatalogModel] = []
    for template in registry.available(role):
        remote_id = template.remote_id
        cards.append(
            CatalogModel(
                remote_id=remote_id,
                name=template.id,
                author=remote_id.split("/", 1)[0],
                pipeline_tag=template.operation.replace("_", "-"),
                formats=sorted(
                    {
                        Path(dependency.path).suffix.lower().lstrip(".")
                        for dependency in template.dependencies
                        if Path(dependency.path).suffix
                    }
                ),
                compatibility="likely",
                compatibility_reasons=["Official ComfyUI workflow available"],
                workflow_template_id=template.id,
                operation=template.operation,
            )
        )
    return sorted(cards, key=lambda item: (item.remote_id.casefold(), item.name))


@router.get("/catalog/item", response_model=CatalogDetail)
async def catalog_item_detail(
    request: Request,
    source: str = Query(default="huggingface", min_length=1, max_length=32),
    item_id: str = Query(alias="id", min_length=1, max_length=500),
    revision: str = "main",
    role: str | None = None,
) -> CatalogDetail:
    services = _services(request)
    try:
        selected_source = services.catalog_sources.get(source)
    except CatalogSourceNotFound as exc:
        raise api_error(404, "catalog-source-not-found", str(exc)) from exc
    if not selected_source.validate_item_id(item_id):
        raise api_error(422, "catalog-item-id-invalid", "invalid catalog item id")
    try:
        detail = await selected_source.inspect(item_id, revision, role)
        return CatalogDetail.model_validate(detail)
    except Exception as exc:
        raise api_error(
            503,
            "catalog-unavailable",
            f"{selected_source.display_name} is temporarily unavailable. "
            "Check your connection and retry.",
        ) from exc


@router.post("/catalog/preflight", response_model=CatalogPreflight)
async def catalog_item_preflight(
    payload: CatalogPreflightRequest,
    request: Request,
    session: SessionDep,
    source: str = Query(default="huggingface", min_length=1, max_length=32),
    item_id: str = Query(alias="id", min_length=1, max_length=500),
) -> CatalogPreflight:
    """Preflight any registered source's item; ids are source-shaped."""

    services = _services(request)
    try:
        selected = services.catalog_sources.get(source)
    except CatalogSourceNotFound as exc:
        raise api_error(404, "catalog-source-unknown", str(exc)) from exc
    if not selected.validate_item_id(item_id):
        raise api_error(422, "catalog-item-id-invalid", "This catalog item id is not valid")
    try:
        return await resolve_catalog_preflight(services, session, item_id, payload, source=source)
    except CatalogUnavailableError as exc:
        raise api_error(503, "catalog-source-unavailable", str(exc)) from exc


@router.get("/catalog/{owner}/{name}", response_model=CatalogDetail)
async def catalog_detail(
    owner: str,
    name: str,
    request: Request,
    revision: str = "main",
    role: str | None = None,
) -> CatalogDetail:
    try:
        detail = await _services(request).catalog.inspect(f"{owner}/{name}", revision, role)
        return CatalogDetail.model_validate(detail)
    except Exception as exc:
        raise api_error(
            503,
            "catalog-unavailable",
            "Hugging Face is temporarily unavailable. Check your connection and retry.",
        ) from exc


class CatalogUnavailableError(RuntimeError):
    """The catalog could not be reached or did not answer usefully."""


@router.post("/catalog/{owner}/{name}/preflight", response_model=CatalogPreflight)
async def catalog_preflight(
    owner: str,
    name: str,
    payload: CatalogPreflightRequest,
    request: Request,
    session: SessionDep,
) -> CatalogPreflight:
    try:
        return await resolve_catalog_preflight(
            _services(request),
            session,
            f"{owner}/{name}",
            payload,
        )
    except CatalogUnavailableError as exc:
        raise api_error(503, "catalog-unavailable", str(exc)) from exc


async def resolve_catalog_preflight(
    services: Services,
    session: Session,
    remote_id: str,
    payload: CatalogPreflightRequest,
    *,
    source: str = "huggingface",
    validate_resolved: Callable[[ResolvedInstallPlan], None] | None = None,
) -> CatalogPreflight:
    """Plan an install without deciding how a caller should report failure.

    Both the catalog route and a reference-recipe install need the whole
    operation - inspection, template matching, adaptive-checkpoint selection,
    dependency resolution and plan persistence - so it lives here rather than in
    a request handler, and raises a domain error the route maps to a status code.

    `validate_resolved` runs immediately before the plan is persisted. A caller
    with its own requirements, such as a recipe that pins exact checksums, must
    reject there rather than afterwards: a committed plan is installable on its
    own through the download endpoint, so a plan persisted and then refused would
    leave behind exactly what the refusal was meant to prevent.
    """

    catalog = services.catalog_sources.get(source)
    try:
        # A LoRA is inspected through the LoRA catalog role however it was
        # asked for. A provider shown a LoRA card under an image role
        # classifies it as unsupported, which is correct of the provider and
        # useless to us.
        inspection_role = (
            "lora"
            if "lora" in {payload.auxiliary_kind, payload.workflow_reference_kind}
            else payload.role
        )
        raw_detail = await catalog.inspect(remote_id, payload.revision, inspection_role)
        detail = CatalogDetail.model_validate(raw_detail)
        # CivitAI identities live under each normalized file's metadata; hoist
        # them once so every downstream consumer - checks, file sources, the
        # planner - sees one uniform shape. The model id stands where a
        # repository would and the version where a revision would, so the
        # tamper comparison covers every provider.
        detail = detail.model_copy(
            update={
                "files": [
                    {
                        **item,
                        # Production normalization puts both identities at the
                        # file top level; nested metadata carries the version
                        # but never the file id. Top level always wins.
                        "source_version_id": item.get("source_version_id")
                        or item["metadata"].get("source_version_id"),
                        "source_file_id": item.get("source_file_id"),
                        "source_remote_id": item.get("source_remote_id")
                        or item["metadata"].get("source_model_id"),
                        "source_revision": item.get("source_revision")
                        or item.get("source_version_id")
                        or item["metadata"].get("source_version_id"),
                        "source_filename": item.get("source_filename") or item.get("filename"),
                    }
                    if isinstance(item.get("metadata"), dict)
                    and item["metadata"].get("provider") == "civitai"
                    else item
                    for item in detail.files
                ]
            }
        )
        if payload.role == "chat" and payload.engine == "llama.cpp":
            try:
                initial_selection = (
                    list(payload.selected_files)
                    if payload.selected_files
                    else automatic_gguf_selection(
                        detail.files,
                        collect_system_info(services.settings).memory_total_bytes,
                    )
                )
                if not automatic_mmproj_selection(detail.files, initial_selection):
                    companion = await services.catalog.discover_vision_projector(
                        detail.model.remote_id,
                        initial_selection,
                    )
                    if companion:
                        detail = detail.model_copy(update={"files": [*detail.files, companion]})
            except (GGUFSelectionError, httpx.HTTPError, ValueError):
                pass
    except Exception as exc:
        raise CatalogUnavailableError(
            f"{catalog.display_name} is temporarily unavailable. Check your connection and retry."
        ) from exc
    system = collect_system_info(services.settings)

    def assess(
        target: CatalogDetail, request: CatalogPreflightRequest | None = None
    ) -> CatalogPreflight:
        """Assess with the request's exact ids, every time.

        A wrapper rather than the same argument at every call site: one that
        forgets it silently plans the provider's primary variant, which looks
        exactly like a successful install of something else.

        A caller that rewrote the request - to pin a revision, or to substitute
        template-selected filenames - passes its own version, and the ids ride
        along with it. Dropping them there would be the same silent fallback in
        a place nobody was looking.
        """
        return assess_catalog_install(
            target,
            request or payload,
            services.settings,
            system,
            selected_file_ids=payload.selected_file_ids,
        )

    async def finalize(
        result: CatalogPreflight,
        resolved_detail: CatalogDetail,
    ) -> CatalogPreflight:
        metadata: dict[str, bytes] = {}
        inspect_prefix = getattr(catalog, "inspect_file_prefix", None)
        repo_files = {
            str(item.get("filename") or "")
            for item in resolved_detail.files
            if item.get("filename")
        }
        metadata_paths = sorted(
            path
            for path in repo_files
            if PurePosixPath(path).name.casefold()
            in {"config.json", "model_index.json", "generation_config.json"}
        )[:16]
        selected_binary_paths = [
            path
            for path in result.selected_files
            if path.casefold().endswith((".gguf", ".safetensors"))
        ][:8]

        file_metadata, ambiguous_files = catalog_file_index(
            resolved_detail.files,
            provider=source,
            prefer_duplicate_variants=not payload.selected_files,
        )
        # Offered from the resolver's own safety predicate, never a second
        # opinion beside it: a chooser that lists a row exact selection would
        # refuse is a chooser that hands out dead ends.
        #
        # Only for a request that named files and could not settle one. An
        # answered request has nothing left to choose, and returning the
        # chooser again would ask the same question twice.
        variants = (
            {
                filename: [
                    CatalogFileVariant(
                        source_file_id=str(item.get("source_file_id") or ""),
                        filename=filename,
                        size_bytes=item.get("size") if isinstance(item.get("size"), int) else None,
                        precision=(
                            str(item["source_file_precision"])
                            if isinstance(item.get("source_file_precision"), str)
                            else None
                        ),
                    )
                    for item in safe_civitai_file_variants(resolved_detail.files, filename)
                ]
                # Only names this request asked for. A version can publish
                # duplicate attachments nobody selected, and offering a choice
                # about those turns an install that succeeded into a question.
                for filename in sorted(ambiguous_files & set(payload.selected_files))
            }
            if source == "civitai" and not payload.selected_file_ids
            else {}
        )
        variants = {name: rows for name, rows in variants.items() if len(rows) > 1}

        async def fetch_prefix(path: str) -> tuple[str, bytes] | None:
            if not callable(inspect_prefix):
                return None
            try:
                source = file_metadata.get(path) or {}
                source_remote_id = str(
                    source.get("source_remote_id") or resolved_detail.model.remote_id
                )
                source_revision = str(source.get("source_revision") or resolved_detail.revision)
                source_filename = str(source.get("source_filename") or path)
                if path.casefold().endswith(".safetensors"):
                    prefix = await inspect_prefix(
                        source_remote_id,
                        source_revision,
                        source_filename,
                        max_bytes=8,
                    )
                    if len(prefix) < 8:
                        return path, prefix
                    header_size = int.from_bytes(prefix[:8], "little")
                    limit = min(8 + header_size, MAX_WEIGHT_HEADER_BYTES + 8)
                elif path.casefold().endswith(".gguf"):
                    limit = MAX_WEIGHT_HEADER_BYTES
                else:
                    limit = MAX_METADATA_BYTES
                return (
                    path,
                    await inspect_prefix(
                        source_remote_id,
                        source_revision,
                        source_filename,
                        max_bytes=limit,
                    ),
                )
            except Exception:
                # Staged bytes are inspected again before activation. Remote
                # prefix inspection is an optimization, never a trust bypass.
                return None

        inspected_prefixes = await asyncio.gather(
            *(fetch_prefix(path) for path in [*metadata_paths, *selected_binary_paths])
        )
        metadata.update(item for item in inspected_prefixes if item is not None)
        if not metadata and resolved_detail.model.architecture:
            metadata["catalog-config.json"] = json.dumps(
                {"model_type": resolved_detail.model.architecture},
                separators=(",", ":"),
            ).encode()
        inspection_error: str | None = None
        try:
            inspection = inspect_repository_metadata(
                metadata,
                result.selected_files,
                role=payload.role,
            )
        except ModelManifestError as exc:
            inspection_error = str(exc)
            inspection = inspect_repository_metadata(
                {},
                result.selected_files,
                role=payload.role,
            )
        if source == "civitai" and "lora" in {
            payload.auxiliary_kind,
            payload.workflow_reference_kind,
        }:
            # CivitAI file-prefix inspection is deliberately unavailable, so a
            # bare safetensors would inspect as "checkpoint" and block as a
            # kind mismatch. The provider's own typed declaration substitutes
            # for planning; the manager's mandatory staged-byte LoRA
            # inspection still gates activation after download.
            #
            # Either ownership field, matching the role selection above. A
            # workflow-owned LoRA sends `workflow_reference_kind` and must not
            # also send `auxiliary_kind` - the planner refuses that as
            # conflicting ownership - so naming only the auxiliary field made
            # every workflow-owned CivitAI LoRA fail as a kind mismatch, for
            # having declared its ownership correctly.
            declared_loras = {
                str(item.get("filename") or "")
                for item in resolved_detail.files
                if isinstance(item.get("metadata"), dict)
                and str(item["metadata"].get("model_type") or "").casefold() == "lora"
            }
            if declared_loras and all(path in declared_loras for path in result.selected_files):
                inspection = dataclasses.replace(
                    inspection,
                    components=tuple(
                        dataclasses.replace(component, kind="lora", target_folder="loras")
                        if component.path in declared_loras
                        else component
                        for component in inspection.components
                    ),
                )
        # The chosen row, not whichever row shares its destination. A filename
        # index re-resolves an answered choice back to the provider's primary
        # variant, which is the whole failure the exact id exists to prevent.
        try:
            selected_metadata = selected_catalog_file_metadata(
                resolved_detail.files,
                result.selected_files,
                provider=source,
                prefer_duplicate_variants=not payload.selected_files,
                selected_file_ids=payload.selected_file_ids,
                expected_sha256=result.expected_sha256,
            )
        except ExactCivitaiFileSelectionError as exc:
            raise api_error(422, "catalog-file-variant-invalid", str(exc)) from exc
        workflow_component_folders: dict[str, str] = {}
        workflow_contract_error: str | None = None
        if result.workflow_template_id:
            try:
                selected_template = ComfyTemplateRegistry(services.settings).get(
                    result.workflow_template_id,
                    payload.role,
                    remote_id=result.remote_id,
                    revision=result.revision,
                    selected_files=result.selected_files,
                    comfy_paths=result.comfy_paths,
                )
                if (
                    selected_template.sha256 != result.workflow_template_sha256
                    or selected_template.selected_files != result.selected_files
                    or selected_template.comfy_paths != result.comfy_paths
                ):
                    raise ValueError("workflow template changed; run the install check again")
                workflow_component_folders = selected_template.component_folders
            except ValueError as exc:
                workflow_contract_error = str(exc)
        # CivitAI file identities live under the normalized file's metadata;
        # the planner reads them at the top level, so hoist without mutating
        # the cached detail.
        resolved = resolve_install_plan(
            remote_id=result.remote_id,
            revision=result.revision,
            role=payload.role,
            engine=payload.engine,
            selected_files=selected_metadata,
            provider=source,
            inspection=inspection,
            workflow_template_id=result.workflow_template_id,
            workflow_template_sha256=result.workflow_template_sha256,
            comfy_paths=result.comfy_paths,
            workflow_component_folders=workflow_component_folders,
            source_remote_id=result.source_remote_id,
            auxiliary_kind=payload.auxiliary_kind,
            workflow_reference_kind=payload.workflow_reference_kind,
        )
        if inspection_error:
            resolved = resolved.blocked(
                "metadata_inspection_failed",
                inspection_error,
            )
        if workflow_contract_error:
            resolved = resolved.blocked(
                "workflow_contract_changed",
                workflow_contract_error,
            )
        if not result.can_install:
            resolved = resolved.blocked(
                "preflight_blocked",
                next(
                    (check.detail for check in result.checks if check.status == "block"),
                    "The install check did not pass.",
                ),
            )
        if validate_resolved:
            validate_resolved(resolved)
        plan = persist_install_plan(session, resolved)
        session.commit()
        return result.model_copy(update={"install_plan": plan, "file_variants": variants})

    if payload.auxiliary_kind:
        auxiliary_folder = comfy_folder_for_kind(payload.auxiliary_kind)
        if payload.auxiliary_kind not in AUXILIARY_ASSET_KINDS or auxiliary_folder is None:
            raise api_error(
                422,
                "auxiliary-kind-unsupported",
                "Unsupported model asset kind.",
            )
        if payload.role != "image":
            result = assess(detail)
            return await finalize(
                result.model_copy(
                    update={
                        "can_install": False,
                        "checks": [
                            *result.checks,
                            CatalogPreflightCheck(
                                id="auxiliary-role",
                                label="Asset role",
                                status="block",
                                detail="LoRA assets currently extend image workflows.",
                            ),
                        ],
                    }
                ),
                detail,
            )
        return await finalize(
            assess(detail).model_copy(
                update={
                    "comfy_paths": {auxiliary_folder: "."},
                }
            ),
            detail,
        )

    if payload.workflow_reference_kind:
        # A workflow named one exact file. Template ranking exists to guess
        # what a repository is for, and there is nothing left to guess here.
        # Letting it run is how asking for one 4GB text encoder planned the
        # repository's official four-file bundle instead: a diffusion model, an
        # unrelated LoRA, the encoder actually wanted, and a VAE already on
        # disk, together over 19GB.
        # One identity, in exactly one form. A retry that answers an ambiguous
        # filename supplies the exact provider id and no filename at all, so
        # counting filenames alone refused the very request the refusal asked
        # for.
        named = len(payload.selected_files) + len(payload.selected_file_ids)
        if named != 1:
            raise api_error(
                422,
                "workflow-asset-file-not-exact",
                "A workflow asset install must name exactly one file, by name or by "
                "exact provider id. Run the install check again with the exact file "
                "the workflow needs.",
            )
        return await finalize(
            assess(detail),
            detail,
        )

    if (
        payload.role == "chat"
        or payload.engine != "comfyui"
        or services.settings.media_engine != "comfyui"
    ):
        return await finalize(
            assess(detail),
            detail,
        )

    runtime_status = services.runtimes.status("comfyui")
    if runtime_status.security_status == "blocked" and runtime_status.state != "ready":
        result = assess(detail)
        checks = [
            *[check for check in result.checks if check.id != "runtime"],
            CatalogPreflightCheck(
                id="runtime",
                label="Runtime security",
                status="block",
                detail=runtime_status.security_message or runtime_status.message,
            ),
        ]
        return await finalize(
            result.model_copy(update={"can_install": False, "checks": checks}),
            detail,
        )

    registry = ComfyTemplateRegistry(services.settings)
    candidates = registry.matches(detail.model.remote_id, payload.role)
    requested_template = (payload.workflow_template_id or "").strip()
    if requested_template:
        # The user chose an exact variant; preflight that one or refuse.
        # Silently ranking a different template is how a text-to-video request
        # once installed a speech-to-video workflow with an audio encoder.
        candidates = [item for item in candidates if item.id == requested_template]
        if not candidates:
            raise api_error(
                422,
                "workflow-template-mismatch",
                "The selected workflow does not belong to this repository and role. "
                "Refresh the catalog and choose a workflow again.",
            )
    if not candidates:
        resolved_payload = payload.model_copy(update={"revision": detail.revision})
        result = assess(detail, resolved_payload)
        adaptive = registry.adaptive_checkpoint(
            detail.model.remote_id,
            detail.revision,
            result.selected_files,
            payload.role,
        )
        if adaptive:
            checks = [
                *[
                    (
                        CatalogPreflightCheck(
                            id="runtime",
                            label="Runtime compatibility",
                            status="warn",
                            detail=(
                                "This is a safe single-file checkpoint layout. ComfyUI "
                                "must accept the checkpoint before LM Atelier activates it."
                            ),
                        )
                        if check.id == "runtime"
                        else check
                    )
                    for check in result.checks
                ],
                CatalogPreflightCheck(
                    id="workflow-template",
                    label="Automatic runtime setup",
                    status="warn",
                    detail=(
                        "LM Atelier will use ComfyUI's standard checkpoint workflow and "
                        "keep the model inactive if live runtime validation fails."
                    ),
                ),
            ]
            return await finalize(
                result.model_copy(
                    update={
                        "source_remote_id": detail.model.remote_id,
                        "comfy_paths": adaptive.comfy_paths,
                        "workflow_template_id": adaptive.id,
                        "workflow_template_sha256": adaptive.sha256,
                        "checks": checks,
                    }
                ),
                detail,
            )
        checks = [
            *[check for check in result.checks if check.id != "selection"],
            CatalogPreflightCheck(
                id="workflow-template",
                label="Automatic runtime setup",
                status="block",
                detail=(
                    "No official workflow or safe adaptive standard-checkpoint contract "
                    "is available for this model."
                ),
            ),
        ]
        return await finalize(
            result.model_copy(
                update={
                    "source_remote_id": detail.model.remote_id,
                    "selected_files": [],
                    "expected_sha256": {},
                    "download_bytes": 0,
                    "can_install": False,
                    "checks": checks,
                }
            ),
            detail,
        )

    inspected: dict[tuple[str, str], CatalogDetail] = {}
    viable: list[tuple[int, int, ComfyTemplate, CatalogDetail]] = []
    for candidate in candidates:
        if viable and candidate.score < viable[0][0]:
            break
        bundled_files: list[dict[str, Any]] = []
        primary_detail: CatalogDetail | None = None
        unavailable = False
        for dependency in candidate.dependencies:
            key = (dependency.remote_id.casefold(), dependency.revision)
            source_detail = inspected.get(key)
            if source_detail is None:
                try:
                    raw_source = await services.catalog.inspect(
                        dependency.remote_id,
                        dependency.revision,
                        payload.role,
                    )
                    source_detail = CatalogDetail.model_validate(raw_source)
                except Exception:
                    unavailable = True
                    break
                inspected[key] = source_detail
            source_file = next(
                (
                    item
                    for item in source_detail.files
                    if str(item.get("filename") or "") == dependency.path
                ),
                None,
            )
            if source_file is None:
                unavailable = True
                break
            if dependency.remote_id.casefold() == candidate.remote_id.casefold():
                primary_detail = source_detail
                bundled_files.append(dict(source_file))
            else:
                bundled_files.append(
                    {
                        **source_file,
                        "filename": dependency.path,
                        "source_remote_id": dependency.remote_id,
                        "source_revision": source_detail.revision,
                        "source_filename": dependency.path,
                    }
                )
        if unavailable or primary_detail is None:
            continue
        any_gated = any(
            inspected[(dependency.remote_id.casefold(), dependency.revision)].model.gated
            for dependency in candidate.dependencies
        )
        bundled_model = primary_detail.model.model_copy(
            update={"gated": primary_detail.model.gated or any_gated}
        )
        bundled_detail = primary_detail.model_copy(
            update={"model": bundled_model, "files": bundled_files}
        )
        size = sum(int(item.get("size") or 0) for item in bundled_files)
        viable.append((candidate.score, size, candidate, bundled_detail))
    if not viable:
        result = assess(detail)
        checks = [
            *[check for check in result.checks if check.id != "selection"],
            CatalogPreflightCheck(
                id="workflow-template",
                label="Automatic runtime setup",
                status="block",
                detail=(
                    "ComfyUI advertises a matching workflow, but its complete safe model "
                    "bundle is not available from the declared repository."
                ),
            ),
        ]
        return await finalize(
            result.model_copy(
                update={
                    "source_remote_id": detail.model.remote_id,
                    "selected_files": [],
                    "expected_sha256": {},
                    "download_bytes": 0,
                    "can_install": False,
                    "checks": checks,
                }
            ),
            detail,
        )

    _, _, template, resolved_detail = min(
        viable,
        key=lambda item: (-item[0], item[1], item[2].id),
    )
    resolved_payload = payload.model_copy(
        update={
            "revision": resolved_detail.revision,
            "selected_files": template.selected_files,
        }
    )
    if payload.selected_file_ids:
        # The template chose the files here, so an exact id was answering a
        # question this path never asked. Refused rather than dropped: silently
        # ignoring it would install the template's bundle while the caller
        # believed it had named one exact variant.
        raise api_error(
            422,
            "catalog-file-variant-not-applicable",
            "This install resolves its files from a workflow template, so it cannot "
            "also name an exact provider file. Run the install check again without one.",
        )
    result = assess(resolved_detail, resolved_payload)
    checks = [
        *[
            (
                CatalogPreflightCheck(
                    id="runtime",
                    label="Runtime compatibility",
                    status="pass",
                    detail=(
                        "The selected files match the official workflow shipped with "
                        "the active ComfyUI runtime."
                    ),
                )
                if check.id == "runtime"
                else check
            )
            for check in result.checks
        ],
        CatalogPreflightCheck(
            id="workflow-template",
            label="Automatic runtime setup",
            status="pass",
            detail=(
                "LM Atelier found a complete official ComfyUI model bundle and workflow; "
                "both will be configured automatically."
            ),
        ),
    ]
    return await finalize(
        result.model_copy(
            update={
                "source_remote_id": detail.model.remote_id,
                "comfy_paths": template.comfy_paths,
                "workflow_template_id": template.id,
                "workflow_template_sha256": template.sha256,
                "checks": checks,
            }
        ),
        resolved_detail,
    )


def _planned_download_fields(plan: InstallPlan | None) -> dict[str, Any]:
    """The download fields an immutable plan authorises, and nothing else.

    This is both the comparison used to reject a tampered request and the source
    a recipe install builds its request from, so the two can never disagree.
    """

    if not plan:
        raise ValueError("install plan not found; run the install check again")
    if plan.status != "planned":
        raise ValueError("install plan is no longer active; run the install check again")
    if plan.compatibility != "supported":
        raise ValueError(plan.failure_reason or "this model layout is unsupported")
    if plan.resolver_version != INSTALL_RESOLVER_VERSION:
        raise ValueError("install contract changed; run the install check again")
    runtime = plan.runtime_contract_json
    if (
        runtime.get("workflow_template_id")
        and runtime.get("workflow_compiler_version") != COMFY_TEMPLATE_COMPILER_VERSION
    ):
        raise ValueError("workflow contract changed; run the install check again")
    return {
        "remote_id": plan.remote_id,
        "revision": plan.revision,
        "role": plan.role,
        "engine": plan.engine,
        "allow_patterns": [
            str(item.get("path") or "")
            for item in plan.artifacts_json
            if item.get("required", True)
        ],
        "expected_sha256": {
            str(item["path"]): str(item["sha256"])
            for item in plan.artifacts_json
            if item.get("sha256")
        },
        "file_sources": {}
        if plan.provider == "civitai"
        else {
            str(item["path"]): {
                "remote_id": str(item["source_remote_id"]),
                "revision": str(item["source_revision"]),
                "filename": str(item["source_path"]),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
                "source_version_id": item.get("source_version_id"),
                "source_file_id": item.get("source_file_id"),
            }
            for item in plan.artifacts_json
            if item.get("source_remote_id")
            and item.get("source_revision")
            and item.get("source_path")
        },
        "source_remote_id": runtime.get("source_remote_id"),
        "comfy_paths": runtime.get("comfy_paths") or {},
        "workflow_template_id": runtime.get("workflow_template_id"),
        "workflow_template_sha256": runtime.get("workflow_template_sha256"),
        "auxiliary_kind": runtime.get("auxiliary_kind"),
    }


@router.post("/downloads", response_model=JobOut, status_code=202)
async def create_download(payload: DownloadRequest, request: Request, session: SessionDep) -> Job:
    manager: DownloadManager = _services(request).downloads
    try:
        if payload.install_plan_id:
            plan = session.get(InstallPlan, payload.install_plan_id)
            expected = _planned_download_fields(plan)
            supplied = payload.model_dump()
            mismatched = [key for key, value in expected.items() if supplied.get(key) != value]
            if mismatched:
                raise ValueError(
                    "install request no longer matches its immutable plan; "
                    "run the install check again"
                )
        return manager.create(session, payload)
    except ValueError as exc:
        raise api_error(422, "download-request-invalid", str(exc)) from exc


@router.post("/downloads/cleanup", response_model=StorageCleanupResult)
async def cleanup_partial_downloads(request: Request, session: SessionDep) -> StorageCleanupResult:
    removed_count, reclaimed_bytes = _services(request).downloads.cleanup_partials(session)
    return StorageCleanupResult(removed_count=removed_count, reclaimed_bytes=reclaimed_bytes)


@router.post("/downloads/{job_id}/pause", response_model=JobOut)
async def pause_download(job_id: str, request: Request, session: SessionDep) -> Job:
    if not await _services(request).downloads.pause(job_id):
        raise api_error(409, "download-not-pausable", "download is not running or cannot be paused")
    session.expire_all()
    job = session.get(Job, job_id)
    if not job:
        raise api_error(404, "download-job-not-found", "download job not found")
    return job


@router.post("/downloads/{job_id}/resume", response_model=JobOut)
async def resume_download(job_id: str, request: Request, session: SessionDep) -> Job:
    if not _services(request).downloads.resume(job_id):
        raise api_error(
            409, "download-not-resumable", "download is not paused, failed, or interrupted"
        )
    session.expire_all()
    job = session.get(Job, job_id)
    if not job:
        raise api_error(404, "download-job-not-found", "download job not found")
    return job


@router.get("/edit-templates", response_model=list[EditTemplateOut])
async def list_edit_templates(session: SessionDep) -> list[EditTemplate]:
    """Enabled one-click edits, built-ins first, then by name."""

    return list(
        session.scalars(
            select(EditTemplate)
            .where(EditTemplate.enabled.is_(True))
            .order_by(EditTemplate.builtin.desc(), EditTemplate.name)
        ).all()
    )


@router.post("/edit-templates", response_model=EditTemplateOut, status_code=201)
async def create_edit_template(payload: EditTemplateCreate, session: SessionDep) -> EditTemplate:
    """Save an edit that worked as a reusable one-click template."""

    existing = session.scalar(select(EditTemplate).where(EditTemplate.name == payload.name))
    if existing:
        raise api_error(
            409, "edit-template-name-taken", "A template with this name already exists."
        )
    # Saved from a result someone liked, so the record of what produced it is
    # the run's, not the machine's current state - those differ the moment a
    # profile is switched between the edit and the save.
    capture = None
    if payload.from_run_id:
        run = session.get(Run, payload.from_run_id)
        if not run:
            raise api_error(404, "run-not-found", "That run no longer exists.")
        capture = capture_recipe(run.provenance_json)
    template = EditTemplate(
        name=payload.name,
        description=payload.description,
        instruction=payload.instruction,
        operation="image_to_image",
        settings_json=capture.settings if capture else payload.settings_json,
        trigger_words_json=[],
        content_rating="general",
        workflow_revision_id=capture.workflow_revision_id if capture else None,
        model_profile_id=capture.model_profile_id if capture else None,
        mask_mode=capture.mask_mode if capture else "none",
        builtin=False,
        enabled=True,
    )
    session.add(template)
    session.commit()
    return template


@router.delete("/edit-templates/{template_id}", status_code=204)
async def delete_edit_template(template_id: str, session: SessionDep) -> Response:
    """Built-ins disable instead: deleting one would resurrect it at next seed."""

    template = session.get(EditTemplate, template_id)
    if not template:
        raise api_error(404, "edit-template-not-found", "edit template not found")
    if template.builtin:
        template.enabled = False
    else:
        session.delete(template)
    session.commit()
    return Response(status_code=204)


@router.get("/recipes", response_model=list[ReferenceRecipe])
async def list_recipes() -> list[ReferenceRecipe]:
    return list_reference_recipes()


@router.get("/recipes/{recipe_id}", response_model=ReferenceRecipe)
async def get_recipe(recipe_id: str) -> ReferenceRecipe:
    recipe = get_reference_recipe(recipe_id)
    if not recipe:
        raise api_error(404, "reference-recipe-not-found", "reference recipe not found")
    return recipe


@router.post("/recipes/{recipe_id}/install", response_model=JobOut, status_code=202)
async def install_recipe(recipe_id: str, request: Request, session: SessionDep) -> Job:
    """Install a shipped recipe through the same verified path as the catalog.

    A recipe used to be downloaded directly, which skipped the install plan and
    with it staged inspection, component manifests, workflow compilation, the
    activation probe and the evidence write - so it produced a model that could
    never report ready. It now runs the ordinary preflight, and the resulting plan
    must match what the recipe pins before anything is transferred.
    """

    recipe = get_reference_recipe(recipe_id)
    if not recipe:
        raise api_error(404, "reference-recipe-not-found", "reference recipe not found")
    if not all(recipe.remote_id.partition("/")[::2]):
        raise api_error(
            422,
            "reference-recipe-repository-invalid",
            "this recipe does not name a valid repository",
        )
    try:
        preflight = await resolve_catalog_preflight(
            _services(request),
            session,
            recipe.remote_id,
            CatalogPreflightRequest(
                revision=recipe.revision,
                role=recipe.role,
                engine=recipe.engine,
                selected_files=[file.path for file in recipe.files],
            ),
            validate_resolved=lambda resolved: _assert_recipe_pins_hold(recipe, resolved),
        )
    except CatalogUnavailableError as exc:
        raise api_error(503, "catalog-unavailable", str(exc)) from exc
    except ValueError as exc:
        # Refused before persistence, so no installable plan is left behind.
        session.rollback()
        raise api_error(422, "recipe-plan-unresolvable", str(exc)) from exc
    plan = preflight.install_plan
    try:
        fields = _planned_download_fields(session.get(InstallPlan, plan.id) if plan else None)
        return _services(request).downloads.create(
            session,
            DownloadRequest(
                install_plan_id=plan.id if plan else None,
                recipe_id=recipe.id,
                recipe_version=recipe.version,
                default_settings=recipe.default_settings,
                **fields,
            ),
        )
    except ValueError as exc:
        raise api_error(422, "recipe-settings-invalid", str(exc)) from exc


def _assert_recipe_pins_hold(recipe: ReferenceRecipe, plan: ResolvedInstallPlan | None) -> None:
    """Refuse to install anything other than exactly what the recipe pins.

    Preflight resolves a repository, and for media it may select a bundle a
    workflow template declares. That is right for a catalog install the user is
    steering, but a recipe exists to promise specific bytes, so a resolution that
    drifts from its pins has to fail rather than quietly install something else.

    This runs on the resolved plan, before persistence. A committed plan is
    installable on its own through the download endpoint, so refusing after
    persisting would leave behind the very thing the refusal prevents.
    """

    if not plan:
        raise ValueError("the install check did not produce a plan for this recipe")
    if plan.compatibility != "supported":
        raise ValueError(
            plan.failure_reason or "this recipe cannot be installed automatically on this machine"
        )
    if plan.remote_id != recipe.remote_id or plan.revision != recipe.revision:
        raise ValueError("this recipe resolved to a different repository or revision")
    pinned = {file.path: file.sha256 for file in recipe.files}
    resolved = {artifact.path: artifact.sha256 for artifact in plan.artifacts if artifact.required}
    if set(resolved) != set(pinned):
        raise ValueError("this recipe resolved to a different set of files")
    drifted = [path for path, digest in pinned.items() if digest and resolved[path] != digest]
    if drifted:
        raise ValueError("this recipe resolved to different file contents")
    unverified = [path for path, digest in resolved.items() if not digest]
    if unverified:
        raise ValueError("this recipe resolved files without a verifiable checksum")


@router.get("/models", response_model=list[ModelInstallOut])
async def list_models(request: Request, session: SessionDep) -> list[ModelInstallOut]:
    installs = list(
        session.scalars(
            select(ModelInstall)
            .where(ModelInstall.active.is_(True))
            .order_by(ModelInstall.updated_at.desc())
        ).all()
    )
    services = _services(request)
    evidence_by_install = {
        install.id: evidence
        for install in installs
        if (
            evidence := current_capability_evidence(
                session,
                install,
                services.settings,
                services.runtimes,
            )
        )
    }
    return [
        ModelInstallOut.model_validate(install).model_copy(
            update={
                "readiness": (
                    "ready"
                    if install.id in evidence_by_install
                    else (
                        "unsupported"
                        if install.compatibility == CompatibilityLevel.UNSUPPORTED.value
                        else "unverified"
                    )
                ),
                "capability_evidence": (
                    ModelCapabilityEvidenceOut.model_validate(evidence_by_install[install.id])
                    if install.id in evidence_by_install
                    else None
                ),
            }
        )
        for install in installs
    ]


@router.post("/references", response_model=ReferenceSubjectOut, status_code=201)
async def create_reference_subject(
    payload: ReferenceSubjectCreate,
    session: SessionDep,
) -> ReferenceSubject:
    try:
        subject = create_subject(
            session,
            name=payload.name,
            kind=payload.kind,
            mention_slug=payload.mention_slug,
            description=payload.description,
            aliases=payload.aliases,
            tags=payload.tags,
        )
    except ReferenceError as exc:
        raise api_error(422, "reference-invalid", str(exc)) from exc
    session.commit()
    session.refresh(subject)
    return subject


@router.get("/references", response_model=ReferenceSubjectPage)
async def list_reference_subjects(
    session: SessionDep,
    kind: str | None = None,
    search: str | None = None,
    include_archived: bool = False,
    limit: int = DEFAULT_PAGE,
    offset: int = 0,
) -> ReferenceSubjectPage:
    try:
        rows, total = list_subjects(
            session,
            kind=kind,
            include_archived=include_archived,
            search=search,
            limit=limit,
            offset=offset,
        )
    except ReferenceError as exc:
        raise api_error(422, "reference-query-invalid", str(exc)) from exc
    return ReferenceSubjectPage(
        items=[ReferenceSubjectOut.model_validate(row, from_attributes=True) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


def _subject_or_404(session: Session, subject_id: str) -> ReferenceSubject:
    subject = session.get(ReferenceSubject, subject_id)
    if not subject:
        raise api_error(404, "reference-not-found", "That reference does not exist.")
    return subject


@router.get("/references/{subject_id}", response_model=ReferenceSubjectOut)
async def read_reference_subject(subject_id: str, session: SessionDep) -> ReferenceSubject:
    return _subject_or_404(session, subject_id)


@router.patch("/references/{subject_id}", response_model=ReferenceSubjectOut)
async def update_reference_subject(
    subject_id: str,
    payload: ReferenceSubjectUpdate,
    session: SessionDep,
) -> ReferenceSubject:
    subject = _subject_or_404(session, subject_id)
    try:
        if payload.name is not None:
            rename_subject(
                session, subject, name=payload.name, follow_mention=payload.follow_mention
            )
        if payload.archived is not None:
            set_archived(session, subject, payload.archived)
        if payload.favorite is not None:
            set_favorite(session, subject, payload.favorite)
        if (
            payload.description is not None
            or payload.aliases is not None
            or payload.tags is not None
        ):
            set_details(
                session,
                subject,
                description=payload.description,
                aliases=payload.aliases,
                tags=payload.tags,
            )
    except ReferenceError as exc:
        raise api_error(422, "reference-invalid", str(exc)) from exc
    session.commit()
    session.refresh(subject)
    return subject


@router.put("/references/{subject_id}/cover", response_model=ReferenceSubjectOut)
async def set_reference_cover(
    subject_id: str,
    payload: ReferenceCoverIn,
    session: SessionDep,
) -> ReferenceSubject:
    subject = _subject_or_404(session, subject_id)
    try:
        set_cover(session, subject, artifact_id=payload.artifact_id)
    except ReferenceError as exc:
        raise api_error(422, "reference-cover-not-held", str(exc)) from exc
    session.commit()
    session.refresh(subject)
    return subject


@router.delete("/references/{subject_id}/cover", response_model=ReferenceSubjectOut)
async def clear_reference_cover(subject_id: str, session: SessionDep) -> ReferenceSubject:
    subject = _subject_or_404(session, subject_id)
    clear_cover(session, subject)
    session.commit()
    session.refresh(subject)
    return subject


@router.get("/references/{subject_id}/deletion-impact", response_model=ReferenceDeletionImpact)
async def read_reference_deletion_impact(
    subject_id: str, session: SessionDep
) -> ReferenceDeletionImpact:
    """What deleting would destroy, answered before anything is destroyed.

    A separate read rather than a field on the subject: it is a question about
    consequences, asked at the moment someone is deciding, and computing it on
    every list would cost a scan per subject to answer a question nobody asked.
    """

    subject = _subject_or_404(session, subject_id)
    impact = deletion_impact(session, subject)
    return ReferenceDeletionImpact(
        reference_subject_id=impact.reference_subject_id,
        name=impact.name,
        asset_count=impact.asset_count,
        exclusive_artifact_ids=list(impact.exclusive_artifact_ids),
    )


@router.delete("/references/{subject_id}", status_code=204)
async def delete_reference_subject(
    subject_id: str,
    session: SessionDep,
    acknowledged_assets: int | None = None,
) -> Response:
    """Permanently delete a subject, but only against a seen impact.

    `acknowledged_assets` must match what `deletion-impact` currently reports.
    Archiving is the ordinary removal; this path destroys membership rows and is
    not undoable, so it refuses unless the caller is deleting the thing they
    were shown rather than something that has changed since.
    """

    subject = _subject_or_404(session, subject_id)
    impact = deletion_impact(session, subject)
    if acknowledged_assets is None or acknowledged_assets != impact.asset_count:
        raise api_error(
            409,
            "reference-impact-not-acknowledged",
            "Confirm the deletion impact first; this reference now holds "
            f"{impact.asset_count} image(s).",
        )
    session.delete(subject)
    session.commit()
    return Response(status_code=204)


@router.post(
    "/references/{subject_id}/assets",
    response_model=ReferenceAssetAttached,
    status_code=201,
)
async def attach_reference_asset(
    subject_id: str,
    payload: ReferenceAssetAttach,
    request: Request,
    session: SessionDep,
) -> ReferenceAssetAttached:
    """Add one image to a reference, reporting anything it closely resembles.

    The similarity scan reads image bytes, so it is given a reader that verifies
    each file against its recorded checksum. A file that fails that check is
    skipped rather than compared, because an unverifiable image is not evidence
    of anything - least of all of being a duplicate.
    """

    subject = _subject_or_404(session, subject_id)
    services = _services(request)

    def read_verified(artifact_id: str) -> bytes:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        return services.artifacts.read_verified_bytes(
            artifact,
            maximum_bytes=services.settings.max_upload_bytes,
        )

    try:
        attached = attach_asset(
            session,
            subject,
            artifact_id=payload.artifact_id,
            caption=payload.caption,
            purpose=payload.purpose,
            view_label=payload.view_label,
            read_bytes=read_verified,
        )
    except ReferenceError as exc:
        raise api_error(422, "reference-asset-invalid", str(exc)) from exc
    session.commit()
    session.refresh(attached.asset)
    return ReferenceAssetAttached(
        asset=ReferenceAssetOut.model_validate(attached.asset, from_attributes=True),
        similar=[
            ReferenceSimilarAsset(
                reference_asset_id=item.reference_asset_id,
                artifact_id=item.artifact_id,
                mean_absolute_difference=item.mean_absolute_difference,
            )
            for item in attached.similar
        ],
    )


@router.get("/references/{subject_id}/assets", response_model=list[ReferenceAssetOut])
async def list_reference_assets(subject_id: str, session: SessionDep) -> list[ReferenceAsset]:
    subject = _subject_or_404(session, subject_id)
    return list(subject.assets)


@router.post(
    "/references/{subject_id}/assets/{asset_id}/review",
    response_model=ReferenceAssetReviewed,
)
async def review_reference_image(
    subject_id: str,
    asset_id: str,
    payload: ReferenceAssetReviewRequest,
    request: Request,
    session: SessionDep,
) -> ReferenceAssetReviewed:
    """Settle one exact unchecked image review, with exact retries idempotent."""

    services = _services(request)
    try:
        result = review_reference_asset(
            session,
            services.artifacts,
            subject_id=subject_id,
            asset_id=asset_id,
            expected_state=payload.expected_state,
            expected_version=payload.expected_version,
            decision=payload.decision,
            reasons=list(payload.reasons),
            maximum_bytes=services.settings.max_upload_bytes,
            maximum_pixels=services.settings.vision_max_pixels,
        )
    except ReferenceReviewNotFound as exc:
        raise api_error(404, "reference-asset-not-attached", str(exc)) from exc
    except ReferenceReviewConflict as exc:
        raise api_error(
            409,
            "reference-asset-review-conflict",
            str(exc),
            current_state=exc.state,
            current_version=exc.version,
        ) from exc
    except ReferenceReviewInvalid as exc:
        raise api_error(422, "reference-asset-review-invalid", str(exc)) from exc
    session.commit()
    return ReferenceAssetReviewed(
        asset=ReferenceAssetOut.model_validate(result.asset, from_attributes=True),
        review=ReferenceAssetReviewEventOut.model_validate(result.review, from_attributes=True),
        idempotent=result.idempotent,
    )


@router.delete("/references/{subject_id}/assets/{asset_id}", status_code=204)
async def detach_reference_asset(
    subject_id: str,
    asset_id: str,
    session: SessionDep,
) -> Response:
    subject = _subject_or_404(session, subject_id)
    try:
        detach_asset(session, subject, asset_id=asset_id)
    except ReferenceError as exc:
        raise api_error(404, "reference-asset-not-attached", str(exc)) from exc
    session.commit()
    return Response(status_code=204)


@router.get("/model-assets", response_model=list[ModelAssetOut])
async def list_model_assets(
    session: SessionDep,
    kind: str | None = None,
) -> list[ModelAssetInstall]:
    statement = select(ModelAssetInstall)
    if kind:
        if kind not in AUXILIARY_ASSET_KINDS:
            raise api_error(422, "model-asset-kind-unsupported", "unsupported model asset kind")
        statement = statement.where(ModelAssetInstall.kind == kind)
    return list(
        session.scalars(statement.order_by(ModelAssetInstall.name, ModelAssetInstall.id)).all()
    )


@router.put(
    "/model-assets/{asset_id}/prompt-grammar",
    response_model=AdapterPromptGrammarOut,
)
async def review_model_asset_prompt_grammar(
    asset_id: str,
    payload: AdapterPromptGrammarReview,
    session: SessionDep,
) -> AdapterPromptGrammar:
    """Record how this adapter must be prompted, as reviewed on this machine.

    A PUT rather than a POST: there is exactly one grammar per adapter, and a
    second would raise the question of which one a rewriter believed. Re-review
    replaces, and a re-review that fails leaves the previous one standing.
    """

    asset = session.get(ModelAssetInstall, asset_id)
    if not asset:
        raise api_error(404, "model-asset-not-found", "That model asset is not installed.")
    try:
        row = review_adapter_grammar(
            session,
            model_asset_install_id=asset.id,
            asset_sha256=payload.asset_sha256,
            source_identity=payload.source_identity,
            source_text=payload.source_text,
            payload=payload.grammar,
            approve_prose=tuple(payload.approve_prose),
            verified_values={
                slot: frozenset(values) for slot, values in payload.verified_values.items()
            },
        )
    except PromptGrammarError as exc:
        # The grammar is refused rather than stored partially. A grammar this
        # build cannot read is one it must never act on.
        raise api_error(422, "prompt-grammar-invalid", str(exc)) from exc
    session.commit()
    session.refresh(row)
    return row


@router.get(
    "/model-assets/{asset_id}/prompt-grammar",
    response_model=AdapterPromptGrammarOut,
)
async def read_model_asset_prompt_grammar(
    asset_id: str,
    session: SessionDep,
) -> AdapterPromptGrammar:
    row = session.scalar(
        select(AdapterPromptGrammar).where(AdapterPromptGrammar.model_asset_install_id == asset_id)
    )
    if not row:
        raise api_error(
            404,
            "prompt-grammar-not-reviewed",
            "No prompt grammar has been reviewed for this adapter on this machine.",
        )
    return row


@router.patch("/model-assets/{asset_id}", response_model=ModelAssetOut)
async def update_model_asset(
    asset_id: str,
    payload: ModelAssetUpdate,
    request: Request,
    session: SessionDep,
) -> ModelAssetInstall:
    services = _services(request)
    asset = session.get(ModelAssetInstall, asset_id)
    if not asset:
        raise api_error(404, "model-asset-not-found", "model asset not found")
    values = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not values:
        return asset
    lora_fields = {
        "use_case",
        "auto_apply",
        "default_model_strength",
        "default_clip_strength",
    }
    if set(values) & lora_fields and asset.kind != "lora":
        raise api_error(
            422,
            "automatic-selection-lora-only",
            "automatic selection metadata is only available for LoRAs",
        )
    if "use_case" in values:
        values["use_case"] = values["use_case"].strip()
    for field in ("default_model_strength", "default_clip_strength"):
        value = values.get(field)
        if value is not None and not math.isfinite(value):
            raise api_error(422, "number-not-finite", f"{field} must be finite")
    next_use_case = values.get("use_case", asset.use_case).strip()
    next_auto_apply = values.get("auto_apply", asset.auto_apply)
    if next_auto_apply and not next_use_case:
        raise api_error(
            422,
            "automatic-selection-use-case-required",
            "automatic LoRA selection requires a use case",
        )
    if next_auto_apply and not asset.verified_at:
        raise api_error(
            409,
            "automatic-selection-needs-verified-lora",
            "only a verified LoRA can be selected automatically",
        )
    if values.get("active") is True and not asset.verified_at:
        raise api_error(
            409, "model-asset-not-verified", "only a verified model asset can be enabled"
        )

    active_changed = "active" in values and values["active"] != asset.active

    def apply_values() -> None:
        for field, value in values.items():
            setattr(asset, field, value)
        session.commit()

    if active_changed:
        async with services.scheduler.lease("primary"):
            previous_values = {field: getattr(asset, field) for field in values}
            was_running = next(
                worker.running for worker in services.processes.statuses() if worker.name == "media"
            )
            apply_values()
            if was_running:
                try:
                    await services.processes.start_media()
                except Exception:
                    for field, value in previous_values.items():
                        setattr(asset, field, value)
                    session.commit()
                    with suppress(Exception):
                        await services.processes.start_media()
                    raise
    else:
        apply_values()
    session.refresh(asset)
    return asset


@router.delete("/model-assets/{asset_id}", status_code=204)
async def delete_model_asset(
    asset_id: str,
    request: Request,
    session: SessionDep,
) -> Response:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        asset = session.get(ModelAssetInstall, asset_id)
        if not asset:
            raise api_error(404, "model-asset-not-found", "model asset not found")
        was_running = next(
            worker.running for worker in services.processes.statuses() if worker.name == "media"
        )
        deletion_error: BaseException | None = None
        try:
            if was_running:
                await services.processes.stop("media")
            quarantine = _delete_model_asset_locked(asset, services.settings.model_dir, session)
            try:
                await asyncio.to_thread(_finalize_model_quarantine, quarantine)
            except Exception:
                logger.warning(
                    "Deleted auxiliary model files remain safely quarantined at %s",
                    quarantine,
                    exc_info=True,
                )
        except BaseException as exc:
            deletion_error = exc
            raise
        finally:
            if was_running and not next(
                worker.running for worker in services.processes.statuses() if worker.name == "media"
            ):
                try:
                    await services.processes.start_media()
                except Exception:
                    if deletion_error is None:
                        raise
                    logger.exception(
                        "Could not restore the media worker after auxiliary deletion failed"
                    )
    return Response(status_code=204)


def _delete_model_asset_locked(
    asset: ModelAssetInstall,
    model_dir: Path,
    session: Session,
) -> Path | None:
    model_root = model_dir.resolve()
    try:
        path = _managed_model_path(model_root, asset.local_path)
        recover_model_delete_quarantines(session, model_root, strict=True)
    except ValueError as exc:
        raise api_error(422, "model-asset-delete-refused", str(exc)) from exc
    moves: list[tuple[Path, Path]] = []
    quarantine: Path | None = None
    commit_started = False
    try:
        session.flush()
        if (
            path is not None
            and path.exists()
            and not _model_asset_path_is_shared(
                session,
                asset,
                model_root,
                path,
            )
        ):
            _ensure_model_tree_link_free(path)
            quarantine = _new_model_quarantine(model_root, asset.id)
            staged = quarantine / "payload"
            os.replace(path, staged)
            moves.append((staged, path))
        session.delete(asset)
        session.flush()
        commit_started = True
        session.commit()
    except Exception:
        with suppress(Exception):
            session.rollback()
        committed = _model_asset_delete_was_committed(asset.id) if commit_started else False
        if committed is True:
            return quarantine
        if committed is None:
            logger.error(
                "Could not determine auxiliary deletion outcome; leaving files in quarantine %s",
                quarantine,
                exc_info=True,
            )
            raise
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Auxiliary deletion rollback left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            with suppress(Exception):
                _finalize_model_quarantine(quarantine)
        raise
    return quarantine


def _model_asset_path_is_shared(
    session: Session,
    asset: ModelAssetInstall,
    model_root: Path,
    path: Path,
) -> bool:
    candidates: list[ModelInstall | ModelAssetInstall] = [
        *session.scalars(select(ModelInstall)).all(),
        *session.scalars(select(ModelAssetInstall).where(ModelAssetInstall.id != asset.id)).all(),
    ]
    for candidate in candidates:
        try:
            sibling = _managed_model_path(model_root, candidate.local_path)
        except ValueError:
            return True
        if sibling is not None and (
            sibling == path or sibling in path.parents or path in sibling.parents
        ):
            return True
    return False


def _model_asset_delete_was_committed(asset_id: str) -> bool | None:
    try:
        with SessionLocal() as verification:
            return verification.get(ModelAssetInstall, asset_id) is None
    except Exception:
        logger.exception("Could not verify the auxiliary model deletion database outcome")
        return None


@router.get("/models/storage", response_model=ModelStorageInfo)
async def model_storage(request: Request, session: SessionDep) -> ModelStorageInfo:
    settings = _services(request).settings
    partials = list(settings.download_dir.glob("*.partial"))
    return ModelStorageInfo(
        installed_bytes=_path_size(settings.model_dir),
        partial_download_bytes=sum(_path_size(path) for path in partials),
        catalog_cache_bytes=_path_size(settings.catalog_cache_dir),
        installed_count=(
            session.scalar(select(func.count(ModelInstall.id)).where(ModelInstall.active.is_(True)))
            or 0
        )
        + (
            session.scalar(
                select(func.count(ModelAssetInstall.id)).where(
                    ModelAssetInstall.active.is_(True),
                    ModelAssetInstall.verified_at.is_not(None),
                )
            )
            or 0
        ),
        partial_download_count=len(partials),
    )


@router.get("/catalog/civitai/{model_id}/versions", response_model=CatalogVersions)
async def catalog_model_versions(
    model_id: str, request: Request, session: SessionDep
) -> CatalogVersions:
    """Every installable version of one model, and which are already here.

    A version is what installs, so a card that groups them still has to let
    someone choose one deliberately. This is the list that choice is made
    from; picking a row goes back through the ordinary preflight and install
    for that exact version, and nothing about the verified path changes.

    Installed state is read from the same manifest field update checks use.
    Where a kind does not record a provider version - checkpoints today - the
    answer is `null` rather than `false`: saying "not installed" about
    something we cannot see is how a person ends up with a second copy.
    """

    services = _services(request)
    source = services.catalog_sources.get("civitai")
    if not isinstance(source, CivitaiCatalog):
        raise api_error(503, "provider-unavailable", "CivitAI catalog is not available")
    if not source.validate_item_id(model_id):
        raise api_error(422, "catalog-item-id-invalid", "That is not a CivitAI model id.")
    try:
        summary = await source.versions(model_id)
    except CatalogUnavailableError as exc:
        raise api_error(503, "catalog-unavailable", str(exc)) from exc
    except ValueError as exc:
        raise api_error(404, "catalog-item-not-found", str(exc)) from exc

    installed = {
        identity.version_id: identity
        for identity in installed_civitai_identities(session)
        if identity.model_id == model_id
    }
    rows = []
    for version in summary.get("versions") or []:
        version_id = str(version.get("version_id") or "")
        match = installed.get(version_id)
        rows.append(
            CatalogVersionRow(
                version_id=version_id,
                version_name=version.get("version_name"),
                published_at=version.get("published_at"),
                base_model=version.get("base_model"),
                size_bytes=int(version.get("size_bytes") or 0),
                changelog=version.get("changelog"),
                # `False` is only safe once this model has proved it records
                # versions at all - which one recorded identity demonstrates.
                # With none, silence is the honest answer: an install that
                # stores no version is indistinguishable from no install.
                installed=True if match else (False if installed else None),
                installed_as=match.name if match else None,
            )
        )
    return CatalogVersions(
        model_id=str(summary.get("model_id") or model_id),
        model_name=summary.get("model_name"),
        versions=rows,
    )


@router.get("/models/updates", response_model=list[ModelUpdateOut])
async def check_model_updates(request: Request, session: SessionDep) -> list[ModelUpdateOut]:
    """Compare installed provider versions on demand; the request is the consent.

    Nothing polls in the background - this asks the provider only when the
    user asks, and only about models whose install manifests name an exact
    version. A provider that cannot answer yields "unknown", never a guess.
    """

    services = _services(request)
    identities = installed_civitai_identities(session)
    summaries: dict[str, dict[str, Any] | None] = {}
    if identities:
        source = services.catalog_sources.get("civitai")
        if not isinstance(source, CivitaiCatalog):  # pragma: no cover - registry invariant
            raise api_error(503, "provider-unavailable", "CivitAI catalog is not available")
        for identity in identities:
            if identity.model_id in summaries:
                continue
            try:
                summaries[identity.model_id] = await source.versions(identity.model_id)
            except Exception:  # noqa: BLE001 - one model must not hide the rest
                summaries[identity.model_id] = None
    report: list[ModelUpdateOut] = []
    for identity in identities:
        summary = summaries.get(identity.model_id)
        candidate = newer_version(identity, summary) if summary is not None else None
        state: Literal["update_available", "current", "unknown"] = (
            "unknown" if summary is None else "current" if candidate is None else "update_available"
        )
        report.append(
            ModelUpdateOut(
                install_id=identity.install_id,
                name=identity.name,
                kind=identity.kind,
                model_id=identity.model_id,
                installed_version_id=identity.version_id,
                installed_version_name=identity.version_name,
                state=state,
                update_version_id=candidate.version_id if candidate else None,
                update_version_name=candidate.version_name if candidate else None,
                update_published_at=candidate.published_at if candidate else None,
                update_base_model=candidate.base_model if candidate else None,
                update_changelog=candidate.changelog if candidate else None,
            )
        )
    return report


@router.post("/models/import", response_model=ModelInstallOut, status_code=201)
async def import_model(payload: ModelImport, session: SessionDep) -> ModelInstall:
    path = Path(payload.local_path).expanduser().resolve(strict=True)
    comfy_paths = _comfy_import_paths(path) if payload.engine == "comfyui" else {}
    blocked = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    files = [child for child in path.rglob("*") if child.is_file()] if path.is_dir() else [path]
    unsafe = [child for child in files if child.suffix.lower() in blocked]
    if unsafe:
        raise api_error(
            422, "model-file-format-blocked", "pickle-compatible model files are blocked by default"
        )
    size = sum(child.stat().st_size for child in files)
    install = ModelInstall(
        id=new_id("model"),
        name=payload.name,
        role=payload.role,
        engine=payload.engine,
        local_path=str(path),
        size_bytes=size,
        compatibility=CompatibilityLevel.ADVANCED.value,
        manifest_json={
            "imported": True,
            "path_type": "directory" if path.is_dir() else "file",
            "file_count": len(files),
            "pickle_compatible_weights": False,
            **({"comfy_paths": comfy_paths} if comfy_paths else {}),
        },
    )
    session.add(install)
    session.flush()
    ensure_profile_for_install(session, install)
    session.commit()
    session.refresh(install)
    return install


def _comfy_import_paths(path: Path) -> dict[str, str]:
    """Expose exact, ordinary ComfyUI model folders under an imported root."""

    if not path.is_dir():
        return {}
    result: dict[str, str] = {}
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if child.name not in COMFY_MODEL_FOLDERS:
            continue
        if is_link_or_reparse(child, missing="assume_link", unreadable="assume_link"):
            raise api_error(
                422,
                "model-path-unsafe",
                "ComfyUI model folders cannot use filesystem links",
            )
        if child.is_dir():
            result[child.name] = child.name
    return result


@router.post("/models/{model_id}/activate", response_model=JobOut, status_code=202)
async def activate_model(model_id: str, request: Request, session: SessionDep) -> Job:
    """Re-prove an installed model against the current runtime and hardware.

    Activation evidence is bound to the runtime, workflow contract and hardware it
    was gathered on, so an upgrade can leave a working install reporting that it
    must be rechecked. Without this the only remedy was to delete the model and
    download it again.
    """

    install = session.get(ModelInstall, model_id)
    if not install:
        raise api_error(404, "model-not-found", "model not found")
    try:
        return _services(request).downloads.reactivate(session, install)
    except ValueError as exc:
        raise api_error(422, "model-activation-invalid", str(exc)) from exc


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    request: Request,
    session: SessionDep,
    delete_profiles: bool = False,
) -> Response:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        install = session.get(ModelInstall, model_id)
        if not install:
            raise api_error(404, "model-not-found", "model not found")
        worker_name = "chat" if install.role == ModelRole.CHAT.value else "media"
        _ensure_worker_idle(session, worker_name)
        if worker_name == "media" and any(
            worker.name == "media" and worker.running for worker in services.processes.statuses()
        ):
            raise api_error(
                409, "media-worker-running", "stop the media worker before deleting this model"
            )
        quarantine = _delete_model_locked(
            model_id,
            request,
            session,
            delete_profiles=delete_profiles,
        )
        try:
            await asyncio.to_thread(_finalize_model_quarantine, quarantine)
        except Exception:
            # The database commit is authoritative. Startup recovery will
            # finish this same-volume deletion without another user action.
            logger.warning(
                "Deleted model files remain safely quarantined at %s",
                quarantine,
                exc_info=True,
            )
        return Response(status_code=204)


def _delete_model_locked(
    model_id: str,
    request: Request,
    session: Session,
    *,
    delete_profiles: bool,
) -> Path | None:
    install = session.get(ModelInstall, model_id)
    if not install:
        raise api_error(404, "model-not-found", "model not found")
    model_root = _services(request).settings.model_dir.resolve()
    try:
        path = _managed_model_path(model_root, install.local_path)
        recover_model_delete_quarantines(session, model_root, strict=True)
    except ValueError as exc:
        raise api_error(422, "model-delete-refused", str(exc)) from exc

    profiles = list(
        session.scalars(
            select(ModelProfile).where(ModelProfile.model_install_id == install.id)
        ).all()
    )
    if profiles and not delete_profiles:
        raise api_error(
            409, "model-in-use-by-profile", "delete profiles that use this model before deleting it"
        )
    profile_ids = {profile.id for profile in profiles}
    if profile_ids and any(
        worker.running and worker.profile_id in profile_ids
        for worker in _services(request).processes.statuses()
    ):
        raise api_error(
            409, "model-loaded-by-worker", "unload the active worker before deleting this model"
        )
    if profile_ids:
        affected_chats = session.scalars(
            select(Chat).where(
                or_(
                    Chat.active_chat_profile_id.in_(profile_ids),
                    Chat.active_vision_profile_id.in_(profile_ids),
                    Chat.active_image_profile_id.in_(profile_ids),
                    Chat.active_video_profile_id.in_(profile_ids),
                )
            )
        ).all()
        for chat in affected_chats:
            if chat.active_chat_profile_id in profile_ids:
                chat.active_chat_profile_id = AUTO_PROFILE_ID
            if chat.active_vision_profile_id in profile_ids:
                chat.active_vision_profile_id = AUTO_PROFILE_ID
            if chat.active_image_profile_id in profile_ids:
                chat.active_image_profile_id = AUTO_PROFILE_ID
            if chat.active_video_profile_id in profile_ids:
                chat.active_video_profile_id = AUTO_PROFILE_ID
        for profile in profiles:
            session.delete(profile)
    moves: list[tuple[Path, Path]] = []
    quarantine: Path | None = None
    commit_started = False
    try:
        # Flush relationship and chat changes before touching the filesystem.
        # A constraint failure here therefore cannot move any model content.
        session.flush()
        if path is not None:
            moves, quarantine = _quarantine_model_files(
                session,
                install,
                model_root,
                path,
            )
        session.delete(install)
        # Force database errors while the quarantined files are still
        # restorable, rather than deferring every check into commit().
        session.flush()
        commit_started = True
        session.commit()
    except Exception as exc:
        try:
            session.rollback()
        except Exception:
            logger.exception("Database rollback failed during model deletion")
        if commit_started:
            committed = _model_delete_was_committed(model_id)
            if committed is True:
                logger.warning(
                    "Model deletion commit completed despite a reported database error",
                    exc_info=True,
                )
                return quarantine
            if committed is None:
                logger.error(
                    "Could not determine model deletion outcome; leaving files in quarantine %s",
                    quarantine,
                    exc_info=True,
                )
                raise
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Model deletion rollback left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            try:
                _finalize_model_quarantine(quarantine)
            except Exception:
                logger.warning(
                    "Restored model files but could not prune quarantine %s",
                    quarantine,
                    exc_info=True,
                )
        if isinstance(exc, ValueError):
            raise api_error(422, "model-delete-refused", str(exc)) from exc
        raise
    return quarantine


_MODEL_DELETE_QUARANTINE = ".delete-pending"
_MODEL_DELETE_MARKER = ".model-id"


def _model_delete_was_committed(model_id: str) -> bool | None:
    """Resolve an ambiguous commit error from a fresh database transaction."""

    try:
        with SessionLocal() as verification:
            return verification.get(ModelInstall, model_id) is None
    except Exception:
        logger.exception("Could not verify the model deletion database outcome")
        return None


def _managed_model_path(model_root: Path, value: str) -> Path | None:
    """Return a confined, link-free managed path or None for external imports."""

    raw = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    try:
        relative = raw.relative_to(model_root)
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] == _MODEL_DELETE_QUARANTINE:
        return None
    cursor = model_root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and _model_path_is_link(cursor):
            raise ValueError("managed model paths cannot use filesystem links")
    resolved = raw.resolve(strict=False)
    if model_root not in resolved.parents or resolved == model_root:
        raise ValueError("managed model path escapes model storage")
    return resolved


def _model_path_is_link(path: Path) -> bool:
    return is_link_or_reparse(
        path,
        missing="assume_regular",
        unreadable="assume_link",
    )


def _ensure_model_tree_link_free(path: Path) -> None:
    if _model_path_is_link(path):
        raise ValueError("managed model paths cannot use filesystem links")
    if not path.is_dir():
        return
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                candidate = Path(entry.path)
                if _model_path_is_link(candidate):
                    raise ValueError("managed model directories cannot contain filesystem links")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(candidate)


def _quarantine_model_files(
    session: Session,
    install: ModelInstall,
    model_root: Path,
    path: Path,
) -> tuple[list[tuple[Path, Path]], Path | None]:
    if not path.exists():
        return [], None
    siblings = _related_model_installs(session, install, model_root, path)
    quarantine: Path | None = None
    moves: list[tuple[Path, Path]] = []
    try:
        if not siblings:
            _ensure_model_tree_link_free(path)
            quarantine = _new_model_quarantine(model_root, install.id)
            staged = quarantine / "payload"
            os.replace(path, staged)
            return [(staged, path)], quarantine

        if not path.is_dir():
            # Another install owns the same managed file.
            return [], None

        retained_paths, retained_roots = _retained_model_paths(
            siblings,
            model_root,
        )
        for relative in _manifest_model_files(install):
            candidate = _confined_manifest_file(path, relative)
            if not candidate.exists():
                continue
            if _model_path_is_link(candidate):
                raise ValueError("managed model files cannot use filesystem links")
            if not candidate.is_file():
                raise ValueError("managed model manifests may only reference files")
            if candidate in retained_paths or any(
                candidate == root or root in candidate.parents for root in retained_roots
            ):
                continue
            if quarantine is None:
                quarantine = _new_model_quarantine(model_root, install.id)
            staged = _safe_quarantine_file_path(quarantine, relative)
            os.replace(candidate, staged)
            moves.append((staged, candidate))
        return moves, quarantine
    except Exception:
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Model staging failure left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            try:
                _finalize_model_quarantine(quarantine)
            except Exception:
                logger.warning(
                    "Could not prune a failed model deletion quarantine %s",
                    quarantine,
                    exc_info=True,
                )
        raise


def _related_model_installs(
    session: Session,
    install: ModelInstall,
    model_root: Path,
    path: Path,
) -> list[tuple[ModelInstall, Path]]:
    related: list[tuple[ModelInstall, Path]] = []
    for sibling in session.scalars(select(ModelInstall).where(ModelInstall.id != install.id)).all():
        try:
            sibling_path = _managed_model_path(model_root, sibling.local_path)
        except ValueError:
            # A linked sibling is never evidence that it is safe to remove
            # content through the current install.
            continue
        if sibling_path is None:
            continue
        if sibling_path == path or sibling_path in path.parents or path in sibling_path.parents:
            related.append((sibling, sibling_path))
    return related


def _retained_model_paths(
    siblings: list[tuple[ModelInstall, Path]],
    model_root: Path,
) -> tuple[set[Path], set[Path]]:
    retained_paths: set[Path] = set()
    retained_roots: set[Path] = set()
    for sibling, sibling_path in siblings:
        if sibling_path.is_file():
            retained_paths.add(sibling_path)
            continue
        try:
            files = _manifest_model_files(sibling)
        except ValueError:
            retained_roots.add(sibling_path)
            continue
        if not files:
            retained_roots.add(sibling_path)
            continue
        for relative in files:
            try:
                candidate = _confined_manifest_file(sibling_path, relative)
            except ValueError:
                retained_roots.add(sibling_path)
                break
            if candidate == model_root or model_root not in candidate.parents:
                retained_roots.add(sibling_path)
                break
            retained_paths.add(candidate)
    return retained_paths, retained_roots


def _manifest_model_files(install: ModelInstall) -> list[PurePosixPath]:
    raw_files = install.manifest_json.get("files", [])
    if not isinstance(raw_files, list):
        raise ValueError("managed model manifest files must be a list")
    files: list[PurePosixPath] = []
    seen: set[str] = set()
    for value in raw_files:
        if not isinstance(value, str) or not value:
            raise ValueError("managed model manifest contains an invalid file path")
        relative = PurePosixPath(value.replace("\\", "/"))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
        ):
            raise ValueError("managed model manifest contains an unsafe file path")
        identity = relative.as_posix().casefold()
        if identity in seen:
            continue
        seen.add(identity)
        files.append(relative)
    return files


def _confined_manifest_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and _model_path_is_link(cursor):
            raise ValueError("managed model files cannot use filesystem links")
    resolved = candidate.resolve(strict=False)
    if root not in resolved.parents:
        raise ValueError("managed model manifest path escapes its install directory")
    return resolved


def _new_model_quarantine(model_root: Path, model_id: str) -> Path:
    parent = model_root / _MODEL_DELETE_QUARANTINE
    if parent.exists() and _model_path_is_link(parent):
        raise ValueError("model deletion quarantine cannot use a filesystem link")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.resolve().parent != model_root:
        raise ValueError("model deletion quarantine escapes model storage")
    quarantine = parent / new_id("delete")
    quarantine.mkdir(mode=0o700)
    try:
        marker = quarantine / _MODEL_DELETE_MARKER
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(model_id)
            handle.flush()
            os.fsync(handle.fileno())
        marker.chmod(0o600)
    except Exception:
        with suppress(OSError):
            quarantine.rmdir()
        raise
    return quarantine


def _safe_quarantine_file_path(
    quarantine: Path,
    relative: PurePosixPath,
) -> Path:
    if _model_path_is_link(quarantine) or not quarantine.is_dir():
        raise ValueError("model deletion quarantine contains a filesystem link")
    files = quarantine / "files"
    if _model_path_is_link(files):
        raise ValueError("model deletion quarantine contains a filesystem link")
    if not files.exists():
        files.mkdir()
    if _model_path_is_link(files) or not files.is_dir():
        raise ValueError("model deletion quarantine contains an unsafe files directory")
    files_root = files.resolve()
    if files_root.parent != quarantine.resolve():
        raise ValueError("model deletion quarantine escapes model storage")
    cursor = files
    for part in relative.parts[:-1]:
        cursor /= part
        if _model_path_is_link(cursor):
            raise ValueError("model deletion quarantine contains a filesystem link")
        if cursor.exists():
            if not cursor.is_dir():
                raise ValueError("model deletion quarantine contains a filesystem link")
        else:
            cursor.mkdir()
            if _model_path_is_link(cursor) or not cursor.is_dir():
                raise ValueError("model deletion quarantine contains a filesystem link")
    staged = cursor / relative.name
    if staged.exists() or _model_path_is_link(staged):
        raise ValueError("model deletion quarantine contains an unexpected file")
    resolved_parent = staged.parent.resolve()
    if resolved_parent != files_root and files_root not in resolved_parent.parents:
        raise ValueError("model deletion quarantine escapes model storage")
    return staged


def _restore_model_moves(moves: list[tuple[Path, Path]]) -> None:
    for staged, original in reversed(moves):
        if not staged.exists():
            continue
        if original.exists():
            raise RuntimeError("cannot restore a quarantined model over an existing path")
        original.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, original)


def _finalize_model_quarantine(quarantine: Path | None) -> None:
    if quarantine is None or not quarantine.exists():
        return
    if _model_path_is_link(quarantine) or not quarantine.is_dir():
        raise OSError("refusing to follow an unsafe model deletion quarantine")
    marker = quarantine / _MODEL_DELETE_MARKER
    if _model_path_is_link(marker) or not marker.is_file():
        raise OSError("model deletion quarantine has no safe ownership marker")
    for child in quarantine.iterdir():
        if child == marker:
            continue
        if _model_path_is_link(child):
            raise OSError("refusing to follow a link in model deletion quarantine")
        if child.is_dir():
            try:
                _ensure_model_tree_link_free(child)
            except ValueError as exc:
                raise OSError("refusing to follow a link in model deletion quarantine") from exc
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()
        else:
            raise OSError("model deletion quarantine contains an unsupported entry")
    # Remove the ownership marker last. If payload cleanup fails partway
    # through, the next recovery pass can still determine whether to restore
    # the remaining files or finish deleting them.
    marker.unlink(missing_ok=True)
    quarantine.rmdir()
    with suppress(OSError):
        quarantine.parent.rmdir()


def recover_model_delete_quarantines(
    session: Session,
    model_root: Path,
    *,
    strict: bool = False,
) -> None:
    parent = model_root / _MODEL_DELETE_QUARANTINE
    if not parent.exists():
        return
    if _model_path_is_link(parent) or not parent.is_dir():
        raise ValueError("model deletion quarantine is not a safe directory")
    for quarantine in list(parent.iterdir()):
        if not quarantine.name.startswith("delete_"):
            continue
        try:
            if not quarantine.is_dir() or _model_path_is_link(quarantine):
                raise ValueError("model deletion quarantine contains a filesystem link")
            marker = quarantine / _MODEL_DELETE_MARKER
            if not marker.exists():
                if not any(quarantine.iterdir()):
                    quarantine.rmdir()
                    continue
                raise ValueError("model deletion quarantine has no ownership marker")
            if _model_path_is_link(marker) or not marker.is_file():
                raise ValueError("model deletion quarantine contains an unsafe marker")
            marker_model_id = marker.read_text(encoding="utf-8")
            if (
                not marker_model_id
                or marker_model_id != marker_model_id.strip()
                or len(marker_model_id) > 200
            ):
                raise ValueError("model deletion quarantine has an invalid owner")
            install: ModelInstall | ModelAssetInstall | None = session.get(
                ModelInstall, marker_model_id
            )
            if install is None:
                install = session.get(ModelAssetInstall, marker_model_id)
            if install is None:
                _finalize_model_quarantine(quarantine)
                continue
            path = _managed_model_path(model_root, install.local_path)
            if path is None:
                raise ValueError("model deletion quarantine belongs to an external model")
            _restore_model_quarantine(quarantine, path)
        except (OSError, UnicodeError, ValueError):
            if strict:
                raise
            logger.warning(
                "Could not reconcile model deletion quarantine %s",
                quarantine,
                exc_info=True,
            )
    with suppress(OSError):
        parent.rmdir()


def _restore_model_quarantine(quarantine: Path, path: Path) -> None:
    moves: list[tuple[Path, Path]] = []
    payload = quarantine / "payload"
    if payload.exists():
        _ensure_model_tree_link_free(payload)
        moves.append((payload, path))
    files = quarantine / "files"
    if files.exists():
        if _model_path_is_link(files) or not files.is_dir():
            raise ValueError("model deletion quarantine contains an unsafe files directory")
        _ensure_model_tree_link_free(files)
        for staged in files.rglob("*"):
            if _model_path_is_link(staged):
                raise ValueError("model deletion quarantine contains a filesystem link")
            if not staged.is_file():
                continue
            relative = staged.relative_to(files)
            original = _confined_manifest_file(
                path,
                PurePosixPath(*relative.parts),
            )
            moves.append((staged, original))
    try:
        _restore_model_moves(moves)
    except RuntimeError as exc:
        raise ValueError("model deletion recovery needs manual conflict resolution") from exc
    _finalize_model_quarantine(quarantine)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@router.get("/profiles", response_model=list[ModelProfileOut])
async def list_profiles(
    request: Request,
    session: SessionDep,
    role: str | None = None,
) -> list[ModelProfileOut]:
    statement = (
        select(ModelProfile)
        .outerjoin(ModelInstall, ModelInstall.id == ModelProfile.model_install_id)
        .where(
            or_(
                ModelProfile.model_install_id.is_(None),
                and_(
                    ModelInstall.active.is_(True),
                    ModelInstall.role == ModelProfile.role,
                    ModelInstall.engine == ModelProfile.engine,
                ),
            )
        )
        .order_by(ModelProfile.role, ModelProfile.name)
    )
    if role:
        statement = statement.where(ModelProfile.role == role)
    services = _services(request)
    results: list[ModelProfileOut] = []
    for profile in session.scalars(statement).all():
        evidence = None
        if profile.model_install_id:
            install = session.get(ModelInstall, profile.model_install_id)
            if install:
                evidence = current_capability_evidence(
                    session,
                    install,
                    services.settings,
                    services.runtimes,
                )
        results.append(
            ModelProfileOut.model_validate(profile).model_copy(
                update={"input_modalities": evidence_input_modalities(evidence)}
            )
        )
    return results


@router.post("/profiles", response_model=ModelProfileOut, status_code=201)
async def create_profile(
    payload: ModelProfileCreate,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    _validated_profile_install(
        session,
        model_install_id=payload.model_install_id,
        role=payload.role,
        engine=payload.engine,
    )
    fields = await _engine_role_fields(
        request,
        payload.role,
        engine=payload.engine,
        allow_inactive=True,
    )
    if payload.is_default:
        for profile in session.scalars(
            select(ModelProfile).where(ModelProfile.role == payload.role)
        ).all():
            profile.is_default = False
    try:
        load_settings = validate_settings(
            payload.load_settings, [field for field in fields if field.scope == "load"]
        )
        request_settings = validate_settings(
            payload.request_settings, [field for field in fields if field.scope != "load"]
        )
    except ValueError as exc:
        raise api_error(422, "profile-settings-invalid", str(exc)) from exc
    profile = ModelProfile(
        name=payload.name,
        use_case=payload.use_case,
        role=payload.role,
        engine=payload.engine,
        model_install_id=payload.model_install_id,
        load_settings_json=load_settings,
        request_settings_json=request_settings,
        is_default=payload.is_default,
    )
    session.add(profile)
    session.flush()
    ensure_legacy_profile_workflow(session, profile)
    session.commit()
    session.refresh(profile)
    return profile


@router.patch("/profiles/{profile_id}", response_model=ModelProfileOut)
async def update_profile(
    profile_id: str,
    payload: ModelProfileUpdate,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise api_error(404, "profile-not-found", "profile not found")
    try:
        validate_profile_binding(session, profile)
    except LookupError as exc:
        raise api_error(404, "profile-not-found", str(exc)) from exc
    except ValueError as exc:
        raise api_error(422, "profile-binding-invalid", str(exc)) from exc
    values = payload.model_dump(exclude_unset=True)
    fields = (
        await _engine_role_fields(
            request,
            profile.role,
            engine=profile.engine,
            allow_inactive=True,
        )
        if {"load_settings", "request_settings"} & values.keys()
        else []
    )
    if "is_default" in values:
        is_default = bool(values.pop("is_default"))
        if is_default:
            for sibling in session.scalars(
                select(ModelProfile).where(ModelProfile.role == profile.role)
            ).all():
                sibling.is_default = False
        profile.is_default = is_default
    if "load_settings" in values:
        try:
            profile.load_settings_json = validate_settings(
                values.pop("load_settings") or {},
                [field for field in fields if field.scope == "load"],
            )
        except ValueError as exc:
            raise api_error(422, "profile-load-settings-invalid", str(exc)) from exc
    if "request_settings" in values:
        try:
            profile.request_settings_json = validate_settings(
                values.pop("request_settings") or {},
                [field for field in fields if field.scope != "load"],
            )
        except ValueError as exc:
            raise api_error(422, "profile-request-settings-invalid", str(exc)) from exc
    for key, value in values.items():
        setattr(profile, key, value)
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    session.refresh(profile)
    return profile


@router.delete("/profiles/{profile_id}", status_code=204)
async def delete_profile(profile_id: str, request: Request, session: SessionDep) -> Response:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        profile = session.get(ModelProfile, profile_id)
        if not profile:
            raise api_error(404, "profile-not-found", "profile not found")
        worker_name = "chat" if profile.role == ModelRole.CHAT.value else "media"
        _ensure_worker_idle(session, worker_name)
        if any(
            worker.running and worker.profile_id == profile.id
            for worker in services.processes.statuses()
        ):
            raise api_error(
                409,
                "profile-loaded-by-worker",
                "unload the active worker before deleting its profile",
            )
        retire_legacy_profile_workflow(session, profile)
        session.delete(profile)
        session.flush()
        reconcile_legacy_workflow_compatibility(session)
        session.commit()
        return Response(status_code=204)


@router.post("/profiles/{profile_id}/clone", response_model=ModelProfileOut, status_code=201)
async def clone_profile(
    profile_id: str,
    payload: ModelProfileClone,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    source = session.get(ModelProfile, profile_id)
    if not source:
        raise api_error(404, "profile-not-found", "profile not found")
    return await create_profile(
        ModelProfileCreate(
            name=payload.name or f"{source.name} copy",
            use_case=source.use_case,
            role=cast(Literal["chat", "image", "video"], source.role),
            engine=source.engine,
            model_install_id=source.model_install_id,
            load_settings=source.load_settings_json,
            request_settings=source.request_settings_json,
        ),
        request,
        session,
    )


@router.post("/profiles/{profile_id}/reset", response_model=ModelProfileOut)
async def reset_profile(profile_id: str, session: SessionDep) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise api_error(404, "profile-not-found", "profile not found")
    try:
        validate_profile_binding(session, profile)
    except LookupError as exc:
        raise api_error(404, "profile-not-found", str(exc)) from exc
    except ValueError as exc:
        raise api_error(422, "profile-settings-invalid", str(exc)) from exc
    profile.load_settings_json = {}
    profile.request_settings_json = {}
    ensure_legacy_profile_workflow(session, profile)
    session.commit()
    session.refresh(profile)
    return profile


@router.get("/profiles/{profile_id}/export", response_model=ModelProfileBundle)
async def export_profile(profile_id: str, session: SessionDep) -> ModelProfileBundle:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise api_error(404, "profile-not-found", "profile not found")
    return ModelProfileBundle(
        name=profile.name,
        use_case=profile.use_case,
        role=cast(Literal["chat", "image", "video"], profile.role),
        engine=profile.engine,
        model_install_id=profile.model_install_id,
        load_settings=profile.load_settings_json,
        request_settings=profile.request_settings_json,
    )


@router.post("/profiles/import", response_model=ModelProfileOut, status_code=201)
async def import_profile(
    payload: ModelProfileBundle,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    return await create_profile(
        ModelProfileCreate(
            name=payload.name,
            use_case=payload.use_case,
            role=payload.role,
            engine=payload.engine,
            model_install_id=payload.model_install_id,
            load_settings=payload.load_settings,
            request_settings=payload.request_settings,
        ),
        request,
        session,
    )


@router.get("/presets", response_model=list[PresetOut])
async def list_presets(session: SessionDep, role: str | None = None) -> list[GenerationPreset]:
    statement = select(GenerationPreset).order_by(GenerationPreset.role, GenerationPreset.name)
    if role:
        statement = statement.where(GenerationPreset.role == role)
    return list(session.scalars(statement).all())


@router.post("/presets", response_model=PresetOut, status_code=201)
async def create_preset(
    payload: PresetCreate,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    fields = await _engine_role_fields(request, payload.role)
    try:
        values = validate_settings(
            payload.settings,
            [field for field in fields if field.scope != "load"],
        )
    except ValueError as exc:
        raise api_error(422, "preset-invalid", str(exc)) from exc
    if payload.is_default:
        for sibling in session.scalars(
            select(GenerationPreset).where(GenerationPreset.role == payload.role)
        ).all():
            sibling.is_default = False
    preset = GenerationPreset(
        name=payload.name,
        role=payload.role,
        settings_json=values,
        is_default=payload.is_default,
    )
    session.add(preset)
    session.commit()
    session.refresh(preset)
    return preset


@router.patch("/presets/{preset_id}", response_model=PresetOut)
async def update_preset(
    preset_id: str,
    payload: PresetUpdate,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise api_error(404, "preset-not-found", "preset not found")
    values = payload.model_dump(exclude_unset=True)
    if "settings" in values:
        fields = await _engine_role_fields(request, preset.role)
        try:
            preset.settings_json = validate_settings(
                values.pop("settings") or {},
                [field for field in fields if field.scope != "load"],
            )
        except ValueError as exc:
            raise api_error(422, "preset-invalid", str(exc)) from exc
    if "is_default" in values:
        is_default = bool(values.pop("is_default"))
        if is_default:
            for sibling in session.scalars(
                select(GenerationPreset).where(GenerationPreset.role == preset.role)
            ).all():
                sibling.is_default = False
        preset.is_default = is_default
    for key, value in values.items():
        setattr(preset, key, value)
    session.commit()
    session.refresh(preset)
    return preset


@router.delete("/presets/{preset_id}", status_code=204)
async def delete_preset(preset_id: str, session: SessionDep) -> Response:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise api_error(404, "preset-not-found", "preset not found")
    owners: list[Project | Chat] = []
    owners.extend(session.scalars(select(Project)).all())
    owners.extend(session.scalars(select(Chat)).all())
    for owner in owners:
        bindings = (
            dict(owner.generation_preset_ids_json)
            if isinstance(owner.generation_preset_ids_json, dict)
            else {}
        )
        if bindings.get(preset.role) != preset.id:
            continue
        bindings.pop(preset.role, None)
        scoped = (
            dict(owner.generation_settings_json)
            if isinstance(owner.generation_settings_json, dict)
            else {}
        )
        direct = scoped.get(preset.role)
        scoped[preset.role] = {
            **preset.settings_json,
            **(direct if isinstance(direct, dict) else {}),
        }
        owner.generation_preset_ids_json = bindings
        owner.generation_settings_json = scoped
    session.delete(preset)
    session.commit()
    return Response(status_code=204)


@router.post("/presets/{preset_id}/clone", response_model=PresetOut, status_code=201)
async def clone_preset(
    preset_id: str,
    payload: PresetClone,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    source = session.get(GenerationPreset, preset_id)
    if not source:
        raise api_error(404, "preset-not-found", "preset not found")
    return await create_preset(
        PresetCreate(
            name=payload.name or f"{source.name} copy",
            role=cast(Literal["chat", "image", "video"], source.role),
            settings=source.settings_json,
        ),
        request,
        session,
    )


@router.post("/presets/{preset_id}/reset", response_model=PresetOut)
async def reset_preset(preset_id: str, session: SessionDep) -> GenerationPreset:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise api_error(404, "preset-not-found", "preset not found")
    preset.settings_json = {}
    session.commit()
    session.refresh(preset)
    return preset


@router.get("/presets/{preset_id}/export", response_model=PresetBundle)
async def export_preset(preset_id: str, session: SessionDep) -> PresetBundle:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise api_error(404, "preset-not-found", "preset not found")
    return PresetBundle(
        name=preset.name,
        role=cast(Literal["chat", "image", "video"], preset.role),
        settings=preset.settings_json,
    )


@router.post("/presets/import", response_model=PresetOut, status_code=201)
async def import_preset(
    payload: PresetBundle,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    return await create_preset(
        PresetCreate(name=payload.name, role=payload.role, settings=payload.settings),
        request,
        session,
    )


def _require_media_worker_stopped(request: Request) -> None:
    if any(
        worker.name == "media" and worker.running
        for worker in _services(request).processes.statuses()
    ):
        raise api_error(
            409, "media-worker-running", "stop the media worker before changing custom nodes"
        )


async def _custom_node_lifecycle(
    request: Request,
    session: SessionDep,
) -> AsyncIterator[None]:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        _ensure_worker_idle(session, "media")
        _require_media_worker_stopped(request)
        yield


CustomNodeLifecycleDep = Annotated[None, Depends(_custom_node_lifecycle)]


@router.get("/custom-nodes", response_model=list[CustomNodeOut])
async def list_custom_nodes(session: SessionDep) -> list[CustomNodeInstall]:
    return list(session.scalars(select(CustomNodeInstall).order_by(CustomNodeInstall.name)).all())


@router.post("/custom-nodes", response_model=CustomNodeOut, status_code=201)
async def install_custom_node(
    payload: CustomNodeInstallRequest,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
    try:
        source_url = _services(request).custom_nodes.normalize_source(payload.source_url)
    except ValueError as exc:
        raise api_error(422, "custom-node-request-invalid", str(exc)) from exc
    existing = session.scalar(
        select(CustomNodeInstall).where(CustomNodeInstall.source_url == source_url)
    )
    if existing:
        raise api_error(
            409, "custom-node-source-duplicate", "this custom node source is already managed"
        )
    try:
        install = await _services(request).custom_nodes.install(
            session,
            name=payload.name,
            source_url=source_url,
            revision=payload.revision,
        )
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        session.rollback()
        raise api_error(422, "custom-node-install-failed", str(exc)) from exc
    session.commit()
    session.refresh(install)
    return install


@router.patch("/custom-nodes/{node_id}", response_model=CustomNodeOut)
async def update_custom_node(
    node_id: str,
    payload: CustomNodeUpdateRequest,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise api_error(404, "custom-node-install-not-found", "custom node install not found")
    try:
        await _services(request).custom_nodes.update(install, payload.revision)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        session.rollback()
        raise api_error(422, "custom-node-update-failed", str(exc)) from exc
    session.commit()
    session.refresh(install)
    return install


@router.post("/custom-nodes/{node_id}/trust", response_model=CustomNodeOut)
async def trust_custom_node(
    node_id: str,
    payload: CustomNodeTrustRequest,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise api_error(404, "custom-node-install-not-found", "custom node install not found")
    reviewed_node_types: tuple[str, ...] | None = None
    if payload.trusted:
        try:
            await _services(request).custom_nodes.verify(install)
            reviewed_node_types = reviewed_custom_node_types({"node_types": payload.node_types})
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            raise api_error(422, "custom-node-trust-failed", str(exc)) from exc
    install.trusted = payload.trusted
    install.security_json = {
        **install.security_json,
        **({"node_types": list(reviewed_node_types)} if reviewed_node_types is not None else {}),
        "reviewed_at": utcnow().isoformat(),
        "trusted_by_local_user": payload.trusted,
    }
    session.commit()
    session.refresh(install)
    return install


@router.post("/custom-nodes/{node_id}/rollback", response_model=CustomNodeOut)
async def rollback_custom_node(
    node_id: str,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise api_error(404, "custom-node-install-not-found", "custom node install not found")
    try:
        await _services(request).custom_nodes.rollback(install)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        session.rollback()
        raise api_error(422, "custom-node-rollback-failed", str(exc)) from exc
    session.commit()
    session.refresh(install)
    return install


@router.delete("/custom-nodes/{node_id}", status_code=204)
async def remove_custom_node(
    node_id: str,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> Response:
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise api_error(404, "custom-node-install-not-found", "custom node install not found")
    _services(request).custom_nodes.remove(install)
    session.delete(install)
    session.commit()
    return Response(status_code=204)


_CHAT_WORKFLOW_PROFILE_FIELDS: dict[WorkflowSelectorCapability, str] = {
    "chat": "active_chat_profile_id",
    "vision": "active_vision_profile_id",
    "image": "active_image_profile_id",
    "video": "active_video_profile_id",
}
_PROJECT_WORKFLOW_REVISION_FIELDS = {
    "image": "image_workflow_revision_id",
    "video": "video_workflow_revision_id",
}
_SELECTOR_OPERATIONS: dict[WorkflowSelectorCapability, frozenset[Operation]] = {
    "chat": frozenset({Operation.TEXT}),
    "vision": frozenset({Operation.TEXT}),
    "image": frozenset({Operation.TEXT_TO_IMAGE, Operation.IMAGE_TO_IMAGE}),
    "video": frozenset({Operation.TEXT_TO_VIDEO, Operation.IMAGE_TO_VIDEO}),
}
_WORKFLOW_SELECTION_RESPONSE_MODES: frozenset[WorkflowSelectionResponseMode] = frozenset(
    {"default", "inherit", "automatic", "family", "revision", "legacy"}
)


def _workflow_selection_response_mode(value: str) -> WorkflowSelectionResponseMode:
    if value not in _WORKFLOW_SELECTION_RESPONSE_MODES:
        raise api_error(500, "workflow-selection-corrupt", "workflow selection mode is invalid")
    return value


def _workflow_selector_capability(value: str) -> WorkflowSelectorCapability:
    if value not in _SELECTOR_OPERATIONS:
        raise api_error(
            500,
            "workflow-preference-corrupt",
            "workflow preference selector capability is invalid",
        )
    return value


def _workflow_preference(
    session: Session,
    family_id: str,
    capability: WorkflowSelectorCapability,
) -> WorkflowPreference | None:
    return session.scalar(
        select(WorkflowPreference).where(
            WorkflowPreference.workflow_family_id == family_id,
            WorkflowPreference.selector_capability == capability,
        )
    )


def _selectable_workflow_family(
    session: Session,
    family_id: str,
    capability: WorkflowSelectorCapability,
) -> WorkflowFamily:
    family = session.get(WorkflowFamily, family_id)
    if family is None:
        raise api_error(404, "workflow-family-not-found", "workflow family not found")
    preference = _workflow_preference(session, family.id, capability)
    if family.archived or not family.enabled or preference is None or not preference.enabled:
        raise api_error(
            422,
            "workflow-family-not-selectable",
            "workflow family is not enabled for this selector",
        )
    operations = {operation.value for operation in _SELECTOR_OPERATIONS[capability]}
    compatible_definition = session.scalar(
        select(WorkflowDefinition.id)
        .where(
            WorkflowDefinition.family_id == family.id,
            WorkflowDefinition.operation.in_(operations),
        )
        .limit(1)
    )
    if compatible_definition is None:
        raise api_error(
            422,
            "workflow-family-operation-unavailable",
            "workflow family has no compatible operation variant",
        )
    return family


def _chat_workflow_selection_out(
    session: Session,
    chat: Chat,
    capability: WorkflowSelectorCapability,
) -> WorkflowSelectionOut:
    selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == chat.id,
            ChatWorkflowSelection.selector_capability == capability,
        )
    )
    if selection is not None:
        return WorkflowSelectionOut(
            selector_capability=capability,
            mode=_workflow_selection_response_mode(selection.mode),
            workflow_family_id=selection.workflow_family_id,
        )
    legacy_profile_id = getattr(chat, _CHAT_WORKFLOW_PROFILE_FIELDS[capability])
    if legacy_profile_id is None:
        mode: WorkflowSelectionResponseMode = "default"
    elif legacy_profile_id == AUTO_PROFILE_ID:
        mode = "automatic"
    else:
        mode = "legacy"
    return WorkflowSelectionOut(
        selector_capability=capability,
        mode=mode,
        legacy_profile_id=legacy_profile_id if mode == "legacy" else None,
    )


def _project_workflow_selection_out(
    session: Session,
    project: Project,
    capability: Literal["image", "video"],
) -> WorkflowSelectionOut:
    selection = session.scalar(
        select(ProjectWorkflowSelection).where(
            ProjectWorkflowSelection.project_id == project.id,
            ProjectWorkflowSelection.selector_capability == capability,
        )
    )
    if selection is None:
        legacy_revision_id = getattr(project, _PROJECT_WORKFLOW_REVISION_FIELDS[capability])
        return WorkflowSelectionOut(
            selector_capability=capability,
            mode="revision" if legacy_revision_id else "inherit",
            workflow_revision_id=legacy_revision_id,
        )
    return WorkflowSelectionOut(
        selector_capability=capability,
        mode=_workflow_selection_response_mode(selection.mode),
        workflow_family_id=selection.workflow_family_id,
        workflow_revision_id=selection.workflow_revision_id,
    )


def _revision_readiness(
    session: Session,
    revision: WorkflowRevision,
    *,
    expected_engine: str,
    operation: Operation,
) -> tuple[Literal["ready", "setup_required", "review_required", "unavailable"], str | None]:
    if revision.engine != expected_engine:
        return "unavailable", "engine_mismatch"
    if operation != Operation.TEXT and expected_engine != "mock" and not revision.api_graph_json:
        return "unavailable", "revision_not_executable"
    if not revision.trusted:
        return "review_required", "revision_untrusted"
    if revision.dependency_contract_sha256 is None:
        return "ready", None
    activation = session.scalar(
        select(WorkflowActivation).where(
            WorkflowActivation.workflow_revision_id == revision.id,
            WorkflowActivation.is_active.is_(True),
            WorkflowActivation.state == "ready",
            WorkflowActivation.invalidated_at.is_(None),
        )
    )
    launch_sha256 = (
        activation.details_json.get("launch_sha256")
        if activation is not None and isinstance(activation.details_json, dict)
        else None
    )
    if (
        activation is None
        or activation.dependency_contract_sha256 != revision.dependency_contract_sha256
        or not isinstance(launch_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", launch_sha256) is None
    ):
        return "setup_required", "activation_not_ready"
    return "ready", None


def _workflow_family_variant_out(
    session: Session,
    services: Services,
    family: WorkflowFamily,
    definition: WorkflowDefinition,
    compatibility: WorkflowProfileCompatibility | None,
    ready_offers: Mapping[str, tuple[WorkflowInstallOffer, ...]],
) -> WorkflowFamilyVariantOut:
    readiness: WorkflowVariantReadiness
    reason: str | None
    operation = Operation(definition.operation)
    expected_engine = (
        services.settings.chat_engine
        if operation == Operation.TEXT
        else services.settings.media_engine
    )
    revision = (
        session.get(WorkflowRevision, definition.current_revision_id)
        if definition.current_revision_id
        else None
    )
    if compatibility is not None and revision is None:
        profile = session.get(ModelProfile, compatibility.model_profile_id)
        if profile is not None:
            if operation == Operation.TEXT:
                install = (
                    session.get(ModelInstall, profile.model_install_id)
                    if profile.model_install_id
                    else None
                )
                profile_ready = profile.engine == expected_engine and (
                    expected_engine == "mock" or (install is not None and install.active)
                )
                if profile_ready:
                    readiness, reason = "ready", None
                else:
                    readiness, reason = "setup_required", "model_unavailable"
            else:
                revision = services.orchestrator.legacy_workflow_revision(
                    session,
                    profile,
                    operation,
                )
                if revision is None:
                    readiness, reason = "setup_required", "operation_unavailable"
                else:
                    readiness, reason = _revision_readiness(
                        session,
                        revision,
                        expected_engine=expected_engine,
                        operation=operation,
                    )
        else:
            readiness, reason = "setup_required", "model_unavailable"
    elif revision is None or revision.workflow_id != definition.id:
        readiness, reason = "setup_required", "current_revision_missing"
        revision = None
    else:
        readiness, reason = _revision_readiness(
            session,
            revision,
            expected_engine=expected_engine,
            operation=operation,
        )
    if family.archived:
        readiness, reason = "unavailable", "family_archived"
    elif not family.enabled:
        readiness, reason = "unavailable", "family_disabled"
    setup_resolution: Literal["reviewed_download_available", "attention_required"] | None = None
    install_offer_id: str | None = None
    if readiness == "setup_required":
        setup_resolution = "attention_required"
        if revision is not None:
            matching_offers = [
                offer
                for offer in ready_offers.get(revision.id, ())
                if offer.workflow_artifact_sha256 == revision.artifact_sha256
                and offer.dependency_contract_sha256 == revision.dependency_contract_sha256
            ]
            if len(matching_offers) == 1:
                setup_resolution = "reviewed_download_available"
                install_offer_id = matching_offers[0].id
    return WorkflowFamilyVariantOut(
        id=definition.id,
        variant_key=definition.variant_key or "",
        name=definition.name,
        operation=operation,
        current_revision_id=revision.id if revision else None,
        current_revision_version=revision.version if revision else None,
        engine=revision.engine if revision else None,
        capabilities=list(revision.capabilities_json) if revision else [],
        trusted=revision.trusted if revision else compatibility is not None,
        readiness=readiness,
        readiness_reason=reason,
        setup_resolution=setup_resolution,
        install_offer_id=install_offer_id,
    )


def _ready_workflow_install_offers(
    session: Session,
    *,
    family_id: str | None = None,
    selector_capability: WorkflowSelectorCapability | None = None,
    include_archived: bool = True,
) -> dict[str, tuple[WorkflowInstallOffer, ...]]:
    query = (
        select(WorkflowInstallOffer)
        .join(
            WorkflowRevision,
            WorkflowRevision.id == WorkflowInstallOffer.workflow_revision_id,
        )
        .join(
            WorkflowDefinition,
            and_(
                WorkflowDefinition.id == WorkflowRevision.workflow_id,
                WorkflowDefinition.current_revision_id == WorkflowRevision.id,
            ),
        )
        .join(WorkflowFamily, WorkflowFamily.id == WorkflowDefinition.family_id)
        .where(
            WorkflowInstallOffer.status == "ready",
            WorkflowInstallOffer.invalidated_at.is_(None),
        )
    )
    if family_id is not None:
        query = query.where(WorkflowFamily.id == family_id)
    else:
        if not include_archived:
            query = query.where(WorkflowFamily.archived.is_(False))
        if selector_capability is not None:
            query = query.join(WorkflowPreference).where(
                WorkflowPreference.selector_capability == selector_capability
            )
    grouped: dict[str, list[WorkflowInstallOffer]] = {}
    for offer in session.scalars(
        query.order_by(WorkflowInstallOffer.workflow_revision_id, WorkflowInstallOffer.id)
    ).unique():
        grouped.setdefault(offer.workflow_revision_id, []).append(offer)
    return {revision_id: tuple(offers) for revision_id, offers in grouped.items()}


def _workflow_family_out(
    session: Session,
    services: Services,
    family: WorkflowFamily,
    ready_offers: Mapping[str, tuple[WorkflowInstallOffer, ...]],
) -> WorkflowFamilyOut:
    compatibility = session.scalar(
        select(WorkflowProfileCompatibility).where(
            WorkflowProfileCompatibility.workflow_family_id == family.id
        )
    )
    return WorkflowFamilyOut(
        id=family.id,
        name=family.name,
        description=family.description,
        use_case=family.use_case,
        tags=[item for item in family.tags_json if isinstance(item, str)],
        enabled=family.enabled,
        archived=family.archived,
        compatibility=compatibility is not None,
        variants=[
            _workflow_family_variant_out(
                session,
                services,
                family,
                definition,
                compatibility,
                ready_offers,
            )
            for definition in sorted(
                family.definitions,
                key=lambda item: (item.operation, item.variant_key or "", item.id),
            )
        ],
        preferences=[
            WorkflowFamilyPreferenceOut(
                selector_capability=_workflow_selector_capability(preference.selector_capability),
                enabled=preference.enabled,
                is_default=preference.is_default,
                sort_order=preference.sort_order,
            )
            for preference in sorted(
                family.preferences,
                key=lambda item: (item.selector_capability, item.sort_order, item.id),
            )
        ],
        created_at=family.created_at,
        updated_at=family.updated_at,
    )


def _workflow_family_row(session: Session, family_id: str) -> WorkflowFamily:
    family = session.scalar(
        select(WorkflowFamily)
        .options(
            selectinload(WorkflowFamily.definitions),
            selectinload(WorkflowFamily.preferences),
        )
        .where(WorkflowFamily.id == family_id)
    )
    if family is None:
        raise api_error(404, "workflow-family-not-found", "workflow family not found")
    return family


def _workflow_family_compatibility_profile(
    session: Session,
    family_id: str,
) -> ModelProfile | None:
    mapping = session.scalar(
        select(WorkflowProfileCompatibility).where(
            WorkflowProfileCompatibility.workflow_family_id == family_id
        )
    )
    return session.get(ModelProfile, mapping.model_profile_id) if mapping is not None else None


def _workflow_family_removal_impact_out(
    impact: WorkflowFamilyRemovalImpact,
) -> WorkflowFamilyRemovalImpactOut:
    return WorkflowFamilyRemovalImpactOut(
        family_id=impact.family_id,
        archive_blocked=impact.archive_blocked,
        revision_count=impact.revision_count,
        current_revision_count=impact.current_revision_count,
        chat_selection_count=impact.chat_selection_count,
        project_selection_count=impact.project_selection_count,
        project_revision_pin_count=impact.project_revision_pin_count,
        active_run_count=impact.active_run_count,
        queued_step_count=impact.queued_step_count,
        historical_run_count=impact.historical_run_count,
        active_activation_count=impact.active_activation_count,
        default_for=[_workflow_selector_capability(value) for value in impact.default_for],
        dependencies=[
            WorkflowDependencyImpactOut(
                resource_kind=dependency.resource_kind,
                resource_id=dependency.resource_id,
                resource_name=dependency.resource_name,
                binding_count=dependency.binding_count,
                revision_count=dependency.revision_count,
                current_revision=dependency.current_revision,
                shared=dependency.shared,
                other_workflow_count=dependency.other_workflow_count,
                other_family_ids=list(dependency.other_family_ids),
            )
            for dependency in impact.dependencies
        ],
    )


@router.get("/workflow-families", response_model=list[WorkflowFamilyOut])
async def list_workflow_families(
    request: Request,
    session: SessionDep,
    selector_capability: WorkflowSelectorCapability | None = None,
    include_archived: bool = False,
) -> list[WorkflowFamilyOut]:
    query = select(WorkflowFamily).options(
        selectinload(WorkflowFamily.definitions),
        selectinload(WorkflowFamily.preferences),
    )
    if not include_archived:
        query = query.where(WorkflowFamily.archived.is_(False))
    if selector_capability is not None:
        query = query.join(WorkflowPreference).where(
            WorkflowPreference.selector_capability == selector_capability
        )
    families = list(
        session.scalars(query.order_by(WorkflowFamily.name, WorkflowFamily.id)).unique()
    )
    services = _services(request)
    ready_offers = _ready_workflow_install_offers(
        session,
        selector_capability=selector_capability,
        include_archived=include_archived,
    )
    return [_workflow_family_out(session, services, family, ready_offers) for family in families]


@router.get("/workflow-families/{family_id}", response_model=WorkflowFamilyOut)
async def get_workflow_family(
    family_id: str,
    request: Request,
    session: SessionDep,
) -> WorkflowFamilyOut:
    family = _workflow_family_row(session, family_id)
    ready_offers = _ready_workflow_install_offers(session, family_id=family_id)
    return _workflow_family_out(session, _services(request), family, ready_offers)


@router.patch("/workflow-families/{family_id}", response_model=WorkflowFamilyOut)
async def update_workflow_family(
    family_id: str,
    payload: WorkflowFamilyUpdate,
    request: Request,
    session: SessionDep,
) -> WorkflowFamilyOut:
    family = _workflow_family_row(session, family_id)
    values = payload.model_dump(exclude_unset=True)
    disabling = values.get("enabled") is False and family.enabled
    archiving = values.get("archived") is True and not family.archived
    if disabling or archiving:
        impact = workflow_family_removal_impact(session, family)
        if impact.archive_blocked:
            raise api_error(
                409,
                "workflow-family-in-use",
                "clear active workflow selections and defaults before disabling this family",
                chat_selection_count=impact.chat_selection_count,
                project_selection_count=impact.project_selection_count,
                default_for=list(impact.default_for),
            )
    final_archived = values.get("archived", family.archived)
    if final_archived and values.get("enabled") is True:
        raise api_error(
            422,
            "workflow-family-archived",
            "an archived workflow family cannot be enabled",
        )
    compatibility_profile = _workflow_family_compatibility_profile(session, family.id)
    if values.get("name") is not None:
        name = values["name"].strip()
        if not name:
            raise api_error(422, "workflow-family-name-empty", "workflow family name is empty")
        family.name = name
        if compatibility_profile is not None:
            compatibility_profile.name = name
    if values.get("description") is not None:
        family.description = values["description"].strip()
    if values.get("use_case") is not None:
        family.use_case = values["use_case"].strip()
        if compatibility_profile is not None:
            compatibility_profile.use_case = family.use_case
    if values.get("tags") is not None:
        family.tags_json = _normalized_workflow_family_tags(values["tags"])
    if archiving:
        family.archived = True
        family.enabled = False
        for preference in family.preferences:
            preference.enabled = False
            preference.is_default = False
    else:
        if "archived" in values:
            family.archived = values["archived"]
        if "enabled" in values:
            family.enabled = values["enabled"]
    if compatibility_profile is not None and ({"name", "use_case"} & values.keys()):
        ensure_legacy_profile_workflow(session, compatibility_profile)
    session.commit()
    family = _workflow_family_row(session, family.id)
    ready_offers = _ready_workflow_install_offers(session, family_id=family.id)
    return _workflow_family_out(session, _services(request), family, ready_offers)


def _normalized_workflow_family_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = raw_tag.strip()
        key = tag.casefold()
        if not tag or len(tag) > 100:
            raise api_error(
                422,
                "workflow-family-tag-invalid",
                "workflow family tags must contain 1 to 100 characters",
            )
        if key not in seen:
            normalized.append(tag)
            seen.add(key)
    return normalized


@router.put(
    "/workflow-families/{family_id}/preferences/{selector_capability}",
    response_model=WorkflowFamilyPreferenceOut,
)
async def update_workflow_family_preference(
    family_id: str,
    selector_capability: WorkflowSelectorCapability,
    payload: WorkflowFamilyPreferenceUpdate,
    session: SessionDep,
) -> WorkflowFamilyPreferenceOut:
    family = _workflow_family_row(session, family_id)
    if payload.is_default and not payload.enabled:
        raise api_error(
            422,
            "workflow-default-disabled",
            "a default workflow preference must be enabled",
        )
    if payload.enabled and (family.archived or not family.enabled):
        raise api_error(
            422,
            "workflow-family-not-selectable",
            "enable and restore the workflow family before adding it to a selector",
        )
    compatible_operations = {
        operation.value for operation in _SELECTOR_OPERATIONS[selector_capability]
    }
    has_variant = session.scalar(
        select(WorkflowDefinition.id)
        .where(
            WorkflowDefinition.family_id == family.id,
            WorkflowDefinition.operation.in_(compatible_operations),
        )
        .limit(1)
    )
    if has_variant is None:
        raise api_error(
            422,
            "workflow-family-operation-unavailable",
            "workflow family has no compatible operation variant",
        )
    if not payload.enabled:
        reference_count = workflow_family_selector_reference_count(
            session,
            family.id,
            selector_capability,
        )
        if reference_count:
            raise api_error(
                409,
                "workflow-preference-in-use",
                "clear active workflow selections before disabling this preference",
                selector_reference_count=reference_count,
            )
    preference = _workflow_preference(session, family.id, selector_capability)
    compatibility_profile = _workflow_family_compatibility_profile(session, family.id)
    if payload.is_default:
        for previous_default in session.scalars(
            select(WorkflowPreference).where(
                WorkflowPreference.selector_capability == selector_capability,
                WorkflowPreference.is_default.is_(True),
                WorkflowPreference.workflow_family_id != family.id,
            )
        ).all():
            previous_default.is_default = False
        # The database enforces one default per selector with a partial unique
        # index. Flush the old default first so SQLite cannot order the two
        # updates in the unsafe direction inside the final commit.
        session.flush()
    if preference is None:
        preference = WorkflowPreference(
            workflow_family_id=family.id,
            selector_capability=selector_capability,
        )
        session.add(preference)
    preference.enabled = payload.enabled
    preference.is_default = payload.is_default
    preference.sort_order = payload.sort_order
    if compatibility_profile is not None:
        if payload.is_default:
            for sibling in session.scalars(
                select(ModelProfile).where(
                    ModelProfile.role == compatibility_profile.role,
                    ModelProfile.id != compatibility_profile.id,
                )
            ).all():
                sibling.is_default = False
        compatibility_profile.is_default = payload.is_default
        reconcile_legacy_workflow_compatibility(session)
    session.commit()
    session.refresh(preference)
    return WorkflowFamilyPreferenceOut(
        selector_capability=selector_capability,
        enabled=preference.enabled,
        is_default=preference.is_default,
        sort_order=preference.sort_order,
    )


@router.get(
    "/workflow-families/{family_id}/removal-impact",
    response_model=WorkflowFamilyRemovalImpactOut,
)
async def get_workflow_family_removal_impact(
    family_id: str,
    session: SessionDep,
) -> WorkflowFamilyRemovalImpactOut:
    family = _workflow_family_row(session, family_id)
    return _workflow_family_removal_impact_out(workflow_family_removal_impact(session, family))


@router.get(
    "/workflow-dependencies/{resource_kind}/{resource_id}/consumers",
    response_model=WorkflowResourceConsumersOut,
)
async def get_workflow_dependency_consumers(
    resource_kind: WorkflowDependencyResourceKind,
    resource_id: str,
    session: SessionDep,
) -> WorkflowResourceConsumersOut:
    consumers = workflow_resource_consumers(session, resource_kind, resource_id)
    return WorkflowResourceConsumersOut(
        resource_kind=resource_kind,
        resource_id=resource_id,
        resource_name=workflow_resource_name(session, resource_kind, resource_id),
        consumers=[
            WorkflowResourceConsumerOut(
                workflow_id=consumer.workflow_id,
                workflow_name=consumer.workflow_name,
                workflow_family_id=consumer.workflow_family_id,
                workflow_family_name=consumer.workflow_family_name,
                revision_ids=list(consumer.revision_ids),
                binding_count=consumer.binding_count,
                current_revision=consumer.current_revision,
            )
            for consumer in consumers
        ],
    )


@router.get(
    "/chats/{chat_id}/workflow-selections",
    response_model=list[WorkflowSelectionOut],
)
async def list_chat_workflow_selections(
    chat_id: str,
    session: ConversationSessionDep,
) -> list[WorkflowSelectionOut]:
    chat = session.get(Chat, chat_id)
    if chat is None:
        raise api_error(404, "chat-not-found", "chat not found")
    return [
        _chat_workflow_selection_out(session, chat, capability)
        for capability in ("chat", "vision", "image", "video")
    ]


@router.put(
    "/chats/{chat_id}/workflow-selections/{selector_capability}",
    response_model=WorkflowSelectionOut,
)
async def set_chat_workflow_selection(
    chat_id: str,
    selector_capability: WorkflowSelectorCapability,
    payload: ChatWorkflowSelectionIn,
    session: ConversationSessionDep,
) -> WorkflowSelectionOut:
    chat = session.get(Chat, chat_id)
    if chat is None:
        raise api_error(404, "chat-not-found", "chat not found")
    selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == chat.id,
            ChatWorkflowSelection.selector_capability == selector_capability,
        )
    )
    legacy_field = _CHAT_WORKFLOW_PROFILE_FIELDS[selector_capability]
    if payload.mode == "default":
        if selection is not None:
            session.delete(selection)
        setattr(chat, legacy_field, None)
    else:
        family_id: str | None = None
        legacy_profile_id: str | None = AUTO_PROFILE_ID
        if payload.mode == "family":
            family = _selectable_workflow_family(
                session,
                payload.workflow_family_id,
                selector_capability,
            )
            family_id = family.id
            compatibility = session.scalar(
                select(WorkflowProfileCompatibility).where(
                    WorkflowProfileCompatibility.workflow_family_id == family.id
                )
            )
            legacy_profile_id = (
                compatibility.model_profile_id if compatibility is not None else None
            )
        if selection is None:
            selection = ChatWorkflowSelection(
                chat_id=chat.id,
                selector_capability=selector_capability,
                mode=payload.mode,
                workflow_family_id=family_id,
            )
            session.add(selection)
        else:
            selection.mode = payload.mode
            selection.workflow_family_id = family_id
        setattr(chat, legacy_field, legacy_profile_id)
    session.commit()
    return _chat_workflow_selection_out(session, chat, selector_capability)


@router.get(
    "/projects/{project_id}/workflow-selections",
    response_model=list[WorkflowSelectionOut],
)
async def list_project_workflow_selections(
    project_id: str,
    session: SessionDep,
) -> list[WorkflowSelectionOut]:
    project = session.get(Project, project_id)
    if project is None:
        raise api_error(404, "project-not-found", "project not found")
    return [
        _project_workflow_selection_out(session, project, capability)
        for capability in ("image", "video")
    ]


@router.put(
    "/projects/{project_id}/workflow-selections/{selector_capability}",
    response_model=WorkflowSelectionOut,
)
async def set_project_workflow_selection(
    project_id: str,
    selector_capability: Literal["image", "video"],
    payload: ProjectWorkflowSelectionIn,
    session: SessionDep,
) -> WorkflowSelectionOut:
    project = session.get(Project, project_id)
    if project is None:
        raise api_error(404, "project-not-found", "project not found")
    selection = session.scalar(
        select(ProjectWorkflowSelection).where(
            ProjectWorkflowSelection.project_id == project.id,
            ProjectWorkflowSelection.selector_capability == selector_capability,
        )
    )
    legacy_field = _PROJECT_WORKFLOW_REVISION_FIELDS[selector_capability]
    if payload.mode == "inherit":
        if selection is not None:
            session.delete(selection)
        setattr(project, legacy_field, None)
    else:
        family_id: str | None = None
        revision_id: str | None = None
        if payload.mode == "family":
            family = _selectable_workflow_family(
                session,
                payload.workflow_family_id,
                selector_capability,
            )
            family_id = family.id
        elif payload.mode == "revision":
            revision = session.get(WorkflowRevision, payload.workflow_revision_id)
            definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
            operations = {
                operation.value for operation in _SELECTOR_OPERATIONS[selector_capability]
            }
            if revision is None or definition is None:
                raise api_error(404, "workflow-revision-not-found", "workflow revision not found")
            if definition.operation not in operations:
                raise api_error(
                    422,
                    "workflow-revision-selector-mismatch",
                    "workflow revision does not match this selector",
                )
            revision_id = revision.id
        if selection is None:
            selection = ProjectWorkflowSelection(
                project_id=project.id,
                selector_capability=selector_capability,
                mode=payload.mode,
                workflow_family_id=family_id,
                workflow_revision_id=revision_id,
            )
            session.add(selection)
        else:
            selection.mode = payload.mode
            selection.workflow_family_id = family_id
            selection.workflow_revision_id = revision_id
        setattr(project, legacy_field, revision_id)
    session.commit()
    return _project_workflow_selection_out(session, project, selector_capability)


@router.get("/workflows", response_model=list[WorkflowOut])
async def list_workflows(session: SessionDep) -> list[WorkflowDefinition]:
    definitions = list(
        session.scalars(
            select(WorkflowDefinition)
            .options(selectinload(WorkflowDefinition.revisions))
            .order_by(WorkflowDefinition.name)
        ).all()
    )
    # Package drafts exist only to give dependency preparation a saved subject.
    # Until compilation creates the executable revision, presenting one as an
    # ordinary selectable workflow makes the library look broken.
    return [
        definition
        for definition in definitions
        if not is_workflow_package_draft(
            next(
                (
                    revision
                    for revision in definition.revisions
                    if revision.id == definition.current_revision_id
                ),
                None,
            )
        )
    ]


@router.post("/workflows", response_model=WorkflowOut, status_code=201)
async def create_workflow(payload: WorkflowCreate, session: SessionDep) -> WorkflowDefinition:
    try:
        validate_lora_workflow_contract(
            payload.api_graph,
            payload.input_schema,
            payload.dependencies,
        )
        validate_workflow_edit_calibration(payload.input_schema)
    except ValueError as exc:
        raise api_error(422, "workflow-invalid", str(exc)) from exc
    definition = WorkflowDefinition(
        name=payload.name,
        operation=payload.operation.value,
        description=payload.description,
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine=payload.engine,
        engine_version=payload.engine_version,
        ui_graph_json=payload.ui_graph,
        api_graph_json=payload.api_graph,
        input_schema_json=payload.input_schema,
        dependencies_json=payload.dependencies,
        trusted=payload.trusted,
        # Every revision carries its artifact identity, not only compiled ones -
        # otherwise a hand-authored or imported workflow cannot take part in
        # capability evidence or in pin migration across recompiles.
        artifact_sha256=workflow_artifact_contract(
            operation=definition.operation,
            engine=payload.engine,
            api_graph=payload.api_graph,
            input_schema=payload.input_schema,
            dependencies=payload.dependencies,
        ),
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    ensure_workflow_family_ownership(session, definition, revision)
    session.commit()
    created = session.scalar(
        select(WorkflowDefinition)
        .options(selectinload(WorkflowDefinition.revisions))
        .where(WorkflowDefinition.id == definition.id)
    )
    if not created:
        raise api_error(500, "workflow-reload-failed", "workflow could not be reloaded")
    return created


@router.patch("/workflows/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(
    workflow_id: str, payload: WorkflowUpdate, session: SessionDep
) -> WorkflowDefinition:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition:
        raise api_error(404, "workflow-not-found", "workflow not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(definition, key, value)
    session.commit()
    updated = session.scalar(
        select(WorkflowDefinition)
        .options(selectinload(WorkflowDefinition.revisions))
        .where(WorkflowDefinition.id == workflow_id)
    )
    if not updated:
        raise api_error(500, "workflow-reload-failed", "workflow could not be reloaded")
    return updated


@router.get("/workflows/{workflow_id}/export", response_model=WorkflowBundle)
async def export_workflow(workflow_id: str, session: SessionDep) -> WorkflowBundle:
    definition, revision = _workflow_and_revision(session, workflow_id)
    return _workflow_bundle(definition, revision)


async def _vouch_for_role_workflows(
    services: Services,
    session: Session,
    role: Literal["chat", "image", "video"],
) -> int:
    """Trust any workflow for this role that this machine can rebuild itself.

    An imported workflow arrives untrusted, and `workflow_untrusted` is a
    blocking readiness check - so setup verification refused to start, and told
    the user to review a ComfyUI graph. That is a dead end for anyone who cannot
    read one, and it is unnecessary whenever the graph recompiles here
    byte-identically from a template already installed.

    Runs before verification because that is the moment the media runtime is up
    and the answer is obtainable. Anything that cannot be re-derived is left
    untrusted, exactly as before.
    """
    operations = MEDIA_OPERATIONS_BY_ROLE[role]
    if not operations:
        return 0
    vouched = 0
    definitions = session.scalars(
        select(WorkflowDefinition).where(WorkflowDefinition.operation.in_(operations))
    ).all()
    for definition in definitions:
        if not definition.current_revision_id:
            continue
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        if not revision or revision.trusted:
            continue
        decision = await _derive_trust_for_revision(services, definition, revision)
        if decision.trusted:
            revision.trusted = True
            vouched += 1
    if vouched:
        session.commit()
    return vouched


async def _derive_trust_for_revision(
    services: Services,
    definition: WorkflowDefinition,
    revision: WorkflowRevision,
) -> TrustDecision:
    """Ask whether this machine can vouch for one revision by rebuilding it.

    One implementation, shared by the explicit endpoint and by setup
    verification, so the two can never answer differently.
    """
    # Templates are indexed by media role, which the definition records as an
    # operation.
    role = "video" if "video" in definition.operation else "image"
    identity = recorded_template_identity(revision.dependencies_json)
    template: ComfyTemplate | None = None
    recompiled_graph: dict[str, Any] | None = None
    if identity:
        template_id, _ = identity
        try:
            template = services.downloads.comfy_templates.get(template_id, role)
        except (ValueError, LookupError, FileNotFoundError):
            template = None
        if template:
            try:
                describe_nodes = getattr(services.engines.media, "object_info", None)
                if not callable(describe_nodes):
                    raise RuntimeError("this media engine cannot describe its nodes")
                object_info = await describe_nodes()
                recompiled_graph = services.downloads.comfy_templates.compile(
                    template_id,
                    role,
                    object_info,
                ).api_graph
            except Exception:
                # The runtime is not up, or the template no longer compiles here.
                # Either way this machine cannot vouch for the graph right now,
                # which the decision reports as a refusal rather than an error -
                # nothing is wrong, it just cannot be proven yet.
                recompiled_graph = None

    return derive_trust(
        dependencies=revision.dependencies_json,
        stored_api_graph=revision.api_graph_json,
        installed_template_sha256=template.sha256 if template else None,
        recompiled_api_graph=recompiled_graph,
        uses_only_core_nodes=template is not None,
    )


@router.post("/workflows/{workflow_id}/derive-trust", response_model=TrustDerivation)
async def derive_workflow_trust(
    workflow_id: str, request: Request, session: SessionDep
) -> TrustDerivation:
    """Trust a workflow this machine can rebuild for itself.

    Import forces `trusted=False` and execution refuses untrusted revisions, so
    an imported setup arrives inert. If recompiling the recorded template here
    yields byte-identical bytes, the graph was derived on this machine from a
    template already shipped here - the same assertion the compiler makes during
    a normal install - and no human review adds anything. Anything that cannot be
    re-derived still requires review, and this says which case it is.
    """
    definition, revision = _workflow_and_revision(session, workflow_id)
    if revision.trusted:
        return TrustDerivation(
            version=TRUST_DERIVATION_VERSION,
            trusted=True,
            reason="already_trusted",
            message="This workflow is already trusted.",
        )
    decision = await _derive_trust_for_revision(_services(request), definition, revision)
    if decision.trusted:
        revision.trusted = True
        session.commit()
    return TrustDerivation(**decision.as_dict())


def _workflow_editor_shell_asset_response(name: str, media_type: str) -> Response:
    try:
        content = workflow_editor_shell_asset(name)
    except ComfyEditorBridgeError as exc:
        raise api_error(500, exc.code, str(exc)) from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@router.get(SHELL_SCRIPT_ROUTE.removeprefix("/api"), include_in_schema=False)
async def workflow_editor_shell_script() -> Response:
    return _workflow_editor_shell_asset_response("shell.js", "application/javascript")


@router.get(SHELL_STYLE_ROUTE.removeprefix("/api"), include_in_schema=False)
async def workflow_editor_shell_style() -> Response:
    return _workflow_editor_shell_asset_response("shell.css", "text/css")


@router.get("/workflow-editor/shell", include_in_schema=False)
async def workflow_editor_shell(request: Request) -> HTMLResponse:
    services = _services(request)
    if services.processes.workflow_editor_runtime_identity() is None:
        raise api_error(
            409,
            "workflow-editor-runtime-not-ready",
            "Start the supported media runtime before opening this workflow.",
        )
    shell_origin = f"{request.url.scheme}://{request.url.netloc}"
    if workflow_editor_origins_conflict(services.settings.comfy_url, shell_origin):
        raise api_error(
            409,
            "workflow-editor-origin-conflict",
            "The media runtime must use a separate loopback origin for native editing.",
        )
    try:
        document = workflow_editor_shell_document(services.settings.comfy_url)
        csp = workflow_editor_shell_csp(services.settings.comfy_url)
    except ComfyEditorBridgeError as exc:
        raise api_error(500, exc.code, str(exc)) from exc
    return HTMLResponse(
        document,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": csp,
        },
    )


@router.get("/workflows/{workflow_id}/open-target", response_model=WorkflowOpenTarget)
async def workflow_open_target(
    workflow_id: str, request: Request, session: SessionDep
) -> WorkflowOpenTarget:
    definition, revision = _workflow_and_revision(session, workflow_id)
    if not revision.ui_graph_json:
        raise api_error(
            422, "workflow-ui-graph-absent", "this workflow has no ComfyUI user-interface graph"
        )
    filename = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in definition.name
    ).strip("-")
    return WorkflowOpenTarget(
        url=_services(request).settings.comfy_url,
        filename=f"{filename or 'workflow'}.comfyui.json",
        ui_graph=revision.ui_graph_json,
    )


@router.post(
    "/workflows/{workflow_id}/editor-sessions",
    response_model=WorkflowEditorSessionOut,
    status_code=201,
)
async def start_workflow_editor_session(
    workflow_id: str,
    request: Request,
    response: Response,
    session: SessionDep,
) -> WorkflowEditorSessionOut:
    definition, revision = _workflow_and_revision(session, workflow_id)
    if revision.engine != "comfyui":
        raise api_error(
            422,
            "workflow-editor-engine-unsupported",
            "Native editing requires a ComfyUI workflow.",
        )
    if not revision.ui_graph_json:
        raise api_error(
            422,
            "workflow-editor-ui-graph-missing",
            "This workflow has no ComfyUI user-interface graph.",
        )
    if not revision.api_graph_json:
        raise api_error(
            409,
            "workflow-editor-api-graph-missing",
            "This workflow has no executable ComfyUI prompt to verify.",
        )

    services = _services(request)
    runtime_identity = services.processes.workflow_editor_runtime_identity()
    if runtime_identity is None:
        raise api_error(
            409,
            "workflow-editor-runtime-not-ready",
            "Start the supported media runtime before opening this workflow.",
        )
    describe_nodes = getattr(services.engines.media, "object_info", None)
    if not callable(describe_nodes):
        raise api_error(
            503,
            "workflow-editor-node-inventory-unavailable",
            "The media runtime cannot describe its workflow nodes.",
        )
    invalidate_nodes = getattr(services.engines.media, "invalidate_object_info_cache", None)
    if callable(invalidate_nodes):
        invalidate_nodes()
    try:
        object_info = await describe_nodes()
    except Exception as exc:  # noqa: BLE001 - every runtime failure is unavailable here
        raise api_error(
            503,
            "workflow-editor-node-inventory-unavailable",
            "The media runtime could not describe its workflow nodes.",
        ) from exc
    if not isinstance(object_info, Mapping):
        raise api_error(
            503,
            "workflow-editor-node-inventory-invalid",
            "The media runtime returned an invalid workflow node inventory.",
        )
    try:
        prepared = prepare_workflow_revision_compilation(
            revision.ui_graph_json,
            object_info,
            definition.operation,
            revision.api_graph_json,
            revision.input_schema_json,
        )
        compilation = compile_comfyui_ui_graph(
            prepared.ui_graph,
            prepared.object_info,
        )
        compiled_api_graph = prepared.bind(compilation.api_graph)
        compiled_sha256 = workflow_api_graph_sha256(compiled_api_graph)
        stored_sha256 = workflow_api_graph_sha256(revision.api_graph_json)
    except (
        WorkflowCompilationError,
        WorkflowPackageError,
        WorkflowPackageInputError,
    ) as exc:
        raise api_error(
            422,
            "workflow-editor-graph-cannot-compile",
            "This workflow cannot be opened for verified native editing.",
            reason_code=exc.code,
        ) from exc
    except WorkflowEditorSessionError as exc:
        raise api_error(
            422,
            "workflow-editor-graph-invalid",
            "This workflow cannot be opened for verified native editing.",
            reason_code=exc.code,
        ) from exc
    if compiled_sha256 != stored_sha256:
        raise api_error(
            409,
            "workflow-editor-graph-prompt-mismatch",
            "The visual workflow no longer matches its executable prompt.",
        )
    if services.processes.workflow_editor_runtime_identity() != runtime_identity:
        raise api_error(
            409,
            "workflow-editor-runtime-changed",
            "The media runtime changed while the workflow editor was opening.",
        )
    try:
        editor_session = services.workflow_editor_sessions.start(
            workflow_id=workflow_id,
            base_revision_id=revision.id,
            base_ui_graph=revision.ui_graph_json,
            base_api_graph=revision.api_graph_json,
            runtime_identity=runtime_identity,
        )
    except WorkflowEditorSessionError as exc:
        if exc.code == "workflow-editor-capacity":
            raise api_error(429, exc.code, str(exc)) from exc
        raise api_error(
            422,
            "workflow-editor-session-invalid",
            "The workflow editor session could not be created.",
            reason_code=exc.code,
        ) from exc
    response.headers["Cache-Control"] = "no-store"
    return WorkflowEditorSessionOut(
        id=editor_session.id,
        protocol_version=editor_session.protocol_version,
        workflow_id=editor_session.workflow_id,
        base_revision_id=editor_session.base_revision_id,
        base_graph_sha256=editor_session.base_graph_sha256,
        base_prompt_sha256=editor_session.base_prompt_sha256,
        created_at=editor_session.created_at,
        expires_at=editor_session.expires_at,
        ui_graph=revision.ui_graph_json,
        nonce=editor_session.nonce,
    )


@router.post(
    "/workflows/{workflow_id}/editor-sessions/{session_id}/consume",
    response_model=WorkflowEditorReturnOut,
)
async def consume_workflow_editor_session(
    workflow_id: str,
    session_id: str,
    payload: WorkflowEditorConsumeIn,
    request: Request,
    response: Response,
    session: SessionDep,
) -> WorkflowEditorReturnOut:
    services = _services(request)
    runtime_identity = services.processes.workflow_editor_runtime_identity()
    if runtime_identity is None:
        raise api_error(
            409,
            "workflow-editor-runtime-not-ready",
            "The supported media runtime is no longer available.",
        )
    try:
        editor_session = services.workflow_editor_sessions.authorize_return(
            session_id=session_id,
            nonce=payload.nonce,
            workflow_id=workflow_id,
            base_revision_id=payload.base_revision_id,
            runtime_identity=runtime_identity,
        )
    except WorkflowEditorSessionError as exc:
        if exc.code == "workflow-editor-session-not-found":
            raise api_error(404, exc.code, str(exc)) from exc
        if exc.code == "workflow-editor-session-authentication-failed":
            raise api_error(403, exc.code, str(exc)) from exc
        if exc.code in {
            "workflow-editor-session-mismatch",
            "workflow-editor-runtime-changed",
        }:
            raise api_error(409, exc.code, str(exc)) from exc
        raise api_error(
            422,
            "workflow-editor-session-invalid",
            "The workflow editor return could not be authenticated.",
            reason_code=exc.code,
        ) from exc

    definition = session.get(WorkflowDefinition, editor_session.workflow_id)
    base_revision = session.get(WorkflowRevision, editor_session.base_revision_id)
    if (
        definition is None
        or base_revision is None
        or base_revision.workflow_id != definition.id
        or not base_revision.ui_graph_json
        or not base_revision.api_graph_json
    ):
        raise api_error(
            409,
            "workflow-editor-base-revision-changed",
            "The workflow revision opened by this editor session is no longer available.",
        )
    try:
        base_graph_sha256 = workflow_ui_graph_sha256(base_revision.ui_graph_json)
        base_prompt_sha256 = workflow_api_graph_sha256(base_revision.api_graph_json)
    except WorkflowEditorSessionError as exc:
        raise api_error(
            409,
            "workflow-editor-base-revision-invalid",
            "The workflow revision opened by this editor session can no longer be verified.",
            reason_code=exc.code,
        ) from exc
    if (
        base_graph_sha256 != editor_session.base_graph_sha256
        or base_prompt_sha256 != editor_session.base_prompt_sha256
    ):
        raise api_error(
            409,
            "workflow-editor-base-revision-changed",
            "The workflow revision changed while the native editor was open.",
        )

    try:
        workflow_ui_graph_sha256(payload.ui_graph)
        returned_prompt_sha256 = workflow_api_graph_sha256(payload.api_prompt)
    except WorkflowEditorSessionError as exc:
        raise api_error(
            422,
            "workflow-editor-return-invalid",
            "The workflow editor returned an invalid graph or prompt.",
            reason_code=exc.code,
        ) from exc

    describe_nodes = getattr(services.engines.media, "object_info", None)
    if not callable(describe_nodes):
        raise api_error(
            503,
            "workflow-editor-node-inventory-unavailable",
            "The media runtime cannot describe its workflow nodes.",
        )
    invalidate_nodes = getattr(services.engines.media, "invalidate_object_info_cache", None)
    if callable(invalidate_nodes):
        invalidate_nodes()
    try:
        object_info = await describe_nodes()
    except Exception as exc:  # noqa: BLE001 - every runtime failure is unavailable here
        raise api_error(
            503,
            "workflow-editor-node-inventory-unavailable",
            "The media runtime could not describe its workflow nodes.",
        ) from exc
    if not isinstance(object_info, Mapping):
        raise api_error(
            503,
            "workflow-editor-node-inventory-invalid",
            "The media runtime returned an invalid workflow node inventory.",
        )
    try:
        prepared = prepare_workflow_revision_compilation(
            payload.ui_graph,
            object_info,
            definition.operation,
            base_revision.api_graph_json,
            base_revision.input_schema_json,
        )
        compilation = compile_comfyui_ui_graph(
            prepared.ui_graph,
            prepared.object_info,
        )
        raw_compiled_api_graph = {key: dict(value) for key, value in compilation.api_graph.items()}
        raw_compiled_prompt_sha256 = workflow_api_graph_sha256(raw_compiled_api_graph)
        compiled_api_graph = prepared.bind(raw_compiled_api_graph)
    except (
        WorkflowCompilationError,
        WorkflowPackageError,
        WorkflowPackageInputError,
    ) as exc:
        raise api_error(
            422,
            "workflow-editor-return-cannot-compile",
            "The returned visual workflow cannot be compiled.",
            reason_code=exc.code,
        ) from exc
    except WorkflowEditorSessionError as exc:
        raise api_error(
            422,
            "workflow-editor-return-invalid",
            "The workflow editor returned an invalid graph or prompt.",
            reason_code=exc.code,
        ) from exc
    if raw_compiled_prompt_sha256 != returned_prompt_sha256:
        raise api_error(
            422,
            "workflow-editor-return-prompt-mismatch",
            "The returned visual workflow does not match its executable prompt.",
        )
    if services.processes.workflow_editor_runtime_identity() != runtime_identity:
        raise api_error(
            409,
            "workflow-editor-runtime-changed",
            "The media runtime changed while the workflow editor return was verified.",
        )

    session.refresh(definition, attribute_names=["current_revision_id"])
    if not definition.current_revision_id:
        raise api_error(
            409,
            "workflow-editor-current-revision-missing",
            "The workflow no longer has a current revision.",
        )
    try:
        result = services.workflow_editor_sessions.consume(
            session_id=session_id,
            nonce=payload.nonce,
            workflow_id=workflow_id,
            base_revision_id=payload.base_revision_id,
            current_revision_id=definition.current_revision_id,
            returned_ui_graph=payload.ui_graph,
            returned_api_graph=compiled_api_graph,
            runtime_identity=runtime_identity,
        )
    except WorkflowEditorSessionError as exc:
        if exc.code == "workflow-editor-session-not-found":
            raise api_error(404, exc.code, str(exc)) from exc
        if exc.code == "workflow-editor-session-authentication-failed":
            raise api_error(403, exc.code, str(exc)) from exc
        if exc.code in {
            "workflow-editor-session-mismatch",
            "workflow-editor-runtime-changed",
        }:
            raise api_error(409, exc.code, str(exc)) from exc
        raise api_error(
            422,
            "workflow-editor-return-invalid",
            "The workflow editor return could not be consumed.",
            reason_code=exc.code,
        ) from exc
    response.headers["Cache-Control"] = "no-store"
    return WorkflowEditorReturnOut(
        validated_return_id=result.validated_return_id,
        session_id=result.session_id,
        workflow_id=result.workflow_id,
        base_revision_id=result.base_revision_id,
        current_revision_id=result.current_revision_id,
        base_graph_sha256=result.base_graph_sha256,
        returned_graph_sha256=result.returned_graph_sha256,
        base_prompt_sha256=result.base_prompt_sha256,
        returned_prompt_sha256=result.returned_prompt_sha256,
        changed=result.changed,
        forked=result.forked,
        delta=WorkflowEditorGraphDeltaOut(
            node_count_delta=result.delta.node_count_delta,
            link_count_delta=result.delta.link_count_delta,
            added_node_types=result.delta.added_node_types,
            removed_node_types=result.delta.removed_node_types,
            added_asset_filenames=result.delta.added_asset_filenames,
            removed_asset_filenames=result.delta.removed_asset_filenames,
        ),
        expires_at=result.expires_at,
    )


def _workflow_editor_draft_revision_id(validated_return_id: str) -> str:
    digest = hashlib.sha256(validated_return_id.encode("utf-8")).hexdigest()
    return f"wfrev_editor_{digest[:27]}"


def _workflow_runtime_binding_paths(
    graph: Mapping[str, Any],
) -> frozenset[tuple[tuple[str | int, ...], str]]:
    bindings: set[tuple[tuple[str | int, ...], str]] = set()
    stack: list[tuple[tuple[str | int, ...], Any]] = [((), graph)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(((*path, str(key)), item) for key, item in value.items())
        elif isinstance(value, list):
            stack.extend(((*path, index), item) for index, item in enumerate(value))
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            bindings.add((path, value))
    return frozenset(bindings)


def _workflow_editor_draft_dependencies(
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(dependencies))
    for provenance_key in ("template_id", "template_sha256", "compiler_version"):
        copied.pop(provenance_key, None)
    return copied


def _workflow_editor_dependency_signature(
    ui_graph: Mapping[str, Any],
) -> tuple[Any, ...]:
    analysis = analyze_comfyui_workflow_package(ui_graph)
    assets = tuple(
        (
            asset.filename,
            asset.suffix,
            asset.policy,
            asset.kind,
            asset.source_url,
        )
        for asset in analysis.asset_references
    )
    return (
        analysis.required_node_types,
        analysis.custom_packages,
        assets,
        analysis.node_provenance,
        analysis.operation_guess,
    )


def _workflow_editor_draft_matches(
    revision: WorkflowRevision,
    *,
    workflow_id: str,
    engine: str,
    engine_version: str | None,
    ui_graph: Mapping[str, Any],
    api_graph: Mapping[str, Any],
    input_schema: Mapping[str, Any],
    capabilities: list[str],
    dependencies: Mapping[str, Any],
    artifact_sha256: str,
) -> bool:
    return (
        revision.workflow_id == workflow_id
        and revision.engine == engine
        and revision.engine_version == engine_version
        and revision.ui_graph_json == dict(ui_graph)
        and revision.api_graph_json == dict(api_graph)
        and revision.input_schema_json == dict(input_schema)
        and revision.capabilities_json == capabilities
        and revision.dependencies_json == dict(dependencies)
        and revision.dependency_contract_sha256 is None
        and revision.artifact_sha256 == artifact_sha256
        and not revision.trusted
    )


@router.post(
    "/workflows/{workflow_id}/editor-drafts",
    response_model=WorkflowEditorDraftOut,
)
async def create_workflow_editor_draft(
    workflow_id: str,
    payload: WorkflowEditorDraftCreateIn,
    request: Request,
    session: SessionDep,
) -> WorkflowEditorDraftOut:
    services = _services(request)
    runtime_identity = services.processes.workflow_editor_runtime_identity()
    if runtime_identity is None:
        raise api_error(
            409,
            "workflow-editor-runtime-not-ready",
            "The supported media runtime is no longer available.",
        )
    try:
        validated = services.workflow_editor_sessions.validated_return(
            validated_return_id=payload.validated_return_id,
            workflow_id=workflow_id,
            runtime_identity=runtime_identity,
        )
    except WorkflowEditorSessionError as exc:
        if exc.code == "workflow-editor-validated-return-not-found":
            raise api_error(404, exc.code, str(exc)) from exc
        if exc.code in {
            "workflow-editor-validated-return-mismatch",
            "workflow-editor-runtime-changed",
        }:
            raise api_error(409, exc.code, str(exc)) from exc
        raise api_error(
            422,
            "workflow-editor-validated-return-invalid",
            "The validated editor return could not be used.",
            reason_code=exc.code,
        ) from exc
    if not validated.result.changed:
        raise api_error(
            409,
            "workflow-editor-return-unchanged",
            "The editor return contains no workflow changes to save.",
        )

    definition = session.get(WorkflowDefinition, workflow_id)
    base_revision = session.get(WorkflowRevision, validated.result.base_revision_id)
    if definition is None or base_revision is None or base_revision.workflow_id != definition.id:
        raise api_error(
            409,
            "workflow-editor-base-revision-changed",
            "The workflow revision opened by this editor session is no longer available.",
        )
    try:
        returned_ui_graph = json.loads(validated.returned_ui_graph_json)
        returned_api_graph = json.loads(validated.returned_api_graph_json)
    except json.JSONDecodeError as exc:
        raise api_error(
            409,
            "workflow-editor-validated-return-corrupt",
            "The validated editor return can no longer be read.",
        ) from exc
    if not isinstance(returned_ui_graph, dict) or not isinstance(returned_api_graph, dict):
        raise api_error(
            409,
            "workflow-editor-validated-return-corrupt",
            "The validated editor return does not contain workflow graphs.",
        )
    try:
        base_dependency_signature = _workflow_editor_dependency_signature(
            base_revision.ui_graph_json
        )
        returned_dependency_signature = _workflow_editor_dependency_signature(returned_ui_graph)
    except WorkflowPackageError as exc:
        raise api_error(
            409,
            "workflow-editor-validated-return-corrupt",
            "The validated editor return can no longer be analyzed.",
            reason_code=exc.code,
        ) from exc
    if base_dependency_signature != returned_dependency_signature:
        raise api_error(
            422,
            "workflow-editor-draft-dependencies-changed",
            "Review and prepare dependency-changing edits before saving them as a revision.",
        )
    if _workflow_runtime_binding_paths(
        base_revision.api_graph_json
    ) != _workflow_runtime_binding_paths(returned_api_graph):
        raise api_error(
            422,
            "workflow-editor-draft-bindings-changed",
            "Review and prepare runtime input changes before saving them as a revision.",
        )
    try:
        validate_lora_workflow_contract(
            returned_api_graph,
            base_revision.input_schema_json,
            base_revision.dependencies_json,
        )
        validate_workflow_edit_calibration(base_revision.input_schema_json)
    except ValueError as exc:
        raise api_error(
            422,
            "workflow-editor-draft-contract-invalid",
            "The edited workflow does not satisfy its declared input contract.",
        ) from exc

    draft_revision_id = _workflow_editor_draft_revision_id(payload.validated_return_id)
    try:
        services.workflow_editor_sessions.bind_validated_return_to_draft(
            validated_return_id=payload.validated_return_id,
            workflow_id=workflow_id,
            runtime_identity=runtime_identity,
            draft_revision_id=draft_revision_id,
        )
    except WorkflowEditorSessionError as exc:
        if exc.code in {
            "workflow-editor-validated-return-not-found",
            "workflow-editor-validated-return-mismatch",
            "workflow-editor-validated-return-already-applied",
            "workflow-editor-runtime-changed",
        }:
            raise api_error(409, exc.code, str(exc)) from exc
        raise api_error(
            422,
            "workflow-editor-validated-return-invalid",
            "The validated editor return could not be bound to a draft.",
            reason_code=exc.code,
        ) from exc

    capabilities: list[str] = []
    dependencies = _workflow_editor_draft_dependencies(base_revision.dependencies_json)
    input_schema = dict(base_revision.input_schema_json)
    artifact_sha256 = workflow_artifact_contract(
        operation=definition.operation,
        engine=base_revision.engine,
        api_graph=returned_api_graph,
        input_schema=input_schema,
        dependencies=dependencies,
    )
    existing = session.get(WorkflowRevision, draft_revision_id)
    created = False
    if existing is None:
        version = (
            session.scalar(
                select(func.max(WorkflowRevision.version)).where(
                    WorkflowRevision.workflow_id == workflow_id
                )
            )
            or 0
        )
        existing = WorkflowRevision(
            id=draft_revision_id,
            workflow_id=workflow_id,
            version=version + 1,
            engine=base_revision.engine,
            engine_version=base_revision.engine_version,
            ui_graph_json=returned_ui_graph,
            api_graph_json=returned_api_graph,
            input_schema_json=input_schema,
            capabilities_json=capabilities,
            dependencies_json=dependencies,
            dependency_contract_sha256=None,
            trusted=False,
            artifact_sha256=artifact_sha256,
        )
        session.add(existing)
        try:
            session.commit()
            created = True
        except IntegrityError:
            session.rollback()
            existing = session.get(WorkflowRevision, draft_revision_id)
            if existing is None:
                raise api_error(
                    409,
                    "workflow-editor-draft-version-conflict",
                    "The workflow changed while the editor draft was being saved. Retry the save.",
                ) from None
    if not _workflow_editor_draft_matches(
        existing,
        workflow_id=workflow_id,
        engine=base_revision.engine,
        engine_version=base_revision.engine_version,
        ui_graph=returned_ui_graph,
        api_graph=returned_api_graph,
        input_schema=input_schema,
        capabilities=capabilities,
        dependencies=dependencies,
        artifact_sha256=artifact_sha256,
    ):
        raise api_error(
            409,
            "workflow-editor-draft-collision",
            "The validated editor return conflicts with an existing draft.",
        )
    session.refresh(definition, attribute_names=["current_revision_id"])
    return WorkflowEditorDraftOut(
        workflow_id=workflow_id,
        base_revision_id=base_revision.id,
        draft_revision_id=existing.id,
        current_revision_id=definition.current_revision_id,
        version=existing.version,
        created=created,
        forked=definition.current_revision_id != base_revision.id,
        trusted=False,
        review_required=True,
    )


@router.post(
    "/workflows/{workflow_id}/editor-sessions/{session_id}/cancel",
    status_code=204,
)
async def cancel_workflow_editor_session(
    workflow_id: str,
    session_id: str,
    payload: WorkflowEditorCancelIn,
    request: Request,
) -> Response:
    try:
        _services(request).workflow_editor_sessions.cancel(
            session_id=session_id,
            nonce=payload.nonce,
            workflow_id=workflow_id,
        )
    except WorkflowEditorSessionError as exc:
        if exc.code == "workflow-editor-session-not-found":
            raise api_error(404, exc.code, str(exc)) from exc
        if exc.code == "workflow-editor-session-authentication-failed":
            raise api_error(403, exc.code, str(exc)) from exc
        if exc.code == "workflow-editor-session-mismatch":
            raise api_error(409, exc.code, str(exc)) from exc
        raise api_error(
            422,
            "workflow-editor-session-invalid",
            "The workflow editor session could not be cancelled.",
            reason_code=exc.code,
        ) from exc
    return Response(status_code=204)


def _local_asset_filenames(session: Session) -> set[str]:
    """Filenames this machine holds, as a workflow might reference them."""

    filenames: set[str] = set()
    for install in session.scalars(select(ModelInstall)).all():
        files = install.manifest_json.get("files")
        if not isinstance(files, list):
            continue
        for entry in files:
            if isinstance(entry, str) and entry:
                filenames.add(entry)
                filenames.add(PurePosixPath(entry.replace("\\", "/")).name)
    for asset in session.scalars(select(ModelAssetInstall)).all():
        filenames.add(PurePosixPath(asset.local_path.replace("\\", "/")).name)
        # The manager stores local_path as the install directory; the name a
        # workflow actually references lives in the manifest. Without these,
        # a verified LoRA reads as missing during analysis and import.
        manifest = asset.manifest_json
        comfy_name = manifest.get("comfy_name")
        if isinstance(comfy_name, str) and comfy_name:
            filenames.add(comfy_name)
            filenames.add(PurePosixPath(comfy_name.replace("\\", "/")).name)
        asset_files = manifest.get("files")
        if isinstance(asset_files, list):
            for entry in asset_files:
                if isinstance(entry, str) and entry:
                    filenames.add(entry)
                    filenames.add(PurePosixPath(entry.replace("\\", "/")).name)
    return filenames


def _installed_package_versions(session: Session) -> dict[str, set[str]]:
    """Version evidence for installed custom-node packages, exact only.

    Git installs contribute their pinned revisions. Registry installs contribute
    exact declared versions only after their immutable records are trusted and
    active. Unmatched pins remain unresolved rather than being inferred.
    """

    versions = installed_comfy_registry_versions(session)
    for install in session.scalars(
        select(CustomNodeInstall).where(
            CustomNodeInstall.active.is_(True),
            CustomNodeInstall.trusted.is_(True),
        )
    ).all():
        versions.setdefault(install.name, set()).add(install.revision)
    return versions


async def _verified_launchable_registry_packages(
    services: Services,
) -> dict[tuple[str, str], frozenset[str]]:
    """Package-bound nodes an unscoped launch can load after exact verification."""

    resolver = getattr(services.processes, "trusted_comfy_registry_package_node_types", None)
    if not callable(resolver):
        return {}
    try:
        packages = await resolver()
    except Exception:  # noqa: BLE001 - analysis must fail closed, not fail outright
        logger.warning(
            "Installed Registry packages could not be verified for workflow package use."
        )
        return {}
    return {
        (str(key[0]), str(key[1])): frozenset(str(node_type) for node_type in node_types)
        for key, node_types in packages.items()
    }


async def _verified_launchable_manual_packages(
    services: Services,
) -> dict[tuple[str, str], frozenset[str]]:
    """Reviewed package-bound nodes an unscoped manual launch can load."""

    resolver = getattr(services.processes, "trusted_comfy_custom_node_package_node_types", None)
    if not callable(resolver):
        return {}
    try:
        packages = await resolver()
    except Exception:  # noqa: BLE001 - analysis must fail closed, not fail outright
        logger.warning(
            "Installed manual custom-node packages could not be verified for workflow package use."
        )
        return {}
    return {
        (str(key[0]), str(key[1])): frozenset(str(node_type) for node_type in node_types)
        for key, node_types in packages.items()
    }


async def _verified_launchable_packages(
    services: Services,
) -> dict[tuple[str, str], frozenset[str]]:
    registry = await _verified_launchable_registry_packages(services)
    manual = await _verified_launchable_manual_packages(services)
    combined = dict(registry)
    for key, node_types in manual.items():
        previous = combined.get(key)
        if previous is not None and previous != node_types:
            combined.pop(key)
            logger.warning(
                "Installed custom-node package identity %s %s has conflicting ownership.",
                key[0],
                key[1],
            )
            continue
        combined[key] = node_types
    return combined


async def _packages_awaiting_review(services: Services) -> frozenset[tuple[str, str]]:
    """Trusted installs whose reviewed node inventory was never recorded.

    Absent evidence and an absent package resolve identically and need opposite
    actions, so the analysis is told which is which. Failing closed here means
    reporting the package as simply unresolved, which is the older, less helpful
    answer rather than a wrong one.
    """

    resolver = getattr(
        services.processes, "trusted_comfy_custom_node_packages_awaiting_review", None
    )
    if not callable(resolver):
        return frozenset()
    try:
        packages = await resolver()
    except Exception:  # noqa: BLE001 - analysis must not fail outright over advice
        logger.warning("Installed custom-node packages could not be checked for review state.")
        return frozenset()
    # Rebuilt rather than returned, so an untyped resolver cannot widen what
    # this promises - the same shape the launchable-package reader uses.
    return frozenset((str(name), str(revision)) for name, revision in packages)


def _packages_that_did_not_load(
    services: Services,
    analysis: ComfyWorkflowPackageAnalysis,
    available_node_types: set[str],
) -> list[tuple[str, Path | None]]:
    """Packages *this workflow requires* whose nodes are absent after a restart.

    Scoped to the analysis on purpose. The media runtime launches with only the
    packages a particular activation needs, so most installed packages are
    legitimately absent from any given inventory. Reporting every trusted
    install whose nodes are missing would name whichever unrelated package
    happened to be out of scope, and name it as the thing that broke an import
    it has nothing to do with.

    So a package is only a candidate when the workflow declares it, at the
    revision the workflow declares, and the nodes still missing are nodes this
    workflow actually asked that package for.

    Returns each package name with the requirements file it ships, if any, so
    the refusal can distinguish a package that cannot import from one that
    failed for a reason only the runtime log knows.
    """

    from .custom_nodes import CustomNodeManager, reviewed_custom_node_types
    from .db import SessionLocal
    from .models import CustomNodeInstall

    required: dict[tuple[str, str], set[str]] = {}
    for requirement in analysis.custom_packages:
        for version in requirement.versions:
            required[(requirement.package_id, version)] = set(requirement.node_types)
    if not required:
        return []

    failed: list[tuple[str, Path | None]] = []
    with SessionLocal() as session:
        installs = list(
            session.scalars(
                select(CustomNodeInstall).where(
                    CustomNodeInstall.active.is_(True),
                    CustomNodeInstall.trusted.is_(True),
                )
            ).all()
        )
        manager = CustomNodeManager(services.settings)
        for install in installs:
            wanted = required.get((install.name, install.revision))
            if wanted is None:
                continue
            reviewed = set(reviewed_custom_node_types(install.security_json))
            # Only the nodes this workflow asked this package for. A package can
            # own nodes nobody here needs, and their absence proves nothing.
            owed = reviewed & wanted
            if not owed or owed <= available_node_types:
                continue
            failed.append((install.name, manager.python_requirements_path(install)))
    return failed


def _launchable_package_node_types(
    analysis: ComfyWorkflowPackageAnalysis,
    packages: Mapping[tuple[str, str], frozenset[str]],
) -> set[str]:
    """Nodes owned by every exact package/version declared for the requirement."""

    launchable: set[str] = set()
    for requirement in analysis.custom_packages:
        required = set(requirement.node_types)
        versions = set(requirement.versions)
        if versions and all(
            required <= packages.get((requirement.package_id, version), frozenset())
            for version in versions
        ):
            launchable.update(required)
    return launchable


def _verified_package_versions(
    analysis: ComfyWorkflowPackageAnalysis,
    packages: Mapping[tuple[str, str], frozenset[str]],
) -> dict[str, set[str]]:
    """Only versions whose reviewed owner covers every node the graph attributes."""

    verified: dict[str, set[str]] = {}
    for requirement in analysis.custom_packages:
        required = set(requirement.node_types)
        for version in requirement.versions:
            if required <= packages.get((requirement.package_id, version), frozenset()):
                verified.setdefault(requirement.package_id, set()).add(version)
    return verified


_REGISTRY_PREPARE_TASKS: dict[str, asyncio.Task[None]] = {}


async def _cancel_registry_preparation(job_id: str) -> bool:
    """Cancel the live task first, then mark the row; never the reverse.

    Marking the row while the task keeps mutating disk and network would let
    a later success write COMPLETE over CANCELLED.
    """

    task = _REGISTRY_PREPARE_TASKS.get(job_id)
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if not job or job.status in {
            JobStatus.COMPLETE.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }:
            return False
        job.status = JobStatus.CANCELLED.value
        update_job_progress(job, stage="cancelled", indeterminate=True)
        session.commit()
    return True


async def shutdown_registry_preparations() -> None:
    """Cancel and await every live preparation task at service shutdown."""

    tasks = [task for task in _REGISTRY_PREPARE_TASKS.values() if not task.done()]
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError, Exception):
            await task
    _REGISTRY_PREPARE_TASKS.clear()


_MAX_NODE_TYPE_CHARACTERS = 200


def _prepared_node_types(values: list[str]) -> tuple[str, ...]:
    """The exact node identities this package must provide, or a typed refusal.

    Checked here rather than at the lifecycle because a job that must fail on
    its first step is a worse answer than a 422. Persistence refuses an empty
    or malformed set on purpose; nothing about that should be relaxed, so the
    same rules are applied before anything is queued.
    """
    cleaned = [value.strip() for value in values]
    if any(
        not value or len(value) > _MAX_NODE_TYPE_CHARACTERS or _has_control_character(value)
        for value in cleaned
    ):
        raise api_error(
            422,
            "workflow-package-node-type-invalid",
            "A node type must be non-empty, printable, and within the length limit.",
        )
    if len(set(cleaned)) != len(cleaned):
        raise api_error(
            422,
            "workflow-package-node-type-repeated",
            "The same node type was named more than once. Send each exactly once.",
        )
    return tuple(cleaned)


def _has_control_character(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)


def _analyzed_package_node_types(
    ui_graph: dict[str, Any], package_id: str, version: str
) -> tuple[str, ...]:
    """Derive one exact package identity from the submitted source graph."""

    try:
        analysis = analyze_comfyui_workflow_package(ui_graph)
    except WorkflowPackageError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    matches = [
        requirement
        for requirement in analysis.custom_packages
        if requirement.package_id == package_id
    ]
    if len(matches) != 1:
        raise api_error(
            422,
            "workflow-package-requirement-not-found",
            f"The workflow does not declare exactly one custom-node package named "
            f"{package_id}; it declares "
            + (", ".join(sorted(item.package_id for item in analysis.custom_packages)) or "none")
            + ".",
        )
    requirement = matches[0]
    if requirement.versions != (version,):
        # Both halves named. A workflow can declare a version that cannot be
        # installed at all - a Registry release whose own pins have no wheels
        # for this interpreter, say - and the reader's next move depends
        # entirely on knowing which version was asked for and which the
        # workflow wrote down. Saying only that they differ leaves them to go
        # and find both.
        declared = ", ".join(requirement.versions) or "no version"
        raise api_error(
            422,
            "workflow-package-version-mismatch",
            f"This asked to prepare {package_id} {version}, but the workflow declares "
            f"{declared}. A workflow is prepared with the version it names.",
        )
    return _prepared_node_types(list(requirement.node_types))


#: A comparison is only worth doing if it is bounded; a graph larger than this
#: is refused rather than serialized twice to find out it did not match.
MAX_COMPARED_GRAPH_CHARACTERS = 8_000_000


def _canonical_graph(graph: dict[str, Any]) -> str:
    """One bounded string for a graph, so two of them can be compared exactly.

    Node-type names alone are not the graph. Two workflows can require the
    same class names while declaring different packages, versions, or links -
    which is exactly the substitution this comparison exists to catch.
    """
    encoded = json.dumps(
        graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    if len(encoded) > MAX_COMPARED_GRAPH_CHARACTERS:
        raise api_error(
            422,
            "workflow-graph-too-large",
            "That workflow is too large to compare against the stored revision.",
        )
    return encoded


def _workflow_package_draft_identity(canonical_graph: str) -> tuple[str, str, str]:
    """Return stable local identities for one exact source graph."""

    digest = hashlib.sha256(canonical_graph.encode("utf-8")).hexdigest()
    short = digest[:24]
    return f"wfpkgdraft_{short}", f"wfpkgdrev_{short}", digest


def _workflow_with_revisions(session: Session, workflow_id: str) -> WorkflowDefinition:
    definition = session.scalar(
        select(WorkflowDefinition)
        .options(selectinload(WorkflowDefinition.revisions))
        .where(WorkflowDefinition.id == workflow_id)
    )
    if not definition:
        raise api_error(
            500,
            "workflow-reload-failed",
            "The workflow could not be reloaded.",
        )
    return definition


def _validated_workflow_package_draft(
    session: Session,
    payload: WorkflowPackageImportRequest,
) -> tuple[WorkflowDefinition, WorkflowRevision] | None:
    """Resolve only a draft derived from this exact submitted graph."""

    supplied = (payload.draft_workflow_id, payload.draft_revision_id)
    if not any(supplied):
        return None
    if not all(supplied):
        raise api_error(
            422,
            "workflow-package-draft-identity-incomplete",
            "Both workflow draft identities are required.",
        )
    canonical = _canonical_graph(payload.ui_graph)
    expected_workflow_id, initial_revision_id, digest = _workflow_package_draft_identity(canonical)
    if payload.draft_workflow_id != expected_workflow_id:
        raise api_error(
            422,
            "workflow-package-draft-identity-mismatch",
            "The workflow draft does not match the submitted package.",
        )
    definition = session.get(WorkflowDefinition, expected_workflow_id)
    initial_revision = session.get(WorkflowRevision, initial_revision_id)
    selected_revision = session.get(WorkflowRevision, payload.draft_revision_id)
    if (
        not definition
        or not initial_revision
        or not selected_revision
        or initial_revision.workflow_id != definition.id
        or selected_revision.workflow_id != definition.id
        or initial_revision.dependencies_json != workflow_package_draft_dependencies(digest)
        or _canonical_graph(initial_revision.ui_graph_json) != canonical
        or _canonical_graph(selected_revision.ui_graph_json) != canonical
    ):
        raise api_error(
            409,
            "workflow-package-draft-mismatch",
            "The stored workflow draft does not match the submitted package.",
        )
    return definition, initial_revision


def _authorized_workflow_context(
    session: Session, payload: WorkflowPackagePrepareRequest
) -> tuple[str, tuple[str, ...], dict[str, Any]] | None:
    """The revision an omission candidate is about, its node set, and its graph.

    Everything downstream comes from the stored graph. Re-analyzing what a
    caller sends proves that graph is internally consistent; it does not bind
    it to anything this machine saved, and a proof about a graph nobody stored
    is a proof about nothing.

    A submitted graph is compared whole rather than by the node types it needs.
    Comparing type names would let a caller submit one package's metadata for a
    class the stored graph attributes to another, prepare the first, and bind
    the proof to the second.
    """
    if not payload.workflow_revision_id:
        return None
    revision = session.get(WorkflowRevision, payload.workflow_revision_id)
    if not revision:
        raise api_error(404, "workflow-revision-not-found", "That workflow revision is unknown.")
    stored = revision.ui_graph_json
    if payload.ui_graph and _canonical_graph(payload.ui_graph) != _canonical_graph(stored):
        raise api_error(
            422,
            "workflow-graph-mismatch",
            "The submitted workflow does not match the stored revision it names.",
        )
    try:
        analysis = analyze_comfyui_workflow_package(stored)
    except WorkflowPackageError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    # The analyzer's complete set, not the selected package's declared subset:
    # the proof is that the workflow runs, and every type it needs is part of
    # that whether or not this package supplies it.
    return revision.id, tuple(sorted(set(analysis.required_node_types))), stored


async def _run_workflow_package_preparation(
    services: Services,
    job_id: str,
    package_id: str,
    version: str,
    node_types: tuple[str, ...],
    renew_install_id: str | None = None,
    authorized_workflow: tuple[str, tuple[str, ...]] | None = None,
) -> None:
    """One durable preparation job: lease held, worker state told truthfully."""

    def report(name: str, done: int | None, total: int | None) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            update_job_progress(
                job,
                stage=name,
                completed_units=done,
                total_units=total,
                unit="bytes" if total is not None else None,
                indeterminate=total is None,
            )
            session.commit()

    try:
        async with services.scheduler.job_lease(job_id, resource="media_compute", group="primary"):
            media_stopped = _media_worker_truly_stopped(services)
            # The composition opens its session only around the atomic
            # prepare step; resolution and closure run session-free.
            preparation = await prepare_workflow_package(
                SessionLocal,
                package_id=package_id,
                version=version,
                node_types=node_types,
                context=PreparationContext.from_settings(services.settings),
                media_worker_stopped=media_stopped,
                interpreter_probe=probe_comfy_registry_runtime_target,
                registry_client=ComfyRegistryClient(),
                project_client=ComfyRegistryWheelProjectClient(),
                metadata_client=ComfyRegistryWheelMetadataClient(),
                archive_downloader=ComfyRegistryArchiveDownloader(),
                wheel_downloader=ComfyRegistryWheelDownloader(),
                phase=report,
                renew_install_id=renew_install_id,
                authorized_workflow=authorized_workflow,
            )
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                if job and job.status != JobStatus.CANCELLED.value:
                    job.status = JobStatus.COMPLETE.value
                    job.payload_json = {
                        **job.payload_json,
                        "preparation": {
                            "install_id": preparation.install_id,
                            "installed_path": preparation.installed_path,
                            "wheel_environment_path": preparation.wheel_environment_path,
                            "archive_sha256": preparation.archive_sha256,
                            "manifest_sha256": preparation.manifest_sha256,
                            "wheel_closure_sha256": preparation.wheel_closure_sha256,
                            "wheel_environment_sha256": preparation.wheel_environment_sha256,
                            "reused_wheel_environment": preparation.reused_wheel_environment,
                        },
                    }
                    update_job_progress(
                        job,
                        stage=(
                            "Dependencies refreshed; trust unchanged"
                            if renew_install_id is not None
                            else "Prepared, inactive and untrusted"
                        ),
                    )
                session.commit()
    except asyncio.CancelledError:
        raise
    except WorkflowPackagePreparationError as exc:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if job and job.status != JobStatus.CANCELLED.value:
                job.status = JobStatus.FAILED.value
                job.error = str(exc)
                job.payload_json = {**job.payload_json, "error_code": exc.code}
                update_job_progress(job, stage="Preparation refused")
            session.commit()
    except Exception as exc:  # noqa: BLE001 - the job must never die silently
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if job and job.status != JobStatus.CANCELLED.value:
                job.status = JobStatus.FAILED.value
                job.error = str(exc)
                update_job_progress(job, stage="Preparation failed")
            session.commit()
    await services.scheduler.publish_job(job_id)


def _queue_registry_preparation(
    session: Session,
    services: Services,
    *,
    package_id: str,
    version: str,
    node_types: tuple[str, ...],
    renew_install_id: str | None = None,
    authorized_workflow: tuple[str, tuple[str, ...]] | None = None,
) -> Job:
    payload: dict[str, Any] = {
        "package_id": package_id,
        "version": version,
        "node_types": list(node_types),
    }
    if authorized_workflow is not None:
        revision_id, required_node_types = authorized_workflow
        payload["authorized_workflow"] = {
            "workflow_revision_id": revision_id,
            "required_node_types": list(required_node_types),
        }
    if renew_install_id is not None:
        payload["renew_install_id"] = renew_install_id
    job = Job(
        kind=JobKind.REGISTRY_PREPARE.value,
        status=JobStatus.QUEUED.value,
        payload_json=payload,
        cancellable=True,
    )
    session.add(job)
    session.commit()
    task = asyncio.create_task(
        _run_workflow_package_preparation(
            services,
            job.id,
            package_id,
            version,
            node_types,
            renew_install_id,
            authorized_workflow,
        ),
        name=f"registry-prepare-{job.id}",
    )
    _REGISTRY_PREPARE_TASKS[job.id] = task

    def _discard(done: asyncio.Task[None], key: str = job.id) -> None:
        _REGISTRY_PREPARE_TASKS.pop(key, None)

    task.add_done_callback(_discard)
    return job


@router.post("/workflows/packages/drafts", response_model=WorkflowOut, status_code=201)
async def ensure_workflow_package_draft(
    payload: WorkflowPackageDraftRequest, session: SessionDep
) -> WorkflowDefinition:
    """Persist an exact package graph without making it executable."""

    try:
        analyze_comfyui_workflow_package(payload.ui_graph)
    except WorkflowPackageError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    canonical = _canonical_graph(payload.ui_graph)
    workflow_id, revision_id, digest = _workflow_package_draft_identity(canonical)
    definition = session.get(WorkflowDefinition, workflow_id)
    revision = session.get(WorkflowRevision, revision_id)
    if bool(definition) != bool(revision):
        raise api_error(
            409,
            "workflow-package-draft-collision",
            "The workflow draft identity is already in use.",
        )
    if definition and revision:
        if (
            revision.workflow_id != definition.id
            or revision.dependencies_json != workflow_package_draft_dependencies(digest)
            or _canonical_graph(revision.ui_graph_json) != canonical
        ):
            raise api_error(
                409,
                "workflow-package-draft-collision",
                "The workflow draft identity is already in use.",
            )
        # Metadata remains editable while the graph is still only a draft.
        if definition.current_revision_id == revision.id:
            definition.name = payload.name
            definition.description = payload.description
            session.commit()
        return _workflow_with_revisions(session, definition.id)

    dependencies = workflow_package_draft_dependencies(digest)
    definition = WorkflowDefinition(
        id=workflow_id,
        name=payload.name,
        operation=payload.operation.value,
        description=payload.description,
    )
    session.add(definition)
    revision = WorkflowRevision(
        id=revision_id,
        workflow_id=workflow_id,
        version=1,
        engine="comfyui",
        ui_graph_json=payload.ui_graph,
        api_graph_json={},
        input_schema_json={},
        dependencies_json=dependencies,
        trusted=False,
        artifact_sha256=workflow_artifact_contract(
            operation=payload.operation.value,
            engine="comfyui",
            api_graph={},
            input_schema={},
            dependencies=dependencies,
        ),
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.commit()
    return _workflow_with_revisions(session, definition.id)


@router.post("/workflows/packages/prepare", response_model=JobOut, status_code=202)
async def prepare_workflow_package_endpoint(
    payload: WorkflowPackagePrepareRequest, request: Request, session: SessionDep
) -> Job:
    """Queue one package preparation; the result stays inactive and untrusted."""

    services = _services(request)
    # Re-analyze the source graph before judging the machine. The package name,
    # version, and node closure have to agree independently of browser state.
    # Resolved before queueing so an unknown or mismatched revision is a typed
    # refusal rather than a job that fails on its first step.
    authorized = _authorized_workflow_context(session, payload)
    # The stored graph chooses the package requirement as well as the node set.
    # Letting the submitted graph choose it would let a caller prepare one
    # package while binding the proof to a graph that names another.
    node_types = _analyzed_package_node_types(
        authorized[2] if authorized else payload.ui_graph,
        payload.package_id,
        payload.version,
    )
    # Then refuse when the machine cannot prepare at all - a job that must fail
    # on its first step is a worse answer than a typed 422.
    try:
        PreparationContext.from_settings(services.settings)
    except WorkflowPackagePreparationError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    return _queue_registry_preparation(
        session,
        services,
        package_id=payload.package_id,
        version=payload.version,
        node_types=node_types,
        authorized_workflow=(authorized[0], authorized[1]) if authorized else None,
    )


_REGISTRY_ACTIVATION_STATUS: dict[str, int] = {
    "registry_install_not_found": 404,
    "media_worker_running": 409,
    "registry_install_untrusted": 409,
    "registry_install_verification_failed": 409,
}


def _registry_activation_failure(exc: ComfyRegistryActivationError) -> Exception:
    return api_error(_REGISTRY_ACTIVATION_STATUS.get(exc.code, 500), exc.code, str(exc))


def _registry_install_out(
    install: ComfyRegistryInstall,
    disk_state: RegistryInstallDiskState,
) -> RegistryInstallOut:
    review = install.review_json if isinstance(install.review_json, dict) else {}
    reviewed_at = review.get("reviewed_at")
    activated_at = review.get("activated_at")
    return RegistryInstallOut(
        id=install.id,
        package_id=install.package_id,
        package_version=install.package_version,
        node_types=[str(node_type) for node_type in install.node_types_json],
        archive_sha256=install.archive_sha256,
        manifest_sha256=install.manifest_sha256,
        wheel_closure_sha256=install.wheel_closure_sha256,
        wheel_environment_sha256=install.wheel_environment_sha256,
        disk_status=disk_state.status,
        node_files_present=disk_state.node_files_present,
        wheel_environment_present=disk_state.wheel_environment_present,
        trusted=install.trusted,
        active=install.active,
        reviewed_at=reviewed_at if isinstance(reviewed_at, str) else None,
        activated_at=activated_at if isinstance(activated_at, str) else None,
        review=_registry_review_out(review),
    )


def _registry_review_out(review: Mapping[str, object]) -> RegistryInstallReviewOut | None:
    """Report what staging recorded, or nothing rather than a reassuring zero.

    An install prepared before this record existed has no findings to show.
    Rendering that as an empty list would read as "nothing to worry about",
    which is a different claim from "we did not look".
    """
    if not review.get("review_required"):
        return None
    return RegistryInstallReviewOut(
        file_count=_review_count(review.get("file_count")),
        expanded_bytes=_review_count(review.get("expanded_bytes")),
        python_file_count=_review_count(review.get("python_file_count")),
        install_scripts=_review_paths(review.get("install_scripts")),
        startup_hooks=_review_paths(review.get("startup_hooks")),
        native_files=_review_paths(review.get("native_files")),
        dependency_manifests=_review_paths(review.get("dependency_manifests")),
        top_level_entries=_review_paths(review.get("top_level_entries")),
        registry_warnings=_review_paths(review.get("registry_warnings")),
    )


def _review_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _review_paths(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)][:64]


def _loaded_registry_install(session: Session, install_id: str) -> ComfyRegistryInstall:
    install = session.get(ComfyRegistryInstall, install_id)
    if install is None:  # pragma: no cover - the activation call guarantees the row
        raise _registry_activation_failure(
            ComfyRegistryActivationError(
                "registry_install_not_found", "Registry package install was not found"
            )
        )
    return install


def _media_worker_truly_stopped(services: Services) -> bool:
    media = next(
        (status for status in services.processes.statuses() if status.name == "media"),
        None,
    )
    return media is None or (not media.running and media.state != "starting")


def _registry_activation_context(services: Services) -> PreparationContext:
    try:
        return PreparationContext.from_settings(services.settings)
    except WorkflowPackagePreparationError as exc:
        raise api_error(422, exc.code, str(exc)) from exc


def _registry_install_disk_state(
    services: Services,
    install: ComfyRegistryInstall,
) -> RegistryInstallDiskState:
    custom_node_root = (
        services.settings.comfy_directory / "custom_nodes"
        if services.settings.comfy_directory is not None
        else None
    )
    return inspect_registry_install_disk_state(
        install,
        custom_node_root=custom_node_root,
        environment_root=registry_wheel_environment_root(services.settings.registry_dir),
    )


@router.get("/workflows/packages/installs", response_model=list[RegistryInstallOut])
async def list_registry_installs(request: Request, session: SessionDep) -> list[RegistryInstallOut]:
    """List every prepared package with its trust and activation state."""

    services = _services(request)
    installs = session.scalars(
        select(ComfyRegistryInstall).order_by(
            ComfyRegistryInstall.package_id, ComfyRegistryInstall.package_version
        )
    ).all()
    return [
        _registry_install_out(install, _registry_install_disk_state(services, install))
        for install in installs
    ]


@router.post(
    "/workflows/packages/installs/{install_id}/renew",
    response_model=JobOut,
    status_code=202,
)
async def renew_registry_install(install_id: str, request: Request, session: SessionDep) -> Job:
    """Rebuild an inactive package's dependencies for the current media runtime."""

    services = _services(request)
    async with services.scheduler.lease("primary"):
        return _queue_registry_install_renewal(session, services, install_id)


def _queue_registry_install_renewal(
    session: Session,
    services: Services,
    install_id: str,
) -> Job:
    install = _loaded_registry_install(session, install_id)
    if install.active:
        raise api_error(
            409,
            "registry-install-active",
            "Deactivate the Registry package before refreshing its dependencies",
        )
    disk_state = _registry_install_disk_state(services, install)
    if not disk_state.node_files_present:
        raise api_error(
            409,
            "registry-install-files-missing",
            "Remove this incomplete Registry package and prepare it again",
        )
    if not _media_worker_truly_stopped(services):
        raise api_error(
            409,
            "media-worker-running",
            "Stop the media worker before refreshing Registry dependencies",
        )
    try:
        PreparationContext.from_settings(services.settings)
    except WorkflowPackagePreparationError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    node_types = _prepared_node_types(list(install.node_types_json))
    return _queue_registry_preparation(
        session,
        services,
        package_id=install.package_id,
        version=install.package_version,
        node_types=node_types,
        renew_install_id=install.id,
    )


@router.post(
    "/workflows/packages/installs/{install_id}/review",
    response_model=RegistryInstallOut,
)
async def review_registry_install(
    install_id: str,
    payload: RegistryInstallReviewRequest,
    request: Request,
    session: SessionDep,
) -> RegistryInstallOut:
    """Record the explicit local trust decision; revoking also deactivates."""

    services = _services(request)
    context = _registry_activation_context(services)
    async with services.scheduler.lease("primary"):
        try:
            review_comfy_registry_install(
                session,
                install_id=install_id,
                trusted=payload.trusted,
                custom_node_root=context.custom_node_root,
                environment_root=registry_wheel_environment_root(context.state_root),
                media_worker_stopped=_media_worker_truly_stopped(services),
            )
        except ComfyRegistryActivationError as exc:
            raise _registry_activation_failure(exc) from exc
    install = _loaded_registry_install(session, install_id)
    return _registry_install_out(install, _registry_install_disk_state(services, install))


@router.post(
    "/workflows/packages/installs/{install_id}/activate",
    response_model=RegistryInstallOut,
)
async def activate_registry_install(
    install_id: str, request: Request, session: SessionDep
) -> RegistryInstallOut:
    """Activate one trusted package; startup failure restores the prior runtime."""

    services = _services(request)
    context = _registry_activation_context(services)
    async with services.scheduler.lease("primary"):
        try:
            await activate_comfy_registry_install(
                session,
                install_id=install_id,
                custom_node_root=context.custom_node_root,
                environment_root=registry_wheel_environment_root(context.state_root),
                media_worker_stopped=_media_worker_truly_stopped(services),
                start_media=services.processes.start_media,
                # The same read startup verifies against, so a proof cannot be
                # made about an inventory nobody else saw.
                read_node_inventory=services.processes.comfy_node_inventory,
            )
        except ComfyRegistryActivationError as exc:
            raise _registry_activation_failure(exc) from exc
    install = _loaded_registry_install(session, install_id)
    return _registry_install_out(install, _registry_install_disk_state(services, install))


@router.post(
    "/workflows/packages/installs/{install_id}/deactivate",
    response_model=RegistryInstallOut,
)
async def deactivate_registry_install(
    install_id: str, request: Request, session: SessionDep
) -> RegistryInstallOut:
    """Deactivate one package and restart the media runtime without it."""

    services = _services(request)
    async with services.scheduler.lease("primary"):
        try:
            await deactivate_comfy_registry_install(
                session,
                install_id=install_id,
                media_worker_stopped=_media_worker_truly_stopped(services),
                start_media=services.processes.start_media,
            )
        except ComfyRegistryActivationError as exc:
            raise _registry_activation_failure(exc) from exc
    install = _loaded_registry_install(session, install_id)
    return _registry_install_out(install, _registry_install_disk_state(services, install))


_REGISTRY_REMOVAL_STATUS: dict[str, int] = {
    "registry_install_not_found": 404,
    "registry_install_active": 409,
    "registry_install_in_use": 409,
    "registry_install_busy": 409,
    "registry_install_path_invalid": 409,
    "registry_install_remove_failed": 500,
    "registry_install_restore_failed": 500,
}


@router.delete("/workflows/packages/installs/{install_id}", status_code=204)
async def delete_registry_install(
    install_id: str,
    request: Request,
    session: SessionDep,
) -> Response:
    """Remove one inactive prepared package and its exclusively owned files."""

    services = _services(request)
    custom_node_root = (
        services.settings.comfy_directory / "custom_nodes"
        if services.settings.comfy_directory is not None
        else None
    )
    async with services.scheduler.lease("primary"):
        try:
            remove_comfy_registry_install(
                session,
                install_id=install_id,
                custom_node_root=custom_node_root,
                environment_root=registry_wheel_environment_root(services.settings.registry_dir),
            )
        except ComfyRegistryReconciliationError as exc:
            raise api_error(
                _REGISTRY_REMOVAL_STATUS.get(exc.code, 500),
                exc.code,
                str(exc),
            ) from exc
    return Response(status_code=204)


def _rebuild_asset_binding(
    session: Session,
    ui_graph: dict[str, Any],
    selections: list[WorkflowAssetSelectionIn],
) -> tuple[WorkflowAssetBindingPlan, dict[str, InstallPlan]]:
    """Re-analyze and re-bind from current server records.

    Analysis is stateless by design and the browser keeps the graph, so a
    review is always recomputed here: the client's report, digests, sizes,
    kinds, and bound assets are never trusted - only its explicit id-based
    selections are.
    """

    try:
        analysis = analyze_comfyui_workflow_package(
            ui_graph,
            available_asset_filenames=_local_asset_filenames(session),
            installed_package_versions=_installed_package_versions(session),
        )
    except WorkflowPackageError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    plan_ids = {selection.install_plan_id for selection in selections}
    plans = {
        plan.id: plan
        for plan in session.scalars(select(InstallPlan).where(InstallPlan.id.in_(plan_ids))).all()
    }
    raw_selections = [
        WorkflowAssetPlanSelection(
            reference_filename=selection.reference_filename,
            install_plan_id=selection.install_plan_id,
            artifact_path=selection.artifact_path,
        )
        for selection in selections
    ]
    try:
        materialized_selections, plans = materialize_workflow_asset_aliases(
            session,
            analysis.asset_references,
            raw_selections,
            plans,
        )
        binding = bind_workflow_assets_to_install_plans(
            analysis.asset_references,
            materialized_selections,
            plans,
        )
    except (WorkflowAssetAliasError, WorkflowAssetBindingError) as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    return binding, plans


def _asset_review_out(binding: WorkflowAssetBindingPlan) -> WorkflowAssetReviewOut:
    return WorkflowAssetReviewOut(
        binding_plan_hash=binding.plan_hash,
        assets=[
            BoundWorkflowAssetOut(
                reference_filename=asset.reference_filename,
                kind=asset.kind,
                install_plan_id=asset.install_plan_id,
                install_plan_hash=asset.install_plan_hash,
                provider=asset.provider,
                remote_id=asset.remote_id,
                revision=asset.revision,
                artifact_path=asset.artifact_path,
                artifact_kind=asset.artifact_kind,
                target_folder=asset.target_folder,
                size_bytes=asset.size_bytes,
                sha256=asset.sha256,
            )
            for asset in binding.assets
        ],
        download_count=len({asset.install_plan_id for asset in binding.assets}),
        total_bytes=sum(asset.size_bytes for asset in binding.assets),
    )


def _workflow_install_offer_out(offer: WorkflowInstallOffer) -> WorkflowInstallOfferOut:
    return WorkflowInstallOfferOut(
        id=offer.id,
        workflow_revision_id=offer.workflow_revision_id,
        workflow_artifact_sha256=offer.workflow_artifact_sha256,
        dependency_contract_sha256=offer.dependency_contract_sha256,
        binding_plan_sha256=offer.binding_plan_sha256,
        offer_sha256=offer.offer_sha256,
        assets=[BoundWorkflowAssetOut.model_validate(asset) for asset in offer.assets_json],
        plan_count=offer.plan_count,
        total_bytes=offer.total_bytes,
        status=cast(
            "Literal['ready', 'queued', 'invalidated', 'completed', 'expired']",
            offer.status,
        ),
        queued_at=offer.queued_at,
        completed_at=offer.completed_at,
        invalidated_at=offer.invalidated_at,
        invalidation_code=offer.invalidation_code,
        invalidation_reason=offer.invalidation_reason,
    )


async def _workflow_install_inventory(
    request: Request,
    session: Session,
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    describe_nodes = getattr(_services(request).engines.media, "object_info", None)
    if not callable(describe_nodes):
        raise api_error(
            503,
            "media-runtime-unavailable",
            "Start the media worker to review workflow downloads.",
        )
    try:
        object_info = await describe_nodes()
    except Exception as exc:  # noqa: BLE001 - any runtime failure means unavailable
        raise api_error(
            503,
            "media-runtime-unavailable",
            "Start the media worker to review workflow downloads.",
        ) from exc
    if not isinstance(object_info, Mapping):
        raise api_error(
            503,
            "media-runtime-unavailable",
            "The media worker returned an invalid node inventory.",
        )
    return (
        {str(node_type) for node_type in object_info},
        _local_asset_filenames(session),
        _installed_package_versions(session),
    )


@router.post("/workflows/packages/assets/review", response_model=WorkflowAssetReviewOut)
async def review_workflow_assets(
    payload: WorkflowAssetReviewRequest, session: SessionDep
) -> WorkflowAssetReviewOut:
    """Bind explicit selections to immutable plans and report the cost."""

    binding, _plans = _rebuild_asset_binding(session, payload.ui_graph, payload.selections)
    # Alias plans are durable confirmation records. The install call rebuilds
    # the same plan from the original provider selection and must recover the
    # identical plan id/hash the user reviewed.
    session.commit()
    return _asset_review_out(binding)


@router.post("/workflows/packages/assets/install", response_model=list[JobOut], status_code=202)
async def install_workflow_assets(
    payload: WorkflowAssetQueueRequest, request: Request, session: SessionDep
) -> list[Job]:
    """Queue the reviewed binding: one download per distinct plan.

    Independent downloads are not a rollback unit. Every created or reused
    job is returned; a verified partial install stays reusable, while the
    workflow itself remains unresolved until a fresh analysis says every
    asset is present.
    """

    binding, plans = _rebuild_asset_binding(session, payload.ui_graph, payload.selections)
    try:
        requests = compose_workflow_asset_download_requests(
            binding, plans, expected_binding_plan_hash=payload.binding_plan_hash
        )
    except WorkflowAssetDownloadError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    manager: DownloadManager = _services(request).downloads
    # Validate every request before starting any: `create` commits and starts
    # each transfer, so refusing halfway would leave earlier downloads running
    # behind a 422 that claims nothing was queued.
    try:
        validated = [manager.validated_request(session, download) for download in requests]
    except ValueError as exc:
        raise api_error(422, "asset-download-refused", str(exc)) from exc
    jobs = [manager.create(session, download) for download in validated]
    session.commit()
    return jobs


@router.post(
    "/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
    response_model=WorkflowInstallOfferOut,
    status_code=201,
)
async def review_workflow_install_offer(
    workflow_id: str,
    revision_id: str,
    payload: WorkflowInstallOfferCreate,
    request: Request,
    session: SessionDep,
) -> WorkflowInstallOfferOut:
    """Persist one complete offer without accepting graph or digest claims."""

    node_types, asset_filenames, package_versions = await _workflow_install_inventory(
        request,
        session,
    )
    selections = [
        WorkflowAssetPlanSelection(
            reference_filename=selection.reference_filename,
            install_plan_id=selection.install_plan_id,
            artifact_path=selection.artifact_path,
        )
        for selection in payload.selections
    ]
    try:
        offer = create_workflow_install_offer(
            session,
            workflow_id=workflow_id,
            revision_id=revision_id,
            selections=selections,
            available_node_types=node_types,
            available_asset_filenames=asset_filenames,
            installed_package_versions=package_versions,
        )
    except WorkflowInstallOfferError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    session.commit()
    session.refresh(offer)
    return _workflow_install_offer_out(offer)


@router.post(
    "/workflow-install-offers/{offer_id}/install",
    response_model=list[JobOut],
    status_code=202,
)
async def install_workflow_offer(
    offer_id: str,
    request: Request,
    session: SessionDep,
) -> list[Job]:
    """Queue only an opaque offer after rebuilding every server-owned identity."""

    node_types, asset_filenames, package_versions = await _workflow_install_inventory(
        request,
        session,
    )
    try:
        offer, downloads = revalidate_workflow_install_offer(
            session,
            offer_id,
            available_node_types=node_types,
            available_asset_filenames=asset_filenames,
            installed_package_versions=package_versions,
        )
    except WorkflowInstallOfferError as exc:
        session.commit()
        raise api_error(422, exc.code, str(exc)) from exc

    manager: DownloadManager = _services(request).downloads
    try:
        validated = [manager.validated_request(session, download) for download in downloads]
    except ValueError as exc:
        invalidate_workflow_install_offer(
            offer,
            code="asset-download-refused",
            reason=str(exc),
        )
        session.commit()
        raise api_error(422, "asset-download-refused", str(exc)) from exc
    jobs = [manager.create(session, download) for download in validated]
    mark_workflow_install_offer_queued(offer)
    session.commit()
    return jobs


@router.post("/workflows/packages/import", response_model=WorkflowOut, status_code=201)
async def import_workflow_package(
    payload: WorkflowPackageImportRequest, request: Request, session: SessionDep
) -> WorkflowDefinition:
    """Compile a fully resolved ComfyUI package into an untrusted workflow.

    The one gate is the analyzer's own `ready`, re-computed here rather than
    trusted from the browser, and compilation needs the live runtime's node
    definitions - a graph compiled against guessed definitions could change
    behavior silently.
    """

    draft = _validated_workflow_package_draft(session, payload)
    services = _services(request)
    describe_nodes = getattr(services.engines.media, "object_info", None)
    if not callable(describe_nodes):
        raise api_error(
            503, "media-runtime-unavailable", "Start the media worker to compile workflows"
        )
    try:
        object_info = await describe_nodes()
    except Exception as exc:  # noqa: BLE001 - any runtime failure means "not up"
        raise api_error(
            503, "media-runtime-unavailable", "Start the media worker to compile workflows"
        ) from exc
    available_node_types = {str(node_type) for node_type in object_info}
    asset_filenames = _local_asset_filenames(session)
    package_versions = _installed_package_versions(session)
    try:
        analysis = analyze_comfyui_workflow_package(
            payload.ui_graph,
            available_node_types=available_node_types,
            available_asset_filenames=asset_filenames,
            installed_package_versions=package_versions,
        )
    except WorkflowPackageError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    verified_package_versions = package_versions
    if analysis.custom_packages:
        launchable_packages = await _verified_launchable_packages(services)
        launchable_node_types = _launchable_package_node_types(analysis, launchable_packages)
        verified_package_versions = _verified_package_versions(analysis, launchable_packages)
        try:
            analysis = analyze_comfyui_workflow_package(
                payload.ui_graph,
                available_node_types=available_node_types | launchable_node_types,
                available_asset_filenames=asset_filenames,
                installed_package_versions=verified_package_versions,
                packages_awaiting_review=await _packages_awaiting_review(services),
            )
        except WorkflowPackageError as exc:
            raise api_error(422, exc.code, str(exc)) from exc
    if not analysis.ready:
        raise api_error(
            422,
            "package-not-resolved",
            "Resolve everything in the package review before importing",
        )
    if not set(analysis.required_node_types) <= available_node_types:
        _ensure_worker_idle(session, "media")
        async with services.scheduler.lease("primary"):
            session.expire_all()
            _ensure_worker_idle(session, "media")
            try:
                await services.processes.start_media()
                object_info = await describe_nodes()
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                raise api_error(
                    422,
                    "workflow-package-node-refresh-failed",
                    "The installed custom-node packages could not be loaded for workflow import.",
                ) from exc
        available_node_types = {str(node_type) for node_type in object_info}
        try:
            analysis = analyze_comfyui_workflow_package(
                payload.ui_graph,
                available_node_types=available_node_types,
                available_asset_filenames=asset_filenames,
                installed_package_versions=verified_package_versions,
            )
        except WorkflowPackageError as exc:
            raise api_error(422, exc.code, str(exc)) from exc
        if not analysis.ready:
            # Name the package rather than the symptom. "A node is missing" is
            # true and useless: the node is missing because something that owns
            # it did not load, and that has a different fix from installing
            # anything at all.
            unloaded = _packages_that_did_not_load(services, analysis, available_node_types)
            if unloaded:
                # Partitioned rather than summarised. A set where one package
                # ships requirements and another does not has two causes, and
                # asserting either for both would be a confident wrong answer.
                needing = sorted(name for name, path in unloaded if path is not None)
                other = sorted(name for name, path in unloaded if path is None)
                # No pip command. Installing requirements by hand invalidates the
                # prepared runtime baseline, which the worker compares for
                # equality before it starts - so the command would fix the import
                # and stop the runtime, which is worse than saying nothing.
                remedy = (
                    " Prepare the package from the workflow package review:"
                    " preparation installs a package's dependency closure and records"
                    " the runtime's package baseline, which installing by hand cannot do."
                )
                if needing and not other:
                    listed = ", ".join(needing)
                    raise api_error(
                        422,
                        "custom-node-package-requirements-missing",
                        f"{listed} is installed and trusted but did not load. It declares"
                        " Python requirements, and installing a pinned repository does not"
                        " install what it imports, so its nodes never register with the"
                        f" runtime.{remedy}",
                    )
                if needing:
                    listed = ", ".join(needing)
                    others = ", ".join(other)
                    raise api_error(
                        422,
                        "custom-node-package-did-not-load",
                        f"{listed} and {others} are installed and trusted but did not load."
                        f" {listed} declares Python requirements that installing a pinned"
                        f" repository does not install. Why {others} failed is recorded in"
                        f" the media worker log.{remedy}",
                    )
                listed = ", ".join(other)
                raise api_error(
                    422,
                    "custom-node-package-did-not-load",
                    f"{listed} is installed and trusted but did not load, so the nodes it"
                    " provides are unavailable. The media worker log records why the import"
                    f" failed.{remedy}",
                )
            raise api_error(
                422,
                "package-not-resolved",
                "The refreshed media runtime did not load every required workflow node.",
            )
    try:
        prepared = prepare_workflow_package_compilation(
            payload.ui_graph,
            object_info,
            payload.operation,
        )
        compilation = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
        compiled_api_graph = prepared.bind(compilation.api_graph)
    except (
        WorkflowCompilationError,
        WorkflowPackageError,
        WorkflowPackageInputError,
    ) as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    input_schema = prepared.input_schema
    if draft:
        definition, initial_revision = draft
        current_revision = session.get(WorkflowRevision, definition.current_revision_id)
        if not current_revision:
            raise api_error(
                409,
                "workflow-package-draft-mismatch",
                "The stored workflow draft has no current revision.",
            )
        if current_revision.id != initial_revision.id:
            if (
                definition.operation != payload.operation.value
                or _canonical_graph(current_revision.ui_graph_json)
                != _canonical_graph(payload.ui_graph)
                or current_revision.api_graph_json != compiled_api_graph
                or current_revision.input_schema_json != input_schema
            ):
                raise api_error(
                    409,
                    "workflow-package-draft-already-finalized",
                    "The workflow draft was already finalized differently.",
                )
            definition.name = payload.name
            definition.description = payload.description
            session.commit()
            return _workflow_with_revisions(session, definition.id)
        if initial_revision.api_graph_json:
            raise api_error(
                409,
                "workflow-package-draft-mismatch",
                "The stored workflow draft is already executable.",
            )
        definition.name = payload.name
        definition.operation = payload.operation.value
        definition.description = payload.description
        session.flush()
        await create_workflow_revision(
            definition.id,
            WorkflowRevisionCreate(
                ui_graph=payload.ui_graph,
                api_graph=compiled_api_graph,
                input_schema=input_schema,
                trusted=False,
            ),
            session,
        )
        return _workflow_with_revisions(session, definition.id)
    return await create_workflow(
        WorkflowCreate(
            name=payload.name,
            operation=payload.operation,
            description=payload.description,
            engine="comfyui",
            ui_graph=payload.ui_graph,
            api_graph=compiled_api_graph,
            input_schema=input_schema,
            trusted=False,
        ),
        session,
    )


@router.post("/workflows/packages/analyze", response_model=WorkflowPackageAnalysisOut)
async def analyze_workflow_package(
    payload: WorkflowPackageAnalyzeRequest, request: Request, session: SessionDep
) -> WorkflowPackageAnalysisOut:
    """Report what a raw ComfyUI package needs, without persisting or executing it."""

    services = _services(request)
    available_node_types: set[str] = set()
    node_inventory_available = False
    describe_nodes = getattr(services.engines.media, "object_info", None)
    if callable(describe_nodes):
        try:
            object_info = await describe_nodes()
            available_node_types = {str(node_type) for node_type in object_info}
            node_inventory_available = True
        except Exception:  # noqa: BLE001
            # The media runtime is not up, so node availability is unknown.
            # The report says so instead of failing - and instead of letting
            # "every node missing" masquerade as a finding.
            node_inventory_available = False
    asset_filenames = _local_asset_filenames(session)
    package_versions = _installed_package_versions(session)
    try:
        analysis = analyze_comfyui_workflow_package(
            payload.ui_graph,
            available_node_types=available_node_types,
            available_asset_filenames=asset_filenames,
            installed_package_versions=package_versions,
        )
    except WorkflowPackageError as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    verified_package_versions = package_versions
    if analysis.custom_packages:
        launchable_packages = await _verified_launchable_packages(services)
        launchable_node_types = _launchable_package_node_types(analysis, launchable_packages)
        verified_package_versions = _verified_package_versions(analysis, launchable_packages)
        try:
            analysis = analyze_comfyui_workflow_package(
                payload.ui_graph,
                available_node_types=available_node_types | launchable_node_types,
                available_asset_filenames=asset_filenames,
                installed_package_versions=verified_package_versions,
                packages_awaiting_review=await _packages_awaiting_review(services),
            )
        except WorkflowPackageError as exc:
            raise api_error(422, exc.code, str(exc)) from exc
    # Authors often record where a model came from. A filename search cannot
    # find a file inside a repository, so those links are frequently the only
    # way an asset is findable at all - read from the graph, validated here,
    # and still resolved by the ordinary preflight path.
    candidates = collect_source_candidates(
        payload.ui_graph,
        allowed_hosts=services.catalog_sources.host_map(),
        asset_filenames=[asset.filename for asset in analysis.asset_references],
    )
    return WorkflowPackageAnalysisOut(
        format_version=analysis.format_version,
        frontend_version=analysis.frontend_version,
        node_count=analysis.node_count,
        link_count=analysis.link_count,
        subgraph_count=analysis.subgraph_count,
        operation_guess=analysis.operation_guess,
        truncated=analysis.truncated,
        required_node_types=list(analysis.required_node_types),
        frontend_node_types=list(analysis.frontend_node_types),
        missing_node_types=list(analysis.missing_node_types),
        missing_nodes=[
            WorkflowMissingNodeOut(
                node_type=missing.node_type,
                count=missing.count,
                package_id=missing.package_id,
            )
            for missing in analysis.missing_nodes
        ],
        custom_packages=[
            WorkflowPackageRequirementOut(
                package_id=package.package_id,
                versions=list(package.versions),
                node_types=list(package.node_types),
                locally_resolved=package.locally_resolved,
            )
            for package in analysis.custom_packages
        ],
        asset_references=[
            WorkflowAssetReferenceOut(
                filename=asset.filename,
                suffix=asset.suffix,
                policy=asset.policy,
                kind=asset.kind,
                source_url=asset.source_url,
                present_locally=asset.present_locally,
                source_candidates=[
                    WorkflowSourceCandidateOut(**candidate.as_dict())
                    for candidate in candidates.get(asset.filename, ())
                ],
            )
            for asset in analysis.asset_references
        ],
        issues=[
            WorkflowPackageIssueOut(
                code=issue.code,
                count=issue.count,
                node_types=list(issue.node_types),
                severity=issue.severity,
            )
            for issue in analysis.issues
        ],
        ready=analysis.ready,
        runtime_nodes_available=analysis.runtime_nodes_available,
        dependencies_resolved=analysis.dependencies_resolved,
        source_candidates=[
            WorkflowSourceCandidateOut(**candidate.as_dict())
            for candidate in candidates.get("", ())
        ],
        node_inventory_available=node_inventory_available,
    )


@router.post("/workflows/import", response_model=WorkflowOut, status_code=201)
async def import_workflow(payload: WorkflowBundle, session: SessionDep) -> WorkflowDefinition:
    return await create_workflow(
        WorkflowCreate(
            name=payload.name,
            operation=payload.operation,
            description=payload.description,
            engine=payload.engine,
            engine_version=payload.engine_version,
            ui_graph=payload.ui_graph,
            api_graph=payload.api_graph,
            input_schema=payload.input_schema,
            dependencies=payload.dependencies,
            trusted=False,
        ),
        session,
    )


@router.post("/workflows/{workflow_id}/clone", response_model=WorkflowOut, status_code=201)
async def clone_workflow(
    workflow_id: str, payload: WorkflowClone, session: SessionDep
) -> WorkflowDefinition:
    definition, revision = _workflow_and_revision(session, workflow_id)
    return await create_workflow(
        WorkflowCreate(
            name=payload.name or f"{definition.name} copy",
            operation=Operation(definition.operation),
            description=definition.description,
            engine=revision.engine,
            engine_version=revision.engine_version,
            ui_graph=revision.ui_graph_json,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
            trusted=revision.trusted,
        ),
        session,
    )


@router.post(
    "/workflows/{workflow_id}/revisions",
    response_model=WorkflowRevisionOut,
    status_code=201,
)
async def create_workflow_revision(
    workflow_id: str, payload: WorkflowRevisionCreate, session: SessionDep
) -> WorkflowRevision:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition:
        raise api_error(404, "workflow-not-found", "workflow not found")
    try:
        validate_lora_workflow_contract(
            payload.api_graph,
            payload.input_schema,
            payload.dependencies,
        )
        validate_workflow_edit_calibration(payload.input_schema)
    except ValueError as exc:
        raise api_error(422, "workflow-revision-invalid", str(exc)) from exc
    version = (
        session.scalar(
            select(func.max(WorkflowRevision.version)).where(
                WorkflowRevision.workflow_id == workflow_id
            )
        )
        or 0
    )
    current = session.get(WorkflowRevision, definition.current_revision_id)
    engine = current.engine if current else "comfyui"
    revision = WorkflowRevision(
        workflow_id=workflow_id,
        version=version + 1,
        engine=engine,
        engine_version=payload.engine_version,
        ui_graph_json=payload.ui_graph,
        api_graph_json=payload.api_graph,
        input_schema_json=payload.input_schema,
        dependencies_json=payload.dependencies,
        trusted=payload.trusted,
        artifact_sha256=workflow_artifact_contract(
            operation=definition.operation,
            engine=engine,
            api_graph=payload.api_graph,
            input_schema=payload.input_schema,
            dependencies=payload.dependencies,
        ),
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    ensure_workflow_family_ownership(session, definition, revision)
    session.commit()
    session.refresh(revision)
    return revision


@router.post(
    "/workflows/{workflow_id}/revisions/{revision_id}/restore",
    response_model=WorkflowRevisionOut,
    status_code=201,
)
async def restore_workflow_revision(
    workflow_id: str, revision_id: str, session: SessionDep
) -> WorkflowRevision:
    source = session.get(WorkflowRevision, revision_id)
    if not source or source.workflow_id != workflow_id:
        raise api_error(404, "workflow-revision-not-found", "workflow revision not found")
    return await create_workflow_revision(
        workflow_id,
        WorkflowRevisionCreate(
            engine_version=source.engine_version,
            ui_graph=source.ui_graph_json,
            api_graph=source.api_graph_json,
            input_schema=source.input_schema_json,
            dependencies=source.dependencies_json,
            trusted=source.trusted,
        ),
        session,
    )


@router.post("/workflows/{workflow_id}/validate")
async def validate_workflow(
    workflow_id: str, request: Request, session: SessionDep
) -> dict[str, Any]:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition or not definition.current_revision_id:
        raise api_error(404, "workflow-not-found", "workflow not found")
    revision = session.get(WorkflowRevision, definition.current_revision_id)
    if not revision:
        raise api_error(404, "workflow-revision-not-found", "workflow revision not found")
    errors = await _services(request).engines.media.validate_workflow(revision.api_graph_json)
    warnings: list[str] = []
    if revision.engine == "comfyui" and not revision.trusted:
        errors.append(
            "workflow revision is not trusted; review it and create a trusted revision "
            "before execution"
        )
    # Asking the engine what a role offers is I/O and can fail for reasons that
    # have nothing to do with this schema. It stays outside the block below so
    # its failure cannot be reported as "invalid workflow input schema", and so
    # whatever it says about the engine's internals does not reach the caller
    # through an error text meant to describe the user's own document.
    role = "video" if "video" in definition.operation else "image"
    base_fields = await _engine_role_fields(
        request,
        role,
        engine=revision.engine,
        allow_inactive=True,
    )
    try:
        declared_fields = workflow_settings(base_fields, revision.input_schema_json)
        validate_settings(defaults(declared_fields), declared_fields)
        validate_workflow_edit_calibration(revision.input_schema_json)
    except ValueError as exc:
        # Explaining why a schema is invalid is what this endpoint is for, so the
        # message survives - but it now comes only from our own validators,
        # which describe the document the caller supplied.
        errors.append(f"invalid workflow input schema: {exc}")
    dependencies = revision.dependencies_json
    errors.extend(node_dependency_errors(session, dependencies))
    required_models = dependencies.get("models", [])
    installed = session.scalars(select(ModelInstall).where(ModelInstall.active.is_(True))).all()
    installed_values = {
        value for model in installed for value in (model.id, model.name, model.local_path)
    }
    for dependency in required_models if isinstance(required_models, list) else []:
        value = (
            dependency.get("id") or dependency.get("name") or dependency.get("path")
            if isinstance(dependency, dict)
            else dependency
        )
        if value and str(value) not in installed_values:
            errors.append(f"missing model dependency: {value}")
    required_engine_version = dependencies.get("engine_version")
    if required_engine_version:
        capabilities = await _services(request).engines.media.capabilities()
        if str(required_engine_version) != capabilities.version:
            errors.append(
                "media engine version does not match the workflow requirement: "
                f"expected {required_engine_version}, found {capabilities.version}"
            )
    minimum_vram = dependencies.get("minimum_vram_bytes")
    if isinstance(minimum_vram, int) and minimum_vram > 0:
        system = collect_system_info(_services(request).settings)
        accelerators = [device for device in system.devices if device.kind != "cpu"]
        capacity = max(
            (device.total_memory_bytes or 0 for device in accelerators),
            default=0,
        )
        available_values = [
            device.available_memory_bytes
            for device in accelerators
            if device.available_memory_bytes is not None
        ]
        available = max(available_values, default=None)
        if not capacity:
            warnings.append("no accelerator memory capacity was detected for this workflow")
        elif capacity < minimum_vram:
            errors.append("accelerator memory capacity is below the workflow requirement")
        elif available is not None and available < minimum_vram:
            warnings.append(
                "currently available accelerator memory is below the workflow requirement"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "revision_id": revision.id,
    }


def _workflow_and_revision(
    session: Session, workflow_id: str
) -> tuple[WorkflowDefinition, WorkflowRevision]:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition or not definition.current_revision_id:
        raise api_error(404, "workflow-not-found", "workflow not found")
    revision = session.get(WorkflowRevision, definition.current_revision_id)
    if not revision:
        raise api_error(404, "workflow-revision-not-found", "workflow revision not found")
    return definition, revision


def _workflow_bundle(definition: WorkflowDefinition, revision: WorkflowRevision) -> WorkflowBundle:
    return WorkflowBundle(
        name=definition.name,
        operation=Operation(definition.operation),
        description=definition.description,
        engine=revision.engine,
        engine_version=revision.engine_version,
        ui_graph=revision.ui_graph_json,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
        dependencies=revision.dependencies_json,
        trusted=revision.trusted,
        source_revision=revision.version,
    )
