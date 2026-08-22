export type RoutingMode = "auto" | "text" | "image" | "video";
export type EngineRole = "chat" | "image" | "video";
export type GenerationSettingsByRole = Partial<Record<EngineRole, Record<string, unknown>>>;
export type GenerationPresetIdsByRole = Partial<Record<EngineRole, string | null>>;

export interface Project {
  id: string;
  name: string;
  description: string;
  instructions: string;
  archived: boolean;
  pinned: boolean;
  image_workflow_revision_id: string | null;
  video_workflow_revision_id: string | null;
  generation_settings_json?: GenerationSettingsByRole;
  generation_preset_ids_json?: GenerationPresetIdsByRole;
  created_at: string;
  updated_at: string;
}

export interface Artifact {
  id: string;
  sha256: string;
  kind: string;
  media_type: string;
  size_bytes: number;
  original_name: string | null;
  metadata_json: Record<string, unknown>;
  /** Pins against automatic cleanup only; explicit deletion always wins.
   * Optional so view fixtures stay lean; the server always sends it. */
  favorite?: boolean;
  created_at: string;
  url?: string | null;
  generation_identity?: GenerationIdentity | null;
}

export interface ArtifactLibraryItem extends Artifact {
  reference_count: number;
  chat_ids: string[];
  project_ids: string[];
}

export interface GenerationIdentity {
  model_profile_name: string | null;
  workflow_family_name: string | null;
  workflow_definition_name: string | null;
  workflow_version: number | null;
}

export interface ArtifactStorageInfo {
  total_bytes: number;
  total_count: number;
  referenced_bytes: number;
  referenced_count: number;
  unreferenced_bytes: number;
  unreferenced_count: number;
  temporary_bytes: number;
  temporary_count: number;
  eligible_bytes: number;
  eligible_count: number;
  retention_pending_count?: number;
  disk_free_bytes: number;
  warning: boolean;
  retention_days: number;
  temporary_retention_hours: number;
}

export interface ArtifactCleanupResult {
  dry_run: boolean;
  marked_count: number;
  retention_pending_count: number;
  removed_count: number;
  reclaimed_bytes: number;
}

export interface ArtifactDeleteResult {
  artifact_id: string;
  reference_count: number;
  removed_count: number;
  reclaimed_bytes: number;
}

export interface MessagePart {
  id: string;
  position: number;
  type: "text" | "image" | "video" | "attachment" | "progress" | "error" | "generation_metadata";
  text: string | null;
  artifact_id: string | null;
  metadata_json: Record<string, unknown>;
  artifact?: Artifact | null;
}

export interface Message {
  id: string;
  chat_id: string;
  parent_id: string | null;
  role: "user" | "assistant" | "system" | "tool";
  status: "complete" | "pending" | "failed" | "cancelled";
  transcript_visible?: boolean;
  active_response_revision_id?: string | null;
  parts: MessagePart[];
  /** What this turn referred to, as it stood when the turn was accepted.
   *  Empty for every message that named nothing, which is almost all of them. */
  references?: MessageReference[];
  response_revisions?: ResponseRevision[];
  /** The local preference verdict on the base response; revisions carry their own. */
  feedback?: "up" | "down" | null;
  created_at: string;
  updated_at: string;
}

/** One recorded reference, snapshotted rather than joined.
 *
 * The name and mention are the values the turn used, not the subject's current
 * ones, and `reference_subject_id` carries no promise the subject still
 * exists. Rendering a mention from this - never by scanning the message text -
 * is what keeps a renamed subject from rewriting an old message, and a deleted
 * one from erasing the record that it was used.
 */
export interface MessageReference {
  reference_subject_id: string;
  mention_slug: string;
  subject_name: string;
  subject_kind: string;
  role?: string | null;
  strength?: number | null;
  source: string;
  reference_asset_ids_json: string[];
  artifact_ids_json: string[];
}

export interface ResponseRevision {
  id: string;
  message_id: string;
  run_id: string | null;
  sequence: number;
  status: "complete" | "pending" | "failed" | "cancelled";
  parts: MessagePart[];
  feedback?: "up" | "down" | null;
  created_at: string;
  updated_at: string;
}

export interface Chat {
  id: string;
  project_id: string | null;
  title: string;
  archived: boolean;
  pinned: boolean;
  routing_mode: RoutingMode;
  confirm_uncertain_media: boolean;
  active_chat_profile_id: string | null;
  active_vision_profile_id?: string | null;
  active_image_profile_id: string | null;
  active_video_profile_id: string | null;
  active_head_message_id: string | null;
  vision_settings_json?: Record<string, unknown>;
  /** Whether this conversation may reach the internet. Never inherited. */
  web_settings_json?: WebSettings;
  generation_settings_json?: GenerationSettingsByRole;
  generation_preset_ids_json?: GenerationPresetIdsByRole;
  // Empty unless this chat was forked from a message in another one.
  origin_json?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ChatDetail extends Chat {
  messages: Message[];
}
export interface PromptHelperDetail extends ChatDetail {
  draft_prompt: string;
}

export interface Run {
  id: string;
  idempotency_key: string | null;
  chat_id: string;
  user_message_id: string;
  assistant_message_id: string;
  work_plan_id?: string | null;
  work_step_id?: string | null;
  operation: string;
  status: string;
  standalone_prompt: string;
  profile_id: string | null;
  vision_profile_id?: string | null;
  workflow_revision_id: string | null;
  settings_json: Record<string, unknown>;
  provenance_json: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
}

export interface TurnAccepted {
  run: Run;
  user_message: Message;
  assistant_message: Message;
}

export interface Job {
  id: string;
  kind: string;
  status: string;
  run_id: string | null;
  work_plan_id?: string | null;
  work_step_id?: string | null;
  progress: number;
  phase: string;
  progress_json?: ProgressV2;
  queue_resource?: string | null;
  queue_group?: string | null;
  queue_priority?: number;
  queue_ticket?: string | null;
  enqueued_at?: string | null;
  claim_expires_at?: string | null;
  heartbeat_at?: string | null;
  payload_json: Record<string, unknown>;
  result_json: Record<string, unknown>;
  error: string | null;
  attempt: number;
  cancellable: boolean;
  created_at: string;
  updated_at: string;
  // Added to JobOut by #220 for phase durations; the browser type never
  // gained them, so the data was unreachable until the contract gate caught it.
  started_at: string | null;
  completed_at: string | null;
}

export interface ProgressV2 {
  version: 2;
  stage: string;
  stage_progress: number | null;
  overall_progress: number | null;
  completed_units: number | null;
  total_units: number | null;
  unit: string | null;
  bytes_reused: number;
  rate_bytes_per_second: number | null;
  eta_seconds: number | null;
  file_index: number | null;
  file_count: number | null;
  queue_resource: string | null;
  queue_position: number | null;
  queue_length: number | null;
  blocked_by: string[];
  indeterminate: boolean;
  stage_started_at?: string | null;
  stage_elapsed_ms?: number;
  completed_stages?: Array<{
    stage: string;
    duration_ms: number;
  }>;
  updated_at: string;
}

export interface WorkStep {
  id: string;
  plan_id: string;
  run_id: string | null;
  ordinal: number;
  display_group: string | null;
  operation: string;
  status: string;
  prompt: string;
  profile_id: string | null;
  workflow_revision_id: string | null;
  settings_json: Record<string, unknown>;
  input_bindings_json: Array<Record<string, unknown>>;
  output_contract_json: Array<Record<string, unknown>>;
  queue_class: string;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface WorkPlan {
  id: string;
  chat_id: string;
  idempotency_key: string | null;
  source_action: string;
  persistence_scope: "durable";
  status: string;
  context_head_message_id: string | null;
  transcript_sequence: number;
  priority: number;
  planner_version: string;
  failure_policy: string;
  summary_json: Record<string, unknown>;
  steps: WorkStep[];
  created_at: string;
  updated_at: string;
}

export interface SettingField {
  key: string;
  label: string;
  type: "boolean" | "integer" | "number" | "string" | "enum" | "array" | "object";
  default: unknown;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
  multiple_of?: number | null;
  choices: unknown[];
  scope: "load" | "request" | "workflow";
  visibility: "basic" | "advanced" | "expert";
  restart_required: boolean;
  available: boolean;
  unavailable_reason: string | null;
  help: string;
}

export interface EngineCapabilities {
  engine: string;
  version: string;
  roles: string[];
  operations: string[];
  formats: string[];
  devices: string[];
  input_modalities?: string[];
  streaming: boolean;
  tool_calling: boolean;
  settings: SettingField[];
  settings_by_role?: Partial<Record<EngineRole, SettingField[]>>;
  healthy: boolean;
  details: Record<string, unknown>;
}

export interface ToolCapabilityProbe {
  engine: string;
  version: string;
  advertised: boolean;
  passed: boolean;
  tool_name: string | null;
  arguments: Record<string, unknown> | null;
  error: string | null;
}

export interface ModelProfile {
  id: string;
  model_install_id: string | null;
  name: string;
  use_case: string;
  role: "chat" | "image" | "video";
  engine: string;
  load_settings_json: Record<string, unknown>;
  request_settings_json: Record<string, unknown>;
  is_default: boolean;
  input_modalities?: string[];
}

export interface ModelProfileBundle {
  format: "lm-atelier-profile";
  version: 1;
  name: string;
  use_case: string;
  role: "chat" | "image" | "video";
  engine: string;
  model_install_id: string | null;
  load_settings: Record<string, unknown>;
  request_settings: Record<string, unknown>;
}

export interface GenerationPreset {
  id: string;
  name: string;
  role: "chat" | "image" | "video";
  settings_json: Record<string, unknown>;
  is_default: boolean;
}

export interface GenerationPresetBundle {
  format: "lm-atelier-preset";
  version: 1;
  name: string;
  role: "chat" | "image" | "video";
  settings: Record<string, unknown>;
}

export interface WorkerStatus {
  name: "chat" | "media";
  state: "stopped" | "starting" | "ready" | "exited";
  managed: boolean;
  running: boolean;
  pid: number | null;
  profile_id: string | null;
  command: string[];
  exit_code: number | null;
  estimated_memory_bytes: number | null;
  startup_duration_ms?: number | null;
  current_memory_bytes: number | null;
  peak_memory_bytes: number | null;
  active_jobs: number;
  queued_jobs: number;
  failure_detail?: string | null;
  failure_code?:
    | "oom_vram"
    | "oom_host"
    | "port_in_use"
    | "model_incompatible"
    | "executable_missing"
    | "startup_timeout"
    | "crashed"
    | "unknown"
    | null;
  failure_remedy?: string | null;
  stderr_tail?: string | null;
  log_path?: string | null;
}

export interface WorkerSettings {
  worker_startup_seconds: number;
}

export interface WorkerResetResult {
  worker: WorkerStatus;
  cancelled_jobs: number;
}

export interface WorkerLogTail {
  name: "chat" | "media";
  text: string;
  truncated: boolean;
  log_bytes: number;
}

export interface WorkerLogLocation {
  path: string;
}

export interface RuntimeStatus {
  engine: "llama.cpp" | "vllm" | "comfyui";
  release: string;
  state: "missing" | "installing" | "ready" | "failed" | "unsupported";
  supported: boolean;
  managed: boolean;
  progress: number;
  progress_json?: ProgressV2 | null;
  downloaded_bytes: number;
  size_bytes: number | null;
  distribution: string;
  license: string;
  security_status?: "checksum-pinned" | "blocked";
  security_message?: string;
  message: string;
}

export interface SetupReadinessCheck {
  code: string;
  status: "pass" | "pending" | "fail";
  message: string;
  action: string | null;
}

export interface SetupRoleReadiness {
  role: "chat" | "image" | "video";
  state: "ready" | "in_progress" | "action_required";
  verification_level: "generation_probe";
  engine: string | null;
  job_id: string | null;
  verification_id: string | null;
  install_id: string | null;
  profile_id: string | null;
  workflow_revision_id: string | null;
  next_action: string | null;
  checks: SetupReadinessCheck[];
}

export interface SetupReadinessReport {
  version: 2;
  state: "ready" | "in_progress" | "action_required";
  roles: SetupRoleReadiness[];
}

export interface SetupVerification {
  id: string;
  role: "chat" | "image" | "video";
  state: "queued" | "running" | "ready" | "failed";
  job_id: string | null;
  failure_code: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface BackupInfo {
  name: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
  verified: boolean;
  restore_pending: boolean;
  media_included: boolean;
  media_size_bytes: number;
}

export interface ModelInstall {
  id: string;
  source_id: string | null;
  name: string;
  role: string;
  engine: string;
  local_path: string;
  size_bytes: number;
  compatibility: string;
  manifest_json: Record<string, unknown>;
  active: boolean;
  readiness: "ready" | "unverified" | "unsupported";
  capability_evidence: {
    id: string;
    evidence_key: string;
    result: string;
    runtime_build: string;
    probed_at: string;
  } | null;
  created_at: string;
  updated_at: string;
}

export interface ModelAssetInstall {
  id: string;
  source_id: string | null;
  name: string;
  kind: string;
  family: string | null;
  size_bytes: number;
  manifest_json: Record<string, unknown>;
  active: boolean;
  use_case: string;
  auto_apply: boolean;
  default_model_strength: number;
  default_clip_strength: number;
  verified_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ModelStorageInfo {
  installed_bytes: number;
  partial_download_bytes: number;
  catalog_cache_bytes: number;
  installed_count: number;
  partial_download_count: number;
}

export interface StorageCleanupResult {
  removed_count: number;
  reclaimed_bytes: number;
}

export interface CatalogModel {
  provider: string;
  remote_id: string;
  name: string;
  author: string | null;
  pipeline_tag: string | null;
  tags: string[];
  downloads: number | null;
  likes: number | null;
  trending_score: number | null;
  created_at: string | null;
  last_modified: string | null;
  gated: boolean | string | null;
  private: boolean;
  library_name: string | null;
  architecture: string | null;
  formats: string[];
  quantizations: string[];
  parameter_count: number | null;
  license_id: string | null;
  total_size_bytes: number | null;
  compatibility: string;
  compatibility_reasons: string[];
  required_runtime?: string | null;
  /** The model this card is a version of, when the provider has one.
   *
   * A CivitAI card is one version, because a version is what installs and
   * what the download path binds to. These let the library list versions
   * under one parent without giving that up. Absent where the repository is
   * already the installable thing, and a card without them renders as before.
   */
  parent_model_id?: string | null;
  parent_model_name?: string | null;
  /** How many versions this card stands for. More than one means the card
   * must open a chooser rather than install, because the point of version
   * identity is that the person picked one. */
  version_count?: number;
  /** How many of them are already here, or absent when that cannot be known.
   * Not zero: a kind that records no provider version cannot be matched
   * against these at all, and "0 of 12" would be a claim the data cannot
   * support. */
  installed_version_count?: number | null;
  workflow_template_id?: string | null;
  operation?: string | null;
  content_rating?: ContentRating;
}

export type ContentRating = "general" | "mature" | "unknown";

export interface ExchangeDeletion {
  chat_id: string;
  user_message_id: string;
  message_ids: string[];
  run_ids: string[];
  job_ids: string[];
  work_plan_ids: string[];
  released_artifact_ids: string[];
  retained_artifact_ids: string[];
  new_head_message_id: string | null;
}

export interface CatalogVersionRow {
  version_id: string;
  version_name?: string | null;
  published_at?: string | null;
  base_model?: string | null;
  size_bytes: number;
  changelog?: string | null;
  /** True, false, or unknown - and unknown is a real answer, not a default.
   * Checkpoint installs record no provider version, so for those we cannot
   * tell. Rendering unknown as "not installed" is how someone installs a
   * second copy of what they already have. */
  installed?: boolean | null;
  installed_as?: string | null;
}

export interface CatalogVersions {
  model_id: string;
  model_name?: string | null;
  versions: CatalogVersionRow[];
}

export interface CatalogPage {
  items: CatalogModel[];
  next_cursor: string | null;
  stale?: boolean;
}

export interface CatalogDetail {
  model: CatalogModel;
  revision: string;
  files: Array<{ filename: string; size: number | null; sha256: string | null }>;
}

export interface CatalogFileVariant {
  source_file_id: string;
  filename: string;
  size_bytes: number | null;
  precision: string | null;
}

export interface CatalogPreflight {
  remote_id: string;
  source_remote_id: string | null;
  revision: string;
  selected_files: string[];
  expected_sha256: Record<string, string>;
  file_sources?: Record<string, {
    remote_id: string;
    revision: string;
    filename: string;
    size_bytes: number | null;
    sha256: string | null;
  }>;
  comfy_paths: Record<string, string>;
  workflow_template_id: string | null;
  workflow_template_sha256: string | null;
  download_bytes: number;
  available_disk_bytes: number;
  estimated_ram_bytes: number | null;
  estimated_vram_bytes: number | null;
  can_install: boolean;
  /** The choices behind a filename this version could not settle. Present
   * only for names that are genuinely ambiguous, so a list of one never
   * turns an ordinary install into a decision. */
  file_variants?: Record<string, CatalogFileVariant[]>;
  auxiliary_kind?: string | null;
  content_rating?: ContentRating;
  install_plan: {
    id: string;
    plan_hash: string;
    compatibility: "supported" | "unsupported" | "trusted_extension_required";
    family: string | null;
    failure_code: string | null;
    failure_reason: string | null;
  } | null;
  checks: Array<{
    id: string;
    label: string;
    status: "pass" | "warn" | "block";
    detail: string;
  }>;
}

/** One missing workflow file bound to an exact plan artifact. */
export interface BoundWorkflowAsset {
  reference_filename: string;
  kind: string;
  install_plan_id: string;
  install_plan_hash: string;
  provider: string;
  remote_id: string;
  revision: string;
  artifact_path: string;
  artifact_kind: string;
  target_folder: string;
  size_bytes: number;
  sha256: string;
}

/** The server's review of a selection set; the hash confirms the queue. */
export interface WorkflowAssetReview {
  binding_plan_hash: string;
  assets: BoundWorkflowAsset[];
  download_count: number;
  total_bytes: number;
}

export interface RecipeFile {
  path: string;
  size_bytes: number | null;
  sha256: string | null;
}

export interface ReferenceRecipe {
  id: string;
  version: number;
  name: string;
  summary: string;
  role: "chat" | "image" | "video";
  engine: "llama.cpp" | "vllm" | "comfyui";
  operations: string[];
  license_id: string;
  status: "reference-candidate" | "certified";
  certified: boolean;
  remote_id: string;
  revision: string;
  files: RecipeFile[];
  total_size_bytes: number | null;
  hardware: {
    tier: "cpu" | "midrange-gpu" | "high-end-gpu";
    minimum_ram_gb: number;
    recommended_ram_gb: number;
    minimum_vram_gb: number | null;
    recommended_vram_gb: number | null;
    guidance: string;
  };
  default_settings: Record<string, unknown>;
  workflow_path: string | null;
  node_policy: string | null;
  notes: string[];
}

export interface WorkflowRevision {
  id: string;
  workflow_id: string;
  version: number;
  engine: string;
  engine_version: string | null;
  ui_graph_json: Record<string, unknown>;
  api_graph_json: Record<string, unknown>;
  input_schema_json: Record<string, unknown>;
  dependencies_json: Record<string, unknown>;
  trusted: boolean;
  created_at: string;
}

export interface Workflow {
  id: string;
  name: string;
  operation: string;
  description: string;
  current_revision_id: string | null;
  revisions: WorkflowRevision[];
}

export interface WorkflowEditorSession {
  id: string;
  protocol_version: number;
  workflow_id: string;
  base_revision_id: string;
  base_graph_sha256: string;
  base_prompt_sha256: string;
  created_at: string;
  expires_at: string;
  ui_graph: Record<string, unknown>;
  nonce: string;
}

export interface WorkflowEditorGraphDelta {
  node_count_delta: number;
  link_count_delta: number;
  added_node_types: string[];
  removed_node_types: string[];
  added_asset_filenames: string[];
  removed_asset_filenames: string[];
}

export interface WorkflowEditorReturn {
  validated_return_id: string;
  session_id: string;
  workflow_id: string;
  base_revision_id: string;
  current_revision_id: string;
  base_graph_sha256: string;
  returned_graph_sha256: string;
  base_prompt_sha256: string;
  returned_prompt_sha256: string;
  changed: boolean;
  forked: boolean;
  delta: WorkflowEditorGraphDelta;
  expires_at: string;
}

export interface WorkflowEditorDraft {
  workflow_id: string;
  base_revision_id: string;
  draft_revision_id: string;
  current_revision_id: string | null;
  version: number;
  created: boolean;
  forked: boolean;
  trusted: false;
  review_required: true;
}

export interface CustomNodeInstall {
  id: string;
  name: string;
  source_url: string;
  revision: string;
  previous_revision: string | null;
  tree_hash: string;
  trusted: boolean;
  active: boolean;
  security_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface WorkflowBundle {
  format: "lm-atelier-workflow";
  version: 1;
  name: string;
  operation: string;
  description: string;
  engine: string;
  engine_version: string | null;
  ui_graph: Record<string, unknown>;
  api_graph: Record<string, unknown>;
  input_schema: Record<string, unknown>;
  dependencies: Record<string, unknown>;
  trusted: boolean;
  source_revision: number | null;
}

export interface SystemInfo {
  platform: string;
  platform_release: string;
  distribution: string;
  distribution_version: string;
  architecture: string;
  python_version: string;
  cpu_model: string;
  cpu_count: number;
  memory_total_bytes: number;
  memory_available_bytes: number;
  disk_total_bytes: number;
  disk_free_bytes: number;
  ffmpeg_available: boolean;
  support: PlatformAssessment;
  devices: Array<{
    id: string;
    name: string;
    kind: string;
    total_memory_bytes: number | null;
    available_memory_bytes: number | null;
    backend: string | null;
    details: Record<string, unknown>;
  }>;
}

export interface ApplicationInfo {
  version: string;
  data_directory: string;
  log_directory: string;
  max_media_outputs_per_plan: number;
  // The installation-wide gate. False means no chat can open its own, and
  // the UI says so rather than offering a switch that does nothing.
  web_access_enabled: boolean;
}

export interface WebSettings {
  allow_url_fetch: boolean;
}

export type CredentialProvider = "huggingface" | "civitai";

export interface CredentialStatus {
  provider: CredentialProvider;
  configured: boolean;
  source: "none" | "environment" | "credential_vault";
  vault_available: boolean;
}

export interface PlatformAssessment {
  platform_status: "target" | "experimental" | "unsupported";
  platform_label: string;
  accelerator_status: "primary" | "experimental" | "cpu-only";
  accelerator_label: string;
  certification_status: "hardware-pending" | "experimental" | "unsupported";
  chat_ready: boolean;
  reference_media_ready: boolean;
  vram_tier_gb: number | null;
  messages: string[];
}

export interface PlatformMatrixEntry {
  id: string;
  name: string;
  status: "target" | "experimental";
  operating_systems: string[];
  architectures: string[];
  accelerator: string;
  workloads: string[];
  vram_tiers_gb: number[];
  evidence: string;
  notes: string[];
}

export interface AppEvent {
  sequence: number;
  type: string;
  entity_id: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

/** The router's answer for an unsent draft, so the composer need not guess. */
export interface DraftClassification {
  references_prior_visual: boolean;
}

/** Analyzer report for a raw ComfyUI package (field names frozen with the
 * backend DTO; paired in the TypeScript contract gate). */
export interface WorkflowPackageRequirement {
  package_id: string;
  versions: string[];
  node_types: string[];
  locally_resolved: boolean;
}

/** A source link the workflow's author wrote down, validated server-side.
 *
 * A suggestion of what to look up - never a download instruction. The browser
 * cannot supply one of these, and installing still goes through the ordinary
 * immutable-plan path.
 */
export interface WorkflowSourceCandidate {
  provider: string;
  remote_id: string;
  revision: string | null;
  filename: string | null;
  url: string;
}

export interface WorkflowAssetReference {
  filename: string;
  suffix: string;
  policy: "supported" | "blocked" | "unsupported";
  kind: "checkpoint" | "configuration" | "embedding" | "lora" | "upscaler" | "vae";
  source_url: string | null;
  present_locally: boolean;
  /** Present only when the author's text names this exact file. */
  source_candidates: WorkflowSourceCandidate[];
}

export interface WorkflowPackageIssue {
  code: string;
  count: number;
  node_types: string[];
  severity: "blocking" | "advisory";
}

export interface WorkflowMissingNode {
  node_type: string;
  count: number;
  package_id: string | null;
}

export interface WorkflowPackageAnalysis {
  format_version: string;
  frontend_version: string | null;
  node_count: number;
  link_count: number;
  subgraph_count: number;
  operation_guess: "image" | "unknown" | "video";
  truncated: boolean;
  required_node_types: string[];
  frontend_node_types: string[];
  missing_node_types: string[];
  missing_nodes: WorkflowMissingNode[];
  custom_packages: WorkflowPackageRequirement[];
  asset_references: WorkflowAssetReference[];
  issues: WorkflowPackageIssue[];
  /** The one trust/activation gate the browser obeys; analyzer-computed. */
  ready: boolean;
  runtime_nodes_available: boolean;
  dependencies_resolved: boolean;
  /** False when the media runtime could not list nodes - "missing" is then
   * "unknown" and must be presented that way. */
  node_inventory_available: boolean;
  /** Links the author recorded that name no particular file, for the user to
   * assign. Most authors write display names, so this is the common case. */
  source_candidates: WorkflowSourceCandidate[];
}

/** A one-click edit: a named instruction scaffold over an edit workflow. */
export interface EditTemplate {
  id: string;
  name: string;
  description: string;
  instruction: string;
  operation: string;
  settings_json: Record<string, unknown>;
  /** What produced the result this was saved from. Null on anything saved
   * before recipes: nobody recorded it, and today's binding is not it. */
  workflow_revision_id: string | null;
  model_profile_id: string | null;
  mask_mode: string;
  trigger_words_json: string[];
  content_rating: "general" | "mature" | "unknown";
  builtin: boolean;
  enabled: boolean;
}

/** The identities a completed preparation is bound to; trusting and
 * activating what they describe are separate explicit steps. */
export interface WorkflowPackagePreparation {
  install_id: string;
  installed_path: string;
  wheel_environment_path: string;
  archive_sha256: string;
  manifest_sha256: string;
  wheel_closure_sha256: string;
  wheel_environment_sha256: string;
  reused_wheel_environment: boolean;
}

/** One prepared package and the two explicit decisions it is waiting for. */
/** One installed asset's staleness verdict; "unknown" means the provider
 * could not answer, never a guess. */
export interface ModelUpdate {
  install_id: string;
  name: string;
  kind: string;
  model_id: string;
  installed_version_id: string;
  installed_version_name: string | null;
  state: "update_available" | "current" | "unknown";
  update_version_id: string | null;
  update_version_name: string | null;
  update_published_at: string | null;
  update_base_model: string | null;
  update_changelog: string | null;
}

/** What staging found, so trusting a package can be an informed act. */
export interface RegistryInstallReview {
  file_count: number;
  expanded_bytes: number;
  python_file_count: number;
  install_scripts: string[];
  startup_hooks: string[];
  native_files: string[];
  dependency_manifests: string[];
  top_level_entries: string[];
  registry_warnings: string[];
}

export interface RegistryInstall {
  id: string;
  package_id: string;
  package_version: string;
  node_types: string[];
  archive_sha256: string;
  manifest_sha256: string;
  wheel_closure_sha256: string | null;
  wheel_environment_sha256: string | null;
  disk_status: "ready" | "node_files_missing" | "wheel_environment_missing" | "files_missing";
  node_files_present: boolean;
  wheel_environment_present: boolean;
  trusted: boolean;
  active: boolean;
  reviewed_at: string | null;
  activated_at: string | null;
  // Absent for a package prepared before this record existed. Absent means
  // "not looked at", which is not the same as "nothing found".
  review: RegistryInstallReview | null;
}

export type WorkflowSelectorCapability = "chat" | "vision" | "image" | "video";

/** Whether a variant can actually run right now, and why not when it cannot. */
export type WorkflowVariantReadiness =
  | "ready"
  | "setup_required"
  // The revision runs here, but nobody has vouched for it yet. Absent
  // from this union it fell through to "cannot run on this machine",
  // which is untrue and points at the wrong remedy.
  | "review_required"
  | "unavailable";

export type WorkflowSetupResolution =
  | "reviewed_download_available"
  | "attention_required";

export type WorkflowSelectionMode =
  | "default"
  | "inherit"
  | "automatic"
  | "family"
  | "revision"
  | "legacy";

export interface WorkflowFamilyVariant {
  id: string;
  variant_key: string;
  name: string;
  operation: string;
  current_revision_id: string | null;
  current_revision_version: number | null;
  engine: string | null;
  capabilities: string[];
  trusted: boolean;
  readiness: WorkflowVariantReadiness;
  readiness_reason: string | null;
  // Optional for rolling compatibility with a backend that predates the
  // additive setup-resolution projection.
  setup_resolution?: WorkflowSetupResolution | null;
  install_offer_id?: string | null;
}

export interface WorkflowFamilyPreference {
  selector_capability: WorkflowSelectorCapability;
  enabled: boolean;
  is_default: boolean;
  sort_order: number;
}

/** A family and the operation variants it resolves to.
 *
 * `compatibility` marks a family generated from a legacy profile rather than
 * authored as one, which is worth saying out loud rather than hiding: those
 * resolve to their original profile and behave exactly as they did.
 */
export interface WorkflowFamily {
  id: string;
  name: string;
  description: string;
  use_case: string;
  tags: string[];
  enabled: boolean;
  archived: boolean;
  compatibility: boolean;
  variants: WorkflowFamilyVariant[];
  preferences: WorkflowFamilyPreference[];
  created_at: string;
  updated_at: string;
}

export interface WorkflowSelection {
  selector_capability: WorkflowSelectorCapability;
  mode: WorkflowSelectionMode;
  workflow_family_id: string | null;
  workflow_revision_id: string | null;
  legacy_profile_id: string | null;
}

export type ChatWorkflowSelectionInput =
  | { mode: "default" }
  | { mode: "automatic" }
  | { mode: "family"; workflow_family_id: string };

export type ProjectWorkflowSelectionInput =
  | { mode: "inherit" }
  | { mode: "automatic" }
  | { mode: "family"; workflow_family_id: string }
  | { mode: "revision"; workflow_revision_id: string };

export type WorkflowDependencyResourceKind =
  | "model_profile"
  | "model_install"
  | "model_asset"
  | "custom_node"
  | "registry_package"
  | "runtime";

export interface WorkflowFamilyUpdate {
  name?: string;
  description?: string;
  use_case?: string;
  tags?: string[];
  enabled?: boolean;
  archived?: boolean;
}

export interface WorkflowFamilyPreferenceUpdate {
  enabled: boolean;
  is_default: boolean;
  sort_order: number;
}

/** One thing a family depends on, and whether anything else depends on it too. */
export interface WorkflowDependencyImpact {
  resource_kind: string;
  resource_id: string;
  resource_name: string;
  binding_count: number;
  revision_count: number;
  current_revision: boolean;
  shared: boolean;
  other_workflow_count: number;
  other_family_ids: string[];
}

/** What archiving a family would touch.
 *
 * Removal is archival by design: immutable revisions, exact project pins,
 * queued steps, run history, and shared bytes all survive it. These counts
 * exist so the question can say that, rather than implying things vanish.
 */
export interface WorkflowFamilyRemovalImpact {
  family_id: string;
  removal_strategy: "archive";
  archive_blocked: boolean;
  revision_count: number;
  current_revision_count: number;
  chat_selection_count: number;
  project_selection_count: number;
  project_revision_pin_count: number;
  active_run_count: number;
  queued_step_count: number;
  historical_run_count: number;
  active_activation_count: number;
  default_for: WorkflowSelectorCapability[];
  dependencies: WorkflowDependencyImpact[];
}

export interface WorkflowResourceConsumer {
  workflow_id: string;
  workflow_name: string;
  workflow_family_id: string | null;
  workflow_family_name: string | null;
  revision_ids: string[];
  binding_count: number;
  current_revision: boolean;
}

export interface WorkflowResourceConsumers {
  resource_kind: WorkflowDependencyResourceKind;
  resource_id: string;
  resource_name: string;
  consumers: WorkflowResourceConsumer[];
}

export interface StudioToolCapability {
  kind: string;
  workflow_class: string;
  available: boolean;
  reason: string | null;
}

export interface StudioCapabilityReport {
  tools: StudioToolCapability[];
}

export type ReferenceKind =
  | "person"
  | "character"
  | "object"
  | "product"
  | "place"
  | "style"
  | "wardrobe"
  | "pose"
  | "composition"
  | "other";

export interface ReferenceSubject {
  id: string;
  name: string;
  mention_slug: string;
  kind: string;
  description: string | null;
  aliases_json: string[];
  tags_json: string[];
  cover_artifact_id: string | null;
  favorite: boolean;
  archived: boolean;
}

export interface ReferenceSubjectPage {
  items: ReferenceSubject[];
  total: number;
  limit: number;
  offset: number;
}

export interface ReferenceAsset {
  id: string;
  reference_subject_id: string;
  artifact_id: string;
  caption: string | null;
  purpose: string;
  view_label: string | null;
  sort_order: number;
  validation_state: string;
  validation_reasons_json: string[];
  width: number | null;
  height: number | null;
  review_version: number;
}

export interface ReferenceAssetReviewed {
  asset: ReferenceAsset;
  review: {
    id: string;
    result_version: number;
    decision: "usable" | "weak" | "rejected";
    decision_sha256: string;
  };
  idempotent: boolean;
}

/** An image already held that closely resembles one just added. */
export interface ReferenceSimilarAsset {
  reference_asset_id: string;
  artifact_id: string;
  mean_absolute_difference: number;
}

export interface ReferenceAssetAttached {
  asset: ReferenceAsset;
  /** Advice, not a refusal - an empty list can also mean the scan could not run. */
  similar: ReferenceSimilarAsset[];
}

/** What deleting a reference would destroy, answered before anything is. */
export interface ReferenceDeletionImpact {
  reference_subject_id: string;
  name: string;
  asset_count: number;
  exclusive_artifact_ids: string[];
}
