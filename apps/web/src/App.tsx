import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Download,
  Film,
  Folder,
  GitBranch,
  HardDrive,
  Image as ImageIcon,
  Library,
  LoaderCircle,
  Menu,
  MessageSquare,
  MoreHorizontal,
  Pin,
  Pencil,
  Plus,
  Quote,
  RotateCcw,
  Search,
  Send,
  SlidersHorizontal,
  Sparkles,
  Star,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Upload,
  Wand2,
  Workflow as WorkflowIcon,
  X,
} from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ActiveChatWorkflowSelector } from "./ActiveChatWorkflowSelector";
import { CopyTextButton } from "./CopyTextButton";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import { api } from "./api";
import { formatBytes } from "./format";
import {
} from "./imageEditStrength";
import { GlobalNotices } from "./GlobalNotices";
import {
  artifactSource,
  mediaOriginForPart,
  mediaOriginLabel,
  editLineageForResult,
  editSourceUrlForResult,
  messagePartsForTranscript,
  priorVisibleMediaByMessage,
  type EditLineageStep,
  type MediaOrigin,
} from "./messageMedia";
import { useLiveEvents } from "./useLiveEvents";
import { ErrorCallout } from "./ErrorCallout";
import { FirstFailure } from "./FirstFailure";
import type { VisualTarget } from "./libraryEditTargets";
import { EmptyState } from "./EmptyState";
import { AtelierMark } from "./AtelierMark";
import { EditingStudio } from "./EditingStudio";
import { MessageTimestamp } from "./MessageTimestamp";
import { PendingResponseStatus } from "./PendingResponseStatus";
import { MarkdownText } from "./MarkdownText";
import { MentionText } from "./MentionText";
import { MessageField } from "./MessageField";
import { OutputCountControl } from "./OutputCountControl";
import { mediaOutputCountForTurn, useMediaOutputCount } from "./mediaOutputCount";
import type { TurnReference } from "./mentionDraft";
import { useComposerMentions } from "./useComposerMentions";
import { useConfirm } from "./useConfirm";
import { focusMainContent, roleForMode } from "./viewHelpers";
import { ArtifactPart } from "./ArtifactPart";
import { ImageStudioIcon } from "./ImageStudioIcon";
import { FirstRunSetup } from "./SetupWizard";
import { ChatManager } from "./ChatManager";
import { type View } from "./rooms";
import { useWorkspaceChrome, type SidebarLayout } from "./sidebarLayout";
import { SidebarResizer } from "./SidebarResizer";
import { SidebarFooter } from "./SidebarFooter";
import { SetupSurface } from "./SetupSurface";
import { ThemeToggle } from "./ThemeToggle";
import { WorkflowConsumers } from "./WorkflowConsumers";
import { operationForTurn, revisionForTurn, schemaForRevision } from "./turnWorkflow";
import type { WorkflowFamily, WorkflowSelection } from "./types";
import { PromptDialog } from "./ConfirmDialog";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import { ProjectManager } from "./ProjectManager";
import { SettingsView } from "./SettingsView";
import { MediaLibraryView } from "./MediaLibraryView";
import { ReferencesLibrary } from "./ReferencesLibrary";
import { MediaOutputPlan } from "./MediaOutputPlan";
import { ModelCard } from "./ModelCard";
import { WorkflowsView } from "./WorkflowsView";
import { VersionChooser } from "./VersionChooser";
import { StudioView } from "./StudioView";
import { RecipeCard } from "./RecipeCard";
import { ModelUpdatesPanel } from "./ModelUpdatesPanel";
import { useProjectMutations } from "./useProjectMutations";
import { AttachControls } from "./AttachControls";
import { JobsPanel } from "./JobsPanel";
import { editVisionNote, workshopTranscript } from "./promptWorkshop";
import { useComposerUploads } from "./useComposerUploads";
import type { ComposerAttachment } from "./useComposerUploads";
import { useDraftClassification } from "./useDraftClassification";
import { useFirstRunSetup } from "./useFirstRunSetup";
import { useGenerationModeSelection } from "./useGenerationModeSelection";
import { useMessageActions } from "./useMessageActions";
import {
  normalizeSettingsForFields,
  promptPreviewSettings,
  resolveCapabilitySettings,
  resolveWorkflowSettings,
} from "./settings";
import type {
  CatalogModel,
  CatalogPreflight,
  Chat,
  ChatDetail,
  EngineCapabilities,
  EngineRole,
  GenerationPreset,
  Message,
  MessageReference,
  MessagePart,
  ModelAssetInstall,
  ModelInstall,
  ModelProfile,
  Project,
  RoutingMode,
  SetupReadinessReport,
  TurnAccepted,
  Workflow,
  WorkPlan,
} from "./types";

type PendingTurn = { id: string; text: string; mode: RoutingMode };
type SendTurnVariables = PendingTurn & {
  chatId: string;
  artifacts: string[];
  settings: Record<string, unknown>;
  /** Subject ids chosen from the mention picker, never parsed from the text. */
  references: TurnReference[]; outputCount?: number;
  stopCurrent?: boolean;
};

const SETUP_DISMISSED_KEY = "lm-atelier-setup-dismissed";
const CURRENT_CHAT_KEY = "local-lm-chat";


/** The library's Edit action: attach the selection in the chat composer,
 * switch to image mode, and open the studio. */
function PartView({
  part,
  liveText,
  markdown = false,
  references,
  origin,
  onEditImage,
  onOpenStudio,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  compareSourceUrl,
  lineage,
}: {
  part: MessagePart;
  liveText?: string;
  markdown?: boolean;
  /** What the turn recorded referring to, so a text part can mark exactly
   *  those and nothing it found by reading the prose. */
  references?: MessageReference[];
  origin: MediaOrigin | null;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onOpenStudio?: (part: MessagePart) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  if (part.type === "text") {
    const text = liveText || part.text || "";
    return markdown ? <MarkdownText text={text} /> : <MentionText text={text} references={references} />;
  }
  if (part.type === "image" || part.type === "video" || part.type === "attachment") {
    return <ArtifactPart part={part} origin={origin} onEditImage={onEditImage} onOpenStudio={onOpenStudio} onAnimateImage={onAnimateImage} onReferenceMedia={onReferenceMedia} onToggleFavorite={onToggleFavorite} compareSourceUrl={compareSourceUrl} lineage={lineage} />;
  }
  if (part.type === "progress") {
    const progress = Number(part.metadata_json.progress ?? 0);
    const indeterminate = part.metadata_json.indeterminate === true;
    return (
      <div className="generation-progress" role="status" aria-live="polite">
        <Sparkles size={17} />
        <div>
          <span>{part.text || "Working"}</span>
          <div className="progress-track">
            <div
              className={indeterminate ? "indeterminate" : undefined}
              style={indeterminate ? undefined : { width: `${progress * 100}%` }}
            />
          </div>
        </div>
      </div>
    );
  }
  if (part.type === "error") return <div className="message-error" role="alert">{part.text}</div>;
  return <div className="message-error" role="alert">Unsupported message part: {String(part.type)}</div>;
}

function MessageBubble({
  message,
  liveText,
  hiddenInputArtifactIds,
  onRegenerate,
  onEdit,
  onSelectRevision,
  onCancelQueued,
  onEditImage,
  onOpenStudio,
  onAnimateImage,
  onReferenceMedia,
  onToggleFavorite,
  onQuote,
  onDeleteExchange,
  onForkThread,
  onFeedback,
  compareSourceUrl,
  lineage,
}: {
  message: Message;
  liveText?: string;
  hiddenInputArtifactIds?: ReadonlySet<string>;
  onRegenerate?: (messageId: string) => void;
  onEdit?: (messageId: string, text: string) => void;
  onSelectRevision?: (messageId: string, revisionId: string) => void;
  onCancelQueued?: () => void;
  onEditImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onOpenStudio?: (part: MessagePart) => void;
  onAnimateImage?: (part: MessagePart, origin: MediaOrigin) => void;
  onReferenceMedia?: (part: MessagePart, origin: MediaOrigin) => void;
  onToggleFavorite?: (part: MessagePart) => void;
  onQuote?: (text: string) => void;
  onDeleteExchange?: (messageId: string) => void;
  onForkThread?: (messageId: string) => void;
  onFeedback?: (messageId: string, revisionId: string | null, rating: "up" | "down" | null) => void;
  compareSourceUrl?: string | null;
  lineage?: EditLineageStep[];
}) {
  const visibleParts = messagePartsForTranscript(message, hiddenInputArtifactIds);
  const userText = visibleParts.filter((part) => part.type === "text").map((part) => part.text || "").join("\n");
  const copyableText = (liveText || userText).trim();
  const chatProgress = visibleParts.find(
    (part) => part.type === "progress" && part.metadata_json.activity === "chat",
  );
  const hasVisibleText = Boolean(copyableText);
  const hasMediaProgress = visibleParts.some(
    (part) => part.type === "progress" && part.metadata_json.activity !== "chat",
  );
  const showChatStartup = message.role === "assistant"
    && message.status === "pending"
    && !hasVisibleText
    && (Boolean(chatProgress) || !hasMediaProgress);
  const renderedParts = chatProgress
    ? visibleParts.filter((part) => part.id !== chatProgress.id)
    : visibleParts;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(userText);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const metadata = message.parts.find((part) => part.type === "generation_metadata")?.metadata_json;
  const context = metadata?.context as Record<string, unknown> | undefined;
  const provenance = metadata?.provenance as Record<string, unknown> | undefined;
  const routing = provenance?.routing as Record<string, unknown> | undefined;
  const operation = typeof routing?.operation === "string" ? routing.operation : undefined;
  const modelSelection = provenance?.model_selection as Record<string, unknown> | undefined;
  const autoProfileName = modelSelection?.mode === "auto"
    ? String(modelSelection.profile_name ?? "")
    : "";
  const autoMatchedTerms = modelSelection?.mode === "auto" && Array.isArray(modelSelection.matched_terms)
    ? modelSelection.matched_terms.filter((term): term is string => typeof term === "string").slice(0, 3)
    : [];
  const autoSelectionDetail = autoMatchedTerms.length
    ? ` · matched ${autoMatchedTerms.join(", ")}`
    : modelSelection?.mode === "auto" && modelSelection.fallback
      ? " · general fallback"
      : "";
  const auxiliaryAssets = provenance?.auxiliary_assets as Record<string, unknown> | undefined;
  const loraSelection = auxiliaryAssets?.selection as Record<string, unknown> | undefined;
  const automaticLoras = loraSelection?.mode === "automatic" && Array.isArray(loraSelection.selected)
    ? loraSelection.selected.filter(
        (item): item is Record<string, unknown> => Boolean(item) && typeof item === "object",
      )
    : [];
  const automaticLoraNames = automaticLoras
    .map((item) => String(item.name ?? ""))
    .filter(Boolean);
  const automaticLoraTerms = Array.from(new Set(automaticLoras.flatMap((item) => (
    Array.isArray(item.matched_terms)
      ? item.matched_terms.filter((term): term is string => typeof term === "string")
      : []
  )))).slice(0, 3);
  const appliedTriggerWords = Array.isArray(auxiliaryAssets?.trigger_words_applied)
    ? auxiliaryAssets.trigger_words_applied.filter(
        (word): word is string => typeof word === "string" && Boolean(word),
      )
    : [];
  const usage = context?.usage as Record<string, unknown> | undefined;
  const inputTokens = Number(usage?.prompt_tokens ?? context?.input_tokens ?? 0);
  const contextLimit = Number(context?.context_limit ?? 0);
  const omitted = Number(context?.messages_omitted ?? 0);
  const contextCompaction = context?.compaction as Record<string, unknown> | undefined;
  const compactedMessages = contextCompaction?.active
    ? Number(contextCompaction.source_message_count ?? omitted)
    : 0;
  const completedRevisions = (message.response_revisions ?? [])
    .filter((revision) => revision.status === "complete")
    .sort((left, right) => left.sequence - right.sequence);
  const activeRevisionIndex = completedRevisions.findIndex(
    (revision) => revision.id === message.active_response_revision_id,
  );
  const revisionIndex = activeRevisionIndex >= 0
    ? activeRevisionIndex
    : Math.max(0, completedRevisions.length - 1);
  const regenerationPending = (message.response_revisions ?? []).some(
    (revision) => revision.status === "pending",
  );
  const userMessageMeta = message.role === "user" && message.status === "complete" && !editing ? <div className="message-meta"><MessageTimestamp at={message.created_at} />{confirmingDelete ? <span className="delete-confirm"><span>Also deletes the answer and its media.</span><button className="danger" onClick={() => { setConfirmingDelete(false); onDeleteExchange?.(message.id); }}>Delete turn</button><button onClick={() => setConfirmingDelete(false)}>Keep</button></span> : <span className="message-actions">{onEdit && <button onClick={() => setEditing(true)} aria-label="Edit message" title="Edit"><Pencil size={14} /></button>}{copyableText && <CopyTextButton text={copyableText} label="Copy user message" buttonText="" />}{onDeleteExchange && <button aria-label="Delete this turn" title="Delete turn" onClick={() => setConfirmingDelete(true)}><Trash2 size={14} /></button>}</span>}</div> : null;
  const messageActionPartIndex = userMessageMeta ? renderedParts.map((part) => part.type).lastIndexOf("text") : -1;
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : <Bot size={19} />}</div>
      <div className="message-content">
        {editing ? <div className="message-edit"><textarea aria-label="Edit message" rows={4} value={draft} onChange={(event) => setDraft(event.target.value)} /><div><button onClick={() => { setDraft(userText); setEditing(false); }}>Cancel</button><button className="primary" disabled={!draft.trim()} onClick={() => { onEdit?.(message.id, draft.trim()); setEditing(false); }}>Send edited message</button></div></div> : renderedParts.map((part, index) => <Fragment key={part.id}><PartView part={part} liveText={liveText} markdown={message.role === "assistant"} references={message.references} origin={mediaOriginForPart(part, operation, message.role === "assistant" ? "generated" : null)} onEditImage={onEditImage} onOpenStudio={onOpenStudio} onAnimateImage={onAnimateImage} onReferenceMedia={onReferenceMedia} onToggleFavorite={onToggleFavorite} compareSourceUrl={message.role === "assistant" ? compareSourceUrl : undefined} lineage={message.role === "assistant" ? lineage : undefined} />{index === messageActionPartIndex && userMessageMeta}</Fragment>)}
        {liveText && !visibleParts.some((part) => part.type === "text") && (
          <MarkdownText text={liveText} />
        )}
        {showChatStartup && <PendingResponseStatus label={chatProgress?.text || "Starting chat"} startedAt={message.created_at} />}
        {messageActionPartIndex < 0 && userMessageMeta}
        {message.role === "assistant" && message.status === "cancelled" && !visibleParts.some((part) => part.type === "error") && (
          <div className="message-meta"><span>Generation cancelled</span></div>
        )}
        {message.role === "assistant" && message.status === "complete" && (
          <div className="message-meta">
            <MessageTimestamp at={message.created_at} />
            {autoProfileName && <span>Auto chose {autoProfileName}{autoSelectionDetail}</span>}
            {automaticLoraNames.length > 0 && (
              <span>
                LoRA Auto used {automaticLoraNames.join(", ")}
                {automaticLoraTerms.length > 0 ? ` — matched ${automaticLoraTerms.join(", ")}` : ""}
              </span>
            )}
            {appliedTriggerWords.length > 0 && <span>Added trigger words: {appliedTriggerWords.join(", ")}</span>}
            {contextLimit > 0 && (
              <span>
                Context {inputTokens.toLocaleString()} / {contextLimit.toLocaleString()} tokens
                {omitted > 0 && compactedMessages === 0
                  ? ` · ${omitted} earlier message${omitted === 1 ? "" : "s"} omitted`
                  : ""}
              </span>
            )}
            {compactedMessages > 0 && (
              <span>
                Compacted {compactedMessages} earlier message
                {compactedMessages === 1 ? "" : "s"} · full transcript preserved
              </span>
            )}
            {regenerationPending && <span>Regenerating…</span>}
            {/* Always visible, unlike the hover actions below: cycling between
                answers is navigation, and a control the user cannot see is a
                control they do not know exists. */}
            {completedRevisions.length > 1 && onSelectRevision && (
              <span className="response-revision-controls">
                <button
                  disabled={revisionIndex <= 0}
                  title="Previous answer"
                  onClick={() => onSelectRevision(
                    message.id,
                    completedRevisions[revisionIndex - 1]!.id,
                  )}
                  aria-label="Previous response revision"
                >
                  <ChevronLeft size={14} />
                </button>
                <span>{revisionIndex + 1} / {completedRevisions.length}</span>
                <button
                  disabled={revisionIndex >= completedRevisions.length - 1}
                  title="Next answer"
                  onClick={() => onSelectRevision(
                    message.id,
                    completedRevisions[revisionIndex + 1]!.id,
                  )}
                  aria-label="Next response revision"
                >
                  <ChevronRight size={14} />
                </button>
              </span>
            )}
            {/* Revealed on hover, and on keyboard focus - hover alone would
                put these actions out of reach without a mouse. */}
            <span className="message-actions">
              {onFeedback && (() => {
                const displayed = completedRevisions[revisionIndex] ?? null;
                const current = displayed ? displayed.feedback ?? null : message.feedback ?? null;
                const send = (rating: "up" | "down") => onFeedback(
                  message.id,
                  displayed?.id ?? null,
                  current === rating ? null : rating,
                );
                return (
                  <>
                    <button
                      aria-label="Good response"
                      title="Good response (stored locally)"
                      aria-pressed={current === "up"}
                      onClick={() => send("up")}
                    >
                      <ThumbsUp size={14} fill={current === "up" ? "currentColor" : "none"} />
                    </button>
                    <button
                      aria-label="Poor response"
                      title="Poor response (stored locally)"
                      aria-pressed={current === "down"}
                      onClick={() => send("down")}
                    >
                      <ThumbsDown size={14} fill={current === "down" ? "currentColor" : "none"} />
                    </button>
                  </>
                );
              })()}
              {copyableText && (
                <CopyTextButton
                  text={copyableText}
                  label="Copy assistant message"
                  buttonText=""
                />
              )}
              {copyableText && onQuote && (
                <button onClick={() => onQuote(copyableText)} aria-label="Quote response" title="Quote">
                  <Quote size={14} />
                </button>
              )}
              {onRegenerate && (
                <button onClick={() => onRegenerate(message.id)} aria-label="Regenerate response" title="Regenerate">
                  <RotateCcw size={14} />
                </button>
              )}
              {onForkThread && (
                <button onClick={() => onForkThread(message.id)} aria-label="Start a new thread here" title="New thread from here">
                  <GitBranch size={14} />
                </button>
              )}
            </span>
          </div>
        )}
        {message.role === "assistant" && message.status !== "complete" && copyableText && (
          <div className="message-meta"><CopyTextButton text={copyableText} label="Copy assistant message" /></div>
        )}
        {message.role === "assistant" && message.status === "pending" && onCancelQueued && (
          <div className="message-meta">
            <button onClick={onCancelQueued}><X size={13} /> Cancel queued item</button>
          </div>
        )}
      </div>
    </article>
  );
}


function SettingsDrawer({
  open,
  onClose,
  mode,
  engines,
  values,
  onValues,
  presets,
  presetId,
  onPreset,
  workflowSchema,
  inheritedValues,
  inheritedPresetId,
  profileValues,
  imageEdit,
  imageEditPrompt,
}: {
  open: boolean;
  onClose: () => void;
  mode: RoutingMode;
  engines: EngineCapabilities[];
  values: Record<string, unknown>;
  onValues: (values: Record<string, unknown>) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  workflowSchema?: Record<string, unknown>;
  inheritedValues?: Record<string, unknown>;
  inheritedPresetId?: string | null;
  profileValues?: Record<string, unknown>;
  imageEdit: boolean;
  imageEditPrompt: string;
}) {
  const role = roleForMode(mode);
  if (!open) return null;
  return (
    <AccessibleDialog
      title={`${role[0].toUpperCase() + role.slice(1)} settings`}
      eyebrow="Chat defaults"
      closeLabel="Close settings"
      onClose={onClose}
      className="settings-drawer"
      backdropClassName="settings-drawer-backdrop"
    >
      <GenerationSettingsPanel
        role={role}
        engines={engines}
        values={values}
        onValues={onValues}
        presets={presets}
        presetId={presetId}
        onPreset={onPreset}
        workflowSchema={workflowSchema}
        inheritedValues={inheritedValues}
        inheritedPresetId={inheritedPresetId}
        profileValues={profileValues}
        imageEdit={imageEdit}
        imageEditPrompt={imageEditPrompt}
        resetLabel="Reset chat overrides"
        onReset={() => onValues({})}
      />
    </AccessibleDialog>
  );
}

function promptHelperMessageText(message: Message): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("\n")
    .trim();
}

function PromptHelperDialog({
  sourceChat,
  initialDraft,
  engines,
  workflows,
  editSourceArtifactIds,
  onAccept,
  onClose,
}: {
  sourceChat: ChatDetail;
  initialDraft: string;
  engines: EngineCapabilities[];
  workflows: Workflow[];
  // When improving an image-edit instruction, the source image rides along so
  // a vision-capable helper grounds its rewrite in what the picture shows.
  editSourceArtifactIds?: string[];
  onAccept: (draft: string) => void;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const started = useRef(false);
  const [adoptedAssistantId, setAdoptedAssistantId] = useState<string | null>(null);
  const [helperId, setHelperId] = useState<string | null>(null);
  const [draft, setDraft] = useState(initialDraft);
  const [instruction, setInstruction] = useState("");
  const [working, setWorking] = useState(true);
  const [closing, setClosing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const helper = useQuery({
    queryKey: ["prompt-helper", helperId],
    queryFn: () => api.promptHelper(helperId!),
    enabled: Boolean(helperId),
    refetchInterval: 500,
  });

  const refresh = useCallback((id: string) => {
    void client.invalidateQueries({ queryKey: ["prompt-helper", id] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  }, [client]);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let active = true;
    void (async () => {
      try {
        const created = await api.createPromptHelper(sourceChat.id, initialDraft);
        if (!active) {
          await api.deletePromptHelper(created.id).catch(() => undefined);
          return;
        }
        setHelperId(created.id);
        await api.sendTurn(
          created.id,
          editSourceArtifactIds?.length
            ? "Improve the current draft as an editing instruction for the attached source image. "
              + "Ground it in what the image actually shows and preserve everything it does not ask to change. "
              + "Return the complete revised prompt only."
            : "Improve the current draft. Return the complete revised prompt only.",
          "text",
          editSourceArtifactIds ?? [],
          {},
        );
        if (active) refresh(created.id);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "Could not start prompt workshop");
      } finally {
        if (active) setWorking(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [editSourceArtifactIds, initialDraft, refresh, sourceChat.id]);

  const helperMessages = helper.data ? activeBranchMessages(helper.data) : [];
  const transcript = workshopTranscript(helperMessages);
  const pending = helperMessages.some((message) => message.status === "pending");
  const latestAssistant = [...helperMessages].reverse().find(
    (message) => message.role === "assistant" && message.status === "complete",
  );
  const latestAssistantText = latestAssistant ? promptHelperMessageText(latestAssistant) : "";
  const visionNote = editVisionNote(latestAssistant, Boolean(editSourceArtifactIds?.length));

  const send = async (mode: "text" | "image" | "video", text: string) => {
    if (!helperId || !draft.trim() || !text.trim()) return;
    setWorking(true);
    setError(null);
    try {
      await api.updatePromptHelper(helperId, draft.trim());
      const role = roleForMode(mode);
      const engine = engines.find((item) => item.roles.includes(role));
      const schema = workflowSchemaForTurn(workflows, mode, false);
      const fields = resolveWorkflowSettings(resolveCapabilitySettings(engine, role), schema);
      await api.sendTurn(
        helperId,
        text.trim(),
        mode,
        [],
        mode === "text" ? {} : promptPreviewSettings(fields),
      );
      setInstruction("");
      refresh(helperId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prompt workshop request failed");
    } finally {
      setWorking(false);
    }
  };

  const finish = async (accept: boolean) => {
    if (accept && !draft.trim()) return;
    setClosing(true);
    setError(null);
    try {
      if (helperId) await api.deletePromptHelper(helperId);
      if (accept) onAccept(draft.trim());
      else onClose();
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not close prompt workshop");
      setClosing(false);
    }
  };

  const unavailable = working || pending || closing || !helperId;
  return (
    <AccessibleDialog
      title="Prompt workshop"
      eyebrow="Draft with a local model"
      closeLabel="Cancel prompt workshop"
      onClose={() => void finish(false)}
      className="prompt-helper-dialog"
    >
      <div className="prompt-helper-body">
        <label className="prompt-helper-draft">
          <span>Draft prompt</span>
          <textarea
            aria-label="Draft prompt"
            value={draft}
            maxLength={20_000}
            rows={7}
            onChange={(event) => setDraft(event.target.value)}
          />
        </label>
        <div className="prompt-helper-conversation" aria-live="polite">
          {transcript.length === 0 && !error && (
            <div className="submission-progress"><LoaderCircle size={17} /><span>Starting workshop…</span></div>
          )}
          {transcript.map((message) => <MessageBubble key={message.id} message={message} />)}
        </div>
        {visionNote && <small className="prompt-helper-vision-note">{visionNote}</small>}
        {latestAssistant && latestAssistantText && adoptedAssistantId !== latestAssistant.id && (
          <button
            type="button"
            className="secondary compact-button prompt-helper-adopt"
            disabled={unavailable}
            onClick={() => {
              setAdoptedAssistantId(latestAssistant.id);
              setDraft(latestAssistantText);
              if (helperId) {
                void api.updatePromptHelper(helperId, latestAssistantText).catch(() => undefined);
              }
            }}
          >
            <Check size={14} /> Use latest response as draft
          </button>
        )}
        <div className="prompt-helper-compose">
          <textarea
            aria-label="Prompt workshop instruction"
            placeholder="Ask for a change…"
            value={instruction}
            rows={2}
            onChange={(event) => setInstruction(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send("text", instruction);
              }
            }}
          />
          <button
            type="button"
            className="primary compact-button"
            disabled={unavailable || !instruction.trim()}
            onClick={() => void send("text", instruction)}
          >
            <Send size={14} /> Ask helper
          </button>
        </div>
        <div className="prompt-helper-actions">
          <span>
            <button
              type="button"
              className="secondary compact-button"
              disabled={unavailable || !engines.some((engine) => engine.roles.includes("image"))}
              onClick={() => void send("image", draft)}
            >
              <ImageIcon size={14} /> Preview image
            </button>
            <button
              type="button"
              className="secondary compact-button"
              disabled={unavailable || !engines.some((engine) => engine.roles.includes("video"))}
              onClick={() => void send("video", draft)}
            >
              <Film size={14} /> Preview video
            </button>
          </span>
          <span>
            <button type="button" disabled={closing} onClick={() => void finish(false)}>Cancel</button>
            <button
              type="button"
              className="primary"
              disabled={unavailable || !draft.trim()}
              onClick={() => void finish(true)}
            >
              <Check size={15} /> Use prompt
            </button>
          </span>
        </div>
        {(error || helper.error) && (
          <div className="callout error" role="alert">{error || helper.error?.message}</div>
        )}
      </div>
    </AccessibleDialog>
  );
}
function Composer({
  chat,
  engines,
  profiles,
  stoppable,
  settings,
  onSettings,
  presets,
  presetId,
  onPreset,
  onMode,
  onSend,
  onStop,
  onStopAndSend,
  maxMediaOutputsPerPlan,
  workflows,
  project,
  visualTarget,
  quoteTarget,
}: {
  chat: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  stoppable: boolean;
  settings: Record<string, unknown>;
  onSettings: (settings: Record<string, unknown>) => void;
  presets: GenerationPreset[];
  presetId: string | null;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>, references: TurnReference[], outputCount?: number) => void;
  onStop: () => void;
  onStopAndSend: (
    text: string,
    mode: RoutingMode,
    artifacts: string[],
    settings: Record<string, unknown>,
    references: TurnReference[], outputCount?: number,
  ) => void;
  maxMediaOutputsPerPlan: number; workflows: Workflow[];
  project?: Project;
  visualTarget?: VisualTarget | null;
  quoteTarget?: { text: string; requestId: number } | null;
}) {
  const [text, setText] = useState("");
  const { outputCount, setOutputCount, resetOutputCount } = useMediaOutputCount();
  const mentions = useComposerMentions();
  const { mode, changeMode, currentMode } = useGenerationModeSelection(chat.routing_mode, onMode);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [promptHelperDraft, setPromptHelperDraft] = useState<string | null>(null);
  const [studioOpen, setStudioOpen] = useState(false);
  const [templateSettings, setTemplateSettings] = useState<{ name: string; settings: Record<string, unknown> } | null>(null);
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const { uploading, uploadError, setUploadError, uploadFiles, uploadPastedImages } = useComposerUploads(
    (attachment) => setAttachments((current) => [...current, attachment]),
  );
  const [dropActive, setDropActive] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const textInput = useRef<HTMLTextAreaElement>(null);
  const consumedVisualRequest = useRef<number | null>(null);
  useEffect(() => {
    if (!visualTarget || consumedVisualRequest.current === visualTarget.requestId) return;
    consumedVisualRequest.current = visualTarget.requestId;
    setAttachments((current) => {
      const additions = [visualTarget.attachment, ...(visualTarget.extraAttachments ?? [])]
        .filter((addition) => !current.some((item) => item.id === addition.id));
      return additions.length ? [...current, ...additions] : current;
    });
    if (visualTarget.mode) changeMode(visualTarget.mode);
    if (visualTarget.mode === "video") {
      window.setTimeout(() => {
        setText((current) => current.trim() ? current : "Animate this image");
      }, 0);
    }
    if (visualTarget.studio) {
      // After the attach renders, like the Animate prefill above.
      window.setTimeout(() => setStudioOpen(true), 0);
    }
    textInput.current?.focus();
  }, [visualTarget, changeMode]);
  const consumedQuoteRequest = useRef<number | null>(null);
  useEffect(() => {
    if (!quoteTarget || consumedQuoteRequest.current === quoteTarget.requestId) return;
    consumedQuoteRequest.current = quoteTarget.requestId;
    const quoted = quoteTarget.text
      .trim()
      .split("\n")
      .map((line) => `> ${line}`)
      .join("\n");
    setText((current) => (current.trim() ? `${quoted}\n\n${current}` : `${quoted}\n\n`));
    textInput.current?.focus();
  }, [quoteTarget]);
  const branchMessages = activeBranchMessages(chat);
  const priorVisual = branchMessages.some((message) =>
    message.parts.some((part) =>
      Boolean(part.artifact_id)
      && (part.type === "image" || part.type === "video")
      && part.metadata_json.preview !== true
    )
  );
  const priorImage = branchMessages.some((message) =>
    message.parts.some((part) =>
      Boolean(part.artifact_id)
      && part.type === "image"
      && part.metadata_json.preview !== true
    )
  );
  const usePriorVisual = useDraftClassification(chat.id, text, mode, priorVisual);
  const imageEdit = mode === "image" && (
    attachments.some((attachment) => attachment.kind === "image")
    || (priorImage && usePriorVisual)
  );
  const needsWorkflowSchema = mode === "image" || mode === "video";
  const families = useQuery({
    queryKey: ["workflow-families"],
    queryFn: () => api.workflowFamilies(),
    enabled: needsWorkflowSchema,
  });
  const selections = useQuery({
    queryKey: ["chat", chat?.id, "workflow-selections"],
    queryFn: () => api.chatWorkflowSelections(chat!.id),
    enabled: needsWorkflowSchema && Boolean(chat?.id),
  });
  const projectSelections = useQuery({ queryKey: ["project", project?.id, "workflow-selections"],
    queryFn: () => api.projectWorkflowSelections(project!.id),
    enabled: needsWorkflowSchema && Boolean(project?.id) });
  const imageProfile = profiles.find((profile) => profile.id === chat.active_image_profile_id)
    ?? profiles.find((profile) => profile.role === "image" && profile.is_default);
  const profileValues = {
    ...(imageProfile?.load_settings_json ?? {}),
    ...(imageProfile?.request_settings_json ?? {}),
  };
  const workflowSchema = workflowSchemaForTurn(
    workflows,
    mode,
    attachments.length > 0 || usePriorVisual,
    families.data ?? [],
    selections.data?.find((one) => one.selector_capability === mode),
    project ? projectSelections.data?.find((one) => one.selector_capability === mode) : null,
  );

  const submit = (stopCurrent = false) => {
    if (!text.trim()) return;
    const selectedMode = currentMode();
    const role = roleForMode(selectedMode);
    const engine = engines.find((item) => item.roles.includes(role));
    const fields = resolveWorkflowSettings(
      resolveCapabilitySettings(engine, role),
      workflowSchema,
    );
    const dispatch = stopCurrent ? onStopAndSend : onSend;
    const requestedOutputCount = mediaOutputCountForTurn(selectedMode, outputCount);
    dispatch(
      text.trim(),
      selectedMode,
      attachments.map((item) => item.id),
      selectedMode === "auto"
        ? {}
        : normalizeSettingsForFields(
            templateSettings ? { ...settings, ...templateSettings.settings } : settings,
            fields,
          ),
      mentions.forText(text),
      requestedOutputCount,
    );
    setText("");
    mentions.clear();
    setAttachments([]);
    setTemplateSettings(null);
    resetOutputCount();
  };

  return (
    <>
      <div
        className={`composer-wrap${dropActive ? " drop-active" : ""}`}
        onDragOver={(event) => {
          if (!Array.from(event.dataTransfer.types).includes("Files")) return;
          event.preventDefault();
          setDropActive(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget.contains(event.relatedTarget as Node)) return;
          setDropActive(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDropActive(false);
          const dropped = Array.from(event.dataTransfer.files);
          const files = dropped.filter(
            (file) => file.type.startsWith("image/") || file.type.startsWith("video/"),
          );
          setUploadError(
            files.length < dropped.length ? "Only images and videos can be attached." : "",
          );
          void uploadFiles(files);
        }}
      >
        {dropActive && <div className="drop-hint">Drop images or videos to attach</div>}
        {uploadError && <ErrorCallout message={uploadError} />}
        {attachments.length > 0 && (
          <div className="attachment-strip">
            {attachments.map((attachment) => {
              const source = attachment.artifact?.url || artifactSource(attachment.id)!;
              const name = attachment.artifact?.original_name || attachment.id;
              const label = mediaOriginLabel(attachment.origin, attachment.kind);
              return (
                <article className="attachment-card" key={attachment.id}>
                  <a
                    className="attachment-preview"
                    href={source}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`Preview ${name}`}
                  >
                    {attachment.kind === "image"
                      ? <img src={source} alt="" />
                      : <video src={source} muted preload="metadata" />}
                  </a>
                  <span className="attachment-summary">
                    <strong>{label}</strong>
                    <small title={name}>{name}</small>
                  </span>
                  <span className="attachment-actions">
                    {attachment.kind === "image" && (
                      <>
                        <button
                          className="attachment-edit"
                          aria-label="Edit attached image"
                          onClick={() => {
                            onMode("image");
                            textInput.current?.focus();
                          }}
                        >
                          Edit
                        </button>
                        <button
                          className="attachment-edit"
                          aria-label="Animate attached image"
                          onClick={() => {
                            onMode("video");
                            setText((current) => current.trim() ? current : "Animate this image");
                            textInput.current?.focus();
                          }}
                        >
                          Animate
                        </button>
                      </>
                    )}
                    <button
                      aria-label={`Remove ${label}: ${name}`}
                      onClick={() => setAttachments((items) => (
                        items.filter((item) => item.id !== attachment.id)
                      ))}
                    >
                      <X size={12} />
                    </button>
                  </span>
                </article>
              );
            })}
          </div>
        )}
        {templateSettings && (
          <div className="template-settings-chip">
            <span>{templateSettings.name} settings apply to this send</span>
            <button aria-label="Remove template settings" onClick={() => setTemplateSettings(null)}><X size={12} /></button>
          </div>
        )}
        <div className="composer">
          <MessageField field={textInput} value={text} onChange={setText} onSubmit={submit} onMention={mentions.add} onPasteFiles={(files) => { void uploadPastedImages(files); }} />
          <div className="composer-tools">
            <div className="left-tools">
              <AttachControls disabled={uploading} onPickFile={() => fileInput.current?.click()} onAttach={(attachment) => setAttachments((current) => [...current, attachment])} />
              <input ref={fileInput} hidden multiple type="file" accept="image/*,video/*" onChange={(event) => { setUploadError(""); void uploadFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
              <button
                className="icon-button"
                onClick={() => setPromptHelperDraft(text.trim())}
                disabled={!text.trim()}
                aria-label="Open prompt workshop"
                title="Improve this prompt"
              >
                <Sparkles size={18} />
              </button>
              <label className={`mode-select mode-${mode}`}>
                {mode === "auto" && <Sparkles size={15} />}
                {mode === "text" && <MessageSquare size={15} />}
                {mode === "image" && <ImageIcon size={15} />}
                {mode === "video" && <Film size={15} />}
                <select aria-label="Generation mode" value={mode} onChange={(event) => changeMode(event.target.value as RoutingMode)}>
                  <option value="auto">Auto</option><option value="text">Text</option><option value="image">Image</option><option value="video">Video</option>
                </select>
                <ChevronDown size={13} />
              </label>
              <OutputCountControl mode={mode} maximum={maxMediaOutputsPerPlan} value={outputCount} onChange={setOutputCount} />
              <div className="composer-workflow-selector">
                <WorkflowIcon aria-hidden="true" size={15} />
                <ActiveChatWorkflowSelector chatId={chat.id} routingMode={mode} />
              </div>
              {imageEdit && <button className="icon-button" onClick={() => setStudioOpen(true)} aria-label="Open editing studio" title="One-click edits"><Wand2 size={18} /></button>}
              <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Turn settings"><SlidersHorizontal size={18} /></button>
            </div>
            <span className="composer-submit-actions">
              {stoppable && (
                <button
                  className="send-button stop"
                  onClick={onStop}
                  aria-label="Stop current response"
                  title="Stop current response"
                >
                  <CircleStop size={18} />
                </button>
              )}
              {stoppable && text.trim() && (
                <button
                  className="secondary stop-and-send"
                  onClick={() => submit(true)}
                  aria-label="Stop current response and send"
                >
                  Stop and send
                </button>
              )}
              <button
                className="send-button"
                disabled={!text.trim()}
                onClick={() => submit()}
                aria-label="Send"
              >
                <Send size={18} />
              </button>
            </span>
          </div>
        </div>
      </div>
      {studioOpen && <EditingStudio currentInstruction={text} onClose={() => setStudioOpen(false)} onPick={(instruction, template) => { setText(instruction); setTemplateSettings(Object.keys(template.settings_json).length ? { name: template.name, settings: template.settings_json } : null); setStudioOpen(false); window.setTimeout(() => textInput.current?.focus(), 0); }} imageCount={attachments.filter((item) => item.kind === "image").length} onApplyToEach={(instruction, template) => {
        const role = roleForMode("image");
        const engine = engines.find((item) => item.roles.includes(role));
        const fields = resolveWorkflowSettings(resolveCapabilitySettings(engine, role), workflowSchema);
        const merged = normalizeSettingsForFields({ ...settings, ...template.settings_json }, fields);
        // One ordinary edit turn per image: each queues, verifies, and retries
        // alone; the pending-work bound errs clearly rather than truncating.
        // No references: these are edits of the attached images themselves,
        // not a mention-driven turn, and the instruction was not composed in
        // the field that tracks mentions.
        for (const item of attachments.filter((entry) => entry.kind === "image")) onSend(instruction, "image", [item.id], merged, []);
        setAttachments([]); setText(""); setTemplateSettings(null); setStudioOpen(false);
      }} />}
      {promptHelperDraft !== null && (
        <PromptHelperDialog
          sourceChat={chat}
          initialDraft={promptHelperDraft}
          engines={engines}
          workflows={workflows}
          // Only explicit attachments ground the workshop: the helper chat has
          // no lineage, so a prior-image reference has nothing to resolve to.
          editSourceArtifactIds={imageEdit
            ? attachments.filter((item) => item.kind === "image").map((item) => item.id)
            : undefined}
          onAccept={(nextDraft) => {
            setText(nextDraft);
            setPromptHelperDraft(null);
            window.setTimeout(() => textInput.current?.focus(), 0);
          }}
          onClose={() => setPromptHelperDraft(null)}
        />
      )}
      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        mode={mode}
        engines={engines}
        values={settings}
        onValues={onSettings}
        presets={presets}
        presetId={presetId}
        onPreset={onPreset}
        workflowSchema={workflowSchema}
        inheritedValues={project?.generation_settings_json?.[roleForMode(mode)]}
        inheritedPresetId={project?.generation_preset_ids_json?.[roleForMode(mode)]}
        profileValues={profileValues}
        imageEdit={imageEdit}
        imageEditPrompt={text}
      />
    </>
  );
}

function activeBranchMessages(chat: ChatDetail): Message[] {
  const visibleMessages = chat.messages.filter(
    (message) => message.transcript_visible !== false,
  );
  if (!chat.active_head_message_id) return visibleMessages;
  const byId = new Map(visibleMessages.map((message) => [message.id, message]));
  const lineage: Message[] = [];
  const visited = new Set<string>();
  let current = byId.get(chat.active_head_message_id);
  while (current && !visited.has(current.id)) {
    visited.add(current.id);
    lineage.unshift(current);
    current = current.parent_id ? byId.get(current.parent_id) : undefined;
  }
  return lineage.length > 0 ? lineage : visibleMessages;
}

function workflowSchemaForTurn(
  workflows: Workflow[],
  mode: RoutingMode,
  hasAttachments: boolean,
  families: WorkflowFamily[] = [],
  chatSelection: WorkflowSelection | null | undefined = null,
  projectSelection: WorkflowSelection | null | undefined = null,
): Record<string, unknown> | undefined {
  if (mode !== "image" && mode !== "video") return undefined;
  const operation = operationForTurn(mode, hasAttachments);
  const revisionId = revisionForTurn(
    families,
    mode,
    chatSelection,
    projectSelection,
    operation,
  );
  return schemaForRevision(workflows, revisionId, operation);
}

function ChatView({
  onOpenStudio,
  chat,
  engines,
  profiles,
  workflows,
  project,
  liveText,
  pendingTurns,
  workPlans,
  settings,
  presets,
  presetId,
  onSettings,
  onPreset,
  onMode,
  onSend,
  onRegenerate,
  onSelectRevision,
  onEdit,
  onStop,
  onStopAndSend,
  maxMediaOutputsPerPlan,
  onCancelPlan,
  onCancelStep,
  onRetryStep,
  onDeleteExchange,
  onForkThread,
  libraryEdit,
}: {
  onOpenStudio: (artifactId: string) => void;
  chat?: ChatDetail;
  engines: EngineCapabilities[];
  profiles: ModelProfile[];
  workflows: Workflow[];
  project?: Project;
  liveText: Record<string, string>;
  pendingTurns: PendingTurn[];
  workPlans: WorkPlan[];
  settings: Record<string, unknown>;
  presets: GenerationPreset[];
  presetId: string | null;
  onSettings: (settings: Record<string, unknown>) => void;
  onPreset: (presetId: string | null) => void;
  onMode: (mode: RoutingMode) => void;
  onSend: (text: string, mode: RoutingMode, artifacts: string[], settings: Record<string, unknown>, references: TurnReference[], outputCount?: number) => void;
  onRegenerate: (messageId: string, settings: Record<string, unknown>) => void;
  onSelectRevision: (messageId: string, revisionId: string) => void;
  onEdit: (
    messageId: string,
    text: string,
    mode: RoutingMode,
    settings: Record<string, unknown>,
  ) => void;
  onStop: () => void;
  onStopAndSend: (
    text: string,
    mode: RoutingMode,
    artifacts: string[],
    settings: Record<string, unknown>,
    references: TurnReference[], outputCount?: number,
  ) => void;
  maxMediaOutputsPerPlan: number; onCancelPlan: (planId: string) => void;
  onCancelStep: (stepId: string) => void;
  onRetryStep: (stepId: string) => void;
  onDeleteExchange: (messageId: string) => void;
  onForkThread: (messageId: string) => void;
  libraryEdit?: VisualTarget | null;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const followMessages = useRef(true);
  const previousChatId = useRef<string | undefined>(undefined);
  const [visualTarget, setVisualTarget] = useState<VisualTarget | null>(null);
  const favoriteClient = useQueryClient();
  const feedback = useMutation({
    mutationFn: ({ messageId, revisionId, rating }: {
      messageId: string;
      revisionId: string | null;
      rating: "up" | "down" | null;
    }) => api.setResponseFeedback(messageId, rating, revisionId),
    onSuccess: () => void favoriteClient.invalidateQueries({ queryKey: ["chat"] }),
  });
  const toggleFavorite = useMutation({
    mutationFn: ({ artifactId, next }: { artifactId: string; next: boolean }) =>
      api.favoriteArtifact(artifactId, next),
    onSuccess: () => {
      void favoriteClient.invalidateQueries({ queryKey: ["chat"] });
      void favoriteClient.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const consumedLibraryEdit = useRef<number | null>(null);
  useEffect(() => {
    if (!libraryEdit || consumedLibraryEdit.current === libraryEdit.requestId) return;
    consumedLibraryEdit.current = libraryEdit.requestId;
    setVisualTarget(libraryEdit);
  }, [libraryEdit]);
  const [quoteTarget, setQuoteTarget] = useState<{ text: string; requestId: number } | null>(null);
  useEffect(() => {
    if (previousChatId.current !== chat?.id) {
      previousChatId.current = chat?.id;
      followMessages.current = true;
    }
    if (followMessages.current && typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chat?.id, chat?.messages, liveText, pendingTurns]);
  const trackMessageScroll = () => {
    const viewport = messagesRef.current;
    if (!viewport) return;
    followMessages.current = (
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
    ) <= 96;
  };
  if (!chat) return <EmptyState icon={<MessageSquare />} title="Start a local conversation" body="Create a chat and choose a model. Conversations stay on this machine." />;
  const messages = activeBranchMessages(chat);
  const priorVisibleMedia = priorVisibleMediaByMessage(messages);
  const stoppable = messages.some(
    (message) => message.status === "pending"
      || (message.response_revisions ?? []).some(
        (revision) => revision.status === "pending",
      ),
  );
  const busy = stoppable || pendingTurns.length > 0;
  const planByAssistantMessage = new Map(
    workPlans.flatMap((plan) => {
      const assistantMessageIds = Array.isArray(plan.summary_json.assistant_message_ids)
        ? plan.summary_json.assistant_message_ids.filter(
          (messageId): messageId is string => typeof messageId === "string",
        )
        : [];
      const legacyAssistantMessageId = plan.summary_json.assistant_message_id;
      if (typeof legacyAssistantMessageId === "string") {
        assistantMessageIds.push(legacyAssistantMessageId);
      }
      return [...new Set(assistantMessageIds)].map(
        (messageId) => [messageId, plan] as const,
      );
    }),
  );
  return (
    <div className="chat-view">
      <div className="chat-header">
        <div><small>{chat.project_id ? "Project chat" : "Unfiled chat"}</small><h1>{chat.title}</h1></div>
      </div>
      {/* Reported here because the global list belongs to a component the
          transcript cannot reach. */}
      <FirstFailure of={[feedback, toggleFavorite]} />
      <div className="messages" ref={messagesRef} onScroll={trackMessageScroll}>
        {messages.length === 0 && pendingTurns.length === 0 ? (
          <EmptyState icon={<Sparkles />} title="What should we make?" body="Ask anything or create an image or video. Auto mode picks the model." />
        ) : messages.map((message, messageIndex) => {
          const messagePlan = planByAssistantMessage.get(message.id);
          const compareSourceUrl = message.role === "assistant"
            ? editSourceUrlForResult(messages, messageIndex)
            : null;
          const lineage = compareSourceUrl ? editLineageForResult(messages, messageIndex) : undefined;
          const isPrimaryOutput = messagePlan?.summary_json.assistant_message_id === message.id;
          return (
            <Fragment key={message.id}>
              {messagePlan && messagePlan.steps.length > 1 && isPrimaryOutput && messagePlan.steps.some((step) => step.status !== "complete") && (
                <MediaOutputPlan
                  plan={messagePlan}
                  onCancelStep={onCancelStep}
                  onRetryStep={onRetryStep}
                />
              )}
              <MessageBubble
                message={message}
                liveText={liveText[message.id]}
                compareSourceUrl={compareSourceUrl}
                lineage={lineage}
                onFeedback={(messageId, revisionId, rating) =>
                  feedback.mutate({ messageId, revisionId, rating })}
                onToggleFavorite={(part) => part.artifact_id && toggleFavorite.mutate({
                  artifactId: part.artifact_id,
                  next: !part.artifact?.favorite,
                })}
                hiddenInputArtifactIds={priorVisibleMedia.get(message.id)}
                onRegenerate={busy ? undefined : (messageId) => onRegenerate(
                  messageId,
                  chat.routing_mode === "auto" ? {} : settings,
                )}
                onSelectRevision={busy ? undefined : onSelectRevision}
                onEdit={busy ? undefined : (messageId, text) => onEdit(
                  messageId,
                  text,
                  chat.routing_mode,
                  chat.routing_mode === "auto" ? {} : settings,
                )}
                onCancelQueued={
                  messagePlan && messagePlan.steps.length <= 1 && messagePlan.status === "queued"
                    ? () => onCancelPlan(messagePlan.id)
                    : undefined
                }
                onOpenStudio={busy ? undefined : (part) => onOpenStudio(part.artifact_id!)}
                onEditImage={busy ? undefined : (part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: "image",
                  requestId: Date.now(),
                })}
                onAnimateImage={busy ? undefined : (part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: "video",
                  requestId: Date.now(),
                })}
                onReferenceMedia={busy ? undefined : (part, origin) => setVisualTarget({
                  attachment: {
                    id: part.artifact_id!,
                    kind: part.type === "video" ? "video" : "image",
                    artifact: part.artifact,
                    origin,
                  },
                  mode: null,
                  requestId: Date.now(),
                })}
                onQuote={(text) => setQuoteTarget({ text, requestId: Date.now() })}
                onDeleteExchange={busy ? undefined : onDeleteExchange}
                onForkThread={busy ? undefined : onForkThread}
              />
            </Fragment>
          );
        })}
        {pendingTurns.map((pendingTurn) => (
          <Fragment key={pendingTurn.id}>
            <article className="message user optimistic">
              <div className="avatar">You</div>
              <div className="message-content"><div className="message-text">{pendingTurn.text}</div></div>
            </article>
            <article
              className="message assistant optimistic"
              aria-live="polite"
            >
              <div className="avatar"><Bot size={19} /></div>
              <div className="message-content">
                <div className="submission-progress">
                  <LoaderCircle size={17} />
                  <span>{pendingTurn.mode === "auto" ? "Choosing mode and model…" : "Starting…"}</span>
                </div>
              </div>
            </article>
          </Fragment>
        ))}
        <div ref={endRef} />
      </div>
      <Composer chat={chat} engines={engines} profiles={profiles} stoppable={stoppable} settings={settings} onSettings={onSettings} presets={presets} presetId={presetId} onPreset={onPreset} onMode={onMode} onSend={onSend} onStop={onStop} onStopAndSend={onStopAndSend} maxMediaOutputsPerPlan={maxMediaOutputsPerPlan} workflows={workflows} project={project} visualTarget={visualTarget} quoteTarget={quoteTarget} />
    </div>
  );
}

function InstalledModelRow({
  model,
  profile,
  creating,
  deleting,
  saving,
  defaulting,
  onCreate,
  onDelete,
  onSaveUseCase,
  onSetDefault,
}: {
  model: ModelInstall;
  profile?: ModelProfile;
  creating: boolean;
  deleting: boolean;
  saving: boolean;
  defaulting: boolean;
  onCreate: () => void;
  onDelete: () => void;
  onSaveUseCase: (value: string) => Promise<boolean>;
  onSetDefault: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(profile?.use_case ?? "");
  const startEditing = () => {
    setDraft(profile?.use_case ?? "");
    setEditing(true);
  };
  const save = async () => {
    if (await onSaveUseCase(draft.trim())) setEditing(false);
  };
  return (
    <div className={editing ? "editing" : ""}>
      <span className="badge">{model.role}</span>
      <span className="model-install-copy">
        <strong>{model.name}</strong>
        <small>{model.readiness === "ready" ? "Runtime verified" : model.readiness === "unsupported" ? "Unsupported" : "Not runtime verified"}</small>
        {model.role === "chat" && model.readiness === "ready" && profile?.input_modalities?.includes("text") && (
          <small>{profile.input_modalities.includes("image") ? "Vision capable" : "Text only"}</small>
        )}
        {profile?.use_case && <small>{profile.use_case}</small>}
      </span>
      <span className="model-install-size">{formatBytes(model.size_bytes)}</span>
      <span className="row-actions">
        {profile?.is_default
          ? <span className="badge tested">Default</span>
          : <button className="secondary compact-button" aria-label={`Set ${model.name} as default ${model.role} model`} disabled={creating || defaulting} onClick={onSetDefault}>{defaulting ? "Setting..." : "Set default"}</button>}
        {profile
          ? <button className="secondary compact-button" aria-label={`Edit use case for ${model.name}`} onClick={startEditing} disabled={editing || saving}>Edit use case</button>
          : <button className="secondary compact-button" aria-label={`Add ${model.name} to model selectors`} disabled={creating} onClick={onCreate}>Add to selectors</button>}
        <button className="secondary compact-button danger" aria-label={`Delete ${model.name}`} disabled={deleting} onClick={onDelete}>Delete</button>
      </span>
      {editing && profile && (
        <form className="model-use-case-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <textarea aria-label={`Best uses for ${model.name}`} rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Programming, illustration, cinematic video…" />
          <button type="button" className="secondary compact-button" disabled={saving} onClick={() => setEditing(false)}>Cancel</button>
          <button type="submit" className="primary compact-button" disabled={saving || draft.trim() === profile.use_case.trim()}>{saving ? "Saving…" : "Save"}</button>
        </form>
      )}
    </div>
  );
}

type ModelAssetUpdateValues = Partial<Pick<
  ModelAssetInstall,
  "active" | "use_case" | "auto_apply" | "default_model_strength" | "default_clip_strength"
>>;

function InstalledAssetRow({
  asset,
  saving,
  deleting,
  onUpdate,
  onDelete,
}: {
  asset: ModelAssetInstall;
  saving: boolean;
  deleting: boolean;
  onUpdate: (values: ModelAssetUpdateValues) => Promise<boolean>;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [useCase, setUseCase] = useState(asset.use_case);
  const [autoApply, setAutoApply] = useState(asset.auto_apply);
  const [modelStrength, setModelStrength] = useState(String(asset.default_model_strength));
  const [clipStrength, setClipStrength] = useState(String(asset.default_clip_strength));
  const beginEditing = () => {
    setUseCase(asset.use_case);
    setAutoApply(asset.auto_apply);
    setModelStrength(String(asset.default_model_strength));
    setClipStrength(String(asset.default_clip_strength));
    setEditing(true);
  };
  const parsedModelStrength = Number(modelStrength);
  const parsedClipStrength = Number(clipStrength);
  const strengthsValid = Number.isFinite(parsedModelStrength)
    && Math.abs(parsedModelStrength) <= 4
    && Number.isFinite(parsedClipStrength)
    && Math.abs(parsedClipStrength) <= 4;
  const unchanged = useCase.trim() === asset.use_case
    && autoApply === asset.auto_apply
    && parsedModelStrength === asset.default_model_strength
    && parsedClipStrength === asset.default_clip_strength;
  const save = async () => {
    if (!strengthsValid || (autoApply && !useCase.trim())) return;
    const saved = await onUpdate({
      use_case: useCase.trim(),
      auto_apply: autoApply,
      default_model_strength: parsedModelStrength,
      default_clip_strength: parsedClipStrength,
    });
    if (saved) setEditing(false);
  };
  return (
    <div className={editing ? "editing" : ""}>
      <span className="badge">{asset.kind.replace("_", " ")}</span>
      <span className="model-install-copy">
        <strong>{asset.name}</strong>
        <small>{asset.active ? "Ready" : "Disabled"}{asset.family ? ` · ${asset.family}` : ""}</small>
        {asset.kind === "lora" && asset.auto_apply && asset.use_case && (
          <small>Auto · {asset.use_case}</small>
        )}
      </span>
      <span className="model-install-size">{formatBytes(asset.size_bytes)}</span>
      <span className="row-actions">
        <button
          className="secondary compact-button"
          disabled={!asset.verified_at || saving}
          onClick={() => void onUpdate({ active: !asset.active })}
        >
          {asset.active ? "Disable" : "Enable"}
        </button>
        {asset.kind === "lora" && (
          <button className="secondary compact-button" disabled={editing || saving} onClick={beginEditing}>
            Edit Auto rules
          </button>
        )}
        <button className="secondary compact-button danger" disabled={deleting} onClick={onDelete}>Delete</button>
      </span>
      {editing && asset.kind === "lora" && (
        <form className="model-use-case-editor lora-auto-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <label>
            Use case
            <textarea aria-label={`Auto use case for ${asset.name}`} rows={2} value={useCase} onChange={(event) => setUseCase(event.target.value)} placeholder="Watercolor landscapes, product photography…" />
          </label>
          <label>
            Model strength
            <input aria-label={`Default model strength for ${asset.name}`} type="number" min="-4" max="4" step="0.05" value={modelStrength} onChange={(event) => setModelStrength(event.target.value)} />
          </label>
          <label>
            CLIP strength
            <input aria-label={`Default CLIP strength for ${asset.name}`} type="number" min="-4" max="4" step="0.05" value={clipStrength} onChange={(event) => setClipStrength(event.target.value)} />
          </label>
          <label className="lora-auto-toggle">
            <input aria-label={`Use ${asset.name} automatically`} type="checkbox" checked={autoApply} onChange={(event) => setAutoApply(event.target.checked)} />
            Use automatically
          </label>
          <span className="row-actions">
            <button type="button" className="secondary compact-button" disabled={saving} onClick={() => setEditing(false)}>Cancel</button>
            <button type="submit" className="primary compact-button" disabled={saving || unchanged || !strengthsValid || (autoApply && !useCase.trim())}>{saving ? "Saving…" : "Save"}</button>
          </span>
        </form>
      )}
    </div>
  );
}

interface PendingInstall {
  model: CatalogModel;
  preflight: CatalogPreflight;
  installRole: string;
  engine: string;
  auxiliaryKind: "lora" | null;
}

function ModelsView({ initialRole }: { initialRole: EngineRole }) {
  const [choosingVersions, setChoosingVersions] = useState<CatalogModel | null>(null);
  const [confirmDialog, confirm] = useConfirm();
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [catalogSource, setCatalogSource] = useState("huggingface");
  const [role, setRole] = useState<string>(initialRole);
  const [sort, setSort] = useState("trending");
  const [compatibility, setCompatibility] = useState("");
  const [quantization, setQuantization] = useState("");
  const [maxSizeGb, setMaxSizeGb] = useState("");
  const [updatedWithinDays, setUpdatedWithinDays] = useState("");
  const [installedChatCapability, setInstalledChatCapability] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [importName, setImportName] = useState("");
  const [importPath, setImportPath] = useState("");
  const [importRole, setImportRole] = useState("chat");
  const [importEngine, setImportEngine] = useState("llama.cpp");
  const catalogFilters = {
    compatibility,
    quantization,
    max_size_bytes: maxSizeGb ? String(Number(maxSizeGb) * 1024 ** 3) : "",
    updated_within_days: updatedWithinDays,
  };
  const catalog = useInfiniteQuery({
    queryKey: ["catalog", submitted, role, sort, catalogFilters, catalogSource],
    queryFn: ({ pageParam }) =>
      api.catalog(submitted, role, sort, pageParam, catalogFilters, catalogSource),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const rawCatalogItems = useMemo(() => catalog.data?.pages.flatMap((page) => page.items) ?? [], [catalog.data]);
  const catalogItems = rawCatalogItems;
  const catalogIsStale = catalog.data?.pages.some((page) => page.stale) ?? false;
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: api.recipes });
  const installed = useQuery({ queryKey: ["models"], queryFn: api.models });
  const modelAssets = useQuery({ queryKey: ["model-assets"], queryFn: () => api.modelAssets() });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3_000 });
  const storage = useQuery({ queryKey: ["model-storage"], queryFn: api.modelStorage });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const runtimes = useQuery({ queryKey: ["runtimes"], queryFn: api.runtimes });
  const machine = useQuery({ queryKey: ["system"], queryFn: api.system });
  const runtimeFor = (model: CatalogModel) => runtimes.data?.find(
    (runtime) => runtime.engine === model.required_runtime,
  );
  // Preflight and transfer are separate steps so the user sees what a download
  // will cost before it starts; the numbers were previously computed and dropped.
  const [pendingInstall, setPendingInstall] = useState<PendingInstall | null>(null);
  const download = useMutation({
    mutationFn: async ({ model, selectedRole }: { model: CatalogModel; selectedRole: string }) => {
      const auxiliaryKind = selectedRole === "lora" ? "lora" : null;
      const installRole = auxiliaryKind ? "image" : selectedRole;
      const engine = model.required_runtime ?? (installRole === "chat" ? "llama.cpp" : "comfyui");
      // A CivitAI card's remote id is its exact version; that is also the
      // revision it pins. Hugging Face keeps floating "main".
      const revision = model.provider === "civitai" ? model.remote_id : "main";
      const preflight = auxiliaryKind
        ? await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            revision,
            [],
            auxiliaryKind,
            null,
            model.provider,
          )
        : await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            revision,
            [],
            null,
            // Preflight the exact workflow this card represents; a repository
            // can ship several and ranking must not answer for the user.
            model.workflow_template_id ?? null,
            model.provider,
          );
      if (!preflight.can_install) {
        const blockers = preflight.checks
          .filter((check) => check.status === "block")
          .map((check) => check.detail);
        throw new Error(blockers.join(" ") || "This model cannot be installed safely.");
      }
      if (!preflight.install_plan || preflight.install_plan.compatibility !== "supported") {
        throw new Error(
          preflight.install_plan?.failure_reason
          || "LM Atelier cannot safely activate this model with the current runtime.",
        );
      }
      return { model, preflight, installRole, engine, auxiliaryKind } satisfies PendingInstall;
    },
    onSuccess: (ready) => setPendingInstall(ready),
  });
  const confirmInstall = useMutation({
    mutationFn: ({ preflight, installRole, engine, auxiliaryKind }: PendingInstall) => {
      const downloadArguments = [
        preflight.remote_id,
        preflight.source_remote_id,
        installRole,
        engine,
        preflight.revision,
        preflight.selected_files,
        preflight.expected_sha256,
        preflight.file_sources ?? {},
        preflight.comfy_paths,
        preflight.workflow_template_id,
        preflight.workflow_template_sha256,
        preflight.install_plan?.id ?? null,
      ] as const;
      const contentRating = preflight.content_rating ?? "unknown";
      return auxiliaryKind
        ? api.download(...downloadArguments, auxiliaryKind, contentRating)
        : api.download(...downloadArguments, null, contentRating);
    },
    onSuccess: () => {
      setPendingInstall(null);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const installRecipe = useMutation({
    mutationFn: (recipeId: string) => api.installRecipe(recipeId),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const createProfile = useMutation({
    mutationFn: (model: ModelInstall) => api.createProfile(model),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["profiles"] }),
  });
  const updateUseCase = useMutation({
    mutationFn: ({ profileId, useCase }: { profileId: string; useCase: string }) =>
      api.updateProfile(profileId, { use_case: useCase }),
    onSuccess: (updated) => {
      client.setQueryData<ModelProfile[]>(["profiles"], (current) =>
        current?.map((profile) => profile.id === updated.id ? updated : profile) ?? [updated],
      );
      void client.invalidateQueries({ queryKey: ["profiles"] });
    },
  });
  const setDefaultModel = useMutation({
    mutationFn: ({ model, profile }: { model: ModelInstall; profile?: ModelProfile }) =>
      profile
        ? api.updateProfile(profile.id, { is_default: true })
        : api.createProfile(model, true),
    onSuccess: (updated) => {
      client.setQueryData<ModelProfile[]>(["profiles"], (current) => {
        const siblings = (current ?? []).map((profile) => (
          profile.role === updated.role
            ? { ...profile, is_default: profile.id === updated.id }
            : profile
        ));
        return siblings.some((profile) => profile.id === updated.id)
          ? siblings
          : [...siblings, updated];
      });
      void client.invalidateQueries({ queryKey: ["profiles"] });
    },
  });
  const deleteModel = useMutation({
    mutationFn: (modelId: string) => api.deleteModel(modelId, true),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["profiles"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const updateModelAsset = useMutation({
    mutationFn: ({ id, values }: { id: string; values: ModelAssetUpdateValues }) =>
      api.updateModelAsset(id, values),
    onSuccess: (updated) => {
      client.setQueryData<ModelAssetInstall[]>(["model-assets"], (current) =>
        current?.map((asset) => asset.id === updated.id ? updated : asset) ?? [updated],
      );
      void client.invalidateQueries({ queryKey: ["model-assets"] });
    },
  });
  const deleteModelAsset = useMutation({
    mutationFn: api.deleteModelAsset,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["model-assets"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const cleanupDownloads = useMutation({
    mutationFn: api.cleanupDownloads,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["model-storage"] }),
  });
  const importModel = useMutation({
    mutationFn: () => api.importModel({ name: importName, local_path: importPath, role: importRole, engine: importEngine }),
    onSuccess: () => {
      setImportOpen(false);
      setImportName("");
      setImportPath("");
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const installedRemoteIds = new Set(
    (
      role === "lora"
        ? modelAssets.data
          ?.filter((asset) => asset.kind === "lora" && asset.active)
          .map((asset) => asset.manifest_json.remote_id)
        : installed.data
          ?.filter((model) => model.role === role && model.active)
          ?.flatMap((model) => [
            model.manifest_json.remote_id,
            model.manifest_json.source_remote_id,
          ])
    )?.filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const activeDownloadIds = new Set(
    jobs.data
      ?.filter((job) =>
        job.kind === "download"
        && (
          role === "lora"
            ? job.payload_json.auxiliary_kind === "lora"
            : job.payload_json.role === role && !job.payload_json.auxiliary_kind
        )
        && ["queued", "running", "paused"].includes(job.status)
      )
      .map((job) => job.payload_json.remote_id)
      .filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const installedModels = (installed.data ?? []).filter((model) => {
    if (!installedChatCapability) return true;
    if (model.role !== "chat" || model.readiness !== "ready") return false;
    const profile = profiles.data?.find((candidate) => candidate.model_install_id === model.id);
    const modalities = profile?.input_modalities ?? [];
    return installedChatCapability === "vision"
      ? modalities.includes("image")
      : modalities.includes("text") && !modalities.includes("image");
  });
  const installedTemplateIds = new Set(installed.data?.filter((model) => model.role === role && model.active).map((model) => model.manifest_json.workflow_template_id).filter((value): value is string => typeof value === "string") ?? []);
  // A workflow card is installed only when ITS template is: variants share one
  // repository, and installing one must not disable the others.
  const statusFor = (model: CatalogModel): "idle" | "preparing" | "downloading" | "installed" => (
    (model.workflow_template_id ? installedTemplateIds.has(model.workflow_template_id) : installedRemoteIds.has(model.remote_id))
      ? "installed"
      : activeDownloadIds.has(model.remote_id)
        ? "downloading"
        : download.isPending && download.variables?.model.remote_id === model.remote_id && (download.variables?.model.workflow_template_id ?? null) === (model.workflow_template_id ?? null)
          ? "preparing"
          : "idle"
  );
  return (
    <div className="page-view">
      <header className="page-header"><div><h1>Model library</h1></div><div className="storage-actions"><div className="storage-pill"><HardDrive size={17} />{storage.data?.installed_count ?? installed.data?.length ?? 0} installed · {formatBytes(storage.data?.installed_bytes)}</div><button className="secondary compact-button" onClick={() => setImportOpen(true)}><Folder size={16} />Import local</button>{Boolean(storage.data?.partial_download_count) && <button className="secondary compact-button" disabled={cleanupDownloads.isPending} onClick={() => cleanupDownloads.mutate()}>Clean {storage.data?.partial_download_count} partial</button>}</div></header>
      <ModelUpdatesPanel onInstall={(model, selectedRole) => download.mutate({ model, selectedRole })} />
      <section className="recipe-section">
        <div className="section-heading"><div><h2>Reference recipes</h2></div></div>
        {recipes.isLoading && <div className="loading-line" />}
        <FirstFailure of={[recipes, installRecipe]} />
        <div className="recipe-grid">{recipes.data?.map((recipe) => <RecipeCard key={recipe.id} recipe={recipe} pending={installRecipe.isPending && installRecipe.variables === recipe.id} onInstall={() => installRecipe.mutate(recipe.id)} />)}</div>
      </section>
      <div className="toolbar">
        <form className="search-box" onSubmit={(event) => { event.preventDefault(); setSubmitted(query); }}><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models" /></form>
        <select aria-label="Model role" value={role} onChange={(event) => setRole(event.target.value)}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option><option value="lora">LoRA</option></select>
        <select aria-label="Model source" value={catalogSource} onChange={(event) => setCatalogSource(event.target.value)}><option value="huggingface">Hugging Face</option><option value="civitai">CivitAI</option></select><select aria-label="Model order" value={sort} onChange={(event) => setSort(event.target.value)}><option value="trending">Trending</option><option value="downloads">Downloads</option><option value="likes">Likes</option><option value="newest">Newest</option><option value="updated">Recently updated</option><option value="compatible">Compatible first</option></select>
      </div>
      <div className="catalog-filters"><select aria-label="Compatibility filter" value={compatibility} onChange={(event) => setCompatibility(event.target.value)}><option value="">All compatibility</option><option value="likely">Automatic test available</option><option value="advanced_import">Advanced import</option><option value="unsupported">Unsupported</option></select><select aria-label="Last updated filter" value={updatedWithinDays} onChange={(event) => setUpdatedWithinDays(event.target.value)}><option value="">Updated any time</option><option value="7">Updated this week</option><option value="30">Updated this month</option><option value="90">Updated in 3 months</option><option value="365">Updated this year</option></select><input aria-label="Quantization filter" placeholder="Quantization (Q4_K_M, FP8…)" value={quantization} onChange={(event) => setQuantization(event.target.value)} /><input aria-label="Maximum download size" type="number" min="0" placeholder="Max download (GB)" value={maxSizeGb} onChange={(event) => setMaxSizeGb(event.target.value)} /></div>
      {(installed.data?.length ?? 0) > 0 && <section>
        <div className="section-heading">
          <h2>Installed models</h2>
          <select
            aria-label="Installed chat capability"
            value={installedChatCapability}
            onChange={(event) => setInstalledChatCapability(event.target.value)}
          >
            <option value="">All capabilities</option>
            <option value="text">Text only</option>
            <option value="vision">Vision capable</option>
          </select>
        </div>
        <div className="profile-table model-installs">{installedModels.map((model) => {
        const profile = profiles.data?.find((candidate) => candidate.model_install_id === model.id);
        return <InstalledModelRow
          key={model.id}
          model={model}
          profile={profile}
          creating={createProfile.isPending && createProfile.variables?.id === model.id}
          deleting={deleteModel.isPending && deleteModel.variables === model.id}
          saving={updateUseCase.isPending && updateUseCase.variables?.profileId === profile?.id}
          defaulting={setDefaultModel.isPending && setDefaultModel.variables?.model.id === model.id}
          onCreate={() => createProfile.mutate(model)}
          onDelete={() => void confirm({ title: `Delete ${model.name}?`, question: "This removes the model file and its saved settings from local storage. Downloading it again is the only way back.", detail: <WorkflowConsumers kind="model_install" resourceId={model.id} />, confirmLabel: "Delete model" }).then((ok) => ok && deleteModel.mutate(model.id))}
          onSaveUseCase={async (value) => {
            if (!profile) return false;
            try {
              await updateUseCase.mutateAsync({ profileId: profile.id, useCase: value });
              return true;
            } catch {
              return false;
            }
          }}
          onSetDefault={() => setDefaultModel.mutate({ model, profile })}
        />;
        })}</div>
      </section>}
      {(modelAssets.data?.length ?? 0) > 0 && <section>
        <div className="section-heading"><h2>Installed workflow assets</h2></div>
        <div className="profile-table model-installs">
          {modelAssets.data?.map((asset) => (
            <InstalledAssetRow
              key={asset.id}
              asset={asset}
              saving={updateModelAsset.isPending && updateModelAsset.variables?.id === asset.id}
              deleting={deleteModelAsset.isPending && deleteModelAsset.variables === asset.id}
              onUpdate={async (values) => {
                try {
                  await updateModelAsset.mutateAsync({ id: asset.id, values });
                  return true;
                } catch {
                  return false;
                }
              }}
              onDelete={() => void confirm({ title: `Delete ${asset.name}?`, question: "This removes the file from local storage.", detail: <WorkflowConsumers kind="model_asset" resourceId={asset.id} />, confirmLabel: "Delete" }).then((ok) => ok && deleteModelAsset.mutate(asset.id))}
            />
          ))}
        </div>
      </section>}
      {pendingInstall && (
        <InstallConfirmDialog
          name={pendingInstall.model.name || pendingInstall.model.remote_id}
          preflight={pendingInstall.preflight}
          system={machine.data}
          pending={confirmInstall.isPending}
          onConfirm={() => confirmInstall.mutate(pendingInstall)}
          onCancel={() => setPendingInstall(null)}
        />
      )}
      <FirstFailure of={[createProfile, download, confirmInstall, deleteModel, cleanupDownloads, updateUseCase, setDefaultModel, updateModelAsset, deleteModelAsset]} />
      {/* isFetching, not isLoading: the latter is only true the first
          time, so changing a filter swapped the results with no sign
          anything had happened - which reads as the page refreshing
          itself for no reason. */}
      {catalog.isFetching && !catalog.isFetchingNextPage && (
        <div className="catalog-loading" role="status">
          <div className="loading-line" />
          <span>{catalogItems.length > 0 ? "Finding models…" : "Loading the catalogue…"}</span>
        </div>
      )}
      <ErrorCallout message={catalog.error?.message} action={<button className="secondary compact-button" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Retry</button>} />
      {catalogIsStale && !catalog.error && <div className="callout warning action-callout" role="status"><span>Showing saved results while Hugging Face is unavailable.</span><button className="secondary compact-button" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Refresh</button></div>}
      <div className={`model-grid ${catalog.isFetching && !catalog.isFetchingNextPage ? "superseded" : ""}`}>{catalogItems.map((model) => <ModelCard key={model.remote_id} model={model} role={role} runtime={runtimeFor(model)} status={statusFor(model)} onDownload={() => download.mutate({ model, selectedRole: role })} onChooseVersion={model.provider === "civitai" && model.parent_model_id ? () => setChoosingVersions(model) : undefined} />)}</div>
      {choosingVersions?.parent_model_id && (
        <VersionChooser modelId={choosingVersions.parent_model_id} modelName={choosingVersions.parent_model_name ?? choosingVersions.name}
          onClose={() => setChoosingVersions(null)}
          onChoose={(versionId) => { setChoosingVersions(null); download.mutate({ model: { ...choosingVersions, remote_id: versionId }, selectedRole: role }); }} />
      )}
      {catalog.hasNextPage && <div className="load-more"><button className="secondary" disabled={catalog.isFetchingNextPage} onClick={() => void catalog.fetchNextPage()}>{catalog.isFetchingNextPage ? "Loading…" : "Load more models"}</button></div>}
      {importOpen && (
        <AccessibleDialog
          title="Import a local model"
          eyebrow="Advanced import"
          closeLabel="Close local import"
          onClose={() => setImportOpen(false)}
        >
          <p>Register a local file or folder. Pickle-compatible formats are blocked as unsafe, and imports require review before use.</p>
          <label>Display name<input value={importName} onChange={(event) => setImportName(event.target.value)} /></label>
          <label>Absolute local path<input value={importPath} onChange={(event) => setImportPath(event.target.value)} placeholder="/path/to/model.gguf" /></label>
          <label>Role<select value={importRole} onChange={(event) => { const next = event.target.value; setImportRole(next); setImportEngine(next === "chat" ? "llama.cpp" : "comfyui"); }}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select></label>
          <label>Runtime<select value={importEngine} onChange={(event) => setImportEngine(event.target.value)}><option value="llama.cpp">llama.cpp</option><option value="vllm">vLLM (ModelOpt/NVFP4)</option><option value="comfyui">ComfyUI</option></select></label>
          {importModel.error && <ErrorCallout message={importModel.error.message} />}
          <footer><button className="secondary" onClick={() => setImportOpen(false)}>Cancel</button><button className="primary" disabled={!importName.trim() || !importPath.trim() || importModel.isPending} onClick={() => importModel.mutate()}>{importModel.isPending ? "Importing…" : "Import model"}</button></footer>
        </AccessibleDialog>
      )}
      {confirmDialog}
    </div>
  );
}




function Sidebar({
  projects,
  chats,
  engines,
  presets,
  currentChatId,
  view,
  setupState,
  onChat,
  onSetup,
  onView,
  onNewChat,
  onNewProject,
  onExportProject,
  onImportProject,
  onUpdateChat,
  onDeleteChat,
  onUpdateProject,
  onDeleteProject,
  sidebar,
}: {
  projects: Project[];
  chats: Chat[];
  engines: EngineCapabilities[];
  presets: GenerationPreset[];
  currentChatId: string | null;
  view: View;
  setupState?: SetupReadinessReport["state"] | undefined;
  onChat: (id: string) => void;
  onSetup: () => void;
  onView: (view: View) => void;
  onNewChat: (projectId?: string | null) => void;
  onNewProject: (name: string) => void;
  onExportProject: (id: string, includeMedia?: boolean) => void;
  onImportProject: (file: File) => void;
  onUpdateChat: (id: string, values: Partial<Chat>) => void;
  onDeleteChat: (id: string, deleteGeneratedMedia: boolean) => void;
  onUpdateProject: (id: string, values: Partial<Project>) => void;
  onDeleteProject: (id: string) => void;
  sidebar: SidebarLayout;
}) {
  const [naming, setNaming] = useState(false);
  const [closedProjects, setClosedProjects] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [managedChat, setManagedChat] = useState<Chat | null>(null);
  const [managedProject, setManagedProject] = useState<Project | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const projectImport = useRef<HTMLInputElement>(null);
  const normalizedSearch = search.trim().toLowerCase();
  const visibleChats = chats.filter((chat) => (showArchived || !chat.archived) && (!normalizedSearch || chat.title.toLowerCase().includes(normalizedSearch)));
  const visibleProjects = projects.filter((project) => (showArchived || !project.archived) && (!normalizedSearch || project.name.toLowerCase().includes(normalizedSearch) || visibleChats.some((chat) => chat.project_id === project.id)));
  const unfiled = visibleChats.filter((chat) => !chat.project_id);
  const chatRow = (chat: Chat) => <div className="sidebar-chat-row" key={chat.id}><button className={`chat-main ${view === "chat" && currentChatId === chat.id ? "active" : ""}`} aria-current={view === "chat" && currentChatId === chat.id ? "page" : undefined} onClick={() => { onChat(chat.id); setMobileOpen(false); }}><span>{chat.title}</span>{chat.archived && <small>Archived</small>}</button><button className={`inline-add sidebar-pin ${chat.pinned ? "pinned" : ""}`} aria-label={chat.pinned ? `Unpin ${chat.title}` : `Pin ${chat.title}`} aria-pressed={chat.pinned} title={chat.pinned ? "Unpin" : "Pin"} onClick={() => onUpdateChat(chat.id, { pinned: !chat.pinned })}><Pin size={13} /></button><button className="inline-add" aria-label={`Manage ${chat.title}`} onClick={() => setManagedChat(chat)}><MoreHorizontal size={13} /></button></div>;
  return (
    <>
    <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><AtelierMark /></div><span>LM Atelier<small>Local creative studio</small></span><button className="icon-button mobile-menu" aria-label="Toggle navigation" aria-expanded={mobileOpen} onClick={() => setMobileOpen((open) => !open)}><Menu /></button></div>
      <button className="new-chat" onClick={() => { onNewChat(null); setMobileOpen(false); }}><Plus size={18} />New chat</button>
      <nav className="primary-nav"><button className={view === "media" ? "active" : ""} aria-current={view === "media" ? "page" : undefined} onClick={() => { onView("media"); setMobileOpen(false); }}><ImageIcon />Media library</button><button className={view === "models" ? "active" : ""} aria-current={view === "models" ? "page" : undefined} onClick={() => { onView("models"); setMobileOpen(false); }}><Library />Model library</button><button className={view === "references" ? "active" : ""} aria-current={view === "references" ? "page" : undefined} onClick={() => { onView("references"); setMobileOpen(false); }}><Star />References</button><button className={view === "workflows" ? "active" : ""} aria-current={view === "workflows" ? "page" : undefined} onClick={() => { onView("workflows"); setMobileOpen(false); }}><WorkflowIcon />Workflows</button><button className={view === "studio" ? "active" : ""} aria-current={view === "studio" ? "page" : undefined} onClick={() => { onView("studio"); setMobileOpen(false); }}><ImageStudioIcon />Image Studio</button></nav>
      <div className="workspace-search"><Search size={14} /><input aria-label="Search projects and chats" placeholder="Search workspace" value={search} onChange={(event) => setSearch(event.target.value)} /><button className={showArchived ? "active" : ""} aria-pressed={showArchived} onClick={() => setShowArchived((value) => !value)}>Archived</button></div>
      <div className="workspace-tree" role="region" aria-label="Projects and chats">
        <div className="sidebar-section">
          <div className="section-title"><span>Projects</span><input ref={projectImport} hidden type="file" accept=".zip,.lm-atelier.zip,application/zip" onChange={(event) => { const file = event.target.files?.[0]; if (file) onImportProject(file); event.target.value = ""; }} /><button aria-label="Import project" onClick={() => projectImport.current?.click()}><Upload size={14} /></button><button aria-label="New project" onClick={() => setNaming(true)}><Plus size={15} /></button></div>
          {visibleProjects.map((project) => {
            const open = !closedProjects.has(project.id);
            const projectMatches = normalizedSearch && project.name.toLowerCase().includes(normalizedSearch);
            const projectChats = chats.filter((chat) => chat.project_id === project.id && (showArchived || !chat.archived) && (!normalizedSearch || projectMatches || chat.title.toLowerCase().includes(normalizedSearch)));
            return (
              <div className="project-group" key={project.id}>
                <div className="project-row">
                  <button className="project-main" aria-expanded={open} onClick={() => setClosedProjects((current) => {
                    const next = new Set(current);
                    if (open) next.add(project.id);
                    else next.delete(project.id);
                    return next;
                  })}>
                    <ChevronDown className={open ? "" : "closed"} size={14} />
                    <Folder size={16} />
                    <span>{project.name}</span>
                  </button>
                  <button className="inline-add" onClick={() => { onNewChat(project.id); setMobileOpen(false); }} aria-label={`New chat in ${project.name}`}><Plus size={13} /></button>
                  <button className="inline-add" onClick={() => onExportProject(project.id)} aria-label={`Export ${project.name}`}><Download size={13} /></button>
                  <button className="inline-add" onClick={() => setManagedProject(project)} aria-label={`Manage ${project.name}`}><MoreHorizontal size={13} /></button>
                </div>
                {open && <div className="chat-list">{projectChats.map(chatRow)}</div>}
              </div>
            );
          })}
        </div>
        {unfiled.length > 0 && <div className="sidebar-section"><div className="section-title"><span>Chats</span></div><div className="chat-list standalone">{unfiled.map(chatRow)}</div></div>}
      </div>
      {naming && <PromptDialog title="New project" label="Project name" confirmLabel="Create project" placeholder="Portrait studies" onCancel={() => setNaming(false)} onConfirm={(name) => { setNaming(false); onNewProject(name); }} />}
      <SidebarFooter setupState={setupState} view={view} onSetup={onSetup} onView={onView} onNavigate={() => setMobileOpen(false)} />
      {managedChat && <ChatManager chat={managedChat} projects={projects} onClose={() => setManagedChat(null)} onSave={(values) => { onUpdateChat(managedChat.id, values); setManagedChat(null); }} onDelete={(deleteGeneratedMedia) => { onDeleteChat(managedChat.id, deleteGeneratedMedia); setManagedChat(null); }} />}
      {managedProject && <ProjectManager project={managedProject} engines={engines} presets={presets} onClose={() => setManagedProject(null)} onSave={(values) => { onUpdateProject(managedProject.id, values); setManagedProject(null); }} onDelete={() => { onDeleteProject(managedProject.id); setManagedProject(null); }} onExport={(includeMedia) => onExportProject(managedProject.id, includeMedia)} />}
    </aside>
      <SidebarResizer layout={sidebar} />
    </>
  );
}

export default function App() {
  const client = useQueryClient();
  const [view, setView] = useState<View>("chat");
  const { appearance, sidebar } = useWorkspaceChrome();
  const [studioSource, setStudioSource] = useState<{ artifactId: string; chatId: string | null } | null>(null);
  const [modelLibraryRole, setModelLibraryRole] = useState<EngineRole>("chat");
  const [setupOpen, setSetupOpen] = useState<boolean | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(() => localStorage.getItem(CURRENT_CHAT_KEY));
  const [liveText, setLiveText] = useState<Record<string, string>>({});
  const [chatDrafts, setChatDrafts] = useState<Record<string, Partial<Chat>>>({});
  const [pendingTurns, setPendingTurns] = useState<Record<string, PendingTurn[]>>({});
  const setupReadiness = useQuery({
    queryKey: ["setup-readiness"],
    queryFn: api.setupReadiness,
    refetchInterval: (query) => query.state.data?.state === "ready" ? false : 3_000,
  });
  const setupVisible = setupOpen ?? Boolean(setupReadiness.data && setupReadiness.data.state !== "ready" && sessionStorage.getItem(SETUP_DISMISSED_KEY) !== "1");
  const [firstRunSetup, exitFirstRunSetup] = useFirstRunSetup();
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: () => api.projects(true),
  });
  const chats = useQuery({ queryKey: ["chats"], queryFn: () => api.chats(null, true) });
  const firstActiveChatId = chats.data?.find((candidate) => !candidate.archived)?.id ?? null;
  const activeChatId = currentChatId ?? firstActiveChatId;
  const chat = useQuery({ queryKey: ["chat", activeChatId], queryFn: () => api.chat(activeChatId!), enabled: Boolean(activeChatId) });
  const workPlans = useQuery({
    queryKey: ["work-plans", activeChatId],
    queryFn: () => api.workPlans(activeChatId!),
    enabled: Boolean(activeChatId),
    refetchInterval: 3_000,
  });
  const engines = useQuery({ queryKey: ["engines"], queryFn: api.engines });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: api.workflows });
  const applicationInfo = useQuery({ queryKey: ["about"], queryFn: api.about });
  const eventsConnected = useLiveEvents(client, setLiveText);

  const createChat = useMutation({
    mutationFn: (projectId?: string | null) => api.createChat(projectId),
    onSuccess: (created) => {
      setCurrentChatId(created.id);
      localStorage.setItem(CURRENT_CHAT_KEY, created.id);
      setView("chat");
      focusMainContent();
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["projects"] }),
  });
  const applyAcceptedTurn = (chatId: string, accepted: TurnAccepted) => {
    client.setQueryData<ChatDetail>(["chat", chatId], (current) => {
      if (!current) return current;
      const messageIds = new Set(current.messages.map((message) => message.id));
      const acceptedMessages = [accepted.user_message, accepted.assistant_message]
        .filter((message) => !messageIds.has(message.id));
      return {
        ...current,
        active_head_message_id: accepted.assistant_message.id,
        messages: [...current.messages, ...acceptedMessages],
      };
    });
    void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
    void client.invalidateQueries({ queryKey: ["chats"] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
    void client.invalidateQueries({ queryKey: ["work-plans", chatId] });
  };
  const send = useMutation({
    mutationFn: ({ chatId, id, text, mode, artifacts, settings, references, outputCount, stopCurrent }: SendTurnVariables) => {
      const optionalOutputCount: [] | [number] = outputCount === undefined ? [] : [outputCount];
      return stopCurrent
        ? api.stopAndSendTurn(chatId, text, mode, artifacts, settings, id, references, ...optionalOutputCount)
        : api.sendTurn(chatId, text, mode, artifacts, settings, id, "turns", undefined, references, ...optionalOutputCount);
    },
    onMutate: ({ chatId, id, text, mode }) => {
      setPendingTurns((current) => ({
        ...current,
        [chatId]: [...(current[chatId] ?? []), { id, text, mode }],
      }));
    },
    onSuccess: (accepted, { chatId }) => applyAcceptedTurn(chatId, accepted),
    onSettled: (_accepted, _error, { chatId, id }) => {
      setPendingTurns((current) => {
        const remaining = (current[chatId] ?? []).filter((pending) => pending.id !== id);
        const next = { ...current };
        if (remaining.length) next[chatId] = remaining;
        else delete next[chatId];
        return next;
      });
    },
  });
  const cancelWorkPlan = useMutation({
    mutationFn: api.cancelWorkPlan,
    onSuccess: (plan) => {
      void client.invalidateQueries({ queryKey: ["chat", plan.chat_id] });
      void client.invalidateQueries({ queryKey: ["work-plans", plan.chat_id] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const { deleteExchange, forkThread } = useMessageActions(setCurrentChatId, setView);
  const refreshWorkStep = () => {
    void client.invalidateQueries({ queryKey: ["chat", activeChatId] });
    void client.invalidateQueries({ queryKey: ["work-plans", activeChatId] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
  };
  const cancelWorkStep = useMutation({
    mutationFn: api.cancelWorkStep,
    onSuccess: refreshWorkStep,
  });
  const retryWorkStep = useMutation({
    mutationFn: api.retryWorkStep,
    onSuccess: refreshWorkStep,
  });
  const regenerate = useMutation({
    mutationFn: ({ messageId, settings }: { chatId: string; messageId: string; settings: Record<string, unknown> }) =>
      api.regenerateMessage(messageId, settings),
    onSuccess: (_accepted, { chatId }) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const selectResponseRevision = useMutation({
    mutationFn: ({ messageId, revisionId }: { chatId: string; messageId: string; revisionId: string }) =>
      api.selectResponseRevision(messageId, revisionId),
    onSuccess: (_message, { chatId }) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId], exact: true });
    },
  });
  const branch = useMutation({
    mutationFn: ({ messageId, text, mode, settings }: { chatId: string; messageId: string; text: string; mode: RoutingMode; settings: Record<string, unknown> }) =>
      api.branchMessage(messageId, text, mode, settings),
    onSuccess: (accepted, { chatId }) => applyAcceptedTurn(chatId, accepted),
  });
  const stop = useMutation({
    mutationFn: (chatId: string) => api.cancelChat(chatId),
    onSuccess: (_job, chatId) => {
      void client.invalidateQueries({ queryKey: ["chat", chatId] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const updateChat = useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Chat> }) => api.updateChat(id, values),
    onMutate: ({ id, values }) => {
      void client.cancelQueries({ queryKey: ["chat", id] });
      const previousChat = client.getQueryData<ChatDetail>(["chat", id]);
      const previousChats = client.getQueryData<Chat[]>(["chats"]);
      client.setQueryData<ChatDetail>(["chat", id], (current) => (
        current ? { ...current, ...values } : current
      ));
      client.setQueryData<Chat[]>(["chats"], (current) => current?.map((item) => (
        item.id === id ? { ...item, ...values } : item
      )));
      return { previousChat, previousChats };
    },
    onError: (_error, { id }, context) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      if (context?.previousChat) client.setQueryData(["chat", id], context.previousChat);
      if (context?.previousChats) client.setQueryData(["chats"], context.previousChats);
    },
    onSuccess: (updated, { id, values }) => {
      if (updated) {
        client.setQueryData<ChatDetail>(["chat", id], (current) => (
          current ? { ...current, ...updated } : current
        ));
        client.setQueryData<Chat[]>(["chats"], (current) => current?.map((item) => (
          item.id === id ? { ...item, ...updated } : item
        )));
        setChatDrafts((current) => {
          const draft = current[id];
          if (!draft) return current;
          const remaining = { ...draft };
          for (const key of Object.keys(values) as (keyof Chat)[]) {
            if (remaining[key] === values[key]) delete remaining[key];
          }
          const next = { ...current };
          if (Object.keys(remaining).length) next[id] = remaining;
          else delete next[id];
          return next;
        });
      }
    },
    onSettled: (updated, error, { id }) => {
      if (updated || error) {
        void client.invalidateQueries({ queryKey: ["chat", id] });
        void client.invalidateQueries({ queryKey: ["chats"] });
      }
    },
  });
  const manageChat = useMutation({
    mutationFn: ({ id, values }: { id: string; values: Partial<Chat> }) => api.updateChat(id, values),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["chat"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const deleteChat = useMutation({
    mutationFn: ({ id, deleteGeneratedMedia }: { id: string; deleteGeneratedMedia: boolean }) => api.deleteChat(id, deleteGeneratedMedia),
    onMutate: async ({ id: deletedId }) => {
      await client.cancelQueries({ queryKey: ["chats"] });
      const previousChats = client.getQueryData<Chat[]>(["chats"]) ?? [];
      const remainingChats = previousChats.filter((candidate) => candidate.id !== deletedId);
      const previousCurrentChatId = currentChatId;
      client.setQueryData<Chat[]>(["chats"], remainingChats);
      if (activeChatId === deletedId) {
        const nextChatId = remainingChats.find((candidate) => !candidate.archived)?.id ?? null;
        setCurrentChatId(nextChatId);
        if (nextChatId) localStorage.setItem(CURRENT_CHAT_KEY, nextChatId);
        else localStorage.removeItem(CURRENT_CHAT_KEY);
      }
      client.removeQueries({ queryKey: ["chat", deletedId], exact: true });
      return { previousChats, previousCurrentChatId };
    },
    onSuccess: (_value, { id: deletedId }) => {
      setChatDrafts((current) => {
        const next = { ...current };
        delete next[deletedId];
        return next;
      });
      void client.invalidateQueries({ queryKey: ["artifacts"] });
      void client.invalidateQueries({ queryKey: ["artifact-storage"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (_error, _deletedChat, context) => {
      if (!context) return;
      client.setQueryData(["chats"], context.previousChats);
      setCurrentChatId(context.previousCurrentChatId);
      if (context.previousCurrentChatId) localStorage.setItem(CURRENT_CHAT_KEY, context.previousCurrentChatId);
      else localStorage.removeItem(CURRENT_CHAT_KEY);
    },
    onSettled: () => void client.invalidateQueries({ queryKey: ["chats"] }),
  });
  const { updateProject, deleteProject, exportProject, importProject } = useProjectMutations({
    client,
    onImportedChat: (chatId) => {
      setCurrentChatId(chatId);
      localStorage.setItem(CURRENT_CHAT_KEY, chatId);
      setView("chat");
    },
  });

  const openLibraryImage = useCallback((artifactId: string) => {
    setStudioSource({ artifactId, chatId: null });
    setView("studio");
    focusMainContent();
  }, []);

  const allChats = useMemo(() => chats.data ?? [], [chats.data]);
  const allProjects = useMemo(() => projects.data ?? [], [projects.data]);
  // One place that knows what opening the library means, since three
  // different surfaces send people there.
  const openWorkflows = () => { setView("workflows"); focusMainContent(); };
  const activeContent = useMemo(() => {
    if (view === "studio") {
      return (
        <StudioView
          sourceArtifactId={studioSource?.artifactId ?? null}
          sourceChatId={studioSource?.chatId ?? null}
          onOpenArtifact={(artifactId) => setStudioSource({ artifactId, chatId: null })}
          onOpenWorkflows={openWorkflows}
          onClose={() => setStudioSource(null)}/>
      );
    }
    if (view === "media") return <MediaLibraryView onEditImage={openLibraryImage} />;
    if (view === "models") return <ModelsView key={modelLibraryRole} initialRole={modelLibraryRole} />;
    if (view === "references") return <ReferencesLibrary />;
    if (view === "workflows") return <WorkflowsView />;
    if (view === "settings") return <SettingsView engines={engines.data ?? []} />;
    const displayedChat = chat.data
      ? { ...chat.data, ...(chatDrafts[chat.data.id] ?? {}) }
      : undefined;
    const selectedRole = roleForMode(displayedChat?.routing_mode ?? "auto");
    const scopedSettings = displayedChat?.generation_settings_json?.[selectedRole] ?? {};
    const presetId = displayedChat?.generation_preset_ids_json?.[selectedRole] ?? null;
    const persistActiveChat = (values: Partial<Chat>) => {
      if (!displayedChat) return;
      setChatDrafts((current) => ({
        ...current,
        [displayedChat.id]: { ...(current[displayedChat.id] ?? {}), ...values },
      }));
      client.setQueryData<ChatDetail>(["chat", displayedChat.id], (current) => (
        current ? { ...current, ...values } : current
      ));
      updateChat.mutate({ id: displayedChat.id, values });
    };
    return <ChatView key={displayedChat?.id ?? "empty-chat"} onOpenStudio={(artifactId) => { setStudioSource({ artifactId, chatId: displayedChat?.id ?? null }); setView("studio"); focusMainContent(); }} chat={displayedChat} engines={engines.data ?? []} profiles={profiles.data ?? []} presets={presets.data ?? []} workflows={workflows.data ?? []} project={allProjects.find((item) => item.id === displayedChat?.project_id)} liveText={liveText} pendingTurns={displayedChat ? pendingTurns[displayedChat.id] ?? [] : []} workPlans={workPlans.data ?? []} settings={scopedSettings} presetId={presetId} maxMediaOutputsPerPlan={applicationInfo.data?.max_media_outputs_per_plan ?? 1} onSettings={(settings) => {
      if (!displayedChat) return;
      const role = roleForMode(displayedChat.routing_mode);
      persistActiveChat({
        generation_settings_json: {
          ...(displayedChat.generation_settings_json ?? {}),
          [role]: settings,
        },
      });
    }} onPreset={(selectedPresetId) => {
      if (!displayedChat) return;
      const role = roleForMode(displayedChat.routing_mode);
      const bindings = { ...(displayedChat.generation_preset_ids_json ?? {}) };
      if (selectedPresetId) bindings[role] = selectedPresetId;
      else delete bindings[role];
      persistActiveChat({ generation_preset_ids_json: bindings });
    }} onMode={(mode) => {
      persistActiveChat({ routing_mode: mode });
    }} onRegenerate={(messageId, settings) => {
      if (displayedChat) regenerate.mutate({ chatId: displayedChat.id, messageId, settings });
    }} onSelectRevision={(messageId, revisionId) => {
      if (displayedChat) {
        selectResponseRevision.mutate({
          chatId: displayedChat.id,
          messageId,
          revisionId,
        });
      }
    }} onEdit={(messageId, text, mode, settings) => {
      if (displayedChat) branch.mutate({
        chatId: displayedChat.id,
        messageId,
        text,
        mode,
        settings,
      });
    }} onStop={() => {
      if (displayedChat) stop.mutate(displayedChat.id);
    }} onStopAndSend={(text, mode, artifacts, settings, references, outputCount) => {
      if (displayedChat) {
        send.mutate({ chatId: displayedChat.id, id: crypto.randomUUID(), text, mode, artifacts, settings, references, outputCount, stopCurrent: true });
      }
    }} onDeleteExchange={deleteExchange.mutate} onForkThread={forkThread.mutate} onCancelPlan={(planId) => {
      cancelWorkPlan.mutate(planId);
    }} onCancelStep={(stepId) => {
      cancelWorkStep.mutate(stepId);
    }} onRetryStep={(stepId) => {
      retryWorkStep.mutate(stepId);
    }} onSend={(text, mode, artifacts, settings, references, outputCount) => {
      if (displayedChat) {
        send.mutate({ chatId: displayedChat.id, id: crypto.randomUUID(), text, mode, artifacts, settings, references, outputCount });
      }
    }} />;
  }, [studioSource, view, modelLibraryRole, engines.data, profiles.data, presets.data, workflows.data, applicationInfo.data, allProjects, chat.data, chatDrafts, liveText, pendingTurns, workPlans.data, send, regenerate, selectResponseRevision, branch, stop, cancelWorkPlan, cancelWorkStep, retryWorkStep, updateChat, deleteExchange, forkThread, client, openLibraryImage]);

  if (firstRunSetup && setupReadiness.data) {
    return <FirstRunSetup report={setupReadiness.data} onExit={exitFirstRunSetup} onOpenModels={(role) => { exitFirstRunSetup(); setModelLibraryRole(role); setView("models"); }} onOpenWorkflows={() => { exitFirstRunSetup(); setView("workflows"); }} />;
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Sidebar projects={allProjects} chats={allChats} engines={engines.data ?? []} presets={presets.data ?? []} currentChatId={activeChatId} view={view} setupState={setupReadiness.data?.state} onSetup={() => setSetupOpen(true)} onChat={(id) => { setCurrentChatId(id); localStorage.setItem(CURRENT_CHAT_KEY, id); setView("chat"); focusMainContent(); }} onView={(nextView) => { setView(nextView); focusMainContent(); }} onNewChat={(projectId) => createChat.mutate(projectId)} onNewProject={(name) => createProject.mutate(name)} onExportProject={(id, includeMedia) => exportProject.mutate({ id, includeMedia })} onImportProject={(file) => importProject.mutate(file)} onUpdateChat={(id, values) => manageChat.mutate({ id, values })} onDeleteChat={(id, deleteGeneratedMedia) => deleteChat.mutate({ id, deleteGeneratedMedia })} onUpdateProject={(id, values) => updateProject.mutate({ id, values })} onDeleteProject={(id) => deleteProject.mutate(id)} sidebar={sidebar} />
      <main id="main-content" tabIndex={-1}>{activeContent}</main>
      <ThemeToggle appearance={appearance} />
      <SetupSurface
        open={setupOpen}
        visible={setupVisible}
        report={setupReadiness.data}
        error={setupReadiness.error}
        onRetry={() => void setupReadiness.refetch()}
        onDismiss={() => {
          sessionStorage.setItem(SETUP_DISMISSED_KEY, "1");
          setSetupOpen(false);
        }}
        onClose={() => setSetupOpen(false)}
        onOpenModels={(role) => {
          setModelLibraryRole(role);
          setView("models");
          setSetupOpen(false);
          focusMainContent();
        }}
        onOpenWorkflows={() => { setSetupOpen(false); openWorkflows(); }}
      />
      <JobsPanel />
      <GlobalNotices connected={eventsConnected} mutations={[send, regenerate, selectResponseRevision, branch, stop, cancelWorkPlan, cancelWorkStep, retryWorkStep, updateChat, createChat, createProject, exportProject, importProject, manageChat, deleteChat, updateProject, deleteProject, deleteExchange, forkThread]} />
    </div>
  );
}
