from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DDL,
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from .artifact_library_schema import CREATE_TRIGGER_SQL, DROP_TRIGGER_SQL
from .db import Base
from .domain import (
    ArtifactKind,
    CompatibilityLevel,
    JobKind,
    JobStatus,
    MessageRole,
    MessageStatus,
    ModelRole,
    Operation,
    PartType,
    RoutingMode,
    RunStatus,
    new_id,
    utcnow,
)
from .media_organization_schema import (
    CREATE_MEDIA_ORGANIZATION_TRIGGER_SQL,
    DROP_MEDIA_ORGANIZATION_TRIGGER_SQL,
)
from .reference_review_schema import (
    CREATE_REFERENCE_REVIEW_TRIGGER_SQL,
    DROP_REFERENCE_REVIEW_TRIGGER_SQL,
)


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def _install_sqlite_trigger(statement: str) -> Callable[..., None]:
    """Create a missing canonical trigger and refuse an existing divergent one.

    Alembic owns upgrades and deliberately uses strict trigger statements.
    ``create_all`` is also called during application startup, where metadata
    ``after_create`` events run even when no table is created. An existing
    name is insufficient authority: a stale or weakened body must fail closed.
    """

    parts = statement.split(None, 3)
    if len(parts) < 4 or parts[:2] != ["CREATE", "TRIGGER"]:
        raise ValueError("SQLite trigger statement has no CREATE TRIGGER clause")
    trigger_name = parts[2]
    canonical = statement.strip()

    def install(_target: object, connection: Connection, **_kwargs: object) -> None:
        if connection.dialect.name != "sqlite":
            return
        installed = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (trigger_name,),
        ).scalar_one_or_none()
        if installed is None:
            connection.exec_driver_sql(canonical)
            return
        if not isinstance(installed, str) or installed.strip() != canonical:
            raise RuntimeError(
                f"SQLite trigger {trigger_name} does not match the application schema"
            )

    return install


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("proj"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    image_workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="SET NULL"), nullable=True
    )
    video_workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="SET NULL"), nullable=True
    )
    generation_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation_preset_ids_json: Mapped[dict[str, str | None]] = mapped_column(JSON, default=dict)

    chats: Mapped[list[Chat]] = relationship(back_populates="project")


class Chat(TimestampMixin, Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("chat"))
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(240), default="New chat")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    scope: Mapped[str] = mapped_column(String(24), default="standard", index=True)
    draft_prompt: Mapped[str] = mapped_column(Text, default="")
    routing_mode: Mapped[str] = mapped_column(String(16), default=RoutingMode.AUTO.value)
    confirm_uncertain_media: Mapped[bool] = mapped_column(Boolean, default=True)
    active_chat_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_vision_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_image_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_video_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    active_head_message_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    generation_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation_preset_ids_json: Mapped[dict[str, str | None]] = mapped_column(JSON, default=dict)
    vision_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Whether this conversation may reach the internet. Empty means no. It is
    # deliberately per-chat and never copied into a new one: permission that
    # spreads by default is permission nobody granted.
    web_settings_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default="{}"
    )
    # Where a forked thread came from, empty for chats created directly.
    origin_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, server_default="{}")

    project: Mapped[Project | None] = relationship(back_populates="chats")
    messages: Mapped[list[Message]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="chat", cascade="all, delete-orphan")


class Message(TimestampMixin, Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("msg"))
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True
    )
    role: Mapped[str] = mapped_column(String(16), default=MessageRole.USER.value)
    status: Mapped[str] = mapped_column(String(16), default=MessageStatus.COMPLETE.value)
    transcript_visible: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="1",
        index=True,
    )
    active_response_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)

    chat: Mapped[Chat] = relationship(back_populates="messages")
    parts: Mapped[list[MessagePart]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="MessagePart.position",
    )
    response_revisions: Mapped[list[ResponseRevision]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="ResponseRevision.sequence",
        foreign_keys="ResponseRevision.message_id",
    )
    feedback_rows: Mapped[list[ResponseFeedback]] = relationship(
        cascade="all, delete-orphan", foreign_keys="ResponseFeedback.message_id"
    )
    # Read-only here. `message_references` is the sole writer, because these
    # rows are written once and never revised - a relationship that could
    # append or reorder them would be a way to rewrite history from the side.
    references: Mapped[list[MessageReference]] = relationship(
        order_by="MessageReference.position",
        viewonly=True,
    )

    @property
    def feedback(self) -> str | None:
        """The base response's verdict; revisions carry their own."""
        row = next((item for item in self.feedback_rows if item.response_revision_id is None), None)
        return row.rating if row else None


class MessagePart(TimestampMixin, Base):
    __tablename__ = "message_parts"
    __table_args__ = (UniqueConstraint("message_id", "position", name="uq_message_part_position"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("part"))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32), default=PartType.TEXT.value)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    message: Mapped[Message] = relationship(back_populates="parts")
    artifact: Mapped[Artifact | None] = relationship()


class WorkPlan(TimestampMixin, Base):
    __tablename__ = "work_plans"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "transcript_sequence",
            name="uq_work_plan_transcript_sequence",
        ),
        UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_work_plan_chat_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("plan"))
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_action: Mapped[str] = mapped_column(String(32), default="send")
    persistence_scope: Mapped[str] = mapped_column(String(16), default="durable")
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    context_head_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    transcript_sequence: Mapped[int] = mapped_column(Integer)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    planner_version: Mapped[str] = mapped_column(String(32), default="legacy-turn-v1")
    failure_policy: Mapped[str] = mapped_column(String(32), default="stop_dependents")
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    steps: Mapped[list[WorkStep]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="WorkStep.ordinal",
    )


class WorkStep(TimestampMixin, Base):
    __tablename__ = "work_steps"
    __table_args__ = (
        UniqueConstraint("plan_id", "ordinal", name="uq_work_step_ordinal"),
        UniqueConstraint("run_id", name="uq_work_step_run"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("step"))
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("work_plans.id", ondelete="CASCADE"),
        index=True,
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    display_group: Mapped[str | None] = mapped_column(String(80), nullable=True)
    operation: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    prompt: Mapped[str] = mapped_column(Text, default="")
    profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    workflow_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_bindings_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    output_contract_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    queue_class: Mapped[str] = mapped_column(String(32), default="interactive_compute")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    plan: Mapped[WorkPlan] = relationship(back_populates="steps")


class WorkStepDependency(Base):
    __tablename__ = "work_step_dependencies"

    step_id: Mapped[str] = mapped_column(
        ForeignKey("work_steps.id", ondelete="CASCADE"),
        primary_key=True,
    )
    depends_on_step_id: Mapped[str] = mapped_column(
        ForeignKey("work_steps.id", ondelete="CASCADE"),
        primary_key=True,
    )


class Run(TimestampMixin, Base):
    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_runs_chat_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("run"))
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    user_message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    assistant_message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), unique=True
    )
    work_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    work_step_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_steps.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    operation: Mapped[str] = mapped_column(String(32), default=Operation.TEXT.value)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.PENDING.value, index=True)
    standalone_prompt: Mapped[str] = mapped_column(Text, default="")
    profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    vision_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    workflow_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    chat: Mapped[Chat] = relationship(back_populates="runs")


class ResponseFeedback(TimestampMixin, Base):
    """One local preference verdict on a response or one of its revisions.

    The run is the provenance anchor - it already records the model, profile,
    and effective settings - so feedback stores a pointer, never a copy. A
    verdict is per (message, revision) and overwrites in place; clearing
    deletes the row. Nothing here trains weights; consumers are local
    preference matching and evaluation.
    """

    __tablename__ = "response_feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "response_revision_id", name="uq_response_feedback_target"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("fb"))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    response_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("response_revisions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    rating: Mapped[str] = mapped_column(String(8))


class ResponseRevision(TimestampMixin, Base):
    __tablename__ = "response_revisions"
    __table_args__ = (
        UniqueConstraint("message_id", "sequence", name="uq_response_revision_sequence"),
        UniqueConstraint("run_id", name="uq_response_revision_run"),
        Index(
            "uq_response_revision_pending_message",
            "message_id",
            unique=True,
            sqlite_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("rev"))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default=MessageStatus.PENDING.value, index=True)

    feedback_rows: Mapped[list[ResponseFeedback]] = relationship(
        cascade="all, delete-orphan", foreign_keys="ResponseFeedback.response_revision_id"
    )

    @property
    def feedback(self) -> str | None:
        return self.feedback_rows[0].rating if self.feedback_rows else None

    message: Mapped[Message] = relationship(
        back_populates="response_revisions",
        foreign_keys=[message_id],
    )
    parts: Mapped[list[ResponseRevisionPart]] = relationship(
        back_populates="revision",
        cascade="all, delete-orphan",
        order_by="ResponseRevisionPart.position",
    )


class ResponseRevisionPart(TimestampMixin, Base):
    __tablename__ = "response_revision_parts"
    __table_args__ = (
        UniqueConstraint(
            "response_revision_id",
            "position",
            name="uq_response_revision_part_position",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("revpart"))
    response_revision_id: Mapped[str] = mapped_column(
        ForeignKey("response_revisions.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32), default=PartType.TEXT.value)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    revision: Mapped[ResponseRevision] = relationship(back_populates="parts")
    artifact: Mapped[Artifact | None] = relationship()


class TurnCreationClaim(Base):
    """Short-lived database lease that deduplicates turn planning across orchestrators."""

    __tablename__ = "turn_creation_claims"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_turn_creation_claim_chat_id_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("claim"))
    chat_id: Mapped[str] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_token: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(24), default=ArtifactKind.OTHER.value)
    media_type: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(Integer)
    relative_path: Mapped[str] = mapped_column(Text, unique=True)
    original_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Pins against automatic cleanup only; explicit deletion always wins.
    favorite: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class ArtifactLibraryEntry(TimestampMixin, Base):
    """Durable user-visible membership, separate from content-addressed bytes."""

    __tablename__ = "artifact_library_entries"
    __table_args__ = (
        UniqueConstraint("artifact_id", name="uq_artifact_library_entry_artifact"),
        CheckConstraint(
            "length(display_name) BETWEEN 1 AND 500 AND instr(display_name, char(0)) = 0",
            name="ck_library_entry_display_name",
        ),
        CheckConstraint("favorite IN (0, 1)", name="ck_library_entry_favorite_boolean"),
        CheckConstraint("state IN ('visible', 'trashed')", name="ck_library_entry_state"),
        CheckConstraint("version > 0", name="ck_library_entry_version_positive"),
        CheckConstraint(
            "(state = 'visible' AND deleted_at IS NULL AND recovery_id IS NULL) OR "
            "(state = 'trashed' AND deleted_at IS NOT NULL AND recovery_id IS NOT NULL)",
            name="ck_library_entry_recovery_consistent",
        ),
        Index("ix_library_entry_state_created", "state", "created_at", "id"),
        Index("ix_library_entry_favorite_created", "favorite", "created_at", "id"),
        Index(
            "ux_library_entry_recovery_id",
            "recovery_id",
            unique=True,
            sqlite_where=text("recovery_id IS NOT NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"))
    display_name: Mapped[str] = mapped_column(String(500))
    favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    state: Mapped[str] = mapped_column(String(16), default="visible")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recovery_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class MediaCollection(TimestampMixin, Base):
    """A durable manual Media Library collection; smart queries are separate."""

    __tablename__ = "media_collections"
    __table_args__ = (
        CheckConstraint("kind = 'manual'", name="ck_media_collection_kind"),
        CheckConstraint(
            "length(name) BETWEEN 1 AND 200 AND name = trim(name) AND name NOT GLOB '*[^ -~]*'",
            name="ck_media_collection_name",
        ),
        CheckConstraint(
            "length(description) <= 2000 AND description NOT GLOB '*[^ -~]*'",
            name="ck_media_collection_description",
        ),
        CheckConstraint("version > 0", name="ck_media_collection_version_positive"),
    )

    id: Mapped[str] = mapped_column(String(43), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), default="manual")
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)


class MediaCollectionMembership(Base):
    """One ordered strong reference from a manual collection to a library entry."""

    __tablename__ = "media_collection_memberships"
    __table_args__ = (
        UniqueConstraint(
            "collection_id", "position", name="uq_media_collection_membership_position"
        ),
        CheckConstraint("position >= 0", name="ck_media_collection_membership_position"),
        CheckConstraint(
            "note IS NULL OR (length(note) BETWEEN 1 AND 1000 AND note = trim(note) "
            "AND note NOT GLOB '*[^ -~]*')",
            name="ck_media_collection_membership_note",
        ),
    )

    collection_id: Mapped[str] = mapped_column(
        ForeignKey("media_collections.id", ondelete="CASCADE"), primary_key=True
    )
    entry_id: Mapped[str] = mapped_column(
        ForeignKey("artifact_library_entries.id", ondelete="RESTRICT"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MediaTag(TimestampMixin, Base):
    """A normalized explicit Media Library tag."""

    __tablename__ = "media_tags"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_media_tag_slug"),
        CheckConstraint(
            "length(slug) BETWEEN 1 AND 80 AND slug NOT GLOB '*[^a-z0-9-]*' "
            "AND slug NOT LIKE '-%' AND slug NOT LIKE '%-' AND slug NOT LIKE '%--%'",
            name="ck_media_tag_slug",
        ),
        CheckConstraint(
            "length(label) BETWEEN 1 AND 200 AND label = trim(label) AND label NOT GLOB '*[^ -~]*'",
            name="ck_media_tag_label",
        ),
        CheckConstraint(
            "color IS NULL OR (length(color) = 7 AND substr(color, 1, 1) = '#' "
            "AND substr(color, 2) NOT GLOB '*[^0-9a-f]*')",
            name="ck_media_tag_color",
        ),
        CheckConstraint("version > 0", name="ck_media_tag_version_positive"),
    )

    id: Mapped[str] = mapped_column(String(41), primary_key=True)
    slug: Mapped[str] = mapped_column(String(80))
    label: Mapped[str] = mapped_column(String(200))
    color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class MediaTagAssignment(Base):
    """One explicit strong tag reference to a Media Library entry."""

    __tablename__ = "media_tag_assignments"

    tag_id: Mapped[str] = mapped_column(
        ForeignKey("media_tags.id", ondelete="CASCADE"), primary_key=True
    )
    entry_id: Mapped[str] = mapped_column(
        ForeignKey("artifact_library_entries.id", ondelete="RESTRICT"), primary_key=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


for _statement in CREATE_TRIGGER_SQL:
    event.listen(
        Base.metadata,
        "after_create",
        _install_sqlite_trigger(_statement),
    )
for _statement in DROP_TRIGGER_SQL:
    event.listen(
        Base.metadata,
        "before_drop",
        DDL(_statement).execute_if(dialect="sqlite"),  # type: ignore[no-untyped-call]
    )
for _statement in CREATE_MEDIA_ORGANIZATION_TRIGGER_SQL:
    event.listen(
        Base.metadata,
        "after_create",
        _install_sqlite_trigger(_statement),
    )
for _statement in DROP_MEDIA_ORGANIZATION_TRIGGER_SQL:
    event.listen(
        Base.metadata,
        "before_drop",
        DDL(_statement).execute_if(dialect="sqlite"),  # type: ignore[no-untyped-call]
    )
for _statement in CREATE_REFERENCE_REVIEW_TRIGGER_SQL:
    event.listen(
        Base.metadata,
        "after_create",
        _install_sqlite_trigger(_statement),
    )
for _statement in DROP_REFERENCE_REVIEW_TRIGGER_SQL:
    event.listen(
        Base.metadata,
        "before_drop",
        DDL(_statement).execute_if(dialect="sqlite"),  # type: ignore[no-untyped-call]
    )


class ModelSource(TimestampMixin, Base):
    __tablename__ = "model_sources"
    __table_args__ = (
        UniqueConstraint("provider", "remote_id", "revision", name="uq_model_source"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("source"))
    provider: Mapped[str] = mapped_column(String(32), default="huggingface")
    remote_id: Mapped[str] = mapped_column(String(500))
    revision: Mapped[str] = mapped_column(String(200), default="main")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class InstallPlan(TimestampMixin, Base):
    __tablename__ = "install_plans"
    __table_args__ = (
        UniqueConstraint("plan_hash", name="uq_install_plan_hash"),
        Index("ix_install_plan_source", "provider", "remote_id", "revision", "role"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("plan"))
    provider: Mapped[str] = mapped_column(String(32), default="huggingface")
    remote_id: Mapped[str] = mapped_column(String(500))
    revision: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(16))
    engine: Mapped[str] = mapped_column(String(32))
    architecture: Mapped[str | None] = mapped_column(String(200), nullable=True)
    family: Mapped[str | None] = mapped_column(String(100), nullable=True)
    plan_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    resolver_version: Mapped[str] = mapped_column(String(40))
    compatibility: Mapped[str] = mapped_column(String(40))
    artifacts_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    runtime_contract_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    activation_probe_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(24), default="planned", index=True)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class ModelInstall(TimestampMixin, Base):
    __tablename__ = "model_installs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("model"))
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_sources.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(300), index=True)
    role: Mapped[str] = mapped_column(String(16), default=ModelRole.CHAT.value)
    engine: Mapped[str] = mapped_column(String(32))
    local_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    compatibility: Mapped[str] = mapped_column(
        String(24), default=CompatibilityLevel.ADVANCED.value
    )
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class ModelAssetInstall(TimestampMixin, Base):
    __tablename__ = "model_asset_installs"

    id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: new_id("asset"),
    )
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(300), index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    family: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    local_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    use_case: Mapped[str] = mapped_column(Text, default="")
    auto_apply: Mapped[bool] = mapped_column(Boolean, default=False)
    default_model_strength: Mapped[float] = mapped_column(Float, default=1.0)
    default_clip_strength: Mapped[float] = mapped_column(Float, default=1.0)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ModelComponentManifest(TimestampMixin, Base):
    __tablename__ = "model_component_manifests"
    __table_args__ = (
        UniqueConstraint(
            "model_install_id",
            "relative_path",
            name="uq_model_component_path",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: new_id("component"),
    )
    model_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"),
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(40))
    relative_path: Mapped[str] = mapped_column(Text)
    target_folder: Mapped[str] = mapped_column(String(80))
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    required: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelCapabilityEvidence(TimestampMixin, Base):
    __tablename__ = "model_capability_evidence"
    __table_args__ = (
        UniqueConstraint(
            "model_install_id",
            "evidence_key",
            name="uq_model_capability_evidence_install_key",
        ),
        Index("ix_model_capability_evidence_install_result", "model_install_id", "result"),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: new_id("evidence"),
    )
    model_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"),
        index=True,
    )
    evidence_key: Mapped[str] = mapped_column(String(64), index=True)
    result: Mapped[str] = mapped_column(String(24))
    component_hashes_json: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    runtime_build: Mapped[str] = mapped_column(String(200))
    adapter_contract_version: Mapped[int] = mapped_column(Integer)
    launch_contract_version: Mapped[str] = mapped_column(String(40))
    workflow_contract_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    hardware_class: Mapped[str] = mapped_column(String(200))
    # What the proving machine offered, so a proof survives a driver update or a
    # PATH change. Null on rows written before envelopes existed, which fall back
    # to comparing `hardware_class` for equality.
    hardware_envelope_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    probe_version: Mapped[str] = mapped_column(String(40))
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    probed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SetupVerification(TimestampMixin, Base):
    __tablename__ = "setup_verifications"
    __table_args__ = (
        UniqueConstraint("evidence_key", name="uq_setup_verification_evidence_key"),
        Index("ix_setup_verifications_role_state", "role", "state"),
    )

    id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: new_id("verify"),
    )
    role: Mapped[str] = mapped_column(String(16))
    evidence_key: Mapped[str] = mapped_column(String(64), unique=True)
    state: Mapped[str] = mapped_column(String(24), default="queued")
    model_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"),
    )
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("model_profiles.id", ondelete="CASCADE"),
    )
    workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="CASCADE"),
        nullable=True,
    )
    chat_id: Mapped[str | None] = mapped_column(String(40), nullable=True, unique=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    input_artifact_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelProfile(TimestampMixin, Base):
    __tablename__ = "model_profiles"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("profile"))
    model_install_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_installs.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    use_case: Mapped[str] = mapped_column(Text, default="")
    role: Mapped[str] = mapped_column(String(16), default=ModelRole.CHAT.value)
    engine: Mapped[str] = mapped_column(String(32))
    load_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    request_settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class GenerationPreset(TimestampMixin, Base):
    __tablename__ = "generation_presets"
    __table_args__ = (UniqueConstraint("role", "name", name="uq_preset_role_name"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("preset"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    role: Mapped[str] = mapped_column(String(16), default=ModelRole.CHAT.value, index=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class EditTemplate(TimestampMixin, Base):
    """A one-click edit: a named instruction scaffold over an edit workflow.

    Templates are data, not machinery - applying one composes an ordinary
    image turn from the scaffold and settings, so verification, retries, and
    revision cycling behave exactly as they do for a hand-written edit.
    """

    __tablename__ = "edit_templates"
    __table_args__ = (UniqueConstraint("name", name="uq_edit_template_name"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("edittpl"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # The complete edit instruction; "{subject}" marks where an optional
    # user addition is spliced in. Plain text otherwise.
    instruction: Mapped[str] = mapped_column(Text)
    operation: Mapped[str] = mapped_column(String(32), default="image_to_image")
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trigger_words_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    content_rating: Mapped[str] = mapped_column(String(16), default="general", index=True)
    # What produced the result this recipe was saved from. Nullable because a
    # template saved before recipes recorded none of it, and guessing a
    # binding would claim knowledge the record does not have.
    workflow_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model_profile_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # "none", "selection", or "inverse": whether the recipe expects a mask,
    # which decides whether applying it can be one click at all.
    mask_mode: Mapped[str] = mapped_column(String(16), default="none")
    # Seeded templates ship with the app and may be refreshed on upgrade;
    # user-saved ones never are.
    builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class WorkflowFamily(TimestampMixin, Base):
    __tablename__ = "workflow_families"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: new_id("wffamily")
    )
    name: Mapped[str] = mapped_column(String(240), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    use_case: Mapped[str] = mapped_column(Text, default="")
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    definitions: Mapped[list[WorkflowDefinition]] = relationship(
        back_populates="family", passive_deletes=True
    )
    preferences: Mapped[list[WorkflowPreference]] = relationship(
        back_populates="family",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WorkflowDefinition(TimestampMixin, Base):
    __tablename__ = "workflow_definitions"
    __table_args__ = (
        UniqueConstraint(
            "family_id",
            "variant_key",
            name="uq_workflow_definition_family_variant",
        ),
        CheckConstraint(
            "family_id IS NULL OR variant_key IS NOT NULL",
            name="ck_workflow_definition_family_variant",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: new_id("workflow")
    )
    family_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_families.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    variant_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    name: Mapped[str] = mapped_column(String(240), index=True)
    operation: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text, default="")
    current_revision_id: Mapped[str | None] = mapped_column(String(40), nullable=True)

    family: Mapped[WorkflowFamily | None] = relationship(
        back_populates="definitions", foreign_keys=[family_id]
    )
    revisions: Mapped[list[WorkflowRevision]] = relationship(
        back_populates="definition",
        cascade="all, delete-orphan",
        foreign_keys="WorkflowRevision.workflow_id",
    )


class WorkflowPreference(TimestampMixin, Base):
    __tablename__ = "workflow_preferences"
    __table_args__ = (
        UniqueConstraint(
            "workflow_family_id",
            "selector_capability",
            name="uq_workflow_preference_family_selector",
        ),
        CheckConstraint(
            "length(trim(selector_capability)) > 0",
            name="ck_workflow_preference_selector_nonempty",
        ),
        CheckConstraint(
            "NOT is_default OR enabled",
            name="ck_workflow_preference_default_enabled",
        ),
        Index(
            "ix_workflow_preferences_selector_order",
            "selector_capability",
            "enabled",
            "sort_order",
        ),
        Index(
            "uq_workflow_preferences_default_selector",
            "selector_capability",
            unique=True,
            sqlite_where=text("is_default = 1"),
            postgresql_where=text("is_default IS TRUE"),
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfpref"))
    workflow_family_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_families.id", ondelete="CASCADE"), index=True
    )
    selector_capability: Mapped[str] = mapped_column(String(32))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    family: Mapped[WorkflowFamily] = relationship(
        back_populates="preferences", foreign_keys=[workflow_family_id]
    )


class WorkflowProfileCompatibility(TimestampMixin, Base):
    """Local bridge from one legacy profile to its generated workflow family."""

    __tablename__ = "workflow_profile_compatibility"
    __table_args__ = (
        CheckConstraint(
            _lowercase_sha256_check("source_fingerprint_sha256"),
            name="ck_workflow_profile_compatibility_fingerprint_sha256",
        ),
    )

    model_profile_id: Mapped[str] = mapped_column(
        ForeignKey("model_profiles.id", ondelete="CASCADE"), primary_key=True
    )
    workflow_family_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_families.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    source_fingerprint_sha256: Mapped[str] = mapped_column(String(64))

    model_profile: Mapped[ModelProfile] = relationship(foreign_keys=[model_profile_id])
    workflow_family: Mapped[WorkflowFamily] = relationship(foreign_keys=[workflow_family_id])


class ChatWorkflowSelection(TimestampMixin, Base):
    """Workflow-first chat choice; absence deliberately preserves legacy semantics."""

    __tablename__ = "chat_workflow_selections"
    __table_args__ = (
        UniqueConstraint(
            "chat_id",
            "selector_capability",
            name="uq_chat_workflow_selection_capability",
        ),
        CheckConstraint(
            "length(trim(selector_capability)) > 0",
            name="ck_chat_workflow_selection_capability_nonempty",
        ),
        CheckConstraint(
            "(mode = 'automatic' AND workflow_family_id IS NULL) OR "
            "(mode = 'family' AND workflow_family_id IS NOT NULL)",
            name="ck_chat_workflow_selection_mode_target",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfsel"))
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    selector_capability: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(16))
    workflow_family_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_families.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    chat: Mapped[Chat] = relationship(foreign_keys=[chat_id])
    workflow_family: Mapped[WorkflowFamily | None] = relationship(foreign_keys=[workflow_family_id])


class ProjectWorkflowSelection(TimestampMixin, Base):
    """Workflow-first project choice with exact-revision pin support."""

    __tablename__ = "project_workflow_selections"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "selector_capability",
            name="uq_project_workflow_selection_capability",
        ),
        CheckConstraint(
            "length(trim(selector_capability)) > 0",
            name="ck_project_workflow_selection_capability_nonempty",
        ),
        CheckConstraint(
            "(mode = 'automatic' AND workflow_family_id IS NULL "
            "AND workflow_revision_id IS NULL) OR "
            "(mode = 'family' AND workflow_family_id IS NOT NULL "
            "AND workflow_revision_id IS NULL) OR "
            "(mode = 'revision' AND workflow_family_id IS NULL "
            "AND workflow_revision_id IS NOT NULL)",
            name="ck_project_workflow_selection_mode_target",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfsel"))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    selector_capability: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(16))
    workflow_family_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_families.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    workflow_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    project: Mapped[Project] = relationship(foreign_keys=[project_id])
    workflow_family: Mapped[WorkflowFamily | None] = relationship(foreign_keys=[workflow_family_id])
    workflow_revision: Mapped[WorkflowRevision | None] = relationship(
        foreign_keys=[workflow_revision_id]
    )


class WorkflowRevision(TimestampMixin, Base):
    __tablename__ = "workflow_revisions"
    __table_args__ = (
        UniqueConstraint("workflow_id", "version", name="uq_workflow_version"),
        CheckConstraint(
            "dependency_contract_sha256 IS NULL OR ("
            + _lowercase_sha256_check("dependency_contract_sha256")
            + ")",
            name="ck_workflow_revision_dependency_contract_sha256",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfrev"))
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_definitions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    engine: Mapped[str] = mapped_column(String(32), default="comfyui")
    engine_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ui_graph_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    api_graph_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_schema_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    capabilities_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    dependencies_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dependency_contract_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Identifies what this revision executes, so capability evidence survives a
    # compiler change that does not alter the compiled output.
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    trusted: Mapped[bool] = mapped_column(Boolean, default=False)

    definition: Mapped[WorkflowDefinition] = relationship(
        back_populates="revisions", foreign_keys=[workflow_id]
    )
    dependency_slots: Mapped[list[WorkflowDependencySlot]] = relationship(
        back_populates="revision",
        passive_deletes="all",
        order_by="WorkflowDependencySlot.ordinal",
    )
    activations: Mapped[list[WorkflowActivation]] = relationship(
        back_populates="revision",
        passive_deletes="all",
    )


class CustomNodeInstall(TimestampMixin, Base):
    __tablename__ = "custom_node_installs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("node"))
    name: Mapped[str] = mapped_column(String(240), index=True)
    source_url: Mapped[str] = mapped_column(String(1000), unique=True)
    revision: Mapped[str] = mapped_column(String(40))
    previous_revision: Mapped[str | None] = mapped_column(String(40), nullable=True)
    installed_path: Mapped[str] = mapped_column(Text)
    tree_hash: Mapped[str] = mapped_column(String(64))
    trusted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    security_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ComfyRegistryInstall(TimestampMixin, Base):
    __tablename__ = "comfy_registry_installs"
    __table_args__ = (
        UniqueConstraint(
            "package_id",
            "package_version",
            name="uq_comfy_registry_install_package_version",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: new_id("registry")
    )
    package_id: Mapped[str] = mapped_column(String(100), index=True)
    package_version: Mapped[str] = mapped_column(String(100))
    registry_record_id: Mapped[str] = mapped_column(String(1000), unique=True)
    repository_url: Mapped[str] = mapped_column(String(1000))
    download_url: Mapped[str] = mapped_column(String(1000))
    archive_sha256: Mapped[str] = mapped_column(String(64))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    installed_path: Mapped[str] = mapped_column(Text, unique=True)
    node_types_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    pip_dependencies_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    review_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    wheel_closure_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    wheel_environment_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    wheel_environment_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    trusted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class WorkflowDependencySlot(TimestampMixin, Base):
    __tablename__ = "workflow_dependency_slots"
    __table_args__ = (
        UniqueConstraint(
            "workflow_revision_id",
            "name",
            name="uq_workflow_dependency_slot_revision_name",
        ),
        UniqueConstraint(
            "workflow_revision_id",
            "ordinal",
            name="uq_workflow_dependency_slot_revision_ordinal",
        ),
        UniqueConstraint(
            "id",
            "workflow_revision_id",
            name="uq_workflow_dependency_slot_id_revision",
        ),
        CheckConstraint(
            "length(trim(name)) > 0",
            name="ck_workflow_dependency_slot_name_nonempty",
        ),
        CheckConstraint(
            "resource_kind IN ('model_profile', 'model_install', 'model_asset', "
            "'custom_node', 'registry_package', 'runtime')",
            name="ck_workflow_dependency_slot_resource_kind",
        ),
        CheckConstraint(
            "satisfaction IN ('all_of', 'any_of')",
            name="ck_workflow_dependency_slot_satisfaction",
        ),
        CheckConstraint(
            "required IN (false, true)",
            name="ck_workflow_dependency_slot_required",
        ),
        CheckConstraint("ordinal >= 0", name="ck_workflow_dependency_slot_ordinal"),
        CheckConstraint(
            _lowercase_sha256_check("contract_sha256"),
            name="ck_workflow_dependency_slot_contract_sha256",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfslot"))
    workflow_revision_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    resource_kind: Mapped[str] = mapped_column(String(32))
    required: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    satisfaction: Mapped[str] = mapped_column(
        String(16), default="all_of", server_default=text("'all_of'")
    )
    requirements_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, server_default=text("'[]'")
    )
    contract_sha256: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int] = mapped_column(Integer)

    revision: Mapped[WorkflowRevision] = relationship(
        back_populates="dependency_slots", foreign_keys=[workflow_revision_id]
    )
    bindings: Mapped[list[WorkflowDependencyBinding]] = relationship(
        back_populates="slot",
        foreign_keys=(
            "[WorkflowDependencyBinding.workflow_dependency_slot_id, "
            "WorkflowDependencyBinding.workflow_revision_id]"
        ),
        viewonly=True,
    )


class WorkflowActivation(TimestampMixin, Base):
    __tablename__ = "workflow_activations"
    __table_args__ = (
        UniqueConstraint(
            "workflow_revision_id",
            "binding_sha256",
            name="uq_workflow_activation_revision_binding",
        ),
        UniqueConstraint(
            "id",
            "workflow_revision_id",
            name="uq_workflow_activation_id_revision",
        ),
        CheckConstraint(
            "state IN ('ready', 'stale', 'disabled')",
            name="ck_workflow_activation_state",
        ),
        CheckConstraint(
            _lowercase_sha256_check("dependency_contract_sha256"),
            name="ck_workflow_activation_contract_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("binding_sha256"),
            name="ck_workflow_activation_binding_sha256",
        ),
        CheckConstraint(
            "NOT is_active OR (state = 'ready' AND invalidated_at IS NULL)",
            name="ck_workflow_activation_active_ready",
        ),
        CheckConstraint(
            "is_active IN (false, true)",
            name="ck_workflow_activation_is_active",
        ),
        Index(
            "uq_workflow_activation_active_revision",
            "workflow_revision_id",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active IS TRUE"),
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfact"))
    workflow_revision_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="CASCADE"), index=True
    )
    resolver_version: Mapped[str] = mapped_column(String(40))
    dependency_contract_sha256: Mapped[str] = mapped_column(String(64))
    binding_sha256: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(
        String(16), default="ready", server_default=text("'ready'"), index=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), index=True
    )
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidation_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default=text("'{}'")
    )

    revision: Mapped[WorkflowRevision] = relationship(
        back_populates="activations", foreign_keys=[workflow_revision_id]
    )
    bindings: Mapped[list[WorkflowDependencyBinding]] = relationship(
        back_populates="activation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys=(
            "[WorkflowDependencyBinding.workflow_activation_id, "
            "WorkflowDependencyBinding.workflow_revision_id]"
        ),
    )


class WorkflowDependencyBinding(TimestampMixin, Base):
    __tablename__ = "workflow_dependency_bindings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workflow_activation_id", "workflow_revision_id"],
            ["workflow_activations.id", "workflow_activations.workflow_revision_id"],
            name="fk_workflow_dependency_binding_activation_revision",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workflow_dependency_slot_id", "workflow_revision_id"],
            ["workflow_dependency_slots.id", "workflow_dependency_slots.workflow_revision_id"],
            name="fk_workflow_dependency_binding_slot_revision",
        ),
        UniqueConstraint(
            "workflow_activation_id",
            "workflow_dependency_slot_id",
            "requirement_key",
            name="uq_workflow_dependency_binding_assignment",
        ),
        CheckConstraint(
            "length(trim(requirement_key)) > 0",
            name="ck_workflow_dependency_binding_requirement_nonempty",
        ),
        CheckConstraint(
            _lowercase_sha256_check("resource_identity_sha256"),
            name="ck_workflow_dependency_binding_identity_sha256",
        ),
        CheckConstraint(
            "(CASE WHEN model_profile_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN model_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN model_asset_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN custom_node_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN comfy_registry_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN runtime_key IS NOT NULL THEN 1 ELSE 0 END) = 1",
            name="ck_workflow_dependency_binding_one_locator",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfbind"))
    workflow_revision_id: Mapped[str] = mapped_column(String(40), index=True)
    workflow_activation_id: Mapped[str] = mapped_column(String(40), index=True)
    workflow_dependency_slot_id: Mapped[str] = mapped_column(String(40), index=True)
    requirement_key: Mapped[str] = mapped_column(String(100))
    model_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profiles.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    model_install_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_installs.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    model_asset_install_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_asset_installs.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    custom_node_install_id: Mapped[str | None] = mapped_column(
        ForeignKey("custom_node_installs.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    comfy_registry_install_id: Mapped[str | None] = mapped_column(
        ForeignKey("comfy_registry_installs.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    runtime_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mount_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default=text("'{}'")
    )
    resource_identity_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default=text("'{}'")
    )
    resource_identity_sha256: Mapped[str] = mapped_column(String(64))

    activation: Mapped[WorkflowActivation] = relationship(
        back_populates="bindings",
        foreign_keys=[workflow_activation_id, workflow_revision_id],
    )
    slot: Mapped[WorkflowDependencySlot] = relationship(
        back_populates="bindings",
        foreign_keys=[workflow_dependency_slot_id, workflow_revision_id],
        viewonly=True,
    )
    model_profile: Mapped[ModelProfile | None] = relationship(foreign_keys=[model_profile_id])
    model_install: Mapped[ModelInstall | None] = relationship(foreign_keys=[model_install_id])
    model_asset_install: Mapped[ModelAssetInstall | None] = relationship(
        foreign_keys=[model_asset_install_id]
    )
    custom_node_install: Mapped[CustomNodeInstall | None] = relationship(
        foreign_keys=[custom_node_install_id]
    )
    comfy_registry_install: Mapped[ComfyRegistryInstall | None] = relationship(
        foreign_keys=[comfy_registry_install_id]
    )


class WorkflowTrustAttestation(TimestampMixin, Base):
    """What this machine verified about one revision, and what it verified it against.

    Deliberately not a boolean on the revision. Derived trust proves a revision
    came from a template this build compiled; attestation proves something
    weaker - that at one moment, on this machine, every node type the graph
    executes resolved to something installed and reviewed, and every asset it
    names was present. Recording the evidence rather than a verdict is what
    keeps the two from being confused, because conflating them would turn
    import into a way to run an unreviewed package.

    The identity columns are what make the claim checkable later. An attestation
    is about one artifact on one runtime carrying one whitelist, so if any of
    those move, the answer is stale rather than wrong - and staleness is
    computed when read, never stored, because the thing that invalidates it
    happens elsewhere and would not come back to update a row.
    """

    __tablename__ = "workflow_trust_attestations"
    __table_args__ = (
        UniqueConstraint("workflow_revision_id", name="uq_workflow_trust_attestation_revision"),
        CheckConstraint(
            _lowercase_sha256_check("artifact_sha256"),
            name="ck_workflow_trust_attestation_artifact_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("node_inventory_sha256"),
            name="ck_workflow_trust_attestation_node_inventory_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("whitelist_sha256"),
            name="ck_workflow_trust_attestation_whitelist_sha256",
        ),
        CheckConstraint(
            "runtime_contract_sha256 IS NULL OR ("
            + _lowercase_sha256_check("runtime_contract_sha256")
            + ")",
            name="ck_workflow_trust_attestation_runtime_contract_sha256",
        ),
        CheckConstraint(
            "launch_scope_sha256 IS NULL OR ("
            + _lowercase_sha256_check("launch_scope_sha256")
            + ")",
            name="ck_workflow_trust_attestation_launch_scope_sha256",
        ),
    )

    # "wfattest_" plus 32 hex is 41 characters, so String(40) would truncate.
    id: Mapped[str] = mapped_column(
        String(48), primary_key=True, default=lambda: new_id("wfattest")
    )
    workflow_revision_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="CASCADE"), index=True
    )
    artifact_sha256: Mapped[str] = mapped_column(String(64), index=True)
    runtime_contract_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    runtime_managed: Mapped[bool] = mapped_column(Boolean, default=False)
    node_inventory_sha256: Mapped[str] = mapped_column(String(64))
    whitelist_sha256: Mapped[str] = mapped_column(String(64))
    launch_scope_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    required_node_types_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    declared_dependencies_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    resolution_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    attested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AdapterPromptGrammar(TimestampMixin, Base):
    """How one installed adapter expects to be prompted, and what file that was true of.

    An adapter can be trained to expect a particular prompt shape, and one
    prompted the wrong way does not fail - it produces confident output in the
    wrong form. Recording the shape is what lets a prompt be written correctly,
    and what lets the application say a request is outside what the loaded stack
    can express instead of rendering something adjacent.

    Keyed to the asset's content digest and not only to the install row. An
    install row is mutable and a file can be replaced underneath it, so binding
    the grammar to the row alone would let a new file silently inherit the old
    file's description of itself. `asset_sha256` is what the grammar was written
    about; if the install no longer hashes to it, the grammar is stale rather
    than wrong, and staleness is computed on read for the same reason it is for
    an attestation.

    `grammar_json` holds the normalized form only. The document it came from is
    third-party text on its way to a model that rewrites what the user wrote,
    which makes it an instruction channel, so the raw file is quarantined
    outside this row and only its digest is kept. `examples_reviewed` gates the
    one part of a normalized grammar that is still free text.
    """

    __tablename__ = "adapter_prompt_grammars"
    __table_args__ = (
        # One grammar per install. A second would raise the question of which one
        # a rewriter believed, and the answer has to be that there is only one.
        UniqueConstraint("model_asset_install_id", name="uq_adapter_prompt_grammar_install"),
        CheckConstraint(
            _lowercase_sha256_check("asset_sha256"),
            name="ck_adapter_prompt_grammar_asset_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("source_sha256"),
            name="ck_adapter_prompt_grammar_source_sha256",
        ),
    )

    # "promptgrammar_" plus 32 hex is 46 characters, so String(40) would truncate.
    id: Mapped[str] = mapped_column(
        String(48), primary_key=True, default=lambda: new_id("promptgrammar")
    )
    model_asset_install_id: Mapped[str] = mapped_column(
        ForeignKey("model_asset_installs.id", ondelete="CASCADE"), index=True
    )
    asset_sha256: Mapped[str] = mapped_column(String(64), index=True)
    source_identity: Mapped[str] = mapped_column(Text)
    source_sha256: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    grammar_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # The digest of what a rewriter would actually act on, including the two
    # overlays that change emitted text. Storing the grammar alone would leave
    # the overlays unauthenticated, so widening approval or verification would
    # change the output without changing anything review is bound to.
    grammar_sha256: Mapped[str] = mapped_column(String(64), default="")
    approved_prose_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Reviewed evidence, not observations. Nothing generated may write here; a
    # future observations table accumulates separately and can only promote a
    # value through an explicit review.
    verified_values_json: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict)
    # What the fit was judged against. A compiler change must make the evidence
    # stale rather than overflow at turn time.
    compiler_version: Mapped[str] = mapped_column(String(64), default="")
    compiler_ceiling: Mapped[int] = mapped_column(Integer, default=0)
    fits: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ReferenceSubject(TimestampMixin, Base):
    """A subject the user has taught this application about, addressable by name.

    The name and the mention are deliberately separate columns. A display name is
    whatever a person actually calls someone; a mention slug is an addressing
    token with exactly one canonical form, so that two subjects can never occupy
    what a reader sees as the same `@name`.

    Nothing here is a quality signal. `favorite` is organisation only - it says a
    person wanted this near the top of a list, not that its images are good - and
    reading it as ranking input would quietly turn a bookmark into a preference
    the user never expressed.
    """

    __tablename__ = "reference_subjects"
    __table_args__ = (
        # The mention is the addressing key, so uniqueness is the whole point.
        # Slugs are canonicalised before they arrive, which is what makes a
        # plain unique index sufficient rather than needing a case-folded one.
        UniqueConstraint("mention_slug", name="uq_reference_subject_mention_slug"),
        CheckConstraint("length(trim(name)) > 0", name="ck_reference_subject_name_present"),
        CheckConstraint("length(trim(mention_slug)) > 0", name="ck_reference_subject_slug_present"),
    )

    # "refsubject_" plus 32 hex is 43 characters, so String(40) would truncate.
    id: Mapped[str] = mapped_column(
        String(48), primary_key=True, default=lambda: new_id("refsubject")
    )
    name: Mapped[str] = mapped_column(String(120))
    mention_slug: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Cleared rather than cascading: losing a cover image must not lose the
    # subject, because the images are replaceable and the identity is not.
    cover_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True
    )
    favorite: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # Archive is the normal way to remove a subject. Permanent deletion has to
    # be impact-aware, because past runs recorded what they used and that record
    # is history rather than a pointer to a current row.
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    assets: Mapped[list[ReferenceAsset]] = relationship(
        back_populates="subject",
        cascade="all, delete-orphan",
        order_by="ReferenceAsset.sort_order",
    )


class ReferenceAsset(TimestampMixin, Base):
    """One image belonging to a subject, and what it is there to show.

    This row expresses membership and role. It does not own bytes: the image
    lives in the content-addressed artifact store, which already counts
    references, so the same photograph used by two subjects is stored once.
    Creating a second media filesystem here would mean two things to keep
    consistent and two things to leak.
    """

    __tablename__ = "reference_assets"
    __table_args__ = (
        # The same image twice under one subject is a duplicate, not a second
        # view of it. Detecting that here costs nothing and stops a set being
        # silently weighted toward whichever picture was added twice.
        UniqueConstraint(
            "reference_subject_id", "artifact_id", name="uq_reference_asset_membership"
        ),
        # Named here rather than left to `index=True`, because these are the
        # names the migration created and therefore the names every existing
        # database has. Letting the model derive its own would mean a schema
        # built by migration and one built from this metadata differ by index
        # name, and autogenerate would propose renaming them on real data.
        Index("ix_reference_assets_subject", "reference_subject_id"),
        Index("ix_reference_assets_artifact", "artifact_id"),
        CheckConstraint("review_version > 0", name="ck_reference_asset_review_version"),
    )

    id: Mapped[str] = mapped_column(
        String(48), primary_key=True, default=lambda: new_id("refasset")
    )
    reference_subject_id: Mapped[str] = mapped_column(
        ForeignKey("reference_subjects.id", ondelete="CASCADE")
    )
    # Restricted, not cascading: an artifact still used by a Reference must not
    # be removable out from under it by an unrelated cleanup.
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"))
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    purpose: Mapped[str] = mapped_column(String(40), default="other")
    view_label: Mapped[str | None] = mapped_column(String(60), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    # Starts unchecked, which is not the same as usable. An image nobody has
    # looked at must not let an unreviewed set claim a reviewed set's fidelity.
    validation_state: Mapped[str] = mapped_column(String(30), default="unchecked")
    validation_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Version 1 means unchecked. The first and only decision advances it to 2.
    review_version: Mapped[int] = mapped_column(Integer, default=1)

    subject: Mapped[ReferenceSubject] = relationship(back_populates="assets")


class ReferenceAssetReviewEvent(Base):
    """One immutable human decision over exact retained artifact bytes."""

    __tablename__ = "reference_asset_review_events"
    __table_args__ = (
        UniqueConstraint(
            "reference_asset_id", "result_version", name="uq_reference_asset_review_version"
        ),
        CheckConstraint("expected_version > 0", name="ck_reference_review_expected_version"),
        CheckConstraint(
            "result_version = expected_version + 1", name="ck_reference_review_result_version"
        ),
        CheckConstraint("width > 0 AND height > 0", name="ck_reference_review_dimensions"),
        CheckConstraint(
            _lowercase_sha256_check("artifact_sha256"), name="ck_reference_review_artifact_sha256"
        ),
        CheckConstraint(
            _lowercase_sha256_check("decision_sha256"), name="ck_reference_review_decision_sha256"
        ),
    )

    id: Mapped[str] = mapped_column(String(88), primary_key=True)
    # A snapshot, not a foreign key: detach must not erase the review history.
    reference_asset_id: Mapped[str] = mapped_column(String(48), index=True)
    artifact_id: Mapped[str] = mapped_column(String(80))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    reviewer_kind: Mapped[str] = mapped_column(String(32))
    expected_state: Mapped[str] = mapped_column(String(30))
    expected_version: Mapped[int] = mapped_column(Integer)
    result_version: Mapped[int] = mapped_column(Integer)
    decision: Mapped[str] = mapped_column(String(30))
    reasons_json: Mapped[list[str]] = mapped_column(JSON)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    decision_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MessageReference(TimestampMixin, Base):
    """What one turn referred to, recorded as it stood at the time.

    This is history rather than a link. A subject can be renamed, archived or
    deleted long after a turn used it, and the question this row exists to
    answer - why does that picture look like that - has to keep its answer
    afterwards. So the identifying fields are snapshots and not a join:
    `reference_subject_id` is stored without a foreign key, and the mention,
    name and kind are copied in beside it. A live foreign key would make the
    record evaporate at exactly the moment someone asked.

    That also means a later rename cannot rewrite what a past turn recorded.
    The row is written once and never revised.

    The artifact ids are copied for the same reason and are likewise not foreign
    keys. A cleanup that removes unreferenced bytes must not be able to erase
    the record that those bytes were once used.
    """

    __tablename__ = "message_references"
    __table_args__ = (
        # One entry per slot, so a retry that re-records a turn cannot quietly
        # double what that turn referred to.
        UniqueConstraint("message_id", "position", name="uq_message_reference_position"),
        CheckConstraint("position >= 0", name="ck_message_reference_position"),
        CheckConstraint(
            "length(trim(reference_subject_id)) > 0",
            name="ck_message_reference_subject_present",
        ),
        # Named for the same reason as on `reference_assets`: these are what the
        # migration created, so this metadata has to agree with it.
        Index("ix_message_references_message", "message_id"),
        Index("ix_message_references_subject", "reference_subject_id"),
    )

    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=lambda: new_id("msgref"))
    # The turn that referred to it. This one is a real foreign key: deleting a
    # user's turn is meant to remove what it produced, and a reference record
    # outliving its own message would be an orphan nobody could interpret.
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer, default=0)
    reference_subject_id: Mapped[str] = mapped_column(String(48))
    mention_slug: Mapped[str] = mapped_column(String(64))
    subject_name: Mapped[str] = mapped_column(String(120))
    subject_kind: Mapped[str] = mapped_column(String(40))
    role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    # How it came to be here. A typed mention and something inherited from the
    # surrounding context differ in how much the user actually asserted, and a
    # later question about why an image contains someone must tell them apart.
    source: Mapped[str] = mapped_column(String(40), default="mention")
    reference_asset_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    artifact_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)


class WorkflowInstallOffer(TimestampMixin, Base):
    """One reviewed, content-bound way to make a workflow locally installable."""

    __tablename__ = "workflow_install_offers"
    __table_args__ = (
        CheckConstraint(
            _lowercase_sha256_check("workflow_artifact_sha256"),
            name="ck_workflow_install_offer_artifact_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("dependency_contract_sha256"),
            name="ck_workflow_install_offer_contract_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("binding_plan_sha256"),
            name="ck_workflow_install_offer_binding_sha256",
        ),
        CheckConstraint(
            _lowercase_sha256_check("offer_sha256"),
            name="ck_workflow_install_offer_sha256",
        ),
        CheckConstraint(
            "status IN ('ready', 'queued', 'invalidated', 'completed', 'expired')",
            name="ck_workflow_install_offer_status",
        ),
        CheckConstraint("plan_count > 0", name="ck_workflow_install_offer_plan_count"),
        CheckConstraint("total_bytes > 0", name="ck_workflow_install_offer_total_bytes"),
        Index(
            "ix_workflow_install_offer_revision_status",
            "workflow_revision_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfoffer"))
    workflow_revision_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_revisions.id", ondelete="CASCADE"), index=True
    )
    workflow_artifact_sha256: Mapped[str] = mapped_column(String(64))
    dependency_contract_sha256: Mapped[str] = mapped_column(String(64))
    binding_plan_sha256: Mapped[str] = mapped_column(String(64))
    offer_sha256: Mapped[str] = mapped_column(String(64), index=True)
    selections_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    assets_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    plan_count: Mapped[int] = mapped_column(Integer)
    total_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(16), default="ready", server_default=text("'ready'"), index=True
    )
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidation_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("job"))
    kind: Mapped[str] = mapped_column(String(16), default=JobKind.CHAT.value)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    work_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    work_step_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_steps.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    phase: Mapped[str] = mapped_column(String(120), default="queued")
    progress_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    queue_resource: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    queue_group: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    queue_priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    queue_ticket: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claim_owner: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    cancellable: Mapped[bool] = mapped_column(Boolean, default=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AppSetting(TimestampMixin, Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    value_json: Mapped[Any] = mapped_column(JSON)


@event.listens_for(Session, "before_flush")
def _guard_artifact_reference_flush(
    session: Session,
    flush_context: object,
    instances: object,
) -> None:
    """Register JSON Artifact authority with every production model import."""

    from .artifact_library import guard_artifact_reference_flush

    guard_artifact_reference_flush(session, flush_context, instances)
