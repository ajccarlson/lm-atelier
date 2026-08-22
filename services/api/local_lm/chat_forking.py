"""Fork a conversation into a new chat from one message.

Distinct from edit-and-branch, which creates a sibling inside one chat's
tree: this makes a separate thread with its own head, title, and history, so
a tangent stops competing with the original conversation for the same
transcript.

What travels: the message lineage up to and including the chosen message,
copied as new rows so the fork is independently editable, plus the source
chat's routing mode, profile bindings, and generation settings. What does not
travel: runs, jobs, and work plans - the fork inherits the *conversation*,
not the execution history that produced it, and re-running is the user's
choice. Artifacts are referenced, never copied; they are content-addressed,
so both chats point at the same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .message_references import carry_message_references_if_absent
from .models import Chat, Message, MessagePart
from .workflow_compatibility import copy_chat_workflow_selections

MAX_FORK_TITLE = 240


class ForkError(ValueError):
    """Base class so callers map every refusal to one error family."""


class ForkSourceNotFound(ForkError):
    def __init__(self) -> None:
        super().__init__("the message to fork from was not found")


@dataclass(frozen=True)
class ChatFork:
    chat_id: str
    source_chat_id: str
    source_message_id: str
    copied_message_ids: list[str]


def fork_chat_from_message(session: Session, message_id: str) -> ChatFork:
    """Create a new chat carrying the history up to `message_id`."""

    source = session.get(Message, message_id)
    if not source:
        raise ForkSourceNotFound
    origin = session.get(Chat, source.chat_id)
    if not origin:
        raise ForkSourceNotFound

    lineage = _lineage_to(session, source)
    fork = Chat(
        project_id=origin.project_id,
        title=_fork_title(origin.title),
        routing_mode=origin.routing_mode,
        confirm_uncertain_media=origin.confirm_uncertain_media,
        active_chat_profile_id=origin.active_chat_profile_id,
        active_vision_profile_id=origin.active_vision_profile_id,
        active_image_profile_id=origin.active_image_profile_id,
        active_video_profile_id=origin.active_video_profile_id,
        generation_settings_json=dict(origin.generation_settings_json or {}),
        generation_preset_ids_json=dict(origin.generation_preset_ids_json or {}),
        vision_settings_json=dict(origin.vision_settings_json or {}),
    )
    session.add(fork)
    session.flush()
    copy_chat_workflow_selections(session, origin, fork)

    copied: list[str] = []
    parent_id: str | None = None
    for message in lineage:
        clone = Message(
            chat_id=fork.id,
            parent_id=parent_id,
            role=message.role,
            status=message.status,
            transcript_visible=message.transcript_visible,
        )
        session.add(clone)
        session.flush()
        for part in message.parts:
            session.add(
                MessagePart(
                    message_id=clone.id,
                    position=part.position,
                    type=part.type,
                    text=part.text,
                    # A reference, not a copy: artifacts are content-addressed
                    # and already shared between chats.
                    artifact_id=part.artifact_id,
                    metadata_json=dict(part.metadata_json or {}),
                )
            )
        # References are immutable history, just like the copied message. Use
        # their sole writer so every identity/snapshot field travels verbatim
        # and remains bound to the newly created message rather than the source.
        carry_message_references_if_absent(
            session,
            source_message_id=message.id,
            target_message_id=clone.id,
        )
        parent_id = clone.id
        copied.append(clone.id)

    fork.active_head_message_id = parent_id
    # Recorded in both directions so neither chat is a mystery later.
    fork.origin_json = {
        "forked_from_chat_id": origin.id,
        "forked_from_message_id": source.id,
        "turn_count": len(lineage),
    }
    session.flush()
    return ChatFork(
        chat_id=fork.id,
        source_chat_id=origin.id,
        source_message_id=source.id,
        copied_message_ids=copied,
    )


def _lineage_to(session: Session, source: Message) -> list[Message]:
    """Oldest to newest, ending at `source`."""

    rows: list[Message] = []
    current_id: str | None = source.id
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts))
            .where(Message.id == current_id, Message.chat_id == source.chat_id)
        )
        if not message:
            break
        rows.append(message)
        current_id = message.parent_id
    rows.reverse()
    return rows


def _fork_title(title: str) -> str:
    candidate = f"{title} (thread)"
    return candidate[:MAX_FORK_TITLE] if len(candidate) > MAX_FORK_TITLE else candidate
