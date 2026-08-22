"""Forking a thread from a message."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import local_lm.chat_forking as chat_forking_module
from local_lm.chat_forking import fork_chat_from_message
from local_lm.db import SessionLocal
from local_lm.message_references import (
    ResolvedReference,
    message_references,
    record_message_references,
)
from local_lm.models import Artifact, Chat, Message, MessageReference
from local_lm.references import MentionSource


async def wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    for _ in range(400):
        payload = (await client.get(f"/api/runs/{run_id}")).json()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return cast(dict[Any, Any], payload)
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish in time")


async def _turn(client: AsyncClient, chat_id: str, text: str) -> dict:  # type: ignore[type-arg]
    accepted = await client.post(f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "text"})
    assert accepted.status_code == 202
    payload = accepted.json()
    await wait_for_run(client, payload["run"]["id"])
    return cast(dict[Any, Any], payload)


async def test_forking_copies_the_history_up_to_the_chosen_message(
    client: AsyncClient,
) -> None:
    source = (await client.post("/api/chats", json={"title": "Original"})).json()
    assert (
        await client.patch(f"/api/chats/{source['id']}", json={"routing_mode": "image"})
    ).status_code == 200
    first = await _turn(client, source["id"], "First question")
    await _turn(client, source["id"], "Second question")

    forked = await client.post(f"/api/messages/{first['assistant_message']['id']}/fork")
    assert forked.status_code == 201
    fork = forked.json()

    assert fork["id"] != source["id"]
    assert fork["title"] == "Original (thread)"
    # Settings travel so the fork behaves like the conversation it came from.
    assert fork["routing_mode"] == "image"
    assert fork["origin_json"]["forked_from_chat_id"] == source["id"]
    assert fork["origin_json"]["forked_from_message_id"] == first["assistant_message"]["id"]

    detail = (await client.get(f"/api/chats/{fork['id']}")).json()
    texts = [
        part["text"]
        for message in detail["messages"]
        for part in message["parts"]
        if part["type"] == "text"
    ]
    assert "First question" in texts
    # History stops at the forked message: the later turn stays behind.
    assert "Second question" not in texts
    assert detail["active_head_message_id"] == detail["messages"][-1]["id"]

    # The original is untouched - forking is not moving.
    original = (await client.get(f"/api/chats/{source['id']}")).json()
    assert len(original["messages"]) == 4
    assert original["origin_json"] == {}

    with SessionLocal() as session:
        fork_message_ids = select(Message.id).where(Message.chat_id == fork["id"])
        assert (
            session.scalar(
                select(func.count())
                .select_from(MessageReference)
                .where(MessageReference.message_id.in_(fork_message_ids))
            )
            == 0
        )


async def test_the_fork_is_independently_editable(client: AsyncClient) -> None:
    source = (await client.post("/api/chats", json={"title": "Shared start"})).json()
    turn = await _turn(client, source["id"], "Common ground")
    fork = (await client.post(f"/api/messages/{turn['assistant_message']['id']}/fork")).json()

    with SessionLocal() as session:
        copied = session.scalars(select(Message.id).where(Message.chat_id == fork["id"])).all()
        original = session.scalars(select(Message.id).where(Message.chat_id == source["id"])).all()
    # New rows, not shared ones: editing the fork must not rewrite history.
    assert not set(copied) & set(original)

    await _turn(client, fork["id"], "Only in the fork")
    unchanged = (await client.get(f"/api/chats/{source['id']}")).json()
    assert len(unchanged["messages"]) == 2


async def test_forking_an_unknown_message_is_refused(client: AsyncClient) -> None:
    response = await client.post("/api/messages/msg_missing/fork")
    assert response.status_code == 404
    assert response.json()["code"] == "fork-source-not-found"


@pytest.mark.parametrize("title", ["A" * 240])
async def test_a_long_title_stays_within_its_bound(client: AsyncClient, title: str) -> None:
    source = (await client.post("/api/chats", json={"title": title})).json()
    turn = await _turn(client, source["id"], "Question")

    forked = await client.post(f"/api/messages/{turn['assistant_message']['id']}/fork")
    assert forked.status_code == 201
    with SessionLocal() as session:
        chat = session.get(Chat, forked.json()["id"])
        assert chat is not None
        assert len(chat.title) <= 240


async def test_fork_carries_exact_reference_snapshots_in_order(client: AsyncClient) -> None:
    source = (await client.post("/api/chats", json={"title": "Referenced history"})).json()
    first_snapshot = (
        ResolvedReference(
            reference_subject_id="refsubject_deleted",
            mention_slug="original-name",
            subject_name="Original Name",
            subject_kind="person",
            reference_asset_ids=("refasset_second", "refasset_first"),
            artifact_ids=("artifact_second", "artifact_first"),
            role="subject",
            strength=0.65,
            source=MentionSource.INHERITED_CONTEXT,
        ),
        ResolvedReference(
            reference_subject_id="refsubject_place",
            mention_slug="old-studio",
            subject_name="Old Studio",
            subject_kind="place",
            role="style",
            strength=0.25,
            source=MentionSource.MENTION,
        ),
    )
    second_snapshot = (
        ResolvedReference(
            reference_subject_id="refsubject_object",
            mention_slug="blue-vase",
            subject_name="Blue Vase",
            subject_kind="object",
            source=MentionSource.MENTION,
        ),
    )
    with SessionLocal() as session:
        session.add_all(
            [
                Artifact(
                    id="artifact_first",
                    sha256="1" * 64,
                    kind="image",
                    media_type="image/png",
                    size_bytes=1,
                    relative_path="fork/first.png",
                ),
                Artifact(
                    id="artifact_second",
                    sha256="2" * 64,
                    kind="image",
                    media_type="image/png",
                    size_bytes=1,
                    relative_path="fork/second.png",
                ),
            ]
        )
        root = Message(chat_id=source["id"], role="user", status="complete")
        session.add(root)
        session.flush()
        leaf = Message(
            chat_id=source["id"],
            parent_id=root.id,
            role="assistant",
            status="complete",
        )
        session.add(leaf)
        session.flush()
        record_message_references(session, root.id, first_snapshot)
        record_message_references(session, leaf.id, second_snapshot)
        session.commit()
        root_id, leaf_id = root.id, leaf.id
        source_reference_ids = tuple(
            session.scalars(
                select(MessageReference.id)
                .where(MessageReference.message_id.in_((root_id, leaf_id)))
                .order_by(MessageReference.message_id, MessageReference.position)
            ).all()
        )

    response = await client.post(f"/api/messages/{leaf_id}/fork")
    assert response.status_code == 201
    fork_id = response.json()["id"]

    with SessionLocal() as session:
        copied_root = session.scalar(
            select(Message).where(Message.chat_id == fork_id, Message.parent_id.is_(None))
        )
        assert copied_root is not None
        copied_leaf = session.scalar(
            select(Message).where(
                Message.chat_id == fork_id,
                Message.parent_id == copied_root.id,
            )
        )
        assert copied_leaf is not None

        assert message_references(session, copied_root.id) == first_snapshot
        assert message_references(session, copied_leaf.id) == second_snapshot
        assert message_references(session, root_id) == first_snapshot
        assert message_references(session, leaf_id) == second_snapshot

        source_rows = session.scalars(
            select(MessageReference)
            .where(MessageReference.message_id.in_((root_id, leaf_id)))
            .order_by(MessageReference.message_id, MessageReference.position)
        ).all()
        assert tuple(row.id for row in source_rows) == source_reference_ids
        copied_rows = session.scalars(
            select(MessageReference)
            .where(MessageReference.message_id.in_((copied_root.id, copied_leaf.id)))
            .order_by(MessageReference.message_id, MessageReference.position)
        ).all()
        assert len(copied_rows) == 3
        assert not {row.id for row in copied_rows} & set(source_reference_ids)
        copied_message_ids = {copied_root.id, copied_leaf.id}
        assert {row.message_id for row in copied_rows} <= copied_message_ids
        copied_messages = [session.get(Message, row.message_id) for row in copied_rows]
        assert all(message is not None for message in copied_messages)
        assert all(message.chat_id == fork_id for message in copied_messages if message is not None)


async def test_reference_copy_failure_rolls_back_the_whole_fork(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = (
        ResolvedReference(
            reference_subject_id="refsubject_private",
            mention_slug="private-subject",
            subject_name="Private Subject",
            subject_kind="person",
        ),
    )
    source_chat = (await client.post("/api/chats", json={"title": "Rollback source"})).json()
    with SessionLocal() as session:
        root = Message(chat_id=source_chat["id"], role="user", status="complete")
        session.add(root)
        session.flush()
        leaf = Message(
            chat_id=source_chat["id"],
            parent_id=root.id,
            role="assistant",
            status="complete",
        )
        session.add(leaf)
        session.flush()
        record_message_references(session, root.id, snapshot)
        record_message_references(session, leaf.id, snapshot)
        session.commit()
        source_chat_id = source_chat["id"]
        source_message_ids = (root.id, leaf.id)
        source_reference_ids = tuple(
            session.scalars(select(MessageReference.id).order_by(MessageReference.message_id)).all()
        )

    real_carry = chat_forking_module.carry_message_references_if_absent  # type: ignore[attr-defined]
    calls = 0

    def fail_after_second_copy(
        session: Session,
        *,
        source_message_id: str,
        target_message_id: str,
    ) -> None:
        nonlocal calls
        calls += 1
        real_carry(
            session,
            source_message_id=source_message_id,
            target_message_id=target_message_id,
        )
        if calls == 2:
            raise RuntimeError("injected fork failure")

    monkeypatch.setattr(
        chat_forking_module,
        "carry_message_references_if_absent",
        fail_after_second_copy,
    )
    with SessionLocal() as session:
        with pytest.raises(RuntimeError, match="^injected fork failure$"):
            fork_chat_from_message(session, source_message_ids[1])
        session.rollback()

    with SessionLocal() as session:
        assert session.scalars(select(Chat.id)).all() == [source_chat_id]
        assert set(session.scalars(select(Message.id)).all()) == set(source_message_ids)
        rows = session.scalars(select(MessageReference).order_by(MessageReference.message_id)).all()
        assert tuple(row.id for row in rows) == source_reference_ids
        assert {row.message_id for row in rows} == set(source_message_ids)
