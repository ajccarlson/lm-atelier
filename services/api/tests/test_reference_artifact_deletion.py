"""An image a Reference is holding survives the paths that delete images.

`ReferenceAsset.artifact_id` is `ondelete="RESTRICT"`, and `models.py` says the
constraint exists so an unrelated cleanup cannot take an image a Reference
depends on. Nothing tested that invariant end to end, which is why these exist.

The invariant holds. Three separate things enforce it: `_delete_artifact`
refuses when the id is in `referenced_artifact_ids`; a `before_flush` listener
registered on every `Session` refuses the same set, so a path that never calls
that helper is still stopped; and the column is `ondelete="RESTRICT"`, so the
database refuses last.

**What these tests do not prove.** None of them can tell you the Reference is
what saved the image, and I checked rather than assumed. `referenced_artifact_ids`
counts `ArtifactLibraryEntry.artifact_id` alongside `ReferenceAsset.artifact_id`,
and every image a Reference can hold also has a Media Library entry - membership
is granted at all three sites that create one, and no path deletes such a row,
because the library models removal as state on the row rather than its absence.
So membership covers every case the Reference covers.

Deleting `ReferenceAsset.artifact_id` from that set leaves all four tests
passing. Only deleting `ArtifactLibraryEntry.artifact_id` as well turns the last
one red. That is the measurement, and it says these tests are pinned by
membership.

This is not a gap that can be closed from here: no reachable state holds an
image in a Reference without also holding it in the library, so isolating the
Reference would mean fabricating a state the API cannot produce, and a test
built on one would be pinning fiction. What these do assert is the behaviour a
person depends on - the image is still there afterwards - across the three
routes claude/R787 named, which had no coverage at all. If the Media Library
rule is ever relaxed, the last test turns red and the isolation question becomes
answerable for real; until then it is honest to say the guarantee is joint.

Written against a report (claude/R787) that turned out to describe an older base
where `artifact_library.py` did not exist yet. On that revision the failures
were real; here they are not, and the honest residue of the exercise is
coverage for an invariant that had none.
"""

from __future__ import annotations

import io

import pytest
from httpx2 import AsyncClient
from PIL import Image

from local_lm.artifact_library import ArtifactReferenceDataError
from local_lm.db import SessionLocal
from local_lm.models import Artifact


def still_stored(artifact_id: str) -> bool:
    """Whether the row survives.

    Asked of the database rather than `GET /api/artifacts`, which is the media
    *library* view - it joins through `MessagePart`, so an image uploaded
    directly and held only by a Reference is correctly absent from it and
    reading that as "deleted" would be reading the wrong question.
    """

    with SessionLocal() as session:
        return session.get(Artifact, artifact_id) is not None


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (110, 120, 130)).save(buffer, format="PNG")
    return buffer.getvalue()


async def held_by_a_reference(client: AsyncClient, name: str) -> str:
    """An uploaded image that a Reference subject is holding."""

    upload = await client.post(
        "/api/artifacts",
        params={"kind": "image"},
        files={"file": (f"{name}.png", png(), "image/png")},
    )
    assert upload.status_code in (200, 201)
    artifact_id = upload.json()["id"]

    subject = await client.post("/api/references", json={"name": name, "kind": "person"})
    assert subject.status_code == 201
    attached = await client.post(
        f"/api/references/{subject.json()['id']}/assets",
        json={"artifact_id": artifact_id, "purpose": "identity"},
    )
    assert attached.status_code == 201
    assert isinstance(artifact_id, str)
    return artifact_id


async def test_deleting_a_referenced_image_refuses_rather_than_failing(
    client: AsyncClient,
) -> None:
    """A refusal the user can act on, and the image still there afterwards.

    Which refusal is deliberately not asserted: today it is Media Library
    membership rather than the reference, and pinning that message would make
    this a test of the wrong rule.
    """

    artifact_id = await held_by_a_reference(client, "Held")

    deleted = await client.delete(f"/api/artifacts/{artifact_id}")

    assert deleted.status_code == 409, deleted.text
    # And the image survives, because refusing has to mean refusing.
    assert still_stored(artifact_id)


async def test_a_chat_with_a_referenced_image_still_deletes(client: AsyncClient) -> None:
    """Deleting a chat must not be blocked by an image somebody kept.

    Keeping the image is the right answer - it is in their Reference library
    now, which is a deliberate act of keeping - and the chat must still go. If
    the sweep ever tried to take that image instead, the constraint would stop
    it mid-transaction and the chat would survive its own deletion, leaving no
    way to remove it by this route.
    """

    chat = await client.post("/api/chats", json={"title": "Held media"})
    assert chat.status_code == 201
    chat_id = chat.json()["id"]

    artifact_id = await held_by_a_reference(client, "Kept")

    deleted = await client.delete(f"/api/chats/{chat_id}", params={"delete_generated_media": True})

    assert deleted.status_code in (200, 204)
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404
    # The Reference keeps its picture.
    assert still_stored(artifact_id)


async def test_retention_does_not_try_to_take_a_referenced_image(
    client: AsyncClient,
) -> None:
    """`cleanup_retention` must treat a Reference as a reason to keep.

    If it selects the image, the constraint stops it mid-sweep and retention
    stops running at all - which is the failure `models.py` names when it
    explains why the constraint is RESTRICT rather than SET NULL.
    """

    artifact_id = await held_by_a_reference(client, "Retained")

    swept = await client.post("/api/artifacts/cleanup", json={"dry_run": False})

    assert swept.status_code in (200, 201)
    assert still_stored(artifact_id)


async def test_deleting_the_row_directly_is_refused_before_the_database_sees_it(
    client: AsyncClient,
) -> None:
    """The listener, reached the way an unwritten path would reach it.

    The routes above all go through a helper that checks first. This deletes the
    row through the session instead, asking nobody, which is what a future path
    would do by accident - and the `before_flush` listener answers rather than
    letting the statement run and surface a database error as a 500.

    It does not isolate the Reference; see the module docstring for the
    measurement showing membership is what holds this up today. Its value is the
    layer: refusal happens in Python, before the delete is issued, on a path
    that called no helper.
    """

    artifact_id = await held_by_a_reference(client, "Alone")

    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        session.delete(artifact)
        with pytest.raises(ArtifactReferenceDataError):
            session.flush()

    assert still_stored(artifact_id)
