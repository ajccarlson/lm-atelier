from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.artifact_library import ArtifactReferenceDataError
from local_lm.db import Base
from local_lm.models import Artifact, ReferenceAsset, ReferenceSubject
from local_lm.references import ReferenceKind, ValidationState, slugify_mention


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")

    # SQLite ignores foreign keys unless asked, and every deletion rule in this
    # slice is expressed as one. Without this the tests would pass while the
    # constraints did nothing.
    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(connection, _record):  # type: ignore[no-untyped-def]
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


_artifacts = 0


def _artifact(session: Session) -> Artifact:
    global _artifacts
    _artifacts += 1
    artifact = Artifact(
        id=f"art_{_artifacts}",
        sha256=f"{_artifacts:064x}",
        kind="image",
        media_type="image/png",
        size_bytes=1,
        relative_path=f"{_artifacts}/a",
    )
    session.add(artifact)
    session.flush()
    return artifact


def _subject(session: Session, name: str = "Ada Lovelace", **changes: object) -> ReferenceSubject:
    values: dict[str, object] = {
        "name": name,
        "mention_slug": slugify_mention(name),
        "kind": ReferenceKind.PERSON.value,
    }
    values.update(changes)
    subject = ReferenceSubject(**values)  # type: ignore[arg-type]
    session.add(subject)
    session.flush()
    return subject


def test_a_subject_starts_unarchived_and_unfavourited(session: Session) -> None:
    subject = _subject(session)
    session.commit()
    assert subject.favorite is False
    assert subject.archived is False
    assert subject.aliases_json == []
    assert subject.mention_slug == "ada-lovelace"


def test_two_subjects_cannot_share_one_mention(session: Session) -> None:
    """The mention is the addressing key. Two subjects behind one `@name` would
    make every mention a coin toss."""

    _subject(session, "Ada Lovelace")
    session.commit()
    # Different display name, same canonical mention - which is exactly the
    # collision the unique index exists to stop.
    with pytest.raises(IntegrityError):
        _subject(session, "ada  lovelace")
    session.rollback()


def test_a_subject_with_no_name_is_refused(session: Session) -> None:
    session.add(ReferenceSubject(name="   ", mention_slug="x", kind="person"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_the_same_image_cannot_be_added_to_one_subject_twice(session: Session) -> None:
    """A duplicate is not a second view of the subject; it would silently weight
    the set toward whichever picture was added twice."""

    subject = _subject(session)
    artifact = _artifact(session)
    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    session.commit()

    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    with pytest.raises(IntegrityError):
        session.commit()


def test_one_image_may_belong_to_two_subjects(session: Session) -> None:
    """The artifact store is content-addressed and already counts references, so
    a photograph showing two subjects is stored once and referenced twice."""

    first, second = _subject(session, "Ada"), _subject(session, "Grace")
    artifact = _artifact(session)
    session.add(ReferenceAsset(reference_subject_id=first.id, artifact_id=artifact.id))
    session.add(ReferenceAsset(reference_subject_id=second.id, artifact_id=artifact.id))
    session.commit()
    assert session.query(ReferenceAsset).count() == 2


def test_an_image_in_use_cannot_be_deleted_out_from_under_a_subject(session: Session) -> None:
    """An unrelated cleanup must not be able to empty a Reference."""

    subject = _subject(session)
    artifact = _artifact(session)
    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    session.commit()

    session.delete(artifact)
    with pytest.raises(ArtifactReferenceDataError):
        session.commit()
    session.rollback()


def test_restrict_membership_wins_when_the_same_id_is_also_a_cover(
    session: Session,
) -> None:
    """A SET NULL cover pointer does not excuse deleting a RESTRICT asset."""

    artifact = _artifact(session)
    subject = _subject(session, cover_artifact_id=artifact.id)
    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    session.commit()

    session.delete(artifact)
    with pytest.raises(ArtifactReferenceDataError):
        session.commit()
    session.rollback()


def test_losing_a_cover_image_does_not_lose_the_subject(session: Session) -> None:
    """Images are replaceable; the identity is not."""

    artifact = _artifact(session)
    subject = _subject(session, cover_artifact_id=artifact.id)
    session.commit()

    session.delete(artifact)
    session.commit()
    session.refresh(subject)
    assert subject.cover_artifact_id is None
    assert session.query(ReferenceSubject).count() == 1


def test_deleting_a_subject_takes_its_membership_rows_with_it(session: Session) -> None:
    subject = _subject(session)
    artifact = _artifact(session)
    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    session.commit()

    session.delete(subject)
    session.commit()
    assert session.query(ReferenceAsset).count() == 0
    # The image itself survives - it is content-addressed and may be used
    # elsewhere. Membership ending is not a reason to destroy the bytes.
    assert session.query(Artifact).count() == 1


def test_an_image_starts_unchecked_rather_than_usable(session: Session) -> None:
    """An image nobody has looked at must not let an unreviewed set claim the
    fidelity of a reviewed one."""

    subject = _subject(session)
    artifact = _artifact(session)
    asset = ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id)
    session.add(asset)
    session.commit()
    assert asset.validation_state == ValidationState.UNCHECKED.value
    assert asset.validation_state != ValidationState.USABLE.value
    assert asset.validation_reasons_json == []


def test_assets_come_back_in_the_order_the_user_arranged(session: Session) -> None:
    subject = _subject(session)
    for position in (2, 0, 1):
        session.add(
            ReferenceAsset(
                reference_subject_id=subject.id,
                artifact_id=_artifact(session).id,
                sort_order=position,
            )
        )
    session.commit()
    session.refresh(subject)
    assert [asset.sort_order for asset in subject.assets] == [0, 1, 2]
