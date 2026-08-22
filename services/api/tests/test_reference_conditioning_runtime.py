from __future__ import annotations

import io
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.message_references import ResolvedReference
from local_lm.models import Artifact
from local_lm.reference_conditioning_runtime import (
    ReferenceConditioningError,
    bind_selected_reference_images,
)
from local_lm.reference_library import attach_asset, create_subject
from local_lm.reference_review import review_reference_asset


@pytest.fixture
def context(tmp_path: Path) -> Generator[tuple[Session, ArtifactStore], None, None]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _foreign_keys(connection: Any, _record: object) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    with Session(engine) as session:
        yield session, ArtifactStore(settings)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), (80, 120, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _selection(
    session: Session,
    store: ArtifactStore,
    *,
    review: bool,
) -> tuple[ResolvedReference, str]:
    artifact = store.ingest_bytes(
        session,
        _png(),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="reference.png",
    )
    subject = create_subject(session, name="Selected subject", kind="person")
    asset = attach_asset(
        session,
        subject,
        artifact_id=artifact.id,
        purpose="identity",
    ).asset
    session.commit()
    if review:
        review_reference_asset(
            session,
            store,
            subject_id=subject.id,
            asset_id=asset.id,
            expected_state="unchecked",
            expected_version=1,
            decision="usable",
            reasons=[],
            maximum_bytes=1_000_000,
            maximum_pixels=1_000_000,
        )
        session.commit()
    return (
        ResolvedReference(
            reference_subject_id=subject.id,
            mention_slug=subject.mention_slug,
            subject_name=subject.name,
            subject_kind=subject.kind,
            reference_asset_ids=(asset.id,),
            artifact_ids=(artifact.id,),
        ),
        artifact.id,
    )


def test_reviewed_exact_selection_binds_bytes_and_review_event(
    context: tuple[Session, ArtifactStore],
) -> None:
    session, store = context
    reference, artifact_id = _selection(session, store, review=True)

    result = bind_selected_reference_images(
        session,
        store,
        (reference,),
        maximum_bytes=1_000_000,
    )

    assert len(result) == 1
    assert result[0].artifact_id == artifact_id
    assert result[0].review_version == 2
    assert result[0].review_event_id.startswith("refreview:sha256:")
    assert result[0].provenance()["bytes_verified"] is True


def test_unreviewed_selection_refuses_instead_of_substituting(
    context: tuple[Session, ArtifactStore],
) -> None:
    session, store = context
    reference, _ = _selection(session, store, review=False)

    with pytest.raises(ReferenceConditioningError, match="Review the selected"):
        bind_selected_reference_images(
            session,
            store,
            (reference,),
            maximum_bytes=1_000_000,
        )


def test_changed_retained_bytes_refuse_before_queue(
    context: tuple[Session, ArtifactStore],
) -> None:
    session, store = context
    reference, artifact_id = _selection(session, store, review=True)
    artifact = session.get(Artifact, artifact_id)
    assert artifact is not None
    store.resolve(artifact).write_bytes(b"changed")

    with pytest.raises(ValueError, match="size does not match"):
        bind_selected_reference_images(
            session,
            store,
            (reference,),
            maximum_bytes=1_000_000,
        )


def test_duplicate_selection_refuses(
    context: tuple[Session, ArtifactStore],
) -> None:
    session, store = context
    reference, _ = _selection(session, store, review=True)

    with pytest.raises(ReferenceConditioningError, match="more than once"):
        bind_selected_reference_images(
            session,
            store,
            (reference, reference),
            maximum_bytes=1_000_000,
        )
