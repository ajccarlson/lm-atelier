from __future__ import annotations

import io
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Update

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import ReferenceAsset, ReferenceAssetReviewEvent
from local_lm.reference_library import attach_asset, create_subject
from local_lm.reference_review import (
    ReferenceReviewConflict,
    ReferenceReviewInvalid,
    review_reference_asset,
)


@pytest.fixture
def review_context(tmp_path: Path) -> Generator[tuple[Session, ArtifactStore], None, None]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _foreign_keys(connection: Any, _record: object) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    with Session(engine) as session:
        yield session, ArtifactStore(settings)


def _png(size: tuple[int, int] = (8, 6), shade: int = 120) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(buffer, format="PNG")
    return buffer.getvalue()


def _asset(
    session: Session,
    store: ArtifactStore,
    content: bytes,
    *,
    name: str = "reference.png",
) -> tuple[str, str]:
    artifact = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name=name,
    )
    subject = create_subject(session, name=f"Subject {artifact.sha256[:8]}", kind="person")
    asset = attach_asset(session, subject, artifact_id=artifact.id).asset
    session.commit()
    return subject.id, asset.id


def _review(
    session: Session,
    store: ArtifactStore,
    *,
    subject_id: str,
    asset_id: str,
    decision: str = "usable",
    reasons: list[str] | None = None,
    expected_version: int = 1,
    maximum_pixels: int = 1_000_000,
):  # type: ignore[no-untyped-def]
    return review_reference_asset(
        session,
        store,
        subject_id=subject_id,
        asset_id=asset_id,
        expected_state="unchecked",
        expected_version=expected_version,
        decision=decision,
        reasons=reasons or [],
        maximum_bytes=1_000_000,
        maximum_pixels=maximum_pixels,
    )


def test_review_is_one_conditional_transition_with_an_immutable_digest(
    review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, _png())

    result = _review(
        session,
        store,
        subject_id=subject_id,
        asset_id=asset_id,
        decision="weak",
        reasons=["soft focus"],
    )
    session.commit()

    assert result.idempotent is False
    assert result.asset.validation_state == "weak"
    assert result.asset.validation_reasons_json == ["soft focus"]
    assert (result.asset.width, result.asset.height, result.asset.review_version) == (8, 6, 2)
    assert result.review.id == f"refreview:sha256:{result.review.decision_sha256}"
    assert result.review.artifact_sha256
    assert result.review.reviewer_kind == "local-human"


def test_exact_retry_is_idempotent_but_conflict_and_rereview_are_refused(
    review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, _png())
    first = _review(
        session,
        store,
        subject_id=subject_id,
        asset_id=asset_id,
        decision="rejected",
        reasons=["wrong subject"],
    )
    session.commit()

    same = _review(
        session,
        store,
        subject_id=subject_id,
        asset_id=asset_id,
        decision="rejected",
        reasons=["wrong subject"],
    )
    assert same.idempotent is True
    assert same.review.id == first.review.id
    assert session.scalar(select(func.count()).select_from(ReferenceAssetReviewEvent)) == 1

    with pytest.raises(ReferenceReviewConflict):
        _review(
            session,
            store,
            subject_id=subject_id,
            asset_id=asset_id,
            decision="weak",
            reasons=["wrong subject"],
        )
    with pytest.raises(ReferenceReviewConflict):
        _review(
            session,
            store,
            subject_id=subject_id,
            asset_id=asset_id,
            decision="rejected",
            reasons=["wrong subject"],
            expected_version=2,
        )


def test_lost_race_changes_nothing_owned_by_the_loser(
    review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, _png())
    original_execute = session.execute
    raced = False

    def racing_execute(statement: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        nonlocal raced
        if (
            not raced
            and isinstance(statement, Update)
            and statement.table.name == "reference_assets"
        ):
            raced = True
            session.connection().exec_driver_sql(
                """
                UPDATE reference_assets
                SET validation_state = 'rejected',
                    validation_reasons_json = '["winner"]',
                    width = 7,
                    height = 9,
                    review_version = 2
                WHERE id = ?
                """,
                (asset_id,),
            )
        return original_execute(statement, *args, **kwargs)

    monkeypatch.setattr(session, "execute", racing_execute)
    with pytest.raises(ReferenceReviewConflict):
        _review(
            session,
            store,
            subject_id=subject_id,
            asset_id=asset_id,
            decision="weak",
            reasons=["loser"],
        )

    current = original_execute(
        select(
            ReferenceAsset.validation_state,
            ReferenceAsset.validation_reasons_json,
            ReferenceAsset.width,
            ReferenceAsset.height,
            ReferenceAsset.review_version,
        ).where(ReferenceAsset.id == asset_id)
    ).one()
    assert tuple(current) == ("rejected", ["winner"], 7, 9, 2)
    assert session.scalar(select(func.count()).select_from(ReferenceAssetReviewEvent)) == 0


@pytest.mark.parametrize(
    "content",
    [
        b"not an image",
        _png()[:-12],
    ],
)
def test_corrupt_and_truncated_images_never_settle(
    review_context: tuple[Session, ArtifactStore],
    content: bytes,
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, content)

    with pytest.raises(ReferenceReviewInvalid, match="complete decoding"):
        _review(session, store, subject_id=subject_id, asset_id=asset_id)

    asset = session.get(ReferenceAsset, asset_id)
    assert asset is not None
    assert (asset.validation_state, asset.validation_reasons_json) == ("unchecked", [])
    assert (asset.width, asset.height, asset.review_version) == (None, None, 1)
    assert session.scalar(select(func.count()).select_from(ReferenceAssetReviewEvent)) == 0


def test_pixel_cap_refuses_before_full_decode_authority(
    review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, _png((20, 20)))

    with pytest.raises(ReferenceReviewInvalid, match="pixel limit"):
        _review(
            session,
            store,
            subject_id=subject_id,
            asset_id=asset_id,
            maximum_pixels=399,
        )
    asset = session.get(ReferenceAsset, asset_id)
    assert asset is not None and asset.validation_state == "unchecked"


def test_review_event_cannot_be_updated_or_deleted(
    review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, _png())
    result = _review(session, store, subject_id=subject_id, asset_id=asset_id)
    session.commit()

    with pytest.raises(IntegrityError, match="immutable"):
        session.connection().exec_driver_sql(
            "UPDATE reference_asset_review_events SET decision = 'weak' WHERE id = ?",
            (result.review.id,),
        )
    session.rollback()
    with pytest.raises(IntegrityError, match="immutable"):
        session.connection().exec_driver_sql(
            "DELETE FROM reference_asset_review_events WHERE id = ?",
            (result.review.id,),
        )
    session.rollback()


def test_assets_cannot_begin_settled_or_change_reviewed_artifact_identity(
    review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = review_context
    subject_id, asset_id = _asset(session, store, _png())
    asset = session.get(ReferenceAsset, asset_id)
    assert asset is not None
    session.add(
        ReferenceAsset(
            reference_subject_id=asset.reference_subject_id,
            artifact_id=asset.artifact_id,
            validation_state="usable",
            validation_reasons_json=[],
            width=8,
            height=6,
            review_version=2,
        )
    )
    with pytest.raises(IntegrityError, match="must begin unchecked"):
        session.flush()
    session.rollback()

    subject_id, asset_id = _asset(session, store, _png(shade=121), name="other.png")
    result = _review(session, store, subject_id=subject_id, asset_id=asset_id)
    session.commit()
    with pytest.raises(IntegrityError, match="identity is immutable"):
        session.connection().exec_driver_sql(
            "UPDATE reference_assets SET artifact_id = ? WHERE id = ?",
            ("sha256:" + "f" * 64, result.asset.id),
        )
    session.rollback()
