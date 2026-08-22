"""Bind explicitly selected, reviewed Reference images to one queued media run.

This is deliberately narrower than automatic Reference selection: a turn must
name the exact ReferenceAsset ids it wants.  The binding rechecks subject
membership, the immutable human-review event, the content-addressed artifact
row, and the retained bytes inside the turn transaction.  The returned facts
are safe to persist in Run provenance and to use as media inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .message_references import ResolvedReference
from .models import Artifact, ReferenceAsset, ReferenceAssetReviewEvent, ReferenceSubject
from .references import ReferencePurpose, ValidationState

MAX_CONDITIONING_IMAGES = 9


@dataclass(frozen=True, slots=True)
class BoundReferenceImage:
    reference_subject_id: str
    reference_asset_id: str
    artifact_id: str
    artifact_sha256: str
    review_event_id: str
    review_version: int
    review_decision_sha256: str

    def provenance(self) -> dict[str, object]:
        return {
            "reference_subject_id": self.reference_subject_id,
            "reference_asset_id": self.reference_asset_id,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "review_event_id": self.review_event_id,
            "review_version": self.review_version,
            "review_decision_sha256": self.review_decision_sha256,
            "explicit_selection": True,
            "review_verified": True,
            "bytes_verified": True,
        }


class ReferenceConditioningError(ValueError):
    """An exact selected Reference cannot safely condition this run."""


def bind_selected_reference_images(
    session: Session,
    artifacts: ArtifactStore,
    references: tuple[ResolvedReference, ...],
    *,
    maximum_bytes: int,
) -> tuple[BoundReferenceImage, ...]:
    selected = [
        (reference.reference_subject_id, asset_id, artifact_id)
        for reference in references
        for asset_id, artifact_id in zip(
            reference.reference_asset_ids,
            reference.artifact_ids,
            strict=True,
        )
    ]
    if not selected:
        return ()
    if len(selected) > MAX_CONDITIONING_IMAGES:
        raise ReferenceConditioningError(
            f"A generation can use at most {MAX_CONDITIONING_IMAGES} selected Reference images."
        )
    if len({asset_id for _, asset_id, _ in selected}) != len(selected):
        raise ReferenceConditioningError("A selected Reference image was included more than once.")

    bound: list[BoundReferenceImage] = []
    for subject_id, asset_id, artifact_id in selected:
        subject = session.get(ReferenceSubject, subject_id)
        if subject is None or subject.archived:
            raise ReferenceConditioningError("A selected Reference is unavailable.")
        asset = session.get(ReferenceAsset, asset_id)
        if (
            asset is None
            or asset.reference_subject_id != subject_id
            or asset.artifact_id != artifact_id
        ):
            raise ReferenceConditioningError("A selected Reference image is no longer attached.")
        if asset.purpose != ReferencePurpose.IDENTITY.value:
            raise ReferenceConditioningError(
                "A selected Reference image must have the identity purpose."
            )
        if asset.validation_state != ValidationState.USABLE.value:
            raise ReferenceConditioningError(
                "Review the selected Reference image as usable before generation."
            )
        event = session.scalar(
            select(ReferenceAssetReviewEvent).where(
                ReferenceAssetReviewEvent.reference_asset_id == asset.id,
                ReferenceAssetReviewEvent.result_version == asset.review_version,
            )
        )
        if (
            event is None
            or event.decision != ValidationState.USABLE.value
            or event.artifact_id != asset.artifact_id
        ):
            raise ReferenceConditioningError(
                "The selected Reference image does not have a matching usable review."
            )
        artifact = session.get(Artifact, artifact_id)
        if (
            artifact is None
            or artifact.id != f"sha256:{artifact.sha256}"
            or event.artifact_sha256 != artifact.sha256
        ):
            raise ReferenceConditioningError("The selected Reference image identity changed.")
        content = artifacts.read_verified_bytes(artifact, maximum_bytes=maximum_bytes)
        if not content:
            raise ReferenceConditioningError("The selected Reference image is empty.")
        bound.append(
            BoundReferenceImage(
                reference_subject_id=subject_id,
                reference_asset_id=asset.id,
                artifact_id=artifact.id,
                artifact_sha256=artifact.sha256,
                review_event_id=event.id,
                review_version=event.result_version,
                review_decision_sha256=event.decision_sha256,
            )
        )
    return tuple(bound)
