"""One authoritative human review transition for a Reference image.

Image decoding happens before the write, but it grants no authority by itself.
The authority transition is one conditional SQL statement from the exact
unchecked state/version the caller saw. The immutable event is inserted only
after that statement wins and in the same transaction.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

from PIL import Image, UnidentifiedImageError
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .domain import utcnow
from .models import Artifact, ReferenceAsset, ReferenceAssetReviewEvent
from .references import ValidationState

ReviewDecision = Literal["usable", "weak", "rejected"]
REVIEWER_KIND = "local-human"
MAX_REVIEW_REASONS = 16
MAX_REVIEW_REASON_CHARS = 500


class ReferenceReviewError(ValueError):
    """A Reference review could not be completed."""


class ReferenceReviewNotFound(ReferenceReviewError):
    """The named subject membership does not exist."""


class ReferenceReviewConflict(ReferenceReviewError):
    """The expected unchecked state/version no longer owns the transition."""

    def __init__(self, state: str, version: int) -> None:
        super().__init__(f"reference asset review is already {state} at version {version}")
        self.state = state
        self.version = version


class ReferenceReviewInvalid(ReferenceReviewError):
    """The decision or retained image bytes cannot support a review."""


@dataclass(frozen=True)
class ReferenceReviewResult:
    asset: ReferenceAsset
    review: ReferenceAssetReviewEvent
    idempotent: bool


@dataclass(frozen=True)
class _AssetSnapshot:
    id: str
    subject_id: str
    artifact_id: str
    artifact_sha256: str
    state: str
    version: int


def _decision(value: object) -> ReviewDecision:
    if value not in {
        ValidationState.USABLE.value,
        ValidationState.WEAK.value,
        ValidationState.REJECTED.value,
    }:
        raise ReferenceReviewInvalid("review decision must be usable, weak, or rejected")
    return value


def _reasons(values: object, decision: ReviewDecision) -> tuple[str, ...]:
    if not isinstance(values, list) or len(values) > MAX_REVIEW_REASONS:
        raise ReferenceReviewInvalid(f"a review has at most {MAX_REVIEW_REASONS} reasons")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ReferenceReviewInvalid("review reasons must be text")
        reason = value.strip()
        if not reason or len(reason) > MAX_REVIEW_REASON_CHARS:
            raise ReferenceReviewInvalid(
                f"each review reason must contain 1 to {MAX_REVIEW_REASON_CHARS} characters"
            )
        if reason in cleaned:
            raise ReferenceReviewInvalid("review reasons must be distinct")
        cleaned.append(reason)
    if decision in {ValidationState.WEAK.value, ValidationState.REJECTED.value} and not cleaned:
        raise ReferenceReviewInvalid(f"a {decision} review needs at least one reason")
    return tuple(cleaned)


def _snapshot(
    session: Session, *, subject_id: str, asset_id: str
) -> tuple[_AssetSnapshot, Artifact]:
    with session.no_autoflush:
        row = session.execute(
            select(
                ReferenceAsset.id,
                ReferenceAsset.reference_subject_id,
                ReferenceAsset.artifact_id,
                ReferenceAsset.validation_state,
                ReferenceAsset.review_version,
            ).where(
                ReferenceAsset.id == asset_id,
                ReferenceAsset.reference_subject_id == subject_id,
            )
        ).one_or_none()
        if row is None:
            raise ReferenceReviewNotFound("that image is not attached to this reference")
        artifact = session.get(Artifact, row.artifact_id)
    if artifact is None:
        raise ReferenceReviewInvalid("the attached artifact no longer exists")
    if not artifact.media_type.casefold().startswith("image/"):
        raise ReferenceReviewInvalid("only a retained image can be reviewed as a Reference asset")
    return (
        _AssetSnapshot(
            id=row.id,
            subject_id=row.reference_subject_id,
            artifact_id=row.artifact_id,
            artifact_sha256=artifact.sha256,
            state=row.validation_state,
            version=row.review_version,
        ),
        artifact,
    )


def _verified_dimensions(content: bytes, *, maximum_pixels: int) -> tuple[int, int]:
    if type(maximum_pixels) is not int or maximum_pixels < 1:
        raise ReferenceReviewInvalid("reference image pixel limit is invalid")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as candidate:
                width, height = candidate.size
                if width < 1 or height < 1 or width * height > maximum_pixels:
                    raise ReferenceReviewInvalid("reference image exceeds the pixel limit")
                if getattr(candidate, "n_frames", 1) != 1:
                    raise ReferenceReviewInvalid("animated reference images are not reviewable")
                # verify() walks the encoded structure and detects truncated
                # streams without trusting a successful header read.
                candidate.verify()
            with Image.open(io.BytesIO(content)) as decoded:
                if decoded.size != (width, height):
                    raise ReferenceReviewInvalid(
                        "reference image dimensions changed while decoding"
                    )
                if decoded.width * decoded.height > maximum_pixels:
                    raise ReferenceReviewInvalid("reference image exceeds the pixel limit")
                # A second open plus load() forces the complete pixel stream.
                decoded.load()
    except ReferenceReviewInvalid:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise ReferenceReviewInvalid("reference image failed complete decoding") from exc
    return width, height


def _digest(
    snapshot: _AssetSnapshot,
    *,
    expected_state: str,
    expected_version: int,
    decision: ReviewDecision,
    reasons: tuple[str, ...],
    width: int,
    height: int,
) -> str:
    payload = {
        "artifact_id": snapshot.artifact_id,
        "artifact_sha256": snapshot.artifact_sha256,
        "decision": decision,
        "expected_state": expected_state,
        "expected_version": expected_version,
        "height": height,
        "reasons": list(reasons),
        "reference_asset_id": snapshot.id,
        "reviewer_kind": REVIEWER_KIND,
        "width": width,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _matching_retry(
    session: Session,
    snapshot: _AssetSnapshot,
    *,
    expected_state: str,
    expected_version: int,
    decision: ReviewDecision,
    reasons: tuple[str, ...],
) -> ReferenceReviewResult | None:
    with session.no_autoflush:
        review = session.scalar(
            select(ReferenceAssetReviewEvent).where(
                ReferenceAssetReviewEvent.reference_asset_id == snapshot.id,
                ReferenceAssetReviewEvent.result_version == expected_version + 1,
            )
        )
    if (
        review is None
        or review.expected_state != expected_state
        or review.expected_version != expected_version
        or review.decision != decision
        or tuple(review.reasons_json) != reasons
        or review.artifact_id != snapshot.artifact_id
        or review.artifact_sha256 != snapshot.artifact_sha256
        or snapshot.state != review.decision
        or snapshot.version != review.result_version
    ):
        return None
    asset = session.get(ReferenceAsset, snapshot.id, populate_existing=True)
    if asset is None:
        return None
    return ReferenceReviewResult(asset=asset, review=review, idempotent=True)


def review_reference_asset(
    session: Session,
    artifacts: ArtifactStore,
    *,
    subject_id: str,
    asset_id: str,
    expected_state: str,
    expected_version: int,
    decision: object,
    reasons: object,
    maximum_bytes: int,
    maximum_pixels: int,
) -> ReferenceReviewResult:
    """Settle one unchecked membership, or return its exact prior decision."""

    if expected_state != ValidationState.UNCHECKED.value:
        raise ReferenceReviewInvalid("expected_state must be unchecked")
    if type(expected_version) is not int or expected_version < 1:
        raise ReferenceReviewInvalid("expected_version must be a positive integer")
    settled = _decision(decision)
    cleaned_reasons = _reasons(reasons, settled)
    snapshot, artifact = _snapshot(session, subject_id=subject_id, asset_id=asset_id)

    if snapshot.state != expected_state or snapshot.version != expected_version:
        retry = _matching_retry(
            session,
            snapshot,
            expected_state=expected_state,
            expected_version=expected_version,
            decision=settled,
            reasons=cleaned_reasons,
        )
        if retry is not None:
            return retry
        raise ReferenceReviewConflict(snapshot.state, snapshot.version)

    try:
        content = artifacts.read_verified_bytes(artifact, maximum_bytes=maximum_bytes)
    except (OSError, ValueError) as exc:
        raise ReferenceReviewInvalid("reference image bytes failed retained verification") from exc
    width, height = _verified_dimensions(content, maximum_pixels=maximum_pixels)
    decision_sha256 = _digest(
        snapshot,
        expected_state=expected_state,
        expected_version=expected_version,
        decision=settled,
        reasons=cleaned_reasons,
        width=width,
        height=height,
    )
    reviewed_at: datetime = utcnow()

    # Core SQL deliberately avoids ORM synchronization. No state, reasons, or
    # dimensions are assigned to an ORM object before this compare-and-swap.
    with session.no_autoflush:
        changed = cast(
            CursorResult[Any],
            session.execute(
                update(ReferenceAsset)
                .execution_options(synchronize_session=False)
                .where(
                    ReferenceAsset.id == snapshot.id,
                    ReferenceAsset.reference_subject_id == snapshot.subject_id,
                    ReferenceAsset.artifact_id == snapshot.artifact_id,
                    ReferenceAsset.validation_state == expected_state,
                    ReferenceAsset.review_version == expected_version,
                )
                .values(
                    validation_state=settled,
                    validation_reasons_json=list(cleaned_reasons),
                    width=width,
                    height=height,
                    review_version=expected_version + 1,
                    updated_at=reviewed_at,
                )
            ),
        )
    if changed.rowcount != 1:
        current = session.execute(
            select(
                ReferenceAsset.validation_state,
                ReferenceAsset.review_version,
                ReferenceAsset.artifact_id,
            ).where(ReferenceAsset.id == snapshot.id)
        ).one_or_none()
        if current is None or current.artifact_id != snapshot.artifact_id:
            raise ReferenceReviewConflict("missing", expected_version)
        raced = _AssetSnapshot(
            id=snapshot.id,
            subject_id=snapshot.subject_id,
            artifact_id=snapshot.artifact_id,
            artifact_sha256=snapshot.artifact_sha256,
            state=current.validation_state,
            version=current.review_version,
        )
        retry = _matching_retry(
            session,
            raced,
            expected_state=expected_state,
            expected_version=expected_version,
            decision=settled,
            reasons=cleaned_reasons,
        )
        if retry is not None:
            return retry
        raise ReferenceReviewConflict(current.validation_state, current.review_version)

    review = ReferenceAssetReviewEvent(
        id=f"refreview:sha256:{decision_sha256}",
        reference_asset_id=snapshot.id,
        artifact_id=snapshot.artifact_id,
        artifact_sha256=snapshot.artifact_sha256,
        reviewer_kind=REVIEWER_KIND,
        expected_state=expected_state,
        expected_version=expected_version,
        result_version=expected_version + 1,
        decision=settled,
        reasons_json=list(cleaned_reasons),
        width=width,
        height=height,
        decision_sha256=decision_sha256,
        reviewed_at=reviewed_at,
    )
    session.add(review)
    session.flush()
    # Refresh only after the conditional write and event insert have succeeded.
    asset = session.get(ReferenceAsset, snapshot.id, populate_existing=True)
    if asset is None:  # pragma: no cover - protected by the same transaction
        raise ReferenceReviewConflict("missing", expected_version + 1)
    return ReferenceReviewResult(asset=asset, review=review, idempotent=False)
