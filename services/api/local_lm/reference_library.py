"""Creating, renaming, archiving and deleting the subjects a user can name.

Two rules drive most of this and neither is obvious from the data model alone.

**Archiving is the removal a user wants; deletion is the one they have to mean.**
A subject is referenced by past runs, and those references are history rather
than pointers - a run recorded what it used, and that record has to stay true
after the subject is gone. So deletion reports what it would cost before it
happens, and archiving exists so that the common case never needs that
conversation at all.

**A rename is not a new subject.** The display name changes, the mention may
change with it, and neither rewrites what an old run meant. Only the addressing
token is unique; the human-facing name never is, because two people can
reasonably be called the same thing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import String, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .image_edit_difference import compare_images
from .models import Artifact, ReferenceAsset, ReferenceSubject
from .references import (
    MAX_NAME,
    MAX_SLUG,
    ReferenceError,
    ReferenceKind,
    ReferencePurpose,
    ValidationState,
    parse_kind,
    parse_purpose,
    slugify_mention,
    valid_mention_slug,
)

MAX_PAGE = 200
DEFAULT_PAGE = 50
# Bounded so a subject cannot carry an unbounded JSON blob that every list
# response then has to serialise.
MAX_DESCRIPTION = 4000
MAX_ALIASES = 32
MAX_TAGS = 32
MAX_SLUG_COLLISION_SUFFIX = 999

_INVALID_MENTION = "a mention must use lowercase letters, digits and hyphens"
_MENTION_TAKEN = "that mention is already in use"
_MENTION_EXHAUSTED = "a unique mention could not be assigned"
# Image comparison decodes every candidate. Skip a larger advisory scan rather
# than turning one attachment into unbounded retained-byte reads and image
# work. Attachment remains uncapped; an empty similarity result already means
# that comparison could not run, not that every stored image was compared.
MAX_SIMILARITY_CANDIDATES = 64


@dataclass(frozen=True)
class DeletionImpact:
    """What permanently deleting a subject would take with it."""

    reference_subject_id: str
    name: str
    asset_count: int
    # Artifacts this subject is the only user of. Everything else it holds is
    # shared, and shared bytes are not this subject's to destroy.
    exclusive_artifact_ids: tuple[str, ...]

    @property
    def shared_artifact_count(self) -> int:
        return self.asset_count - len(self.exclusive_artifact_ids)


def _available_slug(session: Session, desired: str, *, exclude_id: str | None = None) -> str:
    """The canonical slug, or a numbered variant when it is already taken.

    Suffixing rather than refusing, because the collision is usually two people
    with the same name rather than a mistake, and making the second one fail
    would push the user into inventing a worse name themselves.
    """

    if not valid_mention_slug(desired):
        raise ReferenceError(_INVALID_MENTION)

    def is_taken(candidate: str) -> bool:
        query = select(ReferenceSubject.id).where(ReferenceSubject.mention_slug == candidate)
        if exclude_id is not None:
            query = query.where(ReferenceSubject.id != exclude_id)
        return session.scalar(query) is not None

    if not is_taken(desired):
        return desired
    for suffix in range(2, MAX_SLUG_COLLISION_SUFFIX + 1):
        ending = f"-{suffix}"
        stem = desired[: MAX_SLUG - len(ending)].rstrip("-")
        candidate = f"{stem}{ending}"
        if not is_taken(candidate):
            return candidate
    raise ReferenceError(_MENTION_EXHAUSTED)


def create_subject(
    session: Session,
    *,
    name: str,
    kind: ReferenceKind | str,
    mention_slug: str | None = None,
    description: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
) -> ReferenceSubject:
    """Add a subject, giving it a mention nobody else answers to."""

    cleaned = (name or "").strip()
    if not cleaned:
        raise ReferenceError("a subject needs a name")
    if len(cleaned) > MAX_NAME:
        raise ReferenceError(f"a name is at most {MAX_NAME} characters")

    if mention_slug is None:
        slug = _available_slug(session, slugify_mention(cleaned))
    else:
        slug = mention_slug.strip()
        if not valid_mention_slug(slug):
            raise ReferenceError(_INVALID_MENTION)
        if session.scalar(select(ReferenceSubject.id).where(ReferenceSubject.mention_slug == slug)):
            # An explicitly chosen mention is refused rather than silently
            # renumbered: the user asked for that exact one.
            raise ReferenceError(_MENTION_TAKEN)

    cleaned_description = (description or "").strip()
    if len(cleaned_description) > MAX_DESCRIPTION:
        raise ReferenceError(f"a description is at most {MAX_DESCRIPTION} characters")
    subject = ReferenceSubject(
        name=cleaned,
        mention_slug=slug,
        kind=parse_kind(kind).value,
        description=cleaned_description or None,
        aliases_json=_clean_labels(list(aliases or []), MAX_ALIASES, "aliases"),
        tags_json=_clean_labels(list(tags or []), MAX_TAGS, "tags"),
    )
    session.add(subject)
    session.flush()
    return subject


def rename_subject(
    session: Session, subject: ReferenceSubject, *, name: str, follow_mention: bool = False
) -> ReferenceSubject:
    """Change what a subject is called, and optionally how it is addressed.

    The mention deliberately does not follow by default. Past turns recorded a
    display name for provenance, but a live chat draft may hold the mention, and
    silently changing the addressing token underneath someone mid-sentence is
    worse than letting the two drift apart.
    """

    cleaned = (name or "").strip()
    if not cleaned:
        raise ReferenceError("a subject needs a name")
    if len(cleaned) > MAX_NAME:
        raise ReferenceError(f"a name is at most {MAX_NAME} characters")
    mention_slug = subject.mention_slug
    if follow_mention:
        mention_slug = _available_slug(session, slugify_mention(cleaned), exclude_id=subject.id)
    subject.name = cleaned
    subject.mention_slug = mention_slug
    session.flush()
    return subject


def set_details(
    session: Session,
    subject: ReferenceSubject,
    *,
    description: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
) -> ReferenceSubject:
    """Correct what a subject says about itself.

    These could be set when a subject was created and never afterwards, which
    made a typo in a description permanent and an alias impossible to withdraw.

    `None` means leave that field alone. An empty string or an empty list means
    clear it - the distinction matters, because "I am not editing the aliases"
    and "this subject has no aliases any more" are different instructions and a
    single nullable value cannot carry both.
    """

    if description is not None:
        cleaned = description.strip()
        if len(cleaned) > MAX_DESCRIPTION:
            raise ReferenceError(f"a description is at most {MAX_DESCRIPTION} characters")
        subject.description = cleaned or None
    if aliases is not None:
        subject.aliases_json = _clean_labels(aliases, MAX_ALIASES, "aliases")
    if tags is not None:
        subject.tags_json = _clean_labels(tags, MAX_TAGS, "tags")
    session.flush()
    return subject


def _clean_labels(values: list[str], limit: int, field: str) -> list[str]:
    """Trim, drop blanks, and collapse repeats while keeping the order given.

    Case-insensitive de-duplication, because two aliases differing only in case
    are one alias to anyone reading them and would both have to be matched by
    anything that later resolves a name.
    """

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ReferenceError(f"{field} must be text")
        trimmed = value.strip()
        if not trimmed:
            continue
        if len(trimmed) > MAX_NAME:
            raise ReferenceError(f"each of the {field} is at most {MAX_NAME} characters")
        folded = trimmed.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        cleaned.append(trimmed)
    if len(cleaned) > limit:
        raise ReferenceError(f"a subject has at most {limit} {field}")
    return cleaned


def set_archived(session: Session, subject: ReferenceSubject, archived: bool) -> ReferenceSubject:
    """Archive or restore. Reversible on purpose - this is the normal removal."""

    subject.archived = bool(archived)
    session.flush()
    return subject


def set_favorite(session: Session, subject: ReferenceSubject, favorite: bool) -> ReferenceSubject:
    """Mark for the user's own convenience.

    Organisation only. Nothing may read this as a quality signal or feed it into
    ranking - it says someone wanted this near the top of a list, not that its
    images are good.
    """

    subject.favorite = bool(favorite)
    session.flush()
    return subject


def list_subjects(
    session: Session,
    *,
    kind: ReferenceKind | str | None = None,
    include_archived: bool = False,
    search: str | None = None,
    limit: int = DEFAULT_PAGE,
    offset: int = 0,
) -> tuple[list[ReferenceSubject], int]:
    """A page of subjects and the total that matched.

    Archived subjects are excluded unless asked for, because archiving is the
    removal a user expects to be honoured everywhere by default.
    """

    if limit < 1 or limit > MAX_PAGE:
        raise ReferenceError(f"a page is between 1 and {MAX_PAGE} subjects")
    if offset < 0:
        raise ReferenceError("an offset cannot be negative")

    filters: list[ColumnElement[bool]] = []
    if not include_archived:
        filters.append(ReferenceSubject.archived.is_(False))
    if kind is not None:
        filters.append(ReferenceSubject.kind == parse_kind(kind).value)
    if search and search.strip():
        pattern = f"%{search.strip().casefold()}%"
        # Aliases are searched as well as the name, because that is the whole
        # reason a subject has them: the person who wrote "Ada Lovelace" and
        # the person looking for "Countess Lovelace" mean the same subject, and
        # a search that only knows the display name makes the alias decorative.
        #
        # Matched against the stored JSON text rather than a joined table. The
        # alternative is a second table for what is usually two or three short
        # strings, and the false positives this can produce need the search term
        # to contain JSON punctuation, which no useful search does.
        filters.append(
            or_(
                func.lower(ReferenceSubject.name).like(pattern),
                func.lower(func.cast(ReferenceSubject.aliases_json, String)).like(pattern),
            )
        )

    total = session.scalar(select(func.count()).select_from(ReferenceSubject).where(*filters))
    rows = session.scalars(
        select(ReferenceSubject)
        .where(*filters)
        # Favourites first as an organisation aid, then most recently touched.
        .order_by(
            ReferenceSubject.favorite.desc(),
            ReferenceSubject.updated_at.desc(),
            ReferenceSubject.id,
        )
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), int(total or 0)


def deletion_impact(session: Session, subject: ReferenceSubject) -> DeletionImpact:
    """What deleting this subject would destroy, computed before anything is.

    Only artifacts nobody else references count as exclusive. A photograph
    showing two subjects belongs to both, and removing one of them is not
    permission to delete the picture.
    """

    asset_rows = session.scalars(
        select(ReferenceAsset).where(ReferenceAsset.reference_subject_id == subject.id)
    ).all()
    artifact_ids = [row.artifact_id for row in asset_rows]

    exclusive: list[str] = []
    for artifact_id in artifact_ids:
        others = session.scalar(
            select(func.count())
            .select_from(ReferenceAsset)
            .where(
                ReferenceAsset.artifact_id == artifact_id,
                ReferenceAsset.reference_subject_id != subject.id,
            )
        )
        if not others:
            exclusive.append(artifact_id)

    return DeletionImpact(
        reference_subject_id=subject.id,
        name=subject.name,
        asset_count=len(asset_rows),
        exclusive_artifact_ids=tuple(dict.fromkeys(exclusive)),
    )


@dataclass(frozen=True)
class SimilarAsset:
    """An image already in this subject that closely resembles a new one."""

    reference_asset_id: str
    artifact_id: str
    mean_absolute_difference: float


@dataclass(frozen=True)
class AttachedAsset:
    """A newly attached image, and anything the caller should know about it."""

    asset: ReferenceAsset
    # Near-duplicates are reported rather than refused. Two similar shots of one
    # subject are often deliberate - a second angle, a better exposure - so the
    # person adding them is the only one who can say which. What is worth saying
    # is that the set may now be weighted toward one look.
    similar: tuple[SimilarAsset, ...] = ()


def attach_asset(
    session: Session,
    subject: ReferenceSubject,
    *,
    artifact_id: str,
    caption: str | None = None,
    purpose: ReferencePurpose | str = ReferencePurpose.OTHER,
    view_label: str | None = None,
    read_bytes: Callable[[str], bytes] | None = None,
) -> AttachedAsset:
    """Add one image to a subject, refusing an exact repeat and flagging a near one.

    The exact case is refused because it cannot be what anyone wanted: the
    artifact store is content-addressed, so the same `artifact_id` is the same
    bytes, and a set holding one picture twice is silently weighted toward it.

    `read_bytes` is optional and injected rather than reached for. Without it the
    near-duplicate scan is skipped and the attachment still succeeds - similarity
    is advice, and advice that can fail must not be able to block the operation
    it advises on.
    """

    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise ReferenceError("that image is not in the artifact store")

    exact = session.scalar(
        select(ReferenceAsset.id).where(
            ReferenceAsset.reference_subject_id == subject.id,
            ReferenceAsset.artifact_id == artifact_id,
        )
    )
    if exact is not None:
        raise ReferenceError(
            f"{subject.name} already holds that exact image; adding it twice would "
            "weight the set toward one picture"
        )

    similar: list[SimilarAsset] = []
    if read_bytes is not None:
        candidates = session.scalars(
            select(ReferenceAsset)
            .where(ReferenceAsset.reference_subject_id == subject.id)
            .order_by(ReferenceAsset.sort_order, ReferenceAsset.id)
            .limit(MAX_SIMILARITY_CANDIDATES + 1)
        ).all()
        if len(candidates) > MAX_SIMILARITY_CANDIDATES:
            candidates = []
        try:
            incoming = read_bytes(artifact_id) if candidates else b""
        except (OSError, KeyError, ValueError):
            incoming = b""
        if incoming:
            for row in candidates:
                try:
                    other = read_bytes(row.artifact_id)
                except (OSError, KeyError, ValueError):
                    continue
                if not other:
                    continue
                difference = compare_images(incoming, other)
                # `comparable` false means one of them could not be read at all.
                # Reporting that as "identical" would be the worst possible
                # reading of "we could not tell".
                if difference.comparable and not difference.changed:
                    similar.append(
                        SimilarAsset(
                            reference_asset_id=row.id,
                            artifact_id=row.artifact_id,
                            mean_absolute_difference=round(difference.mean_absolute_difference, 4),
                        )
                    )

    highest_order = session.scalar(
        select(func.max(ReferenceAsset.sort_order)).where(
            ReferenceAsset.reference_subject_id == subject.id
        )
    )
    next_order = 0 if highest_order is None else highest_order + 1
    asset = ReferenceAsset(
        reference_subject_id=subject.id,
        artifact_id=artifact_id,
        caption=(caption or None),
        purpose=parse_purpose(purpose).value,
        view_label=(view_label or None),
        sort_order=next_order,
        validation_state=ValidationState.UNCHECKED.value,
    )
    session.add(asset)
    session.flush()
    return AttachedAsset(asset, tuple(similar))


def detach_asset(session: Session, subject: ReferenceSubject, *, asset_id: str) -> None:
    """Remove one image from a subject.

    Only the membership goes. The artifact is content-addressed and may be in
    use elsewhere, so ending this subject's claim on it is not permission to
    destroy the bytes.
    """

    asset = session.get(ReferenceAsset, asset_id)
    if asset is None or asset.reference_subject_id != subject.id:
        raise ReferenceError("that image is not attached to this reference")
    if subject.cover_artifact_id == asset.artifact_id:
        # The cover would otherwise point at an image the subject no longer has.
        subject.cover_artifact_id = None
    session.delete(asset)
    session.flush()


def set_cover(session: Session, subject: ReferenceSubject, *, artifact_id: str) -> ReferenceSubject:
    """Choose which of a subject's own images stands for it in a list.

    Restricted to images the subject already holds. A cover allowed to point at
    any artifact would be a second, weaker kind of membership - one the deletion
    impact does not count and the detach rule above does not clear - so a
    reference can only be represented by a picture it actually has.
    """

    held = session.scalar(
        select(ReferenceAsset).where(
            ReferenceAsset.reference_subject_id == subject.id,
            ReferenceAsset.artifact_id == artifact_id,
        )
    )
    if held is None:
        raise ReferenceError("a cover has to be one of this reference's own images")
    subject.cover_artifact_id = artifact_id
    session.flush()
    return subject


def clear_cover(session: Session, subject: ReferenceSubject) -> ReferenceSubject:
    """Stop an image standing for the subject without removing it from the set."""

    subject.cover_artifact_id = None
    session.flush()
    return subject
