from __future__ import annotations

import io
from collections.abc import Generator

import pytest
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import Artifact, ReferenceAsset, ReferenceSubject
from local_lm.reference_library import (
    MAX_PAGE,
    MAX_SIMILARITY_CANDIDATES,
    attach_asset,
    clear_cover,
    create_subject,
    deletion_impact,
    detach_asset,
    list_subjects,
    rename_subject,
    set_archived,
    set_cover,
    set_details,
    set_favorite,
)
from local_lm.references import MAX_SLUG, ReferenceError, ReferenceKind, valid_mention_slug


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")

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


def _attach(session: Session, subject: ReferenceSubject, artifact: Artifact) -> None:
    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    session.flush()


def test_a_subject_gets_a_mention_derived_from_its_name(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind=ReferenceKind.PERSON)
    assert subject.mention_slug == "ada-lovelace"
    assert subject.kind == "person"


def test_two_people_may_share_a_name_but_not_a_mention(session: Session) -> None:
    """The collision is usually two real people called the same thing, not a
    mistake. Refusing would push the user into inventing a worse name."""

    first = create_subject(session, name="Ada Lovelace", kind="person")
    second = create_subject(session, name="Ada Lovelace", kind="person")
    assert first.name == second.name
    assert (first.mention_slug, second.mention_slug) == ("ada-lovelace", "ada-lovelace-2")


def test_a_maximum_length_derived_mention_reserves_room_for_each_suffix(
    session: Session,
) -> None:
    name = "a" * MAX_SLUG
    subjects = [create_subject(session, name=name, kind="person") for _ in range(11)]
    slugs = [subject.mention_slug for subject in subjects]

    assert slugs[:2] == [name, "a" * (MAX_SLUG - 2) + "-2"]
    assert slugs[8:11] == [
        "a" * (MAX_SLUG - 2) + "-9",
        "a" * (MAX_SLUG - 3) + "-10",
        "a" * (MAX_SLUG - 3) + "-11",
    ]
    assert all(len(slug) <= MAX_SLUG and valid_mention_slug(slug) for slug in slugs)


def test_a_mention_the_user_chose_is_refused_rather_than_renumbered(session: Session) -> None:
    """Silently handing back a different mention than the one asked for would
    mean the user's chats reference something that does not exist."""

    create_subject(session, name="Ada", kind="person", mention_slug="ada")
    with pytest.raises(ReferenceError) as caught:
        create_subject(session, name="Someone else", kind="person", mention_slug="ada")
    assert str(caught.value) == "that mention is already in use"


def test_an_explicit_mention_honors_the_exact_length_boundary(session: Session) -> None:
    boundary = "a" * MAX_SLUG
    subject = create_subject(session, name="Ada", kind="person", mention_slug=boundary)
    assert subject.mention_slug == boundary

    private_value = "z" * (MAX_SLUG + 1)
    with pytest.raises(ReferenceError) as caught:
        create_subject(session, name="Grace", kind="person", mention_slug=private_value)
    assert str(caught.value) == "a mention must use lowercase letters, digits and hyphens"
    assert private_value not in str(caught.value)
    assert session.query(ReferenceSubject).count() == 1


@pytest.mark.parametrize("bad", ["Ada Lovelace", "ada_lovelace", "-ada", "", "a" * 80])
def test_an_unusable_mention_is_refused(session: Session, bad: str) -> None:
    with pytest.raises(ReferenceError):
        create_subject(session, name="Ada", kind="person", mention_slug=bad)


def test_collision_exhaustion_is_bounded_and_leaves_no_partial_subject(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.reference_library.MAX_SLUG_COLLISION_SUFFIX", 3)
    for _ in range(3):
        create_subject(session, name="Ada", kind="person")

    with pytest.raises(ReferenceError) as caught:
        create_subject(session, name="Ada", kind="person")

    assert str(caught.value) == "a unique mention could not be assigned"
    assert session.query(ReferenceSubject).count() == 3


def test_collision_exhaustion_does_not_partially_rename_a_subject(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.reference_library.MAX_SLUG_COLLISION_SUFFIX", 3)
    for _ in range(3):
        create_subject(session, name="Grace", kind="person")
    subject = create_subject(session, name="Ada", kind="person")

    with pytest.raises(ReferenceError) as caught:
        rename_subject(session, subject, name="Grace", follow_mention=True)

    assert str(caught.value) == "a unique mention could not be assigned"
    assert subject.name == "Ada"
    assert subject.mention_slug == "ada"


def test_renaming_does_not_move_the_mention_by_default(session: Session) -> None:
    """A live chat draft may already hold the mention. Changing the addressing
    token underneath someone mid-sentence is worse than letting the two drift."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    rename_subject(session, subject, name="Augusta Ada King")
    assert subject.name == "Augusta Ada King"
    assert subject.mention_slug == "ada-lovelace"

    rename_subject(session, subject, name="Augusta Ada King", follow_mention=True)
    assert subject.mention_slug == "augusta-ada-king"


def test_a_following_mention_still_avoids_a_collision(session: Session) -> None:
    create_subject(session, name="Grace Hopper", kind="person")
    subject = create_subject(session, name="Ada", kind="person")
    rename_subject(session, subject, name="Grace Hopper", follow_mention=True)
    assert subject.mention_slug == "grace-hopper-2"


def test_archiving_hides_a_subject_without_destroying_it(session: Session) -> None:
    """Archiving is the removal a user wants; it has to be reversible."""

    subject = create_subject(session, name="Ada", kind="person")
    set_archived(session, subject, True)
    visible, total = list_subjects(session)
    assert visible == [] and total == 0

    hidden, hidden_total = list_subjects(session, include_archived=True)
    assert [item.id for item in hidden] == [subject.id] and hidden_total == 1

    set_archived(session, subject, False)
    restored, _ = list_subjects(session)
    assert [item.id for item in restored] == [subject.id]


def test_favourites_sort_first_without_being_a_quality_signal(session: Session) -> None:
    """Organisation only - it says someone wanted this near the top, not that
    its images are good."""

    plain = create_subject(session, name="Bravo", kind="person")
    marked = create_subject(session, name="Alpha", kind="person")
    set_favorite(session, marked, True)
    listed, _ = list_subjects(session)
    assert listed[0].id == marked.id
    assert plain.favorite is False


def test_listing_filters_by_kind_and_name(session: Session) -> None:
    create_subject(session, name="Ada Lovelace", kind="person")
    create_subject(session, name="Brass Lamp", kind="object")

    people, total = list_subjects(session, kind="person")
    assert total == 1 and people[0].name == "Ada Lovelace"

    found, _ = list_subjects(session, search="LAMP")
    assert [item.name for item in found] == ["Brass Lamp"]

    assert list_subjects(session, search="nothing here")[1] == 0


def test_a_page_is_bounded(session: Session) -> None:
    for index in range(5):
        create_subject(session, name=f"Subject {index}", kind="person")
    page, total = list_subjects(session, limit=2)
    assert len(page) == 2 and total == 5
    assert len(list_subjects(session, limit=2, offset=4)[0]) == 1

    for bad in (0, -1, MAX_PAGE + 1):
        with pytest.raises(ReferenceError):
            list_subjects(session, limit=bad)
    with pytest.raises(ReferenceError):
        list_subjects(session, offset=-1)


def test_deletion_impact_counts_only_what_is_this_subject_s_to_lose(
    session: Session,
) -> None:
    """A photograph showing two subjects belongs to both. Removing one of them
    is not permission to delete the picture."""

    subject = create_subject(session, name="Ada", kind="person")
    other = create_subject(session, name="Grace", kind="person")

    private_image = _artifact(session)
    shared_image = _artifact(session)
    _attach(session, subject, private_image)
    _attach(session, subject, shared_image)
    _attach(session, other, shared_image)

    impact = deletion_impact(session, subject)
    assert impact.asset_count == 2
    assert impact.exclusive_artifact_ids == (private_image.id,)
    assert impact.shared_artifact_count == 1
    assert impact.name == "Ada"


def test_deletion_impact_is_computed_before_anything_is_destroyed(
    session: Session,
) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    _attach(session, subject, _artifact(session))
    session.commit()

    deletion_impact(session, subject)

    assert session.query(ReferenceSubject).count() == 1
    assert session.query(ReferenceAsset).count() == 1
    assert session.query(Artifact).count() == 1


def _png(shade: int, size: tuple[int, int] = (48, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(buffer, format="PNG")
    return buffer.getvalue()


def _named_artifact(session: Session, marker: str) -> Artifact:
    artifact = Artifact(
        id=f"art_{marker}",
        sha256=f"{abs(hash(marker)):064x}"[:64],
        kind="image",
        media_type="image/png",
        size_bytes=1,
        relative_path=f"{marker}/a",
    )
    session.add(artifact)
    session.flush()
    return artifact


def test_the_same_image_twice_is_refused_with_a_reason(session: Session) -> None:
    """The store is content-addressed, so the same id is the same bytes. A set
    holding one picture twice is silently weighted toward it."""

    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "one")
    attach_asset(session, subject, artifact_id="art_one")

    with pytest.raises(ReferenceError) as caught:
        attach_asset(session, subject, artifact_id="art_one")
    assert "already holds that exact image" in str(caught.value)
    assert session.query(ReferenceAsset).count() == 1


def test_a_near_duplicate_is_reported_rather_than_refused(session: Session) -> None:
    """Two similar shots are often deliberate - a second angle, a better
    exposure - so the person adding them is the only one who can decide."""

    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "first")
    _named_artifact(session, "second")
    images = {"art_first": _png(120), "art_second": _png(120)}

    attach_asset(session, subject, artifact_id="art_first", read_bytes=images.__getitem__)
    result = attach_asset(session, subject, artifact_id="art_second", read_bytes=images.__getitem__)

    assert result.asset.id, "the attachment still happened"
    assert len(result.similar) == 1
    assert result.similar[0].artifact_id == "art_first"
    assert session.query(ReferenceAsset).count() == 2


def test_a_genuinely_different_image_is_not_flagged(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "dark")
    _named_artifact(session, "light")
    images = {"art_dark": _png(10), "art_light": _png(240)}

    attach_asset(session, subject, artifact_id="art_dark", read_bytes=images.__getitem__)
    result = attach_asset(session, subject, artifact_id="art_light", read_bytes=images.__getitem__)
    assert result.similar == ()


def test_similarity_is_advice_and_cannot_block_the_attachment(session: Session) -> None:
    """Advice that can fail must not be able to block the operation it advises
    on. An unreadable neighbour is skipped, not treated as identical."""

    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "good")
    _named_artifact(session, "broken")
    images = {"art_good": _png(120), "art_broken": b"not an image"}

    attach_asset(session, subject, artifact_id="art_good", read_bytes=images.__getitem__)
    result = attach_asset(session, subject, artifact_id="art_broken", read_bytes=images.__getitem__)
    assert result.asset.id
    assert result.similar == (), "unreadable is not a similarity claim"

    # A reader that raises outright is also survivable.
    _named_artifact(session, "third")

    def explode(_artifact_id: str) -> bytes:
        raise OSError("disk gone")

    assert attach_asset(session, subject, artifact_id="art_third", read_bytes=explode).asset.id


@pytest.mark.parametrize("failure", [OSError, KeyError, ValueError])
@pytest.mark.parametrize("unavailable", ["incoming", "existing"])
def test_unavailable_similarity_evidence_never_blocks_or_reviews_an_attachment(
    session: Session,
    failure: type[Exception],
    unavailable: str,
) -> None:
    private_detail = "private retained-artifact authority detail"
    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "existing")
    _named_artifact(session, "incoming")
    existing = attach_asset(session, subject, artifact_id="art_existing").asset
    images = {"art_existing": _png(120), "art_incoming": _png(120)}

    def read(artifact_id: str) -> bytes:
        if artifact_id == f"art_{unavailable}":
            raise failure(private_detail)
        return images[artifact_id]

    result = attach_asset(
        session,
        subject,
        artifact_id="art_incoming",
        read_bytes=read,
    )
    rows = list(
        session.query(ReferenceAsset)
        .filter(ReferenceAsset.reference_subject_id == subject.id)
        .order_by(ReferenceAsset.sort_order)
    )

    assert result.similar == ()
    assert [row.artifact_id for row in rows] == ["art_existing", "art_incoming"]
    assert result.asset is rows[1]
    assert existing is rows[0]
    assert all(row.validation_state == "unchecked" for row in rows)
    assert all(row.validation_reasons_json == [] for row in rows)
    assert private_detail not in repr(result)


def test_without_a_reader_the_scan_is_skipped_not_faked(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "a1")
    _named_artifact(session, "a2")
    attach_asset(session, subject, artifact_id="art_a1")
    assert attach_asset(session, subject, artifact_id="art_a2").similar == ()


def _fill_similarity_candidates(
    session: Session,
    subject: ReferenceSubject,
    count: int,
) -> list[ReferenceAsset]:
    rows: list[ReferenceAsset] = []
    for index in range(count):
        artifact = _named_artifact(session, f"candidate_{index:03d}")
        rows.append(
            ReferenceAsset(
                reference_subject_id=subject.id,
                artifact_id=artifact.id,
                sort_order=index,
            )
        )
    session.add_all(rows)
    session.flush()
    return rows


def test_similarity_scan_accepts_the_exact_candidate_boundary(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    _fill_similarity_candidates(session, subject, MAX_SIMILARITY_CANDIDATES)
    incoming = _named_artifact(session, "boundary_incoming")
    image = _png(120)
    calls: list[str] = []

    def read(artifact_id: str) -> bytes:
        calls.append(artifact_id)
        return image

    result = attach_asset(session, subject, artifact_id=incoming.id, read_bytes=read)

    assert len(result.similar) == MAX_SIMILARITY_CANDIDATES
    assert calls[0] == incoming.id
    assert calls[1:] == [f"art_candidate_{index:03d}" for index in range(64)]
    assert result.asset.sort_order == MAX_SIMILARITY_CANDIDATES


def test_similarity_scan_plus_one_skips_advice_and_still_attaches(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    existing = _fill_similarity_candidates(session, subject, MAX_SIMILARITY_CANDIDATES + 1)
    incoming = _named_artifact(session, "oversize_incoming")
    calls: list[str] = []

    def read(artifact_id: str) -> bytes:
        calls.append(artifact_id)
        return _png(120)

    result = attach_asset(session, subject, artifact_id=incoming.id, read_bytes=read)

    assert result.similar == ()
    assert calls == []
    rows = session.query(ReferenceAsset).order_by(ReferenceAsset.sort_order).all()
    assert [row.id for row in rows[:-1]] == [row.id for row in existing]
    assert rows[-1].id == result.asset.id
    assert rows[-1].artifact_id == incoming.id
    assert rows[-1].sort_order == MAX_SIMILARITY_CANDIDATES + 1


def test_exact_duplicate_outside_similarity_window_is_still_authoritative(
    session: Session,
) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    existing = _fill_similarity_candidates(session, subject, MAX_SIMILARITY_CANDIDATES + 1)
    duplicate = existing[-1]
    calls: list[str] = []

    def read(artifact_id: str) -> bytes:
        calls.append(artifact_id)
        return _png(120)

    with pytest.raises(ReferenceError, match="already holds that exact image"):
        attach_asset(
            session,
            subject,
            artifact_id=duplicate.artifact_id,
            read_bytes=read,
        )

    assert calls == []
    assert session.query(ReferenceAsset).count() == MAX_SIMILARITY_CANDIDATES + 1


def test_skipping_similarity_uses_full_subject_for_next_order(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    existing = _fill_similarity_candidates(session, subject, MAX_SIMILARITY_CANDIDATES + 1)
    existing[-1].sort_order = 500
    incoming = _named_artifact(session, "unscanned_incoming")
    session.flush()

    result = attach_asset(session, subject, artifact_id=incoming.id)

    assert result.similar == ()
    assert result.asset.sort_order == 501
    assert session.query(ReferenceAsset).count() == MAX_SIMILARITY_CANDIDATES + 2


def test_an_image_outside_the_store_cannot_be_attached(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    with pytest.raises(ReferenceError):
        attach_asset(session, subject, artifact_id="art_nowhere")


def test_attachments_keep_the_order_they_arrived_in(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    for marker in ("p1", "p2", "p3"):
        _named_artifact(session, marker)
        attach_asset(session, subject, artifact_id=f"art_{marker}")
    session.refresh(subject)
    assert [row.sort_order for row in subject.assets] == [0, 1, 2]


def test_detaching_takes_the_membership_and_leaves_the_bytes(session: Session) -> None:
    """The artifact may be in use elsewhere, so ending this subject's claim is
    not permission to destroy it."""

    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "keep")
    attached = attach_asset(session, subject, artifact_id="art_keep").asset

    detach_asset(session, subject, asset_id=attached.id)
    assert session.query(ReferenceAsset).count() == 0
    assert session.get(Artifact, "art_keep") is not None


def test_detaching_the_cover_clears_it_rather_than_dangling(session: Session) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    _named_artifact(session, "cover")
    attached = attach_asset(session, subject, artifact_id="art_cover").asset
    subject.cover_artifact_id = "art_cover"
    session.flush()

    detach_asset(session, subject, asset_id=attached.id)
    assert subject.cover_artifact_id is None


def test_detaching_something_that_belongs_elsewhere_is_refused(session: Session) -> None:
    first = create_subject(session, name="Ada", kind="person")
    second = create_subject(session, name="Grace", kind="person")
    _named_artifact(session, "shared")
    attached = attach_asset(session, first, artifact_id="art_shared").asset

    with pytest.raises(ReferenceError):
        detach_asset(session, second, asset_id=attached.id)
    assert session.query(ReferenceAsset).count() == 1


def test_a_cover_has_to_be_one_of_the_subjects_own_images(session: Session) -> None:
    """A cover allowed to point anywhere would be a second, weaker membership -
    one deletion impact does not count and detaching does not clear."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    stranger = _artifact(session)

    with pytest.raises(ReferenceError, match="one of this reference's own images"):
        set_cover(session, subject, artifact_id=stranger.id)
    assert subject.cover_artifact_id is None


def test_an_image_the_subject_holds_can_stand_for_it(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")
    artifact = _artifact(session)
    _attach(session, subject, artifact)

    set_cover(session, subject, artifact_id=artifact.id)

    assert subject.cover_artifact_id == artifact.id


def test_clearing_a_cover_keeps_the_image(session: Session) -> None:
    """Removing what represents a subject is not removing what it holds."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    artifact = _artifact(session)
    _attach(session, subject, artifact)
    set_cover(session, subject, artifact_id=artifact.id)

    clear_cover(session, subject)

    assert subject.cover_artifact_id is None
    assert subject.assets


def test_detaching_the_cover_image_clears_the_cover(session: Session) -> None:
    """Otherwise the cover points at a picture the subject no longer has."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    artifact = _artifact(session)
    _attach(session, subject, artifact)
    set_cover(session, subject, artifact_id=artifact.id)
    asset = subject.assets[0]

    detach_asset(session, subject, asset_id=asset.id)

    assert subject.cover_artifact_id is None


def test_a_subject_is_found_by_an_alias_not_only_its_name(session: Session) -> None:
    """The whole reason a subject has aliases. A search that only knows the
    display name leaves them decorative, which is what they were."""

    create_subject(session, name="Ada Lovelace", kind="person", aliases=["Countess Lovelace"])

    found, total = list_subjects(session, search="countess")

    assert total == 1
    assert [one.name for one in found] == ["Ada Lovelace"]


def test_searching_still_matches_the_name(session: Session) -> None:
    create_subject(session, name="Ada Lovelace", kind="person", aliases=["AAL"])
    create_subject(session, name="Grace Hopper", kind="person")

    found, _ = list_subjects(session, search="hopper")

    assert [one.name for one in found] == ["Grace Hopper"]


def test_details_can_be_corrected_after_creation(session: Session) -> None:
    """A typo in a description used to be permanent."""

    subject = create_subject(
        session, name="Ada Lovelace", kind="person", description="Mathemetician"
    )

    set_details(session, subject, description="Mathematician", aliases=["Countess Lovelace"])

    assert subject.description == "Mathematician"
    assert subject.aliases_json == ["Countess Lovelace"]


def test_creation_and_correction_use_the_same_detail_rules(session: Session) -> None:
    subject = create_subject(
        session,
        name="Ada Lovelace",
        kind="person",
        description="  Mathematician  ",
        aliases=["Countess", " countess ", ""],
        tags=["historic", " HISTORIC "],
    )

    assert subject.description == "Mathematician"
    assert subject.aliases_json == ["Countess"]
    assert subject.tags_json == ["historic"]


def test_omitting_a_field_leaves_it_alone_and_emptying_it_clears_it(session: Session) -> None:
    """Two different instructions that one nullable value could not carry."""

    subject = create_subject(
        session, name="Ada Lovelace", kind="person", description="Mathematician", tags=["historic"]
    )

    set_details(session, subject, aliases=["AAL"])
    assert subject.description == "Mathematician", "an omitted field is not an instruction"
    assert subject.tags_json == ["historic"]

    set_details(session, subject, description="", tags=[])
    assert subject.description is None
    assert subject.tags_json == []
    assert subject.aliases_json == ["AAL"], "clearing one field does not clear another"


def test_aliases_differing_only_in_case_are_one_alias(session: Session) -> None:
    """Both would have to be matched by anything resolving a name later, and to
    a reader they are the same word twice."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")

    set_details(session, subject, aliases=["Countess", "  countess  ", "COUNTESS", ""])

    assert subject.aliases_json == ["Countess"]


def test_too_many_aliases_are_refused(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")

    with pytest.raises(ReferenceError, match="at most"):
        set_details(session, subject, aliases=[f"name-{index}" for index in range(33)])
