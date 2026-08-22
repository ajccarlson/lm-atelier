from __future__ import annotations

from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowSelection,
    ProjectWorkflowSelection,
    Run,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkStep,
)
from local_lm.workflow_compatibility import AUTO_PROFILE_ID, compatibility_family_id


def _ready_image_family(
    session: Session,
    *,
    name: str,
    use_case: str = "",
) -> tuple[WorkflowFamily, WorkflowRevision]:
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family,
        variant_key="create",
        name=f"{name} create",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        api_graph_json={"node": {"class_type": "MockImage"}},
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="image")
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return family, revision


async def test_profile_and_chat_legacy_writes_are_mirrored_and_retired(
    client: AsyncClient,
) -> None:
    profile_response = await client.post(
        "/api/profiles",
        json={
            "name": "Compatibility image",
            "role": "image",
            "engine": "mock",
        },
    )
    assert profile_response.status_code == 201
    profile = profile_response.json()
    family_id = compatibility_family_id(profile["id"])
    chat = (await client.post("/api/chats", json={"title": "Compatibility"})).json()

    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": profile["id"]},
    )
    assert selected.status_code == 200
    with SessionLocal() as session:
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        assert selection.mode == "family"
        assert selection.workflow_family_id == family_id
        assert session.get(WorkflowFamily, family_id) is not None

    deleted = await client.delete(f"/api/profiles/{profile['id']}")
    assert deleted.status_code == 204
    refreshed = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert refreshed["active_image_profile_id"] == AUTO_PROFILE_ID
    with SessionLocal() as session:
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        assert selection.mode == "automatic"
        assert selection.workflow_family_id is None
        assert session.get(WorkflowFamily, family_id) is None


async def test_project_api_mirrors_exact_revision_pins_and_legacy_null(
    client: AsyncClient,
) -> None:
    revision_id = "wfrev_compatibility_api"
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            id="workflow_compatibility_api",
            name="Compatibility API",
            operation="text_to_image",
        )
        revision = WorkflowRevision(
            id=revision_id,
            definition=definition,
            version=1,
            trusted=True,
        )
        session.add_all([definition, revision])
        session.commit()

    created = await client.post(
        "/api/projects",
        json={
            "name": "Compatibility pin",
            "image_workflow_revision_id": revision_id,
        },
    )
    assert created.status_code == 201
    project = created.json()
    with SessionLocal() as session:
        selection = session.scalar(
            select(ProjectWorkflowSelection).where(
                ProjectWorkflowSelection.project_id == project["id"],
                ProjectWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        assert selection.mode == "revision"
        assert selection.workflow_revision_id == revision_id

    cleared = await client.patch(
        f"/api/projects/{project['id']}",
        json={"image_workflow_revision_id": None},
    )
    assert cleared.status_code == 200
    with SessionLocal() as session:
        assert (
            session.scalar(
                select(ProjectWorkflowSelection).where(
                    ProjectWorkflowSelection.project_id == project["id"],
                    ProjectWorkflowSelection.selector_capability == "image",
                )
            )
            is None
        )


async def test_explicit_chat_family_queues_its_current_operation_revision(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Workflow family"})).json()
    with SessionLocal() as session:
        family, revision = _ready_image_family(session, name="Selected family")
        session.commit()
        family_id = family.id
        revision_id = revision.id
    selected_response = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert selected_response.status_code == 200, selected_response.json()
    assert selected_response.json() == {
        "selector_capability": "image",
        "mode": "family",
        "workflow_family_id": family_id,
        "workflow_revision_id": None,
        "legacy_profile_id": None,
    }

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a blue ceramic bowl", "mode": "image"},
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    assert run["profile_id"] is None
    selected = run["provenance_json"]["model_selection"]
    assert selected["mode"] == "explicit"
    assert selected["compatibility_only"] is True
    assert selected["workflow_family_id"] == family_id
    assert selected["workflow_revision_id"] == revision_id
    witness = run["provenance_json"]["workflow"]
    assert witness["family_id"] == family_id
    assert witness["family_name"] == "Selected family"
    assert witness["definition_name"] == "Selected family create"
    assert witness["revision_id"] == revision_id
    assert witness["selection"] == {
        "source": "workflow_family",
        "mode": "explicit",
        "score": 0,
        "matched_terms": [],
        "fallback": False,
        "compatibility": False,
    }


async def test_automatic_chat_selection_ranks_ready_workflow_families(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Workflow Auto"})).json()
    with SessionLocal() as session:
        selected_family, selected_revision = _ready_image_family(
            session,
            name="Diagram specialist",
            use_case="architectural diagrams and technical illustration",
        )
        _ready_image_family(
            session,
            name="Portrait specialist",
            use_case="portraits and landscapes",
        )
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None and selection.mode == "automatic"
        session.commit()
        family_id = selected_family.id
        revision_id = selected_revision.id
    selected_response = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "automatic"},
    )
    assert selected_response.status_code == 200

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw an architectural diagram", "mode": "image"},
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    selected = run["provenance_json"]["model_selection"]
    assert selected["mode"] == "auto"
    assert selected["compatibility_only"] is True
    assert selected["workflow_family_id"] == family_id
    assert "architectural" in selected["matched_terms"]
    witness = run["provenance_json"]["workflow"]
    assert witness["family_id"] == family_id
    assert witness["revision_id"] == revision_id
    assert witness["selection"]["source"] == "workflow_family"
    assert witness["selection"]["mode"] == "auto"
    assert "architectural" in witness["selection"]["matched_terms"]


async def test_exact_revision_witness_does_not_claim_the_legacy_profile_ran(
    client: AsyncClient,
) -> None:
    profile_response = await client.post(
        "/api/profiles",
        json={
            "name": "Legacy portrait profile",
            "role": "image",
            "engine": "mock",
        },
    )
    assert profile_response.status_code == 201
    profile = profile_response.json()
    chat = (await client.post("/api/chats", json={"title": "Exact workflow"})).json()
    selected_profile = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": profile["id"]},
    )
    assert selected_profile.status_code == 200
    with SessionLocal() as session:
        family, revision = _ready_image_family(session, name="Exact revision family")
        session.commit()
        family_id = family.id
        definition_id = revision.workflow_id
        revision_id = revision.id

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Draw a blue bowl with the exact workflow",
            "mode": "image",
            "workflow_revision_id": revision_id,
        },
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["profile_id"] == profile["id"]
    assert run["workflow_revision_id"] == revision_id
    legacy_selection = run["provenance_json"]["model_selection"]
    assert legacy_selection["profile_id"] == profile["id"]
    assert legacy_selection["profile_name"] == "Legacy portrait profile"
    assert legacy_selection["compatibility_only"] is True
    witness = run["provenance_json"]["workflow"]
    assert witness["family_id"] == family_id
    assert witness["family_name"] == "Exact revision family"
    assert witness["definition_id"] == definition_id
    assert witness["definition_name"] == "Exact revision family create"
    assert witness["revision_id"] == revision_id
    assert witness["selection"] == {
        "source": "resolved_revision",
        "mode": "revision",
    }
    with SessionLocal() as session:
        queued_run = session.get(Run, run["id"])
        assert queued_run is not None and queued_run.work_step_id is not None
        work_step = session.get(WorkStep, queued_run.work_step_id)
        assert work_step is not None
        assert work_step.workflow_revision_id == queued_run.workflow_revision_id == revision_id


async def test_chat_family_overrides_the_project_family_without_changing_its_revision(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Family project"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Project family", "project_id": project["id"]},
        )
    ).json()
    with SessionLocal() as session:
        project_family, project_revision = _ready_image_family(
            session,
            name="Project workflow",
        )
        chat_family, chat_revision = _ready_image_family(
            session,
            name="Chat workflow",
        )
        session.commit()
        project_family_id = project_family.id
        project_revision_id = project_revision.id
        chat_family_id = chat_family.id
        chat_revision_id = chat_revision.id
    chat_selected = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": chat_family_id},
    )
    assert chat_selected.status_code == 200
    project_selected = await client.put(
        f"/api/projects/{project['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": project_family_id},
    )
    assert project_selected.status_code == 200, project_selected.json()

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a project diagram", "mode": "image"},
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == chat_revision_id
    assert run["workflow_revision_id"] != project_revision_id
    assert run["provenance_json"]["model_selection"]["workflow_family_id"] == chat_family_id

    chat_inherited = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "default"},
    )
    assert chat_inherited.status_code == 200, chat_inherited.json()
    inherited = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw the inherited project diagram", "mode": "image"},
    )
    assert inherited.status_code == 202, inherited.json()
    inherited_run = inherited.json()["run"]
    assert inherited_run["workflow_revision_id"] == project_revision_id
    assert (
        inherited_run["provenance_json"]["model_selection"]["workflow_family_id"]
        == project_family_id
    )

    text_accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Summarize the diagram", "mode": "text"},
    )

    assert text_accepted.status_code == 202, text_accepted.json()


async def test_workflow_family_catalog_reports_variants_preferences_and_readiness(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        family, revision = _ready_image_family(
            session,
            name="Catalog family",
            use_case="product illustrations",
        )
        session.commit()
        family_id = family.id
        revision_id = revision.id

    response = await client.get("/api/workflow-families?selector_capability=image")

    assert response.status_code == 200
    card = next(item for item in response.json() if item["id"] == family_id)
    assert card["use_case"] == "product illustrations"
    assert card["compatibility"] is False
    assert card["preferences"] == [
        {
            "selector_capability": "image",
            "enabled": True,
            "is_default": False,
            "sort_order": 0,
        }
    ]
    assert card["variants"][0] == {
        "id": card["variants"][0]["id"],
        "variant_key": "create",
        "name": "Catalog family create",
        "operation": "text_to_image",
        "current_revision_id": revision_id,
        "current_revision_version": 1,
        "engine": "mock",
        "capabilities": [],
        "trusted": True,
        "readiness": "ready",
        "readiness_reason": None,
        "setup_resolution": None,
        "install_offer_id": None,
    }


async def test_selector_endpoints_clear_to_default_and_inherit(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Selector reset"})).json()
    automatic = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "automatic"},
    )
    assert automatic.status_code == 200
    default = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "default"},
    )
    assert default.status_code == 200
    assert default.json()["mode"] == "default"

    project = (await client.post("/api/projects", json={"name": "Selector reset"})).json()
    inherited = await client.put(
        f"/api/projects/{project['id']}/workflow-selections/image",
        json={"mode": "inherit"},
    )
    assert inherited.status_code == 200
    assert inherited.json()["mode"] == "inherit"

    invalid = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "automatic", "workflow_family_id": "wffamily_typo"},
    )
    assert invalid.status_code == 422
