from __future__ import annotations

import io

import pytest
from httpx2 import AsyncClient
from PIL import Image


def _png(size: tuple[int, int] = (12, 10)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _attached(client: AsyncClient, content: bytes) -> tuple[str, str]:
    subject = (
        await client.post("/api/references", json={"name": "Reviewed", "kind": "person"})
    ).json()
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("review.png", content, "image/png")},
    )
    assert uploaded.status_code in (200, 201), uploaded.text
    attached = await client.post(
        f"/api/references/{subject['id']}/assets",
        json={"artifact_id": uploaded.json()["id"], "purpose": "identity"},
    )
    assert attached.status_code == 201, attached.text
    return subject["id"], attached.json()["asset"]["id"]


@pytest.mark.anyio
async def test_api_review_exact_retry_and_conflicts_are_explicit(client: AsyncClient) -> None:
    subject_id, asset_id = await _attached(client, _png())
    route = f"/api/references/{subject_id}/assets/{asset_id}/review"
    decision = {
        "expected_state": "unchecked",
        "expected_version": 1,
        "decision": "weak",
        "reasons": ["soft focus"],
    }

    first = await client.post(route, json=decision)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["idempotent"] is False
    assert body["asset"]["validation_state"] == "weak"
    assert body["asset"]["validation_reasons_json"] == ["soft focus"]
    assert (body["asset"]["width"], body["asset"]["height"]) == (12, 10)
    assert body["asset"]["review_version"] == 2
    assert body["review"]["reviewer_kind"] == "local-human"
    assert body["review"]["id"] == f"refreview:sha256:{body['review']['decision_sha256']}"

    retry = await client.post(route, json=decision)
    assert retry.status_code == 200
    assert retry.json()["idempotent"] is True
    assert retry.json()["review"]["id"] == body["review"]["id"]

    conflicting = await client.post(route, json={**decision, "decision": "usable", "reasons": []})
    assert conflicting.status_code == 409
    assert conflicting.json()["code"] == "reference-asset-review-conflict"
    assert conflicting.json()["current_state"] == "weak"
    assert conflicting.json()["current_version"] == 2

    rereview = await client.post(route, json={**decision, "expected_version": 2})
    assert rereview.status_code == 409
    assert rereview.json()["code"] == "reference-asset-review-conflict"


@pytest.mark.anyio
async def test_api_corrupt_image_refuses_without_changing_asset(client: AsyncClient) -> None:
    subject_id, asset_id = await _attached(client, _png()[:-12])
    route = f"/api/references/{subject_id}/assets/{asset_id}/review"

    refused = await client.post(
        route,
        json={
            "expected_state": "unchecked",
            "expected_version": 1,
            "decision": "usable",
            "reasons": [],
        },
    )
    assert refused.status_code == 422
    assert refused.json()["code"] == "reference-asset-review-invalid"

    listed = await client.get(f"/api/references/{subject_id}/assets")
    asset = listed.json()[0]
    assert asset["validation_state"] == "unchecked"
    assert asset["validation_reasons_json"] == []
    assert asset["width"] is None and asset["height"] is None
    assert asset["review_version"] == 1
