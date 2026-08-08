from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import FastAPI, File, Response, UploadFile
from httpx2 import ASGITransport, AsyncClient

from local_lm.config import Settings
from local_lm.main import create_app
from local_lm.security import (
    MAX_JSON_BODY_BYTES,
    MULTIPART_OVERHEAD_BYTES,
    JsonBodyLimitMiddleware,
    SecurityHeadersMiddleware,
    SessionSecurity,
    UploadBodyLimitMiddleware,
)


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_all_application_subprocesses_use_an_explicit_environment() -> None:
    package_root = Path(__file__).resolve().parents[1] / "local_lm"
    subprocess_calls = {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.run",
    }
    missing: list[str] = []

    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _qualified_name(node.func) not in subprocess_calls:
                continue
            if not any(keyword.arg == "env" for keyword in node.keywords):
                missing.append(f"{source_path.relative_to(package_root)}:{node.lineno}")

    assert missing == [], "subprocesses without an explicit safe environment: " + ", ".join(missing)


def test_public_preview_rejects_non_loopback_binding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOCAL_LM_ALLOW_LAN", "true")
    settings = Settings(data_dir=tmp_path / "lan", host="0.0.0.0")

    with pytest.raises(ValueError, match="loopback binding only"):
        settings.prepare()


@pytest.mark.parametrize(
    "worker_url",
    [
        "https://127.0.0.1:12341",
        "http://example.com:12341",
        "http://127.0.0.1:12341/api",
        "http://user:secret@127.0.0.1:12341",
        "http://127.0.0.1:12341?token=secret",
    ],
)
def test_worker_urls_are_restricted_to_plain_loopback_origins(worker_url: str) -> None:
    with pytest.raises(ValueError, match="worker URLs"):
        Settings(llama_url=worker_url)


def test_worker_url_normalizes_a_trailing_slash() -> None:
    settings = Settings(
        llama_url="http://localhost:12341/",
        comfy_url="http://[::1]:8188/",
    )

    assert settings.llama_url == "http://localhost:12341"
    assert settings.comfy_url == "http://[::1]:8188"


async def test_non_dev_api_requires_cookie_and_csrf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "secure", dev=False)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            assert (await client.get("/api/projects")).status_code == 401
            session = await client.post("/api/session")
            assert session.status_code == 200
            csrf = session.json()["csrf_token"]
            assert session.json()["event_sequence"] == 0
            assert isinstance(session.json()["event_epoch"], str)
            assert session.json()["event_epoch"]
            assert "frame-ancestors 'none'" in session.headers["content-security-policy"]
            assert session.headers["referrer-policy"] == "no-referrer"
            assert (await client.get("/api/projects")).status_code == 200
            denied = await client.post("/api/projects", json={"name": "Denied"})
            assert denied.status_code == 403
            allowed = await client.post(
                "/api/projects",
                json={"name": "Allowed"},
                headers={"x-local-lm-csrf": csrf},
            )
            assert allowed.status_code == 201


async def test_browser_origins_are_limited_to_the_local_application(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "origins", dev=False, port=12340)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            rejected = await client.post(
                "/api/session",
                headers={"origin": "https://malicious.example"},
            )
            assert rejected.status_code == 403
            assert rejected.json()["detail"] == "untrusted browser origin"
            assert rejected.headers["x-frame-options"] == "DENY"
            assert "frame-ancestors 'none'" in rejected.headers["content-security-policy"]

            allowed = await client.post(
                "/api/session",
                headers={"origin": "http://127.0.0.1:12340"},
            )
            assert allowed.status_code == 200


async def test_release_mode_excludes_developer_hosts_and_openapi(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "release-surface", dev=False)
    app = create_app(settings)
    assert app.openapi_url is None

    accepted_statuses: list[int] = []
    rejected_responses: list[tuple[int, str]] = []
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        for host in ("127.0.0.1", "localhost", "[::1]"):
            async with AsyncClient(transport=transport, base_url=f"http://{host}:12340") as client:
                accepted_statuses.append((await client.get("/api/ready")).status_code)
        for host in ("testserver", "testclient", "malicious.example"):
            async with AsyncClient(transport=transport, base_url=f"http://{host}:12340") as client:
                rejected = await client.get("/api/ready")
                rejected_responses.append((rejected.status_code, rejected.text))

    assert accepted_statuses == [200, 200, 200]
    assert rejected_responses == [(400, "Invalid host header")] * 3


async def test_developer_mode_retains_test_hosts_and_openapi(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "developer-surface",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)
    assert app.openapi_url == "/openapi.json"

    responses: list[tuple[int, str]] = []
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        for host in ("testserver", "testclient"):
            async with AsyncClient(transport=transport, base_url=f"http://{host}:12340") as client:
                response = await client.get("/openapi.json")
                responses.append((response.status_code, response.json()["info"]["title"]))

    assert responses == [(200, "LM Atelier API")] * 2


def test_websocket_origin_policy_allows_non_browser_clients_and_local_ui(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    security = SessionSecurity(
        Settings(data_dir=tmp_path / "websocket-origins", dev=False, port=12340)
    )

    assert security._valid_origin(None)
    assert security._valid_origin("http://localhost:12340")
    assert security._valid_origin("http://[::1]:12340")
    assert not security._valid_origin("null")
    assert not security._valid_origin("https://127.0.0.1:12340")
    assert not security._valid_origin("http://127.0.0.1.attacker.test:12340")


async def test_json_request_body_limit_covers_streamed_bodies_without_content_length(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        data_dir=tmp_path / "json-body-limit",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)

    async def oversized_body():  # type: ignore[no-untyped-def]
        yield b'{"name":"'
        yield b"x" * MAX_JSON_BODY_BYTES
        yield b'"}'

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            session = await client.post("/api/session")
            response = await client.post(
                "/api/projects",
                content=oversized_body(),
                headers={
                    "content-type": "application/json",
                    "x-local-lm-csrf": session.json()["csrf_token"],
                },
            )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        f"JSON request body exceeds the {MAX_JSON_BODY_BYTES}-byte limit"
    )


async def test_json_body_replay_waits_on_the_real_connection_after_the_body() -> None:
    downstream_receive_started = asyncio.Event()
    allow_disconnect = asyncio.Event()

    async def downstream(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        body = await receive()
        assert body == {
            "type": "http.request",
            "body": b'{"ok":true}',
            "more_body": False,
        }
        pending_receive = asyncio.create_task(receive())
        await asyncio.wait_for(downstream_receive_started.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert not pending_receive.done()
        allow_disconnect.set()
        assert await pending_receive == {"type": "http.disconnect"}
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = JsonBodyLimitMiddleware(downstream)
    source_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    source_messages.put_nowait(
        {
            "type": "http.request",
            "body": b'{"ok":true}',
            "more_body": False,
        }
    )
    receive_count = 0

    async def receive() -> dict[str, object]:
        nonlocal receive_count
        receive_count += 1
        if receive_count > 1:
            downstream_receive_started.set()
            await allow_disconnect.wait()
            return {"type": "http.disconnect"}
        message = await source_messages.get()
        return message

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "path": "/api/example",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,  # type: ignore[arg-type]
        send,  # type: ignore[arg-type]
    )

    assert sent == [
        {"type": "http.response.start", "status": 204, "headers": []},
        {"type": "http.response.body", "body": b""},
    ]


async def test_upload_limit_rejects_streamed_multipart_before_endpoint_runs() -> None:
    app = FastAPI()
    endpoint_ran = False

    @app.post("/api/artifacts")
    async def consume_upload(file: Annotated[UploadFile, File()]) -> dict[str, str]:
        nonlocal endpoint_ran
        endpoint_ran = True
        return {"name": file.filename or ""}

    app.add_middleware(
        UploadBodyLimitMiddleware,
        artifact_max_bytes=8,
        project_max_bytes=16,
    )

    async def oversized_body():  # type: ignore[no-untyped-def]
        yield (
            b"--test\r\n"
            b'Content-Disposition: form-data; name="file"; filename="oversized.bin"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
        )
        yield b"x" * (MULTIPART_OVERHEAD_BYTES + 9)
        yield b"\r\n--test--\r\n"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/artifacts",
            content=oversized_body(),
            headers={"content-type": "multipart/form-data; boundary=test"},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "upload request exceeds the 8-byte limit"
    assert endpoint_ran is False


async def test_security_headers_block_remote_active_content_and_preserve_stricter_csp() -> None:
    app = FastAPI()

    @app.get("/")
    async def index() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/artifact")
    async def artifact() -> Response:
        return Response(
            "image",
            headers={"Content-Security-Policy": "sandbox; default-src 'none'"},
        )

    app.add_middleware(SecurityHeadersMiddleware)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")
        artifact_response = await client.get("/artifact")

    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "img-src 'self' data: blob:" in csp
    connect_directive = next(
        directive.strip()
        for directive in csp.split(";")
        if directive.strip().startswith("connect-src")
    )
    assert connect_directive == "connect-src 'self' ws://127.0.0.1:* ws://localhost:*"
    assert "frame-ancestors 'none'" in csp
    assert "https:" not in csp
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert "camera=()" in response.headers["permissions-policy"]
    assert artifact_response.headers["content-security-policy"] == "sandbox; default-src 'none'"


async def test_each_refusal_says_which_one_it_was(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Three refusals a caller must tell apart to know what to do next.

    An untrusted origin means the request came from somewhere it should not.
    A missing session means authenticate. A failed CSRF check means the token
    is stale and should be fetched again. All three were 401 or 403 with prose,
    so a client could only match on wording the next improvement would break.

    They are raised in middleware, which runs outside the app's exception
    handlers, so the codes reach nobody unless the middleware carries them.
    That is what this pins.
    """
    settings = Settings(data_dir=tmp_path / "codes", dev=False, port=12340)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            no_session = await client.get("/api/projects")
            assert no_session.status_code == 401
            assert no_session.json()["code"] == "session-required"

            session = await client.post("/api/session")
            csrf = session.json()["csrf_token"]

            no_csrf = await client.post("/api/projects", json={"name": "Denied"})
            assert no_csrf.status_code == 403
            assert no_csrf.json()["code"] == "csrf-invalid"

            bad_origin = await client.post(
                "/api/projects",
                json={"name": "Denied"},
                headers={"origin": "https://malicious.example", "x-local-lm-csrf": csrf},
            )
            assert bad_origin.status_code == 403
            assert bad_origin.json()["code"] == "origin-untrusted"
            # The prose is unchanged, so anything already matching on it keeps
            # working; the code is a sibling rather than a replacement.
            assert bad_origin.json()["detail"] == "untrusted browser origin"


def test_the_session_cookie_is_not_a_durable_on_disk_secret(tmp_path: Path) -> None:
    """It used to be the secret itself, read from state/session-secret and
    handed to the browser unchanged, so it was identical after every restart and
    the CSRF token derived from it could never rotate."""

    from local_lm.config import Settings
    from local_lm.security import SessionSecurity

    settings = Settings(data_dir=tmp_path / "session")
    settings.prepare()

    first = SessionSecurity(settings)
    second = SessionSecurity(settings)

    response = Response()
    first.issue_session(response)
    cookie = response.headers["set-cookie"]

    # Nothing on disk carries it, so nothing on disk leaks it.
    assert not list(settings.state_dir.glob("session-secret*"))
    # A later run of the service does not reissue the same session.
    assert first.csrf_token != second.csrf_token
    assert not first._valid_cookie(second._session_token)
    assert first._session_token in cookie
    assert first.csrf_token not in cookie
