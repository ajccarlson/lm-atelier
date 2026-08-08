from __future__ import annotations

import hashlib
import hmac
import secrets

from fastapi import Request, Response, WebSocket, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .api_errors import api_error
from .config import Settings

SESSION_COOKIE = "local_lm_session"
CSRF_HEADER = "x-local-lm-csrf"
MAX_JSON_BODY_BYTES = 4 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
SECURITY_HEADERS = {
    b"content-security-policy": (
        b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        b"img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; "
        b"connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
        b"worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
        b"form-action 'self'; frame-ancestors 'none'"
    ),
    b"cross-origin-opener-policy": b"same-origin",
    b"cross-origin-resource-policy": b"same-origin",
    b"permissions-policy": (
        b"camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), bluetooth=()"
    ),
    b"referrer-policy": b"no-referrer",
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
}


def trusted_browser_origins(settings: Settings) -> frozenset[str]:
    origins = {
        f"http://127.0.0.1:{settings.port}",
        f"http://localhost:{settings.port}",
        f"http://[::1]:{settings.port}",
    }
    if settings.dev:
        origins.update(
            {
                "http://127.0.0.1:5173",
                "http://localhost:5173",
                "http://[::1]:5173",
            }
        )
    return frozenset(origins)


class JsonBodyLimitMiddleware:
    """Bound JSON bodies while leaving separately streamed upload endpoints intact."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_JSON_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not str(scope.get("path", "")).startswith("/api")
            or not self._is_json(scope)
        ):
            await self.app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }
            # Preserve the real connection lifecycle after replaying the body.
            # Returning synthetic disconnect messages without awaiting the
            # server receive channel can drive downstream listeners into a
            # full-core hot loop for the rest of the application lifetime.
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _is_json(scope: Scope) -> bool:
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
        return media_type == b"application/json" or media_type.endswith(b"+json")

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"JSON request body exceeds the {self.max_bytes}-byte limit"},
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
        await response(scope, receive, send)


class UploadBodyLimitMiddleware:
    """Bound multipart requests before Starlette materializes uploaded files."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        artifact_max_bytes: int,
        project_max_bytes: int,
    ) -> None:
        self.app = app
        self.limits = {
            "/api/artifacts": artifact_max_bytes + MULTIPART_OVERHEAD_BYTES,
            "/api/projects/import": project_max_bytes + MULTIPART_OVERHEAD_BYTES,
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self._limit(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    await self._reject(scope, receive, send, limit)
                    return
            except ValueError:
                pass

        received = 0
        exceeded = False
        rejection_sent = False

        async def limited_receive() -> Message:
            nonlocal exceeded, received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _UploadLimitExceeded
            return message

        async def limited_send(message: Message) -> None:
            nonlocal rejection_sent
            if exceeded:
                if message["type"] == "http.response.start" and not rejection_sent:
                    rejection_sent = True
                    await self._reject(scope, receive, send, limit)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except _UploadLimitExceeded:
            if not rejection_sent:
                await self._reject(scope, receive, send, limit)

    def _limit(self, scope: Scope) -> int | None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            return None
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
        if media_type != b"multipart/form-data":
            return None
        return self.limits.get(str(scope.get("path", "")))

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        payload_limit = max(0, limit - MULTIPART_OVERHEAD_BYTES)
        response = JSONResponse(
            {"detail": f"upload request exceeds the {payload_limit}-byte limit"},
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
        await response(scope, receive, send)


class SecurityHeadersMiddleware:
    """Apply browser isolation headers without replacing stricter route headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _value in headers}
                headers.extend(
                    (name, value)
                    for name, value in SECURITY_HEADERS.items()
                    if name not in existing
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _UploadLimitExceeded(Exception):
    pass


class SessionSecurity:
    """Hold the browser's session in memory, for this run of the service only.

    The session token was previously a secret kept on disk and handed to the
    browser unchanged, so the cookie was not a token standing for a session - it
    was the durable credential itself, identical after every restart, with a
    CSRF token derived from it that could therefore never rotate either.

    Nothing needs that persistence. The page asks for a session when it loads,
    so a token minted per run is enough, and both values change whenever the
    service does.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session_token = secrets.token_urlsafe(48)
        self.csrf_token = hmac.new(
            self._session_token.encode(), b"local-lm-csrf-v1", hashlib.sha256
        ).hexdigest()

    def issue_session(self, response: Response) -> str:
        response.set_cookie(
            SESSION_COOKIE,
            self._session_token,
            httponly=True,
            secure=False,
            samesite="strict",
            path="/",
        )
        return self.csrf_token

    def _valid_cookie(self, cookie: str | None) -> bool:
        return bool(cookie and hmac.compare_digest(cookie, self._session_token))

    def _valid_origin(self, origin: str | None) -> bool:
        if origin is None:
            return True
        return origin.lower() in trusted_browser_origins(self.settings)

    def validate_origin(self, origin: str | None) -> None:
        if not self._valid_origin(origin):
            raise api_error(
                status.HTTP_403_FORBIDDEN,
                "origin-untrusted",
                "untrusted browser origin",
            )

    def validate_request(self, request: Request) -> None:
        if self.settings.dev and request.client and request.client.host == "testclient":
            return
        self.validate_origin(request.headers.get("origin"))
        if not self._valid_cookie(request.cookies.get(SESSION_COOKIE)):
            raise api_error(status.HTTP_401_UNAUTHORIZED, "session-required", "session required")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            csrf = request.headers.get(CSRF_HEADER)
            if not csrf or not hmac.compare_digest(csrf, self.csrf_token):
                raise api_error(status.HTTP_403_FORBIDDEN, "csrf-invalid", "CSRF check failed")

    async def validate_websocket(self, websocket: WebSocket) -> bool:
        if self.settings.dev and websocket.client and websocket.client.host == "testclient":
            return True
        return self._valid_origin(websocket.headers.get("origin")) and self._valid_cookie(
            websocket.cookies.get(SESSION_COOKIE)
        )
