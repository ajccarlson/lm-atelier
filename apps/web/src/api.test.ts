import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.resetModules();
  sessionStorage.clear();
  localStorage.clear();
});

it("parses the bounded visible Media Library page before returning it", async () => {
  const sha = "a".repeat(64);
  const response = {
    items: [{
      id: `libentry:sha256:${sha}`,
      artifact_id: `sha256:${sha}`,
      version: 1,
      state: "visible",
      display_name: "Published image",
      favorite: false,
      kind: "image",
      media_type: "image/png",
      size_bytes: 10,
      created_at: "2026-08-12T12:00:00Z",
      updated_at: "2026-08-12T12:00:00Z",
    }],
    next_cursor: null,
  };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify(response), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  const controller = new AbortController();
  await expect(api.artifactLibrary(
    { kind: "image", query: " portrait ", favorite: true },
    "opaque_cursor",
    20,
    controller.signal,
  )).resolves.toMatchObject({ items: [{ artifact_id: `sha256:${sha}` }] });
  expect(fetchMock.mock.calls[1][0]).toBe(
    "/api/artifact-library?limit=20&query=+portrait+&state=visible&kind=image&favorite=true&cursor=opaque_cursor",
  );
  expect(fetchMock.mock.calls[1][1]?.signal).toBe(controller.signal);
  expect(fetchMock.mock.calls[1][1]?.method).toBeUndefined();
});

it("rejects a malformed Media Library response atomically", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      items: [{ artifact_id: "private/path" }],
      next_cursor: null,
    }), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.artifactLibrary(
    { kind: "", query: "", favorite: false },
    null,
    20,
  )).rejects.toThrow("The Media Library response was invalid.");
});

it("rejects a Media Library cursor self-loop", async () => {
  const cursor = `cGF5bG9hZA.${"a".repeat(43)}`;
  const items = Array.from({ length: 20 }, (_, index) => {
    const sha = index.toString(16).padStart(64, "0");
    return {
      id: `libentry:sha256:${sha}`,
      artifact_id: `sha256:${sha}`,
      version: 1,
      state: "visible",
      display_name: `Item ${index}`,
      favorite: false,
      kind: "image",
      media_type: "image/png",
      size_bytes: 1,
      created_at: "2026-08-12T12:00:00Z",
      updated_at: "2026-08-12T12:00:00Z",
    };
  });
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      items,
      next_cursor: cursor,
    }), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.artifactLibrary(
    { kind: "", query: "", favorite: false },
    cursor,
    20,
  )).rejects.toThrow("The Media Library response was invalid.");
});

it("requests the read-only setup readiness contract", async () => {
  const report = { version: 2, state: "ready", roles: [] };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify(report), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.setupReadiness()).resolves.toEqual(report);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/setup/readiness");
  expect(fetchMock.mock.calls[1][1]?.method).toBeUndefined();
});

it("starts setup verification with the local CSRF contract", async () => {
  const verification = {
    id: "verify-image",
    role: "image",
    state: "queued",
    job_id: "job-image",
    failure_code: null,
    started_at: null,
    completed_at: null,
  };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify(verification), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.verifySetupRole("image")).resolves.toEqual(verification);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/setup/verify/image");
  expect(fetchMock.mock.calls[1][1]?.method).toBe("POST");
  expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-local-lm-csrf")).toBe("csrf");
});

it("keeps native workflow editor authority in authenticated request bodies", async () => {
  const session = {
    id: "editor-session",
    protocol_version: 2,
    workflow_id: "workflow/one",
    base_revision_id: "revision-one",
    ui_graph: {},
    nonce: "secret-nonce",
  };
  const returned = {
    validated_return_id: "validated-return",
    changed: true,
  };
  const draft = {
    draft_revision_id: "draft-one",
    review_required: true,
  };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ csrf_token: "csrf" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
    .mockResolvedValueOnce(new Response(JSON.stringify(session), {
      status: 201,
      headers: { "content-type": "application/json" },
    }))
    .mockResolvedValueOnce(new Response(JSON.stringify(returned), {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
    .mockResolvedValueOnce(new Response(JSON.stringify(draft), {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.startWorkflowEditor("workflow/one");
  await api.consumeWorkflowEditor("workflow/one", "editor/session", {
    nonce: "secret-nonce",
    base_revision_id: "revision-one",
    ui_graph: { nodes: [] },
    api_prompt: { 1: {} },
  });
  await api.createWorkflowEditorDraft("workflow/one", "validated-return");
  await api.cancelWorkflowEditor("workflow/one", "editor/session", "secret-nonce");

  expect(fetchMock.mock.calls.slice(1).map((call) => call[0])).toEqual([
    "/api/workflows/workflow%2Fone/editor-sessions",
    "/api/workflows/workflow%2Fone/editor-sessions/editor%2Fsession/consume",
    "/api/workflows/workflow%2Fone/editor-drafts",
    "/api/workflows/workflow%2Fone/editor-sessions/editor%2Fsession/cancel",
  ]);
  expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
    nonce: "secret-nonce",
    base_revision_id: "revision-one",
    ui_graph: { nodes: [] },
    api_prompt: { 1: {} },
  });
  expect(JSON.parse(String(fetchMock.mock.calls[3][1]?.body))).toEqual({
    validated_return_id: "validated-return",
  });
  expect(JSON.parse(String(fetchMock.mock.calls[4][1]?.body))).toEqual({
    nonce: "secret-nonce",
  });
  for (const call of fetchMock.mock.calls.slice(1)) {
    expect(new Headers(call[1]?.headers).get("x-local-lm-csrf")).toBe("csrf");
    expect(String(call[0])).not.toContain("secret-nonce");
  }
});
it("returns the complete uploaded artifact for composer previews", async () => {
  const artifact = {
    id: "artifact-uploaded",
    sha256: "uploaded",
    kind: "input",
    media_type: "image/png",
    size_bytes: 6,
    original_name: "source.png",
    metadata_json: { uploaded: true },
    created_at: "2026-07-30T00:00:00Z",
    url: "/api/artifacts/artifact-uploaded/content",
  };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify(artifact), {
        status: 201,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  const file = new File(["pixels"], "source.png", { type: "image/png" });
  await expect(api.upload(file)).resolves.toEqual(artifact);
  expect(fetchMock.mock.calls[1][0]).toBe("/api/artifacts");
  expect(fetchMock.mock.calls[1][1]?.method).toBe("POST");
  expect(fetchMock.mock.calls[1][1]?.body).toBeInstanceOf(FormData);
  expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-local-lm-csrf")).toBe("csrf");
});
it("confirms a bounded ordered plan without changing Auto mode", async () => {
  const confirm = vi.fn(() => true);
  vi.stubGlobal("confirm", confirm);
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        detail: {
          code: "ordered_plan_confirmation_required",
          message: "Confirm the ordered multi-model plan.",
          plan: {
            planner_version: "ordered-work-v1",
            steps: [
              { id: "story", mode: "text", prompt: "Write", depends_on: [], inputs: [] },
              { id: "image", mode: "image", prompt: "Draw", depends_on: ["story"], inputs: [] },
              { id: "video", mode: "video", prompt: "Animate", depends_on: ["image"], inputs: [] },
            ],
          },
          estimate: {
            video_duration_seconds: 4,
            estimated_bytes: 2 * 1024 ** 3,
          },
        },
      }), {
        status: 409,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ accepted: true }), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.sendTurn(
    "chat-ordered",
    "Write a story, then draw it, then animate it",
    "auto",
    [],
    {},
    "ordered-key",
  );

  expect(confirm).toHaveBeenCalledWith(
    "3-step plan: text → image → video · about 4 seconds of video · up to 2 GB working space. Start it?",
  );
  const retryBody = JSON.parse(String(fetchMock.mock.calls[2][1]?.body));
  expect(retryBody.mode).toBe("auto");
  expect(retryBody.confirm_media).toBe(true);
  expect(retryBody.idempotency_key).toBe("ordered-key");
});

it("opens the event socket from the sequence returned by session initialization", async () => {
  const urls: string[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: (() => void) | null = null;

    constructor(url: string) {
      urls.push(url);
    }

    close() {}
  }

  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 742 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    ),
  );
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());

  expect(urls).toHaveLength(1);
  expect(new URL(urls[0]).searchParams.get("after")).toBe("742");
  dispose();
});

it("sends turn overrides with edited branches and regenerated responses", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({}), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    ));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.branchMessage(
    "message-user",
    "Count to 1000",
    "text",
    { max_tokens: 4096 },
  );
  await api.regenerateMessage("message-assistant", { max_tokens: 4096 });

  expect(fetchMock.mock.calls[1][0]).toBe("/api/messages/message-user/branch");
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    text: "Count to 1000",
    mode: "text",
    input_artifact_ids: [],
    settings: { max_tokens: 4096 },
  });
  expect(fetchMock.mock.calls[2][0]).toBe("/api/messages/message-assistant/regenerate");
  expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
    settings: { max_tokens: 4096 },
  });
});

it("uses the explicit stop-and-send endpoint and preserves its idempotency key", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({}), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.stopAndSendTurn(
    "chat/one",
    "Use this instead",
    "text",
    [],
    { max_tokens: 128 },
    "client-turn-7",
  );

  expect(fetchMock.mock.calls[1][0]).toBe("/api/chats/chat/one/stop-and-send");
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
    text: "Use this instead",
    mode: "text",
    idempotency_key: "client-turn-7",
  });
});

it("uses the recovery and unsuccessful-job action contracts", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ id: "job/retry" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ name: "backup one.sqlite3" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ name: "backup one.sqlite3", restore_pending: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.retryJob("job/retry");
  await api.verifyBackup("backup one.sqlite3");
  await api.restoreBackup("backup one.sqlite3");
  await api.deleteBackup("backup one.sqlite3");

  expect(fetchMock.mock.calls.slice(1).map(([url, init]) => [url, init?.method])).toEqual([
    ["/api/jobs/job%2Fretry/retry", "POST"],
    ["/api/backups/backup%20one.sqlite3/verify", "POST"],
    ["/api/backups/backup%20one.sqlite3/restore", "POST"],
    ["/api/backups/backup%20one.sqlite3", "DELETE"],
  ]);
});

it("requests transactional profile cleanup when deleting an installed model", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.deleteModel("model-1", true);

  expect(fetchMock.mock.calls[1][0]).toBe(
    "/api/models/model-1?delete_profiles=true",
  );
  expect(fetchMock.mock.calls[1][1]?.method).toBe("DELETE");
});

it("retries session initialization after a transient startup failure", async () => {
  const fetchMock = vi.fn()
    .mockRejectedValueOnce(new Error("service is starting"))
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-recovered" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.projects()).rejects.toThrow("service is starting");
  await expect(api.projects()).resolves.toEqual([]);

  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
    "/api/session",
    "/api/session",
    "/api/projects?include_archived=false&query=",
  ]);
});

it("refreshes an expired session once before retrying an API request", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "session required" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ id: "project-1", name: "Recovered" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.createProject("Recovered")).resolves.toMatchObject({
    id: "project-1",
  });

  const firstHeaders = new Headers(fetchMock.mock.calls[1][1]?.headers);
  const retriedHeaders = new Headers(fetchMock.mock.calls[3][1]?.headers);
  expect(firstHeaders.get("x-local-lm-csrf")).toBe("csrf-old");
  expect(retriedHeaders.get("x-local-lm-csrf")).toBe("csrf-new");
});

it("keeps retrying event initialization while the local service starts", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockRejectedValueOnce(new Error("service is starting"))
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 9 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const onStatus = vi.fn();
  const dispose = await connectEvents(vi.fn(), onStatus);

  expect(urls).toHaveLength(0);
  expect(onStatus).toHaveBeenCalledWith(false);
  await vi.advanceTimersByTimeAsync(1_000);
  expect(urls).toHaveLength(1);
  expect(new URL(urls[0]).searchParams.get("after")).toBe("9");
  dispose();
});

it("renews the session after an authenticated event socket is rejected", async () => {
  vi.useFakeTimers();
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor() {
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  expect(sockets).toHaveLength(1);

  sockets[0].onclose?.({ code: 4401 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(sockets).toHaveLength(2);
  dispose();
});

it("replays events from zero when the service sequence resets after a restart", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old", event_sequence: 742 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new", event_sequence: 3 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  expect(new URL(urls[0]).searchParams.get("after")).toBe("742");

  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(new URL(urls[1]).searchParams.get("after")).toBe("0");
  dispose();
});

it("replays from zero when a restarted service has already advanced beyond the old sequence", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        csrf_token: "csrf-old",
        event_epoch: "old-process",
        event_sequence: 3,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        csrf_token: "csrf-new",
        event_epoch: "new-process",
        event_sequence: 15,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  expect(new URL(urls[0]).searchParams.get("after")).toBe("3");

  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(new URL(urls[1]).searchParams.get("after")).toBe("0");
  dispose();
});

it("retains the last received sequence during a same-service reconnect", async () => {
  vi.useFakeTimers();
  const urls: string[] = [];
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor(url: string) {
      urls.push(url);
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 10 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf", event_sequence: 15 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const dispose = await connectEvents(vi.fn(), vi.fn());
  sockets[0].onmessage?.({
    data: JSON.stringify({ sequence: 12, type: "generation.progress", payload: {} }),
  } as MessageEvent);
  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);

  expect(new URL(urls[1]).searchParams.get("after")).toBe("12");
  dispose();
});

it("notifies the client after each event socket reconnect, not the initial open", async () => {
  vi.useFakeTimers();
  const sockets: FakeWebSocket[] = [];

  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((message: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;

    constructor() {
      sockets.push(this);
    }

    close() {}
  }

  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-old", event_sequence: 10 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf-new", event_sequence: 10 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const { connectEvents } = await import("./api");
  const onReconnect = vi.fn();
  const dispose = await connectEvents(vi.fn(), vi.fn(), onReconnect);
  sockets[0].onopen?.();
  expect(onReconnect).not.toHaveBeenCalled();

  sockets[0].onclose?.({ code: 1006 } as CloseEvent);
  await vi.advanceTimersByTimeAsync(1_000);
  sockets[1].onopen?.();
  expect(onReconnect).toHaveBeenCalledTimes(1);
  dispose();
});

it("recovers a stale CSRF token instead of failing permanently", async () => {
  // resetSession() clears the token on every socket close, so a request that
  // started just before a close goes out with an empty header. The server
  // answers 403 for that, not 401, and only 401 used to be retried - which made
  // a narrow race into a permanent dead end.
  const chat = { id: "chat-1", title: "Recovered" };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "first" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "CSRF check failed" }), {
        status: 403,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "second" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify(chat), {
        status: 201,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.createChat(null)).resolves.toEqual(chat);
  // A fresh session was fetched and the retry carried the new token.
  expect(fetchMock.mock.calls[2][0]).toBe("/api/session");
  expect(new Headers(fetchMock.mock.calls[3][1]?.headers).get("x-local-lm-csrf")).toBe("second");
});

it("does not retry a 403 that is a genuine refusal", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "operation is not permitted" }), {
        status: 403,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api, ApiError } = await import("./api");
  await expect(api.createChat(null)).rejects.toBeInstanceOf(ApiError);
  // Session, then the request. No retry, no second session fetch.
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("surfaces the stable error code beside the human-readable detail", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({ detail: "Stop the media worker first", code: "media_worker_running" }),
        { status: 409, headers: { "content-type": "application/json" } },
      ),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api, ApiError } = await import("./api");
  const failure = await api.reviewRegistryInstall("install-1", true).then(
    () => null,
    (error: unknown) => error,
  );
  expect(failure).toBeInstanceOf(ApiError);
  expect((failure as InstanceType<typeof ApiError>).code).toBe("media_worker_running");
  expect((failure as Error).message).toBe("Stop the media worker first");
});

it("routes a CivitAI preflight through the source-aware endpoint", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ can_install: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.catalogPreflight("201", "image", "comfyui", "201", [], "lora", null, "civitai");
  expect(fetchMock.mock.calls[1][0]).toBe("/api/catalog/preflight?source=civitai&id=201");
  // The Hugging Face path is untouched.
  fetchMock.mockResolvedValueOnce(
    new Response(JSON.stringify({ can_install: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );
  await api.catalogPreflight("owner/name", "chat", "llama.cpp", "main", []);
  expect(fetchMock.mock.calls[2][0]).toBe("/api/catalog/owner/name/preflight");
});

it("retries a stale token when the refusal carries the csrf code", async () => {
  // The retry used to key on the exact sentence "CSRF check failed". Rewording
  // it would have turned a recoverable stale token into a hard refusal, and
  // the sentence looked entirely safe to change.
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "first" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "reworded entirely", code: "csrf-invalid" }), {
        status: 403,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "second" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.editTemplates()).resolves.toEqual([]);

  // Session, refused request, fresh session, replay. A GET carries no CSRF
  // header, so the replay is what proves the retry rather than the header.
  expect(fetchMock).toHaveBeenCalledTimes(4);
  expect(fetchMock.mock.calls[2][0]).toBe("/api/session");
  expect(fetchMock.mock.calls[3][0]).toBe("/api/edit-templates");
});

it("still recognizes the older wording, for a server that predates the code", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "first" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "CSRF check failed" }), {
        status: 403,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "second" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await expect(api.editTemplates()).resolves.toEqual([]);
});

it("drops event frames it cannot act on, and keeps the ones it can", async () => {
  const { readEvent } = await import("./api");

  // onmessage runs long after the try around the connection returned, so a
  // throw here would leave the socket open and the stream silent.
  expect(readEvent("not json at all")).toBeNull();
  expect(readEvent("null")).toBeNull();
  expect(readEvent("[1,2,3]")).toBeNull();
  expect(readEvent(new Blob())).toBeNull();

  // A sequence that is not a real number would carry NaN into the ?after= of
  // every later reconnect, through Math.max.
  expect(readEvent(JSON.stringify({ type: "run.updated" }))).toBeNull();
  expect(readEvent(JSON.stringify({ sequence: "12", type: "run.updated" }))).toBeNull();
  expect(readEvent(JSON.stringify({ sequence: Number.NaN, type: "run.updated" }))).toBeNull();

  const good = readEvent(
    JSON.stringify({
      sequence: 7,
      type: "run.updated",
      entity_id: null,
      payload: {},
      created_at: "2026-08-07T00:00:00Z",
    }),
  );
  expect(good?.sequence).toBe(7);
});

it("sends the references the person chose, and nothing it inferred", async () => {
  // The whole reference path in one assertion: ids travel as data. The server
  // refuses to recover references by reading a prompt, so the client must
  // never send something it worked out from the text.
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ accepted: true }), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.sendTurn(
    "chat-1",
    // Two mentions written, one of them never chosen from the picker.
    "draw @ada-lovelace beside @grace-hopper",
    "image",
    [],
    {},
    "reference-key",
    "turns",
    undefined,
    [{ reference_subject_id: "ref-1", source: "mention" }],
  );

  const body = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
  expect(body.references).toEqual([{ reference_subject_id: "ref-1", source: "mention" }]);
});

it("sends an empty reference list when nothing was chosen", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ accepted: true }), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.sendTurn("chat-1", "draw @ada-lovelace", "image", [], {}, "no-reference-key");

  const body = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
  expect(body.references).toEqual([]);
});

it("sends an explicit media output count and never leaks it into Auto", async () => {
  vi.stubGlobal("confirm", vi.fn(() => true));
  const accepted = () => new Response(JSON.stringify({ accepted: true }), {
    status: 202,
    headers: { "content-type": "application/json" },
  });
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ csrf_token: "csrf" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(accepted())
    .mockResolvedValueOnce(
      new Response(JSON.stringify({
        detail: {
          code: "route_confirmation_required",
          plan: { operation: "text_to_image" },
          estimate: {},
        },
      }), {
        status: 409,
        headers: { "content-type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(accepted());
  vi.stubGlobal("fetch", fetchMock);

  const { api } = await import("./api");
  await api.sendTurn(
    "chat-1", "four images", "image", [], {}, "image-count", "turns", undefined, [], 4,
  );
  await api.sendTurn(
    "chat-1", "choose for me", "auto", [], {}, "auto-count", "turns", undefined, [], 4,
  );

  const imageBody = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
  const autoBody = JSON.parse(String(fetchMock.mock.calls[2][1]?.body));
  const confirmedAutoBody = JSON.parse(String(fetchMock.mock.calls[3][1]?.body));
  expect(imageBody.output_count).toBe(4);
  expect(autoBody).not.toHaveProperty("output_count");
  expect(confirmedAutoBody.mode).toBe("image");
  expect(confirmedAutoBody.confirm_media).toBe(true);
  expect(confirmedAutoBody).not.toHaveProperty("output_count");
});
