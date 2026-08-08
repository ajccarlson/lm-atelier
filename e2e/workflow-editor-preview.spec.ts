import {
  expect,
  test,
  type APIRequestContext,
  type Frame,
  type Page,
  type Response as PlaywrightResponse,
} from "@playwright/test";

const WORKFLOW_NAME = "Synthetic browser protocol fixture";
const INITIAL_LABEL = "camera";
const EDITED_LABEL = "studio";
const MANUAL_PROTOCOL_LABEL = "manual-protocol-load";
type WorkflowRevision = {
  id: string;
  version: number;
  trusted: boolean;
  ui_graph_json: Record<string, unknown>;
  api_graph_json: Record<string, unknown>;
};

type Workflow = {
  id: string;
  name: string;
  current_revision_id: string;
  revisions: WorkflowRevision[];
};

type WorkflowEditorReturn = {
  validated_return_id: string;
};

function uiGraph(label = INITIAL_LABEL): Record<string, unknown> {
  return {
    version: 0.4,
    nodes: [
      {
        id: 1,
        type: "Source",
        mode: 0,
        inputs: [],
        outputs: [{ name: "IMAGE", type: "IMAGE", links: [] }],
        widgets_values: [label, 42, "randomize"],
      },
    ],
    links: [],
  };
}

function apiGraph(label = INITIAL_LABEL): Record<string, unknown> {
  return {
    "1": {
      inputs: { label, seed: 42 },
      class_type: "Source",
      _meta: { title: "Source image" },
    },
  };
}

function parseCsp(value: string): Record<string, string[]> {
  const parsed: Record<string, string[]> = {};
  for (const rawDirective of value.split(";")) {
    const fields = rawDirective.trim().split(/\s+/).filter(Boolean);
    if (fields.length === 0) continue;
    const [name, ...sources] = fields;
    if (name in parsed) throw new Error(`duplicate CSP directive: ${name}`);
    parsed[name] = sources;
  }
  return parsed;
}

async function createSession(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/session");
  expect(response.ok()).toBeTruthy();
  const payload = await response.json() as { csrf_token: string };
  return payload.csrf_token;
}

async function createWorkflow(request: APIRequestContext): Promise<Workflow> {
  const csrfToken = await createSession(request);
  const response = await request.post("/api/workflows", {
    headers: { "x-local-lm-csrf": csrfToken },
    data: {
      name: WORKFLOW_NAME,
      operation: "text_to_image",
      description: "Synthetic workflow used only for browser protocol certification.",
      engine: "comfyui",
      engine_version: "0.28.0",
      ui_graph: uiGraph(),
      api_graph: apiGraph(),
      input_schema: {},
      dependencies: {},
      trusted: true,
    },
  });
  expect(response.status(), await response.text()).toBe(201);
  return response.json() as Promise<Workflow>;
}

async function listWorkflow(
  request: APIRequestContext,
  workflowId: string,
): Promise<Workflow> {
  const response = await request.get("/api/workflows");
  expect(response.ok()).toBeTruthy();
  const workflows = await response.json() as Workflow[];
  const workflow = workflows.find((candidate) => candidate.id === workflowId);
  expect(workflow).toBeTruthy();
  return workflow!;
}

async function dismissSetup(page: Page): Promise<void> {
  const dialog = page.getByRole("dialog", { name: "Set up LM Atelier" });
  const appeared = await dialog.waitFor({ state: "visible", timeout: 5_000 })
    .then(() => true)
    .catch(() => false);
  if (!appeared) return;
  await dialog.getByRole("button", { name: "Not now" }).click();
  await expect(dialog).toBeHidden();
}

async function probeHostileConnects(
  hostileFrame: Frame,
  shellOrigin: string,
): Promise<{ forged: unknown[]; extraPorts: unknown[] }> {
  return hostileFrame.evaluate(async (targetOrigin) => {
    const delay = (milliseconds: number) => new Promise(
      (resolve) => window.setTimeout(resolve, milliseconds),
    );
    const attempt = async (portCount: number): Promise<unknown[]> => {
      const channels = Array.from({ length: portCount }, () => new MessageChannel());
      const responses: unknown[] = [];
      for (const channel of channels) {
        channel.port1.onmessage = (event) => responses.push(event.data);
        channel.port1.start();
      }
      window.top!.postMessage(
        { source: "lm-atelier", protocol: 2, type: "connect" },
        targetOrigin,
        channels.map((channel) => channel.port2),
      );
      await delay(50);
      for (const channel of channels) {
        channel.port1.postMessage({
          source: "lm-atelier",
          protocol: 2,
          type: "load",
          nonce: "forged-browser-protocol-nonce",
          graph: { nodes: [], links: [] },
        });
      }
      await delay(200);
      for (const channel of channels) channel.port1.close();
      return responses;
    };
    return {
      forged: await attempt(1),
      extraPorts: await attempt(2),
    };
  }, shellOrigin);
}

async function connectLegitimateProtocolOpener(
  page: Page,
  shellOrigin: string,
): Promise<string[]> {
  return page.evaluate(async ({ graph, targetOrigin }) => {
    const host = window as typeof window & { syntheticProtocolPopup?: Window | null };
    const popup = host.syntheticProtocolPopup;
    if (!popup) throw new Error("synthetic protocol popup is unavailable");
    const channel = new MessageChannel();
    const messages: string[] = [];
    return new Promise<string[]>((resolve, reject) => {
      const timeout = window.setTimeout(
        () => reject(new Error("legitimate protocol connection timed out")),
        2_000,
      );
      channel.port1.onmessage = (event) => {
        const message = event.data as { type?: unknown };
        if (typeof message?.type !== "string") return;
        messages.push(message.type);
        if (message.type === "connected") {
          channel.port1.postMessage({
            source: "lm-atelier",
            protocol: 2,
            type: "load",
            nonce: "legitimate-browser-protocol-nonce",
            graph,
          });
        } else if (message.type === "loaded") {
          window.clearTimeout(timeout);
          channel.port1.close();
          resolve(messages);
        }
      };
      channel.port1.start();
      popup.postMessage(
        { source: "lm-atelier", protocol: 2, type: "connect" },
        targetOrigin,
        [channel.port2],
      );
    });
  }, { graph: uiGraph(MANUAL_PROTOCOL_LABEL), targetOrigin: shellOrigin });
}

async function probeLegitimateConnectAttempt(
  page: Page,
  shellOrigin: string,
  portCount: number,
): Promise<unknown[]> {
  return page.evaluate(async ({ targetOrigin, transferredPortCount }) => {
    const host = window as typeof window & { syntheticProtocolPopup?: Window | null };
    const popup = host.syntheticProtocolPopup;
    if (!popup) throw new Error("synthetic protocol popup is unavailable");
    const channels = Array.from(
      { length: transferredPortCount },
      () => new MessageChannel(),
    );
    const responses: unknown[] = [];
    for (const channel of channels) {
      channel.port1.onmessage = (event) => responses.push(event.data);
      channel.port1.start();
    }
    popup.postMessage(
      { source: "lm-atelier", protocol: 2, type: "connect" },
      targetOrigin,
      channels.map((channel) => channel.port2),
    );
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    for (const channel of channels) channel.port1.close();
    return responses;
  }, { targetOrigin: shellOrigin, transferredPortCount: portCount });
}

test("certifies the synthetic native-editor browser protocol without claiming managed ComfyUI", async ({
  context,
  page,
  request,
}) => {
  const baseURL = process.env.LM_ATELIER_E2E_BASE_URL;
  const comfyOrigin = process.env.LM_ATELIER_E2E_COMFY_ORIGIN;
  const attackerOrigin = process.env.LM_ATELIER_E2E_ATTACKER_ORIGIN;
  test.skip(
    !baseURL || !comfyOrigin || !attackerOrigin,
    "requires the isolated synthetic workflow-editor protocol runner",
  );
  expect(baseURL, "the protocol runner must provide the product origin").toBeTruthy();
  expect(comfyOrigin, "the protocol runner must provide the synthetic Comfy origin").toBeTruthy();
  expect(attackerOrigin, "the protocol runner must provide the hostile origin").toBeTruthy();
  const shellOrigin = new URL(baseURL!).origin;
  const allowedOrigins = new Set([
    new URL(baseURL!).origin,
    new URL(comfyOrigin!).origin,
    new URL(attackerOrigin!).origin,
  ]);
  const unexpectedRequests: string[] = [];
  const browserErrors: string[] = [];

  await context.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (["http:", "https:", "ws:", "wss:"].includes(url.protocol)
      && !allowedOrigins.has(url.origin)) {
      unexpectedRequests.push(`${url.protocol}//${url.host}${url.pathname}`);
      await route.abort();
      return;
    }
    await route.continue();
  });
  const watchPage = (candidate: Page) => {
    candidate.on("pageerror", (error) => browserErrors.push(error.message));
    candidate.on("console", (message) => {
      if (message.type() === "error") {
        browserErrors.push(message.text());
      }
    });
  };
  watchPage(page);
  context.on("page", watchPage);

  const created = await createWorkflow(request);

  const hostileOpener = await context.newPage();
  await hostileOpener.goto(`${baseURL}/`);
  await dismissSetup(hostileOpener);
  const isolatedPopupPromise = hostileOpener.waitForEvent("popup");
  await hostileOpener.evaluate((shellURL) => window.open(shellURL, "_blank", "popup"), `${baseURL}/api/workflow-editor/shell`);
  const isolatedPopup = await isolatedPopupPromise;
  await isolatedPopup.waitForLoadState("domcontentloaded");
  expect(await isolatedPopup.evaluate(() => window.opener !== null)).toBe(true);
  await hostileOpener.goto(`${attackerOrigin}/`);
  await expect(hostileOpener.getByRole("heading", { name: "Hostile protocol origin" })).toBeVisible();
  expect(await isolatedPopup.evaluate(() => window.opener === null)).toBe(true);
  await isolatedPopup.close();
  await hostileOpener.close();

  await page.goto("/");
  await dismissSetup(page);
  const shellResponsePromise = context.waitForEvent("response",
    (response) => response.url() === `${baseURL}/api/workflow-editor/shell`,
  );
  const protocolPopupPromise = page.waitForEvent("popup");
  await page.evaluate((shellURL) => {
    const host = window as typeof window & { syntheticProtocolPopup?: Window | null };
    host.syntheticProtocolPopup = window.open(shellURL, "_blank", "popup");
  }, `${baseURL}/api/workflow-editor/shell`);
  const [protocolPopup, shellResponse] = await Promise.all([
    protocolPopupPromise,
    shellResponsePromise,
  ]);
  await protocolPopup.waitForLoadState("domcontentloaded");

  const headers = await shellResponse.allHeaders();
  expect(headers["cross-origin-opener-policy"]).toBe("same-origin");
  expect(headers["cache-control"]).toBe("no-store");
  const csp = headers["content-security-policy"];
  expect(csp).toBeTruthy();
  expect(parseCsp(csp!)).toEqual({
    "default-src": ["'none'"],
    "script-src": ["'self'"],
    "style-src": ["'self'"],
    "frame-src": [comfyOrigin!],
    "connect-src": ["'none'"],
    "img-src": ["'none'"],
    "object-src": ["'none'"],
    "base-uri": ["'none'"],
    "form-action": ["'none'"],
    "frame-ancestors": ["'none'"],
  });
  expect(csp).not.toContain("'unsafe-inline'");
  expect(csp).not.toContain("'unsafe-eval'");

  const editorIframe = protocolPopup.locator('iframe[title="ComfyUI workflow editor"]');
  expect(await editorIframe.getAttribute("referrerpolicy")).toBe("no-referrer");
  expect((await editorIframe.getAttribute("sandbox"))?.split(/\s+/).sort()).toEqual([
    "allow-downloads",
    "allow-forms",
    "allow-modals",
    "allow-same-origin",
    "allow-scripts",
  ]);
  expect(await editorIframe.getAttribute("src")).toBe(`${comfyOrigin}/`);

  const editorFrame = protocolPopup.frameLocator('iframe[title="ComfyUI workflow editor"]');
  await expect(editorFrame.getByRole("heading", { name: "Synthetic browser protocol editor" })).toBeVisible();
  await expect(protocolPopup.locator("#workflow-editor-status")).toBeHidden();
  const hostileFrameLocator = editorFrame.frameLocator('iframe[title="Hostile protocol probe"]');
  await expect(hostileFrameLocator.getByRole("heading", { name: "Hostile protocol origin" })).toBeVisible();
  const hostileFrame = protocolPopup.frames().find(
    (candidate) => candidate.url() === `${attackerOrigin}/`,
  );
  expect(hostileFrame).toBeTruthy();
  expect(await probeHostileConnects(hostileFrame!, shellOrigin)).toEqual({
    forged: [],
    extraPorts: [],
  });
  expect(await probeLegitimateConnectAttempt(page, shellOrigin, 2)).toEqual([]);
  expect(await connectLegitimateProtocolOpener(page, shellOrigin)).toEqual([
    "connected",
    "loaded",
  ]);
  expect(await probeLegitimateConnectAttempt(page, shellOrigin, 1)).toEqual([]);
  await expect(editorFrame.getByRole("textbox", { name: "Source label" })).toHaveValue(
    MANUAL_PROTOCOL_LABEL,
  );
  await protocolPopup.close();

  const untouched = await listWorkflow(request, created.id);
  expect(untouched.current_revision_id).toBe(created.current_revision_id);
  expect(untouched.revisions).toHaveLength(1);
  expect(untouched.revisions[0].ui_graph_json).toEqual(uiGraph());
  expect(untouched.revisions[0].api_graph_json).toEqual(apiGraph());

  await page.getByRole("button", { name: "Workflows" }).click();
  await page.getByText(WORKFLOW_NAME, { exact: true }).click();

  const popupPromise = page.waitForEvent("popup");
  const sessionResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "POST"
    && response.url().endsWith(`/api/workflows/${created.id}/editor-sessions`)
  ));
  await page.getByRole("button", { name: "Edit in ComfyUI (preview)" }).click();
  const [popup, sessionResponse] = await Promise.all([popupPromise, sessionResponsePromise]);
  await popup.waitForLoadState("domcontentloaded");

  expect(sessionResponse.status()).toBe(201);
  const editorSession = await sessionResponse.json() as { nonce: string };
  expect(editorSession.nonce).toBeTruthy();
  expect(popup.url()).toBe(`${baseURL}/api/workflow-editor/shell`);
  expect(popup.url()).not.toContain(editorSession.nonce);
  expect(await popup.evaluate(() => Boolean(window.opener))).toBe(true);
  expect(await popup.evaluate(() => window.opener?.location.origin)).toBe(shellOrigin);

  const frame = popup.frameLocator('iframe[title="ComfyUI workflow editor"]');
  await expect(frame.getByRole("heading", { name: "Synthetic browser protocol editor" })).toBeVisible();
  await expect(popup.locator("#workflow-editor-status")).toBeHidden();
  await expect(frame.getByRole("textbox", { name: "Source label" })).toHaveValue(INITIAL_LABEL);
  await expect(frame.getByRole("button", { name: "Save to LM Atelier" })).toBeVisible();
  expect(await popup.locator("body").innerText()).not.toContain(editorSession.nonce);

  const exactConsumeResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "POST"
    && response.url().includes(`/api/workflows/${created.id}/editor-sessions/`)
    && response.url().endsWith("/consume")
  ));
  await frame.getByRole("textbox", { name: "Source label" }).fill(EDITED_LABEL);
  await frame.getByRole("button", { name: "Save to LM Atelier" }).click();

  const consumeResponse = await exactConsumeResponsePromise;
  expect(consumeResponse.status()).toBe(200);
  const returned = await consumeResponse.json() as WorkflowEditorReturn;
  expect(returned.validated_return_id).toBeTruthy();
  await expect(page.getByText("Draft v2 saved for review.", { exact: true })).toBeVisible();
  await expect.poll(() => popup.isClosed()).toBe(true);

  const saved = await listWorkflow(request, created.id);
  expect(saved.current_revision_id).toBe(created.current_revision_id);
  expect(saved.revisions).toHaveLength(2);
  const current = saved.revisions.find(
    (revision) => revision.id === saved.current_revision_id,
  );
  const draft = saved.revisions.find(
    (revision) => revision.id !== saved.current_revision_id,
  );
  expect(current?.ui_graph_json).toEqual(uiGraph());
  expect(current?.api_graph_json).toEqual(apiGraph());
  expect(draft).toMatchObject({ version: 2, trusted: false });
  expect(draft?.ui_graph_json).toEqual(uiGraph(EDITED_LABEL));
  expect(draft?.api_graph_json).toEqual(apiGraph(EDITED_LABEL));
  expect(JSON.stringify(draft)).not.toContain(editorSession.nonce);
  expect(JSON.stringify(draft)).not.toContain(returned.validated_return_id);

  expect(unexpectedRequests).toEqual([]);
  expect(browserErrors).toEqual([]);
});
