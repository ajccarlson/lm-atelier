import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReferencesLibrary } from "./ReferencesLibrary";
import { api } from "./api";
import type { ReferenceSubject } from "./types";

vi.mock("./api", () => ({
  api: {
    references: vi.fn(),
    createReference: vi.fn(),
    updateReference: vi.fn(),
    referenceDeletionImpact: vi.fn(),
    deleteReference: vi.fn(),
  },
}));

const mocked = vi.mocked(api);

function subject(overrides: Partial<ReferenceSubject> = {}): ReferenceSubject {
  return {
    id: "ref-1",
    name: "Ada Lovelace",
    mention_slug: "ada-lovelace",
    kind: "person",
    description: null,
    aliases_json: [],
    tags_json: [],
    cover_artifact_id: null,
    favorite: false,
    archived: false,
    ...overrides,
  };
}

function renderLibrary() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReferencesLibrary />
    </QueryClientProvider>,
  );
}

function show(items: ReferenceSubject[] = [subject()]) {
  mocked.references.mockResolvedValue({ items, total: items.length, limit: 50, offset: 0 });
  return renderLibrary();
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("references library", () => {
  it("shows the mention beside the name rather than implying it", async () => {
    // A rename does not move the mention by default, so the two can legitimately
    // disagree. Showing only the name would leave no way to know what to type.
    show([subject({ name: "Grace Hopper", mention_slug: "ada-lovelace" })]);

    expect(await screen.findByText("Grace Hopper")).toBeTruthy();
    expect(screen.getByText("@ada-lovelace")).toBeTruthy();
  });

  it("keeps an archived reference visible but receded", async () => {
    show([subject({ archived: true })]);

    const row = (await screen.findByText("Ada Lovelace")).closest("li");
    // Archiving is a removal from view, not from the record - so it is still
    // listed when asked for, and still readable when it is.
    expect(row?.className).toContain("archived");
  });

  it("asks what deletion would destroy before offering to do it", async () => {
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-1",
      name: "Ada Lovelace",
      asset_count: 3,
      exclusive_artifact_ids: ["art-1"],
    });
    show();

    fireEvent.click(await screen.findByLabelText("Delete permanently"));

    await waitFor(() => expect(mocked.referenceDeletionImpact).toHaveBeenCalledWith("ref-1"));
    expect(await screen.findByText(/holds 3 image/)).toBeTruthy();
    expect(screen.getByText(/1 image\(s\) are used only here/)).toBeTruthy();
    // Nothing is deleted by asking.
    expect(mocked.deleteReference).not.toHaveBeenCalled();
  });

  it("deletes only against the impact it showed", async () => {
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-1",
      name: "Ada Lovelace",
      asset_count: 2,
      exclusive_artifact_ids: [],
    });
    mocked.deleteReference.mockResolvedValue(undefined);
    show();

    fireEvent.click(await screen.findByLabelText("Delete permanently"));
    fireEvent.click(await screen.findByText("Delete permanently", { selector: "button.danger" }));

    // The count travels with the request: the server refuses to destroy
    // something other than what the user was looking at.
    await waitFor(() => expect(mocked.deleteReference).toHaveBeenCalledWith("ref-1", 2));
  });

  it("says an image shared with another reference would not be lost", async () => {
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-1",
      name: "Ada Lovelace",
      asset_count: 4,
      exclusive_artifact_ids: [],
    });
    show();

    fireEvent.click(await screen.findByLabelText("Delete permanently"));
    expect(await screen.findByText(/No images would be lost/)).toBeTruthy();
  });

  it("offers only kinds the server accepts", async () => {
    show([]);

    fireEvent.click(await screen.findByText("New reference"));
    const options = [...screen.getByRole("combobox").querySelectorAll("option")].map(
      (option) => option.textContent,
    );
    // A workflow declares which kinds it can condition on, so an invented one
    // could never be matched against anything.
    expect(options).toContain("person");
    expect(options).not.toContain("spaceship");
  });

  it("shows the chosen cover beside the name", async () => {
    // Queried by class rather than by role: the alt is deliberately empty
    // because the name is right beside it, which makes the image decorative
    // and gives it no accessible role to find.
    const { container } = show([subject({ cover_artifact_id: "art-9" })]);

    await screen.findByText("Ada Lovelace");
    const image = container.querySelector("img.reference-cover");
    expect(image?.getAttribute("src")).toContain("art-9");
  });

  it("shows no placeholder when no cover was chosen", async () => {
    // A column of empty boxes would make the list harder to scan, which is the
    // opposite of what a cover is for.
    const { container } = show([subject({ cover_artifact_id: null })]);

    await screen.findByText("Ada Lovelace");
    expect(container.querySelector("img.reference-cover")).toBeNull();
  });

  it("pages through every reference instead of stopping at the first fifty", async () => {
    mocked.references.mockImplementation(async (_search, _archived, limit, offset) => {
      const pageLimit = limit ?? 50;
      const pageOffset = offset ?? 0;
      expect(pageLimit).toBe(50);
      return pageOffset === 0
        ? {
            items: Array.from({ length: 50 }, (_, index) =>
              subject({ id: `ref-${index + 1}`, name: `Reference ${index + 1}` }),
            ),
            total: 51,
            limit: pageLimit,
            offset: pageOffset,
          }
        : {
            items: [subject({ id: "ref-51", name: "Last reference" })],
            total: 51,
            limit: pageLimit,
            offset: pageOffset,
          };
    });
    renderLibrary();

    expect(await screen.findByText("Showing 1-50 of 51")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByText("Last reference")).toBeTruthy();
    expect(screen.getByText("Showing 51-51 of 51")).toBeTruthy();
    expect(mocked.references).toHaveBeenLastCalledWith("", false, 50, 50);

    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    await waitFor(() => expect(mocked.references).toHaveBeenLastCalledWith("", false, 50, 0));
  });

  it("returns to the first page when a filter changes", async () => {
    mocked.references.mockImplementation(async (_search, _archived, limit, offset) => ({
      items: [subject({ name: (offset ?? 0) === 0 ? "First page" : "Second page" })],
      total: 51,
      limit: limit ?? 50,
      offset: offset ?? 0,
    }));
    renderLibrary();

    fireEvent.click(await screen.findByRole("button", { name: "Next" }));
    expect(await screen.findByText("Second page")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Search references"), {
      target: { value: "Ada" },
    });
    await waitFor(() => expect(mocked.references).toHaveBeenLastCalledWith("Ada", false, 50, 0));

    fireEvent.click(screen.getByLabelText("Show archived"));
    await waitFor(() => expect(mocked.references).toHaveBeenLastCalledWith("Ada", true, 50, 0));
  });

  it("returns to the previous page after deleting its only item", async () => {
    let deleted = false;
    mocked.references.mockImplementation(async (_search, _archived, limit, offset) => ({
      items:
        deleted && (offset ?? 0) > 0
          ? []
          : [subject({ id: (offset ?? 0) === 0 ? "ref-1" : "ref-51" })],
      total: deleted ? 50 : 51,
      limit: limit ?? 50,
      offset: offset ?? 0,
    }));
    mocked.referenceDeletionImpact.mockResolvedValue({
      reference_subject_id: "ref-51",
      name: "Ada Lovelace",
      asset_count: 0,
      exclusive_artifact_ids: [],
    });
    mocked.deleteReference.mockImplementation(async () => {
      deleted = true;
    });
    renderLibrary();

    fireEvent.click(await screen.findByRole("button", { name: "Next" }));
    await screen.findByText("Showing 51-51 of 51");
    expect(mocked.references).toHaveBeenLastCalledWith("", false, 50, 50);
    fireEvent.click(screen.getByLabelText("Delete permanently"));
    fireEvent.click(await screen.findByText("Delete permanently", { selector: "button.danger" }));

    await waitFor(() => expect(mocked.references).toHaveBeenLastCalledWith("", false, 50, 0));
  });

  it("returns to the previous page when archiving removes its only visible item", async () => {
    let archived = false;
    mocked.references.mockImplementation(async (_search, includeArchived, limit, offset) => ({
      items:
        archived && !includeArchived && (offset ?? 0) > 0
          ? []
          : [subject({ id: (offset ?? 0) === 0 ? "ref-1" : "ref-51" })],
      total: archived && !includeArchived ? 50 : 51,
      limit: limit ?? 50,
      offset: offset ?? 0,
    }));
    mocked.updateReference.mockImplementation(async () => {
      archived = true;
      return subject({ id: "ref-51", archived: true });
    });
    renderLibrary();

    fireEvent.click(await screen.findByRole("button", { name: "Next" }));
    await screen.findByText("Showing 51-51 of 51");
    fireEvent.click(screen.getByRole("button", { name: "Archive" }));

    await waitFor(() => expect(mocked.references).toHaveBeenLastCalledWith("", false, 50, 0));
    expect(screen.queryByText("Showing 50-50 of 50")).toBeNull();
  });
});
