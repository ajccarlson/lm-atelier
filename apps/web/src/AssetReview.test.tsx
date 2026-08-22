import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { api } from "./api";
import { AssetReview } from "./AssetReview";
import type { ReferenceAsset } from "./types";

vi.mock("./api", () => ({ api: { reviewReferenceAsset: vi.fn() } }));
const mocked = vi.mocked(api);

const ASSET: ReferenceAsset = {
  id: "refasset-1",
  reference_subject_id: "refsubject-1",
  artifact_id: "sha256:abc",
  caption: null,
  purpose: "identity",
  view_label: null,
  sort_order: 0,
  validation_state: "unchecked",
  validation_reasons_json: [],
  width: null,
  height: null,
  review_version: 1,
};

function show(asset: ReferenceAsset = ASSET, onReviewed = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <AssetReview subjectId="refsubject-1" asset={asset} onReviewed={onReviewed} />
    </QueryClientProvider>,
  );
  return onReviewed;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("quotes the exact unchecked review version when allowing an image", async () => {
  const reviewed = vi.fn();
  mocked.reviewReferenceAsset.mockResolvedValue({
    asset: { ...ASSET, validation_state: "usable", review_version: 2 },
    review: { id: "event-1", result_version: 2, decision: "usable", decision_sha256: "a" },
    idempotent: false,
  });
  show(ASSET, reviewed);

  fireEvent.click(screen.getByRole("button", { name: /Review image 1/ }));

  await waitFor(() =>
    expect(mocked.reviewReferenceAsset).toHaveBeenCalledWith(
      "refsubject-1",
      "refasset-1",
      {
        expected_state: "unchecked",
        expected_version: 1,
        decision: "usable",
        reasons: [],
      },
    ),
  );
  await waitFor(() => expect(reviewed).toHaveBeenCalled());
});

it("requires and sends a rejection reason", async () => {
  mocked.reviewReferenceAsset.mockResolvedValue({
    asset: { ...ASSET, validation_state: "rejected", review_version: 2 },
    review: { id: "event-2", result_version: 2, decision: "rejected", decision_sha256: "b" },
    idempotent: false,
  });
  show();

  fireEvent.click(screen.getByRole("button", { name: "Reject image 1" }));
  const submit = screen.getByRole("button", { name: "Reject" });
  expect(submit).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Why image 1 is not usable"), {
    target: { value: "  wrong view  " },
  });
  fireEvent.click(submit);

  await waitFor(() =>
    expect(mocked.reviewReferenceAsset).toHaveBeenCalledWith(
      "refsubject-1",
      "refasset-1",
      expect.objectContaining({ decision: "rejected", reasons: ["wrong view"] }),
    ),
  );
});

it("does not offer a second decision for a settled image", () => {
  show({
    ...ASSET,
    validation_state: "usable",
    validation_reasons_json: [],
    width: 512,
    height: 512,
    review_version: 2,
  });

  expect(screen.getByText("Verified 512 × 512")).toBeTruthy();
  expect(screen.queryByRole("button", { name: /Review image/ })).toBeNull();
  expect(screen.queryByRole("button", { name: /Reject image/ })).toBeNull();
});
