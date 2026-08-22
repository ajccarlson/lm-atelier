import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "./api";
import type { ReferenceAsset } from "./types";

/** Settle the one human review that permits an exact Reference image to run. */
export function AssetReview({
  subjectId,
  asset,
  onReviewed,
}: {
  subjectId: string;
  asset: ReferenceAsset;
  onReviewed: () => void;
}) {
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const settled = asset.validation_state !== "unchecked";
  const review = useMutation({
    mutationFn: (decision: "usable" | "rejected") =>
      api.reviewReferenceAsset(subjectId, asset.id, {
        expected_state: "unchecked",
        expected_version: asset.review_version,
        decision,
        reasons: decision === "rejected" ? [reason.trim()] : [],
      }),
    onSuccess: () => {
      setError(null);
      setRejecting(false);
      setReason("");
      onReviewed();
    },
    onError: (value: unknown) =>
      setError(value instanceof Error ? value.message : "The review was not saved."),
  });

  if (settled) {
    return (
      <div className="asset-review">
        {asset.width && asset.height ? (
          <span className="muted">Verified {asset.width} × {asset.height}</span>
        ) : null}
        {asset.validation_reasons_json.map((item) => (
          <span className="muted" key={item}>{item}</span>
        ))}
      </div>
    );
  }

  return (
    <div className="asset-review">
      {rejecting ? (
        <div className="row-actions">
          <input
            aria-label={`Why image ${asset.sort_order + 1} is not usable`}
            value={reason}
            placeholder="What is wrong with it?"
            onChange={(event) => setReason(event.target.value)}
          />
          <button
            className="secondary compact-button danger"
            disabled={!reason.trim() || review.isPending}
            onClick={() => review.mutate("rejected")}
          >
            Reject
          </button>
          <button className="secondary compact-button" onClick={() => setRejecting(false)}>
            Cancel
          </button>
        </div>
      ) : (
        <div className="row-actions">
          <button
            className="secondary compact-button"
            disabled={review.isPending}
            aria-label={`Review image ${asset.sort_order + 1} and allow it in generations`}
            onClick={() => review.mutate("usable")}
          >
            Review and use
          </button>
          <button
            className="secondary compact-button danger"
            disabled={review.isPending}
            aria-label={`Reject image ${asset.sort_order + 1}`}
            onClick={() => setRejecting(true)}
          >
            Reject
          </button>
        </div>
      )}
      {error ? <p className="error-text" role="status">{error}</p> : null}
    </div>
  );
}
