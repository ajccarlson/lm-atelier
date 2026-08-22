import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ImageIcon, Plus, Trash2 } from "lucide-react";
import { AssetReview } from "./AssetReview";
import { api } from "./api";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { LibraryImagePicker } from "./LibraryImagePicker";
import { artifactSource } from "./messageMedia";
import type { ArtifactLibraryItem, ReferenceSimilarAsset, ReferenceSubject } from "./types";

/** What one image is for. Closed, because a preparation recipe decides what to
 *  do with an image from its purpose - one nobody implements contributes
 *  nothing, silently. */
const PURPOSES = [
  "identity",
  "appearance",
  "clothing",
  "pose",
  "style",
  "environment",
  "detail",
  "product_view",
  "other",
] as const;

export function ReferenceDetail({
  subject,
  onBack,
}: {
  subject: ReferenceSubject;
  onBack: () => void;
}) {
  const client = useQueryClient();
  const [purpose, setPurpose] = useState<string>("identity");
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Held after an attach rather than shown transiently: the whole point is that
  // the person who just added the image gets to decide what to do about it.
  const [similar, setSimilar] = useState<ReferenceSimilarAsset[]>([]);
  // Refusals are kept per image rather than collapsed into one message. Adding
  // six pictures and being told only "that did not work" hides which five
  // landed, and the answer changes what to do next.
  const [refused, setRefused] = useState<string[]>([]);

  const assets = useQuery({
    queryKey: ["reference-assets", subject.id],
    queryFn: () => api.referenceAssets(subject.id),
  });

  const refresh = () => client.invalidateQueries({ queryKey: ["reference-assets", subject.id] });
  const fail = (reason: unknown) =>
    setError(reason instanceof Error ? reason.message : "That did not work");

  const attach = useMutation({
    mutationFn: async (items: ArtifactLibraryItem[]) => {
      const reports: ReferenceSimilarAsset[] = [];
      const failures: string[] = [];
      // One at a time, and one refusal does not abandon the rest: the set
      // already holding image three is no reason to drop four, five and six.
      for (const item of items) {
        try {
          const result = await api.attachReferenceAsset(subject.id, {
            artifact_id: item.id,
            purpose,
          });
          reports.push(...result.similar);
        } catch (reason) {
          failures.push(reason instanceof Error ? reason.message : "That image was not added");
        }
      }
      return { reports, failures };
    },
    onSuccess: ({ reports, failures }) => {
      setSimilar(reports);
      setRefused(failures);
      setError(null);
      void refresh();
    },
    onError: fail,
  });

  // A comma-separated string is what a person types; the array is what the
  // server stores. Splitting on save rather than on every keystroke means a
  // half-typed "Countess Lovelace" is never briefly two names.
  const asText = (values: string[]) => values.join(", ");
  const asList = (text: string) =>
    text
      .split(",")
      .map((one) => one.trim())
      .filter(Boolean);

  const initialDetails = {
    description: subject.description ?? "",
    aliases: asText(subject.aliases_json),
    tags: asText(subject.tags_json),
  };
  const [savedDetails, setSavedDetails] = useState(initialDetails);
  const [draft, setDraft] = useState(initialDetails);
  const edited =
    draft.description !== savedDetails.description ||
    draft.aliases !== savedDetails.aliases ||
    draft.tags !== savedDetails.tags;

  const details = useMutation({
    mutationFn: () => {
      const changed: {
        description?: string;
        aliases?: string[];
        tags?: string[];
      } = {};
      if (draft.description !== savedDetails.description) changed.description = draft.description;
      if (draft.aliases !== savedDetails.aliases) changed.aliases = asList(draft.aliases);
      if (draft.tags !== savedDetails.tags) changed.tags = asList(draft.tags);
      return api.updateReference(subject.id, changed);
    },
    onSuccess: (updated) => {
      // The server trims and de-duplicates. Its response is the value that was
      // actually saved; retaining the submitted draft would leave the form
      // looking perpetually dirty after a canonicalisation.
      const canonical = {
        description: updated.description ?? "",
        aliases: asText(updated.aliases_json),
        tags: asText(updated.tags_json),
      };
      setSavedDetails(canonical);
      setDraft(canonical);
      setError(null);
      void client.invalidateQueries({ queryKey: ["references"] });
    },
    onError: fail,
  });
  const save = () => details.mutate();

  // Invalidates the subject list rather than the asset list: the cover lives on
  // the subject, and this view reads the subject from the list it was opened
  // from, so refreshing the assets alone would leave the star where it was.
  const cover = useMutation({
    mutationFn: (artifactId: string | null) =>
      artifactId === null
        ? api.clearReferenceCover(subject.id)
        : api.setReferenceCover(subject.id, artifactId),
    onSuccess: () => {
      setError(null);
      void client.invalidateQueries({ queryKey: ["references"] });
    },
    onError: fail,
  });

  const detach = useMutation({
    mutationFn: (assetId: string) => api.detachReferenceAsset(subject.id, assetId),
    onSuccess: () => {
      setSimilar([]);
      setRefused([]);
      void refresh();
    },
    onError: fail,
  });

  const items = assets.data ?? [];

  return (
    <section className="page-view reference-detail" aria-labelledby="reference-detail-heading">
      <header className="page-header">
        <div>
          <h1 id="reference-detail-heading">{subject.name}</h1>
          <p className="muted">
            Written as <code>@{subject.mention_slug}</code> in a chat. {items.length} image
            {items.length === 1 ? "" : "s"}.
          </p>
        </div>
        <div className="row-actions">
          <button className="secondary" onClick={onBack}>
            Back to references
          </button>
        </div>
      </header>

      {error ? (
        <ErrorCallout
          message={error}
          action={
            <button className="secondary compact-button" onClick={() => setError(null)}>
              Dismiss
            </button>
          }
        />
      ) : null}

      {refused.length > 0 ? (
        <ErrorCallout
          message={`${refused.length} image${refused.length === 1 ? " was" : "s were"} not added: ${refused.join("; ")}`}
          action={
            <button className="secondary compact-button" onClick={() => setRefused([])}>
              Dismiss
            </button>
          }
        />
      ) : null}

      {similar.length > 0 ? (
        // Advice, not a refusal. The image was added; this only says the set may
        // now lean toward one look, which the person adding it is best placed
        // to judge.
        <ErrorCallout
          message={
            `That image closely resembles ${similar.length} already here. It was added anyway - ` +
            "remove it if the set is now weighted toward one look."
          }
          action={
            <button className="secondary compact-button" onClick={() => setSimilar([])}>
              Understood
            </button>
          }
        />
      ) : null}

      {/* Held in local state and saved explicitly rather than on every
          keystroke: an alias list that rewrites itself mid-word would fight
          whoever is typing it. */}
      <div className="row-actions">
        <label>
          Description
          <input
            value={draft.description}
            aria-label="Description"
            onChange={(event) => setDraft({ ...draft, description: event.target.value })}
          />
        </label>
        <label>
          Also known as
          <input
            value={draft.aliases}
            aria-label="Other names, separated by commas"
            placeholder="Countess Lovelace, AAL"
            onChange={(event) => setDraft({ ...draft, aliases: event.target.value })}
          />
        </label>
        <label>
          Tags
          <input
            value={draft.tags}
            aria-label="Tags, separated by commas"
            onChange={(event) => setDraft({ ...draft, tags: event.target.value })}
          />
        </label>
        <button className="secondary" disabled={!edited || details.isPending} onClick={save}>
          Save details
        </button>
      </div>

      <div className="row-actions">
        <button className="primary" disabled={attach.isPending} onClick={() => setPicking(true)}>
          <Plus />
          Add images
        </button>
      </div>

      {items.length === 0 && !assets.isLoading ? (
        <EmptyState
          icon={<Plus />}
          title="No images yet"
          body="Add a few from the media library so generations have something to work from."
        />
      ) : null}

      <ul className="reference-asset-grid">
        {items.map((asset) => (
          <li key={asset.id}>
            <img
              src={artifactSource(asset.artifact_id) ?? undefined}
              alt={asset.caption ?? `${subject.name}, ${asset.purpose}`}
              loading="lazy"
            />
            <div className="detail-title">
              <span className="badge">{asset.purpose}</span>
              {/* Unchecked is not a synonym for usable: an image nobody has
                  looked at must not let the set claim a reviewed set's
                  fidelity. */}
              <span className="badge">{asset.validation_state}</span>
            </div>
            <div className="row-actions">
              {/* Pressing the current cover clears it, so there is a way back
                  to no cover at all without removing the picture. */}
              <button
                className="secondary compact-button"
                aria-pressed={subject.cover_artifact_id === asset.artifact_id}
                aria-label={
                  subject.cover_artifact_id === asset.artifact_id
                    ? `Stop image ${asset.sort_order + 1} standing for ${subject.name}`
                    : `Use image ${asset.sort_order + 1} for ${subject.name}`
                }
                disabled={cover.isPending}
                onClick={() =>
                  cover.mutate(
                    subject.cover_artifact_id === asset.artifact_id ? null : asset.artifact_id,
                  )
                }
              >
                <ImageIcon />
              </button>
              <button
                className="secondary compact-button danger"
                aria-label={`Remove image ${asset.sort_order + 1}`}
                onClick={() => detach.mutate(asset.id)}
              >
                <Trash2 />
              </button>
            </div>
            <AssetReview subjectId={subject.id} asset={asset} onReviewed={refresh} />
          </li>
        ))}
      </ul>

      {picking ? (
        <LibraryImagePicker
          title={`Add images of ${subject.name}`}
          confirmLabel="Add"
          onClose={() => setPicking(false)}
          onConfirm={(chosen) => attach.mutate(chosen)}
        >
          {/* Chosen here rather than after the fact: the purpose applies to
              everything picked in this pass, and asking once is the difference
              between labelling a set and labelling six images one at a time. */}
          <label>
            Purpose
            <select value={purpose} onChange={(event) => setPurpose(event.target.value)}>
              {PURPOSES.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        </LibraryImagePicker>
      ) : null}
    </section>
  );
}
