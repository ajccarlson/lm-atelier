import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Star, Archive, Trash2 } from "lucide-react";
import { api } from "./api";
import { AccessibleDialog } from "./AccessibleDialog";
import { EmptyState } from "./EmptyState";
import { ErrorCallout } from "./ErrorCallout";
import { ReferenceDetail } from "./ReferenceDetail";
import { artifactSource } from "./messageMedia";
import type { ReferenceDeletionImpact, ReferenceSubject } from "./types";

/** The kinds the server accepts. A closed set, because a workflow declares
 *  which kinds it can condition on - an invented one could never be matched. */
const KINDS = [
  "person",
  "character",
  "object",
  "product",
  "place",
  "style",
  "wardrobe",
  "pose",
  "composition",
  "other",
] as const;

const REFERENCE_PAGE_SIZE = 50;

export function ReferencesLibrary() {
  const client = useQueryClient();
  const [search, setSearch] = useState("");
  const [includeArchived, setIncludeArchived] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [kind, setKind] = useState<string>("person");
  const [error, setError] = useState<string | null>(null);
  // Deletion is confirmed against the impact the user was actually shown, so
  // the impact is held here rather than re-fetched at the moment of deleting.
  const [pendingDelete, setPendingDelete] = useState<ReferenceDeletionImpact | null>(null);
  const [opened, setOpened] = useState<ReferenceSubject | null>(null);
  const [offset, setOffset] = useState(0);

  const references = useQuery({
    queryKey: ["references", search, includeArchived, offset],
    queryFn: () => api.references(search, includeArchived, REFERENCE_PAGE_SIZE, offset),
  });

  const refresh = () => client.invalidateQueries({ queryKey: ["references"] });
  const fail = (reason: unknown) =>
    setError(reason instanceof Error ? reason.message : "That did not work");

  const create = useMutation({
    mutationFn: () => api.createReference({ name, kind }),
    onSuccess: () => {
      setCreating(false);
      setName("");
      setError(null);
      setOffset(0);
      void refresh();
    },
    onError: fail,
  });

  const update = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Record<string, unknown> }) =>
      api.updateReference(id, body),
    onSuccess: () => void refresh(),
    onError: fail,
  });

  const remove = useMutation({
    mutationFn: (impact: ReferenceDeletionImpact) =>
      api.deleteReference(impact.reference_subject_id, impact.asset_count),
    onSuccess: () => {
      setPendingDelete(null);
      void refresh();
    },
    onError: fail,
  });

  const askToDelete = async (subject: ReferenceSubject) => {
    try {
      setPendingDelete(await api.referenceDeletionImpact(subject.id));
    } catch (reason) {
      fail(reason);
    }
  };

  const items = references.data?.items ?? [];
  const total = references.data?.total ?? 0;
  const shownFrom = total === 0 ? 0 : Math.min(offset + 1, total);
  const shownThrough = Math.min(offset + items.length, total);
  const hasPrevious = offset > 0;
  const hasNext = offset + items.length < total;

  useEffect(() => {
    if (!references.data || offset === 0 || items.length > 0) return;
    const lastOffset =
      total === 0 ? 0 : Math.floor((total - 1) / REFERENCE_PAGE_SIZE) * REFERENCE_PAGE_SIZE;
    if (lastOffset >= offset) return;
    const timeout = window.setTimeout(() => {
      setOffset((current) => (current === offset ? lastOffset : current));
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [items.length, offset, references.data, total]);

  // The detail view replaces the list rather than nesting inside it, so the
  // back control is the only way out and cannot be confused with the nav.
  if (opened) {
    const current = items.find((item) => item.id === opened.id) ?? opened;
    return <ReferenceDetail subject={current} onBack={() => setOpened(null)} />;
  }

  return (
    <section className="page-view reference-library" aria-labelledby="references-heading">
      <header className="page-header">
        <h1 id="references-heading">References</h1>
        <p className="muted">
          Subjects you can name in a chat with <code>@</code>, and the images that show them.
        </p>
        <div className="row-actions">
          <input
            type="search"
            value={search}
            aria-label="Search references"
            placeholder="Search by name"
            onChange={(event) => {
              setSearch(event.target.value);
              setOffset(0);
            }}
          />
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={includeArchived}
              onChange={(event) => {
                setIncludeArchived(event.target.checked);
                setOffset(0);
              }}
            />
            Show archived
          </label>
          <button className="primary" onClick={() => setCreating(true)}>
            <Plus />
            New reference
          </button>
        </div>
      </header>

      {error ? (
        <ErrorCallout
          message={error}
          action={<button className="secondary compact-button" onClick={() => setError(null)}>Dismiss</button>}
        />
      ) : null}

      {references.isLoading ? <p>Loading references…</p> : null}

      {!references.isLoading && total === 0 ? (
        <EmptyState
          icon={<Plus />}
          title="No references yet"
          body="Create one, add a few images of it, then write @its-name in any chat."
        />
      ) : null}

      <ul className="reference-list">
        {items.map((subject) => (
          <li key={subject.id} className={subject.archived ? "archived" : ""}>
            {/* Only when one was chosen. A placeholder for every subject
                without a cover would add a column of empty boxes and make the
                list harder to scan, which is the opposite of the point. */}
            {subject.cover_artifact_id ? (
              <img
                className="reference-cover"
                src={artifactSource(subject.cover_artifact_id) ?? undefined}
                alt=""
                loading="lazy"
              />
            ) : null}
            <div className="detail-title">
              <button className="link-button" onClick={() => setOpened(subject)}>
                <strong>{subject.name}</strong>
              </button>
              {/* The mention is the addressing token and never changes silently
                  with the name, so it is shown next to it rather than implied. */}
              <code>@{subject.mention_slug}</code>
              <span className="badge">{subject.kind}</span>
              {subject.archived ? <span className="badge">Archived</span> : null}
            </div>
            <div className="row-actions">
              <button
                className="secondary compact-button"
                aria-pressed={subject.favorite}
                aria-label={subject.favorite ? "Remove from favourites" : "Add to favourites"}
                onClick={() =>
                  update.mutate({ id: subject.id, body: { favorite: !subject.favorite } })
                }
              >
                <Star />
              </button>
              <button
                className="secondary compact-button"
                aria-label={subject.archived ? "Restore" : "Archive"}
                onClick={() =>
                  update.mutate({ id: subject.id, body: { archived: !subject.archived } })
                }
              >
                <Archive />
              </button>
              <button className="secondary compact-button danger" aria-label="Delete permanently" onClick={() => void askToDelete(subject)}>
                <Trash2 />
              </button>
            </div>
          </li>
        ))}
      </ul>

      {total > 0 ? (
        <nav className="row-actions" aria-label="Reference pages">
          <p className="muted" aria-live="polite">
            Showing {shownFrom}-{shownThrough} of {total}
          </p>
          <button
            className="secondary compact-button"
            disabled={!hasPrevious || references.isFetching}
            onClick={() => setOffset(Math.max(0, offset - REFERENCE_PAGE_SIZE))}
          >
            Previous
          </button>
          <button
            className="secondary compact-button"
            disabled={!hasNext || references.isFetching}
            onClick={() => setOffset(offset + REFERENCE_PAGE_SIZE)}
          >
            Next
          </button>
        </nav>
      ) : null}

      {creating ? (
        <AccessibleDialog
          title="New reference"
          eyebrow="References"
          closeLabel="Close"
          onClose={() => setCreating(false)}
        >
        <label>
          Name
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <label>
          Kind
          <select value={kind} onChange={(event) => setKind(event.target.value)}>
            {KINDS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <p className="muted">
          The mention is derived from the name. If another reference already answers to it, this
          one gets a numbered variant rather than being refused.
        </p>
        <div className="row-actions">
          <button className="secondary" onClick={() => setCreating(false)}>Cancel</button>
          <button
            className="primary"
            disabled={!name.trim() || create.isPending}
            onClick={() => create.mutate()}
          >
            Create
          </button>
        </div>
        </AccessibleDialog>
      ) : null}

      {pendingDelete ? (
        <AccessibleDialog
          title="Delete this reference permanently?"
          eyebrow="References"
          closeLabel="Cancel"
          onClose={() => setPendingDelete(null)}
        >
          <>
            <p>
              <strong>{pendingDelete.name}</strong> holds {pendingDelete.asset_count} image
              {pendingDelete.asset_count === 1 ? "" : "s"}.
            </p>
            {/* Only images nobody else references are actually lost. A picture
                showing two subjects belongs to both, and removing one of them
                is not permission to destroy it. */}
            <p>
              {pendingDelete.exclusive_artifact_ids.length === 0
                ? "No images would be lost - every one is used elsewhere too."
                : `${pendingDelete.exclusive_artifact_ids.length} image(s) are used only here and would be lost.`}
            </p>
            <p className="muted">
              Archiving hides a reference without destroying anything, and can be undone. This
              cannot.
            </p>
            <div className="row-actions">
              <button className="secondary" onClick={() => setPendingDelete(null)}>Cancel</button>
              <button
                className="secondary compact-button danger"
                disabled={remove.isPending}
                onClick={() => remove.mutate(pendingDelete)}
              >
                Delete permanently
              </button>
            </div>
          </>
        </AccessibleDialog>
      ) : null}
    </section>
  );
}
