"""SQLite guards for authoritative, immutable Reference asset reviews."""

from __future__ import annotations

ASSET_REVIEW_INSERT_TRIGGER = """
CREATE TRIGGER reference_asset_review_insert_guard
BEFORE INSERT ON reference_assets
BEGIN
  SELECT CASE WHEN NEW.validation_state != 'unchecked'
                        OR NEW.review_version != 1
                        OR NEW.width IS NOT NULL OR NEW.height IS NOT NULL
                        OR NOT json_valid(NEW.validation_reasons_json)
                        OR json_type(NEW.validation_reasons_json) != 'array'
                        OR json_array_length(NEW.validation_reasons_json) != 0
    THEN RAISE(ABORT, 'reference asset must begin unchecked') END;
END
"""

ASSET_REVIEW_UPDATE_TRIGGER = """
CREATE TRIGGER reference_asset_review_update_guard
BEFORE UPDATE OF validation_state, validation_reasons_json, width, height, review_version
ON reference_assets
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id
                        OR NEW.reference_subject_id != OLD.reference_subject_id
                        OR NEW.artifact_id != OLD.artifact_id
    THEN RAISE(ABORT, 'reference asset review identity changed') END;
  SELECT CASE WHEN OLD.validation_state != 'unchecked'
    THEN RAISE(ABORT, 'reference asset review is already settled') END;
  SELECT CASE WHEN NEW.validation_state NOT IN ('usable', 'weak', 'rejected')
    THEN RAISE(ABORT, 'reference asset review decision is invalid') END;
  SELECT CASE WHEN NEW.review_version != OLD.review_version + 1
    THEN RAISE(ABORT, 'reference asset review version is stale') END;
  SELECT CASE WHEN NOT json_valid(NEW.validation_reasons_json)
                        OR json_type(NEW.validation_reasons_json) != 'array'
                        OR (SELECT count(*) FROM json_each(NEW.validation_reasons_json)) > 16
                        OR EXISTS (
                          SELECT 1 FROM json_each(NEW.validation_reasons_json)
                          WHERE type != 'text' OR length(value) NOT BETWEEN 1 AND 500
                        )
    THEN RAISE(ABORT, 'reference asset review reasons are invalid') END;
  SELECT CASE WHEN NEW.width IS NULL OR NEW.height IS NULL
                        OR NEW.width < 1 OR NEW.height < 1
    THEN RAISE(ABORT, 'reference asset review dimensions are invalid') END;
END
"""

ASSET_REVIEW_IDENTITY_TRIGGER = """
CREATE TRIGGER reference_asset_review_identity_guard
BEFORE UPDATE OF id, reference_subject_id, artifact_id ON reference_assets
WHEN OLD.validation_state != 'unchecked'
BEGIN
  SELECT RAISE(ABORT, 'reviewed reference asset identity is immutable');
END
"""

REVIEW_EVENT_INSERT_TRIGGER = """
CREATE TRIGGER reference_asset_review_event_insert_guard
BEFORE INSERT ON reference_asset_review_events
BEGIN
  SELECT CASE WHEN NEW.id != 'refreview:sha256:' || NEW.decision_sha256
    THEN RAISE(ABORT, 'reference asset review identity is invalid') END;
  SELECT CASE WHEN length(NEW.decision_sha256) != 64
                        OR lower(NEW.decision_sha256) != NEW.decision_sha256
                        OR replace(replace(replace(replace(replace(replace(replace(replace(
                           replace(replace(replace(replace(replace(replace(replace(replace(
                           NEW.decision_sha256,
                           '0',''),'1',''),'2',''),'3',''),'4',''),'5',''),'6',''),'7',''),
                           '8',''),'9',''),'a',''),'b',''),'c',''),'d',''),'e',''),'f','') != ''
    THEN RAISE(ABORT, 'reference asset review digest is invalid') END;
  SELECT CASE WHEN NEW.reviewer_kind != 'local-human'
                        OR NEW.decision NOT IN ('usable', 'weak', 'rejected')
                        OR NEW.expected_state != 'unchecked'
                        OR NEW.result_version != NEW.expected_version + 1
                        OR NEW.width < 1 OR NEW.height < 1
                        OR NOT json_valid(NEW.reasons_json)
                        OR json_type(NEW.reasons_json) != 'array'
    THEN RAISE(ABORT, 'reference asset review event is invalid') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM reference_assets AS asset
    JOIN artifacts AS artifact ON artifact.id = asset.artifact_id
    WHERE asset.id = NEW.reference_asset_id
      AND asset.artifact_id = NEW.artifact_id
      AND artifact.sha256 = NEW.artifact_sha256
      AND asset.validation_state = NEW.decision
      AND asset.validation_reasons_json = NEW.reasons_json
      AND asset.width = NEW.width
      AND asset.height = NEW.height
      AND asset.review_version = NEW.result_version
  ) THEN RAISE(ABORT, 'reference asset review event does not match settled asset') END;
END
"""

REVIEW_EVENT_UPDATE_TRIGGER = """
CREATE TRIGGER reference_asset_review_event_update_guard
BEFORE UPDATE ON reference_asset_review_events
BEGIN
  SELECT RAISE(ABORT, 'reference asset review events are immutable');
END
"""

REVIEW_EVENT_DELETE_TRIGGER = """
CREATE TRIGGER reference_asset_review_event_delete_guard
BEFORE DELETE ON reference_asset_review_events
BEGIN
  SELECT RAISE(ABORT, 'reference asset review events are immutable');
END
"""

CREATE_REFERENCE_REVIEW_TRIGGER_SQL = (
    ASSET_REVIEW_INSERT_TRIGGER,
    ASSET_REVIEW_UPDATE_TRIGGER,
    ASSET_REVIEW_IDENTITY_TRIGGER,
    REVIEW_EVENT_INSERT_TRIGGER,
    REVIEW_EVENT_UPDATE_TRIGGER,
    REVIEW_EVENT_DELETE_TRIGGER,
)

DROP_REFERENCE_REVIEW_TRIGGER_SQL = (
    "DROP TRIGGER reference_asset_review_event_delete_guard",
    "DROP TRIGGER reference_asset_review_event_update_guard",
    "DROP TRIGGER reference_asset_review_event_insert_guard",
    "DROP TRIGGER reference_asset_review_identity_guard",
    "DROP TRIGGER reference_asset_review_update_guard",
    "DROP TRIGGER reference_asset_review_insert_guard",
)
