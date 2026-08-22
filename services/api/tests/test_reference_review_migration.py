from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.reference_review_schema import CREATE_REFERENCE_REVIEW_TRIGGER_SQL


def test_reference_review_migration_preserves_unchecked_assets_and_installs_guards(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "reference-review-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f9b7a1c42d60")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-13 12:00:00"
    digest = "a" * 64
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO artifacts
              (id, sha256, kind, media_type, size_bytes, relative_path,
               original_name, metadata_json, favorite, created_at, updated_at)
            VALUES (?, ?, 'input', 'image/png', 1, ?, 'legacy.png', '{}', 0, ?, ?)
            """,
            (f"sha256:{digest}", digest, f"aa/aa/{digest}", timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_subjects
              (id, name, mention_slug, kind, description, aliases_json, tags_json,
               cover_artifact_id, favorite, archived, created_at, updated_at)
            VALUES
              ('refsubject_legacy', 'Legacy', 'legacy', 'person', NULL, '[]', '[]',
               NULL, 0, 0, ?, ?)
            """,
            (timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_assets
              (id, reference_subject_id, artifact_id, caption, purpose, view_label,
               sort_order, validation_state, validation_reasons_json, width, height,
               created_at, updated_at)
            VALUES
              ('refasset_legacy', 'refsubject_legacy', ?, NULL, 'identity', NULL,
               0, 'unchecked', '[]', NULL, NULL, ?, ?)
            """,
            (f"sha256:{digest}", timestamp, timestamp),
        )

    command.upgrade(config, "head")
    expected_triggers = {statement.split()[2] for statement in CREATE_REFERENCE_REVIEW_TRIGGER_SQL}
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT validation_state, validation_reasons_json, width, height, review_version "
            "FROM reference_assets WHERE id = 'refasset_legacy'"
        ).fetchone() == ("unchecked", "[]", None, None, 1)
        assert connection.execute(
            "SELECT count(*) FROM reference_asset_review_events"
        ).fetchone() == (0,)
        migrated_triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        assert expected_triggers <= migrated_triggers

    command.downgrade(config, "f9b7a1c42d60")
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reference_assets)").fetchall()
        }
        assert "review_version" not in columns
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'reference_asset_review_events'"
            ).fetchone()
            is None
        )
