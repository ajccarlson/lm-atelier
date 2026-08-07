from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError

from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.database_migrations import (
    DatabaseUpgradeError,
    DatabaseVersionError,
    alembic_config,
    upgrade_database,
)


def test_unknown_database_revision_has_actionable_error(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "newer-database")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    unknown_revision = "newer_lm_atelier_revision"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY)")
        connection.execute(
            "INSERT INTO alembic_version VALUES (?)",
            (unknown_revision,),
        )

    with pytest.raises(
        DatabaseVersionError,
        match="database schema revision that this build does not recognize",
    ) as captured:
        upgrade_database(settings)

    assert isinstance(captured.value.__cause__, CommandError)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            unknown_revision,
        )


def test_unrelated_alembic_command_error_is_not_reclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "migration-failure")
    expected = CommandError("unrelated migration failure")

    def fail_upgrade(*_args: object, **_kwargs: object) -> None:
        raise expected

    monkeypatch.setattr(command, "upgrade", fail_upgrade)

    with pytest.raises(CommandError) as captured:
        upgrade_database(settings)

    assert captured.value is expected


def test_migrations_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "migrations")
    settings.prepare()
    config = alembic_config(settings)
    upgrade_database(settings)
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
        chat_columns = {row[1] for row in connection.execute("PRAGMA table_info(chats)").fetchall()}
        project_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
        }
        profile_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(model_profiles)").fetchall()
        }
        asset_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(model_asset_installs)").fetchall()
        }
        registry_install_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(comfy_registry_installs)").fetchall()
        }
        workflow_definition_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(workflow_definitions)").fetchall()
        }
        workflow_revision_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(workflow_revisions)").fetchall()
        }
        workflow_family_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(workflow_families)").fetchall()
        }
        workflow_preference_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(workflow_preferences)").fetchall()
        }
        workflow_preference_indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(workflow_preferences)").fetchall()
        }
        workflow_install_offer_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(workflow_install_offers)").fetchall()
        }
        workflow_install_offer_foreign_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(workflow_install_offers)"
            ).fetchall()
        }
        workflow_install_offer_indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(workflow_install_offers)").fetchall()
        }
        workflow_default_index_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND name = 'uq_workflow_preferences_default_selector'
            """
        ).fetchone()
        workflow_definition_foreign_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(workflow_definitions)"
            ).fetchall()
        }
        workflow_preference_foreign_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(workflow_preferences)"
            ).fetchall()
        }
        workflow_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_definitions)").fetchall()
            if row[1] == "id"
        )
        workflow_revision_definition_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_revisions)").fetchall()
            if row[1] == "workflow_id"
        )
        registry_install_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(comfy_registry_installs)").fetchall()
            if row[1] == "id"
        )
        component_manifest_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_component_manifests)").fetchall()
            if row[1] == "id"
        )
        capability_evidence_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_capability_evidence)").fetchall()
            if row[1] == "id"
        )
        unique_run_indexes = {
            tuple(
                column[2]
                for column in connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall()
            )
            for index in connection.execute("PRAGMA index_list(runs)").fetchall()
            if index[2]
        }
    assert {
        "projects",
        "messages",
        "generation_presets",
        "custom_node_installs",
        "comfy_registry_installs",
        "model_asset_installs",
        "turn_creation_claims",
        "workflow_families",
        "workflow_preferences",
        "workflow_dependency_slots",
        "workflow_activations",
        "workflow_dependency_bindings",
        "workflow_profile_compatibility",
        "chat_workflow_selections",
        "project_workflow_selections",
        "workflow_install_offers",
    } <= tables
    assert {
        "active_head_message_id",
        "active_vision_profile_id",
        "generation_settings_json",
        "generation_preset_ids_json",
        "vision_settings_json",
    } <= chat_columns
    assert {
        "image_workflow_revision_id",
        "video_workflow_revision_id",
        "generation_settings_json",
        "generation_preset_ids_json",
    } <= project_columns
    assert "use_case" in profile_columns
    assert {
        "source_id",
        "name",
        "kind",
        "family",
        "local_path",
        "size_bytes",
        "manifest_json",
        "active",
        "verified_at",
    } <= asset_columns
    assert {
        "package_id",
        "package_version",
        "registry_record_id",
        "archive_sha256",
        "manifest_sha256",
        "installed_path",
        "node_types_json",
        "pip_dependencies_json",
        "review_json",
        "wheel_closure_sha256",
        "wheel_environment_sha256",
        "wheel_environment_path",
        "trusted",
        "active",
    } <= registry_install_columns
    assert {"family_id", "variant_key"} <= workflow_definition_columns
    assert {"capabilities_json", "dependency_contract_sha256"} <= workflow_revision_columns
    assert {
        "id",
        "name",
        "description",
        "use_case",
        "tags_json",
        "enabled",
        "archived",
    } <= workflow_family_columns
    assert {
        "id",
        "workflow_family_id",
        "selector_capability",
        "enabled",
        "is_default",
        "sort_order",
    } <= workflow_preference_columns
    assert {
        "ix_workflow_preferences_selector_order",
        "uq_workflow_preferences_default_selector",
    } <= workflow_preference_indexes
    assert {
        "id",
        "workflow_revision_id",
        "workflow_artifact_sha256",
        "dependency_contract_sha256",
        "binding_plan_sha256",
        "offer_sha256",
        "selections_json",
        "assets_json",
        "plan_count",
        "total_bytes",
        "status",
        "queued_at",
        "completed_at",
        "invalidated_at",
        "invalidation_code",
        "invalidation_reason",
    } <= workflow_install_offer_columns
    assert {
        "ix_workflow_install_offers_workflow_revision_id",
        "ix_workflow_install_offers_offer_sha256",
        "ix_workflow_install_offers_status",
        "ix_workflow_install_offer_revision_status",
    } <= workflow_install_offer_indexes
    assert workflow_default_index_sql is not None
    assert "WHERE is_default = 1" in workflow_default_index_sql[0]
    assert (
        "family_id",
        "workflow_families",
        "id",
        "SET NULL",
    ) in workflow_definition_foreign_keys
    assert (
        "workflow_family_id",
        "workflow_families",
        "id",
        "CASCADE",
    ) in workflow_preference_foreign_keys
    assert (
        "workflow_revision_id",
        "workflow_revisions",
        "id",
        "CASCADE",
    ) in workflow_install_offer_foreign_keys
    assert ("chat_id", "idempotency_key") in unique_run_indexes
    assert ("idempotency_key",) not in unique_run_indexes
    assert workflow_id_type == "VARCHAR(64)"
    assert workflow_revision_definition_type == "VARCHAR(64)"
    assert registry_install_id_type == "VARCHAR(64)"
    assert component_manifest_id_type == "VARCHAR(64)"
    assert capability_evidence_id_type == "VARCHAR(64)"

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_workflow_family_migration_preserves_existing_revisions(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "workflow-family-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "d8b3e6f92a41")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-03 00:00:00"
    definition = (
        "workflow_existing",
        "Existing workflow",
        "image_to_image",
        "Existing description",
        "revision_existing",
        timestamp,
        timestamp,
    )
    second_definition = (
        "workflow_existing_alternative",
        "Existing alternative",
        "image_to_image",
        "Another workflow with the same legacy operation",
        None,
        timestamp,
        timestamp,
    )
    revision = (
        "revision_existing",
        "workflow_existing",
        3,
        "comfyui",
        "0.3.50",
        '{"nodes":[{"id":1}]}',
        '{"1":{"class_type":"KSampler"}}',
        '{"type":"object"}',
        '{"model_install_ids":["model_existing"]}',
        "c" * 64,
        1,
        timestamp,
        timestamp,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO workflow_definitions (
                id, name, operation, description, current_revision_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            definition,
        )
        connection.execute(
            """
            INSERT INTO workflow_definitions (
                id, name, operation, description, current_revision_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            second_definition,
        )
        connection.execute(
            """
            INSERT INTO workflow_revisions (
                id, workflow_id, version, engine, engine_version,
                ui_graph_json, api_graph_json, input_schema_json,
                dependencies_json, artifact_sha256, trusted,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            revision,
        )
        connection.commit()

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        migrated_definition = connection.execute(
            """
            SELECT id, name, operation, description, current_revision_id,
                   created_at, updated_at, family_id, variant_key
            FROM workflow_definitions
            WHERE id = 'workflow_existing'
            """
        ).fetchone()
        migrated_revision = connection.execute(
            """
            SELECT id, workflow_id, version, engine, engine_version,
                   ui_graph_json, api_graph_json, input_schema_json,
                   capabilities_json, dependencies_json, artifact_sha256, trusted,
                   created_at, updated_at
            FROM workflow_revisions
            WHERE id = 'revision_existing'
            """
        ).fetchone()
        assert migrated_definition == (*definition, None, None)
        assert migrated_revision == (*revision[:8], "[]", *revision[8:])
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM workflow_definitions
            WHERE operation = 'image_to_image'
              AND family_id IS NULL
              AND variant_key IS NULL
            """
        ).fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM workflow_families").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM workflow_preferences").fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO workflow_families (
                id, name, description, use_case, tags_json, enabled, archived,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wffamily_delete_probe",
                "Delete probe",
                "",
                "",
                "[]",
                1,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            UPDATE workflow_definitions
            SET family_id = 'wffamily_delete_probe', variant_key = 'edit'
            WHERE id = 'workflow_existing'
            """
        )
        connection.execute(
            """
            INSERT INTO workflow_preferences (
                id, workflow_family_id, selector_capability, enabled,
                is_default, sort_order, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wfpref_delete_probe",
                "wffamily_delete_probe",
                "image",
                1,
                1,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute("DELETE FROM workflow_families WHERE id = 'wffamily_delete_probe'")
        assert connection.execute(
            "SELECT family_id, variant_key FROM workflow_definitions WHERE id = 'workflow_existing'"
        ).fetchone() == (None, "edit")
        assert connection.execute("SELECT COUNT(*) FROM workflow_preferences").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_revisions WHERE id = 'revision_existing'"
        ).fetchone() == (1,)
        connection.commit()

    command.downgrade(config, "d8b3e6f92a41")

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        restored_definition = connection.execute(
            """
            SELECT id, name, operation, description, current_revision_id,
                   created_at, updated_at
            FROM workflow_definitions
            WHERE id = 'workflow_existing'
            """
        ).fetchone()
        restored_revision = connection.execute(
            """
            SELECT id, workflow_id, version, engine, engine_version,
                   ui_graph_json, api_graph_json, input_schema_json,
                   dependencies_json, artifact_sha256, trusted,
                   created_at, updated_at
            FROM workflow_revisions
            WHERE id = 'revision_existing'
            """
        ).fetchone()
        assert "workflow_families" not in tables
        assert "workflow_preferences" not in tables
        assert restored_definition == definition
        assert restored_revision == revision
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_definitions WHERE operation = 'image_to_image'"
        ).fetchone() == (2,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_workflow_family_migration_refuses_lossy_downgrade(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "workflow-family-downgrade")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-03 00:00:00"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO workflow_families (
                id, name, description, use_case, tags_json, enabled, archived,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wffamily_keep",
                "Keep this family",
                "",
                "",
                "[]",
                1,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="Cannot downgrade workflow families"):
        command.downgrade(config, "d8b3e6f92a41")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "b6a1e4d92c70",
        )
        assert connection.execute(
            "SELECT name FROM workflow_families WHERE id = 'wffamily_keep'"
        ).fetchone() == ("Keep this family",)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_generated_identifier_width_migration_preserves_existing_rows(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "generated-identifier-widths")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "e6c42a9b13fd")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-02 00:00:00"
    workflow_id = f"workflow_{'a' * 32}"
    revision_id = f"wfrev_{'b' * 32}"
    registry_id = f"registry_{'c' * 32}"
    model_id = f"model_{'d' * 32}"
    component_id = f"component_{'e' * 32}"
    evidence_id = f"evidence_{'f' * 32}"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO model_installs (
                id, name, role, engine, local_path, size_bytes, compatibility,
                manifest_json, active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                "Existing model",
                "image",
                "comfyui",
                "models/existing.safetensors",
                1,
                "ready",
                "{}",
                1,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO model_component_manifests (
                id, model_install_id, kind, relative_path, target_folder,
                required, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                component_id,
                model_id,
                "checkpoint",
                "existing.safetensors",
                "checkpoints",
                1,
                "{}",
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO model_capability_evidence (
                id, model_install_id, evidence_key, result,
                component_hashes_json, runtime_build, adapter_contract_version,
                launch_contract_version, hardware_class, probe_version,
                details_json, probed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                model_id,
                "existing-evidence",
                "passed",
                "{}",
                "test-runtime",
                1,
                "test-contract",
                "test-hardware",
                "test-probe",
                "{}",
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO workflow_definitions (
                id, name, operation, description, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                "Existing workflow",
                "text_to_image",
                "",
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO workflow_revisions (
                id, workflow_id, version, engine, ui_graph_json, api_graph_json,
                input_schema_json, dependencies_json, trusted, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                workflow_id,
                1,
                "comfyui",
                "{}",
                "{}",
                "{}",
                "{}",
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO comfy_registry_installs (
                id, package_id, package_version, registry_record_id, repository_url,
                download_url, archive_sha256, manifest_sha256, installed_path,
                node_types_json, pip_dependencies_json, review_json, trusted, active,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                registry_id,
                "example-node",
                "1.2.3",
                "record-123",
                "https://github.com/example/node",
                "https://api.comfy.org/example.zip",
                "d" * 64,
                "e" * 64,
                "example-node",
                "[]",
                "[]",
                "{}",
                0,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        workflow_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_definitions)").fetchall()
            if row[1] == "id"
        )
        revision_workflow_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_revisions)").fetchall()
            if row[1] == "workflow_id"
        )
        registry_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(comfy_registry_installs)").fetchall()
            if row[1] == "id"
        )
        component_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_component_manifests)").fetchall()
            if row[1] == "id"
        )
        evidence_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_capability_evidence)").fetchall()
            if row[1] == "id"
        )
        assert workflow_type == "VARCHAR(64)"
        assert revision_workflow_type == "VARCHAR(64)"
        assert registry_type == "VARCHAR(64)"
        assert component_type == "VARCHAR(64)"
        assert evidence_type == "VARCHAR(64)"
        assert connection.execute(
            "SELECT id FROM workflow_definitions WHERE id = ?",
            (workflow_id,),
        ).fetchone() == (workflow_id,)
        assert connection.execute(
            "SELECT workflow_id FROM workflow_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone() == (workflow_id,)
        assert connection.execute(
            "SELECT id FROM comfy_registry_installs WHERE id = ?",
            (registry_id,),
        ).fetchone() == (registry_id,)
        assert connection.execute(
            "SELECT id FROM model_component_manifests WHERE id = ?",
            (component_id,),
        ).fetchone() == (component_id,)
        assert connection.execute(
            "SELECT id FROM model_capability_evidence WHERE id = ?",
            (evidence_id,),
        ).fetchone() == (evidence_id,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_generation_defaults_migration_backfills_existing_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "existing-generation-defaults")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f7a2c9e51b40")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-07-25 00:00:00"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO projects (
                id, name, description, instructions, archived, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("project-existing", "Existing", "", "", 0, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO chats (
                id, project_id, title, archived, routing_mode, confirm_uncertain_media,
                active_chat_profile_id, active_image_profile_id, active_video_profile_id,
                active_head_message_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chat-existing",
                "project-existing",
                "Existing",
                0,
                "auto",
                1,
                None,
                None,
                None,
                None,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        rows = [
            connection.execute(
                """
                SELECT generation_settings_json, generation_preset_ids_json
                FROM projects
                WHERE id = ?
                """,
                ("project-existing",),
            ).fetchone(),
            connection.execute(
                """
                SELECT generation_settings_json, generation_preset_ids_json
                FROM chats
                WHERE id = ?
                """,
                ("chat-existing",),
            ).fetchone(),
        ]
        for row in rows:
            assert row is not None
            assert json.loads(row[0]) == {}
            assert json.loads(row[1]) == {}


def test_sanitized_v017_database_upgrades_to_head_without_data_loss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "v017-upgrade")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    fixture = Path(__file__).parent / "fixtures" / "v0.1.7-sanitized.sql"
    with sqlite3.connect(database) as connection:
        connection.executescript(fixture.read_text(encoding="utf-8"))

    upgrade_database(settings)

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        project = connection.execute(
            """
            SELECT name, instructions, generation_settings_json, generation_preset_ids_json
            FROM projects
            WHERE id = 'project_v017'
            """
        ).fetchone()
        chat = connection.execute(
            """
            SELECT title, active_head_message_id,
                   generation_settings_json, generation_preset_ids_json,
                   active_vision_profile_id, vision_settings_json
            FROM chats
            WHERE id = 'chat_v017'
            """
        ).fetchone()
        run = connection.execute(
            """
            SELECT chat_id, idempotency_key, settings_json, provenance_json
            FROM runs
            WHERE id = 'run_v017'
            """
        ).fetchone()
        messages = connection.execute(
            """
            SELECT messages.id, message_parts.text
            FROM messages
            JOIN message_parts ON message_parts.message_id = messages.id
            WHERE messages.chat_id = 'chat_v017'
            ORDER BY messages.created_at
            """
        ).fetchall()
        profile = connection.execute(
            "SELECT use_case FROM model_profiles WHERE id = 'profile_v017'"
        ).fetchone()
        workflow = connection.execute(
            "SELECT trusted FROM workflow_revisions WHERE id = 'revision_v017'"
        ).fetchone()

    assert project is not None
    assert project["name"] == "Synthetic project"
    assert project["instructions"] == "Keep answers concise"
    assert json.loads(project["generation_settings_json"]) == {}
    assert json.loads(project["generation_preset_ids_json"]) == {}
    assert chat is not None
    assert chat["title"] == "Synthetic chat"
    assert chat["active_head_message_id"] == "assistant_v017"
    assert json.loads(chat["generation_settings_json"]) == {}
    assert json.loads(chat["generation_preset_ids_json"]) == {}
    assert chat["active_vision_profile_id"] == "__auto__"
    assert json.loads(chat["vision_settings_json"]) == {
        "max_images": 4,
        "max_video_frames": 6,
        "include_prior_visual": True,
    }
    assert run is not None
    assert run["chat_id"] == "chat_v017"
    assert run["idempotency_key"] == "legacy-idempotency-key"
    assert json.loads(run["settings_json"]) == {"max_tokens": 256}
    assert json.loads(run["provenance_json"]) == {
        "input_artifact_ids": ["artifact_v017"],
        "synthetic": True,
    }
    assert [tuple(row) for row in messages] == [
        ("user_v017", "Synthetic prompt"),
        ("assistant_v017", "Synthetic response"),
    ]
    assert profile is not None and profile["use_case"] == "writing"
    assert workflow is not None and workflow["trusted"] == 1


def test_idempotency_migration_downgrade_preserves_cross_chat_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "idempotency-downgrade")
    settings.prepare()
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    fixture = Path(__file__).parent / "fixtures" / "v0.1.7-sanitized.sql"
    with sqlite3.connect(database) as connection:
        connection.executescript(fixture.read_text(encoding="utf-8"))
    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
                INSERT INTO chats (
                    id, project_id, title, archived, routing_mode, confirm_uncertain_media,
                    active_chat_profile_id, active_vision_profile_id,
                    active_image_profile_id, active_video_profile_id,
                    created_at, updated_at, active_head_message_id,
                    generation_settings_json, generation_preset_ids_json,
                    vision_settings_json
                )
                SELECT
                    'chat_v017_other', project_id, 'Other synthetic chat', archived,
                    routing_mode, confirm_uncertain_media, active_chat_profile_id,
                    active_vision_profile_id, active_image_profile_id, active_video_profile_id,
                    created_at, updated_at, 'assistant_v017_other',
                    generation_settings_json, generation_preset_ids_json,
                    vision_settings_json
            FROM chats
            WHERE id = 'chat_v017'
            """
        )
        connection.execute(
            """
            INSERT INTO messages (
                id, chat_id, parent_id, role, status, created_at, updated_at
            ) VALUES (
                'user_v017_other', 'chat_v017_other', NULL, 'user', 'complete',
                '2026-07-25 00:00:00', '2026-07-25 00:00:00'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO messages (
                id, chat_id, parent_id, role, status, created_at, updated_at
            ) VALUES (
                'assistant_v017_other', 'chat_v017_other', 'user_v017_other',
                'assistant', 'complete',
                '2026-07-25 00:00:01', '2026-07-25 00:00:01'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO runs (
                id, idempotency_key, chat_id, user_message_id, assistant_message_id,
                operation, status, standalone_prompt, profile_id, workflow_revision_id,
                settings_json, provenance_json, error, started_at, completed_at,
                duration_ms, created_at, updated_at
            )
            SELECT
                'run_v017_other', idempotency_key, 'chat_v017_other',
                'user_v017_other', 'assistant_v017_other', operation, status,
                standalone_prompt, profile_id, workflow_revision_id, settings_json,
                provenance_json, error, started_at, completed_at, duration_ms,
                created_at, updated_at
            FROM runs
            WHERE id = 'run_v017'
            """
        )
        connection.commit()

    command.downgrade(config, "a4d7c2e91b63")
    with sqlite3.connect(database) as connection:
        keys = connection.execute(
            """
            SELECT id, idempotency_key
            FROM runs
            WHERE id IN ('run_v017', 'run_v017_other')
            ORDER BY id
            """
        ).fetchall()
        assert len(keys) == 2
        assert keys[0][1] != keys[1][1]
        assert all(key for _run_id, key in keys)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM runs
            WHERE id IN ('run_v017', 'run_v017_other')
            """
        ).fetchone() == (2,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_artifact_hash_backfills_existing_revisions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An existing installation keeps its evidence rather than re-proving models.

    The backfill reads stored JSON only, so it needs no live runtime, and it must
    agree with what the application computes for the same revision afterwards.
    """
    from alembic import command

    from local_lm.model_planner import workflow_artifact_contract

    settings = Settings(data_dir=tmp_path / "backfill")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c6e9b2f41d30")

    api_graph = {"3": {"class_type": "KSampler"}}
    input_schema = {"type": "object"}
    dependencies = {"template_id": "example", "model_install_ids": ["local_only"]}
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "insert into workflow_definitions (id, name, description, operation, created_at,"
            " updated_at) values ('wf_1', 'Example', '', 'text_to_image', '2026-01-01',"
            " '2026-01-01')"
        )
        connection.execute(
            "insert into workflow_revisions (id, workflow_id, version, engine, ui_graph_json,"
            " api_graph_json, input_schema_json, dependencies_json, trusted, created_at,"
            " updated_at) values ('rev_1', 'wf_1', 1, 'comfyui', ?, ?, ?, ?, 1, '2026-01-01',"
            " '2026-01-01')",
            (
                json.dumps({"nodes": []}),
                json.dumps(api_graph),
                json.dumps(input_schema),
                json.dumps(dependencies),
            ),
        )
        connection.commit()

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        digest = connection.execute(
            "select artifact_sha256 from workflow_revisions where id = 'rev_1'"
        ).fetchone()[0]

    assert digest == workflow_artifact_contract(
        operation="text_to_image",
        engine="comfyui",
        api_graph=api_graph,
        input_schema=input_schema,
        dependencies=dependencies,
    )


def _parent_revision(settings: Settings) -> str:
    script = ScriptDirectory.from_config(alembic_config(settings))
    head = script.get_revision("head")
    assert head.down_revision, "expected the migration chain to have more than one revision"
    return str(head.down_revision)


def test_a_failed_upgrade_gives_the_data_back_on_the_next_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this guards is not hypothetical: SQLite's driver leaves DDL
    standing when the transaction around it fails, while the revision marker
    rolls back. So the failure is injected after real DDL has been applied."""

    settings = Settings(data_dir=tmp_path / "interrupted-upgrade")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    upgrade_database(settings)
    _run(
        database,
        ("CREATE TABLE canary (id TEXT PRIMARY KEY)", ()),
        ("INSERT INTO canary VALUES ('irreplaceable')", ()),
        ("UPDATE alembic_version SET version_num = ?", (_parent_revision(settings),)),
    )

    def half_apply_then_die(*_args: object, **_kwargs: object) -> None:
        _run(database, ("CREATE TABLE added_by_the_migration (id TEXT PRIMARY KEY)", ()))
        raise RuntimeError("interrupted partway")

    monkeypatch.setattr(command, "upgrade", half_apply_then_die)

    with pytest.raises(DatabaseUpgradeError, match="Restart LM Atelier"):
        upgrade_database(settings)

    # Without the restore the install is finished: the schema holds half a
    # migration the data says was never applied.
    assert _table_exists(database, "added_by_the_migration")

    assert BackupManager(settings).apply_pending_restore()

    assert not _table_exists(database, "added_by_the_migration")
    assert _query(database, "SELECT id FROM canary") == [("irreplaceable",)]


def test_data_from_a_newer_build_is_never_restored_over(tmp_path: Path) -> None:
    """Nothing was applied, so there is nothing to give back, and replacing the
    data would destroy what a newer build wrote."""

    settings = Settings(data_dir=tmp_path / "newer-build")
    settings.prepare()
    _run(
        settings.state_dir / "local-lm.sqlite3",
        ("CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY)", ()),
        ("INSERT INTO alembic_version VALUES ('from_a_newer_build')", ()),
    )

    with pytest.raises(DatabaseVersionError):
        upgrade_database(settings)

    assert not (settings.state_dir / "restore-on-next-start.json").exists()


def test_an_upgrade_with_nothing_to_do_does_not_copy_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "already-current")
    settings.prepare()
    upgrade_database(settings)
    before = {path.name for path in settings.backup_dir.glob("*.sqlite3")}

    upgrade_database(settings)

    assert {path.name for path in settings.backup_dir.glob("*.sqlite3")} == before


def _run(database: Path, *statements: tuple[str, tuple[object, ...]]) -> None:
    """Windows will not let a restore replace a file that is still open."""

    with closing(sqlite3.connect(database)) as connection:
        for sql, parameters in statements:
            connection.execute(sql, parameters)
        connection.commit()


def _query(database: Path, sql: str) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(sql).fetchall()


def _table_exists(database: Path, name: str) -> bool:
    return bool(
        _query(database, f"SELECT name FROM sqlite_master WHERE type='table' AND name = '{name}'")
    )


def test_the_restore_is_armed_before_the_upgrade_not_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that is killed never reaches an exception handler, so the
    request has to already be on disk while the upgrade is in flight."""

    settings = Settings(data_dir=tmp_path / "armed-first")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    upgrade_database(settings)
    _run(
        database,
        ("UPDATE alembic_version SET version_num = ?", (_parent_revision(settings),)),
    )
    marker = settings.state_dir / "restore-on-next-start.json"
    armed_during_upgrade: list[bool] = []

    def observe(*_args: object, **_kwargs: object) -> None:
        armed_during_upgrade.append(marker.is_file())

    monkeypatch.setattr(command, "upgrade", observe)

    upgrade_database(settings)

    assert armed_during_upgrade == [True]
    # And withdrawn once the upgrade survived, so an ordinary start does not
    # roll itself back.
    assert not marker.exists()


def test_a_brand_new_database_is_not_snapshotted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """There is nothing to restore to, and trying produced a traceback on the
    first launch of every new install - describing a loss that cannot happen."""

    settings = Settings(data_dir=tmp_path / "first-launch")
    settings.prepare()

    with caplog.at_level("WARNING"):
        upgrade_database(settings)

    assert not list(settings.backup_dir.glob("*.sqlite3"))
    assert not (settings.state_dir / "restore-on-next-start.json").exists()
    assert "Could not snapshot" not in caplog.text
    # And it really did upgrade.
    assert _recorded_revisions_for(settings)


def _recorded_revisions_for(settings: Settings) -> set[str]:
    with closing(sqlite3.connect(settings.state_dir / "local-lm.sqlite3")) as connection:
        return {row[0] for row in connection.execute("SELECT version_num FROM alembic_version")}
