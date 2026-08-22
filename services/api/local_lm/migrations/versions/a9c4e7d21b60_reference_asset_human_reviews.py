"""record authoritative human reviews of retained Reference asset bytes

Revision ID: a9c4e7d21b60
Revises: f9b7a1c42d60
Create Date: 2026-08-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from local_lm.reference_review_schema import (
    CREATE_REFERENCE_REVIEW_TRIGGER_SQL,
    DROP_REFERENCE_REVIEW_TRIGGER_SQL,
)

revision: str = "a9c4e7d21b60"
down_revision: str | None = "f9b7a1c42d60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def _begin_write_fence() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        connection.exec_driver_sql("UPDATE alembic_version SET version_num = version_num")


def upgrade() -> None:
    _begin_write_fence()
    connection = op.get_bind()
    invalid = connection.execute(
        sa.text(
            """
            SELECT id FROM reference_assets
            WHERE validation_state != 'unchecked'
               OR width IS NOT NULL OR height IS NOT NULL
               OR NOT json_valid(validation_reasons_json)
               OR json_type(validation_reasons_json) != 'array'
               OR json_array_length(validation_reasons_json) != 0
            LIMIT 1
            """
        )
    ).scalar_one_or_none()
    if invalid is not None:
        raise RuntimeError(
            "reference asset review migration found authority without a human review event"
        )
    with op.batch_alter_table("reference_assets") as batch:
        batch.add_column(
            sa.Column("review_version", sa.Integer(), nullable=False, server_default="1")
        )
        batch.create_check_constraint("ck_reference_asset_review_version", "review_version > 0")
    op.create_table(
        "reference_asset_review_events",
        sa.Column("id", sa.String(length=88), nullable=False),
        sa.Column("reference_asset_id", sa.String(length=48), nullable=False),
        sa.Column("artifact_id", sa.String(length=80), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewer_kind", sa.String(length=32), nullable=False),
        sa.Column("expected_state", sa.String(length=30), nullable=False),
        sa.Column("expected_version", sa.Integer(), nullable=False),
        sa.Column("result_version", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=30), nullable=False),
        sa.Column("reasons_json", sa.JSON(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("decision_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("decision_sha256"),
        sa.UniqueConstraint(
            "reference_asset_id", "result_version", name="uq_reference_asset_review_version"
        ),
        sa.CheckConstraint("expected_version > 0", name="ck_reference_review_expected_version"),
        sa.CheckConstraint(
            "result_version = expected_version + 1", name="ck_reference_review_result_version"
        ),
        sa.CheckConstraint("width > 0 AND height > 0", name="ck_reference_review_dimensions"),
        sa.CheckConstraint(
            _lowercase_sha256_check("artifact_sha256"), name="ck_reference_review_artifact_sha256"
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("decision_sha256"), name="ck_reference_review_decision_sha256"
        ),
    )
    op.create_index(
        "ix_reference_asset_review_events_reference_asset_id",
        "reference_asset_review_events",
        ["reference_asset_id"],
    )
    for statement in CREATE_REFERENCE_REVIEW_TRIGGER_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_REFERENCE_REVIEW_TRIGGER_SQL:
        op.execute(statement)
    op.drop_index(
        "ix_reference_asset_review_events_reference_asset_id",
        table_name="reference_asset_review_events",
    )
    op.drop_table("reference_asset_review_events")
    with op.batch_alter_table("reference_assets") as batch:
        batch.drop_constraint("ck_reference_asset_review_version", type_="check")
        batch.drop_column("review_version")
