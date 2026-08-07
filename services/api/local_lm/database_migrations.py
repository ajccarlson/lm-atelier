from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError

from .backups import BackupManager
from .config import Settings

logger = logging.getLogger(__name__)


class DatabaseVersionError(RuntimeError):
    """The database revision is not supported by this application build."""


class DatabaseUpgradeError(RuntimeError):
    """The upgrade failed and the previous data is waiting to be restored."""


def alembic_config(settings: Settings) -> Config:
    package_root = Path(__file__).resolve().parent
    config = Config()
    config.set_main_option("script_location", str(package_root / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.attributes["settings"] = settings
    return config


def _recorded_revisions(database: Path) -> set[str]:
    """Read the revisions the data claims, without opening the ORM."""

    if not database.is_file():
        return set()
    try:
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    except sqlite3.DatabaseError:
        # No version table yet, or unreadable. Either way the upgrade has work
        # to do and deserves the protection below.
        return set()
    return {str(row[0]) for row in rows}


def _upgrade_plan(config: Config, settings: Settings) -> tuple[bool, bool]:
    """Whether an upgrade has work to do, and whether anything could be lost.

    A database with no recorded revision has nothing to restore to. Snapshotting
    one fails verification - there is not yet an `alembic_version` table to
    verify - and the warning that follows is a traceback on the first launch of
    every new install, describing a loss that cannot happen.
    """

    database = settings.state_dir / "local-lm.sqlite3"
    recorded = _recorded_revisions(database)
    try:
        heads = set(ScriptDirectory.from_config(config).get_heads())
    except CommandError:
        return True, bool(recorded)
    return recorded != heads, bool(recorded)


def upgrade_database(settings: Settings) -> None:
    """Bring the data to this build's schema, or give it back unchanged.

    A partly-applied upgrade cannot be undone in place. SQLite's Python driver
    leaves DDL standing when the surrounding transaction fails, while the
    revision marker - ordinary row data - rolls back. The schema then holds half
    of a migration that the data still says was never applied, so every later
    start replays it and fails on what already exists. Guards inside a migration
    cannot close this, because a killed process never reaches them.

    So the upgrade is made recoverable rather than atomic: snapshot first, and on
    failure ask for that snapshot back on the next start. The restore itself
    already runs early enough to matter - `apply_pending_restore` happens before
    the database is opened.
    """

    config = alembic_config(settings)
    manager = BackupManager(settings)
    snapshot: str | None = None
    pending, recoverable = _upgrade_plan(config, settings)
    if pending and recoverable:
        try:
            snapshot = manager.create().name
            # Armed before the work, not after it. An exception handler only
            # runs for an exception; a process killed mid-upgrade - the case
            # that motivates all of this - never reaches one, and would
            # otherwise leave half-applied DDL with nothing asking for it back.
            manager.request_restore(snapshot)
        except Exception:
            # A missing snapshot is worth reporting but not worth refusing to
            # start over: without it the upgrade is exactly as safe as it was
            # before this protection existed.
            snapshot = None
            logger.warning("Could not snapshot the data before upgrading", exc_info=True)
    try:
        command.upgrade(config, "head")
    except CommandError as error:
        # The data is newer than this build. Nothing was applied, so there is
        # nothing to give back and a restore would be the wrong answer.
        if not isinstance(error.__cause__, ResolutionError):
            raise
        if snapshot is not None:
            manager.cancel_restore()
        raise DatabaseVersionError(
            "This LM Atelier data uses a database schema revision that this "
            "build does not recognize. Install the latest LM Atelier version "
            "and keep the existing data folder."
        ) from error
    except Exception as error:
        if snapshot is None:
            raise
        raise DatabaseUpgradeError(
            "The data could not be upgraded to this version of LM Atelier. "
            "Restart LM Atelier to return to your data as it was before the "
            "upgrade started."
        ) from error
    if snapshot is not None:
        manager.cancel_restore()
