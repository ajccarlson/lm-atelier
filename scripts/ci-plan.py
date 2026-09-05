"""Select the fail-closed verification plan for a GitHub Actions run."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}")
CONTRACT_DOCUMENTS = {
    ".github/release_template.md",
    "docs/troubleshooting.md",
    "readme.md",
}
DEPENDENCY_FILES = {
    "apps/web/package.json",
    "package-lock.json",
    "package.json",
    "services/api/pyproject.toml",
    "services/api/uv.lock",
}
WINDOWS_PATHS = {
    ".github/workflows/ci.yml",
    "packaging/lmatelier.spec",
    "scripts/build-windows-installer.ps1",
    "scripts/ci-plan.py",
    "scripts/smoke-windows-installer.ps1",
    "scripts/verify.ps1",
}
WINDOWS_PATH_PREFIXES = (
    "scripts/",
    "packaging/windows/",
    "services/api/",
)


def normalized_path(value: str) -> str:
    """Return a repository path in the normalized form used by the policy."""

    return value.strip().replace("\\", "/").removeprefix("./").casefold()


def is_lightweight_documentation(path: str) -> bool:
    """Return whether a path is documentation with no executable contract."""

    normalized = normalized_path(path)
    return normalized.endswith(".md") and normalized not in CONTRACT_DOCUMENTS


def requires_dependency_audit(paths: Iterable[str]) -> bool:
    """Return whether changed dependency inputs need an online audit."""

    return any(normalized_path(path) in DEPENDENCY_FILES for path in paths)


def requires_windows_verification(paths: Iterable[str]) -> bool:
    """Return whether changed paths exercise Windows-specific behavior."""

    normalized = (normalized_path(path) for path in paths)
    return any(
        path in WINDOWS_PATHS
        or any(path.startswith(prefix) for prefix in WINDOWS_PATH_PREFIXES)
        for path in normalized
    )


def classify_develop_changes(paths: Iterable[str]) -> tuple[str, bool]:
    """Classify a develop PR and whether it changes dependency inputs."""

    changed = tuple(path for path in paths if normalized_path(path))
    if changed and all(is_lightweight_documentation(path) for path in changed):
        return "documentation", False
    return "full", requires_dependency_audit(changed)


def git(*arguments: str) -> str:
    """Run a read-only git query and return one stripped line."""

    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def changed_paths(base_sha: str, head_sha: str) -> tuple[str, ...]:
    """Return every path in a pull request without GitHub's path-filter cap."""

    output = git("diff", "--name-only", "--diff-filter=ACDMRTUXB", base_sha, head_sha)
    return tuple(line for line in output.splitlines() if line)


def require_sha(label: str, value: str) -> str:
    """Reject absent or malformed event SHAs before passing them to git."""

    normalized = value.casefold()
    if SHA.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a full commit SHA")
    git("rev-parse", "--verify", f"{normalized}^{{commit}}")
    return normalized


def validate_develop_promotion(
    *,
    base_ref: str,
    head_ref: str,
    base_sha: str,
    head_sha: str,
) -> None:
    """Require an exact protected develop-to-main promotion graph."""

    if base_ref != "main" or head_ref != "develop":
        raise ValueError("main accepts promotions only from the develop branch")
    base = require_sha("base SHA", base_sha)
    head = require_sha("head SHA", head_sha)
    if git("rev-parse", "origin/main") != base:
        raise ValueError("promotion base does not match the current origin/main")
    if git("rev-parse", "origin/develop") != head:
        raise ValueError("promotion head does not match the current origin/develop")
    common = git("merge-base", base, head)
    base_tree = git("rev-parse", f"{base}^{{tree}}")
    common_tree = git("rev-parse", f"{common}^{{tree}}")
    if base_tree != common_tree:
        raise ValueError(
            "main contains source changes not present in the develop lineage"
        )


def validate_merge_group(
    *, base_ref: str, head_ref: str, base_sha: str, head_sha: str
) -> tuple[str, str]:
    """Bind a develop queue plan to the checked-out integration commit."""

    if base_ref != "refs/heads/develop":
        raise ValueError("merge groups must target refs/heads/develop")
    if not head_ref.startswith("refs/heads/gh-readonly-queue/develop/"):
        raise ValueError("merge group head must use the develop queue ref")
    base = require_sha("base SHA", base_sha)
    head = require_sha("head SHA", head_sha)
    if git("rev-parse", "HEAD") != head:
        raise ValueError("checkout does not match the merge group head")
    if base == head or git("merge-base", base, head) != base:
        raise ValueError("merge group head must integrate its event base")
    return base, head


def write_outputs(
    path: Path,
    *,
    mode: str,
    dependency_audit: bool,
    windows: bool,
) -> None:
    """Append the selected plan to GitHub's step-output file."""

    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"mode={mode}\n")
        output.write(f"dependency_audit={str(dependency_audit).lower()}\n")
        output.write(f"windows={str(windows).lower()}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--head-ref", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--github-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.event == "workflow_dispatch":
        mode, dependency_audit, windows = "full", True, True
    elif arguments.event == "merge_group":
        base, head = validate_merge_group(
            base_ref=arguments.base_ref,
            head_ref=arguments.head_ref,
            base_sha=arguments.base_sha,
            head_sha=arguments.head_sha,
        )
        paths = changed_paths(base, head)
        mode, dependency_audit = classify_develop_changes(paths)
        windows = mode == "full" and requires_windows_verification(paths)
    elif arguments.event == "pull_request" and arguments.base_ref == "main":
        validate_develop_promotion(
            base_ref=arguments.base_ref,
            head_ref=arguments.head_ref,
            base_sha=arguments.base_sha,
            head_sha=arguments.head_sha,
        )
        mode, dependency_audit, windows = "promotion", False, False
    elif arguments.event == "pull_request" and arguments.base_ref == "develop":
        base = require_sha("base SHA", arguments.base_sha)
        head = require_sha("head SHA", arguments.head_sha)
        paths = changed_paths(base, head)
        mode, dependency_audit = classify_develop_changes(paths)
        windows = mode == "full" and requires_windows_verification(paths)
    else:
        raise SystemExit("Unsupported CI event or pull-request target")
    write_outputs(
        arguments.github_output,
        mode=mode,
        dependency_audit=dependency_audit,
        windows=windows,
    )
    print(
        f"Verification plan: {mode}; dependency audit: {dependency_audit}; "
        f"Windows: {windows}"
    )


if __name__ == "__main__":
    main()
