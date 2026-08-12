"""Validate GitHub workflow syntax and repository security policy."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_ROOT = Path(".github/workflows")
PUBLIC_CONFIGURATION = Path("scripts/configure-public-repository.ps1")
ACTION_USE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<target>\S+)"
    r"(?:\s+#\s*(?P<version>\S+))?\s*$"
)
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
VERSION_COMMENT = re.compile(r"v[0-9]+(?:\.[0-9]+){1,2}(?:-[0-9A-Za-z.-]+)?")
PULL_REQUEST_TARGET = re.compile(r"^\s*pull_request_target\s*:", re.MULTILINE)
ALLOWED_ACTIONS_BLOCK = re.compile(
    r"^\$AllowedActionPatterns\s*=\s*@\(\s*$"
    r"(?P<body>.*?)"
    r"^\)\s*$",
    re.MULTILINE | re.DOTALL,
)
QUOTED_ACTION = re.compile(
    r'^\s*"(?P<target>[^"]+@[0-9a-f]{40})",?\s*$',
    re.MULTILINE,
)


def iter_permissions(
    value: Any,
    location: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], dict[str, Any] | str]]:
    """Return every workflow- or job-level permissions declaration."""

    declarations: list[tuple[tuple[str, ...], dict[str, Any] | str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "permissions" and isinstance(child, (dict, str)):
                declarations.append((location + (str(key),), child))
            declarations.extend(iter_permissions(child, location + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            declarations.extend(iter_permissions(child, location + (str(index),)))
    return declarations


def validate_permissions(path: Path, workflow: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    top_level = workflow.get("permissions")
    if top_level != {"contents": "read"}:
        errors.append(f"{path}: top-level permissions must be exactly contents: read")

    allowed_writes = {
        (
            "jobs",
            "release-candidate",
            "permissions",
        ): {
            "id-token",
            "attestations",
            "artifact-metadata",
        },
        (
            "jobs",
            "draft-release",
            "permissions",
        ): {
            "contents",
        },
    }
    for location, declaration in iter_permissions(workflow):
        if declaration == "read-all":
            continue
        if isinstance(declaration, str):
            errors.append(
                f"{path}: unsupported permissions declaration {declaration!r}"
            )
            continue
        for scope, access in declaration.items():
            reviewed_write = (
                path.name == "release.yml"
                and scope in allowed_writes.get(location, set())
                and access == "write"
            )
            if reviewed_write:
                continue
            if access not in {"read", "none"}:
                errors.append(
                    f"{path}: {scope} permission must not grant {access!r} access"
                )
    return errors


def validate_action_pins(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        match = ACTION_USE.match(line)
        if not match:
            continue
        target = match.group("target")
        if target.startswith("docker://"):
            errors.append(
                f"{path}:{line_number}: docker actions are prohibited; "
                "use a reviewed SHA-pinned action"
            )
            continue
        if target.startswith("./"):
            continue
        action, separator, reference = target.rpartition("@")
        if not separator or not action or not FULL_COMMIT.fullmatch(reference):
            errors.append(
                f"{path}:{line_number}: external action must use a full commit SHA"
            )
        version = match.group("version")
        if version is None or not VERSION_COMMENT.fullmatch(version):
            errors.append(
                f"{path}:{line_number}: pinned action needs an auditable version comment"
            )
    return errors


def external_actions(content: str) -> set[str]:
    """Return every external action reference in one workflow."""

    actions: set[str] = set()
    for line in content.splitlines():
        match = ACTION_USE.match(line)
        if not match:
            continue
        target = match.group("target")
        if not target.startswith(("./", "docker://")):
            actions.add(target)
    return actions


def validate_action_allowlist(workflow_actions: set[str]) -> list[str]:
    """Keep the applied GitHub allowlist equal to the reviewed workflow pins."""

    content = PUBLIC_CONFIGURATION.read_text(encoding="utf-8")
    block = ALLOWED_ACTIONS_BLOCK.search(content)
    if block is None:
        return [f"{PUBLIC_CONFIGURATION}: missing $AllowedActionPatterns declaration"]
    configured = {
        match.group("target") for match in QUOTED_ACTION.finditer(block.group("body"))
    }
    if configured == workflow_actions:
        return []
    missing = sorted(workflow_actions - configured)
    unused = sorted(configured - workflow_actions)
    details: list[str] = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unused:
        details.append(f"unused {', '.join(unused)}")
    return [
        (
            f"{PUBLIC_CONFIGURATION}: selected Actions allowlist is out of sync "
            f"with workflows ({'; '.join(details)})"
        )
    ]


def validate_checkout_credentials(path: Path, workflow: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return [f"{path}: workflow must define jobs"]
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("uses")
            if not isinstance(action, str) or not action.startswith(
                "actions/checkout@"
            ):
                continue
            inputs = step.get("with")
            if (
                not isinstance(inputs, dict)
                or inputs.get("persist-credentials") is not False
            ):
                errors.append(
                    f"{path}: checkout in job {job_name} must disable persisted credentials"
                )
    return errors


def validate_environment_contexts(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Reject contexts GitHub does not expose to workflow- or job-level env."""

    errors: list[str] = []
    declarations: list[tuple[str, Any]] = [("workflow", workflow.get("env"))]
    jobs = workflow.get("jobs")
    if isinstance(jobs, dict):
        declarations.extend(
            (f"job {job_name}", job.get("env"))
            for job_name, job in jobs.items()
            if isinstance(job, dict)
        )
    for location, declaration in declarations:
        if not isinstance(declaration, dict):
            continue
        for name, value in declaration.items():
            if isinstance(value, str) and "${{ runner." in value:
                errors.append(
                    f"{path}: {location} env {name} cannot use the runner context"
                )
    return errors


def validate_untrusted_triggers(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    if PULL_REQUEST_TARGET.search(content):
        errors.append(
            f"{path}: pull_request_target is prohibited for untrusted contribution CI"
        )
    if path.name == "ci.yml" and "run-ci" in content:
        errors.append(
            f"{path}: normal pull-request CI must not require the run-ci label"
        )
    return errors


# The merge gate is the only required context that cannot be satisfied by a job
# not running, so it is the only thing standing between a draft-skipped pull
# request and a merge with no verification. Checking that it *exists* would be
# checking the wrong thing - the first version of it existed and still exited
# zero on a draft, which handed the draft head a green required context. So this
# runs the gate's own script against the event/result matrix and asserts the
# conclusions.
MERGE_GATE_MATRIX = (
    # (label, env, expected exit code)
    (
        "draft must fail closed",
        {
            "DRAFT": "true",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "draft fails even when nothing else ran",
        {
            "DRAFT": "true",
            "PLAN": "skipped",
            "UBUNTU": "skipped",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "skipped plan is not a passing plan",
        {
            "DRAFT": "false",
            "PLAN": "skipped",
            "UBUNTU": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "skipped Ubuntu is not a passing Ubuntu",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "skipped",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "cancelled Ubuntu is not a passing Ubuntu",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "cancelled",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "failed Ubuntu fails the gate",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "failure",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "Windows may be absent when the plan does not require it",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        0,
    ),
    (
        "a plan-required Windows must not be skipped",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "a plan-required Windows must not have failed",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "failure",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "everything the plan required succeeded",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        0,
    ),
)


def usable_bash() -> str | None:
    """A bash that runs a script and reports its exit code faithfully.

    On a Windows runner `bash` is WSL's, which is present on PATH and fails
    everything when no distribution is installed. Git's bash sits beside git
    itself. Neither can be trusted by name, so each candidate has to prove it
    executes a script and returns its status.
    """
    candidates: list[str] = []
    git = shutil.which("git")
    if git:
        # Git ships bash at <root>/bin/bash.exe, with git at <root>/cmd/git.exe.
        candidates.append(str(Path(git).resolve().parents[1] / "bin" / "bash.exe"))
    found = shutil.which("bash")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "exit 7"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 7:
            return candidate
    return None


def validate_merge_gate(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Prove the merge gate refuses, by running it rather than reading it."""

    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    job = (workflow.get("jobs") or {}).get("merge-gate")
    if not isinstance(job, dict):
        return [f"{path}: ci.yml has no merge-gate job to enforce required checks"]

    condition = str(job.get("if", ""))
    if "always()" not in condition:
        problems.append(
            f"{path}: merge-gate must use always() or it skips exactly when it is needed"
        )
    required_needs = {"verification-plan", "compatibility", "windows-compatibility"}
    declared = set(job.get("needs") or [])
    missing = required_needs - declared
    if missing:
        problems.append(
            f"{path}: merge-gate does not depend on {sorted(missing)}, so it can "
            "conclude before those jobs decide"
        )

    scripts = [step.get("run") for step in job.get("steps") or [] if step.get("run")]
    if len(scripts) != 1:
        return problems + [
            f"{path}: expected exactly one run step in merge-gate, found {len(scripts)}"
        ]
    script = scripts[0]

    shell = usable_bash()
    if shell is None:
        # Reported rather than passed over: the Ubuntu leg always has a real
        # bash and enforces this, so a developer machine without one should say
        # so out loud instead of appearing to have checked.
        print(f"{path}: no usable bash; merge-gate behaviour NOT evaluated here")
        return problems

    for label, environment, expected in MERGE_GATE_MATRIX:
        full = {
            **os.environ,
            "ACTION": "synchronize",
            "HEAD_SHA": "0" * 40,
            **environment,
        }
        try:
            finished = subprocess.run(
                [shell, "-c", script],
                env=full,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            problems.append(f"{path}: merge-gate script could not be executed: {error}")
            break
        actual = 0 if finished.returncode == 0 else 1
        if actual != expected:
            problems.append(
                f"{path}: merge-gate case '{label}' returned {finished.returncode}, "
                f"expected {'success' if expected == 0 else 'failure'}. "
                f"env={environment} output={finished.stdout.strip()[-160:]}"
            )
    return problems


def main() -> None:
    paths = sorted(WORKFLOW_ROOT.glob("*.yml"))
    paths += sorted(WORKFLOW_ROOT.glob("*.yaml"))
    if not paths:
        raise SystemExit("No GitHub workflow files found")

    errors: list[str] = []
    workflow_actions: set[str] = set()
    for path in paths:
        content = path.read_text(encoding="utf-8")
        try:
            workflow = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            errors.append(f"{path}: invalid YAML: {exc}")
            continue
        if not isinstance(workflow, dict):
            errors.append(f"{path}: workflow root must be a mapping")
            continue
        errors.extend(validate_permissions(path, workflow))
        errors.extend(validate_action_pins(path, content))
        errors.extend(validate_checkout_credentials(path, workflow))
        errors.extend(validate_environment_contexts(path, workflow))
        errors.extend(validate_untrusted_triggers(path, content))
        errors.extend(validate_merge_gate(path, workflow))
        workflow_actions.update(external_actions(content))
        print(path)

    errors.extend(validate_action_allowlist(workflow_actions))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
