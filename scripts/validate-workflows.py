"""Validate GitHub workflow syntax and repository security policy."""

from __future__ import annotations

import os
import re
import subprocess
import sys
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
# A fully valid, fully green environment. The malformed-authority cases below
# start here and break exactly one field, so each one isolates its own input.
GREEN_MERGE_GATE = {
    "EVENT_NAME": "pull_request",
    "DRAFT": "false",
    "ACTION": "synchronize",
    "HEAD_SHA": "0" * 40,
    "PLAN": "success",
    "UBUNTU": "success",
    "WINDOWS": "success",
    "WINDOWS_REQUIRED": "true",
}

GREEN_MERGE_GROUP = {
    "EVENT_NAME": "merge_group",
    "REPOSITORY_PRIVATE": "false",
    "ACTION": "checks_requested",
    "BASE_REF": "refs/heads/develop",
    "BASE_SHA": "1" * 40,
    "HEAD_REF": "refs/heads/gh-readonly-queue/develop/pr-1-" + "0" * 40,
    "HEAD_SHA": "0" * 40,
    "RUN_SHA": "0" * 40,
    "DRAFT": None,
    "BASE_CHANGED": None,
    "PLAN": "success",
    "UBUNTU": "success",
    "WINDOWS": "success",
    "WINDOWS_REQUIRED": "true",
}

MERGE_GATE_MATRIX = (
    ("verified merge group without PR-only fields", GREEN_MERGE_GROUP, 0),
    (
        "merge group with unrequired Windows",
        {**GREEN_MERGE_GROUP, "WINDOWS": "skipped", "WINDOWS_REQUIRED": "false"},
        0,
    ),
    *(
        (f"merge group refuses {name}={value!r}", {**GREEN_MERGE_GROUP, name: value}, 1)
        for name, value in (
            ("EVENT_NAME", None),
            ("EVENT_NAME", "push"),
            ("ACTION", "destroyed"),
            ("REPOSITORY_PRIVATE", None),
            ("REPOSITORY_PRIVATE", "true"),
            ("BASE_REF", "refs/heads/main"),
            ("BASE_REF", None),
            ("BASE_SHA", None),
            ("BASE_SHA", "0" * 40),
            ("HEAD_REF", "refs/heads/develop"),
            ("HEAD_SHA", "0" * 39),
            ("RUN_SHA", "2" * 40),
            ("PLAN", "skipped"),
            ("PLAN", "failure"),
            ("PLAN", "cancelled"),
            ("UBUNTU", "skipped"),
            ("UBUNTU", "failure"),
            ("UBUNTU", "cancelled"),
            ("WINDOWS", "skipped"),
            ("WINDOWS", "failure"),
            ("WINDOWS", "cancelled"),
            ("WINDOWS_REQUIRED", None),
        )
    ),
    (
        # Production's shape, not a synthetic green one: the plan is skipped,
        # Ubuntu runs on always() and refuses, and WINDOWS_REQUIRED is absent
        # because the job that sets it never ran.
        "a title or body edit must not authorize",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "false",
            "DRAFT": "false",
            "PLAN": "skipped",
            "UBUNTU": "failure",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": None,
        },
        1,
    ),
    (
        "a title or body edit must not authorize even if everything went green",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "false",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "an edited event that changed the base reverifies and may authorize",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        0,
    ),
    (
        # On a synchronize the answer is not READ, so only the domain check can
        # refuse it. Without this case that check has no consequence any test
        # can observe, and deleting it would look free.
        "an unreadable base-change answer is refused even when it is not read",
        {
            "ACTION": "synchronize",
            "BASE_CHANGED": "maybe",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "an unreadable base-change answer is not a permissive one",
        {
            "ACTION": "edited",
            "BASE_CHANGED": None,
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    # (label, env, expected exit code); a None value removes the variable
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
    # Authority the gate cannot read is not authority it may assume. Each case
    # below starts from the fully green environment above and breaks exactly
    # one input. `None` removes the variable entirely.
    #
    # Unset `DRAFT` once meant "not a draft" and unset `WINDOWS_REQUIRED` meant
    # "Windows is not required", so a job that failed to bind its environment
    # would have been authorized by the absence of the evidence against it.
    *(
        (label, {**GREEN_MERGE_GATE, **override}, 1)
        for label, override in (
            ("a missing draft state is not a non-draft", {"DRAFT": None}),
            ("an empty draft state is not a non-draft", {"DRAFT": ""}),
            ("a capitalised draft state is not recognised", {"DRAFT": "False"}),
            (
                "a missing Windows policy is not 'not required'",
                {"WINDOWS_REQUIRED": None},
            ),
            ("a non-boolean Windows policy is refused", {"WINDOWS_REQUIRED": "yes"}),
            ("a missing action is refused", {"ACTION": None}),
            ("an unsubscribed action is refused", {"ACTION": "closed"}),
            ("a missing head is refused", {"HEAD_SHA": None}),
            ("a truncated head is refused", {"HEAD_SHA": "0" * 39}),
            ("an uppercase head is refused", {"HEAD_SHA": "A" * 40}),
            ("an unknown plan conclusion is refused", {"PLAN": "neutral"}),
            ("an uppercase Ubuntu conclusion is refused", {"UBUNTU": "SUCCESS"}),
            ("an unknown Windows conclusion is refused", {"WINDOWS": "timed_out"}),
            # The one case where a result's domain is what refuses rather than
            # the success comparison: an unrequired Windows leg is never
            # examined, so without a domain check an unreadable conclusion
            # would sail through unread.
            (
                "an unreadable Windows conclusion is refused even when unrequired",
                {"WINDOWS": "timed_out", "WINDOWS_REQUIRED": "false"},
            ),
        )
    ),
    # Converting back to draft has to withdraw authorization even though every
    # job in this run reports success. Without the trigger the workflow never
    # re-runs and the previous green stays attached; with it, this case is what
    # replaces that green with a refusal.
    (
        "converted back to draft withdraws a fully green head",
        {
            "ACTION": "converted_to_draft",
            "DRAFT": "true",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    # An `edited` event is how a base-branch change arrives. The verdict must
    # come from this run's results rather than from which action delivered it,
    # so the same inputs decide the same way under a different action.
    (
        "an edited base is authorized only by this run's results",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        0,
    ),
    (
        "an edited base does not excuse a missing Ubuntu leg",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "skipped",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
)


MERGE_GATE_SCRIPT = Path("scripts/ci-merge-gate.py")
MERGE_GATE_COMMAND = "python scripts/ci-merge-gate.py"
MERGE_GATE_CONDITION = (
    "always() && (github.event_name == 'pull_request' || "
    "github.event_name == 'merge_group')"
)


def event_expression(field: str, fallback: str) -> str:
    """Select a merge-group field or the existing PR/dispatch binding."""

    return (
        "${{ github.event_name == 'merge_group' && github.event.merge_group."
        + field
        + " || "
        + fallback
        + " }}"
    )


EVENT_HEAD_SHA = event_expression("head_sha", "github.event.pull_request.head.sha")
EVENT_BASE_SHA = event_expression("base_sha", "github.event.pull_request.base.sha")
EVENT_BASE_REF = event_expression("base_ref", "github.base_ref")
EVENT_HEAD_REF = event_expression("head_ref", "github.head_ref")
VERIFICATION_REF = event_expression(
    "head_sha",
    "github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha",
)

# The whole job, declared rather than sampled. Six rounds of review on this file
# established that any check phrased as "the fields I thought of are correct"
# accepts the field nobody thought of, and GitHub has plenty: a job-level
# `continue-on-error` tolerates the decision failing, `defaults.run.shell`,
# `container` and a different `runs-on` all change what executes without
# touching a single pinned step field. So the schema is exhaustive and equality
# is the test: unknown keys are refused because they are unknown.
MERGE_GATE_JOB = {
    "name": "Merge gate",
    "if": MERGE_GATE_CONDITION,
    "needs": ["verification-plan", "compatibility", "windows-compatibility"],
    "runs-on": "ubuntu-24.04",
    "timeout-minutes": 5,
}
# The decision's whole authority, bound expression by expression. A key that
# goes missing becomes an unreadable input rather than a permissive default,
# and one that reads the wrong expression answers about a different thing.
MERGE_GATE_ENV = {
    "EVENT_NAME": "${{ github.event_name }}",
    "REPOSITORY_PRIVATE": "${{ github.event.repository.private }}",
    "BASE_REF": EVENT_BASE_REF,
    "BASE_SHA": EVENT_BASE_SHA,
    "HEAD_REF": EVENT_HEAD_REF,
    "RUN_SHA": "${{ github.sha }}",
    "DRAFT": "${{ github.event.pull_request.draft }}",
    "ACTION": "${{ github.event.action }}",
    "HEAD_SHA": EVENT_HEAD_SHA,
    "PLAN": "${{ needs.verification-plan.result }}",
    "UBUNTU": "${{ needs.compatibility.result }}",
    "WINDOWS": "${{ needs.windows-compatibility.result }}",
    "WINDOWS_REQUIRED": "${{ needs.verification-plan.outputs.windows }}",
    # Which KIND of edited event this is. Without it the decision cannot
    # tell a base change, which must reverify, from a title or body edit,
    # which verified nothing.
    "BASE_CHANGED": (
        "${{ github.event.changes.base.ref.from != ''"
        " || github.event.changes.base.sha.from != '' }}"
    ),
}

# The gate runs a tracked file, so the job has to fetch that file before it can
# decide anything, and it must fetch the copy belonging to the head under
# judgement. Without these two steps the job fails for want of a script rather
# than for want of verification - which fails closed, but can never authorize a
# valid head either, so the gate is useless in both directions.
MERGE_GATE_SETUP = (
    {
        "label": "checkout",
        "missing": f"so {MERGE_GATE_SCRIPT.name} is not present when the job runs",
        "action": "actions/checkout",
        "ref": "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "required_with": {
            # The PR head or generated merge-group head being judged.
            "ref": EVENT_HEAD_SHA,
            # This job reads a verdict. It never needs to write, and a token
            # left on disk is reachable by anything the head runs later.
            "persist-credentials": False,
        },
    },
    {
        "label": "python setup",
        "missing": "so the gate would run on whatever interpreter the runner happens to have",
        "action": "actions/setup-python",
        "ref": "5fda3b95a4ea91299a34e894583c3862153e4b97",
        "required_with": {"python-version": "3.12"},
    },
)

# Each step in full, in order. The `with` and `env` mappings are compared whole,
# so an added input is refused the same way a changed one is.
MERGE_GATE_STEPS = (
    {
        "uses": f"{MERGE_GATE_SETUP[0]['action']}@{MERGE_GATE_SETUP[0]['ref']}",
        "with": dict(MERGE_GATE_SETUP[0]["required_with"]),
    },
    {
        "uses": f"{MERGE_GATE_SETUP[1]['action']}@{MERGE_GATE_SETUP[1]['ref']}",
        "with": dict(MERGE_GATE_SETUP[1]["required_with"]),
    },
    {
        "name": "Require real verification for this exact head",
        "env": dict(MERGE_GATE_ENV),
        "run": MERGE_GATE_COMMAND,
    },
)


def exactly_equal(expected: Any, actual: Any) -> bool:
    """Equality that also requires the same type, all the way down.

    Python treats `False == 0` and `True == 1`, so a YAML
    `persist-credentials: 0` satisfies an expected `False` under ordinary
    mapping equality, and a pinned `timeout-minutes: 5` is satisfied by `True`.
    For values whose whole purpose is to be pinned, that is a bypass rather
    than a convenience, and it hides inside nested `with` and `env` mappings
    where a top-level type check never looks.
    """
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        if len(expected) != len(actual):
            return False
        # Keys need the same treatment as values, and a set comparison cannot
        # give it: {False} == {0} and {True} == {1}, so two mappings with
        # differently typed keys compare equal and even collapse into one
        # entry. Match each key to one of the same type, consuming it, so a
        # single actual entry cannot satisfy two expected ones.
        remaining = list(actual.items())
        for key, value in expected.items():
            found = next(
                (
                    index
                    for index, (other, _) in enumerate(remaining)
                    if type(other) is type(key) and other == key
                ),
                None,
            )
            if found is None:
                return False
            if not exactly_equal(value, remaining.pop(found)[1]):
                return False
        return True
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(exactly_equal(item, other) for item, other in zip(expected, actual))
    return bool(expected == actual)


def validate_merge_gate_schema(path: Path, job: dict[str, Any]) -> list[str]:
    """Compare the whole job against its declaration, key for key.

    Everything here is equality against an exhaustive schema rather than a set
    of individual assertions. The difference matters: an assertion answers the
    question it was given, while equality answers the question nobody thought
    to ask. `continue-on-error` at job level is the example that made this
    necessary - it lets the decision fail and the job succeed, and no amount of
    checking the fields we already pin would have noticed it.
    """
    problems: list[str] = []
    expected_keys = set(MERGE_GATE_JOB) | {"steps"}
    unexpected = sorted(set(job) - expected_keys)
    if unexpected:
        problems.append(
            f"{path}: merge-gate declares {unexpected}, which can change whether the "
            "decision runs or whether its result is the job's answer"
        )
    for key, want in MERGE_GATE_JOB.items():
        got = job.get(key)
        if not exactly_equal(want, got):
            problems.append(
                f"{path}: merge-gate {key} must be exactly {want!r}, found {got!r}"
            )

    steps = job.get("steps")
    if not isinstance(steps, list) or len(steps) != len(MERGE_GATE_STEPS):
        found = len(steps) if isinstance(steps, list) else steps
        return problems + [
            (
                f"{path}: merge-gate must have exactly {len(MERGE_GATE_STEPS)} steps - "
                f"its audited setup then one decision - found {found!r}"
            )
        ]
    for index, (step, want_step) in enumerate(zip(steps, MERGE_GATE_STEPS)):
        if not isinstance(step, dict):
            problems.append(f"{path}: merge-gate step {index} is not a mapping")
            continue
        extra = sorted(set(step) - set(want_step))
        if extra:
            problems.append(
                f"{path}: merge-gate step {index} declares {extra}; nothing may alter how "
                "or whether that step runs"
            )
        for key, want in want_step.items():
            got = step.get(key)
            if isinstance(want, str) and isinstance(got, str):
                got = got.strip()
            if not exactly_equal(want, got):
                problems.append(
                    f"{path}: merge-gate step {index} {key} must be exactly {want!r}, "
                    f"found {got!r}"
                )
    return problems


def validate_merge_gate_steps(path: Path, job: dict[str, Any]) -> list[str]:
    """The gate's steps must fetch the decision before making it.

    The job is bound exactly rather than sampled. Order and duplicates matter -
    a second checkout can replace the script that is about to run, and a
    setup-python after the decision provisions an interpreter nothing uses -
    but so does everything else in the list. An extra step between the setup
    and the decision runs with the gate's authority, and an execution control
    on the decision step can detach its exit code from what the job concludes.
    `continue-on-error: true` is the sharpest of those: the step fails, the job
    succeeds, and the gate reports green having refused.
    """
    problems: list[str] = []
    steps = job.get("steps") or []
    # A malformed `steps:` must produce a diagnostic, not a traceback. Anything
    # that is not a list of mappings crashed this function before it could
    # report, which turns a policy violation into a stack trace and loses the
    # message the operator needs.
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return problems + [
            f"{path}: merge-gate steps must be a list of mappings, found {type(steps).__name__}"
        ]
    if len(steps) != len(MERGE_GATE_SETUP) + 1:
        problems.append(
            f"{path}: merge-gate must have exactly {len(MERGE_GATE_SETUP) + 1} steps - "
            f"its audited setup then one decision - found {len(steps)}"
        )
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        for control in ("if", "continue-on-error", "working-directory", "shell"):
            if control in step:
                problems.append(
                    f"{path}: merge-gate step {index} declares {control!r}, which can "
                    "detach its result from the job's conclusion"
                )
    if str(job.get("if", "")).strip() != MERGE_GATE_CONDITION:
        problems.append(
            f"{path}: merge-gate condition must be exactly {MERGE_GATE_CONDITION!r}, "
            f"found {str(job.get('if', '')).strip()!r}"
        )
    uses = [
        (index, str(step.get("uses", "")))
        for index, step in enumerate(steps)
        if step.get("uses")
    ]
    if len(uses) != len(MERGE_GATE_SETUP):
        problems.append(
            f"{path}: merge-gate must use exactly {len(MERGE_GATE_SETUP)} actions, found "
            f"{len(uses)}; an extra action runs with the gate's authority"
        )
    decision = [index for index, step in enumerate(steps) if step.get("run")]
    if len(decision) == 1:
        # The decision reads its authority from this mapping. A dropped key is
        # an unreadable input, which the predicate refuses - the binding is
        # pinned here so that refusal is never reached by accident.
        declared = steps[decision[0]].get("env") or {}
        if set(declared) != set(MERGE_GATE_ENV):
            missing = sorted(set(MERGE_GATE_ENV) - set(declared))
            extra = sorted(set(declared) - set(MERGE_GATE_ENV))
            problems.append(
                f"{path}: merge-gate decision env must be exactly "
                f"{sorted(MERGE_GATE_ENV)}; missing={missing} unexpected={extra}"
            )
        for name, expression in MERGE_GATE_ENV.items():
            if name in declared and str(declared[name]).strip() != expression:
                problems.append(
                    f"{path}: merge-gate {name} must read {expression!r}, found "
                    f"{str(declared[name]).strip()!r}"
                )

    for position, expected in enumerate(MERGE_GATE_SETUP):
        matches = [
            (index, value)
            for index, value in uses
            if value.startswith(f"{expected['action']}@")
        ]
        if not matches:
            problems.append(
                f"{path}: merge-gate has no {expected['label']} step, {expected['missing']}"
            )
            continue
        if len(matches) > 1:
            problems.append(
                f"{path}: merge-gate uses {expected['action']} {len(matches)} times; a second "
                "one can replace what the first fetched"
            )
        index, value = matches[0]
        pinned = value.split("@", 1)[1].split()[0]
        if not FULL_COMMIT.fullmatch(pinned):
            problems.append(
                f"{path}: merge-gate pins {expected['action']} to {pinned!r}, which is mutable; "
                "a moving ref can change what the gate runs"
            )
        elif pinned != expected["ref"]:
            problems.append(
                f"{path}: merge-gate pins {expected['action']} to {pinned}, not the "
                f"repository's audited {expected['ref']}"
            )
        if index != position:
            problems.append(
                f"{path}: merge-gate runs {expected['action']} at step {index}, expected "
                f"position {position}; the decision must come after its setup"
            )
        declared = steps[index].get("with") or {}
        for key, want in expected["required_with"].items():
            if declared.get(key) != want:
                problems.append(
                    f"{path}: merge-gate {expected['label']} needs {key}={want!r}, "
                    f"found {declared.get(key)!r}"
                )

    decision = [index for index, step in enumerate(steps) if step.get("run")]
    if decision and decision[0] < len(MERGE_GATE_SETUP):
        problems.append(
            f"{path}: merge-gate runs a command at step {decision[0]}, before its "
            "checkout and interpreter are in place"
        )
    return problems


# Every action the gate depends on. `converted_to_draft` and `edited` are here
# because without them a head keeps an authorization earned under conditions
# that no longer hold: a pull request converted back to draft, or repointed at a
# different base branch, would otherwise carry its old success forward.
REQUIRED_PULL_REQUEST_ACTIONS = frozenset(
    {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "edited",
    }
)
PULL_REQUEST_ACTIONS = [
    "opened",
    "synchronize",
    "reopened",
    "ready_for_review",
    "converted_to_draft",
    "edited",
]
EXPECTED_TRIGGER_KEYS = frozenset(
    {"pull_request", "merge_group", "schedule", "workflow_dispatch"}
)
PROTECTED_BRANCHES = ["develop", "main"]
EXPECTED_SCHEDULE = [{"cron": "23 9 * * 2"}]
# Whitespace-normalised, because the expression is written across lines in YAML.
EXPECTED_CONCURRENCY_GROUP = (
    "ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}-${{ "
    "github.event_name == 'pull_request' && github.event.action || github.event_name }}"
)


def validate_pull_request_triggers(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Pin the events the gate runs on, and the identity that keeps its runs apart.

    Exact equality rather than "contains what we need". A subset check accepts
    additions, and an addition changes authority: subscribing to `closed`, for
    instance, would run the gate on a merged pull request. The same applies to
    the branch list, the concurrency expression and the cancellation policy -
    each is a decision, so each is pinned rather than sampled.
    """
    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    # `on` is YAML's boolean true once parsed, which is a well-known trap here.
    triggers = workflow.get("on") or workflow.get(True) or {}
    pull_request = triggers.get("pull_request") if isinstance(triggers, dict) else None
    if not isinstance(pull_request, dict):
        return [f"{path}: ci.yml has no pull_request trigger"]

    # The exact list, in order, without duplicates. A set comparison accepts a
    # repeated entry, and a repeated entry is a signal that the file was edited
    # by something that did not understand it.
    declared = list(pull_request.get("types") or [])
    if declared != PULL_REQUEST_ACTIONS:
        problems.append(
            f"{path}: pull_request types must be exactly {PULL_REQUEST_ACTIONS}, found "
            f"{declared}"
        )
    # No other trigger keys. A `push` trigger would run this workflow, and its
    # gate, on events with none of the pull request context the gate reads.
    if set(triggers) != EXPECTED_TRIGGER_KEYS:
        missing = sorted(EXPECTED_TRIGGER_KEYS - set(triggers))
        extra = sorted(set(triggers) - EXPECTED_TRIGGER_KEYS)
        problems.append(
            f"{path}: triggers must be exactly {sorted(EXPECTED_TRIGGER_KEYS)}; "
            f"missing={missing} unexpected={extra}"
        )
    if not exactly_equal({"types": ["checks_requested"]}, triggers.get("merge_group")):
        problems.append(f"{path}: merge_group must request exactly checks_requested")
    # workflow_dispatch takes no inputs. A dispatch contract is a way to vary a
    # run's behaviour from outside the file.
    if triggers.get("workflow_dispatch") is not None:
        problems.append(
            f"{path}: workflow_dispatch must declare no inputs, found "
            f"{triggers.get('workflow_dispatch')!r}"
        )
    if list(pull_request.get("branches") or []) != PROTECTED_BRANCHES:
        problems.append(
            f"{path}: pull_request must target exactly {PROTECTED_BRANCHES}, found "
            f"{pull_request.get('branches')!r}"
        )

    if triggers.get("schedule") != EXPECTED_SCHEDULE:
        problems.append(
            f"{path}: ci.yml must keep its schedule {EXPECTED_SCHEDULE}, found "
            f"{triggers.get('schedule')!r}"
        )
    if "workflow_dispatch" not in triggers:
        problems.append(f"{path}: ci.yml must keep its workflow_dispatch trigger")

    # The event action belongs in the concurrency identity. With one group per
    # pull request, a draft `synchronize` and the `ready_for_review` that
    # follows it cancel each other, and the survivor is whichever GitHub
    # started last - which is how a non-draft pull request came to report every
    # required check as skipped while its last real run tested an earlier
    # commit. Comparing the whole expression also catches a truncated or
    # reordered identity, which a substring check cannot.
    concurrency = workflow.get("concurrency") or {}
    group = " ".join(str(concurrency.get("group", "")).split())
    if group != EXPECTED_CONCURRENCY_GROUP:
        problems.append(
            f"{path}: the concurrency group must be exactly {EXPECTED_CONCURRENCY_GROUP!r}, "
            f"found {group!r}"
        )
    if concurrency.get("cancel-in-progress") is not True:
        problems.append(
            f"{path}: concurrency must set cancel-in-progress: true, found "
            f"{concurrency.get('cancel-in-progress')!r}"
        )
    return problems


def validate_merge_gate(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Prove the merge gate refuses, by running it rather than reading it."""

    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    job = (workflow.get("jobs") or {}).get("merge-gate")
    if not isinstance(job, dict):
        return [f"{path}: ci.yml has no merge-gate job to enforce required checks"]

    # One exhaustive comparison replaces the individual condition, needs and
    # step assertions that used to live here. Those answered only the questions
    # they were given; this refuses anything the schema does not name, which is
    # where every one of the accepted drifts came from.
    problems.extend(validate_merge_gate_schema(path, job))
    problems.extend(validate_merge_gate_steps(path, job))

    declared_steps = job.get("steps") or []
    if not isinstance(declared_steps, list) or not all(
        isinstance(step, dict) for step in declared_steps
    ):
        # Already reported by the schema and the step checks; stop here rather
        # than crashing on a shape neither of them accepted.
        return problems
    scripts = [step.get("run") for step in declared_steps if step.get("run")]
    if len(scripts) != 1:
        return problems + [
            f"{path}: expected exactly one run step in merge-gate, found {len(scripts)}"
        ]
    # The step must invoke the shipped predicate, because that is the thing the
    # cases below actually execute. A step that inlined its own logic would be
    # tested by nothing.
    if scripts[0].strip() != MERGE_GATE_COMMAND:
        return problems + [
            (
                f"{path}: merge-gate must run exactly {MERGE_GATE_COMMAND!r}, so the "
                f"workflow and this validator exercise the same decision; "
                f"found {scripts[0].strip()!r}"
            )
        ]
    if not MERGE_GATE_SCRIPT.is_file():
        return problems + [f"{path}: {MERGE_GATE_SCRIPT} is missing"]

    for label, environment, expected in MERGE_GATE_MATRIX:
        full = {
            **os.environ,
            "EVENT_NAME": "pull_request",
            "ACTION": "synchronize",
            "HEAD_SHA": "0" * 40,
            "BASE_CHANGED": "false",
            **{name: value for name, value in environment.items() if value is not None},
        }
        # A case can assert what happens when a variable is absent entirely,
        # which is different from asserting what happens when it is empty.
        for name, value in environment.items():
            if value is None:
                full.pop(name, None)
        try:
            finished = subprocess.run(
                [sys.executable, str(MERGE_GATE_SCRIPT)],
                env=full,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # A failure to execute is a problem, never a pass. The previous
            # version printed a note and returned the existing list, so a
            # machine that could not run the gate reported success - the same
            # fail-open this gate exists to prevent.
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


MERGE_GROUP_CONDITION = (
    "( github.event_name == 'merge_group' && "
    "github.event.action == 'checks_requested' && "
    "github.event.repository.private == false && "
    "github.event.merge_group.base_ref == 'refs/heads/develop' )"
)
VERIFICATION_PLAN_CONDITION = (
    "github.event_name == 'workflow_dispatch' || " + MERGE_GROUP_CONDITION + " || "
    "( github.event_name == 'pull_request' && "
    "github.event.repository.private == false && "
    "github.event.pull_request.draft == false && "
    "( github.event.action != 'edited' || "
    "github.event.changes.base.ref.from != '' || "
    "github.event.changes.base.sha.from != '' ) )"
)
COMPATIBILITY_CONDITION = (
    "always() && ( github.event_name == 'workflow_dispatch' || "
    + MERGE_GROUP_CONDITION
    + " || ( github.event_name == 'pull_request' && "
    "github.event.repository.private == false && "
    "github.event.pull_request.draft == false ) )"
)
PLAN_ENV = {
    "BASE_REF": EVENT_BASE_REF,
    "BASE_SHA": EVENT_BASE_SHA,
    "EVENT_NAME": "${{ github.event_name }}",
    "HEAD_REF": EVENT_HEAD_REF,
    "HEAD_SHA": EVENT_HEAD_SHA,
}
PLAN_COMMAND = (
    "python3 scripts/ci-plan.py \\\n"
    '  --event "$EVENT_NAME" \\\n'
    '  --base-ref "$BASE_REF" \\\n'
    '  --head-ref "$HEAD_REF" \\\n'
    '  --base-sha "$BASE_SHA" \\\n'
    '  --head-sha "$HEAD_SHA" \\\n'
    '  --github-output "$GITHUB_OUTPUT"'
)


def validate_verification_bindings(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Keep queue planning and verification bound to one integration commit."""

    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    jobs = workflow.get("jobs") or {}
    expected_checkout = {
        "uses": f"{MERGE_GATE_SETUP[0]['action']}@{MERGE_GATE_SETUP[0]['ref']}",
        "with": {
            "ref": VERIFICATION_REF,
            "fetch-depth": 0,
            "persist-credentials": False,
        },
    }
    for name, condition, needs in (
        ("verification-plan", VERIFICATION_PLAN_CONDITION, None),
        ("compatibility", COMPATIBILITY_CONDITION, "verification-plan"),
        (
            "windows-compatibility",
            "needs.verification-plan.outputs.windows == 'true'",
            "verification-plan",
        ),
    ):
        job = jobs.get(name)
        if not isinstance(job, dict):
            problems.append(f"{path}: missing {name} verification job")
            continue
        if " ".join(str(job.get("if", "")).split()) != condition:
            problems.append(f"{path}: {name} must keep its event condition")
        if not exactly_equal(needs, job.get("needs")):
            problems.append(f"{path}: {name} must depend on its verification plan")
        steps = job.get("steps")
        if not isinstance(steps, list) or not all(
            isinstance(step, dict) for step in steps
        ):
            problems.append(f"{path}: {name} steps must be a list of mappings")
            continue
        checkouts = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        if not exactly_equal([expected_checkout], checkouts):
            problems.append(
                f"{path}: {name} must check out the integration SHA exactly once"
            )
        if "continue-on-error" in job or any(
            "continue-on-error" in step for step in steps
        ):
            problems.append(f"{path}: {name} must report verification failures")
        if name == "verification-plan":
            expected_outputs = {
                output: "${{ steps.plan.outputs." + output + " }}"
                for output in ("mode", "dependency_audit", "windows")
            }
            expected_plan = {"id": "plan", "env": PLAN_ENV, "run": PLAN_COMMAND}
            normalized = [
                {**step, "run": step["run"].strip()} if "run" in step else step
                for step in steps
            ]
            if not exactly_equal([expected_checkout, expected_plan], normalized):
                problems.append(
                    f"{path}: verification-plan must run the bound planner exactly"
                )
            if not exactly_equal(expected_outputs, job.get("outputs")):
                problems.append(
                    f"{path}: verification-plan must publish all planner outputs"
                )
        elif name == "compatibility":
            required_plan = {
                "name": "Require a valid verification plan",
                "if": "needs.verification-plan.result != 'success'",
                "run": "exit 1",
            }
            if not steps or not exactly_equal(required_plan, steps[0]):
                problems.append(
                    f"{path}: compatibility must refuse an unsuccessful plan"
                )
            docs = [
                step
                for step in steps
                if step.get("name") == "Validate documentation-only change"
            ]
            if len(docs) != 1 or not exactly_equal(
                {"BASE_SHA": EVENT_BASE_SHA, "HEAD_SHA": EVENT_HEAD_SHA},
                docs[0].get("env"),
            ):
                problems.append(
                    f"{path}: documentation verification must use the event SHAs"
                )
    return problems


def validate_workflow_document(
    path: Path, content: str, workflow: dict[str, Any]
) -> list[str]:
    """Every policy this repository applies to one workflow file.

    Extracted from `main` so that a regression can enter where production
    enters. Calling the individual validators from a test proves each rule in
    isolation and proves nothing about whether the rule is still wired in - a
    check that stops being called is indistinguishable from one that passes,
    and only a caller-level entry point can tell the difference.
    """
    errors: list[str] = []
    errors.extend(validate_permissions(path, workflow))
    errors.extend(validate_action_pins(path, content))
    errors.extend(validate_checkout_credentials(path, workflow))
    errors.extend(validate_environment_contexts(path, workflow))
    errors.extend(validate_untrusted_triggers(path, content))
    errors.extend(validate_pull_request_triggers(path, workflow))
    errors.extend(validate_merge_gate(path, workflow))
    errors.extend(validate_verification_bindings(path, workflow))
    return errors


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
        errors.extend(validate_workflow_document(path, content, workflow))
        workflow_actions.update(external_actions(content))
        print(path)

    errors.extend(validate_action_allowlist(workflow_actions))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
