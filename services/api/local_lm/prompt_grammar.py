"""How an adapter expects to be prompted, reduced to something safe to act on.

An adapter can be trained to expect a particular prompt shape, and one prompted
the wrong way does not fail: it produces confident output in the wrong form.
Recording that shape is what lets a prompt be written correctly.

The record arrives as a document distributed with the adapter, and its output is
placed in the instruction of a model that rewrites what the user wrote. That
makes every character of it an instruction channel, so this module is written as
a refusal engine rather than a parser. A template is reduced to a trigger, slot
placeholders and a tiny punctuation alphabet; identifiers must match a
conservative ASCII pattern; and prose survives only when a digest of that exact
text was approved on this machine.

Nothing here accepts the document's own claim to have been verified. A published
value is a claim about the world, and one such claim has already been observed
to be false: a vendor listed a value the model does not implement, and prompting
it degraded to the stem rather than failing. Verification is overlaid from local
evidence and is never read out of the file.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA_VERSION = 1

MAX_SLOTS = 8
MAX_VALUES_PER_SLOT = 64
MAX_EXAMPLES = 8
MAX_PROSE = 2000
MAX_TEMPLATE = 300
MAX_INSTRUCTION_CHARS = 8000

# Deliberately narrow, and ASCII by construction. Because this is a fullmatch
# over explicit ranges, every non-ASCII character is refused without needing to
# enumerate them - which is what keeps bidi overrides, zero-width joiners and
# lookalike glyphs out. A leading "@" is permitted because published triggers
# use it; nothing else may lead.
_IDENTIFIER = re.compile(r"@?[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
# A template is punctuation and placeholders, not language. This is the entire
# alphabet allowed between them.
_TEMPLATE_SEPARATOR = re.compile(r"[ ,;:()\[\]/|-]+")
_PLACEHOLDER = re.compile(r"<([A-Za-z0-9_]{1,32})>")

_GRAMMAR_KEYS = frozenset(
    {"schema_version", "trigger", "slots", "template", "description_guidance", "examples"}
)
_SLOT_KEYS = frozenset({"name", "required", "values"})


class PromptGrammarError(ValueError):
    """A published grammar could not be reduced to something safe to act on."""


def prose_digest(text: str) -> str:
    """The identity under which one exact piece of prose may be approved."""

    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SlotValue:
    value: str
    verified: bool = False


@dataclass(frozen=True)
class GrammarSlot:
    name: str
    required: bool
    values: tuple[SlotValue, ...]

    def published(self, candidate: str) -> bool:
        return any(item.value == candidate for item in self.values)

    def verified(self, candidate: str) -> bool:
        return any(item.value == candidate and item.verified for item in self.values)

    def verified_values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.values if item.verified)


@dataclass(frozen=True)
class PromptGrammar:
    schema_version: int
    trigger: str
    slots: tuple[GrammarSlot, ...]
    template: str
    description_guidance: str | None
    examples: tuple[str, ...]

    def slot(self, name: str) -> GrammarSlot | None:
        return next((item for item in self.slots if item.name == name), None)


def _strict_bool(value: object, field: str) -> bool:
    """A structural flag must be a real boolean.

    Coercing with `bool()` would let the string "false" mean true, which is the
    kind of difference a hand-written document is very likely to contain.
    """

    if type(value) is not bool:
        raise PromptGrammarError(f"{field} must be true or false")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise PromptGrammarError(f"{field} must be text")
    cleaned = value.strip()
    if not _IDENTIFIER.fullmatch(cleaned):
        raise PromptGrammarError(
            f"{field} must be a short plain identifier; {cleaned[:32]!r} is not"
        )
    return cleaned


def _prose(value: object, field: str, approved: frozenset[str]) -> str | None:
    """Return this text only if a digest of exactly it was approved here.

    Approval is content-bound rather than a flag, because a flag approves a
    field while the danger is in the characters. Re-editing the document
    silently revokes its own approval, which is the intended behaviour.
    """

    if not isinstance(value, str):
        raise PromptGrammarError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned:
        raise PromptGrammarError(f"{field} must not be empty")
    if len(cleaned) > MAX_PROSE:
        raise PromptGrammarError(f"{field} is longer than {MAX_PROSE} characters")
    return cleaned if prose_digest(cleaned) in approved else None


def _parse_template(template: object, trigger: str, slots: tuple[GrammarSlot, ...]) -> str:
    """Reduce a template to placeholders and punctuation, or refuse it.

    A template that still contains words is prose, and prose in this position is
    an instruction to the rewriting model. Refusing here is what stops the
    document from writing part of its own system prompt.
    """

    if not isinstance(template, str):
        raise PromptGrammarError("template must be text")
    cleaned = template.strip()
    if not cleaned or len(cleaned) > MAX_TEMPLATE:
        raise PromptGrammarError(f"template must be 1 to {MAX_TEMPLATE} characters")

    declared = {slot.name for slot in slots}
    placeholders = _PLACEHOLDER.findall(cleaned)
    if len(placeholders) != len(set(placeholders)):
        raise PromptGrammarError("template repeats a placeholder")
    unknown = sorted(set(placeholders) - declared - {"description"})
    if unknown:
        raise PromptGrammarError(f"template uses undeclared placeholder(s): {', '.join(unknown)}")
    missing = sorted({slot.name for slot in slots if slot.required} - set(placeholders))
    if missing:
        raise PromptGrammarError(f"template omits required slot(s): {', '.join(missing)}")

    # Whatever is not a placeholder must be the trigger or punctuation. Anything
    # left over is a word, and a word here is prose.
    remainder = _PLACEHOLDER.sub(" ", cleaned)
    words = [item for item in _TEMPLATE_SEPARATOR.split(remainder) if item]
    if words.count(trigger) != 1:
        raise PromptGrammarError("template must name the trigger exactly once")
    leftover = [item for item in words if item != trigger]
    if leftover:
        raise PromptGrammarError(
            "template must contain only the trigger, placeholders and punctuation; "
            f"found {leftover[0][:32]!r}"
        )
    return cleaned


def normalize_grammar(
    payload: object,
    *,
    approved_prose: frozenset[str] = frozenset(),
    verified_values: Mapping[str, frozenset[str]] | None = None,
) -> PromptGrammar:
    """Reduce a published grammar to a bounded one, or refuse it.

    `approved_prose` holds digests of exact texts approved on this machine.
    `verified_values` holds the values locally observed to work, per slot. The
    payload cannot contribute to either: it is the thing being checked.
    """

    if not isinstance(payload, dict):
        raise PromptGrammarError("a grammar must be an object")
    unknown_keys = sorted(set(payload) - _GRAMMAR_KEYS)
    if unknown_keys:
        raise PromptGrammarError(f"unknown grammar key(s): {', '.join(unknown_keys)}")

    version = payload.get("schema_version", SCHEMA_VERSION)
    if type(version) is not int or version != SCHEMA_VERSION:
        raise PromptGrammarError(f"grammar schema version must be {SCHEMA_VERSION}")

    trigger = _identifier(payload.get("trigger"), "trigger")
    evidence = {key: frozenset(value) for key, value in (verified_values or {}).items()}

    raw_slots = payload.get("slots", [])
    if not isinstance(raw_slots, list):
        raise PromptGrammarError("slots must be a list")
    if len(raw_slots) > MAX_SLOTS:
        raise PromptGrammarError(f"a grammar carries at most {MAX_SLOTS} slots")

    slots: list[GrammarSlot] = []
    seen: set[str] = set()
    for entry in raw_slots:
        if not isinstance(entry, dict):
            raise PromptGrammarError("each slot must be an object")
        unknown_slot_keys = sorted(set(entry) - _SLOT_KEYS)
        if unknown_slot_keys:
            raise PromptGrammarError(f"unknown slot key(s): {', '.join(unknown_slot_keys)}")
        name = _identifier(entry.get("name"), "slot name")
        if name in seen:
            raise PromptGrammarError(f"slot {name!r} is declared twice")
        seen.add(name)

        raw_values = entry.get("values", [])
        if not isinstance(raw_values, list) or not raw_values:
            raise PromptGrammarError(f"slot {name!r} must list its permitted values")
        if len(raw_values) > MAX_VALUES_PER_SLOT:
            raise PromptGrammarError(
                f"slot {name!r} carries more than {MAX_VALUES_PER_SLOT} values"
            )

        locally_verified = evidence.get(name, frozenset())
        values: list[SlotValue] = []
        for item in raw_values:
            # Values are plain identifiers only. An object here would invite a
            # `verified` field, and a downloaded file must have no way to say
            # that this machine verified anything.
            candidate = _identifier(item, f"value in slot {name!r}")
            if any(existing.value == candidate for existing in values):
                raise PromptGrammarError(f"slot {name!r} repeats the value {candidate!r}")
            values.append(SlotValue(candidate, verified=candidate in locally_verified))
        slots.append(
            GrammarSlot(name, _strict_bool(entry.get("required", False), "required"), tuple(values))
        )

    frozen_slots = tuple(slots)
    template = _parse_template(payload.get("template"), trigger, frozen_slots)

    raw_examples = payload.get("examples", [])
    if not isinstance(raw_examples, list):
        raise PromptGrammarError("examples must be a list")
    if len(raw_examples) > MAX_EXAMPLES:
        raise PromptGrammarError(f"a grammar carries at most {MAX_EXAMPLES} examples")
    examples = tuple(
        text
        for text in (_prose(item, "example", approved_prose) for item in raw_examples)
        if text is not None
    )

    guidance_value = payload.get("description_guidance")
    guidance = (
        None if guidance_value is None else _prose(guidance_value, "guidance", approved_prose)
    )

    return PromptGrammar(
        schema_version=SCHEMA_VERSION,
        trigger=trigger,
        slots=frozen_slots,
        template=template,
        description_guidance=guidance,
        examples=examples,
    )


def unsupported_slot_request(grammar: PromptGrammar, slot_name: str, candidate: str) -> str | None:
    """Why this request cannot be relied on, or `None` if it can.

    Published and verified are separate answers on purpose. A value the vendor
    lists but this machine has never seen work is the case that fails silently -
    it degrades to something adjacent and renders with full confidence - so it
    is reported rather than permitted.
    """

    slot = grammar.slot(slot_name)
    if slot is None:
        return f"this adapter has no {slot_name} setting"
    if slot.verified(candidate):
        return None
    if slot.published(candidate):
        return (
            f"{candidate!r} is listed for {slot_name} but has never been seen to work here. "
            "A listed value that the model does not implement produces a confident "
            "result for something else rather than an error."
        )
    # Deliberately no nearest-match suggestion. Substituting the closest value
    # was measured and rejected: a value asserts its own meaning and the model
    # follows it over the surrounding description.
    return (
        f"this adapter does not support {candidate!r} for {slot_name}. Describing the "
        f"subject more closely may work; naming a different {slot_name} will produce "
        "that other one instead."
    )


def rewriter_instruction(grammar: PromptGrammar) -> str:
    """The instruction fragment that teaches a rewriter this grammar.

    Only verified values are offered. Prose appears only when its exact text was
    approved here, so a grammar whose document was edited after approval renders
    as structure alone rather than silently regaining its voice.
    """

    lines = [f"Write the prompt in this format: {grammar.template}"]
    for slot in grammar.slots:
        verified = slot.verified_values()
        if not verified:
            continue
        requirement = "Required" if slot.required else "Optional"
        lines.append(f"{requirement} {slot.name}, one of: {', '.join(verified)}.")
    if grammar.description_guidance:
        lines.append(grammar.description_guidance)
    if grammar.examples:
        lines.append("Worked examples:")
        lines.extend(grammar.examples)
    rendered = "\n".join(lines)
    if len(rendered) > MAX_INSTRUCTION_CHARS:
        raise PromptGrammarError(f"the rendered grammar exceeds {MAX_INSTRUCTION_CHARS} characters")
    return rendered
