from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

import local_lm.prompt_model_invocation as invocation_module
from local_lm.adapters.base import ChatAdapter, ChatEvent, ChatRequest
from local_lm.prompt_model_invocation import (
    MAX_PROMPT_MODEL_ARGUMENT_FRAGMENTS,
    MAX_PROMPT_MODEL_EVENTS,
    MAX_PROMPT_MODEL_NAME_FRAGMENTS,
    PROMPT_MODEL_INVOCATION_FAILED,
    PromptModelInvocationData,
    PromptModelInvocationError,
    PromptModelInvocationItem,
    invoke_prompt_model_values,
)
from local_lm.prompt_model_values import (
    PROMPT_MODEL_VALUES_TOOL_NAME,
    PromptModelSlotContract,
    PromptModelSlotSpec,
    prompt_model_slot_contract,
    prompt_model_values_sha256,
)
from local_lm.prompt_templates import (
    MAX_TEMPLATE_DOCUMENT_CHARS,
    MAX_TEMPLATE_SLOTS,
    PromptTemplateVariationScope,
    parse_prompt_template_contract,
)
from local_lm.schemas import EngineCapabilities


def _contract() -> PromptModelSlotContract:
    template = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": "{{subject}} in {{style}}, {{lighting}}",
            "slots": [
                {
                    "name": "subject",
                    "mode": "input",
                    "variation_scope": "batch",
                },
                {
                    "name": "style",
                    "mode": "model",
                    "variation_scope": "batch",
                    "guidance": "a concise visual medium",
                },
                {
                    "name": "lighting",
                    "mode": "model",
                    "variation_scope": "item",
                    "guidance": "a different lighting treatment",
                },
            ],
            "resource_policy": {"mode": "inherited"},
        }
    )
    return prompt_model_slot_contract(template, item_count=2)


def _data() -> PromptModelInvocationData:
    return PromptModelInvocationData(
        template_text="{{subject}} in {{style}}, {{lighting}}",
        batch_values=(("subject", "a lighthouse"),),
        items=(
            PromptModelInvocationItem(ordinal=1, values=()),
            PromptModelInvocationItem(ordinal=2, values=()),
        ),
    )


def _payload() -> dict[str, object]:
    return {
        "version": 1,
        "batch_values": {"style": "oil paint"},
        "items": [
            {"ordinal": 1, "values": {"lighting": "soft window light"}},
            {"ordinal": 2, "values": {"lighting": "hard rim light"}},
        ],
    }


def _call(
    *,
    name: object = PROMPT_MODEL_VALUES_TOOL_NAME,
    arguments: object | None = None,
    index: object = 0,
    call_id: object | None = "call_1",
) -> ChatEvent:
    function: dict[str, object] = {"name": name}
    function["arguments"] = (
        json.dumps(_payload(), separators=(",", ":")) if arguments is None else arguments
    )
    call: dict[str, object] = {"index": index, "function": function}
    if call_id is not None:
        call["id"] = call_id
    return ChatEvent(type="tool_delta", data={"tool_calls": [call]})


class SequenceAdapter(ChatAdapter):
    def __init__(self, attempts: list[list[ChatEvent] | BaseException]) -> None:
        self._attempts = list(attempts)
        self.requests: list[ChatRequest] = []

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        response = self._attempts.pop(0)

        async def events() -> AsyncIterator[ChatEvent]:
            if isinstance(response, BaseException):
                raise response
            for event in response:
                yield event

        return events()

    async def capabilities(self) -> EngineCapabilities:
        raise AssertionError("capabilities are outside this boundary")

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        raise AssertionError("token counting is outside this boundary")

    async def cancel(self, run_id: str) -> None:
        raise AssertionError("cancellation is outside this boundary")

    async def close(self) -> None:
        raise AssertionError("adapter ownership remains with the caller")


async def _failed(
    adapter: SequenceAdapter,
    *,
    contract: PromptModelSlotContract | None = None,
    data: PromptModelInvocationData | None = None,
) -> PromptModelInvocationError:
    with pytest.raises(PromptModelInvocationError) as caught:
        await invoke_prompt_model_values(
            adapter,
            contract=contract or _contract(),
            data=data or _data(),
        )
    assert str(caught.value) == PROMPT_MODEL_INVOCATION_FAILED
    assert repr(caught.value) == "PromptModelInvocationError('Prompt model invocation failed.')"
    return caught.value


@pytest.mark.asyncio
async def test_valid_call_returns_codec_values_digest_and_content_free_evidence() -> None:
    contract = _contract()
    data = _data()
    before = copy.deepcopy(data)
    adapter = SequenceAdapter([[_call(), ChatEvent(type="complete")]])

    result = await invoke_prompt_model_values(adapter, contract=contract, data=data)

    assert result.values.batch_values == (("style", "oil paint"),)
    assert result.values.items[1].values == (("lighting", "hard rim light"),)
    assert result.values_sha256 == prompt_model_values_sha256(result.values, contract=contract)
    assert result.attempts == (
        invocation_module.PromptModelAttemptEvidence(
            attempt=1,
            event_count=2,
            call_count=1,
            name_fragment_count=1,
            argument_fragment_count=1,
            aggregate_characters=result.attempts[0].aggregate_characters,
            aggregate_bytes=result.attempts[0].aggregate_bytes,
        ),
    )
    assert result.attempts[0].aggregate_characters > 0
    assert result.attempts[0].aggregate_bytes >= result.attempts[0].aggregate_characters
    assert data == before
    rendered = repr(result)
    assert "oil paint" not in rendered
    assert "hard rim light" not in rendered
    assert "attempts=" not in rendered
    assert "lighthouse" not in repr(data)
    assert "lighthouse" not in repr(data.items[0])


@pytest.mark.asyncio
async def test_valid_fragmented_call_and_evidence_are_stable() -> None:
    raw = json.dumps(_payload(), separators=(",", ":"))
    events = [
        ChatEvent(
            type="tool_delta",
            data={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "supply_prompt_"},
                    }
                ]
            },
        ),
        ChatEvent(
            type="tool_delta",
            data={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "name": "model_values",
                            "arguments": raw[: len(raw) // 2],
                        },
                    }
                ]
            },
        ),
        ChatEvent(
            type="tool_delta",
            data={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": raw[len(raw) // 2 :]},
                    }
                ]
            },
        ),
        ChatEvent(type="complete"),
    ]
    first = await invoke_prompt_model_values(
        SequenceAdapter([events]), contract=_contract(), data=_data()
    )
    second = await invoke_prompt_model_values(
        SequenceAdapter([events]), contract=_contract(), data=_data()
    )
    assert first.values == second.values
    assert first.values_sha256 == second.values_sha256
    assert first.attempts == second.attempts
    assert first.attempts[0].name_fragment_count == 2
    assert first.attempts[0].argument_fragment_count == 2


@pytest.mark.asyncio
async def test_request_has_fixed_authority_one_exact_tool_and_untrusted_json_data() -> None:
    adapter = SequenceAdapter([[_call(), ChatEvent(type="complete")]])
    data = _data()

    await invoke_prompt_model_values(adapter, contract=_contract(), data=data)

    request = adapter.requests[0]
    assert request.persistence_scope == "ephemeral"
    assert request.scope_id is None
    assert request.settings == {"temperature": 0, "max_tokens": 4096}
    assert [message["role"] for message in request.messages] == ["system", "user"]
    system = request.messages[0]["content"]
    user = request.messages[1]["content"]
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert data.template_text not in system
    assert "a concise visual medium" not in system
    decoded = json.loads(user)
    assert decoded["template_text"] == data.template_text
    assert decoded["model_slots"]["batch"] == [
        {"guidance": "a concise visual medium", "name": "style"}
    ]
    assert decoded["invocation_values"]["batch"] == {"subject": "a lighthouse"}
    assert len(request.tools) == 1
    function = request.tools[0]["function"]
    assert isinstance(function, dict)
    assert function["name"] == PROMPT_MODEL_VALUES_TOOL_NAME
    assert "a concise visual medium" not in json.dumps(request.tools)
    assert "a different lighting treatment" not in json.dumps(request.tools)
    joined = json.dumps(request.messages) + json.dumps(request.tools)
    for forbidden in (
        "transcript",
        "attachment",
        "credential",
        "workflow_revision_id",
        "lora",
        "queue",
        "media_path",
    ):
        assert forbidden not in joined.lower()


@pytest.mark.asyncio
async def test_invalid_then_valid_uses_one_fixed_non_echoing_repair() -> None:
    invalid_secret = "DO-NOT-ECHO-invalid-private-output"
    adapter = SequenceAdapter(
        [
            [_call(arguments=invalid_secret)],
            [_call(call_id="call_2"), ChatEvent(type="complete")],
        ]
    )

    result = await invoke_prompt_model_values(adapter, contract=_contract(), data=_data())

    assert [attempt.attempt for attempt in result.attempts] == [1, 2]
    assert len(adapter.requests) == 2
    first_user = adapter.requests[0].messages[-1]["content"]
    second_user = adapter.requests[1].messages[-1]["content"]
    assert first_user == second_user
    second_serialized = json.dumps(adapter.requests[1].messages)
    assert invalid_secret not in second_serialized
    assert "prior response was invalid" in second_serialized.lower()
    assert adapter.requests[0].tools == adapter.requests[1].tools


@pytest.mark.asyncio
async def test_live_caller_contract_mutation_cannot_change_snapshotted_authority() -> None:
    caller_contract = _contract()
    expected_contract = _contract()

    class MutatingAdapter(SequenceAdapter):
        def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            self.requests.append(request)
            attempt = len(self.requests)

            async def events() -> AsyncIterator[ChatEvent]:
                if attempt == 1:
                    await asyncio.sleep(0)
                    object.__setattr__(caller_contract, "item_count", 1)
                    object.__setattr__(
                        caller_contract.batch_slots[0],
                        "name",
                        "forged_style",
                    )
                    object.__setattr__(
                        caller_contract.batch_slots[0],
                        "guidance",
                        "forged guidance",
                    )
                    yield _call(arguments="{")
                else:
                    yield _call(call_id="call_2")
                yield ChatEvent(type="complete")

            return events()

    adapter = MutatingAdapter([])
    result = await invoke_prompt_model_values(
        adapter,
        contract=caller_contract,
        data=_data(),
    )

    assert caller_contract.item_count == 1
    assert caller_contract.batch_slots[0].name == "forged_style"
    assert len(adapter.requests) == 2
    for request in adapter.requests:
        user_data = json.loads(request.messages[-1]["content"])
        assert user_data["model_slots"]["item_count"] == 2
        assert user_data["model_slots"]["batch"] == [
            {"guidance": "a concise visual medium", "name": "style"}
        ]
        parameters = request.tools[0]["function"]["parameters"]
        assert isinstance(parameters, dict)
        batch_values = parameters["properties"]["batch_values"]
        assert isinstance(batch_values, dict)
        assert set(batch_values["properties"]) == {"style"}
        items = parameters["properties"]["items"]
        assert isinstance(items, dict)
        assert items["minItems"] == 2
        assert items["maxItems"] == 2
    assert len(result.values.items) == 2
    assert result.values_sha256 == prompt_model_values_sha256(
        result.values,
        contract=expected_contract,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [ChatEvent(type="delta", text="prose only")],
        [],
        # Keep a terminal event so removing the exact tool-name comparison is
        # the only reason this otherwise-valid stream could be accepted.
        [_call(name="wrong_tool"), ChatEvent(type="complete")],
        # The first call is valid by itself. Only the one-call ceiling can
        # prevent a second well-formed entry from being silently truncated.
        [
            ChatEvent(
                type="tool_delta",
                data={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": PROMPT_MODEL_VALUES_TOOL_NAME,
                                "arguments": json.dumps(_payload(), separators=(",", ":")),
                            },
                        },
                        {
                            "index": 1,
                            "function": {"name": "b", "arguments": "{}"},
                        },
                    ]
                },
            ),
            ChatEvent(type="complete"),
        ],
        [_call(name="wrong_tool")],
        [
            ChatEvent(
                type="tool_delta",
                data={
                    "tool_calls": [
                        {"index": 0, "function": {"name": "a", "arguments": "{}"}},
                        {"index": 1, "function": {"name": "b", "arguments": "{}"}},
                    ]
                },
            )
        ],
        [_call(index=1)],
        [_call(index=True)],
        [_call(index="0")],
        [_call(arguments={"version": 1})],
        [
            ChatEvent(
                type="tool_delta",
                data={"tool_calls": [{"index": 0, "function": "not-an-object"}]},
            )
        ],
        [
            ChatEvent(
                type="tool_delta",
                data={"tool_calls": [{"index": 0, "function": {}}]},
            )
        ],
        [_call(arguments='{"version":1,"version":1,"batch_values":{},"items":[]}')],
        [_call(arguments="{")],
    ],
)
async def test_invalid_outputs_get_one_repair_then_the_exact_fixed_failure(
    events: list[ChatEvent],
) -> None:
    adapter = SequenceAdapter([events, events])
    await _failed(adapter)
    assert len(adapter.requests) == 2


@pytest.mark.asyncio
async def test_mixed_string_and_dict_argument_fragments_are_refused() -> None:
    first = ChatEvent(
        type="tool_delta",
        data={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": PROMPT_MODEL_VALUES_TOOL_NAME,
                        "arguments": '{"version":',
                    },
                }
            ]
        },
    )
    second = ChatEvent(
        type="tool_delta",
        data={"tool_calls": [{"index": 0, "function": {"arguments": {"version": 1}}}]},
    )
    adapter = SequenceAdapter([[first, second], [first, second]])
    await _failed(adapter)


@pytest.mark.asyncio
async def test_error_event_and_adapter_exception_are_terminal_without_repair() -> None:
    for response in (
        [ChatEvent(type="error", text="private adapter error")],
        RuntimeError("private adapter exception"),
    ):
        adapter = SequenceAdapter([response])
        await _failed(adapter)
        assert len(adapter.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["error", "cancelled"])
async def test_terminal_event_type_precedes_malformed_text_and_data(
    event_type: str,
) -> None:
    class Hostile:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("malformed terminal payload was inspected")

        def __repr__(self) -> str:
            raise AssertionError("malformed terminal payload was rendered")

    event = ChatEvent(type=event_type)
    event.text = Hostile()  # type: ignore[assignment]
    event.data = Hostile()  # type: ignore[assignment]
    for events in ([event], [_call(), ChatEvent(type="complete"), event]):
        adapter = SequenceAdapter([events])
        await _failed(adapter)
        assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_truncated_cancelled_and_forged_event_streams_are_refused() -> None:
    truncated = [_call()]
    await _failed(SequenceAdapter([truncated, truncated]))

    cancelled = [_call(), ChatEvent(type="cancelled")]
    adapter = SequenceAdapter([cancelled])
    await _failed(adapter)
    assert len(adapter.requests) == 1

    forged = [_call(), ChatEvent(type="forged_empty"), ChatEvent(type="complete")]
    await _failed(SequenceAdapter([forged, forged]))


@pytest.mark.asyncio
async def test_timeout_is_terminal_and_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingAdapter(SequenceAdapter):
        def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
            self.requests.append(request)

            async def events() -> AsyncIterator[ChatEvent]:
                await asyncio.sleep(1)
                yield _call()

            return events()

    monkeypatch.setattr(invocation_module, "PROMPT_MODEL_INVOCATION_TIMEOUT_SECONDS", 0.001)
    adapter = HangingAdapter([])
    await _failed(adapter)
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_cooperative_cancellation_is_not_converted() -> None:
    adapter = SequenceAdapter([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await invoke_prompt_model_values(adapter, contract=_contract(), data=_data())


@pytest.mark.asyncio
async def test_event_argument_and_fragment_limits_are_enforced() -> None:
    too_many_events = [ChatEvent(type="usage") for _ in range(MAX_PROMPT_MODEL_EVENTS + 1)]
    huge = "x" * (MAX_TEMPLATE_DOCUMENT_CHARS + 1)
    name_event = ChatEvent(
        type="tool_delta",
        data={
            "tool_calls": [
                {
                    "index": 0,
                    "function": {
                        "name": PROMPT_MODEL_VALUES_TOOL_NAME,
                        "arguments": huge,
                    },
                }
            ]
        },
    )
    fragment = ChatEvent(
        type="tool_delta",
        data={"tool_calls": [{"index": 0, "function": {"arguments": "x"}}]},
    )
    start = ChatEvent(
        type="tool_delta",
        data={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": PROMPT_MODEL_VALUES_TOOL_NAME},
                }
            ]
        },
    )
    too_many_fragments = [start] + [
        fragment for _ in range(MAX_PROMPT_MODEL_ARGUMENT_FRAGMENTS + 1)
    ]
    name_fragments = [
        ChatEvent(
            type="tool_delta",
            data={"tool_calls": [{"index": 0, "function": {"name": "x"}}]},
        )
        for _ in range(MAX_PROMPT_MODEL_NAME_FRAGMENTS + 1)
    ]
    for events in (
        too_many_events,
        [name_event],
        too_many_fragments,
        name_fragments,
    ):
        await _failed(SequenceAdapter([events, events]))


@pytest.mark.asyncio
async def test_post_terminal_event_and_repeated_call_id_are_refused() -> None:
    raw = json.dumps(_payload(), separators=(",", ":"))
    split = len(raw) // 2
    repeated_id = [
        ChatEvent(
            type="tool_delta",
            data={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": PROMPT_MODEL_VALUES_TOOL_NAME},
                    }
                ]
            },
        ),
        ChatEvent(
            type="tool_delta",
            data={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_2",
                        "function": {"arguments": "{}"},
                    }
                ]
            },
        ),
    ]
    post_terminal = [_call(), ChatEvent(type="complete"), ChatEvent(type="heartbeat")]
    completing_after_terminal = [
        _call(arguments=raw[:split]),
        ChatEvent(type="complete"),
        ChatEvent(
            type="tool_delta",
            data={"tool_calls": [{"index": 0, "function": {"arguments": raw[split:]}}]},
        ),
    ]
    usage_after_terminal = [_call(), ChatEvent(type="complete"), ChatEvent(type="usage")]
    for events in (
        repeated_id,
        post_terminal,
        completing_after_terminal,
        usage_after_terminal,
    ):
        await _failed(SequenceAdapter([events, events]))


@pytest.mark.asyncio
async def test_invalid_invocation_data_and_contract_fail_before_adapter_use() -> None:
    invalid_data = [
        PromptModelInvocationData("", (), _data().items),
        PromptModelInvocationData("{{style}}", (("bad-name", "x"),), _data().items),
        PromptModelInvocationData("{{style}}", (("subject", ""),), _data().items),
        PromptModelInvocationData(
            "{{style}}",
            (),
            (
                PromptModelInvocationItem(ordinal=2, values=()),
                PromptModelInvocationItem(ordinal=1, values=()),
            ),
        ),
    ]
    for data in invalid_data:
        adapter = SequenceAdapter([])
        await _failed(adapter, data=data)
        assert adapter.requests == []

    contract = _contract()
    object.__setattr__(contract, "item_count", 99)
    adapter = SequenceAdapter([])
    await _failed(adapter, contract=contract)
    assert adapter.requests == []


def _sized_contract(
    batch_count: int,
    item_count: int,
) -> PromptModelSlotContract:
    return PromptModelSlotContract(
        version=1,
        item_count=2,
        batch_slots=tuple(
            PromptModelSlotSpec(
                name=f"batch_{index}",
                variation_scope=PromptTemplateVariationScope.BATCH,
                guidance="batch guidance",
            )
            for index in range(batch_count)
        ),
        item_slots=tuple(
            PromptModelSlotSpec(
                name=f"item_{index}",
                variation_scope=PromptTemplateVariationScope.ITEM,
                guidance="item guidance",
            )
            for index in range(item_count)
        ),
    )


@pytest.mark.asyncio
async def test_snapshot_slot_cap_is_non_vacuous_and_combined() -> None:
    at_cap = _sized_contract(MAX_TEMPLATE_SLOTS // 2, MAX_TEMPLATE_SLOTS // 2)
    called = SequenceAdapter([[ChatEvent(type="error")]])
    await _failed(called, contract=at_cap)
    assert len(called.requests) == 1

    over_cap = _sized_contract(
        MAX_TEMPLATE_SLOTS // 2,
        (MAX_TEMPLATE_SLOTS // 2) + 1,
    )
    refused = SequenceAdapter([])
    await _failed(refused, contract=over_cap)
    assert refused.requests == []


@pytest.mark.asyncio
async def test_very_oversized_exact_tuple_is_refused_before_slot_access() -> None:
    class HostileSlot:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("oversized slot was inspected")

        def __repr__(self) -> str:
            raise AssertionError("oversized slot was rendered")

    hostile_slot = HostileSlot()
    oversized = cast(
        tuple[PromptModelSlotSpec, ...],
        (hostile_slot,) * 200_000,
    )
    contract = PromptModelSlotContract(
        version=1,
        item_count=2,
        batch_slots=oversized,
        item_slots=(),
    )
    adapter = SequenceAdapter([])

    await _failed(adapter, contract=contract)

    assert adapter.requests == []


@pytest.mark.asyncio
async def test_contract_and_tuple_subclasses_are_refused_before_descriptors() -> None:
    class HostileTuple(tuple[PromptModelSlotSpec, ...]):
        def __len__(self) -> int:
            raise AssertionError("tuple subclass length was read")

        def __iter__(self) -> Any:
            raise AssertionError("tuple subclass was iterated")

    tuple_contract = PromptModelSlotContract(
        version=1,
        item_count=2,
        batch_slots=cast(
            tuple[PromptModelSlotSpec, ...],
            HostileTuple(_contract().batch_slots),
        ),
        item_slots=_contract().item_slots,
    )
    tuple_adapter = SequenceAdapter([])
    await _failed(tuple_adapter, contract=tuple_contract)
    assert tuple_adapter.requests == []

    class HostileContract(PromptModelSlotContract):
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("contract subclass descriptor was read")

    hostile_contract = object.__new__(HostileContract)
    contract_adapter = SequenceAdapter([])
    await _failed(contract_adapter, contract=hostile_contract)
    assert contract_adapter.requests == []

    missing_fields = object.__new__(PromptModelSlotContract)
    missing_adapter = SequenceAdapter([])
    await _failed(missing_adapter, contract=missing_fields)
    assert missing_adapter.requests == []


@pytest.mark.asyncio
async def test_non_json_event_data_and_oversized_utf8_are_refused() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    bad_events = [
        ChatEvent(type="tool_delta", data=cyclic),
        ChatEvent(type="usage", text="😀" * 65_537),
        ChatEvent(type="usage", data={"value": float("nan")}),
    ]
    for event in bad_events:
        await _failed(SequenceAdapter([[event], [event]]))


@pytest.mark.asyncio
async def test_each_invocation_value_tuple_is_bounded_before_iteration() -> None:
    at_cap = tuple((f"slot_{index}", "x") for index in range(MAX_TEMPLATE_SLOTS))
    accepted = PromptModelInvocationData(
        template_text="{{style}}",
        batch_values=at_cap,
        items=(
            PromptModelInvocationItem(1, ()),
            PromptModelInvocationItem(2, ()),
        ),
    )
    await invoke_prompt_model_values(
        SequenceAdapter([[_call(), ChatEvent(type="complete")]]),
        contract=_contract(),
        data=accepted,
    )

    over_cap = PromptModelInvocationData(
        template_text="{{style}}",
        batch_values=at_cap + (("overflow", "x"),),
        items=accepted.items,
    )
    adapter = SequenceAdapter([])
    await _failed(adapter, data=over_cap)
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_large_event_containers_fail_at_the_container_bound() -> None:
    mapping = {f"k{index}": index for index in range(12_000)}
    sequence = list(range(12_000))
    for data in (mapping, {"values": sequence}):
        event = ChatEvent(type="usage", data=data)
        await _failed(SequenceAdapter([[event], [event]]))


@pytest.mark.asyncio
async def test_hostile_subclasses_and_repr_or_equality_hooks_are_never_called() -> None:
    class HostileStr(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile equality ran")

        def __hash__(self) -> int:
            raise AssertionError("hostile hash ran")

        def __repr__(self) -> str:
            raise AssertionError("hostile repr ran")

    class HostileValue:
        def __repr__(self) -> str:
            raise AssertionError("hostile repr ran")

    bad_data = (
        PromptModelInvocationData(
            "{{style}}",
            ((HostileStr("subject"), "x"),),
            _data().items,
        ),
        PromptModelInvocationData(
            "{{style}}",
            (("subject", HostileStr("x")),),
            _data().items,
        ),
        PromptModelInvocationData(
            "{{style}}",
            (("subject", HostileValue()),),  # type: ignore[arg-type]
            _data().items,
        ),
    )
    for data in bad_data:
        adapter = SequenceAdapter([])
        await _failed(adapter, data=data)
        assert adapter.requests == []

    class HostileEvent(ChatEvent):
        def __repr__(self) -> str:
            raise AssertionError("hostile event repr ran")

    hostile_event = HostileEvent(type="tool_delta")
    await _failed(SequenceAdapter([[hostile_event], [hostile_event]]))
