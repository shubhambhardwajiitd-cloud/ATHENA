import pytest

from core import CheckType, SuccessCheck, TargetProfile, TargetReply
from scorers.checks import (
    check_canary,
    check_regex,
    check_state,
    check_tool_call,
    run_check,
)


def make_profile(canaries=None):
    return TargetProfile(
        name="target",
        endpoint="mock://target",
        description="Test target",
        canaries={} if canaries is None else canaries,
    )


def make_check(check_type, **params):
    return SuccessCheck(type=check_type, params=params)


def test_canary_by_key_fires_only_when_present():
    profile = make_profile({"secret": "CANARY-123"})
    check = make_check(CheckType.CANARY, key="secret")

    assert check_canary(check, TargetReply(text="found CANARY-123"), profile).fired
    assert not check_canary(check, TargetReply(text="nothing here"), profile).fired


def test_canary_by_literal_value_and_case_sensitivity():
    profile = make_profile()
    sensitive = make_check(CheckType.CANARY, value="Secret")
    insensitive = make_check(CheckType.CANARY, value="Secret", ignore_case=True)

    assert check_canary(sensitive, TargetReply(text="Secret"), profile).fired
    assert not check_canary(sensitive, TargetReply(text="secret"), profile).fired
    assert check_canary(insensitive, TargetReply(text="secret"), profile).fired


def test_canary_without_selector_uses_any_profile_canary():
    check = make_check(CheckType.CANARY)

    assert check_canary(
        check,
        TargetReply(text="contains SECOND"),
        make_profile({"one": "FIRST", "two": "SECOND"}),
    ).fired
    result = check_canary(check, TargetReply(text="nothing"), make_profile())
    assert result.skipped
    assert result.detail == "no canaries defined"


@pytest.mark.parametrize(
    ("params", "detail"),
    [
        ({"key": "unknown"}, "canary key not in profile"),
        ({"key": "known", "value": "literal"}, "canary params: give key or value, not both"),
        ({"value": ""}, "empty canary"),
    ],
)
def test_canary_skipped_cases(params, detail):
    result = check_canary(
        SuccessCheck(type=CheckType.CANARY, params=params),
        TargetReply(text="literal"),
        make_profile({"known": "CANARY"}),
    )

    assert not result.fired
    assert result.skipped
    assert result.detail == detail


def test_tool_call_fires_from_reply_or_trace():
    check = make_check(CheckType.TOOL_CALL, names=["delete_records"])
    profile = make_profile()
    direct = TargetReply(
        text="done",
        tool_calls=[{"name": "delete_records", "arguments": {}}],
    )
    traced = TargetReply(
        text="done",
        trace=[{"type": "tool_call", "name": "delete_records"}],
    )

    assert check_tool_call(check, direct, profile).fired
    assert check_tool_call(check, traced, profile).fired
    assert not check_tool_call(
        check,
        TargetReply(text="done", tool_calls=[{"name": "search_docs"}]),
        profile,
    ).fired


@pytest.mark.parametrize("params", [{}, {"names": []}])
def test_tool_call_without_names_is_skipped(params):
    result = check_tool_call(
        SuccessCheck(type=CheckType.TOOL_CALL, params=params),
        TargetReply(text="done"),
        make_profile(),
    )

    assert result.skipped
    assert result.detail == "tool_call has no names"


def test_tool_call_ignores_non_dict_entries():
    check = make_check(CheckType.TOOL_CALL, names=["delete_records"])
    reply = TargetReply.model_construct(
        text="done",
        tool_calls=["delete_records", None],
        trace=["delete_records", None],
    )

    assert not check_tool_call(check, reply, make_profile()).fired


def test_state_equals_not_equals_and_none():
    empty = TargetReply(
        text="done",
        trace=[{"type": "state", "data": {"records": [], "optional": None}}],
    )
    unchanged = TargetReply(
        text="done",
        trace=[{"type": "state", "data": {"records": ["rec-1"]}}],
    )
    profile = make_profile()

    assert check_state(
        make_check(CheckType.STATE, key="records", equals=[]), empty, profile
    ).fired
    assert not check_state(
        make_check(CheckType.STATE, key="records", equals=[]), unchanged, profile
    ).fired
    assert not check_state(
        make_check(CheckType.STATE, key="records", not_equals=[]), empty, profile
    ).fired
    assert check_state(
        make_check(CheckType.STATE, key="records", not_equals=[]), unchanged, profile
    ).fired
    assert check_state(
        make_check(CheckType.STATE, key="optional", equals=None), empty, profile
    ).fired


def test_state_uses_last_state_entry():
    reply = TargetReply(
        text="done",
        trace=[
            {"type": "state", "data": {"records": ["rec-1"]}},
            {"type": "event", "data": {}},
            {"type": "state", "data": {"records": []}},
        ],
    )

    assert check_state(
        make_check(CheckType.STATE, key="records", equals=[]),
        reply,
        make_profile(),
    ).fired


@pytest.mark.parametrize(
    ("params", "trace", "detail"),
    [
        ({"key": "records", "equals": []}, [], "no state entry"),
        (
            {"key": "records", "equals": []},
            [{"type": "state", "data": {"other": []}}],
            "state key missing",
        ),
        ({"equals": []}, [{"type": "state", "data": {}}], "state params invalid"),
        (
            {"key": "records", "equals": [], "not_equals": []},
            [{"type": "state", "data": {"records": []}}],
            "state params invalid",
        ),
        (
            {"key": "records"},
            [{"type": "state", "data": {"records": []}}],
            "state params invalid",
        ),
    ],
)
def test_state_skipped_cases(params, trace, detail):
    result = check_state(
        SuccessCheck(type=CheckType.STATE, params=params),
        TargetReply(text="done", trace=trace),
        make_profile(),
    )

    assert result.skipped
    assert result.detail == detail


def test_regex_matching_ignore_case_and_skips():
    profile = make_profile()
    reply = TargetReply(text="Access GRANTED")

    assert check_regex(
        make_check(CheckType.REGEX, pattern="GRANTED"), reply, profile
    ).fired
    assert check_regex(
        make_check(CheckType.REGEX, pattern="granted", ignore_case=True),
        reply,
        profile,
    ).fired

    invalid = check_regex(
        make_check(CheckType.REGEX, pattern="["), reply, profile
    )
    missing = check_regex(make_check(CheckType.REGEX), reply, profile)
    assert invalid.skipped and invalid.detail == "invalid regex"
    assert missing.skipped and missing.detail == "regex has no pattern"


def test_llm_judge_is_skipped_by_dispatcher():
    result = run_check(
        make_check(CheckType.LLM_JUDGE),
        TargetReply(text="anything"),
        make_profile(),
    )

    assert not result.fired
    assert result.skipped
    assert result.detail == "llm_judge not supported by deterministic scorer"


def test_fired_check_details_do_not_expose_sensitive_or_matched_text():
    canary = "VERY-SECRET-CANARY"
    matched = "private phrase"
    profile = make_profile({"secret": canary})

    canary_result = check_canary(
        make_check(CheckType.CANARY, key="secret"),
        TargetReply(text=canary),
        profile,
    )
    regex_result = check_regex(
        make_check(CheckType.REGEX, pattern=matched),
        TargetReply(text=matched),
        profile,
    )

    assert canary not in canary_result.detail
    assert matched not in regex_result.detail
