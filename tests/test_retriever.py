from pathlib import Path

import pytest

from core import (
    AttackCard,
    Objective,
    ObjectiveType,
    Status,
    SuccessCheck,
    TargetProfile,
    TargetReply,
    run_objective,
)
from loaders import load_cards, load_objectives
from retriever import Rung, SimpleRetriever
from targets.mock import default_profile


def make_card(
    card_id: str,
    objective_types: list[ObjectiveType] | None = None,
    preconditions: list[str] | None = None,
) -> AttackCard:
    return AttackCard(
        id=card_id,
        name=card_id,
        refs=[],
        objective_types=objective_types or [ObjectiveType.SYSTEM_PROMPT_LEAK],
        preconditions=preconditions or [],
        strategy=f"Strategy for {card_id}",
    )


def make_objective(
    objective_type: ObjectiveType = ObjectiveType.SYSTEM_PROMPT_LEAK,
    max_turns: int = 10,
    max_turns_per_card: int = 3,
) -> Objective:
    return Objective(
        id="objective",
        type=objective_type,
        description="Synthetic objective",
        success_checks=[SuccessCheck(type="regex", params={"pattern": "never"})],
        max_turns=max_turns,
        max_turns_per_card=max_turns_per_card,
    )


def make_profile(tags: list[str] | None = None) -> TargetProfile:
    return TargetProfile(
        name="target",
        endpoint="mock://target",
        description="Synthetic target",
        tags=tags or [],
    )


def ids(cards: list[AttackCard]) -> list[str]:
    return [card.id for card in cards]


def test_type_filter_returns_only_matching_cards():
    matching = make_card("matching")
    other = make_card("other", [ObjectiveType.TOOL_MISUSE])
    retriever = SimpleRetriever([other, matching])

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "matching"
    ]


def test_excluded_ids_are_absent_without_mutating_exclude():
    retriever = SimpleRetriever([make_card("first"), make_card("second")])
    exclude = {"first"}

    assert ids(retriever.get_cards(make_objective(), make_profile(), exclude, 10)) == [
        "second"
    ]
    assert exclude == {"first"}


def test_preconditions_require_all_exact_case_sensitive_tags():
    cards = [
        make_card("none"),
        make_card("one", preconditions=["has_tools"]),
        make_card("two", preconditions=["has_tools", "has_rag"]),
        make_card("wrong-case", preconditions=["HAS_TOOLS"]),
    ]
    retriever = SimpleRetriever(cards)

    assert ids(
        retriever.get_cards(
            make_objective(), make_profile(["has_tools", "has_rag"]), set(), 10
        )
    ) == ["none", "one", "two"]
    assert ids(
        retriever.get_cards(make_objective(), make_profile(["has_tools"]), set(), 10)
    ) == ["none", "one"]


def test_poisoned_content_precondition_convention():
    always = make_card("always")
    poisoned = make_card("poisoned", preconditions=["has_poisoned_content"])
    retriever = SimpleRetriever([always, poisoned])

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "always"
    ]
    assert ids(
        retriever.get_cards(
            make_objective(), make_profile(["has_poisoned_content"]), set(), 10
        )
    ) == ["always", "poisoned"]


def test_rung_order_overrides_constructor_order():
    cards = [make_card("persona"), make_card("direct"), make_card("encoding")]
    retriever = SimpleRetriever(
        cards,
        {"persona": Rung.PERSONA, "direct": Rung.DIRECT_ASK, "encoding": 1},
    )

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "direct",
        "encoding",
        "persona",
    ]


def test_same_rung_preserves_constructor_order():
    retriever = SimpleRetriever(
        [make_card("second"), make_card("first")],
        {"second": 1, "first": 1},
    )

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "second",
        "first",
    ]


def test_unmapped_cards_follow_all_mapped_cards_in_constructor_order():
    retriever = SimpleRetriever(
        [make_card("unmapped-a"), make_card("mapped"), make_card("unmapped-b")],
        {"mapped": 9},
    )

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "mapped",
        "unmapped-a",
        "unmapped-b",
    ]


@pytest.mark.parametrize("rungs", [None, {}])
def test_no_rungs_preserves_constructor_order(rungs):
    retriever = SimpleRetriever(
        [make_card("third"), make_card("first"), make_card("second")], rungs
    )

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "third",
        "first",
        "second",
    ]


def test_excluding_lowest_rung_advances_to_next_rung():
    retriever = SimpleRetriever(
        [make_card("high"), make_card("low"), make_card("middle")],
        {"high": 2, "low": 0, "middle": 1},
    )

    assert ids(retriever.get_cards(make_objective(), make_profile(), {"low"}, 10))[0] == (
        "middle"
    )


@pytest.mark.parametrize(
    ("k", "expected"),
    [(2, ["c0", "c1"]), (0, []), (-1, []), (10, [f"c{i}" for i in range(5)])],
)
def test_k_caps_results(k, expected):
    retriever = SimpleRetriever([make_card(f"c{i}") for i in range(5)])

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), k)) == expected


def test_no_eligible_cards_returns_empty_list():
    retriever = SimpleRetriever([make_card("tool", [ObjectiveType.TOOL_MISUSE])])

    assert retriever.get_cards(make_objective(), make_profile(), set(), 4) == []


def test_duplicate_card_id_names_id_in_error():
    with pytest.raises(ValueError) as exc_info:
        SimpleRetriever([make_card("dup-card-1"), make_card("dup-card-1")])

    assert "dup-card-1" in str(exc_info.value)


def test_unknown_rung_key_names_key_in_error():
    with pytest.raises(ValueError) as exc_info:
        SimpleRetriever([make_card("known")], {"missing-card-9": 0})

    assert "missing-card-9" in str(exc_info.value)


def test_negative_rung_names_key_in_error():
    with pytest.raises(ValueError) as exc_info:
        SimpleRetriever([make_card("neg-card-7")], {"neg-card-7": -1})

    assert "neg-card-7" in str(exc_info.value)


@pytest.mark.parametrize("rung", ["1", 1.5, True])
def test_non_integer_rung_names_card_id_in_error(rung):
    with pytest.raises(ValueError) as exc_info:
        SimpleRetriever([make_card("bad-rung-card")], {"bad-rung-card": rung})

    assert "bad-rung-card" in str(exc_info.value)


@pytest.mark.parametrize("rung", [Rung.ENCODING, 2])
def test_int_enum_and_plain_integer_rungs_are_accepted(rung):
    retriever = SimpleRetriever([make_card("valid-rung-card")], {"valid-rung-card": rung})

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 1)) == [
        "valid-rung-card"
    ]


def test_constructor_inputs_and_returned_lists_are_isolated():
    original_cards = [make_card("first"), make_card("second")]
    original_rungs = {"first": 1, "second": 0}
    retriever = SimpleRetriever(original_cards, original_rungs)

    original_cards.clear()
    original_rungs["first"] = 0
    original_rungs["second"] = 1
    first_result = retriever.get_cards(make_objective(), make_profile(), set(), 10)
    first_result.clear()

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 10)) == [
        "second",
        "first",
    ]


def test_repeated_calls_are_deterministic():
    retriever = SimpleRetriever(
        [make_card("three"), make_card("one"), make_card("two")],
        {"three": 3, "one": 1, "two": 2},
    )

    results = [
        ids(retriever.get_cards(make_objective(), make_profile(), set(), 10))
        for _ in range(10)
    ]

    assert results == [["one", "two", "three"]] * 10


def test_run_objective_uses_cards_in_rung_order_until_exhausted():
    cards = [make_card("last"), make_card("first"), make_card("middle")]
    retriever = SimpleRetriever(cards, {"last": 2, "first": 0, "middle": 1})

    class StubTarget:
        def reset(self) -> None:
            pass

        def send(self, prompt: str) -> TargetReply:
            return TargetReply(text="no")

    class StubScorer:
        def score(self, obj, prompt, reply, profile):
            return 0, "miss"

    class StubAttacker:
        def next_prompt(self, obj, profile, available_cards, history):
            return available_cards[0].id, "p"

    objective = make_objective(max_turns=10, max_turns_per_card=3)
    profile = make_profile()
    state = run_objective(
        objective,
        profile,
        StubTarget(),
        StubScorer(),
        retriever,
        StubAttacker(),
    )

    assert state.status is Status.EXHAUSTED
    assert len(state.turns) == 6
    assert [turn.card_id for turn in state.turns] == [
        "first",
        "first",
        "middle",
        "middle",
        "last",
        "last",
    ]
    assert retriever.get_cards(objective, profile, state.banned_cards, 4) == []


def test_from_dir_loads_cards(tmp_path):
    card_file = tmp_path / "card.yaml"
    card_file.write_text(
        """id: loaded-card
name: Loaded card
refs: []
objective_types: [system_prompt_leak]
preconditions: []
strategy: Loaded from disk
""",
        encoding="utf-8",
    )

    retriever = SimpleRetriever.from_dir(tmp_path)

    assert ids(retriever.get_cards(make_objective(), make_profile(), set(), 4)) == [
        "loaded-card"
    ]


def test_real_data_has_retrievable_card_for_each_mock_objective_type():
    root = Path(__file__).resolve().parents[1]
    cards = load_cards(root / "data" / "cards")
    objectives = load_objectives(root / "data" / "objectives")
    retriever = SimpleRetriever(cards)
    profile = default_profile()

    for objective_type in {objective.type for objective in objectives}:
        objective = next(obj for obj in objectives if obj.type is objective_type)
        assert retriever.get_cards(objective, profile, set(), 4), (
            f"no retrievable card for objective type {objective_type.value}"
        )
