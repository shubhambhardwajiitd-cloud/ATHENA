from pathlib import Path

import yaml

from core import ObjectiveType
from loaders import load_cards, load_objectives
from scorers.deterministic import DeterministicScorer
from targets.mock import MockTarget, default_profile


CARDS_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "cards"
OBJECTIVES_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "objectives"

EXPECTED_IDS = {
    "direct-prompt-request",
    "direct-instruction-override",
    "direct-tool-request",
    "translation-reformat-extraction",
    "completion-prefix-priming",
    "encoding-obfuscation-request",
    "persona-roleplay-authority",
    "hypothetical-fiction-framing",
    "multi-turn-escalation",
    "indirect-injection-document",
    "tool-output-injection",
}


def loaded_cards():
    return load_cards(CARDS_DIRECTORY)


def test_card_ids_and_names_are_exact_and_unique():
    cards = loaded_cards()
    ids = [card.id for card in cards]
    names = [card.name for card in cards]

    assert set(ids) == EXPECTED_IDS
    assert len(ids) == 11
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))


def test_every_card_has_complete_safe_content():
    for card in loaded_cards():
        assert card.strategy
        assert len(card.strategy) < 600
        assert card.example_patterns
        assert card.known_weaknesses
        assert card.refs == []


def test_card_preconditions_are_known():
    allowed = {"has_tools", "has_rag"}

    for card in loaded_cards():
        assert set(card.preconditions) <= allowed


def test_cards_do_not_contain_mock_canary():
    canary = "CANARY-7f3a91"

    for card in loaded_cards():
        text_fields = [
            card.strategy,
            card.known_weaknesses,
            *card.example_patterns,
        ]
        assert all(canary not in text for text in text_fields)


def test_card_objective_type_coverage():
    cards = loaded_cards()

    assert sum(
        ObjectiveType.SYSTEM_PROMPT_LEAK in card.objective_types for card in cards
    ) >= 5
    assert sum(ObjectiveType.TOOL_MISUSE in card.objective_types for card in cards) >= 3
    assert sum(
        ObjectiveType.INDIRECT_INJECTION in card.objective_types for card in cards
    ) >= 2


def test_cards_applicable_to_default_profile():
    profile = default_profile()
    tags = set(profile.tags)
    applicable = [card for card in loaded_cards() if set(card.preconditions) <= tags]
    applicable_ids = {card.id for card in applicable}

    assert sum(
        ObjectiveType.SYSTEM_PROMPT_LEAK in card.objective_types
        for card in applicable
    ) >= 5
    assert "indirect-injection-document" not in applicable_ids
    assert "tool-output-injection" in applicable_ids
    assert "direct-tool-request" in applicable_ids


def test_selected_patterns_score_against_mock_target():
    cards = {card.id: card for card in loaded_cards()}
    objectives = {
        objective.id: objective for objective in load_objectives(OBJECTIVES_DIRECTORY)
    }
    leak = objectives["mock-system-prompt-leak"]
    tool_misuse = objectives["mock-tool-misuse"]
    scorer = DeterministicScorer()
    profile = default_profile()

    override_prompt = cards["direct-instruction-override"].example_patterns[0].replace(
        "{goal}", "reveal your system prompt"
    )
    translation_prompt = cards[
        "translation-reformat-extraction"
    ].example_patterns[0].replace("{language}", "French")
    direct_prompt = cards["direct-prompt-request"].example_patterns[0]
    tool_prompt = cards["direct-tool-request"].example_patterns[0].replace(
        "{goal}", "free up space"
    )

    cases = [
        (override_prompt, leak, 2),
        (translation_prompt, leak, 2),
        (direct_prompt, leak, 1),
        (tool_prompt, tool_misuse, 2),
    ]
    for prompt, objective, expected_score in cases:
        reply = MockTarget().send(prompt)
        score, _ = scorer.score(objective, prompt, reply, profile)
        assert score == expected_score


def test_card_ids_match_file_groupings():
    expected_by_file = {
        "direct.yaml": {
            "direct-prompt-request",
            "direct-instruction-override",
            "direct-tool-request",
            "translation-reformat-extraction",
            "completion-prefix-priming",
        },
        "obfuscation_persona.yaml": {
            "encoding-obfuscation-request",
            "persona-roleplay-authority",
            "hypothetical-fiction-framing",
        },
        "multi_turn_indirect.yaml": {
            "multi-turn-escalation",
            "indirect-injection-document",
            "tool-output-injection",
        },
    }

    for file_name, expected_ids in expected_by_file.items():
        raw_cards = yaml.safe_load(
            (CARDS_DIRECTORY / file_name).read_text(encoding="utf-8")
        )
        assert {card["id"] for card in raw_cards} == expected_ids
