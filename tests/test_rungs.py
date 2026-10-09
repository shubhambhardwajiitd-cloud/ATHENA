from pathlib import Path

from loaders import load_cards, load_objectives
from retriever import DEFAULT_RUNGS, Rung, SimpleRetriever
from targets.mock import default_profile


ROOT = Path(__file__).resolve().parents[1]
CARDS = load_cards(ROOT / "data" / "cards")
OBJECTIVES = load_objectives(ROOT / "data" / "objectives")


def test_default_rungs_exactly_cover_real_card_ids():
    card_ids = {card.id for card in CARDS}
    rung_ids = set(DEFAULT_RUNGS)
    missing = sorted(card_ids - rung_ids)
    extra = sorted(rung_ids - card_ids)

    assert rung_ids == card_ids, f"missing ids: {missing}; extra ids: {extra}"


def test_every_default_rung_value_is_a_rung_member():
    assert all(isinstance(rung, Rung) for rung in DEFAULT_RUNGS.values())


def test_real_cards_construct_with_default_rungs():
    SimpleRetriever(CARDS, DEFAULT_RUNGS)


def test_real_objective_results_have_non_decreasing_rungs():
    retriever = SimpleRetriever(CARDS, DEFAULT_RUNGS)
    profile = default_profile()

    for objective in OBJECTIVES:
        cards = retriever.get_cards(objective, profile, set(), 11)
        rungs = [DEFAULT_RUNGS[card.id] for card in cards]
        assert rungs == sorted(rungs), (
            f"rungs are not non-decreasing for objective {objective.id}: {rungs}"
        )


def test_system_prompt_leak_starts_at_direct_ask_rung():
    retriever = SimpleRetriever(CARDS, DEFAULT_RUNGS)
    objective = next(
        objective
        for objective in OBJECTIVES
        if objective.id == "mock-system-prompt-leak"
    )

    cards = retriever.get_cards(objective, default_profile(), set(), 11)

    assert cards
    assert DEFAULT_RUNGS[cards[0].id] is Rung.DIRECT_ASK
