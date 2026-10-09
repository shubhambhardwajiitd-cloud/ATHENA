"""Simple deterministic attack-card retrieval."""

from enum import IntEnum
from pathlib import Path

from core import AttackCard, Objective, TargetProfile


class Rung(IntEnum):
    """Preference levels in the strategy ladder, from earliest to latest."""

    DIRECT_ASK = 0
    ENCODING = 1
    PERSONA = 2
    MULTI_TURN = 3
    INDIRECT_INJECTION = 4


class SimpleRetriever:
    """Retrieve cards in rung-then-load-order, not relevance-ranked.

    The rung map is external because ``AttackCard`` has no rung field, as
    specified by design section 13.
    """

    def __init__(
        self,
        cards: list[AttackCard],
        rungs: dict[str, int] | None = None,
    ) -> None:
        card_ids: set[str] = set()
        for card in cards:
            if card.id in card_ids:
                raise ValueError(f"duplicate card id: {card.id}")
            card_ids.add(card.id)

        rung_copy = dict(rungs) if rungs is not None else {}
        for card_id, rung in rung_copy.items():
            if card_id not in card_ids:
                raise ValueError(f"rung references unknown card id: {card_id}")
            if not isinstance(rung, int) or isinstance(rung, bool):
                raise ValueError(f"rung must be an int for card id: {card_id}")
            if rung < 0:
                raise ValueError(f"negative rung for card id: {card_id}")

        self._cards: tuple[AttackCard, ...] = tuple(cards)
        self._rungs: dict[str, int] = rung_copy

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        rungs: dict[str, int] | None = None,
    ) -> "SimpleRetriever":
        """Load cards from a directory and construct a retriever."""
        from loaders import load_cards

        return cls(load_cards(path), rungs=rungs)

    def get_cards(
        self,
        obj: Objective,
        profile: TargetProfile,
        exclude: set[str],
        k: int,
    ) -> list[AttackCard]:
        """Return up to ``k`` applicable, non-excluded cards."""
        if k <= 0:
            return []

        eligible: list[tuple[int, AttackCard]] = []
        for index, card in enumerate(self._cards):
            if (
                obj.type in card.objective_types
                and card.id not in exclude
                and all(tag in profile.tags for tag in card.preconditions)
            ):
                eligible.append((index, card))

        eligible.sort(
            key=lambda item: (
                item[1].id not in self._rungs,
                self._rungs.get(item[1].id, 0),
                item[0],
            )
        )
        return [card for _, card in eligible[:k]]
