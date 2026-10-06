from enum import Enum
from typing import Protocol
from pydantic import BaseModel, Field


class ObjectiveType(str, Enum):
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    TOOL_MISUSE = "tool_misuse"
    DATA_EXFILTRATION = "data_exfiltration"
    POLICY_BYPASS = "policy_bypass"
    INDIRECT_INJECTION = "indirect_injection"
    GOAL_HIJACK = "goal_hijack"


class CheckType(str, Enum):
    CANARY = "canary"
    TOOL_CALL = "tool_call"
    STATE = "state"
    REGEX = "regex"
    LLM_JUDGE = "llm_judge"


class SuccessCheck(BaseModel):
    type: CheckType
    params: dict = Field(default_factory=dict)
    level: int = 2  # score if this check fires: 1 partial, 2 full


class Objective(BaseModel):
    id: str
    type: ObjectiveType
    description: str
    technique_refs: list[str] = []
    success_checks: list[SuccessCheck]
    max_turns: int = 10
    max_turns_per_card: int = 3


class TargetProfile(BaseModel):
    name: str
    endpoint: str
    description: str
    tags: list[str] = []
    tools: list[str] = []
    canaries: dict[str, str] = {}


class AttackCard(BaseModel):
    id: str
    name: str
    refs: list[str]
    objective_types: list[ObjectiveType]
    preconditions: list[str] = []
    strategy: str
    example_patterns: list[str] = []
    converters: list[str] = []
    known_weaknesses: str = ""


class TargetReply(BaseModel):
    text: str
    tool_calls: list[dict] = []
    trace: list[dict] = []


class Turn(BaseModel):
    index: int
    card_id: str
    prompt: str
    reply: TargetReply
    score: int = 0  # 0 none, 1 partial, 2 full
    failure_note: str = ""


class Status(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    EXHAUSTED = "exhausted"


class ObjectiveState(BaseModel):
    objective: Objective
    turns: list[Turn] = []
    banned_cards: set[str] = set()
    status: Status = Status.RUNNING

    @property
    def best_score(self) -> int:
        return max((t.score for t in self.turns), default=0)


class Target(Protocol):
    def reset(self) -> None: ...
    def send(self, prompt: str) -> TargetReply: ...


class Scorer(Protocol):
    def score(self, obj: Objective, prompt: str, reply: TargetReply,
              profile: TargetProfile) -> tuple[int, str]:
        """Return (score 0-2, short failure or progress note)."""


class Retriever(Protocol):
    def get_cards(self, obj: Objective, profile: TargetProfile,
                  exclude: set[str], k: int) -> list[AttackCard]: ...


class Attacker(Protocol):
    def next_prompt(self, obj: Objective, profile: TargetProfile,
                    cards: list[AttackCard], history: list[Turn]
                    ) -> tuple[str, str]:
        """Return (card_id used, prompt to send)."""


def run_objective(obj: Objective, profile: TargetProfile, target: Target,
                  scorer: Scorer, retriever: Retriever, attacker: Attacker,
                  cards_per_turn=4) -> ObjectiveState:
    state = ObjectiveState(objective=obj)
    uses: dict[str, int] = {}
    misses: dict[str, int] = {}
    target.reset()
    for i in range(obj.max_turns):
        cards = retriever.get_cards(obj, profile, state.banned_cards, cards_per_turn)
        if not cards:
            break
        card_id, prompt = attacker.next_prompt(obj, profile, cards, state.turns)
        reply = target.send(prompt)
        score, note = scorer.score(obj, prompt, reply, profile)
        state.turns.append(Turn(index=i, card_id=card_id, prompt=prompt,
                                reply=reply, score=score, failure_note=note))
        if score >= 2:
            state.status = Status.SUCCESS
            return state
        uses[card_id] = uses.get(card_id, 0) + 1
        if score == 0:
            misses[card_id] = misses.get(card_id, 0) + 1
        if misses.get(card_id, 0) >= 2 or uses[card_id] >= obj.max_turns_per_card:
            state.banned_cards.add(card_id)
    state.status = Status.EXHAUSTED
    return state
