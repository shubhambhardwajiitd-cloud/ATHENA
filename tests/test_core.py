from core import (
    AttackCard,
    CheckType,
    Objective,
    ObjectiveState,
    ObjectiveType,
    Status,
    SuccessCheck,
    TargetProfile,
    TargetReply,
    Turn,
    run_objective,
)


class StubTarget:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.reset_calls = 0
        self.send_calls = []
        self.events = []

    def reset(self):
        self.reset_calls += 1
        self.events.append("reset")

    def send(self, prompt):
        self.send_calls.append(prompt)
        self.events.append("send")
        return next(self.replies)


class StubScorer:
    def __init__(self, results):
        self.results = iter(results)

    def score(self, obj, prompt, reply, profile):
        return next(self.results)


class StubRetriever:
    def __init__(self, cards):
        self.cards = cards
        self.exclude_calls = []

    def get_cards(self, obj, profile, exclude, k):
        self.exclude_calls.append(set(exclude))
        return [card for card in self.cards if card.id not in exclude][:k]


class StubAttacker:
    def __init__(self, results):
        self.results = iter(results)

    def next_prompt(self, obj, profile, cards, history):
        return next(self.results)


def make_objective(**overrides):
    values = {
        "id": "objective-1",
        "type": ObjectiveType.POLICY_BYPASS,
        "description": "Test objective",
        "success_checks": [SuccessCheck(type=CheckType.REGEX)],
    }
    values.update(overrides)
    return Objective(**values)


def make_profile():
    return TargetProfile(name="target", endpoint="sandbox", description="Test target")


def make_card(card_id):
    return AttackCard(
        id=card_id,
        name=f"Card {card_id}",
        refs=[],
        objective_types=[ObjectiveType.POLICY_BYPASS],
        strategy="Test strategy",
    )


def make_replies(count):
    return [TargetReply(text=f"reply-{i}") for i in range(count)]


def test_schema_defaults_and_mutable_defaults_are_not_shared():
    objective_one = make_objective()
    objective_two = make_objective(id="objective-2")
    check_one = SuccessCheck(type=CheckType.CANARY)
    check_two = SuccessCheck(type=CheckType.CANARY)
    profile_one = make_profile()
    profile_two = make_profile()
    card_one = make_card("A")
    card_two = make_card("B")
    reply_one = TargetReply(text="first")
    reply_two = TargetReply(text="second")
    state_one = ObjectiveState(objective=objective_one)
    state_two = ObjectiveState(objective=objective_two)

    assert objective_one.max_turns == 10
    assert objective_one.max_turns_per_card == 3
    assert check_one.level == 2
    assert state_one.turns == []
    assert state_one.banned_cards == set()
    assert state_one.status is Status.RUNNING

    objective_one.technique_refs.append("ref")
    check_one.params["key"] = "value"
    profile_one.tags.append("tag")
    profile_one.tools.append("tool")
    profile_one.canaries["name"] = "value"
    card_one.preconditions.append("condition")
    card_one.example_patterns.append("pattern")
    card_one.converters.append("converter")
    reply_one.tool_calls.append({"name": "tool"})
    reply_one.trace.append({"event": "send"})
    state_one.turns.append(
        Turn(
            index=0,
            card_id="A",
            prompt="prompt",
            reply=TargetReply(text="reply"),
        )
    )
    state_one.banned_cards.add("A")

    assert objective_two.technique_refs == []
    assert check_two.params == {}
    assert profile_two.tags == []
    assert profile_two.tools == []
    assert profile_two.canaries == {}
    assert card_two.preconditions == []
    assert card_two.example_patterns == []
    assert card_two.converters == []
    assert reply_two.tool_calls == []
    assert reply_two.trace == []
    assert state_two.turns == []
    assert state_two.banned_cards == set()


def test_best_score_is_zero_without_turns_and_maximum_with_turns():
    state = ObjectiveState(objective=make_objective())
    assert state.best_score == 0

    state.turns.extend(
        [
            Turn(index=0, card_id="A", prompt="p0", reply=TargetReply(text="r0"), score=1),
            Turn(index=1, card_id="B", prompt="p1", reply=TargetReply(text="r1"), score=0),
            Turn(index=2, card_id="C", prompt="p2", reply=TargetReply(text="r2"), score=2),
        ]
    )
    assert state.best_score == 2


def test_success_on_first_turn_stops_loop():
    target = StubTarget(make_replies(2))
    state = run_objective(
        make_objective(),
        make_profile(),
        target,
        StubScorer([(2, "success")]),
        StubRetriever([make_card("A")]),
        StubAttacker([("A", "prompt-0")]),
    )

    assert state.status is Status.SUCCESS
    assert len(state.turns) == 1
    assert target.send_calls == ["prompt-0"]


def test_target_reset_called_once_before_first_send():
    target = StubTarget(make_replies(1))
    run_objective(
        make_objective(max_turns=1),
        make_profile(),
        target,
        StubScorer([(0, "miss")]),
        StubRetriever([make_card("A")]),
        StubAttacker([("A", "prompt-0")]),
    )

    assert target.reset_calls == 1
    assert target.events == ["reset", "send"]


def test_card_is_banned_after_two_misses():
    retriever = StubRetriever([make_card("A"), make_card("B")])
    state = run_objective(
        make_objective(max_turns=3),
        make_profile(),
        StubTarget(make_replies(3)),
        StubScorer([(0, "miss"), (0, "miss"), (0, "miss")]),
        retriever,
        StubAttacker([("A", "p0"), ("A", "p1"), ("B", "p2")]),
    )

    assert "A" in state.banned_cards
    assert "A" in retriever.exclude_calls[2]


def test_partial_then_miss_keeps_card_alive():
    state = run_objective(
        make_objective(max_turns=2),
        make_profile(),
        StubTarget(make_replies(2)),
        StubScorer([(1, "partial"), (0, "miss")]),
        StubRetriever([make_card("A")]),
        StubAttacker([("A", "p0"), ("A", "p1")]),
    )

    assert "A" not in state.banned_cards


def test_card_is_banned_at_max_turns_per_card():
    state = run_objective(
        make_objective(max_turns=2, max_turns_per_card=2),
        make_profile(),
        StubTarget(make_replies(2)),
        StubScorer([(1, "partial"), (1, "partial")]),
        StubRetriever([make_card("A")]),
        StubAttacker([("A", "p0"), ("A", "p1")]),
    )

    assert "A" in state.banned_cards


def test_exhausted_at_max_turns():
    cards = [make_card("A"), make_card("B"), make_card("C")]
    state = run_objective(
        make_objective(max_turns=3),
        make_profile(),
        StubTarget(make_replies(3)),
        StubScorer([(0, "miss"), (0, "miss"), (0, "miss")]),
        StubRetriever(cards),
        StubAttacker([("A", "p0"), ("B", "p1"), ("C", "p2")]),
    )

    assert state.status is Status.EXHAUSTED
    assert len(state.turns) == 3


def test_exhausted_when_retriever_returns_no_cards():
    target = StubTarget([])
    state = run_objective(
        make_objective(),
        make_profile(),
        target,
        StubScorer([]),
        StubRetriever([]),
        StubAttacker([]),
    )

    assert state.status is Status.EXHAUSTED
    assert state.turns == []
    assert target.send_calls == []


def test_turn_records_all_stub_values_with_sequential_indices():
    replies = [TargetReply(text="first"), TargetReply(text="second")]
    state = run_objective(
        make_objective(max_turns=2),
        make_profile(),
        StubTarget(replies),
        StubScorer([(1, "progress"), (0, "blocked")]),
        StubRetriever([make_card("A"), make_card("B")]),
        StubAttacker([("A", "prompt-a"), ("B", "prompt-b")]),
    )

    assert [turn.index for turn in state.turns] == [0, 1]
    assert state.turns[0].card_id == "A"
    assert state.turns[0].prompt == "prompt-a"
    assert state.turns[0].reply is replies[0]
    assert state.turns[0].score == 1
    assert state.turns[0].failure_note == "progress"
    assert state.turns[1].card_id == "B"
    assert state.turns[1].prompt == "prompt-b"
    assert state.turns[1].reply is replies[1]
    assert state.turns[1].score == 0
    assert state.turns[1].failure_note == "blocked"


def test_success_after_partial_scores():
    target = StubTarget(make_replies(3))
    state = run_objective(
        make_objective(max_turns=5, max_turns_per_card=3),
        make_profile(),
        target,
        StubScorer([(1, "partial"), (1, "partial"), (2, "success")]),
        StubRetriever([make_card("A"), make_card("B")]),
        StubAttacker([("A", "p0"), ("A", "p1"), ("B", "p2")]),
    )

    assert state.status is Status.SUCCESS
    assert len(state.turns) == 3
    assert state.best_score == 2
    assert len(target.send_calls) == 3
