"""Rule-based scorer for deterministic ATHENA success checks."""

from core import Objective, TargetProfile, TargetReply
from scorers.checks import run_check


class DeterministicScorer:
    """Score replies by evaluating every configured deterministic check."""

    def score(
        self,
        obj: Objective,
        prompt: str,
        reply: TargetReply,
        profile: TargetProfile,
    ) -> tuple[int, str]:
        """Return the highest fired level and a safe one-line summary."""
        del prompt
        if not obj.success_checks:
            return 0, "no success checks defined"

        fired: list[str] = []
        skipped: list[str] = []
        score = 0

        for check in obj.success_checks:
            result = run_check(check, reply, profile)
            configured_label = check.params.get("label")
            label = (
                configured_label
                if isinstance(configured_label, str) and configured_label
                else check.type.value
            )
            label = label.replace("\r", " ").replace("\n", " ")

            if result.fired:
                level = max(0, min(2, check.level))
                score = max(score, level)
                fired.append(f"{label}(L{level})")
            if result.skipped:
                detail = result.detail.replace("\r", " ").replace("\n", " ")
                skipped.append(f"{label} ({detail})")

        note = "fired: " + ", ".join(fired) if fired else "no checks fired"
        if skipped:
            note += "; skipped: " + ", ".join(skipped)
        return score, note
