"""Default strategy-ladder rungs for ATHENA's attack cards."""

from .simple import Rung


DEFAULT_RUNGS: dict[str, Rung] = {
    "direct-prompt-request": Rung.DIRECT_ASK,
    "direct-instruction-override": Rung.DIRECT_ASK,
    "direct-tool-request": Rung.DIRECT_ASK,
    "encoding-obfuscation-request": Rung.ENCODING,
    "translation-reformat-extraction": Rung.ENCODING,
    "completion-prefix-priming": Rung.ENCODING,
    "persona-roleplay-authority": Rung.PERSONA,
    "hypothetical-fiction-framing": Rung.PERSONA,
    "multi-turn-escalation": Rung.MULTI_TURN,
    "indirect-injection-document": Rung.INDIRECT_INJECTION,
    "tool-output-injection": Rung.INDIRECT_INJECTION,
}
