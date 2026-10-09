"""Environment-based configuration for the LLM client."""

import math
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, SecretStr

from .types import LLMConfigError


class LLMConfig(BaseModel):
    base_url: str
    model: str
    api_key: SecretStr
    timeout: float = 60.0
    max_retries: int = 5
    daily_token_limit: int | None = None
    reasoning_effort: str | None = None


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise LLMConfigError("dotenv file must be saved as UTF-8") from exc
    except OSError as exc:
        raise LLMConfigError(f"could not read dotenv file: {path}") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise LLMConfigError(f"missing required environment variable {name}")
    return value


def _optional_number(
    values: Mapping[str, str],
    name: str,
    converter: type[int] | type[float],
    default: int | float | None,
) -> int | float | None:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = converter(raw)
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(f"{name} must be numeric and non-negative") from exc
    if parsed < 0 or (converter is float and not math.isfinite(parsed)):
        raise LLMConfigError(f"{name} must be numeric and non-negative")
    return parsed


def load_llm_config(
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> LLMConfig:
    """Load config from one explicit dotenv file and an overriding environment."""

    values = _read_dotenv(dotenv_path) if dotenv_path is not None else {}
    values.update(os.environ if env is None else env)

    timeout = _optional_number(values, "ATHENA_LLM_TIMEOUT", float, 60.0)
    max_retries = _optional_number(values, "ATHENA_LLM_MAX_RETRIES", int, 5)
    daily_limit = _optional_number(
        values, "ATHENA_LLM_DAILY_TOKEN_LIMIT", int, None
    )
    reasoning_effort = values.get("ATHENA_LLM_REASONING_EFFORT") or None

    return LLMConfig(
        base_url=_required(values, "ATHENA_LLM_BASE_URL"),
        model=_required(values, "ATHENA_LLM_MODEL"),
        api_key=SecretStr(_required(values, "ATHENA_LLM_API_KEY")),
        timeout=float(timeout),
        max_retries=int(max_retries),
        daily_token_limit=None if daily_limit is None else int(daily_limit),
        reasoning_effort=reasoning_effort,
    )
