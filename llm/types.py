"""Shared types and errors for LLM clients."""

from typing import Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse: ...


class LLMError(Exception):
    """Base error raised by an LLM client."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class LLMConfigError(LLMError, ValueError):
    """The LLM configuration is missing or invalid."""


class LLMAuthError(LLMError):
    """The provider rejected the supplied credentials."""


class LLMModelError(LLMError):
    """The configured model is invalid or unavailable."""


class LLMQuotaExhaustedError(LLMError):
    """The provider or local daily quota has been exhausted."""


class LLMRateLimitError(LLMError):
    """A retryable provider rate limit was reached."""


class LLMServerError(LLMError):
    """The provider returned a retryable server error."""


class LLMTimeoutError(LLMError):
    """A retryable timeout or transport failure occurred."""


class LLMResponseError(LLMError):
    """The provider returned an unusable response."""


class LLMOutputTruncatedError(LLMResponseError):
    """The provider exhausted its output allowance before returning content."""
