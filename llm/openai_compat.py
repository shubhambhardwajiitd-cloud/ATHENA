"""Synchronous client for OpenAI-compatible chat-completions APIs."""

import datetime
import time
from collections.abc import Callable

import httpx

from .config import LLMConfig
from .types import (
    LLMAuthError,
    LLMError,
    LLMModelError,
    LLMOutputTruncatedError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)


class TokenBudget:
    """Track a UTC-day token budget in-process only.

    The counter is not persisted. Provider-side daily windows may not align with
    the UTC date used by this local guard.
    """

    def __init__(
        self,
        daily_limit: int | None,
        clock: Callable[[], datetime.date] = lambda: datetime.datetime.now(
            datetime.UTC
        ).date(),
    ) -> None:
        self.daily_limit = daily_limit
        self._clock = clock
        self._date = clock()
        self.used = 0

    def _reset_if_needed(self) -> None:
        today = self._clock()
        if today != self._date:
            self._date = today
            self.used = 0

    def check(self) -> None:
        self._reset_if_needed()
        if self.daily_limit is not None and self.used >= self.daily_limit:
            raise LLMQuotaExhaustedError("local daily token budget exhausted")

    def record(self, n: int) -> None:
        self._reset_if_needed()
        self.used += n


class OpenAICompatClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        budget: TokenBudget | None = None,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._budget = budget or TokenBudget(config.daily_token_limit)
        self._client = httpx.Client(transport=transport, timeout=config.timeout)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        body: dict[str, object] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._config.reasoning_effort is not None:
            body["reasoning_effort"] = self._config.reasoning_effort

        total_tries = self._config.max_retries + 1
        for attempt in range(1, total_tries + 1):
            try:
                self._budget.check()
            except LLMQuotaExhaustedError as exc:
                exc.attempts = attempt
                raise

            try:
                response = self._client.post(
                    f"{self._config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": (
                            f"Bearer {self._config.api_key.get_secret_value()}"
                        )
                    },
                    json=body,
                )
                result = self._handle_response(response, attempt, messages)
            except (httpx.TimeoutException, httpx.TransportError):
                error: LLMError = LLMTimeoutError(
                    "LLM request timed out or encountered a transport error",
                    attempts=attempt,
                )
                retry_after = None
            except LLMError as exc:
                error = exc
                retry_after = response.headers.get("Retry-After")
            else:
                return result

            if not self._is_retryable(error) or attempt == total_tries:
                raise error
            self._sleep(self._retry_delay(attempt, retry_after))

        raise AssertionError("retry loop did not return or raise")

    def _handle_response(
        self,
        response: httpx.Response,
        attempt: int,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        status = response.status_code
        if status in {401, 403}:
            raise LLMAuthError("LLM authentication failed", attempts=attempt)
        if status == 404:
            raise LLMModelError("LLM model was not found", attempts=attempt)
        if status == 429:
            body_lower = response.text.lower()
            # Groq exposes daily quota failures only through message wording.
            daily_markers = ("per day", "tokens per day", "tpd", "daily")
            if any(marker in body_lower for marker in daily_markers):
                raise LLMQuotaExhaustedError(
                    "LLM daily quota exhausted", attempts=attempt
                )
            raise LLMRateLimitError("LLM rate limit reached", attempts=attempt)
        if 500 <= status <= 599:
            raise LLMServerError(
                f"LLM server returned HTTP {status}", attempts=attempt
            )
        if 400 <= status <= 499:
            if status == 400 and "model" in response.text.lower():
                raise LLMModelError("LLM model request was rejected", attempts=attempt)
            raise LLMError(f"LLM request failed with HTTP {status}", attempts=attempt)

        try:
            payload = response.json()
        except (ValueError, TypeError):
            raise LLMResponseError("LLM response has an invalid schema", attempts=attempt)

        if not isinstance(payload, dict):
            raise LLMResponseError("LLM response has an invalid schema", attempts=attempt)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("LLM response has an invalid schema", attempts=attempt)
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMResponseError("LLM response has an invalid schema", attempts=attempt)
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise LLMResponseError("LLM response has an invalid schema", attempts=attempt)

        content = message["content"]
        finish_reason = choice.get("finish_reason")
        usage = payload.get("usage")
        prompt_tokens = self._usage_value(usage, "prompt_tokens")
        completion_tokens = self._usage_value(usage, "completion_tokens")

        if status == 200:
            self._budget.record(
                self._tokens_used(
                    prompt_tokens,
                    completion_tokens,
                    messages,
                    content,
                )
            )

        if not isinstance(content, str) or not content.strip():
            if finish_reason == "length":
                raise LLMOutputTruncatedError(
                    "LLM output was truncated; raise max_tokens or lower reasoning effort",
                    attempts=attempt,
                )
            raise LLMResponseError("LLM response content is empty", attempts=attempt)

        response_model = payload.get("model")
        model = (
            response_model
            if isinstance(response_model, str) and response_model
            else self._config.model
        )
        return LLMResponse(
            text=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _is_retryable(error: LLMError) -> bool:
        return isinstance(
            error,
            (LLMRateLimitError, LLMServerError, LLMTimeoutError, LLMResponseError),
        ) and not isinstance(error, LLMOutputTruncatedError)

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return float(min(60, 2**attempt))

    @staticmethod
    def _usage_value(usage: object, key: str) -> int | None:
        if not isinstance(usage, dict):
            return None
        value = usage.get(key)
        return value if type(value) is int and value >= 0 else None

    @staticmethod
    def _tokens_used(
        prompt_tokens: int | None,
        completion_tokens: int | None,
        messages: list[dict[str, str]],
        content: object,
    ) -> int:
        if prompt_tokens is not None and completion_tokens is not None:
            return prompt_tokens + completion_tokens
        message_chars = sum(
            len(value)
            for message in messages
            if isinstance(message, dict)
            for value in [message.get("content")]
            if isinstance(value, str)
        )
        content_chars = len(content) if isinstance(content, str) else 0
        return (message_chars + content_chars) // 4
