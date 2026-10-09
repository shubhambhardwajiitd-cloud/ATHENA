import datetime
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from llm import (
    LLMAuthError,
    LLMClient,
    LLMConfig,
    LLMConfigError,
    LLMError,
    LLMModelError,
    LLMOutputTruncatedError,
    LLMQuotaExhaustedError,
    LLMResponseError,
    LLMServerError,
    OpenAICompatClient,
    StubLLMClient,
    TokenBudget,
    load_llm_config,
)


def config(**changes) -> LLMConfig:
    values = {
        "base_url": "https://example.test/v1/",
        "model": "test-model",
        "api_key": SecretStr("sk-test-SECRET-123"),
        "max_retries": 2,
    }
    values.update(changes)
    return LLMConfig(**values)


def response(
    text: str = "answer",
    *,
    status: int = 200,
    model: object | None = "returned-model",
    usage: dict[str, object] | None = None,
    finish_reason: str = "stop",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    payload = {
        "choices": [
            {"message": {"content": text}, "finish_reason": finish_reason}
        ]
    }
    if model is not None:
        payload["model"] = model
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(status, json=payload, headers=headers)


def scripted_client(items, *, settings=None, sleeps=None, budget=None):
    queue = list(items)

    def handler(request):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = OpenAICompatClient(
        settings or config(),
        transport=httpx.MockTransport(handler),
        sleep=(sleeps if sleeps is not None else []).append,
        budget=budget,
    )
    return client, queue


def test_success_parses_response_and_exact_request():
    requests = []

    def handler(request):
        requests.append(request)
        return response(
            usage={"prompt_tokens": 7, "completion_tokens": 3}
        )

    client = OpenAICompatClient(config(), transport=httpx.MockTransport(handler))
    messages = [{"role": "user", "content": "hello"}]

    result = client.generate(messages)

    assert result.text == "answer"
    assert result.model == "returned-model"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert str(requests[0].url) == "https://example.test/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer sk-test-SECRET-123"
    assert set(json.loads(requests[0].content)) == {
        "model",
        "messages",
        "temperature",
        "max_tokens",
    }


def test_reasoning_effort_is_only_additional_body_key():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(model=None)

    settings = config(reasoning_effort="low")
    client = OpenAICompatClient(settings, transport=httpx.MockTransport(handler))
    result = client.generate([], temperature=0.2, max_tokens=50)

    assert set(bodies[0]) == {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "reasoning_effort",
    }
    assert bodies[0]["reasoning_effort"] == "low"
    assert result.model == settings.model


@pytest.mark.parametrize(
    ("headers", "expected_sleep"),
    [({"Retry-After": "3"}, 3.0), ({}, 2.0)],
)
def test_rate_limit_retries(headers, expected_sleep):
    sleeps = []
    client, _ = scripted_client(
        [httpx.Response(429, text="requests per minute", headers=headers), response()],
        sleeps=sleeps,
    )

    assert client.generate([]).text == "answer"
    assert sleeps == [expected_sleep]


def test_server_errors_back_off_and_recover():
    sleeps = []
    client, _ = scripted_client(
        [httpx.Response(500), httpx.Response(502), response()], sleeps=sleeps
    )

    assert client.generate([]).text == "answer"
    assert sleeps == [2.0, 4.0]


def test_server_error_exhaustion_reports_total_attempts():
    requests = 0
    sleeps = []

    def handler(request):
        nonlocal requests
        requests += 1
        return httpx.Response(500)

    client = OpenAICompatClient(
        config(max_retries=2),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    with pytest.raises(LLMServerError) as exc_info:
        client.generate([])

    assert requests == 3
    assert exc_info.value.attempts == 3
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried(status):
    sleeps = []
    client, queue = scripted_client([httpx.Response(status), response()], sleeps=sleeps)

    with pytest.raises(LLMAuthError) as exc_info:
        client.generate([])

    assert exc_info.value.attempts == 1
    assert len(queue) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("status", "body", "error_type"),
    [(404, "", LLMModelError), (400, "unknown model", LLMModelError),
     (400, "bad temperature", LLMError)],
)
def test_nonretryable_http_errors(status, body, error_type):
    sleeps = []
    client, queue = scripted_client(
        [httpx.Response(status, text=body), response()], sleeps=sleeps
    )

    with pytest.raises(error_type) as exc_info:
        client.generate([])

    if error_type is LLMError:
        assert type(exc_info.value) is LLMError
    assert exc_info.value.attempts == 1
    assert len(queue) == 1
    assert sleeps == []


def test_daily_provider_quota_is_not_retried():
    sleeps = []
    client, queue = scripted_client(
        [httpx.Response(429, text="tokens per day (TPD) exceeded"), response()],
        sleeps=sleeps,
    )

    with pytest.raises(LLMQuotaExhaustedError):
        client.generate([])

    assert len(queue) == 1
    assert sleeps == []


def test_transport_timeout_is_retried():
    request = httpx.Request("POST", "https://example.test")
    client, _ = scripted_client([httpx.ReadTimeout("timeout", request=request), response()])

    assert client.generate([]).text == "answer"


@pytest.mark.parametrize(
    "bad_response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"model": "x"}),
        response(text="   "),
    ],
)
def test_bad_responses_are_retried_then_raise(bad_response):
    sleeps = []
    client, _ = scripted_client([bad_response, bad_response, bad_response], sleeps=sleeps)

    with pytest.raises(LLMResponseError) as exc_info:
        client.generate([])

    assert exc_info.value.attempts == 3
    assert sleeps == [2.0, 4.0]


def test_empty_length_response_is_nonretryable_truncation():
    sleeps = []
    client, queue = scripted_client(
        [response(text="", finish_reason="length"), response()], sleeps=sleeps
    )

    with pytest.raises(LLMOutputTruncatedError, match="raise max_tokens") as exc_info:
        client.generate([])

    assert exc_info.value.attempts == 1
    assert len(queue) == 1
    assert sleeps == []


def test_truncated_response_usage_is_recorded_before_error():
    budget = TokenBudget(1000)
    client, _ = scripted_client(
        [
            response(
                text="",
                finish_reason="length",
                usage={"prompt_tokens": 10, "completion_tokens": 90},
            )
        ],
        budget=budget,
    )

    with pytest.raises(LLMOutputTruncatedError):
        client.generate([])

    assert budget.used == 100


def test_each_empty_retry_records_its_usage_once():
    budget = TokenBudget(1000)
    empty = response(
        text="",
        finish_reason="stop",
        usage={"prompt_tokens": 5, "completion_tokens": 5},
    )
    client, _ = scripted_client([empty, empty, empty], budget=budget)

    with pytest.raises(LLMResponseError):
        client.generate([])

    assert budget.used == 30


def test_success_usage_is_recorded_exactly_once():
    budget = TokenBudget(100)
    client, _ = scripted_client(
        [response(usage={"prompt_tokens": 2, "completion_tokens": 3})],
        budget=budget,
    )

    client.generate([])

    assert budget.used == 5


@pytest.mark.parametrize(
    "invalid_response",
    [httpx.Response(500), httpx.Response(200, json={"choices": []})],
)
def test_non_200_and_malformed_schema_record_no_usage(invalid_response):
    budget = TokenBudget(100)
    client, _ = scripted_client(
        [invalid_response], settings=config(max_retries=0), budget=budget
    )

    with pytest.raises(LLMError):
        client.generate([])

    assert budget.used == 0


def test_non_integer_usage_is_ignored_and_estimated():
    budget = TokenBudget(100)
    client, _ = scripted_client(
        [response(text="12345678", usage={"prompt_tokens": "7", "completion_tokens": "7"})],
        budget=budget,
    )

    result = client.generate([{"role": "user", "content": "12345678"}])

    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert budget.used == 4


@pytest.mark.parametrize("response_model", [123, ""])
def test_invalid_response_model_falls_back_to_config(response_model):
    client, _ = scripted_client([response(model=response_model)])

    assert client.generate([]).model == "test-model"


def test_budget_blocks_request_at_limit():
    requests = 0

    def handler(request):
        nonlocal requests
        requests += 1
        return response()

    budget = TokenBudget(100)
    budget.record(100)
    client = OpenAICompatClient(
        config(), transport=httpx.MockTransport(handler), budget=budget
    )

    with pytest.raises(LLMQuotaExhaustedError):
        client.generate([])
    assert requests == 0


def test_budget_resets_when_utc_date_changes_and_none_is_unlimited():
    day = [datetime.date(2026, 1, 1)]
    budget = TokenBudget(1, clock=lambda: day[0])
    budget.record(1)
    with pytest.raises(LLMQuotaExhaustedError):
        budget.check()
    day[0] = datetime.date(2026, 1, 2)
    budget.check()
    assert budget.used == 0

    unlimited = TokenBudget(None)
    unlimited.record(10**12)
    unlimited.check()


def test_usage_absent_is_estimated_from_content_characters():
    budget = TokenBudget(100)
    client, _ = scripted_client([response(text="12345678", usage=None)], budget=budget)

    client.generate([{"role": "user", "content": "12345678"}])

    assert budget.used == 4


def test_failed_call_records_only_later_success_usage():
    budget = TokenBudget(100)
    client, _ = scripted_client(
        [httpx.Response(500), response(usage={"prompt_tokens": 2, "completion_tokens": 3})],
        budget=budget,
    )

    client.generate([])

    assert budget.used == 5


def test_secrets_never_appear_in_config_errors_or_logs(caplog):
    settings = config()
    client, _ = scripted_client([httpx.Response(401)], settings=settings)

    with pytest.raises(LLMAuthError) as exc_info:
        client.generate([{"role": "user", "content": "private"}])

    rendered = " ".join(
        [str(settings), repr(settings), str(settings.model_dump()),
         str(exc_info.value), repr(exc_info.value), caplog.text]
    )
    assert "sk-test-SECRET-123" not in rendered


def test_config_loads_environment_and_defaults():
    settings = load_llm_config(
        env={
            "ATHENA_LLM_BASE_URL": "https://example.test/v1",
            "ATHENA_LLM_MODEL": "model",
            "ATHENA_LLM_API_KEY": "secret",
            "ATHENA_LLM_REASONING_EFFORT": "",
        }
    )

    assert settings.timeout == 60.0
    assert settings.max_retries == 5
    assert settings.reasoning_effort is None


def test_dotenv_parsing_and_environment_precedence(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        """# ignored
export ATHENA_LLM_BASE_URL='https://file.test/v1'
ATHENA_LLM_MODEL="file-model"
ATHENA_LLM_API_KEY=file-secret
ATHENA_LLM_TIMEOUT=12.5
ATHENA_LLM_MAX_RETRIES=4
ATHENA_LLM_DAILY_TOKEN_LIMIT=99
ATHENA_LLM_REASONING_EFFORT=low
""",
        encoding="utf-8",
    )

    settings = load_llm_config(
        env={"ATHENA_LLM_MODEL": "environment-model"}, dotenv_path=dotenv
    )

    assert settings.base_url == "https://file.test/v1"
    assert settings.model == "environment-model"
    assert settings.api_key.get_secret_value() == "file-secret"
    assert settings.timeout == 12.5
    assert settings.max_retries == 4
    assert settings.daily_token_limit == 99
    assert settings.reasoning_effort == "low"


def test_dotenv_with_utf8_bom_loads_first_key(tmp_path):
    dotenv = tmp_path / ".env"
    content = (
        "ATHENA_LLM_BASE_URL=https://bom.test/v1\n"
        "ATHENA_LLM_MODEL=model\n"
        "ATHENA_LLM_API_KEY=secret\n"
    ).encode("utf-8")
    dotenv.write_bytes(b"\xef\xbb\xbf" + content)

    settings = load_llm_config(env={}, dotenv_path=dotenv)

    assert settings.base_url == "https://bom.test/v1"


def test_utf16_dotenv_raises_safe_config_error(tmp_path):
    dotenv = tmp_path / ".env"
    secret_value = "VALUE-MUST-NOT-APPEAR"
    dotenv.write_text(
        f"ATHENA_LLM_API_KEY={secret_value}\n",
        encoding="utf-16",
    )

    with pytest.raises(LLMConfigError) as exc_info:
        load_llm_config(env={}, dotenv_path=dotenv)

    assert "UTF-8" in str(exc_info.value)
    assert secret_value not in str(exc_info.value)


@pytest.mark.parametrize("timeout", ["nan", "inf"])
def test_non_finite_timeout_is_rejected(timeout):
    env = {
        "ATHENA_LLM_BASE_URL": "url",
        "ATHENA_LLM_MODEL": "model",
        "ATHENA_LLM_API_KEY": "key",
        "ATHENA_LLM_TIMEOUT": timeout,
    }

    with pytest.raises(LLMConfigError) as exc_info:
        load_llm_config(env=env)

    assert "ATHENA_LLM_TIMEOUT" in str(exc_info.value)


@pytest.mark.parametrize(
    ("env", "variable"),
    [
        ({"ATHENA_LLM_BASE_URL": "url", "ATHENA_LLM_MODEL": "model"},
         "ATHENA_LLM_API_KEY"),
        ({"ATHENA_LLM_BASE_URL": "url", "ATHENA_LLM_MODEL": "model",
          "ATHENA_LLM_API_KEY": "key", "ATHENA_LLM_TIMEOUT": "slow"},
         "ATHENA_LLM_TIMEOUT"),
        ({"ATHENA_LLM_BASE_URL": "url", "ATHENA_LLM_MODEL": "model",
          "ATHENA_LLM_API_KEY": "key", "ATHENA_LLM_MAX_RETRIES": "-1"},
         "ATHENA_LLM_MAX_RETRIES"),
    ],
)
def test_invalid_config_names_variable_without_value(env, variable):
    with pytest.raises(LLMConfigError) as exc_info:
        load_llm_config(env=env)

    assert variable in str(exc_info.value)
    assert "slow" not in str(exc_info.value)


def test_stub_returns_raises_records_and_exhausts():
    scripted = LLMError("scripted")
    stub = StubLLMClient(["first", scripted])
    messages = [{"role": "user", "content": "hello"}]

    assert stub.generate(messages, temperature=0.1, max_tokens=9).text == "first"
    with pytest.raises(LLMError, match="scripted"):
        stub.generate([])
    with pytest.raises(LLMError, match="stub exhausted"):
        stub.generate([])
    assert stub.calls[0] == {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 9,
    }
    assert len(stub.calls) == 3


def use_client(client: LLMClient) -> str:
    return client.generate([]).text


def test_protocol_conformance_for_both_clients():
    stub = StubLLMClient(["stub"])
    real, _ = scripted_client([response(text="real")])

    assert use_client(stub) == "stub"
    assert use_client(real) == "real"


def test_repository_environment_hygiene():
    root = Path(__file__).resolve().parents[1]
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    example = (root / ".env.example").read_text(encoding="utf-8")
    assert example
    for line in example.splitlines():
        if line.startswith("ATHENA_LLM_API_KEY="):
            assert line == "ATHENA_LLM_API_KEY="
