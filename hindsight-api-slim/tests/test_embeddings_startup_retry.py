"""The startup dimension probe rides out a transient 429 instead of killing the API (#6709).

``MemoryEngine.initialize()`` runs the embedder's ``initialize()`` inside the lifespan
TaskGroup; an ``openai.RateLimitError`` from the dimension-detection probe used to escape
it and exit the process with "Application startup failed". The probe now retries transient
failures with the same bounded policy the knowledge embedder uses for ingest; permanent
errors still fail fast, and a known model or explicit dimensions still skip the probe.
"""

import asyncio
from typing import Any

import httpx
import pytest
from openai import AuthenticationError, RateLimitError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from hindsight_api.engine.embeddings import OpenAIEmbeddings


def _rate_limit_error() -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))
    return RateLimitError(
        "Error code: 429", response=response, body={"code": "rate_limit_exceeded", "message": "endpoint busy"}
    )


def _auth_error() -> AuthenticationError:
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))
    return AuthenticationError("Error code: 401", response=response, body={"code": "invalid_api_key"})


def _ok_response(dim: int = 768) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse.model_construct(
        data=[Embedding(embedding=[0.1] * dim, index=0, object="embedding")],
        model="custom-embed",
        object="list",
        usage=Usage.model_construct(prompt_tokens=1, total_tokens=1),
    )


class _ScriptedEmbeddingsAPI:
    """Raises/returns each scripted step in order; repeats the last step once exhausted."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0

    async def create(self, **_: Any) -> CreateEmbeddingResponse:
        self.calls += 1
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return step


class _ScriptedClient:
    def __init__(self, api: _ScriptedEmbeddingsAPI) -> None:
        self.api_key = "sk-test"
        self.embeddings = api


@pytest.fixture
def no_sleep(monkeypatch) -> list[float]:
    """Record backoff sleeps instead of taking them; the retry logic is what's under test."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture
def install_api(monkeypatch):
    """Route initialize()'s AsyncOpenAI construction to a scripted fake client."""

    def install(api: _ScriptedEmbeddingsAPI) -> None:
        monkeypatch.setattr("openai.AsyncOpenAI", lambda **_: _ScriptedClient(api))

    return install


@pytest.mark.asyncio
async def test_probe_retries_transient_429_then_detects_dimension(no_sleep, install_api):
    api = _ScriptedEmbeddingsAPI([_rate_limit_error(), _rate_limit_error(), _ok_response(768)])
    install_api(api)
    # An unknown model forces the dimension-detection probe (the startup call from #6709).
    backend = OpenAIEmbeddings(api_key="sk-test", model="custom-embed")

    await backend.initialize()

    assert backend.dimension == 768
    assert api.calls == 3
    assert len(no_sleep) == 2  # backed off between attempts instead of exiting


@pytest.mark.asyncio
async def test_probe_gives_up_after_bounded_attempts(no_sleep, install_api):
    api = _ScriptedEmbeddingsAPI([_rate_limit_error()])
    install_api(api)
    backend = OpenAIEmbeddings(api_key="sk-test", model="custom-embed")

    # A still-rate-limited endpoint after the bounded budget must still surface the
    # error (startup fails loudly), not hang or silently continue undimensioned.
    with pytest.raises(RateLimitError):
        await backend.initialize()

    assert api.calls == 6  # RetryPolicy(max_retries=5) -> 6 attempts, no more


@pytest.mark.asyncio
async def test_probe_auth_error_fails_fast(no_sleep, install_api):
    api = _ScriptedEmbeddingsAPI([_auth_error()])
    install_api(api)
    backend = OpenAIEmbeddings(api_key="sk-test", model="custom-embed")

    with pytest.raises(AuthenticationError):
        await backend.initialize()

    assert api.calls == 1
    assert no_sleep == []


@pytest.mark.asyncio
async def test_known_model_skips_the_probe(no_sleep, install_api):
    api = _ScriptedEmbeddingsAPI([_ok_response()])
    install_api(api)
    backend = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small")

    await backend.initialize()

    assert backend.dimension == 1536
    assert api.calls == 0
    assert no_sleep == []


@pytest.mark.asyncio
async def test_explicit_dimensions_skip_the_probe(no_sleep, install_api):
    api = _ScriptedEmbeddingsAPI([_ok_response()])
    install_api(api)
    backend = OpenAIEmbeddings(api_key="sk-test", model="custom-embed", dimensions=1024)

    await backend.initialize()

    assert backend.dimension == 1024
    assert api.calls == 0
    assert no_sleep == []
