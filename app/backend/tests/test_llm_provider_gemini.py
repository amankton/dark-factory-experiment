"""
Tests for LLM_PROVIDER=gemini request shaping.

Gemini is reached through its OpenAI-compatible endpoint, which differs from
OpenRouter in three ways these tests lock in:
  1. The system prompt is sent as a plain string (no Anthropic cache_control blocks).
  2. Reasoning effort is sent as top-level `reasoning_effort`, not OpenRouter's
     `extra_body={"reasoning": ...}`.
  3. Embedding requests pass `dimensions=1536` so Gemini's 3072-dim default
     is truncated to match the stored vectors.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from openai import omit


class _FakeStream:
    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        delta = SimpleNamespace(content="hello", tool_calls=None)
        yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop")])


async def _run_stream_chat(provider: str, effort: str = "") -> dict[str, Any]:
    from backend.llm.openrouter import stream_chat

    mock_create = AsyncMock(return_value=_FakeStream())
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    blocks = [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "catalog"},
    ]
    with (
        patch("backend.llm.openrouter._get_async_client", return_value=fake_client),
        patch("backend.llm.openrouter.build_system_prompt", new=AsyncMock(return_value=blocks)),
        patch("backend.llm.openrouter.LLM_PROVIDER", provider),
        patch("backend.llm.openrouter.LLM_REASONING_EFFORT", effort),
    ):
        async for _ in stream_chat(messages=[{"role": "user", "content": "hi"}]):
            pass
    kwargs: dict[str, Any] = mock_create.call_args.kwargs
    return kwargs


class TestGeminiChatRequest:
    async def test_system_prompt_flattened_to_string(self) -> None:
        kwargs = await _run_stream_chat("gemini")
        assert kwargs["messages"][0] == {"role": "system", "content": "sys\n\ncatalog"}

    async def test_openrouter_keeps_content_blocks(self) -> None:
        kwargs = await _run_stream_chat("openrouter")
        system_content = kwargs["messages"][0]["content"]
        assert isinstance(system_content, list)
        assert system_content[0]["cache_control"] == {"type": "ephemeral"}

    async def test_reasoning_effort_top_level_for_gemini(self) -> None:
        kwargs = await _run_stream_chat("gemini", effort="low")
        assert kwargs["reasoning_effort"] == "low"
        assert "extra_body" not in kwargs

    async def test_reasoning_effort_extra_body_for_openrouter(self) -> None:
        kwargs = await _run_stream_chat("openrouter", effort="low")
        assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}
        assert "reasoning_effort" not in kwargs


class TestGeminiEmbeddings:
    def _fake_client(self) -> MagicMock:
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
        )
        return client

    def test_gemini_requests_1536_dimensions(self) -> None:
        from backend.rag import embeddings

        client = self._fake_client()
        with (
            patch.object(embeddings, "_get_client", return_value=client),
            patch.object(embeddings, "LLM_PROVIDER", "gemini"),
        ):
            embeddings.embed_text("hello")
            embeddings.embed_batch(["hello"])
        for call in client.embeddings.create.call_args_list:
            assert call.kwargs["dimensions"] == 1536

    def test_openrouter_omits_dimensions(self) -> None:
        from backend.rag import embeddings

        client = self._fake_client()
        with (
            patch.object(embeddings, "_get_client", return_value=client),
            patch.object(embeddings, "LLM_PROVIDER", "openrouter"),
        ):
            embeddings.embed_text("hello")
        assert client.embeddings.create.call_args.kwargs["dimensions"] is omit
