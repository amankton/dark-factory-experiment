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


class TestGeminiToolCallFinishReason:
    """Gemini ends tool-call rounds with finish_reason="stop", not "tool_calls"."""

    async def _run(self, provider: str) -> tuple[AsyncMock, list[str]]:
        from backend.llm.openrouter import stream_chat

        def tool_call(**kw: Any) -> SimpleNamespace:
            # Mirrors Gemini's stream: index=None and a thought signature.
            return SimpleNamespace(
                index=None,
                id=kw.get("id"),
                type="function" if kw.get("id") else None,
                function=SimpleNamespace(name=kw.get("name"), arguments=kw.get("arguments")),
                extra_content=kw.get("extra_content"),
            )

        def chunk(**kw: Any) -> SimpleNamespace:
            delta = SimpleNamespace(content=kw.get("content"), tool_calls=kw.get("tool_calls"))
            choice = SimpleNamespace(delta=delta, finish_reason=kw.get("finish_reason"))
            return SimpleNamespace(choices=[choice])

        class _Stream:
            def __init__(self, chunks: list[SimpleNamespace]) -> None:
                self._chunks = chunks

            def __aiter__(self):
                return self._gen()

            async def _gen(self):
                for c in self._chunks:
                    yield c

        round1 = _Stream(
            [
                chunk(
                    tool_calls=[
                        tool_call(
                            id="c1",
                            name="search_videos",
                            arguments='{"query":"x"}',
                            extra_content={"google": {"thought_signature": "sig-1"}},
                        ),
                        tool_call(
                            id="c2",
                            name="get_video_transcript",
                            arguments='{"video_id":"v"}',
                            extra_content={"google": {"thought_signature": "sig-2"}},
                        ),
                    ]
                ),
                chunk(finish_reason="stop"),
            ]
        )
        round2 = _Stream([chunk(content="answer"), chunk(finish_reason="stop")])
        create = AsyncMock(side_effect=[round1, round2])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        executed: list[str] = []

        async def exec_tool(name: str, raw_args: str) -> str:
            executed.append(name)
            return "result"

        with (
            patch("backend.llm.openrouter._get_async_client", return_value=fake_client),
            patch(
                "backend.llm.openrouter.build_system_prompt",
                new=AsyncMock(return_value=[{"type": "text", "text": "sys"}]),
            ),
            patch("backend.llm.openrouter.LLM_PROVIDER", provider),
        ):
            async for _ in stream_chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "search_videos"}}],
                tool_executor=exec_tool,
                max_tool_calls=3,
            ):
                pass
        return create, executed

    async def test_gemini_executes_tools_on_stop(self) -> None:
        create, executed = await self._run("gemini")
        assert executed == ["search_videos", "get_video_transcript"]
        assert create.call_count == 2

    async def test_gemini_thought_signatures_echoed_back(self) -> None:
        create, _ = await self._run("gemini")
        messages = create.call_args_list[1].kwargs["messages"]
        assistant = next(m for m in messages if m["role"] == "assistant")
        signatures = [
            tc["extra_content"]["google"]["thought_signature"] for tc in assistant["tool_calls"]
        ]
        assert signatures == ["sig-1", "sig-2"]

    async def test_openrouter_behaviour_unchanged(self) -> None:
        create, executed = await self._run("openrouter")
        assert executed == []
        assert create.call_count == 1


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
