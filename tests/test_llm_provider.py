"""Unit tests for src/llm/provider.py - all mocked, no network/GPU."""

from __future__ import annotations

import pytest

from llm import provider as llm_provider


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_provider.provider_name() == "groq"


def test_groq_missing_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm_provider.ProviderError, match="Groq API key"):
        llm_provider.get_llm_client(api_key="")


def test_groq_uses_key_and_default_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(llm_provider, "ChatGroq", _FakeClient)

    client = llm_provider.get_llm_client(api_key="gsk_fake")

    assert isinstance(client, _FakeClient)
    assert client.kwargs["model"] == llm_provider.DEFAULT_GROQ_MODEL
    assert client.kwargs["api_key"] == "gsk_fake"


def test_groq_falls_back_to_env_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    monkeypatch.setattr(llm_provider, "ChatGroq", _FakeClient)

    client = llm_provider.get_llm_client(api_key="")

    assert client.kwargs["api_key"] == "gsk_env"


def test_ollama_needs_no_key_and_uses_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(llm_provider, "ChatOpenAI", _FakeClient)

    client = llm_provider.get_llm_client()

    assert isinstance(client, _FakeClient)
    assert client.kwargs["model"] == llm_provider.DEFAULT_OLLAMA_MODEL
    assert client.kwargs["base_url"] == llm_provider.DEFAULT_OLLAMA_BASE_URL


def test_ollama_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    monkeypatch.setattr(llm_provider, "ChatOpenAI", _FakeClient)

    client = llm_provider.get_llm_client()

    assert client.kwargs["model"] == "qwen3:8b"
    assert client.kwargs["base_url"] == "http://gpu-box:11434/v1"


def test_openai_compatible_requires_base_url_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(llm_provider.ProviderError, match="LLM_BASE_URL"):
        llm_provider.get_llm_client()


def test_openai_compatible_builds_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm-host:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-fake")
    monkeypatch.setattr(llm_provider, "ChatOpenAI", _FakeClient)

    client = llm_provider.get_llm_client()

    assert client.kwargs["base_url"] == "http://vllm-host:8000/v1"
    assert client.kwargs["model"] == "some-model"
    assert client.kwargs["api_key"] == "sk-fake"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(llm_provider.ProviderError, match="Unknown LLM_PROVIDER"):
        llm_provider.get_llm_client()
