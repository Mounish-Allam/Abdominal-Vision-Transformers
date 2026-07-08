"""Unit tests for src/report_generator.py - all mocked, no network/GPU/real index."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import report_generator


def _stats():
    return {
        "total": 224 * 224,
        0: {"pixels": 30000, "pct": 60.0},
        1: {"pixels": 10000, "pct": 20.0},
        2: {"pixels": 3000, "pct": 6.0},
        3: {"pixels": 3000, "pct": 6.0},
        4: {"pixels": 4000, "pct": 8.0},
    }


class _FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages):
        return SimpleNamespace(content="Canned report text.")


def test_generate_report_no_api_key_returns_message(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    report, passages_md = report_generator.generate_report(_stats(), api_key="")
    assert "API key" in report
    assert passages_md == ""


def test_generate_report_use_rag_false_skips_retrieval(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(report_generator, "ChatGroq", _FakeLLM)

    def _boom():
        raise AssertionError("_get_vectorstore should not be called when use_rag=False")

    monkeypatch.setattr(report_generator, "_get_vectorstore", _boom)

    report, passages_md = report_generator.generate_report(_stats(), use_rag=False)

    assert report == "Canned report text."
    assert passages_md == ""


def test_generate_report_falls_back_when_index_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(report_generator, "ChatGroq", _FakeLLM)

    def _raise_not_found():
        raise FileNotFoundError("no index")

    monkeypatch.setattr(report_generator, "_get_vectorstore", _raise_not_found)

    report, passages_md = report_generator.generate_report(_stats(), use_rag=True)

    assert report == "Canned report text."
    assert "RAG unavailable" in passages_md
    assert "build_index.py" in passages_md


def test_prompts_request_findings_and_impression_sections():
    for system_prompt in (report_generator._GROUNDED_SYSTEM, report_generator._LEGACY_SYSTEM):
        assert "Findings:" in system_prompt
        assert "Impression:" in system_prompt
