"""LLM provider selection for clinical report generation.

Every report-generation call goes through `get_llm_client()` here. No other module
should import `langchain_groq`/`langchain_openai` directly or hardcode a base URL
or model name literal - that keeps the report LLM a swappable, honestly documented
dependency rather than one vendor baked into the code.

Supported providers, selected via the LLM_PROVIDER env var:
  - "groq" (default): Groq's free-tier hosted Llama 3.3 70B. Zero local setup;
    needs GROQ_API_KEY (env var, or a key pasted into the Gradio UI).
  - "ollama": a locally running Ollama server, reached over its OpenAI-compatible
    endpoint (default http://localhost:11434/v1, override with OLLAMA_BASE_URL).
    No API key needed. Trades model size (8B vs. 70B) for keeping data fully
    on-prem - the right story for a medical demo. Requires `ollama pull <model>`
    first (default model: qwen3:8b).
  - "openai_compatible": any other OpenAI-compatible chat-completions endpoint
    (e.g. vLLM), configured via LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.

Swapping providers changes report *prose* only - it has no effect on segmentation
Dice, which comes from the Swin+DAF network alone.
"""

from __future__ import annotations

import os

try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class ProviderError(RuntimeError):
    """The configured LLM provider is missing a key, endpoint, or model."""


def provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()


def get_llm_client(api_key: str = "", temperature: float = 0.2):
    """Return a LangChain chat client for the provider selected via LLM_PROVIDER.

    `api_key` is only consulted by the "groq" provider (e.g. a key pasted into the
    Gradio UI, falling back to GROQ_API_KEY). Raises ProviderError with a
    user-facing message if the selected provider is misconfigured.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ProviderError("Run: pip install langchain-groq langchain-openai langchain-core")

    provider = provider_name()

    if provider == "groq":
        key = api_key.strip() or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ProviderError(
                "No Groq API key provided. Get a free key at console.groq.com and enter it above."
            )
        model = os.environ.get("LLM_MODEL", DEFAULT_GROQ_MODEL)
        return ChatGroq(model=model, temperature=temperature, api_key=key)

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        model = os.environ.get("LLM_MODEL", DEFAULT_OLLAMA_MODEL)
        # Ollama's OpenAI-compatible endpoint ignores the API key but the client
        # requires a non-empty string.
        return ChatOpenAI(model=model, temperature=temperature, base_url=base_url, api_key="ollama")

    if provider == "openai_compatible":
        base_url = os.environ.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "")
        if not base_url or not model:
            raise ProviderError(
                "LLM_PROVIDER=openai_compatible requires LLM_BASE_URL and LLM_MODEL to be set."
            )
        key = os.environ.get("LLM_API_KEY", "") or "not-needed"
        return ChatOpenAI(model=model, temperature=temperature, base_url=base_url, api_key=key)

    raise ProviderError(
        f"Unknown LLM_PROVIDER '{provider}'. Use one of: groq, ollama, openai_compatible."
    )
