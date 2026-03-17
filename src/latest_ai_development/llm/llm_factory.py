from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from latest_ai_development.config.settings import get_models_config, get_settings
from latest_ai_development.secrets_manager import resolve_openai_api_key


def get_llm(**overrides: Any):
    """
    Create an LLM instance based on configuration.

    Supports:
    - OpenAI
    - Ollama
    """

    settings = get_settings()
    models_config = get_models_config()

    provider = settings.provider.lower()

    providers = models_config.get("providers", {})
    provider_config = providers.get(provider)

    if not provider_config:
        raise ValueError(f"Provider '{provider}' not configured in models.yaml")

    model_name = overrides.get(
        "model_name",
        provider_config.get("chat_model"),
    )

    temperature = overrides.get(
        "temperature",
        provider_config.get("temperature", 0.7),
    )

    # OPENAI
    if provider == "openai":

        api_key = resolve_openai_api_key()

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY or OPENAI_API_KEY_SECRET."
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    # OLLAMA
    if provider == "ollama":

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported provider: {provider}")