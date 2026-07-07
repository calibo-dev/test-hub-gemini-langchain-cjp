from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from latest_ai_development.config.settings import (
    get_models_config,
    get_settings,
    normalize_provider_name,
)

load_dotenv()


def resolve_api_key(
    secret_name: str,
) -> str:
    """
    Resolve a provider API key from its provider-specific secret name.
    """
    secret_name = secret_name.strip()

    if not secret_name:
        return ""

    from latest_ai_development.secrets_manager import SecretsManager

    return (SecretsManager().get_secret(secret_name) or "").strip()


def get_llm(**overrides: Any):
    """
    Create an LLM instance based on configuration.

    Supports:
    - OpenAI
    - Anthropic
    - Gemini
    - Ollama
    """

    settings = get_settings()
    models_config = get_models_config()

    provider = normalize_provider_name(settings.provider)

    providers = models_config.get("providers", {})
    provider_config = providers.get(provider)

    if not provider_config:
        raise ValueError(f"Provider '{settings.provider}' not configured in models.yaml")

    model_name = overrides.get(
        "model_name",
        provider_config.get("chat_model"),
    )

    temperature = overrides.get(
        "temperature",
        provider_config.get("temperature", 0.7),
    )

    max_tokens = overrides.get(
        "max_tokens",
        provider_config.get("max_tokens"),
    )

    # OPENAI
    if provider == "openai":

        api_key = resolve_api_key(settings.openai_api_key_secret)

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY_SECRET."
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    # ANTHROPIC
    if provider == "anthropicai":

        api_key = resolve_api_key(settings.anthropicai_api_key_secret)

        if not api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPICAI_API_KEY_SECRET."
            )

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the 'langchain-anthropic' package. "
                "Run `uv sync` to install template dependencies."
            ) from exc

        model_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key,
        }
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        return ChatAnthropic(**model_kwargs)

    # GEMINI
    if provider == "geminiai":

        api_key = resolve_api_key(settings.geminiai_api_key_secret)

        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINIAI_API_KEY_SECRET."
            )

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Gemini provider requires the 'langchain-google-genai' package. "
                "Run `uv sync` to install template dependencies."
            ) from exc

        model_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key,
        }
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        return ChatGoogleGenerativeAI(**model_kwargs)

    # OLLAMA
    if provider == "ollama":

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported provider: {provider}")
