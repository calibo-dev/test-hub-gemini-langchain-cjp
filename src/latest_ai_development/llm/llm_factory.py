from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from latest_ai_development.config.settings import get_models_config, get_settings
from dotenv import load_dotenv

load_dotenv()

def resolve_openai_api_key() -> str:
    """
    Resolve the OpenAI API key without changing the shared secrets manager file.

    Priority:
    1. OPENAI_API_KEY
    2. API_KEY_SECRET
    3. OPENAI_API_KEY_SECRET
    """
    direct_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    print(f"Direct API Key: {direct_api_key}")  # Debugging line

    if direct_api_key:
        print("Using direct API key")
        return direct_api_key

    secret_name = (
        os.getenv("API_KEY_SECRET", "").strip()
        or os.getenv("OPENAI_API_KEY_SECRET", "").strip()
    )

    print(f"Resolved Secret Name: {secret_name}")  # Debugging line
    
    if not secret_name:
        print(f"Secret Name is empty")  # Debug
        return ""
    
    print(f"Secret Name: {secret_name}")  # Debugging line

    from latest_ai_development.secrets_manager import SecretsManager

    return (SecretsManager().get_secret(secret_name) or "").strip()


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
                "Set OPENAI_API_KEY, API_KEY_SECRET, or OPENAI_API_KEY_SECRET."
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
