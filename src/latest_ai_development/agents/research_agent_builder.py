from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_fallback_llms, get_llm
from latest_ai_development.llm.model_attempt_logging import (
    LoggedModelFallbackMiddleware,
    ModelAttemptLoggingMiddleware,
    get_model_attempts,
)
from latest_ai_development.prompts.prompt_builder import build_research_system_prompt
from latest_ai_development.tools.tool_registry import get_tools


def build_research_agent():
    """
    Build the research stage agent loop.

    The agent lets the model decide when to call registered tools and stops
    when the model produces a final research note response.
    """
    stages_config = get_stages_config()
    stage_cfg = stages_config.get("research")

    if not stage_cfg:
        raise ValueError("Missing 'research' configuration in stages.yaml")

    tools = get_tools() if stage_cfg.get("use_tools", True) else []
    system_prompt = build_research_system_prompt(stage_cfg)
    model_config = stage_cfg.get("model")
    primary_attempt, fallback_attempts = get_model_attempts("research", model_config)
    primary_llm = get_llm(model_config, section="primary")
    fallback_llms = get_fallback_llms(model_config)

    middleware = []
    if fallback_llms:
        middleware.append(
            LoggedModelFallbackMiddleware(
                primary_attempt,
                fallback_attempts,
                fallback_llms[0],
                *fallback_llms[1:],
            )
        )
    else:
        middleware.append(ModelAttemptLoggingMiddleware(primary_attempt))

    return create_agent(
        model=primary_llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        name="research_agent",
    )


def extract_final_message_text(result: dict[str, Any]) -> str:
    """
    Extract plain text from a LangChain agent result.

    Supports string message content and provider-standard content blocks.
    """
    messages = result.get("messages") or []
    if not messages:
        return ""

    message = messages[-1]
    content_blocks = getattr(message, "content_blocks", None)
    if content_blocks:
        return _content_blocks_to_text(content_blocks)

    content = getattr(message, "content", message)
    return _content_to_text(content)


def _content_blocks_to_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            text = block.get("text") or block.get("content")
            if text:
                parts.append(str(text))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts).strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return _content_blocks_to_text(content)

    if content is None:
        return ""

    return str(content)
