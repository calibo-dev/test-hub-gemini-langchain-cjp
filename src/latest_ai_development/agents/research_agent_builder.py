from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.prompts.prompt_builder import build_research_system_prompt
from latest_ai_development.tools.tool_registry import get_tools

logger = logging.getLogger(__name__)


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
    primary_model = model_config.get("primary", {}) if isinstance(model_config, dict) else {}
    fallback_model = model_config.get("fallback") if isinstance(model_config, dict) else None

    logger.info(
        "Trying primary model | stage=research | provider=%s | modelId=%s",
        primary_model.get("provider"),
        primary_model.get("modelId"),
    )
    if isinstance(fallback_model, dict):
        logger.info(
            "Trying fallback model | stage=research | provider=%s | modelId=%s",
            fallback_model.get("provider"),
            fallback_model.get("modelId"),
        )

    llm_candidates = get_llm_candidates(model_config)

    agents = [
        create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            name="research_agent" if index == 0 else f"research_agent_fallback_{index}",
        )
        for index, llm in enumerate(llm_candidates)
    ]

    if len(agents) == 1:
        return agents[0]

    return agents[0].with_fallbacks(agents[1:])


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
