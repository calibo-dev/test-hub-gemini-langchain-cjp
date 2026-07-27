from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.prompts.prompt_builder import build_research_prompt
from latest_ai_development.tools.tool_registry import get_tools

logger = logging.getLogger(__name__)


def build_research_chain():
    """
    Build the research stage LangChain pipeline.

    Pipeline:
        Prompt → LLM (optional tool calling) → Output Parser
    """

    # Load stage configuration
    stages_config = get_stages_config()
    stage_cfg = stages_config.get("research")

    if not stage_cfg:
        raise ValueError("Missing 'research' configuration in stages.yaml")

    # Build prompt using hybrid prompt builder
    prompt = build_research_prompt(stage_cfg)

    # Stage configuration flags
    use_tools = stage_cfg.get("use_tools", True)
    # Load tools
    tools = get_tools()
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

    # Enable tool calling only if configured
    if use_tools and tools:
        llm_candidates = [llm.bind_tools(tools) for llm in llm_candidates]

    # Output parser
    parser = StrOutputParser()
    chains = [
        prompt | llm | parser
        for llm in llm_candidates
    ]

    if len(chains) == 1:
        return chains[0]

    return chains[0].with_fallbacks(chains[1:])
