from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.llm.model_attempt_logging import (
    get_model_attempts,
    with_model_attempt_logging,
)
from latest_ai_development.prompts.prompt_builder import build_research_prompt
from latest_ai_development.tools.tool_registry import get_tools


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

    llm_candidates = get_llm_candidates(model_config)

    # Enable tool calling only if configured
    if use_tools and tools:
        llm_candidates = [llm.bind_tools(tools) for llm in llm_candidates]

    # Output parser
    parser = StrOutputParser()
    primary_attempt, fallback_attempts = get_model_attempts("research", model_config)
    attempts = [primary_attempt, *fallback_attempts]
    chains = [
        prompt | with_model_attempt_logging(llm, attempts[index], retry=index > 0) | parser
        for index, llm in enumerate(llm_candidates)
    ]

    if len(chains) == 1:
        return chains[0]

    return chains[0].with_fallbacks(chains[1:])
