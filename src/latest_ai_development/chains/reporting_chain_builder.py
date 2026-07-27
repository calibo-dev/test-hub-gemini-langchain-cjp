from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.llm.model_attempt_logging import (
    get_model_attempts,
    with_model_attempt_logging,
)
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt


def build_reporting_chain():
    """
    Build the reporting stage LangChain pipeline.

    Pipeline:
        Prompt → LLM → Output Parser
    """

    # Load stage configuration
    stages_cfg = get_stages_config()
    stage_cfg = stages_cfg.get("reporting")

    if not stage_cfg:
        raise ValueError("Missing 'reporting' configuration in stages.yaml")

    # Build prompt using hybrid prompt builder
    prompt = build_reporting_prompt(stage_cfg)

    model_config = stage_cfg.get("model")
    primary_attempt, fallback_attempts = get_model_attempts("reporting", model_config)
    attempts = [primary_attempt, *fallback_attempts]
    parser = StrOutputParser()
    chains = [
        prompt | with_model_attempt_logging(llm, attempts[index], retry=index > 0) | parser
        for index, llm in enumerate(get_llm_candidates(model_config))
    ]

    if len(chains) == 1:
        return chains[0]

    return chains[0].with_fallbacks(chains[1:])
