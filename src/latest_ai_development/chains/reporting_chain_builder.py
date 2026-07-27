from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt

logger = logging.getLogger(__name__)


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
    primary_model = model_config.get("primary", {}) if isinstance(model_config, dict) else {}
    fallback_model = model_config.get("fallback") if isinstance(model_config, dict) else None

    logger.info(
        "Trying primary model | stage=reporting | provider=%s | modelId=%s",
        primary_model.get("provider"),
        primary_model.get("modelId"),
    )
    if isinstance(fallback_model, dict):
        logger.info(
            "Trying fallback model | stage=reporting | provider=%s | modelId=%s",
            fallback_model.get("provider"),
            fallback_model.get("modelId"),
        )

    parser = StrOutputParser()
    chains = [
        prompt | llm | parser
        for llm in get_llm_candidates(model_config)
    ]

    if len(chains) == 1:
        return chains[0]

    return chains[0].with_fallbacks(chains[1:])
