from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.llm.model_attempt_logging import (
    get_model_attempts,
    with_model_attempt_logging,
)
from latest_ai_development.prompts.prompt_builder import build_response_generation_prompt


def build_response_generation_chain():
    stages_cfg = get_stages_config()
    stage_cfg = stages_cfg.get("response_generation_agent")

    if not stage_cfg:
        raise ValueError("Missing 'response_generation_agent' configuration in stages.yaml")

    prompt = build_response_generation_prompt(stage_cfg)

    model_config = stage_cfg.get("model")
    primary_attempt, fallback_attempts = get_model_attempts("response_generation_agent", model_config)
    attempts = [primary_attempt, *fallback_attempts]
    parser = StrOutputParser()
    chains = [
        prompt
        | with_model_attempt_logging(llm, attempts[index], retry=index > 0)
        | parser
        for index, llm in enumerate(get_llm_candidates(model_config))
    ]

    if len(chains) == 1:
        return chains[0]

    return chains[0].with_fallbacks(chains[1:])
