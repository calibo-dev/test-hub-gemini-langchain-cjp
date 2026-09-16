from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.llm.model_attempt_logging import (
    get_model_attempts,
    with_model_attempt_logging,
)
from latest_ai_development.prompts.prompt_builder import build_validator_prompt


def build_validator_chain():
    stages_cfg = get_stages_config()
    stage_cfg = stages_cfg.get("validator_agent")

    if not stage_cfg:
        raise ValueError("Missing 'validator_agent' configuration in stages.yaml")

    parser = JsonOutputParser(
        schema=[
            {
                "name": "validated_query",
                "description": "Validated SQL SELECT statement safe for execution.",
                "type": "string",
            },
            {
                "name": "out_of_scope",
                "description": "True when the SQL query is invalid or out of scope.",
                "type": "boolean",
            },
        ]
    )

    format_instructions = parser.get_format_instructions()
    prompt = build_validator_prompt(stage_cfg, format_instructions)

    model_config = stage_cfg.get("model")
    primary_attempt, fallback_attempts = get_model_attempts("validator_agent", model_config)
    attempts = [primary_attempt, *fallback_attempts]
    chains = [
        prompt
        | with_model_attempt_logging(llm, attempts[index], retry=index > 0)
        | parser
        for index, llm in enumerate(get_llm_candidates(model_config))
    ]

    if len(chains) == 1:
        return chains[0]

    return chains[0].with_fallbacks(chains[1:])
