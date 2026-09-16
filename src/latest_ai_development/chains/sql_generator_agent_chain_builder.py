from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm_candidates
from latest_ai_development.llm.model_attempt_logging import (
    get_model_attempts,
    with_model_attempt_logging,
)
from latest_ai_development.prompts.prompt_builder import build_sql_generator_prompt


def build_sql_generator_chain():
    stages_cfg = get_stages_config()
    stage_cfg = stages_cfg.get("sql_generator_agent")

    if not stage_cfg:
        raise ValueError("Missing 'sql_generator_agent' configuration in stages.yaml")

    parser = JsonOutputParser(
        schema=[
            {
                "name": "sql_query",
                "description": "Candidate read-only SQL SELECT statement for ak_country.",
                "type": "string",
            },
            {
                "name": "out_of_scope",
                "description": "True when the user query cannot be answered from the ak_country table.",
                "type": "boolean",
            },
        ]
    )

    format_instructions = parser.get_format_instructions()
    prompt = build_sql_generator_prompt(stage_cfg, format_instructions)

    model_config = stage_cfg.get("model")
    primary_attempt, fallback_attempts = get_model_attempts("sql_generator_agent", model_config)
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
