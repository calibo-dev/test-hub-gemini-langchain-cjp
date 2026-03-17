from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm
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

    # Stage configuration flags
    temperature_override = stage_cfg.get("temperature_override")

    # Load LLM (allow stage-level override)
    llm = (
        get_llm(temperature=temperature_override)
        if temperature_override is not None
        else get_llm()
    )

    # Output parser
    parser = StrOutputParser()

    # Runnable pipeline
    chain = prompt | llm | parser

    return chain