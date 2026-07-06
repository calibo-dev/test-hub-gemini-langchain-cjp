from __future__ import annotations

from typing import Any

from latest_ai_development.agents.research_agent_builder import (
    build_research_agent,
    extract_final_message_text,
)
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.prompts.prompt_builder import build_research_user_prompt
from latest_ai_development.stages.base_stage import BaseStage


class ResearchStage(BaseStage):

    def __init__(self) -> None:
        self.agent = build_research_agent()
        self.stage_cfg = get_stages_config()["research"]

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:

        topic = inputs.get("topic")

        if not topic:
            raise ValueError("Missing required input: topic")

        user_prompt = build_research_user_prompt(self.stage_cfg, inputs)
        result = self.agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
            }
        )
        research_notes = extract_final_message_text(result)

        return {
            "topic": topic,
            "research_notes": research_notes,
        }
