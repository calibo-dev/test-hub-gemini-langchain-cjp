from __future__ import annotations

from typing import Dict, Any

from latest_ai_development.chains.research_chain_builder import build_research_chain
from latest_ai_development.stages.base_stage import BaseStage


class ResearchStage(BaseStage):

    def __init__(self) -> None:
        self.chain = build_research_chain()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        topic = inputs.get("topic")

        if not topic:
            raise ValueError("Missing required input: topic")

        research_notes = self.chain.invoke(
            {
                "topic": topic
            }
        )

        return {
            "topic": topic,
            "research_notes": research_notes,
        }