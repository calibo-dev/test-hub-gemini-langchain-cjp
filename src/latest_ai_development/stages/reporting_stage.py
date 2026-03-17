from __future__ import annotations

from typing import Dict, Any

from latest_ai_development.chains.reporting_chain_builder import build_reporting_chain
from latest_ai_development.stages.base_stage import BaseStage


class ReportingStage(BaseStage):

    def __init__(self) -> None:
        self.chain = build_reporting_chain()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        topic = inputs.get("topic")
        research_notes = inputs.get("research_notes")

        if not topic:
            raise ValueError("Missing required input: topic")

        if not research_notes:
            raise ValueError("Missing required input: research_notes")

        report = self.chain.invoke(
            {
                "topic": topic,
                "research_notes": research_notes,
            }
        )

        return {
            "topic": topic,
            "research_notes": research_notes,
            "report": report,
        }