from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.agents.research_agent_builder import (
    build_research_agent,
    extract_final_message_text,
)
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.prompts.prompt_builder import build_research_user_prompt
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class ResearchStage(BaseStage):
    component_name = "research_agent"

    def __init__(self) -> None:
        self.agent = build_research_agent()
        self.stage_cfg = get_stages_config()["research"]

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"

        topic = inputs.get("topic")

        if not topic:
            logger.error(
                "Research stage validation failed | flow_run_id=%s | component=%s | missing=topic",
                flow_run_id,
                self.component_name,
            )
            raise ValueError("Missing required input: topic")

        logger.debug(
            "Research stage started | flow_run_id=%s | component=%s | topic=%s",
            flow_run_id,
            self.component_name,
            topic,
        )

        started_at = time.perf_counter()
        try:
            user_prompt = build_research_user_prompt(self.stage_cfg, inputs)
            result = self.agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                }
            )
            research_notes = extract_final_message_text(result)
        except Exception:
            logger.exception(
                "Research stage execution failed | flow_run_id=%s | component=%s | "
                "topic=%s | elapsed=%.2fs",
                flow_run_id,
                self.component_name,
                topic,
                time.perf_counter() - started_at,
            )
            raise

        if not research_notes:
            logger.warning(
                "Research stage produced empty research notes | flow_run_id=%s | "
                "component=%s | topic=%s",
                flow_run_id,
                self.component_name,
                topic,
            )

        logger.debug(
            "Research stage completed | flow_run_id=%s | component=%s | topic=%s | "
            "notes_chars=%d | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            topic,
            len(research_notes),
            time.perf_counter() - started_at,
        )

        return {
            "topic": topic,
            "research_notes": research_notes,
        }
