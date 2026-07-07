from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.chains.reporting_chain_builder import build_reporting_chain
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class ReportingStage(BaseStage):
    component_name = "reporting_chain"

    def __init__(self) -> None:
        self.chain = build_reporting_chain()

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"

        topic = inputs.get("topic")
        research_notes = inputs.get("research_notes")

        if not topic:
            logger.error(
                "Reporting stage validation failed | flow_run_id=%s | component=%s | missing=topic",
                flow_run_id,
                self.component_name,
            )
            raise ValueError("Missing required input: topic")

        if not research_notes:
            logger.error(
                "Reporting stage validation failed | flow_run_id=%s | "
                "component=%s | missing=research_notes",
                flow_run_id,
                self.component_name,
            )
            raise ValueError("Missing required input: research_notes")

        logger.debug(
            "Reporting stage started | flow_run_id=%s | component=%s | topic=%s",
            flow_run_id,
            self.component_name,
            topic,
        )

        started_at = time.perf_counter()
        try:
            report = self.chain.invoke(
                {
                    "topic": topic,
                    "research_notes": research_notes,
                    "current_year": inputs.get("current_year", ""),
                    "knowledge_context": inputs.get("knowledge_context", ""),
                }
            )
        except Exception:
            logger.exception(
                "Reporting stage execution failed | flow_run_id=%s | component=%s | "
                "topic=%s | elapsed=%.2fs",
                flow_run_id,
                self.component_name,
                topic,
                time.perf_counter() - started_at,
            )
            raise

        logger.debug(
            "Reporting stage completed | flow_run_id=%s | component=%s | "
            "topic=%s | report_chars=%d | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            topic,
            len(report) if isinstance(report, str) else len(str(report)),
            time.perf_counter() - started_at,
        )

        return {
            "topic": topic,
            "research_notes": research_notes,
            "report": report,
        }
