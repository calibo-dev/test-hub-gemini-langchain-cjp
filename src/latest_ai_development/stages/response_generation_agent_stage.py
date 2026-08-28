from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.chains.response_generation_agent_chain_builder import (
    build_response_generation_chain,
)
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class ResponseGenerationAgentStage(BaseStage):
    component_name = "response_generation_agent"

    def __init__(self) -> None:
        self.chain = build_response_generation_chain()
        stages_cfg = get_stages_config()
        self.stage_cfg = stages_cfg.get(self.component_name)
        if not self.stage_cfg:
            raise ValueError("Missing 'response_generation_agent' configuration in stages.yaml")

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"
        results = inputs.get("results", "")
        out_of_scope = bool(inputs.get("out_of_scope"))
        user_query = inputs.get("user_query", "")

        payload = {
            "results": results,
            "out_of_scope": out_of_scope,
            "user_query": user_query,
            "current_year": inputs.get("current_year", ""),
            "knowledge_context": inputs.get("knowledge_context", ""),
        }

        logger.info(
            "Response generation stage started | flow_run_id=%s | component=%s",
            flow_run_id,
            self.component_name,
        )
        start_at = time.perf_counter()

        response = self.chain.invoke(payload)
        response_text = response.strip()

        logger.info(
            "Response generation stage completed | flow_run_id=%s | component=%s | char_count=%d | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            len(response_text),
            time.perf_counter() - start_at,
        )

        return {"response": response_text}
