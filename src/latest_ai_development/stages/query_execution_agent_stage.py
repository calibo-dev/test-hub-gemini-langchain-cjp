from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.agents.query_execution_agent_builder import (
    build_query_execution_agent,
    extract_final_message_text,
)
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.prompts.prompt_builder import build_query_execution_user_prompt
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class QueryExecutionAgentStage(BaseStage):
    component_name = "query_execution_agent"

    def __init__(self) -> None:
        self.agent = build_query_execution_agent()
        stages_cfg = get_stages_config()
        self.stage_cfg = stages_cfg.get(self.component_name)
        if not self.stage_cfg:
            raise ValueError("Missing 'query_execution_agent' configuration in stages.yaml")

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"
        out_of_scope = bool(inputs.get("out_of_scope"))

        if out_of_scope:
            logger.info(
                "Query execution stage skipped (out_of_scope) | flow_run_id=%s | component=%s",
                flow_run_id,
                self.component_name,
            )
            return {"results": "", "out_of_scope": True}

        validated_query = inputs.get("validated_query")
        if not validated_query:
            raise ValueError("validated_query is required for the query_execution_agent stage when not out_of_scope")

        user_prompt = build_query_execution_user_prompt(self.stage_cfg, inputs)

        logger.info(
            "Query execution stage started | flow_run_id=%s | component=%s",
            flow_run_id,
            self.component_name,
        )
        start_at = time.perf_counter()

        try:
            result = self.agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
            results_text = extract_final_message_text(result).strip()
        except Exception:
            logger.exception(
                "Query execution stage failed | flow_run_id=%s | component=%s | elapsed=%.2fs",
                flow_run_id,
                self.component_name,
                time.perf_counter() - start_at,
            )
            raise

        logger.info(
            "Query execution stage completed | flow_run_id=%s | component=%s | char_count=%d | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            len(results_text),
            time.perf_counter() - start_at,
        )

        return {"results": results_text, "out_of_scope": False}
