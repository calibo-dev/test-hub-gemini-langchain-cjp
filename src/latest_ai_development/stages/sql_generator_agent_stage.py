from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.chains.sql_generator_agent_chain_builder import build_sql_generator_chain
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class SqlGeneratorAgentStage(BaseStage):
    component_name = "sql_generator_agent"

    def __init__(self) -> None:
        self.chain = build_sql_generator_chain()
        stages_cfg = get_stages_config()
        self.stage_cfg = stages_cfg.get(self.component_name)
        if not self.stage_cfg:
            raise ValueError("Missing 'sql_generator_agent' configuration in stages.yaml")

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"
        user_query = inputs.get("user_query")

        if not user_query:
            raise ValueError("user_query is required for the sql_generator_agent stage")

        payload = {
            "user_query": user_query,
            "metadata": inputs.get("metadata", ""),
            "current_year": inputs.get("current_year", ""),
            "knowledge_context": inputs.get("knowledge_context", ""),
        }

        logger.info(
            "SQL generation stage started | flow_run_id=%s | component=%s",
            flow_run_id,
            self.component_name,
        )
        start_at = time.perf_counter()

        parsed = self.chain.invoke(payload)

        # Keep the workflow contract strict: the SQL generator must return
        # "sql_query". Do not accept "sql" or other alternate key names here.
        sql_query = _ensure_string(parsed.get("sql_query"))
        out_of_scope = bool(parsed.get("out_of_scope"))

        if not sql_query and not out_of_scope:
            logger.warning(
                "SQL generation returned no query; marking request out of scope | "
                "flow_run_id=%s | component=%s | parsed_keys=%s",
                flow_run_id,
                self.component_name,
                sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
            )
            out_of_scope = True

        logger.info(
            "SQL generation stage completed | flow_run_id=%s | component=%s | out_of_scope=%s | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            out_of_scope,
            time.perf_counter() - start_at,
        )

        return {"sql_query": sql_query.strip(), "out_of_scope": out_of_scope}


def _ensure_string(value: Any) -> str:
    """Return a trimmed string for optional parsed model values."""
    if value is None:
        return ""
    return str(value).strip()
