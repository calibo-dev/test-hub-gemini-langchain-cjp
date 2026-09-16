from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.chains.validator_agent_chain_builder import build_validator_chain
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class ValidatorAgentStage(BaseStage):
    component_name = "validator_agent"

    def __init__(self) -> None:
        self.chain = build_validator_chain()
        stages_cfg = get_stages_config()
        self.stage_cfg = stages_cfg.get(self.component_name)
        if not self.stage_cfg:
            raise ValueError("Missing 'validator_agent' configuration in stages.yaml")

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"
        out_of_scope = bool(inputs.get("out_of_scope"))
        # Keep the workflow contract strict: validator input must be "sql_query".
        # Do not accept "sql" or other alternate key names here.
        sql_query = _ensure_string(inputs.get("sql_query"))

        # When an earlier stage marks the request out of scope, there may be no
        # SQL to validate. Propagate that state instead of failing the workflow.
        if out_of_scope and not sql_query:
            logger.info(
                "Validator stage skipped because request is out of scope | "
                "flow_run_id=%s | component=%s",
                flow_run_id,
                self.component_name,
            )
            return {"validated_query": "", "out_of_scope": True}

        if not sql_query:
            raise ValueError("sql_query is required for the validator_agent stage")

        payload = {
            "sql_query": sql_query,
            "out_of_scope": out_of_scope,
        }

        logger.info(
            "Validator stage started | flow_run_id=%s | component=%s",
            flow_run_id,
            self.component_name,
        )
        start_at = time.perf_counter()

        parsed = self.chain.invoke(payload)

        # Keep the validator contract strict: output must be "validated_query".
        validated_query = _ensure_string(parsed.get("validated_query"))
        out_of_scope = bool(parsed.get("out_of_scope"))

        if not validated_query and not out_of_scope:
            logger.warning(
                "Validator returned no validated query; marking request out of scope | "
                "flow_run_id=%s | component=%s | parsed_keys=%s",
                flow_run_id,
                self.component_name,
                sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
            )
            out_of_scope = True

        logger.info(
            "Validator stage completed | flow_run_id=%s | component=%s | out_of_scope=%s | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            out_of_scope,
            time.perf_counter() - start_at,
        )

        return {"validated_query": validated_query, "out_of_scope": out_of_scope}


def _ensure_string(value: Any) -> str:
    """Return a trimmed string for optional parsed model values."""
    if value is None:
        return ""
    return str(value).strip()
