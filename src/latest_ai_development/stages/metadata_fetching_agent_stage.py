from __future__ import annotations

import logging
import time
from typing import Any

from latest_ai_development.agents.metadata_fetching_agent_builder import (
    build_metadata_fetching_agent,
    extract_final_message_text,
)
from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.prompts.prompt_builder import build_metadata_fetching_user_prompt
from latest_ai_development.stages.base_stage import BaseStage

logger = logging.getLogger(__name__)


class MetadataFetchingAgentStage(BaseStage):
    component_name = "metadata_fetching_agent"

    def __init__(self) -> None:
        self.agent = build_metadata_fetching_agent()
        stages_cfg = get_stages_config()
        self.stage_cfg = stages_cfg.get(self.component_name)
        if not self.stage_cfg:
            raise ValueError("Missing 'metadata_fetching_agent' configuration in stages.yaml")

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        flow_run_id = str(inputs.get("flow_run_id") or "unknown").strip() or "unknown"

        logger.info(
            "Metadata stage started | flow_run_id=%s | component=%s",
            flow_run_id,
            self.component_name,
        )
        start_at = time.perf_counter()
        metadata = ""
        try:
            user_prompt = build_metadata_fetching_user_prompt(self.stage_cfg, inputs)
            result = self.agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
            metadata = extract_final_message_text(result).strip()
        except Exception:
            logger.exception(
                "Metadata stage execution failed | flow_run_id=%s | component=%s | elapsed=%.2fs",
                flow_run_id,
                self.component_name,
                time.perf_counter() - start_at,
            )
            raise

        if not metadata:
            logger.warning(
                "Metadata stage produced empty metadata | flow_run_id=%s | component=%s",
                flow_run_id,
                self.component_name,
            )

        logger.info(
            "Metadata stage completed | flow_run_id=%s | component=%s | chars=%d | elapsed=%.2fs",
            flow_run_id,
            self.component_name,
            len(metadata),
            time.perf_counter() - start_at,
        )

        return {"metadata": metadata}
